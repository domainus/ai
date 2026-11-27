"""Checks for the Cycle 5 C5-02 per-seed iso-loss PTQ and Muon-mitigation benchmark.

Search, selection, statistics, geometry, guard and validator tests are pure Python or tiny tensors.
The pipeline tests train the ``tiny`` preset on synthetic shards; on an sm_100+ GPU they include
real hardware W4A4 rows. The committed canonical artifact is a ``cycle5_c5_02.v2`` postprocess of the
``cycle5_c5_02.v1`` measurements: its tests need the local git repository and no GPU.
"""

import copy
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats

pytest.importorskip("triton")  # the benchmark imports the Blackwell NVFP4 kernels

from research.polar_opt import harness_tier2_real as h  # noqa: E402
from research.quant_ptq import bench_c5_02_isoloss_ptq_mitigation as b  # noqa: E402
from research.quant_ptq import micro_rotated_ptq as q  # noqa: E402
from tests.test_micro_rotated_ptq import _fake_data  # noqa: E402

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"
FP4_GEMM = CUDA and hasattr(F, "scaled_mm") and torch.cuda.get_device_capability()[0] >= 10
ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "cycle5_c5_02_isoloss_ptq_mitigation.json"
SPEC = b.CANONICAL["adamw"]
SOURCE_COMMIT = "da148c296a4b8750fa9e67c0145e4f0789ecba54"  # the commit of the cycle5_c5_02.v1 measurements
SOURCE_SHA256 = "4944c2832056524f994a2d4931950983244ac20e547cc9e281d5e7e0defeddb8"
NONCANONICAL = {"mode": "noncanonical", "certified_canonical": False, "postprocessed": False}


# ------------------------------------------------------------------ Stage A search and selection (pure)
def test_canonical_protocol_is_the_predeclared_one():
    p = b.CANONICAL
    assert p["seeds"] == [1, 2, 3, 4, 5] and p["tolerance_nats"] == 0.02 and p["preset"] == "50m"
    assert p["muon"] == {"steps": 320, "lr": 0.02, "auxiliary_lr": 0.001} and p["adamw"]["lr"] == 0.001
    assert p["adamw_cooldown"]["muon_steps"] + p["adamw_cooldown"]["cooldown_steps"] == 320
    assert p["adamw_cooldown"]["cooldown_steps"] == 64 and p["qat_cooldown"]["cooldown_steps"] == 48
    assert p["qat_cooldown"]["muon_steps"] == 272 and p["awq"]["alphas"] == [0.0, 0.25, 0.5, 0.75]
    assert p["calibration_sequences"] == 128 and p["weight_formats"] == list(q.FORMATS)
    assert SPEC == {"lr": 0.001, "cosine_steps": 464, "first_eval_step": 400, "eval_grid": 16, "max_steps": 2560}
    grid = b.adamw_eval_steps(SPEC)
    assert grid[0] == 400 and grid[-1] == 2560 and SPEC["cosine_steps"] in grid and len(grid) == 136
    assert all(y - x == 16 for x, y in zip(grid, grid[1:]))
    assert 320 * 32768 == 10_485_760  # T_muon in tokens
    for quick_key in ("adamw_cooldown", "qat_cooldown"):
        assert b.QUICK[quick_key]["muon_steps"] + b.QUICK[quick_key]["cooldown_steps"] == b.QUICK["muon"]["steps"]


def test_adamw_schedule_is_cosine_to_the_floor_then_the_floor():
    cosine = SPEC["cosine_steps"]
    assert b.c4.warmup_for(cosine) == 46
    assert b.adamw_multiplier(0, SPEC) == pytest.approx(1 / 46) and b.adamw_multiplier(45, SPEC) == 1.0
    for step in range(cosine):
        assert b.adamw_multiplier(step, SPEC) == b.schedule_multiplier(step, cosine)  # the C4-01b schedule
    assert b.schedule_multiplier(cosine, cosine) == pytest.approx(b.MIN_LR_RATIO)
    assert all(b.adamw_multiplier(step, SPEC) == b.MIN_LR_RATIO for step in (cosine, cosine + 1, 2559, 5000))
    assert b.adamw_multiplier(399, SPEC) < 0.16  # the first measured step is already near the floor LR
    assert b.adamw_stops(0.0) and b.adamw_stops(-1e-9) and not b.adamw_stops(1e-9)


def _evals(*points):
    return [{"step": s, "val_nll": v, "checkpoint_sha256": f"{s:064x}"} for s, v in points]


def _rows(muon_val, evaluations):
    return [{"arm": b.MUON, "steps": 320, "val_nll": muon_val, "tokens": 320 * 32768},
            {"arm": b.COOL_ADAMW}, {"arm": b.COOL_QAT},
            {"arm": b.ADAMW, "lr": 0.001, "cosine_steps": 464, "evaluations": evaluations}]


def test_selection_takes_the_measured_step_nearest_to_muon_then_fewer_steps():
    rows = _rows(5.34, _evals((400, 5.45), (416, 5.40), (432, 5.352), (448, 5.331)))
    s = b.select_adamw(rows, 0.02)
    assert (s["matched"], s["t_adamw"], s["adamw_evaluations"]) == (True, 448, 4)
    assert s["val_gap_muon_minus_adamw"] == 5.34 - 5.331 and s["abs_val_gap"] == abs(5.34 - 5.331)
    assert s["adamw_tokens"] == 448 * 32768 and s["adamw_over_muon_tokens"] == 448 / 320
    assert s["adamw_floor_lr_steps"] == 448 - 464 and s["adamw_cosine_steps"] == 464
    assert s["adamw_checkpoint_sha256"] == f"{448:064x}" and s["adamw_stopped_at_or_below_muon"] is True
    # The step before the crossing wins when it is nearer.
    near = b.select_adamw(_rows(5.34, _evals((400, 5.45), (416, 5.343), (432, 5.33))), 0.02)
    assert near["t_adamw"] == 416 and near["adamw_floor_lr_steps"] == -48
    # An exact tie in |gap| goes to the earlier step.
    assert b.select_adamw(_rows(5.0, _evals((400, 5.25), (416, 4.75))), 0.02)["t_adamw"] == 400
    # The tolerance edge matches; just beyond it does not, and the nearest step is still returned.
    assert b.select_adamw(_rows(5.0, _evals((400, 5.02))), 0.02)["matched"] is True
    far = b.select_adamw(_rows(5.0, _evals((400, 5.0201))), 0.02)
    assert far["matched"] is False and far["t_adamw"] == 400 and far["adamw_stopped_at_or_below_muon"] is False
    # A run that is already below Muon at the first measured step is reported, not hidden.
    low = b.select_adamw(_rows(5.0, _evals((400, 4.9))), 0.02)
    assert low["matched"] is False and low["adamw_stopped_at_or_below_muon"] is True
    assert not any("test" in key for key in s)
    with pytest.raises(ValueError, match="no measured AdamW step"):
        b.select_adamw(_rows(5.0, []), 0.02)
    with pytest.raises(ValueError, match="no measured AdamW step"):
        b.select_adamw(_rows(5.0, [])[:3], 0.02)


def test_adamw_run_stops_at_the_first_measured_step_at_or_below_muon(tmp_path):
    """The run is driven by the val results alone; the returned state is the measured step nearest to Muon."""
    config, shards, init = _tiny_setup(tmp_path)
    cpu, protocol = torch.device("cpu"), copy.deepcopy(b.QUICK)
    train = shards.open("train")["train"]
    spec = protocol["adamw"]
    curve = iter([5.50, 5.42, 5.36, 5.31, 5.2])  # a falling val curve; Muon is at 5.35
    seen = []

    def evaluate(model):
        seen.append(b.e2e._checkpoint_sha256(model))
        return {"nll": next(curve), "tokens": 1024}

    state, row = b.train_adamw_matched(config, train, cpu, protocol, seed=3, init=init, target_val_nll=5.35,
                                       evaluate=evaluate)
    steps = [e["step"] for e in row["evaluations"]]
    assert steps == b.adamw_eval_steps(spec)[:4] == [24, 32, 40, 48]  # stops at the first step below 5.35
    assert [e["val_gap"] for e in row["evaluations"]] == pytest.approx([0.15, 0.07, 0.01, -0.04])
    assert [e["checkpoint_sha256"] for e in row["evaluations"]] == seen and len(set(seen)) == 4
    assert [e["lr_multiplier"] for e in row["evaluations"]] == [b.adamw_multiplier(s - 1, spec) for s in steps]
    assert row["evaluations"][2]["lr_multiplier"] == b.MIN_LR_RATIO  # steps after the cosine run at the floor LR
    assert (row["steps"], row["trained_steps"], row["val_nll"]) == (40, 48, 5.36)  # 0.01 above beats 0.04 below
    assert row["checkpoint_sha256"] == seen[2] and row["tokens"] == 40 * config.step_tokens
    assert row["warmup_steps"] == b.c4.warmup_for(spec["cosine_steps"]) and row["cosine_steps"] == spec["cosine_steps"]
    assert [p[0] for p in row["train_curve"]] == [1, 10, 20, 30, 40] and row["evaluated_splits"] == ["val"]
    assert b.e2e._checkpoint_sha256(b._new_model(config, cpu, state)) == seen[2]  # the weights of step 40
    # Without a crossing the run uses the whole grid and reports the last (nearest) step.
    never = iter([6.0 - 0.01 * i for i in range(10)])
    _, capped = b.train_adamw_matched(config, train, cpu, protocol, seed=3, init=init, target_val_nll=5.0,
                                      evaluate=lambda model: {"nll": next(never), "tokens": 1024})
    assert [e["step"] for e in capped["evaluations"]] == b.adamw_eval_steps(spec)
    assert capped["steps"] == capped["trained_steps"] == spec["max_steps"]
    # The same run is deterministic on the CPU: the weights of a measured step do not depend on the val values.
    again = iter([5.50, 5.2])
    _, short = b.train_adamw_matched(config, train, cpu, protocol, seed=3, init=init, target_val_nll=5.35,
                                     evaluate=lambda model: {"nll": next(again), "tokens": 1024})
    assert [e["checkpoint_sha256"] for e in short["evaluations"]] == seen[:2]


# ------------------------------------------------------------------ statistics
def test_t_quantile_is_the_scipy_value_for_four_degrees_of_freedom():
    assert b.T975_DF4 == pytest.approx(scipy_stats.t.ppf(0.975, 4), rel=1e-13)
    assert b.SIGN_FLIP_MIN_P == 0.0625 and b.N_SEEDS == 5


@pytest.mark.parametrize("diffs", [[0.011, 0.014, 0.009, 0.012, 0.010], [-0.3, 0.2, 0.05, 0.1, -0.02],
                                   [1e-6, 2e-6, 3.5e-6, 2.5e-6, 1.5e-6], [5.0, 5.0001, 4.9999, 5.0002, 5.0],
                                   [-0.02, -0.03, -0.025, -0.021, -0.5]])
def test_paired_stats_match_scipy(diffs):
    s = b.paired_stats(diffs)
    ref = scipy_stats.ttest_1samp(diffs, 0.0)
    assert s["n"] == 5 and s["df"] == 4 and s["diffs"] == diffs
    assert s["mean"] == pytest.approx(np.mean(diffs), rel=1e-12)
    assert s["sd"] == pytest.approx(np.std(diffs, ddof=1), rel=1e-12)
    assert s["t"] == pytest.approx(ref.statistic, rel=1e-10)
    assert s["p_two_sided"] == pytest.approx(ref.pvalue, rel=1e-9)
    low, high = ref.confidence_interval(0.95)
    assert s["ci95"] == pytest.approx([low, high], rel=1e-9, abs=1e-15)


def test_paired_t_p_value_is_stable_at_large_t_and_exact_at_the_edges():
    for t in (0.0, 1e-3, 1.0, 50.0, 1e3, 1e5, 1e7):
        u = 4 / (4 + t * t)
        s = math.sqrt(1 - u)
        stable = u * u * (3 + u) / (4 * (1 + s * (1 + u / 2)))
        assert stable == pytest.approx(2 * scipy_stats.t.sf(t, 4), rel=1e-11)
    same = b.paired_stats([0.25] * 5)
    assert same["sd"] == 0 and same["t"] is None and same["p_two_sided"] == 0.0 and same["p_sign_flip"] == 0.0625
    zero = b.paired_stats([0.0] * 5)
    assert zero["p_two_sided"] == 1.0 and zero["p_sign_flip"] == 1.0 and zero["ci95"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="exactly 5"):
        b.paired_stats([0.1, 0.2, 0.3])


def test_sign_flip_p_value_is_the_exact_enumeration():
    assert b.sign_flip_p([0.1, 0.2, 0.3, 0.4, 0.5]) == 2 / 32  # all one sign: only the two extreme patterns
    assert b.sign_flip_p([-0.1, -0.2, -0.3, -0.4, -0.5]) == 2 / 32
    assert b.sign_flip_p([0.1, -0.1, 0.2, -0.2, 0.0]) == 1.0  # observed sum 0: every pattern is as extreme
    rng = np.random.default_rng(1)
    for _ in range(50):
        d = rng.normal(0.3, 1.0, 5).tolist()
        observed = abs(sum(d))
        sums = [abs(sum(s * x for s, x in zip(signs, d))) for signs in itertools.product((1, -1), repeat=5)]
        assert b.sign_flip_p(d) == sum(v >= observed - 1e-12 for v in sums) / 32
        assert b.sign_flip_p(d) >= b.SIGN_FLIP_MIN_P and b.sign_flip_p(d) == b.sign_flip_p([-x for x in d])
    ref = scipy_stats.permutation_test(([0.4, 0.1, 0.3, -0.05, 0.2],), np.mean, permutation_type="samples",
                                       n_resamples=np.inf, alternative="two-sided")
    assert b.sign_flip_p([0.4, 0.1, 0.3, -0.05, 0.2]) == pytest.approx(ref.pvalue)


def test_holm_bonferroni_is_the_step_down_procedure():
    assert b.holm_bonferroni([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert b.holm_bonferroni([0.0625] * 6) == pytest.approx([0.375] * 6)  # sign-flip floor never passes six cells
    assert b.holm_bonferroni([0.5, 0.9]) == [1.0, 1.0]


# ------------------------------------------------------------------ activation-aware scaling and geometry
def test_awq_scales_have_unit_geometric_mean_and_reduce_to_one_at_alpha_zero():
    mean_abs = torch.tensor([[0.5, 2.0, 8.0, 0.0], [1.0, 1.0, 4.0, 0.25]], dtype=torch.float64)
    ones = b.awq_scales(mean_abs, 0.0)
    assert ones.dtype == torch.float32 and torch.equal(ones, torch.ones(2, 4))
    s = b.awq_scales(mean_abs, 0.5)
    assert s[0, 3] == 1.0  # a channel without input energy keeps scale 1
    assert s[0, :3].double().log().sum().abs() < 1e-6 and s[1].double().log().sum().abs() < 1e-6
    assert s[0, 2] / s[0, 0] == pytest.approx((8.0 / 0.5) ** 0.5, rel=1e-6)  # s_j proportional to mean|x_j|^alpha
    assert torch.allclose(b.awq_scales(mean_abs, 1.0)[1], mean_abs[1].float(), rtol=1e-6)  # geometric mean is 1


def test_scaled_quantization_is_exact_bookkeeping_and_alpha_zero_is_the_plain_method():
    torch.manual_seed(0)
    L, rows, cols = 2, 32, 64
    W = torch.randn(L, rows, cols)
    X = torch.randn(L, 512, cols, dtype=torch.float64) * torch.logspace(-1, 1, cols, dtype=torch.float64)
    H = X.mT @ X / 512
    scale = b.awq_scales(X.abs().mean(1), 0.5)
    # The Hessian of the scaled input x / s is D^-1 H D^-1.
    Xs = X / scale.double()[:, None, :]
    assert torch.allclose(b.scale_hessian(H, scale), Xs.mT @ Xs / 512, rtol=1e-12)
    # Output error of Q(W D) D^-1 against H equals the error of Q(W D) against the scaled Hessian.
    fmt, rot = q.FORMATS["nvfp4_g16"], q.BlockHadamard(16)
    Ws = W * scale[:, None, :]
    Qr, _ = q.gptq_quantize(rot.forward(Ws), rot.rotate_hessian(b.scale_hessian(H, scale)), fmt)
    effective = rot.inverse(Qr) / scale[:, None, :]
    direct = q.output_error(W, effective, H)
    scaled = q.output_error(Ws, rot.inverse(Qr), b.scale_hessian(H, scale))
    assert torch.allclose(direct, scaled, rtol=1e-5)
    # With unit scales the scaled path is the plain rotation + GPTQ, bit for bit.
    ones = b.awq_scales(X.abs().mean(1), 0.0)
    plain, _ = q.gptq_quantize(rot.forward(W), rot.rotate_hessian(H), fmt)
    unit, _ = q.gptq_quantize(rot.forward(W * ones[:, None, :]), rot.rotate_hessian(b.scale_hessian(H, ones)), fmt)
    assert torch.equal(plain, unit) and torch.equal(rot.inverse(unit) / ones[:, None, :], rot.inverse(plain))
    # The hardware input map divides by the scales before the rotation.
    x = torch.randn(7, cols)
    mapped = b.ScaledRotation(rot, 1.0 / scale[0]).forward(x)
    assert torch.allclose(mapped, rot.forward(x / scale[0]), rtol=1e-5, atol=1e-5)
    layer = b.ScaledInput(torch.nn.Identity(), 1.0 / scale[0])
    assert torch.allclose(layer(x.to(torch.bfloat16)), x.to(torch.bfloat16).float() / scale[0], rtol=1e-5,
                          atol=1e-6)
    assert b.select_alpha([0.0, 0.25, 0.5, 0.75], [5.2, 5.1, 5.1, 5.3]) == 0.25  # tie: the smaller alpha
    assert b.select_alpha([0.0, 0.25, 0.5, 0.75], [5.0, 5.1, 5.2, 5.3]) == 0.0


def test_layer_geometry_definitions_and_the_error_identity():
    torch.manual_seed(1)
    W = torch.randn(3, 24, 48)
    X = torch.randn(3, 400, 48, dtype=torch.float64) * torch.logspace(-1, 1, 48, dtype=torch.float64)
    H = X.mT @ X / 400
    rank = b.stable_rank(W)
    for i in range(3):
        sv = torch.linalg.svdvals(W[i].double())
        assert rank[i].item() == pytest.approx((sv.square().sum() / sv[0] ** 2).item(), rel=1e-10)
    assert b.stable_rank(W.mT)[0].item() == pytest.approx(rank[0].item(), rel=1e-10)  # tall or wide
    eye = torch.eye(48, dtype=torch.float64).expand(3, 48, 48)
    assert torch.allclose(b.alignment(W, eye), torch.ones(3, dtype=torch.float64))  # isotropic reference
    top = torch.linalg.eigh(H[0]).eigenvectors[:, -1]
    assert b.alignment(top[None, None, :].float(), H[:1]).item() > 5  # a weight on the top input direction
    assert b.alignment(torch.zeros(1, 4, 48), H[:1]).item() == 0.0
    W_hat = q.quantize_rtn(W, q.FORMATS["int2_asym_g16"])
    g = b.error_geometry(W, W_hat, H)
    signal = b.alignment(W, H).tolist()
    reference = (q.output_error(W, W_hat, H) / q.output_error(W, torch.zeros_like(W), H)).tolist()
    for i in range(3):
        assert g["output_rel_error"][i] == pytest.approx(reference[i], rel=1e-7)  # output_error subtracts in FP32
        assert g["output_rel_error"][i] == pytest.approx(
            g["weight_rel_error"][i] * g["error_alignment"][i] / signal[i], rel=1e-10)
    assert b.layer_names(2) == ["blocks.0.qkv", "blocks.1.qkv", "blocks.0.out", "blocks.1.out", "blocks.0.gate",
                                "blocks.1.gate", "blocks.0.up", "blocks.1.up", "blocks.0.down", "blocks.1.down"]


def test_method_names_and_expected_grids():
    assert b.parse_method("rtn") == ("identity", "rtn", False)
    assert b.parse_method("awq_adaptive_butterfly_b16_gptq") == ("adaptive_butterfly_b16", "gptq", True)
    assert b.methods_of(b.MUON, b.CANONICAL) == [*b.BASE_METHODS, *b.AWQ_METHODS]
    assert b.methods_of(b.COOL_QAT, b.CANONICAL) == list(b.BASE_METHODS)
    assert len(b.expected_combos(b.MUON, b.CANONICAL, True)) == 37  # 24 fake-quant + 12 hardware + 1 emulation
    assert len(b.expected_combos(b.COOL_ADAMW, b.CANONICAL, True)) == 25
    assert len(b.expected_combos(b.ADAMW, b.CANONICAL, False)) == 24
    hw = [c for c in b.expected_combos(b.MUON, b.CANONICAL, True) if c[2] in b.HW_MODES]
    assert {c[0] for c in hw} == {"nvfp4_g16"} and {c[1] for c in hw} == {*b.BASE_METHODS, *b.AWQ_METHODS}


# ------------------------------------------------------------------ training arms
def test_fake_quant_linear_quantizes_weight_and_input_with_straight_through_gradients():
    torch.manual_seed(0)
    fmt = q.FORMATS["nvfp4_g16"]
    linear = torch.nn.Linear(32, 16, bias=False)
    x = torch.randn(3, 5, 32, requires_grad=True)
    layer = b.FakeQuantLinear(linear, fmt)
    out = layer(x)
    wq, xq = q.quantize_rtn(linear.weight.detach(), fmt), q.quantize_rtn(x.detach().reshape(-1, 32), fmt)
    assert torch.allclose(out, F.linear(xq.reshape(3, 5, 32), wq), atol=1e-6)
    assert not torch.equal(wq, linear.weight.detach()) and not torch.equal(xq, x.detach().reshape(-1, 32))
    upstream = torch.randn_like(out)
    (out * upstream).sum().backward()
    # Straight-through: the gradients are those of the linear map at the quantized operands.
    assert torch.allclose(linear.weight.grad, upstream.reshape(-1, 16).T @ xq, atol=1e-5)
    assert torch.allclose(x.grad, upstream @ wq, atol=1e-5)
    assert layer.linear is linear  # the optimizer keeps updating the FP32 master weight


def _tiny_setup(tmp_path, seed=3):
    data_dir = _fake_data(tmp_path, sizes=(200_000, 20_000, 20_000))
    shards = b.Shards(data_dir, verify=False, expected=None)
    return h.PRESETS["tiny"], shards, h.initial_state(h.PRESETS["tiny"], seed)


def test_branch_from_a_snapshot_continues_the_muon_run(tmp_path):
    """Restoring a snapshot and running the remaining Muon steps gives the weights of the full run."""
    config, shards, init = _tiny_setup(tmp_path)
    cpu, protocol, seed = torch.device("cpu"), copy.deepcopy(b.QUICK), 3
    train = shards.open("train")["train"]
    total, split = protocol["muon"]["steps"], protocol["qat_cooldown"]["muon_steps"]
    model, row, snapshots = b.train_muon(config, train, cpu, protocol, seed=seed, init=init, snapshot_steps=(split,))
    assert row["snapshots"] == {str(split): snapshots[split]["sha256"]} and row["steps"] == total
    assert snapshots[split]["sha256"] != row["checkpoint_sha256"] != b.e2e._checkpoint_sha256(
        b._new_model(config, cpu, init))
    assert [p[0] for p in row["train_curve"]] == [1, 10, 20] and row["warmup_steps"] == 20
    branch, optimizer = b._restore(config, cpu, protocol, snapshots[split])
    assert b.e2e._checkpoint_sha256(branch) == snapshots[split]["sha256"]
    stream = b._stream_at(train, config, seed, cpu, split)
    reference = h.BatchStream(train, config, seed, cpu)
    for _ in range(split):
        reference.next_starts()
    assert np.array_equal(stream.next_starts(), reference.next_starts())
    b._run_steps(branch, optimizer, b._stream_at(train, config, seed, cpu, split), split, total,
                 lambda step: b.schedule_multiplier(step, total), "branch")
    assert b.e2e._checkpoint_sha256(branch.eval()) == row["checkpoint_sha256"]


def test_cooldown_arms_branch_from_the_snapshot_and_change_only_the_last_steps(tmp_path):
    config, shards, init = _tiny_setup(tmp_path)
    cpu, protocol, seed = torch.device("cpu"), copy.deepcopy(b.QUICK), 3
    train = shards.open("train")["train"]
    steps = (protocol["adamw_cooldown"]["muon_steps"], protocol["qat_cooldown"]["muon_steps"])
    _, base, snapshots = b.train_muon(config, train, cpu, protocol, seed=seed, init=init, snapshot_steps=steps)
    shas = {base["checkpoint_sha256"]}
    for arm, step in zip((b.COOL_ADAMW, b.COOL_QAT), steps):
        before = copy.deepcopy(snapshots[step]["model"])
        model, row = b.train_cooldown(arm, config, train, cpu, protocol, seed=seed, init=init, snapshot=snapshots[step])
        assert row["arm"] == arm and row["steps"] == protocol["muon"]["steps"] and row["tokens"] == base["tokens"]
        assert row["branch"] == {"from_arm": b.MUON, "from_step": step, "snapshot_sha256": base["snapshots"][str(step)],
                                 "cooldown_steps": protocol["muon"]["steps"] - step}
        assert row["init_fingerprint"] == base["init_fingerprint"]
        assert row["stream_fingerprint"] == base["stream_fingerprint"]
        assert row["train_curve"][0][0] == step + 1 and math.isfinite(row["train_loss"])
        assert all(torch.equal(before[k], snapshots[step]["model"][k]) for k in before)  # the snapshot is not mutated
        assert not any(isinstance(m, b.FakeQuantLinear) for m in model.modules())  # the plain layers are restored
        assert not model.training
        shas.add(row["checkpoint_sha256"])
    assert len(shas) == 3  # both interventions change the final weights
    with pytest.raises(ValueError, match="Unknown cooldown arm"):
        b.train_cooldown("muon_other", config, train, cpu, protocol, seed=seed, init=init, snapshot=snapshots[steps[0]])


def test_adamw_cooldown_updates_matrices_with_fresh_adamw_and_keeps_the_auxiliary_state(tmp_path):
    config, _, init = _tiny_setup(tmp_path)
    cpu, protocol = torch.device("cpu"), copy.deepcopy(b.QUICK)
    model = b._new_model(config, cpu, init)
    muon, _ = b._muon_optimizer(model, protocol, cpu)
    cooldown = b.CooldownAdamW(muon, 0.001)
    assert cooldown.auxiliary is muon.auxiliary
    matrix_ids = {id(p) for p in muon.matrices.values()}
    assert {id(p) for g in cooldown.matrix.param_groups for p in g["params"]} == matrix_ids
    assert all(g["weight_decay"] == b.WEIGHT_DECAY for g in cooldown.matrix.param_groups)
    cooldown.set_lr_multiplier(0.25)
    assert all(g["lr"] == pytest.approx(0.00025) for g in cooldown.matrix.param_groups)
    assert all(g["lr"] == pytest.approx(0.25 * g["base_lr"]) for g in muon.auxiliary.param_groups)
    # Terminal LR of the cooldown equals the terminal LR of the AdamW arm (same multiplier floor).
    total = b.CANONICAL["muon"]["steps"]
    assert b.schedule_multiplier(total, total) == pytest.approx(b.MIN_LR_RATIO)
    assert b.schedule_multiplier(256, total) == pytest.approx(
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (256 - 32) / (320 - 32))))
    assert b.CANONICAL["adamw_cooldown"]["lr"] == b.CANONICAL["adamw"]["lr"]


# ------------------------------------------------------------------ Stage A isolation
def test_stage_a_runs_without_test_bin_and_refuses_a_mapping_that_holds_it(tmp_path):
    config, shards, _ = _tiny_setup(tmp_path)
    (tmp_path / "test.bin").unlink()  # Stage A must not need the file at all
    view = shards.open("train", "val")
    assert shards.opened == ["train", "val"] and set(view) == {"train", "val"}
    protocol = copy.deepcopy(b.QUICK)
    unit = b.stage_a_seed(config, view, torch.device("cpu"), protocol, 7, tmp_path / "ckpt", lambda *a: None)
    assert unit["split_view"] == ["train", "val"] and unit["seed"] == 7
    assert [r["arm"] for r in unit["rows"]] == [b.MUON, b.COOL_ADAMW, b.COOL_QAT, b.ADAMW]
    assert not any("test" in key for key in b._keys(unit))
    for row in unit["rows"]:
        assert row["evaluated_splits"] == ["val"] and row["val_tokens"] == 16 * 64
        assert ((np.asarray(view["val"][:1024]) >= 1000) & (np.asarray(view["val"][:1024]) < 2000)).sum() > 1000
    assert len({(r["init_fingerprint"], r["stream_fingerprint"]) for r in unit["rows"]}) == 1
    assert unit["selection"] == b.select_adamw(unit["rows"], protocol["tolerance_nats"])
    adamw = unit["rows"][3]
    measured = adamw["evaluations"]
    assert [e["step"] for e in measured] == b.adamw_eval_steps(protocol["adamw"])[:len(measured)]
    assert all(e["val_gap"] == e["val_nll"] - unit["rows"][0]["val_nll"] for e in measured)
    assert not any(b.adamw_stops(e["val_gap"]) for e in measured[:-1])
    assert b.adamw_stops(measured[-1]["val_gap"]) or measured[-1]["step"] == protocol["adamw"]["max_steps"]
    assert adamw["steps"] == unit["selection"]["t_adamw"] and adamw["trained_steps"] == measured[-1]["step"]
    # The four final state dicts are saved and load back with the recorded SHA-256.
    assert b.checkpoints_verified(unit, tmp_path / "ckpt")
    for arm, entry in unit["checkpoints"].items():
        assert entry["file"] == b.checkpoint_name(7, arm)
        model = b.load_checkpoint(config, torch.device("cpu"), tmp_path / "ckpt" / entry["file"], entry["sha256"])
        assert not model.training
        assert b.state_sha256(model.state_dict()) == b.e2e._checkpoint_sha256(model) == entry["sha256"]
    with pytest.raises(RuntimeError, match="differs from the Stage A record"):
        b.load_checkpoint(config, torch.device("cpu"), tmp_path / "ckpt" / b.checkpoint_name(7, b.MUON), "0" * 64)
    (tmp_path / "ckpt" / b.checkpoint_name(7, b.ADAMW)).unlink()
    assert not b.checkpoints_verified(unit, tmp_path / "ckpt")
    with pytest.raises(RuntimeError, match="exactly train and val"):
        b.stage_a_seed(config, {**view, "test": view["val"]}, torch.device("cpu"), protocol, 7, tmp_path / "c2",
                       lambda *a: None)
    with pytest.raises(FileNotFoundError):
        shards.open("test")


def test_checkpoint_digest_verifier_rejects_missing_unreadable_corrupt_and_other_weights(tmp_path):
    """H3: file presence is not enough. The verifier reloads the tensors on the CPU and hashes them."""
    torch.manual_seed(0)
    state = {"b.weight": torch.randn(3, 4), "a.bias": torch.randn(4).double(), "c.steps": torch.arange(5)}
    path = tmp_path / "ckpt" / "state.pt"
    b._save_state(state, path)
    digest = b.state_sha256(state)
    reference = hashlib.sha256()  # the definition of e2e._checkpoint_sha256: sorted names, NUL, shape, FP32 bytes
    for name in sorted(state):
        array = state[name].float().contiguous().numpy()
        reference.update(name.encode() + b"\0" + str(array.shape).encode() + array.tobytes())
    assert digest == reference.hexdigest() and b._is_hex(digest, 64)
    assert b.checkpoint_file_matches_sha256(path, digest)
    assert not b.checkpoint_file_matches_sha256(path, "0" * 64)  # another recorded digest
    assert not b.checkpoint_file_matches_sha256(tmp_path / "ckpt" / "missing.pt", digest)
    assert not b.checkpoint_file_matches_sha256(tmp_path / "ckpt", digest)  # a directory
    whole = path.read_bytes()
    for broken in (b"", b"not a checkpoint", whole[:len(whole) // 2]):  # empty, garbage, truncated
        path.write_bytes(broken)
        assert not b.checkpoint_file_matches_sha256(path, digest)
        with pytest.raises(RuntimeError, match="cannot be read"):
            b.load_checkpoint(h.PRESETS["tiny"], torch.device("cpu"), path, digest)
    b._save_state({**state, "b.weight": state["b.weight"] + 1e-3}, path)  # readable, other tensor content
    assert not b.checkpoint_file_matches_sha256(path, digest)
    torch.save([1, 2, 3], path)  # readable, not a state dict
    assert not b.checkpoint_file_matches_sha256(path, digest)
    b._save_state(state, path)
    unit = {"checkpoints": {b.MUON: {"file": "state.pt", "sha256": digest}}}
    assert b.checkpoints_verified(unit, tmp_path / "ckpt")
    entry = unit["checkpoints"][b.MUON]
    assert b.stage_b_unit_verified({"checkpoint_sha256": digest}, entry, tmp_path / "ckpt")
    assert not b.stage_b_unit_verified({"checkpoint_sha256": "0" * 64}, entry, tmp_path / "ckpt")
    path.write_bytes(b"lost")
    assert not b.checkpoints_verified(unit, tmp_path / "ckpt")
    assert not b.stage_b_unit_verified({"checkpoint_sha256": digest}, entry, tmp_path / "ckpt")


def test_shards_check_pinned_hashes(tmp_path):
    data_dir = _fake_data(tmp_path)
    with pytest.raises(ValueError, match="differ from the pinned hashes"):
        b.Shards(data_dir, verify=False, expected=q.EXPECTED_SHA256)
    shards = b.Shards(data_dir, verify=False, expected={"train": "fake-train", "val": "fake-val", "test": "fake-test"})
    assert shards.opened == [] and set(shards.summary()["splits"]) == {"train", "val", "test"}
    shards.open("val")
    shards.open("train", "val")
    assert shards.opened == ["val", "train"]
    # Without verification a size check (a stat, not a read) keeps the manifest token counts honest.
    manifest = json.loads((data_dir / "manifest.json").read_text())
    manifest["splits"]["test"]["tokens"] += 1
    (data_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="test: size"):
        b.Shards(data_dir, verify=False, expected=None)


def test_verified_shards_read_every_split_before_any_stage_maps_one(tmp_path, monkeypatch):
    """B3: ``Shards(verify=True)`` reads test.bin at startup; ``opened`` records only the later ``open`` calls."""
    data_dir = _fake_data(tmp_path)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    read = []

    def verify(directory):  # stands in for real_text_data.verify_manifest: SHA-256, then token range
        for name in ("train", "val", "test"):
            for _ in range(2):
                read.append(name)
                (directory / f"{name}.bin").read_bytes()
        return manifest
    monkeypatch.setattr(b.real_text_data, "verify_manifest", verify)
    shards = b.Shards(data_dir, verify=True, expected=None)
    assert read == ["train", "train", "val", "val", "test", "test"] and shards.opened == []
    assert set(shards.open("train", "val")) == {"train", "val"} and shards.opened == ["train", "val"]
    shards.open("test")
    assert shards.opened == ["train", "val", "test"]
    for text in (b.__doc__, b.Shards.__doc__):
        assert "twice" in " ".join(text.split()) and "test.bin" in text
    assert "not a previously unseen study-level holdout" in " ".join(b.__doc__.split())


# ------------------------------------------------------------------ guards, provenance and resume
def _error(capsys, argv):
    with pytest.raises(SystemExit):
        b.parse_args(argv)
    return capsys.readouterr().err


def test_cli_reserves_the_canonical_filename(tmp_path, capsys):
    args = b.parse_args([])
    assert args.canonical and args.output == b.CANONICAL_OUTPUT and args.protocol == b.CANONICAL
    assert args.checkpoint_dir == b.CANONICAL_CHECKPOINT_DIR and not args.resume
    assert b.parse_args(["--seeds", "1", "2", "3", "4", "5", "--device", "cuda", "--resume"]).canonical
    quick = b.parse_args(["--quick"])
    assert not quick.canonical and quick.output == b.QUICK_OUTPUT and not b.reserved(quick.output)
    assert quick.checkpoint_dir == b.QUICK_OUTPUT.with_name(b.QUICK_OUTPUT.name + ".ckpt")
    other = tmp_path / "other" / "results"
    other.mkdir(parents=True)
    link = tmp_path / "alias.json"
    link.symlink_to(b.CANONICAL_OUTPUT)
    for target in (b.CANONICAL_OUTPUT, other / b.CANONICAL_OUTPUT.name, link, tmp_path / "cycle5_c5_02_x.json"):
        assert b.reserved(target)
        for extra in (["--quick"], ["--seeds", "1"], ["--seeds", "5", "4", "3", "2", "1"], ["--device", "cpu"],
                      ["--max-eval-windows", "64"], ["--no-verify"]):
            assert "Refusing to write the reserved" in _error(capsys, [*extra, "--output", str(target)])
    assert not b.reserved(tmp_path / "c5_02.json")
    # The canonical protocol writes only the canonical path of this checkout.
    for target in (other / b.CANONICAL_OUTPUT.name, link, tmp_path / "plain.json"):
        assert "canonical protocol writes" in _error(capsys, ["--output", str(target)])
    assert "needs an explicit --output" in _error(capsys, ["--seeds", "1"])
    assert "must be distinct" in _error(capsys, ["--seeds", "1", "1", "--output", str(tmp_path / "x.json")])
    free = b.parse_args(["--seeds", "0", "--output", str(tmp_path / "x.json")])
    assert not free.canonical and free.protocol["seeds"] == [0] and free.checkpoint_dir == tmp_path / "x.json.ckpt"


def test_cli_postprocess_mode_reads_a_full_commit_sha_and_measures_nothing(tmp_path, capsys):
    args = b.parse_args(["--postprocess-from-git", SOURCE_COMMIT])
    assert args.postprocess_from_git == SOURCE_COMMIT and args.output == b.CANONICAL_OUTPUT
    assert args.canonical is False and args.protocol is None  # it selects no measuring protocol
    assert b.parse_args([]).postprocess_from_git is None and b.parse_args(["--quick"]).postprocess_from_git is None
    free = b.parse_args(["--postprocess-from-git", SOURCE_COMMIT, "--output", str(tmp_path / "copy.json")])
    assert free.output == tmp_path / "copy.json"
    for short in ("da148c2", "HEAD", SOURCE_COMMIT.upper(), SOURCE_COMMIT + "0"):
        assert "full 40-hex SHA" in _error(capsys, ["--postprocess-from-git", short])
    for extra in (["--quick"], ["--seeds", "1", "2"], ["--device", "cpu"], ["--max-eval-windows", "4"],
                  ["--data-dir", str(tmp_path)], ["--no-verify"], ["--checkpoint-dir", str(tmp_path)], ["--resume"]):
        assert "measures nothing" in _error(capsys, ["--postprocess-from-git", SOURCE_COMMIT, *extra])
    other = tmp_path / "results"
    other.mkdir()
    link = tmp_path / "cycle5_c5_02_alias.json"
    link.symlink_to(b.CANONICAL_OUTPUT)
    for target in (other / b.CANONICAL_OUTPUT.name, link, tmp_path / "cycle5_c5_02_other.json"):
        assert "A postprocess writes" in _error(capsys, ["--postprocess-from-git", SOURCE_COMMIT, "--output",
                                                         str(target)])


def test_postprocess_into_a_reserved_file_needs_a_clean_committed_tree(tmp_path, monkeypatch):
    clean = {"git_sha": "a" * 40, "git_dirty": False}
    b.require_clean_tree(clean, b.CANONICAL_OUTPUT)  # no GPU demand: a postprocess runs on any host
    b.require_clean_tree({"git_sha": None, "git_dirty": True}, tmp_path / "free.json")  # not a reserved name
    for git in ({"git_sha": "a" * 40, "git_dirty": True}, {"git_sha": "a" * 40, "git_dirty": None},
                {"git_sha": None, "git_dirty": False}):
        with pytest.raises(RuntimeError, match="clean, committed git tree"):
            b.require_clean_tree(git, b.CANONICAL_OUTPUT)
    # The guard runs before the source is read or anything is written.
    args = b.parse_args(["--postprocess-from-git", SOURCE_COMMIT])
    monkeypatch.setattr(b.h, "git_info", lambda: {"git_sha": "a" * 40, "git_dirty": True})
    monkeypatch.setattr(b, "postprocess_from_git", lambda *a, **k: pytest.fail("the source was read before the guard"))
    before = RESULT.read_bytes() if RESULT.exists() else None
    with pytest.raises(RuntimeError, match="clean, committed git tree"):
        b.main(["--postprocess-from-git", SOURCE_COMMIT, "--silent"])
    assert (RESULT.read_bytes() if RESULT.exists() else None) == before and args.output == b.CANONICAL_OUTPUT


def test_canonical_run_requires_an_sm_120_device_and_a_clean_tree(tmp_path, monkeypatch):
    clean = {"git_sha": "a" * 40, "git_dirty": False}
    cpu, out = torch.device("cpu"), tmp_path / "free.json"
    b.require_canonical_host(cpu, {"git_sha": None, "git_dirty": True}, out, False)  # not canonical: no demand
    for output, canonical in ((b.CANONICAL_OUTPUT, True), (tmp_path / "cycle5_c5_02_other.json", False)):
        with pytest.raises(RuntimeError, match="requires a Blackwell"):
            b.require_canonical_host(cpu, clean, output, canonical)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for capability, ok in (((9, 0), False), ((10, 0), False), ((12, 0), True), ((12, 1), True)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, c=capability: c)
        if ok:
            b.require_canonical_host(torch.device("cuda"), clean, b.CANONICAL_OUTPUT, True)
        else:
            with pytest.raises(RuntimeError, match="requires a Blackwell"):
                b.require_canonical_host(torch.device("cuda"), clean, b.CANONICAL_OUTPUT, True)
    for git in ({"git_sha": "a" * 40, "git_dirty": True}, {"git_sha": "a" * 40, "git_dirty": None},
                {"git_sha": None, "git_dirty": False}, {"git_sha": "xyz", "git_dirty": False}):
        with pytest.raises(RuntimeError, match="clean, committed git tree"):
            b.require_canonical_host(torch.device("cuda"), git, b.CANONICAL_OUTPUT, True)


def test_canonical_guard_runs_before_any_data_or_gpu_work(tmp_path, monkeypatch):
    args = b.parse_args([])
    monkeypatch.setattr(b.h, "git_info", lambda: {"git_sha": "a" * 40, "git_dirty": True})
    monkeypatch.setattr(b, "Shards", lambda *a, **k: pytest.fail("data was opened before the guard"))
    with pytest.raises(RuntimeError, match="requires a Blackwell|clean, committed git tree"):
        b.run(args)


def test_code_hash_is_the_transitive_import_closure(monkeypatch, tmp_path):
    modules = b.code_modules()
    for needed in ("research.quant_ptq.bench_c5_02_isoloss_ptq_mitigation", "research.quant_ptq.bench_isoloss_ptq",
                   "research.quant_ptq.micro_rotated_ptq", "research.quant_ptq.nvfp4_inference_blackwell",
                   "research.quant_ptq.bench_nvfp4_e2e", "research.polar_opt.harness_tier2_real",
                   "research.polar_opt.harness_tier2_lm", "research.polar_opt.real_text_data",
                   "research.polar_opt.tier2_candidates", "research.polar_opt.schatten_switch_blackwell",
                   "research.polar_opt.polar_switch_blackwell", "research.polar_opt.kernels_blackwell"):
        assert needed in modules
    assert list(modules) == sorted(set(modules)) and all(b._module_path(m).is_file() for m in modules)
    original, real_path = b.code_sha256(), b._module_path
    for name in ("research.polar_opt.kernels_blackwell", "research.polar_opt.tier2_candidates",
                 "research.quant_ptq.micro_rotated_ptq", "research.quant_ptq.bench_c5_02_isoloss_ptq_mitigation"):
        edited = tmp_path / (name + ".py")
        edited.write_bytes(real_path(name).read_bytes() + b"\n# edit\n")
        monkeypatch.setattr(b, "_module_path",
                            lambda n, name=name, edited=edited: edited if n == name else real_path(n))
        assert b.code_sha256() != original
    monkeypatch.setattr(b, "_module_path", real_path)
    assert b.code_sha256() == original


def test_import_closure_and_code_hash_of_the_producer_commit_come_from_its_git_blobs():
    """The v1 producer 0355d16: 16 modules and the code hash that the result recorded (audit, provenance)."""
    producer = "0355d16f258133b5fb7aec8e90fd223f781595e9"
    modules = b._code_modules_at_commit(producer)
    assert len(modules) == 16 and list(modules) == sorted(modules) and b.DRIVER_MODULE in modules
    assert b._code_sha256_at_commit(producer, modules) == \
        "424055b21b6f894ef850151501425839e5731e9919ac7399138d75788f7dc85b"
    assert b._code_sha256_at_commit(producer, modules[:-1]) != b._code_sha256_at_commit(producer, modules)
    for bad in ("0" * 40, "HEAD", "xyz"):
        with pytest.raises(ValueError, match="not a commit object"):
            b._code_modules_at_commit(bad)
    tree = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{producer}^{{tree}}"], capture_output=True,
                          text=True, check=True).stdout.strip()
    with pytest.raises(ValueError, match="not a commit object"):  # a tree object is not a commit
        b._require_git_commit(tree, "git_sha")
    with pytest.raises(ValueError, match="cannot read"):
        b._git_blob(producer, b.RESULT_REPO_PATH)  # the producer commit has no result file yet


def test_resume_key_covers_code_runtime_data_and_the_seed_specific_config():
    protocol, config = copy.deepcopy(b.CANONICAL), {"width": 512}
    env = {"torch": "2.14.1+cu130", "triton": "3.8.0", "cuda": "13.0", "driver": "615.78.08",
           "device": "NVIDIA GeForce RTX 5080", "capability": [12, 0], "python": "3.13"}
    data = dict(q.EXPECTED_SHA256)
    base = b.unit_config_hash(protocol, config, "c" * 64, env, data)
    assert b._is_hex(base, 16)
    assert b.unit_config_hash({**protocol, "seeds": [1]}, config, "c" * 64, env, data) == base  # per-seed units
    assert b.unit_config_hash(protocol, config, "c" * 64, {**env, "python": "3.14"}, data) == base
    changed = [b.unit_config_hash(protocol, config, "d" * 64, env, data),
               b.unit_config_hash(protocol, {"width": 768}, "c" * 64, env, data),
               b.unit_config_hash(protocol, config, "c" * 64, env, {**data, "val": "0" * 64}),
               *(b.unit_config_hash(protocol, config, "c" * 64, {**env, key: "other"}, data) for key in b.ENV_KEYS),
               *(b.unit_config_hash({**protocol, key: value}, config, "c" * 64, env, data)
                 for key, value in (("calibration_seed", 1), ("tolerance_nats", 0.03), ("butterfly_steps", 10),
                                    ("adamw", {**protocol["adamw"], "initial_steps": 480}),
                                    ("awq", {**protocol["awq"], "alphas": [0.0, 0.5]})))]
    assert base not in changed and len(set(changed)) == len(changed)
    run = b.run_config_hash(False, protocol, config, "c" * 64, "a" * 40, env, data)
    assert run != b.run_config_hash(False, {**protocol, "seeds": [1]}, config, "c" * 64, "a" * 40, env, data)
    assert run != b.run_config_hash(False, protocol, config, "c" * 64, "b" * 40, env, data)
    assert run != b.run_config_hash(True, protocol, config, "c" * 64, "a" * 40, env, data)
    # The schema version is part of both digests: a v1 source keeps its v1 hashes after a postprocess.
    assert b.SCHEMA_VERSION == "cycle5_c5_02.v2" and b.SOURCE_SCHEMA_VERSION == "cycle5_c5_02.v1"
    assert base == b.unit_config_hash(protocol, config, "c" * 64, env, data, b.SCHEMA_VERSION)
    assert base != b.unit_config_hash(protocol, config, "c" * 64, env, data, b.SOURCE_SCHEMA_VERSION)
    assert run != b.run_config_hash(False, protocol, config, "c" * 64, "a" * 40, env, data, b.SOURCE_SCHEMA_VERSION)


def test_unit_log_resumes_by_key_and_hash(tmp_path):
    path = tmp_path / "out.json.units.jsonl"
    assert b.unit_log_path(tmp_path / "out.json") == path
    log = b.UnitLog(path, "hash-a", resume=False)
    assert log.get(("stage_a", 1)) is None
    stored = log.put(("stage_a", 1), {"value": 1.5, "tuple": (1, 2)})
    assert stored == {"value": 1.5, "tuple": [1, 2], "resumed_from_unit_log": False}
    log.put(("stage_b", 1, b.MUON), {"checkpoint_sha256": "x"})
    with path.open("a") as handle:
        handle.write('{"key": ["stage_a", 2, "hash-a"], "value": {"cut')  # a line cut off by an interruption
    again = b.UnitLog(path, "hash-a", resume=True)
    assert again.get(("stage_a", 1)) == {"value": 1.5, "tuple": [1, 2], "resumed_from_unit_log": True}
    assert again.get(("stage_a", 2)) is None and again.get(("stage_b", 1, b.ADAMW)) is None
    assert again.get(("stage_b", 1, b.MUON), lambda v: v["checkpoint_sha256"] == "y") is None  # unusable unit
    assert again.get(("stage_b", 1, b.MUON), lambda v: v["checkpoint_sha256"] == "x") is not None
    assert (again.loaded, again.run) == (2, 0)
    assert b.UnitLog(path, "hash-b", resume=True).get(("stage_a", 1)) is None  # other code, runtime or protocol
    assert b.UnitLog(path, "hash-a", resume=False).units == {} and path.read_text() == ""  # no --resume: start empty
    assert b.UnitLog(None, "hash-a", resume=True).put(("stage_a", 1), {"v": 1})["v"] == 1


# ------------------------------------------------------------------ pipeline
def _quick_argv(tmp_path, *extra):
    data_dir = tmp_path / "shards"
    if not data_dir.exists():
        data_dir.mkdir()
        _fake_data(data_dir, sizes=(200_000, 20_000, 20_000))
    return ["--quick", "--no-verify", "--silent", "--data-dir", str(data_dir), "--device", DEVICE,
            "--output", str(tmp_path / "quick.json"), *extra]


@pytest.fixture(scope="module")
def quick_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("c5_02_quick")


@pytest.fixture(scope="module")
def quick(quick_dir):
    """One five-seed --quick run. The unit log and the checkpoints stay for the resume tests."""
    canonical_before = RESULT.read_bytes() if RESULT.exists() else None
    args = b.parse_args(_quick_argv(quick_dir))
    payload = b.c4.write_json_atomic(args.output, b.run(args))
    assert json.loads(args.output.read_text()) == payload
    assert (RESULT.read_bytes() if RESULT.exists() else None) == canonical_before
    return payload


def test_quick_pipeline_end_to_end(quick, quick_dir):
    assert b.validate_results(quick) == NONCANONICAL
    assert quick["schema_version"] == "cycle5_c5_02.v2" and quick["quick"] and not quick["canonical"]
    assert quick["postprocess"] is None and list(quick)[-4:] == ["postprocess", "analysis", "derived", "caveats"]
    assert quick["protocol"]["seeds"] == [1, 2, 3, 4, 5] and quick["meta"]["splits_opened"] == ["train", "val", "test"]
    assert quick["meta"]["unit_log"] == {"resume": False, "units_loaded": 0, "units_run": 25}
    stage_a, stage_b = quick["stage_a"], quick["stage_b"]
    assert stage_a["split_view"] == stage_a["splits_opened_at_end"] == ["train", "val"]
    assert [(c["seed"], c["arm"]) for c in stage_b["checkpoints"]] == [(s, a) for s in range(1, 6) for a in b.ARMS]
    hardware = stage_b["checkpoints"][0]["hardware_w4a4"]["available"]
    assert hardware == FP4_GEMM
    for unit in stage_a["seeds"]:
        assert not any("test" in key for key in b._keys(unit))
        assert [r["arm"] for r in unit["rows"]] == [b.MUON, b.COOL_ADAMW, b.COOL_QAT, b.ADAMW]
        assert unit["selection"]["adamw_evaluations"] == len(unit["rows"][3]["evaluations"]) >= 1
        assert unit["selection"]["t_adamw"] == unit["rows"][3]["steps"] == unit["rows"][3]["tokens"] // 512
    final = {u["seed"]: u["checkpoints"] for u in stage_a["seeds"]}
    for c in stage_b["checkpoints"]:
        entry = final[c["seed"]][c["arm"]]
        assert c["checkpoint_sha256"] == c["checkpoint_sha256_after_ptq"] == entry["sha256"]
        assert c["baseline_val"]["nll"] == pytest.approx(entry["val_nll"], abs=1e-9)  # the saved weights, not a retrain
        assert (quick_dir / "quick.json.ckpt" / c["checkpoint_file"]).is_file()
        assert c["baseline_val"]["tokens"] == c["baseline_test"]["tokens"] == 16 * 64
        assert [(r["format"], r["method"], r["mode"]) for r in c["grid"]] == b.expected_combos(
            c["arm"], quick["protocol"], hardware)
        awq = [r for r in c["grid"] if r["activation_aware_scaling"]]
        assert bool(awq) == (c["arm"] in b.AWQ_ARMS) and len(c["awq_selection"]) == (8 if awq else 0)
        for r in awq:
            assert r["awq_alpha"] in quick["protocol"]["awq"]["alphas"]
        for r in c["grid"]:
            if r["mode"].endswith("fakequant"):
                assert len(r["geometry"]["output_rel_error"]) == 10 and r["output_rel_error_mean"] > 0
        assert len(c["diagnostics"]["per_layer"]) == 10
        assert all(row["weight_stable_rank"] >= 1 for row in c["diagnostics"]["per_layer"])
    analysis = quick["analysis"]
    assert [c["name"] for c in analysis["contrasts"]] == [
        "muon_vs_adamw", "adamw_cooldown_vs_muon", "adamw_cooldown_vs_adamw", "qat_cooldown_vs_muon",
        "qat_cooldown_vs_adamw", "awq_vs_muon", "awq_vs_adamw", "awq_on_adamw"]
    cells = 6 if hardware else 4
    selection = analysis["method_selection"]
    for contrast in analysis["contrasts"]:
        for family in b.FAMILIES:
            assert len(contrast[family]) == cells
            for entry in contrast[family]:
                block = entry["test"]["delta_nll_diff"]
                assert block["n"] == 5 and block["df"] == 4 and block["p_sign_flip"] >= 0.0625
                assert block["p_holm"] >= block["p_two_sided"] - 1e-15
                assert block["p_holm_across_contrasts"] >= block["p_holm"] - 1e-15
        for entry in contrast["own_best"]:  # each side at the method that its own arm selects on val
            key = f"{entry['format']}/{entry['mode']}"
            side_x = selection["per_arm_awq" if contrast["kind"] == "awq" else "per_arm"][contrast["arm_x"]]
            assert entry["method_x"] == side_x[key]["best_by_val"]
            assert entry["method_y"] == selection["per_arm"][contrast["arm_y"]][key]["best_by_val"]
        if contrast["kind"] == "awq":
            assert contrast["bf16_nll_diff"] is None and len(contrast["cells"]) == 2 * cells
            assert all(c["method_x"] == "awq_" + c["method_y"] for c in contrast["cells"])
        else:
            assert contrast["bf16_nll_diff"]["test"]["n"] == 5
    iso = analysis["iso_loss"]
    assert len(iso["pairs"]) == 5 and all(p["same_init_and_stream"] and p["stage_b_val_equals_stage_a"]
                                          for p in iso["pairs"])
    assert quick["derived"]["checkpoints"] == 20 and quick["derived"]["analysis_available"] is True
    for family in b.FAMILIES:
        assert set(quick["derived"][f"{family}_test"]) == {c["name"] for c in analysis["contrasts"]}
    assert len(quick["caveats"]) >= 9


def test_alpha_zero_rows_are_the_plain_rows_and_other_alphas_are_new_measurements(quick):
    seen = set()
    for c in quick["stage_b"]["checkpoints"]:
        rows = {(r["format"], r["method"], r["mode"]): r for r in c["grid"]}
        for (fmt, method, mode), r in rows.items():
            if not r["activation_aware_scaling"]:
                continue
            twin = rows[(fmt, method.removeprefix("awq_"), mode)]
            seen.add(r["awq_alpha"] == 0)
            if r["awq_alpha"] == 0:
                assert all(r[f"{s}_nll"] == twin[f"{s}_nll"] for s in b.SPLITS)
            elif mode.endswith("fakequant"):
                assert r["geometry"] != twin["geometry"]
        for s in c["awq_selection"]:
            assert s["selected_alpha"] == b.select_alpha(s["alphas"], s["val_nll"])
            assert s["val_tokens"] == 16 * 64 and s["selected_on"] == "val"
            chosen = rows[(s["format"], s["method"], b.fakequant_mode(s["format"]))]
            assert chosen["val_nll"] == min(s["val_nll"]) == s["val_nll"][s["alphas"].index(s["selected_alpha"])]
            plain = rows[(s["format"], s["method"].removeprefix("awq_"), b.fakequant_mode(s["format"]))]
            assert s["val_nll"][0] == plain["val_nll"] >= chosen["val_nll"]  # never worse than alpha = 0 on val
    assert seen  # the grid holds activation-aware rows


@pytest.mark.skipif(not FP4_GEMM, reason="needs native NVFP4 scaled_mm (sm_100+)")
def test_quick_pipeline_runs_real_hardware_w4a4(quick):
    hw = quick["analysis"]["hardware"]
    assert hw["fp4_gemms_per_forward"] == [10] and hw["fp4_gemms_per_forward_emulated"] == [0]
    assert hw["fp4_gemms_per_forward_bf16_baseline"] == [0]
    assert hw["w4a4_rows"] == 5 * (2 * 12 + 2 * 8) and hw["decode_exact_rows"] == hw["decode_checked_rows"] == 100
    assert hw["decode_checked_matrices"] == 1000
    # H2: rows with alpha = 0 are copies. Unique executions are counted next to the totals.
    rows = [r for c in quick["stage_b"]["checkpoints"] for r in c["grid"]]
    copied_hw = sum(r.get("awq_alpha") == 0 and r["mode"] in b.HW_MODES for r in rows)
    copied_decode = sum(r.get("awq_alpha") == 0 and "hw_weights_checked_matrices" in r for r in rows)
    assert copied_hw == 2 * copied_decode
    assert (hw["w4a4_inherited_zero_alpha_rows"], hw["w4a4_unique_profiled_rows"]) == (copied_hw, 200 - copied_hw)
    assert (hw["decode_inherited_zero_alpha_rows"], hw["decode_unique_checked_rows"]) == (copied_decode,
                                                                                          100 - copied_decode)
    assert hw["decode_inherited_zero_alpha_matrices"] == 10 * copied_decode
    assert hw["decode_unique_checked_matrices"] == 1000 - 10 * copied_decode
    assert {k: quick["derived"]["hardware"][k] for k in hw} == hw
    parity = quick["analysis"]["hardware_vs_emulation"]
    assert len(parity["rows"]) == 20 and parity["passed"] and parity["max_abs_nll_difference"] <= 1e-3


def test_resume_reuses_every_unit_and_recomputes_after_a_lost_checkpoint(quick, quick_dir):
    args = b.parse_args(_quick_argv(quick_dir, "--resume"))
    resumed = b.run(args)
    assert resumed["meta"]["unit_log"] == {"resume": True, "units_loaded": 25, "units_run": 0}
    assert all(u["resumed_from_unit_log"] for u in resumed["stage_a"]["seeds"] + resumed["stage_b"]["checkpoints"])
    strip = lambda p: [{k: v for k, v in u.items() if k != "resumed_from_unit_log"}  # noqa: E731
                       for u in p["stage_a"]["seeds"] + p["stage_b"]["checkpoints"]]
    assert strip(resumed) == strip(quick) and resumed["analysis"] == quick["analysis"]
    # A Stage A unit without its saved weights is not usable: the seed is trained again.
    (quick_dir / "quick.json.ckpt" / b.checkpoint_name(5, b.ADAMW)).unlink()
    again = b.run(b.parse_args(_quick_argv(quick_dir, "--resume")))
    log = again["meta"]["unit_log"]
    assert log["units_loaded"] + log["units_run"] == 25 and 1 <= log["units_run"] <= 5 and log["units_loaded"] >= 20
    assert again["stage_a"]["seeds"][4]["resumed_from_unit_log"] is False
    assert all(u["resumed_from_unit_log"] for u in again["stage_a"]["seeds"][:4])
    assert b.validate_results(again) == NONCANONICAL


def _resume(quick_dir):
    return b.run(b.parse_args(_quick_argv(quick_dir, "--resume")))


def _verified(payload, quick_dir):
    return all(b.checkpoints_verified(unit, quick_dir / "quick.json.ckpt") for unit in payload["stage_a"]["seeds"])


def test_resume_regenerates_a_seed_whose_retained_checkpoint_is_unreadable_or_has_other_weights(quick, quick_dir):
    """H3, Stage A and Stage B units both cached: the retained file is still there, but it is not the weights.

    File presence passed the old check, and the cached Stage B units were then trusted by their recorded
    digest alone. Now the tensor digest of every retained file is recomputed before a unit is reused.
    """
    ckpt = quick_dir / "quick.json.ckpt"
    before = _resume(quick_dir)
    assert before["meta"]["unit_log"] == {"resume": True, "units_loaded": 25, "units_run": 0}
    damaged = ckpt / b.checkpoint_name(2, b.MUON)
    donor = ckpt / b.checkpoint_name(3, b.MUON)
    for content in (b"unreadable", donor.read_bytes()):  # corrupt bytes, then valid weights of another seed
        damaged.write_bytes(content)
        assert damaged.is_file() and not _verified(before, quick_dir)
        again = _resume(quick_dir)
        log = again["meta"]["unit_log"]
        assert log["units_loaded"] + log["units_run"] == 25 and 1 <= log["units_run"] <= 5
        assert again["stage_a"]["seeds"][1]["resumed_from_unit_log"] is False  # seed 2 is trained again
        assert all(u["resumed_from_unit_log"] for i, u in enumerate(again["stage_a"]["seeds"]) if i != 1)
        for record in again["stage_b"]["checkpoints"]:  # a reused Stage B unit has a verified retained file
            entry = again["stage_a"]["seeds"][record["seed"] - 1]["checkpoints"][record["arm"]]
            assert record["checkpoint_sha256"] == entry["sha256"]
            assert b.checkpoint_file_matches_sha256(ckpt / entry["file"], entry["sha256"])
        assert _verified(again, quick_dir) and b.validate_results(again) == NONCANONICAL
        before = again


@pytest.mark.parametrize("content", ["unreadable", "other weights"])
def test_resume_fails_when_a_checkpoint_is_lost_after_stage_a_and_only_stage_b_is_cached(quick, quick_dir,
                                                                                         monkeypatch, content):
    """H3, Stage B unit cached: the Stage A check passed, then the retained file changed.

    The cached Stage B unit still carries the recorded digest. It is not reused: the unit is computed
    again, and loading the unusable file raises instead of returning the cached scores.
    """
    ckpt = quick_dir / "quick.json.ckpt"
    target, donor = ckpt / b.checkpoint_name(4, b.COOL_QAT), ckpt / b.checkpoint_name(1, b.COOL_QAT)
    original, real = target.read_bytes(), b.checkpoints_verified

    def verified_then_lost(unit, checkpoint_dir):
        usable = real(unit, checkpoint_dir)
        if unit["seed"] == 4:  # after the Stage A check of this seed, before its Stage B units are read
            target.write_bytes(b"unreadable" if content == "unreadable" else donor.read_bytes())
        return usable
    monkeypatch.setattr(b, "checkpoints_verified", verified_then_lost)
    try:
        with pytest.raises(RuntimeError, match="cannot be read" if content == "unreadable"
                           else "differs from the Stage A record"):
            _resume(quick_dir)
    finally:
        target.write_bytes(original)
    monkeypatch.setattr(b, "checkpoints_verified", real)
    restored = _resume(quick_dir)
    assert restored["meta"]["unit_log"] == {"resume": True, "units_loaded": 25, "units_run": 0}


def _as_canonical(payload, monkeypatch):
    """Dress a --quick payload as a canonical one, so that every canonical branch of the validator runs."""
    p = copy.deepcopy(payload)
    hashes = {k: v["sha256"] for k, v in p["data"]["splits"].items()}
    fake_sha = "a" * 40
    orig_require, orig_modules, orig_code, orig_caps, orig_summary = (
        b._require_git_commit, b._code_modules_at_commit, b._code_sha256_at_commit,
        b._driver_capabilities_at_commit, b._matches_current_summary_code)
    monkeypatch.setattr(b, "CANONICAL", copy.deepcopy(p["protocol"]))
    monkeypatch.setattr(b, "CANONICAL_DATA", {"source": copy.deepcopy(p["data"]["source"]),
                                              "encoding": p["data"]["encoding"],
                                              "tokens": {k: v["tokens"] for k, v in p["data"]["splits"].items()}})
    monkeypatch.setattr(b.q, "EXPECTED_SHA256", hashes)
    monkeypatch.setattr(b, "_require_git_commit",
                        lambda sha, label: None if sha == fake_sha else orig_require(sha, label))
    monkeypatch.setattr(b, "_code_modules_at_commit",
                        lambda sha: b.code_modules() if sha == fake_sha else orig_modules(sha))
    monkeypatch.setattr(b, "_code_sha256_at_commit",
                        lambda sha, modules: p["meta"]["code_sha256"] if sha == fake_sha else orig_code(sha, modules))
    monkeypatch.setattr(b, "_driver_capabilities_at_commit",
                        lambda sha: (b.SCHEMA_VERSION, True) if sha == fake_sha else orig_caps(sha))
    monkeypatch.setattr(b, "_matches_current_summary_code",
                        lambda sha: True if sha == fake_sha else orig_summary(sha))
    p["quick"], p["canonical"] = False, True
    p["data"].update(sha256_verified=True, expected_sha256=dict(hashes))
    p["meta"].update(git_sha=fake_sha, git_dirty=False,
                     checkpoint_dir=str(b.CANONICAL_CHECKPOINT_DIR),
                     command=["/venv/bin/python", f"research/quant_ptq/{b.SCRIPT_NAME}"])
    p["run_config_hash"] = b.run_config_hash(False, p["protocol"], p["config"], p["meta"]["code_sha256"], fake_sha,
                                             p["meta"]["environment"], hashes)
    return p


@pytest.mark.skipif(not FP4_GEMM, reason="a canonical result needs hardware W4A4 rows (sm_100+)")
def test_canonical_branches_of_the_validator_run_on_a_dressed_quick_payload(quick, monkeypatch):
    if torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("a canonical result needs an sm_120+ device")
    p = _as_canonical(quick, monkeypatch)
    certified = {"mode": "canonical", "certified_canonical": True, "postprocessed": False}
    assert b.validate_results(p, canonical=True) == certified
    assert b.validate_results(p) == certified  # the payload's own flag selects the canonical checks
    with pytest.raises(ValueError, match="labeled canonical"):
        b.validate_results(p, canonical=False)
    _rejects(p, lambda x: _set(x, ["meta", "git_dirty"], True), "clean tree", canonical=True)
    _rejects(p, lambda x: _set(x, ["meta", "environment", "capability"], [10, 0]), "sm_120|unit_config_hash",
             canonical=True)
    _rejects(p, lambda x: _set(x, ["meta", "command"], ["python", "other.py"]), "canonical invocation",
             canonical=True)
    _rejects(p, lambda x: _set(x, ["meta", "command"], [*x["meta"]["command"], "--seeds", "1"]),
             "valid runnable invocation|canonical protocol", canonical=True)
    _rejects(p, lambda x: _set(x, ["meta", "code_modules"], x["meta"]["code_modules"][:-1]), "import closure",
             canonical=True)
    _rejects(p, lambda x: _set(x, ["meta", "code_sha256"], "0" * 64), "unit_config_hash|committed git blobs",
             canonical=True)
    _rejects(p, lambda x: _set(x, ["data", "sha256_verified"], False), "pinned|expected_sha256", canonical=True)
    _rejects(p, lambda x: _set(x, ["meta", "gpu_telemetry", "end"], None), "gpu_telemetry", canonical=True)
    _rejects(p, lambda x: _set(x, ["stage_b", "checkpoints", 0, "baseline_test", "tokens"], 512),
             "does not cover the full test.bin", canonical=True)
    # Canonical data identity, orthogonalizer and call-site certification.
    _rejects(p, lambda x: _set(x, ["data", "encoding"], "cl100k_base"), "data.encoding", canonical=True)
    _rejects(p, lambda x: _set(x, ["data", "source", "name"], "other corpus"), "data.source", canonical=True)
    _rejects(p, lambda x: _set(x, ["data", "splits", "train", "tokens"], 200_064),
             "split token counts|Calibration", canonical=True)

    def cpu_stand_in(x):
        for unit in x["stage_a"]["seeds"]:
            for row in unit["rows"][:3]:
                row["orthogonalizer"] = "polar_express eager BF16 (CPU stand-in: the SS3_54 kernel needs CUDA)"
    _rejects(p, cpu_stand_in, "one orthogonalizer", canonical=True)
    # CLI metadata consistency on a fresh canonical run (Astra re-check secondary findings).
    _rejects(p, lambda x: _set(x, ["meta", "command"], [*x["meta"]["command"], "--resume"]),
             "--resume flag differs", canonical=True)
    _rejects(p, lambda x: _set(x, ["meta", "command"], [*x["meta"]["command"], "--checkpoint-dir", "elsewhere"]),
             "--checkpoint-dir differs", canonical=True)
    _rejects(p, lambda x: _set(x, ["meta", "command"], [*x["meta"]["command"], "--data-dir", "elsewhere"]),
             "--data-dir differs", canonical=True)
    _rejects(p, lambda x: _set(x, ["meta", "command"], [*x["meta"]["command"], "--checkpoint-d", "elsewhere"]),
             "valid runnable invocation", canonical=True)
    _rejects(p, lambda x: _set(x, ["meta", "command"], [*x["meta"]["command"], "--data-d", "elsewhere"]),
             "valid runnable invocation", canonical=True)
    _rejects(p, lambda x: _set(x, ["meta", "checkpoint_dir"], "/tmp/elsewhere"),
             "canonical checkpoint directory", canonical=True)
    # A v1 producer commit (0355d16) or an earlier v2 commit (9473f90) cannot be certified as a current fresh v2 producer.
    v1_sha = "0355d16f258133b5fb7aec8e90fd223f781595e9"
    v1_code = "424055b21b6f894ef850151501425839e5731e9919ac7399138d75788f7dc85b"
    old_v2_sha = "9473f90b3eb340cb162cb5d6f96bc6ec3e4dff32"
    old_v2_code = "12ded47a9e5dd26c0e893285aad190f3f40e9ecf36d4e2bc1a17438a1c98ba33"

    def claim_commit_producer(git_sha, code_sha):
        def edit(x):
            hashes = {k: v["sha256"] for k, v in x["data"]["splits"].items()}
            x["meta"].update(git_sha=git_sha, code_sha256=code_sha)
            x["meta"]["unit_config_hash"] = b.unit_config_hash(
                x["protocol"], x["config"], code_sha, x["meta"]["environment"], hashes, b.SCHEMA_VERSION)
            x["run_config_hash"] = b.run_config_hash(
                False, x["protocol"], x["config"], code_sha, git_sha, x["meta"]["environment"], hashes,
                b.SCHEMA_VERSION)
        return edit
    _rejects(p, claim_commit_producer(v1_sha, v1_code), "declares schema 'cycle5_c5_02.v1'", canonical=True)
    _rejects(p, claim_commit_producer(old_v2_sha, old_v2_code), "summary code differs from the validating checkout",
             canonical=True)


def test_a_rejected_payload_is_kept_next_to_the_output(quick, quick_dir, monkeypatch):
    args = b.parse_args(_quick_argv(quick_dir, "--resume"))

    def reject(payload, canonical=None):
        raise ValueError("rejected for the test")
    monkeypatch.setattr(b, "validate_results", reject)
    before = args.output.read_bytes()
    with pytest.raises(ValueError, match="rejected for the test"):
        b.run(args)
    kept = b.rejected_path(args.output)
    assert kept.name == "quick.json.rejected" and args.output.read_bytes() == before
    assert json.loads(kept.read_text())["stage_b"]["checkpoints"][0]["grid"]
    kept.unlink()


def test_main_writes_the_result_and_removes_the_unit_log(quick, tmp_path, monkeypatch):
    out = tmp_path / "result.json"
    log = b.unit_log_path(out)
    log.write_text("")
    monkeypatch.setattr(b, "run", lambda args: quick)
    data_dir = tmp_path / "d"
    data_dir.mkdir()
    payload = b.main(["--quick", "--silent", "--data-dir", str(data_dir), "--output", str(out)])
    assert payload == quick and json.loads(out.read_text()) == quick and not log.exists()


# ------------------------------------------------------------------ validator
def _set(obj, path, value):
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


def _get(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def _edited(payload, edit):
    changed = copy.deepcopy(payload)
    edit(changed)
    return changed


def _rejects(payload, edit, match, canonical=None):
    with pytest.raises((ValueError, KeyError), match=match):
        b.validate_results(_edited(payload, edit), canonical=canonical)


def _refresh(p):
    """Recompute the dependent summaries, as a party that rewrites a leaf consistently would."""
    p["analysis"] = json.loads(json.dumps(b.analyze(p["stage_a"], p["stage_b"], p["protocol"])))
    p["derived"] = json.loads(json.dumps(b.derive(p)))
    p["caveats"] = b.caveats(p)


def test_validate_results_rejects_stage_a_tampering(quick):
    unit = ["stage_a", "seeds", 0]
    adamw = [*unit, "rows", 3]
    measured = quick["stage_a"]["seeds"][0]["rows"][3]["evaluations"]

    def bump(path, delta):
        return lambda p: _set(p, path, _get(p, path) + delta)

    _rejects(quick, bump([*adamw, "val_nll"], 0.5), "val_ppl|nearest to Muon")
    _rejects(quick, bump([*adamw, "evaluations", 0, "val_nll"], 0.5), "val_gap")
    _rejects(quick, bump([*adamw, "evaluations", 0, "val_gap"], 0.5), "val_gap")
    _rejects(quick, bump([*adamw, "evaluations", 0, "step"], 8), "predeclared val grid")
    _rejects(quick, bump([*adamw, "evaluations", 0, "lr_multiplier"], 0.5), "lr_multiplier")
    _rejects(quick, lambda p: _set(p, [*adamw, "evaluations", 0, "checkpoint_sha256"], "xyz"), "measurement fields")
    _rejects(quick, lambda p: _set(p, [*adamw, "evaluations", 0, "test_nll"], 5.0), "test.bin field")
    _rejects(quick, lambda p: _set(p, [*adamw, "evaluations"], []), "no measured step")
    _rejects(quick, bump([*adamw, "trained_steps"], 8), "nearest to Muon")
    _rejects(quick, lambda p: _set(p, [*adamw, "checkpoint_sha256"], "0" * 64), "nearest to Muon")
    _rejects(quick, lambda p: _set(p, [*adamw, "cosine_steps"], 48), "differs from the protocol")
    _rejects(quick, lambda p: _set(p, [*adamw, "lr"], 0.0015), "differs from the protocol")
    _rejects(quick, lambda p: _set(p, [*adamw, "warmup_steps"], 3), "schedule differs")
    _rejects(quick, lambda p: _set(p, [*adamw, "steps"], 36), "token count")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 0, "test_nll"], 5.0), "keys differ|test.bin field")
    _rejects(quick, lambda p: _set(p, [*unit, "test_gap"], 0.0), "keys differ|test.bin field")
    _rejects(quick, lambda p: _set(p, ["stage_a", "split_view"], ["train", "val", "test"]),
             "before Shards.open maps test.bin")
    _rejects(quick, lambda p: _set(p, ["stage_a", "splits_opened_at_end"], ["train", "val", "test"]),
             "before Shards.open maps test.bin")
    _rejects(quick, lambda p: _set(p, [*unit, "split_view"], ["train", "val", "test"]), "must not see test.bin")
    _rejects(quick, lambda p: _set(p, ["meta", "splits_opened"], ["test", "train", "val"]), "order train, val, test")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 0, "auxiliary_lr"], 99.0), "differs from the protocol")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 0, "tokens"], 7), "token count differs")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 0, "evaluated_splits"], ["val", "test"]), "read val only")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 1, "branch", "snapshot_sha256"], "0" * 64),
             "does not branch from the Muon snapshot")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 2, "branch", "from_step"], 0), "does not branch|train curve")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 1, "stream_fingerprint"], "0" * 16), "share the initial state")
    _rejects(quick, lambda p: _set(p, [*unit, "selection", "t_adamw"], 640), "selection")
    _rejects(quick, lambda p: _set(p, [*unit, "selection", "matched"], False), "selection")
    _rejects(quick, lambda p: _set(p, [*unit, "checkpoints", b.ADAMW, "sha256"], "0" * 64), "checkpoints")
    _rejects(quick, lambda p: _get(p, [*unit, "rows"]).pop(), "rows must be")
    _rejects(quick, lambda p: _get(p, ["stage_a", "seeds"]).pop(), "one unit per seed")
    # Val coverage of Stage A in every mode, locked recipes, loss domains, durations and scalar types.
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 0, "val_tokens"], 512), "val.bin is not covered completely")
    _rejects(quick, lambda p: _set(p, [*adamw, "evaluations", 0, "val_tokens"], 512), "measurement fields")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 1, "cooldown", "optimizer"], "Muon"), "cooldown record")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 1, "cooldown", "fake_quant"], "nvfp4_g16"), "cooldown record")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 2, "cooldown", "lr"], 0.001), "cooldown record")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 2, "orthogonalizer"], "another kernel"), "one orthogonalizer")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 0, "orthogonalizer"], "fused AdamW"), "its orthogonalizer")
    _rejects(quick, lambda p: _set(p, [*adamw, "orthogonalizer"], "SS3_54"), "fused AdamW")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 0, "train_loss_note"], "mean of all steps"), "loss note")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 0, "train_loss"], -0.5), "finite and nonnegative")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 0, "train_seconds"], -1.0), "finite and nonnegative")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 0, "train_curve", 1, 1], -2.0), "train curve")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 0, "train_curve", 1, 0], 10.0), "train curve")
    for i, row in enumerate(quick["stage_a"]["seeds"][0]["rows"]):
        if row["train_curve"][-1][0] == row["steps"]:  # the last step is a curve point: both values must agree
            _rejects(quick, bump([*unit, "rows", i, "last_train_loss"], 0.25), "last_train_loss")
    _rejects(quick, lambda p: _set(p, [*adamw, "eval_seconds"], -1.0), "differs from the protocol")
    _rejects(quick, lambda p: _set(p, [*unit, "seconds"], -1.0), "seconds must be")
    _rejects(quick, lambda p: _set(p, ["stage_a", "seconds"], -1.0), "stage_a.seconds")
    _rejects(quick, bump(["stage_a", "seconds"], 1.0), "meta.stage_seconds.stage_a")
    _rejects(quick, lambda p: _set(p, [*unit, "resumed_from_unit_log"], 0), "must be a bool")
    _rejects(quick, lambda p: _set(p, [*unit, "resumed_from_unit_log"], True), "marked as resumed")
    _rejects(quick, lambda p: _set(p, [*unit, "seed"], True), "one unit per seed")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 0, "steps"], 24.0), "schedule differs")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 0, "tokens"], float(quick["stage_a"]["seeds"][0]["rows"][0][
        "tokens"])), "token count differs")
    _rejects(quick, lambda p: _set(p, [*adamw, "evaluations", 0, "step"], float(measured[0]["step"])),
             "predeclared val grid")
    _rejects(quick, lambda p: _set(p, [*unit, "rows", 1, "branch", "cooldown_steps"], 8.0), "does not branch")

    def consistent(p, mutate):
        """Rewrite the AdamW measurements and every field that follows from them."""
        rows = _get(p, [*unit, "rows"])
        mutate(rows[3]["evaluations"], rows[0]["val_nll"])
        for e in rows[3]["evaluations"]:
            e["val_gap"] = e["val_nll"] - rows[0]["val_nll"]

    def stop_too_early(p):  # the last measured step is above Muon and below the step cap
        def mutate(evaluations, muon):
            evaluations[-1]["val_nll"] = muon + 0.3
        consistent(p, mutate)

    def run_on_after_the_crossing(p):  # an earlier measured step was already at or below Muon
        def mutate(evaluations, muon):
            evaluations[0]["val_nll"] = muon - 0.3
        consistent(p, mutate)
    if measured[-1]["step"] < quick["protocol"]["adamw"]["max_steps"]:
        _rejects(quick, stop_too_early, "stopped before it reached Muon")
    if len(measured) > 1:
        _rejects(quick, run_on_after_the_crossing, "went on after it was at or below Muon")


def test_validate_results_rejects_stage_b_tampering(quick):
    c = ["stage_b", "checkpoints", 0]
    grid = quick["stage_b"]["checkpoints"][0]["grid"]
    fake = next(i for i, r in enumerate(grid) if r["method"] == "gptq" and r["mode"].endswith("fakequant"))
    awq = next(i for i, r in enumerate(grid) if r["activation_aware_scaling"] and r["mode"].endswith("fakequant"))
    _rejects(quick, lambda p: _set(p, [*c, "checkpoint_sha256"], "0" * 64), "saved Stage A checkpoint")
    _rejects(quick, lambda p: _set(p, [*c, "checkpoint_sha256_after_ptq"], "0" * 64), "saved Stage A checkpoint")
    _rejects(quick, lambda p: _set(p, [*c, "baseline_val", "nll"], 1.0), "ppl|differs from its Stage A value")
    _rejects(quick, lambda p: _set(p, [*c, "grid", fake, "test_delta_nll"], 0.123), "test_delta_nll")
    _rejects(quick, lambda p: _set(p, [*c, "grid", fake, "val_tokens"], 1), "token count differs")
    _rejects(quick, lambda p: _set(p, [*c, "grid", fake, "test_ppl"], 1.0), "test_ppl")
    _rejects(quick, lambda p: _set(p, [*c, "grid", fake, "rounding"], "rtn"), "method fields differ")
    _rejects(quick, lambda p: _get(p, [*c, "grid"]).pop(), "PTQ grid differs from the protocol")
    _rejects(quick, lambda p: _set(p, [*c, "grid", fake, "geometry", "weight_rel_error", 3], 0.5),
             "weight_rel_error_mean|weight error x error")
    _rejects(quick, lambda p: _set(p, [*c, "grid", fake, "output_rel_error_mean"], 0.5), "output_rel_error_mean")
    _rejects(quick, lambda p: _get(p, [*c, "grid", fake, "geometry", "error_alignment"]).pop(), "one value per layer")
    _rejects(quick, lambda p: _set(p, [*c, "grid", awq, "awq_alpha"], 0.33), "alpha differs from the val selection")
    _rejects(quick, lambda p: _set(p, [*c, "awq_selection", 0, "selected_alpha"], 0.33), "not the val.bin minimum")
    _rejects(quick, lambda p: _set(p, [*c, "awq_selection", 0, "selected_on"], "test"), "not the val.bin minimum")
    _rejects(quick, lambda p: _set(p, [*c, "awq_selection", 0, "alphas"], [0.0, 0.9]), "not the val.bin minimum")
    _rejects(quick, lambda p: _set(p, [*c, "awq_selection", 0, "val_tokens"], 5), "must cover the val windows")

    def other_candidate(p):  # a consistent table whose selected candidate is not the measured row
        table = _get(p, [*c, "awq_selection"])
        target = next(t for t in table if (t["format"], t["method"]) == (grid[awq]["format"], grid[awq]["method"]))
        index = target["alphas"].index(target["selected_alpha"])
        target["val_nll"][index] -= 1e-3
    _rejects(quick, other_candidate, "not the candidate that the val selection measured|must be the unscaled row")
    _rejects(quick, lambda p: _get(p, [*c, "awq_selection"]).pop(), "AWQ selection table")
    _rejects(quick, lambda p: _set(p, [*c, "diagnostics", "per_layer", 0, "name"], "blocks.9.qkv"), "in order")
    _rejects(quick, lambda p: _set(p, [*c, "diagnostics", "per_layer", 0, "weight_stable_rank"], 0.5), "out of range")
    _rejects(quick, lambda p: _set(p, [*c, "diagnostics", "summary", "signal_alignment", "all", "mean"], 9.0),
             "diagnostics.summary")
    _rejects(quick, lambda p: _set(p, [*c, "best", "nvfp4_g16/w4a16_fakequant", "best_by_val"], "rtn"), "best")
    _rejects(quick, lambda p: _set(p, ["stage_b", "calibration", "split"], "test"), "Calibration windows")
    _rejects(quick, lambda p: _set(p, ["stage_b", "calibration", "starts", 0], -64), "Calibration starts")
    _rejects(quick, lambda p: _set(p, ["stage_b", "calibration", "seed"], 1), "Calibration windows")
    _rejects(quick, lambda p: _get(p, ["stage_b", "calibration", "starts"]).reverse(), "Calibration starts")
    _rejects(quick, lambda p: _set(p, ["stage_b", "calibration", "tokens"], 256.0), "Calibration windows")
    # Both baselines cover their split completely, in every mode.
    cover = b.expected_coverage(20_000, 64, 16)
    assert cover == {"tokens": 1024, "windows": 16, "span": [0, 1025], "split_tokens": 20_000}
    assert b.expected_coverage(2_002_330, 512, None) == {"tokens": b.VAL_TOKENS, "windows": 3910,
                                                         "span": [0, 2_001_921], "split_tokens": 2_002_330}
    assert b.expected_coverage(2_000_081, 512, None)["tokens"] == b.TEST_TOKENS
    for split in b.SPLITS:
        assert {k: quick["stage_b"]["checkpoints"][0][f"baseline_{split}"][k] for k in cover} == cover
        for key, value in (("windows", 1), ("span", [0, 1]), ("split_tokens", 2), ("tokens", 1024.0),
                           ("span", [64, 1089])):
            _rejects(quick, lambda p: _set(p, [*c, f"baseline_{split}", key], value),
                     f"does not cover the full {split}.bin")
        _rejects(quick, lambda p: _set(p, [*c, f"baseline_{split}", "documents"], 3), "keys differ")
    # Fitted rotations: one record per kind for every fitted candidate, with consistent contents.
    fits = quick["stage_b"]["checkpoints"][0]["butterfly_fits"]
    key = "nvfp4_g16/adaptive_butterfly_b16_gptq"
    assert key in fits and list(fits[key]) == list(q.LINEAR_KINDS)
    fit = [*c, "butterfly_fits", key, "qkv"]
    _rejects(quick, lambda p: _get(p, [*c, "butterfly_fits"]).pop(key), "butterfly_fits must hold")
    _rejects(quick, lambda p: _set(p, [*c, "butterfly_fits", "nvfp4_g16/gptq"], fits[key]), "butterfly_fits must hold")
    _rejects(quick, lambda p: _get(p, [*c, "butterfly_fits", key]).pop("down"), "one fit per projection kind")
    _rejects(quick, lambda p: _get(p, fit).pop("lr"), "keys differ")
    _rejects(quick, lambda p: _set(p, [*fit, "steps"], 150), "steps, LR or layer counts")
    _rejects(quick, lambda p: _set(p, [*fit, "lr"], 0.1), "steps, LR or layer counts")
    _rejects(quick, lambda p: _set(p, [*fit, "layers"], 12), "steps, LR or layer counts")
    _rejects(quick, lambda p: _set(p, [*fit, "layers_improved"], 3), "steps, LR or layer counts|fitted error")
    _rejects(quick, lambda p: _set(p, [*fit, "layers_improved"], True), "steps, LR or layer counts")
    _rejects(quick, lambda p: _set(p, [*fit, "weight_sq_error_fitted"],
                                   2.0 * _get(p, [*fit, "weight_sq_error_hadamard_start"])), "fitted error")
    _rejects(quick, lambda p: _set(p, [*fit, "relative_error_reduction"], 0.5), "relative_error_reduction")

    def unimproved_with_lower_error(p):
        record = _get(p, fit)
        record["weight_sq_error_fitted"] = 0.5 * record["weight_sq_error_hadamard_start"]
        record["relative_error_reduction"], record["layers_improved"] = 0.5, 0
    _rejects(quick, unimproved_with_lower_error, "equal start when layers_improved == 0")

    def one_ulp_increase_when_unimproved(p):
        record = _get(p, fit)
        start = float(np.float32(0.001))
        fitted = float(np.nextafter(np.float32(start), np.float32(1)))
        record["weight_sq_error_hadamard_start"] = start
        record["weight_sq_error_fitted"] = fitted
        record["relative_error_reduction"] = 1.0 - fitted / start
        record["layers_improved"] = 0
    _rejects(quick, one_ulp_increase_when_unimproved, "0 <= fitted <= start")

    def improved_with_rounded_equal_fp32_sum(p):
        record = _get(p, fit)
        record["weight_sq_error_hadamard_start"] = 190367301632.0
        record["weight_sq_error_fitted"] = 190367301632.0
        record["relative_error_reduction"], record["layers_improved"] = 0.0, 1
    assert b.validate_results(_edited(quick, improved_with_rounded_equal_fp32_sum)) == NONCANONICAL

    # Real CPU fit reproducer where 1 layer improves while the FP32 layer sum stays equal.
    g = torch.Generator().manual_seed(7)
    w = torch.randn(2, 1, 16, generator=g)
    rot = q.Butterfly(16, q.Butterfly.initial_angles((2,), 16, 16))
    fmt = q.FORMATS["nvfp4_g16"]
    z = q.quantize_rtn(rot.forward(w), fmt)
    noise = torch.randn(2, 1, 16, generator=g)
    w[:1] = rot.inverse(z + 0.001 * noise)[:1] * 100_000_000
    _, real_fit = q.fit_butterfly(w, fmt, steps=5)
    assert real_fit["layers_improved"] == 1
    assert real_fit["weight_sq_error_hadamard_start"] == real_fit["weight_sq_error_fitted"] == 190367301632.0
    assert real_fit["relative_error_reduction"] == 0.0
    real_accepted = _edited(quick, lambda p: _set(p, fit, {**real_fit, "layers": _get(p, [*fit, "layers"])}))
    assert b.validate_results(real_accepted) == NONCANONICAL
    # Quantizer metadata, durations and exact scalar types of the grid rows.
    rtn = next(i for i, r in enumerate(grid) if r["method"] == "rtn" and r["mode"].endswith("fakequant"))
    assert grid[rtn]["gptq_damping_max"] is None and grid[fake]["gptq_damping_max"] >= b.GPTQ_DAMP
    _rejects(quick, lambda p: _set(p, [*c, "grid", rtn, "gptq_damping_max"], -0.01), "gptq_damping_max")
    _rejects(quick, lambda p: _set(p, [*c, "grid", fake, "gptq_damping_max"], None), "gptq_damping_max")
    for bad in (-0.01, 0.02, 0.0, b.GPTQ_DAMP * 10 ** 8):
        _rejects(quick, lambda p: _set(p, [*c, "grid", fake, "gptq_damping_max"], bad), "gptq_damping_max")
    for k in range(8):  # the damping fallback of q.gptq_quantize: gptq_damp * 10^k
        b.validate_results(_edited(quick, lambda p: _set(p, [*c, "grid", fake, "gptq_damping_max"],
                                                         b.GPTQ_DAMP * 10 ** k)))
    _rejects(quick, lambda p: _set(p, [*c, "grid", fake, "seconds"], -1.0), "seconds must be")
    _rejects(quick, lambda p: _set(p, [*c, "grid", fake, "quantize_seconds"], -1.0), "quantize_seconds")
    _rejects(quick, lambda p: _set(p, [*c, "awq_selection", 0, "seconds"], -1.0), "seconds must be")
    _rejects(quick, lambda p: _set(p, [*c, "seconds"], -1.0), "seconds must be")
    _rejects(quick, lambda p: _set(p, ["stage_b", "seconds"], -1.0), "stage_b.seconds")
    _rejects(quick, lambda p: _set(p, ["stage_b", "seconds"], _get(p, ["stage_b", "seconds"]) + 1.0),
             "meta.stage_seconds.stage_b")
    _rejects(quick, lambda p: _set(p, [*c, "resumed_from_unit_log"], True), "marked as resumed")
    _rejects(quick, lambda p: _set(p, [*c, "resumed_from_unit_log"], 1), "must be a bool")
    _rejects(quick, lambda p: _set(p, [*c, "grid", fake, "activation_aware_scaling"], 0), "method fields differ")
    _rejects(quick, lambda p: _set(p, [*c, "grid", awq, "activation_aware_scaling"], 1), "method fields differ")
    _rejects(quick, lambda p: _set(p, [*c, "grid", fake, "test_tokens"], 1024.0), "token count differs")
    _rejects(quick, lambda p: _set(p, [*c, "seed"], True), "one checkpoint per")
    _rejects(quick, lambda p: _set(p, [*c, "hardware_w4a4", "available"], int(FP4_GEMM)), "hardware_w4a4 keys differ")
    _rejects(quick, lambda p: _set(p, [*c, "hardware_w4a4", "note"], 5), "hardware_w4a4.note")
    _rejects(quick, lambda p: _set(p, [*c, "hardware_w4a4", "note"], "ok" if FP4_GEMM else None), "hardware_w4a4.note")
    _rejects(quick, lambda p: _get(p, ["stage_b", "checkpoints"]).reverse(), "one checkpoint per")

    def zero_alpha_must_be_the_twin(p):
        rows = _get(p, [*c, "grid"])
        sel = _get(p, [*c, "awq_selection"])
        target = next(s for s in sel if (s["format"], s["method"]) == (rows[awq]["format"], rows[awq]["method"]))
        target["val_nll"] = [0.0, 1.0, 1.0, 1.0]
        target["selected_alpha"] = 0.0
        for r in rows:
            if (r["format"], r["method"]) == (rows[awq]["format"], rows[awq]["method"]) and "awq_alpha" in r:
                r["awq_alpha"] = 0.0
        rows[awq]["test_nll"] += 0.25
        rows[awq]["test_ppl"] = math.exp(rows[awq]["test_nll"])
        base = _get(p, [*c, "baseline_test"])
        rows[awq]["test_delta_nll"] = rows[awq]["test_nll"] - base["nll"]
        rows[awq]["test_relative_delta_ppl_pct"] = 100.0 * (rows[awq]["test_ppl"] / base["ppl"] - 1.0)
    _rejects(quick, zero_alpha_must_be_the_twin, "alpha = 0 must be the unscaled row|not the candidate")
    _rejects(quick, lambda p: _set(p, [*c, "awq_selection", 0, "val_nll", 0], 0.0),
             "alpha = 0 must be the unscaled row|not the val.bin minimum")
    if FP4_GEMM:
        hw = next(i for i, r in enumerate(grid) if r["mode"] == "w4a4_hw_dynamic")
        static = next(i for i, r in enumerate(grid) if r["mode"] == "w4a4_hw_static")
        nv = next(i for i, r in enumerate(grid) if r["format"] == "nvfp4_g16" and r["mode"].endswith("fakequant"))
        _rejects(quick, lambda p: _set(p, [*c, "grid", hw, "fp4_gemms_per_forward"], 9), "FP4 GEMMs per forward")
        _rejects(quick, lambda p: _set(p, [*c, "grid", static, "static_scale_calibration"], "test.bin windows"),
                 "train.bin calibration windows")
        _rejects(quick, lambda p: _set(p, [*c, "grid", nv, "hw_weights_decode_to_fakequant_weights"], False),
                 "decode to the fake-quant weights")
        _rejects(quick, lambda p: _set(p, [*c, "grid", nv, "hw_weights_checked_matrices"], 9),
                 "decode to the fake-quant weights")
        _rejects(quick, lambda p: _set(p, [*c, "hardware_w4a4", "fp4_gemms_per_forward_bf16_baseline"], 3),
                 "must not run FP4 GEMMs")
        _rejects(quick, lambda p: _set(p, [*c, "hardware_w4a4", "fp4_gemms_per_forward_bf16_baseline"], False),
                 "must not run FP4 GEMMs")
        _rejects(quick, lambda p: _set(p, [*c, "grid", hw, "fp4_gemms_per_forward"], 10.0), "FP4 GEMMs per forward")
        _rejects(quick, lambda p: _set(p, [*c, "grid", nv, "hw_weights_decode_to_fakequant_weights"], 1),
                 "decode to the fake-quant weights")
        low, high = grid[static]["static_global_scale_range"]
        assert 0 < low <= high
        for scales in ([-0.5, -1.0], [high, low] if low < high else [0.0, high], [0.0, high], [low], [low, high, high],
                       [low, "1.0"], None):
            _rejects(quick, lambda p: _set(p, [*c, "grid", static, "static_global_scale_range"], scales),
                     "static_global_scale_range")


def test_validate_results_rejects_summary_provenance_and_schema_tampering(quick):
    _rejects(quick, lambda p: _set(p, ["analysis", "contrasts", 0, "primary", 0, "test", "delta_nll_diff", "p_holm"],
                                   0.001), "analysis")
    _rejects(quick, lambda p: _set(p, ["analysis", "contrasts", 3, "cells", 0, "test", "nll_diff", "p_sign_flip"],
                                   0.03125), "analysis")
    _rejects(quick, lambda p: _set(p, ["analysis", "iso_loss", "all_seeds_confirmed_on_test"], None), "analysis")
    _rejects(quick, lambda p: _set(p, ["analysis", "t_quantile_0p975_df4"], 4.3), "analysis")
    _rejects(quick, lambda p: _set(p, ["analysis"], None), "analysis")
    _rejects(quick, lambda p: _set(p, ["derived", "iso_loss", "seeds_matched_on_val"], 99), "derived")
    _rejects(quick, lambda p: _set(p, ["derived", "grid_rows"], 1), "derived")
    _rejects(quick, lambda p: _get(p, ["caveats"]).pop(), "caveats")
    _rejects(quick, lambda p: _set(p, ["run_config_hash"], "0" * 16), "run_config_hash")
    _rejects(quick, lambda p: _set(p, ["meta", "code_sha256"], "0" * 64), "unit_config_hash|run_config_hash")
    _rejects(quick, lambda p: _set(p, ["meta", "unit_config_hash"], "0" * 16), "unit_config_hash")
    _rejects(quick, lambda p: _set(p, ["meta", "git_sha"], "xyz"), "git_sha")
    hashes = {k: v["sha256"] for k, v in quick["data"]["splits"].items()}

    def bogus_exploratory_commit(p):
        zero = "0" * 40
        p["meta"]["git_sha"] = zero
        p["run_config_hash"] = b.run_config_hash(
            p["quick"], p["protocol"], p["config"], p["meta"]["code_sha256"], zero,
            p["meta"]["environment"], hashes, b.SCHEMA_VERSION)
    _rejects(quick, bogus_exploratory_commit, "not a commit object")
    _rejects(quick, lambda p: _set(p, ["meta", "extra"], 1), "meta keys differ")
    _rejects(quick, lambda p: _set(p, ["meta", "numerics", "allow_tf32"], True), "full-precision numerics")
    _rejects(quick, lambda p: _set(p, ["meta", "unit_log", "units_run"], 3), "unit_log")
    _rejects(quick, lambda p: _set(p, ["meta", "total_seconds"], 0.001), "total_seconds")
    _rejects(quick, lambda p: _set(p, ["meta", "timestamp_end"], "yesterday"), "invalid ISO timestamp")
    _rejects(quick, lambda p: _set(p, ["meta", "command", 0], "bash"), "Python invocation")
    _rejects(quick, lambda p: _set(p, ["meta", "code_modules"], ["research.quant_ptq.micro_rotated_ptq"]),
             "code_modules")
    _rejects(quick, lambda p: _set(p, ["data", "roles", "matching"], "test"), "split roles")
    _rejects(quick, lambda p: _set(p, ["data", "splits", "val", "sha256"], "other"), "unit_config_hash|run_config_hash")
    _rejects(quick, lambda p: _set(p, ["schema_version"], "cycle4_c4_01b.v1"), "schema or task")
    _rejects(quick, lambda p: _set(p, ["extra"], 1), "top-level keys")
    _rejects(quick, lambda p: _set(p, ["config", "layers"], 3), "model config differs")
    _rejects(quick, lambda p: _set(p, ["protocol", "tolerance_nats"], 5.0), "unit_config_hash|run_config_hash")
    _rejects(quick, lambda p: _set(p, ["protocol", "seeds"], [1, 2, 3, 4, 4]), "distinct integers")
    _rejects(quick, lambda p: _set(p, ["protocol", "seeds"], [True, 2, 3, 4, 5]), "protocol keys or value types")
    _rejects(quick, lambda p: _set(p, ["protocol", "butterfly_steps"], 5.0), "protocol keys or value types")
    _rejects(quick, lambda p: _set(p, ["protocol", "max_eval_windows"], True), "protocol keys or value types")
    _rejects(quick, lambda p: _set(p, ["protocol", "adamw", "max_trials"], 9), "protocol keys or value types")
    _rejects(quick, lambda p: _set(p, ["postprocess"], {}), "postprocess keys")
    _rejects(quick, lambda p: p.pop("postprocess"), "top-level keys")
    _rejects(quick, lambda p: _set(p, ["schema_version"], b.SOURCE_SCHEMA_VERSION), "schema or task")
    _rejects(quick, lambda p: _set(p, ["quick"], 1), "flags must be bool")
    _rejects(quick, lambda p: _set(p, ["canonical"], 0), "flags must be bool")
    _rejects(quick, lambda p: _set(p, ["meta", "git_dirty"], 1), "git_dirty must be a bool")
    _rejects(quick, lambda p: _set(p, ["meta", "unit_log", "resume"], 0), "unit_log")
    _rejects(quick, lambda p: _set(p, ["meta", "unit_log"], {"resume": False, "units_loaded": 1, "units_run": 24}),
             "unit_log")
    _rejects(quick, lambda p: _set(p, ["meta", "unit_log"], {"resume": True, "units_loaded": 1, "units_run": 24}),
             "marked as resumed")
    _rejects(quick, lambda p: _set(p, ["meta", "stage_seconds", "stage_b"], -1.0), "stage_seconds")
    _rejects(quick, lambda p: _set(p, ["meta", "splits_opened"], ["train", "val"]), "order train, val, test")
    # A NUL byte is rejected everywhere, not only in meta.
    for where in (["data", "directory"], ["meta", "checkpoint_dir"], ["task"], ["caveats", 0],
                  ["stage_a", "seeds", 0, "rows", 0, "orthogonalizer"],
                  ["stage_b", "checkpoints", 0, "checkpoint_file"]):
        _rejects(quick, lambda p: _set(p, where, _get(p, where) + "\x00"), "NUL byte")
    _rejects(quick, lambda p: _set(p, ["data", "splits", "val", "sha\x00256"], "x"), "NUL byte")
    # Data description: exact keys and types (the canonical identity is checked in canonical mode).
    _rejects(quick, lambda p: _set(p, ["data", "license"], "unknown"), "data keys")
    _rejects(quick, lambda p: _set(p, ["data", "splits", "val", "documents"], 3), "data.splits.val")
    _rejects(quick, lambda p: _set(p, ["data", "splits", "val", "tokens"], 20_000.0), "data.splits.val")
    _rejects(quick, lambda p: _set(p, ["data", "splits", "test", "tokens"], True), "data.splits.test")
    _rejects(quick, lambda p: _set(p, ["data", "source"], "fineweb"), "data.source")
    _rejects(quick, lambda p: _set(p, ["data", "source", "name"], ["a"]), "data.source")
    _rejects(quick, lambda p: _set(p, ["data", "encoding"], ""), "data.encoding")
    _rejects(quick, lambda p: _set(p, ["data", "encoding"], 2), "data.encoding")
    _rejects(quick, lambda p: _set(p, ["data", "directory"], None), "data.directory")
    _rejects(quick, lambda p: _set(p, ["data", "sha256_verified"], 0), "sha256_verified must be a bool")
    _rejects(quick, lambda p: _set(p, ["data", "sha256_verified"], True), "expected_sha256")
    _rejects(quick, lambda p: _set(p, ["data", "expected_sha256"], {"train": "x", "val": "y", "test": "z"}),
             "expected_sha256")
    # Nested meta records.
    _rejects(quick, lambda p: _set(p, ["meta", "environment", "torch"], 2), "meta.environment invalid")
    _rejects(quick, lambda p: _set(p, ["meta", "environment", "host_cpus"], "16"), "meta.environment invalid")
    _rejects(quick, lambda p: _set(p, ["meta", "environment", "env", "PYTHONPATH"], 1), "meta.environment invalid")
    _rejects(quick, lambda p: _set(p, ["meta", "environment", "env", "HOME"], "/root"), "meta.environment invalid")
    _rejects(quick, lambda p: _set(p, ["meta", "environment", "hostname"], "bazzite"), "meta.environment keys")
    _rejects(quick, lambda p: _get(p, ["meta", "environment"]).pop("numpy"), "meta.environment keys")
    _rejects(quick, lambda p: _set(p, ["meta", "gpu_telemetry", "start"], "idle"), "gpu_telemetry")
    _rejects(quick, lambda p: _set(p, ["meta", "gpu_telemetry", "end"], {"memory.used": 1.0}), "gpu_telemetry")
    _rejects(quick, lambda p: _set(p, ["meta", "gpu_telemetry", "end"], None if CUDA else {"error": "none"}),
             "gpu_telemetry")
    if CUDA:
        _rejects(quick, lambda p: _set(p, ["meta", "environment", "capability"], [12.0, 0]), "device fields")
        _rejects(quick, lambda p: _set(p, ["meta", "environment", "device"], 5080), "device fields")
        _rejects(quick, lambda p: _get(p, ["meta", "environment"]).pop("driver"), "meta.environment keys")
        _rejects(quick, lambda p: _set(p, ["meta", "gpu_telemetry", "end", "memory.used"], None), "gpu_telemetry")
        _rejects(quick, lambda p: _set(p, ["meta", "gpu_telemetry", "end", "fan"], 30.0), "gpu_telemetry")
        b.validate_results(_edited(quick, lambda p: _set(p, ["meta", "gpu_telemetry", "end"], {"error": "timeout"})))
    with pytest.raises(ValueError, match="Out of range float values"):
        nan = copy.deepcopy(quick)
        nan["stage_b"]["checkpoints"][0]["baseline_test"]["nll"] = float("nan")
        b.validate_results(nan)
    # A quick or CPU result is never a canonical one.
    _rejects(quick, lambda p: None, "not labeled canonical", canonical=True)

    def relabel(p):
        p["quick"], p["canonical"] = False, True
    _rejects(quick, relabel, "protocol differs from the canonical", canonical=None)
    _rejects(quick, lambda p: _set(p, ["canonical"], True), "protocol differs from the canonical|cannot be --quick",
             canonical=True)


def test_exact_json_type_helpers():
    assert b._same({"a": [1, 2.5, None, True, "x"]}, {"a": [1, 2.5, None, True, "x"]})
    assert b._same({"a": 1, "b": 2}, {"b": 2, "a": 1})  # key order is not a JSON property
    for found, expected in ((True, 1), (1, True), (1, 1.0), (1.0, 1), (0, False), ([1], (1,)), ({"a": 1}, {"a": 1.0}),
                            ([1, 2], [1]), ({"a": 1}, {"a": 1, "b": 2}), (None, 0), ("1", 1)):
        assert not b._same(found, expected)
    assert b._same_types({"a": [1, 2, 3], "b": None}, {"a": [5], "b": None})
    assert not b._same_types({"a": [1, True]}, {"a": [5]}) and not b._same_types({"a": 1.0}, {"a": 1})
    assert b._is_int(0) and b._is_int(3, 1) and not b._is_int(0, 1)
    assert not any(b._is_int(x) for x in (True, 1.0, "1", None, -1))
    assert b._nonneg(0) and b._nonneg(2.5) and not any(b._nonneg(x) for x in (-1e-9, True, float("inf"), "1", None))
    assert b._has_nul({"a": [{"b": "x\x00"}]}) and b._has_nul({"k\x00": 1}) and not b._has_nul({"a": ["x", 1, None]})


def test_call_site_certification_needs_the_requested_mode_and_the_payload_flag_to_agree(quick):
    """H1: a caller states the mode it expects; a payload labeled canonical never gets relaxed checks."""
    assert b.validate_results(quick) == b.validate_results(quick, canonical=False) == NONCANONICAL
    with pytest.raises(ValueError, match="not labeled canonical"):
        b.validate_results(quick, canonical=True)
    labeled = _edited(quick, lambda p: _set(p, ["canonical"], True))
    with pytest.raises(ValueError, match="labeled canonical: it cannot be validated with the noncanonical checks"):
        b.validate_results(labeled, canonical=False)
    for mode in (None, True):  # the label alone selects every canonical check
        with pytest.raises(ValueError, match="protocol differs from the canonical|cannot be --quick"):
            b.validate_results(labeled, canonical=mode)
    for mode in (0, 1, "canonical"):
        with pytest.raises(ValueError, match="must be a bool or None"):
            b.validate_results(quick, canonical=mode)


def test_validator_recomputes_summaries_but_cannot_attest_consistently_rewritten_leaves(quick):
    """The documented limit: a leaf rewritten together with every dependent summary still passes."""
    p = copy.deepcopy(quick)
    row = next(r for r in p["stage_b"]["checkpoints"][0]["grid"] if r["method"] == "rtn")
    base = p["stage_b"]["checkpoints"][0]["baseline_test"]
    row["test_nll"] += 0.01
    row["test_ppl"] = math.exp(row["test_nll"])
    row["test_delta_nll"] = row["test_nll"] - base["nll"]
    row["test_relative_delta_ppl_pct"] = 100.0 * (row["test_ppl"] / base["ppl"] - 1.0)
    with pytest.raises(ValueError, match="best|analysis"):
        b.validate_results(p)  # the summaries no longer follow from the rows
    p["stage_b"]["checkpoints"][0]["best"] = b.best_methods(p["stage_b"]["checkpoints"][0]["grid"],
                                                           p["protocol"]["base_methods"])
    _refresh(p)
    assert b.validate_results(p) == NONCANONICAL
    assert "not an attestation" in b.validate_results.__doc__ and "Threat model" in b.validate_results.__doc__
    assert "Call-site certification" in b.validate_results.__doc__


def test_analysis_needs_exactly_five_seeds(quick):
    p = copy.deepcopy(quick)
    p["protocol"]["seeds"] = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        b.analyze(p["stage_a"], p["stage_b"], p["protocol"])
    assert b.derive({**quick, "analysis": None})["analysis_available"] is False
    assert "No paired analysis" in b.caveats({**quick, "analysis": None})[0]


# ------------------------------------------------------------------ audit fixtures (C5-02 audit, B1)
def _grid_row(p, predicate):
    """The first grid row, in checkpoint and row order, that satisfies ``predicate`` (None without one)."""
    return next((r for c in p["stage_b"]["checkpoints"] for r in c["grid"] if predicate(r)), None)


def _stage_a_row(p, arm):
    return next(r for r in p["stage_a"]["seeds"][0]["rows"] if r["arm"] == arm)  # seed 1


def _update(target, **values):
    target.update(values)


# The 18 malformed records of the audit table "Field or invariant / Observed accepted fixture". Every one is a
# single edit of an otherwise unmodified result, and the audited validator accepted every one in both modes.
# (name, edit, message of the check that rejects it now, applies to a noncanonical run)
ASTRA_B1_FIXTURES = [
    ("calibration sample: first start 0 with the original seed",
     lambda p: _set(p, ["stage_b", "calibration", "starts", 0], 0), "Calibration starts", True),
    ("full test coverage: one window",
     lambda p: _set(p, ["stage_b", "checkpoints", 0, "baseline_test", "windows"], 1),
     "does not cover the full test.bin", True),
    ("full test coverage: span [0, 1]",
     lambda p: _set(p, ["stage_b", "checkpoints", 0, "baseline_test", "span"], [0, 1]),
     "does not cover the full test.bin", True),
    ("full test coverage: two available split tokens",
     lambda p: _set(p, ["stage_b", "checkpoints", 0, "baseline_test", "split_tokens"], 2),
     "does not cover the full test.bin", True),
    ("calibration schema: undeclared units member",
     lambda p: _set(p, ["stage_b", "calibration", "units"], "tokens"), "Calibration record keys", True),
    ("fitted rotations: empty butterfly_fits",
     lambda p: _set(p, ["stage_b", "checkpoints", 0, "butterfly_fits"], {}), "butterfly_fits must hold", True),
    ("quantizer metadata: negative, reversed static-scale range",
     lambda p: _update(_grid_row(p, lambda r: r["mode"] == "w4a4_hw_static"), static_global_scale_range=[-0.5, -1.0]),
     "static_global_scale_range", True),
    ("quantizer metadata: negative GPTQ damping on an RTN row",
     lambda p: _update(_grid_row(p, lambda r: r["method"] == "rtn"), gptq_damping_max=-0.01),
     "gptq_damping_max", True),
    ("cooldown recipe: AdamW cooldown with matrix LR 0.75",
     lambda p: _set(_stage_a_row(p, b.COOL_ADAMW), ["cooldown", "lr"], 0.75), "cooldown record", True),
    ("cooldown recipe: QAT cooldown without fake quantization",
     lambda p: _set(_stage_a_row(p, b.COOL_QAT), ["cooldown", "fake_quant"], "none"), "cooldown record", True),
    ("training loss: last Muon loss -1",
     lambda p: _update(_stage_a_row(p, b.MUON), last_train_loss=-1.0), "finite and nonnegative", True),
    ("durations: negative Stage B duration",
     lambda p: _set(p, ["stage_b", "seconds"], -1.0), "stage_b.seconds", True),
    ("durations: negative first-row duration",
     lambda p: _set(p, ["stage_b", "checkpoints", 0, "grid", 0, "seconds"], -1.0), "seconds must be", True),
    ("resume counts: a resumed first checkpoint with zero loaded units",
     lambda p: _set(p, ["stage_b", "checkpoints", 0, "resumed_from_unit_log"], True), "marked as resumed", True),
    ("data description: unrecognized tokenizer encoding",
     lambda p: _set(p, ["data", "encoding"], "cl100k_base"), "data.encoding", False),  # canonical identity
    ("data description: NUL in the data directory",
     lambda p: _set(p, ["data", "directory"], p["data"]["directory"] + "\x00"), "NUL byte", True),
    ("scalar types: integer zero in a boolean scaling flag",
     lambda p: _update(_grid_row(p, lambda r: r["activation_aware_scaling"] is False), activation_aware_scaling=0),
     "method fields differ", True),
    ("scalar types: boolean true in the seed of the first Muon row",
     lambda p: _update(_stage_a_row(p, b.MUON), seed=True), "wrong seed", True),
]


# The tokenizer identity is pinned for canonical results only; every other fixture applies to any run.
NONCANONICAL_B1_FIXTURES = [fixture[:3] for fixture in ASTRA_B1_FIXTURES if fixture[3]]


def test_the_audit_table_has_eighteen_fixtures():
    assert len(ASTRA_B1_FIXTURES) == 18 and len({name for name, *_ in ASTRA_B1_FIXTURES}) == 18
    assert [name for name, *_, noncanonical in ASTRA_B1_FIXTURES if not noncanonical] == [
        "data description: unrecognized tokenizer encoding"]


@pytest.mark.parametrize("name, edit, match", NONCANONICAL_B1_FIXTURES, ids=[f[0] for f in NONCANONICAL_B1_FIXTURES])
def test_validator_rejects_the_audit_fixture_on_a_fresh_noncanonical_run(quick, name, edit, match):
    if "static-scale" in name and not FP4_GEMM:
        pytest.skip("static W4A4 rows need native NVFP4 scaled_mm (sm_100+)")
    assert quick["meta"]["unit_log"]["units_loaded"] == 0 and quick["stage_b"]["calibration"]["starts"][0] != 0
    assert quick["stage_a"]["seeds"][0]["seed"] == 1
    for mode in (None, False):  # the audited validator accepted these in both modes
        _rejects(quick, edit, match, canonical=mode)


# ------------------------------------------------------------------ committed canonical artifact
CERTIFIED = {"mode": "canonical", "certified_canonical": True, "postprocessed": True}
VOLATILE = ("postprocess_git_sha", "postprocess_git_dirty", "postprocess_code_sha256", "postprocess_timestamp")


@pytest.fixture(scope="module")
def canonical():
    if not RESULT.exists():
        pytest.skip(f"{RESULT.name} has not been produced yet")
    payload = json.loads(RESULT.read_text())
    if payload["schema_version"] == b.SOURCE_SCHEMA_VERSION:
        pytest.skip(f"{RESULT.name} is the {b.SOURCE_SCHEMA_VERSION} source: run --postprocess-from-git first")
    return payload


def _source_blob():
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{SOURCE_COMMIT}:{b.RESULT_REPO_PATH}"],
                          capture_output=True, check=True).stdout


def test_committed_canonical_artifact(canonical):
    payload = canonical
    assert b.validate_results(payload, canonical=True) == b.validate_results(payload) == CERTIFIED
    assert payload["schema_version"] == "cycle5_c5_02.v2"
    assert payload["canonical"] and not payload["quick"] and payload["protocol"] == b.CANONICAL
    assert payload["meta"]["git_dirty"] is False and payload["meta"]["environment"]["capability"] >= [12, 0]
    assert payload["stage_a"]["split_view"] == payload["stage_a"]["splits_opened_at_end"] == ["train", "val"]
    stage_b = payload["stage_b"]
    assert [(c["seed"], c["arm"]) for c in stage_b["checkpoints"]] == [(s, a) for s in range(1, 6) for a in b.ARMS]
    for c in stage_b["checkpoints"]:
        assert (c["baseline_val"]["tokens"], c["baseline_test"]["tokens"]) == (b.VAL_TOKENS, b.TEST_TOKENS)
        assert (c["baseline_val"]["windows"], c["baseline_test"]["windows"]) == (3910, 3906)
        assert c["baseline_test"]["span"] == [0, 1_999_873] and c["baseline_test"]["split_tokens"] == 2_000_081
        assert len(c["grid"]) == (37 if c["arm"] in b.AWQ_ARMS else 25)
    assert stage_b["calibration"]["starts"][0] == 346112 and len(stage_b["calibration"]["starts"]) == 128
    assert sum(len(kinds) for c in stage_b["checkpoints"] for kinds in c["butterfly_fits"].values()) == 510
    hw = payload["analysis"]["hardware"]
    assert hw["fp4_gemms_per_forward"] == [60] and hw["fp4_gemms_per_forward_emulated"] == [0]
    assert hw["fp4_gemms_per_forward_bf16_baseline"] == [0]
    assert hw["decode_exact_rows"] == hw["decode_checked_rows"] == 5 * (2 * 6 + 2 * 4)
    assert hw["decode_checked_matrices"] == 60 * hw["decode_checked_rows"]
    parity = payload["analysis"]["hardware_vs_emulation"]
    assert parity["passed"] and len(parity["rows"]) == 20 and parity["max_abs_nll_difference"] <= 1e-3
    iso = payload["analysis"]["iso_loss"]
    assert [p["seed"] for p in iso["pairs"]] == [1, 2, 3, 4, 5]
    for pair, unit in zip(iso["pairs"], payload["stage_a"]["seeds"]):
        assert pair["stage_a_abs_val_gap"] == unit["selection"]["abs_val_gap"]
        assert pair["abs_val_gap"] == pytest.approx(unit["selection"]["abs_val_gap"], abs=1e-9)
        assert pair["val_gap_within_tolerance"] == unit["selection"]["matched"]
    for contrast in payload["analysis"]["contrasts"]:
        assert len(contrast["primary"]) == len(contrast["own_best"]) == 6
        for entry in contrast["primary"] + contrast["own_best"]:
            for field in b.ENDPOINTS:
                block = entry["test"][f"{field}_diff"]
                ref = scipy_stats.ttest_1samp(block["diffs"], 0.0)
                if block["sd"] > 0:
                    assert block["p_two_sided"] == pytest.approx(ref.pvalue, rel=1e-9)
                assert block["p_sign_flip"] == b.sign_flip_p(block["diffs"]) >= 0.0625


def test_committed_artifact_is_a_postprocess_that_keeps_the_v1_measurements_bit_for_bit(canonical):
    """The raw measurements are those of da148c2 / 0355d16; only analysis, derived and caveats are new."""
    blob = _source_blob()
    assert hashlib.sha256(blob).hexdigest() == SOURCE_SHA256 == b.CANONICAL_SOURCE_ARTIFACTS[SOURCE_COMMIT]
    source = json.loads(blob)
    assert source["schema_version"] == "cycle5_c5_02.v1" and "postprocess" not in source
    assert b.RAW_MEASUREMENT_KEYS == ("quick", "canonical", "meta", "run_config_hash", "preset", "config", "protocol",
                                      "data", "stage_a", "stage_b")
    assert set(canonical) == {"schema_version", "task", *b.RAW_MEASUREMENT_KEYS, "postprocess", "analysis", "derived",
                              "caveats"}
    dumps = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)  # noqa: E731
    for key in b.RAW_MEASUREMENT_KEYS:
        assert b._same(canonical[key], source[key]) and dumps(canonical[key]) == dumps(source[key]), key
    post = canonical["postprocess"]
    assert tuple(post) == b.POSTPROCESS_KEYS == (
        "source_schema_version", "source_result_git_sha", "source_artifact_sha256",
        "source_raw_measurements_sha256", "postprocess_git_sha", "postprocess_git_dirty", "postprocess_code_sha256",
        "postprocess_code_modules", "postprocess_command", "postprocess_timestamp", "retained_evidence_sha256",
        "reason")
    assert post["source_schema_version"] == "cycle5_c5_02.v1" and post["source_result_git_sha"] == SOURCE_COMMIT
    assert post["source_artifact_sha256"] == SOURCE_SHA256 and post["postprocess_git_dirty"] is False
    raw = {key: source[key] for key in b.RAW_MEASUREMENT_KEYS}
    assert post["source_raw_measurements_sha256"] == hashlib.sha256(dumps(raw).encode()).hexdigest() \
        == b.raw_measurements_sha256(canonical)
    assert post["retained_evidence_sha256"] == {
        "stopped_canonical_log": "d877d6862b98b8d140aeb2436f0085ce18173ccada7252e0d140910e64c0d046",
        "final_canonical_log": "cfde4b04bc416c04cb5d98c39f94ee59312b62c2f24bfe840acd35c065b843fe",
        "seed0_shakedown_json": "39f2d27e6661e89f7f3e7f4cef7bab7ed78441b73236e81bc92fb8c303421803",
        "floor_lr_probe_output": "adf722e3ab295959d11bb6a0889d8c8e376d2c65f5a540656caecfb5ba44a2a5"}
    assert post["postprocess_command"][1:] == [f"research/quant_ptq/{b.SCRIPT_NAME}", "--postprocess-from-git",
                                               SOURCE_COMMIT]
    assert post["postprocess_code_modules"] == list(b.code_modules())
    assert post["postprocess_code_sha256"] == b._code_sha256_at_commit(post["postprocess_git_sha"], b.code_modules())
    # The producer is the v1 code of 0355d16: its hashes stay v1 hashes, and its code is not this code.
    meta = canonical["meta"]
    assert meta["git_sha"] == "0355d16f258133b5fb7aec8e90fd223f781595e9" != post["postprocess_git_sha"]
    assert meta["code_sha256"] == "424055b21b6f894ef850151501425839e5731e9919ac7399138d75788f7dc85b" \
        != post["postprocess_code_sha256"]
    assert (meta["unit_config_hash"], canonical["run_config_hash"]) == ("37922272335f8cda", "519f50bec7c85095")
    assert "Nothing was retrained" in post["reason"]
    # The postprocess is a pure function of the source: the same payload apart from where and when it ran.
    again = b.postprocess_from_git(SOURCE_COMMIT, post["postprocess_command"])

    def strip(p):
        return {**p, "postprocess": {k: v for k, v in p["postprocess"].items() if k not in VOLATILE}}
    assert strip(again) == strip(canonical) and list(again) == list(canonical)
    head = b.h.git_info()
    if head["git_dirty"] is False and b.code_sha256() == b._code_sha256_at_commit(head["git_sha"], b.code_modules()):
        assert b.validate_results(again, canonical=True) == CERTIFIED  # a clean, committed tree certifies it again


def test_postprocess_refuses_a_source_that_is_not_a_committed_v1_result(monkeypatch):
    with pytest.raises(ValueError, match="not a commit object"):
        b.postprocess_from_git("0" * 40, ["python", b.SCRIPT_NAME])
    with pytest.raises(ValueError, match="cannot read"):  # the producer commit has no result file
        b.postprocess_from_git("0355d16f258133b5fb7aec8e90fd223f781595e9", ["python", b.SCRIPT_NAME])
    source = json.loads(_source_blob())
    for edit in (lambda p: _set(p, ["schema_version"], "cycle5_c5_02.v2"), lambda p: _set(p, ["task"], "other"),
                 lambda p: p.pop("stage_b")):
        monkeypatch.setattr(b, "_git_blob", lambda sha, path, edit=edit: json.dumps(_edited(source, edit)).encode())
        with pytest.raises(ValueError, match="is not a cycle5_c5_02.v1 result"):
            b.postprocess_from_git(SOURCE_COMMIT, ["python", b.SCRIPT_NAME])


def test_committed_canonical_artifact_rejects_postprocess_provenance_tampering(canonical):
    post = ["postprocess"]

    def rejects(edit, match):
        _rejects(canonical, edit, match, canonical=True)
    # Shape of the record.
    rejects(lambda p: _set(p, post, None), "unit_config_hash")  # a v1 measurement is not a fresh v2 run
    rejects(lambda p: _get(p, post).pop("reason"), "postprocess keys")
    rejects(lambda p: _set(p, [*post, "note"], "x"), "postprocess keys")
    rejects(lambda p: _set(p, [*post, "source_schema_version"], "cycle5_c5_02.v2"), "source_schema_version")
    rejects(lambda p: _set(p, [*post, "reason"], "cosmetic"), "postprocess.reason")
    rejects(lambda p: _set(p, [*post, "postprocess_git_dirty"], 0), "must be a bool")
    rejects(lambda p: _set(p, [*post, "postprocess_timestamp"], "today"), "invalid ISO timestamp")
    rejects(lambda p: _set(p, [*post, "postprocess_timestamp"], p["meta"]["timestamp_start"]),
            "before the end of the measuring run")
    rejects(lambda p: _set(p, [*post, "source_artifact_sha256"], "xyz"), "64-hex")
    rejects(lambda p: _set(p, [*post, "postprocess_git_sha"], "HEAD"), "40-hex")
    # Source artifact and raw measurements.
    rejects(lambda p: _set(p, [*post, "source_artifact_sha256"], "0" * 64), "committed bytes of the source artifact")
    rejects(lambda p: _set(p, [*post, "source_result_git_sha"], "0" * 40), "not a commit object")
    rejects(lambda p: _set(p, [*post, "source_result_git_sha"], p["meta"]["git_sha"]), "cannot read")
    rejects(lambda p: _set(p, [*post, "source_raw_measurements_sha256"], "0" * 64), "source_raw_measurements_sha256")
    rejects(lambda p: _set(p, ["meta", "loadavg_end", 0], p["meta"]["loadavg_end"][0] + 1.0),
            "meta differs from the source artifact")
    rejects(lambda p: _set(p, ["stage_b", "checkpoints", 0, "grid", 0, "quantize_seconds"], 0.5),
            "stage_b differs from the source artifact")
    rejects(lambda p: _set(p, ["stage_a", "seeds", 0, "rows", 0, "train_seconds"], 1.0),
            "stage_a differs from the source artifact")
    rejects(lambda p: _set(p, ["data", "directory"], "/data/real_text"), "data differs from the source artifact")

    def rewrite_a_leaf_and_every_summary(p):  # consistent inside the JSON, but not the committed measurement
        c = p["stage_b"]["checkpoints"][0]
        row, base = next(r for r in c["grid"] if r["method"] == "rtn"), c["baseline_test"]
        row["test_nll"] += 0.01
        row["test_ppl"] = math.exp(row["test_nll"])
        row["test_delta_nll"] = row["test_nll"] - base["nll"]
        row["test_relative_delta_ppl_pct"] = 100.0 * (row["test_ppl"] / base["ppl"] - 1.0)
        c["best"] = b.best_methods(c["grid"], p["protocol"]["base_methods"])
        _refresh(p)
        p["postprocess"]["source_raw_measurements_sha256"] = b.raw_measurements_sha256(p)
    rejects(rewrite_a_leaf_and_every_summary, "stage_b differs from the source artifact")
    # Producer code and postprocessing code.
    rejects(lambda p: _set(p, ["meta", "code_modules"], p["meta"]["code_modules"][:-1]), "differs from the source")
    rejects(lambda p: _set(p, [*post, "postprocess_code_modules"], p["postprocess"]["postprocess_code_modules"][:-1]),
            "postprocess_code_modules differ")
    rejects(lambda p: _set(p, [*post, "postprocess_code_sha256"], "0" * 64), "git blobs at postprocess_git_sha")
    rejects(lambda p: _set(p, [*post, "postprocess_git_sha"], "0" * 40), "not a commit object")
    rejects(lambda p: _set(p, [*post, "postprocess_git_sha"], p["meta"]["git_sha"]),
            "git blobs at postprocess_git_sha")  # the v1 producer commit holds other code
    # Astra re-check Blocking 1a: claiming v1 producer commit as both postprocessor and blob source is rejected.
    rejects(lambda p: _update(_get(p, post), postprocess_git_sha=p["meta"]["git_sha"],
                              postprocess_code_sha256=p["meta"]["code_sha256"]),
            "must commit cycle5_c5_02.v2 with --postprocess-from-git")
    # Astra re-check Blocking 1b: stripping postprocess and recomputing v2 hashes on v1 producer fails.
    def v1_as_fresh_v2(p):
        p["postprocess"] = None
        hashes = {k: v["sha256"] for k, v in p["data"]["splits"].items()}
        p["meta"]["unit_config_hash"] = b.unit_config_hash(
            p["protocol"], p["config"], p["meta"]["code_sha256"], p["meta"]["environment"], hashes, b.SCHEMA_VERSION)
        p["run_config_hash"] = b.run_config_hash(
            False, p["protocol"], p["config"], p["meta"]["code_sha256"], p["meta"]["git_sha"],
            p["meta"]["environment"], hashes, b.SCHEMA_VERSION)
    rejects(v1_as_fresh_v2, "declares schema 'cycle5_c5_02.v1'")
    # Astra v3 signoff Blocking 1: earlier v2 commits (9473f90, a7c4be2) cannot be credited for current summaries.
    for old_v2_sha, old_v2_code in (
            ("9473f90b3eb340cb162cb5d6f96bc6ec3e4dff32",
             "12ded47a9e5dd26c0e893285aad190f3f40e9ecf36d4e2bc1a17438a1c98ba33"),
            ("a7c4be23646cee1c3e413b4bd4297984b6e95c9e",
             "3cb1c6b8e6fb117186c9532c75b96e96b14a16d020ba4e244bc9c9e3806d40b4")):
        rejects(lambda p, s=old_v2_sha, c=old_v2_code: _update(
                    _get(p, post), postprocess_git_sha=s, postprocess_code_sha256=c),
                "summary code differs from the validating checkout")

        def old_v2_as_fresh(p, s=old_v2_sha, c=old_v2_code):
            p["postprocess"] = None
            hashes = {k: v["sha256"] for k, v in p["data"]["splits"].items()}
            p["meta"].update(git_sha=s, code_sha256=c)
            p["meta"]["unit_config_hash"] = b.unit_config_hash(
                p["protocol"], p["config"], c, p["meta"]["environment"], hashes, b.SCHEMA_VERSION)
            p["run_config_hash"] = b.run_config_hash(
                False, p["protocol"], p["config"], c, s, p["meta"]["environment"], hashes, b.SCHEMA_VERSION)
        rejects(old_v2_as_fresh, "summary code differs from the validating checkout")
    # Exact hashes from Astra v3 signoff for 9473f90 on the unmodified v1 producer metadata.
    def astra_exact_9473f90_fresh_repro(p):
        p["postprocess"] = None
        p["meta"].update(
            git_sha="9473f90b3eb340cb162cb5d6f96bc6ec3e4dff32",
            code_sha256="12ded47a9e5dd26c0e893285aad190f3f40e9ecf36d4e2bc1a17438a1c98ba33",
            unit_config_hash="346808d5e324305a")
        p["run_config_hash"] = "cc4be683cba148ee"
    rejects(astra_exact_9473f90_fresh_repro, "summary code differs from the validating checkout")
    # Astra re-check Blocking 3a/3b on canonical artifact: 1-ULP increase fails, rounded FP32 tie with 1 improved passes.
    bf_key = "nvfp4_g16/adaptive_butterfly_b16_gptq"
    bf_path = ["stage_b", "checkpoints", 0, "butterfly_fits", bf_key, "qkv"]

    def canon_one_ulp_increase(p):
        rec = _get(p, bf_path)
        start = float(np.float32(0.001))
        fitted = float(np.nextafter(np.float32(start), np.float32(1)))
        rec.update(weight_sq_error_hadamard_start=start, weight_sq_error_fitted=fitted,
                   relative_error_reduction=1.0 - fitted / start, layers_improved=0)
    rejects(canon_one_ulp_increase, "0 <= fitted <= start")

    rejects(lambda p: _set(p, [*post, "postprocess_git_dirty"], True), "clean tree")
    # Command.
    command = canonical["postprocess"]["postprocess_command"]
    for bad, match in ((command[:2], "--postprocess-from-git with the source result commit"),
                       ([*command[:3], "0355d16f258133b5fb7aec8e90fd223f781595e9"], "source result commit"),
                       ([*command[:3], "da148c2"], "valid runnable invocation"),
                       ([*command, "--resume"], "valid runnable invocation"),
                       ([command[0], command[1], "--postprocess-f", SOURCE_COMMIT], "valid runnable invocation"),
                       ([*command, "--output", "/tmp/copy.json"], "must target the canonical output path"),
                       (["bash", *command[1:]], "Python invocation"),
                       (["not-a-python-interpreter", *command[1:]], "Python invocation"),
                       ([command[0], "other.py", *command[2:]], "not an invocation"),
                       ([command[0], "tests/test_bench_c5_02_isoloss_ptq_mitigation.py", *command[2:]],
                        "not an invocation"),
                       ("python", "Python invocation")):
        rejects(lambda p: _set(p, [*post, "postprocess_command"], bad), match)
    rejects(lambda p: _set(p, ["meta", "command"], command), "source artifact|canonical protocol")
    rejects(lambda p: _set(p, ["meta", "command"], ["not-a-python-interpreter", *p["meta"]["command"][1:]]),
            "Python invocation")
    rejects(lambda p: _set(p, ["meta", "command"],
                           [p["meta"]["command"][0], "tests/test_bench_c5_02_isoloss_ptq_mitigation.py"]),
            "canonical invocation")
    rejects(lambda p: _set(p, ["meta", "command"], [*p["meta"]["command"], "--checkpoint-d", "elsewhere"]),
            "valid runnable invocation")
    rejects(lambda p: _set(p, ["meta", "command"], [*p["meta"]["command"], "--data-d", "elsewhere"]),
            "valid runnable invocation")
    # Retained audit evidence.
    evidence = [*post, "retained_evidence_sha256"]
    rejects(lambda p: _set(p, [*evidence, "final_canonical_log"], "0" * 64), "pinned digests")
    rejects(lambda p: _set(p, [*evidence, "final_canonical_log"], "cfde4b04"), "64-hex digests")
    rejects(lambda p: _get(p, evidence).pop("seed0_shakedown_json"), "64-hex digests")
    rejects(lambda p: _set(p, [*evidence, "pilot_script"], "0" * 64), "64-hex digests")


def test_canonical_postprocess_must_read_the_pinned_source(canonical, monkeypatch):
    monkeypatch.setattr(b, "CANONICAL_SOURCE_ARTIFACTS", {SOURCE_COMMIT: "0" * 64})
    with pytest.raises(ValueError, match="pinned committed C5-02 result"):
        b.validate_results(canonical, canonical=True)


def test_postprocess_cli_refuses_to_write_noncanonical_source_to_reserved_output(quick, tmp_path, monkeypatch):
    """Astra re-check Blocking 2: --postprocess-from-git into reserved output refuses noncanonical payloads."""
    res_dir = tmp_path / "results"
    res_dir.mkdir()
    reserved_out = res_dir / b.CANONICAL_OUTPUT.name
    reserved_out.write_text('{"sentinel": true}\n')
    monkeypatch.setattr(b, "CANONICAL_OUTPUT", reserved_out)
    monkeypatch.setattr(b.h, "git_info", lambda: {"git_sha": "a" * 40, "git_dirty": False})
    bad_payload = copy.deepcopy(quick)
    bad_payload["postprocess"] = {
        "source_schema_version": "cycle5_c5_02.v1", "source_result_git_sha": SOURCE_COMMIT,
        "source_artifact_sha256": "0" * 64, "source_raw_measurements_sha256": "0" * 64,
        "postprocess_git_sha": "a" * 40, "postprocess_git_dirty": False,
        "postprocess_code_sha256": "0" * 64, "postprocess_code_modules": list(b.code_modules()),
        "postprocess_command": ["/venv/bin/python", f"research/quant_ptq/{b.SCRIPT_NAME}",
                                "--postprocess-from-git", SOURCE_COMMIT],
        "postprocess_timestamp": "2026-11-27T06:00:00-0700",
        "retained_evidence_sha256": dict(b.RETAINED_EVIDENCE_SHA256), "reason": b.POSTPROCESS_REASON}
    monkeypatch.setattr(b, "postprocess_from_git", lambda commit, cmd: copy.deepcopy(bad_payload))
    with pytest.raises(ValueError, match="not labeled canonical"):
        b.main(["--postprocess-from-git", SOURCE_COMMIT, "--silent"])
    assert reserved_out.read_text() == '{"sentinel": true}\n'

    # Even if validate_results is bypassed or relaxed, the postprocess gate refuses uncertified reserved writes.
    monkeypatch.setattr(b, "validate_results", lambda p, canonical=None: {
        "mode": "noncanonical", "certified_canonical": False, "postprocessed": True})
    with pytest.raises(ValueError, match="Refusing to write noncanonical postprocessed result"):
        b.main(["--postprocess-from-git", SOURCE_COMMIT, "--silent"])
    assert reserved_out.read_text() == '{"sentinel": true}\n'

    # Non-reserved explicit --output still permits a validated exploratory postprocess.
    free_out = tmp_path / "exploratory.json"
    orig_validate = b.validate_results
    monkeypatch.undo()
    monkeypatch.setattr(b, "postprocess_from_git", lambda commit, cmd: copy.deepcopy(bad_payload))
    monkeypatch.setattr(b, "validate_results", lambda p, canonical=None: {
        "mode": "noncanonical", "certified_canonical": False, "postprocessed": True})
    written = b.main(["--postprocess-from-git", SOURCE_COMMIT, "--output", str(free_out), "--silent"])
    assert written["canonical"] is False and free_out.is_file()
    assert orig_validate is not None


@pytest.mark.parametrize("name, edit, match, noncanonical", ASTRA_B1_FIXTURES, ids=[f[0] for f in ASTRA_B1_FIXTURES])
def test_canonical_validator_rejects_the_audit_fixture(canonical, name, edit, match, noncanonical):
    """Each fixture is rejected by the check of its own invariant, before the source comparison runs."""
    assert canonical["stage_b"]["calibration"]["starts"][0] == 346112
    assert canonical["meta"]["unit_log"]["units_loaded"] == 0
    assert _stage_a_row(canonical, b.MUON)["train_curve"][-1][1] > 0 and canonical["data"]["encoding"] == "gpt2"
    _rejects(canonical, edit, match, canonical=True)


def test_call_site_certification_of_the_committed_artifact(canonical):
    """H1: the three relaxations that the audit observed with an explicit noncanonical request are gone."""
    with pytest.raises(ValueError, match="labeled canonical"):
        b.validate_results(canonical, canonical=False)
    for edit in (lambda p: _set(p, ["meta", "git_dirty"], True),  # a dirty-tree record
                 lambda p: _set(p, ["meta", "command"], [p["meta"]["command"][0], "other.py"]),  # another entry point
                 lambda p: _set(p, ["data", "sha256_verified"], False)):  # unverified shards
        for mode in (None, True):
            _rejects(canonical, edit, "clean tree|canonical invocation|expected_sha256", canonical=mode)
        _rejects(canonical, edit, "labeled canonical", canonical=False)
    _rejects(canonical, lambda p: _set(p, ["canonical"], False), "not labeled canonical", canonical=True)
    _rejects(canonical, lambda p: _set(p, ["canonical"], False), "run_config_hash|source artifact", canonical=False)
    # Nested records and types.
    for edit, match in (
            (lambda p: _set(p, ["meta", "environment", "torch"], 2.14), "meta.environment invalid"),
            (lambda p: _set(p, ["meta", "environment", "capability"], [12, "0"]), "device fields"),
            (lambda p: _set(p, ["meta", "environment", "env"], []), "meta.environment invalid"),
            (lambda p: _set(p, ["meta", "gpu_telemetry", "after_stage_a"], None), "gpu_telemetry"),
            (lambda p: _set(p, ["meta", "gpu_telemetry", "start", "clocks.sm"], [277]), "gpu_telemetry"),
            (lambda p: _set(p, ["meta", "gpu_telemetry", "start"], []), "gpu_telemetry"),
            (lambda p: _set(p, ["stage_b", "checkpoints", 3, "hardware_w4a4", "note"], 0), "hardware_w4a4.note"),
            (lambda p: _set(p, ["stage_b", "checkpoints", 3, "hardware_w4a4", "note"], b.HARDWARE_SKIP_NOTE),
             "hardware_w4a4.note"),
            (lambda p: _set(p, ["stage_b", "checkpoints", 3, "hardware_w4a4", "available"], False),
             "needs hardware W4A4")):
        _rejects(canonical, edit, match, canonical=True)


def test_committed_canonical_artifact_rejects_tampering(canonical):
    payload = canonical
    _rejects(payload, lambda p: _set(p, ["meta", "git_dirty"], True), "clean tree", canonical=True)
    _rejects(payload, lambda p: _set(p, ["meta", "git_sha"], "0" * 40), "run_config_hash|not a commit", canonical=True)
    _rejects(payload, lambda p: _set(p, ["protocol", "adamw", "max_trials"], 9), "protocol keys or value types",
             canonical=True)
    _rejects(payload, lambda p: _set(p, ["protocol", "tolerance_nats"], 0.03), "canonical C5-02 protocol",
             canonical=True)
    _rejects(payload, lambda p: _set(p, ["data", "sha256_verified"], False), "pinned|expected_sha256", canonical=True)
    _rejects(payload, lambda p: _set(p, ["meta", "command"], ["python", "other.py"]), "canonical invocation",
             canonical=True)
    # Canonical data identity: FineWeb-Edu, GPT-2 BPE, pinned split lengths.
    _rejects(payload, lambda p: _set(p, ["data", "source", "config"], "sample-100BT"), "data.source", canonical=True)
    _rejects(payload, lambda p: _set(p, ["data", "source", "streaming"], 1), "data.source", canonical=True)
    _rejects(payload, lambda p: _set(p, ["data", "splits", "test", "tokens"], 2_000_082), "split token counts",
             canonical=True)
    # Val coverage, as for test: one window, a short span, a short split.
    for key, value in (("windows", 1), ("span", [0, 1]), ("split_tokens", 2), ("tokens", 1024)):
        _rejects(payload, lambda p: _set(p, ["stage_b", "checkpoints", 7, "baseline_val", key], value),
                 "does not cover the full val.bin|differs from its Stage A value", canonical=True)
    _rejects(payload, lambda p: _set(p, ["stage_a", "seeds", 2, "rows", 3, "val_tokens"], 1_999_872),
             "val.bin is not covered completely", canonical=True)
    # Training metadata of the canonical recipes.
    muon = ["stage_a", "seeds", 0, "rows", 0]
    assert _get(payload, [*muon, "train_curve"])[-1][0] == _get(payload, [*muon, "steps"]) == 320
    _rejects(payload, lambda p: _set(p, [*muon, "last_train_loss"], _get(p, [*muon, "last_train_loss"]) + 0.25),
             "last_train_loss", canonical=True)
    _rejects(payload, lambda p: _set(p, [*muon, "orthogonalizer"], "polar_express eager BF16"), "one orthogonalizer",
             canonical=True)

    def cpu_stand_in_everywhere(p):
        for unit in p["stage_a"]["seeds"]:
            for row in unit["rows"][:3]:
                row["orthogonalizer"] = "polar_express eager BF16 (CPU stand-in: the SS3_54 kernel needs CUDA)"
    _rejects(payload, cpu_stand_in_everywhere, "SS3_54 Triton kernel when canonical", canonical=True)

    def one_field_of_a_zero_alpha_copy(field, value):  # the copy must equal its twin in every measured field
        def edit(p):
            row = _grid_row(p, lambda r: r.get("awq_alpha") == 0 and field in r)
            row[field] = value(row[field])
        return edit
    for field, value in (("seconds", lambda v: v + 1.0), ("quantize_seconds", lambda v: v + 1.0),
                         ("gptq_damping_max", lambda v: 10 * v), ("hw_weights_checked_matrices", lambda v: 59),
                         ("fp4_gemms_per_forward", lambda v: 59),
                         ("static_global_scale_range", lambda v: [v[0], 2 * v[1]]),
                         ("geometry", lambda v: {**v, "weight_rel_error": [2 * x for x in v["weight_rel_error"]]})):
        _rejects(payload, one_field_of_a_zero_alpha_copy(field, value),
                 "alpha = 0 must be the unscaled row|decode to the fake-quant weights|FP4 GEMMs per forward|"
                 "weight_rel_error_mean", canonical=True)

    def break_parity(p):
        c = p["stage_b"]["checkpoints"][0]
        row = next(r for r in c["grid"] if r["mode"] == b.EMULATED_MODE)
        for split in b.SPLITS:
            base = c[f"baseline_{split}"]
            row[f"{split}_nll"] += 0.01
            row[f"{split}_ppl"] = math.exp(row[f"{split}_nll"])
            row[f"{split}_delta_nll"] = row[f"{split}_nll"] - base["nll"]
            row[f"{split}_relative_delta_ppl_pct"] = 100.0 * (row[f"{split}_ppl"] / base["ppl"] - 1.0)
        _refresh(p)
    _rejects(payload, break_parity, "parity threshold", canonical=True)


def test_hardware_and_decode_counts_report_unique_executions_next_to_the_totals(canonical):
    """H2: 20 of 200 hardware rows and 600 of 6,000 matrix comparisons are inherited from alpha = 0 twins."""
    expected = {"w4a4_rows": 200, "w4a4_unique_profiled_rows": 180, "w4a4_inherited_zero_alpha_rows": 20,
                "decode_checked_rows": 100, "decode_unique_checked_rows": 90, "decode_inherited_zero_alpha_rows": 10,
                "decode_checked_matrices": 6000, "decode_unique_checked_matrices": 5400,
                "decode_inherited_zero_alpha_matrices": 600, "decode_exact_rows": 100}
    for section in ("analysis", "derived"):
        assert {k: canonical[section]["hardware"][k] for k in expected} == expected
    rows = [r for c in canonical["stage_b"]["checkpoints"] for r in c["grid"]]
    copies = [r for r in rows if r.get("awq_alpha") == 0]
    assert len(copies) == 58 and all(b.inherited(r) for r in copies)  # 58 copied rows in the 620-row grid
    assert sum(r["mode"] in b.HW_MODES for r in copies) == 20
    assert sum(r["hw_weights_checked_matrices"] for r in copies if "hw_weights_checked_matrices" in r) == 600
    assert not b.inherited({"awq_alpha": 0.25}) and not b.inherited({"method": "gptq"})
    assert "20 of the 200 hardware W4A4 rows and 600 of the 6000 matrix decode comparisons are inherited" \
        in canonical["caveats"][-1]
    for key in ("w4a4_unique_profiled_rows", "decode_inherited_zero_alpha_matrices", "decode_unique_checked_rows"):
        _rejects(canonical, lambda p: _set(p, ["analysis", "hardware", key], p["analysis"]["hardware"][key] + 1),
                 "analysis", canonical=True)
        _rejects(canonical, lambda p: _set(p, ["derived", "hardware", key], p["derived"]["hardware"][key] + 1),
                 "derived", canonical=True)
        _rejects(canonical, lambda p: _get(p, ["analysis", "hardware"]).pop(key), "analysis", canonical=True)

    def shift_one_row_between_unique_and_inherited(p):  # consistent in both sections: only the recount sees it
        for section in ("analysis", "derived"):
            p[section]["hardware"]["w4a4_unique_profiled_rows"] += 1
            p[section]["hardware"]["w4a4_inherited_zero_alpha_rows"] -= 1
    recount = _edited(canonical, shift_one_row_between_unique_and_inherited)
    with pytest.raises(ValueError, match="recount of unique and inherited rows"):
        b._validate_counts(recount)


# ------------------------------------------------------------------ token exposure and caveats (B2, B3, C1-C3)
def _exposure_payload(steps):
    """A minimal payload for ``token_exposure``: (selected, trained) AdamW steps per seed, canonical sizes."""
    rows = lambda selected, trained: [{"arm": b.MUON, "steps": 320},  # noqa: E731
                                      {"arm": b.ADAMW, "steps": selected, "trained_steps": trained}]
    return {"preset": "50m", "config": {"micro_batch": 16, "grad_accum": 4, "sequence_length": 512},
            "data": {"splits": {"train": {"tokens": 26_004_518}}},
            "stage_a": {"seeds": [{"seed": i + 1, "rows": rows(*pair)} for i, pair in enumerate(steps)]}}


def test_token_exposure_statement_is_generated_from_the_selected_and_the_terminal_steps(canonical):
    exposure = b.token_exposure(canonical)
    assert [(e["seed"], e["adamw_selected_steps"], e["adamw_trained_steps"]) for e in exposure] == [
        (1, 720, 720), (2, 944, 944), (3, 880, 896), (4, 784, 800), (5, 672, 672)]
    rounded = lambda key: [round(e[key], 3) for e in exposure]  # noqa: E731
    assert rounded("muon_token_equivalents") == [0.403] * 5
    assert rounded("adamw_selected_token_equivalents") == [0.907, 1.19, 1.109, 0.988, 0.847]
    assert rounded("adamw_trained_token_equivalents") == [0.907, 1.19, 1.129, 1.008, 0.847]
    assert exposure[1]["adamw_selected_token_equivalents"] == 944 * 32768 / 26_004_518
    for row, e, unit in zip(canonical["derived"]["stage_a"], exposure, canonical["stage_a"]["seeds"]):
        assert {k: row[k] for k in ("adamw_trained_steps", "adamw_selected_token_equivalents",
                                    "adamw_trained_token_equivalents")} == {
            "adamw_trained_steps": e["adamw_trained_steps"],
            "adamw_selected_token_equivalents": e["adamw_selected_token_equivalents"],
            "adamw_trained_token_equivalents": e["adamw_trained_token_equivalents"]}
        assert row["adamw_selected_token_equivalents"] == pytest.approx(unit["rows"][3]["train_epochs"], rel=1e-12)
        assert row["t_adamw"] == e["adamw_selected_steps"] <= row["adamw_trained_steps"]
    text = b.exposure_caveat(canonical)
    assert canonical["caveats"][6] == text  # "Caveat 7" of the audit
    for claim in ("BatchStream draws with replacement", "64 independent 513-token slices", "no sequential cursor",
                  "26,004,518 tokens of train.bin", "not in epochs of distinct tokens",
                  "draw 0.403 token equivalents (320 steps)",
                  "selected AdamW checkpoints are at steps 672-944, 0.847-1.190 token equivalents",
                  "the selected checkpoints of seeds 2 and 3 are above one token equivalent (1.190 and 1.109)",
                  "actually trained 672-944 steps, 0.847-1.190 token equivalents",
                  "seed 3 trained 896 steps = 1.129 token equivalents before the selection of step 880 = 1.109",
                  "seed 4 trained 800 steps = 1.008 token equivalents before the selection of step 784 = 0.988",
                  "A compute comparison must count every step actually run"):
        assert claim in text, claim
    assert not any("less than one epoch" in note for note in canonical["caveats"])
    stale = "One 50m model family, short training (less than one epoch of a 26M-token set), one calibration draw."
    _rejects(canonical, lambda p: _set(p, ["caveats", 6], stale), "caveats", canonical=True)
    _rejects(canonical, lambda p: _set(p, ["derived", "stage_a", 3, "adamw_trained_steps"], 784), "derived",
             canonical=True)
    _rejects(canonical, lambda p: _get(p, ["derived", "stage_a", 3]).pop("adamw_trained_token_equivalents"),
             "derived", canonical=True)


def test_token_exposure_statement_follows_other_step_counts():
    below = b.exposure_caveat(_exposure_payload([(480, 480), (640, 640)]))
    assert "no selected checkpoint is above one token equivalent" in below
    assert "(every seed selected its last measured step)" in below
    assert "steps 480-640, 0.605-0.806 token equivalents" in below and "seed 1 trained" not in below
    one = b.exposure_caveat(_exposure_payload([(800, 816), (640, 640)]))
    assert "the selected checkpoint of seed 1 is above one token equivalent (1.008)" in one
    assert "seed 1 trained 816 steps = 1.028 token equivalents before the selection of step 800 = 1.008" in one
    assert "actually trained 640-816 steps, 0.806-1.028 token equivalents" in one
    same = b.exposure_caveat(_exposure_payload([(720, 720), (720, 720)]))  # one value is not written as a range
    assert "steps 720, 0.907 token equivalents" in same and "trained 720 steps, 0.907 token equivalents" in same
    past_one = b.token_exposure(_exposure_payload([(784, 800)]))[0]  # trained past one equivalent, selected below
    assert past_one["adamw_selected_token_equivalents"] < 1.0 < past_one["adamw_trained_token_equivalents"]


def test_quick_caveats_carry_the_dynamic_exposure_statement(quick):
    text = b.exposure_caveat(quick)
    config = h.PRESETS["tiny"]
    slices = config.micro_batch * config.grad_accum
    assert text in quick["caveats"] and "One tiny model family" in text
    assert f"{slices} independent {config.sequence_length + 1}-token slices" in text
    for row, unit in zip(quick["derived"]["stage_a"], quick["stage_a"]["seeds"]):
        adamw = unit["rows"][3]
        assert row["adamw_trained_steps"] == adamw["trained_steps"] >= row["t_adamw"]
        assert row["adamw_selected_token_equivalents"] == adamw["steps"] * config.step_tokens / 200_000
        assert row["adamw_trained_token_equivalents"] == adamw["trained_steps"] * config.step_tokens / 200_000
    changed = copy.deepcopy(quick)  # the statement follows the terminal step: one more measured interval
    changed["stage_a"]["seeds"][0]["rows"][3]["trained_steps"] += 8
    assert b.exposure_caveat(changed) != text and b.derive(changed)["stage_a"][0]["adamw_trained_steps"] \
        == quick["derived"]["stage_a"][0]["adamw_trained_steps"] + 8


def test_caveats_state_the_split_access_the_pilot_chronology_and_the_limits_of_the_claims(canonical):
    notes = canonical["caveats"]
    assert notes == b.caveats(canonical) and len(notes) == 9
    text = " ".join(notes)
    for claim in (
            # B3: split access and pilot chronology
            "the selected step's weights (preceding the stopping step on seeds 3 and 4) are evaluated without retraining",
            "test.bin is held out from the canonical selection rules",
            "It is not a previously unseen study-level holdout",
            "timing_seed0.log and timing_seed0b.log", "SHA-256 39f2d27e...",
            "scored the same test.bin before the initial commit 05ea4ae and the amendment 0355d16",
            "seed 1 had matched at step 640", "still 0.0925 nats above it at step 1280",
            "That attempt has no Stage B scores",
            "startup verification read test.bin (hashes and document boundaries) and scored no model",
            "every one stopped below step 1280",
            # C1
            "sensitivity check at different losses, not an equal-loss proof", "val NLL 5.235-5.414",
            "primary AdamW test delta-NLL means at LR 0.001 are 1.38-4.02x those of the C4-01b AdamW recipe at LR 0.0015",
            "NVFP4 W4A16 2.60x, dynamic W4A4 1.42x, static W4A4 1.38x, MXFP4 1.97x, INT2 g16 3.54x, INT2 g32 4.02x",
            "Validation stopping, the additional token exposure, the floor-LR phase and the fresh restart are coupled",
            "cannot isolate which of them causes the mean BF16 test gap of +0.0037 nats",
            "supports no optimizer-wide PTQ ratio claim",
            # C2
            "The QAT cooldown targets identity-coordinate nvfp4_g16 round-to-nearest only",
            "The weight-only selection is a proxy", "can miss the alpha that is best under hardware W4A4",
            "No explicit spectral penalty was tested",
            # C3
            "share one corpus, one calibration draw and one set of test documents",
            "assume normal paired differences (df = 4)", "over the 6 cells of one family of one contrast",
            "over the 48 cells of that family across the 8 contrasts", "cannot be below 0.0625",
            "restricted to the 50m preset in this training regime on NVIDIA GeForce RTX 5080 (sm_120)"):
        assert claim in text, claim
    for stale in ("before test.bin was opened", "test.bin is the independent confirmation", "at equal val NLL",
                  "less than one epoch", "stopped weights are evaluated", "about three times"):
        assert stale not in text, stale
    source = Path(b.__file__).read_text()
    for stale in ("is not opened by the process until", "test.bin had not been opened",
                  "does not change the PTQ sensitivity of AdamW at equal loss", "test.bin opened only afterwards",
                  "test.bin is not opened before this loop ends", "checkpoints_present"):
        assert stale not in source, stale
