"""Checks for the Cycle 5 C5-02 per-seed iso-loss PTQ and Muon-mitigation benchmark.

Search, selection, statistics, geometry, guard and validator tests are pure Python or tiny tensors.
The pipeline tests train the ``tiny`` preset on synthetic shards; on an sm_100+ GPU they include
real hardware W4A4 rows.
"""

import copy
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

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
    assert b.checkpoints_present(unit, tmp_path / "ckpt")
    for arm, entry in unit["checkpoints"].items():
        assert entry["file"] == b.checkpoint_name(7, arm)
        model = b.load_checkpoint(config, torch.device("cpu"), tmp_path / "ckpt" / entry["file"], entry["sha256"])
        assert not model.training
    with pytest.raises(RuntimeError, match="differs from the Stage A record"):
        b.load_checkpoint(config, torch.device("cpu"), tmp_path / "ckpt" / b.checkpoint_name(7, b.MUON), "0" * 64)
    (tmp_path / "ckpt" / b.checkpoint_name(7, b.ADAMW)).unlink()
    assert not b.checkpoints_present(unit, tmp_path / "ckpt")
    with pytest.raises(RuntimeError, match="exactly train and val"):
        b.stage_a_seed(config, {**view, "test": view["val"]}, torch.device("cpu"), protocol, 7, tmp_path / "c2",
                       lambda *a: None)
    with pytest.raises(FileNotFoundError):
        shards.open("test")


def test_shards_check_pinned_hashes(tmp_path):
    data_dir = _fake_data(tmp_path)
    with pytest.raises(ValueError, match="differ from the pinned hashes"):
        b.Shards(data_dir, verify=False, expected=q.EXPECTED_SHA256)
    shards = b.Shards(data_dir, verify=False, expected={"train": "fake-train", "val": "fake-val", "test": "fake-test"})
    assert shards.opened == [] and set(shards.summary()["splits"]) == {"train", "val", "test"}
    shards.open("val")
    shards.open("train", "val")
    assert shards.opened == ["val", "train"]


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
    b.validate_results(quick)
    assert quick["schema_version"] == "cycle5_c5_02.v1" and quick["quick"] and not quick["canonical"]
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
    b.validate_results(again)


def _as_canonical(payload, monkeypatch):
    """Dress a --quick payload as a canonical one, so that every canonical branch of the validator runs."""
    p = copy.deepcopy(payload)
    tokens = p["stage_b"]["checkpoints"][0]["baseline_val"]["tokens"]
    hashes = {k: v["sha256"] for k, v in p["data"]["splits"].items()}
    monkeypatch.setattr(b, "CANONICAL", copy.deepcopy(p["protocol"]))
    monkeypatch.setattr(b, "VAL_TOKENS", tokens)
    monkeypatch.setattr(b, "TEST_TOKENS", tokens)
    monkeypatch.setattr(b.q, "EXPECTED_SHA256", hashes)
    monkeypatch.setattr(b, "_code_sha256_at_commit", lambda sha, modules: p["meta"]["code_sha256"])
    p["quick"], p["canonical"] = False, True
    p["data"].update(sha256_verified=True, expected_sha256=dict(hashes))
    p["meta"].update(git_sha="a" * 40, git_dirty=False,
                     command=["/venv/bin/python", f"research/quant_ptq/{b.SCRIPT_NAME}"])
    p["run_config_hash"] = b.run_config_hash(False, p["protocol"], p["config"], p["meta"]["code_sha256"], "a" * 40,
                                             p["meta"]["environment"], hashes)
    return p


@pytest.mark.skipif(not FP4_GEMM, reason="a canonical result needs hardware W4A4 rows (sm_100+)")
def test_canonical_branches_of_the_validator_run_on_a_dressed_quick_payload(quick, monkeypatch):
    if torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("a canonical result needs an sm_120+ device")
    p = _as_canonical(quick, monkeypatch)
    b.validate_results(p, canonical=True)
    b.validate_results(p)  # the payload's own flag selects the canonical checks
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
    _rejects(p, lambda x: _set(x, ["data", "sha256_verified"], False), "pinned", canonical=True)
    _rejects(p, lambda x: _set(x, ["meta", "gpu_telemetry", "end"], None), "gpu_telemetry", canonical=True)
    _rejects(p, lambda x: _set(x, ["stage_b", "checkpoints", 0, "baseline_test", "tokens"], 512),
             "token count differs|full val.bin and test.bin", canonical=True)
    monkeypatch.setattr(b, "VAL_TOKENS", 2_001_920)
    _rejects(p, lambda x: None, "val.bin is not covered completely", canonical=True)


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


def _rejects(payload, edit, match, canonical=None):
    broken = copy.deepcopy(payload)
    edit(broken)
    with pytest.raises((ValueError, KeyError), match=match):
        b.validate_results(broken, canonical=canonical)


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
    _rejects(quick, lambda p: _set(p, ["stage_a", "split_view"], ["train", "val", "test"]), "before test.bin is opened")
    _rejects(quick, lambda p: _set(p, ["stage_a", "splits_opened_at_end"], ["train", "val", "test"]),
             "before test.bin is opened")
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
    _rejects(quick, lambda p: _set(p, ["stage_b", "calibration", "starts", 0], -64), "Calibration windows")
    _rejects(quick, lambda p: _set(p, ["stage_b", "calibration", "seed"], 1), "Calibration windows")
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
    with pytest.raises(ValueError, match="Out of range float values"):
        nan = copy.deepcopy(quick)
        nan["stage_b"]["checkpoints"][0]["baseline_test"]["nll"] = float("nan")
        b.validate_results(nan)
    # A quick or CPU result is never a canonical one.
    _rejects(quick, lambda p: None, "cannot be --quick", canonical=True)

    def relabel(p):
        p["quick"], p["canonical"] = False, True
    _rejects(quick, relabel, "protocol differs from the canonical", canonical=None)


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
    b.validate_results(p)
    assert "not an attestation" in b.validate_results.__doc__ and "Threat model" in b.validate_results.__doc__


def test_analysis_needs_exactly_five_seeds(quick):
    p = copy.deepcopy(quick)
    p["protocol"]["seeds"] = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        b.analyze(p["stage_a"], p["stage_b"], p["protocol"])
    assert b.derive({**quick, "analysis": None})["analysis_available"] is False
    assert "No paired analysis" in b.caveats({**quick, "analysis": None})[0]


# ------------------------------------------------------------------ committed canonical artifact
def _canonical():
    if not RESULT.exists():
        pytest.skip(f"{RESULT.name} has not been produced yet")
    return json.loads(RESULT.read_text())


def test_committed_canonical_artifact():
    payload = _canonical()
    b.validate_results(payload, canonical=True)
    assert payload["canonical"] and not payload["quick"] and payload["protocol"] == b.CANONICAL
    assert payload["meta"]["git_dirty"] is False and payload["meta"]["environment"]["capability"] >= [12, 0]
    assert payload["stage_a"]["split_view"] == payload["stage_a"]["splits_opened_at_end"] == ["train", "val"]
    stage_b = payload["stage_b"]
    assert [(c["seed"], c["arm"]) for c in stage_b["checkpoints"]] == [(s, a) for s in range(1, 6) for a in b.ARMS]
    for c in stage_b["checkpoints"]:
        assert (c["baseline_val"]["tokens"], c["baseline_test"]["tokens"]) == (b.VAL_TOKENS, b.TEST_TOKENS)
        assert len(c["grid"]) == (37 if c["arm"] in b.AWQ_ARMS else 25)
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


def test_committed_canonical_artifact_rejects_tampering():
    payload = _canonical()
    _rejects(payload, lambda p: _set(p, ["meta", "git_dirty"], True), "clean tree", canonical=True)
    _rejects(payload, lambda p: _set(p, ["meta", "git_sha"], "0" * 40), "run_config_hash|not a commit", canonical=True)
    _rejects(payload, lambda p: _set(p, ["protocol", "adamw", "max_trials"], 9), "canonical C5-02 protocol",
             canonical=True)
    _rejects(payload, lambda p: _set(p, ["data", "sha256_verified"], False), "pinned", canonical=True)
    _rejects(payload, lambda p: _set(p, ["meta", "command"], ["python", "other.py"]), "canonical invocation",
             canonical=True)

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
