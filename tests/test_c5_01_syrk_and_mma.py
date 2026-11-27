"""Checks for Cycle 5 C5-01: native long-K chunked SYRK and the exact sm_120 MMA model."""

import copy
import json
import math
import random

import pytest
import torch

from research.polar_opt import bench_c4_03_wide_and_gram_drift as c3
from research.polar_opt import bench_c5_01_syrk_and_mma as c5
from research.polar_opt import blackwell_mma_model as mm
from research.polar_opt import kernels_blackwell as kbw
from research.polar_opt import polar_switch_blackwell as psb
from research.polar_opt import schatten_switch_blackwell as ssb

CUDA = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (12, 0)
needs_sm120 = pytest.mark.skipif(not CUDA, reason="needs the RTX 5080 (sm_120) Tensor Cores")


def _row(values: dict, dtype: str) -> list[float]:
    row = [0.0] * 16
    for lane, v in values.items():
        row[lane] = float(torch.tensor(v, dtype=torch.float64).to(mm.DTYPES[dtype]))
    return row


# ------------------------------------------------------------------ constants and wiring
def test_gate_constants_are_shared_by_the_three_modules():
    assert (kbw.SYRK_CHUNK_GATE, kbw.SYRK_CHUNK_SIZE, kbw.SYRK_CHUNK_MAX_BYTES) == (65536, 4096, 1 << 28)
    assert psb.SYRK_CHUNK_GATE == ssb.SYRK_CHUNK_GATE == kbw.SYRK_CHUNK_GATE
    assert ssb.SYRK_CHUNK_SIZE == kbw.SYRK_CHUNK_SIZE
    assert psb._syrk_chunked is ssb._syrk_chunked is kbw._syrk_chunked
    with c5.ungated():
        assert kbw.SYRK_CHUNK_GATE == psb.SYRK_CHUNK_GATE == ssb.SYRK_CHUNK_GATE == 1 << 60
    assert psb.SYRK_CHUNK_GATE == ssb.SYRK_CHUNK_GATE == kbw.SYRK_CHUNK_GATE == 65536


# ------------------------------------------------------------------ the MMA model (CPU)
def test_model_reproduces_the_audit_witnesses_and_the_c4_03_rule_does_not():
    fp16 = _row(mm.ASTRA_FP16_WITNESS, "fp16")
    assert mm.sm120_mma_accumulate(fp16, fp16, dtype="fp16") == float.fromhex(mm.ASTRA_FP16_HARDWARE)
    assert mm.c4_03_accumulate(fp16, fp16) == float.fromhex(mm.ASTRA_FP16_C4_03)
    bf16 = _row(mm.BF16_WITNESS, "bf16")
    assert mm.sm120_mma_accumulate(bf16, bf16, dtype="bf16") == float.fromhex(mm.ASTRA_BF16_HARDWARE)
    assert mm.c4_03_accumulate(bf16, bf16) == float.fromhex(mm.ASTRA_BF16_C4_03)
    rng = random.Random(0)
    for _ in range(50):  # the model has no lane order
        lanes = rng.sample(range(16), 4)
        row = [0.0] * 16
        for lane, v in zip(lanes, (x for x in fp16 if x)):
            row[lane] = v
        assert mm.sm120_mma_accumulate(row, row, dtype="fp16") == float.fromhex(mm.ASTRA_FP16_HARDWARE)


def test_model_reproduces_the_audit_exponent_table():
    for case in mm.ASTRA_EXPONENT_TABLE:
        a, b = [case["a"]] * case["count"], [case["b"]] * case["count"]
        assert mm.sm120_mma_accumulate(a, b, dtype="bf16") == case["hardware"], case["name"]


def test_model_rules_on_targeted_cases():
    one = mm.sm120_mma_accumulate
    assert one([2.0 ** -12] * 16, [2.0 ** -13] * 16, 1.0) == 1.0 + 4 * 2.0 ** -23  # 2 guard bits keep 2^-25
    assert one([2.0 ** -12] * 16, [2.0 ** -14] * 16, 1.0) == 1.0  # 2^-26 is dropped
    assert one([2.0 ** -12] * 3, [2.0 ** -13] * 3, 1.0) == 1.0  # 0.75 ulp: toward zero
    assert one([-2.0 ** -12], [2.0 ** -13], 1.0) == 1.0 - 2.0 ** -24  # below the binade, toward zero
    assert one([2.0 ** 127] * 2, [1.0, 1.0], dtype="bf16") == math.inf  # not the RZ largest finite
    assert one([2.0 ** 70, -2.0 ** 70, 1.0], [2.0 ** 70, 2.0 ** 70, 1.0], dtype="bf16") == 0.0
    assert math.copysign(1.0, one([0.0], [0.0], -0.0)) == 1.0  # an exact zero is +0
    assert math.isnan(one([math.inf], [0.0], dtype="bf16")) and math.isnan(one([math.inf, math.inf], [1.0, -1.0]))
    assert one([-math.inf], [1.0], -2.0) == -math.inf


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_scalar_and_vectorized_models_agree(dtype):
    gen = torch.Generator().manual_seed(3)
    for category in mm.STRESS_CATEGORIES:
        a, b, acc = mm._stress_inputs(category, dtype, 96, 48, gen)
        vector = mm.sm120_mma_batched(a, b, acc, dtype=dtype)
        scalar = torch.tensor([mm.sm120_mma_accumulate(a[i].tolist(), b[i].tolist(), acc[i].item(), dtype=dtype)
                               for i in range(len(a))])
        assert torch.equal(mm._bits(vector), mm._bits(scalar)), category
    a, b, acc = mm._stress_inputs("random_mantissa", dtype, 64, 16, gen)
    for i in range(len(a)):
        assert mm.variant_accumulate(a[i].tolist(), b[i].tolist(), acc[i].item(), dtype=dtype) == \
            mm.sm120_mma_accumulate(a[i].tolist(), b[i].tolist(), acc[i].item(), dtype=dtype)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_vectorized_diagonal_predictor_equals_the_exact_chain(dtype):
    gen = torch.Generator().manual_seed(4)
    for k, lo, hi in ((16, -12, -3), (1000, -9, -1), (20000, -12, -2), (4096, -24, -14)):
        row = mm.normal_mantissa((k,), gen, lo, hi).to(mm.DTYPES[dtype]).double()
        exact = mm.sm120_mma_batched(row[None], row[None], dtype=dtype).item()
        assert mm.sm120_predict_diag(row, dtype=dtype) == exact, (k, lo)


# ------------------------------------------------------------------ guards and provenance
def test_reserved_outputs_and_cli_guards(tmp_path):
    other = tmp_path / "checkout" / "results" / c5.CANONICAL_OUTPUT.name
    alias = tmp_path / "alias.json"
    alias.symlink_to(other)
    novel = tmp_path / "cycle5_c5_01_other.json"
    assert all(c5.reserved(p) for p in (c5.CANONICAL_OUTPUT, other, alias, novel))
    assert not c5.reserved(tmp_path / "custom.json") and not c5.reserved(c5.QUICK_OUTPUT)
    args = c5.parse_args([])
    assert args.canonical and args.output == c5.CANONICAL_OUTPUT and args.protocol == c5.CANONICAL
    assert args.protocol is not c5.CANONICAL
    assert not c5.parse_args(["--quick"]).canonical
    for argv in (["--quick", "--output", str(c5.CANONICAL_OUTPUT)], ["--quick", "--output", str(alias)],
                 ["--quick", "--output", str(novel)], ["--output", str(tmp_path / "custom.json")],
                 ["--output", str(novel)]):
        with pytest.raises(SystemExit):
            c5.parse_args(argv)


def test_canonical_runs_need_sm_120_and_a_clean_tree(monkeypatch, tmp_path):
    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    clean, dirty = {"git_sha": "a" * 40, "git_dirty": False}, {"git_sha": "a" * 40, "git_dirty": True}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    c5.require_canonical_host(cuda, clean, c5.CANONICAL_OUTPUT, True)
    c5.require_canonical_host(cpu, dirty, tmp_path / "custom.json", False)  # not canonical: no guard
    for git in (dirty, {"git_sha": None, "git_dirty": None}):
        with pytest.raises(RuntimeError, match="clean"):
            c5.require_canonical_host(cuda, git, c5.CANONICAL_OUTPUT, True)
    with pytest.raises(RuntimeError, match="sm_120"):
        c5.require_canonical_host(cpu, clean, c5.CANONICAL_OUTPUT, True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(RuntimeError, match="sm_120"):
        c5.require_canonical_host(cuda, clean, tmp_path / "cycle5_c5_01_x.json", False)
    monkeypatch.setattr(c5, "set_numerics", lambda device: pytest.fail("guards must run first"))
    monkeypatch.setattr(c5.h, "git_info", lambda: dirty)
    with pytest.raises(RuntimeError):
        c5.run(c5.parse_args([]))


def test_code_sha256_covers_the_kernels_and_the_model():
    modules = c5.code_modules()
    for needed in c5.REQUIRED_CODE:
        assert any(m.endswith("." + needed) for m in modules), needed
    assert len(c5.code_sha256()) == 64 and c5.code_sha256() == c5.code_sha256()


def test_williams_order_and_timing_summary():
    assert [c5.williams_order(r) for r in range(4)] == [["native", "control"], ["control", "native"]] * 2
    rounds = [{"round": r, "order": c5.williams_order(r), "native": {"gpu_ms": 1.0 + r, "wall_ms": 2.0},
               "control": {"gpu_ms": 1.0 + r, "wall_ms": 1.0}} for r in range(5)]
    summary = c5.timing_summary(rounds)
    assert summary["gpu"]["median_ratio"] == 1.0 and summary["wall"]["median_ratio"] == 2.0
    assert summary["gpu"]["paired"]["ci95_low"] == summary["gpu"]["paired"]["ci95_high"] == 1.0


# ------------------------------------------------------------------ kernels on the GPU
@needs_sm120
def test_short_k_is_bitwise_the_pre_c5_01_code_with_unchanged_launches():
    legacy = c5.load_legacy()
    device = torch.device("cuda")
    for shape in ([256, 1024], [2, 64, 256], [300, 70], [1, 512], [4, 65536], [65535, 2]):
        G = c5._input(shape, 1, device)
        for kernel in ("ps3_55", "ss3_54", "ss3_53", "gram_ns", "triton_syrk"):
            for dtype in ("fp16", "bf16"):
                native = c5.circuit(kernel, dtype)(G)
                c5.share_autotune(legacy)
                assert c5.bit_equal(native, c5.circuit(kernel, dtype, legacy["modules"])(G)), (shape, kernel, dtype)
    G = c5._input([512, 2048], 0, device)
    for kernel, launches in c5.LAUNCHES.items():
        native = c5.circuit(kernel, "fp16")
        assert c5.profiled_launches(native, G) == launches
        c5.share_autotune(legacy)
        assert launches == c5.profiled_launches(
            c5.circuit(kernel, "fp16", legacy["modules"]), G)
        assert c5.sync_free(native, G)


@needs_sm120
def test_chunked_syrk_is_symmetric_sync_free_and_rejects_overflowing_strides():
    device = torch.device("cuda")
    for shape in ([4, 131072], [70000, 5], [2, 3, 69633]):
        X = (c5._input(shape, 2, device) / 300.0).half()
        view = X.mT if X.shape[-2] > X.shape[-1] else X
        view = view if view.ndim == 3 else view[None]
        for out_dtype in (torch.float16, torch.bfloat16, torch.float32):
            out = torch.empty(view.shape[0], view.shape[1], view.shape[1], dtype=out_dtype, device=device)
            kbw._syrk_chunked(view, out)
            assert c5.bit_equal(out, out.mT.contiguous())
        exact = c3.gram64(view[0])
        assert (1 - out[0].double().diagonal() / exact.diagonal()).abs().max() < 5e-5  # no long-K drift
        assert c5.sync_free(lambda _: kbw._syrk_chunked(view, out), None)
    big = torch.empty(1, 2, 1 << 25, dtype=torch.float16, device=device)[:, :, ::1 << 24]  # column stride 2^24
    with pytest.raises(ValueError, match="int32"):
        kbw._syrk_chunked(big, torch.empty(1, 2, 2, device=device))


@needs_sm120
def test_long_k_circuits_equal_the_audited_chunked_gram_and_replay_in_graphs():
    device = torch.device("cuda")
    for shape in ([4, 65537], [69633, 2], [2, 3, 100000]):
        G = c5._input(shape, 7, device)
        for kernel in c5.KERNELS:
            for dtype in ("fp16", "bf16"):
                call = c5.circuit(kernel, dtype)
                native = call(G)
                assert c5.bit_equal(native, c5._chunked_reference(call, G)), (shape, kernel, dtype)
    for kernel in c5.KERNELS:
        result = c5.graph_check(c5.circuit(kernel, "fp16"), [4, 131072], 21, device)
        assert result["bitwise_equal"] and result["max_abs_diff"] == 0.0
        assert c5.sync_free(c5.circuit(kernel, "fp16"), c5._input([4, 131072], 1, device))


@needs_sm120
def test_c4_03_chunked_gram_still_hooks_the_long_k_launches():
    G = c5._input([4, 1 << 17], 0, torch.device("cuda"))
    mitigation = c3.ChunkedGram(4096, 65536)
    with mitigation.patched():
        assert ssb.SYRK_CHUNK_GATE == 1 << 60
        c3.kernel_call("ss3_54", "fp16")(G)
    assert mitigation.launches > 0 and ssb.SYRK_CHUNK_GATE == 65536


@needs_sm120
def test_live_mma_matches_the_model_on_witnesses_and_random_dots():
    for values, dtype, target in ((mm.ASTRA_FP16_WITNESS, "fp16", mm.ASTRA_FP16_HARDWARE),
                                  (mm.BF16_WITNESS, "bf16", mm.ASTRA_BF16_HARDWARE)):
        row = torch.tensor([_row(values, dtype)])
        measured, ops = mm.probe_dots(row, row, dtype=dtype)
        assert measured.item() == float.fromhex(target)
        ptx = dtype.replace("fp", "f")
        assert ops == [f"mma.sync.aligned.m16n8k16.row.col.f32.{ptx}.{ptx}.f32"]
    gen = torch.Generator().manual_seed(11)
    for dtype in ("fp16", "bf16"):
        for category in mm.STRESS_CATEGORIES:
            a, b, acc = mm._stress_inputs(category, dtype, 512, 16, gen)
            measured, _ = mm.probe_dots(a, b, acc, dtype=dtype)
            assert torch.equal(mm._bits(measured), mm._bits(mm.sm120_mma_batched(a, b, acc, dtype=dtype))), category
        a = mm.normal_mantissa((64, 2048), gen, -9, -3).to(mm.DTYPES[dtype]).double()
        for bk in (16, 128):
            measured, _ = mm.probe_dots(a, a, dtype=dtype, bk=bk)
            assert torch.equal(mm._bits(measured), mm._bits(mm.sm120_mma_batched(a, a, dtype=dtype)))


# ------------------------------------------------------------------ the driver
@pytest.fixture(scope="module")
def quick(tmp_path_factory):
    if not CUDA:
        pytest.skip("needs the RTX 5080 (sm_120)")
    path = tmp_path_factory.mktemp("c5") / "quick.json"
    return c5.main(["--quick", "--silent", "--output", str(path)])


def _edit(payload, path, value=None, delta=None):
    payload = copy.deepcopy(payload)
    node = payload
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = node[path[-1]] + delta if delta is not None else value
    return payload


def _tamper_checks(payload, canonical):
    c5.validate_results(payload, canonical=canonical)
    cell = payload["timing"]["cells"][0]
    for edited in (
        _edit(payload, ["timing", "cells", 0, "rounds", 1, "native", "gpu_ms"], delta=1e-3),
        _edit(payload, ["timing", "cells", 0, "rounds", 0, "order"], list(reversed(cell["rounds"][0]["order"]))),
        _edit(payload, ["timing", "cells", 0, "summary", "wall", "median_ratio"], delta=1e-6),
        _edit(payload, ["long_k", "opus_l6", 0, "samples", 0, "native", "wall_ms"], delta=1.0),
        _edit(payload, ["long_k", "grid", 0, "cells", "ss3_54_fp16", "sigma_max"], 5.0),
        _edit(payload, ["mma", "stress", 0, "bit_matches"], delta=-1),
        _edit(payload, ["zero_regression", "rows", 0, "bitwise_equal"], False),
        _edit(payload, ["meta", "code_sha256"], "0" * 64),
        _edit(payload, ["meta", "environment", "driver"], None),
        _edit(payload, ["protocol", "gate"], 1 << 17),
        _edit(payload, ["schema_version"], "cycle5_c5_01.v0"),
    ):
        with pytest.raises(ValueError):
            c5.validate_results(edited, canonical=canonical)


def test_quick_driver_payload_is_valid_and_rejects_tampering(quick):
    _tamper_checks(quick, False)
    with pytest.raises(ValueError):
        c5.validate_results(quick, canonical=True)  # --quick is never canonical
    d = quick["derived"]
    assert d["zero_regression"]["bitwise_equal"] == d["zero_regression"]["cells"]
    assert d["zero_regression"]["launches_at_512x2048"] == c5.LAUNCHES
    assert d["long_k"]["grid_above_design_ceiling"] == 0
    assert d["mma"]["all_dot_products"][0] == d["mma"]["all_dot_products"][1]


def test_committed_canonical_artifact():
    if not c5.CANONICAL_OUTPUT.exists():
        pytest.skip("the canonical C5-01 artifact is not committed in this checkout")
    payload = json.loads(c5.CANONICAL_OUTPUT.read_text())
    _tamper_checks(payload, True)
    assert payload["canonical"] is True and payload["meta"]["git_dirty"] is False
    assert payload["meta"]["environment"]["capability"] >= [12, 0]
    d = payload["derived"]
    assert d["zero_regression"]["bitwise_equal"] == d["zero_regression"]["cells"] > 0
    assert d["long_k"]["grid_cells"] == 654 and d["long_k"]["grid_above_design_ceiling"] == 0
    assert d["long_k"]["graphs_max_abs_diff"] == 0.0
    assert d["mma"]["all_dot_products"][1] >= 50_000
