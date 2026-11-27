"""Checks for Cycle 5 C5-01: native long-K chunked SYRK and the exact sm_120 MMA model."""

import copy
import json
import math
import random
import struct
from fractions import Fraction

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
    # alignment floor: with C = 0 and products below 2^-133, terms still align to 2^-158
    assert one([2.0 ** -70, -2.0 ** -80], [2.0 ** -70, 2.0 ** -78], dtype="bf16") == 2.0 ** -140 - 2.0 ** -149
    assert one([2.0 ** -70, -2.0 ** -80], [2.0 ** -70, 2.0 ** -79], dtype="bf16") == 2.0 ** -140
    assert one([2.0 ** -70, -2.0 ** -80], [2.0 ** -70, 2.0 ** -79], 2.0 ** -149, dtype="bf16") == 2.0 ** -140 + 2.0 ** -149


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



# Raw words of the 16 NaN cases (8 per operand dtype): A lanes, B lanes, C word -> 0x7fffffff.
_WORDS = {"fp16": {"inf": 0x7C00, "snan": 0x7C01, "qnan": 0x7E00, "one": 0x3C00, "minus_one": 0xBC00},
          "bf16": {"inf": 0x7F80, "snan": 0x7F81, "qnan": 0x7FC0, "one": 0x3F80, "minus_one": 0xBF80}}


def _nan_cases():
    cases = []
    for dtype, w in _WORDS.items():
        cases += [(dtype, "inf_x_0", [w["inf"]], [0], 0), (dtype, "inf_minus_inf", [w["inf"], w["inf"]],
                                                            [w["one"], w["minus_one"]], 0),
                  (dtype, "snan_a", [w["snan"]], [w["one"]], 0), (dtype, "qnan_a", [w["qnan"]], [w["one"]], 0),
                  (dtype, "snan_b", [w["one"]], [w["snan"]], 0), (dtype, "qnan_b", [w["one"]], [w["qnan"]], 0),
                  (dtype, "snan_c", [w["one"]], [w["one"]], 0x7F800001),
                  (dtype, "qnan_c", [w["one"]], [w["one"]], 0x7FC00000)]
    return cases


NAN_CASES = _nan_cases()
# Astra's BF16 negative-underflow witness: (-2^-80) * (+2^-80) truncates to a zero magnitude -> +0.
UNDERFLOW_WITNESS = ("bf16", "neg_underflow", [0x9780], [0x1780], 0)


def _raw16(words) -> torch.Tensor:
    """16 FP16/BF16 lane words (zero padded) as int16."""
    return torch.tensor([w - (1 << 16) if w >= 1 << 15 else w for w in words] + [0] * (16 - len(words)),
                        dtype=torch.int16)


def _decode(words, dtype: str) -> torch.Tensor:
    """FP16/BF16 words -> FP64 values of the 16 lanes."""
    return _raw16(words).view(mm.DTYPES[dtype]).double()


def _acc(word: int) -> torch.Tensor:
    return torch.tensor([word - (1 << 32) if word >= 1 << 31 else word], dtype=torch.int32).view(torch.float32)


def _word32(x) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _model_words(dtype, a_words, b_words, c_word, device) -> list[int]:
    """Output words of the scalar (struct and torch encodings) and batched (CPU / CUDA) models."""
    a, b, c = _decode(a_words, dtype), _decode(b_words, dtype), _acc(c_word)
    scalar = mm.sm120_mma_accumulate(a.tolist(), b.tolist(), c.item(), dtype=dtype)
    words = [_word32(scalar), int(mm._bits(torch.tensor([scalar], dtype=torch.float32))[0]) & 0xFFFFFFFF]
    for dev in ("cpu", *([device] if device else [])):
        vector = mm.sm120_mma_batched(a[None].to(dev), b[None].to(dev), c.double().to(dev), dtype=dtype)
        words.append(int(mm._bits(vector)[0]) & 0xFFFFFFFF)
    return words


def _hardware_word(dtype, a_words, b_words, c_word) -> int:
    """One live k16 instruction on raw words: A row 0, B column 0, C[0, 0] (the loads keep every bit)."""
    A = torch.zeros(1, 16, 16, dtype=torch.int16)
    B = torch.zeros(1, 16, 16, dtype=torch.int16)
    C = torch.zeros(1, 16, 16, dtype=torch.int32)
    A[0, 0], B[0, :, 0], C[0, 0, 0] = _raw16(a_words), _raw16(b_words), _acc(c_word).view(torch.int32)[0]
    D, _ = mm.probe_tiles(A.view(mm.DTYPES[dtype]), B.view(mm.DTYPES[dtype]), C.view(torch.float32))
    return int(D.view(torch.int32)[0, 0, 0]) & 0xFFFFFFFF


@pytest.mark.parametrize("case", NAN_CASES, ids=[f"{c[0]}-{c[1]}" for c in NAN_CASES])
def test_model_nan_word_is_exactly_the_mma_nan(case):
    dtype, _, a_words, b_words, c_word = case
    device = "cuda" if torch.cuda.is_available() else None
    assert _model_words(dtype, a_words, b_words, c_word, device) == [mm.NAN_BITS] * (3 + (device is not None))


def test_model_negative_underflow_is_plus_zero():
    dtype, _, a_words, b_words, c_word = UNDERFLOW_WITNESS
    device = "cuda" if torch.cuda.is_available() else None
    assert _model_words(dtype, a_words, b_words, c_word, device) == [0] * (3 + (device is not None))


def _underflow_batch(count: int, seed: int):
    """BF16 dots whose products and accumulators straddle the FP32 subnormal floor 2^-149, both signs."""
    gen = torch.Generator().manual_seed(seed)
    a = mm.normal_mantissa((count, 16), gen, -80, -66).to(torch.bfloat16).double()
    b = mm.normal_mantissa((count, 16), gen, -80, -66).to(torch.bfloat16).double()
    lanes = torch.rand(count, 16, generator=gen)
    a = torch.where(lanes < 0.5, 0.0, a)  # most rows hold a few products
    a[:count // 4, 1:] = 0.0  # a single product (the witness form)
    acc = torch.where(torch.rand(count, generator=gen) < 0.5, 0.0, mm.normal_mantissa((count,), gen, -150, -126))
    return a, b, acc.float().double()


@pytest.mark.parametrize("seed", [0, 1])
def test_model_bf16_underflow_batches_scalar_equals_batched_on_cpu_and_cuda(seed):
    a, b, acc = _underflow_batch(2048, seed)
    scalar = torch.tensor([mm.sm120_mma_accumulate(a[i].tolist(), b[i].tolist(), acc[i].item(), dtype="bf16")
                           for i in range(len(a))], dtype=torch.float32)
    cpu = mm.sm120_mma_batched(a, b, acc, dtype="bf16")
    assert torch.equal(mm._bits(scalar), mm._bits(cpu))
    assert int((mm._bits(cpu) == 0).sum()) > 0 and bool((cpu < 0).any())  # +0 results and negative results occur
    assert not bool((mm._bits(cpu) == -(1 << 31)).any())  # no -0 word (0x80000000)
    if torch.cuda.is_available():
        cuda = mm.sm120_mma_batched(a.cuda(), b.cuda(), acc.cuda(), dtype="bf16")
        assert torch.equal(mm._bits(cuda), mm._bits(cpu))


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
    assert c5.parse_args(["--output", str(c5.CANONICAL_OUTPUT)]).canonical
    real_alias = tmp_path / "real_alias.json"
    real_alias.symlink_to(c5.CANONICAL_OUTPUT)
    for argv in (["--quick", "--output", str(c5.CANONICAL_OUTPUT)], ["--quick", "--output", str(alias)],
                 ["--quick", "--output", str(novel)], ["--output", str(tmp_path / "custom.json")],
                 ["--output", str(novel)],
                 ["--output", str(other)],  # the canonical basename in another checkout
                 ["--output", str(alias)],  # a symlink to it
                 ["--output", str(real_alias)]):  # a symlink to this checkout's canonical file
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


# ------------------------------------------------------------------ single rounding FP64 -> FP16 / BF16 / FP32
_FORMATS = {torch.float16: (10, -14, 15, 16), torch.bfloat16: (7, -126, 127, 16), torch.float32: (23, -126, 127, 32)}


def _round_once(x: float, dtype: torch.dtype) -> int | None:
    """Independent oracle: the IEEE-754 round-to-nearest-even word of ``x`` (None for NaN), with Fraction."""
    p, emin, emax, width = _FORMATS[dtype]
    sign = 1 << (width - 1) if math.copysign(1.0, x) < 0 else 0
    inf = ((1 << (width - 1 - p)) - 1) << p
    if math.isnan(x):
        return None
    if math.isinf(x):
        return sign | inf
    f = abs(Fraction(x))
    if f == 0:
        return sign
    quantum = Fraction(2) ** (max(math.frexp(abs(x))[1] - 1, emin) - p)
    value = round(f / quantum) * quantum  # Fraction.__round__ rounds half to even
    if value == 0:
        return sign
    if value >= Fraction(2) ** (emax + 1):
        return sign | inf
    e = math.frexp(float(value))[1] - 1
    if e < emin:
        return sign | int(value / Fraction(2) ** (emin - p))
    return sign | ((e + 1 - emin) << p) | (int(value / Fraction(2) ** (e - p)) - (1 << p))


_SINGLE_ROUND_CASES = [
    # Astra's K = 65,537 witnesses (FP16, BF16, FP32) and their negatives
    1 + 2 ** -11 + 2 ** -26, 1 + 2 ** -8 + 2 ** -26, 1 + 2 ** -24 + 2 ** -48,
    # exact ties: to even, both directions
    1 + 2 ** -11, 1 + 3 * 2 ** -11, 1 + 2 ** -8, 1 + 3 * 2 ** -8, 1 + 2 ** -24, 1 + 3 * 2 ** -24,
    # subnormal midpoints and their successors (FP16 2^-25, BF16 / FP32 2^-134, 2^-150)
    2 ** -25, 2 ** -25 + 2 ** -60, 3 * 2 ** -25, 2 ** -134, 2 ** -134 + 2 ** -170, 3 * 2 ** -134, 2 ** -150,
    2 ** -150 + 2 ** -200,
    # largest subnormal + half a quantum: carry into the smallest normal
    1023.5 * 2 ** -24, 127.5 * 2 ** -133, 1023.25 * 2 ** -24, 2 ** -14 * (1 - 2 ** -12),
    # overflow boundaries
    65504.0, 65519.99, 65520.0, 65536.0, (2 - 2 ** -7) * 2 ** 127, (2 - 2 ** -8) * 2 ** 127,
    math.nextafter((2 - 2 ** -8) * 2 ** 127, 0), (2 - 2 ** -24) * 2 ** 127, 1e300, 5e-324,
    0.0, math.inf, math.nan,
]


def _random_fp64(count: int, seed: int) -> list[float]:
    """Random FP64 words over every destination range, plus values one FP64 ulp around destination midpoints."""
    gen = torch.Generator().manual_seed(seed)
    words = torch.randint(-(1 << 63), (1 << 63) - 1, (count,), generator=gen, dtype=torch.int64)
    exponent = torch.randint(1023 - 160, 1023 + 140, (count,), generator=gen, dtype=torch.int64)
    words[: count * 9 // 10] = (words[: count * 9 // 10] & ~(0x7FF << 52)) | (exponent[: count * 9 // 10] << 52)
    values = words.view(torch.float64).tolist()  # the last tenth: fully random words (NaN, inf, subnormal)
    for dtype, (p, emin, _, _) in _FORMATS.items():
        base = torch.randn(count // 10, generator=gen, dtype=torch.float64) * torch.exp2(
            torch.randint(emin - p - 2, 18, (count // 10,), generator=gen).double())
        for v in base.to(dtype).double().tolist():
            if math.isfinite(v) and v != 0:
                half = 2.0 ** (max(math.frexp(abs(v))[1] - 1, emin) - p - 1)
                mid = v + math.copysign(half, v)
                values += [mid, math.nextafter(mid, math.inf), math.nextafter(mid, -math.inf)]
    return values


def _store_words(values: list[float], dtype: torch.dtype) -> list[int]:
    src = torch.tensor(values, dtype=torch.float64, device="cuda")
    out = torch.empty(len(values), dtype=dtype, device="cuda")
    kbw._store_fp64_single_round(src, out)
    mask = (1 << _FORMATS[dtype][3]) - 1
    return [w & mask for w in out.view(torch.int16 if dtype != torch.float32 else torch.int32).tolist()]


@needs_sm120
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_store_fp64_single_round_equals_a_fraction_oracle(dtype):
    values = [*_SINGLE_ROUND_CASES, *(-v for v in _SINGLE_ROUND_CASES), *_random_fp64(12000, 5)]
    assert len(values) >= 10_000
    words = _store_words(values, dtype)
    qnan = {torch.float16: 0x7E00, torch.bfloat16: 0x7FC0}.get(dtype)
    for x, word in zip(values, words):
        expected = _round_once(x, dtype)
        if expected is None:  # NaN: the quiet NaN of FP16 / BF16, any NaN for the FP32 conversion
            assert word == qnan if qnan else math.isnan(struct.unpack("<f", struct.pack("<I", word))[0]), hex(word)
        else:
            assert word == expected, (x.hex(), hex(word), hex(expected))
    finite = [x for x in values if not math.isnan(x)]
    signbit = 1 << (_FORMATS[dtype][3] - 1)
    assert [w ^ signbit for w in _store_words(finite, dtype)] == _store_words([-x for x in finite], dtype)


@needs_sm120
def test_store_fp64_single_round_astra_witness_words():
    for dtype, x, words in ((torch.float16, 1 + 2 ** -11 + 2 ** -26, (0x3C01, 0xBC01)),
                            (torch.bfloat16, 1 + 2 ** -8 + 2 ** -26, (0x3F81, 0xBF81)),
                            (torch.float32, 1 + 2 ** -24 + 2 ** -48, (0x3F800001, 0xBF800001))):
        assert tuple(_store_words([x, -x], dtype)) == words
    assert _store_words([2 ** -25, 2 ** -25 + 2 ** -60, -2 ** -25], torch.float16) == [0x0000, 0x0001, 0x8000]
    assert _store_words([2 ** -134, 2 ** -134 + 2 ** -170], torch.bfloat16) == [0x0000, 0x0001]
    assert _store_words([1023.5 * 2 ** -24, 65520.0, -65520.0], torch.float16) == [0x0400, 0x7C00, 0xFC00]
    assert _store_words([math.nan, -math.nan, math.inf, -math.inf, 0.0, -0.0], torch.bfloat16) == \
        [0x7FC0, 0x7FC0, 0x7F80, 0xFF80, 0x0000, 0x8000]


@needs_sm120
@pytest.mark.parametrize("out_dtype, lanes, words", [
    (torch.float16, [1.0, 2 ** -6, 2 ** -6, 2 ** -13], (0x3C01, 0xBC01)),
    (torch.bfloat16, [1.0, 2 ** -4, 2 ** -13], (0x3F81, 0xBF81)),
    (torch.float32, [1.0, 2 ** -12, 2 ** -24], (0x3F800001, 0xBF800001)),
])
def test_chunked_syrk_rounds_the_fp64_sum_once(out_dtype, lanes, words):
    """Astra's K = 65,537 witness: one product per chunk, the FP64 sum just above a destination midpoint."""
    row = torch.zeros(65537, dtype=torch.float64)
    for i, v in enumerate(lanes):
        row[4096 * i] = v
    X = torch.stack([row, -row]).half().cuda()
    width = torch.int16 if out_dtype != torch.float32 else torch.int32
    mask = 0xFFFF if out_dtype != torch.float32 else 0xFFFFFFFF
    positive, negative = words
    for gram in (kbw.triton_syrk(X, out_dtype), kbw.triton_syrk(X.T.contiguous().T, out_dtype)):
        assert [[w & mask for w in r] for r in gram.view(width).tolist()] == [[positive, negative],
                                                                              [negative, positive]]
    G = c5._input([4, 65537], 3, torch.device("cuda"))  # the historical hook rounds the same way
    for kernel in c5.KERNELS:
        call = c5.circuit(kernel, "bf16")
        assert c5.bit_equal(call(G), c5._chunked_reference(call, G))


@needs_sm120
def test_chunked_syrk_caps_programs_per_launch_at_grid_y_and_max_bytes():
    device = torch.device("cuda")
    sizes = []

    def record(kernel, grid, *args):
        sizes.append(args[0].shape[0])
        kbw._direct_launch(kernel, grid, *args)

    X = torch.ones(1, 1 << 28, dtype=torch.float16, device=device)  # 512 MiB: 65,536 chunks
    out = torch.empty(1, 1, 1, dtype=torch.float32, device=device)
    kbw._syrk_chunked(X[None], out, record)
    assert sizes == [65535, 1] and out.item() == 268435456.0
    assert kbw.triton_syrk(X, torch.float32).item() == 268435456.0  # the public API on a contiguous input
    del X, out
    torch.cuda.empty_cache()
    gen = torch.Generator().manual_seed(9)
    X = torch.randint(-2, 3, (256, 70000), generator=gen).half()  # integer Gram: every partial and sum exact
    exact = (X.double() @ X.double().T).float()
    view = X.cuda()[None]
    capped = torch.empty(1, 256, 256, dtype=torch.float32, device=device)
    sizes.clear()
    kbw._syrk_chunked(view, capped, record, max_bytes=1 << 20)  # 4 chunk Gram matrices per launch
    assert sizes == [4, 4, 4, 4, 1, 1]  # 17 chunks, then the 368-column remainder
    default = torch.empty_like(capped)
    sizes.clear()
    kbw._syrk_chunked(view, default, record)
    assert sizes == [17, 1]
    assert c5.bit_equal(capped[0].cpu(), exact) and c5.bit_equal(default, capped)


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
    for kernel in ("ps3_55", "ss3_54"):
        mitigation = c3.ChunkedGram(4096, 65536)
        with mitigation.patched():
            assert psb.SYRK_CHUNK_GATE == ssb.SYRK_CHUNK_GATE == 1 << 60
            c3.kernel_call(kernel, "fp16")(G)
        assert mitigation.launches > 0 and psb.SYRK_CHUNK_GATE == ssb.SYRK_CHUNK_GATE == 65536


@needs_sm120
def test_c4_03_hook_honors_a_custom_gate_above_65536():
    """``ChunkedGram(4096, 131072)`` at K = 100,000: the single unchunked launch on both circuits."""
    G = c5._input([4, 100000], 983, torch.device("cuda"))
    gram, launch = psb._gram, ssb._launch
    for kernel in ("ps3_55", "ss3_54"):
        call = c5.circuit(kernel, "fp16")
        with c5.ungated():
            single = call(G)
        mitigation = c3.ChunkedGram(4096, 131072)
        with mitigation.patched():
            assert psb.SYRK_CHUNK_GATE == ssb.SYRK_CHUNK_GATE == 1 << 60
            hooked = call(G)
        assert mitigation.launches == 0 and c5.bit_equal(hooked, single), kernel
        assert not c5.bit_equal(call(G), single)  # the native path chunks above 65,536
        assert psb.SYRK_CHUNK_GATE == ssb.SYRK_CHUNK_GATE == 65536
    with pytest.raises(RuntimeError, match="inside the hook"):
        with c3.ChunkedGram(4096, 131072).patched():
            raise RuntimeError("inside the hook")
    assert psb.SYRK_CHUNK_GATE == ssb.SYRK_CHUNK_GATE == 65536
    assert psb._gram is gram and ssb._launch is launch


@needs_sm120
def test_graph_equality_replays_both_graphs_before_comparing():
    def capture(value):
        out = torch.zeros(4, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out.fill_(value)
        return graph, out

    divergent = {"native": capture(1.0), "control": capture(2.0)}
    assert c5.bit_equal(divergent["native"][1], divergent["control"][1])  # captured, not yet run: both zero
    assert not c5.graphs_bitwise_equal(divergent, torch.ones(4, device="cuda"))
    assert divergent["native"][1].tolist() == [1.0] * 4 and divergent["control"][1].tolist() == [2.0] * 4
    same = {"native": capture(1.0), "control": capture(1.0)}
    assert c5.graphs_bitwise_equal(same, torch.ones(4, device="cuda"))
    assert not c5.graphs_bitwise_equal(same, torch.full((4,), 2.0, device="cuda"))  # both differ from eager


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


@needs_sm120
def test_live_mma_raw_words_of_nan_and_underflow_cases_equal_the_model():
    for dtype, name, a_words, b_words, c_word in (*NAN_CASES, UNDERFLOW_WITNESS):
        expected = 0 if name == "neg_underflow" else mm.NAN_BITS
        assert _hardware_word(dtype, a_words, b_words, c_word) == expected, (dtype, name)
        assert _model_words(dtype, a_words, b_words, c_word, "cuda") == [expected] * 4, (dtype, name)
    for seed in (0, 1):
        a, b, acc = _underflow_batch(2048, seed)
        measured, _ = mm.probe_dots(a, b, acc, dtype="bf16")
        assert torch.equal(mm._bits(measured), mm._bits(mm.sm120_mma_batched(a, b, acc, dtype="bf16")))
        assert torch.equal(mm._bits(measured), mm._bits(mm.sm120_mma_batched(a.cuda(), b.cuda(), acc.cuda(),
                                                                            dtype="bf16")))
        scalar = torch.tensor([mm.sm120_mma_accumulate(a[i].tolist(), b[i].tolist(), acc[i].item(), dtype="bf16")
                               for i in range(256)], dtype=torch.float32)
        assert torch.equal(mm._bits(measured[:256]), mm._bits(scalar))


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
    for edit, match in _structural_probes(canonical):
        with pytest.raises(ValueError, match=match):
            c5.validate_results(_rederived(payload, edit), canonical=canonical)


def _rederived(payload, edit):
    """``payload`` after ``edit``, with ``derived`` recomputed so only the probed check can fail."""
    payload = copy.deepcopy(payload)
    edit(payload)
    payload["derived"] = json.loads(json.dumps(c5.derive(payload), allow_nan=False))
    return payload


def _set(path, value):
    def edit(payload):
        node = payload
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return edit


def _mma_mismatch(payload):
    record = next(r for r in payload["mma"]["stress"] if r["bit_matches"] == r["cases"])
    record["bit_matches"] -= 1
    record["mismatches"] = [{"index": 0, "hardware": "0x1.0p+0", "model": "0x1.000002p+0"}]


def _structural_probes(canonical):
    """Astra's validator bypasses (C5-01 audit): each must now be rejected."""
    probes = [
        (_set(["timing", "cells", 0, "dtype"], "fp64"), "dtype differs"),
        (lambda p: p["long_k"]["opus_l6"].pop(), "opus_l6 rows"),
        (_set(["long_k", "opus_l6", 0, "shape"], [2, 16]), "opus_l6 rows"),
        (_set(["long_k", "opus_l6", 0, "dtype"], "fp64"), "opus_l6 rows"),
        (_set(["long_k", "grid", 0, "shape"], [0, 0]), "shape differs"),
        (lambda p: p["long_k"]["syrk"].pop(), "syrk rows"),
        (lambda p: p["long_k"]["boundaries"].pop(0), "boundaries rows"),
        (lambda p: p["long_k"]["inputs"].pop(), "inputs rows"),
        (_set(["long_k", "graphs", 0, "shape"], [2, 16]), "graphs rows"),
    ]
    if canonical:
        probes += [
            (_mma_mismatch, "promotion: mma all_dot_products"),
            (_set(["mma", "edge_cases", 0, "model_equals_hardware"], False), "promotion: mma edge_cases"),
            (_set(["mma", "edge_cases", 0, "vectorized_equals_hardware"], False), "promotion: mma edge_cases"),
            (_set(["mma", "witnesses", 0, "configs_equal_hardware"], False), "promotion: mma witnesses"),
            (_set(["mma", "syrk_configs", 0, "fused_store_matches"], 0), "promotion: mma syrk_fused_store"),
            (lambda p: p["meta"]["code_modules"].append("research.polar_opt.bogus"), "code_modules"),
            (lambda p: p["meta"]["code_modules"].append(p["meta"]["code_modules"][0]), "code_modules"),
            (_set(["timing", "cells", 0, "outputs_bitwise_equal"], False), "timing cell's outputs"),
            (_set(["long_k", "graphs", 0, "bitwise_equal"], False), "promotion: long_k graphs_bitwise"),
            (_set(["long_k", "syrk", 0, "symmetric"], False), "promotion: long_k syrk_symmetric"),
            (_set(["long_k", "syrk", 0, "sync_free"], False), "promotion: long_k syrk_sync_free"),
            (_set(["long_k", "inputs", 0, "bitwise_equal_c4_03"], False), "promotion: long_k inputs_bitwise_c4_03"),
            (_set(["long_k", "opus_l6", 0, "bitwise_equal_c4_03"], False), "promotion: long_k opus_l6_bitwise_c4_03"),
            (_set(["long_k", "boundaries", 0, "bitwise_equal_reference"], False),
             "promotion: long_k boundaries_bitwise_reference"),
            (_set(["long_k", "opus_l6", 0, "native", "sigma_max"], 2.0), "Opus L6 cell exceeds"),
        ]
    return probes


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
    assert d["mma"]["all_dot_products"][0] == d["mma"]["all_dot_products"][1]
    assert d["mma"]["edge_cases"][0] == d["mma"]["edge_cases"][1]
    assert d["timing"]["outputs_bitwise_equal"] == d["timing"]["cells"]
    assert all(len(r["samples"]) == 4 for r in payload["long_k"]["opus_l6"])
