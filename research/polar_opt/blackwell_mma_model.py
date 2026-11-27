"""Exact model of the RTX 5080 (sm_120) Tensor-Core MMA used by the Blackwell Triton kernels (C5-01).

Instruction: ``mma.sync.aligned.m16n8k16.row.col.f32.{f16,bf16}.{f16,bf16}.f32``, the PTX that
Triton 3.8 emits for ``tl.dot`` on FP16/BF16 operands with an FP32 accumulator (``probe_ptx``
and ``syrk_raw`` read it from the compiled kernels). One instruction computes, per output entry,
``D = C + sum_{k < 16} a_k b_k``. Live probes on the RTX 5080 give this arithmetic:

1. Products are exact.
2. Alignment. ``E = max(e(C), max_k e(a_k) + e(b_k))`` over the non-zero terms. ``e`` is the
   *field* exponent: ``floor(log2 |x|)`` of a normal number and the smallest normal exponent of a
   subnormal one (FP16 -14, BF16 and FP32 -126). A product enters with the sum of its input
   exponents, before its significand in ``[1, 4)`` is normalized. This is the difference from the
   C4-03 rule, which aligned to the binade of the product itself.
3. Every addend (16 products and ``C``) is truncated toward zero to a multiple of ``2^(E - 25)``:
   24 bits plus 2 guard bits on the scale ``2^E``, no sticky bit.
4. The 17 truncated addends are summed exactly in one adder: no lane groups, no tree with
   intermediate normalization, and the result does not depend on the lane positions.
5. The sum is truncated toward zero to 24 significant bits (FP32 subnormals: multiples of
   ``2^-149``). A magnitude of ``2^128`` or more gives ``+-inf``, not the largest finite number.
   An exact zero is ``+0`` (also for ``C = -0``).
6. Specials. NaN anywhere, ``inf * 0``, or infinities of both signs give the NaN ``0x7fffffff``;
   otherwise an infinity gives itself.

A longer dot product is a chain: Triton walks K in ascending k16 slices (``tl.dot`` with ``BK`` of
32, 64 or 128 issues 2, 4 or 8 chained instructions) and each instruction takes the previous FP32
``D`` as ``C``. ``sm120_mma_accumulate`` is the scalar reference (Python integers),
``sm120_mma_batched`` the same arithmetic vectorized in FP64 (every intermediate is an exact
integer below 2^33 times a power of two), and ``sm120_predict_diag`` the diagonal of a very long
Gram row, vectorized over the binades of the accumulator.

The evidence (witnesses, exponent limits, lane permutations, alternative hypotheses and a
randomized stress suite over all 11 ``_CONFIGS``) is produced by ``characterize`` and recorded in
``results/cycle5_c5_01_syrk_and_mma.json``.
"""

from __future__ import annotations

import hashlib
import itertools
import math

import torch
import triton
import triton.language as tl

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
EMIN = {"fp16": -14, "bf16": -126}  # smallest normal exponent of the operand format
FP32_EMIN = -126
GUARD = 2  # bits kept below the 24-bit significand on the alignment scale
GROUP = 16  # products per instruction (k16)
NAN_BITS = 0x7FFFFFFF  # the NaN the MMA writes
_U = 320  # integer unit 2^-320 of the scalar reference: products of BF16 values reach 2^-266


# ======================================================================================
# Scalar reference (Python integers)
# ======================================================================================
def _to_int(x: float) -> int:
    numerator, denominator = float(x).as_integer_ratio()
    return (numerator << _U) // denominator  # exact: the denominator is a power of two <= 2^149


def _field_exponent(x: float, emin: int) -> int:
    return max(math.frexp(abs(x))[1] - 1, emin)


def _rz(value: int, shift: int) -> int:
    """Drop the lowest ``shift`` bits of the magnitude (round toward zero)."""
    if shift <= 0:
        return value
    return (abs(value) >> shift << shift) * (1 if value >= 0 else -1)


def _special(a, b, acc: float) -> float | None:
    """NaN / infinity result of one instruction, or None for finite inputs."""
    if math.isnan(acc) or any(math.isnan(x) or math.isnan(y) for x, y in zip(a, b)):
        return math.nan
    signs = set()
    if math.isinf(acc):
        signs.add(acc > 0)
    for x, y in zip(a, b):
        if math.isinf(x) or math.isinf(y):
            if x == 0 or y == 0:
                return math.nan
            signs.add((x > 0) == (y > 0))
    if len(signs) == 2:
        return math.nan
    return math.inf if signs == {True} else -math.inf if signs else None


def sm120_mma_group(a, b, acc: float = 0.0, *, dtype: str = "fp16") -> float:
    """One k16 instruction ``acc + sum(a_k b_k)`` (at most 16 products) with the arithmetic above."""
    special = _special(a, b, acc)
    if special is not None:
        return special
    emin = EMIN[dtype]
    terms, exponents = [], []
    for x, y in zip(a, b):
        if x != 0 and y != 0:
            terms.append(_to_int(x) * _to_int(y) >> _U)  # exact: both are multiples of 2^-160
            exponents.append(_field_exponent(x, emin) + _field_exponent(y, emin))
    if acc != 0:
        terms.append(_to_int(acc))
        exponents.append(_field_exponent(acc, FP32_EMIN))
    if not terms:
        return 0.0
    E = max(exponents)
    total = sum(_rz(t, _U + E - 23 - GUARD) for t in terms)
    if total == 0:
        return 0.0
    shift = max(abs(total).bit_length() - 24, _U - 149)  # 24 bits, FP32 subnormals at 2^-149
    value = _rz(total, shift)
    if abs(value) >= 1 << (_U + 128):
        return math.copysign(math.inf, value)
    return value / (1 << _U)  # exact: at most 24 significant bits


def sm120_mma_accumulate(a, b, acc: float = 0.0, *, dtype: str = "fp16") -> float:
    """``acc + sum(a_k b_k)`` as Triton issues it on sm_120: a chain of k16 instructions over K.

    ``a`` and ``b`` hold exact FP16/BF16 values (``dtype``), ``acc`` an FP32 value. A short last
    slice is zero-padded, as the masked loads of the kernels do.
    """
    a, b = [float(x) for x in a], [float(y) for y in b]
    for start in range(0, len(a), GROUP):
        acc = sm120_mma_group(a[start:start + GROUP], b[start:start + GROUP], acc, dtype=dtype)
    return float(acc)


# ======================================================================================
# Vectorized model (FP64 on any device)
# ======================================================================================
def _exponents(x: torch.Tensor, emin: int) -> torch.Tensor:
    """Field exponent of every element as FP64; zeros get -inf."""
    exponent = torch.frexp(x)[1].double() - 1.0
    return torch.where(x == 0, -math.inf, torch.clamp(exponent, min=float(emin)))


def _group_batched(a: torch.Tensor, b: torch.Tensor, acc: torch.Tensor, emin: int) -> torch.Tensor:
    """One instruction for every row: ``a``, ``b`` (N, <=16) FP64, ``acc`` (N,) FP64 -> (N,) FP64."""
    finite = torch.isfinite(a) & torch.isfinite(b)
    zero_lane = (a == 0) | (b == 0)
    nan = torch.isnan(acc) | (torch.isnan(a) | torch.isnan(b) | (~finite & zero_lane)).any(-1)
    inf_lane = ~finite & ~zero_lane & ~(torch.isnan(a) | torch.isnan(b))
    positive = (inf_lane & ((a > 0) == (b > 0))).any(-1) | (acc == math.inf)
    negative = (inf_lane & ((a > 0) != (b > 0))).any(-1) | (acc == -math.inf)
    nan = nan | (positive & negative)

    a = torch.where(finite, a, 0.0)
    b = torch.where(finite, b, 0.0)
    c = torch.where(torch.isfinite(acc), acc, 0.0)
    products = a * b  # exact: 22 (FP16) or 16 (BF16) significand bits, exponents within FP64
    e_products = torch.where(products == 0, -math.inf, _exponents(a, emin) + _exponents(b, emin))
    E = torch.maximum(e_products.amax(-1), _exponents(c, FP32_EMIN))
    live = torch.isfinite(E)
    E = torch.where(live, E, 0.0)
    unit = torch.ldexp(torch.ones_like(E), (E - 23 - GUARD).long())
    # Every truncated addend is an integer below 2^27 units: the sum of 17 is exact in FP64.
    total = torch.trunc(products / unit[:, None]).sum(-1) + torch.trunc(c / unit)
    bits = torch.frexp(total)[1].double()  # significant bits of the integer |total|
    shift = torch.clamp(torch.maximum(bits - 24, -124.0 - E), min=0.0)
    scale = torch.ldexp(torch.ones_like(shift), shift.long())
    value = torch.trunc(total / scale) * scale * unit
    value = torch.where(live & (total != 0), value, 0.0)
    value = torch.where(value.abs() >= 2.0 ** 128, torch.copysign(torch.full_like(value, math.inf), value), value)
    value = torch.where(positive, math.inf, torch.where(negative, -math.inf, value))
    return torch.where(nan, math.nan, value)


def sm120_mma_batched(A: torch.Tensor, B: torch.Tensor, acc: torch.Tensor | None = None, *,
                      dtype: str = "fp16") -> torch.Tensor:
    """``sm120_mma_accumulate`` for many dot products: rows of ``A`` and ``B`` (N, K) -> (N,) FP32."""
    A, B = A.double(), B.double()
    value = torch.zeros(A.shape[0], dtype=torch.float64, device=A.device) if acc is None else acc.double()
    for start in range(0, A.shape[-1], GROUP):
        value = _group_batched(A[:, start:start + GROUP], B[:, start:start + GROUP], value, EMIN[dtype])
    return value.float()


def sm120_predict_diag(row: torch.Tensor, *, dtype: str = "fp16", prefix_groups: int = 1 << 12) -> float:
    """``sm120_mma_accumulate(row, row)`` of one long row, vectorized over the binades of the sum.

    The products are squares, so the accumulator only grows. Once its exponent is at least the
    largest field exponent ``2 e(x)`` of the row, every instruction aligns to the accumulator: in
    one binade ``[2^E, 2^(E+1))`` a product keeps ``floor(p / 2^(E-25))`` units and a group adds
    ``floor(sum / 4)`` ulps of ``2^(E-23)``, one cumulative sum per binade. Until then, and for the
    group that crosses a binade, the exact rule runs directly.
    """
    x = row.detach().double().flatten()
    if x.numel() == 0 or not bool((x != 0).any()):
        return 0.0
    if not bool(torch.isfinite(x).all()):
        return sm120_mma_accumulate(x.tolist(), x.tolist(), dtype=dtype)
    products = torch.nn.functional.pad(x.square(), (0, -x.numel() % GROUP)).view(-1, GROUP)
    top = int(_exponents(x, EMIN[dtype]).max().item()) * 2
    head = x[:prefix_groups * GROUP].tolist()
    acc, start = 0.0, 0
    while start < len(products) and (acc == 0.0 or _field_exponent(acc, FP32_EMIN) < top):
        if start >= prefix_groups:  # a pathological row: exact rule for the rest
            tail = x[start * GROUP:].tolist()
            return sm120_mma_accumulate(tail, tail, acc, dtype=dtype)
        chunk = head[start * GROUP:(start + 1) * GROUP]
        acc, start = sm120_mma_group(chunk, chunk, acc, dtype=dtype), start + 1
    while start < len(products):
        exponent = _field_exponent(acc, FP32_EMIN)
        if exponent >= 127:
            return sm120_mma_accumulate(x[start * GROUP:].tolist(), x[start * GROUP:].tolist(), acc, dtype=dtype)
        unit = 2.0 ** (exponent - 23 - GUARD)
        kept = torch.floor(products[start:] / unit).sum(1)  # per group, in units (exact)
        running = acc / (unit * 2 ** GUARD) + torch.cumsum(torch.floor(kept / 2 ** GUARD), 0)
        crossed = torch.nonzero(running >= 2.0 ** 24)
        if crossed.numel() == 0:
            return running[-1].item() * unit * 2 ** GUARD
        index = int(crossed[0])
        before = running[index - 1].item() * unit * 2 ** GUARD if index else acc
        total = (before / unit + kept[index].item()) * unit  # exact sum of the crossing group
        ulp = 2.0 ** (math.frexp(total)[1] - 1 - 23)
        acc, start = math.floor(total / ulp) * ulp, start + index + 1
    return acc


# ======================================================================================
# Alternative hypotheses (each one is falsified by ``characterize``)
# ======================================================================================
HYPOTHESES = {
    "sm120 (field exponents, 2 guard bits, RZ, 16-wide)": {},
    "C4-03 (product binade, 2 guard bits)": {"exponent": "product"},
    "1 guard bit": {"guard": 1},
    "3 guard bits": {"guard": 3},
    "2 x 8 lane groups": {"lanes": 8},
    "4 x 4 lane groups": {"lanes": 4},
    "pairwise adder tree": {"tree": True},
    "accumulator added after the products": {"inject": "after"},
    "two's-complement truncation (floor)": {"term": "floor"},
    "final round to nearest even": {"final": "rn"},
}


def variant_accumulate(a, b, acc: float = 0.0, *, dtype: str = "fp16", exponent: str = "field",
                       guard: int = GUARD, lanes: int = GROUP, inject: str = "aligned", term: str = "rz",
                       final: str = "rz", tree: bool = False) -> float:
    """A family of k16 models around ``sm120_mma_accumulate`` (finite inputs only).

    ``exponent``: ``field`` (sum of input exponents) or ``product`` (binade of the product).
    ``lanes``: products per sub-adder; the sub-sums are normalized to FP32 one after the other.
    ``inject``: ``aligned`` (the accumulator is one more addend) or ``after`` (added to the
    normalized product sum). ``term`` / ``final``: ``rz``, ``floor`` or ``rn`` (nearest even).
    ``tree``: products summed by a binary tree of 2-input adders, each normalized to FP32, then
    the accumulator.
    """
    emin = EMIN[dtype]

    def cut(value: int, shift: int, mode: str) -> int:
        if shift <= 0:
            return value
        if mode == "rz":
            return _rz(value, shift)
        if mode == "floor":
            return value >> shift << shift
        low = abs(value) >> shift << shift
        rest = abs(value) - low
        if 2 * rest > 1 << shift or (2 * rest == 1 << shift and (low >> shift) & 1):
            low += 1 << shift
        return low * (1 if value >= 0 else -1)

    def add(values: list[tuple[int, int]]) -> int:  # (integer value, alignment exponent)
        values = [(v, e) for v, e in values if v != 0]
        if not values:
            return 0
        E = max(e for _, e in values)
        total = sum(cut(v, _U + E - 23 - guard, term) for v, _ in values)
        if total == 0:
            return 0
        return cut(total, max(abs(total).bit_length() - 24, _U - 149), final)

    def exp_of(value: int) -> int:
        return _field_exponent(value / (1 << _U), FP32_EMIN)

    value = _to_int(acc)
    for start in range(0, len(a), GROUP):
        products = []
        for x, y in zip(a[start:start + GROUP], b[start:start + GROUP]):
            if x != 0 and y != 0:
                p = _to_int(x) * _to_int(y) >> _U
                e = (_field_exponent(x, emin) + _field_exponent(y, emin) if exponent == "field"
                     else math.frexp(abs(x * y))[1] - 1)
                products.append((p, e))
        products += [(0, 0)] * (GROUP - len(products))
        if tree:
            while len(products) > 1:
                sums = [add(products[i:i + 2]) for i in range(0, len(products), 2)]
                products = [(v, exp_of(v) if v else 0) for v in sums]
            value = add([(value, exp_of(value) if value else 0), products[0]])
        elif inject == "after":
            partial = 0
            for s in range(0, GROUP, lanes):
                partial = add([(partial, exp_of(partial) if partial else 0), *products[s:s + lanes]])
            value = add([(value, exp_of(value) if value else 0), (partial, exp_of(partial) if partial else 0)])
        else:
            for s in range(0, GROUP, lanes):
                value = add([(value, exp_of(value) if value else 0), *products[s:s + lanes]])
    return value / (1 << _U)


def c4_03_accumulate(a, b, acc: float = 0.0) -> float:
    """The C4-03 rule (``bench_c4_03_wide_and_gram_drift.mma_accumulate``), for comparison."""
    from research.polar_opt.bench_c4_03_wide_and_gram_drift import mma_accumulate

    return mma_accumulate(list(a), list(b), acc)


# ======================================================================================
# Hardware probes
# ======================================================================================
@triton.jit
def _mma_probe_kernel(a_ptr, b_ptr, c_ptr, o_ptr, K, BK: tl.constexpr):
    """``D = C + A @ B`` for one (16, K) x (K, 16) tile per program, ``BK`` columns per ``tl.dot``."""
    pid = tl.program_id(0).to(tl.int64)
    rm = tl.arange(0, 16)
    rn = tl.arange(0, 16)
    rk = tl.arange(0, BK)
    a_base = a_ptr + pid * 16 * K
    b_base = b_ptr + pid * K * 16
    acc = tl.load(c_ptr + pid * 256 + rm[:, None] * 16 + rn[None, :])
    for k in range(0, K // BK):
        a = tl.load(a_base + rm[:, None] * K + k * BK + rk[None, :])
        b = tl.load(b_base + (k * BK + rk[:, None]) * 16 + rn[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(o_ptr + pid * 256 + rm[:, None] * 16 + rn[None, :], acc)


def _mma_ops(compiled) -> list[str]:
    import re

    return sorted(set(re.findall(r"mma\.[\w.]+", compiled.asm["ptx"])))


def probe_tiles(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, *, bk: int = 16,
                num_warps: int = 1) -> tuple[torch.Tensor, list[str]]:
    """Run ``_mma_probe_kernel``: A (P, 16, K), B (P, K, 16) FP16/BF16, C (P, 16, 16) FP32 -> (D, PTX MMA ops)."""
    A, B, C = A.cuda().contiguous(), B.cuda().contiguous(), C.cuda().float().contiguous()
    P, _, K = A.shape
    if K % bk:
        raise ValueError("K must be a multiple of bk")
    D = torch.empty_like(C)
    compiled = _mma_probe_kernel[(P,)](A, B, C, D, K, BK=bk, num_warps=num_warps)
    return D, _mma_ops(compiled)


def probe_dots(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor | None = None, *, dtype: str,
               bk: int = 16) -> tuple[torch.Tensor, list[str]]:
    """Hardware ``c + a . b`` for many independent rows ``a``, ``b`` (N, K): 16 rows per tile diagonal."""
    work = DTYPES[dtype]
    N, K = a.shape
    pad, tail = -N % 16, -K % bk  # zero rows fill the last tile, zero columns the last slice
    a = torch.nn.functional.pad(a.double(), (0, tail, 0, pad)).view(-1, 16, K + tail)
    b = torch.nn.functional.pad(b.double(), (0, tail, 0, pad)).view(-1, 16, K + tail)
    C = torch.zeros(a.shape[0], 16, 16, dtype=torch.float32)
    if c is not None:
        C.diagonal(dim1=-2, dim2=-1).copy_(torch.nn.functional.pad(c.float().cpu(), (0, pad)).view(-1, 16))
    D, ops = probe_tiles(a.to(work), b.to(work).mT, C, bk=bk, num_warps=1 if bk == 16 else 4)
    return D.diagonal(dim1=-2, dim2=-1).reshape(-1)[:N].cpu(), ops


def _syrk_grid(n: int, batch: int, block: int):
    tiles = -(-n // block)
    return tiles * (tiles + 1) // 2, batch


def config_record(config) -> dict:
    return {"BLOCK": config.kwargs["BLOCK"], "BK": config.kwargs["BK"], "num_warps": config.num_warps,
            "num_stages": config.num_stages}


def syrk_raw(view: torch.Tensor, config: dict, out_dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, list[str]]:
    """The production ``_syrk_kernel`` on a (batch, n, k) view under one explicit configuration.

    The JIT function is launched directly, so neither the autotuner nor a cached plan picks the
    configuration. Returns the stored Gram matrices and the MMA instructions of the PTX.
    """
    from research.polar_opt.kernels_blackwell import _syrk_kernel

    batch, n, k = view.shape
    out = torch.empty(batch, n, n, dtype=out_dtype, device=view.device)
    compiled = _syrk_kernel.fn[_syrk_grid(n, batch, config["BLOCK"])](
        view, out, n, k, *view.stride(), BLOCK=config["BLOCK"], BK=config["BK"],
        num_warps=config["num_warps"], num_stages=config["num_stages"])
    return out, _mma_ops(compiled)


# ======================================================================================
# Witnesses and input generators
# ======================================================================================
# audits/AUDIT_cycle4_c4_03_gpt62_astra.md, "Finite normal counterexample": FP16 row of length 16.
ASTRA_FP16_WITNESS = {6: 0.00162506103515625, 11: 0.19140625, 14: 0.005443572998046875,
                      15: 0.00036072731018066406}
ASTRA_FP16_HARDWARE = "0x1.2c63f4p-5"
ASTRA_FP16_C4_03 = "0x1.2c63f2p-5"
# The audit gives only the bits of its five-term BF16 diagonal (hardware 0x1.627972p-5, C4-03
# model 0x1.627970p-5). This row is a reconstruction with the same two results, found by search.
BF16_WITNESS = {8: 0.2080078125, 10: 0.000301361083984375, 11: 5.2928924560546875e-05,
                13: 0.0003223419189453125, 14: 0.0018310546875}
ASTRA_BF16_HARDWARE = "0x1.627972p-5"
ASTRA_BF16_C4_03 = "0x1.627970p-5"
# "Where exact agreement ends": BF16 off-diagonal inputs (products, count) and the hardware result.
ASTRA_EXPONENT_TABLE = (
    {"name": "2^-133 * 2^10", "a": 2.0 ** -133, "b": 2.0 ** 10, "count": 1, "hardware": 2.0 ** -123},
    {"name": "2^-70 * 2^-70", "a": 2.0 ** -70, "b": 2.0 ** -70, "count": 1, "hardware": 2.0 ** -140},
    {"name": "2^-80 * 2^-80", "a": 2.0 ** -80, "b": 2.0 ** -80, "count": 1, "hardware": 0.0},
    {"name": "256 x 2^-76 * 2^-77", "a": 2.0 ** -76, "b": 2.0 ** -77, "count": 256, "hardware": 2.0 ** -149},
    {"name": "2^70 * 2^70", "a": 2.0 ** 70, "b": 2.0 ** 70, "count": 1, "hardware": math.inf},
)
_INF, _NAN = math.inf, math.nan
# (name, dtype, a, b, acc): FP16/FP32 subnormals, overflow, cancellation, signed zero, specials.
EDGE_CASES = (
    ("fp16 subnormal products", "fp16", [2.0 ** -20] * 8, [3 * 2.0 ** -20] * 8, 0.0),
    ("fp16 subnormal x normal dominates", "fp16", [2.0 ** -20, 1.0], [1.0, 2.0 ** -22], 0.0),
    ("fp16 subnormal x normal + acc", "fp16", [2.0 ** -20] * 4, [2.0 ** -3] * 4, 2.0 ** -30),
    ("fp16 zero lane against 2^15", "fp16", [0.0] + [1.375 * 2.0 ** -14] * 15, [2.0 ** 15] + [1.3 * 2.0 ** -10] * 15, 0.0),
    ("fp16 largest product", "fp16", [65504.0] * 16, [65504.0] * 16, 0.0),
    ("fp32 subnormal acc + product", "bf16", [2.0 ** -70], [2.0 ** -70], 2.0 ** -145),
    ("fp32 subnormal acc alone", "bf16", [0.0], [0.0], 2.0 ** -145),
    ("fp32 negative subnormal acc", "bf16", [0.0], [0.0], -2.0 ** -149),
    ("fp32 overflow to inf (2^127 + 2^127)", "bf16", [2.0 ** 127, 2.0 ** 127], [1.0, 1.0], 0.0),
    ("fp32 overflow from acc", "bf16", [2.0 ** 127], [1.0], 3e38),
    ("fp32 negative overflow", "bf16", [-2.0 ** 127], [1.0], -3e38),
    ("fp32 max + 2^103 stays finite", "bf16", [2.0 ** 103], [1.0], 3.4028234663852886e38),
    ("2^140 - 2^140 + 1 (aligned away)", "bf16", [2.0 ** 70, -2.0 ** 70, 1.0], [2.0 ** 70, 2.0 ** 70, 1.0], 0.0),
    ("acc 1 + 16 x 2^-25", "fp16", [2.0 ** -12] * 16, [2.0 ** -13] * 16, 1.0),
    ("acc 1 + 16 x 2^-26 (dropped)", "fp16", [2.0 ** -12] * 16, [2.0 ** -14] * 16, 1.0),
    ("acc 1.5 + 16 x 2^-25", "bf16", [2.0 ** -12] * 16, [2.0 ** -13] * 16, 1.5),
    ("acc 1.5 - 16 x 2^-25", "bf16", [-2.0 ** -12] * 16, [2.0 ** -13] * 16, 1.5),
    ("acc 1 + 3 x 2^-25 (RZ)", "fp16", [2.0 ** -12] * 3, [2.0 ** -13] * 3, 1.0),
    ("acc 1 - 2^-25 (RZ below a binade)", "fp16", [-2.0 ** -12], [2.0 ** -13], 1.0),
    ("acc -1 + 1 = +0", "fp16", [1.0], [1.0], -1.0),
    ("acc -0, zero products = +0", "fp16", [0.0], [0.0], -0.0),
    ("acc -0, -0 product = +0", "fp16", [-0.0], [1.0], -0.0),
    ("inf * 0", "bf16", [_INF], [0.0], 0.0),
    ("inf * 1", "bf16", [_INF], [1.0], 0.0),
    ("inf - inf", "bf16", [_INF, _INF], [1.0, -1.0], 0.0),
    ("acc inf, -inf product", "bf16", [-_INF], [1.0], _INF),
    ("NaN operand", "fp16", [_NAN, 1.0], [1.0, 1.0], 0.0),
    ("NaN acc", "fp16", [1.0], [1.0], _NAN),
)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def normal_mantissa(shape, gen: torch.Generator, lo: int = -12, hi: int = -3, signed: bool = True) -> torch.Tensor:
    """FP64 values ``+-m 2^e``: ``m`` uniform in [1, 2), ``e`` uniform in [lo, hi] (the audit's inputs)."""
    m = 1.0 + torch.rand(shape, generator=gen, dtype=torch.float64)
    e = torch.randint(lo, hi + 1, shape, generator=gen).double()
    sign = torch.where(torch.rand(shape, generator=gen) < 0.5, -1.0, 1.0).double() if signed else 1.0
    return sign * m * torch.exp2(e)


def _bits(x: torch.Tensor) -> torch.Tensor:
    """FP32 bit patterns with every NaN mapped to ``NAN_BITS`` (the MMA's NaN)."""
    x = x.float().cpu()
    return torch.where(torch.isnan(x), torch.tensor(NAN_BITS, dtype=torch.int32), x.view(torch.int32))


def _hex(x: float) -> str:
    return float(x).hex() if math.isfinite(x) else repr(x)


def _rounded(x: float, dtype: str) -> float:
    return float(torch.tensor(x, dtype=torch.float64).to(DTYPES[dtype]))


# ======================================================================================
# Characterization (the canonical C5-01 MMA section)
# ======================================================================================
def _count(hardware: torch.Tensor, model: torch.Tensor, label: str, inputs=None, limit: int = 4) -> dict:
    """Bit-match count of two FP32 tensors and up to ``limit`` minimal failing witnesses."""
    hw, md = _bits(hardware).reshape(-1), _bits(model).reshape(-1)
    equal = hw == md
    record = {"category": label, "cases": int(equal.numel()), "bit_matches": int(equal.sum()),
              "hardware_sha256": hashlib.sha256(hw.numpy().tobytes()).hexdigest()}
    failing = torch.nonzero(~equal).flatten()[:limit].tolist()
    record["mismatches"] = [{"index": i, "hardware": _hex(hardware.reshape(-1)[i].item()),
                             "model": _hex(model.reshape(-1)[i].item()),
                             **(inputs(i) if inputs else {})} for i in failing]
    return record


def _witness(label: str, values: dict, dtype: str, hardware_hex: str, c4_hex: str) -> dict:
    row = [0.0] * GROUP
    for lane, v in values.items():
        row[lane] = _rounded(v, dtype)
    measured, ops = probe_dots(torch.tensor([row]), torch.tensor([row]), dtype=dtype)
    configs = []
    from research.polar_opt.kernels_blackwell import _CONFIGS

    view = torch.tensor([row], dtype=torch.float64).to(DTYPES[dtype]).cuda()[None]
    for config in _CONFIGS:
        out, _ = syrk_raw(view, config_record(config))
        configs.append(out[0, 0, 0].item())
    model = sm120_mma_accumulate(row, row, dtype=dtype)
    # Every ordered placement of the non-zero values in the 16 lanes (one diagonal entry each).
    lanes = torch.tensor(list(itertools.permutations(range(GROUP), len(values))))
    nonzero = torch.tensor([_rounded(v, dtype) for v in values.values()], dtype=torch.float64)
    rows = torch.zeros(len(lanes), GROUP, dtype=torch.float64).scatter_(1, lanes, nonzero.expand(len(lanes), -1))
    permuted, _ = probe_dots(rows, rows, dtype=dtype)
    return {"name": label, "dtype": dtype, "row": row, "ptx_mma": ops,
            "audit_hardware": hardware_hex, "audit_c4_03_model": c4_hex,
            "probe_hardware": _hex(measured[0].item()), "syrk_configs_hardware": [_hex(v) for v in configs],
            "sm120_model": _hex(model), "c4_03_model": _hex(c4_03_accumulate(row, row)),
            "model_equals_hardware": model == measured[0].item() == float.fromhex(hardware_hex),
            "configs_equal_hardware": all(v == float.fromhex(hardware_hex) for v in configs),
            "permutations": len(lanes),
            "permutations_equal_hardware": int((_bits(permuted) == _bits(torch.tensor([measured[0].item()]))).sum())}


def _exponent_table() -> list[dict]:
    rows = []
    for case in ASTRA_EXPONENT_TABLE:
        k = case["count"]
        a, b = torch.full((1, k), case["a"]), torch.full((1, k), case["b"])
        measured, _ = probe_dots(a, b, dtype="bf16")
        model = sm120_mma_accumulate(a[0].tolist(), b[0].tolist(), dtype="bf16")
        stored = _rounded(measured[0].item(), "bf16")
        rows.append({**{k2: v for k2, v in case.items() if k2 != "hardware"}, "audit_hardware": _hex(case["hardware"]),
                     "probe_hardware": _hex(measured[0].item()), "sm120_model": _hex(model),
                     "c4_03_model": _hex(c4_03_accumulate(a[0].tolist(), b[0].tolist())),
                     "fused_bf16_store": _hex(stored),
                     "model_equals_hardware": model == measured[0].item() == case["hardware"]})
    return rows


def _edge_cases() -> list[dict]:
    rows = []
    for name, dtype, a, b, acc in EDGE_CASES:
        a = [_rounded(x, dtype) for x in a]
        b = [_rounded(x, dtype) for x in b]
        measured, _ = probe_dots(torch.tensor([a]), torch.tensor([b]), torch.tensor([acc]), dtype=dtype)
        model = sm120_mma_accumulate(a, b, acc, dtype=dtype)
        vector = sm120_mma_batched(torch.tensor([a]), torch.tensor([b]), torch.tensor([acc]), dtype=dtype)
        hw_bits = int(_bits(measured)[0])
        rows.append({"name": name, "dtype": dtype, "acc": _hex(acc), "hardware": _hex(measured[0].item()),
                     "hardware_bits": f"0x{hw_bits & 0xFFFFFFFF:08x}", "sm120_model": _hex(model),
                     "model_equals_hardware": hw_bits == int(_bits(torch.tensor([model]))[0]),
                     "vectorized_equals_hardware": hw_bits == int(_bits(vector)[0])})
    return rows


def _stress_inputs(category: str, dtype: str, count: int, k: int, gen: torch.Generator):
    """(a, b, acc) FP64 tensors of ``count`` dot products of length ``k`` for one stress category."""
    work = DTYPES[dtype]
    shape = (count, k)
    if category == "random_mantissa":
        a, b = normal_mantissa(shape, gen), normal_mantissa(shape, gen)
        acc = torch.where(torch.rand(count, generator=gen) < 0.5, 0.0, normal_mantissa((count,), gen, -8, -2))
    elif category == "positive_diagonal":
        a = normal_mantissa(shape, gen, signed=False)
        b, acc = a, torch.zeros(count, dtype=torch.float64)
    elif category == "wide_exponent":
        lo, hi = (-26, 15) if dtype == "fp16" else (-140, 60)
        a, b = normal_mantissa(shape, gen, lo, hi), normal_mantissa(shape, gen, lo, hi)
        acc = torch.where(torch.rand(count, generator=gen) < 0.5, 0.0, normal_mantissa((count,), gen, lo, hi))
    elif category == "cancellation":
        a = normal_mantissa(shape, gen, -8, -2)
        b = normal_mantissa(shape, gen, -8, -2)
        half = k // 2
        a[:, half:2 * half] = -a[:, :half].to(work).double()  # exact negatives of the first half...
        b[:, half:2 * half] = (b[:, :half] * (1 + 2.0 ** -9 * torch.randint(-2, 3, (count, half), generator=gen)))
        acc = -(a[:, :1] * b[:, :1]).flatten() * torch.where(torch.rand(count, generator=gen) < 0.5, 0.0, 1.0)
    elif category == "subnormal":
        lo, hi = (-25, -13) if dtype == "fp16" else (-134, -120)
        a = normal_mantissa(shape, gen, lo, hi)
        b = torch.where(torch.rand(shape, generator=gen) < 0.5, normal_mantissa(shape, gen, lo, hi),
                        normal_mantissa(shape, gen, -4, 4))
        acc = torch.where(torch.rand(count, generator=gen) < 0.5, 0.0,
                          normal_mantissa((count,), gen, -150 if dtype == "bf16" else -40, -120 if dtype == "bf16" else -20))
    elif category == "overflow":
        lo, hi = (12, 15) if dtype == "fp16" else (55, 66)
        a, b = normal_mantissa(shape, gen, lo, hi), normal_mantissa(shape, gen, lo, hi)
        acc = torch.where(torch.rand(count, generator=gen) < 0.5, 0.0,
                          normal_mantissa((count,), gen, 125 if dtype == "bf16" else 25, 127 if dtype == "bf16" else 31))
    elif category == "specials":
        a, b = normal_mantissa(shape, gen, -6, 6), normal_mantissa(shape, gen, -6, 6)
        pick = torch.rand(shape, generator=gen)
        specials = torch.tensor([math.inf, -math.inf, math.nan, 0.0, -0.0], dtype=torch.float64)
        a = torch.where(pick < 0.08, specials[torch.randint(0, 5, shape, generator=gen)], a)
        pick = torch.rand(shape, generator=gen)
        b = torch.where(pick < 0.08, specials[torch.randint(0, 5, shape, generator=gen)], b)
        acc = torch.where(torch.rand(count, generator=gen) < 0.1, specials[torch.randint(0, 5, (count,), generator=gen)],
                          normal_mantissa((count,), gen, -4, 4))
    else:
        raise ValueError(category)
    a = a.to(work).double()
    b = a if category == "positive_diagonal" else b.to(work).double()
    return a, b, acc.float().double()


STRESS_CATEGORIES = ("random_mantissa", "positive_diagonal", "wide_exponent", "cancellation", "subnormal",
                     "overflow", "specials")


def _stress(protocol: dict) -> list[dict]:
    """Single-instruction categories and multi-instruction chains through ``_mma_probe_kernel``."""
    records = []
    for dtype in DTYPES:
        for c_index, category in enumerate(STRESS_CATEGORIES):
            gen = _generator(protocol["stress_seed"] + 100 * c_index + (dtype == "bf16"))
            a, b, acc = _stress_inputs(category, dtype, protocol["stress_dots"], GROUP, gen)
            measured, _ = probe_dots(a, b, acc, dtype=dtype)
            model = sm120_mma_batched(a.cuda(), b.cuda(), acc.cuda(), dtype=dtype).cpu()
            sample = protocol["scalar_crosscheck"]
            scalar = torch.tensor([sm120_mma_accumulate(a[i].tolist(), b[i].tolist(), acc[i].item(), dtype=dtype)
                                   for i in range(sample)])
            record = _count(measured, model, f"{category}/{dtype}",
                            lambda i, a=a, b=b, acc=acc: {"a": a[i].tolist(), "b": b[i].tolist(), "acc": acc[i].item()})
            record.update({"dtype": dtype, "k": GROUP, "bk": GROUP, "kind": "single",
                           "scalar_crosscheck": [int((_bits(scalar) == _bits(model[:sample])).sum()), sample]})
            records.append(record)
        for k, bk in protocol["chains"]:
            gen = _generator(protocol["stress_seed"] + 7 * k + bk + (dtype == "bf16"))
            dots = protocol["chain_dots"]
            a = normal_mantissa((dots, k), gen, -9, -3).to(DTYPES[dtype]).double()
            b = torch.where(torch.rand(dots, 1, generator=gen) < 0.25, a.abs(),
                            normal_mantissa((dots, k), gen, -9, -3).to(DTYPES[dtype]).double())
            measured, _ = probe_dots(a, b, dtype=dtype, bk=bk)
            model = sm120_mma_batched(a.cuda(), b.cuda(), dtype=dtype).cpu()
            record = _count(measured, model, f"chain_k{k}_bk{bk}/{dtype}")
            record.update({"dtype": dtype, "k": k, "bk": bk, "kind": "chain", "scalar_crosscheck": None})
            records.append(record)
    return records


def _syrk_configs(protocol: dict) -> list[dict]:
    """Every Gram entry of ``_syrk_kernel`` (raw FP32 and the fused store) under all 11 ``_CONFIGS``."""
    from research.polar_opt.kernels_blackwell import _CONFIGS

    records = []
    for dtype in DTYPES:
        for shape in protocol["syrk_shapes"]:
            gen = _generator(protocol["stress_seed"] + shape[0] * 131 + shape[1] + (dtype == "bf16"))
            n, k = shape
            X = (normal_mantissa((n, k), gen, -9, -1) * torch.where(torch.rand(n, k, generator=gen) < 0.1, 0.0, 1.0))
            X = X.to(DTYPES[dtype])
            i, j = torch.triu_indices(n, n)
            rows = X.double()
            model = sm120_mma_batched(rows[i].cuda(), rows[j].cuda(), dtype=dtype).cpu()
            for config in _CONFIGS:
                pin = config_record(config)
                raw, ops = syrk_raw(X.cuda()[None], pin)
                stored, _ = syrk_raw(X.cuda()[None], pin, DTYPES[dtype])
                record = _count(raw[0][i, j], model, f"syrk_{n}x{k}/{dtype}")
                fused = _bits(stored[0][i, j].float()) == _bits(model.to(DTYPES[dtype]).float())
                record.update({"dtype": dtype, "shape": shape, "config": pin, "ptx_mma": ops, "kind": "syrk",
                               "fused_store_matches": int(fused.sum()),
                               "symmetric": bool(torch.equal(raw[0].view(torch.int32), raw[0].mT.view(torch.int32)))})
                records.append(record)
    return records


def _audit_seeds(protocol: dict) -> list[dict]:
    """Fresh draws with the audit's seeds, shapes and distributions (its generator code is not recorded).

    seed 419: 9 x 2033 rows spread over powers of two, raw Gram (all configurations), and the same
    rows Frobenius-normalized. seed 927: 4,096 independent 2 x 16 matrices, raw off-diagonal and
    first diagonal. seed 719: 32,768 such matrices, the fused low-precision off-diagonal store.
    """
    from research.polar_opt.kernels_blackwell import _CONFIGS

    records = []
    for dtype in DTYPES:
        work = DTYPES[dtype]
        gen = _generator(419)
        X = torch.randn(9, 2033, generator=gen, dtype=torch.float64) * torch.exp2(
            torch.randint(-6, 7, (9, 2033), generator=gen).double())
        for label, matrix in (("seed419_9x2033", X), ("seed419_9x2033_unit_frobenius", X / X.norm())):
            matrix = matrix.to(work)
            i, j = torch.triu_indices(9, 9)
            model = sm120_mma_batched(matrix.double()[i].cuda(), matrix.double()[j].cuda(), dtype=dtype).cpu()
            for config in _CONFIGS:
                raw, _ = syrk_raw(matrix.cuda()[None], config_record(config))
                record = _count(raw[0][i, j], model, f"{label}/{dtype}")
                record.update({"dtype": dtype, "config": config_record(config), "kind": "audit_seed"})
                records.append(record)
        for seed, count in ((927, 4096), (719, 32768)):
            gen = _generator(seed)
            X = normal_mantissa((count, 2, GROUP), gen).to(work)
            rows = X.double()
            off = sm120_mma_batched(rows[:, 0].cuda(), rows[:, 1].cuda(), dtype=dtype).cpu()
            diag = sm120_mma_batched(rows[:, 0].cuda(), rows[:, 0].cuda(), dtype=dtype).cpu()
            for config in _CONFIGS:
                pin = config_record(config)
                raw, _ = syrk_raw(X.cuda(), pin)
                if seed == 927:
                    for part, hw, md in (("offdiag", raw[:, 0, 1], off), ("diag0", raw[:, 0, 0], diag)):
                        record = _count(hw, md, f"seed927_{count}x2x16_{part}/{dtype}")
                        record.update({"dtype": dtype, "config": pin, "kind": "audit_seed"})
                        records.append(record)
                else:
                    stored, _ = syrk_raw(X.cuda(), pin, work)
                    record = _count(raw[:, 0, 1], off, f"seed719_{count}x2x16_offdiag/{dtype}")
                    record["fused_store_matches"] = int((_bits(stored[:, 0, 1].float()) ==
                                                        _bits(off.to(work).float())).sum())
                    record.update({"dtype": dtype, "config": pin, "kind": "audit_seed"})
                    records.append(record)
    return records


def _hypotheses(protocol: dict) -> list[dict]:
    """Bit matches of each alternative model on the witnesses and on random normal-mantissa dots."""
    rows = []
    for dtype in DTYPES:
        gen = _generator(protocol["stress_seed"] + 99 + (dtype == "bf16"))
        a, b, acc = _stress_inputs("random_mantissa", dtype, protocol["hypothesis_dots"], GROUP, gen)
        measured, _ = probe_dots(a, b, acc, dtype=dtype)
        hw = _bits(measured)
        witness = ASTRA_FP16_WITNESS if dtype == "fp16" else BF16_WITNESS
        row = [0.0] * GROUP
        for lane, v in witness.items():
            row[lane] = _rounded(v, dtype)
        target = float.fromhex(ASTRA_FP16_HARDWARE if dtype == "fp16" else ASTRA_BF16_HARDWARE)
        for name, options in HYPOTHESES.items():
            predicted = torch.tensor([variant_accumulate(a[i].tolist(), b[i].tolist(), acc[i].item(), dtype=dtype,
                                                         **options) for i in range(len(a))])
            value = variant_accumulate(row, row, dtype=dtype, **options)
            rows.append({"hypothesis": name, "options": options, "dtype": dtype, "cases": len(a),
                         "bit_matches": int((_bits(predicted) == hw).sum()), "witness": _hex(value),
                         "witness_equals_hardware": value == target})
    return rows


def characterize(protocol: dict) -> dict:
    """Every MMA measurement of C5-01. Needs the RTX 5080 (or another sm_120 device)."""
    _, ops = probe_tiles(torch.zeros(1, 16, 16, dtype=torch.float16), torch.zeros(1, 16, 16, dtype=torch.float16),
                         torch.zeros(1, 16, 16))
    _, ops_bf16 = probe_tiles(torch.zeros(1, 16, 16, dtype=torch.bfloat16), torch.zeros(1, 16, 16, dtype=torch.bfloat16),
                              torch.zeros(1, 16, 16))
    return {
        "ptx_mma": {"fp16": ops, "bf16": ops_bf16},
        "model": {"name": "sm120_mma_accumulate", "group": GROUP, "guard_bits": GUARD,
                  "alignment": "max(e(C), e(a_k) + e(b_k)); e = field exponent (subnormal at emin)",
                  "addend_rounding": "toward zero at 2^(E-25)", "sum": "exact, one 17-term adder, lane-order free",
                  "normalization": "toward zero to 24 bits, FP32 subnormals at 2^-149, |x| >= 2^128 -> inf",
                  "zero": "+0", "nan_bits": f"0x{NAN_BITS:08x}",
                  "chain": "ascending k16 slices, FP32 D feeds the next C"},
        "witnesses": [_witness("Astra FP16 4-term diagonal", ASTRA_FP16_WITNESS, "fp16", ASTRA_FP16_HARDWARE,
                               ASTRA_FP16_C4_03),
                      _witness("BF16 5-term diagonal (reconstructed)", BF16_WITNESS, "bf16", ASTRA_BF16_HARDWARE,
                               ASTRA_BF16_C4_03)],
        "exponent_table": _exponent_table(),
        "edge_cases": _edge_cases(),
        "hypotheses": _hypotheses(protocol),
        "stress": _stress(protocol),
        "syrk_configs": _syrk_configs(protocol),
        "audit_seeds": _audit_seeds(protocol),
    }
