"""Schatten-8 pre-scaled, register-minimal Polar Switch circuits on Blackwell (RTX 5080, sm_120).

Each segment is the Polar Switch grammar of ``polar_switch_blackwell``:

    B[0] = X X^T                                    ``_syrk_kernel``
    B[dst] = a * B[src]^2 + b * B[add] + c * I      ``_symprod_kernel`` / ``_symprod_ptr_kernel``:
                                                    FP16/BF16 operands, FP32 accumulation, one rounding
    X <- B[last dst] @ X                            ``_apply`` (cuBLAS)

No new matrix multiplication kernels are defined here: ``_syrk_kernel``, ``_CONFIGS``,
``_tri_tile``, and ``_store_sym`` come from ``kernels_blackwell``; ``_launch``, ``_symprod_kernel``,
``_triangle_grid``, and ``_apply`` come from ``polar_switch_blackwell``.

The difference from ``PS3_55`` is the scale. ``PS3_55`` divides by the Frobenius norm, which is
about 3.2 times the top singular value of a Muon momentum. Here the first squaring of segment 1
is the plain ``R^2``, and the Schatten-8 bound ``lambda_max(R) <= ||R^2||_F^(1/2)`` is read from
it. The squaring is part of the circuit, so the bound costs no ``n^3`` work.

Scale handling (the Cycle 1 P0 findings & zero-sync CUDA-graph execution)
-------------------------------------------------------------------------
* Power-of-two prescale (Fable R1). For ``n >= 1024``, ``X_h = fl(64 * G / ||G||_F)``, so
  ``R_h = fl(X_h X_h^T)`` has trace 4096 and its off-diagonal entries are normal FP16 numbers
  also at ``n >= 4096``. ``R^2`` is stored as ``2^-11 R_h^2``. Powers of two change no
  mantissa. Below 1024 the prescale is 1 (one launch less); the residual term below covers the
  at most ``n 2^-25`` of subnormal rounding.
* Rigorous bound (Fable R1). The chain is a polynomial of the stored ``R_h``, and the bound is
  computed for that same matrix. It includes the rounding of the stored square and the cast
  residual ``||X_h X_h^T - R_h||_F <= u ||R_h||_F + n 2^-25``, so it also bounds
  ``sigma_max(X_h)^2``. See ``_upper``.
* Headroom (Fable H2). ``sigma_hat = u_safe * bound`` with ``u_safe = 1.05``, and segment 2
  gets the same factor on the output of segment 1. A circuit is used in BF16 only if the
  coherent rank-one sweeps pass in BF16 (``bf16_verified``).
* Normalization & zero-sync CUDA-graph path (Fable R3). All norm reductions, FP64 overflow/
  underflow-safe scaling, Schatten-8 ``kappa`` folding, and elementwise row-0 updates run in
  device-side Triton kernels with per-matrix coefficients stored in GPU memory. Zero device-to-host
  synchronizations occur, making ``schatten_switch_blackwell`` 100% CUDA-graph capturable while
  giving each batch member its own exact scale and returning exact zeros for dead, tiny
  (``||G||_F <= 1e-12``), or non-finite inputs.

Inputs have shape ``(..., rows, columns)``; tall, wide, square, odd, and ``min(rows, columns)
== 1`` are supported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor

from research.polar_opt.kernels_blackwell import _CONFIGS, _store_sym, _syrk_kernel, _tri_tile
from research.polar_opt.polar_switch_blackwell import (
    _apply,
    _launch,
    _symprod_kernel,
    _triangle_grid,
)

_WORK_DTYPES = (torch.float16, torch.bfloat16)
PRESCALE = 64.0  # ||X_h||_F for n >= PRESCALE_MIN_N; trace(R_h) = 4096
PRESCALE_MIN_N = 1024
FRO_FLOOR = 1e-12  # inputs at or below this Frobenius norm return zeros
_SLACK = 1.0 + 2.0 ** -12  # FP32 accumulation and FP32 norm reductions
_REDUCE_BLOCKS = 64
# dtype -> (unit roundoff, largest absolute rounding error of a subnormal entry)
_ROUNDING = {torch.float16: (2.0 ** -11, 2.0 ** -25), torch.bfloat16: (2.0 ** -8, 0.0),
             torch.float64: (0.0, 0.0)}
_TABLE_CACHE: dict[tuple, tuple[Tensor, Tensor]] = {}


# --------------------------------------------------------------------------------------
# Circuits
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SchattenSwitchCircuit:
    """Segments ``(residual, rows)`` with rows ``(dst, src, add, a, b, c)``.

    The table is written for ``x = sigma / sigma_hat``: segment 1 acts on ``x^2``. Its first
    row is ``(1, 0, 0, 1.0, b, c)``: buffer 1 holds ``R^2 + b R + c I``. ``b`` and ``c`` cost
    one elementwise pass each, because ``R^2`` must exist before the scale is known.

    ``lo`` and ``error`` are the design record: ``|f - 1| <= error`` for ``x`` in
    ``[lo, 1 / u_safe]``, and ``0 <= f <= 1 + error`` on ``[0, 1]`` in exact arithmetic.
    """

    name: str
    table: tuple
    buffers: int
    u_safe: float
    lo: float
    error: float
    bf16_verified: bool = False

    def __post_init__(self):
        first = self.table[0][1]
        if len(first) < 2 or tuple(first[0][:4]) != (1, 0, 0, 1.0):
            raise ValueError("Segment 1 must start with the row (1, 0, 0, 1.0, b, c) and have two rows")
        for _, rows in self.table:
            for dst, src, add, *_ in rows:
                if dst == src or max(dst, src, add) >= self.buffers:
                    raise ValueError(f"Invalid register row {(dst, src, add)} for {self.buffers} buffers")

    @property
    def squarings(self) -> tuple[int, ...]:
        return tuple(len(rows) for _, rows in self.table)

    @property
    def elementwise_passes(self) -> int:
        """Extra ``n^2`` passes that form the first row (0 for a pure square)."""
        return sum(bool(v) for v in self.table[0][1][0][4:6])

    def flops_formula(self) -> str:
        return f"({3 * len(self.table)}α + {sum(self.squarings)})n³"

    def scalar(self, x):
        """Exact-arithmetic map of ``x = sigma / sigma_hat`` (float64 NumPy or Python values)."""
        for residual, rows in self.table:
            work = {0: x * x}
            for dst, src, add, a, b, c in rows:
                work[dst] = a * work[src] ** 2 + b * work[add] + c
            x = x * work[dst] + residual * x
        return x

    def to_dict(self) -> dict:
        return {"name": self.name, "buffers": self.buffers, "u_safe": self.u_safe, "lo": self.lo,
                "error": self.error, "bf16_verified": self.bf16_verified,
                "squarings": list(self.squarings), "flops": self.flops_formula(),
                "table": [[residual, [list(row) for row in rows]] for residual, rows in self.table]}

    @classmethod
    def from_dict(cls, d: dict) -> "SchattenSwitchCircuit":
        table = tuple((float(residual), tuple((int(r[0]), int(r[1]), int(r[2]), *map(float, r[3:]))
                                              for r in rows)) for residual, rows in d["table"])
        return cls(d["name"], table, int(d["buffers"]), float(d["u_safe"]), float(d["lo"]),
                   float(d["error"]), bool(d.get("bf16_verified", False)))


def theoretical_flops(variant, shape) -> dict:
    """Leading FLOPs: SYRK and squares count one triangle (``n^3``), ``Q @ X`` is dense."""
    circuit = _lookup(variant)
    n, m = sorted(shape[-2:])
    batch = math.prod(shape[:-2])
    segments, squares = len(circuit.table), sum(circuit.squarings)
    return {"symmetric_flops": batch * (3 * segments * m * n * n + squares * n ** 3),
            "formula": circuit.flops_formula(), "squarings": squares,
            "matrix_kernel_launches": 2 * segments + squares}


# Discovered circuits: copied verbatim from results/schatten_switch_search_cycle2.json.
_CIRCUIT_SPECS: dict = {
    "SS3_54": {
        "name": "SS3_54",
        "buffers": 3,
        "u_safe": 1.05,
        "lo": 0.00286,
        "error": 0.07982588482306929,
        "bf16_verified": False,
        "squarings": [5, 4],
        "flops": "(6α + 9)n³",
        "table": [
            [0.0, [
                [1, 0, 0, 1.0, -1.035678364265212, 0.0],
                [2, 1, 1, -44.95361416143577, -13.51553474507387, -0.015878595332442222],
                [1, 2, 2, -4.569196296690892, 5.092643625703066, -0.9179839421935934],
                [2, 1, 2, 1.4182468064291267, 0.32580136819175276, -0.4130735383448542],
                [0, 2, 0, 44.21037324366673, -1.0366534520808317, 1.1258551001231605],
            ]],
            [0.0, [
                [1, 0, 0, 1.0, -1.2041354084316795, 0.3052004453155656],
                [0, 1, 0, -11.240335609969362, 0.4504520793471352, 0.6056874882617449],
                [1, 0, 0, -1.3154792860430617, 1.3183888243288422, -0.16196259793937848],
                [0, 1, 0, 7.874247039551604, -2.8721964431448384, 3.7453983723791087],
            ]],
        ],
    },
    "SS3_53": {
        "name": "SS3_53",
        "buffers": 3,
        "u_safe": 1.05,
        "lo": 0.003,
        "error": 0.08754726536476465,
        "bf16_verified": False,
        "squarings": [5, 3],
        "flops": "(6α + 8)n³",
        "table": [
            [0.0, [
                [1, 0, 0, 1.0, -1.032065172692518, 0.001612489358901909],
                [2, 1, 1, -44.83263019846355, -13.422258161051742, -0.004608695563609776],
                [1, 2, 2, -4.342606921772448, 4.829840332286762, -0.8696248777205423],
                [2, 1, 2, 1.3884720693911183, 0.30675032509703015, -0.3803835190493494],
                [0, 2, 0, 42.34652174703252, -1.3667245866292133, 1.489476103920553],
            ]],
            [0.0, [
                [1, 0, 0, 1.0, -1.3619545837223892, 0.36463278427406026],
                [2, 1, 0, -10.626973938205918, -0.8699357489607156, 0.4129312898142153],
                [0, 2, 0, 6.067460139311178, -3.0603231352476334, 2.802011219993987],
            ]],
        ],
    },
    "SS3_34": {
        "name": "SS3_34",
        "buffers": 3,
        "u_safe": 1.05,
        "lo": 0.008,
        "error": 0.06731792066746589,
        "bf16_verified": True,
        "squarings": [3, 4],
        "flops": "(6α + 7)n³",
        "table": [
            [0.0, [
                [1, 0, 0, 1.0, -1.0847316915124348, 0.0],
                [2, 1, 1, 40.259825481661466, 14.228945413711514, 1.0],
                [0, 2, 0, 12.154614532184107, -0.24753624441972114, 0.34411502761971247],
            ]],
            [0.0, [
                [1, 0, 0, 1.0, -1.2071156728892756, 0.3003555159291965],
                [0, 1, 0, -11.982584290661986, 0.4175081099000911, 0.6308126244781913],
                [1, 0, 0, -1.3476549024766409, 1.334861767996618, -0.12595987968659186],
                [0, 1, 0, 7.423423319413903, -2.8900647896347476, 3.760051860736799],
            ]],
        ],
    },
    "SS3_35": {
        "name": "SS3_35",
        "buffers": 3,
        "u_safe": 1.05,
        "lo": 0.004,
        "error": 0.083091071092388,
        "bf16_verified": True,
        "squarings": [3, 5],
        "flops": "(6α + 8)n³",
        "table": [
            [0.0, [
                [1, 0, 0, 1.0, -1.0932562497193754, 0.0],
                [2, 1, 1, 39.96168457119626, 14.266031068292616, 0.9999999999999999],
                [0, 2, 0, 12.502107468117961, -0.11270651203942335, 0.17461401290128586],
            ]],
            [0.0, [
                [1, 0, 0, 1.0, -1.094787000847377, 0.006046824763652966],
                [2, 1, 1, 32.52847526518725, 11.903537182257438, 0.926832022503135],
                [0, 2, 0, 1.1443577301804355, -0.4951256590911481, -0.14435773018043516],
                [1, 0, 1, 1.2879235989542381, -0.6561721992471682, -0.28395584065060986],
                [0, 1, 0, 15.770356077625026, 6.036080816655014, 3.5066207509983927],
            ]],
        ],
    },
}

SCHATTEN_SWITCH_CIRCUITS: dict[str, SchattenSwitchCircuit] = {
    name: SchattenSwitchCircuit.from_dict(spec) for name, spec in _CIRCUIT_SPECS.items()}
DEFAULT_VARIANT = next(iter(SCHATTEN_SWITCH_CIRCUITS), None)


def _lookup(variant) -> SchattenSwitchCircuit:
    if isinstance(variant, SchattenSwitchCircuit):
        return variant
    key = str(variant).upper().replace("-", "_")
    if key not in SCHATTEN_SWITCH_CIRCUITS:
        raise ValueError(f"Unsupported Schatten Switch variant {variant!r}; "
                         f"available: {sorted(SCHATTEN_SWITCH_CIRCUITS)}")
    return SCHATTEN_SWITCH_CIRCUITS[key]


# --------------------------------------------------------------------------------------
# Host arithmetic shared by the native path and the reference
# --------------------------------------------------------------------------------------
def _prescale(n: int) -> tuple[float, float]:
    """``(P, alpha)``: ``||X_h||_F = P`` and the stored square is ``alpha * R_h^2 <= 2^13``.

    The factor 8 of FP16 range that remains is for the first row ``R^2 + b R + c I``.
    """
    P = PRESCALE if n >= PRESCALE_MIN_N else 1.0
    return P, 2.0 ** 13 / P ** 4


def _trace_bound(n: int, dtype: torch.dtype) -> float:
    """``lambda_max(X_h X_h^T) <= ||X_h||_F^2`` for ``X_h = fl(P * G / ||G||_F)``."""
    return _prescale(n)[0] ** 2 * (1.0 + _ROUNDING[dtype][0]) ** 2 * _SLACK


def _upper(norm_r: float, norm_s: float, n: int, dtype: torch.dtype) -> float:
    """Upper bound on ``lambda_max(R_h)`` and on ``lambda_max(X_h X_h^T)``.

    ``norm_r = ||R_h||_F`` and ``norm_s = ||S||_F`` with ``S = fl(alpha * R_h^2)``, both for the
    stored half-precision buffers. With unit roundoff ``u`` and subnormal error ``e`` per entry:

        ||alpha R_h^2||_F <= (1 + u) ||S||_F + n e          (rounding of the stored square)
        lambda_max(R_h)   <= ||R_h^2||_F^(1/2)              (Schatten-8)
        ||X_h X_h^T - R_h||_F <= u ||R_h||_F + n e          (cast residual of the Gram matrix)

    ``_SLACK`` covers the FP32 accumulation of the products and of the two norms. The trace
    bound ``||X_h||_F^2`` is taken when it is smaller.
    """
    u, e = _ROUNDING[dtype]
    schatten8 = math.sqrt(((1.0 + u) * norm_s * _SLACK + n * e) / _prescale(n)[1])
    residual = u * norm_r * _SLACK + n * e
    return min(schatten8 + residual, _trace_bound(n, dtype))


def _frobenius_kappa(circuit, n: int, dtype: torch.dtype) -> float:
    return 1.0 / (circuit.u_safe ** 2 * _trace_bound(n, dtype))


def _explicit_factors(circuit, sigma_hat, count: int, n: int, dtype: torch.dtype) -> list[float]:
    """Operand scale for an explicit ``sigma_hat``: ``||X_h||_F = P / factor``.

    The coefficients stay at the Frobenius scale and the operand shrinks, which is the
    arithmetic of a real input whose bound is far above this singular value.
    """
    values = sigma_hat.reshape(-1).tolist() if isinstance(sigma_hat, Tensor) else [float(sigma_hat)]
    values = values * count if len(values) == 1 else values
    if len(values) != count or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("sigma_hat needs one positive finite value per matrix")
    scale = _prescale(n)[0] * math.sqrt(_frobenius_kappa(circuit, n, dtype))
    return [scale * v for v in values]


def _kappas(circuit, scaling, explicit, norms_r, norms_s, n, dtype) -> list[float]:
    """Per matrix ``kappa`` with ``x^2 = kappa * lambda(R_h)``. A zero matrix gets a finite value."""
    count = len(norms_r)
    if explicit or scaling == "frobenius":
        return [_frobenius_kappa(circuit, n, dtype)] * count
    out = []
    for norm_r, norm_s in zip(norms_r, norms_s):
        upper = _upper(norm_r, norm_s, n, dtype) if norm_s > 0 else _trace_bound(n, dtype)
        out.append(1.0 / (circuit.u_safe ** 2 * upper))
    return out


def _fold(rows, kappa: float, alpha: float) -> tuple[float, float, list]:
    """Stored coefficients of segment 1 for one scale.

    Buffer 0 holds ``R_h = x^2 / kappa`` and buffer 1 holds ``t1 * (x^4 + b x^2 + c)`` with
    ``t1 = alpha / kappa^2``. Later buffers hold their table value, and the last one is
    multiplied by ``sqrt(kappa)`` so that ``Q @ X_h`` has the singular values ``f(x)``.
    Returns the two elementwise coefficients of the first row and the remaining rows.
    """
    b0, c0 = rows[0][4], rows[0][5]
    factor = {0: 1.0 / kappa, 1: alpha / kappa ** 2}
    folded = []
    for j, (dst, src, add, a, b, c) in enumerate(rows[1:], 1):
        t = math.sqrt(kappa) if j == len(rows) - 1 else 1.0
        folded.append((dst, src, add, a * t / factor[src] ** 2, b * t / factor[add], c * t))
        factor[dst] = t
    return alpha * b0 / kappa, alpha * c0 / kappa ** 2, folded


def _check(circuit: SchattenSwitchCircuit, dtype, scaling, allow_unsafe_bf16):
    if scaling not in ("schatten8", "frobenius"):
        raise ValueError("scaling must be 'schatten8' or 'frobenius'")
    if dtype == torch.bfloat16 and not circuit.bf16_verified and not allow_unsafe_bf16:
        raise ValueError(
            f"Circuit {circuit.name!r} is not verified on coherent rank-1 BF16 sweeps; use "
            f"dtype=torch.float16, pass allow_unsafe_bf16=True, or choose from "
            f"{sorted(k for k, c in SCHATTEN_SWITCH_CIRCUITS.items() if c.bf16_verified)}.")


# --------------------------------------------------------------------------------------
# Native path (sync-free, CUDA-graph capturable)
# --------------------------------------------------------------------------------------
def _seg1_table_tensors(circuit: SchattenSwitchCircuit, device: torch.device) -> tuple[Tensor, Tensor]:
    """Cache segment-1 coefficients (float64) and register triples (int32) on ``device``."""
    if circuit.buffers > 3:
        raise ValueError(f"Sync-free kernel supports at most 3 buffers, got {circuit.buffers}")
    seg1_rows = circuit.table[0][1]
    key = (circuit.name, seg1_rows, device)
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    coeffs = torch.tensor([[row[3], row[4], row[5]] for row in seg1_rows], dtype=torch.float64, device=device)
    regs = torch.tensor([[row[0], row[1], row[2]] for row in seg1_rows], dtype=torch.int32, device=device)
    _TABLE_CACHE[key] = (coeffs, regs)
    return coeffs, regs


@triton.autotune(configs=[triton.Config({"BLOCK": 4096}, num_warps=4)], key=[])
@triton.jit
def _safe_norm_part_kernel(x_ptr, m_ptr, s_ptr, nf_ptr, size, num_blocks, BLOCK: tl.constexpr):
    """LAPACK/Blue-style overflow/underflow-safe partial Frobenius norm reduction per matrix."""
    pid = tl.program_id(0)
    matrix = tl.program_id(1).to(tl.int64)
    stride_i32 = num_blocks * BLOCK
    stride = stride_i32.to(tl.int64)
    n_steps = tl.cdiv(size, stride_i32)

    m_cur = tl.full((), 0.0, dtype=tl.float64)
    s_cur = tl.full((), 0.0, dtype=tl.float64)
    nf_cur = tl.full((), 0, dtype=tl.int32)
    x_base = x_ptr + matrix * size

    for step in range(0, n_steps):
        cur_offs = step.to(tl.int64) * stride + pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = cur_offs < size
        v = tl.load(x_base + cur_offs, mask=mask, other=0.0).to(tl.float32)
        av = tl.abs(v)
        is_nf = (v != v) | (av > 3.4028234663852886e38)
        nf_cur = nf_cur | (tl.sum(is_nf.to(tl.int32), axis=0) > 0).to(tl.int32)
        av_clean = tl.where(is_nf, 0.0, av)
        m_chunk = tl.max(av_clean, axis=0)
        has_val = m_chunk > 1e-30
        inv_m = tl.where(has_val, 1.0 / tl.where(has_val, m_chunk, 1.0), 0.0)
        ratio = av_clean * inv_m
        s_chunk = tl.where(has_val, tl.sum(ratio * ratio, axis=0), 0.0)

        mc = tl.where(has_val, m_chunk, 0.0).to(tl.float64)
        sc = s_chunk.to(tl.float64)
        m_new = tl.maximum(m_cur, mc)
        m_safe = tl.where(m_new > 0.0, m_new, 1.0)
        r_old = m_cur / m_safe
        r_new = mc / m_safe
        s_cur = s_cur * (r_old * r_old) + sc * (r_new * r_new)
        m_cur = m_new

    out_idx = matrix * 64 + pid.to(tl.int64)
    tl.store(m_ptr + out_idx, m_cur)
    tl.store(s_ptr + out_idx, s_cur)
    tl.store(nf_ptr + out_idx, nf_cur)


@triton.autotune(configs=[triton.Config({"BLOCK": 4096}, num_warps=4)], key=[])
@triton.jit
def _normalize_safe_kernel(
    x_ptr, m_ptr, s_ptr, nf_ptr, factors_ptr, o_ptr,
    size, num_blocks, has_factors, prescale, fro_floor,
    BLOCK: tl.constexpr,
):
    """Reduce partial scaled norms in FP64 and normalize to target operand norm (zeros if invalid)."""
    matrix = tl.program_id(1).to(tl.int64)
    local = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    idx = tl.arange(0, 64)
    bmask = idx < num_blocks

    m_vec = tl.load(m_ptr + matrix * 64 + idx, mask=bmask, other=0.0)
    s_vec = tl.load(s_ptr + matrix * 64 + idx, mask=bmask, other=0.0)
    nf_vec = tl.load(nf_ptr + matrix * 64 + idx, mask=bmask, other=0)

    any_nf = tl.sum(nf_vec, axis=0) > 0
    M = tl.max(m_vec, axis=0)
    M_safe = tl.where(M > 0.0, M, 1.0)
    ratios = m_vec / M_safe
    ssq = tl.sum(s_vec * (ratios * ratios), axis=0)
    sqrt_ssq = tl.sqrt(ssq)
    fro = M * sqrt_ssq
    valid = (~any_nf) & (M > 0.0) & (fro == fro) & (fro < 1e308) & (fro > fro_floor)

    prescale64 = tl.full((), 0.0, dtype=tl.float64) + prescale
    factor = tl.where(has_factors != 0, tl.load(factors_ptr + matrix).to(tl.float64), 1.0)
    eff_prescale = prescale64 / factor
    unit_scale = tl.where(valid, eff_prescale / tl.where(sqrt_ssq > 0.0, sqrt_ssq, 1.0), 0.0).to(tl.float32)
    M_f32 = M_safe.to(tl.float32)

    mask = local < size
    x = tl.load(x_ptr + matrix * size + local, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(valid, (x / M_f32) * unit_scale, 0.0)
    tl.store(o_ptr + matrix * size + local, y.to(o_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=[triton.Config({"BLOCK": 4096}, num_warps=4)], key=[])
@triton.jit
def _dual_norm_part_kernel(r_ptr, s_ptr, part_r_ptr, part_s_ptr, nn, num_blocks, BLOCK: tl.constexpr):
    """Single-pass partial Frobenius-squared reduction of symmetric R_h and S in FP64."""
    pid = tl.program_id(0)
    matrix = tl.program_id(1).to(tl.int64)
    stride_i32 = num_blocks * BLOCK
    stride = stride_i32.to(tl.int64)
    n_steps = tl.cdiv(nn, stride_i32)

    acc_r = tl.full((), 0.0, dtype=tl.float64)
    acc_s = tl.full((), 0.0, dtype=tl.float64)
    r_base = r_ptr + matrix * nn
    s_base = s_ptr + matrix * nn

    for step in range(0, n_steps):
        cur_offs = step.to(tl.int64) * stride + pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = cur_offs < nn
        r = tl.load(r_base + cur_offs, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(s_base + cur_offs, mask=mask, other=0.0).to(tl.float32)
        acc_r += tl.sum(r * r, axis=0).to(tl.float64)
        acc_s += tl.sum(s * s, axis=0).to(tl.float64)

    out_idx = matrix * 64 + pid.to(tl.int64)
    tl.store(part_r_ptr + out_idx, acc_r)
    tl.store(part_s_ptr + out_idx, acc_s)


@triton.autotune(configs=[triton.Config({"BLOCK": 4096}, num_warps=4)], key=[])
@triton.jit
def _fold_and_row0_axpy_kernel(
    r_ptr, s_ptr, part_r_ptr, part_s_ptr, tbl_ptr, regs_ptr, out_coeffs_ptr,
    n, nn, batch, num_blocks, num_rows, use_frobenius, has_b0, has_c0,
    u, e, prescale, alpha, u_safe, slack, residual,
    BLOCK: tl.constexpr,
):
    """Compute per-matrix Schatten-8 kappa, fold Segment-1 table, and fuse row-0 update into S."""
    pid = tl.program_id(0)
    matrix = tl.program_id(1).to(tl.int64)
    idx = tl.arange(0, 64)
    bmask = idx < num_blocks

    sum_r = tl.sum(tl.load(part_r_ptr + matrix * 64 + idx, mask=bmask, other=0.0), axis=0)
    sum_s = tl.sum(tl.load(part_s_ptr + matrix * 64 + idx, mask=bmask, other=0.0), axis=0)
    norm_r = tl.sqrt(sum_r)
    norm_s = tl.sqrt(sum_s)
    ok = (norm_r == norm_r) & (norm_r < 1e308) & (norm_s == norm_s) & (norm_s < 1e308)

    zero64 = tl.full((), 0.0, dtype=tl.float64)
    u64 = zero64 + u
    e64 = zero64 + e
    prescale64 = zero64 + prescale
    alpha64 = zero64 + alpha
    u_safe64 = zero64 + u_safe
    slack64 = zero64 + slack
    res64 = zero64 + residual

    trace_bound = prescale64 * prescale64 * (1.0 + u64) * (1.0 + u64) * slack64
    schatten8 = tl.sqrt(((1.0 + u64) * norm_s * slack64 + n * e64) / alpha64)
    res_bound = u64 * norm_r * slack64 + n * e64
    schatten_upper = tl.where(norm_s > 0.0, tl.minimum(schatten8 + res_bound, trace_bound), trace_bound)
    upper = tl.where(use_frobenius != 0, trace_bound, schatten_upper)

    kappa = tl.where(ok, 1.0 / (u_safe64 * u_safe64 * upper), 0.0)
    k_safe = tl.where(ok, kappa, 1.0)
    sqrt_k = tl.sqrt(k_safe)

    b0_raw = tl.load(tbl_ptr + 1)
    c0_raw = tl.load(tbl_ptr + 2)
    b0_t = tl.where(ok, alpha64 * b0_raw / k_safe, 0.0).to(tl.float32)
    c0_t = tl.where(ok, alpha64 * c0_raw / (k_safe * k_safe), 0.0).to(tl.float32)
    ok_f32 = tl.where(ok, 1.0, 0.0).to(tl.float32)

    if pid == 0:
        tl.store(out_coeffs_ptr + matrix * 4 + 0, tl.where(ok, res64 * sqrt_k, 0.0).to(tl.float32))
        tl.store(out_coeffs_ptr + matrix * 4 + 1, tl.full((), 0.0, dtype=tl.float32))
        tl.store(out_coeffs_ptr + matrix * 4 + 2, tl.full((), 0.0, dtype=tl.float32))
        tl.store(out_coeffs_ptr + matrix * 4 + 3, ok_f32)

        f0 = 1.0 / k_safe
        f1 = alpha64 / (k_safe * k_safe)
        f2 = tl.full((), 1.0, dtype=tl.float64)
        for j in range(1, num_rows):
            dst = tl.load(regs_ptr + j * 3 + 0)
            src = tl.load(regs_ptr + j * 3 + 1)
            add = tl.load(regs_ptr + j * 3 + 2)
            a = tl.load(tbl_ptr + j * 3 + 0)
            b = tl.load(tbl_ptr + j * 3 + 1)
            c = tl.load(tbl_ptr + j * 3 + 2)
            f_src = tl.where(src == 0, f0, tl.where(src == 1, f1, f2))
            f_add = tl.where(add == 0, f0, tl.where(add == 1, f1, f2))
            t = tl.where(j == num_rows - 1, sqrt_k, 1.0)
            fa = tl.where(ok, a * t / (f_src * f_src), 0.0).to(tl.float32)
            fb = tl.where(ok, b * t / f_add, 0.0).to(tl.float32)
            fc = tl.where(ok, c * t, 0.0).to(tl.float32)
            base_c = (j.to(tl.int64) * batch + matrix) * 4
            tl.store(out_coeffs_ptr + base_c + 0, fa)
            tl.store(out_coeffs_ptr + base_c + 1, fb)
            tl.store(out_coeffs_ptr + base_c + 2, fc)
            tl.store(out_coeffs_ptr + base_c + 3, ok_f32)
            f0 = tl.where(dst == 0, t, f0)
            f1 = tl.where(dst == 1, t, f1)
            f2 = tl.where(dst == 2, t, f2)

    if (has_b0 != 0) | (has_c0 != 0):
        local = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = local < nn
        s_val = tl.load(s_ptr + matrix * nn + local, mask=mask, other=0.0).to(tl.float32)
        if has_b0 != 0:
            r_val = tl.load(r_ptr + matrix * nn + local, mask=mask, other=0.0).to(tl.float32)
            s_val = s_val + b0_t * r_val
        if has_c0 != 0:
            row = local // n
            col = local - row * n
            s_val = s_val + tl.where(row == col, c0_t, 0.0)
        s_val = tl.where(ok, s_val, 0.0)
        tl.store(s_ptr + matrix * nn + local, s_val.to(s_ptr.dtype.element_ty), mask=mask)


# restore_value: the output can alias C, so autotune trials must not accumulate into it.
@triton.autotune(configs=_CONFIGS, key=["N"], restore_value=["o_ptr"])
@triton.jit
def _symprod_ptr_kernel(
    a_ptr, b_ptr, c_ptr, o_ptr, coeff_ptr, N,
    BLOCK: tl.constexpr, BK: tl.constexpr,
):
    """Per-matrix device-coefficient variant of ``_symprod_kernel``."""
    bidx = tl.program_id(1).to(tl.int64)
    offset = bidx * N * N
    i, j = _tri_tile(tl.program_id(0), tl.cdiv(N, BLOCK))
    rm = i * BLOCK + tl.arange(0, BLOCK)
    rn = j * BLOCK + tl.arange(0, BLOCK)
    rk = tl.arange(0, BK)
    l_ptrs = a_ptr + offset + rm[:, None].to(tl.int64) * N + rk[None, :]
    r_ptrs = b_ptr + offset + rk[:, None] * N + rn[None, :].to(tl.int64)
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BK)):
        kmask = rk < N - k * BK
        left = tl.load(l_ptrs, mask=(rm[:, None] < N) & kmask[None, :], other=0.0)
        right = tl.load(r_ptrs, mask=kmask[:, None] & (rn[None, :] < N), other=0.0)
        acc = tl.dot(left, right, acc)
        l_ptrs += BK
        r_ptrs += BK * N
    tile = (rm[:, None] < N) & (rn[None, :] < N)
    c = tl.load(c_ptr + offset + rm[:, None].to(tl.int64) * N + rn[None, :], mask=tile, other=0.0)
    alpha = tl.load(coeff_ptr + bidx * 4 + 0)
    beta = tl.load(coeff_ptr + bidx * 4 + 1)
    gamma = tl.load(coeff_ptr + bidx * 4 + 2)
    ok = tl.load(coeff_ptr + bidx * 4 + 3)
    raw = alpha * acc + beta * c.to(tl.float32) + gamma * (rm[:, None] == rn[None, :]).to(tl.float32)
    out = tl.where(ok != 0.0, raw, 0.0)
    _store_sym(o_ptr + offset, out, i, j, N, BLOCK)


@torch.compiler.disable
@torch.no_grad()
def schatten_switch_blackwell(G: Tensor, variant=None, *, dtype: torch.dtype = torch.float16,
                              scaling: str = "schatten8", sigma_hat=None,
                              allow_unsafe_bf16: bool = False) -> Tensor:
    """Orthogonalize ``G`` with a Schatten-8 pre-scaled Polar Switch circuit.

    ``variant`` is a name in ``SCHATTEN_SWITCH_CIRCUITS`` or a ``SchattenSwitchCircuit``.
    ``scaling="frobenius"`` uses the Frobenius norm as the bound (for comparison only).
    ``sigma_hat`` is a test hook: an explicit scale in units of ``||G||_F`` per matrix, so that
    ``x = sigma / (sigma_hat ||G||_F)``. It bypasses the bound and is unsafe if it is smaller
    than ``sigma_max(G) / ||G||_F``.
    The result has the dtype and shape of ``G``.
    """
    circuit = _lookup(DEFAULT_VARIANT if variant is None else variant)
    if dtype not in _WORK_DTYPES:
        raise TypeError(f"Working dtype must be float16 or bfloat16, received {dtype}")
    _check(circuit, dtype, scaling, allow_unsafe_bf16)
    if G.ndim < 2 or not G.is_cuda or not G.is_floating_point():
        raise ValueError("Expected a floating-point CUDA tensor of shape (..., rows, columns)")
    if G.numel() == 0:
        return G.clone()

    X32 = G.reshape(-1, *G.shape[-2:]).float().contiguous()
    batch, rows, columns = X32.shape
    out_dtype = G.dtype
    device = X32.device
    n, k = min(rows, columns), max(rows, columns)
    size = rows * columns
    nn = n * n
    prescale, alpha = _prescale(n)
    u, e = _ROUNDING[dtype]

    m_part = torch.empty((batch, _REDUCE_BLOCKS), dtype=torch.float64, device=device)
    s_part = torch.empty_like(m_part)
    nf_part = torch.empty((batch, _REDUCE_BLOCKS), dtype=torch.int32, device=device)
    if sigma_hat is not None:
        factors = torch.tensor(_explicit_factors(circuit, sigma_hat, batch, n, dtype),
                               dtype=torch.float64, device=device)
        has_factors = 1
    else:
        factors = m_part
        has_factors = 0

    size_blocks = min(_REDUCE_BLOCKS, triton.cdiv(size, 4096))
    _launch(_safe_norm_part_kernel, lambda c: (size_blocks, batch, 1),
            X32, m_part, s_part, nf_part, size, size_blocks)
    X = torch.empty(X32.shape, dtype=dtype, device=device)
    _launch(_normalize_safe_kernel, lambda c: (triton.cdiv(size, c["BLOCK"]), batch, 1),
            X32, m_part, s_part, nf_part, factors, X,
            size, size_blocks, has_factors, float(prescale), float(FRO_FLOOR))

    work = X.new_empty(circuit.buffers, batch, n, n)
    grid = _triangle_grid(n, batch)
    table = circuit.table
    seg1_residual, seg1_rows = table[0]
    out_coeffs = torch.empty((len(seg1_rows), batch, 4), dtype=torch.float32, device=device)

    view = X.mT if rows > columns else X
    _launch(_syrk_kernel, grid, view, work[0], n, k, *view.stride())
    _launch(_symprod_kernel, grid, work[0], work[0], work[0], work[1], n, float(alpha), 0.0, 0.0)

    nn_blocks = min(_REDUCE_BLOCKS, triton.cdiv(nn, 4096))
    _launch(_dual_norm_part_kernel, lambda c: (nn_blocks, batch, 1),
            work[0], work[1], m_part, s_part, nn, nn_blocks)

    tbl_coeffs, tbl_regs = _seg1_table_tensors(circuit, device)
    has_b0 = int(seg1_rows[0][4] != 0.0)
    has_c0 = int(seg1_rows[0][5] != 0.0)
    use_frobenius = int(has_factors != 0 or scaling == "frobenius")
    row0_grid = (lambda c: (triton.cdiv(nn, c["BLOCK"]), batch, 1)) if (has_b0 or has_c0) else (lambda c: (1, batch, 1))
    _launch(_fold_and_row0_axpy_kernel, row0_grid,
            work[0], work[1], m_part, s_part, tbl_coeffs, tbl_regs, out_coeffs,
            n, nn, batch, nn_blocks, len(seg1_rows), use_frobenius, has_b0, has_c0,
            float(u), float(e), float(prescale), float(alpha),
            float(circuit.u_safe), float(_SLACK), float(seg1_residual))

    for j, (dst, src, add, *_rest) in enumerate(seg1_rows[1:], 1):
        _launch(_symprod_ptr_kernel, grid, work[src], work[src], work[add], work[dst], out_coeffs[j], n)

    last = len(table) - 1
    seg1_out_dtype = out_dtype if last == 0 else dtype
    if seg1_residual == 0.0:
        X = _apply(work[dst], X, 0.0, seg1_out_dtype)
    else:
        prod = _apply(work[dst], X, 0.0, torch.float32)
        res_scale = out_coeffs[0, :, 0].view(-1, 1, 1)
        X = (prod + res_scale * X.float()).to(seg1_out_dtype)

    for phase, (residual, iterations) in enumerate(table[1:], 1):
        view = X.mT if rows > columns else X
        _launch(_syrk_kernel, grid, view, work[0], n, k, *view.stride())
        for dst, src, add, a, b, c in iterations:
            _launch(_symprod_kernel, grid, work[src], work[src], work[add], work[dst], n, a, b, c)
        X = _apply(work[dst], X, residual, out_dtype if phase == last else dtype)

    return X.reshape(G.shape)


# --------------------------------------------------------------------------------------
# Reference: the same arithmetic in plain PyTorch, on CPU or CUDA
# --------------------------------------------------------------------------------------
def _sigma_max(M: Tensor) -> float:
    """Largest singular value in FP64: exact for a short side up to 1024, else power iteration
    from the all-ones vector plus noise (a lower estimate; it converges fast for a dominant
    top singular value)."""
    M = M.double()
    if min(M.shape) <= 1024:
        return torch.linalg.matrix_norm(M, ord=2).item()
    generator = torch.Generator(device=M.device).manual_seed(0)
    v = 1.0 + 0.01 * torch.randn(M.shape[1], dtype=torch.float64, device=M.device, generator=generator)
    for _ in range(100):
        v = M.mT @ (M @ v)
        v = v / v.norm()
    return (M @ v).norm().item()


def _product(A: Tensor, B: Tensor, dtype: torch.dtype) -> Tensor:
    """``A @ B`` from operands already stored in ``dtype``: FP32 accumulation (FP64 for float64)."""
    wide = torch.float64 if dtype == torch.float64 else torch.float32
    return A.to(wide) @ B.to(wide)


@torch.no_grad()
def schatten_switch_reference(G: Tensor, variant=None, *, dtype: torch.dtype = torch.float64,
                              scaling: str = "schatten8", sigma_hat=None, return_info: bool = False):
    """Reference for ``schatten_switch_blackwell`` in plain PyTorch (any device).

    ``dtype=torch.float64`` is the exact-arithmetic matrix reference. ``float16`` / ``bfloat16``
    emulate the kernels: operands in ``dtype``, FP32 accumulation, one rounding per operation.
    Norm reductions run in FP64, so the bound is the same on CPU and CUDA. Each matrix runs
    with its own scale. ``info`` reports, per matrix and in units of ``||G||_F``: ``upper``
    (the bound on ``sigma_max``), ``sigma_hat``, ``sigma_max_operand`` (FP64 top singular value
    of the rounded operand), and the measured and bounded cast residual of the Gram matrix.
    """
    circuit = _lookup(DEFAULT_VARIANT if variant is None else variant)
    if dtype not in _ROUNDING:
        raise TypeError(f"dtype must be float64, float16 or bfloat16, received {dtype}")
    if scaling not in ("schatten8", "frobenius"):
        raise ValueError("scaling must be 'schatten8' or 'frobenius'")
    X = G.reshape(-1, *G.shape[-2:]).double()
    rows, columns = X.shape[-2:]
    n = min(rows, columns)
    prescale, alpha = _prescale(n)
    eye = torch.eye(n, dtype=torch.float64, device=X.device)
    outputs, info = [], {key: [] for key in ("upper", "sigma_hat", "sigma_max_operand",
                                             "cast_residual", "cast_residual_bound", "valid")}
    factors = None if sigma_hat is None else _explicit_factors(circuit, sigma_hat, len(X), n, dtype)
    u, e = _ROUNDING[dtype]
    for index, M in enumerate(X):
        amax = M.abs().max() if M.numel() else M.new_zeros(())
        fro = (M / amax).norm() * amax if torch.isfinite(amax) and amax > 0 else amax
        valid = bool(torch.isfinite(fro)) and fro.item() > FRO_FLOOR
        info["valid"].append(valid)
        if not valid:
            outputs.append(torch.zeros_like(M))
            for key in ("upper", "sigma_hat", "sigma_max_operand", "cast_residual", "cast_residual_bound"):
                info[key].append(0.0)
            continue
        operand_norm = prescale if factors is None else prescale / factors[index]
        Xh = (operand_norm * (M / amax) / (M / amax).norm()).to(dtype)
        view = Xh.mT if rows > columns else Xh
        exact = _product(view, view.mT, dtype)
        R = exact.to(dtype)
        S = (alpha * _product(R, R, dtype)).to(dtype)
        norm_r, norm_s = R.double().norm().item(), S.double().norm().item()
        kappa = _kappas(circuit, scaling, factors is not None, [norm_r], [norm_s], n, dtype)[0]
        info["upper"].append(math.sqrt(_upper(norm_r, norm_s, n, dtype)) / operand_norm)
        info["sigma_hat"].append(1.0 / (operand_norm * math.sqrt(kappa)))
        info["sigma_max_operand"].append(_sigma_max(Xh) / operand_norm)
        info["cast_residual"].append((exact.double() - R.double()).norm().item())
        info["cast_residual_bound"].append(u * norm_r * _SLACK + n * e)
        b0, c0, folded = _fold(circuit.table[0][1], kappa, alpha)
        work = {0: R, 1: (S.double() + b0 * R.double() + c0 * eye).to(dtype)}
        for phase, (residual, iterations) in enumerate(circuit.table):
            if phase:
                view = Xh.mT if rows > columns else Xh
                work = {0: _product(view, view.mT, dtype).to(dtype)}
            else:
                iterations, residual = folded, residual * math.sqrt(kappa)
            for dst, src, add, a, b, c in iterations:
                value = a * _product(work[src], work[src], dtype).double() + b * work[add].double() + c * eye
                work[dst] = value.to(dtype)
            Q = work[dst]
            product = _product(Xh, Q, dtype) if rows > columns else _product(Q, Xh, dtype)
            Xh = (product.double() + residual * Xh.double()).to(dtype)
        outputs.append(Xh.double())
    out = torch.stack(outputs).to(G.dtype).reshape(G.shape)
    return (out, info) if return_info else out
