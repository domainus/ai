"""Schatten-8 pre-scaled, register-minimal Polar Switch circuits on Blackwell (RTX 5080, sm_120).

Each segment is the Polar Switch grammar of ``polar_switch_blackwell``:

    B[0] = X X^T                                    ``_syrk_kernel``
    B[dst] = a * B[src]^2 + b * B[add] + c * I      ``_symprod_kernel`` / ``_symprod_ptr_kernel``:
                                                    FP16/BF16 operands, FP32 accumulation, one rounding
    X <- B[last dst] @ X                            ``_apply`` (cuBLAS)

``_syrk_kernel``, ``_CONFIGS``, ``_tri_tile``, and ``_store_sym`` come from ``kernels_blackwell``;
``_launch``, ``_symprod_kernel``, ``_triangle_grid``, and ``_apply`` come from
``polar_switch_blackwell``. The first SYRK and the first square of segment 1 have variants here
(``_syrk_norm_kernel``, ``_square_norm_kernel``) with the same tile loop and one more epilogue.

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
  synchronizations occur in natural calls (``sigma_hat=None``), making ``schatten_switch_blackwell``
  CUDA-graph capturable after one eager warm-up call per ``(circuit, shape, dtype, device)`` while
  giving each batch member its own exact scale and returning exact zeros for dead, tiny
  (``||G||_F <= 1e-12``), or non-finite inputs (note: inputs are cast to FP32 before norm
  reduction, so ``float64`` entries exceeding ``FLT_MAX`` are treated as non-finite). "FP32
  accumulation, one rounding" applies to the Triton kernels; ``_apply`` uses cuBLAS.
* Launch fusion (Cycle 3, C3-03). Segment 1 has three auxiliary launches: the norm partials,
  the normalization, and ``_fold_and_row0_axpy_kernel``. The norms of ``R_h`` and of
  ``S = fl(alpha * R_h^2)`` are not a separate pass: the kernel that computes a tile also writes
  the sum of squares of the rounded tile (``_tile_sumsq``, one slot per tile, no atomics), so
  a CUDA-graph replay repeats the eager result bit for bit. All device scalars of a call are
  in one scratch allocation that the call owns (``_scratch_layout``).

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
_SLACK = 1.0 + 2.0 ** -12  # FP32 accumulation of the products and of the norm partials
_REDUCE_BLOCKS = 256  # partial slots per matrix of the input norm
_NORM_BLOCK = 4096  # elements per step of one norm partial
_TILE_CHUNK = 64  # tile partials are summed in chunks of this size
_ELEMENTWISE_BLOCK = 4096  # elements per step of an elementwise kernel
_NORMALIZE_PROGRAMS, _NORMALIZE_WARPS = 64, 4  # most programs per matrix, warps per program
_FOLD_PROGRAMS, _FOLD_WARPS = 64, 4
_FOLD_ROWS = 8  # most rows of segment 1 (one vector operation folds all of them)
_MIN_BLOCK = min(config.kwargs["BLOCK"] for config in _CONFIGS)
# dtype -> (unit roundoff, largest absolute rounding error of a subnormal entry)
_ROUNDING = {torch.float16: (2.0 ** -11, 2.0 ** -25), torch.bfloat16: (2.0 ** -8, 0.0),
             torch.float64: (0.0, 0.0)}
_TABLE_CACHE: dict[tuple, Tensor] = {}


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
    return P, 2.0 ** 13 / P ** 4  # a power of two: the kernels also use the exact 1 / alpha


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

    ``_SLACK`` covers the FP32 accumulation of the products and of the two norms (the native
    path sums at most one tile row of exact squares in FP32, then FP64). The trace bound
    ``||X_h||_F^2`` is taken when it is smaller.
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
def _seg1_table(circuit: SchattenSwitchCircuit, device: torch.device) -> Tensor:
    """Cached float64 rows ``(a, b, c, src_code, add_code)`` of segment 1 on ``device``.

    A code names the scale of the buffer that a row reads, so the kernel folds every row
    without the register history: 0 is buffer 0 (``x^2 / kappa``), 1 is the first row
    (``alpha / kappa^2`` times its table value), 2 is a later destination (its table value).
    """
    seg1_rows = circuit.table[0][1]
    key = (circuit.name, seg1_rows, device)
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    if len(seg1_rows) > _FOLD_ROWS:
        raise ValueError(f"Segment 1 supports at most {_FOLD_ROWS} rows, got {len(seg1_rows)}")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Warm up schatten_switch_blackwell eagerly once per (circuit, shape, dtype, device) "
            "before CUDA-graph capture"
        )
    codes, values = {0: 0.0, 1: 1.0}, [[*seg1_rows[0][3:6], 2.0, 2.0]]
    for dst, src, add, a, b, c in seg1_rows[1:]:
        values.append([a, b, c, codes.get(src, 2.0), codes.get(add, 2.0)])
        codes[dst] = 2.0
    table = torch.tensor(values, dtype=torch.float64, device=device)
    _TABLE_CACHE[key] = table
    return table


def _scratch_layout(n: int, rows: int) -> tuple[int, int, int, int]:
    """Offsets in the float64 scratch row of one matrix: ``(off_r, off_s, off_c, width)``.

    One allocation per call holds every device-side scalar, so a call owns its scratch (safe
    for concurrent streams and graphs):

        [0, 3 * _REDUCE_BLOCKS)  input norm partials: sums of squares, maxima, non-finite flags
        [off_r, off_s)           Frobenius-squared partial of each stored tile of ``R_h``
        [off_s, off_c)           the same for ``S = fl(alpha * R_h^2)``
        [off_c, width)           segment-1 coefficient rows ``(a, b, c, ok)``; row 0 holds
                                 ``(residual * sqrt(kappa), unused, unused, ok)``

    A tile region has room for the smallest autotune ``BLOCK``, because the tile count of the
    chosen configuration is known only after the kernel ran.
    """
    tiles = -(-n // _MIN_BLOCK)
    capacity = tiles * (tiles + 1) // 2
    off_r = 3 * _REDUCE_BLOCKS
    return off_r, off_r + capacity, off_r + 2 * capacity, off_r + 2 * capacity + 4 * rows


def _tile_count(n: int, config: dict) -> int:
    tiles = -(-n // config["BLOCK"])
    return tiles * (tiles + 1) // 2


@triton.autotune(configs=[triton.Config({"BLOCK": _NORM_BLOCK, "REDUCE_BLOCKS": _REDUCE_BLOCKS}, num_warps=4)], key=[])
@triton.jit
def _safe_norm_part_kernel(
    x_ptr, sc_ptr, size, num_blocks, sc_stride,
    BLOCK: tl.constexpr, REDUCE_BLOCKS: tl.constexpr,
):
    """Overflow- and underflow-safe partial sums of squares of each matrix (Blue-style scaling).

    A chunk is scaled by its largest entry and summed in FP32. The chunk sums accumulate in
    FP64, where the square of every finite FP32 value is in range.
    """
    pid = tl.program_id(0)
    matrix = tl.program_id(1).to(tl.int64)
    stride_i32 = num_blocks * BLOCK
    stride = stride_i32.to(tl.int64)
    n_steps = tl.cdiv(size, stride_i32)

    q_cur = tl.full((), 0.0, dtype=tl.float64)
    m_cur = tl.full((), 0.0, dtype=tl.float32)
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
        mc = tl.where(has_val, m_chunk, 0.0)
        mc64 = mc.to(tl.float64)
        q_cur += s_chunk.to(tl.float64) * (mc64 * mc64)
        m_cur = tl.maximum(m_cur, mc)

    out = sc_ptr + matrix * sc_stride + pid.to(tl.int64)
    tl.store(out, q_cur)
    tl.store(out + REDUCE_BLOCKS, m_cur.to(tl.float64))
    tl.store(out + 2 * REDUCE_BLOCKS, nf_cur.to(tl.float64))


@triton.autotune(configs=[triton.Config({"BLOCK": _ELEMENTWISE_BLOCK, "REDUCE_BLOCKS": _REDUCE_BLOCKS},
                                        num_warps=_NORMALIZE_WARPS)], key=[])
@triton.jit
def _normalize_safe_kernel(
    x_ptr, sc_ptr, factors_ptr, o_ptr,
    size, num_blocks, num_programs, sc_stride, has_factors, prescale, fro_floor, dtype_max,
    BLOCK: tl.constexpr, REDUCE_BLOCKS: tl.constexpr,
):
    """Reduce the norm partials in FP64 and normalize to the operand norm (zeros if invalid).

    Every thread of every program repeats the FP64 reduction, so a matrix gets at most
    ``num_programs`` programs, and each one walks its blocks in strided steps.
    """
    pid = tl.program_id(0).to(tl.int64)
    matrix = tl.program_id(1).to(tl.int64)
    idx = tl.arange(0, REDUCE_BLOCKS)
    bmask = idx < num_blocks
    part = sc_ptr + matrix * sc_stride + idx

    total = tl.sum(tl.load(part, mask=bmask, other=0.0), axis=0)
    M = tl.max(tl.load(part + REDUCE_BLOCKS, mask=bmask, other=0.0).to(tl.float32), axis=0)
    any_nf = tl.max(tl.load(part + 2 * REDUCE_BLOCKS, mask=bmask, other=0.0).to(tl.float32), axis=0) > 0.0
    M_safe = tl.where(M > 0.0, M, 1.0)
    fro = tl.sqrt(total)

    # With s = ||G / M||_F = fro / M: the scale is prescale / s, and it is in range if
    # prescale <= dtype_max * s. Both are written without a second FP64 division.
    zero64 = tl.full((), 0.0, dtype=tl.float64)
    scaled_max = zero64 + prescale
    if has_factors != 0:
        scaled_max = scaled_max / tl.load(factors_ptr + matrix).to(tl.float64)
    scaled_max = scaled_max * M_safe.to(tl.float64)
    valid = (
        (~any_nf) & (M > 0.0) & (fro == fro) & (fro < 1e308) & (fro > fro_floor)
        & (scaled_max <= (zero64 + dtype_max) * fro)
    )
    unit_scale = tl.where(valid, scaled_max / tl.where(fro > 0.0, fro, 1.0), 0.0).to(tl.float32)

    span_i32 = num_programs * BLOCK
    span = span_i32.to(tl.int64)
    for step in range(0, tl.cdiv(size, span_i32)):
        local = step.to(tl.int64) * span + pid * BLOCK + tl.arange(0, BLOCK)
        mask = local < size
        x = tl.load(x_ptr + matrix * size + local, mask=mask, other=0.0).to(tl.float32)
        y = tl.where(valid, (x / M_safe) * unit_scale, 0.0)
        tl.store(o_ptr + matrix * size + local, y.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _tile_sumsq(stored, i, j, N, BLOCK: tl.constexpr):
    """Sum of squares that tile ``(i, j)`` adds to the stored symmetric matrix, in FP64.

    ``stored`` is the tile after rounding to the buffer dtype. ``_store_sym`` writes an entry
    above the diagonal twice (itself and its mirror), a diagonal entry once, and no entry below
    the diagonal of a diagonal tile. The square of an FP16 or BF16 value is exact in FP32. One
    row (at most ``BLOCK`` squares) is summed in FP32, and the row sums in FP64.
    """
    rm = i * BLOCK + tl.arange(0, BLOCK)
    rn = j * BLOCK + tl.arange(0, BLOCK)
    v = stored.to(tl.float32)
    inside = (rm[:, None] < N) & (rn[None, :] < N)
    weight = (rm[:, None] <= rn[None, :]).to(tl.float32) + (rm[:, None] < rn[None, :]).to(tl.float32)
    rows = tl.sum(tl.where(inside, weight * (v * v), 0.0), axis=1)
    return tl.sum(rows.to(tl.float64), axis=0)


@triton.autotune(configs=_CONFIGS, key=["N", "K"])
@triton.jit
def _syrk_norm_kernel(
    x_ptr, c_ptr, sc_ptr, N, K, sxb, sxn, sxk, part_off, sc_stride,
    BLOCK: tl.constexpr, BK: tl.constexpr,
):
    """``_syrk_kernel`` that also writes ``_tile_sumsq`` of each tile to the scratch row.

    The norm of ``R_h`` comes from the tiles in registers: no second pass over the matrix, no
    atomics, and one fixed summation order per configuration.
    """
    pid = tl.program_id(0)
    bidx = tl.program_id(1).to(tl.int64)
    i, j = _tri_tile(pid, tl.cdiv(N, BLOCK))
    rm = i * BLOCK + tl.arange(0, BLOCK)
    rn = j * BLOCK + tl.arange(0, BLOCK)
    rk = tl.arange(0, BK)
    x_base = x_ptr + bidx * sxb
    a_ptrs = x_base + rm[:, None].to(tl.int64) * sxn + rk[None, :] * sxk
    b_ptrs = x_base + rk[:, None] * sxk + rn[None, :].to(tl.int64) * sxn
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kmask = rk < K - k * BK
        a = tl.load(a_ptrs, mask=(rm[:, None] < N) & kmask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=kmask[:, None] & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * sxk
        b_ptrs += BK * sxk
    stored = acc.to(c_ptr.dtype.element_ty)
    _store_sym(c_ptr + bidx * N * N, stored, i, j, N, BLOCK)
    tl.store(sc_ptr + bidx * sc_stride + part_off + pid, _tile_sumsq(stored, i, j, N, BLOCK))


@triton.autotune(configs=_CONFIGS, key=["N"])
@triton.jit
def _square_norm_kernel(
    a_ptr, o_ptr, sc_ptr, N, alpha, part_off, sc_stride,
    BLOCK: tl.constexpr, BK: tl.constexpr,
):
    """``out = fl(alpha * A @ A)`` for symmetric ``A``, plus ``_tile_sumsq`` of each tile.

    The first square of segment 1. ``out`` must not alias ``A``.
    """
    pid = tl.program_id(0)
    bidx = tl.program_id(1).to(tl.int64)
    offset = bidx * N * N
    i, j = _tri_tile(pid, tl.cdiv(N, BLOCK))
    rm = i * BLOCK + tl.arange(0, BLOCK)
    rn = j * BLOCK + tl.arange(0, BLOCK)
    rk = tl.arange(0, BK)
    l_ptrs = a_ptr + offset + rm[:, None].to(tl.int64) * N + rk[None, :]
    r_ptrs = a_ptr + offset + rk[:, None].to(tl.int64) * N + rn[None, :]
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BK)):
        kmask = rk < N - k * BK
        left = tl.load(l_ptrs, mask=(rm[:, None] < N) & kmask[None, :], other=0.0)
        right = tl.load(r_ptrs, mask=kmask[:, None] & (rn[None, :] < N), other=0.0)
        acc = tl.dot(left, right, acc)
        l_ptrs += BK
        r_ptrs += BK * N
    stored = (alpha * acc).to(o_ptr.dtype.element_ty)
    _store_sym(o_ptr + offset, stored, i, j, N, BLOCK)
    tl.store(sc_ptr + bidx * sc_stride + part_off + pid, _tile_sumsq(stored, i, j, N, BLOCK))


@triton.autotune(configs=[triton.Config({"BLOCK": _ELEMENTWISE_BLOCK, "CHUNK": _TILE_CHUNK, "ROWS": _FOLD_ROWS},
                                        num_warps=_FOLD_WARPS)], key=[])
@triton.jit
def _fold_and_row0_axpy_kernel(
    r_ptr, s_ptr, sc_ptr, tbl_ptr,
    n, nn, num_programs, sc_stride, off_r, tiles_r, off_s, tiles_s, off_c, num_rows,
    use_frobenius, has_b0, has_c0,
    u, e, prescale, alpha, inv_alpha, u_safe, slack, residual,
    BLOCK: tl.constexpr, CHUNK: tl.constexpr, ROWS: tl.constexpr,
):
    """Reduce the tile partials to the Schatten-8 ``kappa`` of each matrix, fold the segment-1
    table with it, and apply the first row ``S += b0 R + c0 I``, in one launch.

    Programs ``0 .. num_programs - 1`` of a matrix apply the first row in strided steps.
    Program ``num_programs`` writes the coefficient rows. An FP64 division or square root
    costs about ten multiplications, and every thread of every program computes the bound. So
    the bound is ``w = 1 / kappa`` (no division), and the programs per matrix are few.
    """
    pid = tl.program_id(0)
    matrix = tl.program_id(1).to(tl.int64)
    sc = sc_ptr + matrix * sc_stride
    idx = tl.arange(0, CHUNK)

    sum_r = tl.full((), 0.0, dtype=tl.float64)
    for chunk in range(0, tl.cdiv(tiles_r, CHUNK)):
        at = chunk * CHUNK + idx
        sum_r += tl.sum(tl.load(sc + off_r + at, mask=at < tiles_r, other=0.0), axis=0)
    sum_s = tl.full((), 0.0, dtype=tl.float64)
    for chunk in range(0, tl.cdiv(tiles_s, CHUNK)):
        at = chunk * CHUNK + idx
        sum_s += tl.sum(tl.load(sc + off_s + at, mask=at < tiles_s, other=0.0), axis=0)
    norm_r = tl.sqrt(sum_r)
    norm_s = tl.sqrt(sum_s)
    ok = (norm_r == norm_r) & (norm_r < 1e308) & (norm_s == norm_s) & (norm_s < 1e308)

    zero64 = tl.full((), 0.0, dtype=tl.float64)
    u64 = zero64 + u
    e64 = zero64 + e
    prescale64 = zero64 + prescale
    alpha64 = zero64 + alpha
    inv_alpha64 = zero64 + inv_alpha
    u_safe64 = zero64 + u_safe
    slack64 = zero64 + slack

    trace_bound = prescale64 * prescale64 * (1.0 + u64) * (1.0 + u64) * slack64
    schatten8 = tl.sqrt(((1.0 + u64) * norm_s * slack64 + n * e64) * inv_alpha64)
    res_bound = u64 * norm_r * slack64 + n * e64
    schatten_upper = tl.where(norm_s > 0.0, tl.minimum(schatten8 + res_bound, trace_bound), trace_bound)
    upper = tl.where(use_frobenius != 0, trace_bound, schatten_upper)
    w = tl.where(ok, u_safe64 * u_safe64 * upper, 1.0)

    if pid == num_programs:
        # Row j reads buffers with the scales 1 / g: g = kappa for buffer 0, kappa^2 / alpha
        # for the first row, and 1 for a later destination (the codes of ``_seg1_table``).
        # The last row carries sqrt(kappa), so that Q @ X_h has the singular values f(x).
        kappa = 1.0 / w
        sqrt_k = tl.sqrt(kappa)
        g1 = kappa * kappa * inv_alpha64
        ok_flag = tl.where(ok, 1.0, 0.0) + zero64
        coeff = sc + off_c
        tl.store(coeff, tl.where(ok, (zero64 + residual) * sqrt_k, 0.0))
        tl.store(coeff + 3, ok_flag)

        rows = tl.arange(0, ROWS)
        folded = (rows >= 1) & (rows < num_rows)
        a = tl.load(tbl_ptr + rows * 5 + 0, mask=folded, other=0.0)
        b = tl.load(tbl_ptr + rows * 5 + 1, mask=folded, other=0.0)
        c = tl.load(tbl_ptr + rows * 5 + 2, mask=folded, other=0.0)
        src = tl.load(tbl_ptr + rows * 5 + 3, mask=folded, other=2.0)
        add = tl.load(tbl_ptr + rows * 5 + 4, mask=folded, other=2.0)
        g_src = tl.where(src == 0.0, kappa, tl.where(src == 1.0, g1, 1.0))
        g_add = tl.where(add == 0.0, kappa, tl.where(add == 1.0, g1, 1.0))
        t = tl.where(rows == num_rows - 1, sqrt_k, 1.0)
        tl.store(coeff + rows * 4 + 0, tl.where(ok, a * t * (g_src * g_src), 0.0), mask=folded)
        tl.store(coeff + rows * 4 + 1, tl.where(ok, b * t * g_add, 0.0), mask=folded)
        tl.store(coeff + rows * 4 + 2, tl.where(ok, c * t, 0.0), mask=folded)
        tl.store(coeff + rows * 4 + 3, ok_flag + tl.zeros((ROWS,), dtype=tl.float64), mask=folded)
    else:
        b0_t = tl.where(ok, alpha64 * tl.load(tbl_ptr + 1) * w, 0.0).to(tl.float32)
        c0_t = tl.where(ok, alpha64 * tl.load(tbl_ptr + 2) * (w * w), 0.0).to(tl.float32)
        span_i32 = num_programs * BLOCK
        span = span_i32.to(tl.int64)
        for step in range(0, tl.cdiv(nn, span_i32)):
            local = step.to(tl.int64) * span + pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
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
# coeff_off selects the coefficient row: it changes per launch and must not specialize.
@triton.autotune(configs=_CONFIGS, key=["N"], restore_value=["o_ptr"])
@triton.jit(do_not_specialize=["coeff_off"])
def _symprod_ptr_kernel(
    a_ptr, b_ptr, c_ptr, o_ptr, sc_ptr, N, sc_stride, coeff_off,
    BLOCK: tl.constexpr, BK: tl.constexpr,
):
    """``_symprod_kernel`` with the coefficients ``(a, b, c, ok)`` of each matrix read from its
    scratch row at ``coeff_off`` (FP64 values, rounded to FP32 here)."""
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
    coeff = sc_ptr + bidx * sc_stride + coeff_off
    alpha = tl.load(coeff + 0).to(tl.float32)
    beta = tl.load(coeff + 1).to(tl.float32)
    gamma = tl.load(coeff + 2).to(tl.float32)
    ok = tl.load(coeff + 3)
    raw = alpha * acc + beta * c.to(tl.float32) + gamma * (rm[:, None] == rn[None, :]).to(tl.float32)
    out = tl.where(ok != 0.0, raw, 0.0)
    _store_sym(o_ptr + offset, out, i, j, N, BLOCK)


def _fold_segment1(circuit: SchattenSwitchCircuit, R: Tensor, S: Tensor, scratch: Tensor, tiles_r: int,
                   tiles_s: int, *, k: int, use_frobenius: bool) -> None:
    """Launch ``_fold_and_row0_axpy_kernel`` on the stored ``R_h`` and ``S`` of shape (batch, n, n).

    ``scratch`` holds ``tiles_r`` and ``tiles_s`` tile partials per matrix (``_scratch_layout``).
    The kernel writes the coefficient rows there and updates ``S`` in place. ``k`` is the long
    side of the input: above 2^16 columns the slack grows with the FP32 Gram bias (Fable M1).
    """
    batch, n, _ = R.shape
    nn = n * n
    seg1_residual, seg1_rows = circuit.table[0]
    prescale, alpha = _prescale(n)
    u, e = _ROUNDING[R.dtype]
    k_slack = 1.0 if k <= 65536 else 1.0 / max(1.0 - 1.2e-8 * k, 0.2)
    off_r, off_s, off_c, stride = _scratch_layout(n, len(seg1_rows))
    has_b0 = int(seg1_rows[0][4] != 0.0)
    has_c0 = int(seg1_rows[0][5] != 0.0)
    programs = min(_FOLD_PROGRAMS, -(-nn // _ELEMENTWISE_BLOCK)) if (has_b0 or has_c0) else 0
    _launch(_fold_and_row0_axpy_kernel, lambda c: (programs + 1, batch, 1),
            R, S, scratch, _seg1_table(circuit, R.device),
            n, nn, programs, stride, off_r, tiles_r, off_s, tiles_s, off_c, len(seg1_rows),
            int(use_frobenius), has_b0, has_c0,
            float(u), float(e), float(prescale), float(alpha), 1.0 / alpha,
            float(circuit.u_safe), float(_SLACK * k_slack), float(seg1_residual))


@torch.compiler.disable
@torch.no_grad()
def schatten_switch_blackwell(G: Tensor, variant=None, *, dtype: torch.dtype = torch.float16,
                              scaling: str = "schatten8", sigma_hat=None,
                              allow_unsafe_bf16: bool = False) -> Tensor:
    """Orthogonalize ``G`` with a Schatten-8 pre-scaled Polar Switch circuit.

    ``variant`` is a name in ``SCHATTEN_SWITCH_CIRCUITS`` or a ``SchattenSwitchCircuit``.
    ``scaling="frobenius"`` uses the Frobenius norm as the bound (for comparison only).
    ``sigma_hat`` is a test hook: an explicit scale in units of ``||G||_F`` per matrix, so that
    ``x = sigma / (sigma_hat ||G||_F)``. It bypasses the bound, runs host setup (not capturable),
    and is unsafe if smaller than ``sigma_max(G) / ||G||_F``. CUDA-graph capture is supported after
    one eager warm-up call per ``(variant, shape, dtype, device)``. ``float64`` inputs exceeding
    the FP32 finite range map to non-finite and return zeros. The result has the dtype and shape
    of ``G``.
    """
    circuit = _lookup(DEFAULT_VARIANT if variant is None else variant)
    if dtype not in _WORK_DTYPES:
        raise TypeError(f"Working dtype must be float16 or bfloat16, received {dtype}")
    _check(circuit, dtype, scaling, allow_unsafe_bf16)
    if len(circuit.table[0][1]) > _FOLD_ROWS:
        raise ValueError(f"Segment 1 supports at most {_FOLD_ROWS} rows, got {len(circuit.table[0][1])}")
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
    prescale, alpha = _prescale(n)
    dtype_max = float(torch.finfo(dtype).max)
    table = circuit.table
    seg1_residual, seg1_rows = table[0]

    off_r, off_s, off_c, stride = _scratch_layout(n, len(seg1_rows))
    scratch = torch.empty((batch, stride), dtype=torch.float64, device=device)
    if sigma_hat is not None:
        factors = torch.tensor(_explicit_factors(circuit, sigma_hat, batch, n, dtype),
                               dtype=torch.float64, device=device)
        has_factors = 1
    else:
        factors = scratch
        has_factors = 0

    size_blocks = min(_REDUCE_BLOCKS, -(-size // _NORM_BLOCK))
    _launch(_safe_norm_part_kernel, lambda c: (size_blocks, batch, 1), X32, scratch, size, size_blocks, stride)
    X = torch.empty(X32.shape, dtype=dtype, device=device)
    programs = min(_NORMALIZE_PROGRAMS, -(-size // _ELEMENTWISE_BLOCK))
    _launch(_normalize_safe_kernel, lambda c: (programs, batch, 1),
            X32, scratch, factors, X,
            size, size_blocks, programs, stride, has_factors, float(prescale), float(FRO_FLOOR), dtype_max)

    work = X.new_empty(circuit.buffers, batch, n, n).unbind(0)
    grid = _triangle_grid(n, batch)
    view = X.mT if rows > columns else X
    tiles_r = _tile_count(n, _launch(_syrk_norm_kernel, grid, view, work[0], scratch, n, k, *view.stride(),
                                     off_r, stride))
    tiles_s = _tile_count(n, _launch(_square_norm_kernel, grid, work[0], work[1], scratch, n, float(alpha),
                                     off_s, stride))

    _fold_segment1(circuit, work[0], work[1], scratch, tiles_r, tiles_s, k=k,
                   use_frobenius=has_factors != 0 or scaling == "frobenius")
    for j, (dst, src, add, *_rest) in enumerate(seg1_rows[1:], 1):
        _launch(_symprod_ptr_kernel, grid, work[src], work[src], work[add], work[dst], scratch,
                n, stride, off_c + 4 * j)

    if sigma_hat is not None:
        X = torch.where((scratch[:, off_c + 3] != 0.0)[:, None, None], X, 0.0)

    last = len(table) - 1
    seg1_out_dtype = out_dtype if last == 0 else dtype
    if seg1_residual == 0.0:
        X = _apply(work[dst], X, 0.0, seg1_out_dtype)
    else:
        prod = _apply(work[dst], X, 0.0, torch.float32)
        res_scale = scratch[:, off_c].float().view(-1, 1, 1)
        X = (prod + res_scale * X.float()).to(seg1_out_dtype)
    if sigma_hat is not None:
        X = torch.where(torch.isfinite(X).all(dim=(-2, -1), keepdim=True), X, 0.0)

    for phase, (residual, iterations) in enumerate(table[1:], 1):
        view = X.mT if rows > columns else X
        _launch(_syrk_kernel, grid, view, work[0], n, k, *view.stride())
        for dst, src, add, a, b, c in iterations:
            _launch(_symprod_kernel, grid, work[src], work[src], work[add], work[dst], n, a, b, c)
        X = _apply(work[dst], X, residual, out_dtype if phase == last else dtype)
        if sigma_hat is not None:
            X = torch.where(torch.isfinite(X).all(dim=(-2, -1), keepdim=True), X, 0.0)

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
