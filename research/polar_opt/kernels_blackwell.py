"""Triton / FP8 kernels for polar orthogonalization on NVIDIA Blackwell (RTX 5080, sm_120).

Contents
--------
* ``triton_syrk(X)``            symmetric Gram ``X @ X.mT``: only upper-triangle tiles are
                                 computed, each tile is mirrored into the lower triangle, so the
                                 result is *bitwise* symmetric.  fp32 accumulation, one rounding.
* ``triton_gram_poly_step(A, b, c, a=0)``  fused ``b*A + c*(A @ A) + a*I`` for symmetric ``A``
                                 (same triangular tiling, one read of ``A`` as epilogue input,
                                 one rounding on output).
* ``polar_express_triton``      full Gram-form PolarExpress circuit built from the two kernels
                                 plus one cuBLAS ``baddbmm`` for ``a*X + B@X``.
* ``polar_express_fp8``         Experimental FP8-early / BF16-late (or FP16-late) circuit
                                 (default ``fp8_steps=1``). NOTE: ``fp8_steps >= 2`` (including
                                 with ``on_misaligned="pad"``) diverges on benign small-``n``
                                 (``n <= 256``) uniform/Gaussian spectra (``sigma_max`` up to
                                 ``114,600``), and even when stable FP8 PE is 20-77% slower than
                                 FP16 ``polar_switch_blackwell`` on LLM shapes.
* ``polar_express_torch``       plain-PyTorch reference of the same circuit (cuBLAS baseline,
                                 also used in fp64 as the algorithmic reference).
* ``polar_svd_fp64``            exact polar factor ``U V^T`` in fp64.

All entry points accept ``(N, M)`` or ``(B, N, M)`` tensors.  Nothing here uses
``torch.compile`` so ``TORCH_COMPILE_DISABLE=1`` has no effect on these paths.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from research.polar_opt.baselines import _AMSEL_COEFFICIENTS

# --------------------------------------------------------------------------------------
# Polar Express coefficients (Amsel et al. 2025, l = 1e-3, degree-5 odd polynomials).
# Matched verbatim to baselines._AMSEL_COEFFICIENTS (cushion=0.02, safety_factor_eps=0.01)
# so polar_express_triton and baselines.polar_express share the exact polynomial table.
# --------------------------------------------------------------------------------------
SAFETY = 1.01
POLAR_EXPRESS_COEFFS = list(_AMSEL_COEFFICIENTS)

FP8_MAX = 448.0  # largest finite float8_e4m3fn
_FP8 = torch.float8_e4m3fn


# --------------------------------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------------------------------
def _cfg(block, bk, warps, stages):
    return triton.Config({"BLOCK": block, "BK": bk}, num_warps=warps, num_stages=stages)


# sm_120 has ~99 KB of shared memory per CTA (vs 227 KB on sm_90/100), so stage counts are kept
# low for the big tiles; configs that do not fit are skipped by the autotuner.  BLOCK=32 exists
# because a small N yields few upper-triangle tiles and 84 SMs would otherwise idle.
_CONFIGS = [
    _cfg(128, 64, 8, 3), _cfg(128, 64, 4, 3), _cfg(128, 32, 8, 4), _cfg(128, 32, 4, 4),
    _cfg(128, 64, 8, 2), _cfg(64, 64, 4, 4), _cfg(64, 64, 4, 3), _cfg(64, 32, 4, 5),
    _cfg(64, 128, 4, 3), _cfg(32, 64, 4, 4), _cfg(32, 128, 4, 3),
]


@triton.jit
def _tri_tile(pid, T):
    """Linear index in a row-major walk of the upper triangle of a TxT tile grid -> (i, j), i <= j."""
    T_i32 = T + tl.zeros((), dtype=tl.int32)
    Tf = T_i32.to(tl.float32)
    t = pid.to(tl.float32)
    disc = (2.0 * Tf + 1.0) * (2.0 * Tf + 1.0) - 8.0 * t
    i = tl.floor(((2.0 * Tf + 1.0) - tl.sqrt(tl.maximum(disc, 0.0))) * 0.5).to(tl.int32)
    # row i starts at i*T - i*(i-1)/2 and holds T-i tiles; fix any fp32 rounding by +-1.
    start = i * T_i32 - (i * (i - 1)) // 2
    i = tl.where(pid < start, i - 1, i)
    start = i * T_i32 - (i * (i - 1)) // 2
    i = tl.where(pid >= start + (T_i32 - i), i + 1, i)
    start = i * T_i32 - (i * (i - 1)) // 2
    return i, i + (pid - start)


@triton.jit
def _store_sym(out_ptr, val, i, j, N, BLOCK: tl.constexpr):
    """Store tile (i, j) of a symmetric NxN row-major matrix and its transpose at (j, i).

    Diagonal tiles contribute only their upper triangle (including the diagonal) directly and
    its mirror for the strict lower triangle, so out[r, c] == out[c, r] holds bitwise.
    """
    rm = i * BLOCK + tl.arange(0, BLOCK)
    rn = j * BLOCK + tl.arange(0, BLOCK)
    v = val.to(out_ptr.dtype.element_ty)
    upper = (rm[:, None] < N) & (rn[None, :] < N) & (rm[:, None] <= rn[None, :])
    tl.store(out_ptr + rm[:, None].to(tl.int64) * N + rn[None, :], v, mask=upper)
    lower = (rn[:, None] < N) & (rm[None, :] < N) & (rm[None, :] < rn[:, None])
    tl.store(out_ptr + rn[:, None].to(tl.int64) * N + rm[None, :], tl.trans(v), mask=lower)


# int64 audit (C5-01). Row offsets ``rm * sxn`` / ``rn * sxn``, the batch offset ``bidx * sxb``
# and every store offset are int64. ``rk * sxk`` and the step ``BK * sxk`` are int32 when Triton
# types ``sxk`` as int32 (|sxk| < 2^31), so they need ``128 * |sxk| < 2^31``; the K walk itself
# advances 64-bit pointers, so no offset grows with K. ``_syrk_chunked`` checks the bound.
@triton.autotune(configs=_CONFIGS, key=["N", "K"])
@triton.jit
def _syrk_kernel(
    x_ptr, c_ptr, N, K, sxb, sxn, sxk,
    BLOCK: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    bidx = tl.program_id(1).to(tl.int64)
    T = tl.cdiv(N, BLOCK)
    i, j = _tri_tile(pid, T)

    rm = i * BLOCK + tl.arange(0, BLOCK)
    rn = j * BLOCK + tl.arange(0, BLOCK)
    rk = tl.arange(0, BK)
    x_base = x_ptr + bidx * sxb
    a_ptrs = x_base + rm[:, None].to(tl.int64) * sxn + rk[None, :] * sxk  # X[rm, k]
    b_ptrs = x_base + rk[:, None] * sxk + rn[None, :].to(tl.int64) * sxn  # X[rn, k]^T
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kmask = rk < K - k * BK
        a = tl.load(a_ptrs, mask=(rm[:, None] < N) & kmask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=kmask[:, None] & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * sxk
        b_ptrs += BK * sxk
    _store_sym(c_ptr + bidx * N * N, acc, i, j, N, BLOCK)


@triton.autotune(configs=_CONFIGS, key=["N"])
@triton.jit
def _poly_kernel(
    a_ptr, o_ptr, N, alpha, beta, gamma,
    BLOCK: tl.constexpr, BK: tl.constexpr,
):
    """out = alpha*A + beta*(A @ A) + gamma*I for a symmetric, contiguous (B, N, N) A."""
    pid = tl.program_id(0)
    bidx = tl.program_id(1).to(tl.int64)
    T = tl.cdiv(N, BLOCK)
    i, j = _tri_tile(pid, T)

    rm = i * BLOCK + tl.arange(0, BLOCK)
    rn = j * BLOCK + tl.arange(0, BLOCK)
    rk = tl.arange(0, BK)
    base = a_ptr + bidx * N * N
    l_ptrs = base + rm[:, None].to(tl.int64) * N + rk[None, :]  # A[rm, k]
    r_ptrs = base + rk[:, None].to(tl.int64) * N + rn[None, :]  # A[k, rn]
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BK)):
        kmask = rk < N - k * BK
        l = tl.load(l_ptrs, mask=(rm[:, None] < N) & kmask[None, :], other=0.0)
        r = tl.load(r_ptrs, mask=kmask[:, None] & (rn[None, :] < N), other=0.0)
        acc = tl.dot(l, r, acc)
        l_ptrs += BK
        r_ptrs += BK * N

    tile_mask = (rm[:, None] < N) & (rn[None, :] < N)
    a_tile = tl.load(base + rm[:, None].to(tl.int64) * N + rn[None, :], mask=tile_mask, other=0.0).to(tl.float32)
    out = beta * acc + alpha * a_tile + gamma * (rm[:, None] == rn[None, :]).to(tl.float32)
    _store_sym(o_ptr + bidx * N * N, out, i, j, N, BLOCK)


@triton.jit
def _quant_kernel(x_ptr, out_ptr, amax_ptr, n, BLOCK: tl.constexpr):
    """out = fp8_e4m3(x * 448 / amax); amax lives in device memory so no host sync is needed."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * (448.0 / tl.load(amax_ptr))
    tl.store(out_ptr + offs, y.to(tl.float8e4nv), mask=mask)


@triton.jit
def _axpy_cast_kernel(u_ptr, x_ptr, out_ptr, a, n, BLOCK: tl.constexpr):
    """out = (u + a * x) rounded once to out's dtype; u is fp32, x is fp16/bf16."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    u = tl.load(u_ptr + offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (u + a * x).to(out_ptr.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------------------
# Python wrappers
# --------------------------------------------------------------------------------------
_HALF = (torch.float16, torch.bfloat16)


def _as_batched(X: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if X.ndim == 2:
        return X.unsqueeze(0), True
    if X.ndim == 3:
        return X, False
    raise ValueError(f"expected (N, M) or (B, N, M), got {tuple(X.shape)}")


def _tri_grid(N: int, B: int):
    def grid(meta):
        T = triton.cdiv(N, meta["BLOCK"])
        return (T * (T + 1) // 2, B)

    return grid


# --------------------------------------------------------------------------------------
# Long-K SYRK (C5-01)
# --------------------------------------------------------------------------------------
# One K loop of FP32 Tensor-Core accumulates loses low bits of every product once the
# accumulator outgrows it, so the diagonal of a Gram matrix drifts down by about 5e-9 per
# product (C4-03: -15.6% at K = 2^25). Above the gate the Gram matrix is the FP64 sum of
# chunk Gram matrices instead; at or below it, the single-launch kernel runs unchanged.
SYRK_CHUNK_GATE = 65536  # K above this is chunked
SYRK_CHUNK_SIZE = 4096  # columns per chunk: the FP32 accumulation length of one partial
SYRK_CHUNK_MAX_BYTES = 1 << 28  # FP32 partial Gram matrices per launch: at most 256 MiB
_MAX_GRID_Y = 65535  # CUDA limit of gridDim.y: one chunk Gram matrix per grid-y program
_INT32_LIMIT = 1 << 31
_MAX_BK = max(config.kwargs["BK"] for config in _CONFIGS)


def _direct_launch(kernel, grid, *args):
    kernel[grid](*args)


@triton.jit
def _cast_fp64_to_half_rn_kernel(src_ptr, dst_ptr, numel, IS_BF16: tl.constexpr, BLOCK: tl.constexpr):
    """dst = src rounded once to nearest even, from FP64 bits (int64 view) to FP16 / BF16 bits (int16 view).

    Integer arithmetic on the raw words, so there is no intermediate FP32 rounding. ``sig`` is the
    53-bit significand, ``e`` the unbiased exponent (FP64 subnormals use -1022). The result in units
    of the destination quantum is ``sig >> shift``, rounded on the dropped bits; ``shift`` grows by
    ``EMIN - e`` below the smallest normal and is clamped at 54, where every dropped value is below
    half a quantum (shifts stay in [42, 54], so they are defined for 64-bit words). Adding the
    rounded significand to ``(e - EMIN) << P`` lets a carry move into the next binade, from the
    largest subnormal into the smallest normal, and from the largest finite value into infinity
    (clamped to the infinity word). NaN becomes the quiet NaN of the destination.
    """
    P: tl.constexpr = 7 if IS_BF16 else 10  # stored significand bits
    EMIN: tl.constexpr = -126 if IS_BF16 else -14  # smallest normal exponent
    INF: tl.constexpr = 0x7F80 if IS_BF16 else 0x7C00
    QNAN: tl.constexpr = 0x7FC0 if IS_BF16 else 0x7E00
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    bits = tl.load(src_ptr + offs, mask=mask, other=0).to(tl.uint64, bitcast=True)
    sign = (bits >> 63) << 15
    field = (bits >> 52) & 0x7FF
    frac = bits & 0xFFFFFFFFFFFFF
    sig = tl.where(field != 0, frac | 0x10000000000000, frac)
    e = tl.maximum(field.to(tl.int64), 1) - 1023
    shift = tl.minimum(52 - P + tl.maximum(EMIN - e, 0), 54).to(tl.uint64)
    one = tl.full([BLOCK], 1, tl.uint64)
    kept = sig >> shift
    dropped = sig & ((one << shift) - 1)
    half = one << (shift - 1)
    up = (dropped > half) | ((dropped == half) & ((kept & 1) == 1))
    magnitude = (tl.maximum(e - EMIN, 0).to(tl.uint64) << P) + kept + up.to(tl.uint64)
    magnitude = tl.minimum(magnitude, INF)
    word = tl.where(field == 0x7FF, tl.where(frac != 0, QNAN, sign | INF), sign | magnitude)
    tl.store(dst_ptr + offs, word.to(tl.uint16).to(tl.int16, bitcast=True), mask=mask)


def _store_fp64_single_round(src: torch.Tensor, out: torch.Tensor) -> None:
    """Write the FP64 tensor ``src`` into ``out`` (same shape) with one IEEE-754 round to nearest even.

    PyTorch's CUDA ``copy_`` from FP64 to FP16 / BF16 rounds through FP32 (double rounding), so the
    two half formats use ``_cast_fp64_to_half_rn_kernel``; FP32 is a direct conversion. No host
    synchronization: the call is CUDA-graph capturable.
    """
    if out.dtype == torch.float32:
        out.copy_(src)
        return
    if out.dtype not in _HALF or src.dtype != torch.float64 or src.shape != out.shape or not out.is_contiguous():
        raise ValueError("need an FP64 source and a contiguous FP16 / BF16 output of the same shape")
    src = src.contiguous()
    numel = src.numel()
    if numel:
        block = 1024
        _cast_fp64_to_half_rn_kernel[(triton.cdiv(numel, block),)](
            src.view(torch.int64), out.view(torch.int16), numel, IS_BF16=out.dtype == torch.bfloat16, BLOCK=block)


def _syrk_chunked(view: torch.Tensor, out: torch.Tensor, launch_fn=None, grid_fn=None, *,
                  chunk: int = SYRK_CHUNK_SIZE, max_bytes: int = SYRK_CHUNK_MAX_BYTES) -> None:
    """Write the Gram matrices ``view @ view.mT`` of a (batch, n, k) view into ``out`` (batch, n, n).

    ``view`` may have any strides, for example ``X.mT`` of a tall ``X`` (``sxn = 1``, ``sxk = n``).
    ``out`` is contiguous float16, bfloat16 or float32. Each matrix is split into ``k // chunk``
    zero-copy strided chunks (count, n, chunk) plus a ``k % chunk`` remainder. One ``_syrk_kernel``
    launch computes up to ``min(65535, max(1, max_bytes // (4 n^2)))`` FP32 chunk Gram matrices (one
    per grid-y program; CUDA allows at most 65,535). They are summed in FP64 and the sum is rounded
    exactly once to ``out``: IEEE-754 round to nearest even, ties to even, with subnormals and
    overflow to infinity (``_store_fp64_single_round``; there is no intermediate FP32 rounding).
    Every partial is bitwise symmetric and the FP64 sum and the rounding are elementwise, so ``out``
    is bitwise symmetric. No host synchronization: the call is CUDA-graph capturable after one
    eager warm-up per shape.

    ``launch_fn(kernel, grid, *args)`` and ``grid_fn(n, batch)`` default to a direct Triton launch
    on the triangle grid; ``polar_switch_blackwell`` passes its cached launcher.
    """
    launch_fn = launch_fn or _direct_launch
    grid_fn = grid_fn or _tri_grid
    batch, n, k = view.shape
    if chunk <= 0 or out.shape != (batch, n, n) or not out.is_contiguous():
        raise ValueError("need chunk > 0 and a contiguous (batch, n, n) output")
    sxk = abs(view.stride(2))
    if _MAX_BK * sxk >= _INT32_LIMIT > sxk:  # see the int64 audit above _syrk_kernel
        raise ValueError(f"column stride {sxk} overflows the int32 in-tile offsets of _syrk_kernel")
    count, rest = divmod(k, chunk)
    step = min(_MAX_GRID_Y, max(1, max_bytes // (4 * n * n)))
    total = torch.zeros((batch, n, n), dtype=torch.float64, device=view.device)
    for index in range(batch):
        matrix = view[index]
        # (n, count * chunk) -> (count, n, chunk): column block c of the matrix is chunk c.
        chunks = matrix[:, :count * chunk].unflatten(1, (count, chunk)).transpose(0, 1)
        pieces = [chunks[s:s + step] for s in range(0, count, step)]
        if rest:
            pieces.append(matrix[None, :, count * chunk:])
        for piece in pieces:
            parts = torch.empty((len(piece), n, n), dtype=torch.float32, device=view.device)
            launch_fn(_syrk_kernel, grid_fn(n, len(piece)), piece, parts, n, piece.shape[-1], *piece.stride())
            total[index] += parts.sum(0, dtype=torch.float64)
    _store_fp64_single_round(total, out)


def triton_syrk(X: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Symmetric Gram matrix ``X @ X.mT`` for fp16/bf16 ``X`` of shape (N, M) or (B, N, M).

    Computes only upper-triangle tiles (~half of the MMA work), accumulates in fp32, rounds once
    to ``out_dtype`` (default: input dtype) and mirrors tiles so the output is exactly symmetric.
    For ``M > SYRK_CHUNK_GATE`` the FP32 partials of ``SYRK_CHUNK_SIZE`` columns are summed in
    FP64 (``_syrk_chunked``).
    """
    if X.dtype not in _HALF:
        raise TypeError(f"triton_syrk supports float16/bfloat16, got {X.dtype}")
    Xb, squeeze = _as_batched(X)
    B, N, K = Xb.shape
    out = torch.empty((B, N, N), dtype=out_dtype or X.dtype, device=X.device)
    if out.numel():
        if K > SYRK_CHUNK_GATE:
            _syrk_chunked(Xb, out)
        else:
            _syrk_kernel[_tri_grid(N, B)](Xb, out, N, K, Xb.stride(0), Xb.stride(1), Xb.stride(2))
    return out[0] if squeeze else out


def triton_gram_poly_step(A: torch.Tensor, b: float, c: float, a: float = 0.0) -> torch.Tensor:
    """Fused ``b*A + c*(A @ A) + a*I`` for a symmetric fp16/bf16 ``A`` of shape (N, N) or (B, N, N).

    ``A`` must be exactly symmetric (e.g. the output of :func:`triton_syrk`); the result is then
    exactly symmetric too.  fp32 accumulation, a single rounding on output.
    """
    if A.dtype not in _HALF:
        raise TypeError(f"triton_gram_poly_step supports float16/bfloat16, got {A.dtype}")
    Ab, squeeze = _as_batched(A)
    Ab = Ab.contiguous()
    B, N, N2 = Ab.shape
    if N != N2:
        raise ValueError("A must be square")
    out = torch.empty_like(Ab)
    if out.numel():
        _poly_kernel[_tri_grid(N, B)](Ab, out, N, float(b), float(c), float(a))
    return out[0] if squeeze else out


# --------------------------------------------------------------------------------------
# Circuits
# --------------------------------------------------------------------------------------
def _normalize(X: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Frobenius-normalize in fp32 (or fp64 when ``dtype==float64``), then cast to ``dtype``.

    Note: ``polar_express_triton`` intentionally normalizes in FP32, whereas
    ``baselines.polar_express`` matches upstream BF16 normalization.
    """
    Xf = X.double() if dtype == torch.float64 else X.float()
    nrm = Xf.norm(dim=(-2, -1), keepdim=True) * SAFETY + 1e-7
    return (Xf / nrm).to(dtype)


def _wide_view(X: torch.Tensor):
    """Return (wide matrix with N <= M, transposed?)."""
    if X.shape[-2] > X.shape[-1]:
        return X.mT.contiguous(), True
    return X.contiguous(), False


def polar_express_torch(X: torch.Tensor, steps: int = 5, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Plain PyTorch Gram-form PolarExpress (cuBLAS).  ``dtype=torch.float64`` gives the exact-arithmetic reference."""
    dtype = dtype or X.dtype
    Xb, transposed = _wide_view(_normalize(X, dtype))
    for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
        A = Xb @ Xb.mT
        B = b * A + c * (A @ A)
        Xb = a * Xb + B @ Xb
    return Xb.mT if transposed else Xb


def _triton_step(X: torch.Tensor, a: float, b: float, c: float, fold_identity: bool = False) -> torch.Tensor:
    """One Gram-form step on a batched, contiguous, wide (B, N, M) fp16/bf16 tensor."""
    A = triton_syrk(X)
    if fold_identity:  # B' = a*I + b*A + c*A^2 ; X <- B' X
        return torch.bmm(triton_gram_poly_step(A, b, c, a), X)
    return torch.baddbmm(X, triton_gram_poly_step(A, b, c), X, beta=a)


def polar_express_triton(
    X: torch.Tensor, steps: int = 5, dtype: torch.dtype | None = None, fold_identity: bool = False
) -> torch.Tensor:
    """PolarExpress with Triton syrk + fused polynomial kernels (fp16/bf16, fp32 accumulation)."""
    dtype = dtype or X.dtype
    Xb, squeeze = _as_batched(X)
    Xw, transposed = _wide_view(_normalize(Xb, dtype))
    for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
        Xw = _triton_step(Xw, a, b, c, fold_identity)
    out = Xw.mT if transposed else Xw
    return out[0] if squeeze else out


# ---- FP8 ------------------------------------------------------------------------------
def _fp8_quant(t: torch.Tensor):
    """Per-tensor dynamic e4m3 quantization.  Returns (q, scale) with ``t ~= q * scale``.

    ``scale`` stays a 0-dim fp32 CUDA tensor, so there is no host sync.
    """
    t = t.contiguous()
    amax = torch.linalg.vector_norm(t, ord=float("inf"), dtype=torch.float32).clamp_min_(1e-12)
    q = torch.empty(t.shape, dtype=_FP8, device=t.device)
    n = t.numel()
    _quant_kernel[(triton.cdiv(n, 4096),)](t, q, amax, n, BLOCK=4096)
    return q, amax / FP8_MAX


def _scaled_mm_nt(qa: torch.Tensor, sa, qb: torch.Tensor, sb, out_dtype) -> torch.Tensor:
    """``dequant(qa) @ dequant(qb).mT`` with both operands row-major (rows x K).

    ``_scaled_mm`` wants mat2 column-major, which is exactly ``qb.t()`` of a contiguous ``qb``.
    """
    return torch._scaled_mm(qa, qb.t(), sa, sb, out_dtype=out_dtype)


def _fp8_gram_step(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """One Gram-form step with all three GEMMs in FP8 (e4m3, fp32 accumulate / fp32 output).

    ``X`` is a contiguous (N, M) fp16/bf16 tensor, N <= M, N and M multiples of 16.  Intermediates
    ``A`` and ``B`` stay in fp32 between GEMMs; ``X`` itself is carried in the working dtype so the
    ``a*X`` term is exact and only the update ``B@X`` carries FP8 error.
    """
    qx, sx = _fp8_quant(X)
    A = _scaled_mm_nt(qx, sx, qx, sx, torch.float32)  # X X^T
    qa, sa = _fp8_quant(A)
    AA = _scaled_mm_nt(qa, sa, qa, sa, torch.float32)  # A A^T == A A (A symmetric)
    # B = b*A + c*AA = c * (AA + (b/c) * A): quantize the bracket, fold c into its scale.
    qb, sb = _fp8_quant(torch.add(AA, A, alpha=b / c))  # B symmetric, so B^T == B
    qxt = qx.t().contiguous()  # (M, N): mat2 of B @ X must be column-major
    BX = _scaled_mm_nt(qb, sb * c, qxt, sx, torch.float32)  # B @ (qxt^T) = B @ X
    out = torch.empty_like(X)
    _axpy_cast_kernel[(triton.cdiv(X.numel(), 4096),)](BX, X, out, a, X.numel(), BLOCK=4096)
    return out


def polar_express_fp8(
    X: torch.Tensor,
    steps: int = 5,
    fp8_steps: int = 1,
    dtype: torch.dtype | None = None,
    on_misaligned: str = "fallback",
    return_meta: bool = False,
):
    """Experimental FP8-early / BF16-late (or FP16-late) PolarExpress.

    Steps ``[0, fp8_steps)`` run their GEMMs in e4m3 through ``torch._scaled_mm`` with per-tensor
    dynamic scaling; the other steps use the Triton fp16/bf16 kernels with fp32 accumulation.
    ``_scaled_mm`` is unbatched, so batched inputs loop over the batch (each matrix has its own scale).

    Default ``fp8_steps=1`` keeps benign small-``n`` (``n <= 256``) uniform/Gaussian spectra bounded;
    ``fp8_steps >= 2`` (including with ``on_misaligned="pad"``) diverges on those inputs because e4m3
    error at step 1 pushes ``sigma_max`` past the polynomial stability basin.

    ``_scaled_mm`` requires both matrix dims to be multiples of 16.  If they are not:
      * ``on_misaligned="fallback"`` (default): run the whole circuit on the Triton bf16/fp16 path;
      * ``on_misaligned="pad"``: zero-pad rows/cols up to a multiple of 16 and crop the result.
    ``meta`` (when ``return_meta``) records which path actually ran and why.
    """
    dtype = dtype or X.dtype
    if dtype not in _HALF:
        raise TypeError("late-step dtype must be float16 or bfloat16")
    if on_misaligned not in ("fallback", "pad"):
        raise ValueError("on_misaligned must be 'fallback' or 'pad'")
    Xb, squeeze = _as_batched(X)
    Xw, transposed = _wide_view(_normalize(Xb, dtype))
    n, m = Xw.shape[-2:]
    meta = dict(
        path="fp8-early", dtype=str(dtype).replace("torch.", ""), steps=steps, fp8_steps_requested=fp8_steps,
        fp8_steps_used=min(fp8_steps, steps), orig_shape=tuple(X.shape), compute_shape=tuple(Xw.shape),
        transposed=transposed, padded_shape=None, reason=None,
    )

    pn, pm = -n % 16, -m % 16
    if (pn or pm) and fp8_steps > 0:
        if on_misaligned == "pad":
            Xw = F.pad(Xw, (0, pm, 0, pn))
            meta["padded_shape"] = tuple(Xw.shape)
        else:
            meta.update(path="fallback-bf16", fp8_steps_used=0,
                        reason=f"_scaled_mm needs dims % 16 == 0, got {n}x{m}")
    fp8_n = meta["fp8_steps_used"]

    for step, (a, b, c) in enumerate(POLAR_EXPRESS_COEFFS[:steps]):
        if step < fp8_n:
            Xw = torch.stack([_fp8_gram_step(x, a, b, c) for x in Xw])
        else:
            Xw = _triton_step(Xw, a, b, c)
    if meta["padded_shape"] is not None:
        Xw = Xw[..., :n, :m]
    out = Xw.mT if transposed else Xw
    out = out[0] if squeeze else out
    return (out, meta) if return_meta else out


# --------------------------------------------------------------------------------------
# Exact reference
# --------------------------------------------------------------------------------------
def polar_svd_fp64(X: torch.Tensor) -> torch.Tensor:
    """Exact polar factor ``U V^T`` of ``X`` (any shape (..., N, M)) computed in fp64 on the CPU."""
    U, _, Vh = torch.linalg.svd(X.detach().cpu().double(), full_matrices=False)
    return (U @ Vh).to(X.device)
