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


def triton_syrk(X: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Symmetric Gram matrix ``X @ X.mT`` for fp16/bf16 ``X`` of shape (N, M) or (B, N, M).

    Computes only upper-triangle tiles (~half of the MMA work), accumulates in fp32, rounds once
    to ``out_dtype`` (default: input dtype) and mirrors tiles so the output is exactly symmetric.
    """
    if X.dtype not in _HALF:
        raise TypeError(f"triton_syrk supports float16/bfloat16, got {X.dtype}")
    Xb, squeeze = _as_batched(X)
    B, N, K = Xb.shape
    out = torch.empty((B, N, N), dtype=out_dtype or X.dtype, device=X.device)
    if out.numel():
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
