"""Correctness tests for the Blackwell Triton / FP8 polar kernels (needs a CUDA GPU).

Kernel checks compare against fp64 references of the *same* operation, so tolerances are set by
fp16/bf16 rounding only.  Circuit checks compare against (a) the same circuit in fp64 and
(b) the exact SVD polar factor (whose gap is dominated by the 5-step truncation, ~0.09).
"""

import pytest
import torch

from research.polar_opt.kernels_blackwell import (
    POLAR_EXPRESS_COEFFS,
    polar_express_fp8,
    polar_express_torch,
    polar_express_triton,
    polar_svd_fp64,
    triton_gram_poly_step,
    triton_syrk,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DTYPES = [torch.bfloat16, torch.float16]
# unit roundoff * generous constant for fp32-accumulated products of unit-scale data
KERNEL_TOL = {torch.bfloat16: 1e-2, torch.float16: 2e-3}
POW2 = [(256, 1024), (1024, 4096), (2048, 8192)]
ODD = [(127, 513), (515, 1029), (1024, 3072)]
BATCHED = [(4, 512, 2048), (8, 256, 1024)]
TALL = [(1024, 256), (3072, 1024), (1029, 515)]
SHAPES = POW2 + ODD + BATCHED + TALL


def rand(shape, dtype, scale=1.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=g) * scale).to(dtype)


def rel(a, ref):
    return ((a.double() - ref).norm(dim=(-2, -1)) / ref.norm(dim=(-2, -1))).max().item()


def check_clean(t):
    assert torch.isfinite(t.float()).all(), "NaN/Inf in output"


# ---------------------------------------------------------------- triton_syrk
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_syrk_matches_fp64_and_is_exactly_symmetric(shape, dtype):
    X = rand(shape, dtype, scale=0.1)
    A = triton_syrk(X)
    check_clean(A)
    assert A.dtype == dtype and A.shape == (*shape[:-1], shape[-2])
    ref = X.double() @ X.double().mT
    assert rel(A, ref) < KERNEL_TOL[dtype]
    assert (A - A.mT).abs().max().item() == 0.0


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_syrk_is_as_accurate_as_cublas(dtype):
    X = rand((1024, 4096), dtype, scale=0.1)
    ref = X.double() @ X.double().mT
    assert rel(triton_syrk(X), ref) <= 1.5 * rel(X @ X.mT, ref) + 1e-6


def test_syrk_fp32_output_and_strided_input():
    X = rand((300, 700), torch.bfloat16, scale=0.1)
    ref = X.double() @ X.double().mT
    A = triton_syrk(X, out_dtype=torch.float32)  # fp32 out: no output rounding at all
    assert A.dtype == torch.float32 and rel(A, ref) < 1e-5 and (A - A.mT).abs().max() == 0
    Xt = rand((700, 300), torch.bfloat16, scale=0.1).mT  # non-contiguous view
    assert rel(triton_syrk(Xt), Xt.double() @ Xt.double().mT) < KERNEL_TOL[torch.bfloat16]


def test_syrk_rejects_fp32():
    with pytest.raises(TypeError):
        triton_syrk(torch.randn(16, 16, device="cuda"))


# ---------------------------------------------------------- fused polynomial
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("shape", [(256, 1024), (127, 513), (515, 1029), (4, 512, 2048), (1024, 3072)], ids=str)
@pytest.mark.parametrize("identity", [0.0, 1.7])
def test_poly_step_matches_fp64(shape, dtype, identity):
    # Normalize so that ||A||_2 ~ 1: keeps A @ A well inside the fp16 range.
    X = rand(shape, dtype, scale=1.0)
    X = (X.float() / X.float().norm(dim=(-2, -1), keepdim=True)).to(dtype)
    A = triton_syrk(X)
    b, c = -2.9478499167379106, 0.5448431082926601
    P = triton_gram_poly_step(A, b, c, identity)
    check_clean(P)
    Ad = A.double()
    n = Ad.shape[-1]
    ref = b * Ad + c * (Ad @ Ad) + identity * torch.eye(n, dtype=torch.float64, device="cuda")
    assert P.dtype == dtype
    assert rel(P, ref) < KERNEL_TOL[dtype]
    assert (P - P.mT).abs().max().item() == 0.0


# ------------------------------------------------------------------ circuits
def exact_polar(X):
    return polar_svd_fp64(X.float())


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_triton_circuit_matches_fp64_circuit_and_svd(shape, dtype):
    X32 = rand(shape, torch.float32)
    X = X32.to(dtype)
    Q = polar_express_triton(X)
    check_clean(Q)
    assert Q.shape == X.shape and Q.dtype == dtype
    ref64 = polar_express_torch(X32, dtype=torch.float64)  # same algorithm, exact arithmetic
    assert rel(Q, ref64) < 3e-2  # ~ rounding of 5 half-precision steps
    # Never meaningfully worse than the cuBLAS half-precision implementation.
    assert rel(Q, ref64) <= 1.5 * rel(polar_express_torch(X), ref64) + 1e-3
    # Against the true polar factor the 5-step truncation (~0.09) dominates.
    assert rel(Q, exact_polar(X32)) < 0.15


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_triton_circuit_fold_identity_equivalent(dtype):
    X = rand((512, 2048), dtype)
    ref64 = polar_express_torch(X.float(), dtype=torch.float64)
    assert rel(polar_express_triton(X, fold_identity=True), ref64) < 3e-2


def test_coefficients_converge_in_fp64():
    """Sanity check of the coefficient table: 8 exact steps give an orthogonal matrix."""
    X = rand((256, 1024), torch.float64)
    Q = polar_express_torch(X, steps=len(POLAR_EXPRESS_COEFFS), dtype=torch.float64)
    eye = torch.eye(256, dtype=torch.float64, device="cuda")
    assert (Q @ Q.mT - eye).norm().item() < 1e-6


# ------------------------------------------------------------ FP8-early circuit
# Shapes with min(n, m) >= 256 are stable with two FP8 steps; small n needs fp8_steps=1
# (see test_fp8_small_matrix_one_step).  Dims are multiples of 16 so FP8 really runs.
FP8_ALIGNED = [(256, 1024), (1024, 4096), (2048, 8192), (1024, 3072), (4, 512, 2048), (8, 256, 1024), (3072, 1024)]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("fp8_steps", [1, 2])
@pytest.mark.parametrize("shape", FP8_ALIGNED, ids=str)
def test_fp8_early_matches_half_precision_path(shape, fp8_steps, dtype):
    X32 = rand(shape, torch.float32)
    X = X32.to(dtype)
    Q, meta = polar_express_fp8(X, fp8_steps=fp8_steps, return_meta=True)
    check_clean(Q)
    assert meta["path"] == "fp8-early" and meta["fp8_steps_used"] == fp8_steps
    assert meta["reason"] is None and meta["padded_shape"] is None
    assert Q.shape == X.shape and Q.dtype == dtype
    ref64 = polar_express_torch(X32, dtype=torch.float64)
    base = rel(polar_express_triton(X), ref64)
    # FP8 error (~1-3% per FP8 step at n=256) is only partly absorbed by the later polishing steps.
    assert rel(Q, ref64) <= base + 0.035 * fp8_steps
    assert rel(Q, exact_polar(X32)) < 0.15


def test_fp8_zero_steps_equals_triton_path():
    X = rand((256, 1024), torch.bfloat16)
    Q, meta = polar_express_fp8(X, fp8_steps=0, return_meta=True)
    assert meta["fp8_steps_used"] == 0
    assert torch.equal(Q, polar_express_triton(X))


@pytest.mark.parametrize("shape", [(127, 513), (515, 1029), (130, 1000), (1024, 3075)], ids=str)
def test_fp8_misaligned_falls_back_with_diagnostics(shape):
    X = rand(shape, torch.bfloat16)
    Q, meta = polar_express_fp8(X, fp8_steps=2, return_meta=True)  # default: fallback
    assert meta["path"] == "fallback-bf16" and meta["fp8_steps_used"] == 0
    assert "% 16" in meta["reason"] and meta["padded_shape"] is None
    check_clean(Q)
    assert torch.equal(Q, polar_express_triton(X))  # exactly the non-FP8 result


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("shape", [(127, 513), (515, 1029), (1029, 515)], ids=str)
def test_fp8_misaligned_pad_mode(shape, dtype):
    X32 = rand(shape, torch.float32)
    X = X32.to(dtype)
    Q, meta = polar_express_fp8(X, fp8_steps=1, on_misaligned="pad", return_meta=True)
    n, m = sorted(shape)
    assert meta["path"] == "fp8-early" and meta["fp8_steps_used"] == 1
    assert meta["padded_shape"][-2:] == (-(-n // 16) * 16, -(-m // 16) * 16)
    assert Q.shape == X.shape
    check_clean(Q)
    ref64 = polar_express_torch(X32, dtype=torch.float64)
    assert rel(Q, ref64) <= rel(polar_express_triton(X), ref64) + 0.03


@pytest.mark.parametrize("shape", [(64, 512), (127, 513), (128, 528)], ids=str)
def test_fp8_small_matrix_one_step(shape):
    """With few rows, one FP8 step is safe; two steps can leave the polynomial's stable region."""
    X32 = rand(shape, torch.float32)
    X = X32.to(torch.bfloat16)
    Q = polar_express_fp8(X, fp8_steps=1, on_misaligned="pad")
    check_clean(Q)
    assert rel(Q, exact_polar(X32)) < 0.15


def test_fp8_rejects_fp32_dtype():
    with pytest.raises(TypeError):
        polar_express_fp8(torch.randn(64, 256, device="cuda"), dtype=torch.float32)
