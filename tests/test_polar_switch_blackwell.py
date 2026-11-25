"""Tests for the Blackwell Triton Polar Switch and Gram Newton-Schulz (needs a CUDA GPU).

Two implementations of an FP16 circuit cannot agree bitwise: the FP32 accumulation order differs
between Triton and cuBLAS, and later steps amplify one-ulp differences. Equivalence is therefore
checked in two ways: the relative Frobenius distance to the eager result, and the error against
the FP64 SVD polar factor, which must equal the eager error within 3e-3.
"""

import pytest
import torch

from research.polar_opt.baselines import _normalized_fp16, gram_ns, polar_switch, svd_polar_fp64
from research.polar_opt.harness_tier1 import synthetic_cases
from research.polar_opt.polar_switch_blackwell import (
    _gram,
    _prepare,
    _symprod,
    gram_ns_blackwell,
    polar_switch_blackwell,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

POW2 = [(256, 1024), (1024, 4096), (512, 512)]
ODD = [(127, 513), (515, 1029), (255, 255)]
TALL = [(1024, 256), (513, 127), (1029, 515)]
BATCHED = [(4, 512, 2048), (3, 300, 100), (2, 3, 64, 200)]
SHAPES = POW2 + ODD + TALL + BATCHED
VARIANTS = ["PS2_55", "PS3_55"]
BF16, FP16 = torch.bfloat16, torch.float16
LOW_PRECISION = [
    pytest.param(dict(dtype=BF16), id="bf16"),
    pytest.param(dict(fp8_sketch=True), id="fp8-sketch+fp16"),
    pytest.param(dict(dtype=BF16, fp8_sketch=True), id="fp8-sketch+bf16"),
]
# Measured distances to eager are at most 3e-3 (Polar Switch) and 1.1e-2 (Gram Newton-Schulz).
EAGER_TOL = {"polar_switch": 2e-2, "gram_ns": 3e-2}
ERROR_TOL = 3e-3


def rand(shape, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, device="cuda", generator=generator)


def rel(a, reference):
    """Largest per-matrix relative Frobenius distance."""
    a, reference = a.double(), reference.double()
    return ((a - reference).norm(dim=(-2, -1)) / reference.norm(dim=(-2, -1))).max().item()


def check_output(Q, G):
    assert Q.shape == G.shape and Q.dtype == G.dtype
    assert torch.isfinite(Q).all(), "NaN/Inf in output"


def sigma_max(Q):
    return torch.linalg.svdvals(Q.double().cpu()).max().item()


def sigma_bound(name, options):
    """Largest singular value allowed. Eager reaches 1.11 (Polar Switch) and 1.15 (Gram NS).

    BF16 Gram Newton-Schulz is the exception. Its Gram-space recurrence overshoots to 1.3-1.8 on
    ill-conditioned inputs. The eager cuBLAS recurrence does the same in BF16 (1.5-1.65), so the
    cause is the 8-bit mantissa, not the kernels. The bound there is the harness divergence limit.
    """
    return 2.0 if name == "gram_ns" and options.get("dtype") == BF16 else 1.25


def two_level_rank_one(n=384, m=1152):
    """Rank one with entries of two magnitudes: FP8 rounding biases its Gram matrix upward."""
    generator = torch.Generator().manual_seed(1)
    u = torch.where(torch.rand(n, generator=generator) < 0.5, 1.0, 0.53)
    v = torch.where(torch.rand(m, generator=generator) < 0.5, 1.0, 0.53)
    return (u[:, None] * v[None, :]).cuda()


# ------------------------------------------------------------------ equivalence to eager
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_polar_switch_matches_eager_and_svd(shape, variant):
    G = rand(shape)
    Q, eager, P = polar_switch_blackwell(G, variant), polar_switch(G, variant), svd_polar_fp64(G)
    check_output(Q, G)
    assert rel(Q, eager) < EAGER_TOL["polar_switch"]
    assert abs(rel(Q, P) - rel(eager, P)) < ERROR_TOL
    assert rel(Q, P) < 0.2


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_gram_ns_matches_eager_and_svd(shape):
    G = rand(shape)
    Q, eager, P = gram_ns_blackwell(G), gram_ns(G), svd_polar_fp64(G)
    check_output(Q, G)
    assert rel(Q, eager) < EAGER_TOL["gram_ns"]
    assert abs(rel(Q, P) - rel(eager, P)) < ERROR_TOL
    assert rel(Q, P) < 0.2


@pytest.mark.parametrize("steps,resets", [(0, (2,)), (1, (2,)), (3, (1,)), (5, ()), (7, (2, 4)), (5, (0, 3))])
@pytest.mark.parametrize("shape", [(515, 1029), (1029, 515), (255, 255)], ids=str)
def test_gram_ns_steps_and_restarts_match_eager(shape, steps, resets):
    G = rand(shape, seed=1)
    Q = gram_ns_blackwell(G, steps, resets)
    check_output(Q, G)
    assert rel(Q, gram_ns(G, steps, resets)) < EAGER_TOL["gram_ns"]


# ------------------------------------------------------------------ BF16 and FP8 sketch
@pytest.mark.parametrize("options", LOW_PRECISION)
@pytest.mark.parametrize("shape", [(256, 1024), (515, 1029), (1029, 515), (4, 512, 2048)], ids=str)
def test_low_precision_variants_keep_polar_accuracy(shape, options):
    G = rand(shape, seed=2)
    P = svd_polar_fp64(G)
    for variant in VARIANTS:
        Q = polar_switch_blackwell(G, variant, **options)
        check_output(Q, G)
        assert rel(Q, P) < rel(polar_switch(G, variant), P) + 0.01
    Q = gram_ns_blackwell(G, **options)
    check_output(Q, G)
    assert rel(Q, P) < rel(gram_ns(G), P) + 0.01


@pytest.mark.parametrize("options", LOW_PRECISION)
def test_low_precision_variants_on_square_input(options):
    # A square Gaussian matrix has singular values near zero. The FP8 sketch rotates those
    # directions, which costs about 0.02 of polar error. It must stay bounded.
    G = rand((512, 512), seed=3)
    P = svd_polar_fp64(G)
    for variant in VARIANTS:
        Q = polar_switch_blackwell(G, variant, **options)
        check_output(Q, G)
        assert rel(Q, P) < rel(polar_switch(G, variant), P) + 0.04
    assert rel(gram_ns_blackwell(G, **options), P) < rel(gram_ns(G), P) + 0.04


# ------------------------------------------------------------------ hard spectra, no NaN/Inf
HARD = list(synthetic_cases([128], [1, 4], seed=123))


@pytest.mark.parametrize("options", [pytest.param({}, id="fp16"), *LOW_PRECISION])
@pytest.mark.parametrize("case", HARD, ids=lambda case: case.name)
def test_hard_spectra_stay_finite_and_bounded(case, options):
    """Power-law, geometric, spike, and rank-one spectra: finite, with bounded singular values.

    FP16 must reproduce the eager error (measured difference at most 8e-4). BF16 and the FP8
    sketch are not equivalent to eager here: at n = 128 they add up to 0.07 of polar error on
    steep spectra (0.045 at n = 512). The margin only asserts that the loss stays bounded.
    """
    G = case.matrix.cuda()
    P = svd_polar_fp64(G)
    margin = 0.10 if options else 5e-3
    for name, blackwell, eager in (
        ("PS2_55", lambda: polar_switch_blackwell(G, "PS2_55", **options), lambda: polar_switch(G)),
        ("PS3_55", lambda: polar_switch_blackwell(G, "PS3_55", **options),
         lambda: polar_switch(G, "PS3_55")),
        ("gram_ns", lambda: gram_ns_blackwell(G, **options), gram_ns),
    ):
        Q = blackwell()
        check_output(Q, G)
        assert sigma_max(Q) < sigma_bound(name, options), name
        assert rel(Q, P) < rel(eager() if name != "gram_ns" else eager(G), P) + margin, name


@pytest.mark.parametrize("options", [pytest.param({}, id="fp16"), *LOW_PRECISION])
def test_degenerate_inputs_stay_finite(options):
    spike = rand((96, 384), seed=4) * 1e-3
    spike[5, 9] = 1.0
    for G in (torch.zeros(96, 384, device="cuda"), rand((96, 384)) * 1e-9, rand((96, 384)) * 1e6,
              spike, two_level_rank_one()):
        for variant in VARIANTS:
            Q = polar_switch_blackwell(G, variant, **options)
            check_output(Q, G)
            assert sigma_max(Q) < sigma_bound(variant, options)
        Q = gram_ns_blackwell(G, **options)
        check_output(Q, G)
        assert sigma_max(Q) < sigma_bound("gram_ns", options)
    assert not polar_switch_blackwell(torch.zeros(96, 384, device="cuda"), **options).any()


def test_fp8_sketch_is_rescaled_to_the_exact_trace():
    """Without the rescale, FP8 rounding pushes the top eigenvalue above 1 and the circuit diverges."""
    G = two_level_rank_one()
    X, norm = _prepare(G, FP16)
    sketch = X.new_empty(1, 384, 384)
    _gram(X, sketch, norm)
    energy = X.double().square().sum().item()
    assert sketch.double().diagonal(dim1=-2, dim2=-1).sum().item() == pytest.approx(energy, rel=2e-3)
    assert torch.linalg.eigvalsh(sketch.double()).max().item() <= energy * (1 + 2e-3)
    assert (sketch - sketch.mT).abs().max().item() == 0.0
    # The raw FP8 Gram matrix of this input overestimates the only nonzero eigenvalue by 0.9
    # percent. The Polar Switch polynomial is 0.25 at eigenvalue 1.0 and 2.6 at 1.01.
    scale = X.abs().max().item() / 448
    quantized = (X.float() / scale).to(torch.float8_e4m3fn).double() * scale
    assert torch.linalg.eigvalsh(quantized[0] @ quantized[0].mT).max().item() > 1.005 * energy


# ------------------------------------------------------------------ kernels and plumbing
@pytest.mark.parametrize("dtype", [FP16, BF16], ids=str)
@pytest.mark.parametrize("batch,n", [(1, 127), (3, 200), (1, 1024)])
def test_symprod_kernel_matches_fp64(batch, n, dtype):
    S = rand((batch, n, n), seed=5)
    S = (S + S.mT) / (4 * n**0.5)
    A = S.to(dtype)
    B = (0.5 * (S @ S) + 0.25 * S).to(dtype)
    B = torch.triu(B) + torch.triu(B, 1).mT  # exactly symmetric, and it commutes with A up to rounding
    C = B.clone()
    out = torch.empty_like(A)
    _symprod(A, B, C, out, 1.5, -0.75, 0.3)
    eye = torch.eye(n, dtype=torch.float64, device="cuda")
    reference = 1.5 * (A.double() @ B.double()) - 0.75 * C.double() + 0.3 * eye
    reference = torch.triu(reference) + torch.triu(reference, 1).mT
    tolerance = 1e-2 if dtype == BF16 else 2e-3
    assert rel(out, reference) < tolerance
    assert (out - out.mT).abs().max().item() == 0.0
    _symprod(A, B, C, C, 1.5, -0.75, 0.3)  # the output can alias the additive operand
    assert torch.equal(C, out)
    with pytest.raises(ValueError):
        _symprod(A, B, C, A, 1.0, 1.0)


@pytest.mark.parametrize("shape", [(127, 513), (1029, 515), (4, 512, 2048)], ids=str)
def test_prepare_matches_eager_normalization(shape):
    G = rand(shape, seed=6)
    X, norm = _prepare(G, FP16)
    eager = _normalized_fp16(G)
    assert X.shape == eager.shape and X.is_contiguous()
    # The kernel's FP32 division can differ by one ulp before the single rounding to FP16.
    assert rel(X, eager) < 1e-5
    assert torch.allclose(norm, G.reshape(eager.shape).norm(dim=(-2, -1)))
    assert rel(_prepare(G, BF16)[0], eager) < 5e-3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64], ids=str)
def test_input_dtype_is_restored_and_input_is_not_modified(dtype):
    G = rand((200, 600), seed=7).to(dtype)
    before = G.clone()
    P = svd_polar_fp64(G)
    for Q in (polar_switch_blackwell(G), polar_switch_blackwell(G, "ps3-55"), gram_ns_blackwell(G),
              polar_switch_blackwell(G, fp8_sketch=True)):
        check_output(Q, G)
        assert rel(Q, P) < 0.12
    assert torch.equal(G, before)


def test_batched_result_equals_per_matrix_results():
    G = rand((2, 3, 64, 200), seed=8)
    for function in (polar_switch_blackwell, gram_ns_blackwell):
        batched = function(G)
        for index in ((0, 0), (1, 2)):
            assert rel(batched[index], function(G[index])) < 5e-3


def test_first_call_and_cached_launch_agree_bitwise():
    # The first call on a new shape runs the autotuner on kernels that update a buffer in place.
    G = rand((97, 331), seed=9)
    for function in (polar_switch_blackwell, gram_ns_blackwell):
        first = function(G)
        assert torch.equal(first, function(G))
        assert torch.equal(first, function(G.clone()))


def test_noncontiguous_and_offset_inputs():
    base = rand((301, 2, 96), seed=10)
    G = base[1:, 0].mT  # (96, 300) view: transposed, strided, and offset
    Q = polar_switch_blackwell(G)
    check_output(Q, G)
    assert rel(Q, polar_switch_blackwell(G.contiguous())) < 5e-3
    assert rel(gram_ns_blackwell(G), gram_ns_blackwell(G.contiguous())) < 5e-3


def test_empty_and_invalid_inputs():
    assert polar_switch_blackwell(torch.empty(0, 8, device="cuda")).shape == (0, 8)
    assert gram_ns_blackwell(torch.empty(3, 0, 8, device="cuda")).shape == (3, 0, 8)
    G = rand((32, 64))
    with pytest.raises(ValueError):
        polar_switch_blackwell(G, "PS4_55")
    with pytest.raises(TypeError):
        polar_switch_blackwell(G, dtype=torch.float32)
    with pytest.raises(ValueError):
        polar_switch_blackwell(G.cpu())
    with pytest.raises(ValueError):
        gram_ns_blackwell(G[0])


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (64, 1)], ids=str)
def test_min_dim_one_shapes_supported(shape):
    G = rand(shape, seed=11)
    for variant in VARIANTS:
        Q = polar_switch_blackwell(G, variant)
        check_output(Q, G)
        assert rel(Q, polar_switch(G, variant)) < EAGER_TOL["polar_switch"]
    Qg = gram_ns_blackwell(G)
    check_output(Qg, G)
    assert rel(Qg, gram_ns(G)) < EAGER_TOL["gram_ns"]


def test_torch_compile_does_not_poison_launch_cache():
    from research.polar_opt.polar_switch_blackwell import _PLANS

    polar_switch_blackwell(rand((256, 1024), seed=12), "PS2_55")
    G = rand((128, 512), seed=13)
    compiled_out = torch.compile(lambda x: polar_switch_blackwell(x, "PS2_55"))(G)
    torch._dynamo.reset()
    eager_out = polar_switch_blackwell(G, "PS2_55")
    check_output(eager_out, G)
    assert torch.equal(compiled_out, eager_out)
    assert all(plan[0] is not None and hasattr(plan[0], "__getitem__") for plan in _PLANS.values())


def test_fp8_sketch_huge_scale_returns_zero_not_nan():
    G = rand((64, 128), seed=14) * 1e25
    Q = polar_switch_blackwell(G, "PS2_55", fp8_sketch=True)
    check_output(Q, G)
    assert Q.abs().max().item() == 0.0
