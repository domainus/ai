import math
from functools import partial

import pytest
import torch

from research.polar_opt.baselines import (
    gram_ns,
    jordan_ns5,
    optimal_composition,
    optimal_cubic,
    optimal_quintic,
    polar_express,
    polar_switch,
    svd_polar_fp64,
    theoretical_cost,
)


METHODS = [
    pytest.param(jordan_ns5, torch.bfloat16, 0.46, id="jordan"),
    pytest.param(polar_express, torch.bfloat16, 0.20, id="polar-express"),
    pytest.param(gram_ns, torch.float32, 0.25, id="gram-ns"),
    pytest.param(polar_switch, torch.float32, 0.16, id="ps2"),
    pytest.param(partial(polar_switch, variant="PS3_55"), torch.float32, 0.16, id="ps3"),
]


@pytest.fixture(params=[
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable"
    )),
])
def device(request):
    return request.param


def _matrix(shape, device, condition=2):
    rows, cols = shape[-2:]
    batch = shape[:-2]
    n = min(rows, cols)
    generator = torch.Generator().manual_seed(227)
    u = torch.linalg.qr(torch.randn(*batch, rows, n, generator=generator, dtype=torch.float64)).Q
    v = torch.linalg.qr(torch.randn(*batch, cols, n, generator=generator, dtype=torch.float64)).Q
    s = torch.logspace(0, -math.log10(condition), n, dtype=torch.float64)
    return ((u * s) @ v.mT).float().to(device)


@pytest.mark.parametrize("shape", [(8, 8), (8, 19), (19, 8), (2, 3, 7, 16)])
def test_fp64_reduced_svd(shape, device):
    g = _matrix(shape, device, condition=1e6)
    factor = svd_polar_fp64(g)
    assert factor.dtype == torch.float64
    assert factor.shape == shape
    assert torch.linalg.svdvals(factor).sub(1).abs().amax().item() < 1e-12
    oriented_g, oriented_q = (g.mT, factor.mT) if shape[-2] > shape[-1] else (g, factor)
    positive = oriented_g.double() @ oriented_q.mT
    torch.testing.assert_close(positive, positive.mT, atol=1e-12, rtol=1e-12)
    assert torch.linalg.eigvalsh(positive).amin().item() > 0


@pytest.mark.parametrize("method,dtype,error_limit", METHODS)
@pytest.mark.parametrize("shape", [(11, 11), (7, 21), (28, 7), (2, 3, 5, 17)])
def test_baseline_polar_error(method, dtype, error_limit, shape, device):
    matrix = _matrix(shape, device)
    before = matrix.clone()
    result = method(matrix)
    exact = svd_polar_fp64(matrix)
    assert result.dtype == dtype
    assert result.shape == matrix.shape
    assert torch.isfinite(result).all()
    errors = (result.double() - exact).norm(dim=(-2, -1)) / math.sqrt(min(shape[-2:]))
    assert errors.amax().item() < error_limit
    torch.testing.assert_close(matrix, before, atol=0, rtol=0)


@pytest.mark.parametrize("method,dtype,error_limit", METHODS)
def test_transpose_and_batch_equivariance(method, dtype, error_limit, device):
    matrix = _matrix((2, 7, 31), device)
    result = method(matrix).float()
    transposed = method(matrix.mT).mT.float()
    batch_members = torch.stack([method(item).float() for item in matrix])
    torch.testing.assert_close(result, transposed, atol=0.016, rtol=0.02)
    torch.testing.assert_close(result, batch_members, atol=0.016, rtol=0.02)


@pytest.mark.parametrize("method,dtype,error_limit", METHODS)
def test_zero_rank_one_and_ill_conditioned_inputs_are_finite(method, dtype, error_limit, device):
    ill = _matrix((12, 96), device, condition=1e6)
    rank_one = torch.ones_like(ill)
    zero = torch.zeros_like(ill)
    output = method(torch.stack([zero, rank_one, ill]))
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[0]).item() == 0
    assert torch.linalg.svdvals(output.float())[1, 0].item() > 0.5


def test_remez_coefficients_match_published_configuration():
    coefficients = optimal_composition(1e-3, 10, degree=5, safety_factor_eps=1e-2, cushion=0.02)
    expected_prefix = [
        (8.237312490495555, -23.157747414558187, 16.6805684114459),
        (4.082441999064831, -2.8930477353325843, 0.5252849256975642),
        (3.926347992254648, -2.854746803476524, 0.531802242289498),
        (3.298218713308519, -2.42454198102671, 0.48632008358844137),
        (2.297036943455259, -1.636625581259032, 0.4002628455953629),
    ]
    assert len(coefficients) == 10
    for actual, expected in zip(coefficients, expected_prefix):
        assert actual == pytest.approx(expected, rel=1e-12)
    assert optimal_quintic(1, 1) == pytest.approx((15 / 8, -10 / 8, 3 / 8))
    assert optimal_cubic(1, 1) == pytest.approx((1.5, -0.5))


def test_gram_restart_changes_the_finite_precision_path(device):
    matrix = _matrix((12, 36), device, condition=100)
    restarted = gram_ns(matrix, reset_iterations=(2,))
    unbroken = gram_ns(matrix, reset_iterations=())
    assert not torch.equal(restarted, unbroken)
    exact = svd_polar_fp64(matrix)
    for candidate in (restarted, unbroken):
        assert torch.isfinite(candidate).all()
        assert (candidate - exact).norm().item() / math.sqrt(12) < 0.30


def test_polar_switch_variants_are_distinct(device):
    matrix = _matrix((12, 24), device)
    assert not torch.equal(polar_switch(matrix, "PS2_55"), polar_switch(matrix, "PS3_55"))
    with pytest.raises(ValueError, match="Unsupported Polar Switch variant"):
        polar_switch(matrix, "other")


def test_theoretical_costs_distinguish_dense_and_symmetric_work():
    n, m = 16, 128
    n3 = n**3
    direct = theoretical_cost("polar_express", (n, m))
    gram = theoretical_cost("gram_ns", (n, m))
    ps2 = theoretical_cost("PS2_55", (n, m))
    ps3 = theoretical_cost("PS3_55", (3, n, m))
    assert direct.flops == (20 * 8 + 10) * n3
    assert direct.symmetric_flops == (15 * 8 + 5) * n3
    assert gram.flops == (8 * 8 + 28) * n3
    assert gram.symmetric_flops == (6 * 8 + 14) * n3
    assert ps2.symmetric_flops == (6 * 8 + 10) * n3
    assert ps3.flops == 3 * ps2.flops
    assert ps3.peak_intermediate_bytes >= 3 * ps2.peak_intermediate_bytes
