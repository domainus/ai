"""Tests for the Schatten-8 pre-scaled Polar Switch circuits (needs a CUDA GPU).

The native path (Triton kernels, FP16 buffers) is checked against two references: the float64
matrix reference ``schatten_switch_reference`` and the float64 scalar map of each table. The
Cycle 1 audit findings are regression tests: R1 (subnormal Gram entries at n = 4096 and the
rigor of the bound), R2 (coherent rank-one sign sweep), and R3 (tiny inputs).
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.polar_opt import schatten_switch_blackwell as ssb
from research.polar_opt.baselines import svd_polar_fp64
from research.polar_opt.harness_tier1 import _quality, harvested_cases, synthetic_cases
from research.polar_opt.polar_switch_blackwell import _PLANS, polar_switch_blackwell
from research.polar_opt.schatten_switch_blackwell import (
    DEFAULT_VARIANT,
    SCHATTEN_SWITCH_CIRCUITS,
    schatten_switch_blackwell,
    schatten_switch_reference,
    theoretical_flops,
)
from research.polar_opt.schatten_switch_search import (
    CAP,
    PS3_MAX,
    PS3_MEDIAN,
    identity_plus_constant,
    rank1_sweep,
    spectral_bound,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = json.loads((ROOT / "results" / "schatten_switch_search_cycle2.json").read_text())
TRAJECTORIES = ROOT / "artifacts" / "harvested_trajectories.pt"
NAMES = sorted(SCHATTEN_SWITCH_CIRCUITS)
BF16_NAMES = [name for name in NAMES if SCHATTEN_SWITCH_CIRCUITS[name].bf16_verified]
FP16, BF16 = torch.float16, torch.bfloat16
SHAPES = [(256, 1024), (515, 1029), (1029, 515), (512, 512), (255, 255), (3, 300, 100), (2, 3, 64, 200)]


def rand(shape, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, device="cuda", generator=generator)


def rel(a, reference):
    """Largest per-matrix relative Frobenius distance."""
    a, reference = a.double(), reference.double()
    return ((a - reference).norm(dim=(-2, -1)) / reference.norm(dim=(-2, -1))).max().item()


def with_spectrum(singular_values, m, seed=0):
    generator = torch.Generator().manual_seed(seed)
    n = len(singular_values)
    u = torch.linalg.qr(torch.randn(n, n, generator=generator, dtype=torch.float64)).Q
    v = torch.linalg.qr(torch.randn(m, n, generator=generator, dtype=torch.float64)).Q
    return ((u * singular_values) @ v.mT).float().cuda()


# ------------------------------------------------------------------ tables and the scalar record
def test_registry_matches_the_search_artifact():
    assert NAMES and DEFAULT_VARIANT in SCHATTEN_SWITCH_CIRCUITS
    for name in NAMES:
        circuit = SCHATTEN_SWITCH_CIRCUITS[name]
        assert sum(circuit.squarings) < 10, "PS3_55 has 10 squarings"
        assert circuit.buffers in (2, 3)
        assert circuit.u_safe >= 1.05
        assert circuit.to_dict() == ARTIFACT["selected"][name]["circuit"]
        cost = theoretical_flops(name, (1024, 4096))
        assert cost["symmetric_flops"] < (6 * 4 + 10) * 1024 ** 3
        assert cost["matrix_kernel_launches"] < 14


@pytest.mark.parametrize("name", NAMES)
def test_scalar_design_record(name):
    """In exact arithmetic: the band error on [lo, 1 / u_safe], and bounds on the whole basin."""
    circuit = SCHATTEN_SWITCH_CIRCUITS[name]
    x = np.unique(np.r_[np.linspace(0, 1, 40001), np.geomspace(1e-7, 1, 20001)])
    f = circuit.scalar(x)
    band = (x >= circuit.lo) & (x <= 1 / circuit.u_safe)
    assert np.abs(f[band] - 1).max() <= circuit.error + 1e-6
    assert circuit.error <= 0.12
    assert f.min() >= -1e-9 and f.max() <= 1.15, "bounded on [0, 1], which is [0, u_safe] in units of the bound"
    first = dataclasses.replace(circuit, table=circuit.table[:1]).scalar(x)
    assert first.min() >= -1e-9 and first.max() <= 1 / circuit.u_safe + 1e-9, "segment 2 keeps u_safe of headroom"
    second = x.copy()
    work = {0: second * second}
    for dst, src, add, a, b, c in circuit.table[1][1]:
        work[dst] = a * work[src] ** 2 + b * work[add] + c
    second = x * work[dst]
    assert second.min() >= -1e-9 and second.max() <= 1.15


# ------------------------------------------------------------------ equivalence to the references
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_native_matches_float64_matrix_reference(shape, name):
    G = rand(shape)
    Q = schatten_switch_blackwell(G, name)
    assert Q.shape == G.shape and Q.dtype == G.dtype and torch.isfinite(Q).all()
    exact = schatten_switch_reference(G, name)  # float64 arithmetic, same scale formula
    emulated = schatten_switch_reference(G, name, dtype=FP16)
    P = svd_polar_fp64(G)
    assert rel(Q, exact) < 2e-2
    assert rel(Q, emulated) < 2e-2
    assert abs(rel(Q, P) - rel(exact, P)) < 5e-3
    assert rel(Q, P) < 0.2


@pytest.mark.parametrize("name", NAMES)
def test_singular_values_follow_the_float64_scalar_map(name):
    circuit = SCHATTEN_SWITCH_CIRCUITS[name]
    s = torch.logspace(0, -3.5, 192, dtype=torch.float64)
    G = with_spectrum(s, 384)
    _, info = schatten_switch_reference(G, name, return_info=True)
    x = (s / s.norm()).numpy() / info["sigma_hat"][0]
    assert x.max() <= 1 / circuit.u_safe + 1e-6
    expected = np.sort(circuit.scalar(x))
    actual = np.sort(torch.linalg.svdvals(schatten_switch_blackwell(G, name).double()).cpu().numpy())
    assert np.abs(actual - expected).max() < 0.03


@pytest.mark.parametrize("name", NAMES)
def test_batched_result_equals_per_matrix_results(name):
    # Members with different stable ranks get different scales: this is the batch rescale path.
    flat = torch.ones(128, dtype=torch.float64)
    steep = torch.logspace(0, -3, 128, dtype=torch.float64)
    spike = torch.full((128,), 1e-2, dtype=torch.float64)
    spike[0] = 1.0
    G = torch.stack([with_spectrum(flat, 256, 1), 7.0 * with_spectrum(steep, 256, 2), 1e-3 * with_spectrum(spike, 256, 3)])
    batched = schatten_switch_blackwell(G, name)
    P = svd_polar_fp64(G)
    for index in range(3):
        single = schatten_switch_blackwell(G[index], name)
        # The steep member has FP16 noise of about 2 percent (the reference in FP16 differs from
        # the reference in FP64 by the same amount), so the two paths agree only to that level.
        assert rel(batched[index], single) < 3e-2
        assert abs(rel(batched[index], P[index]) - rel(single, P[index])) < 5e-3
    nested = schatten_switch_blackwell(G.reshape(1, 3, 128, 256), name)
    assert torch.equal(nested.reshape(G.shape), batched)


# ------------------------------------------------------------------ Fable R1: subnormals and the bound
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("ratio", [0.15, 0.2, 0.246, 0.3])
def test_fable_r1_identity_plus_constant_4096_fp16(ratio, name):
    """``p I + c 11^T`` at 4096 x 4096: trace-1 Gram entries would be FP16 subnormals."""
    Q = schatten_switch_blackwell(identity_plus_constant(4096, ratio), name, dtype=FP16)
    assert torch.isfinite(Q).all()
    assert spectral_bound(Q) <= CAP


def test_fable_r1_identity_plus_constant_8192_fp16():
    Q = schatten_switch_blackwell(identity_plus_constant(8192, 0.2), DEFAULT_VARIANT, dtype=FP16)
    assert torch.isfinite(Q).all()
    assert spectral_bound(Q) <= CAP


def two_level_rank_one(n, device):
    u = torch.ones(n, device=device)
    u[n // 2:] = 1.013
    return u[:, None] * u[None, :]


@pytest.mark.parametrize("dtype", [FP16, BF16], ids=str)
@pytest.mark.parametrize("device,n", [("cpu", 2048), ("cuda", 4096)])
def test_fable_r1_bound_is_rigorous_on_cpu_and_cuda(device, n, dtype):
    """The scale is an upper bound on sigma_max of the rounded operand, and it is not loose."""
    circuit = SCHATTEN_SWITCH_CIRCUITS[DEFAULT_VARIANT]
    for G in (identity_plus_constant(n, 0.246).to(device), two_level_rank_one(n, device),
              torch.randn(n, n, generator=torch.Generator().manual_seed(n)).to(device)):
        _, info = schatten_switch_reference(G, DEFAULT_VARIANT, dtype=dtype, return_info=True)
        top, upper, scale = info["sigma_max_operand"][0], info["upper"][0], info["sigma_hat"][0]
        assert top <= upper, "the bound must hold for the rounded operand"
        assert scale == pytest.approx(circuit.u_safe * upper, rel=1e-12)
        assert info["cast_residual"][0] <= info["cast_residual_bound"][0]
    # Low stable rank: the Schatten-8 bound is tight (the Frobenius bound would be 1.0).
    _, info = schatten_switch_reference(two_level_rank_one(n, device), DEFAULT_VARIANT, dtype=dtype, return_info=True)
    assert info["upper"][0] / info["sigma_max_operand"][0] < (1.003 if dtype == FP16 else 1.01)


def test_fable_r1_cpu_and_cuda_bounds_agree():
    G = identity_plus_constant(1024, 0.246)
    _, cpu = schatten_switch_reference(G.cpu(), DEFAULT_VARIANT, dtype=FP16, return_info=True)
    _, gpu = schatten_switch_reference(G, DEFAULT_VARIANT, dtype=FP16, return_info=True)
    assert cpu["upper"][0] == pytest.approx(gpu["upper"][0], rel=1e-5)


@pytest.mark.parametrize("name", NAMES)
def test_native_scale_equals_reference_scale(name):
    """The native output equals the reference with the scale passed explicitly."""
    G = rand((256, 512), seed=5)
    _, info = schatten_switch_reference(G, name, dtype=FP16, return_info=True)
    explicit = schatten_switch_blackwell(G, name, sigma_hat=info["sigma_hat"][0])
    assert rel(schatten_switch_blackwell(G, name), explicit) < 1e-2


# ------------------------------------------------------------------ Fable R2: coherent rank-one sweep
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("shape,points", [((64, 128), 1200), ((256, 256), 400)], ids=str)
def test_fable_r2_rank_one_sign_sweep_fp16(shape, points, name):
    """Exact rank-one sign matrices with sigma / bound swept over [0, u_safe]: real matrix code."""
    result = rank1_sweep(SCHATTEN_SWITCH_CIRCUITS[name], FP16, shape, points)
    assert result["finite"] and result["passed"]
    assert result["max_sigma"] <= result["max_frobenius"] + 1e-9 <= CAP + 1e-9
    assert result["max_abs_deviation_from_scalar"] < 0.2  # coherent FP16 rounding, largest for SS3_53
    assert result["max_batch_vs_single"] < 0.02


@pytest.mark.parametrize("name", BF16_NAMES)
@pytest.mark.parametrize("shape,points", [((64, 128), 1200), ((256, 256), 400)], ids=str)
def test_fable_r2_rank_one_sign_sweep_bf16(shape, points, name):
    result = rank1_sweep(SCHATTEN_SWITCH_CIRCUITS[name], BF16, shape, points)
    assert result["finite"] and result["passed"] and result["max_frobenius"] <= CAP


def test_bf16_needs_a_verified_circuit():
    G = rand((64, 128))
    unverified = dataclasses.replace(SCHATTEN_SWITCH_CIRCUITS[DEFAULT_VARIANT], bf16_verified=False)
    with pytest.raises(ValueError, match="BF16"):
        schatten_switch_blackwell(G, unverified, dtype=BF16)
    assert torch.isfinite(schatten_switch_blackwell(G, unverified, dtype=BF16, allow_unsafe_bf16=True)).all()
    for name in NAMES:
        if name not in BF16_NAMES:
            with pytest.raises(ValueError, match="BF16"):
                schatten_switch_blackwell(G, name, dtype=BF16)
    for name in BF16_NAMES:
        Q = schatten_switch_blackwell(G, name, dtype=BF16)
        assert torch.isfinite(Q).all() and rel(Q, svd_polar_fp64(G)) < 0.2


# ------------------------------------------------------------------ Fable R3: normalization
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("scale", [1e-14, 1e-25, 1e-30, 1e-38])
def test_fable_r3_tiny_inputs_return_exact_zeros(scale, name):
    G = rand((64, 128), seed=1) * scale
    Q = schatten_switch_blackwell(G, name)
    assert Q.shape == G.shape and not Q.any()


def test_fable_r3_dead_and_nonfinite_inputs_return_exact_zeros():
    assert not schatten_switch_blackwell(torch.zeros(96, 384, device="cuda")).any()
    for value in (float("nan"), float("inf")):
        G = rand((32, 64), seed=2)
        G[3, 5] = value
        assert not schatten_switch_blackwell(G).any()


@pytest.mark.parametrize("scale", [1e-9, 1e6, 1e25, 1e30])
def test_fable_r3_scale_invariance_without_overflow(scale):
    G = rand((64, 128), seed=3)
    base = schatten_switch_blackwell(G)
    Q = schatten_switch_blackwell(G * scale)
    assert torch.isfinite(Q).all() and rel(Q, base) < 1e-2


def test_fable_r3_mixed_batch():
    scales = torch.tensor([1.0, 1e-30, 1e25, 0.0], device="cuda")
    G = rand((4, 32, 64), seed=4) * scales[:, None, None]
    Q = schatten_switch_blackwell(G)
    assert torch.isfinite(Q).all()
    assert not Q[1].any() and not Q[3].any()
    for index in (0, 2):
        assert rel(Q[index], svd_polar_fp64(G[index])) < 0.2


# ------------------------------------------------------------------ plumbing
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (64, 1)], ids=str)
def test_min_dim_one_shapes_supported(shape, name):
    circuit = SCHATTEN_SWITCH_CIRCUITS[name]
    G = rand(shape, seed=11)
    Q = schatten_switch_blackwell(G, name)
    assert Q.shape == G.shape and torch.isfinite(Q).all()
    assert rel(Q, G / G.norm()) < circuit.error + 0.03  # the polar factor of a vector is the unit vector


def test_torch_compile_does_not_poison_launch_cache():
    schatten_switch_blackwell(rand((256, 1024), seed=12))
    G = rand((128, 512), seed=13)
    compiled = torch.compile(lambda x: schatten_switch_blackwell(x))(G)
    torch._dynamo.reset()
    eager = schatten_switch_blackwell(G)
    assert torch.isfinite(eager).all() and torch.equal(compiled, eager)
    assert all(plan[0] is not None and hasattr(plan[0], "__getitem__") for plan in _PLANS.values())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64], ids=str)
def test_input_dtype_is_restored_and_input_is_not_modified(dtype):
    G = rand((200, 600), seed=7).to(dtype)
    before = G.clone()
    P = svd_polar_fp64(G)
    for name in NAMES:
        Q = schatten_switch_blackwell(G, name)
        assert Q.dtype == dtype and Q.shape == G.shape and torch.isfinite(Q).all()
        assert rel(Q, P) < 0.12
    assert torch.equal(G, before)


def test_noncontiguous_and_offset_inputs():
    base = rand((301, 2, 96), seed=10)
    G = base[1:, 0].mT  # (96, 300) view: transposed, strided, and offset
    Q = schatten_switch_blackwell(G)
    assert Q.shape == G.shape and torch.isfinite(Q).all()
    assert rel(Q, schatten_switch_blackwell(G.contiguous())) < 5e-3


def test_first_call_and_cached_launch_agree_bitwise():
    G = rand((97, 331), seed=9)
    for name in NAMES:
        first = schatten_switch_blackwell(G, name)
        assert torch.equal(first, schatten_switch_blackwell(G, name))
        assert torch.equal(first, schatten_switch_blackwell(G.clone(), name))


def test_empty_and_invalid_inputs():
    assert schatten_switch_blackwell(torch.empty(0, 8, device="cuda")).shape == (0, 8)
    assert schatten_switch_blackwell(torch.empty(3, 0, 8, device="cuda")).shape == (3, 0, 8)
    G = rand((32, 64))
    with pytest.raises(ValueError):
        schatten_switch_blackwell(G, "PS3_55")
    with pytest.raises(TypeError):
        schatten_switch_blackwell(G, dtype=torch.float32)
    with pytest.raises(ValueError):
        schatten_switch_blackwell(G.cpu())
    with pytest.raises(ValueError):
        schatten_switch_blackwell(G[0])
    with pytest.raises(ValueError):
        schatten_switch_blackwell(G, scaling="spectral")
    assert rel(schatten_switch_blackwell(G, DEFAULT_VARIANT.lower().replace("_", "-")), schatten_switch_blackwell(G)) == 0.0


def test_frobenius_scaling_is_safe_but_lifts_less():
    G = with_spectrum(torch.logspace(0, -3, 128, dtype=torch.float64), 256, seed=6)
    P = svd_polar_fp64(G)
    Q = schatten_switch_blackwell(G, scaling="frobenius")
    assert torch.isfinite(Q).all() and torch.linalg.svdvals(Q.double()).max().item() <= CAP
    assert rel(schatten_switch_blackwell(G), P) < rel(Q, P)


# ------------------------------------------------------------------ Tier-1: 117 cases
@pytest.fixture(scope="module")
def tier1_cases():
    if not TRAJECTORIES.is_file():
        pytest.skip(f"missing {TRAJECTORIES}")
    cases = list(synthetic_cases([128, 256], [1, 2, 4, 8], 123)) + list(harvested_cases(TRAJECTORIES, all_layers=True))
    out = []
    for case in cases:
        G = case.matrix.cuda()
        P = svd_polar_fp64(G)
        out.append((case.suite, case.name, G, P, P.cpu()))
    return out


def run_tier1(name, dtype, cases):
    """Per-suite errors over the 117 cases; every case must be finite and below the cap."""
    rows = {}
    for suite, case, G, P, P_cpu in cases:
        quality = _quality(schatten_switch_blackwell(G, name, dtype=dtype), P, P_cpu, 2.0)
        assert not quality["diverged"], f"{case}: {quality}"
        assert quality["sigma_max"] <= CAP, case
        rows.setdefault(suite, []).append(quality["relative_frobenius_error"])
    return rows


@pytest.mark.parametrize("name", NAMES)
def test_tier1_117_cases_accuracy_and_zero_divergence(name, tier1_cases):
    assert len(tier1_cases) == 117
    record = ARTIFACT["selected"][name]
    rows = run_tier1(name, FP16, tier1_cases)
    harvested = np.array(rows["harvested_momentum"])
    assert len(harvested) == 45 and sum(map(len, rows.values())) == 117
    for suite, errors in rows.items():
        stored = record["tier1"][suite]
        assert stored["diverged"] == 0
        assert np.median(errors) == pytest.approx(stored["median_relative_frobenius_error"], abs=5e-3)
        assert max(errors) == pytest.approx(stored["max_relative_frobenius_error"], abs=1e-2)
    # The exact-arithmetic score that selected the circuit predicts the FP16 matrix result.
    assert np.median(harvested) == pytest.approx(record["scalar"]["harvested_median"], abs=5e-3)
    assert harvested.max() == pytest.approx(record["scalar"]["harvested_max"], abs=1e-2)
    if record["parity_with_ps3_55"]:
        assert np.median(harvested) <= PS3_MEDIAN and harvested.max() <= PS3_MAX + 2e-3


@pytest.mark.parametrize("name", BF16_NAMES)
def test_tier1_117_cases_zero_divergence_bf16(name, tier1_cases):
    rows = run_tier1(name, BF16, tier1_cases)
    stored = ARTIFACT["selected"][name]["tier1_bf16"]
    assert stored["all"]["diverged"] == 0
    assert np.median(rows["harvested_momentum"]) == pytest.approx(
        stored["harvested_momentum"]["median_relative_frobenius_error"], abs=1e-2)


def test_default_variant_is_at_ps3_parity_with_fewer_squarings():
    assert ARTIFACT["default_variant"] == DEFAULT_VARIANT
    record = ARTIFACT["selected"][DEFAULT_VARIANT]
    assert record["parity_with_ps3_55"] and record["squarings"] < 10
    assert record["tier1"]["all"]["diverged"] == 0 and record["tier1"]["all"]["cases"] == 117
    reference = ARTIFACT["tier1"]["summary"]["harvested_momentum"]["ps3_55_blackwell_fp16"]
    ours = record["tier1"]["harvested_momentum"]
    assert ours["median_relative_frobenius_error"] <= reference["median_relative_frobenius_error"]
    assert ours["max_relative_frobenius_error"] <= reference["max_relative_frobenius_error"]


# ------------------------------------------------------------------ quick GPU latency check
def _event_times(calls, G, rounds=12, warmup=4):
    """Fastest CUDA-event interval per method, methods interleaved (robust to a shared GPU)."""
    for call in calls.values():
        for _ in range(warmup):
            call(G)
    torch.cuda.synchronize()
    best = {name: float("inf") for name in calls}
    for _ in range(rounds):
        for name, call in calls.items():
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            out = call(G)
            end.record()
            torch.cuda.synchronize()
            best[name] = min(best[name], start.elapsed_time(end))
            del out
    return best


@pytest.mark.parametrize("shape", [(512, 2048), (1024, 4096), (2048, 8192), (4096, 4096)], ids=str)
def test_quick_gpu_latency_against_ps3_55(shape, record_property):
    """Sanity only: the coordinator runs the clean head-to-head on an idle GPU."""
    G = rand(shape, seed=20)
    calls = {"ps3_55_blackwell_fp16": lambda x: polar_switch_blackwell(x, "PS3_55", dtype=FP16)}
    for name in NAMES:
        calls[name] = lambda x, name=name: schatten_switch_blackwell(x, name)
    times = _event_times(calls, G)
    base = times.pop("ps3_55_blackwell_fp16")
    speedups = {name: base / value for name, value in times.items()}
    record_property("ps3_55_blackwell_fp16_ms", base)
    for name, value in times.items():
        record_property(f"{name}_ms", value)
    print(f"\n{shape}: ps3_55 {base:.3f} ms; " + "; ".join(
        f"{name} {times[name]:.3f} ms ({speedups[name]:.2f}x)" for name in times))
    # Below about n = 2048 the call is bound by host kernel-launch overhead of the extra
    # norm/fold kernels, so the circuits are slower than PS3_55 there. That is a measured property,
    # not a test failure; the bound only catches a gross regression.
    for name, value in speedups.items():
        assert value > 0.3, f"{name} is far slower than PS3_55 on {shape}"
    if min(shape) >= 2048:  # squarings dominate: fewer squarings must show
        assert max(speedups.values()) > 1.0
        assert speedups[DEFAULT_VARIANT] > 0.95


@pytest.mark.parametrize("name", NAMES)
def test_cuda_graph_capture_and_replay_exact_match(name):
    for shape in [(512, 2048), (4, 256, 512)]:
        static_in = rand(shape, seed=31)
        for _ in range(5):
            schatten_switch_blackwell(static_in, name)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(5):
                schatten_switch_blackwell(static_in, name)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = schatten_switch_blackwell(static_in, name)
        torch.cuda.synchronize()

        graph.replay()
        torch.cuda.synchronize()
        eager = schatten_switch_blackwell(static_in, name)
        assert torch.isfinite(static_out).all()
        assert (static_out - eager).abs().max().item() < 1e-5

        if len(shape) == 3:
            scales = torch.tensor([1.0, 1e-30, 1e25, 0.0], device="cuda")
            fresh = rand(shape, seed=32) * scales[:, None, None]
        else:
            fresh = rand(shape, seed=32) * 1e20
        static_in.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        eager_fresh = schatten_switch_blackwell(fresh, name)
        assert torch.isfinite(static_out).all()
        assert (static_out - eager_fresh).abs().max().item() < 1e-5
        if len(shape) == 3:
            assert not static_out[1].any() and not static_out[3].any()


def test_fable_m1_long_k_bound_rigor_and_stability():
    """Fable M1: compensate Triton Gram downward bias for very long inner dimension K > 2^16."""
    G = torch.ones(1, 1 << 25, device="cuda")
    Q = schatten_switch_blackwell(G, "SS3_35")
    assert torch.isfinite(Q).all()
    assert Q.double().norm().item() <= 1.25


def test_fable_l1_sigma_hat_overflow_returns_exact_zeros():
    """Fable L1: overflowing sigma_hat paths return exact finite zeros with no 0*Inf NaNs."""
    G = rand((64, 128), seed=41)
    q_fp16 = schatten_switch_blackwell(G, "SS3_54", sigma_hat=1e-10)
    q_bf16 = schatten_switch_blackwell(G, "SS3_35", dtype=BF16, sigma_hat=1e-3)
    assert torch.isfinite(q_fp16).all() and not q_fp16.any()
    assert torch.isfinite(q_bf16).all() and not q_bf16.any()


def test_custom_pure_square_and_nonzero_residual_circuits():
    """Opus L7: exercise pure-square (b0=0, c0=0) and non-zero residual branches."""
    G = rand((3, 97, 331), seed=42)
    pure = ssb.SchattenSwitchCircuit(
        name="TEST_PURE", table=((0.0, ((1, 0, 0, 1.0, 0.0, 0.0), (0, 1, 0, 0.0, 0.5, 0.5))),),
        buffers=2, u_safe=1.05, lo=0.5, error=0.5,
    )
    res_circuit = ssb.SchattenSwitchCircuit(
        name="TEST_RES", table=((0.05, ((1, 0, 0, 1.0, -1.0, 0.0), (0, 1, 0, 0.0, 0.5, 0.45))),),
        buffers=2, u_safe=1.05, lo=0.5, error=0.5,
    )
    for circuit in (pure, res_circuit):
        out = schatten_switch_blackwell(G, circuit, dtype=FP16)
        ref = schatten_switch_reference(G, circuit, dtype=FP16)
        assert torch.isfinite(out).all()
        assert rel(out, ref) < 5e-3


# ------------------------------------------------------------------ Cycle 3 C3-02: early-spectrum mode
def test_early_srank_spectra_are_synthetic_and_near_rank_one():
    from research.polar_opt import schatten_switch_search as search

    spectra = search.early_srank_spectra()
    again = search.early_srank_spectra()
    assert len(spectra) == len(search.EARLY_SIZES) * len(search.EARLY_SRANKS) * (
        len(search.EARLY_POWERS) + len(search.EARLY_SECONDARIES) * len(search.EARLY_FLOORS))
    for row, other in zip(spectra, again):
        s = row["s"]
        assert np.array_equal(s, other["s"]) and len(s) == row["n"]
        assert 1.03 - 1e-9 <= np.sum(s ** 2) / s[0] ** 2 <= 1.60 + 1e-9
        assert s[0] == s.max() == 1.0 and s.min() > 0
    # Anti-overfitting: the search never reads the held-out real momenta.
    assert "harvested_nesterov_real" not in Path(search.__file__).read_text()


@pytest.mark.parametrize("name", NAMES)
def test_kernel_launch_record_matches_the_profiler_and_is_sync_free(name):
    from research.polar_opt.schatten_switch_search import kernel_launches, measure_launches

    circuit = SCHATTEN_SWITCH_CIRCUITS[name]
    record = kernel_launches(circuit, 512)
    assert record["matrix_products"] == theoretical_flops(name, (512, 2048))["matrix_kernel_launches"]
    for shape in [(512, 2048), (1024, 1024)]:
        measured = measure_launches(lambda G: schatten_switch_blackwell(G, name), shape)
        assert measured["kernels"] == record["total"] and measured["host_reads"] == 0 == record["host_reads"]


@pytest.mark.parametrize("name", ["SS3_54", "SS3_35"])
def test_early_mode_gpu_gates_pass_for_registered_circuits(name):
    from research.polar_opt.schatten_switch_search import graph_replay, r3_long_k

    circuit = SCHATTEN_SWITCH_CIRCUITS[name]
    assert r3_long_k(circuit, FP16)["passed"]
    assert graph_replay(circuit, FP16)["passed"]
    if circuit.bf16_verified:
        assert graph_replay(circuit, BF16)["passed"]


def test_heldout_eval_loads_validated_candidates(tmp_path):
    from research.polar_opt.eval_heldout_nesterov import _candidate_methods

    fp16_only = dataclasses.replace(SCHATTEN_SWITCH_CIRCUITS["SS3_54"], name="X_54")
    both = dataclasses.replace(SCHATTEN_SWITCH_CIRCUITS["SS3_35"], name="X_35")
    path = tmp_path / "search.json"
    path.write_text(json.dumps({
        "validated": {
            "X_54": {"circuit": fp16_only.to_dict(), "fp16_passed": True, "bf16_verified": False},
            "X_35": {"circuit": both.to_dict(), "fp16_passed": True, "bf16_verified": True},
        }
    }))
    methods = _candidate_methods(path)
    assert sorted(methods) == ["cand_X_35_bf16", "cand_X_35_fp16", "cand_X_54_fp16"]
    G = rand((256, 512), seed=3)
    assert torch.equal(methods["cand_X_54_fp16"](G), schatten_switch_blackwell(G, "SS3_54"))
