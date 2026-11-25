"""Adversarial tests for warm spectral bounds, the guard, and the short Gram circuits.

CPU tests use the exact FP16/BF16 emulation in ``circuits`` (half operands, FP32
accumulation). CUDA tests run the same code on tensor cores and must agree.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from research.polar_opt.baselines import PS2_55, PS3_55, POLAR_EXPRESS_COEFFICIENTS, svd_polar_fp64
from research.polar_opt.circuits import (
    CAP,
    CIRCUITS,
    FP16_ONLY_CIRCUITS,
    SpectralPrescaledPolar,
    WarmSpectralScaler,
    run_circuit,
)
from research.polar_opt.spectral_scaling import (
    CALIBRATED_GAMMA,
    EstimatorConfig,
    WarmSpectralEstimator,
    calibrate_gamma,
    gram_bounds,
    matrix_bounds,
    round_to,
    stability_basin,
)

TRAJ = Path(__file__).resolve().parents[1] / "artifacts" / "harvested_trajectories.pt"
CUDA = torch.cuda.is_available()
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
# Rounding slack on top of the exact-arithmetic design error (relative Frobenius).
SLACK = {"fp16": 0.02, "bf16": 0.05}
ALL = {**CIRCUITS, **FP16_ONLY_CIRCUITS}
NAMES = sorted(ALL)
# (circuit, dtype) pairs whose scalar-rounded basin is claimed: BF16-safe circuits in both
# precisions, FP16-only circuits (e.g. the recommended S1-k4+5) in FP16.
CASES = [(n, dt) for n in sorted(CIRCUITS) for dt in ("fp16", "bf16")] + \
        [(n, "fp16") for n in sorted(FP16_ONLY_CIRCUITS)]


def spectral(n, m, s, seed=0):
    """``(n, m)`` float64 matrix with singular values ``s`` (length min(n, m))."""
    g = torch.Generator().manual_seed(seed)
    k = min(n, m)
    U = torch.linalg.qr(torch.randn(n, k, dtype=torch.float64, generator=g))[0]
    V = torch.linalg.qr(torch.randn(m, k, dtype=torch.float64, generator=g))[0]
    return (U * s) @ V.mT, U, V


def loguniform(k, lo, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.exp(torch.empty(k, dtype=torch.float64).uniform_(np.log(lo), 0, generator=g))
    s[0] = 1.0
    return s


def rel(Y, P):
    return ((Y.double().cpu() - P).norm() / P.norm()).item()


def sigma_max(M):
    return torch.linalg.matrix_norm(M.double(), ord=2).item()


def guarded(est, M):
    """Run the estimator on the Gram matrix of the Frobenius-normalized wide view, like circuits do."""
    W = M.mT if M.shape[-2] > M.shape[-1] else M
    W = (W / W.norm()).float()
    R = W @ W.mT
    return est.estimate(R, gram=True, A2=R @ R), sigma_max(W)


# --------------------------------------------------------------------------------------
# Rigorous bounds
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [1, 2, 4, 8])
@pytest.mark.parametrize("tall", [False, True])
def test_rigorous_bounds_bracket_sigma_max(alpha, tall):
    n = 48
    for seed, s in enumerate([loguniform(n, 1e-6), torch.ones(n, dtype=torch.float64),
                              torch.cat([torch.tensor([100.0]), torch.ones(n - 1)]).double()]):
        M, _, _ = spectral(n, alpha * n, s, seed)
        M = M.mT if tall else M
        true = sigma_max(M)
        lo, hi = matrix_bounds(M)
        assert lo.item() <= true * (1 + 1e-6) and hi.item() >= true * (1 - 1e-6)
        W = M.mT if tall else M
        A = W @ W.mT
        glo, ghi = gram_bounds(A, A @ A)
        assert glo.item() <= true ** 2 * (1 + 1e-6) <= ghi.item() * (1 + 2e-6)


def test_schatten8_bound_is_tight_on_harvested_spectra():
    if not TRAJ.exists():
        pytest.skip("no harvested trajectories")
    pairs = torch.load(TRAJ)["pairs"]
    ratios = []
    for p in pairs:
        W = p["current"].double()
        W = W.mT if W.shape[0] > W.shape[1] else W
        A = W @ W.mT
        ratios.append(gram_bounds(A, A @ A)[1].sqrt().item() / sigma_max(W))
    # Frobenius / sigma_max is 3.19 (median) on the same matrices.
    assert min(ratios) >= 1 - 1e-9 and max(ratios) < 1.25 and np.median(ratios) < 1.12


# --------------------------------------------------------------------------------------
# Adversarial estimator traps
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("defer", [True, False])
@pytest.mark.parametrize("spike", [3.0, 10.0, 100.0])
def test_orthogonal_warm_start_trap(defer, spike):
    """Warm vector exactly orthogonal to a new rank-1 spike: power iteration sees nothing."""
    n = 128
    s = torch.logspace(0, -3, n, dtype=torch.float64)
    M0, U, V = spectral(n, 2 * n, s, seed=3)
    est = WarmSpectralEstimator(EstimatorConfig(steps=1, defer_escalation=defer))
    out0, _ = guarded(est, M0)
    s1 = s.clone()
    s1[-1] = spike  # the spike lives on the old *bottom* direction, orthogonal to v_{t-1}
    M1 = (U * s1) @ V.mT
    out1, true1 = guarded(est, M1)
    # Deferred mode: the warm power estimate is fooled in this call (immediate mode
    # already ran the Krylov escalation and found the spike) ...
    if defer and spike >= 5:
        assert out1.power.item() < true1 / 2
    # ... but the guarded scale never leaves the stability basin.
    assert out1.sigma.item() >= true1 / 1.12
    assert bool(out1.escalated.item()) or out1.sigma.item() >= true1
    # The next call (deferred refresh or no refresh needed) finds the spike.
    out2, true2 = guarded(est, M1)
    assert abs(out2.power.item() / true2 - 1) < 0.02


def test_rank_one_spike_appearing_from_noise():
    n = 96
    g = torch.Generator().manual_seed(5)
    est = WarmSpectralEstimator()
    noise = torch.randn(n, 3 * n, dtype=torch.float64, generator=g) * 1e-3
    guarded(est, noise)
    u = torch.randn(n, 1, dtype=torch.float64, generator=g)
    v = torch.randn(1, 3 * n, dtype=torch.float64, generator=g)
    M = noise + 100 * 1e-3 * np.sqrt(3 * n) * (u / u.norm()) @ (v / v.norm())
    out, true = guarded(est, M)
    assert out.sigma.item() >= true / 1.12


@pytest.mark.parametrize("kappa", [1e3, 1e6])
def test_extreme_condition_number_stays_in_basin(kappa):
    n = 128
    s = torch.logspace(0, -np.log10(kappa), n, dtype=torch.float64)
    M, _, _ = spectral(n, 4 * n, s, seed=7)
    est = WarmSpectralEstimator()
    for drift in range(4):  # streaming with a slowly rotating top direction
        Mt = M + 0.05 * drift * spectral(n, 4 * n, s.flip(0), seed=8 + drift)[0]
        out, true = guarded(est, Mt)
        assert true / 1.15 <= out.sigma.item() <= out.upper.item() + 1e-6


def test_streaming_trajectories_never_leave_basin():
    """Cold start on M_{t-1}, warm on M_t, for all 45 real pairs; plus gamma calibration."""
    if not TRAJ.exists():
        pytest.skip("no harvested trajectories")
    pairs = torch.load(TRAJ)["pairs"]
    worst_low, worst_high = np.inf, 0.0
    for p in pairs:
        for cfg in (EstimatorConfig(steps=1), EstimatorConfig(steps=2), EstimatorConfig(probes=1)):
            est = WarmSpectralEstimator(cfg)
            guarded(est, p["previous"])
            out, true = guarded(est, p["current"])
            r = out.sigma.item() / true
            worst_low, worst_high = min(worst_low, r), max(worst_high, r)
    assert worst_low >= 1.0 - 1e-4  # calibrated gamma: no underestimate on real data
    assert worst_high <= 1.25
    cal = calibrate_gamma([(p["previous"], p["current"]) for p in pairs[::4]], 1)
    assert cal["gamma"] <= CALIBRATED_GAMMA[1] + 1e-3


# --------------------------------------------------------------------------------------
# Stability basin [0, 1 + delta]
# --------------------------------------------------------------------------------------
def _switch(table, buffers):
    def f(s):
        for residual, rows in table:
            work = [s * s] + [None] * (buffers - 1)
            for dst, src, add, a, b, c in rows:
                work[dst] = a * work[src] ** 2 + b * work[add] + c
            s = s * work[dst] + residual * s
        return s
    return f


def test_baseline_basins_need_frobenius_scaling():
    """PS2-55/PS3-55 diverge just past 1 and PolarExpress past ~1.05: they cannot take an
    underestimated spectral scale, which is why the new circuits are designed with a basin."""
    assert stability_basin(_switch(PS2_55, 2)) < 1.01
    assert stability_basin(_switch(PS3_55, 3)) < 1.01

    def pe(s):
        for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
            s = a * s + b * s ** 3 + c * s ** 5
        return s
    assert stability_basin(pe) < 1.1


@pytest.mark.parametrize("name,dt", CASES + [(n, "fp64") for n in NAMES])
def test_circuit_basin(name, dt):
    c = ALL[name]
    rnd = None if dt == "fp64" else round_to(DTYPES[dt])
    assert c.basin_u(rnd=rnd) >= 1.12, "circuit must tolerate a 12% spectral underestimate"


# --------------------------------------------------------------------------------------
# Circuits: emulated and real FP16/BF16 against the FP64 SVD polar factor
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,dt", CASES)
@pytest.mark.parametrize("alpha", [1, 2, 4, 8])
def test_emulated_polar_error_rectangular(name, alpha, dt):
    c = ALL[name]
    n = 64
    s = loguniform(n, c.lo, seed=alpha)
    M, _, _ = spectral(n, alpha * n, s, seed=alpha)
    for G in (M, M.mT):  # wide and tall
        Y = run_circuit(G.float(), c, dtype=DTYPES[dt], sigma=torch.tensor(1.0))
        assert Y.shape == G.shape and Y.dtype == torch.float32
        assert rel(Y, svd_polar_fp64(G)) <= c.error + SLACK[dt]


@pytest.mark.parametrize("name,dt", CASES)
@pytest.mark.parametrize("overshoot", [1.0, 1.1])
def test_underestimated_scale_stays_bounded(name, dt, overshoot):
    """Scale by sigma_max / overshoot: singular values enter (1, overshoot]; output stays <= CAP."""
    c = ALL[name]
    n = 64
    s = torch.linspace(1, c.lo, n, dtype=torch.float64)
    M, _, _ = spectral(n, 2 * n, s, seed=9)
    Y = run_circuit(M.float(), c, dtype=DTYPES[dt], sigma=torch.tensor(1 / overshoot))
    sv = torch.linalg.svdvals(Y.double())
    assert torch.isfinite(Y).all() and sv.max().item() <= max(CAP, 1 + c.error) + SLACK[dt]


@pytest.mark.parametrize("name", NAMES)
def test_batched_shapes_and_zero_input(name):
    c = ALL[name]
    G = torch.randn(2, 3, 40, 24)
    G[1, 2] = 0
    Y = SpectralPrescaledPolar(c)(G, key="w")
    assert Y.shape == G.shape and torch.isfinite(Y).all() and Y[1, 2].abs().max() == 0


@pytest.mark.parametrize("name", NAMES)
def test_trajectories_emulated_fp16(name):
    """Streaming warm scaling on real pairs, CPU FP16 emulation: finite, and no worse than
    Frobenius scaling of the same circuit (spectral scaling must help, not hurt)."""
    if not TRAJ.exists():
        pytest.skip("no harvested trajectories")
    c = ALL[name]
    pairs = torch.load(TRAJ)["pairs"][::5]
    warm, frob = [], []
    for p in pairs:
        polar = SpectralPrescaledPolar(c)
        polar(p["previous"], key=0)
        Y = polar(p["current"], key=0)
        P = svd_polar_fp64(p["current"])
        assert torch.isfinite(Y).all()
        warm.append(rel(Y, P))
        frob.append(rel(run_circuit(p["current"], c), P))
    assert np.median(warm) <= np.median(frob) + 1e-3


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("name,dt", CASES)
@pytest.mark.parametrize("triton", [False, True])
def test_cuda_matches_emulation(name, dt, triton):
    c = ALL[name]
    n = 128
    M, _, _ = spectral(n, 4 * n, loguniform(n, c.lo, seed=2), seed=2)
    P = svd_polar_fp64(M)
    cpu = rel(run_circuit(M.float(), c, dtype=DTYPES[dt], sigma=torch.tensor(1.0)), P)
    gpu = rel(run_circuit(M.float().cuda(), c, dtype=DTYPES[dt], sigma=torch.tensor(1.0),
                          use_triton_syrk=triton), P)
    assert gpu <= c.error + SLACK[dt]
    assert abs(gpu - cpu) <= (2e-3 if dt == "fp16" else 1e-2)


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("name,dt", CASES)
def test_cuda_trajectories_warm_streaming(name, dt):
    if not TRAJ.exists():
        pytest.skip("no harvested trajectories")
    c = ALL[name]
    scaler = WarmSpectralScaler()
    for i, p in enumerate(torch.load(TRAJ)["pairs"]):
        run_circuit(p["previous"].cuda(), c, dtype=DTYPES[dt], scaler=scaler, key=i)
        Y, info = run_circuit(p["current"].cuda(), c, dtype=DTYPES[dt], scaler=scaler, key=i,
                              return_info=True, use_triton_syrk=True)
        W = p["current"].double()
        true = sigma_max(W) / W.norm().item()
        assert torch.isfinite(Y).all()
        assert info.sigma.item() >= true / c.basin_u()
        assert torch.linalg.svdvals(Y.double().cpu()).max().item() <= max(CAP, 1 + c.error) + SLACK[dt]


def test_schatten8_stateless_on_trajectories_and_adversarial():
    """Stateless Schatten-8 scaling never underestimates sigma_max and stays bounded everywhere."""
    c = CIRCUITS["S1-k3+4-lo0.005"]
    polar = SpectralPrescaledPolar(c, dtype=torch.float16, scaling="schatten8", use_triton_syrk=CUDA)
    # Orthogonal spike + extreme kappa = 1e6 + rank-1
    for s in (
        torch.logspace(0, -6, 64, dtype=torch.float64),
        torch.cat([torch.tensor([100.0]), torch.full((63,), 1e-3, dtype=torch.float64)]),
        torch.cat([torch.tensor([1.0]), torch.zeros(63, dtype=torch.float64)]),
    ):
        M, _, _ = spectral(64, 128, s, seed=11)
        M_in = M.float().cuda() if CUDA else M.float()
        Y = polar(M_in)
        assert torch.isfinite(Y).all()
        assert torch.linalg.svdvals(Y.double().cpu()).max().item() <= max(CAP, 1 + c.error) + SLACK["fp16"]

    if TRAJ.exists():
        pairs = torch.load(TRAJ)["pairs"]
        for p in pairs[::3]:
            cur = p["current"].cuda() if CUDA else p["current"]
            Y = polar(cur)
            assert torch.isfinite(Y).all()
            assert torch.linalg.svdvals(Y.double().cpu()).max().item() <= max(CAP, 1 + c.error) + SLACK["fp16"]


def test_block_krylov_nonfinite_guard():
    """Non-finite operator outputs trigger safe fallback instead of LinAlgError."""
    from research.polar_opt.spectral_scaling import block_krylov

    def bad_op(v):
        return torch.full_like(v, float("nan"))

    lam, vec = block_krylov(bad_op, torch.randn(16, 4), depth=2)
    assert lam.shape == (1,) or lam.ndim == 0
    assert vec.shape == (16, 1)


def test_fp16_only_circuits_refuse_bf16():
    for name in FP16_ONLY_CIRCUITS:
        with pytest.raises(ValueError):
            SpectralPrescaledPolar(name, dtype=torch.bfloat16)


def test_recommended_circuit_beats_ps2_on_trajectories():
    """S1-k4+5 (FP16, (6a + 9)n^3) vs PS2-55 ((6a + 10)n^3) on all 45 real pairs, CPU emulation."""
    if not TRAJ.exists():
        pytest.skip("no harvested trajectories")
    from research.polar_opt.baselines import polar_switch
    c = FP16_ONLY_CIRCUITS["S1-k4+5-lo0.002"]
    ours, ps2 = [], []
    for p in torch.load(TRAJ)["pairs"]:
        polar = SpectralPrescaledPolar(c)
        polar(p["previous"], key=0)
        P = svd_polar_fp64(p["current"])
        ours.append(rel(polar(p["current"], key=0), P))
        ps2.append(rel(polar_switch(p["current"], "PS2_55"), P))
    assert np.median(ours) < np.median(ps2) and np.quantile(ours, 0.9) < np.quantile(ps2, 0.9)
