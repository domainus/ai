"""Adversarial tests for spectral pre-scaling and Gram polar circuits (T1-03)."""

from __future__ import annotations


import numpy as np
import pytest
import torch

from research.polar_opt import circuit_search as cs
from research.polar_opt.circuits import (
    CIRCUITS,
    CircuitConfig,
    WarmSpectralScaler,
    gram_square_products,
    polar_spectral_circuit,
    svd_polar,
)
from research.polar_opt.spectral_scaling import (
    REASON_LOWER_INCONSISTENT,
    GuardConfig,
    block_krylov_lower,
    frobenius_upper,
    gram_schatten_upper,
    hoelder_upper,
    hybrid_guard_estimate,
    power_iteration,
    rowcol_lower,
    spectral_counterexample,
)

F64 = torch.float64
LOW = {"fp16": torch.float16, "bf16": torch.bfloat16}
# Circuits each precision is expected to handle (from the held-out validation).
FP16_CIRCUITS = sorted(CIRCUITS)
BF16_CIRCUITS = ["SP3-1r1-l0.05", "SP4-1r1-l0.05", "SP4-1r1-l0.02-bf16"]
# Held-out worst-case limits with margin (measured: fp16 <= 0.029, bf16 <= 0.075).
ERR_LIMIT = {"fp16": 0.04, "bf16": 0.10}


def orth(n: int, k: int, g: torch.Generator) -> torch.Tensor:
    return torch.linalg.qr(torch.randn(n, k, generator=g, dtype=F64)).Q


def from_sv(s, m: int, n: int, seed: int) -> torch.Tensor:
    s = torch.as_tensor(np.array(s, dtype=float), dtype=F64)
    g = torch.Generator().manual_seed(seed)
    k = s.numel()
    return (orth(m, k, g) * s) @ orth(n, k, g).mT


def smax(X: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(X.to(F64), 2)


def run(X, cfg, dtype_key=None, scaler=None, **kw):
    dt = LOW[dtype_key] if dtype_key else F64
    Xr = X.to(dt).to(F64) if dtype_key else X
    O = polar_spectral_circuit(Xr, cfg, scaler, dtype=dt, emulate=dtype_key is not None, **kw)
    return Xr, O.to(F64)


def subspace_err(Xr: torch.Tensor, O: torch.Tensor, ell: float) -> float:
    """Max ``|1 - u_i^T O v_i|`` over singular directions with ``sigma_i >= 2 ell sigma_max``."""
    U, S, Vh = torch.linalg.svd(Xr, full_matrices=False)
    d = torch.diagonal(U.mT @ O @ Vh.mT, dim1=-2, dim2=-1)
    good = S >= 2 * ell * S[..., :1]
    return float((1 - d[good]).abs().max())


# --------------------------------------------------------------------------
# Rigorous bounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64, 64), (64, 256), (256, 64), (3, 32, 96)])
def test_bounds_bracket_sigma_max(shape):
    g = torch.Generator().manual_seed(1)
    M = torch.randn(*shape, generator=g, dtype=F64)
    M = M + 5 * torch.randn(*shape[:-1], 1, generator=g, dtype=F64) * torch.randn(
        *shape[:-2], 1, shape[-1], generator=g, dtype=F64
    )
    s = smax(M)
    v = torch.randn(*shape[:-2], shape[-1], generator=g, dtype=F64)
    lowers = [
        rowcol_lower(M)[0],
        power_iteration(M, v, 1).sigma,
        power_iteration(M, v, 2).sigma,
        block_krylov_lower(M, torch.randn(*shape[:-2], shape[-1], 3, generator=g, dtype=F64), 2)[0],
    ]
    Gs = M @ M.mT if shape[-2] <= shape[-1] else M.mT @ M
    uppers = [frobenius_upper(M), hoelder_upper(M), gram_schatten_upper(Gs, 1), gram_schatten_upper(Gs @ Gs, 2)]
    tol = 1e-5  # estimators compute in fp32
    for lo in lowers:
        assert torch.all(lo <= s * (1 + tol))
    for up in uppers:
        assert torch.all(up >= s * (1 - tol))
    # Schatten-8 is never looser than Schatten-4 or Frobenius.
    assert torch.all(uppers[3] <= uppers[2] * (1 + tol))
    assert torch.all(uppers[2] <= uppers[0] * (1 + tol))


# --------------------------------------------------------------------------
# Orthogonal warm-start trap
# --------------------------------------------------------------------------


def _spike_trap(n=256, spike=100.0, seed=5):
    g = torch.Generator().manual_seed(seed)
    U, V = orth(n, n, g), orth(n, n, g)
    s = torch.ones(n, dtype=F64)
    s[0] = spike
    return (U * s) @ V.mT, U, V


def test_trap_power_iteration_fails_alone():
    M, U, V = _spike_trap()
    # Exactly orthogonal warm start: the spike stays invisible to power steps.
    for k in (1, 2):
        pw = power_iteration(M, V[:, 1], k)
        assert float(pw.sigma) < 1.1, "documented failure mode: 100x underestimate"


def test_trap_guard_detects_and_recovers():
    M, U, V = _spike_trap()
    r = hybrid_guard_estimate(M, V[:, 1], torch.tensor(1.0, dtype=F64))
    assert int(r.reasons) & REASON_LOWER_INCONSISTENT
    assert float(r.sigma_safe) >= 100.0 / 1.05
    assert float(r.sigma_safe) <= float(frobenius_upper(M)) * (1 + 1e-6)
    assert abs(float((r.v * V[:, 0]).sum())) > 0.99  # warm start now on the spike


@pytest.mark.parametrize("dkey", ["fp16", "bf16"])
def test_trap_gram_scaler_certified_and_circuit_accurate(dkey):
    M, U, V = _spike_trap()
    sc = WarmSpectralScaler(u_basin=1.05)
    sc.v = U[:, 1].clone()  # Gram-space trap: orthogonal to the spike direction
    cfg = CIRCUITS["SP4-1r1-l0.02-bf16" if dkey == "bf16" else "SP4-1r1-l0.02-fp16"]
    Xr, O = run(M, cfg, dkey, sc)
    x_max = float(smax(Xr) / sc.last["sigma_hat"])
    assert x_max <= 1.05 * (1 + 1e-3)
    assert torch.isfinite(O).all()
    assert float(smax(O)) <= 1.1
    assert abs(1 - float(U[:, 0] @ O @ V[:, 0])) < 0.02  # spike direction polished


def test_silent_counterexample_task_formula_vs_certified():
    M, v_trap, s1 = spectral_counterexample()
    lit = hybrid_guard_estimate(M, v_trap, torch.tensor(1.2, dtype=F64))
    assert int(lit.reasons) == 0, "no trigger fires on the counterexample"
    assert s1 / float(lit.sigma_safe) > 2.0, "documented failure: overshoot ~2.4"

    # The overshoot drives any registered circuit into its blow-up region.
    cfg = CIRCUITS["SP4-1r1-l0.02-fp16"]
    O = polar_spectral_circuit(M / lit.sigma_safe, cfg, prescale="none", dtype=torch.float16, emulate=True)
    assert (not torch.isfinite(O).all()) or float(smax(O.to(F64))) > 10

    cert = hybrid_guard_estimate(M, v_trap, torch.tensor(1.2, dtype=F64), GuardConfig(u_basin=1.05))
    assert s1 / float(cert.sigma_safe) <= 1.05 * (1 + 1e-6)

    sc = WarmSpectralScaler(u_basin=1.05)
    sc.v = torch.linalg.svd(M).U[:, 1].clone()
    Xr, O = run(M, cfg, "fp16", sc)
    assert float(smax(Xr) / sc.last["sigma_hat"]) <= 1.05 * (1 + 1e-3)
    assert torch.isfinite(O).all()
    assert float(torch.linalg.matrix_norm(O - svd_polar(Xr), 2)) < 0.02


# --------------------------------------------------------------------------
# Extreme condition numbers and aspect ratios
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dkey,name", [("fp16", n) for n in FP16_CIRCUITS] + [("bf16", n) for n in BF16_CIRCUITS])
@pytest.mark.parametrize("alpha,tall", [(1, False), (4, True), (8, False)])
def test_kappa_1e6_bounded_and_accurate(dkey, name, alpha, tall):
    n = 96
    s = np.geomspace(3.7, 3.7e-6, n)  # kappa = 1e6, arbitrary global scale
    X = from_sv(s, n, alpha * n, seed=alpha + 10 * tall)
    if tall:
        X = X.mT.contiguous()
    cfg = CIRCUITS[name]
    Xr, O = run(X, cfg, dkey, WarmSpectralScaler(u_basin=cfg.u))
    assert O.shape == X.shape
    assert torch.isfinite(O).all()
    assert float(smax(O)) <= 1.08
    assert subspace_err(Xr, O, cfg.ell) <= ERR_LIMIT[dkey]


@pytest.mark.parametrize("alpha", [1, 2, 4, 8])
@pytest.mark.parametrize("tall", [False, True])
@pytest.mark.parametrize("dkey", ["fp16", "bf16"])
def test_aspect_ratios_batched_vs_fp64_svd(alpha, tall, dkey):
    n, ell = 48, 0.02
    name = "SP4-1r1-l0.02-" + dkey
    rng = np.random.default_rng(alpha)
    Xs = []
    for b in range(3):
        s = np.exp(rng.uniform(np.log(ell * 1.05), 0.0, n)) * 10.0 ** (b - 1)
        X = from_sv(s, n, alpha * n, seed=100 * alpha + b)
        Xs.append(X.mT if tall else X)
    X = torch.stack(Xs)
    sc = WarmSpectralScaler(u_basin=1.05)
    Xr, O = run(X, name, dkey, sc)
    assert O.shape == X.shape
    err = torch.linalg.matrix_norm(O - svd_polar(Xr), 2)
    assert torch.isfinite(O).all()
    assert float(err.max()) <= ERR_LIMIT[dkey]
    # Exact arithmetic reaches the design error.
    _, O64 = run(X, name, None, WarmSpectralScaler(u_basin=1.05))
    e64 = float(torch.linalg.matrix_norm(O64 - svd_polar(X), 2).max())
    assert e64 <= cs.exact_error(CIRCUITS[name]) + 1e-6


# --------------------------------------------------------------------------
# Streaming trajectory drift
# --------------------------------------------------------------------------


def test_streaming_drift_certified_and_tight():
    r = cs.streaming_summary(T=120, n=64, alpha=4, seed=3)
    assert r["certified_max_overshoot"] <= 1.05 * (1 + 1e-3)
    assert r["certified_conservatism_median"] <= 1.10
    assert r["certified_conservatism_max"] <= 1.25
    # Uncertified 1-step power iteration misses the spike event by more than the
    # basin allows (13% here, 38% in the n = 128 run of the results JSON).
    assert r["power1_right_underestimate_max"] > 1.05


def test_streaming_rotation_with_circuit_no_nans():
    g = torch.Generator().manual_seed(7)
    n, N = 48, 192
    sc = WarmSpectralScaler(u_basin=1.05)
    M = torch.zeros(n, N, dtype=F64)
    A, B = torch.randn(n, 4, generator=g, dtype=F64), torch.randn(N, 4, generator=g, dtype=F64)
    for t in range(40):
        # Rotate the dominant subspace fast and flip the top direction mid-way.
        A = A + 0.3 * torch.randn(n, 4, generator=g, dtype=F64)
        B = B + 0.3 * torch.randn(N, 4, generator=g, dtype=F64)
        G = A @ B.mT + 0.1 * torch.randn(n, N, generator=g, dtype=F64)
        if t == 20:
            G = G + 50 * torch.outer(torch.randn(n, generator=g, dtype=F64), torch.randn(N, generator=g, dtype=F64))
        M = 0.9 * M + G
        Xr, O = run(M, "SP4-1r1-l0.02-bf16", "bf16", sc)
        assert float(smax(Xr) / sc.last["sigma_hat"]) <= 1.05 * (1 + 1e-3)
        assert torch.isfinite(O).all() and float(smax(O)) <= 1.08


# --------------------------------------------------------------------------
# Low precision: emulated and native
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dkey,name", [("fp16", n) for n in FP16_CIRCUITS] + [("bf16", n) for n in BF16_CIRCUITS])
def test_interval_accuracy_emulated(dkey, name):
    cfg = CIRCUITS[name]
    n = 128
    s = cs.interval_spectrum(cfg.ell, cfg.u, n, seed=77)
    X = from_sv(s, n, 2 * n, seed=78)
    Xr, O = run(X, cfg, dkey, prescale="none")
    assert torch.isfinite(O).all()
    assert float(torch.linalg.matrix_norm(O - svd_polar(Xr), 2)) <= ERR_LIMIT[dkey]


def _devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dkey", ["fp16", "bf16"])
def test_native_low_precision_devices(device, dkey):
    dt = LOW[dkey]
    n = 64
    s = np.r_[np.geomspace(1.0, 0.03, n // 2), np.full(n // 2, 1e-7)]
    X = torch.stack([from_sv(s, n, 3 * n, seed=s_) for s_ in (1, 2)])
    Xd = X.to(device=device, dtype=dt)
    try:
        O = polar_spectral_circuit(Xd, "SP4-1r1-l0.02-" + dkey, WarmSpectralScaler(u_basin=1.05))
    except (RuntimeError, TypeError) as e:  # dtype unsupported on this backend
        pytest.skip(f"{device}/{dkey} unsupported: {e}")
    assert O.dtype == dt and O.device.type == device
    O = O.cpu().to(F64)
    Xr = Xd.cpu().to(F64)
    assert torch.isfinite(O).all()
    assert float(smax(O).max()) <= 1.1
    assert subspace_err(Xr, O, 0.02) <= 0.12


def test_zero_and_degenerate_batch_members():
    X = torch.zeros(3, 16, 40, dtype=F64)
    X[1] = from_sv(np.geomspace(1, 0.05, 16), 16, 40, seed=0)
    X[2, 0, 0] = 1e-30  # rank-1 and tiny
    for dkey in ("fp16", "bf16", None):
        _, O = run(X, "SP3-1r1-l0.05", dkey, WarmSpectralScaler())
        assert torch.isfinite(O).all()
        assert float(O[0].abs().max()) == 0.0


# --------------------------------------------------------------------------
# Minimax design, basin, and cost model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lo,hi", [(0.02, 1.05), (0.3, 1.6), (0.9, 1.1)])
def test_minimax_equioscillates(lo, hi):
    c, E = cs.minimax_odd(lo, hi, 5)
    assert cs.alternation_count(c, lo, hi) >= 4  # 3 coefficients -> 4 points
    x = cs.design_grid(lo, hi, 20000)
    assert abs(np.abs(1 - cs.odd_eval(c, x)).max() - E) < 1e-6


def test_greedy_error_decreases_with_steps():
    errs = [cs.exact_error(CircuitConfig("g", cs.greedy_circuit(0.01, 1.05, [5] * T), (), 0.01, 1.05))
            for T in range(1, 6)]
    assert all(a > b for a, b in zip(errs, errs[1:]))
    assert errs[3] < 0.02  # 4 steps suffice for ell = 0.01 in exact arithmetic


def test_registered_circuits_have_overshoot_margin():
    for name, cfg in CIRCUITS.items():
        e = cs.exact_error(cfg)
        assert cs.exact_error(cfg, u=cfg.u * cs.OVERSHOOT) <= e + 1e-3, name
        assert cs.stability_basin(cfg, max(0.05, 2 * e)) >= cfg.u * cs.OVERSHOOT, name


def test_uninflated_greedy_explodes_just_past_u():
    """Documented failure: zero-margin greedy circuits diverge at 0.5% overshoot."""
    cfg = CircuitConfig("g", cs.greedy_circuit(0.02, 1.05, [5] * 4), (), 0.02, 1.05)
    assert abs(cfg.poly(1.05) - 1) < 1e-3
    assert cfg.poly(1.05 * 1.005) > 10


def test_flop_model():
    q = (1.0, -1.0, 0.5)
    for T in range(2, 7):
        assert gram_square_products([q] * T) == 4 * T - 3
    assert gram_square_products([q]) == 1
    for a in (1, 2, 4, 8):
        assert CIRCUITS["SP4-1r1-l0.02-fp16"].flops(a) == cs.ps2_55_flops(a)  # 6a + 10
        assert CIRCUITS["SP3-0r-l0.05-fp16"].flops(a) == 3 * a + 9
        assert CIRCUITS["SP3-1r1-l0.05"].flops(a) == 6 * a + 6


def test_results_json_matches_registry():
    import json
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "results" / "circuit_search_cycle1.json"
    if not path.exists():
        pytest.skip("results file not generated")
    data = json.loads(path.read_text())
    by_name = {c["name"]: c for c in data["circuits"]}
    for cfg in CIRCUITS.values():
        src = cfg.notes.split(";")[0]
        assert src in by_name, src
        assert np.allclose(np.array(by_name[src]["coefficients"]), np.array(cfg.steps), rtol=0, atol=0)
