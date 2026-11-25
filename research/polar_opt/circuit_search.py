"""Search for short polar circuits on spectrally pre-scaled intervals.

Run ``python -m research.polar_opt.circuit_search`` to write
``results/circuit_search_cycle1.json``.

Pipeline
--------
1. Minimax design (exact arithmetic). ``minimax_odd`` solves the discrete
   minimax problem ``min_p max_x |1 - p(x)|`` for an odd polynomial by linear
   programming on a dense geometric+linear grid. ``greedy_circuit`` composes
   step-wise optimal polynomials, the Polar Express construction, which is
   optimal for the composed problem in exact arithmetic.
2. Structure enumeration. One coefficient sequence can run as one Gram pass
   (0 restarts, ``3 alpha + k``) or split by restarts (``3 alpha`` per pass).
3. Rounding-aware evaluation. ``circuits.polar_spectral_circuit`` runs with
   ``emulate=True`` (operands and stored results rounded to FP16 or BF16, fp32
   accumulation). Error is ``||O - U V^T||_2`` in fp64 against the SVD polar of
   the rounded input. Training matrices have spectra that fill ``[ell, u]``,
   plus one ``kappa = 1e6`` matrix that exposes Gram-space blow-up.
4. Refinement. Nelder-Mead (or differential evolution) on the coefficients
   minimizes ``max(simulated errors, exact error on [ell, u])``.
5. Held-out validation with fresh seeds, ``n = 256``, ``alpha in {1, 2, 4, 8}``,
   wide and tall shapes. A refined circuit replaces the greedy one only when
   its validation error is lower.

Baselines are computed with the same simulator: Jordan NS5 (published
coefficients) and a Polar-Express-style greedy circuit re-derived here for
``[1e-3, 1]``. The official Polar Express and Polar Switch coefficient lists
are not used. PS2-55/PS3-55 appear only as the FLOP formula
``(6 alpha + 10) n^3`` supplied with the task.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy
import torch
from scipy.optimize import differential_evolution, linprog, minimize

from .circuits import (
    JORDAN_NS5,
    CircuitConfig,
    WarmSpectralScaler,
    polar_spectral_circuit,
    polar_xspace_circuit,
    svd_polar,
)
from .spectral_scaling import (
    GuardConfig,
    hybrid_guard_estimate,
    power_iteration,
    spectral_counterexample,
)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
ALPHAS = (1, 2, 4, 8)

# --------------------------------------------------------------------------
# Exact-arithmetic minimax design
# --------------------------------------------------------------------------


def design_grid(lo: float, hi: float, n: int = 2000) -> np.ndarray:
    return np.unique(np.concatenate([np.geomspace(lo, hi, n), np.linspace(lo, hi, n)]))


def odd_eval(c, x):
    x2 = x * x
    h = np.zeros_like(x)
    for cj in reversed(c):
        h = h * x2 + cj
    return x * h


def minimax_odd(lo: float, hi: float, degree: int = 5) -> tuple[np.ndarray, float]:
    """Best odd polynomial of ``degree`` approximating 1 on ``[lo, hi]`` (LP)."""
    x = design_grid(lo, hi)
    k = (degree + 1) // 2
    Phi = np.stack([x ** (2 * j + 1) for j in range(k)], 1)
    one = np.ones((len(x), 1))
    A = np.vstack([np.hstack([Phi, -one]), np.hstack([-Phi, -one])])
    b = np.concatenate([np.ones(len(x)), -np.ones(len(x))])
    res = linprog(
        np.r_[np.zeros(k), 1.0], A_ub=A, b_ub=b,
        bounds=[(None, None)] * k + [(0, None)], method="highs",
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[:k], float(res.x[k])


def alternation_count(c, lo: float, hi: float) -> int:
    """Number of sign-alternating extrema of ``1 - p`` at the max error level."""
    x = design_grid(lo, hi, 20000)
    e = 1 - odd_eval(c, x)
    E = np.abs(e).max()
    idx = np.where(np.abs(e) > E * (1 - 1e-3))[0]
    signs, last = [], None
    for i in idx:
        s = np.sign(e[i])
        if s != last:
            signs.append(s)
            last = s
    return len(signs)


def greedy_circuit(ell: float, u: float, degrees, inflate: float = 1.0):
    """Greedy composition; ``inflate`` widens each design upper edge (safety).

    With ``inflate = 1`` every step puts its maximum ``1 + E_t`` exactly at the
    next step's design edge, so the composed polynomial explodes just past
    ``u`` (for example ``P(1.055) ~ 1e7`` for ``ell = 0.02, u = 1.05, T = 4``).
    ``inflate = 1 + eta`` designs step ``t`` on ``[lo_t, hi_t (1 + eta)]`` and
    gives every step an overshoot margin.
    """
    lo, hi, steps = ell, u, []
    for d in degrees:
        c, _ = minimax_odd(lo, hi * inflate, d)
        y = odd_eval(c, design_grid(lo, hi))
        lo, hi = float(y.min()), float(y.max())
        steps.append(tuple(float(v) for v in c))
    return tuple(steps)


def exact_error(cfg: CircuitConfig, ell: float | None = None, u: float | None = None) -> float:
    lo, hi = ell or cfg.ell, u or cfg.u
    x = design_grid(lo, hi, 20000)
    return float(np.abs(1 - cfg.poly(x)).max())


def stability_basin(cfg: CircuitConfig, tol: float, x_max: float = 4.0) -> float:
    """Largest ``u*`` with ``|1 - P(x)| <= tol`` on ``[u, u*]`` (overshoot basin).

    Returns ``nan`` if the design edge already violates ``tol``.
    """
    x = np.linspace(cfg.u, x_max, 40001)
    with np.errstate(over="ignore", invalid="ignore"):
        bad = ~(np.abs(1 - cfg.poly(x)) <= tol)
    if not bad.any():
        return float(x_max)
    i = int(np.argmax(bad))
    return float(x[i - 1]) if i > 0 else float("nan")


def basin_profile(cfg: CircuitConfig, rel=(0.001, 0.005, 0.01, 0.02, 0.05)) -> dict:
    """``P(u (1 + d))`` for small overshoots ``d``: how fast the circuit leaves 1."""
    with np.errstate(over="ignore", invalid="ignore"):
        return {f"+{d:g}": float(cfg.poly(cfg.u * (1 + d))) for d in rel}


# Relative overshoot past ``u`` that every circuit must tolerate. Low-precision
# rounding of the scaled matrix moves singular values by O(eps).
OVERSHOOT = 1.005
INFLATES = (1.0, 1.01, 1.02)


# --------------------------------------------------------------------------
# Test matrices and simulated evaluation
# --------------------------------------------------------------------------


def make_matrix(sigmas, alpha: int, seed: int, tall: bool = False) -> torch.Tensor:
    """``U diag(sigmas) V^T`` in fp64, shape ``n x alpha n`` (or transposed)."""
    s = torch.as_tensor(np.array(sigmas, dtype=float), dtype=torch.float64)
    n = s.numel()
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64)).Q
    V = torch.linalg.qr(torch.randn(alpha * n, n, generator=g, dtype=torch.float64)).Q
    X = (U * s) @ V.mT
    return X.mT.contiguous() if tall else X


def interval_spectrum(ell: float, u: float, n: int, seed: int) -> np.ndarray:
    """Endpoints, clusters at both edges, and a random log-uniform fill."""
    rng = np.random.default_rng(seed)
    k = max(n // 8, 1)
    fill = np.exp(rng.uniform(np.log(ell), np.log(u), n - 2 - 2 * k))
    return np.sort(np.r_[ell, u, ell * (1 + 0.05 * rng.random(k)), u * (1 - 0.01 * rng.random(k)), fill])[::-1]


def kappa_spectrum(n: int, kappa: float = 1e6) -> np.ndarray:
    return np.geomspace(1.0, 1.0 / kappa, n)


def momentum_spectrum(n: int, p: float, kappa: float) -> np.ndarray:
    return np.maximum(np.arange(1, n + 1, dtype=float) ** (-p), 1.0 / kappa)


def _round_input(X: torch.Tensor, dtype) -> torch.Tensor:
    return X.to(dtype).to(torch.float64)


def eval_case(cfg: CircuitConfig, case: dict, dtype) -> dict:
    """Simulated error of ``cfg`` on one prepared case."""
    Xr = case["X"][dtype]
    if case["kind"] == "interval":
        O = polar_spectral_circuit(Xr, cfg, dtype=dtype, emulate=True, prescale="none")
    else:
        O = polar_spectral_circuit(Xr, cfg, dtype=dtype, emulate=True)
    O = O.to(torch.float64)
    finite = bool(torch.isfinite(O).all())
    if not finite:
        return {"err": float("inf"), "finite": False, "norm": float("inf")}
    norm = float(torch.linalg.matrix_norm(O, 2))
    if case["kind"] == "interval":
        err = float(torch.linalg.matrix_norm(O - case["P"][dtype], 2))
    else:
        # Error on singular directions inside the design range, plus overshoot.
        U, S, Vh = case["svd"][dtype]
        d = torch.diagonal(U.mT @ O @ Vh.mT, dim1=-2, dim2=-1)
        good = S >= 2 * cfg.ell * S[..., :1]
        err = max(float((1 - d[good]).abs().max()), norm - 1.0)
    return {"err": err, "finite": True, "norm": norm}


def prepare_cases(specs) -> list[dict]:
    cases = []
    for sp in specs:
        X = make_matrix(sp["sigmas"], sp["alpha"], sp["seed"], sp.get("tall", False))
        c = {"kind": sp["kind"], "X": {}, "P": {}, "svd": {}, "label": sp["label"]}
        for dt in DTYPES.values():
            Xr = _round_input(X, dt)
            c["X"][dt] = Xr
            if sp["kind"] == "interval":
                c["P"][dt] = svd_polar(Xr)
            else:
                c["svd"][dt] = torch.linalg.svd(Xr, full_matrices=False)
        cases.append(c)
    return cases


def train_specs(ell: float, u: float, n: int = 64) -> list[dict]:
    return [
        {"kind": "interval", "sigmas": interval_spectrum(ell, u, n, 11), "alpha": 1, "seed": 101, "label": "int-a1"},
        {"kind": "interval", "sigmas": interval_spectrum(ell, u, n, 12), "alpha": 4, "seed": 102, "label": "int-a4"},
        {"kind": "interval", "sigmas": interval_spectrum(ell, u, n, 13), "alpha": 2, "seed": 104, "tall": True,
         "label": "int-a2-tall"},
        {"kind": "interval", "sigmas": interval_spectrum(ell, u, n, 14), "alpha": 8, "seed": 105, "label": "int-a8"},
        {"kind": "kappa", "sigmas": kappa_spectrum(n), "alpha": 4, "seed": 103, "label": "kappa1e6-a4"},
        {"kind": "interval", "sigmas": interval_spectrum(ell, u * OVERSHOOT, n, 15), "alpha": 4, "seed": 106,
         "label": "overshoot-a4"},
    ]


def validation_specs(ell: float, u: float, n: int = 256) -> list[dict]:
    specs = []
    for i, a in enumerate(ALPHAS):
        specs.append({"kind": "interval", "sigmas": interval_spectrum(ell, u, n, 900 + i), "alpha": a,
                      "seed": 9000 + i, "tall": i % 2 == 1, "label": f"int-a{a}"})
    for i, a in enumerate((1, 8)):
        specs.append({"kind": "kappa", "sigmas": kappa_spectrum(n), "alpha": a, "seed": 9100 + i,
                      "tall": i == 1, "label": f"kappa1e6-a{a}"})
    specs.append({"kind": "interval", "sigmas": interval_spectrum(ell, u * OVERSHOOT, n, 950), "alpha": 2,
                  "seed": 9200, "label": "overshoot-a2"})
    return specs


def worst(cfg: CircuitConfig, cases, dtype) -> dict:
    rs = [eval_case(cfg, c, dtype) for c in cases]
    return {
        "worst_err": max(r["err"] for r in rs),
        "all_finite": all(r["finite"] for r in rs),
        "max_norm": max(r["norm"] for r in rs),
        "per_case": {c["label"]: round(r["err"], 6) for c, r in zip(cases, rs)},
    }


# --------------------------------------------------------------------------
# Refinement
# --------------------------------------------------------------------------


def refine(cfg: CircuitConfig, cases, dtype, method: str = "nelder-mead", maxiter: int = 1500):
    """Rounding-aware coefficient refinement; returns ``(config, objective)``."""
    shapes = [len(c) for c in cfg.steps]
    x0 = np.concatenate([np.asarray(c, float) for c in cfg.steps])

    def unpack(x):
        out, i = [], 0
        for k in shapes:
            out.append(tuple(x[i:i + k]))
            i += k
        return out

    def objective(x):
        c = cfg.with_steps(unpack(x))
        try:
            e = max(eval_case(c, case, dtype)["err"] for case in cases)
        except Exception:
            return 1e6
        e = max(e, exact_error(c, u=c.u * OVERSHOOT))
        return e if math.isfinite(e) else 1e6

    f0 = objective(x0)
    if method == "de":
        span = np.maximum(np.abs(x0) * 0.08, 1e-3)
        res = differential_evolution(objective, list(zip(x0 - span, x0 + span)), maxiter=maxiter // 50,
                                     popsize=8, seed=0, polish=False, tol=1e-8)
    else:
        simplex = [x0] + [x0 + np.eye(len(x0))[i] * 0.02 * max(abs(x0[i]), 0.05) for i in range(len(x0))]
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxfev": maxiter, "initial_simplex": np.array(simplex),
                                "xatol": 1e-7, "fatol": 1e-7, "adaptive": True})
    if res.fun < f0:
        return cfg.with_steps(unpack(res.x)), float(res.fun), float(f0)
    return cfg, float(f0), float(f0)


# --------------------------------------------------------------------------
# Cost references
# --------------------------------------------------------------------------


def xspace_flops(T: int, alpha: float, model: str = "sym") -> float:
    return T * ((3 * alpha + 1) if model == "sym" else (4 * alpha + 2))


def ps2_55_flops(alpha: float) -> float:
    return 6 * alpha + 10


def flops_table(cfg: CircuitConfig) -> dict:
    out = {}
    for a in ALPHAS:
        f = cfg.flops(a)
        out[str(a)] = {
            "sym_n3": f,
            "dense_n3": cfg.flops(a, "dense"),
            "vs_PE5_xspace": round(f / xspace_flops(5, a), 4),
            "vs_PE6_xspace": round(f / xspace_flops(6, a), 4),
            "vs_PS2_55_brief": round(f / ps2_55_flops(a), 4),
        }
    return out


# --------------------------------------------------------------------------
# Search driver
# --------------------------------------------------------------------------

ELLS = (0.005, 0.01, 0.02, 0.05)
US = (1.02, 1.05, 1.10)
TS = (2, 3, 4, 5, 6)


def structures(T: int):
    yield ()
    for r in range(1, min(T, 4)):
        yield (r,)


def _name(ell, u, T, restarts, inflate=1.0):
    r = "0r" if not restarts else f"1r@{restarts[0]}"
    return f"SP{T}-{r}-l{ell:g}-u{u:g}-m{inflate - 1:g}"


def _screen_one(args):
    ell, u = args
    torch.set_num_threads(1)
    cases = prepare_cases(train_specs(ell, u))
    rows = []
    for T in TS:
        for inf in INFLATES:
            steps = greedy_circuit(ell, u, [5] * T, inflate=inf)
            for rs in structures(T):
                cfg = CircuitConfig(_name(ell, u, T, rs, inf), steps, rs, ell, u)
                row = {"name": cfg.name, "ell": ell, "u": u, "T": T, "restarts": list(rs), "inflate": inf,
                       "exact_err": exact_error(cfg), "exact_err_overshoot": exact_error(cfg, u=u * OVERSHOOT),
                       "flops_a4_sym": cfg.flops(4), "square_products": cfg.square_products(),
                       "passes": len(cfg.passes())}
                for key, dt in DTYPES.items():
                    w = worst(cfg, cases, dt)
                    row[f"train_{key}"] = w["worst_err"]
                    row[f"train_{key}_finite"] = w["all_finite"]
                rows.append(row)
    return rows


def _pareto(rows, key, err_cap=0.15):
    """Cost/error Pareto front; keep the cheapest point and the 3 most accurate."""
    rows = sorted((r for r in rows if r[key] < err_cap), key=lambda r: (r["flops_a4_sym"], r[key]))
    front, best = [], float("inf")
    for r in rows:
        if r[key] < best * 0.9:
            front.append(r)
            best = r[key]
    keep = front[:1] + front[-3:]
    return [r for i, r in enumerate(keep) if r not in keep[:i]]


def _refine_one(job):
    torch.set_num_threads(1)
    ell, u, T, rs, inf, key, method, maxiter = job
    dt = DTYPES[key]
    steps = greedy_circuit(ell, u, [5] * T, inflate=inf)
    base = CircuitConfig(_name(ell, u, T, rs, inf), steps, tuple(rs), ell, u)
    train = prepare_cases(train_specs(ell, u))
    t0 = time.time()
    ref, f_ref, f0 = refine(base, train, dt, method=method, maxiter=maxiter)
    val = prepare_cases(validation_specs(ell, u))
    v_base = worst(base, val, dt)
    v_ref = worst(ref, val, dt)
    other = "bf16" if key == "fp16" else "fp16"
    # Accept refined coefficients only for a clear (>= 5%) held-out gain.
    use_ref = v_ref["worst_err"] < 0.95 * v_base["worst_err"] and v_ref["all_finite"]
    final = ref if use_ref else base
    final = CircuitConfig(final.name + f"-{key}" + ("-ref" if use_ref else ""), final.steps,
                          final.restarts, ell, u, notes=f"tuned for {key}" if use_ref else "greedy")
    v_final = v_ref if use_ref else v_base
    v_other = worst(final, val, DTYPES[other])
    return {
        "name": final.name,
        "target_dtype": key,
        "ell": ell,
        "u": u,
        "T": T,
        "restarts": list(rs),
        "inflate": inf,
        "refined": use_ref,
        "method": method,
        "seconds": round(time.time() - t0, 1),
        "coefficients": [list(c) for c in final.steps],
        "greedy_coefficients": [list(c) for c in steps],
        "exact_err": exact_error(final),
        "exact_err_overshoot": exact_error(final, u=u * OVERSHOOT),
        "basin_profile": basin_profile(final),
        "basin_u_tol0.05": stability_basin(final, 0.05),
        "basin_u_tol0.2": stability_basin(final, 0.2),
        "train_objective_greedy": f0,
        "train_objective_refined": f_ref,
        "validation": {key: v_final, f"{key}_greedy": v_base, other: v_other},
        "flops": flops_table(final),
        "square_products": final.square_products(),
        "passes": len(final.passes()),
    }


def baselines() -> dict:
    pe = {T: greedy_circuit(1e-3, 1.0, [5] * T, inflate=1.01) for T in (5, 6)}
    out = {"jordan_ns5": {"coefficients": [list(c) for c in JORDAN_NS5], "source": "github.com/KellerJordan/Muon",
                          "flops": {str(a): xspace_flops(5, a) for a in ALPHAS}}}
    for T, steps in pe.items():
        cfg = CircuitConfig(f"PEg{T}", steps, (), 1e-3, 1.0)
        out[f"pe_greedy_{T}"] = {
            "coefficients": [list(c) for c in steps],
            "source": "re-derived greedy minimax on [1e-3, 1] with 1.01 inflation; NOT the official Polar Express list",
            "exact_err_[1e-3,1]": exact_error(cfg),
            "flops_xspace": {str(a): xspace_flops(T, a) for a in ALPHAS},
        }
    out["ps2_55"] = {"flops": {str(a): ps2_55_flops(a) for a in ALPHAS},
                     "source": "(6 alpha + 10) n^3 formula supplied by the task brief; coefficients unavailable here, "
                               "accuracy not compared"}
    out["ps3_55"] = {"flops": None, "source": "no cost formula or coefficients available; not compared"}
    return out


def end_to_end(circuits: list[dict]) -> list[dict]:
    """Same matrices, each method with its own scaling. Full-spectrum polar error."""
    torch.set_num_threads(4)
    pe = {T: greedy_circuit(1e-3, 1.0, [5] * T, inflate=1.01) for T in (5, 6)}
    rows = []
    for kappa in (20.0, 50.0):
        for p in (0.3, 1.0):
            for a in (1, 4):
                X = make_matrix(momentum_spectrum(256, p, kappa), a, seed=int(7000 + kappa + 10 * p + a))
                for key, dt in DTYPES.items():
                    Xr = _round_input(X, dt)
                    P = svd_polar(Xr)

                    def err(O):
                        O = O.to(torch.float64)
                        return float(torch.linalg.matrix_norm(O - P, 2)) if torch.isfinite(O).all() else float("inf")

                    row = {"kappa": kappa, "decay_p": p, "alpha": a, "dtype": key,
                           "frobenius_over_sigma_max": float(torch.linalg.matrix_norm(Xr) / torch.linalg.matrix_norm(Xr, 2))}
                    row["NS5_frob"] = {"err": err(polar_xspace_circuit(Xr, JORDAN_NS5, dtype=dt, emulate=True)),
                                       "flops": xspace_flops(5, a)}
                    for T, st in pe.items():
                        row[f"PEg{T}_frob_xspace"] = {
                            "err": err(polar_xspace_circuit(Xr, st, dtype=dt, emulate=True, safety=1.01)),
                            "flops": xspace_flops(T, a)}
                    for T, rs in ((6, (2,)), (7, (2, 4))):
                        cfg = CircuitConfig(f"PEg{T}_frob_gram", greedy_circuit(1e-3, 1.0, [5] * T, inflate=1.01),
                                            rs, 1e-3, 1.0, prescale="frobenius")
                        row[f"PEg{T}_frob_gram_r{'-'.join(map(str, rs))}"] = {
                            "err": err(polar_spectral_circuit(Xr, cfg, dtype=dt, emulate=True)),
                            "flops": cfg.flops(a)}
                    for c in circuits:
                        if c["target_dtype"] != key:
                            continue
                        cfg = CircuitConfig(c["name"], tuple(tuple(v) for v in c["coefficients"]),
                                            tuple(c["restarts"]), c["ell"], c["u"])
                        sc = WarmSpectralScaler(u_basin=c["u"], power_steps=2)
                        e = err(polar_spectral_circuit(Xr, cfg, sc, dtype=dt, emulate=True))
                        smax = float(torch.linalg.matrix_norm(Xr, 2))
                        row[c["name"]] = {"err": e, "flops": cfg.flops(a),
                                          "x_max": smax / float(sc.last["sigma_hat"]),
                                          "x_min": float(torch.linalg.svdvals(Xr)[-1]) / float(sc.last["sigma_hat"])}
                    rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Adversarial stress summaries (mirrors tests/test_spectral_bounds.py)
# --------------------------------------------------------------------------


def _orth(n, g):
    return torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64)).Q


def stress_summary() -> dict:
    torch.set_num_threads(4)
    out = {}
    g = torch.Generator().manual_seed(5)
    n = 256
    U, V = _orth(n, g), _orth(n, g)
    s = torch.ones(n, dtype=torch.float64)
    s[0] = 100.0
    M = (U * s) @ V.mT
    trap = V[:, 1]
    p1 = power_iteration(M, trap, 1)
    p2 = power_iteration(M, trap, 2)
    lit = hybrid_guard_estimate(M, trap, torch.tensor(1.0, dtype=torch.float64))
    out["orthogonal_trap_100x"] = {
        "sigma_max": 100.0,
        "power1_lower": float(p1.sigma), "power2_lower": float(p2.sigma),
        "guard_sigma_safe": float(lit.sigma_safe), "guard_reasons_bitmask": int(lit.reasons),
        "guard_overshoot": 100.0 / float(lit.sigma_safe),
    }
    Mc, vt, smax = spectral_counterexample()
    lit = hybrid_guard_estimate(Mc, vt, torch.tensor(1.2, dtype=torch.float64))
    cert = hybrid_guard_estimate(Mc, vt, torch.tensor(1.2, dtype=torch.float64), GuardConfig(u_basin=1.05))
    out["silent_counterexample_n512"] = {
        "sigma_max": smax,
        "task_formula_sigma_safe": float(lit.sigma_safe), "task_formula_reasons": int(lit.reasons),
        "task_formula_overshoot": smax / float(lit.sigma_safe),
        "certified_rowcol_fro_overshoot": smax / float(cert.sigma_safe),
        "certified_rowcol_fro_sigma_hat_over_sigma_max": float(cert.sigma_safe) / smax,
    }
    sc = WarmSpectralScaler(u_basin=1.05)
    X0 = Mc / torch.linalg.matrix_norm(Mc)
    G = X0 @ X0.mT
    a2 = torch.linalg.matrix_norm(G)
    G = G / a2
    sc.v = torch.linalg.svd(Mc).U[:, 1]  # left-space trap vector
    sgram = sc(G, G @ G)
    smax_rel = float(torch.linalg.matrix_norm(X0, 2) / a2.sqrt())
    out["silent_counterexample_n512"]["gram_certified_overshoot"] = smax_rel / float(sgram)
    out["silent_counterexample_n512"]["gram_certified_sigma_hat_over_sigma_max"] = float(sgram) / smax_rel
    return out


def streaming_summary(T: int = 300, n: int = 128, alpha: int = 4, seed: int = 0) -> dict:
    """Muon-like momentum stream with drifting low-rank signal and a spike event."""
    torch.set_num_threads(4)
    g = torch.Generator().manual_seed(seed)
    N = alpha * n
    beta = 0.95
    r = 8
    A = torch.randn(n, r, generator=g, dtype=torch.float64)
    B = torch.randn(N, r, generator=g, dtype=torch.float64)
    M = torch.zeros(n, N, dtype=torch.float64)
    sc = WarmSpectralScaler(u_basin=1.05, power_steps=1)
    ratios, cons, p1 = [], [], []
    v_right = None
    for t in range(T):
        A = A + 0.05 * torch.randn(n, r, generator=g, dtype=torch.float64)
        B = B + 0.05 * torch.randn(N, r, generator=g, dtype=torch.float64)
        Gt = (A @ B.mT) / math.sqrt(N) + torch.randn(n, N, generator=g, dtype=torch.float64) / math.sqrt(N)
        if t == T // 2:
            uu = torch.randn(n, generator=g, dtype=torch.float64)
            vv = torch.randn(N, generator=g, dtype=torch.float64)
            Gt = Gt + 200 * torch.outer(uu / uu.norm(), vv / vv.norm())
        M = beta * M + (1 - beta) * Gt
        X0 = M / torch.linalg.matrix_norm(M)
        G = X0 @ X0.mT
        a2 = torch.linalg.matrix_norm(G)
        G = G / a2
        s = sc(G, G @ G)
        true = float(torch.linalg.matrix_norm(X0, 2) / a2.sqrt())
        ratios.append(true / float(s))
        cons.append(float(s) / true)
        if v_right is not None:
            pr = power_iteration(M, v_right, 1)
            p1.append(float(torch.linalg.matrix_norm(M, 2)) / float(pr.sigma))
            v_right = pr.v
        else:
            v_right = torch.randn(N, generator=g, dtype=torch.float64)
    q = lambda a, x: float(np.quantile(a, x))
    return {
        "steps": T, "n": n, "alpha": alpha, "beta": beta, "spike_step": T // 2,
        "certified_max_overshoot": max(ratios),
        "certified_conservatism_median": q(cons, 0.5), "certified_conservatism_p99": q(cons, 0.99),
        "certified_conservatism_max": max(cons),
        "power1_right_underestimate_median": q(p1, 0.5), "power1_right_underestimate_max": max(p1),
        "calibrated_gamma_power1_p100": max(p1) * 1.02,
    }


def headline(circuits: list[dict], e2e: list[dict]) -> list[dict]:
    """Per method and dtype: error range over the end-to-end matrices it is designed for.

    A spectral circuit with design ``ell`` is scored only on matrices with
    ``1 / kappa >= ell`` (its validity range). Baselines are scored on all.
    """
    out = []
    for key in DTYPES:
        rows = [r for r in e2e if r["dtype"] == key]
        names = [k for k, v in rows[0].items() if isinstance(v, dict)]
        for nm in names:
            c = next((c for c in circuits if c["name"] == nm), None)
            valid = [r for r in rows if c is None or 1.0 / r["kappa"] >= c["ell"] - 1e-12]
            if not valid:
                continue
            errs = [r[nm]["err"] for r in valid]
            a4 = [r[nm]["flops"] for r in valid if r["alpha"] == 4]
            out.append({"method": nm, "dtype": key, "n_matrices": len(valid),
                        "kappa_max": max(r["kappa"] for r in valid),
                        "err_min": min(errs), "err_max": max(errs),
                        "flops_a4_sym_n3": a4[0] if a4 else None})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results/circuit_search_cycle1.json")
    ap.add_argument("--maxiter", type=int, default=1200)
    ap.add_argument("--method", default="nelder-mead", choices=["nelder-mead", "de"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)
    t0 = time.time()

    exact = []
    for ell in (1e-3, *ELLS):
        for u in (1.0, *US):
            for T in range(1, 8):
                cfg = CircuitConfig("x", greedy_circuit(ell, u, [5] * T), (), ell, u)
                exact.append({"ell": ell, "u": u, "T": T, "exact_err": exact_error(cfg)})

    jobs = [(ell, u) for ell in ELLS for u in US]
    with ProcessPoolExecutor(args.workers) as ex:
        screen = [r for rows in ex.map(_screen_one, jobs) for r in rows]
    print(f"screened {len(screen)} structures in {time.time() - t0:.0f}s", flush=True)

    ref_jobs = []
    for key in DTYPES:
        for ell in ELLS:
            rows = [r for r in screen if r["ell"] == ell and r["u"] == 1.05 and r[f"train_{key}_finite"]]
            for r in _pareto(rows, f"train_{key}"):
                ref_jobs.append((ell, 1.05, r["T"], tuple(r["restarts"]), r["inflate"], key, args.method,
                                 args.maxiter))
    with ProcessPoolExecutor(args.workers) as ex:
        refined = list(ex.map(_refine_one, ref_jobs))
    print(f"refined {len(refined)} circuits in {time.time() - t0:.0f}s", flush=True)

    e2e = end_to_end(refined)
    stress = stress_summary()
    stream = streaming_summary()

    result = {
        "experiment": "T1-03 spectral pre-scaled polar circuits (cycle 1)",
        "headline": headline(refined, e2e),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": round(time.time() - t0, 1),
        "env": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                "scipy": scipy.__version__, "machine": platform.machine(), "system": platform.system()},
        "simulation_model": "operands and stored results rounded to fp16/bf16 (RNE), fp32 accumulation and "
                            "epilogue, one rounding per stored matrix; reference = fp64 SVD polar of the rounded input",
        "cost_model": {
            "units": "n^3 with n = min(m, N), alpha = max/min dimension",
            "sym": "SYRK alpha, Q@X 2 alpha, symmetric n x n product 1; Gram quintic pass of T steps = 4T-3 products",
            "dense": "Gram 2 alpha, Q@X 2 alpha, n x n product 2",
            "xspace_quintic_step_sym": "3 alpha + 1",
        },
        "caveats": [
            "All accuracy numbers are from fp16/bf16 rounding emulation on CPU (fp32 accumulate), not GPU kernels; "
            "no wall-clock latency is measured.",
            "Test matrices are synthetic (random singular vectors, prescribed spectra); real Muon momentum "
            "(T1-02) has not been used, so the achievable ell after spectral scaling is unverified.",
            "Polar Express baselines are re-derived greedy circuits, not the official coefficient list. Polar Switch "
            "is compared by the brief's FLOP formula only; PS3-55 is not compared.",
            "FLOPs use the 'sym' model (symmetric n x n products at n^3); the 'dense' model is also reported.",
            "A spectral circuit is only valid for sigma_min / sigma_hat >= ell; below that, directions are "
            "partially orthogonalized (bounded, not polished).",
            "0-restart Gram circuits are unstable in bf16 for T >= 4: rounding makes Gram eigenvalues negative and "
            "y -> y h(y)^2 amplifies them by a_t^2 per step.",
        ],
        "baselines": baselines(),
        "exact_greedy_table": exact,
        "screen": screen,
        "circuits": refined,
        "end_to_end": e2e,
        "adversarial": {"stress": stress, "streaming": stream},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, default=float)
    print(f"wrote {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
