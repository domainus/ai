"""Search short Gram-form polar circuits for spectrally pre-scaled Muon momenta.

Pipeline
--------
1. Degree floor: LP minimax error of the best odd polynomial ``s * q(s^2)`` with
   ``deg q = 2^k`` on ``[lo, 1]``. No circuit with ``k`` squares per Gram pass
   can beat it in exact arithmetic.
2. Chain fit: full-history squaring chains (``circuits.ChainSegment``). Outer
   variables (inside the squares) by variable projection + Gauss-Newton
   (``scipy.optimize.least_squares``) with Lawson reweighting from many random
   starts; output coefficients by an exact LP (HiGHS) with the basin
   constraints ``0 <= f <= 1 + E`` on ``[0, lo]`` and ``0 <= f <= CAP`` on
   ``[1, basin]``.
3. One-restart composites: a short first segment lifts the bottom of the
   spectrum, a second segment polishes from the lifted interval.
4. Rounding-aware selection and polish: the best restarts are re-scored on
   emulated FP16 and BF16 matrices (FP16/BF16 operands, FP32 accumulation, one
   rounding per inner step), then Nelder-Mead tunes the output coefficients on
   training matrices and keeps the change only if held-out matrices improve.
5. Evaluation against PolarExpress, PS2-55 and PS3-55: scalar error and
   stability basin, synthetic matrices (CPU emulation and real CUDA), the 45
   harvested trajectory pairs with the streaming warm estimator, and latency.

Run: ``PYTHONPATH=. python research/polar_opt/circuit_search.py --output results/circuit_search_cycle1.json``
"""

from __future__ import annotations

import os

# One BLAS thread per fit worker; the pool provides the parallelism.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares, linprog, minimize

from research.polar_opt import baselines as bl
from research.polar_opt.circuits import CAP, ChainSegment, GramCircuit, WarmSpectralScaler, run_circuit
from research.polar_opt.spectral_scaling import (
    EstimatorConfig,
    basin_report,
    calibrate_gamma,
    round_to,
)

ROUND = {"fp16": round_to(torch.float16), "bf16": round_to(torch.bfloat16)}
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


# --------------------------------------------------------------------------------------
# 1. Degree floor
# --------------------------------------------------------------------------------------
def degree_floor(lo: float, k: int, n: int = 1500) -> float:
    """Minimax error of the best ``s * q(s^2)``, ``deg q = 2^k``, approximating 1 on ``[lo, 1]``."""
    s = np.geomspace(lo, 1, n)
    V = np.polynomial.chebyshev.chebvander(2 * s * s - 1, 2 ** k) * s[:, None]
    m = V.shape[1]
    A = np.vstack([np.c_[V, -np.ones(n)], np.c_[-V, -np.ones(n)]])
    b = np.r_[np.ones(n), -np.ones(n)]
    r = linprog(np.r_[np.zeros(m), 1], A_ub=A, b_ub=b, bounds=[(None, None)] * m + [(0, None)],
                method="highs")
    return float(r.x[-1]) if r.status == 0 else float("nan")


# --------------------------------------------------------------------------------------
# 2. Chain fit
# --------------------------------------------------------------------------------------
def n_outer(k: int) -> int:
    """Op 0: ``L = R + x``. Op j >= 1: ``L = B_j + sum_{i<j} c_i B_i + c`` (B_j's weight fixed to 1)."""
    return 1 + sum(j + 1 for j in range(1, k))


U_FP16 = 2.0 ** -11
# Weight of the first-order rounding sensitivity. The measure is a worst case
# (constant shift of every L_j by its max) and overestimates the observed FP16
# matrix error by ~35x; u/16 keeps both FP16 and BF16 emulated errors within
# 1e-2 of the exact-arithmetic error in the k=6, lo=0.02 sweep, while rho = 0
# gives FP16 error 1.9 and BF16 error 490 for the same chain length.
RHO = U_FP16 / 16
# On the overshoot interval (1, basin] the circuit must stay in [HI_FLOOR, CAP]:
# bounded (no divergence) and still pointing the right way.
HI_FLOOR = 0.5


def _grid(lo: float, basin: float, k: int):
    """Fit points: [lo, 1] (geometric + Chebyshev in lambda), [0, lo) and (1, basin]."""
    n_cheb = 6 * 2 ** k
    lam = 0.5 * (1 + np.cos(np.pi * (np.arange(n_cheb) + 0.5) / n_cheb))
    s_in = np.unique(np.r_[np.geomspace(lo, 1, 400), np.sqrt(lam[lam >= lo * lo])])
    s_low = np.linspace(0, lo, 101)[:-1]
    s_hi = np.linspace(1, basin, max(32, min(512, 4 * 2 ** k)))[1:]
    return np.r_[s_in, s_low, s_hi], len(s_in), len(s_low)


def _buffers(x, s, k, norms=None, tangents=False):
    """Buffers ``B_0..B_k`` on ``s``; each ``L_j`` (j >= 1) is divided by its max (or ``norms``).

    With ``tangents``, also return ``D[j]``: the derivative of every basis
    column ``s * B_i`` when ``L_j`` is shifted by its own max (a first-order
    model of rounding ``L_j`` to FP16 with relative error 1).
    """
    lam = s * s
    L0 = lam + x[0]
    B, Ls, used, coefs, i = [lam, L0 ** 2], [L0], [1.0], [], 1
    for j in range(1, k):
        c = x[i:i + j + 1]
        i += j + 1
        L = B[j] + c[-1] + sum(ci * bi for ci, bi in zip(c[:-1], B[:j]))
        m = norms[j] if norms is not None else np.abs(L).max() + 1e-300
        used.append(m)
        coefs.append(c)
        Ls.append(L / m)
        B.append((L / m) ** 2)
    if not tangents:
        return B, used
    D = []
    for j in range(k):
        dB = [np.zeros_like(s)] * (j + 1) + [2 * Ls[j] * np.abs(Ls[j]).max()]
        for t in range(j + 1, k):
            c = coefs[t - 1]
            dL = (dB[t] + sum(ci * db for ci, db in zip(c[:-1], dB[:t]))) / used[t]
            dB.append(2 * Ls[t] * dL)
        D.append(np.stack(dB + [np.zeros_like(s)], 1) * s[:, None])
    return B, used, D


def _lp(V, D, n_in, n_low, rho):
    """Minimize ``E + rho * sum_j S_j``: ``|f - 1| <= E`` on the fit points, basin
    constraints elsewhere, and ``S_j >= max |d f / d L_j|`` (rounding sensitivity)."""
    sc = np.abs(V).max(0) + 1e-300
    Vn = V / sc
    Dn = [d / sc for d in D]
    k, m, N = len(D), Vn.shape[1], len(Vn)
    o, z = np.ones, np.zeros
    Vi, Vl, Vh = Vn[:n_in], Vn[n_in:n_in + n_low], Vn[n_in + n_low:]
    nh = len(Vh)
    # variables: c (m), E, S_0..S_{k-1}
    def row(M, e=0.0, sj=None):
        out = np.zeros((len(M), m + 1 + k))
        out[:, :m] = M
        out[:, m] = e
        if sj is not None:
            out[:, m + 1 + sj] = -1
        return out
    A = [row(Vi, -1), row(-Vi, -1), row(Vl, -1), row(-Vl), row(Vh), row(-Vh)]
    b = [o(n_in), -o(n_in), o(n_low), z(n_low), CAP * o(nh), -HI_FLOOR * o(nh)]
    stride = max(1, N // 600)  # sensitivity on a subsample keeps the LP small
    for j, d in enumerate(Dn):
        A += [row(d[::stride], sj=j), row(-d[::stride], sj=j)]
        b += [z(len(d[::stride]))] * 2
    cost = np.r_[z(m), 1, rho * o(k)]
    r = linprog(cost, A_ub=np.vstack(A), b_ub=np.concatenate(b),
                bounds=[(None, None)] * m + [(0, None)] * (1 + k), method="highs")
    if r.status != 0:
        return float("inf"), None, None
    return float(r.x[m]), r.x[:m] / sc, float(r.x[m + 1:].sum())


def _design(x, k, lo, basin, rho=RHO):
    s, n_in, n_low = _grid(lo, basin, k)
    with np.errstate(all="ignore"):
        B, norms, D = _buffers(x, s, k, tangents=True)
    V = np.stack(B + [np.ones_like(s)], 1) * s[:, None]
    if not (np.isfinite(V).all() and all(np.isfinite(d).all() for d in D)):
        return float("inf"), None, None, None
    E, v, sens = _lp(V, D, n_in, n_low, rho)
    return E, v, norms, sens


def to_segment(x, v, norms, k) -> ChainSegment:
    """Convert fit variables to the runtime format ``(w_I, w_0..w_j)`` / ``(v_I, v_0..v_k)``."""
    ops = [(float(x[0]), 1.0)]
    i = 1
    for j in range(1, k):
        c = x[i:i + j + 1]
        i += j + 1
        m = norms[j]
        ops.append(tuple(float(t) for t in [c[-1] / m, *(ci / m for ci in c[:-1]), 1.0 / m]))
    out = (float(v[-1]), *(float(t) for t in v[:-1]))
    return ChainSegment(tuple(ops), out)


def _vp_residual(x, s, w, n_in, k, rho, n_low):
    with np.errstate(all="ignore"):
        B, _, D = _buffers(x, s, k, tangents=True)
        V = np.stack(B + [np.ones_like(s)], 1) * s[:, None]
    if not (np.isfinite(V).all() and all(np.isfinite(d).all() for d in D)):
        return np.full(len(s) + 1, 1e3)
    sc = np.abs(V).max(0) + 1e-300
    Vn = V / sc
    sw = np.sqrt(w)
    c = np.linalg.lstsq(Vn * sw[:, None], sw, rcond=None)[0]
    f = Vn @ c
    r = np.empty(len(s) + 1)
    r[:n_in] = sw[:n_in] * (f[:n_in] - 1)
    floor = np.r_[np.zeros(n_low), np.full(len(s) - n_in - n_low, HI_FLOOR)]
    rest = f[n_in:]
    r[n_in:-1] = 3 * np.where(rest < floor, rest - floor, np.where(rest > CAP, rest - CAP, 0))
    sens = sum(np.abs((d / sc) @ c).max() for d in D)
    r[-1] = np.sqrt(n_in) * rho * sens
    return r


def fit_chain(k: int, lo: float, basin: float, restarts: int, seed: int = 0, keep: int = 4,
              rho: float = RHO) -> list[dict]:
    """Return the ``keep`` best distinct fits for ``k`` squares on ``[lo, 1]``.

    The score is ``E + rho * S`` (minimax error plus first-order FP16 rounding
    sensitivity); ``error`` reports ``E`` alone.
    """
    rng = np.random.default_rng(seed)
    s, n_in, n_low = _grid(lo, basin, k)
    fits = []
    for _ in range(restarts):
        x = np.r_[rng.uniform(-1, 0.2), rng.uniform(-2, 2, n_outer(k) - 1)]
        w = np.ones(len(s))
        try:
            for _ in range(6):  # Lawson reweighting toward the minimax solution
                x = least_squares(_vp_residual, x, args=(s, w, n_in, k, rho, n_low), method="trf",
                                  max_nfev=100 * len(x)).x
                r = np.abs(_vp_residual(x, s, np.ones(len(s)), n_in, k, rho, n_low)[:n_in])
                w[:n_in] *= r + 1e-12
                w[:n_in] /= w[:n_in].mean()
        except (ValueError, np.linalg.LinAlgError):
            continue
        E, v, norms, sens = _design(x, k, lo, basin, rho)
        if v is not None and np.isfinite(E):
            fits.append((E + rho * sens, E, sens, x, v, norms))
    fits.sort(key=lambda t: t[0])
    out, seen = [], []
    for score, E, sens, x, v, norms in fits:
        if any(abs(score - e) < 1e-6 for e in seen):
            continue
        seen.append(score)
        seg = to_segment(x, v, norms, k)
        out.append({"error": E, "sensitivity": sens, "score": score, "segment": seg})
        if len(out) == keep:
            break
    return out


def _run_pool(jobs, workers):
    """Run fit jobs in parallel, printing each as it finishes; results keep job order."""
    out = [None] * len(jobs)
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(_fit_job, j): i for i, j in enumerate(jobs)}
        for fut in as_completed(futures):
            f = out[futures[fut]] = fut.result()
            best = f["fits"][0]["error"] if f["fits"] else float("nan")
            print(f"fit k={f['k']} lo={f['lo']:.4g} best={best:.4g} ({f['seconds']:.0f}s)", flush=True)
    return out


def _fit_job(args):
    k, lo, basin, restarts, seed = args
    t = time.perf_counter()
    fits = fit_chain(k, lo, basin, restarts, seed)
    return {"k": k, "lo": lo, "basin": basin, "seconds": time.perf_counter() - t, "fits": fits}


# --------------------------------------------------------------------------------------
# Scalar maps of the baselines (exact arithmetic)
# --------------------------------------------------------------------------------------
def _odd_composition(coefficients):
    def f(s):
        for a, b, c in coefficients:
            s = a * s + b * s ** 3 + c * s ** 5
        return s
    return f


def _switch(table, buffers):
    def f(s):
        for residual, rows in table:
            work = [s * s] + [None] * (buffers - 1)
            for dst, src, add, a, b, c in rows:
                work[dst] = a * work[src] ** 2 + b * work[add] + c
            s = s * work[dst] + residual * s
        return s
    return f


BASELINES = {
    "polar_express_5": (_odd_composition(bl.POLAR_EXPRESS_COEFFICIENTS), "(15α + 5)n³", lambda a: 15 * a + 5),
    "PS2_55": (_switch(bl.PS2_55, 2), "(6α + 10)n³", lambda a: 6 * a + 10),
    "PS3_55": (_switch(bl.PS3_55, 3), "(6α + 10)n³", lambda a: 6 * a + 10),
    "jordan_ns5": (_odd_composition((bl._JORDAN,) * 5), "(15α + 5)n³", lambda a: 15 * a + 5),
}


def scalar_error(f, lo: float) -> float:
    s = np.geomspace(lo, 1, 4001)
    with np.errstate(all="ignore"):
        return float(np.nanmax(np.abs(f(s) - 1)))


# --------------------------------------------------------------------------------------
# Matrix test sets
# --------------------------------------------------------------------------------------
def synthetic(n: int, alpha: int, lo: float, seed: int, kind: str = "loguniform") -> torch.Tensor:
    """``(n, alpha n)`` matrix with sigma_max = 1 and the given spectrum on ``[lo, 1]``."""
    g = torch.Generator().manual_seed(seed)
    m = alpha * n
    if kind == "loguniform":
        s = torch.exp(torch.empty(n, dtype=torch.float64).uniform_(math.log(lo), 0, generator=g))
    else:  # Muon-like: fast decay with effective rank ~3
        s = torch.exp(-torch.arange(n, dtype=torch.float64) / 3.0).clamp_min(lo)
    s[0] = 1.0
    U = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, generator=g))[0]
    V = torch.linalg.qr(torch.randn(m, n, dtype=torch.float64, generator=g))[0]
    return (U * s) @ V.mT


def rel_error(Y: torch.Tensor, P: torch.Tensor) -> float:
    Y = Y.double().cpu()
    if not torch.isfinite(Y).all():
        return float("inf")
    return ((Y - P).norm() / P.norm()).item()


def circuit_on(G, circuit, dtype, device, sigma=None):
    return run_circuit(G.to(device).float(), circuit, dtype=DTYPES[dtype], sigma=sigma)


# --------------------------------------------------------------------------------------
# 4. Rounding-aware selection and polish
# --------------------------------------------------------------------------------------
def _sim_score(circuit: GramCircuit, mats, dtype: str) -> float:
    errs = [rel_error(run_circuit(G, circuit, dtype=DTYPES[dtype], sigma=torch.ones(())), P)
            for G, P in mats]
    return float(np.mean(errs))


def _test_mats(lo, seeds, n=96, alpha=2):
    out = []
    for sd in seeds:
        G = synthetic(n, alpha, lo, sd)
        out.append((G.float(), bl.svd_polar_fp64(G)))
    return out


def polish(circuit: GramCircuit, dtype: str, iters: int) -> tuple[GramCircuit, dict]:
    """Nelder-Mead on the last segment's output coefficients under emulated rounding."""
    train = _test_mats(circuit.lo, (11, 12))
    held = _test_mats(circuit.lo, (21, 22, 23))
    seg = circuit.segments[-1]
    base = np.array(seg.out)
    err0 = circuit.error

    def make(v):
        segs = circuit.segments[:-1] + (ChainSegment(seg.ops, tuple(map(float, v))),)
        return GramCircuit(circuit.name, segs, circuit.lo, circuit.basin, circuit.error)

    def obj(v):
        c = make(v)
        e = scalar_error(c.scalar, c.lo)
        return _sim_score(c, train, dtype) + 10 * max(0.0, e - 1.1 * err0)

    before = _sim_score(circuit, held, dtype)
    if iters <= 0:
        return circuit, {"held_out_before": before, "held_out_after": before, "accepted": False}
    r = minimize(obj, base, method="Nelder-Mead",
                 options={"maxiter": iters, "xatol": 1e-9, "fatol": 1e-9, "initial_simplex":
                          np.vstack([base] + [base + np.eye(len(base))[i] * 1e-3 * max(1, abs(base[i]))
                                              for i in range(len(base))])})
    cand = make(r.x)
    after = _sim_score(cand, held, dtype)
    ok = after < before and cand.basin_u() >= min(circuit.basin_u(), circuit.basin) * 0.999
    if ok:
        cand = GramCircuit(cand.name, cand.segments, cand.lo, cand.basin,
                           scalar_error(cand.scalar, cand.lo))
    return (cand if ok else circuit), {"held_out_before": before, "held_out_after": after, "accepted": ok}


# --------------------------------------------------------------------------------------
# 5. Evaluation
# --------------------------------------------------------------------------------------
def describe(c: GramCircuit, alphas=(1, 2, 4, 8)) -> dict:
    rep = basin_report(c.scalar, lo=c.lo)
    return {
        **c.to_dict(), "error": scalar_error(c.scalar, c.lo),
        "flops_by_alpha": {str(a): c.flops(a) for a in alphas},
        "basin_fp64": c.basin_u(), "at_overshoot": rep["at_overshoot"],
        "basin_fp16_scalar": c.basin_u(rnd=ROUND["fp16"]),
        "basin_bf16_scalar": c.basin_u(rnd=ROUND["bf16"]),
        "error_fp16_scalar": scalar_error(lambda s: c.scalar(s, ROUND["fp16"]), c.lo),
        "error_bf16_scalar": scalar_error(lambda s: c.scalar(s, ROUND["bf16"]), c.lo),
    }


def baseline_table(los, alphas=(1, 2, 4, 8)) -> dict:
    out = {}
    for name, (f, formula, cost) in BASELINES.items():
        rep = basin_report(f, lo=min(los))
        out[name] = {
            "flops": formula, "flops_by_alpha": {str(a): cost(a) for a in alphas},
            "error_by_lo": {str(lo): scalar_error(f, lo) for lo in los},
            "basin_fp64": rep["basin_u"], "at_overshoot": rep["at_overshoot"],
            "max_on_unit": rep["max_on_unit"],
        }
    return out


def _baseline_matrix(name, G, device):
    G = G.to(device).float()
    if name == "polar_express_bf16":
        return bl.polar_express(G)
    if name == "gram_ns_fp16":
        return bl.gram_ns(G)
    if name in ("PS2_55", "PS3_55"):
        return bl.polar_switch(G, name)
    if name == "jordan_ns5":
        return bl.jordan_ns5(G)
    raise KeyError(name)


BASELINE_RUNS = ("polar_express_bf16", "gram_ns_fp16", "PS2_55", "PS3_55", "jordan_ns5")


def synthetic_eval(circuits: dict, device: str, n: int, alphas, los, seeds) -> dict:
    """Relative Frobenius polar error on synthetic matrices (exact sigma_max = 1 is known)."""
    rows = []
    for alpha in alphas:
        for lo in los:
            for kind in ("loguniform", "muon"):
                for sd in seeds:
                    G = synthetic(n, alpha, lo, 1000 + sd, kind)
                    P = bl.svd_polar_fp64(G)
                    Gf = G.float()
                    case = {"alpha": alpha, "lo": lo, "kind": kind, "seed": sd}
                    for name, c in circuits.items():
                        for dt in ("fp16", "bf16"):
                            case[f"{name}|{dt}|oracle"] = rel_error(
                                circuit_on(Gf, c, dt, device, sigma=torch.ones(())), P)
                            case[f"{name}|{dt}|cpu_emulated"] = rel_error(
                                circuit_on(Gf, c, dt, "cpu", sigma=torch.ones(())), P)
                            case[f"{name}|{dt}|frobenius"] = rel_error(circuit_on(Gf, c, dt, device), P)
                    for b in BASELINE_RUNS:
                        case[b] = rel_error(_baseline_matrix(b, Gf, device), P)
                    rows.append(case)
    return {"cases": rows, "summary": _summarize(rows)}


def _summarize(rows) -> dict:
    keys = [k for k in rows[0] if k not in ("alpha", "lo", "kind", "seed", "parameter", "phase")]
    out = {}
    for key in keys:
        v = np.array([r[key] for r in rows if isinstance(r.get(key), (int, float))], dtype=float)
        if not len(v):
            continue
        fin = v[np.isfinite(v)]
        out[key] = {"median": float(np.median(fin)) if len(fin) else None,
                    "p90": float(np.quantile(fin, .9)) if len(fin) else None,
                    "max": float(fin.max()) if len(fin) else None,
                    "nonfinite": int((~np.isfinite(v)).sum()), "count": int(len(v))}
    return out


def trajectory_eval(circuits: dict, path: Path, device: str, estimator: EstimatorConfig) -> dict:
    """Stream each harvested pair: cold estimate on ``previous``, warm estimate on ``current``."""
    data = torch.load(path, map_location="cpu")
    rows = []
    for pair in data["pairs"]:
        prev, cur = pair["previous"].float(), pair["current"].float()
        P = bl.svd_polar_fp64(cur)
        W = cur.double().mT if cur.shape[0] > cur.shape[1] else cur.double()
        s = torch.linalg.svdvals(W)
        row = {"parameter": pair["parameter"], "phase": pair["phase"],
               "kappa": (s[0] / s[-1]).item(), "fro_over_spectral": (s.norm() / s[0]).item()}
        for name, c in circuits.items():
            for dt in ("fp16", "bf16"):
                scaler = WarmSpectralScaler(estimator)
                run_circuit(prev.to(device), c, dtype=DTYPES[dt], scaler=scaler, key="p")
                Y, info = run_circuit(cur.to(device), c, dtype=DTYPES[dt], scaler=scaler, key="p",
                                      return_info=True)
                row[f"{name}|{dt}|warm"] = rel_error(Y, P)
                if dt == "fp16":
                    true = (s[0] / W.norm()).item()  # sigma_max of the Frobenius-normalized matrix
                    row["sigma_hat_over_sigma_max"] = info.sigma.item() / true
                    row["upper_over_sigma_max"] = info.upper.item() / true
                    row["escalated"] = bool(info.escalated.item())
                    row["certified"] = bool(info.certified.item())
                row[f"{name}|{dt}|frobenius"] = rel_error(
                    run_circuit(cur.to(device), c, dtype=DTYPES[dt]), P)
                row[f"{name}|{dt}|oracle"] = rel_error(
                    run_circuit(cur.to(device), c, dtype=DTYPES[dt], sigma=s[0:1].float()), P)
        for b in BASELINE_RUNS:
            row[b] = rel_error(_baseline_matrix(b, cur, device), P)
        rows.append(row)
    summary = _summarize(rows)
    for key in ("escalated", "certified"):
        summary[key] = int(sum(r.get(key, False) for r in rows))
    return {"pairs": len(rows), "cases": rows, "summary": summary}


def timing(circuits: dict, shapes, repeats: int = 20) -> dict:
    """Median CUDA latency (ms) with CUDA events; the GPU may be shared, so treat as indicative."""
    if not torch.cuda.is_available():
        return {}
    out = {}
    for shape in shapes:
        G = torch.randn(*shape, device="cuda")
        runs = {b: (lambda b=b: _baseline_matrix(b, G, "cuda")) for b in ("gram_ns_fp16", "PS2_55", "polar_express_bf16")}
        for name, c in circuits.items():
            for tri in (False, True):
                sc = WarmSpectralScaler()
                runs[f"{name}|fp16|{'triton' if tri else 'cublas'}"] = (
                    lambda c=c, tri=tri, sc=sc: run_circuit(G, c, dtype=torch.float16, use_triton_syrk=tri,
                                                            scaler=sc, key=0))
        res = {}
        for label, fn in runs.items():
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            times = []
            for _ in range(repeats):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                fn()
                b.record()
                torch.cuda.synchronize()
                times.append(a.elapsed_time(b))
            res[label] = float(np.median(times))
        out["x".join(map(str, shape))] = res
    return out


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
# lo = sigma_min / sigma_hat covered. The task range is [0.005, 0.05]; 0.001-0.002 is
# added because the median harvested momentum has 16% of its singular values below
# 0.01 sigma_max and the p90 matrix has 25% below 0.005 sigma_max.
ZERO_RESTART_GRID = {0.002: (8,), 0.005: (7, 8), 0.01: (6, 7, 8), 0.02: (5, 6, 7), 0.05: (4, 5, 6)}
SEG1_GRID = {0.001: (3, 4), 0.002: (3, 4), 0.005: (2, 3), 0.01: (2, 3)}
SEG2_K = (3, 4, 5)


def _seg_dict(seg: ChainSegment) -> dict:
    return {"ops": [list(o) for o in seg.ops], "out": list(seg.out)}


def _seg_load(d: dict) -> ChainSegment:
    return ChainSegment(tuple(tuple(o) for o in d["ops"]), tuple(d["out"]))


def _dump_fits(fits) -> list:
    return [{**{k: v for k, v in f.items() if k != "fits"},
             "fits": [{**{k: v for k, v in t.items() if k != "segment"}, "segment": _seg_dict(t["segment"])}
                      for t in f["fits"]]} for f in fits]


def _load_fits(rows) -> list:
    return [{**f, "fits": [{**t, "segment": _seg_load(t["segment"])} for t in f["fits"]]} for f in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", type=Path, default=Path("results/circuit_search_cycle1.json"))
    ap.add_argument("--trajectories", type=Path, default=Path("artifacts/harvested_trajectories.pt"))
    ap.add_argument("--basin", type=float, default=1.15, help="overshoot u the circuit must tolerate")
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--polish-iters", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quick", action="store_true", help="tiny grid for smoke tests")
    ap.add_argument("--fit-cache", type=Path, default=None,
                    help="JSON with chain fits: loaded if it exists, else written after fitting")
    args = ap.parse_args()
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    zgrid = {0.05: (4,)} if args.quick else ZERO_RESTART_GRID
    s1grid = {0.01: (2,)} if args.quick else SEG1_GRID
    restarts = 4 if args.quick else args.restarts

    los = sorted({*zgrid, *s1grid})
    floors = {str(lo): {str(k): degree_floor(lo, k) for k in range(2, 9)} for lo in los}
    print("degree floors", json.dumps(floors), flush=True)

    cache = None
    if args.fit_cache is not None and args.fit_cache.exists():
        cache = json.loads(args.fit_cache.read_text())
        print(f"loaded fits from {args.fit_cache}", flush=True)
    jobs = [(k, lo, args.basin, restarts, 17 * k + int(1e4 * lo)) for lo, ks in zgrid.items() for k in ks]
    jobs += [(k, lo, args.basin, restarts, 31 * k + int(1e4 * lo)) for lo, ks in s1grid.items() for k in ks]
    if cache is not None:
        fitted = _load_fits(cache["fitted"])
    else:
        fitted = _run_pool(jobs, args.workers)

    # One-restart composites: normalize segment 1 so its max on [0, basin] is 1,
    # then fit segment 2 on the lifted interval [lo2, 1] (its input never exceeds 1).
    seg1 = [f for f in fitted if f["lo"] in s1grid and f["k"] in s1grid[f["lo"]] and f["fits"]]
    seg2_jobs, seg1_info = [], []
    for f in seg1:
        seg = f["fits"][0]["segment"]
        x = np.linspace(0, args.basin, 20001)
        y = seg.scalar(x)
        top = float(y.max())
        lo2 = float(seg.scalar(np.geomspace(f["lo"], 1, 4001)).min()) / top
        seg1_info.append((f, seg.scaled(1 / top), lo2))
        for k2 in ((2,) if args.quick else SEG2_K):
            seg2_jobs.append((k2, lo2, 1.02, restarts, 7 * k2 + len(seg2_jobs)))
    if cache is not None:
        seg2_fits = _load_fits(cache["seg2"])
    else:
        seg2_fits = _run_pool(seg2_jobs, args.workers)
        if args.fit_cache is not None:
            args.fit_cache.parent.mkdir(parents=True, exist_ok=True)
            args.fit_cache.write_text(json.dumps({"fitted": _dump_fits(fitted), "seg2": _dump_fits(seg2_fits)}))

    candidates = []
    for f in fitted:
        if f["lo"] in zgrid and f["k"] in zgrid[f["lo"]]:
            for rank, fit in enumerate(f["fits"]):
                candidates.append(GramCircuit(f"S0-k{f['k']}-lo{f['lo']}-r{rank}", (fit["segment"],),
                                              f["lo"], args.basin, fit["error"]))
    for f, s1, lo2 in seg1_info:
        for g in seg2_fits:
            if abs(g["lo"] - lo2) > 1e-12 or not g["fits"]:
                continue
            c = GramCircuit(f"S1-k{f['k']}+{g['k']}-lo{f['lo']}", (s1, g["fits"][0]["segment"]),
                            f["lo"], args.basin, 0.0)
            candidates.append(GramCircuit(c.name, c.segments, c.lo, c.basin, scalar_error(c.scalar, c.lo)))

    # Rounding-aware selection: per (family, k, lo) keep the restart with the best FP16 emulated score.
    sel_mats = {lo: _test_mats(lo, (1, 2)) for lo in los}
    best = {}
    for c in candidates:
        if c.basin_u() < 1 + 0.8 * (args.basin - 1):  # dense check; the LP enforces it on a grid
            continue
        fam = c.name.rsplit("-r", 1)[0]
        score = _sim_score(c, sel_mats[c.lo], "fp16")
        if fam not in best or score < best[fam][0]:
            best[fam] = (score, c)
    chosen, polish_log = {}, {}
    for fam, (score, c) in sorted(best.items()):
        c = GramCircuit(fam, c.segments, c.lo, c.basin, c.error)
        c2, log = polish(c, "fp16", 0 if args.quick else args.polish_iters)
        chosen[fam] = c2
        polish_log[fam] = {"selection_fp16_score": score, **log}
        print(f"{fam}: err={c2.error:.4g} flops={c2.flops_formula()} basin={c2.basin_u():.3f} "
              f"polish={log}", flush=True)

    # Pareto front per lo: (squares, restarts) vs error; evaluate a compact subset on matrices.
    circuits = {k: v for k, v in chosen.items()}
    focus = _focus(circuits)
    print("focus:", list(focus), flush=True)

    gammas = {}
    if args.trajectories.exists():
        pairs = [(p["previous"], p["current"]) for p in torch.load(args.trajectories)["pairs"]]
        gammas = {str(st): calibrate_gamma(pairs, st, device=args.device) for st in (1, 2)}
    synth = synthetic_eval(focus, args.device, 128 if args.quick else 256,
                           (1, 4) if args.quick else (1, 2, 4, 8),
                           (0.05,) if args.quick else (0.002, 0.01, 0.05), (0,) if args.quick else (0, 1))
    traj = (trajectory_eval(focus, args.trajectories, args.device, EstimatorConfig())
            if args.trajectories.exists() else None)
    tim = timing(focus, [(1024, 1024)] if args.quick else [(1024, 1024), (1024, 4096), (2048, 2048), (2048, 8192)])

    result = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__, "seconds": None,
        "settings": {"basin": args.basin, "cap": CAP, "restarts": restarts, "quick": args.quick,
                     "rho": RHO,
                     "zero_restart_grid": {str(k): v for k, v in zgrid.items()},
                     "seg1_grid": {str(k): v for k, v in s1grid.items()}, "seg2_k": SEG2_K,
                     "precision_model": "FP16/BF16 operands, FP32 accumulation, FP32 Gram buffers, "
                                        "one rounding per operand; CPU path is an exact emulation",
                     "flops_model": "leading symmetric FLOPs: SYRK αn³, square n³, Q@W 2αn³"},
        "degree_floor": floors,
        "fits": [{"k": f["k"], "lo": f["lo"], "basin": f["basin"], "seconds": f["seconds"],
                  "errors": [t["error"] for t in f["fits"]]} for f in fitted + seg2_fits],
        "circuits": {k: describe(v) for k, v in circuits.items()},
        "polish": polish_log,
        "focus": list(focus),
        "baselines_scalar": baseline_table(los),
        "gamma_calibration": gammas,
        "synthetic": synth,
        "trajectories": traj,
        "timing_ms": tim,
    }
    result["seconds"] = time.perf_counter() - t0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, allow_nan=True) + "\n")
    print(f"wrote {args.output} in {result['seconds']:.0f}s", flush=True)


def _focus(circuits: dict) -> dict:
    """Per lo, the cheapest 0-restart circuit with error <= 0.16 and the best 1-restart circuit."""
    out = {}
    for lo in sorted({c.lo for c in circuits.values()}):
        zero = sorted((c for c in circuits.values() if c.lo == lo and c.restarts == 0),
                      key=lambda c: (c.flops(), c.error))
        ok = [c for c in zero if c.error <= 0.16]
        for c in (ok[:2] if ok else zero[-1:]):
            out[c.name] = c
        one = sorted((c for c in circuits.values() if c.lo == lo and c.restarts == 1),
                     key=lambda c: (c.error > 0.16, c.flops(), c.error))
        if one:
            out[one[0].name] = one[0]
    return out


if __name__ == "__main__":
    main()
