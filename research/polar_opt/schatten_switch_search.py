"""Search register-minimal Polar Switch circuits for Schatten-8 pre-scaled inputs (Cycle 2, T1-04).

A circuit is a list of segments. Each segment runs natively on ``_symprod_kernel``:

    B[0] = X X^T                                     (SYRK)
    B[dst] = a * B[src]^2 + b * B[add] + c * I       (k squarings, 2 or 3 FP16 buffers)
    X <- B[last dst] @ X                             (restart)

On a singular value ``x = sigma / sigma_hat`` one segment is ``x -> x * p_k(x^2)``. The input is
scaled by the rigorous Schatten-8 bound, so the design interval is ``x in [lo, 1]`` and not the
Frobenius interval of ``PS3_55``. The search asks how many squarings that saves.

Pipeline
--------
1. Segment fits. Every register program (``dst``, ``src``, ``add``) is enumerated, and its
   coefficients are fitted by a penalty Levenberg-Marquardt solve followed by an exact LP on the
   last operation. A ladder in ``lo`` carries each solution from an easy interval to a hard one.
   Two problems are solved on the normalized interval ``[0, 1]``:
     lift       ``1 - E <= f <= 1`` on ``[lo, 1]`` and ``0 <= f <= 1`` on ``[0, lo]``
     symmetric  ``|f - 1| <= E`` on ``[lo, 1]`` and ``0 <= f <= 1 + E`` on ``[0, lo]``
   The first operation of segment 1 is ``pure`` (``R^2``), ``semi`` (``R^2 + b R``), or
   ``general`` (``R^2 + b R + c I``). ``R^2`` is the matrix the Schatten-8 bound needs, so the
   bound costs no extra squaring. ``semi`` and ``general`` add one and two elementwise passes.
2. Stronger optima. Random starts stop in poor local optima (about half the lift of
   ``PS3_55``'s own segments), so every family gets an iterated local search at three rungs
   and sweeps along the ladder. The two ``PS3_55`` segments seed the 5-squaring families.
3. Composites. A lift segment and a symmetric segment are joined with rounding headroom: the
   scale is ``sigma_hat = u * U`` and segment 2 sees at most ``1 / u`` (``u = 1.05``). Both
   segments are bounded on ``[0, 1]``, which is ``[0, u]`` in units of their nominal input.
   Zero-restart chains (``k = 6, 7``) are fitted with the same solver.
4. Scalar scoring on the 45 harvested momenta (exact arithmetic), then a joint polish of both
   segments on that score with the safety bounds as hard penalties.
5. Native verification on the GPU with ``schatten_switch_blackwell``: coherent rank-1 sign
   matrices swept over ``[0, u]`` in FP16 and BF16, the ``p I + c 11^T`` family at n = 4096 and
   8192, the 117 Tier-1 cases, and a short latency check against ``ps3_55_blackwell_fp16``.
   Per number of squarings, the best-scoring candidate that passes the FP16 checks is selected.

Early-spectrum re-polish (opt-in, ``--include-early-srank``, Cycle 3 C3-02): the registered 5+4,
5+3, 3+5 and 3+4 circuits (and the best fit-cache composites of those splits) are polished again
with purely synthetic near-rank-1 spectra (stable rank 1.03-1.60) added to the targets, then
checked on the GPU (Fable R1 at n = 1023..8192, R2 sweeps, R3 at K = 2^20, CUDA-graph replay).
It replaces the full search and takes a few minutes. Real held-out momenta are never read here.

Run (about 50 minutes on 8 cores and one GPU for a full search):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
        research/polar_opt/schatten_switch_search.py --output results/schatten_switch_search_cycle2.json
"""

from __future__ import annotations

import os

# One BLAS thread per fit worker; the pool provides the parallelism.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, linprog

from research.polar_opt import baselines as bl

MAX_WORKERS = 8  # host-load limit for the CPU search
PS3_MEDIAN, PS3_MAX = 0.064, 0.3208  # PS3_55 on the harvested momenta (parity targets)
FIRST_PARAMS = {"pure": 0, "semi": 1, "general": 2}
LIFT_LADDER = (0.3, 0.2, 0.14, 0.1, 0.07, 0.05, 0.035, 0.025, 0.017, 0.012, 0.008, 0.0055, 0.004,
               0.003, 0.0022, 0.0016, 0.0012)
SYM_LADDER = (0.5, 0.4, 0.3, 0.22, 0.16, 0.12, 0.09, 0.07, 0.055, 0.045, 0.035, 0.028, 0.022,
              0.017, 0.013, 0.01)
ZERO_LADDER = (0.5, 0.35, 0.25, 0.18, 0.13, 0.09, 0.065, 0.045, 0.032, 0.022, 0.015, 0.01, 0.007,
               0.005, 0.0035)


# --------------------------------------------------------------------------------------
# 1. Register programs and the scalar chain
# --------------------------------------------------------------------------------------
def topologies(k: int, buffers: int) -> list[tuple[tuple[int, int, int], ...]]:
    """All register programs ``(dst, src, add)`` with ``src`` = the previous ``dst``.

    ``dst != src`` because the kernel output cannot alias its product operand. Programs that
    read the same history values and keep the same history values live are the same program.
    """
    found = {}

    def extend(j, history, src, program, key):
        if j == k:
            found.setdefault(tuple(key), tuple(program))
            return
        for dst in range(buffers):
            if dst == src:
                continue
            for add in ([0] if j == 0 else [r for r in range(buffers) if history[r] is not None]):
                after = list(history)
                after[dst] = j + 1
                live = tuple(sorted(h for h in after if h is not None))
                extend(j + 1, after, dst, program + [(dst, src, add)], key + [(history[add], live)])

    extend(0, [0] + [None] * (buffers - 1), 0, [], [])
    return list(found.values())


def chain(theta: np.ndarray, topo, lam: np.ndarray, first: str, jac: bool = True):
    """Value (and Jacobian) of the last buffer on ``lam``.

    Operation 0 is ``lam^2 + b0 lam + c0`` with the gauge ``a0 = 1``; ``b0`` and ``c0`` exist
    only for ``first`` = "semi" / "general". Later operations have free ``(a, b, c)``.
    """
    n0 = FIRST_PARAMS[first]
    rest = theta[n0:].reshape(-1, 3)
    b0 = theta[0] if n0 >= 1 else 0.0
    c0 = theta[1] if n0 == 2 else 0.0
    value = {0: lam}
    grad = {0: np.zeros((lam.size, theta.size))} if jac else None
    for j, (dst, src, add) in enumerate(topo):
        if j == 0:
            new = lam * lam + b0 * lam + c0
            if jac:
                d = np.zeros((lam.size, theta.size))
                if n0 >= 1:
                    d[:, 0] = lam
                if n0 == 2:
                    d[:, 1] = 1.0
        else:
            a, b, c = rest[j - 1]
            ps, pa = value[src], value[add]
            new = a * ps * ps + b * pa + c
            if jac:
                d = (2 * a * ps)[:, None] * grad[src] + b * grad[add]
                i = n0 + 3 * (j - 1)
                d[:, i] += ps * ps
                d[:, i + 1] += pa
                d[:, i + 2] += 1.0
        value[dst] = new
        if jac:
            grad[dst] = d
    return (value[dst], grad[dst]) if jac else value[dst]


def rows_from_theta(theta, topo, first: str, scale: float = 1.0) -> tuple:
    """Table rows ``(dst, src, add, a, b, c)``; ``scale`` multiplies the output."""
    n0 = FIRST_PARAMS[first]
    theta = np.asarray(theta, dtype=float)
    rest = theta[n0:].reshape(-1, 3)
    coefficients = [(1.0, float(theta[0]) if n0 >= 1 else 0.0, float(theta[1]) if n0 == 2 else 0.0)]
    coefficients += [tuple(float(v) for v in row) for row in rest]
    a, b, c = coefficients[-1]
    coefficients[-1] = (a * scale, b * scale, c * scale)
    return tuple((dst, src, add, *coefficients[j]) for j, (dst, src, add) in enumerate(topo))


def segment_scalar(rows, x):
    """``x -> x * p_k(x^2)`` for table rows, in float64."""
    x = np.asarray(x, dtype=np.float64)
    work = {0: x * x}
    for dst, src, add, a, b, c in rows:
        work[dst] = a * work[src] ** 2 + b * work[add] + c
    return x * work[dst]


def table_scalar(table, x):
    """Scalar map of a whole table ``((residual, rows), ...)``."""
    for residual, rows in table:
        x = segment_scalar(rows, x) + residual * np.asarray(x, dtype=np.float64)
    return x


def normalize_rows(rows) -> tuple:
    """Rescale every intermediate buffer to ``max |B| = 1`` on ``[0, 1]`` (a pure gauge change).

    The first row keeps ``a = 1`` because its buffer is the Schatten-8 square, and the last row
    keeps the output scale.
    """
    lam = np.linspace(0.0, 1.0, 4001)
    value, factor, out = {0: lam}, {0: 1.0}, []  # stored buffer = factor * original buffer
    for j, (dst, src, add, a, b, c) in enumerate(rows):
        new = a * value[src] ** 2 + b * value[add] + c
        t = 1.0 if j in (0, len(rows) - 1) else 1.0 / max(float(np.abs(new).max()), 1e-300)
        out.append((dst, src, add, a * t / factor[src] ** 2, b * t / factor[add], c * t))
        value[dst], factor[dst] = new, t
    return tuple(out)


# --------------------------------------------------------------------------------------
# 2. Segment fit: penalty Levenberg-Marquardt, exact LP on the last operation, ladder in lo
# --------------------------------------------------------------------------------------
def _grid(lo: float, n_in: int = 240, n_low: int = 50):
    x_in = np.unique(np.r_[np.geomspace(lo, 1, n_in), np.linspace(lo, 1, n_in // 2)])
    return x_in, np.linspace(0, lo, n_low + 1)[1:-1]


def _residual(z, topo, first, grid, penalty, gam):
    """Residual and Jacobian of ``E^2 + penalty^2 * sum(violation^2)``; ``z = (theta, E)``."""
    theta, E = z[:-1], z[-1]
    x_in, x_low = grid
    x = np.r_[x_in, x_low]
    with np.errstate(all="ignore"):
        p, d = chain(theta, topo, x * x, first)
        f, J = x * p, x[:, None] * d
    if not (np.isfinite(f).all() and np.isfinite(J).all()):
        return None
    ni = len(x_in)
    res, jac = [], []

    def hinge(value, dvalue, dE):
        active = value > 0
        res.append(np.where(active, penalty * value, 0.0))
        jac.append(np.c_[np.where(active[:, None], penalty * dvalue, 0.0),
                         np.where(active, penalty * dE, 0.0)])

    hinge(f[:ni] - 1 - gam * E, J[:ni], -gam)
    hinge(1 - f[:ni] - E, -J[:ni], -1.0)
    hinge(f[ni:] - 1 - gam * E, J[ni:], -gam)
    hinge(-f[ni:], -J[ni:], 0.0)
    res.append(np.array([E]))
    jac.append(np.r_[np.zeros(theta.size), 1.0][None, :])
    return np.concatenate(res), np.vstack(jac)


def _lp_last(theta, topo, first, lo, gam):
    """Exact minimax over the last operation's ``(a, b, c)`` on a dense grid."""
    x_in = np.unique(np.r_[np.geomspace(lo, 1, 1200), np.linspace(lo, 1, 500)])
    x_low = np.linspace(0, lo, 201)[1:-1]
    x = np.r_[x_in, x_low]
    trial = np.array(theta, dtype=float)
    basis = []
    for unit in np.eye(3):
        trial[-3:] = unit
        with np.errstate(all="ignore"):
            basis.append(x * chain(trial, topo, x * x, first, jac=False))
    V = np.stack(basis, 1)
    if not np.isfinite(V).all():
        return np.inf, theta
    scale = np.abs(V).max(0) + 1e-300
    V = V / scale
    ni, nl = len(x_in), len(x_low)
    ones = np.ones
    A = np.vstack([np.c_[V[:ni], -gam * ones(ni)], np.c_[-V[:ni], -ones(ni)],
                   np.c_[V[ni:], -gam * ones(nl)], np.c_[-V[ni:], np.zeros(nl)]])
    b = np.r_[ones(ni), -ones(ni), ones(nl), np.zeros(nl)]
    r = linprog([0, 0, 0, 1], A_ub=A, b_ub=b, bounds=[(None, None)] * 3 + [(0, None)], method="highs")
    if r.status != 0:
        return np.inf, theta
    out = np.array(theta, dtype=float)
    out[-3:] = r.x[:3] / scale
    return float(r.x[3]), out


def solve_segment(theta, E0, topo, first, lo, gam, stages=((30, 60), (300, 80), (3000, 80))):
    """Refine ``theta`` for one ``lo``; returns ``(E, theta)`` with exact bounds from the LP."""
    grid = _grid(lo)
    z = np.r_[theta, E0]
    for penalty, evaluations in stages:
        if _residual(z, topo, first, grid, penalty, gam) is None:
            return np.inf, theta

        def fun(v, penalty=penalty):
            r = _residual(v, topo, first, grid, penalty, gam)
            return np.full(8, 1e6) if r is None else r[0]

        def jac(v, penalty=penalty):
            r = _residual(v, topo, first, grid, penalty, gam)
            return np.zeros((8, v.size)) if r is None else r[1]

        z = least_squares(fun, z, jac=jac, method="trf", max_nfev=evaluations, x_scale="jac").x
    return _lp_last(z[:-1], topo, first, lo, gam)


def _random_theta(rng, k, first):
    head = {"pure": [], "semi": [rng.uniform(-2.5, 1.0)],
            "general": [rng.uniform(-2.5, 1.0), rng.uniform(-0.3, 0.3)]}[first]
    n = k - 1
    rest = np.c_[rng.choice([-1, 1], n) * np.exp(rng.uniform(-2, 3, n)),
                 rng.uniform(-4, 4, n) * np.exp(rng.uniform(-2, 1, n)), rng.uniform(-2, 2, n)]
    return np.r_[head, rest.ravel()]


def _ladder_job(job):
    """Fit one register program down a ladder of ``lo``; returns ``{lo: [(E, theta), ...]}``."""
    topo, first, gam, ladder, seed, starts, keep = job
    rng = np.random.default_rng(seed)
    population = []
    for _ in range(starts):
        try:
            E, theta = solve_segment(_random_theta(rng, len(topo), first), 1.0, topo, first, ladder[0],
                                     gam, stages=((3, 50), (30, 60), (300, 60)))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        if np.isfinite(E) and E < 0.999:
            population.append((E, theta))
    population.sort(key=lambda t: t[0])
    current = []
    for E, theta in population:
        if all(abs(E - other) > 1e-4 for other, _ in current):
            current.append((E, theta))
        if len(current) == keep:
            break
    out = {ladder[0]: [(E, theta.tolist()) for E, theta in current]}
    for lo in ladder[1:]:
        following = []
        for E, theta in current:
            try:
                E2, theta2 = solve_segment(np.asarray(theta), min(1.0, 1.2 * E + 0.02), topo, first, lo, gam)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
            if np.isfinite(E2):
                following.append((E2, theta2))
        following.sort(key=lambda t: t[0])
        current = following
        out[lo] = [(E, theta.tolist()) for E, theta in current]
    return topo, out


def fit_family(k: int, buffers: int, first: str, mode: str, ladder, *, starts: int, keep: int,
               workers: int, max_topos: int, seed: int = 0, top: int = 6, sample: int = 700) -> dict:
    """Best fits per ladder rung for one family. ``mode`` is "lift" or "sym".

    Families with more than ``max_topos`` register programs are screened on the first three
    rungs with 3 starts, and only the best ``max_topos`` programs run the full ladder. At most
    ``sample`` programs (a seeded random subset) enter the screen.
    """
    gam = 0.0 if mode == "lift" else 1.0
    programs = topologies(k, buffers)
    total = len(programs)
    if total > sample:  # deep chains: screen a seeded random sample of the register programs
        order = np.random.default_rng(seed).permutation(total)[:sample]
        programs = [programs[i] for i in sorted(order)]
    began = time.perf_counter()
    with ProcessPoolExecutor(min(workers, MAX_WORKERS)) as pool:
        if len(programs) > max_topos:
            jobs = [(t, first, gam, ladder[:3], seed + 7919 * i, 3, 1) for i, t in enumerate(programs)]
            screen = list(pool.map(_ladder_job, jobs, chunksize=4))
            ranked = sorted(screen, key=lambda r: r[1][ladder[2]][0][0] if r[1].get(ladder[2]) else np.inf)
            selected = [t for t, _ in ranked[:max_topos]]
        else:
            selected = programs
        jobs = [(t, first, gam, ladder, seed + 104729 * i + 1, starts, keep) for i, t in enumerate(selected)]
        results = list(pool.map(_ladder_job, jobs, chunksize=1))
    rungs = {}
    for lo in ladder:
        rows = sorted(((r[lo][0][0], r[lo][0][1], [list(op) for op in t]) for t, r in results if r.get(lo)),
                      key=lambda row: row[0])
        rungs[str(lo)] = [{"E": E, "theta": theta, "topo": topo} for E, theta, topo in rows[:top]]
    return {"k": k, "buffers": buffers, "first": first, "mode": mode, "programs": total,
            "programs_screened": len(programs),
            "programs_fitted": len(selected), "starts": starts, "seconds": time.perf_counter() - began,
            "rungs": rungs}


FAMILIES = (
    # (k, buffers, first, mode, ladder, max_topos)
    (3, 3, "pure", "lift", LIFT_LADDER, 200), (3, 3, "semi", "lift", LIFT_LADDER, 200),
    (3, 3, "general", "lift", LIFT_LADDER, 200),
    (4, 3, "pure", "lift", LIFT_LADDER, 200), (4, 3, "semi", "lift", LIFT_LADDER, 200),
    (4, 3, "general", "lift", LIFT_LADDER, 200),
    (4, 2, "semi", "lift", LIFT_LADDER, 200),
    (5, 3, "pure", "lift", LIFT_LADDER, 48), (5, 3, "semi", "lift", LIFT_LADDER, 48),
    (5, 3, "general", "lift", LIFT_LADDER, 48),
    (3, 3, "general", "sym", SYM_LADDER, 200), (4, 3, "general", "sym", SYM_LADDER, 200),
    (4, 2, "general", "sym", SYM_LADDER, 200), (5, 3, "general", "sym", SYM_LADDER, 48),
    (6, 3, "semi", "sym", ZERO_LADDER, 16), (7, 3, "semi", "sym", ZERO_LADDER, 12),
)
# Rungs where the iterated local search runs; a sweep carries its results to the other rungs.
ILS_RUNGS = {"lift": (0.008, 0.004, 0.0022), "sym": (0.12, 0.07, 0.045), "zero": (0.022, 0.01, 0.005)}


def family_key(k, buffers, first, mode) -> str:
    return f"{mode}-k{k}-b{buffers}-{first}"


def _ladder_of(family: dict) -> tuple:
    return tuple(float(lo) for lo in family["rungs"])


def run_fits(cache: dict, save, workers: int, starts: int, zero_restart: bool) -> None:
    """Random multi-start ladder fits for every family that is not in ``cache``."""
    for k, buffers, first, mode, ladder, max_topos in FAMILIES:
        key = family_key(k, buffers, first, mode)
        if key in cache or (k >= 6) != zero_restart:
            continue
        cache[key] = fit_family(k, buffers, first, mode, ladder, starts=starts, keep=2, workers=workers,
                                max_topos=max_topos)
        save()
        best = {lo: round(rows[0]["E"], 4) for lo, rows in cache[key]["rungs"].items() if rows}
        print(f"fit {key}: {cache[key]['programs_fitted']}/{cache[key]['programs']} programs, "
              f"{cache[key]['seconds']:.0f}s, E by lo {best}", flush=True)


# --------------------------------------------------------------------------------------
# 3. Stronger optimum: PS3_55 anchors, iterated local search, sweeps along the ladder
# --------------------------------------------------------------------------------------
def theta_from_rows(rows, first: str = "general"):
    """Table rows -> ``(theta, topo)`` in the gauge ``a0 = 1`` (inverse of ``rows_from_theta``)."""
    gauge = {rows[0][0]: rows[0][3]}
    theta = [rows[0][4] / rows[0][3], rows[0][5] / rows[0][3]][:FIRST_PARAMS[first]]
    for dst, src, add, a, b, c in rows[1:]:
        theta += [a * gauge.get(src, 1.0) ** 2, b * gauge.get(add, 1.0), c]
        gauge[dst] = 1.0
    return np.array(theta), tuple((r[0], r[1], r[2]) for r in rows)


def _ils_job(job):
    """Iterated local search at one rung: perturb the best point, refit, keep improvements."""
    theta, topo, first, gam, lo, iterations, seed, sigma = job
    rng = np.random.default_rng(seed)
    try:
        best = solve_segment(np.asarray(theta, dtype=float), 1.0, topo, first, lo, gam)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return np.inf, list(theta), topo, lo
    for _ in range(iterations):
        size = best[1].size
        trial = best[1] * np.exp(sigma * rng.standard_normal(size)) + 0.02 * sigma * rng.standard_normal(size)
        try:
            E, fitted = solve_segment(trial, min(1.0, 1.3 * best[0] + 0.02), topo, first, lo, gam,
                                      stages=((30, 40), (300, 60), (3000, 60)))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        if E < best[0] - 1e-6:
            best = (E, fitted)
    return best[0], best[1].tolist(), topo, lo


def _sweep_job(job):
    """Continue one solution along consecutive rungs; returns ``[(lo, E, theta), ...]``."""
    theta, E, topo, first, gam, rungs = job
    out, theta = [], np.asarray(theta, dtype=float)
    for lo in rungs:
        try:
            E, theta = solve_segment(theta, min(1.0, 1.2 * E + 0.02), topo, first, lo, gam)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            break
        if not np.isfinite(E):
            break
        out.append((lo, E, theta.tolist()))
    return topo, out


def _insert(family: dict, lo: float, E: float, theta, topo, top: int = 6) -> None:
    rows = family["rungs"].setdefault(str(lo), [])
    rows.append({"E": float(E), "theta": list(map(float, theta)), "topo": [list(op) for op in topo]})
    rows.sort(key=lambda row: row["E"])
    unique = []
    for row in rows:
        if all(abs(row["E"] - other["E"]) > 1e-6 or row["topo"] != other["topo"] for other in unique):
            unique.append(row)
    family["rungs"][str(lo)] = unique[:top]


def ps3_anchors() -> dict:
    """PS3_55's two segments as seeds for the 5-squaring families (zhehangdu/polar-switch, MIT)."""
    lift_general = theta_from_rows(bl.PS3_55[0][1], "general")
    lift_semi = theta_from_rows(bl.PS3_55[0][1], "semi")
    sym_general = theta_from_rows(bl.PS3_55[1][1], "general")
    return {"lift-k5-b3-general": [(0.0022, *lift_general)], "lift-k5-b3-semi": [(0.0022, *lift_semi)],
            "sym-k5-b3-general": [(0.045, *sym_general)]}


def improve_family(family: dict, seeds, workers: int, iterations: int, candidates: int = 4) -> None:
    """Iterated local search at a few rungs, then sweeps down and up the ladder (in place)."""
    gam = 0.0 if family["mode"] == "lift" else 1.0
    first, ladder = family["first"], _ladder_of(family)
    zero = family["k"] >= 6
    rungs = [lo for lo in ILS_RUNGS["zero" if zero else family["mode"]] if lo in ladder]
    jobs = []
    for lo in rungs:
        starts = [(row["theta"], tuple(map(tuple, row["topo"]))) for row in family["rungs"][str(lo)][:candidates]]
        starts += [(theta, topo) for seed_lo, theta, topo in seeds if seed_lo == lo or seed_lo not in rungs]
        for i, (theta, topo) in enumerate(starts):
            for j, sigma in enumerate((0.25, 0.6)):
                jobs.append((list(theta), topo, first, gam, lo, iterations, 1000 * i + j, sigma))
    began = time.perf_counter()
    with ProcessPoolExecutor(min(workers, MAX_WORKERS)) as pool:
        found = [r for r in pool.map(_ils_job, jobs, chunksize=1) if np.isfinite(r[0])]
        for E, theta, topo, lo in found:
            _insert(family, lo, E, theta, topo)
        sweeps = []
        for E, theta, topo, lo in found:
            i = ladder.index(lo)
            sweeps.append((theta, E, topo, first, gam, ladder[i + 1:]))
            sweeps.append((theta, E, topo, first, gam, ladder[:i][::-1]))
        for topo, results in pool.map(_sweep_job, sweeps, chunksize=1):
            for lo, E, theta in results:
                _insert(family, lo, E, theta, topo)
    family["improve_seconds"] = family.get("improve_seconds", 0.0) + time.perf_counter() - began
    family["improved"] = family.get("improved", 0) + 1


def run_improve(cache: dict, save, workers: int, iterations: int, rounds: int, zero_restart: bool) -> None:
    anchors = ps3_anchors()
    for k, buffers, first, mode, ladder, _ in FAMILIES:
        key = family_key(k, buffers, first, mode)
        if (k >= 6) != zero_restart:
            continue
        family = cache[key]
        while family.get("improved", 0) < rounds:
            improve_family(family, anchors.get(key, ()) if not family.get("improved") else (), workers, iterations)
            save()
            best = {lo: round(rows[0]["E"], 4) for lo, rows in family["rungs"].items() if rows}
            print(f"improve {key} round {family['improved']}: {family['improve_seconds']:.0f}s, E by lo {best}",
                  flush=True)


# --------------------------------------------------------------------------------------
# 4. Composites and the exact-arithmetic score on the harvested momenta
# --------------------------------------------------------------------------------------
def harvested_spectra(path: Path) -> list[dict]:
    """FP64 singular values of the 45 harvested momenta (all layers), computed on the CPU."""
    import torch

    data = torch.load(path, map_location="cpu", weights_only=True)
    out = []
    for pair in data["pairs"]:
        M = pair["current"].double()
        out.append({"name": f"{pair['phase']}/{pair['parameter']}", "n": min(M.shape),
                    "s": torch.linalg.svdvals(M).numpy()})
    return out


def nominal_x(s: np.ndarray, n: int, u_safe: float) -> np.ndarray:
    """``sigma / sigma_hat`` as the FP16 runtime scales it, from an exact spectrum."""
    import torch

    from research.polar_opt import schatten_switch_blackwell as ssb

    prescale, alpha = ssb._prescale(n)
    lam = (prescale * s / np.linalg.norm(s)) ** 2
    upper = ssb._upper(float(np.sqrt((lam ** 2).sum())), float(alpha * np.sqrt((lam ** 4).sum())), n, torch.float16)
    return np.sqrt(lam / upper) / u_safe


def harvested_score(f, spectra, u_safe: float | None) -> dict:
    """Relative Frobenius polar error per matrix: ``sqrt(mean((f(x) - 1)^2))``.

    ``u_safe=None`` scores a Frobenius-normalized map (``PS3_55``).
    """
    errors = []
    for row in spectra:
        x = row["s"] / np.linalg.norm(row["s"]) if u_safe is None else nominal_x(row["s"], row["n"], u_safe)
        with np.errstate(all="ignore"):
            errors.append(float(np.sqrt(np.mean((f(x) - 1.0) ** 2))))
    errors = np.array(errors)
    return {"median": float(np.median(errors)), "max": float(errors.max()), "errors": errors.tolist()}


def relabel(rows) -> tuple:
    """Swap registers 1 and 2 if needed so that the first row writes register 1."""
    if rows[0][0] == 1:
        return tuple(rows)
    swap = {0: 0, 1: 2, 2: 1}
    return tuple((swap[d], swap[s], swap[a], *rest) for d, s, a, *rest in rows)


def compose(seg1: dict, seg2: dict | None, u: float, name: str, lo: float, error: float):
    """Join a lift segment and a symmetric segment with headroom ``u`` (see the module docstring).

    Segment 1 is bounded by 1 on ``[0, 1]`` and its output is divided by ``u``, so segment 2 is
    used on ``[0, 1 / u]`` of its bounded interval ``[0, 1]``. ``seg2=None`` is a zero-restart chain.
    """
    from research.polar_opt.schatten_switch_blackwell import SchattenSwitchCircuit

    scale = 1.0
    if seg2 is not None:  # exact bound: the LP holds f1 <= 1 only on its grid
        dense = np.unique(np.r_[np.linspace(0, 1, 40001), np.geomspace(1e-7, 1, 20001)])
        top = float(segment_scalar(rows_from_theta(seg1["theta"], seg1["topo"], seg1["first"]), dense).max())
        scale = 1.0 / (u * max(1.0, top))
    rows1 = relabel(normalize_rows(rows_from_theta(seg1["theta"], seg1["topo"], seg1["first"], scale)))
    table = [(0.0, rows1)]
    if seg2 is not None:
        table.append((0.0, normalize_rows(rows_from_theta(seg2["theta"], seg2["topo"], seg2["first"]))))
    buffers = 1 + max(max(row[:3]) for _, rows in table for row in rows)
    return SchattenSwitchCircuit(name, tuple(table), buffers, u, lo, error)


def describe(circuit, spectra) -> dict:
    """Scalar facts of a circuit in exact arithmetic. The band is ``[circuit.lo, 1 / u_safe]``."""
    u = circuit.u_safe
    x = np.unique(np.r_[np.geomspace(1e-6, 1, 20001), np.linspace(0, 1, 20001)])
    with np.errstate(all="ignore"):
        f = circuit.scalar(x)
    inside = x <= 1 / u
    bad = np.nonzero(inside & (np.abs(f - 1) > 0.1))[0]
    lo_10 = float(x[bad[-1] + 1]) if len(bad) and x[bad[-1]] < 1 / u - 1e-3 else (None if len(bad) else 0.0)
    band = inside & (x >= circuit.lo)
    score = harvested_score(circuit.scalar, spectra, u)
    return {"slope_at_zero": float(circuit.scalar(1e-9) / 1e-9), "half_gain_x": float(x[np.argmax(f >= 0.5)]),
            "lo_for_error_0.1": lo_10, "band_lo": circuit.lo, "band_error": float(np.abs(f[band] - 1).max()),
            "max_on_unit": float(f.max()), "min_on_unit": float(f.min()),
            "value_at_headroom_edge": float(circuit.scalar(1.0)),
            "harvested_median": score["median"], "harvested_max": score["max"],
            "parity": max(score["median"] / PS3_MEDIAN, score["max"] / PS3_MAX)}


def _pair_job(job):
    """Refit the symmetric segment at the exact ``lo2`` that a lift segment needs."""
    seg1, row2, first2, lo2, tag = job
    try:
        E2, theta2 = solve_segment(np.asarray(row2["theta"]), min(1.0, row2["E"] + 0.02),
                                   tuple(map(tuple, row2["topo"])), first2, lo2, 1.0)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return tag, seg1, None
    if not np.isfinite(E2) or E2 > row2["E"] + 1e-9:  # the rung solution is valid on the larger interval
        E2, theta2 = row2["E"], row2["theta"]
    return tag, seg1, {"theta": list(map(float, theta2)), "topo": row2["topo"], "first": first2, "lo": lo2, "E": E2}


def composites(cache: dict, spectra, u: float, workers: int, splits) -> list[dict]:
    """All (lift family, rung) x symmetric family pairs for the given ``(k1, k2)`` splits."""
    jobs = []
    for k1, k2 in splits:
        for key1, fam1 in cache.items():
            if fam1["mode"] != "lift" or fam1["k"] != k1:
                continue
            for key2, fam2 in cache.items():
                if fam2["mode"] != "sym" or fam2["k"] != k2 or fam2["buffers"] != fam1["buffers"]:
                    continue
                rungs2 = sorted(float(lo) for lo, rows in fam2["rungs"].items() if rows)
                for lo1, rows1 in fam1["rungs"].items():
                    if not rows1:
                        continue
                    lo2 = (1.0 - rows1[0]["E"]) / u
                    below = [r for r in rungs2 if r <= lo2]
                    if not below or lo2 > 0.5:
                        continue
                    seg1 = {"theta": rows1[0]["theta"], "topo": rows1[0]["topo"], "first": fam1["first"],
                            "lo": float(lo1), "E": rows1[0]["E"]}
                    for row2 in fam2["rungs"][str(below[-1])][:2]:
                        jobs.append((seg1, row2, fam2["first"], lo2, (key1, key2)))
    out = []
    with ProcessPoolExecutor(min(workers, MAX_WORKERS)) as pool:
        for (key1, key2), seg1, seg2 in pool.map(_pair_job, jobs, chunksize=4):
            if seg2 is None:
                continue
            k1, k2 = len(seg1["topo"]), len(seg2["topo"])
            circuit = compose(seg1, seg2, u, f"{k1}+{k2}", seg1["lo"], seg2["E"])
            out.append({"k1": k1, "k2": k2, "first": seg1["first"], "families": [key1, key2], "u_safe": u,
                        "seg1": seg1, "seg2": seg2, **describe(circuit, spectra)})
    return out


# --------------------------------------------------------------------------------------
# 5. Joint polish on the harvested score with hard safety bounds
# --------------------------------------------------------------------------------------
def polish(seg1: dict, seg2: dict, u: float, spectra, targets, *, band_lo: float, band_error: float = 0.1,
           evaluations: int = 300) -> tuple[dict, dict, dict]:
    """Tune both segments so every harvested error is at most its target (PS3_55's error).

    Hard bounds (penalties, then an exact rescale): ``0 <= f1 <= 1`` on ``[0, 1]``,
    ``0 <= f2 <= 1 + band_error`` on ``[0, 1]``, ``|f - 1| <= band_error`` on
    ``[band_lo, 1 / u]`` and ``f >= 1 - 2 band_error`` on ``(1 / u, 1]``.
    """
    topo1, topo2 = tuple(map(tuple, seg1["topo"])), tuple(map(tuple, seg2["topo"]))
    first1, first2 = seg1["first"], seg2["first"]
    n1 = len(seg1["theta"])
    xs = [nominal_x(row["s"], row["n"], u) for row in spectra]
    grid = np.unique(np.r_[np.linspace(0, 1, 241), np.geomspace(1e-4, 1, 160), 1 - np.geomspace(1e-4, 0.2, 60)])
    band = np.geomspace(band_lo, 1 / u, 240)
    edge = np.linspace(1 / u, 1, 40)
    targets = np.asarray(targets)

    def f1(theta, x):
        return x * chain(theta, topo1, x * x, first1, jac=False)

    def f2(theta, y):
        return y * chain(theta, topo2, y * y, first2, jac=False)

    def residual(z, penalty=60.0):
        t1, t2 = z[:n1], z[n1:]
        with np.errstate(all="ignore"):
            parity = [np.sqrt(np.mean((f2(t2, f1(t1, x) / u) - 1) ** 2)) for x in xs]
            g1, g2 = f1(t1, grid), f2(t2, grid)
            fb, fe = f2(t2, f1(t1, band) / u), f2(t2, f1(t1, edge) / u)
        hinge = lambda v: penalty * np.maximum(v, 0.0)
        r = np.concatenate([(np.array(parity) / targets) ** 2, hinge(g1 - 1), hinge(-g1), hinge(g2 - 1 - band_error),
                            hinge(-g2), hinge(np.abs(fb - 1) - band_error), hinge(1 - 2 * band_error - fe)])
        return np.where(np.isfinite(r), r, 1e6)

    z0 = np.r_[seg1["theta"], seg2["theta"]]
    z = least_squares(residual, z0, method="trf", x_scale="jac", max_nfev=evaluations).x
    t1, t2 = z[:n1].copy(), z[n1:].copy()
    dense = np.unique(np.r_[np.linspace(0, 1, 20001), np.geomspace(1e-6, 1, 4001)])
    t1[-3:] /= max(1.0, float(f1(t1, dense).max()))  # exact: f1 <= 1 on [0, 1]
    new1 = {**seg1, "theta": t1.tolist()}
    new2 = {**seg2, "theta": t2.tolist()}
    before, after = float(np.sum(residual(z0) ** 2)), float(np.sum(residual(np.r_[t1, t2]) ** 2))
    return new1, new2, {"objective_before": before, "objective_after": after, "accepted": after < before}


# --------------------------------------------------------------------------------------
# 6. Native verification on the GPU
# --------------------------------------------------------------------------------------
CAP = 1.25  # largest singular value allowed on the safety sweeps
DTYPE_NAMES = {"fp16": "float16", "bf16": "bfloat16"}


def _sign_matrix(shape, seed):
    import torch

    g = torch.Generator().manual_seed(seed)
    u = (torch.randint(2, (shape[0],), generator=g) * 2 - 1).float()
    v = (torch.randint(2, (shape[1],), generator=g) * 2 - 1).float()
    return (u[:, None] * v[None, :]).cuda()


def rank1_sweep(circuit, dtype, shape=(64, 128), points: int = 1200, seed: int = 0) -> dict:
    """Coherent rank-one sign matrix with an explicit scale: ``x = sigma / sigma_hat`` over ``[0, 1]``.

    ``x = 1`` is ``sigma_max = u_safe * bound``, so the sweep covers ``[0, u_safe]`` in units of
    the bound. The largest output singular value is reported exactly (FP64 SVD) and by the
    Frobenius norm, which is an upper bound. The batch call uses one scale per member; every
    30th point is repeated as a single-matrix call.
    """
    import torch

    from research.polar_opt.schatten_switch_blackwell import schatten_switch_blackwell

    M = _sign_matrix(shape, seed)
    x = torch.linspace(0.002, 1.0, points, dtype=torch.float64)
    out = schatten_switch_blackwell(M.expand(points, *shape).contiguous(), circuit, dtype=dtype,
                                    sigma_hat=(1.0 / x).float().cuda(), allow_unsafe_bf16=True).double()
    finite = bool(torch.isfinite(out).all())
    top = torch.linalg.matrix_norm(out, ord=2).cpu() if finite else torch.full((points,), float("inf"))
    frobenius = out.norm(dim=(-2, -1)).cpu()
    model = torch.from_numpy(np.asarray(circuit.scalar(x.numpy())))
    single = []
    for i in range(0, points, 30):
        one = schatten_switch_blackwell(M, circuit, dtype=dtype, sigma_hat=1.0 / x[i].item(),
                                        allow_unsafe_bf16=True).double()
        single.append(abs(torch.linalg.matrix_norm(one, ord=2).item() - top[i].item()))
    worst = int(top.argmax())
    return {"shape": list(shape), "points": points, "finite": finite, "max_sigma": float(top.max()),
            "max_sigma_at_x": float(x[worst]), "max_frobenius": float(frobenius.max()),
            "max_scalar_model": float(model.max()), "max_abs_deviation_from_scalar": float((top - model).abs().max()),
            "max_batch_vs_single": float(max(single)), "passed": finite and float(frobenius.max()) <= CAP}


def identity_plus_constant(n: int, ratio: float):
    """``p I + c 11^T`` with singular values ``(s1, p, ..., p)``, ``p = ratio * s1`` (audit R1)."""
    import torch

    s1 = 1.0 / np.sqrt(1.0 + (n - 1) * ratio * ratio)
    p = ratio * s1
    G = torch.full((n, n), (s1 - p) / n, dtype=torch.float32, device="cuda")
    G.diagonal().add_(p)
    return G


def spectral_bound(Q) -> float:
    """Rigorous upper bound on the largest singular value: ``sqrt(||Q||_1 ||Q||_inf)``.

    It is tight for ``a I + b 11^T`` with ``b > 0``, the shape of the outputs checked with it.
    """
    Q = Q.double().abs()
    return float((Q.sum(-1).max() * Q.sum(-2).max()).sqrt())


def subnormal_family(circuit, dtype, n: int = 4096, ratios=None) -> dict:
    """Audit R1: Schatten-8 scaling on ``p I + c 11^T`` at a size where trace-1 Gram entries are
    FP16 subnormals. Cycle 1 circuits returned ``Inf`` or a top gain of 5.4 on this family."""
    import torch

    from research.polar_opt.schatten_switch_blackwell import schatten_switch_blackwell

    ratios = np.round(np.linspace(0.15, 0.30, 16), 4) if ratios is None else ratios
    rows = []
    for ratio in ratios:
        Q = schatten_switch_blackwell(identity_plus_constant(n, float(ratio)), circuit, dtype=dtype,
                                      allow_unsafe_bf16=True)
        finite = bool(torch.isfinite(Q).all())
        rows.append({"ratio": float(ratio), "finite": finite,
                     "sigma_max_bound": spectral_bound(Q) if finite else float("inf")})
        del Q
    worst = max(r["sigma_max_bound"] for r in rows)
    return {"n": n, "cases": len(rows), "nonfinite": sum(not r["finite"] for r in rows),
            "max_sigma_bound": worst if np.isfinite(worst) else None,
            "passed": all(r["finite"] for r in rows) and worst <= CAP, "rows": rows}


def tier1(circuits: dict, trajectories: Path, repeats: int = 5) -> dict:
    """The 117 Tier-1 cases: accuracy, divergence and wall-clock latency next to ``PS3_55``."""
    import torch

    from research.polar_opt.harness_tier1 import Registry, benchmark, harvested_cases, summarize, synthetic_cases
    from research.polar_opt.polar_switch_blackwell import polar_switch_blackwell
    from research.polar_opt.schatten_switch_blackwell import schatten_switch_blackwell

    registry = Registry()
    registry.register("ps3_55_blackwell_fp16", lambda x: polar_switch_blackwell(x, "PS3_55", dtype=torch.float16))
    for name, (circuit, dtype) in circuits.items():
        registry.register(name, lambda x, c=circuit, d=dtype: schatten_switch_blackwell(x, c, dtype=d, allow_unsafe_bf16=True))
    cases = list(synthetic_cases([128, 256], [1, 2, 4, 8], 123)) + list(harvested_cases(trajectories, all_layers=True))
    results = benchmark(cases, registry, torch.device("cuda"), warmup=3, repeats=repeats, seed=123, divergence_limit=2.0)
    summary = summarize(results)
    keep = ("cases", "divergence_rate", "nonfinite_rate", "median_relative_frobenius_error",
            "max_relative_frobenius_error", "sigma_max", "median_case_latency_ms")
    out = {suite: {name: {k: row[k] for k in keep} for name, row in methods.items()} for suite, methods in summary.items()}
    for suite in out.values():
        for row in suite.values():
            row["diverged"] = round(row["divergence_rate"] * row["cases"])
    return {"case_count": len(cases), "repeats": repeats, "summary": out}


def latency(circuits: dict, shapes=((512, 2048), (1024, 4096), (2048, 8192), (4096, 4096)), iters: int = 15,
            warmup: int = 4) -> dict:
    """Short GPU-event and wall-clock check against ``ps3_55_blackwell_fp16`` (not a clean benchmark)."""
    import torch

    from research.polar_opt.baselines import svd_polar_fp64
    from research.polar_opt.polar_switch_blackwell import _time, polar_switch_blackwell
    from research.polar_opt.schatten_switch_blackwell import schatten_switch_blackwell

    calls = {"ps3_55_blackwell_fp16": lambda G: polar_switch_blackwell(G, "PS3_55", dtype=torch.float16)}
    for name, (circuit, dtype) in circuits.items():
        calls[name] = lambda G, c=circuit, d=dtype: schatten_switch_blackwell(G, c, dtype=d, allow_unsafe_bf16=True)
    flush = torch.empty(4 * torch.cuda.get_device_properties(0).L2_cache_size, dtype=torch.uint8, device="cuda")
    out = {"host_load_start": os.getloadavg()[0], "iters": iters, "warmup": warmup,
           "clock": "CUDA events (gpu) and synchronized host clock (wall), L2 flushed, random order", "shapes": {}}
    for shape in shapes:
        torch.manual_seed(0)
        G = torch.randn(*shape, device="cuda")
        P = svd_polar_fp64(G)
        stats = _time(calls, G, flush, warmup, iters, 0)
        base = stats["ps3_55_blackwell_fp16"]
        rows = {}
        for name, call in calls.items():
            Q = call(G).double()
            rows[name] = {"gpu_ms": stats[name]["gpu"]["median_ms"], "gpu_iqr_ms": stats[name]["gpu"]["iqr_ms"],
                          "wall_ms": stats[name]["wall"]["median_ms"],
                          "speedup_gpu": base["gpu"]["median_ms"] / stats[name]["gpu"]["median_ms"],
                          "speedup_wall": base["wall"]["median_ms"] / stats[name]["wall"]["median_ms"],
                          "rel_fro_err": float(((Q - P).norm() / P.norm()).item())}
        out["shapes"]["x".join(map(str, shape))] = rows
        del G, P
    out["kernel_launches_measured"] = {name: measure_launches(call) for name, call in calls.items()}
    out["host_load_end"] = os.getloadavg()[0]
    return out


def gram_restart_probe(trajectories: Path) -> dict:
    """Negative result: PS3_55 with a Gram-space restart (``R' = Q R Q``, one SYRK, one ``Q @ X``).

    It would cut the cost to ``(3 alpha + 13) n^3``. In FP16 the two products that form ``R'``
    amplify rounding by ``||Q||`` (about 46), and the small eigenvalues of ``R'`` are lost.
    """
    import torch

    from research.polar_opt.baselines import svd_polar_fp64
    from research.polar_opt.harness_tier1 import harvested_cases, synthetic_cases
    from research.polar_opt.polar_switch_blackwell import _apply, _gram, _prepare, _symprod

    def run(G):
        X, _ = _prepare(G, torch.float16)
        n = min(X.shape[-2:])
        w = [X.new_empty(len(X), n, n) for _ in range(6)]
        _gram(X, w[0])
        w[3].copy_(w[0])
        regs = [w[0], w[1], w[2]]
        for dst, src, add, a, b, c in bl.PS3_55[0][1]:
            _symprod(regs[src], regs[src], regs[add], regs[dst], a, b, c)
        Q1 = regs[dst]
        _symprod(Q1, w[3], w[3], w[4], 1.0, 0.0)
        _symprod(w[4], Q1, w[4], w[5], 1.0, 0.0)
        regs2 = [w[5]] + [r for r in regs if r is not Q1]
        for dst, src, add, a, b, c in bl.PS3_55[1][1]:
            _symprod(regs2[src], regs2[src], regs2[add], regs2[dst], a, b, c)
        _symprod(regs2[dst], Q1, Q1, w[3], 1.0, 0.0)
        return _apply(w[3], X, 0.0, G.dtype).reshape(G.shape)

    suites = {}
    cases = list(synthetic_cases([128, 256], [1, 2, 4, 8], 123)) + list(harvested_cases(trajectories, all_layers=True))
    for case in cases:
        G = case.matrix.cuda()
        Q = run(G).double()
        P = svd_polar_fp64(G)
        finite = bool(torch.isfinite(Q).all())
        error = ((Q - P).norm() / min(Q.shape) ** 0.5).item() if finite else float("inf")
        top = torch.linalg.svdvals(Q).max().item() if finite else float("inf")
        suites.setdefault(case.suite, []).append((error, top))
    out = {}
    for suite, rows in suites.items():
        errors = np.array([r[0] for r in rows])
        tops = np.array([r[1] for r in rows])
        out[suite] = {"cases": len(rows), "median_error": float(np.median(errors)),
                      "nonfinite": int((~np.isfinite(errors)).sum()), "diverged": int((~(tops <= 2.0)).sum()),
                      "max_sigma": float(tops[np.isfinite(tops)].max()) if np.isfinite(tops).any() else None}
    return {"flops": "(3α + 13)n³ instead of (6α + 10)n³", "verdict": "rejected: diverges in FP16", "suites": out}


# --------------------------------------------------------------------------------------
# 7. Driver
# --------------------------------------------------------------------------------------
SPLITS = ((3, 4), (4, 4), (4, 5), (5, 4), (3, 5), (5, 3))
U_FP16, U_BF16 = 1.05, 1.15


def ps3_scalar(x):
    return table_scalar(bl.PS3_55, x)


def ranked_candidates(grid: list, spectra, targets, u: float, per_class: int = 3) -> dict:
    """``{(k1, k2): [records by parity]}``: the best ``per_class`` composites of every
    ``(k1, k2, first row)`` class, plus the polished version of the best one when it helps."""
    classes = {}
    for row in grid:
        classes.setdefault((row["k1"], row["k2"], row["first"]), []).append(row)
    out = {}
    for (k1, k2, first), rows in sorted(classes.items()):
        rows.sort(key=lambda r: r["parity"])
        kept = [{**row, "polish": None} for row in rows[:per_class]]
        row = rows[0]
        band_lo, band_error = 1.3 * row["seg1"]["lo"], max(0.1, 1.05 * row["seg2"]["E"])
        try:
            seg1, seg2, log = polish(row["seg1"], row["seg2"], u, spectra, targets, band_lo=band_lo, band_error=band_error)
            facts = describe(compose(seg1, seg2, u, "polished", band_lo, band_error), spectra)
            ok = (facts["parity"] < row["parity"] and facts["min_on_unit"] >= -1e-9
                  and facts["max_on_unit"] <= 1 + band_error + 0.01 and facts["band_error"] <= band_error + 0.01)
            log = {**log, "parity_before": row["parity"], "parity_after": facts["parity"], "kept": ok}
            if ok:
                kept.append({**row, **facts, "seg1": seg1, "seg2": seg2, "polish": log})
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            log = {"error": str(error)}
        out.setdefault((k1, k2), []).extend(kept)
        print(f"class {k1}+{k2} {first}: raw parity {row['parity']:.3f} (median {row['harvested_median']:.4f}, "
              f"max {row['harvested_max']:.4f}), polish {log}", flush=True)
    for rows in out.values():
        rows.sort(key=lambda r: r["parity"])
    return out


def zero_restart(cache: dict, spectra, u: float) -> dict:
    """Best single-segment chain per ``k``: the harvested score shows why a restart is needed."""
    out = {}
    for key, family in cache.items():
        if family["k"] < 6:
            continue
        rows = []
        for lo, fits in family["rungs"].items():
            if not fits:
                continue
            seg = {"theta": fits[0]["theta"], "topo": fits[0]["topo"], "first": family["first"]}
            circuit = compose(seg, None, u, key, float(lo), fits[0]["E"])
            rows.append({"lo": float(lo), "E": fits[0]["E"], **describe(circuit, spectra), "circuit": circuit.to_dict()})
        rows.sort(key=lambda r: r["parity"])
        out[key] = {"k": family["k"], "first": family["first"], "degree_bound_on_slope": 2 ** (family["k"] + 1) + 1,
                    "best": rows[0] if rows else None,
                    "by_lo": [{k: r[k] for k in ("lo", "E", "slope_at_zero", "harvested_median", "harvested_max")}
                              for r in sorted(rows, key=lambda r: -r["lo"])]}
    return out


def verify(circuit, dtype_name: str) -> dict:
    """Safety checks with the real matrix code. BF16 gets denser sweeps: its windows are narrow."""
    import torch

    dtype = getattr(torch, DTYPE_NAMES[dtype_name])
    dense = dtype_name == "bf16"
    sweeps = [rank1_sweep(circuit, dtype, (64, 128), 2400 if dense else 1200),
              rank1_sweep(circuit, dtype, (256, 256), 800 if dense else 600, seed=1)]
    if dense:
        sweeps.append(rank1_sweep(circuit, dtype, (128, 512), 1600, seed=2))
    out = {"rank1_sweeps": sweeps, "passed": all(r["passed"] for r in sweeps)}
    if dtype_name == "fp16":
        out["identity_plus_constant_4096"] = subnormal_family(circuit, dtype)
        out["identity_plus_constant_8192"] = subnormal_family(circuit, dtype, n=8192, ratios=[0.15, 0.3])
        out["passed"] = out["passed"] and out["identity_plus_constant_4096"]["passed"] \
            and out["identity_plus_constant_8192"]["passed"]
    return out


def kernel_launches(circuit, n: int) -> dict:
    """GPU kernels of one natural call (``sigma_hat=None``) on a contiguous FP32 matrix.

    Analytic launch count applies only to zero-residual segment-1 circuits
    (``circuit.table[0][0] == 0.0``, where segment 1 has no identity residual so
    its square reuses the SYRK matrix product count) on contiguous FP32 inputs.
    Matrix products: per segment SYRK + squares + ``Q @ X``. Four reductions and elementwise
    kernels: the safe norm partials, the normalize-and-prescale pass, the dual norm of ``R`` and
    ``R^2``, and the fold kernel, which also applies the first-row ``b R + c I`` (so ``semi`` and
    ``general`` first rows add no launch). The path is sync-free, so there are no host reads, and
    the count does not depend on ``n``. ``measure_launches`` checks these numbers with the profiler.
    """
    if circuit.table[0][0] != 0.0:
        raise ValueError("kernel_launches analytic count supports zero-residual segment-1 circuits only")
    products = 2 * len(circuit.table) + sum(circuit.squarings)
    return {"matrix_products": products, "total": products + 4, "host_reads": 0}


def measure_launches(call, shape=(512, 2048)) -> dict:
    """Profiled CUDA kernel count of one warm call, and whether it synchronizes with the host."""
    import torch

    G = torch.randn(*shape, device="cuda")
    call(G)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        call(G)
        torch.cuda.synchronize()
    kernels = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
                  and not e.name.startswith(("Memcpy", "Memset")))
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        call(G)
        host_reads = 0
    except RuntimeError:
        host_reads = 1  # at least one synchronizing host read
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    return {"shape": list(shape), "kernels": kernels, "host_reads": host_reads}


def git_info() -> dict:
    """Commit and dirty state of the checkout that ran the search (tracked files only)."""
    import subprocess

    root = Path(__file__).resolve().parents[2]

    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                                  timeout=10, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(status) if status is not None else None}


def _clean(value):
    """JSON without NaN or Inf: non-finite floats become ``None``."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _strip(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in ("seg1", "seg2")}


# --------------------------------------------------------------------------------------
# 8. Opt-in early-spectrum re-polish (Cycle 3, C3-02: --include-early-srank)
# --------------------------------------------------------------------------------------
# Real step-20 momenta have stable rank about 1.2: one dominant singular value and a tail far
# below it, where every Schatten Switch circuit lifts less than PS3_55. This mode adds purely
# synthetic near-rank-1 spectra to the polish targets. It never reads real held-out momenta;
# those are scored only out of sample by ``eval_heldout_nesterov.py``.
EARLY_SRANKS = (1.03, 1.08, 1.15, 1.25, 1.4, 1.6)
EARLY_SIZES = (512, 1024)
EARLY_POWERS = (0.5, 1.0, 1.25)
EARLY_SECONDARIES = (3, 10)
EARLY_FLOORS = (1e-3, 2e-3, 4e-3)  # floor energy n f^2 stays below the smallest tail energy 0.03
EARLY_SPLITS = ((5, 4), (5, 3), (3, 5), (3, 4))
# Polish jobs (weight, band error): early target = weight * PS3_55 error on that spectrum; band
# error None keeps the start's design error, 0.11 lets the band loosen to lift the tail.
EARLY_JOBS = ((1.0, None), (1.6, None), (2.5, None), (1.0, 0.11), (0.7, 0.11))
BF16_SPLITS = {(3, 5), (3, 4)}  # splits whose registered circuit passes the BF16 sweeps
R1_SIZES = (1023, 1024, 2048, 4096, 8192)
R1_RATIOS = (0.15, 0.175, 0.2, 0.246, 0.3)


def early_srank_spectra(sizes=EARLY_SIZES, sranks=EARLY_SRANKS, seed: int = 0) -> list[dict]:
    """Synthetic near-rank-1 spectra: a unit spike plus a tail of energy ``srank - 1``.

    Two tail shapes, both with a wide dynamic range (as an early gradient has: a few strong
    directions and a weak bulk):
      ``pow{p}``         ``i^-p`` for ``p`` in 0.5, 1, 1.25
      ``spk{r}_f{f}``    ``r`` geometric secondary values (1 down to 0.2) over a noise floor: the
                         singular values of a Gaussian ``(n - 1 - r) x 2n`` matrix with RMS ``f``
    Stable rank ``||s||^2 / s_1^2`` equals ``srank`` and the tail stays below the spike.
    """
    rng = np.random.default_rng(seed)
    out = []
    for n in sizes:
        i = np.arange(1, n, dtype=np.float64)
        bulk = np.sort(np.linalg.svd(rng.standard_normal((n - 1, 2 * n)), compute_uv=False))[::-1]
        bulk /= np.sqrt(np.mean(bulk ** 2))
        for srank in sranks:
            energy = srank - 1.0
            tails = {f"pow{p}": i ** -p * np.sqrt(energy / np.sum(i ** (-2 * p))) for p in EARLY_POWERS}
            for r in EARLY_SECONDARIES:
                for f in EARLY_FLOORS:
                    floor = f * bulk[:n - 1 - r]
                    secondary = np.geomspace(1.0, 0.2, r)
                    secondary *= np.sqrt((energy - np.sum(floor ** 2)) / np.sum(secondary ** 2))
                    tails[f"spk{r}_f{f}"] = np.r_[secondary, floor]
            for kind, tail in tails.items():
                s = np.r_[1.0, np.sort(tail)[::-1]]
                out.append({"name": f"early/{kind}/n{n}/srank{srank}", "n": n, "s": s,
                            "srank": float(np.sum(s ** 2))})
    return out


def _early_scores(f, cycle1, early, u) -> dict:
    """Exact-arithmetic error on the Cycle 1 momenta and on the synthetic early spectra."""
    a, b = harvested_score(f, cycle1, u), harvested_score(f, early, u)
    e = np.array(b["errors"])
    return {"cycle1_median": a["median"], "cycle1_max": a["max"], "early_median": b["median"],
            "early_p90": float(np.quantile(e, 0.9)), "early_max": b["max"], "early_errors": b["errors"]}


def _early_polish_job(job):
    tag, weight, seg1, seg2, spectra, targets, band_lo, band_error, evaluations = job
    try:
        with np.errstate(all="ignore"):
            new1, new2, log = polish(seg1, seg2, U_FP16, spectra, targets, band_lo=band_lo, band_error=band_error,
                                     evaluations=evaluations)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return tag, weight, band_error, None, None, {"error": str(error)}
    return tag, weight, band_error, new1, new2, log


def early_starts(artifact: dict, cache: dict, cycle1, early, workers: int, per_split: int) -> list[dict]:
    """Start points: the registered circuits' segments and the best cache composites per split."""
    starts = []
    for name, record in artifact["selected"].items():
        split = tuple(record["circuit"]["squarings"])
        if split in EARLY_SPLITS:
            starts.append({"tag": name, "split": split, "first": record["first_row"],
                           "seg1": record["segments"]["seg1"], "seg2": record["segments"]["seg2"],
                           "band_lo": record["scalar"]["band_lo"], "band_error": record["scalar"]["band_error"]})
    if cache and per_split:
        grid = composites(cache, cycle1 + early, U_FP16, workers, EARLY_SPLITS)
        for split in EARLY_SPLITS:
            rows = sorted((r for r in grid if (r["k1"], r["k2"]) == split), key=lambda r: r["harvested_median"])
            for i, row in enumerate(rows[:per_split]):
                starts.append({"tag": f"cache_{split[0]}{split[1]}_{i}", "split": split, "first": row["first"],
                               "seg1": row["seg1"], "seg2": row["seg2"], "band_lo": 1.3 * row["seg1"]["lo"],
                               "band_error": max(0.1, 1.05 * row["seg2"]["E"])})
    return starts


def r1_extended(circuit, dtype, sizes=R1_SIZES, ratios=R1_RATIOS) -> dict:
    """Fable R1 over ``n = 1023..8192`` (``8192`` with two ratios to bound the run time)."""
    rows = {str(n): subnormal_family(circuit, dtype, n=n, ratios=ratios if n < 8192 else (0.175, 0.3))
            for n in sizes}
    return {"passed": all(r["passed"] for r in rows.values()),
            "max_sigma_bound": max((r["max_sigma_bound"] or float("inf")) for r in rows.values()),
            "by_n": {n: {k: r[k] for k in ("cases", "nonfinite", "max_sigma_bound", "passed")} for n, r in rows.items()}}


def r3_long_k(circuit, dtype, k: int = 1 << 20) -> dict:
    """Fable R3 / M1: long inner dimension ``K = 2^20`` stays bounded; tiny and dead inputs give zeros."""
    import torch

    from research.polar_opt.schatten_switch_blackwell import schatten_switch_blackwell

    ones = schatten_switch_blackwell(torch.ones(1, k, device="cuda"), circuit, dtype=dtype, allow_unsafe_bf16=True)
    G = torch.randn(16, k, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))
    Q = schatten_switch_blackwell(G, circuit, dtype=dtype, allow_unsafe_bf16=True).double()
    top = float(torch.linalg.eigvalsh(Q @ Q.mT).max().clamp_min(0).sqrt())
    tiny = schatten_switch_blackwell(torch.full((64, 128), 1e-20, device="cuda"), circuit, dtype=dtype,
                                     allow_unsafe_bf16=True)
    dead = schatten_switch_blackwell(torch.zeros(64, 128, device="cuda"), circuit, dtype=dtype, allow_unsafe_bf16=True)
    out = {"k": k, "rank1_norm": float(ones.double().norm()), "random_sigma_max": top,
           "finite": bool(torch.isfinite(ones).all() and torch.isfinite(Q).all()),
           "tiny_and_dead_are_zero": bool(not tiny.any() and not dead.any())}
    out["passed"] = out["finite"] and out["tiny_and_dead_are_zero"] and out["rank1_norm"] <= CAP and top <= CAP
    return out


def graph_replay(circuit, dtype, shapes=((512, 2048), (4, 256, 512))) -> dict:
    """CUDA-graph capture after warm-up; replay must equal the eager call bit for bit."""
    import torch

    from research.polar_opt.schatten_switch_blackwell import schatten_switch_blackwell

    call = lambda x: schatten_switch_blackwell(x, circuit, dtype=dtype, allow_unsafe_bf16=True)  # noqa: E731
    rows = []
    for shape in shapes:
        static = torch.randn(*shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(31))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                call(static)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call(static)
        static.copy_(torch.randn(*shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(32)) * 1e3)
        graph.replay()
        torch.cuda.synchronize()
        rows.append({"shape": list(shape), "bitwise_equal": bool(torch.equal(out, call(static)))})
    return {"passed": all(r["bitwise_equal"] for r in rows), "shapes": rows}


def _resolve_search_output(output: Path | None, include_early_srank: bool, seed_artifact: Path) -> Path:
    """Resolve the search output path, keeping early re-polish distinct from its seed artifact."""
    if include_early_srank:
        resolved = Path("results/schatten_switch_early_polish.json") if output is None else output
        if resolved.resolve() == seed_artifact.resolve():
            raise ValueError(
                f"Early search --output ({resolved}) must differ from --seed-artifact ({seed_artifact})"
            )
        return resolved
    return Path("results/schatten_switch_search_cycle2.json") if output is None else output


def run_early_search(args) -> None:
    """Targeted re-polish of the 5+4, 5+3, 3+5 and 3+4 circuits with synthetic early spectra."""
    import torch
    from dataclasses import replace

    if args.output.resolve() == args.seed_artifact.resolve():
        raise ValueError(
            f"Early search --output ({args.output}) must differ from --seed-artifact ({args.seed_artifact})"
        )
    began = time.perf_counter()
    artifact = json.loads(args.seed_artifact.read_text())
    cache = json.loads(args.fit_cache.read_text()) if args.fit_cache and args.fit_cache.exists() else {}
    cache_note = "used" if cache else "none"
    if cache and not all(isinstance(v, dict) and "mode" in v and "rungs" in v for v in cache.values()):
        cache, cache_note = {}, "ignored: not a Schatten Switch family cache (run_fits format)"
        print(f"--fit-cache {args.fit_cache} {cache_note}; starts are the registered circuits only", flush=True)
    cycle1 = harvested_spectra(args.trajectories)
    early = early_srank_spectra()
    ps3_c1 = np.array(harvested_score(ps3_scalar, cycle1, None)["errors"])
    ps3_early = np.array(harvested_score(ps3_scalar, early, None)["errors"])
    starts = early_starts(artifact, cache, cycle1, early, args.workers, args.early_cache_starts)
    print(f"early re-polish: {len(starts)} starts, {len(early)} synthetic spectra "
          f"(PS3_55 early median {np.median(ps3_early):.4f})", flush=True)

    jobs = []
    for start in starts:
        f = compose(start["seg1"], start["seg2"], U_FP16, "start", start["band_lo"], start["band_error"]).scalar
        start["scores"] = _early_scores(f, cycle1, early, U_FP16)
        # Cycle 1 targets keep the start's own error (its mid/late win) and PS3_55 parity.
        own = np.array(harvested_score(f, cycle1, U_FP16)["errors"])
        t1 = np.minimum(np.maximum(own, 0.5 * PS3_MEDIAN), 0.97 * np.maximum(ps3_c1, PS3_MEDIAN))
        for weight, band_error in EARLY_JOBS:
            targets = np.r_[t1, weight * np.maximum(ps3_early, 1e-3)]
            jobs.append((start["tag"], weight, start["seg1"], start["seg2"], cycle1 + early, targets,
                         start["band_lo"], band_error or start["band_error"], args.early_evaluations))
    by_tag = {s["tag"]: s for s in starts}
    records = []
    with ProcessPoolExecutor(min(args.workers, MAX_WORKERS)) as pool:
        for tag, weight, band_error, seg1, seg2, log in pool.map(_early_polish_job, jobs, chunksize=1):
            start = by_tag[tag]
            row = {"start": tag, "split": list(start["split"]), "first": start["first"], "weight": weight,
                   "design_band_error": band_error, "polish": log, "start_scores": {k: v for k, v in start["scores"].items() if k != "early_errors"}}
            if seg1 is not None:
                circuit = compose(seg1, seg2, U_FP16, "early", start["band_lo"], band_error)
                facts = describe(circuit, cycle1)
                safe = (facts["min_on_unit"] >= -1e-9 and facts["max_on_unit"] <= 1 + band_error + 0.01
                        and facts["band_error"] <= band_error + 0.01)
                scores = _early_scores(circuit.scalar, cycle1, early, U_FP16)
                row.update({"seg1": seg1, "seg2": seg2, "band_lo": start["band_lo"],
                            "band_error": max(band_error, facts["band_error"]), "scalar": facts, "safe": safe,
                            "scores": {k: v for k, v in scores.items() if k != "early_errors"},
                            "cycle1_parity": scores["cycle1_median"] <= PS3_MEDIAN
                            and scores["cycle1_max"] <= PS3_MAX + 2e-3})
            records.append(row)
            s = row.get("scores")
            print(f"  {tag:<12} w={weight} E={band_error:.3f}: " + (f"early median {start['scores']['early_median']:.4f} -> "
                  f"{s['early_median']:.4f}, cycle1 median {start['scores']['cycle1_median']:.4f} -> "
                  f"{s['cycle1_median']:.4f}, safe {row['safe']}" if s else f"failed {log}"), flush=True)
    search_seconds = time.perf_counter() - began

    # GPU validation: per split, the best safe polished circuits at Cycle 1 parity.
    validated = {}
    for split in EARLY_SPLITS:
        pool_rows = sorted((r for r in records if tuple(r["split"]) == split and r.get("safe") and r["cycle1_parity"]),
                           key=lambda r: r["scores"]["early_median"])
        base = min((s["scores"]["early_median"] for s in starts if s["split"] == split and s["tag"].startswith("SS")),
                   default=np.inf)
        for i, row in enumerate([r for r in pool_rows if r["scores"]["early_median"] < 0.9 * base][:args.early_validate]):
            buffers = 1 + max(max(op) for op in row["seg1"]["topo"] + row["seg2"]["topo"])
            name = f"SS{buffers}_{split[0]}{split[1]}E{i}"
            circuit = compose(row["seg1"], row["seg2"], U_FP16, name, row["band_lo"], row["band_error"])
            check = {"fp16": verify(circuit, "fp16"), "r1_fp16": r1_extended(circuit, torch.float16),
                     "r3_fp16": r3_long_k(circuit, torch.float16), "graph_fp16": graph_replay(circuit, torch.float16)}
            gates = [check[k]["passed"] for k in ("fp16", "r1_fp16", "r3_fp16", "graph_fp16")]
            if split in BF16_SPLITS:
                check.update({"bf16": verify(circuit, "bf16"), "r1_bf16": r1_extended(circuit, torch.bfloat16)})
                bf16_ok = check["bf16"]["passed"] and check["r1_bf16"]["passed"]
                if bf16_ok:
                    circuit = replace(circuit, bf16_verified=True)
                    check["graph_bf16"] = graph_replay(circuit, torch.bfloat16)
                    bf16_ok = check["graph_bf16"]["passed"]
                    circuit = replace(circuit, bf16_verified=bf16_ok)
            passed = all(gates)
            validated[name] = {"circuit": circuit.to_dict(), "start": row["start"], "weight": row["weight"],
                               "split": list(split), "first_row": row["first"], "scores": row["scores"],
                               "scalar": row["scalar"], "segments": {"seg1": row["seg1"], "seg2": row["seg2"]},
                               "checks": check, "fp16_passed": passed, "bf16_verified": circuit.bf16_verified,
                               "kernel_launches": kernel_launches(circuit, 512)}
            print(f"validate {name} (from {row['start']}, w={row['weight']}): early median "
                  f"{row['scores']['early_median']:.4f}, FP16 gates {gates}, BF16 {circuit.bf16_verified}", flush=True)

    from research.polar_opt.schatten_switch_blackwell import SCHATTEN_SWITCH_CIRCUITS, SchattenSwitchCircuit

    timed = {name: (SCHATTEN_SWITCH_CIRCUITS[name], torch.float16) for name in sorted(SCHATTEN_SWITCH_CIRCUITS)}
    timed.update({name: (SchattenSwitchCircuit.from_dict(v["circuit"]), torch.float16)
                  for name, v in validated.items() if v["fp16_passed"]})
    timing = latency(timed, shapes=((512, 512), (512, 1408), (1024, 4096), (4096, 4096)), iters=10, warmup=3)
    result = {
        "schema_version": 1, "mode": "early_srank_repolish", **git_info(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": torch.cuda.get_device_name(),
        "torch": torch.__version__, "command": [os.path.basename(__file__), *sys.argv[1:]],
        "settings": {"u_safe": U_FP16, "cap": CAP, "splits": [list(s) for s in EARLY_SPLITS],
                     "jobs": [list(j) for j in EARLY_JOBS], "evaluations": args.early_evaluations,
                     "cache_starts_per_split": args.early_cache_starts, "validate_per_split": args.early_validate,
                     "seed_artifact": str(args.seed_artifact), "fit_cache": str(args.fit_cache) if args.fit_cache else None, "fit_cache_status": cache_note,
                     "targets": "Cycle 1 momenta: min(max(start error, PS3_MEDIAN / 2), 0.97 max(PS3_55 error, "
                                "PS3_MEDIAN)); early spectra: weight * PS3_55 error",
                     "selection": "safe, Cycle 1 parity with PS3_55, early median < 0.9 x registered circuit",
                     "held_out": "real Nesterov momenta are never read here"},
        "early_spectra": [{"name": e["name"], "n": e["n"], "srank": e["srank"]} for e in early],
        "ps3_55_reference": {"cycle1_median": float(np.median(ps3_c1)), "cycle1_max": float(ps3_c1.max()),
                             "early_median": float(np.median(ps3_early)), "early_p90": float(np.quantile(ps3_early, 0.9)),
                             "early_max": float(ps3_early.max())},
        "starts": [{"tag": s["tag"], "split": list(s["split"]), "first": s["first"], "band_lo": s["band_lo"],
                    "band_error": s["band_error"], "scores": {k: v for k, v in s["scores"].items() if k != "early_errors"}}
                   for s in starts],
        "polish_records": [{k: v for k, v in r.items() if k not in ("seg1", "seg2")} for r in records],
        "validated": validated,
        "registered_kernel_launches": {name: kernel_launches(c, 512) for name, (c, _) in timed.items()},
        "latency_sanity": timing, "search_seconds": search_seconds, "seconds": time.perf_counter() - began,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_clean(result), indent=1, allow_nan=False) + "\n")
    print(f"wrote {args.output} in {result['seconds']:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=None,
                        help="output JSON path (defaults to results/schatten_switch_search_cycle2.json, or "
                             "results/schatten_switch_early_polish.json with --include-early-srank)")
    parser.add_argument("--fit-cache", type=Path, default=None,
                        help="JSON with the segment fits: loaded if it exists, updated after every family")
    parser.add_argument("--trajectories", type=Path, default=Path("artifacts/harvested_trajectories.pt"))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--starts", type=int, default=10)
    parser.add_argument("--ils-iterations", type=int, default=120)
    parser.add_argument("--ils-rounds", type=int, default=2)
    parser.add_argument("--skip-search", action="store_true", help="use the fit cache as it is")
    parser.add_argument("--fits-only", action="store_true")
    early = parser.add_argument_group("early-spectrum re-polish (opt-in; replaces the full search)")
    early.add_argument("--include-early-srank", action="store_true",
                       help="re-polish the 5+4, 5+3, 3+5, 3+4 circuits with synthetic near-rank-1 spectra "
                            "(stable rank 1.03-1.60) added to the polish targets, then validate on the GPU")
    early.add_argument("--seed-artifact", type=Path, default=Path("results/schatten_switch_search_cycle2.json"),
                       help="search JSON whose selected circuits seed the re-polish")
    early.add_argument("--early-cache-starts", type=int, default=2,
                       help="extra starts per split from the fit cache composites (needs --fit-cache)")
    early.add_argument("--early-evaluations", type=int, default=200)
    early.add_argument("--early-validate", type=int, default=2, help="GPU-validated candidates per split")
    args = parser.parse_args()
    try:
        args.output = _resolve_search_output(args.output, args.include_early_srank, args.seed_artifact)
    except ValueError as exc:
        parser.error(str(exc))
    if args.include_early_srank:
        if args.fit_cache and not args.fit_cache.exists():
            parser.error(f"--fit-cache {args.fit_cache} does not exist")
        run_early_search(args)
        return
    began = time.perf_counter()
    cache = json.loads(args.fit_cache.read_text()) if args.fit_cache and args.fit_cache.exists() else {}

    def save():
        if args.fit_cache:
            args.fit_cache.parent.mkdir(parents=True, exist_ok=True)
            args.fit_cache.write_text(json.dumps(cache))

    if not args.skip_search:
        for zero in (False, True):
            run_fits(cache, save, args.workers, args.starts, zero)
            run_improve(cache, save, args.workers, args.ils_iterations, args.ils_rounds, zero)
    if args.fits_only:
        return

    import torch
    from dataclasses import replace

    spectra = harvested_spectra(args.trajectories)
    ps3 = harvested_score(ps3_scalar, spectra, None)
    targets = 0.97 * np.maximum(np.array(ps3["errors"]), PS3_MEDIAN)
    print(f"PS3_55 scalar on harvested momenta: median {ps3['median']:.4f}, max {ps3['max']:.4f}", flush=True)
    grid = composites(cache, spectra, U_FP16, args.workers, SPLITS)  # the last CPU pool; the GPU starts below
    ranked = ranked_candidates(grid, spectra, targets, U_FP16)
    zero = zero_restart(cache, spectra, U_FP16)

    # Per number of squarings: the best candidate that passes the FP16 safety checks. Each
    # elementwise pass of the first row counts as 0.02 of parity (one more kernel launch).
    passes = {"pure": 0, "semi": 1, "general": 2}

    def build(record, suffix=""):
        k1, k2 = record["k1"], record["k2"]
        buffers = 1 + max(max(op) for op in record["seg1"]["topo"] + record["seg2"]["topo"])
        return compose(record["seg1"], record["seg2"], U_FP16, f"SS{buffers}_{k1}{k2}{suffix}", record["band_lo"],
                       record["band_error"])

    def entry(record, circuit, check):
        return {"circuit": circuit, "squarings": record["k1"] + record["k2"], "first_row": record["first"],
                "scalar": _strip(record), "segments": {"seg1": record["seg1"], "seg2": record["seg2"]}, "fp16": check}

    selected, attempts, pools = {}, [], {}
    for total in sorted({k1 + k2 for k1, k2 in ranked}, reverse=True):
        pools[total] = sorted((r for (k1, k2), rows in ranked.items() if k1 + k2 == total for r in rows),
                              key=lambda r: r["parity"] + 0.02 * passes[r["first"]])
        for record in pools[total][:10]:
            circuit = build(record)
            check = verify(circuit, "fp16")
            attempts.append({"name": circuit.name, "first_row": record["first"], "polished": record["polish"] is not None,
                             "parity": record["parity"], "fp16_passed": check["passed"],
                             "max_rank1_sigma": max(r["max_frobenius"] for r in check["rank1_sweeps"]),
                             "max_identity_plus_constant": check["identity_plus_constant_4096"]["max_sigma_bound"]})
            print(f"{total} squarings: {circuit.name} ({record['first']}, parity {record['parity']:.3f}) FP16 safety "
                  f"{check['passed']}", flush=True)
            if check["passed"]:
                selected[circuit.name] = entry(record, circuit, check)
                break

    # BF16. First the FP16 table itself; else another candidate with the same number of squarings
    # that passes both dtypes; else the selected segments joined with u = 1.15.
    for name, record in list(selected.items()):
        same = verify(record["circuit"], "bf16")
        record["bf16"] = same
        if same["passed"]:
            record["circuit"] = replace(record["circuit"], bf16_verified=True)
            print(f"{name}: BF16 passes with the FP16 table", flush=True)
            continue
        found = False
        for other in pools[record["squarings"]][:8]:
            circuit = build(other)
            suffix = "" if circuit.name != name else "B"
            circuit = build(other, suffix)
            if circuit.name in selected or circuit.table == record["circuit"].table:
                continue
            bf16 = verify(circuit, "bf16")
            fp16 = verify(circuit, "fp16") if bf16["passed"] else {"passed": False}
            attempts.append({"name": circuit.name, "first_row": other["first"], "polished": other["polish"] is not None,
                             "parity": other["parity"], "fp16_passed": fp16["passed"], "bf16_passed": bf16["passed"]})
            if bf16["passed"] and fp16["passed"]:
                selected[circuit.name] = {**entry(other, replace(circuit, bf16_verified=True), fp16), "bf16": bf16}
                print(f"{name}: BF16 fails; {circuit.name} ({other['first']}, parity {other['parity']:.3f}) passes "
                      f"FP16 and BF16", flush=True)
                found = True
                break
        if found:
            continue
        segs = record["segments"]
        wide = compose(segs["seg1"], segs["seg2"], U_BF16, name + "_U115", record["circuit"].lo, record["circuit"].error)
        facts = describe(wide, spectra)
        wide = replace(wide, error=facts["band_error"])
        fp16, bf16 = verify(wide, "fp16"), verify(wide, "bf16")
        print(f"{name}: BF16 fails; u = 1.15 variant FP16 {fp16['passed']}, BF16 {bf16['passed']}", flush=True)
        if fp16["passed"] and bf16["passed"]:
            selected[wide.name] = {"circuit": replace(wide, bf16_verified=True), "squarings": record["squarings"],
                                   "first_row": record["first_row"], "scalar": facts, "segments": segs,
                                   "fp16": fp16, "bf16": bf16}
        else:
            record["bf16_u115_variant"] = {"fp16_passed": fp16["passed"], "bf16_passed": bf16["passed"]}

    runs = {name: (record["circuit"], torch.float16) for name, record in selected.items()}
    runs.update({name + "@bf16": (record["circuit"], torch.bfloat16) for name, record in selected.items()
                 if record["circuit"].bf16_verified})
    tier = tier1(runs, args.trajectories)
    for name, record in selected.items():
        record["tier1"] = {suite: rows[name] for suite, rows in tier["summary"].items()}
        harvested = record["tier1"]["harvested_momentum"]
        record["parity_with_ps3_55"] = bool(
            record["tier1"]["all"]["diverged"] == 0
            and harvested["median_relative_frobenius_error"] <= PS3_MEDIAN
            and harvested["max_relative_frobenius_error"] <= PS3_MAX + 2e-3)
        if record["circuit"].bf16_verified:
            record["tier1_bf16"] = {suite: rows[name + "@bf16"] for suite, rows in tier["summary"].items()}
            if record["tier1_bf16"]["all"]["diverged"]:
                record["circuit"] = replace(record["circuit"], bf16_verified=False)
        record["kernel_launches"] = kernel_launches(record["circuit"], 512)
    timing = latency({name: run for name, run in runs.items() if "@" not in name})
    probe = gram_restart_probe(args.trajectories)

    result = {
        "schema_version": 1, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **git_info(),
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "settings": {"u_safe_fp16": U_FP16, "u_safe_bf16": U_BF16, "cap": CAP, "workers": min(args.workers, MAX_WORKERS),
                     "starts": args.starts, "ils_iterations": args.ils_iterations, "ils_rounds": args.ils_rounds,
                     "splits": [list(split) for split in SPLITS], "parity_targets": {"median": PS3_MEDIAN, "max": PS3_MAX},
                     "grammar": "B[dst] = a*B[src]^2 + b*B[add] + c*I on 2 or 3 FP16 buffers; src = previous dst",
                     "first_row": "pure: R^2; semi: R^2 + b R (one elementwise pass); general: + c I (two passes)",
                     "score": "exact-arithmetic sqrt(mean((f(sigma/sigma_hat) - 1)^2)) per harvested matrix, "
                              "sigma_hat = u_safe * runtime FP16 bound; the polish used this score",
                     "anchors": "PS3_55 segments seed the 5-squaring families (zhehangdu/polar-switch, MIT)"},
        "ps3_55_reference": {"squarings": 10, "flops": "(6α + 10)n³", "harvested_median": ps3["median"],
                             "harvested_max": ps3["max"], "slope_at_zero": float(ps3_scalar(1e-9) / 1e-9)},
        "families": {key: {k: v for k, v in family.items() if k != "rungs"} | {
            "best_by_lo": {lo: rows[0] for lo, rows in family["rungs"].items() if rows}}
            for key, family in cache.items()},
        "composite_grid": [_strip(row) | {"lo1": row["seg1"]["lo"], "E1": row["seg1"]["E"], "lo2": row["seg2"]["lo"],
                                          "E2": row["seg2"]["E"]} for row in grid],
        "best_scalar_per_split": {f"{k1}+{k2}": _strip(rows[0]) for (k1, k2), rows in sorted(ranked.items())},
        "selection_attempts": attempts,
        "zero_restart": zero,
        "selected": {name: {"circuit": record["circuit"].to_dict(),
                            "elementwise_passes": record["circuit"].elementwise_passes,
                            **{k: v for k, v in record.items() if k != "circuit"}}
                     for name, record in selected.items()},
        "default_variant": next((name for name, record in selected.items() if record["parity_with_ps3_55"]), None),
        "tier1": tier, "latency_sanity": timing, "negative_results": {"gram_space_restart": probe},
    }
    reference = tier["summary"]["harvested_momentum"]["ps3_55_blackwell_fp16"]
    result["summary"] = {"ps3_55_blackwell_fp16": {
        "squarings": 10, "flops": "(6α + 10)n³", "harvested_median": reference["median_relative_frobenius_error"],
        "harvested_max": reference["max_relative_frobenius_error"], "tier1_diverged": 0,
        "tier1_latency_ms": tier["summary"]["all"]["ps3_55_blackwell_fp16"]["median_case_latency_ms"]}}
    for name, record in selected.items():
        harvested = record["tier1"]["harvested_momentum"]
        result["summary"][name] = {
            "squarings": record["squarings"], "flops": record["circuit"].flops_formula(),
            "first_row": record["first_row"], "bf16_verified": record["circuit"].bf16_verified,
            "parity_with_ps3_55": record["parity_with_ps3_55"],
            "harvested_median": harvested["median_relative_frobenius_error"],
            "harvested_max": harvested["max_relative_frobenius_error"],
            "tier1_diverged": record["tier1"]["all"]["diverged"],
            "tier1_latency_ms": record["tier1"]["all"]["median_case_latency_ms"],
            "gpu_speedup_vs_ps3_55": {shape: rows[name]["speedup_gpu"] for shape, rows in timing["shapes"].items()}}
    result["seconds"] = time.perf_counter() - began
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_clean(result), indent=1, allow_nan=False) + "\n")
    print(f"wrote {args.output} in {result['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
