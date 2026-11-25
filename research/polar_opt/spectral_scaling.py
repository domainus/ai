"""Warm-started spectral-norm estimators for streaming Muon momentum matrices.

Muon divides each momentum matrix ``M_t`` by a scale ``s`` before it runs a
polynomial polar circuit. The circuit is designed for singular values in
``[ell, u]``. Frobenius scaling (``s = ||M||_F``) is always safe but compresses
the spectrum by ``||M||_F / sigma_max = sqrt(stable rank)``. Spectral scaling
(``s ~ sigma_max``) keeps the spectrum wide, so a shorter circuit suffices. The
risk is underestimation: if ``s < sigma_max`` the top singular value enters the
circuit at ``x = sigma_max / s > 1``.

Bounds used here (all for ``M`` of shape ``(..., m, n)``):

* Rigorous upper bounds on ``sigma_max``:
  ``||M||_F``; Hoelder ``sqrt(||M||_1 ||M||_inf)``; Schatten-4k norms
  ``||G^k||_F^{1/(2k)}`` from the Gram matrix ``G = M M^T``. The Schatten-8 bound
  (k = 2) is free inside a Gram circuit because the first quintic step already
  forms ``G^2``.
* Rigorous lower bounds: the largest row or column L2 norm, and every power or
  Krylov Rayleigh value ``||M w|| / ||w||``.
* Estimates (not bounds): ``gamma * lower`` with a calibrated margin ``gamma``.

Hybrid guard (task formula)::

    sigma_safe = min(||M||_F, max(sigma_cheap, gamma * sigma_power))

Extra power or block-Krylov work runs when one of these triggers fires:
a rigorous lower bound exceeds the power estimate, the Rayleigh value jumps
against the previous step, the relative residual ``||M^T u - sigma v|| / sigma``
is large, or the new vector has low overlap with the warm start.

Worst-case underestimation (documented failure modes)
-----------------------------------------------------
1. Orthogonal warm-start trap. If ``v_{t-1}`` is exactly orthogonal to the new
   top right singular vector ``v_1``, every power step stays in ``v_1^perp`` in
   exact arithmetic. The residual can be zero, so no residual test detects it.
   Rounding re-injects ``v_1`` only at rate ``eps * (sigma_1/sigma_2)^(2k)``.
2. With the task formula, the only remaining protection is ``sigma_cheap``.
   Because ``sum_j ||M e_j||^2 = ||M||_F^2 >= sigma_max^2``, the largest column
   norm is at least ``sigma_max / sqrt(n)``. The worst-case overshoot is
   therefore ``x_max = sigma_max / sigma_safe <= sqrt(min(m, n))``, for example
   32 for ``n = 1024``. A quintic circuit evaluated at ``x = 32`` overflows FP16
   after one step. ``spectral_counterexample`` builds a concrete instance: a
   spike ``sigma_1 = 3`` appears while the warm start sits exactly on the
   previous top vector ``v_2`` (``sigma_2 = 1.2``) over a flat bulk at 1. The
   residual is zero, the overlap is 1, the Rayleigh value does not jump, and
   every row/column norm stays below 1.2 because the spike energy spreads over
   ``n`` coordinates. No trigger fires and the overshoot is
   ``3 / (1.2 gamma) ~ 2.4``. The largest spike that stays hidden grows like
   ``sigma_2 sqrt(n / (2 ln n))``.
3. Fix (``certified`` mode): with any rigorous upper bound ``U``, use
   ``sigma_safe = max(sigma_safe, U / u_basin)``. Then
   ``x_max = sigma_max / sigma_safe <= U / sigma_safe <= u_basin`` always.
   The cost is conservatism ``sigma_safe / sigma_max <= max(gamma, U / (u sigma_max))``.
   With ``U = min(Frobenius, Hoelder)`` (the only bounds available without a
   Gram matrix) this can be large: 6.2x on the counterexample. The Gram path in
   ``circuits.WarmSpectralScaler`` uses the free Schatten-8 bound instead.
   It reaches 0.96x there and a median of 1.06x on a drifting momentum
   stream. Spiky spectra make ``U`` tight, and flat spectra make power
   iteration accurate, so the two bounds cover each other.

Stability basin
---------------
A circuit designed on ``[ell, u]`` tolerates overshoot up to its basin edge
``u* >= u``. Past ``u*`` the leading ``x^5`` term dominates and composition
grows double-exponentially. ``circuit_search.stability_basin`` measures ``u*``
for each circuit. Underestimates are harmless while
``sigma_max / sigma_safe <= u*``. Certified mode keeps this ratio at or below
``u_basin`` by construction.

The polynomial circuits depend only on the ratios ``sigma_i / s``. Rounding
inside an FP16/BF16 Gram matrix makes the Schatten bound inexact at the level
``O(eps)``. ``gram_schatten_upper`` multiplies by ``(1 + 2 eps)`` to cover that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = [
    "GuardConfig",
    "GuardResult",
    "PowerResult",
    "frobenius_upper",
    "hoelder_upper",
    "gram_schatten_upper",
    "rowcol_lower",
    "power_iteration",
    "block_krylov_lower",
    "hybrid_guard_estimate",
    "calibrate_gamma",
    "max_overshoot",
    "spectral_counterexample",
    "REASON_LOWER_INCONSISTENT",
    "REASON_RAYLEIGH_JUMP",
    "REASON_RESIDUAL",
    "REASON_LOW_OVERLAP",
    "REASON_NO_WARM_START",
]

REASON_LOWER_INCONSISTENT = 1
REASON_RAYLEIGH_JUMP = 2
REASON_RESIDUAL = 4
REASON_LOW_OVERLAP = 8
REASON_NO_WARM_START = 16

_TINY = 1e-30


def _norm(x: torch.Tensor, dim=-1, keepdim=False) -> torch.Tensor:
    return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / _norm(v, keepdim=True).clamp_min(_TINY)


def _mv(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (M @ v.unsqueeze(-1)).squeeze(-1)


def _mtv(M: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return (M.mT @ u.unsqueeze(-1)).squeeze(-1)


# --------------------------------------------------------------------------
# Rigorous bounds
# --------------------------------------------------------------------------


def frobenius_upper(M: torch.Tensor) -> torch.Tensor:
    """``||M||_F >= sigma_max``. Shape ``(...)``."""
    return _norm(M.float(), dim=(-2, -1))


def hoelder_upper(M: torch.Tensor) -> torch.Tensor:
    """``sqrt(||M||_1 ||M||_inf) >= sigma_max`` (max abs column/row sum)."""
    A = M.float().abs()
    return torch.sqrt(A.sum(-2).amax(-1) * A.sum(-1).amax(-1))


def gram_schatten_upper(Gk: torch.Tensor, k: int, eps: float = 0.0) -> torch.Tensor:
    """Upper bound ``||G^k||_F^{1/(2k)} >= sigma_max(M)`` where ``G = M M^T``.

    This is the Schatten-``4k`` norm of ``M``. ``eps`` is the unit roundoff of
    the dtype that stored ``G^k``; the result is inflated by ``(1 + 2 eps)``.
    """
    s = _norm(Gk.float(), dim=(-2, -1)).pow(1.0 / (2 * k))
    return s * (1.0 + 2.0 * eps)


def rowcol_lower(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Largest row/column L2 norm (``<= sigma_max``) and a probe vector.

    The probe lies in the right singular space (length ``n``). For a column
    winner ``j`` it is ``e_j``. For a row winner ``i`` it is ``M^T e_i``
    normalized, which satisfies ``||M w|| >= ||M^T e_i||``.
    """
    Mf = M.float()
    col = _norm(Mf, dim=-2)  # (..., n)
    row = _norm(Mf, dim=-1)  # (..., m)
    cmax, cidx = col.max(-1)
    rmax, ridx = row.max(-1)
    n = M.shape[-1]
    e_col = torch.nn.functional.one_hot(cidx, n).to(Mf.dtype)
    row_vec = torch.gather(Mf, -2, ridx[..., None, None].expand(*ridx.shape, 1, n)).squeeze(-2)
    probe = torch.where((cmax >= rmax)[..., None], e_col, _unit(row_vec))
    return torch.maximum(cmax, rmax), probe


# --------------------------------------------------------------------------
# Warm-started power iteration and block Krylov
# --------------------------------------------------------------------------


@dataclass
class PowerResult:
    sigma: torch.Tensor  # rigorous lower bound after the last step, (...)
    v: torch.Tensor  # new right warm-start vector, (..., n)
    residual: torch.Tensor  # relative residual ||M^T u - s v|| / s of the last step
    first_rayleigh: torch.Tensor  # ||M v_prev||: the 0-step Rayleigh value
    overlap: torch.Tensor  # |<v_prev, v_new>|
    matvecs: int


def power_iteration(M: torch.Tensor, v: torch.Tensor, steps: int = 1) -> PowerResult:
    """Warm-started power iteration on ``M^T M``; each step costs 2 matvecs.

    For a unit ``v`` the values ``s1 = ||M v||`` and ``s2 = ||M^T M v|| / ||M v||``
    are both lower bounds on ``sigma_max`` and ``s2 >= s1``. The residual of the
    singular pair ``(s1, u = Mv/s1, v)`` is ``sqrt(s2^2 - s1^2)``, so it is free.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    Mf = M.float()
    v0 = _unit(v.float())
    vk = v0
    first = None
    s1 = s2 = None
    for _ in range(steps):
        w = _mv(Mf, vk)
        s1 = _norm(w)
        if first is None:
            first = s1
        z = _mtv(Mf, w)
        zn = _norm(z)
        s2 = zn / s1.clamp_min(_TINY)
        vk = z / zn.clamp_min(_TINY)[..., None]
    res = torch.sqrt((s2 * s2 - s1 * s1).clamp_min(0.0)) / s1.clamp_min(_TINY)
    return PowerResult(
        sigma=torch.maximum(s1, s2),
        v=vk,
        residual=res,
        first_rayleigh=first,
        overlap=(v0 * vk).sum(-1).abs(),
        matvecs=2 * steps,
    )


def block_krylov_lower(
    M: torch.Tensor, V0: torch.Tensor, steps: int = 2
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Block Krylov Rayleigh-Ritz lower bound on ``sigma_max``.

    ``V0`` has shape ``(..., n, b)`` (warm start plus probes). The basis is
    ``[V0, A V0, ..., A^steps V0]`` with ``A = M^T M``, orthonormalized after
    every block. The bound is ``sigma_max(M Q) <= sigma_max(M)`` for the
    orthonormal ``Q``. Returns ``(sigma, top right Ritz vector, matvecs)``.
    """
    Mf = M.float()
    blocks = [torch.linalg.qr(V0.float()).Q]
    matvecs = 0
    for _ in range(steps):
        W = Mf.mT @ (Mf @ blocks[-1])
        matvecs += 2 * W.shape[-1]
        blocks.append(torch.linalg.qr(W).Q)
    Q = torch.linalg.qr(torch.cat(blocks, dim=-1)).Q
    B = Mf @ Q
    matvecs += Q.shape[-1]
    evals, evecs = torch.linalg.eigh(B.mT @ B)  # ascending
    sigma = evals[..., -1].clamp_min(0.0).sqrt()
    v = _unit((Q @ evecs[..., -1:]).squeeze(-1))
    return sigma, v, matvecs


# --------------------------------------------------------------------------
# Hybrid guard
# --------------------------------------------------------------------------


@dataclass
class GuardConfig:
    """Parameters of :func:`hybrid_guard_estimate`.

    ``u_basin=None`` gives the literal task formula. A float enables certified
    mode: ``sigma_safe >= U / u_basin`` with ``U = min(Frobenius, Hoelder)``.
    """

    gamma: float = 1.05
    power_steps: int = 1
    jump_tol: float = math.log(1.5)  # |log(sigma_t / sigma_{t-1})|
    residual_tol: float = 0.25
    overlap_tol: float = 0.5
    probes: int = 4
    fallback_steps: int = 2
    u_basin: float | None = None


@dataclass
class GuardResult:
    sigma_safe: torch.Tensor
    sigma_power: torch.Tensor  # rigorous lower bound (after any fallback)
    sigma_cheap: torch.Tensor  # rigorous lower bound (row/col norms)
    upper: torch.Tensor  # rigorous upper bound min(F, Hoelder)
    v: torch.Tensor  # new warm start
    reasons: torch.Tensor  # int bitmask per batch element
    matvecs: int
    extra: dict = field(default_factory=dict)

    @property
    def triggered(self) -> torch.Tensor:
        return self.reasons != 0


def hybrid_guard_estimate(
    M: torch.Tensor,
    v_prev: torch.Tensor | None,
    sigma_prev: torch.Tensor | None = None,
    cfg: GuardConfig | None = None,
    generator: torch.Generator | None = None,
) -> GuardResult:
    """Guarded spectral estimate for one streaming step.

    ``sigma_safe = min(||M||_F, max(sigma_cheap, gamma * sigma_power))``, where
    ``sigma_power`` comes from a warm-started power iteration. When a trigger
    fires, a block-Krylov pass with the warm vector, the row/column probe, and
    random Gaussian probes replaces ``sigma_power`` (if larger).
    """
    cfg = cfg or GuardConfig()
    Mf = M.float()
    batch, n = M.shape[:-2], M.shape[-1]
    fro = frobenius_upper(Mf)
    upper = torch.minimum(fro, hoelder_upper(Mf))
    cheap, probe = rowcol_lower(Mf)

    reasons = torch.zeros(batch, dtype=torch.int64, device=M.device)
    if v_prev is None:
        v_prev = probe
        reasons |= REASON_NO_WARM_START
    pw = power_iteration(Mf, v_prev, cfg.power_steps)
    matvecs = pw.matvecs
    sigma_pow = pw.sigma

    reasons |= torch.where(cheap > sigma_pow * (1 + 1e-6), REASON_LOWER_INCONSISTENT, 0)
    if sigma_prev is not None:
        jump = (sigma_pow.clamp_min(_TINY) / sigma_prev.float().clamp_min(_TINY)).log().abs()
        reasons |= torch.where(jump > cfg.jump_tol, REASON_RAYLEIGH_JUMP, 0)
    reasons |= torch.where(pw.residual > cfg.residual_tol, REASON_RESIDUAL, 0)
    reasons |= torch.where(pw.overlap < cfg.overlap_tol, REASON_LOW_OVERLAP, 0)
    # A zero matrix has nothing to estimate.
    reasons = torch.where(fro > 0, reasons, torch.zeros_like(reasons))

    v_new = pw.v
    if bool((reasons != 0).any()):
        cols = [pw.v, probe]
        k_rand = max(cfg.probes - 2, 0)
        if k_rand:
            g = torch.randn(*batch, n, k_rand, generator=generator, dtype=Mf.dtype, device="cpu")
            cols.append(g.to(M.device))
        V0 = torch.cat([c[..., None] if c.dim() == pw.v.dim() else c for c in cols], dim=-1)
        s_k, v_k, mv = block_krylov_lower(Mf, V0, cfg.fallback_steps)
        matvecs += mv
        use = (reasons != 0) & (s_k > sigma_pow)
        sigma_pow = torch.where(use, s_k, sigma_pow)
        v_new = torch.where(use[..., None], v_k, v_new)

    est = torch.maximum(cheap, cfg.gamma * sigma_pow)
    sigma_safe = torch.minimum(fro, est)
    if cfg.u_basin is not None:
        sigma_safe = torch.minimum(upper, torch.maximum(sigma_safe, upper / cfg.u_basin))
    return GuardResult(
        sigma_safe=sigma_safe,
        sigma_power=sigma_pow,
        sigma_cheap=cheap,
        upper=upper,
        v=v_new,
        reasons=reasons,
        matvecs=matvecs,
    )


# --------------------------------------------------------------------------
# Calibration and diagnostics
# --------------------------------------------------------------------------


def calibrate_gamma(ratios, quantile: float = 1.0, margin: float = 1.02) -> float:
    """Pick ``gamma`` from observed ratios ``sigma_true / sigma_power`` (>= 1).

    ``quantile=1.0`` uses the worst observed ratio. The margin covers unseen
    drift. The result is only an empirical calibration, not a bound.
    """
    r = torch.as_tensor(ratios, dtype=torch.float64).flatten()
    return float(torch.quantile(r, quantile) * margin)


def max_overshoot(sigma_true: torch.Tensor, sigma_safe: torch.Tensor) -> float:
    """Largest ``x_max = sigma_true / sigma_safe`` seen by the circuit."""
    return float((sigma_true / sigma_safe.clamp_min(_TINY)).max())


def spectral_counterexample(
    n: int = 512,
    spike: float = 3.0,
    second: float = 1.2,
    bulk: float = 1.0,
    dtype=torch.float64,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Instance where the task-formula guard underestimates ``sigma_max`` silently.

    Returns ``(M, v_trap, sigma_max)`` with
    ``M = U diag(spike, second, bulk, ..., bulk) V^T`` (square ``n x n``) and
    ``v_trap = V[:, 1]``, the top vector before the spike appeared. Power
    iteration from ``v_trap`` returns ``second`` exactly (zero residual,
    overlap 1). The row/column norms stay below ``second`` while
    ``(spike^2 - bulk^2) * 2 ln(n) / n < second^2 - bulk^2``.
    """
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=dtype)).Q
    V = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=dtype)).Q
    s = torch.full((n,), bulk, dtype=dtype)
    s[0], s[1] = spike, second
    M = (U * s) @ V.mT
    return M, V[:, 1].clone(), float(spike)
