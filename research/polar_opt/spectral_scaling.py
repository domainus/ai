"""Warm-started spectral-norm estimates and guards for streaming Muon momentum.

A polar circuit maps each singular value ``x = sigma_i / sigma_hat`` through a
scalar function ``f``. Frobenius scaling (``sigma_hat = ||M||_F``) is always safe
but wastes the interval: real Muon momenta have ``||M||_F / sigma_max ~ 3.2``.
Spectral scaling with ``sigma_hat ~ sigma_max`` shrinks the circuit's work, but
a power-iteration estimate is a *lower* bound on ``sigma_max``. This module
combines both kinds of information:

* rigorous lower bounds: max row/column L2 norm, ``||M v||`` for unit ``v``,
  Ritz values, ``max diag(A)`` for a Gram matrix ``A``;
* rigorous upper bounds: ``||M||_F``, ``sqrt(||M||_1 ||M||_inf)``, and for a
  Gram matrix ``A = W W^T`` also ``||A||_F`` (Schatten-4), ``||A||_inf`` and
  ``||A^2||_F^(1/2)`` (Schatten-8);
* a warm-started power or block-Krylov estimate ``sigma_p`` (lower bound),
  inflated by a calibrated margin ``gamma``.

The guard is ``sigma_safe = min(U, max(L, gamma * sigma_p))``. Taking ``min``
with a rigorous upper bound ``U`` and ``max`` with a rigorous lower bound ``L``
never moves the estimate away from ``sigma_max``. Only ``gamma * sigma_p`` can
underestimate. Triggers detect the cases where it does (Rayleigh jumps, a warm
vector that is not an eigenvector, a jump in the ``U / sigma_p`` ratio, or a
lower bound above the estimate) and escalate to a block-Krylov solve with
random probes.

No power-type estimate is a rigorous upper bound: a spike orthogonal to every
probe is invisible in exact arithmetic. The remaining risk is covered by the
circuit's stability basin. If ``sigma_max / sigma_safe <= u_basin`` the circuit
output stays in ``[0, cap]``. ``stability_basin`` measures ``u_basin`` in FP64
and under scalar FP16/BF16 rounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import Tensor

_TINY = 1e-30

# Calibrated on artifacts/harvested_trajectories.pt (45 consecutive Muon EMA
# pairs, 3 layers x 5 matrices x early/mid/late) by
# ``calibrate_gamma(..., margin=0.02)``: the warm vector comes from the
# estimator's own cold start on the previous matrix. Worst sigma_max / sigma_p
# was 1.1186 (1 step) and 1.0995 (2 steps); medians 1.035 and 1.0076. Keys are
# warm power steps. circuit_search.py recomputes and reports these values.
CALIBRATED_GAMMA = {1: 1.141, 2: 1.122}
# Margin after a block-Krylov solve (cold start or escalation), whose top Ritz
# value is within 1e-3 of lambda_max on every harvested matrix.
KRYLOV_GAMMA = 1.02


# --------------------------------------------------------------------------------------
# Rigorous bounds
# --------------------------------------------------------------------------------------
def matrix_bounds(M: Tensor) -> tuple[Tensor, Tensor]:
    """Return rigorous ``(lower, upper)`` bounds on ``sigma_max`` of ``(..., m, n)``.

    lower = max(max row L2 norm, max column L2 norm)
    upper = min(||M||_F, sqrt(||M||_1 * ||M||_inf))
    """
    W = M.float()
    lower = torch.maximum(W.norm(dim=-1).amax(-1), W.norm(dim=-2).amax(-1))
    one = W.abs().sum(-2).amax(-1)
    inf = W.abs().sum(-1).amax(-1)
    upper = torch.minimum(W.norm(dim=(-2, -1)), (one * inf).sqrt())
    return lower, upper


def gram_bounds(A: Tensor, A2: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Return rigorous ``(lower, upper)`` bounds on ``lambda_max`` of PSD ``A = W W^T``.

    lower = max diag(A)
    upper = min(tr A, ||A||_F, ||A^2||_F^(1/2) if ``A2`` is given else ||A||_inf)

    ``||A||_F = (sum sigma^4)^(1/2)`` is the Schatten-4 norm of ``W`` squared. It
    is free in a Gram-form circuit and much tighter than ``||W||_F^2`` for the
    fast-decaying spectra of Muon momenta.
    """
    A = A.float()
    lower = A.diagonal(dim1=-2, dim2=-1).amax(-1)
    upper = torch.minimum(A.diagonal(dim1=-2, dim2=-1).sum(-1), A.norm(dim=(-2, -1)))
    if A2 is not None:
        upper = torch.minimum(upper, A2.float().norm(dim=(-2, -1)).sqrt())
    else:  # ||A||_inf costs a full pass; Schatten-8 is almost always tighter
        upper = torch.minimum(upper, A.abs().sum(-1).amax(-1))
    return lower, upper


# --------------------------------------------------------------------------------------
# Operators: everything works on a PSD operator A of the small side.
# --------------------------------------------------------------------------------------
class _GramOperator:
    """``A v`` for ``A = W W^T`` (``gram=False``) or a given Gram matrix (``gram=True``)."""

    def __init__(self, M: Tensor, gram: bool):
        if gram:
            if M.shape[-1] != M.shape[-2]:
                raise ValueError("A Gram matrix must be square")
            self.A, self.W = M.float(), None
        else:
            W = M.mT if M.shape[-2] > M.shape[-1] else M
            self.A, self.W = None, W.float()
        self.n = M.shape[-1] if gram else min(M.shape[-2:])
        self.batch = M.shape[:-2]
        self.device = M.device
        self.cost = M.shape[-1] ** 2 if gram else 2 * M.shape[-1] * M.shape[-2]
        self.matvecs = 0

    def __call__(self, V: Tensor) -> Tensor:
        """Apply A to ``V`` of shape (..., n, b)."""
        self.matvecs += V.shape[-1] * (1 if self.A is not None else 2)
        if self.A is not None:
            return self.A @ V
        return self.W @ (self.W.mT @ V)

    def bounds(self, A2: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Rigorous (lower, upper) on lambda_max = sigma_max^2."""
        if self.A is not None:
            return gram_bounds(self.A, A2)
        lower, upper = matrix_bounds(self.W)
        return lower.square(), upper.square()

    def random(self, b: int, generator: torch.Generator | None) -> Tensor:
        return torch.randn(*self.batch, self.n, b, device=self.device, generator=generator)


def _unit(v: Tensor) -> Tensor:
    return v / v.norm(dim=-2, keepdim=True).clamp_min(_TINY)


def power_iteration(op: Callable[[Tensor], Tensor], v: Tensor, steps: int):
    """Run ``steps`` applications of PSD ``A`` from unit vectors ``v`` of shape (..., n, 1).

    Returns ``(lam_low, v_new, align, residual)`` where ``lam_low = ||A v_last||``
    is a rigorous lower bound on ``lambda_max``, ``align = |<v, A v>| / ||A v||`` at
    the first step (1 means ``v`` was an eigenvector) and ``residual`` is the
    relative eigen-residual ``||A v - theta v|| / ||A v||`` at the last step.
    """
    if steps < 1:
        raise ValueError("Need at least one power step")
    v = _unit(v)
    align = None
    for _ in range(steps):
        w = op(v)
        lam = w.norm(dim=-2, keepdim=True)
        theta = (v * w).sum(-2, keepdim=True)
        if align is None:
            align = (theta.abs() / lam.clamp_min(_TINY)).squeeze(-1).squeeze(-1)
        residual = ((w - theta * v).norm(dim=-2, keepdim=True) / lam.clamp_min(_TINY))
        v = w / lam.clamp_min(_TINY)
    return lam.squeeze(-1).squeeze(-1), v, align, residual.squeeze(-1).squeeze(-1)


def block_krylov(op: Callable[[Tensor], Tensor], V: Tensor, depth: int):
    """Rayleigh-Ritz on the block Krylov space ``[V, A V, ..., A^depth V]``.

    ``V`` is (..., n, b): typically the warm vector plus random probes. Returns
    ``(lam_low, v_top)``. The top Ritz value is a rigorous lower bound on
    ``lambda_max`` in exact arithmetic (Cauchy interlacing). A random probe has
    overlap ~1/sqrt(n) with any fixed direction, so a hidden spike that is
    ``r`` times larger is amplified by ``r^(2 depth)`` and found quickly.
    """
    blocks = [V]
    Q = torch.linalg.qr(V.double())[0].to(V.dtype)
    for _ in range(depth):
        Q = torch.linalg.qr(op(Q).double())[0].to(V.dtype)
        blocks.append(Q)
    K = torch.linalg.qr(torch.cat(blocks, dim=-1).double())[0]
    AK = op(K.to(V.dtype)).double()
    T = K.mT @ AK
    T_sym = (T + T.mT) / 2
    if not torch.isfinite(T_sym).all():
        return power_iteration(op, V[..., :1], 1)[:2]
    try:
        theta, Y = torch.linalg.eigh(T_sym)
    except torch.linalg.LinAlgError:
        return power_iteration(op, V[..., :1], max(1, depth))[:2]
    v = (K @ Y[..., -1:]).to(V.dtype)
    return theta[..., -1].clamp_min(0).to(V.dtype), _unit(v)


# --------------------------------------------------------------------------------------
# Warm-started streaming estimator with a guard
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EstimatorConfig:
    """Knobs for :class:`WarmSpectralEstimator`.

    ``steps``: warm power steps per call (1 or 2). ``gamma``: overshoot margin
    (default ``CALIBRATED_GAMMA[steps]``). ``probes``: random probe columns
    added to every call (block hybrid; 0 = pure power). Triggers: relative
    Rayleigh jump above ``jump_tol``, first-step alignment below ``align_tol``,
    ``U / sigma_p`` growth above ``ratio_tol``, or a rigorous lower bound above
    ``gamma * sigma_p``. A trigger escalates to block Krylov with
    ``escalate_probes`` random columns and depth ``escalate_depth``.

    ``defer_escalation`` (default) avoids a host sync: a triggered matrix uses
    the rigorous upper bound ``U`` for this call (always safe), and the Krylov
    refresh of the warm vector runs at the start of the next call, when the
    trigger flags are long computed. With ``False`` the refresh runs at once.
    """

    steps: int = 1
    gamma: float | None = None
    probes: int = 0
    jump_tol: float = 0.25
    align_tol: float = 0.9
    ratio_tol: float = 0.25
    escalate_probes: int = 4
    escalate_depth: int = 4
    cold_probes: int = 4
    cold_depth: int = 8
    krylov_gamma: float = KRYLOV_GAMMA
    defer_escalation: bool = True

    @property
    def margin(self) -> float:
        if self.gamma is not None:
            return self.gamma
        return CALIBRATED_GAMMA.get(self.steps, CALIBRATED_GAMMA[2])


@dataclass
class SpectralEstimate:
    """Per-matrix result. ``sigma`` is the value to divide by; all values are in sigma units."""

    sigma: Tensor
    lower: Tensor
    upper: Tensor
    power: Tensor
    escalated: Tensor
    certified: Tensor
    triggers: dict = field(default_factory=dict)
    matvecs: int = 0
    flops: int = 0

    def certified_within(self, basin: float) -> Tensor:
        """True where ``sigma_max / sigma <= basin`` is proved by the rigorous upper bound."""
        return self.upper <= basin * self.sigma


class WarmSpectralEstimator:
    """Streaming spectral-norm estimate for one parameter's momentum ``M_t``.

    The state is the previous top vector on the small side and the previous
    ``sigma_p`` and ``U / sigma_p`` ratio. ``estimate(M)`` takes the matrix
    ``(..., m, n)`` or, with ``gram=True``, its Gram matrix ``A = W W^T`` on the
    small side (then each application costs ``n^2`` instead of ``2 m n``).
    """

    def __init__(self, config: EstimatorConfig | None = None, seed: int = 0):
        self.config = config or EstimatorConfig()
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.vector: Tensor | None = None
        self.sigma_prev: Tensor | None = None
        self.ratio_prev: Tensor | None = None
        self._pending: list = []  # queued (host flags, CUDA event or None)
        self._generator: torch.Generator | None = None
        self._key = None

    def _gen(self, device) -> torch.Generator:
        if self._generator is None or self._generator.device != torch.device(device):
            self._generator = torch.Generator(device=device).manual_seed(self.seed)
        return self._generator

    def _stash_flags(self, flags: Tensor) -> None:
        """Queue a non-blocking host copy of trigger flags (read by a later call)."""
        if flags.is_cuda:
            host = torch.empty(flags.shape, dtype=torch.bool, pin_memory=True)
            host.copy_(flags, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        else:
            host, event = flags.clone(), None
        self._pending.append((host, event))

    def _ready_flags(self) -> Tensor | None:
        """OR of every queued flag set whose copy has landed; never blocks."""
        ready = None
        while self._pending and (self._pending[0][1] is None or self._pending[0][1].query()):
            host, _ = self._pending.pop(0)
            ready = host if ready is None else ready | host
        return ready

    @torch.no_grad()
    def estimate(self, M: Tensor, *, gram: bool = False, A2: Tensor | None = None) -> SpectralEstimate:
        cfg = self.config
        op = _GramOperator(M, gram)
        gen = self._gen(M.device)
        key = (tuple(op.batch), op.n, str(M.device))
        lam_lo, lam_hi = op.bounds(A2)
        deferred = None
        if self.vector is None or self._key != key:
            V = torch.cat([op.random(1, gen), op.random(cfg.cold_probes, gen)], dim=-1)
            lam_p, v = block_krylov(op, V, cfg.cold_depth)
            escalated = torch.ones_like(lam_p, dtype=torch.bool)
            triggers = {"cold_start": escalated.clone()}
        elif (pending := self._ready_flags()) is not None and bool(pending.any()):
            # Refresh requested by an earlier call's triggers, read without a host sync.
            V = torch.cat([self.vector, op.random(cfg.escalate_probes, gen)], dim=-1)
            lam_p, v = block_krylov(op, V, cfg.escalate_depth)
            escalated = pending.to(lam_lo.device)
            lam_w, v_w, align_w, _ = power_iteration(op, self.vector, cfg.steps)
            lam_p = torch.where(escalated, torch.maximum(lam_p, lam_w), lam_w)
            v = torch.where(escalated[..., None, None], v, v_w)
            sigma_w = lam_w.sqrt()
            ratio_w = lam_hi.sqrt() / sigma_w.clamp_min(_TINY)
            triggers = {
                "deferred_refresh": escalated.clone(),
                "jump": (~escalated) & ((sigma_w / self.sigma_prev.clamp_min(_TINY) - 1).abs() > cfg.jump_tol),
                "misaligned": (~escalated) & (align_w < cfg.align_tol),
                "ratio_growth": (~escalated) & (ratio_w > (1 + cfg.ratio_tol) * self.ratio_prev),
                "lower_bound": (~escalated) & (lam_lo.sqrt() > cfg.margin * sigma_w),
            }
            hit_w = torch.zeros_like(escalated)
            for k_flag in ("jump", "misaligned", "ratio_growth", "lower_bound"):
                hit_w |= triggers[k_flag]
            if cfg.defer_escalation:
                deferred = hit_w
            elif hit_w.any():
                escalated = escalated | hit_w
                V2 = torch.cat([v, op.random(cfg.escalate_probes, gen)], dim=-1)
                lam_k2, v_k2 = block_krylov(op, V2, cfg.escalate_depth)
                lam_p = torch.where(hit_w, torch.maximum(lam_k2, lam_p), lam_p)
                v = torch.where(hit_w[..., None, None], v_k2, v)
        else:
            v0 = self.vector
            if cfg.probes:
                lam_p, v = block_krylov(op, torch.cat([v0, op.random(cfg.probes, gen)], -1), cfg.steps)
                _, _, align, _ = power_iteration(op, v0, 1)
            else:
                lam_p, v, align, _ = power_iteration(op, v0, cfg.steps)
            sigma_p = lam_p.sqrt()
            ratio = lam_hi.sqrt() / sigma_p.clamp_min(_TINY)
            g = cfg.margin
            triggers = {
                "jump": (sigma_p / self.sigma_prev.clamp_min(_TINY) - 1).abs() > cfg.jump_tol,
                "misaligned": align < cfg.align_tol,
                "ratio_growth": ratio > (1 + cfg.ratio_tol) * self.ratio_prev,
                "lower_bound": lam_lo.sqrt() > g * sigma_p,
            }
            hit = torch.zeros_like(sigma_p, dtype=torch.bool)
            for flag in triggers.values():
                hit |= flag
            if cfg.defer_escalation:
                deferred = hit
                escalated = torch.zeros_like(hit)
            else:
                escalated = hit
                if escalated.any():
                    V = torch.cat([v, op.random(cfg.escalate_probes, gen)], dim=-1)
                    lam_k, v_k = block_krylov(op, V, cfg.escalate_depth)
                    lam_p = torch.where(escalated, torch.maximum(lam_k, lam_p), lam_p)
                    v = torch.where(escalated[..., None, None], v_k, v)
        sigma_p = lam_p.sqrt()
        lower, upper = lam_lo.sqrt(), lam_hi.sqrt()
        gamma = torch.where(escalated, cfg.krylov_gamma, cfg.margin)
        sigma = torch.minimum(upper, torch.maximum(lower, gamma * sigma_p))
        if deferred is not None:
            sigma = torch.where(deferred, upper, sigma)
        if deferred is not None:
            self._stash_flags(deferred)
        self.vector = v
        self.sigma_prev = sigma_p
        self.ratio_prev = upper / sigma_p.clamp_min(_TINY)
        self._key = key
        return SpectralEstimate(
            sigma=sigma, lower=torch.maximum(lower, sigma_p), upper=upper, power=sigma_p,
            escalated=escalated if deferred is None else escalated | deferred,
            certified=upper <= sigma, triggers=triggers,
            matvecs=op.matvecs, flops=op.matvecs * op.cost,
        )


# --------------------------------------------------------------------------------------
# Calibration of gamma on real trajectories
# --------------------------------------------------------------------------------------
def _wide_fro(M: Tensor) -> Tensor:
    W = M.float()
    W = W.mT if W.shape[-2] > W.shape[-1] else W
    return W / W.norm(dim=(-2, -1), keepdim=True)


@torch.no_grad()
def calibrate_gamma(pairs, steps: int, *, margin: float = 0.02, gram: bool = True,
                    device="cpu", seed: int = 0) -> dict:
    """Measure ``sigma_max / sigma_p`` on consecutive momentum pairs.

    For each ``(previous, current)`` pair a fresh estimator cold-starts on
    ``previous`` and then makes one warm call with ``steps`` power steps on
    ``current`` (no triggers, ``gamma = 1``). ``gamma = max ratio * (1 + margin)``.
    The sample is small (45 pairs), so the margin and the basin still matter.
    """
    ratios = []
    for previous, current in pairs:
        est = WarmSpectralEstimator(
            EstimatorConfig(steps=steps, gamma=1.0, jump_tol=float("inf"), align_tol=0.0,
                            ratio_tol=float("inf")), seed=seed)
        mats = []
        for M in (previous, current):
            W = _wide_fro(M.to(device))
            mats.append(W @ W.mT if gram else W)
        est.estimate(mats[0], gram=gram)
        out = est.estimate(mats[1], gram=gram)
        true = torch.linalg.matrix_norm(_wide_fro(current).double(), ord=2).item()
        ratios.append(true / out.power.double().item())
    r = np.array(ratios)
    return {
        "steps": steps, "pairs": len(r), "margin": margin,
        "gamma": float(r.max() * (1 + margin)),
        "ratio": {q: float(np.quantile(r, p)) for q, p in
                  (("min", 0), ("p50", .5), ("p90", .9), ("max", 1))},
    }


# --------------------------------------------------------------------------------------
# Stability basin of a scalar circuit map
# --------------------------------------------------------------------------------------
def stability_basin(f: Callable[[np.ndarray], np.ndarray], *, cap: float = 1.25,
                    x_max: float = 2.0, num: int = 40001) -> float:
    """Largest ``u >= 1`` with ``0 <= f(x) <= cap`` for every ``x`` in ``[0, u]``.

    ``f`` maps scaled singular values to output singular values (exact for any
    odd matrix polynomial in exact arithmetic, since it acts on each singular
    value independently). Returns 0 when ``f`` already leaves ``[0, cap]`` in
    ``[0, 1]``. The grid spacing is ``x_max / (num - 1)``.
    """
    x = np.linspace(0.0, x_max, num)
    with np.errstate(all="ignore"):
        y = np.asarray(f(x), dtype=np.float64)
    bad = ~np.isfinite(y) | (y < -1e-12) | (y > cap)
    if not bad.any():
        return float(x_max)
    first = int(np.argmax(bad))
    return 0.0 if x[first] <= 1.0 else float(x[first - 1])


def basin_report(f: Callable[[np.ndarray], np.ndarray], *, lo: float, cap: float = 1.25) -> dict:
    """Characterize ``f`` on ``[0, 1 + delta]``.

    Reports the max error on ``[lo, 1]``, the range on ``[0, 1]``, values at the
    overshoots 1.02/1.05/1.10/1.15 and ``basin_u``. A scaling with
    ``sigma_max / sigma_hat <= basin_u`` can never diverge.
    """
    x = np.geomspace(lo, 1.0, 4001)
    with np.errstate(all="ignore"):
        err = float(np.nanmax(np.abs(np.asarray(f(x)) - 1)))
        unit = np.asarray(f(np.linspace(0, 1, 4001)))
        over = {f"{u:.2f}": float(np.asarray(f(np.array([u])))[0]) for u in (1.02, 1.05, 1.10, 1.15)}
    return {
        "lo": lo, "max_error": err, "min_on_unit": float(np.nanmin(unit)),
        "max_on_unit": float(np.nanmax(unit)), "at_overshoot": over,
        "basin_u": stability_basin(f, cap=cap), "cap": cap,
    }


def round_to(dtype: torch.dtype) -> Callable[[np.ndarray], np.ndarray]:
    """Scalar rounding model: round a float64 array to ``dtype`` and back."""
    def rnd(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float64)).to(dtype).double().numpy()
    return rnd
