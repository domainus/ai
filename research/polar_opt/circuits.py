"""Gram-space polar circuits with warm-started spectral pre-scaling.

A circuit is a list of odd polynomials ``p_t(x) = sum_j c_{t,j} x^(2j+1)``. For a
wide matrix ``X`` (``m <= n``) each step is ``X <- X h_t(X^T X)`` with
``h_t(y) = sum_j c_{t,j} y^j``. The Gram form never touches ``X`` inside a pass::

    G = X X^T; R = G; Q = I
    for each step: H = h(R); Q = H Q; R = H R H      (R is the Gram of Q X)
    X <- Q X                                          (end of the pass)

A *restart* ends a pass early and re-forms ``G`` from the new ``X``. That costs
``3 alpha n^3`` but removes the rounding error that builds up in ``Q`` and ``R``.

Pre-scaling (all certified, ``O(n^2)`` extra work):

1. ``X0 = X / ||X||_F``.
2. ``G = X0 X0^T``; divide by ``||G||_F`` (Schatten-4 bound, keeps FP16 G^2
   away from underflow).
3. ``G^2`` is the first step's ``R^2``. ``U = ||G^2||_F^{1/4}`` is a rigorous
   Schatten-8 upper bound on ``sigma_max``.
4. ``WarmSpectralScaler`` adds a warm-started power lower bound ``L`` from
   ``G^2 v`` and sets ``s = min(U, max(L, U / u))``. Then
   ``sigma_max / s <= u`` always, so the circuit never sees ``x > u``.

FLOP model (``n`` = small side, ``alpha = N / n``), units of ``n^3``:
``sym``: SYRK ``alpha``, final ``Q X`` ``2 alpha``, each ``n x n`` product of
commuting symmetric matrices ``1``. ``dense``: Gram ``2 alpha``, apply
``2 alpha``, each square product ``2``. A quintic pass with ``T`` steps needs
``4T - 3`` square products (``T >= 2``) or 1 (``T = 1``).

``emulate=True`` runs fp32 arithmetic but rounds every matmul operand and every
stored matrix to ``dtype``. This models FP16/BF16 tensor cores: low-precision
operands, fp32 accumulation, fp32 epilogue, one rounding per stored result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .spectral_scaling import gram_schatten_upper

__all__ = [
    "CircuitConfig",
    "WarmSpectralScaler",
    "polar_spectral_circuit",
    "polar_xspace_circuit",
    "svd_polar",
    "gram_square_products",
    "CIRCUITS",
    "DEFAULT_CIRCUIT",
    "JORDAN_NS5",
]

_TINY = 1e-30


# --------------------------------------------------------------------------
# Configuration and FLOP model
# --------------------------------------------------------------------------


def gram_square_products(pass_steps) -> int:
    """Number of ``n x n`` products for one Gram pass (see module docstring)."""
    T = len(pass_steps)
    k = 0
    for t, c in enumerate(pass_steps):
        k += max(len(c) - 2, 0)  # powers R^2 .. R^(nc-1)
        k += 1 if t > 0 else 0  # Q <- H Q
        k += 2 if t < T - 1 else 0  # R <- H R H
    return k


@dataclass(frozen=True)
class CircuitConfig:
    """A Gram-space polynomial circuit.

    ``steps[t]`` holds the odd coefficients ``(c0, c1, c2, ...)`` of
    ``p_t(x) = c0 x + c1 x^3 + c2 x^5 + ...``. ``restarts`` lists the step
    indices that start a new Gram pass. ``[ell, u]`` is the design interval.
    ``prescale`` is ``"schatten8"`` (certified, default), ``"frobenius"``, or
    ``"none"`` (the caller already scaled ``X``).
    """

    name: str
    steps: tuple[tuple[float, ...], ...]
    restarts: tuple[int, ...] = ()
    ell: float = 1e-3
    u: float = 1.0
    prescale: str = "schatten8"
    notes: str = ""

    def passes(self) -> list[tuple[tuple[float, ...], ...]]:
        cuts = [0, *sorted(self.restarts), len(self.steps)]
        return [self.steps[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]

    def square_products(self) -> int:
        k = sum(gram_square_products(p) for p in self.passes())
        if self.prescale == "schatten8" and len(self.steps[0]) < 3:
            k += 1  # G^2 is not otherwise needed
        return k

    def flops(self, alpha: float, model: str = "sym") -> float:
        """Cost in units of ``n^3``."""
        P, k = len(self.passes()), self.square_products()
        if model == "sym":
            return 3 * alpha * P + k
        if model == "dense":
            return 4 * alpha * P + 2 * k
        raise ValueError(model)

    def poly(self, x):
        """Composed scalar polynomial (works on floats, numpy arrays, tensors)."""
        for c in self.steps:
            x2 = x * x
            h = 0.0
            for cj in reversed(c):
                h = h * x2 + cj
            x = x * h
        return x

    def with_steps(self, steps, **kw) -> "CircuitConfig":
        d = dict(self.__dict__)
        d.update(kw)
        d["steps"] = tuple(tuple(float(v) for v in c) for c in steps)
        return CircuitConfig(**d)


# --------------------------------------------------------------------------
# Arithmetic model
# --------------------------------------------------------------------------


class _Arith:
    """Native low precision, or fp32 emulation with explicit rounding."""

    def __init__(self, dtype: torch.dtype, emulate: bool):
        self.dtype = dtype
        self.emulate = emulate and dtype not in (torch.float32, torch.float64)
        self.work = torch.float32 if self.emulate else dtype
        if dtype == torch.float64:
            self.work = torch.float64
        self.eps = torch.finfo(dtype).eps

    def store(self, x: torch.Tensor) -> torch.Tensor:
        if self.emulate:
            return x.to(self.dtype).to(self.work)
        return x.to(self.dtype)

    def mm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Operands are already stored values; fp32 accumulation in emulation.
        return a @ b

    def epi(self, x: torch.Tensor) -> torch.Tensor:
        """Upcast for an fp32 epilogue."""
        return x.to(torch.float64 if self.work == torch.float64 else torch.float32)


def _eye_like(R: torch.Tensor) -> torch.Tensor:
    return torch.eye(R.shape[-1], dtype=R.dtype, device=R.device).expand_as(R)


def _sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.mT)


def _poly_h(c, R, powers, ar: _Arith) -> torch.Tensor:
    """``H = c0 I + c1 R + c2 R^2 + ...`` in the fp32 epilogue, stored once."""
    H = c[0] * _eye_like(ar.epi(R))
    for j in range(1, len(c)):
        H = H + c[j] * ar.epi(powers[j])
    return ar.store(H)


def _gram_pass(X, steps, ar: _Arith, G, G2=None):
    """Run one Gram pass; returns the new stored ``X``."""
    R, Q, T = G, None, len(steps)
    for t, c in enumerate(steps):
        powers = {1: R}
        for j in range(2, len(c)):
            if j == 2 and t == 0 and G2 is not None:
                powers[2] = G2
            else:
                powers[j] = ar.store(ar.mm(powers[j - 1], R))
        H = _poly_h(c, R, powers, ar)
        Q = H if Q is None else ar.store(_sym(ar.epi(ar.mm(H, Q))))
        if t < T - 1:
            RH = ar.store(ar.mm(R, H))
            R = ar.store(_sym(ar.epi(ar.mm(H, RH))))
    return ar.store(ar.mm(Q, X))


# --------------------------------------------------------------------------
# Warm-started spectral scaler
# --------------------------------------------------------------------------


@dataclass
class WarmSpectralScaler:
    """Streaming, certified estimate of ``sigma_max`` from ``G`` and ``G^2``.

    Keep one instance per parameter. The warm vector ``v`` lives in the Gram
    space (length ``min(m, n)``). Each power step is one ``G^2`` matvec
    (``n^2`` flops), worth two power steps on ``M^T M``.

    ``s = min(U, max(gamma * L, U / u_basin))`` with the Schatten-8 upper bound
    ``U`` and the power lower bound ``L``. Guarantee: ``sigma_max / s <= u_basin``
    for any warm start, including one orthogonal to the top singular vector.
    """

    u_basin: float = 1.05
    gamma: float = 1.0
    power_steps: int = 1
    probes: int = 0
    seed: int = 0
    v: torch.Tensor | None = None
    last: dict = field(default_factory=dict)

    def reset(self) -> None:
        self.v = None

    def __call__(self, G: torch.Tensor, G2: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
        """Return the scale ``s`` (shape ``(...)``) relative to ``G``'s ``X``."""
        G2f = G2.float()
        U = gram_schatten_upper(G2f, 2, eps)
        v = self.v
        if v is None or v.shape != G2f.shape[:-1] or v.device != G2f.device:
            # Start from the row of G with the largest diagonal (row norm).
            idx = G.float().diagonal(dim1=-2, dim2=-1).argmax(-1)
            v = torch.gather(G.float(), -2, idx[..., None, None].expand(*idx.shape, 1, G.shape[-1]))
            v = v.squeeze(-2)
        V = v.float()[..., None]
        if self.probes:
            gen = torch.Generator().manual_seed(self.seed)
            self.seed += 1
            rnd = torch.randn(*G2f.shape[:-1], self.probes, generator=gen).to(G2f.device)
            V = torch.cat([V, rnd], dim=-1)
        V = V / torch.linalg.vector_norm(V, dim=-2, keepdim=True).clamp_min(_TINY)
        L = torch.zeros_like(U)
        for _ in range(self.power_steps):
            W = G2f @ V
            nrm = torch.linalg.vector_norm(W, dim=-2)  # ||G^2 v|| <= sigma^4
            L, best = nrm.max(-1)
            V = W / nrm[..., None, :].clamp_min(_TINY)
        L = L.clamp_min(0).pow(0.25)
        v_new = torch.gather(V, -1, best[..., None, None].expand(*best.shape, V.shape[-2], 1)).squeeze(-1)
        self.v = v_new.detach()
        s = torch.minimum(U, torch.maximum(self.gamma * L, U / self.u_basin))
        s = torch.where(U > 0, s, torch.ones_like(s))
        self.last = {"upper": U, "lower": L, "scale": s}
        return s


# --------------------------------------------------------------------------
# Circuits
# --------------------------------------------------------------------------


def _as_wide(X: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if X.shape[-2] > X.shape[-1]:
        return X.mT, True
    return X, False


def polar_spectral_circuit(
    X: torch.Tensor,
    config: "CircuitConfig | str",
    scaler: WarmSpectralScaler | None = None,
    *,
    dtype: torch.dtype | None = None,
    emulate: bool = False,
    prescale: str | None = None,
) -> torch.Tensor:
    """Approximate polar factor of ``X`` (shape ``(..., m, n)``) with a Gram circuit.

    ``dtype`` is the storage/operand precision (default ``X.dtype``). Tall
    inputs are transposed so the Gram matrix is ``min(m, n)`` square. With a
    ``scaler`` the certified warm-started scale is used. Otherwise
    ``prescale`` (default ``config.prescale``) selects ``"schatten8"``
    (``s = U / u``), ``"frobenius"``, or ``"none"``. Returns ``X.dtype``.
    """
    cfg = CIRCUITS[config] if isinstance(config, str) else config
    mode = prescale or cfg.prescale
    out_dtype = X.dtype
    ar = _Arith(dtype or X.dtype, emulate)
    Xw, transposed = _as_wide(X)

    if mode == "none":
        X0 = ar.store(Xw.to(ar.work))
        fro = torch.ones(X0.shape[:-2] + (1, 1), dtype=X0.dtype, device=X0.device)
    else:
        Xf = Xw.to(torch.float64 if ar.work == torch.float64 else torch.float32)
        fro = torch.linalg.vector_norm(Xf, dim=(-2, -1), keepdim=True)
        X0 = ar.store(Xf / fro.clamp_min(_TINY))

    passes = cfg.passes()
    G = ar.store(ar.mm(X0, X0.mT))
    G2 = None
    if mode in ("schatten8",) or scaler is not None:
        # Schatten-4 normalization keeps G^2 in the normal range of FP16.
        a2 = torch.linalg.vector_norm(ar.epi(G), dim=(-2, -1), keepdim=True) * (1 + 2 * ar.eps)
        a2 = torch.where(a2 > 0, a2, torch.ones_like(a2))
        G = ar.store(ar.epi(G) / a2)
        G2 = ar.store(ar.mm(G, G))
        if scaler is not None:
            s = scaler(G, G2, ar.eps)[..., None, None]
            # sigma_hat in the units of X: sigma_max(X) / sigma_hat <= u_basin.
            scaler.last["sigma_hat"] = (fro * a2.sqrt() * s).squeeze(-1).squeeze(-1)
        else:
            s = gram_schatten_upper(ar.epi(G2), 2, ar.eps)[..., None, None] / cfg.u
            s = torch.where(s > 0, s, torch.ones_like(s))
        G = ar.store(ar.epi(G) / s**2)
        G2 = ar.store(ar.epi(G2) / s**4)
        X0 = ar.store(ar.epi(X0) / (a2.sqrt() * s))

    Xc = X0
    for i, steps in enumerate(passes):
        if i > 0:
            G, G2 = ar.store(ar.mm(Xc, Xc.mT)), None
        Xc = _gram_pass(Xc, steps, ar, G, G2)
    out = Xc.mT if transposed else Xc
    return out.to(out_dtype)


def polar_xspace_circuit(
    X: torch.Tensor,
    steps,
    *,
    dtype: torch.dtype | None = None,
    emulate: bool = False,
    prescale: str = "frobenius",
    safety: float = 1.0,
) -> torch.Tensor:
    """Classic X-space iteration ``X <- c0 X + (c1 A + c2 A^2) X``, ``A = X X^T``.

    Used for baselines (Jordan NS5, Polar-Express-style greedy). Cost per
    quintic step: ``(3 alpha + 1) n^3`` (sym model). ``safety`` divides the
    Frobenius scale, e.g. ``1.01`` maps ``sigma_max`` to at most ``1/1.01``.
    """
    ar = _Arith(dtype or X.dtype, emulate)
    Xw, transposed = _as_wide(X)
    Xf = Xw.to(torch.float64 if ar.work == torch.float64 else torch.float32)
    if prescale == "frobenius":
        Xf = Xf / (torch.linalg.vector_norm(Xf, dim=(-2, -1), keepdim=True).clamp_min(_TINY) * safety)
    Xc = ar.store(Xf)
    for c in steps:
        A = ar.store(ar.mm(Xc, Xc.mT))
        powers = {1: A}
        for j in range(2, len(c)):
            powers[j] = ar.store(ar.mm(powers[j - 1], A))
        B = 0.0
        for j in range(1, len(c)):
            B = B + c[j] * ar.epi(powers[j])
        BX = ar.mm(ar.store(B), Xc)
        Xc = ar.store(c[0] * ar.epi(Xc) + ar.epi(BX))
    out = Xc.mT if transposed else Xc
    return out.to(X.dtype)


def svd_polar(X: torch.Tensor) -> torch.Tensor:
    """Exact polar factor ``U V^T`` in fp64 (reference)."""
    U, _, Vh = torch.linalg.svd(X.to(torch.float64), full_matrices=False)
    return U @ Vh


# Jordan's Muon NS5 coefficients (github.com/KellerJordan/Muon), 5 steps.
JORDAN_NS5 = ((3.4445, -4.7750, 2.0315),) * 5


# --------------------------------------------------------------------------
# Registry of discovered circuits (filled from results/circuit_search_cycle1.json)
# --------------------------------------------------------------------------

CIRCUITS: dict[str, CircuitConfig] = {}


def _register(cfg: CircuitConfig) -> CircuitConfig:
    CIRCUITS[cfg.name] = cfg
    return cfg


# Copied from results/circuit_search_cycle1.json (run of circuit_search.main).
# All use u = 1.05: pair them with WarmSpectralScaler(u_basin=1.05) or the
# default "schatten8" prescale. "SPT-0r" = T quintic steps in one Gram pass;
# "SPT-1rK" = one restart before step K. Cost (sym model, n^3): 0r: 3a + 4T - 3;
# SP3-1r1 6a + 6; SP4-1r1 6a + 10 (= PS2-55 formula); SP5-1r2 6a + 14.
_register(CircuitConfig(
    name="SP3-0r-l0.05-fp16",
    steps=(
        (6.442851150186531, -15.632045706045407, 10.086891935857436),
        (2.5994826603205374, -1.8726135123601726, 0.4293545566573855),
        (1.929102580531148, -1.299189927372965, 0.3714077742790561),
    ),
    restarts=(),
    ell=0.05,
    u=1.05,
    notes="SP3-0r-l0.05-u1.05-m0.01-fp16-ref; refined for fp16; inflate 1.01; "
    "held-out worst fp16 0.0222, bf16 0.1956; exact 1.26e-02",
))
_register(CircuitConfig(
    name="SP3-1r1-l0.05",
    steps=(
        (6.44323847616669, -15.65077005552096, 10.07424952573664),
        (2.565786590094792, -1.868594962139617, 0.4275894997921128),
        (1.9283773022570454, -1.2965726070914796, 0.37053404503538423),
    ),
    restarts=(1,),
    ell=0.05,
    u=1.05,
    notes="SP3-1r@1-l0.05-u1.05-m0.01-fp16; greedy minimax; inflate 1.01; "
    "held-out worst fp16 0.0101, bf16 0.0245; exact 9.15e-03",
))
_register(CircuitConfig(
    name="SP4-1r1-l0.05",
    steps=(
        (6.3924504914326885, -15.237210083621369, 9.619199124720579),
        (2.5589622892793447, -1.834135608063992, 0.41140920498925765),
        (1.9225947261951633, -1.2790146991731515, 0.3598606727305219),
        (1.8566752460814284, -1.2133309901387563, 0.35666099950936203),
    ),
    restarts=(1,),
    ell=0.05,
    u=1.05,
    notes="SP4-1r@1-l0.05-u1.05-m0.02-fp16; greedy minimax; inflate 1.02; "
    "held-out worst fp16 0.0027, bf16 0.0235; exact 5.26e-06",
))
_register(CircuitConfig(
    name="SP4-1r1-l0.02-fp16",
    steps=(
        (7.310518015705668, -18.678976613729734, 12.210956699425243),
        (3.233050619695339, -2.380263265823123, 0.4825752535512917),
        (2.2436946478980153, -1.5959846999450769, 0.3993728233315667),
        (1.878338120365188, -1.2443845091644479, 0.36625694413636267),
    ),
    restarts=(1,),
    ell=0.02,
    u=1.05,
    notes="SP4-1r@1-l0.02-u1.05-m0.01-fp16; greedy minimax; inflate 1.01; "
    "held-out worst fp16 0.0098, bf16 0.0767; exact 9.20e-04",
))
_register(CircuitConfig(
    name="SP4-1r1-l0.02-bf16",
    steps=(
        (7.245215268771397, -18.15682538174988, 11.639145240852288),
        (3.21692333576289, -2.3254100806393856, 0.46203341400597814),
        (2.243552512123268, -1.574684713741587, 0.3859444135374623),
        (1.8707857083050765, -1.2268730592171089, 0.3565166127571125),
    ),
    restarts=(1,),
    ell=0.02,
    u=1.05,
    notes="SP4-1r@1-l0.02-u1.05-m0.02-bf16; greedy minimax; inflate 1.02; "
    "held-out worst bf16 0.0745, fp16 0.0091; exact 1.22e-03",
))
_register(CircuitConfig(
    name="SP5-1r2-l0.01-fp16",
    steps=(
        (7.581045655636782, -19.31740879918861, 12.439506515963672),
        (3.62749680217399, -2.5605161728448085, 0.49228063989302007),
        (2.691382210511111, -1.942686153935957, 0.4220329476923091),
        (1.964294707047183, -1.31894540490489, 0.36297325749416354),
        (1.8568403014679378, -1.2136250235645227, 0.3564644339665493),
    ),
    restarts=(2,),
    ell=0.01,
    u=1.05,
    notes="SP5-1r@2-l0.01-u1.05-m0.02-fp16-ref; refined for fp16; inflate 1.02; "
    "held-out worst fp16 0.0169, bf16 0.1312; exact 4.47e-04",
))

DEFAULT_CIRCUIT = {"fp16": "SP4-1r1-l0.02-fp16", "bf16": "SP4-1r1-l0.02-bf16"}
