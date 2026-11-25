"""Gram-form squaring-chain polar circuits with warm spectral pre-scaling.

A segment works on the wide view ``W`` (``n <= m``) and its Gram ``R = W W^T``:

    B_0 = R
    L_j = w_I I + sum_{i<=j} w_i B_i          (elementwise, FP32)
    B_{j+1} = L_j @ L_j                       (one symmetric product)
    Q   = v_I I + sum_{i<=k} v_i B_i
    W  <- Q @ W

On each singular value this is ``s -> s * q(s^2)`` with ``deg q = 2^k``. One
segment costs ``alpha n^3`` (SYRK) + ``k n^3`` (symmetric squares) + ``2 alpha n^3``
(``Q @ W``) = ``(3 alpha + k) n^3``; each restart adds another segment. Keeping
``R``, ``R2``, and all ``k + 1`` scaled Gram buffers live requires ``max_k + 4``
FP32 ``n^2`` buffers, which lets ``k`` squares reach the minimax error floor of
a degree-``2^k`` polynomial.

Precision model (sparse-error emulation): every matrix product takes FP16/BF16
operands and accumulates in FP32; Gram buffers stay in FP32 and are rounded once
when they become an operand. In FP16, ``_square`` prescales by an exact power of
two before casting so trace-1 subnormals on ``n >= 4096`` cannot corrupt ``R^2``
or underbound Schatten-8; ``run_circuit`` includes the FP16/BF16 cast residual
``||L - L_h||_F`` in the Schatten-8 upper bound and runs CPU norm reductions in
FP64 to avoid CPU FP32 summation drift.

``SpectralPrescaledPolar`` defaults to stateless rigorous ``scaling="schatten8"``,
which reuses ``R^2`` for zero extra ``n^3`` work and eliminates multi-step drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from research.polar_opt.spectral_scaling import (
    EstimatorConfig,
    SpectralEstimate,
    WarmSpectralEstimator,
    stability_basin,
)

_HALF = (torch.float16, torch.bfloat16)
CAP = 1.25  # basin ceiling: a circuit is "stable" on [0, u] if 0 <= f <= CAP there
# Circuits verified under full coherent rank-1 BF16 matrix sweeps (sigma_max <= 1.25 for x <= 1).
BF16_SAFE_CIRCUITS = frozenset({"S0-k4-lo0.05", "S1-k2+4-lo0.01"})


# --------------------------------------------------------------------------------------
# Circuit description
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChainSegment:
    """One Gram pass. ``ops[j] = (w_I, w_0, ..., w_j)``; ``out = (v_I, v_0, ..., v_k)``."""

    ops: tuple[tuple[float, ...], ...]
    out: tuple[float, ...]

    def __post_init__(self):
        for j, op in enumerate(self.ops):
            if len(op) != j + 2:
                raise ValueError(f"op {j} needs {j + 2} coefficients, got {len(op)}")
        if len(self.out) != len(self.ops) + 2:
            raise ValueError("out needs v_I plus one coefficient per buffer")

    @property
    def squarings(self) -> int:
        return len(self.ops)

    def scalar(self, s, rnd=None):
        """Map singular values ``s`` (float64 array) the way the matrix code does.

        ``rnd`` (e.g. ``spectral_scaling.round_to(torch.float16)``) rounds every
        value that becomes a half-precision operand: ``s``, ``R`` before the
        first square, each ``L_j`` and ``Q``. FP32 buffers are not rounded.
        """
        r = (lambda x: x) if rnd is None else rnd
        s = r(np.asarray(s, dtype=np.float64))
        lam = s * s
        bufs = [lam]
        for j, (wI, *w) in enumerate(self.ops):
            if j == 0:  # (p R + q I)^2 expanded so R^2 is reused (see _run_segment)
                p = w[0]
                bufs.append(p * p * r(lam) ** 2 + 2 * p * wI * lam + wI * wI)
            else:
                bufs.append(r(wI + sum(c * b for c, b in zip(w, bufs))) ** 2)
        vI, *v = self.out
        q = vI + sum(c * b for c, b in zip(v, bufs))
        return r(r(q) * s)

    def scaled(self, factor: float) -> "ChainSegment":
        """Same segment with its output multiplied by ``factor``."""
        return ChainSegment(self.ops, tuple(factor * c for c in self.out))


@dataclass(frozen=True)
class GramCircuit:
    """Segments with a restart (``W <- Q W``, then ``R <- W W^T``) between them.

    Design record: ``|f - 1| <= error`` on ``[lo, 1]`` and ``0 <= f <= CAP`` on
    ``[0, basin]`` in exact arithmetic, where ``f`` maps ``sigma / sigma_hat``.
    """

    name: str
    segments: tuple[ChainSegment, ...]
    lo: float
    basin: float
    error: float

    @property
    def restarts(self) -> int:
        return len(self.segments) - 1

    @property
    def squarings(self) -> tuple[int, ...]:
        return tuple(s.squarings for s in self.segments)

    def flops(self, alpha: float = 1.0) -> float:
        """Leading symmetric FLOPs in units of ``n^3`` (SYRK and squares count ``n^3``)."""
        return sum(3 * alpha + s.squarings for s in self.segments)

    def flops_formula(self) -> str:
        return f"({3 * len(self.segments)}α + {sum(self.squarings)})n³"

    def scalar(self, s, rnd=None):
        for seg in self.segments:
            s = seg.scalar(s, rnd)
        return s

    def basin_u(self, cap: float | None = None, rnd=None) -> float:
        """Largest ``u`` with ``0 <= f <= max(CAP, 1 + error)`` on ``[0, u]``."""
        cap = max(CAP, 1 + 1.001 * self.error) if cap is None else cap
        return stability_basin(lambda x: self.scalar(x, rnd), cap=cap)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "lo": self.lo, "basin": self.basin, "error": self.error,
            "restarts": self.restarts, "squarings": list(self.squarings),
            "flops": self.flops_formula(),
            "segments": [{"ops": [list(o) for o in s.ops], "out": list(s.out)} for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GramCircuit":
        segs = tuple(ChainSegment(tuple(tuple(map(float, o)) for o in s["ops"]),
                                  tuple(map(float, s["out"]))) for s in d["segments"])
        return cls(d["name"], segs, float(d["lo"]), float(d["basin"]), float(d["error"]))


# --------------------------------------------------------------------------------------
# Matrix kernels: FP16/BF16 operands, FP32 accumulation, FP32 Gram buffers
# --------------------------------------------------------------------------------------
def _syrk(X: Tensor, use_triton: bool) -> Tensor:
    """FP32 ``X @ X^T`` from a half-precision ``(B, n, m)`` X (exactly symmetric with Triton)."""
    if use_triton and X.is_cuda:
        from research.polar_opt.kernels_blackwell import triton_syrk
        return triton_syrk(X.contiguous(), out_dtype=torch.float32)
    if X.is_cuda:
        return torch.bmm(X, X.mT, out_dtype=torch.float32)
    Xf = X.float()
    return torch.bmm(Xf, Xf.mT)


def _square(L: Tensor, dtype: torch.dtype, use_triton: bool, return_cast_err: bool = False):
    """``L @ L`` for symmetric FP32 ``L``; the operand is rounded once to ``dtype``.

    In FP16, a power-of-two prescale avoids subnormal truncation (critical at n >= 4096 when
    trace(R) = 1 places off-diagonal entries below 2^-14) without altering any mantissa bit.
    """
    if dtype == torch.float16 and L.numel() > 0:
        amax = L.abs().amax(dim=(-2, -1), keepdim=True)
        exp = (13.0 - torch.ceil(torch.log2(amax.clamp_min(1e-38)))).clamp(-14.0, 50.0)
        nonzero = amax > 0
        scale_up = torch.where(nonzero, torch.exp2(exp), torch.ones_like(amax))
        scale_down_1 = torch.where(nonzero, torch.exp2(-exp), torch.ones_like(amax))
        scale_down_2 = torch.where(nonzero, torch.exp2(-2.0 * exp), torch.ones_like(amax))
        L_h = (L * scale_up).to(dtype)
        R2 = _syrk(L_h, use_triton) * scale_down_2
        if return_cast_err:
            diff = L - L_h.float() * scale_down_1
            err = diff.norm(dim=(-2, -1)) if L.is_cuda else diff.double().norm(dim=(-2, -1))
            return R2, err
        return R2
    L_h = L.to(dtype)
    R2 = _syrk(L_h, use_triton)
    if return_cast_err:
        diff = L - L_h.float()
        err = diff.norm(dim=(-2, -1)) if L.is_cuda else diff.double().norm(dim=(-2, -1))
        return R2, err
    return R2


def _mm(Q: Tensor, W: Tensor, dtype: torch.dtype) -> Tensor:
    """``Q @ W`` with half operands, FP32 accumulation and a half-precision result."""
    Q = Q.to(dtype)
    if W.is_cuda:
        return torch.bmm(Q, W)
    return torch.bmm(Q.float(), W.float()).to(dtype)


def _combine(bufs, coefficients, diag: float) -> Tensor:
    """``diag * I + sum c_i B_i`` with in-place adds (few kernels, no identity matrix)."""
    out = bufs[0] * coefficients[0]
    for c, b in zip(coefficients[1:], bufs[1:]):
        if c:
            out.add_(b, alpha=c)
    out.diagonal(dim1=-2, dim2=-1).add_(diag)
    return out


def _run_segment(W: Tensor, R: Tensor, R2: Tensor, seg: ChainSegment, scale: Tensor,
                 dtype: torch.dtype, use_triton: bool) -> Tensor:
    """Run one segment. ``scale`` ((B, 1, 1)) multiplies every singular value first.

    The scale is folded into ``R``, ``R2`` and ``Q``, so ``W`` is never rescaled.
    """
    c2 = scale.square()
    bufs = [R * c2]
    for j, (wI, *w) in enumerate(seg.ops):
        if j == 0:  # (p R + q I)^2 = p^2 R^2 + 2 p q R + q^2 I, reusing R2
            p = w[0]
            B = (R2 * c2.square()).mul_(p * p).add_(bufs[0], alpha=2 * p * wI)
            B.diagonal(dim1=-2, dim2=-1).add_(wI * wI)
        else:
            B = _square(_combine(bufs, w, wI), dtype, use_triton)
        bufs.append(B)
    vI, *v = seg.out
    return _mm(_combine(bufs, v, vI).mul_(scale), W, dtype)


def run_circuit(G: Tensor, circuit: GramCircuit, *, sigma: Tensor | None = None,
                dtype: torch.dtype = torch.float16, use_triton_syrk: bool = False,
                scaler: "WarmSpectralScaler | None" = None, key=None,
                schatten8: bool = False, return_info: bool = False):
    """Orthogonalize ``G`` of shape ``(..., m, n)`` with ``circuit``; returns ``G.dtype``.

    Scaling, in order of preference: ``scaler`` (guarded warm spectral estimate
    on the Gram matrix), ``schatten8=True`` (stateless rigorous Schatten-8 upper
    bound from ``R`` and ``R2``), an explicit ``sigma`` (shape ``G.shape[:-2]``,
    in ``G``'s units), else the Frobenius norm (always safe, but wastes interval).
    """
    if dtype not in _HALF:
        raise TypeError("dtype must be float16 or bfloat16")
    shape = G.shape
    X = G.reshape(-1, *shape[-2:]).float()
    tall = X.shape[-2] > X.shape[-1]
    W = X.mT if tall else X
    # Overflow/underflow-safe Frobenius normalization; dead/sub-1e-12/non-finite inputs map to 0.
    amax = W.abs().amax(dim=(-2, -1), keepdim=True)
    tiny = torch.finfo(torch.float32).tiny
    W_unit = torch.where(amax > 0, W / amax.clamp_min(tiny), torch.zeros_like(W))
    fro_unit = W_unit.norm(dim=(-2, -1), keepdim=True) if W.is_cuda else W_unit.double().norm(dim=(-2, -1), keepdim=True).float()
    fro = fro_unit * amax
    valid = (fro > 1e-12) & torch.isfinite(fro)
    W = torch.where(valid, W_unit / fro_unit.clamp_min(tiny), torch.zeros_like(W)).to(dtype).contiguous()
    info = None
    for i, seg in enumerate(circuit.segments):
        R = _syrk(W, use_triton_syrk)
        if schatten8 and i == 0:
            R2, cast_err = _square(R, dtype, use_triton_syrk, return_cast_err=True)
        else:
            R2 = _square(R, dtype, use_triton_syrk)
            cast_err = None
        if i:
            scale = torch.ones_like(fro)
        elif scaler is not None:
            info = scaler.estimate(R, R2, key=key)
            sig = info.sigma.reshape(-1, 1, 1)
            scale = torch.where(sig > 0, 1.0 / sig.clamp_min(1e-30), 1.0)  # zero input stays zero
        elif schatten8:
            R_n = R if R.is_cuda else R.double()
            R2_n = R2 if R2.is_cuda else R2.double()
            ce_n = cast_err if R.is_cuda else cast_err.double()
            tr = R_n.diagonal(dim1=-2, dim2=-1).sum(-1)
            s4 = R_n.norm(dim=(-2, -1))
            s8 = R2_n.norm(dim=(-2, -1)).sqrt() + ce_n
            upper = torch.minimum(tr, torch.minimum(s4, s8)).clamp_min(0.0).sqrt().float().reshape(-1, 1, 1)
            scale = torch.where(upper > 0, 1.0 / (upper * 1.005).clamp_min(1e-30), 1.0)
        elif sigma is not None:
            sig = sigma.reshape(-1, 1, 1).float().to(fro.device)
            scale = torch.where(sig > 0, fro / sig.clamp_min(1e-30), 1.0)
        else:
            scale = torch.ones_like(fro)
        W = _run_segment(W, R, R2, seg, scale, dtype, use_triton_syrk)
    out = (W.mT if tall else W).to(G.dtype).reshape(shape)
    return (out, info) if return_info else out


# --------------------------------------------------------------------------------------
# Warm spectral pre-scaling
# --------------------------------------------------------------------------------------
class WarmSpectralScaler:
    """Per-parameter :class:`WarmSpectralEstimator` states that operate on Gram matrices."""

    def __init__(self, config: EstimatorConfig | None = None, seed: int = 0):
        self.config = config or EstimatorConfig()
        self.seed = seed
        self.states: dict = {}

    def estimate(self, R: Tensor, R2: Tensor | None = None, key=None) -> SpectralEstimate:
        """Guarded ``sigma_max`` estimate of the normalized ``W`` with ``R = W W^T``, ``R2 = R^2``."""
        est = self.states.get(key)
        if est is None:
            est = self.states[key] = WarmSpectralEstimator(self.config, self.seed)
        return est.estimate(R, gram=True, A2=R2)

    def reset(self) -> None:
        self.states.clear()


class SpectralPrescaledPolar:
    """Muon orthogonalizer: guarded warm spectral scaling followed by a short Gram circuit.

    >>> polar = SpectralPrescaledPolar("S1-k3+4-lo0.005")
    >>> update = polar(momentum, key="blocks.0.qkv")   # key selects the warm state
    """

    def __init__(self, circuit: str | GramCircuit = "S1-k3+4-lo0.005", *, dtype: torch.dtype = torch.float16,
                 use_triton_syrk: bool = False, estimator: EstimatorConfig | None = None,
                 scaling: str = "schatten8", seed: int = 0, allow_unsafe_bf16: bool = False):
        if scaling not in ("spectral", "schatten8", "frobenius"):
            raise ValueError("scaling must be 'spectral', 'schatten8', or 'frobenius'")
        if isinstance(circuit, str):
            if circuit in CIRCUITS:
                self.circuit = CIRCUITS[circuit]
            elif circuit in FP16_ONLY_CIRCUITS:
                self.circuit = FP16_ONLY_CIRCUITS[circuit]
            else:
                raise KeyError(f"Unknown circuit {circuit!r}")
        else:
            self.circuit = circuit
        if dtype == torch.bfloat16 and not allow_unsafe_bf16 and self.circuit.name not in BF16_SAFE_CIRCUITS:
            raise ValueError(
                f"Circuit {self.circuit.name!r} failed coherent rank-1 BF16 matrix safety; use "
                f"dtype=torch.float16, pass allow_unsafe_bf16=True, or choose from "
                f"BF16_SAFE_CIRCUITS={sorted(BF16_SAFE_CIRCUITS)}."
            )
        self.dtype, self.use_triton_syrk = dtype, use_triton_syrk
        self.schatten8 = scaling == "schatten8"
        self.scaler = WarmSpectralScaler(estimator, seed) if scaling == "spectral" else None

    def __call__(self, G: Tensor, key=None, return_info: bool = False):
        return run_circuit(G, self.circuit, dtype=self.dtype, use_triton_syrk=self.use_triton_syrk,
                           scaler=self.scaler, key=key, schatten8=self.schatten8, return_info=return_info)


# --------------------------------------------------------------------------------------
# Discovered circuits (copied verbatim from results/circuit_search_cycle1.json)
# --------------------------------------------------------------------------------------
_CIRCUIT_SPECS = {
    "S0-k4-lo0.05": {
        "name": "S0-k4-lo0.05",
        "lo": 0.05,
        "basin": 1.15,
        "error": 0.19296479556245394,
        "segments": [
            {
                "ops": [
                    [-0.7589783259477789, 1.0],
                    [-0.82049388437767, 0.3055401729591701, 3.1603157561340214],
                    [-0.7704125323423244, 0.5856728989414185, -0.12277595923306574, 1.8411373902931703],
                    [-0.11754132896453053, -0.050401341300462726, -0.17359994624134928, 0.26800426497654645, 0.94953898305169],
                ],
                "out": [1.7894167332911306, -1.6432540478540254, 1.995077215999924, 2.042559902771412, -1.1816998843397863, 14.548396009414853],
            }
        ],
    },
    "S0-k6-lo0.05": {
        "name": "S0-k6-lo0.05",
        "lo": 0.05,
        "basin": 1.15,
        "error": 0.05147676226335207,
        "segments": [
            {
                "ops": [
                    [-0.15920863026601803, 1.0],
                    [-1.0196907728200255, 0.4392029693955648, 0.776836369015596],
                    [-4.6764174951696615, 7.871893273206732, -5.088887068089007, 3.8054074899242645],
                    [-0.7122316409606168, -0.0971035791545849, 0.3277062679092741, -0.3013563198972298, 1.5217554528558246],
                    [0.14230722318583827, 0.0684518552136749, -0.3762308646435235, 0.17698166331890275, -1.4262739787355028, 0.4368269131186607],
                    [3.0812159377472867, -1.8858837790040897, -0.10830918902173745, 1.1826738427400307, -6.475740438171224, -4.478044319673922, 2.412538522803807],
                ],
                "out": [-18.522856613497307, 1.8498153299593776, 8.144261599798046, -7.71192093023467, 53.66329252272975, 34.623055267060515, -16.607474536187375, 4.658394720664786],
            }
        ],
    },
    "S1-k2+4-lo0.01": {
        "name": "S1-k2+4-lo0.01",
        "lo": 0.01,
        "basin": 1.15,
        "error": 0.14584267982443233,
        "segments": [
            {
                "ops": [
                    [-0.6548699888153007, 1.0],
                    [0.28275837883204424, -0.7013757086453993, 1.672458334730463],
                ],
                "out": [-0.16082252483417483, 0.6541427428907861, -1.1572094396564119, 6.607498091204056],
            },
            {
                "ops": [
                    [-0.5423367161692504, 1.0],
                    [-0.5726719066076308, -0.24656641683071742, 5.346876025960638],
                    [-1.0111054362223821, 0.726939524133541, 1.7530800571648635, 1.4954735527541387],
                    [-0.2769230528715032, -0.053479818016690606, 0.9227703474607576, 0.28641855313484155, 0.719090875284547],
                ],
                "out": [2.3001419332309645, -2.805301013341339, 6.39336892606388, 0.5090075753778688, -0.9728725686042748, 13.354039541929104],
            },
        ],
    },
    "S1-k3+4-lo0.01": {
        "name": "S1-k3+4-lo0.01",
        "lo": 0.01,
        "basin": 1.15,
        "error": 0.06997519519165285,
        "segments": [
            {
                "ops": [
                    [-0.4786071185573397, 1.0],
                    [0.3727700132300974, -1.5250206216920974, 2.738221054238937],
                    [-0.5519814432887283, 0.3584463139173963, 0.03557702102976633, 1.5438320010093083],
                ],
                "out": [-0.22395333305488763, 0.14033527941715893, 0.5887749511567563, 1.4897258962568356, 10.120253454687122],
            },
            {
                "ops": [
                    [-0.4647234149723282, 1.0],
                    [0.553671805379356, -0.9592784833393869, 2.0666418154926727],
                    [0.39453983663911146, -0.8397851710074604, 0.40282978703387884, 0.5184618793629486],
                    [1.912460347353119, -3.5080249261561405, 3.8450881663005356, -11.333651915473917, 9.590776134465694],
                ],
                "out": [0.4071084220246475, 1.5927375410610987, -3.3911420997690636, 9.23203837528111, -4.344184705825304, 7.114486699348014],
            },
        ],
    },
    "S1-k3+4-lo0.005": {
        "name": "S1-k3+4-lo0.005",
        "lo": 0.005,
        "basin": 1.15,
        "error": 0.1377700265777,
        "segments": [
            {
                "ops": [
                    [-0.49385096118172467, 1.0],
                    [0.3722923786346221, -1.3897652519284596, 2.5737454683867553],
                    [-0.5857354837124857, 0.5756277553406401, -0.3304742728677731, 1.666334448253571],
                ],
                "out": [-0.19684360492448746, 0.25634953815610806, 0.3574158073231677, 0.858837271924139, 10.890086565099466],
            },
            {
                "ops": [
                    [-0.5700236178310539, 1.0],
                    [-0.2067237331531353, -0.2946231541932569, 3.713831143970203],
                    [-0.07165604316106963, 0.4808932770692318, -3.9195679515112203, 2.3452292045242067],
                    [-0.0791478749138764, -0.14437162783826044, 0.8500788080458147, -0.49010460005934403, 1.2930389819648147],
                ],
                "out": [1.9122490067395033, -1.4413903652171893, 2.755710207599569, 2.8580314348752234, -1.237727464082057, 13.95082318964976],
            },
        ],
    },
    "S1-k3+5-lo0.01": {
        "name": "S1-k3+5-lo0.01",
        "lo": 0.01,
        "basin": 1.15,
        "error": 0.0556448923797106,
        "segments": [
            {
                "ops": [
                    [-0.4786071185573397, 1.0],
                    [0.3727700132300974, -1.5250206216920974, 2.738221054238937],
                    [-0.5519814432887283, 0.3584463139173963, 0.03557702102976633, 1.5438320010093083],
                ],
                "out": [-0.22395333305488763, 0.14033527941715893, 0.5887749511567563, 1.4897258962568356, 10.120253454687122],
            },
            {
                "ops": [
                    [-0.2820541125327334, 1.0],
                    [0.6552872538977976, -2.604829392829646, 4.333037716996138],
                    [-0.2232304942455419, -0.2557012432092832, 1.6191100257104876, 1.0944229694425522],
                    [-1.0647633987443015, 3.0369279044454696, -4.487008290889673, 0.45477639637931533, 1.9669488039364234],
                    [0.1545849407673523, -0.31328742041141855, -0.07915599107734347, -0.5244991735663058, 0.7833919131022006, 0.5928195367617428],
                ],
                "out": [2.295811282511387, -2.779797578856914, 2.1231150998725843, 4.787556998426757, -0.3860020714735408, -4.264064067241661, 22.627473294686602],
            },
        ],
    },
}

_FP16_ONLY_SPECS = {
    "S1-k4+5-lo0.002": {
        "name": "S1-k4+5-lo0.002",
        "lo": 0.002,
        "basin": 1.15,
        "error": 0.10978755597514434,
        "segments": [
            {
                "ops": [
                    [-0.3309490930070439, 1.0],
                    [0.5652871258967422, -2.9892079102197298, 3.96899097778878],
                    [-0.7881335507218657, -1.0360958682187476, 1.881582767309963, 1.5820488664235473],
                    [-0.6659382816131821, -0.10422731411097381, 0.6560179729489231, 0.720507160960855, 0.8735792419053264],
                ],
                "out": [0.9025216792170797, 0.29153230083540144, -0.8456855011330409, -0.6211974863997396, -1.065619105658214, 23.380439404743573],
            },
            {
                "ops": [
                    [-0.5208699491496268, 1.0],
                    [0.3830778179905685, -1.6371196284327647, 1.186246138403688],
                    [-0.6458311495270223, -1.0026335543272449, -1.1559414163434292, 1.5750284321424666],
                    [-2.89528052486269, -8.644573029539355, -11.708827340189927, 13.829660768721125, 6.3963792152495555],
                    [-0.7945864509616654, 0.7113004825010036, 2.3916655826947153, -0.3546075661351371, 0.19842723437278234, 1.3157165685432721],
                ],
                "out": [-3.307690014550009, -24.010847414588753, -19.642854324482457, 31.4518950863311, 12.552823558029537, 5.485084622416678, 13.621959150024194],
            },
        ],
    },
    "S0-k7-lo0.02": {
        "name": "S0-k7-lo0.02",
        "lo": 0.02,
        "basin": 1.15,
        "error": 0.061649688506316025,
        "segments": [
            {
                "ops": [
                    [-0.2801000019373751, 1.0],
                    [0.21364051573444895, -1.9432763554995611, 1.4104887386965395],
                    [0.32716756481947534, -5.708063236511931, 4.4066862703894945, 3.110166215910059],
                    [-0.7376983558100594, -2.5611492843494803, 1.6351297076010136, 1.776207069557724, 1.3682681414877633],
                    [-0.8687141751257533, -0.446304361006281, -0.026163232369099527, -0.06095082580799407, 0.5615768763311066, 0.8438199338197522],
                    [0.04109831539219202, 0.5227114753853409, 0.4043781228195334, -0.976364937682246, -0.43581872448209685, -0.7626077700504612, 0.7661714967915595],
                    [1.32060609912352, 1.502370730659386, 0.3855904375816301, 0.28722838202062045, -1.755951753525383, -2.659493913416208, -2.3003132233712567, 2.1965457130536628],
                ],
                "out": [39.27250897605925, 14.346962532770565, 19.92700243572992, 22.498373637292897, -42.20388233639485, -69.75572940275275, -57.19245246563515, 68.20993300439561, 64.06822064411914],
            }
        ],
    },
}

CIRCUITS: dict[str, GramCircuit] = {k: GramCircuit.from_dict(v) for k, v in _CIRCUIT_SPECS.items()}
FP16_ONLY_CIRCUITS: dict[str, GramCircuit] = {k: GramCircuit.from_dict(v) for k, v in _FP16_ONLY_SPECS.items()}
