"""muP width-transfer coordinate check for AdamW and Muon (CPU only, Cycle 3 T4-01).

The harness trains one small pre-RMSNorm causal Transformer at several widths and measures how
the change of each activation scales with the width. A parameterization transfers across widths
when that change does not depend on the width.

Model
-----
``emb -> [x + attn(norm(x)); x + mlp(norm(x))] * layers -> norm -> W_head``, with RoPE, separate
``q/k/v/o`` matrices, the gated MLP ``down(silu(gate(h)) * up(h))`` and an untied readout. The
head dimension is fixed, so the number of heads grows with the width.

Parameterizations (``d`` = width, ``d_in`` = fan-in, ``m = d_base / d``)
-----------------------------------------------------------------------
============  ===================  ==============================  =======================
parameter     ``sp``               ``mup`` + AdamW                 ``mup`` + Muon
============  ===================  ==============================  =======================
embedding     std 0.02, lr eta     std 1/sqrt(d_base), lr eta      same, lr eta_aux (AdamW)
hidden 2D     std 0.02, lr eta     std 1/sqrt(d_in),               std 1/sqrt(d_in),
                                   lr eta * d_base / d_in          dW = -eta sqrt(d_out/d_in) polar(M)
RMSNorm       1.0, lr eta          1.0, lr eta                     1.0, lr eta_aux (AdamW)
readout       std 0.02, lr eta,    zero, logits * m,               zero, logits * m,
              logits * 1           lr eta * m^readout_lr_power     lr eta_aux * m^readout_lr_power
attention     1/sqrt(head_dim)     1/head_dim                      1/head_dim
============  ===================  ==============================  =======================

``sp`` with Muon (``sp_muon``) is the naive control: ``dW = -eta polar(M)`` without the
``sqrt(d_out / d_in)`` factor, on the SP initialization, readout and attention scale. Under
Muon, ``eta_aux`` is fixed (0.01, the best muP AdamW learning rate of this model) while the
learning-rate check sweeps ``eta``.

Readout learning rate
---------------------
With Adam, an update of the readout moves a logit by ``multiplier * lr * Theta(d)``, because the
sign-like update is aligned with all ``d`` coordinates of the feature. The logit multiplier
``m`` therefore needs a width-independent learning rate (``readout_lr_power = 0``, the
``mup.MuReadout`` + ``MuAdam`` rule, and the default here). The T4-01 task text asked for the
multiplier ``m`` and also the learning rate ``eta * m`` (``readout_lr_power = 1``). That
combination scales the logit change as ``1 / d``; it is kept as the control ``readout_lr_power=1``
and it fails the gate on the logits (``beta = -1``). See ``results/cycle3_mup_coord_check_cpu.json``.

Precision of the polar step
---------------------------
The model, the gradients and the momentum are FP32. ``svd_exact`` runs in FP64. Every other
orthogonalizer emulates the low-precision GPU kernels on the CPU: operands are stored in the
working dtype, products accumulate in FP32, and each stored result is rounded once.

* ``SS3_*``: ``schatten_switch_reference(..., dtype=torch.float16)``, imported unchanged.
* ``PS3_55``: the ``baselines.PS3_55`` table in emulated FP16 (``polar_switch_reference``).
* ``polar_express``: the ``baselines`` quintics in emulated BF16, the dtype of the baseline
  (``polar_express_reference``).

Native FP16 / BF16 matrix products are 100 times slower than FP32 on a CPU, which is why the two
baselines are emulated instead of called. ``polar_dtype=torch.float64`` runs the same polynomial
in exact arithmetic and separates the polynomial from the rounding.

Run ``python -m research.polar_opt.mup_coord_check --out results/cycle3_mup_coord_check_cpu.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from multiprocessing import get_context
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from research.polar_opt.baselines import _AMSEL_COEFFICIENTS, _SWITCH_TABLES, _take_steps
from research.polar_opt.harness_tier2_real import git_info
from research.polar_opt.schatten_switch_blackwell import (
    SCHATTEN_SWITCH_CIRCUITS,
    schatten_switch_reference,
)

TINY = 1e-30  # values at or below this are not used in a logarithm
PASS_BETA = 0.15  # muP gate: every |beta| is below this
FAIL_BETA = 0.25  # negative control: some |beta| is at or above this
WIDTHS = (128, 256, 512, 1024)
SEEDS = (0, 1, 2)
PROBE_STEPS = (0, 1, 2, 4, 8, 12)
GATE_STEPS = (2, 4, 8, 12)
LR_GRIDS = {"muon": (0.005, 0.01, 0.02, 0.04, 0.08), "adamw": (0.001, 0.003, 0.01, 0.03)}
LR_WIDTHS = (128, 256, 512)
LR_STEPS = 20
ADAM_BETAS = (0.9, 0.95)
# The gradient of a hidden matrix is proportional to m * |W_head| and shrinks with the width. The
# usual eps = 1e-8 would damp the wide models more, so eps is far below every gradient here.
ADAM_EPS = 1e-12
MUON_BETA = 0.95


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    width: int = 128
    parameterization: str = "mup"  # "sp" or "mup"
    d_base: int = 128
    layers: int = 2
    head_dim: int = 32
    mlp_mult: int = 2
    vocab_size: int = 256
    seq_len: int = 32
    batch_size: int = 8
    num_batches: int = 4  # distinct seeded batches; training cycles through them
    readout_lr_power: float = 0.0  # muP readout lr = eta * (d_base / width) ** power

    def __post_init__(self):
        if self.parameterization not in ("sp", "mup"):
            raise ValueError("parameterization must be 'sp' or 'mup'")
        if self.width % self.head_dim or self.head_dim % 2:
            raise ValueError("width must be a multiple of an even head_dim")

    @property
    def mup(self) -> bool:
        return self.parameterization == "mup"

    @property
    def heads(self) -> int:
        return self.width // self.head_dim

    @property
    def attention_scale(self) -> float:
        return 1.0 / self.head_dim if self.mup else 1.0 / math.sqrt(self.head_dim)

    @property
    def logit_scale(self) -> float:
        return self.d_base / self.width if self.mup else 1.0


def _rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.width
        self.heads, self.scale = config.heads, config.attention_scale
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = (layer(x).view(batch, length, self.heads, -1).transpose(1, 2)
                   for layer in (self.q, self.k, self.v))
        scores = (_rope(q, cos, sin) @ _rope(k, cos, sin).mT) * self.scale
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        mixed = scores.masked_fill(future, float("-inf")).softmax(-1) @ v
        return self.o(mixed.transpose(1, 2).reshape(batch, length, width))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d, hidden = config.width, config.mlp_mult * config.width
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm, self.attn = nn.RMSNorm(config.width), Attention(config)
        self.mlp_norm, self.mlp = nn.RMSNorm(config.width), MLP(config)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class CoordTransformer(nn.Module):
    """Pre-RMSNorm causal Transformer. ``resid_final`` and ``logits`` are identity hook points."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.resid_final = nn.Identity()  # the residual stream before the final norm
        self.norm = nn.RMSNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.logits = nn.Identity()
        position = torch.arange(config.seq_len, dtype=torch.float32)
        frequency = 10000 ** (-torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        angles = position[:, None] * frequency
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def forward(self, tokens: Tensor) -> Tensor:
        length = tokens.shape[1]
        cos, sin = self.rope_cos[:length], self.rope_sin[:length]
        x = self.emb(tokens)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.logits(self.head(self.norm(self.resid_final(x))) * self.config.logit_scale)

    def probe_points(self) -> dict[str, nn.Module]:
        points = {"emb": self.emb}
        for i, block in enumerate(self.blocks):
            points[f"block{i}_attn_out"] = block.attn
            points[f"block{i}_mlp_out"] = block.mlp
        return points | {"resid_final": self.resid_final, "logits": self.logits}


def probe_names(config: ModelConfig) -> list[str]:
    blocks = [f"block{i}_{part}_out" for i in range(config.layers) for part in ("attn", "mlp")]
    return ["emb", *blocks, "resid_final", "logits"]


@dataclass(frozen=True)
class ParamSpec:
    """Initialization and step size of one parameter. ``lr_mult`` multiplies the AdamW learning
    rate (``eta``, or ``eta_aux`` under Muon); ``muon_scale`` multiplies ``eta * polar(M)``."""

    kind: str  # "embedding", "hidden", "norm" or "head"
    fan_out: int
    fan_in: int
    init_std: float
    lr_mult: float
    muon_scale: float | None = None


def parameter_specs(model: CoordTransformer) -> dict[str, ParamSpec]:
    """The parameterization table of the module docstring, for every parameter of ``model``."""
    config = model.config
    specs = {}
    for name, p in model.named_parameters():
        if p.ndim == 1:
            specs[name] = ParamSpec("norm", p.numel(), p.numel(), 1.0, 1.0)
            continue
        fan_out, fan_in = p.shape
        if name == "emb.weight":
            std = 1.0 / math.sqrt(config.d_base) if config.mup else 0.02
            specs[name] = ParamSpec("embedding", fan_out, fan_in, std, 1.0)
        elif name == "head.weight":
            std = 0.0 if config.mup else 0.02
            lr = (config.d_base / fan_in) ** config.readout_lr_power if config.mup else 1.0
            specs[name] = ParamSpec("head", fan_out, fan_in, std, lr)
        elif config.mup:
            specs[name] = ParamSpec("hidden", fan_out, fan_in, 1.0 / math.sqrt(fan_in),
                                    config.d_base / fan_in, math.sqrt(fan_out / fan_in))
        else:
            specs[name] = ParamSpec("hidden", fan_out, fan_in, 0.02, 1.0, 1.0)
    return specs


def build_model(config: ModelConfig, seed: int) -> CoordTransformer:
    """Model with seeded initial values. The values depend on the seed and the width only."""
    model = CoordTransformer(config)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, spec in parameter_specs(model).items():
            p = model.get_parameter(name)
            if spec.kind == "norm":
                p.fill_(1.0)
            elif spec.init_std == 0.0:
                p.zero_()
            else:
                p.normal_(0.0, spec.init_std, generator=generator)
    return model


def make_tokens(config: ModelConfig, seed: int) -> Tensor:
    """``(num_batches, batch_size, seq_len + 1)`` tokens of a seeded Markov chain.

    Each token has four successors with probabilities 0.5, 0.25, 0.15 and 0.1. The tokens depend
    on the seed and not on the width, so every width trains on the same data.
    """
    generator = torch.Generator().manual_seed(1_000_003 * (seed + 1))
    successors = torch.randint(config.vocab_size, (config.vocab_size, 4), generator=generator)
    count = config.num_batches * config.batch_size
    choice = torch.multinomial(torch.tensor([0.5, 0.25, 0.15, 0.1]), count * config.seq_len,
                               replacement=True, generator=generator).view(count, config.seq_len)
    tokens = torch.empty(count, config.seq_len + 1, dtype=torch.long)
    tokens[:, 0] = torch.randint(config.vocab_size, (count,), generator=generator)
    for t in range(config.seq_len):
        tokens[:, t + 1] = successors[tokens[:, t], choice[:, t]]
    return tokens.view(config.num_batches, config.batch_size, config.seq_len + 1)


# --------------------------------------------------------------------------------------
# Polar orthogonalizers
# --------------------------------------------------------------------------------------
NATIVE_POLAR_DTYPE = {"svd_exact": torch.float64, "polar_express": torch.bfloat16,
                      "PS3_55": torch.float16,
                      **{name: torch.float16 for name in SCHATTEN_SWITCH_CIRCUITS}}
POLAR_METHODS = tuple(NATIVE_POLAR_DTYPE)


def _wide(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _stored(X: Tensor, dtype: torch.dtype) -> Tensor:
    """Round ``X`` to the working dtype and return it in the accumulation dtype."""
    return X.to(dtype).to(_wide(dtype))


def svd_exact(G: Tensor, rtol: float | None = None) -> Tensor:
    """FP64 ``U @ Vh`` on the numerical row space of ``G``.

    Singular values at or below ``rtol * sigma_max`` are treated as zero. The default is the
    numerical rank at the precision of the input, ``max(rows, columns) * eps(G.dtype)``: a
    gradient of fewer tokens than the width has exact zeros that FP32 rounding fills with noise,
    and ``U @ Vh`` would turn that noise into unit singular values. A zero matrix returns zeros.
    """
    U, S, Vh = torch.linalg.svd(G.double(), full_matrices=False)
    if rtol is None:
        rtol = max(G.shape[-2:]) * torch.finfo(G.dtype).eps
    return (U * (S > rtol * S[..., :1])[..., None, :]) @ Vh


def polar_express_reference(G: Tensor, steps: int = 5, dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """``baselines.polar_express`` (Amsel et al.) for one matrix, with emulated ``dtype`` storage."""
    tall = G.shape[-2] > G.shape[-1]
    X = _stored(G.mT if tall else G, dtype)
    X = _stored(X / (1.01 * X.norm() + 1e-7), dtype)
    for a, b, c in _take_steps(_AMSEL_COEFFICIENTS, steps):
        A = _stored(X @ X.mT, dtype)
        B = _stored(b * A + _stored(c * A, dtype) @ A, dtype)
        X = _stored(a * X + B @ X, dtype)
    return X.mT if tall else X


def polar_switch_reference(G: Tensor, variant: str = "PS3_55", dtype: torch.dtype = torch.float16) -> Tensor:
    """``baselines.polar_switch`` (Du and Su) for one matrix, with emulated ``dtype`` storage."""
    table, _ = _SWITCH_TABLES[variant]
    tall = G.shape[-2] > G.shape[-1]
    X = G.to(_wide(dtype))
    X = _stored(X / (X.norm() + 1e-7), dtype)
    for residual, rows in table:
        view = X.mT if tall else X
        work = {0: _stored(view @ view.mT, dtype)}
        for dst, src, add, a, b, c in rows:
            value = a * (work[src] @ work[src].mT) + b * work[add]
            value.diagonal().add_(c)
            work[dst] = _stored(value, dtype)
        product = X @ work[dst] if tall else work[dst] @ X
        X = _stored(product + residual * X, dtype)
    return X


def orthogonalize(G: Tensor, method: str, dtype: torch.dtype | None = None) -> Tensor:
    """Approximate polar factor of one matrix, in the dtype of ``G``.

    ``dtype`` is the working dtype of the iteration (default ``NATIVE_POLAR_DTYPE[method]``).
    Zero and non-finite inputs return zeros.
    """
    if method not in NATIVE_POLAR_DTYPE:
        raise ValueError(f"Unknown orthogonalizer {method!r}; available: {POLAR_METHODS}")
    if G.ndim != 2:
        raise ValueError("Expected one matrix")
    dtype = NATIVE_POLAR_DTYPE[method] if dtype is None else dtype
    if not torch.isfinite(G).all() or not G.any():
        return torch.zeros_like(G)
    if method == "svd_exact":
        out = svd_exact(G)
    elif method == "polar_express":
        out = polar_express_reference(G, dtype=dtype)
    elif method == "PS3_55":
        out = polar_switch_reference(G, method, dtype=dtype)
    else:
        out = schatten_switch_reference(G, method, dtype=dtype)
    return out.to(G.dtype)


# --------------------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------------------
class Muon:
    """Nesterov Muon on the hidden matrices; AdamW on the embedding, the norms and the readout.

    ``dW = -lr * muon_scale * polar(lerp(g, M, beta))`` with ``M = lerp(M, g, 1 - beta)``.
    """

    def __init__(self, model: CoordTransformer, *, lr: float, lr_aux: float, polar: str,
                 polar_dtype: torch.dtype | None = None, beta: float = MUON_BETA):
        specs = parameter_specs(model)
        named = dict(model.named_parameters())
        self.lr, self.beta, self.polar, self.polar_dtype = lr, beta, polar, polar_dtype
        self.hidden = {name: (named[name], spec.muon_scale) for name, spec in specs.items()
                       if spec.kind == "hidden"}
        self.momentum = {name: torch.zeros_like(p) for name, (p, _) in self.hidden.items()}
        self.auxiliary = _adamw(model, lr_aux, [name for name in specs if name not in self.hidden])

    @torch.no_grad()
    def step(self) -> None:
        for name, (p, scale) in self.hidden.items():
            momentum = self.momentum[name].lerp_(p.grad, 1 - self.beta)
            update = orthogonalize(p.grad.lerp(momentum, self.beta), self.polar, self.polar_dtype)
            p.add_(update, alpha=-self.lr * scale)
        self.auxiliary.step()


def _adamw(model: CoordTransformer, lr: float, names=None) -> torch.optim.AdamW:
    specs = parameter_specs(model)
    groups = [{"params": [model.get_parameter(name)], "lr": lr * specs[name].lr_mult, "name": name}
              for name in (specs if names is None else names)]
    return torch.optim.AdamW(groups, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=0.0)


def build_optimizer(model: CoordTransformer, optimizer: str, *, lr: float, lr_aux: float | None = None,
                    polar: str = "svd_exact", polar_dtype: torch.dtype | None = None):
    """``adamw`` with per-parameter muP learning rates, or ``muon`` with the AdamW auxiliary."""
    if optimizer == "adamw":
        return _adamw(model, lr)
    if optimizer == "muon":
        if lr_aux is None:
            raise ValueError("Muon needs lr_aux")
        return Muon(model, lr=lr, lr_aux=lr_aux, polar=polar, polar_dtype=polar_dtype)
    raise ValueError("optimizer must be 'adamw' or 'muon'")


def resolve(parameterization: str, optimizer: str | None = None) -> tuple[str, str]:
    """``("sp" | "mup", "adamw" | "muon")``. ``"sp_muon"`` is ``("sp", "muon")``."""
    if parameterization == "sp_muon":
        return "sp", "muon"
    return parameterization, optimizer or "adamw"


# --------------------------------------------------------------------------------------
# Probes and one training run
# --------------------------------------------------------------------------------------
def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def rms(x: Tensor) -> float | None:
    """Root mean square in FP64; ``None`` when it is not finite."""
    return _finite(x.double().square().mean().sqrt().item())


def spectral_norm(M: Tensor, iterations: int = 40) -> float | None:
    """Power-iteration estimate (a lower bound) of the top singular value, in FP64.

    A zero matrix gives 0 and a non-finite matrix gives ``None``.
    """
    M = M.double()
    v = torch.randn(M.shape[1], dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    for _ in range(iterations):
        v = M.mT @ (M @ v)
        norm = v.norm()
        if not torch.isfinite(norm):
            return None
        if not norm > TINY:
            return 0.0
        v = v / norm
    return _finite((M @ v).norm().item())


class Probe:
    """Forward hooks that keep the output of every probe point for one input."""

    def __init__(self, model: CoordTransformer, tokens: Tensor):
        self.model, self.tokens, self.active, self.store = model, tokens, False, {}
        for name, module in model.probe_points().items():
            module.register_forward_hook(lambda _m, _i, out, name=name: self._keep(name, out))

    def _keep(self, name: str, out: Tensor) -> None:
        if self.active:
            self.store[name] = out.detach()

    @torch.no_grad()
    def capture(self) -> dict[str, Tensor]:
        self.active, self.store = True, {}
        try:
            self.model(self.tokens)
        finally:
            self.active = False
        return self.store


def gradients_finite(loss: Tensor, model: nn.Module) -> bool:
    return bool(torch.isfinite(loss)) and all(
        bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)


def _loss(model: CoordTransformer, batch: Tensor) -> Tensor:
    return F.cross_entropy(model(batch[:, :-1]).flatten(0, 1), batch[:, 1:].flatten())


def run_trial(config: ModelConfig, *, optimizer: str, lr: float, seed: int, steps: int,
              lr_aux: float | None = None, polar: str = "svd_exact",
              polar_dtype: torch.dtype | None = None, probe_steps=()) -> dict:
    """Train for ``steps`` updates and return the losses and, at ``probe_steps``, the statistics.

    Every returned number is finite or ``None``.

    ``activation_rms[name][i]``, ``delta_rms`` (against step 0) and ``step_delta_rms`` (against
    the previous step) belong to ``probe_steps[i]`` and to the first batch as the fixed probe
    input. ``updates[parameter]`` describes the update that reached that step: its RMS, for a
    matrix its spectral norm, and for a Muon matrix ``polar_mean_sq_sv``. A non-finite loss or
    gradient stops the run before the update: ``nonfinite_step`` is set and later entries are
    ``None``.
    """
    model = build_model(config, seed)
    tokens = make_tokens(config, seed)
    opt = build_optimizer(model, optimizer, lr=lr, lr_aux=lr_aux, polar=polar, polar_dtype=polar_dtype)
    specs = parameter_specs(model)
    names = probe_names(config)
    probe_steps = tuple(probe_steps)
    probe = Probe(model, tokens[0, :, :-1])
    stats = {key: {name: [None] * len(probe_steps) for name in names}
             for key in ("activation_rms", "delta_rms", "step_delta_rms")}
    updates = {}
    for name, p in model.named_parameters():
        keys = ["update_rms"] + ["update_spectral"] * (p.ndim == 2)
        keys += ["polar_mean_sq_sv"] * (optimizer == "muon" and specs[name].kind == "hidden")
        updates[name.removesuffix(".weight")] = {key: [None] * len(probe_steps) for key in keys}
    losses, nonfinite_step, first, previous = [], None, None, None

    for step in range(steps + 1):
        if probe_steps:
            current = probe.capture()
            first = first or current
            if step in probe_steps:
                i = probe_steps.index(step)
                for name in names:
                    stats["activation_rms"][name][i] = rms(current[name])
                    stats["delta_rms"][name][i] = rms(current[name] - first[name])
                    if previous is not None:
                        stats["step_delta_rms"][name][i] = rms(current[name] - previous[name])
            previous = current
        if step == steps:
            break
        loss = _loss(model, tokens[step % config.num_batches])
        model.zero_grad(set_to_none=True)
        loss.backward()
        if not gradients_finite(loss, model):
            nonfinite_step = step
            break
        losses.append(loss.item())
        record = step + 1 in probe_steps
        before = {name: p.detach().clone() for name, p in model.named_parameters()} if record else None
        opt.step()
        if record:
            i = probe_steps.index(step + 1)
            for name, p in model.named_parameters():
                delta, entry = p.detach() - before[name], updates[name.removesuffix(".weight")]
                entry["update_rms"][i] = rms(delta)
                if "update_spectral" in entry:
                    entry["update_spectral"][i] = spectral_norm(delta)
                if "polar_mean_sq_sv" in entry:
                    # Mean squared singular value of the polar output: 1 for an exact full-rank
                    # polar factor, lower for missing rank or under-lifted singular values.
                    entry["polar_mean_sq_sv"][i] = _finite(delta.double().square().sum().item() / (
                        min(delta.shape) * (lr * specs[name].muon_scale) ** 2))

    final_loss = None
    if nonfinite_step is None:
        with torch.no_grad():
            final_loss = _finite(torch.stack([_loss(model, batch) for batch in tokens]).mean().item())
    return {"width": config.width, "seed": seed, "lr": lr, "losses": losses, "final_loss": final_loss,
            "nonfinite_step": nonfinite_step, **stats, "updates": updates}


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def _valid(value) -> bool:
    return value is not None and math.isfinite(value) and value > TINY


def loglog_slope(widths, values) -> float | None:
    """OLS slope of ``log(values)`` on ``log(widths)``. ``None`` unless every value is finite and
    above ``TINY`` and there are two distinct widths."""
    if len(widths) != len(values) or len(set(widths)) < 2 or not all(_valid(v) for v in values):
        return None
    x = [math.log(w) for w in widths]
    y = [math.log(v) for v in values]
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    return (sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
            / sum((a - mean_x) ** 2 for a in x))


def max_over_min(values) -> float | None:
    return max(values) / min(values) if values and all(_valid(v) for v in values) else None


def geometric_mean(values) -> float | None:
    values = list(values)
    if not values or not all(_valid(v) for v in values):
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def width_scaling(trials: list[dict], key: str, steps=PROBE_STEPS) -> dict:
    """Per probe and step: seed geometric mean per width, ``beta`` and ``max_over_min_ratio``."""
    widths = sorted({t["width"] for t in trials})
    out = {}
    for name in trials[0][key]:
        out[name] = {}
        for i, step in enumerate(steps):
            by_width = [geometric_mean(t[key][name][i] for t in trials if t["width"] == w) for w in widths]
            out[name][str(step)] = {"by_width": by_width, "beta": loglog_slope(widths, by_width),
                                    "max_over_min_ratio": max_over_min(by_width)}
    return out


def gate(scaling: dict, gate_steps=GATE_STEPS) -> dict:
    """The coordinate-check gate on ``beta`` of ``delta_rms`` for every probe at ``gate_steps``.

    ``passed``: every beta exists and ``|beta| < PASS_BETA``. ``fails_clearly`` (required of a
    negative control): a beta is missing (zero or non-finite change) or ``|beta| >= FAIL_BETA``.
    """
    betas = {(name, step): scaling[name][str(step)]["beta"] for name in scaling for step in gate_steps}
    present = {key: value for key, value in betas.items() if value is not None}
    missing = len(betas) - len(present)
    worst = max(present, key=lambda key: abs(present[key]), default=None)
    max_abs = abs(present[worst]) if worst else None
    return {"passed": missing == 0 and max_abs is not None and max_abs < PASS_BETA,
            "fails_clearly": missing > 0 or (max_abs is not None and max_abs >= FAIL_BETA),
            "max_abs_beta": max_abs, "missing_betas": missing,
            "worst": {"probe": worst[0], "step": worst[1], "beta": present[worst]} if worst else None}


def update_scaling(trials: list[dict], steps=PROBE_STEPS) -> dict:
    """Per parameter and probe step: the seed geometric mean of each update statistic per width,
    and its width exponent ``beta``."""
    widths = sorted({t["width"] for t in trials})
    out = {}
    for name, entry in trials[0]["updates"].items():
        out[name] = {}
        for key in entry:
            out[name][key] = {}
            for i, step in enumerate(steps[1:], 1):
                by_width = [geometric_mean(t["updates"][name][key][i] for t in trials if t["width"] == w)
                            for w in widths]
                out[name][key][str(step)] = {"by_width": by_width, "beta": loglog_slope(widths, by_width)}
    return out


def summarize_coord(trials: list[dict], probe_steps=PROBE_STEPS, gate_steps=GATE_STEPS) -> dict:
    scaling = width_scaling(trials, "delta_rms", probe_steps)
    return {"gate": gate(scaling, gate_steps),
            "nonfinite_trials": sum(t["nonfinite_step"] is not None for t in trials),
            "delta_rms": scaling,
            "step_delta_rms": width_scaling(trials, "step_delta_rms", probe_steps),
            "activation_rms": width_scaling(trials, "activation_rms", probe_steps),
            "updates": update_scaling(trials, probe_steps)}


def summarize_lr_transfer(trials: list[dict], d_base: int) -> dict:
    """Seed-mean final loss per width and learning rate, and how the best learning rate moves.

    A diverged run counts as an infinite loss. ``best_lr_interpolated`` is the vertex of the
    parabola through the best grid point and its neighbours in ``log(lr)``. ``regret`` is the
    loss at the best learning rate of the base width minus the best loss of that width.
    """
    widths = sorted({t["width"] for t in trials})
    lrs = sorted({t["lr"] for t in trials})
    table, best, interpolated = {}, {}, {}
    for w in widths:
        row = []
        for lr in lrs:
            values = [t["final_loss"] for t in trials if t["width"] == w and t["lr"] == lr]
            finite = all(v is not None and math.isfinite(v) for v in values)
            row.append(sum(values) / len(values) if finite else math.inf)
        table[w] = row
        i = min(range(len(lrs)), key=row.__getitem__)
        best[w], interpolated[w] = lrs[i], lrs[i]
        if 0 < i < len(lrs) - 1 and all(math.isfinite(v) for v in row[i - 1:i + 2]):
            x0, x1, x2 = (math.log(lr) for lr in lrs[i - 1:i + 2])
            y0, y1, y2 = row[i - 1:i + 2]
            curvature = (y0 - y1) / ((x0 - x1) * (x0 - x2)) + (y2 - y1) / ((x2 - x1) * (x2 - x0))
            if curvature > 0:
                slope = (y2 - y1) / (x2 - x1) - curvature * (x2 - x1)  # derivative at x1
                interpolated[w] = math.exp(min(max(x1 - slope / (2 * curvature), x0), x2))
    base = d_base if d_base in best else widths[0]
    base_index = lrs.index(best[base])
    return {
        "widths": widths, "lr_grid": lrs,
        "final_loss": {str(w): table[w] for w in widths},
        "best_lr": {str(w): best[w] for w in widths},
        "best_loss": {str(w): min(table[w]) for w in widths},
        "best_lr_interpolated": {str(w): interpolated[w] for w in widths},
        "best_at_grid_edge": {str(w): best[w] in (lrs[0], lrs[-1]) for w in widths},
        "same_best_lr": len(set(best.values())) == 1,
        "best_lr_log2_shift": math.log2(best[widths[-1]] / best[widths[0]]),
        "best_lr_interpolated_log2_shift": math.log2(interpolated[widths[-1]] / interpolated[widths[0]]),
        "regret_at_base_best_lr": {str(w): table[w][base_index] - min(table[w]) for w in widths},
    }


def orthogonalizer_comparison(coord: dict, lr_transfer: dict, gate_steps=GATE_STEPS,
                              reference: str = "mup_muon/svd_exact") -> dict:
    """muP Muon per orthogonalizer: the gate, the distance of ``beta`` from ``svd_exact``, the
    lift of the update and the best learning rate per width.

    ``hidden_polar_mean_sq_sv[step]`` is the geometric mean over the hidden matrices of the mean
    squared singular value of the polar output, per width. A value below that of ``svd_exact``
    is under-lift of small singular values; its ``beta`` shows how the under-lift grows with width.
    """
    out = {}
    for name, entry in coord.items():
        if not name.startswith("mup_muon/"):
            continue
        row = {"gate_passed": entry["gate"]["passed"], "max_abs_beta": entry["gate"]["max_abs_beta"]}
        if reference in coord:
            mine, exact = entry["delta_rms"], coord[reference]["delta_rms"]
            pairs = [(mine[probe][str(step)]["beta"], exact[probe][str(step)]["beta"])
                     for probe in mine for step in gate_steps]
            row["max_abs_beta_difference_from_svd_exact"] = max(
                (abs(a - b) for a, b in pairs if a is not None and b is not None), default=None)
        widths = sorted({t["width"] for t in entry["trials"]})
        lifts = [kind["polar_mean_sq_sv"] for kind in entry["updates"].values() if "polar_mean_sq_sv" in kind]
        row["hidden_polar_mean_sq_sv"] = {}
        for step in gate_steps:
            by_width = [geometric_mean(lift[str(step)]["by_width"][i] for lift in lifts) for i in range(len(widths))]
            row["hidden_polar_mean_sq_sv"][str(step)] = {"by_width": by_width, "beta": loglog_slope(widths, by_width)}
        if name in lr_transfer:
            lr = lr_transfer[name]
            row |= {"best_lr": lr["best_lr"], "same_best_lr": lr["same_best_lr"],
                    "best_lr_interpolated_log2_shift": lr["best_lr_interpolated_log2_shift"],
                    "max_regret_at_base_best_lr": max(lr["regret_at_base_best_lr"].values())}
        out[name] = row
    return out


# --------------------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    """One row of the report: a parameterization, an optimizer and, for Muon, an orthogonalizer."""

    name: str
    parameterization: str
    optimizer: str
    polar: str | None = None
    polar_dtype: str | None = None  # None: NATIVE_POLAR_DTYPE
    readout_lr_power: float = 0.0
    expected: str = "open"  # "pass", "fail" or "open"
    lr_transfer: bool = True  # include in the learning-rate transfer check

    def config(self, base: ModelConfig, width: int) -> ModelConfig:
        return replace(base, width=width, parameterization=self.parameterization,
                       readout_lr_power=self.readout_lr_power)

    def polar_torch_dtype(self) -> torch.dtype | None:
        return getattr(torch, self.polar_dtype) if self.polar_dtype else None

    def describe(self) -> dict:
        out = asdict(self)
        if self.polar:
            out["polar_dtype"] = str(self.polar_torch_dtype() or NATIVE_POLAR_DTYPE[self.polar]).removeprefix("torch.")
        return out


def default_variants() -> list[Variant]:
    expected = {"svd_exact": "pass", "polar_express": "pass", "PS3_55": "pass"}
    return [
        Variant("sp_adamw", "sp", "adamw", expected="fail"),
        Variant("mup_adamw", "mup", "adamw", expected="pass"),
        Variant("mup_adamw_readout_lr_power1", "mup", "adamw", readout_lr_power=1.0, expected="fail",
                lr_transfer=False),
        Variant("sp_muon/svd_exact", "sp", "muon", "svd_exact", expected="fail"),
        *(Variant(f"mup_muon/{polar}", "mup", "muon", polar, expected=expected.get(polar, "open"))
          for polar in POLAR_METHODS),
        Variant("mup_muon_readout_lr_power1/PS3_55", "mup", "muon", "PS3_55",
                readout_lr_power=1.0, expected="fail", lr_transfer=False),
    ]


def _run_task(task: dict) -> tuple[dict, dict]:
    variant, base = task["variant"], task["base"]
    start = time.perf_counter()
    trial = run_trial(variant.config(base, task["width"]), optimizer=variant.optimizer, lr=task["lr"],
                      lr_aux=task["lr_aux"], seed=task["seed"], steps=task["steps"], polar=variant.polar or "svd_exact",
                      polar_dtype=variant.polar_torch_dtype(), probe_steps=task["probe_steps"])
    trial["seconds"] = time.perf_counter() - start
    return task, trial


def _worker_init() -> None:
    torch.set_num_threads(1)


def run_tasks(tasks: list[dict], jobs: int) -> list[tuple[dict, dict]]:
    """Run the tasks, widest first, with one BLAS thread each. The result order is the task order.

    One thread per task makes the numbers independent of ``jobs``.
    """
    order = sorted(range(len(tasks)), key=lambda i: -tasks[i]["width"] ** 3 * tasks[i]["steps"])
    results: dict[int, tuple[dict, dict]] = {}
    if jobs <= 1:
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for i in order:
                results[i] = _run_task(tasks[i])
        finally:
            torch.set_num_threads(threads)
    else:
        with ProcessPoolExecutor(jobs, mp_context=get_context("spawn"), initializer=_worker_init) as pool:
            futures = {pool.submit(_run_task, tasks[i]): i for i in order}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
    return [results[i] for i in range(len(tasks))]


def run(*, base: ModelConfig = ModelConfig(), variants: list[Variant] | None = None,
        widths=WIDTHS, seeds=SEEDS, probe_steps=PROBE_STEPS, gate_steps=GATE_STEPS,
        lr_adamw: float = 0.003, lr_muon: float = 0.02, lr_aux: float = 0.01,
        lr_grids: dict | None = None, lr_widths=LR_WIDTHS, lr_seeds=(0,), lr_steps: int = LR_STEPS,
        jobs: int = 1) -> dict:
    """Coordinate check and learning-rate transfer for every variant."""
    variants = default_variants() if variants is None else variants
    lr_grids = LR_GRIDS if lr_grids is None else lr_grids
    start = time.perf_counter()
    tasks = []
    for variant in variants:
        common = {"variant": variant, "base": base, "lr_aux": lr_aux}
        lr = lr_muon if variant.optimizer == "muon" else lr_adamw
        tasks += [{**common, "check": "coord", "width": w, "seed": s, "lr": lr, "steps": max(probe_steps),
                   "probe_steps": tuple(probe_steps)} for w in widths for s in seeds]
        if variant.lr_transfer:
            tasks += [{**common, "check": "lr", "width": w, "seed": s, "lr": lr, "steps": lr_steps,
                       "probe_steps": ()} for w in lr_widths for s in lr_seeds for lr in lr_grids[variant.optimizer]]
    results = run_tasks(tasks, jobs)

    coord, lr_transfer = {}, {}
    for variant in variants:
        mine = [(task, trial) for task, trial in results if task["variant"] == variant]
        trials = [trial for task, trial in mine if task["check"] == "coord"]
        if trials:
            summary = summarize_coord(trials, probe_steps, gate_steps)
            verdict = summary["gate"]
            matches = {"pass": verdict["passed"], "fail": verdict["fails_clearly"]}.get(variant.expected)
            coord[variant.name] = {
                **variant.describe(), "lr": trials[0]["lr"],
                "lr_aux": lr_aux if variant.optimizer == "muon" else None,
                "matches_expectation": matches, **summary,
                "trials": [{k: v for k, v in t.items() if k not in ("lr", "updates")} for t in trials]}
        trials = [trial for task, trial in mine if task["check"] == "lr"]
        if trials:
            lr_transfer[variant.name] = {
                **variant.describe(), "lr_aux": lr_aux if variant.optimizer == "muon" else None,
                **summarize_lr_transfer(trials, base.d_base),
                "trials": [{k: t[k] for k in ("width", "seed", "lr", "final_loss", "nonfinite_step", "seconds")}
                           | {"first_loss": t["losses"][0] if t["losses"] else None} for t in trials]}

    return {
        "schema": "cycle3_mup_coord_check/v1",
        **git_info(),
        "device": "cpu", "torch": torch.__version__, "threads_per_task": 1,
        "jobs": jobs, "elapsed_seconds": time.perf_counter() - start,
        "config": {
            "model": {k: v for k, v in asdict(base).items()
                      if k not in ("width", "parameterization", "readout_lr_power")},
            "widths": list(widths), "seeds": list(seeds), "probe_steps": list(probe_steps),
            "gate_steps": list(gate_steps), "pass_beta": PASS_BETA, "fail_beta": FAIL_BETA,
            "gate_statistic": "OLS slope of log(seed geometric mean of ||h(t) - h(0)||_RMS) on log(width)",
            "lr_adamw": lr_adamw, "lr_muon": lr_muon, "lr_aux": lr_aux,
            "adam_betas": list(ADAM_BETAS), "adam_eps": ADAM_EPS, "weight_decay": 0.0,
            "muon_beta": MUON_BETA, "muon_nesterov": True,
            "lr_transfer": {"lr_grids": {k: list(v) for k, v in lr_grids.items()}, "widths": list(lr_widths),
                            "seeds": list(lr_seeds), "steps": lr_steps,
                            "loss": "mean cross-entropy over the training batches after the last step"},
            "precision": {
                "model_and_gradients": "float32",
                "polar_native_dtype": {k: str(v).removeprefix("torch.") for k, v in NATIVE_POLAR_DTYPE.items()},
                "emulation": "operands stored in the working dtype, FP32 accumulation, one rounding per "
                             "stored result; svd_exact is FP64 on the numerical row space",
            },
        },
        "orthogonalizer_comparison": orthogonalizer_comparison(coord, lr_transfer, gate_steps),
        "coord_check": coord,
        "lr_transfer": lr_transfer,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _clean(value):
    """JSON value with six significant digits; non-finite numbers become ``None``."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float):
        return float(f"{value:.6g}") if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict) -> None:
    text = json.dumps(_clean(payload), indent=1, allow_nan=False)
    # Keep lists of scalars, and then objects without a nested object, on one line.
    for opening, closing, inner in (("[", "]", r"[^\[\]{}]"), ("{", "}", r"[^{}]")):
        pattern = re.escape(opening) + r"\s+(" + inner + r"*?)\s+" + re.escape(closing)
        text = re.sub(pattern, lambda m: opening + re.sub(r"\s+", " ", m.group(1)) + closing, text)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true",
                        help="smoke run: widths 128 and 256, one seed, short learning-rate check")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="worker processes, one BLAS thread each")
    parser.add_argument("--widths", type=int, nargs="+", default=list(WIDTHS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--lr-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--d-base", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--mlp-mult", type=int, default=2)
    parser.add_argument("--lr-adamw", type=float, default=0.003)
    parser.add_argument("--lr-muon", type=float, default=0.02)
    parser.add_argument("--lr-aux", type=float, default=0.01,
                        help="AdamW learning rate of the embedding, norms and readout under Muon")
    args = parser.parse_args(argv)

    base = ModelConfig(d_base=args.d_base, layers=args.layers, head_dim=args.head_dim, mlp_mult=args.mlp_mult)
    options = {"widths": tuple(args.widths), "seeds": tuple(args.seeds), "lr_seeds": tuple(args.lr_seeds)}
    if args.quick:
        options = {"widths": (128, 256), "seeds": (0,), "lr_seeds": (0,), "lr_widths": (128, 256), "lr_steps": 4,
                   "lr_grids": {"muon": (0.01, 0.04), "adamw": (0.003, 0.01)}}
    payload = run(base=base, lr_adamw=args.lr_adamw, lr_muon=args.lr_muon, lr_aux=args.lr_aux,
                  jobs=args.jobs, **options)
    payload["quick"] = args.quick
    write_json(args.out, payload)
    for name, entry in payload["coord_check"].items():
        verdict = entry["gate"]
        print(f"{name:40s} expected={entry['expected']:5s} passed={verdict['passed']!s:5s} "
              f"max|beta|={verdict['max_abs_beta']}")
    for name, entry in payload["lr_transfer"].items():
        print(f"{name:40s} best_lr={entry['best_lr']} same={entry['same_best_lr']}")
    return payload


if __name__ == "__main__":
    main()
