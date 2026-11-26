"""Real-text Tier-2 language-model benchmark for Muon orthogonalizers (Cycle 2, T3-01).

This harness replaces the Cycle 1 synthetic motif stream. It fixes the Tier-2 method flaws in
the Cycle 1 audits:

* Data: GPT-2 tokens from ``real_text_data.py``. Training samples come only from ``train.bin``.
  LR selection uses ``val.bin``. The final report uses ``test.bin``. The shards are disjoint
  by document, and the manifest checksums are checked before any run.
* Pairing: for one seed, every optimizer starts from the same initial weights (one CPU state
  dict, loaded into each run) and sees the same training batches (a NumPy stream seeded by the
  seed alone). Each row stores fingerprints of both.
* Update layout: with ``--update-dtype fp32`` (default) every Muon orthogonalizer output is
  converted once to a contiguous FP32 tensor. Its unbound slices then match the FP32 master
  parameters in dtype and stride, so ``torch._foreach_add_`` takes the single-kernel path for
  every comparator. ``native`` keeps each kernel's own dtype and strides, for an ablation.
* Schedule and grids: linear warmup times cosine decay, separate LR grids for AdamW and Muon,
  several seeds, and token or wall-clock budgets.
* Wall budgets: only model and optimizer compute counts against the budget. A step that ends
  after the deadline is undone (parameters are restored from a pre-step copy), so the reported
  model is the last one that finished before the deadline. Overshoot is zero by construction.
* Telemetry: CUDA-event time per step for fwd+bwd, polar orthogonalization, the FP32 layout cast,
  the foreach parameter update, and the whole optimizer step; tokens/s; validation and test loss
  and perplexity; git SHA; load average; GPU clocks, temperature and power from ``nvidia-smi``.
* Statistics: for each budget, the best LR per optimizer is chosen on validation loss. Test
  loss and throughput are then compared per seed against a reference optimizer, with the mean
  and a 95% Student-t interval.

Optimizers: ``adamw`` (fused) and Muon with ``jordan_eager``, ``pe_eager``, ``pe_compiled``,
``pe_triton``, ``ps2_eager``, ``ps3_eager``, ``ps2_blackwell`` and ``ps3_blackwell``
orthogonalizers. Register more with ``--extra-optimizer NAME=MODULE:FACTORY``. A FACTORY with no
required arguments is called once per run and must return ``f(G) -> U`` for a stacked FP32
``G`` of shape (batch, rows, columns). Any other callable is used directly as ``f``.

Examples:
    PYTHONPATH=. .venv/bin/python research/polar_opt/harness_tier2_real.py --preset 50m \\
        --optimizers adamw muon_pe_eager muon_ps3_blackwell --seeds 0 1 2 --budget both
    PYTHONPATH=. .venv/bin/python research/polar_opt/harness_tier2_real.py --preset 50m \\
        --harvest-nesterov
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from research.polar_opt.harness_tier2_lm import Transformer
from research.polar_opt.real_text_data import DEFAULT_DIR, PADDED_VOCAB_SIZE, verify_manifest

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------
# Model configuration
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """Field names match ``harness_tier2_lm.Transformer``; batch fields are added here."""

    vocab_size: int = PADDED_VOCAB_SIZE
    width: int = 512
    layers: int = 12
    heads: int = 8
    hidden: int = 1408
    sequence_length: int = 512
    micro_batch: int = 16
    grad_accum: int = 4

    @property
    def step_tokens(self) -> int:
        return self.micro_batch * self.grad_accum * self.sequence_length


PRESETS = {
    "tiny": ModelConfig(width=64, layers=2, heads=4, hidden=128, sequence_length=64,
                        micro_batch=4, grad_accum=2),
    # 38.5M non-embedding + 25.8M tied embedding; 32,768 tokens per optimizer step.
    "50m": ModelConfig(width=512, layers=12, heads=8, hidden=1408, sequence_length=512,
                       micro_batch=16, grad_accum=4),
    # GPT-2 small scale: 85M non-embedding + 38.6M tied embedding; 32,768 tokens per step.
    # Micro-batch 8 keeps peak memory near 7 GB (16 x 4 peaks at 11.3 GB on a 16 GB card).
    "120m": ModelConfig(width=768, layers=12, heads=12, hidden=2048, sequence_length=512,
                        micro_batch=8, grad_accum=8),
}


def parameter_counts(config: ModelConfig) -> dict:
    with torch.device("meta"):
        model = Transformer(config)
    total = sum(p.numel() for p in model.parameters())
    embedding = model.embedding.weight.numel()
    return {"total": total, "embedding": embedding, "non_embedding": total - embedding,
            "muon_matrices": sum(p.numel() for p in model.matrix_parameters().values())}


def initial_state(config: ModelConfig, seed: int) -> dict[str, Tensor]:
    """CPU initial weights for ``seed``. Every optimizer of that seed loads this one dict."""
    torch.manual_seed(seed)
    return {k: v.detach().clone() for k, v in Transformer(config).state_dict().items()}


def state_fingerprint(state: dict[str, Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
class ShardData:
    """Read-only memmaps of the verified train/val/test shards."""

    def __init__(self, directory: Path = DEFAULT_DIR, verify: bool = True):
        directory = Path(directory)
        self.manifest = (verify_manifest(directory) if verify
                         else json.loads((directory / "manifest.json").read_text()))
        self.splits = {name: np.memmap(directory / entry["file"], dtype=np.uint16, mode="r")
                       for name, entry in self.manifest["splits"].items()}

    @property
    def train(self) -> np.ndarray:
        return self.splits["train"]

    def summary(self) -> dict:
        return {"source": self.manifest["source"], "encoding": self.manifest["encoding"],
                "splits": {k: {"tokens": v["tokens"], "sha256": v["sha256"]}
                           for k, v in self.manifest["splits"].items()}}


class BatchStream:
    """Uniform random training windows. The stream depends only on (tokens, config, seed)."""

    def __init__(self, tokens: np.ndarray, config: ModelConfig, seed: int, device: torch.device):
        self.tokens, self.config, self.device = tokens, config, device
        self.rng = np.random.default_rng(seed)
        self.span = np.arange(config.sequence_length + 1)

    def next_starts(self) -> np.ndarray:
        count = self.config.grad_accum * self.config.micro_batch
        return self.rng.integers(0, len(self.tokens) - len(self.span) + 1, size=count)

    def next_batch(self) -> tuple[Tensor, Tensor]:
        """Return (inputs, targets), each (grad_accum, micro_batch, sequence_length) int64."""
        cfg = self.config
        windows = torch.from_numpy(self.tokens[self.next_starts()[:, None] + self.span].astype(np.int64))
        windows = windows.view(cfg.grad_accum, cfg.micro_batch, -1)
        if self.device.type == "cuda":
            windows = windows.pin_memory()
        windows = windows.to(self.device)
        return windows[..., :-1], windows[..., 1:]


def stream_fingerprint(tokens: np.ndarray, config: ModelConfig, seed: int, steps: int = 3) -> str:
    stream = BatchStream(tokens, config, seed, torch.device("cpu"))
    return hashlib.sha256(b"".join(stream.next_starts().tobytes() for _ in range(steps))).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------------------
def _adamw(parameters, lr: float, weight_decay: float, cuda: bool) -> torch.optim.AdamW:
    """Fused AdamW. Matrices and the embedding decay; norm weights do not."""
    parameters = list(parameters)
    groups = [{"params": [p for p in parameters if p.ndim >= 2], "weight_decay": weight_decay},
              {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0}]
    groups = [g for g in groups if g["params"]]
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=cuda)
    for group in optimizer.param_groups:
        group["base_lr"] = lr
    return optimizer


class StepTimer:
    """CUDA-event marks inside one step. The time between two marks belongs to the later label."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.marks: list[tuple[str, torch.cuda.Event]] = []

    def mark(self, label: str) -> None:
        if self.enabled:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.marks.append((label, event))

    def collect(self) -> dict[str, float]:
        """Return per-label milliseconds. Call only after the step has synchronized."""
        out: dict[str, float] = defaultdict(float)
        for (_, a), (label, b) in zip(self.marks, self.marks[1:]):
            out[label] += a.elapsed_time(b)
        self.marks = []
        return dict(out)


class AdamW:
    """Fused AdamW on every parameter, with the same interface as ``Muon``."""

    def __init__(self, model, *, lr: float, weight_decay: float):
        self.inner = _adamw(model.parameters(), lr, weight_decay, next(model.parameters()).is_cuda)

    def set_lr_multiplier(self, multiplier: float) -> None:
        for group in self.inner.param_groups:
            group["lr"] = group["base_lr"] * multiplier

    @torch.no_grad()
    def step(self, timer: StepTimer | None = None) -> None:
        self.inner.step()
        if timer:
            timer.mark("adamw_fused")


class Muon:
    """Muon with Nesterov momentum on hidden matrices; fused AdamW on embedding and norms.

    ``orthogonalize`` receives the stacked post-Nesterov matrix of each same-shape group in FP32.
    ``update_dtype="fp32"`` converts its result to a contiguous FP32 tensor in one pass, so the
    unbound updates match the FP32 parameters and ``_foreach_add_`` stays on its fast path.
    ``capture(names, nesterov)``, when set, sees the exact orthogonalizer input of each group.
    """

    def __init__(self, model, orthogonalize, *, lr: float, auxiliary_lr: float, weight_decay: float,
                 beta: float = 0.95, update_dtype: str = "fp32"):
        if update_dtype not in ("fp32", "native"):
            raise ValueError("update_dtype must be 'fp32' or 'native'")
        self.orthogonalize, self.update_dtype = orthogonalize, update_dtype
        self.base_lr = self.lr = lr
        self.weight_decay, self.beta = weight_decay, beta
        self.capture = None
        self.matrices = model.matrix_parameters()
        self.momentum = {name: torch.zeros_like(p) for name, p in self.matrices.items()}
        groups = defaultdict(list)
        for name, p in self.matrices.items():
            groups[tuple(p.shape)].append(name)
        self.groups = list(groups.values())
        matrix_ids = {id(p) for p in self.matrices.values()}
        self.auxiliary = _adamw([p for p in model.parameters() if id(p) not in matrix_ids],
                                auxiliary_lr, weight_decay, next(model.parameters()).is_cuda)

    def set_lr_multiplier(self, multiplier: float) -> None:
        self.lr = self.base_lr * multiplier
        for group in self.auxiliary.param_groups:
            group["lr"] = group["base_lr"] * multiplier

    def prepare(self, updates: Tensor) -> Tensor:
        if self.update_dtype == "fp32":
            return updates.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        return updates

    @torch.no_grad()
    def step(self, timer: StepTimer | None = None) -> None:
        mark = timer.mark if timer else (lambda label: None)
        for names in self.groups:
            parameters = [self.matrices[name] for name in names]
            gradients = [p.grad for p in parameters]
            momenta = [self.momentum[name] for name in names]
            torch._foreach_lerp_(momenta, gradients, 1 - self.beta)
            nesterov = torch.stack(torch._foreach_lerp(gradients, momenta, self.beta))
            if self.capture is not None:
                self.capture(names, nesterov)
            mark("momentum")
            updates = self.orthogonalize(nesterov)
            mark("polar_ortho")
            updates = self.prepare(updates)
            mark("layout_cast")
            rows, columns = nesterov.shape[-2:]
            torch._foreach_mul_(parameters, 1 - self.lr * self.weight_decay)
            torch._foreach_add_(parameters, list(updates.unbind()),
                                alpha=-self.lr * math.sqrt(max(1, rows / columns)))
            mark("foreach_update")
        self.auxiliary.step()
        mark("auxiliary_adamw")


def _partial(function, *args, **kwargs):
    return functools.partial(function, *args, **kwargs)


@functools.cache
def _compiled_polar_express():
    from research.polar_opt.baselines import polar_express

    torch._dynamo.config.cache_size_limit = max(getattr(torch._dynamo.config, "cache_size_limit", 8), 32)
    return torch.compile(polar_express, dynamic=False)


def builtin_factories() -> dict:
    """Return name -> zero-argument factory of an orthogonalizer (None for AdamW)."""
    from research.polar_opt import tier2_candidates as candidates
    from research.polar_opt.baselines import jordan_ns5, polar_express, polar_switch

    return {
        "adamw": None,
        "muon_jordan_eager": lambda: jordan_ns5,
        "muon_pe_eager": lambda: polar_express,
        "muon_pe_compiled": _compiled_polar_express,
        "muon_pe_triton": lambda: candidates.pe_triton_bf16,
        "muon_ps2_eager": lambda: _partial(polar_switch, variant="PS2_55"),
        "muon_ps3_eager": lambda: _partial(polar_switch, variant="PS3_55"),
        "muon_ps2_blackwell": lambda: candidates.ps2_blackwell_fp16,
        "muon_ps3_blackwell": lambda: candidates.ps3_blackwell_fp16,
    }


DEFAULT_OPTIMIZERS = ["adamw", "muon_pe_eager", "muon_pe_compiled", "muon_pe_triton",
                      "muon_ps2_eager", "muon_ps3_eager", "muon_ps2_blackwell", "muon_ps3_blackwell"]


def _requires_arguments(obj) -> bool:
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return True
    params = list(signature.parameters.values())
    if any(
        p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        for p in params
    ):
        return True
    return any(p.kind == p.VAR_POSITIONAL for p in params)


def load_factory(spec: str):
    """``MODULE:ATTR`` -> zero-argument factory. A zero-argument ATTR is the factory itself;
    any other callable (``f(G)``) is wrapped so each run receives it unchanged."""
    module, _, attribute = spec.partition(":")
    if not module or not attribute:
        raise ValueError(f"Expected MODULE:FACTORY, received {spec!r}")
    obj = getattr(importlib.import_module(module), attribute)
    if not callable(obj):
        raise ValueError(f"{spec} is not callable")
    if inspect.isclass(obj) and _requires_arguments(obj):
        raise ValueError(f"{spec} is a class that needs arguments; wrap it in a zero-argument factory")
    return (lambda: obj) if _requires_arguments(obj) else obj


def parse_extra(entries: list[str], factories: dict) -> list[str]:
    names = []
    for entry in entries:
        name, sep, spec = entry.partition("=")
        if not sep or not name or name in factories:
            raise ValueError(f"Duplicate or malformed --extra-optimizer {entry!r}")
        factories[name] = load_factory(spec)
        names.append(name)
    return names


def build_optimizer(name: str, factory, model, *, lr: float, auxiliary_lr: float,
                    weight_decay: float, update_dtype: str):
    if factory is None:
        return AdamW(model, lr=lr, weight_decay=weight_decay)
    return Muon(model, factory(), lr=lr, auxiliary_lr=auxiliary_lr, weight_decay=weight_decay,
                update_dtype=update_dtype)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def lr_multiplier(
    step: int,
    progress: float,
    warmup_steps: int,
    min_ratio: float,
    *,
    total_steps: int | None = None,
    warmup_fraction: float | None = None,
) -> float:
    """Linear warmup decoupled from cosine decay down to ``min_ratio``.

    In token mode (``warmup_fraction is None``), warmup spans ``step < warmup_steps`` and cosine
    decay follows across ``[warmup_steps, total_steps]`` (or ``progress`` if ``total_steps`` is
    omitted). In wall mode (``warmup_fraction`` given), warmup spans ``progress < warmup_fraction``
    so peak LR stays independent of optimizer throughput.
    """
    if warmup_fraction is not None and warmup_fraction > 0 and warmup_steps > 0:
        p = min(max(progress, 0.0), 1.0)
        if p < warmup_fraction:
            return min(1.0, max(1.0 / max(1, warmup_steps), p / warmup_fraction))
        decay_p = min(max((p - warmup_fraction) / max(1e-9, 1.0 - warmup_fraction), 0.0), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * decay_p))
    if warmup_steps > 0 and step < warmup_steps:
        return min(1.0, (step + 1) / warmup_steps)
    if total_steps is not None:
        decay_p = min(max((step - warmup_steps) / max(1, total_steps - warmup_steps), 0.0), 1.0)
    else:
        decay_p = min(max(progress, 0.0), 1.0)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * decay_p))


def _loss(model, x: Tensor, y: Tensor, reduction: str = "mean") -> Tensor:
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        logits = model(x)
        return F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction=reduction)  # FP32 under autocast


def train_step(forward, batch: tuple[Tensor, Tensor], optimizer, timer: StepTimer | None = None) -> Tensor:
    """One optimizer step over ``grad_accum`` micro-batches. Returns the mean loss (device)."""
    inputs, targets = batch
    if timer:
        timer.mark("start")
    total = torch.zeros((), device=inputs.device)
    for x, y in zip(inputs, targets):
        loss = _loss(forward, x, y) / len(inputs)
        loss.backward()
        total += loss.detach()
    if timer:
        timer.mark("fwd_bwd")
    optimizer.step(timer)
    return total


@torch.no_grad()
def evaluate(forward, tokens: np.ndarray, config: ModelConfig, eval_tokens: int, device: torch.device) -> float:
    """Mean next-token loss on the first ``eval_tokens`` of a split, in disjoint windows."""
    seq, batch = config.sequence_length, config.micro_batch
    windows = min(eval_tokens, len(tokens) - 1) // seq // batch * batch
    if windows == 0:
        raise ValueError("Evaluation split is shorter than one batch")
    total = torch.zeros((), dtype=torch.float64, device=device)
    offsets = np.arange(seq + 1)
    for first in range(0, windows, batch):
        starts = np.arange(first, first + batch) * seq
        chunk = torch.from_numpy(tokens[starts[:, None] + offsets].astype(np.int64)).to(device)
        total += _loss(forward, chunk[:, :-1], chunk[:, 1:], reduction="sum").double()
    return (total / (windows * seq)).item()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_telemetry(device: torch.device) -> dict | None:
    """One ``nvidia-smi`` sample: SM/memory clocks (MHz), temperature (C), power (W), etc."""
    if device.type != "cuda":
        return None
    fields = ["clocks.sm", "clocks.mem", "temperature.gpu", "power.draw", "utilization.gpu",
              "pstate", "memory.used", "clocks_event_reasons.active"]
    uuid = f"GPU-{torch.cuda.get_device_properties(device).uuid}"
    try:
        out = subprocess.run(["nvidia-smi", "-i", uuid, f"--query-gpu={','.join(fields)}",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": f"{type(error).__name__}: {error}"}
    sample = {"time": time.time()}
    for key, value in zip(fields, (v.strip() for v in out.split(","))):
        try:
            sample[key] = float(value)
        except ValueError:
            sample[key] = value
    return sample


def git_info() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                                  timeout=10, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(status) if status is not None else None}


def _median_ms(samples: list[dict], key: str) -> float | None:
    if not any(key in s for s in samples):
        return None
    return median(s.get(key, 0.0) for s in samples)


def summarize_timing(samples: list[dict]) -> dict:
    """Median per-step CUDA-event milliseconds by segment."""
    if not samples:
        return {}
    optimizer = [sum(v for k, v in s.items() if k not in ("fwd_bwd", "step_wall_ms")) for s in samples]
    return {
        "steps_timed": len(samples),
        "fwd_bwd_ms": _median_ms(samples, "fwd_bwd"),
        "polar_ortho_ms": _median_ms(samples, "polar_ortho"),
        "layout_cast_ms": _median_ms(samples, "layout_cast"),
        "foreach_update_ms": _median_ms(samples, "foreach_update"),
        "momentum_ms": _median_ms(samples, "momentum"),
        "auxiliary_adamw_ms": _median_ms(samples, "auxiliary_adamw"),
        "adamw_fused_ms": _median_ms(samples, "adamw_fused"),
        "optimizer_step_ms": median(optimizer),
        "step_wall_ms": _median_ms(samples, "step_wall_ms"),
    }


def run_trial(config: ModelConfig, data: ShardData, *, device: torch.device, seed: int,
              optimizer_name: str, factory, lr: float, auxiliary_lr: float = 2e-3,
              weight_decay: float = 0.01, update_dtype: str = "fp32", tokens: int | None = None,
              seconds: float | None = None, lr_warmup_steps: int = 50,
              wall_warmup_fraction: float = 0.0625, min_lr_ratio: float = 0.1,
              compile_warmup_steps: int = 3, eval_tokens: int = 1 << 20, init: dict | None = None,
              compile_model: bool = False, reusable_model: tuple[Transformer, object] | None = None,
              log_every: int = 10, telemetry_every: int = 100, clock=time.perf_counter) -> dict:
    """Train one fresh (optimizer, LR, seed) trial under exactly one budget and evaluate it."""
    if (tokens is None) == (seconds is None):
        raise ValueError("Specify exactly one of tokens or seconds")
    step_tokens = config.step_tokens
    if tokens is not None and (tokens <= 0 or tokens % step_tokens):
        raise ValueError(f"Token budget must be a positive multiple of {step_tokens}")
    if seconds is not None and not (math.isfinite(seconds) and seconds > 0):
        raise ValueError("Wall budget must be positive and finite")
    budget = "tokens" if tokens is not None else "wall"
    init = init if init is not None else initial_state(config, seed)
    if reusable_model is not None:
        model, forward = reusable_model
    else:
        with torch.device(device):
            model = Transformer(config)
        if compile_model:
            torch._dynamo.config.cache_size_limit = max(getattr(torch._dynamo.config, "cache_size_limit", 8), 32)
            forward = torch.compile(model)
        else:
            forward = model
    model.load_state_dict(init)
    model.train()
    settings = dict(lr=lr, auxiliary_lr=auxiliary_lr, weight_decay=weight_decay, update_dtype=update_dtype)

    # Untimed warmup on a disposable stream: compilation, autotuning and plan caches.
    if compile_warmup_steps:
        warm_optimizer = build_optimizer(optimizer_name, factory, model, **settings)
        warm_stream = BatchStream(data.train, config, seed + 1_000_000, device)
        for _ in range(compile_warmup_steps):
            train_step(forward, warm_stream.next_batch(), warm_optimizer)
            model.zero_grad(set_to_none=True)
        _sync(device)
        del warm_optimizer, warm_stream
        model.load_state_dict(init)

    optimizer = build_optimizer(optimizer_name, factory, model, **settings)
    stream = BatchStream(data.train, config, seed, device)
    parameters = list(model.parameters())
    snapshot = [p.detach().clone() for p in parameters] if budget == "wall" else None
    timer = StepTimer(device.type == "cuda")
    total_steps = tokens // step_tokens if tokens is not None else None

    row = {"optimizer": optimizer_name, "lr": lr, "seed": seed, "budget": budget,
           "update_dtype": update_dtype if factory is not None else None,
           "auxiliary_lr": auxiliary_lr if factory is not None else None,
           "init_fingerprint": state_fingerprint(init),
           "stream_fingerprint": stream_fingerprint(data.train, config, seed),
           "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "loadavg_start": os.getloadavg(),
           "gpu_start": gpu_telemetry(device)}
    gpu_samples, timing, curve = [], [], []
    elapsed, steps, discarded, last_loss, diverged = 0.0, 0, None, None, False
    while total_steps is None or steps < total_steps:
        if seconds is not None and elapsed >= seconds:
            break
        progress = steps / total_steps if total_steps else elapsed / seconds
        warmup_frac = wall_warmup_fraction if seconds is not None else None
        optimizer.set_lr_multiplier(
            lr_multiplier(
                steps,
                progress,
                lr_warmup_steps,
                min_lr_ratio,
                total_steps=total_steps,
                warmup_fraction=warmup_frac,
            )
        )
        batch = stream.next_batch()
        if snapshot is not None:
            torch._foreach_copy_(snapshot, [p.detach() for p in parameters])
        _sync(device)
        start = clock()
        loss = train_step(forward, batch, optimizer, timer)
        _sync(device)
        step_seconds = clock() - start
        loss = loss.item()
        model.zero_grad(set_to_none=True)
        segments = timer.collect()
        if seconds is not None and elapsed + step_seconds > seconds:
            # The step finished after the deadline: undo it and report the pre-step model.
            with torch.no_grad():
                torch._foreach_copy_([p.detach() for p in parameters], snapshot)
            discarded = step_seconds
            break
        elapsed += step_seconds
        steps += 1
        timing.append({**segments, "step_wall_ms": step_seconds * 1e3})
        if not math.isfinite(loss):
            diverged = True
            break
        last_loss = loss
        if steps % log_every == 0 or steps == 1:
            curve.append((steps, round(loss, 5)))
        if telemetry_every and steps % telemetry_every == 0:
            gpu_samples.append(gpu_telemetry(device))

    row["gpu_end"] = gpu_telemetry(device)
    row["loadavg_end"] = os.getloadavg()
    model.eval()
    val = evaluate(model, data.splits["val"], config, eval_tokens, device) if not diverged else float("nan")
    test = evaluate(model, data.splits["test"], config, eval_tokens, device) if not diverged else float("nan")
    finite = math.isfinite(val) and math.isfinite(test)
    row.update({
        "steps": steps, "tokens": steps * step_tokens, "target_tokens": tokens, "target_seconds": seconds,
        "train_seconds": elapsed, "discarded_step_seconds": discarded,
        "wall_budget_overshoot_seconds": 0.0 if seconds is not None else None,
        "tokens_per_second": steps * step_tokens / elapsed if elapsed and steps > 0 else None,
        "last_train_loss": last_loss,
        "val_loss": val if finite else None, "val_ppl": math.exp(val) if finite else None,
        "test_loss": test if finite else None, "test_ppl": math.exp(test) if finite else None,
        "diverged": diverged or not finite or steps == 0,
        "timing": summarize_timing(timing[1:] if len(timing) > 2 else timing),
        "train_curve": curve, "gpu_samples": gpu_samples,
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    return row


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def t_interval(values: list[float], level: float = 0.95) -> dict:
    """Mean and two-sided Student-t confidence interval."""
    from scipy.stats import t

    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "ci95": None}
    m = mean(values)
    if n == 1:
        return {"n": 1, "mean": m, "sd": None, "ci95": None}
    sd = stdev(values)
    half = t.ppf(0.5 + level / 2, n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "ci95": [m - half, m + half]}


def _mean_or_none(values) -> float | None:
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def summarize(rows: list[dict], reference: str) -> dict:
    """Best LR per (budget, optimizer) by mean validation loss, then paired test comparisons."""
    cells = defaultdict(list)
    for row in rows:
        cells[(row["budget"], row["optimizer"], row["update_dtype"], row["lr"])].append(row)
    grid = []
    for (budget, name, layout, lr), group in sorted(cells.items(), key=lambda kv: str(kv[0])):
        ok = [r for r in group if not r["diverged"]]
        grid.append({"budget": budget, "optimizer": name, "update_dtype": layout, "lr": lr,
                     "runs": len(group), "diverged": len(group) - len(ok),
                     "val_loss_mean": mean(r["val_loss"] for r in ok) if len(ok) == len(group) else None,
                     "test_loss_mean": mean(r["test_loss"] for r in ok) if len(ok) == len(group) else None,
                     "tokens_per_second_mean": _mean_or_none([r["tokens_per_second"] for r in group])})
    best = {}
    group_keys = sorted(
        {(g["budget"], g["optimizer"], g["update_dtype"]) for g in grid},
        key=lambda k: (k[0], k[1], "" if k[2] is None else k[2]),
    )
    for budget, name, layout in group_keys:
        options = [g for g in grid if (g["budget"], g["optimizer"], g["update_dtype"]) == (budget, name, layout)]
        scored = [g for g in options if g["val_loss_mean"] is not None]
        if not scored:
            continue
        chosen = min(scored, key=lambda g: g["val_loss_mean"])
        lrs = sorted(g["lr"] for g in options)
        best[(budget, name, layout)] = {**chosen, "lr_grid": lrs,
                                        "lr_at_grid_edge": len(lrs) < 3 or chosen["lr"] in (lrs[0], lrs[-1])}
    comparisons = []
    for (budget, name, layout), choice in sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][1], "" if kv[0][2] is None else kv[0][2])):
        ref_key = next(
            (cand for cand in ((budget, reference, layout), (budget, reference, "fp32"), (budget, reference, None)) if cand in best),
            None,
        )
        if ref_key is None or ref_key == (budget, name, layout):
            continue
        ref_rows = cells[(budget, reference, ref_key[2], best[ref_key]["lr"])]
        mine_rows = cells[(budget, name, layout, choice["lr"])]
        ref = {r["seed"]: r for r in ref_rows}
        mine = {r["seed"]: r for r in mine_rows}
        if len(ref) != len(ref_rows) or len(mine) != len(mine_rows):
            raise ValueError(f"Duplicate seed found in cell {(budget, name, layout, choice['lr'])}")
        seeds = sorted(set(ref) & set(mine))
        tps_ratios = [
            mine[s]["tokens_per_second"] / ref[s]["tokens_per_second"]
            for s in seeds
            if mine[s].get("tokens_per_second") and ref[s].get("tokens_per_second")
        ]
        comparisons.append({
            "budget": budget, "optimizer": name, "update_dtype": layout, "lr": choice["lr"],
            "reference": reference, "reference_lr": best[ref_key]["lr"], "reference_update_dtype": ref_key[2],
            "seeds": seeds,
            "test_loss_diff": t_interval([mine[s]["test_loss"] - ref[s]["test_loss"] for s in seeds]),
            "val_loss_diff": t_interval([mine[s]["val_loss"] - ref[s]["val_loss"] for s in seeds]),
            "tokens_per_second_ratio": t_interval(tps_ratios),
        })
    return {"grid": grid, "best_lr": [{"budget": k[0], "optimizer": k[1], **v} for k, v in best.items()],
            "paired_vs_reference": comparisons,
            "note": "Negative loss differences favour the optimizer. LR is chosen on validation loss "
                    "only; test loss is reported for the chosen LR."}


# --------------------------------------------------------------------------------------
# Nesterov harvesting
# --------------------------------------------------------------------------------------
def harvest_nesterov(config: ModelConfig, data: ShardData, *, device: torch.device, factory, lr: float,
                     auxiliary_lr: float, weight_decay: float, steps: int, seed: int = 999,
                     lr_warmup_steps: int = 50, min_lr_ratio: float = 0.1, starts: list[int] | None = None,
                     layers: list[int] | None = None) -> dict:
    """Train one held-out-seed trajectory and keep consecutive post-Nesterov Muon inputs.

    Each pair is (N_t, N_{t+1}) for one matrix, where N is the exact tensor passed to the
    orthogonalizer: lerp(gradient, momentum, beta) after the momentum update. Singular values
    are exact FP64 (``torch.linalg.svdvals`` on ``.double()``).
    """
    starts = starts or sorted({max(1, steps // 20), steps // 2, steps - 1})
    if steps < 2 or min(starts) < 1 or max(starts) + 1 > steps:
        raise ValueError("Each start t needs 1 <= t and t + 1 <= steps")
    layers = layers if layers is not None else sorted({0, config.layers // 2, config.layers - 1})
    torch.manual_seed(seed)
    with torch.device(device):
        model = Transformer(config)
    optimizer = Muon(model, factory(), lr=lr, auxiliary_lr=auxiliary_lr, weight_decay=weight_decay)
    stream = BatchStream(data.train, config, seed, device)
    capture_steps = {t for s in starts for t in (s, s + 1)}
    captured: dict[int, dict[str, Tensor]] = {}

    def capture_into(step):
        def capture(names, nesterov):
            for i, name in enumerate(names):
                if int(name.split(".")[0]) in layers:
                    captured.setdefault(step, {})[name] = nesterov[i].detach().float().cpu().clone()
        return capture

    losses = []
    for step in range(1, steps + 1):
        optimizer.capture = capture_into(step) if step in capture_steps else None
        optimizer.set_lr_multiplier(lr_multiplier(step - 1, (step - 1) / steps, lr_warmup_steps, min_lr_ratio,
                                                  total_steps=steps))
        losses.append(train_step(model, stream.next_batch(), optimizer).item())
        model.zero_grad(set_to_none=True)
    pairs = []
    for t in starts:
        phase = "early" if t == starts[0] else "late" if t == starts[-1] else "mid"
        for name in sorted(captured[t]):
            a, b = captured[t][name], captured[t + 1][name]
            pairs.append({
                "name": name, "layer": int(name.split(".")[0]), "phase": phase, "step_t": t,
                "shape": list(a.shape), "N_t": a, "N_t1": b,
                "sv_t": torch.linalg.svdvals(a.to(device).double()).cpu(),
                "sv_t1": torch.linalg.svdvals(b.to(device).double()).cpu(),
            })
    return {"pairs": pairs, "losses": losses, "starts": starts, "layers": layers, "seed": seed}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _config_hash(
    config: ModelConfig,
    data: ShardData,
    *,
    budget: str,
    target_tokens: int | None,
    target_seconds: float | None,
    auxiliary_lr: float,
    weight_decay: float,
    lr_warmup_steps: int,
    wall_warmup_fraction: float,
    min_lr_ratio: float,
    compile_model: bool,
    eval_tokens: int,
) -> str:
    payload = {
        "config": asdict(config),
        "data_sha256": {k: v["sha256"] for k, v in sorted(data.summary()["splits"].items())},
        "budget": budget,
        "target_tokens": target_tokens,
        "target_seconds": target_seconds,
        "auxiliary_lr": auxiliary_lr,
        "weight_decay": weight_decay,
        "lr_warmup_steps": lr_warmup_steps,
        "wall_warmup_fraction": wall_warmup_fraction,
        "min_lr_ratio": min_lr_ratio,
        "compile_model": compile_model,
        "eval_tokens": eval_tokens,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _run_key(row: dict) -> tuple:
    return (row["optimizer"], row["lr"], row["seed"], row["budget"], row["update_dtype"],
            row.get("run_config_hash"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="50m")
    parser.add_argument("--micro-batch", type=int, help="override the preset micro-batch")
    parser.add_argument("--grad-accum", type=int, help="override the preset gradient accumulation")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimizers", nargs="+", default=DEFAULT_OPTIMIZERS)
    parser.add_argument("--extra-optimizer", action="append", default=[], metavar="NAME=MODULE:FACTORY")
    parser.add_argument("--adamw-lrs", nargs="+", type=float, default=[5e-4, 1e-3, 2e-3, 4e-3])
    parser.add_argument("--muon-lrs", nargs="+", type=float, default=[0.005, 0.01, 0.02, 0.04])
    parser.add_argument("--muon-aux-lr", type=float, default=2e-3, help="AdamW LR for Muon's embedding/norms")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--update-dtype", choices=["fp32", "native"], default="fp32")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", choices=["tokens", "wall", "both"], default="both")
    parser.add_argument("--tokens", type=int, default=26_214_400, help="token budget (800 steps of 32,768)")
    parser.add_argument("--seconds", type=float, default=120.0, help="wall budget of model+optimizer compute")
    parser.add_argument("--lr-warmup-steps", type=int, default=50)
    parser.add_argument("--wall-warmup-fraction", type=float, default=0.0625,
                        help="wall-budget linear warmup fraction of total seconds")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--compile-warmup-steps", type=int, default=3)
    parser.add_argument("--eval-tokens", type=int, default=1 << 20)
    parser.add_argument("--compile-model", action="store_true", help="torch.compile the model (all runs)")
    parser.add_argument("--reference", default="muon_pe_eager")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cycle2_tier2_real.json")
    parser.add_argument("--resume", action="store_true", help="skip runs already in OUTPUT.jsonl (without it, OUTPUT.jsonl is replaced)")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--telemetry-every", type=int, default=100)
    parser.add_argument("--harvest-nesterov", action="store_true")
    parser.add_argument("--harvest-optimizer", default="muon_pe_eager")
    parser.add_argument("--harvest-lr", type=float, default=0.02)
    parser.add_argument("--harvest-steps", type=int, default=400)
    parser.add_argument("--harvest-seed", type=int, default=999)
    parser.add_argument("--harvest-output", type=Path, default=ROOT / "artifacts" / "harvested_nesterov_real.pt")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    config = PRESETS[args.preset]
    config = replace(config, micro_batch=args.micro_batch or config.micro_batch,
                     grad_accum=args.grad_accum or config.grad_accum)
    factories = builtin_factories()
    try:
        extra = parse_extra(args.extra_optimizer, factories)
    except (ValueError, ImportError, AttributeError) as error:
        parser.error(str(error))
    selected = list(dict.fromkeys([*args.optimizers, *extra]))
    unknown = set(selected) - factories.keys()
    if unknown:
        parser.error(f"Unknown optimizer(s): {', '.join(sorted(unknown))}")
    if any(not (math.isfinite(v) and v > 0) for v in [*args.adamw_lrs, *args.muon_lrs, args.muon_aux_lr]):
        parser.error("Learning rates must be positive and finite")
    data = ShardData(args.data_dir)
    meta = {
        **git_info(), "command": [sys.executable, *sys.argv], "args": {k: str(v) for k, v in vars(args).items()},
        "preset": args.preset, "config": asdict(config), "step_tokens": config.step_tokens,
        "parameters": parameter_counts(config), "data": data.summary(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "host_cpus": os.cpu_count(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "loadavg_start": os.getloadavg(),
    }

    if args.harvest_nesterov:
        if factories.get(args.harvest_optimizer) is None:
            parser.error("--harvest-optimizer must be a Muon optimizer")
        result = harvest_nesterov(config, data, device=device, factory=factories[args.harvest_optimizer],
                                  lr=args.harvest_lr, auxiliary_lr=args.muon_aux_lr,
                                  weight_decay=args.weight_decay, steps=args.harvest_steps,
                                  seed=args.harvest_seed, lr_warmup_steps=args.lr_warmup_steps,
                                  min_lr_ratio=args.min_lr_ratio)
        result["meta"] = {**meta, "optimizer": args.harvest_optimizer, "lr": args.harvest_lr,
                          "input": "post-Nesterov lerp(grad, momentum, beta) passed to the orthogonalizer"}
        args.harvest_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, args.harvest_output)
        print(f"Saved {len(result['pairs'])} pairs to {args.harvest_output}")
        return

    budgets = ["tokens", "wall"] if args.budget == "both" else [args.budget]
    budget_hashes = {
        b: _config_hash(
            config,
            data,
            budget=b,
            target_tokens=args.tokens if b == "tokens" else None,
            target_seconds=args.seconds if b == "wall" else None,
            auxiliary_lr=args.muon_aux_lr,
            weight_decay=args.weight_decay,
            lr_warmup_steps=args.lr_warmup_steps,
            wall_warmup_fraction=args.wall_warmup_fraction,
            min_lr_ratio=args.min_lr_ratio,
            compile_model=args.compile_model,
            eval_tokens=args.eval_tokens,
        )
        for b in budgets
    }
    log_path = args.output.with_suffix(".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.resume and log_path.exists():
        rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        for r in rows:
            if r.get("budget") not in budget_hashes or r.get("run_config_hash") != budget_hashes[r["budget"]]:
                raise ValueError(
                    f"Cannot --resume from {log_path}: existing row ({r.get('budget')!r}, "
                    f"{r.get('run_config_hash')!r}) != current {budget_hashes!r}"
                )
    elif log_path.exists():
        log_path.unlink()
    done = {_run_key(r) for r in rows}
    reusable_model = None
    if args.compile_model:
        torch._dynamo.config.cache_size_limit = max(getattr(torch._dynamo.config, "cache_size_limit", 8), 32)
        with torch.device(device):
            shared_model = Transformer(config)
        reusable_model = (shared_model, torch.compile(shared_model))
    inits: dict[int, dict] = {}
    for seed in args.seeds:
        inits = {seed: initial_state(config, seed)}  # keep one seed's weights in memory
        for budget in budgets:
            cfg_hash = budget_hashes[budget]
            plan = [(name, lr) for name in selected
                    for lr in (args.adamw_lrs if factories[name] is None else args.muon_lrs)]
            # Counterbalance run order across seeds and budgets.
            random.Random(f"{seed}-{budget}").shuffle(plan)
            for order, (name, lr) in enumerate(plan):
                layout = args.update_dtype if factories[name] is not None else None
                if (name, lr, seed, budget, layout, cfg_hash) in done:
                    continue
                row = run_trial(
                    config, data, device=device, seed=seed, optimizer_name=name, factory=factories[name],
                    lr=lr, auxiliary_lr=args.muon_aux_lr, weight_decay=args.weight_decay,
                    update_dtype=args.update_dtype, tokens=args.tokens if budget == "tokens" else None,
                    seconds=args.seconds if budget == "wall" else None, lr_warmup_steps=args.lr_warmup_steps,
                    wall_warmup_fraction=args.wall_warmup_fraction, min_lr_ratio=args.min_lr_ratio,
                    compile_warmup_steps=args.compile_warmup_steps, eval_tokens=args.eval_tokens,
                    init=inits[seed], compile_model=args.compile_model, reusable_model=reusable_model,
                    log_every=args.log_every, telemetry_every=args.telemetry_every)
                row["run_config_hash"] = cfg_hash
                row["order_in_block"] = order
                rows.append(row)
                done.add(_run_key(row))
                with log_path.open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                t = row["timing"]
                print(f"[{budget}] seed {seed} {name:<22} lr {lr:<7g} steps {row['steps']:>5} "
                      f"val {row['val_loss'] or float('nan'):.4f} test {row['test_loss'] or float('nan'):.4f} "
                      f"tok/s {row['tokens_per_second'] or 0:,.0f} fwd+bwd {t.get('fwd_bwd_ms') or 0:.1f} ms "
                      f"opt {t.get('optimizer_step_ms') or 0:.2f} ms polar {t.get('polar_ortho_ms') or 0:.2f} ms",
                      flush=True)
                torch.cuda.empty_cache() if device.type == "cuda" else None
    meta["loadavg_end"] = os.getloadavg()
    _write(args.output, {"meta": meta, "summary": summarize(rows, args.reference), "runs": rows})
    print(f"Wrote {args.output} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
