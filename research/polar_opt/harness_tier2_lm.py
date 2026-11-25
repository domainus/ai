"""Matched multi-seed language-model benchmark on a generated token stream."""

import argparse
import importlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite, sqrt
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from research.polar_opt.baselines import jordan_ns5, polar_express, polar_switch


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 1024
    width: int = 512
    layers: int = 6
    heads: int = 8
    hidden: int = 1344
    sequence_length: int = 128
    batch_size: int = 16


def _rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.attn_norm = nn.RMSNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)
        self.mlp_norm = nn.RMSNorm(config.width)
        self.gate = nn.Linear(config.width, config.hidden, bias=False)
        self.up = nn.Linear(config.width, config.hidden, bias=False)
        self.down = nn.Linear(config.hidden, config.width, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, sequence, width = x.shape
        head_dim = width // self.heads
        q, k, v = self.qkv(self.attn_norm(x)).view(
            batch, sequence, 3, self.heads, head_dim
        ).permute(2, 0, 3, 1, 4).unbind()
        q = _rope(F.rms_norm(q, (head_dim,), eps=1e-5), cos, sin)
        k = _rope(F.rms_norm(k, (head_dim,), eps=1e-5), cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(attn.transpose(1, 2).reshape(batch, sequence, width))
        h = self.mlp_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.width % config.heads or (config.width // config.heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.RMSNorm(config.width)
        head_dim = config.width // config.heads
        position = torch.arange(config.sequence_length, dtype=torch.float32)
        frequency = 10000 ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angles = position[:, None] * frequency
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        for block in self.blocks:
            for projection in (block.out, block.down):
                nn.init.normal_(projection.weight, std=0.02 / sqrt(2 * config.layers))

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embedding(tokens)
        cos = self.rope_cos[:tokens.shape[1]]
        sin = self.rope_sin[:tokens.shape[1]]
        for block in self.blocks:
            x = block(x, cos, sin)
        return F.linear(self.norm(x), self.embedding.weight)

    def matrix_parameters(self):
        return {name: p for name, p in self.blocks.named_parameters() if p.ndim == 2}


class TokenStream:
    """Generate sequences with local motifs and dependencies across eight-token phrases."""

    def __init__(self, config: ModelConfig, seed: int, device: torch.device):
        self.config = config
        self.device = device
        self.rng = torch.Generator(device=device).manual_seed(seed)
        position = torch.arange(config.sequence_length + 1, device=device)
        self.phrase = position // 8
        self.slot = position % 8

    def next_batch(self) -> tuple[Tensor, Tensor]:
        cfg = self.config
        motif, stride, topic = torch.randint(
            cfg.vocab_size, (3, cfg.batch_size, 1), device=self.device, generator=self.rng
        ).unbind()
        base = motif + self.phrase * (stride % 16 + 1)
        tokens = base + self.slot * (topic % 8 + 1) + (self.slot % 3) * (stride % 32)
        tokens = torch.where(self.slot == 0, topic + self.phrase % 4, tokens)
        tokens = tokens.remainder(cfg.vocab_size)
        return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


class Muon:
    """Muon on hidden matrices and AdamW on embedding and norm parameters."""

    def __init__(self, model, orthogonalize, lr, auxiliary_lr, weight_decay, beta=0.95):
        self.orthogonalize = orthogonalize
        self.lr, self.weight_decay, self.beta = lr, weight_decay, beta
        self.matrices = model.matrix_parameters()
        self.momentum = {name: torch.zeros_like(p) for name, p in self.matrices.items()}
        groups = defaultdict(list)
        for name, p in self.matrices.items():
            groups[p.shape].append(name)
        self.groups = list(groups.values())
        matrix_ids = {id(p) for p in self.matrices.values()}
        self.auxiliary = _adamw(
            [p for p in model.parameters() if id(p) not in matrix_ids],
            auxiliary_lr, weight_decay, next(model.parameters()).is_cuda,
        )

    @torch.no_grad()
    def step(self):
        for names in self.groups:
            gradients = [self.matrices[name].grad for name in names]
            momenta = [self.momentum[name] for name in names]
            torch._foreach_lerp_(momenta, gradients, 1 - self.beta)
            nesterov = torch.stack(torch._foreach_lerp(gradients, momenta, self.beta))
            updates = self.orthogonalize(nesterov)
            rows, columns = nesterov.shape[-2:]
            parameters = [self.matrices[name] for name in names]
            torch._foreach_mul_(parameters, 1 - self.lr * self.weight_decay)
            torch._foreach_add_(
                parameters, list(updates.unbind()), alpha=-self.lr * sqrt(max(1, rows / columns))
            )
        self.auxiliary.step()


def _adamw(parameters, lr, weight_decay, cuda):
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, fused=cuda)


def train_step(model, stream, optimizer, device) -> Tensor:
    model.zero_grad(set_to_none=True)
    x, targets = stream.next_batch()
    loss = _loss(model, x, targets, device)
    loss.backward()
    optimizer.step()
    return loss.detach()


def _loss(model, x, targets, device):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(x)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


@torch.no_grad()
def _evaluate(model, stream, batches, device):
    model.eval()
    losses = [_loss(model, *stream.next_batch(), device) for _ in range(batches)]
    return torch.stack(losses).mean().item()


def configure_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return device


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _setup(config, seed, device, method, lr, auxiliary_lr, decay):
    torch.manual_seed(seed)
    model = Transformer(config).to(device)
    optimizer = (_adamw(model.parameters(), lr, decay, device.type == "cuda") if method is None
                 else Muon(model, method, lr, auxiliary_lr, decay))
    return model, optimizer


def run(config, *, device, seed, method, lr, auxiliary_lr, weight_decay, tokens, seconds,
        warmup_steps, eval_batches):
    """Run one fresh seed/LR trial with exactly one of the two budgets."""
    if (tokens is None) == (seconds is None):
        raise ValueError("Specify exactly one budget")
    step_tokens = config.batch_size * config.sequence_length
    if tokens is not None and (tokens <= 0 or tokens % step_tokens):
        raise ValueError(f"Token budget must be a positive multiple of {step_tokens}")
    if seconds is not None and (not isfinite(seconds) or seconds <= 0):
        raise ValueError("Wall-clock budget must be positive and finite")
    if warmup_steps < 0 or eval_batches <= 0:
        raise ValueError("Expected nonnegative warmup and positive evaluation batch counts")

    if warmup_steps:
        model, optimizer = _setup(config, seed, device, method, lr, auxiliary_lr, weight_decay)
        warmup = TokenStream(config, seed + 1_000_000, device)
        for _ in range(warmup_steps):
            train_step(model, warmup, optimizer, device)
        _sync(device)
        del model, optimizer, warmup

    model, optimizer = _setup(config, seed, device, method, lr, auxiliary_lr, weight_decay)
    stream = TokenStream(config, seed + 2_000_000, device)
    evaluation = TokenStream(config, 3_000_000, device)
    _sync(device)
    start = perf_counter()
    steps = 0
    elapsed = 0.0
    diverged = False
    while (steps * step_tokens < tokens if tokens is not None else elapsed < seconds):
        loss = train_step(model, stream, optimizer, device).item()
        elapsed = perf_counter() - start
        steps += 1
        if not isfinite(loss):
            diverged = True
            break
    validation = _evaluate(model, evaluation, eval_batches, device)
    return {
        "seed": seed, "lr": lr, "steps": steps, "tokens": steps * step_tokens,
        "target_tokens": tokens, "target_seconds": seconds, "train_seconds": elapsed,
        "wall_budget_overshoot_seconds": max(0.0, elapsed - seconds) if seconds else None,
        "tokens_per_second": steps * step_tokens / elapsed,
        "last_train_loss": loss if isfinite(loss) else None,
        "validation_loss": validation if isfinite(validation) else None,
        "diverged": diverged or not isfinite(validation),
    }


def load_callable(spec):
    module, attribute = spec.split(":", 1)
    function = getattr(importlib.import_module(module), attribute)
    if not callable(function):
        raise ValueError(f"{spec} is not callable")
    return function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 124, 125])
    parser.add_argument("--lr", nargs="+", type=float, default=[0.0003, 0.001, 0.003])
    parser.add_argument("--auxiliary-lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizers", nargs="+",
                        default=["adamw", "muon_pe", "muon_ps2", "muon_ps3"])
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=MODULE:FUNCTION")
    parser.add_argument("--budget", choices=["tokens", "wall", "both"], default="both")
    parser.add_argument("--tokens", type=int, default=614400)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("results/tier2_lm.json"))
    parser.add_argument("--tiny", action="store_true", help="Use a small model for smoke tests")
    args = parser.parse_args()
    device = configure_device(args.device)
    config = (ModelConfig(width=64, layers=2, heads=4, hidden=128, sequence_length=32, batch_size=2)
              if args.tiny else ModelConfig())

    methods = {
        "adamw": None, "muon_jordan": jordan_ns5, "muon_pe": polar_express,
        "muon_ps2": polar_switch, "muon_ps3": lambda x: polar_switch(x, "PS3_55"),
    }
    selected = args.optimizers.copy()
    for entry in args.candidate:
        name, spec = entry.split("=", 1)
        if not name or name in methods:
            parser.error(f"Duplicate or empty candidate name {name!r}")
        methods[name] = load_callable(spec)
        if name not in selected:
            selected.append(name)
    unknown = set(selected) - methods.keys()
    if unknown:
        parser.error(f"Unknown optimizer(s): {', '.join(sorted(unknown))}")
    if any(not isfinite(lr) or lr <= 0 for lr in [*args.lr, args.auxiliary_lr]):
        parser.error("Learning rates must be positive and finite")

    budgets = ["tokens", "wall"] if args.budget == "both" else [args.budget]
    results = []
    for seed, lr, budget, name in product(args.seeds, args.lr, budgets, selected):
        row = run(
            config, device=device, seed=seed, method=methods[name], lr=lr,
            auxiliary_lr=args.auxiliary_lr, weight_decay=args.weight_decay,
            tokens=args.tokens if budget == "tokens" else None,
            seconds=args.seconds if budget == "wall" else None,
            warmup_steps=args.warmup_steps, eval_batches=args.eval_batches,
        )
        row.update(optimizer=name, budget=budget)
        results.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    output = {
        "config": asdict(config), "model_parameters": sum(p.numel() for p in Transformer(config).parameters()),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch": str(torch.__version__), "auxiliary_lr": args.auxiliary_lr,
        "weight_decay": args.weight_decay, "warmup_steps": args.warmup_steps,
        "eval_batches": args.eval_batches, "candidate_entry_points": args.candidate, "runs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
