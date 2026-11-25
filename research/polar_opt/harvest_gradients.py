"""Train a compact causal Transformer and harvest consecutive Muon EMA matrices."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from research.polar_opt.baselines import jordan_ns5
from research.polar_opt.harness_tier2_lm import (
    ModelConfig,
    Muon,
    TokenStream,
    Transformer,
    configure_device,
    train_step,
)


def _values(s: Tensor) -> dict:
    top, bottom = s[0].item(), s[-1].item()
    ratio = s.norm().item() / top if top else None
    return {
        "sigma_max": top, "sigma_min": bottom,
        "kappa": top / bottom if bottom else None,
        "effective_rank": ratio, "frobenius_spectral_ratio": ratio,
    }


def spectrum_pair(previous: Tensor, current: Tensor) -> dict:
    """Measure reduced FP64 SVDs on CPU with a fixed thread count in the CLI."""
    u0, s0, vh0 = torch.linalg.svd(previous.double(), full_matrices=False)
    u1, s1, vh1 = torch.linalg.svd(current.double(), full_matrices=False)
    before, after = _values(s0), _values(s1)
    top0, top1 = before["sigma_max"], after["sigma_max"]
    return {
        "previous": before, "current": after,
        "top_left_cosine": min(1.0, abs(torch.dot(u0[:, 0], u1[:, 0]).item())),
        "top_right_cosine": min(1.0, abs(torch.dot(vh0[0], vh1[0]).item())),
        "relative_sigma_max_change": abs(top1 - top0) / top0 if top0 else None,
    }


def _summarize(records):
    columns = {
        **{key: [row["current"][key] for row in records] for key in records[0]["current"]},
        **{key: [row[key] for row in records] for key in (
            "top_left_cosine", "top_right_cosine", "relative_sigma_max_change"
        )},
    }
    summary = {}
    for key, values in columns.items():
        valid = torch.tensor([v for v in values if v is not None], dtype=torch.float64)
        if valid.numel():
            quantiles = valid.quantile(torch.tensor([0., .25, .5, .75, 1.], dtype=valid.dtype))
            summary[key] = dict(zip(("min", "p25", "median", "p75", "max"), quantiles.tolist()))
        else:
            summary[key] = None
    return summary


def harvest(config, device, *, steps, seed, muon_lr, auxiliary_lr):
    if steps < 6:
        raise ValueError("Need at least six steps for disjoint early, mid, and late pairs")
    torch.manual_seed(seed)
    model = Transformer(config).to(device)
    optimizer = Muon(model, jordan_ns5, muon_lr, auxiliary_lr, weight_decay=0.01)
    stream = TokenStream(config, seed + 2_000_000, device)
    windows = {"early": (1, 2), "mid": (steps // 2, steps // 2 + 1), "late": (steps - 1, steps)}
    phases = {step: phase for phase, pair in windows.items() for step in pair}
    layers = {0, config.layers // 2, config.layers - 1}
    names = [name for name in optimizer.momentum if int(name.split(".")[0]) in layers]
    captures = {phase: [] for phase in windows}
    losses = []
    for step in range(1, steps + 1):
        loss = train_step(model, stream, optimizer, device)
        if step in phases:
            snapshot = {name: optimizer.momentum[name].detach().to("cpu", copy=True) for name in names}
            captures[phases[step]].append(snapshot)
        if step == 1 or step % 25 == 0 or step == steps:
            value = loss.item()
            if not torch.isfinite(loss).item():
                raise ArithmeticError(f"Training diverged at step {step}")
            losses.append({"step": step, "loss": value})
            print(f"step {step:3d}/{steps}  loss {value:.5f}", flush=True)

    pairs = []
    for phase, (previous, current) in captures.items():
        for name in names:
            pairs.append({
                "phase": phase, "parameter": f"blocks.{name}", "steps": list(windows[phase]),
                "previous": previous[name], "current": current[name],
            })
    metadata = {
        "schema_version": 1, "model": asdict(config),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch": str(torch.__version__), "seed": seed, "steps": steps,
        "tokens": steps * config.batch_size * config.sequence_length,
        "muon_lr": muon_lr, "adamw_lr": auxiliary_lr, "momentum_beta": optimizer.beta,
        "matrix_update": "Nesterov Muon with Jordan NS5; stored matrices are the pre-Nesterov EMA",
        "selected_layers": sorted(layers), "snapshot_steps": windows, "training_losses": losses,
    }
    return metadata, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--auxiliary-lr", type=float, default=0.001)
    parser.add_argument("--svd-threads", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/harvested_trajectories.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/harvested_spectra.json"))
    args = parser.parse_args()
    torch.set_num_threads(args.svd_threads)
    metadata, pairs = harvest(
        ModelConfig(), configure_device(args.device),
        steps=args.steps, seed=args.seed, muon_lr=args.muon_lr, auxiliary_lr=args.auxiliary_lr,
    )

    records = []
    for pair in pairs:
        records.append({
            "phase": pair["phase"], "parameter": pair["parameter"], "steps": pair["steps"],
            "shape": list(pair["current"].shape),
            **spectrum_pair(pair["previous"], pair["current"]),
        })
    stats = {
        **metadata, "trajectory_count": len(pairs), "svd_dtype": "float64", "svd_device": "cpu",
        "summary": _summarize(records),
        "summary_by_phase": {
            phase: _summarize([row for row in records if row["phase"] == phase])
            for phase in metadata["snapshot_steps"]
        },
        "records": records,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "pairs": pairs}, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(pairs)} pairs to {args.checkpoint} and spectral statistics to {args.output}")


if __name__ == "__main__":
    main()
