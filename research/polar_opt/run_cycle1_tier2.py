"""Cycle 1 matched-LR, multi-seed Tier-2 LM pretraining benchmark on RTX 5080."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import torch

from research.polar_opt.baselines import polar_express, polar_switch
from research.polar_opt.harness_tier2_lm import ModelConfig, Transformer, configure_device, run as tier2_run
from research.polar_opt.tier2_candidates import (
    ShapeKeyedSpectralPolar,
    pe_triton_bf16,
    ps2_blackwell_fp16,
    ps3_blackwell_fp16,
)


def _method_factories():
    return {
        "adamw": lambda: None,
        "muon_pe_eager": lambda: (lambda x: polar_express(x, steps=5)),
        "muon_pe_triton": lambda: pe_triton_bf16,
        "muon_ps2_eager": lambda: (lambda x: polar_switch(x, "PS2_55")),
        "muon_ps2_blackwell": lambda: ps2_blackwell_fp16,
        "muon_ps3_blackwell": lambda: ps3_blackwell_fp16,
        "muon_schatten8_S1_k3p4_lo005": lambda: ShapeKeyedSpectralPolar("S1-k3+4-lo0.005", scaling="schatten8"),
        "muon_schatten8_S1_k4p5_lo002": lambda: ShapeKeyedSpectralPolar("S1-k4+5-lo0.002", scaling="schatten8"),
        "muon_schatten8_S0_k6_lo005": lambda: ShapeKeyedSpectralPolar("S0-k6-lo0.05", scaling="schatten8"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", nargs="+", type=int, default=[123, 124])
    ap.add_argument("--muon-lrs", nargs="+", type=float, default=[0.01, 0.02])
    ap.add_argument("--adamw-lrs", nargs="+", type=float, default=[0.001, 0.003])
    ap.add_argument("--tokens", type=int, default=409600)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--eval-batches", type=int, default=12)
    ap.add_argument("--output", type=Path, default=Path("results/cycle1_tier2_lm.json"))
    args = ap.parse_args()

    device = configure_device(args.device)
    cfg = ModelConfig()
    factories = _method_factories()
    runs = []

    for seed in args.seeds:
        for opt_name, factory in factories.items():
            lrs = args.adamw_lrs if opt_name == "adamw" else args.muon_lrs
            for lr in lrs:
                for budget in ("tokens", "wall"):
                    row = tier2_run(
                        cfg,
                        device=device,
                        seed=seed,
                        method=factory(),
                        lr=lr,
                        auxiliary_lr=0.001,
                        weight_decay=0.01,
                        tokens=args.tokens if budget == "tokens" else None,
                        seconds=args.seconds if budget == "wall" else None,
                        warmup_steps=args.warmup_steps,
                        eval_batches=args.eval_batches,
                    )
                    row.update(optimizer=opt_name, budget=budget)
                    runs.append(row)
                    print(json.dumps(row, allow_nan=False), flush=True)

    grouped = defaultdict(list)
    for r in runs:
        grouped[(r["optimizer"], r["budget"], r["lr"])].append(r)

    best_by_opt_budget = {}
    for (opt, budget, lr), rows in grouped.items():
        if any(r["diverged"] for r in rows):
            continue
        val_mean = mean(r["validation_loss"] for r in rows)
        tps_mean = mean(r["tokens_per_second"] for r in rows)
        sec_mean = mean(r["train_seconds"] for r in rows)
        key = f"{opt}|{budget}"
        cand = {
            "optimizer": opt,
            "budget": budget,
            "best_lr": lr,
            "mean_validation_loss": val_mean,
            "mean_tokens_per_second": tps_mean,
            "mean_train_seconds": sec_mean,
            "seeds": len(rows),
        }
        if key not in best_by_opt_budget or val_mean < best_by_opt_budget[key]["mean_validation_loss"]:
            best_by_opt_budget[key] = cand

    payload = {
        "schema_version": 1,
        "config": asdict(cfg),
        "model_parameters": sum(p.numel() for p in Transformer(cfg).parameters()),
        "device": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
        "tokens_budget": args.tokens,
        "wall_seconds_budget": args.seconds,
        "seeds": args.seeds,
        "notes": (
            "Heuristic warm-power spectral scaling (scaling='spectral', defer_escalation=True) "
            "diverged at step 11 on 0.qkv.weight[1] due to multi-step gradual drift pushing "
            "true_sigma/sigma_hat to 1.191 (> 1.160 basin) without tripping 1-step jump triggers. "
            "Stateless rigorous Schatten-8 scaling (scaling='schatten8') eliminates underestimation "
            "and host syncs entirely while retaining true_sigma/upper <= 1.0."
        ),
        "summary": sorted(best_by_opt_budget.values(), key=lambda x: (x["budget"], x["mean_validation_loss"])),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
