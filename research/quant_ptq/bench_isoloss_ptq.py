"""Iso-loss Muon-versus-AdamW post-training quantization on Blackwell (Cycle 4 C4-01b).

Cycle 3 T5-01 compared the PTQ loss of a 320-step Muon model and a 320-step AdamW model. Their
baseline perplexities were far apart, so that comparison was not quality matched. This benchmark
first matches the two baselines, then compares their PTQ sensitivity on three paired seeds.

Stage A  Seed 0, ``train.bin`` and ``val.bin`` only. Train ``muon_ss3_54`` for T_muon steps and
         ``adamw`` for at most 768 steps (LR 0.001 or 0.0015). Choose (T_muon, T_adamw, adamw_lr)
         with |val NLL difference| <= 0.02 nats: the largest T_muon in (320, 256, 224, 192) that
         AdamW can match (AdamW steps on a grid of 8). Every trial is a separate warmup + cosine
         run and is stored.
         ``test.bin`` is locked during this stage (a read raises).
Stage B  Seeds 1, 2, 3 with the locked Stage A choice. For each seed both optimizers start from
         the same ``initial_state`` and read the same batch stream. Each of the six checkpoints
         gets: full ``val.bin`` and ``test.bin`` NLL, weight / activation / Hessian distribution
         diagnostics, FP64-Hessian weight-only PTQ (4 formats x 4 methods) and hardware NVFP4
         W4A4 through ``NVFP4Linear`` (4 methods x dynamic / static activation scales, plus one
         FP32 emulation row). Calibration and static scales use 128 ``train.bin`` windows only.
Analysis Per (format, method, mode): signed per-seed differences ``muon - adamw`` of the test NLL
         increase and of the quantized test NLL, with the exact df = 2 paired t-test, 95% Student-t
         interval and Holm-Bonferroni correction. Methods are selected on ``val.bin`` only.

Run (canonical, about 70 minutes on an RTX 5080; ``--resume`` continues an interrupted run):
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_isoloss_ptq.py \\
        --data-dir artifacts/real_text --output results/cycle4_c4_01b_isoloss_ptq.json
Smoke run:
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_isoloss_ptq.py --quick --output /tmp/iso.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from research.polar_opt import harness_tier2_lm, harness_tier2_real as h
from research.polar_opt.harness_tier2_lm import Transformer
from research.quant_ptq import bench_nvfp4_e2e as e2e
from research.quant_ptq import micro_rotated_ptq as q
from research.quant_ptq import nvfp4_inference_blackwell as nb

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "cycle4_c4_01b.v1"
CANONICAL_OUTPUT = ROOT / "results" / "cycle4_c4_01b_isoloss_ptq.json"
QUICK_OUTPUT = ROOT / "results" / "cycle4_c4_01b_isoloss_ptq_quick.json"

MUON, ADAMW = "muon_ss3_54", "adamw"
OPTIMIZERS = (MUON, ADAMW)
WEIGHT_DECAY, MIN_LR_RATIO, GPTQ_DAMP = 0.01, 0.1, 0.01
METHODS = ("rtn", "gptq", "hadamard_b16_gptq", "adaptive_butterfly_b16_gptq")
HW_FORMAT = "nvfp4_g16"
HW_MODES = ("w4a4_hw_dynamic", "w4a4_hw_static")
EMULATED_METHOD, EMULATED_MODE = "hadamard_b16_gptq", "w4a4_emulated_fp32_dynamic"
EMULATION_PARITY_NATS = 1e-3  # hardware FP4 GEMM (BF16 output) against the FP32 GEMM of the same operands
VAL_TOKENS, TEST_TOKENS = 2_001_920, 1_999_872  # complete 512-token windows of val.bin and test.bin
# Student-t 0.975 quantile with 2 degrees of freedom, as fixed by the C4-01b protocol (the value that
# older SciPy releases return for ``t.ppf(0.975, 2)``). The closed form sqrt(2 * 0.95^2 / (1 - 0.95^2)),
# which SciPy 1.18 also returns, is 1.6e-10 smaller. Both are stored; the difference changes an interval
# bound by 4e-11 of its half-width.
T975_DF2 = 4.302652729911275
T975_DF2_CLOSED_FORM = math.sqrt(2 * 0.9025 / 0.0975)
DIAGNOSTICS = ("weight_excess_kurtosis", "weight_max_over_rms", "activation_excess_kurtosis",
               "activation_max_over_rms", "hessian_diag_max_over_mean")
SPLITS = ("val", "test")
TOL = 1e-12


def fakequant_mode(fmt: q.QuantFormat) -> str:
    return f"w{fmt.element_bits}a16_fakequant"


def warmup_for(steps: int) -> int:
    return max(20, steps // 10)


def hardware_w4a4_available(device: torch.device) -> bool:
    return (device.type == "cuda" and hasattr(F, "scaled_mm")
            and torch.cuda.get_device_capability(device) >= (10, 0))


# ======================================================================================
# Data access ledger
# ======================================================================================
class SplitLedger(dict):
    """``ShardData.splits`` that counts reads per split and can lock splits for a stage."""

    def __init__(self, splits: dict):
        super().__init__(splits)
        self.reads = {name: 0 for name in splits}
        self.locked: frozenset[str] = frozenset()

    def __getitem__(self, name: str):
        if name in self.locked:
            raise RuntimeError(f"{name}.bin is locked in this stage")
        self.reads[name] += 1
        return super().__getitem__(name)

    def get(self, name: str, default=None):
        if name in self:
            return self[name]
        if name in self.locked:
            raise RuntimeError(f"{name}.bin is locked in this stage")
        return default

    def _ensure_unlocked(self) -> None:
        for k in self:
            if k in self.locked:
                raise RuntimeError(f"{k}.bin is locked in this stage")

    def values(self):
        self._ensure_unlocked()
        return [self[k] for k in self]

    def items(self):
        self._ensure_unlocked()
        return [(k, self[k]) for k in self]

    def copy(self):
        self._ensure_unlocked()
        return {k: self[k] for k in self}

    def pop(self, *args, **kwargs):
        raise RuntimeError("SplitLedger does not allow mutating split access")

    def popitem(self):
        raise RuntimeError("SplitLedger does not allow mutating split access")

    def setdefault(self, *args, **kwargs):
        raise RuntimeError("SplitLedger does not allow mutating split access")

    @contextlib.contextmanager
    def forbid(self, *names: str):
        previous, self.locked = self.locked, self.locked | frozenset(names)
        try:
            yield self
        finally:
            self.locked = previous


def load_data(args) -> h.ShardData:
    data = q.load_data(args.data_dir, verify=not args.no_verify,
                       expected=None if args.no_verify else q.EXPECTED_SHA256)
    data.splits = SplitLedger(data.splits)
    return data


# ======================================================================================
# Training
# ======================================================================================
def train_checkpoint(optimizer: str, config: h.ModelConfig, train: np.ndarray, device: torch.device, *,
                     seed: int, steps: int, lr: float, init: dict[str, Tensor]):
    """One warmup + cosine run from ``init`` on the seed's batch stream. Reads ``train`` only.

    The loop is the token-budget path of ``harness_tier2_real.run_trial`` without its final
    val/test evaluation, so that Stage A cannot touch ``test.bin``.
    """
    spec = q.MODEL_SPECS[optimizer]
    factory, orthogonalizer = q._factory(spec["optimizer"], device)
    with torch.device(device):
        model = Transformer(config)
    model.load_state_dict(init)
    model.train()
    opt = h.build_optimizer(optimizer, factory, model, lr=lr, auxiliary_lr=spec["auxiliary_lr"] or 0.0,
                            weight_decay=WEIGHT_DECAY, update_dtype="fp32")
    stream = h.BatchStream(train, config, seed, device)
    warmup = warmup_for(steps)
    losses = []
    began = time.perf_counter()
    for step in range(steps):
        opt.set_lr_multiplier(h.lr_multiplier(step, step / steps, warmup, MIN_LR_RATIO, total_steps=steps))
        losses.append(h.train_step(model, stream.next_batch(), opt).item())
        model.zero_grad(set_to_none=True)
        if not math.isfinite(losses[-1]):
            raise RuntimeError(f"{optimizer} seed {seed} diverged at step {step + 1}")
    seconds = time.perf_counter() - began
    model.eval()
    tokens = steps * config.step_tokens
    row = {"optimizer": optimizer, "seed": seed, "steps": steps, "warmup_steps": warmup, "lr": lr,
           "auxiliary_lr": spec["auxiliary_lr"], "weight_decay": WEIGHT_DECAY, "min_lr_ratio": MIN_LR_RATIO,
           "tokens": tokens, "train_epochs": tokens / len(train), "orthogonalizer": orthogonalizer,
           "init_fingerprint": h.state_fingerprint(init),
           "stream_fingerprint": h.stream_fingerprint(train, config, seed),
           "train_loss": math.fsum(losses[-10:]) / len(losses[-10:]), "last_train_loss": losses[-1],
           "train_loss_note": "train_loss is the mean of the last 10 optimizer steps",
           "train_curve": [[i + 1, v] for i, v in enumerate(losses) if i == 0 or (i + 1) % 10 == 0],
           "train_seconds": seconds, "checkpoint_sha256": e2e._checkpoint_sha256(model)}
    return model, row


# ======================================================================================
# Stage A: seed-0 iso-loss calibration on val.bin
# ======================================================================================
def calibration_search(run, *, t_muon_candidates, muon_lr: float, adamw_lrs, adamw_max_steps: int,
                       step_quantum: int, tolerance: float, max_probes: int = 10,
                       search: list | None = None) -> None:
    """Run the pilot trials that ``select_isoloss`` needs. ``run(optimizer, steps, lr)`` gives val NLL.

    AdamW is first trained at the full step budget for every LR. Then, for T_muon from large to
    small and for the AdamW LR with the lower full-budget val NLL first, the AdamW step count is
    searched on the ``step_quantum`` grid: ``hi`` is the AdamW trial that is better than Muon and
    nearest to it, ``lo`` the nearest worse trial with fewer steps (0 steps if there is none).
    The next trial is the grid point nearest to the linear interpolation between them (the
    midpoint from 0 steps). An LR stops when a trial is within half the tolerance, when AdamW is
    never better, or after ``max_probes`` trials. AdamW runs can be unstable, so val NLL is not
    always monotone in the step count: when no grid point is left between ``lo`` and ``hi``,
    ``lo`` is set aside as an outlier and the search goes on from the next worse trial.
    The other LR is tried only if the first one has no trial within the tolerance. The search
    stops at the first T_muon with a match.
    """
    search = [] if search is None else search
    adamw = {lr: {adamw_max_steps: run(ADAMW, adamw_max_steps, lr)} for lr in adamw_lrs}  # lr -> steps -> val NLL
    for t_muon in sorted(t_muon_candidates, reverse=True):
        target = run(MUON, t_muon, muon_lr)

        def closest(lr) -> float:
            return min(abs(v - target) for v in adamw[lr].values())

        for lr in sorted(adamw_lrs, key=lambda v: (adamw[v][adamw_max_steps], v)):
            outliers = set()
            for _ in range(max_probes):
                excess = {steps: v - target for steps, v in adamw[lr].items()}  # > 0: AdamW is worse than Muon
                better = {steps: e for steps, e in excess.items() if e < 0}
                if closest(lr) <= tolerance / 2 or not better:
                    break
                hi = max(better, key=lambda steps: (better[steps], -steps))
                worse = {steps: e for steps, e in excess.items() if e > 0 and steps < hi and steps not in outliers}
                lo = min(worse, key=lambda steps: (worse[steps], -steps)) if worse else 0
                free = [steps for steps in range(lo + step_quantum, hi, step_quantum) if steps not in adamw[lr]]
                if not free:
                    if not worse or closest(lr) <= tolerance:
                        break
                    outliers.add(lo)
                    continue
                guess = lo + (hi - lo) * worse[lo] / (worse[lo] - better[hi]) if worse else hi / 2
                steps = min(free, key=lambda v: (abs(v - guess), v))
                adamw[lr][steps] = run(ADAMW, steps, lr)
                search.append({"t_muon": t_muon, "adamw_lr": lr, "adamw_steps": steps,
                               "adamw_minus_muon_val_nll": adamw[lr][steps] - target, "bracket_steps": [lo, hi]})
            if closest(lr) <= tolerance:
                break
        if min(closest(lr) for lr in adamw_lrs) <= tolerance:
            return


def non_monotone_trials(rows: list[dict], tolerance: float) -> list[dict]:
    """AdamW pilot trials that are worse by more than ``tolerance`` than a trial with fewer steps."""
    out = []
    for lr in sorted({r["lr"] for r in rows if r["optimizer"] == ADAMW}):
        trials = sorted((r for r in rows if r["optimizer"] == ADAMW and r["lr"] == lr), key=lambda r: r["steps"])
        for i, r in enumerate(trials):
            if i and r["val_nll"] - min(t["val_nll"] for t in trials[:i]) > tolerance:
                best = min(trials[:i], key=lambda t: (t["val_nll"], t["steps"]))
                out.append({"adamw_lr": lr, "steps": r["steps"], "val_nll": r["val_nll"],
                            "better_trial_steps": best["steps"], "better_trial_val_nll": best["val_nll"],
                            "val_nll_excess": r["val_nll"] - best["val_nll"]})
    return out


def select_isoloss(rows: list[dict], *, tolerance: float, t_muon_candidates, adamw_max_steps: int) -> dict:
    """Pure Stage A choice from the stored pilot rows (val NLL only).

    The largest candidate T_muon with an AdamW row (steps <= ``adamw_max_steps``) within
    ``tolerance`` wins; among its matches the smallest |difference|, then fewer steps, then the
    lower LR. With no match at all, the closest pair is returned with ``matched = False``.
    """
    muon = {r["steps"]: r for r in rows if r["optimizer"] == MUON}
    adamw = [r for r in rows if r["optimizer"] == ADAMW and r["steps"] <= adamw_max_steps]
    if not muon or not adamw:
        raise ValueError("The pilot needs at least one Muon row and one AdamW row")

    def closest(t: int):
        return min(adamw, key=lambda a: (abs(muon[t]["val_nll"] - a["val_nll"]), a["steps"], a["lr"]))

    present = [t for t in sorted(t_muon_candidates, reverse=True) if t in muon]
    if not present:
        raise ValueError("No Muon pilot row is a T_muon candidate")
    pairs = [(t, closest(t)) for t in present]
    matches = [(t, a) for t, a in pairs if abs(muon[t]["val_nll"] - a["val_nll"]) <= tolerance]
    t, a = matches[0] if matches else min(pairs, key=lambda p: (abs(muon[p[0]]["val_nll"] - p[1]["val_nll"]), -p[0]))
    gap = muon[t]["val_nll"] - a["val_nll"]
    return {"matched": bool(matches), "tolerance_nats": tolerance, "t_muon": t, "t_adamw": a["steps"],
            "adamw_lr": a["lr"], "muon_lr": muon[t]["lr"], "muon_val_nll": muon[t]["val_nll"],
            "adamw_val_nll": a["val_nll"], "val_gap_muon_minus_adamw": gap, "abs_val_gap": abs(gap),
            "muon_tokens": muon[t]["tokens"], "adamw_tokens": a["tokens"],
            "adamw_over_muon_tokens": a["tokens"] / muon[t]["tokens"],
            "t_muon_candidates_evaluated": present,
            "rule": "largest T_muon candidate with an AdamW row within the tolerance on seed-0 val.bin; "
                    "then smallest |difference|, fewer AdamW steps, lower AdamW LR"}


def stage_a_pilot(config: h.ModelConfig, data: h.ShardData, device: torch.device, protocol: dict, units,
                  log) -> dict:
    """Seed-0 pilot. ``test.bin`` is locked for the whole stage; rows hold val NLL only."""
    ledger, seed = data.splits, protocol["pilot_seed"]
    before = dict(ledger.reads)
    rows, memo, search = [], {}, []
    began = time.perf_counter()
    with ledger.forbid("test"):
        train, val = ledger["train"], ledger["val"]
        init = h.initial_state(config, seed)

        def trial(optimizer: str, steps: int, lr: float) -> dict:
            model, row = train_checkpoint(optimizer, config, train, device, seed=seed, steps=steps, lr=lr,
                                          init=init)
            result = q.evaluate_split(model, val, config.sequence_length, config.micro_batch, device,
                                      protocol["max_eval_windows"])
            row.update(val_nll=result["nll"], val_ppl=result["ppl"], val_tokens=result["tokens"],
                       evaluated_splits=["val"])
            del model
            return row

        def run(optimizer: str, steps: int, lr: float) -> float:
            key = (optimizer, steps, lr)
            if key not in memo:
                row = units.cached(("pilot", seed, optimizer, steps, lr), lambda: trial(optimizer, steps, lr))
                rows.append(row)
                memo[key] = row["val_nll"]
                log(f"  A seed {seed} {optimizer:<12} steps {steps:4d} lr {lr:<7g} train {row['train_loss']:.4f} "
                    f"val NLL {row['val_nll']:.4f} ({row['train_seconds']:.0f}s)")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            return memo[key]

        calibration_search(run, t_muon_candidates=protocol["t_muon_candidates"], muon_lr=protocol["muon"]["lr"],
                           adamw_lrs=protocol["adamw_lrs"], adamw_max_steps=protocol["adamw_max_steps"],
                           step_quantum=protocol["adamw_step_quantum"], tolerance=protocol["tolerance_nats"],
                           search=search)
    selection = select_isoloss(rows, tolerance=protocol["tolerance_nats"],
                               t_muon_candidates=protocol["t_muon_candidates"],
                               adamw_max_steps=protocol["adamw_max_steps"])
    return {"seed": seed, "splits_used": ["train", "val"],
            "split_reads": {name: ledger.reads[name] - before[name] for name in before},
            "rows": rows, "search": search, "selection": selection,
            "adamw_non_monotone": non_monotone_trials(rows, protocol["tolerance_nats"]),
            "seconds": time.perf_counter() - began}


# ======================================================================================
# Distribution diagnostics
# ======================================================================================
class Moments:
    """Streaming FP64 power sums of all elements of a tensor stream: kurtosis and max / rms."""

    def __init__(self, device=None):
        self.sums = torch.zeros(4, dtype=torch.float64, device=device)
        self.amax = torch.zeros((), dtype=torch.float64, device=device)
        self.count = 0

    def update(self, x: Tensor) -> None:
        x = x.detach().double().flatten()
        self.count += x.numel()
        x2 = x * x
        self.sums += torch.stack((x.sum(), x2.sum(), (x2 * x).sum(), (x2 * x2).sum()))
        self.amax = torch.maximum(self.amax, x.abs().max())

    def result(self) -> dict:
        """Same definitions as ``micro_rotated_ptq.excess_kurtosis`` and ``max_over_rms``."""
        s1, s2, s3, s4 = (self.sums / self.count).tolist()
        m2 = s2 - s1 * s1
        m4 = s4 - 4 * s1 * s3 + 6 * s1 * s1 * s2 - 3 * s1 ** 4
        return {"excess_kurtosis": m4 / (m2 * m2) - 3.0 if m2 > 0 else 0.0,
                "max_over_rms": self.amax.item() / math.sqrt(s2) if s2 > 0 else 0.0, "elements": self.count}


@torch.no_grad()
def collect_calibration_statistics(model, windows: np.ndarray, batch: int, device: torch.device):
    """One forward pass over ``windows``: FP64 Hessians and input moments of every quantized linear."""
    linears = q.quantized_linears(model)
    moments = {name: Moments(device) for name in linears}

    def hook(name):
        def record(module, inputs):
            with torch.autocast(device_type=device.type, enabled=False):
                moments[name].update(inputs[0])
        return record

    handles = [m.register_forward_pre_hook(hook(name)) for name, m in linears.items()]
    try:
        hessians = q.collect_hessians(model, windows, batch, device)
    finally:
        for handle in handles:
            handle.remove()
    return hessians, moments


def summarize_diagnostics(per_layer: list[dict]) -> dict:
    """Mean / median / min / max of every diagnostic over all layers and per projection kind."""
    return {metric: {"all": q._summary([r[metric] for r in per_layer]),
                     "by_kind": {kind: q._summary([r[metric] for r in per_layer if r["kind"] == kind])
                                 for kind in q.LINEAR_KINDS}}
            for metric in DIAGNOSTICS}


def distribution_diagnostics(weights: dict[str, Tensor], hessians: dict[str, Tensor],
                             moments: dict[str, Moments]) -> dict:
    per_layer = []
    for name, w in weights.items():
        act, diag = moments[name].result(), hessians[name].diagonal()
        per_layer.append({"name": name, "kind": name.rsplit(".", 1)[1],
                          "weight_excess_kurtosis": q.excess_kurtosis(w), "weight_max_over_rms": q.max_over_rms(w),
                          "activation_excess_kurtosis": act["excess_kurtosis"],
                          "activation_max_over_rms": act["max_over_rms"],
                          "hessian_diag_max_over_mean": (diag.max() / diag.mean()).item()})
    return {"per_layer": per_layer, "summary": summarize_diagnostics(per_layer),
            "definitions": {
                "weight": "FP32 master weight of each of the quantized linear layers",
                "activation": "input of the layer, all elements pooled over the train.bin calibration windows "
                              "(BF16-autocast forward of the unquantized model, FP64 moments)",
                "excess_kurtosis": "central m4 / m2^2 - 3", "max_over_rms": "max |x| / sqrt(mean x^2)",
                "hessian_diag_max_over_mean": "max_j H_jj / mean_j H_jj for the FP64 H = X^T X / N "
                                              "(largest input-channel energy over the mean)"}}


# ======================================================================================
# Stage B: PTQ study of one checkpoint
# ======================================================================================
def _quantize(Wr: Tensor, Hr: Tensor | None, fmt: q.QuantFormat, rounding: str):
    """Quantize a rotated weight stack. Returns (values, NVFP4 encoding or None, GPTQ damping list)."""
    encode = fmt.kind == "nvfp4"
    if rounding == "gptq":
        Qr, info = q.gptq_quantize(Wr, Hr, fmt, damp=GPTQ_DAMP, return_encoding=encode)
        return Qr, info.get("encoding"), info["damping_lambda"]
    if not encode:
        return q.quantize_rtn(Wr, fmt), None, []
    encs = [q.nvfp4_encode(w) for w in Wr]
    return torch.stack([e["values"] for e in encs]), {
        key: torch.stack([e[key] for e in encs]) for key in ("codes", "block_scale", "global_scale")}, []


def best_methods(rows: list[dict]) -> dict:
    """Per (format, mode) cell: the method with the lowest summed val NLL, and the same on test."""
    cells: dict[tuple, dict] = {}
    for r in rows:
        totals = cells.setdefault((r["format"], r["mode"]), {})
        entry = totals.setdefault(r["method"], {"val": [], "test": []})
        for split in SPLITS:
            entry[split].append(r[f"{split}_nll"])
    out = {}
    for (fmt, mode), totals in cells.items():
        by = {split: min(totals, key=lambda m: math.fsum(totals[m][split])) for split in SPLITS}
        out[f"{fmt}/{mode}"] = {"format": fmt, "mode": mode, "methods": list(totals), "best_by_val": by["val"],
                                "best_by_test": by["test"], "confirmed_on_test": by["val"] == by["test"]}
    return out


def checkpoint_study(model, config: h.ModelConfig, data: h.ShardData, device: torch.device, calibration: dict,
                     protocol: dict, log, label: str) -> dict:
    """Baselines, diagnostics and the PTQ grid of one trained checkpoint. Weights are restored."""
    t0 = time.perf_counter()
    seq, batch, windows = config.sequence_length, config.micro_batch, protocol["max_eval_windows"]
    val, test = data.splits["val"], data.splits["test"]
    sha_before = e2e._checkpoint_sha256(model)

    def evaluate_both():
        return {"val": q.evaluate_split(model, val, seq, batch, device, windows),
                "test": q.evaluate_split(model, test, seq, batch, device, windows)}

    baseline = evaluate_both()
    log(f"  B {label} BF16 val NLL {baseline['val']['nll']:.4f} test NLL {baseline['test']['nll']:.4f}")
    hessians, moments = collect_calibration_statistics(model, calibration["array"], batch, device)
    linears = q.quantized_linears(model)
    kinds = {k: [f"blocks.{i}.{k}" for i in range(config.layers)] for k in q.LINEAR_KINDS}
    weights = {n: m.weight.detach().float().clone() for n, m in linears.items()}
    diagnostics = distribution_diagnostics(weights, hessians, moments)
    W = {k: torch.stack([weights[n] for n in names]) for k, names in kinds.items()}
    Hc = {k: torch.stack([hessians[n] for n in names]) for k, names in kinds.items()}
    reference_error = {k: q.output_error(W[k], torch.zeros_like(W[k]), Hc[k]) for k in kinds}
    del hessians, moments
    hardware_ok = hardware_w4a4_available(device)

    def apply(source: dict[str, Tensor]) -> None:
        with torch.no_grad():
            for kind, names in kinds.items():
                for i, name in enumerate(names):
                    linears[name].weight.copy_(source[kind][i])

    def row(fmt_name: str, method: str, mode: str, evals: dict, seconds: float, **extra) -> dict:
        family, rounding = q.parse_method(method)
        out = {"format": fmt_name, "method": method, "mode": mode, "rotation": family, "rounding": rounding}
        for split in SPLITS:
            result, base = evals[split], baseline[split]
            if result["tokens"] != base["tokens"]:
                raise RuntimeError("Evaluation token count changed between runs")
            out.update({f"{split}_nll": result["nll"], f"{split}_ppl": result["ppl"],
                        f"{split}_tokens": result["tokens"], f"{split}_delta_nll": result["nll"] - base["nll"],
                        f"{split}_relative_delta_ppl_pct": 100.0 * (result["ppl"] / base["ppl"] - 1.0)})
        out.update(seconds=seconds, **extra)
        log(f"  B {label} {fmt_name:<13} {method:<28} {mode:<26} val dNLL {out['val_delta_nll']:+.4f} "
            f"test dNLL {out['test_delta_nll']:+.4f} ({out['test_relative_delta_ppl_pct']:+.2f}%)")
        return out

    grid, butterfly_fits = [], {}
    for fmt_name in protocol["weight_formats"]:
        fmt = q.FORMATS[fmt_name]
        rotations = {}
        for kind in q.LINEAR_KINDS:
            rotations[(kind, "identity")] = q.Rotation()
            rotations[(kind, "hadamard_b16")] = q.BlockHadamard(16)
            rotations[(kind, "adaptive_butterfly_b16")], butterfly_fits[f"{fmt_name}/{kind}"] = q.make_rotation(
                "adaptive_butterfly_b16", W[kind], fmt, butterfly_steps=protocol["butterfly_steps"])
        for method in protocol["methods"]:
            family, rounding = q.parse_method(method)
            t1 = time.perf_counter()
            effective, rotated, encodings, damping, relative_error = {}, {}, {}, [], []
            for kind in kinds:
                rot = rotations[(kind, family)]
                Hr = rot.rotate_hessian(Hc[kind]) if rounding == "gptq" else None
                rotated[kind], encodings[kind], lam = _quantize(rot.forward(W[kind]), Hr, fmt, rounding)
                effective[kind] = rot.inverse(rotated[kind])
                damping += lam
                relative_error += (q.output_error(W[kind], effective[kind], Hc[kind]) / reference_error[kind]).tolist()
            extra = {"quantize_seconds": time.perf_counter() - t1,
                     "gptq_damping_max": max(damping) if damping else None,
                     "calibration_relative_output_error_mean": float(np.mean(relative_error))}
            hardware = {}
            if fmt_name == HW_FORMAT:
                # Hardware operands of every matrix decode bit for bit (int32 views) to the fake-quant
                # weights in rotated coordinates.
                checked, exact = 0, True
                for kind, names in kinds.items():
                    rot, enc = rotations[(kind, family)], encodings[kind]
                    for i, name in enumerate(names):
                        layer = nb.NVFP4Linear.from_encoding(
                            enc["codes"][i], enc["block_scale"][i], enc["global_scale"][i],
                            rotation=e2e.ROTATION_OF[family],
                            angles=rot.angles[i] if isinstance(rot, q.Butterfly) else None)
                        checked += 1
                        exact &= bool(torch.equal(nb.dequantized_weight(layer).view(torch.int32),
                                                  rotated[kind][i].contiguous().view(torch.int32)))
                        hardware[name] = layer
                extra.update(hw_weights_checked_matrices=checked, hw_weights_decode_to_fakequant_weights=exact)

            t1 = time.perf_counter()
            apply(effective)
            evals = evaluate_both()
            apply(W)
            grid.append(row(fmt_name, method, fakequant_mode(fmt), evals, time.perf_counter() - t1, **extra))
            if not (hardware and hardware_ok):
                continue

            t1 = time.perf_counter()  # dynamic per-tensor activation scale
            previous = e2e._swap(model, hardware)
            evals = evaluate_both()
            e2e._swap(model, previous)
            grid.append(row(fmt_name, method, "w4a4_hw_dynamic", evals, time.perf_counter() - t1))

            t1 = time.perf_counter()  # static activation scales from the train.bin calibration windows
            layer_rotations = {name: e2e._layer_rotation(rotations[(name.split(".")[2], family)],
                                                         int(name.split(".")[1])) for name in hardware}
            static = e2e.calibrate_activation_scales(model, calibration["array"], batch, device, layer_rotations)
            for name, layer in hardware.items():
                layer.act_global_scale = static[name]
            previous = e2e._swap(model, hardware)
            evals = evaluate_both()
            e2e._swap(model, previous)
            scales = [v.item() for v in static.values()]
            grid.append(row(fmt_name, method, "w4a4_hw_static", evals, time.perf_counter() - t1,
                            static_scale_calibration="train.bin calibration windows, BF16 model, max |x R^T| / 2688",
                            static_global_scale_range=[min(scales), max(scales)]))

            if method == EMULATED_METHOD:  # same W4A4 operands, FP32 GEMM: isolates the FP4 hardware GEMM
                for layer in hardware.values():
                    layer.act_global_scale = None
                t1 = time.perf_counter()
                previous = e2e._swap(model, {n: e2e.EmulatedW4A4Linear(m) for n, m in hardware.items()})
                evals = evaluate_both()
                e2e._swap(model, previous)
                grid.append(row(fmt_name, method, EMULATED_MODE, evals, time.perf_counter() - t1))
            del hardware
        del rotations
        if device.type == "cuda":
            torch.cuda.empty_cache()
    sha_after = e2e._checkpoint_sha256(model)
    if sha_after != sha_before:
        raise RuntimeError("The PTQ study did not restore the checkpoint weights")
    return {"checkpoint_sha256": sha_before, "checkpoint_sha256_after_ptq": sha_after,
            "baseline_val": baseline["val"], "baseline_test": baseline["test"], "diagnostics": diagnostics,
            "butterfly_fits": butterfly_fits, "grid": grid, "best": best_methods(grid),
            "hardware_w4a4": {"available": hardware_ok,
                              "note": None if hardware_ok else "skipped: needs CUDA sm_100+ with F.scaled_mm"},
            "seconds": time.perf_counter() - t0}


def stage_b(config: h.ModelConfig, data: h.ShardData, device: torch.device, protocol: dict, selection: dict,
            units, log) -> dict:
    """Locked paired runs: for each seed, Muon and AdamW from one init and one batch stream."""
    calibration = q.calibration_windows(data, protocol["calibration_sequences"], config.sequence_length,
                                        protocol["calibration_seed"])
    plan = {MUON: (selection["t_muon"], selection["muon_lr"]), ADAMW: (selection["t_adamw"], selection["adamw_lr"])}
    checkpoints = []
    for seed in protocol["seeds"]:
        init = h.initial_state(config, seed)
        for optimizer in OPTIMIZERS:
            steps, lr = plan[optimizer]

            def study() -> dict:
                model, training = train_checkpoint(optimizer, config, data.splits["train"], device, seed=seed,
                                                   steps=steps, lr=lr, init=init)
                log(f"  B seed {seed} {optimizer}: {steps} steps in {training['train_seconds']:.0f}s, "
                    f"train loss {training['train_loss']:.4f}")
                record = {"seed": seed, "optimizer": optimizer, "training": training,
                          **checkpoint_study(model, config, data, device, calibration, protocol, log,
                                             f"s{seed} {optimizer}")}
                if record["checkpoint_sha256"] != training["checkpoint_sha256"]:
                    raise RuntimeError("Checkpoint changed between training and the PTQ study")
                return record

            checkpoints.append(units.cached(("checkpoint", seed, optimizer, steps, lr), study))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {"seeds": list(protocol["seeds"]), "t_muon": selection["t_muon"], "t_adamw": selection["t_adamw"],
            "adamw_lr": selection["adamw_lr"],
            "calibration": {k: v for k, v in calibration.items() if k != "array"}, "checkpoints": checkpoints}


# ======================================================================================
# Statistics (pure functions of the stored rows; ``validate_results`` recomputes them)
# ======================================================================================
def mean_sd(values: list[float]) -> dict:
    """Mean and sample standard deviation (n - 1)."""
    n = len(values)
    mean = math.fsum(values) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {"values": list(values), "mean": mean, "sd": sd}


def paired_t_df2(diffs: list[float]) -> dict:
    """Exact paired t statistics of three differences (2 degrees of freedom).

    The Student-t CDF with df = 2 is 1/2 + t / (2 sqrt(t^2 + 2)), so the two-sided p-value is
    1 - |t| / sqrt(t^2 + 2). It is computed as 2 / (s (s + |t|)) with s = sqrt(t^2 + 2), which is
    the same number without cancellation at large |t|. With sd = 0, p is 1 (mean 0) or 0.
    """
    if len(diffs) != 3:
        raise ValueError("The exact df = 2 formulas need exactly three paired differences")
    stats = mean_sd(diffs)
    mean, sd = stats["mean"], stats["sd"]
    se = sd / math.sqrt(3)
    if sd > 0:
        t = mean / se
        s = math.sqrt(t * t + 2)
        p = 2 / (s * (s + abs(t)))
    else:
        t, p = None, (1.0 if mean == 0 else 0.0)
    return {"n": 3, "df": 2, "diffs": list(diffs), "mean": mean, "sd": sd, "se": se, "t": t,
            "ci95": [mean - T975_DF2 * se, mean + T975_DF2 * se], "p_two_sided": p}


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(pvalues)
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: (pvalues[j], j))):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def analyze(stage_b_result: dict, tolerance: float) -> dict:
    """Paired Muon-versus-AdamW statistics over the Stage B seeds. Differences are muon - adamw."""
    seeds = stage_b_result["seeds"]
    ckpt = {(c["seed"], c["optimizer"]): c for c in stage_b_result["checkpoints"]}
    if len(ckpt) != len(seeds) * 2 or any((s, o) not in ckpt for s in seeds for o in OPTIMIZERS):
        raise ValueError("Stage B needs exactly one checkpoint per (seed, optimizer)")
    muon, adamw = [ckpt[(s, MUON)] for s in seeds], [ckpt[(s, ADAMW)] for s in seeds]

    # Baseline (BF16) quality match of the locked configuration.
    pairs = []
    for m, a in zip(muon, adamw):
        tm, ta = m["training"], a["training"]
        pair = {"seed": m["seed"], "init_fingerprint": tm["init_fingerprint"],
                "stream_fingerprint": tm["stream_fingerprint"],
                "same_init_and_stream": (tm["init_fingerprint"] == ta["init_fingerprint"]
                                         and tm["stream_fingerprint"] == ta["stream_fingerprint"]),
                "muon_checkpoint_sha256": m["checkpoint_sha256"], "adamw_checkpoint_sha256": a["checkpoint_sha256"],
                "muon_train_loss": tm["train_loss"], "adamw_train_loss": ta["train_loss"]}
        for split in SPLITS:
            vm, va = m[f"baseline_{split}"]["nll"], a[f"baseline_{split}"]["nll"]
            pair.update({f"{split}_nll_muon": vm, f"{split}_nll_adamw": va,
                         f"{split}_gap_muon_minus_adamw": vm - va, f"abs_{split}_gap": abs(vm - va)})
        pair["val_gap_within_tolerance"] = pair["abs_val_gap"] <= tolerance
        pairs.append(pair)
    baseline = {"tolerance_nats": tolerance, "pairs": pairs,
                "all_seeds_within_tolerance_on_val": all(p["val_gap_within_tolerance"] for p in pairs)}
    for split in SPLITS:
        baseline[f"{split}_gap_muon_minus_adamw"] = paired_t_df2([p[f"{split}_gap_muon_minus_adamw"] for p in pairs])
        baseline[f"mean_abs_{split}_gap"] = math.fsum(p[f"abs_{split}_gap"] for p in pairs) / len(pairs)
        baseline[f"max_abs_{split}_gap"] = max(p[f"abs_{split}_gap"] for p in pairs)

    # Every (format, method, mode) combination must exist once in every checkpoint.
    def table(c):
        rows = {(r["format"], r["method"], r["mode"]): r for r in c["grid"]}
        if len(rows) != len(c["grid"]):
            raise ValueError("Duplicate PTQ grid row")
        return rows

    tables = {key: table(c) for key, c in ckpt.items()}
    combos = list(tables[(seeds[0], MUON)])
    if any(list(t) != combos for t in tables.values()):
        raise ValueError("PTQ grids differ between checkpoints")
    comparisons = []
    for combo in combos:
        entry = dict(zip(("format", "method", "mode"), combo))
        rows_m = [tables[(s, MUON)][combo] for s in seeds]
        rows_a = [tables[(s, ADAMW)][combo] for s in seeds]
        for split in SPLITS:
            block = {}
            for field in ("delta_nll", "relative_delta_ppl_pct", "nll"):
                block[f"muon_{field}"] = mean_sd([r[f"{split}_{field}"] for r in rows_m])
                block[f"adamw_{field}"] = mean_sd([r[f"{split}_{field}"] for r in rows_a])
            block["delta_nll_diff"] = paired_t_df2([m[f"{split}_delta_nll"] - a[f"{split}_delta_nll"]
                                                    for m, a in zip(rows_m, rows_a)])
            block["abs_nll_diff"] = paired_t_df2([m[f"{split}_nll"] - a[f"{split}_nll"]
                                                  for m, a in zip(rows_m, rows_a)])
            entry[split] = block
        comparisons.append(entry)
    for split in SPLITS:  # secondary: Holm over every combination
        for metric in ("delta_nll_diff", "abs_nll_diff"):
            adjusted = holm_bonferroni([c[split][metric]["p_two_sided"] for c in comparisons])
            for c, p in zip(comparisons, adjusted):
                c[split][metric]["p_holm_all_combinations"] = p

    # Method selection on val (confirmed on test), pooled and per optimizer.
    pooled_rows = [r for c in ckpt.values() for r in c["grid"]]
    selection = {"pooled": best_methods(pooled_rows),
                 MUON: best_methods([r for c in muon for r in c["grid"]]),
                 ADAMW: best_methods([r for c in adamw for r in c["grid"]])}

    # Primary family: one comparison per (format, mode) cell with more than one method, at the
    # method that the pooled val NLL selects. Endpoints are read on test.
    by_combo = {(c["format"], c["method"], c["mode"]): c for c in comparisons}
    primary = []
    for cell in selection["pooled"].values():
        if len(cell["methods"]) < 2:
            continue
        c = by_combo[(cell["format"], cell["best_by_val"], cell["mode"])]["test"]
        primary.append({"format": cell["format"], "mode": cell["mode"], "method": cell["best_by_val"],
                        "method_selected_on": "val", "best_by_test": cell["best_by_test"],
                        "confirmed_on_test": cell["confirmed_on_test"],
                        "muon_delta_nll": c["muon_delta_nll"], "adamw_delta_nll": c["adamw_delta_nll"],
                        "muon_relative_delta_ppl_pct": c["muon_relative_delta_ppl_pct"],
                        "adamw_relative_delta_ppl_pct": c["adamw_relative_delta_ppl_pct"],
                        "muon_nll": c["muon_nll"], "adamw_nll": c["adamw_nll"],
                        **{metric: {k: v for k, v in c[metric].items() if k != "p_holm_all_combinations"}
                           for metric in ("delta_nll_diff", "abs_nll_diff")}})
    for metric in ("delta_nll_diff", "abs_nll_diff"):
        for entry, p in zip(primary, holm_bonferroni([e[metric]["p_two_sided"] for e in primary])):
            entry[metric]["p_holm"] = p
            entry[metric]["significant_holm_0p05"] = p < 0.05

    # Hardware FP4 GEMM against the FP32 GEMM of the same quantized operands.
    emulation = []
    for key, rows in tables.items():
        hw = rows.get((HW_FORMAT, EMULATED_METHOD, "w4a4_hw_dynamic"))
        emu = rows.get((HW_FORMAT, EMULATED_METHOD, EMULATED_MODE))
        if hw and emu:
            emulation.append({"seed": key[0], "optimizer": key[1],
                              **{f"{split}_nll_hw_minus_emulated": hw[f"{split}_nll"] - emu[f"{split}_nll"]
                                 for split in SPLITS}})
    worst = max((abs(e[f"{s}_nll_hw_minus_emulated"]) for e in emulation for s in SPLITS), default=None)
    parity = {"method": EMULATED_METHOD, "rows": emulation, "max_abs_nll_difference": worst,
              "threshold_nats": EMULATION_PARITY_NATS, "passed": worst is not None and worst <= EMULATION_PARITY_NATS}

    # Paired diagnostics (descriptive; p-values are not corrected).
    diagnostics = {}
    for metric in DIAGNOSTICS:
        diagnostics[metric] = {}
        for scope in ("all", *q.LINEAR_KINDS):
            def pick(c, stat):
                s = c["diagnostics"]["summary"][metric]
                return (s["all"] if scope == "all" else s["by_kind"][scope])[stat]
            diagnostics[metric][scope] = {
                stat: {"muon": [pick(c, stat) for c in muon], "adamw": [pick(c, stat) for c in adamw],
                       **paired_t_df2([pick(m, stat) - pick(a, stat) for m, a in zip(muon, adamw)])}
                for stat in ("mean", "median", "max")}
    return {"sign_convention": "every difference is muon - adamw; a negative delta_nll_diff means that Muon "
                               "loses less quality under PTQ, a negative abs_nll_diff that its quantized NLL is lower",
            "t_quantile_0p975_df2": T975_DF2, "t_quantile_0p975_df2_closed_form": T975_DF2_CLOSED_FORM,
            "t_quantile_note": "ci95 uses t_quantile_0p975_df2, the constant fixed by the protocol. The closed "
                               "form sqrt(2 * 0.9025 / 0.0975) is 1.6e-10 smaller (4e-11 of an interval half-width). "
                               "p-values use the exact df = 2 formula 1 - |t| / sqrt(t^2 + 2).",
            "baseline": baseline, "comparisons": comparisons,
            "method_selection": selection, "primary": primary,
            "primary_note": "Holm-Bonferroni family: the (format, mode) cells with more than one method, each at "
                            "the method with the lowest pooled val NLL, test endpoints. delta_nll_diff is the "
                            "primary endpoint; abs_nll_diff is corrected as a separate family.",
            "hardware_vs_emulation": parity, "diagnostics_paired": diagnostics}


def summarize(payload: dict) -> dict:
    """Headline numbers copied from the stored stages and the analysis."""
    selection, analysis = payload["stage_a_pilot"]["selection"], payload["analysis"]
    base = analysis["baseline"]
    return {
        "calibrated": {k: selection[k] for k in ("matched", "t_muon", "t_adamw", "adamw_lr", "abs_val_gap")},
        "stage_b_baseline": {
            "abs_val_gap_per_seed": [p["abs_val_gap"] for p in base["pairs"]],
            "abs_test_gap_per_seed": [p["abs_test_gap"] for p in base["pairs"]],
            "mean_abs_val_gap": base["mean_abs_val_gap"], "mean_abs_test_gap": base["mean_abs_test_gap"],
            "all_seeds_within_tolerance_on_val": base["all_seeds_within_tolerance_on_val"]},
        "primary_test": [{"format": e["format"], "mode": e["mode"], "method": e["method"],
                          "muon_delta_nll_mean": e["muon_delta_nll"]["mean"],
                          "adamw_delta_nll_mean": e["adamw_delta_nll"]["mean"],
                          "delta_nll_diff_mean": e["delta_nll_diff"]["mean"],
                          "delta_nll_diff_ci95": e["delta_nll_diff"]["ci95"],
                          "delta_nll_diff_p": e["delta_nll_diff"]["p_two_sided"],
                          "delta_nll_diff_p_holm": e["delta_nll_diff"]["p_holm"],
                          "abs_nll_diff_mean": e["abs_nll_diff"]["mean"],
                          "abs_nll_diff_ci95": e["abs_nll_diff"]["ci95"],
                          "abs_nll_diff_p_holm": e["abs_nll_diff"]["p_holm"]} for e in analysis["primary"]],
        "any_primary_delta_nll_diff_significant_holm_0p05": any(
            e["delta_nll_diff"]["significant_holm_0p05"] for e in analysis["primary"]),
        "hardware_vs_emulation_max_abs_nll_difference": analysis["hardware_vs_emulation"]["max_abs_nll_difference"]}


def caveats(payload: dict) -> list[str]:
    selection, base = payload["stage_a_pilot"]["selection"], payload["analysis"]["baseline"]
    unstable = payload["stage_a_pilot"]["adamw_non_monotone"]
    notes = [
        "Three paired seeds give 2 degrees of freedom: the 95% interval is 4.30 standard errors wide on each "
        "side, and small consistent effects cannot pass a Holm correction. Absence of significance is not "
        "evidence of equal PTQ sensitivity.",
        "The iso-loss match is calibrated on seed 0 and then locked. The Stage B baselines of seeds 1 to 3 "
        "differ from seed 0, so per-seed baseline gaps can exceed the calibration tolerance; they are reported.",
        "Iso-loss is not iso-compute: AdamW uses "
        f"{selection['adamw_over_muon_tokens']:.2f}x the training tokens of Muon to reach the same val NLL.",
        "One 50m model family, short training (less than one epoch of a 26M-token set), one calibration set. "
        "PTQ sensitivity of longer-trained or larger models can differ.",
        "Hessians come from one forward pass of the full-precision model (parallel GPTQ). Static activation "
        "scales are calibrated on the BF16 model, not on the upstream-quantized model.",
        "Dynamic W4A4 activation scales are per forward batch (16 x 512 tokens), so they depend on the batch.",
        "Diagnostics p-values are descriptive and not corrected for multiplicity.",
    ]
    if unstable:
        worst = max(unstable, key=lambda u: u["val_nll_excess"])
        notes.insert(0, f"AdamW val NLL is not monotone in the step count on the pilot seed: {len(unstable)} trial(s) "
                        f"are worse than a shorter trial by more than the tolerance (worst: {worst['steps']} steps, "
                        f"LR {worst['adamw_lr']:g}, +{worst['val_nll_excess']:.3f} nats). AdamW runs at these LRs "
                        "have run-to-run instability, which limits how well a seed-0 match transfers to other seeds.")
    if not selection["matched"]:
        notes.insert(0, "Stage A found no (T_muon, T_adamw, LR) within the tolerance; the closest pair was used. "
                        "The comparison is not quality matched.")
    if not base["all_seeds_within_tolerance_on_val"]:
        notes.insert(0, "At least one Stage B seed has a baseline val NLL gap above the tolerance "
                        f"(max {base['max_abs_val_gap']:.4f} nats).")
    return notes


# ======================================================================================
# Validation
# ======================================================================================
def _assert_close(found, expected, path: str, tol: float = TOL) -> None:
    """Deep equality; numbers may differ by ``tol`` (absolute, or relative above magnitude 1)."""
    if isinstance(expected, dict):
        if not isinstance(found, dict) or list(found) != list(expected):
            raise ValueError(f"{path}: keys differ from the recomputed value")
        for key, value in expected.items():
            _assert_close(found[key], value, f"{path}.{key}", tol)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(found, (list, tuple)) or len(found) != len(expected):
            raise ValueError(f"{path}: length differs from the recomputed value")
        for i, value in enumerate(expected):
            _assert_close(found[i], value, f"{path}[{i}]", tol)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if found is not expected and found != expected or isinstance(found, bool) != isinstance(expected, bool):
            raise ValueError(f"{path}: {found!r} differs from the recomputed {expected!r}")
    else:
        if isinstance(found, bool) or not isinstance(found, (int, float)) or \
                not abs(found - expected) <= tol * max(1.0, abs(expected)):
            raise ValueError(f"{path}: {found!r} differs from the recomputed {expected!r}")


def _keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _keys(value)


def _is_hex(value, length: int) -> bool:
    return (isinstance(value, str) and len(value) == length
            and all(c in "0123456789abcdefABCDEF" for c in value))


def _validate_training(row: dict, path: str, config: dict, protocol: dict) -> None:
    if row["warmup_steps"] != warmup_for(row["steps"]) or row["weight_decay"] != WEIGHT_DECAY \
            or row["min_lr_ratio"] != MIN_LR_RATIO:
        raise ValueError(f"{path}: schedule differs from warmup max(20, T // 10), weight decay 0.01, min LR 0.1")
    step_tokens = config["micro_batch"] * config["grad_accum"] * config["sequence_length"]
    if row["tokens"] != row["steps"] * step_tokens:
        raise ValueError(f"{path}: token count {row['tokens']!r} differs from steps * {step_tokens}")
    expected_aux = protocol["muon"]["auxiliary_lr"] if row["optimizer"] == MUON else None
    if row["auxiliary_lr"] != expected_aux:
        raise ValueError(f"{path}: auxiliary_lr {row['auxiliary_lr']!r} differs from {expected_aux!r}")
    if not _is_hex(row["checkpoint_sha256"], 64):
        raise ValueError(f"{path}: missing checkpoint SHA-256")


def validate_results(payload: dict, *, canonical: bool | None = None) -> None:
    """Raise if a result is incomplete, non-finite, leaks ``test.bin`` into Stage A, or if any stored
    summary differs from its recomputation from the stored rows by more than 1e-12.

    ``canonical`` (default: ``not payload["quick"]``) adds the checks of the full RTX 5080 run:
    pinned shard hashes, full val/test token counts, the complete 25-row grid with hardware W4A4,
    the locked protocol constants, a clean git tree, and an sm_120+ device.
    """
    json.dumps(payload, allow_nan=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected schema version")
    canonical = (not payload["quick"]) if canonical is None else canonical
    meta = payload["meta"]
    for key in ("git_sha", "git_dirty", "command", "environment", "gpu_telemetry", "loadavg_start", "loadavg_end",
                "config_hash", "code_sha256"):
        if key not in meta:
            raise ValueError(f"Missing meta key {key!r}")
    if not _is_hex(meta["git_sha"], 40):
        raise ValueError("meta.git_sha must be a 40-char hex SHA")
    if not _is_hex(meta["code_sha256"], 64):
        raise ValueError("meta.code_sha256 must be a 64-char hex SHA-256")
    if not _is_hex(meta["config_hash"], 16):
        raise ValueError("meta.config_hash must be a 16-char hex digest")
    if not isinstance(meta["git_dirty"], bool):
        raise ValueError("meta.git_dirty must be a bool")
    protocol, data, pilot, stage = payload["protocol"], payload["data"], payload["stage_a_pilot"], payload["stage_b"]
    tolerance, seeds = protocol["tolerance_nats"], protocol["seeds"]
    config = payload["config"]
    layers, seq_len = config["layers"], config["sequence_length"]
    matrices = layers * len(q.LINEAR_KINDS)
    train_tokens = data["splits"]["train"]["tokens"]

    # Data hygiene.
    if (data.get("training_split") != "train" or data["calibration_split"] != "train"
            or data["static_scale_split"] != "train" or data["selection_split"] != "val"
            or data["evaluation_split"] != "test" or stage["calibration"]["split"] != "train"):
        raise ValueError("Calibration must use train.bin, selection val.bin and evaluation test.bin")
    cal = stage["calibration"]
    starts = cal.get("starts")
    if (cal["sequences"] != protocol["calibration_sequences"] or cal["seed"] != protocol["calibration_seed"]
            or cal.get("tokens") != protocol["calibration_sequences"] * seq_len
            or not isinstance(starts, list) or len(starts) != protocol["calibration_sequences"]
            or len(set(starts)) != len(starts)
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 or s > train_tokens - seq_len for s in starts)):
        raise ValueError("Calibration windows differ from the protocol")
    if canonical:
        found = {k: v["sha256"] for k, v in data["splits"].items()}
        if not data["sha256_verified"] or data["expected_sha256"] != q.EXPECTED_SHA256 \
                or any(found.get(k) != v for k, v in q.EXPECTED_SHA256.items()):
            raise ValueError("Shard SHA-256 values differ from the pinned train/val/test hashes")
        if payload["quick"] or meta["environment"].get("capability", [0, 0]) < [12, 0]:
            raise ValueError("A canonical result needs a full run on a Blackwell (sm_120+) CUDA device")
        locked = dict(_CANONICAL_PROTOCOL, max_eval_windows=None)
        if any(protocol.get(k) != v for k, v in locked.items()):
            raise ValueError("Protocol differs from the canonical C4-01b protocol")
        if meta["git_dirty"] is not False:
            raise ValueError("A canonical result must come from a clean git tree")

    # Stage A: seed 0, val.bin only.
    if pilot["seed"] != protocol["pilot_seed"] or pilot["seed"] in seeds:
        raise ValueError("The Stage A seed must be disjoint from the Stage B seeds")
    if pilot["split_reads"].get("test") != 0 or set(pilot["splits_used"]) != {"train", "val"}:
        raise ValueError("Stage A must not read test.bin")
    if not isinstance(pilot.get("search"), list):
        raise ValueError("Stage A must include the search trace list")
    if any("test" in key for key in _keys([pilot["rows"], pilot["search"], pilot["selection"]])):
        raise ValueError("Stage A holds a test.bin field")
    for i, row in enumerate(pilot["rows"]):
        _validate_training(row, f"stage_a_pilot.rows[{i}]", config, protocol)
        if row["seed"] != pilot["seed"] or row["evaluated_splits"] != ["val"]:
            raise ValueError("Every Stage A row must be a seed-0 run evaluated on val.bin only")
        steps, lr = row["steps"], row["lr"]
        if row["optimizer"] == ADAMW and (steps > protocol["adamw_max_steps"] or lr not in protocol["adamw_lrs"]):
            raise ValueError("A Stage A AdamW row is outside the step budget or the LR grid")
        if row["optimizer"] == MUON and (steps not in protocol["t_muon_candidates"] or lr != protocol["muon"]["lr"]):
            raise ValueError("A Stage A Muon row is outside the candidate grid")
        if canonical and (row["val_tokens"] != VAL_TOKENS or not row["tokens"] < train_tokens):
            raise ValueError("Stage A rows need the full val.bin and less than one epoch of train.bin")
        _assert_close(row["val_ppl"], math.exp(row["val_nll"]), f"stage_a_pilot.rows[{i}].val_ppl")
    full_adamw_lrs = {r["lr"] for r in pilot["rows"]
                      if r["optimizer"] == ADAMW and r["steps"] == protocol["adamw_max_steps"]}
    if any(lr not in full_adamw_lrs for lr in protocol["adamw_lrs"]):
        raise ValueError("Stage A must evaluate every AdamW LR at adamw_max_steps")
    selection = select_isoloss(pilot["rows"], tolerance=tolerance, t_muon_candidates=protocol["t_muon_candidates"],
                               adamw_max_steps=protocol["adamw_max_steps"])
    _assert_close(pilot["selection"], selection, "stage_a_pilot.selection")
    _assert_close(pilot["adamw_non_monotone"], non_monotone_trials(pilot["rows"], tolerance),
                  "stage_a_pilot.adamw_non_monotone")
    larger = [t for t in protocol["t_muon_candidates"] if t > selection["t_muon"]]
    if selection["matched"] and set(larger) - set(selection["t_muon_candidates_evaluated"]):
        raise ValueError("A larger T_muon candidate was not evaluated before the selected one")
    if canonical and not selection["matched"]:
        raise ValueError("The canonical run needs a Stage A match within the tolerance")

    # Stage B: locked configuration, paired seeds, complete grids.
    if (stage["seeds"] != seeds or len(seeds) != 3 or stage["t_muon"] != selection["t_muon"]
            or stage["t_adamw"] != selection["t_adamw"] or stage["adamw_lr"] != selection["adamw_lr"]):
        raise ValueError("Stage B does not use the locked Stage A selection on three seeds")
    plan = {MUON: (selection["t_muon"], selection["muon_lr"]), ADAMW: (selection["t_adamw"], selection["adamw_lr"])}
    expected_combos = [(f, m, fakequant_mode(q.FORMATS[f]))
                       for f in protocol["weight_formats"] for m in protocol["methods"]]
    hw_combos = ([(HW_FORMAT, m, mode) for m in protocol["methods"] for mode in HW_MODES]
                 + [(HW_FORMAT, EMULATED_METHOD, EMULATED_MODE)])
    expected_layers = {f"blocks.{i}.{k}" for i in range(layers) for k in q.LINEAR_KINDS}
    if [(c["seed"], c["optimizer"]) for c in stage["checkpoints"]] != [(s, o) for s in seeds for o in OPTIMIZERS]:
        raise ValueError("Stage B needs one Muon and one AdamW checkpoint per seed, in order")
    for c in stage["checkpoints"]:
        path = f"stage_b[{c['seed']}, {c['optimizer']}]"
        training = c["training"]
        _validate_training(training, path, config, protocol)
        if (training["seed"], training["optimizer"], (training["steps"], training["lr"])) != \
                (c["seed"], c["optimizer"], plan[c["optimizer"]]):
            raise ValueError(f"{path}: training differs from the locked selection")
        if not (c["checkpoint_sha256"] == c["checkpoint_sha256_after_ptq"] == training["checkpoint_sha256"]):
            raise ValueError(f"{path}: checkpoint SHA-256 values differ")
        base = {split: c[f"baseline_{split}"] for split in SPLITS}
        if canonical and (base["val"]["tokens"], base["test"]["tokens"]) != (VAL_TOKENS, TEST_TOKENS):
            raise ValueError(f"{path}: baseline does not cover the full val.bin and test.bin")
        for split in SPLITS:
            _assert_close(base[split]["ppl"], math.exp(base[split]["nll"]), f"{path}.baseline_{split}.ppl")
        combos = [(r["format"], r["method"], r["mode"]) for r in c["grid"]]
        hardware = c["hardware_w4a4"]["available"]
        if sorted(combos) != sorted(expected_combos + (hw_combos if hardware else [])) or (canonical and not hardware):
            raise ValueError(f"{path}: incomplete PTQ grid")
        for r in c["grid"]:
            where = f"{path}.{r['format']}/{r['method']}/{r['mode']}"
            for split in SPLITS:
                if r[f"{split}_tokens"] != base[split]["tokens"]:
                    raise ValueError(f"{where}: {split} token count differs from the baseline")
                _assert_close(r[f"{split}_ppl"], math.exp(r[f"{split}_nll"]), f"{where}.{split}_ppl")
                _assert_close(r[f"{split}_delta_nll"], r[f"{split}_nll"] - base[split]["nll"],
                              f"{where}.{split}_delta_nll")
                _assert_close(r[f"{split}_relative_delta_ppl_pct"],
                              100.0 * (r[f"{split}_ppl"] / base[split]["ppl"] - 1.0),
                              f"{where}.{split}_relative_delta_ppl_pct")
            if r["format"] == HW_FORMAT and r["mode"].endswith("fakequant") and (
                    r.get("hw_weights_decode_to_fakequant_weights") is not True
                    or r.get("hw_weights_checked_matrices") != matrices):
                raise ValueError(f"{where}: hardware weights must decode to the fake-quant weights in all "
                                 f"{matrices} matrices")
            if r["mode"] == "w4a4_hw_static":
                cal_desc = str(r.get("static_scale_calibration", ""))
                if "train.bin" not in cal_desc or "test" in cal_desc.lower():
                    raise ValueError(f"{where}: static activation scale calibration must come from train.bin")
        per_layer = c["diagnostics"]["per_layer"]
        if len(per_layer) != matrices:
            raise ValueError(f"{path}: diagnostics must cover all {matrices} linear layers")
        if {r["name"] for r in per_layer} != expected_layers or any(
                r["kind"] != r["name"].rsplit(".", 1)[1] for r in per_layer):
            raise ValueError(f"{path}: diagnostics layer names or kinds do not match model projections")
        _assert_close(c["diagnostics"]["summary"], summarize_diagnostics(per_layer),
                      f"{path}.diagnostics.summary")
        _assert_close(c["best"], best_methods(c["grid"]), f"{path}.best")

    # Analysis, summary and caveats are pure functions of the rows above.
    analysis = analyze(stage, tolerance)
    _assert_close(payload["analysis"], analysis, "analysis")
    if not all(p["same_init_and_stream"] for p in analysis["baseline"]["pairs"]):
        raise ValueError("Muon and AdamW of a seed must share the initial state and the batch stream")
    if canonical and not analysis["hardware_vs_emulation"]["passed"]:
        raise ValueError("Hardware W4A4 differs from its FP32 emulation by more than the parity threshold")
    _assert_close(payload["summary"], summarize(payload), "summary")
    _assert_close(payload["caveats"], caveats(payload), "caveats")


# ======================================================================================
# Resume log and output
# ======================================================================================
class UnitLog:
    """Finished units (pilot rows, checkpoint studies) as JSON lines next to the output.

    A unit is keyed by (stage, seed, optimizer, steps, lr, config_hash). ``--resume`` reloads the
    units of an interrupted run with the same ``config_hash``; without it the file starts empty.
    Model weights are never stored, so an interrupted unit is trained again.
    """

    def __init__(self, path: Path | None, config_hash: str, resume: bool):
        self.path, self.config_hash, self.units = path, config_hash, {}
        self.loaded = self.run = 0
        if path is None:
            return
        if resume and path.exists():
            for line in path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut off by the interruption
                self.units[tuple(record["key"])] = record["value"]
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    def cached(self, key: tuple, compute) -> dict:
        key = (*key, self.config_hash)
        if key in self.units:
            self.loaded += 1
            return {**self.units[key], "resumed_from_unit_log": True}
        value = json.loads(json.dumps(compute(), allow_nan=False))  # plain JSON types, as after a reload
        self.run += 1
        if self.path is not None:
            with self.path.open("a") as handle:
                handle.write(json.dumps({"key": list(key), "value": value}, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return {**value, "resumed_from_unit_log": False}

def unit_log_path(output: Path) -> Path:
    return output.with_name(output.name + ".units.jsonl")


def code_sha256() -> str:
    """SHA-256 of the source files that define training, quantization, data, and Muon kernels."""
    digest = hashlib.sha256()
    extra_modules = (
        importlib.import_module(name)
        for name in (
            "research.polar_opt.real_text_data",
            "research.polar_opt.tier2_candidates",
            "research.polar_opt.schatten_switch_blackwell",
            "research.polar_opt.polar_switch_blackwell",
            "research.polar_opt.kernels_blackwell",
        )
    )
    for module in (sys.modules[__name__], q, nb, e2e, h, harness_tier2_lm, *extra_modules):
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> dict:
    """Lossless atomic write: strict JSON (no NaN), read back and compared, synced, then renamed.

    Returns the payload as read back (plain JSON types).
    """
    text = e2e._dumps_json(payload)  # 2-space JSON, leaf lists on one line, allow_nan=False
    reloaded = json.loads(text)
    if reloaded != json.loads(json.dumps(payload, allow_nan=False)):
        raise ValueError("JSON round trip is not lossless")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    return reloaded


# ======================================================================================
# Driver
# ======================================================================================
def build_protocol(args) -> dict:
    spec = q.MODEL_SPECS[MUON]
    return {"preset": args.preset, "pilot_seed": args.pilot_seed, "seeds": list(args.seeds),
            "tolerance_nats": args.tolerance, "t_muon_candidates": list(args.t_muon_candidates),
            "adamw_lrs": list(args.adamw_lrs), "adamw_max_steps": args.adamw_max_steps,
            "adamw_step_quantum": args.adamw_step_quantum,
            "muon": {"optimizer": MUON, "lr": spec["lr"], "auxiliary_lr": spec["auxiliary_lr"]},
            "adamw": {"optimizer": ADAMW}, "weight_decay": WEIGHT_DECAY, "min_lr_ratio": MIN_LR_RATIO,
            "schedule": "linear warmup over max(20, T // 10) steps, then cosine decay to min_lr_ratio",
            "calibration_sequences": args.calibration_sequences, "calibration_seed": args.calibration_seed,
            "max_eval_windows": args.max_eval_windows, "butterfly_steps": args.butterfly_steps,
            "gptq_damp": GPTQ_DAMP, "weight_formats": list(q.FORMATS), "methods": list(METHODS),
            "hardware_format": HW_FORMAT, "hardware_modes": list(HW_MODES),
            "emulated": [EMULATED_METHOD, EMULATED_MODE]}


def require_blackwell(device: torch.device, output: Path | None) -> None:
    """The canonical filename promises an RTX 5080-class (sm_120+) result."""
    if output is not None and _targets_canonical_filename(output) and (
            device.type != "cuda" or not torch.cuda.is_available()
            or torch.cuda.get_device_capability(device) < (12, 0)):
        raise RuntimeError(f"{CANONICAL_OUTPUT.name} requires a Blackwell (sm_120+) CUDA device")


def run_experiment(args) -> dict:
    device = torch.device(args.device)
    output = getattr(args, "output", None)
    require_blackwell(device, output)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    log = (lambda *a, **k: None) if args.silent else (lambda *a, **k: print(*a, **k, flush=True))
    started = time.perf_counter()
    data = load_data(args)
    config = h.PRESETS[args.preset]
    protocol = build_protocol(args)
    environment = q.environment(device)
    hashed = {"schema_version": SCHEMA_VERSION, "config": asdict(config), "protocol": protocol,
              "data_sha256": {k: v["sha256"] for k, v in data.summary()["splits"].items()},
              "code_sha256": code_sha256(), "torch": environment["torch"],
              "triton": environment.get("triton"), "cuda": environment.get("cuda"),
              "device": environment.get("device", "cpu")}
    config_hash = hashlib.sha256(json.dumps(hashed, sort_keys=True).encode()).hexdigest()[:16]
    units = UnitLog(unit_log_path(output) if output is not None else None, config_hash, args.resume)
    meta = {**h.git_info(), "command": [sys.executable, *sys.argv], "args": {k: str(v) for k, v in vars(args).items()},
            "environment": environment, "config_hash": config_hash, "code_sha256": hashed["code_sha256"],
            "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "loadavg_start": list(os.getloadavg()),
            "gpu_telemetry": {"start": h.gpu_telemetry(device)},
            "numerics": {"allow_tf32": False, "cudnn_allow_tf32": False,
                         "allow_fp16_reduced_precision_reduction": False,
                         "allow_bf16_reduced_precision_reduction": False}}
    log(f"git {meta['git_sha']} dirty={meta['git_dirty']} device={environment.get('device', 'cpu')} "
        f"config {config_hash} resume units {len(units.units)}")
    payload = {"schema_version": SCHEMA_VERSION,
               "task": "Cycle 4 C4-01b: iso-loss multi-seed Muon-vs-AdamW PTQ and Blackwell hardware W4A4",
               "quick": args.quick, "meta": meta, "preset": args.preset, "config": asdict(config),
               "protocol": protocol,
               "data": {**data.summary(), "directory": str(args.data_dir), "sha256_verified": not args.no_verify,
                        "expected_sha256": None if args.no_verify else q.EXPECTED_SHA256,
                        "training_split": "train", "calibration_split": "train", "static_scale_split": "train",
                        "selection_split": "val", "evaluation_split": "test",
                        "test_used_for": "Stage B evaluation only; locked during Stage A"}}
    payload["stage_a_pilot"] = stage_a_pilot(config, data, device, protocol, units, log)
    selection = payload["stage_a_pilot"]["selection"]
    meta["gpu_telemetry"]["after_stage_a"] = h.gpu_telemetry(device)
    log(f"Stage A: matched={selection['matched']} T_muon {selection['t_muon']} T_adamw {selection['t_adamw']} "
        f"adamw_lr {selection['adamw_lr']:g} |val gap| {selection['abs_val_gap']:.4f}")
    payload["stage_b"] = stage_b(config, data, device, protocol, selection, units, log)
    payload["analysis"] = analyze(payload["stage_b"], protocol["tolerance_nats"])
    payload["summary"] = summarize(payload)
    payload["caveats"] = caveats(payload)
    meta["unit_log"] = {"resume": args.resume, "units_loaded": units.loaded, "units_run": units.run}
    meta["split_reads_total"] = dict(data.splits.reads)
    meta["gpu_telemetry"]["end"] = h.gpu_telemetry(device)
    meta["loadavg_end"] = list(os.getloadavg())
    meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["total_seconds"] = time.perf_counter() - started
    payload = json.loads(json.dumps(payload, allow_nan=False))
    validate_results(payload, canonical=_is_canonical_args(args))
    return payload


_CANONICAL_PROTOCOL = {"preset": "50m", "pilot_seed": 0, "seeds": [1, 2, 3], "tolerance_nats": 0.02,
                       "t_muon_candidates": [320, 256, 224, 192], "adamw_lrs": [0.001, 0.0015],
                       "adamw_max_steps": 768, "adamw_step_quantum": 8, "calibration_sequences": 128,
                       "calibration_seed": 20261126, "butterfly_steps": 150}
_CANONICAL = {**_CANONICAL_PROTOCOL, "tolerance": 0.02, "device": "cuda", "max_eval_windows": None,
              "no_verify": False, "quick": False}
del _CANONICAL["tolerance_nats"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    c = _CANONICAL
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--device", default=c["device"])
    parser.add_argument("--preset", default=c["preset"], choices=sorted(h.PRESETS))
    parser.add_argument("--pilot-seed", type=int, default=c["pilot_seed"], help="Stage A seed")
    parser.add_argument("--seeds", type=int, nargs=3, default=c["seeds"], help="the three Stage B seeds")
    parser.add_argument("--tolerance", type=float, default=c["tolerance"], help="iso-loss val NLL tolerance (nats)")
    parser.add_argument("--t-muon-candidates", type=int, nargs="+", default=c["t_muon_candidates"])
    parser.add_argument("--adamw-lrs", type=float, nargs="+", default=c["adamw_lrs"])
    parser.add_argument("--adamw-max-steps", type=int, default=c["adamw_max_steps"])
    parser.add_argument("--adamw-step-quantum", type=int, default=c["adamw_step_quantum"])
    parser.add_argument("--calibration-sequences", type=int, default=c["calibration_sequences"])
    parser.add_argument("--calibration-seed", type=int, default=c["calibration_seed"])
    parser.add_argument("--max-eval-windows", type=int, default=None, help="limit val/test windows (default: all)")
    parser.add_argument("--butterfly-steps", type=int, default=c["butterfly_steps"])
    parser.add_argument("--no-verify", action="store_true", help="skip SHA-256 checks (synthetic shards only)")
    parser.add_argument("--quick", action="store_true", help="tiny model, few steps, short evaluation")
    parser.add_argument("--resume", action="store_true", help="reuse finished units of an interrupted run")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _is_canonical_args(args: argparse.Namespace) -> bool:
    return all(getattr(args, k) == v for k, v in _CANONICAL.items())


def _targets_canonical_filename(path: Path) -> bool:
    """True for the canonical filename in any checkout, directly or through a symlink."""
    return path.name == CANONICAL_OUTPUT.name or path.resolve().name == CANONICAL_OUTPUT.name


def _resolve_output(args: argparse.Namespace) -> Path:
    canonical = _is_canonical_args(args)
    if args.output is None:
        if args.quick:
            return QUICK_OUTPUT
        if canonical:
            return CANONICAL_OUTPUT
        raise ValueError(f"Non-canonical run requires an explicit --output path to avoid overwriting "
                         f"{CANONICAL_OUTPUT.name}")
    if _targets_canonical_filename(args.output) and not canonical:
        raise ValueError(f"Refusing to overwrite canonical {CANONICAL_OUTPUT.name} from a --quick or "
                         "non-canonical run; choose a different --output path")
    return args.output


def parse_args(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.data_dir is None:
        args.data_dir = q.default_data_dir()
    if args.pilot_seed in args.seeds or len(set(args.seeds)) != 3:
        parser.error("--seeds must be three distinct seeds that exclude --pilot-seed")
    if args.adamw_max_steps % args.adamw_step_quantum:
        parser.error("--adamw-max-steps must be a multiple of --adamw-step-quantum")
    try:
        args.output = _resolve_output(args)  # judge the guard on the arguments as given
    except ValueError as exc:
        parser.error(str(exc))
    if args.quick:
        args.preset, args.t_muon_candidates = "tiny", [32, 24]
        args.adamw_max_steps, args.adamw_step_quantum = 64, 8
        args.calibration_sequences, args.max_eval_windows = 8, args.max_eval_windows or 64
        args.butterfly_steps = 10
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    payload = run_experiment(args)
    payload = write_json_atomic(args.output, payload)
    unit_log_path(args.output).unlink(missing_ok=True)
    if not args.silent:
        print(f"Wrote {args.output}")
    return payload


if __name__ == "__main__":
    main()
