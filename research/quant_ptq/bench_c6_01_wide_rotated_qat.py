"""Cycle 6 C6-01: wide Hadamard-rotated Muon W4A4 QAT and cross-block activation quantization mitigation.

C5-02 (50m) and C5-03 (120m) found that activation quantization causes about 80% of the hardware
NVFP4 W4A4 loss, that block-16 rotations only mix inside one NVFP4 group, that identity-coordinate
QAT costs BF16 loss, and that AWQ + butterfly helps modestly. This benchmark tests rotations that mix
across groups (``rotated_qat``) and STE fake W4A4 training in those coordinates.

Part A  50m, three seeds, plain ``muon_ss3_54`` checkpoints. Rotation-width probe: identity, block 16,
        block 64, full-width Kronecker-Hadamard, block-16 butterfly, and the two cross-mix + butterfly
        composites, each with GPTQ, on real hardware NVFP4 W4A4 (dynamic and static scales), plus
        activation statistics in the rotated coordinates (kurtosis, max / RMS, group-of-16 crest
        factor, NVFP4 rounding error) and AWQ variants.
Part B  50m, the same three seeds, equal token budgets and batches. Arms: plain Muon; identity QAT
        (cooldown, full run); block-16 QAT cooldown; wide QAT cooldown; wide QAT cooldown with the dual
        BF16 + W4A4 loss. The wide family is the better of block 64 and full width in Part A on val.bin.
Part C  120m. The three preserved C5-03 checkpoints (640 steps, digests pinned), a plain continuation
        of ``cooldown_steps`` extra steps (the token-matched control) and the same extra steps with the
        rotated QAT cooldown that Part B selects on val.bin.

Split hygiene. One freeze for the whole study: every training run, calibration, static scale, AWQ
alpha, and every selection (wide family, best PTQ methods, transfer arm) uses train.bin and val.bin.
The guarantee is zero pre-freeze test scoring or selection: ``ScoringLedger`` refuses a test scoring
before the freeze and a val scoring after it, and the scoring view (``Shards.open``) maps test.bin
only after the freeze. It is not a guarantee that no test.bin byte is read earlier: the manifest
verification of ``real_text_data.py`` hashes and memory-maps test.bin before the freeze, and
``splits_opened_before_freeze`` and the test-access log count the scoring interface only.

Freeze coverage. A ``cycle6_c6_01.v2`` run puts the AWQ choice of every (part, seed, arm, method) into
the freeze: the selected alpha, the val.bin scores of every alpha and the SHA-256 of the selected
effective weights. ``selections_sha256`` covers them, and the run appends that digest to the
test-access log before it maps test.bin. The preserved ``cycle6_c6_01.v1`` result (producer 1b4ea1c)
differs: its pre-test ``selections_sha256`` covered the family, arm and method names only, not the AWQ
alphas. The validator binds the v1 AWQ selections with ``HISTORICAL_V1_AWQ_SELECTIONS_SHA256``. That
is an anti-tamper pin computed after the run, not a contemporaneous freeze, and the validation receipt
says so (``awq_in_contemporaneous_freeze``: False for v1, True for v2).

Method menus. At 50m the plain Muon arm chooses its best method from all probe methods, including the
cross-mix butterfly composites (the v1 run selected ``cross_full_butterfly_b16_gptq``). The 50m QAT
arms are not offered those composites. A 50m best-vs-best contrast therefore compares deployment
pipelines; use the same-method contrasts for claims about the effect of training. At 120m the winning
method of the v1 run was offered to both the QAT arm and its token-matched control.

Only two modes exist:

* the canonical run (no flag): real FineWeb-Edu shards with pinned SHA-256 values, an sm_120+ device,
  a clean committed tree, and the canonical output path, which must not exist yet;
* ``--synthetic-shakedown``: generated Markov shards and a small model. It cannot read real text.

There is no real-data smoke mode, so no development run can score the real test.bin. A canonical run
appends one line to ``artifacts/c6_01_run_evidence/test_access_log.jsonl`` before it maps test.bin and
records the earlier lines in the result. A shakedown keeps the same record for its synthetic test.bin
in ``<output>.work/run_evidence``. ``verify_external_evidence`` checks a result against the retained
checkpoint files and that log.

Run (canonical, alone on the GPU, from a clean commit; about three hours on an RTX 5080):
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_c6_01_wide_rotated_qat.py
Shakedown (about two minutes):
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_c6_01_wide_rotated_qat.py \\
        --synthetic-shakedown --output /tmp/c6_01_shakedown.json
"""

from __future__ import annotations

import argparse
import ast
import copy
import fnmatch
import functools
import hashlib
import io
import json
import math
import os
import shutil
import statistics
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import timedelta
from importlib.abc import SourceLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import Tensor, nn

from research.polar_opt import harness_tier2_real as h
from research.polar_opt.harness_tier2_lm import Transformer
from research.quant_ptq import bench_c5_02_isoloss_ptq_mitigation as c502
from research.quant_ptq import bench_c5_03_nvfp4_120m_fusion as c503
from research.quant_ptq import bench_isoloss_ptq as c4
from research.quant_ptq import bench_nvfp4_e2e as e2e
from research.quant_ptq import micro_rotated_ptq as q
from research.quant_ptq import nvfp4_inference_blackwell as nb
from research.quant_ptq import rotated_qat as rq

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "cycle6_c6_01.v2"
# The preserved canonical result of producer 1b4ea1c. It stays valid, with every check that its records allow.
HISTORICAL_SCHEMA_V1 = "cycle6_c6_01.v1"
SCHEMAS = (HISTORICAL_SCHEMA_V1, SCHEMA_VERSION)
# SHA-256 of ``awq_freeze`` of the v1 result (alpha, val.bin scores and effective-weight digest of all 36 AWQ
# selections). The v1 freeze digest did not cover them; this pin was computed after the run.
HISTORICAL_V1_AWQ_SELECTIONS_SHA256 = "2b8a654c4a356c90e151f439ae3134051d2929c213cce8cce19d636338a43347"
TASK = ("Cycle 6 C6-01: wide Hadamard-rotated Muon W4A4 QAT and cross-block activation quantization mitigation "
        "(50m three-seed study, 120m real-checkpoint hardware NVFP4 transfer) on Blackwell")
DRIVER_MODULE = "research.quant_ptq.bench_c6_01_wide_rotated_qat"
SCRIPT_REPO_PATH = "/".join(DRIVER_MODULE.split(".")) + ".py"
RESULT_REPO_PATH = "results/cycle6_c6_01_wide_rotated_qat.json"
CANONICAL_OUTPUT = ROOT / RESULT_REPO_PATH
RESERVED_PATTERN = "cycle6_c6_01*.json"
CANONICAL_CHECKPOINT_DIR = ROOT / "artifacts" / "c6_01_checkpoints"
TEST_ACCESS_LOG_NAME = "test_access_log.jsonl"
TEST_ACCESS_LOG = ROOT / "artifacts" / "c6_01_run_evidence" / TEST_ACCESS_LOG_NAME
TEST_ACCESS_LOG_REPO_PATH = "artifacts/c6_01_run_evidence/test_access_log.jsonl"
SHAKEDOWN_TEST_ACCESS_LOG = "<output>.work/run_evidence/test_access_log.jsonl"  # the synthetic test.bin only
EXPECTED_ROLES = {"training": "train", "calibration": "train", "static_scales": "train",
                  "awq_alpha_selection": "val", "method_and_arm_selection": "val", "evaluation": "test"}
C5_03_RESULT_REPO_PATH = c503.RESULT_REPO_PATH
C5_03_RESULT_SHA256 = "e0e435ef2ea8598df22bace95ab76fd10c43b2ff394bf635ab8d1548d8bcad82"
# The modules that measure, analyze, summarize and validate. ``meta.git_sha`` must hold these bytes.
SUMMARY_MODULES = (
    DRIVER_MODULE,
    "research.quant_ptq.rotated_qat",
    "research.quant_ptq.bench_c5_03_nvfp4_120m_fusion",
    "research.quant_ptq.bench_c5_02_isoloss_ptq_mitigation",
    "research.quant_ptq.bench_isoloss_ptq",
    "research.quant_ptq.bench_nvfp4_e2e",
    "research.quant_ptq.micro_rotated_ptq",
    "research.quant_ptq.nvfp4_inference_blackwell",
    "research.polar_opt.harness_tier2_real",
)
TOL = 1e-12
N_SEEDS = 3
SIGN_FLIP_MIN_P = 2 / 2 ** N_SEEDS
SPLITS = ("val", "test")
ENDPOINTS = ("nll", "delta_nll")

# The preserved C5-03 baselines (120m, muon_ps3_blackwell, 640 steps): file and tensor digests of the
# committed C5-03 result (``part_c.seeds[i].training``).
PINNED_BASELINES = {
    1: {"file": "seed1_120m_muon_ps3_blackwell_640steps.pt",
        "checkpoint_sha256": "ad9dc6a00b80e5b82b14eb01927ea0cd7b1c64825adcf7e3c4e016f566d8c7f9",
        "checkpoint_file_sha256": "448b0297dfd632839fcdcae4fd530fdc070d785da5a5bf35d7e886377a5ab29b"},
    2: {"file": "seed2_120m_muon_ps3_blackwell_640steps.pt",
        "checkpoint_sha256": "82158f688be42b5acb46d9a33997c06317943da24db7ee2c11f796f935e338c4",
        "checkpoint_file_sha256": "0b02ffee244aaa6f811740759c166df20a4f3541ef60f0b7f223a98567aee3ef"},
    3: {"file": "seed3_120m_muon_ps3_blackwell_640steps.pt",
        "checkpoint_sha256": "d705cbc5ea2a95296ecd5a899ef76b9ed06cff8a1bd04a75a20c5e354c9d1b38",
        "checkpoint_file_sha256": "057e78c783ff966cc368b1292cd4514ab6f2f9d06032dc9334f77e72b6154856"},
}

PRESETS = {
    # Widths 64 (one Hadamard block) and 192 (= 3 x 64: the DCT-IV factor), so the shakedown runs every rotation path.
    "shakedown": h.ModelConfig(vocab_size=512, width=64, layers=2, heads=4, hidden=192, sequence_length=64,
                               micro_batch=8, grad_accum=2),
    "50m": h.PRESETS["50m"],
    "120m": h.PRESETS["120m"],
}

MODES = ("w4a16_fakequant", "w4a4_hw_dynamic", "w4a4_hw_static")
HW_MODES = MODES[1:]
DYNAMIC = "w4a4_hw_dynamic"
EMULATED_MODE = "w4a4_emulated_fp32_dynamic"
BF16 = "@bf16"  # the unquantized checkpoint, in a contrast
STATIC_CALIBRATION = "train.bin calibration windows through the BF16 model, max |map(x)| / 2688"

# Development record (synthetic Markov shards only; scratch runs, never part of a result, no real split read).
# Model: width 128, 4 layers, hidden 384, 128-token windows; Muon 300 steps, cooldown from step 255; two seeds;
# val NLL means, BF16 / best hardware W4A4 dynamic (Muon: 1.7589 / 1.7654):
#   identity QAT cooldown          1.7595 / 1.7646      identity QAT full run      1.7636 / 1.7684
#   block-16 QAT cooldown          1.7601 / 1.7654      block-64 QAT cooldown      1.7595 / 1.7655
#   full-width QAT cooldown        1.7593 / 1.7654      full-width QAT full run    1.7631 / 1.7689
#   full-width dual loss cooldown, BF16 weight 0.25 / 0.5 / 0.75: 1.7591 / 1.7654, 1.7589 / 1.7654, 1.7589 / 1.7659
# Decisions taken from it: (1) the dual loss uses BF16 weight 0.5 (BF16 NLL unchanged, W4A4 not worse);
# (2) a rotated full-run arm is not part of the canonical study: both full runs lost about 0.004 nats of BF16 and
# W4A4 NLL to their cooldowns, so only the identity full run stays, as the reference the task asks for;
# (3) a logit-distillation term was dropped: the full-vocabulary KL of a 50m micro-batch does not fit in 16 GB.
# The synthetic differences between the cooldown arms are within seed noise (about 0.0005 nats); they did not
# choose between rotation widths. That choice is made on val.bin inside the canonical run.
# A separate timing probe trained random-init 50m / 120m models on train.bin for a few steps (no split scored):
# plain 272 / 440 ms per step, identity QAT 470-570 / 726, full-width QAT 545-670 / 844, dual 810-930 / 1260.

# Part A / B arms (50m): arm -> None (plain Muon) or (QAT family or "@wide", schedule, loss)
MUON = "muon_ss3_54"
ARMS_AB = {
    MUON: None,
    "qat_identity_cooldown": ("identity", "cooldown", "qat"),
    "qat_identity_full": ("identity", "full", "qat"),
    "qat_b16_cooldown": ("hadamard_b16", "cooldown", "qat"),
    "qat_wide_cooldown": ("@wide", "cooldown", "qat"),
    "qat_wide_dual_cooldown": ("@wide", "cooldown", "dual"),
}
QAT_ARMS = tuple(a for a in ARMS_AB if a != MUON)
COOLDOWN_ARMS = tuple(a for a, spec in ARMS_AB.items() if spec is not None and spec[1] == "cooldown")
TRANSFER_CANDIDATES = ("qat_b16_cooldown", "qat_wide_cooldown", "qat_wide_dual_cooldown")  # rotated cooldowns
WIDE_QAT_FAMILIES = ("hadamard_b64", "kron_hadamard_full")
# Part C arms (120m)
BASELINE, CONTINUE, QAT_C = "c5_03_baseline", "bf16_continue", "qat_cooldown"
ARMS_C = (BASELINE, CONTINUE, QAT_C)
PARTS = ("ab", "c")

C5_HADAMARD = "hadamard_b16_gptq"
C5_BUTTERFLY = "adaptive_butterfly_b16_gptq"
C5_BEST = "awq_w4a4_adaptive_butterfly_b16_gptq"  # the val-selected C5-03 method
PROBE_FAMILIES = tuple(rq.FAMILIES)
PROBE_METHODS = (
    "rtn", "gptq", "hadamard_b16_rtn", C5_HADAMARD, C5_BUTTERFLY, "hadamard_b64_rtn", "hadamard_b64_gptq",
    "kron_hadamard_full_rtn", "kron_hadamard_full_gptq", "cross_b64_butterfly_b16_gptq",
    "cross_full_butterfly_b16_gptq", "awq_w4a4_hadamard_b16_gptq", C5_BEST, "awq_w4a4_hadamard_b64_gptq",
    "awq_w4a4_kron_hadamard_full_gptq")
PROBE_EMULATED = "kron_hadamard_full_gptq"

CANONICAL = {
    "device": "cuda", "seeds": [1, 2, 3],
    "weight_decay": c4.WEIGHT_DECAY, "min_lr_ratio": c4.MIN_LR_RATIO, "gptq_damp": c4.GPTQ_DAMP,
    "calibration_sequences": 128, "calibration_seed": 20261126, "butterfly_steps": 150,
    "awq_alphas": [0.0, 0.25, 0.5], "emulation_parity_nats": 1e-3, "max_eval_windows": None,
    "dual_bf16_weight": 0.5,
    "part_ab": {"preset": "50m", "optimizer": MUON, "steps": 320, "cooldown_from": 272, "lr": 0.02,
                "auxiliary_lr": 0.001},
    "part_c": {"preset": "120m", "optimizer": "muon_ps3_blackwell", "baseline": "c5_03_preserved",
               "baseline_steps": 640, "baseline_warmup_steps": 64, "cooldown_steps": 96, "lr": 0.02,
               "auxiliary_lr": 0.001},
}
SHAKEDOWN = {
    **CANONICAL, "calibration_sequences": 16, "butterfly_steps": 5, "max_eval_windows": 96,
    "part_ab": {**CANONICAL["part_ab"], "preset": "shakedown", "steps": 40, "cooldown_from": 30},
    "part_c": {**CANONICAL["part_c"], "preset": "shakedown", "baseline": "trained_in_run", "baseline_steps": 30,
               "baseline_warmup_steps": 6, "cooldown_steps": 8},
}
SYNTHETIC_SHARDS = {"seed": 0, "vocab": 512, "branching": 6, "document_tokens": 400,
                    "documents": {"train": 1200, "val": 120, "test": 120}}


# ======================================================================================
# Method grids
# ======================================================================================
def _dedupe(items) -> list:
    return list(dict.fromkeys(items))


def arm_qat(part: str, arm: str, selections: dict) -> dict | None:
    """``{"family", "schedule", "loss"}`` of a QAT arm (None for a BF16-trained arm)."""
    if part == "ab":
        spec = ARMS_AB[arm]
    else:
        spec = None if arm != QAT_C else (selections["transfer_family"], "cooldown", selections["transfer_loss"])
    if spec is None:
        return None
    family = selections["qat_wide_family"] if spec[0] == "@wide" else spec[0]
    return {"family": family, "schedule": spec[1], "loss": spec[2]}


def unit_methods(part: str, arm: str, selections: dict) -> list[str]:
    """The PTQ methods of one checkpoint, in grid order. AWQ + butterfly needs no plain twin in the grid."""
    qat = arm_qat(part, arm, selections)
    if part == "ab" and arm == MUON:
        return list(PROBE_METHODS)
    if qat is not None:
        own = [rq.method_name(qat["family"], "rtn"), rq.method_name(qat["family"], "gptq")]
        tail = [] if part == "ab" else [selections["best_wide_ptq_method"]]
        return _dedupe([*own, C5_HADAMARD, C5_BEST, *tail])
    transfer = rq.method_name(selections["transfer_family"], "gptq")
    return _dedupe([C5_HADAMARD, C5_BEST, rq.method_name(selections["qat_wide_family"], "gptq"),
                    selections["best_wide_ptq_method"], transfer])


def emulated_method(part: str, arm: str, selections: dict) -> str:
    """The method whose hardware rows get an FP32 emulation twin: the widest path of the checkpoint."""
    qat = arm_qat(part, arm, selections)
    if qat is not None:
        return rq.method_name(qat["family"], "gptq")
    return PROBE_EMULATED if part == "ab" else rq.method_name(selections["qat_wide_family"], "gptq")


def expected_grid(methods: list[str], emulated: str) -> list[tuple[str, str]]:
    out = []
    for method in methods:
        out += [(method, mode) for mode in MODES]
        if method == emulated:
            out.append((method, EMULATED_MODE))
    return out


# ======================================================================================
# Selections on val.bin (pure functions of the stored rows; the validator recomputes them)
# ======================================================================================
def _units(units: list[dict], part: str, arm: str) -> list[dict]:
    found = sorted((u for u in units if u["part"] == part and u["arm"] == arm), key=lambda u: u["seed"])
    if len(found) != N_SEEDS:
        raise ValueError(f"part {part} arm {arm}: expected {N_SEEDS} seeds, found {len(found)}")
    return found


def _row(unit: dict, method: str, mode: str) -> dict:
    return next(r for r in unit["grid"] if r["method"] == method and r["mode"] == mode)


def summed_nll(units: list[dict], part: str, arm: str, method: str, split: str = "val", mode: str = DYNAMIC) -> float:
    return math.fsum(_row(u, method, mode)[f"{split}_nll"] for u in _units(units, part, arm))


def best_method(units: list[dict], part: str, arm: str, methods, split: str = "val") -> str:
    """The method with the lowest hardware dynamic NLL summed over seeds (ties: list order)."""
    methods = list(methods)
    return min(methods, key=lambda m: (summed_nll(units, part, arm, m, split), methods.index(m)))


def select_probe(units: list[dict]) -> dict:
    """Part A selections on val.bin from the plain Muon checkpoints."""
    wide = best_method(units, "ab", MUON, [rq.method_name(f, "gptq") for f in WIDE_QAT_FAMILIES])
    return {"qat_wide_family": rq.parse_method(wide)[0],
            "best_wide_ptq_method": best_method(units, "ab", MUON, [m for m in PROBE_METHODS if rq.is_wide(m)]),
            "best_ptq_method": best_method(units, "ab", MUON, PROBE_METHODS)}


def select_transfer(units: list[dict], selections: dict) -> dict:
    """Part B selections on val.bin: the best method of every arm and the rotated cooldown that Part C runs."""
    best = {arm: best_method(units, "ab", arm, unit_methods("ab", arm, selections)) for arm in ARMS_AB}
    arm = min(TRANSFER_CANDIDATES, key=lambda a: (summed_nll(units, "ab", a, best[a]), TRANSFER_CANDIDATES.index(a)))
    qat = arm_qat("ab", arm, selections)
    return {"arm_best_method": {"ab": best}, "transfer_arm": arm, "transfer_family": qat["family"],
            "transfer_loss": qat["loss"]}


def select_part_c(units: list[dict], selections: dict) -> dict:
    return {arm: best_method(units, "c", arm, unit_methods("c", arm, selections)) for arm in ARMS_C}


def selections_sha256(selections: dict) -> str:
    return hashlib.sha256(json.dumps(selections, sort_keys=True).encode()).hexdigest()


def awq_freeze(units: list[dict]) -> list[dict]:
    """The AWQ choice of every (part, seed, arm, method), from the val.bin records alone.

    A v2 run stores this list in ``selections["awq_selections"]`` before it maps test.bin, so
    ``selections_sha256`` covers it.
    """
    return [{"part": u["part"], "seed": u["seed"], "arm": u["arm"], "method": a["method"],
             "selected_alpha": a["selected_alpha"], "val_nll_hw_dynamic": list(a["val_nll_hw_dynamic"]),
             "effective_weight_sha256": _row(u, a["method"], "w4a16_fakequant")["effective_weights_sha256"]}
            for u in units for a in u["awq_selection"]]


# ======================================================================================
# Training
# ======================================================================================
def _new_model(config: h.ModelConfig, device, state: dict) -> Transformer:
    with torch.device(device):
        model = Transformer(config)
    model.load_state_dict(state)
    return model.train()


def run_steps(model, optimizer, stream: h.BatchStream, first: int, last: int, multiplier, label: str, *,
              wrappers: dict | None = None, bf16_weight: float | None = None, at_step=None):
    """Optimizer steps ``first .. last - 1``. Returns (losses, wall seconds per step; ``.item()`` synchronizes)."""
    losses, seconds = [], []
    for step in range(first, last):
        if at_step is not None:
            at_step(step)
        optimizer.set_lr_multiplier(multiplier(step))
        began = time.perf_counter()
        losses.append(rq.train_step(model, stream.next_batch(), optimizer, wrappers, bf16_weight).item())
        model.zero_grad(set_to_none=True)
        seconds.append(time.perf_counter() - began)
        if not math.isfinite(losses[-1]):
            raise RuntimeError(f"{label} diverged at step {step + 1}")
    return losses, seconds


def run_arm_steps(model, optimizer, stream, first: int, last: int, multiplier, label: str, qat: dict | None,
                  bf16_weight: float, at_step=None):
    if qat is None:
        return run_steps(model, optimizer, stream, first, last, multiplier, label, at_step=at_step)
    with rq.fake_quant(model, qat["family"]) as wrappers:
        return run_steps(model, optimizer, stream, first, last, multiplier, label, wrappers=wrappers,
                         bf16_weight=bf16_weight if qat["loss"] == "dual" else None)


def training_record(part: str, arm: str, seed: int, config: h.ModelConfig, train: np.ndarray, model, path: Path, *,
                    optimizer: str, lr: float, auxiliary_lr: float, steps: int, first_step: int, losses: list[float],
                    step_seconds: list[float], qat: dict | None, branch: dict | None, init_fingerprint: str | None,
                    protocol: dict) -> dict:
    """Record of a checkpoint after ``steps`` optimizer steps; ``losses`` are the steps ``first_step .. steps - 1``."""
    c502._save_state(model.state_dict(), path)
    tokens = steps * config.step_tokens
    return {"source": "trained_in_run", "part": part, "arm": arm, "seed": seed, "optimizer": optimizer, "lr": lr,
            "auxiliary_lr": auxiliary_lr, "weight_decay": protocol["weight_decay"],
            "min_lr_ratio": protocol["min_lr_ratio"], "steps": steps, "first_step": first_step,
            "step_tokens": config.step_tokens, "tokens": tokens, "train_epochs": tokens / len(train),
            "init_fingerprint": init_fingerprint, "stream_fingerprint": h.stream_fingerprint(train, config, seed),
            "train_loss": math.fsum(losses[-10:]) / len(losses[-10:]), "last_train_loss": losses[-1],
            "train_curve": [[first_step + i + 1, v] for i, v in enumerate(losses)
                            if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(losses)],
            "train_seconds": math.fsum(step_seconds), "step_seconds": list(step_seconds),
            "qat": None if qat is None else {**qat, "bf16_weight": protocol["dual_bf16_weight"]
                                             if qat["loss"] == "dual" else None,
                                             "fake_quant_steps": steps - first_step},
            "branch": branch, "checkpoint_file": path.name, "checkpoint_sha256": e2e._checkpoint_sha256(model.eval()),
            "checkpoint_file_sha256": c503.file_sha256(path)}


def checkpoint_name(part: str, seed: int, arm: str) -> str:
    return f"{part}_seed{seed}_{arm}.pt"


def _muon_protocol(spec: dict) -> dict:
    return {"muon": {"lr": spec["lr"], "auxiliary_lr": spec["auxiliary_lr"]}}


def _cpu_snapshot(snapshot: dict) -> dict:
    """A ``c502._snapshot`` moved to the CPU (weights, Muon momentum, auxiliary AdamW state)."""
    def cpu(obj):
        if isinstance(obj, Tensor):
            return obj.cpu()
        if isinstance(obj, dict):
            return {k: cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(cpu(v) for v in obj)
        return obj
    return cpu(snapshot)


def optimizer_snapshot_sha256(snapshot: dict) -> str:
    """SHA-256 of the optimizer state of a ``c502._snapshot``: Muon momentum and the auxiliary AdamW state dict.

    Keys in sorted order; every tensor with its dtype, shape and bytes; every other value by ``repr``.
    """
    digest = hashlib.sha256()

    def feed(obj) -> None:
        if isinstance(obj, Tensor):
            t = obj.detach().cpu().contiguous()
            digest.update(f"tensor:{t.dtype}:{tuple(t.shape)}\0".encode())
            digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, dict):
            digest.update(f"dict:{len(obj)}\0".encode())
            for key in sorted(obj, key=str):
                digest.update(f"{key!r}\0".encode())
                feed(obj[key])
        elif isinstance(obj, (list, tuple)):
            digest.update(f"list:{len(obj)}\0".encode())
            for item in obj:
                feed(item)
        else:
            digest.update(f"{type(obj).__name__}:{obj!r}\0".encode())

    feed({"momentum": snapshot["momentum"], "auxiliary": snapshot["auxiliary"]})
    return digest.hexdigest()


def train_ab_arm(arm: str, qat: dict | None, config, train, device, protocol: dict, seed: int, init: dict,
                 snapshot: dict | None, path: Path):
    """One 50m arm. Cooldown arms branch from ``snapshot`` and run the same last batches as the Muon run.

    A cooldown records the digests of the state it starts from, measured after the restore: the weights
    (``snapshot_sha256``) and the optimizer (``optimizer_snapshot_sha256``). Both must equal the snapshot.
    """
    spec = protocol["part_ab"]
    total, first = spec["steps"], 0
    multiplier = lambda step: c502.schedule_multiplier(step, total)  # noqa: E731
    branch, snapshots = None, {}
    if qat is not None and qat["schedule"] == "cooldown":
        first = spec["cooldown_from"]
        model, optimizer = c502._restore(config, device, _muon_protocol(spec), snapshot)
        stream = c502._stream_at(train, config, seed, device, first)
        branch = {"from_arm": MUON, "from_step": first, "snapshot_sha256": e2e._checkpoint_sha256(model),
                  "optimizer_snapshot_sha256": optimizer_snapshot_sha256(
                      {"momentum": optimizer.momentum, "auxiliary": optimizer.auxiliary.state_dict()})}
        if (branch["snapshot_sha256"], branch["optimizer_snapshot_sha256"]) != \
                (snapshot["sha256"], snapshot["optimizer_sha256"]):
            raise RuntimeError(f"{arm} seed {seed}: the restored state differs from the step-{first} snapshot")
    else:
        model = _new_model(config, device, init)
        optimizer, _ = c502._muon_optimizer(model, _muon_protocol(spec), device)
        stream = h.BatchStream(train, config, seed, device)

    def at_step(step: int) -> None:
        if qat is None and step == spec["cooldown_from"]:
            snapshots[step] = _cpu_snapshot(c502._snapshot(model, optimizer))
            snapshots[step]["optimizer_sha256"] = optimizer_snapshot_sha256(snapshots[step])

    losses, seconds = run_arm_steps(model, optimizer, stream, first, total, multiplier, f"{arm} seed {seed}", qat,
                                    protocol["dual_bf16_weight"], at_step)
    record = training_record("ab", arm, seed, config, train, model, path, optimizer=spec["optimizer"], lr=spec["lr"],
                             auxiliary_lr=spec["auxiliary_lr"], steps=total, first_step=first, losses=losses,
                             step_seconds=seconds, qat=qat, branch=branch,
                             init_fingerprint=h.state_fingerprint(init), protocol=protocol)
    return model, record, snapshots.get(spec["cooldown_from"])


def _ps3_optimizer(model, spec: dict, protocol: dict):
    factory = h.builtin_factories()[spec["optimizer"]]
    return h.build_optimizer(spec["optimizer"], factory, model, lr=spec["lr"], auxiliary_lr=spec["auxiliary_lr"],
                             weight_decay=protocol["weight_decay"], update_dtype="fp32")


def train_shakedown_baseline(config, train, device, protocol: dict, seed: int, path: Path):
    """The shakedown stand-in for a preserved C5-03 checkpoint: the C5-03 recipe on the synthetic shards."""
    spec = protocol["part_c"]
    steps = spec["baseline_steps"]
    init = h.initial_state(config, seed)
    model = _new_model(config, device, init)
    optimizer = _ps3_optimizer(model, spec, protocol)
    multiplier = lambda step: h.lr_multiplier(step, step / steps, spec["baseline_warmup_steps"],  # noqa: E731
                                              protocol["min_lr_ratio"], total_steps=steps)
    losses, seconds = run_steps(model, optimizer, h.BatchStream(train, config, seed, device), 0, steps, multiplier,
                                f"baseline seed {seed}")
    return training_record("c", BASELINE, seed, config, train, model, path, optimizer=spec["optimizer"], lr=spec["lr"],
                           auxiliary_lr=spec["auxiliary_lr"], steps=steps, first_step=0, losses=losses,
                           step_seconds=seconds, qat=None, branch=None, init_fingerprint=h.state_fingerprint(init),
                           protocol=protocol)


def train_c_arm(arm: str, qat: dict | None, config, train, device, protocol: dict, seed: int, baseline: dict,
                baseline_path: Path, path: Path):
    """``cooldown_steps`` extra steps from a baseline checkpoint at the terminal LR of its schedule.

    The C5-03 checkpoints hold weights only, so Muon momentum and the auxiliary AdamW start fresh. The
    batches are the steps ``baseline_steps ..`` of the seed's stream: new tokens for both arms.
    """
    spec = protocol["part_c"]
    first, total = spec["baseline_steps"], spec["baseline_steps"] + spec["cooldown_steps"]
    state = torch.load(baseline_path, map_location="cpu", weights_only=True)
    model = _new_model(config, device, state)
    if e2e._checkpoint_sha256(model) != baseline["checkpoint_sha256"]:
        raise RuntimeError(f"{baseline_path.name}: tensor SHA-256 differs from the baseline record")
    optimizer = _ps3_optimizer(model, spec, protocol)
    stream = c502._stream_at(train, config, seed, device, first)
    losses, seconds = run_arm_steps(model, optimizer, stream, first, total, lambda step: protocol["min_lr_ratio"],
                                    f"{arm} seed {seed}", qat, protocol["dual_bf16_weight"])
    # No optimizer snapshot exists: both continuations start a fresh optimizer from the same weights.
    branch = {"from_arm": BASELINE, "from_step": first, "snapshot_sha256": baseline["checkpoint_sha256"],
              "optimizer_snapshot_sha256": None}
    return model, training_record("c", arm, seed, config, train, model, path, optimizer=spec["optimizer"],
                                  lr=spec["lr"], auxiliary_lr=spec["auxiliary_lr"], steps=total, first_step=first,
                                  losses=losses, step_seconds=seconds, qat=qat, branch=branch, init_fingerprint=None,
                                  protocol=protocol)


def preserved_baseline_record(seed: int) -> dict:
    """Training record of a preserved C5-03 checkpoint (not trained in this run)."""
    pin = PINNED_BASELINES[seed]
    return {"source": "c5_03_preserved", "part": "c", "arm": BASELINE, "seed": seed,
            "checkpoint_file": pin["file"], "checkpoint_sha256": pin["checkpoint_sha256"],
            "checkpoint_file_sha256": pin["checkpoint_file_sha256"]}


# ======================================================================================
# Checkpoint study: PTQ candidates, hardware W4A4 scoring, val then test
# ======================================================================================
def _hw_counts(model) -> dict:
    """What runs in the block linear slots (the AWQ and cross-mix wrappers are looked through)."""
    counts = {"nvfp4_modules": 0, "bf16_block_linears": 0, "emulated_modules": 0, "cross_mix_modules": 0}
    for block in model.blocks:
        for kind in q.LINEAR_KINDS:
            m = getattr(block, kind)
            if isinstance(m, c502.ScaledInput):
                m = m.layer
            if isinstance(m, rq.CrossInput):
                counts["cross_mix_modules"] += 1
                m = m.layer
            counts["nvfp4_modules"] += isinstance(m, nb.NVFP4Linear)
            counts["bf16_block_linears"] += type(m) is nn.Linear
            counts["emulated_modules"] += isinstance(m, e2e.EmulatedW4A4Linear)
    return counts


ROW_BASE = {"method": None, "mode": None, "rotation": None, "rounding": None, "awq": None, "awq_alpha": None,
            "wide": None, "fp4_gemms_per_forward": None, "nvfp4_modules": None, "bf16_block_linears": None,
            "emulated_modules": None, "cross_mix_modules": None, "cupti": None, "hw_weights_checked_matrices": None,
            "hw_weights_decode_exact_matrices": None, "effective_weights_sha256": None,
            "effective_weights_sha256_test": None, "quantize_seconds": None, "gptq_damping_max": None,
            "static_global_scale_range": None}
ROW_KEYS = set(ROW_BASE) | {f"{s}_{k}" for s in SPLITS for k in (
    "nll", "ppl", "tokens", "delta_nll", "relative_delta_ppl_pct")} | {f"seconds_{s}" for s in SPLITS}


def new_row(method: str, mode: str, **extra) -> dict:
    family, rounding, awq = rq.parse_method(method)
    return {**ROW_BASE, "method": method, "mode": mode, "rotation": family, "rounding": rounding, "awq": awq,
            "wide": family in rq.WIDE_FAMILIES, **extra}


class Unit:
    """One checkpoint (part, seed, arm): its training record, PTQ candidates and grid rows."""

    def __init__(self, part: str, seed: int, arm: str, config: h.ModelConfig, selections: dict, training: dict,
                 checkpoint_dir: Path):
        self.part, self.seed, self.arm, self.config = part, seed, arm, config
        self.methods = unit_methods(part, arm, selections)
        self.emulated = emulated_method(part, arm, selections)
        self.probe = part == "ab" and arm == MUON
        self.training, self.checkpoint_dir = training, Path(checkpoint_dir)
        self.rows: dict[tuple[str, str], dict] = {}
        self.cands: dict[str, dict] = {}
        self.awq_selection, self.butterfly_fits, self.baseline = [], {}, {}
        self.geometry = self.activation_statistics = self.bf16_fp4_gemms = self.bf16_cupti = None
        self.seconds = {"phase1": 0.0, "phase2": 0.0}

    @property
    def label(self) -> str:
        return f"{self.part.upper()} seed {self.seed} {self.arm}"

    def record_phase1(self) -> dict:
        """The val.bin view of the unit, for the selection rules (no test field exists yet)."""
        return {"part": self.part, "seed": self.seed, "arm": self.arm, "grid": list(self.rows.values()),
                "awq_selection": list(self.awq_selection)}

    def record(self) -> dict:
        return {"part": self.part, "seed": self.seed, "arm": self.arm, "training": self.training,
                "methods": list(self.methods), "emulated_method": self.emulated,
                "baseline_val": self.baseline["val"], "baseline_test": self.baseline["test"],
                "bf16_fp4_gemms_per_forward": self.bf16_fp4_gemms, "bf16_cupti": self.bf16_cupti,
                "geometry": self.geometry, "activation_statistics": self.activation_statistics,
                "butterfly_fits": self.butterfly_fits, "awq_selection": self.awq_selection,
                "grid": [self.rows[key] for key in expected_grid(self.methods, self.emulated)],
                "seconds": dict(self.seconds)}


def _fill(row: dict, split: str, result: dict, base: dict, seconds: float) -> None:
    if result["tokens"] != base["tokens"]:
        raise RuntimeError("Evaluation token count changed between runs")
    row.update({f"{split}_nll": result["nll"], f"{split}_ppl": result["ppl"], f"{split}_tokens": result["tokens"],
                f"{split}_delta_nll": result["nll"] - base["nll"],
                f"{split}_relative_delta_ppl_pct": 100.0 * (result["ppl"] / base["ppl"] - 1.0),
                f"seconds_{split}": seconds})


def _layer_summary(values: dict[str, Tensor]) -> dict:
    """Mean / median / min / max over all layers and the mean per projection kind."""
    flat = [v for kind in q.LINEAR_KINDS for v in values[kind].tolist()]
    return {"all": q._summary(flat), "by_kind": {kind: float(values[kind].double().mean()) for kind in q.LINEAR_KINDS}}


class Study:
    """Phase 1 (train.bin + val.bin) of every checkpoint, one freeze, then Phase 2 (test.bin)."""

    def __init__(self, protocol: dict, shards, device, log):
        self.protocol, self.shards, self.device, self.log = protocol, shards, device, log
        self.ledger = c503.ScoringLedger()
        self.view = shards.open("train", "val")
        self.calibration = {}
        self.units: list[Unit] = []

    # ---------------------------------------------------------------- helpers
    def calibrate_for(self, config: h.ModelConfig) -> dict:
        """One seeded train.bin draw per sequence length (the C5-03 draw for 512-token windows)."""
        key = config.sequence_length
        if key not in self.calibration:
            self.calibration[key] = q.calibration_windows(
                SimpleNamespace(splits={"train": self.view["train"]}), self.protocol["calibration_sequences"], key,
                self.protocol["calibration_seed"])
        return self.calibration[key]

    def kinds(self, config) -> dict[str, list[str]]:
        return {k: [f"blocks.{i}.{k}" for i in range(config.layers)] for k in q.LINEAR_KINDS}

    def score(self, unit: Unit, model, split: str) -> dict:
        return self.ledger.score(model, split, self.view[split], unit.config, self.device,
                                 self.protocol["max_eval_windows"])

    def apply(self, unit: Unit, model, stacks: dict[str, Tensor]) -> None:
        linears = q.quantized_linears(model)
        with torch.no_grad():
            for kind, names in self.kinds(unit.config).items():
                for i, name in enumerate(names):
                    linears[name].weight.copy_(stacks[kind][i])

    def candidate(self, unit: Unit, W, Hc, A, family: str, rounding: str, alpha: float) -> dict:
        """Quantized stacks of every kind in (scaled and) rotated coordinates, and their NVFP4Linear layers."""
        began = time.perf_counter()
        cand = {"family": family, "rounding": rounding, "alpha": alpha, "rotations": {}, "scales": {},
                "inverse_scales": {}, "effective": {}, "fits": {}, "layers": {}}
        damping, checked, exact = [], 0, 0
        for kind, names in self.kinds(unit.config).items():
            scale = c502.awq_scales(A[kind], alpha) if alpha else None
            Ws = W[kind] if scale is None else W[kind] * scale[:, None, :]
            rot, fit = rq.make_rotation(family, Ws, butterfly_steps=self.protocol["butterfly_steps"])
            if fit:
                cand["fits"][kind] = fit
            Hr = None
            if rounding == "gptq":
                Hr = rot.rotate_hessian(Hc[kind] if scale is None else c502.scale_hessian(Hc[kind], scale))
            Qr, enc, lam = c4._quantize(rot.forward(Ws), Hr, rq.FMT, rounding)
            eff = rot.inverse(Qr)
            cand["effective"][kind] = eff if scale is None else eff / scale[:, None, :]
            cand["rotations"][kind] = rot
            cand["scales"][kind] = scale
            cand["inverse_scales"][kind] = None if scale is None else 1.0 / scale
            damping += lam
            for i, name in enumerate(names):
                layer = nb.NVFP4Linear.from_encoding(
                    enc["codes"][i], enc["block_scale"][i], enc["global_scale"][i],
                    rotation=rq.kernel_rotation(family), angles=rot.angles[i] if hasattr(rot, "angles") else None)
                checked += 1
                exact += bool(torch.equal(nb.dequantized_weight(layer).view(torch.int32),
                                          Qr[i].contiguous().view(torch.int32)))
                cand["layers"][name] = layer
            del Qr, enc
        cand["effective_sha256"] = c503._stack_sha256(cand["effective"])
        cand["info"] = {"quantize_seconds": time.perf_counter() - began,
                        "gptq_damping_max": max(damping) if damping else None,
                        "hw_weights_checked_matrices": checked, "hw_weights_decode_exact_matrices": exact,
                        "effective_weights_sha256": cand["effective_sha256"]}
        return cand

    def modules(self, unit: Unit, cand: dict, emulate: bool = False) -> dict:
        out = {}
        for kind, names in self.kinds(unit.config).items():
            inv, cross = cand["inverse_scales"][kind], rq.cross_of(cand["rotations"][kind])
            for i, name in enumerate(names):
                m = e2e.EmulatedW4A4Linear(cand["layers"][name]) if emulate else cand["layers"][name]
                if cross is not None:
                    m = rq.CrossInput(m, cross)
                out[name] = m if inv is None else c502.ScaledInput(m, inv[i])
        return out

    def maps(self, unit: Unit, cand: dict) -> dict:
        """Input map of every layer: ``x -> (x / s) R^T``."""
        out = {}
        for kind, names in self.kinds(unit.config).items():
            rot, inv = cand["rotations"][kind], cand["inverse_scales"][kind]
            for i, name in enumerate(names):
                layer_rot = rq.layer_rotation(rot, i)
                out[name] = layer_rot if inv is None else c502.ScaledRotation(layer_rot, inv[i])
        return out

    @staticmethod
    def set_static(cand: dict, scales: dict | None) -> None:
        for name, layer in cand["layers"].items():
            layer.act_global_scale = None if scales is None else scales[name].to(layer.w_packed_bytes.device)

    def effective_again(self, unit: Unit, cand: dict) -> dict:
        """W4A16 weights rebuilt from the retained NVFP4 layers (decode, inverse rotation, AWQ scale)."""
        out = {}
        for kind, names in self.kinds(unit.config).items():
            Qr = torch.stack([nb.dequantized_weight(cand["layers"][n]) for n in names])
            eff = cand["rotations"][kind].inverse(Qr)
            scale = cand["scales"][kind]
            out[kind] = eff if scale is None else eff / scale[:, None, :]
        cand["effective_sha256_test"] = c503._stack_sha256(out)
        return out

    def measure(self, unit: Unit, model, cand: dict, mode: str, split: str, static: dict | None = None,
                evidence: bool = False):
        """(evaluation, hardware counts) of one candidate in one mode on one split. Restores the model.

        The counts are the modules in the block linear slots. ``evidence`` adds one profiled forward:
        the ``aten::_scaled_mm_v2`` dispatches and the CUPTI kernel launches (every grid row, on val.bin).
        """
        if mode == "w4a16_fakequant":
            self.apply(unit, model, cand["effective"] if split == "val" else self.effective_again(unit, cand))
            try:
                return self.score(unit, model, split), {}
            finally:
                self.apply(unit, model, self.W)
        self.set_static(cand, static)
        previous = e2e._swap(model, self.modules(unit, cand, emulate=mode == EMULATED_MODE))
        try:
            counts = _hw_counts(model)
            if evidence:
                profile = rq.forward_profile(model, self.probe)
                counts.update(fp4_gemms_per_forward=profile["scaled_mm_dispatches"], cupti=profile["cupti"])
            return self.score(unit, model, split), counts
        finally:
            e2e._swap(model, previous)
            self.set_static(cand, None)

    def evidence(self, unit: Unit, model, cand: dict, mode: str, static: dict | None = None) -> dict:
        """The profiled forward of a candidate that was scored without it (the selected AWQ candidate)."""
        self.set_static(cand, static)
        previous = e2e._swap(model, self.modules(unit, cand, emulate=mode == EMULATED_MODE))
        try:
            profile = rq.forward_profile(model, self.probe)
            return {"fp4_gemms_per_forward": profile["scaled_mm_dispatches"], "cupti": profile["cupti"]}
        finally:
            e2e._swap(model, previous)
            self.set_static(cand, None)

    def to(self, cand: dict, device) -> None:
        for name, layer in cand["layers"].items():
            cand["layers"][name] = layer.to(device)
        cand["rotations"] = {k: rq.rotation_to(rot, device) for k, rot in cand["rotations"].items()}
        for key in ("scales", "inverse_scales"):
            cand[key] = {k: None if v is None else v.to(device) for k, v in cand[key].items()}
        if cand.get("static") is not None:
            cand["static"] = {n: v.to(device) for n, v in cand["static"].items()}

    def load_stacks(self, unit: Unit, model) -> None:
        linears = q.quantized_linears(model)
        self.W = {k: torch.stack([linears[n].weight.detach().float().clone() for n in ns])
                  for k, ns in self.kinds(unit.config).items()}
        self.probe = torch.from_numpy(self.calibrate_for(unit.config)["array"][:2]).to(self.device)

    # ---------------------------------------------------------------- phase 1
    def phase1(self, unit: Unit, model) -> None:
        began = time.perf_counter()
        config, calibration = unit.config, self.calibrate_for(unit.config)["array"]
        names = [n for ns in self.kinds(config).values() for n in ns]
        unit.baseline["val"] = self.score(unit, model, "val")
        self.log(f"  {unit.label}: BF16 val NLL {unit.baseline['val']['nll']:.4f}")
        hessians, _, mean_abs = c502.collect_calibration_statistics(model, calibration, config.micro_batch, self.device)
        self.load_stacks(unit, model)
        Hc = {k: torch.stack([hessians[n] for n in ns]) for k, ns in self.kinds(config).items()}
        A = {k: torch.stack([mean_abs[n] for n in ns]) for k, ns in self.kinds(config).items()}
        del hessians, mean_abs
        unit.geometry = {"weight_stable_rank": _layer_summary({k: c502.stable_rank(self.W[k]) for k in self.W}),
                         "signal_alignment": _layer_summary({k: c502.alignment(self.W[k], Hc[k]) for k in self.W})}
        profile = rq.forward_profile(model, self.probe)
        unit.bf16_fp4_gemms, unit.bf16_cupti = profile["scaled_mm_dispatches"], profile["cupti"]

        def run_modes(method: str, cand: dict, known: tuple | None = None, **extra) -> None:
            for mode in MODES:
                t2 = time.perf_counter()
                static, row = None, new_row(method, mode, **extra)
                if mode == "w4a16_fakequant":
                    row.update(cand["info"])
                if mode == "w4a4_hw_static":
                    cand["static"] = static = c503.calibrate(model, names, self.maps(unit, cand), calibration,
                                                             config.micro_batch, self.device)
                    values = [v.item() for v in static.values()]
                    row["static_global_scale_range"] = [min(values), max(values)]
                if known is not None and mode == DYNAMIC:
                    result, counts = known[0], {**known[1], **self.evidence(unit, model, cand, mode)}
                else:
                    result, counts = self.measure(unit, model, cand, mode, "val", static, evidence=True)
                row.update(counts)
                _fill(row, "val", result, unit.baseline["val"], time.perf_counter() - t2)
                unit.rows[(method, mode)] = row
            if method == unit.emulated:
                t2 = time.perf_counter()
                row = new_row(method, EMULATED_MODE)
                result, counts = self.measure(unit, model, cand, EMULATED_MODE, "val", evidence=True)
                row.update(counts)
                _fill(row, "val", result, unit.baseline["val"], time.perf_counter() - t2)
                unit.rows[(method, EMULATED_MODE)] = row
            self.log(f"  {unit.label}: {method:<38} val dNLL " + " ".join(
                f"{m.removeprefix('w4a4_').removeprefix('hw_')}:{unit.rows[(method, m)]['val_delta_nll']:+.4f}"
                for m in MODES))
            if cand["fits"]:
                unit.butterfly_fits[method] = cand["fits"]
            self.to(cand, "cpu")
            cand["effective"] = None  # rebuilt from the layers for the test scoring
            unit.cands[method] = cand

        probe_rotations = {}
        for method in unit.methods:
            family, rounding, awq = rq.parse_method(method)
            if not awq:
                cand = self.candidate(unit, self.W, Hc, A, family, rounding, 0.0)
                if unit.probe and rounding == "gptq":
                    probe_rotations[family] = self.maps(unit, cand)
                run_modes(method, cand)
                continue
            t2 = time.perf_counter()
            scores, kept = {}, {}
            for alpha in self.protocol["awq_alphas"]:
                cand = self.candidate(unit, self.W, Hc, A, family, rounding, alpha)
                kept[alpha] = (cand, self.measure(unit, model, cand, DYNAMIC, "val"))
                scores[alpha] = kept[alpha][1][0]["nll"]
            alphas = list(self.protocol["awq_alphas"])
            chosen = c502.select_alpha(alphas, [scores[a] for a in alphas])
            unit.awq_selection.append({"method": method, "alphas": alphas, "val_nll_hw_dynamic": [scores[a] for a in alphas],
                                       "selected_alpha": chosen, "selected_on": "val.bin, w4a4_hw_dynamic",
                                       "seconds": time.perf_counter() - t2})
            cand, known = kept.pop(chosen)
            del kept
            run_modes(method, cand, known, awq_alpha=chosen)
            torch.cuda.empty_cache()
        if unit.probe:
            stats = rq.rotated_activation_statistics(model, probe_rotations, calibration, config.micro_batch, self.device)
            unit.activation_statistics = {
                family: {metric: _layer_summary({k: torch.tensor([per_layer[n][metric] for n in ns], dtype=torch.float64)
                                                 for k, ns in self.kinds(config).items()})
                         for metric in rq.ACTIVATION_METRICS} for family, per_layer in stats.items()}
        if e2e._checkpoint_sha256(model) != unit.training["checkpoint_sha256"]:
            raise RuntimeError(f"{unit.label}: Phase 1 did not restore the checkpoint weights")
        self.W = None
        del Hc, A, probe_rotations
        torch.cuda.empty_cache()
        unit.seconds["phase1"] = time.perf_counter() - began
        self.units.append(unit)

    # ---------------------------------------------------------------- freeze and phase 2
    def freeze(self) -> list[str]:
        opened = list(self.shards.opened)
        self.ledger.freeze()
        return opened

    def open_test(self) -> None:
        self.view = {**self.view, **self.shards.open("test")}

    def phase2(self, unit: Unit) -> None:
        began = time.perf_counter()
        model = c503.load_checkpoint(unit.config, self.device, unit.checkpoint_dir / unit.training["checkpoint_file"],
                                     unit.training)
        unit.baseline["test"] = self.score(unit, model, "test")
        self.load_stacks(unit, model)
        for method, cand in unit.cands.items():
            self.to(cand, self.device)
            for mode in (*MODES, EMULATED_MODE):
                row = unit.rows.get((method, mode))
                if row is None:
                    continue
                t2 = time.perf_counter()
                result, counts = self.measure(unit, model, cand, mode, "test",
                                              cand.get("static") if mode == "w4a4_hw_static" else None)
                for key, value in counts.items():
                    if row[key] != value:
                        raise RuntimeError(f"{unit.label} {method}/{mode}: hardware count {key} changed between "
                                           "val and test")
                if mode == "w4a16_fakequant":
                    row["effective_weights_sha256_test"] = cand["effective_sha256_test"]
                _fill(row, "test", result, unit.baseline["test"], time.perf_counter() - t2)
            self.to(cand, "cpu")
        if e2e._checkpoint_sha256(model) != unit.training["checkpoint_sha256"]:
            raise RuntimeError(f"{unit.label}: Phase 2 did not restore the checkpoint weights")
        self.log(f"  {unit.label}: test BF16 {unit.baseline['test']['nll']:.4f}; hw dynamic NLL " + " ".join(
            f"{m}:{unit.rows[(m, DYNAMIC)]['test_nll']:.4f}" for m in unit.methods))
        del model
        self.W = None
        unit.cands.clear()
        torch.cuda.empty_cache()
        unit.seconds["phase2"] = time.perf_counter() - began


# ======================================================================================
# The study pipeline
# ======================================================================================
def run_study(protocol: dict, shards, device, checkpoint_dir: Path, baseline_dir: Path | None, log,
              before_test=lambda frozen_sha: None) -> dict:
    """Stages A, B and C on train.bin and val.bin, the freeze, then every test.bin scoring.

    ``before_test(selections_sha256)`` runs after the freeze and before test.bin is mapped (the access log).
    """
    began = time.perf_counter()
    seeds, stage_seconds = protocol["seeds"], {}
    study = Study(protocol, shards, device, log)
    train = study.view["train"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ab, pc = protocol["part_ab"], protocol["part_c"]
    cfg_ab, cfg_c = PRESETS[ab["preset"]], PRESETS[pc["preset"]]
    selections: dict = {}

    def study_unit(part, seed, arm, config, model, record, directory=checkpoint_dir) -> None:
        unit = Unit(part, seed, arm, config, selections, record, directory)
        study.phase1(unit, model)

    # Stage A: plain Muon checkpoints and the rotation-width probe.
    t0 = time.perf_counter()
    inits, snapshots = {}, {}
    for seed in seeds:
        inits[seed] = h.initial_state(cfg_ab, seed)
        model, record, snapshots[seed] = train_ab_arm(MUON, None, cfg_ab, train, device, protocol, seed, inits[seed],
                                                      None, checkpoint_dir / checkpoint_name("ab", seed, MUON))
        log(f"A seed {seed}: {MUON} {record['steps']} steps in {record['train_seconds']:.0f}s, "
            f"train loss {record['train_loss']:.4f}")
        study_unit("ab", seed, MUON, cfg_ab, model, record)
        del model
        torch.cuda.empty_cache()
    selections.update(select_probe([u.record_phase1() for u in study.units]))
    log(f"A selections (val.bin): {selections}")
    stage_seconds["stage_a"] = time.perf_counter() - t0

    # Stage B: QAT arms on the same seeds, batches and token budget.
    t0 = time.perf_counter()
    for seed in seeds:
        for arm in QAT_ARMS:
            qat = arm_qat("ab", arm, selections)
            model, record, _ = train_ab_arm(arm, qat, cfg_ab, train, device, protocol, seed, inits[seed],
                                            snapshots[seed], checkpoint_dir / checkpoint_name("ab", seed, arm))
            log(f"B seed {seed}: {arm} ({qat['family']}, {qat['loss']}) steps {record['first_step']}..{record['steps']} "
                f"in {record['train_seconds']:.0f}s, train loss {record['train_loss']:.4f}")
            study_unit("ab", seed, arm, cfg_ab, model, record)
            del model
            torch.cuda.empty_cache()
        del snapshots[seed], inits[seed]
    selections.update(select_transfer([u.record_phase1() for u in study.units], selections))
    log(f"B selections (val.bin): transfer arm {selections['transfer_arm']} ({selections['transfer_family']}, "
        f"{selections['transfer_loss']}); best methods {selections['arm_best_method']['ab']}")
    stage_seconds["stage_b"] = time.perf_counter() - t0

    # Stage C: the 120m baselines, the plain continuation and the selected QAT cooldown.
    t0 = time.perf_counter()
    baselines = {}
    for seed in seeds:
        if pc["baseline"] == "c5_03_preserved":
            baselines[seed] = (preserved_baseline_record(seed), Path(baseline_dir))
        else:
            path = checkpoint_dir / checkpoint_name("c", seed, BASELINE)
            baselines[seed] = (train_shakedown_baseline(cfg_c, train, device, protocol, seed, path), checkpoint_dir)
        record, directory = baselines[seed]
        base_path = directory / record["checkpoint_file"]
        model = c503.load_checkpoint(cfg_c, device, base_path, record)
        study_unit("c", seed, BASELINE, cfg_c, model, record, directory)
        del model
        torch.cuda.empty_cache()
        for arm in (CONTINUE, QAT_C):
            qat = arm_qat("c", arm, selections)
            model, arm_record = train_c_arm(arm, qat, cfg_c, train, device, protocol, seed, record, base_path,
                                            checkpoint_dir / checkpoint_name("c", seed, arm))
            log(f"C seed {seed}: {arm} steps {arm_record['first_step']}..{arm_record['steps']} in "
                f"{arm_record['train_seconds']:.0f}s, train loss {arm_record['train_loss']:.4f}")
            study_unit("c", seed, arm, cfg_c, model, arm_record)
            del model
            torch.cuda.empty_cache()
    selections["arm_best_method"]["c"] = select_part_c([u.record_phase1() for u in study.units], selections)
    stage_seconds["stage_c"] = time.perf_counter() - t0

    # Freeze, then test.bin.
    phase1_opened = study.freeze()
    selections["awq_selections"] = awq_freeze([u.record_phase1() for u in study.units])
    frozen = copy.deepcopy(selections)
    frozen_sha = selections_sha256(frozen)
    log(f"freeze after {study.ledger.counts['val']} val.bin scorings; selections {frozen_sha[:16]} "
        f"({len(frozen['awq_selections'])} AWQ selections)")
    before_test(frozen_sha)
    study.open_test()
    t0 = time.perf_counter()
    for unit in study.units:
        study.phase2(unit)
    stage_seconds["test_scoring"] = time.perf_counter() - t0
    calibration = {str(k): {key: v for key, v in cal.items() if key != "array"} for k, cal in study.calibration.items()}
    return {"configs": {"ab": asdict(cfg_ab), "c": asdict(cfg_c)}, "calibration": calibration,
            "splits_opened_before_freeze": phase1_opened, "scoring_ledger": study.ledger.record(),
            "selections": frozen, "selections_sha256": frozen_sha, "units": [u.record() for u in study.units],
            "stage_seconds": stage_seconds, "seconds": time.perf_counter() - began}


# ======================================================================================
# Statistics (pure functions of the stored rows; the validator recomputes them)
# ======================================================================================
def throughput(unit: dict, reference: dict) -> dict:
    """Median step time of an arm against the plain run of the same seed over the same optimizer steps."""
    tr, ref = unit["training"], reference["training"]
    seconds = tr["step_seconds"]
    ref_seconds = ref["step_seconds"][tr["first_step"] - ref["first_step"]:tr["steps"] - ref["first_step"]]
    median, ref_median = statistics.median(seconds), statistics.median(ref_seconds)
    return {"steps": [tr["first_step"], tr["steps"]], "step_ms_median": 1e3 * median,
            "tokens_per_second": tr["step_tokens"] / median, "reference_arm": reference["arm"],
            "reference_step_ms_median": 1e3 * ref_median,
            "reference_tokens_per_second": ref["step_tokens"] / ref_median, "step_time_ratio": median / ref_median}


def contrast_specs(selections: dict) -> list[tuple[str, str, tuple, tuple]]:
    """(part, name, x, y) with x, y = (arm, method or BF16). Differences are x - y on hardware dynamic rows."""
    best_ab, best_c = selections["arm_best_method"]["ab"], selections["arm_best_method"]["c"]
    wide_ptq = selections["best_wide_ptq_method"]
    out = [("ab", "probe_wide_ptq_vs_hadamard_b16", (MUON, wide_ptq), (MUON, C5_HADAMARD)),
           ("ab", "probe_wide_ptq_vs_awq_butterfly_b16", (MUON, wide_ptq), (MUON, C5_BEST)),
           ("ab", "probe_hadamard_b64_vs_hadamard_b16", (MUON, "hadamard_b64_gptq"), (MUON, C5_HADAMARD)),
           ("ab", "probe_kron_full_vs_hadamard_b16", (MUON, "kron_hadamard_full_gptq"), (MUON, C5_HADAMARD))]
    for arm in QAT_ARMS:
        family = arm_qat("ab", arm, selections)["family"]
        out += [("ab", f"{arm}_bf16_vs_muon_bf16", (arm, BF16), (MUON, BF16)),
                ("ab", f"{arm}_best_vs_muon_best", (arm, best_ab[arm]), (MUON, best_ab[MUON]))]
        for rounding in ("gptq", "rtn"):
            method = rq.method_name(family, rounding)
            out.append(("ab", f"{arm}_own_{rounding}_vs_muon_same_method", (arm, method), (MUON, method)))
    own = rq.method_name(selections["transfer_family"], "gptq")
    out += [("c", "wide_ptq_vs_c5_03_awq_butterfly", (BASELINE, wide_ptq), (BASELINE, C5_BEST)),
            ("c", "wide_ptq_vs_c5_03_hadamard", (BASELINE, wide_ptq), (BASELINE, C5_HADAMARD)),
            ("c", "qat_best_vs_c5_03_awq_butterfly", (QAT_C, best_c[QAT_C]), (BASELINE, C5_BEST)),
            ("c", "qat_best_vs_c5_03_hadamard", (QAT_C, best_c[QAT_C]), (BASELINE, C5_HADAMARD)),
            ("c", "qat_best_vs_bf16_continue_best", (QAT_C, best_c[QAT_C]), (CONTINUE, best_c[CONTINUE])),
            ("c", "qat_own_gptq_vs_bf16_continue_same_method", (QAT_C, own), (CONTINUE, own)),
            ("c", "bf16_continue_best_vs_c5_03_awq_butterfly", (CONTINUE, best_c[CONTINUE]), (BASELINE, C5_BEST)),
            ("c", "qat_bf16_vs_baseline_bf16", (QAT_C, BF16), (BASELINE, BF16)),
            ("c", "qat_bf16_vs_bf16_continue_bf16", (QAT_C, BF16), (CONTINUE, BF16)),
            ("c", "bf16_continue_bf16_vs_baseline_bf16", (CONTINUE, BF16), (BASELINE, BF16)),
            ("c", "qat_best_vs_baseline_bf16", (QAT_C, best_c[QAT_C]), (BASELINE, BF16))]
    return out


def _series(units: list[dict], part: str, side: tuple, split: str, endpoint: str) -> list[float]:
    arm, method = side
    out = []
    for unit in _units(units, part, arm):
        if method == BF16:
            out.append(unit[f"baseline_{split}"]["nll"] if endpoint == "nll" else 0.0)
        else:
            out.append(_row(unit, method, DYNAMIC)[f"{split}_{endpoint}"])
    return out


def verdict(stats: dict) -> str:
    """Pre-declared reading of a contrast x - y of an NLL: lower is better; 'significant' is Holm t-test p < 0.05."""
    if stats["mean"] == 0:
        return "ties (no difference)"
    side = "helps" if stats["mean"] < 0 else "loses"
    return f"{side} (significant)" if stats["p_holm"] < 0.05 else f"ties ({side} direction, not significant)"


def analyze(units: list[dict], selections: dict, protocol: dict, reference: dict | None) -> dict:
    cells = {}
    for part, arms in (("ab", ARMS_AB), ("c", ARMS_C)):
        for arm in arms:
            group = _units(units, part, arm)
            entry = {"bf16": {split: c4.mean_sd([u[f"baseline_{split}"]["nll"] for u in group]) for split in SPLITS},
                     "methods": {}}
            for method, mode in [(r["method"], r["mode"]) for r in group[0]["grid"]]:
                rows = [_row(u, method, mode) for u in group]
                cell = {split: {"nll": c4.mean_sd([r[f"{split}_nll"] for r in rows]),
                                "delta_nll": c4.mean_sd([r[f"{split}_delta_nll"] for r in rows]),
                                "relative_delta_ppl_pct": c4.mean_sd([r[f"{split}_relative_delta_ppl_pct"]
                                                                     for r in rows])} for split in SPLITS}
                entry["methods"][f"{method}/{mode}"] = cell
            for method in group[0]["methods"]:  # share of the dynamic W4A4 loss that activations add to W4A16
                for split in SPLITS:
                    w4a4 = entry["methods"][f"{method}/{DYNAMIC}"][split]["delta_nll"]["mean"]
                    w4a16 = entry["methods"][f"{method}/w4a16_fakequant"][split]["delta_nll"]["mean"]
                    entry["methods"][f"{method}/{DYNAMIC}"][split]["activation_share_of_delta"] = \
                        (w4a4 - w4a16) / w4a4 if w4a4 > 0 else None
            cells[f"{part}/{arm}"] = entry
    contrasts = []
    for part, name, x, y in contrast_specs(selections):
        entry = {"part": part, "name": name, "x": list(x), "y": list(y), "mode": DYNAMIC}
        for split in SPLITS:
            entry[split] = {}
            for endpoint in ENDPOINTS:
                diffs = [a - b for a, b in zip(_series(units, part, x, split, endpoint),
                                               _series(units, part, y, split, endpoint))]
                entry[split][endpoint] = {**c4.paired_t_df2(diffs), "p_sign_flip": c502.sign_flip_p(diffs)}
        contrasts.append(entry)
    for part in PARTS:  # Holm-Bonferroni inside one part, split and endpoint
        family = [e for e in contrasts if e["part"] == part]
        for split in SPLITS:
            for endpoint in ENDPOINTS:
                for key, src in (("p_holm", "p_two_sided"), ("p_sign_flip_holm", "p_sign_flip")):
                    adjusted = c4.holm_bonferroni([e[split][endpoint][src] for e in family])
                    for e, p in zip(family, adjusted):
                        e[split][endpoint][key] = p
    for e in contrasts:
        for split in SPLITS:
            for endpoint in ENDPOINTS:
                e[split][endpoint]["verdict"] = verdict(e[split][endpoint])
    rows = [r for u in units for r in u["grid"]]
    hw = [r for r in rows if r["mode"] in HW_MODES]
    w4a16 = [r for r in rows if r["mode"] == "w4a16_fakequant"]
    parity = []
    for u in units:
        for split in SPLITS:
            hw_nll = _row(u, u["emulated_method"], DYNAMIC)[f"{split}_nll"]
            em_nll = _row(u, u["emulated_method"], EMULATED_MODE)[f"{split}_nll"]
            parity.append({"part": u["part"], "seed": u["seed"], "arm": u["arm"], "method": u["emulated_method"],
                           "split": split, "hw_nll": hw_nll, "emulated_nll": em_nll, "abs_diff": abs(hw_nll - em_nll)})
    max_diff = max(p["abs_diff"] for p in parity)
    speed = {}
    for part, arms, ref_arm in (("ab", QAT_ARMS, MUON), ("c", (QAT_C,), CONTINUE)):
        for arm in arms:
            per_seed = [throughput(u, r) for u, r in zip(_units(units, part, arm), _units(units, part, ref_arm))]
            speed[f"{part}/{arm}"] = {"per_seed": per_seed,
                                      "step_time_ratio": c4.mean_sd([t["step_time_ratio"] for t in per_seed]),
                                      "tokens_per_second": c4.mean_sd([t["tokens_per_second"] for t in per_seed]),
                                      "reference_tokens_per_second": c4.mean_sd(
                                          [t["reference_tokens_per_second"] for t in per_seed])}
    probe = _units(units, "ab", MUON)
    activation = {family: {metric: c4.mean_sd([u["activation_statistics"][family][metric]["all"]["mean"] for u in probe])
                           for metric in rq.ACTIVATION_METRICS} for family in PROBE_FAMILIES}
    geometry = {f"{part}/{arm}": {metric: c4.mean_sd([u["geometry"][metric]["all"]["mean"] for u in _units(units, part, arm)])
                                  for metric in ("weight_stable_rank", "signal_alignment")}
                for part, arms in (("ab", ARMS_AB), ("c", ARMS_C)) for arm in arms}
    reproduction = None
    if reference is not None:
        entries = []
        for u in _units(units, "c", BASELINE):
            ref = reference["seeds"][str(u["seed"])]
            for split in SPLITS:
                entries.append({"seed": u["seed"], "row": "bf16", "split": split, "c5_03_nll": ref["bf16"][split],
                                "nll": u[f"baseline_{split}"]["nll"],
                                "abs_diff": abs(u[f"baseline_{split}"]["nll"] - ref["bf16"][split])})
                for method in (C5_HADAMARD, C5_BEST):
                    for mode, theirs in ref["methods"][method].items():
                        mine = _row(u, method, mode)[f"{split}_nll"]
                        entries.append({"seed": u["seed"], "row": f"{method}/{mode}", "split": split,
                                        "c5_03_nll": theirs[split], "nll": mine, "abs_diff": abs(mine - theirs[split])})
        reproduction = {"rows": entries, "max_abs_nll_difference": max(e["abs_diff"] for e in entries)}
    return {
        "n_seeds": N_SEEDS, "df": N_SEEDS - 1, "sign_flip_min_p": SIGN_FLIP_MIN_P,
        "endpoints": "nll: final NLL of the quantized (or BF16) model; delta_nll: quantized minus the BF16 NLL of "
                     "the same checkpoint (0 for a BF16 side). Contrast differences are x - y; lower is better.",
        "cells": cells, "contrasts": contrasts,
        "hardware": {"fp4_gemms_per_forward": sorted({r["fp4_gemms_per_forward"] for r in hw}),
                     "fp4_gemms_per_forward_emulated": sorted({r["fp4_gemms_per_forward"] for r in rows
                                                               if r["mode"] == EMULATED_MODE}),
                     "fp4_gemms_per_forward_bf16": sorted({u["bf16_fp4_gemms_per_forward"] for u in units}),
                     "bf16_block_linears_in_hw_rows": sorted({r["bf16_block_linears"] for r in hw}),
                     "cupti_fp4_gemm_kernels": sorted({r["cupti"]["fp4_gemm_kernels"] for r in hw}),
                     "cupti_nvfp4_quant_kernels": sorted({r["cupti"]["nvfp4_quant_kernels"] for r in hw}),
                     "hw_rows": len(hw), "wide_hw_rows": sum(r["wide"] for r in hw),
                     "decode_checked_rows": len(w4a16),
                     "decode_checked_matrices": sum(r["hw_weights_checked_matrices"] for r in w4a16),
                     "decode_exact_matrices": sum(r["hw_weights_decode_exact_matrices"] for r in w4a16)},
        "hardware_vs_emulation": {"rows": parity, "max_abs_nll_difference": max_diff,
                                  "tolerance_nats": protocol["emulation_parity_nats"],
                                  "passed": max_diff <= protocol["emulation_parity_nats"]},
        "throughput": speed, "activation_statistics": activation, "geometry": geometry,
        "c5_03_reproduction": reproduction,
    }


def c5_03_reference() -> dict:
    """The published C5-03 rows of the comparator methods on the preserved checkpoints (committed result)."""
    blob = (ROOT / C5_03_RESULT_REPO_PATH).read_bytes()
    source = json.loads(blob)
    modes = {"w4a16_fakequant": "w4a16_fakequant", DYNAMIC: DYNAMIC, "w4a4_hw_static": "w4a4_hw_static_bf16cal"}
    seeds = {}
    for s in source["part_c"]["seeds"]:
        grid = {(r["method"], r["mode"]): r for r in s["grid"]}
        seeds[str(s["seed"])] = {
            "checkpoint_sha256": s["training"]["checkpoint_sha256"],
            "checkpoint_file_sha256": s["training"]["checkpoint_file_sha256"],
            "bf16": {split: s[f"baseline_{split}"]["nll"] for split in SPLITS},
            "methods": {method: {mine: {split: grid[(method, theirs)][f"{split}_nll"] for split in SPLITS}
                                 for mine, theirs in modes.items()} for method in (C5_HADAMARD, C5_BEST)},
            "awq_alpha": grid[(C5_BEST, DYNAMIC)]["awq_alpha"]}
    return {"result": C5_03_RESULT_REPO_PATH, "result_sha256": hashlib.sha256(blob).hexdigest(),
            "awq_alphas": source["protocol"]["part_c"]["awq_alphas"], "seeds": seeds}


def _parse_c5_03_blob(blob: bytes) -> dict:
    """Parse and SHA-256-verify a C5-03 result blob against ``C5_03_RESULT_SHA256``."""
    digest = hashlib.sha256(blob).hexdigest()
    if digest != C5_03_RESULT_SHA256:
        raise ValueError(f"{C5_03_RESULT_REPO_PATH} SHA-256 {digest} differs from pinned {C5_03_RESULT_SHA256}")
    source = json.loads(blob)
    modes = {"w4a16_fakequant": "w4a16_fakequant", DYNAMIC: DYNAMIC, "w4a4_hw_static": "w4a4_hw_static_bf16cal"}
    seeds = {}
    for s in source["part_c"]["seeds"]:
        grid = {(r["method"], r["mode"]): r for r in s["grid"]}
        seeds[str(s["seed"])] = {
            "checkpoint_sha256": s["training"]["checkpoint_sha256"],
            "checkpoint_file_sha256": s["training"]["checkpoint_file_sha256"],
            "bf16": {split: s[f"baseline_{split}"]["nll"] for split in SPLITS},
            "methods": {method: {mine: {split: grid[(method, theirs)][f"{split}_nll"] for split in SPLITS}
                                 for mine, theirs in modes.items()} for method in (C5_HADAMARD, C5_BEST)},
            "awq_alpha": grid[(C5_BEST, DYNAMIC)]["awq_alpha"]}
    return {"result": C5_03_RESULT_REPO_PATH, "result_sha256": digest,
            "awq_alphas": source["protocol"]["part_c"]["awq_alphas"], "seeds": seeds}


def _trusted_c5_03_reference(git_sha: str | None = None) -> dict:
    """Load the pinned C5-03 reference from Git (when ``git_sha`` is given) and the checkout file."""
    disk_ref = _parse_c5_03_blob((ROOT / C5_03_RESULT_REPO_PATH).read_bytes())
    if git_sha is None:
        return disk_ref
    blob_ref = _parse_c5_03_blob(c503._git_blob(git_sha, C5_03_RESULT_REPO_PATH))
    _require(_same(disk_ref, blob_ref), "checkout C5-03 result differs from the producer commit's Git blob")
    return blob_ref


# ======================================================================================
# Hardware parity of the cross-mix path (recorded with the results)
# ======================================================================================
def cross_mix_parity(device) -> dict:
    """The fused quantizer on cross-mixed operands, bit for bit against the eager reference.

    For every wide family and width: ``quantize_activation(C x, R16)`` equals
    ``quantize_activation_ref(C x, R16)`` (dynamic and static scale), with ``C`` the FP32 cross-block mix.
    """
    failures, cases = [], 0
    for M, K in ((1, 128), (37, 384), (129, 512), (64, 768), (300, 1408), (16, 2048)):
        g = torch.Generator(device=device).manual_seed(M * 31 + K)
        x = (torch.randn(M, K, generator=g, device=device) * 3).to(torch.bfloat16)
        x[:, :16] *= 30
        W = torch.randn(1, 32, K, generator=g, device=device) * 0.05
        for family in rq.WIDE_FAMILIES:
            rot, _ = rq.make_rotation(family, W, butterfly_steps=2)
            angles = rot.angles[0] if hasattr(rot, "angles") else None
            mixed = rq.cross_of(rot).forward(x.float())
            name = rq.kernel_rotation(family)
            tables = nb.butterfly_tables(angles) if angles is not None else None
            ref = nb.quantize_activation_ref(mixed, name, angles)
            for static in (None, ref.global_scale * 0.5):
                cases += 1
                want = nb.quantize_activation_ref(mixed, name, angles, static)
                if not c503._same_operands(nb.quantize_activation(mixed, name, tables, static), want):
                    failures.append([M, K, family, static is not None])
    return {"cases": cases, "bitwise_failures": failures, "passed": not failures,
            "reference": "FP32 cross-block mix, then micro_rotated_ptq block-16 rotation + nvfp4_encode + pack_e2m1 "
                         "+ to_blocked"}


# ======================================================================================
# Summary and caveats
# ======================================================================================
def summarize(payload: dict) -> dict:
    a, sel = payload["analysis"], payload["selections"]

    def contrast(name: str) -> dict:
        e = next(c for c in a["contrasts"] if c["name"] == name)
        return {"x": e["x"], "y": e["y"], **{
            f"{split}_{endpoint}": {k: e[split][endpoint][k] for k in ("mean", "ci95", "p_two_sided", "p_holm",
                                                                      "p_sign_flip", "verdict")}
            for split in SPLITS for endpoint in ENDPOINTS}}

    def arm(part: str, name: str) -> dict:
        cell, best = a["cells"][f"{part}/{name}"], sel["arm_best_method"][part][name]
        row = cell["methods"][f"{best}/{DYNAMIC}"]
        return {"bf16_test_nll": cell["bf16"]["test"]["mean"], "best_method_by_val": best,
                "best_w4a4_dynamic_test_nll": row["test"]["nll"]["mean"],
                "best_w4a4_dynamic_test_delta_nll": row["test"]["delta_nll"]["mean"],
                "best_w4a4_static_test_nll": cell["methods"][f"{best}/w4a4_hw_static"]["test"]["nll"]["mean"],
                "best_w4a4_dynamic_test_relative_delta_ppl_pct": row["test"]["relative_delta_ppl_pct"]["mean"]}

    probe = a["cells"][f"ab/{MUON}"]["methods"]
    return {
        "selections": sel,
        "part_a_rotation_width": {m: {"test_delta_nll_w4a16": probe[f"{m}/w4a16_fakequant"]["test"]["delta_nll"]["mean"],
                                      "test_delta_nll_w4a4_dynamic": probe[f"{m}/{DYNAMIC}"]["test"]["delta_nll"]["mean"],
                                      "test_delta_nll_w4a4_static": probe[f"{m}/w4a4_hw_static"]["test"]["delta_nll"]["mean"],
                                      "val_delta_nll_w4a4_dynamic": probe[f"{m}/{DYNAMIC}"]["val"]["delta_nll"]["mean"],
                                      "activation_share_of_test_delta": probe[f"{m}/{DYNAMIC}"]["test"][
                                          "activation_share_of_delta"]} for m in PROBE_METHODS},
        "part_a_activation_statistics": {f: {m: v["mean"] for m, v in metrics.items()}
                                         for f, metrics in a["activation_statistics"].items()},
        "part_b_arms": {name: {**arm("ab", name), **({} if name == MUON else {
            "step_time_ratio_vs_muon": a["throughput"][f"ab/{name}"]["step_time_ratio"]["mean"],
            "tokens_per_second": a["throughput"][f"ab/{name}"]["tokens_per_second"]["mean"]})} for name in ARMS_AB},
        "part_b_contrasts": {c["name"]: contrast(c["name"]) for c in a["contrasts"] if c["part"] == "ab"},
        "part_c_arms": {name: arm("c", name) for name in ARMS_C},
        "part_c_qat_step_time_ratio_vs_bf16_continue": a["throughput"][f"c/{QAT_C}"]["step_time_ratio"]["mean"],
        "part_c_contrasts": {c["name"]: contrast(c["name"]) for c in a["contrasts"] if c["part"] == "c"},
        "hardware": a["hardware"],
        "hardware_vs_emulation_max_abs_nll_difference": a["hardware_vs_emulation"]["max_abs_nll_difference"],
        "c5_03_reproduction_max_abs_nll_difference": (a["c5_03_reproduction"] or {}).get("max_abs_nll_difference"),
    }


def _caveats_v1(payload: dict) -> list[str]:
    protocol, sel, data = payload["protocol"], payload["selections"], payload["data"]
    ab, pc = protocol["part_ab"], protocol["part_c"]
    cfg_ab, cfg_c = PRESETS[ab["preset"]], PRESETS[pc["preset"]]
    env = (payload.get("meta") or {}).get("environment") or {}
    cap = env.get("capability")
    device = f"{env.get('device') or protocol['device']}" + (f" (sm_{cap[0]}{cap[1]})" if cap else "")
    extra = pc["cooldown_steps"] * cfg_c.step_tokens
    base_tokens = pc["baseline_steps"] * cfg_c.step_tokens
    notes = [
        f"Statistics have df = {N_SEEDS - 1} ({N_SEEDS} paired seeds): the two-sided t-test has little power and the "
        f"exact sign-flip p-value cannot fall below {SIGN_FLIP_MIN_P:g}, so no sign-flip result can be significant. "
        "Holm-Bonferroni runs inside one part, split and endpoint; 'ties' means the difference is not significant, "
        "not that it is zero.",
        f"Part B arms share the token budget: {ab['steps']} steps ({ab['steps'] * cfg_ab.step_tokens:,} tokens), the "
        f"same batches, and cooldown arms branch from the Muon run at step {ab['cooldown_from']}.",
        f"Part C token budgets differ. The baseline has {pc['baseline_steps']} steps ({base_tokens:,} tokens); "
        f"{CONTINUE} and {QAT_C} add {pc['cooldown_steps']} steps ({extra:,} tokens, +{100 * extra / base_tokens:.1f}%) "
        f"on new train.bin batches at the terminal learning rate (x{protocol['min_lr_ratio']}). {CONTINUE} is the "
        f"token-matched control for {QAT_C}; a contrast of {QAT_C} with the baseline mixes the QAT effect with the "
        "extra tokens.",
        "Part C cooldowns start Muon momentum and the auxiliary AdamW from zero" + (
            ": the preserved C5-03 checkpoints hold weights only." if pc["baseline"] == "c5_03_preserved"
            else " (as the canonical run must: the preserved checkpoints hold weights only)."),
        f"Selections use val.bin only: the wide QAT family ({sel['qat_wide_family']}), the best wide PTQ method "
        f"({sel['best_wide_ptq_method']}), the transfer arm ({sel['transfer_arm']}), the best method of every arm, and "
        "the AWQ alpha of every (checkpoint, method). The 'best method' of an arm is chosen per arm, so a best-vs-best "
        "contrast compares deployment pipelines, not one method.",
        f"AWQ alphas are {protocol['awq_alphas']} (C5-03 used [0.0, 0.25, 0.5, 0.75]); alpha is chosen per checkpoint "
        "on val.bin with live hardware W4A4 dynamic scoring.",
        "Wide rotations run a cross-block mix as FP32 PyTorch matrix products before the fused NVFP4 quantizer "
        "(only the block-16 part is in the kernel). Inference latency of that extra pass is not measured here, and "
        "the hardware rows are perplexity results, not speed results.",
        "QAT fake quantization is FP32 round-to-nearest NVFP4 of both operands with a BF16 GEMM; training does not "
        "run the FP4 GEMM. Final rows requantize the weights with RTN or GPTQ and run the native FP4 GEMM.",
        f"Step times are wall clock per optimizer step on one {device} that also drives a desktop; the ratio to the "
        "plain run of the same seed over the same steps is the throughput result. Part B full-run arms compare all "
        f"{ab['steps']} steps, cooldown arms the last {ab['steps'] - ab['cooldown_from']}.",
        f"The dual loss weight ({protocol['dual_bf16_weight']} BF16 / {1 - protocol['dual_bf16_weight']} W4A4), the "
        "cooldown lengths and the arm list were fixed before the canonical run; QAT settings were developed on the "
        "synthetic shakedown only.",
        "GPU training is seeded (initial state, batch stream) but not claimed to be bit-reproducible; the checkpoints "
        "are retained outside git with their tensor and file SHA-256 values.",
    ]
    if payload["shakedown"]:
        notes.append(
            f"SHAKEDOWN: synthetic Markov shards ({data['source'].get('name')}) and the {ab['preset']} model. The "
            "numbers test the pipeline and say nothing about real text; no real split was read.")
        return notes
    access = data["test_access"]
    prior = len(access["prior_openings"])
    notes.append(
        "test.bin is held out from every selection of this run (it is mapped once, after the freeze), but it is not "
        "an unseen holdout at study level: earlier cycles scored it, and C5-03 scored the same preserved 120m "
        f"checkpoints with {C5_HADAMARD} and {C5_BEST}.")
    notes.append(
        f"The C6-01 test-access log ({access['log']}) had {prior} earlier opening{'s' if prior != 1 else ''} of "
        "test.bin by this driver when this run mapped it" + (
            "." if prior == 0 else ": " + "; ".join(f"{e.get('git_sha', '?')[:7]} at {e.get('timestamp')}"
                                                    for e in access["prior_openings"]) + "."))
    rep = payload["analysis"]["c5_03_reproduction"]
    notes.append(
        f"The {BASELINE} rows re-measure C5-03 methods on the same checkpoints in this session (max |NLL difference| "
        f"to the published C5-03 rows: {rep['max_abs_nll_difference']:.2e} nats). The contrasts use the rows of this "
        "run, not the published values.")
    return notes


_V1_HELD_OUT_NOTE = "test.bin is held out from every selection of this run (it is mapped once, after the freeze)"


def caveats(payload: dict) -> list[str]:
    """The caveats of a result. ``_caveats_v1`` is the retained v1 list: the preserved result reproduces it verbatim.

    A v2 result has the same notes with the held-out note reworded (the scoring view maps test.bin after the
    freeze; the manifest verification reads it earlier) and three more: unequal method menus, the exact
    split-hygiene guarantee, and what the freeze digest covers in v1 and in v2.
    """
    notes = _caveats_v1(payload)
    if payload["schema_version"] == HISTORICAL_SCHEMA_V1:
        return notes
    protocol, sel = payload["protocol"], payload["selections"]
    ab, pc = protocol["part_ab"], protocol["part_c"]
    best_ab, best_c = sel["arm_best_method"]["ab"], sel["arm_best_method"]["c"]
    held = [i for i, note in enumerate(notes) if note.startswith(_V1_HELD_OUT_NOTE)]
    if len(held) != (0 if payload["shakedown"] else 1):
        raise RuntimeError("the v1 held-out note is not where the v2 caveats expect it")
    for i in held:
        notes[i] = (
            "test.bin is held out from every selection of this run: the scoring view maps it once, after the "
            "freeze. It is not an unseen holdout at study level: earlier cycles scored it, and C5-03 scored the "
            f"same preserved 120m checkpoints with {C5_HADAMARD} and {C5_BEST}.")
    composites = [m for m in PROBE_METHODS if m.startswith("cross_")]
    qat_menu = max(len(unit_methods("ab", arm, sel)) for arm in QAT_ARMS)
    shared = set(unit_methods("c", QAT_C, sel)) & set(unit_methods("c", CONTINUE, sel))
    notes.append(
        f"Method menus differ in Part B ({ab['preset']}). {MUON} chooses its best method from all "
        f"{len(PROBE_METHODS)} probe methods (chosen: {best_ab[MUON]}), which include the cross-mix butterfly "
        f"composites ({', '.join(composites)}). Every Part B QAT arm chooses from its own RTN and GPTQ methods, "
        f"{C5_HADAMARD} and {C5_BEST} (at most {qat_menu} methods) and is not offered those composites. A Part B "
        "best-vs-best contrast therefore compares deployment pipelines with unequal menus; use the same-method "
        "contrasts (*_own_gptq_vs_muon_same_method, *_own_rtn_vs_muon_same_method) for claims about the effect of "
        f"training. In Part C ({pc['preset']}), the best methods of {QAT_C} ({best_c[QAT_C]}) and {CONTINUE} "
        f"({best_c[CONTINUE]}) are " + (
            "offered to both arms." if {best_c[QAT_C], best_c[CONTINUE]} <= shared
            else "not offered to both arms, so the same caution applies there."))
    notes.append(
        "Split hygiene is a guarantee of zero pre-freeze test scoring or selection: the scoring ledger refuses a "
        "test.bin scoring before the freeze, and no selection reads a test.bin score. It is not a guarantee that no "
        "test.bin byte is read earlier: " + (
            "the run writes and hashes its synthetic test.bin when it generates the shards"
            if payload["shakedown"] else
            "the manifest verification of real_text_data.py hashes and memory-maps test.bin (SHA-256, token range, "
            "document counts) before the freeze") +
        ". splits_opened_before_freeze and the test-access log count the scoring interface only.")
    notes.append(
        f"Freeze coverage ({SCHEMA_VERSION}): selections_sha256 covers the wide family, the transfer arm, the best "
        f"method of every arm and all {len(sel['awq_selections'])} AWQ selections (the selected alpha, the val.bin "
        "score of every alpha and the SHA-256 of the selected effective weights). The run appends that digest to "
        f"the test-access log before it maps test.bin. The historical {HISTORICAL_SCHEMA_V1} result differs: its "
        "pre-test digest covered the family, arm and method names only, not the AWQ alphas. The validator binds "
        "the v1 AWQ selections with a digest that was pinned after the run (an anti-tamper pin, not a "
        "contemporaneous freeze).")
    return notes


# ======================================================================================
# Guards and provenance
# ======================================================================================
_require, _same, _assert_close = c502._require, c502._same, c502._assert_close
_is_number, _nonneg, _is_int, _is_hex = c502._is_number, c502._nonneg, c502._is_int, c503._is_hex


def reserved(path: Path) -> bool:
    path = Path(path)
    return any(fnmatch.fnmatchcase(name, RESERVED_PATTERN) for name in (path.name, path.resolve().name))


def require_canonical_host(device, git: dict, output: Path) -> None:
    """The canonical run needs an sm_120+ device, a clean committed tree and a canonical output that does not exist."""
    if device.type != "cuda" or not torch.cuda.is_available() or \
            tuple(torch.cuda.get_device_capability(device)) < (12, 0):
        raise RuntimeError(f"{Path(output).name} requires a Blackwell (sm_120+) CUDA device")
    if git.get("git_dirty") is not False or not _is_hex(git.get("git_sha"), 40):
        raise RuntimeError("The canonical run needs a clean, committed git tree (git_dirty must be False)")
    if Path(output).exists() or Path(output).is_symlink():
        raise RuntimeError(f"{output} exists: the canonical run scores test.bin once and does not overwrite a result")


def _import_closure(source_of) -> tuple[str, ...]:
    found, todo = set(), [DRIVER_MODULE]
    while todo:
        name = todo.pop()
        if name in found:
            continue
        text = source_of(name)
        if text is None:
            continue
        found.add(name)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("research"):
                todo += [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
            elif isinstance(node, ast.Import):
                todo += [alias.name for alias in node.names if alias.name.startswith("research")]
    return tuple(sorted(found))


def _module_path(name: str) -> Path:
    return ROOT.joinpath(*name.split(".")).with_suffix(".py")


@functools.cache
def code_modules() -> tuple[str, ...]:
    """Every ``research.*`` module this driver imports, directly or transitively (static ``ast`` closure)."""
    return _import_closure(lambda name: _module_path(name).read_text() if _module_path(name).is_file() else None)


def code_sha256() -> str:
    digest = hashlib.sha256()
    for name in code_modules():
        digest.update(name.encode() + b"\0" + _module_path(name).read_bytes())
    return digest.hexdigest()


@functools.cache
def _code_modules_at_commit(git_sha: str) -> tuple[str, ...]:
    listing = c503._git("ls-tree", "-r", "--name-only", git_sha, "--", "research")
    if listing.returncode != 0:
        raise ValueError(f"cannot list the tree of commit {git_sha}")
    files = set(listing.stdout.decode().splitlines())
    return _import_closure(lambda n: c503._git_blob(git_sha, c503._module_repo_path(n)).decode()
                           if c503._module_repo_path(n) in files else None)


@functools.cache
def _schema_at_commit(git_sha: str) -> str | None:
    """The ``SCHEMA_VERSION`` that the driver committed at ``git_sha`` declares."""
    tree = ast.parse(c503._git_blob(git_sha, SCRIPT_REPO_PATH).decode())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == "SCHEMA_VERSION" and isinstance(node.value, ast.Constant):
            return node.value.value
    return None


# The driver definitions that recompute a result from its rows: the constants, the method grids, the selection
# rules, ``analyze`` and ``summarize``. A validating checkout must hold the producer's AST of every one of them.
SUMMARY_DEFINITIONS = (
    "TASK", "TOL", "N_SEEDS", "SIGN_FLIP_MIN_P", "SPLITS", "ENDPOINTS", "PINNED_BASELINES", "PRESETS", "MODES",
    "HW_MODES", "DYNAMIC", "EMULATED_MODE", "BF16", "MUON", "ARMS_AB", "QAT_ARMS", "TRANSFER_CANDIDATES",
    "WIDE_QAT_FAMILIES", "BASELINE", "CONTINUE", "QAT_C", "ARMS_C", "PARTS", "C5_HADAMARD", "C5_BUTTERFLY",
    "C5_BEST", "PROBE_FAMILIES", "PROBE_METHODS", "PROBE_EMULATED", "CANONICAL", "SHAKEDOWN",
    "C5_03_RESULT_REPO_PATH", "_dedupe", "arm_qat", "unit_methods", "emulated_method", "expected_grid", "_units",
    "_row", "summed_nll", "best_method", "select_probe", "select_transfer", "select_part_c", "selections_sha256",
    "throughput", "contrast_specs", "_series", "verdict", "analyze", "c5_03_reference", "summarize")


_DYNAMIC_MUTATORS = frozenset({"globals", "locals", "setattr", "delattr", "exec", "eval", "__import__", "vars"})


def _definitions(source: str) -> dict[str, str]:
    """``ast.dump`` of every top-level assignment and function (decorators, args, return type and body)."""
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = "".join((
                "".join(ast.dump(d) for d in node.decorator_list),
                ast.dump(node.args),
                "" if node.returns is None else ast.dump(node.returns),
                "".join(ast.dump(stmt) for stmt in node.body),
            ))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names = [target] if isinstance(target, ast.Name) else getattr(target, "elts", [])
                for name in names:
                    if isinstance(name, ast.Name):
                        out[name.id] = ast.dump(target) + ast.dump(node.value)
    return out


def _target_names(node: ast.AST):
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            yield from _target_names(elt)
    elif isinstance(node, ast.Starred):
        yield from _target_names(node.value)


def _is_main_guard(node: ast.AST) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
            and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1 and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__")


def _walk_module_scope(node: ast.AST):
    """Yield every AST node executed in module scope (skipping function/lambda bodies and ``__main__``)."""
    yield node
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return
    for child in ast.iter_child_nodes(node):
        if not _is_main_guard(child):
            yield from _walk_module_scope(child)


def _module_ast_clean(committed_tree: ast.Module, current_tree: ast.Module, tracked: set[str]) -> bool:
    """Reject module-scope rebindings, augmented/annotated assignments, shadowed imports, globals or mutators."""
    committed_imports = {ast.dump(n) for n in committed_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    current_imports = [n for n in current_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not committed_imports <= {ast.dump(n) for n in current_imports}:
        return False
    import_names = {alias.asname or alias.name.split(".")[0] for node in current_imports for alias in node.names}
    guarded = tracked | import_names
    top_ids = {id(n) for n in current_tree.body}
    bindings = dict.fromkeys(guarded, 0)
    for node in ast.walk(current_tree):
        if isinstance(node, ast.Global) and any(name in guarded for name in node.names):
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _DYNAMIC_MUTATORS:
            return False
    for node in _walk_module_scope(current_tree):
        if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.Delete, ast.NamedExpr)):
            targets = [node.target] if not isinstance(node, ast.Delete) else node.targets
            if any(name in guarded for tgt in targets for name in _target_names(tgt)):
                return False
        elif isinstance(node, ast.ClassDef) and node.name in guarded:
            return False
        elif isinstance(node, (ast.For, ast.AsyncFor)) and any(n in guarded for n in _target_names(node.target)):
            return False
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            if any(n in guarded for item in node.items if item.optional_vars is not None
                   for n in _target_names(item.optional_vars)):
                return False
        elif isinstance(node, ast.ExceptHandler) and node.name in guarded:
            return False
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in guarded:
                if id(node) not in top_ids:
                    return False
                bindings[node.name] += 1
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                for name in _target_names(tgt):
                    if name in guarded:
                        if id(node) not in top_ids:
                            return False
                        bindings[name] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if name in guarded:
                    if id(node) not in top_ids:
                        return False
                    bindings[name] += 1
    return all(bindings[name] == 1 for name in guarded)


class _GitSourceLoader(SourceLoader):
    def __init__(self, source: bytes, origin: str):
        self._source = source
        self._origin = origin

    def get_filename(self, fullname: str) -> str:
        return self._origin

    def get_data(self, path: str) -> bytes:
        return self._source


@functools.cache
def _producer_namespace(git_sha: str) -> dict:
    """Load the committed driver Git blob at ``git_sha`` into an isolated module namespace."""
    for module in SUMMARY_MODULES:
        if module != DRIVER_MODULE and c503._git_blob(git_sha, c503._module_repo_path(module)) != _module_path(module).read_bytes():
            raise ValueError(f"summary module {module} differs from producer commit {git_sha}")
    blob = c503._git_blob(git_sha, SCRIPT_REPO_PATH)
    name = f"_c6_01_producer_{git_sha[:12]}"
    spec = spec_from_loader(name, _GitSourceLoader(blob, f"<git:{git_sha}:{SCRIPT_REPO_PATH}>"))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.__dict__


def _same_summary_definitions(committed: str, current: str, schema: str | None) -> bool:
    """True iff ``current`` recomputes a result of ``schema`` with the definitions of the ``committed`` driver.

    The caveats belong to the schema: the v1 list is ``caveats`` in a v1 driver and ``_caveats_v1`` here.
    Also rejects module-scope rebindings, augmented/annotated assignments, shadowed imports, globals or mutators.
    """
    if schema == HISTORICAL_SCHEMA_V1:
        pairs = [("caveats", "_caveats_v1")]
        tracked = set(SUMMARY_DEFINITIONS) | {"_caveats_v1", "caveats"}
    elif schema == SCHEMA_VERSION:
        pairs = [("caveats", "caveats"), ("_caveats_v1", "_caveats_v1")]
        tracked = set(SUMMARY_DEFINITIONS) | {"caveats", "_caveats_v1"}
    else:
        return False
    committed_tree, current_tree = ast.parse(committed), ast.parse(current)
    if not _module_ast_clean(committed_tree, current_tree, tracked):
        return False
    theirs, mine = _definitions(committed), _definitions(current)
    return all(a in theirs and b in mine and theirs[a] == mine[b]
               for a, b in [*((name, name) for name in SUMMARY_DEFINITIONS), *pairs])


def _matches_current_summary_code(git_sha: str) -> bool:
    """True iff the validating checkout recomputes a result with the code of its producer commit ``git_sha``.

    Every ``SUMMARY_MODULES`` file must have the committed bytes, with one exception. The driver may
    differ, because validator corrections land after a canonical run. Then every ``SUMMARY_DEFINITIONS``
    entry and the caveats of the producer's schema must have the committed AST.
    """
    for module in SUMMARY_MODULES:
        committed, current = c503._git_blob(git_sha, c503._module_repo_path(module)), _module_path(module).read_bytes()
        if committed != current and (module != DRIVER_MODULE or not _same_summary_definitions(
                committed.decode(), current.decode(), _schema_at_commit(git_sha))):
            return False
    return True


def run_config_hash(shakedown: bool, protocol: dict, code: str, git_sha, env: dict, data_sha256: dict,
                    schema: str = SCHEMA_VERSION) -> str:
    """``schema`` is the schema version of the run that measured."""
    return c503._digest({"schema": schema, "shakedown": shakedown, "protocol": protocol, "code_sha256": code,
                         "git_sha": git_sha, "env": {k: env.get(k) for k in c503.ENV_KEYS},
                         "data_sha256": data_sha256})


def _guarded_relative(path: Path, *, canonical_dest: bool = False) -> Path:
    """``path`` relative to ``ROOT`` (no symlink resolution, no ``..``); a canonical destination is the canonical file."""
    if ".." in path.parts:
        raise ValueError(f"Refusing to write {path}: parent traversal ('..') is not allowed")
    try:
        relative = c503._relative_under_root(path)
    except ValueError as exc:
        if "parent traversal" in str(exc):
            raise
        raise ValueError(f"The canonical result goes to {CANONICAL_OUTPUT}, not {path}" if canonical_dest
                         else f"{path} is a reserved result path outside {ROOT}") from None
    if canonical_dest and relative != CANONICAL_OUTPUT.relative_to(ROOT):
        raise ValueError(f"The canonical result goes to {CANONICAL_OUTPUT}, not {path}")
    if not relative.parts or relative.name in ("", ".", ".."):
        raise ValueError(f"Refusing to write invalid path {path}")
    return relative


def verify_canonical_output_path(path: Path) -> Path:
    """One destination rule for the CLI, the validator and the writer: the canonical file of this checkout,
    reached without ``..`` and without a symlink at any existing directory component or at the file."""
    path = Path(path)
    relative = _guarded_relative(path, canonical_dest=True)
    pre = c503._open_verified_parent_fd(path, relative, create=False)
    if pre is not None:
        os.close(pre[0])
    return relative


def write_json_atomic(path: Path, payload: dict, *, canonical_dest: bool = False) -> dict:
    """Lossless atomic write through one verified ``dir_fd`` and an ``O_EXCL`` same-directory temp file.

    A canonical or reserved destination must lie under ``ROOT`` without ``..`` or a symlink on the way;
    ``canonical_dest`` also binds it to ``CANONICAL_OUTPUT``. A failure before ``os.replace`` leaves the
    previous destination untouched and no staging file.
    """
    path = Path(path)
    guarded = canonical_dest or reserved(path)
    text = e2e._dumps_json(payload) + "\n"
    expected = json.loads(json.dumps(payload, allow_nan=False))
    if guarded:
        relative = _guarded_relative(path, canonical_dest=canonical_dest)
        dir_fd, filename = c503._open_verified_parent_fd(path, relative, create=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        dir_fd, filename = os.open(path.parent, c503._DIR_OPEN_FLAGS), path.name
    tmp_name = None
    try:
        fd, tmp_name = c503._create_temp_at(dir_fd, filename)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        rfd = os.open(tmp_name, os.O_RDONLY | c503._NOFOLLOW_FLAGS, dir_fd=dir_fd)
        with os.fdopen(rfd, "r", encoding="utf-8") as handle:
            reloaded = json.load(handle)
        if reloaded != expected:
            raise ValueError("JSON round trip is not lossless")
        if guarded:
            c503._check_not_symlink(filename, dir_fd, path, path)
        os.replace(tmp_name, filename, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        tmp_name = None
        os.fsync(dir_fd)
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name, dir_fd=dir_fd)
            except FileNotFoundError:
                pass
        os.close(dir_fd)
    return reloaded


def _read_access_log(log_path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()]


def record_test_access(git: dict, cfg_hash: str, frozen_sha: str, log_path: Path | None = None,
                       label: str = TEST_ACCESS_LOG_REPO_PATH) -> dict:
    """Append this run and its freeze digest to the test-access log before test.bin is mapped.

    Returns the record for ``data.test_access``, with the earlier lines of the log.
    """
    log_path = TEST_ACCESS_LOG if log_path is None else log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prior = _read_access_log(log_path) if log_path.exists() else []
    entry = {"git_sha": git["git_sha"], "run_config_hash": cfg_hash, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "selections_sha256": frozen_sha}
    with log_path.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {"log": label, "prior_openings": prior, "this_opening": entry}


def _artifact_dir(name: str) -> Path:
    """``artifacts/<name>`` here, else in the main checkout of this worktree."""
    local = ROOT / "artifacts" / name
    return local if local.exists() else q.default_data_dir().parent / name


def default_baseline_dir() -> Path:
    """``artifacts/c5_03_checkpoints`` here, else in the main checkout of this worktree."""
    return _artifact_dir("c5_03_checkpoints")


def shakedown_evidence_dir(work_dir: Path) -> Path:
    return Path(work_dir) / "run_evidence"


# ======================================================================================
# Driver
# ======================================================================================
def run(args) -> dict:
    protocol = copy.deepcopy(args.protocol)
    device = torch.device(protocol["device"])
    git = h.git_info()
    if args.canonical:
        require_canonical_host(device, git, args.output)  # before any GPU work and before any shard is read
    if device.type != "cuda" or not c4.hardware_w4a4_available(device):
        raise RuntimeError("C6-01 needs a CUDA device with native NVFP4 scaled_mm (sm_100+)")
    numerics = c503.set_numerics()
    rq.prime_cupti(device)  # before any other profiler session of this process
    log = (lambda *a, **k: None) if args.silent else (lambda *a, **k: print(*a, **k, flush=True))
    started, timestamp_start = time.perf_counter(), time.strftime("%Y-%m-%dT%H:%M:%S%z")
    env = q.environment(device)
    code = code_sha256()
    if args.shakedown:
        data_dir = args.work_dir / "shards"
        if data_dir.exists():
            shutil.rmtree(data_dir)
        rq.write_synthetic_shards(data_dir, **SYNTHETIC_SHARDS)
        shards = c502.Shards(data_dir, verify=False, expected=None)
    else:
        data_dir = args.data_dir
        shards = c502.Shards(data_dir, verify=True, expected=q.EXPECTED_SHA256)
    summary = shards.summary()
    synthetic = summary["source"].get("name") == rq.SYNTHETIC_SOURCE
    found_sha = {k: v["sha256"] for k, v in summary["splits"].items()}
    if args.shakedown and (not synthetic or set(found_sha.values()) & set(q.EXPECTED_SHA256.values())):
        raise RuntimeError("The shakedown reads generated synthetic shards only")
    if args.canonical and (synthetic or found_sha != q.EXPECTED_SHA256):
        raise RuntimeError("The canonical run reads the pinned FineWeb-Edu shards only")
    data = {**summary, "directory": str(data_dir), "sha256_verified": not args.shakedown, "synthetic": synthetic,
            "roles": dict(EXPECTED_ROLES), "test_access": None}
    cfg_hash = run_config_hash(args.shakedown, protocol, code, git["git_sha"], env, found_sha)
    meta = {**git, "command": _invocation(args.argv), "argv": list(args.argv), "environment": env, "numerics": numerics,
            "code_sha256": code, "code_modules": list(code_modules()), "timestamp_start": timestamp_start,
            "loadavg_start": list(os.getloadavg()), "gpu_telemetry": {"start": h.gpu_telemetry(device)}}
    log(f"[c6-01] git {git['git_sha']} dirty={git['git_dirty']} device={env.get('device')} run {cfg_hash} "
        f"canonical={args.canonical} shakedown={args.shakedown}")
    payload = {"schema_version": SCHEMA_VERSION, "task": TASK, "shakedown": args.shakedown, "canonical": args.canonical,
               "meta": meta, "run_config_hash": cfg_hash, "protocol": protocol, "data": data}
    payload["kernel_parity"] = c503.kernel_parity(device, args.shakedown)
    payload["cross_mix_parity"] = cross_mix_parity(device)
    log(f"kernel parity {payload['kernel_parity']['cases']} cases passed={payload['kernel_parity']['passed']}; "
        f"cross-mix parity {payload['cross_mix_parity']['cases']} cases passed={payload['cross_mix_parity']['passed']}")
    if not (payload["kernel_parity"]["passed"] and payload["cross_mix_parity"]["passed"]):
        raise RuntimeError("The NVFP4 quantizer is not bit-exact on this host")

    # The canonical log is for the real test.bin; a shakedown keeps the same record in its work directory.
    evidence_dir = TEST_ACCESS_LOG.parent if args.canonical else shakedown_evidence_dir(args.work_dir)

    def before_test(frozen_sha: str) -> None:
        data["test_access"] = record_test_access(
            git, cfg_hash, frozen_sha, evidence_dir / TEST_ACCESS_LOG_NAME,
            TEST_ACCESS_LOG_REPO_PATH if args.canonical else SHAKEDOWN_TEST_ACCESS_LOG)

    baseline_dir = None if args.shakedown else args.baseline_checkpoint_dir
    study = run_study(protocol, shards, device, args.checkpoint_dir, baseline_dir, log, before_test)
    payload.update(study)
    payload["c5_03_reference"] = None if args.shakedown else _trusted_c5_03_reference()
    payload["analysis"] = analyze(payload["units"], payload["selections"], protocol, payload["c5_03_reference"])
    payload["summary"] = summarize(payload)
    meta["splits_opened"] = list(shards.opened)
    meta["gpu_telemetry"]["end"] = h.gpu_telemetry(device)
    meta["loadavg_end"] = list(os.getloadavg())
    meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["total_seconds"] = time.perf_counter() - started
    payload["caveats"] = caveats(payload)
    payload = json.loads(json.dumps(payload, allow_nan=False))
    try:
        validate_results(payload, canonical=args.canonical, verify_external=True, checkpoint_dir=args.checkpoint_dir,
                         baseline_dir=baseline_dir, evidence_dir=evidence_dir)
    except Exception:
        # Keep the measurements of a long run: the rejected payload goes next to the output, never to it.
        write_json_atomic(rejected_path(args.output), payload)
        raise
    return payload


def _invocation(argv: list[str]) -> list[str]:
    """The effective command line (equal to sys.argv when the script runs from the command line)."""
    return [sys.executable, str(Path(__file__).resolve().relative_to(ROOT)), *argv]


def rejected_path(output: Path) -> Path:
    return Path(output).with_name(Path(output).name + ".rejected")


# ======================================================================================
# Validation
# ======================================================================================
TOP_KEYS = {"schema_version", "task", "shakedown", "canonical", "meta", "run_config_hash", "protocol", "data",
            "kernel_parity", "cross_mix_parity", "configs", "calibration", "splits_opened_before_freeze",
            "scoring_ledger", "selections", "selections_sha256", "units", "stage_seconds", "seconds",
            "c5_03_reference", "analysis", "summary", "caveats"}
META_KEYS = {"git_sha", "git_dirty", "command", "argv", "environment", "numerics", "code_sha256", "code_modules",
             "timestamp_start", "timestamp_end", "loadavg_start", "loadavg_end", "gpu_telemetry", "total_seconds",
             "splits_opened"}
DATA_KEYS = {"source", "encoding", "splits", "directory", "sha256_verified", "synthetic", "roles", "test_access"}
UNIT_KEYS = {"part", "seed", "arm", "training", "methods", "emulated_method", "baseline_val", "baseline_test",
             "bf16_fp4_gemms_per_forward", "bf16_cupti", "geometry", "activation_statistics", "butterfly_fits",
             "awq_selection", "grid", "seconds"}
SELECTION_KEYS = {"qat_wide_family", "best_wide_ptq_method", "best_ptq_method", "arm_best_method", "transfer_arm",
                  "transfer_family", "transfer_loss"}
TRAINED_KEYS = {"source", "part", "arm", "seed", "optimizer", "lr", "auxiliary_lr", "weight_decay", "min_lr_ratio",
                "steps", "first_step", "step_tokens", "tokens", "train_epochs", "init_fingerprint",
                "stream_fingerprint", "train_loss", "last_train_loss", "train_curve", "train_seconds", "step_seconds",
                "qat", "branch", "checkpoint_file", "checkpoint_sha256", "checkpoint_file_sha256"}
PRESERVED_KEYS = {"source", "part", "arm", "seed", "checkpoint_file", "checkpoint_sha256", "checkpoint_file_sha256"}
STAGES = ("stage_a", "stage_b", "stage_c", "test_scoring")
# A v2 run adds the freeze digest to its test-access line, the AWQ selections to the frozen selections and the
# optimizer snapshot digest to a branch.
OPENING_KEYS = {HISTORICAL_SCHEMA_V1: {"git_sha", "run_config_hash", "timestamp"},
                SCHEMA_VERSION: {"git_sha", "run_config_hash", "timestamp", "selections_sha256"}}
BRANCH_KEYS = {HISTORICAL_SCHEMA_V1: {"from_arm", "from_step", "snapshot_sha256"},
               SCHEMA_VERSION: {"from_arm", "from_step", "snapshot_sha256", "optimizer_snapshot_sha256"}}
AWQ_SELECTION_KEYS = {"method", "alphas", "val_nll_hw_dynamic", "selected_alpha", "selected_on", "seconds"}
ROW_COUNTERS = ("fp4_gemms_per_forward", "nvfp4_modules", "bf16_block_linears", "emulated_modules",
                "cross_mix_modules")
# Range of a per-layer metric: (lower bound, the bound is exclusive), and an upper bound where one exists.
METRIC_LOW = {"weight_stable_rank": (1.0, False), "signal_alignment": (0.0, True), "excess_kurtosis": (-2.0, False),
              "max_over_rms": (1.0, False), "group16_max_over_rms": (1.0, False),
              "group16_outlier_share": (0.0, False), "nvfp4_relative_sq_error": (0.0, False)}
METRIC_HIGH = {"group16_outlier_share": 1.0, "group16_max_over_rms": 4.0}


def _is_count(x) -> bool:
    """A kernel, dispatch or module count: a JSON integer >= 0 (``True`` is not a count)."""
    return type(x) is int and x >= 0


def _parsed_command(command, label: str) -> argparse.Namespace:
    """``command`` as a runnable invocation of this script, parsed by its own CLI (no abbreviations)."""
    _require(isinstance(command, list) and len(command) >= 2 and all(isinstance(x, str) and x for x in command)
             and c502._is_python_executable(command[0]), f"{label} must be a Python invocation")
    _require(command[1] == SCRIPT_REPO_PATH, f"{label} is not an invocation of {SCRIPT_REPO_PATH}")
    err = io.StringIO()
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            return parse_args(command[2:])
    except SystemExit as exc:
        raise ValueError(f"{label} is not a valid runnable invocation: {err.getvalue().strip()}") from exc


def _validate_meta(payload: dict, canonical: bool) -> None:
    meta = payload["meta"]
    _require(isinstance(meta, dict) and set(meta) == META_KEYS, "meta keys differ from the schema")
    _require(meta["git_sha"] is None or _is_hex(meta["git_sha"], 40), "meta.git_sha must be 40-hex or None")
    if meta["git_sha"] is not None:
        c503._require_git_commit(meta["git_sha"], "meta.git_sha")
    _require(meta["git_dirty"] is None or type(meta["git_dirty"]) is bool, "meta.git_dirty must be a bool")
    _require(_same(meta["numerics"], c503.NUMERICS), "full-precision numerics are required")
    command, argv = meta["command"], meta["argv"]
    _require(isinstance(argv, list) and isinstance(command, list) and argv == command[2:],
             "meta.argv must be the arguments of meta.command")
    parsed = _parsed_command(command, "meta.command")
    _require(parsed.canonical == payload["canonical"] and parsed.shakedown == payload["shakedown"]
             and _same(parsed.protocol, payload["protocol"]),
             "meta.argv does not select the recorded protocol and mode")
    modules = meta["code_modules"]
    _require(isinstance(modules, list) and all(isinstance(m, str) and m.startswith("research.") for m in modules)
             and len(set(modules)) == len(modules) and DRIVER_MODULE in modules
             and set(SUMMARY_MODULES) <= set(modules),
             "meta.code_modules must list the driver and its research.* imports")
    _require(_is_hex(meta["code_sha256"], 64), "meta.code_sha256 must be 64-hex")
    env = meta["environment"]
    _require(isinstance(env, dict) and all(k in env for k in c503.ENV_KEYS), "meta.environment incomplete")
    _require(_nonneg(meta["total_seconds"]) and _nonneg(payload["seconds"])
             and meta["total_seconds"] >= payload["seconds"] - 1e-6, "meta durations invalid")
    start = c502._timestamp(meta["timestamp_start"], "meta.timestamp_start")
    end = c502._timestamp(meta["timestamp_end"], "meta.timestamp_end")
    _require(end >= start and abs((end - start).total_seconds() - meta["total_seconds"]) <= 5.0,
             "meta timestamps are inconsistent with meta.total_seconds")
    if meta["git_sha"] is not None:
        _require(start >= c503._commit_timestamp(meta["git_sha"]),
                 "meta.timestamp_start is before the producer commit meta.git_sha")
    for field in ("loadavg_start", "loadavg_end"):
        _require(isinstance(meta[field], list) and len(meta[field]) == 3 and all(_nonneg(x) for x in meta[field]),
                 f"meta.{field} invalid")
    _require(meta["splits_opened"] == ["train", "val", "test"], "meta.splits_opened must be train, val, then test")
    if canonical:
        verify_canonical_output_path(parsed.output)
        _require(_is_hex(meta["git_sha"], 40) and meta["git_dirty"] is False,
                 "a canonical result needs a 40-hex git_sha and a clean tree (git_dirty False)")
        _require(tuple(env["capability"]) >= (12, 0), "a canonical result needs an sm_120+ device")
        _require(modules == list(_code_modules_at_commit(meta["git_sha"])),
                 "meta.code_modules differ from the import closure at meta.git_sha")
        _require(meta["code_sha256"] == c503._code_sha256_at_commit(meta["git_sha"], tuple(modules)),
                 "meta.code_sha256 differs from the committed git blobs")
        _require(_schema_at_commit(meta["git_sha"]) == payload["schema_version"],
                 "meta.git_sha declares another schema than the payload")
        _require(_matches_current_summary_code(meta["git_sha"]),
                 "meta.git_sha summary code differs from the validating checkout")
    else:
        _require(modules == list(code_modules()), "meta.code_modules differ from the import closure of this driver")
        _require(meta["code_sha256"] == code_sha256(), "meta.code_sha256 differs from this checkout")
    data_sha = {k: v["sha256"] for k, v in payload["data"]["splits"].items()}
    _require(payload["run_config_hash"] == run_config_hash(payload["shakedown"], payload["protocol"],
                                                           meta["code_sha256"], meta["git_sha"], env, data_sha,
                                                           payload["schema_version"]),
             "run_config_hash differs")


def _opening_time(entry, where: str, canonical: bool):
    """One line of the test-access log (written by a v1 or a v2 run). Returns its timestamp."""
    _require(isinstance(entry, dict) and any(set(entry) == keys for keys in OPENING_KEYS.values()),
             f"{where}: keys differ from a test-access log entry")
    _require((_is_hex(entry["git_sha"], 40) or (entry["git_sha"] is None and not canonical))
             and _is_hex(entry["run_config_hash"], 16)
             and ("selections_sha256" not in entry or _is_hex(entry["selections_sha256"], 64)),
             f"{where}: git_sha, run_config_hash or selections_sha256 invalid")
    return c502._timestamp(entry["timestamp"], f"{where}.timestamp")


def _validate_test_access(payload: dict, canonical: bool) -> None:
    """``data.test_access``: the line this run appended before it mapped test.bin, and the earlier lines.

    The opening names this run, carries the freeze digest (v2), lies inside the run interval, after stages
    A-C and before the test scoring. ``verify_external_evidence`` compares the record with the log file.
    """
    access, meta = payload["data"]["test_access"], payload["meta"]
    _require(isinstance(access, dict) and set(access) == {"log", "prior_openings", "this_opening"}
             and access["log"] == (TEST_ACCESS_LOG_REPO_PATH if canonical else SHAKEDOWN_TEST_ACCESS_LOG)
             and isinstance(access["prior_openings"], list) and isinstance(access["this_opening"], dict),
             "data.test_access must record this run and the earlier openings of test.bin")
    this = access["this_opening"]
    _require(set(this) == OPENING_KEYS[payload["schema_version"]],
             "data.test_access.this_opening keys differ from the schema")
    opened = _opening_time(this, "data.test_access.this_opening", canonical)
    _require(this["git_sha"] == meta["git_sha"] and this["run_config_hash"] == payload["run_config_hash"],
             "data.test_access.this_opening must name this run (git_sha, run_config_hash)")
    if "selections_sha256" in this:
        _require(this["selections_sha256"] == payload["selections_sha256"],
                 "data.test_access.this_opening.selections_sha256 differs from the freeze digest selections_sha256")
    start = c502._timestamp(meta["timestamp_start"], "meta.timestamp_start")
    end = c502._timestamp(meta["timestamp_end"], "meta.timestamp_end")
    _require(start - timedelta(seconds=1) <= opened <= end + timedelta(seconds=1),
             "data.test_access.this_opening.timestamp is outside the run interval "
             "(meta.timestamp_start .. meta.timestamp_end)")
    stages = payload["stage_seconds"]  # timestamps have a resolution of one second
    _require((opened - start).total_seconds() >= sum(stages[k] for k in STAGES[:3]) - 2.0
             and (end - opened).total_seconds() >= stages["test_scoring"] - 2.0,
             "data.test_access.this_opening.timestamp must follow stages A-C and precede the test scoring")
    for i, entry in enumerate(access["prior_openings"]):
        where = f"data.test_access.prior_openings[{i}]"
        _require(_opening_time(entry, where, canonical) <= opened, f"{where} is later than this opening")


def _validate_data(payload: dict, canonical: bool) -> None:
    data = payload["data"]
    _require(isinstance(data, dict) and set(data) == DATA_KEYS, "data keys differ from the schema")
    _require(_same(data["roles"], EXPECTED_ROLES),
             "data.roles differs from the split roles of the protocol (train.bin: training, calibration and static "
             "scales; val.bin: every selection; test.bin: evaluation only)")
    found = {k: v["sha256"] for k, v in data["splits"].items()}
    _require(set(found) == {"train", "val", "test"} and all(_is_hex(v, 64) for v in found.values()),
             "data.splits must hold the 64-hex SHA-256 of train, val and test")
    real = set(found.values()) & set(q.EXPECTED_SHA256.values())
    if canonical:
        _require(data["sha256_verified"] is True and data["synthetic"] is False and found == q.EXPECTED_SHA256,
                 "canonical shards must be SHA-256 verified against the pinned FineWeb-Edu hashes")
        _require({k: v["tokens"] for k, v in data["splits"].items()} == c502.CANONICAL_DATA["tokens"],
                 "canonical split token counts differ")
    else:
        _require(data["synthetic"] is True and data["source"].get("name") == rq.SYNTHETIC_SOURCE and not real
                 and data["sha256_verified"] is False,
                 "a shakedown reads generated synthetic shards only (no pinned real-text hash)")
    _validate_test_access(payload, canonical)


@functools.cache
def _expected_stream_fingerprint(train_tokens: int, preset: str, seed: int) -> str:
    """``h.stream_fingerprint`` of a seed. The batch starts depend on the length of train.bin, not on its tokens."""
    return h.stream_fingerprint(range(train_tokens), PRESETS[preset], seed)


@functools.cache
def _expected_init_fingerprint(preset: str, seed: int) -> str:
    """``h.state_fingerprint`` of the initial weights of ``preset`` and ``seed`` (CPU, RNG-isolated)."""
    with torch.random.fork_rng(devices=[]):
        return h.state_fingerprint(h.initial_state(PRESETS[preset], seed))


def _validate_training(u: dict, payload: dict, where: str) -> None:
    protocol, sel = payload["protocol"], payload["selections"]
    part, arm, seed, tr = u["part"], u["arm"], u["seed"], u["training"]
    spec = protocol["part_ab" if part == "ab" else "part_c"]
    config = PRESETS[spec["preset"]]
    train_tokens = payload["data"]["splits"]["train"]["tokens"]
    _require(isinstance(tr, dict) and (tr.get("part"), tr.get("arm"), tr.get("seed")) == (part, arm, seed)
             and _is_hex(tr.get("checkpoint_sha256"), 64) and _is_hex(tr.get("checkpoint_file_sha256"), 64),
             f"{where}: training record identity or digests invalid")
    preserved = part == "c" and arm == BASELINE and spec["baseline"] == "c5_03_preserved"
    if preserved:
        _require(set(tr) == PRESERVED_KEYS and _same(tr, preserved_baseline_record(seed)),
                 f"{where}: the baseline must be the preserved C5-03 checkpoint with its pinned digests")
        return
    _require(set(tr) == TRAINED_KEYS and tr["source"] == "trained_in_run", f"{where}: training record keys differ")
    qat = arm_qat(part, arm, sel)
    if part == "ab":
        total, first = spec["steps"], spec["cooldown_from"] if qat and qat["schedule"] == "cooldown" else 0
        branch_arm = MUON
    else:
        base = spec["baseline_steps"]
        total, first = (base, 0) if arm == BASELINE else (base + spec["cooldown_steps"], base)
        branch_arm = BASELINE
    _require(tr["optimizer"] == spec["optimizer"] and tr["lr"] == spec["lr"] and tr["auxiliary_lr"] == spec["auxiliary_lr"]
             and tr["weight_decay"] == protocol["weight_decay"] and tr["min_lr_ratio"] == protocol["min_lr_ratio"]
             and tr["steps"] == total and tr["first_step"] == first and tr["step_tokens"] == config.step_tokens
             and tr["tokens"] == total * config.step_tokens and tr["checkpoint_file"] == checkpoint_name(part, seed, arm),
             f"{where}: training record differs from the protocol")
    _assert_close(tr["train_epochs"], tr["tokens"] / train_tokens, f"{where}.train_epochs")
    # Paired controls: the arms of a seed start from one initialization and read one batch stream.
    _require(_is_hex(tr["init_fingerprint"], 16) if part == "ab" or arm == BASELINE else tr["init_fingerprint"] is None,
             f"{where}: init_fingerprint must be 16-hex (a run from the seed's initial state) or None (a Part C "
             "continuation)")
    _require(tr["stream_fingerprint"] == _expected_stream_fingerprint(train_tokens, spec["preset"], seed),
             f"{where}: stream_fingerprint differs from the batch stream of seed {seed}")
    seconds = tr["step_seconds"]
    _require(isinstance(seconds, list) and len(seconds) == total - first and all(_nonneg(s) and s > 0 for s in seconds),
             f"{where}: one positive wall time per optimizer step is required")
    _assert_close(tr["train_seconds"], math.fsum(seconds), f"{where}.train_seconds")
    _require(tr["train_seconds"] >= math.fsum(seconds) - 1e-3, f"{where}: train_seconds is below the step times")
    # The curve holds the first step, every tenth step and the last step, with a positive finite loss each.
    curve, n = tr["train_curve"], total - first
    _require(isinstance(curve, list) and curve and all(
        isinstance(pt, list) and len(pt) == 2 and type(pt[0]) is int and first + 1 <= pt[0] <= total
        and _is_number(pt[1]) and pt[1] > 0 for pt in curve),
        f"{where}: train_curve must hold [step, loss] pairs with steps in {first + 1}..{total} and a positive loss")
    steps = [pt[0] for pt in curve]
    _require(steps[0] == first + 1 and steps[-1] == total and all(a < b for a, b in zip(steps, steps[1:]))
             and steps == [first + i + 1 for i in range(n) if i == 0 or (i + 1) % 10 == 0 or i + 1 == n],
             f"{where}: train_curve steps must increase from {first + 1} to {total} (first, every tenth, last)")
    _require(curve[-1][1] == tr["last_train_loss"] and _is_number(tr["train_loss"]) and tr["train_loss"] > 0,
             f"{where}: training curve record invalid")
    if qat is None:
        _require(tr["qat"] is None, f"{where}: a BF16-trained arm has no QAT record")
    else:
        want = {**qat, "bf16_weight": protocol["dual_bf16_weight"] if qat["loss"] == "dual" else None,
                "fake_quant_steps": total - first}
        _require(_same(tr["qat"], want), f"{where}: QAT record differs from the arm definition and the selections")
    if first == 0:
        _require(tr["branch"] is None, f"{where}: a full run has no branch")
    else:
        origin = next(x for x in payload["units"] if (x["part"], x["seed"], x["arm"]) == (part, seed, branch_arm))
        branch = tr["branch"]
        _require(isinstance(branch, dict) and set(branch) == BRANCH_KEYS[payload["schema_version"]]
                 and branch["from_arm"] == branch_arm and type(branch["from_step"]) is int
                 and branch["from_step"] == first and _is_hex(branch["snapshot_sha256"], 64)
                 and (part == "ab" or branch["snapshot_sha256"] == origin["training"]["checkpoint_sha256"]),
                 f"{where}: a cooldown must branch from the {branch_arm} run of its seed")
        if "optimizer_snapshot_sha256" in branch:  # Part C starts a fresh optimizer: no snapshot exists
            _require(_is_hex(branch["optimizer_snapshot_sha256"], 64) if part == "ab"
                     else branch["optimizer_snapshot_sha256"] is None,
                     f"{where}: branch.optimizer_snapshot_sha256 must be 64-hex (Part B) or None (Part C)")


def _validate_siblings(payload: dict) -> None:
    """The paired controls of a seed: one initialization, one batch stream and one branch state.

    Every Part A/B arm starts from the same initial weights. Every trained arm of a part reads the same
    batch stream. The ``COOLDOWN_ARMS`` restore one step-``cooldown_from`` snapshot (weights and, in v2,
    optimizer), and the two Part C continuations load one baseline checkpoint.
    """
    units, cooldown_from = payload["units"], payload["protocol"]["part_ab"]["cooldown_from"]
    ab_preset, pc = payload["protocol"]["part_ab"]["preset"], payload["protocol"]["part_c"]
    for seed in payload["protocol"]["seeds"]:
        group = {(u["part"], u["arm"]): u["training"] for u in units if u["seed"] == seed}
        _require(len({group[("ab", arm)]["init_fingerprint"] for arm in ARMS_AB}) == 1,
                 f"seed {seed}: the Part A/B arms do not share one init_fingerprint")
        _require(group[("ab", MUON)]["init_fingerprint"] == _expected_init_fingerprint(ab_preset, seed),
                 f"seed {seed}: init_fingerprint differs from the initial weights of {ab_preset}")
        if pc["baseline"] != "c5_03_preserved":
            _require(group[("c", BASELINE)]["init_fingerprint"] == _expected_init_fingerprint(pc["preset"], seed),
                     f"seed {seed}: Part C init_fingerprint differs from the initial weights of {pc['preset']}")
        for part in PARTS:
            streams = {tr["stream_fingerprint"] for (p, _), tr in group.items()
                       if p == part and tr["source"] == "trained_in_run"}
            _require(len(streams) == 1, f"seed {seed}: the part {part} arms do not share one stream_fingerprint")
        _require(all((group[("ab", arm)]["branch"] is not None) == (arm in COOLDOWN_ARMS) for arm in ARMS_AB),
                 f"seed {seed}: the cooldown arms, and no other Part A/B arm, have a branch")
        cooldowns = [group[("ab", arm)]["branch"] for arm in COOLDOWN_ARMS]
        _require(all(_same(branch, cooldowns[0]) for branch in cooldowns),
                 f"seed {seed}: the cooldown arms do not branch from one step-{cooldown_from} snapshot (sibling "
                 "branch records differ)")
        _require(isinstance(group[("c", CONTINUE)]["branch"], dict)
                 and _same(group[("c", CONTINUE)]["branch"], group[("c", QAT_C)]["branch"]),
                 f"seed {seed}: {CONTINUE} and {QAT_C} do not branch from one baseline checkpoint")


def _validate_layer_summary(summary, metric: str, where: str, *, upper_bound: float | None = None) -> None:
    """A ``_layer_summary``: finite numbers, ordered extrema, per-kind means inside them, the range of the metric."""
    _require(isinstance(summary, dict) and set(summary) == {"all", "by_kind"} and isinstance(summary["all"], dict)
             and set(summary["all"]) == {"mean", "median", "min", "max"} and isinstance(summary["by_kind"], dict)
             and list(summary["by_kind"]) == list(q.LINEAR_KINDS)
             and all(_is_number(v) for v in (*summary["all"].values(), *summary["by_kind"].values())),
             f"{where}: summary keys or numbers invalid")
    a, kinds = summary["all"], list(summary["by_kind"].values())
    eps = 1e-9 * max(1.0, abs(a["min"]), abs(a["max"]))
    inside = (a["mean"], a["median"], *kinds)
    _require(a["min"] <= a["max"] and all(a["min"] - eps <= v <= a["max"] + eps for v in inside)
             and abs(a["mean"] - math.fsum(kinds) / len(kinds)) <= eps,  # every kind has the same number of layers
             f"{where}: summary bounds violated (min <= mean, median and per-kind means <= max)")
    (low, exclusive), high = METRIC_LOW[metric], min(METRIC_HIGH.get(metric, math.inf),
                                                     math.inf if upper_bound is None else upper_bound)
    tol = 1e-6 * max(1.0, abs(low), 0.0 if math.isinf(high) else abs(high))
    _require((a["min"] > low if exclusive else a["min"] >= low - tol) and a["max"] <= high + tol,
             f"{where}: values outside the range of {metric}")


def _validate_unit(u: dict, payload: dict, coverage: dict) -> tuple[int, int]:
    """One checkpoint. Returns its (val, test) scoring counts."""
    protocol, sel = payload["protocol"], payload["selections"]
    part, arm = u["part"], u["arm"]
    where = f"unit {part}/{u['seed']}/{arm}"
    _require(set(u) == UNIT_KEYS, f"{where}: unit keys differ from the schema")
    config = PRESETS[protocol["part_ab" if part == "ab" else "part_c"]["preset"]]
    n_layers = config.layers * len(q.LINEAR_KINDS)
    _validate_training(u, payload, where)
    methods = unit_methods(part, arm, sel)
    _require(u["methods"] == methods and u["emulated_method"] == emulated_method(part, arm, sel),
             f"{where}: methods differ from the arm definition and the selections")
    for split in SPLITS:
        base = u[f"baseline_{split}"]
        _require(all(base[k] == v for k, v in coverage[part][split].items()), f"{where}: {split} coverage differs")
        _assert_close(base["ppl"], math.exp(base["nll"]), f"{where}.baseline_{split}.ppl")
    _require(_is_count(u["bf16_fp4_gemms_per_forward"]) and u["bf16_fp4_gemms_per_forward"] == 0
             and isinstance(u["bf16_cupti"], dict) and set(u["bf16_cupti"]) == set(rq.CUPTI_KEYS)
             and all(_is_count(v) for v in u["bf16_cupti"].values()) and u["bf16_cupti"]["fp4_gemm_kernels"] == 0
             and u["bf16_cupti"]["nvfp4_quant_kernels"] == 0 and u["bf16_cupti"]["nvfp4_amax_kernels"] == 0
             and _is_int(u["bf16_cupti"]["non_fp4_gemm_kernels"], n_layers),
             f"{where}: the BF16 model must run no FP4 GEMM and no NVFP4 quantizer kernel (CUPTI)")
    grid = u["grid"]
    _require([(r["method"], r["mode"]) for r in grid] == expected_grid(methods, u["emulated_method"]),
             f"{where}: grid rows differ")
    for r in grid:
        rw = f"{where} {r['method']}/{r['mode']}"
        _require(set(r) == ROW_KEYS, f"{rw}: row keys differ")
        family, rounding, awq = rq.parse_method(r["method"])
        _require((r["rotation"], r["rounding"], r["awq"], r["wide"]) == (family, rounding, awq, family in rq.WIDE_FAMILIES),
                 f"{rw}: method fields")
        _require((type(r["awq_alpha"]) is float and r["awq_alpha"] in protocol["awq_alphas"]) if awq
                 else r["awq_alpha"] is None, f"{rw}: awq_alpha must be a protocol alpha (AWQ rows only)")
        for split in SPLITS:
            base = u[f"baseline_{split}"]
            _require(r[f"{split}_tokens"] == base["tokens"] and _nonneg(r[f"seconds_{split}"]),
                     f"{rw}: {split} tokens or seconds invalid")
            _assert_close(r[f"{split}_ppl"], math.exp(r[f"{split}_nll"]), f"{rw}.{split}_ppl")
            _assert_close(r[f"{split}_delta_nll"], r[f"{split}_nll"] - base["nll"], f"{rw}.{split}_delta_nll")
            _assert_close(r[f"{split}_relative_delta_ppl_pct"], 100.0 * (r[f"{split}_ppl"] / base["ppl"] - 1.0),
                          f"{rw}.{split}_relative_delta_ppl_pct")
        cross = n_layers if family in rq.WIDE_FAMILIES else 0
        cupti, decode = r["cupti"], (r["hw_weights_checked_matrices"], r["hw_weights_decode_exact_matrices"])
        if r["mode"] == "w4a16_fakequant":  # the BF16 model with decoded weights: no module is swapped or profiled
            _require(all(r[k] is None for k in ROW_COUNTERS) and cupti is None and all(_is_count(v) for v in decode),
                     f"{rw}: a W4A16 row has int32 decode counts and no hardware counter")
        else:
            _require(all(_is_count(r[k]) for k in ROW_COUNTERS) and isinstance(cupti, dict)
                     and set(cupti) == set(rq.CUPTI_KEYS) and all(_is_count(v) for v in cupti.values())
                     and decode == (None, None),
                     f"{rw}: the module, dispatch and CUPTI kernel counts must be integers >= 0")
        if r["mode"] in HW_MODES:
            _require(r["fp4_gemms_per_forward"] == n_layers and r["nvfp4_modules"] == n_layers
                     and r["bf16_block_linears"] == 0 and r["emulated_modules"] == 0
                     and r["cross_mix_modules"] == cross,
                     f"{rw}: every block linear must be a native FP4 GEMM ({n_layers} per forward, no fallback)")
            _require(isinstance(cupti, dict) and set(cupti) == set(rq.CUPTI_KEYS)
                     and cupti["fp4_gemm_kernels"] == n_layers and cupti["nvfp4_quant_kernels"] == n_layers
                     and cupti["nvfp4_amax_kernels"] == (n_layers if r["mode"] == DYNAMIC else 0)
                     and _is_int(cupti["non_fp4_gemm_kernels"])
                     and (cupti["non_fp4_gemm_kernels"] >= u["bf16_cupti"]["non_fp4_gemm_kernels"]  # + the cross mix
                          if family in rq.WIDE_FAMILIES else
                          cupti["non_fp4_gemm_kernels"] == u["bf16_cupti"]["non_fp4_gemm_kernels"] - n_layers),
                     f"{rw}: CUPTI must show {n_layers} FP4 GEMM and quantizer kernels and no BF16 GEMM fallback")
        elif r["mode"] == EMULATED_MODE:
            # FP32 GEMMs on the operands of the fused NVFP4 quantizer (dynamic scale): the quantizer and amax
            # kernels run once per layer and no FP4 GEMM runs. An FP32 GEMM replaces every BF16 GEMM with one
            # kernel or more (the count depends on the matrix size), and a wide row adds the cross mix.
            _require(r["fp4_gemms_per_forward"] == 0 and r["emulated_modules"] == n_layers
                     and r["bf16_block_linears"] == 0 and r["nvfp4_modules"] == 0 and r["cross_mix_modules"] == cross
                     and cupti["fp4_gemm_kernels"] == 0 and cupti["nvfp4_quant_kernels"] == n_layers
                     and cupti["nvfp4_amax_kernels"] == n_layers
                     and cupti["non_fp4_gemm_kernels"] >= u["bf16_cupti"]["non_fp4_gemm_kernels"],
                     f"{rw}: the emulation must run no FP4 GEMM, {n_layers} NVFP4 quantizer and amax kernels and "
                     "at least the GEMM kernels of the BF16 model")
        else:
            _require(r["hw_weights_checked_matrices"] == n_layers
                     and r["hw_weights_decode_exact_matrices"] == n_layers
                     and _is_hex(r["effective_weights_sha256"], 64)
                     and r["effective_weights_sha256_test"] == r["effective_weights_sha256"],
                     f"{rw}: weights must decode exactly ({n_layers}/{n_layers}, int32) and be rebuilt identically")
        static = r["mode"] == "w4a4_hw_static"
        _require((isinstance(r["static_global_scale_range"], list) and len(r["static_global_scale_range"]) == 2
                  and 0 < r["static_global_scale_range"][0] <= r["static_global_scale_range"][1])
                 if static else r["static_global_scale_range"] is None, f"{rw}: static scale range")
    rows = {(r["method"], r["mode"]): r for r in grid}
    awq_methods = [m for m in methods if rq.parse_method(m)[2]]
    _require([a["method"] for a in u["awq_selection"]] == awq_methods, f"{where}: one AWQ selection per AWQ method")
    for a in u["awq_selection"]:
        _require(set(a) == AWQ_SELECTION_KEYS and _nonneg(a["seconds"]) and isinstance(a["val_nll_hw_dynamic"], list)
                 and all(_is_number(v) and v > 0 for v in a["val_nll_hw_dynamic"]),
                 f"{where}: AWQ selection record invalid")
        _require(a["alphas"] == protocol["awq_alphas"] and a["selected_on"] == "val.bin, w4a4_hw_dynamic"
                 and len(a["val_nll_hw_dynamic"]) == len(a["alphas"])
                 and a["selected_alpha"] == c502.select_alpha(a["alphas"], a["val_nll_hw_dynamic"]),
                 f"{where}: the selected AWQ alpha is not the val.bin minimum")
        r = rows[(a["method"], DYNAMIC)]
        _require(r["awq_alpha"] == a["selected_alpha"]
                 and r["val_nll"] == a["val_nll_hw_dynamic"][a["alphas"].index(a["selected_alpha"])]
                 and all(rows[(a["method"], m)]["awq_alpha"] == a["selected_alpha"] for m in MODES),
                 f"{where}: the AWQ rows are not the selected candidate")
    fitted = {m for m in methods if "butterfly" in m}
    _require(set(u["butterfly_fits"]) == fitted, f"{where}: butterfly fits differ from the fitted methods")
    for key, kinds in u["butterfly_fits"].items():
        _require(list(kinds) == list(q.LINEAR_KINDS), f"{where}: one butterfly fit per projection kind")
        for kind, fit in kinds.items():
            _require(set(fit) == c503.BUTTERFLY_FIT_KEYS and fit["steps"] == protocol["butterfly_steps"]
                     and fit["layers"] == config.layers and 0 <= fit["weight_sq_error_fitted"]
                     <= fit["weight_sq_error_hadamard_start"], f"{where}.butterfly_fits[{key}].{kind}: fit record")
    _require(isinstance(u["geometry"], dict) and list(u["geometry"]) == ["weight_stable_rank", "signal_alignment"],
             f"{where}: geometry metrics differ")
    max_stable_rank = float(min(config.width, config.hidden))
    for metric, summary in u["geometry"].items():
        _validate_layer_summary(summary, metric, f"{where}: geometry.{metric}",
                                upper_bound=max_stable_rank if metric == "weight_stable_rank" else None)
    if part == "ab" and arm == MUON:
        stats = u["activation_statistics"]
        _require(isinstance(stats, dict) and list(stats) == list(PROBE_FAMILIES)
                 and all(isinstance(v, dict) and list(v) == list(rq.ACTIVATION_METRICS) for v in stats.values()),
                 f"{where}: activation statistics of every rotation family are required")
        for family, metrics in stats.items():
            for metric, summary in metrics.items():
                _validate_layer_summary(summary, metric, f"{where}: activation_statistics.{family}.{metric}")
    else:
        _require(u["activation_statistics"] is None, f"{where}: only the probe checkpoints have activation statistics")
    sec = u["seconds"]
    _require(isinstance(sec, dict) and set(sec) == {"phase1", "phase2"} and all(_nonneg(v) for v in sec.values()),
             f"{where}: unit.seconds invalid")
    _require(sec["phase1"] >= math.fsum(r["seconds_val"] for r in grid)
             + math.fsum(a["seconds"] for a in u["awq_selection"]) - 1e-3,
             f"{where}: seconds.phase1 is below the recorded validation and AWQ times")
    _require(sec["phase2"] >= math.fsum(r["seconds_test"] for r in grid) - 1e-3,
             f"{where}: seconds.phase2 is below the recorded test scoring times")
    plain = sum(not rq.parse_method(m)[2] for m in methods)
    val = 1 + len(MODES) * plain + 1 + len(awq_methods) * (len(protocol["awq_alphas"]) + len(MODES) - 1)
    return val, 1 + len(MODES) * len(methods) + 1


def _validate_c5_03_rows(units: list[dict], reference: dict) -> int:
    """The baseline rows against the committed C5-03 result: exact, 0.0 nats on every comparator row.

    The checkpoints, the calibration draw, the methods and the alpha are those of C5-03, and the scoring
    is deterministic, so a re-measurement that differs is not the published comparator. Returns the row
    count: seeds x splits x (BF16 + two methods x three modes) = 42 for the canonical study.
    """
    compared = 0
    for u in _units(units, "c", BASELINE):
        theirs = reference["seeds"][str(u["seed"])]
        _require(_row(u, C5_BEST, DYNAMIC)["awq_alpha"] == theirs["awq_alpha"],
                 f"C5-03 comparator seed {u['seed']}: the AWQ alpha differs from the committed C5-03 result")
        for split in SPLITS:
            pairs = [("bf16", u[f"baseline_{split}"]["nll"], theirs["bf16"][split])]
            pairs += [(f"{method}/{mode}", _row(u, method, mode)[f"{split}_nll"], published[split])
                      for method in (C5_HADAMARD, C5_BEST) for mode, published in theirs["methods"][method].items()]
            for name, mine, published in pairs:
                _require(_is_number(mine) and mine - published == 0.0,
                         f"C5-03 comparator seed {u['seed']} {name} {split}: NLL {mine!r} differs from the committed "
                         f"C5-03 value {published!r} (an exact reproduction is required)")
            compared += len(pairs)
    _require(compared == N_SEEDS * len(SPLITS) * (1 + 2 * len(MODES)), "C5-03 comparator rows are incomplete")
    return compared


def _validate_study(payload: dict, canonical: bool) -> None:
    protocol, units, sel = payload["protocol"], payload["units"], payload["selections"]
    ab, pc = protocol["part_ab"], protocol["part_c"]
    _require(_same(payload["configs"], {"ab": asdict(PRESETS[ab["preset"]]), "c": asdict(PRESETS[pc["preset"]])}),
             "configs differ from the presets of the protocol")
    seeds = protocol["seeds"]
    expected = ([("ab", s, MUON) for s in seeds] + [("ab", s, a) for s in seeds for a in QAT_ARMS]
                + [("c", s, a) for s in seeds for a in ARMS_C])
    _require(isinstance(units, list) and [(u["part"], u["seed"], u["arm"]) for u in units] == expected,
             "units differ from the study order (stage A, stage B, stage C)")
    v1 = payload["schema_version"] == HISTORICAL_SCHEMA_V1
    _require(isinstance(sel, dict) and set(sel) == SELECTION_KEYS | (set() if v1 else {"awq_selections"})
             and sel["qat_wide_family"] in WIDE_QAT_FAMILIES
             and sel["transfer_arm"] in TRANSFER_CANDIDATES and sel["best_wide_ptq_method"] in PROBE_METHODS
             and rq.is_wide(sel["best_wide_ptq_method"]), "selections keys or values invalid")
    split_tokens = {k: v["tokens"] for k, v in payload["data"]["splits"].items()}
    coverage, calibration = {}, payload["calibration"]
    for part, spec in (("ab", ab), ("c", pc)):
        config = PRESETS[spec["preset"]]
        coverage[part] = {split: c502.expected_coverage(split_tokens[split], config.sequence_length,
                                                        protocol["max_eval_windows"]) for split in SPLITS}
        c503._validate_calibration(calibration[str(config.sequence_length)], protocol, config.sequence_length,
                                   split_tokens["train"])
    _require(set(calibration) == {str(PRESETS[s["preset"]].sequence_length) for s in (ab, pc)},
             "calibration draws differ from the sequence lengths of the presets")
    if canonical:
        _require(all(coverage[p]["val"]["tokens"] == c502.VAL_TOKENS and coverage[p]["test"]["tokens"] == c502.TEST_TOKENS
                     for p in PARTS), "canonical scoring must cover the full val.bin and test.bin")
    val_scorings = test_scorings = 0
    for u in units:
        val, test = _validate_unit(u, payload, coverage)
        val_scorings, test_scorings = val_scorings + val, test_scorings + test
    _validate_siblings(payload)
    # Selections: recomputed from the val.bin rows, in the order the study made them.
    want = select_probe(units)
    _require(all(sel[k] == v for k, v in want.items()), "Part A selections differ from the val.bin results")
    transfer = select_transfer(units, sel)
    _require(all(_same(sel[k], v) for k, v in transfer.items() if k != "arm_best_method")
             and _same(sel["arm_best_method"].get("ab"), transfer["arm_best_method"]["ab"]),
             "Part B selections differ from the val.bin results")
    _require(_same(sel["arm_best_method"], {"ab": transfer["arm_best_method"]["ab"], "c": select_part_c(units, sel)}),
             "Part C best methods differ from the val.bin results")
    # AWQ: the alpha, the val.bin scores and the effective-weight digest of every (part, seed, arm, method).
    if v1:  # outside the v1 freeze digest: bound by a digest that was pinned after the run
        _require(selections_sha256(awq_freeze(units)) == HISTORICAL_V1_AWQ_SELECTIONS_SHA256,
                 f"the AWQ selections differ from the pinned selections of the preserved {HISTORICAL_SCHEMA_V1} result "
                 "(HISTORICAL_V1_AWQ_SELECTIONS_SHA256)")
    else:  # inside the freeze digest
        _require(_same(sel["awq_selections"], awq_freeze(units)),
                 "selections.awq_selections differ from the AWQ selections and effective-weight digests of the units")
    _require(payload["selections_sha256"] == selections_sha256(sel), "selections_sha256 differs")
    _require(payload["splits_opened_before_freeze"] == ["train", "val"],
             "only train.bin and val.bin may be mapped before the freeze")
    _require(_same(payload["scoring_ledger"], {"val_scorings": val_scorings, "test_scorings": test_scorings,
                                               "frozen_after_val_scorings": val_scorings,
                                               "test_scored_only_after_freeze": True}),
             "scoring ledger differs: every val.bin scoring must precede the freeze and every test.bin scoring")
    stage_min = {
        "stage_a": math.fsum(u["training"]["train_seconds"] + u["seconds"]["phase1"]
                             for u in units if u["part"] == "ab" and u["arm"] == MUON),
        "stage_b": math.fsum(u["training"]["train_seconds"] + u["seconds"]["phase1"]
                             for u in units if u["part"] == "ab" and u["arm"] != MUON),
        "stage_c": math.fsum(u["training"].get("train_seconds", 0.0) + u["seconds"]["phase1"]
                             for u in units if u["part"] == "c"),
        "test_scoring": math.fsum(u["seconds"]["phase2"] for u in units),
    }
    _require(set(payload["stage_seconds"]) == set(STAGES) and all(_nonneg(v) for v in payload["stage_seconds"].values())
             and all(payload["stage_seconds"][k] >= stage_min[k] - 1e-3 for k in STAGES)
             and payload["seconds"] >= sum(payload["stage_seconds"].values()) - 1e-6,
             "stage durations invalid or below sequential unit times")
    reference = payload["c5_03_reference"]
    if pc["baseline"] == "c5_03_preserved":
        _require(_same(reference, _trusted_c5_03_reference(payload["meta"]["git_sha"] if canonical else None)),
                 "c5_03_reference differs from the committed C5-03 result of this checkout")
        _require(all(reference["seeds"][str(s)]["checkpoint_sha256"] == PINNED_BASELINES[s]["checkpoint_sha256"]
                     and reference["seeds"][str(s)]["checkpoint_file_sha256"] == PINNED_BASELINES[s]["checkpoint_file_sha256"]
                     for s in seeds), "the pinned baseline digests differ from the committed C5-03 result")
        compared = _validate_c5_03_rows(units, reference)
    else:
        _require(reference is None, "a shakedown has no C5-03 reference")
    _assert_close(payload["analysis"], analyze(units, sel, protocol, reference), "analysis")
    a = payload["analysis"]
    if reference is not None:
        rep = a["c5_03_reproduction"]
        _require(len(rep["rows"]) == compared and all(e["abs_diff"] == 0.0 for e in rep["rows"])
                 and rep["max_abs_nll_difference"] == 0.0,
                 "analysis.c5_03_reproduction must show an exact reproduction (0.0 nats on every comparator row)")
    _require(a["hardware_vs_emulation"]["passed"] is True and len(a["hardware_vs_emulation"]["rows"]) == 2 * len(units),
             "hardware W4A4 must match the FP32 emulation within the tolerance for every checkpoint and split")
    n_ab = PRESETS[ab["preset"]].layers * len(q.LINEAR_KINDS)
    n_c = PRESETS[pc["preset"]].layers * len(q.LINEAR_KINDS)
    hw = a["hardware"]
    _require(hw["fp4_gemms_per_forward"] == sorted({n_ab, n_c}) and hw["bf16_block_linears_in_hw_rows"] == [0]
             and hw["fp4_gemms_per_forward_bf16"] == [0] and hw["fp4_gemms_per_forward_emulated"] == [0]
             and hw["decode_exact_matrices"] == hw["decode_checked_matrices"] and hw["wide_hw_rows"] > 0,
             "hardware evidence incomplete: FP4 GEMM counts, fallback count or int32 decode checks")


def verify_external_evidence(payload: dict, *, checkpoint_dir: Path | None = None, baseline_dir: Path | None = None,
                             evidence_dir: Path | None = None) -> dict:
    """Check a result against the files that its JSON cannot hold. Raises ``ValueError`` on any difference.

    * Every checkpoint: the SHA-256 of the file (streamed) and the tensor SHA-256 of its state dict (sorted
      names, shapes and FP32 bytes) against ``training.checkpoint_file_sha256`` and ``checkpoint_sha256``.
      Checkpoints trained in the run are in ``checkpoint_dir`` (24 in the canonical study); the preserved
      C5-03 checkpoints are in ``baseline_dir`` (3).
    * The test-access log in ``evidence_dir``: its first lines are ``data.test_access.prior_openings`` and
      then ``this_opening``. For a v2 result, that line holds the freeze digest that the run logged before
      it mapped test.bin.

    A directory that is not given is the one of the recorded command for a shakedown (``<output>.work``),
    and ``artifacts/`` of this checkout or of its main checkout for a canonical result. The function reads
    files only. It does not validate the payload: ``validate_results(..., verify_external=True)`` does both.
    """
    if payload["shakedown"]:
        recorded = _parsed_command(payload["meta"]["command"], "meta.command")
        checkpoint_dir = recorded.checkpoint_dir if checkpoint_dir is None else checkpoint_dir
        evidence_dir = shakedown_evidence_dir(recorded.work_dir) if evidence_dir is None else evidence_dir
    else:
        checkpoint_dir = _artifact_dir(CANONICAL_CHECKPOINT_DIR.name) if checkpoint_dir is None else checkpoint_dir
        baseline_dir = default_baseline_dir() if baseline_dir is None else baseline_dir
        evidence_dir = _artifact_dir(TEST_ACCESS_LOG.parent.name) if evidence_dir is None else evidence_dir
    directories = {"trained_in_run": checkpoint_dir, "c5_03_preserved": baseline_dir}
    verified, recorded_files, size = dict.fromkeys(directories, 0), set(), 0
    for u in payload["units"]:
        tr, where = u["training"], f"unit {u['part']}/{u['seed']}/{u['arm']}"
        name = tr["checkpoint_file"]
        _require(directories.get(tr["source"]) is not None and isinstance(name, str) and Path(name).name == name,
                 f"{where}: no directory for a {tr['source']!r} checkpoint named {name!r}")
        file = Path(directories[tr["source"]]) / name
        _require(file.is_file(), f"{where}: checkpoint file {file} is missing")
        _require(c503.file_sha256(file) == tr["checkpoint_file_sha256"],
                 f"{where}: the SHA-256 of {file} differs from training.checkpoint_file_sha256")
        _require(c502.checkpoint_file_matches_sha256(file, tr["checkpoint_sha256"]),
                 f"{where}: the tensor SHA-256 of {file} differs from training.checkpoint_sha256")
        verified[tr["source"]] += 1
        recorded_files.add(file)
        size += file.stat().st_size
    access = payload["data"]["test_access"]
    log_path = Path(evidence_dir) / TEST_ACCESS_LOG_NAME
    _require(log_path.is_file(), f"the external test-access log {log_path} is missing")
    try:
        lines = _read_access_log(log_path)
    except ValueError as exc:
        raise ValueError(f"the external test-access log {log_path} is not JSON lines") from exc
    expected = [*access["prior_openings"], access["this_opening"]]
    _require(len(lines) >= len(expected) and all(_same(line, entry) for line, entry in zip(lines, expected)),
             f"data.test_access differs from the external test-access log {log_path}: its first {len(expected)} "
             "line(s) must be the recorded prior openings and this opening")
    return {"checkpoints_trained_in_run_verified": verified["trained_in_run"],
            "checkpoints_c5_03_preserved_verified": verified["c5_03_preserved"],
            "checkpoint_bytes_hashed": size,
            "unrecorded_checkpoint_files": sorted(str(f) for d in {f.parent for f in recorded_files}
                                                  for f in d.glob("*.pt") if f not in recorded_files),
            "test_access_log": str(log_path), "test_access_log_sha256": c503.file_sha256(log_path),
            "test_access_log_entries": len(lines), "test_access_log_verified": True}


def validate_results(payload: dict, canonical: bool | None = None, *, verify_external: bool = False,
                     checkpoint_dir: Path | None = None, baseline_dir: Path | None = None,
                     evidence_dir: Path | None = None) -> dict:
    """Offline check of a C6-01 result. Recomputes every selection, statistic, summary and caveat from the rows.

    ``canonical`` states what the caller expects (None: the payload's label); a disagreement raises. A
    canonical result is bound to ``meta.git_sha`` (import closure, blobs, schema, and the summary code of
    this checkout, see ``_matches_current_summary_code``), to the pinned shard and baseline digests, to the
    committed C5-03 result (an exact reproduction of its comparator rows) and to the canonical output path
    (no symlink, no alias). It cannot re-measure: it trusts the recorded NLLs and timings that pass its
    consistency checks.

    Two schemas are valid. ``cycle6_c6_01.v2`` is the schema of new runs: the freeze digest covers the AWQ
    selections and is in the test-access line of the run. ``cycle6_c6_01.v1`` is valid for the preserved
    canonical result only. Its freeze digest covered the family, arm and method names, not the AWQ alphas;
    the validator binds its AWQ selections with ``HISTORICAL_V1_AWQ_SELECTIONS_SHA256``, a pin that was
    computed after the run. The receipt reports this as ``awq_in_contemporaneous_freeze`` (False for v1).

    ``verify_external=True``, or any of the three directories, also runs ``verify_external_evidence`` (the
    checkpoint files and the test-access log) and adds its report as ``external_evidence``.
    """
    try:
        receipt = _validate_payload(payload, canonical)
        if verify_external or any(d is not None for d in (checkpoint_dir, baseline_dir, evidence_dir)):
            receipt["external_evidence"] = verify_external_evidence(
                payload, checkpoint_dir=checkpoint_dir, baseline_dir=baseline_dir, evidence_dir=evidence_dir)
        return receipt
    except (KeyError, TypeError, IndexError, AttributeError, StopIteration) as exc:  # a record of the wrong shape
        raise ValueError(f"malformed payload: {type(exc).__name__}: {exc}") from exc


def _validate_payload(payload: dict, canonical: bool | None) -> dict:
    json.dumps(payload, allow_nan=False)
    _require(isinstance(payload, dict) and set(payload) == TOP_KEYS, "top-level keys differ from the schema")
    schema = payload["schema_version"]
    _require(isinstance(schema, str) and schema in SCHEMAS and payload["task"] == TASK, "schema or task differs")
    _require(type(payload["shakedown"]) is bool and type(payload["canonical"]) is bool
             and payload["canonical"] is not payload["shakedown"],
             "a result is either the canonical run or a synthetic shakedown")
    _require(canonical is None or type(canonical) is bool, "the requested canonical mode must be a bool or None")
    if canonical is True and not payload["canonical"]:
        raise ValueError("canonical certification was requested, but the payload is not labeled canonical")
    if canonical is False and payload["canonical"]:
        raise ValueError("the payload is labeled canonical: it cannot be validated with the noncanonical checks")
    canonical = payload["canonical"]
    _require(canonical or schema == SCHEMA_VERSION,
             f"{HISTORICAL_SCHEMA_V1} is valid for the preserved canonical result only: a shakedown is {SCHEMA_VERSION}")
    _require(_same(payload["protocol"], CANONICAL if canonical else SHAKEDOWN),
             "the protocol differs from the " + ("canonical C6-01 protocol" if canonical else "shakedown protocol"))
    _validate_data(payload, canonical)
    _validate_meta(payload, canonical)
    kp, cm = payload["kernel_parity"], payload["cross_mix_parity"]
    _require(kp["passed"] is True and kp["bitwise_failures"] == [] and kp["cases"] == (279 if canonical else 87),
             "kernel parity must pass every case bit for bit")
    _require(cm["passed"] is True and cm["bitwise_failures"] == [] and cm["cases"] == 6 * 2 * len(rq.WIDE_FAMILIES),
             "cross-mix kernel parity must pass every case bit for bit")
    _validate_study(payload, canonical)
    _assert_close(payload["summary"], summarize(payload), "summary")
    _require(payload["caveats"] == caveats(payload), "caveats differ from the recomputed caveats")
    if canonical:
        _require(SIGN_FLIP_MIN_P == 2.0 / (2 ** N_SEEDS), "SIGN_FLIP_MIN_P differs from 2 / 2**N_SEEDS")
        pns = _producer_namespace(payload["meta"]["git_sha"])
        p_analysis = pns["analyze"](payload["units"], payload["selections"], payload["protocol"],
                                    payload["c5_03_reference"])
        p_summary = pns["summarize"]({**payload, "analysis": p_analysis})
        p_caveats = pns["caveats"](payload)
        _assert_close(payload["analysis"], p_analysis, "producer_blob.analysis")
        _assert_close(payload["summary"], p_summary, "producer_blob.summary")
        _require(payload["caveats"] == p_caveats, "caveats differ from the producer commit's recomputed caveats")
    return {"certified_canonical": canonical, "shakedown": payload["shakedown"], "schema_version": schema,
            "awq_in_contemporaneous_freeze": schema == SCHEMA_VERSION}


# ======================================================================================
# CLI
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument("--synthetic-shakedown", action="store_true",
                        help="generated Markov shards and a small model: the only non-canonical mode (no real text)")
    parser.add_argument("--data-dir", type=Path, default=None, help="canonical run: the verified shard directory")
    parser.add_argument("--baseline-checkpoint-dir", type=Path, default=None,
                        help="canonical run: the directory of the preserved C5-03 checkpoints (digests are pinned)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="where trained state dicts go (default: artifacts/c6_01_checkpoints, or <output>.work)")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    args.argv = argv
    args.shakedown = args.synthetic_shakedown
    args.canonical = not args.shakedown
    if args.shakedown:
        if args.data_dir is not None or args.baseline_checkpoint_dir is not None:
            parser.error("--synthetic-shakedown generates its shards and baselines: remove --data-dir / "
                         "--baseline-checkpoint-dir")
        if args.output is None:
            parser.error("--synthetic-shakedown needs an explicit --output")
        if reserved(args.output):
            parser.error(f"Refusing to write the reserved {args.output.name} from a shakedown; choose a path outside "
                         f"{RESERVED_PATTERN}")
        args.protocol = copy.deepcopy(SHAKEDOWN)
        args.work_dir = args.output.with_name(args.output.name + ".work")
        if args.checkpoint_dir is None:
            args.checkpoint_dir = args.work_dir / "checkpoints"
        return args
    args.protocol = copy.deepcopy(CANONICAL)
    args.output = CANONICAL_OUTPUT if args.output is None else args.output
    try:
        verify_canonical_output_path(args.output)
    except ValueError as exc:
        parser.error(f"The canonical run writes {CANONICAL_OUTPUT} of this checkout ({exc}); every other run is "
                     "--synthetic-shakedown")
    args.work_dir = None
    if args.data_dir is None:
        args.data_dir = q.default_data_dir()
    if args.baseline_checkpoint_dir is None:
        args.baseline_checkpoint_dir = default_baseline_dir()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = CANONICAL_CHECKPOINT_DIR
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    payload = write_json_atomic(args.output, run(args), canonical_dest=args.canonical)
    if not args.silent:
        print(json.dumps({k: payload["summary"][k] for k in ("selections", "part_b_arms", "part_c_arms")}, indent=1))
        print(f"Wrote {args.output}")
    return payload


if __name__ == "__main__":
    main()
