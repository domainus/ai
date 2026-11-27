"""Cycle 5 C5-02: per-seed iso-loss Muon-versus-AdamW PTQ and three Muon PTQ-sensitivity mitigations.

C4-01b matched AdamW to Muon on one pilot seed and then locked the recipe. Only one of its three
held-out seeds met the 0.02-nat target, and its checkpoints were not kept. This benchmark matches
every seed on ``val.bin`` alone, keeps the matched weights, and tests three mitigations on the
same five paired seeds.

Stage A  Training and matching. The stage receives a mapping with ``train`` and ``val`` only;
         ``test.bin`` is not opened by the process until the stage has finished for every seed.
         Per seed (one ``initial_state``, one batch stream):
           ``muon_ss3_54``          320 steps, snapshots after steps 256 and 272.
           ``muon_adamw_cooldown``  branch of the step-256 snapshot: the last 64 steps update the
                                    hidden matrices with fresh AdamW (LR 0.001 x the same
                                    schedule multiplier, so the terminal LR equals the AdamW arm).
           ``muon_qat_cooldown``    branch of the step-272 snapshot: the last 48 Muon steps run
                                    with NVFP4 fake quantization (straight-through) of the weight
                                    and the input of every quantized linear.
           ``adamw``                one run at LR 0.001: warmup + cosine to the floor LR at step
                                    464, then the floor LR. Val NLL is measured every 16 steps
                                    from step 400; the run stops at the first measured step where
                                    AdamW is at or below Muon (cap 2560 steps). ``select_adamw``
                                    takes the measured step with the smallest |val NLL(adamw) -
                                    val NLL(muon)| (then fewer steps); it is matched when the gap
                                    is <= 0.02 nats.
         Every measured step is stored. The four final state dicts of a seed are saved (not
         committed).
Stage B  Evaluation of the saved checkpoints (no retraining, same SHA-256). Each checkpoint gets
         full ``val.bin`` / ``test.bin`` NLL, FP64-Hessian weight-only PTQ (4 formats x 4
         methods), hardware NVFP4 W4A4 through ``NVFP4Linear`` (dynamic and static activation
         scales, FP4 GEMMs counted by the profiler, int32 decode checks, one FP32 emulation row),
         and per-layer geometry (stable rank, signal / error alignment with the Hessian,
         unweighted against Hessian-weighted error).
         ``awq_*`` rows (the two baseline arms): activation-aware channel scales
         ``s_j = mean(|x_j|)^alpha`` before the block-16 rotation and GPTQ. ``alpha`` is chosen
         per (checkpoint, format, rotation) on ``val.bin`` (weight-only fake quantization);
         ``alpha = 0`` is the plain rotation + GPTQ row, bit for bit.
Analysis Paired per-seed differences (df = 4): Student-t test and 95% interval, the exact
         two-sided sign-flip p-value (2^5 sign patterns, minimum 0.0625) and Holm-Bonferroni
         corrections, for the test NLL increase and the quantized test NLL. Contrasts: Muon
         against AdamW, each cooldown arm and the activation-aware scaling against Muon and
         against AdamW; with one val-selected method for both sides ("primary") and with each
         side's own val-selected method ("own_best").

``cycle5_c5_02*.json`` is reserved in every checkout and through symlinks: only the canonical
protocol, on an sm_120+ device and a clean git tree, may write it. Both guards run before any
GPU work. ``validate_results`` documents what an offline check can and cannot establish.

Run (alone on the GPU, from a clean commit, about 3.3 hours on an RTX 5080: each of the 620 grid
rows reads the complete val.bin and test.bin; ``--resume`` continues an interrupted run):
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_c5_02_isoloss_ptq_mitigation.py
Smoke run:
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_c5_02_isoloss_ptq_mitigation.py --quick \\
        --output /tmp/c5_02.json
"""

from __future__ import annotations

import argparse
import ast
import copy
import fnmatch
import functools
import hashlib
import io
import itertools
import json
import math
import os
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from research.polar_opt import harness_tier2_real as h
from research.polar_opt import real_text_data
from research.polar_opt.harness_tier2_lm import Transformer
from research.quant_ptq import bench_isoloss_ptq as c4
from research.quant_ptq import bench_nvfp4_e2e as e2e
from research.quant_ptq import micro_rotated_ptq as q
from research.quant_ptq import nvfp4_inference_blackwell as nb

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "cycle5_c5_02.v1"
TASK = ("Cycle 5 C5-02: per-seed validation-only iso-loss Muon-vs-AdamW PTQ and Muon PTQ-sensitivity "
        "mitigations (AdamW cooldown, NVFP4 QAT cooldown, activation-aware scaling) on Blackwell")
SCRIPT_NAME = "bench_c5_02_isoloss_ptq_mitigation.py"
CANONICAL_OUTPUT = ROOT / "results" / "cycle5_c5_02_isoloss_ptq_mitigation.json"
QUICK_OUTPUT = ROOT / "results" / "c5_02_quick.json"
CANONICAL_CHECKPOINT_DIR = ROOT / "artifacts" / "c5_02_checkpoints"
RESERVED_PATTERN = "cycle5_c5_02*.json"

MUON, ADAMW, COOL_ADAMW, COOL_QAT = "muon_ss3_54", "adamw", "muon_adamw_cooldown", "muon_qat_cooldown"
ARMS = (MUON, ADAMW, COOL_ADAMW, COOL_QAT)
WEIGHT_DECAY, MIN_LR_RATIO, GPTQ_DAMP = c4.WEIGHT_DECAY, c4.MIN_LR_RATIO, c4.GPTQ_DAMP
BASE_METHODS = ("rtn", "gptq", "hadamard_b16_gptq", "adaptive_butterfly_b16_gptq")
AWQ_ROTATIONS = ("hadamard_b16", "adaptive_butterfly_b16")
AWQ_METHODS = tuple(f"awq_{r}_gptq" for r in AWQ_ROTATIONS)
AWQ_ARMS = (MUON, ADAMW)
HW_FORMAT, QAT_FORMAT = "nvfp4_g16", "nvfp4_g16"
HW_MODES = ("w4a4_hw_dynamic", "w4a4_hw_static")
EMULATED_METHOD, EMULATED_MODE = "hadamard_b16_gptq", "w4a4_emulated_fp32_dynamic"
EMULATION_PARITY_NATS = 1e-3
STATIC_CALIBRATION = "train.bin calibration windows, BF16 model, max |(x / s) R^T| / 2688"
VAL_TOKENS, TEST_TOKENS = 2_001_920, 1_999_872  # complete 512-token windows of val.bin and test.bin
SPLITS = ("val", "test")
N_SEEDS = 5
T975_DF4 = 2.7764451051977987  # Student-t 0.975 quantile, 4 degrees of freedom
SIGN_FLIP_MIN_P = 2 / 2 ** N_SEEDS
LAYER_METRICS = ("weight_excess_kurtosis", "weight_max_over_rms", "activation_excess_kurtosis",
                 "activation_max_over_rms", "hessian_diag_max_over_mean", "weight_fro_norm",
                 "weight_stable_rank", "signal_alignment")
GEOMETRY = ("weight_rel_error", "error_alignment", "output_rel_error")
ENDPOINTS = ("delta_nll", "nll")
TOL = 1e-12
ENV_KEYS = ("torch", "triton", "cuda", "driver", "device", "capability")

# Design-time pilot (seed 0, val.bin only, never part of a result). Muon: val NLL 5.343 to 5.346.
# * Fresh warmup + cosine AdamW runs at LR 0.001 with 456 .. 704 steps: two runs of one configuration differed
#   by 0.06 nats (640 steps) and 0.07 nats (472 steps), and 496 / 512 / 528 / 536 steps gave 5.393 / 5.500 /
#   5.338 / 5.262. A match by fresh runs is therefore a lottery.
# * Inside one run the val NLL falls smoothly at the floor LR (about 0.0044 nats per 8 steps), so one run per
#   seed is stopped on a 16-step grid instead. Three runs with ``cosine_steps`` = 464 were 0.05 to 0.28 nats
#   above Muon when the floor LR was reached; the worst needs about 500 floor-LR steps.
# * The floor-LR phase does not change the PTQ sensitivity of AdamW at equal loss: identity-GPTQ val NLL
#   increase 0.0010 (NVFP4) / 0.0096 (INT2 g16) at val NLL 5.41 of a stopped run, against 0.0010 / 0.0106 and
#   0.0008 / 0.0106 for fresh cosine runs of 544 and 608 steps (val NLL 5.30 and 5.24).
# Protocol amendment before any Stage B work (val.bin only; test.bin had not been opened). The first canonical
# attempt (commit 05ea4ae, ``max_steps`` = 1280) was stopped during Stage A: seed 1 matched at 640 steps
# (gap 0.003 nats), but the AdamW run of seed 2 was 0.58 nats above Muon at step 400 and still 0.09 nats above
# it at step 1280. AdamW at LR 0.001 has such slow runs, so the cap is 2560 steps (about 0.9 nats of floor-LR
# progress). Nothing else changed. A run that needs more is reported as unmatched.
CANONICAL = {
    "preset": "50m", "device": "cuda", "seeds": [1, 2, 3, 4, 5], "tolerance_nats": 0.02,
    "muon": {"steps": 320, "lr": 0.02, "auxiliary_lr": 0.001},
    "adamw": {"lr": 0.001, "cosine_steps": 464, "first_eval_step": 400, "eval_grid": 16, "max_steps": 2560},
    "adamw_cooldown": {"muon_steps": 256, "cooldown_steps": 64, "lr": 0.001},
    "qat_cooldown": {"muon_steps": 272, "cooldown_steps": 48, "format": QAT_FORMAT},
    "weight_decay": WEIGHT_DECAY, "min_lr_ratio": MIN_LR_RATIO,
    "schedule": "linear warmup over max(20, T // 10) steps, then cosine decay to min_lr_ratio at step T "
                "(Muon: T = 320; AdamW: T = cosine_steps, then min_lr_ratio until the run stops); both cooldown "
                "arms keep the multiplier of the 320-step Muon run",
    "calibration_sequences": 128, "calibration_seed": 20261126, "butterfly_steps": 150, "gptq_damp": GPTQ_DAMP,
    "weight_formats": list(q.FORMATS), "base_methods": list(BASE_METHODS),
    "awq": {"alphas": [0.0, 0.25, 0.5, 0.75], "rotations": list(AWQ_ROTATIONS), "arms": list(AWQ_ARMS)},
    "hardware_format": HW_FORMAT, "hardware_modes": list(HW_MODES), "emulated": [EMULATED_METHOD, EMULATED_MODE],
    "max_eval_windows": None,
}
QUICK = {
    **CANONICAL, "preset": "tiny",
    "muon": {"steps": 24, "lr": 0.02, "auxiliary_lr": 0.001},
    "adamw": {"lr": 0.001, "cosine_steps": 32, "first_eval_step": 24, "eval_grid": 8, "max_steps": 56},
    "adamw_cooldown": {"muon_steps": 16, "cooldown_steps": 8, "lr": 0.001},
    "qat_cooldown": {"muon_steps": 18, "cooldown_steps": 6, "format": QAT_FORMAT},
    "calibration_sequences": 4, "butterfly_steps": 5,
    "max_eval_windows": 16,
}


def fakequant_mode(fmt_name: str) -> str:
    return c4.fakequant_mode(q.FORMATS[fmt_name])


def parse_method(method: str) -> tuple[str, str, bool]:
    """(rotation family, rounding, activation-aware scaling) of a method name."""
    awq = method.startswith("awq_")
    family, rounding = q.parse_method(method.removeprefix("awq_"))
    return family, rounding, awq


def methods_of(arm: str, protocol: dict) -> list[str]:
    extra = [f"awq_{r}_gptq" for r in protocol["awq"]["rotations"]] if arm in protocol["awq"]["arms"] else []
    return [*protocol["base_methods"], *extra]


def expected_combos(arm: str, protocol: dict, hardware: bool) -> list[tuple[str, str, str]]:
    """The (format, method, mode) rows of one checkpoint, in the order the study writes them."""
    combos = []
    for fmt in protocol["weight_formats"]:
        for method in methods_of(arm, protocol):
            combos.append((fmt, method, fakequant_mode(fmt)))
            if hardware and fmt == protocol["hardware_format"]:
                combos += [(fmt, method, mode) for mode in protocol["hardware_modes"]]
                if method == protocol["emulated"][0]:
                    combos.append((fmt, method, protocol["emulated"][1]))
    return combos


# ======================================================================================
# Guards and provenance
# ======================================================================================
def _is_hex(value, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def reserved(path: Path) -> bool:
    """True for a ``cycle5_c5_02*.json`` name, in any checkout or through a symlink."""
    path = Path(path)
    return any(fnmatch.fnmatchcase(name, RESERVED_PATTERN) for name in (path.name, path.resolve().name))


def require_canonical_host(device: torch.device, git: dict, output: Path, canonical: bool) -> None:
    """A canonical run (or a reserved filename) needs an sm_120+ CUDA device and a clean, committed tree."""
    if not (canonical or reserved(output)):
        return
    if device.type != "cuda" or not torch.cuda.is_available() or \
            tuple(torch.cuda.get_device_capability(device)) < (12, 0):
        raise RuntimeError(f"{Path(output).name} requires a Blackwell (sm_120+) CUDA device")
    if git.get("git_dirty") is not False or not _is_hex(git.get("git_sha"), 40):
        raise RuntimeError("A canonical run needs a clean, committed git tree (git_dirty must be False)")


def _module_path(name: str) -> Path:
    return ROOT.joinpath(*name.split(".")).with_suffix(".py")


@functools.cache
def code_modules() -> tuple[str, ...]:
    """Every ``research.*`` module this driver imports, directly or transitively (static ``ast`` closure)."""
    found, todo = set(), ["research.quant_ptq.bench_c5_02_isoloss_ptq_mitigation"]
    while todo:
        name = todo.pop()
        if name in found or not _module_path(name).is_file():
            continue
        found.add(name)
        for node in ast.walk(ast.parse(_module_path(name).read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("research"):
                todo += [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
            elif isinstance(node, ast.Import):
                todo += [alias.name for alias in node.names if alias.name.startswith("research")]
    return tuple(sorted(found))


def code_sha256() -> str:
    digest = hashlib.sha256()
    for name in code_modules():
        digest.update(name.encode() + b"\0" + _module_path(name).read_bytes())
    return digest.hexdigest()


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def run_config_hash(quick: bool, protocol: dict, config: dict, code: str, git_sha: str | None, env: dict,
                    data_sha256: dict) -> str:
    return _digest({"schema": SCHEMA_VERSION, "quick": quick, "protocol": protocol, "config": config,
                    "code_sha256": code, "git_sha": git_sha, "env": {k: env.get(k) for k in ENV_KEYS},
                    "data_sha256": data_sha256})


def unit_config_hash(protocol: dict, config: dict, code: str, env: dict, data_sha256: dict) -> str:
    """Resume key of a per-seed unit: everything that determines it, without the seed list.

    Transitive code, torch / triton / cuda / driver / device, shard hashes, the model config and the
    protocol (calibration seed included). A changed dependency or runtime therefore reuses nothing.
    """
    return _digest({"schema": SCHEMA_VERSION, "protocol": {k: v for k, v in protocol.items() if k != "seeds"},
                    "config": config, "code_sha256": code, "env": {k: env.get(k) for k in ENV_KEYS},
                    "data_sha256": data_sha256})


class UnitLog:
    """Finished units as JSON lines next to the output, keyed by (stage, seed, arm, unit hash).

    ``--resume`` reloads the units of an interrupted run; without it the file starts empty.
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

    def get(self, key: tuple, usable=lambda value: True) -> dict | None:
        value = self.units.get((*key, self.config_hash))
        if value is None or not usable(value):
            return None
        self.loaded += 1
        return {**value, "resumed_from_unit_log": True}

    def put(self, key: tuple, value: dict) -> dict:
        value = json.loads(json.dumps(value, allow_nan=False))  # plain JSON types, as after a reload
        self.run += 1
        self.units[(*key, self.config_hash)] = value
        if self.path is not None:
            with self.path.open("a") as handle:
                handle.write(json.dumps({"key": [*key, self.config_hash], "value": value}, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return {**value, "resumed_from_unit_log": False}


def unit_log_path(output: Path) -> Path:
    return output.with_name(output.name + ".units.jsonl")


# ======================================================================================
# Data: splits are opened one by one, so a stage can be given a mapping without test.bin
# ======================================================================================
class Shards:
    """Verified shard directory. ``open`` memory-maps single splits and records the order."""

    def __init__(self, directory: Path, *, verify: bool, expected: dict | None):
        self.directory = Path(directory)
        self.manifest = (real_text_data.verify_manifest(self.directory) if verify
                         else json.loads((self.directory / "manifest.json").read_text()))
        found = {k: v["sha256"] for k, v in self.manifest["splits"].items()}
        if expected and any(found.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Shard SHA-256 values differ from the pinned hashes: {found}")
        self.opened: list[str] = []

    def open(self, *names: str) -> dict[str, np.ndarray]:
        view = {}
        for name in names:
            entry = self.manifest["splits"][name]
            view[name] = np.memmap(self.directory / entry["file"], dtype=np.uint16, mode="r")
            if name not in self.opened:
                self.opened.append(name)
        return view

    def summary(self) -> dict:
        return {"source": self.manifest["source"], "encoding": self.manifest["encoding"],
                "splits": {k: {"tokens": v["tokens"], "sha256": v["sha256"]}
                           for k, v in self.manifest["splits"].items()}}


# ======================================================================================
# Training arms
# ======================================================================================
def schedule_multiplier(step: int, total: int) -> float:
    return h.lr_multiplier(step, step / total, c4.warmup_for(total), MIN_LR_RATIO, total_steps=total)


def adamw_multiplier(step: int, spec: dict) -> float:
    """AdamW LR multiplier of step ``step``: warmup + cosine to the floor at ``cosine_steps``, then the floor."""
    return schedule_multiplier(step, spec["cosine_steps"]) if step < spec["cosine_steps"] else MIN_LR_RATIO


def adamw_eval_steps(spec: dict) -> list[int]:
    """The step grid on which the AdamW run is measured on val.bin."""
    return list(range(spec["first_eval_step"], spec["max_steps"] + 1, spec["eval_grid"]))


def adamw_stops(val_gap: float) -> bool:
    """The run stops at the first measured step where AdamW is at or below Muon on val.bin."""
    return val_gap <= 0.0


class FakeQuantLinear(nn.Module):
    """A linear layer with fake-quantized weight and input, both with straight-through gradients.

    Identity coordinates, round to nearest: the weight uses groups of 16 along the input dimension
    and one FP32 tensor scale; the input uses groups of 16 and the dynamic scale of the forward
    batch. The fake-quantized FP32 operands go to the BF16 autocast GEMM of the training step.
    """

    def __init__(self, linear: nn.Linear, fmt: q.QuantFormat):
        super().__init__()
        self.linear, self.fmt = linear, fmt

    def forward(self, x: Tensor) -> Tensor:
        w = self.linear.weight
        with torch.autocast(device_type=x.device.type, enabled=False):
            wq = w + (q.quantize_rtn(w.detach(), self.fmt) - w.detach())
            x2 = x.reshape(-1, x.shape[-1]).float()
            xq = (x2 + (q.quantize_rtn(x2.detach(), self.fmt) - x2.detach())).reshape(x.shape)
        return F.linear(xq, wq)


class CooldownAdamW:
    """Fresh fused AdamW on the hidden matrices; the auxiliary AdamW of the Muon run continues."""

    def __init__(self, muon: h.Muon, lr: float):
        parameters = list(muon.matrices.values())
        self.matrix = h._adamw(parameters, lr, WEIGHT_DECAY, parameters[0].is_cuda)
        self.auxiliary = muon.auxiliary

    def set_lr_multiplier(self, multiplier: float) -> None:
        for group in (*self.matrix.param_groups, *self.auxiliary.param_groups):
            group["lr"] = group["base_lr"] * multiplier

    def step(self, timer=None) -> None:
        self.matrix.step()
        self.auxiliary.step()


def _new_model(config: h.ModelConfig, device: torch.device, state: dict[str, Tensor]) -> Transformer:
    with torch.device(device):
        model = Transformer(config)
    model.load_state_dict(state)
    return model.train()


def _muon_optimizer(model, protocol: dict, device: torch.device):
    factory, orthogonalizer = q._factory(MUON, device)
    spec = protocol["muon"]
    return h.build_optimizer(MUON, factory, model, lr=spec["lr"], auxiliary_lr=spec["auxiliary_lr"],
                             weight_decay=WEIGHT_DECAY, update_dtype="fp32"), orthogonalizer


def _snapshot(model, optimizer: h.Muon) -> dict:
    return {"model": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "momentum": {k: v.detach().clone() for k, v in optimizer.momentum.items()},
            "auxiliary": copy.deepcopy(optimizer.auxiliary.state_dict()),
            "sha256": e2e._checkpoint_sha256(model)}


def _restore(config: h.ModelConfig, device: torch.device, protocol: dict, snapshot: dict):
    """Model and Muon optimizer in the exact state of ``snapshot`` (weights, momentum, auxiliary AdamW)."""
    model = _new_model(config, device, snapshot["model"])
    optimizer, _ = _muon_optimizer(model, protocol, device)
    with torch.no_grad():
        for name, buffer in optimizer.momentum.items():
            buffer.copy_(snapshot["momentum"][name])
    optimizer.auxiliary.load_state_dict(copy.deepcopy(snapshot["auxiliary"]))
    return model, optimizer


def _stream_at(train: np.ndarray, config: h.ModelConfig, seed: int, device: torch.device, step: int) -> h.BatchStream:
    """The seed's batch stream, advanced past the first ``step`` optimizer steps."""
    stream = h.BatchStream(train, config, seed, device)
    for _ in range(step):
        stream.next_starts()
    return stream


def _run_steps(model, optimizer, stream: h.BatchStream, first: int, last: int, multiplier, label: str,
               at_step=None) -> list[float]:
    """Optimizer steps ``first .. last - 1`` with LR multiplier ``multiplier(step)``. Reads the stream only."""
    losses = []
    for step in range(first, last):
        if at_step is not None:
            at_step(step)
        optimizer.set_lr_multiplier(multiplier(step))
        losses.append(h.train_step(model, stream.next_batch(), optimizer).item())
        model.zero_grad(set_to_none=True)
        if not math.isfinite(losses[-1]):
            raise RuntimeError(f"{label} diverged at step {step + 1}")
    return losses


def _training_row(arm: str, sha256: str, config: h.ModelConfig, train: np.ndarray, *, seed: int, steps: int,
                  lr: float, auxiliary_lr: float | None, init: dict, losses: list[float], first_step: int,
                  seconds: float, orthogonalizer: str, schedule_steps: int | None = None, **extra) -> dict:
    """Training record of a checkpoint after ``steps`` steps; ``losses`` are its own last phase."""
    tokens = steps * config.step_tokens
    return {"arm": arm, "seed": seed, "steps": steps, "warmup_steps": c4.warmup_for(schedule_steps or steps), "lr": lr,
            "auxiliary_lr": auxiliary_lr, "weight_decay": WEIGHT_DECAY, "min_lr_ratio": MIN_LR_RATIO,
            "tokens": tokens, "train_epochs": tokens / len(train), "orthogonalizer": orthogonalizer,
            "init_fingerprint": h.state_fingerprint(init),
            "stream_fingerprint": h.stream_fingerprint(train, config, seed),
            "train_loss": math.fsum(losses[-10:]) / len(losses[-10:]), "last_train_loss": losses[-1],
            "train_loss_note": "train_loss is the mean of the last 10 optimizer steps (of this arm's own phase)",
            "train_curve": [[first_step + i + 1, v] for i, v in enumerate(losses) if i == 0 or (i + 1) % 10 == 0],
            "train_seconds": seconds, "checkpoint_sha256": sha256, **extra}


def train_muon(config, train, device, protocol: dict, *, seed: int, init: dict, snapshot_steps=()):
    """The 320-step Muon run. Returns (model, row, {step: snapshot taken before that step})."""
    steps = protocol["muon"]["steps"]
    model = _new_model(config, device, init)
    optimizer, orthogonalizer = _muon_optimizer(model, protocol, device)
    snapshots = {}

    def at_step(step: int) -> None:
        if step in snapshot_steps:
            snapshots[step] = _snapshot(model, optimizer)

    began = time.perf_counter()
    losses = _run_steps(model, optimizer, h.BatchStream(train, config, seed, device), 0, steps,
                        lambda step: schedule_multiplier(step, steps), f"{MUON} seed {seed}", at_step)
    row = _training_row(MUON, e2e._checkpoint_sha256(model.eval()), config, train, seed=seed, steps=steps,
                        lr=protocol["muon"]["lr"],
                        auxiliary_lr=protocol["muon"]["auxiliary_lr"], init=init, losses=losses, first_step=0,
                        seconds=time.perf_counter() - began, orthogonalizer=orthogonalizer,
                        snapshots={str(step): snap["sha256"] for step, snap in sorted(snapshots.items())})
    return model, row, snapshots


def train_adamw_matched(config, train, device, protocol: dict, *, seed: int, init: dict, target_val_nll: float,
                        evaluate):
    """One AdamW run, measured on the val grid and stopped by ``adamw_stops``.

    ``evaluate(model)`` gives the val.bin result of the current weights. Returns the state dict of the
    measured step nearest to ``target_val_nll`` (then fewer steps) and the training record of that step.
    """
    spec = protocol["adamw"]
    model = _new_model(config, device, init)
    optimizer = h.build_optimizer(ADAMW, None, model, lr=spec["lr"], auxiliary_lr=0.0, weight_decay=WEIGHT_DECAY,
                                  update_dtype="fp32")
    stream = h.BatchStream(train, config, seed, device)
    losses, evaluations, best, state, done, eval_seconds = [], [], None, None, 0, 0.0
    began = time.perf_counter()
    for target in adamw_eval_steps(spec):
        losses += _run_steps(model, optimizer, stream, done, target, lambda step: adamw_multiplier(step, spec),
                             f"{ADAMW} seed {seed}")
        done = target
        t1 = time.perf_counter()
        result = evaluate(model.eval())
        gap = result["nll"] - target_val_nll
        evaluations.append({"step": target, "lr_multiplier": adamw_multiplier(target - 1, spec),
                            "val_nll": result["nll"], "val_tokens": result["tokens"], "val_gap": gap,
                            "checkpoint_sha256": e2e._checkpoint_sha256(model)})
        if best is None or (abs(gap), target) < best:
            best, state = (abs(gap), target), {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.train()
        eval_seconds += time.perf_counter() - t1
        if adamw_stops(gap):
            break
    chosen = next(e for e in evaluations if e["step"] == best[1])
    row = _training_row(ADAMW, chosen["checkpoint_sha256"], config, train, seed=seed, steps=chosen["step"],
                        lr=spec["lr"], auxiliary_lr=None, init=init, losses=losses[:chosen["step"]], first_step=0,
                        seconds=time.perf_counter() - began - eval_seconds, orthogonalizer="fused AdamW",
                        schedule_steps=spec["cosine_steps"], cosine_steps=spec["cosine_steps"], trained_steps=done,
                        eval_seconds=eval_seconds, evaluations=evaluations)
    row.update(val_nll=chosen["val_nll"], val_ppl=math.exp(chosen["val_nll"]), val_tokens=chosen["val_tokens"],
               evaluated_splits=["val"])
    return state, row


def train_cooldown(arm: str, config, train, device, protocol: dict, *, seed: int, init: dict, snapshot: dict):
    """A cooldown arm: the Muon run up to ``snapshot``, then its own last steps on the same batches."""
    total = protocol["muon"]["steps"]
    spec = protocol["adamw_cooldown" if arm == COOL_ADAMW else "qat_cooldown"]
    first = spec["muon_steps"]
    if first + spec["cooldown_steps"] != total:
        raise ValueError(f"{arm}: muon_steps + cooldown_steps must equal the Muon step count")
    model, muon = _restore(config, device, protocol, snapshot)
    stream = _stream_at(train, config, seed, device, first)
    _, orthogonalizer = q._factory(MUON, device)
    began = time.perf_counter()
    multiplier = lambda step: schedule_multiplier(step, total)  # noqa: E731
    if arm == COOL_ADAMW:
        losses = _run_steps(model, CooldownAdamW(muon, spec["lr"]), stream, first, total, multiplier,
                            f"{arm} seed {seed}")
        extra = {"cooldown": {"optimizer": "fused AdamW on the hidden matrices (fresh moments); auxiliary AdamW "
                                           "continues", "lr": spec["lr"]}}
    elif arm == COOL_QAT:
        fmt = q.FORMATS[spec["format"]]
        wrapped = {name: FakeQuantLinear(linear, fmt) for name, linear in q.quantized_linears(model).items()}
        previous = e2e._swap(model, wrapped)
        try:
            losses = _run_steps(model, muon, stream, first, total, multiplier, f"{arm} seed {seed}")
        finally:
            e2e._swap(model, previous)
        extra = {"cooldown": {"optimizer": "Muon continues (same momentum and auxiliary AdamW)",
                              "fake_quant": f"{spec['format']} round-to-nearest, straight-through, weight and "
                                            "input of every quantized linear, identity coordinates"}}
    else:
        raise ValueError(f"Unknown cooldown arm {arm!r}")
    row = _training_row(arm, e2e._checkpoint_sha256(model.eval()), config, train, seed=seed, steps=total,
                        lr=protocol["muon"]["lr"],
                        auxiliary_lr=protocol["muon"]["auxiliary_lr"], init=init, losses=losses, first_step=first,
                        seconds=time.perf_counter() - began, orthogonalizer=orthogonalizer,
                        branch={"from_arm": MUON, "from_step": first, "snapshot_sha256": snapshot["sha256"],
                                "cooldown_steps": spec["cooldown_steps"]}, **extra)
    return model, row


# ======================================================================================
# Stage A: per-seed AdamW stopping step on val.bin (a pure function of the stored measurements)
# ======================================================================================
def select_adamw(rows: list[dict], tolerance: float) -> dict:
    """The measured AdamW step nearest to Muon on val.bin (then fewer steps). Reads val NLL only."""
    muon = next(r for r in rows if r["arm"] == MUON)
    adamw = next((r for r in rows if r["arm"] == ADAMW), None)
    if adamw is None or not adamw["evaluations"]:
        raise ValueError("The seed has no measured AdamW step")
    evaluations = adamw["evaluations"]
    best = min(evaluations, key=lambda e: (abs(e["val_nll"] - muon["val_nll"]), e["step"]))
    gap = muon["val_nll"] - best["val_nll"]
    tokens = best["step"] * (muon["tokens"] // muon["steps"])
    return {"matched": abs(gap) <= tolerance, "tolerance_nats": tolerance, "t_muon": muon["steps"],
            "t_adamw": best["step"], "adamw_lr": adamw["lr"], "adamw_cosine_steps": adamw["cosine_steps"],
            "adamw_floor_lr_steps": best["step"] - adamw["cosine_steps"], "muon_val_nll": muon["val_nll"],
            "adamw_val_nll": best["val_nll"], "val_gap_muon_minus_adamw": gap, "abs_val_gap": abs(gap),
            "muon_tokens": muon["tokens"], "adamw_tokens": tokens, "adamw_over_muon_tokens": tokens / muon["tokens"],
            "adamw_evaluations": len(evaluations),
            "adamw_stopped_at_or_below_muon": adamw_stops(evaluations[-1]["val_nll"] - muon["val_nll"]),
            "adamw_checkpoint_sha256": best["checkpoint_sha256"],
            "rule": "the measured AdamW step with the smallest |val NLL difference| to the seed's Muon run on "
                    "val.bin; then fewer AdamW steps"}


def _save_state(state: dict[str, Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({k: v.detach().cpu() for k, v in state.items()}, tmp)
    tmp.replace(path)


def checkpoint_name(seed: int, arm: str) -> str:
    return f"seed{seed}_{arm}.pt"


def stage_a_seed(config: h.ModelConfig, view: dict, device: torch.device, protocol: dict, seed: int,
                 checkpoint_dir: Path, log) -> dict:
    """Train the four arms of ``seed`` and match AdamW to Muon. ``view`` holds train and val only."""
    if set(view) != {"train", "val"}:
        raise RuntimeError("Stage A needs a split mapping with exactly train and val")
    train, val = view["train"], view["val"]
    began = time.perf_counter()
    init = h.initial_state(config, seed)
    rows = []

    def evaluate(model) -> dict:
        return q.evaluate_split(model, val, config.sequence_length, config.micro_batch, device,
                                protocol["max_eval_windows"])

    def finish(row: dict) -> None:
        rows.append(row)
        log(f"  A seed {seed} {row['arm']:<20} steps {row['steps']:4d} train {row['train_loss']:.4f} "
            f"val NLL {row['val_nll']:.4f} ({row['train_seconds']:.0f}s)")

    def finish_model(model, row: dict) -> None:
        result = evaluate(model)
        row.update(val_nll=result["nll"], val_ppl=result["ppl"], val_tokens=result["tokens"],
                   evaluated_splits=["val"])
        _save_state(model.state_dict(), checkpoint_dir / checkpoint_name(seed, row["arm"]))
        finish(row)

    cool = {COOL_ADAMW: protocol["adamw_cooldown"]["muon_steps"], COOL_QAT: protocol["qat_cooldown"]["muon_steps"]}
    model, row, snapshots = train_muon(config, train, device, protocol, seed=seed, init=init,
                                       snapshot_steps=tuple(cool.values()))
    finish_model(model, row)
    for arm, step in cool.items():
        model, row = train_cooldown(arm, config, train, device, protocol, seed=seed, init=init,
                                    snapshot=snapshots[step])
        finish_model(model, row)
    del snapshots, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    state, row = train_adamw_matched(config, train, device, protocol, seed=seed, init=init,
                                     target_val_nll=rows[0]["val_nll"], evaluate=evaluate)
    _save_state(state, checkpoint_dir / checkpoint_name(seed, ADAMW))
    del state
    finish(row)
    for e in row["evaluations"]:
        log(f"    adamw step {e['step']:4d} lr x{e['lr_multiplier']:.3f} val NLL {e['val_nll']:.4f} "
            f"gap {e['val_gap']:+.4f}")
    selection = select_adamw(rows, protocol["tolerance_nats"])
    if (selection["t_adamw"], selection["adamw_checkpoint_sha256"]) != (row["steps"], row["checkpoint_sha256"]):
        raise RuntimeError("The saved AdamW checkpoint is not the selected step")
    return {"seed": seed, "split_view": sorted(view), "rows": rows, "selection": selection,
            "checkpoints": {r["arm"]: {"file": checkpoint_name(seed, r["arm"]), "sha256": r["checkpoint_sha256"],
                                       "val_nll": r["val_nll"]} for r in (rows[0], rows[3], rows[1], rows[2])},
            "seconds": time.perf_counter() - began}


def load_checkpoint(config: h.ModelConfig, device: torch.device, path: Path, sha256: str) -> Transformer:
    """The saved state dict of a Stage A arm; raises unless its SHA-256 is the recorded one."""
    model = _new_model(config, device, torch.load(path, map_location="cpu", weights_only=True)).eval()
    if e2e._checkpoint_sha256(model) != sha256:
        raise RuntimeError(f"{path.name}: checkpoint SHA-256 differs from the Stage A record")
    return model


def checkpoints_present(unit: dict, checkpoint_dir: Path) -> bool:
    return all((checkpoint_dir / entry["file"]).is_file() for entry in unit["checkpoints"].values())


# ======================================================================================
# Stage B: calibration statistics, geometry and activation-aware scaling
# ======================================================================================
@torch.no_grad()
def collect_calibration_statistics(model, windows: np.ndarray, batch: int, device: torch.device):
    """One pass over ``windows``: FP64 Hessians, input moments and per-channel mean |x| of every linear."""
    linears = q.quantized_linears(model)
    moments = {name: c4.Moments(device) for name in linears}
    abs_sum = {name: torch.zeros(m.in_features, dtype=torch.float64, device=device) for name, m in linears.items()}
    rows = dict.fromkeys(linears, 0)

    def hook(name):
        def record(module, inputs):
            with torch.autocast(device_type=device.type, enabled=False):
                x = inputs[0].detach().reshape(-1, module.in_features)
                moments[name].update(x)
                abs_sum[name] += x.double().abs().sum(0)
                rows[name] += x.shape[0]
        return record

    handles = [m.register_forward_pre_hook(hook(name)) for name, m in linears.items()]
    try:
        hessians = q.collect_hessians(model, windows, batch, device)
    finally:
        for handle in handles:
            handle.remove()
    return hessians, moments, {name: abs_sum[name] / rows[name] for name in linears}


def awq_scales(mean_abs: Tensor, alpha: float) -> Tensor:
    """FP32 channel scales ``mean(|x_j|)^alpha`` of a layer stack (L, K), geometric mean 1 per layer.

    Channels without input energy get scale 1. ``alpha = 0`` gives exactly 1 everywhere.
    """
    mean_abs = mean_abs.double()
    if alpha == 0:
        return torch.ones_like(mean_abs, dtype=torch.float32)
    live = mean_abs > 0
    logs = torch.where(live, mean_abs.clamp_min(torch.finfo(torch.float64).tiny).log(), torch.zeros_like(mean_abs))
    center = logs.sum(-1, keepdim=True) / live.sum(-1, keepdim=True).clamp_min(1)
    return torch.where(live, torch.exp(alpha * (logs - center)), torch.ones_like(mean_abs)).float()


def scale_hessian(H: Tensor, scale: Tensor) -> Tensor:
    """Hessian of the scaled input x / s: D^-1 H D^-1 (FP64)."""
    s = scale.double()
    return H / (s[:, :, None] * s[:, None, :])


class ScaledRotation(q.Rotation):
    """``x -> (x / s) R^T``: the input map of an activation-aware scaled, rotated layer."""

    def __init__(self, rotation: q.Rotation, inverse_scale: Tensor):
        self.rotation, self.inverse_scale = rotation, inverse_scale

    def forward(self, x: Tensor) -> Tensor:
        return self.rotation.forward(x * self.inverse_scale)


class ScaledInput(nn.Module):
    """Divides the FP32 input by the channel scales, then calls the hardware layer."""

    def __init__(self, layer: nn.Module, inverse_scale: Tensor):
        super().__init__()
        self.layer = layer
        self.register_buffer("inverse_scale", inverse_scale.float())

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x.float() * self.inverse_scale)


def stable_rank(W: Tensor) -> Tensor:
    """||W||_F^2 / sigma_1^2 per layer of a stack, from the FP64 eigenvalues of the smaller Gram matrix."""
    W = W.double()
    gram = W @ W.mT if W.shape[-2] <= W.shape[-1] else W.mT @ W
    return gram.diagonal(dim1=-2, dim2=-1).sum(-1) / torch.linalg.eigvalsh(gram)[..., -1]


def alignment(B: Tensor, H: Tensor) -> Tensor:
    """tr(B H B^T) / (||B||_F^2 tr(H) / d) per layer: 1 when B ignores the input covariance."""
    B = B.double()
    energy = B.square().sum(dim=(-2, -1))
    reference = energy * H.diagonal(dim1=-2, dim2=-1).mean(-1)
    return torch.where(reference > 0, ((B @ H) * B).sum(dim=(-2, -1)) / reference, torch.zeros_like(energy))


def error_geometry(W: Tensor, W_hat: Tensor, H: Tensor) -> dict[str, list[float]]:
    """Per-layer weight error, error alignment and output error of a quantized stack.

    ``output_rel_error = weight_rel_error * error_alignment / signal_alignment`` holds exactly.
    """
    E = W.double() - W_hat.double()
    signal = ((W.double() @ H) * W.double()).sum(dim=(-2, -1))
    return {"weight_rel_error": (E.square().sum(dim=(-2, -1)) / W.double().square().sum(dim=(-2, -1))).tolist(),
            "error_alignment": alignment(E, H).tolist(),
            "output_rel_error": (((E @ H) * E).sum(dim=(-2, -1)) / signal).tolist()}


def summarize_layers(per_layer: list[dict]) -> dict:
    """Mean / median / min / max of every layer metric over all layers and per projection kind."""
    return {metric: {"all": q._summary([r[metric] for r in per_layer]),
                     "by_kind": {kind: q._summary([r[metric] for r in per_layer if r["kind"] == kind])
                                 for kind in q.LINEAR_KINDS}}
            for metric in LAYER_METRICS}


def layer_names(layers: int) -> list[str]:
    """Kind-major layer order of every per-layer list in a checkpoint record."""
    return [f"blocks.{i}.{kind}" for kind in q.LINEAR_KINDS for i in range(layers)]


def select_alpha(alphas: list[float], val_nll: list[float]) -> float:
    """The alpha with the lowest val.bin NLL; a tie goes to the smaller alpha."""
    return min(zip(val_nll, alphas))[1]


def best_methods(rows: list[dict], methods: list[str]) -> dict:
    """Per (format, mode) cell: the method of ``methods`` with the lowest summed val NLL, and the same on test."""
    cells: dict[tuple, dict] = {}
    for r in rows:
        if r["method"] in methods:
            entry = cells.setdefault((r["format"], r["mode"]), {}).setdefault(r["method"], {"val": [], "test": []})
            for split in SPLITS:
                entry[split].append(r[f"{split}_nll"])
    out = {}
    for (fmt, mode), totals in cells.items():
        by = {split: min(totals, key=lambda m: math.fsum(totals[m][split])) for split in SPLITS}
        out[f"{fmt}/{mode}"] = {"format": fmt, "mode": mode, "methods": list(totals), "best_by_val": by["val"],
                                "best_by_test": by["test"], "confirmed_on_test": by["val"] == by["test"]}
    return out


@torch.no_grad()
def fp4_gemms_per_forward(model, tokens: Tensor) -> int:
    """``aten::_scaled_mm_v2`` dispatches of one forward pass, counted by the profiler."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        with torch.autocast(device_type=tokens.device.type, dtype=torch.bfloat16):
            model(tokens)
    return sum(event.count for event in profile.key_averages() if event.key == "aten::_scaled_mm_v2")


def checkpoint_study(model, arm: str, config: h.ModelConfig, view: dict, device: torch.device, calibration: dict,
                     protocol: dict, log, label: str) -> dict:
    """Baselines, layer diagnostics and the PTQ grid of one saved checkpoint. Weights are restored."""
    t0 = time.perf_counter()
    seq, batch, windows = config.sequence_length, config.micro_batch, protocol["max_eval_windows"]
    val, test = view["val"], view["test"]
    sha_before = e2e._checkpoint_sha256(model)
    awq_spec = protocol["awq"]

    def evaluate(split: str) -> dict:
        return q.evaluate_split(model, val if split == "val" else test, seq, batch, device, windows)

    def evaluate_both(val_result: dict | None = None) -> dict:
        return {"val": val_result or evaluate("val"), "test": evaluate("test")}

    baseline = evaluate_both()
    log(f"  B {label} BF16 val NLL {baseline['val']['nll']:.4f} test NLL {baseline['test']['nll']:.4f}")
    hessians, moments, mean_abs = collect_calibration_statistics(model, calibration["array"], batch, device)
    linears = q.quantized_linears(model)
    kinds = {k: [f"blocks.{i}.{k}" for i in range(config.layers)] for k in q.LINEAR_KINDS}
    names = layer_names(config.layers)
    weights = {n: m.weight.detach().float().clone() for n, m in linears.items()}
    W = {k: torch.stack([weights[n] for n in ns]) for k, ns in kinds.items()}
    Hc = {k: torch.stack([hessians[n] for n in ns]) for k, ns in kinds.items()}
    A = {k: torch.stack([mean_abs[n] for n in ns]) for k, ns in kinds.items()}
    signal = {k: alignment(W[k], Hc[k]).tolist() for k in kinds}
    rank = {k: stable_rank(W[k]).tolist() for k in kinds}
    per_layer = []
    for kind, ns in kinds.items():
        for i, name in enumerate(ns):
            act, diag, w = moments[name].result(), hessians[name].diagonal(), weights[name]
            per_layer.append({"name": name, "kind": kind,
                              "weight_excess_kurtosis": q.excess_kurtosis(w), "weight_max_over_rms": q.max_over_rms(w),
                              "activation_excess_kurtosis": act["excess_kurtosis"],
                              "activation_max_over_rms": act["max_over_rms"],
                              "hessian_diag_max_over_mean": (diag.max() / diag.mean()).item(),
                              "weight_fro_norm": w.double().square().sum().sqrt().item(),
                              "weight_stable_rank": rank[kind][i], "signal_alignment": signal[kind][i]})
    del hessians, moments, mean_abs
    hardware_ok = c4.hardware_w4a4_available(device)
    probe = torch.from_numpy(calibration["array"][:2]).to(device)
    baseline_gemms = fp4_gemms_per_forward(model, probe) if hardware_ok else None

    def apply(source: dict[str, Tensor]) -> None:
        with torch.no_grad():
            for kind, ns in kinds.items():
                for i, name in enumerate(ns):
                    linears[name].weight.copy_(source[kind][i])

    def row(fmt_name: str, method: str, mode: str, evals: dict, seconds: float, **extra) -> dict:
        family, rounding, awq = parse_method(method)
        out = {"format": fmt_name, "method": method, "mode": mode, "rotation": family, "rounding": rounding,
               "activation_aware_scaling": awq}
        for split in SPLITS:
            result, base = evals[split], baseline[split]
            if result["tokens"] != base["tokens"]:
                raise RuntimeError("Evaluation token count changed between runs")
            out.update({f"{split}_nll": result["nll"], f"{split}_ppl": result["ppl"],
                        f"{split}_tokens": result["tokens"], f"{split}_delta_nll": result["nll"] - base["nll"],
                        f"{split}_relative_delta_ppl_pct": 100.0 * (result["ppl"] / base["ppl"] - 1.0)})
        out.update(seconds=seconds, **extra)
        log(f"  B {label} {fmt_name:<13} {method:<32} {mode:<26} val dNLL {out['val_delta_nll']:+.4f} "
            f"test dNLL {out['test_delta_nll']:+.4f} ({out['test_relative_delta_ppl_pct']:+.2f}%)")
        return out

    def quantize(fmt, family: str, rounding: str, alpha: float) -> dict:
        """One candidate: quantized stacks of every kind in scaled + rotated coordinates, and back."""
        t1 = time.perf_counter()
        cand = {"alpha": alpha, "family": family, "rotations": {}, "scales": {}, "rotated": {}, "encodings": {},
                "effective": {}, "fits": {}}
        damping, geometry = [], {key: [] for key in GEOMETRY}
        for kind in kinds:
            scale = awq_scales(A[kind], alpha) if alpha else None
            Ws = W[kind] if scale is None else W[kind] * scale[:, None, :]
            if family == "adaptive_butterfly_b16":
                rot, cand["fits"][kind] = q.make_rotation(family, Ws, fmt, butterfly_steps=protocol["butterfly_steps"])
            else:
                rot = q.Rotation() if family == "identity" else q.BlockHadamard(16)
            Hr = None
            if rounding == "gptq":
                Hr = rot.rotate_hessian(Hc[kind] if scale is None else scale_hessian(Hc[kind], scale))
            Qr, encoding, lam = c4._quantize(rot.forward(Ws), Hr, fmt, rounding)
            eff = rot.inverse(Qr)
            eff = eff if scale is None else eff / scale[:, None, :]
            cand["rotations"][kind], cand["scales"][kind] = rot, scale
            cand["rotated"][kind], cand["encodings"][kind], cand["effective"][kind] = Qr, encoding, eff
            damping += lam
            for key, values in error_geometry(W[kind], eff, Hc[kind]).items():
                geometry[key] += values
        cand["extra"] = {"quantize_seconds": time.perf_counter() - t1,
                         "gptq_damping_max": max(damping) if damping else None, "geometry": geometry,
                         **{f"{key}_mean": math.fsum(values) / len(values) for key, values in geometry.items()}}
        return cand

    def hardware_layers(cand: dict) -> tuple[dict, dict, dict]:
        """NVFP4Linear layers of a candidate; every packed weight is decoded and compared (int32 views)."""
        layers, modules, checked, exact = {}, {}, 0, True
        for kind, ns in kinds.items():
            rot, enc, scale = cand["rotations"][kind], cand["encodings"][kind], cand["scales"][kind]
            for i, name in enumerate(ns):
                layer = nb.NVFP4Linear.from_encoding(
                    enc["codes"][i], enc["block_scale"][i], enc["global_scale"][i],
                    rotation=e2e.ROTATION_OF[cand["family"]],
                    angles=rot.angles[i] if isinstance(rot, q.Butterfly) else None)
                checked += 1
                exact &= bool(torch.equal(nb.dequantized_weight(layer).view(torch.int32),
                                          cand["rotated"][kind][i].contiguous().view(torch.int32)))
                layers[name] = layer
                modules[name] = layer if scale is None else ScaledInput(layer, 1.0 / scale[i])
        return layers, modules, {"hw_weights_checked_matrices": checked,
                                 "hw_weights_decode_to_fakequant_weights": exact}

    def input_maps(cand: dict) -> dict:
        maps = {}
        for kind, ns in kinds.items():
            rot, scale = cand["rotations"][kind], cand["scales"][kind]
            for i, name in enumerate(ns):
                layer_rot = e2e._layer_rotation(rot, i)
                maps[name] = layer_rot if scale is None else ScaledRotation(layer_rot, 1.0 / scale[i])
        return maps

    def hardware_rows(fmt_name: str, method: str, cand: dict, layers: dict, modules: dict, **extra) -> list[dict]:
        rows = []
        t1 = time.perf_counter()  # dynamic per-tensor activation scale
        previous = e2e._swap(model, modules)
        gemms = fp4_gemms_per_forward(model, probe)
        evals = evaluate_both()
        e2e._swap(model, previous)
        rows.append(row(fmt_name, method, "w4a4_hw_dynamic", evals, time.perf_counter() - t1,
                        fp4_gemms_per_forward=gemms, **extra))
        t1 = time.perf_counter()  # static activation scales from the train.bin calibration windows
        static = e2e.calibrate_activation_scales(model, calibration["array"], batch, device, input_maps(cand))
        for name, layer in layers.items():
            layer.act_global_scale = static[name]
        previous = e2e._swap(model, modules)
        gemms = fp4_gemms_per_forward(model, probe)
        evals = evaluate_both()
        e2e._swap(model, previous)
        scales = [v.item() for v in static.values()]
        rows.append(row(fmt_name, method, "w4a4_hw_static", evals, time.perf_counter() - t1,
                        fp4_gemms_per_forward=gemms, static_scale_calibration=STATIC_CALIBRATION,
                        static_global_scale_range=[min(scales), max(scales)], **extra))
        for layer in layers.values():
            layer.act_global_scale = None
        if method == EMULATED_METHOD:  # same W4A4 operands, FP32 GEMM: isolates the FP4 hardware GEMM
            t1 = time.perf_counter()
            previous = e2e._swap(model, {n: e2e.EmulatedW4A4Linear(m) for n, m in layers.items()})
            gemms = fp4_gemms_per_forward(model, probe)
            evals = evaluate_both()
            e2e._swap(model, previous)
            rows.append(row(fmt_name, method, EMULATED_MODE, evals, time.perf_counter() - t1,
                            fp4_gemms_per_forward=gemms))
        return rows

    def finish(fmt_name: str, method: str, cand: dict, val_result: dict | None = None, **extra) -> list[dict]:
        """The fake-quant row of a candidate and, for the hardware format, its W4A4 rows.

        ``val_result`` is the val.bin result of the fake-quantized candidate when it is already known.
        """
        info = {**cand["extra"], **extra}
        layers = modules = None
        if fmt_name == HW_FORMAT:
            layers, modules, decode = hardware_layers(cand)
            info.update(decode)
        t1 = time.perf_counter()
        apply(cand["effective"])
        evals = evaluate_both(val_result)
        apply(W)
        rows = [row(fmt_name, method, fakequant_mode(fmt_name), evals, time.perf_counter() - t1, **info)]
        if layers is not None and hardware_ok:
            rows += hardware_rows(fmt_name, method, cand, layers, modules, **extra)
        return rows

    grid, butterfly_fits, awq_selection = [], {}, []
    for fmt_name in protocol["weight_formats"]:
        fmt = q.FORMATS[fmt_name]
        plain = {}
        for method in protocol["base_methods"]:
            family, rounding, _ = parse_method(method)
            cand = quantize(fmt, family, rounding, 0.0)
            if cand["fits"]:
                butterfly_fits[f"{fmt_name}/{method}"] = cand["fits"]
            rows = finish(fmt_name, method, cand)
            grid += rows
            if arm in awq_spec["arms"] and rounding == "gptq" and family in awq_spec["rotations"]:
                plain[family] = (cand, rows)
            del cand
        for family in (awq_spec["rotations"] if arm in awq_spec["arms"] else ()):
            method = f"awq_{family}_gptq"
            t1 = time.perf_counter()
            candidates = {a: plain[family][0] if a == 0 else quantize(fmt, family, "gptq", a)
                          for a in awq_spec["alphas"]}
            on_val = {}
            for a in awq_spec["alphas"]:
                if a == 0:  # the unscaled model itself: the val.bin result of its fake-quant row
                    twin = plain[family][1][0]
                    on_val[a] = {"nll": twin["val_nll"], "ppl": twin["val_ppl"], "tokens": twin["val_tokens"]}
                else:
                    apply(candidates[a]["effective"])
                    on_val[a] = evaluate("val")
            apply(W)
            alpha = select_alpha(awq_spec["alphas"], [on_val[a]["nll"] for a in awq_spec["alphas"]])
            awq_selection.append({"format": fmt_name, "method": method, "rotation": family,
                                  "alphas": list(awq_spec["alphas"]),
                                  "val_nll": [on_val[a]["nll"] for a in awq_spec["alphas"]],
                                  "val_tokens": on_val[alpha]["tokens"], "selected_alpha": alpha,
                                  "selected_on": "val", "seconds": time.perf_counter() - t1})
            if alpha == 0:  # its rows are copied, not measured again
                grid += [{**r, "method": method, "activation_aware_scaling": True, "awq_alpha": alpha}
                         for r in plain[family][1] if r["mode"] != EMULATED_MODE]
            else:
                if candidates[alpha]["fits"]:
                    butterfly_fits[f"{fmt_name}/{method}"] = candidates[alpha]["fits"]
                grid += finish(fmt_name, method, candidates[alpha], val_result=on_val[alpha], awq_alpha=alpha)
            del candidates
        del plain
        if device.type == "cuda":
            torch.cuda.empty_cache()
    sha_after = e2e._checkpoint_sha256(model)
    if sha_after != sha_before:
        raise RuntimeError("The PTQ study did not restore the checkpoint weights")
    return {"checkpoint_sha256": sha_before, "checkpoint_sha256_after_ptq": sha_after,
            "baseline_val": baseline["val"], "baseline_test": baseline["test"],
            "diagnostics": {"layer_names": names, "per_layer": per_layer, "summary": summarize_layers(per_layer)},
            "butterfly_fits": butterfly_fits, "grid": grid, "awq_selection": awq_selection,
            "best": best_methods(grid, protocol["base_methods"]),
            "hardware_w4a4": {"available": hardware_ok, "fp4_gemms_per_forward_bf16_baseline": baseline_gemms,
                              "note": None if hardware_ok else "skipped: needs CUDA sm_100+ with F.scaled_mm"},
            "seconds": time.perf_counter() - t0}


# ======================================================================================
# Statistics (pure functions of the stored rows; ``validate_results`` recomputes them)
# ======================================================================================
def holm_bonferroni(pvalues: list[float]) -> list[float]:
    return c4.holm_bonferroni(pvalues)


def sign_flip_p(diffs: list[float]) -> float:
    """Exact two-sided sign-flip p-value of the mean: the share of the 2^n sign patterns whose
    |sum| is at least the observed |sum| (relative tolerance 1e-12 for ties)."""
    observed = abs(math.fsum(diffs))
    count = sum(abs(math.fsum(s * d for s, d in zip(signs, diffs))) >= observed * (1 - 1e-12)
                for signs in itertools.product((1.0, -1.0), repeat=len(diffs)))
    return count / 2 ** len(diffs)


def paired_stats(diffs: list[float]) -> dict:
    """Paired statistics of five differences (4 degrees of freedom).

    The Student-t CDF with df = 4 gives the two-sided p-value 1 - s (1 + u / 2) with
    u = 4 / (4 + t^2) and s = sqrt(1 - u). It is computed as u^2 (3 + u) / (4 (1 + s (1 + u / 2))),
    the same number without cancellation at large |t|. With sd = 0, p is 1 (mean 0) or 0.
    """
    if len(diffs) != N_SEEDS:
        raise ValueError(f"The exact df = {N_SEEDS - 1} formulas need exactly {N_SEEDS} paired differences")
    stats = c4.mean_sd(diffs)
    mean, sd = stats["mean"], stats["sd"]
    se = sd / math.sqrt(N_SEEDS)
    if sd > 0:
        t = mean / se
        u = 4 / (4 + t * t)
        s = math.sqrt(1 - u)
        p = u * u * (3 + u) / (4 * (1 + s * (1 + u / 2)))
    else:
        t, p = None, (1.0 if mean == 0 else 0.0)
    return {"n": N_SEEDS, "df": N_SEEDS - 1, "diffs": list(diffs), "mean": mean, "sd": sd, "se": se, "t": t,
            "ci95": [mean - T975_DF4 * se, mean + T975_DF4 * se], "p_two_sided": p,
            "p_sign_flip": sign_flip_p(diffs)}


ARM_CONTRASTS = (("muon_vs_adamw", MUON, ADAMW),
                 ("adamw_cooldown_vs_muon", COOL_ADAMW, MUON), ("adamw_cooldown_vs_adamw", COOL_ADAMW, ADAMW),
                 ("qat_cooldown_vs_muon", COOL_QAT, MUON), ("qat_cooldown_vs_adamw", COOL_QAT, ADAMW))
AWQ_CONTRASTS = (("awq_vs_muon", MUON, MUON), ("awq_vs_adamw", MUON, ADAMW), ("awq_on_adamw", ADAMW, ADAMW))
FAMILIES = ("primary", "own_best")  # Holm-Bonferroni families of every contrast


def _tables(stage_b: dict, seeds: list[int]) -> dict:
    """(seed, arm) -> {(format, method, mode): row}, with completeness checks."""
    ckpt = {(c["seed"], c["arm"]): c for c in stage_b["checkpoints"]}
    if len(ckpt) != len(seeds) * len(ARMS) or any((s, a) not in ckpt for s in seeds for a in ARMS):
        raise ValueError("Stage B needs exactly one checkpoint per (seed, arm)")
    tables = {}
    for key, c in ckpt.items():
        rows = {(r["format"], r["method"], r["mode"]): r for r in c["grid"]}
        if len(rows) != len(c["grid"]):
            raise ValueError("Duplicate PTQ grid row")
        tables[key] = rows
    for arm in ARMS:
        if any(list(tables[(s, arm)]) != list(tables[(seeds[0], arm)]) for s in seeds):
            raise ValueError("PTQ grids differ between the seeds of an arm")
    return tables


def _cell_stats(rows_x: list[dict], rows_y: list[dict]) -> dict:
    """Paired statistics x - y of one cell, per split and endpoint."""
    out = {}
    for split in SPLITS:
        block = {}
        for field in ("delta_nll", "relative_delta_ppl_pct", "nll"):
            block[f"x_{field}"] = c4.mean_sd([r[f"{split}_{field}"] for r in rows_x])
            block[f"y_{field}"] = c4.mean_sd([r[f"{split}_{field}"] for r in rows_y])
        for field in ENDPOINTS:
            block[f"{field}_diff"] = paired_stats([x[f"{split}_{field}"] - y[f"{split}_{field}"]
                                                   for x, y in zip(rows_x, rows_y)])
        out[split] = block
    return out


def _holm_cells(cells: list[dict], key: str) -> None:
    for split in SPLITS:
        for field in ENDPOINTS:
            blocks = [c[split][f"{field}_diff"] for c in cells]
            for block, p in zip(blocks, holm_bonferroni([b["p_two_sided"] for b in blocks])):
                block[key] = p


def analyze(stage_a: dict, stage_b: dict, protocol: dict) -> dict:
    """Paired statistics over the five seeds. Every difference is x - y of the named contrast."""
    seeds, tolerance = protocol["seeds"], protocol["tolerance_nats"]
    tables = _tables(stage_b, seeds)
    ckpt = {(c["seed"], c["arm"]): c for c in stage_b["checkpoints"]}
    pilot = {s["seed"]: s for s in stage_a["seeds"]}

    # Per-seed iso-loss match: Stage A selection (val only), confirmed on the saved checkpoints.
    pairs = []
    for seed in seeds:
        m, a, sel = ckpt[(seed, MUON)], ckpt[(seed, ADAMW)], pilot[seed]["selection"]
        rows = {arm: next(r for r in pilot[seed]["rows"]
                          if r["checkpoint_sha256"] == ckpt[(seed, arm)]["checkpoint_sha256"]) for arm in ARMS}
        pair = {"seed": seed, "t_adamw": sel["t_adamw"], "adamw_floor_lr_steps": sel["adamw_floor_lr_steps"],
                "adamw_evaluations": sel["adamw_evaluations"],
                "adamw_over_muon_tokens": sel["adamw_over_muon_tokens"], "stage_a_matched": sel["matched"],
                "stage_a_abs_val_gap": sel["abs_val_gap"],
                "same_init_and_stream": len({(r["init_fingerprint"], r["stream_fingerprint"])
                                             for r in pilot[seed]["rows"]}) == 1,
                "checkpoint_sha256": {arm: ckpt[(seed, arm)]["checkpoint_sha256"] for arm in ARMS},
                "stage_b_val_equals_stage_a": all(
                    abs(ckpt[(seed, arm)]["baseline_val"]["nll"] - rows[arm]["val_nll"]) <= 1e-9 for arm in ARMS)}
        for split in SPLITS:
            vm, va = m[f"baseline_{split}"]["nll"], a[f"baseline_{split}"]["nll"]
            pair.update({f"{split}_nll_muon": vm, f"{split}_nll_adamw": va,
                         f"{split}_gap_muon_minus_adamw": vm - va, f"abs_{split}_gap": abs(vm - va),
                         f"{split}_gap_within_tolerance": abs(vm - va) <= tolerance})
        pairs.append(pair)
    iso = {"tolerance_nats": tolerance, "pairs": pairs,
           "seeds_matched_on_val": sum(p["val_gap_within_tolerance"] for p in pairs),
           "seeds_confirmed_on_test": sum(p["test_gap_within_tolerance"] for p in pairs),
           "all_seeds_matched_on_val": all(p["val_gap_within_tolerance"] for p in pairs),
           "all_seeds_confirmed_on_test": all(p["test_gap_within_tolerance"] for p in pairs)}
    for split in SPLITS:
        iso[f"{split}_gap_muon_minus_adamw"] = paired_stats([p[f"{split}_gap_muon_minus_adamw"] for p in pairs])
        iso[f"mean_abs_{split}_gap"] = math.fsum(p[f"abs_{split}_gap"] for p in pairs) / len(pairs)
        iso[f"max_abs_{split}_gap"] = max(p[f"abs_{split}_gap"] for p in pairs)

    # Method selection on val, pooled over every arm and seed (base methods) and over the AWQ arms (AWQ methods).
    base, awq_methods = protocol["base_methods"], [f"awq_{r}_gptq" for r in protocol["awq"]["rotations"]]
    pooled = best_methods([r for c in stage_b["checkpoints"] for r in c["grid"]], base)
    pooled_awq = best_methods([r for c in stage_b["checkpoints"] if c["arm"] in protocol["awq"]["arms"]
                               for r in c["grid"]], awq_methods)
    own = {arm: best_methods([r for s in seeds for r in ckpt[(s, arm)]["grid"]], base) for arm in ARMS}
    own_awq = {arm: best_methods([r for s in seeds for r in ckpt[(s, arm)]["grid"]], awq_methods)
               for arm in protocol["awq"]["arms"]}
    selection = {"base_pooled": pooled, "awq_pooled": pooled_awq, "per_arm": own, "per_arm_awq": own_awq}
    primary_keys = [key for key, cell in pooled.items() if len(cell["methods"]) > 1]

    def families(method_x, method_y, own_x, own_y) -> dict:
        """Primary cells with one method for both sides, and with each side's own val-selected method."""
        cell = lambda k: {"format": pooled[k]["format"], "mode": pooled[k]["mode"]}  # noqa: E731
        return {"primary": [{**cell(k), "method_x": method_x[k]["best_by_val"],
                             "method_y": method_y[k]["best_by_val"]} for k in primary_keys],
                "own_best": [{**cell(k), "method_x": own_x[k]["best_by_val"],
                              "method_y": own_y[k]["best_by_val"]} for k in primary_keys]}

    def rows_of(arm, combo):
        return [tables[(s, arm)][combo] for s in seeds]

    contrasts = []
    for name, arm_x, arm_y in ARM_CONTRASTS:
        cells = [{"format": f, "mode": mode, "method_x": method, "method_y": method,
                  **_cell_stats(rows_of(arm_x, (f, method, mode)), rows_of(arm_y, (f, method, mode)))}
                 for (f, method, mode) in tables[(seeds[0], arm_x)] if (f, method, mode) in tables[(seeds[0], arm_y)]]
        bf16 = {split: paired_stats([ckpt[(s, arm_x)][f"baseline_{split}"]["nll"]
                                     - ckpt[(s, arm_y)][f"baseline_{split}"]["nll"] for s in seeds])
                for split in SPLITS}
        contrasts.append({"name": name, "kind": "arm", "arm_x": arm_x, "arm_y": arm_y, "bf16_nll_diff": bf16,
                          "cells": cells, **families(pooled, pooled, own[arm_x], own[arm_y])})
    for name, arm_x, arm_y in AWQ_CONTRASTS:
        cells = [{"format": f, "mode": mode, "method_x": method, "method_y": method.removeprefix("awq_"),
                  **_cell_stats(rows_of(arm_x, (f, method, mode)),
                                rows_of(arm_y, (f, method.removeprefix("awq_"), mode)))}
                 for (f, method, mode) in tables[(seeds[0], arm_x)] if method in awq_methods]
        contrasts.append({"name": name, "kind": "awq", "arm_x": arm_x, "arm_y": arm_y, "bf16_nll_diff": None,
                          "cells": cells, **families(pooled_awq, pooled, own_awq[arm_x], own[arm_y])})
    for contrast in contrasts:
        _holm_cells(contrast["cells"], "p_holm_all_cells")
        for family in FAMILIES:
            for entry in contrast[family]:
                entry.update(_cell_stats(
                    rows_of(contrast["arm_x"], (entry["format"], entry["method_x"], entry["mode"])),
                    rows_of(contrast["arm_y"], (entry["format"], entry["method_y"], entry["mode"]))))
            _holm_cells(contrast[family], "p_holm")
            for entry in contrast[family]:
                for split in SPLITS:
                    for field in ENDPOINTS:
                        block = entry[split][f"{field}_diff"]
                        block["significant_holm_0p05"] = block["p_holm"] < 0.05
    for family in FAMILIES:
        _holm_cells([entry for contrast in contrasts for entry in contrast[family]], "p_holm_across_contrasts")

    # Hardware FP4 GEMM against the FP32 GEMM of the same quantized operands.
    emulation = []
    for (seed, arm), rows in tables.items():
        hw = rows.get((HW_FORMAT, EMULATED_METHOD, "w4a4_hw_dynamic"))
        emu = rows.get((HW_FORMAT, EMULATED_METHOD, EMULATED_MODE))
        if hw and emu:
            emulation.append({"seed": seed, "arm": arm,
                              **{f"{split}_nll_hw_minus_emulated": hw[f"{split}_nll"] - emu[f"{split}_nll"]
                                 for split in SPLITS}})
    worst = max((abs(e[f"{s}_nll_hw_minus_emulated"]) for e in emulation for s in SPLITS), default=None)
    parity = {"method": EMULATED_METHOD, "rows": emulation, "max_abs_nll_difference": worst,
              "threshold_nats": EMULATION_PARITY_NATS, "passed": worst is not None and worst <= EMULATION_PARITY_NATS}
    hw_rows = [r for rows in tables.values() for r in rows.values() if r["mode"] in HW_MODES]
    decode_rows = [r for rows in tables.values() for r in rows.values() if "hw_weights_checked_matrices" in r]
    hardware = {"w4a4_rows": len(hw_rows),
                "fp4_gemms_per_forward": sorted({r["fp4_gemms_per_forward"] for r in hw_rows}),
                "fp4_gemms_per_forward_emulated": sorted({r["fp4_gemms_per_forward"] for rows in tables.values()
                                                          for r in rows.values() if r["mode"] == EMULATED_MODE}),
                "fp4_gemms_per_forward_bf16_baseline": sorted(
                    {c["hardware_w4a4"]["fp4_gemms_per_forward_bf16_baseline"] for c in stage_b["checkpoints"]
                     if c["hardware_w4a4"]["available"]}),
                "decode_checked_rows": len(decode_rows),
                "decode_checked_matrices": sum(r["hw_weights_checked_matrices"] for r in decode_rows),
                "decode_exact_rows": sum(r["hw_weights_decode_to_fakequant_weights"] is True for r in decode_rows)}

    # Layer geometry: per-arm seed means of the all-layer mean, and paired tests for the arm contrasts.
    def layer_mean(seed, arm, metric):
        return ckpt[(seed, arm)]["diagnostics"]["summary"][metric]["all"]["mean"]

    layers = {metric: {"per_arm": {arm: c4.mean_sd([layer_mean(s, arm, metric) for s in seeds]) for arm in ARMS},
                       "contrasts": {name: paired_stats([layer_mean(s, x, metric) - layer_mean(s, y, metric)
                                                         for s in seeds]) for name, x, y in ARM_CONTRASTS}}
              for metric in LAYER_METRICS}
    quant = []
    for arm in ARMS:
        for (f, method, mode) in tables[(seeds[0], arm)]:
            if mode == fakequant_mode(f):
                quant.append({"arm": arm, "format": f, "method": method,
                              **{key: c4.mean_sd([tables[(s, arm)][(f, method, mode)][f"{key}_mean"] for s in seeds])
                                 for key in GEOMETRY}})
    def alpha_of(seed, arm, sel):
        return next(x for x in ckpt[(seed, arm)]["awq_selection"]
                    if (x["format"], x["method"]) == (sel["format"], sel["method"]))["selected_alpha"]

    alphas = [{"arm": arm, "format": sel["format"], "method": sel["method"],
               "selected_alpha": [alpha_of(s, arm, sel) for s in seeds]}
              for arm in protocol["awq"]["arms"] for sel in ckpt[(seeds[0], arm)]["awq_selection"]]
    return {"sign_convention": "every difference is x - y of the contrast (arm_x with method_x minus arm_y with "
                               "method_y); a negative delta_nll_diff means that x loses less quality under PTQ, a "
                               "negative nll_diff that its quantized NLL is lower",
            "t_quantile_0p975_df4": T975_DF4, "sign_flip_min_p": SIGN_FLIP_MIN_P,
            "inference_note": "p_two_sided is the exact df = 4 Student-t p-value of the paired differences; "
                              "p_sign_flip is the exact two-sided sign-flip permutation p-value over the 2^5 sign "
                              "patterns (it cannot be below 0.0625, so it never passes a Holm correction over "
                              "more than one cell). p_holm: Holm-Bonferroni over the cells of one family "
                              "('primary' or 'own_best') of one contrast, split and endpoint; "
                              "p_holm_across_contrasts: over that family of all contrasts; p_holm_all_cells: over "
                              "every cell of one contrast.",
            "primary_note": "The families cover the (format, mode) cells with more than one method. 'primary': "
                            "arm contrasts use the base method with the lowest val NLL pooled over all arms and "
                            "seeds for both sides; AWQ contrasts compare the val-selected AWQ method (pooled over "
                            "the AWQ arms) with that base method. 'own_best': each side uses the method with the "
                            "lowest val NLL of its own arm (AWQ contrasts: the arm's best AWQ method against the "
                            "other arm's best base method), i.e. the best deployable result of each side. "
                            "'cells' of an AWQ contrast compare each AWQ method with the same rotation without "
                            "scaling (alpha = 0).",
            "iso_loss": iso, "method_selection": selection, "contrasts": contrasts,
            "hardware_vs_emulation": parity, "hardware": hardware,
            "layer_geometry": layers, "quantization_geometry": quant, "awq_selected_alpha": alphas}


def derive(payload: dict) -> dict:
    """Headline numbers, copied from the stored stages and the analysis."""
    stage_a, stage_b, analysis = payload["stage_a"], payload["stage_b"], payload["analysis"]
    keys = ("matched", "t_adamw", "adamw_floor_lr_steps", "adamw_evaluations", "abs_val_gap",
            "val_gap_muon_minus_adamw", "adamw_over_muon_tokens")
    out = {"stage_a": [{"seed": s["seed"], **{k: s["selection"][k] for k in keys}} for s in stage_a["seeds"]],
           "checkpoints": len(stage_b["checkpoints"]),
           "grid_rows": sum(len(c["grid"]) for c in stage_b["checkpoints"]),
           "analysis_available": analysis is not None}
    if analysis is None:
        return out
    iso = analysis["iso_loss"]
    out["iso_loss"] = {k: iso[k] for k in ("seeds_matched_on_val", "seeds_confirmed_on_test",
                                           "all_seeds_matched_on_val", "all_seeds_confirmed_on_test",
                                           "mean_abs_val_gap", "max_abs_val_gap", "mean_abs_test_gap",
                                           "max_abs_test_gap")}
    out["iso_loss"]["abs_val_gap_per_seed"] = [p["abs_val_gap"] for p in iso["pairs"]]
    out["iso_loss"]["abs_test_gap_per_seed"] = [p["abs_test_gap"] for p in iso["pairs"]]
    for family in FAMILIES:
        out[f"{family}_test"] = {}
        for contrast in analysis["contrasts"]:
            rows = []
            for e in contrast[family]:
                t = e["test"]
                rows.append({"format": e["format"], "mode": e["mode"], "method_x": e["method_x"],
                             "method_y": e["method_y"], "x_delta_nll_mean": t["x_delta_nll"]["mean"],
                             "y_delta_nll_mean": t["y_delta_nll"]["mean"], "x_nll_mean": t["x_nll"]["mean"],
                             "y_nll_mean": t["y_nll"]["mean"],
                             **{f"{field}_diff_{key}": t[f"{field}_diff"][key] for field in ENDPOINTS
                                for key in ("mean", "ci95", "p_two_sided", "p_holm", "p_holm_across_contrasts",
                                            "p_sign_flip")}})
            out[f"{family}_test"][contrast["name"]] = {
                "arm_x": contrast["arm_x"], "arm_y": contrast["arm_y"], "cells": rows,
                **{f"{field}_diff_negative_cells": sum(r[f"{field}_diff_mean"] < 0 for r in rows)
                   for field in ENDPOINTS},
                **{f"{field}_diff_significant_holm_0p05": sum(
                    e["test"][f"{field}_diff"]["significant_holm_0p05"] for e in contrast[family])
                   for field in ENDPOINTS}}
    out["hardware"] = {**analysis["hardware"],
                       "emulation_max_abs_nll_difference": analysis["hardware_vs_emulation"]["max_abs_nll_difference"],
                       "emulation_parity_passed": analysis["hardware_vs_emulation"]["passed"]}
    out["layer_geometry_arm_means"] = {metric: {arm: analysis["layer_geometry"][metric]["per_arm"][arm]["mean"]
                                                for arm in ARMS}
                                       for metric in ("weight_stable_rank", "signal_alignment", "weight_fro_norm")}
    return out


def caveats(payload: dict) -> list[str]:
    analysis, protocol = payload["analysis"], payload["protocol"]
    notes = [
        "Five paired seeds give 4 degrees of freedom. The Student-t p-values assume normal paired differences; "
        "the exact sign-flip p-value cannot be below 0.0625 and therefore cannot pass a Holm correction over "
        "several cells. Absence of significance is not evidence of equal PTQ sensitivity.",
        "AdamW is one run per seed, stopped on val.bin, and the stopped weights are evaluated without "
        "retraining. The val match of Stage B is therefore the Stage A match by construction; test.bin is the "
        "independent confirmation. Training is not bit-reproducible on this host: fresh AdamW runs of one "
        "configuration differed by 0.06 to 0.07 nats on val.bin in the design pilot, so a retrain gives other "
        "weights and another stopping step.",
        "Iso-loss is not iso-compute and not iso-schedule: AdamW uses more training tokens than Muon "
        "(adamw_over_muon_tokens per seed) and ends with a seed-dependent number of floor-LR steps "
        "(adamw_floor_lr_steps), while Muon ends when its cosine reaches the floor.",
        "The AdamW cosine length and the first measured step come from a seed-0 design pilot (val.bin only). "
        "The AdamW step cap was raised from 1280 to 2560 after a first canonical attempt was stopped in Stage A "
        "(val.bin only, before test.bin was opened): the AdamW run of seed 2 was 0.58 nats above Muon at step 400 "
        "and unmatched at step 1280. In the pilot the PTQ loss of AdamW at LR 0.001 was about the same with and "
        "without the floor-LR phase at equal val NLL; it is about three times the PTQ loss of the C4-01b AdamW "
        "recipe (LR 0.0015), so the Muon-versus-AdamW ratio depends on the AdamW recipe.",
        "Both cooldown arms branch from snapshots of the seed's Muon run, so their first 256 / 272 steps are that "
        "run's steps. Their BF16 baselines differ from Muon; bf16_nll_diff and nll_diff report the net effect.",
        "The alpha of the activation-aware scaling is selected per checkpoint, format and rotation on val.bin "
        "with weight-only fake quantization; the hardware W4A4 rows reuse that alpha.",
        "One 50m model family, short training (less than one epoch of a 26M-token set), one calibration draw. "
        "PTQ sensitivity of longer-trained or larger models can differ.",
        "Hessians come from one forward pass of the full-precision model (parallel GPTQ). Static activation "
        "scales are calibrated on the BF16 model. Dynamic W4A4 scales are per forward batch (16 x 512 tokens).",
        "Layer-geometry p-values are descriptive and not corrected for multiplicity.",
    ]
    if analysis is None:
        notes.insert(0, f"No paired analysis: it needs exactly {N_SEEDS} seeds, this run has "
                        f"{len(protocol['seeds'])}.")
    elif not analysis["iso_loss"]["all_seeds_matched_on_val"]:
        notes.insert(0, "At least one seed has no measured AdamW step within the val tolerance "
                        f"(max |val gap| {analysis['iso_loss']['max_abs_val_gap']:.4f} nats). Its closest step "
                        "is used; the comparison of that seed is not quality matched.")
    return notes


# ======================================================================================
# Validation
# ======================================================================================
def _require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _assert_close(found, expected, path: str, tol: float = TOL) -> None:
    """Deep equality with exact ``int`` / ``bool`` / ``str`` / ``None`` types and close finite floats."""
    if isinstance(expected, dict):
        if not isinstance(found, dict) or list(found) != list(expected):
            raise ValueError(f"{path}: keys differ from the recomputed value")
        for key, value in expected.items():
            _assert_close(found[key], value, f"{path}.{key}", tol)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(found, list) or len(found) != len(expected):
            raise ValueError(f"{path}: length or type differs from the recomputed value")
        for i, value in enumerate(expected):
            _assert_close(found[i], value, f"{path}[{i}]", tol)
    elif isinstance(expected, (bool, str, int)) or expected is None:
        if type(found) is not type(expected) or found != expected:
            raise ValueError(f"{path}: {found!r} differs from the recomputed {expected!r}")
    else:
        if not _is_number(found) or not abs(found - expected) <= tol * max(1.0, abs(expected)):
            raise ValueError(f"{path}: {found!r} differs from the recomputed {expected!r}")


def _keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _keys(value)


def _has_nul(obj) -> bool:
    if isinstance(obj, str):
        return "\x00" in obj
    if isinstance(obj, dict):
        return any(_has_nul(k) or _has_nul(v) for k, v in obj.items())
    return isinstance(obj, list) and any(_has_nul(x) for x in obj)


def _require_git_commit(sha: str, label: str) -> None:
    kind = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-t", sha], capture_output=True, text=True)
    if kind.returncode != 0 or kind.stdout.strip() != "commit":
        raise ValueError(f"{label} {sha} is not a commit object")


@functools.cache
def _code_sha256_at_commit(git_sha: str, modules: tuple[str, ...]) -> str:
    _require_git_commit(git_sha, "git_sha")
    digest = hashlib.sha256()
    for module in modules:
        blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{git_sha}:{'/'.join(module.split('.'))}.py"],
                              capture_output=True)
        if blob.returncode != 0:
            raise ValueError(f"cannot read {module} at git_sha {git_sha}")
        digest.update(module.encode() + b"\0" + blob.stdout)
    return digest.hexdigest()


def _timestamp(text, field: str) -> datetime:
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"meta.{field}: invalid ISO timestamp {text!r}") from exc


META_KEYS = {"git_sha", "git_dirty", "command", "environment", "numerics", "code_sha256", "code_modules",
             "unit_config_hash", "checkpoint_dir", "timestamp_start", "timestamp_end", "loadavg_start",
             "loadavg_end", "gpu_telemetry", "stage_seconds", "total_seconds", "unit_log", "splits_opened"}
NUMERICS = {"allow_tf32": False, "cudnn_allow_tf32": False, "allow_fp16_reduced_precision_reduction": False,
            "allow_bf16_reduced_precision_reduction": False,
            "autocast": "bfloat16 forward on CUDA (harness_tier2_real._loss)"}


def _validate_meta(payload: dict, *, canonical: bool) -> None:
    meta = payload["meta"]
    _require(isinstance(meta, dict) and set(meta) == META_KEYS, "meta keys differ from the schema")
    _require(not _has_nul(meta), "meta strings must not contain NUL bytes")
    _require(meta["git_sha"] is None or _is_hex(meta["git_sha"], 40), "meta.git_sha must be 40-hex or None")
    _require(meta["git_dirty"] is None or isinstance(meta["git_dirty"], bool), "meta.git_dirty must be a bool")
    command = meta["command"]
    _require(isinstance(command, list) and len(command) >= 2 and all(isinstance(x, str) and x for x in command)
             and "python" in Path(command[0]).name.lower(), "meta.command must be a Python invocation")
    env = meta["environment"]
    _require(isinstance(env, dict) and isinstance(env.get("torch"), str) and isinstance(env.get("python"), str),
             "meta.environment invalid")
    _require(meta["numerics"] == NUMERICS, "full-precision numerics are required")
    modules = meta["code_modules"]
    _require(isinstance(modules, list) and all(isinstance(m, str) and m.startswith("research.") for m in modules)
             and len(set(modules)) == len(modules)
             and "research.quant_ptq.bench_c5_02_isoloss_ptq_mitigation" in modules,
             "meta.code_modules must list the driver and its research.* imports")
    _require(_is_hex(meta["code_sha256"], 64) and _is_hex(meta["unit_config_hash"], 16), "meta hashes invalid")
    data_sha = {k: v["sha256"] for k, v in payload["data"]["splits"].items()}
    _require(meta["unit_config_hash"] == unit_config_hash(payload["protocol"], payload["config"], meta["code_sha256"],
                                                          env, data_sha), "meta.unit_config_hash differs")
    stages = meta["stage_seconds"]
    _require(isinstance(stages, dict) and set(stages) == {"stage_a", "stage_b"}
             and all(_is_number(v) and v >= 0 for v in stages.values()), "meta.stage_seconds invalid")
    _require(_is_number(meta["total_seconds"]) and meta["total_seconds"] >= sum(stages.values()) - 1e-6,
             "meta.total_seconds is below the sum of the stage times")
    start = _timestamp(meta["timestamp_start"], "timestamp_start")
    end = _timestamp(meta["timestamp_end"], "timestamp_end")
    _require(start.timestamp() >= 1_700_000_000 and end >= start
             and abs((end - start).total_seconds() - meta["total_seconds"]) <= 5.0,
             "meta timestamps are inconsistent with meta.total_seconds")
    for field in ("loadavg_start", "loadavg_end"):
        _require(isinstance(meta[field], list) and len(meta[field]) == 3
                 and all(_is_number(x) and x >= 0 for x in meta[field]), f"meta.{field} invalid")
    _require(isinstance(meta["gpu_telemetry"], dict)
             and set(meta["gpu_telemetry"]) == {"start", "after_stage_a", "end"}, "meta.gpu_telemetry keys differ")
    log = meta["unit_log"]
    _require(isinstance(log, dict) and set(log) == {"resume", "units_loaded", "units_run"}
             and isinstance(log["resume"], bool) and all(type(log[k]) is int and log[k] >= 0
                                                         for k in ("units_loaded", "units_run"))
             and log["units_loaded"] + log["units_run"] == len(payload["protocol"]["seeds"]) * (1 + len(ARMS)),
             "meta.unit_log must account for one Stage A unit per seed and one Stage B unit per checkpoint")
    _require(meta["splits_opened"] == ["train", "val", "test"], "splits must be opened in the order train, val, test")
    _require(isinstance(meta["checkpoint_dir"], str) and meta["checkpoint_dir"], "meta.checkpoint_dir invalid")
    if canonical:
        _require(_is_hex(meta["git_sha"], 40) and meta["git_dirty"] is False,
                 "a canonical result needs a 40-hex git_sha and a clean tree")
        _require(all(env.get(k) is not None for k in ENV_KEYS) and env["capability"] >= [12, 0],
                 "a canonical result needs an sm_120+ CUDA device and a recorded runtime")
        _require(command[1].endswith(SCRIPT_NAME) and "--quick" not in command,
                 "meta.command is not the canonical invocation")
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                parsed = parse_args(command[2:])
        except SystemExit as exc:
            raise ValueError("meta.command is not a valid runnable invocation") from exc
        _require(parsed.canonical is True, "meta.command does not select the canonical protocol")
        _require(modules == list(code_modules()), "meta.code_modules differ from the static import closure")
        _require(meta["code_sha256"] == _code_sha256_at_commit(meta["git_sha"], tuple(modules)),
                 "meta.code_sha256 differs from the committed git blobs")
        for side in ("start", "after_stage_a", "end"):  # one nvidia-smi sample each, or its recorded error
            sample = meta["gpu_telemetry"][side]
            _require(isinstance(sample, dict) and ("error" in sample or _is_number(sample.get("memory.used"))),
                     f"meta.gpu_telemetry.{side} invalid")


TRAINING_KEYS = {"arm", "seed", "steps", "warmup_steps", "lr", "auxiliary_lr", "weight_decay", "min_lr_ratio",
                 "tokens", "train_epochs", "orthogonalizer", "init_fingerprint", "stream_fingerprint", "train_loss",
                 "last_train_loss", "train_loss_note", "train_curve", "train_seconds", "checkpoint_sha256",
                 "val_nll", "val_ppl", "val_tokens", "evaluated_splits"}


def _validate_training(row: dict, path: str, config: dict, protocol: dict, train_tokens: int) -> None:
    extra = {MUON: {"snapshots"}, ADAMW: {"cosine_steps", "trained_steps", "eval_seconds", "evaluations"},
             COOL_ADAMW: {"branch", "cooldown"}, COOL_QAT: {"branch", "cooldown"}}
    _require(row.get("arm") in ARMS and set(row) == TRAINING_KEYS | extra[row["arm"]], f"{path}: keys differ")
    arm, muon, total = row["arm"], protocol["muon"], protocol["muon"]["steps"]
    schedule_steps = protocol["adamw"]["cosine_steps"] if arm == ADAMW else total
    _require(type(row["steps"]) is int and row["warmup_steps"] == c4.warmup_for(schedule_steps)
             and row["weight_decay"] == WEIGHT_DECAY and row["min_lr_ratio"] == MIN_LR_RATIO,
             f"{path}: schedule differs from warmup max(20, T // 10), weight decay 0.01, min LR ratio 0.1")
    step_tokens = config["micro_batch"] * config["grad_accum"] * config["sequence_length"]
    _require(row["tokens"] == row["steps"] * step_tokens, f"{path}: token count differs from steps * {step_tokens}")
    _assert_close(row["train_epochs"], row["tokens"] / train_tokens, f"{path}.train_epochs")
    if arm == ADAMW:
        spec = protocol["adamw"]
        _require(row["lr"] == spec["lr"] and row["auxiliary_lr"] is None
                 and row["cosine_steps"] == spec["cosine_steps"], f"{path}: AdamW run differs from the protocol")
    else:
        _require(row["steps"] == total and row["lr"] == muon["lr"] and row["auxiliary_lr"] == muon["auxiliary_lr"],
                 f"{path}: Muon-based arm differs from the protocol")
    _require(_is_hex(row["checkpoint_sha256"], 64) and _is_hex(row["init_fingerprint"], 16)
             and _is_hex(row["stream_fingerprint"], 16), f"{path}: missing hashes")
    _require(row["evaluated_splits"] == ["val"] and _is_number(row["val_nll"]), f"{path}: Stage A rows read val only")
    _assert_close(row["val_ppl"], math.exp(row["val_nll"]), f"{path}.val_ppl")
    curve = row["train_curve"]
    first = 0 if arm in (MUON, ADAMW) else row["branch"]["from_step"]
    count = row["steps"] - first
    _require(isinstance(curve, list) and [p[0] for p in curve] == [first + i + 1 for i in range(count)
                                                                  if i == 0 or (i + 1) % 10 == 0]
             and all(_is_number(p[1]) for p in curve), f"{path}: train curve differs from the schedule")


def _validate_adamw_run(row: dict, muon: dict, path: str, spec: dict) -> None:
    """Replay the predeclared val grid and stop rule of the AdamW run from its stored measurements."""
    evaluations = row["evaluations"]
    _require(isinstance(evaluations, list) and evaluations, f"{path}: the AdamW run has no measured step")
    grid = adamw_eval_steps(spec)
    _require([e["step"] for e in evaluations] == grid[:len(evaluations)],
             f"{path}: measured AdamW steps differ from the predeclared val grid")
    for e in evaluations:
        _require(set(e) == {"step", "lr_multiplier", "val_nll", "val_tokens", "val_gap", "checkpoint_sha256"}
                 and _is_number(e["val_nll"]) and _is_hex(e["checkpoint_sha256"], 64)
                 and e["val_tokens"] == muon["val_tokens"], f"{path}: AdamW measurement fields differ")
        _assert_close(e["val_gap"], e["val_nll"] - muon["val_nll"], f"{path}.evaluations[{e['step']}].val_gap")
        _assert_close(e["lr_multiplier"], adamw_multiplier(e["step"] - 1, spec),
                      f"{path}.evaluations[{e['step']}].lr_multiplier")
    gaps = [e["val_nll"] - muon["val_nll"] for e in evaluations]  # recomputed, as in ``select_adamw``
    _require(not any(adamw_stops(g) for g in gaps[:-1]),
             f"{path}: the AdamW run went on after it was at or below Muon")
    _require(adamw_stops(gaps[-1]) or evaluations[-1]["step"] == grid[-1],
             f"{path}: the AdamW run stopped before it reached Muon or the step cap")
    chosen = min(zip(evaluations, gaps), key=lambda pair: (abs(pair[1]), pair[0]["step"]))[0]
    _require(row["trained_steps"] == evaluations[-1]["step"]
             and (row["steps"], row["val_nll"], row["val_tokens"], row["checkpoint_sha256"])
             == (chosen["step"], chosen["val_nll"], chosen["val_tokens"], chosen["checkpoint_sha256"]),
             f"{path}: the AdamW record is not the measured step nearest to Muon")


def _validate_stage_a(payload: dict, *, canonical: bool) -> dict:
    stage, protocol, config = payload["stage_a"], payload["protocol"], payload["config"]
    seeds, tolerance = protocol["seeds"], protocol["tolerance_nats"]
    train_tokens = payload["data"]["splits"]["train"]["tokens"]
    _require(set(stage) == {"split_view", "splits_opened_at_end", "seeds", "seconds"}, "stage_a keys differ")
    _require(stage["split_view"] == ["train", "val"] and stage["splits_opened_at_end"] == ["train", "val"],
             "Stage A must run on a train + val mapping before test.bin is opened")
    _require([s["seed"] for s in stage["seeds"]] == seeds, "Stage A needs one unit per seed, in order")
    _require(not any("test" in key for key in _keys(stage)), "Stage A holds a test.bin field")
    cool = {COOL_ADAMW: protocol["adamw_cooldown"], COOL_QAT: protocol["qat_cooldown"]}
    by_seed = {}
    for unit in stage["seeds"]:
        seed, rows = unit["seed"], unit["rows"]
        path = f"stage_a[{seed}]"
        _require(set(unit) == {"seed", "split_view", "rows", "selection", "checkpoints", "seconds",
                               "resumed_from_unit_log"}, f"{path}: keys differ")
        _require(unit["split_view"] == ["train", "val"], f"{path}: Stage A must not see test.bin")
        _require([r["arm"] for r in rows] == [MUON, COOL_ADAMW, COOL_QAT, ADAMW],
                 f"{path}: rows must be Muon, the two cooldowns and the AdamW run")
        for i, r in enumerate(rows):
            _validate_training(r, f"{path}.rows[{i}]", config, protocol, train_tokens)
            _require(r["seed"] == seed, f"{path}.rows[{i}]: wrong seed")
            if canonical:
                _require(r["val_tokens"] == VAL_TOKENS, f"{path}.rows[{i}]: val.bin is not covered completely")
        _require(len({(r["init_fingerprint"], r["stream_fingerprint"]) for r in rows}) == 1,
                 f"{path}: the arms of a seed must share the initial state and the batch stream")
        muon = rows[0]
        _require(sorted(muon["snapshots"]) == sorted({str(c["muon_steps"]) for c in cool.values()})
                 and all(_is_hex(v, 64) for v in muon["snapshots"].values()), f"{path}: Muon snapshots differ")
        for r in rows[1:3]:
            spec = cool[r["arm"]]
            _require(r["branch"] == {"from_arm": MUON, "from_step": spec["muon_steps"],
                                     "snapshot_sha256": muon["snapshots"][str(spec["muon_steps"])],
                                     "cooldown_steps": spec["cooldown_steps"]}
                     and spec["muon_steps"] + spec["cooldown_steps"] == protocol["muon"]["steps"],
                     f"{path}: {r['arm']} does not branch from the Muon snapshot of the protocol")
        _validate_adamw_run(rows[3], muon, path, protocol["adamw"])
        _assert_close(unit["selection"], select_adamw(rows, tolerance), f"{path}.selection")
        final = {r["arm"]: r for r in rows}
        _assert_close(unit["checkpoints"], {arm: {"file": checkpoint_name(seed, arm),
                                                  "sha256": final[arm]["checkpoint_sha256"],
                                                  "val_nll": final[arm]["val_nll"]} for arm in ARMS},
                      f"{path}.checkpoints")
        by_seed[seed] = final
    return by_seed


GRID_KEYS = {"format", "method", "mode", "rotation", "rounding", "activation_aware_scaling", "seconds",
             *(f"{s}_{k}" for s in SPLITS for k in ("nll", "ppl", "tokens", "delta_nll", "relative_delta_ppl_pct"))}
FAKEQUANT_KEYS = {"quantize_seconds", "gptq_damping_max", "geometry", *(f"{k}_mean" for k in GEOMETRY)}


def _validate_stage_b(payload: dict, final: dict, *, canonical: bool) -> None:
    stage, protocol, config = payload["stage_b"], payload["protocol"], payload["config"]
    seeds, layers, seq_len = protocol["seeds"], config["layers"], config["sequence_length"]
    matrices, names = layers * len(q.LINEAR_KINDS), layer_names(layers)
    train_tokens = payload["data"]["splits"]["train"]["tokens"]
    _require(set(stage) == {"calibration", "checkpoints", "seconds"}, "stage_b keys differ")
    cal, starts = stage["calibration"], stage["calibration"].get("starts")
    _require(cal.get("split") == "train" and cal.get("sequences") == protocol["calibration_sequences"]
             and cal.get("seed") == protocol["calibration_seed"] and cal.get("seq_len") == seq_len
             and cal.get("tokens") == protocol["calibration_sequences"] * seq_len
             and isinstance(starts, list) and len(starts) == protocol["calibration_sequences"]
             and len(set(starts)) == len(starts)
             and all(type(s) is int and 0 <= s <= train_tokens - seq_len and s % seq_len == 0 for s in starts),
             "Calibration windows differ from the protocol (train.bin, aligned, distinct)")
    _require([(c["seed"], c["arm"]) for c in stage["checkpoints"]] == [(s, a) for s in seeds for a in ARMS],
             "Stage B needs one checkpoint per (seed, arm), in order")
    alphas = protocol["awq"]["alphas"]
    for c in stage["checkpoints"]:
        seed, arm = c["seed"], c["arm"]
        path = f"stage_b[{seed}, {arm}]"
        _require(set(c) == {"seed", "arm", "checkpoint_file", "checkpoint_sha256", "checkpoint_sha256_after_ptq",
                            "baseline_val", "baseline_test", "diagnostics", "butterfly_fits", "grid", "awq_selection",
                            "best", "hardware_w4a4", "seconds", "resumed_from_unit_log"}, f"{path}: keys differ")
        trained = final[seed][arm]
        _require(c["checkpoint_sha256"] == c["checkpoint_sha256_after_ptq"] == trained["checkpoint_sha256"]
                 and c["checkpoint_file"] == checkpoint_name(seed, arm),
                 f"{path}: Stage B must evaluate the saved Stage A checkpoint and restore it")
        base = {split: c[f"baseline_{split}"] for split in SPLITS}
        _require(abs(base["val"]["nll"] - trained["val_nll"]) <= 1e-9
                 and base["val"]["tokens"] == trained["val_tokens"],
                 f"{path}: the val NLL of the saved checkpoint differs from its Stage A value")
        if canonical:
            _require((base["val"]["tokens"], base["test"]["tokens"]) == (VAL_TOKENS, TEST_TOKENS),
                     f"{path}: baseline does not cover the full val.bin and test.bin")
        for split in SPLITS:
            _assert_close(base[split]["ppl"], math.exp(base[split]["nll"]), f"{path}.baseline_{split}.ppl")
        hw = c["hardware_w4a4"]
        _require(set(hw) == {"available", "fp4_gemms_per_forward_bf16_baseline", "note"}
                 and isinstance(hw["available"], bool), f"{path}: hardware_w4a4 keys differ")
        hardware = hw["available"]
        _require(hardware or not canonical, f"{path}: a canonical result needs hardware W4A4")
        _require(hw["fp4_gemms_per_forward_bf16_baseline"] == (0 if hardware else None),
                 f"{path}: the BF16 baseline must not run FP4 GEMMs")
        diagnostics = c["diagnostics"]
        per_layer = diagnostics["per_layer"]
        _require(diagnostics["layer_names"] == names and [r["name"] for r in per_layer] == names
                 and all(r["kind"] == r["name"].rsplit(".", 1)[1] and set(r) == {"name", "kind", *LAYER_METRICS}
                         for r in per_layer), f"{path}: diagnostics must cover the {matrices} projections in order")
        _require(all(1.0 - 1e-9 <= r["weight_stable_rank"] and r["weight_fro_norm"] > 0 and r["signal_alignment"] > 0
                     for r in per_layer), f"{path}: layer geometry out of range")
        _assert_close(diagnostics["summary"], summarize_layers(per_layer), f"{path}.diagnostics.summary")
        combos = [(r["format"], r["method"], r["mode"]) for r in c["grid"]]
        _require(combos == expected_combos(arm, protocol, hardware), f"{path}: PTQ grid differs from the protocol")
        selected = {}
        _require([(s["format"], s["method"]) for s in c["awq_selection"]]
                 == [(f, m) for f in protocol["weight_formats"] for m in methods_of(arm, protocol)
                     if m.startswith("awq_")], f"{path}: AWQ selection table differs from the protocol")
        for s in c["awq_selection"]:
            where = f"{path}.awq_selection[{s['format']}/{s['method']}]"
            _require(set(s) == {"format", "method", "rotation", "alphas", "val_nll", "val_tokens",
                                "selected_alpha", "selected_on", "seconds"}, f"{where}: keys differ")
            _require(s["alphas"] == alphas and len(s["val_nll"]) == len(alphas)
                     and all(_is_number(v) for v in s["val_nll"]) and s["selected_on"] == "val"
                     and s["rotation"] == parse_method(s["method"])[0]
                     and s["selected_alpha"] == select_alpha(alphas, s["val_nll"]),
                     f"{where}: alpha is not the val.bin minimum of the protocol grid")
            _require(s["val_tokens"] == base["val"]["tokens"], f"{where}: alpha selection must cover the val windows")
            selected[(s["format"], s["method"])] = s
        plain = {(r["format"], r["method"], r["mode"]): r for r in c["grid"]}
        for (fmt, method), s in selected.items():
            twin = plain[(fmt, method.removeprefix("awq_"), fakequant_mode(fmt))]
            _require(0.0 not in alphas or s["val_nll"][alphas.index(0.0)] == twin["val_nll"],
                     f"{path}.awq_selection[{fmt}/{method}]: alpha = 0 must be the unscaled row")
        for r in c["grid"]:
            where = f"{path}.{r['format']}/{r['method']}/{r['mode']}"
            family, rounding, awq = parse_method(r["method"])
            fake, hw_mode = r["mode"] == fakequant_mode(r["format"]), r["mode"] in HW_MODES
            keys = set(GRID_KEYS)
            if fake:
                keys |= FAKEQUANT_KEYS
                if r["format"] == HW_FORMAT:
                    keys |= {"hw_weights_checked_matrices", "hw_weights_decode_to_fakequant_weights"}
            else:
                keys |= {"fp4_gemms_per_forward"}
            if awq and (fake or hw_mode):
                keys |= {"awq_alpha"}
            if r["mode"] == "w4a4_hw_static":
                keys |= {"static_scale_calibration", "static_global_scale_range"}
            _require(set(r) == keys, f"{where}: keys differ from the schema")
            _require((r["rotation"], r["rounding"], r["activation_aware_scaling"]) == (family, rounding, awq),
                     f"{where}: method fields differ")
            for split in SPLITS:
                _require(r[f"{split}_tokens"] == base[split]["tokens"], f"{where}: {split} token count differs")
                _assert_close(r[f"{split}_ppl"], math.exp(r[f"{split}_nll"]), f"{where}.{split}_ppl")
                _assert_close(r[f"{split}_delta_nll"], r[f"{split}_nll"] - base[split]["nll"],
                              f"{where}.{split}_delta_nll")
                _assert_close(r[f"{split}_relative_delta_ppl_pct"],
                              100.0 * (r[f"{split}_ppl"] / base[split]["ppl"] - 1.0),
                              f"{where}.{split}_relative_delta_ppl_pct")
            if "awq_alpha" in r:
                table = selected[(r["format"], r["method"])]
                _require(r["awq_alpha"] == table["selected_alpha"], f"{where}: alpha differs from the val selection")
                _require(not fake or r["val_nll"] == table["val_nll"][alphas.index(r["awq_alpha"])],
                         f"{where}: the row is not the candidate that the val selection measured")
                if r["awq_alpha"] == 0:  # alpha = 0 is the plain rotation + GPTQ row
                    twin = plain[(r["format"], r["method"].removeprefix("awq_"), r["mode"])]
                    _require(all(r[f"{s}_nll"] == twin[f"{s}_nll"] for s in SPLITS),
                             f"{where}: alpha = 0 must be the unscaled row")
            if fake:
                geometry = r["geometry"]
                _require(set(geometry) == set(GEOMETRY) and all(
                    isinstance(geometry[k], list) and len(geometry[k]) == matrices
                    and all(_is_number(v) and v >= 0 for v in geometry[k]) for k in GEOMETRY),
                    f"{where}: geometry must hold one value per layer")
                for k in GEOMETRY:
                    _assert_close(r[f"{k}_mean"], math.fsum(geometry[k]) / matrices, f"{where}.{k}_mean")
                signal = {row["name"]: row["signal_alignment"] for row in per_layer}
                for i, name in enumerate(names):
                    predicted = geometry["weight_rel_error"][i] * geometry["error_alignment"][i] / signal[name]
                    _require(abs(geometry["output_rel_error"][i] - predicted)
                             <= 1e-6 * max(geometry["output_rel_error"][i], 1e-300),
                             f"{where}: output error of {name} is not weight error x error / signal alignment")
                if r["format"] == HW_FORMAT:
                    _require(r["hw_weights_decode_to_fakequant_weights"] is True
                             and r["hw_weights_checked_matrices"] == matrices,
                             f"{where}: hardware weights must decode to the fake-quant weights in all "
                             f"{matrices} matrices")
            else:
                _require(r["fp4_gemms_per_forward"] == (matrices if hw_mode else 0),
                         f"{where}: expected {matrices if hw_mode else 0} FP4 GEMMs per forward pass")
            if r["mode"] == "w4a4_hw_static":
                _require(r["static_scale_calibration"] == STATIC_CALIBRATION,
                         f"{where}: static activation scales must come from the train.bin calibration windows")
        _assert_close(c["best"], best_methods(c["grid"], protocol["base_methods"]), f"{path}.best")


def validate_results(payload: dict, *, canonical: bool | None = None) -> None:
    """Offline consistency, provenance and promotion-gate validator.

    What it establishes from the JSON and the local git repository alone:

    * schema, strict JSON (no NaN / infinity) and the exact protocol (``canonical``: the locked C5-02
      protocol, pinned shard hashes, complete val/test coverage, sm_120+ device, clean tree);
    * provenance: ``git_sha`` is a commit object (``git cat-file -t``), ``code_sha256`` equals the
      transitive ``research.*`` blobs at that commit, and ``run_config_hash`` / ``unit_config_hash``
      are the recomputed digests;
    * Stage A isolation as recorded: a train + val mapping, test.bin opened only afterwards, no
      test field in any Stage A record, and the AdamW search of every seed replayed with
      ``adamw_eval_steps`` / ``adamw_stops`` / ``select_adamw`` from the stored val NLLs;
    * Stage B identity: every evaluated checkpoint has the SHA-256 and the val NLL of its Stage A
      record; complete grids; per-row arithmetic; alpha = val.bin minimum; 60/60 FP4 GEMMs and
      int32 decode checks as recorded; per-layer error identity;
    * ``analysis``, ``derived`` and ``caveats`` equal their recomputation from the stored rows (1e-12);
    * canonical gate: hardware-versus-FP32-emulation parity within 1e-3 nats.

    Threat model. This is an offline consistency check, not an attestation. It cannot prove that a
    leaf measurement (an NLL, a checkpoint hash, a GEMM count, a timing) was produced by the
    recorded code on the recorded GPU: a party that rewrites leaves and recomputes every dependent
    summary passes. The saved checkpoints are not in git, so their bytes are not checked here.
    Independent evidence needs a live replay on an RTX 5080 from the retained checkpoints or from
    training (training is not bit-reproducible on this host). Scientific outcomes (matched seeds,
    significance, sign of an effect) are reported, never required.
    """
    json.dumps(payload, allow_nan=False)
    top = {"schema_version", "task", "quick", "canonical", "meta", "run_config_hash", "preset", "config",
           "protocol", "data", "stage_a", "stage_b", "analysis", "derived", "caveats"}
    _require(isinstance(payload, dict) and set(payload) == top, "top-level keys differ from the schema")
    _require(payload["schema_version"] == SCHEMA_VERSION and payload["task"] == TASK, "schema or task differs")
    _require(isinstance(payload["quick"], bool) and isinstance(payload["canonical"], bool),
             "quick / canonical flags must be bool")
    canonical = payload["canonical"] if canonical is None else canonical
    protocol, config, data, meta = payload["protocol"], payload["config"], payload["data"], payload["meta"]
    if canonical:
        _require(payload["canonical"] is True and payload["quick"] is False, "a canonical result cannot be --quick")
        _require(protocol == CANONICAL, "protocol differs from the canonical C5-02 protocol")
    _require(payload["preset"] == protocol["preset"] and config == asdict(h.PRESETS[protocol["preset"]]),
             "model config differs from the preset")
    seeds = protocol["seeds"]
    _require(isinstance(seeds, list) and seeds and all(type(s) is int for s in seeds) and len(set(seeds)) == len(seeds),
             "protocol.seeds must be distinct integers")

    # Data hygiene.
    _require(data.get("roles") == {"training": "train", "calibration": "train", "static_scales": "train",
                                   "matching": "val", "method_and_alpha_selection": "val", "evaluation": "test"},
             "split roles differ: train for training and calibration, val for selection, test for evaluation")
    _require(set(data["splits"]) == {"train", "val", "test"}, "data.splits must hold train, val and test")
    if canonical:
        found = {k: v["sha256"] for k, v in data["splits"].items()}
        _require(data["sha256_verified"] is True and data["expected_sha256"] == q.EXPECTED_SHA256
                 and found == q.EXPECTED_SHA256, "shard SHA-256 values differ from the pinned train/val/test hashes")
    _validate_meta(payload, canonical=bool(canonical))
    _require(payload["run_config_hash"] == run_config_hash(
        payload["quick"], protocol, config, meta["code_sha256"], meta["git_sha"], meta["environment"],
        {k: v["sha256"] for k, v in data["splits"].items()}), "run_config_hash differs from the recomputed digest")

    final = _validate_stage_a(payload, canonical=bool(canonical))
    _validate_stage_b(payload, final, canonical=bool(canonical))

    # Analysis, derived numbers and caveats are pure functions of the rows above.
    if len(seeds) == N_SEEDS:
        _assert_close(payload["analysis"], json.loads(json.dumps(analyze(payload["stage_a"], payload["stage_b"],
                                                                         protocol))), "analysis")
        iso = payload["analysis"]["iso_loss"]
        _require(all(p["same_init_and_stream"] and p["stage_b_val_equals_stage_a"] for p in iso["pairs"]),
                 "the arms of a seed must share init and stream, and Stage B must evaluate the Stage A weights")
    else:
        _require(payload["analysis"] is None, f"the paired analysis needs exactly {N_SEEDS} seeds")
    _assert_close(payload["derived"], json.loads(json.dumps(derive(payload))), "derived")
    _assert_close(payload["caveats"], caveats(payload), "caveats")
    if canonical:
        _require(payload["analysis"] is not None, "a canonical result needs the paired analysis")
        _require(payload["analysis"]["hardware_vs_emulation"]["passed"] is True
                 and len(payload["analysis"]["hardware_vs_emulation"]["rows"]) == len(seeds) * len(ARMS),
                 "hardware W4A4 differs from its FP32 emulation by more than the parity threshold")


# ======================================================================================
# Driver
# ======================================================================================
def set_numerics(device: torch.device) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return dict(NUMERICS)


def run(args) -> dict:
    device = torch.device(args.protocol["device"])
    git = h.git_info()
    require_canonical_host(device, git, args.output, args.canonical)  # before any GPU work
    numerics = set_numerics(device)
    log = (lambda *a, **k: None) if args.silent else (lambda *a, **k: print(*a, **k, flush=True))
    started, timestamp_start = time.perf_counter(), time.strftime("%Y-%m-%dT%H:%M:%S%z")
    protocol = copy.deepcopy(args.protocol)
    config = h.PRESETS[protocol["preset"]]
    shards = Shards(args.data_dir, verify=not args.no_verify, expected=None if args.no_verify else q.EXPECTED_SHA256)
    data_sha = {k: v["sha256"] for k, v in shards.summary()["splits"].items()}
    env = q.environment(device)
    code = code_sha256()
    unit_hash = unit_config_hash(protocol, asdict(config), code, env, data_sha)
    cfg_hash = run_config_hash(args.quick, protocol, asdict(config), code, git["git_sha"], env, data_sha)
    units = UnitLog(unit_log_path(args.output), unit_hash, args.resume)
    checkpoint_dir = Path(args.checkpoint_dir)
    meta = {**git, "command": [sys.executable, *sys.argv], "environment": env, "numerics": numerics,
            "code_sha256": code, "code_modules": list(code_modules()), "unit_config_hash": unit_hash,
            "checkpoint_dir": str(checkpoint_dir), "timestamp_start": timestamp_start,
            "loadavg_start": list(os.getloadavg()), "gpu_telemetry": {"start": h.gpu_telemetry(device)},
            "stage_seconds": {}}
    log(f"[c5-02] git {git['git_sha']} dirty={git['git_dirty']} device={env.get('device', 'cpu')} "
        f"run {cfg_hash} units {unit_hash} resume units {len(units.units)} canonical={args.canonical}")
    payload = {"schema_version": SCHEMA_VERSION, "task": TASK, "quick": args.quick, "canonical": args.canonical,
               "meta": meta, "run_config_hash": cfg_hash, "preset": protocol["preset"], "config": asdict(config),
               "protocol": protocol,
               "data": {**shards.summary(), "directory": str(args.data_dir), "sha256_verified": not args.no_verify,
                        "expected_sha256": None if args.no_verify else q.EXPECTED_SHA256,
                        "roles": {"training": "train", "calibration": "train", "static_scales": "train",
                                  "matching": "val", "method_and_alpha_selection": "val", "evaluation": "test"}}}

    # Stage A: every seed, on a mapping without test.bin. test.bin is not opened before this loop ends.
    began = time.perf_counter()
    view_a = shards.open("train", "val")
    stage_a_units = []
    for seed in protocol["seeds"]:
        unit = units.get(("stage_a", seed), lambda value: checkpoints_present(value, checkpoint_dir))
        if unit is None:
            unit = units.put(("stage_a", seed),
                             stage_a_seed(config, view_a, device, protocol, seed, checkpoint_dir, log))
        sel = unit["selection"]
        log(f"Stage A seed {seed}: matched={sel['matched']} T_adamw {sel['t_adamw']} "
            f"({sel['adamw_evaluations']} measured steps), val gap (muon - adamw) "
            f"{sel['val_gap_muon_minus_adamw']:+.4f} ({unit['seconds']:.0f}s)")
        stage_a_units.append(unit)
    payload["stage_a"] = {"split_view": sorted(view_a), "splits_opened_at_end": list(shards.opened),
                          "seeds": stage_a_units, "seconds": time.perf_counter() - began}
    meta["stage_seconds"]["stage_a"] = payload["stage_a"]["seconds"]
    meta["gpu_telemetry"]["after_stage_a"] = h.gpu_telemetry(device)

    # Stage B: the saved checkpoints on val.bin and test.bin.
    began = time.perf_counter()
    view_b = {**view_a, **shards.open("test")}
    calibration = q.calibration_windows(SimpleNamespace(splits=view_a), protocol["calibration_sequences"],
                                        config.sequence_length, protocol["calibration_seed"])
    checkpoints = []
    for unit in stage_a_units:
        for arm in ARMS:
            seed, entry = unit["seed"], unit["checkpoints"][arm]
            record = units.get(("stage_b", seed, arm), lambda value: value["checkpoint_sha256"] == entry["sha256"])
            if record is None:
                model = load_checkpoint(config, device, checkpoint_dir / entry["file"], entry["sha256"])
                record = units.put(("stage_b", seed, arm), {
                    "seed": seed, "arm": arm, "checkpoint_file": entry["file"],
                    **checkpoint_study(model, arm, config, view_b, device, calibration, protocol, log,
                                       f"s{seed} {arm}")})
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            log(f"Stage B seed {seed} {arm}: {len(record['grid'])} rows ({record['seconds']:.0f}s)")
            checkpoints.append(record)
    payload["stage_b"] = {"calibration": {k: v for k, v in calibration.items() if k != "array"},
                          "checkpoints": checkpoints, "seconds": time.perf_counter() - began}
    meta["stage_seconds"]["stage_b"] = payload["stage_b"]["seconds"]
    meta["unit_log"] = {"resume": args.resume, "units_loaded": units.loaded, "units_run": units.run}
    meta["splits_opened"] = list(shards.opened)
    meta["gpu_telemetry"]["end"] = h.gpu_telemetry(device)
    meta["loadavg_end"] = list(os.getloadavg())
    meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["total_seconds"] = time.perf_counter() - started
    payload = json.loads(json.dumps(payload, allow_nan=False))
    payload["analysis"] = (analyze(payload["stage_a"], payload["stage_b"], protocol)
                           if len(protocol["seeds"]) == N_SEEDS else None)
    payload["derived"] = derive(payload)
    payload["caveats"] = caveats(payload)
    payload = json.loads(json.dumps(payload, allow_nan=False))
    try:
        validate_results(payload, canonical=args.canonical)
    except Exception:
        # Keep the measurements of a long run: the rejected payload goes next to the output, never to it.
        c4.write_json_atomic(rejected_path(args.output), payload)
        raise
    return payload


def rejected_path(output: Path) -> Path:
    return output.with_name(output.name + ".rejected")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="tiny model, few steps, short evaluation (never canonical)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="seeds (default 1 2 3 4 5; any other list is not canonical and has no paired analysis "
                             "unless it has five seeds)")
    parser.add_argument("--device", default=None, help="default: cuda")
    parser.add_argument("--max-eval-windows", type=int, default=None, help="limit val/test windows (not canonical)")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--no-verify", action="store_true", help="skip SHA-256 checks (synthetic shards only)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="where Stage A saves the final state dicts (default: artifacts/c5_02_checkpoints for "
                             "the canonical run, <output>.ckpt otherwise)")
    parser.add_argument("--resume", action="store_true", help="reuse finished units of an interrupted run")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    protocol = copy.deepcopy(QUICK if args.quick else CANONICAL)
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds must be distinct")
        protocol["seeds"] = list(args.seeds)
    if args.device is not None:
        protocol["device"] = args.device
    if args.max_eval_windows is not None:
        protocol["max_eval_windows"] = args.max_eval_windows
    args.protocol = protocol
    args.canonical = not args.quick and not args.no_verify and protocol == CANONICAL
    if args.output is None:
        if args.canonical:
            args.output = CANONICAL_OUTPUT
        elif args.quick:
            args.output = QUICK_OUTPUT
        else:
            parser.error(f"A non-canonical run needs an explicit --output (not {RESERVED_PATTERN})")
    if reserved(args.output) and not args.canonical:
        parser.error(f"Refusing to write the reserved {args.output.name} from a --quick or non-canonical run; "
                     f"choose a path outside {RESERVED_PATTERN}")
    if args.canonical and (args.output.is_symlink() or args.output.resolve() != CANONICAL_OUTPUT.resolve()):
        parser.error(f"The canonical protocol writes {CANONICAL_OUTPUT} of this checkout (no symlink, no other "
                     "checkout); change the protocol or use --quick for other paths")
    if args.data_dir is None:
        args.data_dir = q.default_data_dir()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = (CANONICAL_CHECKPOINT_DIR if args.canonical
                               else args.output.with_name(args.output.name + ".ckpt"))
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    payload = c4.write_json_atomic(args.output, run(args))
    unit_log_path(args.output).unlink(missing_ok=True)
    if not args.silent:
        print(json.dumps(payload["derived"].get("iso_loss", payload["derived"]["stage_a"]), indent=1))
        print(f"Wrote {args.output}")
    return payload


if __name__ == "__main__":
    main()
