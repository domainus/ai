"""Cycle 5 C5-02: per-seed iso-loss Muon-versus-AdamW PTQ and three Muon PTQ-sensitivity mitigations.

C4-01b matched AdamW to Muon on one pilot seed and then locked the recipe. Only one of its three
held-out seeds met the 0.02-nat target, and its checkpoints were not kept. This benchmark matches
every seed on ``val.bin`` alone, keeps the matched weights, and tests three mitigations on the
same five paired seeds.

Stage A  Training and matching. The stage receives a mapping with ``train`` and ``val`` only:
         ``Shards.open`` maps ``train.bin`` and ``val.bin`` for Stage A and maps ``test.bin`` for the
         first time for the Stage B scoring. This is not the first read of ``test.bin`` by the
         process: ``Shards(verify=True)`` reads all three shard files twice at startup (SHA-256, then
         token range and document boundaries for the manifest and disjointness checks). That
         verification scores no model.
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

``test.bin`` is held out from the canonical selection rules: training, stopping, checkpoint
selection, calibration, alpha selection and method selection read ``train.bin`` and ``val.bin``
only. It is not a previously unseen study-level holdout. Seed-0 shakedown runs (the logs
``timing_seed0.log`` and ``timing_seed0b.log``, and the shakedown JSON with SHA-256 ``39f2d27e...``)
scored the same ``test.bin`` before the initial commit ``05ea4ae`` and the amendment ``0355d16``.

``cycle5_c5_02*.json`` is reserved in every checkout and through symlinks: only the canonical
protocol, on an sm_120+ device and a clean git tree, may write it. Both guards run before any
GPU work. ``validate_results`` documents what an offline check can and cannot establish.

Schema ``cycle5_c5_02.v2`` adds the top-level ``postprocess`` field. It is ``None`` for a run that
measures. ``--postprocess-from-git <commit>`` reads the committed ``cycle5_c5_02.v1`` result of that
commit with ``git show``, keeps every raw measurement unchanged (``RAW_MEASUREMENT_KEYS``),
recomputes only ``analysis``, ``derived`` and ``caveats``, and records the provenance of the source
and of the postprocessing code in ``postprocess``. It needs no GPU and does not retrain.

Run (alone on the GPU, from a clean commit, about 3.3 hours on an RTX 5080: each measured grid row
reads the complete val.bin and test.bin; 562 of the 620 canonical rows are measured, 58 are copies of
alpha = 0 twins. ``--resume`` continues an interrupted run; it reuses a finished unit only after the
tensor digest of every retained checkpoint of that unit is verified):
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_c5_02_isoloss_ptq_mitigation.py
Smoke run:
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_c5_02_isoloss_ptq_mitigation.py --quick \\
        --output /tmp/c5_02.json
Postprocess the committed v1 measurements (clean tree, no GPU):
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_c5_02_isoloss_ptq_mitigation.py \\
        --postprocess-from-git da148c296a4b8750fa9e67c0145e4f0789ecba54
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
import re
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
SCHEMA_VERSION = "cycle5_c5_02.v2"
SOURCE_SCHEMA_VERSION = "cycle5_c5_02.v1"  # the schema that ``--postprocess-from-git`` reads
TASK = ("Cycle 5 C5-02: per-seed validation-only iso-loss Muon-vs-AdamW PTQ and Muon PTQ-sensitivity "
        "mitigations (AdamW cooldown, NVFP4 QAT cooldown, activation-aware scaling) on Blackwell")
SCRIPT_NAME = "bench_c5_02_isoloss_ptq_mitigation.py"
RESULT_REPO_PATH = "results/cycle5_c5_02_isoloss_ptq_mitigation.json"
CANONICAL_OUTPUT = ROOT / RESULT_REPO_PATH
DRIVER_MODULE = "research.quant_ptq.bench_c5_02_isoloss_ptq_mitigation"
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
# Scored targets of the complete 512-token windows of the canonical val.bin and test.bin (3,910 and 3,906
# windows). For reference: the validator recomputes the coverage from the split lengths (``expected_coverage``).
VAL_TOKENS, TEST_TOKENS = 2_001_920, 1_999_872
# Identity of the canonical shards (FineWeb-Edu, GPT-2 BPE). The SHA-256 values are ``q.EXPECTED_SHA256``.
CANONICAL_DATA = {
    "source": {"dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "split": "train",
               "streaming": True, "name": "fineweb-edu", "fallback_from": None},
    "encoding": "gpt2",
    "tokens": {"train": 26_004_518, "val": 2_002_330, "test": 2_000_081},
}
ROLES = {"training": "train", "calibration": "train", "static_scales": "train", "matching": "val",
         "method_and_alpha_selection": "val", "evaluation": "test"}
CANONICAL_ORTHOGONALIZER = "schatten_switch_blackwell SS3_54 FP16 (Triton)"
ADAMW_ORTHOGONALIZER = "fused AdamW"
TRAIN_LOSS_NOTE = "train_loss is the mean of the last 10 optimizer steps (of this arm's own phase)"
BUTTERFLY_FAMILY, BUTTERFLY_LR = "adaptive_butterfly_b16", 0.01  # ``q.fit_butterfly`` default LR
GPTQ_DAMP_TRIES = 8  # ``q.gptq_quantize`` multiplies the damping by 10 at most 7 times
HARDWARE_SKIP_NOTE = "skipped: needs CUDA sm_100+ with F.scaled_mm"

# Raw measurements of a result. ``--postprocess-from-git`` copies them unchanged from the source.
RAW_MEASUREMENT_KEYS = ("quick", "canonical", "meta", "run_config_hash", "preset", "config", "protocol", "data",
                        "stage_a", "stage_b")
POSTPROCESS_KEYS = ("source_schema_version", "source_result_git_sha", "source_artifact_sha256",
                    "source_raw_measurements_sha256", "postprocess_git_sha", "postprocess_git_dirty",
                    "postprocess_code_sha256", "postprocess_code_modules", "postprocess_command",
                    "postprocess_timestamp", "retained_evidence_sha256", "reason")
# The committed v1 result that the canonical postprocess reads: result commit -> SHA-256 of the file.
CANONICAL_SOURCE_ARTIFACTS = {
    "da148c296a4b8750fa9e67c0145e4f0789ecba54": "4944c2832056524f994a2d4931950983244ac20e547cc9e281d5e7e0defeddb8"}
# Run evidence that is retained outside git (artifacts/c5_02_design_pilot/), as hashed by the C5-02 audit:
# aborted_canonical_05ea4ae.log, canonical_0355d16.log, timing_seed0b.json and probe_hold.out.
RETAINED_EVIDENCE_SHA256 = {
    "stopped_canonical_log": "d877d6862b98b8d140aeb2436f0085ce18173ccada7252e0d140910e64c0d046",
    "final_canonical_log": "cfde4b04bc416c04cb5d98c39f94ee59312b62c2f24bfe840acd35c065b843fe",
    "seed0_shakedown_json": "39f2d27e6661e89f7f3e7f4cef7bab7ed78441b73236e81bc92fb8c303421803",
    "floor_lr_probe_output": "adf722e3ab295959d11bb6a0889d8c8e376d2c65f5a540656caecfb5ba44a2a5",
}
POSTPROCESS_REASON = (
    "Remediation of the C5-02 audit (audits/AUDIT_cycle5_c5_02_gpt62_astra.md): the raw measurements of the "
    "source result are unchanged; analysis, derived and caveats are recomputed with the corrected token-exposure "
    "statement, the corrected split-access and pilot chronology, the unique and inherited hardware counts and "
    "the tightened caveats, and the result is checked by the hardened validator. Nothing was retrained.")
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

# Design-time pilot (seed 0, never part of a result). Muon: val NLL 5.343 to 5.346.
# * Fresh warmup + cosine AdamW runs at LR 0.001 with 456 .. 704 steps: two runs of one configuration differed
#   by 0.06 nats (640 steps) and 0.07 nats (472 steps), and 496 / 512 / 528 / 536 steps gave 5.393 / 5.500 /
#   5.338 / 5.262. A match by fresh runs is therefore a lottery.
# * Inside one run the val NLL falls smoothly at the floor LR (about 0.0044 nats per 8 steps), so one run per
#   seed is stopped on a 16-step grid instead. Three runs with ``cosine_steps`` = 464 were 0.05 to 0.28 nats
#   above Muon when the floor LR was reached; the worst needs about 500 floor-LR steps.
# * Floor-LR probe (val.bin and train calibration only). It is a sensitivity check at different losses, not
#   an equal-loss proof: identity-GPTQ val NLL increase 0.0010 (NVFP4) / 0.0096 (INT2 g16) at val NLL 5.414 of
#   a run stopped at 848 steps, against 0.0010 / 0.0106 at 5.298 and 0.0008 / 0.0106 at 5.235 for fresh cosine
#   runs of 544 and 608 steps. The magnitudes are similar across 5.235 to 5.414; a floor-phase effect at equal
#   loss is not excluded.
# * Seed-0 shakedowns of the complete pipeline (timing_seed0.log, timing_seed0b.log) scored test.bin before
#   the initial commit 05ea4ae and before the amendment 0355d16. test.bin is therefore held out from the
#   canonical selection rules, but it is not a previously unseen study-level holdout.
# Protocol amendment. The first canonical attempt (commit 05ea4ae, ``max_steps`` = 1280) was stopped during
# Stage A and has no Stage B scores: seed 1 matched at 640 steps (gap 0.003 nats), but the AdamW run of seed 2
# was 0.58 nats above Muon at step 400 and still 0.0925 nats above it at step 1280, where the log ends. Its
# startup verification had read test.bin (hashes and document boundaries); no model was scored on it. AdamW at
# LR 0.001 has such slow runs, so the cap is 2560 steps (about 0.9 nats of floor-LR progress). Nothing else
# changed. The fresh runs of 0355d16 follow other trajectories; every one of them stopped below step 1280. A
# run that needs more than the cap is reported as unmatched.
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


def require_clean_tree(git: dict, output: Path) -> None:
    """A postprocess that writes a reserved filename needs a clean, committed tree (no GPU)."""
    if reserved(output) and (git.get("git_dirty") is not False or not _is_hex(git.get("git_sha"), 40)):
        raise RuntimeError(f"Postprocessing into {Path(output).name} needs a clean, committed git tree "
                           "(git_dirty must be False)")


def _module_path(name: str) -> Path:
    return ROOT.joinpath(*name.split(".")).with_suffix(".py")


def _import_closure(source_of) -> tuple[str, ...]:
    """Static ``ast`` import closure of the driver. ``source_of(module)`` gives the source text or None."""
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


@functools.cache
def code_modules() -> tuple[str, ...]:
    """Every ``research.*`` module this driver imports, directly or transitively (static ``ast`` closure)."""
    return _import_closure(lambda name: _module_path(name).read_text() if _module_path(name).is_file() else None)


def code_sha256() -> str:
    digest = hashlib.sha256()
    for name in code_modules():
        digest.update(name.encode() + b"\0" + _module_path(name).read_bytes())
    return digest.hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)


def _require_git_commit(sha: str, label: str) -> None:
    kind = _git("cat-file", "-t", sha) if _is_hex(sha, 40) else None
    if kind is None or kind.returncode != 0 or kind.stdout.strip() != b"commit":
        raise ValueError(f"{label} {sha} is not a commit object")


def _git_blob(git_sha: str, repo_path: str) -> bytes:
    """The committed bytes of ``repo_path`` at ``git_sha`` (``git show <sha>:<path>``)."""
    blob = _git("show", f"{git_sha}:{repo_path}")
    if blob.returncode != 0:
        raise ValueError(f"cannot read {repo_path} at commit {git_sha}")
    return blob.stdout


def _module_repo_path(name: str) -> str:
    return "/".join(name.split(".")) + ".py"


@functools.cache
def _code_modules_at_commit(git_sha: str) -> tuple[str, ...]:
    """The static import closure of the driver, computed from the git blobs of ``git_sha``."""
    _require_git_commit(git_sha, "git_sha")
    listing = _git("ls-tree", "-r", "--name-only", git_sha, "--", "research")
    if listing.returncode != 0:
        raise ValueError(f"cannot list the tree of commit {git_sha}")
    files = set(listing.stdout.decode().splitlines())

    def source_of(name: str) -> str | None:
        repo_path = _module_repo_path(name)
        return _git_blob(git_sha, repo_path).decode() if repo_path in files else None

    return _import_closure(source_of)


@functools.cache
def _code_sha256_at_commit(git_sha: str, modules: tuple[str, ...]) -> str:
    _require_git_commit(git_sha, "git_sha")
    digest = hashlib.sha256()
    for module in modules:
        blob = _git("show", f"{git_sha}:{_module_repo_path(module)}")
        if blob.returncode != 0:
            raise ValueError(f"cannot read {module} at git_sha {git_sha}")
        digest.update(module.encode() + b"\0" + blob.stdout)
    return digest.hexdigest()


@functools.cache
def _driver_capabilities_at_commit(git_sha: str) -> tuple[str | None, bool]:
    """``(declared SCHEMA_VERSION, supports --postprocess-from-git)`` of the driver committed at ``git_sha``."""
    _require_git_commit(git_sha, "git_sha")
    try:
        tree = ast.parse(_git_blob(git_sha, _module_repo_path(DRIVER_MODULE)).decode())
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot parse {DRIVER_MODULE} at git_sha {git_sha}") from exc
    schema_version = None
    has_postprocess_fn = False
    has_postprocess_cli = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == "SCHEMA_VERSION" and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            schema_version = node.value.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "SCHEMA_VERSION" and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            schema_version = node.value.value
        elif isinstance(node, ast.FunctionDef) and node.name == "postprocess_from_git":
            has_postprocess_fn = True
        elif isinstance(node, ast.FunctionDef) and node.name == "build_parser":
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) \
                        and child.func.attr == "add_argument" and child.args \
                        and isinstance(child.args[0], ast.Constant) \
                        and child.args[0].value == "--postprocess-from-git":
                    has_postprocess_cli = True
    return schema_version, has_postprocess_fn and has_postprocess_cli


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def run_config_hash(quick: bool, protocol: dict, config: dict, code: str, git_sha: str | None, env: dict,
                    data_sha256: dict, schema: str = SCHEMA_VERSION) -> str:
    """``schema`` is the schema version of the run that measured (the source schema after a postprocess)."""
    return _digest({"schema": schema, "quick": quick, "protocol": protocol, "config": config,
                    "code_sha256": code, "git_sha": git_sha, "env": {k: env.get(k) for k in ENV_KEYS},
                    "data_sha256": data_sha256})


def unit_config_hash(protocol: dict, config: dict, code: str, env: dict, data_sha256: dict,
                     schema: str = SCHEMA_VERSION) -> str:
    """Resume key of a per-seed unit: everything that determines it, without the seed list.

    Transitive code, torch / triton / cuda / driver / device, shard hashes, the model config and the
    protocol (calibration seed included). A changed dependency or runtime therefore reuses nothing.
    """
    return _digest({"schema": schema, "protocol": {k: v for k, v in protocol.items() if k != "seeds"},
                    "config": config, "code_sha256": code, "env": {k: env.get(k) for k in ENV_KEYS},
                    "data_sha256": data_sha256})


def raw_measurements_sha256(payload: dict) -> str:
    """SHA-256 of the raw measurements in one canonical JSON form (sorted keys, no whitespace)."""
    raw = {key: payload[key] for key in RAW_MEASUREMENT_KEYS}
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


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
# Data: ``Shards.open`` maps splits one by one, so a stage can be given a mapping without test.bin
# ======================================================================================
class Shards:
    """Verified shard directory. ``open`` memory-maps single splits and records the order.

    Split access. With ``verify=True`` the constructor reads ``train.bin``, ``val.bin`` and ``test.bin``
    twice each: once for the SHA-256 and once for the token range and the document boundaries
    (manifest and disjointness checks of ``real_text_data.verify_manifest``). No model is scored there.
    ``opened`` records the ``open`` calls only: train and val for Stage A, then test for the Stage B
    scoring. It is therefore the order in which the splits are mapped for training and scoring, not
    the order of the first file read.
    """

    def __init__(self, directory: Path, *, verify: bool, expected: dict | None):
        self.directory = Path(directory)
        self.manifest = (real_text_data.verify_manifest(self.directory) if verify
                         else json.loads((self.directory / "manifest.json").read_text()))
        found = {k: v["sha256"] for k, v in self.manifest["splits"].items()}
        if expected and any(found.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Shard SHA-256 values differ from the pinned hashes: {found}")
        if not verify:  # ``verify_manifest`` makes the same size check; here it is a stat, not a read
            for name, entry in self.manifest["splits"].items():
                size = (self.directory / entry["file"]).stat().st_size
                if size != 2 * entry["tokens"]:
                    raise ValueError(f"{name}: size {size} != 2 * {entry['tokens']} tokens")
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
            "train_loss_note": TRAIN_LOSS_NOTE,
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
                        seconds=time.perf_counter() - began - eval_seconds, orthogonalizer=ADAMW_ORTHOGONALIZER,
                        schedule_steps=spec["cosine_steps"], cosine_steps=spec["cosine_steps"], trained_steps=done,
                        eval_seconds=eval_seconds, evaluations=evaluations)
    row.update(val_nll=chosen["val_nll"], val_ppl=math.exp(chosen["val_nll"]), val_tokens=chosen["val_tokens"],
               evaluated_splits=["val"])
    return state, row


def cooldown_recipe(arm: str, protocol: dict) -> dict:
    """The ``cooldown`` record of a cooldown arm. It follows from the locked protocol alone."""
    if arm == COOL_ADAMW:
        return {"optimizer": "fused AdamW on the hidden matrices (fresh moments); auxiliary AdamW continues",
                "lr": protocol["adamw_cooldown"]["lr"]}
    if arm == COOL_QAT:  # identity coordinates and round to nearest only: no GPTQ, no rotation
        return {"optimizer": "Muon continues (same momentum and auxiliary AdamW)",
                "fake_quant": f"{protocol['qat_cooldown']['format']} round-to-nearest, straight-through, weight and "
                              "input of every quantized linear, identity coordinates"}
    raise ValueError(f"Unknown cooldown arm {arm!r}")


def train_cooldown(arm: str, config, train, device, protocol: dict, *, seed: int, init: dict, snapshot: dict):
    """A cooldown arm: the Muon run up to ``snapshot``, then its own last steps on the same batches."""
    total = protocol["muon"]["steps"]
    extra = {"cooldown": cooldown_recipe(arm, protocol)}
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
    else:
        fmt = q.FORMATS[spec["format"]]
        wrapped = {name: FakeQuantLinear(linear, fmt) for name, linear in q.quantized_linears(model).items()}
        previous = e2e._swap(model, wrapped)
        try:
            losses = _run_steps(model, muon, stream, first, total, multiplier, f"{arm} seed {seed}")
        finally:
            e2e._swap(model, previous)
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


def state_sha256(state: dict[str, Tensor]) -> str:
    """Tensor-content SHA-256 of a state dict: the digest of ``e2e._checkpoint_sha256`` without a model.

    Sorted names, a NUL separator, the shape and the contiguous FP32 bytes of every tensor, on the CPU.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().float().contiguous().cpu().numpy()
        digest.update(name.encode("utf-8") + b"\0" + str(array.shape).encode("utf-8") + array.tobytes())
    return digest.hexdigest()


def checkpoint_file_matches_sha256(path: Path, sha256: str) -> bool:
    """True when ``path`` loads (CPU, ``weights_only=True``) and its tensor-content SHA-256 is ``sha256``.

    False for a missing, unreadable, corrupt or different file. No GPU and no model are needed.
    """
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        return isinstance(state, dict) and bool(state) and state_sha256(state) == sha256
    except Exception:  # any load failure means the retained weights cannot be trusted
        return False


def load_checkpoint(config: h.ModelConfig, device: torch.device, path: Path, sha256: str) -> Transformer:
    """The saved state dict of a Stage A arm; raises unless its SHA-256 is the recorded one."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"{path.name}: the retained checkpoint cannot be read") from exc
    model = _new_model(config, device, state).eval()
    if e2e._checkpoint_sha256(model) != sha256:
        raise RuntimeError(f"{path.name}: checkpoint SHA-256 differs from the Stage A record")
    return model


def checkpoints_verified(unit: dict, checkpoint_dir: Path) -> bool:
    """A cached Stage A unit is usable only when every retained checkpoint has its recorded digest."""
    return all(checkpoint_file_matches_sha256(checkpoint_dir / entry["file"], entry["sha256"])
               for entry in unit["checkpoints"].values())


def stage_b_unit_verified(record: dict, entry: dict, checkpoint_dir: Path) -> bool:
    """A cached Stage B unit is usable only for the Stage A digest, and only while the retained file has it."""
    return (record["checkpoint_sha256"] == entry["sha256"]
            and checkpoint_file_matches_sha256(checkpoint_dir / entry["file"], entry["sha256"]))


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


def inherited(row: dict) -> bool:
    """True for an activation-aware row with alpha = 0: its values are copied from the unscaled twin row."""
    return row.get("awq_alpha") == 0


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
    # An alpha = 0 row is a copy of its unscaled twin: it was not profiled or decoded again.
    hw_copies, decode_copies = [r for r in hw_rows if inherited(r)], [r for r in decode_rows if inherited(r)]
    hardware = {"w4a4_rows": len(hw_rows),
                "w4a4_unique_profiled_rows": len(hw_rows) - len(hw_copies),
                "w4a4_inherited_zero_alpha_rows": len(hw_copies),
                "fp4_gemms_per_forward": sorted({r["fp4_gemms_per_forward"] for r in hw_rows}),
                "fp4_gemms_per_forward_emulated": sorted({r["fp4_gemms_per_forward"] for rows in tables.values()
                                                          for r in rows.values() if r["mode"] == EMULATED_MODE}),
                "fp4_gemms_per_forward_bf16_baseline": sorted(
                    {c["hardware_w4a4"]["fp4_gemms_per_forward_bf16_baseline"] for c in stage_b["checkpoints"]
                     if c["hardware_w4a4"]["available"]}),
                "decode_checked_rows": len(decode_rows),
                "decode_unique_checked_rows": len(decode_rows) - len(decode_copies),
                "decode_inherited_zero_alpha_rows": len(decode_copies),
                "decode_checked_matrices": sum(r["hw_weights_checked_matrices"] for r in decode_rows),
                "decode_unique_checked_matrices": sum(r["hw_weights_checked_matrices"] for r in decode_rows
                                                      if not inherited(r)),
                "decode_inherited_zero_alpha_matrices": sum(r["hw_weights_checked_matrices"] for r in decode_copies),
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


def token_exposure(payload: dict) -> list[dict]:
    """Per seed: optimizer steps and training-set token equivalents (tokens drawn / tokens of train.bin).

    ``BatchStream`` draws every window with replacement, so a token equivalent is not an epoch of distinct
    tokens. ``adamw_trained_*`` counts every step that the AdamW run made, also the steps after the selected one.
    """
    config, train_tokens = payload["config"], payload["data"]["splits"]["train"]["tokens"]
    step_tokens = config["micro_batch"] * config["grad_accum"] * config["sequence_length"]
    out = []
    for unit in payload["stage_a"]["seeds"]:
        rows = {r["arm"]: r for r in unit["rows"]}
        muon, adamw = rows[MUON], rows[ADAMW]
        out.append({"seed": unit["seed"], "muon_steps": muon["steps"],
                    "muon_token_equivalents": muon["steps"] * step_tokens / train_tokens,
                    "adamw_selected_steps": adamw["steps"], "adamw_trained_steps": adamw["trained_steps"],
                    "adamw_selected_token_equivalents": adamw["steps"] * step_tokens / train_tokens,
                    "adamw_trained_token_equivalents": adamw["trained_steps"] * step_tokens / train_tokens})
    return out


def derive(payload: dict) -> dict:
    """Headline numbers, copied from the stored stages and the analysis."""
    stage_a, stage_b, analysis = payload["stage_a"], payload["stage_b"], payload["analysis"]
    keys = ("matched", "t_adamw", "adamw_floor_lr_steps", "adamw_evaluations", "abs_val_gap",
            "val_gap_muon_minus_adamw", "adamw_over_muon_tokens")
    exposure_keys = ("adamw_trained_steps", "adamw_selected_token_equivalents", "adamw_trained_token_equivalents")
    out = {"stage_a": [{"seed": s["seed"], **{k: s["selection"][k] for k in keys},
                        **{k: exposure[k] for k in exposure_keys}}
                       for s, exposure in zip(stage_a["seeds"], token_exposure(payload))],
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
    # Totals and, next to them, the unique executions (alpha = 0 rows are copies of their unscaled twins).
    out["hardware"] = {**analysis["hardware"],
                       "emulation_max_abs_nll_difference": analysis["hardware_vs_emulation"]["max_abs_nll_difference"],
                       "emulation_parity_passed": analysis["hardware_vs_emulation"]["passed"]}
    out["layer_geometry_arm_means"] = {metric: {arm: analysis["layer_geometry"][metric]["per_arm"][arm]["mean"]
                                                for arm in ARMS}
                                       for metric in ("weight_stable_rank", "signal_alignment", "weight_fro_norm")}
    return out


def _span(values: list, fmt: str) -> str:
    low, high = format(min(values), fmt), format(max(values), fmt)
    return low if low == high else f"{low}-{high}"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def exposure_caveat(payload: dict) -> str:
    """Token exposure and compute, generated from the selected and the terminal AdamW steps of every seed."""
    config = payload["config"]
    exposure = token_exposure(payload)
    slices, length = config["micro_batch"] * config["grad_accum"], config["sequence_length"] + 1
    train_tokens = payload["data"]["splits"]["train"]["tokens"]
    selected = [e["adamw_selected_token_equivalents"] for e in exposure]
    trained = [e["adamw_trained_token_equivalents"] for e in exposure]
    text = (f"One {payload['preset']} model family, one calibration draw and short training. Training batches are "
            f"BatchStream draws with replacement: every optimizer step takes {slices} independent {length}-token "
            f"slices at uniform random starts (no sequential cursor), so exposure is given in training-set token "
            f"equivalents (tokens drawn / {train_tokens:,} tokens of train.bin), not in epochs of distinct tokens. "
            f"Muon and both cooldown arms draw {_span([e['muon_token_equivalents'] for e in exposure], '.3f')} "
            f"token equivalents ({_span([e['muon_steps'] for e in exposure], 'd')} steps). The selected AdamW "
            f"checkpoints are at steps {_span([e['adamw_selected_steps'] for e in exposure], 'd')}, "
            f"{_span(selected, '.3f')} token equivalents")
    above = [e for e in exposure if e["adamw_selected_token_equivalents"] > 1.0]
    if above:
        names = _join([str(e["seed"]) for e in above])
        values = _join([format(e["adamw_selected_token_equivalents"], ".3f") for e in above])
        text += (f"; the selected checkpoint{'' if len(above) == 1 else 's'} of seed{'' if len(above) == 1 else 's'} "
                 f"{names} {'is' if len(above) == 1 else 'are'} above one token equivalent ({values})")
    else:
        text += "; no selected checkpoint is above one token equivalent"
    text += (f". The AdamW runs actually trained {_span([e['adamw_trained_steps'] for e in exposure], 'd')} steps, "
             f"{_span(trained, '.3f')} token equivalents")
    beyond = [e for e in exposure if e["adamw_trained_steps"] != e["adamw_selected_steps"]]
    if beyond:
        text += ": " + _join([
            f"seed {e['seed']} trained {e['adamw_trained_steps']} steps = "
            f"{e['adamw_trained_token_equivalents']:.3f} token equivalents before the selection of step "
            f"{e['adamw_selected_steps']} = {e['adamw_selected_token_equivalents']:.3f}" for e in beyond])
    else:
        text += " (every seed selected its last measured step)"
    return (text + ". A compute comparison must count every step actually run. More exposure both repeats and adds "
            "training positions and can change generalization and geometry; there is no fixed-step or "
            "fixed-exposure control. PTQ sensitivity of longer-trained or larger models can differ.")


def caveats(payload: dict) -> list[str]:
    analysis, protocol = payload["analysis"], payload["protocol"]
    env = payload["meta"]["environment"]
    capability = env.get("capability")
    device = f"{env['device']} (sm_{capability[0]}{capability[1]})" if capability else "CPU (no CUDA device)"
    if analysis is None:
        families = gap = hardware = ""
    else:
        cells, contrasts = len(analysis["contrasts"][0][FAMILIES[0]]), len(analysis["contrasts"])
        families = (f" Holm-Bonferroni is applied to the paired-t p-values over the {cells} cells of one family of "
                    f"one contrast (p_holm) and over the {cells * contrasts} cells of that family across the "
                    f"{contrasts} contrasts (p_holm_across_contrasts).")
        gap = (f" It cannot isolate which of them causes the mean BF16 test gap of "
               f"{analysis['iso_loss']['test_gap_muon_minus_adamw']['mean']:+.4f} nats (Muon minus AdamW).")
        hw = analysis["hardware"]
        hardware = (f" Rows with alpha = 0 are copies of their unscaled twin rows, not new measurements: "
                    f"{hw['w4a4_inherited_zero_alpha_rows']} of the {hw['w4a4_rows']} hardware W4A4 rows and "
                    f"{hw['decode_inherited_zero_alpha_matrices']} of the {hw['decode_checked_matrices']} matrix "
                    f"decode comparisons are inherited (analysis.hardware gives the unique and the inherited "
                    f"counts).")
    preceded = [s["seed"] for s in payload["stage_a"]["seeds"]
                if next(r for r in s["rows"] if r["arm"] == ADAMW)["steps"]
                < next(r for r in s["rows"] if r["arm"] == ADAMW)["trained_steps"]]
    selected_note = (f" (preceding the stopping step on seed{'' if len(preceded) == 1 else 's'} "
                     f"{_join([str(s) for s in preceded])})" if preceded else "")
    notes = [
        f"Five paired seeds give 4 degrees of freedom. The seeds share one corpus, one calibration draw and one "
        f"set of test documents; the intervals condition on these and on the val-selected methods, and they omit "
        f"the variation of new documents, of a new calibration draw and of a repeated method selection. The "
        f"Student-t p-values and 95% intervals assume normal paired differences (df = 4).{families} The exact "
        f"two-sided sign-flip p-value cannot be below {SIGN_FLIP_MIN_P}, so it cannot pass a 0.05 threshold, with "
        f"or without a correction. Absence of significance is not evidence of equal PTQ sensitivity. Inference is "
        f"restricted to the {payload['preset']} preset in this training regime on {device}.",
        f"AdamW is one run per seed, stopped on val.bin, and the selected step's weights{selected_note} are "
        "evaluated without retraining. The val match of Stage B is therefore the Stage A match by construction. "
        "test.bin is held out from the canonical selection rules: training, stopping, checkpoint selection, "
        "calibration, alpha selection and method selection read train.bin and val.bin only. It is not a previously "
        "unseen study-level holdout: seed-0 shakedown runs (timing_seed0.log and timing_seed0b.log; shakedown JSON "
        "SHA-256 39f2d27e...) scored the same test.bin before the initial commit 05ea4ae and the amendment "
        "0355d16, and the logs cannot show whether a person used those scores for a design decision. Training is "
        "not bit-reproducible on this host: fresh AdamW runs of one configuration differed by 0.06 to 0.07 nats "
        "on val.bin in the design pilot, so a retrain gives other weights and another stopping step.",
        "Iso-loss is not iso-compute and not iso-schedule: AdamW draws more training tokens than Muon "
        "(adamw_over_muon_tokens per seed) and ends with a seed-dependent number of floor-LR steps "
        "(adamw_floor_lr_steps), while Muon ends when its cosine reaches the floor. Validation stopping, the "
        "additional token exposure, the floor-LR phase and the fresh restart are coupled in this recipe, and the "
        f"experiment has no fixed-step or fixed-exposure control.{gap} The selection conditions the val gap on "
        "its target. The result is for this recipe; it supports no optimizer-wide PTQ ratio claim.",
        "The AdamW cosine length and the first measured step come from a seed-0 design pilot. The AdamW step cap "
        "was raised from 1280 to 2560 after the first canonical attempt (commit 05ea4ae) was stopped in Stage A: "
        "seed 1 had matched at step 640, and the AdamW run of seed 2 was 0.58 nats above Muon at step 400 and "
        "still 0.0925 nats above it at step 1280, where its log ends. That attempt has no Stage B scores; its "
        "startup verification read test.bin (hashes and document boundaries) and scored no model. The canonical "
        "run of the amended commit 0355d16 trained fresh trajectories, and every one stopped below step 1280. "
        "The pilot floor-LR probe is a sensitivity check at different losses, not an equal-loss proof: "
        "identity-GPTQ val NLL increase 0.0010 (NVFP4) / 0.0096 (INT2 g16) at val NLL 5.414 of a run stopped at "
        "848 steps, against 0.0010 / 0.0106 at 5.298 and 0.0008 / 0.0106 at 5.235 for fresh cosine runs of 544 "
        "and 608 steps (val NLL 5.235-5.414). In the canonical five-seed run, the primary AdamW test delta-NLL "
        "means at LR 0.001 are 1.38-4.02x those of the C4-01b AdamW recipe at LR 0.0015 (NVFP4 W4A16 2.60x, "
        "dynamic W4A4 1.42x, static W4A4 1.38x, MXFP4 1.97x, INT2 g16 3.54x, INT2 g32 4.02x); seeds, stopping, "
        "duration and LR changed together, so the difference is not attributed to the LR.",
        "Both cooldown arms branch from snapshots of the seed's Muon run, so their first "
        f"{protocol['adamw_cooldown']['muon_steps']} / {protocol['qat_cooldown']['muon_steps']} steps are that "
        "run's steps. Their BF16 baselines differ from Muon; bf16_nll_diff and nll_diff report the net effect. "
        f"The QAT cooldown targets identity-coordinate {protocol['qat_cooldown']['format']} round-to-nearest "
        "only; GPTQ and the rotations change the rounded values or the coordinates, so the arm is a test of that "
        "one target, not of QAT in general. The AdamW cooldown is an optimizer switch. No explicit spectral "
        "penalty was tested; stable rank and alignment are measured, not varied independently.",
        "The alpha of the activation-aware scaling is selected per checkpoint, format and rotation on val.bin "
        "with weight-only fake quantization; the hardware W4A4 rows reuse that alpha. The weight-only selection "
        "is a proxy: only the selected alpha and the unscaled row are scored on hardware, so the selection can "
        "miss the alpha that is best under hardware W4A4. The AWQ results hold for this selection rule; they do "
        "not show that a W4A4-tuned scaling is ineffective.",
        exposure_caveat(payload),
        "Hessians come from one forward pass of the full-precision model (parallel GPTQ). Static activation "
        "scales are calibrated on the BF16 model. Dynamic W4A4 scales are per forward batch (16 x 512 tokens).",
        f"Layer-geometry p-values are descriptive and not corrected for multiplicity.{hardware}",
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


def _nonneg(x) -> bool:
    """A finite, nonnegative JSON number (a loss or a duration)."""
    return _is_number(x) and x >= 0


def _is_int(x, low: int = 0) -> bool:
    """A JSON integer of at least ``low``: ``True`` and ``1.0`` are not integers."""
    return type(x) is int and x >= low


def _optional_str(x) -> bool:
    return x is None or isinstance(x, str)


def _same(found, expected) -> bool:
    """Deep equality with identical JSON types: ``True`` is not ``1``, and ``1`` is not ``1.0``."""
    if type(found) is not type(expected):
        return False
    if isinstance(expected, dict):
        return found.keys() == expected.keys() and all(_same(found[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(found) == len(expected) and all(_same(x, y) for x, y in zip(found, expected))
    return found == expected


def _same_types(found, template) -> bool:
    """``found`` has the keys and the JSON types of ``template``; list lengths are free."""
    if type(found) is not type(template):
        return False
    if isinstance(template, dict):
        return found.keys() == template.keys() and all(_same_types(found[k], v) for k, v in template.items())
    if isinstance(template, list):
        return not template or all(_same_types(x, template[0]) for x in found)
    return True


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


def _timestamp(text, field: str) -> datetime:
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field}: invalid ISO timestamp {text!r}") from exc


SCRIPT_REPO_PARTS = ("research", "quant_ptq", SCRIPT_NAME)


def _is_python_executable(exe: str) -> bool:
    return bool(re.fullmatch(r"python(\d+(\.\d+)*)?(\.exe)?", Path(exe).name.lower()))


def _is_driver_script(arg: str) -> bool:
    parts = Path(arg).parts
    return parts == (SCRIPT_NAME,) or (len(parts) >= len(SCRIPT_REPO_PARTS)
                                       and parts[-len(SCRIPT_REPO_PARTS):] == SCRIPT_REPO_PARTS)


def _cli_option_present(argv: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def _parsed_command(command, label: str) -> argparse.Namespace:
    """``command`` as a runnable invocation of this script, parsed by its own CLI."""
    _require(isinstance(command, list) and len(command) >= 2 and all(isinstance(x, str) and x for x in command)
             and _is_python_executable(command[0]), f"{label} must be a Python invocation")
    _require(_is_driver_script(command[1]), f"{label} is not an invocation of {SCRIPT_NAME}")
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return parse_args(command[2:])
    except SystemExit as exc:
        raise ValueError(f"{label} is not a valid runnable invocation") from exc


def expected_coverage(split_tokens: int, seq_len: int, max_windows: int | None) -> dict:
    """Windows, scored targets and token span of ``q.evaluate_split`` on a split of ``split_tokens`` tokens."""
    windows = (split_tokens - 1) // seq_len
    if max_windows is not None:
        windows = min(windows, max_windows)
    return {"tokens": windows * seq_len, "windows": windows, "span": [0, windows * seq_len + 1],
            "split_tokens": split_tokens}


def _validate_protocol(protocol, *, canonical: bool) -> None:
    free = "max_eval_windows"  # None (complete splits) or a window limit
    _require(isinstance(protocol, dict) and free in protocol
             and _same_types({k: v for k, v in protocol.items() if k != free},
                             {k: v for k, v in CANONICAL.items() if k != free})
             and (protocol[free] is None or _is_int(protocol[free], 1)),
             "protocol keys or value types differ from the schema")
    seeds = protocol["seeds"]
    _require(seeds and all(type(s) is int for s in seeds) and len(set(seeds)) == len(seeds),
             "protocol.seeds must be distinct integers")
    if canonical:
        _require(_same(protocol, CANONICAL), "protocol differs from the canonical C5-02 protocol")


DATA_KEYS = {"source", "encoding", "splits", "directory", "sha256_verified", "expected_sha256", "roles"}


def _validate_data(data, *, canonical: bool) -> None:
    _require(isinstance(data, dict) and set(data) == DATA_KEYS, "data keys differ from the schema")
    _require(_same(data["roles"], ROLES),
             "split roles differ: train for training and calibration, val for selection, test for evaluation")
    splits = data["splits"]
    _require(isinstance(splits, dict) and set(splits) == {"train", "val", "test"},
             "data.splits must hold train, val and test")
    for name, entry in splits.items():
        _require(isinstance(entry, dict) and set(entry) == {"tokens", "sha256"} and _is_int(entry["tokens"], 1)
                 and isinstance(entry["sha256"], str) and entry["sha256"],
                 f"data.splits.{name} must hold a token count and a SHA-256")
    source = data["source"]
    _require(isinstance(source, dict) and source
             and all(v is None or type(v) in (str, bool, int) for v in source.values()),
             "data.source must be a flat description of the corpus")
    _require(isinstance(data["encoding"], str) and data["encoding"], "data.encoding must name the tokenizer")
    _require(isinstance(data["directory"], str) and data["directory"], "data.directory invalid")
    _require(type(data["sha256_verified"]) is bool, "data.sha256_verified must be a bool")
    found, expected = {k: v["sha256"] for k, v in splits.items()}, data["expected_sha256"]
    _require(_same(expected, found) if data["sha256_verified"] else expected is None,
             "data.expected_sha256 must be the verified shard hashes (None without verification)")
    if canonical:
        _require(data["sha256_verified"] is True and _same(found, dict(q.EXPECTED_SHA256)),
                 "shard SHA-256 values differ from the pinned train/val/test hashes")
        _require(_same(source, CANONICAL_DATA["source"]) and data["encoding"] == CANONICAL_DATA["encoding"],
                 "data.source or data.encoding differs from the canonical FineWeb-Edu GPT-2 shards")
        _require(_same({k: v["tokens"] for k, v in splits.items()}, CANONICAL_DATA["tokens"]),
                 "split token counts differ from the pinned canonical shards")


META_KEYS = {"git_sha", "git_dirty", "command", "environment", "numerics", "code_sha256", "code_modules",
             "unit_config_hash", "checkpoint_dir", "timestamp_start", "timestamp_end", "loadavg_start",
             "loadavg_end", "gpu_telemetry", "stage_seconds", "total_seconds", "unit_log", "splits_opened"}
NUMERICS = {"allow_tf32": False, "cudnn_allow_tf32": False, "allow_fp16_reduced_precision_reduction": False,
            "allow_bf16_reduced_precision_reduction": False,
            "autocast": "bfloat16 forward on CUDA (harness_tier2_real._loss)"}
ENVIRONMENT_KEYS = {"python", "platform", "torch", "cuda", "cudnn", "host_cpus", "numpy", "env", "triton"}
ENVIRONMENT_CUDA_KEYS = {"device", "capability", "driver"}
ENVIRONMENT_VARIABLES = {"CUDA_VISIBLE_DEVICES", "PYTHONPATH", "OMP_NUM_THREADS"}
TELEMETRY_KEYS = {"time", "clocks.sm", "clocks.mem", "temperature.gpu", "power.draw", "utilization.gpu", "pstate",
                  "memory.used", "clocks_event_reasons.active"}


def _validate_environment(env, *, canonical: bool) -> None:
    """``q.environment``: the runtime, and the device fields exactly when the run was on CUDA."""
    _require(isinstance(env, dict) and set(env) in (ENVIRONMENT_KEYS, ENVIRONMENT_KEYS | ENVIRONMENT_CUDA_KEYS),
             "meta.environment keys differ from the schema")
    _require(all(isinstance(env[k], str) and env[k] for k in ("python", "platform", "torch", "numpy"))
             and all(_optional_str(env[k]) for k in ("cuda", "triton"))
             and all(env[k] is None or _is_int(env[k]) for k in ("cudnn", "host_cpus"))
             and isinstance(env["env"], dict) and set(env["env"]) == ENVIRONMENT_VARIABLES
             and all(_optional_str(v) for v in env["env"].values()), "meta.environment invalid")
    if "device" in env:
        capability = env["capability"]
        _require(isinstance(env["device"], str) and env["device"] and _optional_str(env["driver"])
                 and isinstance(capability, list) and len(capability) == 2 and all(_is_int(x) for x in capability),
                 "meta.environment device fields invalid")
    if canonical:
        _require(all(env.get(k) is not None for k in ENV_KEYS) and env["capability"] >= [12, 0],
                 "a canonical result needs an sm_120+ CUDA device and a recorded runtime")


def _validate_telemetry(sample, where: str, *, cuda: bool, canonical: bool) -> None:
    """One ``h.gpu_telemetry`` sample: None on the CPU, a recorded error, or the nvidia-smi fields."""
    if sample is None:
        _require(not cuda and not canonical, f"{where}: a CUDA run records a sample or its error")
        return
    _require(cuda and isinstance(sample, dict) and sample, f"{where} invalid")
    if "error" in sample:
        _require(set(sample) == {"error"} and isinstance(sample["error"], str), f"{where} invalid")
    else:
        _require(set(sample) == TELEMETRY_KEYS and _is_number(sample["time"])
                 and all(_is_number(v) or isinstance(v, str) for v in sample.values())
                 and (not canonical or _is_number(sample["memory.used"])), f"{where} invalid")


def _validate_meta(payload: dict, *, canonical: bool, schema: str) -> None:
    """``schema`` is the schema version of the measuring run: the hashes in ``meta`` were made under it."""
    meta = payload["meta"]
    _require(isinstance(meta, dict) and set(meta) == META_KEYS, "meta keys differ from the schema")
    _require(meta["git_sha"] is None or _is_hex(meta["git_sha"], 40), "meta.git_sha must be 40-hex or None")
    if meta["git_sha"] is not None:
        _require_git_commit(meta["git_sha"], "meta.git_sha")
    _require(meta["git_dirty"] is None or type(meta["git_dirty"]) is bool, "meta.git_dirty must be a bool")
    command = meta["command"]
    _require(isinstance(command, list) and len(command) >= 2 and all(isinstance(x, str) and x for x in command)
             and _is_python_executable(command[0]), "meta.command must be a Python invocation")
    env = meta["environment"]
    _validate_environment(env, canonical=canonical)
    _require(_same(meta["numerics"], NUMERICS), "full-precision numerics are required")
    modules = meta["code_modules"]
    _require(isinstance(modules, list) and all(isinstance(m, str) and m.startswith("research.") for m in modules)
             and len(set(modules)) == len(modules) and DRIVER_MODULE in modules,
             "meta.code_modules must list the driver and its research.* imports")
    _require(_is_hex(meta["code_sha256"], 64) and _is_hex(meta["unit_config_hash"], 16), "meta hashes invalid")
    data_sha = {k: v["sha256"] for k, v in payload["data"]["splits"].items()}
    _require(meta["unit_config_hash"] == unit_config_hash(payload["protocol"], payload["config"], meta["code_sha256"],
                                                          env, data_sha, schema), "meta.unit_config_hash differs")
    stages = meta["stage_seconds"]
    _require(isinstance(stages, dict) and set(stages) == {"stage_a", "stage_b"}
             and all(_nonneg(v) for v in stages.values()), "meta.stage_seconds invalid")
    _require(_nonneg(meta["total_seconds"]) and meta["total_seconds"] >= sum(stages.values()) - 1e-6,
             "meta.total_seconds is below the sum of the stage times")
    start = _timestamp(meta["timestamp_start"], "meta.timestamp_start")
    end = _timestamp(meta["timestamp_end"], "meta.timestamp_end")
    _require(start.timestamp() >= 1_700_000_000 and end >= start
             and abs((end - start).total_seconds() - meta["total_seconds"]) <= 5.0,
             "meta timestamps are inconsistent with meta.total_seconds")
    for field in ("loadavg_start", "loadavg_end"):
        _require(isinstance(meta[field], list) and len(meta[field]) == 3
                 and all(_nonneg(x) for x in meta[field]), f"meta.{field} invalid")
    telemetry = meta["gpu_telemetry"]
    _require(isinstance(telemetry, dict) and set(telemetry) == {"start", "after_stage_a", "end"},
             "meta.gpu_telemetry keys differ")
    for side, sample in telemetry.items():  # one nvidia-smi sample each, or its recorded error
        _validate_telemetry(sample, f"meta.gpu_telemetry.{side}", cuda="device" in env, canonical=canonical)
    log = meta["unit_log"]
    _require(isinstance(log, dict) and set(log) == {"resume", "units_loaded", "units_run"}
             and type(log["resume"]) is bool and all(_is_int(log[k]) for k in ("units_loaded", "units_run"))
             and log["units_loaded"] + log["units_run"] == len(payload["protocol"]["seeds"]) * (1 + len(ARMS))
             and (log["resume"] or log["units_loaded"] == 0),
             "meta.unit_log must account for one Stage A unit per seed and one Stage B unit per checkpoint, and a "
             "run without --resume loads no unit")
    _require(meta["splits_opened"] == ["train", "val", "test"],
             "Shards.open must map the splits in the order train, val, test (test.bin for the Stage B scoring)")
    _require(isinstance(meta["checkpoint_dir"], str) and meta["checkpoint_dir"], "meta.checkpoint_dir invalid")
    if canonical:
        _require(_is_hex(meta["git_sha"], 40) and meta["git_dirty"] is False,
                 "a canonical result needs a 40-hex git_sha and a clean tree")
        _require(_is_driver_script(command[1]) and "--quick" not in command,
                 "meta.command is not the canonical invocation")
        parsed = _parsed_command(command, "meta.command")
        _require(parsed.canonical is True and parsed.postprocess_from_git is None,
                 "meta.command does not select the canonical protocol")
        _require(parsed.resume == log["resume"], "meta.command --resume flag differs from meta.unit_log.resume")
        if _cli_option_present(command[2:], "--checkpoint-dir"):
            _require(str(parsed.checkpoint_dir) == meta["checkpoint_dir"],
                     "meta.command --checkpoint-dir differs from meta.checkpoint_dir")
        else:
            _require(Path(meta["checkpoint_dir"]).parts[-2:] == ("artifacts", "c5_02_checkpoints"),
                     "meta.checkpoint_dir differs from the canonical checkpoint directory")
        if _cli_option_present(command[2:], "--data-dir"):
            _require(str(parsed.data_dir) == payload["data"]["directory"],
                     "meta.command --data-dir differs from data.directory")
    if payload["postprocess"] is None:  # a measuring run: the producer is this code
        _require(modules == list(code_modules()), "meta.code_modules differ from the static import closure")
        if canonical:
            _require(modules == list(_code_modules_at_commit(meta["git_sha"])),
                     "meta.code_modules differ from the static import closure at meta.git_sha")
            _require(meta["code_sha256"] == _code_sha256_at_commit(meta["git_sha"], tuple(modules)),
                     "meta.code_sha256 differs from the committed git blobs")
            commit_schema, _ = _driver_capabilities_at_commit(meta["git_sha"])
            _require(commit_schema == SCHEMA_VERSION,
                     f"meta.git_sha declares schema {commit_schema!r}, not {SCHEMA_VERSION!r}")


TRAINING_KEYS = {"arm", "seed", "steps", "warmup_steps", "lr", "auxiliary_lr", "weight_decay", "min_lr_ratio",
                 "tokens", "train_epochs", "orthogonalizer", "init_fingerprint", "stream_fingerprint", "train_loss",
                 "last_train_loss", "train_loss_note", "train_curve", "train_seconds", "checkpoint_sha256",
                 "val_nll", "val_ppl", "val_tokens", "evaluated_splits"}


def _validate_training(row, path: str, config: dict, protocol: dict, *, seed: int, train_tokens: int,
                       val_tokens: int) -> None:
    extra = {MUON: {"snapshots"}, ADAMW: {"cosine_steps", "trained_steps", "eval_seconds", "evaluations"},
             COOL_ADAMW: {"branch", "cooldown"}, COOL_QAT: {"branch", "cooldown"}}
    _require(isinstance(row, dict) and row.get("arm") in ARMS and set(row) == TRAINING_KEYS | extra[row["arm"]],
             f"{path}: keys differ")
    _require(_same(row["seed"], seed), f"{path}: wrong seed")
    arm, muon, total = row["arm"], protocol["muon"], protocol["muon"]["steps"]
    schedule_steps = protocol["adamw"]["cosine_steps"] if arm == ADAMW else total
    _require(_is_int(row["steps"], 1) and _same(row["warmup_steps"], c4.warmup_for(schedule_steps))
             and _same(row["weight_decay"], WEIGHT_DECAY) and _same(row["min_lr_ratio"], MIN_LR_RATIO),
             f"{path}: schedule differs from warmup max(20, T // 10), weight decay 0.01, min LR ratio 0.1")
    step_tokens = config["micro_batch"] * config["grad_accum"] * config["sequence_length"]
    _require(_same(row["tokens"], row["steps"] * step_tokens),
             f"{path}: token count differs from steps * {step_tokens}")
    _assert_close(row["train_epochs"], row["tokens"] / train_tokens, f"{path}.train_epochs")
    if arm == ADAMW:
        spec = protocol["adamw"]
        _require(_same(row["lr"], spec["lr"]) and row["auxiliary_lr"] is None
                 and _same(row["cosine_steps"], spec["cosine_steps"]) and _is_int(row["trained_steps"], 1)
                 and _nonneg(row["eval_seconds"]), f"{path}: AdamW run differs from the protocol")
        _require(row["orthogonalizer"] == ADAMW_ORTHOGONALIZER, f"{path}: the AdamW arm must record fused AdamW")
    else:
        _require(row["steps"] == total and _same(row["lr"], muon["lr"])
                 and _same(row["auxiliary_lr"], muon["auxiliary_lr"]),
                 f"{path}: Muon-based arm differs from the protocol")
        _require(isinstance(row["orthogonalizer"], str) and row["orthogonalizer"]
                 and row["orthogonalizer"] != ADAMW_ORTHOGONALIZER,
                 f"{path}: a Muon-based arm must record its orthogonalizer")
    _require(_is_hex(row["checkpoint_sha256"], 64) and _is_hex(row["init_fingerprint"], 16)
             and _is_hex(row["stream_fingerprint"], 16), f"{path}: missing hashes")
    _require(row["evaluated_splits"] == ["val"] and _nonneg(row["val_nll"]), f"{path}: Stage A rows read val only")
    _require(_same(row["val_tokens"], val_tokens), f"{path}: val.bin is not covered completely")
    _assert_close(row["val_ppl"], math.exp(row["val_nll"]), f"{path}.val_ppl")
    _require(_nonneg(row["train_loss"]) and _nonneg(row["last_train_loss"]) and _nonneg(row["train_seconds"])
             and row["train_loss_note"] == TRAIN_LOSS_NOTE,
             f"{path}: training losses and times must be finite and nonnegative, with the declared loss note")
    curve = row["train_curve"]
    cooldown = {COOL_ADAMW: "adamw_cooldown", COOL_QAT: "qat_cooldown"}
    first = protocol[cooldown[arm]]["muon_steps"] if arm in cooldown else 0  # a cooldown arm logs its own phase
    count = row["steps"] - first
    _require(isinstance(curve, list) and all(isinstance(p, list) and len(p) == 2 and _nonneg(p[1]) for p in curve)
             and _same([p[0] for p in curve], [first + i + 1 for i in range(count) if i == 0 or (i + 1) % 10 == 0]),
             f"{path}: train curve differs from the schedule")
    if curve and curve[-1][0] == row["steps"]:  # the last optimizer step is a curve point
        _assert_close(row["last_train_loss"], curve[-1][1], f"{path}.last_train_loss")


def _validate_adamw_run(row: dict, muon: dict, path: str, spec: dict) -> None:
    """Replay the predeclared val grid and stop rule of the AdamW run from its stored measurements."""
    evaluations = row["evaluations"]
    _require(isinstance(evaluations, list) and evaluations, f"{path}: the AdamW run has no measured step")
    for e in evaluations:
        _require(isinstance(e, dict)
                 and set(e) == {"step", "lr_multiplier", "val_nll", "val_tokens", "val_gap", "checkpoint_sha256"}
                 and _nonneg(e["val_nll"]) and _is_hex(e["checkpoint_sha256"], 64)
                 and _same(e["val_tokens"], muon["val_tokens"]), f"{path}: AdamW measurement fields differ")
    grid = adamw_eval_steps(spec)
    _require(_same([e["step"] for e in evaluations], grid[:len(evaluations)]),
             f"{path}: measured AdamW steps differ from the predeclared val grid")
    for e in evaluations:
        _assert_close(e["val_gap"], e["val_nll"] - muon["val_nll"], f"{path}.evaluations[{e['step']}].val_gap")
        _assert_close(e["lr_multiplier"], adamw_multiplier(e["step"] - 1, spec),
                      f"{path}.evaluations[{e['step']}].lr_multiplier")
    gaps = [e["val_nll"] - muon["val_nll"] for e in evaluations]  # recomputed, as in ``select_adamw``
    _require(not any(adamw_stops(g) for g in gaps[:-1]),
             f"{path}: the AdamW run went on after it was at or below Muon")
    _require(adamw_stops(gaps[-1]) or evaluations[-1]["step"] == grid[-1],
             f"{path}: the AdamW run stopped before it reached Muon or the step cap")
    chosen = min(zip(evaluations, gaps), key=lambda pair: (abs(pair[1]), pair[0]["step"]))[0]
    _require(_same([row["trained_steps"], row["steps"], row["val_nll"], row["val_tokens"], row["checkpoint_sha256"]],
                   [evaluations[-1]["step"], chosen["step"], chosen["val_nll"], chosen["val_tokens"],
                    chosen["checkpoint_sha256"]]),
             f"{path}: the AdamW record is not the measured step nearest to Muon")


def _validate_stage_a(payload: dict, *, canonical: bool) -> dict:
    stage, protocol, config = payload["stage_a"], payload["protocol"], payload["config"]
    seeds, tolerance, splits = protocol["seeds"], protocol["tolerance_nats"], payload["data"]["splits"]
    val_tokens = expected_coverage(splits["val"]["tokens"], config["sequence_length"],
                                   protocol["max_eval_windows"])["tokens"]
    _require(isinstance(stage, dict) and set(stage) == {"split_view", "splits_opened_at_end", "seeds", "seconds"},
             "stage_a keys differ")
    _require(stage["split_view"] == ["train", "val"] and stage["splits_opened_at_end"] == ["train", "val"],
             "Stage A must run on a train + val mapping, before Shards.open maps test.bin for the Stage B scoring")
    _require(_nonneg(stage["seconds"]), "stage_a.seconds must be a finite, nonnegative duration")
    _require(isinstance(stage["seeds"], list) and all(isinstance(u, dict) for u in stage["seeds"])
             and _same([u.get("seed") for u in stage["seeds"]], seeds), "Stage A needs one unit per seed, in order")
    _require(not any("test" in key for key in _keys(stage)), "Stage A holds a test.bin field")
    cool = {COOL_ADAMW: protocol["adamw_cooldown"], COOL_QAT: protocol["qat_cooldown"]}
    by_seed, orthogonalizers = {}, set()
    for unit in stage["seeds"]:
        seed, rows = unit["seed"], unit.get("rows")
        path = f"stage_a[{seed}]"
        _require(set(unit) == {"seed", "split_view", "rows", "selection", "checkpoints", "seconds",
                               "resumed_from_unit_log"}, f"{path}: keys differ")
        _require(_nonneg(unit["seconds"]), f"{path}.seconds must be a finite, nonnegative duration")
        _require(type(unit["resumed_from_unit_log"]) is bool, f"{path}.resumed_from_unit_log must be a bool")
        _require(unit["split_view"] == ["train", "val"], f"{path}: Stage A must not see test.bin")
        _require(isinstance(rows, list) and [isinstance(r, dict) and r.get("arm") for r in rows]
                 == [MUON, COOL_ADAMW, COOL_QAT, ADAMW],
                 f"{path}: rows must be Muon, the two cooldowns and the AdamW run")
        for i, r in enumerate(rows):
            _validate_training(r, f"{path}.rows[{i}]", config, protocol, seed=seed,
                               train_tokens=splits["train"]["tokens"], val_tokens=val_tokens)
        _require(len({(r["init_fingerprint"], r["stream_fingerprint"]) for r in rows}) == 1,
                 f"{path}: the arms of a seed must share the initial state and the batch stream")
        muon = rows[0]
        _require(isinstance(muon["snapshots"], dict)
                 and sorted(muon["snapshots"]) == sorted({str(c["muon_steps"]) for c in cool.values()})
                 and all(_is_hex(v, 64) for v in muon["snapshots"].values()), f"{path}: Muon snapshots differ")
        orthogonalizers |= {r["orthogonalizer"] for r in rows[:3]}
        for r in rows[1:3]:
            spec = cool[r["arm"]]
            _require(_same(r["branch"], {"from_arm": MUON, "from_step": spec["muon_steps"],
                                         "snapshot_sha256": muon["snapshots"][str(spec["muon_steps"])],
                                         "cooldown_steps": spec["cooldown_steps"]})
                     and spec["muon_steps"] + spec["cooldown_steps"] == protocol["muon"]["steps"],
                     f"{path}: {r['arm']} does not branch from the Muon snapshot of the protocol")
            _require(_same(r["cooldown"], cooldown_recipe(r["arm"], protocol)),
                     f"{path}: the cooldown record of {r['arm']} differs from the locked recipe of the protocol")
        _validate_adamw_run(rows[3], muon, path, protocol["adamw"])
        _assert_close(unit["selection"], select_adamw(rows, tolerance), f"{path}.selection")
        final = {r["arm"]: r for r in rows}
        _assert_close(unit["checkpoints"], {arm: {"file": checkpoint_name(seed, arm),
                                                  "sha256": final[arm]["checkpoint_sha256"],
                                                  "val_nll": final[arm]["val_nll"]} for arm in ARMS},
                      f"{path}.checkpoints")
        by_seed[seed] = final
    _require(len(orthogonalizers) == 1 and (not canonical or orthogonalizers == {CANONICAL_ORTHOGONALIZER}),
             "Muon and both cooldown arms must record one orthogonalizer (the SS3_54 Triton kernel when canonical)")
    return by_seed


GRID_KEYS = {"format", "method", "mode", "rotation", "rounding", "activation_aware_scaling", "seconds",
             *(f"{s}_{k}" for s in SPLITS for k in ("nll", "ppl", "tokens", "delta_nll", "relative_delta_ppl_pct"))}
FAKEQUANT_KEYS = {"quantize_seconds", "gptq_damping_max", "geometry", *(f"{k}_mean" for k in GEOMETRY)}
CALIBRATION_KEYS = {"split", "seed", "sequences", "seq_len", "tokens", "starts"}
BASELINE_KEYS = {"nll", "ppl", "tokens", "windows", "span", "split_tokens"}
BUTTERFLY_FIT_KEYS = {"steps", "lr", "layers", "layers_improved", "weight_sq_error_hadamard_start",
                      "weight_sq_error_fitted", "relative_error_reduction"}
COPY_LABELS = ("method", "activation_aware_scaling", "awq_alpha")  # the fields that an alpha = 0 copy changes


def _validate_calibration(cal, protocol: dict, seq_len: int, train_tokens: int) -> None:
    """The record of ``q.calibration_windows``: the exact seeded draw of distinct, aligned train.bin windows."""
    _require(isinstance(cal, dict) and set(cal) == CALIBRATION_KEYS, "Calibration record keys differ from the schema")
    sequences, available = protocol["calibration_sequences"], train_tokens // seq_len
    _require(cal["split"] == "train" and _same(cal["sequences"], sequences)
             and _same(cal["seed"], protocol["calibration_seed"]) and _same(cal["seq_len"], seq_len)
             and _same(cal["tokens"], sequences * seq_len) and 0 < sequences <= available and cal["seed"] >= 0,
             "Calibration windows differ from the protocol (train.bin, seeded, aligned, distinct)")
    draw = np.sort(np.random.default_rng(cal["seed"]).choice(available, size=sequences, replace=False)) * seq_len
    _require(_same(cal["starts"], [int(s) for s in draw]),
             "Calibration starts differ from the seeded train.bin draw of the protocol")


def _validate_butterfly_fits(c: dict, path: str, protocol: dict, layers: int, selected: dict) -> None:
    """One fit record per projection kind for every fitted butterfly candidate that was evaluated."""
    expected = set()
    for fmt in protocol["weight_formats"]:
        for method in methods_of(c["arm"], protocol):
            family, _, awq = parse_method(method)
            if family.startswith("adaptive_butterfly_b") and (
                    not awq or selected[(fmt, method)]["selected_alpha"] != 0):  # alpha = 0 reuses the plain fit
                expected.add(f"{fmt}/{method}")
    fits = c["butterfly_fits"]
    _require(isinstance(fits, dict) and set(fits) == expected,
             f"{path}: butterfly_fits must hold exactly the fitted butterfly candidates of the grid")
    for key, kinds in fits.items():
        _require(isinstance(kinds, dict) and list(kinds) == list(q.LINEAR_KINDS),
                 f"{path}.butterfly_fits[{key}]: one fit per projection kind is required")
        for kind, fit in kinds.items():
            where = f"{path}.butterfly_fits[{key}].{kind}"
            _require(isinstance(fit, dict) and set(fit) == BUTTERFLY_FIT_KEYS, f"{where}: keys differ")
            start, fitted = fit["weight_sq_error_hadamard_start"], fit["weight_sq_error_fitted"]
            _require(_same(fit["steps"], protocol["butterfly_steps"]) and _same(fit["lr"], BUTTERFLY_LR)
                     and _same(fit["layers"], layers) and _is_int(fit["layers_improved"])
                     and fit["layers_improved"] <= layers, f"{where}: steps, LR or layer counts differ")
            _require(_nonneg(start) and _nonneg(fitted) and fitted <= start
                     and (fit["layers_improved"] > 0 or fitted == start),
                     f"{where}: fitted error must satisfy 0 <= fitted <= start, and equal start when "
                     "layers_improved == 0")
            _assert_close(fit["relative_error_reduction"], 1.0 - fitted / max(start, 1e-30),
                          f"{where}.relative_error_reduction")


def _validate_stage_b(payload: dict, final: dict, *, canonical: bool) -> None:
    stage, protocol, config = payload["stage_b"], payload["protocol"], payload["config"]
    seeds, layers, seq_len = protocol["seeds"], config["layers"], config["sequence_length"]
    matrices, names, splits = layers * len(q.LINEAR_KINDS), layer_names(layers), payload["data"]["splits"]
    _require(isinstance(stage, dict) and set(stage) == {"calibration", "checkpoints", "seconds"},
             "stage_b keys differ")
    _require(_nonneg(stage["seconds"]), "stage_b.seconds must be a finite, nonnegative duration")
    _validate_calibration(stage["calibration"], protocol, seq_len, splits["train"]["tokens"])
    _require(isinstance(stage["checkpoints"], list) and all(isinstance(c, dict) for c in stage["checkpoints"])
             and _same([[c.get("seed"), c.get("arm")] for c in stage["checkpoints"]],
                       [[s, a] for s in seeds for a in ARMS]),
             "Stage B needs one checkpoint per (seed, arm), in order")
    alphas, damping = protocol["awq"]["alphas"], [protocol["gptq_damp"] * 10 ** k for k in range(GPTQ_DAMP_TRIES)]
    coverage = {split: expected_coverage(splits[split]["tokens"], seq_len, protocol["max_eval_windows"])
                for split in SPLITS}
    available = set()
    for c in stage["checkpoints"]:
        seed, arm = c["seed"], c["arm"]
        path = f"stage_b[{seed}, {arm}]"
        _require(set(c) == {"seed", "arm", "checkpoint_file", "checkpoint_sha256", "checkpoint_sha256_after_ptq",
                            "baseline_val", "baseline_test", "diagnostics", "butterfly_fits", "grid", "awq_selection",
                            "best", "hardware_w4a4", "seconds", "resumed_from_unit_log"}, f"{path}: keys differ")
        _require(_nonneg(c["seconds"]), f"{path}.seconds must be a finite, nonnegative duration")
        _require(type(c["resumed_from_unit_log"]) is bool, f"{path}.resumed_from_unit_log must be a bool")
        trained = final[seed][arm]
        _require(c["checkpoint_sha256"] == c["checkpoint_sha256_after_ptq"] == trained["checkpoint_sha256"]
                 and c["checkpoint_file"] == checkpoint_name(seed, arm),
                 f"{path}: Stage B must evaluate the saved Stage A checkpoint and restore it")
        base = {split: c[f"baseline_{split}"] for split in SPLITS}
        for split in SPLITS:  # complete coverage of both splits, recomputed from the split length
            cover = coverage[split]
            _require(isinstance(base[split], dict) and set(base[split]) == BASELINE_KEYS,
                     f"{path}.baseline_{split}: keys differ")
            _require(_same({k: base[split][k] for k in cover}, cover),
                     f"{path}: baseline does not cover the full {split}.bin ({cover['windows']} windows of "
                     f"{seq_len} tokens, span {cover['span']}, {cover['split_tokens']} split tokens)")
            _require(_nonneg(base[split]["nll"]), f"{path}.baseline_{split}.nll must be finite and nonnegative")
            _assert_close(base[split]["ppl"], math.exp(base[split]["nll"]), f"{path}.baseline_{split}.ppl")
        _require(abs(base["val"]["nll"] - trained["val_nll"]) <= 1e-9
                 and base["val"]["tokens"] == trained["val_tokens"],
                 f"{path}: the val NLL of the saved checkpoint differs from its Stage A value")
        hw = c["hardware_w4a4"]
        _require(isinstance(hw, dict) and set(hw) == {"available", "fp4_gemms_per_forward_bf16_baseline", "note"}
                 and type(hw["available"]) is bool, f"{path}: hardware_w4a4 keys differ")
        hardware = hw["available"]
        available.add(hardware)
        _require(hardware or not canonical, f"{path}: a canonical result needs hardware W4A4")
        _require(_same(hw["fp4_gemms_per_forward_bf16_baseline"], 0 if hardware else None),
                 f"{path}: the BF16 baseline must not run FP4 GEMMs")
        _require(_same(hw["note"], None if hardware else HARDWARE_SKIP_NOTE),
                 f"{path}: hardware_w4a4.note must be None with hardware W4A4 and the skip reason without it")
        diagnostics = c["diagnostics"]
        _require(isinstance(diagnostics, dict) and set(diagnostics) == {"layer_names", "per_layer", "summary"}
                 and isinstance(diagnostics["per_layer"], list), f"{path}: diagnostics keys differ")
        per_layer = diagnostics["per_layer"]
        _require(diagnostics["layer_names"] == names
                 and all(isinstance(r, dict) and set(r) == {"name", "kind", *LAYER_METRICS} for r in per_layer)
                 and [r["name"] for r in per_layer] == names
                 and all(r["kind"] == r["name"].rsplit(".", 1)[1] for r in per_layer),
                 f"{path}: diagnostics must cover the {matrices} projections in order")
        _require(all(_is_number(r[m]) for r in per_layer for m in LAYER_METRICS)
                 and all(1.0 - 1e-9 <= r["weight_stable_rank"] and r["weight_fro_norm"] > 0
                         and r["signal_alignment"] > 0 for r in per_layer), f"{path}: layer geometry out of range")
        _assert_close(diagnostics["summary"], summarize_layers(per_layer), f"{path}.diagnostics.summary")
        _require(isinstance(c["grid"], list) and all(isinstance(r, dict) for r in c["grid"])
                 and [(r.get("format"), r.get("method"), r.get("mode")) for r in c["grid"]]
                 == expected_combos(arm, protocol, hardware), f"{path}: PTQ grid differs from the protocol")
        selected = {}
        _require(isinstance(c["awq_selection"], list) and all(isinstance(s, dict) for s in c["awq_selection"])
                 and [(s.get("format"), s.get("method")) for s in c["awq_selection"]]
                 == [(f, m) for f in protocol["weight_formats"] for m in methods_of(arm, protocol)
                     if m.startswith("awq_")], f"{path}: AWQ selection table differs from the protocol")
        for s in c["awq_selection"]:
            where = f"{path}.awq_selection[{s['format']}/{s['method']}]"
            _require(set(s) == {"format", "method", "rotation", "alphas", "val_nll", "val_tokens",
                                "selected_alpha", "selected_on", "seconds"}, f"{where}: keys differ")
            _require(_same(s["alphas"], alphas) and isinstance(s["val_nll"], list) and len(s["val_nll"]) == len(alphas)
                     and all(_nonneg(v) for v in s["val_nll"]) and s["selected_on"] == "val"
                     and s["rotation"] == parse_method(s["method"])[0]
                     and _same(s["selected_alpha"], select_alpha(alphas, s["val_nll"])),
                     f"{where}: alpha is not the val.bin minimum of the protocol grid")
            _require(_same(s["val_tokens"], base["val"]["tokens"]),
                     f"{where}: alpha selection must cover the val windows")
            _require(_nonneg(s["seconds"]), f"{where}.seconds must be a finite, nonnegative duration")
            selected[(s["format"], s["method"])] = s
        _validate_butterfly_fits(c, path, protocol, layers, selected)
        plain = {(r["format"], r["method"], r["mode"]): r for r in c["grid"]}
        for (fmt, method), s in selected.items():
            twin = plain[(fmt, method.removeprefix("awq_"), fakequant_mode(fmt))]
            _require(0.0 not in alphas or s["val_nll"][alphas.index(0.0)] == twin["val_nll"],
                     f"{path}.awq_selection[{fmt}/{method}]: alpha = 0 must be the unscaled row")
        signal = {row["name"]: row["signal_alignment"] for row in per_layer}
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
            _require(_same([r["rotation"], r["rounding"], r["activation_aware_scaling"]], [family, rounding, awq]),
                     f"{where}: method fields differ")
            _require(_nonneg(r["seconds"]), f"{where}.seconds must be a finite, nonnegative duration")
            for split in SPLITS:
                _require(_same(r[f"{split}_tokens"], base[split]["tokens"]), f"{where}: {split} token count differs")
                _require(_nonneg(r[f"{split}_nll"]), f"{where}.{split}_nll must be finite and nonnegative")
                _assert_close(r[f"{split}_ppl"], math.exp(r[f"{split}_nll"]), f"{where}.{split}_ppl")
                _assert_close(r[f"{split}_delta_nll"], r[f"{split}_nll"] - base[split]["nll"],
                              f"{where}.{split}_delta_nll")
                _assert_close(r[f"{split}_relative_delta_ppl_pct"],
                              100.0 * (r[f"{split}_ppl"] / base[split]["ppl"] - 1.0),
                              f"{where}.{split}_relative_delta_ppl_pct")
            if "awq_alpha" in r:
                table = selected[(r["format"], r["method"])]
                _require(_same(r["awq_alpha"], table["selected_alpha"]),
                         f"{where}: alpha differs from the val selection")
                _require(not fake or r["val_nll"] == table["val_nll"][alphas.index(r["awq_alpha"])],
                         f"{where}: the row is not the candidate that the val selection measured")
                if r["awq_alpha"] == 0:  # a copy of the plain rotation + GPTQ row: every measured field is the twin's
                    twin = plain[(r["format"], r["method"].removeprefix("awq_"), r["mode"])]
                    _require(_same({k: v for k, v in r.items() if k not in COPY_LABELS},
                                   {k: v for k, v in twin.items() if k not in COPY_LABELS}),
                             f"{where}: alpha = 0 must be the unscaled row in every measured field")
            if fake:
                _require(_nonneg(r["quantize_seconds"]),
                         f"{where}.quantize_seconds must be a finite, nonnegative duration")
                lam = r["gptq_damping_max"]
                _require(lam is None if rounding == "rtn" else _is_number(lam) and any(
                    abs(lam - d) <= 1e-12 * d for d in damping),
                    f"{where}: gptq_damping_max must be None for round-to-nearest and gptq_damp * 10^k "
                    f"(k < {GPTQ_DAMP_TRIES}) for GPTQ")
                geometry = r["geometry"]
                _require(isinstance(geometry, dict) and set(geometry) == set(GEOMETRY) and all(
                    isinstance(geometry[k], list) and len(geometry[k]) == matrices
                    and all(_nonneg(v) for v in geometry[k]) for k in GEOMETRY),
                    f"{where}: geometry must hold one value per layer")
                for k in GEOMETRY:
                    _assert_close(r[f"{k}_mean"], math.fsum(geometry[k]) / matrices, f"{where}.{k}_mean")
                for i, name in enumerate(names):
                    predicted = geometry["weight_rel_error"][i] * geometry["error_alignment"][i] / signal[name]
                    _require(abs(geometry["output_rel_error"][i] - predicted)
                             <= 1e-6 * max(geometry["output_rel_error"][i], 1e-300),
                             f"{where}: output error of {name} is not weight error x error / signal alignment")
                if r["format"] == HW_FORMAT:
                    _require(r["hw_weights_decode_to_fakequant_weights"] is True
                             and _same(r["hw_weights_checked_matrices"], matrices),
                             f"{where}: hardware weights must decode to the fake-quant weights in all "
                             f"{matrices} matrices")
            else:
                _require(_same(r["fp4_gemms_per_forward"], matrices if hw_mode else 0),
                         f"{where}: expected {matrices if hw_mode else 0} FP4 GEMMs per forward pass")
            if r["mode"] == "w4a4_hw_static":
                _require(r["static_scale_calibration"] == STATIC_CALIBRATION,
                         f"{where}: static activation scales must come from the train.bin calibration windows")
                scales = r["static_global_scale_range"]
                _require(isinstance(scales, list) and len(scales) == 2 and all(_is_number(v) for v in scales)
                         and 0 < scales[0] <= scales[1],
                         f"{where}: static_global_scale_range must be [lo, hi] with finite 0 < lo <= hi")
        _assert_close(c["best"], best_methods(c["grid"], protocol["base_methods"]), f"{path}.best")
    _require(len(available) == 1, "hardware W4A4 must be available for every checkpoint of a run, or for none")


def _validate_counts(payload: dict) -> None:
    """Duplicated stage times, resume counts and the unique / inherited hardware counts, recounted from the rows."""
    meta, stage_a, stage_b = payload["meta"], payload["stage_a"], payload["stage_b"]
    for name, stage in (("stage_a", stage_a), ("stage_b", stage_b)):
        _assert_close(stage["seconds"], meta["stage_seconds"][name], f"{name}.seconds (meta.stage_seconds.{name})")
    resumed = sum(u["resumed_from_unit_log"] for u in stage_a["seeds"] + stage_b["checkpoints"])
    _require(resumed == meta["unit_log"]["units_loaded"],
             f"{resumed} units are marked as resumed, but meta.unit_log.units_loaded is "
             f"{meta['unit_log']['units_loaded']}")
    if payload["analysis"] is None:
        return
    matrices = payload["config"]["layers"] * len(q.LINEAR_KINDS)
    rows = [r for c in stage_b["checkpoints"] for r in c["grid"]]
    profiled = [r for r in rows if r["mode"] in HW_MODES]
    decoded = [r for r in rows if r["format"] == HW_FORMAT and r["mode"] == fakequant_mode(r["format"])]
    copies = lambda group: sum(r["activation_aware_scaling"] and r["awq_alpha"] == 0 for r in group)  # noqa: E731
    expected = {"w4a4_rows": len(profiled), "w4a4_unique_profiled_rows": len(profiled) - copies(profiled),
                "w4a4_inherited_zero_alpha_rows": copies(profiled),
                "decode_checked_rows": len(decoded), "decode_unique_checked_rows": len(decoded) - copies(decoded),
                "decode_inherited_zero_alpha_rows": copies(decoded),
                "decode_checked_matrices": matrices * len(decoded),
                "decode_unique_checked_matrices": matrices * (len(decoded) - copies(decoded)),
                "decode_inherited_zero_alpha_matrices": matrices * copies(decoded)}
    for section in ("analysis", "derived"):
        for key, value in expected.items():
            _require(_same(payload[section]["hardware"].get(key), value),
                     f"{section}.hardware.{key} differs from the recount of unique and inherited rows ({value})")


def _validate_postprocess_record(payload: dict) -> None:
    """Shape of ``postprocess``. The git checks are in ``_validate_postprocess_provenance``."""
    post = payload["postprocess"]
    _require(isinstance(post, dict) and set(post) == set(POSTPROCESS_KEYS), "postprocess keys differ from the schema")
    _require(post["source_schema_version"] == SOURCE_SCHEMA_VERSION,
             f"postprocess.source_schema_version must be {SOURCE_SCHEMA_VERSION}")
    _require(all(_is_hex(post[k], 40) for k in ("source_result_git_sha", "postprocess_git_sha")),
             "postprocess commits must be 40-hex")
    _require(all(_is_hex(post[k], 64) for k in ("source_artifact_sha256", "source_raw_measurements_sha256",
                                                "postprocess_code_sha256")), "postprocess digests must be 64-hex")
    _require(type(post["postprocess_git_dirty"]) is bool, "postprocess.postprocess_git_dirty must be a bool")
    modules = post["postprocess_code_modules"]
    _require(isinstance(modules, list) and all(isinstance(m, str) for m in modules),
             "postprocess.postprocess_code_modules must be a list of modules")
    evidence = post["retained_evidence_sha256"]
    _require(isinstance(evidence, dict) and set(evidence) == set(RETAINED_EVIDENCE_SHA256)
             and all(_is_hex(v, 64) for v in evidence.values()),
             "postprocess.retained_evidence_sha256 must hold the 64-hex digests of the four retained audit files")
    _require(post["reason"] == POSTPROCESS_REASON, "postprocess.reason differs from the declared reason")
    _timestamp(post["postprocess_timestamp"], "postprocess.postprocess_timestamp")


def _validate_postprocess_provenance(payload: dict, *, canonical: bool) -> None:
    """Source and code provenance of a postprocessed result, against the local git repository."""
    post, meta = payload["postprocess"], payload["meta"]
    source_sha, post_sha = post["source_result_git_sha"], post["postprocess_git_sha"]
    _require(_is_hex(meta["git_sha"], 40), "a postprocessed result needs the 40-hex git_sha of its producer")
    for label, sha in (("postprocess.source_result_git_sha", source_sha),
                       ("postprocess.postprocess_git_sha", post_sha), ("meta.git_sha", meta["git_sha"])):
        _require_git_commit(sha, label)
    blob = _git_blob(source_sha, RESULT_REPO_PATH)
    _require(hashlib.sha256(blob).hexdigest() == post["source_artifact_sha256"],
             "postprocess.source_artifact_sha256 differs from the committed bytes of the source artifact")
    source = json.loads(blob)
    _require(isinstance(source, dict) and source.get("schema_version") == post["source_schema_version"]
             and source.get("task") == TASK, "the source artifact has another schema version or task")
    for key in RAW_MEASUREMENT_KEYS:  # the raw measurements are the source's, value for value and type for type
        _require(key in source and _same(payload[key], source[key]),
                 f"{key} differs from the source artifact: a postprocess must not change a raw measurement")
    _require(post["source_raw_measurements_sha256"] == raw_measurements_sha256(payload),
             "postprocess.source_raw_measurements_sha256 differs from the raw measurements")
    # Producer code: the import closure, blobs and schema declaration of the commit that measured.
    _require(meta["code_modules"] == list(_code_modules_at_commit(meta["git_sha"])),
             "meta.code_modules differ from the static import closure at meta.git_sha")
    _require(meta["code_sha256"] == _code_sha256_at_commit(meta["git_sha"], tuple(meta["code_modules"])),
             "meta.code_sha256 differs from the git blobs at meta.git_sha")
    producer_schema, _ = _driver_capabilities_at_commit(meta["git_sha"])
    _require(producer_schema == post["source_schema_version"],
             f"meta.git_sha declares schema {producer_schema!r}, not {post['source_schema_version']!r}")
    # Postprocessing code: this code, as committed at postprocess_git_sha.
    _require(post["postprocess_code_modules"] == list(code_modules())
             and post["postprocess_code_modules"] == list(_code_modules_at_commit(post_sha)),
             "postprocess.postprocess_code_modules differ from the static import closure")
    _require(post["postprocess_code_sha256"] == _code_sha256_at_commit(post_sha, tuple(post["postprocess_code_modules"])),
             "postprocess.postprocess_code_sha256 differs from the git blobs at postprocess_git_sha")
    post_schema, has_post_cli = _driver_capabilities_at_commit(post_sha)
    _require(post_schema == SCHEMA_VERSION and has_post_cli,
             f"postprocess.postprocess_git_sha must commit {SCHEMA_VERSION} with --postprocess-from-git")
    _require(post_sha != meta["git_sha"] and post_sha != source_sha,
             "postprocess.postprocess_git_sha must differ from the producer and source commits")
    parsed = _parsed_command(post["postprocess_command"], "postprocess.postprocess_command")
    _require(parsed.postprocess_from_git == source_sha,
             "postprocess.postprocess_command must pass --postprocess-from-git with the source result commit")
    if canonical:
        _require(reserved(parsed.output) and Path(parsed.output).name == CANONICAL_OUTPUT.name,
                 "postprocess.postprocess_command of a canonical result must target the canonical output path")
    _require(_timestamp(post["postprocess_timestamp"], "postprocess.postprocess_timestamp")
             >= _timestamp(meta["timestamp_end"], "meta.timestamp_end"),
             "postprocess.postprocess_timestamp is before the end of the measuring run")
    if canonical:
        _require(post["postprocess_git_dirty"] is False, "a canonical postprocess needs a clean tree")
        _require(CANONICAL_SOURCE_ARTIFACTS.get(source_sha) == post["source_artifact_sha256"],
                 "the source of a canonical postprocess must be the pinned committed C5-02 result")
        _require(_same(post["retained_evidence_sha256"], RETAINED_EVIDENCE_SHA256),
                 "postprocess.retained_evidence_sha256 differs from the pinned digests of the retained audit files")


def validate_results(payload: dict, *, canonical: bool | None = None) -> dict:
    """Offline consistency, provenance and promotion-gate validator. Returns what it certified.

    Call-site certification. ``payload["canonical"]`` selects the checks. A caller that passes
    ``canonical`` states what it expects: ``True`` to promote a result, ``False`` for an exploratory
    one. A disagreement with the payload raises, so a payload labeled canonical is never validated
    with relaxed checks. The return value is ``{"mode": "canonical" | "noncanonical",
    "certified_canonical": bool, "postprocessed": bool}``.

    What it establishes from the JSON and the local git repository alone:

    * schema, strict JSON (no NaN / infinity, no NUL byte in any string), JSON integers and booleans
      with their exact types, and the exact protocol (canonical: the locked C5-02 protocol, the
      FineWeb-Edu GPT-2 shards with pinned hashes and token counts, sm_120+ device, clean tree);
    * provenance of a measuring run (``postprocess`` is None): whenever ``meta.git_sha`` is non-None it is
      checked as a git commit object (``git cat-file -t``), ``run_config_hash`` / ``unit_config_hash`` are the
      recomputed digests, and ``meta.code_modules`` matches this driver's import closure. In canonical mode,
      ``meta.git_dirty`` must be False, ``meta.code_sha256`` must equal the transitive ``research.*`` blobs at
      ``meta.git_sha``, and the committed driver at ``meta.git_sha`` must declare ``SCHEMA_VERSION`` (exploratory
      runs may start from dirty working trees and are not checked against committed blobs);
    * provenance of a postprocessed result: the source artifact is the committed file of
      ``source_result_git_sha`` (byte digest), every ``RAW_MEASUREMENT_KEYS`` value equals the source,
      the producer code is the import closure and the blobs at ``meta.git_sha`` (hashes under the
      source schema), and the postprocessing code is this code at ``postprocess_git_sha``;
    * split access as recorded by ``Shards.open``: Stage A on a train + val mapping, test.bin mapped
      afterwards for the Stage B scoring, no test field in any Stage A record. This is not a record of
      every file read: the startup verification reads all three shard files and scores no model;
    * Stage A: the locked recipes of every arm (optimizer, LR, cooldown record), finite nonnegative
      losses, and the AdamW search of every seed replayed with ``adamw_eval_steps`` /
      ``adamw_stops`` / ``select_adamw`` from the stored val NLLs;
    * Stage B: every evaluated checkpoint has the SHA-256 and the val NLL of its Stage A record; the
      seeded calibration draw and the complete val / test coverage are recomputed from the seed and
      the split lengths; complete grids; per-row arithmetic; fit records, damping and static-scale
      domains; alpha = val.bin minimum, and alpha = 0 rows equal to their unscaled twins; 60/60 FP4
      GEMMs and int32 decode checks as recorded; per-layer error identity;
    * durations are finite and nonnegative, duplicated stage times agree, and the resume markers
      add up to ``meta.unit_log``;
    * ``analysis``, ``derived`` and ``caveats`` equal their recomputation from the stored rows (1e-12),
      and the unique / inherited hardware counts equal an independent recount;
    * canonical gate: hardware-versus-FP32-emulation parity within 1e-3 nats.

    Threat model. This is an offline consistency check, not an attestation. It cannot prove that a
    leaf measurement (an NLL, a checkpoint hash, a GEMM count, a timing) was produced by the
    recorded code on the recorded GPU: a party that rewrites leaves and recomputes every dependent
    summary passes. The saved checkpoints are not in git, so their bytes are not checked here
    (``--resume`` checks their tensor digests before it reuses a unit). The retained audit files are
    not in git either: their digests are pinned, their bytes are not read. Independent evidence
    needs a live replay on an RTX 5080 from the retained checkpoints or from training (training is
    not bit-reproducible on this host). Scientific outcomes (matched seeds, significance, sign of
    an effect) are reported, never required.
    """
    try:
        return _validate_payload(payload, canonical)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:  # a record of the wrong shape or type
        raise ValueError(f"malformed payload: {type(exc).__name__}: {exc}") from exc


def _validate_payload(payload: dict, canonical: bool | None) -> dict:
    json.dumps(payload, allow_nan=False)
    top = {"schema_version", "task", "quick", "canonical", "meta", "run_config_hash", "preset", "config",
           "protocol", "data", "stage_a", "stage_b", "postprocess", "analysis", "derived", "caveats"}
    _require(isinstance(payload, dict) and set(payload) == top, "top-level keys differ from the schema")
    _require(not _has_nul(payload), "the payload must not contain a NUL byte in any string")
    _require(payload["schema_version"] == SCHEMA_VERSION and payload["task"] == TASK, "schema or task differs")
    _require(type(payload["quick"]) is bool and type(payload["canonical"]) is bool,
             "quick / canonical flags must be bool")
    effective = payload["canonical"]
    _require(canonical is None or type(canonical) is bool, "the requested canonical mode must be a bool or None")
    if canonical is False and effective:
        raise ValueError("the payload is labeled canonical: it cannot be validated with the noncanonical checks")
    if canonical is True and not effective:
        raise ValueError("canonical certification was requested, but the payload is not labeled canonical")
    post = payload["postprocess"]
    if post is not None:
        _validate_postprocess_record(payload)
    schema = SCHEMA_VERSION if post is None else post["source_schema_version"]
    protocol, config, data, meta = payload["protocol"], payload["config"], payload["data"], payload["meta"]
    _validate_protocol(protocol, canonical=effective)
    if effective:
        _require(payload["quick"] is False, "a canonical result cannot be --quick")
    _require(isinstance(payload["preset"], str) and payload["preset"] == protocol["preset"]
             and payload["preset"] in h.PRESETS and _same(config, asdict(h.PRESETS[protocol["preset"]])),
             "model config differs from the preset")
    seeds = protocol["seeds"]
    _validate_data(data, canonical=effective)
    _validate_meta(payload, canonical=effective, schema=schema)
    _require(payload["run_config_hash"] == run_config_hash(
        payload["quick"], protocol, config, meta["code_sha256"], meta["git_sha"], meta["environment"],
        {k: v["sha256"] for k, v in data["splits"].items()}, schema),
        "run_config_hash differs from the recomputed digest")

    final = _validate_stage_a(payload, canonical=effective)
    _validate_stage_b(payload, final, canonical=effective)

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
    _validate_counts(payload)
    if effective:
        _require(payload["analysis"] is not None, "a canonical result needs the paired analysis")
        _require(payload["analysis"]["hardware_vs_emulation"]["passed"] is True
                 and len(payload["analysis"]["hardware_vs_emulation"]["rows"]) == len(seeds) * len(ARMS),
                 "hardware W4A4 differs from its FP32 emulation by more than the parity threshold")
    if post is not None:
        _validate_postprocess_provenance(payload, canonical=effective)
    return {"mode": "canonical" if effective else "noncanonical", "certified_canonical": bool(effective),
            "postprocessed": post is not None}


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
    # With verification this reads train.bin, val.bin and test.bin twice each (hashes, then token range and
    # document boundaries). No model is scored. ``shards.opened`` records the later ``open`` calls only.
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
                        "expected_sha256": None if args.no_verify else dict(q.EXPECTED_SHA256),
                        "roles": dict(ROLES)}}

    # Stage A: every seed, on a mapping of train.bin and val.bin only. ``Shards.open`` does not map test.bin
    # before this loop ends, so no Stage A code can score or select on it. (The startup verification above
    # has already read the test.bin file; the pre-commit seed-0 shakedowns scored it. See the module docstring.)
    began = time.perf_counter()
    view_a = shards.open("train", "val")
    stage_a_units = []
    for seed in protocol["seeds"]:
        # A cached unit is reused only when every retained checkpoint still has its recorded tensor digest.
        unit = units.get(("stage_a", seed), lambda value: checkpoints_verified(value, checkpoint_dir))
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

    # Stage B: the saved checkpoints on val.bin and test.bin. This is the first ``Shards.open`` of test.bin
    # and the first time that this run scores a model on it.
    began = time.perf_counter()
    view_b = {**view_a, **shards.open("test")}
    calibration = q.calibration_windows(SimpleNamespace(splits=view_a), protocol["calibration_sequences"],
                                        config.sequence_length, protocol["calibration_seed"])
    checkpoints = []
    for unit in stage_a_units:
        for arm in ARMS:
            seed, entry = unit["seed"], unit["checkpoints"][arm]
            # A cached Stage B unit is trusted only while the retained checkpoint file has the recorded digest.
            # Otherwise the unit is computed again, and ``load_checkpoint`` raises for an unusable file.
            record = units.get(("stage_b", seed, arm),
                               lambda value: stage_b_unit_verified(value, entry, checkpoint_dir))
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
    payload["postprocess"] = None  # this run measured; ``postprocess_from_git`` fills it for a v1 source
    payload = _with_summaries(json.loads(json.dumps(payload, allow_nan=False)))
    try:
        validate_results(payload, canonical=args.canonical)
    except Exception:
        # Keep the measurements of a long run: the rejected payload goes next to the output, never to it.
        c4.write_json_atomic(rejected_path(args.output), payload)
        raise
    return payload


def _with_summaries(payload: dict) -> dict:
    """``analysis``, ``derived`` and ``caveats`` of the raw measurements, as plain JSON types."""
    protocol = payload["protocol"]
    payload["analysis"] = (analyze(payload["stage_a"], payload["stage_b"], protocol)
                           if len(protocol["seeds"]) == N_SEEDS else None)
    payload["derived"] = derive(payload)
    payload["caveats"] = caveats(payload)
    return json.loads(json.dumps(payload, allow_nan=False))


def rejected_path(output: Path) -> Path:
    return output.with_name(output.name + ".rejected")


def postprocess_from_git(commit: str, command: list[str]) -> dict:
    """The v2 payload of the committed v1 result of ``commit``. Deterministic apart from its timestamp.

    The raw measurements (``RAW_MEASUREMENT_KEYS``) are the parsed bytes of ``git show`` and are not
    changed. Only ``analysis``, ``derived`` and ``caveats`` are recomputed. ``postprocess`` records the
    source, the postprocessing code and the digests of the retained audit files.
    """
    _require_git_commit(commit, "--postprocess-from-git")
    blob = _git_blob(commit, RESULT_REPO_PATH)
    source = json.loads(blob)
    if not isinstance(source, dict) or source.get("schema_version") != SOURCE_SCHEMA_VERSION \
            or source.get("task") != TASK or any(key not in source for key in RAW_MEASUREMENT_KEYS):
        raise ValueError(f"{RESULT_REPO_PATH} at {commit} is not a {SOURCE_SCHEMA_VERSION} result of this task")
    git = h.git_info()
    payload = {"schema_version": SCHEMA_VERSION, "task": TASK, **{key: source[key] for key in RAW_MEASUREMENT_KEYS}}
    payload["postprocess"] = {
        "source_schema_version": source["schema_version"], "source_result_git_sha": commit,
        "source_artifact_sha256": hashlib.sha256(blob).hexdigest(),
        "source_raw_measurements_sha256": raw_measurements_sha256(source),
        "postprocess_git_sha": git["git_sha"], "postprocess_git_dirty": git["git_dirty"],
        "postprocess_code_sha256": code_sha256(), "postprocess_code_modules": list(code_modules()),
        "postprocess_command": list(command), "postprocess_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "retained_evidence_sha256": dict(RETAINED_EVIDENCE_SHA256), "reason": POSTPROCESS_REASON}
    return _with_summaries(payload)


def postprocess(args) -> dict:
    """``--postprocess-from-git``: no GPU, no training. A reserved output needs a clean tree and canonical source."""
    require_clean_tree(h.git_info(), args.output)  # before any work
    payload = postprocess_from_git(args.postprocess_from_git, [sys.executable, *sys.argv])
    want_canonical = True if reserved(args.output) else payload["canonical"]
    cert = validate_results(payload, canonical=want_canonical)
    if reserved(args.output) and not cert["certified_canonical"]:
        raise ValueError(f"Refusing to write noncanonical postprocessed result to reserved {args.output.name}")
    return payload


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
    parser.add_argument("--postprocess-from-git", metavar="COMMIT", default=None,
                        help=f"no GPU work: read the committed {SOURCE_SCHEMA_VERSION} result of COMMIT (full 40-hex "
                             "SHA) with git show, keep its raw measurements unchanged and recompute analysis, "
                             "derived and caveats")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.postprocess_from_git is not None:
        measuring = [flag for flag, used in (
            ("--quick", args.quick), ("--seeds", args.seeds is not None), ("--device", args.device is not None),
            ("--max-eval-windows", args.max_eval_windows is not None), ("--data-dir", args.data_dir is not None),
            ("--no-verify", args.no_verify), ("--checkpoint-dir", args.checkpoint_dir is not None),
            ("--resume", args.resume)) if used]
        if measuring:
            parser.error(f"--postprocess-from-git measures nothing: remove {' '.join(measuring)}")
        if not _is_hex(args.postprocess_from_git, 40):
            parser.error("--postprocess-from-git needs the full 40-hex SHA of the result commit")
        args.output = CANONICAL_OUTPUT if args.output is None else args.output
        if reserved(args.output) and (args.output.is_symlink()
                                      or args.output.resolve() != CANONICAL_OUTPUT.resolve()):
            parser.error(f"A postprocess writes {CANONICAL_OUTPUT} of this checkout (no symlink, no other "
                         f"checkout) or a path outside {RESERVED_PATTERN}")
        args.protocol, args.canonical = None, False  # not a measuring run: it selects no protocol
        return args
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
    if args.postprocess_from_git is not None:
        payload = c4.write_json_atomic(args.output, postprocess(args))
    else:
        payload = c4.write_json_atomic(args.output, run(args))
        unit_log_path(args.output).unlink(missing_ok=True)
    if not args.silent:
        print(json.dumps(payload["derived"].get("iso_loss", payload["derived"]["stage_a"]), indent=1))
        print(f"Wrote {args.output}")
    return payload


if __name__ == "__main__":
    main()
