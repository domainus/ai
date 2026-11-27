"""Cycle 4 C4-03: wide-model training transfer, 5-seed common-checkpoint replication, K = 2^25 Gram drift.

The driver wraps ``harness_tier2_real`` and the Blackwell kernels without changing them.
Stages (each one writes one strict JSON file, ``allow_nan=False``):

* ``wide``: compiled end-to-end Muon training at widths 768 (anchor), 1536 and 2048, 16,384
  tokens per step, 64 steps, seeds 1-3, four orthogonalizers. Each row keeps every per-step
  CUDA-event segment (fwd+bwd, polar, layout cast, foreach update) and the peak VRAM.
  -> ``results/cycle4_c4_03_wide.json``
* ``ckpt``: ``PRESETS["120m"]``, seeds 1-5, one fixed-token cosine schedule over 384 steps
  (12,582,912 tokens, < 0.5 epoch). ``val.bin`` and ``test.bin`` are scored at the predeclared
  steps 128, 256 and 384 of the *same* trajectory, so a wall-clock LR schedule cannot confound
  the comparison of ``ss3_54`` / ``ss3_53`` with ``ps3_blackwell``.
  -> ``results/cycle4_c4_03_ckpt.json``
* ``drift``: root cause of the ``sigma_max`` drift at very long ``K`` (Opus L6). It measures the
  Gram matrix of the FP16/BF16 operand with several accumulations against exact FP64, compares
  the Triton SYRK with an exact integer model of the Tensor-Core FP32 accumulate, sweeps
  ``K = 2^10 .. 2^25``, and tests a prototype mitigation (chunked-K SYRK, see ``ChunkedGram``).
  The production kernels stay untouched. -> ``results/cycle4_c4_03_drift.json``
* ``summary``: predeclared paired statistics of ``wide`` (df = 2) and ``ckpt`` (df = 4) and the
  headline numbers of ``drift``. -> ``results/cycle4_c4_03_summary.json``

Statistics: for one seed every optimizer starts from the same weights and sees the same batches.
A contrast is a paired Student-t test over seeds with the exact closed forms for df = 2 and
df = 4 (``paired_t``). Holm-Bonferroni adjusts p within the families named in ``analyze_train``.

The filenames ``cycle4_c4_03_*.json`` are reserved, in any checkout and through symlinks: only
the canonical protocol of that stage, on an sm_120+ CUDA device and a clean git tree, may write
them. Both guards run before any data is read. ``validate_results`` recomputes the
``run_config_hash`` and every derived number.

Run (alone on the GPU, from a clean commit):
    PYTHONPATH=. .venv/bin/python research/polar_opt/bench_c4_03_wide_and_gram_drift.py --stage wide
    PYTHONPATH=. .venv/bin/python research/polar_opt/bench_c4_03_wide_and_gram_drift.py --stage ckpt
    PYTHONPATH=. .venv/bin/python research/polar_opt/bench_c4_03_wide_and_gram_drift.py --stage drift
    PYTHONPATH=. .venv/bin/python research/polar_opt/bench_c4_03_wide_and_gram_drift.py --stage summary
Smoke test (tiny models; on CPU the Muon kernels fall back to eager Polar Express):
    PYTHONPATH=. .venv/bin/python research/polar_opt/bench_c4_03_wide_and_gram_drift.py --stage wide \\
        --quick --output /tmp/c4_03_wide.json
"""

from __future__ import annotations

import argparse
import ast
import copy
import fnmatch
import functools
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from statistics import median

import torch
from torch import Tensor

from research.polar_opt import bench_c4_01a_120m_long as c41
from research.polar_opt import harness_tier2_real as h
from research.quant_ptq import micro_rotated_ptq as q
from research.polar_opt.bench_c4_01a_120m_long import (  # shared, already audited helpers
    StepClock, _assert_close, _is_hex, _relative, _require, default_device, dumps, eval_tokens_used,
    holm_bonferroni, load_rows, mean_sd, resume_log_path, set_numerics, sha256_file, write_json_atomic)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SCHEMA_VERSION = "cycle4_c4_03.v1"
TASK = ("Cycle 4 C4-03: wide-model training transfer (w >= 1536), 5-seed common-checkpoint SS3_54/SS3_53 vs "
        "PS3_55 replication and K = 2^25 FP16 Gram drift root cause on RTX 5080 sm_120")
RUN_STAGES = ("wide", "ckpt", "drift")
TRAIN_STAGES = ("wide", "ckpt")
STAGES = (*RUN_STAGES, "summary")
CANONICAL_OUTPUTS = {stage: RESULTS / f"cycle4_c4_03_{stage}.json" for stage in STAGES}
QUICK_OUTPUTS = {stage: RESULTS / f"c4_03_quick_{stage}.json" for stage in STAGES}
RESERVED_PATTERN = "cycle4_c4_03_*.json"
REQUIRED_CODE = ("bench_c4_03_wide_and_gram_drift", "bench_c4_01a_120m_long", "harness_tier2_real",
                 "harness_tier2_lm", "real_text_data", "tier2_candidates", "baselines", "circuits",
                 "kernels_blackwell", "polar_switch_blackwell", "schatten_switch_blackwell", "micro_rotated_ptq")

MUON = ("muon_pe_compiled", "muon_ps3_blackwell", "muon_ss3_54", "muon_ss3_53")
CUDA_ORTHOGONALIZERS = {name: c41.CUDA_ORTHOGONALIZERS[name] for name in MUON}
# Predeclared contrasts per stage: (reference, candidates). The candidate is the numerator.
CONTRASTS = {"wide": (("muon_pe_compiled", ("muon_ss3_54", "muon_ss3_53", "muon_ps3_blackwell")),
                      ("muon_ps3_blackwell", ("muon_ss3_54", "muon_ss3_53"))),
             "ckpt": (("muon_ps3_blackwell", ("muon_ss3_54", "muon_ss3_53")),)}
RUN_METRICS = (("tokens_per_second", "ratio"), ("polar_ortho_ms", "ratio"), ("polar_fraction", "diff"))
LOSS_METRICS = (("val_loss", "diff"), ("test_loss", "diff"))  # at every predeclared checkpoint
TIMING_KEYS = ("step_wall_ms", "fwd_bwd_ms", "polar_ortho_ms", "layout_cast_ms", "foreach_update_ms",
               "momentum_ms", "auxiliary_adamw_ms", "optimizer_step_ms")
SEGMENTS = ("fwd_bwd", "momentum", "polar_ortho", "layout_cast", "foreach_update", "auxiliary_adamw")

# 16,384 tokens per step at every width, so the polar fraction scales with the width alone.
WIDE_CONFIGS = {
    "w768": h.ModelConfig(width=768, layers=12, heads=12, hidden=2048, sequence_length=512, micro_batch=8,
                          grad_accum=4),
    "w1536": h.ModelConfig(width=1536, layers=6, heads=16, hidden=4096, sequence_length=512, micro_batch=8,
                           grad_accum=4),
    "w2048": h.ModelConfig(width=2048, layers=4, heads=16, hidden=5440, sequence_length=512, micro_batch=8,
                           grad_accum=4),
}
_QUICK_WIDE = {
    "w64": h.ModelConfig(width=64, layers=2, heads=4, hidden=128, sequence_length=64, micro_batch=4, grad_accum=2),
    "w96": h.ModelConfig(width=96, layers=1, heads=4, hidden=256, sequence_length=64, micro_batch=4, grad_accum=2),
}
CKPT_STEPS = (128, 256, 384)  # predeclared common checkpoints of the 384-step schedule

DESIGN_CEILING = 1.08  # Opus L6: the design target of sigma_max is 1 + error = 1.08
PIN_CONFIG = {"BLOCK": 32, "BK": 64, "num_warps": 4, "num_stages": 4}
OPUS_L6 = {  # audits/AUDIT_cycle3_c3_03_claude_opus.md, finding L6 and evidence section 3
    "wide4:ss3_54_fp16": {"shape": [4, 1 << 25], "sigma_max": [1.129, 1.135]},
    "wide4:ps3_55_fp16": {"shape": [4, 1 << 25], "sigma_max": [1.179, 1.179]},
    "tall2:ss3_54_fp16": {"shape": [1 << 25, 2], "sigma_max": [1.150, 1.151]},
    "wide1:ss3_54_fp16": {"shape": [1, 1 << 25], "sigma_max": [1.043, 1.043]},
}
_TRAIN = {"device": "cuda", "muon_lr": 0.02, "muon_aux_lr": 0.001, "weight_decay": 0.01, "update_dtype": "fp32",
          "compile_model": True, "compile_warmup_steps": 3, "min_lr_ratio": 0.1, "no_verify": False}
CANONICAL = {
    "wide": {**_TRAIN, "configs": {k: asdict(v) for k, v in WIDE_CONFIGS.items()}, "optimizers": list(MUON),
             "seeds": [1, 2, 3], "steps": 64, "checkpoints": [64], "lr_warmup_steps": 8, "eval_tokens": 262_144},
    "ckpt": {**_TRAIN, "configs": {"120m": asdict(h.PRESETS["120m"])},
             "optimizers": ["muon_ps3_blackwell", "muon_ss3_54", "muon_ss3_53"], "seeds": [1, 2, 3, 4, 5],
             "steps": CKPT_STEPS[-1], "checkpoints": list(CKPT_STEPS), "lr_warmup_steps": 50,
             "eval_tokens": 2_000_000},
    "drift": {
        "device": "cuda", "ceiling": DESIGN_CEILING, "kernels": ["ps3_55", "ss3_54", "ss3_53"],
        "dtypes": ["fp16", "bf16"],
        # n is the short side. "tall" inputs have shape (K, n). Every integer exponent is swept.
        "families": {
            "wide4": {"n": 4, "orientation": "wide", "k_exponents": list(range(10, 26)), "seeds": [0, 1, 2]},
            "tall2": {"n": 2, "orientation": "tall", "k_exponents": list(range(10, 26)), "seeds": [0, 1, 2]},
            "wide1": {"n": 1, "orientation": "wide", "k_exponents": list(range(20, 26)), "seeds": [0]},
            "wide16": {"n": 16, "orientation": "wide", "k_exponents": list(range(18, 25)), "seeds": [0]},
        },
        "gram_seeds": [0], "gram_chunks": [1024, 4096, 16384, 65536], "prescale_exponent": 12,
        "chunk_k": 4096, "gate_k": 65536, "pin_config": PIN_CONFIG, "latency_reps": 3,
        "model_k": [64, 256, 1024, 4096, 16384], "model_rows": 3, "model_seed": 0,
        "config_shapes": [[4, 262144], [262144, 2], [100, 16384]],
        "svd_check": [[4, 262144], [262144, 2]],
        "normal_shapes": [[512, 2048], [1024, 4096], [2048, 8192], [4096, 4096]], "normal_reps": 20,
    },
}
_QUICK_TRAIN = {"compile_model": False, "compile_warmup_steps": 1, "lr_warmup_steps": 2, "eval_tokens": 2048,
                "steps": 6}
QUICK = {
    "wide": {**CANONICAL["wide"], **_QUICK_TRAIN, "configs": {k: asdict(v) for k, v in _QUICK_WIDE.items()},
             "checkpoints": [6]},
    "ckpt": {**CANONICAL["ckpt"], **_QUICK_TRAIN, "configs": {"tiny": asdict(h.PRESETS["tiny"])},
             "checkpoints": [2, 4, 6]},
    "drift": {**CANONICAL["drift"],
              "families": {"wide4": {"n": 4, "orientation": "wide", "k_exponents": [10, 12, 14], "seeds": [0]},
                           "tall2": {"n": 2, "orientation": "tall", "k_exponents": [10, 12], "seeds": [0]}},
              "gram_chunks": [256, 1024], "chunk_k": 256, "gate_k": 1024, "latency_reps": 1,
              "model_k": [64, 256], "config_shapes": [[4, 4096]], "svd_check": [[4, 4096]],
              "normal_shapes": [[64, 256]], "normal_reps": 2},
}


# ======================================================================================
# Guards, provenance and hashes
# ======================================================================================
def reserved(path: Path) -> bool:
    """True for a ``cycle4_c4_03_*.json`` name, in any checkout or through a symlink."""
    path = Path(path)
    return any(fnmatch.fnmatchcase(name, RESERVED_PATTERN) for name in (path.name, path.resolve().name))


def require_blackwell(device: torch.device, output: Path | None, *, canonical: bool = False) -> None:
    """A reserved filename (or the canonical protocol) promises an RTX 5080-class (sm_120+) result."""
    if (canonical or (output is not None and reserved(output))) and (
            device.type != "cuda" or not torch.cuda.is_available()
            or tuple(torch.cuda.get_device_capability(device)) < (12, 0)):
        raise RuntimeError(f"{Path(output).name if output else 'A canonical run'} requires a Blackwell "
                           "(sm_120+) CUDA device")


def _check_git(git: dict, canonical: bool, output: Path) -> None:
    if (canonical or reserved(output)) and git.get("git_dirty") is not False:
        raise RuntimeError("A canonical run needs a clean, committed git tree (git_dirty must be False)")


def _module_path(name: str) -> Path:
    return ROOT.joinpath(*name.split(".")).with_suffix(".py")


@functools.cache
def code_modules() -> tuple[str, ...]:
    """Every ``research.*`` module that this driver imports, directly or through another module.

    The closure is static (``ast`` over each source file, imports inside functions included),
    so it is the same in every process: driver, harness, data loader, candidate adapters,
    baselines, circuits and Blackwell kernels.
    """
    found, todo = set(), ["research.polar_opt.bench_c4_03_wide_and_gram_drift"]
    while todo:
        name = todo.pop()
        if name in found or not _module_path(name).is_file():
            continue  # a package, an attribute of a module, or not part of this repository
        found.add(name)
        for node in ast.walk(ast.parse(_module_path(name).read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("research"):
                todo += [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
            elif isinstance(node, ast.Import):
                todo += [alias.name for alias in node.names if alias.name.startswith("research")]
    return tuple(sorted(found))


def code_sha256() -> str:
    """SHA-256 over the source of every module in ``code_modules`` (sorted by module name)."""
    digest = hashlib.sha256()
    for name in code_modules():
        digest.update(name.encode() + b"\0" + _module_path(name).read_bytes())
    return digest.hexdigest()


def _env_key(env: dict) -> dict:
    return {k: env.get(k) for k in ("torch", "triton", "cuda", "device", "capability")}


def _hash_payload(stage: str, quick: bool, protocol: dict, data_sha256: dict | None, code: str,
                  git_sha: str | None, env: dict) -> str:
    payload = {"schema": SCHEMA_VERSION, "stage": stage, "quick": quick, "protocol": protocol,
               "data_sha256": data_sha256, "code_sha256": code, "git_sha": git_sha, "env": _env_key(env)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def expected_run_config_hash(payload: dict) -> str:
    """``run_config_hash`` recomputed from the payload's own protocol, shards, code and environment."""
    data = payload.get("data")
    shards = {k: v["sha256"] for k, v in sorted(data["splits"].items())} if data else None
    return _hash_payload(payload["stage"], payload["quick"], payload["protocol"], shards,
                         payload["meta"]["code_sha256"], payload["meta"]["git_sha"], payload["meta"]["environment"])


# ======================================================================================
# Statistics (pure functions; ``validate_results`` recomputes them)
# ======================================================================================
# df = 4: with x = t / sqrt(t^2 + 4) the CDF is 1/2 + (3 x - x^3) / 4. The 0.975 quantile solves
# x^3 - 3 x + 1.9 = 0 on (0, 1): x = 2 cos((acos(-0.95) + 4 pi) / 3), and t = 2 x / sqrt(1 - x^2).
_X975 = 2.0 * math.cos((math.acos(-0.95) + 4.0 * math.pi) / 3.0)
T975_DF4 = 2.0 * _X975 / math.sqrt(1.0 - _X975 * _X975)
T975 = {2: c41.T975_DF2, 4: T975_DF4}


def paired_t(values: list[float], null: float = 0.0) -> dict:
    """Exact paired t test of 3 (df = 2) or 5 (df = 4) per-seed values against ``null``.

    df = 4: the two-sided p is 1 - (3 x - x^3) / 2 with x = |t| / sqrt(t^2 + 4). It is computed
    as c^2 (3 - c) / 2 with c = 1 - x = 4 / (s (s + |t|)), s = sqrt(t^2 + 4), so a large |t| does
    not cancel. With sd = 0, p is 1 (mean == null) or 0.
    """
    n = len(values)
    if n == 3:
        return c41.paired_t_df2(values, null)
    if n != 5:
        raise ValueError("The exact formulas need three (df = 2) or five (df = 4) paired values")
    stats = mean_sd(values)
    mean, sd = stats["mean"], stats["sd"]
    se = sd / math.sqrt(n)
    shifted = mean_sd([v - null for v in values])  # t from v - null: no cancellation for ratios near 1
    if sd > 0:
        t = shifted["mean"] / (shifted["sd"] / math.sqrt(n))
        s = math.sqrt(t * t + 4.0)
        c = 4.0 / (s * (s + abs(t)))
        p = c * c * (3.0 - c) / 2.0
    else:
        t, p = None, (1.0 if shifted["mean"] == 0 else 0.0)
    return {**stats, "n": n, "df": 4, "null": null, "se": se, "t": t,
            "ci95": [mean - T975_DF4 * se, mean + T975_DF4 * se], "p_two_sided": p}


# ======================================================================================
# Training stages (wide, ckpt)
# ======================================================================================
def factories(device: torch.device) -> tuple[dict, dict]:
    """Name -> zero-argument orthogonalizer factory, and a description of each (``c41.factories``)."""
    found, described = c41.factories(device)
    return {name: found[name] for name in MUON}, {name: described[name] for name in MUON}


def muon_groups(config: h.ModelConfig) -> list[dict]:
    """The same-shape groups that ``harness_tier2_real.Muon`` stacks, in its order."""
    with torch.device("meta"):
        model = h.Transformer(config)
    groups: dict = {}
    for parameter in model.matrix_parameters().values():
        groups[tuple(parameter.shape)] = groups.get(tuple(parameter.shape), 0) + 1
    return [{"shape": list(shape), "count": count, "stacked": [count, *shape], "short_side": min(shape),
             "aspect": max(shape) / min(shape)} for shape, count in groups.items()]


def model_record(config: h.ModelConfig) -> dict:
    return {"config": asdict(config), "step_tokens": config.step_tokens,
            "parameters": h.parameter_counts(config), "muon_groups": muon_groups(config)}


def timing_from_raw(row: dict) -> dict:
    """``harness_tier2_real.summarize_timing`` of the stored per-step samples (first step dropped)."""
    wall, segments = row["step_wall_seconds"], row["step_ms"]
    samples = [{**{k: v[i] for k, v in segments.items()}, "step_wall_ms": wall[i] * 1e3} for i in range(len(wall))]
    return h.summarize_timing(samples[1:] if len(samples) > 2 else samples)


def polar_fraction(timing: dict) -> float | None:
    """Median polar orthogonalization time over the median step wall time."""
    polar, wall = timing.get("polar_ortho_ms"), timing.get("step_wall_ms")
    return polar / wall if polar is not None and wall else None


def run_trial(config: h.ModelConfig, data: h.ShardData, *, device: torch.device, seed: int, name: str, factory,
              protocol: dict, init: dict, reusable: tuple, clock=None) -> dict:
    """One Muon trial on the fixed-token schedule, scored at every predeclared checkpoint.

    The loop is the token-budget loop of ``harness_tier2_real.run_trial`` (same warmup, stream,
    schedule and step), plus evaluation inside the trajectory. Evaluation runs under ``no_grad``
    in eval mode and is outside the step clock, so it changes neither the weights nor the timing.
    """
    clock = clock or StepClock()
    total, checkpoints, eval_tokens = protocol["steps"], set(protocol["checkpoints"]), protocol["eval_tokens"]
    settings = dict(lr=protocol["muon_lr"], auxiliary_lr=protocol["muon_aux_lr"],
                    weight_decay=protocol["weight_decay"], update_dtype=protocol["update_dtype"])
    cuda = device.type == "cuda"
    model, forward = reusable
    model.load_state_dict(init)
    model.train()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    if protocol["compile_warmup_steps"]:  # untimed, on a disposable stream: compilation and autotuning
        warm = h.build_optimizer(name, factory, model, **settings)
        if hasattr(getattr(warm, "orthogonalize", None), "set_warmup"):
            warm.orthogonalize.set_warmup(True)
        warm_stream = h.BatchStream(data.train, config, seed + 1_000_000, device)
        for _ in range(protocol["compile_warmup_steps"]):
            h.train_step(forward, warm_stream.next_batch(), warm)
            model.zero_grad(set_to_none=True)
        h._sync(device)
        del warm, warm_stream
        model.load_state_dict(init)
    optimizer = h.build_optimizer(name, factory, model, **settings)
    stream = h.BatchStream(data.train, config, seed, device)
    timer = h.StepTimer(cuda)
    row = {"optimizer": name, "seed": seed, "lr": protocol["muon_lr"], "auxiliary_lr": protocol["muon_aux_lr"],
           "update_dtype": protocol["update_dtype"], "init_fingerprint": h.state_fingerprint(init),
           "stream_fingerprint": h.stream_fingerprint(data.train, config, seed),
           "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "loadavg_start": list(os.getloadavg()),
           "gpu_start": h.gpu_telemetry(device)}
    wall, segments, curve, scored = [], {}, [], []
    elapsed, steps, last_loss, diverged = 0.0, 0, None, False
    while steps < total:
        optimizer.set_lr_multiplier(h.lr_multiplier(steps, steps / total, protocol["lr_warmup_steps"],
                                                    protocol["min_lr_ratio"], total_steps=total))
        batch = stream.next_batch()
        h._sync(device)
        start = clock()
        loss = h.train_step(forward, batch, optimizer, timer)
        h._sync(device)
        step_seconds = clock() - start
        loss = loss.item()
        model.zero_grad(set_to_none=True)
        for label, value in timer.collect().items():
            segments.setdefault(label, []).append(value)
        elapsed += step_seconds
        steps += 1
        wall.append(step_seconds)
        if not math.isfinite(loss):
            diverged = True
            break
        last_loss = loss
        if steps % 10 == 0 or steps == 1:
            curve.append([steps, round(loss, 5)])
        if steps in checkpoints:
            model.eval()
            val = h.evaluate(model, data.splits["val"], config, eval_tokens, device)
            test = h.evaluate(model, data.splits["test"], config, eval_tokens, device)
            model.train()
            if not (math.isfinite(val) and math.isfinite(test)):
                diverged = True
                break
            scored.append({"step": steps, "tokens": steps * config.step_tokens, "train_seconds": elapsed,
                           "val_loss": val, "val_ppl": math.exp(val), "test_loss": test,
                           "test_ppl": math.exp(test)})
    final = scored[-1] if scored and not diverged else {}
    row.update({
        "steps": steps, "tokens": steps * config.step_tokens, "target_steps": total, "train_seconds": elapsed,
        "tokens_per_second": steps * config.step_tokens / elapsed if elapsed else None,
        "last_train_loss": last_loss, "checkpoints": scored, "val_loss": final.get("val_loss"),
        "test_loss": final.get("test_loss"), "diverged": diverged or not final,
        "step_wall_seconds": wall, "step_ms": segments, "train_curve": curve,
        "peak_vram_mib": ({"allocated": torch.cuda.max_memory_allocated(device) / 2 ** 20,
                           "reserved": torch.cuda.max_memory_reserved(device) / 2 ** 20} if cuda else None),
        "gpu_end": h.gpu_telemetry(device), "loadavg_end": list(os.getloadavg()),
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    row["timing"] = timing_from_raw(row)
    row["polar_fraction"] = polar_fraction(row["timing"])
    return row


def train_plan(stage: str, protocol: dict) -> list[tuple[str, int, int, str]]:
    """(config label, seed, order_in_block, optimizer): seeded order per (config, seed) block."""
    plan = []
    for label in protocol["configs"]:
        for seed in protocol["seeds"]:
            block = list(protocol["optimizers"])
            random.Random(f"{seed}-{stage}-{label}").shuffle(block)
            plan += [(label, seed, order, name) for order, name in enumerate(block)]
    return plan


def build_train_protocol(args, described: dict, data: h.ShardData) -> dict:
    protocol = copy.deepcopy(args.protocol)
    splits = data.manifest["splits"]
    configs = {label: h.ModelConfig(**config) for label, config in protocol["configs"].items()}
    protocol.update({
        "eval_tokens_used": {label: {s: eval_tokens_used(splits[s]["tokens"], c, protocol["eval_tokens"])
                                     for s in ("val", "test")} for label, c in configs.items()},
        "tokens": {label: protocol["steps"] * c.step_tokens for label, c in configs.items()},
        "run_order": "per (config, seed): random.Random(f'{seed}-{stage}-{label}').shuffle(optimizers)",
        "schedule": "linear warmup over lr_warmup_steps, then cosine to min_lr_ratio over the fixed steps",
        "pairing": "one CPU initial state and one BatchStream per (config, seed), shared by every optimizer",
        "evaluation": "val.bin and test.bin prefixes in eval mode at each checkpoint of the same trajectory",
        "orthogonalizers": {name: described[name] for name in protocol["optimizers"]}})
    return protocol


def _append(log_path: Path, row: dict) -> None:
    with log_path.open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _start(args) -> tuple:
    """Device, git and logging setup. Both guards run here, before any data is read."""
    device = torch.device(args.protocol["device"])
    require_blackwell(device, args.output, canonical=args.canonical)
    git = h.git_info()
    _check_git(git, args.canonical, args.output)
    log = (lambda *a: None) if args.silent else (lambda *a: print(*a, flush=True))
    return device, git, log, time.perf_counter()


def _meta(args, git: dict, env: dict, numerics: dict, device: torch.device) -> dict:
    return {**git, "command": [sys.executable, *sys.argv], "args": {k: str(v) for k, v in vars(args).items()},
            "environment": env, "numerics": numerics, "code_sha256": code_sha256(),
            "code_modules": list(code_modules()), "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "loadavg_start": list(os.getloadavg()), "gpu_telemetry": {"start": h.gpu_telemetry(device)}}


def _finish(meta: dict, args, loaded: int, rows: list, started: float, device: torch.device) -> None:
    meta.update({"resume": {"resume": args.resume, "rows_loaded": loaded, "rows_run": len(rows) - loaded},
                 "loadavg_end": list(os.getloadavg()), "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "total_seconds": time.perf_counter() - started})
    meta["gpu_telemetry"]["end"] = h.gpu_telemetry(device)


def run_train_stage(args) -> dict:
    """Train every (config, seed, optimizer) of the ``wide`` or ``ckpt`` stage. Returns the payload."""
    stage = args.stage
    device, git, log, started = _start(args)
    numerics = set_numerics(device)
    verify = not args.protocol["no_verify"]
    data = q.load_data(args.data_dir, verify=verify, expected=q.EXPECTED_SHA256 if verify else None)
    found, described = factories(device)
    env = q.environment(device)
    protocol = build_train_protocol(args, described, data)
    shards = {k: v["sha256"] for k, v in sorted(data.summary()["splits"].items())}
    cfg_hash = _hash_payload(stage, args.quick, protocol, shards, code_sha256(), git["git_sha"], env)
    train_tokens = data.manifest["splits"]["train"]["tokens"]
    log_path = resume_log_path(args.output)
    rows = load_rows(log_path, cfg_hash, args.resume)
    loaded = len(rows)
    done = {(r["config"], r["optimizer"], r["seed"]) for r in rows}
    log(f"[{stage}] git {git['git_sha']} dirty={git['git_dirty']} device={env.get('device', 'cpu')} "
        f"hash {cfg_hash} canonical={args.canonical} resumed rows {loaded}")
    meta = _meta(args, git, env, numerics, device)
    torch._dynamo.config.cache_size_limit = max(getattr(torch._dynamo.config, "cache_size_limit", 8), 64)
    plan = train_plan(stage, protocol)
    for label, raw in protocol["configs"].items():
        config = h.ModelConfig(**raw)
        todo = [p for p in plan if p[0] == label and (label, p[3], p[1]) not in done]
        if not todo:
            continue
        with torch.device(device):
            shared = h.Transformer(config)
        reusable = (shared, torch.compile(shared) if protocol["compile_model"] else shared)
        init_seed, init = None, None
        for _, seed, order, name in todo:
            if seed != init_seed:
                init_seed, init = seed, h.initial_state(config, seed)  # one seed's weights in memory
            row = run_trial(config, data, device=device, seed=seed, name=name, factory=found[name],
                            protocol=protocol, init=init, reusable=reusable)
            row.update({"config": label, "run_config_hash": cfg_hash, "order_in_block": order,
                        "epoch_fraction": row["tokens"] / train_tokens})
            row = json.loads(json.dumps(row, allow_nan=False))  # plain JSON types, as after a resume
            rows.append(row)
            _append(log_path, row)
            t, peak = row["timing"], row["peak_vram_mib"] or {}
            log(f"[{stage}] {label} seed {seed} {name:<19} steps {row['steps']:>3} step {t['step_wall_ms']:.1f} ms "
                f"fwd+bwd {t.get('fwd_bwd_ms') or 0:.1f} polar {t.get('polar_ortho_ms') or 0:.2f} "
                f"frac {row['polar_fraction'] or 0:.4f} tok/s {row['tokens_per_second'] or 0:,.0f} "
                f"val {row['val_loss'] or math.nan:.4f} test {row['test_loss'] or math.nan:.4f} "
                f"peak {peak.get('allocated') or 0:,.0f} MiB")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del shared, reusable, init
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _finish(meta, args, loaded, rows, started, device)
    position = {(label, name, seed): (i, order) for i, (label, seed, order, name) in enumerate(plan)}
    rows.sort(key=lambda r: position[(r["config"], r["optimizer"], r["seed"])])
    payload = {"schema_version": SCHEMA_VERSION, "kind": "runs", "stage": stage, "task": TASK, "quick": args.quick,
               "canonical": args.canonical, "meta": meta,
               "models": {label: model_record(h.ModelConfig(**c)) for label, c in protocol["configs"].items()},
               "data": {**data.summary(), "directory": str(args.data_dir), "sha256_verified": verify,
                        "expected_sha256": q.EXPECTED_SHA256 if verify else None, "train_tokens": train_tokens},
               "protocol": protocol, "run_config_hash": cfg_hash, "runs": rows}
    payload["derived"] = derive_train(payload)
    payload = json.loads(json.dumps(payload, allow_nan=False))
    validate_results(payload, canonical=args.canonical)
    return payload


def _mean(values: list) -> float | None:
    return math.fsum(values) / len(values) if values and all(v is not None for v in values) else None


def derive_train(payload: dict) -> dict:
    """Per (config, optimizer) means and the budget checks of one training stage."""
    rows, per = payload["runs"], {}
    for label in payload["protocol"]["configs"]:
        for name in payload["protocol"]["optimizers"]:
            mine = [r for r in rows if (r["config"], r["optimizer"]) == (label, name)]
            entry = {"runs": len(mine), "diverged": sum(r["diverged"] for r in mine),
                     "tokens_per_second_mean": _mean([r["tokens_per_second"] for r in mine]),
                     "polar_fraction_mean": _mean([r["polar_fraction"] for r in mine]),
                     "peak_vram_allocated_mib_max": (max(r["peak_vram_mib"]["allocated"] for r in mine)
                                                     if mine and all(r["peak_vram_mib"] for r in mine) else None),
                     **{f"{k}_mean": _mean([r["timing"].get(k) for r in mine]) for k in TIMING_KEYS}}
            for step in payload["protocol"]["checkpoints"]:
                for split in ("val", "test"):
                    values = [c[f"{split}_loss"] for r in mine for c in r["checkpoints"] if c["step"] == step]
                    entry[f"{split}_loss_step{step}_mean"] = _mean(values) if len(values) == len(mine) else None
            per.setdefault(label, {})[name] = entry
    peaks = [r["peak_vram_mib"]["reserved"] for r in rows if r["peak_vram_mib"]]
    return {"per_config": per, "runs": len(rows), "divergences": sum(r["diverged"] for r in rows),
            "max_epoch_fraction": max((r["epoch_fraction"] for r in rows), default=None),
            "max_peak_vram_reserved_mib": max(peaks) if peaks else None}


RUN_FIELDS = ("seed", "steps", "tokens", "epoch_fraction", "train_seconds", "tokens_per_second", "polar_fraction",
              "diverged", "init_fingerprint", "stream_fingerprint", "order_in_block", "checkpoints")


def per_seed_table(run_payload: dict) -> dict:
    """config -> optimizer -> per-seed rows (sorted by seed), with the timing medians inlined."""
    table: dict = {}
    for label in run_payload["protocol"]["configs"]:
        for name in run_payload["protocol"]["optimizers"]:
            rows = sorted((r for r in run_payload["runs"] if (r["config"], r["optimizer"]) == (label, name)),
                          key=lambda r: r["seed"])
            table.setdefault(label, {})[name] = [
                {**{k: r[k] for k in RUN_FIELDS}, **{k: r["timing"].get(k) for k in TIMING_KEYS},
                 "peak_vram_allocated_mib": (r["peak_vram_mib"] or {}).get("allocated")} for r in rows]
    return table


def analyze_train(stage: str, per_seed: dict, checkpoints: list[int]) -> dict:
    """Predeclared paired contrasts of one training stage with Holm-adjusted p-values.

    ``p_holm_family``: ``wide`` adjusts within (config, metric), across its contrasts; ``ckpt``
    adjusts within (metric, contrast), across the checkpoints. ``p_holm_joint`` adjusts within
    one metric across everything else (configs, contrasts and checkpoints).
    A contrast with a missing value (CPU smoke runs have no CUDA-event timing) is omitted.
    """
    contrasts = []
    for label, table in per_seed.items():
        for reference, candidates in CONTRASTS[stage]:
            for optimizer in candidates:
                if optimizer not in table or reference not in table:
                    continue
                mine, ref = table[optimizer], table[reference]
                seeds = [r["seed"] for r in mine]
                if seeds != [r["seed"] for r in ref] or len(seeds) not in (3, 5):
                    raise ValueError(f"{label} {optimizer} vs {reference}: seeds differ or are not 3 or 5")
                for a, b in zip(mine, ref):
                    if (a["init_fingerprint"], a["stream_fingerprint"]) != (
                            b["init_fingerprint"], b["stream_fingerprint"]):
                        raise ValueError(f"{label} seed {a['seed']}: {optimizer} and {reference} are not paired")
                cells = [(metric, kind, None, [r[metric] for r in mine], [r[metric] for r in ref])
                         for metric, kind in RUN_METRICS]
                for step in checkpoints:
                    def at(rows, metric):
                        return [next((c[metric] for c in r["checkpoints"] if c["step"] == step), None)
                                for r in rows]
                    cells += [(metric, kind, step, at(mine, metric), at(ref, metric)) for metric, kind in LOSS_METRICS]
                for metric, kind, step, x, y in cells:
                    if any(v is None for v in (*x, *y)):
                        continue
                    values = [a / b for a, b in zip(x, y)] if kind == "ratio" else [a - b for a, b in zip(x, y)]
                    pair = f"{optimizer}_vs_{reference}"
                    contrasts.append({
                        "id": f"{label}:{metric}_{kind}" + (f"@{step}" if step else "") + f":{pair}",
                        "config": label, "optimizer": optimizer, "reference": reference, "metric": metric,
                        "kind": kind, "checkpoint": step, "seeds": seeds,
                        "family": (f"{label}|{metric}_{kind}" if stage == "wide"
                                   else f"{label}|{metric}_{kind}|{pair}"),
                        "joint": f"{metric}_{kind}",
                        "stats": paired_t(values, 1.0 if kind == "ratio" else 0.0)})
    sizes = {}
    for field, key in (("p_holm_family", "family"), ("p_holm_joint", "joint")):
        members: dict = {}
        for i, c in enumerate(contrasts):
            members.setdefault(c[key], []).append(i)
        for idx in members.values():
            for i, p in zip(idx, holm_bonferroni([contrasts[i]["stats"]["p_two_sided"] for i in idx])):
                contrasts[i][field] = p
        sizes[field] = {group: len(idx) for group, idx in members.items()}
    per_optimizer = {label: {name: {k: mean_sd([r[k] for r in rows]) for k in (
        "tokens_per_second", "polar_fraction", *TIMING_KEYS) if all(r[k] is not None for r in rows)}
        for name, rows in table.items()} for label, table in per_seed.items()}
    return {"per_optimizer": per_optimizer, "contrasts": contrasts, "holm_group_sizes": sizes}


# ======================================================================================
# Drift stage: exact model of the Tensor-Core FP32 accumulate
# ======================================================================================
_UNIT = 600  # integer unit 2^-600: every product of two FP16 or BF16 values is an integer multiple


def _scaled(x: float) -> int:
    numerator, denominator = float(x).as_integer_ratio()
    return (numerator << _UNIT) // denominator  # exact: the denominator is a power of two <= 2^149


def _truncate(value: int, shift: int) -> int:
    """Drop the lowest ``shift`` bits of the magnitude (round toward zero)."""
    if shift <= 0:
        return value
    return (abs(value) >> shift << shift) * (1 if value >= 0 else -1)


def mma_accumulate(a: list[float], b: list[float], acc: float = 0.0, *, group: int = 16, guard: int = 2) -> float:
    """Binade-aligned integer model of ``acc + sum(a_k * b_k)`` for sm_120 Tensor-Core MMA.

    One MMA instruction takes ``group`` = 16 products and the FP32 accumulator:

    1. Every product ``a_k * b_k`` of two FP16/BF16 values is exact.
    2. All 17 addends are aligned to the largest binary exponent ``E`` among them and truncated
       toward zero below ``2^(E - 23 - guard)``: 24 significand bits plus ``guard`` = 2 guard bits.
    3. Their sum is exact and is truncated toward zero to 24 significant bits.

    Step 2 explains the measured long-K diagonal bias: once the accumulator enters a binade
    where a product falls below the aligned truncation quantum ``2^(E - 25)``, the product is
    dropped (reproducing ``_syrk_kernel`` on 160/160 probe cases and 260/260 sweep diagonals).
    This is NOT a universal bit-exact specification of all Blackwell MMA arithmetic: live
    RTX 5080 probes show 1-ULP disagreements on a 4-term normal FP16 diagonal, signed
    off-diagonals, and FP32 exponent-limit cases.
    """
    value = _scaled(acc)
    for start in range(0, len(a), group):
        addends = [value, *(_scaled(x) * _scaled(y) >> _UNIT for x, y in zip(a[start:start + group],
                                                                              b[start:start + group]))]
        top = max(abs(v).bit_length() for v in addends)
        if top == 0:
            continue
        total = sum(_truncate(v, top - 24 - guard) for v in addends)
        value = _truncate(total, abs(total).bit_length() - 24)
    return value / (1 << _UNIT)


def mma_predict_diag(row: Tensor, *, group: int = 16, guard: int = 2, prefix_groups: int = 512) -> float | None:
    """``mma_accumulate(row, row)`` for one long row, vectorized over the binades of the accumulator.

    The products are squares, so the accumulator only grows. While it stays in one binade
    ``[2^E, 2^(E+1))``, the truncation unit of every product is the constant ``2^(E-23-guard)``
    and a group adds ``floor(sum(floor(p / unit)) / 2^guard)`` units of ``2^(E-23)``: one cumulative
    sum per binade. The first groups (until the accumulator exceeds the largest product) and the
    group that crosses a binade use the exact rule directly. Returns None if the accumulator does
    not exceed the largest product within ``prefix_groups`` groups.
    """
    products = row.detach().double().flatten().square()
    if products.numel() == 0 or not bool(products.max() > 0):
        return 0.0
    products = torch.nn.functional.pad(products, (0, -products.numel() % group)).view(-1, group)
    top = math.frexp(products.max().item())[1] - 1  # exponent of the largest product
    head = row.detach().double().flatten()[:prefix_groups * group].tolist()
    acc, start = 0.0, 0
    while start < len(products) and (acc == 0.0 or math.frexp(acc)[1] - 1 <= top):
        if start >= prefix_groups:
            return None
        chunk = head[start * group:(start + 1) * group]
        acc, start = mma_accumulate(chunk, chunk, acc, group=group, guard=guard), start + 1
    while start < len(products):
        exponent = math.frexp(acc)[1] - 1
        unit = 2.0 ** (exponent - 23 - guard)  # truncation unit of a product
        kept = torch.floor(products[start:] / unit).sum(1)  # per group, in units of ``unit`` (exact)
        running = acc / (unit * 2 ** guard) + torch.cumsum(torch.floor(kept / 2 ** guard), 0)
        crossed = torch.nonzero(running >= 2.0 ** 24)
        if crossed.numel() == 0:
            return running[-1].item() * unit * 2 ** guard
        index = int(crossed[0])
        before = running[index - 1].item() * unit * 2 ** guard if index else acc
        total = (before / unit + kept[index].item()) * unit  # exact sum of the crossing group
        ulp = 2.0 ** (math.frexp(total)[1] - 1 - 23)
        acc, start = math.floor(total / ulp) * ulp, start + index + 1
    return acc


# ======================================================================================
# Drift stage: prototype mitigation (the production kernels are not modified)
# ======================================================================================
def chunk_view(view: Tensor, chunk: int) -> Tensor:
    """(n, K) -> (K // chunk, n, chunk) strided view of the first ``K // chunk * chunk`` columns.

    No copy: ``_syrk_kernel`` takes the three strides, so one launch with ``batch = K // chunk``
    computes every chunk Gram matrix in parallel.
    """
    count = view.shape[1] // chunk
    return view[:, :count * chunk].unflatten(1, (count, chunk)).transpose(0, 1)


class ChunkedGram:
    """Chunked-K SYRK: the prototype mitigation of the long-K Gram bias.

    For ``K > gate`` the Gram matrix is the FP64 sum of ``K / chunk`` chunk Gram matrices. Each
    chunk is the production ``_syrk_kernel`` (FP16/BF16 operands, FP32 Tensor-Core accumulation)
    on a strided view, so the accumulator of a chunk is at most ``chunk`` products long and the
    truncation bias is that of ``K = chunk`` (about 2e-5 for 4096). For ``K <= gate`` every call
    goes to the production code unchanged, so normal LLM shapes are bitwise identical.

    ``patched()`` installs it in ``polar_switch_blackwell._gram`` and in the two SYRK launches of
    ``schatten_switch_blackwell`` for the duration of a ``with`` block. Inside, the Schatten
    bound uses ``k = chunk`` for its FP32 slack, because that is the accumulation length now.
    """

    def __init__(self, chunk: int = 4096, gate: int = 65536, max_bytes: int = 1 << 28):
        if chunk <= 0 or gate < chunk:
            raise ValueError("Need 0 < chunk <= gate")
        self.chunk, self.gate, self.max_bytes, self.launches = chunk, gate, max_bytes, 0

    def gram(self, view: Tensor) -> Tensor:
        """FP64 Gram matrices (batch, n, n) of the rows of ``view`` (batch, n, K), any strides."""
        from research.polar_opt import polar_switch_blackwell as psb
        from research.polar_opt.kernels_blackwell import _syrk_kernel

        batch, n, k = view.shape
        out = torch.zeros(batch, n, n, dtype=torch.float64, device=view.device)
        step = max(1, self.max_bytes // (4 * n * n))  # chunk Gram matrices per launch
        for index in range(batch):
            chunks = chunk_view(view[index], self.chunk)
            pieces = [chunks[i:i + step] for i in range(0, len(chunks), step)]
            if k % self.chunk:
                pieces.append(view[index:index + 1, :, len(chunks) * self.chunk:])
            for piece in pieces:
                parts = torch.empty(len(piece), n, n, dtype=torch.float32, device=view.device)
                psb._launch(_syrk_kernel, psb._triangle_grid(n, len(piece)), piece, parts, n, piece.shape[-1],
                            *piece.stride())
                out[index] += parts.sum(0, dtype=torch.float64)
                self.launches += 1
        return out

    @contextmanager
    def patched(self):
        from research.polar_opt import polar_switch_blackwell as psb
        from research.polar_opt import schatten_switch_blackwell as ssb

        gram, launch, fold, gate = psb._gram, ssb._launch, ssb._fold_segment1, ssb.SYRK_CHUNK_GATE

        def patched_gram(X, out, sketch_norm=None):
            view = X.mT if X.shape[-2] > X.shape[-1] else X
            if sketch_norm is not None or view.shape[-1] <= self.gate:
                return gram(X, out, sketch_norm)
            out.copy_(self.gram(view))

        def patched_launch(kernel, grid, *args):
            if kernel is ssb._syrk_norm_kernel and args[4] > self.gate:
                view, out, scratch, n, _, _, _, _, off_r, _ = args
                out.copy_(self.gram(view))
                capacity = ssb._scratch_layout(n, 1)[1] - off_r
                scratch[:, off_r:off_r + capacity] = 0.0  # one partial holds ||R_h||_F^2 of the stored R_h
                scratch[:, off_r] = out.double().square().sum((-2, -1))
                return {"BLOCK": ssb._MIN_BLOCK}
            if kernel is ssb._syrk_kernel and args[3] > self.gate:
                args[1].copy_(self.gram(args[0]))
                return {"BLOCK": ssb._MIN_BLOCK}
            return launch(kernel, grid, *args)

        def patched_fold(circuit, R, S, scratch, tiles_r, tiles_s, *, k, use_frobenius):
            return fold(circuit, R, S, scratch, tiles_r, tiles_s, k=self.chunk if k > self.gate else k,
                        use_frobenius=use_frobenius)

        # C5-01: the native chunked path of ``ssb`` is switched off, so the historical hook above
        # still sees the long-K ``_syrk_norm_kernel`` / ``_syrk_kernel`` launches.
        psb._gram, ssb._launch, ssb._fold_segment1 = patched_gram, patched_launch, patched_fold
        ssb.SYRK_CHUNK_GATE = 1 << 60
        try:
            yield self
        finally:
            psb._gram, ssb._launch, ssb._fold_segment1 = gram, launch, fold
            ssb.SYRK_CHUNK_GATE = gate


@contextmanager
def pinned_syrk(config: dict = PIN_CONFIG):
    """Run the two K-keyed SYRK kernels with one fixed Triton configuration instead of autotuning.

    One autotune sweep at ``K = 2^25`` costs minutes per (kernel, dtype) because a ``BLOCK=128``
    trial computes a masked 128 x 128 tile for a 4-row matrix. The stored result does not depend
    on the configuration (``drift`` unit ``config_independence`` compares all of them bitwise).
    """
    import triton

    from research.polar_opt import schatten_switch_blackwell as ssb

    kernels = (ssb._syrk_kernel, ssb._syrk_norm_kernel)
    saved = [kernel.configs for kernel in kernels]
    pin = triton.Config({"BLOCK": config["BLOCK"], "BK": config["BK"]}, num_warps=config["num_warps"],
                        num_stages=config["num_stages"])
    for kernel in kernels:
        kernel.configs = [pin]
    try:
        yield
    finally:
        for kernel, configs in zip(kernels, saved):
            kernel.configs = configs


# ======================================================================================
# Drift stage: measurements
# ======================================================================================
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def kernel_call(kernel: str, dtype: str):
    """``f(G) -> Q`` of a production Blackwell orthogonalizer. SS3 in BF16 needs ``allow_unsafe_bf16``."""
    from research.polar_opt.polar_switch_blackwell import polar_switch_blackwell
    from research.polar_opt.schatten_switch_blackwell import schatten_switch_blackwell

    if kernel == "ps3_55":
        return lambda G: polar_switch_blackwell(G, "PS3_55", dtype=DTYPES[dtype])
    return lambda G: schatten_switch_blackwell(G, kernel.upper(), dtype=DTYPES[dtype], allow_unsafe_bf16=True)


def design_ceilings(kernels: list[str]) -> dict:
    """Largest value of each circuit's exact scalar map on ``sigma / sigma_hat`` in [0, 1].

    ``SS3_54`` stays below 1.08, but ``PS3_55`` (1.0810) and ``SS3_53`` (1.0875) exceed the common
    1.08 ceiling by design, so the drift onset is reported against both ceilings.
    """
    import numpy as np

    from research.polar_opt.baselines import _SWITCH_TABLES
    from research.polar_opt.schatten_switch_blackwell import SCHATTEN_SWITCH_CIRCUITS

    grid, out = np.linspace(0.0, 1.0, 100_001), {}
    for kernel in kernels:
        table = (_SWITCH_TABLES["PS3_55"][0] if kernel == "ps3_55"
                 else SCHATTEN_SWITCH_CIRCUITS[kernel.upper()].table)
        x = grid.copy()
        for residual, rows in table:
            work = {0: x * x}
            for dst, src, add, a, b, c in rows:
                work[dst] = a * work[src] ** 2 + b * work[add] + c
            x = x * work[dst] + residual * x
        out[kernel] = float(x.max())
    return out


def gram64(left: Tensor, right: Tensor | None = None, block: int = 1 << 20) -> Tensor:
    """Exact-arithmetic reference ``left @ right^T`` (n, n) in FP64, in column blocks (no K x K memory)."""
    right = left if right is None else right
    out = torch.zeros(left.shape[0], right.shape[0], dtype=torch.float64, device=left.device)
    for start in range(0, left.shape[1], block):
        out += left[:, start:start + block].double() @ right[:, start:start + block].double().mT
    return out


def _short(matrix: Tensor) -> Tensor:
    """(n, K) view with the short side first."""
    return matrix if matrix.shape[0] <= matrix.shape[1] else matrix.mT


def polar_quality(Q: Tensor, G: Tensor, inverse_root: Tensor) -> dict:
    """Singular values of ``Q`` and its error against the exact polar factor, all in FP64.

    With ``A = G G^T`` (short side), the polar factor is ``P = A^(-1/2) G``. Then
    ``||Q - P||_F^2 = tr(Q Q^T) + n - 2 tr(Q G^T A^(-1/2))``, so only n x n matrices are formed.
    The singular values are the square roots of the eigenvalues of the FP64 ``Q Q^T``.
    """
    if not bool(torch.isfinite(Q).all()):
        return {"finite": False, "sigma_min": None, "sigma_max": None, "rel_fro_err": None}
    q_view, g_view = _short(Q), _short(G)
    QQ = gram64(q_view)
    sigma = torch.linalg.eigvalsh(QQ).clamp_min(0.0).sqrt()
    n = QQ.shape[0]
    error = QQ.diagonal().sum() + n - 2.0 * (gram64(q_view, g_view) @ inverse_root).diagonal().sum()
    return {"finite": True, "sigma_min": sigma.min().item(), "sigma_max": sigma.max().item(),
            "rel_fro_err": math.sqrt(max(error.item(), 0.0) / n)}


def _timed(function, reps: int) -> tuple:
    """(result of the first call, median milliseconds of ``reps`` later synchronized calls)."""
    result = function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        begin = time.perf_counter()
        out = function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - begin) * 1e3)
        del out
    return result, median(samples)


def _syrk32(view: Tensor) -> Tensor:
    """Raw FP32 accumulator of the production ``_syrk_kernel`` for one (n, K) view (one K loop)."""
    from research.polar_opt import polar_switch_blackwell as psb
    from research.polar_opt.kernels_blackwell import _syrk_kernel

    n, k = view.shape
    out = torch.empty(1, n, n, dtype=torch.float32, device=view.device)
    psb._launch(_syrk_kernel, psb._triangle_grid(n, 1), view[None], out, n, k, *view[None].stride())
    return out[0]


def _bias(estimate: Tensor, reference: Tensor) -> dict:
    """Relative diagonal bias ``1 - est / ref`` and the largest off-diagonal error over the mean diagonal."""
    estimate = estimate.double()
    diag = 1.0 - estimate.diagonal() / reference.diagonal()
    off = (estimate - reference).abs()
    off = off - torch.diag(off.diagonal())
    return {"diag_bias_mean": diag.mean().item(), "diag_bias_max": diag.max().item(),
            "offdiag_err_max": (off.max() / reference.diagonal().mean()).item()}


def gram_decomposition(G: Tensor, dtype: str, protocol: dict) -> dict:
    """Isolate the three candidate causes on the Frobenius-normalized operand of ``G`` in ``dtype``.

    1. Operand rounding and FP16 subnormals: exact FP64 Gram of the rounded operand against the
       exact FP64 Gram of the unrounded one.
    2. Frobenius pre-scaling: the SYRK of the operand times ``2^p`` (exact, every entry normal)
       against the SYRK of the operand, compared bitwise after the exact rescale.
    3. Accumulation: one sequential K loop of FP32 Tensor-Core accumulates (``_syrk_kernel`` and
       ``_syrk_norm_kernel``), cuBLAS FP16 GEMM, cuBLAS FP32 GEMM, and chunked FP32 accumulation
       summed in FP64, in FP32 (``torch.sum``) and pairwise in FP32.
    """
    from research.polar_opt import polar_switch_blackwell as psb
    from research.polar_opt import schatten_switch_blackwell as ssb

    work, reps = DTYPES[dtype], protocol["latency_reps"]
    X, norm = psb._prepare(G, work)  # the production operand: fl(G / (||G||_F + 1e-7))
    view = _short(X[0])
    n, k = view.shape
    unrounded = gram64(_short(G)) / (norm.double().item() + psb._EPS) ** 2
    exact = gram64(view)
    tiny = torch.finfo(work).tiny
    out = {"operand": {"subnormal_fraction": ((view.abs() < tiny) & (view != 0)).float().mean().item(),
                       "zero_fraction": (view == 0).float().mean().item(),
                       "rms": math.sqrt(exact.diagonal().sum().item() / (n * k)),
                       "smallest_normal": tiny},
           "operand_rounding": _bias(exact, unrounded), "estimators": {}}

    def add(name, function):
        value, ms = _timed(function, reps)
        out["estimators"][name] = {**_bias(value, exact), "ms": ms}
        return value

    sequential = add("triton_sequential", lambda: _syrk32(view))
    scale = 2.0 ** protocol["prescale_exponent"]
    prescaled = _syrk32(view * scale) / scale ** 2
    stored = torch.empty(1, n, n, dtype=work, device=view.device)
    off_r, _, _, stride = ssb._scratch_layout(n, 1)
    scratch = torch.zeros(1, stride, dtype=torch.float64, device=view.device)
    psb._launch(ssb._syrk_norm_kernel, psb._triangle_grid(n, 1), view[None], stored, scratch, n, k,
                *view[None].stride(), off_r, stride)
    out["checks"] = {"prescale_bitwise_equal": bool(torch.equal(prescaled, sequential)),
                     "prescaled_all_normal": bool(((view * scale).abs() >= tiny)[view != 0].all()),
                     "syrk_norm_equals_rounded_syrk": bool(torch.equal(stored[0], sequential.to(work)))}
    left, right = view.contiguous()[None], view.mT.contiguous()[None]  # copies are outside the timed call
    add("cublas_fp16_gemm", lambda: torch.bmm(left, right, out_dtype=torch.float32)[0])
    left, right = left.float(), right.float()
    add("cublas_fp32_gemm", lambda: torch.bmm(left, right)[0])
    del left, right
    for chunk in protocol["gram_chunks"]:
        if chunk < k:
            add(f"triton_chunk{chunk}_fp64sum", lambda c=chunk: ChunkedGram(c, c).gram(view[None])[0])
    chunk = protocol["chunk_k"]
    if chunk < k and k % chunk == 0:
        parts = torch.empty(k // chunk, n, n, dtype=torch.float32, device=view.device)
        pieces = chunk_view(view, chunk)
        psb._launch(ssb._syrk_kernel, psb._triangle_grid(n, len(pieces)), pieces, parts, n, chunk, *pieces.stride())
        out["estimators"][f"triton_chunk{chunk}_fp32sum"] = {**_bias(parts.sum(0), exact), "ms": None}
        while len(parts) > 1 and len(parts) % 2 == 0:  # pairwise FP32 tree (K / chunk is a power of two)
            parts = parts[0::2] + parts[1::2]
        out["estimators"][f"triton_chunk{chunk}_fp32pairwise"] = {**_bias(parts.sum(0), exact), "ms": None}
    rows = min(n, 4)
    predicted = [mma_predict_diag(view[r]) for r in range(rows)]
    measured = sequential.diagonal()[:rows].double().tolist()
    out["mma_model"] = {
        "rows": rows, "predicted_diag": predicted, "measured_diag": measured,
        "bitwise_equal": [p is not None and p == m for p, m in zip(predicted, measured)],
        "predicted_bias_mean": (_mean([1.0 - p / e for p, e in zip(predicted, exact.diagonal()[:rows].tolist())])
                                if all(p is not None for p in predicted) else None)}
    return out


def _input(shape: list[int], seed: int, device: torch.device) -> Tensor:
    return torch.randn(*shape, device=device, generator=torch.Generator(device).manual_seed(seed))


def sweep_cell(family: str, spec: dict, exponent: int, seed: int, protocol: dict, device: torch.device) -> dict:
    """One (shape, seed): every kernel in production and mitigated form, plus the Gram decomposition."""
    n, k = spec["n"], 1 << exponent
    shape = [n, k] if spec["orientation"] == "wide" else [k, n]
    G = _input(shape, seed, device)
    A = gram64(_short(G))
    values, vectors = torch.linalg.eigh(A)
    inverse_root = (vectors / values.sqrt()) @ vectors.mT
    row = {"id": f"sweep:{family}:2^{exponent}:seed{seed}", "unit": "sweep", "family": family, "n": n, "k": k,
           "k_exponent": exponent, "shape": shape, "seed": seed,
           "input": {"sigma_min": values.min().sqrt().item(), "sigma_max": values.max().sqrt().item(),
                     "fro": values.sum().sqrt().item()}, "kernels": {}}
    mitigation = ChunkedGram(protocol["chunk_k"], protocol["gate_k"])
    for kernel in protocol["kernels"]:
        for dtype in protocol["dtypes"]:
            call, cell = kernel_call(kernel, dtype), {}
            Q, ms = _timed(lambda: call(G), protocol["latency_reps"])
            cell["production"] = {**polar_quality(Q, G, inverse_root), "ms": ms}
            del Q
            before = mitigation.launches
            with mitigation.patched():
                Q, ms = _timed(lambda: call(G), protocol["latency_reps"])
            cell["mitigated"] = {**polar_quality(Q, G, inverse_root), "ms": ms,
                                 "chunked": mitigation.launches > before}
            del Q
            row["kernels"][f"{kernel}_{dtype}"] = cell
    if seed in protocol["gram_seeds"]:
        row["gram"] = {dtype: gram_decomposition(G, dtype, protocol) for dtype in protocol["dtypes"]}
    return row


_PROBE_PRODUCTS = ((-10, -10, 1.0), (-11, -11, 1.0), (-11, -12, 1.0), (-12, -12, 1.0), (-12, -13, 1.0),
                   (-13, -13, 1.0), (-13, -14, 1.0), (-14, -14, 1.0), (-12, -12, 1.5), (-12, -13, 1.5),
                   (-12, -13, 1.75), (-12, -14, 1.75), (-12, -12, -1.0), (-12, -13, -1.0), (-12, -13, -1.5),
                   (-13, -13, -1.0))
_PROBE_COUNTS = (15, 31, 63, 1023, 4095)


def unit_mma_probe(protocol: dict, device: torch.device) -> dict:
    """Accumulator 1.0 plus ``count`` equal products ``a * b`` in one off-diagonal Gram entry.

    Row 0 is ``[1, a, a, ...]`` and row 1 is ``[1, b, b, ...]``. ``kept_ulp`` is the measured
    contribution of one product in units of ``2^-23`` (the ulp of 1.0); ``model_ulp`` is the
    same number from ``mma_accumulate``. A product of ``2^-26`` (1/8 ulp) contributes nothing.
    """
    cases = []
    for dtype in protocol["dtypes"]:
        for ea, eb, mantissa in _PROBE_PRODUCTS:
            a, b = 2.0 ** ea, mantissa * 2.0 ** eb
            for count in _PROBE_COUNTS:
                X = torch.zeros(2, count + 1, device=device)
                X[:, 0], X[0, 1:], X[1, 1:] = 1.0, a, b
                X = X.to(DTYPES[dtype])
                rows = X.double().tolist()
                triton_value = _syrk32(X)[0, 1].item()
                cublas = torch.bmm(X[None], X.mT.contiguous()[None], out_dtype=torch.float32)[0, 0, 1].item()
                model = mma_accumulate(rows[0], rows[1])
                cases.append({"dtype": dtype, "a": a, "b": b, "count": count, "exact_ulp": a * b * 2.0 ** 23,
                              "triton_kept_ulp": (triton_value - 1.0) / count * 2.0 ** 23,
                              "cublas_fp16_kept_ulp": (cublas - 1.0) / count * 2.0 ** 23,
                              "model_ulp": (model - 1.0) / count * 2.0 ** 23,
                              "model_equals_triton": model == triton_value})
    return {"id": "mma_probe", "unit": "mma_probe", "cases": cases}


def unit_mma_model(protocol: dict, device: torch.device) -> dict:
    """``mma_accumulate`` against the stored FP32 Gram of ``_syrk_kernel``: every entry, both signs."""
    cases = []
    for dtype in protocol["dtypes"]:
        for k in protocol["model_k"]:
            n = protocol["model_rows"]
            X = (_input([n, k], protocol["model_seed"] + k, device) / math.sqrt(n * k)).to(DTYPES[dtype])
            measured, rows = _syrk32(X).double().tolist(), X.double().tolist()
            pairs = [(i, j) for i in range(n) for j in range(i, n)]
            equal = [mma_accumulate(rows[i], rows[j]) == measured[i][j] for i, j in pairs]
            cases.append({"dtype": dtype, "n": n, "k": k, "entries": len(pairs), "bitwise_equal": sum(equal),
                          "diag_bitwise_equal": sum(e for e, (i, j) in zip(equal, pairs) if i == j),
                          "vectorized_diag_equal": sum(mma_predict_diag(X[i]) == measured[i][i] for i in range(n))})
    return {"id": "mma_model", "unit": "mma_model", "cases": cases}


def unit_config_independence(protocol: dict, device: torch.device) -> dict:
    """The raw FP32 Gram of ``_syrk_kernel`` under every autotune configuration, compared bitwise."""
    import triton

    from research.polar_opt import polar_switch_blackwell as psb
    from research.polar_opt.kernels_blackwell import _CONFIGS

    cases = []
    for shape in protocol["config_shapes"]:
        view = _short((_input(shape, 1, device) / math.sqrt(shape[0] * shape[1])).half())
        results = []
        for config in _CONFIGS:
            pin = {"BLOCK": config.kwargs["BLOCK"], "BK": config.kwargs["BK"], "num_warps": config.num_warps,
                   "num_stages": config.num_stages}
            psb._PLANS.clear()  # a plan keeps its configuration
            with pinned_syrk(pin):
                results.append(_syrk32(view).clone())
        psb._PLANS.clear()
        cases.append({"shape": shape, "configs": len(results),
                      "bitwise_equal": sum(bool(torch.equal(results[0], r)) for r in results)})
    return {"id": "config_independence", "unit": "config_independence", "triton": triton.__version__,
            "cases": cases}


def unit_svd_check(protocol: dict, device: torch.device) -> dict:
    """The Gram-eigenvalue singular values against ``torch.linalg.svdvals`` in FP64."""
    cases = []
    for shape in protocol["svd_check"]:
        G = _input(shape, 0, device)
        Q = kernel_call("ss3_54", "fp16")(G)
        direct = torch.linalg.svdvals(_short(Q).double())
        via_gram = torch.linalg.eigvalsh(gram64(_short(Q))).clamp_min(0.0).sqrt().flip(0)
        cases.append({"shape": shape, "max_abs_diff": (direct - via_gram).abs().max().item(),
                      "sigma_max_svd": direct.max().item()})
    return {"id": "svd_check", "unit": "svd_check", "cases": cases}


def unit_normal_shape(shape: list[int], protocol: dict, device: torch.device) -> dict:
    """A normal LLM shape: the mitigation is not triggered, the output is bitwise equal, latency is equal."""
    G = _input(shape, 0, device)
    mitigation = ChunkedGram(protocol["chunk_k"], protocol["gate_k"])
    row = {"id": f"normal:{shape[0]}x{shape[1]}", "unit": "normal", "shape": shape, "kernels": {}}
    for kernel in protocol["kernels"]:
        call = kernel_call(kernel, "fp16")
        reference = call(G)
        with mitigation.patched():
            patched = call(G)
        torch.cuda.synchronize()
        samples = {"production": [], "mitigated": []}
        for _ in range(protocol["normal_reps"]):  # interleaved
            for name in samples:
                begin = time.perf_counter()
                if name == "mitigated":
                    with mitigation.patched():
                        call(G)
                else:
                    call(G)
                torch.cuda.synchronize()
                samples[name].append((time.perf_counter() - begin) * 1e3)
        row["kernels"][f"{kernel}_fp16"] = {
            "bitwise_equal": bool(torch.equal(reference, patched)), "chunked_launches": mitigation.launches,
            "production_ms": median(samples["production"]), "mitigated_ms": median(samples["mitigated"]),
            "production_samples_ms": samples["production"], "mitigated_samples_ms": samples["mitigated"]}
    return row


def drift_units(protocol: dict) -> list[tuple[str, tuple]]:
    """(row id, arguments) of every unit of the drift stage, in run order."""
    units = [("mma_probe", ()), ("mma_model", ()), ("config_independence", ()), ("svd_check", ())]
    for family, spec in protocol["families"].items():
        units += [(f"sweep:{family}:2^{e}:seed{seed}", (family, e, seed))
                  for e in spec["k_exponents"] for seed in spec["seeds"]]
    return units + [(f"normal:{s[0]}x{s[1]}", (s,)) for s in protocol["normal_shapes"]]


def run_drift_stage(args) -> dict:
    device, git, log, started = _start(args)
    if device.type != "cuda":
        raise RuntimeError("The drift stage runs the Blackwell Triton kernels and needs a CUDA device")
    numerics = set_numerics(device)
    env = q.environment(device)
    protocol = copy.deepcopy(args.protocol)
    cfg_hash = _hash_payload("drift", args.quick, protocol, None, code_sha256(), git["git_sha"], env)
    log_path = resume_log_path(args.output)
    rows = load_rows(log_path, cfg_hash, args.resume)
    loaded = len(rows)
    done = {r["id"] for r in rows}
    log(f"[drift] git {git['git_sha']} dirty={git['git_dirty']} device={env.get('device')} hash {cfg_hash} "
        f"canonical={args.canonical} resumed rows {loaded}")
    meta = _meta(args, git, env, numerics, device)
    simple = {"mma_probe": unit_mma_probe, "mma_model": unit_mma_model, "svd_check": unit_svd_check}
    for unit_id, unit_args in drift_units(protocol):
        if unit_id in done:
            continue
        began = time.perf_counter()
        if unit_id == "config_independence":
            row = unit_config_independence(protocol, device)
        elif unit_id.startswith("normal:"):
            row = unit_normal_shape(*unit_args, protocol, device)  # natural autotuning, as in production
        else:
            with pinned_syrk(protocol["pin_config"]):
                if unit_id in simple:
                    row = simple[unit_id](protocol, device)
                else:
                    family, exponent, seed = unit_args
                    row = sweep_cell(family, protocol["families"][family], exponent, seed, protocol, device)
        row.update({"run_config_hash": cfg_hash, "seconds": time.perf_counter() - began})
        row = json.loads(json.dumps(row, allow_nan=False))
        rows.append(row)
        _append(log_path, row)
        if row["unit"] == "sweep":
            shown = " ".join(f"{name} {cell['production']['sigma_max'] or math.nan:.3f}->"
                             f"{cell['mitigated']['sigma_max'] or math.nan:.3f}"
                             for name, cell in row["kernels"].items())
            log(f"[drift] {unit_id} ({row['seconds']:.1f}s) sigma_max production->mitigated: {shown}")
        else:
            log(f"[drift] {unit_id} ({row['seconds']:.1f}s)")
        torch.cuda.empty_cache()
    _finish(meta, args, loaded, rows, started, device)
    order = {unit_id: i for i, (unit_id, _) in enumerate(drift_units(protocol))}
    rows.sort(key=lambda r: order[r["id"]])
    payload = {"schema_version": SCHEMA_VERSION, "kind": "drift", "stage": "drift", "task": TASK,
               "quick": args.quick, "canonical": args.canonical, "meta": meta, "data": None,
               "protocol": protocol, "opus_l6": OPUS_L6, "run_config_hash": cfg_hash, "rows": rows}
    payload["derived"] = derive_drift(payload)
    payload = json.loads(json.dumps(payload, allow_nan=False))
    validate_results(payload, canonical=args.canonical)
    return payload


def _onset(points: list[tuple[int, float]], ceiling: float) -> dict:
    """First grid ``K`` whose worst-seed ``sigma_max`` exceeds ``ceiling``, with its bracket.

    ``interpolated_k`` is linear in ``log2 K`` between the two bracketing grid points.
    """
    points = sorted(points)
    last = None if not points or math.isinf(points[-1][1]) else points[-1][1]
    above = [i for i, (_, s) in enumerate(points) if s > ceiling]
    if not above:
        return {"first_k_above": None, "last_k_at_or_below": points[-1][0] if points else None,
                "interpolated_k": None, "monotone_above": None, "sigma_max_at_largest_k": last}
    i = above[0]
    (k1, s1), (k0, s0) = points[i], points[i - 1] if i else (None, None)
    crossing = None
    if i and math.isfinite(s1) and s1 > s0:
        crossing = 2.0 ** (math.log2(k0) + (ceiling - s0) / (s1 - s0) * (math.log2(k1) - math.log2(k0)))
    return {"first_k_above": k1, "last_k_at_or_below": k0, "interpolated_k": crossing,
            "monotone_above": all(s > ceiling for _, s in points[i:]), "sigma_max_at_largest_k": last}


def derive_drift(payload: dict) -> dict:
    """Headline numbers of the drift stage, recomputed from its rows by ``validate_results``."""
    protocol, rows = payload["protocol"], payload["rows"]
    sweep = [r for r in rows if r["unit"] == "sweep"]
    names = [f"{kernel}_{dtype}" for kernel in protocol["kernels"] for dtype in protocol["dtypes"]]
    ceiling, design = protocol["ceiling"], design_ceilings(protocol["kernels"])
    own = {name: design[name.rsplit("_", 1)[0]] for name in names}

    def worst(family, name, variant):  # K -> largest sigma_max over seeds (non-finite counts as infinite)
        cells: dict = {}
        for r in sweep:
            if r["family"] == family:
                value = r["kernels"][name][variant]["sigma_max"]
                cells[r["k"]] = max(cells.get(r["k"], 0.0), math.inf if value is None else value)
        return sorted(cells.items())

    # "common": the 1.08 ceiling of Opus L6; "design": the circuit's own exact-arithmetic maximum.
    onset = {family: {name: {variant: {"common": _onset(worst(family, name, variant), ceiling),
                                       "design": _onset(worst(family, name, variant), own[name])}
                             for variant in ("production", "mitigated")} for name in names}
             for family in protocol["families"]}
    sigma = {family: {name: {variant: {str(k): (None if math.isinf(s) else s) for k, s in worst(family, name, variant)}
                             for variant in ("production", "mitigated")} for name in names}
             for family in protocol["families"]}
    reproduction = {}
    for key, target in payload["opus_l6"].items():
        family, name = key.split(":")
        values = [r["kernels"][name]["production"]["sigma_max"] for r in sweep
                  if r["family"] == family and r["shape"] == target["shape"]]
        if values and all(v is not None for v in values):
            reproduction[key] = {"shape": target["shape"], "opus": target["sigma_max"], "measured_min": min(values),
                                 "measured_max": max(values),
                                 # the measured range over seeds overlaps the audit range within 0.01
                                 "reproduced": min(values) <= target["sigma_max"][1] + 0.01
                                 and max(values) >= target["sigma_max"][0] - 0.01}
    gram = [(r, dtype, g) for r in sweep for dtype, g in r.get("gram", {}).items()]
    bias = {}
    for r, dtype, g in gram:
        entry = bias.setdefault(r["family"], {}).setdefault(dtype, {})
        entry[str(r["k"])] = {name: e["diag_bias_mean"] for name, e in g["estimators"].items()}
    slopes = [g["estimators"]["triton_sequential"]["diag_bias_mean"] / r["k"] for r, _, g in gram
              if 1 << 16 <= r["k"] <= 1 << 22]
    model_rows = [flag for _, _, g in gram for flag in g["mma_model"]["bitwise_equal"]]
    exact = [c for r in rows if r["unit"] == "mma_model" for c in r["cases"]]
    probe = [c for r in rows if r["unit"] == "mma_probe" for c in r["cases"]]
    mitigated = [(r, name, cell) for r in sweep for name, cell in r["kernels"].items()]
    chunked = [(r, name, cell) for r, name, cell in mitigated if cell["mitigated"]["chunked"]]
    speedups = [cell["production"]["ms"] / cell["mitigated"]["ms"] for r, _, cell in chunked
                if r["k"] == max(x["k"] for x in sweep)]
    normal = [cell for r in rows if r["unit"] == "normal" for cell in r["kernels"].values()]
    finite = [cell["mitigated"]["sigma_max"] for _, _, cell in mitigated]
    excess = [cell[variant]["sigma_max"] - own[name] if cell[variant]["sigma_max"] is not None else math.inf
              for variant in ("production", "mitigated") for _, name, cell in mitigated]
    excess = {"production": excess[:len(mitigated)], "mitigated": excess[len(mitigated):]}
    return {
        "ceiling": ceiling, "design_ceilings": design, "onset": onset, "sigma_max": sigma, "reproduction": reproduction,
        "gram_diag_bias": bias,
        "triton_bias_per_product_mean": _mean(slopes) if slopes else None,
        "checks": {
            "operand_rounding_bias_max": max((abs(g["operand_rounding"]["diag_bias_max"]) for _, _, g in gram),
                                             default=None),
            "prescale_bitwise_equal": [sum(g["checks"]["prescale_bitwise_equal"] for _, _, g in gram), len(gram)],
            "syrk_norm_equals_rounded_syrk": [sum(g["checks"]["syrk_norm_equals_rounded_syrk"] for _, _, g in gram),
                                              len(gram)],
            "mma_model_sweep_diag_bitwise": [sum(model_rows), len(model_rows)],
            "mma_model_exact_entries_bitwise": [sum(c["bitwise_equal"] for c in exact),
                                                sum(c["entries"] for c in exact)],
            "mma_probe_model_equals_triton": [sum(c["model_equals_triton"] for c in probe), len(probe)],
            "config_independence": [[c["bitwise_equal"], c["configs"]] for r in rows
                                    if r["unit"] == "config_independence" for c in r["cases"]],
            "svd_check_max_abs_diff": max((c["max_abs_diff"] for r in rows if r["unit"] == "svd_check"
                                           for c in r["cases"]), default=None)},
        "mitigation": {
            "chunk_k": protocol["chunk_k"], "gate_k": protocol["gate_k"],
            "cells": len(mitigated), "chunked_cells": len(chunked),
            "sigma_max_worst": (None if not finite or any(v is None for v in finite) else max(finite)),
            "cells_above_common_ceiling": {
                variant: sum(cell[variant]["sigma_max"] is None or cell[variant]["sigma_max"] > ceiling
                             for _, _, cell in mitigated) for variant in ("production", "mitigated")},
            "cells_above_design_ceiling": {variant: sum(v > 0 for v in values) for variant, values in excess.items()},
            "max_excess_over_design_ceiling": {
                variant: (None if not values or math.isinf(max(values)) else max(values))
                for variant, values in excess.items()},
            "unchunked_cells_bitwise_scope": "K <= gate_k calls the production code unchanged",
            "speedup_at_largest_k_min": min(speedups) if speedups else None,
            "speedup_at_largest_k_max": max(speedups) if speedups else None,
            "normal_shapes": {"cells": len(normal), "bitwise_equal": sum(c["bitwise_equal"] for c in normal),
                              "chunked_launches": sum(c["chunked_launches"] for c in normal),
                              "latency_ratio_max": max((c["mitigated_ms"] / c["production_ms"] for c in normal),
                                                       default=None)}},
    }


# ======================================================================================
# Summary
# ======================================================================================
def _source_record(source: dict, path: Path) -> dict:
    env = source["meta"]["environment"]
    return {"path": _relative(path), "sha256": sha256_file(path), "git_sha": source["meta"]["git_sha"],
            "git_dirty": source["meta"]["git_dirty"], "code_sha256": source["meta"]["code_sha256"],
            "run_config_hash": source["run_config_hash"], "quick": source["quick"],
            "canonical": source["canonical"], "device": env.get("device"), "capability": env.get("capability"),
            "protocol": source["protocol"]}


def _train_section(stage: str, source: dict) -> dict:
    per_seed = per_seed_table(source)
    return {"models": source["models"], "per_seed": per_seed,
            "analysis": analyze_train(stage, per_seed, source["protocol"]["checkpoints"]),
            "checks": {k: source["derived"][k] for k in ("runs", "divergences", "max_epoch_fraction",
                                                         "max_peak_vram_reserved_mib")}}


def build_summary(sources: dict, paths: dict, *, quick: bool) -> dict:
    """``sources``: {"wide": payload, "ckpt": payload, "drift": payload}; ``paths``: where they were read."""
    for stage in RUN_STAGES:
        if sources[stage].get("stage") != stage or sources[stage].get("kind") not in ("runs", "drift"):
            raise ValueError(f"{paths[stage]} is not a {stage} payload")
    canonical = all(s["canonical"] and not s["quick"] for s in sources.values())
    payload = {
        "schema_version": SCHEMA_VERSION, "kind": "summary", "stage": "summary", "task": TASK, "quick": quick,
        "canonical": canonical,
        "meta": {**h.git_info(), "command": [sys.executable, *sys.argv],
                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        "sources": {stage: _source_record(sources[stage], paths[stage]) for stage in RUN_STAGES},
        "data": sources["ckpt"]["data"],
        "statistics": {
            "test": "paired Student-t over seeds: df = 2 for wide (three seeds), df = 4 for ckpt (five seeds)",
            "t_quantile_0p975": {"df2": T975[2], "df4": T975[4]},
            "p_two_sided": {"df2": "2 / (s (s + |t|)), s = sqrt(t^2 + 2)",
                            "df4": "c^2 (3 - c) / 2, c = 4 / (s (s + |t|)), s = sqrt(t^2 + 4)"},
            "direction": "diff = optimizer - reference; ratio = optimizer / reference; a negative loss diff "
                         "and a tokens_per_second ratio > 1 favour the optimizer",
            "holm": {"wide": {"p_holm_family": "within (config, metric), across the five contrasts",
                              "p_holm_joint": "within one metric, across configs and contrasts"},
                     "ckpt": {"p_holm_family": "within (metric, contrast), across the checkpoints",
                              "p_holm_joint": "within one metric, across checkpoints and contrasts"}},
            "contrasts": {stage: [[ref, list(c)] for ref, c in plan] for stage, plan in CONTRASTS.items()},
            "metrics": {"per_run": [f"{m}_{k}" for m, k in RUN_METRICS],
                        "per_checkpoint": [f"{m}_{k}" for m, k in LOSS_METRICS]}},
        "wide": _train_section("wide", sources["wide"]), "ckpt": _train_section("ckpt", sources["ckpt"]),
        "drift": {"opus_l6": sources["drift"]["opus_l6"], "derived": sources["drift"]["derived"]},
    }
    return json.loads(json.dumps(payload, allow_nan=False))


# ======================================================================================
# Validation
# ======================================================================================
def _validate_provenance(meta: dict, data: dict | None, canonical: bool, where: str) -> None:
    _require(isinstance(meta.get("git_dirty"), bool), f"{where}: git_dirty must be recorded")
    if canonical:
        _require(_is_hex(meta.get("git_sha"), 40), f"{where}: canonical results need a 40-hex git_sha")
        _require(meta["git_dirty"] is False, f"{where}: canonical results must come from a clean git tree")
        if data is not None:
            _require({k: v["sha256"] for k, v in data["splits"].items()} == q.EXPECTED_SHA256,
                     f"{where}: shard SHA-256 values differ from the pinned q.EXPECTED_SHA256")
            _require(data["sha256_verified"] is True and data["expected_sha256"] == q.EXPECTED_SHA256,
                     f"{where}: shards must be verified against the pins")


def _validate_stage_common(payload: dict, canonical: bool) -> None:
    stage, meta = payload["stage"], payload["meta"]
    _validate_provenance(meta, payload["data"], canonical, stage)
    _require(_is_hex(meta.get("code_sha256"), 64), f"{stage}: code_sha256 must be 64-hex")
    _require(all(any(name.endswith("." + needed) for name in meta["code_modules"]) for needed in REQUIRED_CODE),
             f"{stage}: code_modules does not cover the driver, harness, data, adapters and kernels")
    _require(_is_hex(payload.get("run_config_hash"), 16), f"{stage}: run_config_hash must be 16-hex")
    _require(payload["run_config_hash"] == expected_run_config_hash(payload),
             f"{stage}: run_config_hash differs from the recomputed digest")
    numerics = meta["numerics"]
    _require(numerics["allow_tf32"] is False and numerics["cudnn_allow_tf32"] is False
             and numerics["float32_matmul_precision"] == "highest", "TF32 must be disabled")
    if meta["environment"].get("device") is not None:
        _require(numerics["allow_fp16_reduced_precision_reduction"] is False
                 and numerics["allow_bf16_reduced_precision_reduction"] is False,
                 "reduced-precision FP16/BF16 reductions must be disabled")
    if canonical:
        _require(payload["quick"] is False and payload["canonical"] is True, "a canonical result cannot be --quick")
        _require(list(meta["environment"].get("capability") or [0, 0]) >= [12, 0],
                 "a canonical result needs an sm_120+ CUDA device")
        for key, value in CANONICAL[stage].items():
            _require(payload["protocol"].get(key) == value,
                     f"protocol {key} differs from the canonical {stage} protocol")


def _validate_runs(payload: dict, canonical: bool) -> None:
    stage, protocol, rows = payload["stage"], payload["protocol"], payload["runs"]
    _validate_stage_common(payload, canonical)
    train_tokens = payload["data"]["splits"]["train"]["tokens"]
    _require(payload["data"]["train_tokens"] == train_tokens, "train_tokens differs from the manifest")
    if canonical:
        _require(protocol["orthogonalizers"] == {n: CUDA_ORTHOGONALIZERS[n] for n in protocol["optimizers"]},
                 "protocol orthogonalizers differ from CUDA_ORTHOGONALIZERS")
    steps, checkpoints = protocol["steps"], protocol["checkpoints"]
    _require(checkpoints == sorted(set(checkpoints)) and checkpoints[-1] == steps and checkpoints[0] > 0,
             "checkpoints must be increasing and end at the last step")
    configs = {label: h.ModelConfig(**c) for label, c in protocol["configs"].items()}
    _require(set(payload["models"]) == set(configs), "models differ from the protocol configs")
    for label, config in configs.items():
        _assert_close(payload["models"][label], json.loads(json.dumps(model_record(config))), f"models.{label}")
        _require(protocol["tokens"][label] == steps * config.step_tokens, "protocol tokens differ")
        for split in ("val", "test"):
            _require(protocol["eval_tokens_used"][label][split] == eval_tokens_used(
                payload["data"]["splits"][split]["tokens"], config, protocol["eval_tokens"]),
                "eval_tokens_used differs")
    plan = train_plan(stage, protocol)
    _require([(r["config"], r["seed"], r["order_in_block"], r["optimizer"]) for r in rows] == plan,
             "runs differ from the seeded (config, seed, order_in_block, optimizer) plan")
    fingerprints: dict = {}
    for i, r in enumerate(rows):
        where = f"runs[{i}] {r['config']} {r['optimizer']} seed {r['seed']}"
        config = configs[r["config"]]
        _require(r["run_config_hash"] == payload["run_config_hash"], f"{where}: run_config_hash differs")
        _require(r["diverged"] is False, f"{where}: diverged (0 divergences required)")
        _require((r["lr"], r["auxiliary_lr"], r["update_dtype"]) == (
            protocol["muon_lr"], protocol["muon_aux_lr"], protocol["update_dtype"]), f"{where}: optimizer settings")
        durations = r["step_wall_seconds"]
        _require(r["steps"] == steps == r["target_steps"] == len(durations), f"{where}: step count differs")
        _require(r["tokens"] == steps * config.step_tokens, f"{where}: tokens != steps * step_tokens")
        elapsed, partial = 0.0, {}
        for index, d in enumerate(durations, 1):
            _require(d > 0, f"{where}: non-positive step time")
            elapsed += d  # the trial accumulates in this order
            partial[index] = elapsed
        _assert_close(r["train_seconds"], elapsed, f"{where}.train_seconds")
        _assert_close(r["tokens_per_second"], r["tokens"] / r["train_seconds"], f"{where}.tokens_per_second")
        _assert_close(r["epoch_fraction"], r["tokens"] / train_tokens, f"{where}.epoch_fraction")
        _require(r["epoch_fraction"] < 1, f"{where}: the token budget must stay below one epoch")
        _require(all(len(v) == steps for v in r["step_ms"].values()), f"{where}: per-step segment count differs")
        if payload["meta"]["environment"].get("device") is not None:
            _require(set(r["step_ms"]) == set(SEGMENTS), f"{where}: CUDA-event segments are missing")
            _require(r["peak_vram_mib"] is not None and r["peak_vram_mib"]["allocated"] > 0, f"{where}: peak VRAM")
        _assert_close(r["timing"], json.loads(json.dumps(timing_from_raw(r))), f"{where}.timing")
        _assert_close(r["polar_fraction"], polar_fraction(r["timing"]), f"{where}.polar_fraction")
        _require([c["step"] for c in r["checkpoints"]] == checkpoints, f"{where}: checkpoints differ")
        for c in r["checkpoints"]:
            _require(c["tokens"] == c["step"] * config.step_tokens, f"{where}: checkpoint tokens")
            _assert_close(c["train_seconds"], partial[c["step"]], f"{where}.checkpoint.train_seconds")
            for split in ("val", "test"):
                _assert_close(c[f"{split}_ppl"], math.exp(c[f"{split}_loss"]), f"{where}.{split}_ppl")
        _require((r["val_loss"], r["test_loss"]) == (r["checkpoints"][-1]["val_loss"],
                                                     r["checkpoints"][-1]["test_loss"]), f"{where}: final losses")
        key = (r["config"], r["seed"])
        fp = fingerprints.setdefault(key, (r["init_fingerprint"], r["stream_fingerprint"]))
        _require(fp == (r["init_fingerprint"], r["stream_fingerprint"]), f"{where}: runs of one seed are not paired")
    for label in configs:
        mine = [v for (config, _), v in fingerprints.items() if config == label]
        _require(len({v[0] for v in mine}) == len(mine), f"{label}: seeds share an init fingerprint")
        _require(len({v[1] for v in mine}) == len(mine), f"{label}: seeds share a batch stream")
    _assert_close(payload["derived"], derive_train(payload), "derived")
    if canonical and stage == "ckpt":
        _require(payload["derived"]["max_epoch_fraction"] < 0.5, "the canonical ckpt budget must stay < 0.5 epoch")


def _validate_drift(payload: dict, canonical: bool) -> None:
    _validate_stage_common(payload, canonical)
    protocol, rows = payload["protocol"], payload["rows"]
    _require(payload["opus_l6"] == OPUS_L6, "opus_l6 differs from the recorded audit values")
    _require([r["id"] for r in rows] == [unit_id for unit_id, _ in drift_units(protocol)],
             "rows differ from the units of the protocol")
    names = [f"{kernel}_{dtype}" for kernel in protocol["kernels"] for dtype in protocol["dtypes"]]
    for r in rows:
        _require(r["run_config_hash"] == payload["run_config_hash"], f"{r['id']}: run_config_hash differs")
        if r["unit"] != "sweep":
            continue
        spec = protocol["families"][r["family"]]
        _require((r["n"], r["k"]) == (spec["n"], 1 << r["k_exponent"]) and r["shape"] == (
            [r["n"], r["k"]] if spec["orientation"] == "wide" else [r["k"], r["n"]]), f"{r['id']}: shape differs")
        _require(list(r["kernels"]) == names, f"{r['id']}: kernels differ from the protocol")
        _require(("gram" in r) == (r["seed"] in protocol["gram_seeds"]), f"{r['id']}: Gram decomposition")
        for name, cell in r["kernels"].items():
            for mode in ("production", "mitigated"):
                entry = cell[mode]
                _require(entry["finite"] is True, f"{r['id']} {name}.{mode}: non-finite")
                _require(entry["sigma_min"] >= 0.0 and entry["sigma_max"] >= entry["sigma_min"]
                         and entry["sigma_max"] > 0.0,
                         f"{r['id']} {name}.{mode}: invalid singular value bounds")
                _require(entry["rel_fro_err"] >= 0.0 and entry["ms"] > 0.0,
                         f"{r['id']} {name}.{mode}: invalid error or latency")
            _require(cell["mitigated"]["chunked"] == (r["k"] > protocol["gate_k"]),
                     f"{r['id']} {name}: the mitigation gate differs from K > gate_k")
    _assert_close(payload["derived"], derive_drift(payload), "derived")


def _source_path(recorded: dict) -> Path:
    return Path(recorded["path"]) if Path(recorded["path"]).is_absolute() else ROOT / recorded["path"]


def validate_results(payload: dict, *, canonical: bool | None = None, sources: dict | None = None) -> None:
    """Raise ``ValueError`` unless ``payload`` (a stage or the summary) is internally consistent.

    ``canonical`` (default: ``payload["canonical"]``) adds the canonical checks: locked protocol,
    clean git tree, pinned shard hashes and an sm_120+ device. For a summary, ``sources``
    ({"wide": ..., "ckpt": ..., "drift": ...}) or, when canonical, the files named in
    ``payload["sources"]`` are re-read, hash-checked, validated and re-derived. The recorded
    provenance is compared with the sources also when they are passed directly.
    """
    _require(payload.get("schema_version") == SCHEMA_VERSION, "unknown schema_version")
    json.dumps(payload, allow_nan=False)  # strict JSON: raises on NaN or infinity
    canonical = payload.get("canonical") is True if canonical is None else canonical
    kind = payload.get("kind")
    if kind == "runs":
        _require(payload.get("stage") in TRAIN_STAGES, f"unknown stage {payload.get('stage')!r}")
        return _validate_runs(payload, canonical)
    if kind == "drift":
        _require(payload.get("stage") == "drift", "a drift payload has the stage 'drift'")
        return _validate_drift(payload, canonical)
    _require(kind == "summary", "unknown payload kind")
    _require(set(payload["sources"]) == set(RUN_STAGES), "the summary needs the wide, ckpt and drift stages")
    if canonical:
        _require(payload["quick"] is False and payload["canonical"] is True, "a canonical summary cannot be --quick")
        _validate_provenance(payload["meta"], payload["data"], canonical, "summary")
    if sources is None and canonical:
        sources = {stage: json.loads(_source_path(src).read_text()) if _source_path(src).exists() else None
                   for stage, src in payload["sources"].items()}
        _require(all(s is not None for s in sources.values()), "a canonical source file is missing")
    if sources is not None:
        _require(set(sources) == set(RUN_STAGES), "the summary needs the wide, ckpt and drift stages")
        for stage, src in sources.items():
            _require(src.get("stage") == stage, f"source {stage} is a {src.get('stage')!r} payload")
            validate_results(src, canonical=canonical)
            recorded = payload["sources"][stage]
            _require(_is_hex(recorded.get("sha256"), 64), f"sources.{stage}.sha256 must be 64-hex")
            if canonical:
                path = _source_path(recorded)
                _require(path.exists(), f"canonical source {path} is missing")
                raw = path.read_bytes()
                _require(hashlib.sha256(raw).hexdigest() == recorded["sha256"],
                         f"{path}: SHA-256 differs from the summary")
                _require(src == json.loads(raw),
                         f"sources.{stage} differs from the canonical file {path}")
            for key in ("git_sha", "git_dirty", "code_sha256", "run_config_hash", "quick", "canonical", "protocol"):
                _assert_close(recorded[key], {**src["meta"], **src}[key], f"sources.{stage}.{key}")
            for key in ("device", "capability"):
                _require(recorded.get(key) == src["meta"]["environment"].get(key), f"sources.{stage}.{key} differs")
        for stage in TRAIN_STAGES:
            _assert_close(payload[stage], _train_section(stage, sources[stage]), stage)
        _assert_close(payload["drift"], {"opus_l6": sources["drift"]["opus_l6"],
                                         "derived": sources["drift"]["derived"]}, "drift")
        _assert_close(payload["data"], sources["ckpt"]["data"], "data")
        if canonical:
            metas = [s["meta"] for s in sources.values()]
            _require(len({m["git_sha"] for m in metas}) == 1 and len({m["code_sha256"] for m in metas}) == 1,
                     "canonical stages must come from one commit")
            _require(payload["meta"]["git_sha"] == metas[0]["git_sha"],
                     "canonical summary git_sha differs from its stage sources")
    for stage, src in payload["sources"].items():
        _validate_provenance(src, None, canonical, f"sources.{stage}")
        if canonical:
            _require(src["canonical"] is True and src["quick"] is False, f"source {stage} is not canonical")
            _require(list(src["capability"] or [0, 0]) >= [12, 0], f"source {stage} needs an sm_120+ device")
            _require(src["protocol"] == {**src["protocol"], **CANONICAL[stage]},
                     f"source {stage} protocol differs from the canonical protocol")
    _assert_close(payload["statistics"]["t_quantile_0p975"], {"df2": T975[2], "df4": T975[4]}, "t quantiles")
    for stage in TRAIN_STAGES:
        section, protocol = payload[stage], payload["sources"][stage]["protocol"]
        for label, table in section["per_seed"].items():
            for name, rows in table.items():
                for r in rows:
                    where = f"{stage}.per_seed.{label}.{name} seed {r['seed']}"
                    _require(r["diverged"] is False, f"{where}: diverged")
                    _assert_close(r["tokens_per_second"], r["tokens"] / r["train_seconds"],
                                  f"{where}.tokens_per_second")
                    _require([c["step"] for c in r["checkpoints"]] == protocol["checkpoints"],
                             f"{where}: checkpoints differ")
        _assert_close(section["analysis"], analyze_train(stage, section["per_seed"], protocol["checkpoints"]),
                      f"{stage}.analysis")


# ======================================================================================
# CLI
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--optimizers", nargs="+", default=None, help="wide/ckpt: subset of the Muon orthogonalizers")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="wide/ckpt seeds; drift: seeds of every family")
    p.add_argument("--steps", type=int, default=None, help="wide/ckpt: optimizer steps (the last checkpoint)")
    p.add_argument("--no-verify", action="store_true", help="skip SHA-256 checks (synthetic shards)")
    p.add_argument("--quick", action="store_true", help="tiny models, short sweeps")
    p.add_argument("--resume", action="store_true", help="skip rows already in OUTPUT.jsonl")
    p.add_argument("--silent", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    for stage in RUN_STAGES:
        p.add_argument(f"--{stage}-input", type=Path, default=None, help=f"summary: {stage}-stage payload")
    return p


def _resolve_output(args) -> Path:
    stage = args.stage
    if args.output is None:
        if args.quick:
            return QUICK_OUTPUTS[stage]
        if args.canonical:
            return CANONICAL_OUTPUTS[stage]
        raise ValueError(f"A non-canonical run needs an explicit --output (not {CANONICAL_OUTPUTS[stage].name})")
    if reserved(args.output) and not (args.canonical and args.output.resolve().name == CANONICAL_OUTPUTS[stage].name):
        raise ValueError(f"Refusing to write the reserved {args.output.name} from a --quick, non-canonical or "
                         f"other-stage run; choose a path outside {RESERVED_PATTERN}")
    return args.output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = {stage: getattr(args, f"{stage}_input") for stage in RUN_STAGES}
    overrides = [k for k in ("data_dir", "device", "optimizers", "seeds", "steps") if getattr(args, k) is not None]
    try:
        if args.stage == "summary":
            if overrides or args.resume or args.no_verify:
                parser.error("--stage summary takes no run arguments")
            args.inputs = {stage: path or (QUICK_OUTPUTS if args.quick else CANONICAL_OUTPUTS)[stage]
                           for stage, path in inputs.items()}
            args.canonical = not args.quick  # confirmed against the sources before writing
            args.output = _resolve_output(args)
            return args
        if any(inputs.values()):
            parser.error("--wide-input/--ckpt-input/--drift-input belong to --stage summary")
        protocol = copy.deepcopy((QUICK if args.quick else CANONICAL)[args.stage])
        if args.device is not None:
            protocol["device"] = args.device
        elif args.quick:
            protocol["device"] = default_device()
        if args.stage == "drift":
            if args.optimizers or args.steps or args.no_verify or args.data_dir:
                parser.error("--stage drift takes no training arguments")
            if args.seeds is not None:
                for spec in protocol["families"].values():
                    spec["seeds"] = list(args.seeds)
        else:
            protocol["no_verify"] = bool(args.no_verify)
            if args.optimizers is not None:
                protocol["optimizers"] = list(args.optimizers)
            if args.seeds is not None:
                protocol["seeds"] = list(args.seeds)
            if args.steps is not None:
                if args.steps <= 0:
                    parser.error("--steps must be positive")
                protocol["checkpoints"] = [c for c in protocol["checkpoints"] if c < args.steps] + [args.steps]
                protocol["steps"] = args.steps
            unknown = set(protocol["optimizers"]) - set(MUON)
            if unknown or len(set(protocol["optimizers"])) != len(protocol["optimizers"]):
                parser.error(f"unknown or repeated optimizers {sorted(unknown)}")
            if args.data_dir is None:
                args.data_dir = q.default_data_dir()
        if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
            parser.error("repeated seeds")
        args.protocol = protocol
        args.canonical = not args.quick and protocol == CANONICAL[args.stage]
        args.output = _resolve_output(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def run_summary(args) -> dict:
    sources = {stage: json.loads(Path(path).read_text()) for stage, path in args.inputs.items()}
    payload = build_summary(sources, args.inputs, quick=args.quick)
    if reserved(args.output) and not payload["canonical"]:
        raise ValueError(f"{args.output.name} needs canonical wide, ckpt and drift sources")
    validate_results(payload, canonical=payload["canonical"], sources=sources)
    return payload


def format_tables(summary: dict) -> str:
    """Markdown tables: per-optimizer means, every predeclared contrast, and the drift headline."""
    lines = []
    for stage in TRAIN_STAGES:
        section = summary[stage]
        lines += [f"\n{stage}: per-optimizer means", "| config | optimizer | tok/s | step ms | fwd+bwd ms | polar ms | "
                  "cast ms | foreach ms | polar fraction |", "|---|---|---|---|---|---|---|---|---|"]
        for label, table in section["analysis"]["per_optimizer"].items():
            for name, s in table.items():
                def show(key, fmt):
                    return format(s[key]["mean"], fmt) if key in s else "-"
                lines.append(f"| {label} | {name} | {show('tokens_per_second', ',.0f')} | "
                             f"{show('step_wall_ms', '.2f')} | {show('fwd_bwd_ms', '.2f')} | "
                             f"{show('polar_ortho_ms', '.3f')} | "
                             f"{show('layout_cast_ms', '.4f')} | {show('foreach_update_ms', '.3f')} | "
                             f"{show('polar_fraction', '.4f')} |")
        lines += [f"\n{stage}: paired contrasts",
                  "| config | contrast | metric | step | per-seed | mean | 95% CI | p | Holm family | Holm joint |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for c in section["analysis"]["contrasts"]:
            s = c["stats"]
            lines.append(f"| {c['config']} | {c['optimizer']} vs {c['reference']} | {c['metric']} {c['kind']} | "
                         f"{c['checkpoint'] or ''} | {' '.join(format(v, '+.5f') for v in s['values'])} | "
                         f"{s['mean']:.6g} | [{s['ci95'][0]:.6g}, {s['ci95'][1]:.6g}] | {s['p_two_sided']:.4g} | "
                         f"{c['p_holm_family']:.4g} | {c['p_holm_joint']:.4g} |")
    drift = summary["drift"]["derived"]
    lines += ["\ndrift: onset of sigma_max > the circuit's design ceiling (first grid K above, interpolated K)",
              "| family | kernel | production | mitigated | sigma_max at largest K (production -> mitigated) |",
              "|---|---|---|---|---|"]
    for family, table in drift["onset"].items():
        for name, cell in table.items():
            p, m = cell["production"]["design"], cell["mitigated"]["design"]
            lines.append(f"| {family} | {name} | {p['first_k_above']} ({p['interpolated_k'] or 0:.3g}) | "
                         f"{m['first_k_above']} | {p['sigma_max_at_largest_k'] or math.nan:.4f} -> "
                         f"{m['sigma_max_at_largest_k'] or math.nan:.4f} |")
    lines.append(f"\ndrift checks: {dumps(drift['checks'])}\nmitigation: {dumps(drift['mitigation'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.stage == "summary":
        payload = write_json_atomic(args.output, run_summary(args))
        if not args.silent:
            print(format_tables(payload))
    else:
        payload = write_json_atomic(args.output, (run_drift_stage if args.stage == "drift" else run_train_stage)(args))
        resume_log_path(args.output).unlink(missing_ok=True)
    if not args.silent:
        print(f"Wrote {args.output}")
    return payload


if __name__ == "__main__":
    main()
