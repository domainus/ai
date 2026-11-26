"""Cycle 4 C4-01a: long-budget ``120m`` FineWeb-Edu 3-seed Muon pretraining transfer (RTX 5080).

This driver wraps ``harness_tier2_real`` without changing it. It keeps the ``PRESETS["120m"]``
model (width 768, 12 layers, 12 heads, hidden 2048, sequence 512, micro-batch 8 x 8 accumulation,
32,768 tokens per step), ``--compile-model``, ``--update-dtype fp32``, ``BatchStream``,
``initial_state``, the wall-budget rollback (a step that ends after the deadline is undone, so
overshoot is 0.0 s), and ``--resume`` from ``OUTPUT.jsonl``. The resume log is deleted after the
final atomic write.

Stages (each one writes one strict JSON file, ``allow_nan=False``):

* ``tokens``: fixed token budget, 640 steps = 20,971,520 tokens (0.806 of the 26,004,518-token
  ``train.bin``), 50 warmup steps, ``adamw`` + four Muon orthogonalizers, seeds 1-3.
  -> ``results/cycle4_c4_01a_120m_tokens_21m.json``
* ``wall``: fixed accepted-compute budget of 300 s, warmup over 6.25% of it, four Muon
  orthogonalizers, seeds 1-3. -> ``results/cycle4_c4_01a_120m_wall_300s.json``
* ``summary``: predeclared paired analysis of both files. -> ``results/cycle4_c4_01a_120m_summary.json``

Statistics: for each seed, every optimizer starts from the same weights and sees the same batches.
Each contrast is a paired Student-t test of three per-seed values with df = 2. The two-sided
p-value is ``1 - |t| / sqrt(t^2 + 2)``, computed as ``2 / (s (s + |t|))`` with
``s = sqrt(t^2 + 2)`` to avoid cancellation. Holm-Bonferroni adjusts p within each family
(reference, budget, metric), within each reference across budgets and metrics, and jointly over
all 30 Muon-vs-Muon contrasts.

The canonical filenames ``cycle4_c4_01a_120m_*.json`` are reserved, in any checkout and through
symlinks: only the canonical arguments of that stage, on an sm_120+ CUDA device and a clean git
tree, may write them. ``validate_results`` recomputes every derived number (tolerance 1e-12).

Run (alone on the GPU, from a clean commit):
    PYTHONPATH=. .venv/bin/python research/polar_opt/bench_c4_01a_120m_long.py --stage tokens \\
        --data-dir artifacts/real_text
    PYTHONPATH=. .venv/bin/python research/polar_opt/bench_c4_01a_120m_long.py --stage wall \\
        --data-dir artifacts/real_text
    PYTHONPATH=. .venv/bin/python research/polar_opt/bench_c4_01a_120m_long.py --stage summary
Smoke test (tiny model; Muon kernels fall back to eager Polar Express on CPU):
    PYTHONPATH=. .venv/bin/python research/polar_opt/bench_c4_01a_120m_long.py --stage tokens --quick \\
        --output /tmp/c4_01a_tokens.json
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import median

import torch

from research.polar_opt import harness_tier2_real as h
from research.quant_ptq import micro_rotated_ptq as q

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SCHEMA_VERSION = "cycle4_c4_01a.v1"
TASK = "Cycle 4 C4-01a: long-budget 120m FineWeb-Edu 3-seed Muon pretraining transfer on RTX 5080 sm_120"
STAGES = ("tokens", "wall", "summary")
CANONICAL_OUTPUTS = {"tokens": RESULTS / "cycle4_c4_01a_120m_tokens_21m.json",
                     "wall": RESULTS / "cycle4_c4_01a_120m_wall_300s.json",
                     "summary": RESULTS / "cycle4_c4_01a_120m_summary.json"}
QUICK_OUTPUTS = {stage: RESULTS / f"cycle4_c4_01a_quick_{stage}.json" for stage in STAGES}
RESERVED_PATTERN = "cycle4_c4_01a_120m_*.json"
TOL = 1e-12
T975_DF2 = 0.95 / math.sqrt(2 * 0.975 * 0.025)  # df = 2 quantile: (2p - 1) / sqrt(2 p (1 - p))

MUON = ("muon_pe_compiled", "muon_ps3_blackwell", "muon_ss3_54", "muon_ss3_53")
SS3_FACTORIES = {"muon_ss3_54": "ss3_54_blackwell_fp16", "muon_ss3_53": "ss3_53_blackwell_fp16"}
# Predeclared contrasts: (reference, candidates, budgets). The candidate is the numerator.
MUON_PLAN = (("muon_pe_compiled", ("muon_ss3_54", "muon_ss3_53", "muon_ps3_blackwell"), ("tokens", "wall")),
             ("muon_ps3_blackwell", ("muon_ss3_54", "muon_ss3_53"), ("tokens", "wall")))
ADAMW_PLAN = ("adamw", MUON, ("tokens",))
METRICS = (("tokens_per_second", "ratio"), ("tokens_per_second", "diff"), ("test_loss", "diff"))

_COMMON = {"preset": "120m", "device": "cuda", "seeds": [1, 2, 3], "adamw_lrs": [0.001], "muon_lrs": [0.02],
           "muon_aux_lr": 0.001, "weight_decay": 0.01, "update_dtype": "fp32", "compile_model": True,
           "compile_warmup_steps": 3, "lr_warmup_steps": 50, "min_lr_ratio": 0.1, "eval_tokens": 2_000_000,
           "no_verify": False, "quick": False}
CANONICAL = {
    "tokens": {**_COMMON, "optimizers": ["adamw", *MUON], "tokens": 20_971_520, "seconds": None,
               "wall_warmup_fraction": None},
    "wall": {**_COMMON, "optimizers": list(MUON), "tokens": None, "seconds": 300.0, "wall_warmup_fraction": 0.0625},
}
CANONICAL_STEPS = {"tokens": 640}
_QUICK = {"preset": "tiny", "compile_model": False, "compile_warmup_steps": 1, "lr_warmup_steps": 2,
          "eval_tokens": 2048, "quick": True}
QUICK = {"tokens": {**CANONICAL["tokens"], **_QUICK, "tokens": 8 * h.PRESETS["tiny"].step_tokens},
         "wall": {**CANONICAL["wall"], **_QUICK, "seconds": 0.5}}
TRAIN_KEYS = tuple(k for k in CANONICAL["tokens"] if k not in ("quick",))


# ======================================================================================
# Small helpers
# ======================================================================================
def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def reserved(path: Path) -> bool:
    """True for a ``cycle4_c4_01a_120m_*.json`` name, in any checkout or through a symlink."""
    path = Path(path)
    return any(fnmatch.fnmatchcase(name, RESERVED_PATTERN) for name in (path.name, path.resolve().name))


def require_blackwell(device: torch.device, output: Path | None, *, canonical: bool = False) -> None:
    """A reserved filename (or the canonical arguments) promises an RTX 5080-class (sm_120+) result."""
    if (canonical or (output is not None and reserved(output))) and (
            device.type != "cuda" or not torch.cuda.is_available()
            or tuple(torch.cuda.get_device_capability(device)) < (12, 0)):
        raise RuntimeError(f"{Path(output).name if output else 'A canonical run'} requires a Blackwell "
                           "(sm_120+) CUDA device")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_sha256() -> str:
    """SHA-256 of the files that define this driver, training, data and every Muon kernel."""
    digest = hashlib.sha256()
    names = ("research.polar_opt.harness_tier2_real", "research.polar_opt.harness_tier2_lm",
             "research.polar_opt.real_text_data", "research.polar_opt.tier2_candidates",
             "research.polar_opt.baselines", "research.polar_opt.circuits",
             "research.polar_opt.kernels_blackwell", "research.polar_opt.polar_switch_blackwell",
             "research.polar_opt.schatten_switch_blackwell", "research.quant_ptq.micro_rotated_ptq")
    for module in (sys.modules[__name__], *(importlib.import_module(n) for n in names)):
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


def dumps(payload: dict) -> str:
    return json.dumps(payload, indent=1, allow_nan=False)


def write_json_atomic(path: Path, payload: dict) -> dict:
    """Strict JSON, read back and compared, fsynced, then renamed over the (resolved) target."""
    text = dumps(payload)
    reloaded = json.loads(text)
    if reloaded != json.loads(json.dumps(payload, allow_nan=False)):
        raise ValueError("JSON round trip is not lossless")
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    return reloaded


def resume_log_path(output: Path) -> Path:
    """The harness convention: ``OUTPUT.jsonl`` next to ``OUTPUT.json``."""
    return Path(output).resolve().with_suffix(".jsonl")


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


class StepClock:
    """``perf_counter`` that keeps every value it returns.

    ``run_trial`` calls the clock exactly twice per step (start, end) and computes
    ``clock() - start``, so consecutive pairs give its step times bit for bit, including a
    discarded post-deadline step.
    """

    def __init__(self):
        self.calls: list[float] = []

    def __call__(self) -> float:
        value = time.perf_counter()
        self.calls.append(value)
        return value

    def durations(self) -> list[float]:
        return [b - a for a, b in zip(self.calls[0::2], self.calls[1::2])]


# ======================================================================================
# Optimizers
# ======================================================================================
def factories(device: torch.device) -> tuple[dict, dict]:
    """Name -> zero-argument orthogonalizer factory (None for AdamW), and a description of each.

    On CPU (smoke tests only) every Muon variant uses eager Polar Express, because the Blackwell
    Triton kernels need CUDA. Canonical runs always use the real kernels.
    """
    from research.polar_opt import tier2_candidates as candidates
    from research.polar_opt.baselines import polar_express

    found = h.builtin_factories()
    for name, attribute in SS3_FACTORIES.items():
        found[name] = lambda attribute=attribute: getattr(candidates, attribute)
    found = {name: found[name] for name in ("adamw", *MUON)}
    described = {"adamw": "torch.optim.AdamW fused, betas (0.9, 0.95)",
                 "muon_pe_compiled": "torch.compile(baselines.polar_express), 5 steps",
                 "muon_ps3_blackwell": "tier2_candidates.ps3_blackwell_fp16 (Polar Switch PS3_55, FP16 Triton)",
                 "muon_ss3_54": "tier2_candidates.ss3_54_blackwell_fp16 (Schatten Switch SS3_54, FP16 Triton)",
                 "muon_ss3_53": "tier2_candidates.ss3_53_blackwell_fp16 (Schatten Switch SS3_53, FP16 Triton)"}
    if device.type != "cuda":
        for name in MUON:
            found[name] = lambda: polar_express
            described[name] = "CPU stand-in: baselines.polar_express eager (the real kernel needs CUDA)"
    return found, described


# ======================================================================================
# Training stages
# ======================================================================================
def set_numerics(device: torch.device) -> dict:
    """Full-precision FP32 matmuls: no TF32 and no reduced-precision FP16/BF16 reductions."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return {"allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_fp16_reduced_precision_reduction":
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "allow_bf16_reduced_precision_reduction":
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "autocast": "bfloat16 forward on CUDA (harness_tier2_real._loss)"}


def eval_tokens_used(split_tokens: int, config: h.ModelConfig, eval_tokens: int) -> int:
    """The token count ``harness_tier2_real.evaluate`` scores on one split."""
    seq, batch = config.sequence_length, config.micro_batch
    return min(eval_tokens, split_tokens - 1) // seq // batch * batch * seq


def build_protocol(args, config: h.ModelConfig, data: h.ShardData, described: dict) -> dict:
    splits = data.manifest["splits"]
    return {"budget": args.stage, "preset": args.preset, "optimizers": list(args.optimizers),
            "seeds": list(args.seeds), "adamw_lrs": list(args.adamw_lrs), "muon_lrs": list(args.muon_lrs),
            "muon_aux_lr": args.muon_aux_lr, "weight_decay": args.weight_decay, "update_dtype": args.update_dtype,
            "compile_model": args.compile_model, "compile_warmup_steps": args.compile_warmup_steps,
            "lr_warmup_steps": args.lr_warmup_steps, "wall_warmup_fraction": args.wall_warmup_fraction,
            "min_lr_ratio": args.min_lr_ratio, "tokens": args.tokens, "seconds": args.seconds,
            "eval_tokens": args.eval_tokens,
            "eval_tokens_used": {s: eval_tokens_used(splits[s]["tokens"], config, args.eval_tokens)
                                 for s in ("val", "test")},
            "target_steps": args.tokens // config.step_tokens if args.tokens is not None else None,
            "run_order": "per seed: random.Random(f'{seed}-{budget}').shuffle(optimizer x LR plan), as the harness",
            "schedule": ("linear warmup over lr_warmup_steps, then cosine to min_lr_ratio" if args.stage == "tokens"
                         else "linear warmup over wall_warmup_fraction of seconds, then cosine to min_lr_ratio"),
            "wall_budget": "model+optimizer CUDA-synchronized compute only; a post-deadline step is undone",
            "epoch_note": "BatchStream samples uniform random windows with replacement; epoch_fraction = "
                          "tokens / train.bin tokens",
            "orthogonalizers": {name: described[name] for name in args.optimizers}}


def run_config_hash(args, config: h.ModelConfig, data: h.ShardData, protocol: dict, git: dict, env: dict) -> str:
    """The harness hash plus this driver's code, git commit, kernels and device. ``--resume`` needs it equal."""
    harness_hash = h._config_hash(
        config, data, budget=args.stage, target_tokens=args.tokens, target_seconds=args.seconds,
        auxiliary_lr=args.muon_aux_lr, weight_decay=args.weight_decay, lr_warmup_steps=args.lr_warmup_steps,
        wall_warmup_fraction=args.wall_warmup_fraction if args.wall_warmup_fraction is not None else 0.0,
        min_lr_ratio=args.min_lr_ratio, compile_model=args.compile_model, eval_tokens=args.eval_tokens)
    payload = {"schema": SCHEMA_VERSION, "harness": harness_hash, "protocol": protocol, "quick": args.quick,
               "code_sha256": code_sha256(), "git_sha": git["git_sha"],
               "env": {k: env.get(k) for k in ("torch", "triton", "cuda", "device", "capability")}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_rows(log: Path, cfg_hash: str, resume: bool) -> list[dict]:
    """Rows of an interrupted run. A cut-off last line is dropped; any other mismatch raises."""
    if not resume:
        log.unlink(missing_ok=True)
        return []
    if not log.exists():
        return []
    lines = log.read_text().splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # interrupted mid-write
            raise
    for row in rows:
        if row.get("run_config_hash") != cfg_hash:
            raise ValueError(f"Cannot --resume from {log}: row hash {row.get('run_config_hash')!r} != {cfg_hash!r}")
    return rows


def _check_git(git: dict, canonical: bool, output: Path) -> None:
    if (canonical or reserved(output)) and git.get("git_dirty") is not False:
        raise RuntimeError("A canonical run needs a clean, committed git tree (git_dirty must be False)")


def run_stage(args) -> dict:
    """Train every (seed, optimizer, LR) of one budget and return the payload (not yet written)."""
    device = torch.device(args.device)
    require_blackwell(device, args.output, canonical=args.canonical)
    git = h.git_info()
    _check_git(git, args.canonical, args.output)
    log = (lambda *a: None) if args.silent else (lambda *a: print(*a, flush=True))
    started = time.perf_counter()
    numerics = set_numerics(device)
    data = q.load_data(args.data_dir, verify=not args.no_verify,
                       expected=None if args.no_verify else q.EXPECTED_SHA256)
    config = h.PRESETS[args.preset]
    found, described = factories(device)
    unknown = set(args.optimizers) - found.keys()
    if unknown:
        raise ValueError(f"Unknown optimizer(s): {sorted(unknown)}")
    env = q.environment(device)
    protocol = build_protocol(args, config, data, described)
    cfg_hash = run_config_hash(args, config, data, protocol, git, env)
    train_tokens = data.manifest["splits"]["train"]["tokens"]
    log_path = resume_log_path(args.output)
    rows = load_rows(log_path, cfg_hash, args.resume)
    loaded = len(rows)
    done = {(r["optimizer"], r["lr"], r["seed"]) for r in rows}
    log(f"[{args.stage}] git {git['git_sha']} dirty={git['git_dirty']} device={env.get('device', 'cpu')} "
        f"hash {cfg_hash} canonical={args.canonical} resumed rows {loaded}")
    meta = {**git, "command": [sys.executable, *sys.argv], "args": {k: str(v) for k, v in vars(args).items()},
            "environment": env, "numerics": numerics, "code_sha256": code_sha256(),
            "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "loadavg_start": list(os.getloadavg()),
            "gpu_telemetry": {"start": h.gpu_telemetry(device)}}

    reusable = None
    if args.compile_model:
        torch._dynamo.config.cache_size_limit = max(getattr(torch._dynamo.config, "cache_size_limit", 8), 32)
        with torch.device(device):
            shared = h.Transformer(config)
        reusable = (shared, torch.compile(shared))
    for seed in args.seeds:
        init = h.initial_state(config, seed)  # one seed's weights in memory, as the harness
        plan = [(name, lr) for name in args.optimizers
                for lr in (args.adamw_lrs if found[name] is None else args.muon_lrs)]
        random.Random(f"{seed}-{args.stage}").shuffle(plan)
        for order, (name, lr) in enumerate(plan):
            if (name, lr, seed) in done:
                continue
            clock = StepClock()
            row = h.run_trial(
                config, data, device=device, seed=seed, optimizer_name=name, factory=found[name], lr=lr,
                auxiliary_lr=args.muon_aux_lr, weight_decay=args.weight_decay, update_dtype=args.update_dtype,
                tokens=args.tokens, seconds=args.seconds, lr_warmup_steps=args.lr_warmup_steps,
                wall_warmup_fraction=args.wall_warmup_fraction or 0.0, min_lr_ratio=args.min_lr_ratio,
                compile_warmup_steps=args.compile_warmup_steps, eval_tokens=args.eval_tokens, init=init,
                compile_model=args.compile_model, reusable_model=reusable, clock=clock)
            durations = clock.durations()
            row.update({"run_config_hash": cfg_hash, "order_in_block": order,
                        "epoch_fraction": row["tokens"] / train_tokens,
                        "step_wall_seconds": durations[:row["steps"]]})
            row = json.loads(json.dumps(row, allow_nan=False))  # plain JSON types, as after a resume
            rows.append(row)
            done.add((name, lr, seed))
            with log_path.open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            t = row["timing"]
            log(f"[{args.stage}] seed {seed} {name:<20} lr {lr:<6g} steps {row['steps']:>4} "
                f"train {row['train_seconds']:.2f}s val {row['val_loss'] or math.nan:.4f} "
                f"test {row['test_loss'] or math.nan:.4f} tok/s {row['tokens_per_second'] or 0:,.0f} "
                f"epoch {row['epoch_fraction']:.3f} polar {t.get('polar_ortho_ms') or 0:.2f} ms")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    meta.update({"resume": {"resume": args.resume, "rows_loaded": loaded, "rows_run": len(rows) - loaded},
                 "loadavg_end": list(os.getloadavg()), "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "total_seconds": time.perf_counter() - started})
    meta["gpu_telemetry"]["end"] = h.gpu_telemetry(device)
    rows.sort(key=lambda r: (r["seed"], r["order_in_block"]))
    payload = {"schema_version": SCHEMA_VERSION, "kind": "runs", "stage": args.stage, "task": TASK,
               "quick": args.quick, "canonical": args.canonical, "meta": meta, "preset": args.preset,
               "config": asdict(config), "step_tokens": config.step_tokens,
               "parameters": h.parameter_counts(config),
               "data": {**data.summary(), "directory": str(args.data_dir), "sha256_verified": not args.no_verify,
                        "expected_sha256": None if args.no_verify else q.EXPECTED_SHA256,
                        "train_tokens": train_tokens},
               "protocol": protocol, "run_config_hash": cfg_hash, "runs": rows}
    payload["derived"] = derive_runs(payload)
    payload = json.loads(json.dumps(payload, allow_nan=False))
    validate_results(payload, canonical=args.canonical)
    return payload


def derive_runs(payload: dict) -> dict:
    """Per-optimizer means and the budget checks of one stage."""
    rows = payload["runs"]
    per = {}
    for name in payload["protocol"]["optimizers"]:
        mine = [r for r in rows if r["optimizer"] == name]
        ok = [r for r in mine if not r["diverged"]]
        per[name] = {"runs": len(mine), "diverged": len(mine) - len(ok),
                     **{f"{k}_mean": (math.fsum(r[k] for r in ok) / len(ok) if ok and len(ok) == len(mine) else None)
                        for k in ("steps", "tokens", "train_seconds", "tokens_per_second", "val_loss", "test_loss",
                                  "epoch_fraction")}}
    overshoot = [r["wall_budget_overshoot_seconds"] for r in rows if r["wall_budget_overshoot_seconds"] is not None]
    epochs = [r["epoch_fraction"] for r in rows]
    return {"per_optimizer": per, "runs": len(rows), "divergences": sum(r["diverged"] for r in rows),
            "max_wall_budget_overshoot_seconds": max(overshoot) if overshoot else None,
            "max_epoch_fraction": max(epochs) if epochs else None,
            "all_under_one_epoch": all(e < 1 for e in epochs)}


# ======================================================================================
# Statistics (pure functions of the per-seed rows; ``validate_results`` recomputes them)
# ======================================================================================
def mean_sd(values: list[float]) -> dict:
    n = len(values)
    mean = math.fsum(values) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {"values": list(values), "mean": mean, "sd": sd}


def paired_t_df2(values: list[float], null: float = 0.0) -> dict:
    """Exact paired t test of three per-seed values against ``null`` (df = 2).

    The df = 2 CDF is 1/2 + t / (2 sqrt(t^2 + 2)), so the two-sided p is 1 - |t| / sqrt(t^2 + 2),
    computed as 2 / (s (s + |t|)) with s = sqrt(t^2 + 2). With sd = 0, p is 1 (mean == null) or 0.
    """
    if len(values) != 3:
        raise ValueError("The exact df = 2 formulas need exactly three paired values")
    stats = mean_sd(values)
    mean, sd = stats["mean"], stats["sd"]
    se = sd / math.sqrt(3)
    shifted = mean_sd([v - null for v in values])  # t from v - null: no cancellation for ratios near 1
    if sd > 0:
        t = shifted["mean"] / (shifted["sd"] / math.sqrt(3))
        s = math.sqrt(t * t + 2)
        p = 2 / (s * (s + abs(t)))
    else:
        t, p = None, (1.0 if shifted["mean"] == 0 else 0.0)
    return {**stats, "n": 3, "df": 2, "null": null, "se": se, "t": t,
            "ci95": [mean - T975_DF2 * se, mean + T975_DF2 * se], "p_two_sided": p}


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(pvalues)
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: (pvalues[j], j))):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


ROW_FIELDS = ("seed", "lr", "steps", "tokens", "epoch_fraction", "train_seconds", "tokens_per_second",
              "val_loss", "test_loss", "test_ppl", "diverged", "wall_budget_overshoot_seconds",
              "discarded_step_seconds", "init_fingerprint", "stream_fingerprint", "order_in_block")


def per_seed_table(run_payload: dict) -> dict:
    """optimizer -> per-seed rows (sorted by seed). Each optimizer must have one LR."""
    table = {}
    for name in run_payload["protocol"]["optimizers"]:
        rows = sorted((r for r in run_payload["runs"] if r["optimizer"] == name), key=lambda r: r["seed"])
        if len({r["lr"] for r in rows}) != 1:
            raise ValueError(f"{name}: the summary needs exactly one LR per optimizer")
        table[name] = [{k: r[k] for k in ROW_FIELDS} for r in rows]
    return table


def _contrast_plan() -> list[tuple]:
    plan = []
    for reference, candidates, budgets in (*MUON_PLAN, ADAMW_PLAN):
        for budget in budgets:
            for metric, kind in METRICS:
                for optimizer in candidates:
                    plan.append((reference, budget, metric, kind, optimizer))
    return plan


def analyze(per_seed: dict) -> dict:
    """Predeclared paired contrasts, Holm-adjusted three ways, plus per-optimizer means."""
    per_optimizer = {budget: {name: {k: mean_sd([r[k] for r in rows])
                                     for k in ("steps", "tokens", "epoch_fraction", "train_seconds",
                                               "tokens_per_second", "val_loss", "test_loss")}
                              for name, rows in table.items()}
                     for budget, table in per_seed.items()}
    contrasts = []
    for reference, budget, metric, kind, optimizer in _contrast_plan():
        mine, ref = per_seed[budget][optimizer], per_seed[budget][reference]
        seeds = [r["seed"] for r in mine]
        if seeds != [r["seed"] for r in ref] or len(seeds) != 3:
            raise ValueError(f"{budget} {optimizer} vs {reference}: seeds differ or are not three")
        for a, b in zip(mine, ref):
            if (a["init_fingerprint"], a["stream_fingerprint"]) != (b["init_fingerprint"], b["stream_fingerprint"]):
                raise ValueError(f"{budget} seed {a['seed']}: {optimizer} and {reference} are not paired")
        if kind == "ratio":
            values, null = [a[metric] / b[metric] for a, b in zip(mine, ref)], 1.0
        else:
            values, null = [a[metric] - b[metric] for a, b in zip(mine, ref)], 0.0
        contrasts.append({"id": f"{budget}:{metric}_{kind}:{optimizer}_vs_{reference}", "budget": budget,
                          "optimizer": optimizer, "reference": reference, "metric": metric, "kind": kind,
                          "seeds": seeds, "family": f"{reference}|{budget}|{metric}_{kind}",
                          "stats": paired_t_df2(values, null)})
    groups = {"p_holm_family": lambda c: c["family"],
              "p_holm_reference_joint": lambda c: c["reference"],
              "p_holm_muon_joint": lambda c: "muon" if c["reference"] != "adamw" else None}
    for field, key in groups.items():
        members: dict = {}
        for i, c in enumerate(contrasts):
            members.setdefault(key(c), []).append(i)
        for group, idx in members.items():
            adjusted = holm_bonferroni([contrasts[i]["stats"]["p_two_sided"] for i in idx])
            for i, p in zip(idx, adjusted):
                contrasts[i][field] = p if group is not None else None
    sizes = {field: {} for field in groups}
    for field, key in groups.items():
        for c in contrasts:
            if key(c) is not None:
                sizes[field][key(c)] = sizes[field].get(key(c), 0) + 1
    return {"per_optimizer": per_optimizer, "contrasts": contrasts, "holm_group_sizes": sizes}


def stage_checks(run_payload: dict) -> dict:
    rows = run_payload["runs"]
    return {"runs": len(rows), "divergences": sum(r["diverged"] for r in rows),
            "max_wall_budget_overshoot_seconds": run_payload["derived"]["max_wall_budget_overshoot_seconds"],
            "max_epoch_fraction": max(r["epoch_fraction"] for r in rows),
            "all_under_one_epoch": all(r["epoch_fraction"] < 1 for r in rows),
            "steps": {name: [r["steps"] for r in sorted((r for r in rows if r["optimizer"] == name),
                                                         key=lambda r: r["seed"])]
                      for name in run_payload["protocol"]["optimizers"]}}


def build_summary(sources: dict, paths: dict, *, quick: bool) -> dict:
    """``sources``: {"tokens": run payload, "wall": run payload}; ``paths``: where they were read."""
    for stage in ("tokens", "wall"):
        if sources[stage].get("kind") != "runs" or sources[stage].get("stage") != stage:
            raise ValueError(f"{paths[stage]} is not a {stage} run payload")
    canonical = all(s["canonical"] and not s["quick"] for s in sources.values())
    per_seed = {stage: per_seed_table(sources[stage]) for stage in ("tokens", "wall")}
    tokens = sources["tokens"]
    payload = {
        "schema_version": SCHEMA_VERSION, "kind": "summary", "task": TASK, "quick": quick, "canonical": canonical,
        "meta": {**h.git_info(), "command": [sys.executable, *sys.argv],
                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        "sources": {stage: {"path": _relative(paths[stage]), "sha256": sha256_file(paths[stage]),
                            "git_sha": s["meta"]["git_sha"], "git_dirty": s["meta"]["git_dirty"],
                            "code_sha256": s["meta"]["code_sha256"], "run_config_hash": s["run_config_hash"],
                            "quick": s["quick"], "canonical": s["canonical"],
                            "device": s["meta"]["environment"].get("device"),
                            "capability": s["meta"]["environment"].get("capability"),
                            "protocol": s["protocol"]}
                    for stage, s in sources.items()},
        "preset": tokens["preset"], "config": tokens["config"], "step_tokens": tokens["step_tokens"],
        "parameters": tokens["parameters"], "data": tokens["data"],
        "statistics": {"test": "paired Student-t over seeds, df = 2 (three seeds)",
                       "t_quantile_0p975_df2": T975_DF2,
                       "p_two_sided": "1 - |t|/sqrt(t^2+2), computed as 2/(s(s+|t|)), s = sqrt(t^2+2)",
                       "ratio_null": 1.0, "diff_null": 0.0,
                       "direction": "diff = optimizer - reference; ratio = optimizer / reference; "
                                    "negative test_loss diff and ratio > 1 favour the optimizer",
                       "holm": {"p_holm_family": "within (reference, budget, metric)",
                                "p_holm_reference_joint": "within one reference, across budgets and metrics",
                                "p_holm_muon_joint": "all Muon-vs-Muon contrasts (pe_compiled and ps3_blackwell "
                                                     "references); null for the AdamW contrasts"},
                       "plan": {"muon": [[p[0], list(p[1]), list(p[2])] for p in MUON_PLAN],
                                "adamw": [ADAMW_PLAN[0], list(ADAMW_PLAN[1]), list(ADAMW_PLAN[2])],
                                "metrics": [f"{m}_{k}" for m, k in METRICS]}},
        "per_seed": per_seed,
        "checks": {stage: stage_checks(sources[stage]) for stage in ("tokens", "wall")},
    }
    payload["analysis"] = analyze(per_seed)
    return json.loads(json.dumps(payload, allow_nan=False))


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
        if isinstance(found, bool) != isinstance(expected, bool) or found != expected:
            raise ValueError(f"{path}: {found!r} differs from the recomputed {expected!r}")
    else:
        if isinstance(found, bool) or not isinstance(found, (int, float)) or \
                not abs(found - expected) <= tol * max(1.0, abs(expected)):
            raise ValueError(f"{path}: {found!r} differs from the recomputed {expected!r}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_hex(value, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _validate_provenance(meta: dict, data: dict, canonical: bool, where: str) -> None:
    _require(isinstance(meta.get("git_dirty"), bool), f"{where}: git_dirty must be recorded")
    if canonical:
        _require(_is_hex(meta.get("git_sha"), 40), f"{where}: canonical results need a 40-hex git_sha")
        _require(meta["git_dirty"] is False, f"{where}: canonical results must come from a clean git tree")
        _require({k: v["sha256"] for k, v in data["splits"].items()} == q.EXPECTED_SHA256,
                 f"{where}: shard SHA-256 values differ from the pinned q.EXPECTED_SHA256")
        _require(data["sha256_verified"] is True and data["expected_sha256"] == q.EXPECTED_SHA256,
                 f"{where}: shards must be verified against the pins")


def _validate_runs(payload: dict, canonical: bool) -> None:
    stage, protocol, meta, rows = payload["stage"], payload["protocol"], payload["meta"], payload["runs"]
    _require(stage in ("tokens", "wall"), f"unknown stage {stage!r}")
    _validate_provenance(meta, payload["data"], canonical, stage)
    config = h.ModelConfig(**payload["config"])
    _require(payload["config"] == asdict(h.PRESETS[payload["preset"]]), "config differs from the preset")
    _require(payload["step_tokens"] == config.step_tokens, "step_tokens differs from the config")
    _require(payload["parameters"] == h.parameter_counts(config), "parameter counts differ from the config")
    train_tokens = payload["data"]["splits"]["train"]["tokens"]
    _require(payload["data"]["train_tokens"] == train_tokens, "train_tokens differs from the manifest")
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
        locked = CANONICAL[stage]
        for key in ("preset", "optimizers", "seeds", "muon_lrs", "muon_aux_lr", "weight_decay", "update_dtype",
                    "compile_model", "compile_warmup_steps", "lr_warmup_steps", "wall_warmup_fraction",
                    "min_lr_ratio", "tokens", "seconds", "eval_tokens"):
            _require(protocol[key] == locked[key], f"protocol {key} differs from the canonical {stage} protocol")
        if "adamw" in locked["optimizers"]:
            _require(protocol["adamw_lrs"] == locked["adamw_lrs"], "protocol adamw_lrs differs")
    for split in ("val", "test"):
        _require(protocol["eval_tokens_used"][split] == eval_tokens_used(
            payload["data"]["splits"][split]["tokens"], config, protocol["eval_tokens"]), "eval_tokens_used differs")
    _require((protocol["tokens"] is None) == (stage == "wall") and (protocol["seconds"] is None) == (stage == "tokens"),
             "a stage has exactly one budget")
    target_steps = protocol["tokens"] // config.step_tokens if stage == "tokens" else None
    _require(protocol["target_steps"] == target_steps, "target_steps differs")
    if canonical and stage in CANONICAL_STEPS:
        _require(target_steps == CANONICAL_STEPS[stage], "canonical token budget must be 640 steps")

    # One row per (optimizer, LR, seed) of the plan, with the harness invariants.
    lrs = {name: (protocol["adamw_lrs"] if name == "adamw" else protocol["muon_lrs"]) for name in protocol["optimizers"]}
    expected = sorted((n, lr, s) for n in protocol["optimizers"] for lr in lrs[n] for s in protocol["seeds"])
    _require(sorted((r["optimizer"], r["lr"], r["seed"]) for r in rows) == expected,
             "runs differ from the (optimizer, LR, seed) plan")
    fingerprints: dict = {}
    for i, r in enumerate(rows):
        where = f"runs[{i}] {r['optimizer']} seed {r['seed']}"
        _require(r["run_config_hash"] == payload["run_config_hash"], f"{where}: run_config_hash differs")
        _require(r["budget"] == stage, f"{where}: budget differs")
        _require(r["diverged"] is False, f"{where}: diverged (0 divergences required)")
        _require(r["update_dtype"] == (None if r["optimizer"] == "adamw" else protocol["update_dtype"]),
                 f"{where}: update_dtype differs")
        _require(r["auxiliary_lr"] == (None if r["optimizer"] == "adamw" else protocol["muon_aux_lr"]),
                 f"{where}: auxiliary_lr differs")
        steps, durations = r["steps"], r["step_wall_seconds"]
        _require(isinstance(steps, int) and steps > 0 and len(durations) == steps, f"{where}: step count differs")
        _require(r["tokens"] == steps * config.step_tokens, f"{where}: tokens != steps * step_tokens")
        elapsed = 0.0
        for d in durations:
            _require(d > 0, f"{where}: non-positive step time")
            elapsed += d  # the harness accumulates in this order
        _assert_close(r["train_seconds"], elapsed, f"{where}.train_seconds")
        _assert_close(r["tokens_per_second"], r["tokens"] / r["train_seconds"], f"{where}.tokens_per_second")
        _assert_close(r["epoch_fraction"], r["tokens"] / train_tokens, f"{where}.epoch_fraction")
        _assert_close(r["val_ppl"], math.exp(r["val_loss"]), f"{where}.val_ppl")
        _assert_close(r["test_ppl"], math.exp(r["test_loss"]), f"{where}.test_ppl")
        timed = durations[1:] if len(durations) > 2 else durations
        _require(r["timing"]["steps_timed"] == len(timed), f"{where}: steps_timed differs")
        _assert_close(r["timing"]["step_wall_ms"], median(d * 1e3 for d in timed), f"{where}.timing.step_wall_ms")
        if stage == "tokens":
            _require(steps == target_steps and r["target_tokens"] == protocol["tokens"]
                     and r["target_seconds"] is None and r["wall_budget_overshoot_seconds"] is None
                     and r["discarded_step_seconds"] is None, f"{where}: token-budget invariants")
            _require(r["epoch_fraction"] < 1 and r["tokens"] < train_tokens, f"{where}: token budget must stay < 1 epoch")
        else:
            seconds = protocol["seconds"]
            _require(r["target_seconds"] == seconds and r["target_tokens"] is None, f"{where}: wall-budget targets")
            _require(r["wall_budget_overshoot_seconds"] == 0.0, f"{where}: wall overshoot must be 0.0 s")
            _require(r["train_seconds"] <= seconds, f"{where}: accepted compute exceeds the wall budget")
            d = r["discarded_step_seconds"]
            _require(d is None or (d > 0 and r["train_seconds"] + d > seconds),
                     f"{where}: a discarded step must have crossed the deadline")
            _require(d is not None or r["train_seconds"] >= seconds, f"{where}: the run stopped before the deadline")
        key = r["seed"]
        fp = fingerprints.setdefault(key, (r["init_fingerprint"], r["stream_fingerprint"]))
        _require(fp == (r["init_fingerprint"], r["stream_fingerprint"]), f"{where}: runs of one seed are not paired")
    _require(len({v[1] for v in fingerprints.values()}) == len(fingerprints), "seeds share a batch stream")
    _assert_close(payload["derived"], derive_runs(payload), "derived")
    if canonical:
        _require(payload["derived"]["divergences"] == 0, "canonical runs need 0 divergences")


def validate_results(payload: dict, *, canonical: bool | None = None, sources: dict | None = None) -> None:
    """Raise ``ValueError`` unless ``payload`` (a stage or the summary) is internally consistent.

    ``canonical`` (default: ``payload["canonical"]``) adds the canonical checks: locked protocol,
    clean git tree, pinned shard hashes and an sm_120+ device. For a summary, ``sources``
    ({"tokens": payload, "wall": payload}) or, when canonical, the files named in
    ``payload["sources"]`` are re-read, hash-checked, validated and re-derived.
    """
    _require(payload.get("schema_version") == SCHEMA_VERSION, "unknown schema_version")
    json.dumps(payload, allow_nan=False)  # strict JSON: raises on NaN or infinity
    canonical = payload.get("canonical") is True if canonical is None else canonical
    if payload.get("kind") == "runs":
        _validate_runs(payload, canonical)
        return
    _require(payload.get("kind") == "summary", "unknown payload kind")
    if canonical:
        _require(payload["quick"] is False and payload["canonical"] is True, "a canonical summary cannot be --quick")
        _require(payload["meta"]["git_dirty"] is False, "the canonical summary must come from a clean git tree")
    if sources is None and canonical:
        sources = {}
        for stage, src in payload["sources"].items():
            path = Path(src["path"]) if Path(src["path"]).is_absolute() else ROOT / src["path"]
            _require(path.exists(), f"canonical source {path} is missing")
            _require(sha256_file(path) == src["sha256"], f"{path}: SHA-256 differs from the summary")
            sources[stage] = json.loads(path.read_text())
    if sources is not None:
        _require(set(sources) == {"tokens", "wall"}, "the summary needs the tokens and wall stages")
        for stage, src in sources.items():
            validate_results(src, canonical=canonical)
            recorded = payload["sources"][stage]
            for key in ("git_sha", "git_dirty", "code_sha256", "run_config_hash", "quick", "canonical", "protocol"):
                _assert_close(recorded[key], {**src["meta"], **src}[key], f"sources.{stage}.{key}")
        _assert_close(payload["per_seed"], {s: per_seed_table(sources[s]) for s in ("tokens", "wall")}, "per_seed")
        _assert_close(payload["checks"], {s: stage_checks(sources[s]) for s in ("tokens", "wall")}, "checks")
        if canonical:
            _require(sources["tokens"]["meta"]["git_sha"] == sources["wall"]["meta"]["git_sha"]
                     and sources["tokens"]["meta"]["code_sha256"] == sources["wall"]["meta"]["code_sha256"],
                     "canonical stages must come from one commit")
    for stage, src in payload["sources"].items():
        _validate_provenance(src, payload["data"], canonical, f"sources.{stage}")
    if canonical:
        for stage, src in payload["sources"].items():
            _require(src["canonical"] is True and src["quick"] is False, f"source {stage} is not canonical")
            _require(list(src["capability"] or [0, 0]) >= [12, 0], f"source {stage} needs an sm_120+ device")
            _require(src["git_dirty"] is False, f"source {stage} must come from a clean tree")
    _assert_close(payload["statistics"]["t_quantile_0p975_df2"], T975_DF2, "statistics.t_quantile_0p975_df2")
    step_tokens, train_tokens = payload["step_tokens"], payload["data"]["splits"]["train"]["tokens"]
    for stage, table in payload["per_seed"].items():
        for name, rows in table.items():
            for r in rows:
                where = f"per_seed.{stage}.{name} seed {r['seed']}"
                _require(r["diverged"] is False, f"{where}: diverged")
                _require(r["tokens"] == r["steps"] * step_tokens, f"{where}: tokens != steps * step_tokens")
                _assert_close(r["tokens_per_second"], r["tokens"] / r["train_seconds"], f"{where}.tokens_per_second")
                _assert_close(r["epoch_fraction"], r["tokens"] / train_tokens, f"{where}.epoch_fraction")
                _assert_close(r["test_ppl"], math.exp(r["test_loss"]), f"{where}.test_ppl")
                if stage == "wall":
                    _require(r["wall_budget_overshoot_seconds"] == 0.0, f"{where}: wall overshoot must be 0.0 s")
                elif canonical:
                    _require(r["steps"] == CANONICAL_STEPS["tokens"] and r["epoch_fraction"] < 1,
                             f"{where}: canonical token budget is 640 steps, < 1 epoch")
    _assert_close(payload["analysis"], analyze(payload["per_seed"]), "analysis")
    _require(len(payload["analysis"]["contrasts"]) == len(_contrast_plan()), "contrast count differs")


# ======================================================================================
# CLI
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--preset", choices=sorted(h.PRESETS), default=None)
    p.add_argument("--optimizers", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--adamw-lrs", nargs="+", type=float, default=None)
    p.add_argument("--muon-lrs", nargs="+", type=float, default=None)
    p.add_argument("--muon-aux-lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--update-dtype", choices=["fp32", "native"], default=None)
    p.add_argument("--tokens", type=int, default=None, help="token budget (tokens stage)")
    p.add_argument("--seconds", type=float, default=None, help="accepted-compute wall budget (wall stage)")
    p.add_argument("--lr-warmup-steps", type=int, default=None)
    p.add_argument("--wall-warmup-fraction", type=float, default=None)
    p.add_argument("--min-lr-ratio", type=float, default=None)
    p.add_argument("--compile-warmup-steps", type=int, default=None)
    p.add_argument("--eval-tokens", type=int, default=None)
    p.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--no-verify", action="store_true", default=None, help="skip SHA-256 checks (synthetic shards)")
    p.add_argument("--quick", action="store_true", help="tiny model, short budgets")
    p.add_argument("--resume", action="store_true", help="skip runs already in OUTPUT.jsonl")
    p.add_argument("--silent", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--tokens-input", type=Path, default=None, help="summary: tokens-stage payload")
    p.add_argument("--wall-input", type=Path, default=None, help="summary: wall-stage payload")
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
    if args.stage == "summary":
        given = [k for k in TRAIN_KEYS if getattr(args, k) is not None]
        if given or args.resume or args.data_dir is not None:
            parser.error(f"--stage summary takes no training arguments ({', '.join(given) or 'resume/data-dir'})")
        if args.tokens_input is None:
            args.tokens_input = (QUICK_OUTPUTS if args.quick else CANONICAL_OUTPUTS)["tokens"]
        if args.wall_input is None:
            args.wall_input = (QUICK_OUTPUTS if args.quick else CANONICAL_OUTPUTS)["wall"]
        args.canonical = not args.quick  # confirmed against the sources before writing
        try:
            args.output = _resolve_output(args)
        except ValueError as exc:
            parser.error(str(exc))
        return args
    if args.tokens_input is not None or args.wall_input is not None:
        parser.error("--tokens-input/--wall-input belong to --stage summary")
    if args.stage == "tokens" and (args.seconds is not None or args.wall_warmup_fraction is not None):
        parser.error("--seconds/--wall-warmup-fraction belong to --stage wall")
    if args.stage == "wall" and args.tokens is not None:
        parser.error("--tokens belongs to --stage tokens")
    explicit = {k for k in TRAIN_KEYS if getattr(args, k) is not None}
    base = (QUICK if args.quick else CANONICAL)[args.stage]
    for key in TRAIN_KEYS:
        if getattr(args, key) is None:
            setattr(args, key, base[key])
    if "device" not in explicit and args.quick:
        args.device = default_device()
    args.no_verify = bool(args.no_verify)
    args.canonical = not args.quick and all(getattr(args, k) == v for k, v in CANONICAL[args.stage].items())
    if args.data_dir is None:
        args.data_dir = q.default_data_dir()
    config = h.PRESETS[args.preset]
    if args.stage == "tokens" and (args.tokens <= 0 or args.tokens % config.step_tokens):
        parser.error(f"--tokens must be a positive multiple of {config.step_tokens}")
    if args.stage == "wall" and not (math.isfinite(args.seconds) and args.seconds > 0):
        parser.error("--seconds must be positive and finite")
    if any(not (math.isfinite(v) and v > 0) for v in [*args.adamw_lrs, *args.muon_lrs, args.muon_aux_lr]):
        parser.error("learning rates must be positive and finite")
    unknown = set(args.optimizers) - {"adamw", *MUON}
    if unknown or len(set(args.seeds)) != len(args.seeds):
        parser.error(f"unknown optimizers {sorted(unknown)} or repeated seeds")
    try:
        args.output = _resolve_output(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def run_summary(args) -> dict:
    paths = {"tokens": args.tokens_input, "wall": args.wall_input}
    sources = {stage: json.loads(Path(path).read_text()) for stage, path in paths.items()}
    payload = build_summary(sources, paths, quick=args.quick)
    if reserved(args.output) and not payload["canonical"]:
        raise ValueError(f"{args.output.name} needs canonical tokens and wall sources")
    validate_results(payload, canonical=payload["canonical"], sources=sources)
    return payload


def format_tables(summary: dict) -> str:
    """Markdown tables of the per-optimizer means and every predeclared contrast."""
    lines = []
    for stage, table in summary["analysis"]["per_optimizer"].items():
        lines += [f"\n{stage} budget", "| optimizer | steps | tokens | epoch | tok/s | val loss | test loss |",
                  "|---|---|---|---|---|---|---|"]
        for name, s in table.items():
            lines.append(f"| {name} | {s['steps']['mean']:.2f} | {s['tokens']['mean']:,.0f} | "
                         f"{s['epoch_fraction']['mean']:.4f} | {s['tokens_per_second']['mean']:,.0f} | "
                         f"{s['val_loss']['mean']:.5f} | {s['test_loss']['mean']:.5f} |")
    lines += ["\n| budget | contrast | metric | mean | 95% CI | p | Holm family | Holm ref | Holm muon |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in summary["analysis"]["contrasts"]:
        s = c["stats"]
        muon = "" if c["p_holm_muon_joint"] is None else f"{c['p_holm_muon_joint']:.4g}"
        lines.append(f"| {c['budget']} | {c['optimizer']} vs {c['reference']} | {c['metric']} {c['kind']} | "
                     f"{s['mean']:.6g} | [{s['ci95'][0]:.6g}, {s['ci95'][1]:.6g}] | {s['p_two_sided']:.4g} | "
                     f"{c['p_holm_family']:.4g} | {c['p_holm_reference_joint']:.4g} | {muon} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.stage == "summary":
        payload = write_json_atomic(args.output, run_summary(args))
        if not args.silent:
            print(format_tables(payload))
    else:
        payload = write_json_atomic(args.output, run_stage(args))
        resume_log_path(args.output).unlink(missing_ok=True)
    if not args.silent:
        print(f"Wrote {args.output}")
    return payload


if __name__ == "__main__":
    main()
