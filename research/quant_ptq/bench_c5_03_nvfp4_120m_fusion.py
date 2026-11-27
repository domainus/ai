"""Cycle 5 C5-03: 120m real-checkpoint hardware NVFP4 W4A4 perplexity and small-shape quantizer fusion.

Part A  One linear layer, eight (M, N, K) shapes from decode to large prefill, including the C4-02
        negative caveat (1024, 512, 1408). Methods in one Williams-balanced session: BF16 cuBLAS,
        the pre-quantized NVFP4 GEMM ceiling, the exact pre-C5-03 C4-02 quantizer (embedded below,
        ``_c4_02_*``) and the C5-03 quantizer of ``nvfp4_inference_blackwell``, each for
        ``none``, ``had_b16``, ``bfly_b16`` and ``had_b16_static_gs``, eager and as CUDA graphs.
        Both quantizers feed the same ``scaled_mm``; their outputs are compared bit for bit. An
        ablation (CUDA graphs) at two shapes measures each C5-03 change and the rejected
        cooperative single-launch quantizer.
Part B  Full Transformer forward latency (50m, 120m, w2048_6l; random weights): BF16 eager,
        compiled and graphed, and C4-02 against C5-03 for the four variants, eager and graphed.
Part C  Three seeds of the 120m model trained with ``muon_ps3_blackwell`` for 640 steps
        (20,971,520 tokens). Seven PTQ methods, four evaluation modes, an FP32 emulation of the
        hardware W4A4 path, and paired-seed (df = 2) statistics with Holm-Bonferroni correction.

Split hygiene (Part C). ``Shards(verify=True)`` reads train.bin, val.bin and test.bin at startup to
check their SHA-256 values, token ranges and document boundaries; no model is scored there.
``Shards.open`` then maps train.bin and val.bin for Phase 1 (training, calibration, static scales,
AWQ alpha and method selection on the full val.bin) and maps test.bin only in Phase 2, after every
selection of every seed is frozen. ``meta.splits_opened`` and ``part_c.scoring_ledger`` record it.

Timing design: Williams-balanced orders (every method follows every other method equally often),
CUDA events around each call with a 4x L2 flush (268 MB on the RTX 5080) before every sample, the
raw per-call event times (``event_samples_ms``) and per-round means (``samples_ms``), and a host
wall clock that includes launch overhead. TF32 and reduced-precision reductions are off.

``cycle5_c5_03*.json`` is reserved: only the canonical protocol, on an sm_120+ device, from a clean
committed tree, in this checkout and without a symlink, may write it.

Schema ``cycle5_c5_03.v2`` adds the top-level ``postprocess`` field. It is ``None`` for a run that
measures. ``--postprocess-from-git <commit>`` reads the committed ``cycle5_c5_03.v1`` result of that
commit with ``git show``, keeps every raw measurement unchanged (``RAW_MEASUREMENT_KEYS``; ``part_c``
without its ``analysis``), recomputes only ``part_c.analysis``, ``summary`` and ``caveats``, and records
the provenance of the source and of the postprocessing code in ``postprocess``. It needs no GPU,
trains nothing and scores no split.

Run (canonical, alone on the GPU, from a clean commit; about one hour on an RTX 5080):
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_c5_03_nvfp4_120m_fusion.py \\
        --data-dir artifacts/real_text
Smoke run:
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_c5_03_nvfp4_120m_fusion.py --quick \\
        --output /tmp/c5_03.json
Postprocess the committed v1 measurements (clean tree, no GPU):
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_c5_03_nvfp4_120m_fusion.py \\
        --postprocess-from-git 726b20c9590aba911361e300e7d2079c7bb55cfa
"""

from __future__ import annotations

import argparse
import ast
import copy
import errno
import fnmatch
import functools
import hashlib
import io
import json
import math
import os
import secrets
import stat
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import triton
import triton.language as tl
from torch import Tensor, nn
from triton import knobs
from triton.runtime import driver

from research.polar_opt import harness_tier2_real as h
from research.polar_opt.harness_tier2_lm import Transformer
from research.polar_opt.polar_switch_blackwell import _c_launcher
from research.quant_ptq import bench_c5_02_isoloss_ptq_mitigation as c502
from research.quant_ptq import bench_isoloss_ptq as c4
from research.quant_ptq import bench_nvfp4_e2e as e2e
from research.quant_ptq import micro_rotated_ptq as q
from research.quant_ptq import nvfp4_inference_blackwell as nb
from research.quant_ptq.nvfp4_inference_blackwell import ROTATIONS, NVFP4Activation, scale_layout

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "cycle5_c5_03.v2"
SOURCE_SCHEMA_VERSION = "cycle5_c5_03.v1"  # the schema that ``--postprocess-from-git`` reads
TASK = ("Cycle 5 C5-03: 120m real-checkpoint hardware NVFP4 W4A4 perplexity and small-shape dynamic activation "
        "quantizer fusion on Blackwell")
SCRIPT_NAME = "bench_c5_03_nvfp4_120m_fusion.py"
RESULT_REPO_PATH = "results/cycle5_c5_03_nvfp4_120m_fusion.json"
CANONICAL_OUTPUT = ROOT / RESULT_REPO_PATH
DRIVER_MODULE = "research.quant_ptq.bench_c5_03_nvfp4_120m_fusion"
SCRIPT_REPO_PATH = "/".join(DRIVER_MODULE.split(".")) + ".py"
CANONICAL_PRODUCER_SHA = "38cef483071d649fc74161b5d00359c7c5ad0927"
# The modules that compute ``part_c.analysis``, ``summary``, ``caveats`` and the validator. The credited
# commit (``meta.git_sha`` of a measuring run, ``postprocess_git_sha`` of a postprocess) must hold these bytes.
SUMMARY_MODULES = (
    DRIVER_MODULE,
    "research.quant_ptq.nvfp4_inference_blackwell",
    "research.quant_ptq.bench_c5_02_isoloss_ptq_mitigation",
    "research.quant_ptq.bench_isoloss_ptq",
    "research.quant_ptq.bench_nvfp4_e2e",
    "research.quant_ptq.micro_rotated_ptq",
    "research.polar_opt.harness_tier2_real",
)
QUICK_OUTPUT = ROOT / "results" / "c5_03_quick.json"
CANONICAL_CHECKPOINT_DIR = ROOT / "artifacts" / "c5_03_checkpoints"
RESERVED_PATTERN = "cycle5_c5_03*.json"
LEGACY_GIT_SHA = "08fc1d082495fabd3946769abc81dbc0a14ba7f0"  # the C4-02 quantizer embedded below is from here
LEGACY_REPO_PATH = "research/quant_ptq/nvfp4_inference_blackwell.py"
TOL = 1e-12

# Raw measurements of a result (``part_c`` without ``analysis``). ``--postprocess-from-git`` copies them unchanged.
RAW_MEASUREMENT_KEYS = ("quick", "canonical", "meta", "run_config_hash", "protocol", "data", "kernel_parity",
                        "part_a", "part_b", "part_c")
DERIVED_PART_C_KEYS = ("analysis",)
POSTPROCESS_KEYS = ("source_schema_version", "source_result_git_sha", "source_artifact_sha256",
                    "source_raw_measurements_sha256", "postprocess_git_sha", "postprocess_git_dirty",
                    "postprocess_code_sha256", "postprocess_code_modules", "postprocess_command",
                    "postprocess_timestamp", "retained_evidence_sha256", "reason")
# The committed v1 result that the canonical postprocess reads: result commit -> SHA-256 of the file
# (726b20c holds the canonical run of the clean producer commit 38cef48).
CANONICAL_SOURCE_ARTIFACTS = {
    "726b20c9590aba911361e300e7d2079c7bb55cfa": "9ed6c9fd3879566f78e943f5aa54657b1f0e1c27e543e3e93f11103413da14ce"}
# Study history retained outside git (artifacts/c5_03_run_evidence/): the canonical log of 38cef48 and the
# log and result of the superseded run at 4736807, which scored the same test.bin first.
RETAINED_EVIDENCE_SHA256 = {
    "canonical_38cef48.log": "d6ebd7f655fe300df39764447a17fceee6262df52a7cd1087a8937fd4e210408",
    "superseded_canonical_4736807.log": "807781a3d86bd47e413075186cde26b55645d9dd07cb9d60b002875ac993724e",
    "superseded_result_4736807.json": "aafd8ebe571bb512ac02e5c20403ac900e930cd11e4a74c5c5152a3e3b2f9c5a",
}
POSTPROCESS_REASON = (
    "Remediation of the C5-03 audit and post-remediation re-checks (audits/AUDIT_cycle5_c5_03_gpt62_astra.md, "
    "findings 1-5 and v2-v3 closures): the raw measurements of the canonical result 726b20c (measured by the clean "
    "commit 38cef48) are preserved unchanged, so test.bin is not scored a third time. part_c.analysis, summary "
    "and caveats are recomputed (the caveats bind to the actual protocol, test-split SHA-256, analysis availability "
    "and source, recording the superseded 4736807 run and measurement limits), and the result is checked by the "
    "hardened validator (calibration draw, linear parameter counts, FP4-specific GEMM and exact quantizer kernel "
    "profiles, cooperative launch record, AWQ fields, exact script path, unified canonical path guard and commit "
    "chronology). The exact n_part public launcher contract, verified dir_fd atomic writer, and integer-view "
    "bitwise checks are verified by the test suite and audit replay. Nothing was retrained or re-measured.")

VARIANTS = ("none", "had_b16", "bfly_b16", "had_b16_static_gs")
VARIANT_ROTATION = {"none": "none", "had_b16": "hadamard_b16", "bfly_b16": "adaptive_butterfly_b16",
                    "had_b16_static_gs": "hadamard_b16"}
IMPLS = ("c4_02", "c5_03")
PART_A_METHODS = ("bf16_cublas", "nvfp4_gemm_only", *(f"{i}_{v}" for i in IMPLS for v in VARIANTS))
PART_A_LABELS = {(16, 4096, 4096): "small-batch decode", (128, 4096, 4096): "decode batch 128",
                 (1024, 512, 1408): "50m MLP down (C4-02 caveat)", (1024, 1408, 512): "50m MLP up",
                 (1024, 768, 2048): "120m MLP down", (1024, 2048, 768): "120m MLP up",
                 (1024, 4096, 4096): "prefill 4096", (2048, 8192, 8192): "large prefill"}
ABLATION_VARIANTS = ("none", "had_b16")
ABLATIONS = ("c5_03", "c5_03_no_pdl", "c5_03_ladder_cvt", "c5_03_coop_single_launch", "c4_02")
ABLATION_METHODS = ("bf16_cublas", *(f"{a}_{v}" for v in ABLATION_VARIANTS for a in ABLATIONS))
PART_B_METHODS = ("bf16_eager", "bf16_compiled", "bf16_eager_graph",
                  *(f"{i}_{v}_{m}" for v in VARIANTS for i in IMPLS for m in ("eager", "graph")))

BASE_METHODS = ("rtn", "gptq", "hadamard_b16_rtn", "hadamard_b16_gptq", "adaptive_butterfly_b16_gptq")
AWQ_BASE = {"awq_w4a4_hadamard_b16_gptq": "hadamard_b16_gptq",
            "awq_w4a4_adaptive_butterfly_b16_gptq": "adaptive_butterfly_b16_gptq"}
METHODS = (*BASE_METHODS, *AWQ_BASE)
MODES = ("w4a16_fakequant", "w4a4_hw_dynamic", "w4a4_hw_static_bf16cal", "w4a4_hw_static_seqcal")
HW_MODES = MODES[1:]
EMULATED_METHOD, EMULATED_MODE = "hadamard_b16_gptq", "w4a4_emulated_fp32_dynamic"
SPLITS = ("val", "test")
VAL_TOKENS, TEST_TOKENS = 2_001_920, 1_999_872
TRAIN_TOKENS = 26_004_518
N_SEEDS = 3
SIGN_FLIP_MIN_P = 2 / 2 ** N_SEEDS  # 0.25: the smallest two-sided sign-flip p-value of three seeds
CONTRASTS = (
    ("hadamard_vs_identity_gptq", ("hadamard_b16_gptq", "w4a4_hw_dynamic"), ("gptq", "w4a4_hw_dynamic")),
    ("butterfly_vs_hadamard_gptq", ("adaptive_butterfly_b16_gptq", "w4a4_hw_dynamic"),
     ("hadamard_b16_gptq", "w4a4_hw_dynamic")),
    ("gptq_vs_rtn", ("gptq", "w4a4_hw_dynamic"), ("rtn", "w4a4_hw_dynamic")),
    ("hadamard_gptq_vs_hadamard_rtn", ("hadamard_b16_gptq", "w4a4_hw_dynamic"),
     ("hadamard_b16_rtn", "w4a4_hw_dynamic")),
    ("awq_vs_plain_hadamard_gptq", ("awq_w4a4_hadamard_b16_gptq", "w4a4_hw_dynamic"),
     ("hadamard_b16_gptq", "w4a4_hw_dynamic")),
    ("awq_vs_plain_butterfly_gptq", ("awq_w4a4_adaptive_butterfly_b16_gptq", "w4a4_hw_dynamic"),
     ("adaptive_butterfly_b16_gptq", "w4a4_hw_dynamic")),
    ("static_seqcal_vs_static_bf16cal", ("hadamard_b16_gptq", "w4a4_hw_static_seqcal"),
     ("hadamard_b16_gptq", "w4a4_hw_static_bf16cal")),
    ("dynamic_vs_static_bf16cal", ("hadamard_b16_gptq", "w4a4_hw_dynamic"),
     ("hadamard_b16_gptq", "w4a4_hw_static_bf16cal")),
    ("w4a4_dynamic_vs_w4a16", ("hadamard_b16_gptq", "w4a4_hw_dynamic"), ("hadamard_b16_gptq", "w4a16_fakequant")),
)
STATIC_BF16CAL = "train.bin calibration windows through the BF16 model, max |(x / s) R^T| / 2688"
STATIC_SEQCAL = ("train.bin calibration windows; block i is calibrated with blocks 0..i-1 already W4A4 hardware "
                 "with their own sequential static scales (block i itself BF16), max |(x / s) R^T| / 2688")

PART_A_SHAPES = [[16, 4096, 4096], [128, 4096, 4096], [1024, 512, 1408], [1024, 1408, 512], [1024, 768, 2048],
                 [1024, 2048, 768], [1024, 4096, 4096], [2048, 8192, 8192]]
CANONICAL = {
    "device": "cuda", "parts": "ABC", "seed": 1,
    "timing": {"squares_a": 2, "squares_b": 1, "iters": 10, "l2_flush_multiple": 4, "wall_iters_a": 2,
               "wall_iters_b": 1},
    "part_a": {"shapes": PART_A_SHAPES, "ablation_shapes": [[1024, 512, 1408], [16, 4096, 4096]],
               "coop_replays": 1000},
    "part_b": {"models": {"50m": 16, "120m": 8, "w2048_6l": 4}},
    "part_c": {"preset": "120m", "seeds": [1, 2, 3], "optimizer": "muon_ps3_blackwell", "steps": 640,
               "warmup_steps": 64, "lr": 0.02, "auxiliary_lr": 0.001, "weight_decay": c4.WEIGHT_DECAY,
               "min_lr_ratio": c4.MIN_LR_RATIO, "calibration_sequences": 128, "calibration_seed": 20261126,
               "gptq_damp": c4.GPTQ_DAMP, "butterfly_steps": 150, "awq_alphas": [0.0, 0.25, 0.5, 0.75],
               "emulation_parity_nats": 1e-3, "max_eval_windows": None},
}
QUICK = {
    **CANONICAL,
    "timing": {"squares_a": 1, "squares_b": 1, "iters": 3, "l2_flush_multiple": 4, "wall_iters_a": 2,
               "wall_iters_b": 1},
    "part_a": {"shapes": [[16, 256, 256], [129, 512, 768]], "ablation_shapes": [[129, 512, 768]],
               "coop_replays": 20},
    "part_b": {"models": {"tiny": 4}},
    "part_c": {**CANONICAL["part_c"], "preset": "tiny", "steps": 8, "warmup_steps": 2, "calibration_sequences": 8,
               "butterfly_steps": 5, "max_eval_windows": 16},
}
PART_B_CONFIGS = {"tiny": h.PRESETS["tiny"], "50m": h.PRESETS["50m"], "120m": h.PRESETS["120m"],
                  "w2048_6l": e2e.PART_B_MODELS["w2048_6l"][0]}


# ======================================================================================
# The pre-C5-03 C4-02 quantizer, embedded verbatim from LEGACY_GIT_SHA (names prefixed ``_c4_02``).
# ``tests/test_c5_03_nvfp4_120m_fusion.py`` checks the source against the git blob.
# ======================================================================================
MAX_PARTIALS = 512  # the C4-02 constant that the embedded code uses
_c4_02_PLANS: dict = {}


def _c4_02_tile(M: int, rotation: str) -> tuple[int, int, int]:
    """(rows, 16-input groups, warps) per program. Rows must divide 128 (the scale-row padding).

    From a sweep on the RTX 5080 with a flushed L2 (M in 16..8192, K in 512..8192): narrow
    8-row tiles suit the add-only rotations; the butterfly, with its per-group angle loads,
    prefers 32 x 4 once M fills the GPU.
    """
    if rotation == "adaptive_butterfly_b16":
        return (8, 4, 4) if M < 128 else (32, 4, 4)
    return (8, 8, 4) if M < 128 else (8, 16, 4)


@triton.jit
def _c4_02_nanmax(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _c4_02_pairs(v, R: tl.constexpr, S: tl.constexpr):
    """(R, 16) -> the two halves (R, 8 // S, S) of every stage-S pair (i, i + S) in each block."""
    w = tl.permute(tl.reshape(v, (R, 8 // S, 2, S)), (0, 1, 3, 2))
    return tl.split(w)


@triton.jit
def _c4_02_unpairs(a, d, R: tl.constexpr, S: tl.constexpr):
    return tl.reshape(tl.permute(tl.join(a, d), (0, 1, 3, 2)), (R, 16))


@triton.jit
def _c4_02_butterfly_stage(v, cos_ptr, sin_ptr, groups, gmask, s: tl.constexpr, BM: tl.constexpr,
                     BG: tl.constexpr):
    """Stage ``s``: (a, d) -> (c a + s d, s a - c d) with angles (G, 4, 8) loaded for this tile."""
    R: tl.constexpr = BM * BG
    S: tl.constexpr = 1 << s
    offs = groups[:, None] * 32 + s * 8 + tl.arange(0, 8)[None, :]
    c = tl.load(cos_ptr + offs, mask=gmask[:, None], other=0.0)
    sn = tl.load(sin_ptr + offs, mask=gmask[:, None], other=0.0)
    c = tl.reshape(tl.broadcast_to(c[None, :, :], (BM, BG, 8)), (R, 8 // S, S))
    sn = tl.reshape(tl.broadcast_to(sn[None, :, :], (BM, BG, 8)), (R, 8 // S, S))
    a, d = _c4_02_pairs(v, R, S)
    return _c4_02_unpairs(c * a + sn * d, sn * a - c * d, R, S)


@triton.jit
def _c4_02_load_rotated(x_ptr, cos_ptr, sin_ptr, rows, g0, M, K, G, stride_m, stride_k,
                  ROT: tl.constexpr, BM: tl.constexpr, BG: tl.constexpr):
    """FP32 tile (BM * BG, 16) of rotated activations; out-of-range inputs are 0."""
    R: tl.constexpr = BM * BG
    cols = g0 * 16 + tl.arange(0, BG * 16)
    mask = (rows[:, None] < M) & (cols[None, :] < K)
    ptrs = x_ptr + rows[:, None].to(tl.int64) * stride_m + cols[None, :].to(tl.int64) * stride_k
    v = tl.reshape(tl.load(ptrs, mask=mask, other=0.0).to(tl.float32), (R, 16))
    if ROT == 1:  # orthonormal Walsh-Hadamard (Sylvester order), the add sequence of ``fwht``
        for s in tl.static_range(4):
            a, d = _c4_02_pairs(v, R, 1 << s)
            v = _c4_02_unpairs(a + d, a - d, R, 1 << s)
        v = v * 0.25
    elif ROT == 2:  # adaptive butterfly: four stages of 2x2 reflections, ``Butterfly.forward``
        groups = g0 + tl.arange(0, BG)
        gmask = groups < G
        for s in tl.static_range(4):
            v = _c4_02_butterfly_stage(v, cos_ptr, sin_ptr, groups, gmask, s, BM, BG)
    return v


@triton.jit
def _c4_02_amax_kernel(x_ptr, part_ptr, cos_ptr, sin_ptr, M, K, G, stride_m, stride_k, n_gt, n_tiles,
                 ROT: tl.constexpr, BM: tl.constexpr, BG: tl.constexpr):
    """Pass 1: partial[program] = NaN-propagating max |R x| over a strided set of tiles."""
    pid = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for t in range(pid, n_tiles, tl.num_programs(0)):
        rows = (t // n_gt) * BM + tl.arange(0, BM)
        v = _c4_02_load_rotated(x_ptr, cos_ptr, sin_ptr, rows, (t % n_gt) * BG, M, K, G, stride_m, stride_k,
                          ROT, BM, BG)
        acc = _c4_02_nanmax(acc, tl.reduce(tl.abs(v), None, _c4_02_nanmax))
    tl.store(part_ptr + pid, acc)


@triton.jit
def _c4_02_quant_kernel(x_ptr, q_ptr, s_ptr, gs_ptr, part_ptr, cos_ptr, sin_ptr, M, K, G, stride_m, stride_k,
                  pc4, n_part, ROT: tl.constexpr, DYNAMIC: tl.constexpr, BM: tl.constexpr, BG: tl.constexpr,
                  PBLOCK: tl.constexpr):
    """Pass 2 (or the only pass with a static scale): quantize, pack and swizzle one tile.

    The grid covers the padded scale matrix (ceil(M / 128) * 128 rows, ``4 * pc4`` columns).
    """
    pid_m, pid_g = tl.program_id(0), tl.program_id(1)
    if DYNAMIC:
        p = tl.arange(0, PBLOCK)
        amax = tl.reduce(tl.load(part_ptr + p, mask=p < n_part, other=0.0), None, _c4_02_nanmax)
        raw_gs = tl.div_rn(amax, 2688.0)  # amax / (6 * 448)
        gs = tl.where(raw_gs > 0, raw_gs, 1.0)  # 0, underflowing subnormal, or NaN -> 1
        if (pid_m == 0) & (pid_g == 0):
            tl.store(gs_ptr, gs)
    else:
        gs = tl.load(gs_ptr)
    rows = pid_m * BM + tl.arange(0, BM)
    groups = pid_g * BG + tl.arange(0, BG)
    v = _c4_02_load_rotated(x_ptr, cos_ptr, sin_ptr, rows, pid_g * BG, M, K, G, stride_m, stride_k, ROT, BM, BG)
    v = tl.reshape(v, (BM, BG, 16))

    # E4M3 block scale: clamp(amax / (6 gs), max=448) -> e4m3 (round to nearest even).
    bamax = tl.reduce(tl.abs(v), 2, _c4_02_nanmax)
    block = tl.minimum(tl.div_rn(bamax, 6.0 * gs), 448.0, propagate_nan=tl.PropagateNan.ALL)
    block8 = block.to(tl.float8e4nv)
    scale = block8.to(tl.float32) * gs
    safe = tl.where(scale > 0, scale, 1.0)
    y = tl.div_rn(v, safe[:, :, None])

    # E2M1 codes: the comparisons of ``e2m1_codes`` (ties to even), sign from the IEEE sign bit.
    a = tl.abs(y)
    code = ((a > 0.25).to(tl.uint8) + (a >= 0.75).to(tl.uint8) + (a > 1.25).to(tl.uint8)
            + (a >= 1.75).to(tl.uint8) + (a > 2.5).to(tl.uint8) + (a >= 3.5).to(tl.uint8)
            + (a > 5.0).to(tl.uint8))
    sign = ((y.to(tl.int32, bitcast=True) >> 31) & 1).to(tl.uint8)
    code = code | (sign << 3)
    lo, hi = tl.split(tl.reshape(code, (BM, BG, 8, 2)))
    packed = tl.reshape(lo | (hi << 4), (BM, BG * 8))
    pcols = pid_g * BG * 8 + tl.arange(0, BG * 8)
    half = K // 2
    tl.store(q_ptr + rows[:, None].to(tl.int64) * half + pcols[None, :], packed,
             mask=(rows[:, None] < M) & (pcols[None, :] < half))

    # SWIZZLE_32_4_4: scale (r, c) -> ((r // 128) * pc4 + c // 4) * 512 + (r % 32) * 16 + (r % 128 // 32) * 4 + c % 4.
    valid = (rows[:, None] < M) & (groups[None, :] < G)
    bits = tl.where(valid, block8.to(tl.uint8, bitcast=True), 0).to(tl.float8e4nv, bitcast=True)
    ri = rows % 128
    offs = (((rows // 128).to(tl.int64) * pc4)[:, None] + (groups // 4)[None, :]) * 512 \
        + ((ri % 32) * 16 + (ri // 32) * 4)[:, None] + (groups % 4)[None, :]
    tl.store(s_ptr + offs, bits, mask=groups[None, :] < pc4 * 4)


def _c4_02_launch(kernel, grid: tuple[int, int, int], tensors: tuple, scalars: tuple, constants: tuple,
            num_warps: int) -> None:
    """Launch ``kernel(*tensors, *scalars, *constants)`` with FP contraction disabled.

    The first call per specialization compiles; later calls use Triton's C launcher. The plan key
    holds what Triton specializes on: dtypes, 16-byte alignment and integer values.
    """
    key = (kernel, tensors[0].device, constants, num_warps, *[(t.dtype, t.data_ptr() & 15) for t in tensors],
           *scalars)
    plan = _c4_02_PLANS.get(key)
    if plan is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up NVFP4 kernels eagerly once per shape before CUDA-graph capture")
        compiled = kernel[grid](*tensors, *scalars, *constants, num_warps=num_warps, enable_fp_fusion=False)
        _c4_02_PLANS[key] = compiled, _c_launcher(compiled)
        return
    compiled, direct = plan
    hooks = knobs.runtime
    if direct is None or hooks.launch_enter_hook.calls or hooks.launch_exit_hook.calls:
        compiled[grid](*tensors, *scalars, *constants)
    else:
        launch, head = direct
        active = driver.active
        launch(*grid, active.get_current_stream(active.get_current_device()), *head,
               (*tensors, *scalars, *constants))


def _c4_02_amax(x, M, K, workspace, rot, tables) -> int:
    G = K // 16
    bm, bg, warps = _c4_02_tile(M, rot)
    n_gt = -(-G // bg)
    n_tiles = -(-M // bm) * n_gt
    programs = max(1, min(n_tiles, MAX_PARTIALS))
    cos, sin = tables if tables is not None else (workspace, workspace)
    _c4_02_launch(_c4_02_amax_kernel, (programs, 1, 1), (x, workspace, cos, sin),
            (M, K, G, x.stride(0), x.stride(1), n_gt, max(n_tiles, 1)), (ROTATIONS[rot], bm, bg), warps)
    return programs


def _c4_02_quant(x, M, K, packed_u8, scale, gs, workspace, n_part, rot, tables) -> None:
    pr, pc = scale_layout(M, K)
    bm, bg, warps = _c4_02_tile(M, rot)
    dynamic = workspace is not None
    ws = workspace if dynamic else gs
    cos, sin = tables if tables is not None else (ws, ws)
    _c4_02_launch(_c4_02_quant_kernel, (pr // bm, -(-pc // bg), 1), (x, packed_u8, scale, gs, ws, cos, sin),
            (M, K, K // 16, x.stride(0), x.stride(1), pc // 4, n_part),
            (ROTATIONS[rot], dynamic, bm, bg, MAX_PARTIALS), warps)


def _c4_02_quantize(x: Tensor, M: int, K: int, rotation: str, tables, global_scale: Tensor | None) -> NVFP4Activation:
    pr, pc = scale_layout(M, K)
    packed = torch.empty((M, K // 2), dtype=torch.float4_e2m1fn_x2, device=x.device)
    scale = torch.empty(pr * pc, dtype=torch.float8_e4m3fn, device=x.device)
    packed_u8 = packed.view(torch.uint8)
    if global_scale is None:
        workspace = torch.empty(MAX_PARTIALS + 1, dtype=torch.float32, device=x.device)
        gs = workspace[MAX_PARTIALS]
        n_part = _c4_02_amax(x, M, K, workspace, rotation, tables)
        _c4_02_quant(x, M, K, packed_u8, scale, gs, workspace, n_part, rotation, tables)
    else:
        gs = global_scale
        _c4_02_quant(x, M, K, packed_u8, scale, gs, None, 0, rotation, tables)
    return NVFP4Activation(packed, scale, gs)


class C4_02Linear(nn.Module):
    """C4-02 ``NVFP4Linear.forward``: the embedded C4-02 quantizer with the weights and GEMM of ``layer``."""

    def __init__(self, layer: nb.NVFP4Linear):
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        layer = self.layer
        lead = x.shape[:-1]
        x2 = x.reshape(-1, layer.in_features)
        M = x2.shape[0]
        if M == 0:
            return x.new_empty(*lead, layer.out_features, dtype=layer.out_dtype)
        a = _c4_02_quantize(x2, M, layer.in_features, layer.rotation, layer._tables, layer.act_global_scale)
        out = nb.nvfp4_scaled_mm(a, layer.w_packed, layer.w_scale, layer.w_global, layer.out_dtype)
        return out.reshape(*lead, layer.out_features)


def c4_02_quantize_activation(x: Tensor, rotation: str = "none", tables=None,
                              global_scale: Tensor | None = None) -> NVFP4Activation:
    M, K, tables = nb._check(x, rotation, tables)
    return _c4_02_quantize(x, M, K, rotation, tables, None if global_scale is None else global_scale.reshape(()))


# ======================================================================================
# Rejected investigation: cooperative single-launch dynamic quantizer (ablation only)
# ======================================================================================
@triton.jit
def _coop_quant_kernel(x_ptr, q_ptr, s_ptr, gs_ptr, sync_ptr, cos_ptr, sin_ptr, M, K, G, stride_m, stride_k, pc4,
                       n_gt_a, n_tiles_a, n_gt_q, n_tiles_q, ROT: tl.constexpr, BMA: tl.constexpr,
                       BGA: tl.constexpr, BM: tl.constexpr, BG: tl.constexpr):
    """amax -> grid barrier -> quantize in one cooperative launch.

    ``sync`` (int32[4]) = [arrivals, generation, amax bits of even generations, of odd generations].
    Non-negative FP32 values (and NaN after abs) order like their int32 bits, so ``atomic_max`` on the
    bits is an exact, order-independent NaN-propagating max. The last arrival zeroes the other slot
    and the counter, then bumps the generation (release): the buffer is ready for the next launch,
    also across CUDA-graph replays, with no memset.
    """
    pid = tl.program_id(0)
    P = tl.num_programs(0)
    acc = tl.zeros((), dtype=tl.float32)
    for t in range(pid, n_tiles_a, P):
        rows = (t // n_gt_a) * BMA + tl.arange(0, BMA)
        v = nb._load_rotated(x_ptr, cos_ptr, sin_ptr, rows, (t % n_gt_a) * BGA, M, K, G, stride_m, stride_k,
                             ROT, BMA, BGA)
        acc = nb._nanmax(acc, tl.reduce(tl.abs(v), None, nb._nanmax))
    gen = tl.atomic_add(sync_ptr + 1, 0, sem="acquire")
    slot = gen & 1
    tl.atomic_max(sync_ptr + 2 + slot, acc.to(tl.int32, bitcast=True), sem="relaxed")
    arrived = tl.atomic_add(sync_ptr, 1, sem="acq_rel")
    if arrived == P - 1:
        tl.atomic_xchg(sync_ptr + 3 - slot, 0, sem="relaxed")
        tl.atomic_xchg(sync_ptr, 0, sem="relaxed")
        tl.atomic_add(sync_ptr + 1, 1, sem="release")
    else:
        cur = tl.atomic_add(sync_ptr + 1, 0, sem="acquire")
        while cur == gen:
            cur = tl.atomic_add(sync_ptr + 1, 0, sem="acquire")
    gs = nb._global_scale(tl.atomic_add(sync_ptr + 2 + slot, 0, sem="acquire").to(tl.float32, bitcast=True))
    if pid == 0:
        tl.store(gs_ptr, gs)
    for t in range(pid, n_tiles_q, P):
        pid_m, pid_g = t // n_gt_q, t % n_gt_q
        rows = pid_m * BM + tl.arange(0, BM)
        v = tl.reshape(nb._load_rotated(x_ptr, cos_ptr, sin_ptr, rows, pid_g * BG, M, K, G, stride_m, stride_k,
                                        ROT, BM, BG), (BM, BG, 16))
        nb._store_tile(v, tl.reduce(tl.abs(v), 2, nb._nanmax), q_ptr, s_ptr, gs, pid_m, pid_g, M, K, G, pc4,
                       BM, BG, True)


_COOP: dict = {}


def coop_grid(x: Tensor, rotation: str) -> dict:
    """Grid of the cooperative kernel: 2 programs per SM, at most the occupancy-limited resident capacity.

    ``launch_cooperative_grid`` makes the driver refuse a grid that cannot be co-resident, so the
    barrier cannot deadlock on a too-large grid; the occupancy query records why the grid fits.
    """
    M, K = x.shape
    key = ("grid", x.device, M, K, rotation)
    if key not in _COOP:
        coop_quantize(x, rotation, None, programs=1)  # compiles the kernel for this specialization
        compiled = _COOP[("kernel", x.device, M, K, rotation, x.dtype)]
        from cuda.bindings import driver as cu
        err, per_sm = cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            compiled.function, 32 * compiled.metadata.num_warps, compiled.metadata.shared)
        if int(err) != 0:
            raise RuntimeError(f"cuOccupancyMaxActiveBlocksPerMultiprocessor failed: {err}")
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        resident = sms * int(per_sm)
        _COOP[key] = {"programs": min(2 * sms, resident), "max_active_blocks_per_sm": int(per_sm), "sms": sms,
                      "resident_capacity": resident}
    return _COOP[key]


def coop_quantize(x: Tensor, rotation: str = "none", tables=None, programs: int | None = None) -> NVFP4Activation:
    M, K, tables = nb._check(x, rotation, tables)
    pr, pc = scale_layout(M, K)
    G = K // 16
    sync = _COOP.get(("sync", x.device))
    if sync is None:
        sync = _COOP[("sync", x.device)] = torch.zeros(4, dtype=torch.int32, device=x.device)
    if programs is None:
        programs = coop_grid(x, rotation)["programs"]
    packed = torch.empty((M, K // 2), dtype=torch.float4_e2m1fn_x2, device=x.device)
    scale = torch.empty(pr * pc, dtype=torch.float8_e4m3fn, device=x.device)
    gs = torch.empty((), dtype=torch.float32, device=x.device)
    bma, bga, _ = nb._amax_tile(M, K, rotation)
    bm, bg, warps = nb._quant_tile(M, K, rotation)
    n_gt_a, n_gt_q = -(-G // bga), -(-pc // bg)
    cos, sin = tables if tables is not None else (gs, gs)
    args = (x, packed.view(torch.uint8), scale, gs, sync, cos, sin, M, K, G, x.stride(0), x.stride(1), pc // 4,
            n_gt_a, -(-M // bma) * n_gt_a, n_gt_q, (pr // bm) * n_gt_q, ROTATIONS[rotation], bma, bga, bm, bg)
    key = ("kernel", x.device, M, K, rotation, x.dtype)
    compiled = _COOP.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up the cooperative kernel eagerly before CUDA-graph capture")
        _COOP[key] = _coop_quant_kernel[(programs,)](*args, num_warps=max(warps, 4), enable_fp_fusion=False,
                                                     launch_cooperative_grid=True)
    else:
        compiled[(programs, 1, 1)](*args)
    return NVFP4Activation(packed, scale, gs)


class CoopLinear(nn.Module):
    def __init__(self, layer: nb.NVFP4Linear):
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        layer = self.layer
        a = coop_quantize(x, layer.rotation, layer._tables)
        return nb.nvfp4_scaled_mm(a, layer.w_packed, layer.w_scale, layer.w_global, layer.out_dtype)


@contextmanager
def options(**changes):
    """Temporarily change ``nb.OPTIONS`` (the setting of a CUDA graph is the one at its capture)."""
    saved = dict(nb.OPTIONS)
    nb.OPTIONS.update(changes)
    try:
        yield
    finally:
        nb.OPTIONS.clear()
        nb.OPTIONS.update(saved)


# ======================================================================================
# Timing
# ======================================================================================
def capture_shared(fn, pool):
    """``e2e.capture`` into the shared memory ``pool``.

    Timed graphs replay one at a time on one stream and their outputs are discarded, so they may
    reuse each other's scratch memory; private pools for 19 full-model graphs would not fit in 16 GB.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        fn()
    torch.cuda.synchronize()
    return graph.replay


def capture_under(fn, **opts):
    with options(**opts):
        return e2e.capture(fn)


def kernel_profile(replays: dict, flush: Tensor, reps: int = 20) -> dict:
    """Mean CUPTI kernel duration (us) per kernel name of one replay, for every method (L2 flushed)."""
    out = {}
    for name, replay in replays.items():
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                flush.zero_()
                replay()
            torch.cuda.synchronize()
        totals = defaultdict(float)
        for event in prof.events():
            if event.device_type.name == "CUDA" and "FillFunctor" not in event.name:  # drop the flush
                totals[event.name[:120]] += event.device_time
        out[name] = {k: v / reps for k, v in sorted(totals.items())}
    return out


def _timing_table(timing: dict, base: str) -> dict:
    """ms, wall_ms and speedups of every method of a Williams timing against ``base``."""
    ref = timing[base]["gpu"]["median_ms"]
    return {name: {"ms": t["gpu"]["median_ms"], "wall_ms": t["wall"]["median_ms"],
                   f"speedup_vs_{base}": ref / t["gpu"]["median_ms"]} for name, t in timing.items()}


def summarize_part_a_row(row: dict) -> dict:
    """Per mode: every method's median ms and speedups; per variant C5-03 against C4-02 and BF16."""
    flops = 2.0 * row["m"] * row["n"] * row["k"]
    out = {}
    for mode in ("eager", "cuda_graph"):
        timing = row[mode]
        bf16 = timing["bf16_cublas"]["gpu"]["median_ms"]
        ceiling = timing["nvfp4_gemm_only"]["gpu"]["median_ms"]
        table = {}
        for name in PART_A_METHODS:
            ms = timing[name]["gpu"]["median_ms"]
            table[name] = {"ms": ms, "wall_ms": timing[name]["wall"]["median_ms"], "tflops": flops / (ms * 1e-3) / 1e12,
                           "speedup_vs_bf16": bf16 / ms, "pct_of_gemm_ceiling_retained": 100.0 * ceiling / ms}
        variants = {}
        for v in VARIANTS:
            new, old = table[f"c5_03_{v}"], table[f"c4_02_{v}"]
            variants[v] = {"c5_03_ms": new["ms"], "c4_02_ms": old["ms"], "c5_03_speedup_vs_c4_02": old["ms"] / new["ms"],
                           "c5_03_speedup_vs_bf16": new["speedup_vs_bf16"],
                           "c4_02_speedup_vs_bf16": old["speedup_vs_bf16"],
                           "c5_03_beats_bf16": new["ms"] < bf16, "c5_03_faster_than_c4_02": new["ms"] < old["ms"]}
        out[mode] = {"methods": table, "variants": variants}
    return out


def summarize_part_b_model(row: dict) -> dict:
    timing = row["timing"]
    base = {k: timing[k]["gpu"]["median_ms"] for k in ("bf16_eager", "bf16_compiled", "bf16_eager_graph")}
    table = {}
    for name in PART_B_METHODS:
        ms = timing[name]["gpu"]["median_ms"]
        table[name] = {"ms": ms, "wall_ms": timing[name]["wall"]["median_ms"],
                       "tokens_per_s": row["tokens_per_call"] / (ms * 1e-3),
                       **{f"speedup_vs_{k}": v / ms for k, v in base.items()}}
    variants = {}
    for v in VARIANTS:
        for mode in ("eager", "graph"):
            new, old = table[f"c5_03_{v}_{mode}"]["ms"], table[f"c4_02_{v}_{mode}"]["ms"]
            ref = base["bf16_eager_graph" if mode == "graph" else "bf16_eager"]
            variants[f"{v}_{mode}"] = {"c5_03_ms": new, "c4_02_ms": old, "c5_03_speedup_vs_c4_02": old / new,
                                       "c5_03_speedup_vs_bf16_same_mode": ref / new,
                                       "c5_03_speedup_vs_bf16_compiled": base["bf16_compiled"] / new}
    return {"methods": table, "variants": variants}


# ======================================================================================
# Part A: single-layer latency
# ======================================================================================
def _layers_for(W: Tensor, angles: Tensor, static_gs: Tensor) -> dict:
    return {"none": nb.NVFP4Linear.from_weight(W, "none"),
            "had_b16": nb.NVFP4Linear.from_weight(W, "hadamard_b16"),
            "bfly_b16": nb.NVFP4Linear.from_weight(W, "adaptive_butterfly_b16", angles),
            "had_b16_static_gs": nb.NVFP4Linear.from_weight(W, "hadamard_b16", act_global_scale=static_gs)}


_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _bitwise_equal(a: Tensor, b: Tensor) -> bool:
    """Same shape, dtype and bytes: compares integer views, so -0.0 differs from +0.0 and NaN payloads count."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = _INT_VIEW[a.element_size()]
    return bool(torch.equal(a.contiguous().view(view), b.contiguous().view(view)))


def _same_operands(a: NVFP4Activation, b: NVFP4Activation) -> bool:
    return (torch.equal(a.packed.view(torch.uint8), b.packed.view(torch.uint8))
            and torch.equal(a.scale.view(torch.uint8), b.scale.view(torch.uint8))
            and torch.equal(a.global_scale.reshape(()).view(torch.int32), b.global_scale.reshape(()).view(torch.int32)))


def part_a_shape(device, M: int, N: int, K: int, timing: dict, flush: Tensor, seed: int, log) -> dict:
    g = torch.Generator(device=device).manual_seed(seed + M + N + K)
    x = torch.randn(M, K, generator=g, device=device).to(torch.bfloat16)
    W = (torch.randn(N, K, generator=g, device=device) * 0.02).to(torch.bfloat16)
    angles = torch.rand(K // 16, 4, 8, generator=g, device=device) * 2 * math.pi
    static_gs = nb.quantize_activation(x, "hadamard_b16").global_scale.clone()
    layers = _layers_for(W.float(), angles, static_gs)
    legacy = {v: C4_02Linear(layer) for v, layer in layers.items()}
    pre = nb.quantize_activation(x, "none")
    base = layers["none"]
    methods = {"bf16_cublas": lambda: x @ W.T,
               "nvfp4_gemm_only": lambda: nb.nvfp4_scaled_mm(pre, base.w_packed, base.w_scale, base.w_global)}
    for impl, mods in (("c4_02", legacy), ("c5_03", layers)):
        for v in VARIANTS:
            methods[f"{impl}_{v}"] = (lambda m=mods[v]: m(x))
    checks = {}
    for v, layer in layers.items():
        rot = VARIANT_ROTATION[v]
        ref = nb.quantize_activation_ref(x, rot, angles if rot == "adaptive_butterfly_b16" else None,
                                         layer.act_global_scale)
        checks[f"{v}_c5_03_operands_equal_reference"] = _same_operands(layer.quantize_input(x), ref)
        checks[f"{v}_c4_02_operands_equal_reference"] = _same_operands(
            c4_02_quantize_activation(x, rot, layer.tables(), layer.act_global_scale), ref)
        checks[f"{v}_c5_03_output_equals_c4_02_bitwise"] = _bitwise_equal(layer(x), legacy[v](x))
    eager, eager_orders = e2e.williams_timing(methods, squares=timing["squares_a"], iters=timing["iters"],
                                              flush=flush, wall_iters=timing["wall_iters_a"])
    graphs = {name: e2e.capture(fn) for name, fn in methods.items()}
    graphed, graph_orders = e2e.williams_timing(graphs, squares=timing["squares_a"], iters=timing["iters"],
                                                flush=flush, wall_iters=timing["wall_iters_a"])
    profile = kernel_profile({k: v for k, v in graphs.items() if k != "bf16_cublas"}, flush)
    row = {"m": M, "n": N, "k": K, "label": PART_A_LABELS.get((M, N, K), ""), "checks": checks,
           "c5_03_plan": {v: nb.quantizer_plan(M, K, VARIANT_ROTATION[v], v.endswith("static_gs")) for v in VARIANTS},
           "eager_williams_orders": eager_orders, "cuda_graph_williams_orders": graph_orders,
           "eager": eager, "cuda_graph": graphed, "cuda_graph_kernel_profile_us": profile}
    row["summary"] = summarize_part_a_row(row)
    s = row["summary"]["cuda_graph"]["variants"]
    log(f"  A ({M:5d},{N:5d},{K:5d}) graph ms bf16 {graphed['bf16_cublas']['gpu']['median_ms']:.4f} gemm "
        f"{graphed['nvfp4_gemm_only']['gpu']['median_ms']:.4f} | " + "  ".join(
            f"{v} {s[v]['c4_02_ms']:.4f}->{s[v]['c5_03_ms']:.4f}" for v in VARIANTS))
    return row


def ablation_shape(device, M: int, N: int, K: int, timing: dict, flush: Tensor, seed: int, replays: int, log) -> dict:
    """CUDA-graph ablation of the C5-03 changes and of the rejected cooperative single launch."""
    g = torch.Generator(device=device).manual_seed(seed + 7 * M + N + K)
    x = torch.randn(M, K, generator=g, device=device).to(torch.bfloat16)
    W = (torch.randn(N, K, generator=g, device=device) * 0.02).to(torch.bfloat16)
    layers = {"none": nb.NVFP4Linear.from_weight(W.float(), "none"),
              "had_b16": nb.NVFP4Linear.from_weight(W.float(), "hadamard_b16")}
    fns, graphs, checks = {}, {"bf16_cublas": e2e.capture(lambda: x @ W.T)}, {}
    for v, layer in layers.items():
        coop = CoopLinear(layer)
        legacy = C4_02Linear(layer)
        want = layer(x)
        with options(native_cvt=False):
            checks[f"{v}_ladder_cvt_equals_c5_03"] = _bitwise_equal(layer(x), want)
        with options(pdl=False):
            checks[f"{v}_no_pdl_equals_c5_03"] = _bitwise_equal(layer(x), want)
        checks[f"{v}_coop_equals_c5_03"] = _bitwise_equal(coop(x), want)
        graphs[f"c5_03_{v}"] = e2e.capture(lambda m=layer: m(x))
        graphs[f"c5_03_no_pdl_{v}"] = capture_under(lambda m=layer: m(x), pdl=False)
        graphs[f"c5_03_ladder_cvt_{v}"] = capture_under(lambda m=layer: m(x), native_cvt=False)
        graphs[f"c5_03_coop_single_launch_{v}"] = e2e.capture(lambda m=coop: m(x))
        graphs[f"c4_02_{v}"] = e2e.capture(lambda m=legacy: m(x))
        fns[v] = (layer, coop)
    graphs = {name: graphs[name] for name in ABLATION_METHODS}
    timed, orders = e2e.williams_timing(graphs, squares=timing["squares_a"], iters=timing["iters"], flush=flush,
                                        wall_iters=timing["wall_iters_a"])
    # Cooperative kernel: repeated graph replays with new inputs must stay exact (no deadlock, auto-reset).
    layer, coop = fns["had_b16"]
    x_static = torch.empty_like(x)
    x_static.copy_(x)
    out = {}
    replay = e2e.capture(lambda: out.__setitem__("y", coop(x_static)))
    gen = torch.Generator(device=device).manual_seed(seed + 99)
    exact = 0
    for i in range(replays):
        x_static.copy_(torch.randn(M, K, generator=gen, device=device) * (0.01 + 10.0 * (i % 7)))
        replay()
        exact += _bitwise_equal(out["y"], layer(x_static))
    sync_state = _COOP[("sync", x.device)].tolist()
    grid = coop_grid(x, "hadamard_b16")
    table = _timing_table(timed, "bf16_cublas")
    log(f"  A ablation ({M},{N},{K}) graph ms: " + "  ".join(f"{k} {v['ms']:.4f}" for k, v in table.items()))
    return {"m": M, "n": N, "k": K, "mode": "cuda_graph", "checks": checks, "williams_orders": orders,
            "timing": timed, "summary": table,
            "coop": {"grid": grid, "graph_replays": replays, "graph_replays_bitwise_exact": exact,
                     "sync_counter_after_replays": sync_state[0], "launch_cooperative_grid": True,
                     "enabled_in_nvfp4_inference_blackwell": False}}


def part_a(device, protocol: dict, flush: Tensor, log) -> dict:
    t = protocol["timing"]
    rows = []
    for M, N, K in protocol["part_a"]["shapes"]:
        rows.append(part_a_shape(device, M, N, K, t, flush, protocol["seed"], log))
        torch.cuda.empty_cache()
    ablation = [ablation_shape(device, M, N, K, t, flush, protocol["seed"], protocol["part_a"]["coop_replays"], log)
                for M, N, K in protocol["part_a"]["ablation_shapes"]]
    return {"design": "Williams-balanced orders over all methods; CUDA events, mean of `iters` flushed samples per "
                      "round (raw per-call times in event_samples_ms), 4x L2 flush before every sample; median over "
                      "rounds. `wall_ms`: host clock between synchronizes. c4_02_* is the embedded pre-C5-03 "
                      "quantizer, c5_03_* the current module; both allocate their buffers per call and share the "
                      "NVFP4 weights and scaled_mm.",
            "l2_flush_bytes": flush.numel(), "shapes": rows, "ablation": ablation}


# ======================================================================================
# Part B: full Transformer forward latency
# ======================================================================================
@torch.no_grad()
def calibrate(model, names: list[str], maps: dict, windows: np.ndarray, batch: int, device) -> dict[str, Tensor]:
    """Static FP32 scales amax(map(x)) / 2688 of the inputs of ``names`` (forward pre-hooks, BF16 autocast)."""
    modules = dict(model.named_modules())
    amax = {name: torch.zeros((), device=device) for name in names}

    def hook(name):
        def record(module, inputs):
            with torch.autocast(device_type=device.type, enabled=False):
                z = maps[name].forward(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float())
                amax[name] = torch.maximum(amax[name], z.abs().amax())
        return record

    handles = [modules[name].register_forward_pre_hook(hook(name)) for name in names]
    try:
        for first in range(0, len(windows), batch):
            x = torch.from_numpy(np.asarray(windows[first:first + batch], dtype=np.int64)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                model(x)
    finally:
        for handle in handles:
            handle.remove()
    return {n: q.nvfp4_global_scale(a.reshape(1, 1)).reshape(()) for n, a in amax.items()}


def part_b(device, protocol: dict, flush: Tensor, log) -> dict:
    t = protocol["timing"]
    rows = {}
    for name, batch in protocol["part_b"]["models"].items():
        config = PART_B_CONFIGS[name]
        torch.manual_seed(protocol["seed"])
        with torch.device(device):
            model = Transformer(config).to(torch.bfloat16).eval()
        tokens = torch.randint(0, config.vocab_size, (batch, config.sequence_length), device=device)
        linears = q.quantized_linears(model)
        maps = {n: q.BlockHadamard(16) for n in linears}
        static = calibrate(model, list(linears), maps, tokens.cpu().numpy(), batch, device)
        g = torch.Generator(device=device).manual_seed(protocol["seed"] + 5)
        angles = {n: torch.rand(m.in_features // 16, 4, 8, generator=g, device=device) * 2 * math.pi
                  for n, m in linears.items()}
        variants, keep = {}, [model]
        for v in VARIANTS:
            rot = VARIANT_ROTATION[v]
            new = copy.deepcopy(model)
            old = copy.deepcopy(model)
            for n, m in linears.items():
                layer = nb.NVFP4Linear.from_weight(m.weight.float(), rot,
                                                   angles[n] if rot == "adaptive_butterfly_b16" else None,
                                                   act_global_scale=static[n] if v.endswith("static_gs") else None)
                e2e._swap(new, {n: layer})
                e2e._swap(old, {n: C4_02Linear(layer)})
            variants[v] = {"c5_03": new, "c4_02": old}
            keep += [new, old]
        compiled = torch.compile(model)
        began = time.perf_counter()
        with torch.inference_mode():
            for _ in range(3):
                compiled(tokens)
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - began

        def forward(m):
            def run():
                with torch.inference_mode():
                    return m(tokens)
            return run

        functions = {"bf16_eager": forward(model), "bf16_compiled": forward(compiled)}
        pool = torch.cuda.graph_pool_handle()
        functions["bf16_eager_graph"] = capture_shared(functions["bf16_eager"], pool)
        checks = {}
        with torch.inference_mode():
            reference = model(tokens).float()
            for v in VARIANTS:
                a, b = variants[v]["c5_03"](tokens), variants[v]["c4_02"](tokens)
                checks[f"{v}_c5_03_logits_equal_c4_02_bitwise"] = _bitwise_equal(a, b)
                checks[f"{v}_logits_rel_diff_vs_bf16"] = e2e._rel(a, reference)
        del reference
        for v in VARIANTS:
            for impl in IMPLS:
                functions[f"{impl}_{v}_eager"] = forward(variants[v][impl])
                functions[f"{impl}_{v}_graph"] = capture_shared(functions[f"{impl}_{v}_eager"], pool)
        functions = {k: functions[k] for k in PART_B_METHODS}
        timing, orders = e2e.williams_timing(functions, squares=t["squares_b"], iters=t["iters"], flush=flush,
                                             wall_iters=t["wall_iters_b"])
        row = {"config": asdict(config), "batch": batch, "tokens_per_call": batch * config.sequence_length,
               "linear_parameters": sum(m.weight.numel() for m in linears.values()),
               "nvfp4_layers": len(linears), "compile_seconds": compile_seconds, "checks": checks,
               "williams_orders": orders, "timing": timing}
        row["summary"] = summarize_part_b_model(row)
        log(f"  B {name}: bf16 eager {timing['bf16_eager']['gpu']['median_ms']:.3f} compiled "
            f"{timing['bf16_compiled']['gpu']['median_ms']:.3f} | " + "  ".join(
                f"{k} {v['c4_02_ms']:.3f}->{v['c5_03_ms']:.3f}" for k, v in row["summary"]["variants"].items()
                if k.endswith("graph")))
        rows[name] = row
        del model, compiled, variants, functions, keep, pool
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    return {"design": "random-init BF16 weights, random tokens. NVFP4 variants replace every block linear (qkv, out, "
                      "gate, up, down) by an RTN NVFP4Linear; bfly_b16 uses random angles, had_b16_static_gs scales "
                      "calibrated on the timed tokens. c4_02_* wraps the same layers with the embedded C4-02 "
                      "quantizer. Embedding, RMSNorms, attention (SDPA) and the tied LM head stay BF16. `_graph`: one "
                      "CUDA graph per forward; bf16_compiled: torch.compile default mode.",
            "models": rows}


# ======================================================================================
# Part C: three-seed 120m real-checkpoint W4A4 perplexity
# ======================================================================================
def checkpoint_name(seed: int, spec: dict) -> str:
    return f"seed{seed}_{spec['preset']}_{spec['optimizer']}_{spec['steps']}steps.pt"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_seed(config: h.ModelConfig, train: np.ndarray, device, spec: dict, seed: int, checkpoint_dir: Path,
               log) -> tuple[Transformer, dict]:
    """``spec['steps']`` Muon steps (warmup + cosine to ``min_lr_ratio``) from ``h.initial_state(seed)``."""
    init = h.initial_state(config, seed)
    with torch.device(device):
        model = Transformer(config)
    model.load_state_dict(init)
    model.train()
    factory = h.builtin_factories()[spec["optimizer"]]
    optimizer = h.build_optimizer(spec["optimizer"], factory, model, lr=spec["lr"], auxiliary_lr=spec["auxiliary_lr"],
                                  weight_decay=spec["weight_decay"], update_dtype="fp32")
    stream = h.BatchStream(train, config, seed, device)
    steps, losses = spec["steps"], []
    began = time.perf_counter()
    for step in range(steps):
        optimizer.set_lr_multiplier(h.lr_multiplier(step, step / steps, spec["warmup_steps"], spec["min_lr_ratio"],
                                                    total_steps=steps))
        losses.append(h.train_step(model, stream.next_batch(), optimizer).item())
        model.zero_grad(set_to_none=True)
        if not math.isfinite(losses[-1]):
            raise RuntimeError(f"seed {seed} diverged at step {step + 1}")
        if (step + 1) % 160 == 0:
            log(f"  C seed {seed} step {step + 1}/{steps} loss {losses[-1]:.4f}")
    torch.cuda.synchronize()
    seconds = time.perf_counter() - began
    model.eval()
    del optimizer
    path = Path(checkpoint_dir) / checkpoint_name(seed, spec)
    c502._save_state(model.state_dict(), path)
    tokens = steps * config.step_tokens
    row = {"seed": seed, "optimizer": spec["optimizer"], "steps": steps, "warmup_steps": spec["warmup_steps"],
           "lr": spec["lr"], "auxiliary_lr": spec["auxiliary_lr"], "weight_decay": spec["weight_decay"],
           "min_lr_ratio": spec["min_lr_ratio"], "step_tokens": config.step_tokens, "tokens": tokens,
           "train_epochs": tokens / len(train), "init_fingerprint": h.state_fingerprint(init),
           "stream_fingerprint": h.stream_fingerprint(train, config, seed),
           "train_loss": math.fsum(losses[-10:]) / len(losses[-10:]), "last_train_loss": losses[-1],
           "train_curve": [[i + 1, v] for i, v in enumerate(losses) if i == 0 or (i + 1) % 10 == 0 or i + 1 == steps],
           "train_seconds": seconds, "tokens_per_second": tokens / seconds,
           "checkpoint_file": path.name, "checkpoint_sha256": e2e._checkpoint_sha256(model),
           "checkpoint_file_sha256": file_sha256(path)}
    log(f"  C seed {seed} trained {steps} steps in {seconds:.0f}s, last loss {losses[-1]:.4f}")
    return model, row


def load_checkpoint(config, device, path: Path, record: dict) -> Transformer:
    if file_sha256(path) != record["checkpoint_file_sha256"]:
        raise RuntimeError(f"{path.name}: file SHA-256 differs from the training record")
    state = torch.load(path, map_location="cpu", weights_only=True)
    with torch.device(device):
        model = Transformer(config)
    model.load_state_dict(state)
    model.eval()
    if e2e._checkpoint_sha256(model) != record["checkpoint_sha256"]:
        raise RuntimeError(f"{path.name}: tensor SHA-256 differs from the training record")
    return model


def parse_method(method: str) -> tuple[str, str, bool]:
    awq = method.startswith("awq_w4a4_")
    family, rounding = q.parse_method(method.removeprefix("awq_w4a4_"))
    return family, rounding, awq


def _stack_sha256(stacks: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for kind in q.LINEAR_KINDS:
        digest.update(kind.encode() + b"\0" + stacks[kind].detach().float().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


class ScoringLedger:
    """Counts every scored split in order; test scoring is refused until ``freeze``."""

    def __init__(self):
        self.counts = {"val": 0, "test": 0}
        self.frozen_after_val_scorings = None
        self.first_test_after_freeze = None

    def score(self, model, split: str, tokens: np.ndarray, config, device, max_windows) -> dict:
        if split == "test" and self.frozen_after_val_scorings is None:
            raise RuntimeError("test.bin may be scored only after every selection is frozen")
        if split == "val" and self.frozen_after_val_scorings is not None:
            raise RuntimeError("val.bin scoring after the selection freeze is not part of the protocol")
        self.counts[split] += 1
        if split == "test" and self.first_test_after_freeze is None:
            self.first_test_after_freeze = True
        return q.evaluate_split(model, tokens, config.sequence_length, config.micro_batch, device, max_windows)

    def freeze(self) -> None:
        self.frozen_after_val_scorings = self.counts["val"]

    def record(self) -> dict:
        return {"val_scorings": self.counts["val"], "test_scorings": self.counts["test"],
                "frozen_after_val_scorings": self.frozen_after_val_scorings,
                "test_scored_only_after_freeze": bool(self.first_test_after_freeze)}


def _hw_counts(model) -> dict:
    """What runs in the block linear slots (``ScaledInput`` wrappers are looked through)."""
    inner = [getattr(b, k) for b in model.blocks for k in q.LINEAR_KINDS]
    inner = [m.layer if isinstance(m, c502.ScaledInput) else m for m in inner]
    return {"nvfp4_modules": sum(isinstance(m, nb.NVFP4Linear) for m in inner),
            "bf16_block_linears": sum(type(m) is nn.Linear for m in inner),
            "emulated_modules": sum(isinstance(m, e2e.EmulatedW4A4Linear) for m in inner)}


class Seed:
    """Phase-1 / Phase-2 state of one seed: training record, candidates and grid rows."""

    def __init__(self, seed: int):
        self.seed = seed
        self.training = None
        self.rows: dict[tuple[str, str], dict] = {}
        self.cands: dict[str, dict] = {}
        self.awq_selection = []
        self.butterfly_fits = {}
        self.baseline = {}
        self.seconds = {"train": 0.0, "phase1": 0.0, "phase2": 0.0}
        self.bf16_fp4_gemms = None


def _row(method: str, mode: str, **extra) -> dict:
    family, rounding, awq = parse_method(method)
    return {"method": method, "mode": mode, "rotation": family, "rounding": rounding, "awq": awq,
            "awq_alpha": None, "inherited": False, "fp4_gemms_per_forward": None, "nvfp4_modules": None,
            "bf16_block_linears": None, "emulated_modules": None, "hw_weights_checked_matrices": None,
            "hw_weights_decode_exact_matrices": None, "effective_weights_sha256": None,
            "effective_weights_sha256_test": None, "quantize_seconds": None, "gptq_damping_max": None,
            "static_global_scale_range": None, "static_scale_calibration": None, **extra}


def _fill(row: dict, split: str, result: dict, base: dict, seconds: float) -> None:
    if result["tokens"] != base["tokens"]:
        raise RuntimeError("Evaluation token count changed between runs")
    row.update({f"{split}_nll": result["nll"], f"{split}_ppl": result["ppl"], f"{split}_tokens": result["tokens"],
                f"{split}_delta_nll": result["nll"] - base["nll"],
                f"{split}_relative_delta_ppl_pct": 100.0 * (result["ppl"] / base["ppl"] - 1.0),
                f"seconds_{split}": seconds})


class Study:
    """Part C pipeline. Phase 1 (train.bin + val.bin) for every seed, freeze, then Phase 2 (test.bin)."""

    def __init__(self, protocol: dict, shards, device, checkpoint_dir: Path, log):
        self.spec = protocol["part_c"]
        self.config = h.PRESETS[self.spec["preset"]]
        self.shards, self.device, self.log = shards, device, log
        self.checkpoint_dir = Path(checkpoint_dir)
        self.ledger = ScoringLedger()
        self.kinds = {k: [f"blocks.{i}.{k}" for i in range(self.config.layers)] for k in q.LINEAR_KINDS}
        self.fmt = q.FORMATS["nvfp4_g16"]

    # ---------------------------------------------------------------- helpers
    def score(self, model, split: str) -> dict:
        return self.ledger.score(model, split, self.view[split], self.config, self.device, self.spec["max_eval_windows"])

    def apply(self, model, stacks: dict[str, Tensor]) -> None:
        linears = q.quantized_linears(model)
        with torch.no_grad():
            for kind, names in self.kinds.items():
                for i, name in enumerate(names):
                    linears[name].weight.copy_(stacks[kind][i])

    def candidate(self, W, Hc, A, family: str, rounding: str, alpha: float) -> dict:
        """Quantized stacks of every kind in (scaled and) rotated coordinates, and NVFP4Linear layers."""
        t1 = time.perf_counter()
        cand = {"family": family, "rounding": rounding, "alpha": alpha, "rotations": {}, "scales": {}, "inverse_scales": {},
                "effective": {}, "fits": {}, "layers": {}}
        damping, checked, exact = [], 0, 0
        for kind, names in self.kinds.items():
            scale = c502.awq_scales(A[kind], alpha) if alpha else None
            Ws = W[kind] if scale is None else W[kind] * scale[:, None, :]
            if family == "adaptive_butterfly_b16":
                rot, cand["fits"][kind] = q.make_rotation(family, Ws, self.fmt, butterfly_steps=self.spec["butterfly_steps"])
            else:
                rot = q.Rotation() if family == "identity" else q.BlockHadamard(16)
            Hr = None
            if rounding == "gptq":
                Hr = rot.rotate_hessian(Hc[kind] if scale is None else c502.scale_hessian(Hc[kind], scale))
            Qr, enc, lam = c4._quantize(rot.forward(Ws), Hr, self.fmt, rounding)
            eff = rot.inverse(Qr)
            cand["effective"][kind] = eff if scale is None else eff / scale[:, None, :]
            cand["rotations"][kind] = rot
            cand["scales"][kind] = scale
            cand["inverse_scales"][kind] = None if scale is None else 1.0 / scale
            damping += lam
            for i, name in enumerate(names):
                layer = nb.NVFP4Linear.from_encoding(enc["codes"][i], enc["block_scale"][i], enc["global_scale"][i],
                                                     rotation=e2e.ROTATION_OF[family],
                                                     angles=rot.angles[i] if isinstance(rot, q.Butterfly) else None)
                checked += 1
                exact += bool(torch.equal(nb.dequantized_weight(layer).view(torch.int32),
                                          Qr[i].contiguous().view(torch.int32)))
                cand["layers"][name] = layer
            del Qr, enc
        cand["effective_sha256"] = _stack_sha256(cand["effective"])
        cand["info"] = {"quantize_seconds": time.perf_counter() - t1,
                        "gptq_damping_max": max(damping) if damping else None,
                        "hw_weights_checked_matrices": checked, "hw_weights_decode_exact_matrices": exact,
                        "effective_weights_sha256": cand["effective_sha256"]}
        return cand

    def modules(self, cand: dict, emulate: bool = False) -> dict:
        out = {}
        for kind, names in self.kinds.items():
            inv = cand["inverse_scales"][kind]
            for i, name in enumerate(names):
                layer = cand["layers"][name]
                m = e2e.EmulatedW4A4Linear(layer) if emulate else layer
                out[name] = m if inv is None else c502.ScaledInput(m, inv[i])
        return out

    def maps(self, cand: dict) -> dict:
        out = {}
        for kind, names in self.kinds.items():
            rot, inv = cand["rotations"][kind], cand["inverse_scales"][kind]
            for i, name in enumerate(names):
                layer_rot = e2e._layer_rotation(rot, i)
                out[name] = layer_rot if inv is None else c502.ScaledRotation(layer_rot, inv[i])
        return out

    def set_static(self, cand: dict, scales: dict | None) -> None:
        for name, layer in cand["layers"].items():
            layer.act_global_scale = None if scales is None else scales[name].to(layer.w_packed_bytes.device)

    def seqcal(self, model, cand: dict) -> dict:
        """Block by block: calibrate block i with blocks 0..i-1 already W4A4 (with their sequential scales)."""
        maps, modules, scales = self.maps(cand), self.modules(cand), {}
        self.set_static(cand, None)
        previous = {}
        try:
            for i in range(self.config.layers):
                names = [f"blocks.{i}.{k}" for k in q.LINEAR_KINDS]
                found = calibrate(model, names, maps, self.calibration["array"], self.config.micro_batch, self.device)
                for name in names:
                    scales[name] = found[name]
                    cand["layers"][name].act_global_scale = found[name]
                previous.update(e2e._swap(model, {n: modules[n] for n in names}))
        finally:
            e2e._swap(model, previous)
            self.set_static(cand, None)
        return scales

    def measure(self, model, cand: dict, mode: str, split: str, static: dict | None = None) -> tuple[dict, dict]:
        """(evaluation, hardware counts) of one candidate in one mode on one split. Restores the model."""
        if mode == "w4a16_fakequant":
            self.apply(model, cand["effective"] if split == "val" else self.effective_again(cand))
            try:
                return self.score(model, split), {}
            finally:
                self.apply(model, self.W)
        self.set_static(cand, static)
        previous = e2e._swap(model, self.modules(cand, emulate=mode == EMULATED_MODE))
        try:
            counts = {"fp4_gemms_per_forward": c502.fp4_gemms_per_forward(model, self.probe), **_hw_counts(model)}
            return self.score(model, split), counts
        finally:
            e2e._swap(model, previous)
            self.set_static(cand, None)

    def effective_again(self, cand: dict) -> dict:
        """W4A16 weights rebuilt from the retained NVFP4 layers (decode, inverse rotation, AWQ scale)."""
        out = {}
        for kind, names in self.kinds.items():
            Qr = torch.stack([nb.dequantized_weight(cand["layers"][n]) for n in names])
            eff = cand["rotations"][kind].inverse(Qr)
            scale = cand["scales"][kind]
            out[kind] = eff if scale is None else eff / scale[:, None, :]  # the Phase-1 operation
        cand["effective_sha256_test"] = _stack_sha256(out)
        return out

    def to(self, cand: dict, device) -> None:
        for name, layer in cand["layers"].items():
            cand["layers"][name] = layer.to(device)
        for kind, rot in cand["rotations"].items():
            if isinstance(rot, q.Butterfly):
                cand["rotations"][kind] = q.Butterfly(16, rot.angles.to(device))
        for key in ("scales", "inverse_scales"):
            cand[key] = {k: None if v is None else v.to(device) for k, v in cand[key].items()}
        for key in ("static_bf16cal", "static_seqcal"):
            if cand.get(key) is not None:
                cand[key] = {n: v.to(device) for n, v in cand[key].items()}

    # ---------------------------------------------------------------- phase 1
    def phase1(self, s: Seed) -> None:
        began = time.perf_counter()
        model, s.training = train_seed(self.config, self.view["train"], self.device, self.spec, s.seed,
                                       self.checkpoint_dir, self.log)
        s.seconds["train"] = s.training["train_seconds"]
        t1 = time.perf_counter()
        s.baseline["val"] = self.score(model, "val")
        self.log(f"  C seed {s.seed} BF16 val NLL {s.baseline['val']['nll']:.4f}")
        hessians, _, mean_abs = c502.collect_calibration_statistics(model, self.calibration["array"],
                                                                     self.config.micro_batch, self.device)
        linears = q.quantized_linears(model)
        self.W = {k: torch.stack([linears[n].weight.detach().float().clone() for n in ns]) for k, ns in self.kinds.items()}
        Hc = {k: torch.stack([hessians[n] for n in ns]) for k, ns in self.kinds.items()}
        A = {k: torch.stack([mean_abs[n] for n in ns]) for k, ns in self.kinds.items()}
        del hessians, mean_abs
        self.probe = torch.from_numpy(self.calibration["array"][:2]).to(self.device)
        s.bf16_fp4_gemms = c502.fp4_gemms_per_forward(model, self.probe)

        def run_modes(method: str, cand: dict, known: dict | None = None, **extra) -> None:
            for mode in MODES:
                t2 = time.perf_counter()
                static = None
                row = _row(method, mode, **extra)
                if mode == "w4a16_fakequant":
                    row.update(cand["info"])
                if mode == "w4a4_hw_static_bf16cal":
                    cand["static_bf16cal"] = static = calibrate(model, [n for ns in self.kinds.values() for n in ns],
                                                                self.maps(cand), self.calibration["array"],
                                                                self.config.micro_batch, self.device)
                    row["static_scale_calibration"] = STATIC_BF16CAL
                elif mode == "w4a4_hw_static_seqcal":
                    cand["static_seqcal"] = static = self.seqcal(model, cand)
                    row["static_scale_calibration"] = STATIC_SEQCAL
                if static is not None:
                    values = [v.item() for v in static.values()]
                    row["static_global_scale_range"] = [min(values), max(values)]
                if known is not None and mode == "w4a4_hw_dynamic":
                    result, counts = known["result"], known["counts"]
                else:
                    result, counts = self.measure(model, cand, mode, "val", static)
                row.update(counts)
                _fill(row, "val", result, s.baseline["val"], time.perf_counter() - t2)
                s.rows[(method, mode)] = row
                self.log(f"  C seed {s.seed} {method:<38} {mode:<24} val dNLL {row['val_delta_nll']:+.4f}")
            if method == EMULATED_METHOD:
                t2 = time.perf_counter()
                row = _row(method, EMULATED_MODE)
                result, counts = self.measure(model, cand, EMULATED_MODE, "val")
                row.update(counts)
                _fill(row, "val", result, s.baseline["val"], time.perf_counter() - t2)
                s.rows[(method, EMULATED_MODE)] = row
            self.to(cand, "cpu")
            cand["effective"] = None  # rebuilt from the layers for the test scoring
            s.cands[method] = cand

        plain = {}
        for method in BASE_METHODS:
            family, rounding, _ = parse_method(method)
            cand = self.candidate(self.W, Hc, A, family, rounding, 0.0)
            if cand["fits"]:
                s.butterfly_fits[method] = cand["fits"]
            run_modes(method, cand)
            plain[method] = cand
        for method, base in AWQ_BASE.items():
            family, rounding, _ = parse_method(base)
            t2 = time.perf_counter()
            scores, kept = {0.0: s.rows[(base, "w4a4_hw_dynamic")]["val_nll"]}, {}
            for alpha in self.spec["awq_alphas"]:
                if alpha == 0:
                    continue
                cand = self.candidate(self.W, Hc, A, family, rounding, alpha)
                result, counts = self.measure(model, cand, "w4a4_hw_dynamic", "val")
                scores[alpha] = result["nll"]
                kept[alpha] = (cand, {"result": result, "counts": counts})
            alphas = list(self.spec["awq_alphas"])
            val_nll = [scores[a] for a in alphas]
            chosen = c502.select_alpha(alphas, val_nll)
            s.awq_selection.append({"method": method, "base_method": base, "alphas": alphas, "val_nll_hw_dynamic": val_nll,
                                    "selected_alpha": chosen, "selected_on": "val.bin, w4a4_hw_dynamic",
                                    "seconds": time.perf_counter() - t2})
            self.log(f"  C seed {s.seed} {method} alpha val NLL " + " ".join(f"{a}:{v:.4f}" for a, v in zip(alphas, val_nll))
                     + f" -> {chosen}")
            if chosen == 0:
                for mode in MODES:
                    twin = s.rows[(base, mode)]
                    s.rows[(method, mode)] = {**twin, "method": method, "awq": True, "awq_alpha": 0.0, "inherited": True}
            else:
                cand, known = kept.pop(chosen)
                if cand["fits"]:
                    s.butterfly_fits[method] = cand["fits"]
                run_modes(method, cand, known, awq_alpha=chosen)
            del kept
            torch.cuda.empty_cache()
        if e2e._checkpoint_sha256(model) != s.training["checkpoint_sha256"]:
            raise RuntimeError("Phase 1 did not restore the checkpoint weights")
        model = Hc = A = None  # (the nested run_modes closes over model)
        self.W = None
        torch.cuda.empty_cache()
        s.seconds["phase1"] = time.perf_counter() - t1
        self.log(f"  C seed {s.seed} phase 1 done ({time.perf_counter() - began:.0f}s)")

    # ---------------------------------------------------------------- phase 2
    def phase2(self, s: Seed) -> None:
        t1 = time.perf_counter()
        model = load_checkpoint(self.config, self.device, self.checkpoint_dir / s.training["checkpoint_file"], s.training)
        s.baseline["test"] = self.score(model, "test")
        linears = q.quantized_linears(model)
        self.W = {k: torch.stack([linears[n].weight.detach().float().clone() for n in ns]) for k, ns in self.kinds.items()}
        self.probe = torch.from_numpy(self.calibration["array"][:2]).to(self.device)
        for method, cand in s.cands.items():
            self.to(cand, self.device)
            for mode in (*MODES, EMULATED_MODE):
                row = s.rows.get((method, mode))
                if row is None:
                    continue
                t2 = time.perf_counter()
                static = cand.get("static_bf16cal") if mode == "w4a4_hw_static_bf16cal" else \
                    cand.get("static_seqcal") if mode == "w4a4_hw_static_seqcal" else None
                result, counts = self.measure(model, cand, mode, "test", static)
                for key, value in counts.items():
                    if row[key] != value:
                        raise RuntimeError(f"{method}/{mode}: hardware count {key} changed between val and test")
                if mode == "w4a16_fakequant":
                    row["effective_weights_sha256_test"] = cand["effective_sha256_test"]
                _fill(row, "test", result, s.baseline["test"], time.perf_counter() - t2)
            self.to(cand, "cpu")
            self.log(f"  C seed {s.seed} {method:<38} test dNLL " + " ".join(
                f"{m.removeprefix('w4a4_')}:{s.rows[(method, m)]['test_delta_nll']:+.4f}" for m in MODES))
        for method, base in AWQ_BASE.items():
            if s.rows[(method, MODES[0])]["inherited"]:
                for mode in MODES:
                    s.rows[(method, mode)] = {**s.rows[(base, mode)], "method": method, "awq": True, "awq_alpha": 0.0,
                                              "inherited": True}
        if e2e._checkpoint_sha256(model) != s.training["checkpoint_sha256"]:
            raise RuntimeError("Phase 2 did not restore the checkpoint weights")
        del model
        self.W = None
        s.cands.clear()
        torch.cuda.empty_cache()
        s.seconds["phase2"] = time.perf_counter() - t1

    def run(self) -> dict:
        began = time.perf_counter()
        self.view = self.shards.open("train", "val")
        self.calibration = q.calibration_windows(SimpleNamespace(splits={"train": self.view["train"]}),
                                                 self.spec["calibration_sequences"], self.config.sequence_length,
                                                 self.spec["calibration_seed"])
        seeds = [Seed(seed) for seed in self.spec["seeds"]]
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            self.phase1(s)
        phase1_opened = list(self.shards.opened)
        self.ledger.freeze()
        selections = {"awq": {str(s.seed): s.awq_selection for s in seeds},
                      "best_method_by_val": best_by(seeds_rows(seeds), "val")}
        frozen_sha = hashlib.sha256(json.dumps(selections, sort_keys=True).encode()).hexdigest()
        self.view = {**self.view, **self.shards.open("test")}
        for s in seeds:
            self.phase2(s)
        out_seeds = []
        for s in seeds:
            rows = [s.rows[key] for key in _expected_grid()]
            out_seeds.append({"seed": s.seed, "training": s.training, "baseline_val": s.baseline["val"],
                              "baseline_test": s.baseline["test"], "bf16_fp4_gemms_per_forward": s.bf16_fp4_gemms,
                              "butterfly_fits": s.butterfly_fits, "awq_selection": s.awq_selection, "grid": rows,
                              "seconds": dict(s.seconds)})
        return {"preset": self.spec["preset"], "config": asdict(self.config),
                "calibration": {k: v for k, v in self.calibration.items() if k != "array"},
                "splits_opened_in_phase1": phase1_opened, "scoring_ledger": self.ledger.record(),
                "frozen_selections": selections, "frozen_selections_sha256": frozen_sha,
                "checkpoint_dir": str(self.checkpoint_dir), "seeds": out_seeds,
                "seconds": time.perf_counter() - began}


def seeds_rows(seeds) -> list[list[dict]]:
    return [list(s.rows.values()) for s in seeds]


def best_by(rows_per_seed: list[list[dict]], split: str) -> dict:
    """Per mode: the method with the lowest NLL summed over seeds on ``split`` (ties: method order)."""
    out = {}
    for mode in MODES:
        totals = {m: math.fsum(next(r for r in rows if r["method"] == m and r["mode"] == mode)[f"{split}_nll"]
                               for rows in rows_per_seed) for m in METHODS}
        out[mode] = min(METHODS, key=lambda m: (totals[m], METHODS.index(m)))
    return out


# ======================================================================================
# Statistics (pure functions of the stored rows; the validator recomputes them)
# ======================================================================================
def _grid(seed_record: dict) -> dict:
    return {(r["method"], r["mode"]): r for r in seed_record["grid"]}


def analyze(part_c: dict) -> dict:
    seeds = part_c["seeds"]
    if len(seeds) != N_SEEDS:
        return None
    grids = [_grid(s) for s in seeds]
    cells = {}
    for (method, mode) in grids[0]:
        cells[f"{method}/{mode}"] = {split: {
            "delta_nll": c4.mean_sd([g[(method, mode)][f"{split}_delta_nll"] for g in grids]),
            "relative_delta_ppl_pct": c4.mean_sd([g[(method, mode)][f"{split}_relative_delta_ppl_pct"] for g in grids])}
            for split in SPLITS}
    contrasts = []
    for name, x, y in CONTRASTS:
        entry = {"name": name, "x": list(x), "y": list(y)}
        for split in SPLITS:
            diffs = [g[x][f"{split}_delta_nll"] - g[y][f"{split}_delta_nll"] for g in grids]
            entry[split] = {**c4.paired_t_df2(diffs), "p_sign_flip": c502.sign_flip_p(diffs)}
        contrasts.append(entry)
    for split in SPLITS:
        for key, src in (("p_holm", "p_two_sided"), ("p_sign_flip_holm", "p_sign_flip")):
            for entry, p in zip(contrasts, c4.holm_bonferroni([e[split][src] for e in contrasts])):
                entry[split][key] = p
    rows = [r for s in seeds for r in s["grid"]]
    hw = [r for r in rows if r["mode"] in HW_MODES]
    emul = [(s["seed"], _grid(s)) for s in seeds]
    parity = []
    for seed, g in emul:
        hw_row, em = g[(EMULATED_METHOD, "w4a4_hw_dynamic")], g[(EMULATED_METHOD, EMULATED_MODE)]
        for split in SPLITS:
            parity.append({"seed": seed, "split": split, "hw_nll": hw_row[f"{split}_nll"],
                           "emulated_nll": em[f"{split}_nll"],
                           "abs_diff": abs(hw_row[f"{split}_nll"] - em[f"{split}_nll"])})
    max_diff = max(p["abs_diff"] for p in parity)
    unique = [r for r in rows if not r["inherited"]]
    w4a16 = [r for r in unique if r["mode"] == "w4a16_fakequant"]
    by_val, by_test = best_by([s["grid"] for s in seeds], "val"), best_by([s["grid"] for s in seeds], "test")
    return {
        "n_seeds": N_SEEDS, "df": N_SEEDS - 1, "sign_flip_min_p": SIGN_FLIP_MIN_P,
        "endpoint": "per-seed delta NLL (quantized - BF16) of the same checkpoint; contrast differences are x - y",
        "cells": cells, "contrasts": contrasts,
        "selection": {"best_method_by_val": by_val, "best_method_by_test": by_test,
                      "confirmed_on_test": {m: by_val[m] == by_test[m] for m in MODES},
                      "selected_test_delta_nll": {m: cells[f"{by_val[m]}/{m}"]["test"]["delta_nll"]["mean"] for m in MODES},
                      "awq_selected_alpha": {str(s["seed"]): {a["method"]: a["selected_alpha"] for a in s["awq_selection"]}
                                             for s in seeds}},
        "hardware": {"fp4_gemms_per_forward": sorted({r["fp4_gemms_per_forward"] for r in hw}),
                     "fp4_gemms_per_forward_emulated": sorted({r["fp4_gemms_per_forward"] for r in rows
                                                               if r["mode"] == EMULATED_MODE}),
                     "fp4_gemms_per_forward_bf16_baseline": sorted({s["bf16_fp4_gemms_per_forward"] for s in seeds}),
                     "bf16_block_linears_in_hw_rows": sorted({r["bf16_block_linears"] for r in hw}),
                     "hw_rows": len(hw), "hw_rows_unique": sum(not r["inherited"] for r in hw),
                     "hw_rows_inherited": sum(r["inherited"] for r in hw),
                     "decode_checked_rows": len(w4a16),
                     "decode_checked_matrices": sum(r["hw_weights_checked_matrices"] for r in w4a16),
                     "decode_exact_matrices": sum(r["hw_weights_decode_exact_matrices"] for r in w4a16)},
        "rows": {"total": len(rows), "unique": len(unique), "inherited": len(rows) - len(unique)},
        "hardware_vs_emulation": {"rows": parity, "max_abs_nll_difference": max_diff,
                                  "tolerance_nats": part_c["protocol"]["emulation_parity_nats"],
                                  "passed": max_diff <= part_c["protocol"]["emulation_parity_nats"]},
    }


# ======================================================================================
# Kernel parity (recorded with the results)
# ======================================================================================
def kernel_parity(device, quick: bool) -> dict:
    """Bit-exact C5-03 operands against the eager reference (C4-02 grid, small/medium/unaligned shapes,
    three input dtypes, non-contiguous inputs and special values) and against the embedded C4-02 quantizer."""
    grid = [(m, k) for m in (1, 37, 129) for k in (64, 80, 768)] if quick else \
        [(m, k) for m in (1, 37, 128, 129, 256, 512) for k in (64, 80, 512, 768, 1408, 2048)]
    extra = [(16, 4096), (33, 1408), (1000, 512), (1024, 1408), (257, 2048), (7, 8192)] if not quick else [(33, 1408)]
    failures, cases = [], 0

    def check(x, rotation, angles, static, label):
        nonlocal cases
        cases += 1
        tables = nb.butterfly_tables(angles) if angles is not None else None
        want = nb.quantize_activation_ref(x, rotation, angles, static)
        got = nb.quantize_activation(x, rotation, tables, static)
        old = c4_02_quantize_activation(x, rotation, tables, static)
        if not (_same_operands(got, want) and _same_operands(old, want)):
            failures.append(label)

    for M, K in grid + extra:
        g = torch.Generator(device=device).manual_seed(M * 31 + K)
        x = (torch.randn(M, K, generator=g, device=device) * 3).to(torch.bfloat16)
        x[:, :16] *= 30
        angles = torch.rand(K // 16, 4, 8, generator=g, device=device) * 2 * math.pi
        for rotation in ROTATIONS:
            a_ = angles if rotation == "adaptive_butterfly_b16" else None
            ref = nb.quantize_activation_ref(x, rotation, a_)
            for static in (None, ref.global_scale * 0.5):
                check(x, rotation, a_, static, [M, K, rotation, static is not None, "bf16"])
    special = torch.randn(64, 512, device=device)
    special[0, :8] = torch.tensor([0.0, -0.0, 1e-45, -1e-45, 1e-38, float("inf"), float("-inf"), float("nan")])
    special[1, :16] = torch.logspace(-40, 0, 16, device=device)
    base = torch.randn(300, 1024, device=device)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for x in (base.to(dtype)[::2, 256:768], base.to(dtype).t().contiguous().t()[:, :512], special.to(dtype)):
            angles = torch.rand(x.shape[1] // 16, 4, 8, device=device) * 2 * math.pi
            for rotation in ROTATIONS:
                check(x, rotation, angles if rotation == "adaptive_butterfly_b16" else None, None,
                      [list(x.shape), rotation, str(dtype), x.is_contiguous()])
    return {"cases": cases, "bitwise_failures": failures, "passed": not failures,
            "reference": "micro_rotated_ptq rotation (FP32) + nvfp4_encode + pack_e2m1 + to_blocked; every case also "
                         "checks the embedded C4-02 quantizer"}


# ======================================================================================
# Summary and caveats
# ======================================================================================
def summarize(payload: dict) -> dict:
    out = {}
    if payload.get("part_a"):
        out["part_a"] = [{"mnk": [r["m"], r["n"], r["k"]], "label": r["label"],
                          "cuda_graph": r["summary"]["cuda_graph"]["variants"],
                          "eager": r["summary"]["eager"]["variants"],
                          "bf16_cublas_ms": r["summary"]["cuda_graph"]["methods"]["bf16_cublas"]["ms"],
                          "nvfp4_gemm_only_ms": r["summary"]["cuda_graph"]["methods"]["nvfp4_gemm_only"]["ms"]}
                         for r in payload["part_a"]["shapes"]]
        out["part_a_trailing_bf16_cuda_graph"] = [
            [r["m"], r["n"], r["k"], v] for r in payload["part_a"]["shapes"] for v in VARIANTS
            if not r["summary"]["cuda_graph"]["variants"][v]["c5_03_beats_bf16"]]
        out["part_a_ablation"] = [{"mnk": [a["m"], a["n"], a["k"]], "ms": {k: v["ms"] for k, v in a["summary"].items()}}
                                  for a in payload["part_a"]["ablation"]]
    if payload.get("part_b"):
        out["part_b"] = {name: m["summary"]["variants"] for name, m in payload["part_b"]["models"].items()}
    if payload.get("part_c"):
        c = payload["part_c"]
        out["part_c"] = {"analysis_selection": c["analysis"]["selection"] if c["analysis"] else None,
                         "baseline": [{"seed": s["seed"], "val_nll": s["baseline_val"]["nll"],
                                       "test_nll": s["baseline_test"]["nll"]} for s in c["seeds"]]}
    return out


def caveats(payload: dict) -> list[str]:
    protocol = payload["protocol"]
    parts = protocol["parts"]
    timing = protocol["timing"]
    spec = protocol["part_c"]
    env = (payload.get("meta") or {}).get("environment") or {}
    dev_name = env.get("device") or protocol["device"]
    cap = env.get("capability")
    dev_label = f"{dev_name} (sm_{cap[0]}{cap[1]})" if cap else str(dev_name)
    notes = []
    if "C" in parts:
        n_seeds = len(spec["seeds"])
        part_c = payload.get("part_c")
        has_analysis = (n_seeds == N_SEEDS) if part_c is None else (part_c.get("analysis") is not None)
        if n_seeds == N_SEEDS and has_analysis:
            notes.append(
                f"Part C statistics have df = {N_SEEDS - 1} ({N_SEEDS} paired seeds): the two-sided t-test has "
                f"little power, and the exact sign-flip p-value cannot fall below {SIGN_FLIP_MIN_P:g} for any "
                "contrast, so no sign-flip result can be significant."
            )
        else:
            notes.append(
                f"Part C ran {n_seeds} seed{'s' if n_seeds != 1 else ''}; paired contrast analysis is only "
                f"computed for {N_SEEDS} seeds (part_c.analysis is None)."
            )
        data_rec = payload.get("data") or {}
        src_info = data_rec.get("source") or {}
        enc = data_rec.get("encoding") or "unknown"
        if src_info.get("name") == "fineweb-edu" and enc == "gpt2":
            corpus = "FineWeb-Edu sample-10BT shards, GPT-2 BPE"
        else:
            corpus = f"{src_info.get('name') or src_info.get('dataset') or 'local shards'}, {enc}"
        window_note = (f" (evaluation limited to {spec['max_eval_windows']} windows per split)"
                       if spec["max_eval_windows"] is not None else "")
        notes.append(
            f"Part C uses one corpus ({corpus}), one calibration draw of {spec['calibration_sequences']} "
            f"train.bin sequences (seed {spec['calibration_seed']}) and one model size: the {spec['preset']} "
            f"model trained {spec['steps']} steps with {spec['optimizer']} on one {dev_label}{window_note}. "
            "Deltas may differ for longer training, other optimizers or larger models."
        )
        ckpt_phrase = "1 checkpoint is" if n_seeds == 1 else f"{n_seeds} checkpoints are"
        notes.append(
            f"GPU training is seeded (initial state, batch stream) but not claimed to be bit-reproducible; "
            f"the {ckpt_phrase} retained outside git with their tensor and file SHA-256 values."
        )
        test_sha = ((data_rec.get("splits") or {}).get("test") or {}).get("sha256")
        if test_sha == q.EXPECTED_SHA256["test"]:
            notes.append(
                "test.bin is held out from every selection rule of this run (Phase 2 maps it after the freeze), "
                "but the same test.bin was scored by earlier cycles on other checkpoints; it is not a study-level "
                "unseen holdout."
            )
        else:
            notes.append(
                "test.bin is held out from every selection rule of this run (Phase 2 maps it after the freeze); "
                "its SHA-256 differs from the canonical FineWeb-Edu test split, so earlier-cycle FineWeb-Edu "
                "scoring does not apply to this test split."
            )
        notes.append(
            "AWQ alpha is chosen per seed on val.bin with live hardware W4A4 dynamic scoring; an alpha = 0 choice "
            "makes the AWQ rows copies of the plain rotation + GPTQ rows (counted as inherited, not as measurements)."
        )
    if "A" in parts or "B" in parts:
        notes.append(
            f"Part A/B weights and activations are random; latency does not depend on values. A "
            f"{timing['l2_flush_multiple']}x L2 flush before every sample makes the activation come from DRAM, "
            "which is pessimistic for a layer whose input was just written by the previous kernel."
        )
        notes.append(
            f"CUDA-event resolution on this GPU is about 2 us; per-round values are means of {timing['iters']} "
            "flushed calls and the reported value is the median over Williams rounds."
        )
        if "B" in parts:
            notes.append(
                "bf16_compiled is torch.compile(default); NVFP4 models run eager or as CUDA graphs, not compiled."
            )
        if "A" in parts:
            notes.append(
                "The cooperative single-launch quantizer (ablation) is bit-exact and survives repeated graph replays "
                "but is slower than the two-pass PDL quantizer, so it is not enabled; a single-program fused "
                "quantizer was measured during development at 10-25x slower and dropped."
            )
        notes.append(
            "Static-scale (had_b16_static_gs) differences against C4-02 are near parity: their sign changes between "
            "shapes and between sessions, so no static-scale speedup is claimed. Eager full-forward (Part B) times "
            "vary more between sessions than the CUDA-graph ratios do; the graph ratios are the reproducible "
            "latency result."
        )
    if payload.get("canonical"):
        post = payload.get("postprocess")
        meta_sha = (payload.get("meta") or {}).get("git_sha")
        if (isinstance(post, dict) and post.get("source_result_git_sha") in CANONICAL_SOURCE_ARTIFACTS
                and meta_sha == CANONICAL_PRODUCER_SHA):
            src_short = post["source_result_git_sha"][:7]
            prod_short = meta_sha[:7]
            notes.append(
                f"Study history: a superseded canonical run at commit 4736807 scored the same test.bin before host "
                f"launch-plan caching was added at {prod_short}; this result is postprocessed from the complete "
                f"rerun committed at {src_short} (measured by {prod_short}). Both runs (logs and the superseded "
                f"JSON) are retained in artifacts/c5_03_run_evidence/ with pinned SHA-256 values, and test.bin was "
                f"also scored by earlier cycles, so the test scores here are not a first look at an unseen holdout."
            )
        else:
            producer_short = meta_sha[:7] if isinstance(meta_sha, str) and meta_sha else "uncommitted"
            notes.append(
                f"Study history: earlier C5-03 canonical runs (4736807 and 38cef48) and prior cycles scored the same "
                f"test.bin; this canonical measurement was produced at commit {producer_short}, so its test scores "
                f"are not a first look at an unseen holdout."
            )
    rows = payload.get("part_a", {}).get("shapes", []) if payload.get("part_a") else []
    trailing = [f"{v} at ({r['m']},{r['n']},{r['k']}) {r['summary']['cuda_graph']['variants'][v]['c5_03_ms']:.4f} ms "
                f"vs bf16 {r['summary']['cuda_graph']['methods']['bf16_cublas']['ms']:.4f} ms"
                for r in rows for v in VARIANTS if not r["summary"]["cuda_graph"]["variants"][v]["c5_03_beats_bf16"]]
    if trailing:
        notes.append("C5-03 W4A4 (CUDA graph) still trails BF16 cuBLAS for: " + "; ".join(trailing) + ".")
    elif rows:
        notes.append("C5-03 W4A4 (CUDA graph) beats BF16 cuBLAS at every Part A shape and variant.")
    slower = [f"{v} at ({r['m']},{r['n']},{r['k']})" for r in rows for v in VARIANTS
              if not r["summary"]["cuda_graph"]["variants"][v]["c5_03_faster_than_c4_02"]]
    if slower:
        notes.append("C5-03 is not faster than C4-02 (CUDA graph median) for: " + ", ".join(slower) + ".")
    c = payload.get("part_c")
    if c and c.get("analysis"):
        alphas = c["analysis"]["selection"]["awq_selected_alpha"]
        zero = sum(1 for per in alphas.values() for a in per.values() if a == 0)
        notes.append(f"AWQ selected alpha = 0 in {zero} of {sum(len(p) for p in alphas.values())} (seed, method) "
                     "cases; those rows are inherited copies.")
    return notes


# ======================================================================================
# Guards and provenance
# ======================================================================================
def _is_hex(value, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def reserved(path: Path) -> bool:
    path = Path(path)
    return any(fnmatch.fnmatchcase(name, RESERVED_PATTERN) for name in (path.name, path.resolve().name))


def require_canonical_host(device, git: dict, output: Path, canonical: bool) -> None:
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


def _module_repo_path(name: str) -> str:
    return "/".join(name.split(".")) + ".py"


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


def _git_blob(git_sha: str, repo_path: str) -> bytes:
    blob = _git("show", f"{git_sha}:{repo_path}")
    if blob.returncode != 0:
        raise ValueError(f"cannot read {repo_path} at commit {git_sha}")
    return blob.stdout


def _require_git_commit(sha: str, label: str) -> None:
    kind = _git("cat-file", "-t", sha) if _is_hex(sha, 40) else None
    if kind is None or kind.returncode != 0 or kind.stdout.strip() != b"commit":
        raise ValueError(f"{label} {sha} is not a commit object")


@functools.cache
def _commit_timestamp_cached(root: str, git_sha: str) -> datetime:
    _require_git_commit(git_sha, "git_sha")
    res = subprocess.run(["git", "-C", root, "show", "-s", "--format=%cI", git_sha], capture_output=True)
    if res.returncode != 0:
        raise ValueError(f"cannot read commit timestamp of {git_sha}")
    return datetime.fromisoformat(res.stdout.decode().strip())


def _commit_timestamp(git_sha: str) -> datetime:
    return _commit_timestamp_cached(str(ROOT), git_sha)


@functools.cache
def _code_modules_at_commit(git_sha: str) -> tuple[str, ...]:
    listing = _git("ls-tree", "-r", "--name-only", git_sha, "--", "research")
    if listing.returncode != 0:
        raise ValueError(f"cannot list the tree of commit {git_sha}")
    files = set(listing.stdout.decode().splitlines())
    return _import_closure(lambda n: _git_blob(git_sha, _module_repo_path(n)).decode()
                           if _module_repo_path(n) in files else None)


@functools.cache
def _code_sha256_at_commit(git_sha: str, modules: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for module in modules:
        digest.update(module.encode() + b"\0" + _git_blob(git_sha, _module_repo_path(module)))
    return digest.hexdigest()


@functools.cache
def _driver_capabilities_at_commit(git_sha: str) -> tuple[str | None, bool]:
    """``(declared SCHEMA_VERSION, supports --postprocess-from-git)`` of the driver committed at ``git_sha``."""
    _require_git_commit(git_sha, "git_sha")
    try:
        tree = ast.parse(_git_blob(git_sha, _module_repo_path(DRIVER_MODULE)).decode())
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot parse {DRIVER_MODULE} at git_sha {git_sha}") from exc
    schema_version, has_fn, has_cli = None, False, False
    for node in tree.body:
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else \
            node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(target, ast.Name) and target.id == "SCHEMA_VERSION" and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            schema_version = node.value.value
        elif isinstance(node, ast.FunctionDef) and node.name == "postprocess_from_git":
            has_fn = True
        elif isinstance(node, ast.FunctionDef) and node.name == "build_parser":
            has_cli = any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "add_argument"
                          and c.args and isinstance(c.args[0], ast.Constant)
                          and c.args[0].value == "--postprocess-from-git" for c in ast.walk(node))
    return schema_version, has_fn and has_cli


def _matches_current_summary_code(git_sha: str) -> bool:
    """True iff the committed ``SUMMARY_MODULES`` at ``git_sha`` have the bytes of the validating checkout."""
    return all(_git_blob(git_sha, _module_repo_path(m)) == _module_path(m).read_bytes() for m in SUMMARY_MODULES)


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


ENV_KEYS = ("torch", "triton", "cuda", "driver", "device", "capability")


def run_config_hash(quick: bool, protocol: dict, code: str, git_sha, env: dict, data_sha256: dict,
                    schema: str = SCHEMA_VERSION) -> str:
    """``schema`` is the schema version of the run that measured (the source schema after a postprocess)."""
    return _digest({"schema": schema, "quick": quick, "protocol": protocol, "code_sha256": code,
                    "git_sha": git_sha, "env": {k: env.get(k) for k in ENV_KEYS}, "data_sha256": data_sha256})


def raw_measurements(payload: dict) -> dict:
    """The raw measurements: ``RAW_MEASUREMENT_KEYS``, with ``part_c`` stripped of its derived ``analysis``."""
    raw = {key: payload[key] for key in RAW_MEASUREMENT_KEYS}
    if isinstance(raw["part_c"], dict):
        raw["part_c"] = {k: v for k, v in raw["part_c"].items() if k not in DERIVED_PART_C_KEYS}
    return raw


def raw_measurements_sha256(payload: dict) -> str:
    """SHA-256 of the raw measurements in one canonical JSON form (sorted keys, no whitespace)."""
    text = json.dumps(raw_measurements(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
_NOFOLLOW_FLAGS = os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _relative_under_root(path: Path) -> Path:
    """Path relative to ``ROOT`` without resolving symlinks; rejects ``..`` traversal."""
    if ".." in path.parts:
        raise ValueError(f"Refusing to write {path}: parent traversal ('..') is not allowed")
    root_resolved = ROOT.resolve()
    if not path.is_absolute():
        return (Path.cwd().resolve() / path).relative_to(root_resolved)
    for base in (ROOT, root_resolved):
        try:
            return path.relative_to(base)
        except ValueError:
            pass
    raise ValueError(f"{path} is outside {ROOT}")


def _guarded_relative(path: Path, *, canonical_dest: bool = False) -> Path:
    """Validate ``path`` against ``ROOT``, ``CANONICAL_OUTPUT``, and parent traversal."""
    if ".." in path.parts:
        raise ValueError(f"Refusing to write {path}: parent traversal ('..') is not allowed")
    canonical_rel = CANONICAL_OUTPUT.relative_to(ROOT)
    try:
        relative = _relative_under_root(path)
    except ValueError as exc:
        if "parent traversal" in str(exc):
            raise
        if canonical_dest:
            raise ValueError(f"The canonical result goes to {CANONICAL_OUTPUT}, not {path}") from None
        raise ValueError(f"{path} is a reserved result path outside {ROOT}") from None
    if canonical_dest and relative != canonical_rel:
        raise ValueError(f"The canonical result goes to {CANONICAL_OUTPUT}, not {path}")
    if not relative.parts or relative.name in ("", ".", ".."):
        raise ValueError(f"Refusing to write invalid path {path}")
    return relative


def _check_not_symlink(name: str, dir_fd: int, full_path: Path, node: Path) -> None:
    try:
        st = os.lstat(name, dir_fd=dir_fd)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(st.st_mode):
        raise ValueError(f"Refusing to write {full_path}: {node} is a symlink")


def _open_verified_parent_fd(path: Path, relative: Path, *, create: bool) -> tuple[int, str] | None:
    """Walk ``relative`` from ``ROOT`` with ``O_NOFOLLOW`` and return ``(dir_fd, filename)``."""
    dir_fd = os.open(ROOT.resolve(), _DIR_OPEN_FLAGS)
    cur = ROOT
    try:
        for part in relative.parts[:-1]:
            cur = cur / part
            try:
                next_fd = os.open(part, _DIR_OPEN_FLAGS | os.O_NOFOLLOW, dir_fd=dir_fd)
            except FileNotFoundError:
                if not create:
                    os.close(dir_fd)
                    dir_fd = -1
                    return None
                try:
                    os.mkdir(part, 0o777, dir_fd=dir_fd)
                except FileExistsError:
                    pass
                try:
                    next_fd = os.open(part, _DIR_OPEN_FLAGS | os.O_NOFOLLOW, dir_fd=dir_fd)
                except OSError as exc:
                    if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                        _check_not_symlink(part, dir_fd, path, cur)
                        raise ValueError(f"Refusing to write {path}: {cur} is not a directory") from None
                    raise
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    _check_not_symlink(part, dir_fd, path, cur)
                    raise ValueError(f"Refusing to write {path}: {cur} is not a directory") from None
                raise
            os.close(dir_fd)
            dir_fd = next_fd
        filename = relative.parts[-1]
        _check_not_symlink(filename, dir_fd, path, cur / filename)
        result = (dir_fd, filename)
        dir_fd = -1
        return result
    finally:
        if dir_fd >= 0:
            os.close(dir_fd)


def _verify_canonical_output_path(path: Path) -> Path:
    """Single destination rule shared by CLI parsing, offline validation and the guarded writer.

    Requires ``path`` to bind lexically to ``CANONICAL_OUTPUT`` under ``ROOT`` with no ``..`` traversal
    and no symlink at any existing directory component or at the target file.
    """
    path = Path(path)
    relative = _guarded_relative(path, canonical_dest=True)
    pre = _open_verified_parent_fd(path, relative, create=False)
    if pre is not None:
        os.close(pre[0])
    return relative


def _create_temp_at(dir_fd: int, filename: str) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW_FLAGS
    for _ in range(128):
        candidate = f".{filename}.{secrets.token_hex(8)}.tmp"
        try:
            return os.open(candidate, flags, 0o600, dir_fd=dir_fd), candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not allocate unique temporary file for {filename}")


def _write_json_atomic(path: Path, payload: dict, *, canonical_dest: bool = False) -> dict:
    """Lossless atomic write through one verified ``dir_fd`` and an ``O_EXCL`` same-directory temp file.

    A canonical or reserved destination must lie under ``ROOT`` without parent traversal (``..``) or any
    symlink along the path or its parents below ``ROOT``; ``canonical_dest`` also binds it to
    ``CANONICAL_OUTPUT`` of this checkout. Failures before ``os.replace`` (JSON serialization, write,
    file ``fsync``, round-trip check, or ``os.replace`` itself) leave the previous destination untouched
    with zero staging files; a post-rename directory ``os.fsync(dir_fd)`` failure raises after the new
    file is already visible (leaving crash durability uncertain), also with zero staging files.
    """
    path = Path(path)
    guarded = canonical_dest or reserved(path)
    relative = None
    if guarded:
        relative = _guarded_relative(path, canonical_dest=canonical_dest)
        pre = _open_verified_parent_fd(path, relative, create=False)
        if pre is not None:
            os.close(pre[0])
    text = e2e._dumps_json(payload) + "\n"
    expected = json.loads(json.dumps(payload, allow_nan=False))
    if guarded:
        dir_fd, filename = _open_verified_parent_fd(path, relative, create=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        dir_fd = os.open(path.parent, _DIR_OPEN_FLAGS)
        filename = path.name
    tmp_name = None
    try:
        fd, tmp_name = _create_temp_at(dir_fd, filename)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        rfd = os.open(tmp_name, os.O_RDONLY | _NOFOLLOW_FLAGS, dir_fd=dir_fd)
        with os.fdopen(rfd, "r", encoding="utf-8") as handle:
            reloaded = json.load(handle)
        if reloaded != expected:
            raise ValueError("JSON round trip is not lossless")
        if guarded:
            _check_not_symlink(filename, dir_fd, path, path)
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


def _is_driver_script(arg: str) -> bool:
    return arg == SCRIPT_REPO_PATH


def _parsed_command(command, label: str) -> argparse.Namespace:
    """``command`` as a runnable invocation of this script, parsed by its own CLI (no abbreviations)."""
    _require(isinstance(command, list) and len(command) >= 2 and all(isinstance(x, str) and x for x in command)
             and c502._is_python_executable(command[0]), f"{label} must be a Python invocation")
    _require(_is_driver_script(command[1]), f"{label} is not an invocation of {SCRIPT_REPO_PATH}")
    err = io.StringIO()
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            return parse_args(command[2:])
    except SystemExit as exc:
        raise ValueError(f"{label} is not a valid runnable invocation: {err.getvalue().strip()}") from exc


NUMERICS = {"allow_tf32": False, "cudnn_allow_tf32": False, "allow_fp16_reduced_precision_reduction": False,
            "allow_bf16_reduced_precision_reduction": False}


def set_numerics() -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return dict(NUMERICS)


# ======================================================================================
# Driver
# ======================================================================================
def run(args) -> dict:
    protocol = copy.deepcopy(args.protocol)
    device = torch.device(protocol["device"])
    git = h.git_info()
    require_canonical_host(device, git, args.output, args.canonical)  # before any GPU work
    if device.type != "cuda" or not c4.hardware_w4a4_available(device):
        raise RuntimeError("C5-03 needs a CUDA device with native NVFP4 scaled_mm (sm_100+)")
    numerics = set_numerics()
    log = (lambda *a, **k: None) if args.silent else (lambda *a, **k: print(*a, **k, flush=True))
    started, timestamp_start = time.perf_counter(), time.strftime("%Y-%m-%dT%H:%M:%S%z")
    env = q.environment(device)
    code = code_sha256()
    shards = None
    data = None
    if "C" in protocol["parts"]:
        # Startup verification reads all three shard files (SHA-256, token ranges, document boundaries); no
        # model is scored there. ``shards.opened`` records the later ``open`` calls (train, val, then test).
        shards = c502.Shards(args.data_dir, verify=not args.no_verify,
                             expected=None if args.no_verify else q.EXPECTED_SHA256)
        summary = shards.summary()
        data = {**summary, "directory": str(args.data_dir), "sha256_verified": not args.no_verify,
                "documents": {k: v.get("documents") for k, v in shards.manifest["splits"].items()},
                "startup_verification": "SHA-256 of every shard file and the manifest token / document checks "
                                        "(real_text_data.verify_manifest), before and apart from any scoring",
                "roles": {"training": "train", "calibration": "train", "static_scales": "train",
                          "awq_alpha_selection": "val", "method_selection": "val", "evaluation": "test"}}
    data_sha = {k: v["sha256"] for k, v in data["splits"].items()} if data else {}
    cfg_hash = run_config_hash(args.quick, protocol, code, git["git_sha"], env, data_sha)
    # The effective invocation (equal to sys.argv when the script runs from the command line).
    meta = {**git, "command": _invocation(args.argv),
            "argv": list(args.argv), "environment": env,
            "numerics": numerics, "code_sha256": code, "code_modules": list(code_modules()),
            "timestamp_start": timestamp_start, "loadavg_start": list(os.getloadavg()),
            "gpu_telemetry": {"start": h.gpu_telemetry(device)}, "part_seconds": {}}
    log(f"[c5-03] git {git['git_sha']} dirty={git['git_dirty']} device={env.get('device')} run {cfg_hash} "
        f"canonical={args.canonical}")
    payload = {"schema_version": SCHEMA_VERSION, "task": TASK, "quick": args.quick, "canonical": args.canonical,
               "meta": meta, "run_config_hash": cfg_hash, "protocol": protocol, "data": data}
    t0 = time.perf_counter()
    payload["kernel_parity"] = kernel_parity(device, args.quick)
    meta["part_seconds"]["kernel_parity"] = time.perf_counter() - t0
    log(f"kernel parity: {payload['kernel_parity']['cases']} cases, passed={payload['kernel_parity']['passed']}")
    flush = torch.empty(protocol["timing"]["l2_flush_multiple"] * torch.cuda.get_device_properties(device).L2_cache_size,
                        dtype=torch.uint8, device=device)
    payload["part_a"] = payload["part_b"] = payload["part_c"] = None
    if "A" in protocol["parts"]:
        t0 = time.perf_counter()
        payload["part_a"] = part_a(device, protocol, flush, log)
        meta["part_seconds"]["part_a"] = time.perf_counter() - t0
        meta["gpu_telemetry"]["after_part_a"] = h.gpu_telemetry(device)
    if "B" in protocol["parts"]:
        t0 = time.perf_counter()
        payload["part_b"] = part_b(device, protocol, flush, log)
        meta["part_seconds"]["part_b"] = time.perf_counter() - t0
        meta["gpu_telemetry"]["after_part_b"] = h.gpu_telemetry(device)
    del flush
    torch.cuda.empty_cache()
    if "C" in protocol["parts"]:
        t0 = time.perf_counter()
        study = Study(protocol, shards, device, args.checkpoint_dir, log)
        payload["part_c"] = {"protocol": protocol["part_c"], **study.run()}
        payload["part_c"]["analysis"] = analyze(payload["part_c"])
        meta["part_seconds"]["part_c"] = time.perf_counter() - t0
        meta["gpu_telemetry"]["after_part_c"] = h.gpu_telemetry(device)
        meta["splits_opened"] = list(shards.opened)
    else:
        meta["splits_opened"] = []
    payload["postprocess"] = None  # this run measured; ``postprocess_from_git`` fills it for a v1 source
    payload["summary"] = summarize(payload)
    payload["caveats"] = caveats(payload)
    meta["gpu_telemetry"]["end"] = h.gpu_telemetry(device)
    meta["loadavg_end"] = list(os.getloadavg())
    meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["total_seconds"] = time.perf_counter() - started
    payload = json.loads(json.dumps(payload, allow_nan=False))
    try:
        validate_results(payload, canonical=args.canonical)
    except Exception:
        # Keep the measurements of a long run: the rejected payload goes next to the output, never to it.
        _write_json_atomic(rejected_path(args.output), payload)
        raise
    return payload


def _invocation(argv: list[str]) -> list[str]:
    """The effective command line (equal to sys.argv when the script runs from the command line)."""
    return [sys.executable, str(Path(__file__).resolve().relative_to(ROOT)), *argv]


def rejected_path(output: Path) -> Path:
    return Path(output).with_name(Path(output).name + ".rejected")


def _with_summaries(payload: dict) -> dict:
    """``part_c.analysis``, ``summary`` and ``caveats`` of the raw measurements, as plain JSON types."""
    if payload.get("part_c") is not None:
        payload["part_c"]["analysis"] = analyze(payload["part_c"])
    payload["summary"] = summarize(payload)
    payload["caveats"] = caveats(payload)
    return json.loads(json.dumps(payload, allow_nan=False))


def postprocess_from_git(commit: str, command: list[str]) -> dict:
    """The v2 payload of the committed v1 result of ``commit``. Deterministic apart from its timestamp.

    The raw measurements (``raw_measurements``) are the parsed bytes of ``git show`` and are not changed.
    Only ``part_c.analysis``, ``summary`` and ``caveats`` are recomputed. ``postprocess`` records the
    source, the postprocessing code and the pinned digests of the retained run evidence.
    """
    _require_git_commit(commit, "--postprocess-from-git")
    blob = _git_blob(commit, RESULT_REPO_PATH)
    source = json.loads(blob)
    if not isinstance(source, dict) or source.get("schema_version") != SOURCE_SCHEMA_VERSION \
            or source.get("task") != TASK or any(key not in source for key in RAW_MEASUREMENT_KEYS):
        raise ValueError(f"{RESULT_REPO_PATH} at {commit} is not a {SOURCE_SCHEMA_VERSION} result of this task")
    git = h.git_info()
    payload = {"schema_version": SCHEMA_VERSION, "task": TASK,
               **copy.deepcopy({key: source[key] for key in RAW_MEASUREMENT_KEYS})}
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
    """``--postprocess-from-git``: no GPU, no training, no scoring.

    A reserved output needs a clean tree and the pinned canonical source, both checked before any work,
    and the payload must then certify as canonical.
    """
    if reserved(args.output):
        require_clean_tree(h.git_info(), args.output)
        if args.postprocess_from_git not in CANONICAL_SOURCE_ARTIFACTS:
            raise ValueError(f"{args.output.name} is postprocessed only from the pinned source "
                             f"{', '.join(CANONICAL_SOURCE_ARTIFACTS)}")
    payload = postprocess_from_git(args.postprocess_from_git, _invocation(args.argv))
    want_canonical = True if reserved(args.output) else payload["canonical"]
    cert = validate_results(payload, canonical=want_canonical)
    if reserved(args.output) and not cert["certified_canonical"]:
        raise ValueError(f"Refusing to write noncanonical postprocessed result to reserved {args.output.name}")
    return payload


# ======================================================================================
# Validation
# ======================================================================================
_require, _same, _assert_close = c502._require, c502._same, c502._assert_close
_is_number, _nonneg, _is_int = c502._is_number, c502._nonneg, c502._is_int

TOP_KEYS = {"schema_version", "task", "quick", "canonical", "meta", "run_config_hash", "protocol", "data",
            "kernel_parity", "part_a", "part_b", "part_c", "postprocess", "summary", "caveats"}
META_KEYS = {"git_sha", "git_dirty", "command", "argv", "environment", "numerics", "code_sha256", "code_modules",
             "timestamp_start", "timestamp_end", "loadavg_start", "loadavg_end", "gpu_telemetry", "part_seconds",
             "total_seconds", "splits_opened"}
ROW_KEYS = set(_row("rtn", "w4a16_fakequant")) | {f"{s}_{k}" for s in SPLITS for k in (
    "nll", "ppl", "tokens", "delta_nll", "relative_delta_ppl_pct")} | {f"seconds_{s}" for s in SPLITS}
COPY_FIELDS = ("method", "awq", "awq_alpha", "inherited")  # the fields an alpha = 0 copy changes
BUTTERFLY_FIT_KEYS = {"steps", "lr", "layers", "layers_improved", "weight_sq_error_hadamard_start",
                      "weight_sq_error_fitted", "relative_error_reduction"}


def _validate_protocol(protocol) -> None:
    template = CANONICAL
    free = {("part_c", "max_eval_windows")}
    _require(isinstance(protocol, dict) and set(protocol) == set(template), "protocol keys differ from the schema")
    for key in ("device", "parts", "seed"):
        _require(type(protocol[key]) is type(template[key]), f"protocol.{key} type differs")
    _require(isinstance(protocol["parts"], str) and protocol["parts"] and set(protocol["parts"]) <= set("ABC")
             and "".join(sorted(set(protocol["parts"]))) == protocol["parts"], "protocol.parts invalid")
    for key in ("timing", "part_a", "part_b", "part_c"):
        _require(isinstance(protocol[key], dict) and set(protocol[key]) == set(template[key]),
                 f"protocol.{key} keys differ")
    spec = protocol["part_c"]
    _require(all(type(spec[k]) is type(template["part_c"][k]) for k in spec if ("part_c", k) not in free),
             "protocol.part_c value types differ")
    _require(spec["max_eval_windows"] is None or _is_int(spec["max_eval_windows"], 1), "max_eval_windows invalid")
    seeds = spec["seeds"]
    _require(seeds and all(type(s) is int for s in seeds) and len(set(seeds)) == len(seeds),
             "protocol.part_c.seeds must be distinct integers")
    _require(all(_is_int(v, 1) for v in protocol["timing"].values()), "protocol.timing values must be positive ints")
    for shape in (*protocol["part_a"]["shapes"], *protocol["part_a"]["ablation_shapes"]):
        _require(isinstance(shape, list) and len(shape) == 3 and all(_is_int(v, 1) for v in shape),
                 "Part A shapes must be [M, N, K] integers")
    _require(all(name in PART_B_CONFIGS and _is_int(b, 1) for name, b in protocol["part_b"]["models"].items()),
             "Part B models invalid")


def _validate_meta(payload: dict, canonical: bool, schema: str) -> None:
    """``schema`` is the schema version of the measuring run: ``run_config_hash`` was made under it."""
    meta = payload["meta"]
    _require(isinstance(meta, dict) and set(meta) == META_KEYS, "meta keys differ from the schema")
    _require(meta["git_sha"] is None or _is_hex(meta["git_sha"], 40), "meta.git_sha must be 40-hex or None")
    if meta["git_sha"] is not None:
        _require_git_commit(meta["git_sha"], "meta.git_sha")
    _require(meta["git_dirty"] is None or type(meta["git_dirty"]) is bool, "meta.git_dirty must be a bool")
    _require(_same(meta["numerics"], NUMERICS), "full-precision numerics are required")
    command, argv = meta["command"], meta["argv"]
    _require(isinstance(argv, list) and isinstance(command, list) and argv == command[2:],
             "meta.argv must be the arguments of meta.command")
    parsed = _parsed_command(command, "meta.command")
    _require(parsed.postprocess_from_git is None, "meta.command must be a measuring invocation")
    _require(parsed.canonical == payload["canonical"] and _same(parsed.protocol, payload["protocol"]),
             "meta.argv does not select the recorded protocol and canonical flag")
    modules = meta["code_modules"]
    _require(isinstance(modules, list) and all(isinstance(m, str) and m.startswith("research.") for m in modules)
             and len(set(modules)) == len(modules) and DRIVER_MODULE in modules,
             "meta.code_modules must list the driver and its research.* imports")
    measuring = payload["postprocess"] is None
    if measuring:
        _require(modules == list(code_modules()), "meta.code_modules differ from the static import closure of this driver")
    _require(_is_hex(meta["code_sha256"], 64), "meta.code_sha256 must be 64-hex")
    env = meta["environment"]
    _require(isinstance(env, dict) and all(k in env for k in ENV_KEYS), "meta.environment incomplete")
    _require(_nonneg(meta["total_seconds"]) and all(_nonneg(v) for v in meta["part_seconds"].values())
             and meta["total_seconds"] >= sum(meta["part_seconds"].values()) - 1e-6, "meta durations invalid")
    start = c502._timestamp(meta["timestamp_start"], "meta.timestamp_start")
    end = c502._timestamp(meta["timestamp_end"], "meta.timestamp_end")
    _require(end >= start and abs((end - start).total_seconds() - meta["total_seconds"]) <= 5.0,
             "meta timestamps are inconsistent with meta.total_seconds")
    if meta["git_sha"] is not None:
        _require(start >= _commit_timestamp(meta["git_sha"]),
                 "meta.timestamp_start is before the producer commit meta.git_sha")
    for field in ("loadavg_start", "loadavg_end"):
        _require(isinstance(meta[field], list) and len(meta[field]) == 3 and all(_nonneg(x) for x in meta[field]),
                 f"meta.{field} invalid")
    want_splits = ["train", "val", "test"] if "C" in payload["protocol"]["parts"] else []
    _require(meta["splits_opened"] == want_splits, "meta.splits_opened must be train, val, then test (Part C)")
    if canonical:
        _verify_canonical_output_path(parsed.output)
        _require(_is_hex(meta["git_sha"], 40) and meta["git_dirty"] is False,
                 "a canonical result needs a 40-hex git_sha and a clean tree (git_dirty False)")
        _require(not {a.split("=", 1)[0] for a in argv} & set(RAW_NONCANONICAL_FLAGS),
                 "meta.argv carries a non-canonical flag")
        _require(tuple(env["capability"]) >= (12, 0), "a canonical result needs an sm_120+ device")
    if canonical or not measuring:  # the producer code: the import closure and blobs at meta.git_sha
        _require(_is_hex(meta["git_sha"], 40), "the producer needs a 40-hex meta.git_sha")
        _require(modules == list(_code_modules_at_commit(meta["git_sha"])),
                 "meta.code_modules differ from the import closure at meta.git_sha")
        _require(meta["code_sha256"] == _code_sha256_at_commit(meta["git_sha"], tuple(modules)),
                 "meta.code_sha256 differs from the committed git blobs")
        commit_schema, _ = _driver_capabilities_at_commit(meta["git_sha"])
        _require(commit_schema == schema, f"meta.git_sha declares schema {commit_schema!r}, not {schema!r}")
    if measuring:
        if canonical:  # the credited commit computed summary and caveats: its summary code is this checkout's
            _require(_matches_current_summary_code(meta["git_sha"]),
                     "meta.git_sha summary code differs from the validating checkout")
        else:
            _require(meta["code_sha256"] == code_sha256(), "meta.code_sha256 differs from this checkout")
    data_sha = {k: v["sha256"] for k, v in payload["data"]["splits"].items()} if payload["data"] else {}
    _require(payload["run_config_hash"] == run_config_hash(payload["quick"], payload["protocol"], meta["code_sha256"],
                                                           meta["git_sha"], env, data_sha, schema),
             "run_config_hash differs")


def _validate_dist(dist: dict, rounds: int, per_round: int | None) -> None:
    samples = dist.get("samples_ms")
    _require(isinstance(samples, list) and len(samples) == rounds == dist["rounds"]
             and all(_nonneg(v) for v in samples), "samples_ms missing or of the wrong length")
    if per_round is not None:
        events = dist.get("event_samples_ms")
        _require(isinstance(events, list) and len(events) == rounds, "event_samples_ms missing")
        for ev, mean_ms in zip(events, samples):
            _require(isinstance(ev, list) and len(ev) == per_round and all(_nonneg(v) for v in ev)
                     and abs(sum(ev) / len(ev) - mean_ms) <= 1e-12, "round mean differs from event_samples_ms")
    else:
        _require("event_samples_ms" not in dist, "wall timings have no event samples")
    q1, mid, q3 = np.quantile(samples, [0.25, 0.5, 0.75])
    for key, want in (("median_ms", mid), ("iqr_ms", q3 - q1), ("min_ms", min(samples)), ("max_ms", max(samples)),
                      ("mean_ms", np.mean(samples))):
        _require(abs(dist[key] - float(want)) <= 1e-12, f"timing {key} differs from samples_ms")


def _validate_timing(timing: dict, orders, methods, squares: int, iters: int, wall_iters: int, where: str) -> None:
    square = e2e.williams_square(len(methods))
    expected = [[methods[i] for i in square[r % len(square)]] for r in range(squares * len(square))]
    _require(orders == expected, f"{where}: Williams orders differ from the balanced design")
    _require(list(timing) == list(methods), f"{where}: timed methods differ")
    for name, t in timing.items():
        _require(set(t) == {"gpu", "wall"}, f"{where}.{name}: timing keys")
        _validate_dist(t["gpu"], len(orders), iters)
        _validate_dist(t["wall"], wall_iters * len(orders), None)


COOP_KEYS = {"grid", "graph_replays", "graph_replays_bitwise_exact", "sync_counter_after_replays",
             "launch_cooperative_grid", "enabled_in_nvfp4_inference_blackwell"}
COOP_GRID_KEYS = {"programs", "max_active_blocks_per_sm", "sms", "resident_capacity"}
GEMM_KERNEL_MARKERS = ("gemm", "cutlass", "nvjet", "cublas")
FP4_KERNEL_MARKERS = ("e2m1", "nvfp4", "fp4")
QUANT_KERNEL_NAMES = {
    "c4_02": (_c4_02_amax_kernel.fn.__name__, _c4_02_quant_kernel.fn.__name__),
    "c5_03": (nb._amax_kernel.fn.__name__, nb._quant_kernel.fn.__name__),
}
QUANT_ROLE_MARKERS = ("amax", "quant", "_c4_02_", "_coop_")


def _is_gemm_kernel(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in GEMM_KERNEL_MARKERS)


def _is_fp4_gemm_kernel(name: str) -> bool:
    low = name.lower()
    return _is_gemm_kernel(name) and any(m in low for m in FP4_KERNEL_MARKERS)


def _validate_kernel_profile(method: str, kernels, where: str) -> None:
    """CUPTI kernels of one graph replay: finite nonnegative durations, FP4-specific GEMM and exact quantizer names."""
    _require(isinstance(kernels, dict) and kernels and all(isinstance(k, str) and k and _nonneg(us)
                                                           for k, us in kernels.items()),
             f"{where}: kernel durations must be finite nonnegative microseconds per named kernel")
    fp4_gemms = [k for k in kernels if _is_fp4_gemm_kernel(k)]
    non_fp4_gemms = [k for k in kernels if _is_gemm_kernel(k) and not _is_fp4_gemm_kernel(k)]
    _require(bool(fp4_gemms), f"{where}: no FP4 GEMM kernel")
    _require(not non_fp4_gemms, f"{where}: unexpected non-FP4 GEMM kernel {non_fp4_gemms}")
    quant_like = {k for k in kernels if any(m in k.lower() for m in QUANT_ROLE_MARKERS)}
    if method == "nvfp4_gemm_only":
        _require(not quant_like, f"{where}: the GEMM ceiling runs no quantizer kernel, found {sorted(quant_like)}")
        return
    impl = next(i for i in IMPLS if method.startswith(f"{i}_"))
    static = method.endswith("static_gs")
    amax_name, quant_name = QUANT_KERNEL_NAMES[impl]
    expected = {quant_name} if static else {amax_name, quant_name}
    _require(quant_like == expected,
             f"{where}: a {'static' if static else 'dynamic'} {impl} graph must run exact quantizer "
             f"kernel(s) {sorted(expected)} of its own implementation, found {sorted(quant_like)}")


def _validate_part_a(payload: dict) -> None:
    protocol, a = payload["protocol"], payload["part_a"]
    t = protocol["timing"]
    _require(a["l2_flush_bytes"] == t["l2_flush_multiple"] * (64 << 20) or not payload["canonical"],
             "the canonical L2 flush is 4 x 64 MB")
    _require([[r["m"], r["n"], r["k"]] for r in a["shapes"]] == protocol["part_a"]["shapes"], "Part A shapes differ")
    for r in a["shapes"]:
        where = f"part_a ({r['m']},{r['n']},{r['k']})"
        _require(r["checks"] and all(v is True for v in r["checks"].values()) and len(r["checks"]) == 3 * len(VARIANTS),
                 f"{where}: a bit-exactness check failed")
        for mode, key in (("eager", "eager_williams_orders"), ("cuda_graph", "cuda_graph_williams_orders")):
            _validate_timing(r[mode], r[key], list(PART_A_METHODS), t["squares_a"], t["iters"], t["wall_iters_a"],
                             f"{where}.{mode}")
        _assert_close(r["summary"], summarize_part_a_row(r), f"{where}.summary")
        _require(_same(r["c5_03_plan"], {v: nb.quantizer_plan(r["m"], r["k"], VARIANT_ROTATION[v], v.endswith("static_gs"))
                                         for v in VARIANTS}), f"{where}: c5_03_plan differs from the module")
        profile = r["cuda_graph_kernel_profile_us"]
        _require(isinstance(profile, dict) and set(profile) == set(PART_A_METHODS) - {"bf16_cublas"},
                 f"{where}: kernel profile incomplete")
        for method, kernels in profile.items():
            _validate_kernel_profile(method, kernels, f"{where}.cuda_graph_kernel_profile_us.{method}")
    _require([[x["m"], x["n"], x["k"]] for x in a["ablation"]] == protocol["part_a"]["ablation_shapes"],
             "Part A ablation shapes differ")
    for x in a["ablation"]:
        where = f"part_a.ablation ({x['m']},{x['n']},{x['k']})"
        _require(all(v is True for v in x["checks"].values()) and len(x["checks"]) == 3 * len(ABLATION_VARIANTS),
                 f"{where}: an ablation variant is not bit-exact")
        _validate_timing(x["timing"], x["williams_orders"], list(ABLATION_METHODS), t["squares_a"], t["iters"],
                         t["wall_iters_a"], where)
        _assert_close(x["summary"], _timing_table(x["timing"], "bf16_cublas"), f"{where}.summary")
        coop = x["coop"]
        _require(isinstance(coop, dict) and set(coop) == COOP_KEYS, f"{where}: cooperative record keys differ")
        _require(coop["launch_cooperative_grid"] is True, f"{where}: the cooperative kernel needs a cooperative launch")
        grid = coop["grid"]
        _require(isinstance(grid, dict) and set(grid) == COOP_GRID_KEYS and all(_is_int(v, 1) for v in grid.values())
                 and grid["resident_capacity"] == grid["sms"] * grid["max_active_blocks_per_sm"]
                 and 1 <= grid["programs"] <= grid["resident_capacity"],
                 f"{where}: the cooperative grid must be positive and fit its resident capacity "
                 "(sms x max_active_blocks_per_sm)")
        _require(coop["graph_replays"] == protocol["part_a"]["coop_replays"]
                 and coop["graph_replays_bitwise_exact"] == coop["graph_replays"]
                 and coop["sync_counter_after_replays"] == 0 and coop["enabled_in_nvfp4_inference_blackwell"] is False,
                 f"{where}: cooperative replays must be exact, reset the barrier and stay disabled")


def _validate_part_b(payload: dict) -> None:
    protocol, b = payload["protocol"], payload["part_b"]
    t = protocol["timing"]
    _require(list(b["models"]) == list(protocol["part_b"]["models"]), "Part B models differ")
    for name, m in b["models"].items():
        where = f"part_b.{name}"
        _require(_same(m["config"], asdict(PART_B_CONFIGS[name])) and m["batch"] == protocol["part_b"]["models"][name]
                 and m["tokens_per_call"] == m["batch"] * m["config"]["sequence_length"], f"{where}: config differs")
        _require(m["nvfp4_layers"] == 5 * m["config"]["layers"], f"{where}: every block linear must be NVFP4")
        cfg = m["config"]
        linear = cfg["layers"] * (4 * cfg["width"] * cfg["width"] + 3 * cfg["width"] * cfg["hidden"])
        _require(_is_int(m.get("linear_parameters"), 1) and m["linear_parameters"] == linear,
                 f"{where}: linear_parameters must be the {linear} weights of the qkv, out, gate, up and down linears")
        _require(_nonneg(m.get("compile_seconds")), f"{where}: compile_seconds invalid")
        for v in VARIANTS:
            _require(m["checks"][f"{v}_c5_03_logits_equal_c4_02_bitwise"] is True, f"{where}: {v} logits differ")
            _require(_nonneg(m["checks"][f"{v}_logits_rel_diff_vs_bf16"]), f"{where}: {v} rel diff invalid")
        _validate_timing(m["timing"], m["williams_orders"], list(PART_B_METHODS), t["squares_b"], t["iters"],
                         t["wall_iters_b"], where)
        _assert_close(m["summary"], summarize_part_b_model(m), f"{where}.summary")


def _expected_grid() -> list[tuple[str, str]]:
    out = []
    for method in METHODS:
        out += [(method, mode) for mode in MODES]
        if method == EMULATED_METHOD:
            out.append((method, EMULATED_MODE))
    return out


CALIBRATION_KEYS = {"split", "seed", "sequences", "seq_len", "tokens", "starts"}


def _validate_calibration(cal, spec: dict, seq_len: int, train_tokens: int) -> None:
    """The record of ``q.calibration_windows``: the exact seeded draw of distinct, aligned train.bin windows."""
    _require(isinstance(cal, dict) and set(cal) == CALIBRATION_KEYS, "calibration record keys differ from the schema")
    available = train_tokens // seq_len
    _require(cal["split"] == "train" and _same(cal["sequences"], spec["calibration_sequences"])
             and _same(cal["seed"], spec["calibration_seed"]) and _same(cal["seq_len"], seq_len)
             and _is_int(cal["sequences"], 1) and cal["sequences"] <= available
             and _same(cal["tokens"], cal["sequences"] * cal["seq_len"]),
             "calibration must be the train.bin draw of the protocol (sequences x seq_len tokens)")
    rng = np.random.default_rng(cal["seed"])
    expected = [int(s) for s in np.sort(rng.choice(available, size=cal["sequences"], replace=False)) * seq_len]
    _require(_same(cal["starts"], expected), "calibration starts differ from the seeded train.bin draw of the protocol")


def _validate_part_c(payload: dict) -> None:
    protocol, c, data = payload["protocol"], payload["part_c"], payload["data"]
    spec, canonical = protocol["part_c"], payload["canonical"]
    config = h.PRESETS[spec["preset"]]
    n_layers = config.layers * len(q.LINEAR_KINDS)
    _require(_same(c["protocol"], spec) and c["preset"] == spec["preset"] and _same(c["config"], asdict(config)),
             "part_c protocol or config differs")
    _require(data is not None and type(data["sha256_verified"]) is bool, "data record missing")
    if canonical:
        _require(data["sha256_verified"] is True and {k: v["sha256"] for k, v in data["splits"].items()}
                 == q.EXPECTED_SHA256, "canonical shards must be SHA-256 verified against the pinned hashes")
        _require({k: v["tokens"] for k, v in data["splits"].items()} == c502.CANONICAL_DATA["tokens"],
                 "canonical split token counts differ")
    _require(c["splits_opened_in_phase1"] == ["train", "val"], "Phase 1 must map only train.bin and val.bin")
    split_tokens = {k: v["tokens"] for k, v in data["splits"].items()}
    _validate_calibration(c["calibration"], spec, config.sequence_length, split_tokens["train"])
    seeds = c["seeds"]
    _require([s["seed"] for s in seeds] == spec["seeds"], "part_c seeds differ from the protocol")
    coverage = {split: c502.expected_coverage(split_tokens[split], config.sequence_length, spec["max_eval_windows"])
                for split in SPLITS}
    if canonical:
        _require(coverage["val"]["tokens"] == VAL_TOKENS and coverage["test"]["tokens"] == TEST_TOKENS,
                 "canonical scoring must cover the full val.bin and test.bin")
    val_scorings = test_scorings = 0
    for s in seeds:
        where = f"part_c seed {s['seed']}"
        tr = s["training"]
        tokens = spec["steps"] * config.step_tokens
        _require(tr["seed"] == s["seed"] and tr["optimizer"] == spec["optimizer"] and tr["steps"] == spec["steps"]
                 and tr["warmup_steps"] == spec["warmup_steps"] and tr["lr"] == spec["lr"]
                 and tr["auxiliary_lr"] == spec["auxiliary_lr"] and tr["weight_decay"] == spec["weight_decay"]
                 and tr["min_lr_ratio"] == spec["min_lr_ratio"] and tr["tokens"] == tokens
                 and tr["step_tokens"] == config.step_tokens, f"{where}: training record differs from the protocol")
        _assert_close(tr["train_epochs"], tokens / split_tokens["train"], f"{where}.train_epochs")
        _require(tr["checkpoint_file"] == checkpoint_name(s["seed"], spec) and _is_hex(tr["checkpoint_sha256"], 64)
                 and _is_hex(tr["checkpoint_file_sha256"], 64) and tr["train_curve"][-1][0] == spec["steps"]
                 and math.isfinite(tr["last_train_loss"]) and tr["train_curve"][-1][1] == tr["last_train_loss"],
                 f"{where}: checkpoint or training curve record invalid")
        for split in SPLITS:
            base = s[f"baseline_{split}"]
            _require(all(base[k] == v for k, v in coverage[split].items()), f"{where}: {split} coverage differs")
            _assert_close(base["ppl"], math.exp(base["nll"]), f"{where}.baseline_{split}.ppl")
        _require(s["bf16_fp4_gemms_per_forward"] == 0, f"{where}: the BF16 model must run no FP4 GEMM")
        grid = s["grid"]
        _require([(r["method"], r["mode"]) for r in grid] == _expected_grid(), f"{where}: grid rows differ")
        rows = _grid(s)
        for r in grid:
            rw = f"{where} {r['method']}/{r['mode']}"
            _require(set(r) == ROW_KEYS, f"{rw}: row keys differ")
            family, rounding, awq = parse_method(r["method"])
            _require((r["rotation"], r["rounding"], r["awq"]) == (family, rounding, awq), f"{rw}: method fields")
            if awq:
                _require(r["awq_alpha"] in spec["awq_alphas"] and type(r["awq_alpha"]) is float,
                         f"{rw}: awq_alpha must be one of the protocol alphas")
            else:
                _require(r["awq_alpha"] is None and r["inherited"] is False,
                         f"{rw}: a row without AWQ has no awq_alpha and is not inherited")
            _require(all(_nonneg(r[f"seconds_{split}"]) for split in SPLITS), f"{rw}: seconds_val / seconds_test invalid")
            for split in SPLITS:
                base = s[f"baseline_{split}"]
                _require(r[f"{split}_tokens"] == base["tokens"], f"{rw}: {split} tokens differ from the baseline")
                _assert_close(r[f"{split}_ppl"], math.exp(r[f"{split}_nll"]), f"{rw}.{split}_ppl")
                _assert_close(r[f"{split}_delta_nll"], r[f"{split}_nll"] - base["nll"], f"{rw}.{split}_delta_nll")
                _assert_close(r[f"{split}_relative_delta_ppl_pct"], 100.0 * (r[f"{split}_ppl"] / base["ppl"] - 1.0),
                              f"{rw}.{split}_relative_delta_ppl_pct")
            if r["mode"] in HW_MODES:
                _require(r["fp4_gemms_per_forward"] == n_layers and r["nvfp4_modules"] == n_layers
                         and r["bf16_block_linears"] == 0 and r["emulated_modules"] == 0,
                         f"{rw}: every block linear must be a native FP4 GEMM ({n_layers} per forward)")
            elif r["mode"] == EMULATED_MODE:
                _require(r["fp4_gemms_per_forward"] == 0 and r["emulated_modules"] == n_layers
                         and r["bf16_block_linears"] == 0, f"{rw}: the emulation must run no FP4 GEMM")
            else:
                _require(r["fp4_gemms_per_forward"] is None and r["hw_weights_checked_matrices"] == n_layers
                         and r["hw_weights_decode_exact_matrices"] == n_layers
                         and _is_hex(r["effective_weights_sha256"], 64)
                         and r["effective_weights_sha256_test"] == r["effective_weights_sha256"],
                         f"{rw}: weights must decode exactly ({n_layers}/{n_layers}) and be rebuilt identically")
            static = r["mode"] in ("w4a4_hw_static_bf16cal", "w4a4_hw_static_seqcal")
            _require((r["static_scale_calibration"] == (STATIC_BF16CAL if r["mode"].endswith("bf16cal") else STATIC_SEQCAL))
                     if static else r["static_scale_calibration"] is None, f"{rw}: static calibration record")
            _require((isinstance(r["static_global_scale_range"], list) and len(r["static_global_scale_range"]) == 2
                      and 0 < r["static_global_scale_range"][0] <= r["static_global_scale_range"][1])
                     if static else r["static_global_scale_range"] is None, f"{rw}: static scale range")
        selections = s["awq_selection"]
        _require([a["method"] for a in selections] == list(AWQ_BASE), f"{where}: one AWQ selection per AWQ method")
        val_scorings += 1 + len(BASE_METHODS) * len(MODES) + 1
        test_scorings += 1
        for a in selections:
            base = AWQ_BASE[a["method"]]
            _require(a["base_method"] == base and a["alphas"] == spec["awq_alphas"]
                     and a["selected_on"] == "val.bin, w4a4_hw_dynamic", f"{where}: AWQ selection record")
            _require(a["val_nll_hw_dynamic"][a["alphas"].index(0.0)] == rows[(base, "w4a4_hw_dynamic")]["val_nll"],
                     f"{where}: the alpha = 0 candidate is the plain {base} hardware row")
            _require(a["selected_alpha"] == c502.select_alpha(a["alphas"], a["val_nll_hw_dynamic"]),
                     f"{where}: the selected alpha is not the val.bin minimum")
            val_scorings += len(a["alphas"]) - 1
            for mode in MODES:
                r = rows[(a["method"], mode)]
                _require(r["awq_alpha"] == a["selected_alpha"], f"{where}: AWQ row alpha differs from the selection")
                if a["selected_alpha"] == 0:
                    twin = rows[(base, mode)]
                    _require(r["inherited"] is True and all(_same(r[k], twin[k]) for k in r if k not in COPY_FIELDS),
                             f"{where}: an alpha = 0 AWQ row must equal its twin in every measured field")
                else:
                    _require(r["inherited"] is False, f"{where}: a measured AWQ row is not inherited")
                    test_scorings += 1
                    val_scorings += mode != "w4a4_hw_dynamic"
                    if mode == "w4a4_hw_dynamic":
                        _require(r["val_nll"] == a["val_nll_hw_dynamic"][a["alphas"].index(a["selected_alpha"])],
                                 f"{where}: the selected AWQ hardware row is its selection score")
        _require(all(not r["inherited"] for r in grid if not r["awq"]), f"{where}: only AWQ rows can be inherited")
        test_scorings += len(BASE_METHODS) * len(MODES) + 1
        expected_fits = {"adaptive_butterfly_b16_gptq"} | {
            a["method"] for a in selections if "butterfly" in a["method"] and a["selected_alpha"] != 0}
        _require(set(s["butterfly_fits"]) == expected_fits, f"{where}: butterfly fits differ from the fitted candidates")
        for key, kinds in s["butterfly_fits"].items():
            _require(list(kinds) == list(q.LINEAR_KINDS), f"{where}: one butterfly fit per projection kind")
            for kind, fit in kinds.items():
                fw = f"{where}.butterfly_fits[{key}].{kind}"
                _require(set(fit) == BUTTERFLY_FIT_KEYS and fit["steps"] == spec["butterfly_steps"]
                         and fit["layers"] == config.layers and _is_int(fit["layers_improved"])
                         and fit["layers_improved"] <= config.layers, f"{fw}: fit record")
                start, fitted = fit["weight_sq_error_hadamard_start"], fit["weight_sq_error_fitted"]
                _require(_nonneg(start) and _nonneg(fitted) and 0 <= fitted <= start
                         and (fit["layers_improved"] > 0 or fitted == start),
                         f"{fw}: fitted error must satisfy 0 <= fitted <= start, and equal start when "
                         "layers_improved == 0")
                _assert_close(fit["relative_error_reduction"], 1.0 - fitted / max(start, 1e-30),
                              f"{fw}.relative_error_reduction")
    ledger = c["scoring_ledger"]
    _require(_same(ledger, {"val_scorings": val_scorings, "test_scorings": test_scorings,
                            "frozen_after_val_scorings": val_scorings, "test_scored_only_after_freeze": True}),
             "scoring ledger differs: every val.bin scoring must precede the freeze and every test.bin scoring")
    sel = {"awq": {str(s["seed"]): s["awq_selection"] for s in seeds},
           "best_method_by_val": best_by([s["grid"] for s in seeds], "val")}
    _require(_same(c["frozen_selections"], sel)
             and c["frozen_selections_sha256"] == hashlib.sha256(json.dumps(sel, sort_keys=True).encode()).hexdigest(),
             "frozen selections differ from the val.bin results")
    if len(seeds) == N_SEEDS:
        _assert_close(c["analysis"], analyze(c), "part_c.analysis")
        _require(c["analysis"]["hardware_vs_emulation"]["passed"] is True,
                 "hardware W4A4 must match the FP32 emulation within the tolerance")
        _require(c["analysis"]["hardware"]["fp4_gemms_per_forward"] == [n_layers], "FP4 GEMM count differs")
    else:
        _require(c["analysis"] is None, "the paired analysis needs exactly three seeds")


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
             "postprocess.retained_evidence_sha256 must hold the 64-hex digests of the three retained run files")
    _require(post["reason"] == POSTPROCESS_REASON, "postprocess.reason differs from the declared reason")
    c502._timestamp(post["postprocess_timestamp"], "postprocess.postprocess_timestamp")


def _validate_postprocess_provenance(payload: dict, canonical: bool) -> None:
    """Source and code provenance of a postprocessed result, against the local git repository."""
    post, meta = payload["postprocess"], payload["meta"]
    source_sha, post_sha = post["source_result_git_sha"], post["postprocess_git_sha"]
    for label, sha in (("postprocess.source_result_git_sha", source_sha),
                       ("postprocess.postprocess_git_sha", post_sha), ("meta.git_sha", meta["git_sha"])):
        _require_git_commit(sha, label)
    blob = _git_blob(source_sha, RESULT_REPO_PATH)
    _require(hashlib.sha256(blob).hexdigest() == post["source_artifact_sha256"],
             "postprocess.source_artifact_sha256 differs from the committed bytes of the source artifact")
    source = json.loads(blob)
    _require(isinstance(source, dict) and source.get("schema_version") == post["source_schema_version"]
             and source.get("task") == TASK and all(k in source for k in RAW_MEASUREMENT_KEYS),
             "the source artifact has another schema version or task")
    ours, theirs = raw_measurements(payload), raw_measurements(source)
    for key in RAW_MEASUREMENT_KEYS:  # value for value and type for type
        _require(_same(ours[key], theirs[key]),
                 f"{key} differs from the source artifact: a postprocess must not change a raw measurement")
    _require(post["source_raw_measurements_sha256"] == raw_measurements_sha256(source) == raw_measurements_sha256(payload),
             "postprocess.source_raw_measurements_sha256 differs from the raw measurements of the source")
    # The postprocessing code: this code, as committed at postprocess_git_sha.
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
    _require(_matches_current_summary_code(post_sha),
             "postprocess.postprocess_git_sha summary code differs from the validating checkout")
    parsed = _parsed_command(post["postprocess_command"], "postprocess.postprocess_command")
    _require(parsed.postprocess_from_git == source_sha,
             "postprocess.postprocess_command must pass --postprocess-from-git with the source result commit")
    post_ts = c502._timestamp(post["postprocess_timestamp"], "postprocess.postprocess_timestamp")
    end_ts = c502._timestamp(meta["timestamp_end"], "meta.timestamp_end")
    source_commit_ts = _commit_timestamp(source_sha)
    post_commit_ts = _commit_timestamp(post_sha)
    _require(post_ts >= end_ts, "postprocess.postprocess_timestamp is before the end of the measuring run")
    _require(source_commit_ts >= end_ts,
             "postprocess.source_result_git_sha commit predates the end of the measuring run")
    _require(post_ts >= source_commit_ts, "postprocess.postprocess_timestamp is before the source result commit")
    _require(post_ts >= post_commit_ts, "postprocess.postprocess_timestamp is before the postprocessing commit")
    if canonical:
        try:
            _verify_canonical_output_path(parsed.output)
        except ValueError as exc:
            raise ValueError(
                f"postprocess.postprocess_command of a canonical result must target the canonical output path ({exc})"
            ) from exc
        _require(post["postprocess_git_dirty"] is False, "a canonical postprocess needs a clean tree")
        _require(CANONICAL_SOURCE_ARTIFACTS.get(source_sha) == post["source_artifact_sha256"],
                 "the source of a canonical postprocess must be the pinned committed C5-03 result")
        _require(_same(post["retained_evidence_sha256"], RETAINED_EVIDENCE_SHA256),
                 "postprocess.retained_evidence_sha256 differs from the pinned digests of the retained run files")


def validate_results(payload: dict, canonical: bool | None = None) -> dict:
    """Offline check of a C5-03 result. Recomputes every summary, statistic and selection from the raw rows.

    ``canonical`` states what the caller expects (None: the payload's label); a disagreement raises.
    A measuring result (``postprocess`` None) is bound to ``meta.git_sha``: in canonical mode its import
    closure and blobs, its ``SCHEMA_VERSION`` and its ``SUMMARY_MODULES`` (which must equal this checkout).
    A postprocessed result is bound to the committed source artifact (byte digest, raw measurements
    value for value), to the producer at ``meta.git_sha`` (hashes under ``SOURCE_SCHEMA_VERSION``) and to
    the postprocessing code at ``postprocess_git_sha`` (``SUMMARY_MODULES`` equal to this checkout).

    It cannot re-measure: it trusts the recorded timings and NLLs that pass its consistency checks.
    """
    try:
        return _validate_payload(payload, canonical)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:  # a record of the wrong shape or type
        raise ValueError(f"malformed payload: {type(exc).__name__}: {exc}") from exc


def _validate_payload(payload: dict, canonical: bool | None) -> dict:
    json.dumps(payload, allow_nan=False)
    _require(isinstance(payload, dict) and set(payload) == TOP_KEYS, "top-level keys differ from the schema")
    _require(payload["schema_version"] == SCHEMA_VERSION and payload["task"] == TASK, "schema or task differs")
    _require(type(payload["quick"]) is bool and type(payload["canonical"]) is bool, "quick/canonical must be bools")
    _require(canonical is None or type(canonical) is bool, "the requested canonical mode must be a bool or None")
    if canonical is True and not payload["canonical"]:
        raise ValueError("canonical certification was requested, but the payload is not labeled canonical")
    if canonical is False and payload["canonical"]:
        raise ValueError("the payload is labeled canonical: it cannot be validated with the noncanonical checks")
    canonical = payload["canonical"]
    post = payload["postprocess"]
    if post is not None:
        _validate_postprocess_record(payload)
    schema = SCHEMA_VERSION if post is None else post["source_schema_version"]
    protocol = payload["protocol"]
    _validate_protocol(protocol)
    if canonical:
        _require(payload["quick"] is False and _same(protocol, CANONICAL),
                 "the protocol differs from the canonical C5-03 protocol")
    _validate_meta(payload, canonical, schema)
    kp = payload["kernel_parity"]
    want_cases = (54 + 6 if payload["quick"] else 216 + 36) + 27
    _require(kp["passed"] is True and kp["bitwise_failures"] == [] and kp["cases"] == want_cases,
             "kernel parity must pass every case bit for bit")
    for part, check in (("A", _validate_part_a), ("B", _validate_part_b), ("C", _validate_part_c)):
        key = f"part_{part.lower()}"
        if part in protocol["parts"]:
            _require(payload[key] is not None, f"{key} missing")
            check(payload)
        else:
            _require(payload[key] is None, f"{key} present but not in protocol.parts")
    _assert_close(payload["summary"], summarize(payload), "summary")
    if post is not None:
        _validate_postprocess_provenance(payload, canonical)
    _require(payload["caveats"] == caveats(payload), "caveats differ from the recomputed caveats")
    return {"certified_canonical": canonical, "parts": protocol["parts"], "postprocessed": post is not None}


# ======================================================================================
# CLI
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument("--quick", action="store_true", help="tiny shapes and model, short evaluation (never canonical)")
    parser.add_argument("--parts", default=None, help="subset of A, B, C (default ABC; a subset is not canonical)")
    parser.add_argument("--device", default=None, help="default: cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Part C seeds (default 1 2 3)")
    parser.add_argument("--max-eval-windows", type=int, default=None, help="limit val/test windows (not canonical)")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--no-verify", action="store_true", help="skip SHA-256 checks (synthetic shards only)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="where Part C saves the trained state dicts (default: artifacts/c5_03_checkpoints for "
                             "the canonical run, <output>.ckpt otherwise)")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--postprocess-from-git", metavar="COMMIT", default=None,
                        help=f"no GPU work: read the committed {SOURCE_SCHEMA_VERSION} result of COMMIT (full 40-hex "
                             "SHA) with git show, keep its raw measurements unchanged and recompute part_c.analysis, "
                             "summary and caveats")
    return parser


RAW_NONCANONICAL_FLAGS = ("--quick", "--parts", "--device", "--seeds", "--max-eval-windows", "--no-verify")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    args.argv = argv
    if args.postprocess_from_git is not None:
        measuring = [flag for flag, used in (
            ("--quick", args.quick), ("--parts", args.parts is not None), ("--device", args.device is not None),
            ("--seeds", args.seeds is not None), ("--max-eval-windows", args.max_eval_windows is not None),
            ("--data-dir", args.data_dir is not None), ("--no-verify", args.no_verify),
            ("--checkpoint-dir", args.checkpoint_dir is not None)) if used]
        if measuring:
            parser.error(f"--postprocess-from-git measures nothing: remove {' '.join(measuring)}")
        if not _is_hex(args.postprocess_from_git, 40):
            parser.error("--postprocess-from-git needs the full 40-hex SHA of the result commit")
        args.output = CANONICAL_OUTPUT if args.output is None else args.output
        pinned = args.postprocess_from_git in CANONICAL_SOURCE_ARTIFACTS
        if pinned or reserved(args.output):
            try:
                _verify_canonical_output_path(args.output)
            except ValueError as exc:
                parser.error(
                    f"A postprocess writes {CANONICAL_OUTPUT} of this checkout ({exc})"
                    + ("" if pinned else f", or a path outside {RESERVED_PATTERN}")
                )
        args.protocol, args.canonical = None, False  # not a measuring run: it selects no protocol
        args.explicit_data_dir = args.explicit_checkpoint_dir = None
        return args
    protocol = copy.deepcopy(QUICK if args.quick else CANONICAL)
    if args.parts is not None:
        parts = "".join(sorted(set(args.parts.upper())))
        if not parts or not set(parts) <= set("ABC") or len(parts) != len(args.parts):
            parser.error("--parts must be a non-empty subset of A, B, C without repeats")
        protocol["parts"] = parts
    if args.device is not None:
        protocol["device"] = args.device
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds must be distinct")
        protocol["part_c"]["seeds"] = list(args.seeds)
    if args.max_eval_windows is not None:
        if args.max_eval_windows < 1:
            parser.error("--max-eval-windows must be positive")
        protocol["part_c"]["max_eval_windows"] = args.max_eval_windows
    args.protocol = protocol
    # Canonical: the parsed protocol is the canonical one and no raw flag that changes it was given.
    raw_flags = {a.split("=", 1)[0] for a in argv if a.startswith("--")}
    args.canonical = (not args.quick and not args.no_verify and protocol == CANONICAL
                      and not raw_flags & set(RAW_NONCANONICAL_FLAGS))
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
    if args.canonical:
        try:
            _verify_canonical_output_path(args.output)
        except ValueError as exc:
            parser.error(
                f"The canonical protocol writes {CANONICAL_OUTPUT} of this checkout ({exc}); "
                "change the protocol or use --quick for other paths"
            )
    args.explicit_data_dir, args.explicit_checkpoint_dir = args.data_dir, args.checkpoint_dir
    if args.data_dir is None:
        args.data_dir = q.default_data_dir()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = (CANONICAL_CHECKPOINT_DIR if args.canonical
                               else args.output.with_name(args.output.name + ".ckpt"))
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.postprocess_from_git is not None:
        payload = postprocess(args)
        canonical_dest = reserved(args.output)
    else:
        payload = run(args)
        canonical_dest = args.canonical
    payload = _write_json_atomic(args.output, payload, canonical_dest=canonical_dest)
    if not args.silent:
        print(json.dumps(payload["summary"].get("part_a_trailing_bf16_cuda_graph"), indent=None))
        print(f"Wrote {args.output}")
    return payload


if __name__ == "__main__":
    main()
