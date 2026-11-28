"""Cycle 6 C6-02: small-shape NVFP4 quantizer fusion, the fused GEMM alpha, and wide cross-mix latency (RTX 5080).

Part A  One linear layer at the eight C5-03 shapes, (16, 4096, 4096) .. (2048, 8192, 8192). Methods in one
        Williams-balanced session: BF16 cuBLAS, the pre-quantized NVFP4 GEMM through ``torch._scaled_mm_v2``
        (C5-03) and through cuBLASLt with a fused alpha (C6-02), and the C5-03 quantizer
        (``nvfp4_inference_blackwell``, unchanged; its bytes are pinned) against the C6-02 quantizer
        (``nvfp4_small_shape_blackwell``) for ``none``, ``had_b16``, ``bfly_b16`` and ``had_b16_static_gs``,
        eager and as CUDA graphs, with CUPTI kernel profiles. Outputs are compared bit for bit. A CUDA-graph
        ablation at two shapes switches off each C6-02 change.
Part B  The ~1.6 us elementwise kernel of ``torch._scaled_mm_v2``: the dispatcher trace (``aten::mul`` of the
        two FP32 tensor scales), the kernels of the direct cuBLASLt call, its bit-for-bit equality with
        ``nvfp4_scaled_mm_public`` for every shape and output dtype, and the cuBLASLt heuristic algorithms.
Part C  Full Transformer forwards (50m, 120m, w2048_6l; random weights) as CUDA graphs: BF16, C5-03 against
        C6-02 for the four variants, and the C6-01 wide cross-mix families (``hadamard_b64``,
        ``kron_hadamard_full``, ``cross_full_butterfly_b16``) on the C6-01 path (``rotated_qat.CrossInput``
        over the C5-03 layer), on ``CrossInput`` over the C6-02 layer, and with the fused cross-mix kernel
        (``FastCrossInput``). The Pareto analysis links these latencies to the verified C6-01 test NLLs
        (``results/cycle6_c6_01_wide_rotated_qat.json``, SHA-256 pinned).

Timing: Williams-balanced orders, CUDA events around every call with a 4x L2 flush (268 MB on the RTX 5080)
before every sample, raw per-call event times, and a host wall clock. TF32 and reduced-precision
reductions are off.

``cycle6_c6_02*.json`` is reserved: only the canonical protocol, on an sm_120+ device, from a clean committed
tree, in this checkout, without ``..`` or a symlink, may write it, and it is written once.

Run (canonical, alone on the GPU, from a clean commit):
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_c6_02_nvfp4_small_shape_fusion.py
Shakedown (small shapes and the tiny model; never the canonical file):
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_c6_02_nvfp4_small_shape_fusion.py \\
        --synthetic-shakedown --output /tmp/c6_02_shakedown.json
Validate a result:
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_c6_02_nvfp4_small_shape_fusion.py --validate PATH
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
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from research.polar_opt import harness_tier2_real as h
from research.polar_opt.harness_tier2_lm import Transformer
from research.quant_ptq import bench_c5_02_isoloss_ptq_mitigation as c502
from research.quant_ptq import bench_c5_03_nvfp4_120m_fusion as c503
from research.quant_ptq import bench_nvfp4_e2e as e2e
from research.quant_ptq import micro_rotated_ptq as q
from research.quant_ptq import nvfp4_inference_blackwell as nb
from research.quant_ptq import nvfp4_small_shape_blackwell as fs
from research.quant_ptq import rotated_qat as rq

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "cycle6_c6_02.v1"
TASK = ("Cycle 6 C6-02: Blackwell small-shape NVFP4 quantizer fusion, fused cuBLASLt alpha and wide cross-mix "
        "latency against the C6-01 perplexities")
DRIVER_MODULE = "research.quant_ptq.bench_c6_02_nvfp4_small_shape_fusion"
SCRIPT_REPO_PATH = "/".join(DRIVER_MODULE.split(".")) + ".py"
RESULT_REPO_PATH = "results/cycle6_c6_02_nvfp4_small_shape_fusion.json"
CANONICAL_OUTPUT = ROOT / RESULT_REPO_PATH
RESERVED_PATTERN = "cycle6_c6_02*.json"
# The modules that measure, summarize and validate. The producer commit ``meta.git_sha`` must hold these bytes.
SUMMARY_MODULES = (
    DRIVER_MODULE,
    "research.quant_ptq.nvfp4_small_shape_blackwell",
    "research.quant_ptq.nvfp4_inference_blackwell",
    "research.quant_ptq.rotated_qat",
    "research.quant_ptq.bench_nvfp4_e2e",
    "research.quant_ptq.bench_c5_03_nvfp4_120m_fusion",
    "research.quant_ptq.micro_rotated_ptq",
)
# Inputs that this study reads and must not change: the C5-03 module (its own and the C6-01 validators pin
# these bytes), the C5-03 and C6-01 canonical results.
PINNED_INPUTS = {
    "research/quant_ptq/nvfp4_inference_blackwell.py":
        "2b3a2af2615061d952216fa1691fdbd2f659961a5706f555c76bf35593053192",
    "results/cycle5_c5_03_nvfp4_120m_fusion.json": "e0e435ef2ea8598df22bace95ab76fd10c43b2ff394bf635ab8d1548d8bcad82",
    "results/cycle6_c6_01_wide_rotated_qat.json": "c490a415568244bd1e2489d0004adc025d856e92452cb976c05efc2dcdf66bf9",
}
C6_01_RESULT = "results/cycle6_c6_01_wide_rotated_qat.json"
C5_03_RESULT = "results/cycle5_c5_03_nvfp4_120m_fusion.json"
TOL = 1e-9

VARIANTS = ("none", "had_b16", "bfly_b16", "had_b16_static_gs")
VARIANT_ROTATION = c503.VARIANT_ROTATION
IMPLS = ("c5_03", "c6_02")
GEMM_ONLY = ("nvfp4_gemm_only_c5_03", "nvfp4_gemm_only_c6_02")
PART_A_METHODS = ("bf16_cublas", *GEMM_ONLY, *(f"{i}_{v}" for i in IMPLS for v in VARIANTS))
PART_A_SHAPES = c503.PART_A_SHAPES
PART_A_LABELS = c503.PART_A_LABELS
ABLATION_VARIANTS = ("none", "had_b16", "bfly_b16")
ABLATION_OPTIONS = {  # name -> the ``fs.OPTIONS`` change it makes (captured into its CUDA graph)
    "c6_02": {}, "c6_02_no_pdl": {"pdl": False}, "c6_02_scalar_io": {"vector_io": False},
    "c6_02_torch_gemm": {"lt_gemm": False}, "c6_02_stash": {"stash": True},
    "c6_02_no_delegate": {"delegate_large": False}}
ABLATIONS = (*ABLATION_OPTIONS, "c5_03")
ABLATION_METHODS = ("bf16_cublas", *(f"{a}_{v}" for v in ABLATION_VARIANTS for a in ABLATIONS))
OUT_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
WIDE_FAMILIES = ("hadamard_b64", "kron_hadamard_full", "cross_full_butterfly_b16")
WIDE_PATHS = ("c6_01", "c6_02_crossinput", "c6_02_fused")
PART_C_METHODS = ("bf16_graph", *(f"{i}_{v}" for v in VARIANTS for i in IMPLS),
                  *(f"{p}_{f}" for f in WIDE_FAMILIES for p in WIDE_PATHS))
PART_C_CONFIGS = c503.PART_B_CONFIGS
# Pareto: (C6-02 Part C method, C6-01 cell method) per evaluated C6-01 method. C6-01 cells: 50m = Part A/B
# arm ``ab/muon_ss3_54`` (three seeds), 120m = Part C arm ``c/c5_03_baseline`` (three seeds).
PARETO_CELLS = {"50m": "ab/muon_ss3_54", "120m": "c/c5_03_baseline"}
PARETO_POINTS = {
    "bf16": ("bf16_graph", None),
    "gptq": ("c6_02_none", "gptq/w4a4_hw_dynamic"),
    "hadamard_b16_gptq": ("c6_02_had_b16", "hadamard_b16_gptq/w4a4_hw_dynamic"),
    "hadamard_b16_gptq_static": ("c6_02_had_b16_static_gs", "hadamard_b16_gptq/w4a4_hw_static"),
    "adaptive_butterfly_b16_gptq": ("c6_02_bfly_b16", "adaptive_butterfly_b16_gptq/w4a4_hw_dynamic"),
    "hadamard_b64_gptq": ("c6_02_fused_hadamard_b64", "hadamard_b64_gptq/w4a4_hw_dynamic"),
    "kron_hadamard_full_gptq": ("c6_02_fused_kron_hadamard_full", "kron_hadamard_full_gptq/w4a4_hw_dynamic"),
    "cross_full_butterfly_b16_gptq": ("c6_02_fused_cross_full_butterfly_b16",
                                      "cross_full_butterfly_b16_gptq/w4a4_hw_dynamic"),
}
# The same C6-01 method on the C6-01 hardware path (for the latency gap the fused cross-mix closes)
PARETO_C6_01_PATH = {"hadamard_b64_gptq": "c6_01_hadamard_b64", "kron_hadamard_full_gptq": "c6_01_kron_hadamard_full",
                     "cross_full_butterfly_b16_gptq": "c6_01_cross_full_butterfly_b16"}

CANONICAL = {
    "device": "cuda", "seed": 1,
    "timing": {"squares_a": 2, "squares_c": 2, "iters": 10, "l2_flush_multiple": 4, "wall_iters": 2},
    "part_a": {"shapes": PART_A_SHAPES, "ablation_shapes": [[1024, 512, 1408], [2048, 8192, 8192]]},
    "part_b": {"shapes": PART_A_SHAPES, "out_dtypes": list(OUT_DTYPES), "algo_shape": [1024, 512, 1408]},
    "part_c": {"models": {"50m": 16, "120m": 8, "w2048_6l": 4}, "families": list(WIDE_FAMILIES)},
    "parity": {"quick": False},
}
SHAKEDOWN = {
    "device": "cuda", "seed": 1,
    "timing": {"squares_a": 1, "squares_c": 1, "iters": 3, "l2_flush_multiple": 4, "wall_iters": 1},
    "part_a": {"shapes": [[16, 256, 256], [129, 512, 768]], "ablation_shapes": [[129, 512, 768]]},
    "part_b": {"shapes": [[16, 256, 256], [129, 512, 768]], "out_dtypes": list(OUT_DTYPES),
               "algo_shape": [129, 512, 768]},
    "part_c": {"models": {"tiny": 2}, "families": list(WIDE_FAMILIES)},
    "parity": {"quick": True},
}
CANONICAL_PARITY_MIN_CASES = 279


def _require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


# ======================================================================================
# Timing and profiles
# ======================================================================================
@contextmanager
def options(**changes):
    """Temporarily change ``fs.OPTIONS`` (a CUDA graph keeps the setting of its capture)."""
    saved = dict(fs.OPTIONS)
    fs.OPTIONS.update(changes)
    try:
        yield
    finally:
        fs.OPTIONS.clear()
        fs.OPTIONS.update(saved)


def capture_under(fn, pool=None, **opts):
    with options(**opts):
        return c503.capture_shared(fn, pool) if pool is not None else e2e.capture(fn)


def timing(functions: dict, t: dict, squares: int, flush: Tensor) -> tuple[dict, list]:
    return e2e.williams_timing(functions, squares=squares, iters=t["iters"], flush=flush, wall_iters=t["wall_iters"])


MUL_KERNEL = "MulFunctor"
GEMM_MARKERS = ("gemm", "cutlass", "nvjet", "cublas")
FP4_MARKERS = ("e2m1", "fp4")


def kernel_kind(name: str) -> str:
    """Role of a CUDA kernel in a profile (C6-02 and C5-03 quantizers are told apart by their kernel objects)."""
    low = name.lower()
    if name in (fs._amax_kernel.fn.__name__, nb._amax_kernel.fn.__name__):
        return "amax"
    if name in (fs._quant_kernel.fn.__name__, nb._quant_kernel.fn.__name__):
        return "quant"
    if name == fs._cross_mix_kernel.fn.__name__:
        return "cross_mix"
    if MUL_KERNEL in name:
        return "scale_product"
    if any(m in low for m in GEMM_MARKERS):
        return "fp4_gemm" if any(m in low for m in FP4_MARKERS) else "gemm"
    return "other"


def kernel_profile(replays: dict, flush: Tensor, reps: int = 20) -> dict:
    """Per method: mean CUPTI kernel duration (us) and launches per replay, by kernel name (L2 flushed)."""
    out = {}
    for name, replay in replays.items():
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                flush.zero_()
                replay()
            torch.cuda.synchronize()
        totals, counts = defaultdict(float), defaultdict(int)
        for event in prof.events():
            if event.device_type.name == "CUDA" and "FillFunctor" not in event.name:  # drop the flush
                totals[event.name[:160]] += event.device_time
                counts[event.name[:160]] += 1
        if not totals:
            raise RuntimeError("CUPTI recorded no CUDA kernel (prime CUPTI before any CPU-only profiler session)")
        out[name] = {k: {"us": totals[k] / reps, "launches": counts[k] / reps, "kind": kernel_kind(k)}
                     for k in sorted(totals)}
    return out


def _kinds(profile: dict) -> dict:
    """Launches per replay by kernel kind."""
    out = defaultdict(float)
    for v in profile.values():
        out[v["kind"]] += v["launches"]
    return dict(out)


# ======================================================================================
# Bitwise helpers
# ======================================================================================
_bitwise_equal = c503._bitwise_equal
_same_operands = c503._same_operands


def _mismatches(a: Tensor, b: Tensor) -> int:
    view = c503._INT_VIEW[a.element_size()]
    return int((a.contiguous().view(view) != b.contiguous().view(view)).sum())


# ======================================================================================
# Part A: single-layer latency
# ======================================================================================
def _layers(W: Tensor, angles: Tensor, static_gs: Tensor) -> dict:
    c5 = c503._layers_for(W, angles, static_gs)
    return {v: {"c5_03": layer, "c6_02": fs.FastNVFP4Linear.from_layer(layer)} for v, layer in c5.items()}


def _inputs(device, M: int, N: int, K: int, seed: int):
    g = torch.Generator(device=device).manual_seed(seed + M + N + K)
    x = torch.randn(M, K, generator=g, device=device).to(torch.bfloat16)
    W = (torch.randn(N, K, generator=g, device=device) * 0.02).to(torch.bfloat16)
    angles = torch.rand(K // 16, 4, 8, generator=g, device=device) * 2 * math.pi
    return x, W, angles


def summarize_part_a_row(row: dict) -> dict:
    """Per mode: every method's median ms and speedups; per variant C6-02 against C5-03 and BF16."""
    flops = 2.0 * row["m"] * row["n"] * row["k"]
    out = {}
    for mode in ("eager", "cuda_graph"):
        t = row[mode]
        bf16 = t["bf16_cublas"]["gpu"]["median_ms"]
        table = {name: {"ms": t[name]["gpu"]["median_ms"], "wall_ms": t[name]["wall"]["median_ms"],
                        "tflops": flops / (t[name]["gpu"]["median_ms"] * 1e-3) / 1e12,
                        "speedup_vs_bf16": bf16 / t[name]["gpu"]["median_ms"]} for name in PART_A_METHODS}
        variants = {}
        for v in VARIANTS:
            new, old = table[f"c6_02_{v}"]["ms"], table[f"c5_03_{v}"]["ms"]
            variants[v] = {"c6_02_ms": new, "c5_03_ms": old, "c6_02_speedup_vs_c5_03": old / new,
                           "c6_02_speedup_vs_bf16": bf16 / new, "c5_03_speedup_vs_bf16": bf16 / old,
                           "c6_02_beats_bf16": new < bf16, "c6_02_faster_than_c5_03": new < old}
        gemm = {"c5_03_ms": table[GEMM_ONLY[0]]["ms"], "c6_02_ms": table[GEMM_ONLY[1]]["ms"],
                "c6_02_speedup_vs_c5_03": table[GEMM_ONLY[0]]["ms"] / table[GEMM_ONLY[1]]["ms"]}
        out[mode] = {"methods": table, "variants": variants, "gemm_only": gemm}
    return out


def part_a_shape(device, M: int, N: int, K: int, t: dict, flush: Tensor, seed: int, log) -> dict:
    x, W, angles = _inputs(device, M, N, K, seed)
    static_gs = nb.quantize_activation(x, "hadamard_b16").global_scale.clone()
    layers = _layers(W.float(), angles, static_gs)
    base = layers["none"]["c5_03"]
    pre = nb.quantize_activation(x, "none")
    alpha = (pre.global_scale * base.w_global).reshape(())
    plan = fs.lt_plan(M, N, K, torch.bfloat16, device)
    methods = {"bf16_cublas": lambda: x @ W.T,
               GEMM_ONLY[0]: lambda: nb.nvfp4_scaled_mm(pre, base.w_packed, base.w_scale, base.w_global),
               GEMM_ONLY[1]: lambda: fs.nvfp4_scaled_mm_lt(pre, base.w_packed, base.w_scale, alpha)}
    for impl in IMPLS:
        for v in VARIANTS:
            methods[f"{impl}_{v}"] = (lambda m=layers[v][impl]: m(x))
    checks = {f"{GEMM_ONLY[1]}_equals_public_bitwise": _bitwise_equal(
        methods[GEMM_ONLY[1]](), nb.nvfp4_scaled_mm_public(pre, base.w_packed, base.w_scale, base.w_global))}
    for v, pair in layers.items():
        rot = VARIANT_ROTATION[v]
        ref = nb.quantize_activation_ref(x, rot, angles if rot == "adaptive_butterfly_b16" else None,
                                         pair["c5_03"].act_global_scale)
        checks[f"{v}_c6_02_operands_equal_reference"] = _same_operands(pair["c6_02"].quantize_input(x), ref)
        checks[f"{v}_c6_02_output_equals_c5_03_bitwise"] = _bitwise_equal(pair["c6_02"](x), pair["c5_03"](x))
    eager, eager_orders = timing(methods, t, t["squares_a"], flush)
    graphs = {name: e2e.capture(fn) for name, fn in methods.items()}
    graphed, graph_orders = timing(graphs, t, t["squares_a"], flush)
    profile = kernel_profile({k: v for k, v in graphs.items() if k != "bf16_cublas"}, flush)
    row = {"m": M, "n": N, "k": K, "label": PART_A_LABELS.get((M, N, K), ""), "checks": checks,
           "delegated_to_c5_03": fs.delegated(M, K), "lt_plan": plan.record(),
           "plans": {v: {"c5_03": nb.quantizer_plan(M, K, VARIANT_ROTATION[v], v.endswith("static_gs")),
                         "c6_02": fs.quantizer_plan(M, K, VARIANT_ROTATION[v], v.endswith("static_gs"))}
                     for v in VARIANTS},
           "eager_williams_orders": eager_orders, "cuda_graph_williams_orders": graph_orders,
           "eager": eager, "cuda_graph": graphed, "cuda_graph_kernel_profile": profile}
    row["summary"] = summarize_part_a_row(row)
    s = row["summary"]["cuda_graph"]
    log(f"  A ({M:5d},{N:5d},{K:5d}) graph us bf16 {s['methods']['bf16_cublas']['ms'] * 1e3:.2f} gemm "
        f"{s['gemm_only']['c5_03_ms'] * 1e3:.2f}->{s['gemm_only']['c6_02_ms'] * 1e3:.2f} | " + "  ".join(
            f"{v} {s['variants'][v]['c5_03_ms'] * 1e3:.2f}->{s['variants'][v]['c6_02_ms'] * 1e3:.2f} "
            f"({s['variants'][v]['c6_02_speedup_vs_bf16']:.3f}x)" for v in VARIANTS))
    return row


def ablation_shape(device, M: int, N: int, K: int, t: dict, flush: Tensor, seed: int, log) -> dict:
    """CUDA-graph ablation: each C6-02 change switched off (or the stash / no-delegation alternative on)."""
    x, W, angles = _inputs(device, M, N, K, seed + 7)
    layers = _layers(W.float(), angles, nb.quantize_activation(x, "hadamard_b16").global_scale.clone())
    graphs, checks = {"bf16_cublas": e2e.capture(lambda: x @ W.T)}, {}
    for v in ABLATION_VARIANTS:
        fast, old = layers[v]["c6_02"], layers[v]["c5_03"]
        want = old(x)
        for name, opts in ABLATION_OPTIONS.items():
            with options(**opts):
                checks[f"{v}_{name}_equals_c5_03_bitwise"] = _bitwise_equal(fast(x), want)
            graphs[f"{name}_{v}"] = capture_under(lambda m=fast: m(x), **opts)
        graphs[f"c5_03_{v}"] = e2e.capture(lambda m=old: m(x))
    graphs = {name: graphs[name] for name in ABLATION_METHODS}
    timed, orders = timing(graphs, t, t["squares_a"], flush)
    table = c503._timing_table(timed, "bf16_cublas")
    log(f"  A ablation ({M},{N},{K}) graph us: " + "  ".join(f"{k} {v['ms'] * 1e3:.2f}" for k, v in table.items()))
    return {"m": M, "n": N, "k": K, "mode": "cuda_graph", "delegated_to_c5_03": fs.delegated(M, K),
            "checks": checks, "williams_orders": orders, "timing": timed, "summary": table}


def part_a(device, protocol: dict, flush: Tensor, log) -> dict:
    t = protocol["timing"]
    rows = []
    for M, N, K in protocol["part_a"]["shapes"]:
        rows.append(part_a_shape(device, M, N, K, t, flush, protocol["seed"], log))
        torch.cuda.empty_cache()
    ablation = [ablation_shape(device, M, N, K, t, flush, protocol["seed"], log)
                for M, N, K in protocol["part_a"]["ablation_shapes"]]
    return {"design": "Williams-balanced orders over all methods; CUDA events, mean of `iters` flushed samples per "
                      "round (raw per-call times in event_samples_ms), 4x L2 flush before every sample; median over "
                      "rounds. `wall_ms`: host clock between synchronizes. c5_03_* is nvfp4_inference_blackwell "
                      "(pinned bytes) with torch._scaled_mm_v2; c6_02_* is nvfp4_small_shape_blackwell "
                      "(FastNVFP4Linear sharing the same NVFP4 weights) with the direct cuBLASLt call; both allocate "
                      "their buffers per call. Inputs above fs.DELEGATE_ABOVE elements are delegated to the C5-03 "
                      "path by design.",
            "l2_flush_bytes": flush.numel(), "delegate_above_elements": fs.DELEGATE_ABOVE,
            "shapes": rows, "ablation": ablation}


# ======================================================================================
# Part B: the elementwise kernel of torch._scaled_mm_v2
# ======================================================================================
def _trace(fn) -> dict:
    """CPU operators and CUDA kernels of one call (after a warm-up)."""
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    cpu = [e.name for e in prof.events() if e.device_type.name == "CPU" and e.name.startswith("aten::")]
    cuda = [e.name[:160] for e in prof.events() if e.device_type.name == "CUDA"]
    return {"aten_ops": cpu, "kernels": cuda, "kernel_kinds": [kernel_kind(k) for k in cuda]}


def heuristic_algorithms(M: int, N: int, K: int, device, flush: Tensor, t: dict) -> dict:
    """Every algorithm the cuBLASLt heuristic returns for the NVFP4 GEMM: config, bitwise check, graph time."""
    import ctypes
    lt = fs._lt()
    x, W, _ = _inputs(device, M, N, K, 11)
    layer = nb.NVFP4Linear.from_weight(W.float(), "none")
    a = nb.quantize_activation(x, "none")
    ref = nb.nvfp4_scaled_mm_public(a, layer.w_packed, layer.w_scale, layer.w_global)
    alpha = (a.global_scale * layer.w_global).reshape(())
    plan = fs.lt_plan(M, N, K, torch.bfloat16, device)
    pref = ctypes.c_void_p()
    fs._ck(lt.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "MatmulPreferenceCreate")
    limit = ctypes.c_uint64(fs.LT_WORKSPACE_LIMIT)
    fs._ck(lt.cublasLtMatmulPreferenceSetAttribute(pref, fs._PREF_MAX_WORKSPACE_BYTES, ctypes.byref(limit), 8), "pref")
    results, found = (fs._Heuristic * 16)(), ctypes.c_int()
    fs._ck(lt.cublasLtMatmulAlgoGetHeuristic(plan.handle, plan.desc, *plan.layouts, pref, 16, results,
                                             ctypes.byref(found)), "AlgoGetHeuristic")
    lt.cublasLtMatmulPreferenceDestroy(pref)
    algos, graphs, outs = [], {}, []
    for i in range(found.value):
        config = {}
        for name, attr in fs._ALGO_CONFIG.items():
            value, written = ctypes.c_int32(), ctypes.c_size_t()
            lt.cublasLtMatmulAlgoConfigGetAttribute(ctypes.byref(results[i].algo), attr, ctypes.byref(value), 4,
                                                    ctypes.byref(written))
            config[name] = value.value
        out = torch.full((M, N), float("nan"), dtype=torch.bfloat16, device=device)
        outs.append(out)

        def call(i=i, out=out):
            fs._set(plan.desc, fs._DESC_A_SCALE_POINTER, layer.w_scale.data_ptr(), ctypes.c_void_p)
            fs._set(plan.desc, fs._DESC_B_SCALE_POINTER, a.scale.data_ptr(), ctypes.c_void_p)
            fs._ck(lt.cublasLtMatmul(plan.handle, plan.desc, alpha.data_ptr(), layer.w_packed.data_ptr(),
                                     plan.layouts[0], a.packed.data_ptr(), plan.layouts[1], plan.zero.data_ptr(),
                                     out.data_ptr(), plan.layouts[2], out.data_ptr(), plan.layouts[3],
                                     ctypes.byref(results[i].algo), None, 0,
                                     torch.cuda.current_stream().cuda_stream), "Matmul")
        if results[i].workspaceSize == 0:
            call()
            torch.cuda.synchronize()
            graphs[f"algo_{i}"] = e2e.capture(call)
        algos.append({"index": i, "config": config, "workspace_bytes": int(results[i].workspaceSize),
                      "waves": float(results[i].wavesCount),
                      "bitwise_equal_public": bool(results[i].workspaceSize == 0 and _bitwise_equal(out, ref))})
    timed, orders = timing(graphs, t, 1, flush) if graphs else ({}, [])
    for row in algos:
        name = f"algo_{row['index']}"
        row["graph_ms"] = timed[name]["gpu"]["median_ms"] if name in timed else None
    return {"mnk": [M, N, K], "returned": found.value, "selected": plan.record()["algo"], "algorithms": algos,
            "williams_orders": orders}


def part_b(device, protocol: dict, flush: Tensor, log) -> dict:
    t = protocol["timing"]
    M, N, K = protocol["part_b"]["algo_shape"]
    x, W, _ = _inputs(device, M, N, K, protocol["seed"])
    layer = nb.NVFP4Linear.from_weight(W.float(), "hadamard_b16")
    fast = fs.FastNVFP4Linear.from_layer(layer)
    a = nb.quantize_activation(x, "hadamard_b16")
    alpha = (a.global_scale * layer.w_global).reshape(())
    traces = {"torch_scaled_mm_v2": _trace(lambda: nb.nvfp4_scaled_mm(a, layer.w_packed, layer.w_scale,
                                                                      layer.w_global)),
              "cublaslt_fused_alpha": _trace(lambda: fs.nvfp4_scaled_mm_lt(a, layer.w_packed, layer.w_scale, alpha)),
              "c5_03_layer": _trace(lambda: layer(x)), "c6_02_layer": _trace(lambda: fast(x))}
    exact = []
    for M2, N2, K2 in protocol["part_b"]["shapes"]:
        x2, W2, angles2 = _inputs(device, M2, N2, K2, protocol["seed"] + 3)
        for rot in nb.ROTATIONS:
            lay = nb.NVFP4Linear.from_weight(W2.float(), rot, angles2 if rot == "adaptive_butterfly_b16" else None)
            act = nb.quantize_activation(x2, rot, lay.tables())
            al = (act.global_scale * lay.w_global).reshape(())
            for name, dtype in OUT_DTYPES.items():
                plan = fs.lt_plan(M2, N2, K2, dtype, device)
                out = torch.full((M2, N2), float("nan"), dtype=dtype, device=device)  # C must not be read
                plan(act, lay.w_packed, lay.w_scale, al, out)
                ref = nb.nvfp4_scaled_mm_public(act, lay.w_packed, lay.w_scale, lay.w_global, dtype)
                exact.append({"mnk": [M2, N2, K2], "rotation": rot, "out_dtype": name, "direct": plan.direct,
                              "algo": plan.record()["algo"], "bitwise_equal_public": _bitwise_equal(out, ref)})
    algos = heuristic_algorithms(M, N, K, device, flush, t)
    log(f"  B trace: scaled_mm_v2 kernels {traces['torch_scaled_mm_v2']['kernel_kinds']} | cuBLASLt "
        f"{traces['cublaslt_fused_alpha']['kernel_kinds']} | {sum(r['bitwise_equal_public'] for r in exact)}/"
        f"{len(exact)} bitwise | {algos['returned']} heuristic algorithms")
    return {"design": "Profiler traces of one call (CPU aten ops, CUDA kernels); the direct cuBLASLt call writes into "
                      "a NaN-filled output (beta = 0 must not read C) and is compared bit for bit with "
                      "nvfp4_scaled_mm_public for every shape, rotation and output dtype; every heuristic algorithm "
                      "is timed as a CUDA graph (4x L2 flush).",
            "traces": traces, "exactness": exact, "heuristic": algos}


# ======================================================================================
# Part C: full Transformer forwards and the wide cross-mix
# ======================================================================================
def _wide_modules(model, family: str, angles: dict) -> dict:
    """Per quantized linear: the C5-03 layer of ``family`` and its cross-block mix.

    The layer stores the round-to-nearest NVFP4 of ``W R^T`` with ``R = R16 C`` (``from_weight`` applies the
    in-kernel block-16 rotation ``R16`` to the mixed weight ``C W``), as the C6-01 hardware path.
    """
    rot = rq.kernel_rotation(family)
    out = {}
    for name, m in q.quantized_linears(model).items():
        cross = rq.make_cross(rq.FAMILIES[family][0], m.in_features)
        layer = nb.NVFP4Linear.from_weight(cross.forward(m.weight.float()), rot,
                                           angles[name] if rot == "adaptive_butterfly_b16" else None)
        out[name] = (layer, cross)
    return out


def summarize_part_c_model(row: dict) -> dict:
    t = row["timing"]
    bf16 = t["bf16_graph"]["gpu"]["median_ms"]
    table = {n: {"ms": t[n]["gpu"]["median_ms"], "wall_ms": t[n]["wall"]["median_ms"],
                 "tokens_per_s": row["tokens_per_call"] / (t[n]["gpu"]["median_ms"] * 1e-3),
                 "speedup_vs_bf16_graph": bf16 / t[n]["gpu"]["median_ms"]} for n in PART_C_METHODS}
    variants = {v: {"c5_03_ms": table[f"c5_03_{v}"]["ms"], "c6_02_ms": table[f"c6_02_{v}"]["ms"],
                    "c6_02_speedup_vs_c5_03": table[f"c5_03_{v}"]["ms"] / table[f"c6_02_{v}"]["ms"],
                    "c6_02_speedup_vs_bf16_graph": bf16 / table[f"c6_02_{v}"]["ms"]} for v in VARIANTS}
    wide = {f: {"c6_01_ms": table[f"c6_01_{f}"]["ms"], "c6_02_crossinput_ms": table[f"c6_02_crossinput_{f}"]["ms"],
                "c6_02_fused_ms": table[f"c6_02_fused_{f}"]["ms"],
                "fused_speedup_vs_c6_01": table[f"c6_01_{f}"]["ms"] / table[f"c6_02_fused_{f}"]["ms"],
                "fused_speedup_vs_bf16_graph": bf16 / table[f"c6_02_fused_{f}"]["ms"],
                "fused_overhead_vs_c6_02_had_b16": table[f"c6_02_fused_{f}"]["ms"] / table["c6_02_had_b16"]["ms"]}
            for f in WIDE_FAMILIES}
    return {"methods": table, "variants": variants, "wide": wide}


def part_c_model(device, name: str, batch: int, t: dict, flush: Tensor, seed: int, log) -> dict:
    config = PART_C_CONFIGS[name]
    torch.manual_seed(seed)
    with torch.device(device):
        model = Transformer(config).to(torch.bfloat16).eval()
    tokens = torch.randint(0, config.vocab_size, (batch, config.sequence_length), device=device)
    linears = q.quantized_linears(model)
    static = c503.calibrate(model, list(linears), {n: q.BlockHadamard(16) for n in linears}, tokens.cpu().numpy(),
                            batch, device)
    g = torch.Generator(device=device).manual_seed(seed + 5)
    angles = {n: torch.rand(m.in_features // 16, 4, 8, generator=g, device=device) * 2 * math.pi
              for n, m in linears.items()}
    models, keep = {}, [model]
    for v in VARIANTS:
        rot = VARIANT_ROTATION[v]
        old, new = copy.deepcopy(model), copy.deepcopy(model)
        for n, m in linears.items():
            layer = nb.NVFP4Linear.from_weight(m.weight.float(), rot, angles[n] if rot == "adaptive_butterfly_b16"
                                               else None, act_global_scale=static[n] if v.endswith("static_gs")
                                               else None)
            e2e._swap(old, {n: layer})
            e2e._swap(new, {n: fs.FastNVFP4Linear.from_layer(layer)})
        models[f"c5_03_{v}"], models[f"c6_02_{v}"] = old, new
    fused_modules = {}
    for fam in WIDE_FAMILIES:
        variants = {p: copy.deepcopy(model) for p in WIDE_PATHS}
        fused_modules[fam] = {}
        for n, (layer, cross) in _wide_modules(model, fam, angles).items():
            fast = fs.FastNVFP4Linear.from_layer(layer)
            fused = fs.FastCrossInput(fast, cross).to(device)
            e2e._swap(variants["c6_01"], {n: rq.CrossInput(layer, cross)})
            e2e._swap(variants["c6_02_crossinput"], {n: rq.CrossInput(fast, cross)})
            e2e._swap(variants["c6_02_fused"], {n: fused})
            fused_modules[fam][n] = (fused, cross)
        for p, m in variants.items():
            models[f"{p}_{fam}"] = m
    keep += list(models.values())

    def forward(m):
        def run():
            with torch.inference_mode():
                return m(tokens)
        return run

    checks, parity, rel = {}, {}, {}
    with torch.inference_mode():  # one pair (or triple) of logits at a time: 50m logits are 0.8 GB each
        reference = model(tokens).float()
        for v in VARIANTS:
            a, b = models[f"c5_03_{v}"](tokens), models[f"c6_02_{v}"](tokens)
            checks[f"{v}_c6_02_logits_equal_c5_03_bitwise"] = _bitwise_equal(b, a)
            rel[f"c5_03_{v}"], rel[f"c6_02_{v}"] = e2e._rel(a, reference), e2e._rel(b, reference)
            del a, b
        for fam in WIDE_FAMILIES:
            a, b, c = (models[f"{p}_{fam}"](tokens) for p in WIDE_PATHS)
            checks[f"{fam}_c6_02_crossinput_logits_equal_c6_01_bitwise"] = _bitwise_equal(b, a)
            parity[fam] = {"logits_bitwise_equal_c6_01": _bitwise_equal(c, a),
                           "logits_mismatches": _mismatches(c, a), "logits_numel": c.numel(),
                           "logits_max_abs_diff": float((c.float() - a.float()).abs().max()),
                           "logits_rel_diff_vs_c6_01": e2e._rel(c, a)}
            for p, logits in zip(WIDE_PATHS, (a, b, c)):
                rel[f"{p}_{fam}"] = e2e._rel(logits, reference)
            del a, b, c
            parity[fam].update(_fused_mix_parity(model, tokens, fused_modules[fam]))
    del reference
    pool = torch.cuda.graph_pool_handle()
    functions = {"bf16_graph": c503.capture_shared(forward(model), pool)}
    for k in PART_C_METHODS[1:]:
        functions[k] = c503.capture_shared(forward(models[k]), pool)
    functions = {k: functions[k] for k in PART_C_METHODS}
    profile = {k: _kinds(p) for k, p in kernel_profile(
        {k: functions[k] for k in ("c5_03_had_b16", "c6_02_had_b16", "c6_01_cross_full_butterfly_b16",
                                   "c6_02_fused_cross_full_butterfly_b16")}, flush, reps=3).items()}
    timed, orders = timing(functions, t, t["squares_c"], flush)
    row = {"config": asdict(config), "batch": batch, "tokens_per_call": batch * config.sequence_length,
           "nvfp4_layers": len(linears), "linear_parameters": sum(m.weight.numel() for m in linears.values()),
           "checks": checks, "fused_cross_mix_parity": parity, "logits_rel_diff_vs_bf16": rel,
           "kernel_kinds_per_forward": profile, "williams_orders": orders, "timing": timed}
    row["summary"] = summarize_part_c_model(row)
    s = row["summary"]
    log(f"  C {name}: bf16 graph {s['methods']['bf16_graph']['ms']:.3f} ms | " + "  ".join(
        f"{v} {s['variants'][v]['c5_03_ms']:.3f}->{s['variants'][v]['c6_02_ms']:.3f}" for v in VARIANTS) + " | " +
        "  ".join(f"{f} {s['wide'][f]['c6_01_ms']:.3f}->{s['wide'][f]['c6_02_fused_ms']:.3f}" for f in WIDE_FAMILIES))
    del model, models, functions, keep, pool, fused_modules
    torch.cuda.empty_cache()
    return row


@torch.no_grad()
def _fused_mix_parity(model, tokens: Tensor, modules: dict) -> dict:
    """FP32 bit mismatches of ``FastCrossInput.mixed`` against ``CrossMix.forward(x.float())``.

    The inputs are those of every quantized linear in one BF16 forward of ``model`` (forward pre-hooks).
    """
    counts = {"mixed_values": 0, "mixed_mismatches": 0, "mixed_max_abs_diff": 0.0, "layers_exact": 0, "layers": 0}
    handles = []

    def hook(fused, cross):
        def record(module, inputs):
            x = inputs[0]
            got = fused.mixed(x)
            want = cross.forward(x.float()).reshape(got.shape)
            n = _mismatches(got, want)
            counts["mixed_values"] += got.numel()
            counts["mixed_mismatches"] += n
            counts["layers_exact"] += n == 0
            counts["layers"] += 1
            counts["mixed_max_abs_diff"] = max(counts["mixed_max_abs_diff"], float((got - want).abs().max()))
        return record

    blocks = {name: dict(model.named_modules())[name] for name in modules}
    for name, (fused, cross) in modules.items():
        handles.append(blocks[name].register_forward_pre_hook(hook(fused, cross)))
    try:
        with torch.inference_mode():
            model(tokens)
    finally:
        for hnd in handles:
            hnd.remove()
    counts["mixed_mismatch_fraction"] = counts["mixed_mismatches"] / max(counts["mixed_values"], 1)
    return counts


def c6_01_cells() -> dict:
    """The C6-01 per-method three-seed test NLL means of the Pareto cells (from the SHA-256-pinned result)."""
    data = (ROOT / C6_01_RESULT).read_bytes()
    _require(hashlib.sha256(data).hexdigest() == PINNED_INPUTS[C6_01_RESULT], f"{C6_01_RESULT} is not the pinned file")
    cells = json.loads(data)["analysis"]["cells"]
    out = {}
    for preset, cell in PARETO_CELLS.items():
        c = cells[cell]
        rows = {"bf16": {"test_nll": c["bf16"]["test"]["mean"], "test_delta_nll": 0.0}}
        for point, (_, key) in PARETO_POINTS.items():
            if key is not None and key in c["methods"]:
                m = c["methods"][key]["test"]
                rows[point] = {"test_nll": m["nll"]["mean"], "test_delta_nll": m["delta_nll"]["mean"]}
        out[preset] = {"cell": cell, "methods": rows}
    return out


def pareto(part_c: dict) -> dict:
    """Latency (C6-02 Part C graph ms) against the C6-01 test NLL; the frontier is the non-dominated set."""
    cells = c6_01_cells()
    out = {}
    for preset, cell in cells.items():
        if preset not in part_c["models"]:
            continue
        table = part_c["models"][preset]["summary"]["methods"]
        points = []
        for point, row in cell["methods"].items():
            method = PARETO_POINTS[point][0]
            c6_01_path = PARETO_C6_01_PATH.get(point)
            points.append({"method": point, "part_c_method": method, "latency_ms": table[method]["ms"],
                           "c6_01_path_latency_ms": table[c6_01_path]["ms"] if c6_01_path else None,
                           "test_nll": row["test_nll"], "test_delta_nll": row["test_delta_nll"]})
        points.sort(key=lambda p: (p["latency_ms"], p["test_nll"]))
        frontier, best = [], math.inf
        for p in points:
            if p["test_nll"] < best:
                frontier.append(p["method"])
                best = p["test_nll"]
        for p in points:
            p["on_frontier"] = p["method"] in frontier
        out[preset] = {"c6_01_cell": cell["cell"], "points": points, "frontier": frontier}
    return out


def part_c(device, protocol: dict, flush: Tensor, log) -> dict:
    t = protocol["timing"]
    rows = {name: part_c_model(device, name, batch, t, flush, protocol["seed"], log)
            for name, batch in protocol["part_c"]["models"].items()}
    torch._dynamo.reset()
    return {"design": "random-init BF16 weights and tokens; every block linear (qkv, out, gate, up, down) replaced by "
                      "an RTN NVFP4 layer. Variants: c5_03_* nvfp4_inference_blackwell, c6_02_* FastNVFP4Linear on "
                      "the same weights. Wide families: c6_01_* = rotated_qat.CrossInput (FP32 PyTorch cross-block "
                      "mix) over the C5-03 layer (the C6-01 hardware path), c6_02_crossinput_* the same over "
                      "FastNVFP4Linear, c6_02_fused_* FastCrossInput (one Triton cross-mix kernel). One CUDA graph "
                      "per forward (shared memory pool), Williams-balanced, 4x L2 flush. Embedding, norms, "
                      "attention and the tied head stay BF16.",
            "models": rows}


# ======================================================================================
# Parity and sync checks
# ======================================================================================
def kernel_parity(device, quick: bool) -> dict:
    """Bit-exact C6-02 operands against the eager reference (the C5-03 kernel_parity grid, three input dtypes,
    non-contiguous inputs and special values), the public launchers on dirty buffers, and the packed tables."""
    grid = [(m, k) for m in (1, 37, 129) for k in (64, 80, 768)] if quick else \
        [(m, k) for m in (1, 37, 128, 129, 256, 512) for k in (64, 80, 512, 768, 1408, 2048)]
    extra = [(33, 1408)] if quick else [(16, 4096), (33, 1408), (1000, 512), (1024, 1408), (257, 2048), (7, 8192)]
    failures, cases, kinds = [], 0, defaultdict(int)

    def check(ok: bool, label, kind: str):
        nonlocal cases
        cases += 1
        kinds[kind] += 1
        if not ok:
            failures.append([kind, *label] if isinstance(label, list) else [kind, label])

    def operands(x, rotation, angles, static, label):
        tables = nb.butterfly_tables(angles) if angles is not None else None
        check(_same_operands(fs.quantize_activation(x, rotation, tables, static),
                             nb.quantize_activation_ref(x, rotation, angles, static)), label, "quantize_activation")

    for M, K in grid + extra:
        g = torch.Generator(device=device).manual_seed(M * 31 + K)
        x = (torch.randn(M, K, generator=g, device=device) * 3).to(torch.bfloat16)
        x[:, :16] *= 30
        angles = torch.rand(K // 16, 4, 8, generator=g, device=device) * 2 * math.pi
        for rotation in nb.ROTATIONS:
            a_ = angles if rotation == "adaptive_butterfly_b16" else None
            ref = nb.quantize_activation_ref(x, rotation, a_)
            for static in (None, ref.global_scale * 0.5):
                operands(x, rotation, a_, static, [M, K, rotation, static is not None, "bf16"])
    special = torch.randn(64, 512, device=device)
    special[0, :8] = torch.tensor([0.0, -0.0, 1e-45, -1e-45, 1e-38, float("inf"), float("-inf"), float("nan")])
    special[1, :16] = torch.logspace(-40, 0, 16, device=device)
    base = torch.randn(300, 1024, device=device)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for x in (base.to(dtype)[::2, 256:768], base.to(dtype).t().contiguous().t()[:, :512], special.to(dtype),
                  base.to(dtype)[1:130, ::2]):
            angles = torch.rand(x.shape[1] // 16, 4, 8, device=device) * 2 * math.pi
            for rotation in nb.ROTATIONS:
                operands(x, rotation, angles if rotation == "adaptive_butterfly_b16" else None, None,
                         [list(x.shape), rotation, str(dtype), x.is_contiguous()])
    for M, K in ((37, 80), (129, 768), (1024, 1408)):  # public launchers, dirty buffers, exact n_part
        x = torch.randn(M, K, device=device)
        angles = torch.rand(K // 16, 4, 8, device=device) * 2 * math.pi
        pr, pc = nb.scale_layout(M, K)
        for rotation in nb.ROTATIONS:
            a_ = angles if rotation == "adaptive_butterfly_b16" else None
            tables = nb.butterfly_tables(a_) if a_ is not None else None
            packed = torch.full((M, K // 2), 0xAB, dtype=torch.uint8, device=device)
            scale = torch.full((pr * pc,), 0xCD, dtype=torch.uint8, device=device)
            ws = torch.full((fs.MAX_PARTIALS,), float("nan"), device=device)
            gs = torch.empty((), device=device)
            n = fs.launch_amax(x, ws, rotation, tables)
            fs.launch_quant(x, packed, scale, gs, rotation=rotation, tables=tables, workspace=ws, n_part=n)
            got = nb.NVFP4Activation(packed.view(torch.float4_e2m1fn_x2), scale.view(torch.float8_e4m3fn), gs)
            check(_same_operands(got, nb.quantize_activation_ref(x, rotation, a_)) and 1 <= n <= fs.PASS1_PROGRAMS,
                  [M, K, rotation, "launchers"], "launchers")
    for M, K in ((37, 768), (1024, 1408), (129, 2048)):  # the packed (16, G, 4) tables of FastNVFP4Linear
        x = torch.randn(M, K, device=device).to(torch.bfloat16)
        W = torch.randn(64, K, device=device) * 0.02
        angles = torch.rand(K // 16, 4, 8, device=device) * 2 * math.pi
        layer = nb.NVFP4Linear.from_weight(W, "adaptive_butterfly_b16", angles)
        fast = fs.FastNVFP4Linear.from_layer(layer)
        a, _ = fs._quantize(x, M, K, "adaptive_butterfly_b16", fast._fast_tables, None)
        check(_same_operands(a, nb.quantize_activation_ref(x, "adaptive_butterfly_b16", angles))
              and _bitwise_equal(fast(x), layer(x)), [M, K, "packed_tables"], "packed_tables")
    return {"cases": cases, "case_kinds": dict(kinds), "bitwise_failures": failures, "passed": not failures,
            "reference": "micro_rotated_ptq rotation (FP32) + nvfp4_encode + pack_e2m1 + to_blocked "
                         "(nb.quantize_activation_ref); the packed-table cases also compare the layer output with the "
                         "C5-03 NVFP4Linear bit for bit"}


def sync_check(device) -> dict:
    """Host syncs and device-to-host copies inside eager C6-02 forwards (must be none), and graph replay exactness."""
    out = {}
    W = torch.randn(512, 1408, device=device) * 0.02
    angles = torch.rand(88, 4, 8, device=device) * 2 * math.pi
    x = torch.randn(64, 1408, device=device, dtype=torch.bfloat16)
    cross = rq.make_cross("full", 1408)
    for v in VARIANTS:
        rot = VARIANT_ROTATION[v]
        layer = nb.NVFP4Linear.from_weight(W, rot, angles if rot == "adaptive_butterfly_b16" else None,
                                           act_global_scale=torch.tensor(0.01, device=device)
                                           if v.endswith("static_gs") else None)
        for name, module in (("c6_02", fs.FastNVFP4Linear.from_layer(layer)),
                             ("c6_02_fused_cross", fs.FastCrossInput(fs.FastNVFP4Linear.from_layer(layer), cross))):
            module(x)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                module(x)
                raised = False
            except RuntimeError:
                raised = True
            finally:
                torch.cuda.set_sync_debug_mode("default")
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function("c6_02_forward"):
                    for _ in range(3):
                        module(x)
            torch.cuda.synchronize()
            events = prof.events()
            region = next(e for e in events if e.name == "c6_02_forward" and e.device_type.name == "CPU").time_range
            inside = [e.name for e in events if e.device_type.name == "CPU"
                      and region.start <= e.time_range.start and e.time_range.end <= region.end]
            syncs = [n for n in inside if "Synchronize" in n or "Memcpy" in n or "item" in n.lower()
                     or "local_scalar_dense" in n]
            d2h = [e.name for e in events if "DtoH" in e.name]
            x_static = x.clone()
            res = {}
            replay = e2e.capture(lambda m=module: res.__setitem__("y", m(x_static)))
            exact = 0
            gen = torch.Generator(device=device).manual_seed(5)
            for scale in (1.0, 1e-3, 50.0, 1.0):
                x_static.copy_(torch.randn(64, 1408, generator=gen, device=device) * scale)
                replay()
                exact += _bitwise_equal(res["y"], module(x_static))
            out[f"{name}_{v}"] = {"sync_debug_error_raised": raised, "host_syncs_in_region": len(syncs),
                                  "d2h_copies": len(d2h), "graph_replays": 4, "graph_replays_bitwise_exact": exact}
    return out


# ======================================================================================
# Summary and caveats
# ======================================================================================
def summarize(payload: dict) -> dict:
    a, c = payload["part_a"], payload["part_c"]
    shapes = []
    for r in a["shapes"]:
        s = r["summary"]["cuda_graph"]
        shapes.append({"mnk": [r["m"], r["n"], r["k"]], "label": r["label"], "delegated_to_c5_03": r["delegated_to_c5_03"],
                       "bf16_cublas_ms": s["methods"]["bf16_cublas"]["ms"], "gemm_only": s["gemm_only"],
                       "cuda_graph": s["variants"], "eager": r["summary"]["eager"]["variants"]})
    cells = [(r["mnk"], v) for r in shapes for v in VARIANTS]
    beats = [[*m, v] for m, v in cells if next(x for x in shapes if x["mnk"] == m)["cuda_graph"][v]["c6_02_beats_bf16"]]
    trailing = [[*m, v] for m, v in cells if [*m, v] not in beats]
    regressions = [[*r["mnk"], v] for r in shapes for v in VARIANTS if not r["cuda_graph"][v]["c6_02_faster_than_c5_03"]]
    target = next((r for r in shapes if r["mnk"] == [1024, 512, 1408]), None)
    profile = None
    if target is not None:
        row = next(r for r in a["shapes"] if [r["m"], r["n"], r["k"]] == [1024, 512, 1408])
        profile = {m: {k: round(v["us"], 3) for k, v in row["cuda_graph_kernel_profile"][m].items()}
                   for m in (*GEMM_ONLY, *(f"{i}_{v}" for i in IMPLS for v in VARIANTS))}
    b = payload["part_b"]
    part_b = {"torch_scaled_mm_v2_kernel_kinds": b["traces"]["torch_scaled_mm_v2"]["kernel_kinds"],
              "torch_scaled_mm_v2_has_aten_mul": "aten::mul" in b["traces"]["torch_scaled_mm_v2"]["aten_ops"],
              "cublaslt_kernel_kinds": b["traces"]["cublaslt_fused_alpha"]["kernel_kinds"],
              "bitwise_equal_public": [sum(r["bitwise_equal_public"] for r in b["exactness"]), len(b["exactness"])],
              "heuristic_algorithms": b["heuristic"]["returned"],
              "gemm_only_cuda_graph": {r["label"] or str(r["mnk"]): r["gemm_only"] for r in shapes}}
    part_c = {name: {"variants": m["summary"]["variants"], "wide": m["summary"]["wide"],
                     "bf16_graph_ms": m["summary"]["methods"]["bf16_graph"]["ms"],
                     "fused_cross_mix_parity": {f: {k: v for k, v in p.items() if k in (
                         "logits_bitwise_equal_c6_01", "logits_mismatches", "logits_max_abs_diff",
                         "mixed_mismatch_fraction", "layers_exact", "layers")} for f, p in
                         m["fused_cross_mix_parity"].items()}} for name, m in c["models"].items()}
    return {"part_a_cuda_graph_cells_beating_bf16": [len(beats), len(cells)],
            "part_a_cuda_graph_trailing_bf16": trailing, "part_a_cuda_graph_regressions_vs_c5_03": regressions,
            "part_a": shapes, "part_a_1024x512x1408_kernel_us": profile,
            "part_a_ablation": [{"mnk": [x["m"], x["n"], x["k"]], "ms": {k: v["ms"] for k, v in x["summary"].items()}}
                                for x in a["ablation"]],
            "part_b": part_b, "part_c": part_c, "pareto": payload["pareto"],
            "kernel_parity": [payload["kernel_parity"]["cases"], len(payload["kernel_parity"]["bitwise_failures"])]}


def caveats(payload: dict) -> list[str]:
    s = payload["summary"]
    beats, cells = s["part_a_cuda_graph_cells_beating_bf16"]
    out = [
        "nvfp4_inference_blackwell.py (C5-03) is unchanged: the C5-03 and C6-01 validators pin its bytes, so the C6-02 "
        "fast path is the separate module nvfp4_small_shape_blackwell (drop-in, bit-identical operands and outputs).",
        f"Part A: C6-02 beats BF16 cuBLAS in {beats}/{cells} CUDA-graph cells; trailing cells: "
        f"{s['part_a_cuda_graph_trailing_bf16'] or 'none'}; cells slower than C5-03: "
        f"{s['part_a_cuda_graph_regressions_vs_c5_03'] or 'none'}. Each cell is one Williams-balanced session; "
        "cell-to-cell differences of about 1-2% are within the session-to-session spread measured while tuning (the "
        "GPU also drives the desktop).",
        f"Inputs above {fs.DELEGATE_ABOVE} elements (here only 2048 x 8192) use the C5-03 quantizer and "
        "torch._scaled_mm_v2 by design: there the DRAM-bound C5-03 2D tiles match or beat the C6-02 kernels (see the "
        "c6_02_no_delegate ablation), so C6-02 and C5-03 run the same code at that shape.",
        "Part B: the elementwise kernel is aten::mul of the two FP32 tensor scales inside torch._scaled_mm_v2; the C6-02 "
        "quantizer writes alpha = global_scale * w_global (one rounded FP32 product, the same value) and the GEMM calls "
        "cublasLtMatmul with PyTorch's descriptor (1 MiB heuristic limit, device pointer mode). Equality with "
        "nvfp4_scaled_mm_public is measured bit for bit, not derived; it holds for the cuBLASLt build of this torch "
        "wheel. The direct call runs through ctypes (host cost in eager mode).",
        "Part C models use random weights: latency does not depend on the weight values, but the NLL axis of the Pareto "
        "comes from the C6-01 three-seed test means (pinned C6-01 result), not from these models.",
        "The fused cross-mix (FastCrossInput) reproduces the two-half FMA order of the cuBLAS gemmSN kernels that "
        "torch.matmul picks for most shapes; where cuBLAS picks another kernel the mixed FP32 values differ by "
        "rounding. Its parity against rotated_qat.CrossInput (mixed-value and logits mismatches) is measured per model "
        "and family, so the C6-01 NLLs carry over approximately (near-0 parity), not bit for bit, for the full mix.",
        "Every wide-family linear computes its own cross-mix as in the C6-01 hardware path (gate and up mix the same "
        "input twice); sharing it would be a further model-level change, not measured here.",
        "The butterfly coefficients of FastNVFP4Linear live in a non-persistent (16, G, 4) buffer derived from the "
        "(cos, sin) tables; the public quantize_activation keeps the (G, 4, 8) table API.",
    ]
    return out


# ======================================================================================
# Provenance
# ======================================================================================
def reserved(path: Path) -> bool:
    path = Path(path)
    return any(fnmatch.fnmatchcase(name, RESERVED_PATTERN) for name in (path.name, path.resolve().name))


def _module_path(name: str) -> Path:
    return ROOT.joinpath(*name.split(".")).with_suffix(".py")


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


@functools.cache
def _code_modules_at_commit(git_sha: str) -> tuple[str, ...]:
    listing = c503._git("ls-tree", "-r", "--name-only", git_sha, "--", "research")
    if listing.returncode != 0:
        raise ValueError(f"cannot list the tree of commit {git_sha}")
    files = set(listing.stdout.decode().splitlines())
    return _import_closure(lambda n: c503._git_blob(git_sha, c503._module_repo_path(n)).decode()
                           if c503._module_repo_path(n) in files else None)


def _matches_current_summary_code(git_sha: str) -> bool:
    """True iff every ``SUMMARY_MODULES`` file of the producer commit has the bytes of the validating checkout."""
    return all(c503._git_blob(git_sha, c503._module_repo_path(m)) == _module_path(m).read_bytes()
               for m in SUMMARY_MODULES)


def pinned_inputs() -> dict:
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in PINNED_INPUTS}


def run_config_hash(shakedown: bool, protocol: dict, code: str, git_sha, env: dict) -> str:
    return c503._digest({"schema": SCHEMA_VERSION, "shakedown": shakedown, "protocol": protocol, "code_sha256": code,
                         "git_sha": git_sha, "env": {k: env.get(k) for k in c503.ENV_KEYS}})


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
    """The canonical file of this checkout, reached without ``..`` and without a symlink on the way or at the file."""
    path = Path(path)
    relative = _guarded_relative(path, canonical_dest=True)
    pre = c503._open_verified_parent_fd(path, relative, create=False)
    if pre is not None:
        os.close(pre[0])
    return relative


def write_json_atomic(path: Path, payload: dict, *, canonical_dest: bool = False) -> dict:
    """Lossless atomic write through one verified ``dir_fd`` and an ``O_EXCL`` same-directory temp file."""
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


def require_canonical_host(device, git: dict, output: Path) -> None:
    """The canonical run needs an sm_120+ device, a clean committed tree and no existing canonical result."""
    if device.type != "cuda" or not torch.cuda.is_available() or \
            tuple(torch.cuda.get_device_capability(device)) < (12, 0):
        raise RuntimeError(f"{Path(output).name} requires a Blackwell (sm_120+) CUDA device")
    if git.get("git_dirty") is not False or not c503._is_hex(git.get("git_sha"), 40):
        raise RuntimeError("The canonical run needs a clean, committed git tree (git_dirty must be False)")
    if Path(output).exists() or Path(output).is_symlink():
        raise RuntimeError(f"{output} exists: the canonical result is written once")


# ======================================================================================
# Driver
# ======================================================================================
def run(args) -> dict:
    protocol = copy.deepcopy(SHAKEDOWN if args.synthetic_shakedown else CANONICAL)
    canonical = not args.synthetic_shakedown
    device = torch.device(protocol["device"])
    git = h.git_info()
    if canonical:
        require_canonical_host(device, git, args.output)  # before any GPU work
    _require(pinned_inputs() == PINNED_INPUTS, "a pinned input (C5-03 module, C5-03 or C6-01 result) changed")
    numerics = c503.set_numerics()
    log = (lambda *a, **k: None) if args.silent else (lambda *a, **k: print(*a, **k, flush=True))
    started, timestamp_start = time.perf_counter(), time.strftime("%Y-%m-%dT%H:%M:%S%z")
    env = q.environment(device)
    code = code_sha256()
    cfg_hash = run_config_hash(not canonical, protocol, code, git["git_sha"], env)
    meta = {**git, "command": [sys.executable, SCRIPT_REPO_PATH, *args.argv], "argv": list(args.argv),
            "environment": env, "numerics": numerics, "code_sha256": code, "code_modules": list(code_modules()),
            "timestamp_start": timestamp_start, "gpu_telemetry": {"start": h.gpu_telemetry(device)},
            "part_seconds": {}}
    log(f"[c6-02] git {git['git_sha']} dirty={git['git_dirty']} device={env.get('device')} run {cfg_hash} "
        f"canonical={canonical}")
    rq.prime_cupti(device)
    payload = {"schema_version": SCHEMA_VERSION, "task": TASK, "shakedown": not canonical, "canonical": canonical,
               "meta": meta, "run_config_hash": cfg_hash, "protocol": protocol, "pinned_inputs": pinned_inputs()}
    for name, fn in (("kernel_parity", lambda: kernel_parity(device, protocol["parity"]["quick"])),
                     ("sync_check", lambda: sync_check(device))):
        t0 = time.perf_counter()
        payload[name] = fn()
        meta["part_seconds"][name] = time.perf_counter() - t0
    log(f"kernel parity: {payload['kernel_parity']['cases']} cases, {len(payload['kernel_parity']['bitwise_failures'])} "
        f"failures")
    flush = torch.empty(protocol["timing"]["l2_flush_multiple"] * torch.cuda.get_device_properties(device).L2_cache_size,
                        dtype=torch.uint8, device=device)
    for part, fn in (("part_a", part_a), ("part_b", part_b), ("part_c", part_c)):
        t0 = time.perf_counter()
        payload[part] = fn(device, protocol, flush, log)
        meta["part_seconds"][part] = time.perf_counter() - t0
        meta["gpu_telemetry"][f"after_{part}"] = h.gpu_telemetry(device)
        torch.cuda.empty_cache()
    del flush
    payload["pareto"] = pareto(payload["part_c"])
    payload["summary"] = summarize(payload)
    payload["caveats"] = caveats(payload)
    meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["total_seconds"] = time.perf_counter() - started
    payload = json.loads(json.dumps(payload, allow_nan=False))
    try:
        validate_results(payload, canonical=canonical, output=args.output)
    except Exception:
        write_json_atomic(rejected_path(args.output), payload)  # keep the measurements, never at the output
        raise
    return payload


def rejected_path(output: Path) -> Path:
    output = Path(output)
    name = output.name.replace("cycle6_c6_02", "rejected_c6_02")
    return output.with_name(f"{name}.rejected.json")


# ======================================================================================
# Validation
# ======================================================================================
TOP_KEYS = {"schema_version", "task", "shakedown", "canonical", "meta", "run_config_hash", "protocol", "pinned_inputs",
            "kernel_parity", "sync_check", "part_a", "part_b", "part_c", "pareto", "summary", "caveats"}
META_KEYS = {"git_sha", "git_dirty", "command", "argv", "environment", "numerics", "code_sha256", "code_modules",
             "timestamp_start", "gpu_telemetry", "part_seconds", "timestamp_end", "total_seconds"}


def _finite_positive(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _validate_timing(t: dict, methods, squares: int, iters: int, wall_iters: int, orders, where: str) -> None:
    _require(isinstance(t, dict) and set(t) == set(methods), f"{where}: timed methods differ")
    rounds = squares * len(e2e.williams_square(len(methods)))
    _require(isinstance(orders, list) and len(orders) == rounds
             and all(sorted(o) == sorted(methods) for o in orders), f"{where}: Williams orders are invalid")
    square = e2e.williams_square(len(methods))
    names = list(methods)
    _require(all(o == [names[i] for i in square[r % len(square)]] for r, o in enumerate(orders)),
             f"{where}: orders are not the Williams square of the methods")
    for name, v in t.items():
        gpu, wall = v["gpu"], v["wall"]
        samples = gpu["event_samples_ms"]
        _require(len(samples) == rounds and all(len(s) == iters and all(_finite_positive(x) for x in s)
                                                for s in samples), f"{where}/{name}: raw event samples invalid")
        means = [sum(s) / iters for s in samples]
        _require(all(abs(a - b) <= TOL * max(1.0, abs(b)) for a, b in zip(gpu["samples_ms"], means)),
                 f"{where}/{name}: round means differ from the raw samples")
        for stats, values in ((gpu, gpu["samples_ms"]), (wall, wall["samples_ms"])):
            _require(stats["rounds"] == len(values) and all(_finite_positive(x) for x in values),
                     f"{where}/{name}: invalid samples")
            ref = e2e._stats(values)
            _require(all(abs(stats[k] - ref[k]) <= TOL * max(1.0, abs(ref[k])) for k in
                         ("median_ms", "iqr_ms", "min_ms", "max_ms", "mean_ms")), f"{where}/{name}: statistics differ")
        _require(len(wall["samples_ms"]) == rounds * wall_iters, f"{where}/{name}: wall sample count")


def _close(a, b, where: str) -> None:
    """Recomputed derived values must equal the stored ones (floats within TOL)."""
    if isinstance(a, dict):
        _require(isinstance(b, dict) and set(a) == set(b), f"{where}: keys differ")
        for k in a:
            _close(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, list):
        _require(isinstance(b, list) and len(a) == len(b), f"{where}: lengths differ")
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{where}[{i}]")
    elif isinstance(a, float) and not isinstance(b, bool) and isinstance(b, (int, float)):
        _require(abs(a - b) <= TOL * max(1.0, abs(a)), f"{where}: {b} != recomputed {a}")
    else:
        _require(a == b and type(a) is type(b), f"{where}: {b!r} != recomputed {a!r}")


def _validate_profile(method: str, kernels: dict, delegated: bool, where: str) -> None:
    kinds = _kinds(kernels)
    _require(kinds.get("fp4_gemm", 0) == 1 and kinds.get("gemm", 0) == 0, f"{where}/{method}: one FP4 GEMM expected")
    uses_mul = method.startswith("c5_03") or method == GEMM_ONLY[0] or (delegated and method.startswith("c6_02"))
    _require(kinds.get("scale_product", 0) == (1 if uses_mul else 0),
             f"{where}/{method}: the aten::mul scale product must appear exactly on the torch._scaled_mm_v2 paths")
    if method in GEMM_ONLY:
        _require(kinds.get("amax", 0) == kinds.get("quant", 0) == 0, f"{where}/{method}: no quantizer expected")
        return
    static = method.endswith("static_gs")
    _require(kinds.get("quant", 0) == 1 and kinds.get("amax", 0) == (0 if static else 1),
             f"{where}/{method}: quantizer launches")
    impl_names = {"c5_03": (nb._amax_kernel.fn.__name__, nb._quant_kernel.fn.__name__),
                  "c6_02": (nb._amax_kernel.fn.__name__, nb._quant_kernel.fn.__name__) if delegated else
                  (fs._amax_kernel.fn.__name__, fs._quant_kernel.fn.__name__)}[method[:5]]
    quant = [k for k, v in kernels.items() if v["kind"] in ("amax", "quant")]
    _require(set(quant) <= set(impl_names), f"{where}/{method}: quantizer kernels of the wrong implementation")


def _validate_part_a(payload: dict) -> None:
    a, protocol = payload["part_a"], payload["protocol"]
    t = protocol["timing"]
    _require([[r["m"], r["n"], r["k"]] for r in a["shapes"]] == protocol["part_a"]["shapes"], "part_a shapes differ")
    for r in a["shapes"]:
        where = f"part_a {r['m']}x{r['n']}x{r['k']}"
        _require(r["delegated_to_c5_03"] == fs.delegated(r["m"], r["k"]), f"{where}: delegation record")
        expected_checks = {f"{GEMM_ONLY[1]}_equals_public_bitwise", *(f"{v}_c6_02_operands_equal_reference"
                                                                      for v in VARIANTS),
                           *(f"{v}_c6_02_output_equals_c5_03_bitwise" for v in VARIANTS)}
        _require(set(r["checks"]) == expected_checks and all(x is True for x in r["checks"].values()),
                 f"{where}: a bitwise check failed or is missing")
        for mode in ("eager", "cuda_graph"):
            _validate_timing(r[mode], PART_A_METHODS, t["squares_a"], t["iters"], t["wall_iters"],
                             r[f"{mode}_williams_orders"], f"{where}/{mode}")
        _require(set(r["cuda_graph_kernel_profile"]) == set(PART_A_METHODS) - {"bf16_cublas"}, f"{where}: profiles")
        for method, kernels in r["cuda_graph_kernel_profile"].items():
            _validate_profile(method, kernels, r["delegated_to_c5_03"], where)
        _close(summarize_part_a_row(r), r["summary"], f"{where}.summary")
    _require([[x["m"], x["n"], x["k"]] for x in a["ablation"]] == protocol["part_a"]["ablation_shapes"],
             "part_a ablation shapes differ")
    for x in a["ablation"]:
        where = f"part_a ablation {x['m']}x{x['n']}x{x['k']}"
        _require(set(x["checks"]) == {f"{v}_{n}_equals_c5_03_bitwise" for v in ABLATION_VARIANTS
                                      for n in ABLATION_OPTIONS} and all(v is True for v in x["checks"].values()),
                 f"{where}: an ablation output is not bit-identical")
        _validate_timing(x["timing"], ABLATION_METHODS, t["squares_a"], t["iters"], t["wall_iters"],
                         x["williams_orders"], where)
        _close(c503._timing_table(x["timing"], "bf16_cublas"), x["summary"], f"{where}.summary")


def _validate_part_b(payload: dict) -> None:
    b, protocol = payload["part_b"], payload["protocol"]
    tr = b["traces"]
    _require("aten::mul" in tr["torch_scaled_mm_v2"]["aten_ops"]
             and tr["torch_scaled_mm_v2"]["kernel_kinds"].count("scale_product") == 1,
             "part_b: the torch._scaled_mm_v2 trace must show the aten::mul scale product")
    _require("scale_product" not in tr["cublaslt_fused_alpha"]["kernel_kinds"]
             and tr["cublaslt_fused_alpha"]["kernel_kinds"].count("fp4_gemm") == 1,
             "part_b: the direct cuBLASLt call must run one FP4 GEMM and no scale product")
    _require("scale_product" not in tr["c6_02_layer"]["kernel_kinds"]
             and "scale_product" in tr["c5_03_layer"]["kernel_kinds"], "part_b: layer traces")
    combos = [[m, r, d] for m in protocol["part_b"]["shapes"] for r in nb.ROTATIONS for d in protocol["part_b"]["out_dtypes"]]
    _require([[e["mnk"], e["rotation"], e["out_dtype"]] for e in b["exactness"]] == combos, "part_b: exactness grid")
    _require(all(e["bitwise_equal_public"] is True and e["direct"] is True for e in b["exactness"]),
             "part_b: the direct cuBLASLt output differs from nvfp4_scaled_mm_public")
    hz = b["heuristic"]
    _require(hz["mnk"] == protocol["part_b"]["algo_shape"] and hz["returned"] == len(hz["algorithms"]) >= 1,
             "part_b: heuristic record")
    _require(hz["algorithms"][0]["config"] == hz["selected"], "part_b: the selected algorithm is heuristic result 0")


def _validate_part_c(payload: dict) -> None:
    c, protocol = payload["part_c"], payload["protocol"]
    t = protocol["timing"]
    _require(list(c["models"]) == list(protocol["part_c"]["models"]), "part_c models differ")
    for name, row in c["models"].items():
        where = f"part_c {name}"
        _require(row["batch"] == protocol["part_c"]["models"][name]
                 and row["config"] == asdict(PART_C_CONFIGS[name])
                 and row["tokens_per_call"] == row["batch"] * row["config"]["sequence_length"], f"{where}: config")
        expected = {*(f"{v}_c6_02_logits_equal_c5_03_bitwise" for v in VARIANTS),
                    *(f"{f}_c6_02_crossinput_logits_equal_c6_01_bitwise" for f in WIDE_FAMILIES)}
        _require(set(row["checks"]) == expected and all(v is True for v in row["checks"].values()),
                 f"{where}: a bitwise logits check failed")
        _require(set(row["fused_cross_mix_parity"]) == set(WIDE_FAMILIES), f"{where}: parity records")
        for f, p in row["fused_cross_mix_parity"].items():
            _require(p["layers"] == row["nvfp4_layers"] and 0 <= p["mixed_mismatches"] <= p["mixed_values"]
                     and p["logits_bitwise_equal_c6_01"] == (p["logits_mismatches"] == 0)
                     and abs(p["mixed_mismatch_fraction"] - p["mixed_mismatches"] / p["mixed_values"]) <= TOL,
                     f"{where}/{f}: inconsistent parity record")
        _validate_timing(row["timing"], PART_C_METHODS, t["squares_c"], t["iters"], t["wall_iters"],
                         row["williams_orders"], where)
        kinds = row["kernel_kinds_per_forward"]
        layers = row["nvfp4_layers"]
        old, new = kinds["c5_03_had_b16"], kinds["c6_02_had_b16"]
        fused, c601 = kinds["c6_02_fused_cross_full_butterfly_b16"], kinds["c6_01_cross_full_butterfly_b16"]
        # The BF16 parts of the model launch their own elementwise products; the NVFP4 layers differ by one each.
        _require(old.get("fp4_gemm") == new.get("fp4_gemm") == fused.get("fp4_gemm") == c601.get("fp4_gemm") == layers
                 and old.get("scale_product", 0) - new.get("scale_product", 0) == layers
                 and fused.get("scale_product", 0) == new.get("scale_product", 0)
                 and fused.get("cross_mix") == layers and "cross_mix" not in c601
                 and old.get("quant") == new.get("quant") == fused.get("quant") == layers,
                 f"{where}: kernel launches per forward")
        _close(summarize_part_c_model(row), row["summary"], f"{where}.summary")


def _validate_sync(payload: dict) -> None:
    s = payload["sync_check"]
    _require(set(s) == {f"{n}_{v}" for n in ("c6_02", "c6_02_fused_cross") for v in VARIANTS}, "sync_check keys")
    for k, v in s.items():
        _require(v["sync_debug_error_raised"] is False and v["host_syncs_in_region"] == 0 and v["d2h_copies"] == 0
                 and v["graph_replays_bitwise_exact"] == v["graph_replays"] == 4, f"sync_check {k}: failed")


def _validate_meta(payload: dict, canonical: bool, output) -> None:
    meta = payload["meta"]
    _require(set(meta) == META_KEYS, "meta keys differ")
    env = meta["environment"]
    modules = meta["code_modules"]
    if canonical:
        _require(c503._is_hex(meta["git_sha"], 40) and meta["git_dirty"] is False,
                 "a canonical result needs a 40-hex git_sha and a clean tree (git_dirty False)")
        c503._require_git_commit(meta["git_sha"], "meta.git_sha")
        _require(tuple(env["capability"]) >= (12, 0), "a canonical result needs an sm_120+ device")
        _require(modules == list(_code_modules_at_commit(meta["git_sha"])),
                 "meta.code_modules differ from the import closure at meta.git_sha")
        _require(meta["code_sha256"] == c503._code_sha256_at_commit(meta["git_sha"], tuple(modules)),
                 "meta.code_sha256 differs from the committed git blobs")
        _require(_matches_current_summary_code(meta["git_sha"]),
                 "meta.git_sha summary code differs from the validating checkout")
        _require(meta["argv"] == [] and meta["command"][1:] == [SCRIPT_REPO_PATH],
                 "the canonical command takes no arguments")
        if output is not None:
            verify_canonical_output_path(Path(output))
    else:
        _require(modules == list(code_modules()), "meta.code_modules differ from the import closure of this driver")
        _require(meta["code_sha256"] == code_sha256(), "meta.code_sha256 differs from this checkout")
        _require("--synthetic-shakedown" in meta["argv"], "a shakedown records its flag")
    _require(c502._is_python_executable(meta["command"][0]) and meta["command"][1] == SCRIPT_REPO_PATH
             and meta["command"][2:] == meta["argv"], "meta.command is not an invocation of this driver")
    _require(meta["numerics"] == c503.NUMERICS, "numerics differ")
    start = c502._timestamp(meta["timestamp_start"], "meta.timestamp_start")
    end = c502._timestamp(meta["timestamp_end"], "meta.timestamp_end")
    _require(start <= end and _finite_positive(meta["total_seconds"]), "meta timestamps")
    _require(payload["run_config_hash"] == run_config_hash(payload["shakedown"], payload["protocol"],
                                                           meta["code_sha256"], meta["git_sha"], env),
             "run_config_hash differs")


def validate_results(payload: dict, canonical: bool | None = None, output=None) -> dict:
    """Structure, provenance, bitwise checks and every derived number of a C6-02 result. Raises ``ValueError``."""
    try:
        return _validate(payload, canonical, output)
    except (KeyError, TypeError, IndexError, AttributeError, StopIteration) as exc:
        raise ValueError(f"malformed result: {exc!r}") from exc


def _validate(payload: dict, canonical: bool | None, output) -> dict:
    _require(isinstance(payload, dict) and set(payload) == TOP_KEYS, "top-level keys differ")
    _require(payload["schema_version"] == SCHEMA_VERSION and payload["task"] == TASK, "schema or task differs")
    if canonical is None:
        canonical = payload["canonical"]
    _require(payload["canonical"] is canonical and payload["shakedown"] is (not canonical), "canonical flags")
    _require(payload["protocol"] == (CANONICAL if canonical else SHAKEDOWN), "protocol differs from the declared one")
    _require(payload["pinned_inputs"] == PINNED_INPUTS == pinned_inputs(), "pinned inputs differ")
    _validate_meta(payload, canonical, output)
    kp = payload["kernel_parity"]
    _require(kp["bitwise_failures"] == [] and kp["passed"] is True and kp["cases"] == sum(kp["case_kinds"].values()),
             "kernel parity failed")
    if canonical:
        _require(kp["cases"] >= CANONICAL_PARITY_MIN_CASES, f"canonical parity needs {CANONICAL_PARITY_MIN_CASES}+ cases")
    _validate_sync(payload)
    _validate_part_a(payload)
    _validate_part_b(payload)
    _validate_part_c(payload)
    _close(pareto(payload["part_c"]), payload["pareto"], "pareto")
    _close(summarize(payload), payload["summary"], "summary")
    _require(payload["caveats"] == caveats(payload), "caveats differ")
    return payload


# ======================================================================================
# CLI
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("--synthetic-shakedown", action="store_true",
                   help="small shapes and the tiny model (non-canonical; never the canonical file)")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--validate", type=Path, default=None, help="validate an existing result file and exit")
    p.add_argument("--silent", action="store_true")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    args.argv = argv
    if args.validate is not None:
        if args.synthetic_shakedown or args.output is not None:
            raise SystemExit("--validate takes no other option")
        return args
    if args.synthetic_shakedown:
        if args.output is None:
            raise SystemExit("--synthetic-shakedown needs --output (a non-canonical path)")
        if reserved(args.output):
            raise SystemExit(f"{args.output.name} matches the reserved {RESERVED_PATTERN}: a shakedown never writes it")
    else:
        if args.output is not None or args.silent:
            raise SystemExit("the canonical run takes no options; it writes " + RESULT_REPO_PATH)
        args.output = CANONICAL_OUTPUT
        try:
            verify_canonical_output_path(args.output)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.validate is not None:
        payload = json.loads(Path(args.validate).read_text())
        canonical = payload.get("canonical") is True
        validate_results(payload, canonical=canonical, output=args.validate if canonical else None)
        print(f"{args.validate}: valid ({'canonical' if canonical else 'shakedown'})")
        return payload
    payload = run(args)
    write_json_atomic(args.output, payload, canonical_dest=payload["canonical"])
    if not args.silent:
        print(f"wrote {args.output}")
    return payload


if __name__ == "__main__":
    main()
