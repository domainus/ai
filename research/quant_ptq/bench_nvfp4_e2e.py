"""End-to-end Blackwell NVFP4 W4A4 inference: latency and real perplexity (Cycle 4 C4-02).

Part A  One linear layer, (M, N, K) shapes from decode to large prefill. Methods: BF16 cuBLAS, the
        pre-quantized NVFP4 GEMM ceiling, the eager PyTorch quantizer + GEMM, and the fused Triton
        quantizer (no rotation, Hadamard-16, butterfly-16, Hadamard-16 with a static scale) + GEMM.
        Stage timings: amax pass, quantize/pack/swizzle pass, ``scaled_mm``. Eager and CUDA graph.
Part B  Full Transformer forward latency (50m, 120m, a wide 2048 x 6-layer model). All block linear
        layers become ``NVFP4Linear``; embeddings, norms, attention and the tied LM head stay BF16.
Part C  Real W4A4 perplexity on all 1,999,872 ``test.bin`` tokens of a 320-step 50m SS3_54 Muon
        model: BF16, W4A16 fake quant and hardware W4A4 (``F.scaled_mm``) for RTN, GPTQ and
        rotated GPTQ, with dynamic and static (``train.bin``-calibrated) activation scales.

Timing design: Williams-balanced orders (every method follows every other method equally often),
CUDA events around each call with a 4x L2 flush (268 MB on the RTX 5080) before every sample, and a
host wall clock that includes launch overhead. TF32 and reduced-precision reductions are off.

Run (canonical, about 25 minutes on an RTX 5080):
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_nvfp4_e2e.py \\
        --data-dir artifacts/real_text --output results/cycle4_c4_02_nvfp4_e2e.json
Smoke run:
    PYTHONPATH=. .venv/bin/python research/quant_ptq/bench_nvfp4_e2e.py --quick --output /tmp/c4.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from research.polar_opt import harness_tier2_real as h
from research.polar_opt.harness_tier2_lm import Transformer
from research.quant_ptq import micro_rotated_ptq as q
from research.quant_ptq import nvfp4_inference_blackwell as nb

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "cycle4_c4_02.v1"
CANONICAL_OUTPUT = ROOT / "results" / "cycle4_c4_02_nvfp4_e2e.json"
QUICK_OUTPUT = ROOT / "results" / "cycle4_c4_02_nvfp4_e2e_quick.json"
FMT = q.FORMATS["nvfp4_g16"]

PART_A_SHAPES = ((16, 4096, 4096), (128, 4096, 4096), (1024, 512, 1408), (1024, 2048, 768),
                 (1024, 4096, 4096), (2048, 8192, 8192))
PART_A_LABELS = {(16, 4096, 4096): "small-batch decode", (1024, 512, 1408): "50m MLP down",
                 (1024, 2048, 768): "120m MLP up"}
PART_A_METHODS = ("bf16_cublas", "nvfp4_gemm_only", "nvfp4_e2e_eager_ref", "nvfp4_e2e_triton_none",
                  "nvfp4_e2e_triton_had_b16", "nvfp4_e2e_triton_bfly_b16", "nvfp4_e2e_triton_had_b16_static_gs")
PART_B_MODELS = {
    "50m": (h.PRESETS["50m"], 16),
    "120m": (h.PRESETS["120m"], 8),
    "w2048_6l": (h.ModelConfig(width=2048, layers=6, heads=16, hidden=5632, sequence_length=512,
                               micro_batch=4, grad_accum=1), 4),
}
PART_C_METHODS = ("rtn", "gptq", "hadamard_b16_rtn", "hadamard_b16_gptq", "adaptive_butterfly_b16_gptq")
ROTATION_OF = {"identity": "none", "hadamard_b16": "hadamard_b16", "adaptive_butterfly_b16": "adaptive_butterfly_b16"}


# ======================================================================================
# Timing
# ======================================================================================
def williams_square(n: int) -> list[list[int]]:
    """Williams design: each treatment directly follows every other one equally often.

    Even ``n`` gives n orders; odd ``n`` adds the mirrored orders (2n in total).
    """
    first, low, high = [0], 1, n - 1
    while len(first) < n:
        first.append(low)
        low += 1
        if len(first) < n:
            first.append(high)
            high -= 1
    rows = [[(v + r) % n for v in first] for r in range(n)]
    return rows if n % 2 == 0 else rows + [row[::-1] for row in rows]


def _stats(values: list[float]) -> dict:
    q1, mid, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return {"median_ms": float(mid), "iqr_ms": float(q3 - q1), "min_ms": float(min(values)),
            "max_ms": float(max(values)), "mean_ms": float(np.mean(values)), "rounds": len(values)}


def williams_timing(functions: dict, *, squares: int, iters: int, flush: Tensor, warmup: int = 3,
                    wall_iters: int = 2) -> dict:
    """Per method: CUDA-event ms (mean of ``iters`` flushed samples per round) and host wall ms."""
    names = list(functions)
    square = williams_square(len(names))
    gpu, wall = {n: [] for n in names}, {n: [] for n in names}
    for fn in functions.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    for r in range(squares * len(square)):
        for index in square[r % len(square)]:
            name, fn = names[index], functions[names[index]]
            fn()
            pairs = []
            for _ in range(iters):
                flush.zero_()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                pairs.append((start, end))
            torch.cuda.synchronize()
            gpu[name].append(sum(s.elapsed_time(e) for s, e in pairs) / iters)
            for _ in range(wall_iters):
                flush.zero_()
                torch.cuda.synchronize()
                began = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                wall[name].append((time.perf_counter() - began) * 1e3)
    return {n: {"gpu": _stats(gpu[n]), "wall": _stats(wall[n])} for n in names}


def capture(fn):
    """CUDA graph of one ``fn()`` call (warmed up on a side stream); returns ``graph.replay``."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph.replay


def _rel(a: Tensor, b: Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


# ======================================================================================
# Part A: single-layer latency and stage breakdown
# ======================================================================================
def part_a_shape(device: torch.device, M: int, N: int, K: int, *, squares: int, iters: int, flush: Tensor,
                 seed: int, log) -> dict:
    """Part A for one (M, N, K): correctness checks, eager and CUDA-graph timings, summary."""
    g = torch.Generator(device=device).manual_seed(seed + M + N + K)
    x = torch.randn(M, K, generator=g, device=device).to(torch.bfloat16)
    W = (torch.randn(N, K, generator=g, device=device) * 0.02).to(torch.bfloat16)
    angles = torch.rand(K // 16, 4, 8, generator=g, device=device) * 2 * math.pi
    layers = {"none": nb.NVFP4Linear.from_weight(W.float(), "none"),
              "had_b16": nb.NVFP4Linear.from_weight(W.float(), "hadamard_b16"),
              "bfly_b16": nb.NVFP4Linear.from_weight(W.float(), "adaptive_butterfly_b16", angles)}
    static_gs = nb.quantize_activation(x, "hadamard_b16").global_scale.clone()
    layers["had_b16_static_gs"] = nb.NVFP4Linear.from_weight(W.float(), "hadamard_b16", act_global_scale=static_gs)
    base = layers["none"]
    pre = nb.quantize_activation(x, "none")
    rot_name = {"none": "none", "had_b16": "hadamard_b16", "bfly_b16": "adaptive_butterfly_b16"}

    def gemm(a, layer=base):
        return nb.nvfp4_scaled_mm(a, layer.w_packed, layer.w_scale, layer.w_global)

    methods = {
        "bf16_cublas": lambda: x @ W.T,
        "nvfp4_gemm_only": lambda: gemm(pre),
        "nvfp4_e2e_eager_ref": lambda: gemm(nb.quantize_activation_ref(x)),
        "nvfp4_e2e_triton_none": lambda: layers["none"](x),
        "nvfp4_e2e_triton_had_b16": lambda: layers["had_b16"](x),
        "nvfp4_e2e_triton_bfly_b16": lambda: layers["bfly_b16"](x),
        "nvfp4_e2e_triton_had_b16_static_gs": lambda: layers["had_b16_static_gs"](x),
    }
    # Stage kernels on preallocated buffers (the module allocates one buffer per call).
    pr, pc = nb.scale_layout(M, K)
    packed = torch.empty(M, K // 2, dtype=torch.uint8, device=device)
    scale = torch.empty(pr * pc, dtype=torch.float8_e4m3fn, device=device)
    partials = torch.empty(nb.MAX_PARTIALS, device=device)
    gs = torch.empty((), device=device)
    stages = {}
    for key, rot in rot_name.items():
        tables = layers[key].tables()
        n_part = nb.launch_amax(x, partials, rot, tables)
        stages[f"stage_amax_{key}"] = lambda rot=rot, tables=tables: nb.launch_amax(x, partials, rot, tables)
        stages[f"stage_quant_{key}"] = lambda rot=rot, tables=tables, n=n_part: nb.launch_quant(
            x, packed, scale, gs, rotation=rot, tables=tables, workspace=partials, n_part=n)
    stages["stage_quant_had_b16_static_gs"] = lambda: nb.launch_quant(x, packed, scale, static_gs,
                                                                      rotation="hadamard_b16")
    functions = {**methods, **stages}

    # Correctness of this shape.
    exact = x.float() @ W.float().T
    checks = {"triton_none_equals_eager_ref_bitwise": bool(torch.equal(methods["nvfp4_e2e_triton_none"](),
                                                                       methods["nvfp4_e2e_eager_ref"]()))}
    for key, layer in layers.items():
        act = layer.quantize_input(x)
        ref = nb.dequantize(act, M, K) @ nb.dequantized_weight(layer).T
        y = layer(x)
        checks[f"{key}_rel_err_vs_fp32_dequant"] = _rel(y, ref)
        checks[f"{key}_rel_err_vs_bf16_exact"] = _rel(y, exact)
    del exact

    eager = williams_timing(functions, squares=squares, iters=iters, flush=flush)
    graphs, graph_errors = {}, {}
    for name, fn in functions.items():
        try:
            graphs[name] = capture(fn)
        except Exception as error:  # noqa: BLE001 - record why a method cannot be captured
            graph_errors[name] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
            torch.cuda.synchronize()
    graphed = williams_timing(graphs, squares=squares, iters=iters, flush=flush)
    flops = 2.0 * M * N * K
    row = {"m": M, "n": N, "k": K, "label": PART_A_LABELS.get((M, N, K), ""), "checks": checks,
           "graph_capture_errors": graph_errors, "eager": eager, "cuda_graph": graphed}
    summary = {}
    for mode, timing in (("eager", eager), ("cuda_graph", graphed)):
        if "bf16_cublas" not in timing:
            continue
        bf16 = timing["bf16_cublas"]["gpu"]["median_ms"]
        ceiling = timing["nvfp4_gemm_only"]["gpu"]["median_ms"]
        table = {}
        for name in PART_A_METHODS:
            if name not in timing:
                continue
            ms = timing[name]["gpu"]["median_ms"]
            table[name] = {"ms": ms, "wall_ms": timing[name]["wall"]["median_ms"],
                           "tflops": flops / (ms * 1e-3) / 1e12, "speedup_vs_bf16": bf16 / ms,
                           "pct_of_gemm_ceiling_retained": 100.0 * ceiling / ms}
        for key in rot_name:
            name = "nvfp4_e2e_triton_" + key
            table[name]["stages"] = {"amax_ms": timing[f"stage_amax_{key}"]["gpu"]["median_ms"],
                                     "quant_pack_swizzle_ms": timing[f"stage_quant_{key}"]["gpu"]["median_ms"],
                                     "scaled_mm_ms": ceiling, "ms": table[name]["ms"]}
        table["nvfp4_e2e_triton_had_b16_static_gs"]["stages"] = {
            "amax_ms": 0.0, "quant_pack_swizzle_ms": timing["stage_quant_had_b16_static_gs"]["gpu"]["median_ms"],
            "scaled_mm_ms": ceiling, "ms": table["nvfp4_e2e_triton_had_b16_static_gs"]["ms"]}
        summary[mode] = table
    row["summary"] = summary
    g_tab = summary.get("cuda_graph", summary["eager"])
    log(f"  A ({M:5d},{N:5d},{K:5d}) graph ms: bf16 {g_tab['bf16_cublas']['ms']:.4f}  "
        f"gemm {g_tab['nvfp4_gemm_only']['ms']:.4f}  ref {summary['eager']['nvfp4_e2e_eager_ref']['ms']:.4f}(eager)  "
        f"none {g_tab['nvfp4_e2e_triton_none']['ms']:.4f}  had {g_tab['nvfp4_e2e_triton_had_b16']['ms']:.4f}  "
        f"bfly {g_tab['nvfp4_e2e_triton_bfly_b16']['ms']:.4f}  "
        f"had_static {g_tab['nvfp4_e2e_triton_had_b16_static_gs']['ms']:.4f}")
    return row


def part_a(device: torch.device, shapes, *, squares: int, iters: int, flush: Tensor, seed: int, log) -> dict:
    rows = []
    for M, N, K in shapes:
        rows.append(part_a_shape(device, M, N, K, squares=squares, iters=iters, flush=flush, seed=seed, log=log))
        torch.cuda.empty_cache()
    return {"design": "Williams-balanced orders over all methods and stage kernels; CUDA events, mean of "
                      "`iters` samples per round, 4x L2 flush before every sample; median over rounds. "
                      "`wall_ms`: host clock between synchronizes (includes launch overhead). "
                      "Stage kernels run on preallocated buffers; e2e methods allocate per call.",
            "squares": squares, "iters": iters, "l2_flush_bytes": flush.numel(), "shapes": rows}


# ======================================================================================
# Part B: full Transformer forward latency
# ======================================================================================
def nvfp4_model(model: Transformer, rotation: str) -> Transformer:
    """Copy of a BF16 model with every block linear replaced by an RTN ``NVFP4Linear``."""
    out = copy.deepcopy(model)
    for block in out.blocks:
        for kind in q.LINEAR_KINDS:
            linear = getattr(block, kind)
            setattr(block, kind, nb.NVFP4Linear.from_weight(linear.weight.float(), rotation))
    return out


def part_b(device: torch.device, models: dict, *, squares: int, iters: int, flush: Tensor, seed: int, log) -> dict:
    rows = {}
    for name, (config, batch) in models.items():
        torch.manual_seed(seed)
        with torch.device(device):
            model = Transformer(config).to(torch.bfloat16).eval()
        tokens = torch.randint(0, config.vocab_size, (batch, config.sequence_length), device=device)
        variants = {"none": nvfp4_model(model, "none"), "had_b16": nvfp4_model(model, "hadamard_b16")}
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

        functions = {"bf16_eager": forward(model), "bf16_compiled": forward(compiled),
                     "nvfp4_triton_none": forward(variants["none"]),
                     "nvfp4_triton_had_b16": forward(variants["had_b16"])}
        with torch.inference_mode():
            reference = model(tokens).float()
            checks = {f"{k}_logits_rel_diff_vs_bf16": _rel(v(), reference) for k, v in functions.items()}
        del reference
        for key in ("bf16_eager", "nvfp4_triton_none", "nvfp4_triton_had_b16"):
            functions[key + "_graph"] = capture(functions[key])
        timing = williams_timing(functions, squares=squares, iters=iters, flush=flush, wall_iters=1)
        tokens_per_call = batch * config.sequence_length
        table = {}
        base_eager = timing["bf16_eager"]["gpu"]["median_ms"]
        base_compiled = timing["bf16_compiled"]["gpu"]["median_ms"]
        base_graph = timing["bf16_eager_graph"]["gpu"]["median_ms"]
        for key, t in timing.items():
            ms = t["gpu"]["median_ms"]
            table[key] = {"ms": ms, "wall_ms": t["wall"]["median_ms"], "tokens_per_s": tokens_per_call / (ms * 1e-3),
                          "speedup_vs_bf16_eager": base_eager / ms, "speedup_vs_bf16_compiled": base_compiled / ms,
                          "speedup_vs_bf16_eager_graph": base_graph / ms}
        linear_params = sum(getattr(b, k).weight.numel() for b in model.blocks for k in q.LINEAR_KINDS)
        rows[name] = {"config": asdict(config), "batch": batch, "tokens_per_call": tokens_per_call,
                      "linear_parameters": linear_params,
                      "total_parameters": sum(p.numel() for p in model.parameters()),
                      "compile_seconds": compile_seconds, "checks": checks, "timing": timing, "summary": table}
        log(f"  B {name}: " + "  ".join(f"{k} {v['ms']:.3f}" for k, v in table.items()))
        del model, compiled, variants, functions
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    return {"design": "random-init BF16 weights, random tokens; NVFP4 variants replace every block linear "
                      "(qkv, out, gate, up, down) by an RTN NVFP4Linear with dynamic activation scales. "
                      "Embedding, RMSNorms, attention (SDPA) and the tied LM head stay BF16. `_graph`: one CUDA "
                      "graph per forward. bf16_compiled: torch.compile default mode.",
            "squares": squares, "iters": iters, "models": rows}


# ======================================================================================
# Part C: real W4A4 perplexity
# ======================================================================================
def _swap(model, modules: dict[str, nn.Module]) -> dict[str, nn.Module]:
    """Replace ``blocks.i.kind`` modules; returns the previous ones."""
    previous = {}
    for name, module in modules.items():
        _, i, kind = name.split(".")
        block = model.blocks[int(i)]
        previous[name] = getattr(block, kind)
        setattr(block, kind, module)
    return previous


def _layer_rotation(rot: q.Rotation, i: int) -> q.Rotation:
    return q.Butterfly(16, rot.angles[i]) if isinstance(rot, q.Butterfly) else rot


@torch.no_grad()
def calibrate_activation_scales(model, windows: np.ndarray, batch: int, device: torch.device,
                                rotations: dict[str, q.Rotation]) -> dict[str, Tensor]:
    """Static FP32 activation scales amax(x R^T) / (6 * 448) of every linear input, from ``windows``."""
    linears = q.quantized_linears(model)
    amax = {name: torch.zeros((), device=device) for name in linears}

    def hook(name):
        def record(module, inputs):
            with torch.autocast(device_type=device.type, enabled=False):
                z = rotations[name].forward(inputs[0].detach().reshape(-1, module.in_features).float())
                amax[name] = torch.maximum(amax[name], z.abs().amax())
        return record

    handles = [m.register_forward_pre_hook(hook(n)) for n, m in linears.items()]
    try:
        for first in range(0, len(windows), batch):
            x = torch.from_numpy(windows[first:first + batch]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                model(x)
    finally:
        for handle in handles:
            handle.remove()
    return {n: q.nvfp4_global_scale(a.reshape(1, 1)).reshape(()) for n, a in amax.items()}


class EmulatedW4A4Linear(nn.Module):
    """FP32 emulation of an ``NVFP4Linear``: same quantized operands, FP32 GEMM, no FP4 hardware."""

    def __init__(self, layer: nb.NVFP4Linear):
        super().__init__()
        self.layer = layer
        self.register_buffer("weight_values", nb.dequantized_weight(layer))

    def forward(self, x):
        lead = x.shape[:-1]
        with torch.autocast(device_type=x.device.type, enabled=False):
            x2 = x.reshape(-1, self.layer.in_features)
            values = nb.dequantize(self.layer.quantize_input(x2), *x2.shape)
            return (values @ self.weight_values.T).to(torch.bfloat16).reshape(*lead, -1)


def part_c(args, device: torch.device, log) -> dict:
    t0 = time.perf_counter()
    data = q.load_data(args.data_dir, verify=not args.no_verify, expected=None if args.no_verify else q.EXPECTED_SHA256)
    config = h.PRESETS[args.preset]
    log(f"  C training {args.preset} muon_ss3_54 for {args.steps} steps (seed {args.seed})")
    model, training = q.train_model("muon_ss3_54", config, data, device, steps=args.steps, seed=args.seed,
                                    warmup_steps=args.warmup_steps, eval_tokens=args.harness_eval_tokens)
    log(f"  C trained in {time.perf_counter() - t0:.0f}s, last loss {training['last_train_loss']:.4f}")
    test = data.splits["test"]
    seq, batch = config.sequence_length, config.micro_batch
    calibration = q.calibration_windows(data, args.calibration_sequences, seq, args.calibration_seed)

    def evaluate():
        return q.evaluate_split(model, test, seq, batch, device, args.max_eval_windows)

    baseline = evaluate()
    log(f"  C BF16 test NLL {baseline['nll']:.4f} PPL {baseline['ppl']:.3f} on {baseline['tokens']:,} tokens")
    H = q.collect_hessians(model, calibration["array"], batch, device)
    linears = q.quantized_linears(model)
    kinds = {k: [f"blocks.{i}.{k}" for i in range(config.layers)] for k in q.LINEAR_KINDS}
    W = {k: torch.stack([linears[n].weight.detach().float().clone() for n in names]) for k, names in kinds.items()}
    Hc = {k: torch.stack([H[n] for n in names]) for k, names in kinds.items()}
    del H
    rotations, butterfly_fits = {}, {}
    for kind in q.LINEAR_KINDS:
        rotations[(kind, "identity")] = q.Rotation()
        rotations[(kind, "hadamard_b16")] = q.BlockHadamard(16)
        rotations[(kind, "adaptive_butterfly_b16")], butterfly_fits[kind] = q.make_rotation(
            "adaptive_butterfly_b16", W[kind], FMT, butterfly_steps=args.butterfly_steps)

    def row(mode: str, method: str, result: dict, seconds: float, **extra) -> dict:
        if result["tokens"] != baseline["tokens"]:
            raise RuntimeError("Evaluation token count changed between runs")
        return {"method": method, "mode": mode, "test_nll": result["nll"], "test_ppl": result["ppl"],
                "test_tokens": result["tokens"], "delta_nll": result["nll"] - baseline["nll"],
                "delta_ppl": result["ppl"] - baseline["ppl"],
                "relative_delta_ppl_pct": 100.0 * (result["ppl"] / baseline["ppl"] - 1.0), "seconds": seconds, **extra}

    grid = []
    for method in PART_C_METHODS:
        family, rounding = q.parse_method(method)
        rotation = ROTATION_OF[family]
        t1 = time.perf_counter()
        effective, rotated, hardware, emulate = {}, {}, {}, method == "hadamard_b16_gptq"
        damping = []
        for kind, names in kinds.items():
            rot = rotations[(kind, family)]
            Wr = rot.forward(W[kind])
            if rounding == "gptq":
                Qr, info = q.gptq_quantize(Wr, rot.rotate_hessian(Hc[kind]), FMT, return_encoding=True)
                enc = info["encoding"]
                damping += info["damping_lambda"]
                codes, blocks, scales = enc["codes"], enc["block_scale"], enc["global_scale"]
            else:
                encs = [q.nvfp4_encode(Wr[i]) for i in range(len(names))]
                Qr = torch.stack([e["values"] for e in encs])
                codes = torch.stack([e["codes"] for e in encs])
                blocks = torch.stack([e["block_scale"] for e in encs])
                scales = torch.stack([e["global_scale"] for e in encs])
            effective[kind], rotated[kind] = rot.inverse(Qr), Qr
            for i, name in enumerate(names):
                angles = rot.angles[i] if isinstance(rot, q.Butterfly) else None
                hardware[name] = nb.NVFP4Linear.from_encoding(codes[i], blocks[i], scales[i], rotation=rotation,
                                                              angles=angles)
        quant_seconds = time.perf_counter() - t1
        # The hardware weights decode exactly to the fake-quant weights in rotated coordinates.
        weights_exact = all(torch.equal(nb.dequantized_weight(hardware[names[0]]), rotated[kind][0])
                            for kind, names in kinds.items())

        # W4A16 fake quant (dequantized weights, BF16 autocast activations), as in Cycle 3 T5-01.
        t1 = time.perf_counter()
        with torch.no_grad():
            for kind, names in kinds.items():
                for i, name in enumerate(names):
                    linears[name].weight.copy_(effective[kind][i])
        result = evaluate()
        with torch.no_grad():
            for kind, names in kinds.items():
                for i, name in enumerate(names):
                    linears[name].weight.copy_(W[kind][i])
        grid.append(row("w4a16_fakequant", method, result, time.perf_counter() - t1,
                        quantize_seconds=quant_seconds, gptq_damping_max=max(damping) if damping else None,
                        hw_weights_decode_to_fakequant_weights=weights_exact))

        # Hardware W4A4, dynamic per-tensor activation scale.
        t1 = time.perf_counter()
        previous = _swap(model, hardware)
        result = evaluate()
        _swap(model, previous)
        grid.append(row("w4a4_hw_dynamic", method, result, time.perf_counter() - t1))

        # Hardware W4A4, static activation scales calibrated on the train.bin calibration windows.
        t1 = time.perf_counter()
        layer_rotations = {name: _layer_rotation(rotations[(name.split(".")[2], family)], int(name.split(".")[1]))
                           for name in hardware}
        static = calibrate_activation_scales(model, calibration["array"], batch, device, layer_rotations)
        for name, layer in hardware.items():
            layer.act_global_scale = static[name]
        previous = _swap(model, hardware)
        result = evaluate()
        _swap(model, previous)
        grid.append(row("w4a4_hw_static", method, result, time.perf_counter() - t1,
                        static_scale_calibration="train.bin calibration windows, BF16 model, max |x R^T| / 2688",
                        static_global_scale_range=[min(v.item() for v in static.values()),
                                                   max(v.item() for v in static.values())]))

        if emulate:  # FP32 emulation of the same W4A4 operands: isolates the FP4 GEMM hardware.
            for layer in hardware.values():
                layer.act_global_scale = None
            t1 = time.perf_counter()
            previous = _swap(model, {n: EmulatedW4A4Linear(m) for n, m in hardware.items()})
            result = evaluate()
            _swap(model, previous)
            grid.append(row("w4a4_emulated_fp32_dynamic", method, result, time.perf_counter() - t1))
        for r in grid:
            if r["method"] == method:
                log(f"  C {method:<28} {r['mode']:<26} PPL {r['test_ppl']:9.3f} dNLL {r['delta_nll']:+.4f} "
                    f"({r['relative_delta_ppl_pct']:+.2f}%)")
        del hardware, effective, rotated
        torch.cuda.empty_cache()
    return {"preset": args.preset, "config": asdict(config), "training": training, "baseline_test": baseline,
            "calibration": {k: v for k, v in calibration.items() if k != "array"},
            "butterfly_fits": butterfly_fits, "grid": grid, "seconds": time.perf_counter() - t0,
            "data": {**data.summary(), "directory": str(args.data_dir), "sha256_verified": not args.no_verify,
                     "evaluation_split": "test", "calibration_split": "train",
                     "static_scale_split": "train", "test_used_for": "evaluation only"}}


# ======================================================================================
# Kernel parity summary (recorded with the results)
# ======================================================================================
def kernel_parity(device: torch.device, quick: bool) -> dict:
    shapes = [(m, k) for m in (1, 37, 129) for k in (64, 80, 768)] if quick else \
        [(m, k) for m in (1, 37, 128, 129, 256, 512) for k in (64, 80, 512, 768, 1408, 2048)]
    failures, cases = [], 0
    for M, K in shapes:
        g = torch.Generator(device=device).manual_seed(M * 31 + K)
        x = (torch.randn(M, K, generator=g, device=device) * 3).to(torch.bfloat16)
        angles = torch.rand(K // 16, 4, 8, generator=g, device=device) * 2 * math.pi
        for rotation in nb.ROTATIONS:
            a_ = angles if rotation == "adaptive_butterfly_b16" else None
            tables = nb.butterfly_tables(a_) if a_ is not None else None
            ref = nb.quantize_activation_ref(x, rotation, a_)
            for static in (None, ref.global_scale * 0.5):
                cases += 1
                got = nb.quantize_activation(x, rotation, tables, static)
                want = ref if static is None else nb.quantize_activation_ref(x, rotation, a_, static)
                same = (torch.equal(got.packed.view(torch.uint8), want.packed.view(torch.uint8))
                        and torch.equal(got.scale.view(torch.uint8), want.scale.view(torch.uint8))
                        and torch.equal(got.global_scale, want.global_scale))
                if not same:
                    failures.append([M, K, rotation, static is not None])
    return {"cases": cases, "bitwise_failures": failures, "passed": not failures,
            "reference": "micro_rotated_ptq rotation (FP32) + nvfp4_encode + pack_e2m1 + to_blocked"}


# ======================================================================================
# Driver
# ======================================================================================
def run_experiment(args) -> dict:
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    log = (lambda *a, **k: None) if args.silent else (lambda *a, **k: print(*a, **k, flush=True))
    started = time.perf_counter()
    meta = {**h.git_info(), "command": [sys.executable, *sys.argv], "args": {k: str(v) for k, v in vars(args).items()},
            "environment": q.environment(device), "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "loadavg_start": list(os.getloadavg()), "gpu_telemetry": {"start": h.gpu_telemetry(device)},
            "numerics": {"allow_tf32": False, "cudnn_allow_tf32": False,
                         "allow_fp16_reduced_precision_reduction": False,
                         "allow_bf16_reduced_precision_reduction": False}}
    log(f"git {meta['git_sha']} dirty={meta['git_dirty']} device={meta['environment'].get('device')}")
    payload = {"schema_version": SCHEMA_VERSION, "task": "Cycle 4 C4-02: end-to-end Blackwell NVFP4 W4A4 inference",
               "quick": args.quick, "meta": meta}
    payload["kernel_parity"] = kernel_parity(device, args.quick)
    log(f"kernel parity: {payload['kernel_parity']['cases']} cases, passed={payload['kernel_parity']['passed']}")
    flush = torch.empty(4 * torch.cuda.get_device_properties(device).L2_cache_size, dtype=torch.uint8, device=device)
    if "A" in args.parts:
        shapes = ((16, 256, 256), (129, 512, 768)) if args.quick else PART_A_SHAPES
        payload["part_a"] = part_a(device, shapes, squares=args.squares, iters=args.iters, flush=flush,
                                   seed=args.seed, log=log)
        meta["gpu_telemetry"]["after_part_a"] = h.gpu_telemetry(device)
    if "B" in args.parts:
        models = ({"tiny": (h.PRESETS["tiny"], 4)} if args.quick else PART_B_MODELS)
        payload["part_b"] = part_b(device, models, squares=1 if args.quick else args.squares, iters=args.iters,
                                   flush=flush, seed=args.seed, log=log)
        meta["gpu_telemetry"]["after_part_b"] = h.gpu_telemetry(device)
    del flush
    torch.cuda.empty_cache()
    if "C" in args.parts:
        payload["part_c"] = part_c(args, device, log)
        meta["gpu_telemetry"]["after_part_c"] = h.gpu_telemetry(device)
    payload["summary"] = summarize(payload)
    payload["caveats"] = caveats(payload)
    meta["gpu_telemetry"]["end"] = h.gpu_telemetry(device)
    meta["loadavg_end"] = list(os.getloadavg())
    meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["total_seconds"] = time.perf_counter() - started
    validate_results(payload, parts=args.parts)
    return payload


def summarize(payload: dict) -> dict:
    out = {}
    if "part_a" in payload:
        out["part_a"] = []
        for row in payload["part_a"]["shapes"]:
            entry = {"mnk": [row["m"], row["n"], row["k"]], "label": row["label"]}
            for mode, table in row["summary"].items():
                entry[mode] = {name: {k: v for k, v in t.items() if k != "stages"} for name, t in table.items()}
            graph = row["summary"].get("cuda_graph", {})
            if "nvfp4_e2e_triton_had_b16" in graph:
                entry["cuda_graph_stages_had_b16"] = graph["nvfp4_e2e_triton_had_b16"]["stages"]
            out["part_a"].append(entry)
    if "part_b" in payload:
        out["part_b"] = {name: {k: {"ms": v["ms"], "speedup_vs_bf16_eager": v["speedup_vs_bf16_eager"],
                                    "speedup_vs_bf16_compiled": v["speedup_vs_bf16_compiled"]}
                                for k, v in m["summary"].items()} for name, m in payload["part_b"]["models"].items()}
    if "part_c" in payload:
        c = payload["part_c"]
        out["part_c"] = {"baseline_test_nll": c["baseline_test"]["nll"], "baseline_test_ppl": c["baseline_test"]["ppl"],
                         "test_tokens": c["baseline_test"]["tokens"],
                         "table": {f"{r['method']}/{r['mode']}": {k: r[k] for k in ("test_nll", "test_ppl", "delta_nll",
                                                                                      "relative_delta_ppl_pct")}
                                   for r in c["grid"]}}
        hw = [r for r in c["grid"] if r["mode"] == "w4a4_hw_dynamic"]
        out["part_c"]["best_w4a4_hw_dynamic"] = min(hw, key=lambda r: r["test_nll"])["method"]
    return out


def caveats(payload: dict) -> list[str]:
    notes = [
        "Part A/B weights and activations are random; latency does not depend on values. Part C uses a real "
        "320-step single-seed 50m checkpoint, so W4A4 deltas are for one small, briefly trained model.",
        "Dynamic activation scales cost a second read of the activation (amax pass); the static path skips it "
        "but saturates activations above the train.bin calibration maximum.",
        "Static scales are calibrated on the BF16 model's activations, not on the upstream-quantized model.",
        "Under autocast, NVFP4Linear quantizes whatever dtype reaches it (FP32 or BF16) directly; the W4A16 "
        "reference casts activations to BF16 and weights (dequantized FP32) to BF16 before the GEMM.",
        "bf16_compiled is torch.compile(default); NVFP4 models run eager or as CUDA graphs, not compiled.",
        "CUDA-event resolution on this GPU is about 2 us, so single-call sub-10-us numbers are averages of "
        "quantized samples (mean over iters per round, median over rounds).",
    ]
    rows = payload.get("part_a", {}).get("shapes", [])
    slower = [f"({r['m']},{r['n']},{r['k']})" for r in rows
              if r["summary"].get("cuda_graph", {}).get("nvfp4_e2e_triton_had_b16", {}).get("speedup_vs_bf16", 2) < 1]
    if slower:
        notes.append("NVFP4 e2e (had_b16, CUDA graph) is slower than BF16 cuBLAS at " + ", ".join(slower)
                     + ": activation quantization and the FP4 GEMM do not amortize at these sizes.")
    return notes


def validate_results(payload: dict, parts: str = "ABC") -> None:
    json.dumps(payload, allow_nan=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected schema version")
    for key in ("git_sha", "git_dirty", "command", "environment", "gpu_telemetry", "loadavg_start", "loadavg_end"):
        if key not in payload["meta"]:
            raise ValueError(f"Missing meta key {key!r}")
    if not payload["kernel_parity"]["passed"]:
        raise ValueError(f"Kernel parity failed: {payload['kernel_parity']['bitwise_failures'][:5]}")
    if "C" in parts:
        c = payload["part_c"]
        if c["calibration"]["split"] != "train" or c["data"]["evaluation_split"] != "test":
            raise ValueError("Calibration must use train.bin and evaluation test.bin")
        if any(r["test_tokens"] != c["baseline_test"]["tokens"] for r in c["grid"]):
            raise ValueError("Every Part C row must use the baseline test tokens")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parts", default="ABC", help="subset of A, B, C")
    parser.add_argument("--squares", type=int, default=2, help="complete Williams squares per timing (Part A/B)")
    parser.add_argument("--iters", type=int, default=10, help="flushed samples per method and round")
    parser.add_argument("--preset", default="50m", choices=sorted(h.PRESETS))
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--calibration-sequences", type=int, default=128)
    parser.add_argument("--calibration-seed", type=int, default=20261126)
    parser.add_argument("--max-eval-windows", type=int, default=None, help="limit test windows (default: all)")
    parser.add_argument("--harness-eval-tokens", type=int, default=1 << 20)
    parser.add_argument("--butterfly-steps", type=int, default=150)
    parser.add_argument("--no-verify", action="store_true", help="skip SHA-256 checks (synthetic shards only)")
    parser.add_argument("--quick", action="store_true", help="tiny shapes, tiny model, short evaluation")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


_CANONICAL = {"parts": "ABC", "squares": 2, "iters": 10, "preset": "50m", "steps": 320, "warmup_steps": 32,
              "seed": 1, "calibration_sequences": 128, "calibration_seed": 20261126, "max_eval_windows": None,
              "harness_eval_tokens": 1 << 20, "butterfly_steps": 150, "no_verify": False, "quick": False}


def _is_canonical_args(args: argparse.Namespace) -> bool:
    return all(getattr(args, k) == v for k, v in _CANONICAL.items())


def _resolve_output(args: argparse.Namespace) -> Path:
    canonical = _is_canonical_args(args)
    if args.output is None:
        if args.quick:
            return QUICK_OUTPUT
        if canonical:
            return CANONICAL_OUTPUT
        raise ValueError(f"Non-canonical run requires an explicit --output path to avoid overwriting "
                         f"{CANONICAL_OUTPUT.name}")
    if args.output.resolve() == CANONICAL_OUTPUT.resolve() and not canonical:
        raise ValueError(f"Refusing to overwrite canonical {CANONICAL_OUTPUT.name} from a --quick or "
                         "non-canonical run; choose a different --output path")
    return args.output


def parse_args(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.data_dir is None:
        args.data_dir = q.default_data_dir()
    args.parts = "".join(sorted(set(args.parts.upper())))
    if not set(args.parts) <= set("ABC") or not args.parts:
        parser.error("--parts must be a non-empty subset of ABC")
    if args.quick:
        try:
            args.output = _resolve_output(args)  # judge the guard on the arguments as given
        except ValueError as exc:
            parser.error(str(exc))
        args.preset, args.steps, args.warmup_steps = "tiny", 6, 2
        args.calibration_sequences, args.max_eval_windows = 8, args.max_eval_windows or 64
        args.harness_eval_tokens, args.butterfly_steps, args.iters = 4096, 10, 3
        args.squares = 1
        return args
    try:
        args.output = _resolve_output(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    payload = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(args.output)
    if not args.silent:
        print(f"Wrote {args.output}")
    return payload


if __name__ == "__main__":
    main()
