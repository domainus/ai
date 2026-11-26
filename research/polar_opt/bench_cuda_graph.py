"""CUDA-graph replay benchmark for Muon orthogonalizers on LLM shapes (Cycle 2, T3-01).

Fixes Opus M2 and Astra M2/M4: the Cycle 1 "gpu" clock was CUDA-event time around an eager call,
so small shapes measured host launch latency. Each method now reports three clocks:

* ``cuda_graph_ms``: events around ``graph.replay()`` of a warm capture. No host launches are
  inside the interval, so this is pure GPU execution time.
* ``eager_event_ms``: events around an eager call (the Cycle 1 "gpu" clock).
* ``wall_ms``: host ``perf_counter`` between two synchronizes around an eager call.

Every sample starts after ``flush.zero_()`` on a buffer of 4x the L2 size. Methods run in
counterbalanced blocks: block ``b`` uses row ``b`` of a Williams balanced Latin square. Use a
multiple of the method count as ``--blocks`` (twice it for an odd count) for full balance.
The output keeps raw samples, block order, load average per shape, the git SHA and GPU telemetry.

Register more methods with ``--extra NAME=MODULE:FACTORY`` (same contract as
``harness_tier2_real.py --extra-optimizer``).

Run (alone on the GPU):
    PYTHONPATH=. .venv/bin/python research/polar_opt/bench_cuda_graph.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from research.polar_opt.harness_tier2_real import ROOT, builtin_factories, git_info, gpu_telemetry, parse_extra

SHAPES = [(512, 2048), (1024, 4096), (2048, 8192), (4096, 4096), (4, 1024, 4096)]
DEFAULT_METHODS = ["muon_pe_eager", "muon_pe_compiled", "muon_pe_triton", "muon_ps2_eager",
                   "muon_ps3_eager", "muon_ps2_blackwell", "muon_ps3_blackwell"]


def _stats(samples: list[float]) -> dict:
    q1, q50, q3 = np.quantile(samples, [0.25, 0.5, 0.75])
    return {"median_ms": float(q50), "iqr_ms": float(q3 - q1), "min_ms": float(min(samples)),
            "samples_ms": [round(s, 5) for s in samples]}


def block_order(names: list[str], block: int) -> list[str]:
    """Row ``block`` of a Williams balanced Latin square.

    Over ``n`` blocks (``2n`` for odd ``n``) every method takes every position equally often,
    and every method directly follows every other method equally often.
    """
    n = len(names)
    base = [0]
    for j in range(1, n):
        base.append((j + 1) // 2 if j % 2 else n - j // 2)
    order = [names[(x + block) % n] for x in base]
    return order[::-1] if n % 2 and (block // n) % 2 else order


def _layout(fn, fp32: bool):
    if not fp32:
        return fn
    return lambda G: fn(G).to(dtype=torch.float32, memory_format=torch.contiguous_format)


def capture(fn, G: torch.Tensor, warmup: int):
    """Warm ``fn`` on a side stream, then capture one call. Return (graph, static output)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn(G)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(G)
    torch.cuda.synchronize()
    return graph, out


def bench_shape(methods: dict, shape: tuple, *, flush: torch.Tensor, warmup: int, blocks: int,
                iters: int, seed: int) -> dict:
    """Time every method on one seeded FP32 Gaussian input across isolated clock passes."""
    torch.manual_seed(seed)
    G = torch.randn(*shape, device="cuda")
    return bench_matrix(methods, G, flush=flush, warmup=warmup, blocks=blocks, iters=iters)


def bench_matrix(methods: dict, G: torch.Tensor, *, flush: torch.Tensor, warmup: int, blocks: int,
                 iters: int) -> dict:
    """Time every method on the CUDA input ``G`` across isolated clock passes."""
    shape = tuple(G.shape)
    G_ref = G.detach().clone()
    load_start = os.getloadavg()
    names = list(methods)
    graphs, errors, results = {}, {}, {}
    for name in names:
        for _ in range(warmup):  # eager warmup: autotune, compile and plan caches
            methods[name](G)
        torch.cuda.synchronize()
        try:
            graphs[name] = capture(methods[name], G, warmup)
            graphs[name][0].replay()  # capture records kernels but does not run them
            eager = methods[name](G).float()
            g_out = graphs[name][1].float()
            finite = bool(torch.isfinite(eager).all().item() and torch.isfinite(g_out).all().item())
            diff = (g_out - eager).abs().max().item() if finite else None
            errors[name] = {
                "finite": finite,
                "graph_vs_eager_max_abs": float(diff) if diff is not None and math.isfinite(diff) else None,
            }
        except Exception as error:  # noqa: BLE001 - record and keep the eager clocks
            torch.cuda.synchronize()
            eager = methods[name](G).float()
            finite = bool(torch.isfinite(eager).all().item())
            errors[name] = {"finite": finite, "capture_error": f"{type(error).__name__}: {error}"}
        if not torch.equal(G, G_ref):
            raise RuntimeError(f"Method {name!r} mutated input G in-place")
    samples = {name: {"cuda_graph": [], "eager_event": [], "wall": []} for name in names}
    orders = [block_order(names, block) for block in range(blocks)]

    # Pass 1: CUDA-graph replay (no host launch inside timed region)
    for order in orders:
        for name in order:
            if name not in graphs:
                continue
            for _ in range(iters):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                flush.zero_()
                start.record()
                graphs[name][0].replay()
                end.record()
                samples[name]["cuda_graph"].append((start, end))
    torch.cuda.synchronize()

    # Pass 2: Eager CUDA-event timing (identical idle/flush starting state across all methods)
    for order in orders:
        for name in order:
            for _ in range(iters):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                flush.zero_()
                start.record()
                methods[name](G)
                end.record()
                samples[name]["eager_event"].append((start, end))
    torch.cuda.synchronize()

    # Pass 3: Synchronized host wall-clock timing
    for order in orders:
        for name in order:
            for _ in range(iters):
                flush.zero_()
                torch.cuda.synchronize()
                began = time.perf_counter()
                methods[name](G)
                torch.cuda.synchronize()
                samples[name]["wall"].append((time.perf_counter() - began) * 1e3)
    torch.cuda.synchronize()

    for name in names:
        row = {**errors[name]}
        for clock in ("cuda_graph", "eager_event"):
            values = [s.elapsed_time(e) for s, e in samples[name][clock]]
            row[f"{clock}_ms"] = _stats(values) if values else None
        row["wall_ms"] = _stats(samples[name]["wall"])
        results[name] = row
    graphs.clear()
    del G, G_ref
    torch.cuda.empty_cache()
    return {"shape": list(shape), "loadavg_start": load_start, "loadavg_end": os.getloadavg(),
            "block_orders": orders, "methods": results}


def run(names: list[str], factories: dict, *, shapes=SHAPES, warmup: int = 5, blocks: int | None = None,
        iters: int = 10, seed: int = 0, fp32_layout: bool = False) -> dict:
    if warmup < 5:
        raise ValueError("Use at least 5 warmup calls before capture")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    blocks = blocks or (len(names) if len(names) % 2 == 0 else 2 * len(names))
    properties = torch.cuda.get_device_properties(0)
    flush = torch.empty(4 * properties.L2_cache_size, dtype=torch.uint8, device="cuda")
    methods = {name: _layout(factories[name](), fp32_layout) for name in names}
    device = torch.device("cuda")
    result = {
        "env": {**git_info(), "gpu": properties.name, "l2_cache_bytes": properties.L2_cache_size,
                "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
                "host_cpus": os.cpu_count(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "loadavg_start": os.getloadavg(), "gpu_start": gpu_telemetry(device)},
        "settings": {"warmup": warmup, "blocks": blocks, "iters_per_block": iters, "seed": seed,
                     "input": "torch.randn float32", "flush": "flush.zero_() on 4x L2 before every sample",
                     "layout": "fp32 contiguous" if fp32_layout else "native",
                     "order": "Williams balanced Latin square, one row per block"},
        "shapes": [],
    }
    for shape in shapes:
        row = bench_shape(methods, tuple(shape), flush=flush, warmup=warmup, blocks=blocks, iters=iters, seed=seed)
        row["gpu"] = gpu_telemetry(device)
        result["shapes"].append(row)
        print(f"\n{tuple(shape)}  load {row['loadavg_start'][0]:.2f} -> {row['loadavg_end'][0]:.2f}")
        for name, m in row["methods"].items():
            graph = f"{m['cuda_graph_ms']['median_ms']:8.4f}" if m.get("cuda_graph_ms") else "   n/a  "
            print(f"  {name:<22} graph {graph} ms  event {m['eager_event_ms']['median_ms']:8.4f} ms  "
                  f"wall {m['wall_ms']['median_ms']:8.4f} ms", flush=True)
    result["env"]["loadavg_end"] = os.getloadavg()
    result["env"]["gpu_end"] = gpu_telemetry(device)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--extra", action="append", default=[], metavar="NAME=MODULE:FACTORY")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--blocks", type=int, help="default: method count, doubled when odd (full balance)")
    parser.add_argument("--iters", type=int, default=5, help="samples per method per block")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp32-layout", action="store_true", help="include the contiguous FP32 cast")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cycle2_cuda_graph_bench.json")
    args = parser.parse_args(argv)
    factories = builtin_factories()
    try:
        names = list(dict.fromkeys([*args.methods, *parse_extra(args.extra, factories)]))
    except (ValueError, ImportError, AttributeError) as error:
        parser.error(str(error))
    if any(factories.get(name) is None for name in names):
        parser.error("Methods must be orthogonalizers (not adamw) and must be registered")
    blocks = args.blocks or (len(names) if len(names) % 2 == 0 else 2 * len(names))
    result = run(names, factories, warmup=args.warmup, blocks=blocks, iters=args.iters,
                 seed=args.seed, fp32_layout=args.fp32_layout)
    result["command"] = [sys.executable, *sys.argv]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
