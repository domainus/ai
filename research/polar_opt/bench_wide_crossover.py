"""Wide-layer full-model Muon orthogonalization crossover benchmark (Cycle 3, C3-01).

One sample is one full-model optimizer step of orthogonalization: every stacked same-shape
matrix group of a 12-layer Transformer, in order. For width ``d`` the groups are ``(48, d, d)``
(Q, K, V, O), ``(24, hidden, d)`` (MLP gate and up) and ``(12, d, hidden)`` (MLP down), with
``hidden = mlp_hidden(d)``. Width 768 has two more rows for the ``120m`` preset (hidden 2048):
``w768_120m`` keeps the layout above, and ``w768_120m_harness`` uses the groups that
``harness_tier2_real.py`` really steps (its QKV projection is one fused ``(3d, d)`` matrix).

The method follows ``bench_cuda_graph.py``: seeded FP32 Gaussian inputs, ``flush.zero_()`` on 4x
the L2 size before every sample, Williams balanced block order, and one isolated pass per clock.

* ``cuda_graph``: events around ``graph.replay()`` of a warm capture of the whole step.
* ``eager_event``: events around an eager call of the whole step.

A captured 2048-wide step holds about 3 GiB, so seven live graphs do not fit on a 16 GB card.
The graph pass therefore keeps one graph alive at a time: each block slot captures the step,
replays it once untimed, takes its samples, and checks the replayed outputs against eager.

The run stops unless every output is finite, every graph output equals the eager output bit for
bit in every shape group, and the inputs are unchanged.

Run (alone on the GPU, from a clean commit):
    PYTHONPATH=. .venv/bin/python research/polar_opt/bench_wide_crossover.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import torch

from research.polar_opt import tier2_candidates as candidates
from research.polar_opt.bench_cuda_graph import _stats, block_order, capture
from research.polar_opt.harness_tier2_real import (PRESETS, ROOT, Transformer, builtin_factories,
                                                   git_info, gpu_telemetry)

WIDTHS = [512, 768, 1024, 1536, 2048]
LAYERS = 12
CLOCKS = ("eager_event", "cuda_graph")
REFERENCES = ("ps3_blackwell", "pe_compiled")
CROSSOVER_METHODS = ("ss3_54", "ss3_53")
QUICK = {"widths": [64, 128], "layers": 2, "blocks": 2, "iters": 2}
CANONICAL_OUTPUT = ROOT / "results" / "cycle3_c3_01_wide_crossover.json"
QUICK_OUTPUT = ROOT / "results" / "cycle3_c3_01_wide_crossover_quick.json"
_INT_VIEW = {2: torch.int16, 4: torch.int32, 8: torch.int64}


def method_factories() -> dict:
    """Name -> zero-argument factory of ``f(G) -> U``. PE works in BF16, the others in FP16."""
    builtin = builtin_factories()
    return {
        "pe_eager": builtin["muon_pe_eager"],
        "pe_compiled": builtin["muon_pe_compiled"],
        "ps3_eager": builtin["muon_ps3_eager"],
        "ps3_blackwell": builtin["muon_ps3_blackwell"],
        "ss3_54": lambda: candidates.ss3_54_blackwell_fp16,
        "ss3_53": lambda: candidates.ss3_53_blackwell_fp16,
        "ss3_35": lambda: candidates.ss3_35_blackwell_fp16,
    }


def default_blocks(count: int) -> int:
    """Smallest block count with full Williams balance for ``count`` methods."""
    return count if count % 2 == 0 else 2 * count


def mlp_hidden(width: int) -> int:
    return ((int(round(2.75 * width)) + 127) // 128) * 128


def step_groups(width: int, hidden: int, layers: int = LAYERS) -> list[tuple]:
    return [(4 * layers, width, width), (2 * layers, hidden, width), (layers, width, hidden)]


def harness_groups(preset: str) -> list[tuple]:
    """Stacked Muon groups of a harness preset, in the order its optimizer steps them."""
    with torch.device("meta"):
        matrices = Transformer(PRESETS[preset]).matrix_parameters().values()
    return [(count, *shape) for shape, count in Counter(tuple(p.shape) for p in matrices).items()]


def step_configs(widths=WIDTHS, layers: int = LAYERS) -> list[dict]:
    """One row per grid width, plus the two ``120m`` preset rows next to its width."""
    def row(label, width, hidden, groups, grid):
        return {"label": label, "width": width, "hidden": hidden, "grid": grid,
                "groups": [list(g) for g in groups]}

    preset = PRESETS["120m"]
    configs = []
    for width in widths:
        hidden = mlp_hidden(width)
        configs.append(row(f"w{width}", width, hidden, step_groups(width, hidden, layers), True))
        if (width, layers) == (preset.width, preset.layers):
            configs.append(row(f"w{width}_120m", width, preset.hidden,
                               step_groups(width, preset.hidden, layers), False))
            configs.append(row(f"w{width}_120m_harness", width, preset.hidden,
                               harness_groups("120m"), False))
    return configs


def _timed(call, flush: torch.Tensor, iters: int) -> list[tuple]:
    """Event pairs around ``iters`` calls. Read them only after a synchronize."""
    pairs = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        flush.zero_()
        start.record()
        call()
        end.record()
        pairs.append((start, end))
    return pairs


def _verify(name: str, fn, groups: list, replayed: list) -> list[dict]:
    """Compare each replayed group output with a fresh eager output. Raise on any difference.

    The comparison is on the raw bits, so no difference tensor is needed: finite outputs with
    equal bits have a maximum absolute difference of exactly zero.
    """
    rows = []
    for G, graph_out in zip(groups, replayed):
        eager = fn(G)
        if not (torch.isfinite(graph_out).all().item() and torch.isfinite(eager).all().item()):
            raise RuntimeError(f"Method {name!r} returned non-finite values for group {tuple(G.shape)}")
        bits = _INT_VIEW[eager.element_size()]
        if not torch.equal(graph_out.view(bits), eager.view(bits)):
            max_abs = (graph_out - eager).abs().max().item()
            raise RuntimeError(f"Method {name!r} graph replay is not bitwise equal to eager for group "
                               f"{tuple(G.shape)} (max abs difference {max_abs})")
        rows.append({"shape": list(G.shape), "output_dtype": str(eager.dtype).removeprefix("torch."),
                     "finite": True, "graph_vs_eager_max_abs": 0.0})
        del eager
        torch.cuda.empty_cache()  # the next group needs blocks of other sizes
    return rows


def _graph_slot(name: str, step, fn, groups: list, flush: torch.Tensor, warmup: int, iters: int):
    """Capture the step, time ``iters`` replays, then verify the replayed outputs against eager.

    The graph is released before the eager check. Its memory pool and the eager workspace do not
    fit together at the widest rows.
    """
    graph, static = capture(step, groups, warmup)
    graph.replay()  # capture records kernels but does not run them
    torch.cuda.synchronize()
    pairs = _timed(graph.replay, flush, iters)
    torch.cuda.synchronize()
    elapsed = [start.elapsed_time(end) for start, end in pairs]
    replayed = [out.clone() for out in static]
    del graph, static
    torch.cuda.empty_cache()
    return elapsed, _verify(name, fn, groups, replayed)


def _require_unchanged(groups: list, references: list, context: str) -> None:
    if not all(torch.equal(G.cpu(), reference) for G, reference in zip(groups, references)):
        raise RuntimeError(f"{context} mutated its input in-place")


def bench_step(methods: dict, config: dict, *, flush: torch.Tensor, warmups: int, blocks: int,
               iters: int, seed: int) -> dict:
    """Time the full-model step of ``config`` for every method (name -> ``f(G) -> U``)."""
    torch.manual_seed(seed)
    groups = [torch.randn(*shape, device="cuda") for shape in config["groups"]]
    references = [G.cpu() for G in groups]
    names = list(methods)
    steps = {name: (lambda Gs, fn=fn: tuple(fn(G) for G in Gs)) for name, fn in methods.items()}
    load_start = os.getloadavg()

    checks = {}
    for name in names:  # warm autotune, compile and plan caches; fail before any timing
        for _ in range(warmups):
            steps[name](groups)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _require_unchanged(groups, references, f"Method {name!r}")
        checks[name] = [_graph_slot(name, steps[name], methods[name], groups, flush, warmups, 0)[1]]
        torch.cuda.empty_cache()

    orders = [block_order(names, block) for block in range(blocks)]
    samples = {name: {clock: [] for clock in CLOCKS} for name in names}

    # Pass 1: CUDA-graph replay (no host launch inside the timed region), one live graph at a time.
    for order in orders:
        for name in order:
            elapsed, check = _graph_slot(name, steps[name], methods[name], groups, flush, 0, iters)
            samples[name]["cuda_graph"] += elapsed
            checks[name].append(check)
            torch.cuda.empty_cache()

    # Pass 2: eager CUDA-event timing.
    pairs = [(name, pair) for order in orders for name in order
             for pair in _timed(lambda: steps[name](groups), flush, iters)]
    torch.cuda.synchronize()
    for name, (start, end) in pairs:
        samples[name]["eager_event"].append(start.elapsed_time(end))
    _require_unchanged(groups, references, "A timed pass")

    results = {}
    for name in names:
        results[name] = {
            "finite": True,
            "graph_vs_eager_max_abs": max(g["graph_vs_eager_max_abs"] for c in checks[name] for g in c),
            "input_unchanged": True,
            "verified_graph_captures": len(checks[name]),
            "groups": checks[name][0],
            **{clock: _stats(samples[name][clock]) for clock in CLOCKS},
        }
    for reference in REFERENCES:
        for row in results.values():
            row[f"speedup_vs_{reference}"] = ({
                clock: results[reference][clock]["median_ms"] / row[clock]["median_ms"] for clock in CLOCKS
            } if reference in results else None)
    del groups, references
    torch.cuda.empty_cache()
    return {**config, "loadavg": {"start": load_start, "end": os.getloadavg()},
            "block_orders": orders, "methods": results}


def crossover(steps: list[dict], method: str, clock: str, reference: str = "ps3_blackwell") -> dict:
    """Speedup of ``method`` over ``reference`` on the width grid, and where it crosses 1.00x.

    A crossing lies between two adjacent grid widths with speedups on opposite sides of 1, or at
    an exact-1.0 plateau flanked by opposite sides. ``width_estimate`` interpolates log(speedup)
    linearly in log(width). It is an estimate between two measured widths, not a measurement.
    """
    raw = [(s["width"], s["methods"][method][f"speedup_vs_{reference}"][clock])
           for s in steps if s["grid"]]
    for w, s in raw:
        if not (math.isfinite(w) and w > 0 and math.isfinite(s) and s > 0):
            raise ValueError(f"Grid width and speedup must be finite and positive, got ({w!r}, {s!r})")
    points = sorted(raw)
    for k in range(len(points) - 1):
        if points[k][0] == points[k + 1][0]:
            raise ValueError(f"Duplicate grid width {points[k][0]}")
    crossings = []
    i = 0
    while i < len(points):
        if points[i][1] == 1.0:
            j = i
            while j + 1 < len(points) and points[j + 1][1] == 1.0:
                j += 1
            if i > 0 and j < len(points) - 1 and (points[i - 1][1] - 1.0) * (points[j + 1][1] - 1.0) < 0.0:
                w0, w1 = points[i][0], points[j][0]
                crossings.append({"between_widths": [w0, w1],
                                  "width_estimate": float(w0) if w0 == w1 else math.sqrt(w0 * w1)})
            i = j + 1
        else:
            if i + 1 < len(points):
                (w0, s0), (w1, s1) = points[i], points[i + 1]
                if (s0 - 1.0) * (s1 - 1.0) < 0.0:
                    fraction = math.log(s0) / (math.log(s0) - math.log(s1))
                    crossings.append({"between_widths": [w0, w1], "width_estimate": w0 * (w1 / w0) ** fraction})
            i += 1
    return {"speedup_by_width": {str(w): s for w, s in points},
            "first_width_at_or_above_1x": next((w for w, s in points if s >= 1.0), None),
            "crossings": crossings}


def run(names: list[str] | None = None, *, widths=WIDTHS, layers: int = LAYERS, warmups: int = 5,
        blocks: int | None = None, iters: int = 5, seed: int = 0) -> dict:
    if warmups < 5:
        raise ValueError("Use at least 5 warmup calls before capture")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    factories = method_factories()
    names = list(names or factories)
    methods = {name: factories[name]() for name in names}
    blocks = blocks or default_blocks(len(names))
    properties = torch.cuda.get_device_properties(0)
    flush = torch.empty(4 * properties.L2_cache_size, dtype=torch.uint8, device="cuda")
    device = torch.device("cuda")
    result = {
        "env": {**git_info(), "gpu": properties.name, "l2_cache_bytes": properties.L2_cache_size,
                "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
                "host_cpus": os.cpu_count(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "loadavg": {"start": os.getloadavg()}, "gpu_telemetry": {"start": gpu_telemetry(device)}},
        "settings": {
            "warmups": warmups, "blocks": blocks, "iters": iters, "seed": seed, "layers": layers,
            "widths": list(widths),
            "work_dtype": {name: "bfloat16" if name.startswith("pe_") else "float16" for name in names},
            "output": "native dtype and strides of each method (no FP32 layout cast)",
            "sample": "one full-model step: every shape group of the row, in order",
            "input": "torch.randn float32", "flush": "flush.zero_() on 4x L2 before every sample",
            "order": "Williams balanced Latin square, one row per block",
            "clocks": "isolated passes; cuda_graph recaptures per block slot, one live graph at a time",
            "allow_fp16_reduced_precision_reduction": False,
            "allow_bf16_reduced_precision_reduction": False,
        },
        "steps": [],
    }
    for config in step_configs(widths, layers):
        row = bench_step(methods, config, flush=flush, warmups=warmups, blocks=blocks, iters=iters, seed=seed)
        row["gpu_telemetry"] = gpu_telemetry(device)
        result["steps"].append(row)
        print(f"\n{row['label']}  groups {row['groups']}  "
              f"load {row['loadavg']['start'][0]:.2f} -> {row['loadavg']['end'][0]:.2f}")
        for name, m in row["methods"].items():
            ratio = m["speedup_vs_ps3_blackwell"]
            versus = f"  vs ps3_blackwell {ratio['cuda_graph']:.3f}x" if ratio else ""
            print(f"  {name:<14} graph {m['cuda_graph']['median_ms']:9.3f} ms  "
                  f"event {m['eager_event']['median_ms']:9.3f} ms{versus}", flush=True)
    if "ps3_blackwell" in names:
        result["crossover_vs_ps3_blackwell"] = {
            method: {clock: crossover(result["steps"], method, clock) for clock in CLOCKS}
            for method in CROSSOVER_METHODS if method in names}
    result["env"]["loadavg"]["end"] = os.getloadavg()
    result["env"]["gpu_telemetry"]["end"] = gpu_telemetry(device)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", choices=list(method_factories()))
    parser.add_argument("--widths", nargs="+", type=int, default=WIDTHS)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--blocks", type=int, help="default: method count, doubled when odd (full balance)")
    parser.add_argument("--iters", type=int, default=5, help="samples per method per block")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help=f"fast check that overrides widths, layers, blocks and iters: {QUICK}")
    parser.add_argument("--output", type=Path)
    return parser


def _is_canonical_args(args: argparse.Namespace) -> bool:
    all_methods = list(method_factories())
    return (
        not args.quick
        and (args.methods is None or list(args.methods) == all_methods)
        and list(args.widths) == list(WIDTHS)
        and (args.blocks is None or args.blocks == default_blocks(len(all_methods)))
        and args.iters == 5
        and args.warmups >= 5
    )


def _resolve_output(args: argparse.Namespace) -> Path:
    canonical = _is_canonical_args(args)
    if args.output is None:
        if args.quick:
            return QUICK_OUTPUT
        if canonical:
            return CANONICAL_OUTPUT
        raise ValueError(
            "Non-canonical subset run requires an explicit --output path to avoid "
            f"overwriting {CANONICAL_OUTPUT.name}"
        )
    if args.output.resolve() == CANONICAL_OUTPUT.resolve() and not canonical:
        raise ValueError(
            f"Refusing to overwrite canonical {CANONICAL_OUTPUT.name} from a --quick or "
            "subset run; choose a different --output path"
        )
    return args.output


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = _resolve_output(args)
    except ValueError as exc:
        parser.error(str(exc))
    shape = QUICK if args.quick else {"widths": args.widths, "blocks": args.blocks, "iters": args.iters}
    result = run(args.methods, warmups=args.warmups, seed=args.seed, **shape)
    result["settings"]["quick"] = args.quick
    result["command"] = [sys.executable, *sys.argv]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
