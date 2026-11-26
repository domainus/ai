"""Cycle 3 C3-03: CUDA-graph benchmark and Tier-1 refresh after the small-n launch fusion.

Part ``graph`` repeats the Cycle 2 matrix of ``bench_cuda_graph.py`` (5 LLM shapes, 11 methods,
three clocks) and adds the change of every cell against ``results/cycle2_cuda_graph_bench.json``.

Part ``tier1`` runs the 117 Tier-1 cases (72 synthetic, 45 harvested momentum matrices) for the
four Schatten Switch circuits and four references. Each case reports the error against the
FP64 SVD polar factor and the same three clocks. This replaces the latency metadata in
``results/schatten_switch_search_cycle2.json``, which was measured before the sync-free path.
That file is not changed.

Both parts use ``bench_cuda_graph``: Williams-balanced blocks, a 4x L2 flush before every
sample, at least 5 warm-up calls, and CUDA-graph replay checked against the eager result.
Both outputs also hold the GPU kernels per call of every circuit, counted with
``torch.profiler``, for this tree and for the Cycle 2 code (``--baseline-commit``).

A canonical output is written only by the full default run from a clean git tree. ``--quick``
and subset runs write other files.

Run (alone on the GPU):
    PYTHONPATH=. .venv/bin/python research/polar_opt/bench_c3_03_tier1_refresh.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CANONICAL = {"graph": RESULTS / "cycle3_c3_03_cuda_graph_bench.json",
             "tier1": RESULTS / "cycle3_c3_03_tier1_refresh.json"}
QUICK = {part: path.with_name(path.stem + "_quick.json") for part, path in CANONICAL.items()}
CYCLE2_GRAPH = RESULTS / "cycle2_cuda_graph_bench.json"
BASELINE_COMMIT = "4b7a792"  # Cycle 2 closure: the code before the launch fusion
TRAJECTORIES = ROOT / "artifacts" / "harvested_trajectories.pt"

SHAPES = [(512, 2048), (1024, 4096), (2048, 8192), (4096, 4096), (4, 1024, 4096)]
# Method order of results/cycle2_cuda_graph_bench.json.
GRAPH_METHODS = ["muon_pe_eager", "muon_pe_compiled", "muon_pe_triton", "muon_ps2_eager", "muon_ps3_eager",
                 "muon_ps2_blackwell", "muon_ps3_blackwell", "muon_ss3_54", "muon_ss3_53", "muon_ss3_35",
                 "muon_ss3_34"]
SS3 = ["ss3_54", "ss3_53", "ss3_35", "ss3_34"]
GRAPH_REFERENCES = ["muon_ps3_blackwell", "muon_ps2_blackwell", "muon_ps3_eager", "muon_pe_triton",
                    "muon_pe_compiled", "muon_pe_eager"]
TIER1_METHODS = [*(f"{name}_blackwell_fp16" for name in SS3), "ps3_55_blackwell_fp16", "ps2_55_blackwell_fp16",
                 "pe_triton_bf16", "pe_eager"]
TIER1_REFERENCE = "ps3_55_blackwell_fp16"
LAUNCH_SHAPES = [(512, 2048), (1024, 4096), (2048, 8192)]
CLOCKS = ("cuda_graph_ms", "eager_event_ms", "wall_ms")
DEFAULTS = {"warmups": 5, "iters": 5, "seed": 0, "tier1_seed": 123, "divergence_limit": 2.0}
QUICK_SETTINGS = {"shapes": [(64, 256), (2, 33, 97)], "blocks": 2, "iters": 2, "max_cases": 3,
                  "launch_shapes": [(64, 256)]}


def tier1_callables() -> dict:
    """Name -> orthogonalizer ``f(G)`` of the Tier-1 refresh."""
    from research.polar_opt import tier2_candidates as candidates
    from research.polar_opt.baselines import polar_express

    return {**{f"{name}_blackwell_fp16": getattr(candidates, f"{name}_blackwell_fp16") for name in SS3},
            "ps3_55_blackwell_fp16": candidates.ps3_blackwell_fp16,
            "ps2_55_blackwell_fp16": candidates.ps2_blackwell_fp16,
            "pe_triton_bf16": candidates.pe_triton_bf16, "pe_eager": polar_express}


def graph_factories() -> dict:
    from research.polar_opt import tier2_candidates as candidates
    from research.polar_opt.harness_tier2_real import builtin_factories

    factories = builtin_factories()
    for name in SS3:
        factories[f"muon_{name}"] = lambda name=name: getattr(candidates, f"{name}_blackwell_fp16")
    return factories


def default_blocks(count: int) -> int:
    return count if count % 2 == 0 else 2 * count


# --------------------------------------------------------------------------------------
# GPU kernels per call (torch.profiler)
# --------------------------------------------------------------------------------------
def count_launches(shapes) -> dict:
    """GPU kernels and device-to-host copies of one warm call, per circuit and shape.

    The circuits come from the ``research`` package on ``sys.path``. The parent process runs
    this function once with this tree and once with an export of the baseline commit.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    from research.polar_opt import tier2_candidates as candidates

    calls = {name: getattr(candidates, f"{name}_blackwell_fp16") for name in SS3}
    calls["ps3_55"] = candidates.ps3_blackwell_fp16
    out = {}
    for shape in shapes:
        torch.manual_seed(0)
        G = torch.randn(*shape, device="cuda")
        row = out["x".join(map(str, shape))] = {}
        for name, call in calls.items():
            for _ in range(5):
                call(G)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                call(G)
                torch.cuda.synchronize()
            kernels: dict = {}
            copies = 0
            for event in prof.events():
                if event.device_type != torch.autograd.DeviceType.CUDA:
                    continue
                if event.name.lower().startswith("mem"):  # Memcpy / Memset: not a kernel launch
                    copies += "dtoh" in event.name.lower()
                    continue
                label = "cublas_gemm" if "gemm" in event.name else event.name.split("(")[0].split("<")[0]
                kernels[label] = kernels.get(label, 0) + 1
            row[name] = {"total": sum(kernels.values()), "memcpy_dtoh": copies, "kernels": kernels}
    return out


def _subprocess_launches(root: Path, shapes) -> dict:
    env = {**os.environ, "PYTHONPATH": str(root)}
    command = [sys.executable, str(Path(__file__).resolve()), "--count-launches", json.dumps(shapes)]
    done = subprocess.run(command, env=env, cwd=root, capture_output=True, text=True, check=False)
    if done.returncode:
        raise RuntimeError(f"Launch count failed in {root}:\n{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def launch_table(shapes, baseline_commit: str) -> dict:
    """Kernels per call before (baseline commit) and after (this tree), from the profiler."""
    def git(*args, text=True):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=text, check=True).stdout

    after = _subprocess_launches(ROOT, shapes)
    with tempfile.TemporaryDirectory(prefix="c3-03-baseline.") as scratch:
        archive = git("archive", "--format=tar", baseline_commit, "research", text=False)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(scratch, filter="data")
        before = _subprocess_launches(Path(scratch), shapes)
    table = {"method": "torch.profiler CUDA kernel events of one warm call; Memcpy and Memset are not kernels",
             "baseline_commit": git("rev-parse", baseline_commit).strip(), "shapes": {}}
    for shape, rows in after.items():
        table["shapes"][shape] = {
            name: {"before": before[shape][name]["total"], "after": row["total"],
                   "saved": before[shape][name]["total"] - row["total"],
                   "memcpy_dtoh_before": before[shape][name]["memcpy_dtoh"], "memcpy_dtoh_after": row["memcpy_dtoh"],
                   "kernels_before": before[shape][name]["kernels"], "kernels_after": row["kernels"]}
            for name, row in rows.items()}
    return table


# --------------------------------------------------------------------------------------
# Part 1: the Cycle 2 CUDA-graph matrix
# --------------------------------------------------------------------------------------
def delta_vs_cycle2(result: dict, cycle2: dict, reference: str = "muon_ps3_blackwell") -> list[dict]:
    """Before (Cycle 2 JSON) and after (this run) medians of every cell that both runs have."""
    before_shapes = {tuple(row["shape"]): row["methods"] for row in cycle2["shapes"]}
    rows = []
    for shape_row in result["shapes"]:
        before = before_shapes.get(tuple(shape_row["shape"]))
        if before is None:
            continue
        after = shape_row["methods"]
        cells = {}
        for name in after:
            if name not in before:
                continue
            cell = {}
            for clock in CLOCKS:
                if not (after[name].get(clock) and before[name].get(clock)):
                    continue
                b, a = before[name][clock]["median_ms"], after[name][clock]["median_ms"]
                cell[clock] = {"before": b, "after": a, "delta_us": (a - b) * 1e3, "speedup": b / a}
                if reference in after and reference in before and after[reference].get(clock):
                    cell[clock]["vs_reference_before"] = before[reference][clock]["median_ms"] / b
                    cell[clock]["vs_reference_after"] = after[reference][clock]["median_ms"] / a
            cells[name] = cell
        rows.append({"shape": shape_row["shape"], "reference": reference, "methods": cells})
    return rows


def ss3_speedups(result: dict) -> list[dict]:
    """Reference median / circuit median, per shape, clock and reference of the same run."""
    rows = []
    for shape_row in result["shapes"]:
        methods = shape_row["methods"]
        row = {"shape": shape_row["shape"]}
        for clock in CLOCKS:
            medians = {name: cell[clock]["median_ms"] for name, cell in methods.items() if cell.get(clock)}
            row[clock] = {
                f"muon_{name}": {reference: medians[reference] / medians[f"muon_{name}"]
                                 for reference in GRAPH_REFERENCES if reference in medians}
                for name in SS3 if f"muon_{name}" in medians}
        rows.append(row)
    return rows


def run_graph(*, shapes, warmups: int, blocks: int | None, iters: int, seed: int) -> dict:
    from research.polar_opt import bench_cuda_graph as cg

    result = cg.run(GRAPH_METHODS, graph_factories(), shapes=shapes, warmup=warmups, blocks=blocks,
                    iters=iters, seed=seed)
    result["settings"].update(allow_fp16_reduced_precision_reduction=False,
                              allow_bf16_reduced_precision_reduction=False)
    result["ss3_speedups"] = ss3_speedups(result)
    if CYCLE2_GRAPH.is_file():
        cycle2 = json.loads(CYCLE2_GRAPH.read_text())
        result["delta_vs_cycle2"] = {
            "source": CYCLE2_GRAPH.name, "source_git_sha": cycle2["env"].get("git_sha"),
            "note": "vs_reference is muon_ps3_blackwell median / method median of the same run",
            "shapes": delta_vs_cycle2(result, cycle2)}
    checks = [m for row in result["shapes"] for m in row["methods"].values()]
    result["checks"] = {
        "capture_errors": sum("capture_error" in m for m in checks),
        "nonfinite": sum(not m["finite"] for m in checks),
        "graph_vs_eager_max_abs": max((m.get("graph_vs_eager_max_abs") or 0.0) for m in checks)}
    return result


# --------------------------------------------------------------------------------------
# Part 2: 117 Tier-1 cases
# --------------------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tier1_cases(trajectories: Path, seed: int) -> list:
    """The 117 cases of ``schatten_switch_search.tier1``: 72 synthetic and 45 harvested."""
    from research.polar_opt.harness_tier1 import harvested_cases, synthetic_cases

    return [*synthetic_cases([128, 256], [1, 2, 4, 8], seed), *harvested_cases(trajectories, all_layers=True)]


def summarize_tier1(cases: list[dict], reference: str = TIER1_REFERENCE) -> dict:
    """Aggregate error, largest singular value, divergence and the three clocks per method."""
    def aggregate(rows: list[dict]) -> dict:
        out = {}
        for name in rows[0]["methods"]:
            cells = [row["methods"][name] for row in rows]
            errors = [c["relative_frobenius_error"] for c in cells if "relative_frobenius_error" in c]
            entry = {
                "cases": len(cells), "diverged": sum(bool(c["diverged"]) for c in cells),
                "nonfinite": sum(bool(c["nonfinite"]) for c in cells),
                "capture_errors": sum("capture_error" in c for c in cells),
                "graph_vs_eager_max_abs": max((c.get("graph_vs_eager_max_abs") or 0.0) for c in cells),
                "median_relative_frobenius_error": median(errors) if errors else None,
                "mean_relative_frobenius_error": mean(errors) if errors else None,
                "max_relative_frobenius_error": max(errors) if errors else None,
                "max_sigma_max": max((c["sigma_max"] for c in cells if "sigma_max" in c), default=None)}
            for clock in CLOCKS:
                values = [c[clock]["median_ms"] for c in cells if c.get(clock)]
                entry[f"median_case_{clock}"] = median(values) if values else None
                ratios = [row["methods"][reference][clock]["median_ms"] / row["methods"][name][clock]["median_ms"]
                          for row in rows if row["methods"][name].get(clock) and row["methods"][reference].get(clock)]
                entry[f"median_speedup_vs_{reference}_{clock}"] = median(ratios) if ratios else None
            out[name] = entry
        return out

    suites = {"all": cases}
    for case in cases:
        suites.setdefault(case["suite"], []).append(case)
    return {suite: aggregate(rows) for suite, rows in suites.items()}


def run_tier1(trajectories: Path, *, warmups: int, blocks: int | None, iters: int, seed: int,
              divergence_limit: float, max_cases: int | None = None) -> dict:
    import torch

    from research.polar_opt import bench_cuda_graph as cg
    from research.polar_opt.baselines import svd_polar_fp64
    from research.polar_opt.harness_tier1 import _quality
    from research.polar_opt.harness_tier2_real import git_info, gpu_telemetry

    if warmups < 5:
        raise ValueError("Use at least 5 warmup calls before capture")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    methods = tier1_callables()
    blocks = blocks or default_blocks(len(methods))
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    flush = torch.empty(4 * properties.L2_cache_size, dtype=torch.uint8, device="cuda")
    cases = tier1_cases(trajectories, seed)
    counts = {"total": len(cases), "synthetic": sum(c.suite != "harvested_momentum" for c in cases),
              "harvested": sum(c.suite == "harvested_momentum" for c in cases)}
    if max_cases is not None:
        cases = cases[:max_cases // 2] + cases[len(cases) - (max_cases - max_cases // 2):]
    result = {
        "env": {**git_info(), "gpu": properties.name, "l2_cache_bytes": properties.L2_cache_size,
                "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
                "host_cpus": os.cpu_count(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "loadavg": {"start": os.getloadavg()}, "gpu_telemetry": {"start": gpu_telemetry(device)}},
        "settings": {
            "warmups": warmups, "blocks": blocks, "iters_per_block": iters, "seed": seed,
            "samples_per_clock": blocks * iters, "divergence_limit": divergence_limit,
            "flush": "flush.zero_() on 4x L2 before every sample",
            "order": "Williams balanced Latin square, one row per block, the same for every case",
            "block_orders": [cg.block_order(list(methods), block) for block in range(blocks)],
            "clocks": {"cuda_graph_ms": "CUDA events around graph.replay() of a warm capture",
                       "eager_event_ms": "CUDA events around an eager call",
                       "wall_ms": "host perf_counter between two synchronizes around an eager call"},
            "error": "||Q - P||_F / sqrt(n) against baselines.svd_polar_fp64 (FP64 SVD polar factor)",
            "allow_fp16_reduced_precision_reduction": False, "allow_bf16_reduced_precision_reduction": False,
            "trajectories": str(trajectories), "trajectories_sha256": _sha256(trajectories),
            "cases_available": counts, "cases_run": len(cases), "methods": list(methods),
            "supersedes": "latency fields of results/schatten_switch_search_cycle2.json (tier1 and "
                          "latency_sanity), measured before the sync-free path; that file is unchanged"},
        "cases": [],
    }
    for index, case in enumerate(cases, 1):
        G = case.matrix.to(device)
        reference = svd_polar_fp64(G)
        cpu_reference = reference.cpu()
        quality = {name: _quality(call(G), reference, cpu_reference, divergence_limit)
                   for name, call in methods.items()}
        timing = cg.bench_matrix(methods, G, flush=flush, warmup=warmups, blocks=blocks, iters=iters)
        row = {"suite": case.suite, "name": case.name, "shape": list(G.shape), "metadata": case.metadata,
               "loadavg_start": timing["loadavg_start"], "loadavg_end": timing["loadavg_end"], "methods": {}}
        for name in methods:
            cell = {**quality[name], **timing["methods"][name]}
            for clock in CLOCKS:  # 117 cases x 8 methods: keep the statistics, drop the raw samples
                if cell.get(clock):
                    cell[clock] = {k: v for k, v in cell[clock].items() if k != "samples_ms"}
            row["methods"][name] = cell
        result["cases"].append(row)
        if index % 8 == 0 or index == len(cases):
            print(f"tier1 {index}/{len(cases)} cases", flush=True)
        del G, reference
    result["summary"] = summarize_tier1(result["cases"])
    everything = result["summary"]["all"]
    result["checks"] = {
        "cases": len(result["cases"]),
        "diverged": {name: row["diverged"] for name, row in everything.items()},
        "capture_errors": sum(row["capture_errors"] for row in everything.values()),
        "graph_vs_eager_max_abs": max(row["graph_vs_eager_max_abs"] for row in everything.values())}
    result["env"]["loadavg"]["end"] = os.getloadavg()
    result["env"]["gpu_telemetry"]["end"] = gpu_telemetry(device)
    for name, row in everything.items():
        print(f"  {name:<24} err median {row['median_relative_frobenius_error']:.4f} "
              f"max {row['max_relative_frobenius_error']:.4f}  sigma_max {row['max_sigma_max']:.4f}  "
              f"diverged {row['diverged']}/{row['cases']}  graph {row['median_case_cuda_graph_ms']:.4f} ms  "
              f"event {row['median_case_eager_event_ms']:.4f} ms  wall {row['median_case_wall_ms']:.4f} ms")
    return result


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=["all", "graph", "tier1"], default="all")
    parser.add_argument("--warmups", type=int, default=DEFAULTS["warmups"])
    parser.add_argument("--blocks", type=int, help="default: method count, doubled when odd (full balance)")
    parser.add_argument("--iters", type=int, default=DEFAULTS["iters"], help="samples per method per block")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="seed of the graph benchmark inputs")
    parser.add_argument("--max-cases", type=int, help="Tier-1 subset: first and last cases only")
    parser.add_argument("--trajectories", type=Path, default=TRAJECTORIES)
    parser.add_argument("--baseline-commit", default=BASELINE_COMMIT,
                        help="commit of the code before the fusion, for the kernel counts")
    parser.add_argument("--quick", action="store_true", help=f"fast check: {QUICK_SETTINGS}")
    parser.add_argument("--graph-output", type=Path)
    parser.add_argument("--tier1-output", type=Path)
    parser.add_argument("--count-launches", metavar="SHAPES_JSON", help=argparse.SUPPRESS)
    return parser


def _is_canonical_args(args: argparse.Namespace) -> bool:
    """True for the full default run. Only this run may write a canonical output."""
    return (not args.quick and args.blocks is None and args.max_cases is None
            and args.iters == DEFAULTS["iters"] and args.warmups >= DEFAULTS["warmups"]
            and args.seed == DEFAULTS["seed"] and args.baseline_commit == BASELINE_COMMIT)


def _resolve_outputs(args: argparse.Namespace) -> dict[str, Path]:
    """Output path per requested part. A canonical file is safe from quick and subset runs."""
    canonical = _is_canonical_args(args)
    outputs = {}
    for part in ("graph", "tier1") if args.part == "all" else (args.part,):
        given = getattr(args, f"{part}_output")
        if given is None:
            if args.quick:
                given = QUICK[part]
            elif canonical:
                given = CANONICAL[part]
            else:
                raise ValueError(f"A subset run needs an explicit --{part}-output path, so that it "
                                 f"cannot overwrite {CANONICAL[part].name}")
        elif given.resolve() in {path.resolve() for path in CANONICAL.values()} and not (
                canonical and given.resolve() == CANONICAL[part].resolve()):
            raise ValueError(f"Refusing to overwrite a canonical output with {given.name} from a --quick, "
                             f"subset or other-part run; choose a different --{part}-output path")
        outputs[part] = given
    return outputs


def _require_clean_tree(outputs: dict[str, Path]) -> None:
    from research.polar_opt.harness_tier2_real import git_info

    canonical = [path.name for part, path in outputs.items() if path.resolve() == CANONICAL[part].resolve()]
    if canonical and git_info()["git_dirty"] is not False:
        raise ValueError(f"{', '.join(canonical)} must record git_dirty: false. Commit the code first.")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"\nWrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.count_launches:
        print(json.dumps(count_launches([tuple(shape) for shape in json.loads(args.count_launches)])))
        return
    try:
        outputs = _resolve_outputs(args)
        _require_clean_tree(outputs)
    except ValueError as error:
        parser.error(str(error))
    if "tier1" in outputs and not args.trajectories.is_file():
        parser.error(f"Missing {args.trajectories}; pass --trajectories")
    quick = QUICK_SETTINGS if args.quick else {}
    blocks, iters = quick.get("blocks", args.blocks), quick.get("iters", args.iters)
    command = [sys.executable, *(sys.argv if argv is None else [sys.argv[0], *argv])]
    payloads = {}
    if "graph" in outputs:
        payloads["graph"] = run_graph(shapes=quick.get("shapes", SHAPES), warmups=args.warmups, blocks=blocks,
                                      iters=iters, seed=args.seed)
    if "tier1" in outputs:
        payloads["tier1"] = run_tier1(args.trajectories, warmups=args.warmups, blocks=blocks, iters=iters,
                                      seed=DEFAULTS["tier1_seed"], divergence_limit=DEFAULTS["divergence_limit"],
                                      max_cases=quick.get("max_cases", args.max_cases))
    launches = launch_table(quick.get("launch_shapes", LAUNCH_SHAPES), args.baseline_commit)
    for part, payload in payloads.items():
        payload["settings"]["quick"] = args.quick
        payload["launch_counts"] = launches
        payload["command"] = command
        _write(outputs[part], payload)


if __name__ == "__main__":
    main()
