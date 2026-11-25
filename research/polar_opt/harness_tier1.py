"""Check polar accuracy and cold-L2 wall-clock latency on four matrix suites."""

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from math import sqrt
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from research.polar_opt.baselines import (
    TheoreticalCost,
    gram_ns,
    jordan_ns5,
    polar_express,
    polar_switch,
    svd_polar_fp64,
    theoretical_cost,
)
from research.polar_opt.harness_tier2_lm import configure_device, load_callable


Apply = Callable[[Tensor, Tensor | None], Tensor]
Cost = Callable[[Tensor], TheoreticalCost]


@dataclass(frozen=True)
class Orthogonalizer:
    apply: Apply
    cost: Cost | None


class Registry(dict[str, Orthogonalizer]):
    """Register a pure function of the current matrix and optional previous matrix."""

    def register(self, name, function, *, cost=None, uses_previous=False):
        if not name or name in self:
            raise ValueError(f"Duplicate or empty orthogonalizer name {name!r}")
        apply = function if uses_previous else lambda matrix, previous: function(matrix)
        self[name] = Orthogonalizer(apply, cost)


def baseline_registry() -> Registry:
    registry = Registry()
    for name, function in {
        "svd_polar_fp64": svd_polar_fp64,
        "jordan_ns5": jordan_ns5,
        "polar_express": polar_express,
        "gram_ns": gram_ns,
        "PS2_55": polar_switch,
        "PS3_55": partial(polar_switch, variant="PS3_55"),
    }.items():
        def cost(matrix, method=name):
            return theoretical_cost(method, tuple(matrix.shape), input_dtype=matrix.dtype,
                                    device=matrix.device)

        registry.register(name, function, cost=cost)
    return registry


@dataclass(frozen=True)
class Case:
    suite: str
    name: str
    matrix: Tensor
    previous: Tensor | None = None
    metadata: dict | None = None


def _bases(n, m, rng):
    u = torch.linalg.qr(torch.randn(n, n, generator=rng, dtype=torch.float64)).Q
    v = torch.linalg.qr(torch.randn(m, n, generator=rng, dtype=torch.float64)).Q
    return u, v


def synthetic_cases(sizes, aspects, seed):
    rng = torch.Generator().manual_seed(seed)
    for n in sizes:
        for aspect in aspects:
            m = n * aspect
            shape = f"{n}x{m}"
            u, v = _bases(n, m, rng)

            def with_spectrum(s):
                return ((u * s) @ v.mT).float()

            yield Case("uniform_mp", f"gaussian_mp/{shape}",
                       torch.randn(n, m, generator=rng) / sqrt(m))
            spectrum = 0.1 + 0.9 * torch.rand(n, generator=rng, dtype=torch.float64)
            yield Case("uniform_mp", f"uniform_singular_values/{shape}", with_spectrum(spectrum))
            for gamma in (0.5, 1.0, 1.5):
                spectrum = torch.arange(1, n + 1, dtype=torch.float64).pow(-gamma)
                yield Case("power_law", f"gamma_{gamma}/{shape}", with_spectrum(spectrum),
                           metadata={"gamma": gamma})
            for kappa in (10_000, 1_000_000):
                spectrum = torch.logspace(0, -np.log10(kappa), n, dtype=torch.float64)
                yield Case("adversarial", f"geometric_kappa_{kappa}/{shape}", with_spectrum(spectrum),
                           metadata={"designed_kappa": kappa})

            # A new leading direction perpendicular to the previous leading direction.
            spectrum = torch.full((n,), 1e-6, dtype=torch.float64)
            spectrum[0] = 1
            old_spectrum = spectrum.roll(1)
            yield Case("adversarial", f"orthogonal_spike/{shape}", with_spectrum(spectrum),
                       with_spectrum(old_spectrum),
                       {"designed_kappa": 1_000_000, "designed_top_vector_cosine": 0})
            # Integer outer products preserve exact rank one after FP32 storage.
            sign_u = (torch.randint(2, (n,), generator=rng) * 2 - 1).float()
            sign_v = (torch.randint(2, (m,), generator=rng) * 2 - 1).float()
            yield Case("adversarial", f"rank_one/{shape}", sign_u[:, None] * sign_v[None, :],
                       metadata={"designed_rank": 1})


def harvested_cases(path, all_layers=False):
    data = torch.load(path, map_location="cpu", weights_only=True)
    steps_by_phase = data["metadata"]["snapshot_steps"]
    layers = data["metadata"]["selected_layers"]
    layer_by_phase = dict(zip(steps_by_phase, (layers[0], layers[len(layers) // 2], layers[-1])))
    for pair in data["pairs"]:
        phase, parameter = pair["phase"], pair["parameter"]
        if all_layers or int(parameter.split(".")[1]) == layer_by_phase[phase]:
            yield Case("harvested_momentum", f"{phase}/{parameter}", pair["current"], pair["previous"],
                       {"phase": phase, "parameter": parameter, "steps": pair["steps"]})


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _quality(output, reference, cpu_reference, divergence_limit):
    if output.shape != reference.shape:
        raise ValueError(f"Expected output shape {tuple(reference.shape)}, received {tuple(output.shape)}")
    value = output.detach().to(device="cpu", dtype=torch.float64)
    nan = torch.isnan(value).any().item()
    inf = torch.isinf(value).any().item()
    if nan or inf:
        return {"nan": nan, "inf": inf, "diverged": True, "nonfinite": True}

    # Reduced SVD returns a particular orthogonal completion for rank-deficient inputs.
    sigma = torch.linalg.svdvals(value)
    lo, hi = sigma[-1].item(), sigma[0].item()
    return {
        "nan": False, "inf": False, "nonfinite": False, "diverged": hi > divergence_limit,
        "relative_frobenius_error": (value - cpu_reference).norm().item() / sqrt(len(sigma)),
        "sigma_min": lo, "sigma_max": hi, "worst_singular_deviation": max(1 - lo, hi - 1),
    }


def _latency_stats(samples):
    q1, p50, q3 = np.quantile(samples, [0.25, 0.5, 0.75])
    return {
        "median_ms": float(p50), "p25_ms": float(q1), "p75_ms": float(q3),
        "iqr_ms": float(q3 - q1), "samples_ms": samples,
    }


@torch.inference_mode()
def benchmark(cases, registry, device, *, warmup, repeats, seed, divergence_limit):
    """Interleave method order and exclude cache eviction from the timed interval."""
    l2 = torch.cuda.get_device_properties(device).L2_cache_size if device.type == "cuda" else 0
    eviction = torch.zeros(l2 * 2, dtype=torch.uint8, device=device) if l2 else None
    ordering = Random(seed)
    results = []
    for index, case in enumerate(cases, 1):
        matrix = case.matrix.to(device)
        previous = case.previous.to(device) if case.previous is not None else None
        reference = svd_polar_fp64(matrix)
        cpu_reference = reference.cpu()
        measured = {}
        calls = {}
        for name, method in registry.items():
            row = measured[name] = {"cost": asdict(method.cost(matrix)) if method.cost else None}
            try:
                candidate = reference if name == "svd_polar_fp64" else method.apply(matrix, previous)
                row.update(_quality(candidate, reference, cpu_reference, divergence_limit))
                calls[name] = method.apply
            except (RuntimeError, ValueError) as error:
                if isinstance(error, torch.OutOfMemoryError) or "CUDA error:" in str(error):
                    raise
                row.update(nan=None, inf=None, nonfinite=None, diverged=True, error=str(error))

        names = list(calls)
        samples = {name: [] for name in names}
        for i in range(repeats + warmup):
            ordering.shuffle(names)
            for name in names:
                if i >= warmup and eviction is not None:
                    eviction.add_(1)
                _synchronize(device)
                start = perf_counter()
                output = calls[name](matrix, previous)
                _synchronize(device)
                elapsed_ms = (perf_counter() - start) * 1_000
                if i >= warmup:
                    samples[name].append(elapsed_ms)
                del output
        for name, sample in samples.items():
            measured[name]["latency"] = _latency_stats(sample)

        results.append({
            "suite": case.suite, "name": case.name, "shape": list(matrix.shape),
            "input_dtype": str(matrix.dtype), "has_previous": previous is not None,
            "metadata": case.metadata, "methods": measured,
        })
        if index % 8 == 0 or index == len(cases):
            print(f"{index}/{len(cases)} cases complete", flush=True)
    return results


def summarize(results):
    grouped = defaultdict(list)
    for case in results:
        for name, metrics in case["methods"].items():
            grouped["all", name].append(metrics)
            grouped[case["suite"], name].append(metrics)
    summary = defaultdict(dict)
    for (suite, name), rows in grouped.items():
        errors = [row["relative_frobenius_error"] for row in rows if "relative_frobenius_error" in row]
        latency = [row["latency"] for row in rows if "latency" in row]
        summary[suite][name] = {
            "cases": len(rows), "numeric_results": len(errors), "timed_results": len(latency),
            "nan_rate": sum(row["nan"] is True for row in rows) / len(rows),
            "inf_rate": sum(row["inf"] is True for row in rows) / len(rows),
            "nonfinite_rate": sum(row["nonfinite"] is True for row in rows) / len(rows),
            "exceptions": sum("error" in row for row in rows),
            "divergence_rate": sum(row["diverged"] for row in rows) / len(rows),
            "median_relative_frobenius_error": median(errors) if errors else None,
            "max_relative_frobenius_error": max(errors) if errors else None,
            "sigma_min": min((row["sigma_min"] for row in rows if "sigma_min" in row), default=None),
            "sigma_max": max((row["sigma_max"] for row in rows if "sigma_max" in row), default=None),
            "max_singular_deviation": max((row["worst_singular_deviation"] for row in rows
                                           if "worst_singular_deviation" in row), default=None),
            "median_case_latency_ms": median(row["median_ms"] for row in latency) if latency else None,
            "median_case_latency_iqr_ms": median(row["iqr_ms"] for row in latency) if latency else None,
        }
    return dict(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--aspects", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--divergence-limit", type=float, default=2.0)
    parser.add_argument("--trajectories", type=Path, default=Path("artifacts/harvested_trajectories.pt"))
    parser.add_argument("--skip-harvest", action="store_true")
    parser.add_argument("--all-harvest-layers", action="store_true")
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=MODULE:FUNCTION")
    parser.add_argument("--trajectory-candidate", action="append", default=[],
                        metavar="NAME=MODULE:FUNCTION")
    parser.add_argument("--output", type=Path, default=Path("results/cycle0_baselines.json"))
    args = parser.parse_args()
    if min(args.sizes) < 2 or min(args.aspects) < 1 or args.warmup < 0 or args.repeats < 5:
        parser.error("Expected sizes >= 2, aspects >= 1, nonnegative warmup, and at least 5 repeats")
    if not np.isfinite(args.divergence_limit) or args.divergence_limit <= 1:
        parser.error("Divergence limit must be finite and greater than one")
    torch.set_num_threads(args.cpu_threads)
    device = configure_device(args.device)
    registry = baseline_registry()
    for specs, uses_previous in ((args.candidate, False), (args.trajectory_candidate, True)):
        for entry in specs:
            name, spec = entry.split("=", 1)
            registry.register(name, load_callable(spec), uses_previous=uses_previous)

    if not args.skip_harvest and not args.trajectories.is_file():
        parser.error(f"Missing {args.trajectories}. Run harvest_gradients.py first or use --skip-harvest.")
    cases = list(synthetic_cases(args.sizes, args.aspects, args.seed))
    if not args.skip_harvest:
        cases.extend(harvested_cases(args.trajectories, args.all_harvest_layers))
    print(f"Benchmarking {len(cases)} cases with {len(registry)} methods on {device}", flush=True)
    results = benchmark(cases, registry, device, warmup=args.warmup, repeats=args.repeats,
                        seed=args.seed, divergence_limit=args.divergence_limit)
    output = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "settings": {
            "sizes": args.sizes, "aspects": args.aspects, "seed": args.seed, "cpu_threads": args.cpu_threads,
            "warmup": args.warmup, "repeats": args.repeats, "divergence_limit": args.divergence_limit,
            "cache_flush_bytes": (torch.cuda.get_device_properties(device).L2_cache_size * 2
                                  if device.type == "cuda" else 0),
            "timing": "synchronized host perf_counter, randomized interleaving, cache flush excluded",
            "backend": "eager dense PyTorch", "svd_reference": "FP64 reduced SVD, CUDA gesvd / CPU LAPACK",
            "quality_singular_values": "CPU FP64 torch.linalg.svdvals",
            "harvest_checkpoint": None if args.skip_harvest else str(args.trajectories),
            "all_harvest_layers": args.all_harvest_layers, "candidates": args.candidate,
            "trajectory_candidates": args.trajectory_candidate,
        },
        "summary": summarize(results), "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
