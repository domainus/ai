"""Unified solo RTX 5080 head-to-head benchmark for Cycle 1 baselines and discoveries.

Combines:
1. Large LLM-scale shape timing & accuracy (512x2048 .. 4096x4096 and batched 4x1024x4096)
   measuring synchronized GPU-event ms, host wall-clock ms, L2-flushed medians/IQRs,
   and exact FP64 SVD polar error — reporting warm steady-state latency cleanly alongside
   baseline methods.
2. Full Tier-1 4-suite verification via ``harness_tier1`` (uniform/MP, power-law,
   adversarial κ=10^6 / orthogonal spike / rank-1, and all harvested 19.2M LM momentum
   trajectories).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from math import prod

from research.polar_opt.baselines import (
    TheoreticalCost,
    gram_ns,
    jordan_ns5,
    polar_express,
    polar_switch,
    svd_polar_fp64,
    theoretical_cost,
)
from research.polar_opt.circuits import (
    CIRCUITS,
    FP16_ONLY_CIRCUITS,
    SpectralPrescaledPolar,
    WarmSpectralScaler,
    run_circuit,
)
from research.polar_opt.harness_tier1 import (
    Registry,
    baseline_registry,
    benchmark as tier1_benchmark,
    harvested_cases,
    summarize as tier1_summarize,
    synthetic_cases,
)
from research.polar_opt.harness_tier2_lm import configure_device
from research.polar_opt.kernels_blackwell import polar_express_fp8, polar_express_triton
from research.polar_opt.polar_switch_blackwell import gram_ns_blackwell, polar_switch_blackwell

LLM_SHAPES = [(512, 2048), (1024, 4096), (2048, 8192), (4096, 4096), (4, 1024, 4096)]


def _stats(samples: list[float]) -> dict[str, float]:
    q1, med, q3 = np.quantile(samples, [0.25, 0.5, 0.75])
    return {
        "median_ms": float(med),
        "p25_ms": float(q1),
        "p75_ms": float(q3),
        "iqr_ms": float(q3 - q1),
        "min_ms": float(min(samples)),
    }


def _lookup_circuit(name: str):
    if name in CIRCUITS:
        return CIRCUITS[name], True
    if name in FP16_ONLY_CIRCUITS:
        return FP16_ONLY_CIRCUITS[name], False
    raise KeyError(f"Unknown circuit {name!r}")


def _select_circuits(names: list[str] | None) -> list[str]:
    all_circuits = {**CIRCUITS, **FP16_ONLY_CIRCUITS}
    if names:
        return [n for n in names if n in all_circuits]
    if not all_circuits:
        return []
    chosen = []
    for cand in ("S0-k6-lo0.05", "S1-k2+4-lo0.01", "S1-k3+4-lo0.005", "S1-k4+5-lo0.002"):
        if cand in all_circuits:
            chosen.append(cand)
    return chosen or sorted(all_circuits)[:4]


def _clone_scaler(src: WarmSpectralScaler, key: str) -> WarmSpectralScaler:
    """Shallow-clone a single-key WarmSpectralScaler so timed repeats start from identical state."""
    from research.polar_opt.spectral_scaling import WarmSpectralEstimator

    cloned = WarmSpectralScaler(src.config, src.seed)
    st = src.states[key]
    est = WarmSpectralEstimator(st.config, st.seed)
    est.vector = None if st.vector is None else st.vector.clone()
    est.sigma_prev = None if st.sigma_prev is None else st.sigma_prev.clone()
    est.ratio_prev = None if st.ratio_prev is None else st.ratio_prev.clone()
    est._key = st._key
    cloned.states[key] = est
    return cloned


def _make_spectral_wrapper(circuit_name: str, dtype: torch.dtype = torch.float16, use_triton: bool = True):
    circuit, _ = _lookup_circuit(circuit_name)
    last_prev: Tensor | None = None
    last_base: WarmSpectralScaler | None = None

    def apply(matrix: torch.Tensor, previous: torch.Tensor | None) -> torch.Tensor:
        nonlocal last_prev, last_base
        if previous is None:
            scaler = WarmSpectralScaler()
            return run_circuit(matrix, circuit, dtype=dtype, use_triton_syrk=use_triton, scaler=scaler, key="k")
        same_prev = (
            last_prev is not None
            and last_base is not None
            and previous.shape == last_prev.shape
            and previous.dtype == last_prev.dtype
            and previous.device == last_prev.device
            and torch.equal(previous, last_prev)
        )
        if not same_prev:
            base = WarmSpectralScaler()
            run_circuit(previous, circuit, dtype=dtype, use_triton_syrk=use_triton, scaler=base, key="k")
            if previous.is_cuda:
                torch.cuda.synchronize(previous.device)
            last_prev = previous.detach().clone()
            last_base = base
        scaler = _clone_scaler(last_base, "k")
        return run_circuit(matrix, circuit, dtype=dtype, use_triton_syrk=use_triton, scaler=scaler, key="k")

    return apply


def _make_schatten8_wrapper(circuit_name: str, dtype: torch.dtype = torch.float16, use_triton: bool = True):
    circuit, _ = _lookup_circuit(circuit_name)

    def apply(matrix: torch.Tensor) -> torch.Tensor:
        return run_circuit(matrix, circuit, dtype=dtype, use_triton_syrk=use_triton, schatten8=True)

    return apply


@torch.no_grad()
def benchmark_llm_shapes(device: torch.device, circuit_names: list[str], warmup: int, iters: int, seed: int) -> list[dict]:
    """Benchmark baselines, Blackwell kernels, and steady-state spectral circuits on LLM shapes."""
    l2 = torch.cuda.get_device_properties(device).L2_cache_size
    flush = torch.zeros(l2 * 2, dtype=torch.uint8, device=device)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    order = random.Random(seed)
    rows = []

    for shape in LLM_SHAPES:
        G = torch.randn(shape, generator=rng, dtype=torch.float32).to(device)
        ref = svd_polar_fp64(G).reshape(-1, *shape[-2:])

        # Prepare independent steady-state warm scalers per precision (avoid shared mutable state).
        G_prev = (0.95 * G + 0.05 * torch.randn(shape, generator=rng, dtype=torch.float32).to(device))
        warm_scalers_fp16 = {}
        warm_scalers_bf16 = {}
        for cname in circuit_names:
            c, bf16_safe = _lookup_circuit(cname)
            sc16 = WarmSpectralScaler()
            run_circuit(G_prev, c, dtype=torch.float16, use_triton_syrk=True, scaler=sc16, key="steady")
            warm_scalers_fp16[cname] = sc16
            if bf16_safe:
                scb = WarmSpectralScaler()
                run_circuit(G_prev, c, dtype=torch.bfloat16, use_triton_syrk=True, scaler=scb, key="steady")
                warm_scalers_bf16[cname] = scb

        calls = {
            "polar_express_eager": lambda x: polar_express(x, steps=5),
            "polar_express_triton_bf16": lambda x: polar_express_triton(x, steps=5, dtype=torch.bfloat16),
            "polar_express_fp8x2_bf16": lambda x: polar_express_fp8(x, steps=5, fp8_steps=2, dtype=torch.bfloat16),
            "ps2_55_eager": lambda x: polar_switch(x, "PS2_55"),
            "ps3_55_eager": lambda x: polar_switch(x, "PS3_55"),
            "ps2_55_blackwell_fp16": lambda x: polar_switch_blackwell(x, "PS2_55", dtype=torch.float16),
            "ps3_55_blackwell_fp16": lambda x: polar_switch_blackwell(x, "PS3_55", dtype=torch.float16),
            "gram_ns_blackwell_fp16": lambda x: gram_ns_blackwell(x, steps=5, dtype=torch.float16),
        }
        for cname in circuit_names:
            c, bf16_safe = _lookup_circuit(cname)
            sc16 = warm_scalers_fp16[cname]
            calls[f"{cname}_steady_triton_fp16"] = (
                lambda x, circ=c, scaler=sc16: run_circuit(
                    x, circ, dtype=torch.float16, use_triton_syrk=True, scaler=scaler, key="steady"
                )
            )
            calls[f"{cname}_schatten8_triton_fp16"] = (
                lambda x, circ=c: run_circuit(
                    x, circ, dtype=torch.float16, use_triton_syrk=True, schatten8=True
                )
            )
            if bf16_safe:
                scb = warm_scalers_bf16[cname]
                calls[f"{cname}_steady_triton_bf16"] = (
                    lambda x, circ=c, scaler=scb: run_circuit(
                        x, circ, dtype=torch.bfloat16, use_triton_syrk=True, scaler=scaler, key="steady"
                    )
                )

        names = list(calls)
        quality = {}
        for name in names:
            for _ in range(warmup):
                out = calls[name](G)
            torch.cuda.synchronize(device)
            Q = out.double().reshape(-1, *shape[-2:])
            finite = bool(torch.isfinite(Q).all().item())
            if finite:
                err = (Q - ref).norm(dim=(-2, -1)) / ref.norm(dim=(-2, -1))
                W = Q if Q.shape[-2] <= Q.shape[-1] else Q.mT
                eye = torch.eye(W.shape[-2], dtype=torch.float64, device=device)
                orth = (W @ W.mT - eye).norm(dim=(-2, -1)) / (W.shape[-2] ** 0.5)
                quality[name] = {
                    "finite": True,
                    "rel_fro_err_mean": float(err.mean().item()),
                    "rel_fro_err_max": float(err.max().item()),
                    "orth_err_rms": float(orth.mean().item()),
                }
            else:
                quality[name] = {"finite": False}

        gpu_events = {n: [] for n in names}
        wall_times = {n: [] for n in names}
        for _ in range(iters):
            order.shuffle(names)
            for name in names:
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                flush.zero_()
                s_ev.record()
                out = calls[name](G)
                e_ev.record()
                gpu_events[name].append((s_ev, e_ev))
                del out
            for name in names:
                flush.zero_()
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                out = calls[name](G)
                torch.cuda.synchronize(device)
                wall_times[name].append((time.perf_counter() - t0) * 1e3)
                del out
        torch.cuda.synchronize(device)

        pe_gpu = _stats([s.elapsed_time(e) for s, e in gpu_events["polar_express_eager"]])["median_ms"]
        ps2_gpu = _stats([s.elapsed_time(e) for s, e in gpu_events["ps2_55_eager"]])["median_ms"]
        for name in names:
            g_stat = _stats([s.elapsed_time(e) for s, e in gpu_events[name]])
            w_stat = _stats(wall_times[name])
            rows.append({
                "shape": list(shape),
                "method": name,
                "gpu": g_stat,
                "wall": w_stat,
                "speedup_vs_pe_eager_gpu": pe_gpu / g_stat["median_ms"],
                "speedup_vs_ps2_eager_gpu": ps2_gpu / g_stat["median_ms"],
                **quality[name],
            })
        print(f"LLM shape {shape} complete", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--circuits", nargs="*", default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--tier1-repeats", type=int, default=9)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--trajectories", type=Path, default=Path("artifacts/harvested_trajectories.pt"))
    ap.add_argument("--output", type=Path, default=Path("results/cycle1_head_to_head.json"))
    args = ap.parse_args()

    device = configure_device(args.device)
    loadavg_start = list(os.getloadavg())
    circuit_names = _select_circuits(args.circuits)
    print(f"Selected spectral circuits: {circuit_names}", flush=True)

    llm_rows = benchmark_llm_shapes(device, circuit_names, warmup=args.warmup, iters=args.iters, seed=args.seed)

    # Full Tier-1 verifier across baselines + Cycle 1 Blackwell kernels + top spectral circuits.
    registry = baseline_registry()
    registry.register("polar_express_triton_bf16", lambda x: polar_express_triton(x, steps=5, dtype=torch.bfloat16))
    registry.register(
        "polar_express_fp8x2_bf16",
        lambda x: polar_express_fp8(x, steps=5, fp8_steps=2, dtype=torch.bfloat16),
    )
    registry.register(
        "ps2_55_blackwell_fp16",
        lambda x: polar_switch_blackwell(x, "PS2_55", dtype=torch.float16),
        cost=lambda m: theoretical_cost("PS2_55", tuple(m.shape), input_dtype=m.dtype, device=m.device),
    )
    registry.register(
        "ps3_55_blackwell_fp16",
        lambda x: polar_switch_blackwell(x, "PS3_55", dtype=torch.float16),
        cost=lambda m: theoretical_cost("PS3_55", tuple(m.shape), input_dtype=m.dtype, device=m.device),
    )
    registry.register(
        "gram_ns_blackwell_fp16",
        lambda x: gram_ns_blackwell(x, steps=5, dtype=torch.float16),
        cost=lambda m: theoretical_cost("gram_ns", tuple(m.shape), input_dtype=m.dtype, device=m.device),
    )
    def _circuit_cost(cname: str, m: torch.Tensor) -> TheoreticalCost:
        circ, _ = _lookup_circuit(cname)
        rows, cols = m.shape[-2:]
        n, wide = sorted((rows, cols))
        batches = prod(m.shape[:-2])
        segs = len(circ.segments)
        ksum = sum(s.squarings for s in circ.segments)
        sym_flops = batches * (3 * segs * wide * n**2 + ksum * n**3)
        dense_flops = batches * (4 * segs * wide * n**2 + 2 * ksum * n**3)
        max_k = max(s.squarings for s in circ.segments)
        # W fp16/fp32 + R + R2 + max_k+1 scaled FP32 Gram buffers = max_k+4 buffers
        peak = batches * (4 * n * wide + 4 * (max_k + 4) * n * n)
        return TheoreticalCost(
            dense_flops,
            f"({4 * segs}α + {2 * ksum})n³",
            sym_flops,
            f"({3 * segs}α + {ksum})n³",
            peak,
        )

    for cname in circuit_names:
        registry.register(
            f"{cname}_triton_fp16",
            _make_spectral_wrapper(cname, torch.float16, use_triton=True),
            cost=lambda m, name=cname: _circuit_cost(name, m),
            uses_previous=True,
        )
        registry.register(
            f"{cname}_schatten8_fp16",
            _make_schatten8_wrapper(cname, torch.float16, use_triton=True),
            cost=lambda m, name=cname: _circuit_cost(name, m),
            uses_previous=False,
        )

    cases = list(synthetic_cases([128, 256], [1, 2, 4, 8], args.seed))
    if args.trajectories.is_file():
        cases.extend(harvested_cases(args.trajectories, all_layers=True))
    tier1_results = tier1_benchmark(
        cases, registry, device, warmup=3, repeats=args.tier1_repeats, seed=args.seed, divergence_limit=2.0
    )
    tier1_summary = tier1_summarize(tier1_results)

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = None
    payload = {
        "schema_version": 2,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha,
        "cli_args": vars(args) | {"trajectories": str(args.trajectories), "output": str(args.output)},
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": str(torch.__version__),
        "loadavg_start": loadavg_start,
        "loadavg_end": list(os.getloadavg()),
        "loadavg": list(os.getloadavg()),
        "selected_circuits": circuit_names,
        "llm_shapes": llm_rows,
        "tier1_case_count": len(cases),
        "tier1_summary": tier1_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
