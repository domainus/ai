"""Evaluate orthogonalizers on held-out real FineWeb-Edu post-Nesterov pairs (Cycle 2).

Computes exact FP64 SVD targets once per harvested pair, runs each comparator on CUDA with
full FP32 accumulation precision, and writes summary accuracy and divergence statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from research.polar_opt.baselines import polar_express, svd_polar_fp64
from research.polar_opt.harness_tier1 import _quality
from research.polar_opt.harness_tier2_real import ROOT, git_info, gpu_telemetry
from research.polar_opt.kernels_blackwell import polar_express_triton
from research.polar_opt.polar_switch_blackwell import polar_switch_blackwell
from research.polar_opt.schatten_switch_blackwell import schatten_switch_blackwell


def _methods() -> dict:
    return {
        "pe_eager": lambda g: polar_express(g),
        "pe_triton_bf16": lambda g: polar_express_triton(g, steps=5, dtype=torch.bfloat16),
        "ps2_55_blackwell_fp16": lambda g: polar_switch_blackwell(g, "PS2_55", dtype=torch.float16),
        "ps3_55_blackwell_fp16": lambda g: polar_switch_blackwell(g, "PS3_55", dtype=torch.float16),
        "ss3_54_blackwell_fp16": lambda g: schatten_switch_blackwell(g, "SS3_54", dtype=torch.float16),
        "ss3_53_blackwell_fp16": lambda g: schatten_switch_blackwell(g, "SS3_53", dtype=torch.float16),
        "ss3_35_blackwell_fp16": lambda g: schatten_switch_blackwell(g, "SS3_35", dtype=torch.float16),
        "ss3_34_blackwell_fp16": lambda g: schatten_switch_blackwell(g, "SS3_34", dtype=torch.float16),
        "ss3_35_bf16": lambda g: schatten_switch_blackwell(g, "SS3_35", dtype=torch.bfloat16),
        "ss3_34_bf16": lambda g: schatten_switch_blackwell(g, "SS3_34", dtype=torch.bfloat16),
    }


def _candidate_methods(path: Path, *, include_failed: bool = False) -> dict:
    """``cand_<name>_fp16`` (and ``_bf16`` if verified) for passing validated circuits of a search JSON."""
    from research.polar_opt.schatten_switch_blackwell import SchattenSwitchCircuit

    out = {}
    for name, record in json.loads(path.read_text()).get("validated", {}).items():
        if not include_failed and not record.get("fp16_passed", False):
            continue
        circuit = SchattenSwitchCircuit.from_dict(record["circuit"])
        out[f"cand_{name}_fp16"] = lambda g, c=circuit: schatten_switch_blackwell(g, c, dtype=torch.float16)
        if circuit.bf16_verified and record.get("bf16_verified", circuit.bf16_verified):
            out[f"cand_{name}_bf16"] = lambda g, c=circuit: schatten_switch_blackwell(g, c, dtype=torch.bfloat16)
    return out


def _resolve_eval_output(output: Path | None, candidates: Path | None) -> Path:
    """Resolve the eval output path, requiring a distinct non-baseline path when --candidates is set."""
    default_out = ROOT / "results" / "cycle2_heldout_nesterov_eval.json"
    if candidates is not None:
        if output is None:
            raise ValueError("--output is required when --candidates is set so baseline results are not overwritten")
        if output.resolve() == default_out.resolve():
            raise ValueError(f"--output ({output}) must not overwrite baseline {default_out} when --candidates is set")
        if output.resolve() == candidates.resolve():
            raise ValueError(f"--output ({output}) must differ from --candidates ({candidates})")
        return output
    return default_out if output is None else output


def _summarize(rows: list[dict]) -> dict:
    errs = [r["relative_frobenius_error"] for r in rows if "relative_frobenius_error" in r]
    sig_hi = [r["sigma_max"] for r in rows if "sigma_max" in r]
    sig_lo = [r["sigma_min"] for r in rows if "sigma_min" in r]
    return {
        "cases": len(rows),
        "diverged": sum(1 for r in rows if r["diverged"]),
        "nonfinite": sum(1 for r in rows if r["nonfinite"]),
        "median_relative_frobenius_error": float(np.median(errs)) if errs else None,
        "p90_relative_frobenius_error": float(np.quantile(errs, 0.9)) if errs else None,
        "max_relative_frobenius_error": float(max(errs)) if errs else None,
        "max_sigma_max": float(max(sig_hi)) if sig_hi else None,
        "min_sigma_min": float(min(sig_lo)) if sig_lo else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts" / "harvested_nesterov_real.pt")
    parser.add_argument("--output", type=Path, default=None,
                        help="output JSON path (defaults to results/cycle2_heldout_nesterov_eval.json; "
                             "explicit distinct path required when --candidates is set)")
    parser.add_argument("--divergence-limit", type=float, default=2.0)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="subset of the built-in methods (default: all)")
    parser.add_argument("--candidates", type=Path, default=None,
                        help="search JSON whose 'validated' circuits are evaluated out of sample")
    parser.add_argument("--include-failed-candidates", action="store_true",
                        help="also load validated candidates whose fp16_passed is false or missing")
    args = parser.parse_args(argv)
    try:
        args.output = _resolve_eval_output(args.output, args.candidates)
    except ValueError as exc:
        parser.error(str(exc))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Blackwell held-out Nesterov evaluation")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    bundle = torch.load(args.input, map_location="cpu", weights_only=True)
    pairs = bundle["pairs"]
    load_start = os.getloadavg()
    gpu_start = gpu_telemetry(device)

    targets = []
    for pair in pairs:
        G = pair["N_t"].to(device=device, dtype=torch.float32)
        P = svd_polar_fp64(G)
        targets.append((pair["phase"], pair["name"], pair["step_t"], list(pair["shape"]), G, P, P.cpu()))

    methods = _methods()
    if args.methods:
        unknown = sorted(set(args.methods) - set(methods))
        if unknown:
            raise ValueError(f"Unknown methods {unknown}; available: {sorted(methods)}")
        methods = {name: methods[name] for name in args.methods}
    if args.candidates:
        methods.update(_candidate_methods(args.candidates, include_failed=args.include_failed_candidates))
    summary, by_phase, cases = {}, {}, {}
    for method_name, fn in methods.items():
        all_rows = []
        phase_rows: dict[str, list[dict]] = defaultdict(list)
        case_rows = []
        for phase, param_name, step_t, shape, G, P, P_cpu in targets:
            with torch.no_grad():
                out = fn(G)
            torch.cuda.synchronize(device)
            q = _quality(out, P, P_cpu, args.divergence_limit)
            all_rows.append(q)
            phase_rows[phase].append(q)
            case_rows.append({"phase": phase, "name": param_name, "step_t": step_t, "shape": shape, **q})
        summary[method_name] = _summarize(all_rows)
        by_phase[method_name] = {p: _summarize(rows) for p, rows in sorted(phase_rows.items())}
        cases[method_name] = case_rows
        s = summary[method_name]
        med = f"{s['median_relative_frobenius_error']:.4f}" if s["median_relative_frobenius_error"] is not None else "n/a"
        mx = f"{s['max_relative_frobenius_error']:.4f}" if s["max_relative_frobenius_error"] is not None else "n/a"
        smax = f"{s['max_sigma_max']:.4f}" if s["max_sigma_max"] is not None else "n/a"
        print(f"  {method_name:<24} div={s['diverged']}/{s['cases']}  med={med}  p90={s['p90_relative_frobenius_error']:.4f}  max={mx}  smax={smax}")

    payload = {
        "env": {
            **git_info(),
            "command": [sys.executable, *sys.argv],
            "gpu": torch.cuda.get_device_name(device),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "loadavg_start": load_start,
            "loadavg_end": os.getloadavg(),
            "gpu_start": gpu_start,
            "gpu_end": gpu_telemetry(device),
        },
        "input": str(args.input),
        "harvest_meta": bundle.get("meta", {}),
        "starts": bundle.get("starts"),
        "layers": bundle.get("layers"),
        "pair_count": len(targets),
        "divergence_limit": args.divergence_limit,
        "candidates": str(args.candidates) if args.candidates else None,
        "summary": summary,
        "by_phase": by_phase,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
