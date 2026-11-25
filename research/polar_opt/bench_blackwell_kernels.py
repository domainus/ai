"""Benchmark Triton / FP8 polar kernels against PyTorch (cuBLAS) on an NVIDIA Blackwell GPU.

Timing: CUDA events around each call, warmup (also triggers Triton autotune), L2 flushed before
every timed call, ``torch.cuda.synchronize()`` around the measurement.  Times are eager wall-clock on
the GPU stream, i.e. they include kernel-launch gaps (relevant for the multi-kernel FP8 path).

TFLOPS are *nominal dense* FLOPs (2*N*N*M for a Gram, 4*N*N*M + 2*N^3 per polar step) so that the
ratio of TFLOPS equals the ratio of times; ``triton_syrk`` does ~half of that work.

Run:  PYTHONPATH=. .venv/bin/python research/polar_opt/bench_blackwell_kernels.py
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import triton

from research.polar_opt.kernels_blackwell import (
    polar_express_fp8,
    polar_express_torch,
    polar_express_triton,
    polar_svd_fp64,
    triton_syrk,
)

SHAPES = [(512, 2048), (1024, 4096), (2048, 8192), (4096, 4096), (4, 1024, 4096)]
DTYPES = [torch.bfloat16, torch.float16]
STEPS = 5
OUT = Path(__file__).resolve().parents[2] / "results" / "blackwell_kernels_cycle1.json"

_flush = None


def time_fn(fn, warmup: int = 5, iters: int = 30) -> dict:
    """Median / IQR in ms of ``fn()`` using CUDA events, with an L2 flush before each call."""
    global _flush
    if _flush is None:
        _flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")  # >> 64 MB L2
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    pairs = []
    for _ in range(iters):
        _flush.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        pairs.append((s, e))
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in pairs)
    q = statistics.quantiles(t, n=4)
    return dict(median_ms=statistics.median(t), q1_ms=q[0], q3_ms=q[2], iqr_ms=q[2] - q[0], min_ms=t[0], iters=iters)


def dims(shape):
    b = shape[0] if len(shape) == 3 else 1
    n, m = sorted(shape[-2:])
    return b, n, m


def gram_flops(shape):
    b, n, m = dims(shape)
    return 2.0 * b * n * n * m


def polar_flops(shape):
    b, n, m = dims(shape)
    return STEPS * b * (4.0 * n * n * m + 2.0 * n**3)


def polar_errors(Q: torch.Tensor, P: torch.Tensor) -> dict:
    """Relative Frobenius error to the exact polar factor, and distance of Q from orthogonality."""
    Qd, Pd = Q.double(), P.double()
    rel = (Qd - Pd).norm(dim=(-2, -1)) / Pd.norm(dim=(-2, -1))
    W = Qd if Qd.shape[-2] <= Qd.shape[-1] else Qd.mT
    eye = torch.eye(W.shape[-2], dtype=torch.float64, device=W.device)
    orth = (W @ W.mT - eye).norm(dim=(-2, -1)) / W.shape[-2] ** 0.5
    ok = bool(torch.isfinite(Qd).all())
    return dict(rel_fro_err_mean=rel.mean().item(), rel_fro_err_max=rel.max().item(),
                orth_err_rms=orth.mean().item(), finite=ok)


def row(name, shape, dtype, timing, flops, extra=None):
    r = dict(name=name, shape=list(shape), dtype=str(dtype).replace("torch.", ""), **timing,
             tflops=flops / (timing["median_ms"] * 1e-3) / 1e12)
    r.update(extra or {})
    return r


def fmt(r):
    err = f"  err={r['rel_fro_err_mean']:.4f} orth={r['orth_err_rms']:.4f}" if "rel_fro_err_mean" in r else ""
    path = f"  [{r['path']}]" if r.get("path") not in (None, "fp8-early") else ""
    return (f"  {r['name']:<22}{r['dtype']:<9}{str(tuple(r['shape'])):<18}{r['median_ms']:9.4f} ms "
            f"(IQR {r['iqr_ms']:.4f}) {r['tflops']:8.1f} TFLOPS{err}{path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    dev = torch.cuda.get_device_properties(0)
    results = dict(
        env=dict(gpu=dev.name, capability=list(torch.cuda.get_device_capability()), sm_count=dev.multi_processor_count,
                 torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                 python=platform.python_version(), steps=STEPS, iters=args.iters,
                 timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z")),
        gram=[], polar=[],
    )
    print(f"{dev.name}  sm_{dev.major}{dev.minor}  torch {torch.__version__}  triton {triton.__version__}")

    print("\n== Gram matrix X @ X.T : cuBLAS vs triton_syrk ==")
    for shape in SHAPES:
        for dt in DTYPES:
            torch.manual_seed(0)
            X = (torch.randn(*shape, device="cuda") * 0.05).to(dt)
            Xw = X if X.shape[-2] <= X.shape[-1] else X.mT.contiguous()
            ref = Xw.double() @ Xw.double().mT
            for name, fn in (("cublas X@X.T", lambda: Xw @ Xw.mT), ("triton_syrk", lambda: triton_syrk(Xw))):
                out = fn()
                err = ((out.double() - ref).norm() / ref.norm()).item()
                r = row(name, shape, dt, time_fn(fn, iters=args.iters), gram_flops(shape), dict(rel_fro_err=err))
                results["gram"].append(r)
                print(fmt(r) + f"  rel_err={err:.2e}")

    print("\n== 5-step PolarExpress (error vs fp64 SVD polar factor) ==")
    for shape in SHAPES:
        torch.manual_seed(0)
        X32 = torch.randn(*shape, device="cuda")
        t0 = time.time()
        P = polar_svd_fp64(X32)
        print(f" {shape}: fp64 SVD reference in {time.time() - t0:.1f}s")
        for dt in DTYPES:
            X = X32.to(dt)
            methods = [
                ("torch (cuBLAS)", None, lambda: polar_express_torch(X, STEPS)),
                ("triton fused", None, lambda: polar_express_triton(X, STEPS)),
                ("fp8x1 + late", 1, lambda: polar_express_fp8(X, STEPS, fp8_steps=1)),
                ("fp8x2 + late", 2, lambda: polar_express_fp8(X, STEPS, fp8_steps=2)),
            ]
            for name, k, fn in methods:
                Q = fn()
                extra = polar_errors(Q, P)
                if k is not None:
                    _, meta = polar_express_fp8(X, STEPS, fp8_steps=k, return_meta=True)
                    extra["path"] = meta["path"]
                    extra["fp8_steps_used"] = meta["fp8_steps_used"]
                r = row(name, shape, dt, time_fn(fn, iters=args.iters), polar_flops(shape), extra)
                results["polar"].append(r)
                print(fmt(r))
        del P

    # Speedups relative to the cuBLAS baseline of the same shape/dtype.
    def speedups(rows, base):
        b = {(r["shape"].__str__(), r["dtype"]): r["median_ms"] for r in rows if r["name"] == base}
        for r in rows:
            r["speedup_vs_cublas"] = b[(r["shape"].__str__(), r["dtype"])] / r["median_ms"]

    speedups(results["gram"], "cublas X@X.T")
    speedups(results["polar"], "torch (cuBLAS)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
