"""Measure resident MLX primitives in interleaved blocks, including GPU completion."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np

from research.polar_opt import kernels_mlx as kernels


def benchmark(*, quick=False, warmups=5, repeats=20, shapes=None, dtypes=kernels.DTYPES):
    if warmups < 0 or repeats < 1 or (not quick and (warmups < 5 or repeats < 20)):
        raise ValueError("Full measurements require at least 5 warmups and 20 timed blocks")
    shapes = shapes or ([(16, 64)] if quick else [*kernels.FULL_SHAPES, (4096, 4096)])
    if any(len(shape) != 2 or min(shape) < 1 for shape in shapes):
        raise ValueError("Expected nonempty 2-D shapes")
    if not dtypes or any(dtype not in kernels.DTYPES for dtype in dtypes):
        raise ValueError(f"Expected working dtypes from {kernels.DTYPES}")
    kernels.detect_backend("mlx")
    mx.set_default_device(mx.gpu)
    mx.random.seed(0)
    timestamp = datetime.now(timezone.utc).isoformat()
    before = kernels.thermal_snapshot()
    provenance = kernels.git_provenance()
    rows, cases = [], []
    for n, k in shapes:
        base = mx.random.normal((n, k)) / np.sqrt(k).item()
        for dtype in dtypes:
            case_before = kernels.thermal_snapshot()
            fmt = getattr(mx, dtype)
            x = base.astype(fmt)
            eye = mx.eye(n)
            bias = eye.astype(fmt)
            bias32 = bias.astype(mx.float32)
            mx.eval(x, bias, bias32, eye)

            def affine(gram):
                return (-0.5 * gram.astype(mx.float32) + 2.0 * bias32 - 0.25 * eye).astype(fmt)

            calls = {"dense_gram": lambda: x @ x.T,
                     "metal_syrk": lambda: kernels.metal_syrk(x),
                     "dense_affine": lambda: affine(x @ x.T),
                     "metal_unfused_affine": lambda: affine(kernels.metal_syrk(x)),
                     "metal_symprod": lambda: kernels.metal_symprod(x, bias, -0.5, 2.0, -0.25)}
            tiles = {}
            for name in calls:
                if name.startswith("metal"):
                    fused = name == "metal_symprod"
                    fn = kernels._metal_kernel(fmt, fused, x.size < 8 or (fused and bias.size < 8))
                    if fn is None or n > kernels._METAL_MAX_N:
                        raise RuntimeError(f"No Metal kernel for {name} {dtype} {n}x{k}")
                    tiles[name] = fn.keywords["tile_size"]
            samples, walls, outputs, orders = kernels._time_blocks(
                calls, "mlx", np.random.default_rng(100), warmups, repeats)
            for name in calls:
                ref = "dense_gram" if name in ("dense_gram", "metal_syrk") else "dense_affine"
                diff = mx.max(mx.abs(outputs[name].astype(mx.float32) - outputs[ref].astype(mx.float32))).item()
                # On M5 Max, native FP32 MLX matmul differs from FP64 by
                # ~1e-3 on these unit-scale random Gram matrices.
                tol = {"float32": 0.003, "float16": 0.003, "bfloat16": 0.025}[dtype]
                finite = bool(np.isfinite(diff))
                rows.append(dict(method=name, shape=[n, k], dtype=dtype, tile_size=tiles.get(name),
                                 max_abs_diff_vs_dense=diff if finite else None, abs_tolerance=tol,
                                 diverged=not finite or diff > tol,
                                 **kernels._sample_stats(samples[name], walls[name])))
            cases.append(dict(shape=[n, k], dtype=dtype, blocks=orders, thermal_before=case_before,
                              thermal_after=kernels.thermal_snapshot()))
    sources = [Path(__file__), Path(kernels.__file__)]
    return dict(backend="mlx", library_version=version("mlx"), timestamp=timestamp, **provenance,
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                hardware=kernels.hardware_metadata(), thermal_before=before, thermal_after=kernels.thermal_snapshot(),
                warmups=warmups, repeats=repeats, quick=quick, seed=0, order_seed=100,
                results=rows, method_order=cases, command=[sys.executable, *sys.argv],
                sleep_detected=any(r["sleep_detected"] for r in rows),
                divergences=sum(r["diverged"] for r in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--shapes", type=kernels._parse_shapes)
    parser.add_argument("--dtypes", default="float16,bfloat16,float32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = benchmark(quick=args.quick, warmups=args.warmups if args.warmups is not None else (1 if args.quick else 5),
                     repeats=args.repeats if args.repeats is not None else (2 if args.quick else 20),
                     shapes=args.shapes, dtypes=args.dtypes.split(","))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f'Wrote {len(data["results"])} primitive measurements to {args.output} ({data["divergences"]} diverged)')
