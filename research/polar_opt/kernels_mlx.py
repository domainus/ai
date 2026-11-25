"""Float32 polar kernels for MLX, PyTorch MPS, and NumPy CPU.

The methods accept native 2-D arrays and return native arrays on the same
device. The benchmark uses seeded Gaussian inputs and an independent, reduced
NumPy SVD on the CPU. Input transfers and the SVD are outside timed regions.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

FULL_SHAPES = ((512, 2048), (1024, 4096), (2048, 2048))
QUICK_SHAPES = ((16, 64), (32, 128), (32, 32))
JORDAN = ((3.4445, -4.7750, 2.0315),) * 5

# First five quintics from Amsel et al.'s ten-step optimal_composition,
# l=0.001, cushion=0.02, safety_factor_eps=0.01 (PolarExpress f717e2c).
EXPRESS = (
    (8.237312490495555, -23.157747414558187, 16.6805684114459),
    (4.082441999064831, -2.8930477353325843, 0.5252849256975642),
    (3.926347992254648, -2.854746803476524, 0.531802242289498),
    (3.298218713308519, -2.42454198102671, 0.48632008358844137),
    (2.297036943455259, -1.636625581259032, 0.4002628455953629),
)

# Dao-AILab/gram-newton-schulz e45d0ac, PE preset including the 1.05 safety factor.
GRAM = (
    (7.892582874424408, -20.38301394587957, 13.555306149406924),
    (3.911484868135431, -2.5464635929060884, 0.4268988319673074),
    (3.760657955697423, -2.512819018216563, 0.4323647349070073),
    (3.160399673686287, -2.149649518898498, 0.3996366907664389),
    (2.1910971618617303, -1.441662010214663, 0.328146487623155),
)

# Du and Su, zhehangdu/polar-switch 6f382f1. Both circuits have zero residual.
# Each tuple gives the destination, square source, addend, and coefficients.
PS2_55 = (
    (0, (
        (1, 0, 0, 1, -1.039372688617118, 0.0011376825939118764),
        (0, 1, 1, 164.07644513428048, 49.58061223879703, 0.000500089059050944),
        (1, 0, 0, 1, 4.238519015111072, 2.9096036081096996),
        (0, 1, 0, 1, -0.7099643179341211, -3.188845137257469),
        (1, 0, 1, 1, 0.10116427259090495, 0.36987195573416237),
    )),
    (0, (
        (1, 0, 0, 3.6023274985419302, -6.053717133712312, 0.03621204508077079),
        (0, 1, 1, 1, 2.8221769622596162, 1.4534435713646865),
        (1, 0, 1, 1, 0.5287633553256361, -0.15855532053714858),
        (0, 1, 0, 1, -0.3880284725332113, -0.745819141374713),
        (1, 0, 1, 1, 2.072314176413394, 3.287628403409071),
    )),
)

PS3_55 = (
    (0, (
        (1, 0, 0, -1, 1.0372630529657008, -0.0011080641835321228),
        (2, 1, 1, -169.7011946136095, 50.90200678616221, -0.0004999998950287138),
        (1, 2, 2, -1, 4.259459173066772, -2.9027674651455544),
        (2, 1, 2, 1, 0.6326518033906352, -3.1055845493128005),
        (0, 2, 0, 1, -0.5157563645706955, 0.5614111759107233),
    )),
    (0, (
        (1, 0, 0, 7.281110320128563, -7.974535412410232, 0.041676558977155734),
        (2, 1, 1, 1, 2.6715673993971243, 1.495553622588639),
        (0, 2, 0, 1, -1.103759381672208, -0.3426296621352286),
        (1, 0, 1, 1, -0.3953056991433785, -1.172904844980118),
        (0, 1, 0, 1, 2.6713299568702316, 3.4991981111096626),
    )),
)


def detect_backend(requested="auto"):
    if requested not in ("auto", "mlx", "mps", "cpu"):
        raise ValueError(f"Unknown backend {requested!r}")
    if requested == "cpu":
        return "cpu"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if requested in ("auto", "mlx"):
            try:
                import mlx.core as mx
                if mx.metal.is_available():
                    return "mlx"
            except (ImportError, OSError):
                pass
        if requested in ("auto", "mps"):
            try:
                import torch
                if torch.backends.mps.is_available():
                    return "mps"
            except (ImportError, OSError):
                pass
    if requested != "auto":
        raise RuntimeError(f"{requested} is unavailable on this host")
    return "cpu"


def native_array(value, backend):
    if backend == "mlx":
        import mlx.core as mx
        return mx.array(value, dtype=mx.float32)
    if backend == "mps":
        import torch
        return torch.as_tensor(value, dtype=torch.float32, device="mps")
    return np.asarray(value, dtype=np.float32)


def synchronize(value, backend):
    if backend == "mlx":
        import mlx.core as mx
        mx.eval(value)
        mx.synchronize()
    elif backend == "mps":
        import torch
        torch.mps.synchronize()


def as_numpy(value):
    if type(value).__module__.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalized(x, safety=1.0):
    if x.ndim != 2 or min(x.shape) == 0:
        raise ValueError("Expected a nonempty 2-D real matrix")
    module = type(x).__module__.split(".")[0]
    if module == "torch":
        x = x.float()
    elif module == "mlx":
        import mlx.core as mx
        x = x.astype(mx.float32)
    else:
        x = np.asarray(x, dtype=np.float32)
    if x.shape[0] > x.shape[1]:
        x = x.T
    return x / (safety * (x * x).sum() ** 0.5 + 1e-7)


def _identity(x):
    module = type(x).__module__.split(".")[0]
    if module == "torch":
        import torch
        return torch.eye(x.shape[0], dtype=x.dtype, device=x.device)
    if module == "mlx":
        import mlx.core as mx
        return mx.eye(x.shape[0], dtype=x.dtype)
    return np.eye(x.shape[0], dtype=x.dtype)


def _direct(x, coefficients):
    for a, b, c in coefficients:
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x


def jordan_ns5(matrix):
    x = _direct(_normalized(matrix), JORDAN)
    return x if x.shape == matrix.shape else x.T


def polar_express(matrix):
    x = _direct(_normalized(matrix, safety=1.01), EXPRESS)
    return x if x.shape == matrix.shape else x.T


def gram_ns(matrix):
    x = _normalized(matrix)
    if x.shape[0] == x.shape[1]:
        return _direct(x, GRAM)
    identity = _identity(x)
    # Materialize the wide iterate after step two to limit Gram recurrence error.
    for segment in (GRAM[:2], GRAM[2:]):
        gram = x @ x.T
        product = None
        for step, (a, b, c) in enumerate(segment):
            multiplier = a * identity + b * gram + c * (gram @ gram)
            product = multiplier if product is None else multiplier @ product
            if step + 1 < len(segment):
                gram = multiplier @ gram @ multiplier.T
        x = product @ x
    return x if x.shape == matrix.shape else x.T


def polar_switch(matrix, variant="PS2_55"):
    table = {"PS2_55": PS2_55, "PS3_55": PS3_55}.get(variant.upper().replace("-", "_"))
    if table is None:
        raise ValueError(f"Unknown Polar Switch variant {variant!r}")
    x = _normalized(matrix)
    identity = _identity(x)
    for residual, steps in table:
        buffers = {0: x @ x.T}
        for dst, src, add, a, b, c in steps:
            operand = buffers[src]
            buffers[dst] = a * (operand @ operand.T) + b * buffers[add] + c * identity
        update = buffers[dst] @ x
        x = update + residual * x if residual else update
    return x if x.shape == matrix.shape else x.T


METHODS = {
    "jordan_ns5": jordan_ns5,
    "polar_express": polar_express,
    "gram_ns": gram_ns,
    "PS2_55": polar_switch,
    "PS3_55": lambda x: polar_switch(x, "PS3_55"),
}


def hardware_metadata():
    info = {"machine": platform.machine(), "system": platform.system(),
            "release": platform.release(), "hostname": platform.node()}
    if info["system"] == "Darwin":
        for key in ("machdep.cpu.brand_string", "hw.model", "hw.memsize"):
            try:
                result = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                        text=True, check=True, timeout=3)
            except (OSError, subprocess.SubprocessError):
                continue
            info[key] = result.stdout.strip()
    return info


def benchmark(*, backend="auto", quick=False, repeats=5, seed=0, reference_dtype="float64"):
    if repeats < 1 or reference_dtype not in ("float64", "float32"):
        raise ValueError("Expected positive repeats and a float64 or float32 SVD reference")
    backend = detect_backend(backend)
    if backend == "mlx":
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(seed)
    rows = []
    for shape in QUICK_SHAPES if quick else FULL_SHAPES:
        sample = rng.standard_normal(shape, dtype=np.float32)
        u, _, vh = np.linalg.svd(sample.astype(reference_dtype), full_matrices=False)
        reference = (u @ vh).astype(np.float64)
        reference_norm = np.linalg.norm(reference)
        x = native_array(sample, backend)
        synchronize(x, backend)
        for name, method in METHODS.items():
            synchronize(method(x), backend)
            samples_ms = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                output = method(x)
                synchronize(output, backend)
                samples_ms.append((time.perf_counter_ns() - start) / 1e6)
            output = as_numpy(output).astype(np.float64)
            finite = bool(np.isfinite(output).all())
            relative_error = float(np.linalg.norm(output - reference) / reference_norm) if finite else None
            rows.append({"method": name, "shape": list(shape), "latency_ms": statistics.median(samples_ms),
                         "samples_ms": samples_ms, "polar_error": relative_error, "finite": finite})
    library = {"cpu": "numpy", "mps": "torch", "mlx": "mlx"}[backend]
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    return {"backend": backend, "library": library, "library_version": version(library),
            "thread_environment": {key: os.environ[key] for key in thread_vars if key in os.environ},
            "hardware": hardware_metadata(), "quick": quick, "seed": seed,
            "created_at": datetime.now(timezone.utc).isoformat(), "dtype": "float32",
            "reference": f"numpy_{reference_dtype}_svd", "error_metric": "relative_frobenius",
            "warmups": 1, "repeats": repeats, "results": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "mlx", "mps", "cpu"), default="auto")
    parser.add_argument("--quick", action="store_true", help="Use three small CPU-test shapes")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repeats = args.repeats if args.repeats is not None else (2 if args.quick else 5)
    data = benchmark(backend=args.backend, quick=args.quick, repeats=repeats,
                     seed=args.seed, reference_dtype=args.reference_dtype)
    payload = json.dumps(data, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(f'{data["backend"]}: wrote {len(data["results"])} measurements to {args.output}')
