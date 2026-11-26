"""Mixed-precision polar kernels for MLX, PyTorch MPS, and NumPy CPU.

The methods accept native 2-D arrays and return native arrays on the same
device. The benchmark uses seeded Gaussian inputs and an independent, reduced
NumPy SVD on the CPU. Input transfers and quality checks are outside timing.
Normalization uses FP32. The work dtype selects intermediate buffers, while
the floating input dtype and original orientation select the output format.
NumPy emulates FP16/BF16 storage by rounding FP32 after each buffer write.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from functools import lru_cache, partial
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


DTYPES = ("float32", "float16", "bfloat16")

# A 128-thread group computes one upper 32x32 tile. Each Apple SIMD group
# accumulates four 8x8 blocks. Only the epilogue differs between PS and SYRK.
_METAL_TILE_SOURCE = """
    const uint n = x_shape[0], k = x_shape[1];
    const uint tile = threadgroup_position_in_grid.x;
    uint col = uint((sqrt(8.0f * float(tile) + 1.0f) - 1.0f) * 0.5f);
    // Correct rounding at triangular-number boundaries.
    if (col * (col + 1) / 2 > tile) --col;
    if ((col + 1) * (col + 2) / 2 <= tile) ++col;
    const uint row = tile - col * (col + 1) / 2;
    const uint tid = thread_position_in_threadgroup.x;
    const uint subgroup = simdgroup_index_in_threadgroup;
    const uint sr = (subgroup / 2) * 16, sc = (subgroup % 2) * 16;
    threadgroup float left[32 * 16], right[32 * 16], result[32 * 32];
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint base = 0; base < k; base += 16) {
        for (uint t = tid; t < 32 * 16; t += 128) {
            const uint r = row * 32 + t / 16, c = col * 32 + t / 16;
            const uint h = base + t % 16;
            left[t] = r < n && h < k ? float(x[r * k + h]) : 0.0f;
            right[t] = c < n && h < k ? float(x[c * k + h]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint h = 0; h < 16; h += 8) {
            simdgroup_float8x8 a[2], b[2];
            for (uint i = 0; i < 2; ++i) {
                simdgroup_load(a[i], left + (sr + i * 8) * 16 + h, 16);
                simdgroup_load(b[i], right + (sc + i * 8) * 16 + h, 16, ulong2(0), true);
            }
            for (uint i = 0; i < 2; ++i)
                for (uint j = 0; j < 2; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j)
            simdgroup_store(acc[i][j], result + (sr + i * 8) * 32 + sc + j * 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < 32 * 32; t += 128) {
        const uint r = row * 32 + t / 32, c = col * 32 + t % 32;
        if (r < n && c < n && r <= c) {
"""

_METAL_MIRROR = """
            out[r * n + c] = value;
            if (r != c) out[c * n + r] = value;
        }
    }
"""
_METAL_SYRK_SOURCE = _METAL_TILE_SOURCE + "const T value = T(result[t]);" + _METAL_MIRROR
_METAL_SYMPROD_SOURCE = _METAL_TILE_SOURCE + """
            const T value = T(coeff[0] * result[t] + coeff[1] * float(addend[r * n + c])
                              + (r == c ? coeff[2] : 0.0f));
""" + _METAL_MIRROR


def _launch_metal(kernel, x, *extra):
    import mlx.core as mx
    n = x.shape[0]
    tiles = (n + 31) // 32
    return kernel(inputs=[x, *extra], template=[("T", x.dtype)],
                  grid=(128 * (tiles * (tiles + 1) // 2), 1, 1),
                  threadgroup=(128, 1, 1), output_shapes=[(n, n)],
                  output_dtypes=[x.dtype], stream=mx.gpu)[0]


@lru_cache(maxsize=None)
def _metal_kernel(dtype, fused):
    """Compile and evaluate once per specialization, before any timed launches."""
    import mlx.core as mx
    try:
        kernel = mx.fast.metal_kernel(
            name="polar_symprod" if fused else "polar_syrk",
            input_names=["x", "addend", "coeff"] if fused else ["x"],
            output_names=["out"], ensure_row_contiguous=True,
            header="#include <metal_simdgroup_matrix>\nusing namespace metal;",
            source=_METAL_SYMPROD_SOURCE if fused else _METAL_SYRK_SOURCE)
        probe = mx.ones((32, 32), dtype=dtype)
        extra = (probe, mx.ones((3,))) if fused else ()
        mx.eval(_launch_metal(kernel, probe, *extra))
    except (RuntimeError, ValueError, AttributeError) as exc:
        warnings.warn(f"Metal {'symprod' if fused else 'SYRK'} unavailable for {dtype}, "
                      f"using dense matmul ({exc})", RuntimeWarning, stacklevel=2)
        return None
    return kernel


def metal_syrk(x):
    """Row Gram on MLX GPU. Use native matmul if the shader cannot compile."""
    if x.ndim != 2 or min(x.shape) == 0:
        raise ValueError("Expected a nonempty 2-D Metal input")
    kernel = _metal_kernel(x.dtype, False)
    return _launch_metal(kernel, x) if kernel is not None else x @ x.T


@lru_cache(maxsize=32)
def _metal_coefficients(a, b, c):
    import mlx.core as mx
    return mx.array([a, b, c], dtype=mx.float32)


def metal_symprod(x, addend, a, b, c):
    """Fuse a*(x @ x.T) + b*addend + c*I for a symmetric addend.

    Metal rounds once at the mirrored store. Dense fallback rounds the Gram first.
    """
    import mlx.core as mx
    if x.ndim != 2 or min(x.shape) == 0 or addend.shape != (x.shape[0], x.shape[0]):
        raise ValueError("Expected a nonempty matrix and a square addend matching its row count")
    kernel = _metal_kernel(x.dtype, True)
    if kernel is not None:
        return _launch_metal(kernel, x, addend, _metal_coefficients(a, b, c))
    return (a * (x @ x.T).astype(mx.float32) + b * addend.astype(mx.float32)
            + c * mx.eye(x.shape[0])).astype(x.dtype)


def _float32(x):
    module = type(x).__module__.split(".")[0]
    if module == "torch":
        return x.float()
    if module == "mlx":
        import mlx.core as mx
        return x.astype(mx.float32)
    return np.asarray(x, dtype=np.float32)


def as_numpy(value):
    module = type(value).__module__.split(".")[0]
    if module == "torch":
        import torch
        value = value.detach().cpu()
        return (value.float() if value.dtype == torch.bfloat16 else value).numpy()
    if module == "mlx":
        import mlx.core as mx
        if value.dtype == mx.bfloat16:
            value = value.astype(mx.float32)
    return np.asarray(value)


def _cpu_round(x, dtype):
    """Store work values in FP32, rounding to the requested format at each write."""
    x = np.asarray(x, dtype=np.float32)
    if dtype == "float32":
        return x
    if dtype == "float16":
        return x.astype(np.float16).astype(np.float32)
    bits = x.view(np.uint32)
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    # Preserve NaNs (including those whose payload lies wholly in the low half).
    return np.where(np.isnan(x), np.float32(np.nan), rounded.view(np.float32))


class _Work:
    """Backend casts and operations that write a new working-precision buffer."""

    def __init__(self, x, dtype, use_metal_syrk):
        if dtype not in DTYPES:
            raise ValueError(f"Unknown working dtype {dtype!r}")
        module = type(x).__module__.split(".")[0]
        self.metal = use_metal_syrk and module == "mlx"
        if module == "mlx":
            import mlx.core as mx
            self.cast = lambda value: value.astype(getattr(mx, dtype))
        elif module == "torch":
            import torch
            self.cast = lambda value: value.to(getattr(torch, dtype))
        else:
            self.cast = partial(_cpu_round, dtype=dtype)

    def mm(self, x, y):
        return self.cast(x @ y)

    def gram(self, x):
        return metal_syrk(x) if self.metal else self.mm(x, x.T)


def _normalized(x, safety=1.0):
    if x.ndim != 2 or min(x.shape) == 0:
        raise ValueError("Expected a nonempty 2-D real matrix")
    x = _float32(x)
    if x.shape[0] > x.shape[1]:
        x = x.T
    return x / (safety * (x * x).sum() ** 0.5 + 1e-7)


def _identity(x):
    module = type(x).__module__.split(".")[0]
    if module == "torch":
        import torch
        return torch.eye(x.shape[0], dtype=torch.float32, device=x.device)
    if module == "mlx":
        import mlx.core as mx
        return mx.eye(x.shape[0], dtype=mx.float32)
    return np.eye(x.shape[0], dtype=np.float32)


def _restore(x, matrix):
    if x.shape != matrix.shape:
        x = x.T
    if type(matrix).__module__.startswith("torch"):
        return x.to(matrix.dtype) if matrix.is_floating_point() else x.float()
    if type(matrix).__module__.startswith("mlx"):
        import mlx.core as mx
        return x.astype(matrix.dtype if mx.issubdtype(matrix.dtype, mx.floating) else mx.float32)
    if not np.issubdtype(matrix.dtype, np.floating):
        return _float32(x)
    return x.astype(matrix.dtype, copy=False)


def _direct(x, coefficients, work):
    for a, b, c in coefficients:
        gram = work.gram(x)
        poly = work.cast(b * _float32(gram) + c * _float32(work.gram(gram)))
        x = work.cast(a * _float32(x) + _float32(work.mm(poly, x)))
    return x


def jordan_ns5(matrix, dtype="float32", use_metal_syrk=False):
    work = _Work(matrix, dtype, use_metal_syrk)
    x = _direct(work.cast(_normalized(matrix)), JORDAN, work)
    return _restore(x, matrix)


def polar_express(matrix, dtype="float32", use_metal_syrk=False):
    work = _Work(matrix, dtype, use_metal_syrk)
    x = _direct(work.cast(_normalized(matrix, safety=1.01)), EXPRESS, work)
    return _restore(x, matrix)


def gram_ns(matrix, dtype="float32", use_metal_syrk=False):
    work = _Work(matrix, dtype, use_metal_syrk)
    x = work.cast(_normalized(matrix))
    if x.shape[0] == x.shape[1]:
        return _restore(_direct(x, GRAM, work), matrix)
    identity = _identity(x)
    # Restart after step two. A recurrent Gram may lose exact symmetry to
    # rounding, so only the initial Gram square of each segment uses SYRK.
    for segment in (GRAM[:2], GRAM[2:]):
        gram = work.gram(x)
        product = None
        for step, (a, b, c) in enumerate(segment):
            square = work.gram(gram) if step == 0 else work.mm(gram, gram)
            multiplier = work.cast(a * identity + b * _float32(gram) + c * _float32(square))
            product = multiplier if product is None else work.mm(multiplier, product)
            if step + 1 < len(segment):
                gram = work.mm(work.mm(multiplier, gram), multiplier.T)
        x = work.mm(product, x)
    return _restore(x, matrix)


def polar_switch(matrix, variant="PS2_55", dtype="float32", use_metal_syrk=False):
    table = {"PS2_55": PS2_55, "PS3_55": PS3_55}.get(variant.upper().replace("-", "_"))
    if table is None:
        raise ValueError(f"Unknown Polar Switch variant {variant!r}")
    work = _Work(matrix, dtype, use_metal_syrk)
    x = work.cast(_normalized(matrix))
    identity = None if work.metal else _identity(x)
    for residual, steps in table:
        buffers = {0: work.gram(x)}
        for dst, src, add, a, b, c in steps:
            operand, addend = buffers[src], buffers[add]
            if work.metal:
                buffers[dst] = metal_symprod(operand, addend, a, b, c)
            else:
                buffers[dst] = work.cast(a * _float32(work.gram(operand)) + b * _float32(addend)
                                         + c * identity)
        update = work.mm(buffers[dst], x)
        x = work.cast(update + residual * x) if residual else update
    return _restore(x, matrix)


METHODS = {"jordan_ns5": jordan_ns5, "polar_express": polar_express, "gram_ns": gram_ns,
           "PS2_55": polar_switch, "PS3_55": partial(polar_switch, variant="PS3_55")}


def _block_orders(rng, count):
    """Randomized Latin squares balance every method across all time slots."""
    orders = []
    while len(orders) < count:
        base = list(METHODS)
        rng.shuffle(base)
        for offset in rng.permutation(len(base)):
            orders.append(base[offset:] + base[:offset])
            if len(orders) == count:
                break
    return orders


def _quality(output, reference):
    q = as_numpy(output).astype(np.float64)
    if not np.isfinite(q).all():
        return dict(finite=False, diverged=True, polar_error=None,
                    relative_frobenius_error=None, sigma_max=None, orthogonality_err=None)
    w = q if q.shape[0] <= q.shape[1] else q.T
    gram = w @ w.T
    sigma = float(np.sqrt(np.linalg.eigvalsh(gram)[-1]))
    error = float(np.linalg.norm(q - reference) / np.linalg.norm(reference))
    orth = float(np.linalg.norm(gram - np.eye(len(gram))) / np.sqrt(len(gram)))
    return dict(finite=True, diverged=sigma > 2.0, polar_error=error,
                relative_frobenius_error=error, sigma_max=sigma, orthogonality_err=orth)


def _measure(x, reference, backend, dtype, requested, rng, warmups, repeats):
    calls = {name: partial(fn, x, dtype=dtype, use_metal_syrk=requested)
             for name, fn in METHODS.items()}
    syrk = symprod = False
    if requested and backend == "mlx":
        import mlx.core as mx
        work_dtype = getattr(mx, dtype)
        syrk = _metal_kernel(work_dtype, False) is not None
        symprod = _metal_kernel(work_dtype, True) is not None
    for block in _block_orders(rng, warmups):
        for name in block:
            synchronize(calls[name](), backend)
    orders = _block_orders(rng, repeats)
    times = {name: [] for name in calls}
    outputs = {}
    for block in orders:
        for name in block:
            start = time.perf_counter_ns()
            out = calls[name]()
            synchronize(out, backend)
            times[name].append((time.perf_counter_ns() - start) / 1e6)
            outputs[name] = out
    rows = []
    for name, samples in times.items():
        p25, median, p75 = np.percentile(samples, [25, 50, 75]).tolist()
        rows.append(dict(method=name, shape=list(x.shape), backend=backend, dtype=dtype,
                         use_metal_syrk=syrk, use_metal_symprod=symprod and name.startswith("PS"),
                         output_dtype=str(outputs[name].dtype), median_ms=median, latency_ms=median,
                         iqr_ms=p75-p25, p25_ms=p25, p75_ms=p75, min_ms=min(samples),
                         samples_ms=samples, **_quality(outputs[name], reference)))
    return rows, orders


def benchmark(*, backend="auto", quick=False, warmups=5, repeats=20, seed=0, seeds=None,
              reference_dtype="float64", dtypes=("float32",), shapes=None, use_metal_syrk=False):
    if warmups < 0 or repeats < 1 or reference_dtype not in ("float64", "float32"):
        raise ValueError("Expected nonnegative warmups, positive repeats and a float64 or float32 SVD")
    if isinstance(dtypes, str):
        dtypes = dtypes.split(",")
    if not dtypes or any(dtype not in DTYPES for dtype in dtypes):
        raise ValueError(f"Expected working dtypes from {DTYPES}")
    if shapes is None:
        shapes = QUICK_SHAPES if quick else FULL_SHAPES
    if not shapes or any(len(shape) != 2 or min(shape) < 1 for shape in shapes):
        raise ValueError("Expected positive 2-D shapes")
    seeds = list(seeds) if seeds is not None else [seed]
    if not seeds:
        raise ValueError("Expected at least one input seed")
    backend = detect_backend(backend)
    if backend == "mlx":
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
    timestamp = datetime.now(timezone.utc).isoformat()
    rows, orders = [], []
    for input_seed in seeds:
        rng = np.random.default_rng(input_seed)
        order_rng = np.random.default_rng(np.random.SeedSequence([input_seed, 1]))
        for shape in shapes:
            sample = rng.standard_normal(shape, dtype=np.float32)
            u, _, vh = np.linalg.svd(sample.astype(reference_dtype), full_matrices=False)
            reference = (u @ vh).astype(np.float64)
            x = native_array(sample, backend)
            synchronize(x, backend)
            for dtype in dtypes:
                measured, blocks = _measure(x, reference, backend, dtype, use_metal_syrk,
                                            order_rng, warmups, repeats)
                rows.extend(dict(row, seed=input_seed) for row in measured)
                orders.append(dict(shape=list(shape), dtype=dtype, seed=input_seed, blocks=blocks))
    library = {"cpu": "numpy", "mps": "torch", "mlx": "mlx"}[backend]
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    source = Path(__file__).resolve()
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source.parent,
                                          stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    return {"backend": backend, "library": library, "library_version": version(library),
            "thread_environment": {key: os.environ[key] for key in thread_vars if key in os.environ},
            "hardware": hardware_metadata(), "quick": quick, "seed": seeds[0], "seeds": seeds,
            "created_at": timestamp, "timestamp": timestamp, "git_sha": git_sha,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "dtype": dtypes[0] if len(dtypes) == 1 else list(dtypes), "dtypes": list(dtypes),
            "input_dtype": "float32", "use_metal_syrk_requested": use_metal_syrk,
            "use_metal_syrk": any(r["use_metal_syrk"] for r in rows),
            "reference": f"numpy_{reference_dtype}_svd", "error_metric": "relative_frobenius",
            "divergence_limit": 2.0, "divergences": sum(r["diverged"] for r in rows),
            "warmups": warmups, "repeats": repeats, "method_order": orders, "results": rows}


def _parse_shapes(value):
    try:
        return [tuple(map(int, shape.lower().split("x"))) for shape in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated shapes such as 512x2048,2048x2048") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "mlx", "mps", "cpu"), default="auto")
    parser.add_argument("--quick", action="store_true", help="Use small shapes and default to two repeats")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, help="Timed blocks (default 20, or 2 with --quick)")
    parser.add_argument("--dtypes", default="float32", help="Comma-separated working precisions")
    parser.add_argument("--shapes", type=_parse_shapes, help="Comma-separated shapes, e.g. 512x2048,2048x2048")
    parser.add_argument("--use-metal-syrk", action="store_true", help="Use triangular SYRK and fused PS on MLX")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=0)
    seed_group.add_argument("--seeds", help="Comma-separated independent input seeds")
    parser.add_argument("--reference-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repeats = args.repeats if args.repeats is not None else (2 if args.quick else 20)
    data = benchmark(backend=args.backend, quick=args.quick, warmups=args.warmups, repeats=repeats,
                     seed=args.seed, seeds=list(map(int, args.seeds.split(","))) if args.seeds else None,
                     reference_dtype=args.reference_dtype, dtypes=args.dtypes, shapes=args.shapes,
                     use_metal_syrk=args.use_metal_syrk)
    data["command"] = [sys.executable, *sys.argv]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f'{data["backend"]}: wrote {len(data["results"])} measurements to {args.output} '
          f'({data["divergences"]} diverged)')
