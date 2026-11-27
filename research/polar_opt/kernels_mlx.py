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


# Discovered circuits: copied verbatim from results/schatten_switch_search_cycle2.json.
_CIRCUIT_SPECS: dict = {
    "SS3_54": {
        "name": "SS3_54",
        "buffers": 3,
        "u_safe": 1.05,
        "lo": 0.00286,
        "error": 0.07982588482306929,
        "bf16_verified": False,
        "squarings": [5, 4],
        "flops": "(6α + 9)n³",
        "table": [
            [0.0, [
                [1, 0, 0, 1.0, -1.035678364265212, 0.0],
                [2, 1, 1, -44.95361416143577, -13.51553474507387, -0.015878595332442222],
                [1, 2, 2, -4.569196296690892, 5.092643625703066, -0.9179839421935934],
                [2, 1, 2, 1.4182468064291267, 0.32580136819175276, -0.4130735383448542],
                [0, 2, 0, 44.21037324366673, -1.0366534520808317, 1.1258551001231605],
            ]],
            [0.0, [
                [1, 0, 0, 1.0, -1.2041354084316795, 0.3052004453155656],
                [0, 1, 0, -11.240335609969362, 0.4504520793471352, 0.6056874882617449],
                [1, 0, 0, -1.3154792860430617, 1.3183888243288422, -0.16196259793937848],
                [0, 1, 0, 7.874247039551604, -2.8721964431448384, 3.7453983723791087],
            ]],
        ],
    },
    "SS3_53": {
        "name": "SS3_53",
        "buffers": 3,
        "u_safe": 1.05,
        "lo": 0.003,
        "error": 0.08754726536476465,
        "bf16_verified": False,
        "squarings": [5, 3],
        "flops": "(6α + 8)n³",
        "table": [
            [0.0, [
                [1, 0, 0, 1.0, -1.032065172692518, 0.001612489358901909],
                [2, 1, 1, -44.83263019846355, -13.422258161051742, -0.004608695563609776],
                [1, 2, 2, -4.342606921772448, 4.829840332286762, -0.8696248777205423],
                [2, 1, 2, 1.3884720693911183, 0.30675032509703015, -0.3803835190493494],
                [0, 2, 0, 42.34652174703252, -1.3667245866292133, 1.489476103920553],
            ]],
            [0.0, [
                [1, 0, 0, 1.0, -1.3619545837223892, 0.36463278427406026],
                [2, 1, 0, -10.626973938205918, -0.8699357489607156, 0.4129312898142153],
                [0, 2, 0, 6.067460139311178, -3.0603231352476334, 2.802011219993987],
            ]],
        ],
    },
    "SS3_35": {
        "name": "SS3_35",
        "buffers": 3,
        "u_safe": 1.05,
        "lo": 0.004,
        "error": 0.083091071092388,
        "bf16_verified": True,
        "squarings": [3, 5],
        "flops": "(6α + 8)n³",
        "table": [
            [0.0, [
                [1, 0, 0, 1.0, -1.0932562497193754, 0.0],
                [2, 1, 1, 39.96168457119626, 14.266031068292616, 0.9999999999999999],
                [0, 2, 0, 12.502107468117961, -0.11270651203942335, 0.17461401290128586],
            ]],
            [0.0, [
                [1, 0, 0, 1.0, -1.094787000847377, 0.006046824763652966],
                [2, 1, 1, 32.52847526518725, 11.903537182257438, 0.926832022503135],
                [0, 2, 0, 1.1443577301804355, -0.4951256590911481, -0.14435773018043516],
                [1, 0, 1, 1.2879235989542381, -0.6561721992471682, -0.28395584065060986],
                [0, 1, 0, 15.770356077625026, 6.036080816655014, 3.5066207509983927],
            ]],
        ],
    },
}

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

# Each threadgroup owns one upper tile and writes both halves from the same value.
_METAL_TILE_INDEX = """
    const uint n = x_shape[0], k = x_shape[1];
    const uint tile = threadgroup_position_in_grid.x;
    uint col = uint((sqrt(8.0f * float(tile) + 1.0f) - 1.0f) * 0.5f);
    if (col * (col + 1) / 2 > tile) --col;
    if ((col + 1) * (col + 2) / 2 <= tile) ++col;
    const uint row = tile - col * (col + 1) / 2;
"""

# Metal 4 reads bounded tensor slices in native precision and holds the result
# in an FP32 cooperative tensor through the affine epilogue.
_METAL_TENSOR_SOURCE = _METAL_TILE_INDEX + """
    using namespace mpp::tensor_ops;
    using dims = dextents<int, 2>;
    // MPP requires a mutable tensor type even for its read-only operands.
    tensor<device T, dims, tensor_inline> matrix(const_cast<device T*>(x), dims(k, n));
    auto left = matrix.slice(0, row * B);
    auto right = matrix.slice(0, col * B);
    constexpr auto desc = matmul2d_descriptor(B, B, int(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto acc = op.template get_destination_cooperative_tensor<decltype(left), decltype(right), float>();
    op.run(left, right, acc);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto xy = acc.get_multidimensional_index(i);
        const uint r = row * B + xy[1], c = col * B + xy[0];
        if (r < n && c < n && r <= c) {
            const float product = acc[i];
"""

# Earlier Metal implementations use the 32x32 SIMD-group kernel.
_METAL_SIMDGROUP_SOURCE = _METAL_TILE_INDEX + """
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
            const float product = result[t];
"""

_METAL_MIRROR = """
            out[r * n + c] = value;
            if (r != c) out[c * n + r] = value;
        }
    }
"""
_METAL_AFFINE = """
            const T value = T(coeff[0] * product + coeff[1] * float(addend[r * n + c])
                              + (r == c ? coeff[2] : 0.0f));
"""


_METAL_MAX_N = 65535
_METAL_FAST_DTYPES = ("float16", "bfloat16")


def _launch_metal(kernel, x, *extra, tile_size):
    import mlx.core as mx
    n = x.shape[0]
    tiles = (n + tile_size - 1) // tile_size
    return kernel(inputs=[x, *extra], template=[("T", x.dtype), ("B", tile_size)],
                  grid=(128 * (tiles * (tiles + 1) // 2), 1, 1),
                  threadgroup=(128, 1, 1), output_shapes=[(n, n)],
                  output_dtypes=[x.dtype], stream=mx.gpu)[0]


@lru_cache(maxsize=None)
def _metal_kernel(dtype, fused, tiny=False):
    """Probe the Metal 4 kernel, then the portable SIMD-group implementation."""
    import mlx.core as mx
    name = "symprod" if fused else "syrk"
    epilogue = _METAL_AFFINE if fused else "const T value = T(product);"
    for tile_size, header, body in (
        (64, "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>", _METAL_TENSOR_SOURCE),
        (32, "#include <metal_simdgroup_matrix>", _METAL_SIMDGROUP_SOURCE),
    ):
        # MLX binds inputs with fewer than eight elements in constant memory.
        # The SIMD-group shader can read either address space.
        if tiny and tile_size == 64:
            continue
        suffix = f"tiny_{tile_size}" if tiny else str(tile_size)
        try:
            kernel = mx.fast.metal_kernel(
                name=f"polar_{name}_{suffix}", input_names=["x", "addend", "coeff"] if fused else ["x"],
                output_names=["out"], ensure_row_contiguous=True,
                header=header + "\nusing namespace metal;", source=body + epilogue + _METAL_MIRROR)
            launch = partial(_launch_metal, kernel, tile_size=tile_size)
            m = 1 if tiny else tile_size
            probe = mx.ones((m, m), dtype=dtype)
            extra = (probe, mx.ones((3,))) if fused else ()
            out = launch(probe, *extra)
            mx.eval(out)
            actual = as_numpy(out).astype(np.float32)
            expected = np.full((m, m), float(m), dtype=np.float32)
            if fused:
                expected = expected + 1.0 + np.eye(m, dtype=np.float32)
            if actual.shape != (m, m) or not np.array_equal(actual, expected):
                raise RuntimeError(f"Metal {name} tile={tile_size} probe numerical mismatch")
            return launch
        except (RuntimeError, ValueError, AttributeError) as exc:
            if tile_size == 32:
                warnings.warn(f"Metal {'symprod' if fused else 'SYRK'} unavailable for {dtype}, "
                              f"using dense matmul ({exc})", RuntimeWarning, stacklevel=2)
    return None


def metal_syrk(x):
    """Row Gram on MLX GPU. Use native matmul if the shader cannot compile."""
    if x.ndim != 2 or min(x.shape) == 0:
        raise ValueError("Expected a nonempty 2-D Metal input")
    kernel = _metal_kernel(x.dtype, False, x.size < 8) if x.shape[0] <= _METAL_MAX_N else None
    return kernel(x) if kernel is not None else x @ x.T


@lru_cache(maxsize=32)
def _metal_coefficients(a, b, c):
    import mlx.core as mx
    return mx.array([a, b, c], dtype=mx.float32)


def metal_symprod(x, addend, a, b, c):
    """Fuse a*(x @ x.T) + b*addend + c*I for a symmetric addend.

    The Metal epilogue reads only the upper triangle (r <= c) of addend and
    mirrors it, so callers must pass a symmetric addend. Metal rounds once at
    the mirrored store, whereas the dense fallback rounds the Gram first.
    """
    return _metal_symprod_vec(x, addend, (a, b, c))


def _metal_symprod_vec(x, addend, coeff):
    """Accept cached static coefficients or a lazy three-element FP32 MLX array."""
    import mlx.core as mx
    if x.ndim != 2 or min(x.shape) == 0 or addend.shape != (x.shape[0], x.shape[0]):
        raise ValueError("Expected a nonempty matrix and a square addend matching its row count")
    kernel = (_metal_kernel(x.dtype, True, x.size < 8 or addend.size < 8)
              if x.shape[0] <= _METAL_MAX_N else None)
    if kernel is not None:
        if isinstance(coeff, tuple):
            coeff = _metal_coefficients(*coeff)
        return kernel(x, addend, coeff)
    a, b, c = coeff
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
        self.dtype = dtype
        self.metal = use_metal_syrk and module == "mlx" and dtype in _METAL_FAST_DTYPES
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


def _array_module(x):
    module = type(x).__module__.split(".")[0]
    if module == "mlx":
        import mlx.core as mx
        return mx
    if module == "torch":
        import torch
        return torch
    return np


def _oriented(matrix):
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("Expected a nonempty 2-D real matrix")
    x = _float32(matrix)
    return x.T if x.shape[0] > x.shape[1] else x


def _safe_norm(x, xp):
    """Scale the FP32 sum of squares by an exact, representable power of two."""
    amax = xp.max(xp.abs(x))
    safe_amax = xp.where((amax > 0) & xp.isfinite(amax), amax, 1.0)
    exponent = xp.clip(-xp.floor(xp.log2(safe_amax)), -126, 126)
    inv_scale = 2.0 ** exponent
    scaled = x * inv_scale
    return xp.sqrt((scaled * scaled).sum()) / inv_scale


def _normalized(x, safety=1.0):
    x = _oriented(x)
    xp = _array_module(x)
    norm = _safe_norm(x, xp)
    valid = xp.isfinite(norm)
    return xp.where(valid, x, 0.0) / (safety * xp.where(valid, norm, 1.0) + 1e-7)


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
    for residual, steps in table:
        x = _switch_segment(x, residual, steps, work)
    return _restore(x, matrix)


def _switch_segment(x, residual, steps, work):
    identity = None if work.metal else _identity(x)
    buffers = {0: work.gram(x)}
    for dst, src, add, a, b, c in steps:
        operand, addend = buffers[src], buffers[add]
        if work.metal:
            buffers[dst] = metal_symprod(operand, addend, a, b, c)
        else:
            buffers[dst] = work.cast(a * _float32(work.gram(operand)) + b * _float32(addend)
                                     + c * identity)
    update = work.mm(buffers[dst], x)
    return work.cast(update + residual * x) if residual else update


_SLACK = 1.0 + 2.0**-12
_ROUNDING = {"float16": (2.0**-11, 2.0**-25), "bfloat16": (2.0**-8, 0.0),
             "float32": (2.0**-24, 2.0**-150)}


def _upper(norm_r, norm_s, n, k, dtype, prescale, alpha, xp):
    u, e = _ROUNDING[dtype]
    slack = _SLACK / max(0.5, 1.0 - 1e-8 * k)
    trace = prescale**2 * (1.0 + u)**2 * slack
    schatten = xp.sqrt(((1.0 + u) * norm_s * slack + n * e) / alpha)
    upper = schatten + u * norm_r * slack + n * e
    return xp.where((norm_s > 0) & (upper < trace), upper, trace)


def _dense_scaled_gram(x, alpha, work, xp):
    # Apply alpha to the accumulator. Casting an unscaled square can overflow
    # FP16 even when the scaled square is small (e.g. a coordinate spike).
    if xp is np:
        return work.cast(alpha * (x @ x.T))
    zero = xp.zeros_like(x[:1, :1])
    return xp.addmm(zero, x, x.T, alpha=alpha, beta=0.0)


def _schatten_first_segment(x, rows, kappa, alpha, r, square, work, xp):
    b0, c0 = rows[0][4:]
    factor = {0: 1.0 / kappa, 1: alpha / (kappa * kappa)}
    identity = _identity(x) if c0 or not work.metal else None
    row0 = _float32(square) + (alpha * b0 / kappa) * _float32(r)
    if c0:
        row0 = row0 + (c0 * factor[1]) * identity
    buffers = {0: r, 1: work.cast(row0)}
    one = xp.ones_like(kappa)
    for j, (dst, src, add, a, b, c) in enumerate(rows[1:], 1):
        t = xp.sqrt(kappa) if j == len(rows) - 1 else one
        fa, fb, fc = a * t / (factor[src] * factor[src]), b * t / factor[add], c * t
        operand, addend = buffers[src], buffers[add]
        if work.metal:
            buffers[dst] = _metal_symprod_vec(operand, addend, xp.stack([fa, fb, fc]))
        elif work.dtype == "bfloat16":
            # The coherent rank-one gate requires one BF16 rounding after the
            # affine sum. Rounding the dense square first can exceed the cap.
            wide = _float32(operand)
            buffers[dst] = work.cast(fa * (wide @ wide.T) + fb * _float32(addend) + fc * identity)
        else:
            if j == 1:
                # Buffer 1 stores alpha/kappa² times the polynomial. Undo its
                # power-of-two scale before narrowing the dense square.
                scale = 2.0 ** (-xp.floor(xp.log2(factor[1])))
                operand = work.cast(_float32(operand) * scale)
                fa = fa / (scale * scale)
            buffers[dst] = work.cast(fa * _float32(work.gram(operand))
                                     + fb * _float32(addend) + fc * identity)
        factor[dst] = t
    return work.mm(buffers[dst], x)


def schatten_switch(matrix, variant="SS3_54", dtype="float16", use_metal_syrk=False,
                    allow_unsafe_bf16=False):
    """Two-segment Schatten-8 circuit. All data-dependent scalars stay on device."""
    if variant not in _CIRCUIT_SPECS:
        raise ValueError(f"Unknown Schatten Switch variant {variant!r}")
    spec = _CIRCUIT_SPECS[variant]
    if dtype == "bfloat16" and not spec["bf16_verified"] and not allow_unsafe_bf16:
        raise ValueError(f"{variant} is not verified for bfloat16")
    work = _Work(matrix, dtype, use_metal_syrk)
    x = _oriented(matrix)
    xp = _array_module(x)
    n, k = x.shape
    prescale = 64.0 if n >= 1024 else 1.0
    alpha = 2.0**13 / prescale**4
    norm = _safe_norm(x, xp)
    valid = (norm > 1e-12) & xp.isfinite(norm)
    x = work.cast((xp.where(valid, x, 0.0) / xp.where(valid, norm, 1.0)) * prescale)
    r = work.gram(x)
    square = (_metal_symprod_vec(r, r, _metal_coefficients(alpha, 0.0, 0.0)) if work.metal
              else _dense_scaled_gram(r, alpha, work, xp))
    upper = _upper(_safe_norm(_float32(r), xp), _safe_norm(_float32(square), xp),
                   n, k, dtype, prescale, alpha, xp)
    kappa = 1.0 / (spec["u_safe"]**2 * upper)
    (_, rows), (residual, steps) = spec["table"]
    x = _schatten_first_segment(x, rows, kappa, alpha, r, square, work, xp)
    x = _switch_segment(x, residual, steps, work)
    return _restore(xp.where(valid, x, 0.0), matrix)


METHODS = {"jordan_ns5": jordan_ns5, "polar_express": polar_express, "gram_ns": gram_ns,
           "PS2_55": polar_switch, "PS3_55": partial(polar_switch, variant="PS3_55")}


SCHATTEN_METHODS = {name: partial(schatten_switch, variant=name) for name in _CIRCUIT_SPECS}
ALL_METHODS = {**METHODS, **SCHATTEN_METHODS}


def _block_orders(rng, count, names=None):
    """Randomized Latin squares balance every method across all time slots."""
    orders = []
    while len(orders) < count:
        base = list(METHODS if names is None else names)
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


def thermal_snapshot():
    snapshot = {"timestamp": datetime.now(timezone.utc).isoformat()}
    if platform.system() == "Darwin":
        for key, cmd in (("therm", ["pmset", "-g", "therm"]), ("batt", ["pmset", "-g", "batt"]),
                         ("clamshell", ["ioreg", "-r", "-k", "AppleClamshellState", "-d", "4"])):
            output = subprocess.check_output(cmd, text=True)
            if key == "clamshell":
                output = "\n".join(line.strip() for line in output.splitlines()
                                   if '"AppleClamshellState" =' in line)
            snapshot[key] = output.strip()
    return snapshot


def _time_blocks(calls, backend, rng, warmups, repeats):
    for block in _block_orders(rng, warmups, calls):
        for name in block:
            synchronize(calls[name](), backend)
    orders = _block_orders(rng, repeats, calls)
    samples = {name: [] for name in calls}
    walls = {name: [] for name in calls}
    outputs = {}
    for block in orders:
        for name in block:
            wall = time.time()
            start = time.perf_counter_ns()
            out = calls[name]()
            synchronize(out, backend)
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
            walls[name].append(time.time() - wall)
            outputs[name] = out
    return samples, walls, outputs, orders


def _sample_stats(samples, walls):
    p25, median, p75 = np.percentile(samples, [25, 50, 75]).tolist()
    return dict(samples_ms=samples, sample_wall_s=walls, median_ms=median, latency_ms=median,
                iqr_ms=p75-p25, p25_ms=p25, p75_ms=p75, min_ms=min(samples),
                sleep_detected=any(w > 0.002 * s + 0.5 for w, s in zip(walls, samples)))


def _measure(x, reference, backend, dtype, requested, rng, warmups, repeats, methods, ab_metal):
    paired = ab_metal and backend == "mlx" and dtype in _METAL_FAST_DTYPES
    arms = {}
    skipped = []
    for name, fn in methods.items():
        for metal in ((False, True) if paired else (requested,)):
            arm = "metal" if metal else "dense"
            label = f"{name}__{arm}" if paired else name
            if dtype == "bfloat16" and name in _CIRCUIT_SPECS and not _CIRCUIT_SPECS[name]["bf16_verified"]:
                skipped.append(dict(method=label, base_method=name, reason="bf16_verified=False"))
                continue
            arms[label] = name, metal, partial(fn, x, dtype=dtype, use_metal_syrk=metal)
    if not arms:
        return [], [], skipped
    syrk_fn = symprod_fn = None
    if ((requested or paired) and backend == "mlx" and dtype in _METAL_FAST_DTYPES
            and min(x.shape) <= _METAL_MAX_N):
        import mlx.core as mx
        work_dtype = getattr(mx, dtype)
        syrk_fn = _metal_kernel(work_dtype, False, x.size < 8)
        symprod_fn = _metal_kernel(work_dtype, True, min(x.shape) ** 2 < 8)
    calls = {label: run for label, (_, _, run) in arms.items()}
    samples, walls, outputs, orders = _time_blocks(calls, backend, rng, warmups, repeats)
    rows = []
    for label, (name, metal, _) in arms.items():
        syrk = metal and syrk_fn is not None
        fused = metal and name.startswith(("PS", "SS")) and symprod_fn is not None
        row = dict(method=label, shape=list(x.shape), backend=backend, dtype=dtype,
                   use_metal_syrk=syrk, use_metal_symprod=fused,
                   metal_syrk_tile=syrk_fn.keywords["tile_size"] if syrk else None,
                   metal_symprod_tile=symprod_fn.keywords["tile_size"] if fused else None,
                   output_dtype=str(outputs[label].dtype), **_sample_stats(samples[label], walls[label]),
                   **_quality(outputs[label], reference))
        if paired:
            row.update(base_method=name, arm="metal" if metal else "dense")
        rows.append(row)
    return rows, orders, skipped


def _paired_summary(rows):
    dense = {(row["seed"], tuple(row["shape"]), row["dtype"], row["base_method"]): row["samples_ms"]
             for row in rows if row.get("arm") == "dense"}
    groups = {}
    for row in rows:
        if row.get("arm") == "metal":
            key = tuple(row["shape"]), row["dtype"], row["base_method"]
            times = dense[(row["seed"],) + key]
            groups.setdefault(key, []).extend(d / m for d, m in zip(times, row["samples_ms"], strict=True))
    summary = []
    for (shape, dtype, name), ratios in groups.items():
        p25, median, p75 = np.percentile(ratios, [25, 50, 75]).tolist()
        summary.append(dict(shape=list(shape), dtype=dtype, method=name, median=median, p25=p25,
                            p75=p75, iqr=p75-p25, min=min(ratios), max=max(ratios), n_blocks=len(ratios)))
    return summary


def git_provenance():
    cwd = Path(__file__).resolve().parent
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, text=True)
        return dict(git_sha=sha, git_dirty=bool(dirty))
    except (OSError, subprocess.CalledProcessError):
        return dict(git_sha=None, git_dirty=None)


def benchmark(*, backend="auto", quick=False, warmups=5, repeats=20, seed=0, seeds=None,
              reference_dtype="float64", dtypes=("float32",), shapes=None, use_metal_syrk=False,
              methods=None, ab_metal=False):
    if warmups < 0 or repeats < 1 or reference_dtype not in ("float64", "float32"):
        raise ValueError("Expected nonnegative warmups, positive repeats and a float64 or float32 SVD")
    if isinstance(dtypes, str):
        dtypes = dtypes.split(",")
    if not dtypes or any(dtype not in DTYPES for dtype in dtypes):
        raise ValueError(f"Expected working dtypes from {DTYPES}")
    names = methods.split(",") if isinstance(methods, str) else methods
    if names is not None and (not names or len(set(names)) != len(names)
                              or any(name not in ALL_METHODS for name in names)):
        raise ValueError(f"Expected a nonempty subset of {list(ALL_METHODS)}")
    selected = METHODS if names is None else {name: ALL_METHODS[name] for name in names}
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
    provenance = git_provenance()
    thermal_before = thermal_snapshot()
    rows, orders, skipped = [], [], []
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
                case = dict(shape=list(shape), dtype=dtype, seed=input_seed)
                before = thermal_snapshot()
                measured, blocks, omissions = _measure(x, reference, backend, dtype, use_metal_syrk,
                                                       order_rng, warmups, repeats, selected, ab_metal)
                orders.append(dict(**case, blocks=blocks, thermal_before=before, thermal_after=thermal_snapshot()))
                rows.extend(dict(row, seed=input_seed) for row in measured)
                skipped.extend(dict(**case, **entry) for entry in omissions)
    library = {"cpu": "numpy", "mps": "torch", "mlx": "mlx"}[backend]
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    source = Path(__file__).resolve()
    return {"backend": backend, "library": library, "library_version": version(library),
            "thread_environment": {key: os.environ[key] for key in thread_vars if key in os.environ},
            "hardware": hardware_metadata(), "quick": quick, "seed": seeds[0], "seeds": seeds,
            "created_at": timestamp, "timestamp": timestamp, **provenance,
            "thermal_before": thermal_before, "thermal_after": thermal_snapshot(),
            "sleep_detected": any(row["sleep_detected"] for row in rows),
            "ab_metal": ab_metal, "methods": list(selected), "skipped_methods": skipped,
            "paired_ab_summary": _paired_summary(rows),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "dtype": dtypes[0] if len(dtypes) == 1 else list(dtypes), "dtypes": list(dtypes),
            "input_dtype": "float32", "use_metal_syrk_requested": use_metal_syrk,
            "use_metal_syrk": any(r["use_metal_syrk"] for r in rows),
            "use_metal_symprod": any(r["use_metal_symprod"] for r in rows),
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
    parser.add_argument("--ab-metal", action="store_true", help="Interleave dense and Metal arms on MLX")
    parser.add_argument("--methods", help="Comma-separated method names")
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
                     use_metal_syrk=args.use_metal_syrk, methods=args.methods, ab_metal=args.ab_metal)
    data["command"] = [sys.executable, *sys.argv]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f'{data["backend"]}: wrote {len(data["results"])} measurements to {args.output} '
          f'({data["divergences"]} diverged)')
