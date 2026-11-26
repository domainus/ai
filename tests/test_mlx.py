import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from research.polar_opt import kernels_mlx as kernels

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("method", kernels.METHODS, ids=list(kernels.METHODS))
def test_native_kernel_against_svd(method):
    backend = kernels.detect_backend()
    for shape in ((7, 19), (19, 7), (8, 8)):
        matrix = np.random.default_rng(874).standard_normal(shape, dtype=np.float32)
        u, _, vh = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
        value = kernels.native_array(matrix, backend)
        result = kernels.METHODS[method](value)
        kernels.synchronize(result, backend)
        out = kernels.as_numpy(result)
        assert result.shape == matrix.shape and type(value) is type(result)
        assert np.isfinite(out).all()
        # Jordan's five nonconvergent quintics intentionally leave more distortion.
        tolerance = {"jordan_ns5": .36, "polar_express": .15, "gram_ns": .17,
                     "PS2_55": .14, "PS3_55": .12}[method]
        assert np.linalg.norm(out - u @ vh, ord=2) < tolerance
    zero = kernels.native_array(np.zeros((5, 3), np.float32), backend)
    np.testing.assert_array_equal(kernels.as_numpy(kernels.METHODS[method](zero)), np.zeros((5, 3)))


def test_quick_cli(tmp_path):
    output = tmp_path / "results/run.json"
    subprocess.run([sys.executable, str(ROOT / "research/polar_opt/kernels_mlx.py"),
                    "--quick", "--output", str(output)], check=True)
    record = json.loads(output.read_text())
    assert record["backend"] == kernels.detect_backend()
    assert record["hardware"]["machine"] and record["reference"] == "numpy_float64_svd"
    rows = record["results"]
    assert len(rows) == len(kernels.QUICK_SHAPES) * len(kernels.METHODS)
    assert {(tuple(r["shape"]), r["method"]) for r in rows} == {
        (shape, name) for shape in kernels.QUICK_SHAPES for name in kernels.METHODS}
    for row in rows:
        assert row["finite"] and 0 <= row["polar_error"] < .4
        assert len(row["samples_ms"]) == 2
        assert all(t > 0 and np.isfinite(t) for t in row["samples_ms"])
        assert row["latency_ms"] == pytest.approx(np.median(row["samples_ms"]))


def _has_mlx():
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except (ImportError, OSError):
        return False


BACKENDS = ["cpu"] + (["mlx"] if _has_mlx() else [])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", kernels.DTYPES)
@pytest.mark.parametrize("name", kernels.METHODS)
def test_work_precision_quality(backend, dtype, name):
    method = kernels.METHODS[name]
    # Five-step approximations have different accuracy even in full precision.
    fro_limit, orth_limit = {"jordan_ns5": (.34, .48), "polar_express": (.18, .30),
                             "gram_ns": (.19, .32), "PS2_55": (.18, .30),
                             "PS3_55": (.16, .26)}[name]
    for shape in ((7, 19), (19, 7), (8, 8)):
        matrix = np.random.default_rng(874).standard_normal(shape, dtype=np.float32)
        u, _, vt = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
        x = kernels.native_array(matrix, backend)
        for use_metal in ([False, True] if backend == "mlx" else [False]):
            out = method(x, dtype=dtype, use_metal_syrk=use_metal)
            assert out.dtype == x.dtype and out.shape == x.shape
            q = kernels._quality(out, u @ vt)
            assert q["finite"] and not q["diverged"]
            assert q["polar_error"] < fro_limit
            assert q["orthogonality_err"] < orth_limit
    # Restore the actual input dtype, including non-FP32 inputs.
    if backend == "mlx":
        import mlx.core as mx
        half = x.astype(mx.float16)
    else:
        half = x.astype(np.float16)
    assert method(half, dtype=dtype).dtype == half.dtype


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", kernels.DTYPES)
@pytest.mark.parametrize("name", kernels.METHODS)
def test_zero_and_rank_deficient(backend, dtype, name):
    matrix = np.zeros((5, 3), np.float32)
    for magnitude in (0., 1., 1e-9):
        matrix[2, 1] = magnitude
        x = kernels.native_array(matrix, backend)
        y = kernels.as_numpy(kernels.METHODS[name](x, dtype=dtype, use_metal_syrk=True))
        assert np.isfinite(y).all()
        np.testing.assert_array_equal(y[matrix == 0], 0)
        if magnitude:
            assert 0. < y[2, 1] < 2.


@pytest.mark.parametrize("dtype,mantissa", [("float16", 10), ("bfloat16", 7)])
def test_cpu_round_to_nearest_even(dtype, mantissa):
    ulp = 2. ** -mantissa
    values = np.array([1., 1. + .5 * ulp, 1. + 1.5 * ulp, -1. - .5 * ulp,
                       np.inf, -np.inf, np.nan], np.float32)
    expected = [1., 1., 1. + 2 * ulp, -1., np.inf, -np.inf, np.nan]
    np.testing.assert_array_equal(kernels._cpu_round(values, dtype), expected)


@pytest.mark.parametrize("dtype", kernels.DTYPES)
def test_norm_in_fp32_before_work_cast(dtype):
    # Early FP16 conversion overflows. Early BF16 conversion changes the ratio.
    x = np.array([[0.5242, 0.8301, 0.1307], [0.7128, 0.6302, 0.9205]], np.float32) * 1e6
    work = kernels._Work(x, dtype, False)
    expected = kernels._cpu_round(x / (np.linalg.norm(x) + 1e-7), dtype)
    np.testing.assert_array_equal(work.cast(kernels._normalized(x)), expected)
    with pytest.raises(ValueError, match="nonempty"):
        kernels.polar_switch(np.empty((0, 1), dtype=np.float32), dtype=dtype)
    with pytest.raises(ValueError, match="working dtype"):
        kernels.polar_express(x, dtype="fp8")


def test_interleaved_benchmark(monkeypatch):
    observed = []
    for name, method in kernels.METHODS.copy().items():
        def run(x, name=name, method=method, **options):
            observed.append(name)
            return method(x, **options)
        monkeypatch.setitem(kernels.METHODS, name, run)
    repeats = 7
    data = kernels.benchmark(backend="cpu", shapes=[(7, 19)], seeds=[0, 2], dtypes=kernels.DTYPES,
                             repeats=repeats, warmups=2, use_metal_syrk=True)
    assert data["git_sha"] == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert datetime.fromisoformat(data["timestamp"]).utcoffset().total_seconds() == 0
    assert data["use_metal_syrk_requested"] and not data["use_metal_syrk"]
    assert data["divergences"] == 0
    assert len(data["results"]) == 2 * 3 * len(kernels.METHODS)
    assert len(data["method_order"]) == 6
    pos = 0
    for case in data["method_order"]:
        blocks = case["blocks"]
        assert len(blocks) == repeats
        assert all(set(block) == set(kernels.METHODS) for block in blocks)
        # A complete five-block cycle puts every method in every position.
        assert all(set(column) == set(kernels.METHODS) for column in zip(*blocks[:5]))
        pos += 2 * len(kernels.METHODS)
        for block in blocks:
            assert observed[pos:pos + len(block)] == block
            pos += len(block)
    assert pos == len(observed)
    for row in data["results"]:
        assert row["backend"] == "cpu" and row["dtype"] in kernels.DTYPES
        assert not row["use_metal_syrk"] and not row["use_metal_symprod"]
        assert row["metal_syrk_tile"] is None and row["metal_symprod_tile"] is None
        samples = row["samples_ms"]
        assert len(samples) == repeats
        p25, median, p75 = np.percentile(samples, [25, 50, 75])
        np.testing.assert_allclose([row["p25_ms"], row["median_ms"], row["p75_ms"], row["iqr_ms"],
                                    row["min_ms"]], [p25, median, p75, p75-p25, min(samples)])


def test_quality_detects_finite_and_nonfinite_divergence():
    reference = np.eye(3)
    for factor in (0., 1., 2., 2.5):
        row = kernels._quality(reference * factor, reference)
        assert row["diverged"] == (factor > 2.)
        assert row["sigma_max"] == pytest.approx(factor)
        assert row["orthogonality_err"] == pytest.approx(abs(factor ** 2 - 1.))
        assert row["relative_frobenius_error"] == pytest.approx(abs(factor - 1.))
    row = kernels._quality(np.full((3, 3), np.inf), reference)
    assert row["diverged"] and not row["finite"] and row["sigma_max"] is None


@pytest.mark.parametrize("tile_size", [32, 64])
def test_metal_tile_contract(tile_size):
    for n, k in ((1, 1), (33, 17), (67, 35), (2048, 2048), (4096, 4096)):
        core = SimpleNamespace(gpu=object())
        def kernel(**kwargs):
            tiles = (n + tile_size - 1) // tile_size
            assert kwargs["grid"] == (128 * tiles * (tiles + 1) // 2, 1, 1)
            assert kwargs["threadgroup"] == (128, 1, 1)
            assert kwargs["output_shapes"] == [(n, n)]
            assert dict(kwargs["template"]) == {"T": "T", "B": tile_size}
            return ["tile launch"]
        with patch.dict(sys.modules, {"mlx": SimpleNamespace(core=core), "mlx.core": core}):
            x = SimpleNamespace(shape=(n, k), dtype="T")
            assert kernels._launch_metal(kernel, x, tile_size=tile_size) == "tile launch"


def test_lazy_metal_compile_failure_falls_back_once(monkeypatch):
    # MLX may report a compile error at evaluation, after dispatch succeeded.
    probes = []
    def fail_evaluation(value):
        probes.append(value)
        raise RuntimeError("compiler unavailable")
    core = SimpleNamespace(fast=SimpleNamespace(metal_kernel=lambda **kwargs: object()),
                           ones=np.ones, eval=fail_evaluation, float32=np.float32, eye=np.eye)
    x = np.arange(15, dtype=np.float32).reshape(3, 5)
    gram = x @ x.T
    bias = np.eye(3, dtype=np.float32)
    monkeypatch.setattr(kernels, "_launch_metal", lambda *args, **kwargs: kwargs["tile_size"])
    kernels._metal_kernel.cache_clear()
    try:
        with patch.dict(sys.modules, {"mlx": SimpleNamespace(core=core), "mlx.core": core}):
            with pytest.warns(RuntimeWarning, match="SYRK unavailable"):
                np.testing.assert_array_equal(kernels.metal_syrk(x), gram)
            np.testing.assert_array_equal(kernels.metal_syrk(x), gram)
            with pytest.warns(RuntimeWarning, match="symprod unavailable"):
                np.testing.assert_array_equal(kernels.metal_symprod(x, bias, 1, 2, 3), gram + 5 * bias)
            assert probes == [64, 32, 64, 32]
    finally:
        kernels._metal_kernel.cache_clear()


@pytest.mark.skipif(not _has_mlx(), reason="Requires native Apple Metal")
@pytest.mark.parametrize("dtype", kernels.DTYPES)
def test_live_metal_syrk_and_fused_symprod(dtype):
    import mlx.core as mx
    fmt = getattr(mx, dtype)
    # Fail if these calls fall back. Dense matmul can also produce exact symmetry.
    for shape in ((1, 1), (32, 32), (33, 17), (17, 33), (67, 35), (35, 67)):
        tiny = int(np.prod(shape)) < 8
        for fused in (False, True):
            launch = kernels._metal_kernel(fmt, fused, tiny)
            assert launch is not None
            tile = launch.keywords["tile_size"]
            assert tile in (32, 64)
            if tiny:
                assert tile == 32
        # Transposed input also checks the custom kernel's row-contiguous copy.
        x = mx.array(np.random.default_rng(4).normal(size=shape[::-1]).astype(np.float32)).astype(fmt).T
        xf = kernels.as_numpy(x).astype(np.float64)
        reference = xf @ xf.T
        g = kernels.metal_syrk(x)
        eye = mx.eye(shape[0], dtype=fmt)
        fused = kernels.metal_symprod(x, eye, -.5, 2., -.25)
        for actual, target in ((g, reference), (fused, -.5 * reference + 1.75 * np.eye(shape[0]))):
            assert actual.dtype == fmt
            assert mx.max(mx.abs(actual - actual.T)).item() == 0
            tol = {"float32": 5e-5, "float16": 1e-3, "bfloat16": 5e-3}[dtype]
            np.testing.assert_allclose(kernels.as_numpy(actual), target, rtol=tol, atol=tol)

    # Repeated integer-valued rows give an exact, inexpensive FP64 oracle
    # for every output element, including rectangular K != N and bottom-right tiles.
    for n, k in ((2048, 2048), (2048, 4096), (4096, 4096)):
        basis = np.random.default_rng(n + k).integers(-1, 2, size=(19, k)).astype(np.float64)
        gram = mx.array((basis @ basis.T).astype(np.float32))
        indices = mx.arange(n) % len(basis)
        x = mx.array(basis.astype(np.float32)).astype(fmt)[indices]
        reference = gram[indices[:, None], indices[None, :]]
        eye = mx.eye(n)
        for actual, target in (
            (kernels.metal_syrk(x), reference),
            (kernels.metal_symprod(x, eye.astype(fmt), -.5, 2., -.25), -.5 * reference + 1.75 * eye),
        ):
            assert mx.max(mx.abs(actual - actual.T)).item() == 0
            assert mx.all(actual == target.astype(fmt)).item()


def test_fp32_stays_dense_under_use_metal_syrk():
    MlxArray = type("MlxArray", (), {"__module__": "mlx.core"})
    core = SimpleNamespace(float32="f32", float16="f16", bfloat16="bf16")
    with patch.dict(sys.modules, {"mlx": SimpleNamespace(core=core), "mlx.core": core}):
        x = MlxArray()
        assert kernels._Work(x, "float32", True).metal is False
        assert kernels._Work(x, "float16", True).metal is True
        assert kernels._Work(x, "bfloat16", True).metal is True
        assert kernels._Work(x, "float16", False).metal is False


def test_metal_probe_mismatch_falls_back_and_tiny_names_distinct(monkeypatch):
    built = []
    def make_kernel(**kwargs):
        built.append(kwargs["name"])
        return kwargs["name"]
    def fake_launch(kernel_name, x, *extra, tile_size):
        n = x.shape[0]
        if tile_size == 64:
            return np.zeros((n, n), dtype=np.float32)
        expected = np.full((n, n), float(n), dtype=np.float32)
        return expected + 1.0 + np.eye(n, dtype=np.float32) if extra else expected
    core = SimpleNamespace(fast=SimpleNamespace(metal_kernel=make_kernel),
                           ones=np.ones, eval=lambda v: None, float32=np.float32)
    monkeypatch.setattr(kernels, "_launch_metal", fake_launch)
    kernels._metal_kernel.cache_clear()
    try:
        with patch.dict(sys.modules, {"mlx": SimpleNamespace(core=core), "mlx.core": core}):
            for fused, base in ((False, "syrk"), (True, "symprod")):
                normal = kernels._metal_kernel("float16", fused, tiny=False)
                tiny = kernels._metal_kernel("float16", fused, tiny=True)
                assert normal is not None and tiny is not None
                assert normal.keywords["tile_size"] == 32
                assert tiny.keywords["tile_size"] == 32
        assert built == [
            "polar_syrk_64", "polar_syrk_32", "polar_syrk_tiny_32",
            "polar_symprod_64", "polar_symprod_32", "polar_symprod_tiny_32",
        ]
    finally:
        kernels._metal_kernel.cache_clear()


def test_metal_overflow_and_tiny_addend_guards(monkeypatch):
    seen = []
    monkeypatch.setattr(kernels, "_metal_kernel",
                        lambda dtype, fused, tiny=False: seen.append((dtype, fused, tiny)) or None)
    n_big = kernels._METAL_MAX_N + 1
    class FakeMat:
        ndim = 2
        shape = (n_big, 1)
        size = n_big
        dtype = np.float16
        @property
        def T(self):
            return self
        def __matmul__(self, other):
            return np.ones((2, 2), dtype=np.float16)
    class FakeAddend:
        shape = (n_big, n_big)
        def astype(self, _):
            return 0
    core = SimpleNamespace(float32=np.float32,
                           eye=lambda n: 0.0 if n > 16 else np.eye(n, dtype=np.float32))
    with patch.dict(sys.modules, {"mlx": SimpleNamespace(core=core), "mlx.core": core}):
        big = FakeMat()
        assert kernels.metal_syrk(big).shape == (2, 2)
        assert kernels.metal_symprod(big, FakeAddend(), 1.0, 2.0, 3.0).dtype == np.float16
        assert seen == []

    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    addend = np.eye(2, dtype=np.float32)
    with patch.dict(sys.modules, {"mlx": SimpleNamespace(core=core), "mlx.core": core}):
        out = kernels.metal_symprod(x, addend, 1.0, 2.0, 3.0)
    np.testing.assert_allclose(out, x @ x.T + 5.0 * addend)
    assert seen == [(np.dtype("float32"), True, True)]
