import json
import subprocess
import sys
from pathlib import Path

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
