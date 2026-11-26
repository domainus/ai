"""Checks for the Cycle 3 T5-01 micro-rotated NVFP4 / MXFP4 / INT2 PTQ module.

Math tests run on the CPU. Tests marked ``needs_cuda`` check CUDA parity, the native FP4 GEMM
and the CUDA end-to-end pipeline.
"""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.linalg import hadamard

from research.quant_ptq import micro_rotated_ptq as q

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")
ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "cycle3_t5_01_nvfp4_ptq.json"
FMT = q.FORMATS


def e2m1_reference(x: torch.Tensor) -> torch.Tensor:
    """Independent E2M1 rounding: step 0.5 below 2, step 1 below 4, step 2 up to 6, ties to even."""
    a = x.abs().double()
    out = torch.where(a < 2, torch.round(a * 2) / 2, torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2))
    return (torch.sign(x) * out.clamp(max=6)).float()


def correlated_problem(rows=64, cols=256, tokens=4096, seed=0):
    X = q.correlated_inputs(tokens, cols, 1e3, seed, dtype=torch.float32)
    W = (torch.randn(1, rows, cols, generator=torch.Generator().manual_seed(seed + 1)) * 0.02)
    return W, (X.double().T @ X.double() / tokens).unsqueeze(0)


# ------------------------------------------------------------------ element codecs
def test_e2m1_codebook_is_exact():
    codes = torch.arange(16, dtype=torch.uint8)
    values = q.e2m1_decode(codes)
    assert values[:8].tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    assert values[8:].tolist() == [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    assert torch.equal(q.e2m1_codes(values), codes)  # every code point encodes to itself


def test_e2m1_ties_to_even_and_saturation():
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    assert q.e2m1_decode(q.e2m1_codes(ties)).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
    assert q.e2m1_decode(q.e2m1_codes(-ties)).tolist() == [-0.0, -1.0, -1.0, -2.0, -2.0, -4.0, -4.0]
    big = torch.tensor([6.0, 6.5, 7.0, 100.0, -7.0, -1e30])
    assert q.e2m1_decode(q.e2m1_codes(big)).tolist() == [6.0, 6.0, 6.0, 6.0, -6.0, -6.0]
    x = torch.randn(100_000, generator=torch.Generator().manual_seed(0)) * 3
    assert torch.equal(q.e2m1_decode(q.e2m1_codes(x)), e2m1_reference(x))


def test_float4_packing_roundtrip_and_nibble_order():
    codes = torch.randint(0, 16, (8, 64), generator=torch.Generator().manual_seed(1), dtype=torch.uint8)
    packed = q.pack_e2m1(codes)
    assert packed.dtype == torch.float4_e2m1fn_x2 and packed.shape == (8, 32)
    assert torch.equal(q.unpack_e2m1(packed), codes)
    raw = packed.view(torch.uint8)
    assert torch.equal(raw & 0xF, codes[:, ::2]) and torch.equal(raw >> 4, codes[:, 1::2])
    with pytest.raises(ValueError):
        q.pack_e2m1(torch.zeros(3, dtype=torch.uint8))


def _wide_range_weights(rows=64, cols=256, seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=g) * torch.logspace(-4, 1, rows).unsqueeze(1)


def test_nvfp4_e4m3_block_scales_and_global_scale():
    W = _wide_range_weights()
    enc = q.nvfp4_encode(W)
    block = enc["block_scale"].float()
    assert enc["block_scale"].dtype == torch.float8_e4m3fn
    assert block.max().item() == 448.0  # the largest block maps exactly to the E4M3 maximum
    assert (block <= 448).all() and (block >= 0).all() and torch.isfinite(block).all()
    assert math.isclose(enc["global_scale"].item(), W.abs().max().item() / (6 * 448), rel_tol=1e-7)
    assert torch.equal(q.to_e4m3(block), block)
    # decode(encode) is the RTN fake-quant path, and no element exceeds 6 * scale.
    assert torch.allclose(q.nvfp4_decode(enc), q.quantize_rtn(W, FMT["nvfp4_g16"]), rtol=1e-6, atol=0)
    scale = (block * enc["global_scale"]).repeat_interleave(16, -1)
    assert (q.nvfp4_decode(enc).abs() <= 6 * scale * (1 + 1e-6)).all()
    assert (q.quantize_rtn(torch.zeros(4, 32), FMT["nvfp4_g16"]) == 0).all()


def test_mxfp4_e8m0_power_of_two_scales():
    W = _wide_range_weights(seed=3)
    enc = q.mxfp4_encode(W)
    assert enc["block_scale"].dtype == torch.float8_e8m0fnu
    biased = enc["block_scale"].view(torch.uint8).int()
    exponent = biased - 127
    assert exponent.min() >= -127 and exponent.max() <= 127
    scale = enc["block_scale"].float()
    assert torch.equal(scale, torch.exp2(exponent.float()))
    amax = W.reshape(W.shape[0], -1, 32).abs().amax(-1)
    ratio = amax / scale  # OCP floor rule: amax / 2^(floor(log2 amax) - 2) lies in [4, 8)
    assert (ratio >= 4).all() and (ratio < 8).all()
    assert torch.equal(enc["values"], q.quantize_rtn(W, FMT["mxfp4_g32"]))
    tiny = q.mxfp4_exponent(torch.tensor([1e-45, 0.0, 3e38]))
    assert tiny.tolist() == [-127, -127, 125]


@pytest.mark.parametrize("name", ["int2_asym_g16", "int2_asym_g32"])
def test_int2_asymmetric_bounds(name):
    fmt = FMT[name]
    W = _wide_range_weights(seed=4)
    values, codes = q.quantize_rtn(W, fmt, return_codes=True)
    assert codes.min() >= 0 and codes.max() <= 3
    groups = W.reshape(W.shape[0], -1, fmt.group_size)
    vg = values.reshape(groups.shape)
    lo, hi = groups.amin(-1, keepdim=True), groups.amax(-1, keepdim=True)
    slack = 1e-3 * (hi - lo) + 1e-7
    assert (vg >= lo - slack).all() and (vg <= hi + slack).all()
    assert all(len(torch.unique(row)) <= 4 for row in vg.reshape(-1, fmt.group_size)[:200])
    scale, zero = q.group_params(groups, fmt)
    assert torch.equal(scale, q.to_fp16(scale)) and torch.equal(zero, q.to_fp16(zero))
    flat = torch.full((2, fmt.group_size), 0.125)
    assert torch.equal(q.quantize_rtn(flat, fmt), flat)


def test_effective_bits_per_weight():
    assert FMT["nvfp4_g16"].bits_per_weight() == 4.5
    assert FMT["nvfp4_g16"].bits_per_weight(512 * 512) == 4.5 + 32 / (512 * 512)
    assert FMT["mxfp4_g32"].bits_per_weight() == 4.25
    assert FMT["int2_asym_g16"].bits_per_weight() == 4.0
    assert FMT["int2_asym_g32"].bits_per_weight() == 3.0
    assert q.Butterfly(16, q.Butterfly.initial_angles((), 64, 16)).parameter_bits_per_input == 32.0


# ------------------------------------------------------------------ rotations
def _rotations(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"hadamard_b16": q.BlockHadamard(16), "hadamard_b64": q.BlockHadamard(64),
            "rand_hadamard_b64": q.SignHadamard(64, q.random_signs((n,), seed)),
            "butterfly_random": q.Butterfly(16, torch.rand(n // 16, 4, 8, generator=g, dtype=torch.float64) * 6.3)}


@pytest.mark.parametrize("n", [512, 1408])
def test_rotations_are_orthonormal(n):
    eye = torch.eye(n, dtype=torch.float64)
    for name, rot in _rotations(n).items():
        R = rot.matrix(n)
        assert torch.linalg.matrix_norm(R.T @ R - eye).item() <= 1e-6, name
        w = torch.randn(4, n, dtype=torch.float64)
        assert torch.allclose(rot.inverse(rot.forward(w)), w, atol=1e-12), name


def test_fwht_is_sylvester_hadamard_and_butterfly_starts_at_hadamard():
    for b in (16, 64):
        H = torch.from_numpy(hadamard(b)).double() / math.sqrt(b)
        x = torch.randn(3, 4 * b, dtype=torch.float64)
        expected = (x.reshape(3, 4, b) @ H.T).reshape(3, 4 * b)
        assert torch.allclose(q.fwht(x, b), expected, atol=1e-12)
    x = torch.randn(5, 128, dtype=torch.float64)
    start = q.Butterfly(16, q.Butterfly.initial_angles((), 128, 16, dtype=torch.float64))
    assert torch.allclose(start.forward(x), q.fwht(x, 16), atol=1e-12)


@pytest.mark.parametrize("n", [512, 1408])
def test_pre_quantization_forward_equivalence(n):
    g = torch.Generator().manual_seed(n)
    X = torch.randn(256, n, generator=g, dtype=torch.float64)
    W = torch.randn(96, n, generator=g, dtype=torch.float64)
    for dtype, bound in ((torch.float64, 1e-12), (torch.float32, 1e-5)):
        x, w = X.to(dtype), W.to(dtype)
        y = x @ w.T
        for name, rot in _rotations(n).items():
            # W X^T = (W R^T)(R X^T), written row-wise as (x R^T)(W R^T)^T.
            yr = rot.forward(x) @ rot.forward(w).T
            assert ((y - yr).norm() / y.norm()).item() < bound, (name, dtype)


def test_hessian_rotation_identity():
    n = 256
    X = torch.randn(1024, n, dtype=torch.float64)
    H = X.T @ X / X.shape[0]
    for name, rot in _rotations(n).items():
        R = rot.matrix(n)
        HR = rot.rotate_hessian(H)
        assert torch.allclose(HR, R @ H @ R.T, atol=1e-12), name
        XR = rot.forward(X)
        assert torch.allclose(HR, XR.T @ XR / X.shape[0], atol=1e-12), name


def test_batched_rotations_use_per_layer_parameters():
    W = torch.randn(3, 8, 64, dtype=torch.float64)
    rot, info = q.make_rotation("rand_hadamard_b64", W, seed=10)
    for i in range(3):
        single = q.SignHadamard(64, q.random_signs((64,), info["seeds"][i]))
        assert torch.allclose(rot.forward(W)[i], single.forward(W[i]), atol=1e-12)


def test_adaptive_butterfly_is_orthogonal_and_not_worse_than_hadamard():
    g = torch.Generator().manual_seed(5)
    W = torch.randn(2, 64, 128, generator=g) * 0.02
    W[:, :, 3] *= 20  # an outlier input channel
    for name in ("nvfp4_g16", "int2_asym_g16"):
        rot, info = q.fit_butterfly(W, FMT[name], steps=40, check_every=10)
        assert info["weight_sq_error_fitted"] <= info["weight_sq_error_hadamard_start"]
        for i in range(2):
            R = q.Butterfly(16, rot.angles[i].double()).matrix(128)
            assert torch.linalg.matrix_norm(R.T @ R - torch.eye(128, dtype=torch.float64)).item() <= 1e-6
        Z = rot.forward(W)
        had = q.fwht(W, 16)
        err = (q.quantize_rtn(Z, FMT[name]) - Z).pow(2).sum()
        assert err <= (q.quantize_rtn(had, FMT[name]) - had).pow(2).sum() * (1 + 1e-6)


# ------------------------------------------------------------------ GPTQ / LDLQ
def _gptq_reference(W, H, fmt, damp=0.01):
    """Unblocked OBQ-form GPTQ with explicit inverse-Hessian downdates (no Cholesky trick)."""
    W = W[0].double().clone()
    H = H[0].double().clone()
    H += damp * H.diagonal().mean() * torch.eye(H.shape[0], dtype=H.dtype)
    Hinv = torch.linalg.inv(H)
    Q = torch.zeros_like(W)
    gs = q.nvfp4_global_scale(W.float()) if fmt.kind == "nvfp4" else None
    for j in range(W.shape[1]):
        if j % fmt.group_size == 0:
            scale, zero = q.group_params(W[:, j:j + fmt.group_size].float(), fmt, gs)
        _, qj = q.quantize_with(W[:, j:j + 1].float(), scale, zero, fmt)
        Q[:, j:j + 1] = qj.double()
        err = (W[:, j:j + 1] - Q[:, j:j + 1]) / Hinv[j, j]
        W[:, j:] -= err * Hinv[j:j + 1, j:]
        Hinv = Hinv - Hinv[:, j:j + 1] @ Hinv[j:j + 1, :] / Hinv[j, j]
    return Q.float().unsqueeze(0)


@pytest.mark.parametrize("name", list(FMT))
def test_gptq_strictly_beats_rtn_on_correlated_inputs(name):
    fmt = FMT[name]
    W, H = correlated_problem()
    rtn = q.output_error(W, q.quantize_rtn(W, fmt), H).item()
    gptq = q.output_error(W, q.gptq_quantize(W, H, fmt)[0], H).item()
    assert gptq < 0.9 * rtn
    rot = q.BlockHadamard(64)
    rot_rtn = q.output_error(W, q.quantize_stack(W, H, fmt, rot, "rtn")[0], H).item()
    rot_gptq = q.output_error(W, q.quantize_stack(W, H, fmt, rot, "gptq")[0], H).item()
    assert rot_gptq < rot_rtn


@pytest.mark.parametrize("name", list(FMT))
def test_gptq_cholesky_form_matches_obq_reference(name):
    fmt = FMT[name]
    W, H = correlated_problem(rows=32, cols=128, tokens=2048, seed=7)
    blocked, _ = q.gptq_quantize(W, H, fmt, block_size=64)
    reference = _gptq_reference(W, H, fmt)
    # Different float paths may flip a rare tie; the output errors must agree closely.
    assert (blocked == reference).float().mean().item() > 0.995
    e1, e2 = q.output_error(W, blocked, H).item(), q.output_error(W, reference, H).item()
    assert math.isclose(e1, e2, rel_tol=0.02)
    unblocked, _ = q.gptq_quantize(W, H, fmt, block_size=fmt.group_size)
    assert (blocked == unblocked).float().mean().item() > 0.995


@pytest.mark.parametrize("name", list(FMT))
def test_gptq_with_identity_hessian_is_rtn(name):
    W = torch.randn(2, 16, 128, generator=torch.Generator().manual_seed(8)) * 0.05
    H = torch.eye(128, dtype=torch.float64).expand(2, 128, 128).clone()
    assert torch.equal(q.gptq_quantize(W, H, FMT[name])[0], q.quantize_rtn(W, FMT[name]))


def test_gptq_damping_fallback_and_dead_columns():
    g = torch.Generator().manual_seed(12)
    W = torch.randn(1, 8, 64, generator=g) * 0.1
    basis, _ = torch.linalg.qr(torch.randn(64, 64, generator=g, dtype=torch.float64))
    eig = torch.ones(64, dtype=torch.float64)
    eig[:4] = -0.2  # indefinite with a positive diagonal: 0.01 damping is not enough
    H = (basis * eig @ basis.T).unsqueeze(0)
    assert (H.diagonal(dim1=-2, dim2=-1) > 0).all()
    H[0, 3, :] = H[0, :, 3] = 0.0  # one input channel without energy
    Q, info = q.gptq_quantize(W, H, FMT["nvfp4_g16"])
    assert torch.isfinite(Q).all() and info["damping_lambda"][0] > 0.01 and info["dead_columns"] == 1


# ------------------------------------------------------------------ data hygiene
def _fake_data(tmp_path, sizes=(40_000, 6_000, 6_000)):
    """Shards whose token ranges identify their split: train < 1000 <= val < 2000 <= test."""
    rng = np.random.default_rng(0)
    splits = {}
    for (name, size), base in zip(zip(("train", "val", "test"), sizes), (0, 1000, 2000)):
        tokens = (rng.integers(0, 1000, size) + base).astype(np.uint16)
        tokens[0] = 50256
        tokens.tofile(tmp_path / f"{name}.bin")
        splits[name] = {"file": f"{name}.bin", "tokens": size, "sha256": f"fake-{name}"}
    manifest = {"source": {"name": "synthetic"}, "encoding": "gpt2", "dtype": "uint16", "splits": splits}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_calibration_uses_train_only(tmp_path):
    data = q.load_data(_fake_data(tmp_path), verify=False, expected=None)
    cal = q.calibration_windows(data, 32, 128, seed=3)
    tokens = cal["array"]
    assert cal["split"] == "train" and tokens.shape == (32, 128)
    assert ((tokens < 1000) | (tokens == 50256)).all()  # never a val (1000-1999) or test (2000+) token
    starts = np.array(cal["starts"])
    assert (np.diff(starts) >= 128).all() and starts.max() + 128 <= 40_000  # disjoint windows inside train
    assert cal["starts"] == q.calibration_windows(data, 32, 128, seed=3)["starts"]  # deterministic
    only_train = SimpleNamespace(splits={"train": data.splits["train"]})
    assert np.array_equal(q.calibration_windows(only_train, 32, 128, seed=3)["array"], tokens)


def test_real_shards_match_pinned_hashes():
    directory = q.default_data_dir()
    if not (directory / "manifest.json").exists():
        pytest.skip(f"real shards not found at {directory}")
    data = q.load_data(directory)  # recomputes every SHA-256 and compares with the pins
    assert {k: v["sha256"] for k, v in data.manifest["splits"].items()} == q.EXPECTED_SHA256


# ------------------------------------------------------------------ pipeline and artifact schema
def _run_quick(tmp_path, device):
    data_dir = _fake_data(tmp_path, sizes=(200_000, 20_000, 20_000))
    out = tmp_path / "quick.json"
    payload = q.main(["--quick", "--no-verify", "--silent", "--data-dir", str(data_dir), "--device", device,
                      "--output", str(out)])
    assert json.loads(out.read_text()) == json.loads(json.dumps(payload))
    return payload


@pytest.fixture(scope="module")
def quick_cpu(tmp_path_factory):
    return _run_quick(tmp_path_factory.mktemp("quick_cpu"), "cpu")


def _check_quick(payload):
    q.validate_results(payload)
    assert payload["rotation_checks"]["passed"] and payload["gptq_synthetic_check"]["passed"]
    for name in q.MODEL_SPECS:
        study = payload["models"][name]["study"]
        assert len(study["grid"]) == len(q.FORMATS) * len(q.METHODS)
        assert study["baseline_test"]["tokens"] == 64 * 64
        assert 0 < study["parameters"]["quantized_fraction"] < 1
        for row in study["grid"]:
            if row["rounding"] == "gptq":
                rtn = next(r for r in study["grid"] if r["format"] == row["format"]
                           and r["method"] == ("rtn" if row["rotation"] == "identity" else f"{row['rotation']}_rtn"))
                assert row["calibration_relative_output_error_mean"] < rtn["calibration_relative_output_error_mean"]


def test_quick_pipeline_cpu(quick_cpu):
    _check_quick(quick_cpu)
    assert quick_cpu["native_fp4_gemm_probe"]["status"].startswith("skipped")


def test_result_schema_is_strict(quick_cpu):
    bad = copy.deepcopy(quick_cpu)
    bad["models"]["adamw"]["study"]["grid"][0]["test_nll"] = float("nan")
    with pytest.raises(ValueError):
        q.validate_results(bad)
    leak = copy.deepcopy(quick_cpu)
    leak["data"]["calibration"]["split"] = "test"
    with pytest.raises(ValueError, match="train.bin"):
        q.validate_results(leak)
    short = copy.deepcopy(quick_cpu)
    short["models"]["muon_ss3_54"]["study"]["grid"][3]["test_tokens"] -= 1
    with pytest.raises(ValueError):
        q.validate_results(short)
    missing = copy.deepcopy(quick_cpu)
    del missing["meta"]["git_sha"]
    with pytest.raises(ValueError):
        q.validate_results(missing)


def test_committed_result_artifact():
    if not RESULT.exists():
        pytest.skip("result artifact not generated yet")
    payload = json.loads(RESULT.read_text())
    q.validate_results(payload)
    assert payload["quick"] is False and payload["meta"]["git_dirty"] is False
    assert payload["data"]["sha256_verified"] and payload["data"]["expected_sha256"] == q.EXPECTED_SHA256
    assert payload["data"]["calibration"]["tokens"] == 128 * 512
    for name in q.MODEL_SPECS:
        study = payload["models"][name]["study"]
        assert study["baseline_test"]["tokens"] == (2_000_081 - 1) // 512 * 512
        assert len(study["grid"]) == len(q.FORMATS) * len(q.METHODS)
        assert payload["models"][name]["training"]["steps"] == 320


# ------------------------------------------------------------------ CUDA
@needs_cuda
def test_cuda_matches_cpu_reference():
    W = _wide_range_weights(rows=128, cols=512, seed=9)
    for name, fmt in FMT.items():
        assert torch.equal(q.quantize_rtn(W.cuda(), fmt).cpu(), q.quantize_rtn(W, fmt)), name
    x = torch.randn(64, 512)
    for name, rot in _rotations(512).items():
        if isinstance(rot, q.SignHadamard):
            gpu = q.SignHadamard(64, rot.signs.squeeze(-2).cuda())
        elif isinstance(rot, q.Butterfly):
            gpu = q.Butterfly(16, rot.angles.float().cuda())
            rot = q.Butterfly(16, rot.angles.float())
        else:
            gpu = rot
        assert torch.allclose(gpu.forward(x.cuda()).cpu(), rot.forward(x), atol=1e-5), name
    Wg, H = correlated_problem(rows=32, cols=128, tokens=2048, seed=11)
    for name, fmt in FMT.items():
        cpu = q.gptq_quantize(Wg, H, fmt)[0]
        gpu = q.gptq_quantize(Wg.cuda(), H.cuda(), fmt)[0].cpu()
        # Blocked updates sum in a different order on the GPU; error feedback may then flip a rare
        # rounding, so compare values within FP32 rounding and the resulting output error.
        close = torch.isclose(cpu, gpu, rtol=1e-5, atol=1e-12).float().mean().item()
        assert close > 0.99, name
        assert math.isclose(q.output_error(Wg, cpu, H).item(), q.output_error(Wg, gpu, H).item(), rel_tol=1e-3)
    assert torch.equal(q.nvfp4_global_scale(W.cuda()).cpu(), q.nvfp4_global_scale(W))


@needs_cuda
def test_native_fp4_probe_sanity():
    probe = q.probe_native_fp4_gemm(torch.device("cuda"), shapes=((256, 512),), rounds=2, iterations=3)
    assert probe["pack_unpack_roundtrip"] and probe["float4_e2m1fn_x2_dtype"]
    assert "nvfp4" in probe and "mxfp4" in probe
    if probe["nvfp4"]["supported"]:
        assert probe["nvfp4"]["verified"], probe["nvfp4"]
        row = probe["latency"][0]
        assert row["nvfp4_relative_error_vs_fp32_dequant"] < 1e-2
        assert row["bf16_cublas"]["median_ms"] > 0 and row["nvfp4_scaled_mm"]["median_ms"] > 0
    else:
        assert "latency" not in probe and probe["nvfp4"]["error"]


@needs_cuda
def test_quick_pipeline_cuda(tmp_path):
    payload = _run_quick(tmp_path, "cuda")
    _check_quick(payload)
    assert payload["models"]["muon_ss3_54"]["training"]["orthogonalizer"].startswith("schatten_switch_blackwell")
