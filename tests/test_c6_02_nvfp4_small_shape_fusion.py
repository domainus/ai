"""Checks for Cycle 6 C6-02: the small-shape NVFP4 fast path and its benchmark driver.

Kernel tests compare ``nvfp4_small_shape_blackwell`` bit for bit with the eager reference
(``nvfp4_inference_blackwell.quantize_activation_ref``) and with the unchanged C5-03 module: operands,
layer outputs, the direct cuBLASLt GEMM, the ablation settings, the public launcher contract, CUDA graphs
and the absence of host synchronizations. Cross-mix tests measure the fused kernel against
``rotated_qat.CrossInput``. Driver tests cover the CLI and writer guards, the validator (on a shakedown run,
with tampering probes) and the committed canonical result, whose SHA-256 is pinned here.
"""

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CUDA = torch.cuda.is_available()
FP4 = CUDA and hasattr(F, "scaled_mm") and torch.cuda.get_device_capability()[0] >= 10
needs_fp4 = pytest.mark.skipif(not FP4, reason="needs native NVFP4 scaled_mm (sm_100+)")

if CUDA:
    from research.quant_ptq import bench_c5_03_nvfp4_120m_fusion as c503
    from research.quant_ptq import bench_c6_02_nvfp4_small_shape_fusion as b
    from research.quant_ptq import nvfp4_inference_blackwell as nb
    from research.quant_ptq import nvfp4_small_shape_blackwell as fs
    from research.quant_ptq import rotated_qat as rq

CANONICAL_RESULT = ROOT / "results" / "cycle6_c6_02_nvfp4_small_shape_fusion.json"
CANONICAL_SHA256 = "PENDING_CANONICAL_RUN"
ROTS = ("none", "hadamard_b16", "adaptive_butterfly_b16")
M_SIZES = (1, 37, 128, 129, 256, 512)
K_SIZES = (64, 80, 512, 768, 1408, 2048)


def _angles(K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand(K // 16, 4, 8, generator=g, device="cuda") * 2 * math.pi


def _args(rotation, K, seed=0):
    angles = _angles(K, seed) if rotation == "adaptive_butterfly_b16" else None
    return angles, (nb.butterfly_tables(angles) if angles is not None else None)


def _same(a, r):
    return c503._same_operands(a, r)


def _bits(a, b_):
    return c503._bitwise_equal(a, b_)


# ------------------------------------------------------------------ pinned inputs
@needs_fp4
def test_pinned_inputs_are_unchanged():
    """The C5-03 module (pinned by the C5-03 and C6-01 validators) and both earlier results are untouched."""
    assert b.pinned_inputs() == b.PINNED_INPUTS
    assert set(b.PINNED_INPUTS) == {"research/quant_ptq/nvfp4_inference_blackwell.py", b.C5_03_RESULT, b.C6_01_RESULT}
    assert nb.OPTIONS == {"pdl": True, "native_cvt": True} and nb.PASS1_PROGRAMS == 256


# ------------------------------------------------------------------ bit-exact quantizer
@needs_fp4
@pytest.mark.parametrize("rotation", ROTS)
@pytest.mark.parametrize("K", K_SIZES)
@pytest.mark.parametrize("M", M_SIZES)
def test_quantizer_matches_reference_bit_for_bit(M, K, rotation):
    g = torch.Generator(device="cuda").manual_seed(M * 7919 + K)
    x = (torch.randn(M, K, generator=g, device="cuda") * 2).to(torch.bfloat16)
    x[:, :16] *= 30
    angles, tables = _args(rotation, K)
    ref = nb.quantize_activation_ref(x, rotation, angles)
    assert _same(fs.quantize_activation(x, rotation, tables), ref)
    static = (ref.global_scale * 0.5).clone()  # some blocks saturate at E4M3 448 and E2M1 6
    assert _same(fs.quantize_activation(x, rotation, tables, static), nb.quantize_activation_ref(x, rotation, angles, static))


@needs_fp4
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("rotation", ROTS)
def test_dtypes_strides_and_unaligned_inputs(dtype, rotation):
    base = torch.randn(300, 1056, device="cuda", dtype=dtype)
    for x in (base[::2, 256:768], base.t().contiguous().t()[:, :512], base[1:130, ::2], base[:, 8:520],
              base[:, 4:516], base[3:40, 2:514]):
        angles, tables = _args(rotation, x.shape[1])
        assert _same(fs.quantize_activation(x, rotation, tables), nb.quantize_activation_ref(x, rotation, angles))
    # The load width follows the alignment of the groups: 128-bit, 64-bit or one load per input.
    wide = dtype == torch.float32
    assert fs._vec_ok(base[:, :512]) == 2
    assert fs._vec_ok(base[:, 4:516]) == (2 if wide else 1) and fs._vec_ok(base[:, 2:514]) == (1 if wide else 0)
    assert fs._vec_ok(base.t().contiguous().t()) == 0


@needs_fp4
@pytest.mark.parametrize("rotation", ROTS)
def test_special_values_follow_the_reference(rotation):
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(64, 512, device="cuda")
        x[0, :8] = torch.tensor([0.0, -0.0, 1e-45, -1e-45, 1e-38, float("inf"), float("-inf"), float("nan")])
        x[1, :16] = torch.logspace(-40, 0, 16, device="cuda")
        x[2, :32] = 0.0
        x[2, 3] = -0.0
        angles, tables = _args(rotation, 512)
        x = x.to(dtype)
        assert _same(fs.quantize_activation(x, rotation, tables), nb.quantize_activation_ref(x, rotation, angles))
        for gs in (1e-6, 0.01, 1e6):
            static = torch.tensor(gs, device="cuda")
            assert _same(fs.quantize_activation(x, rotation, tables, static),
                         nb.quantize_activation_ref(x, rotation, angles, static))
    for value in (0.0, float("nan"), float("inf")):
        x = torch.full((40, 256), value, device="cuda")
        angles, tables = _args(rotation, 256)
        assert _same(fs.quantize_activation(x, rotation, tables), nb.quantize_activation_ref(x, rotation, angles))


@needs_fp4
@pytest.mark.parametrize("rotation", ROTS)
@pytest.mark.parametrize("M,K", [(16, 4096), (33, 1408), (1000, 512), (1024, 1408), (257, 2048), (7, 8192)])
def test_benchmark_shapes_are_bit_exact(M, K, rotation):
    g = torch.Generator(device="cuda").manual_seed(M + K)
    x = (torch.randn(M, K, generator=g, device="cuda") * 3).to(torch.bfloat16)
    angles, tables = _args(rotation, K)
    ref = nb.quantize_activation_ref(x, rotation, angles)
    assert _same(fs.quantize_activation(x, rotation, tables), ref)
    assert _same(fs.quantize_activation(x, rotation, tables, ref.global_scale * 0.5),
                 nb.quantize_activation_ref(x, rotation, angles, ref.global_scale * 0.5))


@needs_fp4
def test_large_inputs_are_delegated_to_the_c5_03_quantizer():
    M, K = 2048, 8192
    assert fs.delegated(M, K) and not fs.delegated(1024, 4096) and M * K > fs.DELEGATE_ABOVE
    assert fs.quantizer_plan(M, K, "none")["delegated_to"] == "nvfp4_inference_blackwell"
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    assert _same(fs.quantize_activation(x, "hadamard_b16"), nb.quantize_activation(x, "hadamard_b16"))
    with b.options(delegate_large=False):  # the C6-02 kernels themselves are exact there too
        assert not fs.delegated(M, K)
        assert _same(fs.quantize_activation(x, "hadamard_b16"), nb.quantize_activation_ref(x, "hadamard_b16"))


# ------------------------------------------------------------------ public launchers
@needs_fp4
@pytest.mark.parametrize("rotation", ROTS)
def test_launchers_fill_dirty_buffers_and_padding(rotation):
    for M, K in ((37, 80), (129, 768), (1024, 1408)):
        x = torch.randn(M, K, device="cuda")
        angles, tables = _args(rotation, K)
        pr, pc = nb.scale_layout(M, K)
        packed = torch.full((M, K // 2), 0xAB, dtype=torch.uint8, device="cuda")
        scale = torch.full((pr * pc,), 0xCD, dtype=torch.uint8, device="cuda")
        ws = torch.full((fs.MAX_PARTIALS,), float("nan"), device="cuda")
        gs = torch.empty((), device="cuda")
        n = fs.launch_amax(x, ws, rotation, tables)
        assert 1 <= n <= fs.PASS1_PROGRAMS <= fs.MAX_PARTIALS and n == fs._n_partials(M, K, rotation)
        assert ws[n:].isnan().all()  # only workspace[:n] is written
        fs.launch_quant(x, packed, scale, gs, rotation=rotation, tables=tables, workspace=ws, n_part=n)
        ref = nb.quantize_activation_ref(x, rotation, angles)
        assert _same(nb.NVFP4Activation(packed.view(torch.float4_e2m1fn_x2), scale.view(torch.float8_e4m3fn), gs), ref)
        static = torch.tensor(0.01, device="cuda")
        fs.launch_quant(x, packed, scale, static, rotation=rotation, tables=tables)
        assert _same(nb.NVFP4Activation(packed.view(torch.float4_e2m1fn_x2), scale.view(torch.float8_e4m3fn), static),
                     nb.quantize_activation_ref(x, rotation, angles, static))


@needs_fp4
def test_launch_quant_enforces_the_exact_partial_count_and_buffers():
    M, K = 129, 768
    x = torch.randn(M, K, device="cuda")
    pr, pc = nb.scale_layout(M, K)
    good = {"packed": torch.empty((M, K // 2), dtype=torch.uint8, device="cuda"),
            "scale": torch.empty(pr * pc, dtype=torch.uint8, device="cuda"),
            "global_scale": torch.empty((), device="cuda")}
    ws = torch.empty(fs.MAX_PARTIALS, device="cuda")
    n = fs.launch_amax(x, ws, "hadamard_b16")
    for bad in (-1, 0, n - 1, n + 1, fs.MAX_PARTIALS + 1):
        with pytest.raises(ValueError, match="n_part"):
            fs.launch_quant(x, **good, rotation="hadamard_b16", workspace=ws, n_part=bad)
    with pytest.raises(ValueError, match="n_part"):
        fs.launch_quant(x, **good, rotation="hadamard_b16", workspace=ws, n_part=float(n))
    with pytest.raises(ValueError, match="static"):
        fs.launch_quant(x, **good, rotation="hadamard_b16", n_part=n)
    for key, value, match in (("packed", torch.empty((M, K // 2 + 1), dtype=torch.uint8, device="cuda"), "shape"),
                              ("scale", torch.empty(pr * pc + 1, dtype=torch.uint8, device="cuda"), "elements"),
                              ("global_scale", torch.empty(2, device="cuda"), "elements"),
                              ("packed", good["packed"].float(), "dtype"),
                              ("scale", good["scale"].cpu(), "must be on")):
        with pytest.raises(ValueError, match=match):
            fs.launch_quant(x, **{**good, key: value}, rotation="hadamard_b16", workspace=ws, n_part=n)
    with pytest.raises(ValueError, match="at least"):
        fs.launch_amax(x, torch.empty(n - 1, device="cuda"), "hadamard_b16")
    with pytest.raises(ValueError, match="contiguous"):
        fs.launch_amax(x, torch.empty(2 * fs.MAX_PARTIALS, device="cuda")[::2], "hadamard_b16")
    with pytest.raises(ValueError):
        fs.quantize_activation(torch.randn(4, 40, device="cuda"))
    with pytest.raises(ValueError):
        fs.quantize_activation(torch.randn(4, 64, device="cuda"), "adaptive_butterfly_b16")
    with pytest.raises(ValueError):
        fs.quantize_activation(torch.randn(4, 64, device="cuda"), global_scale=torch.tensor(1.0, device="cuda").double())


# ------------------------------------------------------------------ layer and GEMM
def _layers(W, K, static=None):
    out = {}
    for rotation in ROTS:
        angles, _ = _args(rotation, K, seed=1)
        layer = nb.NVFP4Linear.from_weight(W, rotation, angles, act_global_scale=static)
        out[rotation] = (layer, fs.FastNVFP4Linear.from_layer(layer))
    return out


@needs_fp4
@pytest.mark.parametrize("M,N,K", [(1, 256, 64), (37, 512, 768), (129, 384, 1408), (1024, 512, 1408), (16, 1024, 4096)])
@pytest.mark.parametrize("static", (False, True))
def test_fast_layer_output_equals_the_c5_03_layer_bit_for_bit(M, N, K, static):
    g = torch.Generator(device="cuda").manual_seed(M + N + K)
    x = torch.randn(M, K, generator=g, device="cuda").to(torch.bfloat16)
    W = torch.randn(N, K, generator=g, device="cuda") * 0.02
    for rotation, (layer, fast) in _layers(W, K, torch.tensor(0.004, device="cuda") if static else None).items():
        assert _bits(fast(x), layer(x)), rotation
        assert _bits(fast(x.reshape(1, M, K)), layer(x).reshape(1, M, N))
        assert _bits(fast(x.float()), layer(x.float())) and _bits(fast(x.half()), layer(x.half()))
        assert fast(x[:0]).shape == (0, N)
        assert set(fast.state_dict()) == set(layer.state_dict())


@needs_fp4
def test_fast_layer_ablation_settings_are_exact_and_from_weight_works():
    M, N, K = 129, 384, 1408
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    W = torch.randn(N, K, device="cuda") * 0.02
    angles = _angles(K, 2)
    for rotation in ROTS:
        a = angles if rotation == "adaptive_butterfly_b16" else None
        layer = nb.NVFP4Linear.from_weight(W, rotation, a)
        fast = fs.FastNVFP4Linear.from_weight(W, rotation, a)
        want = layer(x)
        assert _bits(fast(x), want)
        for opts in ({"pdl": False}, {"vector_io": False}, {"lt_gemm": False}, {"stash": True},
                     {"delegate_large": False}):
            with b.options(**opts):
                assert _bits(fast(x), want), (rotation, opts)
    assert fs.packed_tables(*nb.butterfly_tables(angles)).shape == (16, K // 16, 4)
    cos, sin = nb.butterfly_tables(angles)
    p = fs.packed_tables(cos, sin)
    assert torch.equal(p[3, 5], torch.stack((cos.reshape(-1, 32)[5, 6], sin.reshape(-1, 32)[5, 6],
                                             cos.reshape(-1, 32)[5, 7], sin.reshape(-1, 32)[5, 7])))


@needs_fp4
@pytest.mark.parametrize("out_dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("M,N,K", [(1, 256, 64), (37, 96, 64), (1024, 512, 1408), (16, 4096, 4096), (1024, 2048, 768)])
def test_direct_cublaslt_gemm_equals_the_public_api_bit_for_bit(M, N, K, out_dtype):
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    W = torch.randn(N, K, device="cuda") * 0.02
    layer = nb.NVFP4Linear.from_weight(W, "hadamard_b16")
    a = nb.quantize_activation(x, "hadamard_b16")
    alpha = (a.global_scale * layer.w_global).reshape(())
    plan = fs.lt_plan(M, N, K, out_dtype, x.device)
    assert plan.direct and plan.workspace_bytes == 0 and plan.record()["algo"]["splitk_num"] in (0, 1)
    out = torch.full((M, N), float("nan"), dtype=out_dtype, device="cuda")  # beta = 0 must not read C
    plan(a, layer.w_packed, layer.w_scale, alpha, out)
    assert _bits(out, nb.nvfp4_scaled_mm_public(a, layer.w_packed, layer.w_scale, layer.w_global, out_dtype))
    assert _bits(fs.nvfp4_scaled_mm_lt(a, layer.w_packed, layer.w_scale, alpha, out_dtype), out)


@needs_fp4
def test_fused_alpha_is_the_product_of_the_two_tensor_scales():
    x = torch.randn(64, 512, device="cuda").to(torch.bfloat16)
    W = torch.randn(256, 512, device="cuda") * 0.02
    layer = fs.FastNVFP4Linear.from_weight(W, "hadamard_b16")
    for static in (None, torch.tensor(0.0123, device="cuda")):
        q, alpha = fs._quantize(x, 64, 512, "hadamard_b16", None, static, layer.w_global)
        assert alpha.view(torch.int32) == (q.global_scale * layer.w_global).reshape(()).view(torch.int32)


@needs_fp4
@pytest.mark.parametrize("rotation", ROTS)
@pytest.mark.parametrize("static", (False, True))
def test_forward_is_sync_free_and_graph_replays_are_bitwise(rotation, static):
    W = torch.randn(384, 768, device="cuda") * 0.02
    angles, _ = _args(rotation, 768)
    layer = nb.NVFP4Linear.from_weight(W, rotation, angles,
                                       act_global_scale=torch.tensor(0.004, device="cuda") if static else None)
    fast = fs.FastNVFP4Linear.from_layer(layer)
    x_static = torch.randn(37, 768, device="cuda", dtype=torch.bfloat16)
    fast(x_static)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fast(x_static)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    out = {}
    replay = b.e2e.capture(lambda: out.__setitem__("y", fast(x_static)))
    g = torch.Generator(device="cuda").manual_seed(5)
    for scale in (1.0, 1e-3, 50.0, 1.0):
        x_static.copy_(torch.randn(37, 768, generator=g, device="cuda") * scale)
        replay()
        assert _bits(out["y"], layer(x_static))


@needs_fp4
def test_first_launch_or_plan_during_capture_is_refused():
    x = torch.randn(3, 4096 + 96, device="cuda")  # a shape no other test compiled
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="Warm up"):
        with torch.cuda.graph(graph):
            fs.quantize_activation(x)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="Warm up"):
        with torch.cuda.graph(graph):
            fs.lt_plan(5, 160, 4096 + 96, torch.bfloat16, x.device)


# ------------------------------------------------------------------ fused cross-block mix
@needs_fp4
@pytest.mark.parametrize("family", ("hadamard_b64", "kron_hadamard_full", "cross_full_butterfly_b16"))
@pytest.mark.parametrize("K", (256, 768, 1408))
def test_fused_cross_mix_matches_crossinput(family, K):
    cross = rq.make_cross(rq.FAMILIES[family][0], K)
    rot = rq.kernel_rotation(family)
    W = torch.randn(256, K, device="cuda") * 0.02
    layer = fs.FastNVFP4Linear.from_layer(nb.NVFP4Linear.from_weight(cross.forward(W), rot, _angles(K)
                                                                     if rot == "adaptive_butterfly_b16" else None))
    fused, eager = fs.FastCrossInput(layer, cross), rq.CrossInput(layer, cross)
    x = torch.randn(8, 128, K, device="cuda").to(torch.bfloat16)
    got, want = fused.mixed(x), cross.forward(x.float()).reshape(-1, K)
    # The two-half FMA order of cuBLAS gemmSN: bit-exact at this batch (1024 rows).
    assert _bits(got, want)
    assert _bits(fused(x), eager(x))
    for small in (x[0, :37], x[0, :1].float()):  # other cuBLAS kernels: near-0 parity only
        diff = (fused.mixed(small) - cross.forward(small.float()).reshape(-1, K)).abs().max().item()
        assert diff <= 1e-6 * max(1.0, small.float().abs().max().item())


@needs_fp4
def test_fused_cross_mix_chunks_many_blocks_and_rejects_bad_inputs():
    K = 5632  # 44 blocks of 128: three programs of output blocks
    cross = rq.make_cross("full", K)
    layer = fs.FastNVFP4Linear.from_layer(nb.NVFP4Linear.from_weight(torch.randn(64, K, device="cuda") * 0.02,
                                                                     "hadamard_b16"))
    fused = fs.FastCrossInput(layer, cross)
    x = torch.randn(64, K, device="cuda").to(torch.bfloat16)
    want = cross.forward(x.float())
    rel = ((fused.mixed(x) - want).norm() / want.norm()).item()
    assert rel < 1e-6 and fused.h.device == x.device and fused.mix.shape == (44, 44)
    assert fused(x[:0]).shape == (0, 64)
    with pytest.raises(ValueError):
        fs.FastCrossInput(layer, rq.CrossMix(K, 16))
    with pytest.raises(ValueError):
        fused.mixed(x.cpu())


# ------------------------------------------------------------------ driver helpers and guards
@needs_fp4
def test_kernel_kinds_and_profile_rules():
    assert b.kernel_kind(fs._amax_kernel.fn.__name__) == "amax" and b.kernel_kind(nb._quant_kernel.fn.__name__) == "quant"
    assert b.kernel_kind("void at::native::vectorized_elementwise_kernel<4, BinaryFunctor<float, float, float, "
                         "binary_internal::MulFunctor<float> >") == "scale_product"
    assert b.kernel_kind("cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1") == "fp4_gemm"
    assert b.kernel_kind("nvjet_tst_128x64_bf16") == "gemm" and b.kernel_kind(fs._cross_mix_kernel.fn.__name__) == "cross_mix"
    assert b.PART_A_METHODS[0] == "bf16_cublas" and len(b.PART_A_METHODS) == 11
    assert b.CANONICAL["part_a"]["shapes"] == c503.PART_A_SHAPES and len(b.CANONICAL["part_a"]["shapes"]) == 8


@needs_fp4
def test_cli_guards(tmp_path):
    with pytest.raises(SystemExit):
        b.parse_args(["--synthetic-shakedown"])  # needs an output
    with pytest.raises(SystemExit, match="reserved"):
        b.parse_args(["--synthetic-shakedown", "--output", str(tmp_path / "cycle6_c6_02_x.json")])
    with pytest.raises(SystemExit, match="no options"):
        b.parse_args(["--output", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        b.parse_args(["--validate", "x.json", "--synthetic-shakedown"])
    with pytest.raises(SystemExit):
        b.parse_args(["--synthetic", "--output", str(tmp_path / "y.json")])  # no abbreviations
    args = b.parse_args(["--synthetic-shakedown", "--output", str(tmp_path / "y.json")])
    assert args.synthetic_shakedown and args.output == tmp_path / "y.json"
    assert b.parse_args([]).output == b.CANONICAL_OUTPUT
    assert b.reserved(Path("results/cycle6_c6_02_nvfp4_small_shape_fusion.json")) and not b.reserved(tmp_path / "y.json")
    assert b.rejected_path(b.CANONICAL_OUTPUT).name == "rejected_c6_02_nvfp4_small_shape_fusion.json.rejected.json"
    assert not b.reserved(b.rejected_path(b.CANONICAL_OUTPUT))


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    monkeypatch.setattr(b, "ROOT", root)
    monkeypatch.setattr(c503, "ROOT", root)
    monkeypatch.setattr(b, "CANONICAL_OUTPUT", root / b.RESULT_REPO_PATH)
    return root


def _leftovers(directory):
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".tmp"))


@needs_fp4
def test_atomic_writer_guards(sandbox, tmp_path):
    payload = {"a": [1, 2.5, None]}
    assert b.write_json_atomic(b.CANONICAL_OUTPUT, payload, canonical_dest=True) == payload
    assert json.loads(b.CANONICAL_OUTPUT.read_text()) == payload and _leftovers(sandbox / "results") == []
    with pytest.raises(RuntimeError, match="written once"):
        b.require_canonical_host(torch.device("cuda"), {"git_sha": "a" * 40, "git_dirty": False}, b.CANONICAL_OUTPUT)
    with pytest.raises(RuntimeError, match="clean"):
        b.require_canonical_host(torch.device("cuda"), {"git_sha": "a" * 40, "git_dirty": True}, tmp_path / "z.json")
    b.CANONICAL_OUTPUT.unlink()
    target = tmp_path / "outside.json"
    b.CANONICAL_OUTPUT.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        b.write_json_atomic(b.CANONICAL_OUTPUT, payload, canonical_dest=True)
    assert not target.exists()
    b.CANONICAL_OUTPUT.unlink()
    for bad in (sandbox / "results" / ".." / "results" / b.CANONICAL_OUTPUT.name,
                Path("results/../results") / b.CANONICAL_OUTPUT.name):
        with pytest.raises(ValueError, match="parent traversal"):
            b.write_json_atomic(bad, payload, canonical_dest=True)
    with pytest.raises(ValueError, match="canonical result goes to"):
        b.write_json_atomic(sandbox / "results" / "cycle6_c6_02_alias.json", payload, canonical_dest=True)
    with pytest.raises(ValueError, match="canonical result goes to"):
        b.verify_canonical_output_path(tmp_path / "plain.json")
    with pytest.raises(ValueError, match="outside"):
        b.write_json_atomic(tmp_path / "cycle6_c6_02_x.json", payload)  # a reserved name outside ROOT
    (sandbox / "results").rename(sandbox / "real")
    (sandbox / "results").symlink_to(sandbox / "real", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        b.write_json_atomic(b.CANONICAL_OUTPUT, payload, canonical_dest=True)
    assert _leftovers(sandbox / "real") == []


# ------------------------------------------------------------------ validator on a shakedown run
@pytest.fixture(scope="module")
def shakedown(tmp_path_factory):
    out = tmp_path_factory.mktemp("c6_02") / "shakedown.json"
    args = b.parse_args(["--synthetic-shakedown", "--silent", "--output", str(out)])
    payload = b.run(args)
    b.write_json_atomic(out, payload)
    return payload


@needs_fp4
def test_shakedown_is_valid_and_complete(shakedown):
    s = shakedown
    assert b.validate_results(copy.deepcopy(s)) == s and s["canonical"] is False and s["shakedown"] is True
    assert s["kernel_parity"]["bitwise_failures"] == [] and s["kernel_parity"]["cases"] > 100
    assert all(all(v.values()) for r in s["part_a"]["shapes"] for v in [r["checks"]])
    assert s["summary"]["part_b"]["bitwise_equal_public"][0] == s["summary"]["part_b"]["bitwise_equal_public"][1]
    assert s["summary"]["part_b"]["torch_scaled_mm_v2_has_aten_mul"] is True
    assert "scale_product" not in s["summary"]["part_b"]["cublaslt_kernel_kinds"]
    assert s["pareto"] == {}  # the tiny model has no C6-01 cell
    with pytest.raises(ValueError, match="canonical"):
        b.validate_results(copy.deepcopy(s), canonical=True)


def _tamper(payload, fn):
    bad = copy.deepcopy(payload)
    fn(bad)
    return bad


TAMPERS = {
    "extra key": lambda p: p.__setitem__("extra", 1),
    "missing key": lambda p: p.pop("caveats"),
    "schema": lambda p: p.__setitem__("schema_version", "cycle6_c6_02.v0"),
    "protocol": lambda p: p["protocol"]["timing"].__setitem__("iters", 4),
    "pinned input": lambda p: p["pinned_inputs"].__setitem__(b.C6_01_RESULT, "0" * 64),
    "code sha": lambda p: p["meta"].__setitem__("code_sha256", "0" * 64),
    "run config hash": lambda p: p.__setitem__("run_config_hash", "0" * 16),
    "command": lambda p: p["meta"]["command"].__setitem__(1, "other.py"),
    "parity failure": lambda p: p["kernel_parity"]["bitwise_failures"].append(["x"]),
    "parity count": lambda p: p["kernel_parity"].__setitem__("cases", p["kernel_parity"]["cases"] + 1),
    "sync": lambda p: next(iter(p["sync_check"].values())).__setitem__("d2h_copies", 1),
    "graph replay": lambda p: next(iter(p["sync_check"].values())).__setitem__("graph_replays_bitwise_exact", 3),
    "part a check": lambda p: p["part_a"]["shapes"][0]["checks"].__setitem__("none_c6_02_output_equals_c5_03_bitwise",
                                                                             False),
    "part a raw sample": lambda p: p["part_a"]["shapes"][0]["cuda_graph"]["c6_02_none"]["gpu"]["event_samples_ms"][0]
    .__setitem__(0, 1e3),
    "part a median": lambda p: p["part_a"]["shapes"][0]["cuda_graph"]["c6_02_none"]["gpu"].__setitem__("median_ms", 1e-6),
    "part a summary": lambda p: p["part_a"]["shapes"][0]["summary"]["cuda_graph"]["variants"]["none"]
    .__setitem__("c6_02_beats_bf16", True),
    "part a orders": lambda p: p["part_a"]["shapes"][0]["cuda_graph_williams_orders"].reverse(),
    "part a shapes": lambda p: p["part_a"]["shapes"].pop(),
    "part a delegation": lambda p: p["part_a"]["shapes"][0].__setitem__("delegated_to_c5_03", True),
    "profile mul": lambda p: next(iter(p["part_a"]["shapes"][0]["cuda_graph_kernel_profile"]["c6_02_none"].values()))
    .__setitem__("kind", "scale_product"),
    "profile impl": lambda p: p["part_a"]["shapes"][0]["cuda_graph_kernel_profile"].__setitem__(
        "c6_02_none", p["part_a"]["shapes"][0]["cuda_graph_kernel_profile"]["c5_03_none"]),
    "ablation check": lambda p: p["part_a"]["ablation"][0]["checks"].__setitem__("none_c6_02_no_pdl_equals_c5_03_bitwise",
                                                                                 False),
    "ablation summary": lambda p: p["part_a"]["ablation"][0]["summary"]["c6_02_none"].__setitem__("ms", 1.0),
    "part b exact": lambda p: p["part_b"]["exactness"][0].__setitem__("bitwise_equal_public", False),
    "part b trace": lambda p: p["part_b"]["traces"]["cublaslt_fused_alpha"]["kernel_kinds"].append("scale_product"),
    "part b heuristic": lambda p: p["part_b"]["heuristic"]["selected"].__setitem__("tile_id", -1),
    "part c check": lambda p: p["part_c"]["models"]["tiny"]["checks"].__setitem__(
        "had_b16_c6_02_logits_equal_c5_03_bitwise", False),
    "part c parity": lambda p: p["part_c"]["models"]["tiny"]["fused_cross_mix_parity"]["hadamard_b64"].__setitem__(
        "mixed_mismatches", -1),
    "part c summary": lambda p: p["part_c"]["models"]["tiny"]["summary"]["wide"]["hadamard_b64"].__setitem__(
        "c6_02_fused_ms", 1.0),
    "part c kernels": lambda p: p["part_c"]["models"]["tiny"]["kernel_kinds_per_forward"]["c6_02_had_b16"].__setitem__(
        "scale_product", 99.0),
    "summary": lambda p: p["summary"].__setitem__("part_a_cuda_graph_cells_beating_bf16", [8, 8]),
    "caveats": lambda p: p["caveats"].append("x"),
    "pareto": lambda p: p.__setitem__("pareto", {"50m": {}}),
    "canonical flag": lambda p: p.__setitem__("canonical", True),
}


@needs_fp4
@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_validator_rejects_tampering_of_a_shakedown(shakedown, name):
    with pytest.raises(ValueError):
        b.validate_results(_tamper(shakedown, TAMPERS[name]))


# ------------------------------------------------------------------ the committed canonical result
@pytest.fixture(scope="module")
def canonical():
    data = CANONICAL_RESULT.read_bytes()
    assert hashlib.sha256(data).hexdigest() == CANONICAL_SHA256
    return json.loads(data)


@needs_fp4
def test_committed_canonical_result_is_pinned_and_valid(canonical):
    payload = b.validate_results(copy.deepcopy(canonical), canonical=True, output=b.CANONICAL_OUTPUT)
    assert payload["canonical"] is True and payload["meta"]["git_dirty"] is False
    producer = payload["meta"]["git_sha"]
    assert b._matches_current_summary_code(producer)
    assert payload["meta"]["code_modules"] == list(b._code_modules_at_commit(producer))
    assert payload["kernel_parity"]["cases"] >= b.CANONICAL_PARITY_MIN_CASES
    assert payload["kernel_parity"]["bitwise_failures"] == []
    assert [r["mnk"] for r in payload["summary"]["part_a"]] == c503.PART_A_SHAPES
    assert set(payload["pareto"]) == {"50m", "120m"}


@needs_fp4
@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_validator_rejects_tampering_of_the_canonical_result(canonical, name):
    if name == "part c check" or name.startswith("part c"):  # the canonical models are 50m, 120m, w2048_6l
        fn = {"part c check": lambda p: p["part_c"]["models"]["50m"]["checks"].__setitem__(
            "had_b16_c6_02_logits_equal_c5_03_bitwise", False),
              "part c parity": lambda p: p["part_c"]["models"]["120m"]["fused_cross_mix_parity"]["hadamard_b64"]
              .__setitem__("mixed_mismatches", -1),
              "part c summary": lambda p: p["part_c"]["models"]["50m"]["summary"]["wide"]["hadamard_b64"].__setitem__(
                  "c6_02_fused_ms", 1.0),
              "part c kernels": lambda p: p["part_c"]["models"]["w2048_6l"]["kernel_kinds_per_forward"]["c6_02_had_b16"]
              .__setitem__("scale_product", 99.0)}[name]
    elif name == "canonical flag":
        fn = lambda p: p.__setitem__("canonical", False)  # noqa: E731
    else:
        fn = TAMPERS[name]
    with pytest.raises(ValueError):
        b.validate_results(_tamper(canonical, fn), canonical=True)


@needs_fp4
def test_canonical_provenance_tampering_is_rejected(canonical):
    for fn in (lambda p: p["meta"].__setitem__("git_dirty", True),
               lambda p: p["meta"].__setitem__("git_sha", "0" * 40),
               lambda p: p["meta"].__setitem__("argv", ["--synthetic-shakedown"]),
               lambda p: p["meta"]["environment"].__setitem__("capability", [9, 0]),
               lambda p: p["meta"]["code_modules"].pop(),
               lambda p: p["kernel_parity"].__setitem__("cases", 100) or p["kernel_parity"]["case_kinds"].__setitem__(
                   "quantize_activation", 100 - sum(v for k, v in p["kernel_parity"]["case_kinds"].items()
                                                    if k != "quantize_activation"))):
        with pytest.raises(ValueError):
            b.validate_results(_tamper(canonical, fn), canonical=True)
    with pytest.raises(ValueError):
        b.validate_results(copy.deepcopy(canonical), canonical=True, output=ROOT / "results" / "cycle6_c6_02_copy.json")
