"""Checks for the fused Blackwell NVFP4 activation quantizer and ``NVFP4Linear`` (Cycle 4 C4-02).

Parity tests compare the Triton kernels bit for bit with the eager reference
``micro_rotated_ptq.nvfp4_encode + pack_e2m1 + to_blocked`` (after the same FP32 rotation).
GPU tests need CUDA; the native FP4 GEMM tests also need ``F.scaled_mm`` on sm_100+.
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from research.quant_ptq import micro_rotated_ptq as q

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")
FP4_GEMM = CUDA and hasattr(F, "scaled_mm") and torch.cuda.get_device_capability()[0] >= 10
needs_fp4 = pytest.mark.skipif(not FP4_GEMM, reason="needs native NVFP4 scaled_mm (sm_100+)")
ROOT = Path(__file__).resolve().parents[1]

if CUDA:
    from research.quant_ptq import bench_nvfp4_e2e as bench
    from research.quant_ptq import nvfp4_inference_blackwell as nb

ROTS = ("none", "hadamard_b16", "adaptive_butterfly_b16")
M_SIZES = (1, 37, 128, 129, 256, 512)
K_SIZES = (64, 80, 512, 768, 1408, 2048)


def _angles(K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand(K // 16, 4, 8, generator=g, device="cuda") * 2 * math.pi


def _args(rotation, K, seed=0):
    angles = _angles(K, seed) if rotation == "adaptive_butterfly_b16" else None
    return angles, (nb.butterfly_tables(angles) if angles is not None else None)


def _assert_same(a, r):
    assert torch.equal(a.packed.view(torch.uint8), r.packed.view(torch.uint8))
    assert torch.equal(a.scale.view(torch.uint8), r.scale.view(torch.uint8))
    assert torch.equal(a.global_scale.view(torch.int32), r.global_scale.view(torch.int32))


# ------------------------------------------------------------------ bit-exact parity
@needs_cuda
@pytest.mark.parametrize("rotation", ROTS)
@pytest.mark.parametrize("K", K_SIZES)
@pytest.mark.parametrize("M", M_SIZES)
def test_kernel_matches_reference_bit_for_bit(M, K, rotation):
    g = torch.Generator(device="cuda").manual_seed(M * 7919 + K)
    x = (torch.randn(M, K, generator=g, device="cuda") * 2).to(torch.bfloat16)
    x[:, :16] *= 30  # an outlier block
    angles, tables = _args(rotation, K)
    ref = nb.quantize_activation_ref(x, rotation, angles)
    _assert_same(nb.quantize_activation(x, rotation, tables), ref)
    # Static scale: half the dynamic one, so some blocks saturate at E4M3 448 and E2M1 6.
    static = (ref.global_scale * 0.5).clone()
    _assert_same(nb.quantize_activation(x, rotation, tables, static),
                 nb.quantize_activation_ref(x, rotation, angles, static))


@needs_cuda
def test_reference_is_the_micro_rotated_ptq_pipeline():
    x = torch.randn(129, 80, device="cuda")
    enc = q.nvfp4_encode(x)
    ref = nb.quantize_activation_ref(x)
    assert torch.equal(ref.packed.view(torch.uint8), q.pack_e2m1(enc["codes"]).view(torch.uint8))
    assert torch.equal(ref.scale.view(torch.uint8), q.to_blocked(enc["block_scale"]).view(torch.uint8))
    pr, pc = nb.scale_layout(129, 80)
    assert (pr, pc) == (256, 8) and ref.scale.numel() == pr * pc
    # Dequantizing the swizzled operands gives the encoder's values exactly.
    assert torch.equal(nb.dequantize(ref, 129, 80), enc["values"])


@needs_cuda
def test_padding_slots_are_zero_even_in_dirty_memory():
    M, K = 37, 80  # 128 - 37 padded rows, K // 16 = 5 padded to 8 scale columns
    x = torch.randn(M, K, device="cuda")
    pr, pc = nb.scale_layout(M, K)
    packed = torch.full((M, K // 2), 0xAB, dtype=torch.uint8, device="cuda")
    scale = torch.full((pr * pc,), 0xCD, dtype=torch.uint8, device="cuda")
    partials = torch.full((nb.MAX_PARTIALS,), float("nan"), device="cuda")
    gs = torch.empty((), device="cuda")
    n = nb.launch_amax(x, partials, "none")
    nb.launch_quant(x, packed, scale, gs, workspace=partials, n_part=n)
    ref = nb.quantize_activation_ref(x)
    assert torch.equal(scale, ref.scale.view(torch.uint8))
    assert torch.equal(packed, ref.packed.view(torch.uint8))
    perm = q.to_blocked(torch.arange(pr * pc, device="cuda").view(pr, pc))
    logical = torch.empty_like(scale)
    logical[perm] = scale
    logical = logical.view(pr, pc)
    assert (logical[M:] == 0).all() and (logical[:, K // 16:] == 0).all()


@needs_cuda
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("rotation", ROTS)
def test_input_dtypes_and_non_contiguous_inputs(dtype, rotation):
    base = torch.randn(300, 1024, device="cuda", dtype=dtype)
    angles, tables = _args(rotation, 512)
    for x in (base[::2, 256:768], base.t().contiguous().t()[:, :512], base[1:130, ::2]):
        assert not x.is_contiguous()
        _assert_same(nb.quantize_activation(x, rotation, tables), nb.quantize_activation_ref(x, rotation, angles))


@needs_cuda
@pytest.mark.parametrize("rotation", ROTS)
def test_all_zero_activations(rotation):
    x = torch.zeros(129, 512, device="cuda", dtype=torch.bfloat16)
    angles, tables = _args(rotation, 512)
    a = nb.quantize_activation(x, rotation, tables)
    _assert_same(a, nb.quantize_activation_ref(x, rotation, angles))
    assert a.global_scale.item() == 1.0  # amax 0 -> tensor scale 1, as nvfp4_global_scale
    assert (a.scale.view(torch.uint8) == 0).all()
    # Magnitude 0 everywhere. The butterfly can produce -0.0 (e.g. 0 * sin - cos * 0 with cos < 0),
    # which keeps its sign bit (code 8), exactly as in the reference.
    assert (a.packed.view(torch.uint8) & 0x77 == 0).all()
    x[3, 5] = -0.0  # a negative zero keeps its sign bit (code 8) as in e2m1_codes
    x[0, 0] = -0.0
    _assert_same(nb.quantize_activation(x.float(), rotation, tables), nb.quantize_activation_ref(x, rotation, angles))


@needs_cuda
@pytest.mark.parametrize("magnitude", (1e-30, 1e-20, 1e-3, 1e3, 1e20, 1e30))
def test_huge_and_tiny_magnitudes(magnitude):
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(64, 512, generator=g, device="cuda") * magnitude
    x[0, :16] = torch.logspace(-6, 0, 16, device="cuda") * magnitude  # spans subnormal E4M3 block scales
    for rotation in ROTS:
        angles, tables = _args(rotation, 512)
        a = nb.quantize_activation(x, rotation, tables)
        _assert_same(a, nb.quantize_activation_ref(x, rotation, angles))
        assert torch.isfinite(a.global_scale)


@needs_cuda
def test_static_scale_saturates_and_tiny_static_scale():
    x = torch.randn(64, 256, device="cuda") * 4
    for gs in (1e-6, 1e-3, 1.0, 1e6):
        static = torch.tensor(gs, device="cuda")
        a = nb.quantize_activation(x, "hadamard_b16", None, static)
        _assert_same(a, nb.quantize_activation_ref(x, "hadamard_b16", None, static))
        assert a.global_scale.data_ptr() == static.data_ptr()  # passed through, not copied
    a = nb.quantize_activation(x, "none", None, torch.tensor(1e-6, device="cuda"))
    scales = a.scale.view(torch.uint8)
    assert (scales <= 0x7E).all() and (scales == 0x7E).any()  # saturated at 448 (0x7E), never NaN


@needs_cuda
@pytest.mark.parametrize("rotation", ROTS)
def test_nan_and_inf_follow_the_reference(rotation):
    """Documented non-finite behavior, identical to the eager reference.

    Dynamic mode: a NaN anywhere makes amax NaN, so the tensor scale falls back to 1 (the
    ``amax > 0`` test of ``nvfp4_global_scale`` fails), and the NaN block gets the E4M3 NaN
    scale 0x7F. An Inf makes the tensor scale Inf, so every finite block scale becomes 0.
    Non-finite activations therefore destroy the quantization of the whole tensor, exactly as
    in the reference; callers must keep activations finite.
    """
    angles, tables = _args(rotation, 256)
    for bad in (float("nan"), float("inf"), float("-inf")):
        x = torch.randn(40, 256, device="cuda")
        x[7, 33] = bad
        a = nb.quantize_activation(x, rotation, tables)
        _assert_same(a, nb.quantize_activation_ref(x, rotation, angles))
        if math.isnan(bad):
            assert a.global_scale.item() == 1.0
        else:
            assert math.isinf(a.global_scale.item())
        static = torch.tensor(0.01, device="cuda")
        _assert_same(nb.quantize_activation(x, rotation, tables, static),
                     nb.quantize_activation_ref(x, rotation, angles, static))


@needs_cuda
def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        nb.quantize_activation(torch.randn(4, 40, device="cuda"))
    with pytest.raises(ValueError):
        nb.quantize_activation(torch.randn(4, 64, device="cuda"), "hadamard_b64")
    with pytest.raises(ValueError):
        nb.quantize_activation(torch.randn(4, 64, device="cuda"), "adaptive_butterfly_b16")
    with pytest.raises(ValueError):
        nb.quantize_activation(torch.randn(4, 64, device="cuda"), global_scale=torch.tensor(1.0, device="cuda").double())


# ------------------------------------------------------------------ GEMM and module
@needs_fp4
@pytest.mark.parametrize("rotation", ROTS)
@pytest.mark.parametrize("M,N,K", [(1, 256, 64), (37, 512, 768), (129, 384, 1408), (512, 1024, 2048)])
def test_scaled_mm_matches_fp32_dequantized_reference(M, N, K, rotation):
    g = torch.Generator(device="cuda").manual_seed(M + N + K)
    x = torch.randn(M, K, generator=g, device="cuda").to(torch.bfloat16)
    W = torch.randn(N, K, generator=g, device="cuda") * 0.02
    angles, _ = _args(rotation, K, seed=1)
    layer = nb.NVFP4Linear.from_weight(W, rotation, angles)
    y = layer(x)
    act = layer.quantize_input(x)
    reference = nb.dequantize(act, M, K) @ nb.dequantized_weight(layer).T
    assert y.dtype == torch.bfloat16 and y.shape == (M, N)
    assert ((y.float() - reference).norm() / reference.norm()).item() < 1e-2
    # Same operands through the eager reference pipeline give the same GEMM output bit for bit.
    eager = nb.nvfp4_scaled_mm(nb.quantize_activation_ref(x, rotation, angles), layer.w_packed, layer.w_scale,
                               layer.w_global)
    assert torch.equal(y, eager)
    # The rotation cancels: W4A4 is close to the exact product (FP4 noise only).
    exact = x.float() @ W.T
    assert ((y.float() - exact).norm() / exact.norm()).item() < 0.2


@needs_fp4
def test_static_module_and_leading_dims():
    W = torch.randn(256, 512, device="cuda") * 0.02
    x = torch.randn(2, 3, 512, device="cuda", dtype=torch.bfloat16)
    gs = torch.tensor(0.01, device="cuda")
    layer = nb.NVFP4Linear.from_weight(W, "hadamard_b16", act_global_scale=gs)
    y = layer(x)
    assert y.shape == (2, 3, 256)
    ref = nb.nvfp4_scaled_mm(nb.quantize_activation_ref(x.reshape(6, 512), "hadamard_b16", None, gs),
                             layer.w_packed, layer.w_scale, layer.w_global)
    assert torch.equal(y.reshape(6, 256), ref)
    assert layer(x[:, :0]).shape == (2, 0, 256)
    assert "static" in repr(layer)


@needs_fp4
def test_direct_scaled_mm_call_equals_public_api_and_dtype_casts_keep_fp32_scales():
    W = torch.randn(256, 512, device="cuda") * 0.02
    angles = _angles(512)
    layer = nb.NVFP4Linear.from_weight(W, "adaptive_butterfly_b16", angles,
                                       act_global_scale=torch.tensor(0.01, device="cuda"))
    x = torch.randn(70, 512, device="cuda", dtype=torch.bfloat16)
    act = layer.quantize_input(x)
    direct = nb.nvfp4_scaled_mm(act, layer.w_packed, layer.w_scale, layer.w_global)
    public = nb.nvfp4_scaled_mm_public(act, layer.w_packed, layer.w_scale, layer.w_global)
    assert torch.equal(direct, public)
    before = layer(x)
    layer.to(torch.bfloat16)  # as in ``model.to(torch.bfloat16)``
    for name in ("w_global", "act_global_scale", "angles", "cos", "sin"):
        assert getattr(layer, name).dtype == torch.float32, name
    assert torch.equal(layer(x), before)


@needs_cuda
def test_gptq_encoding_reconstructs_q_exactly():
    g = torch.Generator().manual_seed(0)
    W = (torch.randn(2, 32, 128, generator=g) * 0.02).cuda()
    X = q.correlated_inputs(1024, 128, 1e3, 0, "cuda", torch.float32)
    H = (X.double().T @ X.double() / 1024).expand(2, -1, -1).contiguous()
    Q, info = q.gptq_quantize(W, H, q.FORMATS["nvfp4_g16"], return_encoding=True)
    Q2, _ = q.gptq_quantize(W, H, q.FORMATS["nvfp4_g16"])
    assert torch.equal(Q, Q2)  # recording the encoding does not change GPTQ
    enc = info["encoding"]
    for i in range(2):
        values = q.nvfp4_decode({"codes": enc["codes"][i], "block_scale": enc["block_scale"][i],
                                 "global_scale": enc["global_scale"][i]})
        assert torch.equal(values, Q[i])
    with pytest.raises(ValueError):
        q.gptq_quantize(W, H, q.FORMATS["mxfp4_g32"], return_encoding=True)


@needs_cuda
def test_nvfp4_encode_of_rtn_matches_quantize_rtn():
    W = torch.randn(3, 64, 256, device="cuda") * 0.02
    rtn = q.quantize_rtn(W, q.FORMATS["nvfp4_g16"])
    for i in range(3):
        assert torch.equal(q.nvfp4_encode(W[i])["values"], rtn[i])


# ------------------------------------------------------------------ sync-free and CUDA graphs
@needs_fp4
@pytest.mark.parametrize("rotation", ROTS)
@pytest.mark.parametrize("static", (False, True))
def test_forward_has_no_device_to_host_sync(rotation, static):
    W = torch.randn(512, 1024, device="cuda") * 0.02
    angles, _ = _args(rotation, 1024)
    layer = nb.NVFP4Linear.from_weight(W, rotation, angles,
                                       act_global_scale=torch.tensor(0.01, device="cuda") if static else None)
    x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
    layer(x)  # compile
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        layer(x)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function("nvfp4_forward"):
            for _ in range(3):
                layer(x)
    torch.cuda.synchronize()
    events = prof.events()
    region = next(e for e in events if e.name == "nvfp4_forward" and e.device_type.name == "CPU").time_range
    # The profiler itself synchronizes when it stops; only calls inside the forward region count.
    inside = [e for e in events if e.device_type.name == "CPU"
              and region.start <= e.time_range.start and e.time_range.end <= region.end]
    syncs = [e.name for e in inside if "Synchronize" in e.name or "Memcpy" in e.name or "item" in e.name.lower()
             or "local_scalar_dense" in e.name]
    assert not syncs, syncs
    names = [e.name for e in events]
    assert not [n for n in names if "DtoH" in n]
    assert any("scaled_mm" in n or "gemm" in n.lower() or "cutlass" in n.lower() for n in names)
    launched = [n for n in names if "_quant_kernel" in n or "_amax_kernel" in n]
    assert launched, "the Triton kernels should appear in the CUDA trace"


@needs_fp4
@pytest.mark.parametrize("rotation", ROTS)
@pytest.mark.parametrize("static", (False, True))
def test_cuda_graph_capture_and_replay_is_bitwise_equal(rotation, static):
    W = torch.randn(384, 768, device="cuda") * 0.02
    angles, _ = _args(rotation, 768)
    layer = nb.NVFP4Linear.from_weight(W, rotation, angles,
                                       act_global_scale=torch.tensor(0.004, device="cuda") if static else None)
    x_static = torch.randn(37, 768, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        layer(x_static)  # warm up on the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_static = layer(x_static)
    g = torch.Generator(device="cuda").manual_seed(5)
    for scale in (1.0, 1e-3, 50.0, 1.0):
        x_static.copy_(torch.randn(37, 768, generator=g, device="cuda") * scale)
        graph.replay()
        assert torch.equal(y_static, layer(x_static))
        assert torch.equal(y_static, nb.nvfp4_scaled_mm(
            nb.quantize_activation_ref(x_static, rotation, angles, layer.act_global_scale),
            layer.w_packed, layer.w_scale, layer.w_global))


@needs_cuda
def test_first_launch_during_capture_is_refused():
    x = torch.randn(3, 4096 + 64, device="cuda")  # a shape no other test compiled
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="Warm up"):
        with torch.cuda.graph(graph):
            nb.quantize_activation(x)


# ------------------------------------------------------------------ benchmark helpers and CLI guard
@needs_cuda
def test_williams_square_is_balanced():
    for n in (2, 3, 4, 5, 7):
        square = bench.williams_square(n)
        assert all(sorted(row) == list(range(n)) for row in square)
        follows = {}
        for row in square:
            for a, b in zip(row, row[1:]):
                follows[(a, b)] = follows.get((a, b), 0) + 1
        assert len(follows) == n * (n - 1) and len(set(follows.values())) == 1


@needs_cuda
def test_cli_output_guard_protects_canonical_artifact(tmp_path, monkeypatch):
    def fail_run(*args, **kwargs):
        raise AssertionError("run_experiment should not be called")

    monkeypatch.setattr(bench, "run_experiment", fail_run)
    assert bench.parse_args([]).output == bench.CANONICAL_OUTPUT
    assert bench.CANONICAL_OUTPUT == ROOT / "results" / "cycle4_c4_02_nvfp4_e2e.json"
    assert bench.parse_args(["--quick"]).output == bench.QUICK_OUTPUT
    custom = tmp_path / "sub.json"
    assert bench.parse_args(["--steps", "10", "--output", str(custom)]).output == custom
    for argv in (
        ["--quick", "--output", str(bench.CANONICAL_OUTPUT)],
        ["--steps", "10"],
        ["--parts", "A"],
        ["--rounds", "2", "--output", str(bench.CANONICAL_OUTPUT)],
        ["--no-verify", "--output", str(bench.CANONICAL_OUTPUT)],
        ["--max-eval-windows", "8"],
    ):
        with pytest.raises(SystemExit) as exc:
            bench.main(argv)
        assert exc.value.code == 2
