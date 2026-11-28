"""Checks for Cycle 6 C6-02: the small-shape NVFP4 fast path and its benchmark driver.

Kernel tests compare ``nvfp4_small_shape_blackwell`` bit for bit with the eager reference
(``nvfp4_inference_blackwell.quantize_activation_ref``) and with the unchanged C5-03 module: operands,
layer outputs, the direct cuBLASLt GEMM, the ablation settings, the public launcher contract, CUDA graphs
and the absence of host synchronizations. Cross-mix tests measure the fused kernel against
``rotated_qat.CrossInput``. Driver tests cover the CLI and writer guards, the validator (on a shakedown run,
with tampering probes) and the committed canonical result, whose SHA-256 is pinned here.
"""

import copy
import gc
import hashlib
import io
import json
import math
import threading
import weakref
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
# The v3 canonical result and its producer commit. ``None`` only in the producer commit itself, before the
# canonical run (the result file does not exist there); the result commit pins both.
CANONICAL_SHA256 = None
CANONICAL_PRODUCER = None
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
    assert fs.delegated(M, K, "none") and fs.delegated(M, K, "adaptive_butterfly_b16")
    assert not fs.delegated(M, K, "hadamard_b16") and not fs.delegated(1024, 4096, "none")
    assert not fs.delegated(8192, 1408, "adaptive_butterfly_b16") and fs.delegated(8192, 1408, "none")
    assert fs.delegated(4096, 2048, "adaptive_butterfly_b16") and fs.delegated(1024, 2048, "adaptive_butterfly_b16")
    assert not fs.delegated(512, 2048, "adaptive_butterfly_b16") and not fs.delegated(128, 4096, "adaptive_butterfly_b16")
    assert not fs.delegated(1023, 8192, "none") and fs.delegated(1024, 8192, "none")  # from 2^23 elements
    assert fs.quantizer_plan(M, K, "none")["delegated_to"] == "nvfp4_inference_blackwell"
    assert "delegated_to" not in fs.quantizer_plan(M, K, "hadamard_b16")
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    angles, tables = _args("adaptive_butterfly_b16", K)
    for rotation, a, t in (("none", None, None), ("adaptive_butterfly_b16", angles, tables)):
        assert _same(fs.quantize_activation(x, rotation, t), nb.quantize_activation(x, rotation, t))
        with b.options(delegate_large=False):  # the C6-02 kernels themselves are exact there too
            assert not fs.delegated(M, K, rotation)
            assert _same(fs.quantize_activation(x, rotation, t), nb.quantize_activation_ref(x, rotation, a))


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


@needs_fp4
def test_concurrent_direct_gemm_submissions_are_exact():
    """Two host threads, own streams and scales, one shared cached plan: every output equals its serial reference."""
    M, N, K = 32, 128, 256
    work = []
    for seed in (1, 2):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = (torch.randn(M, K, generator=g, device="cuda") * 10.0 ** seed).to(torch.bfloat16)
        layer = nb.NVFP4Linear.from_weight(torch.randn(N, K, generator=g, device="cuda") * 0.02 * seed, "none")
        a = nb.quantize_activation(x, "none")
        alpha = (a.global_scale * layer.w_global).reshape(())
        ref = nb.nvfp4_scaled_mm_public(a, layer.w_packed, layer.w_scale, layer.w_global)
        assert _bits(fs.nvfp4_scaled_mm_lt(a, layer.w_packed, layer.w_scale, alpha), ref)  # warm-up, serial
        work.append((a, layer, alpha, ref))
    assert not torch.equal(work[0][0].scale, work[1][0].scale)
    torch.cuda.synchronize()
    iters = 500
    for _ in range(2):
        barrier, bad, errors = threading.Barrier(2), [None, None], []

        def run(i):
            try:
                a, layer, alpha, ref = work[i]
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    wrong = torch.zeros((), dtype=torch.int64, device="cuda")
                    for _ in range(iters):
                        barrier.wait()
                        out = fs.nvfp4_scaled_mm_lt(a, layer.w_packed, layer.w_scale, alpha)
                        wrong += (out.view(torch.int16) != ref.view(torch.int16)).any()
                stream.synchronize()
                bad[i] = int(wrong)
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)
                barrier.abort()

        threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors and bad == [0, 0]


@needs_fp4
@pytest.mark.parametrize("K,Ms", [(2048, (1023, 1024)), (256, (6, 37))])
def test_butterfly_load_state_dict_rebuilds_the_packed_tables(K, Ms):
    """load_state_dict from a layer with other angles: native and delegated forwards follow the loaded tables."""
    rot = "adaptive_butterfly_b16"
    g = torch.Generator(device="cuda").manual_seed(K)
    src = nb.NVFP4Linear.from_weight(torch.randn(128, K, generator=g, device="cuda") * 0.02, rot, _angles(K, 11))
    fast = fs.FastNVFP4Linear.from_weight(torch.randn(128, K, generator=g, device="cuda") * 0.02, rot, _angles(K, 12))
    if K == 2048:  # both sides of the butterfly delegation limit
        assert not fs.delegated(1023, K, rot) and fs.delegated(1024, K, rot)
    ptr = fast.cos_sin.data_ptr()
    assert fast.load_state_dict(src.state_dict()) is not None
    assert torch.equal(fast.cos_sin, fs.packed_tables(src.cos, src.sin)) and fast.cos_sin.data_ptr() == ptr
    xs = [torch.randn(M, K, generator=g, device="cuda").to(torch.bfloat16) for M in Ms]
    for x in xs:
        assert _bits(fast(x), src(x)), x.shape
    buf = io.BytesIO()
    torch.save(torch.nn.Sequential(src).state_dict(), buf)
    buf.seek(0)
    other = torch.nn.Sequential(fs.FastNVFP4Linear.from_weight(torch.randn(128, K, device="cuda"), rot, _angles(K, 13)))
    other.load_state_dict(torch.load(buf, weights_only=True))
    for x in xs:
        assert _bits(other(x), src(x)), x.shape


@needs_fp4
def test_lt_plan_device_alias_workspace_contract_and_close(monkeypatch):
    M, N, K = 37, 96, 64
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    layer = nb.NVFP4Linear.from_weight(torch.randn(N, K, device="cuda") * 0.02, "none")
    a = nb.quantize_activation(x, "none")
    alpha = (a.global_scale * layer.w_global).reshape(())
    ref = nb.nvfp4_scaled_mm_public(a, layer.w_packed, layer.w_scale, layer.w_global)
    here = torch.device("cuda", torch.cuda.current_device())
    cached = fs.lt_plan(M, N, K, torch.bfloat16, torch.device("cuda"))
    assert cached is fs.lt_plan(M, N, K, torch.bfloat16, here) is fs.lt_plan(M, N, K, torch.bfloat16, "cuda")
    assert fs._device_index(torch.device("cuda")) == fs._device_index(here) == here.index
    plan = fs.LtPlan(M, N, K, torch.bfloat16, torch.device("cuda"))  # a private plan, not cached
    plan.workspace_bytes = 4096  # simulate an algorithm that needs a workspace
    out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    for bad in (None, torch.empty(4095, dtype=torch.uint8, device="cuda"), torch.empty(4096, device="cuda"),
                torch.empty(8192, dtype=torch.uint8, device="cuda")[::2], torch.empty(4096, dtype=torch.uint8)):
        with pytest.raises(ValueError, match="workspace"):
            plan(a, layer.w_packed, layer.w_scale, alpha, out, bad)
    plan(a, layer.w_packed, layer.w_scale, alpha, out, torch.empty(4096, dtype=torch.uint8, device="cuda"))
    assert _bits(out, ref)
    plan.close()
    plan.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        plan(a, layer.w_packed, layer.w_scale, alpha, out, torch.empty(4096, dtype=torch.uint8, device="cuda"))
    seen, call = [], fs.LtPlan.__call__

    def spy(self, *args, **kwargs):
        seen.append(args[5] if len(args) > 5 else kwargs.get("workspace"))
        return call(self, *args, **kwargs)

    monkeypatch.setattr(fs.LtPlan, "__call__", spy)
    monkeypatch.setattr(cached, "workspace_bytes", 4096)
    got = fs.nvfp4_scaled_mm_lt(a, layer.w_packed, layer.w_scale, alpha)  # the helper supplies the workspace
    assert seen[-1] is not None and seen[-1].dtype == torch.uint8 and seen[-1].nbytes >= 4096 and _bits(got, ref)


@needs_fp4
def test_lt_plan_cache_is_a_bounded_lru(monkeypatch):
    monkeypatch.setattr(fs, "LT_PLAN_CACHE_MAX", 2)
    fs.clear_lt_plan_cache()
    K, N = 64, 256
    p1, p2 = fs.lt_plan(1, N, K, torch.bfloat16, "cuda"), fs.lt_plan(2, N, K, torch.bfloat16, "cuda")
    assert fs.lt_plan(1, N, K, torch.bfloat16, "cuda") is p1  # p1 is now the most recent
    p3 = fs.lt_plan(3, N, K, torch.bfloat16, "cuda")
    keys = [k[0] for k in fs._LT_PLANS]
    assert keys == [1, 3] and not p2.closed  # evicted, not closed: a caller may still hold it
    x = torch.randn(2, K, device="cuda").to(torch.bfloat16)
    layer = nb.NVFP4Linear.from_weight(torch.randn(N, K, device="cuda") * 0.02, "none")
    a = nb.quantize_activation(x, "none")
    alpha = (a.global_scale * layer.w_global).reshape(())
    out = torch.empty((2, N), dtype=torch.bfloat16, device="cuda")
    assert _bits(p2(a, layer.w_packed, layer.w_scale, alpha, out),
                 nb.nvfp4_scaled_mm_public(a, layer.w_packed, layer.w_scale, layer.w_global))
    dropped = weakref.ref(p2)
    del p2
    gc.collect()
    assert dropped() is None  # its last reference went: the descriptor and layouts were destroyed
    assert fs.clear_lt_plan_cache() == 2 and p1.closed and p3.closed and not fs._LT_PLANS
    with pytest.raises(RuntimeError, match="closed"):
        p1(a, layer.w_packed, layer.w_scale, alpha, out)


@needs_fp4
def test_tile_overrides_take_effect_without_stale_plans():
    M, K = 17, 256
    x = torch.randn(M, K, device="cuda")
    ref = nb.quantize_activation_ref(x, "none")
    default1, default2 = fs._pass1(M, K, "none"), fs._pass2(M, K, "none", True, fs._n_partials(M, K, "none"))
    assert default1[3] == 64
    fs.TILES["pass1"][(M, K, "none")] = (32, 1, 1, 512)
    fs.TILES["pass2"][(M, K, "none")] = (64, 1, 2)
    try:
        assert fs._pass1(M, K, "none")[3] == 32 and fs.quantizer_plan(M, K, "none")["pass1_tile"] == [32, 1, 1]
        assert fs.quantizer_plan(M, K, "none")["pass2_tile"] == [64, 1, 2]
        assert _same(fs.quantize_activation(x), ref)
    finally:
        del fs.TILES["pass1"][(M, K, "none")], fs.TILES["pass2"][(M, K, "none")]
    assert fs._pass1(M, K, "none") == default1
    assert fs._pass2(M, K, "none", True, fs._n_partials(M, K, "none")) == default2


@needs_fp4
def test_options_scope_is_isolated_per_thread_and_nests():
    barrier, seen, errors = threading.Barrier(2), {}, []

    def run(name, value):
        try:
            with fs.options_scope(pdl=value):
                barrier.wait()
                seen[name] = (fs._option("pdl"), fs.quantizer_plan(129, 768, "none")["pdl"])
                barrier.wait()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=run, args=args) for args in (("a", False), ("b", True))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and seen == {"a": (False, False), "b": (True, True)}
    assert fs.OPTIONS["pdl"] is True and fs._option("pdl") is True
    with b.options(pdl=False):
        with fs.options_scope(vector_io=False):
            assert fs._option("pdl") is False and fs._option("vector_io") is False
        assert fs._option("vector_io") is True and fs._option("pdl") is False
    assert fs._option("pdl") is True and fs.OPTIONS == {"pdl": True, "vector_io": True, "lt_gemm": True,
                                                        "stash": False, "delegate_large": True}
    with pytest.raises(KeyError):
        with fs.options_scope(unknown=1):
            pass


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
    for small in (x[0, :37], x[0, :1].float()):  # other cuBLAS kernels: FP32 rounding (1e-6) only
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


@needs_fp4
@pytest.mark.parametrize("family", ("kron_hadamard_full", "cross_full_butterfly_b16"))
def test_exact_cross_mix_matches_crossinput_at_the_w2048_down_projection(family):
    """K = 5632 (w2048_6l down projection), 2048 rows: exact mode is bit-identical to rotated_qat.CrossInput."""
    K = 5632
    cross = rq.make_cross(rq.FAMILIES[family][0], K)
    rot = rq.kernel_rotation(family)
    W = torch.randn(2048, K, device="cuda") * 0.02
    layer = fs.FastNVFP4Linear.from_layer(nb.NVFP4Linear.from_weight(
        cross.forward(W), rot, _angles(K) if rot == "adaptive_butterfly_b16" else None))
    exact, eager = fs.FastCrossInput(layer, cross, exact=True), rq.CrossInput(layer, cross)
    x = torch.randn(4, 512, K, device="cuda").to(torch.bfloat16)
    assert exact.exact and not fs.FastCrossInput(layer, cross).exact
    assert _bits(exact.mixed(x), cross.forward(x.float()).reshape(-1, K))
    assert _bits(exact(x), eager(x))


@needs_fp4
def test_exact_cross_input_runs_the_inner_layer_under_the_callers_autocast():
    """exact=True disables autocast for the FP32 mix only, like rotated_qat.CrossInput: under CUDA BF16 autocast
    an FP32 nn.Linear inner module returns the same BF16 bits through both wrappers (K = 1408)."""
    K = 1408
    cross = rq.make_cross("full", K)
    torch.manual_seed(0)
    layer = torch.nn.Linear(K, 256, device="cuda")
    exact, eager = fs.FastCrossInput(layer, cross, exact=True).cuda(), rq.CrossInput(layer, cross)
    x = torch.randn(4, 64, K, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        got, want = exact(x), eager(x)
    assert got.dtype == want.dtype == torch.bfloat16
    assert _bits(got, want)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.autocast(device_type="cuda", enabled=False):
        assert exact(x).dtype == torch.float32  # the inner layer follows its caller's context


@needs_fp4
def test_cross_mix_matrices_stay_fp32_under_dtype_conversion():
    K = 1408
    cross = rq.make_cross("full", K)
    W = torch.randn(256, K, device="cuda") * 0.02

    def build():
        return fs.FastNVFP4Linear.from_layer(nb.NVFP4Linear.from_weight(cross.forward(W), "hadamard_b16"))

    x = torch.randn(1024, K, device="cuda").to(torch.bfloat16)
    base = fs.FastCrossInput(build(), cross)
    want_mix, want_out = base.mixed(x), base(x)
    assert (want_mix - cross.forward(x.float())).abs().max().item() < 1e-6
    conversions = (lambda m: m.bfloat16(), lambda m: m.half(), lambda m: m.to(dtype=torch.bfloat16),
                   lambda m: m.to("cuda", torch.float16), lambda m: m.cpu().cuda())
    built = [conv(fs.FastCrossInput(build(), cross)) for conv in conversions]  # after construction
    built += [fs.FastCrossInput(build().bfloat16(), cross), fs.FastCrossInput(build().half(), cross).half()]  # before
    for m in built:
        assert m.h.dtype == m.mix.dtype == torch.float32 and m.h.is_cuda and m.mix.is_cuda
        assert _bits(m.mixed(x), want_mix) and _bits(m(x), want_out)
    moved = fs.FastCrossInput(build(), cross).cpu()
    assert moved.h.device.type == "cpu" and moved.h.dtype == torch.float32
    broken = fs.FastCrossInput(build(), cross)
    broken._buffers["mix"] = broken.mix.bfloat16()
    with pytest.raises(RuntimeError, match="float32"):
        broken.mixed(x)


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
    b.write_json_atomic(b.CANONICAL_OUTPUT, payload, canonical_dest=True)
    with pytest.raises(FileExistsError):  # no-replace publication
        b.write_json_atomic(b.CANONICAL_OUTPUT, {"trial": 2}, canonical_dest=True)
    assert json.loads(b.CANONICAL_OUTPUT.read_text()) == payload and _leftovers(sandbox / "results") == []
    plain = sandbox / "results" / "plain.json"
    b.write_json_atomic(plain, {"trial": 1})
    b.write_json_atomic(plain, {"trial": 2})  # ordinary outputs may be replaced
    assert json.loads(plain.read_text()) == {"trial": 2}
    b.CANONICAL_OUTPUT.unlink()
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


def _first_model(payload: dict) -> dict:
    return next(iter(payload["part_c"]["models"].values()))


def _flip(record: dict, key: str) -> None:
    record[key] = not record[key]


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
    "part a summary": lambda p: _flip(p["part_a"]["shapes"][0]["summary"]["cuda_graph"]["variants"]["none"],
                                      "c6_02_beats_bf16"),
    "part a orders": lambda p: p["part_a"]["shapes"][0]["cuda_graph_williams_orders"].reverse(),
    "part a shapes": lambda p: p["part_a"]["shapes"].pop(),
    "part a delegation": lambda p: p["part_a"]["shapes"][0]["delegated_to_c5_03"].__setitem__("none", True),
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
    # The five bypasses that the v1 validator accepted (audit, "Offline validation and artifact handling"):
    "part b heuristic row": lambda p: p["part_b"]["heuristic"]["algorithms"][0].update(bitwise_equal_public=False,
                                                                                      cuda_graph_ms=-1.0),
    "part b heuristic workspace": lambda p: p["part_b"]["heuristic"]["algorithms"][0].__setitem__("workspace_bytes", -1),
    "part b empty raw trace": lambda p: p["part_b"]["traces"]["cublaslt_fused_alpha"].__setitem__("kernels", []),
    "part b direct aten mul": lambda p: p["part_b"]["traces"]["c6_02_layer"]["aten_ops"].append("aten::mul"),
    "part b raw kind": lambda p: p["part_b"]["traces"]["torch_scaled_mm_v2"]["kernels"].__setitem__(0, "nvjet_bf16"),
    "ablation delegation empty": lambda p: p["part_a"]["ablation"][-1].__setitem__("delegated_to_c5_03", {}),
    "profile count": lambda p: next(iter(p["part_a"]["shapes"][0]["cuda_graph_kernel_profile"]["c6_02_none"].values()))
    .__setitem__("count", 0),
    "profile us": lambda p: next(iter(p["part_a"]["shapes"][0]["cuda_graph_kernel_profile"]["c6_02_none"].values()))
    .__setitem__("us", -1.0),
    "part c negative logits error": lambda p: _first_model(p)["fused_cross_mix_parity"]["kron_hadamard_full"]
    .__setitem__("logits_rel_diff_vs_c6_01", -1.0),
    "part c impossible logits count": lambda p: _first_model(p)["fused_cross_mix_parity"]["kron_hadamard_full"]
    .update(logits_numel=1, logits_mismatches=10 ** 8, logits_bitwise_equal_c6_01=False),
    "part c mixed count": lambda p: _first_model(p)["fused_cross_mix_parity"]["hadamard_b64"].__setitem__(
        "mixed_values", 1),
    "part c zero biconditional": lambda p: _first_model(p)["fused_cross_mix_parity"]["hadamard_b64"].__setitem__(
        "layers_exact", _first_model(p)["fused_cross_mix_parity"]["hadamard_b64"]["layers_exact"] - 1),
    "part c compiled parity": lambda p: _first_model(p)["bf16_compiled_parity"].__setitem__("logits_finite", False),
    "part c compiled timing": lambda p: _first_model(p)["timing"]["bf16_compiled"]["gpu"].__setitem__("median_ms", -1.0),
    "part c compiled profile": lambda p: _first_model(p)["kernel_kinds_per_forward"]["bf16_compiled"].__setitem__(
        "fp4_gemm", 1.0),
    "part c trained eval": lambda p: p["part_c"]["compiled_bf16_trained_eval"].__setitem__("x", {}),
    "part c check": lambda p: _first_model(p)["checks"].__setitem__("had_b16_c6_02_logits_equal_c5_03_bitwise", False),
    "part c parity": lambda p: _first_model(p)["fused_cross_mix_parity"]["hadamard_b64"].__setitem__(
        "mixed_mismatches", -1),
    "part c summary": lambda p: _first_model(p)["summary"]["wide"]["hadamard_b64"].__setitem__("c6_02_fused_ms", 1.0),
    "part c kernels": lambda p: _first_model(p)["kernel_kinds_per_forward"]["c6_02_had_b16"].__setitem__(
        "scale_product", 99.0),
    "summary": lambda p: p["summary"].__setitem__("part_a_cuda_graph_cells_beating_bf16", [8, 8]),
    "caveats": lambda p: p["caveats"].append("x"),
    "pareto": lambda p: p.__setitem__("pareto", {"50m": {}}),
    "canonical flag": lambda p: p.__setitem__("canonical", True),
    "sync boolean count": lambda p: next(iter(p["sync_check"].values())).__setitem__("d2h_copies", False),
    "parity boolean count": lambda p: p["kernel_parity"]["case_kinds"].__setitem__("packed_tables", True)
    or p["kernel_parity"].__setitem__("cases", sum(p["kernel_parity"]["case_kinds"].values())),
    "protocol boolean seed": lambda p: p["protocol"].__setitem__("seed", True) or p.__setitem__(
        "run_config_hash", b.run_config_hash(p["shakedown"], p["protocol"], p["meta"]["code_sha256"],
                                             p["meta"]["git_sha"], p["meta"]["environment"])),
    "part c layer count": lambda p: _first_model(p).__setitem__("linear_parameters", 1),
    "part c logits error keys": lambda p: _first_model(p)["logits_rel_diff_vs_bf16"].pop("c6_02_none"),
}


def _resummarize(p: dict) -> None:
    """Recompute every derived summary, Pareto and caveat after a raw-record change (so only the raw check fails)."""
    for row in p["part_c"]["models"].values():
        row["summary"] = b.summarize_part_c_model(row)
    p["pareto"] = b.pareto(p["part_c"])
    p["summary"] = b.summarize(p)
    p["caveats"] = b.caveats(p)


def _truncate_compiled_means(p: dict) -> None:
    gpu = _first_model(p)["timing"]["bf16_compiled"]["gpu"]
    gpu.update({k: v for k, v in b.e2e._stats(gpu["samples_ms"][:1]).items()})


def _contradictory_compiled_errors(p: dict) -> None:
    cp = _first_model(p)["bf16_compiled_parity"]
    cp.update(logits_max_abs_diff_vs_bf16_graph=0.0, logits_rel_diff_vs_bf16_graph=0.01)
    _first_model(p)["logits_rel_diff_vs_bf16"]["bf16_compiled"] = 0.01


# The four contradictions that the v2 validator accepted (audit, "Post-Remediation v2 Re-Check", finding 2), each
# with its dependent summaries recomputed, and the check that must now reject it.
ASTRA_V2_PROBES = {
    "truncated compiled round means": (_truncate_compiled_means, "GPU round count"),
    "negative compiled launch count": (lambda p: _first_model(p)["kernel_kinds_per_forward"]["bf16_compiled"]
                                       .__setitem__("other", -1.0), "bf16_compiled launch counts"),
    "zero max-abs with positive rel-L2": (_contradictory_compiled_errors, "compiled BF16 logits record"),
    "negative compiled rel error": (lambda p: _first_model(p)["logits_rel_diff_vs_bf16"].__setitem__(
        "bf16_compiled", -_first_model(p)["logits_rel_diff_vs_bf16"]["bf16_compiled"]), "logits errors"),
    "boolean seed": (lambda p: _seed0(p).__setitem__("seed", True), "identity"),
}


def _astra_probe(payload, name):
    fn, match = ASTRA_V2_PROBES[name]

    def mutate(p):
        fn(p)
        _resummarize(p)

    return _tamper(payload, mutate), match


@needs_fp4
@pytest.mark.parametrize("name", sorted(set(ASTRA_V2_PROBES) - {"boolean seed"}))  # the shakedown has no trained eval
def test_validator_rejects_the_astra_v2_probes_on_a_shakedown(shakedown, name):
    bad, match = _astra_probe(shakedown, name)
    with pytest.raises(ValueError, match=match):
        b.validate_results(bad)


@needs_fp4
@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_validator_rejects_tampering_of_a_shakedown(shakedown, name):
    with pytest.raises(ValueError):
        b.validate_results(_tamper(shakedown, TAMPERS[name]))


# ------------------------------------------------------------------ the committed canonical result
@pytest.fixture(scope="module")
def canonical():
    if CANONICAL_SHA256 is None:  # the producer commit, before its canonical run
        pytest.skip("the v3 canonical result is pinned in the commit that adds it")
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
    assert producer == CANONICAL_PRODUCER and payload["schema_version"] == "cycle6_c6_02.v3"
    assert set(payload["pareto"]) == {"50m", "120m"} and set(payload["part_c"]["models"]) == {"50m", "120m", "w2048_6l"}
    for preset, p in payload["pareto"].items():
        assert "bf16_compiled" in [x["method"] for x in p["points"]] and p["frontier_vs_bf16_graph"]
        trained = payload["part_c"]["compiled_bf16_trained_eval"][preset]
        assert [r["seed"] for r in trained["seeds"]] == [1, 2, 3] and trained["test_split_sha256"] == b.C6_01_TEST_SHA256
    for name, m in payload["part_c"]["models"].items():
        assert "bf16_compiled" in m["timing"] and m["bf16_compiled_parity"]["logits_finite"] is True
        assert m["summary"]["methods"]["c6_02_had_b16"]["speedup_vs_bf16_compiled"] > 0
    assert not any("near-0" in c for c in payload["caveats"])
    timing = [c for c in payload["caveats"] if c.startswith("Timing anomalies: ")]
    assert len(timing) == 1 and timing[0] == b._timing_caveat(payload) and "Maximum relative IQR" in timing[0]
    assert any(c.startswith("Fused cross-mix fidelity, w2048_6l") and "latency only" in c for c in payload["caveats"])


def _seed0(payload: dict) -> dict:
    return payload["part_c"]["compiled_bf16_trained_eval"]["50m"]["seeds"][0]


CANONICAL_TAMPERS = {
    "trained checkpoint hash": lambda p: _seed0(p).__setitem__("checkpoint_sha256", "0" * 64),
    "trained nll": lambda p: _seed0(p).__setitem__("test_nll", _seed0(p)["test_nll"] + 0.01),
    "trained windows": lambda p: _seed0(p).__setitem__("windows", 3905),
    "trained split": lambda p: p["part_c"]["compiled_bf16_trained_eval"]["120m"].__setitem__("test_split_sha256", "0" * 64),
    "pareto frontier vs graph": lambda p: p["pareto"]["50m"]["frontier_vs_bf16_graph"].pop(),
    "pareto compiled nll": lambda p: next(x for x in p["pareto"]["120m"]["points"] if x["method"] == "bf16_compiled")
    .__setitem__("test_nll", 9.0),
    "w2048 fidelity": lambda p: p["summary"]["part_c"]["w2048_6l"]["fused_cross_mix_fidelity"].__setitem__(
        "kron_hadamard_full", "logits_bitwise_equal_c6_01"),
}


@needs_fp4
@pytest.mark.parametrize("name", sorted(CANONICAL_TAMPERS))
def test_validator_rejects_canonical_only_tampering(canonical, name):
    with pytest.raises(ValueError):
        b.validate_results(_tamper(canonical, CANONICAL_TAMPERS[name]), canonical=True)


@needs_fp4
@pytest.mark.parametrize("name", sorted(ASTRA_V2_PROBES))
def test_validator_rejects_the_astra_v2_probes_on_the_canonical_result(canonical, name):
    bad, match = _astra_probe(canonical, name)
    with pytest.raises(ValueError, match=match):
        b.validate_results(bad, canonical=True)


@needs_fp4
def test_validator_rejects_the_result_commit_as_producer(canonical):
    """The v1 validator accepted the source-identical result commit as producer (with its recomputed config hash)."""
    log = c503._git("log", "-1", "--format=%H", "--", b.RESULT_REPO_PATH)
    result_commit = log.stdout.decode().strip()
    assert c503._is_hex(result_commit, 40) and result_commit != canonical["meta"]["git_sha"]
    assert b._matches_current_summary_code(result_commit)  # same summary bytes: only the tree check tells them apart

    def substitute(p):
        p["meta"]["git_sha"] = result_commit
        p["run_config_hash"] = b.run_config_hash(p["shakedown"], p["protocol"], p["meta"]["code_sha256"], result_commit,
                                                 p["meta"]["environment"])

    with pytest.raises(ValueError, match="already contains"):
        b.validate_results(_tamper(canonical, substitute), canonical=True)


@needs_fp4
def test_superseded_receipts(tmp_path):
    verified = b.verify_superseded_receipts()
    for commit in ("a716333", "3e46141"):  # the v1 and v2 result blobs in git history
        assert verified[f"{commit}:{b.RESULT_REPO_PATH}"] == "verified"
    assert [a["producer_git_sha"][:7] for a in b.SUPERSEDED_ATTEMPTS] == ["a4ebc54", "47cdedd", "411eaf0"]
    assert b.SUPERSEDED_ATTEMPTS[1]["receipts"][
        "superseded_canonical_47cdedd.json"] == "ceff189c3298905b355705dab89532d3a80bbeaf7343a7d27c804865f83949a2"
    assert b.SUPERSEDED_ATTEMPTS[2]["receipts"] == {
        "superseded_canonical_411eaf0.json": "91deb1d2adc3eab3d26bf848ec4c6193a54178bfb145b92770264113d348583d",
        "canonical_411eaf0.log": "57893b3350a35787dcf9fe02858d9d0eccb8caae411a3af870b62bb078efd13c"}
    assert "1.846x" in b.SUPERSEDED_ATTEMPTS[2]["outcome"]
    present = [k for k, v in verified.items() if v == "verified" and ":" not in k]
    assert set(present) <= {n for a in b.SUPERSEDED_ATTEMPTS for n in a["receipts"]}
    (tmp_path / "canonical_a4ebc54.log").write_text("tampered\n")
    with pytest.raises(ValueError, match="receipt"):
        b.verify_superseded_receipts(tmp_path)


@needs_fp4
@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_validator_rejects_tampering_of_the_canonical_result(canonical, name):
    if name == "canonical flag":
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
