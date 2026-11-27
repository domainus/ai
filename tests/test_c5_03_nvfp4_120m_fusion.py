"""Checks for Cycle 5 C5-03: the tuned NVFP4 activation quantizer and the C5-03 benchmark driver.

Kernel tests compare the C5-03 kernels bit for bit with the eager reference and with the embedded
pre-C5-03 C4-02 quantizer, including the ablation settings (no PDL, comparison-ladder E2M1) and the
rejected cooperative single-launch kernel. Driver tests cover the CLI guard, the validator (on a
quick run and with tampering probes) and, once committed, the canonical artifact.
"""

import ast
import copy
import json
import math
import re
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from research.quant_ptq import micro_rotated_ptq as q

ROOT = Path(__file__).resolve().parents[1]
CUDA = torch.cuda.is_available()
FP4 = CUDA and hasattr(F, "scaled_mm") and torch.cuda.get_device_capability()[0] >= 10
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")
needs_fp4 = pytest.mark.skipif(not FP4, reason="needs native NVFP4 scaled_mm (sm_100+)")

if CUDA:
    from research.quant_ptq import bench_c5_03_nvfp4_120m_fusion as b
    from research.quant_ptq import nvfp4_inference_blackwell as nb

ROTS = ("none", "hadamard_b16", "adaptive_butterfly_b16")


def _angles(K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand(K // 16, 4, 8, generator=g, device="cuda") * 2 * math.pi


def _same(a, r):
    return (torch.equal(a.packed.view(torch.uint8), r.packed.view(torch.uint8))
            and torch.equal(a.scale.view(torch.uint8), r.scale.view(torch.uint8))
            and torch.equal(a.global_scale.reshape(()).view(torch.int32), r.global_scale.reshape(()).view(torch.int32)))


def _all_paths(x, rotation, angles, static=None):
    """(reference, C5-03, C5-03 ladder E2M1, C5-03 without PDL, C4-02) operands."""
    tables = nb.butterfly_tables(angles) if angles is not None else None
    ref = nb.quantize_activation_ref(x, rotation, angles, static)
    out = [nb.quantize_activation(x, rotation, tables, static)]
    with b.options(native_cvt=False):
        out.append(nb.quantize_activation(x, rotation, tables, static))
    with b.options(pdl=False):
        out.append(nb.quantize_activation(x, rotation, tables, static))
    out.append(b.c4_02_quantize_activation(x, rotation, tables, static))
    return ref, out


# ------------------------------------------------------------------ bit-exact parity of the C5-03 kernels
@needs_cuda
@pytest.mark.parametrize("rotation", ROTS)
@pytest.mark.parametrize("M,K", [(1, 16), (3, 48), (16, 4096), (33, 1408), (127, 96), (1000, 512), (257, 2048),
                                 (1024, 1408), (7, 8192)])
def test_small_medium_and_unaligned_shapes_are_bit_exact(M, K, rotation):
    g = torch.Generator(device="cuda").manual_seed(M * 13 + K)
    angles = _angles(K, M) if rotation == "adaptive_butterfly_b16" else None
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = (torch.randn(M, K, generator=g, device="cuda") * 3).to(dtype)
        x[:, :16] *= 40
        ref, paths = _all_paths(x, rotation, angles)
        assert all(_same(p, ref) for p in paths)
        static = (ref.global_scale * 0.5).clone()
        ref, paths = _all_paths(x, rotation, angles, static)
        assert all(_same(p, ref) for p in paths)


@needs_cuda
@pytest.mark.parametrize("rotation", ROTS)
def test_native_e2m1_conversion_on_midpoints_and_special_values(rotation):
    """Every E2M1 midpoint and its float neighbours, +-0, subnormals, +-Inf and both NaN signs."""
    mids = torch.tensor(q._E2M1_MIDPOINTS + q.E2M1_VALUES, device="cuda")
    near = torch.cat([mids, torch.nextafter(mids, torch.full_like(mids, 10)),
                      torch.nextafter(mids, torch.zeros_like(mids))])
    special = torch.tensor([0.0, -0.0, 1e-45, -1e-45, 1.2e-38, -1.2e-38, 6.5, 1e30, float("inf"), float("-inf"),
                            float("nan"), -float("nan")], device="cuda")
    values = torch.cat([near, -near, special])
    x = torch.zeros(40, 256, device="cuda")
    x.view(-1)[: values.numel()] = values
    # A fixed scale of 1 / 6: every block scale is 1 (448 * 6 * gs >= amax for the finite blocks), so the
    # midpoints reach the E2M1 conversion unchanged in the identity case.
    angles = _angles(256, 3) if rotation == "adaptive_butterfly_b16" else None
    for static in (None, torch.tensor(1 / 6, device="cuda"), torch.tensor(1e-3, device="cuda")):
        ref, paths = _all_paths(x, rotation, angles, static)
        assert all(_same(p, ref) for p in paths)
    finite = torch.nan_to_num(x, nan=0.0, posinf=7.0, neginf=-7.0)
    for static in (None, torch.tensor(1 / 6, device="cuda")):
        ref, paths = _all_paths(finite, rotation, angles, static)
        assert all(_same(p, ref) for p in paths)


@needs_cuda
def test_dirty_buffers_padding_and_bounded_partials_through_the_public_launchers():
    for M, K in ((37, 80), (1024, 1408), (129, 4096)):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        pr, pc = nb.scale_layout(M, K)
        packed = torch.full((M, K // 2), 0xAB, dtype=torch.uint8, device="cuda")
        scale = torch.full((pr * pc,), 0xCD, dtype=torch.uint8, device="cuda")
        partials = torch.full((nb.MAX_PARTIALS,), float("nan"), device="cuda")
        gs = torch.empty((), device="cuda")
        n = nb.launch_amax(x, partials, "hadamard_b16")
        assert 1 <= n <= nb.PASS1_PROGRAMS <= nb.MAX_PARTIALS
        assert torch.isnan(partials[n:]).all()  # pass 1 writes exactly its n partials
        nb.launch_quant(x, packed, scale, gs, rotation="hadamard_b16", workspace=partials, n_part=n)
        ref = nb.quantize_activation_ref(x, "hadamard_b16")
        assert torch.equal(scale, ref.scale.view(torch.uint8)) and torch.equal(packed, ref.packed.view(torch.uint8))
        assert torch.equal(gs.view(torch.int32), ref.global_scale.view(torch.int32))
        perm = q.to_blocked(torch.arange(pr * pc, device="cuda").view(pr, pc))
        logical = torch.empty_like(scale)
        logical[perm] = scale
        logical = logical.view(pr, pc)
        assert (logical[M:] == 0).all() and (logical[:, K // 16:] == 0).all()


@needs_cuda
def test_shape_aware_plan():
    for M, K in ((16, 4096), (1024, 512), (1024, 768), (1024, 1408), (1024, 2048), (2048, 8192)):
        _, pc = nb.scale_layout(M, K)
        for rotation in ROTS:
            plan = nb.quantizer_plan(M, K, rotation)
            bm, bg, _ = plan["pass2_tile"]
            assert 128 % bm == 0 and pc % bg == 0, (M, K, rotation, plan)  # no half-empty group tiles
            assert plan["launches"] == 2 and plan["pass1_programs"] <= nb.PASS1_PROGRAMS and plan["pdl"]
        assert nb.quantizer_plan(M, K, "hadamard_b16", static=True)["launches"] == 1
    # The C4-02 caveat shape: 16-group tiles do not divide its 88 scale columns.
    assert nb.quantizer_plan(1024, 1408, "none")["pass2_tile"][1] == 8


# ------------------------------------------------------------------ sync-free, graphs, API
@needs_fp4
@pytest.mark.parametrize("rotation", ROTS)
@pytest.mark.parametrize("static", (False, True))
@pytest.mark.parametrize("opts", ({}, {"pdl": False}, {"native_cvt": False}))
def test_c5_03_graph_replays_are_bitwise_and_sync_free(rotation, static, opts):
    W = torch.randn(512, 1408, device="cuda") * 0.02
    angles = _angles(1408, 1) if rotation == "adaptive_butterfly_b16" else None
    layer = nb.NVFP4Linear.from_weight(W, rotation, angles,
                                       act_global_scale=torch.tensor(0.004, device="cuda") if static else None)
    x = torch.randn(1024, 1408, device="cuda", dtype=torch.bfloat16)
    with b.options(**opts):
        layer(x)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            layer(x)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            layer(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = layer(x)
    g = torch.Generator(device="cuda").manual_seed(11)
    for scale in (1.0, 1e-3, 50.0, 1.0, 1e4):
        x.copy_(torch.randn(1024, 1408, generator=g, device="cuda") * scale)
        graph.replay()
        want = nb.nvfp4_scaled_mm(nb.quantize_activation_ref(x, rotation, angles, layer.act_global_scale),
                                  layer.w_packed, layer.w_scale, layer.w_global)
        assert torch.equal(y, want)


@needs_fp4
def test_profiler_sees_the_quantizer_kernels_and_no_device_to_host_copy():
    layer = nb.NVFP4Linear.from_weight(torch.randn(512, 1408, device="cuda") * 0.02, "hadamard_b16")
    x = torch.randn(1024, 1408, device="cuda", dtype=torch.bfloat16)
    layer(x)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            layer(x)
    torch.cuda.synchronize()
    names = [e.name for e in prof.events()]
    assert not [n for n in names if "DtoH" in n or "local_scalar_dense" in n]
    assert sum("_amax_kernel" in n for n in names) >= 3 and sum("_quant_kernel" in n for n in names) >= 3


@needs_cuda
def test_unwarmed_capture_is_refused_for_the_pdl_path():
    x = torch.randn(5, 4096 + 128, device="cuda")  # a shape no other test compiled
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="Warm up"):
        with torch.cuda.graph(graph):
            nb.quantize_activation(x, "hadamard_b16")


@needs_cuda
def test_public_api_is_backward_compatible():
    for name in ("launch_amax", "launch_quant", "quantize_activation", "quantize_activation_ref", "NVFP4Linear",
                 "scale_layout", "butterfly_tables", "dequantize", "dequantized_weight", "nvfp4_scaled_mm",
                 "nvfp4_scaled_mm_public", "nvfp4_encode_ref", "ROTATIONS", "MAX_PARTIALS"):
        assert name in nb.__all__ and hasattr(nb, name)
    assert nb.MAX_PARTIALS == 512 and nb.OPTIONS == {"pdl": True, "native_cvt": True}
    assert "_amax_kernel" in nb._amax_kernel.fn.__name__ and "_quant_kernel" in nb._quant_kernel.fn.__name__


# ------------------------------------------------------------------ embedded C4-02 baseline and the cooperative kernel
LEGACY_NAMES = ("_tile", "_nanmax", "_pairs", "_unpairs", "_butterfly_stage", "_load_rotated", "_amax_kernel",
                "_quant_kernel", "_launch", "_amax", "_quant", "_quantize")


@needs_cuda
def test_embedded_c4_02_quantizer_is_verbatim_from_the_c4_02_commit():
    blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{b.LEGACY_GIT_SHA}:{b.LEGACY_REPO_PATH}"],
                          capture_output=True, check=True).stdout.decode()
    pattern = re.compile(r"\b(" + "|".join(sorted((*LEGACY_NAMES, "_PLANS"), key=len, reverse=True)) + r")\b")

    def segments(text, names):
        tree = ast.parse(text)
        out = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                decorators = "".join("@" + ast.get_source_segment(text, d) + "\n" for d in node.decorator_list)
                out[node.name] = decorators + ast.get_source_segment(text, node)
        return out

    original = segments(blob, LEGACY_NAMES)
    embedded = segments(Path(b.__file__).read_text(), {"_c4_02" + n for n in LEGACY_NAMES})
    assert len(original) == len(LEGACY_NAMES) == len(embedded)
    for name, text in original.items():
        assert embedded["_c4_02" + name] == pattern.sub(lambda m: "_c4_02" + m.group(1), text), name
    assert b.MAX_PARTIALS == 512


@needs_fp4
@pytest.mark.parametrize("variant", ("none", "had_b16", "bfly_b16", "had_b16_static_gs"))
def test_c4_02_wrapper_and_c5_03_layer_give_the_same_gemm_output(variant):
    W = torch.randn(384, 768, device="cuda") * 0.02
    x = torch.randn(3, 37, 768, device="cuda", dtype=torch.bfloat16)
    static = nb.quantize_activation(x.reshape(-1, 768), "hadamard_b16").global_scale.clone()
    layer = b._layers_for(W, _angles(768, 2), static)[variant]
    assert torch.equal(layer(x), b.C4_02Linear(layer)(x))
    assert b.C4_02Linear(layer)(x[:, :0]).shape == (3, 0, 384)


@needs_fp4
def test_cooperative_single_launch_is_exact_over_1000_replays_and_resets_its_barrier():
    torch.manual_seed(0)
    layer = nb.NVFP4Linear.from_weight(torch.randn(512, 1408, device="cuda") * 0.02, "hadamard_b16")
    coop = b.CoopLinear(layer)
    x = torch.randn(1024, 1408, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(coop(x), layer(x))
    grid = b.coop_grid(x, "hadamard_b16")
    assert grid["programs"] <= grid["resident_capacity"] and grid["max_active_blocks_per_sm"] >= 2
    out = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        coop(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out["y"] = coop(x)
    g = torch.Generator(device="cuda").manual_seed(4)
    exact = 0
    torch.cuda.set_sync_debug_mode("error")
    try:
        for i in range(1000):
            x.copy_(torch.randn(1024, 1408, generator=g, device="cuda") * (0.01 + 10.0 * (i % 7)))
            graph.replay()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for i in range(50):  # the last replay plus fresh checks against the two-pass kernel
        x.copy_(torch.randn(1024, 1408, generator=g, device="cuda"))
        graph.replay()
        exact += bool(torch.equal(out["y"], layer(x)))
    assert exact == 50
    assert b._COOP[("sync", x.device)][0].item() == 0  # arrival counter auto-reset


# ------------------------------------------------------------------ statistics
@needs_cuda
def test_paired_statistics_and_holm():
    diffs = [0.1, 0.2, 0.15]
    from research.quant_ptq import bench_isoloss_ptq as c4
    stats = c4.paired_t_df2(diffs)
    mean = sum(diffs) / 3
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / 2)
    tval = mean / (sd / math.sqrt(3))
    assert stats["p_two_sided"] == pytest.approx(1 - abs(tval) / math.sqrt(tval * tval + 2), rel=1e-12)
    assert b.c502.sign_flip_p(diffs) == b.SIGN_FLIP_MIN_P == 0.25
    assert c4.holm_bonferroni([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


# ------------------------------------------------------------------ CLI guard
@needs_cuda
def test_cli_guard(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("run should not be called")

    monkeypatch.setattr(b, "run", fail)
    args = b.parse_args([])
    assert args.canonical and args.output == b.CANONICAL_OUTPUT and args.checkpoint_dir == b.CANONICAL_CHECKPOINT_DIR
    assert b.CANONICAL_OUTPUT == ROOT / "results" / "cycle5_c5_03_nvfp4_120m_fusion.json"
    assert b.parse_args(["--data-dir", "/x"]).canonical  # the data location does not change the protocol
    assert not b.parse_args(["--quick"]).canonical and b.parse_args(["--quick"]).output == b.QUICK_OUTPUT
    assert not b.parse_args(["--parts", "ABC", "--output", str(tmp_path / "x.json")]).canonical  # raw flag
    other = tmp_path / "other" / "results" / b.CANONICAL_OUTPUT.name
    alias = tmp_path / "alias.json"
    alias.symlink_to(other)
    renamed = tmp_path / "cycle5_c5_03_copy.json"
    for argv in (
        ["--quick", "--output", str(b.CANONICAL_OUTPUT)],
        ["--quick", "--output", str(renamed)],
        ["--parts", "AB"],
        ["--parts", "AB", "--output", str(b.CANONICAL_OUTPUT)],
        ["--seeds", "1", "2", "4"],
        ["--max-eval-windows", "8"],
        ["--device", "cpu", "--output", str(b.CANONICAL_OUTPUT)],
        ["--no-verify", "--output", str(b.CANONICAL_OUTPUT)],
        ["--output", str(other)],
        ["--output", str(alias)],
        ["--qui"],  # abbreviations are not accepted
        ["--parts", "AAB", "--output", str(tmp_path / "y.json")],
        ["--seeds", "1", "1", "--output", str(tmp_path / "y.json")],
    ):
        with pytest.raises(SystemExit) as exc:
            b.main(argv)
        assert exc.value.code == 2, argv


@needs_cuda
def test_canonical_host_guard():
    device = torch.device("cuda")
    out = b.CANONICAL_OUTPUT
    with pytest.raises(RuntimeError, match="clean"):
        b.require_canonical_host(device, {"git_sha": "0" * 40, "git_dirty": True}, out, True)
    with pytest.raises(RuntimeError, match="Blackwell"):
        b.require_canonical_host(torch.device("cpu"), {"git_sha": "0" * 40, "git_dirty": False}, out, True)
    b.require_canonical_host(device, {"git_sha": None, "git_dirty": True}, Path("/tmp/q.json"), False)


# ------------------------------------------------------------------ quick run and validator probes
@pytest.fixture(scope="module")
def quick(tmp_path_factory):
    if not FP4:
        pytest.skip("needs native NVFP4 scaled_mm")
    out = tmp_path_factory.mktemp("c5_03") / "q.json"
    payload = b.main(["--quick", "--silent", "--output", str(out)])
    assert json.loads(out.read_text()) == payload
    return payload


def _get(p, path):
    for k in path:
        p = p[k]
    return p


def _set(p, path, value):
    _get(p, path[:-1])[path[-1]] = value


def _rejects(payload, edit, match, **kw):
    p = copy.deepcopy(payload)
    edit(p)
    with pytest.raises(ValueError, match=match):
        b.validate_results(p, **kw)


@needs_fp4
def test_quick_run_is_valid_and_complete(quick):
    assert b.validate_results(quick) == {"certified_canonical": False, "parts": "ABC"}
    c = quick["part_c"]
    assert quick["meta"]["splits_opened"] == ["train", "val", "test"] and c["splits_opened_in_phase1"] == ["train", "val"]
    assert c["scoring_ledger"]["test_scored_only_after_freeze"] is True
    assert len(c["seeds"]) == 3 and all(len(s["grid"]) == 29 for s in c["seeds"])
    hw = c["analysis"]["hardware"]
    assert hw["fp4_gemms_per_forward"] == [10] and hw["decode_exact_matrices"] == hw["decode_checked_matrices"]
    assert c["analysis"]["hardware_vs_emulation"]["passed"]
    assert [x["name"] for x in c["analysis"]["contrasts"]] == [x[0] for x in b.CONTRASTS]
    for row in quick["part_a"]["shapes"]:
        assert all(row["checks"].values())
        for mode in ("eager", "cuda_graph"):
            for t in row[mode].values():
                assert len(t["gpu"]["event_samples_ms"]) == t["gpu"]["rounds"] == len(row[f"{mode}_williams_orders"]
                                                                                     if mode == "eager" else
                                                                                     row["cuda_graph_williams_orders"])
    for model in quick["part_b"]["models"].values():
        assert all(v for k, v in model["checks"].items() if k.endswith("bitwise"))
    with pytest.raises(ValueError, match="canonical"):
        b.validate_results(quick, canonical=True)


@needs_fp4
def test_validator_rejects_tampering(quick):
    p = quick
    seed0 = ["part_c", "seeds", 0]
    grid0 = [*seed0, "grid"]
    r_hw = next(i for i, r in enumerate(p["part_c"]["seeds"][0]["grid"]) if r["mode"] == "w4a4_hw_dynamic")
    r_fq = next(i for i, r in enumerate(p["part_c"]["seeds"][0]["grid"]) if r["mode"] == "w4a16_fakequant")
    _rejects(p, lambda x: _set(x, ["schema_version"], "cycle5_c5_03.v0"), "schema")
    _rejects(p, lambda x: _set(x, ["meta", "git_dirty"], 1), "git_dirty")
    _rejects(p, lambda x: _set(x, ["meta", "code_sha256"], "0" * 64), "code_sha256")
    _rejects(p, lambda x: _set(x, ["meta", "argv"], ["--quick", "--seeds", "1", "2", "3", "4"]), "argv")
    _rejects(p, lambda x: _set(x, ["protocol", "part_c", "steps"], 9), "argv|run_config_hash")
    _rejects(p, lambda x: _set(x, ["kernel_parity", "passed"], False), "kernel parity")
    _rejects(p, lambda x: _set(x, ["part_a", "shapes", 0, "checks", "none_c5_03_output_equals_c4_02_bitwise"], False),
             "bit-exactness")
    _rejects(p, lambda x: _get(x, ["part_a", "shapes", 0, "eager_williams_orders"]).reverse(), "Williams")
    _rejects(p, lambda x: _set(x, ["part_a", "shapes", 0, "cuda_graph", "c5_03_none", "gpu", "samples_ms", 0], 9.0),
             "event_samples_ms|samples_ms")
    _rejects(p, lambda x: _set(x, ["part_a", "shapes", 0, "summary", "cuda_graph", "variants", "none",
                                   "c5_03_beats_bf16"], True) if not p["part_a"]["shapes"][0]["summary"]["cuda_graph"][
        "variants"]["none"]["c5_03_beats_bf16"] else _set(x, ["part_a", "shapes", 0, "summary", "cuda_graph",
                                                              "variants", "none", "c5_03_beats_bf16"], False), "summary")
    _rejects(p, lambda x: _set(x, ["part_a", "ablation", 0, "coop", "graph_replays_bitwise_exact"], 1), "cooperative")
    _rejects(p, lambda x: _set(x, ["part_b", "models", "tiny", "checks", "none_c5_03_logits_equal_c4_02_bitwise"],
                               False), "logits")
    _rejects(p, lambda x: _set(x, [*grid0, r_hw, "test_nll"], _get(x, [*grid0, r_hw, "test_nll"]) + 0.01), "test_ppl")
    _rejects(p, lambda x: _set(x, [*grid0, r_hw, "fp4_gemms_per_forward"], 9), "native FP4 GEMM")
    _rejects(p, lambda x: _set(x, [*grid0, r_hw, "bf16_block_linears"], 1), "native FP4 GEMM")
    _rejects(p, lambda x: _set(x, [*grid0, r_fq, "hw_weights_decode_exact_matrices"], 9), "decode exactly")
    _rejects(p, lambda x: _set(x, [*grid0, r_fq, "effective_weights_sha256_test"], "0" * 64), "decode exactly")
    _rejects(p, lambda x: _set(x, [*seed0, "baseline_test", "tokens"], 2048), "coverage")
    _rejects(p, lambda x: _set(x, [*seed0, "training", "steps"], 7), "training record")
    _rejects(p, lambda x: _set(x, ["part_c", "scoring_ledger", "test_scorings"], 1), "scoring ledger")
    _rejects(p, lambda x: _set(x, ["part_c", "splits_opened_in_phase1"], ["train", "val", "test"]), "Phase 1")
    _rejects(p, lambda x: _set(x, ["meta", "splits_opened"], ["test", "train", "val"]), "splits_opened")
    _rejects(p, lambda x: _set(x, [*seed0, "awq_selection", 0, "selected_alpha"], 0.75 if _get(
        x, [*seed0, "awq_selection", 0, "selected_alpha"]) != 0.75 else 0.25), "selected alpha|alpha differs")
    _rejects(p, lambda x: _set(x, ["part_c", "analysis", "contrasts", 0, "test", "p_holm"], 0.0), "analysis")
    _rejects(p, lambda x: _get(x, ["caveats"]).pop(0), "caveats")

    def fits(x):
        return _get(x, [*seed0, "butterfly_fits", "adaptive_butterfly_b16_gptq", "qkv"])

    def worse(x):
        f = fits(x)
        f["weight_sq_error_fitted"] = f["weight_sq_error_hadamard_start"] * 1.5
        f["relative_error_reduction"] = 1.0 - f["weight_sq_error_fitted"] / f["weight_sq_error_hadamard_start"]
    _rejects(p, worse, "0 <= fitted <= start")

    def unimproved_but_lower(x):
        f = fits(x)
        f["layers_improved"] = 0
        f["weight_sq_error_fitted"] = f["weight_sq_error_hadamard_start"] * 0.5
        f["relative_error_reduction"] = 0.5
    _rejects(p, unimproved_but_lower, "layers_improved == 0")

    def tamper_inherited(x):
        for s in x["part_c"]["seeds"]:
            for r in s["grid"]:
                if r["inherited"]:
                    r["seconds_val"] += 1.0
                    return
        for s in x["part_c"]["seeds"]:  # no alpha = 0 choice in this quick run: claim one
            for r in s["grid"]:
                if r["awq"]:
                    r["inherited"] = True
                    return
    _rejects(p, tamper_inherited, "twin|inherited|alpha")


# ------------------------------------------------------------------ the committed canonical artifact
@needs_cuda
def test_committed_canonical_artifact():
    if not b.CANONICAL_OUTPUT.exists():
        pytest.skip("canonical result not committed yet")
    payload = json.loads(b.CANONICAL_OUTPUT.read_text())
    assert b.validate_results(payload, canonical=True) == {"certified_canonical": True, "parts": "ABC"}
    assert payload["canonical"] and not payload["quick"] and payload["protocol"] == b.CANONICAL
    assert payload["meta"]["git_dirty"] is False and payload["meta"]["environment"]["capability"] >= [12, 0]
    assert payload["kernel_parity"]["cases"] == 279 and payload["kernel_parity"]["passed"]
    assert [[r["m"], r["n"], r["k"]] for r in payload["part_a"]["shapes"]] == b.PART_A_SHAPES
    c = payload["part_c"]
    assert [s["seed"] for s in c["seeds"]] == [1, 2, 3]
    for s in c["seeds"]:
        assert s["training"]["tokens"] == 20_971_520 and s["training"]["steps"] == 640
        assert (s["baseline_val"]["tokens"], s["baseline_test"]["tokens"]) == (b.VAL_TOKENS, b.TEST_TOKENS)
        assert len(s["grid"]) == 29
    hw = c["analysis"]["hardware"]
    assert hw["fp4_gemms_per_forward"] == [60] and hw["fp4_gemms_per_forward_emulated"] == [0]
    assert hw["decode_exact_matrices"] == hw["decode_checked_matrices"] == 60 * hw["decode_checked_rows"]
    parity = c["analysis"]["hardware_vs_emulation"]
    assert parity["passed"] and len(parity["rows"]) == 6 and parity["max_abs_nll_difference"] <= 1e-3
    assert c["scoring_ledger"]["test_scored_only_after_freeze"]
    assert payload["meta"]["splits_opened"] == ["train", "val", "test"]
