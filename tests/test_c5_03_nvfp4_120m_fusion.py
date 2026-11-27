"""Checks for Cycle 5 C5-03: the tuned NVFP4 activation quantizer and the C5-03 benchmark driver.

Kernel tests compare the C5-03 kernels bit for bit with the eager reference and with the embedded
pre-C5-03 C4-02 quantizer, including the ablation settings (no PDL, comparison-ladder E2M1) and the
rejected cooperative single-launch kernel. Driver tests cover the CLI guard, the validator (on a
quick run and with tampering probes) and, once committed, the canonical artifact.

The committed canonical artifact is a ``cycle5_c5_03.v2`` postprocess of the ``cycle5_c5_03.v1`` result
of 726b20c (measured by 38cef48). Its tests replay the GPT 6.2 Astra audit tampering cases (findings 1-2),
the launcher contract (finding 3), the integer-view bitwise checks (finding 5), the exclusive atomic
writer and the postprocess provenance.
"""

import ast
import copy
import hashlib
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


NONCANONICAL = {"certified_canonical": False, "parts": "ABC", "postprocessed": False}
CERTIFIED = {"certified_canonical": True, "parts": "ABC", "postprocessed": True}
SOURCE_COMMIT = "726b20c9590aba911361e300e7d2079c7bb55cfa"
SOURCE_SHA256 = "9ed6c9fd3879566f78e943f5aa54657b1f0e1c27e543e3e93f11103413da14ce"
PRODUCER_COMMIT = "38cef483071d649fc74161b5d00359c7c5ad0927"
AUDIT_COMMIT = "dbf82ab37a6aec6d76af1c8d9962fc7f5a885bd9"


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
    assert b.validate_results(quick) == NONCANONICAL
    assert quick["schema_version"] == "cycle5_c5_03.v2" and quick["postprocess"] is None
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
    notes = "\n".join(quick["caveats"])
    assert "8 train.bin sequences" in notes
    assert "tiny model trained 8 steps" in notes
    assert "evaluation limited to 16 windows per split" in notes
    assert "means of 3 flushed calls" in notes
    for forbidden in ("128 train.bin sequences", "640 steps", "120m", "38cef48", "4736807"):
        assert forbidden not in notes
    with pytest.raises(ValueError, match="canonical"):
        b.validate_results(quick, canonical=True)


@needs_cuda
def test_caveats_bind_study_history_to_actual_source_and_producer():
    fresh = {
        "canonical": True,
        "postprocess": None,
        "meta": {"git_sha": "a" * 40, "environment": {"device": "NVIDIA GeForce RTX 5080", "capability": [12, 0]}},
        "protocol": b.CANONICAL,
        "data": {"source": {"name": "fineweb-edu"}, "encoding": "gpt2"},
        "part_a": None,
        "part_b": None,
        "part_c": None,
    }
    history = next(n for n in b.caveats(fresh) if n.startswith("Study history:"))
    assert "aaaaaaa" in history
    assert "rerun committed at 726b20c" not in history
    assert "measured by 38cef48" not in history


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
    _rejects(p, lambda x: _set(x, ["postprocess"], {}), "postprocess keys")
    _rejects(p, lambda x: x.pop("postprocess"), "top-level keys")
    _rejects(p, lambda x: _set(x, ["part_b", "models", "tiny", "linear_parameters"], 0), "linear_parameters")
    _rejects(p, lambda x: _set(x, ["part_c", "calibration", "starts"], [-1] * 8), "calibration starts")

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

    qprof = ["part_a", "shapes", 0, "cuda_graph_kernel_profile_us"]

    def qgemm(x, method):
        return next(k for k in _get(x, [*qprof, method]) if b._is_fp4_gemm_kernel(k))

    def replace_kernel(x, method, old, new):
        k = _get(x, [*qprof, method])
        k[new] = k.pop(old)

    _rejects(p, lambda x: replace_kernel(x, "c5_03_had_b16", qgemm(x, "c5_03_had_b16"),
                                         "nvjet_tst_128x128_64x5_2x1_v_bz_TNT"), "no FP4 GEMM")
    _rejects(p, lambda x: replace_kernel(x, "nvfp4_gemm_only", qgemm(x, "nvfp4_gemm_only"),
                                         "cutlass3x_sm120_tensorop_s16816gemm_bf16_bf16_f32_128x128x64_0_tnn"),
             "no FP4 GEMM")
    _rejects(p, lambda x: _get(x, [*qprof, "c5_03_had_b16"]).__setitem__(
        "nvjet_tst_128x128_64x5_2x1_v_bz_TNT", 1.0), "non-FP4 GEMM")
    _rejects(p, lambda x: replace_kernel(x, "c5_03_had_b16", "_amax_kernel", "other_amax_kernel"), "exact quantizer")
    _rejects(p, lambda x: replace_kernel(x, "c5_03_had_b16", "_amax_kernel", "unrelated_kernel"), "exact quantizer")
    _rejects(p, lambda x: replace_kernel(x, "c5_03_had_b16", "_quant_kernel", "other_quant_kernel"), "exact quantizer")
    _rejects(p, lambda x: replace_kernel(x, "c5_03_had_b16", "_quant_kernel", "unrelated_kernel"), "exact quantizer")
    _rejects(p, lambda x: replace_kernel(x, "c4_02_had_b16", "_c4_02_amax_kernel", "other_c4_02_amax_kernel"),
             "exact quantizer")
    _rejects(p, lambda x: _get(x, [*qprof, "nvfp4_gemm_only"]).__setitem__("other_quant_kernel", 1.0),
             "runs no quantizer")


# ------------------------------------------------------------------ finding 3: public launcher buffer contracts
def _launch_buffers(M, K, dynamic=True):
    pr, pc = nb.scale_layout(M, K)
    return {"packed": torch.empty((M, K // 2), dtype=torch.uint8, device="cuda"),
            "scale": torch.empty(pr * pc, dtype=torch.uint8, device="cuda"),
            "global_scale": torch.empty((), device="cuda"),
            "workspace": torch.empty(nb.MAX_PARTIALS, device="cuda") if dynamic else None}


@needs_cuda
def test_launch_amax_validates_its_workspace():
    x = torch.randn(129, 768, device="cuda", dtype=torch.bfloat16)
    n = nb._n_partials(129, 768, "hadamard_b16")
    assert nb.launch_amax(x, torch.empty(n, device="cuda"), "hadamard_b16") == n  # exactly n floats suffice
    ws = torch.empty(nb.MAX_PARTIALS, device="cuda")
    for bad, match in ((ws.double(), "dtype"), (ws.half(), "dtype"), (ws.cpu(), "must be on"),
                       (ws.view(2, -1), "1-D"), (torch.empty(2 * nb.MAX_PARTIALS, device="cuda")[::2], "contiguous"),
                       (torch.empty(n - 1, device="cuda"), f"at least {n}"), (ws.tolist(), "tensor")):
        with pytest.raises(ValueError, match=match):
            nb.launch_amax(x, bad, "hadamard_b16")


@needs_cuda
def test_launch_quant_refuses_a_dynamic_call_without_its_partial_count():
    """Audit finding 3: a populated workspace with the default n_part = 0 used to give scale 1.0 silently."""
    for rotation in ROTS:
        angles = _angles(768, 5) if rotation == "adaptive_butterfly_b16" else None
        tables = nb.butterfly_tables(angles) if angles is not None else None
        x = torch.randn(129, 768, device="cuda", dtype=torch.bfloat16)
        buf = _launch_buffers(129, 768)
        n = nb.launch_amax(x, buf["workspace"], rotation, tables)
        with pytest.raises(ValueError, match="n_part"):
            nb.launch_quant(x, buf["packed"], buf["scale"], buf["global_scale"], rotation=rotation, tables=tables,
                            workspace=buf["workspace"])
        for bad in (-1, 0, n - 1, n + 1, nb.MAX_PARTIALS + 1):
            with pytest.raises(ValueError, match="n_part"):
                nb.launch_quant(x, buf["packed"], buf["scale"], buf["global_scale"], rotation=rotation,
                                tables=tables, workspace=buf["workspace"], n_part=bad)
        with pytest.raises(ValueError, match="at least"):  # workspace shorter than pass-1 partials
            nb.launch_quant(x, buf["packed"], buf["scale"], buf["global_scale"], rotation=rotation, tables=tables,
                            workspace=buf["workspace"][: n - 1], n_part=n)
        for bad in (True, 1.0, torch.tensor(n)):
            with pytest.raises(ValueError, match="n_part must be an int"):
                nb.launch_quant(x, buf["packed"], buf["scale"], buf["global_scale"], rotation=rotation,
                                tables=tables, workspace=buf["workspace"], n_part=bad)
        nb.launch_quant(x, buf["packed"], buf["scale"], buf["global_scale"], rotation=rotation, tables=tables,
                        workspace=buf["workspace"], n_part=n)
        ref = nb.quantize_activation_ref(x, rotation, angles)
        assert torch.equal(buf["packed"], ref.packed.view(torch.uint8))
        assert torch.equal(buf["scale"], ref.scale.view(torch.uint8))
        assert torch.equal(buf["global_scale"].view(torch.int32), ref.global_scale.view(torch.int32))
        assert buf["global_scale"].item() != 1.0


@needs_cuda
def test_launch_quant_rejects_off_by_one_partial_counts_for_129x768_identity():
    """Astra v2 probe: at 129x768 identity, launch_amax returns 27; 26 and 28 must raise ValueError."""
    x = torch.ones((129, 768), device="cuda", dtype=torch.bfloat16)
    x[-1, -1] = 256
    buf = _launch_buffers(129, 768)
    buf["workspace"].fill_(65536.0)
    n = nb.launch_amax(x, buf["workspace"])
    assert n == 27
    for bad in (26, 28):
        with pytest.raises(ValueError, match="n_part"):
            nb.launch_quant(x, buf["packed"], buf["scale"], buf["global_scale"],
                            workspace=buf["workspace"], n_part=bad)
    nb.launch_quant(x, buf["packed"], buf["scale"], buf["global_scale"], workspace=buf["workspace"], n_part=n)
    ref = nb.quantize_activation_ref(x)
    assert torch.equal(buf["packed"], ref.packed.view(torch.uint8))
    assert torch.equal(buf["scale"], ref.scale.view(torch.uint8))
    assert torch.equal(buf["global_scale"].view(torch.int32), ref.global_scale.view(torch.int32))


@needs_cuda
def test_launch_quant_static_mode_takes_no_partial_count():
    x = torch.randn(37, 80, device="cuda")
    buf = _launch_buffers(37, 80, dynamic=False)
    gs = torch.tensor(0.01, device="cuda")
    with pytest.raises(ValueError, match="static launch_quant"):
        nb.launch_quant(x, buf["packed"], buf["scale"], gs, n_part=4)
    # Typed operand views and a (1,)-shaped scale are accepted.
    pr, pc = nb.scale_layout(37, 80)
    packed = torch.empty((37, 40), dtype=torch.float4_e2m1fn_x2, device="cuda")
    scale = torch.empty(pr * pc, dtype=torch.float8_e4m3fn, device="cuda")
    nb.launch_quant(x, packed, scale, gs.reshape(1))
    ref = nb.quantize_activation_ref(x, "none", None, gs)
    assert torch.equal(packed.view(torch.uint8), ref.packed.view(torch.uint8))
    assert torch.equal(scale.view(torch.uint8), ref.scale.view(torch.uint8))


@needs_cuda
def test_launch_quant_validates_dtype_device_shape_and_contiguity_of_every_buffer():
    M, K = 37, 80
    x = torch.randn(M, K, device="cuda")
    buf = _launch_buffers(M, K)
    n = nb.launch_amax(x, buf["workspace"])
    good = dict(buf, n_part=n)
    wide = torch.empty((M, K), dtype=torch.uint8, device="cuda")
    cases = (
        ("packed", buf["packed"].to(torch.int8), "packed must have dtype"),
        ("packed", buf["packed"].cpu(), "packed must be on"),
        ("packed", buf["packed"][:-1], "packed must have shape"),
        ("packed", buf["packed"].reshape(-1), "packed must have shape"),
        ("packed", wide[:, ::2], "packed must be contiguous"),
        ("scale", buf["scale"][:-1], "scale must have"),
        ("scale", buf["scale"].half(), "scale must have dtype"),
        ("scale", buf["scale"].cpu(), "scale must be on"),
        ("scale", torch.empty(2 * buf["scale"].numel(), dtype=torch.uint8, device="cuda")[::2], "contiguous"),
        ("global_scale", buf["global_scale"].half(), "global_scale must have dtype"),
        ("global_scale", torch.empty(2, device="cuda"), "global_scale must have 1"),
        ("global_scale", buf["global_scale"].cpu(), "global_scale must be on"),
        ("workspace", buf["workspace"].double(), "workspace must have dtype"),
        ("workspace", buf["workspace"].view(4, -1), "workspace must be 1-D"),
        ("workspace", torch.empty(2 * nb.MAX_PARTIALS, device="cuda")[::2], "workspace must be contiguous"),
        ("workspace", buf["workspace"].cpu(), "workspace must be on"),
    )
    for key, bad, match in cases:
        kw = dict(good, **{key: bad})
        with pytest.raises(ValueError, match=match):
            nb.launch_quant(x, kw["packed"], kw["scale"], kw["global_scale"], workspace=kw["workspace"],
                            n_part=kw["n_part"])


# ------------------------------------------------------------------ finding 5: integer-view bitwise comparisons
def _bits(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype).view(torch.float32)


@needs_cuda
def test_bitwise_equal_distinguishes_signed_zeros_and_nan_payloads():
    pos, neg = torch.tensor([0.0, 1.0]), torch.tensor([-0.0, 1.0])
    assert torch.equal(pos, neg) and not b._bitwise_equal(pos, neg)  # torch.equal ignores the sign of zero
    nan_a, nan_b = _bits([0x7FC00000]), _bits([0x7FC00001])
    assert not b._bitwise_equal(nan_a, nan_b) and b._bitwise_equal(nan_a, nan_a.clone())
    assert not torch.equal(nan_a, nan_a.clone())  # and torch.equal is never true for NaN
    for dtype in (torch.bfloat16, torch.float16, torch.float64):
        assert not b._bitwise_equal(pos.to(dtype), neg.to(dtype)) and b._bitwise_equal(neg.to(dtype), neg.to(dtype))
    assert not b._bitwise_equal(pos, pos.double()) and not b._bitwise_equal(pos, pos[:1])
    x = torch.randn(4, 6)
    assert b._bitwise_equal(x.t(), x.t().contiguous())  # strides do not matter, bytes do
    assert b._bitwise_equal(pos.cuda(), pos.cuda()) and not b._bitwise_equal(pos.cuda(), neg.cuda())


@needs_fp4
@pytest.mark.parametrize("variant", ("none", "had_b16", "bfly_b16", "had_b16_static_gs"))
def test_part_a_layer_outputs_are_bitwise_equal_as_integer_views(variant):
    W = torch.randn(512, 1408, device="cuda") * 0.02
    x = torch.randn(1024, 1408, device="cuda", dtype=torch.bfloat16)
    static = nb.quantize_activation(x, "hadamard_b16").global_scale.clone()
    layer = b._layers_for(W, _angles(1408, 7), static)[variant]
    new, old = layer(x), b.C4_02Linear(layer)(x)
    assert new.dtype == torch.bfloat16 and b._bitwise_equal(new, old)
    bumped = old.clone()
    bumped.view(torch.int16).view(-1)[0] ^= 1  # one ULP of one output
    assert not b._bitwise_equal(new, bumped)


@needs_fp4
def test_part_b_transformer_logits_are_bitwise_equal_as_integer_views():
    from research.polar_opt.harness_tier2_lm import Transformer
    config = b.PART_B_CONFIGS["tiny"]
    torch.manual_seed(3)
    with torch.device("cuda"):
        model = Transformer(config).to(torch.bfloat16).eval()
    tokens = torch.randint(0, config.vocab_size, (2, config.sequence_length), device="cuda")
    linears = q.quantized_linears(model)
    for v in ("none", "had_b16", "bfly_b16"):
        rot = b.VARIANT_ROTATION[v]
        new, old = copy.deepcopy(model), copy.deepcopy(model)
        for n, m in linears.items():
            angles = _angles(m.in_features, 9) if rot == "adaptive_butterfly_b16" else None
            layer = nb.NVFP4Linear.from_weight(m.weight.float(), rot, angles)
            b.e2e._swap(new, {n: layer})
            b.e2e._swap(old, {n: b.C4_02Linear(layer)})
        with torch.inference_mode():
            assert b._bitwise_equal(new(tokens), old(tokens)), v


# ------------------------------------------------------------------ the exclusive atomic writer
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    monkeypatch.setattr(b, "ROOT", root)
    monkeypatch.setattr(b, "CANONICAL_OUTPUT", root / b.RESULT_REPO_PATH)
    return root


def _leftovers(directory):
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".tmp"))


@needs_cuda
def test_atomic_writer_round_trips_and_leaves_no_temp_file(sandbox):
    payload = {"a": [1, 2.5, None], "b": {"c": "x"}}
    assert b._write_json_atomic(b.CANONICAL_OUTPUT, payload, canonical_dest=True) == payload
    assert json.loads(b.CANONICAL_OUTPUT.read_text()) == payload
    free = sandbox.parent / "elsewhere" / "free.json"
    assert b._write_json_atomic(free, payload) == payload
    assert _leftovers(sandbox / "results") == [] and _leftovers(free.parent) == []


@needs_cuda
def test_atomic_writer_rejects_symlinks_and_other_canonical_destinations(sandbox, tmp_path):
    payload = {"x": 1}
    target = tmp_path / "outside.json"
    b.CANONICAL_OUTPUT.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        b._write_json_atomic(b.CANONICAL_OUTPUT, payload, canonical_dest=True)
    assert not target.exists()
    b.CANONICAL_OUTPUT.unlink()
    (sandbox / "results").rmdir()
    elsewhere = tmp_path / "other_results"
    elsewhere.mkdir()
    (sandbox / "results").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):  # a symlinked parent below ROOT
        b._write_json_atomic(b.CANONICAL_OUTPUT, payload, canonical_dest=True)
    with pytest.raises(ValueError, match="symlink"):  # a reserved name is guarded without canonical_dest too
        b._write_json_atomic(sandbox / "results" / "cycle5_c5_03_copy.json", payload)
    assert list(elsewhere.iterdir()) == []
    with pytest.raises(ValueError, match="outside"):
        b._write_json_atomic(tmp_path / "cycle5_c5_03_x.json", payload)
    with pytest.raises(ValueError, match="canonical result goes to"):
        b._write_json_atomic(tmp_path / "plain.json", payload, canonical_dest=True)
    assert not (tmp_path / "plain.json").exists()


@needs_cuda
def test_atomic_writer_keeps_the_old_file_and_no_temp_file_on_failure(sandbox, monkeypatch):
    out = b.CANONICAL_OUTPUT
    b._write_json_atomic(out, {"old": True}, canonical_dest=True)
    with monkeypatch.context() as m:
        m.setattr(b.e2e, "_dumps_json", lambda obj: '{"lossy": 1}')
        with pytest.raises(ValueError, match="round trip"):
            b._write_json_atomic(out, {"new": True}, canonical_dest=True)
    assert json.loads(out.read_text()) == {"old": True} and _leftovers(out.parent) == []
    with pytest.raises(ValueError, match="JSON compliant"):
        b._write_json_atomic(out, {"nan": float("nan")}, canonical_dest=True)
    assert json.loads(out.read_text()) == {"old": True} and _leftovers(out.parent) == []


@needs_cuda
def test_atomic_writer_creates_its_temp_file_exclusively(sandbox, monkeypatch):
    """The temp file is created with O_EXCL | O_NOFOLLOW inside one verified dir_fd."""
    import os
    opens, replaces = [], []
    real_open, real_replace = os.open, os.replace

    def spy_open(path, flags, mode=0o777, *, dir_fd=None):
        fd = real_open(path, flags, mode, dir_fd=dir_fd)
        opens.append((path, flags, mode, dir_fd, fd))
        return fd

    def spy_replace(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        replaces.append((src, dst, src_dir_fd, dst_dir_fd))
        return real_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(b.os, "open", spy_open)
    monkeypatch.setattr(b.os, "replace", spy_replace)
    b._write_json_atomic(b.CANONICAL_OUTPUT, {"x": 1}, canonical_dest=True)
    tmp_create = [o for o in opens if (o[1] & os.O_CREAT) and (o[1] & os.O_EXCL)]
    assert len(tmp_create) == 1
    tmp_name, flags, mode, create_dir_fd, _ = tmp_create[0]
    assert (flags & os.O_NOFOLLOW) and mode == 0o600 and create_dir_fd is not None
    assert tmp_name.startswith(f".{b.CANONICAL_OUTPUT.name}.") and tmp_name.endswith(".tmp")
    reread = [o for o in opens if o[0] == tmp_name and o[3] == create_dir_fd and not (o[1] & os.O_CREAT)]
    assert len(reread) == 1 and (reread[0][1] & os.O_NOFOLLOW)
    assert replaces == [(tmp_name, b.CANONICAL_OUTPUT.name, create_dir_fd, create_dir_fd)]
    assert not (b.CANONICAL_OUTPUT.parent / tmp_name).exists()
    assert _leftovers(b.CANONICAL_OUTPUT.parent) == []


@needs_cuda
def test_atomic_writer_rejects_parent_traversal_and_preserves_old_on_fs_errors(sandbox, tmp_path, monkeypatch):
    (sandbox / "results" / "sub").mkdir()
    (sandbox / "results" / "link_sub").symlink_to(tmp_path, target_is_directory=True)
    for bad in (
        sandbox / "results" / ".." / "results" / b.CANONICAL_OUTPUT.name,
        sandbox / "results" / "sub" / ".." / b.CANONICAL_OUTPUT.name,
        sandbox / "results" / "link_sub" / ".." / b.CANONICAL_OUTPUT.name,
        Path("results/../results") / b.CANONICAL_OUTPUT.name,
    ):
        with pytest.raises(ValueError, match="parent traversal"):
            b._write_json_atomic(bad, {"x": 1}, canonical_dest=True)
        with pytest.raises(ValueError, match="parent traversal"):
            b._write_json_atomic(bad, {"x": 1})

    out = b.CANONICAL_OUTPUT
    b._write_json_atomic(out, {"old": 1}, canonical_dest=True)
    with monkeypatch.context() as m:
        m.setattr(b.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync boom")))
        with pytest.raises(OSError, match="fsync boom"):
            b._write_json_atomic(out, {"new": 2}, canonical_dest=True)
    assert json.loads(out.read_text()) == {"old": 1} and _leftovers(out.parent) == []
    with monkeypatch.context() as m:
        m.setattr(b.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("replace boom")))
        with pytest.raises(OSError, match="replace boom"):
            b._write_json_atomic(out, {"new": 3}, canonical_dest=True)
    assert json.loads(out.read_text()) == {"old": 1} and _leftovers(out.parent) == []


# ------------------------------------------------------------------ CLI of --postprocess-from-git
def _error(capsys, argv):
    with pytest.raises(SystemExit) as exc:
        b.parse_args(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


@needs_cuda
def test_cli_postprocess_mode_measures_nothing(tmp_path, capsys):
    args = b.parse_args(["--postprocess-from-git", SOURCE_COMMIT])
    assert args.postprocess_from_git == SOURCE_COMMIT and args.output == b.CANONICAL_OUTPUT
    assert args.protocol is None and args.canonical is False
    assert b.parse_args([]).postprocess_from_git is None
    free = b.parse_args(["--postprocess-from-git", PRODUCER_COMMIT, "--output", str(tmp_path / "copy.json")])
    assert free.output == tmp_path / "copy.json"
    for short in ("726b20c", SOURCE_COMMIT.upper(), "HEAD"):
        assert "full 40-hex SHA" in _error(capsys, ["--postprocess-from-git", short])
    for extra in (["--quick"], ["--parts", "ABC"], ["--device", "cuda"], ["--seeds", "1", "2", "3"],
                  ["--max-eval-windows", "4"], ["--data-dir", "/x"], ["--no-verify"], ["--checkpoint-dir", "/x"]):
        assert "measures nothing" in _error(capsys, ["--postprocess-from-git", SOURCE_COMMIT, *extra])
    assert "unrecognized" in _error(capsys, ["--postprocess-f", SOURCE_COMMIT])  # no abbreviations
    other = tmp_path / "other" / "results" / b.CANONICAL_OUTPUT.name
    alias = tmp_path / "alias_cycle5_c5_03.json"
    alias.symlink_to(b.CANONICAL_OUTPUT)
    traversal = ROOT / "results" / ".." / "results" / b.CANONICAL_OUTPUT.name
    for out in (other, alias, tmp_path / "cycle5_c5_03_copy.json", tmp_path / "copy.json", traversal):
        assert "A postprocess writes" in _error(capsys, ["--postprocess-from-git", SOURCE_COMMIT, "--output", str(out)])


@needs_cuda
def test_postprocess_into_the_reserved_file_needs_a_clean_tree_and_the_pinned_source(monkeypatch):
    b.require_clean_tree({"git_sha": "a" * 40, "git_dirty": False}, b.CANONICAL_OUTPUT)
    with pytest.raises(RuntimeError, match="clean"):
        b.require_clean_tree({"git_sha": "a" * 40, "git_dirty": True}, b.CANONICAL_OUTPUT)
    b.require_clean_tree({"git_sha": None, "git_dirty": True}, Path("/tmp/free.json"))
    monkeypatch.setattr(b, "postprocess_from_git", lambda *a, **k: pytest.fail("the source was read before the guard"))
    monkeypatch.setattr(b, "_write_json_atomic", lambda *a, **k: pytest.fail("nothing may be written"))
    monkeypatch.setattr(b.h, "git_info", lambda: {"git_sha": "a" * 40, "git_dirty": True})
    with pytest.raises(RuntimeError, match="clean"):
        b.main(["--postprocess-from-git", SOURCE_COMMIT, "--silent"])
    monkeypatch.setattr(b.h, "git_info", lambda: {"git_sha": "a" * 40, "git_dirty": False})
    with pytest.raises(ValueError, match="pinned source"):
        b.main(["--postprocess-from-git", PRODUCER_COMMIT, "--silent"])


@needs_cuda
def test_postprocess_refuses_a_source_that_is_not_a_committed_v1_result():
    with pytest.raises(ValueError, match="not a commit object"):
        b.postprocess_from_git("0" * 40, ["python", b.SCRIPT_REPO_PATH])
    with pytest.raises(ValueError, match="cannot read"):  # the producer commit holds no result file
        b.postprocess_from_git(PRODUCER_COMMIT, ["python", b.SCRIPT_REPO_PATH])


@needs_cuda
def test_driver_capabilities_at_commit():
    assert b._driver_capabilities_at_commit(PRODUCER_COMMIT) == ("cycle5_c5_03.v1", False)
    assert b._driver_capabilities_at_commit(AUDIT_COMMIT) == ("cycle5_c5_03.v1", False)
    assert b.SCHEMA_VERSION == "cycle5_c5_03.v2" and b.SOURCE_SCHEMA_VERSION == "cycle5_c5_03.v1"
    assert set(b.SUMMARY_MODULES) <= set(b.code_modules()) and b.DRIVER_MODULE in b.SUMMARY_MODULES


# ------------------------------------------------------------------ the committed canonical artifact
@pytest.fixture(scope="module")
def canonical():
    if not CUDA or not b.CANONICAL_OUTPUT.exists():
        pytest.skip("canonical result not committed yet")
    payload = json.loads(b.CANONICAL_OUTPUT.read_text())
    if payload["schema_version"] == "cycle5_c5_03.v1":
        pytest.skip("the committed result is the v1 source: run --postprocess-from-git first")
    return payload


def _source():
    blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{SOURCE_COMMIT}:{b.RESULT_REPO_PATH}"],
                          capture_output=True, check=True).stdout
    return blob, json.loads(blob)


def _rejects_canonical(payload, edit, match):
    _rejects(payload, edit, match, canonical=True)


def test_committed_canonical_artifact(canonical):
    payload = canonical
    assert b.validate_results(payload, canonical=True) == b.validate_results(payload) == CERTIFIED
    assert payload["schema_version"] == "cycle5_c5_03.v2"
    assert payload["canonical"] and not payload["quick"] and payload["protocol"] == b.CANONICAL
    assert payload["meta"]["git_dirty"] is False and payload["meta"]["environment"]["capability"] >= [12, 0]
    assert payload["kernel_parity"]["cases"] == 279 and payload["kernel_parity"]["passed"]
    assert [[r["m"], r["n"], r["k"]] for r in payload["part_a"]["shapes"]] == b.PART_A_SHAPES
    assert payload["part_b"]["models"]["120m"]["linear_parameters"] == 84_934_656
    c = payload["part_c"]
    assert c["calibration"]["tokens"] == 128 * 512 and len(set(c["calibration"]["starts"])) == 128
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
    assert any("4736807" in note and "c5_03_run_evidence" in note for note in payload["caveats"])
    assert any("Static-scale" in note and "sessions" in note for note in payload["caveats"])


def test_committed_artifact_keeps_the_v1_measurements_bit_for_bit(canonical):
    blob, source = _source()
    assert hashlib.sha256(blob).hexdigest() == SOURCE_SHA256 == b.CANONICAL_SOURCE_ARTIFACTS[SOURCE_COMMIT]
    assert source["schema_version"] == "cycle5_c5_03.v1" and "postprocess" not in source
    assert b.raw_measurements(canonical) == b.raw_measurements(source)
    assert b.raw_measurements_sha256(canonical) == b.raw_measurements_sha256(source)
    assert canonical["part_c"]["analysis"] == source["part_c"]["analysis"]  # the statistics did not change
    assert canonical["summary"] == source["summary"]
    assert set(canonical) == b.TOP_KEYS
    post = canonical["postprocess"]
    assert list(post) == list(b.POSTPROCESS_KEYS)
    assert post["source_result_git_sha"] == SOURCE_COMMIT and post["source_artifact_sha256"] == SOURCE_SHA256
    assert post["source_raw_measurements_sha256"] == b.raw_measurements_sha256(source)
    assert post["postprocess_git_dirty"] is False and post["reason"] == b.POSTPROCESS_REASON
    assert "726b20c" in post["reason"] and "not scored a third time" in post["reason"]
    assert post["retained_evidence_sha256"] == b.RETAINED_EVIDENCE_SHA256
    assert post["postprocess_command"][1:] == [f"research/quant_ptq/{b.SCRIPT_NAME}", "--postprocess-from-git",
                                               SOURCE_COMMIT]
    assert canonical["meta"]["git_sha"] == PRODUCER_COMMIT != post["postprocess_git_sha"]
    assert b._driver_capabilities_at_commit(post["postprocess_git_sha"]) == (b.SCHEMA_VERSION, True)
    assert b._matches_current_summary_code(post["postprocess_git_sha"])
    assert not b._matches_current_summary_code(PRODUCER_COMMIT)


def test_postprocess_is_a_pure_function_of_the_source(canonical, monkeypatch):
    post = canonical["postprocess"]
    monkeypatch.setattr(b.h, "git_info", lambda: {"git_sha": post["postprocess_git_sha"], "git_dirty": False})
    again = b.postprocess_from_git(SOURCE_COMMIT, post["postprocess_command"])
    assert b.validate_results(again, canonical=True) == CERTIFIED

    def stable(p):
        return {**p, "postprocess": {k: v for k, v in p["postprocess"].items() if k != "postprocess_timestamp"}}
    assert stable(again) == stable(canonical)


def test_canonical_validator_rejects_the_audit_calibration_cases(canonical):
    """Audit finding 1: 128 copies of -1 and a wrong token count were certified."""
    cal = ["part_c", "calibration"]
    _rejects_canonical(canonical, lambda p: _set(p, [*cal, "starts"], [-1] * 128), "calibration starts")
    _rejects_canonical(canonical, lambda p: _set(p, [*cal, "starts"], sorted(_get(p, [*cal, "starts"]))[::-1]),
                       "calibration starts")
    _rejects_canonical(canonical, lambda p: _get(p, [*cal, "starts"]).__setitem__(0, _get(p, [*cal, "starts"])[1]),
                       "calibration starts")
    _rejects_canonical(canonical, lambda p: _set(p, [*cal, "tokens"], 65_535), "calibration must be")
    _rejects_canonical(canonical, lambda p: _set(p, [*cal, "tokens"], 65_536.0), "calibration must be")
    _rejects_canonical(canonical, lambda p: _set(p, [*cal, "seed"], 1), "calibration must be")
    _rejects_canonical(canonical, lambda p: _set(p, [*cal, "note"], "x"), "calibration record keys")
    _rejects_canonical(canonical, lambda p: _get(p, cal).pop("tokens"), "calibration record keys")


def test_canonical_validator_rejects_the_audit_evidence_cases(canonical):
    """Audit finding 2: zero linear parameters, a negative kernel time, a false cooperative flag, AWQ alpha
    on plain GPTQ; plus the expected kernel roles and the cooperative grid capacity."""
    m120 = ["part_b", "models", "120m"]
    _rejects_canonical(canonical, lambda p: _set(p, [*m120, "linear_parameters"], 0), "linear_parameters")
    _rejects_canonical(canonical, lambda p: _set(p, [*m120, "linear_parameters"], 84_934_656.0), "linear_parameters")
    _rejects_canonical(canonical, lambda p: _set(p, ["part_b", "models", "50m", "linear_parameters"], 84_934_656),
                       "linear_parameters")
    _rejects_canonical(canonical, lambda p: _set(p, [*m120, "compile_seconds"], -1.0), "compile_seconds")

    prof = ["part_a", "shapes", 2, "cuda_graph_kernel_profile_us"]

    def kernels(p, method):
        return _get(p, [*prof, method])

    def gemm_name(p, method):
        return next(k for k in kernels(p, method) if "cutlass" in k)

    _rejects_canonical(canonical, lambda p: kernels(p, "c5_03_had_b16").__setitem__("_quant_kernel", -1.0),
                       "finite nonnegative")
    _rejects_canonical(canonical, lambda p: kernels(p, "c4_02_none").__setitem__("_c4_02_amax_kernel", None),
                       "finite nonnegative")
    _rejects_canonical(canonical, lambda p: kernels(p, "c5_03_none").__setitem__("", 1.0), "finite nonnegative")
    _rejects_canonical(canonical, lambda p: _set(p, [*prof, "c5_03_none"], {}), "finite nonnegative")
    _rejects_canonical(canonical, lambda p: kernels(p, "c5_03_bfly_b16").pop(gemm_name(p, "c5_03_bfly_b16")),
                       "no FP4 GEMM")
    _rejects_canonical(canonical, lambda p: kernels(p, "nvfp4_gemm_only").pop(gemm_name(p, "nvfp4_gemm_only")),
                       "no FP4 GEMM")
    _rejects_canonical(canonical, lambda p: kernels(p, "nvfp4_gemm_only").__setitem__("_quant_kernel", 1.0),
                       "runs no quantizer")

    def swap_role(p, method, old, new):
        k = kernels(p, method)
        k[new] = k.pop(old)
    _rejects_canonical(canonical, lambda p: swap_role(p, "c5_03_none", "_quant_kernel", "_c4_02_quant_kernel"),
                       "own implementation")
    _rejects_canonical(canonical, lambda p: swap_role(p, "c4_02_had_b16", "_c4_02_amax_kernel", "_amax_kernel"),
                       "own implementation")
    _rejects_canonical(canonical, lambda p: kernels(p, "c5_03_had_b16_static_gs").__setitem__("_amax_kernel", 1.0),
                       "static c5_03")
    _rejects_canonical(canonical, lambda p: kernels(p, "c5_03_had_b16").pop("_amax_kernel"), "dynamic c5_03")
    _rejects_canonical(canonical, lambda p: kernels(p, "c4_02_had_b16_static_gs").pop("_c4_02_quant_kernel"),
                       "static c4_02")

    coop = ["part_a", "ablation", 0, "coop"]
    _rejects_canonical(canonical, lambda p: _set(p, [*coop, "launch_cooperative_grid"], False), "cooperative launch")
    _rejects_canonical(canonical, lambda p: _set(p, [*coop, "launch_cooperative_grid"], 1), "cooperative launch")
    _rejects_canonical(canonical, lambda p: _set(p, [*coop, "grid", "resident_capacity"], 10_000), "resident capacity")
    _rejects_canonical(canonical, lambda p: _set(p, [*coop, "grid", "programs"], 0), "resident capacity")
    _rejects_canonical(canonical, lambda p: _set(p, [*coop, "grid", "programs"], 757), "resident capacity")
    _rejects_canonical(canonical, lambda p: _set(p, [*coop, "grid", "sms"], -84), "resident capacity")
    _rejects_canonical(canonical, lambda p: _set(p, [*coop, "grid", "note"], 1), "resident capacity")
    _rejects_canonical(canonical, lambda p: _set(p, [*coop, "note"], 1), "cooperative record keys")

    grid0 = ["part_c", "seeds", 0, "grid"]
    plain = next(i for i, r in enumerate(canonical["part_c"]["seeds"][0]["grid"])
                 if r["method"] == "gptq" and r["mode"] == "w4a4_hw_dynamic")
    awq = next(i for i, r in enumerate(canonical["part_c"]["seeds"][0]["grid"]) if r["awq"])
    _rejects_canonical(canonical, lambda p: _set(p, [*grid0, plain, "awq_alpha"], 0.75), "no awq_alpha")
    _rejects_canonical(canonical, lambda p: _set(p, [*grid0, plain, "inherited"], True), "not inherited")
    _rejects_canonical(canonical, lambda p: _set(p, [*grid0, plain, "seconds_val"], -1.0), "seconds_val")
    _rejects_canonical(canonical, lambda p: _set(p, [*grid0, plain, "seconds_test"], None), "seconds_val")
    _rejects_canonical(canonical, lambda p: _set(p, [*grid0, awq, "awq_alpha"], 0.3), "protocol alphas")


def test_canonical_validator_rejects_postprocess_provenance_tampering(canonical, monkeypatch):
    post = ["postprocess"]

    def update(d, **kw):
        d.update(kw)

    # Raw measurements: every field not covered by another check still has to equal the source.
    _rejects_canonical(canonical, lambda p: _get(p, ["meta", "loadavg_start"]).__setitem__(0, 9.0), "meta differs")
    _rejects_canonical(canonical, lambda p: _set(p, ["part_c", "seconds"], p["part_c"]["seconds"] + 1), "part_c differs")
    _rejects_canonical(canonical, lambda p: _set(p, ["part_c", "checkpoint_dir"], "/elsewhere"), "part_c differs")
    _rejects_canonical(canonical, lambda p: _set(p, ["data", "directory"], "/elsewhere"), "data differs")
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "source_artifact_sha256"], "0" * 64),
                       "source_artifact_sha256")
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "source_raw_measurements_sha256"], "0" * 64),
                       "source_raw_measurements_sha256")
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "source_result_git_sha"], PRODUCER_COMMIT),
                       "cannot read|source result commit")
    # Record shape.
    _rejects_canonical(canonical, lambda p: _get(p, post).pop("reason"), "postprocess keys")
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "note"], "x"), "postprocess keys")
    _rejects_canonical(canonical, lambda p: p.pop("postprocess"), "top-level keys")
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "reason"], "cosmetic"), "postprocess.reason")
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "source_schema_version"], "cycle5_c5_03.v2"),
                       "source_schema_version")
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "postprocess_git_dirty"], 0), "must be a bool")
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "postprocess_git_dirty"], True), "clean tree")
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "postprocess_timestamp"], "today"), "timestamp")
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "postprocess_timestamp"], p["meta"]["timestamp_start"]),
                       "before the end of the measuring run")
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "postprocess_timestamp"], p["meta"]["timestamp_end"]),
                       "before the source result commit")
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "postprocess_timestamp"], "2026-11-27T11:30:16-0700"),
                       "before the postprocessing commit")
    _rejects_canonical(
        canonical,
        lambda p: (
            _set(p, ["meta", "timestamp_start"], "2026-11-27T10:30:28-0700"),
            _set(p, ["meta", "timestamp_end"], "2026-11-27T11:28:48-0700"),
        ),
        "before the producer commit",
    )
    # Postprocessing code.
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "postprocess_git_sha"], "HEAD"), "40-hex")
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "postprocess_git_sha"], "0" * 40), "not a commit object")
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "postprocess_code_sha256"], "0" * 64),
                       "git blobs at postprocess_git_sha")
    _rejects_canonical(canonical, lambda p: _set(p, [*post, "postprocess_code_modules"],
                                                 p["postprocess"]["postprocess_code_modules"][:-1]),
                       "postprocess_code_modules differ")
    for old in (PRODUCER_COMMIT, AUDIT_COMMIT, SOURCE_COMMIT):  # v1 commits cannot be credited as the postprocessor
        _rejects_canonical(canonical, lambda p, s=old: _set(p, [*post, "postprocess_git_sha"], s),
                           "git blobs at postprocess_git_sha")
        code = b._code_sha256_at_commit(old, tuple(b._code_modules_at_commit(old)))
        _rejects_canonical(canonical, lambda p, s=old, c=code: update(_get(p, post), postprocess_git_sha=s,
                                                                      postprocess_code_sha256=c),
                           "must commit cycle5_c5_03.v2")

    def v1_as_fresh_v2(p):  # strip the postprocess and recompute the v2 hash on the v1 producer
        p["postprocess"] = None
        hashes = {k: v["sha256"] for k, v in p["data"]["splits"].items()}
        p["run_config_hash"] = b.run_config_hash(False, p["protocol"], p["meta"]["code_sha256"], p["meta"]["git_sha"],
                                                 p["meta"]["environment"], hashes, b.SCHEMA_VERSION)
    _rejects_canonical(canonical, v1_as_fresh_v2, "declares schema 'cycle5_c5_03.v1'")

    def v1_hash_without_postprocess(p):
        p["postprocess"] = None
    _rejects_canonical(canonical, v1_hash_without_postprocess, "declares schema|run_config_hash")
    # Command.
    command = canonical["postprocess"]["postprocess_command"]
    for bad, match in ((command[:2], "--postprocess-from-git with the source result commit"),
                       ([*command[:3], PRODUCER_COMMIT], "source result commit"),
                       ([*command[:3], "726b20c"], "valid runnable invocation"),
                       ([*command, "--quick"], "valid runnable invocation"),
                       ([*command, "--data-dir", "/x"], "valid runnable invocation"),
                       ([command[0], command[1], "--postprocess-f", SOURCE_COMMIT], "valid runnable invocation"),
                       ([*command, "--output", "/tmp/copy.json"],
                        "valid runnable invocation|must target the canonical output path"),
                       ([command[0], b.SCRIPT_NAME, *command[2:]], "not an invocation"),
                       ([command[0], f"fake/{b.SCRIPT_REPO_PATH}", *command[2:]], "not an invocation"),
                       (["bash", *command[1:]], "Python invocation"),
                       ([command[0], "other.py", *command[2:]], "not an invocation"),
                       ("python", "Python invocation")):
        _rejects_canonical(canonical, lambda p, c=bad: _set(p, [*post, "postprocess_command"], c), match)
    _rejects_canonical(canonical, lambda p: _set(p, ["meta", "command"], command), "argv|measuring invocation")
    _rejects_canonical(canonical, lambda p: _set(p, ["meta", "command"],
                                               [p["meta"]["command"][0], b.SCRIPT_NAME, *p["meta"]["argv"]]),
                       "not an invocation")
    # Retained run evidence.
    evidence = [*post, "retained_evidence_sha256"]
    _rejects_canonical(canonical, lambda p: _set(p, [*evidence, "canonical_38cef48.log"], "0" * 64), "pinned digests")
    _rejects_canonical(canonical, lambda p: _get(p, evidence).pop("superseded_result_4736807.json"), "64-hex digests")
    _rejects_canonical(canonical, lambda p: _set(p, [*evidence, "extra.log"], "0" * 64), "64-hex digests")
    # Caller expectations.
    with pytest.raises(ValueError, match="labeled canonical"):
        b.validate_results(canonical, canonical=False)
    monkeypatch.setattr(b, "CANONICAL_SOURCE_ARTIFACTS", {SOURCE_COMMIT: "0" * 64})
    with pytest.raises(ValueError, match="pinned committed C5-03 result"):
        b.validate_results(canonical, canonical=True)


def test_committed_canonical_caveats_and_summary_are_recomputed(canonical):
    _rejects_canonical(canonical, lambda p: p["caveats"].pop(), "caveats")
    _rejects_canonical(canonical, lambda p: _set(p, ["part_c", "analysis", "contrasts", 0, "test", "p_holm"], 0.0),
                       "analysis")
    _rejects_canonical(canonical, lambda p: _set(p, ["summary", "part_c", "baseline", 0, "val_nll"], 1.0), "summary")
