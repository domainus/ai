"""Tests for the Cycle 3 C3-03 launch fusion of the Schatten Switch circuits (needs a CUDA GPU).

The norms of ``R_h`` and ``S`` come from the tiles of ``_syrk_norm_kernel`` and
``_square_norm_kernel``. These tests check the tile partials against FP64 norms, the fold
kernel against the host arithmetic, the launch and synchronization counts with the profiler,
the bitwise equality of CUDA-graph replay and eager calls, and the output guards of the
benchmark script.
"""

import dataclasses
import json
import math

import pytest
import torch
import triton
from torch.profiler import ProfilerActivity, profile
from triton import knobs

from research.polar_opt import bench_c3_03_tier1_refresh as refresh
from research.polar_opt import harness_tier2_real
from research.polar_opt import polar_switch_blackwell as psb
from research.polar_opt import schatten_switch_blackwell as ssb
from research.polar_opt.kernels_blackwell import _syrk_kernel
from research.polar_opt.polar_switch_blackwell import _launch, _symprod_kernel, _triangle_grid, polar_switch_blackwell
from research.polar_opt.schatten_switch_blackwell import (
    SCHATTEN_SWITCH_CIRCUITS,
    schatten_switch_blackwell,
    theoretical_flops,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

NAMES = sorted(SCHATTEN_SWITCH_CIRCUITS)
FP16, BF16 = torch.float16, torch.bfloat16
LLM_SHAPES = [(512, 2048), (1024, 4096), (2048, 8192), (4096, 4096), (4, 1024, 4096)]
ODD_SHAPES = [(97, 331), (3, 1025, 1100), (5, 33, 17), (515, 1029)]
MIN_ONE_SHAPES = [(1, 1), (1, 7), (7, 1), (4, 1, 9)]
AUXILIARY = ["_safe_norm_part_kernel", "_normalize_safe_kernel", "_fold_and_row0_axpy_kernel"]
_FIXED: dict = {}


def rand(shape, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, device="cuda", generator=generator)


def fixed(kernel, block, bk=64):
    """``kernel`` with one tile configuration, so that two kernels run the same tile loop."""
    key = (kernel, block, bk)
    if key not in _FIXED:
        config = triton.Config({"BLOCK": block, "BK": bk}, num_warps=4, num_stages=3)
        _FIXED[key] = triton.autotune(configs=[config], key=[])(kernel.fn)
    return _FIXED[key]


def offset_view(tensor, elements=1):
    """A copy of ``tensor`` whose data pointer is ``elements`` items after a fresh allocation."""
    base = torch.empty(tensor.numel() + elements, dtype=tensor.dtype, device=tensor.device)
    view = base[elements:].view(tensor.shape)
    view.copy_(tensor)
    return view


def tile_reference(stored, block):
    """FP64 sum of squares that each upper-triangle tile adds to the symmetric ``stored``.

    A tile above the diagonal counts twice (the tile and its mirror). A diagonal tile counts
    its square region once. The order is the row-major walk of ``_tri_tile``.
    """
    batch, n, _ = stored.shape
    tiles = -(-n // block)
    padded = torch.zeros(batch, tiles * block, tiles * block, dtype=torch.float64, device=stored.device)
    padded[:, :n, :n] = stored.double().square()
    sums = padded.view(batch, tiles, block, tiles, block).sum(dim=(2, 4))
    i, j = torch.triu_indices(tiles, tiles, device=stored.device)
    return sums[:, i, j] * torch.where(i == j, 1.0, 2.0)


def check_partials(scratch, offset, stored, block):
    """Tile partials and their sum against FP64; no write outside the tile slots."""
    expected = tile_reference(stored, block)
    count = expected.shape[1]
    partials = scratch[:, offset:offset + count]
    # A row of at most BLOCK exact squares is summed in FP32: relative error below BLOCK * 2^-24.
    bound = block * 2.0 ** -24
    assert torch.all((partials - expected).abs() <= bound * expected), "tile weighting or tail mask"
    exact = torch.linalg.vector_norm(stored, dim=(-2, -1), dtype=torch.float64)
    fused = partials.sum(dim=1).sqrt()
    assert torch.all((fused - exact).abs() <= 0.5 * bound * exact + 1e-300)
    untouched = torch.cat([scratch[:, :offset], scratch[:, offset + count:]], dim=1)
    assert torch.isnan(untouched).all(), "a kernel wrote outside its tile slots"
    return (fused - exact).abs().div(exact.clamp_min(1e-300)).max().item()


TILE_CASES = [  # (batch, n, k), BLOCK
    ((1, 1, 5), 32), ((2, 7, 3), 32), ((3, 33, 17), 32), ((1, 97, 331), 32), ((1, 97, 331), 64),
    ((2, 128, 256), 128), ((2, 129, 300), 128), ((2, 129, 300), 64), ((1, 515, 1029), 128),
    ((1, 515, 1029), 32), ((3, 1025, 1100), 128), ((3, 1025, 1100), 64), ((1, 4097, 4100), 128),
]


@needs_cuda
@pytest.mark.parametrize("dtype", [FP16, BF16], ids=str)
@pytest.mark.parametrize("unaligned", [False, True], ids=["aligned", "unaligned"])
@pytest.mark.parametrize("shape,block", TILE_CASES, ids=str)
def test_tile_partials_equal_float64_norms(shape, block, unaligned, dtype):
    """``_syrk_norm_kernel`` and ``_square_norm_kernel``: same tiles as the plain kernels, and
    partials that reproduce the FP64 Frobenius norm on odd, batched, unaligned and large sizes."""
    batch, n, k = shape
    if n > 2048 and (unaligned or dtype == BF16):
        pytest.skip("the large size runs once")
    prescale, alpha = ssb._prescale(n)
    X = (prescale * rand(shape, seed=n) / math.sqrt(n * k)).to(dtype)  # the operand scale of the circuit
    off_r, off_s, off_c, stride = ssb._scratch_layout(n, 2)
    scratch = torch.full((batch, stride), float("nan"), dtype=torch.float64, device="cuda")
    plain = torch.empty(batch, n, n, dtype=dtype, device="cuda")
    fused = torch.empty_like(plain)
    if unaligned:
        X, scratch, fused = offset_view(X), offset_view(scratch), offset_view(fused)
        assert X.data_ptr() % 16 and scratch.data_ptr() % 16 and fused.data_ptr() % 16
    grid = _triangle_grid(n, batch)

    _launch(fixed(_syrk_kernel, block), grid, X, plain, n, k, *X.stride())
    config = _launch(fixed(ssb._syrk_norm_kernel, block), grid, X, fused, scratch, n, k, *X.stride(), off_r, stride)
    assert ssb._tile_count(n, config) == tile_reference(fused, block).shape[1]
    assert torch.equal(plain, fused) and torch.equal(fused, fused.mT)
    assert check_partials(scratch, off_r, fused, block) < 1e-6

    scratch.fill_(float("nan"))
    R = fused.clone()
    _launch(fixed(_symprod_kernel, block), grid, R, R, R, plain, n, float(alpha), 0.0, 0.0)
    _launch(fixed(ssb._square_norm_kernel, block), grid, R, fused, scratch, n, float(alpha), off_s, stride)
    assert torch.equal(plain, fused) and torch.equal(fused, fused.mT)
    assert check_partials(scratch, off_s, fused, block) < 1e-6


@needs_cuda
def test_tile_partials_weight_diagonal_and_off_diagonal_tiles():
    """All-ones Gram matrix: every entry is K, so a tile partial counts stored entries."""
    n, k, block = 100, 64, 32  # 4 x 4 tiles, the last row and column hold 4 entries
    X = torch.ones(1, n, k, dtype=FP16, device="cuda")
    off_r, _, _, stride = ssb._scratch_layout(n, 2)
    scratch = torch.zeros(1, stride, dtype=torch.float64, device="cuda")
    out = torch.empty(1, n, n, dtype=FP16, device="cuda")
    _launch(fixed(ssb._syrk_norm_kernel, block), _triangle_grid(n, 1), X, out, scratch, n, k, *X.stride(),
            off_r, stride)
    entries = (scratch[0, off_r:off_r + 10] / float(k * k)).tolist()
    sizes = [32, 32, 32, 4]
    expected = [sizes[i] * sizes[j] * (1 if i == j else 2) for i in range(4) for j in range(i, 4)]
    assert entries == expected and sum(expected) == n * n


@needs_cuda
@pytest.mark.parametrize("scaling", ["schatten8", "frobenius"])
@pytest.mark.parametrize("n,tiles", [(33, 1), (97, 3), (400, 70), (1024, 150)])
@pytest.mark.parametrize("name", NAMES)
def test_fold_kernel_matches_host_fold(name, n, tiles, scaling):
    """Coefficient rows and the first-row update against ``_kappas`` and ``_fold`` in FP64.

    The tile partials are split over more than one chunk, and the last matrix has an infinite
    partial: it must get zero coefficients and a zero buffer.
    """
    circuit = SCHATTEN_SWITCH_CIRCUITS[name]
    rows = circuit.table[0][1]
    batch = 3
    # Triton passes a float argument as FP32, so the kernel sees u_safe rounded to FP32.
    host = dataclasses.replace(circuit, u_safe=torch.tensor(circuit.u_safe, dtype=torch.float32).item())
    prescale, alpha = ssb._prescale(n)
    X = (prescale * rand((batch, n, 2 * n), seed=n) / math.sqrt(2.0 * n * n)).half()
    R = (X.float() @ X.float().mT).half()
    R = torch.triu(R) + torch.triu(R, 1).mT
    S = (alpha * (R.float() @ R.float())).half()
    S = torch.triu(S) + torch.triu(S, 1).mT
    before = S.clone()
    off_r, off_s, off_c, stride = ssb._scratch_layout(n, len(rows))
    assert tiles <= off_s - off_r
    scratch = torch.full((batch, stride), float("nan"), dtype=torch.float64, device="cuda")
    weights = torch.rand(tiles, dtype=torch.float64, device="cuda") + 0.1
    weights /= weights.sum()
    sums = {off_r: R.double().square().sum(dim=(-2, -1)), off_s: S.double().square().sum(dim=(-2, -1))}
    for offset, total in sums.items():
        scratch[:, offset:offset + tiles] = total[:, None] * weights
    scratch[2, off_s + tiles - 1] = float("inf")

    ssb._fold_segment1(circuit, R, S, scratch, tiles, tiles, k=2 * n, use_frobenius=scaling == "frobenius")
    coeff = scratch[:, off_c:].view(batch, len(rows), 4)
    eye = torch.eye(n, device="cuda")
    for index in range(2):
        norm_r, norm_s = sums[off_r][index].sqrt().item(), sums[off_s][index].sqrt().item()
        kappa = ssb._kappas(host, scaling, False, [norm_r], [norm_s], n, FP16)[0]
        b0, c0, folded = ssb._fold(rows, kappa, alpha)
        assert coeff[index, 0, 0].item() == pytest.approx(circuit.table[0][0] * math.sqrt(kappa), rel=1e-12, abs=0)
        assert coeff[index, :, 3].tolist() == [1.0] * len(rows)
        expected = torch.tensor([row[3:] for row in folded], dtype=torch.float64, device="cuda")
        assert torch.allclose(coeff[index, 1:, :3], expected, rtol=1e-11, atol=0.0)
        b0, c0 = torch.tensor([b0, c0], dtype=torch.float32).tolist()
        updated = (before[index].float() + b0 * R[index].float() + c0 * eye).half().float()
        # One FP32 rounding (or a fused multiply-add) before the FP16 rounding: at most one FP16
        # ulp, which is 2^-24 for a subnormal entry.
        assert ((S[index].float() - updated).abs() <= 2.0 ** -10 * updated.abs() + 2.0 ** -24).all()
        assert (S[index].float() == updated).float().mean() > 0.99
    assert coeff[2, 0, 3].item() == 0.0 and not coeff[2, 1:].any() and not S[2].any()


@needs_cuda
def test_fold_supports_pure_square_rows_and_rejects_long_tables():
    pure = ssb.SchattenSwitchCircuit(
        name="TEST_PURE_FUSION", table=((0.0, ((1, 0, 0, 1.0, 0.0, 0.0), (0, 1, 0, 0.0, 0.5, 0.5))),),
        buffers=2, u_safe=1.05, lo=0.5, error=0.5)
    G = rand((2, 40, 90), seed=3)
    out = schatten_switch_blackwell(G, pure)
    assert torch.isfinite(out).all()
    reference = ssb.schatten_switch_reference(G, pure, dtype=FP16).double()
    assert ((out.double() - reference).norm() / reference.norm()).item() < 5e-3
    rows = ((1, 0, 0, 1.0, -1.0, 0.0), *(((0, 1, 0, 0.1, 0.5, 0.5), (1, 0, 1, 0.1, 0.5, 0.5)) * 4))
    long = ssb.SchattenSwitchCircuit(name="TEST_LONG_FUSION", table=((0.0, rows),), buffers=2, u_safe=1.05,
                                     lo=0.5, error=0.5)
    with pytest.raises(ValueError, match="at most"):
        schatten_switch_blackwell(G, long)


# ------------------------------------------------------------------ launches and synchronization
def profile_call(call, G):
    """Kernel launches by name and device-to-host copies of one warm call."""
    for _ in range(3):
        call(G)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        call(G)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    kernels = [name for name in names if not name.lower().startswith("mem")]
    return kernels, sum("dtoh" in name.lower() for name in names)


@needs_cuda
@pytest.mark.parametrize("shape", [(97, 331), (512, 2048), (4, 256, 512)], ids=str)
@pytest.mark.parametrize("name", NAMES)
def test_profiler_launch_counts_and_zero_device_to_host_copies(name, shape):
    circuit = SCHATTEN_SWITCH_CIRCUITS[name]
    G = rand(shape, seed=5)
    kernels, copies = profile_call(lambda x: schatten_switch_blackwell(x, name), G)
    counts = {kernel: kernels.count(kernel) for kernel in set(kernels) if kernel.startswith("_")}
    squares = circuit.squarings
    assert counts == {"_safe_norm_part_kernel": 1, "_normalize_safe_kernel": 1, "_syrk_norm_kernel": 1,
                      "_square_norm_kernel": 1, "_fold_and_row0_axpy_kernel": 1,
                      "_symprod_ptr_kernel": squares[0] - 1, "_syrk_kernel": 1, "_symprod_kernel": squares[1]}
    matrix = theoretical_flops(name, shape)["matrix_kernel_launches"]
    assert sum(counts.values()) == matrix - len(circuit.table) + len(AUXILIARY)  # cuBLAS runs Q @ X
    if G.ndim == 2:  # cuBLAS launches one kernel per Q @ X; before the fusion there were 4 auxiliary launches
        assert len(kernels) == matrix + len(AUXILIARY)
    assert copies == 0
    reference, _ = profile_call(lambda x: polar_switch_blackwell(x, "PS3_55"), G)
    assert len(kernels) <= len(reference)


@needs_cuda
def test_benchmark_launch_counter_reports_the_fused_counts():
    table = refresh.count_launches([(128, 512)])["128x512"]
    assert {name: row["total"] for name, row in table.items()} == {
        "ss3_54": 16, "ss3_53": 15, "ss3_35": 15, "ss3_34": 14, "ps3_55": 16}
    assert all(row["memcpy_dtoh"] == 0 for row in table.values())
    assert "_dual_norm_part_kernel" not in table["ss3_54"]["kernels"]
    assert table["ss3_54"]["kernels"]["cublas_gemm"] == 2


@needs_cuda
@pytest.mark.parametrize("name", NAMES)
def test_warm_calls_do_not_synchronize(name):
    """``sync_debug_mode('error')`` raises on every implicit device-to-host synchronization."""
    inputs = [rand(shape, seed=6) for shape in [(97, 331), (4, 256, 512), (512, 2048), (7, 1)]]
    inputs.append(rand((3, 64, 128), seed=7) * torch.tensor([1.0, 0.0, float("nan")], device="cuda")[:, None, None])
    for G in inputs:
        schatten_switch_blackwell(G, name)  # warm-up: autotune and table cache
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        outputs = [schatten_switch_blackwell(G, name) for G in inputs]
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert all(torch.isfinite(out).all() for out in outputs)
    assert not outputs[-1][1:].any()


# ------------------------------------------------------------------ CUDA-graph replay
def capture(call, G):
    """Warm ``call`` eagerly and on a side stream, then capture one call."""
    for _ in range(2):
        call(G)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        call(G)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call(G)
    torch.cuda.synchronize()
    return graph, out


def overwrites(G):
    """Adversarial contents for the static input of a captured graph."""
    yield "fresh", rand(G.shape, seed=99)
    yield "zeros", torch.zeros_like(G)
    yield "tiny", G * 1e-38
    yield "huge", G * 1e37
    for label, value in (("nan", float("nan")), ("+inf", float("inf")), ("-inf", float("-inf"))):
        bad = G.clone()
        bad.view(-1)[bad.numel() // 2] = value
        yield label, bad
    if G.ndim == 3:
        mixed = G.clone()
        scales = [1.0, 0.0, 1e-38, 1e37, float("nan"), float("inf"), float("-inf")]
        for index in range(len(G)):
            scale = scales[index % len(scales)]
            if math.isfinite(scale):
                mixed[index] *= scale
            else:
                mixed[index, 0, 0] = scale
        yield "mixed batch", mixed


@needs_cuda
@pytest.mark.parametrize("shape", LLM_SHAPES + ODD_SHAPES + MIN_ONE_SHAPES, ids=str)
@pytest.mark.parametrize("name", NAMES)
def test_cuda_graph_replay_is_bitwise_equal_to_eager(name, shape):
    def call(x):
        return schatten_switch_blackwell(x, name)

    static = rand(shape, seed=31)
    graph, out = capture(call, static)
    graph.replay()
    eager = call(static)
    assert torch.isfinite(eager).all() and eager.any()
    assert torch.equal(out, eager) and (out - eager).abs().max().item() == 0.0
    for label, content in overwrites(static.clone()):  # a copy: the loop overwrites static
        static.copy_(content)
        graph.replay()
        eager = call(content.clone())
        assert torch.isfinite(out).all(), label
        assert torch.equal(out, eager), label
        if label in ("zeros", "tiny", "nan", "+inf", "-inf") and static.ndim == 2:
            assert not out.any(), label
        if label in ("fresh", "huge"):
            assert out.any(), label
        if label == "mixed batch":
            assert out[0].any() and not out[1].any()


@needs_cuda
def test_interleaved_graphs_and_eager_calls_do_not_share_scratch():
    """Two captured graphs of one circuit and shape, replayed in turn with eager calls between:
    every call owns its scratch, so no result depends on the other graph."""
    def call(x):
        return schatten_switch_blackwell(x, "SS3_54")

    a, b = rand((3, 97, 331), seed=41), rand((3, 97, 331), seed=42) * 7.0
    expected_a, expected_b = call(a), call(b)
    assert not torch.equal(expected_a, expected_b)
    graph_a, out_a = capture(call, a)
    graph_b, out_b = capture(call, b)
    for _ in range(3):
        graph_a.replay()
        graph_b.replay()
        assert torch.equal(call(a), expected_a)
        graph_a.replay()
        assert torch.equal(out_a, expected_a) and torch.equal(out_b, expected_b)


# ------------------------------------------------------------------ the launch plan cache
@needs_cuda
def test_launch_plan_key_holds_what_triton_specializes_on():
    """Device, dtypes, pointer alignment and integers select a plan. Floats do not."""
    n, block = 48, 32
    config = triton.Config({"BLOCK": block, "BK": 64}, num_warps=4, num_stages=3)
    kernel = triton.autotune(configs=[config], key=[])(_symprod_kernel.fn)  # a new kernel: no plans yet
    A = rand((1, n, n), seed=8).half()
    A = A + A.mT
    out = torch.empty_like(A)
    grid = _triangle_grid(n, 1)

    def plans():
        return sum(key[0] is kernel for key in psb._PLANS)

    config = _launch(kernel, grid, A, A, A, out, n, 1.0, 0.0, 0.0)
    assert config == {"BLOCK": block, "BK": 64} and plans() == 1
    first = out.clone()
    assert _launch(kernel, grid, A, A, A, out, n, 2.0, 0.5, -1.0) == config and plans() == 1
    assert torch.allclose(out.float(), 2.0 * first.float() + 0.5 * A.float() - torch.eye(n, device="cuda"),
                          rtol=2e-3, atol=2e-2)
    _launch(kernel, grid, A, A, A, out, n, 1.0, 0.0, 0.0)
    assert torch.equal(out, first), "the cached plan repeats the first launch"

    shifted = offset_view(out)
    _launch(kernel, grid, A, A, A, shifted, n, 1.0, 0.0, 0.0)
    assert plans() == 2 and torch.equal(shifted, first)  # another alignment
    B = A.bfloat16()
    _launch(kernel, grid, B, B, B, torch.empty_like(B), n, 1.0, 0.0, 0.0)
    assert plans() == 3  # another dtype
    small = A[:, :40, :40].contiguous()
    _launch(kernel, lambda c: (3, 1, 1), small, small, small, torch.empty_like(small), 40, 1.0, 0.0, 0.0)
    assert plans() == 4  # another integer
    assert all(plan[0] is not None and hasattr(plan[0], "__getitem__") for plan in psb._PLANS.values())


@needs_cuda
def test_direct_launcher_equals_the_public_path_and_honors_hooks(monkeypatch):
    G = rand((3, 97, 331), seed=9)
    direct = schatten_switch_blackwell(G, "SS3_53")
    assert any(plan[3] is not None for plan in psb._PLANS.values()), "Triton 3.8 takes the direct path"

    seen = []

    def hook(metadata):
        seen.append(metadata)

    knobs.runtime.launch_enter_hook.add(hook)
    try:
        hooked = schatten_switch_blackwell(G, "SS3_53")
    finally:
        knobs.runtime.launch_enter_hook.remove(hook)
    assert len(seen) == 13 and torch.equal(hooked, direct)  # every Triton launch reached the hook

    # Another Triton version keeps the public launcher for new plans.
    saved = dict(psb._PLANS)
    psb._PLANS.clear()
    monkeypatch.setattr(triton, "__version__", "0.0.0")
    try:
        public = schatten_switch_blackwell(G, "SS3_53")
        assert psb._PLANS and all(plan[3] is None for plan in psb._PLANS.values())
        assert torch.equal(schatten_switch_blackwell(G, "SS3_53"), public)
    finally:
        psb._PLANS.clear()
        psb._PLANS.update(saved)
    assert torch.equal(public, direct)


# ------------------------------------------------------------------ benchmark script
def test_cli_output_guard_protects_canonical_outputs(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("no benchmark should run")

    for function in ("run_graph", "run_tier1", "launch_table"):
        monkeypatch.setattr(refresh, function, fail)
    monkeypatch.setattr(harness_tier2_real, "git_info", lambda: {"git_sha": "0" * 40, "git_dirty": False})
    parser = refresh.build_parser()
    assert refresh._resolve_outputs(parser.parse_args([])) == refresh.CANONICAL
    assert refresh._resolve_outputs(parser.parse_args(["--quick"])) == refresh.QUICK
    assert refresh._resolve_outputs(parser.parse_args(["--part", "graph"])) == {"graph": refresh.CANONICAL["graph"]}
    custom = tmp_path / "subset.json"
    subset = parser.parse_args(["--part", "tier1", "--max-cases", "4", "--tier1-output", str(custom)])
    assert refresh._resolve_outputs(subset) == {"tier1": custom}
    assert set(refresh.QUICK.values()).isdisjoint(refresh.CANONICAL.values())

    graph, tier1 = (str(refresh.CANONICAL[part]) for part in ("graph", "tier1"))
    for argv in (
        ["--iters", "1"],
        ["--blocks", "2"],
        ["--max-cases", "8"],
        ["--seed", "1"],
        ["--baseline-commit", "HEAD"],
        ["--warmups", "4", "--part", "graph"],
        ["--quick", "--graph-output", graph],
        ["--quick", "--part", "tier1", "--tier1-output", tier1],
        ["--part", "tier1", "--max-cases", "8", "--tier1-output", tier1],
        ["--part", "graph", "--graph-output", tier1],  # the canonical file of the other part
        ["--iters", "2", "--graph-output", str(custom), "--tier1-output", tier1],
    ):
        with pytest.raises(SystemExit) as exit_info:
            refresh.main(argv)
        assert exit_info.value.code == 2, argv


def test_canonical_outputs_need_a_clean_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(harness_tier2_real, "git_info", lambda: {"git_sha": "0" * 40, "git_dirty": True})
    with pytest.raises(ValueError, match="git_dirty"):
        refresh._require_clean_tree(dict(refresh.CANONICAL))
    refresh._require_clean_tree({"graph": tmp_path / "other.json"})  # other outputs may be dirty
    monkeypatch.setattr(refresh, "run_graph", lambda **kwargs: pytest.fail("no benchmark should run"))
    with pytest.raises(SystemExit) as exit_info:
        refresh.main(["--part", "graph"])
    assert exit_info.value.code == 2
    monkeypatch.setattr(harness_tier2_real, "git_info", lambda: {"git_sha": None, "git_dirty": None})
    with pytest.raises(ValueError, match="git_dirty"):
        refresh._require_clean_tree({"tier1": refresh.CANONICAL["tier1"]})


def test_delta_and_tier1_summary_arithmetic():
    def clocks(graph, wall):
        return {"cuda_graph_ms": {"median_ms": graph}, "eager_event_ms": {"median_ms": graph},
                "wall_ms": {"median_ms": wall}}

    before = {"shapes": [{"shape": [512, 2048], "methods": {
        "muon_ps3_blackwell": clocks(0.125, 0.3), "muon_ss3_54": clocks(0.14, 0.39)}}]}
    after = {"shapes": [
        {"shape": [512, 2048], "methods": {"muon_ps3_blackwell": clocks(0.125, 0.2),
                                           "muon_ss3_54": clocks(0.126, 0.195), "new": clocks(1.0, 1.0)}},
        {"shape": [8, 8], "methods": {"muon_ss3_54": clocks(1.0, 1.0)}}]}
    rows = refresh.delta_vs_cycle2(after, before)
    assert len(rows) == 1 and set(rows[0]["methods"]) == {"muon_ps3_blackwell", "muon_ss3_54"}
    cell = rows[0]["methods"]["muon_ss3_54"]
    assert cell["cuda_graph_ms"]["delta_us"] == pytest.approx(-14.0)
    assert cell["cuda_graph_ms"]["vs_reference_before"] == pytest.approx(0.125 / 0.14)
    assert cell["cuda_graph_ms"]["vs_reference_after"] == pytest.approx(0.125 / 0.126)
    assert cell["wall_ms"]["speedup"] == pytest.approx(2.0)
    full = {"shapes": [{"shape": [8, 8], "methods": {
        **{name: clocks(2.0, 3.0) for name in refresh.GRAPH_REFERENCES},
        **{f"muon_{name}": clocks(1.0, 2.0) for name in refresh.SS3}}}]}
    speedups = refresh.ss3_speedups(full)[0]
    assert speedups["cuda_graph_ms"]["muon_ss3_54"] == {name: 2.0 for name in refresh.GRAPH_REFERENCES}
    assert speedups["wall_ms"]["muon_ss3_34"]["muon_pe_eager"] == 1.5

    def case(suite, error, sigma, diverged, fast, slow):
        quality = {"relative_frobenius_error": error, "sigma_max": sigma, "diverged": diverged,
                   "nonfinite": False, "graph_vs_eager_max_abs": 0.0}
        return {"suite": suite, "methods": {"a": {**quality, **clocks(fast, fast)},
                                            "ps3_55_blackwell_fp16": {**quality, **clocks(slow, slow)}}}

    summary = refresh.summarize_tier1([case("x", 0.1, 1.0, False, 1.0, 2.0), case("x", 0.3, 1.2, False, 1.0, 4.0),
                                       case("y", 0.2, 1.1, True, 2.0, 2.0)])
    row = summary["all"]["a"]
    assert (row["cases"], row["diverged"], row["capture_errors"]) == (3, 1, 0)
    assert row["median_relative_frobenius_error"] == 0.2 and row["max_relative_frobenius_error"] == 0.3
    assert row["mean_relative_frobenius_error"] == pytest.approx(0.2) and row["max_sigma_max"] == 1.2
    assert row["median_case_wall_ms"] == 1.0
    assert row["median_speedup_vs_ps3_55_blackwell_fp16_cuda_graph_ms"] == 2.0
    assert summary["x"]["a"]["cases"] == 2 and summary["y"]["a"]["diverged"] == 1


@needs_cuda
def test_quick_tier1_cli_writes_strict_json(tmp_path):
    if not refresh.TRAJECTORIES.is_file():
        pytest.skip(f"missing {refresh.TRAJECTORIES}")
    output = tmp_path / "nested" / "tier1_quick.json"
    refresh.main(["--quick", "--part", "tier1", "--tier1-output", str(output)])

    def reject(constant):
        raise AssertionError(f"non-strict JSON constant {constant}")

    def unique(pairs):
        keys = [key for key, _ in pairs]
        assert len(keys) == len(set(keys)), "duplicate JSON key"
        return dict(pairs)

    result = json.loads(output.read_text(), parse_constant=reject, object_pairs_hook=unique)
    env, settings = result["env"], result["settings"]
    assert len(env["git_sha"]) == 40 and env["git_dirty"] in (True, False)
    assert set(env["loadavg"]) == set(env["gpu_telemetry"]) == {"start", "end"}
    assert settings["quick"] is True and settings["warmups"] == 5 and settings["cases_run"] == 3
    assert settings["cases_available"] == {"total": 117, "synthetic": 72, "harvested": 45}
    assert settings["methods"] == refresh.TIER1_METHODS and len(settings["trajectories_sha256"]) == 64
    assert settings["allow_fp16_reduced_precision_reduction"] is False
    assert {case["suite"] for case in result["cases"]} >= {"harvested_momentum"}
    for case in result["cases"]:
        for name in refresh.TIER1_METHODS:
            cell = case["methods"][name]
            assert cell["finite"] is True and cell["graph_vs_eager_max_abs"] == 0.0
            assert all(cell[clock]["median_ms"] > 0 and "samples_ms" not in cell[clock] for clock in refresh.CLOCKS)
    assert result["checks"]["capture_errors"] == 0 and result["checks"]["graph_vs_eager_max_abs"] == 0.0
    assert all(count == 0 for count in result["checks"]["diverged"].values())
    counts = result["launch_counts"]
    assert len(counts["baseline_commit"]) == 40
    row = counts["shapes"]["64x256"]
    assert {name: (cell["before"], cell["after"]) for name, cell in row.items()} == {
        "ss3_54": (17, 16), "ss3_53": (16, 15), "ss3_35": (16, 15), "ss3_34": (15, 14), "ps3_55": (16, 16)}
    assert all(cell["memcpy_dtoh_after"] == 0 for cell in row.values())
