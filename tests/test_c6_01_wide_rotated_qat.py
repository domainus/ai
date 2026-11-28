"""Checks for Cycle 6 C6-01: wide rotations, rotated-space NVFP4 W4A4 QAT and the C6-01 benchmark driver.

Math tests (CPU): orthogonality and structure of every rotation family, the straight-through
gradients of ``RotatedFakeQuantLinear`` and the dual loss step. Hardware tests (sm_100+): the cross-mix
+ fused-kernel path against its eager reference and its FP32 emulation, and the CUPTI evidence.
Driver tests: the CLI guard, the atomic writer, one ``--synthetic-shakedown`` run (generated Markov
shards; it cannot read real text) with validator tampering probes and, once committed, the canonical
artifact.

No test here scores the real val.bin or test.bin: the driver has no real-data mode besides the
canonical run, and the shakedown fixture fails if the real data directory is even looked up.
"""

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from research.polar_opt import harness_tier2_real as h
from research.polar_opt.harness_tier2_lm import Transformer
from research.quant_ptq import micro_rotated_ptq as q

rq = pytest.importorskip("research.quant_ptq.rotated_qat")  # needs Triton (the NVFP4 kernel module)
b = pytest.importorskip("research.quant_ptq.bench_c6_01_wide_rotated_qat")
c502, c503, e2e, nb = b.c502, b.c503, b.e2e, b.nb

ROOT = Path(__file__).resolve().parents[1]
CUDA = torch.cuda.is_available()
FP4 = CUDA and hasattr(F, "scaled_mm") and torch.cuda.get_device_capability()[0] >= 10
needs_fp4 = pytest.mark.skipif(not FP4, reason="needs native NVFP4 scaled_mm (sm_100+)")
# SHA-256 of the committed canonical result (set by the commit that adds the result).
CANONICAL_SHA256 = "c490a415568244bd1e2489d0004adc025d856e92452cb976c05efc2dcdf66bf9"

if CUDA:
    # Kineto decides once per process whether CUPTI runs: the first profiler session must ask for CUDA
    # activities. Modules are imported at collection, before any test of any module profiles on the CPU only.
    rq.prime_cupti()

WIDTHS = (64, 128, 192, 384, 512, 768, 1408, 2048)


# ------------------------------------------------------------------ rotation math
@pytest.mark.parametrize("n", WIDTHS)
@pytest.mark.parametrize("family", ("hadamard_b64", "kron_hadamard_full"))
def test_wide_hadamard_is_a_symmetric_orthogonal_involution(n, family):
    rot, fit = rq.make_rotation(family, n=n)
    assert fit == {} and isinstance(rot, rq.WideHadamard)
    R = rot.matrix(n)
    eye = torch.eye(n, dtype=torch.float64)
    assert R.shape == (n, n)
    assert torch.allclose(R @ R.T, eye, atol=1e-12) and torch.allclose(R, R.T, atol=1e-12)
    x = torch.randn(3, 5, n, dtype=torch.float64, generator=torch.Generator().manual_seed(n))
    assert torch.allclose(rot.inverse(rot.forward(x)), x, atol=1e-12)
    assert torch.allclose(rot.forward(x), x @ R.T, atol=1e-12)
    assert torch.allclose(rot.dense(x), rot.forward(x), atol=1e-12)  # the QAT path is the same map
    H = torch.randn(n, n, dtype=torch.float64)
    H = H @ H.T
    assert torch.allclose(rot.rotate_hessian(H), R @ H @ R.T, atol=1e-9)


@pytest.mark.parametrize("n", WIDTHS)
def test_block_64_and_full_width_match_their_definitions(n):
    x = torch.randn(4, n, dtype=torch.float64, generator=torch.Generator().manual_seed(n + 1))
    b64, _ = rq.make_rotation("hadamard_b64", n=n)
    assert torch.allclose(b64.forward(x), q.fwht(x, 64), atol=1e-12)  # block-64 Walsh-Hadamard
    full, _ = rq.make_rotation("kron_hadamard_full", n=n)
    block, pow2, odd = rq.full_factors(n)
    assert odd * pow2 * block == n and odd % 2 == 1 and block == min(128, n & -n)
    dense = torch.kron(torch.kron(rq.dct4_matrix(odd), rq.hadamard_matrix(pow2)), rq.hadamard_matrix(block))
    assert torch.allclose(full.matrix(n), dense, atol=1e-12)  # A_m (x) H_B with A_m = DCT4_odd (x) H_pow2
    if n & (n - 1) == 0:
        assert odd == 1 and torch.allclose(full.forward(x), q.fwht(x, n), atol=1e-12)  # the full-width FWHT
    else:
        assert odd > 1
    # Full width mixes every input into every output; block 64 leaves the other blocks at exactly zero.
    assert full.matrix(n).abs().min() > 1e-4
    assert (b64.matrix(n) == 0).double().mean().item() == pytest.approx(1 - 64 / n)


def test_orthogonal_factors():
    for m in (1, 3, 11, 5):
        C = rq.dct4_matrix(m)
        assert torch.allclose(C, C.T, atol=1e-14) and torch.allclose(C @ C, torch.eye(m, dtype=torch.float64), atol=1e-13)
    H = rq.hadamard_matrix(16)
    assert torch.allclose(H.abs(), torch.full((16, 16), 0.25, dtype=torch.float64)) and torch.allclose(H, H.T)
    assert rq.full_factors(1408) == (128, 1, 11) and rq.full_factors(768) == (128, 2, 3)
    assert rq.full_factors(512) == (128, 4, 1) and rq.full_factors(2048) == (128, 16, 1) and rq.full_factors(64) == (64, 1, 1)
    with pytest.raises(ValueError, match="group size"):
        rq.full_factors(24)
    with pytest.raises(ValueError, match="power of two"):
        rq.CrossMix(96, 48)
    with pytest.raises(ValueError, match="divides"):
        rq.CrossMix(96, 64)
    with pytest.raises(ValueError, match="mix must be"):
        rq.CrossMix(128, 64, torch.eye(3))
    with pytest.raises(ValueError, match="Expected 128 inputs"):
        rq.CrossMix(128, 64).forward(torch.zeros(2, 64))


def test_cross_mix_keeps_the_position_inside_a_group_and_composes_with_the_kernel_rotation():
    """R = R16 . C: the cross mix acts on whole groups of 16, so it commutes with any block-16 rotation."""
    n = 384
    cross = rq.make_cross("full", n)
    x = torch.randn(7, n, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(cross.forward(cross.forward(x)), x, atol=1e-12)
    assert torch.allclose(q.fwht(cross.forward(x), 16), cross.forward(q.fwht(x, 16)), atol=1e-12)
    spike = torch.zeros(1, n, dtype=torch.float64)
    spike[0, 5] = 1.0  # position 5 of group 0 stays at position 5 of every group
    out = cross.forward(spike).reshape(n // 16, 16)
    assert (out[:, 5] != 0).all() and out[:, [i for i in range(16) if i != 5]].abs().max() == 0
    assert rq.make_cross(None, n) is None and rq.CrossMix(n, 16).identity and not cross.identity
    assert rq.cross_of(q.BlockHadamard(16)) is None and rq.cross_of(rq.make_rotation("hadamard_b64", n=n)[0]) is not None
    with pytest.raises(ValueError, match="Unknown cross-block mix"):
        rq.make_cross("b128", n)


def test_families_methods_and_kernel_rotations():
    assert set(rq.FAMILIES) == {"identity", "hadamard_b16", "adaptive_butterfly_b16", "hadamard_b64",
                                "kron_hadamard_full", "cross_b64_butterfly_b16", "cross_full_butterfly_b16"}
    assert rq.WIDE_FAMILIES == ("hadamard_b64", "kron_hadamard_full", "cross_b64_butterfly_b16",
                                "cross_full_butterfly_b16")
    assert rq.QAT_FAMILIES == ("identity", "hadamard_b16", "hadamard_b64", "kron_hadamard_full")
    for family in rq.FAMILIES:
        assert rq.kernel_rotation(family) in nb.ROTATIONS
        for rounding in ("rtn", "gptq"):
            for awq in (False, True):
                assert rq.parse_method(rq.method_name(family, rounding, awq)) == (family, rounding, awq)
    assert rq.method_name("identity", "gptq") == "gptq" and rq.method_name("hadamard_b64", "rtn") == "hadamard_b64_rtn"
    assert rq.parse_method("awq_w4a4_adaptive_butterfly_b16_gptq") == ("adaptive_butterfly_b16", "gptq", True)
    assert rq.is_wide("awq_w4a4_kron_hadamard_full_gptq") and not rq.is_wide("hadamard_b16_gptq")
    assert rq.kernel_rotation("cross_full_butterfly_b16") == "adaptive_butterfly_b16"
    assert rq.kernel_rotation("kron_hadamard_full") == "hadamard_b16" and rq.kernel_rotation("identity") == "none"
    for bad in ("hadamard_b32_gptq", "identity_gptq", "hadamard_b64", "hadamard_b64_gptx"):
        with pytest.raises(ValueError, match="Unknown"):
            rq.parse_method(bad)
    # The C5 families are the C5 objects: the same rotation, so the same PTQ rows as C5-02 / C5-03.
    assert type(rq.make_rotation("identity", n=64)[0]) is q.Rotation
    assert type(rq.make_rotation("hadamard_b16", n=64)[0]) is q.BlockHadamard


@pytest.mark.parametrize("family", ("adaptive_butterfly_b16", "cross_b64_butterfly_b16", "cross_full_butterfly_b16"))
def test_butterfly_families_fit_per_layer_and_stay_orthogonal(family):
    torch.manual_seed(0)
    W = torch.randn(3, 24, 192) * 0.05
    rot, fit = rq.make_rotation(family, W, butterfly_steps=4)
    assert fit["steps"] == 4 and fit["layers"] == 3 and fit["weight_sq_error_fitted"] <= fit["weight_sq_error_hadamard_start"]
    assert rot.angles.shape == (3, 12, 4, 8)
    z = rot.forward(W)
    assert torch.allclose(rot.inverse(z), W, atol=1e-5)
    assert torch.allclose(z.square().sum((-2, -1)), W.square().sum((-2, -1)), rtol=1e-5)  # an isometry per layer
    for i in range(3):  # the per-layer rotation is the stack rotation of that layer
        assert torch.allclose(rq.layer_rotation(rot, i).forward(W[i]), z[i], atol=1e-6)
    moved = rq.rotation_to(rot, "cpu")
    assert type(moved) is type(rot) and torch.equal(moved.angles, rot.angles)
    with pytest.raises(ValueError, match="needs the weight stack"):
        rq.make_rotation(family, n=192)


# ------------------------------------------------------------------ STE fake quantization
def test_identity_fake_quant_is_the_c5_02_module_bit_for_bit():
    torch.manual_seed(1)
    lin = nn.Linear(128, 24, bias=False)
    x = torch.randn(2, 9, 128, requires_grad=True)
    y1 = rq.RotatedFakeQuantLinear(lin, rq.qat_rotation("identity", 128))(x)
    g1 = torch.autograd.grad(y1.square().sum(), [x, lin.weight])
    y2 = c502.FakeQuantLinear(lin, rq.FMT)(x)
    g2 = torch.autograd.grad(y2.square().sum(), [x, lin.weight])
    assert torch.equal(y1, y2) and all(torch.equal(a, c) for a, c in zip(g1, g2))


@pytest.mark.parametrize("family,n", [("hadamard_b16", 128), ("hadamard_b64", 192), ("kron_hadamard_full", 192),
                                      ("kron_hadamard_full", 128)])
def test_rotated_fake_quant_forward_and_straight_through_gradients(family, n):
    """Forward: Q(x R^T) Q(W R^T)^T. Backward: both quantizers are the identity, so with upstream g:
    dL/dW = (g^T Q(x R^T)) R and dL/dx = (g Q(W R^T)) R."""
    torch.manual_seed(2)
    lin = nn.Linear(n, 24, bias=False)
    x = (torch.randn(3, 5, n) * 2).requires_grad_(True)
    rot = rq.qat_rotation(family, n)
    wrapper = rq.RotatedFakeQuantLinear(lin, rot)
    y = wrapper(x)
    g = torch.randn_like(y)
    grad_x, grad_w = torch.autograd.grad((y * g).sum(), [x, lin.weight])
    R = rot.matrix(n, dtype=torch.float32)
    xq = q.quantize_rtn(x.detach().reshape(-1, n) @ R.T, rq.FMT)
    wq = q.quantize_rtn(lin.weight.detach() @ R.T, rq.FMT)
    g2 = g.reshape(-1, 24)
    assert torch.allclose(y.detach().reshape(-1, 24), xq @ wq.T, atol=2e-5)
    assert torch.allclose(grad_w, (g2.T @ xq) @ R, atol=2e-4)
    assert torch.allclose(grad_x.reshape(-1, n), (g2 @ wq) @ R, atol=2e-4)
    # The quantizers change the forward value: the output is not the BF16 linear.
    plain = F.linear(x, lin.weight).detach()
    assert not torch.allclose(y.detach(), plain, atol=1e-3)
    wrapper.active = False
    assert torch.equal(wrapper(x), F.linear(x, lin.weight))


def test_fake_quant_context_wraps_and_restores_every_linear():
    config = b.PRESETS["shakedown"]
    torch.manual_seed(0)
    model = Transformer(config)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with rq.fake_quant(model, "kron_hadamard_full") as wrappers:
        assert set(wrappers) == set(q.quantized_linears(model)) and len(wrappers) == 5 * config.layers
        assert all(isinstance(m, rq.RotatedFakeQuantLinear) and m.active for m in wrappers.values())
        assert isinstance(wrappers["blocks.0.down"].rotation, rq.WideHadamard)
        rq.set_active(wrappers, False)
        assert not any(m.active for m in wrappers.values())
    assert all(type(m) is nn.Linear for m in q.quantized_linears(model).values())
    after = model.state_dict()
    assert list(after) == list(before) and all(torch.equal(after[k], v) for k, v in before.items())
    for family in ("adaptive_butterfly_b16", "cross_full_butterfly_b16"):
        with pytest.raises(ValueError, match="parameter-free"):
            with rq.fake_quant(model, family):
                pass


class _NoStep:
    def step(self, timer=None):
        pass


def _grads(model):
    """Gradients in parameter order (a wrapped linear keeps its slot, so the order does not change)."""
    return [p.grad.clone() for p in model.parameters()]


def test_train_step_plain_matches_the_harness_and_dual_mixes_both_passes():
    config = b.PRESETS["shakedown"]
    torch.manual_seed(0)
    model = Transformer(config)
    g = torch.Generator().manual_seed(5)
    tokens = torch.randint(0, 512, (config.grad_accum, config.micro_batch, config.sequence_length + 1), generator=g)
    batch = (tokens[..., :-1], tokens[..., 1:])
    loss = rq.train_step(model, batch, _NoStep())
    mine = _grads(model)
    model.zero_grad(set_to_none=True)
    theirs_loss = h.train_step(model, batch, _NoStep())
    assert torch.equal(loss, theirs_loss) and all(torch.equal(a, c) for a, c in zip(mine, _grads(model)))
    model.zero_grad(set_to_none=True)
    with rq.fake_quant(model, "hadamard_b64") as wrappers:
        loss_q = rq.train_step(model, batch, _NoStep(), wrappers)  # the QAT loss
        quant = _grads(model)
        model.zero_grad(set_to_none=True)
        loss_dual = rq.train_step(model, batch, _NoStep(), wrappers, bf16_weight=0.25)
        dual = _grads(model)
        assert all(m.active for m in wrappers.values())  # the BF16 pass switches the wrappers back on
    assert torch.allclose(loss_dual, loss_q, atol=1e-6) and loss_q != loss  # the dual step reports the W4A4 loss
    assert len(dual) == len(mine) == len(quant)
    for plain, fake, both in zip(mine, quant, dual):
        assert torch.allclose(both, 0.25 * plain + 0.75 * fake, atol=1e-5)
    assert any(not torch.allclose(fake, plain, atol=1e-6) for plain, fake in zip(mine, quant))


# ------------------------------------------------------------------ synthetic shards and activation statistics
def test_synthetic_shards_are_seeded_disjoint_streams_and_never_the_real_text(tmp_path):
    spec = {"seed": 3, "documents": {"train": 40, "val": 8, "test": 8}, "document_tokens": 100}
    one = rq.write_synthetic_shards(tmp_path / "a", **spec)
    two = rq.write_synthetic_shards(tmp_path / "b", **spec)
    other = rq.write_synthetic_shards(tmp_path / "c", **{**spec, "seed": 4})
    assert one == two and one["source"]["name"] == rq.SYNTHETIC_SOURCE == "synthetic-markov"
    assert one["splits"]["train"]["sha256"] != other["splits"]["train"]["sha256"]
    digests = {v["sha256"] for v in one["splits"].values()}
    assert len(digests) == 3 and not digests & set(q.EXPECTED_SHA256.values())
    shards = c502.Shards(tmp_path / "a", verify=False, expected=None)
    view = shards.open("train", "val", "test")
    assert {k: len(v) for k, v in view.items()} == {"train": 4000, "val": 800, "test": 800}
    train = np.asarray(view["train"]).reshape(40, 100)
    assert (train[:, 0] == 0).all() and train[:, 1:].min() >= 1 and train.max() < 512
    assert hashlib.sha256((tmp_path / "a" / "val.bin").read_bytes()).hexdigest() == one["splits"]["val"]["sha256"]
    # A first-order chain with 6 successors per token: every token is followed by at most 6 distinct tokens.
    followers = {}
    for row in train:
        for a, c in zip(row[1:-1], row[2:]):
            followers.setdefault(int(a), set()).add(int(c))
    assert max(len(v) for v in followers.values()) <= 6


def test_rotated_activation_statistics_match_a_direct_computation():
    config = b.PRESETS["shakedown"]
    torch.manual_seed(0)
    model = Transformer(config).eval()
    windows = torch.randint(0, 512, (8, config.sequence_length), generator=torch.Generator().manual_seed(1)).numpy()
    linears = q.quantized_linears(model)
    maps = {f: {n: rq.make_rotation(f, n=m.in_features)[0] for n, m in linears.items()}
            for f in ("identity", "hadamard_b16", "kron_hadamard_full")}
    stats = rq.rotated_activation_statistics(model, maps, windows, 4, torch.device("cpu"))
    assert list(stats) == list(maps) and all(set(v) == set(linears) for v in stats.values())
    assert all(set(s) == set(rq.ACTIVATION_METRICS) for v in stats.values() for s in v.values())
    captured = []
    handle = linears["blocks.1.down"].register_forward_pre_hook(lambda m, i: captured.append(i[0].detach().reshape(-1, 192)))
    with torch.no_grad():
        for first in (0, 4):
            model(torch.from_numpy(windows[first:first + 4]))
    handle.remove()
    for family in maps:
        zs = [maps[family]["blocks.1.down"].forward(x.float()) for x in captured]
        z = torch.cat(zs).double()
        got = stats[family]["blocks.1.down"]
        assert got["excess_kurtosis"] == pytest.approx(q.excess_kurtosis(z), rel=1e-9)
        assert got["max_over_rms"] == pytest.approx(q.max_over_rms(z), rel=1e-9)
        assert got["group16_max_over_rms"] == pytest.approx(q.group_max_over_rms(z), rel=1e-9)
        err = sum((q.quantize_rtn(x, rq.FMT) - x).double().square().sum() for x in zs)  # one tensor scale per batch
        assert got["nvfp4_relative_sq_error"] == pytest.approx((err / z.square().sum()).item(), rel=1e-9)
        assert 0 <= got["group16_outlier_share"] <= 1
    # A rotation cannot change the energy, but it spreads a spike: one hot channel has far lower kurtosis after it.
    spiky = torch.randn(4096, 192, generator=torch.Generator().manual_seed(2))
    spiky[:, 7] *= 40
    assert q.excess_kurtosis(maps["kron_hadamard_full"]["blocks.1.down"].forward(spiky)) < 0.1 * q.excess_kurtosis(spiky)


# ------------------------------------------------------------------ hardware path
@needs_fp4
@pytest.mark.parametrize("family", rq.WIDE_FAMILIES)
@pytest.mark.parametrize("n", (192, 512, 1408))
def test_cross_mix_plus_kernel_matches_the_reference_and_the_emulation(family, n):
    torch.manual_seed(n)
    W = torch.randn(1, 96, n, device="cuda") * 0.05
    x = (torch.randn(300, n, device="cuda") * 2).to(torch.bfloat16)
    rot, _ = rq.make_rotation(family, W, butterfly_steps=3)
    cross, layer_rot = rq.cross_of(rot), rq.layer_rotation(rot, 0)
    angles = rot.angles[0] if hasattr(rot, "angles") else None
    mixed = cross.forward(x.float())
    # The fused quantizer on the mixed operand is the eager reference bit for bit ...
    tables = nb.butterfly_tables(angles) if angles is not None else None
    got = nb.quantize_activation(mixed, rq.kernel_rotation(family), tables)
    assert c503._same_operands(got, nb.quantize_activation_ref(mixed, rq.kernel_rotation(family), angles))
    # ... and its values are NVFP4 round-to-nearest of the fully rotated activation x R^T.
    assert torch.equal(nb.dequantize(got, *mixed.shape), q.quantize_rtn(layer_rot.forward(x.float()), rq.FMT))
    enc = q.nvfp4_encode(layer_rot.forward(W[0]))
    layer = nb.NVFP4Linear.from_encoding(enc["codes"], enc["block_scale"], enc["global_scale"],
                                         rotation=rq.kernel_rotation(family), angles=angles)
    assert torch.equal(nb.dequantized_weight(layer).view(torch.int32), enc["values"].contiguous().view(torch.int32))
    with torch.autocast("cuda", dtype=torch.bfloat16):  # CrossInput must keep the mix in FP32 under autocast
        hw = rq.CrossInput(layer, cross)(x).float()
        emulated = rq.CrossInput(e2e.EmulatedW4A4Linear(layer), cross)(x).float()
    exact = q.quantize_rtn(layer_rot.forward(x.float()), rq.FMT) @ enc["values"].T  # the QAT forward value
    assert ((hw - emulated).norm() / emulated.norm()).item() < 1e-3
    assert ((emulated - exact).norm() / exact.norm()).item() < 5e-3  # BF16 rounding of the output


@needs_fp4
def test_cross_mix_parity_record():
    record = b.cross_mix_parity(torch.device("cuda"))
    assert record["passed"] and record["bitwise_failures"] == [] and record["cases"] == 6 * 2 * len(rq.WIDE_FAMILIES) == 48


@needs_fp4
def test_forward_profile_counts_fp4_gemms_quantizer_kernels_and_fallbacks():
    config = b.PRESETS["shakedown"]
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = Transformer(config).eval()
    tokens = torch.randint(0, 512, (2, config.sequence_length), device="cuda")
    n = 5 * config.layers
    bf16 = rq.forward_profile(model, tokens)
    assert bf16["scaled_mm_dispatches"] == 0 and bf16["cupti"]["fp4_gemm_kernels"] == 0
    assert bf16["cupti"]["nvfp4_quant_kernels"] == bf16["cupti"]["nvfp4_amax_kernels"] == 0
    assert bf16["cupti"]["non_fp4_gemm_kernels"] >= n + 1  # the block linears and the LM head
    linears = q.quantized_linears(model)
    layers = {name: nb.NVFP4Linear.from_weight(m.weight.float(), "hadamard_b16") for name, m in linears.items()}
    previous = e2e._swap(model, layers)
    hw = rq.forward_profile(model, tokens)
    assert hw["scaled_mm_dispatches"] == n and hw["cupti"]["fp4_gemm_kernels"] == n
    assert hw["cupti"]["nvfp4_quant_kernels"] == hw["cupti"]["nvfp4_amax_kernels"] == n
    assert hw["cupti"]["non_fp4_gemm_kernels"] == bf16["cupti"]["non_fp4_gemm_kernels"] - n  # no BF16 fallback
    assert b._hw_counts(model) == {"nvfp4_modules": n, "bf16_block_linears": 0, "emulated_modules": 0,
                                   "cross_mix_modules": 0}
    # One block linear back to BF16: both the profiler and the module count see the fallback.
    e2e._swap(model, {"blocks.1.out": previous["blocks.1.out"]})
    partial = rq.forward_profile(model, tokens)
    assert partial["scaled_mm_dispatches"] == partial["cupti"]["fp4_gemm_kernels"] == n - 1
    assert partial["cupti"]["non_fp4_gemm_kernels"] == hw["cupti"]["non_fp4_gemm_kernels"] + 1
    assert b._hw_counts(model)["bf16_block_linears"] == 1
    for layer in layers.values():  # the static mode runs no amax pass
        layer.act_global_scale = torch.tensor(1.0, device="cuda")
    e2e._swap(model, layers)
    static = rq.forward_profile(model, tokens)
    assert static["cupti"]["nvfp4_amax_kernels"] == 0 and static["cupti"]["nvfp4_quant_kernels"] == n
    cross = rq.make_cross("full", config.width)
    e2e._swap(model, {"blocks.0.out": c502.ScaledInput(rq.CrossInput(layers["blocks.0.out"], cross),
                                                       torch.ones(config.width, device="cuda"))})
    assert b._hw_counts(model) == {"nvfp4_modules": n, "bf16_block_linears": 0, "emulated_modules": 0,
                                   "cross_mix_modules": 1}


def test_kernel_classification():
    fp4 = "cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256"
    assert rq.classify_kernel(fp4) == "fp4_gemm_kernels"
    assert rq.classify_kernel("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64>") == \
        "non_fp4_gemm_kernels"
    assert rq.classify_kernel("_amax_kernel") == "nvfp4_amax_kernels"
    assert rq.classify_kernel("_quant_kernel") == "nvfp4_quant_kernels"
    assert rq.classify_kernel("void at::native::vectorized_elementwise_kernel<4, ...>") is None
    assert rq.classify_kernel("_c4_02_quant_kernel") is None  # only the kernels of the current module count


# ------------------------------------------------------------------ protocol, method grids and statistics
def test_canonical_protocol_is_the_predeclared_one():
    p = b.CANONICAL
    assert p["seeds"] == [1, 2, 3] and p["max_eval_windows"] is None and p["awq_alphas"] == [0.0, 0.25, 0.5]
    assert p["calibration_sequences"] == 128 and p["calibration_seed"] == 20261126 and p["dual_bf16_weight"] == 0.5
    assert p["emulation_parity_nats"] == 1e-3 and p["butterfly_steps"] == 150
    assert p["part_ab"] == {"preset": "50m", "optimizer": "muon_ss3_54", "steps": 320, "cooldown_from": 272,
                            "lr": 0.02, "auxiliary_lr": 0.001}
    assert p["part_c"] == {"preset": "120m", "optimizer": "muon_ps3_blackwell", "baseline": "c5_03_preserved",
                           "baseline_steps": 640, "baseline_warmup_steps": 64, "cooldown_steps": 96, "lr": 0.02,
                           "auxiliary_lr": 0.001}
    assert b.PRESETS["50m"] == h.PRESETS["50m"] and b.PRESETS["120m"] == h.PRESETS["120m"]
    # Part C uses the C5-03 recipe (optimizer, LR, calibration draw) so its baseline rows re-measure C5-03.
    c5 = c503.CANONICAL["part_c"]
    assert all(p["part_c"][k] == c5[k] for k in ("preset", "optimizer", "lr", "auxiliary_lr"))
    assert (p["part_c"]["baseline_steps"], p["part_c"]["baseline_warmup_steps"]) == (c5["steps"], c5["warmup_steps"])
    assert (p["calibration_sequences"], p["calibration_seed"], p["butterfly_steps"], p["gptq_damp"]) == \
        (c5["calibration_sequences"], c5["calibration_seed"], c5["butterfly_steps"], c5["gptq_damp"])
    assert list(b.ARMS_AB) == ["muon_ss3_54", "qat_identity_cooldown", "qat_identity_full", "qat_b16_cooldown",
                               "qat_wide_cooldown", "qat_wide_dual_cooldown"]
    assert b.ARMS_C == ("c5_03_baseline", "bf16_continue", "qat_cooldown")
    assert b.SHAKEDOWN["part_ab"]["preset"] == b.SHAKEDOWN["part_c"]["preset"] == "shakedown"
    assert b.SHAKEDOWN["part_c"]["baseline"] == "trained_in_run"


def test_pinned_baselines_are_the_committed_c5_03_checkpoints():
    reference = b.c5_03_reference()
    source = json.loads((ROOT / c503.RESULT_REPO_PATH).read_text())
    assert reference["result_sha256"] == hashlib.sha256((ROOT / c503.RESULT_REPO_PATH).read_bytes()).hexdigest()
    for s in source["part_c"]["seeds"]:
        pin = b.PINNED_BASELINES[s["seed"]]
        assert pin["file"] == s["training"]["checkpoint_file"]
        assert pin["file"] == c503.checkpoint_name(s["seed"], source["protocol"]["part_c"])
        assert pin["checkpoint_sha256"] == s["training"]["checkpoint_sha256"]
        assert pin["checkpoint_file_sha256"] == s["training"]["checkpoint_file_sha256"]
        ref = reference["seeds"][str(s["seed"])]
        assert ref["bf16"] == {"val": s["baseline_val"]["nll"], "test": s["baseline_test"]["nll"]}
        row = next(r for r in s["grid"] if r["method"] == b.C5_BEST and r["mode"] == "w4a4_hw_static_bf16cal")
        assert ref["methods"][b.C5_BEST]["w4a4_hw_static"] == {"val": row["val_nll"], "test": row["test_nll"]}
    pin = b.PINNED_BASELINES[2]
    assert b.preserved_baseline_record(2) == {
        "source": "c5_03_preserved", "part": "c", "arm": "c5_03_baseline", "seed": 2, "checkpoint_file": pin["file"],
        "checkpoint_sha256": pin["checkpoint_sha256"], "checkpoint_file_sha256": pin["checkpoint_file_sha256"]}


def test_method_grids_follow_the_selections():
    sel = {"qat_wide_family": "kron_hadamard_full", "best_wide_ptq_method": "awq_w4a4_hadamard_b64_gptq",
           "transfer_family": "kron_hadamard_full", "transfer_loss": "dual"}
    assert b.unit_methods("ab", b.MUON, {}) == list(b.PROBE_METHODS) and len(b.PROBE_METHODS) == 15
    assert b.emulated_method("ab", b.MUON, {}) == "kron_hadamard_full_gptq"
    assert b.arm_qat("ab", "qat_wide_dual_cooldown", sel) == {"family": "kron_hadamard_full", "schedule": "cooldown",
                                                             "loss": "dual"}
    assert b.arm_qat("ab", "qat_identity_full", sel) == {"family": "identity", "schedule": "full", "loss": "qat"}
    assert b.arm_qat("ab", b.MUON, sel) is None and b.arm_qat("c", b.CONTINUE, sel) is None
    assert b.unit_methods("ab", "qat_identity_cooldown", sel) == ["rtn", "gptq", b.C5_HADAMARD, b.C5_BEST]
    assert b.unit_methods("ab", "qat_b16_cooldown", sel) == ["hadamard_b16_rtn", b.C5_HADAMARD, b.C5_BEST]
    assert b.unit_methods("ab", "qat_wide_cooldown", sel) == ["kron_hadamard_full_rtn", "kron_hadamard_full_gptq",
                                                             b.C5_HADAMARD, b.C5_BEST]
    assert b.unit_methods("c", b.BASELINE, sel) == [b.C5_HADAMARD, b.C5_BEST, "kron_hadamard_full_gptq",
                                                    "awq_w4a4_hadamard_b64_gptq"]
    assert b.unit_methods("c", b.QAT_C, sel) == ["kron_hadamard_full_rtn", "kron_hadamard_full_gptq", b.C5_HADAMARD,
                                                 b.C5_BEST, "awq_w4a4_hadamard_b64_gptq"]
    assert b.emulated_method("c", b.QAT_C, sel) == b.emulated_method("c", b.CONTINUE, sel) == "kron_hadamard_full_gptq"
    b16 = {**sel, "transfer_family": "hadamard_b16", "transfer_loss": "qat"}  # a block-16 transfer has no wide own method
    assert b.unit_methods("c", b.QAT_C, b16) == ["hadamard_b16_rtn", b.C5_HADAMARD, b.C5_BEST, "awq_w4a4_hadamard_b64_gptq"]
    assert b.emulated_method("c", b.QAT_C, b16) == b.C5_HADAMARD
    grid = b.expected_grid(["rtn", "gptq"], "gptq")
    assert grid == [("rtn", m) for m in b.MODES] + [("gptq", m) for m in b.MODES] + [("gptq", b.EMULATED_MODE)]
    # Every QAT arm carries the two C5-03 comparator methods, so a best-vs-best contrast is not handicapped.
    for arm in b.QAT_ARMS:
        assert {b.C5_HADAMARD, b.C5_BEST} <= set(b.unit_methods("ab", arm, sel))


def _fake_units(nll):
    """Units with only what the selection rules read: ``nll[(arm, method)]`` is the val NLL of every seed."""
    units = []
    for seed in (1, 2, 3):
        for arm in {a for a, _ in nll}:
            grid = [{"method": m, "mode": b.DYNAMIC, "val_nll": v + 0.001 * seed, "test_nll": 9.0 - v}
                    for (a, m), v in nll.items() if a == arm]
            units.append({"part": "ab", "seed": seed, "arm": arm, "grid": grid})
    return units


def test_selection_rules_read_val_only_and_break_ties_by_order():
    nll = {(b.MUON, m): 5.0 for m in b.PROBE_METHODS}
    nll[(b.MUON, "kron_hadamard_full_gptq")] = 4.99
    nll[(b.MUON, "cross_b64_butterfly_b16_gptq")] = 4.98
    nll[(b.MUON, b.C5_BEST)] = 4.97
    sel = b.select_probe(_fake_units(nll))
    assert sel == {"qat_wide_family": "kron_hadamard_full", "best_wide_ptq_method": "cross_b64_butterfly_b16_gptq",
                   "best_ptq_method": b.C5_BEST}
    nll[(b.MUON, "kron_hadamard_full_gptq")] = 5.0  # a tie of the two wide QAT families goes to the first (block 64)
    assert b.select_probe(_fake_units(nll))["qat_wide_family"] == "hadamard_b64"
    # The test NLL is the mirror image (9 - val): a rule that read it would pick the opposite method.
    assert b.best_method(_fake_units(nll), "ab", b.MUON, b.PROBE_METHODS, split="test") != sel["best_ptq_method"]
    for arm in b.QAT_ARMS:
        for m in b.unit_methods("ab", arm, sel):
            nll[(arm, m)] = 5.0
    nll[("qat_wide_dual_cooldown", b.C5_HADAMARD)] = 4.9
    nll[("qat_identity_cooldown", "rtn")] = 4.5  # the best arm overall, but not a rotated cooldown: not transferred
    transfer = b.select_transfer(_fake_units(nll), sel)
    assert transfer["transfer_arm"] == "qat_wide_dual_cooldown" and transfer["transfer_loss"] == "dual"
    assert transfer["transfer_family"] == "kron_hadamard_full"
    assert transfer["arm_best_method"]["ab"]["qat_identity_cooldown"] == "rtn"
    assert transfer["arm_best_method"]["ab"]["qat_wide_dual_cooldown"] == b.C5_HADAMARD
    assert len(b.selections_sha256(sel)) == 64 and b.selections_sha256(sel) != b.selections_sha256({**sel, "x": 1})
    with pytest.raises(ValueError, match="expected 3 seeds"):
        b.summed_nll(_fake_units(nll)[:5], "ab", b.MUON, "rtn")


def test_verdict_throughput_and_contrast_specs():
    assert b.verdict({"mean": -0.01, "p_holm": 0.01}) == "helps (significant)"
    assert b.verdict({"mean": 0.01, "p_holm": 0.01}) == "loses (significant)"
    assert b.verdict({"mean": -0.01, "p_holm": 0.2}) == "ties (helps direction, not significant)"
    assert b.verdict({"mean": 0.01, "p_holm": 0.05}) == "ties (loses direction, not significant)"
    assert b.verdict({"mean": 0.0, "p_holm": 1.0}) == "ties (no difference)"
    ref = {"arm": "muon", "training": {"first_step": 0, "steps": 6, "step_tokens": 100,
                                       "step_seconds": [9.0, 9.0, 9.0, 1.0, 2.0, 3.0]}}
    arm = {"arm": "qat", "training": {"first_step": 3, "steps": 6, "step_tokens": 100, "step_seconds": [4.0, 5.0, 6.0]}}
    t = b.throughput(arm, ref)  # the reference is the plain run over the same steps 3..5 (median 2 s), not all steps
    assert t["step_time_ratio"] == 2.5 and t["step_ms_median"] == 5000.0 and t["reference_step_ms_median"] == 2000.0
    assert t["tokens_per_second"] == 20.0 and t["reference_tokens_per_second"] == 50.0 and t["steps"] == [3, 6]
    sel = {"qat_wide_family": "hadamard_b64", "best_wide_ptq_method": "hadamard_b64_gptq",
           "transfer_family": "hadamard_b64", "transfer_loss": "qat",
           "arm_best_method": {"ab": {a: b.C5_BEST for a in b.ARMS_AB}, "c": {a: b.C5_BEST for a in b.ARMS_C}}}
    specs = b.contrast_specs(sel)
    names = [name for _, name, _, _ in specs]
    assert len(names) == len(set(names)) == 4 + 4 * len(b.QAT_ARMS) + 11
    by_name = {name: (part, x, y) for part, name, x, y in specs}
    # The deployment contrasts of Part C: the QAT cooldown against both C5-03 methods, its token-matched control
    # and the unquantized baseline.
    assert by_name["qat_best_vs_c5_03_awq_butterfly"] == ("c", (b.QAT_C, b.C5_BEST), (b.BASELINE, b.C5_BEST))
    assert by_name["qat_best_vs_c5_03_hadamard"][2] == (b.BASELINE, b.C5_HADAMARD)
    assert by_name["qat_own_gptq_vs_bf16_continue_same_method"] == ("c", (b.QAT_C, "hadamard_b64_gptq"),
                                                                   (b.CONTINUE, "hadamard_b64_gptq"))
    assert by_name["qat_best_vs_baseline_bf16"][2] == (b.BASELINE, b.BF16)
    assert by_name["qat_identity_full_bf16_vs_muon_bf16"] == ("ab", ("qat_identity_full", b.BF16), (b.MUON, b.BF16))
    assert by_name["qat_b16_cooldown_own_rtn_vs_muon_same_method"] == ("ab", ("qat_b16_cooldown", "hadamard_b16_rtn"),
                                                                      (b.MUON, "hadamard_b16_rtn"))
    # Every contrast side exists in the grid of its arm.
    for part, _, x, y in specs:
        for arm, method in (x, y):
            assert method == b.BF16 or method in b.unit_methods(part, arm, sel), (arm, method)


# ------------------------------------------------------------------ CLI guard, provenance and the atomic writer
def test_cli_guard(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("run should not be called")

    monkeypatch.setattr(b, "run", fail)
    args = b.parse_args([])
    assert args.canonical and not args.shakedown and args.output == b.CANONICAL_OUTPUT and args.protocol == b.CANONICAL
    assert args.checkpoint_dir == b.CANONICAL_CHECKPOINT_DIR
    assert b.CANONICAL_OUTPUT == ROOT / "results" / "cycle6_c6_01_wide_rotated_qat.json"
    located = b.parse_args(["--data-dir", "/x", "--baseline-checkpoint-dir", "/y"])  # locations: digests are pinned
    assert located.canonical and located.protocol == b.CANONICAL
    shake = b.parse_args(["--synthetic-shakedown", "--output", str(tmp_path / "s.json")])
    assert shake.shakedown and not shake.canonical and shake.protocol == b.SHAKEDOWN
    assert shake.work_dir == tmp_path / "s.json.work" and shake.checkpoint_dir == shake.work_dir / "checkpoints"
    other = tmp_path / "other" / "results" / b.CANONICAL_OUTPUT.name
    alias = tmp_path / "alias.json"
    alias.symlink_to(b.CANONICAL_OUTPUT)
    for argv in (
        ["--synthetic-shakedown"],  # no output
        ["--synthetic-shakedown", "--output", str(b.CANONICAL_OUTPUT)],
        ["--synthetic-shakedown", "--output", str(tmp_path / "cycle6_c6_01_copy.json")],  # reserved pattern
        ["--synthetic-shakedown", "--output", str(alias)],  # a symlink to the canonical file
        ["--synthetic-shakedown", "--output", str(tmp_path / "s.json"), "--data-dir", "/x"],
        ["--synthetic-shakedown", "--output", str(tmp_path / "s.json"), "--baseline-checkpoint-dir", "/x"],
        ["--output", str(tmp_path / "real.json")],  # a real-data run that is not the canonical file
        ["--output", str(other)],
        ["--output", str(alias)],
        ["--output", str(ROOT / "results" / ".." / "results" / b.CANONICAL_OUTPUT.name)],
        ["--synthetic"],  # abbreviations are not accepted
        ["--quick"], ["--seeds", "1"], ["--max-eval-windows", "8"], ["--no-verify"],  # no real-data smoke flags
    ):
        with pytest.raises(SystemExit) as exc:
            b.main(argv)
        assert exc.value.code == 2, argv


def test_canonical_host_guard(tmp_path):
    clean = {"git_sha": "0" * 40, "git_dirty": False}
    fresh = tmp_path / "new.json"
    if CUDA and torch.cuda.get_device_capability() >= (12, 0):
        b.require_canonical_host(torch.device("cuda"), clean, fresh)
        with pytest.raises(RuntimeError, match="clean"):
            b.require_canonical_host(torch.device("cuda"), {"git_sha": "0" * 40, "git_dirty": True}, fresh)
        with pytest.raises(RuntimeError, match="clean"):
            b.require_canonical_host(torch.device("cuda"), {"git_sha": None, "git_dirty": False}, fresh)
        fresh.write_text("{}")
        with pytest.raises(RuntimeError, match="scores test.bin once"):  # never a second canonical scoring
            b.require_canonical_host(torch.device("cuda"), clean, fresh)
    with pytest.raises(RuntimeError, match="Blackwell"):
        b.require_canonical_host(torch.device("cpu"), clean, tmp_path / "other.json")


def test_code_provenance_covers_the_whole_import_closure():
    modules = b.code_modules()
    assert set(b.SUMMARY_MODULES) <= set(modules) and b.DRIVER_MODULE in modules
    assert "research.quant_ptq.rotated_qat" in modules and "research.polar_opt.harness_tier2_lm" in modules
    assert modules == tuple(sorted(modules)) and len(b.code_sha256()) == 64
    env = {"torch": "2", "triton": "3", "cuda": "13", "driver": "1", "device": "x", "capability": [12, 0]}
    base = b.run_config_hash(True, b.SHAKEDOWN, "c" * 64, "a" * 40, env, {"train": "t"})
    assert base == b.run_config_hash(True, copy.deepcopy(b.SHAKEDOWN), "c" * 64, "a" * 40, dict(env), {"train": "t"})
    for other in (b.run_config_hash(False, b.SHAKEDOWN, "c" * 64, "a" * 40, env, {"train": "t"}),
                  b.run_config_hash(True, b.CANONICAL, "c" * 64, "a" * 40, env, {"train": "t"}),
                  b.run_config_hash(True, b.SHAKEDOWN, "d" * 64, "a" * 40, env, {"train": "t"}),
                  b.run_config_hash(True, b.SHAKEDOWN, "c" * 64, "b" * 40, env, {"train": "t"}),
                  b.run_config_hash(True, b.SHAKEDOWN, "c" * 64, "a" * 40, {**env, "driver": "2"}, {"train": "t"}),
                  b.run_config_hash(True, b.SHAKEDOWN, "c" * 64, "a" * 40, env, {"train": "u"})):
        assert other != base


def test_test_access_log_appends_and_returns_the_earlier_openings(tmp_path):
    log = tmp_path / "evidence" / "log.jsonl"
    first = b.record_test_access({"git_sha": "a" * 40}, "h1", log)
    assert first["prior_openings"] == [] and first["this_opening"]["git_sha"] == "a" * 40
    assert first["log"] == "artifacts/c6_01_run_evidence/test_access_log.jsonl"
    second = b.record_test_access({"git_sha": "b" * 40}, "h2", log)
    assert second["prior_openings"] == [first["this_opening"]] and second["this_opening"]["run_config_hash"] == "h2"
    assert [json.loads(line)["git_sha"][0] for line in log.read_text().splitlines()] == ["a", "b"]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    monkeypatch.setattr(b, "ROOT", root)
    monkeypatch.setattr(c503, "ROOT", root)  # the path primitives of the writer
    monkeypatch.setattr(b, "CANONICAL_OUTPUT", root / b.RESULT_REPO_PATH)
    return root


def _leftovers(directory):
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".tmp"))


def test_atomic_writer_round_trips_and_leaves_no_temp_file(sandbox, tmp_path):
    payload = {"a": [1, 2.5, None], "b": {"c": "x"}}
    assert b.write_json_atomic(b.CANONICAL_OUTPUT, payload, canonical_dest=True) == payload
    assert json.loads(b.CANONICAL_OUTPUT.read_text()) == payload and _leftovers(sandbox / "results") == []
    plain = tmp_path / "deep" / "dir" / "out.json"
    assert b.write_json_atomic(plain, payload) == payload and _leftovers(plain.parent) == []
    with pytest.raises(ValueError):
        b.write_json_atomic(plain, {"x": float("nan")})
    assert json.loads(plain.read_text()) == payload and _leftovers(plain.parent) == []  # the old file survives


def test_atomic_writer_rejects_symlinks_aliases_and_traversal(sandbox, tmp_path):
    payload = {"v": 1}
    outside = tmp_path / "outside.json"
    for bad in (outside, sandbox / "results" / "other.json", sandbox / "results" / ".." / b.RESULT_REPO_PATH):
        with pytest.raises(ValueError, match="canonical result goes to|parent traversal"):
            b.write_json_atomic(bad, payload, canonical_dest=True)
    with pytest.raises(ValueError, match="reserved result path outside"):
        b.write_json_atomic(tmp_path / "cycle6_c6_01_copy.json", payload)  # reserved name outside the checkout
    # A symlink at the canonical file: refused, and its target is not written.
    target = tmp_path / "target.json"
    target.write_text("old")
    b.CANONICAL_OUTPUT.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        b.write_json_atomic(b.CANONICAL_OUTPUT, payload, canonical_dest=True)
    with pytest.raises(ValueError, match="symlink"):
        b.verify_canonical_output_path(b.CANONICAL_OUTPUT)
    assert target.read_text() == "old" and _leftovers(sandbox / "results") == []
    b.CANONICAL_OUTPUT.unlink()
    # A symlinked results directory: refused as well.
    (sandbox / "results").rmdir()
    real = tmp_path / "elsewhere"
    real.mkdir()
    (sandbox / "results").symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|not a directory"):
        b.write_json_atomic(b.CANONICAL_OUTPUT, payload, canonical_dest=True)
    assert list(real.iterdir()) == []
    assert b.reserved(Path("cycle6_c6_01_x.json")) and not b.reserved(Path("c6_01_shakedown.json"))


def test_atomic_writer_keeps_the_old_file_when_the_replace_fails(sandbox, monkeypatch):
    b.write_json_atomic(b.CANONICAL_OUTPUT, {"v": 1}, canonical_dest=True)

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(b.os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        b.write_json_atomic(b.CANONICAL_OUTPUT, {"v": 2}, canonical_dest=True)
    assert json.loads(b.CANONICAL_OUTPUT.read_text()) == {"v": 1} and _leftovers(sandbox / "results") == []


# ------------------------------------------------------------------ the synthetic shakedown and validator probes
@pytest.fixture(scope="module")
def shakedown(tmp_path_factory):
    """One --synthetic-shakedown run: generated shards, every stage, the freeze, test scoring and validation."""
    if not FP4:
        pytest.skip("needs native NVFP4 scaled_mm")
    out = tmp_path_factory.mktemp("c6_01") / "shakedown.json"
    canonical_before = b.CANONICAL_OUTPUT.read_bytes() if b.CANONICAL_OUTPUT.exists() else None

    def forbidden(*args, **kwargs):
        raise AssertionError("the shakedown must not look up the real data or the preserved checkpoints")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(q, "default_data_dir", forbidden)
        patch.setattr(b, "default_baseline_dir", forbidden)
        patch.setattr(b, "record_test_access", forbidden)  # the access log is for the real test.bin only
        payload = b.main(["--synthetic-shakedown", "--silent", "--output", str(out)])
    assert json.loads(out.read_text()) == payload
    assert (b.CANONICAL_OUTPUT.read_bytes() if b.CANONICAL_OUTPUT.exists() else None) == canonical_before
    return payload


NONCANONICAL = {"certified_canonical": False, "shakedown": True}
CERTIFIED = {"certified_canonical": True, "shakedown": False}


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


def _unit(p, part, seed, arm):
    return next(u for u in p["units"] if (u["part"], u["seed"], u["arm"]) == (part, seed, arm))


def _grid_row(u, method, mode):
    return next(r for r in u["grid"] if (r["method"], r["mode"]) == (method, mode))


@needs_fp4
def test_shakedown_is_valid_complete_and_synthetic(shakedown):
    p = shakedown
    assert b.validate_results(p) == b.validate_results(p, canonical=False) == NONCANONICAL
    assert p["schema_version"] == "cycle6_c6_01.v1" and p["shakedown"] and not p["canonical"]
    data = p["data"]
    assert data["synthetic"] and data["source"]["name"] == "synthetic-markov" and data["test_access"] is None
    assert not {v["sha256"] for v in data["splits"].values()} & set(q.EXPECTED_SHA256.values())
    assert "shakedown.json.work" in data["directory"] and "real_text" not in data["directory"]
    assert p["c5_03_reference"] is None and p["analysis"]["c5_03_reproduction"] is None
    # Split hygiene: one freeze, test.bin mapped and scored only after it.
    assert p["meta"]["splits_opened"] == ["train", "val", "test"] and p["splits_opened_before_freeze"] == ["train", "val"]
    ledger = p["scoring_ledger"]
    assert ledger["test_scored_only_after_freeze"] and ledger["frozen_after_val_scorings"] == ledger["val_scorings"]
    assert len(p["selections_sha256"]) == 64
    # 3 seeds x (6 arms of Part A/B + 3 arms of Part C).
    assert len(p["units"]) == 27 and [u["arm"] for u in p["units"][:3]] == [b.MUON] * 3
    probe = _unit(p, "ab", 1, b.MUON)
    assert [r["method"] for r in probe["grid"] if r["mode"] == b.DYNAMIC] == list(b.PROBE_METHODS)
    assert list(probe["activation_statistics"]) == list(rq.FAMILIES)
    sel = p["selections"]
    assert sel["qat_wide_family"] in b.WIDE_QAT_FAMILIES and sel["transfer_arm"] in b.TRANSFER_CANDIDATES
    for seed in (1, 2, 3):
        wide = _unit(p, "ab", seed, "qat_wide_cooldown")["training"]
        assert wide["qat"] == {"family": sel["qat_wide_family"], "schedule": "cooldown", "loss": "qat",
                               "bf16_weight": None, "fake_quant_steps": 10}
        assert wide["branch"]["from_arm"] == b.MUON and wide["first_step"] == 30 and wide["steps"] == 40
        assert _unit(p, "ab", seed, "qat_identity_full")["training"]["qat"]["fake_quant_steps"] == 40
        dual = _unit(p, "ab", seed, "qat_wide_dual_cooldown")["training"]["qat"]
        assert dual["loss"] == "dual" and dual["bf16_weight"] == 0.5
        base, cont, qat = (_unit(p, "c", seed, a)["training"] for a in b.ARMS_C)
        assert base["steps"] == 30 and cont["steps"] == qat["steps"] == 38 and cont["first_step"] == 30
        assert cont["branch"]["snapshot_sha256"] == qat["branch"]["snapshot_sha256"] == base["checkpoint_sha256"]
        assert cont["qat"] is None and qat["qat"]["family"] == sel["transfer_family"]
        assert cont["tokens"] == qat["tokens"] > base["tokens"]  # the continuation is the token-matched control
        assert len({base["checkpoint_sha256"], cont["checkpoint_sha256"], qat["checkpoint_sha256"]}) == 3
    # Hardware evidence: 10 native FP4 GEMMs per forward in every hardware row, no fallback, exact decode, parity.
    hw = p["analysis"]["hardware"]
    assert hw["fp4_gemms_per_forward"] == [10] and hw["cupti_fp4_gemm_kernels"] == [10]
    assert hw["cupti_nvfp4_quant_kernels"] == [10] and hw["bf16_block_linears_in_hw_rows"] == [0]
    assert hw["fp4_gemms_per_forward_bf16"] == [0] and hw["fp4_gemms_per_forward_emulated"] == [0]
    assert hw["decode_exact_matrices"] == hw["decode_checked_matrices"] == 10 * hw["decode_checked_rows"]
    assert hw["wide_hw_rows"] > 0
    parity = p["analysis"]["hardware_vs_emulation"]
    assert parity["passed"] and len(parity["rows"]) == 54 and parity["max_abs_nll_difference"] <= 1e-3
    assert {r["method"] for r in parity["rows"] if r["arm"] == b.MUON} == {"kron_hadamard_full_gptq"}  # a wide path
    assert p["kernel_parity"]["passed"] and p["cross_mix_parity"]["cases"] == 48
    # Statistics: every contrast has both endpoints on both splits, with df = 2 statistics and a verdict.
    contrasts = p["analysis"]["contrasts"]
    assert [c["name"] for c in contrasts] == [name for _, name, _, _ in b.contrast_specs(sel)]
    for c in contrasts:
        for split in ("val", "test"):
            for endpoint in ("nll", "delta_nll"):
                s = c[split][endpoint]
                assert s["df"] == 2 and 0 <= s["p_two_sided"] <= s["p_holm"] <= 1 and s["p_sign_flip"] >= 0.25
                assert s["verdict"] == b.verdict(s)
    bf16 = next(c for c in contrasts if c["name"] == "qat_bf16_vs_baseline_bf16")
    assert all(d == 0 for d in bf16["test"]["delta_nll"]["diffs"])  # a BF16 side has no quantization delta
    assert set(p["analysis"]["throughput"]) == {f"ab/{a}" for a in b.QAT_ARMS} | {"c/qat_cooldown"}
    assert all(t["step_time_ratio"]["mean"] > 0 for t in p["analysis"]["throughput"].values())
    notes = "\n".join(p["caveats"])
    assert "SHAKEDOWN: synthetic Markov shards" in notes and "+26.7%" in notes and "40 steps" in notes
    for forbidden in ("FineWeb", "C6-01 test-access log", "640 steps", "320 steps"):
        assert forbidden not in notes
    with pytest.raises(ValueError, match="not labeled canonical"):
        b.validate_results(p, canonical=True)


@needs_fp4
def test_validator_rejects_tampered_labels_hashes_and_provenance(shakedown, tmp_path):
    _rejects(shakedown, lambda p: p.update(canonical=True), "either the canonical run or a synthetic shakedown")
    _rejects(shakedown, lambda p: p.update(canonical=True, shakedown=False), "differs from the canonical C6-01 protocol")
    _rejects(shakedown, lambda p: p.update(shakedown=False), "either the canonical run or a synthetic shakedown")
    _rejects(shakedown, lambda p: p.update(schema_version="cycle6_c6_01.v0"), "schema or task differs")
    _rejects(shakedown, lambda p: p.update(extra=1), "top-level keys")
    _rejects(shakedown, lambda p: _set(p, ("protocol", "seeds"), [1, 2, 4]), "protocol differs")
    _rejects(shakedown, lambda p: _set(p, ("protocol", "part_c", "cooldown_steps"), 9), "protocol differs")
    _rejects(shakedown, lambda p: _set(p, ("meta", "code_sha256"), "0" * 64), "code_sha256 differs")
    _rejects(shakedown, lambda p: _set(p, ("meta", "code_modules"), p["meta"]["code_modules"][:-1]), "code_modules")
    _rejects(shakedown, lambda p: p.update(run_config_hash="0" * 16), "run_config_hash differs")
    _rejects(shakedown, lambda p: _set(p, ("meta", "environment", "driver"), "0.0"), "run_config_hash differs")
    _rejects(shakedown, lambda p: _set(p, ("meta", "numerics", "allow_tf32"), True), "full-precision numerics")
    _rejects(shakedown, lambda p: _set(p, ("meta", "splits_opened"), ["train", "test", "val"]), "splits_opened")
    _rejects(shakedown, lambda p: _set(p, ("meta", "timestamp_end"), p["meta"]["timestamp_start"]), "timestamps")
    _rejects(shakedown, lambda p: _set(p, ("meta", "total_seconds"), float("nan")), "JSON compliant")
    # Data: a shakedown that claims a real split, a changed digest, or verified real data.
    real = q.EXPECTED_SHA256["test"]
    _rejects(shakedown, lambda p: _set(p, ("data", "splits", "test", "sha256"), real), "synthetic shards only")
    _rejects(shakedown, lambda p: _set(p, ("data", "splits", "val", "sha256"), "0" * 64), "run_config_hash differs")
    _rejects(shakedown, lambda p: _set(p, ("data", "synthetic"), False), "synthetic shards only")
    _rejects(shakedown, lambda p: _set(p, ("data", "source", "name"), "fineweb-edu"), "synthetic shards only")
    _rejects(shakedown, lambda p: _set(p, ("data", "splits", "val", "sha256"), "xyz"), "64-hex")
    # The recorded command: another script, a real-data flag, a reserved or aliased output path.
    alias = tmp_path / "alias.json"
    alias.symlink_to(b.CANONICAL_OUTPUT)

    def command(*argv):
        def edit(p):
            p["meta"]["argv"] = list(argv)
            p["meta"]["command"] = [*p["meta"]["command"][:2], *argv]
        return edit

    _rejects(shakedown, command("--synthetic-shakedown", "--output", str(b.CANONICAL_OUTPUT)), "not a valid runnable")
    _rejects(shakedown, command("--synthetic-shakedown", "--output", str(alias)), "not a valid runnable")
    _rejects(shakedown, command("--synthetic-shakedown", "--output", "x.json", "--data-dir", "/d"), "not a valid runnable")
    _rejects(shakedown, command(), "does not select the recorded protocol and mode")  # the canonical invocation
    _rejects(shakedown, lambda p: _set(p, ("meta", "command", 1), "research/quant_ptq/other.py"), "not an invocation")
    _rejects(shakedown, lambda p: _set(p, ("meta", "argv"), ["--silent"]), "arguments of meta.command")
    _rejects(shakedown, lambda p: _set(p, ("meta", "git_sha"), "0" * 40), "not a commit object")
    _rejects(shakedown, lambda p: _set(p, ("kernel_parity", "passed"), False), "kernel parity")
    _rejects(shakedown, lambda p: _set(p, ("cross_mix_parity", "cases"), 47), "cross-mix kernel parity")
    _rejects(shakedown, lambda p: _set(p, ("cross_mix_parity", "bitwise_failures"), [[1, 128, "x", False]]), "cross-mix")
    _rejects(shakedown, lambda p: p.pop("caveats"), "top-level keys")
    _rejects(shakedown, lambda p: p["units"].pop(), "units differ from the study order")
    _rejects(shakedown, lambda p: p["units"][0].pop("grid"), "unit keys")


@needs_fp4
def test_validator_rejects_missing_hardware_evidence(shakedown):
    def row(method, mode, part="ab", seed=1, arm=b.MUON):
        return lambda p: _grid_row(_unit(p, part, seed, arm), method, mode)

    hw = row("kron_hadamard_full_gptq", b.DYNAMIC)
    _rejects(shakedown, lambda p: hw(p).update(fp4_gemms_per_forward=9), "native FP4 GEMM")
    _rejects(shakedown, lambda p: hw(p).update(bf16_block_linears=1, nvfp4_modules=9), "native FP4 GEMM")
    _rejects(shakedown, lambda p: hw(p).update(cross_mix_modules=0), "native FP4 GEMM")  # a wide row without its mix
    _rejects(shakedown, lambda p: hw(p).update(cupti=None), "CUPTI")
    _rejects(shakedown, lambda p: hw(p)["cupti"].update(fp4_gemm_kernels=9), "CUPTI")
    _rejects(shakedown, lambda p: hw(p)["cupti"].update(nvfp4_amax_kernels=0), "CUPTI")  # dynamic needs the amax pass
    _rejects(shakedown, lambda p: hw(p)["cupti"].pop("nvfp4_quant_kernels"), "CUPTI")
    plain = row(b.C5_HADAMARD, "w4a4_hw_static")
    _rejects(shakedown, lambda p: plain(p)["cupti"].update(nvfp4_amax_kernels=10), "CUPTI")  # static: no amax pass
    _rejects(shakedown, lambda p: plain(p)["cupti"].update(
        non_fp4_gemm_kernels=plain(p)["cupti"]["non_fp4_gemm_kernels"] + 1), "CUPTI")  # one BF16 GEMM too many
    _rejects(shakedown, lambda p: plain(p).update(static_global_scale_range=None), "static scale range")
    _rejects(shakedown, lambda p: _unit(p, "ab", 1, b.MUON).update(bf16_fp4_gemms_per_forward=1), "BF16 model must run no")
    _rejects(shakedown, lambda p: _unit(p, "ab", 1, b.MUON)["bf16_cupti"].update(non_fp4_gemm_kernels=0),
             "BF16 model must run no")  # a blind profiler is not evidence
    w4a16 = row("kron_hadamard_full_gptq", "w4a16_fakequant")
    _rejects(shakedown, lambda p: w4a16(p).update(hw_weights_decode_exact_matrices=9), "decode exactly")
    _rejects(shakedown, lambda p: w4a16(p).update(effective_weights_sha256_test="0" * 64), "decode exactly")
    emulated = row("kron_hadamard_full_gptq", b.EMULATED_MODE)
    _rejects(shakedown, lambda p: emulated(p).update(fp4_gemms_per_forward=10), "emulation must run no FP4 GEMM")
    _rejects(shakedown, lambda p: _unit(p, "ab", 1, b.MUON)["grid"].remove(emulated(p)), "grid rows differ")
    _rejects(shakedown, lambda p: _unit(p, "c", 2, b.QAT_C).update(emulated_method=b.C5_BEST), "methods differ")

    def break_parity(p):  # a self-consistent result whose hardware NLL is 0.01 nats off its FP32 emulation
        u = _unit(p, "ab", 1, b.MUON)
        r, base = emulated(p), u["baseline_test"]
        r["test_nll"] += 0.01
        r["test_ppl"] = math.exp(r["test_nll"])
        r["test_delta_nll"] = r["test_nll"] - base["nll"]
        r["test_relative_delta_ppl_pct"] = 100.0 * (r["test_ppl"] / base["ppl"] - 1.0)
        p["analysis"] = json.loads(json.dumps(b.analyze(p["units"], p["selections"], p["protocol"], None)))

    _rejects(shakedown, break_parity, "must match the FP32 emulation")


@needs_fp4
def test_validator_rejects_tampered_rows_selections_statistics_and_caveats(shakedown):
    sel = shakedown["selections"]
    dyn = lambda p: _grid_row(_unit(p, "ab", 1, b.MUON), b.C5_HADAMARD, b.DYNAMIC)  # noqa: E731
    _rejects(shakedown, lambda p: dyn(p).update(val_delta_nll=dyn(p)["val_delta_nll"] - 0.01), "val_delta_nll")
    _rejects(shakedown, lambda p: dyn(p).update(test_ppl=dyn(p)["test_ppl"] * 0.99), "test_ppl")
    _rejects(shakedown, lambda p: dyn(p).update(test_tokens=1), "tokens or seconds")
    _rejects(shakedown, lambda p: dyn(p).update(rotation="identity"), "method fields")
    _rejects(shakedown, lambda p: dyn(p).update(awq_alpha=0.25), "awq_alpha")
    _rejects(shakedown, lambda p: _unit(p, "ab", 1, b.MUON)["baseline_val"].update(tokens=1), "coverage differs")

    def better(method):  # make one val NLL the best: a consistent row, so only the selection rules can object
        def edit(p):
            for seed in (1, 2, 3):
                u = _unit(p, "ab", seed, b.MUON)
                r, base = _grid_row(u, method, b.DYNAMIC), u["baseline_val"]
                r["val_nll"] = base["nll"] - 1.0
                r["val_ppl"] = math.exp(r["val_nll"])
                r["val_delta_nll"] = r["val_nll"] - base["nll"]
                r["val_relative_delta_ppl_pct"] = 100.0 * (r["val_ppl"] / base["ppl"] - 1.0)
        return edit

    other = next(f for f in b.WIDE_QAT_FAMILIES if f != sel["qat_wide_family"])
    _rejects(shakedown, better(rq.method_name(other, "gptq")), "Part A selections differ")
    _rejects(shakedown, lambda p: _set(p, ("selections", "qat_wide_family"), other), "differs from the arm definition")
    _rejects(shakedown, lambda p: _set(p, ("selections", "best_ptq_method"), "rtn"), "Part A selections differ")
    _rejects(shakedown, lambda p: _set(p, ("selections", "best_wide_ptq_method"), "rtn"), "selections keys or values")
    _rejects(shakedown, lambda p: _set(p, ("selections", "arm_best_method", "c", b.QAT_C), b.C5_HADAMARD)
             if sel["arm_best_method"]["c"][b.QAT_C] != b.C5_HADAMARD else p.update(selections_sha256="0" * 64),
             "Part C best methods differ|selections_sha256 differs")
    _rejects(shakedown, lambda p: p.update(selections_sha256="0" * 64), "selections_sha256 differs")
    # AWQ: the recorded alpha must be the val.bin minimum, and the rows must be that candidate.
    awq = lambda p: _unit(p, "ab", 1, b.MUON)["awq_selection"][0]  # noqa: E731
    _rejects(shakedown, lambda p: awq(p).update(selected_alpha=next(a for a in awq(p)["alphas"]
                                                                    if a != awq(p)["selected_alpha"])),
             "not the val.bin minimum")
    _rejects(shakedown, lambda p: awq(p).update(selected_on="test.bin"), "not the val.bin minimum")
    _rejects(shakedown, lambda p: _unit(p, "ab", 1, b.MUON)["awq_selection"].pop(), "one AWQ selection per AWQ method")
    # Split hygiene records.
    _rejects(shakedown, lambda p: p["scoring_ledger"].update(val_scorings=p["scoring_ledger"]["val_scorings"] + 1),
             "scoring ledger differs")
    _rejects(shakedown, lambda p: p["scoring_ledger"].update(test_scored_only_after_freeze=False), "scoring ledger differs")
    _rejects(shakedown, lambda p: p["scoring_ledger"].update(frozen_after_val_scorings=3), "scoring ledger differs")
    _rejects(shakedown, lambda p: p.update(splits_opened_before_freeze=["train", "val", "test"]), "before the freeze")
    _rejects(shakedown, lambda p: p["calibration"]["64"].update(split="val"), "calibration must be the train.bin draw")
    _rejects(shakedown, lambda p: p["calibration"]["64"]["starts"].reverse(), "calibration starts differ")
    # Training records.
    tr = lambda p, arm="qat_wide_cooldown", part="ab": _unit(p, part, 1, arm)["training"]  # noqa: E731
    _rejects(shakedown, lambda p: tr(p)["qat"].update(family="identity"), "QAT record differs")
    _rejects(shakedown, lambda p: tr(p, "qat_wide_dual_cooldown")["qat"].update(bf16_weight=0.9), "QAT record differs")
    _rejects(shakedown, lambda p: tr(p).update(steps=41), "training record differs from the protocol")
    _rejects(shakedown, lambda p: tr(p)["step_seconds"].pop(), "one positive wall time per optimizer step")
    _rejects(shakedown, lambda p: tr(p).update(checkpoint_sha256="xyz"), "identity or digests")
    _rejects(shakedown, lambda p: tr(p)["branch"].update(from_step=0), "must branch from")
    _rejects(shakedown, lambda p: tr(p, b.QAT_C, "c")["branch"].update(snapshot_sha256="0" * 64), "must branch from")
    _rejects(shakedown, lambda p: tr(p, b.CONTINUE, "c").update(tokens=1), "training record differs from the protocol")
    _rejects(shakedown, lambda p: tr(p, b.MUON).update(qat={"family": "identity"}), "no QAT record")
    # Statistics, summary and caveats are recomputed.
    first = lambda p: p["analysis"]["contrasts"][0]["test"]["nll"]  # noqa: E731
    _rejects(shakedown, lambda p: first(p).update(p_two_sided=first(p)["p_two_sided"] * 0.5), "analysis")
    _rejects(shakedown, lambda p: first(p).update(verdict="helps (significant)"), "analysis")
    _rejects(shakedown, lambda p: p["analysis"]["contrasts"].pop(), "analysis")
    _rejects(shakedown, lambda p: p["analysis"]["throughput"][f"c/{b.QAT_C}"]["step_time_ratio"].update(mean=1.0), "analysis")
    _rejects(shakedown, lambda p: p["analysis"]["hardware"].update(wide_hw_rows=0), "analysis")
    _rejects(shakedown, lambda p: _set(p, ("summary", "selections", "transfer_arm"), "x"), "summary")
    _rejects(shakedown, lambda p: p["summary"]["part_c_arms"][b.QAT_C].update(best_w4a4_dynamic_test_nll=1.0), "summary")
    _rejects(shakedown, lambda p: p["caveats"].pop(), "caveats differ")
    _rejects(shakedown, lambda p: p["caveats"].__setitem__(2, p["caveats"][2].replace("token-matched control",
                                                                                   "equal-budget arm")), "caveats differ")
    _rejects(shakedown, lambda p: p["caveats"].append("QAT wins."), "caveats differ")


@needs_fp4
def test_shakedown_checkpoints_are_the_recorded_tensors(shakedown):
    """The retained state dicts load to the recorded digests: Phase 2 scored the Phase 1 checkpoints."""
    work = Path(shakedown["meta"]["argv"][-1] + ".work") / "checkpoints"
    for part, arm in (("ab", b.MUON), ("ab", "qat_wide_dual_cooldown"), ("c", b.BASELINE), ("c", b.QAT_C)):
        tr = _unit(shakedown, part, 3, arm)["training"]
        path = work / tr["checkpoint_file"]
        assert c503.file_sha256(path) == tr["checkpoint_file_sha256"]
        assert c502.checkpoint_file_matches_sha256(path, tr["checkpoint_sha256"])
    with pytest.raises(RuntimeError, match="SHA-256 differs"):
        tr = dict(_unit(shakedown, "c", 3, b.BASELINE)["training"], checkpoint_sha256="0" * 64)
        c503.load_checkpoint(b.PRESETS["shakedown"], torch.device("cuda"), work / tr["checkpoint_file"], tr)


# ------------------------------------------------------------------ canonical-only branches, on constructed inputs
def _canonical_data(**changes):
    splits = {k: {"tokens": c502.CANONICAL_DATA["tokens"][k], "sha256": v} for k, v in q.EXPECTED_SHA256.items()}
    access = {"log": b.TEST_ACCESS_LOG_REPO_PATH, "prior_openings": [],
              "this_opening": {"git_sha": "a" * 40, "run_config_hash": "h" * 16, "timestamp": "2026-11-27T12:00:00-0700"}}
    data = {"source": {"name": "fineweb-edu"}, "encoding": "gpt2", "splits": splits, "directory": "/d",
            "sha256_verified": True, "synthetic": False, "roles": {}, "test_access": access}
    data.update(changes)
    return {"data": data, "meta": {"git_sha": "a" * 40}, "run_config_hash": "h" * 16}


def test_canonical_data_record_needs_the_pinned_shards_and_the_test_access_log():
    b._validate_data(_canonical_data(), True)
    wrong = {k: {"tokens": 1, "sha256": v} for k, v in q.EXPECTED_SHA256.items()}
    other = {**_canonical_data()["data"]["splits"], "test": {"tokens": 2_000_081, "sha256": "0" * 64}}
    access = _canonical_data()["data"]["test_access"]
    for changes, match in (({"sha256_verified": False}, "pinned FineWeb-Edu"), ({"synthetic": True}, "pinned FineWeb-Edu"),
                           ({"splits": other}, "pinned FineWeb-Edu"), ({"splits": wrong}, "token counts"),
                           ({"test_access": None}, "test_access"),
                           ({"test_access": {**access, "log": "elsewhere.jsonl"}}, "test_access"),
                           ({"test_access": {**access, "prior_openings": "none"}}, "test_access"),
                           ({"test_access": {**access, "this_opening": {**access["this_opening"], "git_sha": "b" * 40}}},
                            "test_access"),
                           ({"test_access": {**access, "this_opening": {**access["this_opening"],
                                                                        "run_config_hash": "x"}}}, "test_access")):
        with pytest.raises(ValueError, match=match):
            b._validate_data(_canonical_data(**changes), True)
    with pytest.raises(ValueError, match="synthetic shards only"):  # real shards are never a shakedown
        b._validate_data(_canonical_data(), False)


def test_canonical_baseline_record_must_be_the_pinned_preserved_checkpoint():
    payload = {"protocol": b.CANONICAL, "selections": {}, "units": [],
               "data": {"splits": {"train": {"tokens": c502.CANONICAL_DATA["tokens"]["train"]}}}}
    for seed in (1, 2, 3):
        unit = {"part": "c", "arm": b.BASELINE, "seed": seed, "training": b.preserved_baseline_record(seed)}
        b._validate_training(unit, payload, "unit")
        for key, value in (("checkpoint_sha256", "0" * 64), ("checkpoint_file_sha256", "0" * 64),
                           ("checkpoint_file", "other.pt"), ("source", "trained_in_run"), ("seed", seed % 3 + 1)):
            bad = {**unit, "training": {**unit["training"], key: value}}
            with pytest.raises(ValueError, match="pinned digests|identity or digests"):
                b._validate_training(bad, payload, "unit")
        # The digests of another seed's checkpoint are valid hex, but they are not this seed's pin.
        swapped = {**unit, "training": {**b.preserved_baseline_record(seed % 3 + 1), "seed": seed}}
        with pytest.raises(ValueError, match="pinned digests"):
            b._validate_training(swapped, payload, "unit")


@needs_fp4
def test_c5_03_reproduction_and_canonical_caveats_on_a_relabeled_copy(shakedown):
    """The analysis and caveat branches that only a canonical run takes, driven with the shakedown rows."""
    p = copy.deepcopy(shakedown)
    modes = ("w4a16_fakequant", b.DYNAMIC, "w4a4_hw_static")
    reference = {"seeds": {}}
    for seed in (1, 2, 3):
        u = _unit(p, "c", seed, b.BASELINE)
        reference["seeds"][str(seed)] = {
            "bf16": {split: u[f"baseline_{split}"]["nll"] + 0.001 for split in ("val", "test")},
            "methods": {m: {mode: {split: _grid_row(u, m, mode)[f"{split}_nll"] for split in ("val", "test")}
                            for mode in modes} for m in (b.C5_HADAMARD, b.C5_BEST)}}
    analysis = b.analyze(p["units"], p["selections"], p["protocol"], reference)
    rep = analysis["c5_03_reproduction"]
    assert len(rep["rows"]) == 3 * 2 * (1 + 2 * 3) and rep["max_abs_nll_difference"] == pytest.approx(0.001, abs=1e-12)
    assert {r["row"] for r in rep["rows"]} == {"bf16"} | {f"{m}/{mode}" for m in (b.C5_HADAMARD, b.C5_BEST) for mode in modes}
    assert all(r["abs_diff"] == 0 for r in rep["rows"] if r["row"] != "bf16")
    rest = {k: v for k, v in analysis.items() if k != "c5_03_reproduction"}
    assert rest == {k: v for k, v in b.analyze(p["units"], p["selections"], p["protocol"], None).items()
                    if k != "c5_03_reproduction"}
    p.update(shakedown=False, canonical=True, analysis=analysis)
    p["protocol"] = copy.deepcopy(b.CANONICAL)
    opening = {"git_sha": "c" * 40, "run_config_hash": "h", "timestamp": "2026-11-27T10:00:00-0700"}
    p["data"]["test_access"] = {"log": b.TEST_ACCESS_LOG_REPO_PATH, "prior_openings": [], "this_opening": opening}
    notes = "\n".join(b.caveats(p))
    assert "SHAKEDOWN" not in notes and "not an unseen holdout at study level" in notes
    assert "had 0 earlier openings of test.bin by this driver when this run mapped it." in notes
    assert "max |NLL difference| to the published C5-03 rows: 1.00e-03 nats" in notes
    assert "640 steps (20,971,520 tokens)" in notes and "add 96 steps (3,145,728 tokens, +15.0%)" in notes
    assert "320 steps (10,485,760 tokens)" in notes and "branch from the Muon run at step 272" in notes
    assert "the preserved C5-03 checkpoints hold weights only." in notes
    p["data"]["test_access"]["prior_openings"] = [opening]
    assert "had 1 earlier opening of test.bin by this driver when this run mapped it: ccccccc at 2026-11-27T10:00:00-0700." \
        in "\n".join(b.caveats(p))


# ------------------------------------------------------------------ the committed canonical artifact
@pytest.fixture(scope="module")
def canonical():
    if not b.CANONICAL_OUTPUT.exists():
        pytest.skip("canonical result not committed yet")
    return json.loads(b.CANONICAL_OUTPUT.read_text())


def test_committed_canonical_artifact(canonical):
    p = canonical
    assert CANONICAL_SHA256 is not None, "pin the SHA-256 of the committed canonical result"
    assert hashlib.sha256(b.CANONICAL_OUTPUT.read_bytes()).hexdigest() == CANONICAL_SHA256
    tracked = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{b.RESULT_REPO_PATH}"], capture_output=True)
    assert tracked.returncode == 0 and hashlib.sha256(tracked.stdout).hexdigest() == CANONICAL_SHA256
    assert b.validate_results(p, canonical=True) == b.validate_results(p) == CERTIFIED
    assert p["canonical"] and not p["shakedown"] and p["protocol"] == b.CANONICAL
    assert p["meta"]["git_dirty"] is False and p["meta"]["environment"]["capability"] >= [12, 0]
    assert p["meta"]["argv"] == [] or set(a.split("=")[0] for a in p["meta"]["argv"] if a.startswith("--")) <= {
        "--data-dir", "--baseline-checkpoint-dir", "--checkpoint-dir", "--silent"}
    data = p["data"]
    assert {k: v["sha256"] for k, v in data["splits"].items()} == q.EXPECTED_SHA256 and data["sha256_verified"]
    assert data["test_access"]["this_opening"]["git_sha"] == p["meta"]["git_sha"]
    assert p["kernel_parity"]["cases"] == 279 and p["cross_mix_parity"]["cases"] == 48
    assert len(p["units"]) == 27
    for u in p["units"]:
        assert (u["baseline_val"]["tokens"], u["baseline_test"]["tokens"]) == (c502.VAL_TOKENS, c502.TEST_TOKENS)
    for seed in (1, 2, 3):
        base = _unit(p, "c", seed, b.BASELINE)["training"]
        assert base == b.preserved_baseline_record(seed)
        for arm in (b.CONTINUE, b.QAT_C):
            tr = _unit(p, "c", seed, arm)["training"]
            assert tr["steps"] == 736 and tr["first_step"] == 640 and tr["tokens"] == 736 * 32768
            assert tr["branch"]["snapshot_sha256"] == b.PINNED_BASELINES[seed]["checkpoint_sha256"]
    hw = p["analysis"]["hardware"]
    assert hw["fp4_gemms_per_forward"] == [60] and hw["cupti_fp4_gemm_kernels"] == [60]
    assert hw["cupti_nvfp4_quant_kernels"] == [60] and hw["bf16_block_linears_in_hw_rows"] == [0]
    assert hw["decode_exact_matrices"] == hw["decode_checked_matrices"] == 60 * hw["decode_checked_rows"]
    parity = p["analysis"]["hardware_vs_emulation"]
    assert parity["passed"] and len(parity["rows"]) == 54 and parity["max_abs_nll_difference"] <= 1e-3
    assert p["scoring_ledger"]["test_scored_only_after_freeze"] and p["meta"]["splits_opened"] == ["train", "val", "test"]
    assert p["c5_03_reference"] == b.c5_03_reference() and p["analysis"]["c5_03_reproduction"] is not None
    notes = "\n".join(p["caveats"])
    assert "Part C token budgets differ" in notes and "+15.0%" in notes and "not an unseen holdout" in notes
    assert "SHAKEDOWN" not in notes


def test_canonical_validator_rejects_tampering(canonical, tmp_path):
    def rejects(edit, match):
        _rejects(canonical, edit, match, canonical=True)

    rejects(lambda p: _unit(p, "c", 1, b.BASELINE)["training"].update(checkpoint_sha256="0" * 64), "pinned digests")
    rejects(lambda p: _unit(p, "c", 1, b.BASELINE)["training"].update(checkpoint_file="other.pt"), "pinned digests")
    rejects(lambda p: _set(p, ("c5_03_reference", "result_sha256"), "0" * 64), "c5_03_reference differs")
    rejects(lambda p: _set(p, ("c5_03_reference", "seeds", "1", "bf16", "test"), 1.0), "c5_03_reference differs")
    rejects(lambda p: _set(p, ("data", "splits", "test", "sha256"), "0" * 64), "pinned FineWeb-Edu hashes")
    rejects(lambda p: _set(p, ("data", "sha256_verified"), False), "pinned FineWeb-Edu hashes")
    rejects(lambda p: _set(p, ("data", "test_access"), None), "test_access")
    rejects(lambda p: _set(p, ("data", "test_access", "this_opening", "git_sha"), "0" * 40), "test_access")
    rejects(lambda p: _set(p, ("meta", "git_dirty"), True), "clean tree")
    rejects(lambda p: _set(p, ("meta", "environment", "capability"), [10, 0]), "sm_120|run_config_hash")
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD~0"], capture_output=True, text=True).stdout.strip()
    first = subprocess.run(["git", "-C", str(ROOT), "rev-list", "--max-parents=0", "HEAD"], capture_output=True,
                           text=True).stdout.split()[0]
    assert head != first
    rejects(lambda p: _set(p, ("meta", "git_sha"), first), "import closure|code_sha256|cannot read")
    rejects(lambda p: p.update(canonical=False, shakedown=True), "differs from the shakedown protocol")
    alias = tmp_path / "alias.json"
    alias.symlink_to(b.CANONICAL_OUTPUT)

    def output(path):
        def edit(p):
            p["meta"]["argv"] = ["--output", str(path)]
            p["meta"]["command"] = [*p["meta"]["command"][:2], "--output", str(path)]
        return edit

    rejects(output(alias), "not a valid runnable")
    rejects(output(tmp_path / b.CANONICAL_OUTPUT.name), "not a valid runnable")
    rejects(lambda p: p["caveats"].pop(), "caveats differ")
    rejects(lambda p: _set(p, ("protocol", "part_c", "cooldown_steps"), 97), "canonical C6-01 protocol")
    with pytest.raises(ValueError, match="cannot be validated with the noncanonical checks"):
        b.validate_results(canonical, canonical=False)
