"""Checks for Cycle 6 C6-01: wide rotations, rotated-space NVFP4 W4A4 QAT and the C6-01 benchmark driver.

Math tests (CPU): orthogonality and structure of every rotation family, the straight-through
gradients of ``RotatedFakeQuantLinear`` and the dual loss step. Hardware tests (sm_100+): the cross-mix
+ fused-kernel path against its eager reference and its FP32 emulation, and the CUPTI evidence.
Driver tests: the CLI guard, the atomic writer, one ``--synthetic-shakedown`` run (generated Markov
shards; it cannot read real text, schema ``cycle6_c6_01.v2``) with validator tampering probes, and the
committed canonical artifact (the preserved ``cycle6_c6_01.v1`` result, byte-pinned). The probes include
every mutation that the GPT 6.2 Astra audit of C6-01 found certified: split roles, the test-opening time,
checkpoint digests, sibling branch hashes, fingerprints, the training curve, summary extrema, emulation
counters, the AWQ winner and the C5-03 comparator.

No test here scores the real val.bin or test.bin: the driver has no real-data mode besides the
canonical run, and the shakedown fixture fails if the real data directory is even looked up.
"""

import copy
import hashlib
import json
import math
import shutil
import subprocess
from datetime import timedelta
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
    assert b.COOLDOWN_ARMS == ("qat_identity_cooldown", "qat_b16_cooldown", "qat_wide_cooldown", "qat_wide_dual_cooldown")
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


def test_test_access_log_appends_the_freeze_digest_and_returns_the_earlier_openings(tmp_path):
    log = tmp_path / "evidence" / "log.jsonl"
    first = b.record_test_access({"git_sha": "a" * 40}, "h1", "1" * 64, log)
    assert first["prior_openings"] == [] and first["this_opening"]["git_sha"] == "a" * 40
    assert first["log"] == "artifacts/c6_01_run_evidence/test_access_log.jsonl"
    assert set(first["this_opening"]) == b.OPENING_KEYS[b.SCHEMA_VERSION] == {
        "git_sha", "run_config_hash", "timestamp", "selections_sha256"}
    assert first["this_opening"]["selections_sha256"] == "1" * 64  # the freeze digest is on disk before test.bin
    second = b.record_test_access({"git_sha": "b" * 40}, "h2", "2" * 64, log, label=b.SHAKEDOWN_TEST_ACCESS_LOG)
    assert second["prior_openings"] == [first["this_opening"]] and second["this_opening"]["run_config_hash"] == "h2"
    assert second["log"] == "<output>.work/run_evidence/test_access_log.jsonl"
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert lines == [first["this_opening"], second["this_opening"]] == b._read_access_log(log)


def test_optimizer_snapshot_digest_covers_momentum_and_the_auxiliary_state():
    def snapshot():
        return {"model": {"w": torch.ones(2)}, "sha256": "x",
                "momentum": {"blocks.0.up": torch.arange(6, dtype=torch.float32).reshape(2, 3), "b": torch.zeros(4)},
                "auxiliary": {"state": {0: {"step": torch.tensor(3.0), "exp_avg": torch.full((2,), 0.5)}},
                              "param_groups": [{"lr": 0.001, "betas": (0.9, 0.95), "params": [0]}]}}

    base = b.optimizer_snapshot_sha256(snapshot())
    assert len(base) == 64 and base == b.optimizer_snapshot_sha256(snapshot())
    same = snapshot()
    same["model"]["w"] += 1  # the weights have their own digest (snapshot_sha256)
    same["momentum"] = dict(reversed(same["momentum"].items()))  # key order does not matter
    assert b.optimizer_snapshot_sha256(same) == base
    edits = (lambda s: s["momentum"]["b"].__setitem__(0, 1e-6),
             lambda s: s["momentum"].update(b=torch.zeros(2, 2)),  # the same bytes in another shape
             lambda s: s["momentum"].update(b=torch.zeros(4, dtype=torch.float64)),
             lambda s: s["auxiliary"]["state"][0].update(step=torch.tensor(4.0)),
             lambda s: s["auxiliary"]["param_groups"][0].update(lr=0.002),
             lambda s: s["momentum"].pop("b"))
    for edit in edits:
        other = snapshot()
        edit(other)
        assert b.optimizer_snapshot_sha256(other) != base


def test_awq_freeze_lists_alpha_scores_and_effective_weights_of_every_selection():
    units = [{"part": "ab", "seed": 1, "arm": b.MUON,
              "awq_selection": [{"method": b.C5_BEST, "alphas": [0.0, 0.25], "val_nll_hw_dynamic": [5.2, 5.1],
                                 "selected_alpha": 0.25, "selected_on": "val.bin, w4a4_hw_dynamic", "seconds": 1.0}],
              "grid": [{"method": b.C5_BEST, "mode": "w4a16_fakequant", "effective_weights_sha256": "e" * 64}]},
             {"part": "c", "seed": 2, "arm": b.QAT_C, "awq_selection": [], "grid": []}]
    frozen = b.awq_freeze(units)
    assert frozen == [{"part": "ab", "seed": 1, "arm": b.MUON, "method": b.C5_BEST, "selected_alpha": 0.25,
                       "val_nll_hw_dynamic": [5.2, 5.1], "effective_weight_sha256": "e" * 64}]
    base = b.selections_sha256({"awq_selections": frozen})
    for key, value in (("selected_alpha", 0.0), ("val_nll_hw_dynamic", [5.2, 5.0]), ("effective_weight_sha256", "f" * 64)):
        assert b.selections_sha256({"awq_selections": [{**frozen[0], key: value}]}) != base


def test_layer_summary_bounds_and_metric_ranges():
    def summary(**changes):
        return {"all": {"mean": 3.0, "median": 2.5, "min": 1.5, "max": 6.0, **changes},
                "by_kind": dict(zip(q.LINEAR_KINDS, (1.5, 2.0, 2.5, 3.0, 6.0)))}

    b._validate_layer_summary(summary(), "max_over_rms", "x")
    for changes, metric, match in (({"min": 6.0, "max": 1.5}, "max_over_rms", "bounds"),  # inverted extrema
                                   ({"mean": 7.0}, "max_over_rms", "bounds"), ({"median": 1.0}, "max_over_rms", "bounds"),
                                   ({"min": 2.0}, "max_over_rms", "bounds"),  # a per-kind mean below the minimum
                                   ({"mean": 3.5}, "max_over_rms", "bounds"),  # not the mean of the per-kind means
                                   ({"max": float("inf")}, "max_over_rms", "numbers"),
                                   ({"min": True}, "max_over_rms", "numbers"),
                                   ({}, "group16_outlier_share", "range")):  # a share above 1
        with pytest.raises(ValueError, match=match):
            b._validate_layer_summary(summary(**changes), metric, "x")
    low = {"all": {"mean": 0.5, "median": 0.5, "min": 0.5, "max": 0.5}, "by_kind": dict.fromkeys(q.LINEAR_KINDS, 0.5)}
    for metric in ("group16_outlier_share", "nvfp4_relative_sq_error", "excess_kurtosis", "weight_stable_rank"):
        b._validate_layer_summary(low, metric, "x")
    for metric in ("max_over_rms", "group16_max_over_rms"):  # a maximum is never below the RMS
        with pytest.raises(ValueError, match="range"):
            b._validate_layer_summary(low, metric, "x")
    negative = {"all": dict.fromkeys(("mean", "median", "min", "max"), -2.5), "by_kind": dict.fromkeys(q.LINEAR_KINDS, -2.5)}
    for metric in b.METRIC_LOW:  # below every physical lower bound (kurtosis >= -2, an error or a rank >= 0)
        with pytest.raises(ValueError, match="range"):
            b._validate_layer_summary(negative, metric, "x")
    assert set(b.METRIC_LOW) == set(rq.ACTIVATION_METRICS) | {"weight_stable_rank", "signal_alignment"}


def test_stream_fingerprint_is_recomputed_from_the_split_length():
    config = b.PRESETS["shakedown"]
    tokens = np.arange(5000, dtype=np.uint16)
    for seed in (1, 2):
        assert b._expected_stream_fingerprint(5000, "shakedown", seed) == h.stream_fingerprint(tokens, config, seed)
    assert b._expected_stream_fingerprint(5000, "shakedown", 1) != b._expected_stream_fingerprint(5000, "shakedown", 2)
    assert b._expected_stream_fingerprint(5000, "shakedown", 1) != b._expected_stream_fingerprint(5001, "shakedown", 1)


def test_summary_code_of_the_v1_producer_is_the_code_of_this_checkout():
    """The driver changed after the canonical run; the definitions that recompute its result did not."""
    producer = "1b4ea1c17ecc3f6b0e1232693a13ccca25ad4d5d"
    assert b._schema_at_commit(producer) == b.HISTORICAL_SCHEMA_V1 == "cycle6_c6_01.v1"
    assert b.SCHEMA_VERSION == "cycle6_c6_01.v2" and b.SCHEMAS == ("cycle6_c6_01.v1", "cycle6_c6_01.v2")
    assert b._matches_current_summary_code(producer)
    committed = c503._git_blob(producer, b.SCRIPT_REPO_PATH).decode()
    current = Path(b.__file__).read_text()
    assert committed != current
    assert b._same_summary_definitions(committed, current, b.HISTORICAL_SCHEMA_V1)
    assert {"analyze", "summarize", "select_probe", "contrast_specs", "CANONICAL", "PINNED_BASELINES"} <= set(
        b.SUMMARY_DEFINITIONS)
    assert set(b.SUMMARY_DEFINITIONS) | {"caveats", "_caveats_v1"} <= set(b._definitions(current))
    # The v1 caveats are ``caveats`` of the producer and ``_caveats_v1`` here; a v1 driver has no v2 caveats.
    assert b._definitions(committed)["caveats"] == b._definitions(current)["_caveats_v1"]
    assert not b._same_summary_definitions(committed, current, b.SCHEMA_VERSION)
    assert not b._same_summary_definitions(committed, current, None)
    assert b._same_summary_definitions(current, current, b.SCHEMA_VERSION)
    for old, new in (("def verdict(stats: dict) -> str:", "def verdict(stats: dict) -> str:\n    stats = dict(stats)"),
                     ("N_SEEDS = 3", "N_SEEDS = 4"), ('"dual_bf16_weight": 0.5,', '"dual_bf16_weight": 0.25,'),
                     ("def _caveats_v1(payload: dict)", "def _caveats_v0(payload: dict)"),
                     ('"Part C cooldowns start Muon momentum', '"Part C cooldowns restart Muon momentum')):
        assert current.count(old) == 1
        assert not b._same_summary_definitions(committed, current.replace(old, new), b.HISTORICAL_SCHEMA_V1), old


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

    canonical_log = out.with_name("canonical_evidence") / b.TEST_ACCESS_LOG_NAME
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(q, "default_data_dir", forbidden)
        patch.setattr(b, "default_baseline_dir", forbidden)
        patch.setattr(b, "TEST_ACCESS_LOG", canonical_log)  # the canonical access log is for the real test.bin only
        payload = b.main(["--synthetic-shakedown", "--silent", "--output", str(out)])
    assert json.loads(out.read_text()) == payload and not canonical_log.parent.exists()
    assert (b.CANONICAL_OUTPUT.read_bytes() if b.CANONICAL_OUTPUT.exists() else None) == canonical_before
    return payload


# Validation receipts. A v2 result has its AWQ selections in the pre-test freeze digest; the preserved v1 result
# does not (they are bound by a digest that was pinned after the run).
NONCANONICAL = {"certified_canonical": False, "shakedown": True, "schema_version": "cycle6_c6_01.v2",
                "awq_in_contemporaneous_freeze": True}
CERTIFIED = {"certified_canonical": True, "shakedown": False, "schema_version": "cycle6_c6_01.v1",
             "awq_in_contemporaneous_freeze": False}


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


def _work(payload):
    """The work directory of a shakedown: its checkpoints and its test-access log."""
    return Path(payload["meta"]["argv"][-1] + ".work")


def _recompute(p):
    """Recompute everything that is derived from the rows, as a consistent forger would."""
    p["analysis"] = json.loads(json.dumps(b.analyze(p["units"], p["selections"], p["protocol"], p["c5_03_reference"])))
    p["summary"] = json.loads(json.dumps(b.summarize(p)))
    p["caveats"] = b.caveats(p)


def _shift_nll(u, row, split, nats):
    """Move one NLL and keep the row consistent (perplexity, delta to the BF16 baseline)."""
    base = u[f"baseline_{split}"]
    row[f"{split}_nll"] += nats
    row[f"{split}_ppl"] = math.exp(row[f"{split}_nll"])
    row[f"{split}_delta_nll"] = row[f"{split}_nll"] - base["nll"]
    row[f"{split}_relative_delta_ppl_pct"] = 100.0 * (row[f"{split}_ppl"] / base["ppl"] - 1.0)


def _other_awq_winner(p, part="ab", seed=1, arm=None, index=0):
    """A different AWQ winner with consistent records: the winning val.bin score moves to another alpha.

    The dynamic row keeps its val NLL (the winning score), so no selection, statistic or summary changes.
    """
    u = _unit(p, part, seed, arm or b.MUON)
    a = u["awq_selection"][index]
    i = a["alphas"].index(a["selected_alpha"])
    j = (i + 1) % len(a["alphas"])
    scores = a["val_nll_hw_dynamic"]
    assert scores[i] != scores[j]
    scores[i], scores[j] = scores[j], scores[i]
    a["selected_alpha"] = a["alphas"][j]
    for mode in b.MODES:
        _grid_row(u, a["method"], mode)["awq_alpha"] = a["alphas"][j]
    return u, a


def _in_window(p, seconds):
    """A timestamp ``seconds`` after the start of the run, in the format of the driver."""
    start = c502._timestamp(p["meta"]["timestamp_start"], "start")
    return (start + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%S%z")


def _unit(p, part, seed, arm):
    return next(u for u in p["units"] if (u["part"], u["seed"], u["arm"]) == (part, seed, arm))


def _grid_row(u, method, mode):
    return next(r for r in u["grid"] if (r["method"], r["mode"]) == (method, mode))


@needs_fp4
def test_shakedown_is_valid_complete_and_synthetic(shakedown):
    p = shakedown
    assert b.validate_results(p) == b.validate_results(p, canonical=False) == NONCANONICAL
    assert p["schema_version"] == "cycle6_c6_01.v2" and p["shakedown"] and not p["canonical"]
    data = p["data"]
    assert data["synthetic"] and data["source"]["name"] == "synthetic-markov" and data["roles"] == b.EXPECTED_ROLES
    # The freeze: the AWQ choice of every (part, seed, arm, method) is in the digest that the run logged before it
    # mapped test.bin, and the log line is on disk.
    access, sel = data["test_access"], p["selections"]
    assert access["log"] == b.SHAKEDOWN_TEST_ACCESS_LOG and access["prior_openings"] == []
    assert access["this_opening"]["selections_sha256"] == p["selections_sha256"] == b.selections_sha256(sel)
    assert access["this_opening"]["run_config_hash"] == p["run_config_hash"]
    assert b._read_access_log(_work(p) / "run_evidence" / "test_access_log.jsonl") == [access["this_opening"]]
    frozen = sel["awq_selections"]
    assert frozen == b.awq_freeze(p["units"]) and len(frozen) == sum(len(u["awq_selection"]) for u in p["units"]) == 36
    for entry in frozen:
        u = _unit(p, entry["part"], entry["seed"], entry["arm"])
        row = _grid_row(u, entry["method"], "w4a16_fakequant")
        assert entry["effective_weight_sha256"] == row["effective_weights_sha256"] == row["effective_weights_sha256_test"]
        assert entry["selected_alpha"] == _grid_row(u, entry["method"], b.DYNAMIC)["awq_alpha"]
        assert min(entry["val_nll_hw_dynamic"]) == _grid_row(u, entry["method"], b.DYNAMIC)["val_nll"]
    # The retained files: all 27 checkpoints (file and tensor digests) and the test-access log.
    external = b.validate_results(p, verify_external=True)
    assert {k: v for k, v in external.items() if k != "external_evidence"} == NONCANONICAL
    evidence = external["external_evidence"]
    assert evidence["checkpoints_trained_in_run_verified"] == 27 and evidence["checkpoints_c5_03_preserved_verified"] == 0
    assert evidence["test_access_log_verified"] and evidence["test_access_log_entries"] == 1
    assert evidence["unrecorded_checkpoint_files"] == []
    assert b.verify_external_evidence(p, checkpoint_dir=_work(p) / "checkpoints",
                                      evidence_dir=_work(p) / "run_evidence") == evidence
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
        # Paired controls: one initialization and batch stream per seed, one step-30 snapshot for the cooldowns.
        arms = [_unit(p, "ab", seed, a)["training"] for a in b.ARMS_AB]
        assert len({tr["init_fingerprint"] for tr in arms}) == len({tr["stream_fingerprint"] for tr in arms}) == 1
        branches = [tr["branch"] for tr in arms if tr["branch"] is not None]
        assert len(branches) == 4 and all(br == wide["branch"] for br in branches)
        assert set(wide["branch"]) == b.BRANCH_KEYS[b.SCHEMA_VERSION]
        assert len(wide["branch"]["optimizer_snapshot_sha256"]) == 64
        assert wide["branch"]["snapshot_sha256"] != wide["branch"]["optimizer_snapshot_sha256"]
        assert _unit(p, "ab", seed, "qat_identity_full")["training"]["qat"]["fake_quant_steps"] == 40
        dual = _unit(p, "ab", seed, "qat_wide_dual_cooldown")["training"]["qat"]
        assert dual["loss"] == "dual" and dual["bf16_weight"] == 0.5
        base, cont, qat = (_unit(p, "c", seed, a)["training"] for a in b.ARMS_C)
        assert base["steps"] == 30 and cont["steps"] == qat["steps"] == 38 and cont["first_step"] == 30
        assert cont["branch"]["snapshot_sha256"] == qat["branch"]["snapshot_sha256"] == base["checkpoint_sha256"]
        assert cont["branch"] == qat["branch"] and cont["branch"]["optimizer_snapshot_sha256"] is None  # fresh optimizer
        assert cont["init_fingerprint"] is None and len(base["init_fingerprint"]) == 16
        assert cont["stream_fingerprint"] == qat["stream_fingerprint"] == base["stream_fingerprint"]
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
    for forbidden in ("FineWeb", "C6-01 test-access log", "640 steps", "320 steps", "real_text_data.py"):
        assert forbidden not in notes
    # The v2 caveats: unequal method menus, the exact split-hygiene guarantee and the freeze coverage.
    assert "Method menus differ in Part B" in notes and "use the same-method contrasts" in notes
    assert "cross_b64_butterfly_b16_gptq, cross_full_butterfly_b16_gptq" in notes
    assert "zero pre-freeze test scoring or selection" in notes and "writes and hashes its synthetic test.bin" in notes
    assert "Freeze coverage (cycle6_c6_01.v2)" in notes and "all 36 AWQ selections" in notes
    assert "an anti-tamper pin, not a contemporaneous freeze" in notes
    assert p["caveats"][:-3] == b._caveats_v1(p)  # a shakedown has no held-out note to reword
    with pytest.raises(ValueError, match="not labeled canonical"):
        b.validate_results(p, canonical=True)


@needs_fp4
def test_validator_rejects_tampered_labels_hashes_and_provenance(shakedown, tmp_path):
    _rejects(shakedown, lambda p: p.update(canonical=True), "either the canonical run or a synthetic shakedown")
    _rejects(shakedown, lambda p: p.update(canonical=True, shakedown=False), "differs from the canonical C6-01 protocol")
    _rejects(shakedown, lambda p: p.update(shakedown=False), "either the canonical run or a synthetic shakedown")
    _rejects(shakedown, lambda p: p.update(schema_version="cycle6_c6_01.v0"), "schema or task differs")
    _rejects(shakedown, lambda p: p.update(schema_version=["cycle6_c6_01.v2"]), "schema or task differs")
    _rejects(shakedown, lambda p: p.update(schema_version="cycle6_c6_01.v1"), "preserved canonical result only")
    _rejects(shakedown, lambda p: p.update(extra=1), "top-level keys")
    _rejects(shakedown, lambda p: _set(p, ("protocol", "seeds"), [1, 2, 4]), "protocol differs")
    _rejects(shakedown, lambda p: _set(p, ("protocol", "part_c", "cooldown_steps"), 9), "protocol differs")
    _rejects(shakedown, lambda p: _set(p, ("meta", "code_sha256"), "0" * 64), "code_sha256 differs")
    _rejects(shakedown, lambda p: _set(p, ("meta", "code_modules"), p["meta"]["code_modules"][:-1]), "code_modules")
    _rejects(shakedown, lambda p: p.update(run_config_hash="0" * 16), "must name this run")

    def config_hash(p):  # consistently in the payload and in its test-access line
        p["run_config_hash"] = p["data"]["test_access"]["this_opening"]["run_config_hash"] = "0" * 16

    _rejects(shakedown, config_hash, "run_config_hash differs")
    _rejects(shakedown, lambda p: _set(p, ("meta", "environment", "driver"), "0.0"), "run_config_hash differs")
    _rejects(shakedown, lambda p: _set(p, ("meta", "numerics", "allow_tf32"), True), "full-precision numerics")
    _rejects(shakedown, lambda p: _set(p, ("meta", "splits_opened"), ["train", "test", "val"]), "splits_opened")
    _rejects(shakedown, lambda p: _set(p, ("meta", "timestamp_end"), p["meta"]["timestamp_start"]), "run interval")
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
    _rejects(shakedown, lambda p: _set(p, ("meta", "git_sha"), "0" * 40), "must name this run")

    def producer(p):
        p["meta"]["git_sha"] = p["data"]["test_access"]["this_opening"]["git_sha"] = "0" * 40

    _rejects(shakedown, producer, "not a commit object")
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
def test_validator_rejects_wrong_split_roles_and_test_opening_times(shakedown):
    """H1: the split-role map and the opening of test.bin (schema, run, freeze digest, time)."""
    roles = lambda p: p["data"]["roles"]  # noqa: E731
    _rejects(shakedown, lambda p: roles(p).update(training="test"), "data.roles")  # test.bin declared as training data
    _rejects(shakedown, lambda p: roles(p).update(awq_alpha_selection="test"), "data.roles")
    _rejects(shakedown, lambda p: roles(p).update(evaluation="val"), "data.roles")
    _rejects(shakedown, lambda p: roles(p).pop("static_scales"), "data.roles")
    _rejects(shakedown, lambda p: roles(p).update(tuning="test"), "data.roles")
    _rejects(shakedown, lambda p: p["data"].update(roles={}), "data.roles")
    opening = lambda p: p["data"]["test_access"]["this_opening"]  # noqa: E731
    _rejects(shakedown, lambda p: opening(p).update(timestamp="1900-01-01T00:00:00-0700"), "outside the run interval")
    _rejects(shakedown, lambda p: opening(p).update(timestamp=_in_window(p, 3600 * 24)), "outside the run interval")
    _rejects(shakedown, lambda p: opening(p).update(timestamp=_in_window(p, 1)), "follow stages A-C")  # before the freeze
    _rejects(shakedown, lambda p: opening(p).update(timestamp=p["meta"]["timestamp_end"]), "precede the test scoring")
    _rejects(shakedown, lambda p: opening(p).update(timestamp="yesterday"), "invalid ISO timestamp")
    _rejects(shakedown, lambda p: opening(p).pop("selections_sha256"), "this_opening keys")  # a v1 line in a v2 result
    _rejects(shakedown, lambda p: opening(p).update(host="x"), "this_opening keys")
    _rejects(shakedown, lambda p: opening(p).update(selections_sha256="0" * 64), "differs from the freeze digest")
    _rejects(shakedown, lambda p: opening(p).update(selections_sha256="xyz"), "selections_sha256 invalid")
    _rejects(shakedown, lambda p: p["data"].update(test_access=None), "test_access")
    _rejects(shakedown, lambda p: p["data"]["test_access"].update(log=b.TEST_ACCESS_LOG_REPO_PATH), "test_access")
    # Earlier openings: a v1 or a v2 line each, not later than this opening.
    prior = lambda p: p["data"]["test_access"]["prior_openings"]  # noqa: E731
    v1_line = {"git_sha": "a" * 40, "run_config_hash": "b" * 16, "timestamp": "2026-11-27T19:03:38-0700"}
    ok = copy.deepcopy(shakedown)
    prior(ok).extend([v1_line, {**v1_line, "selections_sha256": "c" * 64}])
    assert b.validate_results(ok) == NONCANONICAL
    _rejects(shakedown, lambda p: prior(p).append({**v1_line, "timestamp": _in_window(p, 3600 * 24)}),
             "later than this opening")
    _rejects(shakedown, lambda p: prior(p).append({**v1_line, "timestamp": None}), "invalid ISO timestamp")
    _rejects(shakedown, lambda p: prior(p).append({**v1_line, "git_sha": "main"}), "prior_openings.0.: git_sha")
    _rejects(shakedown, lambda p: prior(p).append({"git_sha": "a" * 40}), "keys differ from a test-access log entry")
    _rejects(shakedown, lambda p: prior(p).append("earlier"), "keys differ from a test-access log entry")
    _rejects(shakedown, lambda p: p["data"]["test_access"].update(prior_openings=None), "test_access")


@needs_fp4
def test_validator_rejects_broken_paired_controls_and_training_curves(shakedown):
    """H2 and M1: sibling branch digests, fingerprints, the optimizer snapshot digest and the training curve."""
    tr = lambda p, arm="qat_wide_cooldown", part="ab", seed=1: _unit(p, part, seed, arm)["training"]  # noqa: E731
    # One valid hash that is not the hash of the siblings.
    _rejects(shakedown, lambda p: tr(p)["branch"].update(snapshot_sha256="0" * 64), "sibling branch records differ")
    _rejects(shakedown, lambda p: tr(p, "qat_b16_cooldown", seed=2)["branch"].update(optimizer_snapshot_sha256="0" * 64),
             "sibling branch records differ")
    _rejects(shakedown, lambda p: tr(p)["branch"].update(optimizer_snapshot_sha256=None), "optimizer_snapshot_sha256 must be")
    _rejects(shakedown, lambda p: tr(p)["branch"].pop("optimizer_snapshot_sha256"), "must branch from")
    _rejects(shakedown, lambda p: tr(p)["branch"].update(from_arm="qat_identity_full"), "must branch from")
    _rejects(shakedown, lambda p: tr(p).update(branch=None), "must branch from")
    _rejects(shakedown, lambda p: tr(p, "qat_identity_full").update(branch=dict(tr(p)["branch"])), "full run has no branch")
    _rejects(shakedown, lambda p: tr(p, b.MUON).update(branch=dict(tr(p)["branch"])), "full run has no branch")
    _rejects(shakedown, lambda p: tr(p, b.CONTINUE, "c")["branch"].update(optimizer_snapshot_sha256="0" * 64),
             "optimizer_snapshot_sha256 must be")
    _rejects(shakedown, lambda p: tr(p, b.CONTINUE, "c")["branch"].update(snapshot_sha256="0" * 64), "must branch from")
    # Fingerprints: the batch stream is recomputed; the initialization is one per seed.
    other_seed = tr(shakedown, seed=2)
    _rejects(shakedown, lambda p: tr(p).update(stream_fingerprint=other_seed["stream_fingerprint"]),
             "stream_fingerprint differs")
    _rejects(shakedown, lambda p: tr(p, b.QAT_C, "c").update(stream_fingerprint="0" * 16), "stream_fingerprint differs")
    _rejects(shakedown, lambda p: tr(p).update(init_fingerprint=other_seed["init_fingerprint"]),
             "do not share one init_fingerprint")
    _rejects(shakedown, lambda p: tr(p, b.MUON).update(init_fingerprint="0" * 16), "do not share one init_fingerprint")
    _rejects(shakedown, lambda p: tr(p).update(init_fingerprint="xyz"), "init_fingerprint must be")
    _rejects(shakedown, lambda p: tr(p, b.CONTINUE, "c").update(init_fingerprint="0" * 16), "init_fingerprint must be")
    _rejects(shakedown, lambda p: tr(p, b.BASELINE, "c").update(init_fingerprint=None), "init_fingerprint must be")

    def all_arms(key, value):  # consistent across the siblings: only the recomputed stream can object
        def edit(p):
            for arm in b.ARMS_AB:
                tr(p, arm)[key] = value
        return edit

    _rejects(shakedown, all_arms("stream_fingerprint", "0" * 16), "stream_fingerprint differs")
    # The training curve: the first step, every tenth step and the last step, each with a positive finite loss.
    curve = lambda p, arm=b.MUON: tr(p, arm)["train_curve"]  # noqa: E731
    assert [pt[0] for pt in curve(shakedown)] == [1, 10, 20, 30, 40]
    assert [pt[0] for pt in curve(shakedown, "qat_wide_cooldown")] == [31, 40]
    _rejects(shakedown, lambda p: curve(p)[1].__setitem__(0, 10_000), "train_curve must hold")  # an impossible step
    _rejects(shakedown, lambda p: curve(p)[1].__setitem__(1, -1.0), "train_curve must hold")  # a negative loss
    _rejects(shakedown, lambda p: curve(p).__setitem__(1, [10_000, -1.0]), "train_curve must hold")
    _rejects(shakedown, lambda p: curve(p)[2].__setitem__(1, 0.0), "train_curve must hold")
    _rejects(shakedown, lambda p: curve(p)[2].__setitem__(0, 20.0), "train_curve must hold")  # a float step
    _rejects(shakedown, lambda p: curve(p)[2].__setitem__(0, True), "train_curve must hold")
    _rejects(shakedown, lambda p: curve(p).insert(1, [0, 5.0]), "train_curve must hold")
    _rejects(shakedown, lambda p: curve(p).__setitem__(slice(1, 3), [curve(p)[2], curve(p)[1]]), "train_curve steps")
    _rejects(shakedown, lambda p: curve(p).pop(2), "train_curve steps")
    _rejects(shakedown, lambda p: curve(p).pop(0), "train_curve steps")
    _rejects(shakedown, lambda p: curve(p)[0].__setitem__(0, 2), "train_curve steps")
    _rejects(shakedown, lambda p: curve(p).insert(2, [15, 5.0]), "train_curve steps")
    _rejects(shakedown, lambda p: curve(p, "qat_wide_cooldown")[0].__setitem__(0, 30), "train_curve must hold")
    _rejects(shakedown, lambda p: tr(p).update(train_curve=[]), "train_curve must hold")
    _rejects(shakedown, lambda p: curve(p)[-1].__setitem__(1, curve(p)[-1][1] + 0.5), "training curve record")
    _rejects(shakedown, lambda p: tr(p).update(train_loss=-1.0), "training curve record")
    _rejects(shakedown, lambda p: tr(p)["step_seconds"].__setitem__(0, 0.0), "one positive wall time")
    _rejects(shakedown, lambda p: tr(p)["step_seconds"].__setitem__(0, -1.0), "one positive wall time")
    _rejects(shakedown, lambda p: tr(p).update(train_seconds=tr(p)["train_seconds"] - 1.0), "train_seconds")


@needs_fp4
def test_validator_rejects_impossible_counters_and_summary_bounds(shakedown):
    """M1: counter types and ranges, the emulation invariants and the bounds of the per-layer summaries."""
    def row(method, mode, part="ab", seed=1, arm=b.MUON):
        return lambda p: _grid_row(_unit(p, part, seed, arm), method, mode)

    emulated = row("kron_hadamard_full_gptq", b.EMULATED_MODE)
    assert emulated(shakedown)["cupti"]["nvfp4_quant_kernels"] == emulated(shakedown)["cupti"]["nvfp4_amax_kernels"] == 10
    for key in rq.CUPTI_KEYS:  # negative kernel counts on an emulation row
        _rejects(shakedown, lambda p, key=key: emulated(p)["cupti"].__setitem__(key, -1), "integers >= 0")
        _rejects(shakedown, lambda p, key=key: emulated(p)["cupti"].__setitem__(key, -emulated(p)["cupti"][key] - 5),
                 "integers >= 0")
    _rejects(shakedown, lambda p: emulated(p)["cupti"].update(fp4_gemm_kernels=False), "integers >= 0")  # a bool is no count
    _rejects(shakedown, lambda p: emulated(p)["cupti"].update(nvfp4_quant_kernels=10.0), "integers >= 0")
    _rejects(shakedown, lambda p: emulated(p).update(bf16_block_linears=False), "integers >= 0")
    _rejects(shakedown, lambda p: emulated(p).update(nvfp4_modules=-10), "integers >= 0")
    _rejects(shakedown, lambda p: emulated(p).update(cupti=None), "integers >= 0")
    _rejects(shakedown, lambda p: emulated(p)["cupti"].update(extra_kernels=1), "integers >= 0")
    _rejects(shakedown, lambda p: emulated(p).update(hw_weights_checked_matrices=10), "integers >= 0")
    # The exact emulation invariants: no FP4 GEMM, the quantizer and amax kernels of every layer, no lost GEMM.
    _rejects(shakedown, lambda p: emulated(p)["cupti"].update(fp4_gemm_kernels=1), "emulation must run no FP4 GEMM")
    _rejects(shakedown, lambda p: emulated(p)["cupti"].update(nvfp4_quant_kernels=0), "emulation must run no FP4 GEMM")
    _rejects(shakedown, lambda p: emulated(p)["cupti"].update(nvfp4_amax_kernels=9), "emulation must run no FP4 GEMM")
    _rejects(shakedown, lambda p: emulated(p)["cupti"].update(non_fp4_gemm_kernels=0), "emulation must run no FP4 GEMM")
    _rejects(shakedown, lambda p: emulated(p).update(bf16_block_linears=1), "emulation must run no FP4 GEMM")
    _rejects(shakedown, lambda p: emulated(p).update(emulated_modules=9), "emulation must run no FP4 GEMM")
    _rejects(shakedown, lambda p: emulated(p).update(cross_mix_modules=0), "emulation must run no FP4 GEMM")
    hw = row("kron_hadamard_full_gptq", b.DYNAMIC)
    _rejects(shakedown, lambda p: hw(p)["cupti"].update(non_fp4_gemm_kernels=-1), "integers >= 0")
    _rejects(shakedown, lambda p: hw(p).update(bf16_block_linears=False, emulated_modules=False), "integers >= 0")
    w4a16 = row("kron_hadamard_full_gptq", "w4a16_fakequant")
    _rejects(shakedown, lambda p: w4a16(p).update(fp4_gemms_per_forward=0), "no hardware counter")
    _rejects(shakedown, lambda p: w4a16(p).update(cupti=dict(hw(p)["cupti"])), "no hardware counter")
    _rejects(shakedown, lambda p: w4a16(p).update(bf16_block_linears=10), "no hardware counter")
    _rejects(shakedown, lambda p: w4a16(p).update(hw_weights_checked_matrices=-10, hw_weights_decode_exact_matrices=-10),
             "no hardware counter")
    unit = lambda p: _unit(p, "ab", 1, b.MUON)  # noqa: E731
    _rejects(shakedown, lambda p: unit(p).update(bf16_fp4_gemms_per_forward=False), "BF16 model must run no")
    _rejects(shakedown, lambda p: unit(p)["bf16_cupti"].update(nvfp4_amax_kernels=False), "BF16 model must run no")
    # Summaries. The analysis reads the means only, so nothing else can object to the extrema.
    def stat(p, family="identity", metric="excess_kurtosis"):
        return unit(p)["activation_statistics"][family][metric]

    def invert(summary):
        def edit(p):
            a = summary(p)["all"]
            a["min"], a["max"] = a["max"], a["min"]
        return edit

    _rejects(shakedown, invert(stat), "summary bounds violated")  # an activation maximum below its minimum
    _rejects(shakedown, invert(lambda p: stat(p, "kron_hadamard_full", "max_over_rms")), "summary bounds violated")
    _rejects(shakedown, invert(lambda p: unit(p)["geometry"]["weight_stable_rank"]), "summary bounds violated")
    _rejects(shakedown, lambda p: stat(p)["all"].update(median=stat(p)["all"]["max"] + 1.0), "summary bounds violated")
    _rejects(shakedown, lambda p: stat(p)["by_kind"].update(down=stat(p)["all"]["max"] + 1.0), "summary bounds violated")
    _rejects(shakedown, lambda p: stat(p)["all"].update(min=-3.0), "outside the range of excess_kurtosis")
    _rejects(shakedown, lambda p: stat(p, "identity", "max_over_rms")["all"].update(min=0.5),
             "outside the range of max_over_rms")
    _rejects(shakedown, lambda p: stat(p, "identity", "group16_max_over_rms")["all"].update(min=0.9),
             "outside the range of group16_max_over_rms")
    _rejects(shakedown, lambda p: stat(p, "identity", "nvfp4_relative_sq_error")["all"].update(min=-1e-3),
             "outside the range of nvfp4_relative_sq_error")
    _rejects(shakedown, lambda p: stat(p, "identity", "group16_outlier_share")["all"].update(max=1.5),
             "outside the range of group16_outlier_share")
    _rejects(shakedown, lambda p: unit(p)["geometry"]["signal_alignment"]["all"].update(min=0.0),
             "outside the range of signal_alignment")
    _rejects(shakedown, lambda p: unit(p)["geometry"]["weight_stable_rank"]["all"].update(min=-1.0),
             "outside the range of weight_stable_rank")
    _rejects(shakedown, lambda p: stat(p)["all"].pop("median"), "summary keys or numbers")
    _rejects(shakedown, lambda p: stat(p)["all"].update(max="large"), "summary keys or numbers")
    _rejects(shakedown, lambda p: unit(p)["geometry"].pop("signal_alignment"), "geometry metrics differ")


@needs_fp4
def test_awq_selections_are_frozen_before_test_and_bound_to_the_log(shakedown, tmp_path):
    """H3 (v2): a different AWQ winner fails against the frozen selections, the freeze digest, the opening record
    and, when every record in the JSON is rewritten, the line that the run appended before it mapped test.bin."""
    _rejects(shakedown, lambda p: _other_awq_winner(p), "selections.awq_selections differ")
    _rejects(shakedown, lambda p: _other_awq_winner(p, "c", 3, b.QAT_C), "selections.awq_selections differ")

    def other_weights(p):  # the same alpha and scores, other quantized weights (consistent on val and test)
        r = _grid_row(_unit(p, "ab", 1, b.MUON), b.C5_BEST, "w4a16_fakequant")
        r["effective_weights_sha256"] = r["effective_weights_sha256_test"] = "0" * 64

    _rejects(shakedown, other_weights, "selections.awq_selections differ")
    _rejects(shakedown, lambda p: p["selections"]["awq_selections"].pop(), "selections.awq_selections differ")
    _rejects(shakedown, lambda p: p["selections"]["awq_selections"][0].update(selected_alpha=0.75),
             "selections.awq_selections differ")
    _rejects(shakedown, lambda p: p["selections"].pop("awq_selections"), "selections keys or values")
    _rejects(shakedown, lambda p: _unit(p, "ab", 1, b.MUON)["awq_selection"][0]["val_nll_hw_dynamic"].__setitem__(0, -1.0),
             "AWQ selection record invalid")
    _rejects(shakedown, lambda p: _unit(p, "ab", 1, b.MUON)["awq_selection"][0].update(note="x"),
             "AWQ selection record invalid")

    forged = copy.deepcopy(shakedown)
    _other_awq_winner(forged)
    forged["selections"]["awq_selections"] = b.awq_freeze(forged["units"])  # the frozen list follows the units
    with pytest.raises(ValueError, match="selections_sha256 differs"):
        b.validate_results(forged)
    forged["selections_sha256"] = b.selections_sha256(forged["selections"])  # and the freeze digest follows
    with pytest.raises(ValueError, match="differs from the freeze digest"):
        b.validate_results(forged)
    forged["data"]["test_access"]["this_opening"]["selections_sha256"] = forged["selections_sha256"]
    with pytest.raises(ValueError, match="summary"):
        b.validate_results(forged)
    _recompute(forged)
    # Every record in the JSON is now consistent. The digest on disk is not: it was written before test.bin.
    assert forged["selections_sha256"] != shakedown["selections_sha256"]
    assert b.validate_results(forged) == NONCANONICAL
    with pytest.raises(ValueError, match="differs from the external test-access log"):
        b.validate_results(forged, verify_external=True)
    with pytest.raises(ValueError, match="differs from the external test-access log"):
        b.verify_external_evidence(forged)
    # The log itself: a missing file, a missing line, an edited line, text that is not JSON.
    evidence = tmp_path / "run_evidence"
    shutil.copytree(_work(shakedown) / "run_evidence", evidence)
    log = evidence / b.TEST_ACCESS_LOG_NAME
    assert b.validate_results(shakedown, evidence_dir=evidence)["external_evidence"]["test_access_log"] == str(log)
    line = log.read_text()
    log.write_text(line + line)  # later openings may follow
    assert b.verify_external_evidence(shakedown, evidence_dir=evidence)["test_access_log_entries"] == 2
    for text, match in (("", "differs from the external"), (line.replace("2026", "2025"), "differs from the external"),
                        ('{"git_sha": null}\n' + line, "differs from the external"), ("not json\n", "is not JSON lines")):
        log.write_text(text)
        with pytest.raises(ValueError, match=match):
            b.verify_external_evidence(shakedown, evidence_dir=evidence)
    with pytest.raises(ValueError, match="is missing"):
        b.verify_external_evidence(shakedown, evidence_dir=tmp_path / "nowhere")


@needs_fp4
def test_external_evidence_rejects_changed_checkpoint_digests(shakedown, tmp_path):
    """M2: the JSON holds hex strings; only the retained files can show that a checkpoint digest is the real one."""
    for part, arm, key, match in (("ab", "qat_wide_cooldown", "checkpoint_sha256", "tensor SHA-256"),
                                  ("ab", b.MUON, "checkpoint_file_sha256", "the SHA-256 of"),
                                  ("c", b.QAT_C, "checkpoint_sha256", "tensor SHA-256"),
                                  ("c", b.CONTINUE, "checkpoint_file_sha256", "the SHA-256 of")):
        p = copy.deepcopy(shakedown)
        _unit(p, part, 2, arm)["training"][key] = "0" * 64
        assert b.validate_results(p) == NONCANONICAL  # a well-formed digest: nothing in the JSON contradicts it
        with pytest.raises(ValueError, match=match):
            b.validate_results(p, verify_external=True)
        with pytest.raises(ValueError, match=match):
            b.verify_external_evidence(p)
    # Changed files: one byte of a checkpoint, a checkpoint of another arm, a missing checkpoint.
    copied = tmp_path / "checkpoints"
    shutil.copytree(_work(shakedown) / "checkpoints", copied)
    assert b.verify_external_evidence(shakedown, checkpoint_dir=copied)["checkpoints_trained_in_run_verified"] == 27
    victim, other = copied / b.checkpoint_name("ab", 1, b.MUON), copied / b.checkpoint_name("ab", 1, "qat_identity_full")
    original = victim.read_bytes()
    victim.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="the SHA-256 of"):
        b.verify_external_evidence(shakedown, checkpoint_dir=copied)
    victim.write_bytes(other.read_bytes())
    with pytest.raises(ValueError, match="the SHA-256 of"):
        b.verify_external_evidence(shakedown, checkpoint_dir=copied)
    victim.unlink()
    with pytest.raises(ValueError, match="is missing"):
        b.validate_results(shakedown, checkpoint_dir=copied)
    victim.write_bytes(original)
    (copied / "stray.pt").write_bytes(b"x")
    assert b.verify_external_evidence(shakedown, checkpoint_dir=copied)["unrecorded_checkpoint_files"] == [
        str(copied / "stray.pt")]
    traversal = copy.deepcopy(shakedown)
    _unit(traversal, "ab", 1, b.MUON)["training"]["checkpoint_file"] = "../checkpoints/" + victim.name
    with pytest.raises(ValueError, match="no directory for"):
        b.verify_external_evidence(traversal, checkpoint_dir=copied)


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
def _canonical_data(schema="cycle6_c6_01.v1", **changes):
    """A payload with what ``_validate_data`` reads: a three-hour canonical run that opened test.bin after two hours."""
    splits = {k: {"tokens": c502.CANONICAL_DATA["tokens"][k], "sha256": v} for k, v in q.EXPECTED_SHA256.items()}
    opening = {"git_sha": "a" * 40, "run_config_hash": "c" * 16, "timestamp": "2026-11-27T12:00:00-0700"}
    if schema == b.SCHEMA_VERSION:
        opening["selections_sha256"] = "5" * 64
    access = {"log": b.TEST_ACCESS_LOG_REPO_PATH, "prior_openings": [], "this_opening": opening}
    data = {"source": {"name": "fineweb-edu"}, "encoding": "gpt2", "splits": splits, "directory": "/d",
            "sha256_verified": True, "synthetic": False, "roles": dict(b.EXPECTED_ROLES), "test_access": access}
    data.update(changes)
    return {"schema_version": schema, "data": data, "run_config_hash": "c" * 16, "selections_sha256": "5" * 64,
            "meta": {"git_sha": "a" * 40, "timestamp_start": "2026-11-27T10:00:00-0700",
                     "timestamp_end": "2026-11-27T13:00:00-0700"},
            "stage_seconds": {"stage_a": 2000.0, "stage_b": 2500.0, "stage_c": 2600.0, "test_scoring": 3500.0}}


def test_canonical_data_record_needs_the_pinned_shards_and_the_test_access_log():
    b._validate_data(_canonical_data(), True)
    b._validate_data(_canonical_data(b.SCHEMA_VERSION), True)
    wrong = {k: {"tokens": 1, "sha256": v} for k, v in q.EXPECTED_SHA256.items()}
    other = {**_canonical_data()["data"]["splits"], "test": {"tokens": 2_000_081, "sha256": "0" * 64}}
    access = _canonical_data()["data"]["test_access"]
    this = access["this_opening"]
    for changes, match in (({"sha256_verified": False}, "pinned FineWeb-Edu"), ({"synthetic": True}, "pinned FineWeb-Edu"),
                           ({"splits": other}, "pinned FineWeb-Edu"), ({"splits": wrong}, "token counts"),
                           ({"test_access": None}, "test_access"),
                           ({"test_access": {**access, "log": "elsewhere.jsonl"}}, "test_access"),
                           ({"test_access": {**access, "log": b.SHAKEDOWN_TEST_ACCESS_LOG}}, "test_access"),
                           ({"test_access": {**access, "prior_openings": "none"}}, "test_access"),
                           ({"test_access": {**access, "this_opening": {**this, "git_sha": "b" * 40}}}, "test_access"),
                           ({"test_access": {**access, "this_opening": {**this, "run_config_hash": "d" * 16}}},
                            "test_access"),
                           ({"test_access": {**access, "this_opening": {**this, "run_config_hash": "x"}}}, "test_access"),
                           # The split-role map: test.bin is evaluation data and nothing else.
                           ({"roles": {**b.EXPECTED_ROLES, "training": "test"}}, "data.roles"),
                           ({"roles": {**b.EXPECTED_ROLES, "method_and_arm_selection": "test"}}, "data.roles"),
                           ({"roles": {}}, "data.roles"),
                           # The opening time: inside the run, after stages A-C (7100 s) and before the test scoring.
                           ({"test_access": {**access, "this_opening": {**this, "timestamp": "1900-01-01T00:00:00-0700"}}},
                            "outside the run interval"),
                           ({"test_access": {**access, "this_opening": {**this, "timestamp": "2026-11-27T13:00:02-0700"}}},
                            "outside the run interval"),
                           ({"test_access": {**access, "this_opening": {**this, "timestamp": "2026-11-27T09:59:58-0700"}}},
                            "outside the run interval"),
                           ({"test_access": {**access, "this_opening": {**this, "timestamp": "2026-11-27T11:30:00-0700"}}},
                            "follow stages A-C"),
                           ({"test_access": {**access, "this_opening": {**this, "timestamp": "2026-11-27T12:30:00-0700"}}},
                            "precede the test scoring"),
                           ({"test_access": {**access, "this_opening": {**this, "timestamp": "2026-11-27 12:00"}}},
                            "invalid ISO timestamp"),
                           ({"test_access": {**access, "this_opening": {**this, "selections_sha256": "5" * 64}}},
                            "this_opening keys"),  # a v2 line in the v1 result
                           ({"test_access": {**access, "prior_openings": [{**this, "git_sha": None}]}}, "git_sha"),
                           ({"test_access": {**access, "prior_openings": [
                               {**this, "timestamp": "2026-11-27T12:00:01-0700"}]}}, "later than this opening")):
        with pytest.raises(ValueError, match=match):
            b._validate_data(_canonical_data(**changes), True)
    assert b.EXPECTED_ROLES == {"training": "train", "calibration": "train", "static_scales": "train",
                                "awq_alpha_selection": "val", "method_and_arm_selection": "val", "evaluation": "test"}
    earlier = {**this, "git_sha": "e" * 40, "timestamp": "2026-11-20T08:00:00-0700"}
    b._validate_data(_canonical_data(test_access={**access, "prior_openings": [earlier]}), True)
    # A v2 opening carries the freeze digest of the run.
    v2 = _canonical_data(b.SCHEMA_VERSION)
    v2["data"]["test_access"]["prior_openings"] = [earlier, {**earlier, "selections_sha256": "6" * 64}]
    b._validate_data(v2, True)
    v2["data"]["test_access"]["this_opening"]["selections_sha256"] = "6" * 64
    with pytest.raises(ValueError, match="differs from the freeze digest"):
        b._validate_data(v2, True)
    del v2["data"]["test_access"]["this_opening"]["selections_sha256"]
    with pytest.raises(ValueError, match="this_opening keys"):
        b._validate_data(v2, True)
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
    v2, v1 = b.caveats(p), b.caveats({**p, "schema_version": b.HISTORICAL_SCHEMA_V1})
    # The retained v1 list is the base of the v2 list: one note reworded (L1) and three notes added (M3, L1, H3).
    assert v1 == b._caveats_v1(p) and len(v2) == len(v1) + 3
    assert [i for i, (old, new) in enumerate(zip(v1, v2)) if old != new] == [11]
    assert v1[11].startswith("test.bin is held out from every selection of this run (it is mapped once, after the freeze)")
    assert v2[11].startswith("test.bin is held out from every selection of this run: the scoring view maps it once")
    added = "\n".join(v2[len(v1):])
    assert "Method menus differ in Part B (50m)" in added and "In Part C (120m), the best methods of qat_cooldown" in added
    assert "use the same-method contrasts (*_own_gptq_vs_muon_same_method, *_own_rtn_vs_muon_same_method)" in added
    assert "zero pre-freeze test scoring or selection" in added
    assert "the manifest verification of real_text_data.py hashes and memory-maps test.bin" in added
    assert "Freeze coverage (cycle6_c6_01.v2)" in added and "not the AWQ alphas" in added
    assert not any("Freeze coverage" in note or "Method menus" in note for note in v1)
    notes = "\n".join(v2)
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
    # The preserved result of producer 1b4ea1c: schema v1. Its pre-test freeze digest covers the family, arm and
    # method names only; the 36 AWQ selections are bound by a digest that was pinned after the run.
    assert p["schema_version"] == b.HISTORICAL_SCHEMA_V1 == b._schema_at_commit(p["meta"]["git_sha"])
    assert p["meta"]["git_sha"] == "1b4ea1c17ecc3f6b0e1232693a13ccca25ad4d5d"
    assert "awq_selections" not in p["selections"] and set(p["selections"]) == b.SELECTION_KEYS
    assert p["selections_sha256"] == b.selections_sha256(p["selections"])
    assert len(b.awq_freeze(p["units"])) == 36
    assert b.selections_sha256(b.awq_freeze(p["units"])) == b.HISTORICAL_V1_AWQ_SELECTIONS_SHA256 == \
        "2b8a654c4a356c90e151f439ae3134051d2929c213cce8cce19d636338a43347"
    assert set(p["data"]["test_access"]["this_opening"]) == b.OPENING_KEYS[b.HISTORICAL_SCHEMA_V1]
    assert p["caveats"] == b._caveats_v1(p) == b.caveats(p)  # the retained v1 list, verbatim
    assert p["data"]["roles"] == b.EXPECTED_ROLES
    rep = p["analysis"]["c5_03_reproduction"]
    assert len(rep["rows"]) == 42 and rep["max_abs_nll_difference"] == 0.0 and all(r["abs_diff"] == 0.0 for r in rep["rows"])
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
    rejects(lambda p: _set(p, ("meta", "git_sha"), first), "test_access")  # the access log names the producer

    def producer(sha):  # another producer commit, consistently: only the code provenance can object
        def edit(p):
            p["meta"]["git_sha"] = p["data"]["test_access"]["this_opening"]["git_sha"] = sha
        return edit

    parent = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{canonical['meta']['git_sha']}~1"],
                            capture_output=True, text=True).stdout.strip()
    for sha in (first, parent):  # commits without the C6-01 driver
        rejects(producer(sha), "import closure at meta.git_sha")
    rejects(lambda p: _set(p, ("meta", "code_sha256"), "0" * 64), "committed git blobs")
    rejects(lambda p: p.update(canonical=False, shakedown=True), "not labeled canonical")
    _rejects(canonical, lambda p: p.update(canonical=False, shakedown=True), "preserved canonical result only")
    rejects(lambda p: p.update(schema_version=b.SCHEMA_VERSION), "this_opening keys")  # a v1 result relabeled v2
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


def test_canonical_validator_rejects_the_bypasses_of_the_astra_audit(canonical):
    """Every mutation that the GPT 6.2 Astra audit found certified on the preserved v1 result is rejected.

    The mutations are applied to in-memory copies; the committed file stays byte-pinned. Two bypasses need
    the retained files (a changed checkpoint digest): see ``test_canonical_external_evidence``.
    """
    def rejects(edit, match):
        _rejects(canonical, edit, match, canonical=True)

    tr = lambda p, arm="qat_wide_cooldown", part="ab", seed=1: _unit(p, part, seed, arm)["training"]  # noqa: E731
    # H1: test.bin declared as the training split; a test opening in 1900 or outside the run.
    rejects(lambda p: p["data"]["roles"].update(training="test"), "data.roles")
    rejects(lambda p: p["data"]["roles"].update(awq_alpha_selection="test"), "data.roles")
    opening = lambda p: p["data"]["test_access"]["this_opening"]  # noqa: E731
    rejects(lambda p: opening(p).update(timestamp="1900-01-01T00:00:00-0700"), "outside the run interval")
    rejects(lambda p: opening(p).update(timestamp=_in_window(p, 3600 * 24)), "outside the run interval")
    rejects(lambda p: opening(p).update(timestamp=_in_window(p, 60)), "follow stages A-C")
    rejects(lambda p: opening(p).update(timestamp=p["meta"]["timestamp_end"]), "precede the test scoring")
    rejects(lambda p: opening(p).update(selections_sha256=p["selections_sha256"]), "this_opening keys")
    # H2: one step-272 branch hash or one batch-stream fingerprint that its siblings contradict.
    rejects(lambda p: tr(p)["branch"].update(snapshot_sha256="0" * 64), "sibling branch records differ")
    rejects(lambda p: tr(p, "qat_identity_cooldown", seed=3)["branch"].update(
        snapshot_sha256=tr(p, "qat_identity_cooldown", seed=2)["branch"]["snapshot_sha256"]), "sibling branch records differ")
    rejects(lambda p: tr(p)["branch"].update(optimizer_snapshot_sha256="0" * 64), "must branch from")  # not a v1 field
    rejects(lambda p: tr(p).update(stream_fingerprint="0" * 16), "stream_fingerprint differs")
    rejects(lambda p: tr(p, b.QAT_C, "c", 2).update(stream_fingerprint=tr(p, b.QAT_C, "c", 3)["stream_fingerprint"]),
            "stream_fingerprint differs")
    rejects(lambda p: tr(p).update(init_fingerprint="0" * 16), "do not share one init_fingerprint")
    rejects(lambda p: tr(p, b.CONTINUE, "c").update(init_fingerprint="0" * 16), "init_fingerprint must be")
    for seed in (1, 2, 3):  # the recorded fingerprints are those of the seeds
        assert tr(canonical, seed=seed)["stream_fingerprint"] == b._expected_stream_fingerprint(26_004_518, "50m", seed)
        assert len({tr(canonical, arm, seed=seed)["branch"]["snapshot_sha256"] for arm in b.QAT_ARMS
                    if arm != "qat_identity_full"}) == 1
    # M1: a training curve with a negative loss at step 10,000, inverted activation extrema, negative CUPTI counts.
    curve = lambda p: tr(p, b.MUON)["train_curve"]  # noqa: E731
    rejects(lambda p: curve(p).__setitem__(1, [10_000, -1.0]), "train_curve must hold")
    rejects(lambda p: curve(p)[1].__setitem__(0, 10_000), "train_curve must hold")
    rejects(lambda p: curve(p)[1].__setitem__(1, -1.0), "train_curve must hold")
    rejects(lambda p: curve(p).pop(5), "train_curve steps")

    def invert(p):
        a = _unit(p, "ab", 1, b.MUON)["activation_statistics"]["identity"]["excess_kurtosis"]["all"]
        a["min"], a["max"] = a["max"], a["min"]

    rejects(invert, "summary bounds violated")
    emulated = lambda p: _grid_row(_unit(p, "ab", 1, b.MUON), "kron_hadamard_full_gptq", b.EMULATED_MODE)  # noqa: E731
    assert emulated(canonical)["cupti"] == {"fp4_gemm_kernels": 0, "non_fp4_gemm_kernels": 193, "nvfp4_amax_kernels": 60,
                                           "nvfp4_quant_kernels": 60}
    for key in rq.CUPTI_KEYS:
        rejects(lambda p, key=key: emulated(p)["cupti"].__setitem__(key, -1), "integers >= 0")
    rejects(lambda p: emulated(p)["cupti"].update(non_fp4_gemm_kernels=-193, nvfp4_quant_kernels=-60), "integers >= 0")
    rejects(lambda p: emulated(p)["cupti"].update(nvfp4_quant_kernels=0), "emulation must run no FP4 GEMM")
    rejects(lambda p: emulated(p)["cupti"].update(non_fp4_gemm_kernels=72), "emulation must run no FP4 GEMM")
    # H3: a different AWQ winner. The v1 freeze digest does not cover it (it stays valid), the pinned digest does.
    def other_winner(p):
        _other_awq_winner(p)
        assert p["selections_sha256"] == canonical["selections_sha256"] == b.selections_sha256(p["selections"])

    rejects(other_winner, "pinned selections of the preserved cycle6_c6_01.v1 result")
    rejects(lambda p: _other_awq_winner(p, "ab", 2, "qat_wide_dual_cooldown"), "pinned selections")
    rejects(lambda p: _other_awq_winner(p, "c", 3, b.QAT_C), "pinned selections")

    def other_scores(p):  # the same winner, other val.bin scores of the losing alphas
        a = _unit(p, "ab", 1, b.MUON)["awq_selection"][0]
        worst = max(range(3), key=a["val_nll_hw_dynamic"].__getitem__)
        a["val_nll_hw_dynamic"][worst] += 0.5

    rejects(other_scores, "pinned selections")

    def other_weights(p):
        r = _grid_row(_unit(p, "ab", 1, b.MUON), b.C5_BEST, "w4a16_fakequant")
        r["effective_weights_sha256"] = r["effective_weights_sha256_test"] = "0" * 64

    rejects(other_weights, "pinned selections")
    rejects(lambda p: p["selections"].update(awq_selections=b.awq_freeze(p["units"])), "selections keys or values")
    # M2: a 0.1-nat discrepancy in a re-measured C5-03 comparator, with every derived field recomputed.
    def comparator(method, mode, split, nats=0.1):
        def edit(p):
            u = _unit(p, "c", 1, b.BASELINE)
            _shift_nll(u, _grid_row(u, method, mode), split, nats)
            _recompute(p)
            rep = p["analysis"]["c5_03_reproduction"]
            assert rep["max_abs_nll_difference"] == pytest.approx(abs(nats), abs=1e-9)
            assert p["caveats"] == b.caveats(p) and f"{rep['max_abs_nll_difference']:.2e} nats" in p["caveats"][-1]
        return edit

    rejects(comparator(b.C5_HADAMARD, "w4a4_hw_static", "test"),
            "C5-03 comparator seed 1 hadamard_b16_gptq/w4a4_hw_static test")
    rejects(comparator(b.C5_BEST, "w4a16_fakequant", "test"), "C5-03 comparator")
    rejects(comparator(b.C5_HADAMARD, "w4a4_hw_static", "test", 1e-12), "exact reproduction is required")

    def bf16(p):  # the unquantized baseline NLL itself
        u = _unit(p, "c", 2, b.BASELINE)
        u["baseline_test"]["nll"] += 0.1
        u["baseline_test"]["ppl"] = math.exp(u["baseline_test"]["nll"])
        for r in u["grid"]:
            _shift_nll(u, r, "test", 0.1)
        _recompute(p)

    rejects(bf16, "C5-03 comparator seed 2 bf16 test")
    rejects(lambda p: _grid_row(_unit(p, "c", 1, b.BASELINE), b.C5_BEST, b.DYNAMIC).update(awq_alpha=0.5),
            "not the selected candidate|AWQ alpha differs")

    def record_only(p):  # the comparison record alone
        p["analysis"]["c5_03_reproduction"]["rows"][0]["abs_diff"] = 0.1

    rejects(record_only, "analysis")


def test_canonical_external_evidence(canonical, tmp_path):
    """The preserved result against the retained files: 24 C6 checkpoints, 3 C5-03 checkpoints, the test-access log."""
    dirs = {"checkpoint_dir": b._artifact_dir("c6_01_checkpoints"), "baseline_dir": b.default_baseline_dir(),
            "evidence_dir": b._artifact_dir("c6_01_run_evidence")}
    if not all(d.is_dir() for d in dirs.values()):
        pytest.skip("the retained C6-01 checkpoints and run evidence are not on this host")
    log_sha256 = hashlib.sha256((dirs["evidence_dir"] / b.TEST_ACCESS_LOG_NAME).read_bytes()).hexdigest()
    receipt = b.validate_results(canonical, canonical=True, verify_external=True)
    assert {k: v for k, v in receipt.items() if k != "external_evidence"} == CERTIFIED
    evidence = receipt["external_evidence"]
    assert evidence["checkpoints_trained_in_run_verified"] == 24 and evidence["checkpoints_c5_03_preserved_verified"] == 3
    assert evidence["checkpoint_bytes_hashed"] == 7_596_634_404 + 1_483_136_331
    assert evidence["unrecorded_checkpoint_files"] == [] and evidence["test_access_log_verified"]
    assert evidence["test_access_log_entries"] >= 1 and evidence["test_access_log_sha256"] == log_sha256
    # A changed digest of a new C6 checkpoint (tensor or file) is well-formed hex: only the file can refute it.
    for part, arm, key, match in (("ab", b.MUON, "checkpoint_sha256", "tensor SHA-256"),
                                  ("ab", b.MUON, "checkpoint_file_sha256", "the SHA-256 of")):
        p = copy.deepcopy(canonical)
        _unit(p, part, 1, arm)["training"][key] = "0" * 64
        assert b.validate_results(p, canonical=True) == CERTIFIED
        with pytest.raises(ValueError, match=match):
            b.validate_results(p, canonical=True, **dirs)
        with pytest.raises(ValueError, match=match):
            b.verify_external_evidence(p, **dirs)
    # The log: the opening in the JSON must be the line on disk.
    p = copy.deepcopy(canonical)
    p["data"]["test_access"]["this_opening"]["timestamp"] = _in_window(p, 9967)  # one second off, inside the run
    assert b.validate_results(p, canonical=True) == CERTIFIED
    with pytest.raises(ValueError, match="differs from the external test-access log"):
        b.verify_external_evidence(p, **dirs)
    with pytest.raises(ValueError, match="is missing"):
        b.verify_external_evidence(canonical, **{**dirs, "evidence_dir": tmp_path})
    with pytest.raises(ValueError, match="is missing"):
        b.verify_external_evidence(canonical, **{**dirs, "checkpoint_dir": tmp_path})
    assert hashlib.sha256((dirs["evidence_dir"] / b.TEST_ACCESS_LOG_NAME).read_bytes()).hexdigest() == log_sha256
