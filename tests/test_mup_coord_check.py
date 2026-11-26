"""Tests for the muP width-transfer coordinate check (CPU only).

``coord`` runs the real coordinate check once, in worker processes: AdamW on the default widths
with six seeds, Muon on the default widths with seed 0. It takes about two minutes on 8 cores.
"""

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from research.polar_opt import baselines
from research.polar_opt import mup_coord_check as mcc
from research.polar_opt.mup_coord_check import (
    FAIL_BETA,
    GATE_STEPS,
    PASS_BETA,
    PROBE_STEPS,
    WIDTHS,
    ModelConfig,
    Variant,
    build_model,
    build_optimizer,
    gate,
    geometric_mean,
    loglog_slope,
    make_tokens,
    max_over_min,
    orthogonalize,
    parameter_specs,
    run_trial,
    summarize_coord,
    summarize_lr_transfer,
    svd_exact,
    width_scaling,
)
from research.polar_opt.schatten_switch_blackwell import (
    SCHATTEN_SWITCH_CIRCUITS,
    schatten_switch_reference,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results" / "cycle3_mup_coord_check_cpu.json"
JOBS = max(1, min(8, (os.cpu_count() or 2) // 2))
SS3 = sorted(SCHATTEN_SWITCH_CIRCUITS)


def strict_json(text: str) -> dict:
    def reject(token):
        raise ValueError(f"non-finite JSON constant {token}")
    return json.loads(text, parse_constant=reject)


def with_spectrum(singular_values, columns, seed=0):
    """FP64 matrix ``U diag(s) V^T`` and its factors."""
    generator = torch.Generator().manual_seed(seed)
    s = torch.as_tensor(singular_values, dtype=torch.float64)
    U = torch.linalg.qr(torch.randn(len(s), len(s), generator=generator, dtype=torch.float64)).Q
    V = torch.linalg.qr(torch.randn(columns, len(s), generator=generator, dtype=torch.float64)).Q
    return (U * s) @ V.mT, U, s, V


def rel(a, reference):
    return ((a.double() - reference.double()).norm() / reference.double().norm()).item()


# --------------------------------------------------------------------------------------
# Parameterization
# --------------------------------------------------------------------------------------
CONFIGS = [
    pytest.param(dict(width=256), id="default"),
    pytest.param(dict(width=128), id="base-width"),
    pytest.param(dict(width=192, d_base=64, mlp_mult=3, head_dim=16, layers=3), id="non-default"),
]


@pytest.mark.parametrize("kwargs", CONFIGS)
def test_mup_table(kwargs):
    config = ModelConfig(parameterization="mup", **kwargs)
    d, d_base, hidden = config.width, config.d_base, config.mlp_mult * config.width
    model = build_model(config, seed=0)
    specs = parameter_specs(model)
    assert len([s for s in specs.values() if s.kind == "hidden"]) == 7 * config.layers
    assert len([s for s in specs.values() if s.kind == "norm"]) == 2 * config.layers + 1

    assert config.logit_scale == d_base / d
    assert config.attention_scale == 1 / config.head_dim
    assert config.heads == d // config.head_dim

    emb = specs["emb.weight"]
    assert (emb.kind, emb.init_std, emb.lr_mult) == ("embedding", 1 / math.sqrt(d_base), 1.0)
    head = specs["head.weight"]
    assert (head.kind, head.init_std, head.lr_mult) == ("head", 0.0, 1.0)
    assert model.head.weight.shape == (config.vocab_size, d)
    assert not model.head.weight.any()  # untied, zero
    assert model.head.weight is not model.emb.weight

    shapes = {"q": (d, d), "k": (d, d), "v": (d, d), "o": (d, d),
              "gate": (hidden, d), "up": (hidden, d), "down": (d, hidden)}
    for name, spec in specs.items():
        p = model.get_parameter(name)
        if spec.kind == "norm":
            assert torch.equal(p, torch.ones_like(p)) and spec.lr_mult == 1.0
            continue
        if spec.kind == "hidden":
            fan_out, fan_in = shapes[name.split(".")[-2]]
            assert p.shape == (fan_out, fan_in)
            assert spec.init_std == 1 / math.sqrt(fan_in)
            assert spec.lr_mult == d_base / fan_in
            assert spec.muon_scale == math.sqrt(fan_out / fan_in)
        if spec.init_std:
            assert p.std().item() == pytest.approx(spec.init_std, rel=0.05), name
            assert abs(p.mean().item()) < 0.05 * spec.init_std


@pytest.mark.parametrize("kwargs", CONFIGS)
def test_sp_table(kwargs):
    config = ModelConfig(parameterization="sp", **kwargs)
    model = build_model(config, seed=0)
    assert config.logit_scale == 1.0
    assert config.attention_scale == 1 / math.sqrt(config.head_dim)
    for name, spec in parameter_specs(model).items():
        assert spec.lr_mult == 1.0
        if spec.kind == "norm":
            continue
        assert spec.init_std == 0.02
        assert model.get_parameter(name).std().item() == pytest.approx(0.02, rel=0.05), name
        if spec.kind == "hidden":
            assert spec.muon_scale == 1.0


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"d_base": 0},
        {"mlp_mult": 0},
        {"layers": 0},
        {"num_batches": 0},
        {"width": 0},
        {"head_dim": 0},
        {"vocab_size": -1},
        {"seq_len": 0},
        {"batch_size": 0},
    ],
)
def test_model_config_rejects_nonpositive_fields(bad_kwargs):
    with pytest.raises(ValueError, match="positive"):
        ModelConfig(**bad_kwargs)


def test_readout_lr_power_is_the_task_text_control():
    config = ModelConfig(width=512, d_base=128, readout_lr_power=1.0)
    assert parameter_specs(build_model(config, 0))["head.weight"].lr_mult == 128 / 512
    sp = ModelConfig(width=512, parameterization="sp", readout_lr_power=1.0)
    assert parameter_specs(build_model(sp, 0))["head.weight"].lr_mult == 1.0


@pytest.mark.parametrize("kwargs", CONFIGS)
@pytest.mark.parametrize("parameterization", ["sp", "mup"])
def test_optimizer_learning_rates(parameterization, kwargs):
    config = ModelConfig(parameterization=parameterization, **kwargs)
    model = build_model(config, seed=0)
    specs = parameter_specs(model)
    eta, eta_aux = 0.01, 0.002

    adamw = build_optimizer(model, "adamw", lr=eta)
    rates = {group["name"]: group["lr"] for group in adamw.param_groups}
    assert set(rates) == set(specs)
    for name, spec in specs.items():
        assert rates[name] == pytest.approx(eta * spec.lr_mult)
    if config.mup:
        assert rates["blocks.0.attn.q.weight"] == pytest.approx(eta * config.d_base / config.width)
        assert rates["blocks.0.mlp.down.weight"] == pytest.approx(
            eta * config.d_base / (config.mlp_mult * config.width))
        assert rates["emb.weight"] == rates["norm.weight"] == rates["head.weight"] == eta

    muon = build_optimizer(model, "muon", lr=eta, lr_aux=eta_aux, polar="svd_exact")
    aux = {group["name"]: group["lr"] for group in muon.auxiliary.param_groups}
    assert set(aux) | set(muon.hidden) == set(specs) and not set(aux) & set(muon.hidden)
    assert all(specs[name].kind == "hidden" for name in muon.hidden)
    assert aux["emb.weight"] == aux["norm.weight"] == aux["head.weight"] == eta_aux


def test_logit_and_attention_scale_are_applied():
    config = ModelConfig(width=256, d_base=64, head_dim=16)
    model = build_model(config, seed=0)
    with torch.no_grad():
        model.head.weight.normal_(generator=torch.Generator().manual_seed(1))
    tokens = make_tokens(config, 0)[0, :, :-1]
    kept = {}
    model.norm.register_forward_hook(lambda _m, _i, out: kept.setdefault("features", out))
    logits = model(tokens)
    assert torch.allclose(logits, kept["features"] @ model.head.weight.mT * (64 / 256), atol=1e-6)

    attn = model.blocks[0].attn
    x = torch.randn(2, config.seq_len, config.width, generator=torch.Generator().manual_seed(2))
    split = lambda t: t.view(2, config.seq_len, config.heads, config.head_dim).transpose(1, 2)
    q = mcc._rope(split(attn.q(x)), model.rope_cos, model.rope_sin)
    k = mcc._rope(split(attn.k(x)), model.rope_cos, model.rope_sin)
    mixed = F.scaled_dot_product_attention(q, k, split(attn.v(x)), is_causal=True, scale=1 / 16)
    expected = attn.o(mixed.transpose(1, 2).reshape(x.shape))
    assert torch.allclose(attn(x, model.rope_cos, model.rope_sin), expected, atol=1e-5)


def test_model_is_causal():
    config = ModelConfig(width=128, parameterization="sp")
    model = build_model(config, seed=0)
    tokens = make_tokens(config, 0)[0, :2, :-1]
    changed = tokens.clone()
    changed[:, 20] = (changed[:, 20] + 1) % config.vocab_size
    a, b = model(tokens), model(changed)
    assert torch.allclose(a[:, :20], b[:, :20], atol=1e-6)
    assert (a[:, 20:] - b[:, 20:]).abs().amax(dim=(0, 2)).min() > 1e-4


@pytest.mark.parametrize("parameterization, scales", [
    ("mup", {"q": 1.0, "gate": math.sqrt(2), "down": math.sqrt(0.5)}),
    ("sp", {"q": 1.0, "gate": 1.0, "down": 1.0}),
])
def test_muon_update_is_spectrally_scaled(parameterization, scales):
    """First step: ``dW = -eta * sqrt(d_out / d_in) * polar(g)`` for muP, ``-eta * polar(g)`` for SP."""
    model = build_model(ModelConfig(width=64, d_base=64, parameterization=parameterization), seed=0)
    generator = torch.Generator().manual_seed(3)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=generator)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    build_optimizer(model, "muon", lr=0.05, lr_aux=0.001, polar="svd_exact").step()
    for key, scale in scales.items():
        name = f"blocks.1.{'mlp' if key in ('gate', 'down') else 'attn'}.{key}.weight"
        p = model.get_parameter(name)
        delta = p.detach() - before[name]
        expected = -0.05 * scale * svd_exact(p.grad).float()
        assert torch.allclose(delta, expected, atol=1e-6), name
        singular = torch.linalg.svdvals(delta.double())
        assert singular.max().item() == pytest.approx(0.05 * scale, rel=1e-4)
        assert singular.min().item() == pytest.approx(0.05 * scale, rel=1e-4)


def test_muon_ignores_zero_gradients():
    """muP step 0: the zero readout gives zero hidden gradients and the weights must not move."""
    config = ModelConfig(width=128)
    model = build_model(config, seed=0)
    batch = make_tokens(config, 0)[0]
    mcc._loss(model, batch).backward()
    assert not model.blocks[0].attn.q.weight.grad.any()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    for polar in ("svd_exact", "polar_express", "PS3_55", "SS3_54"):
        build_optimizer(model, "muon", lr=0.02, lr_aux=0.01, polar=polar).step()
        for name, p in model.named_parameters():
            if name != "head.weight":
                assert torch.equal(p, before[name]), (polar, name)
            assert torch.isfinite(p).all()


# --------------------------------------------------------------------------------------
# Orthogonalizers on the CPU
# --------------------------------------------------------------------------------------
def test_svd_exact():
    G = torch.randn(48, 80, generator=torch.Generator().manual_seed(0))
    Q = svd_exact(G)
    assert Q.dtype == torch.float64
    assert torch.allclose(Q @ Q.mT, torch.eye(48, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(svd_exact(G.mT), Q.mT, atol=1e-12)
    assert not svd_exact(torch.zeros(5, 7)).any()

    # Rank 6 in FP32: the polar factor is a partial isometry, not rounding noise at unit scale.
    low_rank = (torch.randn(40, 6, generator=torch.Generator().manual_seed(1))
                @ torch.randn(6, 64, generator=torch.Generator().manual_seed(2)))
    singular = torch.linalg.svdvals(svd_exact(low_rank))
    assert torch.allclose(singular[:6], torch.ones(6, dtype=torch.float64), atol=1e-12)
    assert singular[6:].max() < 1e-12
    assert torch.linalg.svdvals(svd_exact(low_rank, rtol=0.0)).min() > 0.99


def test_svd_exact_preserves_genuine_small_singular_values_at_large_width():
    sv = np.concatenate([np.geomspace(1.0, 5e-5, 256), np.zeros(256)])
    G, _, _, _ = with_spectrum(sv, 512, seed=7)
    G32 = G.float()
    singular = torch.linalg.svdvals(svd_exact(G32))
    assert torch.allclose(singular[:256], torch.ones(256, dtype=torch.float64), atol=1e-4)
    assert singular[256:].max() < 1e-12


@pytest.mark.parametrize("method", mcc.POLAR_METHODS)
def test_orthogonalize_is_zero_safe_and_keeps_dtype(method):
    G = torch.randn(24, 40, generator=torch.Generator().manual_seed(0))
    with_nan = G.clone()
    with_nan[0, 0] = float("nan")
    for bad in (torch.zeros_like(G), with_nan, G * float("inf")):
        out = orthogonalize(bad, method)
        assert out.shape == G.shape and not out.any()
    for shape in ((24, 40), (40, 24), (32, 32)):
        G = torch.randn(*shape, generator=torch.Generator().manual_seed(1))
        out = orthogonalize(G, method)
        assert out.dtype == torch.float32 and out.shape == G.shape
        assert rel(out, svd_exact(G)) < 0.2
    with pytest.raises(ValueError):
        orthogonalize(G, "newton_schulz")
    with pytest.raises(ValueError):
        orthogonalize(G[None], method)


def _polar_express_scalar(x):
    for a, b, c in baselines._take_steps(baselines._AMSEL_COEFFICIENTS, 5):
        x = a * x + b * x ** 3 + c * x ** 5
    return x


def _switch_scalar(x):
    for residual, rows in baselines.PS3_55:
        work = {0: x * x}
        for dst, src, add, a, b, c in rows:
            work[dst] = a * work[src] ** 2 + b * work[add] + c
        x = x * work[dst] + residual * x
    return x


@pytest.mark.parametrize("columns", [40, 96])
@pytest.mark.parametrize("method, scalar, scale", [
    ("polar_express", _polar_express_scalar, lambda norm: 1.01 * norm + 1e-7),
    ("PS3_55", _switch_scalar, lambda norm: norm + 1e-7),
])
def test_fp64_emulation_is_the_baseline_polynomial(method, scalar, scale, columns):
    """In FP64 the emulation applies the imported coefficients to every singular value."""
    G, U, s, V = with_spectrum(np.geomspace(1.0, 1e-3, 40), columns)
    for matrix, left, right in ((G, U, V), (G.mT, V, U)):
        out = orthogonalize(matrix, method, torch.float64)
        core = left.mT @ out @ right
        expected = scalar((s / scale(G.norm().item())).numpy())
        assert np.abs(torch.diagonal(core).numpy() - expected).max() < 1e-9
        assert (core - torch.diag(torch.diagonal(core))).abs().max() < 1e-9


@pytest.mark.parametrize("shape", [(48, 80), (80, 48), (64, 64)])
def test_low_precision_emulation_matches_the_baselines(shape):
    """Native FP16 / BF16 on the CPU (slow) against FP32 accumulation with one rounding."""
    G = torch.randn(*shape, generator=torch.Generator().manual_seed(0))
    assert rel(orthogonalize(G, "PS3_55"), baselines.polar_switch(G, "PS3_55")) < 0.02
    assert rel(orthogonalize(G, "polar_express"), baselines.polar_express(G).float()) < 0.1
    for method in ("PS3_55", "polar_express"):
        assert 0 < rel(orthogonalize(G, method), orthogonalize(G, method, torch.float64)) < 0.05


@pytest.mark.parametrize("name", SS3)
def test_schatten_switch_reference_runs_on_cpu(name):
    """The imported reference and circuit table, on CPU tensors, against the scalar map."""
    circuit = SCHATTEN_SWITCH_CIRCUITS[name]
    G, U, s, V = with_spectrum(np.geomspace(1.0, 0.02, 40), 72)
    for matrix, left, right in ((G, U, V), (G.mT, V, U)):
        out, info = schatten_switch_reference(matrix, name, dtype=torch.float64, return_info=True)
        assert out.device.type == "cpu"
        core = left.mT @ out @ right
        expected = circuit.scalar((s / (info["sigma_hat"][0] * G.norm())).numpy())
        assert np.abs(torch.diagonal(core).numpy() - expected).max() < 1e-6
        assert (core - torch.diag(torch.diagonal(core))).abs().max() < 1e-6

        single = matrix.float()
        emulated = orthogonalize(single, name)  # FP16 operands, FP32 accumulation
        assert torch.equal(emulated, schatten_switch_reference(single, name, dtype=torch.float16))
        assert 0 < rel(emulated, out) < 0.02
        assert torch.equal(orthogonalize(single, name, torch.float64),
                           schatten_switch_reference(single, name, dtype=torch.float64))


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("exponent", [0.0, 0.7, -1.0, 2.5])
def test_loglog_slope_recovers_a_power_law(exponent):
    values = [3.0 * w ** exponent for w in WIDTHS]
    assert loglog_slope(WIDTHS, values) == pytest.approx(exponent, abs=1e-12)
    assert max_over_min(values) == pytest.approx(8.0 ** abs(exponent))


def test_loglog_slope_is_ordinary_least_squares():
    widths, values = [128, 256, 512, 1024], [1.0, 2.5, 3.0, 9.0]
    expected = np.polyfit(np.log(widths), np.log(values), 1)[0]
    assert loglog_slope(widths, values) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("bad", [0.0, 1e-30, 1e-31, -1.0, float("nan"), float("inf"), None])
def test_statistics_reject_tiny_and_non_finite_values(bad):
    values = [1.0, bad, 4.0, 8.0]
    assert loglog_slope(WIDTHS, values) is None
    assert max_over_min(values) is None
    assert geometric_mean(values) is None
    assert loglog_slope([128], [1.0]) is None and loglog_slope([128, 128], [1.0, 2.0]) is None
    assert loglog_slope(WIDTHS, [1.0, 2.0]) is None


def _synthetic_trials(exponents: dict, seeds=(0, 1)):
    """Trials whose ``delta_rms`` is ``scale(seed) * width ** exponent`` at every step after 0."""
    trials = []
    for width in WIDTHS:
        for seed in seeds:
            stats = {name: [0.0] + [(1 + seed) * width ** e] * (len(PROBE_STEPS) - 1)
                     for name, e in exponents.items()}
            trials.append({"width": width, "seed": seed, "delta_rms": stats})
    return trials


def test_width_scaling_and_gate_on_synthetic_trials():
    scaling = width_scaling(_synthetic_trials({"emb": 0.05, "logits": -0.1}), "delta_rms")
    assert scaling["emb"]["0"] == {"by_width": [None] * 4, "beta": None, "max_over_min_ratio": None}
    assert scaling["emb"]["4"]["beta"] == pytest.approx(0.05)
    assert scaling["emb"]["4"]["by_width"][0] == pytest.approx(math.sqrt(2) * 128 ** 0.05)  # seed geometric mean
    assert scaling["logits"]["12"]["max_over_min_ratio"] == pytest.approx(8 ** 0.1)

    passed = gate(scaling)
    assert passed["passed"] and not passed["fails_clearly"]
    assert passed["max_abs_beta"] == pytest.approx(0.1) and passed["worst"]["probe"] == "logits"

    between = gate(width_scaling(_synthetic_trials({"emb": 0.0, "logits": 0.2}), "delta_rms"))
    assert not between["passed"] and not between["fails_clearly"]
    failed = gate(width_scaling(_synthetic_trials({"emb": 0.0, "logits": -1.0}), "delta_rms"))
    assert not failed["passed"] and failed["fails_clearly"]
    assert (PASS_BETA, FAIL_BETA) == (0.15, 0.25)

    # A zero change at a gate step is a missing beta: neither a pass nor a NaN.
    trials = _synthetic_trials({"emb": 0.0})
    trials[0]["delta_rms"]["emb"][PROBE_STEPS.index(8)] = 0.0
    missing = gate(width_scaling(trials, "delta_rms"))
    assert not missing["passed"] and missing["fails_clearly"] and missing["missing_betas"] == 1


def test_lr_transfer_summary():
    grid = [0.005, 0.01, 0.02, 0.04, 0.08]
    optimum = {128: 0.02, 256: 0.025, 512: 0.07}
    trials = [{"width": w, "lr": lr, "seed": s, "final_loss": 1 + s + math.log(lr / best) ** 2}
              for w, best in optimum.items() for lr in grid for s in (0, 1)]
    for trial in trials:
        if trial["width"] == 512 and trial["lr"] == 0.08 and trial["seed"] == 1:
            trial["final_loss"] = None  # one diverged seed removes that learning rate
    out = summarize_lr_transfer(trials, d_base=128)
    assert out["best_lr"] == {"128": 0.02, "256": 0.02, "512": 0.04}
    assert out["best_lr_interpolated"]["128"] == pytest.approx(0.02)
    assert out["best_lr_interpolated"]["256"] == pytest.approx(0.025, rel=1e-9)  # a parabola in log(lr)
    assert out["final_loss"]["512"][-1] == math.inf
    assert out["best_lr_interpolated"]["512"] == 0.04  # no finite right neighbour
    assert not out["same_best_lr"] and out["best_lr_log2_shift"] == pytest.approx(1.0)
    assert out["best_at_grid_edge"] == {"128": False, "256": False, "512": False}
    assert out["regret_at_base_best_lr"]["128"] == 0.0
    assert out["regret_at_base_best_lr"]["512"] == pytest.approx(math.log(0.02 / 0.07) ** 2 - math.log(0.04 / 0.07) ** 2)


def test_summarize_lr_transfer_all_diverged():
    trials = [
        {"width": w, "lr": lr, "seed": s, "final_loss": None, "eval_loss": None}
        for w in (128, 256)
        for lr in (0.01, 0.04)
        for s in (0, 1)
    ]
    out = summarize_lr_transfer(trials, d_base=128)
    assert out["same_best_lr"] is False
    assert out["best_lr"] == {"128": None, "256": None}
    assert out["best_loss"] == {"128": None, "256": None}
    assert out["best_lr_log2_shift"] is None
    assert out["best_lr_interpolated_log2_shift"] is None
    assert out["regret_at_base_best_lr"] == {"128": None, "256": None}
    text = json.dumps(mcc._clean(out), allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


# --------------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------------
def test_runs_are_deterministic():
    config = ModelConfig(width=128)
    kwargs = dict(optimizer="muon", lr=0.02, lr_aux=0.01, polar="SS3_53", seed=3, steps=4, probe_steps=(0, 1, 2, 4))
    first, second = run_trial(config, **kwargs), run_trial(config, **kwargs)
    assert first == second
    assert first["delta_rms"]["logits"][1] > 0 and len(first["losses"]) == 4
    other = run_trial(config, **{**kwargs, "seed": 4})
    assert other["losses"] != first["losses"]

    # The data depends on the seed only; the initial values on the seed and the shape.
    wide = ModelConfig(width=512)
    assert torch.equal(make_tokens(config, 3), make_tokens(wide, 3))
    assert not torch.equal(make_tokens(config, 3), make_tokens(config, 4))
    assert make_tokens(config, 3).shape == (4, 8, 33)
    a, b = build_model(wide, 3), build_model(wide, 3)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert not torch.equal(a.emb.weight, build_model(wide, 4).emb.weight)


def test_make_tokens_val_split_is_disjoint_same_chain_and_eval_loss_recorded():
    config = ModelConfig(width=64, d_base=64, head_dim=16)
    train_t = make_tokens(config, 2, split="train")
    val_t = make_tokens(config, 2, split="val")
    assert train_t.shape == val_t.shape
    assert not torch.equal(train_t, val_t)
    gen = torch.Generator().manual_seed(1_000_003 * 3)
    successors = torch.randint(config.vocab_size, (config.vocab_size, 4), generator=gen)
    # Every transition in val_t comes from the exact same transition table as train_t.
    nxt = successors[val_t[:, :, :-1]]
    assert (nxt == val_t[:, :, 1:].unsqueeze(-1)).any(dim=-1).all()
    trial = run_trial(config, optimizer="muon", lr=0.02, lr_aux=0.01, polar="SS3_54", seed=2, steps=4, probe_steps=(0, 4))
    assert trial["eval_loss"] is not None and math.isfinite(trial["eval_loss"])
    assert trial["final_loss"] is not None and trial["eval_loss"] != trial["final_loss"]


def test_probe_statistics():
    config = ModelConfig(width=128, parameterization="sp")
    trial = run_trial(config, optimizer="adamw", lr=0.003, seed=0, steps=4, probe_steps=(0, 1, 2, 4))
    assert list(trial["delta_rms"]) == ["emb", "block0_attn_out", "block0_mlp_out", "block1_attn_out",
                                        "block1_mlp_out", "resid_final", "logits"]
    for name in trial["delta_rms"]:
        assert trial["delta_rms"][name][0] == 0.0 and trial["step_delta_rms"][name][0] is None
        # One step after step 0, the two deltas are the same quantity.
        assert trial["step_delta_rms"][name][1] == trial["delta_rms"][name][1] > 0
        assert trial["step_delta_rms"][name][3] < trial["delta_rms"][name][3]
        assert all(value > 0 for value in trial["activation_rms"][name])
    assert trial["activation_rms"]["emb"][0] == pytest.approx(0.02, rel=0.05)

    updates = trial["updates"]
    assert len(updates) == 2 * 7 + 5 + 2
    assert set(updates["blocks.0.attn.q"]) == {"update_rms", "update_spectral"}
    assert set(updates["blocks.0.attn_norm"]) == {"update_rms"}
    assert updates["emb"]["update_rms"][0] is None and updates["emb"]["update_rms"][1] > 0
    # The first Adam step moves every entry of a dense gradient by lr.
    assert updates["head"]["update_rms"][1] == pytest.approx(0.003, rel=1e-3)
    assert updates["head"]["update_spectral"][1] <= 0.003 * math.sqrt(256 * 128) * (1 + 1e-6)

    muon = run_trial(config, optimizer="muon", lr=0.02, lr_aux=0.01, seed=0, steps=2, probe_steps=(0, 1, 2))
    lift = muon["updates"]["blocks.1.mlp.gate"]
    assert lift["polar_mean_sq_sv"][1] == pytest.approx(1.0, abs=1e-3)  # 256 tokens, 128 columns: full rank
    assert lift["update_spectral"][1] == pytest.approx(0.02, rel=1e-3)


def test_mup_first_step_moves_only_the_readout():
    trial = run_trial(ModelConfig(width=128), optimizer="adamw", lr=0.003, seed=0, steps=2, probe_steps=(0, 1, 2))
    assert trial["delta_rms"]["logits"][1] > 0
    for name in ("emb", "block0_attn_out", "block1_mlp_out", "resid_final"):
        assert trial["delta_rms"][name][1] == 0.0 and trial["delta_rms"][name][2] > 0
    assert trial["losses"][0] == pytest.approx(math.log(256), abs=1e-5)  # zero readout: uniform prediction


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_non_finite_gradients_stop_the_run(optimizer):
    """A learning rate of 1e18 overflows FP32: no update with a non-finite gradient, no NaN in the metrics."""
    trials = []
    for width in (128, 256):
        trial = run_trial(ModelConfig(width=width, parameterization="sp"), optimizer=optimizer, lr=1e18,
                          lr_aux=1e18, seed=0, steps=12, probe_steps=PROBE_STEPS)
        assert trial["nonfinite_step"] == 1 and trial["final_loss"] is None
        assert len(trial["losses"]) == 1 and math.isfinite(trial["losses"][0])
        assert trial["delta_rms"]["logits"][2:] == [None] * 4
        trials.append(trial)

    summary = summarize_coord(trials)
    assert summary["nonfinite_trials"] == 2
    assert not summary["gate"]["passed"] and summary["gate"]["fails_clearly"]
    assert summary["gate"]["missing_betas"] == 7 * len(GATE_STEPS)
    assert summary["gate"]["max_abs_beta"] is None
    text = json.dumps(mcc._clean({"summary": summary, "trials": trials}), allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text

    lr = summarize_lr_transfer(trials + [dict(t, lr=0.01, final_loss=2.0) for t in trials], d_base=128)
    assert lr["best_lr"] == {"128": 0.01, "256": 0.01}
    assert mcc._clean(lr)["final_loss"]["128"] == [2.0, None]


def test_gradients_finite():
    model = build_model(ModelConfig(width=64, d_base=64, parameterization="sp"), seed=0)
    loss = mcc._loss(model, make_tokens(model.config, 0)[0])
    loss.backward()
    assert mcc.gradients_finite(loss, model)
    assert not mcc.gradients_finite(loss * float("nan"), model)
    model.blocks[1].mlp.up.weight.grad[3, 5] = float("inf")
    assert not mcc.gradients_finite(loss, model)


def test_muon_lr_zero_and_update_scaling_without_step_zero():
    config = ModelConfig(width=64, d_base=64, head_dim=16)
    trial = run_trial(config, optimizer="muon", lr=0.0, lr_aux=0.0, seed=0, steps=2, probe_steps=(2,))
    assert trial["nonfinite_step"] is None
    assert trial["updates"]["blocks.0.attn.q"]["polar_mean_sq_sv"] == [0.0]
    wide = run_trial(ModelConfig(width=128, d_base=64, head_dim=16), optimizer="muon", lr=0.0, lr_aux=0.0,
                     seed=0, steps=2, probe_steps=(2,))
    scaled = mcc.update_scaling([trial, wide], steps=(2,))
    assert "2" in scaled["blocks.0.attn.q"]["update_rms"]
    assert "2" in scaled["blocks.0.attn.q"]["polar_mean_sq_sv"]


# --------------------------------------------------------------------------------------
# The coordinate-check gate and the learning-rate transfer
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def coord():
    """Gate summaries of the real runs, by variant name, and the AdamW learning-rate transfer."""
    variants = {
        "sp_adamw": (Variant("sp_adamw", "sp", "adamw"), 0.003, (0,)),
        "mup_adamw": (Variant("mup_adamw", "mup", "adamw"), 0.003, tuple(range(6))),
        "mup_adamw_readout_lr_power1": (Variant("control", "mup", "adamw", readout_lr_power=1.0), 0.003, (0,)),
        "sp_muon": (Variant("sp_muon", *mcc.resolve("sp_muon"), "PS3_55"), 0.02, (0,)),
        "mup_muon/svd_exact": (Variant("svd", "mup", "muon", "svd_exact"), 0.02, (0,)),
        "mup_muon/PS3_55": (Variant("ps3", "mup", "muon", "PS3_55"), 0.02, (0,)),
    }
    base = {"base": ModelConfig(), "lr_aux": 0.01}
    tasks = [{**base, "name": name, "check": "coord", "variant": variant, "width": width, "seed": seed, "lr": lr,
              "steps": max(PROBE_STEPS), "probe_steps": PROBE_STEPS}
             for name, (variant, lr, seeds) in variants.items() for width in WIDTHS for seed in seeds]
    tasks += [{**base, "name": name, "check": "lr", "variant": variants[name][0], "width": width, "seed": 0,
               "lr": lr, "steps": mcc.LR_STEPS, "probe_steps": ()}
              for name in ("sp_adamw", "mup_adamw") for width in mcc.LR_WIDTHS for lr in mcc.LR_GRIDS["adamw"]]
    results = mcc.run_tasks(tasks, JOBS)
    select = lambda name, check: [trial for task, trial in results if (task["name"], task["check"]) == (name, check)]
    return {"coord": {name: summarize_coord(select(name, "coord")) for name in variants},
            "lr": {name: summarize_lr_transfer(select(name, "lr"), 128) for name in ("sp_adamw", "mup_adamw")}}


@pytest.mark.parametrize("name", ["mup_adamw", "mup_muon/svd_exact", "mup_muon/PS3_55"])
def test_mup_passes_the_coordinate_check(coord, name):
    summary = coord["coord"][name]
    assert summary["nonfinite_trials"] == 0
    assert summary["gate"]["passed"], summary["gate"]
    assert summary["gate"]["max_abs_beta"] < PASS_BETA
    for probe, steps in summary["delta_rms"].items():
        for step in GATE_STEPS:
            assert abs(steps[str(step)]["beta"]) < PASS_BETA, (probe, step)


@pytest.mark.parametrize("name", ["sp_adamw", "sp_muon"])
def test_sp_fails_the_coordinate_check(coord, name):
    verdict = coord["coord"][name]["gate"]
    assert not verdict["passed"] and verdict["fails_clearly"]
    assert verdict["max_abs_beta"] >= FAIL_BETA
    assert verdict["missing_betas"] == 0  # the control fails on its exponents, not on divergence
    assert coord["coord"][name]["delta_rms"]["resid_final"]["12"]["beta"] > 0.5


def test_muon_orthogonalizers_agree_under_mup(coord):
    """PS3_55 in emulated FP16 gives the width exponents of the exact polar factor."""
    exact, ps3 = (coord["coord"][f"mup_muon/{polar}"]["delta_rms"] for polar in ("svd_exact", "PS3_55"))
    for probe in exact:
        for step in GATE_STEPS:
            assert ps3[probe][str(step)]["beta"] == pytest.approx(exact[probe][str(step)]["beta"], abs=0.03)


def test_readout_learning_rate_of_the_task_text_fails_on_the_logits(coord):
    """Logit multiplier ``d_base / d`` with the learning rate ``eta * d_base / d``: logits move as 1 / d."""
    summary = coord["coord"]["mup_adamw_readout_lr_power1"]
    assert not summary["gate"]["passed"] and summary["gate"]["fails_clearly"]
    for step in (1, 2, 4):
        assert summary["delta_rms"]["logits"][str(step)]["beta"] == pytest.approx(-1.0, abs=0.15)


def test_best_learning_rate_transfers_under_mup_and_shifts_under_sp(coord):
    mup, sp = coord["lr"]["mup_adamw"], coord["lr"]["sp_adamw"]
    assert mup["same_best_lr"] and mup["best_lr_log2_shift"] == 0.0
    assert not any(mup["best_at_grid_edge"].values())
    assert max(mup["regret_at_base_best_lr"].values()) == 0.0
    assert not sp["same_best_lr"] and sp["best_lr_log2_shift"] <= -1.0
    assert sp["regret_at_base_best_lr"]["512"] > 0.1


# --------------------------------------------------------------------------------------
# CLI and artifact
# --------------------------------------------------------------------------------------
def test_cli_quick_writes_valid_json(tmp_path, capsys):
    out = tmp_path / "quick.json"
    mcc.main(["--quick", "--jobs", str(JOBS), "--out", str(out)])
    assert "mup_muon/SS3_34" in capsys.readouterr().out
    payload = strict_json(out.read_text())
    json.dumps(payload, allow_nan=False)
    assert payload["schema"] == "cycle3_mup_coord_check/v1" and payload["quick"] is True
    assert re.fullmatch(r"[0-9a-f]{40}", payload["git_sha"]) and isinstance(payload["git_dirty"], bool)
    assert payload["device"] == "cpu" and payload["config"]["widths"] == [128, 256]
    assert payload["config"]["precision"]["polar_native_dtype"]["SS3_54"] == "float16"

    names = {"sp_adamw", "mup_adamw", "mup_adamw_readout_lr_power1", "sp_muon/svd_exact",
             "mup_muon_readout_lr_power1/PS3_55", *(f"mup_muon/{polar}" for polar in mcc.POLAR_METHODS)}
    assert set(payload["coord_check"]) == names
    assert set(payload["lr_transfer"]) == {name for name in names if "readout" not in name}
    assert set(payload["lr_transfer_val"]) == set(payload["lr_transfer"])
    assert set(payload["orthogonalizer_comparison"]) == {f"mup_muon/{polar}" for polar in mcc.POLAR_METHODS}
    for name, entry in payload["coord_check"].items():
        assert isinstance(entry["gate"]["passed"], bool) and entry["nonfinite_trials"] == 0
        assert isinstance(entry["steady_gate"]["passed"], bool)
        assert len(entry["trials"]) == 2 and set(entry["delta_rms"]) == set(entry["trials"][0]["delta_rms"])
        assert entry["delta_rms"]["logits"]["12"]["beta"] is not None
        if entry["optimizer"] == "muon":
            assert all("hidden_polar_mean_sq_sv" in t for t in entry["trials"])
    assert payload["coord_check"]["mup_muon/polar_express"]["polar_dtype"] == "bfloat16"
    assert payload["coord_check"]["sp_adamw"]["gate"]["fails_clearly"]
    row = payload["orthogonalizer_comparison"]["mup_muon/SS3_34"]
    assert row["max_abs_beta_difference_from_svd_exact"] < 0.05 and set(row["best_lr"]) == {"128", "256"}
    assert set(row["val_best_lr"]) == {"128", "256"}
    assert payload["orthogonalizer_comparison"]["mup_muon/svd_exact"]["max_abs_beta_difference_from_svd_exact"] == 0


def test_cli_quick_honors_explicit_overrides(monkeypatch, tmp_path):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"coord_check": {}, "lr_transfer": {}, "lr_transfer_val": {}}

    monkeypatch.setattr(mcc, "run", fake_run)
    monkeypatch.setattr(mcc, "write_json", lambda *args, **kwargs: None)
    mcc.main(["--quick", "--d-base", "64", "--widths", "64", "128", "256",
              "--seeds", "0", "1", "--lr-seeds", "0", "2", "--out", str(tmp_path / "x.json")])
    assert captured["widths"] == (64, 128, 256)
    assert captured["seeds"] == (0, 1)
    assert captured["lr_seeds"] == (0, 2)
    assert captured["lr_widths"] == (64, 128)


@pytest.mark.skipif(not ARTIFACT.exists(), reason="results/cycle3_mup_coord_check_cpu.json is not generated yet")
def test_artifact_is_consistent_with_its_trials():
    payload = strict_json(ARTIFACT.read_text())
    assert payload["git_dirty"] is False and re.fullmatch(r"[0-9a-f]{40}", payload["git_sha"])
    assert payload["quick"] is False
    assert payload["config"]["widths"] == list(WIDTHS) and payload["config"]["seeds"] == [0, 1, 2]
    assert set(payload["lr_transfer_val"]) == set(payload["lr_transfer"])
    for name, entry in payload["coord_check"].items():
        assert len(entry["trials"]) == len(WIDTHS) * len(payload["config"]["seeds"])
        recomputed = gate(width_scaling(entry["trials"], "delta_rms"))
        assert recomputed["passed"] == entry["gate"]["passed"], name
        assert recomputed["fails_clearly"] == entry["gate"]["fails_clearly"], name
        assert recomputed["max_abs_beta"] == pytest.approx(entry["gate"]["max_abs_beta"], abs=1e-4), name
        recomputed_steady = gate(width_scaling(entry["trials"], "delta_rms"), mcc.STEADY_GATE_STEPS)
        assert recomputed_steady["passed"] == entry["steady_gate"]["passed"], name
        assert recomputed_steady["max_abs_beta"] == pytest.approx(entry["steady_gate"]["max_abs_beta"], abs=1e-4), name
        if entry["optimizer"] == "muon":
            assert all(len(t["hidden_polar_mean_sq_sv"]) == len(PROBE_STEPS) for t in entry["trials"]), name
    for key, loss_field in (("lr_transfer", "final_loss"), ("lr_transfer_val", "eval_loss")):
        for name, entry in payload[key].items():
            trials = payload["lr_transfer"][name]["trials"]
            recomputed = summarize_lr_transfer(trials, payload["config"]["model"]["d_base"], loss_key=loss_field)
            assert recomputed["best_lr"] == entry["best_lr"], (key, name)
            assert recomputed["same_best_lr"] == entry["same_best_lr"], (key, name)
            for width in entry["widths"]:
                assert recomputed["final_loss"][str(width)] == pytest.approx(entry["final_loss"][str(width)], rel=1e-4), (key, name, width)
