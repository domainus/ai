"""Checks for the Cycle 4 C4-03 driver (wide transfer, common checkpoints, Gram drift) and its results."""

import copy
import json
import math
import random
import shutil
from dataclasses import asdict

import pytest
import torch
from scipy import stats

from research.polar_opt import bench_c4_03_wide_and_gram_drift as c3
from research.polar_opt import harness_tier2_real as h
from research.polar_opt import real_text_data as rtd
from research.polar_opt.baselines import polar_express
from research.quant_ptq import micro_rotated_ptq as q
from tests.test_tier2_real import ByteEncoder, _documents

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"
needs_cuda = pytest.mark.skipif(not CUDA, reason="the Blackwell Triton kernels need CUDA")


def _tampered(payload, edit):
    payload = copy.deepcopy(payload)
    edit(payload)
    return payload


def _set(path, value=None, *, delta=None):
    def edit(payload):
        node = payload
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = node[path[-1]] + delta if delta is not None else value
    return edit


# ------------------------------------------------------------------ CLI, protocols and guards
def test_canonical_protocols_and_default_outputs():
    for stage in c3.RUN_STAGES:
        for extra in ([], ["--resume"]):
            args = c3.parse_args(["--stage", stage, *extra])
            assert args.canonical and args.output == c3.CANONICAL_OUTPUTS[stage]
            assert args.protocol == c3.CANONICAL[stage] and args.protocol is not c3.CANONICAL[stage]
        quick = c3.parse_args(["--stage", stage, "--quick"])
        assert quick.output == c3.QUICK_OUTPUTS[stage] and not quick.canonical and not c3.reserved(quick.output)
    assert c3.parse_args(["--stage", "summary"]).output == c3.CANONICAL_OUTPUTS["summary"]
    assert all(c3.reserved(p) for p in c3.CANONICAL_OUTPUTS.values())

    ckpt = c3.CANONICAL["ckpt"]
    config = h.ModelConfig(**ckpt["configs"]["120m"])
    assert config == h.PRESETS["120m"] and config.step_tokens == 32_768
    assert (ckpt["steps"], ckpt["checkpoints"], ckpt["seeds"]) == (384, [128, 256, 384], [1, 2, 3, 4, 5])
    assert ckpt["steps"] * config.step_tokens == 12_582_912 < 0.5 * 26_004_518
    assert ckpt["optimizers"] == ["muon_ps3_blackwell", "muon_ss3_54", "muon_ss3_53"]
    assert (ckpt["muon_lr"], ckpt["muon_aux_lr"], ckpt["weight_decay"], ckpt["lr_warmup_steps"],
            ckpt["min_lr_ratio"], ckpt["eval_tokens"]) == (0.02, 0.001, 0.01, 50, 0.1, 2_000_000)
    assert c3.eval_tokens_used(2_500_000, config, ckpt["eval_tokens"]) == 1_998_848

    wide = c3.CANONICAL["wide"]
    assert [c["width"] for c in wide["configs"].values()] == [768, 1536, 2048]
    assert all(h.ModelConfig(**c).step_tokens == 16_384 for c in wide["configs"].values())
    assert (wide["optimizers"], wide["seeds"], wide["steps"]) == (list(c3.MUON), [1, 2, 3], 64)
    for protocol in (wide, ckpt):
        assert protocol["compile_model"] is True and protocol["update_dtype"] == "fp32"
    drift = c3.CANONICAL["drift"]
    assert drift["families"]["wide4"]["k_exponents"] == list(range(10, 26))
    assert drift["families"]["tall2"]["k_exponents"][-1] == 25 and 25 in drift["families"]["wide1"]["k_exponents"]
    assert drift["gate_k"] == 65536 and drift["chunk_k"] == 4096 and drift["ceiling"] == 1.08


def test_cli_refuses_reserved_filenames_and_bad_overrides(tmp_path, monkeypatch):
    for name in ("run_train_stage", "run_drift_stage", "run_summary"):
        monkeypatch.setattr(c3, name, lambda args: pytest.fail("must not run"))
    other = tmp_path / "other" / "results" / c3.CANONICAL_OUTPUTS["wide"].name
    alias = tmp_path / "alias.json"
    alias.symlink_to(other)
    novel = tmp_path / "results" / "cycle4_c4_03_other.json"
    assert all(c3.reserved(p) for p in (other, alias, novel))
    custom = tmp_path / "custom.json"
    assert not c3.reserved(custom)
    args = c3.parse_args(["--stage", "wide", "--seeds", "4", "5", "6", "--output", str(custom)])
    assert args.output == custom and not args.canonical and args.protocol["seeds"] == [4, 5, 6]
    assert c3.parse_args(["--stage", "wide", "--output", str(other)]).canonical  # any checkout
    args = c3.parse_args(["--stage", "ckpt", "--steps", "200", "--output", str(custom)])
    assert (args.protocol["steps"], args.protocol["checkpoints"]) == (200, [128, 200])
    for argv in (
        ["--stage", "wide", "--quick", "--output", str(c3.CANONICAL_OUTPUTS["wide"])],
        ["--stage", "wide", "--quick", "--output", str(other)],
        ["--stage", "ckpt", "--quick", "--output", str(alias)],
        ["--stage", "drift", "--quick", "--output", str(novel)],
        ["--stage", "summary", "--quick", "--output", str(c3.CANONICAL_OUTPUTS["summary"])],
        ["--stage", "ckpt", "--output", str(other)],  # the wide file from the ckpt stage
        ["--stage", "wide", "--seeds", "1", "2", "--output", str(other)],
        ["--stage", "wide", "--device", "cpu", "--output", str(other)],
        ["--stage", "ckpt", "--steps", "320", "--output", str(tmp_path / c3.CANONICAL_OUTPUTS["ckpt"].name)],
        ["--stage", "wide", "--optimizers", "muon_ss3_54"],  # non-canonical without --output
        ["--stage", "drift", "--seeds", "7"],
        ["--stage", "wide", "--optimizers", "adamw", "--output", str(custom)],
        ["--stage", "wide", "--seeds", "1", "1", "--output", str(custom)],
        ["--stage", "drift", "--steps", "4", "--output", str(custom)],
        ["--stage", "summary", "--seeds", "1"],
        ["--stage", "wide", "--wide-input", str(custom)],
    ):
        with pytest.raises(SystemExit):
            c3.parse_args(argv)
    assert not other.exists()


def test_canonical_runs_need_sm_120_and_a_clean_tree_before_any_data(tmp_path, monkeypatch):
    other = tmp_path / "elsewhere" / c3.CANONICAL_OUTPUTS["ckpt"].name
    alias = tmp_path / "alias.json"
    alias.symlink_to(other)
    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for capability, allowed in (((12, 0), True), ((13, 0), True), ((10, 0), False), ((9, 0), False)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, c=capability: c)
        for path in (c3.CANONICAL_OUTPUTS["drift"], other, alias):
            if allowed:
                c3.require_blackwell(cuda, path)
            else:
                with pytest.raises(RuntimeError, match="sm_120"):
                    c3.require_blackwell(cuda, path)
        if not allowed:
            with pytest.raises(RuntimeError, match="sm_120"):
                c3.require_blackwell(cuda, tmp_path / "custom.json", canonical=True)
        c3.require_blackwell(cuda, tmp_path / "custom.json")
    for path in (c3.CANONICAL_OUTPUTS["wide"], other, alias):
        with pytest.raises(RuntimeError, match="sm_120"):
            c3.require_blackwell(cpu, path)
    # Both guards run before data is read, training starts or a kernel launches.
    monkeypatch.setattr(q, "load_data", lambda *a, **k: pytest.fail("guards must run first"))
    monkeypatch.setattr(c3, "set_numerics", lambda device: pytest.fail("guards must run first"))
    runners = {"wide": c3.run_train_stage, "ckpt": c3.run_train_stage, "drift": c3.run_drift_stage}
    for stage, runner in runners.items():
        target = tmp_path / stage / c3.CANONICAL_OUTPUTS[stage].name
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
        with pytest.raises(RuntimeError, match="sm_120"):
            runner(c3.parse_args(["--stage", stage, "--output", str(target)]))
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
        monkeypatch.setattr(h, "git_info", lambda: {"git_sha": "a" * 40, "git_dirty": True})
        with pytest.raises(RuntimeError, match="clean"):
            runner(c3.parse_args(["--stage", stage, "--output", str(target)]))
        monkeypatch.setattr(h, "git_info", lambda: {"git_sha": None, "git_dirty": None})
        with pytest.raises(RuntimeError, match="clean"):
            runner(c3.parse_args(["--stage", stage]))
        assert not target.exists()


def test_code_sha256_covers_the_transitive_import_closure(tmp_path, monkeypatch):
    modules = c3.code_modules()
    assert modules == tuple(sorted(modules)) and all(m.startswith("research.") for m in modules)
    for needed in c3.REQUIRED_CODE:
        assert any(m.endswith("." + needed) for m in modules), needed
    assert "research.polar_opt.spectral_scaling" in modules  # reached only through circuits.py
    original = c3.code_sha256()
    assert len(original) == 64 and original == c3.code_sha256()
    shutil.copytree(c3.ROOT / "research", tmp_path / "research", ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(c3, "ROOT", tmp_path)
    c3.code_modules.cache_clear()
    try:
        assert c3.code_modules() == modules and c3.code_sha256() == original
        for name in ("kernels_blackwell", "spectral_scaling", "real_text_data", "bench_c4_03_wide_and_gram_drift"):
            path = tmp_path / "research" / "polar_opt" / f"{name}.py"
            text = path.read_text()
            path.write_text(text + "\n# edit\n")
            assert c3.code_sha256() != original, name
            path.write_text(text)
        assert c3.code_sha256() == original
    finally:
        monkeypatch.undo()
        c3.code_modules.cache_clear()
    assert c3.code_sha256() == original


# ------------------------------------------------------------------ statistics
def test_t_quantiles_are_exact():
    assert c3.T975[4] == pytest.approx(stats.t.ppf(0.975, 4), rel=1e-14)
    assert c3.T975[2] == pytest.approx(stats.t.ppf(0.975, 2), rel=1e-14)
    x = c3.T975[4] / math.sqrt(c3.T975[4] ** 2 + 4)
    assert 0.5 + (3 * x - x ** 3) / 4 == pytest.approx(0.975, abs=1e-15)


@pytest.mark.parametrize("values,null", [
    ([0.010, 0.012, 0.009, 0.011, 0.0135], 0.0), ([-0.3, 0.1, 0.25, -0.05, 0.4], 0.0),
    ([1.0018, 1.0009, 1.0031, 0.9997, 1.0012], 1.0), ([-0.0050, -0.0053, -0.0061, 0.0088, -0.0007], 0.0),
    ([5.4, 5.4000001, 5.3999999, 5.4000002, 5.4], 5.0), ([0.010, 0.012, 0.009], 0.0), ([1.002, 0.999, 1.004], 1.0)])
def test_paired_t_matches_scipy(values, null):
    found = c3.paired_t(values, null)
    reference = stats.ttest_1samp(values, null)
    n = len(values)
    assert (found["n"], found["df"]) == (n, n - 1)
    assert found["t"] == pytest.approx(reference.statistic, rel=1e-9)
    assert found["p_two_sided"] == pytest.approx(reference.pvalue, rel=1e-9)
    low, high = reference.confidence_interval(0.95)
    assert found["ci95"] == pytest.approx([low, high], rel=1e-9, abs=1e-15)
    assert found["mean"] == pytest.approx(sum(values) / n) and found["se"] == pytest.approx(stats.sem(values))


def test_paired_t_edge_cases_and_stability():
    constant = c3.paired_t([0.5] * 5, 0.5)
    assert constant["t"] is None and constant["p_two_sided"] == 1.0 and constant["ci95"] == [0.5, 0.5]
    assert c3.paired_t([0.5] * 5)["p_two_sided"] == 0.0
    for values in ([0.1, 0.2, 0.3, 0.4], [0.1, 0.2], []):
        with pytest.raises(ValueError):
            c3.paired_t(values)
    for spread in (1.0, 1e-3, 1e-6, 1e-9, 1e-12):  # a large |t| must not cancel to zero
        found = c3.paired_t([1.0 + spread * d for d in (0.0, 1.0, -1.0, 2.0, -2.0)], 0.0)
        assert 0 < found["p_two_sided"] == pytest.approx(2 * stats.t.sf(found["t"], 4), rel=1e-9)
    assert found["t"] > 1e11 and found["p_two_sided"] < 1e-40


def _synthetic_per_seed(stage):
    protocol = c3.CANONICAL[stage]
    rng = random.Random(stage)
    table = {}
    for label in protocol["configs"]:
        for name in protocol["optimizers"]:
            table.setdefault(label, {})[name] = [
                {"seed": seed, "init_fingerprint": f"{label}{seed}", "stream_fingerprint": f"s{seed}",
                 "tokens_per_second": 4e4 + rng.random(), "polar_ortho_ms": 30 + rng.random(),
                 "polar_fraction": 0.1 + rng.random() / 100,
                 **{k: 1.0 + rng.random() for k in c3.TIMING_KEYS if k != "polar_ortho_ms"},
                 "checkpoints": [{"step": step, "val_loss": 5 + rng.random(), "test_loss": 5 + rng.random()}
                                 for step in protocol["checkpoints"]]} for seed in protocol["seeds"]]
    return table


def test_analysis_runs_every_predeclared_contrast_and_holm_family():
    wide = c3.analyze_train("wide", _synthetic_per_seed("wide"), [64])
    assert len(wide["contrasts"]) == 3 * 5 * (3 + 2) == 75
    assert set(wide["holm_group_sizes"]["p_holm_family"].values()) == {5}
    assert wide["holm_group_sizes"]["p_holm_joint"] == {
        "tokens_per_second_ratio": 15, "polar_ortho_ms_ratio": 15, "polar_fraction_diff": 15, "val_loss_diff": 15,
        "test_loss_diff": 15}
    assert all(c["stats"]["df"] == 2 for c in wide["contrasts"])
    table = _synthetic_per_seed("ckpt")
    ckpt = c3.analyze_train("ckpt", table, [128, 256, 384])
    assert len(ckpt["contrasts"]) == 2 * (3 + 2 * 3) == 18 and all(c["stats"]["df"] == 4 for c in ckpt["contrasts"])
    assert ckpt["holm_group_sizes"]["p_holm_joint"] == {
        "tokens_per_second_ratio": 2, "polar_ortho_ms_ratio": 2, "polar_fraction_diff": 2, "val_loss_diff": 6,
        "test_loss_diff": 6}
    assert sorted(ckpt["holm_group_sizes"]["p_holm_family"].values()) == [1] * 6 + [3] * 4
    one = next(c for c in ckpt["contrasts"] if c["id"] == "120m:test_loss_diff@256:muon_ss3_54_vs_muon_ps3_blackwell")
    values = [a["checkpoints"][1]["test_loss"] - b["checkpoints"][1]["test_loss"]
              for a, b in zip(table["120m"]["muon_ss3_54"], table["120m"]["muon_ps3_blackwell"])]
    assert one["stats"] == c3.paired_t(values, 0.0) and one["seeds"] == [1, 2, 3, 4, 5]
    family = [c for c in ckpt["contrasts"] if c["family"] == one["family"]]
    assert [c["checkpoint"] for c in family] == [128, 256, 384]
    assert [c["p_holm_family"] for c in family] == c3.holm_bonferroni([c["stats"]["p_two_sided"] for c in family])
    joint = [c for c in ckpt["contrasts"] if c["joint"] == "test_loss_diff"]
    assert [c["p_holm_joint"] for c in joint] == c3.holm_bonferroni([c["stats"]["p_two_sided"] for c in joint])
    assert all(c["stats"]["null"] == (1.0 if c["kind"] == "ratio" else 0.0) for c in ckpt["contrasts"])
    table["120m"]["muon_ss3_54"][2]["stream_fingerprint"] = "other"
    with pytest.raises(ValueError, match="not paired"):
        c3.analyze_train("ckpt", table, [128, 256, 384])


# ------------------------------------------------------------------ exact model of the MMA accumulate
def test_mma_accumulate_rule():
    ulp = 2.0 ** -23  # of the accumulator 1.0

    def kept(product_a, product_b, count=15):
        return (c3.mma_accumulate([1.0] + [product_a] * count, [1.0] + [product_b] * count) - 1.0) / ulp

    assert kept(2.0 ** -11, 2.0 ** -11) == 15 * 2.0  # 2 ulp each: exact
    assert kept(2.0 ** -12, 2.0 ** -12) == 7.0  # 15 * 0.5 = 7.5, truncated to 24 bits
    assert kept(2.0 ** -12, 2.0 ** -13) == 3.0  # 1/4 ulp is the last guard bit: kept
    assert kept(2.0 ** -13, 2.0 ** -13) == 0.0  # 1/8 ulp is below both guard bits: dropped
    assert kept(2.0 ** -12, 1.5 * 2.0 ** -13) == 3.0  # 0.375 ulp is truncated to 0.25 ulp
    assert kept(2.0 ** -12, -(2.0 ** -12)) == -7.5  # below 1.0 the ulp is 2^-24
    assert kept(2.0 ** -12, -(2.0 ** -13)) == -4.0  # 1 - 3.75 ulp truncated toward zero at 2^-24
    assert kept(2.0 ** -13, -(2.0 ** -13)) == 0.0
    # A product below the guard bits never accumulates, however many there are.
    assert c3.mma_accumulate([1.0] + [2.0 ** -13] * 4095, [1.0] + [2.0 ** -13] * 4095) == 1.0
    assert c3.mma_accumulate([2.0 ** -13] * 4096, [2.0 ** -13] * 4096) == 4096 * 2.0 ** -26  # small accumulator
    # Short sums are exact, and the result does not depend on a power-of-two scale.
    a, b = [0.5, 0.25, -0.125, 3.0], [1.0, 2.0, 4.0, 0.5]
    assert c3.mma_accumulate(a, b) == 0.5 + 0.5 - 0.5 + 1.5
    generator = torch.Generator().manual_seed(0)
    rows = (torch.randn(2, 500, generator=generator) / 32).half().double()
    plain = c3.mma_accumulate(rows[0].tolist(), rows[1].tolist())
    scaled = c3.mma_accumulate((rows[0] * 2 ** 9).tolist(), (rows[1] * 2 ** 9).tolist())
    assert scaled == plain * 2 ** 18 and c3.mma_accumulate([], []) == 0.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vectorized_diagonal_model_equals_the_exact_model(dtype):
    generator = torch.Generator().manual_seed(1)
    for k in (1, 15, 16, 17, 100, 4096, 30_000):
        row = (torch.randn(k, generator=generator) / math.sqrt(4 * k)).to(dtype)
        values = row.double().tolist()
        assert c3.mma_predict_diag(row) == c3.mma_accumulate(values, values), k
    assert c3.mma_predict_diag(torch.zeros(40, dtype=dtype)) == 0.0
    spike = torch.zeros(20_000, dtype=dtype)
    spike[-1] = 1.0  # the accumulator never exceeds the largest product in the prefix
    assert c3.mma_predict_diag(spike, prefix_groups=8) is None


def test_model_bias_is_linear_in_k_and_downward():
    generator = torch.Generator().manual_seed(2)
    bias = {}
    for exponent in (14, 16, 18):
        k = 1 << exponent
        row = (torch.randn(k, generator=generator) / math.sqrt(k)).half()
        bias[k] = 1.0 - c3.mma_predict_diag(row) / row.double().square().sum().item()
    assert all(4.5e-9 < value / k < 7.5e-9 for k, value in bias.items()), bias  # about 5.6e-9 per product


def test_chunk_view_is_a_strided_view_without_a_copy():
    wide = torch.arange(3 * 20, dtype=torch.float32).reshape(3, 20)
    tall = torch.arange(20 * 2, dtype=torch.float32).reshape(20, 2)
    for view in (wide, tall.mT):
        chunks = c3.chunk_view(view, 8)
        assert chunks.shape == (2, view.shape[0], 8) and chunks.data_ptr() == view.data_ptr()
        for i in range(2):
            assert torch.equal(chunks[i], view[:, 8 * i:8 * i + 8])
    assert c3.chunk_view(wide, 8).stride() == (8, 20, 1) and c3.chunk_view(tall.mT, 8).stride() == (16, 1, 2)
    assert c3.chunk_view(wide, 32).shape == (0, 3, 32)
    for chunk, gate in ((0, 16), (32, 16)):
        with pytest.raises(ValueError):
            c3.ChunkedGram(chunk, gate)


def test_onset_and_design_ceilings():
    points = [(1024, 1.01), (4096, 1.02), (16384, 1.06), (65536, 1.10), (262144, 1.30)]
    onset = c3._onset(points, 1.08)
    assert (onset["first_k_above"], onset["last_k_at_or_below"], onset["monotone_above"]) == (65536, 16384, True)
    assert onset["interpolated_k"] == pytest.approx(2.0 ** 15.0) and onset["sigma_max_at_largest_k"] == 1.30
    never = c3._onset(points[:3], 1.08)
    assert never["first_k_above"] is None and never["last_k_at_or_below"] == 16384
    blown = c3._onset([(1024, 1.0), (4096, math.inf)], 1.08)  # a non-finite output counts as above
    assert blown["first_k_above"] == 4096 and blown["interpolated_k"] is None
    assert blown["sigma_max_at_largest_k"] is None
    json.dumps(blown, allow_nan=False)
    ceilings = c3.design_ceilings(["ps3_55", "ss3_54", "ss3_53"])
    assert ceilings["ss3_54"] == pytest.approx(1.0798, abs=2e-4) and ceilings["ss3_54"] < 1.08
    assert ceilings["ss3_53"] == pytest.approx(1.0875, abs=2e-4)
    assert ceilings["ps3_55"] == pytest.approx(1.0810, abs=2e-4)


# ------------------------------------------------------------------ the trial loop
@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    out = tmp_path_factory.mktemp("c4_03_shards")
    rtd.build_shards(_documents(), out, targets={"train": 60_000, "val": 6_000, "test": 6_000},
                     source={"name": "synthetic"}, encoder=ByteEncoder())
    return out


def test_trial_matches_the_harness_and_evaluation_does_not_perturb_it(shards):
    config, device = h.PRESETS["tiny"], torch.device("cpu")
    data = h.ShardData(shards, verify=False)
    init = h.initial_state(config, 3)
    protocol = {**c3.QUICK["ckpt"], "steps": 6}
    rows = {}
    for checkpoints in ([6], [2, 4, 6]):
        model = h.Transformer(config)
        rows[len(checkpoints)] = c3.run_trial(
            config, data, device=device, seed=3, name="muon_ss3_54", factory=lambda: polar_express,
            protocol={**protocol, "checkpoints": checkpoints}, init=init, reusable=(model, model))
    model = h.Transformer(config)
    reference = h.run_trial(
        config, data, device=device, seed=3, optimizer_name="muon_ss3_54", factory=lambda: polar_express,
        lr=protocol["muon_lr"], auxiliary_lr=protocol["muon_aux_lr"], weight_decay=protocol["weight_decay"],
        update_dtype="fp32", tokens=6 * config.step_tokens, lr_warmup_steps=protocol["lr_warmup_steps"],
        min_lr_ratio=protocol["min_lr_ratio"], compile_warmup_steps=protocol["compile_warmup_steps"],
        eval_tokens=protocol["eval_tokens"], init=init, reusable_model=(model, model))
    one, three = rows[1], rows[3]
    for row in (one, three):
        assert (row["val_loss"], row["test_loss"]) == (reference["val_loss"], reference["test_loss"])
        assert (row["init_fingerprint"], row["stream_fingerprint"]) == (
            reference["init_fingerprint"], reference["stream_fingerprint"])
        assert row["steps"] == 6 and not row["diverged"] and len(row["step_wall_seconds"]) == 6
        assert row["train_seconds"] == pytest.approx(sum(row["step_wall_seconds"]))
        assert row["timing"] == c3.timing_from_raw(row) and row["timing"]["steps_timed"] == 5
    assert [c["step"] for c in three["checkpoints"]] == [2, 4, 6] and three["checkpoints"][-1] == {
        **one["checkpoints"][-1], "train_seconds": three["train_seconds"]}
    assert three["checkpoints"][0]["val_loss"] != three["checkpoints"][2]["val_loss"]  # the loss moves
    assert three["checkpoints"][1]["train_seconds"] == pytest.approx(sum(three["step_wall_seconds"][:4]))
    assert three["last_train_loss"] == one["last_train_loss"] == reference["last_train_loss"]


def test_train_plan_is_seeded_per_config_and_seed():
    protocol = c3.CANONICAL["wide"]
    plan = c3.train_plan("wide", protocol)
    assert len(plan) == 36 and plan == c3.train_plan("wide", protocol)
    for label in protocol["configs"]:
        for seed in protocol["seeds"]:
            block = list(protocol["optimizers"])
            random.Random(f"{seed}-wide-{label}").shuffle(block)
            assert [name for cfg, s, _, name in plan if (cfg, s) == (label, seed)] == block
            assert [order for cfg, s, order, _ in plan if (cfg, s) == (label, seed)] == [0, 1, 2, 3]
    assert len({tuple(name for cfg, s, _, name in plan if (cfg, s) == (label, seed))
                for label in protocol["configs"] for seed in protocol["seeds"]}) > 1
    assert len(c3.train_plan("ckpt", c3.CANONICAL["ckpt"])) == 15
    groups = c3.muon_groups(c3.WIDE_CONFIGS["w2048"])
    assert [g["stacked"] for g in groups] == [[4, 6144, 2048], [4, 2048, 2048], [8, 5440, 2048], [4, 2048, 5440]]
    assert c3.model_record(c3.WIDE_CONFIGS["w1536"])["parameters"]["muon_matrices"] == 169_869_312


# ------------------------------------------------------------------ quick pipeline
@pytest.fixture(scope="module")
def quick(tmp_path_factory, shards):
    tmp = tmp_path_factory.mktemp("c4_03_quick")
    common = ["--quick", "--silent", "--device", DEVICE]
    train = [*common, "--no-verify", "--data-dir", str(shards)]
    out = {stage: c3.main(["--stage", stage, *train, "--output", str(tmp / f"{stage}.json")])
           for stage in c3.TRAIN_STAGES}
    if CUDA:
        out["drift"] = c3.main(["--stage", "drift", *common, "--output", str(tmp / "drift.json")])
        out["summary"] = c3.main(["--stage", "summary", "--quick", "--silent", "--output", str(tmp / "summary.json"),
                                  *[x for s in c3.RUN_STAGES for x in (f"--{s}-input", str(tmp / f"{s}.json"))]])
    return {"dir": tmp, "train": train, "common": common, **out}


def test_quick_training_stages(quick):
    for stage, runs in (("wide", 24), ("ckpt", 15)):
        payload = quick[stage]
        c3.validate_results(payload)
        assert payload["stage"] == stage and payload["quick"] and not payload["canonical"]
        assert not c3.resume_log_path(quick["dir"] / f"{stage}.json").exists()  # removed after the final write
        assert len(payload["runs"]) == runs and payload["derived"]["divergences"] == 0
        assert payload["meta"]["numerics"]["allow_tf32"] is False
        assert payload["run_config_hash"] == c3.expected_run_config_hash(payload)
        assert [(r["config"], r["seed"], r["order_in_block"], r["optimizer"]) for r in payload["runs"]] == \
            c3.train_plan(stage, payload["protocol"])
        by_seed = {}
        for r in payload["runs"]:
            by_seed.setdefault((r["config"], r["seed"]), set()).add((r["init_fingerprint"], r["stream_fingerprint"]))
            assert r["steps"] == 6 == len(r["step_wall_seconds"]) and r["epoch_fraction"] < 1
            assert [c["step"] for c in r["checkpoints"]] == payload["protocol"]["checkpoints"]
        assert all(len(v) == 1 for v in by_seed.values())
        assert len({next(iter(v)) for v in by_seed.values()}) == len(by_seed)
        if CUDA:
            assert "Triton" in payload["protocol"]["orthogonalizers"]["muon_ss3_54"]
            assert all(set(r["step_ms"]) == set(c3.SEGMENTS) and 0 < r["polar_fraction"] < 1
                       and r["peak_vram_mib"]["allocated"] > 0 for r in payload["runs"])
    assert list(quick["wide"]["models"]) == ["w64", "w96"]
    assert [c["step"] for c in quick["ckpt"]["runs"][0]["checkpoints"]] == [2, 4, 6]
    json.dumps(quick["wide"], allow_nan=False)


def test_resume_reuses_finished_rows_and_repairs_a_torn_tail(quick, tmp_path):
    out = tmp_path / "resumed.json"
    log = c3.resume_log_path(out)
    rows = quick["ckpt"]["runs"]
    argv = ["--stage", "ckpt", *quick["train"], "--resume", "--output", str(out)]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows[:4]) + '{"cut off')
    payload = c3.main(argv)
    assert payload["meta"]["resume"] == {"resume": True, "rows_loaded": 4, "rows_run": 11}
    assert not log.exists() and len(payload["runs"]) == 15
    assert payload["runs"][:4] == rows[:4]
    c3.validate_results(payload)
    # Double interruption: the torn tail is repaired on disk before any later append.
    log.write_text("".join(json.dumps(r) + "\n" for r in rows[:2]) + '{"torn')
    loaded = c3.load_rows(log, quick["ckpt"]["run_config_hash"], resume=True)
    assert len(loaded) == 2 and log.read_text().endswith("\n") and "torn" not in log.read_text()
    with log.open("a") as handle:
        handle.write(json.dumps(rows[2]) + "\n" + '{"torn2')
    twice = c3.main(argv)
    assert twice["meta"]["resume"] == {"resume": True, "rows_loaded": 3, "rows_run": 12}
    assert not log.exists() and len(twice["runs"]) == 15
    c3.validate_results(twice)
    log.write_text(json.dumps({**rows[0], "run_config_hash": "0" * 16}) + "\n")
    with pytest.raises(ValueError, match="resume"):
        c3.main(argv)
    log.unlink()
    # Without --resume a stale log is discarded.
    log.write_text(json.dumps(rows[0]) + "\n")
    assert c3.load_rows(log, quick["ckpt"]["run_config_hash"], resume=False) == [] and not log.exists()


def test_validate_results_rejects_tampered_runs(quick):
    wide, ckpt = quick["wide"], quick["ckpt"]
    for payload in (wide, ckpt):
        with pytest.raises(ValueError):
            c3.validate_results(payload, canonical=True)  # a quick result never passes as canonical

    def forged_hash(p):
        p["meta"]["code_sha256"] = "0" * 64

    def swap_rows(p):
        p["runs"][0], p["runs"][1] = p["runs"][1], p["runs"][0]

    edits = [
        (ckpt, _set(["runs", 0, "test_loss"], delta=1e-9)),
        (ckpt, _set(["runs", 0, "checkpoints", 1, "val_loss"], delta=1e-9)),
        (ckpt, _set(["runs", 1, "checkpoints", 0, "step"], 3)),
        (ckpt, _set(["runs", 1, "checkpoints", 0, "train_seconds"], delta=1e-6)),
        (ckpt, _set(["runs", 3, "train_seconds"], delta=1e-9)),
        (ckpt, _set(["runs", 3, "step_wall_seconds", 2], delta=1e-6)),
        (ckpt, _set(["runs", 1, "tokens_per_second"], delta=1e-6)),
        (ckpt, _set(["runs", 2, "tokens"], delta=1)),
        (ckpt, _set(["runs", 2, "steps"], 7)),
        (ckpt, _set(["runs", 4, "epoch_fraction"], delta=1e-9)),
        (ckpt, _set(["runs", 5, "diverged"], True)),
        (ckpt, _set(["runs", 6, "init_fingerprint"], "0" * 16)),
        (ckpt, _set(["runs", 6, "run_config_hash"], "0" * 16)),
        (ckpt, _set(["runs", 7, "timing", "step_wall_ms"], delta=1e-6)),
        (ckpt, _set(["runs", 7, "order_in_block"], 2 if ckpt["runs"][7]["order_in_block"] != 2 else 1)),
        (ckpt, _set(["runs", 8, "lr"], 0.04)),
        (ckpt, _set(["protocol", "muon_lr"], 0.04)),  # the hash covers the whole protocol
        (ckpt, _set(["protocol", "checkpoints"], [2, 4])),
        (ckpt, _set(["meta", "git_sha"], "b" * 40)),
        (ckpt, _set(["meta", "numerics", "allow_tf32"], True)),
        (ckpt, _set(["meta", "code_modules"], ["research.polar_opt.harness_tier2_real"])),
        (ckpt, forged_hash),
        (ckpt, _set(["runs", 0, "checkpoints", 0, "val_ppl"], math.nan)),
        (ckpt, lambda p: p["runs"].pop()),
        (ckpt, swap_rows),
        (ckpt, _set(["derived", "per_config", "tiny", "muon_ss3_54", "test_loss_step4_mean"], delta=1e-9)),
        (wide, _set(["models", "w96", "parameters", "total"], delta=1)),
        (wide, _set(["models", "w64", "muon_groups", 0, "stacked", 0], 3)),
        (wide, _set(["derived", "max_epoch_fraction"], delta=1e-9)),
        (wide, _set(["runs", 5, "config"], "w96")),
        (wide, _set(["data", "splits", "train", "sha256"], "0" * 64)),
    ]
    if CUDA:
        edits += [(wide, _set(["runs", 2, "polar_fraction"], delta=1e-9)),
                  (wide, lambda p: p["runs"][2]["step_ms"].update(
                      polar_ortho=[v + 1.0 for v in p["runs"][2]["step_ms"]["polar_ortho"]])),
                  (wide, _set(["runs", 4, "peak_vram_mib"], None)),
                  (wide, lambda p: p["runs"][0]["step_ms"].pop("layout_cast"))]
    for payload, edit in edits:
        with pytest.raises(ValueError):
            c3.validate_results(_tampered(payload, edit))
    c3.validate_results(wide)
    c3.validate_results(ckpt)  # the fixtures are untouched


@needs_cuda
def test_quick_drift_stage(quick):
    drift = quick["drift"]
    c3.validate_results(drift)
    assert [r["id"] for r in drift["rows"]] == [unit for unit, _ in c3.drift_units(drift["protocol"])]
    checks, mitigation = drift["derived"]["checks"], drift["derived"]["mitigation"]
    for key in ("prescale_bitwise_equal", "syrk_norm_equals_rounded_syrk", "mma_model_sweep_diag_bitwise",
                "mma_model_exact_entries_bitwise", "mma_probe_model_equals_triton"):
        assert checks[key][0] == checks[key][1] > 0, key
    assert all(equal == configs == 11 for equal, configs in checks["config_independence"])
    assert checks["svd_check_max_abs_diff"] < 1e-12
    gate = drift["protocol"]["gate_k"]
    for r in (r for r in drift["rows"] if r["unit"] == "sweep"):
        for cell in r["kernels"].values():
            assert cell["mitigated"]["chunked"] == (r["k"] > gate) and cell["production"]["finite"]
            if r["k"] <= gate:  # the production code ran unchanged
                assert {k: cell["mitigated"][k] for k in ("sigma_max", "rel_fro_err")} == {
                    k: cell["production"][k] for k in ("sigma_max", "rel_fro_err")}
    normal = mitigation["normal_shapes"]
    assert normal["cells"] == normal["bitwise_equal"] == 3 and normal["chunked_launches"] == 0
    assert mitigation["chunked_cells"] == 18 and mitigation["cells"] == 30
    probe = next(r for r in drift["rows"] if r["unit"] == "mma_probe")["cases"]
    dropped = [c for c in probe if abs(c["exact_ulp"]) == 0.125]  # 2^-26 against the accumulator 1.0
    assert dropped and all(c["triton_kept_ulp"] == 0.0 for c in dropped)

    def unpinned(p):
        p["rows"][4]["kernels"]["ss3_54_fp16"]["mitigated"]["chunked"] = True  # K = 1024 is not above the gate

    for edit in (
        _set(["rows", 5, "kernels", "ss3_54_fp16", "production", "sigma_max"], delta=1e-6),
        _set(["rows", 4, "gram", "fp16", "estimators", "triton_sequential", "diag_bias_mean"], delta=1e-9),
        _set(["rows", 4, "gram", "fp16", "checks", "prescale_bitwise_equal"], False),
        _set(["rows", 1, "cases", 0, "bitwise_equal"], 0),
        _set(["rows", 6, "run_config_hash"], "0" * 16),
        _set(["rows", 6, "k"], 1 << 13),
        _set(["derived", "design_ceilings", "ss3_53"], 1.08),
        _set(["derived", "onset", "wide4", "ps3_55_fp16", "production", "common", "first_k_above"], 4096),
        _set(["protocol", "gate_k"], 4096),
        _set(["protocol", "ceiling"], 1.2),
        _set(["opus_l6", "wide4:ps3_55_fp16", "sigma_max"], [1.0, 1.0]),
        _set(["meta", "git_sha"], "c" * 40),
        lambda p: p["rows"].pop(),
        unpinned,
    ):
        with pytest.raises(ValueError):
            c3.validate_results(_tampered(drift, edit))
    with pytest.raises(ValueError):
        c3.validate_results(drift, canonical=True)


@needs_cuda
def test_quick_summary_and_provenance(quick):
    summary = quick["summary"]
    sources = {stage: quick[stage] for stage in c3.RUN_STAGES}
    c3.validate_results(summary, sources=sources)
    c3.validate_results(summary)
    assert summary["canonical"] is False and "|" in c3.format_tables(summary)
    assert len(summary["wide"]["analysis"]["contrasts"]) == 2 * 5 * 5
    assert len(summary["ckpt"]["analysis"]["contrasts"]) == 18
    assert summary["sources"]["ckpt"]["run_config_hash"] == quick["ckpt"]["run_config_hash"]
    for edit in (
        _set(["ckpt", "analysis", "contrasts", 5, "stats", "p_two_sided"], delta=1e-9),
        _set(["ckpt", "analysis", "contrasts", 7, "p_holm_joint"], delta=1e-9),
        _set(["wide", "analysis", "contrasts", 0, "stats", "ci95", 1], delta=1e-9),
        _set(["wide", "per_seed", "w64", "muon_ss3_53", 0, "tokens_per_second"], delta=1e-6),
        _set(["ckpt", "per_seed", "tiny", "muon_ss3_53", 2, "diverged"], True),
        _set(["statistics", "t_quantile_0p975", "df4"], 2.78),
        lambda p: p["ckpt"]["analysis"]["contrasts"].pop(),
        lambda p: p["sources"].pop("drift"),
    ):
        with pytest.raises(ValueError):
            c3.validate_results(_tampered(summary, edit))
    # Provenance and rows are compared with the sources also when the sources are passed directly.
    for edit in (
        _set(["sources", "wide", "git_sha"], "d" * 40),
        _set(["sources", "ckpt", "code_sha256"], "0" * 64),
        _set(["sources", "ckpt", "run_config_hash"], "0" * 16),
        _set(["sources", "drift", "protocol", "gate_k"], 1 << 20),
        _set(["sources", "drift", "capability"], [9, 0]),
        _set(["sources", "wide", "sha256"], "xyz"),
        _set(["drift", "derived", "mitigation", "chunked_cells"], 0),
        _set(["wide", "models", "w64", "parameters", "total"], delta=1),
    ):
        with pytest.raises(ValueError):
            c3.validate_results(_tampered(summary, edit), sources=sources)
    forged = _tampered(summary, _set(["ckpt", "per_seed", "tiny", "muon_ss3_54", 0, "checkpoints", 2, "test_loss"],
                                     delta=-0.1))
    forged["ckpt"]["analysis"] = json.loads(json.dumps(
        c3.analyze_train("ckpt", forged["ckpt"]["per_seed"], [2, 4, 6])))
    c3.validate_results(forged)  # self-consistent, so only the sources expose it
    with pytest.raises(ValueError, match="ckpt"):
        c3.validate_results(forged, sources=sources)
    with pytest.raises(ValueError):
        c3.validate_results(summary, sources={**sources, "wide": quick["ckpt"]})
    with pytest.raises(ValueError):
        c3.validate_results(summary, canonical=True)
    with pytest.raises(ValueError, match="canonical"):
        c3.run_summary(c3.parse_args(["--stage", "summary", *[
            x for s in c3.RUN_STAGES for x in (f"--{s}-input", str(quick["dir"] / f"{s}.json"))]]))


# ------------------------------------------------------------------ kernels on the GPU
@needs_cuda
def test_exact_model_reproduces_the_triton_syrk_bitwise():
    device = torch.device("cuda")
    for dtype in (torch.float16, torch.bfloat16):
        X = (c3._input([3, 2000], 5, device) / 77.0).to(dtype)
        rows = X.double().tolist()
        with c3.pinned_syrk():
            measured = c3._syrk32(X).double().tolist()
        for i in range(3):
            for j in range(i, 3):
                assert c3.mma_accumulate(rows[i], rows[j]) == measured[i][j], (dtype, i, j)
            assert c3.mma_predict_diag(X[i]) == measured[i][i]


@needs_cuda
def test_chunked_gram_removes_the_long_k_bias_and_is_gated():
    from research.polar_opt import polar_switch_blackwell as psb
    from research.polar_opt import schatten_switch_blackwell as ssb

    device = torch.device("cuda")
    originals = (psb._gram, ssb._launch, ssb._fold_segment1)
    k = 1 << 22
    for shape in ([4, k], [k, 2]):
        G = c3._input(shape, 0, device)
        X, _ = psb._prepare(G, torch.float16)
        view = c3._short(X[0])
        exact = c3.gram64(view)
        mitigation = c3.ChunkedGram(4096, 65536)
        with c3.pinned_syrk():
            sequential = c3._bias(c3._syrk32(view), exact)["diag_bias_mean"]
            chunked = c3._bias(mitigation.gram(view[None])[0], exact)["diag_bias_mean"]
            assert 0.015 < sequential < 0.03 and 0 < chunked < 5e-5  # about 5.6e-9 * K against 2e-5
            odd = view[:, :k - 1234]  # a tail that is not a multiple of the chunk
            assert c3._bias(mitigation.gram(odd[None])[0], c3.gram64(odd))["diag_bias_max"] < 5e-5
            for kernel in ("ps3_55", "ss3_54", "ss3_53"):
                before = mitigation.launches
                with mitigation.patched():
                    Q = c3.kernel_call(kernel, "fp16")(G)
                assert mitigation.launches > before and Q.shape == G.shape and Q.dtype == G.dtype
                A = c3.gram64(c3._short(G))
                values, vectors = torch.linalg.eigh(A)
                quality = c3.polar_quality(Q, G, (vectors / values.sqrt()) @ vectors.mT)
                assert quality["sigma_max"] < c3.design_ceilings([kernel])[kernel] and quality["rel_fro_err"] < 0.09
        assert (psb._gram, ssb._launch, ssb._fold_segment1) == originals  # restored after the block
    # Below the gate the production code runs unchanged: bitwise equal, batched and tall inputs too.
    mitigation = c3.ChunkedGram(4096, 65536)
    for shape in ([64, 256], [2, 64, 256], [256, 64]):
        G = c3._input(shape, 1, device)
        for kernel in ("ps3_55", "ss3_54", "ss3_53"):
            reference = c3.kernel_call(kernel, "fp16")(G)
            with mitigation.patched():
                assert torch.equal(c3.kernel_call(kernel, "fp16")(G), reference)
    assert mitigation.launches == 0
    with pytest.raises(RuntimeError):
        with mitigation.patched():
            raise RuntimeError("the patch is removed on an error")
    assert (psb._gram, ssb._launch, ssb._fold_segment1) == originals


# ------------------------------------------------------------------ committed canonical artifacts
def _committed():
    if not all(p.exists() for p in c3.CANONICAL_OUTPUTS.values()):
        pytest.skip("canonical C4-03 artifacts are not committed in this checkout")
    return {stage: json.loads(path.read_text()) for stage, path in c3.CANONICAL_OUTPUTS.items()}


def test_committed_canonical_artifacts():
    found = _committed()
    c3.validate_results(found["summary"], canonical=True)  # re-reads and hash-checks the three stage files
    for stage in c3.RUN_STAGES:
        payload = found[stage]
        c3.validate_results(payload, canonical=True)
        assert payload["canonical"] is True and payload["quick"] is False and payload["meta"]["git_dirty"] is False
        assert payload["meta"]["environment"]["capability"] >= [12, 0]
        assert payload["protocol"] == {**payload["protocol"], **c3.CANONICAL[stage]}
        assert not c3.resume_log_path(c3.CANONICAL_OUTPUTS[stage]).exists()
    assert len({found[s]["meta"]["git_sha"] for s in c3.STAGES}) == 1
    assert len({found[s]["meta"]["code_sha256"] for s in c3.RUN_STAGES}) == 1
    wide, ckpt, drift = (found[s] for s in c3.RUN_STAGES)
    for payload in (wide, ckpt):
        assert {k: v["sha256"] for k, v in payload["data"]["splits"].items()} == q.EXPECTED_SHA256
        assert payload["derived"]["divergences"] == 0
        assert payload["derived"]["max_peak_vram_reserved_mib"] < 16 * 1024
    assert len(wide["runs"]) == 36 and all(r["steps"] == 64 and r["tokens"] == 64 * 16_384 for r in wide["runs"])
    assert {label: m["config"] for label, m in wide["models"].items()} == {
        label: asdict(config) for label, config in c3.WIDE_CONFIGS.items()}
    assert len(ckpt["runs"]) == 15
    assert all(r["tokens"] == 12_582_912 and r["epoch_fraction"] < 0.5
               and [c["step"] for c in r["checkpoints"]] == [128, 256, 384] for r in ckpt["runs"])
    assert ckpt["protocol"]["eval_tokens_used"]["120m"] == {"val": 1_998_848, "test": 1_998_848}
    assert len(found["summary"]["wide"]["analysis"]["contrasts"]) == 75
    assert len(found["summary"]["ckpt"]["analysis"]["contrasts"]) == 18
    checks = drift["derived"]["checks"]
    for key in ("prescale_bitwise_equal", "syrk_norm_equals_rounded_syrk", "mma_model_sweep_diag_bitwise",
                "mma_model_exact_entries_bitwise", "mma_probe_model_equals_triton"):
        assert checks[key][0] == checks[key][1] > 0, key
    assert all(r["reproduced"] for r in drift["derived"]["reproduction"].values())
    assert set(drift["derived"]["reproduction"]) == set(c3.OPUS_L6)
    mitigation = drift["derived"]["mitigation"]
    assert mitigation["sigma_max_worst"] < 1.10
    assert max(r["measured_max"] for r in drift["derived"]["reproduction"].values()) > 1.10
    assert mitigation["normal_shapes"]["cells"] == mitigation["normal_shapes"]["bitwise_equal"] == 12
    assert mitigation["normal_shapes"]["chunked_launches"] == 0


def test_committed_canonical_artifacts_reject_tampering():
    found = _committed()
    sources = {stage: found[stage] for stage in c3.RUN_STAGES}

    def forged_hash(p):
        p["meta"]["code_sha256"] = "0" * 64
        p["run_config_hash"] = "0" * 16
        for row in p.get("runs", p.get("rows")):
            row["run_config_hash"] = "0" * 16

    for stage, edit in (
        ("summary", _set(["ckpt", "analysis", "contrasts", 2, "p_holm_joint"], delta=1e-9)),
        ("summary", _set(["sources", "ckpt", "sha256"], "0" * 64)),
        ("summary", _set(["meta", "git_dirty"], True)),
        ("summary", _set(["meta", "git_sha"], "0" * 40)),
        ("summary", _set(["quick"], True)),
        ("summary", _set(["sources", "drift", "canonical"], False)),
        ("wide", _set(["meta", "git_dirty"], True)),
        ("wide", _set(["meta", "environment", "capability"], [9, 0])),
        ("wide", _set(["data", "splits", "test", "sha256"], "0" * 64)),
        ("wide", _set(["protocol", "lr_warmup_steps"], 20)),
        ("wide", _set(["protocol", "orthogonalizers", "muon_ss3_54"],
                       "CPU stand-in: baselines.polar_express eager (the real kernel needs CUDA)")),
        ("wide", _set(["runs", 0, "order_in_block"], 3 if found["wide"]["runs"][0]["order_in_block"] != 3 else 2)),
        ("wide", forged_hash),
        ("ckpt", _set(["protocol", "checkpoints"], [128, 256, 320])),
        ("ckpt", _set(["protocol", "seeds"], [1, 2, 3])),
        ("ckpt", _set(["runs", 0, "checkpoints", 1, "test_loss"], delta=1e-9)),
        ("ckpt", forged_hash),
        ("drift", _set(["protocol", "chunk_k"], 8192)),
        ("drift", _set(["rows", 10, "kernels", "ss3_54_fp16", "production", "sigma_max"], delta=1e-6)),
        ("drift", forged_hash),
    ):
        with pytest.raises(ValueError):
            c3.validate_results(_tampered(found[stage], edit), canonical=True)
    for edit in (_set(["sources", "wide", "sha256"], "0" * 64), _set(["meta", "git_sha"], "0" * 40),
                 _set(["sources", "ckpt", "git_sha"], "0" * 40)):
        with pytest.raises(ValueError):
            c3.validate_results(_tampered(found["summary"], edit), canonical=True, sources=sources)
