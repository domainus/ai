"""Checks for the Cycle 4 C4-01a long-budget 120m transfer driver and its committed results."""

import copy
import json
import math
from dataclasses import asdict

import pytest
import torch

from research.polar_opt import bench_c4_01a_120m_long as c4
from research.polar_opt import harness_tier2_real as h
from research.polar_opt import real_text_data as rtd
from research.quant_ptq import micro_rotated_ptq as q
from tests.test_tier2_real import ByteEncoder, _documents

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"
TOKENS_ARGS = ["--stage", "tokens", "--tokens", "20971520", "--lr-warmup-steps", "50", "--compile-model",
               "--update-dtype", "fp32", "--seeds", "1", "2", "3", "--muon-lrs", "0.02", "--muon-aux-lr", "0.001",
               "--adamw-lrs", "0.001", "--eval-tokens", "2000000", "--optimizers", "adamw", "muon_pe_compiled",
               "muon_ps3_blackwell", "muon_ss3_54", "muon_ss3_53"]
WALL_ARGS = ["--stage", "wall", "--seconds", "300", "--wall-warmup-fraction", "0.0625", "--compile-model",
             "--update-dtype", "fp32", "--seeds", "1", "2", "3", "--muon-lrs", "0.02", "--muon-aux-lr", "0.001",
             "--eval-tokens", "2000000", "--optimizers", "muon_pe_compiled", "muon_ps3_blackwell", "muon_ss3_54",
             "muon_ss3_53"]


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


# ------------------------------------------------------------------ CLI and output guards
def test_canonical_arguments_and_default_outputs():
    for argv, stage in ((TOKENS_ARGS, "tokens"), (WALL_ARGS, "wall"), (["--stage", "tokens"], "tokens"),
                        (["--stage", "wall"], "wall"), (["--stage", "wall", "--resume"], "wall")):
        args = c4.parse_args(argv)
        assert args.canonical and args.output == c4.CANONICAL_OUTPUTS[stage], argv
    args = c4.parse_args(TOKENS_ARGS)
    assert (args.preset, args.tokens // h.PRESETS["120m"].step_tokens, args.compile_model) == ("120m", 640, True)
    assert h.PRESETS["120m"] == h.ModelConfig(width=768, layers=12, heads=12, hidden=2048, sequence_length=512,
                                              micro_batch=8, grad_accum=8)
    assert h.PRESETS["120m"].step_tokens == 32_768 and 20_971_520 < 26_004_518
    assert c4.parse_args(["--stage", "summary"]).output == c4.CANONICAL_OUTPUTS["summary"]
    for stage in c4.STAGES:
        quick = c4.parse_args(["--stage", stage, "--quick"])
        assert quick.output == c4.QUICK_OUTPUTS[stage] and not quick.canonical and not c4.reserved(quick.output)
    assert c4.parse_args(["--stage", "tokens", "--quick"]).preset == "tiny"


def test_cli_refuses_reserved_filenames_for_quick_and_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(c4, "run_stage", lambda args: pytest.fail("must not train"))
    monkeypatch.setattr(c4, "run_summary", lambda args: pytest.fail("must not summarize"))
    tokens_name, wall_name = (c4.CANONICAL_OUTPUTS[s].name for s in ("tokens", "wall"))
    other_checkout = tmp_path / "other" / "results" / tokens_name
    alias = tmp_path / "alias.json"
    alias.symlink_to(other_checkout)
    novel = tmp_path / "results" / "cycle4_c4_01a_120m_other.json"
    assert all(c4.reserved(p) for p in (other_checkout, alias, novel, c4.CANONICAL_OUTPUTS["summary"]))
    assert not c4.reserved(tmp_path / "custom.json")
    custom = tmp_path / "custom.json"
    assert c4.parse_args(["--stage", "tokens", "--seeds", "4", "5", "6", "--output", str(custom)]).output == custom
    assert c4.parse_args([*TOKENS_ARGS, "--output", str(other_checkout)]).canonical  # any checkout
    for argv in (
        ["--stage", "tokens", "--quick", "--output", str(c4.CANONICAL_OUTPUTS["tokens"])],
        ["--stage", "tokens", "--quick", "--output", str(other_checkout)],
        ["--stage", "tokens", "--quick", "--output", str(alias)],
        ["--stage", "wall", "--quick", "--output", str(novel)],
        ["--stage", "summary", "--quick", "--output", str(c4.CANONICAL_OUTPUTS["summary"])],
        ["--stage", "tokens", "--seeds", "4", "5", "6"],  # non-canonical needs --output
        ["--stage", "tokens", "--seeds", "4", "5", "6", "--output", str(other_checkout)],
        ["--stage", "tokens", "--tokens", "8388608", "--output", str(alias)],
        ["--stage", "tokens", "--lr-warmup-steps", "20", "--output", str(other_checkout)],
        ["--stage", "tokens", "--no-compile-model", "--output", str(other_checkout)],
        ["--stage", "tokens", "--update-dtype", "native", "--output", str(other_checkout)],
        ["--stage", "tokens", "--eval-tokens", "1048576", "--output", str(other_checkout)],
        ["--stage", "tokens", "--device", "cpu", "--output", str(other_checkout)],
        ["--stage", "tokens", "--no-verify", "--output", str(other_checkout)],
        ["--stage", "tokens", "--preset", "50m", "--output", str(other_checkout)],
        ["--stage", "tokens", "--optimizers", "adamw", "muon_pe_compiled", "--output", str(other_checkout)],
        ["--stage", "wall", "--seconds", "100", "--output", str(tmp_path / wall_name)],
        ["--stage", "wall", "--muon-lrs", "0.01", "--output", str(c4.CANONICAL_OUTPUTS["wall"])],
        [*TOKENS_ARGS, "--output", str(tmp_path / wall_name)],  # the other stage's reserved name
        [*WALL_ARGS, "--output", str(novel)],
        ["--stage", "tokens", "--seconds", "300"],  # budget flags belong to their stage
        ["--stage", "wall", "--tokens", "20971520"],
        ["--stage", "summary", "--seeds", "1", "2", "3"],
        ["--stage", "tokens", "--tokens", "1000", "--output", str(custom)],  # not a multiple of 32,768
        ["--stage", "tokens", "--optimizers", "sgd", "--output", str(custom)],
    ):
        with pytest.raises(SystemExit) as exc:
            c4.main(argv)
        assert exc.value.code == 2, argv
    assert not other_checkout.exists() and not custom.exists() and not novel.exists()


def test_canonical_runs_need_sm_120_and_a_clean_tree(tmp_path, monkeypatch):
    other = tmp_path / "elsewhere" / c4.CANONICAL_OUTPUTS["tokens"].name
    alias = tmp_path / "alias.json"
    alias.symlink_to(other)
    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for capability, allowed in (((12, 0), True), ((12, 1), True), ((13, 0), True), ((10, 0), False), ((9, 0), False)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, c=capability: c)
        for path in (c4.CANONICAL_OUTPUTS["wall"], other, alias):
            if allowed:
                c4.require_blackwell(cuda, path)
            else:
                with pytest.raises(RuntimeError, match="sm_120"):
                    c4.require_blackwell(cuda, path)
        if not allowed:
            with pytest.raises(RuntimeError, match="sm_120"):
                c4.require_blackwell(cuda, tmp_path / "custom.json", canonical=True)
        c4.require_blackwell(cuda, tmp_path / "custom.json")
    for path in (c4.CANONICAL_OUTPUTS["tokens"], other, alias):
        with pytest.raises(RuntimeError, match="sm_120"):
            c4.require_blackwell(cpu, path)
    c4.require_blackwell(cpu, tmp_path / "custom.json")
    # ``run_stage`` applies both guards before it reads data or trains.
    monkeypatch.setattr(q, "load_data", lambda *a, **k: pytest.fail("guards must run first"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(RuntimeError, match="sm_120"):
        c4.run_stage(c4.parse_args([*TOKENS_ARGS, "--output", str(other)]))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    monkeypatch.setattr(h, "git_info", lambda: {"git_sha": "a" * 40, "git_dirty": True})
    with pytest.raises(RuntimeError, match="clean"):
        c4.run_stage(c4.parse_args([*WALL_ARGS, "--output", str(tmp_path / "w.json")]))
    assert not other.exists()


def test_atomic_json_is_strict_lossless_and_follows_symlinks(tmp_path):
    payload = {"a": [0.1, 1e-300, 5.396465571794642, -0.0], "b": {"n": None, "s": "x", "f": True}}
    assert c4.write_json_atomic(tmp_path / "x.json", payload) == json.loads(json.dumps(payload))
    assert json.loads((tmp_path / "x.json").read_text()) == payload
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            c4.write_json_atomic(tmp_path / "bad.json", {"v": bad})
    assert not (tmp_path / "bad.json").exists()
    target = tmp_path / "target.json"
    (tmp_path / "link.json").symlink_to(target)
    c4.write_json_atomic(tmp_path / "link.json", payload)
    assert (tmp_path / "link.json").is_symlink() and json.loads(target.read_text()) == payload
    assert c4.resume_log_path(tmp_path / "r.json") == (tmp_path / "r.jsonl").resolve()


def test_step_clock_reproduces_run_trial_step_times():
    clock = c4.StepClock()
    starts = []
    for _ in range(4):
        starts.append(clock())
        clock()
    assert clock.durations() == [clock.calls[2 * i + 1] - starts[i] for i in range(4)]


# ------------------------------------------------------------------ statistics
def test_t_quantile_is_exact_for_two_degrees_of_freedom():
    from scipy.stats import t

    assert c4.T975_DF2 == pytest.approx(t.ppf(0.975, 2), rel=1e-14)
    assert c4.T975_DF2 == pytest.approx(math.sqrt(2 * 0.9025 / 0.0975), rel=1e-15)


@pytest.mark.parametrize("values,null", [([0.010, 0.012, 0.009], 0.0), ([-0.3, 0.1, 0.25], 0.0),
                                         ([1.029, 1.030, 1.0295], 1.0), ([1.0, 1.0001, 0.9998], 1.0)])
def test_paired_t_matches_scipy_and_the_closed_form(values, null):
    from scipy import stats

    found = c4.paired_t_df2(values, null)
    ref = stats.ttest_1samp(values, null)
    assert found["t"] == pytest.approx(ref.statistic, rel=1e-9)
    assert found["p_two_sided"] == pytest.approx(ref.pvalue, rel=1e-10)
    assert found["p_two_sided"] == pytest.approx(1 - abs(found["t"]) / math.sqrt(found["t"] ** 2 + 2), rel=1e-12)
    low, high = stats.t.interval(0.95, 2, loc=found["mean"], scale=stats.sem(values))
    assert found["ci95"] == pytest.approx([low, high], rel=1e-12, abs=1e-15)
    assert found["df"] == 2 and found["n"] == 3 and found["null"] == null


def test_paired_t_edge_cases_and_stability():
    exact = c4.paired_t_df2([1.0, 1.0, 1.0], 1.0)
    assert exact["t"] is None and exact["p_two_sided"] == 1.0 and exact["sd"] == 0.0
    assert c4.paired_t_df2([2.0, 2.0, 2.0], 0.0)["p_two_sided"] == 0.0
    huge = c4.paired_t_df2([1.0, 1.0 + 1e-9, 1.0 - 1e-9], 0.0)  # |t| ~ 1.7e9
    t = huge["t"]
    assert 0 < huge["p_two_sided"] == pytest.approx(1 / t ** 2, rel=1e-6)  # the naive 1 - |t|/s gives 0
    with pytest.raises(ValueError, match="three"):
        c4.paired_t_df2([1.0, 2.0])


def test_holm_bonferroni():
    assert c4.holm_bonferroni([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert c4.holm_bonferroni([0.5, 0.6]) == [1.0, 1.0]
    assert c4.holm_bonferroni([0.02, 0.02]) == [0.04, 0.04]
    assert c4.holm_bonferroni([]) == []


def _synthetic_per_seed():
    rows = {}
    for b, budget in enumerate(("tokens", "wall")):
        names = ["adamw", *c4.MUON] if budget == "tokens" else list(c4.MUON)
        rows[budget] = {n: [{"seed": s, "tokens_per_second": 80_000 + 900 * i + 37 * s * (i + 1) + b,
                             "test_loss": 5.5 - 0.01 * i + 0.003 * s * i, "val_loss": 5.4, "steps": 640,
                             "tokens": 640 * 32_768, "epoch_fraction": 0.8, "train_seconds": 250.0,
                             "init_fingerprint": f"i{s}", "stream_fingerprint": f"s{s}"} for s in (1, 2, 3)]
                        for i, n in enumerate(names)}
    return rows


def test_analysis_runs_every_predeclared_contrast_and_holm_family():
    analysis = c4.analyze(_synthetic_per_seed())
    contrasts = analysis["contrasts"]
    assert len(contrasts) == 18 + 12 + 12 == len({c["id"] for c in contrasts})
    sizes = analysis["holm_group_sizes"]
    assert sizes["p_holm_reference_joint"] == {"muon_pe_compiled": 18, "muon_ps3_blackwell": 12, "adamw": 12}
    assert sizes["p_holm_muon_joint"] == {"muon": 30}
    assert set(sizes["p_holm_family"].values()) == {2, 3, 4} and len(sizes["p_holm_family"]) == 6 + 6 + 3
    for c in contrasts:
        p = c["stats"]["p_two_sided"]
        assert p <= c["p_holm_family"] <= c["p_holm_reference_joint"] <= 1
        if c["reference"] == "adamw":
            assert c["p_holm_muon_joint"] is None and c["budget"] == "tokens"
        else:
            assert c["p_holm_reference_joint"] <= c["p_holm_muon_joint"] <= 1
    ratio = next(c for c in contrasts if c["id"] == "wall:tokens_per_second_ratio:muon_ss3_54_vs_muon_ps3_blackwell")
    table = _synthetic_per_seed()["wall"]
    assert ratio["stats"]["values"] == [a["tokens_per_second"] / b["tokens_per_second"]
                                        for a, b in zip(table["muon_ss3_54"], table["muon_ps3_blackwell"])]
    assert ratio["stats"]["null"] == 1.0
    unpaired = _synthetic_per_seed()
    unpaired["wall"]["muon_ss3_53"][1]["stream_fingerprint"] = "other"
    with pytest.raises(ValueError, match="paired"):
        c4.analyze(unpaired)


# ------------------------------------------------------------------ quick end-to-end pipeline
@pytest.fixture(scope="module")
def quick(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("c4_01a_quick")
    rtd.build_shards(_documents(), tmp / "shards", targets={"train": 60_000, "val": 6_000, "test": 6_000},
                     source={"name": "synthetic"}, encoder=ByteEncoder())
    common = ["--quick", "--no-verify", "--silent", "--data-dir", str(tmp / "shards"), "--device", DEVICE]
    out = {stage: c4.main(["--stage", stage, *common, "--output", str(tmp / f"{stage}.json")])
           for stage in ("tokens", "wall")}
    out["summary"] = c4.main(["--stage", "summary", "--quick", "--silent", "--tokens-input", str(tmp / "tokens.json"),
                              "--wall-input", str(tmp / "wall.json"), "--output", str(tmp / "summary.json")])
    return {"dir": tmp, "common": common, **out}


def test_quick_pipeline_end_to_end(quick):
    tokens, wall, summary = quick["tokens"], quick["wall"], quick["summary"]
    for stage, payload in (("tokens", tokens), ("wall", wall)):
        c4.validate_results(payload)
        assert payload["stage"] == stage and payload["quick"] and not payload["canonical"]
        assert not c4.resume_log_path(quick["dir"] / f"{stage}.json").exists()  # removed after the final write
        assert payload["derived"]["divergences"] == 0 and payload["meta"]["numerics"]["allow_tf32"] is False
    assert len(tokens["runs"]) == 15 and len(wall["runs"]) == 12
    assert all(r["steps"] == 8 and r["epoch_fraction"] < 1 for r in tokens["runs"])
    assert all(r["wall_budget_overshoot_seconds"] == 0.0 and r["train_seconds"] <= 0.5 for r in wall["runs"])
    for payload in (tokens, wall):
        by_seed = {}
        for r in payload["runs"]:
            by_seed.setdefault(r["seed"], set()).add((r["init_fingerprint"], r["stream_fingerprint"]))
        assert all(len(v) == 1 for v in by_seed.values()) and len(by_seed) == 3
    c4.validate_results(summary, sources={"tokens": tokens, "wall": wall})
    c4.validate_results(summary)
    assert summary["canonical"] is False and len(summary["analysis"]["contrasts"]) == 42
    assert summary["checks"]["wall"]["max_wall_budget_overshoot_seconds"] == 0.0
    assert "|" in c4.format_tables(summary)
    if CUDA:
        assert "Triton" in tokens["protocol"]["orthogonalizers"]["muon_ss3_54"]


def test_resume_reuses_finished_rows_and_removes_the_log(quick, tmp_path):
    out = tmp_path / "resumed.json"
    log = c4.resume_log_path(out)
    rows = quick["tokens"]["runs"]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows[:4]) + '{"cut off')
    payload = c4.main(["--stage", "tokens", *quick["common"], "--resume", "--output", str(out)])
    assert payload["meta"]["resume"] == {"resume": True, "rows_loaded": 4, "rows_run": 11}
    assert not log.exists() and len(payload["runs"]) == 15
    c4.validate_results(payload)
    # Double-interruption regression (M3): torn tail must be repaired on disk before any later append.
    log.write_text("".join(json.dumps(r) + "\n" for r in rows[:2]) + '{"torn')
    loaded = c4.load_rows(log, quick["tokens"]["run_config_hash"], resume=True)
    assert len(loaded) == 2 and log.read_text().endswith("\n") and "torn" not in log.read_text()
    with log.open("a") as handle:
        handle.write(json.dumps(rows[2]) + "\n" + '{"torn2')
    resumed_twice = c4.main(["--stage", "tokens", *quick["common"], "--resume", "--output", str(out)])
    assert resumed_twice["meta"]["resume"] == {"resume": True, "rows_loaded": 3, "rows_run": 12}
    assert not log.exists() and len(resumed_twice["runs"]) == 15
    c4.validate_results(resumed_twice)
    log.write_text(json.dumps({**rows[0], "run_config_hash": "0" * 16}) + "\n")
    with pytest.raises(ValueError, match="resume"):
        c4.main(["--stage", "tokens", *quick["common"], "--resume", "--output", str(out)])
    log.unlink()


def test_validate_results_rejects_tampered_runs(quick):
    tokens, wall = quick["tokens"], quick["wall"]
    with pytest.raises(ValueError):
        c4.validate_results(tokens, canonical=True)  # a quick result never passes as canonical
    edits = [
        (tokens, _set(["runs", 0, "test_loss"], delta=1e-9)),
        (tokens, _set(["runs", 3, "train_seconds"], delta=1e-9)),
        (tokens, _set(["runs", 3, "step_wall_seconds", 2], delta=1e-6)),
        (tokens, _set(["runs", 1, "tokens_per_second"], delta=1e-6)),
        (tokens, _set(["runs", 2, "tokens"], delta=tokens["step_tokens"])),
        (tokens, _set(["runs", 2, "steps"], 7)),
        (tokens, _set(["runs", 4, "epoch_fraction"], delta=1e-9)),
        (tokens, _set(["runs", 5, "diverged"], True)),
        (tokens, _set(["runs", 6, "init_fingerprint"], "0" * 16)),
        (tokens, _set(["runs", 6, "run_config_hash"], "0" * 16)),
        (tokens, _set(["runs", 7, "timing", "step_wall_ms"], delta=1e-6)),
        (tokens, _set(["derived", "per_optimizer", "adamw", "test_loss_mean"], delta=1e-9)),
        (tokens, _set(["meta", "numerics", "allow_tf32"], True)),
        (tokens, _set(["runs", 0, "val_ppl"], math.nan)),
        (tokens, lambda p: p["runs"].pop()),
        (wall, _set(["runs", 0, "wall_budget_overshoot_seconds"], 0.01)),
        (wall, _set(["runs", 1, "discarded_step_seconds"], None)),
        (wall, _set(["runs", 2, "step_wall_seconds"], [])),
        (wall, _set(["derived", "max_epoch_fraction"], delta=1e-9)),
    ]
    for payload, edit in edits:
        with pytest.raises(ValueError):
            c4.validate_results(_tampered(payload, edit))
    c4.validate_results(tokens)  # the fixture itself is untouched


def test_validate_results_rejects_tampered_summaries(quick):
    summary, sources = quick["summary"], {"tokens": quick["tokens"], "wall": quick["wall"]}
    for edit in (
        _set(["analysis", "contrasts", 5, "stats", "p_two_sided"], delta=1e-9),
        _set(["analysis", "contrasts", 31, "p_holm_family"], delta=1e-9),
        _set(["analysis", "contrasts", 0, "stats", "ci95", 1], delta=1e-9),
        _set(["analysis", "per_optimizer", "wall", "muon_ss3_54", "test_loss", "mean"], delta=1e-9),
        _set(["per_seed", "tokens", "muon_ss3_53", 0, "test_loss"], delta=1e-9),
        _set(["per_seed", "wall", "muon_ss3_53", 2, "tokens_per_second"], delta=1e-6),
        _set(["per_seed", "wall", "muon_ss3_53", 2, "wall_budget_overshoot_seconds"], 0.1),
        _set(["statistics", "t_quantile_0p975_df2"], 4.3),
        lambda p: p["analysis"]["contrasts"].pop(),
    ):
        with pytest.raises(ValueError):
            c4.validate_results(_tampered(summary, edit))
    # Rows that differ from the source files are rejected even when the analysis is re-derived.
    forged = _tampered(summary, _set(["per_seed", "wall", "muon_ss3_54", 0, "test_loss"], delta=-0.1))
    forged["per_seed"]["wall"]["muon_ss3_54"][0]["test_ppl"] = math.exp(forged["per_seed"]["wall"]["muon_ss3_54"][0]
                                                                         ["test_loss"])
    forged["analysis"] = c4.analyze(forged["per_seed"])
    c4.validate_results(forged)
    with pytest.raises(ValueError, match="per_seed"):
        c4.validate_results(forged, sources=sources)
    with pytest.raises(ValueError):
        c4.validate_results(summary, canonical=True)


# ------------------------------------------------------------------ committed canonical artifacts
def _committed():
    paths = c4.CANONICAL_OUTPUTS
    if not all(p.exists() for p in paths.values()):
        pytest.skip("canonical C4-01a artifacts are not committed in this checkout")
    return {stage: json.loads(path.read_text()) for stage, path in paths.items()}


def test_committed_canonical_artifacts():
    found = _committed()
    c4.validate_results(found["summary"], canonical=True)  # re-reads and hash-checks both stage files
    for stage in ("tokens", "wall"):
        payload = found[stage]
        c4.validate_results(payload, canonical=True)
        assert payload["canonical"] is True and payload["quick"] is False and payload["meta"]["git_dirty"] is False
        assert payload["meta"]["environment"]["capability"] >= [12, 0]
        assert {k: v["sha256"] for k, v in payload["data"]["splits"].items()} == q.EXPECTED_SHA256
        assert payload["config"] == asdict(h.PRESETS["120m"]) and payload["step_tokens"] == 32_768
        assert payload["derived"]["divergences"] == 0
        assert not c4.resume_log_path(c4.CANONICAL_OUTPUTS[stage]).exists()
    assert len(found["tokens"]["runs"]) == 15 and len(found["wall"]["runs"]) == 12
    assert all(r["steps"] == 640 and r["tokens"] == 20_971_520 and r["epoch_fraction"] < 1
               for r in found["tokens"]["runs"])
    assert all(r["wall_budget_overshoot_seconds"] == 0.0 and r["train_seconds"] <= 300.0
               for r in found["wall"]["runs"])
    assert found["tokens"]["meta"]["git_sha"] == found["wall"]["meta"]["git_sha"]
    assert len(found["summary"]["analysis"]["contrasts"]) == 42


def test_committed_canonical_artifacts_reject_tampering():
    found = _committed()
    sources = {"tokens": found["tokens"], "wall": found["wall"]}

    def _forged_hash_all_rows(p):
        p["meta"]["code_sha256"] = "0" * 64
        p["run_config_hash"] = "0" * 16
        for row in p["runs"]:
            row["run_config_hash"] = "0" * 16

    for stage, edit in (
        ("summary", _set(["analysis", "contrasts", 2, "p_holm_muon_joint"], delta=1e-9)),
        ("summary", _set(["sources", "wall", "sha256"], "0" * 64)),
        ("summary", _set(["meta", "git_dirty"], True)),
        ("summary", _set(["meta", "git_sha"], "0" * 40)),
        ("summary", _set(["config", "width"], 123)),
        ("summary", _set(["quick"], True)),
        ("tokens", _set(["meta", "git_dirty"], True)),
        ("tokens", _set(["meta", "environment", "capability"], [9, 0])),
        ("tokens", _set(["data", "splits", "test", "sha256"], "0" * 64)),
        ("tokens", _set(["protocol", "lr_warmup_steps"], 20)),
        ("tokens", _set(["protocol", "orthogonalizers", "muon_ss3_54"],
                         "CPU stand-in: baselines.polar_express eager (the real kernel needs CUDA)")),
        ("tokens", _set(["runs", 0, "order_in_block"], 4)),
        ("tokens", _forged_hash_all_rows),
        ("wall", _set(["protocol", "seconds"], 100.0)),
        ("wall", _set(["runs", 0, "test_loss"], delta=1e-9)),
        ("wall", _set(["runs", 0, "order_in_block"], 3)),
        ("wall", _forged_hash_all_rows),
    ):
        with pytest.raises(ValueError):
            c4.validate_results(_tampered(found[stage], edit), canonical=True)

    # Also test explicit ``sources=...`` summary overload catches M1 tampering and H1 binding.
    for edit in (
        _set(["sources", "wall", "sha256"], "0" * 64),
        _set(["config", "width"], 123),
        _set(["meta", "git_sha"], "0" * 40),
    ):
        with pytest.raises(ValueError):
            c4.validate_results(_tampered(found["summary"], edit), canonical=True, sources=sources)
    tampered_wall = copy.deepcopy(found["wall"])
    tampered_wall["runs"][0]["test_loss"] -= 0.01
    tampered_wall["runs"][0]["test_ppl"] = math.exp(tampered_wall["runs"][0]["test_loss"])
    tampered_wall["derived"] = c4.derive_runs(tampered_wall)
    forged_summary = copy.deepcopy(found["summary"])
    forged_summary["per_seed"]["wall"] = c4.per_seed_table(tampered_wall)
    forged_summary["checks"]["wall"] = c4.stage_checks(tampered_wall)
    forged_summary["analysis"] = c4.analyze(forged_summary["per_seed"])
    with pytest.raises(ValueError, match="canonical file"):
        c4.validate_results(forged_summary, canonical=True, sources={"tokens": found["tokens"], "wall": tampered_wall})
