"""Checks for the Cycle 4 C4-01b iso-loss Muon-vs-AdamW PTQ benchmark.

Selection, statistics, validation and guard tests are pure Python. The pipeline tests train the
``tiny`` preset on synthetic shards; on an sm_100+ GPU they include real hardware W4A4 rows.
"""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats

pytest.importorskip("triton")  # the benchmark imports the Blackwell NVFP4 kernels

from research.polar_opt import harness_tier2_real as h  # noqa: E402
from research.polar_opt.harness_tier2_lm import Transformer  # noqa: E402
from research.quant_ptq import bench_isoloss_ptq as iso  # noqa: E402
from research.quant_ptq import micro_rotated_ptq as q  # noqa: E402
from tests.test_micro_rotated_ptq import _fake_data  # noqa: E402

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"
FP4_GEMM = CUDA and hasattr(F, "scaled_mm") and torch.cuda.get_device_capability()[0] >= 10
ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "cycle4_c4_01b_isoloss_ptq.json"
SELECT = dict(tolerance=0.02, t_muon_candidates=[320, 256, 224, 192], adamw_max_steps=768)
SEARCH = dict(t_muon_candidates=[320, 256, 224, 192], muon_lr=0.02, adamw_lrs=[0.001, 0.0015],
              adamw_max_steps=768, step_quantum=8, tolerance=0.02)


# ------------------------------------------------------------------ Stage A selection (pure)
def _row(optimizer, steps, lr, val):
    return {"optimizer": optimizer, "steps": steps, "lr": lr, "val_nll": val, "tokens": steps * 32768}


def test_selection_uses_t_muon_320_when_adamw_matches_it():
    rows = [_row(iso.MUON, 320, 0.02, 5.40), _row(iso.MUON, 256, 0.02, 5.60),
            _row(iso.ADAMW, 768, 0.001, 5.43), _row(iso.ADAMW, 768, 0.0015, 5.385),
            _row(iso.ADAMW, 640, 0.0015, 5.405), _row(iso.ADAMW, 400, 0.001, 5.60)]
    s = iso.select_isoloss(rows, **SELECT)
    assert (s["matched"], s["t_muon"], s["t_adamw"], s["adamw_lr"]) == (True, 320, 640, 0.0015)
    assert s["abs_val_gap"] == abs(5.40 - 5.405) and s["val_gap_muon_minus_adamw"] == 5.40 - 5.405
    assert s["adamw_over_muon_tokens"] == 2.0


def test_selection_falls_back_to_the_largest_matchable_t_muon():
    rows = [_row(iso.MUON, 320, 0.02, 5.40), _row(iso.MUON, 256, 0.02, 5.52), _row(iso.MUON, 224, 0.02, 5.61),
            _row(iso.MUON, 192, 0.02, 5.75), _row(iso.ADAMW, 768, 0.001, 5.60), _row(iso.ADAMW, 768, 0.0015, 5.66),
            _row(iso.ADAMW, 384, 0.001, 5.76)]
    s = iso.select_isoloss(rows, **SELECT)
    assert (s["matched"], s["t_muon"], s["t_adamw"], s["adamw_lr"]) == (True, 224, 768, 0.001)
    assert s["t_muon_candidates_evaluated"] == [320, 256, 224, 192]
    # Exactly at the tolerance still matches; just beyond it does not.
    edge = [_row(iso.MUON, 320, 0.02, 5.0), _row(iso.ADAMW, 768, 0.001, 5.02)]
    assert iso.select_isoloss(edge, **SELECT)["matched"] is (abs(5.0 - 5.02) <= 0.02)
    assert iso.select_isoloss([edge[0], _row(iso.ADAMW, 768, 0.001, 5.0201)], **SELECT)["matched"] is False


def test_selection_tie_breaks_budget_and_unmatched_fallback():
    muon = _row(iso.MUON, 320, 0.02, 5.5)
    # Equal |difference| (1/128, exact in binary): fewer steps first, then the lower LR. A row above
    # the step budget is ignored.
    rows = [muon, _row(iso.ADAMW, 704, 0.0015, 5.5078125), _row(iso.ADAMW, 640, 0.0015, 5.4921875),
            _row(iso.ADAMW, 640, 0.001, 5.4921875), _row(iso.ADAMW, 800, 0.001, 5.5)]
    s = iso.select_isoloss(rows, **SELECT)
    assert (s["t_adamw"], s["adamw_lr"]) == (640, 0.001)
    # No match anywhere: the closest pair is returned and flagged.
    far = [muon, _row(iso.MUON, 192, 0.02, 5.75), _row(iso.ADAMW, 768, 0.001, 5.90)]  # 5.5 is 0.4 away
    s = iso.select_isoloss(far, **SELECT)
    assert (s["matched"], s["t_muon"], s["t_adamw"]) == (False, 192, 768)
    assert s["abs_val_gap"] == pytest.approx(0.15)
    with pytest.raises(ValueError):
        iso.select_isoloss([muon], **SELECT)


def _search(muon_val, adamw_val):
    """Run the search on analytic val-NLL curves; returns (pilot rows, call log, search log)."""
    calls, memo, search = [], {}, []

    def run(optimizer, steps, lr):
        if (optimizer, steps, lr) not in memo:
            calls.append((optimizer, steps, lr))
            memo[(optimizer, steps, lr)] = muon_val(steps) if optimizer == iso.MUON else adamw_val(steps, lr)
        return memo[(optimizer, steps, lr)]

    iso.calibration_search(run, search=search, **SEARCH)
    return [_row(o, s, lr, v) for (o, s, lr), v in memo.items()], calls, search


def test_search_interpolates_adamw_steps_when_adamw_overshoots():
    rows, calls, search = _search(lambda t: 5.4, lambda s, lr: 5.0 + 250.0 / s + (0.03 if lr == 0.001 else 0.0))
    s = iso.select_isoloss(rows, **SELECT)
    assert (s["matched"], s["t_muon"], s["adamw_lr"]) == (True, 320, 0.0015) and s["abs_val_gap"] <= 0.01
    assert s["t_adamw"] == 632  # 5.0 + 250 / T = 5.4 at T = 625
    adamw = [c for c in calls if c[0] == iso.ADAMW]
    assert all(c[1] % 8 == 0 and 0 < c[1] <= 768 for c in adamw)
    assert [c[1] for c in calls if c[0] == iso.MUON] == [320]  # a smaller T_muon is not needed
    # Two full-budget runs, then only the better LR: the midpoint from 0 steps, then interpolation.
    assert [c[1:] for c in adamw] == [(768, 0.001), (768, 0.0015), (384, 0.0015), (680, 0.0015),
                                      (648, 0.0015), (632, 0.0015)]
    assert [(e["adamw_steps"], e["bracket_steps"]) for e in search] == [
        (384, [0, 768]), (680, [384, 768]), (648, [384, 680]), (632, [384, 648])]
    assert abs(search[-1]["adamw_minus_muon_val_nll"]) <= 0.01 and iso.non_monotone_trials(rows, 0.02) == []


def test_search_steps_down_t_muon_when_adamw_cannot_reach_it():
    muon = {320: 5.40, 256: 5.52, 224: 5.61, 192: 5.75}
    rows, calls, _ = _search(muon.__getitem__, lambda s, lr: 5.2 + 300.0 / s + (0.0 if lr == 0.001 else 0.05))
    s = iso.select_isoloss(rows, **SELECT)
    # AdamW at 768 steps (LR 0.001) has 5.5906: out of reach of T_muon 320 and 256 (no extra trial),
    # 0.0194 better than T_muon 224, so its steps are refined to within half the tolerance.
    assert (s["matched"], s["t_muon"], s["t_adamw"], s["adamw_lr"]) == (True, 224, 744, 0.001)
    assert s["abs_val_gap"] <= 0.01
    assert [c[1] for c in calls if c[0] == iso.MUON] == [320, 256, 224]
    assert [c[1:] for c in calls if c[0] == iso.ADAMW] == [(768, 0.001), (768, 0.0015), (384, 0.001),
                                                           (752, 0.001), (744, 0.001)]


def test_search_survives_unstable_adamw_trials():
    # Two trials near the crossing (T = 527) are 0.25 nats worse than their neighbours, as an
    # unstable AdamW run is. The search must not treat them as "more steps are needed".
    def adamw(steps, lr):
        return 6.24 - 0.0017 * steps + (0.03 if lr == 0.0015 else 0.0) + (0.25 if steps in (528, 536) else 0.0)

    rows, calls, search = _search(lambda t: 5.3446, adamw)
    s = iso.select_isoloss(rows, **SELECT)
    assert (s["matched"], s["t_muon"], s["t_adamw"], s["adamw_lr"]) == (True, 320, 520, 0.001)
    assert [e["adamw_steps"] for e in search] == [384, 528, 520, 536, 544]
    unstable = iso.non_monotone_trials(rows, 0.02)
    assert [(u["steps"], u["better_trial_steps"]) for u in unstable] == [(528, 520), (536, 520)]
    assert all(u["val_nll_excess"] > 0.2 and u["adamw_lr"] == 0.001 for u in unstable)
    # A steep curve whose crossing has no grid trial within the tolerance: bounded, no match for 320.
    rows, calls, _ = _search(lambda t: 5.40 if t == 320 else 9.0,
                             lambda steps, lr: 7.0 - 0.02 * (steps - 4) + (0.3 if lr == 0.0015 else 0.0))
    assert len([c for c in calls if c[0] == iso.ADAMW]) <= 2 + 4 * 2 * 10
    assert iso.select_isoloss(rows, **SELECT)["matched"] is False


def test_search_reports_no_match_after_all_candidates():
    rows, calls, _ = _search(lambda t: 5.0 + t / 3200.0, lambda s, lr: 6.5)
    assert [c[1] for c in calls if c[0] == iso.MUON] == [320, 256, 224, 192]
    assert iso.select_isoloss(rows, **SELECT)["matched"] is False


# ------------------------------------------------------------------ Stage A test.bin isolation
def _quick_args(tmp_path, *extra):
    data_dir = tmp_path / "shards"
    data_dir.mkdir(exist_ok=True)
    _fake_data(data_dir, sizes=(200_000, 20_000, 20_000))
    return iso.parse_args(["--quick", "--no-verify", "--silent", "--data-dir", str(data_dir), "--device", DEVICE,
                           "--output", str(tmp_path / "quick.json"), *extra])


def test_split_ledger_locks_and_counts():
    ledger = iso.SplitLedger({"train": 1, "val": 2, "test": 3})
    assert ledger["test"] == 3 and ledger.reads == {"train": 0, "val": 0, "test": 1}
    assert ledger.get("val") == 2 and ledger.get("missing", 99) == 99
    assert ledger.reads == {"train": 0, "val": 1, "test": 1}
    with ledger.forbid("test"):
        assert ledger["val"] == 2 and ledger.get("val") == 2
        assert ledger.get("missing", 99) == 99
        for call in (
            lambda: ledger["test"],
            lambda: ledger.get("test"),
            lambda: ledger.values(),
            lambda: ledger.items(),
            lambda: ledger.copy(),
        ):
            with pytest.raises(RuntimeError, match="test.bin is locked"):
                call()
    for mut in (lambda: ledger.pop("test"), lambda: ledger.popitem(), lambda: ledger.setdefault("x", 0)):
        with pytest.raises(RuntimeError, match="mutating split access"):
            mut()
    assert ledger.values() == [1, 2, 3]
    assert ledger.items() == [("train", 1), ("val", 2), ("test", 3)]
    assert ledger.copy() == {"train": 1, "val": 2, "test": 3}
    assert ledger["test"] == 3 and ledger.reads == {"train": 3, "val": 6, "test": 5}


def test_stage_a_never_reads_test_bin(tmp_path):
    args = _quick_args(tmp_path)
    args.t_muon_candidates, args.adamw_max_steps = [24], 32
    protocol, config, device = iso.build_protocol(args), h.PRESETS["tiny"], torch.device(DEVICE)
    data = iso.load_data(args)
    assert isinstance(data.splits, iso.SplitLedger)
    # A data object without any test split: Stage A cannot depend on it.
    blind = SimpleNamespace(splits=iso.SplitLedger({k: data.splits[k] for k in ("train", "val")}))
    pilot = iso.stage_a_pilot(config, blind, device, protocol, iso.UnitLog(None, "h", False), lambda *a: None)
    assert pilot["split_reads"] == {"train": 1, "val": 1} and pilot["splits_used"] == ["train", "val"]
    assert {r["optimizer"] for r in pilot["rows"]} == {iso.MUON, iso.ADAMW}
    for row in pilot["rows"]:
        assert row["seed"] == 0 and row["evaluated_splits"] == ["val"] and row["val_tokens"] == 64 * 64
        assert row["warmup_steps"] == max(20, row["steps"] // 10) and len(row["checkpoint_sha256"]) == 64
        assert not any("test" in key for key in row)
    assert pilot["selection"] == iso.select_isoloss(
        pilot["rows"], tolerance=0.02, t_muon_candidates=[24], adamw_max_steps=32)
    # With the real ledger the test split is locked during the stage and free again after it.
    before = data.splits.reads["test"]
    same = iso.stage_a_pilot(config, data, device, protocol, iso.UnitLog(None, "h", False), lambda *a: None)
    assert same["split_reads"]["test"] == 0 and data.splits.reads["test"] == before
    assert data.splits.locked == frozenset()
    # Same seed, init and batch stream for every pilot trial of the stage.
    assert len({r["init_fingerprint"] for r in pilot["rows"] + same["rows"]}) == 1
    assert len({r["stream_fingerprint"] for r in pilot["rows"] + same["rows"]}) == 1


# ------------------------------------------------------------------ exact df = 2 statistics
def test_t_quantile_is_exact_for_two_degrees_of_freedom():
    assert iso.T975_DF2 == 4.302652729911275  # the constant fixed by the protocol
    # The closed form solves CDF(t) = 1/2 + t / (2 sqrt(t^2 + 2)) = 0.975; current SciPy agrees with it.
    closed = iso.T975_DF2_CLOSED_FORM
    assert abs(0.5 + closed / (2 * math.sqrt(closed * closed + 2)) - 0.975) < 1e-15
    assert abs(closed - scipy_stats.t.ppf(0.975, 2)) < 1e-9
    assert 1e-10 < iso.T975_DF2 - closed < 2e-10  # the protocol constant is 1.6e-10 larger


@pytest.mark.parametrize("diffs", [[0.011, 0.014, 0.009], [-0.3, 0.2, 0.05], [1e-6, 2e-6, 3.5e-6],
                                   [5.0, 5.0001, 4.9999], [-0.02, -0.021, 0.4], [0.0, 0.0, 1.0]])
def test_paired_t_matches_scipy_and_the_closed_form(diffs):
    r = iso.paired_t_df2(diffs)
    ref = scipy_stats.ttest_1samp(diffs, 0.0)
    assert r["n"] == 3 and r["df"] == 2 and r["diffs"] == diffs
    assert abs(r["mean"] - np.mean(diffs)) < 1e-15 and abs(r["sd"] - np.std(diffs, ddof=1)) < 1e-12
    assert r["t"] == pytest.approx(ref.statistic, rel=1e-10)
    assert r["p_two_sided"] == pytest.approx(ref.pvalue, rel=1e-9, abs=1e-15)
    t = r["t"]
    assert abs(r["p_two_sided"] - (1 - abs(t) / math.sqrt(t * t + 2))) < 1e-12  # the df = 2 formula
    half = iso.T975_DF2 * r["sd"] / math.sqrt(3)
    assert r["ci95"] == pytest.approx([r["mean"] - half, r["mean"] + half], rel=1e-14, abs=1e-300)
    lo, hi = scipy_stats.t.interval(0.95, 2, loc=np.mean(diffs), scale=scipy_stats.sem(diffs))
    assert r["ci95"] == pytest.approx([lo, hi], rel=1e-9, abs=1e-15)
    # The interval excludes 0 exactly when p < 0.05.
    assert (r["ci95"][0] > 0 or r["ci95"][1] < 0) == (r["p_two_sided"] < 0.05)


def test_paired_t_edge_cases():
    zero = iso.paired_t_df2([0.0, 0.0, 0.0])
    assert zero["t"] is None and zero["p_two_sided"] == 1.0 and zero["ci95"] == [0.0, 0.0]
    constant = iso.paired_t_df2([0.5, 0.5, 0.5])
    assert constant["sd"] == 0.0 and constant["t"] is None and constant["p_two_sided"] == 0.0
    json.dumps([zero, constant], allow_nan=False)
    for bad in ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0]):
        with pytest.raises(ValueError, match="exactly three"):
            iso.paired_t_df2(bad)
    stats = iso.mean_sd([1.0, 2.0, 4.0])
    assert stats["values"] == [1.0, 2.0, 4.0] and stats["mean"] == pytest.approx(7 / 3, rel=1e-15)
    assert stats["sd"] == pytest.approx(math.sqrt(7 / 3), rel=1e-15)  # sample variance 14/3 / 2


def test_holm_bonferroni():
    assert iso.holm_bonferroni([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02], abs=1e-15)
    assert iso.holm_bonferroni([0.5, 0.9, 0.4]) == [1.0, 1.0, 1.0] and iso.holm_bonferroni([0.02]) == [0.02]
    assert iso.holm_bonferroni([]) == []
    rng = np.random.default_rng(5)
    for _ in range(50):
        p = rng.uniform(0, 0.2, size=int(rng.integers(1, 9))).tolist()
        got = iso.holm_bonferroni(p)
        order = np.argsort(p, kind="stable")
        want, running = [0.0] * len(p), 0.0
        for rank, i in enumerate(order):  # definition: max over smaller p of (m - rank) p, capped at 1
            running = max(running, min(1.0, (len(p) - rank) * p[i]))
            want[i] = running
        assert got == want and all(a >= b for a, b in zip(got, p))
        shuffled = rng.permutation(len(p))
        assert [iso.holm_bonferroni([p[i] for i in shuffled])[k] for k in np.argsort(shuffled)] == got


# ------------------------------------------------------------------ distribution diagnostics
def test_streaming_moments_match_the_reference_statistics():
    g = torch.Generator().manual_seed(0)
    gauss = torch.randn(200_000, generator=g, dtype=torch.float64)
    laplace = torch.distributions.Laplace(0.0, 1.0).sample((200_000,)).double() + 0.7  # non-zero mean
    spiky = gauss.clone()
    spiky[123] = 80.0
    for x, kurtosis in ((gauss, 0.0), (laplace, 3.0), (spiky, None)):
        moments = iso.Moments()
        for chunk in x.float().split(33_333):  # FP32 chunks, FP64 sums
            moments.update(chunk)
        got = moments.result()
        assert got["elements"] == x.numel()
        assert got["excess_kurtosis"] == pytest.approx(q.excess_kurtosis(x.float()), rel=1e-9, abs=1e-9)
        assert got["max_over_rms"] == pytest.approx(q.max_over_rms(x.float()), rel=1e-12)
        if kurtosis is not None:
            assert abs(got["excess_kurtosis"] - kurtosis) < 0.2
    assert iso.Moments().update(spiky) is None and got["max_over_rms"] > 50 and got["excess_kurtosis"] > 100
    empty = iso.Moments()
    empty.update(torch.zeros(8))
    assert empty.result() == {"excess_kurtosis": 0.0, "max_over_rms": 0.0, "elements": 8}


def test_calibration_statistics_and_diagnostics_on_a_tiny_model():
    config = h.PRESETS["tiny"]
    torch.manual_seed(3)
    model = Transformer(config).eval()
    with torch.no_grad():
        model.blocks[0].qkv.weight[5, 7] = 3.0  # one weight outlier
    windows = np.random.default_rng(0).integers(0, 1000, size=(8, config.sequence_length)).astype(np.int64)
    hessians, moments = iso.collect_calibration_statistics(model, windows, 4, torch.device("cpu"))
    linears = q.quantized_linears(model)
    assert list(hessians) == list(moments) == list(linears) and len(linears) == config.layers * 5
    reference = q.collect_hessians(model, windows, 4, torch.device("cpu"))
    for name, layer in linears.items():
        assert hessians[name].dtype == torch.float64 and torch.equal(hessians[name], reference[name])
        m = moments[name]
        assert m.count == windows.size * layer.in_features
        # mean(diag(H)) is the mean square of the same inputs that the moments saw.
        assert (m.sums[1] / m.count).item() == pytest.approx(hessians[name].diagonal().mean().item(), rel=1e-12)
    weights = {n: m.weight.detach().float() for n, m in linears.items()}
    diag = iso.distribution_diagnostics(weights, hessians, moments)
    assert [r["name"] for r in diag["per_layer"]] == list(linears)
    by_name = {r["name"]: r for r in diag["per_layer"]}
    spiked = by_name["blocks.0.qkv"]
    assert spiked["weight_max_over_rms"] == pytest.approx(q.max_over_rms(weights["blocks.0.qkv"]))
    assert spiked["weight_max_over_rms"] > 30 > by_name["blocks.1.qkv"]["weight_max_over_rms"]
    assert spiked["weight_excess_kurtosis"] > 100 > abs(by_name["blocks.1.qkv"]["weight_excess_kurtosis"])
    for r in diag["per_layer"]:
        assert r["kind"] in q.LINEAR_KINDS and r["hessian_diag_max_over_mean"] >= 1.0
        assert r["activation_max_over_rms"] >= 1.0 and math.isfinite(r["activation_excess_kurtosis"])
        d = hessians[r["name"]].diagonal()
        assert r["hessian_diag_max_over_mean"] == (d.max() / d.mean()).item()
    assert set(diag["summary"]) == set(iso.DIAGNOSTICS)
    for metric, summary in diag["summary"].items():
        values = [r[metric] for r in diag["per_layer"]]
        assert summary["all"] == {"mean": float(np.mean(values)), "median": float(np.median(values)),
                                  "min": min(values), "max": max(values)}
        assert list(summary["by_kind"]) == list(q.LINEAR_KINDS)
        assert summary["by_kind"]["qkv"]["max"] == max(by_name["blocks.0.qkv"][metric], by_name["blocks.1.qkv"][metric])
    json.dumps(diag, allow_nan=False)


# ------------------------------------------------------------------ output and device guards
def test_cli_output_guard_reserves_the_canonical_filename(tmp_path, monkeypatch):
    def fail_run(*args, **kwargs):
        raise AssertionError("run_experiment should not be called")

    monkeypatch.setattr(iso, "run_experiment", fail_run)
    assert iso.CANONICAL_OUTPUT == ROOT / "results" / "cycle4_c4_01b_isoloss_ptq.json"
    assert iso.parse_args([]).output == iso.CANONICAL_OUTPUT and iso._is_canonical_args(iso.parse_args([]))
    assert iso.parse_args(["--resume"]).output == iso.CANONICAL_OUTPUT  # resuming keeps the run canonical
    assert iso.parse_args(["--quick"]).output == iso.QUICK_OUTPUT != iso.CANONICAL_OUTPUT
    custom = tmp_path / "sub.json"
    assert iso.parse_args(["--tolerance", "0.05", "--output", str(custom)]).output == custom
    other_checkout = tmp_path / "other_checkout" / "results" / iso.CANONICAL_OUTPUT.name
    alias = tmp_path / "alias.json"
    alias.symlink_to(other_checkout)
    assert iso._targets_canonical_filename(other_checkout) and iso._targets_canonical_filename(alias)
    assert not iso._targets_canonical_filename(custom)
    for argv in (
        ["--quick", "--output", str(iso.CANONICAL_OUTPUT)],
        ["--quick", "--output", str(other_checkout)],
        ["--quick", "--output", str(alias)],
        ["--tolerance", "0.05"],
        ["--seeds", "4", "5", "6"],
        ["--seeds", "4", "5", "6", "--output", str(other_checkout)],
        ["--adamw-lrs", "0.002", "--output", str(alias)],
        ["--t-muon-candidates", "320", "--output", str(iso.CANONICAL_OUTPUT)],
        ["--adamw-max-steps", "800", "--output", str(iso.CANONICAL_OUTPUT)],
        ["--butterfly-steps", "10", "--output", str(iso.CANONICAL_OUTPUT)],
        ["--calibration-sequences", "64", "--output", str(iso.CANONICAL_OUTPUT)],
        ["--device", "cpu", "--output", str(iso.CANONICAL_OUTPUT)],
        ["--no-verify", "--output", str(iso.CANONICAL_OUTPUT)],
        ["--max-eval-windows", "8"],
        ["--preset", "tiny", "--output", str(other_checkout)],
        ["--seeds", "0", "1", "2", "--output", str(custom)],  # the pilot seed must stay disjoint
        ["--pilot-seed", "2", "--output", str(custom)],
    ):
        with pytest.raises(SystemExit) as exc:
            iso.main(argv)
        assert exc.value.code == 2, argv
    assert not other_checkout.exists() and not custom.exists()


def test_canonical_filename_requires_an_sm_120_device(tmp_path, monkeypatch):
    other_checkout = tmp_path / "elsewhere" / iso.CANONICAL_OUTPUT.name
    alias = tmp_path / "alias.json"
    alias.symlink_to(other_checkout)
    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for capability, allowed in (((12, 0), True), ((12, 1), True), ((13, 0), True), ((10, 0), False),
                                ((9, 0), False), ((8, 9), False)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, c=capability: c)
        for path in (iso.CANONICAL_OUTPUT, other_checkout, alias):
            if allowed:
                iso.require_blackwell(cuda, path)
            else:
                with pytest.raises(RuntimeError, match="sm_120"):
                    iso.require_blackwell(cuda, path)
        iso.require_blackwell(cuda, tmp_path / "custom.json")  # other names carry no device promise
    for path in (iso.CANONICAL_OUTPUT, other_checkout, alias):
        with pytest.raises(RuntimeError, match="sm_120"):
            iso.require_blackwell(cpu, path)
    iso.require_blackwell(cpu, tmp_path / "custom.json")
    iso.require_blackwell(cpu, None)
    # ``run_experiment`` applies the guard before it loads data or trains.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(iso, "load_data", lambda args: pytest.fail("the device guard must run first"))
    args = iso.parse_args(["--output", str(other_checkout)])
    with pytest.raises(RuntimeError, match="sm_120"):
        iso.run_experiment(args)
    assert not other_checkout.exists()


def test_atomic_json_write_is_lossless_and_strict(tmp_path):
    payload = {"a": [0.1, 1e-300, 5.396465571794642, -0.0], "b": {"t": (1, 2), "n": None, "s": "x", "f": True}}
    out = tmp_path / "nested" / "result.json"
    assert iso.write_json_atomic(out, payload) == json.loads(out.read_text())
    assert json.loads(out.read_text())["a"] == payload["a"] and list(out.parent.iterdir()) == [out]
    with pytest.raises(ValueError):
        iso.write_json_atomic(out, {"x": float("nan")})
    assert json.loads(out.read_text())["b"]["t"] == [1, 2] and list(out.parent.iterdir()) == [out]


def test_unit_log_resumes_by_key_and_config_hash(tmp_path):
    path = tmp_path / "out.json.units.jsonl"
    calls = []

    def compute(value):
        def run():
            calls.append(value)
            return {"value": value, "pair": (1, 2)}
        return run

    log = iso.UnitLog(path, "hash-a", resume=False)
    first = log.cached(("checkpoint", 1, iso.MUON, 320, 0.02), compute(1.5))
    assert first == {"value": 1.5, "pair": [1, 2], "resumed_from_unit_log": False} and (log.run, log.loaded) == (1, 0)
    with path.open("a") as handle:
        handle.write('{"key": ["checkpoint", 1, "adamw"')  # a line cut off by an interruption
    resumed = iso.UnitLog(path, "hash-a", resume=True)
    again = resumed.cached(("checkpoint", 1, iso.MUON, 320, 0.02), compute(9.0))
    assert again == {**first, "resumed_from_unit_log": True} and calls == [1.5] and resumed.loaded == 1
    assert resumed.cached(("checkpoint", 1, iso.ADAMW, 640, 0.001), compute(2.5))["value"] == 2.5
    # Another seed, optimizer, step count, LR or configuration hash is a different unit.
    assert resumed.cached(("checkpoint", 2, iso.MUON, 320, 0.02), compute(3.5))["resumed_from_unit_log"] is False
    other = iso.UnitLog(path, "hash-b", resume=True)
    assert other.cached(("checkpoint", 1, iso.MUON, 320, 0.02), compute(4.5))["value"] == 4.5
    assert calls == [1.5, 2.5, 3.5, 4.5]
    assert iso.UnitLog(path, "hash-a", resume=False).units == {} and path.read_text() == ""
    memory_only = iso.UnitLog(None, "hash-a", resume=True)
    assert memory_only.cached(("pilot", 0, iso.ADAMW, 8, 0.1), compute(5.5))["value"] == 5.5
    assert iso.unit_log_path(Path("results/x.json")) == Path("results/x.json.units.jsonl")


def test_code_sha256_covers_transitive_muon_and_data_modules(monkeypatch, tmp_path):
    import importlib

    baseline = iso.code_sha256()
    for i, mod_name in enumerate((
        "research.polar_opt.real_text_data",
        "research.polar_opt.tier2_candidates",
        "research.polar_opt.schatten_switch_blackwell",
        "research.polar_opt.polar_switch_blackwell",
        "research.polar_opt.kernels_blackwell",
    )):
        mod = importlib.import_module(mod_name)
        fake = tmp_path / f"mod_{i}.py"
        fake.write_bytes(Path(mod.__file__).read_bytes() + b"\n# tampered\n")
        with monkeypatch.context() as m:
            m.setattr(mod, "__file__", str(fake))
            assert iso.code_sha256() != baseline


# ------------------------------------------------------------------ quick end-to-end pipeline
@pytest.fixture(scope="module")
def quick(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("isoloss_quick")
    args = _quick_args(tmp_path)
    canonical_before = RESULT.read_bytes() if RESULT.exists() else None
    payload = iso.main(["--quick", "--no-verify", "--silent", "--data-dir", str(args.data_dir), "--device", DEVICE,
                        "--output", str(args.output)])
    assert json.loads(args.output.read_text()) == payload
    assert not iso.unit_log_path(args.output).exists()  # removed after the final write
    assert (RESULT.read_bytes() if RESULT.exists() else None) == canonical_before
    return payload


def test_quick_pipeline_end_to_end(quick):
    iso.validate_results(quick)
    assert quick["quick"] is True and quick["schema_version"] == iso.SCHEMA_VERSION
    pilot, stage, analysis = quick["stage_a_pilot"], quick["stage_b"], quick["analysis"]
    assert pilot["seed"] == 0 and pilot["split_reads"]["test"] == 0 and pilot["split_reads"]["val"] >= 1
    assert stage["seeds"] == [1, 2, 3] and stage["calibration"]["split"] == "train"
    assert [(c["seed"], c["optimizer"]) for c in stage["checkpoints"]] == [
        (s, o) for s in (1, 2, 3) for o in (iso.MUON, iso.ADAMW)]
    layers = quick["config"]["layers"]
    weight_only = len(q.FORMATS) * len(iso.METHODS)
    for c in stage["checkpoints"]:
        assert len(c["checkpoint_sha256"]) == 64 and c["checkpoint_sha256"] == c["checkpoint_sha256_after_ptq"]
        assert c["training"]["steps"] == (stage["t_muon"] if c["optimizer"] == iso.MUON else stage["t_adamw"])
        assert c["baseline_val"]["tokens"] == c["baseline_test"]["tokens"] == 64 * 64
        assert len(c["diagnostics"]["per_layer"]) == layers * 5
        assert c["hardware_w4a4"]["available"] is FP4_GEMM
        assert len(c["grid"]) == weight_only + (9 if FP4_GEMM else 0)
        nvfp4 = [r for r in c["grid"] if r["format"] == "nvfp4_g16" and r["mode"] == "w4a16_fakequant"]
        assert len(nvfp4) == len(iso.METHODS)
        for r in nvfp4:  # int32 decode equality of every hardware weight matrix
            assert r["hw_weights_checked_matrices"] == layers * 5
            assert r["hw_weights_decode_to_fakequant_weights"] is True
        assert {r["mode"] for r in c["grid"] if r["format"].startswith("int2")} == {"w2a16_fakequant"}
        for cell in c["best"].values():
            assert cell["best_by_val"] in cell["methods"] and cell["best_by_test"] in cell["methods"]
    # Paired seeds share the initial state and the batch stream; checkpoints differ.
    pairs = analysis["baseline"]["pairs"]
    assert [p["seed"] for p in pairs] == [1, 2, 3] and all(p["same_init_and_stream"] for p in pairs)
    assert len({p["init_fingerprint"] for p in pairs}) == 3
    assert len({c["checkpoint_sha256"] for c in stage["checkpoints"]}) == 6
    for p in pairs:
        assert p["abs_val_gap"] == abs(p["val_nll_muon"] - p["val_nll_adamw"])
        assert p["abs_test_gap"] == abs(p["test_nll_muon"] - p["test_nll_adamw"])
    assert len(analysis["comparisons"]) == weight_only + (9 if FP4_GEMM else 0)
    assert len(analysis["primary"]) == (6 if FP4_GEMM else 4)
    for entry in analysis["primary"]:
        cell = analysis["method_selection"]["pooled"][f"{entry['format']}/{entry['mode']}"]
        assert entry["method"] == cell["best_by_val"] and entry["best_by_test"] == cell["best_by_test"]
        assert 0.0 <= entry["delta_nll_diff"]["p_two_sided"] <= entry["delta_nll_diff"]["p_holm"] <= 1.0
    if CUDA:
        assert stage["checkpoints"][0]["training"]["orthogonalizer"].startswith("schatten_switch_blackwell")


@pytest.mark.skipif(not FP4_GEMM, reason="needs native NVFP4 scaled_mm (sm_100+)")
def test_quick_pipeline_runs_real_hardware_w4a4(quick):
    for c in quick["stage_b"]["checkpoints"]:
        modes = [(r["method"], r["mode"]) for r in c["grid"] if r["mode"].startswith("w4a4")]
        assert modes == [(m, mode) for m in iso.METHODS for mode in
                         (*iso.HW_MODES, *([iso.EMULATED_MODE] if m == iso.EMULATED_METHOD else []))]
        static = [r for r in c["grid"] if r["mode"] == "w4a4_hw_static"]
        assert all(0 < r["static_global_scale_range"][0] <= r["static_global_scale_range"][1] for r in static)
    parity = quick["analysis"]["hardware_vs_emulation"]
    assert len(parity["rows"]) == 6 and parity["max_abs_nll_difference"] < 1e-2


def _tampered(payload, edit):
    bad = copy.deepcopy(payload)
    edit(bad)
    return bad


def _set(path, value=None, delta=None):
    """An edit that sets (or shifts by ``delta``) the value at ``path`` (keys and list indices)."""
    def edit(payload):
        node = payload
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = node[path[-1]] + delta if delta is not None else value
    return edit


def test_validate_results_rejects_tampering(quick):
    iso.validate_results(quick)
    row, ckpt = ["stage_b", "checkpoints", 0, "grid", 0], ["stage_b", "checkpoints", 0]
    comparison, primary = ["analysis", "comparisons", 0, "test"], ["analysis", "primary", 0]
    numeric = [
        row + ["test_nll"], row + ["val_nll"], row + ["test_ppl"], row + ["test_delta_nll"], row + ["val_delta_nll"],
        row + ["test_relative_delta_ppl_pct"], ckpt + ["baseline_test", "nll"], ckpt + ["baseline_val", "ppl"],
        ckpt + ["diagnostics", "per_layer", 0, "weight_excess_kurtosis"],
        ckpt + ["diagnostics", "summary", "activation_max_over_rms", "all", "median"],
        ckpt + ["diagnostics", "summary", "hessian_diag_max_over_mean", "by_kind", "down", "max"],
        ["stage_a_pilot", "rows", 0, "val_nll"], ["stage_a_pilot", "selection", "abs_val_gap"],
        ["analysis", "baseline", "pairs", 0, "abs_val_gap"], ["analysis", "baseline", "pairs", 1, "test_nll_adamw"],
        ["analysis", "baseline", "mean_abs_test_gap"], ["analysis", "baseline", "val_gap_muon_minus_adamw", "ci95", 0],
        comparison + ["muon_delta_nll", "mean"], comparison + ["adamw_delta_nll", "sd"],
        comparison + ["muon_relative_delta_ppl_pct", "mean"], comparison + ["adamw_nll", "values", 2],
        comparison + ["delta_nll_diff", "diffs", 1], comparison + ["delta_nll_diff", "mean"],
        comparison + ["delta_nll_diff", "sd"], comparison + ["delta_nll_diff", "ci95", 1],
        comparison + ["delta_nll_diff", "p_two_sided"], comparison + ["delta_nll_diff", "p_holm_all_combinations"],
        comparison + ["abs_nll_diff", "ci95", 0], ["analysis", "comparisons", 3, "val", "abs_nll_diff", "mean"],
        primary + ["delta_nll_diff", "p_holm"], primary + ["abs_nll_diff", "p_holm"],
        primary + ["delta_nll_diff", "ci95", 0],
        ["analysis", "diagnostics_paired", "weight_excess_kurtosis", "all", "median", "mean"],
        ["analysis", "diagnostics_paired", "activation_excess_kurtosis", "qkv", "max", "p_two_sided"],
        ["analysis", "t_quantile_0p975_df2"], ["summary", "primary_test", 0, "delta_nll_diff_mean"],
        ["summary", "stage_b_baseline", "abs_test_gap_per_seed", 2],
    ]
    for path in numeric:
        with pytest.raises(ValueError):  # 1e-9 is far above the 1e-12 recomputation tolerance
            iso.validate_results(_tampered(quick, _set(path, delta=1e-9)))
    # Changes below the tolerance are rounding noise, not tampering.
    iso.validate_results(_tampered(quick, _set(comparison + ["delta_nll_diff", "mean"], delta=1e-14)))
    structural = [
        (_set(["schema_version"], "cycle4_c4_01b.v0"), "schema"),
        (_set(row + ["test_nll"], float("nan")), None),
        (_set(["stage_a_pilot", "split_reads", "test"], 1), "must not read test.bin"),
        (_set(["stage_a_pilot", "splits_used"], ["train", "val", "test"]), "must not read test.bin"),
        (_set(["stage_a_pilot", "rows", 0, "test_nll"], 5.0), "test.bin field"),
        (_set(["stage_a_pilot", "rows", 0, "evaluated_splits"], ["val", "test"]), "val.bin only"),
        (_set(["stage_a_pilot", "rows", 0, "seed"], 1), "val.bin only"),
        (_set(["stage_a_pilot", "seed"], 1), "disjoint"),
        (_set(["stage_a_pilot", "selection", "t_adamw"], 8), None),
        (_set(["stage_a_pilot", "selection", "matched"], not quick["stage_a_pilot"]["selection"]["matched"]), None),
        (_set(["data", "calibration_split"], "test"), "train.bin"),
        (_set(["stage_b", "calibration", "split"], "val"), "train.bin"),
        (_set(["stage_b", "calibration", "seed"], 1), "Calibration windows"),
        (_set(["stage_b", "seeds"], [1, 2]), None),
        (_set(["stage_b", "t_adamw"], 8), "locked Stage A selection"),
        (_set(ckpt + ["training", "steps"], 8), None),
        (_set(ckpt + ["training", "warmup_steps"], 2), "schedule"),
        (_set(ckpt + ["checkpoint_sha256"], "0" * 64), "SHA-256"),
        (_set(ckpt + ["checkpoint_sha256_after_ptq"], "f" * 64), "SHA-256"),
        (_set(ckpt + ["training", "checkpoint_sha256"], ""), "SHA-256"),
        (_set(ckpt + ["training", "init_fingerprint"], "deadbeefdeadbeef"), None),
        (_set(row + ["val_tokens"], 64), "token count"),
        (_set(row + ["test_tokens"], 64), "token count"),
        (_set(row + ["hw_weights_checked_matrices"], 9), "hardware weights must decode"),
        (_set(row + ["hw_weights_decode_to_fakequant_weights"], False), "hardware weights must decode"),
        (_set(row + ["method"], "hadamard_b64_gptq"), "incomplete PTQ grid"),
        (lambda p: p["stage_b"]["checkpoints"][3]["grid"].pop(), "incomplete PTQ grid"),
        (lambda p: p["stage_b"]["checkpoints"][0]["diagnostics"]["per_layer"].pop(), "all 10 linear layers"),
        (lambda p: p["stage_b"]["checkpoints"].reverse(), "per seed"),
        (_set(ckpt + ["best", "nvfp4_g16/w4a16_fakequant", "best_by_val"], "nope"), "best"),
        (_set(primary + ["method"], "rtn" if quick["analysis"]["primary"][0]["method"] != "rtn" else "gptq"), None),
        (_set(["analysis", "baseline", "pairs", 0, "same_init_and_stream"], False), None),
        (_set(["caveats", 0], "edited"), "caveats"),
        (lambda p: p["meta"].pop("config_hash"), "config_hash"),
    ]
    for edit, match in structural:
        with pytest.raises(ValueError, match=match):
            iso.validate_results(_tampered(quick, edit))
    # A quick result can never pass as the canonical one.
    with pytest.raises(ValueError, match="pinned train/val/test hashes"):
        iso.validate_results(quick, canonical=True)
    def pin(p):
        p["data"].update(sha256_verified=True, expected_sha256=q.EXPECTED_SHA256)
        for name, sha in q.EXPECTED_SHA256.items():
            p["data"]["splits"][name]["sha256"] = sha

    pinned = _tampered(quick, pin)
    with pytest.raises(ValueError, match="sm_120"):
        iso.validate_results(pinned, canonical=True)


def test_validate_results_hardened_provenance_and_bounds(quick):
    ckpt = ["stage_b", "checkpoints", 0]
    starts = list(quick["stage_b"]["calibration"]["starts"])
    hardened = [
        (_set(["meta", "git_sha"], "not-a-sha"), "git_sha"),
        (_set(["meta", "code_sha256"], "xyz"), "code_sha256"),
        (_set(["meta", "config_hash"], "1234"), "config_hash"),
        (_set(["meta", "git_dirty"], "false"), "git_dirty"),
        (_set(["data", "training_split"], "test"), "train.bin"),
        (_set(["stage_b", "calibration", "starts"], [-1, *starts[1:]]), "Calibration windows"),
        (_set(["stage_b", "calibration", "starts"], [quick["data"]["splits"]["train"]["tokens"], *starts[1:]]),
         "Calibration windows"),
        (_set(["stage_b", "calibration", "tokens"], 7), "Calibration windows"),
        (_set(["stage_a_pilot", "search"], None), "search trace"),
        (lambda p: p["stage_a_pilot"]["rows"].__setitem__(
            0, {**p["stage_a_pilot"]["rows"][0], "steps": p["protocol"]["adamw_max_steps"] - 8,
                "warmup_steps": iso.warmup_for(p["protocol"]["adamw_max_steps"] - 8),
                "tokens": (p["protocol"]["adamw_max_steps"] - 8) * 64 * 8}),
         "adamw_max_steps"),
        (_set(["stage_a_pilot", "rows", 0, "tokens"], 7), "token count"),
        (_set(ckpt + ["training", "auxiliary_lr"], 99.0), "auxiliary_lr"),
        (_set(ckpt + ["diagnostics", "per_layer", 0, "name"], "blocks.0.bogus"), "layer names"),
        (_set(ckpt + ["diagnostics", "per_layer", 0, "kind"], "out"), "layer names"),
    ]
    for edit, match in hardened:
        with pytest.raises(ValueError, match=match):
            iso.validate_results(_tampered(quick, edit))
    if FP4_GEMM:
        idx = next(i for i, r in enumerate(quick["stage_b"]["checkpoints"][0]["grid"])
                   if r["mode"] == "w4a4_hw_static")
        with pytest.raises(ValueError, match="static activation scale"):
            iso.validate_results(_tampered(
                quick, _set(ckpt + ["grid", idx, "static_scale_calibration"], "test.bin static scales")))


# ------------------------------------------------------------------ committed canonical artifact
def _canonical():
    if not RESULT.exists():
        pytest.skip("canonical result not generated yet")
    return json.loads(RESULT.read_text())


def test_committed_canonical_artifact():
    payload = _canonical()
    iso.validate_results(payload, canonical=True)
    assert payload["quick"] is False and payload["meta"]["git_dirty"] is False
    assert payload["meta"]["environment"]["capability"] >= [12, 0]
    assert {k: v["sha256"] for k, v in payload["data"]["splits"].items()} == q.EXPECTED_SHA256
    pilot, stage = payload["stage_a_pilot"], payload["stage_b"]
    assert pilot["seed"] == 0 and pilot["split_reads"]["test"] == 0 and pilot["selection"]["matched"]
    assert pilot["selection"]["abs_val_gap"] <= 0.02 and pilot["selection"]["t_adamw"] <= 768
    assert all(r["val_tokens"] == 2_001_920 and r["tokens"] < 26_004_518 for r in pilot["rows"])
    assert stage["seeds"] == [1, 2, 3] and stage["calibration"]["tokens"] == 128 * 512
    assert stage["calibration"]["seed"] == 20261126 and len(stage["checkpoints"]) == 6
    for c in stage["checkpoints"]:
        assert len(c["grid"]) == 25 and len(c["diagnostics"]["per_layer"]) == 60
        for r in c["grid"]:
            assert r["val_tokens"] == 2_001_920 and r["test_tokens"] == 1_999_872
            if r["format"] == "nvfp4_g16" and r["mode"] == "w4a16_fakequant":
                assert r["hw_weights_checked_matrices"] == 60 and r["hw_weights_decode_to_fakequant_weights"]
    assert len(payload["analysis"]["comparisons"]) == 25 and len(payload["analysis"]["primary"]) == 6
    assert payload["analysis"]["hardware_vs_emulation"]["passed"]


def test_committed_canonical_artifact_rejects_tampering():
    payload = _canonical()
    row = ["stage_b", "checkpoints", 4, "grid", 7]
    edits = [
        _set(row + ["test_nll"], delta=1e-9),
        _set(["analysis", "primary", 2, "delta_nll_diff", "p_holm"], delta=1e-9),
        _set(["analysis", "comparisons", 11, "test", "abs_nll_diff", "ci95", 1], delta=-1e-9),
        _set(row + ["val_tokens"], 2_001_408),
        _set(["stage_b", "checkpoints", 2, "baseline_test", "tokens"], 1_999_360),
        _set(["stage_a_pilot", "rows", 0, "val_tokens"], 1_048_576),
        _set(["data", "splits", "test", "sha256"], "0" * 64),
        _set(["data", "sha256_verified"], False),
        _set(["meta", "environment", "capability"], [9, 0]),
        _set(["meta", "git_dirty"], True),
        _set(["quick"], True),
        _set(["protocol", "tolerance_nats"], 0.05),
        _set(["stage_b", "checkpoints", 1, "grid", 0, "hw_weights_checked_matrices"], 5),
        lambda p: p["stage_b"]["checkpoints"][5]["grid"].pop(),
    ]
    for edit in edits:
        with pytest.raises(ValueError):
            iso.validate_results(_tampered(payload, edit), canonical=True)
