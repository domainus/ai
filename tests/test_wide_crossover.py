"""Checks for the Cycle 3 wide-layer full-model crossover benchmark."""

import itertools
import json

import pytest
import torch

from research.polar_opt import bench_wide_crossover as wide
from research.polar_opt import harness_tier2_real as h

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
SMALL = {"label": "small", "width": 64, "hidden": 128, "grid": True,
         "groups": [[4, 64, 64], [2, 128, 64], [1, 64, 128]]}


def _bench(methods, blocks=2, iters=2):
    flush = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
    return wide.bench_step(methods, SMALL, flush=flush, warmups=5, blocks=blocks, iters=iters, seed=0)


def test_default_schedule_is_williams_balanced():
    names = list(wide.method_factories())
    blocks = wide.default_blocks(len(names))
    orders = [wide.block_order(names, b) for b in range(blocks)]
    assert len(names) == 7 and blocks == 14 and wide.default_blocks(4) == 4
    assert all(sorted(order) == sorted(names) for order in orders)
    for name in names:  # every method takes every position equally often
        positions = [order.index(name) for order in orders]
        assert all(positions.count(i) == blocks // len(names) for i in range(len(names)))
    follows = [pair for order in orders for pair in itertools.pairwise(order)]
    # First-order carryover balance: each ordered pair is adjacent equally often.
    assert {follows.count(pair) for pair in itertools.permutations(names, 2)} == {blocks // len(names)}


def test_step_configs_cover_the_width_grid_and_the_120m_preset():
    configs = {c["label"]: c for c in wide.step_configs()}
    assert [c["width"] for c in configs.values() if c["grid"]] == [512, 768, 1024, 1536, 2048]
    assert [configs[f"w{w}"]["hidden"] for w in wide.WIDTHS] == [1408, 2176, 2816, 4224, 5632]
    assert configs["w768"]["groups"] == [[48, 768, 768], [24, 2176, 768], [12, 768, 2176]]
    assert configs["w768_120m"]["groups"] == [[48, 768, 768], [24, 2048, 768], [12, 768, 2048]]
    # The harness row is the exact set of matrices that the 120m Muon optimizer steps.
    assert configs["w768_120m_harness"]["groups"] == [[12, 2304, 768], [12, 768, 768],
                                                      [24, 2048, 768], [12, 768, 2048]]
    stacked = sum(g[0] * g[1] * g[2] for g in configs["w768_120m_harness"]["groups"])
    assert stacked == h.parameter_counts(h.PRESETS["120m"])["muon_matrices"]
    assert not configs["w768_120m"]["grid"] and not configs["w768_120m_harness"]["grid"]
    assert [c["label"] for c in wide.step_configs([64, 128], layers=2)] == ["w64", "w128"]


def test_crossover_reports_where_the_speedup_crosses_one():
    def steps(speedups, grid=True):
        return [{"width": w, "grid": grid,
                 "methods": {"ss3_54": {"speedup_vs_ps3_blackwell": {"cuda_graph": s}}}}
                for w, s in speedups]

    rows = steps([(2048, 1.04), (512, 0.90), (1024, 0.96)]) + steps([(768, 2.0)], grid=False)
    found = wide.crossover(rows, "ss3_54", "cuda_graph")
    assert list(found["speedup_by_width"]) == ["512", "1024", "2048"]  # preset rows are excluded
    assert found["first_width_at_or_above_1x"] == 2048
    assert [c["between_widths"] for c in found["crossings"]] == [[1024, 2048]]
    assert 1024 < found["crossings"][0]["width_estimate"] < 2048
    never = wide.crossover(steps([(512, 0.90), (1024, 0.95)]), "ss3_54", "cuda_graph")
    assert never["first_width_at_or_above_1x"] is None and never["crossings"] == []


@needs_cuda
def test_graph_replay_is_bitwise_equal_to_eager_in_every_group():
    factories = wide.method_factories()
    names = ["ps3_eager", "ps3_blackwell", "ss3_54"]
    row = _bench({name: factories[name]() for name in names})
    assert [len(order) for order in row["block_orders"]] == [3, 3]
    for name in names:
        m = row["methods"][name]
        assert m["finite"] is True and m["input_unchanged"] is True and m["graph_vs_eager_max_abs"] == 0.0
        assert m["verified_graph_captures"] == 3  # the pre-timing capture and one per block
        assert [g["shape"] for g in m["groups"]] == SMALL["groups"]
        assert all(g["graph_vs_eager_max_abs"] == 0.0 for g in m["groups"])
        for clock in wide.CLOCKS:
            assert len(m[clock]["samples_ms"]) == 4
            assert 0 < m[clock]["min_ms"] <= m[clock]["median_ms"] and m[clock]["iqr_ms"] >= 0
        assert m["speedup_vs_ps3_blackwell"]["cuda_graph"] > 0 and m["speedup_vs_pe_compiled"] is None
    assert row["methods"]["ps3_blackwell"]["speedup_vs_ps3_blackwell"] == {"eager_event": 1.0, "cuda_graph": 1.0}


@needs_cuda
def test_bench_stops_on_mutation_non_finite_output_and_graph_mismatch():
    def mutate(G):
        G.add_(1.0)
        return G

    with pytest.raises(RuntimeError, match="mutated"):
        _bench({"mutator": mutate})
    with pytest.raises(RuntimeError, match="non-finite"):
        _bench({"nan": lambda G: torch.full_like(G, float("nan"))})
    scale = itertools.count(1)  # a host value that the capture freezes
    with pytest.raises(RuntimeError, match="not bitwise equal"):
        _bench({"stateful": lambda G: G * next(scale)})


@needs_cuda
def test_quick_cli_writes_strict_json(tmp_path):
    output = tmp_path / "nested" / "quick.json"
    wide.main(["--quick", "--output", str(output)])

    def reject(constant):
        raise AssertionError(f"non-strict JSON constant {constant}")

    result = json.loads(output.read_text(), parse_constant=reject)  # NaN and Infinity are rejected
    assert json.dumps(result, allow_nan=False)
    env, settings = result["env"], result["settings"]
    assert len(env["git_sha"]) == 40 and env["git_dirty"] in (True, False)
    assert set(env["loadavg"]) == set(env["gpu_telemetry"]) == {"start", "end"}
    assert settings["quick"] is True and settings["warmups"] == 5
    assert (settings["widths"], settings["layers"], settings["blocks"], settings["iters"]) == ([64, 128], 2, 2, 2)
    assert settings["allow_fp16_reduced_precision_reduction"] is False
    assert settings["allow_bf16_reduced_precision_reduction"] is False
    assert not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    assert [s["label"] for s in result["steps"]] == ["w64", "w128"]
    for step in result["steps"]:
        assert list(step["methods"]) == list(wide.method_factories()) and step["gpu_telemetry"]
        for m in step["methods"].values():
            assert m["finite"] is True and m["graph_vs_eager_max_abs"] == 0.0
            assert all(m[f"speedup_vs_{reference}"][clock] > 0
                       for reference in wide.REFERENCES for clock in wide.CLOCKS)
    assert set(result["crossover_vs_ps3_blackwell"]) == {"ss3_54", "ss3_53"}
    assert set(result["crossover_vs_ps3_blackwell"]["ss3_54"]["cuda_graph"]["speedup_by_width"]) == {"64", "128"}
