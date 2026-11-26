"""Fast checks for the Cycle 2 real-text Tier-2 harness, data shards and CUDA-graph benchmark."""

import json
import math

import numpy as np
import pytest
import torch

from research.polar_opt import bench_cuda_graph as bench
from research.polar_opt import harness_tier2_real as h
from research.polar_opt import real_text_data as rtd

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")
TINY = h.PRESETS["tiny"]


class ByteEncoder:
    """Offline stand-in for tiktoken: one token per UTF-8 byte (ids < 256)."""

    @staticmethod
    def encode_ordinary_batch(texts, num_threads=1):
        return [list(text.encode()) for text in texts]


def _documents(count=900):
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "kappa"]
    docs = [" ".join(rng.choice(words, size=int(rng.integers(20, 60)))) + f" #{i}" for i in range(count)]
    return docs[:300] + docs[:20] + ["", "  "] + docs[300:]  # 20 exact duplicates and 2 empty documents


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("shards")
    rtd.build_shards(_documents(), out, targets={"train": 60_000, "val": 6_000, "test": 6_000},
                     source={"name": "synthetic"}, encoder=ByteEncoder())
    return out


@pytest.fixture(scope="module")
def data(shard_dir):
    return h.ShardData(shard_dir)


# ---------------------------------------------------------------- data shards
def test_shards_are_disjoint_and_match_manifest(shard_dir):
    manifest = rtd.verify_manifest(shard_dir)
    hashes = json.loads((shard_dir / "doc_hashes.json").read_text())
    sets = {name: set(hashes[name]) for name in rtd.SPLITS}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    assert manifest["duplicates_skipped"] == 20 and manifest["empty_skipped"] == 2
    for name in rtd.SPLITS:
        entry = manifest["splits"][name]
        tokens = np.memmap(shard_dir / entry["file"], dtype=np.uint16, mode="r")
        assert tokens.size == entry["tokens"] >= entry["target_tokens"]
        assert all(rtd.route(d) == name for d in hashes[name])


def test_verify_manifest_detects_corruption_and_leakage(tmp_path):
    rtd.build_shards(_documents(300), tmp_path, targets={"train": 9_000, "val": 600, "test": 600},
                     source={"name": "synthetic"}, encoder=ByteEncoder())
    rtd.verify_manifest(tmp_path)
    hashes_path = tmp_path / "doc_hashes.json"
    original = hashes_path.read_text()
    hashes = json.loads(original)
    hashes["val"].append(hashes["train"][0])
    hashes_path.write_text(json.dumps(hashes))
    with pytest.raises(ValueError):
        rtd.verify_manifest(tmp_path)
    hashes_path.write_text(original)
    with (tmp_path / "val.bin").open("r+b") as handle:
        handle.seek(10)
        byte = handle.read(1)
        handle.seek(10)
        handle.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256"):
        rtd.verify_manifest(tmp_path)


@pytest.mark.skipif(not (rtd.DEFAULT_DIR / "manifest.json").exists(), reason="real shards not built")
def test_real_text_shards_on_disk():
    manifest = rtd.verify_manifest(rtd.DEFAULT_DIR)
    splits = manifest["splits"]
    assert manifest["encoding"] == "gpt2" and manifest["padded_vocab_size"] == 50304
    assert manifest["source"]["dataset"] in ("HuggingFaceFW/fineweb-edu", "Salesforce/wikitext")
    assert splits["train"]["tokens"] >= 25_000_000
    assert splits["val"]["tokens"] >= 2_000_000 and splits["test"]["tokens"] >= 2_000_000


# ---------------------------------------------------------------- pairing
def test_initial_weights_and_batch_streams_are_paired(data):
    a, b, other = h.initial_state(TINY, 0), h.initial_state(TINY, 0), h.initial_state(TINY, 1)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.equal(a["embedding.weight"], other["embedding.weight"])
    assert h.state_fingerprint(a) == h.state_fingerprint(b) != h.state_fingerprint(other)
    cpu = torch.device("cpu")
    s1, s2, s3 = (h.BatchStream(data.train, TINY, seed, cpu) for seed in (0, 0, 1))
    for _ in range(3):
        x1, y1 = s1.next_batch()
        x2, y2 = s2.next_batch()
        assert torch.equal(x1, x2) and torch.equal(y1, y2)
        assert torch.equal(x1[..., 1:], y1[..., :-1])  # targets are inputs shifted by one
    assert not torch.equal(s3.next_batch()[0], x1)
    assert x1.shape == (TINY.grad_accum, TINY.micro_batch, TINY.sequence_length)


def test_trials_share_initial_weights_and_stream_across_optimizers(data):
    factories = h.builtin_factories()
    rows = [h.run_trial(TINY, data, device=torch.device("cpu"), seed=3, optimizer_name=name,
                        factory=factories[name], lr=lr, tokens=TINY.step_tokens,
                        compile_warmup_steps=0, eval_tokens=1024, telemetry_every=0)
            for name, lr in (("adamw", 1e-3), ("muon_ps2_eager", 0.02))]
    assert rows[0]["init_fingerprint"] == rows[1]["init_fingerprint"]
    assert rows[0]["stream_fingerprint"] == rows[1]["stream_fingerprint"]


# ---------------------------------------------------------------- update layout
def _tiny_cuda_muon(update_dtype):
    torch.manual_seed(0)
    model = h.Transformer(TINY).cuda()
    x = torch.randint(0, 50257, (2, TINY.sequence_length), device="cuda")
    h._loss(model, x, x).backward()
    from research.polar_opt.baselines import polar_express

    return model, h.Muon(model, polar_express, lr=0.02, auxiliary_lr=1e-3, weight_decay=0.01,
                         update_dtype=update_dtype)


def _count_ops(optimizer, name):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        optimizer.step()
        torch.cuda.synchronize()
    return sum(event.name == name for event in prof.events())


@needs_cuda
def test_fp32_updates_are_contiguous_and_stay_on_the_foreach_fast_path():
    model, optimizer = _tiny_cuda_muon("fp32")
    tall = [names for names in optimizer.groups if optimizer.matrices[names[0]].shape[0] > optimizer.matrices[names[0]].shape[1]]
    assert tall, "the tiny preset must contain tall matrices"
    params = [optimizer.matrices[n] for n in tall[0]]
    native = optimizer.orthogonalize(torch.stack([p.grad for p in params]))
    assert native.dtype == torch.bfloat16  # the Cycle 1 confounder being removed
    updates = optimizer.prepare(native).unbind()
    for p, u in zip(params, updates):
        assert u.dtype == torch.float32 == p.dtype and u.stride() == p.stride() and u.is_contiguous()
    assert _count_ops(optimizer, "aten::_foreach_add_") >= len(optimizer.groups)
    assert _count_ops(optimizer, "aten::add_") == 0
    # The native layout falls back to one aten::add_ per matrix, which this check must detect.
    _, native_optimizer = _tiny_cuda_muon("native")
    assert _count_ops(native_optimizer, "aten::add_") >= len(native_optimizer.matrices)


# ---------------------------------------------------------------- schedule and budgets
def test_lr_multiplier_warmup_and_cosine():
    assert h.lr_multiplier(0, 0.0, 10, 0.1) == pytest.approx(0.1)
    assert h.lr_multiplier(9, 0.0, 10, 0.1) == pytest.approx(1.0)
    assert h.lr_multiplier(500, 0.5, 10, 0.1) == pytest.approx(0.55)
    assert h.lr_multiplier(999, 1.0, 10, 0.1) == pytest.approx(0.1)
    assert h.lr_multiplier(0, 0.3, 0, 1.0) == 1.0
    # Decoupled token warmup + cosine decay
    assert h.lr_multiplier(9, 0.1, 10, 0.1, total_steps=110) == pytest.approx(1.0)
    assert h.lr_multiplier(60, 0.6, 10, 0.1, total_steps=110) == pytest.approx(0.55)
    # Wall-mode warmup fraction reaches peak 1.0 regardless of step count
    assert h.lr_multiplier(1, 0.0625, 50, 0.1, warmup_fraction=0.0625) == pytest.approx(1.0)
    assert h.lr_multiplier(1, 0.53125, 50, 0.1, warmup_fraction=0.0625) == pytest.approx(0.55)


def _cpu_trial(data, **kwargs):
    options = dict(device=torch.device("cpu"), seed=0, optimizer_name="adamw", factory=None, lr=1e-3,
                   lr_warmup_steps=0, min_lr_ratio=1.0, compile_warmup_steps=0, eval_tokens=1024,
                   telemetry_every=0)
    return h.run_trial(TINY, data, **{**options, **kwargs})


def test_token_budget_runs_exact_steps(data):
    row = _cpu_trial(data, tokens=3 * TINY.step_tokens)
    assert row["steps"] == 3 and row["tokens"] == row["target_tokens"] == 3 * TINY.step_tokens
    assert row["wall_budget_overshoot_seconds"] is None and not row["diverged"]
    with pytest.raises(ValueError):
        _cpu_trial(data, tokens=TINY.step_tokens + 1)
    with pytest.raises(ValueError):
        _cpu_trial(data, tokens=TINY.step_tokens, seconds=1.0)


def test_wall_budget_keeps_last_step_before_deadline(data):
    ticks = iter(np.arange(0, 100, 0.4))  # every step takes exactly 0.4 "seconds"
    row = _cpu_trial(data, seconds=1.0, clock=lambda: float(next(ticks)))
    assert row["steps"] == 2 and row["train_seconds"] == pytest.approx(0.8)
    assert row["discarded_step_seconds"] == pytest.approx(0.4)
    assert row["wall_budget_overshoot_seconds"] == 0.0
    two = _cpu_trial(data, tokens=2 * TINY.step_tokens)
    three = _cpu_trial(data, tokens=3 * TINY.step_tokens)
    assert row["val_loss"] == two["val_loss"] and row["test_loss"] == two["test_loss"]
    assert three["val_loss"] != two["val_loss"]  # the discarded third step really was undone


@needs_cuda
def test_cuda_trial_reports_segment_timing(data):
    factory = h.builtin_factories()["muon_ps3_blackwell"]
    row = h.run_trial(TINY, data, device=torch.device("cuda"), seed=0, optimizer_name="muon_ps3_blackwell",
                      factory=factory, lr=0.02, tokens=2 * TINY.step_tokens, eval_tokens=1024,
                      compile_warmup_steps=1, telemetry_every=1)
    t = row["timing"]
    for key in ("fwd_bwd_ms", "polar_ortho_ms", "layout_cast_ms", "foreach_update_ms", "optimizer_step_ms"):
        assert t[key] is not None and t[key] >= 0
    assert t["optimizer_step_ms"] >= t["polar_ortho_ms"] + t["foreach_update_ms"]
    assert row["gpu_start"] and "error" not in row["gpu_start"]
    assert len(row["gpu_samples"]) == 2 and len(row["loadavg_start"]) == 3
    assert math.isclose(row["val_ppl"], math.exp(row["val_loss"]))


# ---------------------------------------------------------------- registry and statistics
CALLS = []


def make_recording_polar():
    """Zero-argument factory: a fresh orthogonalizer per run."""
    from research.polar_opt.baselines import polar_switch

    CALLS.append("factory")
    return lambda G: polar_switch(G, "PS2_55")


def test_extra_optimizer_registration(data, shard_dir, tmp_path):
    factories = {"adamw": None}
    names = h.parse_extra(["cand=tests.test_tier2_real:make_recording_polar",
                           "direct=research.polar_opt.baselines:polar_express"], factories)
    assert names == ["cand", "direct"]
    from research.polar_opt.baselines import polar_express

    assert factories["direct"]() is polar_express
    with pytest.raises(ValueError):
        h.parse_extra(["cand=tests.test_tier2_real:make_recording_polar"], factories)
    with pytest.raises(ValueError):
        h.parse_extra(["bad=research.polar_opt.tier2_candidates:ShapeKeyedSpectralPolar"], {})
    CALLS.clear()
    out = tmp_path / "run.json"
    h.main(["--preset", "tiny", "--device", "cpu", "--data-dir", str(shard_dir), "--optimizers", "adamw",
            "--extra-optimizer", "cand=tests.test_tier2_real:make_recording_polar", "--seeds", "0", "1",
            "--adamw-lrs", "1e-3", "--muon-lrs", "0.02", "--budget", "tokens", "--tokens",
            str(TINY.step_tokens), "--eval-tokens", "1024", "--compile-warmup-steps", "1",
            "--reference", "adamw", "--output", str(out)])
    result = json.loads(out.read_text())
    assert len(result["runs"]) == 4 and CALLS == ["factory"] * 4  # warmup + measured run, two seeds
    assert result["meta"]["git_sha"] and result["meta"]["data"]["splits"]["train"]["sha256"]
    paired = result["summary"]["paired_vs_reference"]
    assert [p["optimizer"] for p in paired] == ["cand"] and paired[0]["test_loss_diff"]["n"] == 2


def test_summary_selects_lr_on_validation_and_pairs_seeds():
    rows = []
    for seed, (ref, cand) in enumerate([(3.0, 2.9), (3.1, 2.95), (3.05, 3.0)]):
        for name, lr, val, test in (("ref", 0.01, ref + 0.1, ref + 0.1), ("ref", 0.02, ref, ref),
                                    ("cand", 0.02, cand, cand), ("cand", 0.04, cand + 0.2, cand - 1)):
            rows.append({"budget": "tokens", "optimizer": name, "update_dtype": "fp32", "lr": lr,
                         "seed": seed, "val_loss": val, "test_loss": test, "diverged": False,
                         "tokens_per_second": 100.0 if name == "ref" else 110.0})
    summary = h.summarize(rows, "ref")
    best = {b["optimizer"]: b for b in summary["best_lr"]}
    assert best["cand"]["lr"] == 0.02 and best["cand"]["lr_at_grid_edge"]  # chosen on val, not test
    (paired,) = summary["paired_vs_reference"]
    diffs = [-0.1, -0.15, -0.05]
    assert paired["test_loss_diff"]["mean"] == pytest.approx(np.mean(diffs))
    half = 4.302652729911275 * np.std(diffs, ddof=1) / math.sqrt(3)  # t(0.975, df=2)
    assert paired["test_loss_diff"]["ci95"] == pytest.approx([np.mean(diffs) - half, np.mean(diffs) + half])
    assert paired["tokens_per_second_ratio"]["mean"] == pytest.approx(1.1)


# ---------------------------------------------------------------- Nesterov harvesting
def test_harvest_saves_post_nesterov_pairs_and_exact_spectra(data):
    seen = []

    def factory():
        from research.polar_opt.baselines import polar_switch

        def orthogonalize(G):
            seen.append(G.detach().clone())
            return polar_switch(G, "PS2_55")
        return orthogonalize

    result = h.harvest_nesterov(TINY, data, device=torch.device("cpu"), factory=factory, lr=0.02,
                                auxiliary_lr=1e-3, weight_decay=0.01, steps=4, seed=999, starts=[1, 3],
                                layers=[0])
    pairs = result["pairs"]
    assert len(pairs) == 2 * 5 and {p["step_t"] for p in pairs} == {1, 3}
    groups_per_step = len(seen) // 4
    for p in pairs:
        assert p["layer"] == 0 and p["N_t"].dtype == torch.float32 and p["N_t"].shape == p["N_t1"].shape
        assert p["sv_t"].dtype == torch.float64 and torch.all(p["sv_t"][:-1] >= p["sv_t"][1:])
        # N_t is exactly one slice of the orthogonalizer input at step t.
        inputs = seen[(p["step_t"] - 1) * groups_per_step:p["step_t"] * groups_per_step]
        assert any(g.shape == p["N_t"].shape and torch.equal(g, p["N_t"]) for G in inputs for g in G)
    assert not torch.equal(pairs[0]["N_t"], pairs[0]["N_t1"])


# ---------------------------------------------------------------- CUDA-graph benchmark
@pytest.mark.parametrize("count", [3, 4, 7])
def test_block_order_is_counterbalanced(count):
    names = [chr(97 + i) for i in range(count)]
    blocks = count if count % 2 == 0 else 2 * count
    orders = [bench.block_order(names, b) for b in range(blocks)]
    assert all(sorted(o) == names for o in orders)
    for n in names:  # every method takes every position equally often
        positions = [order.index(n) for order in orders]
        assert all(positions.count(i) == blocks // count for i in range(count))
    follows = [(o[i], o[i + 1]) for o in orders for i in range(count - 1)]
    pairs = {(a, b) for a in names for b in names if a != b}
    # Williams design: each ordered pair is adjacent equally often (first-order carryover balance).
    assert {follows.count(pair) for pair in pairs} == {blocks // count}


@needs_cuda
def test_cuda_graph_bench_smoke():
    factories = h.builtin_factories()
    h.parse_extra(["extra=tests.test_tier2_real:make_recording_polar"], factories)
    result = bench.run(["muon_ps2_eager", "muon_ps3_blackwell", "extra"], factories,
                       shapes=[(64, 256), (2, 128, 256)], warmup=5, blocks=2, iters=2)
    assert result["env"]["git_sha"] and result["settings"]["flush"].startswith("flush.zero_()")
    for shape in result["shapes"]:
        assert len(shape["loadavg_start"]) == 3 and len(shape["block_orders"]) == 2
        for m in shape["methods"].values():
            assert "capture_error" not in m and m["graph_vs_eager_max_abs"] < 1e-2
            for clock in ("cuda_graph_ms", "eager_event_ms", "wall_ms"):
                assert len(m[clock]["samples_ms"]) == 4 and m[clock]["median_ms"] > 0
    with pytest.raises(ValueError):
        bench.run(["muon_ps2_eager"], factories, shapes=[(64, 256)], warmup=4)


def test_nested_output_resume_and_config_hash_guard(data, shard_dir, tmp_path):
    out = tmp_path / "nested/dir/sweep.json"
    base_args = [
        "--preset", "tiny", "--device", "cpu", "--data-dir", str(shard_dir),
        "--optimizers", "adamw", "muon_ps2_eager", "--seeds", "0",
        "--adamw-lrs", "1e-3", "--muon-lrs", "0.02", "--budget", "tokens",
        "--tokens", str(TINY.step_tokens), "--eval-tokens", "1024",
        "--compile-warmup-steps", "0", "--reference", "adamw", "--output", str(out),
    ]
    h.main(base_args)
    assert out.is_file() and out.with_suffix(".jsonl").is_file()
    first = json.loads(out.read_text())
    assert len(first["runs"]) == 2
    # Resume with same settings succeeds cleanly without adding duplicate rows
    h.main([*base_args, "--resume"])
    resumed = json.loads(out.read_text())
    assert len(resumed["runs"]) == 2
    # Resume with changed tokens fails fast rather than silently mixing runs
    with pytest.raises(ValueError, match="Cannot --resume"):
        h.main([*base_args, "--tokens", str(2 * TINY.step_tokens), "--resume"])


def test_summarize_deterministic_reference_and_zero_steps_guard():
    rows = [
        {"budget": "wall", "optimizer": "muon_pe_eager", "update_dtype": "fp32", "lr": 0.02,
         "seed": 0, "val_loss": 3.0, "test_loss": 3.0, "diverged": False, "tokens_per_second": 100.0},
        {"budget": "wall", "optimizer": "muon_pe_eager", "update_dtype": "native", "lr": 0.02,
         "seed": 0, "val_loss": 3.05, "test_loss": 3.05, "diverged": False, "tokens_per_second": 120.0},
        {"budget": "wall", "optimizer": "cand", "update_dtype": "native", "lr": 0.02,
         "seed": 0, "val_loss": 2.9, "test_loss": 2.9, "diverged": False, "tokens_per_second": 144.0},
        {"budget": "wall", "optimizer": "zero_step", "update_dtype": "fp32", "lr": 0.02,
         "seed": 0, "val_loss": 4.0, "test_loss": 4.0, "diverged": False, "tokens_per_second": None},
    ]
    summary = h.summarize(rows, "muon_pe_eager")
    by_opt = {c["optimizer"]: c for c in summary["paired_vs_reference"]}
    assert by_opt["cand"]["reference_update_dtype"] == "native"
    assert by_opt["cand"]["test_loss_diff"]["mean"] == pytest.approx(2.9 - 3.05)
    assert by_opt["zero_step"]["tokens_per_second_ratio"]["n"] == 0


def test_requires_arguments_and_muon_parameter_partitioning():
    def star_args(*args, **kwargs):
        return args[0]

    def kw_only_factory(*, variant):
        return variant

    class KwOnlyClass:
        def __init__(self, *, variant):
            self.variant = variant

    assert h._requires_arguments(star_args)
    assert h._requires_arguments(kw_only_factory)
    assert h._requires_arguments(KwOnlyClass)
    factories = {}
    h.parse_extra(["star=tests.test_tier2_real:make_recording_polar"], factories)
    assert callable(factories["star"]())
    wrapped = h.load_factory("research.polar_opt.baselines:polar_express")
    assert wrapped() is h.builtin_factories()["muon_pe_eager"]()

    model = h.Transformer(TINY)
    opt = h.Muon(model, lambda g: g, lr=0.02, auxiliary_lr=1e-3, weight_decay=0.01)
    mat_ids = {id(p) for p in opt.matrices.values()}
    aux_ids = {id(p) for g in opt.auxiliary.param_groups for p in g["params"]}
    all_ids = {id(p) for p in model.parameters()}
    assert not (mat_ids & aux_ids) and (mat_ids | aux_ids) == all_ids
    for g in opt.auxiliary.param_groups:
        if all(p.ndim < 2 for p in g["params"]):
            assert g["weight_decay"] == 0.0


@needs_cuda
def test_cuda_graph_bench_mutation_and_nan_guards():
    def mutator():
        def fn(G):
            G.add_(1.0)
            return G
        return fn

    with pytest.raises(RuntimeError, match="mutated"):
        bench.run(["mut"], {"mut": mutator}, shapes=[(32, 64)], warmup=5, blocks=2, iters=1)

    def nan_fn():
        return lambda G: torch.full_like(G, float("nan"))

    res = bench.run(["nan_cand"], {"nan_cand": nan_fn}, shapes=[(32, 64)], warmup=5, blocks=2, iters=1)
    row = res["shapes"][0]["methods"]["nan_cand"]
    assert row["finite"] is False and row["graph_vs_eager_max_abs"] is None
    assert json.dumps(res, allow_nan=False)
