# Pre-Sweep Adversarial Audit Report — Cycle 2 T3-01

- **Auditor:** Gemini 4 Neon (`thought_level: high`) · **Date:** 2026-11-25
- **Scope:** Strictly read-only, zero-GPU audit of the Cycle 2 `T3-01` real-text Tier-2 pretraining harness, data builder, CUDA-graph benchmark, unit tests, and on-disk `FineWeb-Edu` shards in `/home/ryan/Documents/projects/ai`.
- **Verification Executed (CPU-only, zero PyTorch/CUDA imports):**
  - Verified on-disk shard checksums, token ranges, and document disjointness via `real_text_data.py --verify` (pure NumPy/stdlib).
  - Audited exact EOT counts, document length distributions, and 64-token prefix overlap across `train.bin`, `val.bin`, and `test.bin`.
  - Cross-checked all files against Cycle 1 audit findings in [AUDIT_cycle1_gpt62_astra.md](/home/ryan/Documents/projects/ai/audits/AUDIT_cycle1_gpt62_astra.md), [AUDIT_cycle1_gemini_neon.md](/home/ryan/Documents/projects/ai/audits/AUDIT_cycle1_gemini_neon.md), and [AUDIT_cycle1_claude_opus.md](/home/ryan/Documents/projects/ai/audits/AUDIT_cycle1_claude_opus.md).

## Executive Verdict: **`PASS_WITH_CAVEATS`**

> **Pre-Sweep Gate:** The core `T3-01` architecture is substantially cleaner than Cycle 1—zero `CRITICAL` data-leakage, gradient-scaling, or update-layout bugs were found. However, **do not launch the 4–10 hour RTX 5080 sweep or publish CUDA-graph tables until the 4 `HIGH` findings (H1–H4) and top `MEDIUM` fixes (M1–M3) below are patched** (~40 lines total). Launching as-is risks an immediate post-Trial-1 crash on nested output paths, speed-biased wall-clock LR schedules (especially crippling `120m` wall runs), non-comparable `eager_event_ms` timings against non-capturable T1-04 kernels, and silent summary corruption under `--resume` or layout ablations.

---

## Verified Strengths (Cycle 1 Remediation Confirmed)

| Area | Verification Result | Evidence |
|---|---|---|
| **Data shards & disjointness** | **PASS** — `real_text_data.py --verify` passed on disk. `train` (`26,004,518` tok / `24,256` docs), `val` (`2,002,330` / `1,804`), `test` (`2,000,081` / `1,960`) all have exact `EOT` counts matching document counts, token range `[0, 50256]`, `0` exact duplicates across splits, and negligible 64-token prefix overlap (`2/1804` train∩val = `0.11%`, `5/1960` train∩test = `0.26%`). | [real_text_data.py:50-193](/home/ryan/Documents/projects/ai/research/polar_opt/real_text_data.py:50), [manifest.json:1-49](/home/ryan/Documents/projects/ai/artifacts/real_text/manifest.json:1) |
| **Paired init, batches & warmup isolation** | **PASS** — Every optimizer per seed loads the identical CPU `init` state dict, trains on an isolated `np.random.default_rng(seed)` stream, runs warmup on `seed + 1_000_000` with a disposable `factory()` instance, reloads `init`, and constructs a fresh `optimizer` via `factory()`. Fingerprints of both are saved per row. | [harness_tier2_real.py:114-177](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:114), [harness_tier2_real.py:501-530](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:501) |
| **Parameter partitioning & Muon math** | **PASS** — Exhaustive disjoint partition: 2D block weights (`60` matrices) use Muon (Nesterov `β=0.95`, decoupled `1 - lr*wd`, RMS scale `sqrt(max(1, rows/cols))`); tied `embedding.weight` (`2D`) gets fused AdamW with `wd`; 1D `RMSNorm` weights get fused AdamW with `wd=0`. Matches `AdamW` decay partitioning. | [harness_tier2_lm.py:62-94](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_lm.py:62), [harness_tier2_real.py:182-293](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:182) |
| **Contiguous FP32 update de-confounding** | **PASS** — Default `update_dtype="fp32"` converts BF16/non-contiguous tall updates (`pe_*`, `jordan_*`) once per shape group to contiguous FP32 so `torch._foreach_add_` stays on its single-kernel path for all comparators; `ps_*` contiguous FP32 outputs take the zero-copy no-op path. `StepTimer` transparently splits `polar_ortho_ms`, `layout_cast_ms`, and `foreach_update_ms`. | [harness_tier2_real.py:265-290](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:265), [test_tier2_real.py:139-154](/home/ryan/Documents/projects/ai/tests/test_tier2_real.py:139) |
| **Zero-overshoot rollback & post-Nesterov harvest** | **PASS** — Pre-step GPU weight snapshot cleanly restores parameters on deadline overrun before immediate `break` (no mutable buffers exist in `Transformer`, and optimizer/RNG are never used after rollback). `harvest_nesterov` captures the exact post-Nesterov `lerp(grad, momentum, beta)` orthogonalizer inputs with FP64 `svdvals`. | [harness_tier2_real.py:521-556](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:521), [harness_tier2_real.py:662-708](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:662) |
| **Williams Latin square balance** | **PASS** — Verified mathematically for even `n` (`n` blocks) and odd `n` (`2n` blocks with reversal); every method holds every position and precedes every other method equally often. | [bench_cuda_graph.py:49-60](/home/ryan/Documents/projects/ai/research/polar_opt/bench_cuda_graph.py:49) |

---

## Severity-Ranked Findings

### `CRITICAL`
None.

---

### `HIGH`

#### [H1] Interleaved timing clocks in `bench_cuda_graph.py` create an asymmetric GPU runway that biases `eager_event_ms` against non-capturable methods (including T1-04's `schatten_switch_blackwell`)
- **Location:** [bench_cuda_graph.py:111-130](/home/ryan/Documents/projects/ai/research/polar_opt/bench_cuda_graph.py:111)
- **Proof & Reasoning:** Inside the single inner `for _ in range(iters):` loop, each iteration enqueues `cuda_graph` (lines 112–118), then `eager_event` (lines 119–124), then `wall` with two `torch.cuda.synchronize()` calls (lines 125–130).
  1. Because `wall` synchronizes at line 129, every iteration starts from an **idle GPU**.
  2. For a **capturable** method (`name in graphs`, e.g., `muon_ps3_blackwell`), `eager_event` is queued behind `flush1 (~260 µs) + graph.replay() (125 µs – 10 ms) + flush2 (~260 µs)`—giving Python a **≥645 µs to >10 ms GPU backlog** that hides host launch gaps before `start.record()` fires.
  3. For a **non-capturable** method (`name not in graphs`—directly hitting Cycle 2 `T1-04`'s [schatten_switch_blackwell.py:38-39](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:38), which performs a host norm read and fails graph capture), `eager_event` starts from an idle GPU with only `flush2 (~260 µs)` of runway. Even among capturable methods, slower `graph.replay()` gives longer launch masking to `eager_event`.
  4. Additionally, running `iters=5` consecutive iterations of `(graph, eager_event, wall)` for the same `name` executes 15 back-to-back calls of the same method, leaving CPU instruction/Python/allocator caches hot for `wall` and diluting block-level carryover control.
- **Minimal Fix:** Split `cuda_graph`, `eager_event`, and `wall` into **three separate passes** (each iterating the Williams `blocks`, with a `torch.cuda.synchronize()` between passes), or synchronize before each `eager_event` flush so every method sees an identical pre-`start` GPU queue state.

#### [H2] Wall-clock LR schedule couples step-based warmup with time-based cosine decay, creating speed-dependent peak LR bias and collapsing `120m` wall-run LR
- **Location:** [harness_tier2_real.py:373-380](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:373), [harness_tier2_real.py:538-539](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:538), [harness_tier2_real.py:743-744](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:743)
- **Proof & Reasoning:**
  ```python
  warm = min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
  return warm * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
  ```
  1. Cosine decay starts at `progress = 0` *during* warmup instead of *after* warmup, so the multiplier never reaches `1.0`.
  2. Under `budget == "wall"`, warmup counts **steps** (`(step + 1) / warmup_steps`) while cosine decay counts **seconds** (`elapsed / seconds`). A slower optimizer (e.g., `muon_pe_eager`) finishes step `50` at a larger fraction `p_slow > p_fast` of `seconds` than a faster optimizer (`muon_ps3_blackwell`), so the slower optimizer both spends more of its wall budget in warmup *and* peaks at a lower effective LR (`cos(π p_slow) < cos(π p_fast)`).
  3. **Severe sweep-default failure on `120m`:** With default `--seconds 120.0` and `--lr-warmup-steps 50`, a `120m` eager run (`32,768` tok/step, ~1.3–2.2s/step on RTX 5080) completes only ~55–90 steps total. Thus **55–90% of the wall budget is warmup**, and at step 49 (`progress ≈ 0.55–0.90`), the peak LR collapses to **0.12–0.48× nominal LR**, breaking LR bracketing on `120m`.
- **Minimal Fix:** Decouple warmup from cosine decay:
  - Token budget: linear ramp over `step < warmup_steps`, then cosine decay over `(step - warmup_steps) / max(1, total_steps - warmup_steps)`.
  - Wall budget: define warmup by wall-clock fraction (e.g., `warmup_frac = min(0.1, lr_warmup_steps / (tokens / step_tokens))` or `--wall-warmup-fraction 0.0625`), ramping linearly over `elapsed < warmup_seconds` and applying cosine decay over `(elapsed - warmup_seconds) / (seconds - warmup_seconds)`. Also scale `--seconds` per preset so wall budgets cover a meaningful horizon (e.g. ≥ 300–400 steps).

#### [H3] Crash after Trial #1 on new `--output` directories, plus `--resume` silently mixes/skips runs across changed configs
- **Location:** [harness_tier2_real.py:721-722](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:721), [harness_tier2_real.py:806-836](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:806)
- **Proof & Reasoning:**
  1. **Immediate crash:** `log_path = args.output.with_suffix(".jsonl")` is opened in append mode at line 835 (`with log_path.open("a")`) **without** creating `log_path.parent`. `mkdir(parents=True, exist_ok=True)` only runs inside `_write()` at line 845 after all trials finish. Passing e.g. `--output results/t3_01/50m.json` runs Trial #1 to completion and then crashes with `FileNotFoundError`. (Tests missed this because `tmp_path` already exists.)
  2. **Silent contamination on `--resume`:** `_run_key(row)` only hashes `(optimizer, lr, seed, budget, update_dtype)`—omitting `preset`/`config`, `target_tokens`, `target_seconds`, `auxiliary_lr`, `weight_decay`, `lr_warmup_steps`, `min_lr_ratio`, `compile_model`, and `data` SHA-256. Resuming after changing any of those flags silently skips new trials and merges incompatible rows from `.jsonl` into `summarize(rows, args.reference)`.
  3. **Order drift:** Line 820 seeds the shuffle with `f"{seed}-{budget_index}"` instead of `f"{seed}-{budget}"`, so running `--budget wall` alone uses a different shuffle order than the `"wall"` phase of `--budget both`.
- **Minimal Fix:** Call `log_path.parent.mkdir(parents=True, exist_ok=True)` before line 808; add a `run_config_hash` (covering preset/config, budgets, hyperparams, `compile_model`, and shard SHA-256s) to every row and `_run_key`, and on `--resume` reject mismatched `run_config_hash` (or filter to the active plan); change line 820 to `f"{seed}-{budget}"`.

#### [H4] Non-deterministic `reference` layout matching and `None`-division crash in `summarize()`
- **Location:** [harness_tier2_real.py:628-652](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:628)
- **Proof & Reasoning:**
  1. `best` is populated by iterating over an unordered Python `set` at line 628 (`{(g["budget"], g["optimizer"], g["update_dtype"]) for g in grid}`). If `rows` contains both `update_dtype="fp32"` and `"native"` (explicitly supported by `_run_key` for layout ablations), `best` holds both `(budget, reference, "fp32")` and `(budget, reference, "native")`. Line 639 (`ref_key = next((k for k in best if k[0] == budget and k[1] == reference), None)`) then picks whichever layout happens to come first under `PYTHONHASHSEED`, randomly comparing `fp32` runs against a `native` reference or vice versa.
  2. If a wall-clock run completes `0` steps (`elapsed == 0.0`), [harness_tier2_real.py:579](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:579) sets `tokens_per_second = None` while `val_loss` / `test_loss` remain finite (untrained init → `diverged = False`). Line 651 (`mine[s]["tokens_per_second"] / ref[s]["tokens_per_second"]`) then crashes with `TypeError`.
  3. Lines 642–643 (`{r["seed"]: r for r in cells[...]}`) silently overwrite duplicate seeds if `.jsonl` contains duplicates.
- **Minimal Fix:** Resolve `ref_key` deterministically: prefer `(budget, reference, layout)` if present, else `(budget, reference, "fp32")`, else `(budget, reference, None)`. Treat `steps == 0` as invalid (`diverged = True` or skip `None` throughput), and assert unique seeds per cell.

---

### `MEDIUM`

#### [M1] `bench_cuda_graph.py` omits `allow_fp16/bf16_reduced_precision_reduction = False`, crashes on `NaN` at JSON write, and lacks input-mutation guard
- **Location:** [bench_cuda_graph.py:89-104](/home/ryan/Documents/projects/ai/research/polar_opt/bench_cuda_graph.py:89), [bench_cuda_graph.py:146-153](/home/ryan/Documents/projects/ai/research/polar_opt/bench_cuda_graph.py:146), [bench_cuda_graph.py:202](/home/ryan/Documents/projects/ai/research/polar_opt/bench_cuda_graph.py:202)
- **Proof & Reasoning:**
  1. [harness_tier2_real.py:763-765](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:763) disables `allow_fp16_reduced_precision_reduction` and `allow_bf16_reduced_precision_reduction`, whereas `bench_cuda_graph.py` leaves PyTorch's `True` defaults—allowing cuBLAS to pick different precision/kernels in kernel benchmarks vs training.
  2. If any `--extra` candidate returns `NaN`/`Inf`, `(graphs[name][1].float() - eager).abs().max().item()` at line 101 stores `NaN`/`Inf`, and the script runs all 5 shapes only to crash at line 202 (`json.dumps(..., allow_nan=False)`) and discard all results.
  3. All methods share one mutable input `G` (line 89) with no check that `G` was not modified in-place by a candidate.
  4. `run()` defaults to `blocks=4, iters=10` (line 146–147), which is **not** Williams-balanced for the 7 `DEFAULT_METHODS` if `run()` is called programmatically (only `main()` computes `2 * len(names)` at line 197).
- **Minimal Fix:** Disable both reduced-precision reduction flags in `run()`; check finiteness of `eager` and `graphs[name][1]` (storing `finite: bool` and `None` if non-finite); verify `torch.equal(G, G_ref)` after capture/warmup; default `blocks: int | None = None` in `run()` and compute the Williams block count inside `run()`.

#### [M2] `--compile-model` across 192 fresh `Transformer` instances is untested and incurs per-trial recompilation hazards; `@functools.cache` on `_compiled_polar_express` risks Dynamo `recompile_limit=8` fallback
- **Location:** [harness_tier2_real.py:299-303](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:299), [harness_tier2_real.py:505](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:505), [harness_tier2_real.py:571-573](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:571)
- **Proof & Reasoning:**
  1. `run_trial()` instantiates a fresh `Transformer(config)` and wraps it in `torch.compile(model)` on every trial (up to 192 times in one process), then calls `model.eval()` before `evaluate(forward, ...)`, triggering a second compiled eval graph (`self.training == False`) per trial. `tests/test_tier2_real.py` has **zero** coverage of `compile_model=True` or `muon_pe_compiled`. Without reusing one compiled model instance per preset (reloading `init` via `load_state_dict`) or resetting Dynamo cleanly, multi-trial sweeps risk Dynamo cache growth or silent eager fallback if any guard specializes on instance/buffer identity.
  2. `_compiled_polar_express()` uses `torch.compile(polar_express, dynamic=False)` cached process-wide. Each distinct matrix shape adds a guard on `polar_express.__code__` (4 shapes per LM preset, 5 in `bench_cuda_graph.py`); exceeding PyTorch's default `recompile_limit=8` in one process (e.g., running `50m` + `120m` + tests or extra shapes) silently falls back to eager.
- **Minimal Fix:** Add a multi-trial test for `compile_model=True` and `muon_pe_compiled`; pass `evaluate(model, ...)` or hoist a single compiled `Transformer` per `(preset, device)` in `main()` (reloading `init` between trials); bump `torch._dynamo.config.cache_size_limit` (e.g., to `32`) or avoid sharing `_compiled_polar_express` across unrelated shape regimes.

#### [M3] `--extra-optimizer` hook (`_requires_arguments`) misclassifies `*args`/`**kwargs` wrappers and required keyword-only parameters
- **Location:** [harness_tier2_real.py:328-349](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:328)
- **Proof & Reasoning:** `_requires_arguments()` only inspects `POSITIONAL_ONLY` and `POSITIONAL_OR_KEYWORD` parameters.
  - A direct orthogonalizer wrapper with `(*args, **kwargs)` (common in decorators and `torch.compile` wrappers) returns `False` and is mistakenly called as a 0-arg factory `obj()`, crashing inside `build_optimizer()`.
  - A class or function with a required `KEYWORD_ONLY` parameter (e.g., `def make(*, variant): ...`) also returns `False`, bypassing the line 346 class guard and failing lazily inside `run_trial()` instead of at CLI parse time.
- **Minimal Fix:** Include `p.KEYWORD_ONLY` in `_requires_arguments()`, and treat `*args`-only callables as direct orthogonalizers (or require explicit factory convention / validate with a dummy call check at CLI startup). Note: `schatten_switch_blackwell(G, variant=None, ...)` does have a required positional `G` so it passes as-is, but compiled/wrapped variants would hit this bug.

#### [M4] Sweep statistical power, runtime sizing, sampling-with-replacement, and fixed `--muon-aux-lr` on `120m`
- **Location:** [harness_tier2_real.py:159-162](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:159), [harness_tier2_real.py:648-652](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:648), [harness_tier2_real.py:735-743](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:735)
- **Proof & Reasoning:**
  1. **Wide 3-seed CIs:** Default `--seeds 0 1 2` yields `t(0.975, df=2) = 4.303`, giving 95% CI half-width `2.48 × sd` (2× wider than `n=5` seeds recommended in Cycle 1 Opus M3).
  2. **Sweep duration & shard sizing:** Default CLI runs `8 optimizers × 4 LRs × 3 seeds × 2 budgets = 192` trials (~10+ hrs on `50m`, ~16–20 hrs on `120m` eager). Also, `BatchStream` samples `26.2M` tokens **with replacement** from `26.0M`-token `train.bin` (`~37%` of tokens unseen, `~26%` seen `≥2×` with overlapping shifted windows). Do not raise `--tokens` above `26M` without rebuilding `train.bin` to avoid multi-epoch repetition.
  3. **Fixed `--muon-aux-lr 2e-3` on `120m`:** Tied `embedding.weight` is `40.1%` of `50m` (`25.8M/64.2M`) and `31.2%` of `120m` (`38.6M/123.6M`). Fixing `aux_lr` preserves equal 4-point LR tuning budgets between AdamW and Muon, which is fair—but `2e-3` may run hot on `120m` (`width=768`). Pilot-verify `120m` AdamW's winning LR before locking `aux_lr=2e-3` for a 10h sweep.
  4. **Post-selection `val_loss_diff` & ratio CI:** `val_loss_diff` is computed on the selection split (report only `test_loss_diff` as confirmatory). `tokens_per_second_ratio` averages raw ratios rather than log-ratios (asymmetric; minor at low CV).

---

### `LOW`

#### [L1] Silent `nvidia-smi` failure passes tests; missing tests for `--resume`, partitioning, `steps==0`, and layout ablations
- **Location:** [test_tier2_real.py:204](/home/ryan/Documents/projects/ai/tests/test_tier2_real.py:204), [harness_tier2_real.py:437-438](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:437)
- **Reasoning:** `gpu_telemetry()` returns `{"error": ...}` on `nvidia-smi` failure, which is truthy—so `assert row["gpu_start"]` at `test_tier2_real.py:204` passes silently even if GPU telemetry is completely broken. Tests also lack coverage for: (a) `--resume` and nested `--output` dirs, (b) `compile_model=True` / `muon_pe_compiled`, (c) `summarize()` with mixed `fp32`/`native` layouts and `steps == 0`, and (d) exhaustive `Muon` parameter partitioning (`matrices` + `auxiliary` == all params, `wd=0` on 1D norms).

#### [L2] Minor systems & data nits (`StepTimer` event asymmetry, wall snapshot VRAM/L2, off-by-1 token bound, EOT check)
- **Locations:** [harness_tier2_real.py:161](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:161), [harness_tier2_real.py:171](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:171), [harness_tier2_real.py:271-293](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:271), [harness_tier2_real.py:522](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:522), [real_text_data.py:181](/home/ryan/Documents/projects/ai/research/polar_opt/real_text_data.py:181)
- **Reasoning:**
  - `Muon.step()` allocates/records `19` CUDA events inside `step_seconds` vs `3` for `AdamW` (~0.15–0.4 ms/step, ~0.05% of a 500 ms step; equal across all Muon variants).
  - Wall-budget `snapshot` holds `+257 MB` (`50m`) / `+495 MB` (`120m`) extra GPU VRAM and runs `_foreach_copy_` before every step (altering pre-step L2 state vs token-budget runs; equal across optimizers within wall mode). Note also that if mid-trial checkpointing/resumption were ever added after rollback, optimizer moments and `stream.rng` are not rolled back (harmless currently since rollback immediately `break`s).
  - `BatchStream.next_starts()` uses `len(self.tokens) - len(self.span)`, excluding the final token of `train.bin` (`+1` fixes it), and returns non-contiguous slices `windows[..., :-1], windows[..., 1:]` (causing a tiny ~64 KB copy in `y.flatten()` inside `train_step`).
  - `real_text_data.py:181` checks `(data == EOT).sum() < entry["documents"]`; since `encode_ordinary_batch` never emits `50256`, tightening `<` to `!=` is stricter and already holds on disk.

---

## Minimal Pre-Sweep Remediation Checklist (Ordered)

1. **Fix `harness_tier2_real.py` wall-clock LR schedule (`H2`):** Decouple warmup from cosine decay and measure wall-budget warmup as a fixed fraction of `seconds` so peak LR is `1.0` for every optimizer regardless of throughput; tune `--seconds` per preset so wall runs exceed ~300 steps.
2. **Fix `harness_tier2_real.py` `.jsonl` directory creation, `--resume` config hash, and shuffle key (`H3`):** Add `log_path.parent.mkdir(parents=True, exist_ok=True)` before trial loop; hash config/budget/data into `_run_key` and validate on `--resume`; shuffle by `f"{seed}-{budget}"`.
3. **Fix `harness_tier2_real.py` `summarize()` reference lookup & `steps == 0` guard (`H4`):** Deterministically prefer matching `(budget, reference, layout)` → `"fp32"` → `None`, and mark `steps == 0` trials invalid before computing throughput ratios.
4. **Fix `bench_cuda_graph.py` clock isolation, cuBLAS reduction flags, `NaN` safety, and input-mutation check (`H1`, `M1`):** Run `cuda_graph`, `eager_event`, and `wall` in separate passes; disable `allow_fp16/bf16_reduced_precision_reduction`; handle non-finite outputs cleanly before `json.dumps(..., allow_nan=False)`; verify `G` invariance; default `blocks=None` inside `run()`.
5. **Harden `--extra-optimizer`, `--compile-model`, and tests (`M2`, `M3`, `L1`):** Handle `KEYWORD_ONLY` and `*args` wrappers in `_requires_arguments()`; reuse a single compiled `Transformer` per preset (or test multi-trial `--compile-model`); assert `"error" not in row["gpu_start"]` and add CPU unit tests for `--resume`, mixed layouts, `steps == 0`, and parameter partitioning.
6. **Sweep execution discipline (`M4`):** Run a quick pilot on `50m`/`120m` to confirm `--muon-aux-lr` and step times before committing 4–10h; prefer `5` seeds on the final shortlisted comparators (`adamw`, `muon_pe_eager`, `muon_pe_compiled`, `muon_ps3_eager`, `muon_ps3_blackwell`, and T1-04 winner) rather than spending 10+ hours across all 8 defaults at 3 seeds.
