# Cycle 2 Adversarial Audit Report — Auditor 4 (Gemini 4 Neon)

- **Auditor:** Gemini 4 Neon (`thought_level: high`) · **Date:** 2026-11-25
- **Target:** `/home/ryan/Documents/projects/ai` at committed `main` HEAD `877c2a8413e05d89f105360a550eb382e464ba70` (`877c2a8`, code commit `9a23a61`)
- **Mode:** Strictly read-only / zero GPU execution. Pure CPU/stdlib JSON and code verification; `MacBook Pro` untouched.
- **Scope:** (1) Full verification of pre-sweep findings (`H1–H4`, `M1–M4`, `L1–L2`) from [AUDIT_cycle2_t3_01_presweep_gemini_neon.md](/home/ryan/Documents/projects/ai/audits/AUDIT_cycle2_t3_01_presweep_gemini_neon.md); (2) cross-artifact consistency & late-2026 SOTA honesty audit across all Cycle 2 JSONs, Blackwell Triton/CUDA-graph kernels, MLX/Metal code, and Cycle 1 state/docs; (3) promotion gates and falsifiable Cycle 3 roadmap.

---

## Executive Verdict: **`PASS_WITH_CAVEATS`**

> **Bottom Line:** Cycle 2 executes a rigorous, zero-divergence, scientifically honest leap over Cycle 1—**zero `CRITICAL` bugs** were found, every `HIGH` pre-sweep finding (`H1–H4`) was remediated before the RTX 5080 sweep, all benchmark JSONs were produced on a clean tree (`git_dirty: false` at `9a23a61`), pairing/rollback/Williams-balance invariants hold across every single run, and `schatten_switch_blackwell.py` achieves **100% sync-free CUDA-graph capture** (`0/55` capture errors, `0.0` max graph-vs-eager diff).
>
> However, **four `HIGH` promotion/honesty gates** block an unconditional `PASS` and must be reflected accurately when updating `README.md`, `state/RESEARCH_STATE.json`, `state/BACKLOG.md`, and the Cycle 2 infographic:
> 1. **`schatten_switch_search_cycle2.json` contains stale pre-sync-free timings** (`0.42×` at `512×2048`) that contradict `cycle2_cuda_graph_bench.json` (`0.90×`), and `README.md` / `state/*` are still frozen at Cycle 1.
> 2. **`SS3_34` (7 squarings) is NOT a lossless LM drop-in:** its `lo = 0.008` cutoff causes severe early-step Nesterov error (`0.5223` median / `0.8459` max at step 20) and regresses `50m` FineWeb-Edu loss by `+0.043` val / `+0.042` test at 128 steps.
> 3. **No end-to-end `SS3_*` speedup over `PS3_55_blackwell` is demonstrated yet on real LM training:** `50m` matrices all have short side `n = 512` (below the `n >= 1024–2048` kernel crossover), so `SS3_54` (`3.23 ms`) and `SS3_53` (`3.10 ms`) tie/trail `PS3_55_blackwell` (`3.08 ms`) on `50m`, and `120m` (`n = 768`) remains untested. Moreover, `+2.45%` of `50m`'s `+3.18%` throughput win over `PolarExpress` comes from eager `PS3_55` itself (`+0.72%` incremental from Blackwell Triton kernels).
> 4. **Apple M5 Max `T2-02` (`kernels_mlx.py` Metal SYRK/symprod + BF16/FP16) has zero live hardware verification.**

---

## 1. Pre-Sweep Remediation Verification Matrix (`H1–H4`, `M1–M4`, `L1–L2`)

Cross-checked against [AUDIT_cycle2_t3_01_presweep_gemini_neon.md](/home/ryan/Documents/projects/ai/audits/AUDIT_cycle2_t3_01_presweep_gemini_neon.md), committed code at `877c2a8`, and Cycle 2 JSONs:

| ID | Pre-Sweep Finding | Status | Verification Evidence & Residual Notes |
|---|---|---|---|
| **H1** | Interleaved `cuda_graph`/`eager_event`/`wall` clocks created asymmetric GPU runway against non-capturable methods | **REMEDIATED (100%)** | [bench_cuda_graph.py:120-156](/home/ryan/Documents/projects/ai/research/polar_opt/bench_cuda_graph.py:120) splits timing into 3 isolated passes with `torch.cuda.synchronize()` between passes. Plus [schatten_switch_blackwell.py:379-711](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:379) is now fully sync-free and CUDA-graph capturable (`0/55` capture errors in [cycle2_cuda_graph_bench.json](/home/ryan/Documents/projects/ai/results/cycle2_cuda_graph_bench.json)). |
| **H2** | Wall-clock LR schedule coupled step warmup with time cosine, biasing peak LR by speed and collapsing `120m` wall LR | **PARTIALLY REMEDIATED** | Core math **100% fixed**: [harness_tier2_real.py:379-407](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:379) decouples warmup from cosine in both token mode (`total_steps`) and wall mode (`wall_warmup_fraction=0.0625`), reaching exact peak `1.0` regardless of throughput ([test_tier2_real.py:163-168](/home/ryan/Documents/projects/ai/tests/test_tier2_real.py:163)). **Deferred/nits:** `--seconds` default (`120.0`, [harness_tier2_real.py:838](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:838)) is not scaled per preset (`120m` remains untested), and `harvest_nesterov` ([harness_tier2_real.py:756](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:756)) omits `total_steps=steps`. |
| **H3** | Post-Trial-1 crash on nested `--output` dirs; `--resume` silently mixed configs; shuffle key used `budget_index` | **REMEDIATED (minor edge case)** | `log_path.parent.mkdir(parents=True, exist_ok=True)` added before trial loop ([harness_tier2_real.py:921](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:921)); `_config_hash()` covers config, shard SHA-256, budgets, hyperparams, and `compile_model` ([harness_tier2_real.py:783-817](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:783)); shuffle key fixed to `f"{seed}-{budget}"` ([harness_tier2_real.py:949](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:949)); tested in [test_tier2_real.py:332-351](/home/ryan/Documents/projects/ai/tests/test_tier2_real.py:332). **Minor edge case:** lines `926–927` skip hash validation for `.jsonl` rows whose `budget` is not in the current invocation's `budgets`. Verified clean in `cycle2_tier2_real.json` (exact 2 hashes: `c8d85b6e90bb3296` / `bc391caca4de9246`). |
| **H4** | Non-deterministic `reference` layout matching, `steps==0` `None`-division crash, silent duplicate-seed overwrite | **REMEDIATED (100%)** | [harness_tier2_real.py:625-706](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:625) sets `diverged=True` on `steps==0`, sorts `group_keys`, resolves `ref_key` deterministically (`layout -> "fp32" -> None`), raises `ValueError` on duplicate seeds, and filters `None` throughput. Verified in [test_tier2_real.py:354-369](/home/ryan/Documents/projects/ai/tests/test_tier2_real.py:354). |
| **M1** | `bench_cuda_graph.py` missing reduced-precision flags, `NaN` JSON crash, input-mutation check, Williams default in `run()` | **REMEDIATED (100%)** | [bench_cuda_graph.py:91,104-116,172-178](/home/ryan/Documents/projects/ai/research/polar_opt/bench_cuda_graph.py:91) disables both `allow_fp16/bf16_reduced_precision_reduction`, stores `finite` / `None` safely before `allow_nan=False`, enforces `torch.equal(G, G_ref)`, and computes Williams `blocks` inside `run()`. Tested in [test_tier2_real.py:404-420](/home/ryan/Documents/projects/ai/tests/test_tier2_real.py:404). |
| **M2** | `--compile-model` per-trial recompilation & `_compiled_polar_express` `recompile_limit=8` | **PARTIALLY REMEDIATED** | **Code 100% fixed:** `main()` hoists one `reusable_model = (shared_model, torch.compile(shared_model))` ([harness_tier2_real.py:935-940](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:935)), `evaluate(model, ...)` runs uncompiled ([harness_tier2_real.py:618-619](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:618)), and Dynamo `cache_size_limit >= 32` is enforced ([harness_tier2_real.py:303,536,937](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:303)). Empirically rock-steady across 70 compiled trials (`fwd_bwd_ms = 222.74–222.89 ms`). **Remaining gap:** zero unit test in `test_tier2_real.py` exercises `compile_model=True` or `muon_pe_compiled`. |
| **M3** | `_requires_arguments()` misclassified `*args` wrappers and required `KEYWORD_ONLY` params | **REMEDIATED (100%)** | [harness_tier2_real.py:329-340](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:329) checks `KEYWORD_ONLY` and `VAR_POSITIONAL`. Verified in [test_tier2_real.py:372-390](/home/ryan/Documents/projects/ai/tests/test_tier2_real.py:372). |
| **M4** | 3-seed wide CIs, runtime sizing, replacement sampling, fixed `muon_aux_lr=2e-3` on `120m`, `val_loss_diff`/ratio CI | **PARTIALLY REMEDIATED** | Upgraded main sweep to **`n = 5` seeds** (`t_{0.975,4} = 2.776`, halving CI width), piloted `50m` AdamW LR (`0.001` beat `0.002` by `-0.674` val loss) and lowered `--muon-aux-lr` to `0.001`, and kept tokens at `10.49M` (`0.40×` of `26.0M` shard). **Deferred:** `120m` preset was never piloted or run; 5-seed sweep locked single LRs from 1-seed 128-step pilots (`lr_at_grid_edge: true`); `summarize()` still emits raw arithmetic mean of `tokens_per_second_ratio`. |
| **L1** | Silent `nvidia-smi` failure in tests; missing tests for `--resume`, partitioning, `steps==0`, mixed layouts, `compile_model` | **MOSTLY REMEDIATED** | [test_tier2_real.py:210](/home/ryan/Documents/projects/ai/tests/test_tier2_real.py:210) now asserts `"error" not in row["gpu_start"]`; tests added for `--resume`/nested output, mixed layouts, partitioning, and CUDA-graph NaN/mutation (`332–420`). **Remaining:** no `compile_model=True` test and no end-to-end `run_trial(..., seconds=1e-9)` `steps==0` test. |
| **L2** | `StepTimer` 19 vs 3 events, wall snapshot VRAM/L2, `BatchStream` off-by-1 & non-contiguous slices, EOT `<` vs `!=` | **PARTIALLY REMEDIATED** | **Fixed:** EOT check tightened to `!=` ([real_text_data.py:181](/home/ryan/Documents/projects/ai/research/polar_opt/real_text_data.py:181)) and `BatchStream.next_starts()` off-by-1 fixed with `+ 1` ([harness_tier2_real.py:161](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:161)). **Retained by design:** `StepTimer` still records `19` events for `Muon` vs `3` for `AdamW` ([harness_tier2_real.py:271-293](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:271); equal across Muon variants, `<0.1%` vs AdamW), and wall snapshot `_foreach_copy_` remains for exact zero-overshoot rollback. |

---

## 2. Cross-Artifact Consistency & Late-2026 SOTA Honesty Audit

### 2.1 FLOPs Theory vs. Measured CUDA-Graph Speedups ([cycle2_cuda_graph_bench.json](/home/ryan/Documents/projects/ai/results/cycle2_cuda_graph_bench.json))

All 11 methods across all 5 LLM shapes (`22` Williams blocks × `5` iters = `110` samples/method/shape, L2 flushed `4×`, isolated 3-pass clocks, `allow_fp16/bf16_reduced_precision_reduction = False`) captured cleanly with **`0/55` errors**, **`finite = True`**, and **`graph_vs_eager_max_abs = 0.0`**. Comparing leading symmetric FLOPs `(6α + s)n³` (where `α = m/n`, `s` = squarings: `PS3_55 = 10`, `SS3_54 = 9`, `SS3_53/SS3_35 = 8`, `SS3_34 = 7`) against measured `cuda_graph_ms` medians reveals a physically consistent crossover:

| Shape (`α = m/n`) | `pe_eager` (ms) | `pe_comp` (ms) | `ps3_eager` (ms) | `ps3_bw` (ms) | `ss3_54` (9 sq) vs `ps3_bw` (theo) | `ss3_53` (8 sq) vs `ps3_bw` (theo) | `ss3_34` (7 sq) vs `ps3_bw` (theo) | `ss3_34` vs `pe_eager` / `pe_comp` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `(512, 2048)` (`α=4`) | 0.2212 | 0.2022 | 0.1765 | **0.1253** | 0.1393 / `0.900×` (`1.030×`) | 0.1352 / `0.927×` (`1.062×`) | 0.1265 / `0.990×` (`1.097×`) | `1.748×` / `1.598×` |
| `(1024, 4096)` (`α=4`) | 1.1551 | 1.1270 | 0.7680 | 0.5188 | 0.5489 / `0.945×` (`1.030×`) | 0.5280 / `0.983×` (`1.062×`) | **0.5059** / `1.025×` (`1.097×`) | `2.283×` / `2.228×` |
| `(2048, 8192)` (`α=4`) | 7.9379 | 7.5592 | 4.9177 | 3.1087 | 3.0935 / `1.005×` (`1.030×`) | 2.9999 / `1.036×` (`1.062×`) | **2.8853** / `1.077×` (`1.097×`) | `2.751×` / `2.620×` |
| `(4096, 4096)` (`α=1`) | 20.1022 | 19.4718 | 19.8815 | 10.5201 | 10.0746 / `1.044×` (`1.067×`) | 9.4495 / `1.113×` (`1.143×`) | **8.7693** / `1.200×` (`1.231×`) | `2.292×` / `2.220×` |
| `(4, 1024, 4096)` (`α=4`) | 4.3618 | 4.3442 | 3.1257 | 1.8828 | 1.8647 / `1.010×` (`1.030×`) | 1.8096 / `1.040×` (`1.062×`) | **1.7357** / `1.085×` (`1.097×`) | `2.513×` / `2.503×` |

- **Internal consistency verified:** At `n >= 2048` and batched `(4, 1024, 4096)`, every `SS3_*` variant tracks the theoretical `(6α + 10)/(6α + s)` FLOPs ceiling within `1.2–3.1%` (the residual `~2%` is the 4 extra `O(n²)` auxiliary Triton launches for safe FP64 norm reduction + dual Gram/square norm + coefficient folding + `_symprod_ptr_kernel` global coeff loads). At `n = 512`, those `~15–20 µs` of fixed auxiliary launches outweigh saving 1–3 tiny `512×512` squarings, explaining why `SS3_54` (`0.900×`) and `SS3_53` (`0.927×`) trail `PS3_55_blackwell` at `n = 512` — which matches their exact `polar_ortho_ms` ordering in `50m` Tier-2 training (`3.08 ms` `ps3_bw` vs `3.10 ms` `ss3_53` vs `3.23 ms` `ss3_54`)!
- **Dense vs. symmetric FLOPs check against `PolarExpress`:** At `(4096, 4096)` (`α=1`), `pe_eager` runs dense `(20α + 10)n³ = 30n³` vs `ps3_blackwell` symmetric `16n³` (`30/16 = 1.875×`; measured `1.911×`), and `pe_triton` runs symmetric SYRK `20n³` vs `ps3_blackwell` `16n³` (`20/16 = 1.250×`; measured `13.3328 / 10.5201 = 1.267×`). `torch.compile(polar_express)` (`muon_pe_compiled`) gains only `0.4–9.4%` over eager `PE`, confirming Cycle 1's speedup was NOT an uncompiled-eager artifact.

### 2.2 Held-Out Real Nesterov Behavior vs. `50m` FineWeb-Edu Convergence

Cross-checking [cycle2_heldout_nesterov_eval.json](/home/ryan/Documents/projects/ai/results/cycle2_heldout_nesterov_eval.json) (`45` real `50m` FineWeb-Edu post-Nesterov matrices across `early` step 20, `mid` step 200, `late` step 399; `0/45` divergences for all 10 methods) against [cycle2_tier2_pilot_muon.json](/home/ryan/Documents/projects/ai/results/cycle2_tier2_pilot_muon.json) (`128` steps) and [cycle2_tier2_real.json](/home/ryan/Documents/projects/ai/results/cycle2_tier2_real.json) (`5` seeds, `320` steps / `75s` wall) resolves the entire accuracy-vs-speed picture:

| Method | `lo` | All Med Err | Early (s20) Med / Max | Mid (s200) Med / p90 | Late (s399) Med / p90 | `50m` Pilot 128s Val / Test | `50m` 5-Seed Tok Test (`Δ` vs PE) | `50m` 5-Seed Wall Test (`Δ` vs PE) | `50m` Tok/s (`×` vs PE) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pe_eager` | 0.001 | 0.0960 | 0.1310 / 0.6187 | 0.0931 / 0.1934 | 0.0932 / 0.1802 | 6.1546 / 6.1930 | 5.4049 (`ref`) | 5.4158 (`ref`) | 138,585 (`1.000×`) |
| `pe_compiled` | 0.001 | — | — | — | — | — | 5.4012 (`-0.0037`) | 5.4150 (`-0.0008`) | 138,855 (`1.002×`) |
| `ps3_eager` | ~0.001 | — | — | — | — | — | 5.4001 (`-0.0049`) | 5.3898 (`-0.0260`) | 141,974 (`1.025×`) |
| `ps3_blackwell` | ~0.001 | 0.0619 | **0.0822 / 0.5909** | 0.0615 / 0.1720 | 0.0604 / 0.1585 | 6.1616 / 6.2000 | 5.4046 (`-0.0003`) | 5.3897* (`-0.0262*`) | **142,993 (`1.032×`)** |
| `ss3_54` (9 sq) | 0.00286 | **0.0389** | 0.2326 / 0.6903 | **0.0333 / 0.1581** | **0.0329 / 0.1388** | 6.1595 / 6.1980 | 5.4018 (`-0.0032`) | **5.3830 (`-0.0329`)** | 142,876 (`1.031×`) |
| `ss3_53` (8 sq) | 0.003 | 0.0548 | 0.3603 / 0.7441 | 0.0543 / 0.1938 | 0.0536 / 0.1862 | 6.1616 / 6.2002 | **5.3972 (`-0.0077`)** | 5.3844 (`-0.0314`) | 142,975 (`1.032×`) |
| `ss3_34` (7 sq) | 0.008 | 0.2146 | **0.5223 / 0.8459** | 0.0680 / 0.2920 | 0.0464 / 0.3140 | **6.1979 / 6.2351** | *(excluded)* | *(excluded)* | — |

*\*`ps3_blackwell` wall-budget seed 5 hit a `1.558 s` host stall (`320` vs `327` steps); seeds 1–4 averaged `5.3849` test loss @ `143,048 tok/s`.*

### 2.3 Honest Positioning vs. Late-2026 SOTA (`Jordan-5`, `PolarExpress` ICLR '26, `Gram NS` Jul '26, `Polar Switch PS2/PS3_55` Oct '26)

- **What Cycle 2 Verifiably Establishes:**
  1. **Genuine new algorithmic + systems contribution (`SS3_54` & `SS3_53`):** First `_symprod_kernel`-native, 3-buffer, **100% sync-free CUDA-graph capturable** Schatten-8 pre-scaled Polar Switch circuits with strictly fewer squarings than Du & Su's `PS3_55` (`9` and `8` vs `10`), certified `u_safe = 1.05` FP16 headroom, `0/117` Tier-1 + `0/45` held-out FineWeb-Edu Nesterov + `0/70` 5-seed LM divergences, **46% lower mid/late real-momentum polar error** (`SS3_54`: `0.033` vs `PS3_55` `0.061` and `PE` `0.093`), **`1.005–1.044×` (`SS3_54`) and `1.036–1.113×` (`SS3_53`) pure CUDA-graph speedups over `PS3_55_blackwell`** on large/batched LLM shapes (`2048×8192`, `4096×4096`, `(4,1024,4096)`), and **`2.00–2.65×`** over `PolarExpress` (`1.93–2.52×` vs `torch.compile(PE)`).
  2. **Clean real-text `50m` FineWeb-Edu transfer (5 seeds, `compile_model=True`, matched FP32 updates, zero-overshoot wall budgets):** `SS3_54`, `SS3_53`, and `PS3_55_blackwell` all achieve **lossless equal-token convergence parity** with `PolarExpress` (`test_diff` in `[-0.0077, -0.0003]`, all 95% CIs span `0`), **`+3.10% to +3.18%` end-to-end throughput** (`+2.90% to +2.98%` vs `muon_pe_compiled`), and statistically significant **`-0.026` to `-0.033` fixed-wall test loss improvements** (`p < 0.05` across all 5 seeds; `SS3_54` also beats `muon_ps3_eager` by `-0.00682 [-0.0119, -0.0018]` wall loss). All Muon variants beat tuned `AdamW` by **`-0.62` to `-0.64`** test loss.
  3. **BF16-safe reduced-squaring discovery (`SS3_35` & `SS3_34`):** Unlike `SS3_54`/`SS3_53` (FP16-only), `SS3_35` (8 sq) and `SS3_34` (7 sq) pass coherent rank-1 BF16 sweeps (`0/117` Tier-1 BF16, `0/45` held-out Nesterov BF16).
  4. **Verified negative discovery (`gram_space_restart`):** Replacing the second segment's `X X^T` restart with Gram-space recursion (`(3α + 13)n³`) diverges catastrophically in FP16 (`75/117` Tier-1 cases, `33/45` harvested, `σ_max` up to `73,058`), validating why both `Gram NS` and `Polar Switch` require explicit `X`-space restarts.
  5. **Cycle 1 confounder resolved:** [cycle2_tier2_real_native.json](/home/ryan/Documents/projects/ai/results/cycle2_tier2_real_native.json) proves the BF16 vs FP32 `_foreach_add_` layout difference is negligible (`<0.01%` throughput ratio: `1.0317` native vs `1.0318` fp32) at realistic `32,768` tok/step batches.

- **What Cycle 2 Does NOT Establish (Must NOT Be Overclaimed in README / State / Infographic):**
  1. Do **NOT** claim `SS3_54` or `SS3_53` beat `PS3_55_blackwell` in end-to-end LM training speed yet — on `50m` (`n = 512`) they are `0.9992×` / `0.9999×` vs `ps3_blackwell` (`3.23`/`3.10` vs `3.08 ms`), and `120m` (`n = 768`) was not run.
  2. Do **NOT** attribute the full `+3.18%` `50m` speedup over `PolarExpress` solely to Blackwell Triton kernels — `+2.45%` comes from Du & Su's `PS3_55` algorithm in eager PyTorch (`muon_ps3_eager`), and `+0.72%` is the incremental Blackwell Triton win at `n = 512`.
  3. Do **NOT** promote `SS3_34` (`+20%` at `4096×4096`) as a lossless LM drop-in — it regresses `50m` LM loss by `+0.043` val / `+0.042` test due to early-step spectral collapse (`0.5223` median error at step 20).
  4. Do **NOT** claim Apple M5 Max `T2-02` Metal SYRK or BF16/FP16 MLX wins — no M5 Max Cycle 2 benchmark has been executed on hardware.
  5. Do **NOT** present `AdamW` vs `Muon` (`+0.62` gap) as fully 320-step LR-bracketed — LRs were bracketed on 1 seed at 128 steps (`4.19M` tokens) and locked for the 5-seed 320-step sweep.

---

## 3. Severity-Ranked Findings

### `CRITICAL`
None.

---

### `HIGH`

#### [H1] Stale pre-sync-free timings in `schatten_switch_search_cycle2.json` contradict `cycle2_cuda_graph_bench.json`; `README.md` and `state/*` are stale at Cycle 1
- **Locations:** [schatten_switch_search_cycle2.json](/home/ryan/Documents/projects/ai/results/schatten_switch_search_cycle2.json), [cycle2_cuda_graph_bench.json](/home/ryan/Documents/projects/ai/results/cycle2_cuda_graph_bench.json), [README.md](/home/ryan/Documents/projects/ai/README.md), [RESEARCH_STATE.json](/home/ryan/Documents/projects/ai/state/RESEARCH_STATE.json), [BACKLOG.md](/home/ryan/Documents/projects/ai/state/BACKLOG.md)
- **Proof:** `schatten_switch_search_cycle2.json` was written at `16:48:54-0700` *before* the `17:06` sync-free Triton rewrite of `schatten_switch_blackwell.py`. Its `summary` / `latency_sanity` report `SS3_54` at `0.306 ms` (`0.421×` vs `PS3_55`) on `512×2048` and `0.390 ms` Tier-1 median (vs `PS3_55` `0.291 ms`). Post-rewrite `cycle2_cuda_graph_bench.json` (`17:19:27-0700`) measures `SS3_54` at `0.1393 ms` (`0.900×`) — **`2.20×` faster** than the stale search artifact. Meanwhile, no post-rewrite 117-case Tier-1 latency JSON exists, and `README.md` / `state/*` still report Cycle 1 (`531/531` tests, Cycle 2 "In Progress").
- **Action:** Treat `cycle2_cuda_graph_bench.json` as the sole authoritative Cycle 2 kernel latency source; annotate `schatten_switch_search_cycle2.json` as pre-sync-free search/accuracy provenance (or re-run Tier-1 latency once in Cycle 3); update `README.md`, `state/RESEARCH_STATE.json`, `state/BACKLOG.md`, and `reports/generate_infographic.py` using strictly reconciled Cycle 2 numbers.

#### [H2] `SS3_34` (7 squarings) fails LM convergence parity (`+0.043` val / `+0.042` test) due to early-step spectral collapse (`lo = 0.008`)
- **Locations:** [schatten_switch_blackwell.py:203-225](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:203), [cycle2_heldout_nesterov_eval.json](/home/ryan/Documents/projects/ai/results/cycle2_heldout_nesterov_eval.json), [cycle2_tier2_pilot_muon.json](/home/ryan/Documents/projects/ai/results/cycle2_tier2_pilot_muon.json)
- **Proof:** On real `50m` FineWeb-Edu step-20 Nesterov momenta, `SS3_34` suffers `0.5223` median / `0.8044` p90 / `0.8459` max polar error (`min_sigma_min = 0.0001` on `11.out.weight`), and still carries elevated mid/late tails (`p90 = 0.2920 / 0.3140` vs `PS3_55`'s `0.1720 / 0.1585`). In the 128-step `50m` pilot, `muon_ss3_34` loses to `muon_pe_eager` and `muon_ps3_blackwell` at all four LRs (`0.005, 0.01, 0.02, 0.04`), peaking `+0.0433` val / `+0.0421` test worse than `pe_eager`.
- **Action:** Keep `SS3_34` explicitly `CAVEATED` (unpromoted as standalone LM optimizer); only test it in Cycle 3 inside hybrid post-warmup schedules (`PS3_55`/`SS3_54` → `SS3_34`).

#### [H3] `50m` (`n = 512`) is below the `SS3_*` crossover: zero end-to-end speedup over `PS3_55_blackwell` yet, and `120m` real-text scaling is untested
- **Locations:** [cycle2_tier2_real.json](/home/ryan/Documents/projects/ai/results/cycle2_tier2_real.json), [harness_tier2_real.py:92-102](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:92)
- **Proof:** On `50m` (`width=512`, `hidden=1408`), polar time is `4.27%` of step time for `pe_eager` (`10.09 / 236.22 ms`) and `1.34–1.41%` for Blackwell kernels (`3.08–3.23 ms`). Eager `muon_ps3_eager` captures `+2.45%` (`1.0245×`) vs `pe_eager`, `muon_ps3_blackwell` adds `+0.72%` (`1.0072×` vs `ps3_eager`), and `muon_ss3_54` (`0.9992×`) / `muon_ss3_53` (`0.9999×`) do not beat `muon_ps3_blackwell` at `n = 512`. `120m` (`width=768`, `hidden=2048`) was never executed.
- **Action:** Report `50m` `SS3_54`/`SS3_53` accurately as **lossless LM parity + matched `+3.1%` vs `PE`**, with kernel wins over `PS3_55_blackwell` verified at `n >= 2048` (`+0.5%` to `+11.3%`) and pending end-to-end scaling at `120m` / wider widths in Cycle 3.

#### [H4] Apple M5 Max `T2-02` Metal SYRK/symprod and BF16/FP16 MLX kernels have zero live hardware verification
- **Locations:** [kernels_mlx.py:148-266,402-425](/home/ryan/Documents/projects/ai/research/polar_opt/kernels_mlx.py:148), [test_mlx.py:229-249](/home/ryan/Documents/projects/ai/tests/test_mlx.py:229)
- **Proof:** No Cycle 2 Apple Silicon result JSON exists; `test_live_metal_syrk_and_fused_symprod` is skipped on Linux; `kernels_mlx.py` implements no `SS3_*` circuits; and its custom Metal shader uses `32×32×16` tiles with FP32 `threadgroup` staging and `simdgroup_float8x8` MMA for all dtypes (`float32`/`float16`/`bfloat16`), whose throughput vs MLX's native Steel BF16/FP16 GEMM on M5 Max is completely unknown.
- **Action:** Keep `T2-02` marked `UNVERIFIED_ON_HARDWARE` until live M5 Max runs complete.

---

### `MEDIUM`

#### [M1] Transient `1.558 s` host stall on `muon_ps3_blackwell` seed 5 wall-budget run biases wall-budget ranking vs `SS3_54`/`SS3_53`
- **Location:** [cycle2_tier2_real.json](/home/ryan/Documents/projects/ai/results/cycle2_tier2_real.json) (`wall`, `seed=5`, `muon_ps3_blackwell`, `19:19:32–19:20:54`)
- **Proof:** The ONLY run out of 70 where `mean_step - med_step > 1.0 ms` (`234.18` vs `229.32 ms`, `loadavg 2.96 -> 3.61`), dropping seed 5 from `327` to `320` steps (`test_loss = 5.40898` vs `5.3800–5.3894` on seeds 1–4) and dragging 5-seed wall `tps_ratio` from `1.0315×` to `1.0270×` (`test_diff` from ~`-0.031` to `-0.0262`).
- **Action:** Disclose this single-seed stall explicitly in the Cycle 2 summary so `SS3_54` (`-0.0329`) is not falsely claimed to outrun `PS3_55_blackwell` (`-0.0262`) on `50m` wall throughput.

#### [M2] Single-LR lock in 5-seed sweep (`lr_at_grid_edge: true`) transferred from 1-seed, 128-step, uncompiled pilots
- **Locations:** [cycle2_tier2_pilot_adamw.json](/home/ryan/Documents/projects/ai/results/cycle2_tier2_pilot_adamw.json), [cycle2_tier2_pilot_muon.json](/home/ryan/Documents/projects/ai/results/cycle2_tier2_pilot_muon.json), [cycle2_tier2_pilot_ss3_53.json](/home/ryan/Documents/projects/ai/results/cycle2_tier2_pilot_ss3_53.json), [cycle2_tier2_real.json](/home/ryan/Documents/projects/ai/results/cycle2_tier2_real.json)
- **Proof:** `adamw` (`0.001`) and Muon (`0.02`) were cleanly interior on `seed=0`, 128 steps (`4.19M` tok, eager), then locked to 1-element grids for 5 seeds at 320 steps (`10.49M` tok, compiled); `muon_ps3_eager` and `muon_pe_compiled` were not piloted.
- **Action:** Disclose the two-stage pilot-bracket → 5-seed-lock protocol clearly; in Cycle 3 `120m`, pilot at least 2 seeds across the target horizon (or keep a 3-point local bracket `[0.5×, 1×, 2×]` around the pilot peak for AdamW and representative Muon).

#### [M3] Early-step spectral fragility of `SS3_54` (`0.2326`) and `SS3_53` (`0.3603`) on real FineWeb-Edu step-20 momenta
- **Location:** [cycle2_heldout_nesterov_eval.json](/home/ryan/Documents/projects/ai/results/cycle2_heldout_nesterov_eval.json)
- **Proof:** While `SS3_54` slashes mid/late error by `46%` (`0.033` vs `PS3_55` `0.061`), its early-step (step 20) error is `2.8×` (`SS3_54`: `0.2326`, max `0.6903`) to `4.4×` (`SS3_53`: `0.3603`, max `0.7441`) higher than `PS3_55` (`0.0822`, max `0.5909`) because Cycle 2 search polished on Cycle 1's synthetic `19.2M` momenta rather than real FineWeb-Edu post-Nesterov spectra. Harmless on `50m` (`320` steps), but a real risk on deeper/wider models or higher LRs.
- **Action:** Re-polish or add real FineWeb-Edu early-step spectra (`artifacts/harvested_nesterov_real.pt`) into Cycle 3 circuit search and test early-step hybrid switching.

#### [M4] Borderline `+0.00454 [+0.00079, +0.00828]` fixed-token test-loss delta between `muon_ps3_blackwell` and `muon_ps3_eager`
- **Location:** [cycle2_tier2_real.json](/home/ryan/Documents/projects/ai/results/cycle2_tier2_real.json)
- **Proof:** Paired across 5 seeds on fixed tokens, `muon_ps3_blackwell` (`5.40461`) trails `muon_ps3_eager` (`5.40007`) by `+0.00454` (`0.08%` relative), though `muon_ps3_blackwell` vs `muon_pe_eager` is `-0.00033 [-0.01064, +0.00998]`. Likely multiple-testing noise (1 of 9 pairwise contrasts crossing `p<0.05`) or minor FP16-vs-FP32 rounding, but worth verifying on longer `120m` horizons.

#### [M5] Missing unit test for `compile_model=True` / `reusable_model` / `muon_pe_compiled`
- **Location:** [test_tier2_real.py](/home/ryan/Documents/projects/ai/tests/test_tier2_real.py)
- **Proof:** Despite `cycle2_tier2_real.json` running 70 compiled trials, no automated pytest guards `reusable_model` weight-reload isolation or `muon_pe_compiled` Dynamo caching against regressions.

---

### `LOW`

#### [L1] Cross-budget `--resume` hash bypass, `harvest_nesterov` missing `total_steps`, flat `--seconds 120.0`, and `StepTimer` 19 vs 3 events
- **Locations:** [harness_tier2_real.py:271-293,756,838,926-927](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:926)
- Validate or filter non-active budget rows on `--resume`; pass `total_steps=steps` in `harvest_nesterov()`; scale default `--seconds` per preset; note `StepTimer` `19` vs `3` event asymmetry (`<0.1%`, Muon-vs-AdamW only).

#### [L2] Stale host-read comment in `test_schatten_switch.py:447`, `4096×4096` SW power-cap throttle (`0x4`), and `kernels_mlx.py` FP32 norm overflow
- **Locations:** [test_schatten_switch.py:447](/home/ryan/Documents/projects/ai/tests/test_schatten_switch.py:447), [cycle2_cuda_graph_bench.json](/home/ryan/Documents/projects/ai/results/cycle2_cuda_graph_bench.json), [kernels_mlx.py:334](/home/ryan/Documents/projects/ai/research/polar_opt/kernels_mlx.py:334)
- Update stale comment; note RTX 5080 hit `SW Power Cap` (`316W`, `2812 MHz`, `<1%` IQR, Williams-balanced) at `4096×4096`; guard `(x * x).sum()` in `kernels_mlx.py` with max-abs rescaling before squaring for parity with `schatten_switch_blackwell`.

#### [L3] End-to-end `steps==0` test and `BatchStream` replacement sampling above `26M` tokens
- **Locations:** [test_tier2_real.py:362](/home/ryan/Documents/projects/ai/tests/test_tier2_real.py:362), [harness_tier2_real.py:159-162](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:159)
- Add a tiny CPU `run_trial(..., seconds=1e-9)` test asserting `steps == 0 and diverged is True`; if Cycle 3 trains `>=26M` tokens, expand `train.bin` or switch to shuffled epoch indexing.

---

## 4. Promotion Gate Recommendations & Falsifiable Cycle 3 Roadmap

### Promotion Gates (For Immediate Cycle 2 Documentation / Infographic Update)

| Artifact / Method | Gate Decision | Exact Permitted Claim |
|---|---|---|
| **`SS3_54` (FP16, 9 sq) & `SS3_53` (FP16, 8 sq)** | **PROMOTE (`VERIFIED`)** | Sync-free CUDA-graph capturable Schatten-8 Polar Switch circuits; `0/117` Tier-1, `0/45` real Nesterov, `0/70` 5-seed `50m` LM divergences; lossless `50m` FineWeb-Edu convergence parity with `PS3_55` & `PE` (`+3.1%` tok/s, `-0.031` to `-0.033` wall test loss vs `PE`); **`1.005–1.044×` (`SS3_54`) / `1.036–1.113×` (`SS3_53`)** pure CUDA-graph speedup over `PS3_55_blackwell` at `n >= 2048` & `(4,1024,4096)` (`2.00–2.65×` vs `PE`, `1.93–2.52×` vs `PE compiled`). Disclose `n = 512` `50m` tie with `PS3_55_blackwell`. |
| **`PS3_55_blackwell` (FP16, 10 sq)** | **RETAIN (`VERIFIED` SOTA baseline)** | Fastest on small widths (`n <= 1024`, `3.08 ms` polar on `50m`); `+3.18%` tok/s over `PE eager` (`+2.45%` eager `PS3_55` + `+0.72%` Blackwell Triton) and `+2.98%` over `PE compiled`. Disclose seed-5 wall stall (`1.56 s`). |
| **`SS3_35` (BF16/FP16, 8 sq)** | **CAVEATED (`KERNEL_VERIFIED`, LM UNTESTED)** | BF16-safe (`0/117` Tier-1, `0/45` Nesterov), `1.040–1.118×` vs `PS3_55_blackwell` at `n >= 2048`; not yet tested in Tier-2 LM training. |
| **`SS3_34` (BF16/FP16, 7 sq)** | **DO NOT PROMOTE AS LM DROP-IN (`CAVEATED`)** | Fastest kernel (`1.025–1.200×` vs `PS3_55_blackwell`, `2.75×` vs `PE` at `2048×8192`), BF16-safe, **but** regresses `50m` LM loss (`+0.043` val / `+0.042` test) due to `0.5223` early-step Nesterov error (`lo = 0.008`). |
| **`gram_space_restart`** | **REJECTED (`VERIFIED NEGATIVE`)** | Diverges on `75/117` Tier-1 cases in FP16 (`σ_max` up to `73,058`). |
| **`kernels_mlx.py` Metal SYRK / BF16-FP16** | **HOLD (`UNVERIFIED_ON_HARDWARE`)** | Code-reviewed and CPU-tested only; no live M5 Max run yet. |

### Concrete, Falsifiable Cycle 3 Roadmap (Priority-Ordered)

1. **`C3-01` — Real-Text Scaling to `120m` (`width=768, hidden=2048`) & Wide-Layer Proxy (`width=1024..2048`) to Test End-to-End `SS3_*` vs `PS3_55_blackwell` Crossover:**
   - **Hypothesis:** Increasing hidden width (`512 -> 768 -> 1024+`) raises Muon polar share of step time and pushes matrix short side `n` into the regime where `SS3_53`/`SS3_54` beat `PS3_55_blackwell` end-to-end.
   - **Protocol:** Pilot `120m` AdamW & Muon LRs (`aux_lr` re-bracketed!) at `compile_model=True`; run `5` seeds on `FineWeb-Edu` (`>=20M` tokens, `--seconds` scaled to `>=300` steps) comparing `muon_pe_compiled`, `muon_ps3_eager`, `muon_ps3_blackwell`, `muon_ss3_54`, `muon_ss3_53`, and `muon_ss3_35` (BF16/FP16). Add a `~250m` (`width=1024, hidden=3072`) or micro-benchmarked step-time profile if VRAM allows (`16 GB` RTX 5080 with activation checkpointing).
   - **Falsification Criterion:** If `SS3_53`/`SS3_54` fail to achieve `> 1.005×` end-to-end tok/s over `PS3_55_blackwell` at `width >= 1024` (or lose `> +0.005` test loss on `120m`), fuse `_dual_norm_part_kernel` + `_fold_and_row0_axpy_kernel` (or accept `PS3_55_blackwell` as optimal for `n <= 1024` and restrict `SS3_*` dispatch to `n >= 2048` / batched tensors).

2. **`C3-02` — Hybrid Phase-Switched Schedule (`PS3_55`/`SS3_54` Warmup → `SS3_34` Post-Warmup) + Real-Nesterov Re-Polished `SS3_*` Circuits:**
   - **Hypothesis:** Because `SS3_34`'s failure is concentrated in early-step skewed spectra (`0.5223` median at step 20 vs `0.0464` at step 399), switching `PS3_55` (or `SS3_54`) during warmup (`steps 1..40`) → `SS3_34` afterward can recover lossless LM convergence while keeping `SS3_34`'s 7-squaring speed (`1.077–1.200×` at `n >= 2048`).
   - **Secondary track:** Re-run `schatten_switch_search.py` including `artifacts/harvested_nesterov_real.pt` early-step spectra in the polish objective to check whether a `7`- or `8`-squaring circuit with lower `lo` (`<= 0.0035`) eliminates early-step collapse without hybrid switching.
   - **Falsification Criterion:** On `50m` (`5` seeds, `320` steps) and `120m`, hybrid `SS3_54 -> SS3_34` must achieve `test_loss_diff` 95% CI upper bound `<= +0.005` vs `muon_ps3_blackwell` with `0` divergences; if mid/late tail error (`p90 = 0.29–0.31`) still regresses loss by `> +0.010`, formally retire `7`-squaring `SS3_34` for LM pretraining.

3. **`C3-03` — Reduce `SS3_*` Auxiliary Kernel Overhead (`n <= 1024`) & Refresh Post-Rewrite Tier-1 Artifact:**
   - **Engineering target:** Profile the `~15–20 µs` overhead gap of `schatten_switch_blackwell` vs `polar_switch_blackwell` at `512×2048` (`0.1393` vs `0.1253 ms`): fuse `_safe_norm_part` / `_dual_norm_part` single-block fast paths for small `size`/`nn`, eliminate unnecessary `torch.empty` allocations, and check whether `SS3_53`/`SS3_54` can cross `1.00×` at `n = 1024`. Re-record a clean post-sync-free 117-case Tier-1 JSON to supersede the stale timings in `schatten_switch_search_cycle2.json`.

4. **`C3-04` — Live Apple M5 Max (`T2-02`) Thermal/Clamshell-Safe Metal Completion:**
   - **Protocol:** Run `kernels_mlx.py --backend mlx --dtypes float32,float16,bfloat16 --warmups 5 --repeats 20` both with and without `--use-metal-syrk` on plugged-in, awake `MacBook Pro` M5 Max (logging `pmset -g therm` / thermal pressure between shapes to guard against clamshell throttling), plus add `SS3_54`/`SS3_53` to `kernels_mlx.py` and run `pytest tests/test_mlx.py` natively on Metal.
   - **Falsification Criterion:** Custom `_METAL_SYRK_SOURCE`/`_METAL_SYMPROD_SOURCE` must beat native MLX `x @ x.T` on `2048×2048` (eliminating Cycle 1's `-14%` square regression) at `0` divergences and matched SVD accuracy; if FP32 `simdgroup_float8x8` + `32×32` tiles lose to MLX's Steel BF16/FP16 GEMM, upgrade the shader to `64×64` tiles + native half/bfloat threadgroup staging or drop custom Metal SYRK in favor of native MLX.

5. **`C3-05` — Harness Hardening (Low-Cost Pre-Sweep Cleanup):**
   - Fix cross-budget `--resume` validation (`harness_tier2_real.py:926`), add `total_steps=steps` to `harvest_nesterov()`, add pytest coverage for `compile_model=True` / `muon_pe_compiled` and end-to-end `steps==0`, fix `kernels_mlx.py:334` norm overflow, and update stale `test_schatten_switch.py:447` comment.
