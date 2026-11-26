# Cycle 3 C3-02 independent audit

**Executive verdict — PASS_WITH_CAVEATS.** The raw evidence supports recording a negative result for both warmup-to-`SS3_34` schedules and for the `SS3_54E0` re-polish. Keep `ss3_35_bf16` **CAVEATED**. Hold the tooling merge until the output-path collision and the failed-candidate handling below are resolved. Neither defect affected the three committed result files.

Reviewed experiment head `1098549792ded67dc5213d331c64e5f3525d90aa`. Commit `46b987f` introduced the three LM candidates, hooks and tests. Commit `8300823` added the opt-in search, launch accounting, extra evaluation methods and tests. Commit `674b006` handled an incompatible fit cache. Commit `1098549` added only the three results. All independent calculations and tests in this audit ran on CPU.

## Findings and merge actions

| Severity | Finding | Action |
|---|---|---|
| CRITICAL | No critical finding in the recorded experiments. All reported paired intervals and all 28 held-out phase summaries reproduce. | Retain the negative results. |
| HIGH | With `--include-early-srank` alone, the search reads and then overwrites `results/schatten_switch_search_cycle2.json`. The new document has `validated` and lacks the `selected` key needed for the next run. A CPU parser probe intercepted the call before search execution and confirmed that the two resolved paths are identical. Both committed Part B commands used explicit, distinct output paths. | Before merge, give early mode a separate output default or require an explicit output. Reject an output that resolves to its seed artifact. Apply the same protection to the canonical Cycle 2 eval when adding `--candidates` without an explicit output. See `schatten_switch_search.py:1201, 1313, 1319, 1333` and `eval_heldout_nesterov.py:76`. |
| MEDIUM | The search keeps rejected GPU candidates under `validated` with `fp16_passed=false`. Timing filters them, but `_candidate_methods` admits every record to the held-out eval. Flipping only this flag on the real candidate in memory still registered its FP16 method. The actual E0 record passed all four FP16 checks. | Before merge, filter failed FP16 records or require an explicit diagnostic mode that labels them as rejected. Test a rejected record. See `schatten_switch_search.py:1273` and `eval_heldout_nesterov.py:44`. |
| MEDIUM | The committed eval contains a `promotion_gate` that the recorded command cannot regenerate. Its `parent_mid_late_win_retained` is the **mid-only** value. It labels 5+3 as lacking a safe candidate although one attempt had `safe=true` and failed Cycle 1 parity. Its statement that 3+k circuits “cannot reach it at all” exceeds this finite search. | Before merging the findings into the research record, identify the gate as a separate analysis or provide its generator. Name the phase statistic, label the three splits as having no **qualifying** candidate, and restrict the conclusion to the starts and constraints tested. No new GPU run is needed. |
| MEDIUM | The PS3-first hybrid and PS3 control call the same polar function for the first 40 updates. All five seeds have matching logged step-1 losses, but each pair differs by step 10, before switching. At step 40, seed 4 differs by 0.03314 in logged training loss. The fingerprints verify the paired inputs. The saved rows do not identify the source of the trajectory differences. | Report the five-seed comparison of complete schedules. Attributing the entire loss difference to update 41 would require a shared step-40 checkpoint or a measured repeat-control trajectory. |
| LOW | The launch formula describes all five recorded Schatten circuits. A supported custom circuit with a nonzero **first-segment residual** takes an extra eager cast/multiply/add path, even on contiguous FP32 input with natural scaling. The new helper returns the same +4 for that case. | Scope the helper to its supported zero-residual path or account for the extra operations. The existing `TEST_RES` fixture supplies a counterexample. See `schatten_switch_blackwell.py:723` and `tests/test_schatten_switch.py:514`. |
| LOW | A module-wide CUDA skip also skips the new pure-CPU synthetic-spectrum test. The full-search serializer changes the shape of `kernel_launches` while retaining schema version 1. | Move the CPU test outside the CUDA guard. Document or version the schema change for consumers of the full-search output. |

## Part A — Five paired seeds on FineWeb-Edu

The `50m` preset contains 64,303,616 parameters. Each of the 20 rows completed 320 updates, 32,768 tokens per update and 10,485,760 training tokens. Each used matrix LR 0.02, auxiliary LR 0.001, FP32 update layout and the same validation and test windows. BF16 in the candidate name specifies the working precision of its polar circuit. The reference was `muon_ps3_blackwell` (FP16 PS3_55). Only one matrix LR was tested.

### Paired statistics from the raw runs

Rows were joined by seed 1–5. Positive loss differences favor the reference. Each speed ratio is candidate tokens/s divided by reference tokens/s for that seed. I recomputed tokens/s from accepted tokens and train seconds and used the arithmetic mean of the five ratios. The two-sided intervals use sample SD, four degrees of freedom and critical value 2.776445. The t-tests use null values of zero for loss and one for speed. Means, SDs and intervals match all nine stored comparisons within `1e-12`.

| Candidate | Paired metric | Mean | SD | 95% Student-t CI | t(4) | Two-sided p |
|---|---|---:|---:|---|---:|---:|
| PS3_55 → SS3_34 | Test loss Δ | +0.010385 | 0.002994 | [+0.006668, +0.014102] | 7.7568 | 0.001489 |
| | Validation loss Δ | +0.011261 | 0.002965 | [+0.007580, +0.014943] | 8.4925 | 0.001054 |
| | Tokens/s ratio | 1.000307 | 0.001689 | [0.998210, 1.002404] | 0.4069 | 0.704939 |
| SS3_54 → SS3_34 | Test loss Δ | +0.011190 | 0.007614 | [+0.001736, +0.020644] | 3.2862 | 0.030323 |
| | Validation loss Δ | +0.011522 | 0.007528 | [+0.002175, +0.020870] | 3.4224 | 0.026721 |
| | Tokens/s ratio | 1.000500 | 0.002464 | [0.997440, 1.003559] | 0.4536 | 0.673662 |
| SS3_35 BF16 | Test loss Δ | +0.002582 | 0.005542 | [−0.004299, +0.009463] | 1.0418 | 0.356311 |
| | Validation loss Δ | +0.002810 | 0.004914 | [−0.003292, +0.008912] | 1.2785 | 0.270212 |
| | Tokens/s ratio | 0.995566 | 0.002362 | [0.992633, 0.998499] | −4.1972 | 0.013727 |

| Optimizer | Mean validation loss | Mean test loss | Mean tokens/s | Polar ms, median of run medians | Polar ms, mean of run medians | Diverged |
|---|---:|---:|---:|---:|---:|---:|
| PS3_55 | 5.355211 | 5.371874 | 142,870.3 | 3.076320 | 3.076314 | 0/5 |
| PS3_55 → SS3_34 | 5.366473 | 5.382259 | 142,914.1 | 2.910048 | 2.909984 | 0/5 |
| SS3_54 → SS3_34 | 5.366734 | 5.383064 | 142,941.8 | 2.910016 | 2.908294 | 0/5 |
| SS3_35 BF16 | 5.358021 | 5.374457 | 142,236.7 | 3.938752 | 3.938381 | 0/5 |

Both hybrids increased test loss for all five seeds. Their unadjusted intervals exclude zero. As a sensitivity check, Holm adjustment across the three candidate **test-loss** contrasts gives p = 0.00447, 0.06065 and 0.35631 in table order. The second hybrid has weaker statistical support under that choice of family. Neither shows a measurable end-to-end speed advantage. BF16 was slower in every pair. Its loss interval includes both improvement and degradation. A claim of loss equivalence would require an explicit equivalence margin.

### Optimizer state, timing and LR

`Muon.step` calls the orthogonalizer once per same-shape group, then steps auxiliary AdamW and calls `on_step_end()` once outside the loop. Microbatch gradient accumulation precedes this step. During each of the three disposable compile steps, `set_warmup(True)` exercises fast then warm for each group, returns the warm result, and suppresses the counter increment. The harness clears gradients, drops the warm optimizer and stream, restores the initial weights, and constructs fresh Muon, AdamW, hybrid and batch-stream state. The timed hybrid therefore uses the warm function on updates 1–40 and the fast function on updates 41–320. The CPU integration test exercised multiple groups, two trials and separate warm and timed instances.

The token-mode LR warmup used 50 updates in all four arms. Cycle 2's comparable 320-step runs used 20. The switch now occurs during the LR ramp (update 41 uses 41/50 of peak LR). This is a common schedule within C3-02, so the paired operational verdict stands. The relative effect at 20 warmup updates and differences between absolute Cycle 2 and Cycle 3 losses remain untested.

The saved polar median excludes update 1 and sums the polar event times across shape groups within each step. Each hybrid median pools **39 warm and 280 fast** updates. The 160th order statistic is dominated by fast-phase observations but may come from either phase. The per-step event series is absent. The table aggregates the saved run medians, and a fast-only median cannot be recovered from these files. Throughput includes all 320 timed training steps and excludes compilation, evaluation and batch preparation.

### Pairing and provenance

I rebuilt the full CPU initial state for each seed, hashed the sorted parameter names and bytes, and independently generated and hashed the first three batches of training-window offsets. Both 16-hex fingerprints matched all four arms per seed. The stream fingerprint covers a prefix. The source fixes subsequent draws by the seed and batch configuration.

| Seed | Initial weights | Stream offsets |
|---:|---|---|
| 1 | `85659cbc0ffc79c1` | `b9f91c01f18ce738` |
| 2 | `eff774e6f234e5e7` | `f32172b52cfa7de3` |
| 3 | `48a8ed2b9025f468` | `e88efd98cfc4e5e9` |
| 4 | `5991a96f467739ab` | `d4afd0616d1db977` |
| 5 | `ed4aff32b452b660` | `01366092adfa364f` |

SHA-256 of the actual shard bytes matched both the external manifest and the run metadata. Byte sizes also matched the manifest's uint16 token counts.

| Shard | Tokens | SHA-256 |
|---|---:|---|
| train | 26,004,518 | `015ab83ceb742555c2398963a0bd1f3c5502dc52cf70a22bf173f1571326ca3a` |
| val | 2,002,330 | `371b64a537f7b1e306ea174a868b05932a254a98b7fbdbf4146648d93180b0b1` |
| test | 2,000,081 | `4eb46f1424856849d2973923dbf3c886d37eb743cf22053fc7e1eea1f41ab92a` |

The LM JSON records `46b987f9375fca171ad330a24cf05ba4f4bb2d3b`. Search and eval both record `674b0067a4344171851f815cb27f72e350a20544`. All three report `git_dirty=false`. The corresponding commits exist in the reviewed history. The status probes check tracked files only. I parsed each file with duplicate-key and nonfinite-literal rejection, then re-serialized with `allow_nan=False`. All passed. Each source writer also uses `allow_nan=False`.

## Part B — Search inputs and held-out Nesterov evaluation

The search file contains no reference to `harvested_nesterov_real`. The actual command supplied `harvested_trajectories.pt`, the Cycle 1 artifact. The loader reads `current` and `parameter` from that format. The evaluator alone read the real bundle's `N_t` matrices. The search used the four registered Cycle 2 circuits as starts. The incompatible fit cache was detected and ignored. Five polish configurations per start gave 20 attempts. Filtering used scalar safety, Cycle 1 parity and synthetic early median below 90% of the registered start. Only one circuit qualified for native checks.

I regenerated all 108 synthetic spectra on CPU and matched every recorded name, length and stable rank. The grid uses two sizes (512, 1024), six target stable ranks (1.03, 1.08, 1.15, 1.25, 1.4, 1.6) and nine tail shapes. Each has a unit spike. Three tails use power laws with exponents 0.5, 1 and 1.25. Six use 3 or 10 geometric secondary values and Gaussian singular-value floors at 0.001, 0.002 or 0.004. A seeded Gaussian bulk is shared by families of the same size. Tail energy is scaled to stable rank minus one. All tested spectra had a positive tail and maximum 1.

**Holdout boundary.** I found no real case coefficients, tensors, per-case losses or case lookup in the fit or candidate ranking. The published Cycle 2 audit had already disclosed the early stable-rank median (1.24), minimum (1.03), and under-lift on these same cases. The synthetic grid deliberately addresses that finding. The 45 real matrices are held out from coefficient fitting, and this is a repeat evaluation on a previously inspected benchmark. The checked-in evidence supports that distinction. An independent generalization claim would need fresh trajectories. See `audits/AUDIT_cycle2_claude_fable.md:152`.

### Held-out results recomputed from the case rows

I matched all seven methods' case identifiers and shapes to the external 45-pair bundle. It contains five matrix types in each of layers 0, 6 and 11, at steps 20, 200 and 399 of one seed-999 trajectory. Each phase has 15 matrices. The pooled row has 45. The six existing methods' case records match their Cycle 2 records exactly. For each group I sorted relative Frobenius errors, used the median and linearly interpolated p90 at index `0.9(n−1)`, and took independent maxima. All rows had zero nonfinite outputs and zero divergences at the evaluator's σ-max limit of 2.0.

| Method | Phase | Median error | p90 error | Max error | Max σ-max |
|---|---|---:|---:|---:|---:|
| PS3_55 FP16 | early | 0.082240 | 0.499808 | 0.590880 | 1.081208 |
| | mid | 0.061542 | 0.171973 | 0.192382 | 1.081074 |
| | late | 0.060358 | 0.158487 | 0.175121 | 1.081158 |
| | all | 0.061853 | 0.217764 | 0.590880 | 1.081208 |
| SS3_54 FP16 | early | 0.232618 | 0.618117 | 0.690261 | 1.080367 |
| | mid | 0.033275 | 0.158147 | 0.208069 | 1.080593 |
| | late | 0.032866 | 0.138786 | 0.165913 | 1.080317 |
| | all | 0.038852 | 0.291841 | 0.690261 | 1.080593 |
| SS3_53 FP16 | early | 0.360252 | 0.683144 | 0.744102 | 1.087636 |
| | mid | 0.054347 | 0.193786 | 0.255245 | 1.087640 |
| | late | 0.053578 | 0.186249 | 0.203899 | 1.087606 |
| | all | 0.054777 | 0.386873 | 0.744102 | 1.087640 |
| SS3_35 FP16 | early | 0.396789 | 0.711197 | 0.767601 | 1.084979 |
| | mid | 0.056619 | 0.211422 | 0.279220 | 1.085469 |
| | late | 0.055763 | 0.212883 | 0.222564 | 1.085345 |
| | all | 0.063996 | 0.442593 | 0.767601 | 1.085469 |
| SS3_35 BF16 | early | 0.397893 | 0.711980 | 0.768335 | 1.115073 |
| | mid | 0.058882 | 0.213496 | 0.280938 | 1.117252 |
| | late | 0.057003 | 0.215829 | 0.224319 | 1.116662 |
| | all | 0.068862 | 0.443716 | 0.768335 | 1.117252 |
| SS3_34 FP16 | early | 0.522256 | 0.804380 | 0.845893 | 1.067447 |
| | mid | 0.068040 | 0.292022 | 0.386748 | 1.067497 |
| | late | 0.046400 | 0.313969 | 0.319735 | 1.067500 |
| | all | 0.214590 | 0.633693 | 0.845893 | 1.067500 |
| SS3_54E0 FP16 | early | 0.202385 | 0.600088 | 0.675456 | 1.118519 |
| | mid | 0.048218 | 0.154760 | 0.200782 | 1.119059 |
| | late | 0.047434 | 0.136829 | 0.161773 | 1.118515 |
| | all | 0.054313 | 0.271778 | 0.675456 | 1.119059 |

As a separate numerical check, I used the real `0.down.weight` input in each phase, CPU FP64 thin-SVD targets and the CPU FP16-emulation implementation of E0. Early/mid/late errors were 0.055564 / 0.045922 / 0.047482. The corresponding saved GPU errors were 0.055534 / 0.045923 / 0.047500. Across these three cases the largest absolute error difference was 0.000030, and the largest σ-max difference was 0.000161. CPU emulation uses different floating-point operations. The external real-bundle SHA-256 was `f9f6358b4d57a931f74840b986d6b0a1f471933fcdad072cb4755a087df0a6eb`.

### Polynomial slopes and kernel launches

I evaluated the derivative at zero directly from the coefficient tables. For a segment with scalar map `x(q(x²) + residual)`, initialize register 0 to zero and evaluate each row as `a × src² + b × add + c`. The segment derivative is the final register plus the residual. Multiply the two segment derivatives. The E0 table came from the search JSON, the registered SS tables from `schatten_switch_blackwell.py`, and PS3_55 from `baselines.py`. The slopes apply to normalized scalar input.

| Circuit | Segment 1 slope | Segment 2 slope | Slope at zero | Matrix products | Analytic launches | Archived profiler launches | Host reads reported |
|---|---:|---:|---:|---:|---:|---:|---:|
| PS3_55 | 46.4799 | 24.3263 | 1130.6865 | 14 | 16 | 16 | 0 |
| SS3_54 | 45.3362 | 12.8872 | 584.2571 | 13 | 17 | 17 | 0 |
| SS3_54E0 | 45.5741 | 14.2620 | 649.9789 | 13 | 17 | 17 | 0 |
| SS3_53 | 43.8360 | 8.8695 | 388.8021 | 12 | 16 | 16 | 0 |
| SS3_35 | 12.6767 | 25.3131 | 320.8866 | 12 | 16 | 16 | 0 |
| SS3_34 | 12.4987 | 12.4845 | 156.0406 | 11 | 15 | 15 | 0 |

Each Schatten segment needs a Gram product, its listed squarings, and a final matrix application. Segment 1 uses its initial square both for the norm estimate and for row 0. Four further launches perform safe-norm partials, normalization with prescale, dual-norm partials, and the coefficient fold including the row-0 update. This gives `2 × segments + squarings + 4` for these warm, natural-scale, contiguous FP32 calls. The corresponding PS3 path adds a vector norm and normalize kernel to its 14 products. The archived 512 × 2048 traces agree in all six cases. Coefficients and norms stay on device. The profiler helper's host-read field uses PyTorch sync-debug mode and reports a binary indication. It is not a complete count of driver synchronization. Cold autotune, explicit scale, and extra casts lie outside this comparison.

## Disposition

- **Retire standalone FP16 SS3_34 and the two tested 40-update hybrids from LM-pretraining promotion.** The standalone evidence comes from the separate 128-step, seed-0 pilot (+0.042061 test loss versus PE eager, +0.035067 versus PS3_55 at LR 0.02). The two hybrids have five-seed evidence at 50m. Record 120m as untested or explicitly waived after the 50m result. Retirement is a program decision for these candidates. This experiment did not measure other switch points or model sizes.
- **Record SS3_35 BF16 as CAVEATED.** Mean test loss was +0.002582 with an interval crossing zero. Throughput fell 0.4434% on average. These measurements support neither a promotion for speed nor a claim of proven loss parity.
- **Log SS3_54E0 as a negative search result.** Its synthetic early median improved from 0.27962 to 0.24464. The held-out early median fell 13.0% versus SS3_54 but remained 2.461× PS3_55. Mid/late medians still beat PS3_55, retaining 47.14% and 47.01% of the parent's respective advantages. The design-band error rose from 0.07983 to 0.11777 and observed max σ-max rose from 1.08059 to 1.11906. The archived FP16 R1, R2, long-K R3 and graph checks passed. E0 has no BF16 verification. Leave it out of the promoted registry.

**CPU verification.** Ran `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/ryan/Documents/projects/ai/.venv/bin/pytest tests/test_tier2_real.py tests/test_schatten_switch.py -q` with bytecode and the pytest cache disabled. **20 passed, 162 skipped in 12.19s**. One skip was the worktree-local real-shard check, and 161 were CUDA-gated. The audit separately hashed the external shards, regenerated the synthetic spectra and checked the real bundle on CPU. Native GPU kernel and timing evidence comes from the archived experiment.
