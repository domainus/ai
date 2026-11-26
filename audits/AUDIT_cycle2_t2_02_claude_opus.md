# T2-02 audit: M5 Max Metal 4 SYRK, fused symprod, FP16/BF16 MLX suite

- **Auditor:** Claude Opus 5.5 (`effort: high`) · **Date:** 2026-11-25 · **Original Audited Commit:** `efc56dcf1586e811bccc8e9abdd4529720ca1d6d` (kernel `3699494935f9f7ac538eb07779d2969344a553aa`)

**Scope:** branch `exp/t2-02-m5-max` @ `efc56dc` (kernel commit `3699494`), worktree `/home/ryan/.t3/worktrees/ai/t3-819533a1-1f6820c1084c9f95`. No GPU work ran. I recomputed everything from the raw `samples_ms` with stdlib Python (NumPy is not installed here), using my own NumPy-compatible linear-percentile code.

**One disclosure:** to test the merge I ran `git merge-tree --write-tree main HEAD`. That command writes unreferenced tree and blob objects to the shared object store. It changed no ref, index or working-tree file, and `git gc` removes the objects. Nothing else was written outside the scratchpad.

## Verdicts

| Gate | Verdict | Basis |
|---|---|---|
| **(a) Kernel correctness and symmetry** | **PASS_WITH_CAVEATS** | The kernel code is correct on review. The tile-index arithmetic passes 1.4 M emulated checks. All 180 triangular rows match dense to within 0.006 relative Frobenius error, and 36 square FP16/BF16 rows are bit-identical. The live boundary tests cannot be checked from here, and the JSON does not record which kernel tile ran. |
| **(b) Primitive and ≤2048² FP16/BF16 speedups** | **VERIFIED** (FP16/BF16 only) | Pooled medians, clean-seed medians, min-ratios and the interleaved primitive A/B agree within about 3%. **FP32 is a verified regression of 0.66–0.91×.** |
| **(c) Sustained 4096² timings** | **PROVISIONAL** | Every one of the 18 cases at 4096² has thermal ramps. The BF16 PS speedups as stated (3.00–3.16×) are inflated by a dense seed that ran throttled. Min-based ratios are 2.61–2.63×. The clean primitive profile supports the direction but does not show sustained performance. |
| **Overall** | **PASS_WITH_CAVEATS** | Merge as opt-in after the FP32 gate (F1) and the rebase (F6). |

## Provenance checks (all confirmed)

- **Commit and source hash:**
  - `git_sha` is `3699494935f9…` in both JSONs.
  - `source_sha256` is `6dda444b…`. This matches `kernels_mlx.py` at `3699494` and at `efc56dc`, and differs from `7221620` (`9345e3a3…`).
- **Flags:**
  - Dense: requested=false, syrk=false, symprod=false on all 180 rows.
  - Triangular: `use_metal_syrk` is true on all 180 rows. `use_metal_symprod` is true on exactly the 72 PS rows.
  - Output dtype is `float32` on all rows. The input dtype is restored correctly.
- **Divergence:** 0/360 diverged and 0/360 non-finite. The top-level `divergences` field is 0 in both files. Max σ is 1.2034 and max orthogonality error is 0.397.
- **Statistics:** median, p25, p75, IQR, min and `latency_ms` recompute with a maximum absolute difference of **0** for all 360 rows and all 12 primitive cells (original and optimized). `polar_error == relative_frobenius_error` on every row.
- **Latin squares:** all 36 cases in both runs use 20 blocks. Each 5-block cycle is a Latin square, and each method fills each position exactly 4 times. The block orders are identical between the dense and triangular runs (same order RNG), so the runs pair well.
- **Cross-check with Cycle 1:** dense FP32 seed-0 errors are bit-identical to `results/apple_silicon_cycle1.json` (15/15). Latencies at 1024×4096 and 2048² agree within 3%, which confirms the dense baseline.

## Findings, ranked by severity

### HIGH

**F1. The FP32 Metal path is slower, but `use_metal_syrk` enables it for every dtype.**
- Location: `kernels_mlx.py:345` (`self.metal = use_metal_syrk and module == "mlx"`).
- Primitive A/B in the same session: `metal_syrk` is 1.48–2.20× slower than `dense_gram` at every FP32 shape (`optimized.results[*].median_ms`).
- End to end, the triangular FP32 path runs at 0.66–0.94× of dense.
- Fix: gate the Metal path on `dtype in ("float16", "bfloat16")` so FP32 always stays dense.

**F2. The dense-vs-triangular speedups compare two separate sessions, and their thermal states differ.**
- The two runs were consecutive and the lid was closed (`run_conditions`):
  - Dense: 00:00–00:56Z, with 5 pmset events totalling 2937 s.
  - Triangular: 00:56–01:47Z, with 3 "Dark Wake Thermal Emergency" events totalling 2793 s.
- Timed work was only 93.6 s (dense) and 83.0 s (triangular) of about 3300 s and 3100 s of wall time.
- At 4096², samples above 1.5× the row minimum:

  | dtype | dense | triangular |
  |---|---|---|
  | FP32 | 39/300 | 59/300 |
  | FP16 | 47/300 | 62/300 |
  | BF16 | 39/300 | 86/300 |

  Triangular was hotter overall, which biases against it.
- The exception is dense 4096² BF16 seed 1, which ran about 2× slower for the whole case (14.9 s against 7.7–8.1 s for the other seeds). That inflates the dense BF16 medians, so the BF16 PS speedups of 3.00–3.16× are overstated. The min-based ratios are 2.61–2.63×.
- Inside a case, the Latin square balances the methods. Across runs, nothing protects the comparison.

### MEDIUM

**F3. The JSON does not record which kernel tile was selected.**
- Location: `kernels_mlx.py:248-265` and `:491-496`.
- The probe falls from the 64-tile tensor kernel to the 32-tile kernel without a warning, and it checks only that the kernel compiles, not its output.
- The live test `test_mlx.py:231` passes with either kernel.
- Timing gives indirect evidence that the 64-tile path ran: the 32-tile version measured 1.33–3.44× slower than dense, while the triangular FP16 path here is 0.59–0.76× of dense.
- Fixes:
  - Write the tile size into each row.
  - Check the probe output (an all-ones K×K input gives an output equal to K everywhere).
  - Assert the tile size from `launch.keywords["tile_size"]` in the live test.

**F4. The "≤2048² is clean" premise is wrong for some cases.**
- Dense seed 1 is contaminated in three cases:
  - 1024×4096 FP32: ramps to 1.8×.
  - 1024×4096 FP16: starts at 2.1×.
  - 2048² FP16: ramps to 14×, and 45% of `gram_ns` samples exceed 1.5× the minimum.
- Triangular seed 1, 1024×4096 BF16, has a single-block spike of 24.6×.
- The pooled medians still hold up: clean-seed speedups differ from pooled ones by ≤1.5%. The per-seed spread for 2048² FP16 is 1.19–1.41×.

**F5. The primitive profile is assembled by hand.**
- No generator script is in the repo (no file contains `metal_unfused_affine`).
- It has no `git_sha`, `source_sha256`, timestamp, host block or command.
- The "original" 32-tile profile used an unspecified seed, warmups=1 and repeats=3.
- `run_conditions` was added to both run JSONs after the runs.
- Fix: commit the primitive script and emit metadata the same way `benchmark()` does.

**F6. Both source files conflict with `main`.**
- Commit `9a23a61` on `main` carries content identical to `7221620` for `kernels_mlx.py` and `test_mlx.py` (`git diff 7221620 main` on those files is empty).
- `git merge-tree` reports CONFLICT in both files.
- Fix: `git rebase --onto main 7221620 exp/t2-02-m5-max`. This replays only `3699494` and `efc56dc`, and should apply cleanly.

**F7. The fallback and pre-M5 paths were never run on hardware.**
- The tile is chosen by compile success, not by GPU family. On M1–M4 with Metal 4 (MetalPerformancePrimitives available), the 64-tile tensor path would be selected but has never been measured.
- On the 32-tile path, the tiny and normal variants share the kernel name `polar_{syrk,symprod}_32` (`:258`). Their signatures differ (constant vs device address space). If MLX caches the compiled library by name, the two could collide. I could not check MLX source here.
- On M5 this cannot happen, because the tiny variant uses 32 and the normal variant uses 64.

**F8. Goal accounting for the "-14% at 2048²" item.**
- In FP32 the regression is still there: PE/PS3 dense is 0.876×, and the Metal path makes FP32 PS3 slower (8.64 ms vs 7.83 ms).
- In FP16/BF16 it is fixed only by fused symprod: PS3 triangular takes 3.04 ms, against 5.60 ms for PE triangular (1.84×) and 6.69 ms for PE dense (2.2×).

### LOW

- **L1. Unguarded 32-bit overflow.**
  - `r * n + c` overflows for n ≥ 65536 (`:221-222`, `:227`).
  - The grid has `128·T < 2³²`, so n must stay below 524k.
  - `col*(col+1)` overflows at col = 65536, which I confirmed.
  - The float `sqrt` tile decode is exact, with the ±1 correction holding even under ±16-ulp `sqrt` error.
  - Fix: add a guard that falls back to dense when n ≥ 65536.
- **L2. The epilogue reads only the upper triangle of `addend` (`:227`).** A non-symmetric addend is silently symmetrized, while the fallback uses the full matrix. Internal callers pass exactly mirrored buffers, so they are safe. Document this or assert it in tests.
- **L3. API edge case.** A direct `metal_symprod` call with `x.size ≥ 8` but `n² < 8` (for example 2×4) binds a constant-space `addend` to the 64-tile kernel, which was probed with a device-space addend. The polar call paths never hit this.
- **L4. Live test coverage does not match production orientation (`test_mlx.py:229`).**
  - The small live cases all have K < N, but production uses N ≤ K (wide).
  - The large exact-integer oracle (a sound test) covers only square inputs.
  - Add (17, 33), (35, 67) and a 4096×K case with K ≠ N.
- **L5. The benchmark cannot place sleep events on samples.**
  - `perf_counter` uses mach time, which stops during sleep, so a sleep never shows up in a sample. The ramp-then-reset pattern (8 per run) shows thermal throttling instead.
  - There are more resets than recorded pmset events, so the events cannot be mapped 1:1.
  - Fix: record a wall-clock time per sample and the thermal pressure, and run with the lid open under `caffeinate`.
- **L6. Tall inputs are untested on the Metal path.** They reach it as transposed views and are copied by `ensure_row_contiguous`. That cost was never benchmarked, because all shapes are wide or square.

## Independently recomputed table

Each latency is the pooled median of 60 samples (3 seeds × 20 blocks) in ms, shown as dense / triangular. `jordan_ns5` and `gram_ns` track PE within about 2%, and `PS2_55` tracks PS3.

| shape | dtype | PE ms (d / t) | PE speedup | PS3 ms (d / t) | PS3 speedup | PS3 speedup, clean seeds | PS3 speedup, min-ratio | PS3 error (d / t) | samples >1.5×min (d / t) | max row rel. IQR (d / t) |
|---|---|---|---|---|---|---|---|---|---|---|
| 512×2048 | FP32 | 0.842 / 1.274 | 0.66× | 0.679 / 0.885 | 0.77× | 0.77 | 0.78 | .0617 / .0634 | 1 / 0 | .07 / .07 |
| 512×2048 | FP16 | 0.845 / 0.813 | 1.04× | 0.708 / 0.488 | 1.45× | 1.45 | 1.47 | .0636 / .0636 | 0 / 0 | .12 / .06 |
| 512×2048 | BF16 | 0.856 / 0.815 | 1.05× | 0.710 / 0.494 | 1.44× | 1.44 | 1.46 | .0636 / .0632 | 0 / 1 | .09 / .06 |
| 1024×4096 | FP32 | 3.374 / 4.410 | 0.77× | 2.288 / 2.604 | 0.88× | 0.86 | 0.86 | .0626 / .0645 | 18 / 0 | .29 / .07 |
| 1024×4096 | FP16 | 3.465 / 2.913 | 1.19× | 2.241 / 1.235 | 1.81× | 1.79 | 1.83 | .0649 / .0647 | 32 / 0 | .67 / .08 |
| 1024×4096 | BF16 | 3.441 / 2.921 | 1.18× | 2.179 / 1.236 | 1.76× | 1.79 | 1.81 | .0646 / .0626 | 0 / 2 | .04 / .10 |
| 2048² | FP32 | 6.853 / 9.684 | 0.71× | 7.826 / 8.643 | 0.91× | 0.91 | 0.90 | .1143 / .1144 | 0 / 0 | .03 / .02 |
| 2048² | FP16 | 6.729 / 5.619 | 1.20× | 7.877 / 3.059 | 2.58× | 2.57 | 2.59 | .1146 / .1144 | 38 / 0 | .97 / .15 |
| 2048² | BF16 | 6.666 / 5.577 | 1.20× | 7.806 / 3.042 | 2.57× | 2.57 | 2.59 | .1157 / .1139 | 0 / 0 | .02 / .03 |
| 4096² | FP32 | 53.93 / 69.48 | 0.78× | 57.53 / 61.52 | 0.94× | — | 0.92 | .1265 / .1262 | 39 / 59 | .34 / .67 |
| 4096² | FP16 | 46.84 / 37.81 | 1.24× | 52.11 / 20.48 | 2.54× | — | 2.59 | .1263 / .1262 | 47 / 62 | .22 / .45 |
| 4096² | BF16 | 54.13 / 38.53 | 1.40×* | 62.59 / 20.87 | 3.00×* | — | 2.61 | .1278 / .1270 | 39 / 86 | .34 / 1.90 |

\* Inflated by dense seed 1, which ran throttled for the whole case. Use the min-ratio column instead.

**Errors:** across all 180 row pairs, triangular minus dense error ranges from −0.0060 (1024×4096 BF16 PS2_55, where fused rounding helps) to +0.0023. The rows that are bit-identical (FP16/BF16 jordan, PE and gram_ns at 2048² and 4096²) are strong evidence that the Gram is correct.

**Outliers:** the largest sample is 3934 ms in the dense run (4096² FP32 seed 0 `jordan_ns5`, block 2, 74× its minimum) and 951 ms in the triangular run. Only one sample in either run exceeds 1 s.

**Primitive profile (64-tile, interleaved, clean: max/min ≤ 1.58, at most one sample above 1.5×min per cell):**
- FP16/BF16, `syrk / dense_gram`:
  - 512×2048: 0.96 / 1.01
  - 1024×4096: 0.75 / 0.76
  - 2048²: 0.74 / 0.75
  - 4096²: 0.59 / 0.61
- `symprod / dense_affine`: 0.32–0.88.
- FP32 `syrk / dense_gram`: 1.48–2.20, slower everywhere.

## Recommendations

1. Keep `use_metal_syrk` **opt-in**. Gate it to FP16/BF16 and force FP32 to dense (F1). That is a one-line change at `kernels_mlx.py:345`, and the test for the FP32 gate goes in `test_interleaved_benchmark`.
2. Before merging:
   - Rebase with `--onto main 7221620` (F6).
   - Record the selected tile size per row and check the probe output (F3).
   - Add the N≤K live test cases (L4) and the n < 65536 guard (L1).
3. Promote only the FP16/BF16 numbers at ≤2048², using the clean-seed or min-ratio columns. Label 4096² as provisional and quote BF16 PS at about 2.6×, not 3.0–3.2×.
4. Re-run 4096² with the lid open, under `caffeinate -dimsu`, with per-sample timestamps and thermal pressure. Interleave dense and triangular in the same Latin square, adding `use_metal` as a method axis, so the comparison happens within one session.
5. Commit the primitive-profile generator (F5). Update `state/BACKLOG.md` to record that the FP32 2048² regression remains and that it is fixed only in FP16/BF16 (F8).

## Post-Audit Remediation Log

Applied on `exp/t2-02-m5-max` after `origin/main` (`877c2a8`) rebase (with Metal shader source strings byte-identical to M5 Max-verified `3699494`):
- **F6 (Rebase):** Cleanly rebased `exp/t2-02-m5-max` onto `origin/main` (`877c2a8`) with zero conflicts.
- **F1 (FP32 gate):** Gated `_Work.metal` and `_measure()` Metal telemetry on `dtype in _METAL_FAST_DTYPES` (`("float16", "bfloat16")`) so FP32 always uses dense MLX Steel GEMM under `use_metal_syrk=True`. Added `test_fp32_stays_dense_under_use_metal_syrk`.
- **F3 (Probe check & tile telemetry):** Added exact numerical probe verification inside `_metal_kernel` and emitted `metal_syrk_tile` / `metal_symprod_tile` in `_measure()` result rows. Added `test_metal_probe_mismatch_falls_back_and_tiny_names_distinct` and live tile assertions in `test_live_metal_syrk_and_fused_symprod`.
- **F7 (Tiny kernel name isolation):** Distinguished tiny kernel specialization names (`polar_{syrk,symprod}_tiny_{tile_size}`).
- **L1, L2, L3, L4 (Guards & live coverage):** Added `_METAL_MAX_N = 65535` 32-bit index overflow guards in `metal_syrk`/`metal_symprod`, documented upper-triangle `addend` symmetry requirement in `metal_symprod`, guarded `addend.size < 8` constant-space binding (`test_metal_overflow_and_tiny_addend_guards`), and expanded `test_live_metal_syrk_and_fused_symprod` to cover wide production orientations `(17, 33)`, `(35, 67)`, and `(2048, 4096)`.
- **Open items tracked for Cycle 2.5 / 3:** Commit primitive-profile generator (`F5`), record honest FP32 vs FP16/BF16 `2048²` goal accounting in `state/BACKLOG.md` (`F8`), run `tests/test_mlx.py` live on `MacBook Pro` after merge, and perform an open-lid `caffeinate -dimsu` single-session interleaved `4096²` rerun (`F2`, `L5`).
