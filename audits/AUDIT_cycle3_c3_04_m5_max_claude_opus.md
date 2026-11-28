# C3-04 / T2-02 closure audit: MLX Schatten Switch port and M5 Max paired Metal benchmarks

## Verdict: **PASS_WITH_CAVEATS**

The numbers are real. Every statistic in the three result files reproduces exactly from the raw samples. Every table cell and headline claim in `reports/m5_max_cycle3.md` matches the raw data. The MLX circuits match the Blackwell source bit for bit. The BF16 fix and the BF16 safety gate both do what the report says. I found no blocker in the science or the code that ran.

The branch is **not merge-ready as pushed**. It edits `state/BACKLOG.md` and `state/ledger.jsonl`, and both files conflict with `origin/main` (M1). Also, the Metal arm's dense fallback breaks the SS numerical contract on hosts where the shader does not compile (M2). It did not affect these runs. All results come from a closed-lid session, and the report does not mention a clear thermal ramp at 4096² (M3).

**Recommendation:** promote the Tier-1 results as stated, with the closed-lid caveat. Merge the code, tests, results and report after you drop the two `state/*` edits from the branch (or rebase and resolve them on `main`). Fix M2 before anyone enables `use_metal_syrk` off M5-class hardware.

## Audit metadata

| Item | Value |
|---|---|
| Auditor | Claude Opus 5.5 (`effort: high`), independent adversarial review |
| Date | 2026-11-27 |
| Reviewed branch | `origin/exp/c3-m5-max-ab-ss3` |
| Implementation and tests commit | `7eacbb6516ef71d8ff19f63c723b861d3dd799fb` |
| Results and report commit (remote HEAD) | `710240f55b85dd3b45764cdfa03b3c7a6dcec0dc` |
| Branch base | `6236a5216b0f0988b069cd7d785ee9c1b8b91542` (`origin/exp/t2-02-m5-max`) |
| `origin/main` at audit | `1c7b9385fb23a81c043677ce823259c6bd2eb32a` |
| Lineage | `6236a52 → 7eacbb6 → 710240f`, linear, one parent each (verified) |
| Audit host | `bazzite` (Linux x86-64), **CPU only**, `CUDA_VISIBLE_DEVICES=""`, single-threaded BLAS. The RTX 5080 was not touched. MLX is not installed on this host. |
| Scope boundary | The M5 Max was asleep after it pushed `710240f`. I audited the Metal shaders by static analysis. I checked the numerics with the NumPy storage-rounding emulation in `kernels_mlx.py`. I recomputed all timing and telemetry from the raw JSON. **I did not replay anything on live M5 Max hardware.** The 34 MLX-only test parametrizations and the 11 `_has_mlx()` tests are therefore unverified by this audit. |
| Run conditions (from telemetry) | All 82 snapshots, 2026-11-27 11:05:57–11:10:41 UTC: `AppleClamshellState = Yes` (lid closed), AC power, battery 80% not charging, `pmset -g therm` with no thermal or performance warning. |
| Shared object-store disclosure | `git merge-tree --write-tree origin/main 710240f` wrote unreferenced tree and blob objects to the shared object store (as in Cycle 2). It changed no ref, index or working-tree file. `git gc` removes them. |

## Findings, ranked by severity

### HIGH / BLOCKER

None.

### MEDIUM

**M1. The branch edits `state/*`, conflicts with `main`, and marks T2-02 done before the audit.**

- `git diff --stat 6236a52..710240f` touches `state/BACKLOG.md` (+2 −1) and `state/ledger.jsonl` (+1). `README.md` is untouched.
- `git merge-tree --write-tree origin/main 710240f` exits 1:
  ```
  CONFLICT (content): Merge conflict in state/BACKLOG.md
  CONFLICT (content): Merge conflict in state/ledger.jsonl
  ```
  Every other file merges cleanly: code, tests, results, report and the Cycle 2 audit file. `origin/main` moved 93 commits past the base and rewrote both state files.
- `state/BACKLOG.md:14` changes T2-02 to `[x]`. The ledger row added in the same commit says `"status": "PENDING_AUDIT"`, and the same backlog line says "The FP32 goal remains open". So the checkbox claims closure before audit, and the original FP32 −14% goal is not met (dense FP32 PS3_55 is still 0.88× Polar Express at 2048²).
- Fix: before merge, restore both files to `6236a52`. Let the integrator write the state update on `main` after this audit. Keep the T2-02 checkbox open, or split it into "FP16/BF16 closed (Metal)" and "FP32 open".

**M2. The dense fallback of the fused kernel breaks the SS circuits' one-rounding contract.**

- Location: `research/polar_opt/kernels_mlx.py:398-400`, which is reached when `_metal_kernel` returns `None` (both shaders fail to compile or fail the probe, `:356-360`), or when `n > _METAL_MAX_N`.
- `_Work.metal` (`:447`) depends only on the flag and the dtype, not on kernel availability. So `schatten_switch(..., use_metal_syrk=True)` still sends `square` (`:664`), every segment-1 row (`:627`) and segment 2 (`:582`) through `_metal_symprod_vec`.
- The fallback rounds the Gram to the work dtype before it applies the coefficients. This is the round-square-first arithmetic that the dense path avoids on purpose: `_dense_scaled_gram` at `:604-610`, the BF16 wide branch at `:628-632`, and the FP16 power-of-two rescale at `:633-639`.
- CPU reproduction: I replaced `_metal_symprod_vec`, `metal_symprod` and `metal_syrk` with NumPy copies of the fallback code (`:398-400`, `:368`) that use `_cpu_round`, then forced `_Work.metal = True`.

  | Input | Dense arm (current) | Metal arm through the fallback |
  |---|---|---|
  | 1024² coordinate spike `g[0,0]=1e25`, FP16, SS3_54 / SS3_53 / SS3_35 | `y[0,0]` = 0.958 / 0.914 / 0.936 | **NaN** (all three) |
  | SS3_35 BF16 coherent rank-one sweep (64×128, 1201 amplitudes) | max ‖y‖_F = 1.127 | **4.353** (cap 1.25) |

- On M5 Max this path never ran: all 96 Metal rows record tile 64. The risk is a pre-Metal-4 host where the 32-tile shader also fails, or a future MLX regression. The only signal is a `RuntimeWarning`.
- Fix: make the fallback compute in FP32, `xf = x.astype(mx.float32); (a * (xf @ xf.T) + b * addend.astype(mx.float32) + c * I).astype(x.dtype)`. That matches the Metal epilogue's single rounding. Alternatively, set `work.metal = False` when no kernel is available.

**M3. The report leaves out a clear 4096² thermal ramp, and the run used a closed lid.**

- The report says (line 3) that all 82 snapshots show "no thermal or performance warning". That is literally true. But `pmset -g therm` does not show GPU clock reduction, and the raw samples show a ramp:

  | Shape | Dtype | Latency, last 5 / first 5 blocks (median over rows; max) | Spearman(block index, latency), median | Paired dense/metal ratio, last 5 / first 5 (median; range) |
  |---|---|---|---|---|
  | 4096² | FP16 | 1.105 (1.179) | +0.80 | 0.990 (0.931–1.042) |
  | 4096² | BF16 | 1.085 (1.128) | +0.85 | 0.998 (0.976–1.040) |
  | ≤ 2048² | both | 0.989–1.003 | −0.25 to +0.19 | 0.988–1.007 |

  Example: the 4096² FP16 PS3_55 Metal arm for seed 2 rises from 22.3 to 26.8 ms over 20 blocks. Its dense partner rises from 53.6 to 61.8 ms.
- The within-block pairing cancels the ramp. No paired ratio or PS3_55/SS ratio moves by more than 1.4% at the median. No A/B sample exceeds 1.5× its row minimum, so the Cycle 2 tail contamination is gone. **No reported number changes.** Absolute 4096² milliseconds, however, are not sustained-state numbers.
- The `caffeinate -dimsu` claim (line 3) is not in the recorded `command` fields, so the artifacts cannot verify it.
- Fix: add one sentence to the report about the ramp and that it does not affect the ratios. Keep the open-lid replication (Cycle 2 F2/L5) as an open item. If you can, record `powermetrics --samplers gpu_power` (it needs sudo) instead of relying on `pmset`.

### LOW

**L1. `_upper` slack differs from the Blackwell formula** (`kernels_mlx.py:597`).
The MLX code uses `_SLACK / max(0.5, 1 − 1e-8·k)` for every `k`. Blackwell uses `_SLACK` for `k ≤ 65536`, and `_SLACK / max(0.2, 1 − 1.2e-8·k)` above that with a chunked FP64-summed SYRK (`schatten_switch_blackwell.py:774`). For every benchmarked `k ≤ 4096`, MLX is slightly more conservative (+4e-5). For `k > 65536`, MLX is less conservative (for example 1.0101 against 1.0121 at `k = 10⁶`), and the Metal SYRK accumulates the full `k` in FP32 with no chunking. Every coefficient, `u_safe`, `lo`, `bf16_verified`, stage count and fold formula matches exactly (see the table below). Only this slack term differs. Either copy the Blackwell constants or document the deviation.

**L2. `_safe_norm` still overflows when ‖G‖_F > FLT_MAX** (`:489`).
The input `[[3.4e38, 3.4e38]]` is finite FP32, but `sqrt(sum)/inv_scale` gives `inf`. `_normalized` and `schatten_switch` then return zeros, even though `G/‖G‖` is representable. Dividing the scaled matrix by the scaled norm (`scaled / sqrt(sum(scaled²))`) removes this edge. Every other probe passed on both NumPy and PyTorch CPU: 1e25, 1e38, 1e-30, subnormal 1e-41, minimum subnormal, ±0 (sign kept), tall, wide and square. The FP32 norm is formed before the work cast, and `test_norm_in_fp32_before_work_cast` passes.

**L3. The five baseline methods changed behavior for edge inputs.**
- NaN or ±Inf input now returns zeros, where it used to propagate NaN (`:496-497`). This matches the Blackwell convention, but in a training loop it can hide a divergence.
- `_normalized` keeps the `+1e-7` epsilon, while `schatten_switch` divides exactly and has a 1e-12 floor (`:661`). So for 1e-12 < ‖G‖_F ≲ 1e-6, the SS methods return a polar factor and the baseline methods return a scaled-down result (for example ‖G‖ = 5e-30 gives `out ≈ 3e-23`).
- Document both behaviors.

**L4. The 32-bit index guard checks only the short side.**
`_METAL_MAX_N` (`:309`, `:367`, `:393`, `:766`) bounds `n`. The SIMD-group shader indexes `x[r * k + h]` in `uint` (`:271-272`), which wraps when `n·k ≥ 2³²`. The MPP path builds `dextents<int, 2>(k, n)` (`:241`), which may limit `n·k` to below 2³¹. Current optimizer shapes are far below these limits. This is a residual of Cycle 2 L1. Add a guard on `x.size`.

**L5. BF16 block orders are not full Latin squares.**
`_block_orders` (`:683-693`) is balanced only when `repeats` is a multiple of the arm count. FP16 has 10 arms × 20 blocks, which is balanced (spread 0). BF16 has 6 arms × 20 blocks, so slot counts are 3 or 4 (spread 1). The effect on paired ratios is negligible, but the docstring "balance every method across all time slots" overclaims.

**L6. The primitive tolerance comment cites FP64, but the check compares against dense MLX** (`bench_mlx_primitives.py:61-69`).
`max_abs_diff_vs_dense` is a consistency check between dense and Metal, not an accuracy check. The largest ratio of diff to tolerance is 0.326 (BF16 symprod: 0.0078 against 0.025). FP16 and BF16 SYRK are bit-identical to dense at 4096².

**L7. Test quality.**
- `test_live_schatten_rank_one_sweep` and `test_live_schatten_fable_r1` skip without MLX, but the dense half runs on the NumPy emulation. I ported the sweep to CPU and all six circuit/dtype cases behave as claimed (see "Numerical audit"). Add a CPU variant so Linux CI covers the BF16 gate.
- `test_schatten_quality_and_invalid_inputs` uses only n ≤ 13 (prescale 1) and loose limits (`polar_error < .16`, `sigma_max ≤ 1.25`). The prescale-64 path is covered on CPU only by the coordinate spike.
- No test pins the M2 fallback. `test_lazy_metal_compile_failure_falls_back_once` checks only the PS path.
- I found no tautologies. The A/B summary test recomputes percentiles independently and checks Latin rows and columns.

**L8. Unexplained cross-session drift of the FP32 baselines.**
Against `results/m5_max_dense_cycle2.json` (same MLX 0.32.5, same Darwin 27.2.0), the FP32 latencies are 4–13% slower (for example `gram_ns` at 1024×4096: 2.585 against 2.283 ms). Polar errors are no longer bit-identical (|Δ| ≤ 0.001). The error change is expected from the new power-of-two normalization. The extra normalization passes and the session state both plausibly explain the slowdown. No within-session claim depends on this.

**L9. A Claude Opus audit file now contains GPT-authored content.**
`710240f` appends an "M5 Max implementation follow-up" section to `audits/AUDIT_cycle2_t2_02_claude_opus.md`. The table in that section is accurate (I checked all three rows). Still, follow-up evidence belongs in the implementer's report, so that each audit file keeps a single author.

**L10. Tall inputs are still not benchmarked on the Metal path** (Cycle 2 L6). All A/B shapes are wide or square.

## Provenance and hygiene checks

| Check | Result |
|---|---|
| Lineage `6236a52 → 7eacbb6 → 710240f` | Verified. Linear, with one parent each. `origin/exp/c3-m5-max-ab-ss3` is `710240f`. |
| `7eacbb6` scope | `kernels_mlx.py`, `bench_mlx_primitives.py`, `tests/test_mlx.py` only |
| `710240f` scope | Results, report, Cycle 2 audit file, **`state/BACKLOG.md`, `state/ledger.jsonl`** (M1) |
| `README.md` edits | None |
| Baseline `METHODS` keys | `jordan_ns5, polar_express, gram_ns, PS2_55, PS3_55`, identical to `6236a52`. The SS methods are in a separate `SCHATTEN_METHODS` / `ALL_METHODS`. |
| Strict JSON parse (all three files, non-finite constants rejected) | Pass |
| `git_sha` / `git_dirty` (all three) | `7eacbb6516ef71d8ff19f63c723b861d3dd799fb` / `False` |
| `source_sha256` of `kernels_mlx.py` at `7eacbb6` | `eb2bdc11b4da2b9ed48b197ef6a5fbd234033ab21473991ba0d5b92fd957ea15`, matches all three files |
| `source_sha256` of `bench_mlx_primitives.py` at `7eacbb6` | `c8268ca313c293011a350f470f7afffc0a02d1377fc01da75e27134de9a89709`, matches |
| Run order | Primitives 11:05:57–11:06:00Z, A/B 11:06:00–11:10:36Z, FP32 gate 11:10:36–11:10:41Z. Sequential, no overlap. |
| Host | `Ryans-MacBook-Pro.local`, Apple M5 Max, `Mac17,7`, 128 GiB, Darwin 27.2.0, MLX 0.32.5 |
| Merge into `origin/main` | **Conflicts in 2 `state/*` files** (M1). All other paths are clean. |
| Test counts in the report (82 baseline / 101 final) | Consistent by collection arithmetic. On Linux the baseline collects 52, plus 30 MLX-backend parametrizations, which gives 82. The final collects 67, plus 34, which gives 101. Not replayable here. |

## Tests on Linux CPU

```
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=. .venv/bin/pytest tests/test_mlx.py -q
56 passed, 11 skipped in 2.20s
```

All 11 skips are `Requires native Apple Metal`: `test_live_metal_syrk_and_fused_symprod` ×3, `test_live_schatten_fable_r1` ×3, `test_live_schatten_rank_one_sweep` ×4 and `test_primitives_quick_cli` ×1. There were no failures.

## Numerical and code audit

### Circuit specs against `origin/main:research/polar_opt/schatten_switch_blackwell.py`

I parsed both `_CIRCUIT_SPECS` literals with `ast.literal_eval` and compared the dicts for equality.

| Circuit | Equal to `origin/main` | Equal to `6236a52` | `u_safe` | `lo` | `bf16_verified` | Stages (squarings) |
|---|---|---|---|---|---|---|
| SS3_54 | **yes, every field** | yes | 1.05 | 0.00286 | False | 5 + 4 |
| SS3_53 | **yes** | yes | 1.05 | 0.003 | False | 5 + 3 |
| SS3_35 | **yes** | yes | 1.05 | 0.004 | True | 3 + 5 |

The fold matches Blackwell `_fold` / `_fold_and_row0_axpy_kernel` term by term: row 0 is `S + (α b0/κ) R + (α c0/κ²) I`, later rows are `a·t/f_src²`, `b·t/f_add` and `c·t`, and the last row has `t = √κ`. The scale is the same: prescale 64 for n ≥ 1024, `α = 2¹³/P⁴`, `κ = 1/(u_safe² · upper)`, and `upper` is the Schatten-8 bound with the trace-bound minimum. `SS3_34` is the only main circuit that is not ported. The only formula deviation is the `k` slack (L1).

### BF16 gate and the dense BF16 SS3_35 folded-affine fix (CPU emulation)

Coherent rank-one sweep, ported from `test_live_schatten_rank_one_sweep` (64×128 sign matrix, 1201 amplitudes, production trace bound), dense arm:

| Circuit | Dtype | max ‖y‖_F (current) | Round-square-first alternative | Cap 1.25 |
|---|---|---|---|---|
| SS3_54 | FP16 | 1.084 | — | pass |
| SS3_53 | FP16 | 1.090 | — | pass |
| SS3_35 | FP16 | 1.090 | — | pass |
| SS3_35 | BF16 | **1.127** | **4.353** | pass (fix necessary and sufficient) |
| SS3_54 | BF16 (`allow_unsafe_bf16`) | 3.8e8 | 1.3e14 | fail, gate justified |
| SS3_53 | BF16 (`allow_unsafe_bf16`) | 4.7e10 | 8.6e9 | fail, gate justified |

`schatten_switch(..., "bfloat16")` raises `ValueError` for SS3_54 and SS3_53 unless `allow_unsafe_bf16=True`. SS3_35 passes. `_measure` skips the same 48 BF16 arms. The 120 FP16 and 72 BF16 rows match `methods × shapes × seeds × arms`.

Counterexample search at 1024² (prescale 64), FP16 and BF16, comparing PS3_55 with SS3_54, SS3_53 and SS3_35: coherent rank-one signs, coordinate spike plus 1e-6 noise, a four-spike spectrum (1, 0.9, 0.5, 0.25), a log spectrum with κ = 1e6, the Fable R1 matrix `pI + (top−p)/n·11ᵀ` at ratio 0.246, and Gaussian input. All outputs were finite. The largest σ_max was 1.138 (SS3_35, BF16, Gaussian), below the 1.25 cap. SS3_54 in BF16 on rank-one signs gives σ = 0.508, which again shows why the gate exists. At κ = 1e6 every method leaves the singular values below `lo` unconverged, as designed. SS errors (0.677–0.711) are no worse than PS3_55 (0.720–0.722).

FP16 dense `j == 1` rescale (`:633-639`): I compared it with a one-rounding FP16 emulation (the same arithmetic as the Metal epilogue) on Gaussian 1024², Gaussian 512×2048, rank-one signs and Fable R1. The errors agree within 0.001 and σ_max within 0.02. I found no subnormal precision loss.

### Host synchronization

`_safe_norm`, `_normalized`, `_upper`, `_schatten_first_segment` and `schatten_switch` contain no `.item()`, `float()`, `bool()` or `tolist()` on device values. The data-dependent `κ`, `t` and the stacked `(fa, fb, fc)` remain MLX/Torch arrays. Python `if` statements branch only on table constants (`c0`) and on shapes.

### Metal shader static audit

- **Tile decode** (`:226-233`): `col` comes from a float `sqrt` with ±1 correction. At the guard limit (n = 65535, 32-tile, T = 2,098,176), `8T+1` exceeds 2²⁴ by 8,193, so the rounding error moves `sqrt` by about 1e-4. The single correction is enough. `col·(col+1)` stays below 2³².
- **Metal 4 path** (`:237-255`): `tensor<device T, dextents<int,2>>(x, dims(k, n))` places the column extent first. The slices `(0, row·B)` and `(0, col·B)` select row blocks. `matmul2d_descriptor(B, B, dynamic, false, true, false)` computes `left · rightᵀ`. The destination index `xy[0]` is the column and `xy[1]` is the row. The `r ≤ c` upper-tile store with mirroring is consistent. The `const_cast` is the documented MPP requirement. Tiny inputs skip this path because constant-space `x` cannot bind there.
- **SIMD-group path** (`:258-295`): 4 SIMD groups each cover a 16×16 quadrant with 2×2 8×8 accumulators. The transposed `simdgroup_load` of `right` gives `left · rightᵀ`. Threadgroup memory is 8 KiB. The barriers are correct.
- **Coefficients**: static coefficients use `_metal_coefficients` (an LRU-cached FP32 `mx.array`). Lazy coefficients use `xp.stack([fa, fb, fc])`. Both have 3 elements, so MLX binds both in the same constant address space, and the one cached kernel signature serves both.
- **Cache keys**: `_metal_kernel` is cached by `(dtype, fused, tiny)`. Kernel names are `polar_{syrk,symprod}_{64|32|tiny_32}`, with `T` and `B` as template arguments, so there is no name collision (Cycle 2 F7).
- **Fallbacks**: they are correct for PS (documented round-first), but unsafe for SS (M2).

## Independent recomputation from raw samples

All computations used NumPy linear percentiles, single-threaded.

### Summary

| Item | Rows / cells | Result |
|---|---|---|
| Primitive rows: median, p25, p75, IQR, min, `latency_ms` | 60 | 0 mismatches |
| Primitive tiles | 36 Metal rows | all `tile_size = 64`. Dense rows are `None`. |
| Primitive `diverged` / max diff ÷ tolerance | 60 | 0 / 0.326 |
| Primitive Latin squares | 12 cases × 20 blocks × 5 methods | full squares, spread 0 |
| Primitive table in the report (median / IQR / min of within-block ratios) | 48 cells | **all 48 match to the printed precision** |
| A/B rows: stats, `sleep_detected`, `diverged`, `finite` | 192 | 0 mismatches. 0 sleep, 0 diverged, 0 non-finite. |
| A/B tiles | 96 Metal rows | SYRK tile 64 on all 96. Symprod tile 64 on all 72 PS/SS rows. `None` on the 24 Polar Express rows and on all 96 dense rows. |
| `paired_ab_summary` (`median`, `p25`, `p75`, `iqr`, `min`, `max`, `n_blocks`) | 32 | max abs deviation **0**. `n_blocks = 60` on all 32. |
| Report A/B table | 32 rows × 6 statistics | all match |
| Report SS-vs-PS3_55 table (error median over seeds / median of 60 PS3_55÷SS ratios) | 16 rows | **all match** |
| A/B Latin squares | 24 cases | FP16 spread 0; BF16 spread 1 (L5) |
| A/B samples > 1.5 × row min | 3840 | **0** (Cycle 2 had up to 86/300 at 4096²) |
| Primitive samples > 1.5 × row min | 1200 | 100, all at ≤ 2048² sub-ms or few-ms cells. None at 4096². |
| `sleep_detected` recomputed from `sample_wall_s` | 5140 samples | 0 flagged. Largest gap between wall and perf counter: 0.079 ms. |
| Power and thermal snapshots | 82 | 82/82 identical: AC, 80%, not charging, no warning, **clamshell Yes** |
| FP32 gate rows | 10 | `use_metal_syrk = False`, `use_metal_symprod = False`, tiles `None`, 0 diverged, FP32 output, on all 10 |

### Per-seed consistency (paired dense/metal medians)

95 of the 96 per-seed medians lie within −3.9% to +3.0% of the pooled median. The one outlier is 4096² FP16 PS3_55, seed 2, at −6.3% (2.519 / 2.515 / 2.327; IQR 0.214). The seed 2 case at 4096² FP16 was the most ramped (M3). All 2048² cells agree within 1.7%.

### Headline claims checked

| Claim | Source | Recomputed |
|---|---|---|
| 4096² FP16 Metal SS3_54 error 0.0539 vs PS3_55 0.1259, speed 0.89× | report line 5, ledger | 0.0539 / 0.1259 / 0.89× ✓ |
| Metal PS3_55 2.48× its own dense FP16 | ledger | 4096²: 2.484, 2048²: 2.489 ✓ |
| 2048² Metal PS3_55 1.80× Metal Polar Express (FP16 and BF16) | backlog, audit follow-up | 1.800 / 1.799 ✓ |
| 2048² dense FP32 PS3_55 0.88× Polar Express | backlog, audit follow-up | 0.878 within block (0.880 ratio of medians) ✓ |
| 5,140 timed calls, 0 flagged | report line 3 | 1200 + 3840 + 100 = 5140, 0 flagged ✓ |
| Dense and Metal error agree | implicit | largest dense−metal \|Δ error\| = 0.0021 ✓ |
| No divergence | all | 0/60, 0/192, 0/10. Largest σ_max is 1.145 (SS3_35 dense BF16 4096²), below the 2.0 divergence limit and the 1.25 cap. ✓ |

### SS3 vs PS3_55 (accuracy against speed)

SS3_54 has the lowest error in every FP16 cell. It cuts PS3_55 error by about 42% at 512×2048 and 1024×4096 (for example 0.0637 → 0.0372). At the square shapes it cuts error by 52–57% (0.1148 → 0.0557 at 2048², 0.1260 → 0.0540 at 4096²). It costs 0.83–0.89× speed on the Metal arm. On the dense arm it is break-even or faster at ≥ 2048² (1.00–1.03×). SS3_35 is the only BF16 option. It lowers error 3–37% depending on shape, at 0.86–0.94× Metal speed. The Metal kernels favor PS3_55 more than SS. The reason is that PS3_55 does all of its work in fused symprod steps, while SS3 adds the `square` pass and a longer segment 1. The report states these trade-offs correctly and does not overclaim.

## Disposition of the Cycle 2 findings (`AUDIT_cycle2_t2_02_claude_opus.md`)

| ID | Cycle 2 finding | Status at `710240f` | Evidence |
|---|---|---|---|
| F1 | FP32 Metal path slower but enabled | **Closed** | `_Work.metal` is gated on `_METAL_FAST_DTYPES` (`:447`). 10/10 FP32 gate rows are dense with null tiles. The primitive FP32 SYRK/dense ratio of 0.67–0.70× confirms that the gate is needed. |
| F2 | Dense and triangular measured in separate sessions under different thermal states | **Closed (method); open (open-lid replication)** | Arms are paired in the same block, 60 blocks per cell, and paired ratios are immune to the ramp (M3). The lid is still closed. |
| F3 | Selected tile not recorded, probe not checked | **Closed** | `metal_syrk_tile` and `metal_symprod_tile` are recorded per row (all 64). The probe checks the numerical output (`:349-354`). Tests pin the tile on live hardware (`test_mlx.py:235`). |
| F4 | "≤ 2048² is clean" was wrong | **Closed for this run** | 0/3840 A/B samples exceed 1.5× the row minimum. Per-seed medians agree within ±4%. |
| F5 | Primitive profile assembled by hand | **Closed** | `bench_mlx_primitives.py` is committed and emits `git_sha`, `source_sha256` (both files), timestamp, hardware, command and per-case thermal data. Its tolerance comment is misleading (L6). |
| F6 | Source conflicts with `main` | **Closed for source, reopened for state** | Code, tests and results merge cleanly. `state/*` conflicts (M1). |
| F7 | Fallback and pre-M5 paths never run; tiny-kernel name collision | **Partial** | The names are now distinct (closed). Pre-M5 and fallback paths are still not run on hardware, and the fallback is unsafe for SS (M2). |
| F8 | Goal accounting for −14% at 2048² | **Partial** | The backlog states FP32 0.88× and Metal FP16/BF16 1.80× correctly, but marks T2-02 `[x]` (M1). |
| L1 | 32-bit index overflow | **Closed for `n`; residual for `n·k`** | `_METAL_MAX_N = 65535` guards (L4) |
| L2 | Upper-triangle addend | **Closed (documented)** | `metal_symprod` docstring. All internal addends are exactly mirrored or built elementwise from symmetric buffers. |
| L3 | Constant-space addend in the 64-tile kernel | **Closed** | `addend.size < 8` selects the tiny 32-tile kernel (`:392`) |
| L4 | Live tests do not match production orientation | **Closed (as reported; not replayed)** | `(17,33)`, `(35,67)` and `(2048,4096)` were added to the live test |
| L5 | Sleep cannot be placed on samples | **Closed (instrumentation); open (open-lid, real thermal pressure)** | `sample_wall_s` and the sleep flag work (0/5140). `pmset` cannot see the 4096² ramp (M3). |
| L6 | Tall inputs untested on the Metal path | **Open** | All A/B shapes are wide or square (L10) |

## Merge and promotion recommendation for `origin/exp/c3-m5-max-ab-ss3`

1. **Promote** the C3-04 Tier-1 results as written in `reports/m5_max_cycle3.md`, with two added caveats: the run used a closed lid (`AppleClamshellState = Yes`), and absolute 4096² latencies ramp 8–11% within each case while the paired ratios do not (M3).
2. **Before merge:** remove the `state/BACKLOG.md` and `state/ledger.jsonl` edits from the branch, or rebase onto `1c7b938` and resolve them. Keep T2-02 open for FP32 (M1). After that, everything else merges cleanly.
3. **Before enabling `use_metal_syrk` beyond M5 Max:** make the `_metal_symprod_vec` fallback FP32-accurate, or disable `work.metal` when no kernel is available. Add a CPU test that pins this (M2).
4. **Open items for Tier 2 or the next cycle:** live replay of `tests/test_mlx.py` on the M5 Max after the merge (101 expected), an open-lid run under `caffeinate -dimsu` at 4096² with GPU power telemetry, and a CPU variant of the rank-one sweep (L7). L1–L6 and L8–L10 can wait.

## Reproduction notes

- Spec equality: run `ast.literal_eval` on the `_CIRCUIT_SPECS` assignment in both files, then compare the dicts with `==`.
- Fallback emulation (M2): monkeypatch `kernels._metal_symprod_vec = lambda x, A, (a,b,c): r(a*r(x@x.T) + b*A + c*I)` with `r = partial(kernels._cpu_round, dtype=...)`, set `metal_syrk = lambda x: r(x@x.T)` and `_metal_coefficients = lambda *c: c`, then wrap `_Work.__init__` to force `self.metal = use and dtype in _METAL_FAST_DTYPES`.
- Round-square-first alternative (BF16 sweep): call `_schatten_first_segment` with a `_Work` whose `.dtype` is set to a non-`"bfloat16"` tag. BF16 casts are kept, so the dense square is rounded before the affine sum.
- Ramp statistics (M3): for every row, compute `mean(samples[-5:]) / mean(samples[:5])` and the Spearman correlation between block index and latency. For the paired ratios, use the dense and metal samples from the same block index.
