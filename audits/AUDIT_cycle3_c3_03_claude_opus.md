# C3-03 audit: small-`n` launch fusion and fresh Tier-1 latency refresh (RTX 5080, `sm_120`)

- **Auditor:** Claude Opus 5.5 (adversarial, live GPU) · **Date:** 2026-11-26
- **Branch audited:** `origin/exp/c3-03-small-n-fusion` · code `08d4102188606e1b900b8162e7aad8834f5fbc90` · results `a3861e8bd568151e5c329edfc818e618d44e635c` (author: Claude Fable 5.2)
- **Environment:** RTX 5080 (exclusive), torch `2.14.1+cu130`, Triton `3.8.0`, Python 3.13.12. Host load average was 2.9–3.7 during the GPU runs (a CPU-only delegate ran at the same time).
- **Hygiene:** I changed no source, test, result or state file. Scratch scripts were in the session scratchpad and `$HOME/tmp/pi-c303audit.*`. I deleted them after the run. I extracted the baseline tree with `git archive 4b7a792 research | tar -x` (this writes nothing to the repository). For one test run I put a temporary symlink at `artifacts/harvested_trajectories.pt` (git-ignored, read-only target) and removed it after the run. `git status --porcelain` was empty after the run.

## Executive verdict: **PASS_WITH_CAVEATS**

| Gate | Verdict |
|---|---|
| Direct Triton 3.8 C launcher | **PASS.** Bitwise equal to the public path on side streams, graph capture on side streams and user streams, unaligned and strided views, after `_PLANS.clear()`, under `torch.compile` (default and `reduce-overhead`), and with a fake Triton version (fallback). Launch hooks route to the public path. |
| Tile-norm fusion and packed scratch | **PASS.** Tile partials match FP64 within 5.6e-8 relative on 26 shape and dtype cells from `(1,1)` to `(8193,8200)` and batch 16. Warm calls write no slot outside their tiles. Fold coefficients match the host `_kappas`/`_fold` within 3.7e-7. |
| Extreme inputs, `sigma_hat`, custom circuits | **PASS.** Zeros, NaN and Inf return exact zeros, per matrix. Power-of-two rescaling is bitwise invariant. Invalid `sigma_hat` raises an error. Segment 1 with 2–8 rows, `has_b0=0`, a residual, and a one-segment circuit all match the FP16 reference within 1.2e-3. Nine rows raises an error. |
| Zero sync, bitwise graph replay, launch counts | **PASS.** `set_sync_debug_mode("error")` is clean, with 0 `Memcpy DtoH`. Graph replay equals eager bitwise on 180 adversarial overwrites. Kernel counts are 16/15/15/14/16 on every shape tested. |
| Tests, provenance, JSON, recomputation | **PASS.** 466 passed and 10 skipped. With the trajectories present, the 7 data-dependent skips pass, so 473 tests pass and 3 skip by design. Both JSONs are strict, and 1,290 recomputed fields match exactly. |
| Live latency spot-check | **PASS_WITH_CAVEATS.** The JSON ratios reproduce within ±0.01. SS3_54 is still 0.984–0.995× of PS3_55 at (512, 2048) in CUDA-graph time (M1). PS3_55 eager-event time is +2.1 µs at (512, 2048) (L2). |

**Merge disposition:** Merge `exp/c3-03-small-n-fusion` into `main`. I found no correctness, memory-safety or provenance defect.
**Promotion disposition:** Promote both JSONs as the canonical C3-03 latency record. They supersede the latency fields of `schatten_switch_search_cycle2.json`. Do not say "SS3 circuits are faster than PS3_55 at every shape". SS3_54, the most accurate circuit, is still behind PS3_55 at (512, 2048) and at parity on the median small Tier-1 case (M1). The commit message already says this.

## Findings, ranked by severity

No CRITICAL or HIGH findings.

### MEDIUM

**M1. The C3-03 goal is met for SS3_53, SS3_35 and SS3_34, but not fully for SS3_54 at the smallest sizes.**
- The goal was Cycle 2 audit M1: SS3 circuits were slower than `ps3_55_blackwell` at (512, 2048) and (1024, 4096). See the `08d4102` commit message.
- CUDA-graph ratio of PS3_55 median to circuit median (>1 means the circuit is faster):
  - From the JSON (recomputed): SS3_54 is **0.987×** at (512, 2048) and 1.031× at (1024, 4096). SS3_53 is 1.034× and 1.070×. SS3_34 is 1.093× and 1.123×.
  - Live spot-check, two independent runs: SS3_54 is **0.995× and 0.984×** at (512, 2048).
- Tier-1 (117 cases, 128–256 short side): the median per-case speedup of SS3_54 against PS3_55 is **0.991×** (graph), 0.999× (event) and 1.008× (wall). SS3_53 and SS3_35 are 1.040×, and SS3_34 is 1.106×.
- The fusion gains are real: SS3_54 is 1.09–1.10× faster than Cycle 2 at (512, 2048) and (1024, 4096), and 1.72–1.77× faster in wall time at (512, 2048).
- Impact: A default switch from PS3_55 to SS3_54 at `n ≤ 512` gives better accuracy (median error 0.0385 against 0.0641), but it does not give lower latency. Make that decision on accuracy, not on speed.

### LOW

**L1. A CUDA-graph capture of a shape that was not warmed up fails with an opaque CUDA error, not with the clear guard message.**
- The guard is at `research/polar_opt/schatten_switch_blackwell.py:396`, in `_seg1_table`. It runs only when the circuit has never been used on that device. The table is cached per circuit, not per shape.
- For a shape that was not warmed up, `_syrk_norm_kernel` autotunes inside the capture first (`polar_switch_blackwell.py:166`).
- Repro: warm `SS3_54` at (256, 1024), then capture (300, 700). The result is `AcceleratorError: CUDA error: operation failed due to a previous error during capture` (`cudaErrorStreamCaptureInvalidated`). Eager calls after this still work and are bitwise equal.
- The docstring documents the warm-up requirement, so this is about error quality. Fix: in `_launch`, raise a clear error when `plan is None` and `torch.cuda.is_current_stream_capturing()`.
- A capture that changes only the input alignment (warm-up aligned, capture with `ptr % 16 == 4`) works and is bitwise correct. Only single-config kernels see the input pointer.

**L2. PS3_55 eager-event time is +2.1 µs (1.6%) at (512, 2048). The cause is not the direct launcher.**
- Live A/B, new tree against the `4b7a792` tree, alternated new/base/new/base: eager-event is 131.1/131.1 µs against 129.0/129.0 µs. The JSON shows the same direction, 0.1301 → 0.1311 ms (0.992×).
- Over the same A/B, CUDA graph is neutral (124.9–126.8 µs against 124.9 µs), and wall time is **1.31–1.32× faster** (297 → 225 µs).
- In-process A/B in the new tree, direct path against the public path (forced with a no-op launch hook): 130.7/130.9 µs against 130.7/131.1 µs. The direct launcher adds no GPU time.
- The delta is reproducible across processes but small. It is probably process-level (autotune choice or allocation layout).
- So the "zero regression on PS3_55" claim holds for graph and wall time, and there is a 1.6% eager-event caveat at this one shape. At (1024, 4096) and larger, the change is 0.992–1.003×.

**L3. The canonical-output gate ignores `--trajectories` and accepts `--warmups > 5`.**
- Location: `bench_c3_03_tier1_refresh.py:385-389`.
- `--trajectories /other.pt` and `--warmups 10` both resolve to the canonical paths.
- The recorded run used the default file. Its `trajectories_sha256` `310ec381…` matches `/home/ryan/Documents/projects/ai/artifacts/harvested_trajectories.pt`, so there is no impact today.
- All other attacks are refused: `--quick` with a canonical path, a `../` alias, `--blocks`/`--iters`/`--seed` without an output, crossed part paths, and a dirty tree (covered by the test).

**L4. The 8-row limit of segment 1 is checked late.**
- Location: `schatten_switch_blackwell.py:394`.
- A 9-row circuit is accepted by `SchattenSwitchCircuit` and by `schatten_switch_reference`. The native call raises `ValueError` only after four kernels have launched. It raises again on every call, because nothing is cached.
- Fix: check this in `__post_init__`.

**L5. The direct launcher uses the current device's stream (`polar_switch_blackwell.py:182`), but the plan key holds `args[0].device`.**
- This is the same as Triton's public `CompiledKernel.__getitem__` (`get_current_device()`), so this change adds no regression.
- On a multi-GPU host, a tensor that is not on the current device would launch into the wrong context on both paths.
- This cannot be tested on a single-GPU host. It is informational.

**L6 (pre-existing, not caused by C3-03). The largest output singular value drifts at very long K.**
- At shape (4, 2^25), SS3_54 gives σ_max = 1.135 and PS3_55 gives 1.179. At (2^25, 2), SS3_54 gives 1.150. The design target is `1 + error = 1.08`.
- The outputs are **bitwise identical** to the `4b7a792` baseline, so this is the FP16 Gram accumulation bias (Fable M1), not the fusion. PS3_55 drifts more.

**Nit.** `bench_cuda_graph.bench_matrix` (`bench_cuda_graph.py:173`) runs `del G`, but the caller `bench_shape` still holds a reference to `G` during `torch.cuda.empty_cache()`. One input buffer stays allocated slightly longer. Otherwise the 7-line refactor keeps the old behavior: same seed, shape, mutation check and return schema.

## Evidence

### 1. Direct C launcher (`polar_switch_blackwell.py:122-183`)

Code review against Triton 3.8 `CompiledKernel.__getitem__` and `CudaLauncher.__call__`:
- The argument order is `(gx, gy, gz, stream, function, coop, pdl, packed_metadata, launch_metadata, enter_hook, exit_hook, global_scratch, profile_scratch, arg_annotations, kernel_signature, args)`. This matches `driver.py:331-333`.
- The fast path requires all of these: Triton `3.8.*`, a `CudaLauncher`, an unwrapped `launch` (no tensor descriptors), no gsan, and zero global and profile scratch.
- Hooks are checked on every launch (`.calls`).
- The stream is `driver.active.get_current_stream`, which is the same source that Triton uses.

Live probe (warm (512, 2048) and other shapes; each value is the max abs diff or `torch.equal`):

```
plans 13 / direct 13          sidestream_{SS3_54,SS3_53,PS3_55}: 0.0 (busy side stream, 20 GEMMs queued first)
graph (torch.cuda.graph side stream) x3 overwrites (randn, 1e-30*, 1e20*) -> True x9
graph_userstream_* (capture on explicit torch.cuda.Stream) -> True x3
ptr%16 = 4, 8, 12, 0 (fp32 view offset 1..4) -> 0.0 for all 12 cells; noncontig .mT -> 0.0; strided [:, ::2] -> 0.0
_PLANS.clear() then re-plan -> 0.0 x3; graph captured before clear replays correctly after clear+re-plan -> True
launch_enter_hook added -> 14 Triton hook calls per SS3_54 call (16 kernels - 2 cuBLAS), output bitwise; removed -> 0 calls
torch.compile default / reduce-overhead around ss3_54 and ps3_55 -> 0.0 / 0.0 / 0.0 / 0.0
capture with unaligned input after aligned-only warm-up -> True;  capture of a cold shape -> cudaErrorStreamCaptureInvalidated (L1)
triton.__version__ = "3.9.0" + _PLANS.clear() -> direct plans 0/19, SS3_54/SS3_34/PS3_55 bitwise equal (fallback)
```

### 2. Tile-norm fusion and scratch (`schatten_switch_blackwell.py:410-433, 530-610, 613-707, 712-745`)

Review:
- The weights `(rm<=rn)+(rm<rn)` are correct for the upper triangle walked by `_tri_tile` (i ≤ j). They mirror `_store_sym` exactly.
- The scratch capacity uses `_MIN_BLOCK=32`, so every autotune trial stays in bounds.
- `_fold_and_row0_axpy_kernel` has a single config, so it is not benchmarked and the in-place axpy runs once.
- `_symprod_ptr_kernel` keeps `restore_value=["o_ptr"]` for rows where the output aliases the additive operand, such as `(0, 2, 0)`.

Probe on autotuned configs with NaN-filled scratch. "Relative" is the max relative error against FP64 `vector_norm` and the FP64 tile reference:

```
shape             dtype  BLOCK r/s  tiles r/s   capacity  tile_r    tile_s    norm_r    norm_s    sym
(1,1),(1,7),(7,1),(4,1,9)  fp16/bf16   32/32   1/1   1   0.0       0.0       0.0       0.0       True
(97,331)          fp16   32/32      10/10       10        1.6e-08   1.6e-08   1.3e-09   4.0e-09   True
(5,33,17)         fp16   32/64      1/1         1         4.3e-08   3.2e-08   2.1e-08   1.6e-08   True
(515,1029)        fp16   32/32      153/153     153       2.9e-08   3.2e-08   2.0e-09   3.6e-09   True
(3,1025,1100)     fp16   64/64      153/153     561       2.6e-08   2.7e-08   2.9e-09   3.8e-09   True
(16,256,512)      fp16   64/64      10/10       36        2.8e-08   2.2e-08   4.9e-09   6.6e-09   True
(4,1024,4096)     fp16   64/64      136/136     528       2.6e-08   3.1e-08   1.6e-09   3.0e-09   True
(4096,4096)       fp16   64/64      2080/2080   8256      2.8e-08   1.6e-08   3.1e-09   3.6e-10   True
(4097,4100)       fp16   64/64      2145/2145   8385      2.9e-08   1.6e-08   2.4e-09   4.0e-10   True
(8192,8192)       fp16   128/64     2080/8256   32896     2.3e-08   0.0       2.3e-09   0.0       True
(8193,8200)       fp16   64/64      8385/8385   33153     2.9e-08   0.0       2.4e-09   0.0       True
bf16 (same 13 shapes up to 4097x4100): every tile_r/tile_s <= 2.7e-08, norm <= 1.3e-08, symmetric
input-norm partials (FP64 sum of squares) vs FP64: <= 5.6e-08 on every shape
```

On a warm call, every scratch slot between the real tile count and the region capacity stays NaN. Coefficient rows against the host fold, using FP64 norms of the stored buffers:

```
fold_SS3_54_(97,331)       coeff_rel 3.7e-07  first-row S update max abs 1.2e-04 (1 fp16 ulp at |S|~0.1)
fold_SS3_34_(3,1025,1100)  coeff_rel 3.7e-07  ... (4,1024,4096) 3.7e-07 ... (4096,4096) 3.6e-07; warm gap all-NaN: True x8
```

### 3. Extreme inputs, `sigma_hat`, custom circuits

```
(4,256,1024) SS3_54:  zeros, 1e-44*, 1e-38*, 2^-60* (||G||_F <= 1e-12), NaN, +Inf, -Inf, one NaN entry, one Inf entry -> all-zero, finite
                      1e-6* -> 1.7e-4 vs unit, 1e37* -> 1.7e-4, 2^100* -> 0.0 (bitwise), 3.4e38 const / sign*3.4e38 -> finite, non-zero
fp64 inputs:          1e300*, 3.5e38 const (> FLT_MAX) -> zeros (documented); 1e30* -> valid (2.2e-4); 1e-300* -> zeros
mixed batch [1, 1e-30, 1e25, 0, NaN] -> zero flags [F, T, F, T, T]; each matrix bitwise equal to its single-matrix call
power-of-two scale 2^-40, 2^40, 2^100 -> bitwise equal to unit scale
sigma_hat 1.0 / 0.5 -> matches fp16 reference (2.8e-4 / 2.3e-4); 1e-3, 1e-8, 1e-20 (below sigma_max/||G||_F) -> zeros
  (reference gives non-finite there: unsafe regime, documented); NaN, 0, -1, Inf, tensor [1, NaN], wrong length -> ValueError
custom segment-1 rows 2..8 on (64,256), (3,97,331), (300,70): native vs fp16 ref 2.1e-4..7.9e-4; 9 rows -> ValueError (L4)
has_b0=0 (benign rows) 4.6e-4 / 3.8e-4; has_b0=0 and has_c0=0 (programs=0) 2.1e-4; seg-1 residual 0.25 -> 1.2e-3;
one segment 4.6e-4..5.4e-4; one segment + residual 0.5 -> 2.3e-4..4.1e-4
(SS3_54 rows with b0 forced to 0 diverge in the FP64 reference too: the circuit is the cause, not the kernel)
K = 2^25: (1,2^25) σ 1.043; (4,2^25) σ 1.129..1.135; (2^25,2) σ 1.150..1.151: bitwise equal to 4b7a792 (L6)
```

### 4. Regression against the `4b7a792` baseline (same seeded inputs, separate processes)

Out of 69 (circuit, shape) cells, from `(1,1)` to `(4096,4096)`, `(4,1024,4096)`, `(4,2^25)` and `(2^25,2)`:
- 62 are **bitwise identical**, including every PS3_55 cell, the BF16 SS3_34 and SS3_35 cells, `scaling="frobenius"` and `sigma_hat=0.3`.
- 7 differ at the FP16 rounding level, relative 1.1e-4 to 1.2e-3. These are SS3 cells at (97,331), (5,33,17), (3,1025,1100) and (512,2048). The cause is the new norm summation order, which changes κ in the last bits.

### 5. Zero sync, graph replay, launch counts

```
set_sync_debug_mode("error") on a warm call: clean for SS3_54/53/35/34 and PS3_55 on (512,2048), (97,331), (4,1,9), (3,1025,1100)
torch.profiler kernels (excluding Memcpy/Memset), each of the 4 shapes:  SS3_54 16  SS3_53 15  SS3_35 15  SS3_34 14  PS3_55 16  (20/20 match)
Memcpy DtoH: 0 in every profile (PS3_55 has one Memset from torch.linalg.vector_norm)
Graph replay vs eager, 9 static-input overwrites each (randn, zeros, 1e-38*, 1e37*, NaN, +Inf, -Inf, mixed batch, 0.5*randn),
  5 methods x 4 shapes (512,2048), (4,1,9), (97,331), (3,1025,1100): 180/180 torch.equal (NaN-aware), max abs diff 0.0
```

### 6. Tests

```
PYTHONPATH=. /home/ryan/Documents/projects/ai/.venv/bin/pytest tests/test_schatten_switch.py tests/test_schatten_switch_fusion.py tests/test_polar_switch_blackwell.py -q
466 passed, 10 skipped, 2 warnings in 368.44s
  skips: 7 x missing artifacts/harvested_trajectories.pt (untracked in a fresh worktree), 3 x "the large size runs once" (by design)
With a temporary read-only symlink to the trajectories:
pytest ... -k "tier1_117 or quick_tier1 or tile_partials_equal"  ->  56 passed, 3 skipped   (the 7 data-dependent tests pass)
```

### 7. Provenance and strict JSON

- `git diff --stat 08d4102..a3861e8` lists only `results/cycle3_c3_03_cuda_graph_bench.json` (+23,619) and `results/cycle3_c3_03_tier1_refresh.json` (+29,459).
- Both files load with `parse_constant` rejecting NaN and Infinity, and both round-trip with `allow_nan=False`. Neither file has a duplicate key.
- Both record `git_sha = 08d4102188606e1b900b8162e7aad8834f5fbc90` and `git_dirty = false`. `launch_counts.baseline_commit = 4b7a792227eb…`.
- `trajectories_sha256 = 310ec381…cc30ec` matches `sha256sum` of the read-only artifact.
- Telemetry, graph run: SM clock 2722 → 2827 MHz, 36 → 63 °C, load average 2.73 → 2.46. Tier-1 run: 2857 MHz at the end, 45 °C, load average 2.46 → 2.54.

### 8. Recomputation from the raw JSON

```
graph bench: 5 shapes x 11 methods = 55 cells; 110 samples per clock; median(samples_ms) == median_ms (within the 5-digit sample rounding): 165/165
  Williams balance: 22 rows per shape, every method 2x in every position, every ordered adjacent pair 2x (110/110 pairs)
  capture_errors 0/55, nonfinite 0/55, graph_vs_eager_max_abs {0.0}
  ss3_speedups: 360/360 ratios exact; delta_vs_cycle2: 165/165 cells exact (before, after, speedup, delta_us); Cycle 2 sha 9a23a61
tier1: 117 cases (uniform_mp 16, power_law 24, adversarial 32, harvested_momentum 45); Williams 8x8: every position 1x, all 56 pairs 1x
  summary: 600/600 fields recomputed exactly (5 suites x 8 methods x 15 fields)
  diverged 0/117 and nonfinite 0/117 for all 8 methods; graph_vs_eager_max_abs 0.0 on 936/936 cells; capture_errors 0
  "errors match Cycle 2": suite medians vs schatten_switch_search_cycle2.json tier1 differ by <= 1.3e-3 relative (PS3_55 exactly 0)
```

Recomputed CUDA-graph ratios (PS3_55 median / circuit median, same run) and the change since Cycle 2:

| shape | SS3_54 | SS3_53 | SS3_35 | SS3_34 | SS3_54 vs C2 | PS3_55 vs C2 |
|---|---|---|---|---|---|---|
| (512, 2048) | 0.987 | 1.034 | 1.037 | 1.093 | 1.097× | 1.000× |
| (1024, 4096) | 1.031 | 1.070 | 1.075 | 1.123 | 1.094× | 1.002× |
| (2048, 8192) | 1.026 | 1.057 | 1.061 | 1.099 | 1.025× | 1.004× |
| (4096, 4096) | 1.062 | 1.132 | 1.138 | 1.222 | 1.017× | 1.000× |
| (4, 1024, 4096) | 1.017 | 1.049 | 1.055 | 1.089 | 1.026× | 1.019× |

### 9. Live RTX 5080 spot-check

- Method: `bench_cuda_graph.bench_shape`, 3 methods, 6 Williams blocks × 10 samples, 4× L2 flush, 5 warm-ups. Four processes alternated new → base → new → base.
- Telemetry: SM clock 225–2722 MHz at start (idle), 2812–2820 MHz at end. Temperature 35–44 °C at start, 63–67 °C at end. Clock-event reasons `0x400` / `0x4`. Load average 2.9–3.7.

```
                 CUDA graph PS3_55 / circuit             eager event              wall
new run 1/2      SS3_54        SS3_53          SS3_54        SS3_53     SS3_54        SS3_53
512x2048         0.995/0.984   1.032/1.034     0.993/1.005   1.032/1.041  1.013/1.006  1.045/1.031
1024x4096        1.032/1.030   1.068/1.069     1.025/1.026   1.061/1.064  1.041/1.030  1.076/1.074
2048x8192        1.031/1.027   1.062/1.058     1.022/1.025   1.050/1.057  1.024/1.022  1.055/1.054
4096x4096        1.062/1.063   1.132/1.132     1.062/1.062   1.132/1.132  1.061/1.061  1.131/1.130
base (4b7a792)   512x2048 SS3_54 0.897/0.895 graph, 0.757/0.766 wall (the Cycle 2 deficit reproduces)
PS3_55 base_ms/new_ms (2 pairs): graph 0.985/1.000, 1.000/1.001, 0.994/0.999, 1.000/1.000
                                 event 0.984/0.984 (L2), 1.002/0.999, 1.000/0.992, 1.000/1.000
                                 wall 1.320/1.311, 1.032/1.025, 0.994/0.992, 1.000/1.000
in-process direct vs public (forced by a no-op hook), 512x2048 PS3_55: event 130.7/130.9 vs 130.7/131.1 us; wall 225 vs 278 us
graph_vs_eager_max_abs 0.0 and no capture error in all 48 spot-check cells
```

## Commands

```
git fetch origin exp/c3-03-small-n-fusion && git checkout -B exp/c3-03-audit-opus origin/exp/c3-03-small-n-fusion
git diff --stat 08d4102..a3861e8
PYTHONPATH=. /home/ryan/Documents/projects/ai/.venv/bin/pytest tests/test_schatten_switch.py tests/test_schatten_switch_fusion.py tests/test_polar_switch_blackwell.py -q
git archive 4b7a792 research | tar -x -C $HOME/tmp/pi-c303audit.*/base        # baseline tree for the A/B runs
PYTHONPATH=. .venv/bin/python <scratch probe>    # one probe per evidence section (sections 1-5 and 9), deleted after the run
```
