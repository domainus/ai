# Cycle 2 Audit: Claude Opus 5.5 (Adversarial Auditor 1, high effort)

- **Repository:** `/home/ryan/Documents/projects/ai`, `main` HEAD `877c2a8413e05d89f105360a550eb382e464ba70`. The tree was clean apart from two untracked audit files that other auditors wrote.
- **Scope:** sync-free CUDA-graph kernels in `research/polar_opt/schatten_switch_blackwell.py`, `_launch`/`_PLANS` in `polar_switch_blackwell.py`, `bench_cuda_graph.py`, `tier2_candidates.py`, `tests/test_schatten_switch.py`, and `results/cycle2_cuda_graph_bench.json`.
- **Hardware:** RTX 5080 (sm_120), torch 2.14.1+cu130, Triton 3.8.0. Auditor 2 used the same GPU at the same time. Load average and GPU state are recorded with each latency number.
- **Changes made:** none to source, tests, results or `state/`. Scratch scripts lived in the session scratchpad and were deleted.

## Executive Verdict: **PASS_WITH_CAVEATS**

The 5 new kernels are correct and capture cleanly in a CUDA graph:
- Replay output is **bitwise equal** to eager for every input tested.
- A warm call makes **zero** D2H copies and zero implicit synchronizations.
- `_PLANS` keying does not break under any attack I tried.

The committed bench JSON is valid:
- The SHA is clean (`9a23a61`). Its code is identical to `877c2a8`, which only adds results.
- The Williams square is fully balanced.
- There are no capture errors, and graph-vs-eager is 0.0 everywhere.

I found no CRITICAL or HIGH issues. The caveats:
1. The SS3 speedup over PS3_55 holds only at large shapes (see M1).
2. "100% CUDA-graph capturable" needs a per-circuit warm-up (see L1).
3. fp64 inputs above the fp32 range give a silent native/reference mismatch (see M2).

---

## Verified Claims vs Caveats/Gaps

| Claim | Status | Evidence |
|---|---|---|
| Zero D2H syncs in a warm call | **Verified** | `torch.cuda.set_sync_debug_mode("error")` raised nothing. The profiler showed 0 `Memcpy DtoH` and 0 memcpy events of any kind. It also showed 2 sync events, which equals the baseline of one explicit `torch.cuda.synchronize()`. I checked 4 circuits × 3 shapes, `(512,2048)`, `(4,1,9)` and `(3,1025,1100)`, for 12/12 clean. |
| Graph replay == eager | **Verified, bitwise** (`torch.equal`, which is stricter than the test's `<1e-5`) | 4 circuits × 13 shapes × 10 adversarial overwrites of the static input, each replayed 3 times. Result: **0 mismatches** and 0 non-deterministic replays. |
| Exact zeros for dead, tiny or non-finite inputs; per-matrix scale | **Verified** | Zeros, NaN, +Inf, −Inf and 1e-38·randn all give exact zeros. A NaN in one batch member zeroes only that member. The mixed batch `[1, 1e-30, 1e25, 0]` gives zeros at indices 1 and 3 and finite polar factors at 0 and 2. 1e37·randn and an entry of 3.0e38 give finite, non-zero output, so there is no FTZ underflow in the `1/m_chunk` reciprocal at `schatten_switch_blackwell.py:404`. |
| `_PLANS` keying is sound (float args → `None`, int values, dtype, `ptr%16`) | **Verified** | Three runs (first call with `_PLANS.clear()`, cached call, and a call after a second `clear()`) are bitwise equal for SS3_54/53/34, `scaling="frobenius"` (`use_frobenius=1`), `sigma_hat=0.5` (`has_factors=1`) and BF16 SS3_35. I also built custom circuits for `has_b0=0, has_c0=0` (pure square, row-0 grid `(1,batch,1)`), `has_c0` only, a non-zero segment-1 residual (the else branch at `:699-702`) and a one-segment circuit. Shapes were `(97,331)`, `(4,1,9)`, `(5,33,17)`, `(1,1)` and `(2,1030,1031)`, the last on the prescale path. All runs are bitwise stable. Each is within 2.6e-3 relative of the FP16-emulating reference, and BF16 is within 2.1e-2. |
| 16-byte alignment of `out_coeffs[j]` | **Verified analytically** | `out_coeffs[j]` starts at byte `16·j·batch` of a fresh allocation (`:671`), so it is always 16-aligned. `_launch` keys on `data_ptr() % 16` anyway (`polar_switch_blackwell.py:125`). Unaligned `work[1]` slices, as with odd `batch·n²` at `(5,33,17)` or `(4,1,9)`, get their own plans and match bitwise. |
| Bitwise symmetry (`_store_sym`) and output aliasing | **Verified by reading** | Programs exist only for upper tiles. Each program reads C(i,j) before it writes (i,j) and (j,i), so aliasing `dst == add` is safe, as in SS3_54 row `[2,1,2]`. The row-0 axpy at `:565-577` is elementwise on bitwise-symmetric S and R, so its result stays symmetric. |
| `min(rows, cols) == 1` | **Verified** | `(1,1)`, `(1,7)`, `(7,1)`, `(4,1,9)` and `(3,7,1)` capture, replay bitwise and pass the adversarial overwrites. |
| Non-contiguous, offset and unaligned views | **Verified** | An fp32 view at `ptr % 16 == 4` (`buf[1:].view(97,331)`, where `X32` aliases `G`) is bitwise equal to its contiguous clone. The repo test `test_noncontiguous_and_offset_inputs` passes. |
| `@torch.compiler.disable` / `torch.compile` | **Verified** | `torch.compile(lambda x: ss(x)*1.0)` is bitwise equal to eager in both the default mode and `mode="reduce-overhead"`, over 4 calls each. |
| Bench: Williams square, 4× L2 flush, reduced-precision flags off, clean SHA | **Verified** | 11 methods and 22 blocks. Every method takes every position exactly 2 times. All 110 ordered adjacent pairs occur exactly 2 times. Every shape uses the same orders. `flush` is `4*L2` bytes (`bench_cuda_graph.py:180`) and is zeroed before every sample (`:127,139,150`). The flags are set at `:176-177`. `git_sha` is `9a23a61`, `git_dirty=false`, and `git diff 9a23a61 877c2a8` touches only `results/*.json`. |
| Bench: 0 capture errors, `graph_vs_eager_max_abs` | **Verified** | All 55 method×shape rows have `capture_error = None`, `finite = True` and `graph_vs_eager_max_abs = 0.0`. |
| Bench: `median_ms` matches raw samples | **Verified** | `np.median(samples_ms)` reproduces every `median_ms` (5 shapes × 11 methods × 3 clocks) to 1e-4 ms, with n = 110 each. The largest graph-clock IQR among SS3/PS3 rows is 0.094 ms at 4096² (≤1%). |
| "SS3 faster than PS3_55" | **Holds only for n ≥ 2048 (SS3_34 from 1024)** | See M1. |
| "100% CUDA-graph capturable" (module docstring, `:33-38`) | **Caveat** | See L1. |

### SS3 vs PS3_55 (`cuda_graph_ms` medians from the committed JSON; speedup = PS3 / SS3)

| Shape | PS3_55 ms | SS3_34 | SS3_35 | SS3_53 | SS3_54 | PE triton (bf16) |
|---|---|---|---|---|---|---|
| 512×2048 | 0.1253 | 0.990× | 0.950× | 0.927× | **0.900×** | 0.637× |
| 1024×4096 | 0.5188 | 1.026× | 0.985× | 0.983× | **0.945×** | 0.599× |
| 2048×8192 | 3.1087 | 1.077× | 1.040× | 1.036× | 1.005× | 0.546× |
| 4096×4096 | 10.5201 | 1.200× | 1.118× | 1.113× | 1.044× | 0.789× |
| 4×1024×4096 | 1.8828 | 1.085× | 1.046× | 1.040× | 1.010× | 0.571× |

The FLOP ratios predict this ordering. PS3_55 costs `(6α+10)n³`, SS3_54 `(6α+9)`, SS3_53 and SS3_35 `(6α+8)`, and SS3_34 `(6α+7)`. At 4096² (α=1), the predicted speedups are 1.067 / 1.143 / 1.231, and the measured ones are 1.044 / 1.113 / 1.200. The gap is about 2–3 points: 4 new reduction and fold kernels plus 2 pointer-coefficient squarings replace 1 `vector_norm` and 1 normalize kernel.

---

## Severity-Ranked Findings

### CRITICAL
None.

### HIGH
None.

### MEDIUM

**M1. The SS3 speedup is shape-dependent. At small shapes SS3 is slower than PS3_55 on every clock, and much slower on the eager wall clock.**

- **Evidence (committed JSON, `results/cycle2_cuda_graph_bench.json`):**
  - **512×2048, graph clock:** all SS3 circuits are slower than PS3_55 (0.900–0.990×).
  - **512×2048, wall clock:** SS3_34 takes 0.3564 ms, SS3_53 0.3730 ms and SS3_54 0.3846 ms, against 0.2941 ms for PS3_55. That is 18–24% slower.
  - **1024×4096:** SS3_53, SS3_54 and SS3_35 are below parity on the graph clock.
  - **Large shapes:** SS3_54, the first registry entry and so `DEFAULT_VARIANT` (`schatten_switch_blackwell.py:254`), gains only 0.5–4.4%.
- **Source of the small-shape wall overhead:** it is host enqueue cost, not GPU time. I measured enqueue time without a sync at 512×2048 (200 calls, median):
  - PS3_55: **0.292 ms**
  - SS3_34: **0.349 ms** (+57 µs)
  - SS3_54: **0.383 ms** (+91 µs)
- **cProfile breakdown:** SS3_54 makes 15 Triton launches per call. PS3_55 makes 13 Triton launches plus 1 `vector_norm`. Each `_launch` costs about 18–25 µs of host time: key building over up to 21 args with `isinstance`/`data_ptr` (`polar_switch_blackwell.py:124-127`) plus Triton's `launch_metadata`/`runner`. SS3 also adds 4 extra `torch.empty` calls (`:648-650,671`) and the `out_coeffs[j]` view indexing (`:693`).
- **Impact:** under CUDA-graph replay the host cost disappears, and the graph clock is at parity (SS3_34) or 5–10% slower (SS3_53/54) at 512×2048. In an eager Muon step, which is how `tier2_candidates.py` wires these circuits, small layers pay 57–91 µs per matrix.
- **Recommendation:**
  - Report SS3 speedups per shape class.
  - Call SS3_34 the only variant at or above parity for n ≥ 1024.
  - Do not cite SS3_54 as faster below n = 2048.
- **Independent spot-check (`bench_cuda_graph.run`, 5 methods, 10 blocks × 3 iters):** the GPU was **shared with Auditor 2** (utilization 100%, 6.5 GB used at the end). Load was 3.35 → 3.61. SM clock was 2857 MHz and temperature 56 °C.
  - Absolute times are inflated, for example PS3_55 at 21.9 ms on 4096² against 10.5 ms in the JSON.
  - Graph-clock ratios still matched the JSON closely. At 512×2048, SS3_34 was 1.000× and SS3_54 0.905×. At 1024×4096, SS3_34 was 1.028× and SS3_54 0.948×. At 4096², SS3_34 was 1.224×, SS3_53 1.177× and SS3_54 1.023×.
  - All capture errors were None and all graph-vs-eager values were 0.0.

**M2. fp64 inputs whose values exceed the fp32 range silently return zeros, but the reference returns a valid polar factor.**

- **Cause:** `X32 = G.reshape(...).float()` (`schatten_switch_blackwell.py:638`) turns finite fp64 values above about 3.4e38 into `inf`. `_safe_norm_part_kernel` then marks the matrix non-finite (`:399`), and the output is all zeros. `schatten_switch_reference` normalizes in fp64 (`:765-767`), finds the matrix valid, and returns the polar factor.
- **Reproduction:** `G = torch.randn(64,128,device="cuda").double()*1e300`. `schatten_switch_blackwell(G).any()` gives `False`. `schatten_switch_reference(G, dtype=torch.float16).any()` gives `True`.
- **Impact:** in practice this is low, because Muon momenta never reach 1e38. It is still a silent contract break, because the docstring says only non-finite or tiny inputs give zeros. `test_input_dtype_is_restored_and_input_is_not_modified` covers fp64 only at unit scale.
- **Fix:** pre-scale fp64 inputs by `amax` before the cast. Otherwise, document that the native path treats `|x| > FLT_MAX` as non-finite and make the reference match.

### LOW

**L1. "100% CUDA-graph capturable" needs a per-circuit, per-device warm-up, and the `sigma_hat` path cannot be captured.**

- **Reproduced:**
  1. I warmed shape (700,1500) with SS3_54, removed SS3_53's `_TABLE_CACHE` entry and captured SS3_53. Capture failed with `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned`. The failing call is the pageable `torch.tensor(..., device=device)` at `schatten_switch_blackwell.py:373-374`.
  2. A shape that was never warmed fails with `cudaErrorStreamCaptureInvalidated`. The autotuner benchmarks during capture, which is expected Triton behavior.
  3. The `sigma_hat` hook calls `.tolist()` on a tensor, which syncs (`:313`). It then makes a host-to-device `torch.tensor` (`:652`).
- **Impact:** none for the bench or the tests. Both warm the same function first (`bench_cuda_graph.py:96-100`, `tests/test_schatten_switch.py:461-468`).
- **Suggested fix:**
  - Qualify the docstring as "capturable after one eager warm-up call per (circuit, shape, device)".
  - Optionally raise a clear error when `torch.cuda.is_current_stream_capturing()` is true and a cache misses.

**L2. The reduction width is hard-coded as 64 in 7 kernel sites, separate from `_REDUCE_BLOCKS`.**

- **Where:** `schatten_switch_blackwell.py:417,433,436-438,484,500,503-504` use the literal `64` (`matrix * 64`, `tl.arange(0, 64)`). Line 67 sets `_REDUCE_BLOCKS = 64`.
- **Risk:** if someone raises `_REDUCE_BLOCKS` and forgets these sites, partial sums are silently truncated or overlap. The value is correct today.
- **Fix:** pass `_REDUCE_BLOCKS` as a `tl.constexpr`.

**L3. A test comment is stale.**

- **Where:** `tests/test_schatten_switch.py:447-448` says the small-shape slowdown comes from "the one host read of the scale".
- **Why it is wrong:** the path now does no host reads. The cause is host launch overhead (see M1).

**L4. The bench counterbalances inside each clock pass but not across clocks.**

- **What happens:** the passes always run in the order graph → eager event → wall (`bench_cuda_graph.py:120-156`). GPU temperature rose from 48 °C to 67 °C over the run. SM clock stayed at 2812–2850 MHz, so the effect is small.
- **Why it matters little:** comparisons across clocks are not the bench's purpose, and the order of methods within each clock is fully balanced.
- **Odd result:** `muon_pe_eager` is slower under the graph clock than under eager events at 2048×8192 (7.938 vs 7.862 ms) and 4×1024×4096 (4.362 vs 4.301 ms). That likely comes from cuBLAS picking a different algorithm or workspace inside the graph. It does not affect SS3 or PS3.

**L5. `git_dirty` ignores untracked files.**

- **Where:** `harness_tier2_real.py:483` runs `git status --porcelain --untracked-files=no`.
- **Risk:** an untracked module loaded through `--extra` would not mark the run dirty. That does not apply to this run, because `tier2_candidates.py` is tracked and identical at both SHAs.

**L6. Batched results are not bitwise equal to per-matrix results for tall batched shapes. The cause is cuBLAS, and PS3_55 shares it.**

- **Measured:** at `(5,33,17)`, member 0 of the batched call differs from the solo call by up to 6.5e-4 for SS3_54 and 8.6e-4 for PS3_55.
- **Root cause:** `torch.bmm` with batch 5 vs batch 1 is itself not bitwise equal on the same FP16 operands, with FP32 output. The Triton kernels are per-matrix independent.
- **Scope:** this is information only. The repo test `test_batched_result_equals_per_matrix_results` passes with its tolerance.

**L7. Some kernel branches are reached only by custom circuits.**

- **Which branches:** none of the 4 shipped circuits has `b0 == 0` or a non-zero segment-1 residual. So the `has_b0 = 0` path (`:569`), the pure-square row-0 grid (`:685`) and the residual branch (`:699-702`) run only for custom circuits.
- **What I checked:** I tested all three with bounded custom circuits (results above). All are bitwise stable and within about 2e-4 of the FP16 emulation reference.
- **Suggestion:** add one such custom-circuit test so the branches stay covered.

---

## Exact Checks Run and Raw Outputs

Every command ran from the repo root with `PYTHONPATH=. .venv/bin/python`.

1. **Static review.** I read all 803 lines of `schatten_switch_blackwell.py`, `polar_switch_blackwell.py:1-330`, `_store_sym` and `_tri_tile` in `kernels_blackwell.py`, `bench_cuda_graph.py`, `tier2_candidates.py`, the relevant tests, and Triton 3.8 `CompiledKernel.__getitem__`. The runner uses the current stream, so capture on the capture stream is correct. `git diff --stat 9a23a61 877c2a8` lists only 7 `results/*.json` files.
2. **Graph/adversarial matrix.** 4 circuits × shapes {(512,2048), (2048,512), (1024,1024), (97,331), (331,97), (1,1), (1,7), (7,1), (4,1,9), (3,7,1), (5,33,17), (3,1025,1100), (4,256,512)}. For each: warm up eager and on a side stream, capture, replay and compare bitwise with eager. Then overwrite the static input with {randn, zeros, 1e37·randn, 1e-38·randn, 1e-5·randn, entry = 3e38, a single NaN, +Inf, −Inf, mixed `[1,1e-30,1e25,0]`, a NaN in member 1}, replay 3 times, and compare bitwise with eager. Load went from 2.12 to 7.95 during the run because Auditor 2 was active. Raw result: `graph/adversarial failures: 4`. All 4 were `member0 != solo` at (5,33,17), with a difference of 5.96e-08: the cuBLAS batch effect in L6. There were zero graph≠eager results and zero zero-pattern violations. A first pass had wrong harness expectations: I had expected zeros for a whole batch when one member had a NaN, and non-zero output for 1e-20·randn, which is below `FRO_FLOOR`. I corrected those expectations; they were not code defects.
3. **Syncs.**
   ```
   sync ('SS3_54', (512, 2048)) ('no-sync', 0, 2, [])   # (sync_debug_mode, D2H count, sync events, memcpy names)
   ... same for all 12 (circuit, shape) pairs ...
   baseline sync events for one torch.cuda.synchronize(): 2
   ```
4. **Cold capture.**
   ```
   cold capture fails as expected: AcceleratorError ... cudaErrorStreamCaptureInvalidated
   warm-shape/cold-table capture FAILS: RuntimeError Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned.
   ```
5. **`_PLANS` matrix.** All 50 combinations of shipped variant/option and shape gave `bitwise_first_cached_cleared=True`. The reference comparison gave NaN in two cases: `sigma_hat=0.5` on vectors, where `sigma_max/||G||_F = 1 > 0.5`, which the docstring marks unsafe; and my first unbounded custom circuits, which produced fp16 overflow in both native and reference. The bounded custom circuits PURE2, CONLY2 and BC2 then gave `finite=True bitwise=True rel_vs_emul≈2e-4`.
6. **Edge cases.**
   ```
   fp64 1e300: native any nonzero: False  reference any nonzero: True
   unaligned ptr%16: 4 ; unaligned vs aligned bitwise: True
   compile None bitwise eq eager: True ; compile reduce-overhead bitwise eq eager: True
   batched-vs-solo tall SS3_54 bitwise: False 6.5e-04 ; PS3_55 bitwise: False 8.6e-04 ; cuBLAS bmm batch5 vs batch1 bitwise: False
   host enqueue PS3_55: median 0.2921 ms ; SS3_34: 0.3493 ms ; SS3_54: 0.3833 ms   (load 2.78-3.28)
   cProfile 200x SS3_54: 3000 _launch calls (15/call), cumtime 0.079 s of 0.118 s
   ```
7. **JSON analysis.** I recomputed medians from `samples_ms`, counted Williams position and adjacency balance (`positions {2}`, `adjacent pairs {2}`, 110/110), and checked env, capture errors and graph-vs-eager. All passed, as reported above.
8. **Latency spot-check on a shared GPU.** `bench_cuda_graph.run(["muon_pe_triton","muon_ps3_blackwell","ss3_34","ss3_53","ss3_54"], shapes=[(512,2048),(1024,4096),(4096,4096)], blocks=10, iters=3)`. Pre-run state was load 3.35, GPU 1%, 2887 MHz, 42 °C. Post-run state was load 3.61, GPU 100%, 2857 MHz, 56 °C, 6547 MiB. The ratios are in M1. Results were not written to `results/`.
9. **Repo tests.** `pytest tests/test_schatten_switch.py -k "cuda_graph or fable_r3 or min_dim or torch_compile or noncontiguous or first_call or empty_and_invalid or batched_result"` gave `50 passed, 92 deselected in 50.71s`.
