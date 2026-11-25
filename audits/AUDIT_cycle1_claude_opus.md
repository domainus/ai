# Cycle 1 Audit — Adversarial Auditor 1 (Claude Opus 5.5)

**Scope:** Blackwell `sm_120` Triton kernels (`kernels_blackwell.py`, `polar_switch_blackwell.py`), FP8/BF16 numerical precision and stability, Tier-1 and Tier-2 claims.
**Commit audited:** `9db80e7` (working tree, no source edits made).
**Hardware:** RTX 5080 (sm_120, 84 SMs), torch 2.14.1+cu130, Triton 3.8.0.
**Method:** Code reading plus counterexample scripts run with `PYTHONPATH=. .venv/bin/python`. Load note: other auditors shared the host. The 1-minute loadavg was 2.8 at the start of my runs and 11.9 during the timing run (E6). E6 compares eager-launch timing with CUDA-graph replay, so host load is part of what it measures.

---

## 1. Executive Verdict: **PASS_WITH_CAVEATS**

- **PS2_55 / PS3_55 Blackwell FP16 (`polar_switch_blackwell`, `dtype=float16`, `fp8_sketch=False`): PASS.** The results match eager `polar_switch` to rounding on all 117 Tier-1 cases. Large-shape speedups (≥1024×4096) hold up under CUDA-graph replay.
- **`polar_express_fp8` (default `fp8_steps=2`): FAIL.** It diverges on *benign* Gaussian and uniform-spectrum inputs with n ≤ 256 (σ_max up to 4.4×10⁴). Quarantine it before Cycle 2.
- **`fp8_sketch=True`: not promotable.** It is bounded (no divergence), but it costs +18–23% median Tier-1 error. At 1024×4096 it gives no GPU-time gain. It was never run through the head-to-head Tier-1 registry.
- **`_launch` plan cache:** one confirmed process-lifetime poisoning bug (torch.compile) and one confirmed shape crash (`min(rows, cols) == 1`).
- **Small-shape "gpu" speedups are inflated by host launch overhead.** At 512×2048 the reported 2.75–3.08× falls to 1.41× when measured on a CUDA graph.

---

## 2. Verified Strengths

| # | Claim | Evidence |
|---|---|---|
| S1 | PS2/PS3 Blackwell FP16 match eager to rounding | E2 (117 Tier-1 cases incl. 45 harvested): PS2 median err 0.0813 vs 0.0813, max 0.9981 vs 0.9981, σ_max 1.1081 vs 1.1084. PS3 0.0641 vs 0.0641. Worst per-case error increase vs eager: **+8×10⁻⁵** (PS2), **+7×10⁻⁵** (PS3). Agrees with `cycle1_head_to_head.json` `tier1_summary.all` (`ps2_55_blackwell_fp16` median 0.08134, σ_max 1.1081). |
| S2 | `triton_syrk` is bitwise symmetric and correct for odd/strided shapes | E3: shapes (1,5), (37,53), (130,70), (3,257,129): `torch.equal(A, A.mT)` True. Error vs fp64 ≤ 2.4×10⁻⁴ (FP16 level), also for a stride-2 K slice and a transposed view. JSON `blackwell_kernels_cycle1.json` `gram`: speedup 1.13× (512×2048) to 1.90–2.02× (4096², 4×1024×4096) over cuBLAS. Error matches cuBLAS. |
| S3 | `_symprod_kernel` output may alias C | Code: program (i,j), i ≤ j, reads C[i,j] once and then writes O[i,j] and O[j,i]. No program reads a lower tile. `restore_value=["o_ptr"]` (`polar_switch_blackwell.py:48`) also restores C during autotune trials when out aliases C. No Polar Switch row has `dst == src` (checked against both tables in `baselines.py:96–128`). |
| S4 | Pointer-alignment specialization handled | Triton specializes pointers only on 16-byte divisibility. The `_launch` key holds `data_ptr() % 16` plus exact int values, which is stricter than Triton's own key. E4: a misaligned input (`data_ptr % 16 == 4`) gives a relative difference of **0.0** against an aligned clone. |
| S5 | Stream / CUDA-graph compatibility (warm) | `CompiledKernel.__getitem__` launches on `driver.active.get_current_stream()`, which is torch's current stream. E3: side-stream output is bitwise equal. After warmup, CUDA-graph capture and replay of PS2 FP16, PS2 sketch and Gram-NS gives **0.0** relative difference against an eager call. |
| S6 | `fp8_sketch` trace rescale bounds the spectrum | E2: σ_max ≤ 1.1081 (fp16) / 1.1154 (bf16) on all 117 cases. 0 divergences, 0 non-finite. Edge cases same as eager: zeros → 0, scale 1e-20 → 0, one 1e6× outlier entry → rel diff 1.5×10⁻³, NaN input → NaN. A shared amax across the batch is harmless: a flat matrix next to a spiky neighbour gives err 0.0763 vs 0.0758 alone. |
| S7 | Large-shape speedups are real GPU time | E6: at 1024×4096, CUDA-event time ≈ graph replay (PS2 bw 0.510 vs 0.508 ms; eager 0.790 vs 0.766). 4096² is consistent across both result files: PS2 bw 10.19 ms vs eager 19.82 (`polar_switch_blackwell_cycle1.json`), 10.17 vs 19.69 (`cycle1_head_to_head.json`). That is **1.94×**. |
| S8 | Tier-2 Blackwell PS is stable and faster in training | `cycle1_tier2_lm.json`: 16 PS2/PS3-Blackwell runs, 0 diverged. Throughput: PS2 bw 160.8k–164.7k tok/s vs PS2 eager 144.3k–151.5k (≈ +9%). |

---

## 3. Findings (severity-ranked)

### CRITICAL

#### C1. `polar_express_fp8` (default `fp8_steps=2`) diverges on benign inputs; `on_misaligned="pad"` diverges too
- **Where:** `kernels_blackwell.py:333–386` (`_fp8_gram_step` at 313–330).
- **Evidence (E1):** Reproduces exactly the **11/117** Tier-1 divergences (`cycle1_head_to_head.json` `polar_express_fp8x2_bf16.divergence_rate = 0.094`). All 11 are *easy* cases, not adversarial ones:

  | Case | fp8x2 bf16 σ_max | fp8x2 fp16 σ_max | fp8x1 σ_max | Triton bf16 σ_max |
  |---|---|---|---|---|
  | uniform_singular_values/128×128 | **44 200** | 114 600 | 1.124 | 1.124 |
  | gaussian_mp/128×128 | 6 207 | 12 920 | 1.124 | 1.124 |
  | gaussian_mp/128×256 | 150.7 | 363.9 | 1.124 | 1.123 |
  | uniform_singular_values/256×512 | 51.9 | 13.1 | 1.124 | 1.124 |
  | (7 more Gaussian/uniform, n ≤ 256) | 3.4 – 226 | | 1.124 | 1.124 |

  Extra repros: Gaussian 112×304 → σ_max 339; 128×384 → 8.2; 512×2048 → 1.125 (stable). `on_misaligned="pad"` on Gaussian 100×300 (padded to 112×304) → σ_max **3 572**, although its docstring calls padding "exact for the polar factor".
- **Root cause (per-step σ_max, uniform 128×128):** FP8 `[0.136, 1.062, 2.023, 2.326, 10.29, 44 205]` vs BF16 `[0.136, 1.062, 1.967, 1.857, 1.56, 1.124]`. FP8 error in the second FP8 step (step 1) pushes σ_max past the PE design upper bound u₁ = 2 − l₁ ≈ 1.97. Outside its interval each odd quintic expands, and the cascade runs 2.02 → 2.33 → 10.3 → 4.4×10⁴. The FP16 late path diverges as well, so BF16 is not the cause. The cause is ~2⁻⁴ e4m3 relative error, amplified by the cancellation in `AA + (b/c)·A`: `b/c = −5.4` at step 1, and the result is scaled back by `c`. Small n is hit because Frobenius normalization leaves σ_i ~ 1/√n. Many singular values then cluster at the top of the step-1 image.
- **Why the kernel bench missed it:** `blackwell_kernels_cycle1.json` only uses n ≥ 512.
- **Also not a speed win where it is stable:** 512×2048: 0.970 ms vs 0.349 ms Triton fused (0.36×). 4×1024×4096: 3.63 vs 3.31 ms, because the Python loop over the batch (`kernels_blackwell.py:379`) runs one `_scaled_mm` per matrix.

### HIGH

#### H1. `_launch` plan cache is permanently poisoned by `torch.compile`
- **Where:** `polar_switch_blackwell.py:100–119`.
- **Repro (E5), warm process:**
  ```python
  psb.polar_switch_blackwell(torch.randn(256,1024,device="cuda"))        # warm kernels
  G = torch.randn(128,512,device="cuda")
  torch.compile(lambda x: psb.polar_switch_blackwell(x,"PS2_55"))(G)     # TypeError
  torch._dynamo.reset()
  psb.polar_switch_blackwell(G,"PS2_55")    # plain EAGER call -> TypeError: 'NoneType' object is not subscriptable
  ```
  After the compile attempt, `_PLANS` holds **3 entries with `compiled=None`**. Dynamo's user-Triton HOP returns `None` from `kernel[grid](...)`. `kernel.best_config` is a stale attribute from a *different* shape. Every later eager call for that shape fails for the life of the process. In a cold process, compile fails with `AttributeError: 'Autotuner' object has no attribute 'best_config'`.
- **Impact:** Users often `torch.compile` an optimizer step. One attempt then breaks the optimizer for the rest of the process, including after they fall back to eager.

#### H2. `min(rows, columns) == 1` crashes `polar_switch_blackwell` and `gram_ns_blackwell`
- **Where:** `_symprod_kernel` → `_tri_tile(tl.program_id(0), tl.cdiv(N, BLOCK))` (`polar_switch_blackwell.py:61`, `kernels_blackwell.py:75`).
- **Repro (E3/E4):** shapes (1,1), (1,7), (7,1), (64,1) → `CompilationError`, root cause `AttributeError: 'int' object has no attribute 'to'`. Triton specializes `N == 1` to a constexpr, so `T` becomes a Python int. Eager `polar_switch` / `gram_ns` handle these shapes. `triton_syrk` and `polar_express_triton` with n = 1 do run.
- **Contradicts:** module docstring `polar_switch_blackwell.py:14` ("Tall, wide, square, and odd sizes are supported").

### MEDIUM

#### M1. `fp8_sketch=True` costs accuracy, and its soundness argument is incomplete
- **Evidence (E2, 117 Tier-1 cases, FP16 late):** median error PS2 0.0813 → **0.0956 (+18%)**, PS3 0.0641 → **0.0790 (+23%)**. Worst per-case increase: +0.050 (PS2, `geometric_kappa_10000/128x128`), +0.046 (PS3, `gamma_1.5/128x128`). The sketch therefore removes most of PS3's 0.017 accuracy lead over PS2. Gram-NS is unaffected (+0.0001 median), because its restart recomputes R exactly.
- **Low-rank counterexample (E3):** non-integer rank-1 `u vᵀ` input: relative deviation from eager **0.40** (fp16) and **1.61** (bf16). Rank-2 (1e-3 separation): 0.46 / 1.28. σ_max stays bounded, so a divergence check does not see this.
- **Why:** The docstring (`polar_switch_blackwell.py:155–159`) argues that the sketch "adds no noise to the row space of X". That is true. But `Q = poly(A_sketch)` does not commute with the true `XXᵀ`, so `polar(QX) ≠ polar(X)`, and phase 1 cannot undo this. Quantization noise creates near-zero eigen-directions in `A_sketch`. Q applies its large small-λ gain to them and rotates the left singular vectors.
- **Benefit is small:** E6 graph replay at 1024×4096: sketch 0.5079 ms = non-sketch 0.5079 ms. The bench shows −3.7% at 2048×8192 and is *slower* at 512×2048 (0.197 vs 0.128 ms GPU, 0.141 vs 0.125 ms graph).
- **Coverage gap:** `run_cycle1_head_to_head.py:270–290` never registers a sketch variant. Its only validation is the 18-case `spectrum_suite` (n = 512) in `polar_switch_blackwell_cycle1.json`.

#### M2. The "gpu" clock is host-launch-bound at small shapes; small-shape speedups are inflated
- **Claim at fault:** `polar_switch_blackwell.py:390` ("A busy host inflates the wall clock. The GPU clock is not affected.") and `:315–317`.
- **Evidence:** The same method and shape on the same day give different results. PS2 bw 512×2048 is **0.128 ms** (IQR 0.002) in `polar_switch_blackwell_cycle1.json` (load 2.8, 4×L2 flush) and **0.206 ms** (IQR 0.024) in `cycle1_head_to_head.json` (load 5.26 at the end, 2×L2 flush). PS2 eager: 0.394 vs 0.568.
- **E6 (CUDA-event time vs CUDA-graph replay, 2×L2 flush):**

  | 512×2048 | events | graph replay |
  |---|---|---|
  | PS2 bw fp16 | 0.323 | **0.125** |
  | PS2 bw sketch | 0.591 | 0.141 |
  | PS2 eager | 0.755 | **0.176** |

  The true GPU speedup of PS2 bw over PS2 eager at 512×2048 is **1.41×**, not the reported 3.08× / 2.75×. The extra gain comes from fewer Python launches. That is still useful in eager mode, but it is a host-side gain and must not be labelled "gpu". Host enqueue for PS2 bw is ~250 µs per call (E4, 64×256), which exceeds the ~130 µs head start a 128 MB flush gives.
- **Also:** `cycle1_head_to_head.json` records only an end-of-run `loadavg` (`[5.26, 3.76, 4.05]`), with no start value and nothing per shape. `cycle1_tier2_lm.json` records no load, timestamp or git SHA at all.

#### M3. Tier-2 comparisons among Muon variants are within noise
- **Noise floor:** PS2 eager and PS2 Blackwell do the same arithmetic up to rounding (S1). Yet their token-budget validation losses differ per seed and LR by up to 0.010 (e.g. lr 0.02: eager 3.0458 / 3.0420 vs bw 3.0358 / 3.0429). The whole token-budget Muon spread except S0 is 3.0316–3.0439, i.e. 0.012. That is within this floor, with n = 2 seeds.
- **Budgets:** 409 600 tokens = 200 steps, and 3-second wall budgets. Loss is still falling steeply there, so a +9% throughput gain becomes a large wall-budget loss gap (2.875 vs 2.944). That gap will not transfer to real-length runs.
- **LR grid:** `{0.01, 0.02}` for Muon. The best LR is at the grid edge (0.02) in 7/8 token-budget Muon rows, and AdamW's best LR (0.003) is also at its edge. The optimum is not bracketed.

#### M4. `polar_express_triton` and eager `polar_express` are different circuits
- `kernels_blackwell.POLAR_EXPRESS_COEFFS[0] = (8.20516, −22.90193, 16.46072)` (raw paper coefficients / 1.01). `baselines._AMSEL_COEFFICIENTS[0] = (8.23731, −23.15775, 16.68057)` (Remez recompute, cushion 0.02). All five steps differ. Normalization also differs: FP32 in Triton, BF16 in eager.
- Consequence: `speedup_vs_own_eager` and the accuracy deltas (0.0876 vs 0.0942 at 512×2048) compare two different algorithms. The FLOP counts are equal, so timing is roughly comparable. Accuracy is not.

### LOW

| # | Finding | Evidence |
|---|---|---|
| L1 | **CUDA-graph capture of an unseen shape fails.** Autotune runs inside capture → `cudaErrorStreamCaptureInvalidated`. After the failed capture, eager recovers and the result is correct (rel 2.1×10⁻³ vs eager, which is rounding level). Needs a documented "warm every shape first" rule. | E3/E4 |
| L2 | **`_launch` is not thread-safe.** It reads the shared `kernel.best_config` after the launch. Two threads on different shapes can pair one key's CompiledKernel with another key's config. `grid(config)` then uses the wrong BLOCK, so tiles are missing in a `torch.empty` output. *Code inspection; not reproduced.* `_PLANS` also grows without bound under dynamic shapes, and each new (N, K) runs an 11-config autotune. | `polar_switch_blackwell.py:113–116` |
| L3 | **Sketch turns FP32 norm overflow into NaN.** For ‖G‖_F > ~1.8×10¹⁹, eager and non-sketch return silent zeros, but the sketch returns NaN: `energy = (inf/(inf+eps))²`. Both behaviours are wrong; NaN also poisons optimizer state. | E3 "huge scale 1e25" |
| L4 | **int32 index arithmetic.** `rm[:, None] * sxn` and `rm[:, None] * N` in `_syrk/_poly/_symprod_kernel` overflow when N·K or N² ≥ 2³¹ (e.g. a 46 341² FP16 Gram is 4.3 GB and fits in 16 GB). *Code inspection.* | `kernels_blackwell.py:119,148`; `polar_switch_blackwell.py:65` |
| L5 | **`triton_gram_poly_step` gives silently wrong results for non-symmetric A** (rel err 0.98). This is a documented precondition of a public function, but it is not checked. | E3 |
| L6 | **`_apply` writes FP32 directly in the last phase** (Q3). Eager rounds to FP16 and then casts. Effect: one fewer FP16 rounding (~2.4×10⁻⁴ RMS relative) and one fewer elementwise pass (~1% at 4096²). This is not a material unfairness: accuracy matches to 10⁻⁶ in the bench. Disclose it. | `polar_switch_blackwell.py:184–195` vs `baselines.py:245–246` |
| L7 | **Gram-NS BF16 is unbounded,** as its docstring says: σ_max 1.68 / 1.74 in the bench and **1.71** in my 117-case sweep (median err 0.127 vs 0.096 in FP16). The bench still prints its speedups next to the safe variants without a flag. Gram-NS FP16 Blackwell σ_max is 1.156 vs eager 1.143, slightly looser than "up to rounding". | E2; `polar_switch_blackwell_cycle1.json` `spectrum_suite_sigma_max` |
| L8 | **Same `data_ptr()` reuse hazard outside my scope.** `_make_spectral_wrapper` keys `prewarmed_cache` on `previous.data_ptr()` (`run_cycle1_head_to_head.py:113`). A freed and reallocated tensor of the same shape returns a stale warmed scaler. *Code inspection; out of scope.* | |

---

## 4. Answers to the Brief's Questions

1. **`_launch` cache.** The key is correct for alignment and int specialization (S4). Under warm CUDA Graphs and side streams it is correct (S5). It breaks under `torch.compile` with persistent poisoning (H1) and for n = 1 (H2). Cold capture fails (L1). It is racy under threads (L2). The leak is small, but each new shape adds an entry and an autotune.
2. **`polar_express_fp8` exploded** because FP8 error at step 1 pushes σ past the PE interval bound, where the quintics expand (C1). **`fp8_sketch`:** `trace ≤ 0` and `amax == 0` are handled (zero input → 0). Rank-deficient input stays bounded, but its error is O(1) (M1). Norm overflow gives NaN (L3).
3. **PS2/PS3 Blackwell FP16 equivalence** is verified (S1). Writing FP32 directly is a disclosed micro-advantage, not a material unfairness (L6).
4. **Gram-NS BF16** is unbounded, as documented (L7). **Timing integrity:** large-shape numbers are sound (S7). Small-shape "gpu" numbers are host-bound and depend on load (M2). Tier-2 stores no load, timestamp or SHA, and its rankings are within noise (M3).

---

## 5. Required Remediation Before Cycle 2

1. **C1:** Set `fp8_steps` to default 1, or remove the FP8-early path. Delete or fix `on_misaligned="pad"`. Add a release gate: Gaussian and uniform spectra at n ∈ {64, 128, 256}, aspect 1–8, with σ_max < 1.2. Optionally clamp σ after each FP8 step, for example by renormalizing the iterate when its Frobenius bound exceeds `u_k`.
2. **H1:** In `_launch`, decorate the public entry points with `@torch.compiler.disable`, or bypass the cache when `torch.compiler.is_compiling()`. Cache only `isinstance(compiled, CompiledKernel)`. Read the config from `kernel.cache[autotune_key]`, not from the shared `best_config`; this also fixes L2.
3. **H2:** Add `do_not_specialize=["N"]` (or handle `n == 1` in Python with a scalar path). Add n ∈ {1, 2, 15, 17} shapes to the unit tests.
4. **M1:** Do not promote `fp8_sketch`. If kept, register it in the Tier-1 head-to-head, include rank-1/rank-2 non-integer cases, and correct the "sound" docstring.
5. **M2:** Report GPU time from CUDA-graph replay or an nsys kernel sum, and report wall time separately. Record loadavg at start, per shape and at end. Use one flush size (4×L2) across all harnesses. Restate the 512×2048 speedups.
6. **M3:** Tier-2: ≥ 5 seeds, an LR grid that brackets the optimum (e.g. {0.005, 0.01, 0.02, 0.04}), budgets ≥ 10× longer, and confidence intervals. Record a git SHA, timestamp and loadavg in the JSON.
7. **M4:** Make the Triton PE use `_AMSEL_COEFFICIENTS` and the eager normalization, or label it as a different circuit.
8. **L1, L3–L7:** Document the warmup-before-capture rule, guard the overflow (compute the norm with scaling, or use FP64 for the norm), use int64 offsets, add a symmetry check under a debug flag, and flag Gram-NS BF16 as unsafe in the bench output.

---

## Appendix — Reproduction Scripts

The scripts lived in the session scratchpad (`e1_fp8.py`, `e2_sketch.py`, `e3_edges.py`/`e3b.py`, `e4_poison.py`, `e5.py`, `e6_timing.py`). Each finding above includes the minimal snippet or inputs needed to reproduce it. Key one-liners:

```bash
# C1  (seeds 0/1/2 -> sigma_max 323 / 4291 / 1986)
PYTHONPATH=. .venv/bin/python -c "import torch;torch.manual_seed(0);from research.polar_opt.kernels_blackwell import polar_express_fp8 as f;G=torch.randn(112,304,device='cuda');print(torch.linalg.svdvals(f(G,5,2,dtype=torch.bfloat16).double()).max())"
# H2
PYTHONPATH=. .venv/bin/python -c "import torch;from research.polar_opt.polar_switch_blackwell import polar_switch_blackwell as f;f(torch.randn(1,64,device='cuda'))"
```
