# domainus/ai — Autonomous Verifiable AI Research Lab

An open, continuously operating autonomous AI research repository investigating **verifiable, programmatically grounded machine learning systems and numerical algorithms** across a multi-machine fleet (`bazzite` NVIDIA RTX 5080 Blackwell `sm_120`, `Ryan's MacBook Pro` Apple M5 Max MLX/Metal 4), built with cross-model engineering and mandatory 4-model adversarial review (**Claude Opus 5.5**, **Claude Fable 5.2**, **GPT 6.2 Astra**, **Gemini 4 Neon** at `high`).

---

## Cycle 1 Executive Summary (Audited & P0-Remediated, 531/531 Tests Passing)

Cycle 1 benchmarked consumer Blackwell (`sm_120`) Triton kernels, Apple Silicon MLX/MPS kernels, warm-started spectral pre-scaling, and statistically rigorous **Schatten-8** upper-bound polar circuits directly against late-2026 SOTA orthogonalizers: **Polar Express** (`arXiv:2505.16932`), **Gram Newton-Schulz** (`Dao-AILab/gram-newton-schulz`), and **Polar Switch** (`PS2_55` / `PS3_55`, `arXiv:2610.26742`).

### What Is Verified vs. Caveated

1. **Verified Systems Engineering Win (`ps2_55_blackwell_fp16` / `ps3_55_blackwell_fp16`, `fp8_sketch=False`):**
   - Executes Du & Su's exact `PS2_55` / `PS3_55` circuits using autotuned consumer Blackwell (`sm_120`) Triton `_syrk_kernel` + fused symmetric `_symprod_kernel` + plan-cached `_launch` (not new math).
   - **RTX 5080 (`sm_120`) Tier-1 (117 cases incl. 45 harvested LM momenta):** `0.304 ms` (`PS2_55`) / `0.307 ms` (`PS3_55`) = **`1.90×` / `1.89×`** faster than eager `PS2_55`/`PS3_55` (`0.578`/`0.581 ms`), **`1.41×`** faster than eager `PolarExpress` (`0.433 ms`), **`0/117` divergences**, and worst per-case polar error delta `<= 8e-5` vs eager `polar_switch`.
   - **RTX 5080 Large LLM Shapes (`512×2048 .. 4096×4096` & `(4, 1024, 4096)`):** `1.42–2.10×` wall-clock speedup vs eager `PS2_55` (`1.88–2.61×` vs eager `PolarExpress`). Under CUDA-graph replay isolating GPU execution from host Python launch overhead, `PS2_55` Blackwell delivers **`1.41×`** at `512×2048` and **`1.94×`** at `4096×4096` (post-audit `N==1`/index fix preserves `10.46 ms` at `4096×4096`, `1.89×` vs eager).
   - **Stand-alone `triton_syrk`:** `1.13–2.02×` vs cuBLAS across tested shapes (`1.63–2.02×` for `N >= 1024`) with bitwise symmetric outputs.
   - **Clean Tier-2 Like-for-Like Throughput (`19.2M` synthetic LM, 409.6k tokens, LR `0.02`, 2 seeds):** `muon_ps2_blackwell` improves mean end-to-end throughput by **`+9.4%` (`1.094×`)** over eager `muon_ps2_eager` with **`0/72`** divergences across all Tier-2 trials.

2. **Caveated Results (Not Overclaimed):**
   - **Tier-2 Wall-Clock Loss Headline (`-0.224` vs eager `PolarExpress`):** `muon_ps3_blackwell` reached `2.8669` val loss @ `162.4k tok/s` vs `muon_pe_eager` `3.0910` @ `127.2k tok/s` (`+27.7%` tok/s) and `muon_ps2_eager` `2.9440` @ `149.8k tok/s` (`+8.4%` tok/s, `-0.077` loss). Equal-token Muon losses were flat (`3.0316–3.0439`) with seed disagreement (`-0.017` vs `+0.013`), LR grid edge saturation (`0.02` won almost everywhere), uncompiled eager baselines, BF16 vs FP32 `_foreach_add_` dispatch differences, `PS3_55 eager` omitted, and a tiny `2,048` tok/step synthetic 8-token motif stream. Moving to `50M–120M` real text in Cycle 2 (`T3-01`).
   - **Stateless Schatten-8 Circuits (`circuits.py`):** `S1-k4+5-lo0.002_schatten8` achieves the lowest worst-case approximate harvested error (`0.2800` vs `PS3_55` `0.3208`), and `S0-k6-lo0.05` reaches `2.669 ms` at `2048×8192` (`2.98×` vs PE), but unfused FP32 buffer combinations make `circuits.py` **`3.8–4.5×` slower** than `_symprod_kernel`-fused `PS3_55` on Tier-1 (`1.185–1.378 ms` vs `0.307 ms`).
   - **Apple M5 Max MLX/MPS (`results/apple_silicon_cycle1.json`):** `15/15` finite on both MLX and MPS; MLX beat PyTorch MPS in 14/15 cases (`2.3–2.9×` at `>= 1024`). At `1024×4096` FP32, `PS3_55` MLX ran in `2.191 ms` (`0.0626` err) vs `PolarExpress` MLX `3.342 ms` (`0.0949` err) — **`1.52×`** faster and `34%` lower error — **but regressed `-14%` (`0.88×`) on square `2048×2048`** (`7.704 ms` vs `6.747 ms`). Caveat: FP32 only, `warmups=1, repeats=5`.

---

## Reconciled Cycle 1 Leaderboards

### RTX 5080 (`sm_120`) Tier-1 (117-case median ms; 45-case harvested LM error) & Tier-2 (19.2M Synthetic LM, 3s wall budget)

| Method | Status | Tier-1 Med (ms) | Harvested Med Err | Harvested Max Err | Tier-2 Wall Loss | Tier-2 Tok/s |
|---|---|---:|---:|---:|---:|---:|
| **`ps3_55_blackwell_fp16`** | `VERIFIED` | **0.307** | **0.0624** | 0.3208 | 2.8669* | 162,448* |
| **`ps2_55_blackwell_fp16`** | `VERIFIED` | **0.304** | 0.0800 | 0.3317 | 2.8753* | 163,194* |
| `PS3_55 (eager)` | `BASELINE` | 0.581 | 0.0624 | 0.3208 | — | — |
| `PS2_55 (eager)` | `BASELINE` | 0.578 | 0.0800 | 0.3318 | 2.9440 | 149,840 |
| `polar_express (eager)` | `BASELINE` | 0.433 | 0.0958 | 0.3484 | 3.0910 | 127,212 |
| `polar_express_triton_bf16` | `CAVEATED`† | 0.635 | 0.0893 | 0.3471 | 3.0266 | 140,778 |
| `S1-k3+4-lo0.005_schatten8` | `CAVEATED` | 1.185 | 0.0724 | 0.4310 | 3.1158 | 128,560 |
| `S1-k4+5-lo0.002_schatten8` | `CAVEATED` | 1.378 | 0.0629 | **0.2800** | 3.1312 | 121,789 |
| `polar_express_fp8x2_bf16` | `EXCLUDED` | — | — | — | — | — |

*\*Tier-2 wall loss is throughput-driven on a synthetic 2,048-tok/step task with caveats above. †Cycle 1 archive used raw PE/1.01 coefficients with FP32 norm (remediated in P0 to match `_AMSEL_COEFFICIENTS`). `polar_express_fp8x2_bf16` excluded due to `11/117` (`11/16` in `uniform_mp`) divergences (`σ_max` up to `44,205`).*

---

## Verified Negative & Safety Discoveries (And P0 Fixes Applied)

1. **`polar_express_fp8` (`fp8_steps=2`) diverges on benign small-`n` `uniform_mp` / Gaussian spectra (`n <= 256`):**
   - Failed **11/16 (`68.75%`)** `uniform_mp` cases (`σ_max` up to `44,205` BF16 / `114,600` FP16; **`0/32`** on `adversarial`), including with `on_misaligned="pad"`. Also `20–77%` slower than FP16 `PS3_55` on LLM shapes. **P0 fix:** Defaulted `fp8_steps=1` (0 divergences verified across `n in {64,128,256}`).
2. **Heuristic warm-power iteration suffers silent single-step failure on live LM training:**
   - `scaling="spectral"` hit a single-step Rayleigh drop at LM step 11 (`σ_p / σ_max = 0.713`, `ratio_growth` `23.7% < 25%` tol, `true / σ̂ = 1.2295`) without triggering any guard, while only beating free `schatten8` scale by `1.7%` mean on harvested pairs. **P0 fix:** Added certified `u_safe = 1.10` floor, changed `SpectralPrescaledPolar` default to `scaling="schatten8"`, cleared `_pending` on shape change, and isolated `block_krylov` NaNs per batch member.
3. **FP16 `schatten8` subnormal rounding on trace-1 `R` at `n >= 4096`:**
   - On coherent `p·I + c·11ᵀ` (`n >= 4096`), FP16 casting of unscaled trace-1 `R` truncated off-diagonal subnormals, corrupting `R2` and diverging. **P0 fix:** Exact power-of-two `_square` prescaling, explicit `‖R - R_h‖_F` cast error bound in `schatten8`, CPU FP64 norm reductions, and overflow/underflow-safe normalization eliminating subnormal `NaN`s.
4. **`fp8_sketch=True` non-commutation:**
   - Trace rescaling keeps `σ_max` bounded, but `poly(A_sketch)` does not commute with `X X^T`, raising median Tier-1 polar error by `+18–23%` (`40–161%` on rank-1) for `<= 3.7%` Amdahl speedup. Remains unpromoted.
5. **Coherent rank-1 BF16 matrix sweeps:**
   - Scalar BF16 rounding checks missed coherent rank-1 matrices (`σ_max` up to `2.313` in BF16). **P0 fix:** Restricted `BF16_SAFE_CIRCUITS = {"S0-k4-lo0.05", "S1-k2+4-lo0.01"}`.
6. **Infrastructure & benchmark integrity fixes in P0:**
   - Fixed `_make_spectral_wrapper` freed CUDA `data_ptr()` cache aliasing (`11.65%` diff bug), `torch.compile` poisoning of `_PLANS`, `min(rows, cols) == 1` Triton constexpr crash in `_tri_tile`, `fp8_sketch` `1e25` overflow `NaN`, aligned `POLAR_EXPRESS_COEFFS = list(_AMSEL_COEFFICIENTS)`, and updated `reports/generate_infographic.py` for Cycle 1 schemas.

---

## Independent Adversarial Audits (`audits/`)

- **Auditor 1 (Claude Opus 5.5, `high`):** [`AUDIT_cycle1_claude_opus.md`](audits/AUDIT_cycle1_claude_opus.md) — `PASS_WITH_CAVEATS` -> P0 remediated
- **Auditor 2 (Claude Fable 5.2, `high`):** [`AUDIT_cycle1_claude_fable.md`](audits/AUDIT_cycle1_claude_fable.md) — `FAIL` -> P0 remediated (verified R1–R5)
- **Auditor 3 (GPT 6.2 Astra, `high`):** [`AUDIT_cycle1_gpt62_astra.md`](audits/AUDIT_cycle1_gpt62_astra.md) — `FAIL` -> P0 remediated
- **Auditor 4 (Gemini 4 Neon, `high`):** [`AUDIT_cycle1_gemini_neon.md`](audits/AUDIT_cycle1_gemini_neon.md) — `PASS_WITH_CAVEATS` -> P0 remediated

---

## Reproduction Commands

```bash
# Run full verified suite (CPU + Blackwell sm_120 CUDA, 531 tests)
PYTHONPATH=. .venv/bin/pytest -q

# Regenerate daily Cycle 1 HTML infographic
PYTHONPATH=. .venv/bin/python reports/generate_infographic.py --date 2026-11-25

# Standalone Blackwell PolarSwitch benchmark & Apple Silicon MLX benchmark
PYTHONPATH=. .venv/bin/python research/polar_opt/polar_switch_blackwell.py --bench
PYTHONPATH=. .venv/bin/python research/polar_opt/kernels_mlx.py --bench --backend mlx
```

## Fleet & Cycle 2 Launch

- **Active fleet:** `bazzite` (RTX 5080 `sm_120`) and `Ryan's MacBook Pro` (Apple M5 Max, 128 GiB unified). Work previously slated for `Ryan's MacBook Air` has been consolidated onto `MacBook Pro` (MacBook Air retired from the active fleet; MacBook Pro benchmark runs automatically once its relay reconnects).
- **Cycle 2 priorities ([`state/BACKLOG.md`](state/BACKLOG.md), In Progress):**
  1. `T1-04`: `_symprod_kernel`-native 2/3-buffer Schatten-8 circuits with fewer squarings than `PS3_55` and verified segment-2 headroom (`u >= 1.05` FP16 / `>= 1.15` BF16).
  2. `T3-01`: Multi-seed LR-bracketed 50M–120M FineWeb-Edu/WikiText-103 LM pretraining on RTX 5080 with `PS3_55` eager, `torch.compile`, and matched contiguous FP32 update controls.
  3. `T2-02`: Apple M5 Max BF16/FP16 MLX suite + custom Metal triangular `SYRK` kernel.
