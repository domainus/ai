# domainus/ai — Autonomous Verifiable AI Research Lab

An open, continuously operating autonomous AI research repository investigating **verifiable, programmatically grounded machine learning systems and numerical algorithms** across a multi-machine fleet (`bazzite` NVIDIA RTX 5080 Blackwell `sm_120`, `Ryan's MacBook Pro` Apple M5 Max MLX/Metal 4), built with cross-model engineering and mandatory multi-model adversarial review (**Claude Opus 5.5**, **Claude Fable 5.2**, **GPT 6.2 Astra**, **Gemini 4 Neon** at `high`).

---

## Cycle 3 Executive Summary (Audited, Remediated & Merged — 1054 Passed, 6 Skipped on RTX 5080)

Cycle 3 merged and adversarially audited five tracks on `bazzite` (`C3-04` on `Ryan's MacBook Pro` remains queued while the laptop is unreachable):
1. **`C3-03` (Small-`n` Tile-Norm Launch Fusion & Fresh 117-Case Tier-1 Refresh):** Fused deterministic zero-atomic tile-norm accumulation into `_syrk_norm_kernel`/`_square_norm_kernel` plus a direct Triton 3.8 C launcher (`_launch`), cutting auxiliary kernels (`SS3_54` `17->16`, `SS3_53`/`SS3_35` `16->15`, `SS3_34` `15->14`, `0` D2H syncs). `SS3_53` beats `PS3_55_blackwell` on every measured shape (`1.034–1.132×`, up to `2.700×` vs `PE eager`); `SS3_54` beats `PS3_55_blackwell` at `n >= 1024` (`1.017–1.062×`) and wins eager wall at `(512, 2048)` (`0.2256` vs `0.2289 ms`).
2. **`C3-01` (Wide-Layer Full-Model Crossover & Short-Budget `120m` FineWeb-Edu Transfer):** `SS3_53` wins full 12-layer step orthogonalization across `w512..w2048` (`1.019–1.067×` vs `ps3_blackwell`), while `SS3_54` crosses `1.000×` at `w1024` and wins at `w1536` (`1.013×`) and `w2048` (`1.020×`). In 3-seed `120m` transfer (`256`-step / `100s` budgets), Blackwell switches gain `+2.90–3.00%` tok/s over `pe_compiled` (`Holm p <= 0.0018`); longer `>= 20M`-token runs remain queued for Cycle 4 (`ps3_blackwell` retained as conservative `120m` default).
3. **`C3-02` (Hybrid Switching, `SS3_35` BF16 & Early-Spectrum Search):** Falsified `PS3/SS3_54 -> SS3_34` post-warmup switching (`+0.0104`/`+0.0112` `50m` loss regression); `SS3_35` BF16 achieves `50m` parity (`+0.0026`, `0.9956×` tok/s); early-polish `SS3_54E0` was rejected for surrendering `53%` of `SS3_54`'s mid/late lead.
4. **`T4-01` (3-Seed $\mu$P Coordinate Check & Val LR Transfer, CPU):** SP fails (`max |beta| = 2.599` AdamW / `1.553` Muon) and `readout_lr_power=1` fails (`0.975–1.049`), while $\mu$P passes (`0.116` AdamW, `0.078–0.082` Muon across `svd_exact`, `PE`, `PS3_55`, `SS3_*`) with flat/near-flat val LR transfer.
5. **`T5-01` (Blackwell NVFP4 / MXFP4 / INT2 Micro-Rotated PTQ):** Native `F.scaled_mm` NVFP4 (`float4_e2m1fn_x2` + swizzled E4M3FN) verified on `sm_120` at **`511.8` / `757.5 TFLOPS`** (`5.07×` / `6.54×` vs BF16, `4×` L2-flushed, pre-quantized GEMM only; MXFP4 unsupported). With exact FP64 Hessians (`40/40` GPTQ rows `damp=0.01`), GPTQ cuts full `test.bin` (`1,999,872` tok) `ΔNLL` `3–5×` vs RTN; parameter-free `hadamard_b16_gptq` ties `adaptive_butterfly_b16_gptq` on FP4, while adaptive butterfly wins modestly on 2-bit.

---

## Verified Cycle 3 Results & Honest Scope Limits

### 1. RTX 5080 (`sm_120`) Post-Fusion CUDA-Graph Replay & Tier-1 Refresh (`C3-03`)

Sources: [`results/cycle3_c3_03_cuda_graph_bench.json`](results/cycle3_c3_03_cuda_graph_bench.json) (`0/55` capture errors, `0.0` graph-vs-eager diff, `22` Williams blocks × `5` iters, `4×` L2 flush), [`results/cycle3_c3_03_tier1_refresh.json`](results/cycle3_c3_03_tier1_refresh.json) (`117` cases, `0/117` divergences), & [`results/cycle2_heldout_nesterov_eval.json`](results/cycle2_heldout_nesterov_eval.json) (`45` held-out FineWeb-Edu inputs).

| Method (`dtype`) | Status | `(512, 2048)` Graph / Wall (ms) | `(1024, 4096)` Graph (ms) | `(2048, 8192)` Graph (ms) | `4096×4096` Graph (ms) | `(4, 1024, 4096)` Graph (ms) | Tier-1 (`117`) Graph / Med Err | Held-Out (`45`) All Med |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`ss3_54_blackwell_fp16`** (`16` kernels) | `VERIFIED` | 0.1270 (`0.987×`) / **0.2256** | **0.5018** (`1.031×`) | **3.0177** (`1.026×`) | **9.9086** (`1.062×`) | **1.8176** (`1.017×`) | 0.0577 / **0.0385** | **0.0389** |
| **`ss3_53_blackwell_fp16`** (`15` kernels) | `VERIFIED` | **0.1212** (`1.034×`) / 0.2221 | **0.4837** (`1.070×`) | **2.9278** (`1.057×`, `2.70×` PE) | **9.2959** (`1.132×`) | **1.7630** (`1.049×`) | **0.0553** / 0.0607 | 0.0548 |
| **`ps3_55_blackwell_fp16`** (`16` kernels) | `VERIFIED` | 0.1253 (`1.000×`) / 0.2289 | 0.5175 (`1.000×`) | 3.0954 (`1.000×`) | 10.5212 (`1.000×`) | 1.8485 (`1.000×`) | 0.0573 / 0.0641 | 0.0619 |
| **`ps2_55_blackwell_fp16`** | `VERIFIED` | 0.1251 / 0.2242 | 0.5175 | 3.0956 | 10.5159 | 1.8487 | 0.0573 / 0.0813 | 0.0802 |
| `ss3_35_blackwell_fp16` (`15` kernels) | `CAVEATED` | 0.1208 (`1.037×`) / 0.2169 | 0.4813 (`1.075×`) | 2.9172 (`1.061×`) | 9.2447 (`1.138×`) | 1.7524 (`1.055×`) | 0.0553 / 0.0619 | 0.0640 |
| `ss3_34_blackwell_fp16` (`14` kernels) | `CAVEATED` | 0.1147 (`1.093×`) / 0.2139 | 0.4608 (`1.123×`) | 2.8164 (`1.099×`) | 8.6070 (`1.222×`) | 1.6968 (`1.089×`) | 0.0514 / 0.2003 | 0.2146 |
| `pe_triton_bf16` | `BASELINE` | 0.1966 / 0.6170 | 0.8649 | 5.6709 | 13.2680 | 3.2700 | 0.0844 / 0.0936 | 0.0924 |
| `pe_eager` | `BASELINE` | 0.2199 / 0.4516 | 1.1540 | 7.9057 | 20.0259 | 4.3592 | 0.1024 / 0.0987 | 0.0960 |

*Speedups in parentheses are vs same-run `ps3_55_blackwell_fp16`. `SS3_54` remains `1.3%` behind `PS3_55_blackwell` in CUDA-graph time at `(512, 2048)` (`0.987×`) and at parity (`0.991×`) on the Tier-1 median. `results/cycle3_c3_03_tier1_refresh.json` supersedes pre-sync-free latency metadata in `results/schatten_switch_search_cycle2.json`.*

### 2. Wide-Layer Full-Model Crossover & `120m` Short-Budget FineWeb-Edu Transfer (`C3-01`)

Sources: [`results/cycle3_c3_01_wide_crossover.json`](results/cycle3_c3_01_wide_crossover.json) (`735` captures, `0.0` graph-vs-eager diff), [`results/cycle3_c3_01_120m_real_tokens.json`](results/cycle3_c3_01_120m_real_tokens.json), & [`results/cycle3_c3_01_120m_real_wall.json`](results/cycle3_c3_01_120m_real_wall.json) (`seeds 1–3`, `0/33` div, `0.0s` overshoot).

- **12-Layer Wide Step Crossover (CUDA-Graph vs `ps3_blackwell`):**
  - `ss3_53`: `1.019×` (`w512`), `1.039×` (`w768`), `1.040×` (`w768_120m_harness`), `1.035×` (`w1024`), `1.059×` (`w1536`), **`1.067×`** (`w2048`, `138.295` vs `147.551 ms`; `2.439×` vs `pe_compiled`).
  - `ss3_54`: `0.977×` (`w512`), `0.996×` (`w768`), `1.001×` (`w768_120m_harness`), `1.000×` (`w1024`, `w1021` interpolated), `1.013×` (`w1536`), **`1.020×`** (`w2048`, `144.729 ms`).
- **`120m` FineWeb-Edu 3-Seed Transfer (`32,768` tok/step, `lr=0.02` Muon / `0.001` AdamW):**
  - **256 Fixed Tokens (`8.39M` tok):** `muon_ss3_54` (`5.4640` test loss, `86,052` tok/s), `muon_ss3_53` (`5.4658`, `86,101` tok/s), and `muon_ps3_blackwell` (`5.4690`, `86,021` tok/s) gain `+2.90–3.00%` tok/s over `pe_compiled` (`5.4672`, `83,597` tok/s; Holm `p <= 0.0018`).
  - **100s Accepted Compute:** `muon_ss3_53` (`5.4468`, `262.67` steps), `muon_ss3_54` (`5.4527`, `262.0` steps, `-0.0055` vs `ps3_blackwell` `5.4582`, Holm 10-pair `p = 0.0352`, joint `p = 0.0810`), and `ps3_blackwell` (`5.4582`) outpace `pe_compiled` (`5.4833`, `255.0` steps). *Scope limit:* short budgets (`< 20M` tokens); `ps3_blackwell` stays the conservative `120m` default.

### 3. Hybrid Switching Ablation (`C3-02`), $\mu$P Coordinate Check (`T4-01`) & Blackwell NVFP4/INT2 PTQ (`T5-01`)

- **`C3-02` (`50m`, `5` seeds):** [`cycle3_c3_02_hybrid_50m.json`](results/cycle3_c3_02_hybrid_50m.json) & [`cycle3_c3_02_early_polish_eval.json`](results/cycle3_c3_02_early_polish_eval.json) — hybrid switching (`PS3_55`/`SS3_54 -> SS3_34` at `w40`) regresses test loss (`5.3823`/`5.3831` vs `5.3719` `ps3_blackwell`, falsified); `SS3_35` BF16 ties parity (`5.3745`, `+0.0026`); early-polish `SS3_54E0` is not promoted.
- **`T4-01` ($\mu$P CPU, `3` seeds):** [`cycle3_mup_coord_check_cpu.json`](results/cycle3_mup_coord_check_cpu.json) — SP fails (`steady max |beta| = 2.599` AdamW / `1.553` Muon) and `readout_lr_power=1` fails (`0.975–1.049`); $\mu$P passes (`0.116` AdamW, `0.078–0.082` Muon across `svd_exact`, `PE`, `PS3_55`, `SS3_54/53/34/35`).
- **`T5-01` (Blackwell NVFP4 & Micro-Rotated PTQ):** [`cycle3_t5_01_nvfp4_ptq.json`](results/cycle3_t5_01_nvfp4_ptq.json) (`40/40` GPTQ rows `damp=0.01`, `1,999,872` `test.bin` tokens, `4×` L2-flushed native probe):
  - **Native `sm_120` NVFP4 `F.scaled_mm` (GEMM-only, pre-quantized):** `(1024, 4096)` **`0.0671 ms` (`511.8 TFLOPS`, `5.074×` vs BF16 `0.3409 ms`)**; `(2048, 8192)` **`0.3629 ms` (`757.5 TFLOPS`, `6.540×` vs BF16 `2.3735 ms`)**; MXFP4 unsupported on RTX 5080.
  - **`muon_ss3_54` (`50m` seed 1, baseline PPL `220.14`) `ΔNLL` / `PPL`:** `nvfp4_g16` RTN `+0.00950` (`222.24`) -> `had_b64_gptq` **`+0.00212` (`220.61`)** (`bfly_b16_gptq` `+0.00215`); `mxfp4_g32` RTN `+0.01300` -> `bfly_b16_gptq` **`+0.00363`** (`had_b16_gptq` `+0.00365`); `int2_asym_g16` RTN `+0.15383` -> `bfly_b16_gptq` **`+0.03336`** (`had_b16_gptq` `+0.03402`); `int2_asym_g32` RTN `+0.26761` -> `bfly_b16_gptq` **`+0.05345` (`232.23`)**. *Caveats:* undertrained `adamw` (`PPL 409.62`) creates an iso-token `1.86×` PPL confound; NVFP4 synthetic outliers favor unrotated `gptq` (`0.000773`, `6.95×` over RTN) over rotation (`0.001475`).

### 4. Apple M5 Max (`40`-Core GPU, MLX `0.32.5`) 64×64 Metal 4 Triangular SYRK (Cycle 2 Baseline)

Sources: [`results/m5_max_dense_cycle2.json`](results/m5_max_dense_cycle2.json), [`results/m5_max_triangular_cycle2.json`](results/m5_max_triangular_cycle2.json) (**79/79** live Metal tests, `0/360` divergences; `C3-04` open-lid `4096²` rerun queued while `Ryan's MacBook Pro` is unreachable):
- `2048×2048`: Metal `PS3_55` **`3.059 ms` FP16 / `3.042 ms` BF16** (**`2.575×` / `2.566×`** vs dense MLX `PS3_55`, **`2.200×` / `2.192×`** vs dense `PE`); `1024×4096` `1.814×` FP16 / `1.763×` BF16; FP32 cleanly gated to dense MLX Steel.

---

## Independent Adversarial Audits (`audits/`)

### Cycle 3
- **C3-03 (`Claude Opus 5.5`, live RTX 5080, `high`):** [`AUDIT_cycle3_c3_03_claude_opus.md`](audits/AUDIT_cycle3_c3_03_claude_opus.md) — `PASS_WITH_CAVEATS` -> L1/L3/L4 remediated (Triton 3.8 C launcher, tile-norm fusion, `0` D2H syncs, 180/180 bitwise graph overwrites, `SS3_54` `(512,2048)` `0.987×` caveat)
- **C3-01 (`GPT 6.2 Astra`, CPU, `high`):** [`AUDIT_cycle3_c3_01_gpt62_astra.md`](audits/AUDIT_cycle3_c3_01_gpt62_astra.md) — `PASS_WITH_CAVEATS` -> remediated (`crossover()` exact-tie & canonical output guard; short-budget caveat)
- **C3-02 (`GPT 6.2 Astra`, CPU, `high`):** [`AUDIT_cycle3_c3_02_gpt62_astra.md`](audits/AUDIT_cycle3_c3_02_gpt62_astra.md) — `PASS_WITH_CAVEATS` -> remediated (`promotion_gate` accuracy, search/eval CLI collision guards, CPU tests)
- **T4-01 (`Gemini 4 Neon`, CPU, `high`):** [`AUDIT_cycle3_t4_01_gemini_neon.md`](audits/AUDIT_cycle3_t4_01_gemini_neon.md) — `PASS_WITH_CAVEATS` -> remediated (`3`-seed coord & val LR transfer, `rtol=1e-6`, `steady_gate`)
- **T5-01 (`Gemini 4 Neon`, CPU + re-check, `high`):** [`AUDIT_cycle3_t5_01_gemini_neon.md`](audits/AUDIT_cycle3_t5_01_gemini_neon.md) — `PASS_WITH_CAVEATS` -> **`HOLD LIFTED — READY TO MERGE`** (caught & verified fix of CUDA `autocast` BF16 Hessian corruption F1, `4×` L2 flush F4, `hadamard_b16_gptq` F5, CLI guard F3, FP16 clamp F7)

### Cycle 1–2 Archive
- Cycle 2: [`AUDIT_cycle2_claude_opus.md`](audits/AUDIT_cycle2_claude_opus.md), [`AUDIT_cycle2_t2_02_claude_opus.md`](audits/AUDIT_cycle2_t2_02_claude_opus.md), [`AUDIT_cycle2_claude_fable.md`](audits/AUDIT_cycle2_claude_fable.md), [`AUDIT_cycle2_gpt62_astra.md`](audits/AUDIT_cycle2_gpt62_astra.md), [`AUDIT_cycle2_gemini_neon.md`](audits/AUDIT_cycle2_gemini_neon.md), [`AUDIT_cycle2_t3_01_presweep_gemini_neon.md`](audits/AUDIT_cycle2_t3_01_presweep_gemini_neon.md)
- Cycle 1: [`AUDIT_cycle1_claude_opus.md`](audits/AUDIT_cycle1_claude_opus.md), [`AUDIT_cycle1_claude_fable.md`](audits/AUDIT_cycle1_claude_fable.md), [`AUDIT_cycle1_gpt62_astra.md`](audits/AUDIT_cycle1_gpt62_astra.md), [`AUDIT_cycle1_gemini_neon.md`](audits/AUDIT_cycle1_gemini_neon.md)

---

## Reproduction Commands

```bash
# Run full verified suite (1054 passed, 6 skipped on Linux RTX 5080)
PYTHONPATH=. .venv/bin/pytest -q

# Regenerate daily Cycle 3 HTML infographic
PYTHONPATH=. .venv/bin/python reports/generate_infographic.py --date 2026-11-26

# Cycle 3 C3-03 3-clock CUDA-graph & 117-case Tier-1 refresh
PYTHONPATH=. .venv/bin/python research/polar_opt/bench_c3_03_tier1_refresh.py

# Cycle 3 C3-01 wide-layer full-model crossover
PYTHONPATH=. .venv/bin/python research/polar_opt/bench_wide_crossover.py

# Cycle 3 T5-01 Blackwell NVFP4 / MXFP4 / INT2 PTQ
PYTHONPATH=. .venv/bin/python research/quant_ptq/micro_rotated_ptq.py --data-dir artifacts/real_text

# Cycle 3 T4-01 CPU muP coordinate check & LR transfer
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python research/polar_opt/mup_coord_check.py --jobs 8
```
