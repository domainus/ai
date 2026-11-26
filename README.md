# domainus/ai — Autonomous Verifiable AI Research Lab

An open, continuously operating autonomous AI research repository investigating **verifiable, programmatically grounded machine learning systems and numerical algorithms** across a multi-machine fleet (`bazzite` NVIDIA RTX 5080 Blackwell `sm_120`, `Ryan's MacBook Pro` Apple M5 Max MLX/Metal 4), built with cross-model engineering and mandatory multi-model adversarial review (**Claude Opus 5.5**, **Claude Fable 5.2**, **GPT 6.2 Astra**, **Gemini 4 Neon** at `high`).

---

## Cycle 2 Executive Summary (Audited, Remediated & Merged — 742 Passed on Linux + 79/79 Live Metal on M5 Max)

Cycle 2 delivered and adversarially audited three parallel hardware/algorithm tracks:
1. **`T1-04` (`SS3_54` & `SS3_53` Sync-Free CUDA-Graph Schatten-8 Circuits):** First `_symprod_kernel`-native, 3-buffer, **zero-D2H-sync CUDA-graph capturable** Schatten-8 Polar Switch circuits (`9` and `8` squarings vs Du & Su `PS3_55`'s `10`).
2. **`T3-01` (Real-Text `50m` FineWeb-Edu 5-Seed Transfer & 3-Clock CUDA-Graph Benchmark):** Disjoint GPT-2 FineWeb-Edu shards (`26.0M` train / `2.00M` val / `2.00M` test), compiled `Transformer`, interior seed-0 LR selection, matched contiguous FP32 parameter updates, zero-overshoot fixed accepted-compute budgets, and `fp32` vs `native` layout ablations.
3. **`T2-02` (Apple Silicon M5 Max 64×64 Metal 4 Cooperative-Tensor `SYRK` & Fused `symprod`):** Custom Metal shaders for `float16`/`bfloat16` (with `float32` cleanly gated to MLX native Steel GEMM).

---

## Verified Cycle 2 Results & Honest Scope Limits

### 1. RTX 5080 (`sm_120`) Pure CUDA-Graph Replay & Held-Out FineWeb-Edu Accuracy

Sources: [`results/cycle2_cuda_graph_bench.json`](results/cycle2_cuda_graph_bench.json) (`0/55` capture errors, `0.0` graph-vs-eager diff, 22 Williams blocks × 5 iters, 4× L2 flush) & [`results/cycle2_heldout_nesterov_eval.json`](results/cycle2_heldout_nesterov_eval.json) (`45` held-out real `50m` FineWeb-Edu post-Nesterov inputs across steps `20, 200, 399`; `0/45` divergences for all methods).

| Method (`dtype`) | Status | `(512, 2048)` Graph (ms) | `(1024, 4096)` Graph (ms) | `(2048, 8192)` Graph (ms) | `4096×4096` Graph (ms) | `(4, 1024, 4096)` Graph (ms) | Held-Out All Med | Held-Out Step-20 Med | Held-Out Late (s399) Med |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`ss3_54_blackwell_fp16`** (9 sq) | `VERIFIED` | 0.1393 | 0.5489 | **3.0935** (`1.005×`) | **10.0746** (`1.044×`) | **1.8647** (`1.010×`) | **0.0389** | 0.2326 | **0.0329** |
| **`ss3_53_blackwell_fp16`** (8 sq) | `VERIFIED` | 0.1352 | 0.5280 | **2.9999** (`1.036×`) | **9.4495** (`1.113×`) | **1.8096** (`1.040×`) | 0.0548 | 0.3603 | 0.0536 |
| **`ps3_55_blackwell_fp16`** (10 sq) | `VERIFIED` | **0.1253** | 0.5188 | 3.1087 (`1.000×`) | 10.5201 (`1.000×`) | 1.8828 (`1.000×`) | 0.0619 | **0.0822** | 0.0604 |
| **`ps2_55_blackwell_fp16`** (10 sq) | `VERIFIED` | **0.1253** | 0.5196 | 3.1089 | 10.5175 | 1.8833 | 0.0802 | 0.0996 | 0.0768 |
| `ss3_35_blackwell_fp16` (8 sq, BF16-safe) | `CAVEATED` | 0.1319 | 0.5268 | 2.9892 (`1.040×`) | 9.4126 (`1.118×`) | 1.7999 (`1.046×`) | 0.0640 | 0.3968 | 0.0558 |
| `ss3_34_blackwell_fp16` (7 sq, BF16-safe) | `CAVEATED` | 0.1265 | **0.5059** (`1.025×`) | **2.8853** (`1.077×`) | **8.7693** (`1.200×`) | **1.7357** (`1.085×`) | **0.2146** | **0.5223** | 0.0464 |
| `pe_triton_bf16` | `BASELINE` | 0.1966 | 0.8657 | 5.6984 | 13.3328 | 3.2987 | 0.0924 | 0.1206 | 0.0915 |
| `pe_compiled` (`torch.compile`) | `BASELINE` | 0.2022 | 1.1270 | 7.5592 | 19.4718 | 4.3442 | — | — | — |
| `ps3_55` (`eager`) | `BASELINE` | 0.1765 | 0.7680 | 4.9177 | 19.8815 | 3.1257 | — | — | — |
| `pe_eager` | `BASELINE` | 0.2212 | 1.1551 | 7.9379 | 20.1022 | 4.3618 | 0.0960 | 0.1310 | 0.0932 |

*Speedups in parentheses are vs `ps3_55_blackwell_fp16`. Note: `schatten_switch_search_cycle2.json` retains pre-sync-free latency metadata; `cycle2_cuda_graph_bench.json` is authoritative for kernel timing. `SS3_34` is disqualified as a standalone LM drop-in (fails early-step lift and regresses `50m` LM loss); `gram_space_restart` is rejected (`75/117` FP16 divergences).*

### 2. RTX 5080 `50m` Real FineWeb-Edu 5-Seed Transfer (Compiled, Matched FP32 Updates)

Source: [`results/cycle2_tier2_real.json`](results/cycle2_tier2_real.json) (`5` held-out seeds `1–5`, `0/70` divergences, `0.0 s` overshoot; LRs selected on `seed 0` pilots `adamw=0.001`, `muon=0.02`, `aux=0.001`) & [`results/cycle2_tier2_real_native.json`](results/cycle2_tier2_real_native.json) (`native` layout ablation).

| Optimizer (`lr`) | Status | Fixed-Tok Test Loss (`Δ` vs PE, 95% CI) | `75s` Compute Test Loss (`Δ` vs PE, 95% CI) | Mean Tok/s (`×` vs PE) | Polar Ortho (ms) | Opt Step (ms) |
|---|---|---:|---:|---:|---:|---:|
| **`muon_ss3_54`** (`0.02`) | `VERIFIED` | 5.4017 (`-0.0032`, `[-0.0172, +0.0109]`) | **5.3830** (`-0.0329`, `[-0.0395, -0.0262]`) | 142,876 (`1.031×`) | 3.23 | 6.08 |
| **`muon_ss3_53`** (`0.02`) | `VERIFIED` | **5.3972** (`-0.0077`, `[-0.0200, +0.0045]`) | **5.3844** (`-0.0314`, `[-0.0405, -0.0224]`) | 142,975 (`1.032×`) | 3.10 | 5.95 |
| **`muon_ps3_blackwell`** (`0.02`) | `VERIFIED` | 5.4046 (`-0.0003`, `[-0.0106, +0.0100]`) | 5.3897* (`-0.0262*`, `[-0.0417, -0.0107]`) | **142,993** (`1.032×`) | **3.08** | **5.92** |
| `muon_ps3_eager` (`0.02`) | `BASELINE` | 5.4001 (`-0.0049`, `[-0.0120, +0.0023]`) | 5.3898 (`-0.0260`, `[-0.0305, -0.0216]`) | 141,974 (`1.025×`) | 4.69 | 7.56 |
| `muon_pe_compiled` (`0.02`) | `BASELINE` | 5.4012 (`-0.0037`, `[-0.0145, +0.0071]`) | 5.4150 (`-0.0008`, `[-0.0075, +0.0058]`) | 138,855 (`1.002×`) | 9.68 | 12.78 |
| `muon_pe_eager` (`0.02`) | `BASELINE` | 5.4049 (`ref`) | 5.4158 (`ref`) | 138,585 (`1.000×`) | 10.09 | 13.15 |
| `adamw` (`0.001`) | `BASELINE` | 6.0274 (`+0.6224`, `[+0.5812, +0.6637]`) | 6.0573 (`+0.6414`, `[+0.4759, +0.8070]`) | 145,459 (`1.050×`) | — | 2.18 |
| `muon_ss3_34` *(seed-0 pilot)* | `CAVEATED` | *+0.0421 vs `pe_eager` at 128 steps* | — | — | *2.90* | *5.75* |

*\*All four PS3/SS3 wall loss wins survive Holm correction across the 6 comparisons to `muon_pe_eager`. At `50m` (`width=512`), `n=512` sits below the `n >= 2048` `SS3` crossover, so `SS3_54`/`SS3_53` match `PS3_55_blackwell` rather than beating it end-to-end (`120m` queued in Cycle 3), and `+2.45%` of the `+3.18%` throughput gain over `PE` comes from `PS3_55 eager` + `+0.72%` from Blackwell Triton. Seed-5 `muon_ps3_blackwell` wall run retained a `1.56s` host stall (`320` vs `327` steps). Native-dtype ablation (`seeds 1–3`): `1.0317×` (`native`) vs `1.0319×` (`fp32`).*

### 3. Apple M5 Max (`40`-Core GPU, MLX `0.32.5`) 64×64 Metal 4 Triangular SYRK & Symprod

Sources: [`results/m5_max_dense_cycle2.json`](results/m5_max_dense_cycle2.json), [`results/m5_max_triangular_cycle2.json`](results/m5_max_triangular_cycle2.json), [`results/m5_max_primitives_cycle2.json`](results/m5_max_primitives_cycle2.json) (**79/79** live Metal tests, `max|G - G^T| == 0`, `0/360` divergences):

| Shape | Dtype | Status | Dense `PE` (ms) | Dense `PS3_55` (ms) | **Metal `PS3_55`** (ms) | Speedup vs Dense `PS3` / `PE` |
|---|---|---|---:|---:|---:|---:|
| `512×2048` | `float16` | `VERIFIED` | 0.845 | 0.708 | **0.488** | **1.45× / 1.73×** |
| `1024×4096` | `float16` | `VERIFIED` | 3.465 | 2.241 | **1.235** | **1.81× / 2.81×** |
| `2048×2048` | `float16` | `VERIFIED` | 6.729 | 7.877 | **3.059** | **2.58× / 2.20×** |
| `2048×2048` | `bfloat16` | `VERIFIED` | 6.666 | 7.806 | **3.042** | **2.57× / 2.19×** |
| `4096×4096` *(min-ratio)* | `fp16 / bf16` | `PROVISIONAL` | 46.84 / 54.13 | 52.11 / 62.59 | **20.48 / 20.87** | **2.54× / ~2.61× min** |

*Pre-sleep `4096²` FP16 primitive A/B: fused `metal_symprod` **1.372 ms** (`3.12×` vs `4.275 ms` dense affine); `metal_syrk` **1.404 ms** (`1.68×` vs `2.362 ms` dense Gram). Honest FP32 caveat: custom Metal regresses against MLX's native Steel GEMM (`0.77–0.94×`) and is gated off (`F1`) so FP32 routes to dense MLX Steel while FP16/BF16 dispatch to 64×64 Metal. Sustained `4096²` runs are provisional due to closed-lid `Dark Wake Thermal Emergency` sleeps (open-lid `caffeinate -dimsu` rerun queued for Cycle 3).*

---

## Independent Adversarial Audits (`audits/`)

### Cycle 2
- **Auditor 1 (Claude Opus 5.5, `bazzite`, `high`):** [`AUDIT_cycle2_claude_opus.md`](audits/AUDIT_cycle2_claude_opus.md) — `PASS_WITH_CAVEATS` -> remediated (sync-free CUDA-graph bitwise checks, 0 D2H syncs, 0/55 capture errors)
- **Auditor 1 (Claude Opus 5.5, `M5 Max T2-02`, `high`):** [`AUDIT_cycle2_t2_02_claude_opus.md`](audits/AUDIT_cycle2_t2_02_claude_opus.md) — `PASS_WITH_CAVEATS` -> remediated (`F1` FP32 gate, `F3`, `F6` rebase, `F7`, `L1–L4`)
- **Auditor 2 (Claude Fable 5.2, `high`):** [`AUDIT_cycle2_claude_fable.md`](audits/AUDIT_cycle2_claude_fable.md) — `PASS_WITH_CAVEATS` -> remediated (Fable R1–R3 stress verified; long-`K` slack fix, `sigma_hat` NaN guard, held-out step-20 caveat documented)
- **Auditor 3 (GPT 6.2 Astra, `high`):** [`AUDIT_cycle2_gpt62_astra.md`](audits/AUDIT_cycle2_gpt62_astra.md) — `PASS_WITH_CAVEATS` -> remediated (100-run corpus/pairing/CI forensic check to `1e-12`)
- **Auditor 4 (Gemini 4 Neon, `high`):** [`AUDIT_cycle2_gemini_neon.md`](audits/AUDIT_cycle2_gemini_neon.md) — `PASS_WITH_CAVEATS` -> remediated (+ [`AUDIT_cycle2_t3_01_presweep_gemini_neon.md`](audits/AUDIT_cycle2_t3_01_presweep_gemini_neon.md))

### Cycle 1 Archive (Remediated, `531/531` Baseline)
- Audits: [`AUDIT_cycle1_claude_opus.md`](audits/AUDIT_cycle1_claude_opus.md), [`AUDIT_cycle1_claude_fable.md`](audits/AUDIT_cycle1_claude_fable.md), [`AUDIT_cycle1_gpt62_astra.md`](audits/AUDIT_cycle1_gpt62_astra.md), [`AUDIT_cycle1_gemini_neon.md`](audits/AUDIT_cycle1_gemini_neon.md).
- Key Cycle 1 wins & safety discoveries: verified `PS2_55`/`PS3_55` Blackwell FP16 (`0.304/0.307 ms` 117-case Tier-1, `triton_syrk` `1.63–2.02×` at `N >= 1024`), flagged `polar_express_fp8` (`fp8_steps >= 2`) `68.75%` `uniform_mp` divergence, `fp8_sketch` non-commutation, and heuristic step-11 warm-power failure.

---

## Reproduction Commands

```bash
# Run full verified suite (742 passed on Linux RTX 5080; 79/79 live Metal on M5 Max)
PYTHONPATH=. .venv/bin/pytest -q

# Regenerate daily Cycle 2 HTML infographic
PYTHONPATH=. .venv/bin/python reports/generate_infographic.py --date 2026-11-25

# RTX 5080 3-clock CUDA-graph benchmark (SS3_* + PS3_55 + PolarExpress)
PYTHONPATH=. .venv/bin/python research/polar_opt/bench_cuda_graph.py \
  --extra muon_ss3_54=research.polar_opt.tier2_candidates:ss3_54_blackwell_fp16 \
  --extra muon_ss3_53=research.polar_opt.tier2_candidates:ss3_53_blackwell_fp16

# Apple M5 Max MLX / Metal 4 benchmark (run on MacBook Pro)
PYTHONPATH=. .venv/bin/python research/polar_opt/kernels_mlx.py --bench --backend mlx --use-metal-syrk
```
