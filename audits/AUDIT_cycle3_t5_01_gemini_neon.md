# Cycle 3 Track 5 (`T5-01`) Adversarial Audit — Blackwell NVFP4 & 2-Bit Micro-Rotated PTQ

- **Auditor:** Gemini 4 Neon (CPU-only, `CUDA_VISIBLE_DEVICES=""`, zero GPU contact)
- **Date:** 2026-11-26
- **Audited Branch:** `origin/exp/c3-t5-01-nvfp4-ptq`
- **Audited Commits:** code `403b4ec17cdd38f8984d5cba0d663486b9f9e138`, results `23270c11048bf386bfdd58f1292f9c8aaffba536` (authored by Claude Opus 5.5)
- **Audit Scope:** Strictly read-only on source, tests, results, and shards (`research/quant_ptq/micro_rotated_ptq.py`, `tests/test_micro_rotated_ptq.py`, `results/cycle3_t5_01_nvfp4_ptq.json`, and read-only live SHA-256 verification of `/home/ryan/Documents/projects/ai/artifacts/real_text/*.bin`).

---

## Executive Verdict & Promotion Disposition

| Gate | Verdict | Basis |
|---|---|---|
| **(1) Provenance, strict JSON & CPU tests** | **PASS** | `git_sha == 403b4ec17c…`, `git_dirty == false`, `403b4ec..23270c1` touches only `results/cycle3_t5_01_nvfp4_ptq.json`, strict JSON (`allow_nan=False`, 0 duplicate keys, 0 non-standard constants), `34 passed, 3 skipped, 0 failed` on CPU. |
| **(2) Quantizer & hardware-format fidelity** | **PASS** | Exact E2M1 RNE codebook, E4M3FN g16 + FP32 tensor scale, cuBLAS `SWIZZLE_32_4_4` verified, OCP MX v1.0 floor rule verified, INT2 asymmetric g16/g32 & effective bits verified (minor edge cases in **F7**). |
| **(3) Micro-rotations & GPTQ Hessian path** | **FAIL / NEEDS FIX** | Rotations and Cholesky GPTQ algebra are exact on CPU, **but `collect_hessians` silently computes `x.T @ x` in BFLOAT16 inside CUDA `torch.autocast`** (**F1**, `HIGH`). This actively produced indefinite Hessians on `adamw`, forcing 10× damping (`0.01 → 0.1`) on all `16/16` `adamw` GPTQ rows and corrupting `H_cal`/`H_val` layer-error telemetry. |
| **(4) Data disjointness, shard SHA-256 & eval** | **PASS** | Live byte-level SHA-256 of `train/val/test.bin` matches pins; calibration uses `train.bin` only (128 disjoint 512-tok windows); full `test.bin` (`3,906` windows = `1,999,872` predicted tokens) has zero off-by-one/stride/leak bugs; paired fingerprints match Cycle 2 `50m` seed-1 runs. |
| **(5) Native FP4 probe, stress test & claims** | **PASS_WITH_CAVEATS** | All 72 grid rows, `summary`, `synthetic_stress.recovery`, and `native_fp4_gemm_probe` TFLOPS/speedups recompute to `0.000e+00` error; BF16 output rounding explains `1.66e-3` error. Caveats: Muon-vs-AdamW `2.01×` baseline PPL confound (**F2**), missing CLI overwrite guard (**F3**), no L2 flush & GEMM-only latency (**F4**), missing `hadamard_b16_gptq` baseline (**F5**), and NVFP4 spike-pinning anti-rotation mechanism (**F6**). |
| **OVERALL VERDICT** | **`PASS_WITH_CAVEATS`** | **HOLD promotion of final Cycle 3 T5-01 numerical claims and `adamw` GPTQ/layer-error results until F1 (autocast Hessian bug) and F3 (CLI overwrite guard) are patched, a regression test is added, and the RTX 5080 artifact is regenerated.** Core quantizer, rotation, evaluation, and native FP4 probe infrastructure is sound and ready to merge once F1 + F3 are fixed. |

---

## Severity-Ranked Findings

### HIGH

#### F1 (`HIGH`): `collect_hessians` silently computes `x.T @ x` in BFLOAT16 on CUDA due to active `torch.autocast`, causing indefinite Hessians and 10× damping fallback on all 16 `adamw` GPTQ rows
- **Location:** `research/quant_ptq/micro_rotated_ptq.py:606-623` (lines `608-609` inside the line-`618` `with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):` block).
- **Root Cause:** Although line 608 explicitly casts activations to FP32 (`x = inputs[0].detach().reshape(-1, module.in_features).float()`), the `forward_pre_hook` executes **inside** the active CUDA `bfloat16` autocast context without `torch.autocast(device_type=device.type, enabled=False)`. Because `torch.matmul` (`@`) is an autocast-eligible downcast op in PyTorch, `(x.T @ x)` silently downcasts `x` back to `bfloat16` (7-bit mantissa, $\epsilon \approx 2^{-7}$) and produces a `bfloat16` Gram matrix *before* `.double()` is applied.
- **Forensic Proof in Committed Artifact (`results/cycle3_t5_01_nvfp4_ptq.json`):**
  - Every single `adamw` GPTQ row (`16/16` across all 4 formats and all 4 GPTQ methods — including rotated methods, because orthogonal similarity $R H R^T$ preserves negative eigenvalues) records `gptq_damping_max = 0.1` instead of `0.01`, whereas all `16/16` `muon_ss3_54` GPTQ rows record `0.01`.
  - For a true PSD Gram matrix $H = X^T X / N$, $H + 0.01 \cdot \text{mean}(\text{diag}(H))\,I$ is strictly positive definite and Cholesky cannot fail. In `adamw`, low-rank collapsed layers (`stable_rank = 3.38` on `blocks.2.out`, `input_energy_max_over_mean = 151.66` max across layers) suffer `O(2^{-7} \|H\|_2)` BF16 matmul error that drives $\lambda_{\min}(H_{\text{bf16}}) < -0.01\,\text{mean}(\text{diag}(H))$.
- **Independent CPU Reproduction Evidence:**
  - Confirmed `dtype((x.float().T @ x.float())) == torch.bfloat16` inside `torch.autocast("cpu", dtype=torch.bfloat16)`.
  - On a rank-3 + noise activation (`8192 × 512`) mimicking `adamw`'s `stable_rank ≈ 3.4`:
    - **Buggy BF16-via-autocast `H_bug`:** $\lambda_{\min} = -17.69$ (vs $0.01 \cdot \text{mean\_diag} = 3.04$), relative Frobenius error vs true FP64 $H_{\text{true}}$ = `1.67e-3`, Cholesky **fails** at `damp = 0.01` (`info = 30`) and passes at `damp = 0.1`.
    - **True FP32/FP64 (autocast disabled):** $\lambda_{\min} = +1.42\times 10^{-3}$ (FP64) / $-1.04\times 10^{-3}$ (FP32), Cholesky **passes cleanly** at `damp = 0.01` (`info = 0`).
    - **Downstream impact:** On `nvfp4_g16`, `GPTQ(H_bug)` degraded true output error (`tr(E H_true E^T)`) by **1.22×** (`3.56e-5 → 4.33e-5`). Worse, because `H_bug` is indefinite, `output_error(W, Q, H_bug)` evaluated to **negative** values (`-8.35e-6` on NVFP4, `-1.42e-4` on INT2-g32), meaning negative per-layer terms can partially cancel positive errors in `calibration_relative_output_error_mean` / `val_relative_output_error_mean` for `adamw`!
- **Why tests missed it:** CPU unit tests run with `enabled = (device.type == "cuda") == False`, and `test_quick_pipeline_cuda` only trains 6 steps on `tiny` (well-conditioned activations).
- **Mitigating note:** End-to-end `test_nll` / `test_ppl` are genuine forward-pass cross-entropy evaluations on `test.bin` (not computed via `H_bug`), and `muon_ss3_54` stayed at `damp = 0.01` (though its `H_cal`/`H_val` still suffered ~`1.6e-3` BF16 rounding noise).
- **Required Fix:** Wrap the hook body in `with torch.autocast(device_type=device.type, enabled=False):`, cast `x` to `.double()` (or FP32 with symmetrization `0.5 * (H + H.T)`), add a CPU/CUDA autocast regression test asserting `collect_hessians` stays PSD inside `autocast(..., dtype=torch.bfloat16)`, and regenerate `results/cycle3_t5_01_nvfp4_ptq.json`.

#### F2 (`HIGH` — Scientific Confound): Muon-vs-AdamW quantizability comparison is confounded by a `2.01×` baseline PPL gap (`219.91` vs `441.60`) and single-seed (`seed=1`) noise
- **Location:** `research/quant_ptq/micro_rotated_ptq.py:905-908, 1135-1146`; `results/cycle3_t5_01_nvfp4_ptq.json` (`summary.muon_vs_adamw_quantizability`).
- **Evidence:** At 320 steps (`10.49M` tokens, `0.16` tokens/param), `muon_ss3_54` reaches `NLL = 5.3932` (`PPL = 219.91`, mean `stable_rank = 173.2`, min `123.0`), whereas `adamw` is severely undertrained at `NLL = 6.0904` (`PPL = 441.60`, mean `stable_rank = 28.5`, min `3.38`). Consequently, `muon_ss3_54` loses `4.5×–10×` more test NLL than `adamw` across all `36/36` grid cells (`muon_minus_adamw_delta_nll` ranges from `+0.0016` to `+0.2269`). Comparing `ΔNLL` across a `0.697` NLL (`2.01×` PPL) training-quality gap measures **convergence stage / representation rank**, not intrinsic optimizer quantizability. Additionally, `seed=1` only (1 seed) means narrow `< 0.0005` NLL rankings among FP4 GPTQ variants (`+0.00195` vs `+0.00215` vs `+0.00216` on Muon NVFP4) are within single-seed noise.
- **Required Framing:** Explicitly qualify `muon_vs_adamw_quantizability` as iso-token (not iso-loss) and single-seed; defer any claim that Muon is inherently harder to quantize than AdamW to quality-matched (iso-loss), multi-seed checkpoints.

---

### MEDIUM

#### F3 (`MEDIUM`): Missing CLI overwrite guard allows `--quick` or subset runs to silently clobber `results/cycle3_t5_01_nvfp4_ptq.json`
- **Location:** `research/quant_ptq/micro_rotated_ptq.py:1269-1294`.
- **Details:** `--output` defaults to `ROOT / "results" / "cycle3_t5_01_nvfp4_ptq.json"`. Running `python research/quant_ptq/micro_rotated_ptq.py --quick` (or `--max-eval-windows 64`, `--no-verify`, `--steps 10`) without specifying `--output` passes `validate_results()` (which accepts `quick=True` and truncated `test_tokens`) and atomically overwrites the canonical production artifact.
- **Required Fix:** Require explicit `--output` (matching `research/polar_opt/kernels_mlx.py:621`) or raise an error when targeting the canonical path under `--quick`, `--no-verify`, `--max-eval-windows`, or non-default `--preset`/`--steps`.

#### F4 (`MEDIUM`): Native Blackwell FP4 probe omits L2 cache flush and excludes online activation quantization & micro-rotation cost
- **Location:** `research/quant_ptq/micro_rotated_ptq.py:801-818, 874-898`.
- **Details:**
  - `_williams_latency` runs `iterations = 50` calls back-to-back on identical pre-quantized tensors with no L2 cache flush. On RTX 5080 (GB203, 64 MB L2), `1024×4096×4096` NVFP4 (`19.25 MB` working set) and BF16 (`48 MB`) both fit in L2; at `2048×8192×8192`, BF16 (`192 MB`) spills 3× past L2 while NVFP4 inputs + scales (`45 MB`) remain L2-resident, boosting the `6.80×` (`782.0` vs `115.0` TFLOPS) GEMM ratio alongside `sm_120`'s consumer BF16-FP32-accum throughput vs full-rate FP4 tensor cores.
  - Pre-quantized GEMM timing excludes online activation quantization, `to_blocked` scale swizzling, and online activation micro-rotation `x → x R^T` (which cannot be fused into nonlinear `SiLU(gate)*up` before `down`, nor across residual-stream RMSNorms with per-layer `rand_hadamard`/`adaptive_butterfly`). The JSON `design` note honestly discloses `"operands pre-quantized (GEMM only)"`.
  - **Verification:** `2*M*N*K / ms` TFLOPS and speedup ratios recompute exactly (`0.000e+00` error). Independent CPU check shows pure BF16 output rounding of `nvfp4_decode(A) @ nvfp4_decode(B)^T` produces relative error `0.001658` (`256×512×1024`) and `0.001662` (`1024×4096×4096`), matching the probe's `0.001660`–`0.001661` and confirming bit-accurate native NVFP4 accumulation. MXFP4 `F.scaled_mm BlockWise1x32` rejection (`NotImplementedError: MXFP4 scaling only supported in CUDA for B200/B300`) is cleanly captured.

#### F5 (`MEDIUM`): Omitted `hadamard_b16_gptq` baseline confounds block-size vs learned-angle gains; parallel GPTQ ignores cross-layer error compounding
- **Location:** `research/quant_ptq/micro_rotated_ptq.py:629-630, 599-623, 1040`.
- **Details:**
  - `METHODS` includes `hadamard_b16_rtn` (exact start point of `adaptive_butterfly_b16_rtn`) but omits `hadamard_b16_gptq`. Thus in GPTQ rows, `adaptive_butterfly_b16_gptq` is only compared against `gptq` and `b64` rotations. Independent CPU evaluation of `hadamard_b16_gptq` on the exact `synthetic_stress` problem (`512×1024`, $\kappa = 10^4$) shows:
    - `int2_asym_g16`: `gptq = 0.019035`, `had_b64_gptq = 0.010105`, **`had_b16_gptq = 0.007738`**, `adaptive_butterfly_b16_gptq = 0.005755` — **54% of butterfly's gain over `b64` comes purely from matching block size 16 to group size 16!**
    - `mxfp4_g32`: **`had_b16_gptq = 0.001803`** slightly beats `adaptive_butterfly_b16_gptq = 0.001847`.
    - `int2_asym_g32`: `had_b16_gptq = 0.025153` vs `had_b64_gptq = 0.021508` vs `adaptive_butterfly_b16_gptq = 0.016737` (genuine learned-angle gain).
  - Parallel (full-precision) Hessians are honestly documented (`lines 33-34`), but at 2-bit (`10%–15%` RTN / `3%–5%` GPTQ layer output MSE on Muon), sequential layer-by-layer error propagation would yield a more realistic 2-bit GPTQ upper bound.

#### F6 (`MEDIUM` — Scientific Mechanism): NVFP4 synthetic-stress "anti-rotation" (`2.08×` worse RTN, `1.91×` worse GPTQ) is driven by E4M3FN exact spike-pinning, not scale underflow
- **Location:** `research/quant_ptq/micro_rotated_ptq.py:719-769`; `results/cycle3_t5_01_nvfp4_ptq.json` (`synthetic_stress`).
- **Independent CPU Diagnosis:** On the exact `synthetic_stress` matrix (`8` channels spiked `25×` hold `~83%` of $\|W\|_F^2$):
  - **Zero** NVFP4 blocks hit E4M3FN zero or subnormal scales (`bs_min = 1.125`, `zero_bs = 0%`, `subnorm_bs = 0%`).
  - Even with **infinite-precision FP32 block scales**, NVFP4 relative weight MSE still jumps `2.33×` under `hadamard_b64` (`0.004724 → 0.011014`, vs actual NVFP4 `0.005299 → 0.011078`).
  - **Mechanism:** In unrotated NVFP4, fine-grained E4M3FN scales pin each isolated spike directly to E2M1 code `6.0` (near-0% error on 83% of weight energy), while `15.9%` zero-collapse sacrifices only 15 non-spike weights per spiked group (`~0.25%` of total energy). Rotating smears that 83% energy across 64 coordinates into E2M1's interior (`~1.1%` Gaussian MSE floor), doubling error. By contrast, unrotated MXFP4's power-of-2 E8M0 floor clips 41% of blocks by up to 25%, severely damaging the 83%-energy spikes (`2.30%` MSE), so Hadamard rotation halves MXFP4 error (`1.21%`). Any report citing `synthetic_stress` must note this anti-rotation regime for NVFP4.

---

### LOW

#### F7 (`LOW`): Quantizer & accounting edge cases
1. **NVFP4 static `global_scale` during GPTQ (`micro_rotated_ptq.py:484`):** `nvfp4_global_scale(W)` is fixed prior to GPTQ error feedback. Verified on CPU at $\kappa = 10^5$ that updated weights can reach the initial `amax` ceiling (`hit_ceil=True`), triggering `to_e4m3`'s `448.0` clamp safely (no NaN/Inf).
2. **NVFP4 E4M3FN scale underflow codes (`micro_rotated_ptq.py:223-226, 254-256`):** Under extreme `>10^5` inter-block dynamic range where a small block's scale underflows E4M3FN to `0.0`, `safe = 1.0` can store non-zero E2M1 nibbles (`[2, 2, 2, 2]`) that decode identically to `0.0` (`nvfp4_decode == quantize_rtn` is `True`).
3. **INT2 `to_fp16` unclamped overflow (`micro_rotated_ptq.py:146-147, 217`):** Unlike `to_e4m3`, `to_fp16` does not clamp to `±65504`. Verified on CPU that a block span `250,000` overflows `scale` to `inf` and yields `NaN` (`0 * inf`). Benign for neural weights (`O(0.02)`), trivial to harden.
4. **Butterfly FP16 storage vs FP32 simulation (`micro_rotated_ptq.py:360, 404`):** Storage bills FP16 angles (`+0.034439` bits/quantized wt), while evaluation uses FP32 angles. Independent CPU check confirms FP16-rounded angles retain `3.58e-15` orthogonality and shift MSE by only `+0.0008%`.
5. **Whole-model vs quantized-layer bitrate (`59.93%` quantized):** Verified on `50m` meta `Transformer` (`total = 64,303,616`, `quantized = 38,535,168 = 59.9269%`, `embedding = 25,755,648 = 40.0532%`, `norms = 12,800`). Keeping embeddings/norms in BF16 yields `9.11` bits/param whole-model for `nvfp4_g16`. Also note `int2_asym_g16` costs `4.00` bits/wt (`2.0` code + `2.0` FP16 scale/zero) — nearly equal to `mxfp4_g32`'s `4.25` bits/wt while losing `9×–11×` more NLL on Muon.

---

## Independent Verification Summary & Recomputed Tables

### 1. Provenance, Live Shard Hashing, Data Hygiene & CPU Test Suite
- **Git provenance:** `403b4ec17cdd38f8984d5cba0d663486b9f9e138` (`git_dirty: false`) matches `meta.git_sha`. `git diff --stat 403b4ec..23270c1` touches **only** `results/cycle3_t5_01_nvfp4_ptq.json` (`+6019` lines).
- **Strict JSON check:** `json.loads(..., parse_constant=reject, object_pairs_hook=no_dup)` + `json.dumps(..., allow_nan=False)` passed with `0` duplicate keys and `0` non-finite values.
- **CPU Pytest (`CUDA_VISIBLE_DEVICES=""`):** `34 passed, 3 skipped, 0 failed` in `40.33s` (skipped: 3 `@needs_cuda` tests at `tests/test_micro_rotated_ptq.py:387, 414, 428`).
- **Shard SHA-256 (both metadata AND live read-only byte hashing of `/home/ryan/Documents/projects/ai/artifacts/real_text/*.bin`):**
  - `train.bin`: `015ab83ceb742555c2398963a0bd1f3c5502dc52cf70a22bf173f1571326ca3a` (`52,009,036` bytes = `26,004,518` `uint16` tokens) — **EXACT MATCH**
  - `val.bin`: `371b64a537f7b1e306ea174a868b05932a254a98b7fbdbf4146648d93180b0b1` (`4,004,660` bytes = `2,002,330` `uint16` tokens) — **EXACT MATCH**
  - `test.bin`: `4eb46f1424856849d2973923dbf3c886d37eb743cf22053fc7e1eea1f41ab92a` (`4,000,162` bytes = `2,000,081` `uint16` tokens) — **EXACT MATCH**
- **Calibration & evaluation hygiene:**
  - Calibration: `train.bin` only, `128` sorted non-overlapping `512`-token windows (`65,536` tokens; `min_start = 346,112`, `max_end = 25,934,848 <= 26,004,518`, `min_diff = 1,024`, all multiples of `512`). Zero overlap with `val.bin`/`test.bin`.
  - Evaluation (`evaluate_split`, `micro_rotated_ptq.py:575-590`): `windows = (2,000,081 - 1) // 512 = 3,906` → `1,999,872` predicted targets partitioning indices `[1, 1,999,872]` with zero overlap and zero gaps (`208` trailing remainder tokens unpredicted).
- **Cycle 2 baseline consistency:** Both models share identical `init_fingerprint = 85659cbc0ffc79c1` and `stream_fingerprint = b9f91c01f18ce738`, matching `results/cycle2_tier2_real.json` seed-1 `50m` (`320` steps, `10,485,760` tokens). Harness 1M-token test PPLs (`muon_ss3_54: 222.59` vs Cycle 2 `224.43`; `adamw: 443.56` vs Cycle 2 `422.07`) are consistent given `warmup_steps = 32` (vs Cycle 2 `20`) and `compile_model = False` (vs Cycle 2 `True`).

### 2. Quantizer, Rotation & GPTQ Independent CPU Probes
- **NVFP4 (`nvfp4_g16`):** E2M1 codebook & RNE tie-breaking exact (`±0.25→±0, ±0.75→±1, ±1.25→±1, ±1.75→±2, ±2.5→±2, ±3.5→±4, ±5.0→±4`); `to_blocked` verified element-by-element against cuBLAS `SWIZZLE_32_4_4` for `(128,4)` and padded `(130,5)`; `nvfp4_decode == quantize_rtn` exact.
- **OCP MXFP4 (`mxfp4_g32`) floor vs no-clip ceiling CPU probe (`512×1024`):**

| Distribution | Clipped Blocks (`>6s`) | Clipped Elems | Mean / Max Rel Clip Err | Rel MSE (OCP Floor) | Rel MSE (No-Clip Ceil) | Ratio (`Ceil/Floor`) |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian(0, 0.02) | 42.1% | 1.85% | 9.93% / 25.00% | `0.013026` | `0.014068` | `1.080×` (OCP wins) |
| Student-t(df=3) | 41.1% | 1.85% | 11.80% / 25.00% | `0.019965` | `0.022756` | `1.140×` (OCP wins) |
| Synthetic stress | 41.0% | 1.78% | 11.89% / 25.00% | `0.023021` | `0.021621` | `0.939×` (Ceil wins) |

- **GPTQ vs RTN across condition numbers (`64×128`, CPU, true FP64 `H`):** GPTQ strictly beats RTN at every $\kappa \in [1, 10^8]$ across all 4 formats (`gptq/rtn` drops from `0.982–0.991` at $\kappa=1$ to `0.347–0.451` at $\kappa=10^3$ and `0.118–0.167` at $\kappa=10^8$, all at `damp = 0.01`).

### 3. Recomputed `50m` Full-Test Summary (`1,999,872` tokens; max recomputation discrepancy = `0.000e+00`)

| Model (Baseline PPL) | Format (eff bits) | RTN `ΔNLL` (`ΔPPL%`) | Best Rot-RTN `ΔNLL` (method) | Plain GPTQ `ΔNLL` (`ΔPPL%`) | Best Rot-GPTQ `ΔNLL` (`ΔPPL%`, method) | `gptq_damping_max` |
|---|---|---:|---:|---:|---:|---:|
| **Muon `ss3_54`** (`219.91`, NLL `5.39323`) | `nvfp4_g16` (`4.50`–`4.53`) | `+0.00883` (`+0.89%`) | `+0.00764` (`had_b64`) | `+0.00239` (`+0.24%`) | `+0.00195` (`+0.20%`, `bfly_b16`) | `0.01` |
| | `mxfp4_g32` (`4.25`–`4.28`) | `+0.01300` (`+1.31%`) | `+0.01235` (`rhad_b64`) | `+0.00443` (`+0.44%`) | `+0.00346` (`+0.35%`, `had_b64`) | `0.01` |
| | `int2_asym_g16` (`4.00`–`4.03`) | `+0.14561` (`+15.67%`) | `+0.11967` (`bfly_b16`) | `+0.03896` (`+3.97%`) | `+0.03257` (`+3.31%`, `bfly_b16`) | `0.01` |
| | `int2_asym_g32` (`3.00`–`3.03`) | `+0.24992` (`+28.39%`) | `+0.21944` (`bfly_b16`) | `+0.06499` (`+6.72%`) | `+0.05140` (`+5.27%`, `bfly_b16`) | `0.01` |
| **AdamW** (`441.60`, NLL `6.09041`) | `nvfp4_g16` (`4.50`–`4.53`) | `+0.00195` (`+0.19%`) | `+0.00097` (`bfly_b16`) | `+0.00051` (`+0.05%`) | `+0.00036` (`+0.04%`, `bfly_b16`) | **`0.1` (F1)** |
| | `mxfp4_g32` (`4.25`–`4.28`) | `+0.00206` (`+0.21%`) | `+0.00211` (`bfly_b16`) | `+0.00097` (`+0.10%`) | `+0.00088` (`+0.09%`, `bfly_b16`) | **`0.1` (F1)** |
| | `int2_asym_g16` (`4.00`–`4.03`) | `+0.01826` (`+1.84%`) | `+0.01363` (`bfly_b16`) | `+0.00403` (`+0.40%`) | `+0.00424` (`+0.43%`, `bfly_b16`) | **`0.1` (F1)** |
| | `int2_asym_g32` (`3.00`–`3.03`) | `+0.02304` (`+2.33%`) | `+0.02305` (`had_b64`) | `+0.00769` (`+0.77%`) | `+0.00646` (`+0.65%`, `bfly_b16`) | **`0.1` (F1)** |

### 4. Recomputed Synthetic Stress Test & Native NVFP4 GEMM Probe
- **Synthetic stress recovery (`512×1024`, $\kappa = 11,634$, kurtosis `943.9`, `8` channels × `25`):**
  - `nvfp4_g16`: RTN `0.005374` → best **`gptq` `0.000773`** (`6.95×` recovery; `had_b64_rtn = 0.011160` is `2.08×` worse than RTN — see **F6**).
  - `mxfp4_g32`: RTN `0.023006` → best `rand_hadamard_b64_gptq` `0.001602` (`14.36×` recovery).
  - `int2_asym_g16`: RTN `0.087765` → best `adaptive_butterfly_b16_gptq` `0.005755` (`15.25×` recovery; CPU-verified `hadamard_b16_gptq = 0.007738`).
  - `int2_asym_g32`: RTN `0.208050` → best `adaptive_butterfly_b16_gptq` `0.016737` (`12.43×` recovery).
- **Native NVFP4 `F.scaled_mm` GEMM probe (RTX 5080 `sm_120`, pre-quantized GEMM only):**
  - `(1024, 4096, 4096)`: BF16 `102.08` TFLOPS (`0.3366 ms`) vs NVFP4 `607.10` TFLOPS (`0.0566 ms`), speedup `5.947×`, `rel_err = 0.001661`.
  - `(2048, 8192, 8192)`: BF16 `114.95` TFLOPS (`2.3913 ms`) vs NVFP4 `782.01` TFLOPS (`0.3515 ms`), speedup `6.804×`, `rel_err = 0.001656`.

---

## Exact Commands Run (All CPU-Only, `CUDA_VISIBLE_DEVICES=""`)

1. `git fetch origin exp/c3-t5-01-nvfp4-ptq && git checkout -B exp/c3-t5-01-audit-neon origin/exp/c3-t5-01-nvfp4-ptq`
2. `git rev-parse 23270c1 403b4ec && git diff --stat 403b4ec..23270c1 && git diff --stat 4b7a792..403b4ec`
3. `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/ryan/Documents/projects/ai/.venv/bin/pytest tests/test_micro_rotated_ptq.py -q` → **`34 passed, 3 skipped in 40.33s`**
4. `CUDA_VISIBLE_DEVICES="" /home/ryan/Documents/projects/ai/.venv/bin/python <scratch>/audit_inspect.py` → strict JSON (`0` dup keys), live SHA-256 of `train/val/test.bin`, full 72-row & summary recomputation (`max_err = 0.000e+00`).
5. `CUDA_VISIBLE_DEVICES="" /home/ryan/Documents/projects/ai/.venv/bin/python <scratch>/audit_c2_compare.py` → Cycle 2 `50m` seed-1 fingerprint and baseline reconciliation.
6. `CUDA_VISIBLE_DEVICES="" /home/ryan/Documents/projects/ai/.venv/bin/python <scratch>/audit_math_probes.py` & `audit_math_probes2.py` & `audit_math_probes3.py` → E2M1/E4M3FN/`SWIZZLE_32_4_4`/MXFP4 clipping/INT2 edge cases/Butterfly FP16/GPTQ $\kappa \in [1, 10^8]$/BF16 autocast Hessian reproduction/`hadamard_b16_gptq` stress test/BF16 output rounding floor/50M meta parameter accounting.
