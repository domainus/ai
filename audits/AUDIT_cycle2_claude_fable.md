# Adversarial Audit — Cycle 2 (Numerical Rigor, Fable R1–R3, Held-Out Nesterov)

- **Auditor:** Adversarial Auditor 2, Claude Fable 5.2 (high effort) · **Date:** 2026-11-25
- **Commit:** `main` HEAD `877c2a8413e05d89f105360a550eb382e464ba70`, tree clean except other auditors' reports.
- **Hardware:** NVIDIA GeForce RTX 5080, torch 2.14.1+cu130, Triton 3.8.0. No Apple Silicon run.
- **Scope:** [schatten_switch_blackwell.py](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py), [schatten_switch_search.py](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_search.py), [schatten_switch_search_cycle2.json](/home/ryan/Documents/projects/ai/results/schatten_switch_search_cycle2.json), [eval_heldout_nesterov.py](/home/ryan/Documents/projects/ai/research/polar_opt/eval_heldout_nesterov.py), [harvested_nesterov_real.pt](/home/ryan/Documents/projects/ai/artifacts/harvested_nesterov_real.pt), [cycle2_heldout_nesterov_eval.json](/home/ryan/Documents/projects/ai/results/cycle2_heldout_nesterov_eval.json).
- **Method:** Correctness and numerical stress tests only. I ran no timing benchmark. Latency numbers in this report are quoted from committed JSON files and are marked as such.
- **Changes made:** This file only. No source, test, JSON, or `state/` file was changed. All scratch scripts were outside the repository and are deleted.

## Executive Verdict: `PASS_WITH_CAVEATS`

**The five sync-free kernels are correct on every input that a Muon optimizer can produce.** I found no `CRITICAL` defect. Fable R1, R2 and R3 are fixed and I could not break them on grids 4 to 6 times denser than the committed ones.

Three caveats block an unqualified `PASS`:

1. **`HIGH` — the "parity with PS3_55" claim is in-sample.** On the held-out post-Nesterov inputs, all four circuits are 2.8 to 6.4 times worse than `PS3_55` at step 20. `SS3_54` fails the project's own parity gate out of sample (max error 0.690 against 0.591). `SS3_34` must not be used as a `PS3_55` replacement.
2. **`MEDIUM` — the bound is not rigorous on the native path for a long side `K ≥ 2^17`.** The Triton Gram matrix is biased low, linearly in `K`. The "rigorous" test checks only the PyTorch reference. One real divergence exists: `SS3_35` returns norm 8.74 on a constant vector of length `2^25` where `PS3_55` returns 1.11.
3. **`LOW` — several stale or untested statements** (stale latency and launch counts in the search JSON, a vacuous test assertion, a `0·Inf = NaN` leak on the `sigma_hat` test hook, silent zeros for float64 inputs above the FP32 range).

| Area | Result |
|---|---|
| 5 new Triton kernels, natural path (`sigma_hat=None`) | **PASS** — no non-finite output, no cross-member leak, dead members exactly zero |
| Fable R1 (`pI + c11ᵀ`, n = 1023 … 8192) | **PASS** — `σ_max ≤ 1.088` (FP16), `≤ 1.181` (BF16) over 1154 cases |
| Fable R2 (coherent rank-1 sweeps) | **PASS** — 150 of 150 sweeps, worst `σ_max` 1.092 (FP16), 1.149 (BF16) |
| Fable R3 (normalization, tiny / huge / non-finite) | **PASS** for FP32 inputs; float64 inputs above `3.4e38` return zeros (L3) |
| Bound `top ≤ upper` | **PASS** for `K ≤ 2^16`; **FAIL** for `K ≥ 2^17` (M1) |
| Held-out artifact and JSON | **PASS** — 45 distinct inputs, every row reproduced to `1.5e-15` |
| Honesty of the circuits against `PS3_55` | **CAVEATS** — H1, M2 |
| Repository tests `tests/test_schatten_switch.py -k "not latency"` | 138 passed, 4 deselected |

---

## Adversarial Stress-Test Results

All runs are on CUDA at HEAD. "ref16" is `schatten_switch_reference(dtype=float16)`. `CAP` is 1.25.

### Kernels and edge inputs

| # | Edge case (exact inputs) | Outcome |
|---|---|---|
| A | 19 shapes × `SS3_54`, `SS3_35`: `(1,1)`, `(1,17)`, `(64,1)`, `(3,64,1)`, `(5,1,17)`, `(7,3)`, `(3,7)`, `(2,1,1)`, `(33,65)`, `(65,33)`, `(128,256)`, `(4,96,200)`, `(2,3,31,31)`, `(1023,1023)`, `(1024,1024)`, `(2,1024,1100)`, `(1100,1024)`, `(1,4096)`, `(4097,1)` | **PASS 38/38.** Finite, `σ_max ≤ 1.087`, max `|native − ref16|` = `2.0e-3`. `(3,64,1)` and `(5,1,17)` have `work[1].data_ptr() % 16` = 6 and 10; no error, no wrong result |
| B1 | Scale sweep of one 48×80 matrix: `1e-42`, `1e-38`, `1e-31`, `1e-20`, `0.9999e-12`, `1.0001e-12`, `1e-6`, `1`, `1e10`, `1e30`, `1e37`, `3e38` | **PASS.** Zeros at and below the floor. Bitwise equal to the unit-scale output from `1.0001e-12` to `3e38` |
| B2 | Floor boundary, `‖G‖_F = 1e-12·(1 ± 1e-6)`, 81 points, 1×2 matrix | **Minor.** Native and reference disagree on 3 of 81 points (L2). 1×1 matrices at `fl32(1e-12)` and its two neighbours agree |
| B3 | Every entry `±3e38` and `±3.4028234e38` (40×40) | **PASS.** Finite, `|native − ref16|` = `6.9e-7`. The subnormal reciprocal `1/m_chunk` causes no loss |
| B4 | Every entry `1e-30`, `1.1e-30`, `1e-31` | **PASS.** Zeros on both paths |
| B5 | One matrix with entries `1e-42·N(0,1)`, one `3e38`, one `-1e37` | **PASS.** Finite, `|native − ref16|` = `4.3e-4` |
| B6 | Batch member with one 4096-block of `1e-31` next to blocks of `1e-25` | **PASS.** Same zero decision as the reference |
| B7 | float64 input scaled by `1e50` and `1e300` | **FAIL (L3).** Native returns all zeros; reference returns a valid polar factor (`σ_max` 1.047) |
| B8 | float64 input scaled by `1e-50`; float16 and bfloat16 input dtypes | **PASS.** Zeros on both paths; dtype restored |
| C1 | Mixed batch of 6: NaN, `+Inf`, `-Inf`, exact zero, `1e-20`, one clean; shapes `(6,40,72)`, `(6,72,40)`, `(6,1,17)`, `(6,64,1)`, `(6,130,130)`; FP16 and BF16 | **PASS.** Dead members exactly zero. Clean members are bitwise equal to the all-clean batch. No leak |
| C2 | Single NaN, `+Inf`, `-Inf`, all-NaN, all-zero matrix; same 5 shapes × 2 dtypes | **PASS 50/50.** Exact zeros |
| C3 | One NaN at 27 positions of a 300×300 matrix; one `Inf` at 19 positions of a 600×700 matrix (103 blocks, strided reduction) | **PASS.** 0 missed |
| D | `_safe_norm_part_kernel` against FP64 norm: 6 shapes × scales `1e-10`, `1`, `1e20`, `1e37` | **PASS.** Worst relative error `1.2e-7` |
| E | Batch against single call, Triton prefix (`X_h`, `R_h`, `S`, row 0, folded coefficients) | **PASS.** Bitwise equal |
| E2 | Batch against single call, full output | **Minor (L4).** Differs at small `n` (40 and 130 tested); up to `1.5e-2` in BF16. Bitwise equal from 512×1408 upward. The cause is cuBLAS, not the kernels |

### `sigma_hat` test hook

| # | Edge case | Outcome |
|---|---|---|
| F1 | `sigma_hat = 1e-10`, FP16, 64×128 Gaussian and rank-1 sign | **FAIL (L1).** `X_h` overflows to `Inf`, `ok = 0`, output is 8192 NaN: the `0·Inf` leak |
| F2 | `sigma_hat` in `1e-6 … 0.05`, FP16 | Output is silently all zeros (`ok = 0`). BF16 returns NaN for `1e-3 … 0.05`. Inconsistent failure modes (L1) |
| F3 | `sigma_hat = 1e10`, `1e30` | **PASS.** FP16 underflows to exact zeros; BF16 returns `σ ≈ slope · x`. No NaN |
| F4 | Per-matrix vector `[1e-10, 1, 1e10, 1e-4, 2, 1]`, batch of 6 | **PASS for isolation.** Every member equals its single-matrix call, including the NaN pattern. No cross-member leak |
| F5 | `sigma_hat` = `0`, `-1`, `NaN`, `Inf`, wrong length | **PASS.** `ValueError` |
| F6 | `sigma_hat = 1.0`, BF16, `SS3_54` (not BF16-verified) | `σ_max = 4.04e8`. This reproduces the JSON and confirms the BF16 gate is needed |

### Custom circuits

| # | Edge case | Outcome |
|---|---|---|
| G1 | Pure-square first row (`b0 = c0 = 0`), three legitimate circuits composed from the JSON families `lift-k{3,4,5}-b3-pure` + `sym-k5-b3-general`; 7 shapes | **PASS.** `|native − ref16| ≤ 1.0e-3`, dead members zero, rank-1 sweep `σ_max` 1.004 (FP16), 1.022 (BF16) |
| G2 | `residual ≠ 0` on both segments (`0.05, −0.03`; `−0.2, 0.3`; `0.07, 0`) with pure, semi and general first rows; shapes include `(3,64,1)`, `(5,1,17)`, `(1100,1024)` | **PASS.** `|native − ref16| ≤ 1.2e-3`, `|native − ref64| ≤ 1.1e-2` |
| G3 | Single-segment circuits with a residual; 2-buffer pure circuit | **PASS.** `|native − ref16| ≤ 1.4e-3` |
| G4 | Dead members with residual circuits | **PASS.** Exact zeros |
| G5 | Two different circuits with the same `name` | **PASS.** The table cache key holds the rows; no collision |
| G6 | 4-buffer circuit | Native raises `ValueError`; the reference runs it. Documented limit, not a defect |

### CUDA graph

| # | Edge case | Outcome |
|---|---|---|
| H1 | Capture after warm-up, `is_current_stream_capturing() == True`; shapes `(4,64,128)`, `(512,2048)`, `(3,64,1)` BF16, `(5,1,17)` BF16, `(2,1024,1024)`; 6 replays with scales `1e-8 … 1e12`, one NaN member, one zero member | **PASS.** Replay equals eager bitwise. Dead member is zero on replay |
| H2 | Capture without warm-up (new `N`) | **Fails** with `cudaErrorStreamCaptureInvalidated`. Expected: the first call autotunes. Needs documentation (L7) |
| H3 | Capture with a CUDA `sigma_hat` tensor | **Fails**: the hook reads the tensor on the host. Test hook only (L7) |

### Bound rigor and precision of the Python `float` arguments

| # | Edge case | Outcome |
|---|---|---|
| I1 | `λ_max(X_h X_hᵀ) ≤ upper` and `λ_max(R_h) ≤ upper` with the kernel's own FP64 norms. 10 families (Gaussian, rank-1 sign / Gaussian / ones / lognormal, rank-2, one-hot, flat and decaying diagonal, spike + tiny noise) × 13 shapes up to 2048² and `(1, 2^20)` × FP16, BF16 × 3 scales | **PASS for `K ≤ 4096`**: 0 violations. **13 violations, all at `(1, 2^20)`**, worst `top/upper` = 1.00505 (M1) |
| I2 | Native bound against reference bound, shapes up to 300² | **PASS.** Relative difference `≤ 2.6e-6` |
| I3 | `u_safe`, `fro_floor`, `residual` as FP32 registers | `u_safe = 1.0499999523`, `fro_floor = 9.99999996e-13`. `κ` is `9.1e-8` larger than the host value. Folded coefficients differ from the FP64 host fold by `≤ 3.8e-7`. `upper` does not use `u_safe`, so this cannot break `top ≤ upper` (L2) |
| K | Gram bias for `n = 1`, `K = 2^12 … 2^26`, Gaussian and constant vectors, FP16 and BF16 | Bias `1 − R_h/‖X_h‖²`: `4.9e-4` (2^16), `9.8e-4` (2^17), `5.9e-3` (2^20), `4.4e-2` (2^23), `0.156` (2^25), `0.274` (2^26). Linear in `K` (M1) |

### Fable R1 and R2

| # | Edge case | Outcome |
|---|---|---|
| J1 | `pI + c11ᵀ`, `n = 4096`, FP16, 4 circuits × 101 ratios in `[0.003, 1]` (61 of them in `[0.15, 0.30]`) | **PASS.** 0 non-finite. Worst `σ_max` (power iteration): `SS3_54` 1.081, `SS3_53` 1.088, `SS3_35` 1.083, `SS3_34` 1.067 |
| J2 | `n = 8192`, FP16, 4 circuits × 14 ratios in `[0.01, 1]` | **PASS.** Worst 1.083 |
| J3 | `n = 4096` and `8192`, **BF16**, `SS3_35` / `SS3_34` (the search does not run this family in BF16) | **PASS.** Worst 1.181 (`SS3_35`, ratio 0.00547), 1.080 (`SS3_34`) |
| J4 | `n = 1023` (no prescale), `1024`, `2048`; FP16 4 circuits, BF16 2 circuits; 40 ratios each | **PASS.** Worst 1.087 (FP16), 1.161 (BF16) |
| J5 | Exact FP64 `eigvalsh` on the 8 cases with the largest `spectral_bound` | **PASS** on `σ_max` (`≤ 1.181`). But `spectral_bound` is 1.256 and 1.261 on two of them, above `CAP` (L5) |
| L1 | Rank-1 sweeps, `x = σ/σ̂` in `(0, 1]`, 4002 points for small shapes. Matrices: sign, Gaussian, lognormal, ramp, ones, sign + noise `1e-3` and `3e-2`. Shapes: `(64,128)`, `(128,64)`, `(33,47)`, `(256,256)`, `(128,512)`, `(1,64)`, `(200,1)`, `(96,160)`, `(100,100)`, `(1024,1024)`, `(1024,2048)`, `(2048,2048)`. FP16 for all 4 circuits, BF16 for `SS3_35` / `SS3_34`. cuBLAS reduced-precision flags on and off | **PASS 150/150.** 0 non-finite. Worst FP16: `SS3_54` 1.0823, `SS3_53` 1.0903, `SS3_35` 1.0918, `SS3_34` 1.0724. Worst BF16: `SS3_35` 1.1493, `SS3_34` 1.1304 |
| L2 | Agreement with the JSON | `SS3_54` (64,128) FP16 `1.0823087319300841` (JSON `…0688`); `SS3_35` (128,512) BF16 `1.14929199`; `SS3_35` R1 4096 `1.2384`. The JSON was produced by the HEAD kernels |

### Held-out Nesterov

| # | Check | Outcome |
|---|---|---|
| N1 | Artifact content | 45 pairs. 45 distinct `N_t` hashes, 45 distinct `N_t1` hashes, all finite. Steps `{20, 200, 399}` × layers `{0, 6, 11}` × 5 matrices. `sha256 f9f6358b…0a6eb` |
| N2 | Stored spectra | `sv_t`, `sv_t1` equal a fresh FP64 `svdvals` to `2e-13` relative |
| N3 | Recompute of all 10 methods × 45 cases with my own CPU LAPACK polar target | Per-case error equals the JSON to `1.5e-15`; `σ_max` equals exactly. Every summary number reproduces |
| N4 | Is the early regression a numerical failure? | No. Measured error equals the exact-arithmetic scalar prediction to `5e-4` for every circuit |

---

## Severity-Ranked Findings

### `CRITICAL`

None.

---

### `HIGH`

#### [H1] "Parity with `PS3_55`" is an in-sample result. It fails on the held-out post-Nesterov inputs, most of all at step 20

**Location**
- Parity gate and default selection: [schatten_switch_search.py:63](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_search.py:63), [schatten_switch_search.py:1112-1115](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_search.py:1112), [schatten_switch_search.py:1149](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_search.py:1149).
- The polish optimizes on the same 45 spectra: [schatten_switch_search.py:504-514](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_search.py:504), [schatten_switch_search.py:641-683](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_search.py:641).
- Default variant: [schatten_switch_blackwell.py:254](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:254).
- Test that locks the in-sample claim: [test_schatten_switch.py:401-409](/home/ryan/Documents/projects/ai/tests/test_schatten_switch.py:401).
- JSON keys: `selected.SS3_54.parity_with_ps3_55 = true`, `default_variant = "SS3_54"`.

**Evidence.** Relative Frobenius error on the 45 held-out inputs, recomputed independently. "Ratio" is the circuit median divided by the `PS3_55` median. "Wins" counts the cases where the circuit has the lower error.

| Method | Early (step 20) median · ratio · wins | Mid (200) ratio · wins | Late (399) ratio · wins | All: median · p90 · max | Mean-error ratio |
|---|---|---|---|---|---|
| `PS3_55` (10 sq) | 0.0822 · 1.00 · — | 1.00 | 1.00 | 0.0619 · 0.218 · 0.591 | 1.00 |
| `SS3_54` (9 sq) | 0.2326 · **2.83** · 4/15 | 0.54 · 14/15 | 0.54 · 13/15 | 0.0389 · **0.292** · **0.690** | 1.06 |
| `SS3_53` (8 sq) | 0.3603 · **4.38** · 4/15 | 0.88 · 10/15 | 0.89 · 10/15 | 0.0548 · **0.387** · **0.744** | 1.41 |
| `SS3_35` (8 sq) | 0.3968 · **4.82** · 3/15 | 0.92 · 10/15 | 0.92 · 10/15 | 0.0640 · **0.443** · **0.768** | 1.55 |
| `SS3_34` (7 sq) | 0.5223 · **6.35** · 0/15 | 1.11 · 7/15 | 0.77 · 10/15 | **0.2146** · **0.634** · **0.846** | 2.10 |

`SS3_35` and `SS3_34` in BF16 give the same picture (early ratios 4.84 and 6.36).

**Cause.** It is a design property, not a rounding failure (stress row N4).

- The step-20 momenta are almost rank one: stable rank median 1.24, minimum 1.03. For those, `‖G‖_F ≈ σ_max`, so the Schatten-8 scale gains nothing over the Frobenius scale. `PS3_55` already has its top singular value at `x = 0.85`.
- The circuits pay for their saved squarings with a weaker lift. Slope at zero: `PS3_55` 1131, `SS3_54` 584, `SS3_53` 389, `SS3_35` 321, `SS3_34` 156. The input level where the output reaches 0.5: `PS3_55` 0.00047, `SS3_54` 0.00090, `SS3_53` 0.00135, `SS3_35` 0.00163, `SS3_34` 0.00338.
- At step 20, the share of singular values below that level is 10% for `PS3_55`, 17% for `SS3_54`, 24% for `SS3_53`, 27% for `SS3_35` and 46% for `SS3_34`.

**Why the search did not see it.** The polish and the parity gate both use the 45 Cycle 1 momenta in `harvested_trajectories.pt`. Their stable rank is 15.8 (early), 8.8 (mid) and 6.9 (late), minimum 2.81. They are pre-Nesterov EMA matrices of a synthetic model, and their "early" phase is steps 1 and 2. No input with stable rank below 2.8 was in the selection data. The search JSON did carry a warning that the gate ignores: the Tier-1 `power_law` maximum is 0.443 for `PS3_55` and 0.596, 0.678, 0.711, 0.811 for the four circuits.

**Assessment per circuit**

| Circuit | Verdict |
|---|---|
| `SS3_54` | **Caveat required.** Better than `PS3_55` from step 200 on (median 0.54×). Worse at step 20 (median 2.83×, wins 4/15). Fails the project's parity gate out of sample: max 0.690 > 0.591, p90 0.292 > 0.218. The label `parity_with_ps3_55: true` must say "in-sample, stable rank ≥ 2.8" |
| `SS3_53`, `SS3_35` | **Caveat required.** Roughly equal to `PS3_55` at mid and late steps (0.88× to 0.92×). 4.4× to 4.8× worse at step 20. The JSON already marks parity false |
| `SS3_34` | **Disqualify as a `PS3_55` replacement.** Median error 3.5× `PS3_55` overall, 0 wins of 15 at step 20, worse p90 in every phase (0.29 to 0.31 against 0.16 to 0.17 at mid and late steps) |

No circuit diverges: `σ_max ≤ 1.088` (FP16) and `≤ 1.117` (BF16) on all 45 inputs. The regression is under-lift, which is safe but reduces the update on small singular directions during warm-up.

**Limits of the held-out set** (they weaken any claim in either direction): every short side is 512, so the `n ≥ 1024` prescale path is never used. One trajectory, seed 999, produced by `muon_pe_eager` ([harness_tier2_real.py:724-770](/home/ryan/Documents/projects/ai/research/polar_opt/harness_tier2_real.py:724)), so it is open-loop for the Schatten-Switch circuits. "Early" is one step, inside the 50-step learning-rate warm-up. The overall p90 of 45 cases is set by step 20 alone.

**Reproduction** (CPU, about 2 seconds):

```python
import json, numpy as np, torch
d = json.load(open("results/cycle2_heldout_nesterov_eval.json"))
base = d["cases"]["ps3_55_blackwell_fp16"]
for m in ("ss3_54_blackwell_fp16", "ss3_53_blackwell_fp16", "ss3_35_blackwell_fp16", "ss3_34_blackwell_fp16"):
    for ph in ("early", "mid", "late"):
        e = np.array([r["relative_frobenius_error"] for r in d["cases"][m] if r["phase"] == ph])
        b = np.array([r["relative_frobenius_error"] for r in base if r["phase"] == ph])
        print(m, ph, f"median ratio {np.median(e) / np.median(b):.2f}  wins {int((e < b).sum())}/15")
pairs = torch.load("artifacts/harvested_nesterov_real.pt", map_location="cpu", weights_only=True)["pairs"]
for ph in ("early", "mid", "late"):
    sr = [float((p["sv_t"] ** 2).sum() / p["sv_t"][0] ** 2) for p in pairs if p["phase"] == ph]
    print(ph, f"stable rank median {np.median(sr):.2f} min {min(sr):.2f}")
# ss3_54 early 2.83 (4/15) ... ss3_34 early 6.35 (0/15); stable rank early 1.24 (min 1.03), mid 3.97, late 5.60
```

**Fix.** (1) Rename the flag to in-sample parity and add a held-out gate with a per-phase and a p90 criterion. (2) Add near-rank-one inputs (stable rank 1.0 to 1.5) to the polish targets, or route low-stable-rank inputs to `PS3_55`. The ratio `‖R_h‖_F / tr(R_h)` is available on the device; a host branch on it would add a synchronization, so a static per-layer choice is the simpler form. (3) Remove `SS3_34` from the candidate list for `PS3_55` replacement.

---

### `MEDIUM`

#### [M1] The bound is not rigorous on the native path for a long side `K ≥ 2^17`; one real divergence at `K = 2^25`

**Location**
- Claim: [schatten_switch_blackwell.py:26-29](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:26), [schatten_switch_blackwell.py:284-300](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:284) ("`_SLACK` covers the FP32 accumulation of the products"), [schatten_switch_blackwell.py:66](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:66).
- Source of the error: the FP32 accumulation `acc = tl.dot(a, b, acc)` in [kernels_blackwell.py:116-123](/home/ryan/Documents/projects/ai/research/polar_opt/kernels_blackwell.py:116), launched at [schatten_switch_blackwell.py:674](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:674).
- The test checks the reference, not the kernels: [test_schatten_switch.py:171-186](/home/ryan/Documents/projects/ai/tests/test_schatten_switch.py:171).

**Evidence.** The stored Gram matrix `R_h` is lower than the FP64 value of `X_h X_hᵀ`. The bias is always downward and grows linearly with `K`: about `5.6e-9 · K` for a Gaussian vector and up to `9.8e-9 · K` for a constant vector. `_SLACK = 1 + 2^-12` covers it only up to `K ≈ 2^16`. I did not isolate the mechanism; a downward-only, linear bias is what a truncating accumulator gives.

| `K` (n = 1, Gaussian, FP16) | `1 − R_h / ‖X_h‖²` | native `upper` | reference `upper` | `λ_max / upper` |
|---|---|---|---|---|
| `2^13` | `< 1e-5` | 1.000855 | 1.000855 | 0.99915 |
| `2^16` | `4.9e-4` | 1.000366 | 1.000855 | 0.99963 |
| `2^17` | `9.8e-4` | 0.999877 | 1.000855 | **1.00012** |
| `2^20` | `5.9e-3` | 0.994973 | 1.000855 | **1.00505** |
| `2^23` | `4.4e-2` | 0.956372 | — | **1.04562** |
| `2^25` | `0.156` | 0.844471 | — | **1.18417** |

The same holds for `n > 1`: a rank-1 sign matrix of shape `(2, 2^20)` gives `λ_max / upper = 1.00934`, and `(64, 262144)` gives 1.00078.

**Effect on the output.** It is smaller than the bound error suggests, because the polynomial is evaluated on the same biased `R_h`. The output error is multiplicative, about `sqrt(true / stored)` per segment.

- For `K ≤ 2^23` every tested output stayed below 1.07.
- A constant vector `(1, 2^25)`: `SS3_35` returns norm **8.74** (FP16) and **10.26** (BF16). `SS3_34` returns 1.19 and 1.21, `SS3_54` 1.02, `SS3_53` 1.03, `PS3_55` 1.11. The divergence limit is 2.0.
- Realistic shapes are safe: rank-1 `(768, 50257)` gives `λ_max / upper = 0.99944`, output `σ_max = 0.959`.

**Severity reasoning.** A matrix parameter with a side of `1.3e5` breaks the stated guarantee by `1e-4`, and a side of `3.3e7` is needed to diverge. No transformer layer in this project comes near. The finding is `MEDIUM` because the docstring states rigor without a size limit, the only rigor test bypasses the native kernels, and the native and reference bounds silently diverge.

**Reproduction** (CUDA, about 10 seconds):

```python
import torch
from research.polar_opt.kernels_blackwell import triton_syrk
from research.polar_opt.polar_switch_blackwell import polar_switch_blackwell
from research.polar_opt.schatten_switch_blackwell import schatten_switch_blackwell as F, _upper
for logk in (13, 16, 17, 20, 23):
    G = torch.randn(1, 1 << logk, device="cuda")
    X = (G / G.norm()).half()
    true = X.double().square().sum().item()                 # lambda_max(X_h X_h^T) in FP64
    R = triton_syrk(X).double().item()                      # the Gram matrix the bound is computed from
    upper = _upper(R, 2.0 ** 13 * R * R, 1, torch.float16)  # n = 1: ||S||_F = alpha * R_h^2
    print(f"K=2^{logk}: R_h={R:.6f} upper={upper:.6f} true/upper={true / upper:.5f}")
# K=2^13 0.99915 | 2^16 0.99963 | 2^17 1.00013 | 2^20 1.00503 | 2^23 1.04561
G = torch.ones(1, 1 << 25, device="cuda")
print(F(G, "SS3_35").double().norm().item(), polar_switch_blackwell(G, "PS3_55").double().norm().item())
# 8.7402  1.1125
```

**Fix.** Any one of these: (1) make the bound depend on `K`, for example multiply `upper` by `1 / (1 − 1e-8·K)`, which covers the largest bias I measured; (2) take the trace of `R` from the FP64 norm that `_normalize_safe_kernel` already has, and scale `R_h` to it; (3) reject or split `K > 2^16`. Add a native-path rigor test that reads `R_h` and `S` from the device.

#### [M2] The search JSON holds stale latency and launch counts that contradict the newer benchmark

**Location**
- [schatten_switch_search.py:966-972](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_search.py:966): `kernel_launches` hard-codes `"host_reads": 1` and adds one launch for `n ≥ 1024`. The docstring describes the old host-read pipeline.
- JSON keys `selected.*.kernel_launches`, `latency_sanity`, `summary.*.gpu_speedup_vs_ps3_55`.
- Contradicted by [schatten_switch_blackwell.py:33-38](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:33) (zero host synchronizations) and by `results/cycle2_cuda_graph_bench.json`.

**Evidence.** These are committed numbers. I did not re-measure them.

| Source | 512×2048, GPU time, `SS3_54` / `PS3_55` |
|---|---|
| `schatten_switch_search_cycle2.json`, 16:48:54 | 0.306 ms / 0.129 ms (speedup 0.42×) |
| `cycle2_cuda_graph_bench.json`, 17:19:27 | 0.142 ms / 0.130 ms eager events; 0.139 / 0.125 CUDA graph |

The search JSON under-reports the circuits by a factor of two at small sizes and reports one host read where the code has none. A reader of the "verified" search JSON gets the wrong picture in both directions.

Also relevant to H1: with the newer numbers, at `n = 512` (the size of every held-out matrix) `SS3_54` is still not faster than `PS3_55` (0.139 ms against 0.125 ms CUDA graph, 0.385 ms against 0.294 ms wall). On the `50m` model the default circuit is therefore both slower and, at step 20, less accurate. The saved squaring pays off only from about 2048×8192 upward (3.09 ms against 3.11 ms) and at 4096² (10.07 ms against 10.52 ms).

**Reproduction:**

```python
import json
a = json.load(open("results/schatten_switch_search_cycle2.json"))
b = json.load(open("results/cycle2_cuda_graph_bench.json"))
print(a["latency_sanity"]["shapes"]["512x2048"]["SS3_54"]["speedup_gpu"], a["selected"]["SS3_54"]["kernel_launches"]["512"])
m = b["shapes"][0]["methods"]
print(b["shapes"][0]["shape"], m["muon_ss3_54"]["eager_event_ms"], m["muon_ps3_blackwell"]["eager_event_ms"])
```

**Fix.** Regenerate the JSON with HEAD, or delete the `latency_sanity`, `gpu_speedup_vs_ps3_55` and `kernel_launches` fields and point to the CUDA-graph benchmark. Record the git SHA in the search JSON; it has none.

---

### `LOW`

#### [L1] `0·Inf = NaN` leak and inconsistent failure modes on the `sigma_hat` hook

**Location:** [schatten_switch_blackwell.py:447-457](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:447) (`unit_scale` can overflow the working dtype while `valid` stays true), [schatten_switch_blackwell.py:507](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:507) and [schatten_switch_blackwell.py:611](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:611) (`ok = 0` zeroes `Q`), [schatten_switch_blackwell.py:697-702](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:697) (`Q @ X_h` with `X_h = Inf`).

`ok = 0` zeroes the polynomial but not the operand. With `sigma_hat = 1e-10` in FP16, `X_h` is `Inf` and the output is 8192 NaN. With `sigma_hat` between `1e-6` and `0.05`, FP16 returns silent zeros and BF16 returns NaN. The natural path cannot reach this state: without the hook `|X_h| ≤ 64`, so `R_h ≤ 4096` and `S ≤ 2^13`. The hook is documented as unsafe, so this is `LOW`. The module docstring still promises exact zeros and no host read.

```python
import torch
from research.polar_opt.schatten_switch_blackwell import schatten_switch_blackwell as F
G = torch.randn(64, 128, device="cuda")
print(int(torch.isnan(F(G, "SS3_54", sigma_hat=1e-10)).sum()),                               # 8192
      not F(G, "SS3_54", sigma_hat=1e-3).any(),                                              # True: silent zeros
      int(torch.isnan(F(G, "SS3_35", dtype=torch.bfloat16, sigma_hat=1e-3)).sum()))          # 8192
```

**Fix.** Multiply `X` by the per-matrix `ok` flag before `_apply`, or mark the matrix invalid in `_normalize_safe_kernel` when `prescale / factor` exceeds the dtype range.

#### [L2] `u_safe` and `fro_floor` reach the kernels as FP32

**Location:** [schatten_switch_blackwell.py:665](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:665), [schatten_switch_blackwell.py:689-690](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:689), [schatten_switch_blackwell.py:494](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:494), [schatten_switch_blackwell.py:514](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:514).

Triton passes Python floats as FP32. The kernel uses `u_safe = 1.0499999523162842` and `fro_floor = 9.999999960041972e-13`. `u`, `e`, `prescale`, `alpha` and `_SLACK` are exact in FP32. Measured effect: `κ` is `9.08e-8` larger than the host value, and the folded coefficients differ from the FP64 host fold by `3.2e-7` to `3.8e-7` (they are `≤ 5.5e-8` from a host fold that uses `fl32(1.05)`). The floor decision differs from the reference on 3 of 81 points within `±1e-6` of `1e-12`.

**This cannot break the bound.** `upper` does not contain `u_safe`, and the kernel computes it in FP64 from FP64 norms. The headroom is 1.0499999523 instead of 1.05, which is four orders of magnitude below the FP16 unit roundoff. The only cost is that "the same arithmetic" in the reference docstring is not exact.

**Fix.** Store the three constants in the FP64 table tensor that already goes to the device.

#### [L3] float64 inputs above the FP32 range return zeros

**Location:** [schatten_switch_blackwell.py:638](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:638) (`.float()` before the "FP64 overflow-safe" norm).

A finite float64 matrix with entries above `3.4e38` becomes `Inf` in FP32, is flagged non-finite, and returns zeros. The reference returns a valid polar factor. The docstring lists only dead, tiny and non-finite inputs as zero cases.

```python
import torch
from research.polar_opt.schatten_switch_blackwell import schatten_switch_blackwell as F, schatten_switch_reference as REF
G = torch.randn(32, 48, device="cuda", dtype=torch.float64) * 1e50
print(not F(G, "SS3_54").any(), torch.linalg.svdvals(REF(G, "SS3_54", dtype=torch.float16))[0].item())  # True 1.047
```

**Fix.** Document the FP32 input range, or raise for float64 inputs whose FP32 cast is not finite.

#### [L4] Batched and single calls are not bitwise equal at small `n`; the result depends on a global cuBLAS flag

**Location:** [polar_switch_blackwell.py:206-217](/home/ryan/Documents/projects/ai/research/polar_opt/polar_switch_blackwell.py:206) (`_apply`). The flag is set in [eval_heldout_nesterov.py:70-71](/home/ryan/Documents/projects/ai/research/polar_opt/eval_heldout_nesterov.py:70) but not in `schatten_switch_search.py` or in the tests.

The Triton prefix is bitwise identical in both modes. The difference comes from `torch.bmm`. At 6×130×130 in BF16 with default flags the outputs differ by `1.5e-2` and `σ_max` is 1.1194 (batch) against 1.1337 (single). With both `allow_*_reduced_precision_reduction` flags off the difference is `5.3e-3`. From 512×1408 upward the two modes are bitwise equal. The JSON value `max_batch_vs_single: 0.0` is measured on exact sign matrices only. My R2 sweeps pass with the flags on and off, so safety holds in both modes.

```python
import torch
from research.polar_opt.schatten_switch_blackwell import schatten_switch_blackwell as F
G = torch.randn(6, 130, 130, device="cuda", generator=torch.Generator(device="cuda").manual_seed(11))
b = F(G, "SS3_35", dtype=torch.bfloat16)
s = torch.stack([F(G[i], "SS3_35", dtype=torch.bfloat16) for i in range(6)])
print((b - s).abs().max().item())  # 0.0149
```

**Fix.** State that "FP32 accumulation, one rounding" covers the Triton kernels only, or pass `out_dtype=torch.float32` in `_apply` for every segment and cast once.

#### [L5] The R1 selection gate depends on the ratio grid

**Location:** [schatten_switch_search.py:745-751](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_search.py:745), [schatten_switch_search.py:761](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_search.py:761), [schatten_switch_search.py:773](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_search.py:773), [test_schatten_switch.py:152-157](/home/ryan/Documents/projects/ai/tests/test_schatten_switch.py:152). BF16 is not run on this family: [schatten_switch_search.py:958-962](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_search.py:958).

The gate compares `sqrt(‖Q‖₁‖Q‖∞)` with `CAP` on 16 ratios. That bound is not tight when the rank-one part of the output is negative. Off the grid, `SS3_53` gives 1.2560 at `n = 4096`, ratio 0.175, and 1.2667 at `n = 2048`; `SS3_35` in BF16 gives 1.2611 at `n = 1023`. The exact `σ_max` is 1.085 for the first case and 1.161 for the third, and the largest `σ_max` over all 40 ratios at `n = 2048` is 1.087 (power iteration), so the circuits are safe. The gate would have rejected `SS3_53` with a 61-point grid, and the committed test fails if 0.175 is added to its ratio list.

```python
from research.polar_opt.schatten_switch_blackwell import schatten_switch_blackwell as F
from research.polar_opt.schatten_switch_search import identity_plus_constant, spectral_bound
print(spectral_bound(F(identity_plus_constant(4096, 0.175), "SS3_53")))  # 1.2560 > CAP = 1.25
```

**Fix.** Use the exact `σ_max` of the two-dimensional invariant subspace (or power iteration) as the gate, and run the family in BF16 for BF16-verified circuits. I ran it: pass.

#### [L6] A vacuous assertion in the rigor test

**Location:** [test_schatten_switch.py:186](/home/ryan/Documents/projects/ai/tests/test_schatten_switch.py:186).

`assert ratio < 1.003 if dtype == FP16 else 1.01` parses as `assert ((ratio < 1.003) if dtype == FP16 else 1.01)`. For BF16 it asserts the constant `1.01`, which is always true. **Fix:** `assert ratio < (1.003 if dtype == FP16 else 1.01)`.

#### [L7] CUDA-graph capture has two undocumented preconditions

**Location:** [polar_switch_blackwell.py:113-141](/home/ryan/Documents/projects/ai/research/polar_opt/polar_switch_blackwell.py:113), [schatten_switch_blackwell.py:307-318](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:307), [schatten_switch_blackwell.py:651-654](/home/ryan/Documents/projects/ai/research/polar_opt/schatten_switch_blackwell.py:651).

"100% CUDA-graph capturable" holds only after one eager call with the same shape, dtype and variant. A first call inside a capture autotunes and invalidates the capture (`cudaErrorStreamCaptureInvalidated`). The `sigma_hat` path cannot be captured: it calls `.tolist()` and builds a host tensor. After warm-up, replay equals eager bitwise on every shape I tried, including `(3,64,1)` and `(5,1,17)`.

#### [L8] The held-out set does not test what the cycle changed most

All 45 inputs have a short side of 512. The power-of-two prescale (`n ≥ 1024`) and the FP16 subnormal regime (`n ≥ 4096`), which are the Fable R1 fixes, are not exercised by any real-data input. The synthetic R1 family and my sweeps cover them, but no harvested momentum does. See also the limits listed under H1. **Fix:** harvest from the `120m` preset, or add a wider model, before the next selection.

---

## Positive Findings (verified, no action needed)

- **Fable R1 is fixed.** 1154 `pI + c11ᵀ` cases at `n` = 1023 to 8192, FP16 and BF16: 0 non-finite, `σ_max ≤ 1.181`. Cycle 1 returned `Inf` or 5.4 here.
- **Fable R2 is fixed.** 150 coherent rank-1 sweeps, including non-sign vectors, added noise, tall shapes, `n = 1`, and the prescaled path at 1024 and 2048: worst `σ_max` 1.092 (FP16) and 1.149 (BF16). The `bf16_verified` flags are correct: `SS3_54` and `SS3_53` do blow up in BF16 (`4e8`), `SS3_35` and `SS3_34` do not.
- **Fable R3 is fixed for FP32 inputs.** Zeros at and below `1e-12`, bitwise scale invariance from `1.0001e-12` to `3e38`, no NaN from `Inf`, `-Inf` or NaN members, no cross-member contamination.
- **The kernels match the FP16 reference** to `2e-3` on 19 shapes, and pure-square and residual circuits to `1.2e-3`.
- **The held-out JSON is honest about its own numbers.** Every row reproduces to `1.5e-15`, the artifact holds 45 distinct real post-Nesterov inputs, and the file does not hide the early-step regression. The problem in H1 is the parity label in the search JSON, not this file.
- **The committed search JSON matches HEAD** on every safety number I recomputed.
