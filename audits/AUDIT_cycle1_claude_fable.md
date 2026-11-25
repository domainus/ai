# Cycle 1 Adversarial Audit 2 — Spectral Pre-Scaling, Schatten-8 Bound, Accounting, Stability Basins

- **Auditor:** Claude Fable 5.2 (Adversarial Auditor 2)
- **Date:** 2026-11-25
- **Commit audited:** `9db80e7` (working tree as found)
- **Scope:** `spectral_scaling.py`, `circuits.py`, `circuit_search.py`, `run_cycle1_head_to_head.py`, and the three result files `circuit_search_cycle1.json`, `cycle1_head_to_head.json`, `cycle1_tier2_lm.json`.
- **Method:** I read the code, derived error bounds, and ran the project code with `.venv/bin/python`. Most runs used the CPU path. Three short checks used the RTX 5080 (marked "CUDA"). I changed no project file. My scripts were temporary; the short reproducers are in Appendix A.

Symbols used below:

| Symbol | Meaning |
|---|---|
| `W_h` | The Frobenius-normalized input after the cast to FP16 or BF16 |
| `R`, `R_h`, `R2` | `R = W_h W_hᵀ` in FP32; `R_h` = `R` cast to half precision; `R2 = R_h R_hᵀ` in FP32 |
| `U` | The upper bound that the code computes: `sqrt(min(tr R, ‖R‖_F, ‖R2‖_F^½))` |
| `σ̂` | The scale the circuit divides by (`1.005·U` for `schatten8`) |
| `x` | `σ_max(W_h) / σ̂`. The circuit is designed for `x ≤ 1` and tolerates `x ≤ basin` |
| `u` | Unit roundoff: `2⁻¹¹ = 0.049%` (FP16), `2⁻⁸ = 0.39%` (BF16) |

---

## 1. Executive Verdict: **FAIL**

Two claims that Cycle 2 depends on are false, and I reproduced both with the project's own code.

1. **`scaling="schatten8"` is not a rigorous bound in FP16, and it is not divergence-free.** A plain "identity plus constant" matrix of shape 4096×4096 makes `run_circuit(..., dtype=float16, schatten8=True)` return `Inf`. This happens on CPU and on the RTX 5080 (cuBLAS and Triton). On a family of 76 such matrices, `S1-k3+4-lo0.005` returned non-finite output for 8, and `S0-k6-lo0.05` returned a top singular value above 1.25 for 33 (maximum 5.44). `PS3_55` is finite on the same inputs.
2. **`scaling="spectral"` still diverges at step 11 with the current code.** The non-finite fallback and the `deferred_refresh` change do not touch the failing path. The same failure appears with `steps=2`, `probes=1`, and `defer_escalation=False`. `scaling="spectral"` is still the default of `SpectralPrescaledPolar`.

A third claim is also false: four of the six circuits labelled BF16-safe exceed the 1.25 cap on an exact rank-one matrix, with a safe scale (`x ≤ 1`).

What passed: the FLOP formulas are exact, the BF16 Schatten-8 bound stayed inside the 1.005 margin in every test, and the three failures above each have a cheap fix that I validated (Section 4).

---

## 2. Mathematical Verification and Constructed Counterexamples

### 2.1 Question 2 — the `schatten8` bound

#### What the bound proves, and what it does not

In exact arithmetic, `tr R ≥ λ_max(R)` and `‖R‖_F ≥ λ_max(R)`. Both terms bound the matrix the circuit uses. The third term is different:

```
‖R2‖_F^½ = ‖R_h²‖_F^½ ≥ λ_max(R_h)        (not λ_max(R))
```

`R_h` is `R` after rounding to half precision. So the code bounds the wrong matrix by the amount

```
λ_max(R) − λ_max(R_h) ≤ v₁ᵀ (R − R_h) v₁ ≤ Σ_ij |E_ij| |v₁,i| |v₁,j|
```

where `v₁` is the top eigenvector and `E = R − R_h`. The cast of `W` to half precision does not matter here, because everything after it uses `W_h`. The cast of `R` does matter.

#### BF16 with the 1.005 margin: holds

BF16 has no subnormal problem in this range, so `|E_ij| ≤ 2⁻⁸ |R_ij|`.

- The value 0.78% is the BF16 spacing (`2⁻⁷`). The largest rounding error is half of that, `2⁻⁸ = 0.39%`. The bound is on `λ = σ²`, so the effect on `σ` is halved again: **0.195%**. This is below the 0.5% margin.
- For a rank-one `R`, the shortfall is at most `u` in `λ`. Worst case `σ_max/U = sqrt(1 + 2⁻⁸) = 1.00195`.
- General first-order bound, with `x = ‖R − λ₁v₁v₁ᵀ‖_F / λ₁`: shortfall minus Schatten-8 slack is at most `u(1 + x) − x⁴ / (4(n−1))`. The maximum over `x` is `u + 0.75·u^(4/3)·(n−1)^(1/3)`. This stays below the margin `1.005² − 1 = 1.0025%` for **n ≤ about 2350**. For larger `n` this proof does not close, but I could not build a counterexample. My best construction needs `n` of about 2·10⁴.

| BF16 test (real code path) | Worst `σ_max(W_h) / U` | Margin 1.005 held? |
|---|---|---|
| 1,120 random and structured matrices, n = 64…512 (rank-1, spike + bulk, power law, geometric, Gaussian) | 1.00020 | Yes |
| Rank-one, two-level scan, n = 256, 801 points | 1.00098 | Yes |
| "Identity plus constant" lattice family, n = 1024…16384 (scalar model) | 1.00191 | Yes |
| Rank-one, two-level, n = 4096, **CPU path** | 1.00484 | Yes, by 0.016% |

The last row is close because the CPU FP32 `norm` is inaccurate (see below), not because of BF16 rounding.

#### FP16: the bound fails, and the circuit diverges

FP16 values below `2⁻¹⁴ = 6.1·10⁻⁵` are subnormal. Their absolute rounding error is up to `2⁻²⁵`, whatever their size. After Frobenius normalization `tr R = 1`, so off-diagonal entries of `R` are about `λ₁/n`. At `n = 4096` they are subnormal whenever `σ_max / ‖G‖_F < 0.5`, which is almost every matrix.

For a subnormal entry `|E_ij| ≤ 2⁻²⁵`, so the relative shortfall is at most `2⁻²⁵·n / λ₁`. For a spike over an isotropic bulk (bulk eigenvalue `r·λ₁`, trace 1), shortfall minus slack is about `0.75·2⁻²⁵·n²·r` at the best `r = (2⁻²⁵ n)^(1/3)`:

| n | Predicted excess in `λ` | Measured `σ_max(W_h)/U`, real code | Measured `σ_max(W_h)/σ̂` |
|---|---|---|---|
| 2048 | 0.37% (inside margin) | 1.00139 | 0.99640 |
| 4096 | 1.9% | 1.00567 | **1.00067** |
| 8192 | 9.4% | 1.03624 | **1.03108** |
| 16384 | about 47% | 1.20286 | **1.19687** (outside every basin) |

**Counterexample family.** `G = p·I + c·11ᵀ`, singular values `(s₁, p, …, p)`. Then `W_h = (d − c′)·I + c′·11ᵀ` with two FP16 values, so `σ_max(W_h) = d + (n−1)c′` exactly. `R` has one off-diagonal value, between 1.4 and 22 FP16 subnormal steps (22.455 at `n = 4096`, 4.403 at 8192, 1.423 at 16384). Every off-diagonal entry rounds the same way, so the error adds up over all `n²` entries. At `n = 8192` the off-diagonal value is 4.403 steps and rounds to 4.

**The bound is the smaller problem.** The same rounding makes `R2` inconsistent with `R` inside the first operation, `B₁ = p²c₂²·R_h² + 2p·w_I·c₂·R + w_I²·I`. At `n = 4096`, `λ₁(R_h)/λ₁(R) = 0.9806`. For `S1-k3+4-lo0.005` this moves the first segment's output on the top direction from 0.411 to 1.076. The second segment has a basin of 1.019, so the output is 2.74 by my scalar replay and 2.84 in the real run. In that run `x` was 1.00067, so the scale was almost exact. The failure comes from the FP16 cast of the unscaled `R`.

CUDA result, 76 matrices from this family with `s₂/s₁` from 0.15 to 0.30, `n = 4096`, FP16, `schatten8` (identical for cuBLAS and Triton SYRK):

| Circuit | `σ_max/σ̂ > 1` | Top gain > 1.25 | Non-finite output | Max top gain |
|---|---|---|---|---|
| `S1-k3+4-lo0.005` | 8 / 76 | 12 / 76 | **8 / 76** | Inf |
| `S0-k6-lo0.05` | 8 / 76 | **33 / 76** | 0 / 76 | 5.44 |
| `S1-k4+5-lo0.002` | 8 / 76 | 2 / 76 | **2 / 76** | Inf |

Appendix A, R1 is a five-line reproducer (`s₂/s₁ = 0.246`). The Tier-1 divergence limit is 2.0, so these count as divergence.

How likely is this in training? The family needs a top singular vector with equal-magnitude entries and a bulk that does not dither the rounding. A noisy bulk removes the coherence, so real momenta are unlikely to hit it. I still rate it critical, because the notes and the Cycle 2 plan state `σ_max/σ̂ ≤ 1.0` as a fact, and the input is a simple matrix at a benchmarked shape.

#### The CPU path is not an exact emulation for the bound

`torch.norm` on a CPU FP32 tensor has a large summation error for big matrices:

| Device | n | FP32 `‖X‖_F` / exact, constant matrices | Gaussian |
|---|---|---|---|
| CPU | 1024 | 0.99913 … 1.00076 | 0.99999 |
| CPU | 4096 | **0.98600 … 1.01414** | 0.99935 |
| CUDA | 1024, 4096 | 1.000000 | 1.000000 |

On CPU at `n = 4096`, a rank-one two-level matrix (`u = (1,…,1, 1.013,…,1.013)`) gives `‖R‖_F` 1.08% low and `σ_max/σ̂ = 1.00045 > 1` in FP16. The same error shifts `fro` and therefore `W_h`. The claim "CPU path is an exact emulation" holds for the matrix products. It does not hold for the three norm reductions.

#### Overflow of `upper * 1.005`, `c2`, and FP16 `L_j`

No overflow occurs inside the basin:

- `U ≥ σ_max(W_h) ≥ ‖W_h‖_F / √n`, so `scale ≤ √n` and `c2² ≤ n²`. These are far below FP32 limits.
- Each `L_j` is normalized to at most 1 on `[0, 1.15]`. The largest `|q|` is 59.9 (`S0-k7-lo0.02`), so `Q·scale ≤ 59.9·√n = 3834` at `n = 4096`. This is far below 65504.
- Zero, single-entry, single-row, rank-2 and mixed-batch inputs all returned finite output with `schatten8` and with `spectral`.

Overflow does occur in two other cases:

- **Outside the basin.** `L_j` grows like `x^(2^(j+1))`. The second segment maps 1.05 to between 2.2 and 166, and 1.10 to between 21 and 2·10⁵. This is how a bound failure becomes `Inf`.
- **Tiny inputs.** FP32 `norm` underflows when the squares underflow. Then `fro = 0`, the clamp gives `1e-30`, and `W = G / 1e-30` overflows FP16.

| `‖G‖_F` (true) | FP32 `fro` | `schatten8` | `frobenius` | `spectral` | `PS2_55` |
|---|---|---|---|---|---|
| 9.1·10⁻²² | 3.2·10⁻²² (wrong by 2.9×) | finite | finite | finite | 0 |
| 9.1·10⁻²⁴ | 0 | **8192 NaN** | **8192 NaN** | **8192 NaN** | 0 |
| 9.1·10⁻²⁹ | 0 | finite | **8192 NaN** | finite | 0 |
| 9.1·10⁻³⁷ | 0 | 2.4·10⁻⁵ (garbage) | 2.4·10⁻⁵ | 2.4·10⁻⁵ | 0 |
| ≥ 9·10¹⁹ | Inf | 0 | 0 | 0 | 0 |

I tested only these scales, so the edges of the NaN window are not resolved. A layer whose gradient becomes exactly zero has momentum that decays by 0.95 per step, so it must pass through this window. The baselines return 0 there because they use `norm + 1e-7`.

### 2.2 Question 1 — the step-11 failure of `scaling="spectral"`

#### Reproduction with the current code

I replayed the Tier-2 harness on CPU (FP32 forward pass, same model, seed 123, lr 0.02, 5 warm-up steps on the same polar object as the harness does, `S1-k3+4-lo0.005`, FP16). I logged the true `σ_max` of each normalized matrix.

| Estimator setting | First step outside the basin | Matrix | `σ_max/σ̂` | Triggers fired | `U/σ_max` at that call |
|---|---|---|---|---|---|
| Default (`steps=1`, deferred) | **11** | `[6,1536,512]` member 1 | 1.2295 | none | 1.016 |
| `steps=2` | 7 | `[12,1344,512]` member 4 | 1.1747 | none | 1.009 |
| `probes=1` | 24, then non-finite at 29 | `[6,512,512]` member 1 | 1.1621, 1.1783 | none | 1.058, 1.021 |
| `defer_escalation=False` | 7 | `[12,1344,512]`, six members | 1.196 … 1.309 | none | 1.016 … 1.036 |
| Clamp `σ̂ ≥ U/1.10` (my test) | none in 60 steps | — | max 1.088 | — | — |

The default run fails at the same step and on the same matrix as the recorded failure (`0.qkv.weight[1]`, step 11). My ratio is 1.2295 against the recorded 1.191; the forward pass differs (FP32 on CPU, BF16 autocast on CUDA).

#### Root cause: it is one silent step, not slow drift

History of the failing matrix:

| Run step | `σ_p/σ_max` | `U/σ_max` | `σ_max/σ̂` | Path |
|---|---|---|---|---|
| 7 | 0.644 | 1.032 | 0.969 | jump, misaligned, ratio → used `U` |
| 8 | 1.000 | 1.030 | 0.980 | deferred Krylov refresh |
| 9 | 0.921 | 1.037 | 0.964 | no trigger |
| 10 | 0.918 | 1.057 | 0.955 | no trigger |
| 11 | **0.713** | 1.016 | **1.2295** | **no trigger** |

At step 11 the power estimate fell to 71% of `σ_max` in one step. All four triggers stayed silent:

- `ratio_growth`: `U/σ_p` went from 1.152 to 1.425. This is growth of 23.7%, under the 25% tolerance.
- `jump`, `misaligned`, `lower_bound`: under their tolerances.

The trigger design cannot give safety, by arithmetic:

- With `γ = 1.141` and basin 1.16, the circuit leaves its basin when `σ_p < σ_max / (1.141·1.16) = 0.7555·σ_max`.
- `ratio_growth` stays silent while `σ_p` keeps at least `1/1.25 = 0.80` of its previous relative value. One silent step from 0.94 gives 0.752. Two silent steps from 1.0 give 0.64. Both are below 0.7555.
- Every trigger compares with the previous step. `sigma_prev` and `ratio_prev` are overwritten on every call, so the reference moves with the error.

**The estimator already had the proof that it was unsafe.** At the failing call `U/σ̂ = 1.016 × 1.2295 = 1.249 > 1.16`. `SpectralEstimate.certified_within(basin)` computes exactly this test, but nothing uses it. In all four failing runs `U` was 0.9% to 5.8% above the truth.

#### The calibration does not match deployment

`γ` comes from 45 harvested pairs. In each pair the warm vector comes from a converged Krylov solve on the previous matrix. The worst ratio was 1.1186. In the replay, the step directly after a Krylov refresh gave `σ_p/σ_max` of 0.26, 0.53, 0.55 and 0.64 (run steps 1, 3, 5, 7). The harvested phases do not include the first ten steps, where the top direction of the momentum turns fastest.

#### The warm estimator gives almost no gain over the free bound

- Harvested pairs: `σ̂/σ_max` median 1.050 (spectral) against `U/σ_max` median 1.079. In 30 of 45 pairs `σ̂ = U`. The mean gain in scale is 1.7%.
- Tier-2 replay: `U/σ_max` median 1.055 to 1.068, 90th percentile 1.10 to 1.11.
- In the 11 steps of the default run, 74.5% of calls were escalated or deferred. Each one used `U`, or ran a batch Krylov solve with FP64 QR. The "one matvec" cost does not describe this phase.

If the clamp `σ̂ ≥ U/u_safe` is added, the estimator can improve the scale by at most `u_safe / 1.005`. The Track 1 hypothesis (warm power iteration gives a tight, safe scale) is not supported. The Schatten-8 bound from `R2` does the work.

#### Remaining blind spots in the fallback and the deferred refresh

| # | Blind spot | Evidence |
|---|---|---|
| a | The two changes do not act on the failing path. The failing calls have no trigger, so nothing is deferred and no Krylov solve runs. | Table above |
| b | `block_krylov` tests `isfinite(T_sym).all()` over the whole batch. One bad member sends every member to one power step from column 0. At a cold start column 0 is random. The caller still marks the result `escalated` and applies `γ = 1.02`. | Healthy 256×1024 Gram: `σ_max/σ̂ = 0.983` alone, **1.38 to 1.40** when batched with one NaN or Inf member (Appendix A, R4) |
| c | The `LinAlgError` branch has the same contract problem: weak estimate, strong label. In the deferred branch a fallback result gets `γ = 1.02` and not 1.141. | Code reading |
| d | `_pending` is not cleared when the batch shape changes. | `RuntimeError: The size of tensor a (3) must match the size of tensor b (2)` after a shape change on one estimator |
| e | Escalation Krylov (warm vector + 4 probes, depth 4) with `γ = 1.02` can miss a spike orthogonal to the warm vector. Bulk uniform on `[0,1]`, spike ratio 1.03 to 1.4, 8 seeds. | Worst `σ_max/σ̂`: 1.039 (n=1024), 1.098 (n=4096), **1.175 (n=16384)**, **1.186 (n=65536)** |
| f | On CUDA the flags are read only if the copy event has completed. If it has not, the next call takes the plain warm path with references that already contain the bad values. | Code reading; I did not test this race |
| g | The Tier-2 harness uses one polar object for the warm-up model and the measured model. The first measured step starts from the warm vector of another trajectory. | `σ_p/σ_max = 0.26` at run step 1 (the triggers caught it) |
| h | `SpectralPrescaledPolar.__call__` defaults to `key=None`, so all parameters share one state unless the caller passes keys. | Code reading |

### 2.3 Question 3 — FLOPs, memory, and tail damping

#### FLOPs: correct

I counted the products per call by instrumenting `_syrk` and `_mm`:

| Circuit | SYRK(W) | Squares (with `R2`) | `Q@W` | Formula |
|---|---|---|---|---|
| `S0-k6-lo0.05` | 1 | 6 | 1 | (3α + 6)n³ ✓ |
| `S1-k3+4-lo0.005` | 2 | 7 | 2 | (6α + 7)n³ ✓ |
| `S1-k4+5-lo0.002` | 2 | 9 | 2 | (6α + 9)n³ ✓ |
| `PS2_55`, `PS3_55` | 2 | 10 | 2 | (6α + 10)n³ ✓ |

All eight circuits match. `R2` is not counted twice. `squarings = len(ops)` counts operation 0, and `R2` is the single square that operation 0 needs. The `j = 0` expansion does not save a square. Its only benefit is that `R2` exists before the scale is known, so the Schatten-8 term adds no `n³` work. That part of the claim is correct. Its cost is the FP16 cast of the unscaled `R` (Section 2.1).

#### Memory: under-counted by two FP32 Gram buffers

`run_circuit` keeps `R` and `R2` alive next to the scaled copies `R*c2` and `R2*c2²`. Measured on CUDA with `max_memory_allocated`:

| Shape | Circuit | Measured peak | `_circuit_cost` | Ratio |
|---|---|---|---|---|
| 1024×1024 | `S0-k6-lo0.05` | 44 MiB | 36 MiB | 1.22 |
| 1024×1024 | `S1-k3+4-lo0.005` | 36 MiB | 28 MiB | 1.29 |
| 1024×1024 | `S1-k4+5-lo0.002` | 40 MiB | 32 MiB | 1.25 |
| 1024×4096 | `S0-k6-lo0.05` | 54 MiB | 48 MiB | 1.13 |

The true count is `max_k + 4` FP32 `n²` buffers, not `max_k + 2`. For comparison, `PS2_55` holds 2 FP16 buffers plus one FP32 scratch, about `8n²` bytes against about `40n²` bytes for `S0-k6`.

#### The FLOP model does not predict latency

From `cycle1_head_to_head.json` (GPU median, Triton, FP16):

| Shape | Method | Symmetric units | ms | ms per unit | Relative to PS3 |
|---|---|---|---|---|---|
| 4096×4096 | `ps3_55_blackwell` | 16 | 10.15 | 0.634 | 1.0× |
| 4096×4096 | `S0-k6-lo0.05` schatten8 | 9 | 13.51 | 1.501 | 2.4× |
| 4096×4096 | `S1-k3+4-lo0.005` schatten8 | 13 | 15.93 | 1.226 | 1.9× |
| 4096×4096 | `S1-k4+5-lo0.002` schatten8 | 15 | 19.78 | 1.319 | 2.1× |
| 1024×4096 | `ps3_55_blackwell` | 34 | 0.510 | 0.0150 | 1.0× |
| 1024×4096 | `S0-k6-lo0.05` schatten8 | 18 | 0.884 | 0.0491 | 3.3× |
| 1024×4096 | `S1-k3+4-lo0.005` schatten8 | 31 | 1.118 | 0.0361 | 2.4× |

A circuit costs about 2 to 3 times more per counted FLOP. The model leaves out the elementwise work on FP32 buffers. By code inspection one segment with `k` squares makes about `10 + k + (k−1)(k+4)/2` full `n²` passes: 41 for `k = 6`, 46 for `S1-k3+4`, 59 for `S1-k4+5`. No circuit beats `ps3_55_blackwell` at equal coverage of small singular values.

#### Tail damping

Exact-arithmetic gain `f(s)` for `s = σ/σ̂`:

| Circuit | f(1e-4) | f(1e-3) | f(2e-3) | f(5e-3) | f(1e-2) | f(5e-2) | `f′(0)` | `s` where `f = 0.5` |
|---|---|---|---|---|---|---|---|---|
| `S0-k6-lo0.05` | 0.0025 | 0.025 | 0.050 | 0.124 | 0.246 | 0.949 | 24.9 | 2.11·10⁻² |
| `S1-k3+4-lo0.005` | 0.021 | 0.212 | 0.415 | 0.879 | 1.029 | 0.892 | 213.9 | 2.45·10⁻³ |
| `S1-k4+5-lo0.002` | 0.060 | 0.560 | 0.932 | 0.910 | 1.060 | 0.941 | 596.1 | 8.80·10⁻⁴ |
| `polar_express_5` | 0.080 | 0.729 | 1.097 | 0.879 | 0.906 | 0.971 | 803.9 | 6.48·10⁻⁴ |
| `PS2_55` | 0.105 | 0.872 | 1.037 | 1.095 | 0.985 | 1.027 | 1055.8 | 4.97·10⁻⁴ |
| `PS3_55` | 0.113 | 0.898 | 0.997 | 1.073 | 1.073 | 1.039 | 1130.7 | 4.65·10⁻⁴ |

All circuits are non-negative on `[0, lo]`, and all but `S1-k3+5-lo0.01` are monotone there.

- **`S0-k6-lo0.05`.** Below `0.05·σ̂` the map is nearly linear with slope 24.9. A singular value at `0.01·σ̂` leaves at 0.25. The headline error 0.0515 holds only on `[0.05, 1]`. On the harvested momenta the median relative error is 0.50 (oracle scale 0.48) against 0.062 for `PS3_55`. The comment in `circuit_search.py` states that 16% of the singular values of the median harvested matrix are below `0.01·σ_max`. This circuit does not do the same job as `PS3_55`, and its error must not be placed next to the baseline errors without that statement.
- **`S1-k4+5-lo0.002`.** Its slope at zero is 53% of the `PS3_55` slope. In units of `σ/σ̂` it damps the tail more. The two methods use different scales, so the fair unit is `σ/σ_max`. With the median harvested scales (`σ̂ = 1.084·σ_max` for `schatten8`, `σ̂ = 3.19·σ_max` for Frobenius) the half-gain point is `9.5·10⁻⁴·σ_max` for `S1-k4+5` and `1.48·10⁻³·σ_max` for `PS3_55`. So `S1-k4+5` lifts the tail slightly more on low-stable-rank matrices and less when the two scales are close. This agrees with the harvested errors (median 0.063 against 0.062, max 0.275 against 0.321). It is parity for one square less: 6% fewer FLOPs at `α = 1`, 3% at `α = 4`, and about twice the latency.

### 2.4 Question 4 — `RHO = U_FP16 / 16` and why `S1-k4+5-lo0.002` fails in BF16

#### What `RHO` does

The fit minimizes `E + ρ·S`. `E` is the minimax error. `S = Σ_j max|∂f/∂L_j|·max|L_j|` is the first-order output shift when every `L_j` moves by its own maximum times a relative error of 1. With rounding at unit roundoff `u`, the worst output shift is `u·S`. `ρ = 2⁻¹¹/16 = 3.05·10⁻⁵` is tuned for FP16 only, on Gaussian-basis test matrices of size 96. BF16 has 8 times the unit roundoff, and no term in the fit knows this.

I recomputed `S` for the shipped circuits (per segment, exact arithmetic, including the cast of `Q`):

| Circuit | Segment | `S` total | Worst shift FP16 (`u·S`) | Worst shift BF16 | Segment-2 headroom |
|---|---|---|---|---|---|
| `S1-k2+4-lo0.01` | 1 / 2 | 12.5 / 56.5 | 0.006 / 0.028 | 0.049 / 0.221 | 2.0% |
| `S1-k3+4-lo0.005` | 1 / 2 | 34.8 / 72.5 | 0.017 / 0.035 | 0.136 / 0.283 | 1.9% |
| `S1-k3+5-lo0.01` | 1 / 2 | 34.0 / 101.9 | 0.017 / 0.050 | 0.133 / 0.398 | 2.3% |
| `S1-k4+5-lo0.002` | 1 / 2 | 84.0 / 194.8 | 0.041 / 0.095 | **0.328 / 0.761** | 2.0% |
| `S0-k6-lo0.05` | 1 | 241.9 | 0.118 | 0.945 | — |
| `S0-k7-lo0.02` | 1 | 431.9 | 0.211 | 1.687 | — |

#### Why FP16 passes and BF16 fails

1. **Segment 2 has 2% headroom.** `main()` fits it with `basin = 1.02`. The measured second-segment basins are 1.019 to 1.023. Segment 1 is scaled so that its exact maximum on `[0, 1.15]` is 1.
2. **Rounding moves the output of segment 1 above 1.** In the scalar rounding model the maximum of segment 1 of `S1-k4+5-lo0.002` is 1.0010 in FP16 and **1.0312** in BF16. The second-segment basin is 1.0205. So in BF16 some inputs inside `[0, 1]` leave the cap, and `stability_basin` returns 0 by its convention. The scalar BF16 error is 1.06.
3. **Depth.** Segment 1 with `k = 4` has `S = 84`. With `k = 3` it has 34.8. The same model gives 1.0156 for `S1-k3+4-lo0.005` in BF16. That is 0.3% inside its 1.0188 basin.

So the BF16 label rests on 0.3% of slack in a rounding model, and that model is optimistic (next part).

#### The scalar rounding model is not conservative

`ChainSegment.scalar(rnd=…)` rounds each eigenvalue relative to its own size. The matrix code rounds entries. For `L_j = w_I·I + β·vvᵀ` the eigenvalue error is about `u·(|w_I| + |β|)`. It does not shrink when `w_I` and `β` cancel. For a random basis the entry errors average out, which is why the code comment reports a 35-fold overestimate at `n = 96`. For a coherent matrix they do not average out.

Test: exact rank-one sign matrix, explicit `sigma`, 1200 values of `x` in `[0.002, 1.2]`, 64×128, real matrix code:

| Circuit | Label | BF16 scalar basin (JSON) | BF16 matrix: max output for `x ≤ 1` | FP16 matrix: max output for `x ≤ 1` |
|---|---|---|---|---|
| `S0-k4-lo0.05` | BF16-safe | 1.152 | 1.204 | 1.195 |
| `S0-k6-lo0.05` | BF16-safe | 1.121 | **1.364** | 1.074 |
| `S1-k2+4-lo0.01` | BF16-safe | 1.207 | 1.171 | 1.150 |
| `S1-k3+4-lo0.01` | BF16-safe | 1.160 | **1.635** | 1.074 |
| `S1-k3+4-lo0.005` | BF16-safe | 1.160 | **1.287** (2.313 at 256×256) | 1.142 |
| `S1-k3+5-lo0.01` | BF16-safe | 1.160 | **1.536** | 1.066 |
| `S1-k4+5-lo0.002` | FP16 only | 0.000 | 7.778 | 1.153 |
| `S0-k7-lo0.02` | FP16 only | 1.027 | 1.414 | 1.094 |

Four of the six BF16-safe circuits exceed the cap with a scale that overestimates `σ_max`. The windows are narrow (0.3% to 1.4% of `x` in `[0.05, 1]`, near `x = 0.54` and `x = 0.90`). Appendix A, R2 gives one point: `S1-k3+4-lo0.005`, 256×256, `σ̂ = 1.81·σ_max`, output `σ_max = 2.313`. On the `schatten8` path, rank-one and near-rank-one BF16 inputs stayed below 1.18, because `x` is then close to 0.995.

In FP16 the matrix results agree with the scalar basins on this test. The FP16 model fails only through the subnormal mechanism of Section 2.1.

Two more points on `circuit_search.py`:

- The comment on `RHO` says that `u/16` keeps FP16 and BF16 emulated errors within `1e-2` of the exact error "in the k=6, lo=0.02 sweep". The result file gives `S0-k6-lo0.02`: exact 0.2398, FP16 scalar 0.2471, BF16 scalar 0.5156, BF16 basin 0. The BF16 part of the comment does not hold in the recorded metric.
- `GramCircuit.basin` stores the design target 1.15. The measured exact basins are 1.1311 (`S0-k6-lo0.05`) and 1.1249 (`S0-k7-lo0.02`). The cause is that the LP cap is active at its grid points and `f` exceeds 1.25 by less than `5·10⁻⁵` between them (interval width `1.2·10⁻⁴`). The effect is small, but the stored value is not a measurement, and the selection filter accepts anything from 1.12.

---

## 3. Severity-Ranked Findings

### CRITICAL

**C1. FP16 `schatten8` returns `Inf` on a simple 4096×4096 matrix (CPU and CUDA).**
`R` is cast to FP16 at trace 1, where its off-diagonal entries are subnormal. `R2 = R_h²` is then inconsistent with `R` in operation 0, and it bounds the wrong matrix. Evidence: Section 2.1, Appendix A R1. The statements "rigorous upper bound" and "`true_sigma/upper ≤ 1.0`" in `circuits.py`, `cycle1_tier2_lm.json`, `RESEARCH_STATE.json` and `BACKLOG.md` (T1-04) are false for FP16 at `n ≥ 4096`.

**C2. `scaling="spectral"` diverges in real training with the current code and is still the class default.**
Reproduced at step 11, and at steps 7 to 29 under three other settings. No trigger fires. The rigorous bound `U` proves the estimate unsafe at each failing call, and the code ignores it. Evidence: Section 2.2.

### HIGH

**H1. The "BF16-safe" label is not valid.** Four of six labelled circuits exceed the 1.25 cap on an exact rank-one matrix with a safe scale. The label comes from a scalar rounding model that is optimistic for coherent matrices. Tier-1 never ran BF16 circuits on its rank-one case. Evidence: Section 2.4, Appendix A R2.

**H2. Two-segment circuits have 2% headroom in segment 2.** Any error of segment 1 above about 1.02 is amplified (1.05 → 2.2 to 166). This is the amplifier in C1 and H1. It makes the Cycle 2 plan "zero-overshoot `u = 1.005`" unsafe as written: rounding inside the chain moves singular values above 1 even when the input scale is exact.

**H3. Tiny inputs return NaN.** `‖G‖_F = 9·10⁻²⁴` gives 8192 NaN in all three scaling modes, and `9·10⁻²⁹` gives NaN with Frobenius scaling. The baselines return 0. Evidence: Section 2.1, Appendix A R3.

### MEDIUM

**M1. The 1.005 margin has no derivation and no budget.** It is partly used by BF16 rank-one rounding (0.195%), fully used by CPU norm error (up to 0.7% at `n = 4096`), and exceeded by FP16 subnormal rounding.

**M2. `block_krylov` fallbacks have the wrong contract.** Whole-batch fallback, random first column at cold start, and the result is still marked `escalated` with `γ = 1.02`. A healthy batch member got `σ_max/σ̂ = 1.38 to 1.40`. Evidence: Appendix A R4.

**M3. `KRYLOV_GAMMA = 1.02` is not safe for large `n`.** A hidden spike gives `σ_max/σ̂` of 1.098 at `n = 4096` and 1.175 at `n = 16384` after escalation.

**M4. `γ` calibration does not represent deployment.** 45 pairs, converged warm vector, no early steps. Deployment saw `σ_p/σ_max` down to 0.26.

**M5. Scalar-error headlines compare different jobs.** `S0-k6-lo0.05` "error 0.0515" against `PS3_55` 0.1017 hides a median error of 0.50 on harvested momenta.

**M6. The FLOP model does not predict latency.** Circuits cost 1.9 to 3.3 times more per counted FLOP than `ps3_55_blackwell`.

### LOW

**L1.** `_circuit_cost` and the module docstring under-count memory by two FP32 `n²` buffers (measured 13% to 29% above the formula).
**L2.** "CPU path is an exact emulation" is false for `norm` reductions (CPU error up to 1.4% at `n = 4096`).
**L3.** `_pending` is not cleared on a batch-shape change; reproduced `RuntimeError`.
**L4.** `GramCircuit.basin` stores the target 1.15, not the measured 1.131 and 1.125.
**L5.** The `RHO` comment about BF16 is contradicted by the result file.
**L6.** Circuits return a full-size update for inputs down to `‖G‖_F ≈ 1e-22`; baselines return about 0 below `1e-7`. This changes behaviour for dead layers and affects Tier-2 comparability.
**L7.** `key=None` default and shared polar object across harness warm-up and run leak estimator state.

---

## 4. Concrete Remediation Actions Required Before Cycle 2

Actions 1 to 3 are blockers. I validated the first three with the project code.

1. **Fix the FP16 cast of Gram buffers (C1).** Multiply by a power of two before every cast, and undo it after the product: `e = 13 − ceil(log2(max|L|))`, `R2 = syrk((L·2^e).to(fp16))·2^(−2e)`. A power of two changes no mantissa. **Validated on CUDA:** on the 76-matrix family all three circuits went from up to 33 cap violations and 8 non-finite outputs to 0 and 0, and `σ_max/σ̂` became 0.99502 for every input.
2. **Make the bound rigorous in floating point (C1, M1).** Add the rounding term: `λ_max(R) ≤ ‖R2‖_F^½ + ‖R − R_h‖_F`. It costs one `n²` pass. Compute the three norms in FP64 or with a scaled sum. Replace 1.005 with a derived margin. Add the family `p·I + c·11ᵀ` at `n = 4096` and `8192` to Tier-1.
3. **Enforce the certificate or remove the warm estimator (C2).** Add `sigma = max(sigma, upper / u_safe)` in `WarmSpectralEstimator.estimate`, with `u_safe` below the rounded basin. **Validated:** 60 Tier-2 steps, 0 calls outside the basin, maximum `σ_max/σ̂ = 1.088` for `u_safe = 1.10`. Then decide whether the estimator stays: its mean gain over `schatten8` is 1.7% on the harvested pairs. Until then, change the default of `SpectralPrescaledPolar` to `"schatten8"` and mark the `*_steady_*` benchmark rows as unsafe.
4. **Keep a basin in Cycle 2 (H2).** Do not design for `u = 1.005`. The basin of every segment must cover the bound error plus `u_dtype·S` of all earlier rounding. Fit segment 2 with a basin of at least `1 + u·S_seg1`: about 1.05 for FP16 with the current segment 1, and 1.15 or more for BF16, or lower `S`.
5. **Replace the scalar rounding model (H1).** Use an absolute-error model (`L_j ± u·(|w_I| + Σ|w_i|·|B_i|)`) for the basin check. Add a matrix test to selection: exact rank-one sign matrices swept over `x` in `[0, basin]`, both dtypes, at two sizes. Re-issue the BF16 list from that test. Today only `S0-k4-lo0.05` and `S1-k2+4-lo0.01` pass it.
6. **Make `RHO` depend on the dtype (Q4).** Use `ρ ∝ u_dtype`, and add the hard constraint `u_dtype·ΣS_j ≤ headroom` to the LP. Fix the comment.
7. **Guard the normalization (H3, L6).** Compute `fro` after dividing by `amax`, or in FP64. Return zeros for zero or non-finite `fro`. Decide on a noise floor that matches the baselines and document it. Add input scales `1e-20`, `1e-25`, `1e-30`, `1e-38` to Tier-1.
8. **Fix the fallback contract (M2, M3, L3, L7).** Fall back per batch member, return a mask, and use `U` for members that fell back. Clear `_pending` when the key changes. Require an explicit `key`. Raise `KRYLOV_GAMMA` or the depth for `n > 4096`.
9. **Recalibrate or drop `γ` (M4).** If the estimator stays, calibrate on full streaming runs that include steps 1 to 50, and log `σ_max/σ̂` in Tier-2.
10. **Correct the accounting (L1, M6).** Use `max_k + 4` buffers, or free `R` and `R2` after scaling. Report measured ms per `n³` unit next to every FLOP formula.
11. **Report tail coverage honestly (M5).** Give errors per `lo` band and in units of `σ/σ_max`. State the half-gain point for every method. Do not rank a `lo = 0.05` circuit against baselines that cover `10⁻³`.
12. **Store measured basins (L4).** Fit with `CAP − δ` and record `basin_u()` for FP64, FP16 and BF16 matrix tests.

---

## 5. What I Verified as Correct

- `GramCircuit.flops()` and the `PS2_55`/`PS3_55` formulas match the executed products for all eight circuits.
- `R2` is not double-counted, and Schatten-8 scaling adds no `n³` work.
- The BF16 Schatten-8 bound stayed inside the 1.005 margin in every test, and it is provable to first order for `n ≤ about 2350`.
- No overflow of `upper`, `c2`, `Q·scale` or `L_j` occurs inside the basin, also for zero and rank-deficient inputs.
- Recorded circuit errors match my recomputation to four digits.
- On all 45 harvested pairs the estimator kept `σ̂ ≥ σ_max` (minimum `σ̂/σ_max = 1.0007` in `circuit_search_cycle1.json`). The failure appears only in streaming use.

## 6. Limits of This Audit

- The Tier-2 replay ran on CPU with an FP32 forward pass. I did not rerun it on CUDA with BF16 autocast.
- I did not test the CUDA flag race (blind spot f).
- The `n = 16384` case was run for the bound only, not for the full circuit.
- For BF16 above `n ≈ 2350` I have neither a proof nor a counterexample.
- I did not audit the Triton kernels or the timing method. Those belong to other auditors.

---

## Appendix A — Reproducers

I ran every reproducer below as written. R1 to R4 run on CPU in under one minute. Expected output:

```
R1 finite: False
R2 output sigma_max: 2.313 (cap 1.25)
R3 NaN count: 8192 | PS2_55 max: 0.0
R4 sigma_max/sigma_hat of the healthy member: 1.4 | escalated: True
```

```python
# Run: PYTHONPATH=. .venv/bin/python repro.py
import numpy as np, torch
from research.polar_opt import circuits as C, spectral_scaling as SS
from research.polar_opt.baselines import polar_switch
g = torch.Generator().manual_seed(1)
circ = C.CIRCUITS["S1-k3+4-lo0.005"]

# R1: FP16 + schatten8 returns Inf on "identity plus constant" at 4096x4096 (same on CUDA, cuBLAS and Triton)
n, rs = 4096, 0.246
s1 = 1 / np.sqrt(1 + (n - 1) * rs * rs); p = rs * s1; c = (s1 - p) / n
G = torch.full((n, n), c, dtype=torch.float32); G.diagonal().add_(p)      # singular values (s1, p, ..., p)
print("R1 finite:", bool(torch.isfinite(C.run_circuit(G, circ, dtype=torch.float16, schatten8=True)).all()))

# R2: "BF16-safe" circuit, exact rank-one input, SAFE scale (sigma_hat = 1.81 sigma_max)
u = (torch.randint(2, (256,), generator=g) * 2 - 1).float(); v = (torch.randint(2, (256,), generator=g) * 2 - 1).float()
Y = C.run_circuit(u[:, None] * v[None, :], circ, dtype=torch.bfloat16, sigma=torch.tensor(256.0 / 0.5535))
print("R2 output sigma_max:", round(torch.linalg.matrix_norm(Y.double(), ord=2).item(), 3), "(cap 1.25)")

# R3: an input with ||G||_F = 9e-24 returns NaN (PS2_55 returns 0)
Gt = torch.randn(64, 128, generator=g) * 1e-25
print("R3 NaN count:", int(torch.isnan(C.run_circuit(Gt, circ, schatten8=True)).sum()), "| PS2_55 max:", polar_switch(Gt).abs().max().item())

# R4: one non-finite batch member makes a healthy member's cold-start estimate far too small
W = torch.randn(256, 1024, generator=g) / 32; A = W @ W.T; A = A / A.diagonal().sum(); B = A.clone(); B[0, 0] = float("nan")
true = torch.linalg.eigvalsh(A.double())[-1].sqrt().item()
e = SS.WarmSpectralEstimator().estimate(torch.stack([A, B]), gram=True, A2=torch.stack([A @ A, B @ B]))
print("R4 sigma_max/sigma_hat of the healthy member:", round(true / e.sigma[0].item(), 3), "| escalated:", bool(e.escalated[0]))
```

**R5 — step-11 divergence of `scaling="spectral"` (CPU, about 2 minutes).** Wrap the polar object so each call logs the true `σ_max` and `info.sigma`, then follow the harness order (5 warm-up steps and the measured run on the same polar object):

```python
import torch
from research.polar_opt.harness_tier2_lm import ModelConfig, TokenStream, _setup, train_step, configure_device
from research.polar_opt.circuits import SpectralPrescaledPolar

class Logged:
    def __init__(self):
        self.polar = SpectralPrescaledPolar("S1-k3+4-lo0.005", dtype=torch.float16, scaling="spectral"); self.step = 0
    def __call__(self, G):
        out, info = self.polar(G, key=tuple(G.shape), return_info=True)
        X = G.reshape(-1, *G.shape[-2:]).double()
        ratio = torch.linalg.matrix_norm(X, ord=2) / X.norm(dim=(-2, -1)) / info.sigma.double()
        if ratio.max() > 1.16 or not torch.isfinite(out).all():
            print("step", self.step, tuple(G.shape), "member", int(ratio.argmax()), "true/sigma_hat", round(ratio.max().item(), 4),
                  "U/true", round((info.upper.double() / (ratio * info.sigma.double())).flatten()[ratio.argmax()].item(), 3),
                  "triggers", {k: bool(v.flatten()[ratio.argmax()]) for k, v in info.triggers.items()})
        return out

torch.set_num_threads(12)
dev, cfg, m, seed, lr = configure_device("cpu"), ModelConfig(), Logged(), 123, 0.02
for offset, steps in ((1_000_000, 5), (2_000_000, 11)):
    model, opt = _setup(cfg, seed, dev, m, lr, 0.001, 0.01); stream = TokenStream(cfg, seed + offset, dev)
    for i in range(steps):
        m.step = i + 1; train_step(model, stream, opt, dev)
# Expected: step 11 (6, 1536, 512) member 1 true/sigma_hat 1.2295 U/true 1.016, all triggers False
```
