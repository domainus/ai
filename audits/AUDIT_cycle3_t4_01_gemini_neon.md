# Adversarial Audit Report — Cycle 3 Track 4 (`T4-01`: Automated muP Width-Transfer Coordinate-Check Harness)

- **Auditor:** Gemini 4 Neon (High) · **Date:** 2026-11-25
- **Branch / Audited Commits:** `exp/c3-t4-01-audit-neon` branched from `origin/exp/c3-t4-01-mup-coord-check`
  - Artifact commit: `f0a3319b7e40d894c57b6aa56cd5d63c4f44dc44` (`results/cycle3_mup_coord_check_cpu.json`)
  - Parent code commit: `d9ff3678413eba5fdaea6e2bc635c8ef23c68a7e` (`research/polar_opt/mup_coord_check.py`, `tests/test_mup_coord_check.py`)
- **Execution Mode:** Strictly read-only, **CPU-only (`CUDA_VISIBLE_DEVICES=""`)**, zero GPU usage, zero modifications to source, test, or result files.
- **Verification Executed:**
  1. `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/ryan/Documents/projects/ai/.venv/bin/pytest tests/test_mup_coord_check.py -q` → **`67 passed in 145.23s (0:02:25)`**.
  2. Standalone strict JSON + NumPy recomputation of every provenance field, OLS log-log exponent `beta`, `max_over_min_ratio`, gate verdict, weight-space `polar_mean_sq_sv` lift, and LR-transfer metric in `results/cycle3_mup_coord_check_cpu.json`.
  3. Mathematical audit of all muP, SP, AdamW, Muon, and `MuReadout` scaling laws and confounders.
  4. Circuit & precision fidelity audit against `research/polar_opt/baselines.py` and `research/polar_opt/schatten_switch_blackwell.py`, including real Nesterov momentum comparisons and singular-value spectrum inspection for `svd_exact` rank truncation.
  5. Controlled CPU sensitivity probes: `mup_adamw` across LRs `{0.001, 0.003, 0.01, 0.03}` and 6 seeds; `ADAM_EPS = 1e-8` vs `1e-12`; `num_batches = 4` vs `12`; `mup_muon/svd_exact` across `lr_muon ∈ {0.01, 0.02, 0.04}` and `lr_aux ∈ {0.003, 0.01, 0.03}`; `svd_exact` across `rtol ∈ {default, 1e-6, 0.0}`; multi-seed (`seeds=[0, 1, 2]`) and held-out validation LR transfer; and API/config edge cases.

---

## Executive Verdict: **`PASS_WITH_CAVEATS`**

### Promotion Gate Recommendation for `T4-01`

**Promote `T4-01` for the primary coordinate-check claim, with mandatory caveats on the secondary LR-transfer and `muP AdamW` stability claims (or apply the targeted remediations below before locking a publication table):**

1. **Verified & Ready to Promote:**
   - **Core muP + Muon width-transfer coordinate check:** All 4 Schatten Switch circuits (`SS3_54`, `SS3_53`, `SS3_35`, `SS3_34`), `PS3_55`, `polar_express`, and `svd_exact` pass the coordinate check cleanly (`max |beta| = 0.0780–0.0819 < 0.15`), stay within `0.0039` of exact FP64 SVD exponents across all 7 activation probes and gate steps `2, 4, 8, 12`, and remain robust (`max |beta| = 0.0618–0.1021`) across a `4×` sweep of `lr_muon` and a `10×` sweep of `lr_aux`.
   - **Negative controls & `MuReadout` correction:** `sp_adamw` (`max |beta| = 2.5988`), `sp_muon/svd_exact` (`1.5944`), and both `readout_lr_power=1` controls (`1.0629` AdamW, `1.0385` Muon) fail clearly as theoretically predicted. The implementer's correction from `readout_lr_power=1` to `readout_lr_power=0` (`MuReadout` multiplier `d_base / d` + width-independent AdamW readout LR) is mathematically exact and properly isolated via the kept control.
   - **Provenance & arithmetic integrity:** `results/cycle3_mup_coord_check_cpu.json` is clean (`git_sha == d9ff3678413eba5fdaea6e2bc635c8ef23c68a7e`, `git_dirty == false`, strict JSON without `NaN`/`Infinity`), and 100% of stored statistics reproduce independently within 6-significant-digit rounding (`max |Δbeta| = 5.31e-6`).
   - **Exact `SS3_*` & `PS3_55` circuit imports:** All `SS3_*` runs invoke `schatten_switch_reference(..., dtype=torch.float16)` unchanged from `schatten_switch_blackwell.py` (exercising both `P=1` for `w < 1024` and `P=64` for `w = 1024`), and `PS3_55` imports `_SWITCH_TABLES["PS3_55"]` from `baselines.py` with `0.17%–0.71%` relative Frobenius fidelity on real momenta.

2. **Caveated / Do NOT Claim Without Qualification:**
   - **Do NOT claim *"Muon keeps the exact best LR at 0.04 across all widths with 0.0 regret"* as a multi-seed result ([H1]).** The artifact ran LR transfer on **`seed=0` only**, where `w=128` beat `0.02` by only `0.0037` nats (`0.6924` vs `0.6961`; interpolated optimum `0.0291`). Under 3-seed averaging (`seeds=[0, 1, 2]`), `mup_muon/svd_exact`, `SS3_54`, and `SS3_34` **all flip to `best_lr = {128: 0.02, 256: 0.04, 512: 0.04}`** (`same_best_lr = False`, `interp_shift = +0.46 to +0.49` octaves, regret `0.034–0.039` nats). Furthermore, `final_loss` evaluates 5-epoch **training-batch memorization** (`0.14` train vs `4.04` held-out val at `w=512`), not out-of-sample generalization. The supportable claim is that Muon's optimum transfers smoothly across `256 -> 512` (`0.04 -> 0.04` on 3-seed mean) with low base-LR regret (`~0.035` nats vs `0.199` for `sp_muon` and `0.697` for `sp_adamw`), and `SS3_*` tracks `svd_exact` closely.
   - **Qualify the `muP AdamW` coordinate-check margin ([H2]).** `mup_adamw` passes the `< 0.15` gate at `lr=0.003` with a thin margin (`0.1380` on 3 seeds, `0.1090` on 6 seeds) due to an analytically explainable step-2 cross-batch $\Theta(1/\sqrt{d})$ sign-update finite-width residual compounded by a `seed=1` outlier (`-0.3551` on `block1_attn_out@2`). Under the same `< 0.15` gate, `mup_adamw` **fails** at `lr=0.001` (`0.1544`, 3 seeds), **fails** at its own optimal LR `0.01` on 6 seeds (`0.1585`, `block0_mlp_out@8`), and **fails** at `lr=0.03` (`0.3413`).
   - **Qualify weight-space `polar_mean_sq_sv` comparisons at `w=1024`, step 12 ([M1]).** `svd_exact`'s width-growing `rtol = max(m, n) * eps(FP32)` over-truncates 200 genuine singular values (`19.5%`) of `blocks.0.attn.q` at `w=1024`, step 12, artificially pulling the `svd_exact` lift reference down to `0.8817` (`beta = -0.0566`) and understating polynomial under-lift at `w=1024` (activation-space `beta` is unaffected: `< 0.001` shift).

---

## 1. Mathematical Audit of muP Equations & Readout Deviation

### 1.1 Parameterization Audit (`research/polar_opt/mup_coord_check.py:104-260`)

Let $d = \text{width}$, $d_0 = d_{\text{base}}$, $m = d_0 / d$, $h = \text{head\_dim}$ (fixed at 32), $k = \text{mlp\_mult}$ (default 2).

| Component | Implementation | Asymptotic Width Scaling Analysis | Status |
|---|---|---|---|
| **Embedding** (`emb.weight`, $V \times d$) | `init_std = 1/sqrt(d_base)`, `lr_mult = 1.0` | Entry RMS $\Theta(1)$ at $t=0$; AdamW step $\Delta E_{v,j} = \Theta(\eta)$ independent of $d$ → $\|\Delta x_0\|_{\text{RMS}} = \Theta(1)$. | **EXACT** |
| **Pre-RMSNorm** (`*.norm.weight`, $d$) | `init = 1.0`, `lr_mult = 1.0` | Elementwise scale (fan-in 1); $\Delta \gamma_j = \Theta(\eta)$ produces $\Theta(1)$ activation change independent of $d$. | **EXACT** |
| **Hidden 2D — AdamW** ($d_{\text{out}} \times d_{\text{in}}$) | `init_std = 1/sqrt(d_in)`, `lr_mult = d_base / d_in` | Forward init: $(W_0 u)_i = \sum_{j=1}^{d_{\text{in}}} \mathcal{N}(0, 1/d_{\text{in}}) u_j = \Theta(1)$. Aligned Adam update: $(\Delta W u)_i = \sum_{j=1}^{d_{\text{in}}} \Theta(\eta d_0 / d_{\text{in}}) \cdot \Theta(1) = \Theta(\eta d_0) = \Theta(1)$, including rectangular `gate`/`up` ($d_{\text{in}} = d$) and `down` ($d_{\text{in}} = k d$). | **EXACT** |
| **Hidden 2D — Muon** ($d_{\text{out}} \times d_{\text{in}}$) | `init_std = 1/sqrt(d_in)`, `muon_scale = sqrt(d_out / d_in)` | RMS-to-RMS operator norm is $\|\Delta W\|_{\text{RMS}\to\text{RMS}} = \sqrt{d_{\text{in}}/d_{\text{out}}}\|\Delta W\|_2$. Setting $\Delta W = -\eta \sqrt{d_{\text{out}}/d_{\text{in}}} Q$ with $\|Q\|_2 = 1$ enforces $\|\Delta W\|_{\text{RMS}\to\text{RMS}} = \eta = \Theta(1)$ exactly across arbitrary aspect ratios. *(Note [M3]: with fixed `mlp_mult`, `sqrt(d_out/d_in)` is constant across $d$.)* | **EXACT** |
| **Attention scaling** | `1 / head_dim` (muP) vs `1 / sqrt(head_dim)` (SP) | Exact TP-V muP attention logit scaling. Since `head_dim = 32` is fixed across widths, scaling does not vary with $d$ in this sweep, though `1/head_dim` prevents attention saturation under correlated updates. | **EXACT** |
| **Readout (`head.weight`, $V \times d$)** | Untied, zero-init, `logit_scale = d_base / d`, `lr_mult = (d_base / d)**readout_lr_power` (default `0.0`) | Verified analytically below. | **EXACT** |
| **SP negative controls** | `sp_adamw` (`init_std=0.02`, `lr_mult=1`, `logit_scale=1`, `attn=1/sqrt(h)`), `sp_muon/svd_exact` (`muon_scale=1`, SP init/readout/attn) | Forward variance blows up as $\Theta(d)$ in attention (`0.02 sqrt(d) * 0.02 sqrt(d)`) and $\Theta(d^{3/2})$ in gated MLP (`silu(0.02 sqrt(d)) * 0.02 sqrt(d) * 0.02 sqrt(2d)`). Matches measured `sp_muon` exponents (`block0_attn_out@2 = +1.045`, `block0_mlp_out@2 = +1.594`). | **VERIFIED** *(see [M3])* |

### 1.2 Proof of the `MuReadout` + `MuAdam` Rule (`readout_lr_power = 0` vs `1`)

Let logits be $z = m \cdot W_{\text{head}} h$ with $W_{\text{head}}^{(0)} = 0$, $h = \text{RMSNorm}(x_{\text{final}}) \in \mathbb{R}^d$ ($\|h\|_{\text{RMS}} = \Theta(1)$), and $m = d_0 / d$.

1. At step 0, $z^{(0)} = 0$, softmax $p^{(0)} = \frac{1}{V}\mathbf{1}$, loss $\mathcal{L}^{(0)} = \log V = \log 256 = 5.5452$, and the readout gradient is:
   $$g_{\text{head}}^{(0)} = \frac{m}{N}\sum_{n=1}^N \big(p_n^{(0)} - e_{y_n}\big) h_n^\top \quad \implies \quad \big|(g_{\text{head}}^{(0)})_{v,j}\big| = \Theta(m) = \Theta(1/d).$$
   (Empirically verified in Probe 1C: `grad_rms(head)` = `4.020e-3, 2.011e-3, 1.005e-3, 5.005e-4` at $d = 128, 256, 512, 1024$ — exact $1/d$ scaling.)
2. Because Adam normalizes every coordinate by $\sqrt{\hat{v}} + \epsilon$ (with $\epsilon = 10^{-12} \ll |g|$), the step-1 update is **invariant to the gradient's $m$ factor**:
   $$(\Delta W_{\text{head}}^{(1)})_{v,j} = -\eta \cdot m^p \cdot \text{sign}\big((g_{\text{head}}^{(0)})_{v,j}\big) = \Theta(\eta m^p).$$
3. Crucially, $\text{sign}((g_{\text{head}}^{(0)})_{v,j})$ correlates with $h_{n,j}$ across all $d$ coordinates, so the inner product sums coherently as $\Theta(d)$:
   $$\Delta z_v^{(1)} = m \sum_{j=1}^d (\Delta W_{\text{head}}^{(1)})_{v,j}\, h_j = m \cdot \Theta\!\big(d \cdot \eta m^p\big) = \Theta\!\big(\eta d_0 \cdot d^{-p}\big).$$
4. **Consequence:**
   - With `readout_lr_power = 0` ($p = 0$, i.e. `MuReadout` multiplier $m = d_0/d$ + width-independent Adam LR $\eta$): $\Delta z^{(1)} = \Theta(1)$ ($\beta = 0$; measured `-0.0536` at step 1, `-0.0015` at step 12). The backprop signal $\partial\mathcal{L}/\partial h = m (W_{\text{head}}^{(1)})^\top \partial\mathcal{L}/\partial z$ scales cleanly as $\Theta(1/d)$ (measured `grad_rms(b0.q)` = `4.061e-5, 2.228e-5, 9.925e-6, 5.433e-6`).
   - With `readout_lr_power = 1` ($p = 1$, applying BOTH multiplier $m$ AND readout LR $\eta m$): the $1/d$ factor is **double-counted**, forcing $\Delta z^{(1)} = \Theta(1/d)$ ($\beta = -1$; measured `-1.0535` at step 1, `-1.0629` at step 2 for AdamW, `-1.0385` at step 2 for Muon).
   - Equivalence check: Yang et al. (2022) Table 3's unmultiplied parameterization ($z = W_{\text{head}} h$, multiplier $= 1$, init $0$ or $1/d$, Adam LR $\eta / d$) is functionally identical to `MuReadout` ($z = (d_0/d) \tilde{W}_{\text{head}} h$, init $0$, Adam LR $\eta$) via $\tilde{W}_{\text{head}} = (d/d_0) W_{\text{head}}$. The implementer's correction is 100% mathematically sound.

### 1.3 Root-Cause Analysis of Confounders & Thin `muP AdamW` Margin (`0.138` vs `0.15`)

Our controlled CPU probes isolated every candidate confounder:

| Confounder / Axis Tested | Empirical Result (Widths `128–1024`, 12 Steps) | Diagnosis |
|---|---|---|
| **`ADAM_EPS = 1e-12` vs `1e-8`** (`mup_adamw`, `lr=0.003`, 3 seeds) | `1e-12`: `max \|beta\| = 0.13804`; `1e-8`: `0.13823` (`Δ = 0.00019`) | **Not a confounder.** Good hygiene because `grad_rms(b0.q)` scales as $1/d$ down to `5.43e-6` at $d=1024$; prevents `eps` damping if extended to $d \ge 4096$. |
| **`num_batches = 4` vs `12`** (`lr=0.003` AdamW / `0.02` Muon, 3 seeds) | `mup_adamw`: `0.13804` (`nb=4`) vs `0.07011` (`nb=12`); `mup_muon/svd_exact`: `0.08114` (`nb=4`) vs `0.07011` (`nb=12`); `step_delta_logits@12`: `-0.1546` → `-0.1306` | **Minor effect on 12-step coord check** (the `mup_adamw` drop is due to a new `make_tokens` RNG draw avoiding `seed=1`'s unlucky batch pair; see below). **Major effect on 20-step LR transfer** (5-epoch memorization; see [H1]). |
| **`mup_adamw` LR sensitivity** (3 seeds, `nb=4`) | `lr=0.001`: **FAIL** (`0.15437`, `b1_attn@4`); `lr=0.003`: **PASS** (`0.13804`, `b1_attn@2`); `lr=0.01`: **PASS** (`0.14160`, `b0_mlp@8`); `lr=0.03`: **FAIL** (`0.34125`, `b1_mlp@12`, all seeds `>0.51`) | **Significant confounder / fragile gate margin ([H2]).** Only `0.003` and `0.01` pass on 3 seeds, and `0.01` **fails** on 6 seeds (`0.15846`)! |
| **`mup_adamw` seed count** (`seeds 0..2` vs `0..5`) | `lr=0.003`: 3-seed `0.13804` → 6-seed `0.10900` (`PASS`); `lr=0.01`: 3-seed `0.14160` → 6-seed **`0.15846` (`FAIL`)** | **Seed-1 outlier + systematic step-2 negative bias.** Per-seed `b1_attn@2` at `lr=0.003`: `[-0.0778, -0.3551, +0.0189, -0.0882, -0.0698, -0.0818]` (5/6 negative). |
| **`mup_muon/svd_exact` `lr_muon` & `lr_aux` sensitivity** (3 seeds) | `(0.01, 0.01)→0.0770`; `(0.02, 0.01)→0.0811`; `(0.04, 0.01)→0.1019`; `(0.02, 0.003)→0.1021`; `(0.02, 0.03)→0.0618` | **Rock-solid.** Passes `< 0.15` across every tested `(lr_muon, lr_aux)` pair with comfortable margin. |

**Why `mup_adamw` exhibits a systematic negative `beta` at step 2 (and why `mup_muon` does not):**
At step 0, `W_head = 0`, so hidden weights receive zero gradients; at step 1, hidden weights take their **first** update using **batch 1** (`tokens[1]`), while probes are evaluated on **batch 0** (`tokens[0]`). Under `mup_adamw`, every entry of $\Delta W$ has magnitude $\Theta(\eta d_0 / d_{\text{in}})$ (pure coordinatewise sign). Applying $\Delta W$ to a different batch $u$ decomposes into a correlated (aligned) component plus an uncorrelated (cross-batch sign-noise) component:
$$\|\Delta W u\|_{\text{RMS}}^2 \approx \underbrace{C_{\text{aligned}}^2}_{\Theta(1)} + \underbrace{\frac{C_{\text{uncorr}}^2}{d}}_{\Theta(1/d)}.$$
At step 2, cross-batch alignment after a single hidden update is still weak, so the $\Theta(1/\sqrt{d})$ incoherent term ($\beta = -0.5$) inflates `delta_rms` at small width (`w=128`) relative to `w=1024`, causing a systematic negative finite-width slope (`b1_attn@2` is negative on 5 of 6 seeds, averaging `-0.1090` over 6 seeds and `-0.1380` over 3 seeds due to `seed=1`'s `-0.3551` draw). By steps 8 and 12, accumulated aligned features dominate and `mup_adamw`'s `block1_attn_out` slope collapses to `-0.0019` and `+0.0234`. By contrast, Muon's polar factor $Q$ has rank $\le 256$ concentrated in the gradient row space with entry magnitude $\Theta(1/\sqrt{d})$ and zero coordinatewise sign-noise, eliminating the step-2 finite-width dip (`block1_attn_out@2 = -0.0232`).

---

## 2. Exact Circuit & Precision Fidelity Audit

### 2.1 Coefficient & Kernel Import Verification
- **`SS3_54`, `SS3_53`, `SS3_34`, `SS3_35`:** Imported directly via `SCHATTEN_SWITCH_CIRCUITS` and executed via `schatten_switch_reference(G, method, dtype=torch.float16)` (`mup_coord_check.py:79-82, 301, 378`). Zero copied coefficients, zero drift. Notably, `PRESCALE_MIN_N = 1024` in `schatten_switch_blackwell.py:64` means the `widths = [128, 256, 512, 1024]` sweep exercises both the `P = 1.0` (`w < 1024`) and `P = 64.0` (`w = 1024`) branches without any discontinuity in `beta`.
- **`PS3_55`:** Imports `_SWITCH_TABLES["PS3_55"]` from `baselines.py` (`mup_coord_check.py:77, 342`).
- **`polar_express`:** Imports `_AMSEL_COEFFICIENTS` and `_take_steps` from `baselines.py` (`mup_coord_check.py:77, 333`).

### 2.2 CPU Low-Precision Emulation vs Native Baselines (Real Nesterov Momenta)

Tested on actual `CoordTransformer` Nesterov momentum tensors (`seed=0`, updates 2 and 4):

| Matrix (`w`, `upd`, layer) | `PE(ref vs base)` | `PE(ref vs fp64)` | `PE(base vs fp64)` | `PS3_55(ref vs base)` | `PS3_55(ref vs fp64)` |
|---|---:|---:|---:|---:|---:|
| `w=128, upd=2, b0.attn.q` | `0.0441` | `0.0257` | `0.0416` | `0.00381` | `0.00620` |
| `w=128, upd=2, b1.mlp.down` | `0.0735` | `0.0595` | `0.0810` | `0.00418` | `0.01094` |
| `w=256, upd=2, b0.attn.q` | **`0.1443`** | `0.1202` | `0.1668` | `0.00702` | `0.01500` |
| `w=256, upd=2, b1.mlp.down` | **`0.1644`** | `0.1249` | `0.1687` | `0.00713` | `0.01637` |
| `w=256, upd=4, b0.attn.q` | `0.0562` | `0.0432` | `0.0624` | `0.00483` | `0.00743` |
| `w=256, upd=4, b1.mlp.down` | `0.0650` | `0.0495` | `0.0665` | `0.00397` | `0.00747` |

- `polar_switch_reference` (`PS3_55`) matches `baselines.polar_switch` tightly (**`0.17%–0.71%`** relative Frobenius norm).
- `polar_express_reference` diverges from eager `baselines.polar_express` by **`3.5%–16.4%`** on real rank-deficient momenta (see **[M2]**).

### 2.3 `svd_exact` Numerical Rank Truncation Audit (`rtol = max(m, n) * eps(FP32)`)

Singular-value spectrum of `CoordTransformer` Nesterov momentum (`seed=0`) at `rtol = max(m, n) * eps(FP32)`:

| Width | Update | Layer | Shape | `rtol` | `> rtol` (`min/s0`) | `(1e-6, rtol]` (`max/s0`) | `<= 1e-6` |
|---:|---:|---|---|---:|---:|---:|---:|
| 128 | 2 | `b0.attn.q` | `(128, 128)` | `1.53e-5` | `128` (`1.84e-5`) | `0` | `0` |
| 256 | 2 | `b0.attn.q` | `(256, 256)` | `3.05e-5` | `133` (`2.44e-2`) | `0` (`6.84e-8`) | `123` |
| 512 | 2 | `b0.attn.q` | `(512, 512)` | `6.10e-5` | `133` (`5.40e-2`) | `0` (`6.52e-8`) | `379` |
| 1024 | 2 | `b0.attn.q` | `(1024, 1024)` | `1.22e-4` | `133` (`5.20e-2`) | `0` (`4.82e-8`) | `891` |
| 1024 | 2 | `b0.mlp.gate` | `(2048, 1024)` | `2.44e-4` | `256` (`6.32e-4`) | `0` (`3.96e-8`) | `768` |
| 1024 | 4 | `b0.mlp.gate` | `(2048, 1024)` | `2.44e-4` | `728` (`2.52e-4`) | **`39`** (`2.41e-4`) | `257` |
| 512 | 12 | `b0.attn.q` | `(512, 512)` | `6.10e-5` | `500` (`6.30e-5`) | **`11`** (`5.81e-5`) | `1` |
| 1024 | 12 | `b0.attn.q` | `(1024, 1024)` | `1.22e-4` | `822` (`1.23e-4`) | **`200`** (`1.22e-4`) | `2` |
| 1024 | 12 | `b0.mlp.gate` | `(2048, 1024)` | `2.44e-4` | `989` (`2.44e-4`) | **`35`** (`2.43e-4`) | `0` |

**Impact on `mup_muon/svd_exact` coordinate check (`widths 128–1024, seed=0`):**

| `svd_exact` `rtol` | Gate Pass | `max \|beta\|` (worst) | `b0_attn@2` | `b1_mlp@2` | `resid@12` | `logits@8` | `q` `polar_mean_sq_sv@2` (`128..1024`) |
|---|---|---:|---:|---:|---:|---:|---|
| Default (`max(m,n)*eps`) | `True` | `0.07362` (`b0_attn@4`) | `+0.0356` | `+0.0222` | `+0.0405` | `-0.0524` | `[1.0000, 0.5195, 0.2598, 0.1299]` |
| Fixed `1e-6` | `True` | `0.07376` (`b0_attn@4`) | `+0.0356` | `+0.0220` | `+0.0405` | `-0.0524` | `[1.0000, 0.5195, 0.2598, 0.1299]` |
| Naive `0.0` (`U @ Vh`) | `True` | `0.07457` (`b0_attn@4`) | `+0.0374` | `+0.0253` | `+0.0406` | `-0.0511` | `[1.0000, 1.0000, 1.0000, 1.0000]` |

- At step 2, rank deficiency is **genuine** (133 unique tokens in batch 1 for `b0.attn.q`; 256 tokens total for `b0.mlp.gate`) with a 4–6 decade cliff to FP32 noise (`~5e-8`). Truncation is necessary to avoid lifting FP32 noise into unit singular values.
- Activation-space width exponents `beta` are **insensitive** to `rtol` (`< 0.001` across all 3 settings).
- However, at step 12, `rtol = max(m, n) * eps(FP32)` grows linearly with width and **over-truncates 200 genuine singular values (`19.5%`) in `b0.attn.q` at `w=1024`**, biasing the weight-space `polar_mean_sq_sv` reference (see **[M1]**).

---

## 3. Independently Recomputed Tables & Provenance Verification

### 3.1 Provenance & Exact Numerical Consistency
- `schema`: `"cycle3_mup_coord_check/v1"`
- `git_sha`: `"d9ff3678413eba5fdaea6e2bc635c8ef23c68a7e"` (exact match to parent code commit)
- `git_dirty`: `false` · `device`: `"cpu"` · `quick`: `false` · `threads_per_task`: `1` · `jobs`: `8` · `elapsed_seconds`: `677.643`
- Strict JSON parse (`parse_constant` rejecting `NaN`/`Infinity`) and `json.dumps(..., allow_nan=False)`: **PASS**.
- Independent recomputation of all `12` variants × `3` activation stat keys × `7` probes × `6` steps (`delta_rms`, `step_delta_rms`, `activation_rms`), all `gate` verdicts, all `orthogonalizer_comparison` fields, and all `10` `lr_transfer` tables matched the artifact within 6-significant-digit `_clean()` rounding (`max |Δbeta| = 5.31e-6`, `max rel Δby_width = 6.92e-6`, `max rel Δmax_over_min = 7.85e-6`).

### 3.2 Independently Recomputed Coordinate-Check Gate & Weight-Space Lift Table

(`widths = [128, 256, 512, 1024]`, `seeds = [0, 1, 2]`, gate steps `{2, 4, 8, 12}`, `PASS_BETA = 0.15`, `FAIL_BETA = 0.25`)

| Variant | Exp | Gate | `max \|beta\|` | Worst `(probe@step)` | Per-Seed `max \|beta\|` (`s0, s1, s2`) | `max \|Δbeta\|` vs `svd_exact` | `polar_mean_sq_sv@12` (`128..1024`) | `lift beta@12` |
|---|---|---|---:|---|---|---:|---|---:|
| `sp_adamw` | fail | **FAIL** | `2.5988` | `block0_mlp_out@8` (`+2.5988`) | `2.507, 2.650, 2.639` | — | — | — |
| `mup_adamw` | pass | **PASS** | `0.1380` | `block1_attn_out@2` (`-0.1380`) | `0.120, 0.355, 0.066` | — | — | — |
| `mup_adamw_readout_lr_power1` | fail | **FAIL** | `1.0629` | `logits@2` (`-1.0629`) | `1.069, 1.134, 1.002` | — | — | — |
| `sp_muon/svd_exact` | fail | **FAIL** | `1.5944` | `block0_mlp_out@2` (`+1.5944`) | `1.596, 1.593, 1.594` | — | — | — |
| `mup_muon/svd_exact` | pass | **PASS** | `0.0811` | `logits@8` (`-0.0811`) | `0.074, 0.116, 0.111` | `0.0000` | `0.999, 0.998, 0.979, 0.882` | `-0.0566` |
| `mup_muon/polar_express` | pass | **PASS** | `0.0819` | `logits@8` (`-0.0819`) | `0.073, 0.116, 0.111` | `0.0015` | `0.957, 0.918, 0.801, 0.596` | `-0.2247` |
| `mup_muon/PS3_55` | pass | **PASS** | `0.0809` | `logits@8` (`-0.0809`) | `0.073, 0.115, 0.111` | `0.0024` | `0.948, 0.910, 0.790, 0.591` | `-0.2254` |
| `mup_muon/SS3_54` | open | **PASS** | `0.0797` | `logits@8` (`-0.0797`) | `0.074, 0.117, 0.108` | `0.0016` | `0.968, 0.939, 0.827, 0.630` | `-0.2040` |
| `mup_muon/SS3_53` | open | **PASS** | `0.0782` | `logits@8` (`-0.0782`) | `0.072, 0.115, 0.107` | `0.0029` | `0.923, 0.889, 0.753, 0.550` | `-0.2479` |
| `mup_muon/SS3_34` | open | **PASS** | `0.0795` | `logits@8` (`-0.0795`) | `0.073, 0.115, 0.109` | `0.0039` | `0.878, 0.802, 0.608, 0.404` | `-0.3763` |
| `mup_muon/SS3_35` | open | **PASS** | `0.0780` | `logits@8` (`-0.0780`) | `0.073, 0.114, 0.107` | `0.0032` | `0.941, 0.900, 0.748, 0.534` | `-0.2724` |
| `mup_muon_readout_lr_power1/PS3_55` | fail | **FAIL** | `1.0385` | `logits@2` (`-1.0385`) | `1.012, 1.091, 1.013` | — | — | — |

### 3.3 Independently Recomputed LR-Transfer Table: Artifact (`seed=0`) vs Audit Multi-Seed (`seeds=[0, 1, 2]`)

(`widths = [128, 256, 512]`, `steps = 20`, training-batch mean cross-entropy)

| Variant | Artifact (`seed=0`) `best_lr` (`128, 256, 512`) | Artifact `interp` (`128, 256, 512`) | Artifact `same` / `interp_shift` / `max_regret` | Audit 3-Seed (`0,1,2`) `best_lr` (`128, 256, 512`) | Audit 3-Seed `same` / `interp_shift` / `max_regret` |
|---|---|---|---|---|---|
| `sp_adamw` | `0.003, 0.001, 0.001` | `0.0035, 0.0010, 0.0010` | `False` / `-1.806` / `0.6852` | `0.003, 0.001, 0.001` | `False` / `-1.881` / `0.6970` |
| `mup_adamw` | `0.01, 0.01, 0.01` | `0.0124, 0.0102, 0.0101` | `True` / `-0.293` / `0.0000` | **`0.01, 0.01, 0.01`** | **`True`** / `-0.216` / `0.0000` |
| `sp_muon/svd_exact` | `0.02, 0.01, 0.08` | `0.0195, 0.0079, 0.0800` | `False` / `+2.038` / `0.1999` | `0.02, 0.01, 0.08` | `False` / `+1.964` / `0.1987` |
| `mup_muon/svd_exact` | `0.04, 0.04, 0.04` | `0.0291, 0.0407, 0.0352` | `True` / `+0.274` / `0.0000` | **`0.02, 0.04, 0.04`** | **`False`** / `+0.457` / `0.0366` |
| `mup_muon/polar_express` | `0.04, 0.04, 0.04` | `0.0301, 0.0410, 0.0352` | `True` / `+0.225` / `0.0000` | *(not re-run)* | — |
| `mup_muon/PS3_55` | `0.04, 0.04, 0.04` | `0.0304, 0.0409, 0.0354` | `True` / `+0.220` / `0.0000` | *(not re-run)* | — |
| `mup_muon/SS3_54` | `0.04, 0.04, 0.04` | `0.0299, 0.0401, 0.0349` | `True` / `+0.223` / `0.0000` | **`0.02, 0.04, 0.04`** | **`False`** / `+0.463` / `0.0349` |
| `mup_muon/SS3_53` | `0.04, 0.04, 0.04` | `0.0305, 0.0393, 0.0357` | `True` / `+0.230` / `0.0000` | *(not re-run)* | — |
| `mup_muon/SS3_34` | `0.04, 0.04, 0.04` | `0.0312, 0.0404, 0.0358` | `True` / `+0.197` / `0.0000` | **`0.02, 0.04, 0.04`** | **`False`** / `+0.495` / `0.0393` |
| `mup_muon/SS3_35` | `0.04, 0.04, 0.04` | `0.0303, 0.0398, 0.0354` | `True` / `+0.224` / `0.0000` | *(not re-run)* | — |

---

## 4. Severity-Ranked Findings

### `CRITICAL`
None.

---

### `HIGH`

#### [H1] Single-seed (`seed=0`) Muon LR-transfer headline (`same_best_lr = True`, `0.04` at all widths) fails under 3-seed averaging, and `final_loss` measures 5-epoch training-batch memorization
- **Location:** `research/polar_opt/mup_coord_check.py:566-567, 820, 923`, `results/cycle3_mup_coord_check_cpu.json` (`lr_transfer`)
- **Evidence:**
  1. `run()` and the CLI default to `lr_seeds = (0,)`. In the artifact (`seed=0`), at `w=128`, `mup_muon/svd_exact` attains `0.6924` at `lr=0.04` vs `0.6961` at `lr=0.02` — a razor-thin `0.0037`-nat margin, with interpolated optimum `0.02914` right at the geometric midpoint (`0.0283`) of `[0.02, 0.04]`.
  2. Re-running the exact harness with `lr_seeds = (0, 1, 2)` flips **all three** tested Muon variants (`svd_exact`, `SS3_54`, `SS3_34`) to `best_lr = {128: 0.02, 256: 0.04, 512: 0.04}` (`same_best_lr = False`, `best_lr_log2_shift = +1.00`, `interp_shift = +0.46 to +0.49` octaves, `max_regret = 0.0349–0.0393`). On `SS3_54`, **2 of 3 individual seeds** (`seed=1` and `seed=2`) pick `0.02` at `w=128` (`0.04` at `w=512`).
  3. Because `LR_STEPS = 20` cycles through `num_batches = 4` (5 full epochs over 1,024 tokens) and `final_loss` evaluates those **same 4 training batches**, `mup_muon/svd_exact` memorizes the training set (`0.740 → 0.311 → 0.141` across `w=128..512`, 3-seed mean in Probe 3B) while **held-out validation loss** on 4 unseen batches from the same Markov chain is `4.822 → 4.393 → 4.042` at `lr=0.04` (and prefers `lr=0.08` at `w=128, 256` vs `0.04` at `w=512`, hitting the upper grid edge). By contrast, `mup_adamw` cleanly keeps `best_lr = 0.01` across all 3 seeds on both training loss and held-out validation loss.
- **Actionable Remediation:** Default `--lr-seeds 0 1 2` (3 seeds minimum), evaluate `lr_transfer` on held-out batches from the same Markov chain (or use `num_batches >= LR_STEPS` fresh batches without cycling), and expand the Muon LR grid / report continuous `best_lr_interpolated_log2_shift` and base-LR regret rather than relying on binary `same_best_lr` from a single seed.

#### [H2] `mup_adamw` coordinate-check pass is sensitive to LR and seed count, failing at its own optimal LR (`0.01`) on 6 seeds and masking an asymmetric test-vs-artifact seed count
- **Location:** `research/polar_opt/mup_coord_check.py:88, 819`, `tests/test_mup_coord_check.py:525`, `results/cycle3_mup_coord_check_cpu.json` (`coord_check.mup_adamw`)
- **Evidence:**
  1. In the artifact (`lr=0.003`, 3 seeds), `mup_adamw` barely clears `PASS_BETA = 0.15` (`max |beta| = 0.13804` at `block1_attn_out@2`, `seed=1` alone hitting `-0.3551`).
  2. `tests/test_mup_coord_check.py:525` silently increased `mup_adamw` to **6 seeds** (`tuple(range(6))`, lowering `max |beta|` to `0.10900`), while keeping `mup_muon/*` at 1 seed and the CLI artifact at 3 seeds.
  3. Across LRs, `mup_adamw`'s `< 0.15` gate passes only inside a narrow window: `lr=0.001` **fails** (`0.15437`, 3 seeds); `lr=0.003` passes (`0.13804`, 3 seeds; `0.10900`, 6 seeds); `lr=0.01` (its own optimal LR in `lr_transfer`!) passes thinly on 3 seeds (`0.14160`) but **fails on 6 seeds (`0.15846` at `block0_mlp_out@8`)**; `lr=0.03` fails severely (`0.34125`, all seeds `> 0.51`).
  4. As derived in §1.3, the step-2 negative bias is a genuine finite-width $\Theta(1/\sqrt{d})$ cross-batch sign-update artifact at `w=128` (4 heads), disappearing by steps 8–12 (`-0.0019`, `+0.0234`).
- **Actionable Remediation:** Use `seeds = (0, 1, 2, 3, 4, 5)` consistently in the CLI default (or start widths at `256` / record both step-2 transient and steady-state `steps 4, 8, 12` gate metrics) and document the LR-sensitivity envelope of `mup_adamw` explicitly alongside `mup_muon`.

---

### `MEDIUM`

#### [M1] `svd_exact` width-growing `rtol = max(m, n) * eps(FP32)` over-truncates 200 genuine singular values at `w=1024` step 12, distorting the weight-space `polar_mean_sq_sv` reference
- **Location:** `research/polar_opt/mup_coord_check.py:314-325, 558-562, 720-725`
- **Evidence:** Because `rtol = max(m, n) * eps(FP32)` scales linearly with width (`1.53e-5` at `w=128` → `2.44e-4` at `w=1024`), at `w=1024, upd=12` it zeroes out **200 genuine singular values** in `(1e-6, 1.22e-4]` of `blocks.0.attn.q` (`19.5%` of the spectrum; actual FP32 rounding floor is `~5e-8`, 3 orders of magnitude lower) and 35 in `blocks.0.mlp.gate`. While activation-space `beta` changes by `< 0.001` (`0.07362` vs `0.07376` at `rtol=1e-6`), `svd_exact`'s own `hidden_polar_mean_sq_sv[12]` drops artificially to `0.8817` (`beta = -0.0566`) at `w=1024`, understating the true polynomial under-lift gap of `SS3_*`, `PS3_55`, and `polar_express` relative to the exact row-space polar factor.
- **Actionable Remediation:** Use a width-independent FP32 noise floor (e.g. `rtol = 1e-6`, sitting cleanly in the 3–6 decade gap between FP32 noise `~5e-8` and genuine singular values `> 1e-5`), or compute `svd_exact` on FP64-accumulated gradients / report exact rank at `1e-6` alongside `polar_mean_sq_sv`.

#### [M2] `polar_express_reference` emulates fused-epilogue BF16 rather than eager `baselines.polar_express`, producing `3.5%–16.4%` update divergence on real rank-deficient momenta
- **Location:** `research/polar_opt/mup_coord_check.py:328-337`, `research/polar_opt/baselines.py:139-153`
- **Evidence:** In `baselines.polar_express`, `X.norm()`, `b * A`, `(c * A) @ A`, `a * X`, and `B @ X` are each rounded to BF16 separately before addition. In `polar_express_reference`, `b * A + _stored(c * A, dtype) @ A` and `a * X + B @ X` accumulate in FP32 with a single BF16 rounding (matching fused `baddbmm` / compiled epilogues). On real `CoordTransformer` Nesterov momenta at `w=256, upd=2`, this yields `14.4%–16.4%` relative Frobenius divergence from `baselines.polar_express` (vs `0.17%–0.71%` for `PS3_55`). Activation `beta` is insensitive (`0.0015` from `svd_exact`), but the docstring's claim that `polar_express_reference` matches `baselines.polar_express` should either note the fused-epilogue difference or insert the exact intermediate `_stored(..., dtype)` calls.

#### [M3] `sp_muon` negative control does not isolate `sqrt(d_out / d_in)` from SP initialization/readout
- **Location:** `research/polar_opt/mup_coord_check.py:27-28, 254-259, 772`
- **Evidence:** Because every hidden matrix in `CoordTransformer` has constant aspect ratio across width $d$ (`q/k/v/o`: `1`, `gate/up`: `mlp_mult`, `down`: `1/mlp_mult`), Muon's spectral factor `sqrt(d_out / d_in)` is **constant with respect to $d$**. Consequently, `sp_muon/svd_exact`'s coordinate-check failure (`block0_attn_out@2 = +1.045`, `block0_mlp_out@2 = +1.594`) is caused by SP's `0.02` initialization (forward variance scaling as $d$ and $d^{3/2}$) and `logit_scale = 1.0`, NOT by omitting `sqrt(d_out / d_in)` (which is independently unit-tested by `test_muon_update_is_spectrally_scaled`). Similarly, `head_dim = 32` is fixed across widths, so `1/head_dim` attention scaling is not width-tested. Clarify this attribution in documentation so the `sp_muon` failure is not misattributed solely to the `sqrt(d_out / d_in)` prefactor.

#### [M4] Edge-case bugs in `summarize_lr_transfer`, `run_trial`, and `update_scaling`
- **Location:** `research/polar_opt/mup_coord_check.py:561-562, 641, 675-696`
- **Evidence (all reproduced via CPU probes):**
  1. **All-diverged false positive in `summarize_lr_transfer()`:** When every LR diverges for all widths (`final_loss = None`), `table[w]` is all `inf`, `min(range(len(lrs)), key=row.__getitem__)` returns index `0`, producing `best_lr = {128: 0.01, 256: 0.01}`, **`same_best_lr = True`**, and `regret = inf - inf = NaN` (silently converted to `None` by `_clean()`).
  2. **Division by zero in `run_trial()` at Muon `lr = 0.0`:** Line 561 divides by `(lr * specs[name].muon_scale) ** 2` without guarding `lr == 0.0`, raising `ZeroDivisionError`.
  3. **Silent step omission in `update_scaling()`:** Line 641 hardcodes `for i, step in enumerate(steps[1:], 1):`, assuming `steps[0] == 0`. Passing `probe_steps=(2, 4)` silently drops step `2` from all `updates` summaries (`updates` keys = `['4']` while `delta_rms` keys = `['2', '4']`).
- **Actionable Remediation:** Guard all-infinite rows in `summarize_lr_transfer` (`best[w] = None`, `same_best_lr = False`, `regret = None`); guard `lr > 0` in `polar_mean_sq_sv`; iterate `for i, step in enumerate(steps): if step > 0:` in `update_scaling`.

---

### `LOW`

#### [L1] Minor input-validation, CLI, and provenance gaps
- **Location:** `research/polar_opt/mup_coord_check.py:118-122, 849, 936-938`
- **Details:**
  1. `ModelConfig.__post_init__` accepts non-positive `d_base=0`, `mlp_mult=0`, `layers=0`, and `num_batches=0` (failing later or silently zeroing logits).
  2. In `main()`, `--quick` unconditionally overwrites `options`, silently ignoring user-supplied `--widths`, `--seeds`, or `--lr-seeds`, and hardcodes `lr_widths = (128, 256)` even if `--d-base` is changed.
  3. `run()` strips `updates` from per-trial records in `trials` (`line 849`), so `coord_check[v]["updates"]` cannot be independently recomputed from raw per-trial JSON rows (only the aggregate `orthogonalizer_comparison[v]["hidden_polar_mean_sq_sv"]` is recomputable from `coord_check[v]["updates"]`).

---

## 5. Summary of Actionable Remediation Items

1. **Multi-seed & held-out LR transfer ([H1]):** Default `--lr-seeds 0 1 2`, evaluate `final_loss` on held-out batches (or fresh non-cycling batches), and downgrade the artifact claim from exact `0.04` transfer across `128..512` to low-regret transfer (`~0.035` nats; stable `0.04` across `256..512`).
2. **Harmonize `mup_adamw` seed count & document LR envelope ([H2]):** Match CLI and pytest seed counts (6 seeds) or report both transient (`step=2`) and steady-state (`steps 4, 8, 12`) gate summaries alongside the `mup_adamw` LR sensitivity table.
3. **Fix `svd_exact` width-growing `rtol` ([M1]):** Replace `max(G.shape[-2:]) * eps(FP32)` with a fixed `1e-6` floor (or width-invariant threshold) so `polar_mean_sq_sv` at `w=1024`, step 12 is not biased by over-truncating 200 genuine singular values.
4. **Patch edge-case guards ([M4], [L1]):** Guard all-diverged rows in `summarize_lr_transfer`, `lr == 0.0` in `polar_mean_sq_sv`, `step > 0` (instead of `steps[1:]`) in `update_scaling`, and positive bounds in `ModelConfig.__post_init__`.
5. **Document `polar_express_reference` fused-epilogue behavior and `sp_muon` control attribution ([M2], [M3]).**
