# Adversarial Audit Report — Cycle 1 (Auditor 4: Gemini 4 Neon)

- **Repository:** `/home/ryan/Documents/projects/ai` (`domainus/ai`)
- **Audit Scope:** Late-2026 SOTA positioning, intellectual honesty (positive & negative findings), repo reproducibility, data consistency across state/ledger/raw JSONs, infographic compliance & accuracy, and Cycle 2 strategic prioritization.
- **Audit Date:** 2026-11-25
- **Executive Verdict:** **`PASS_WITH_CAVEATS`**

> **Promotion Gate Decision:** Do **not** promote Cycle 1 entries from `PENDING_AUDIT` to `VERIFIED` or publish the Day 0 infographic until the **2 `HIGH`** and **`MEDIUM` data/wording** findings below are remediated. The core Blackwell `sm_120` Triton `SYRK`/`_symprod_kernel` engineering (`ps2_55_blackwell_fp16`, `ps3_55_blackwell_fp16`), the 550-test verification suite, and the honest discovery of heuristic warm-power slow drift are solid and reproducible—but summary prose, the FP8 failure diagnosis, and `reports/generate_infographic.py` contain concrete defects that would mislead readers if published as-is.

---

## 1. Late-2026 SOTA Positioning & Intellectual Honesty Review

### Comparison Against Late-2026 Literature

| Upstream Work | Core Contribution | What Cycle 1 Reuses Directly | What Cycle 1 Adds (Honest Classification) |
| :--- | :--- | :--- | :--- |
| **Polar Express** (Amsel, Persson, Musco, Gower — ICLR 2026 / `arXiv:2505.16932`) | Remez-optimal odd quintic polynomial compositions + safety factor schedule | Coefficients & Gram-form iteration (`baselines.py`, `kernels_blackwell.py`, `kernels_mlx.py`) | **Systems + Negative Result:** Blackwell Triton `SYRK`+`_poly_kernel` port (`polar_express_triton_bf16`) and empirical falsification of naive 2-step e4m3 FP8-early (`polar_express_fp8x2_bf16`) via 68.75% divergence on `uniform_mp` spectra. |
| **Gram Newton-Schulz** (Zhang, Amsel, Chen, Dao — 2026, `Dao-AILab/gram-newton-schulz`) | Gram-space recurrence + restarts + datacenter H100/B200 QuACK symmetric kernels | Exact recurrence, Dao `1.05` PE preset, reset schedule (`gram_ns`, `gram_ns_blackwell`) | **Systems Port:** Consumer Blackwell (`sm_120`, 99 KB shared mem/CTA) Triton port without QuACK/B200 (`1.53–2.30×` vs eager PE across LLM shapes; `0.424 ms` vs `0.570 ms` eager on Tier-1). |
| **Polar Switch** (Du & Su — Oct 2026, `arXiv:2610.26742`, `zhehangdu/polar-switch`) | Register-minimal multi-buffer squaring-chain circuits (`PS2-55`, `PS3-55`) over symmetric products | Exact `PS2_55` & `PS3_55` circuit tables (`baselines.py`, `polar_switch_blackwell.py`, `kernels_mlx.py`) | **Systems Engineering (High Value) + Apple Silicon Port (Incremental):** Autotuned `sm_120` Triton `_syrk_kernel` + fused `_symprod_kernel` (bitwise upper-triangle mirroring, FP32 accumulation, 1 rounding) + plan-cached `_launch` skipping ~12 µs Python dispatch overhead (`1.88–2.61×` vs eager PE; `1.52–2.78×` vs eager PS2/PS3; `0/117` Tier-1 & `0/72` Tier-2 divergence). MLX `float32` port on Apple M5 Max (`1.52×` vs PE at `1024×4096`, but `-14%` slower at `2048×2048`). |
| **Cycle 1 Track 1 (Original)** (`spectral_scaling.py`, `circuits.py`, `circuit_search.py`) | N/A (repo-originated) | Builds on Gram-form squaring chains | **Genuine Negative Discovery + Algorithmic Pivot (Not Yet Pareto-Winning):** (1) Falsified heuristic warm-started power iteration on a live 19.2M LM (gradual multi-step drift reached `true_sigma / sigma_hat = 1.191 > 1.160` at step 11 without triggering 1-step jump guards). (2) Pivoted to stateless, sync-free Schatten-8 upper-bound pre-scaling reusing `R²` (`0/72` Tier-2 divergence; `S1-k4+5-lo0.002_schatten8` hits `0.2800` max harvested error vs `PS3_55` `0.3208`). (3) Rounding-aware variable-projection + Lawson + HiGHS LP search. **Honest caveat:** Currently `3.8–4.5×` slower than `ps3_55_blackwell_fp16` (`1.185–1.378 ms` vs `0.307 ms` Tier-1) and loses on Tier-2 wall-clock loss (`3.116–3.131` vs `2.867`). |

### Guardrails Against Overclaiming

1. **Systems Engineering vs. New Mathematics:** Cycle 1's winning method (`ps3_55_blackwell_fp16`) executes **Du & Su's exact `PS3_55` polynomial circuit**, not a new mathematical algorithm. Its win is **consumer Blackwell (`sm_120`) Triton kernel engineering** (bringing QuACK-style symmetric-product acceleration to RTX 5080 without B200/QuACK). README, state summaries, and reports must state this explicitly.
2. **Tier-2 Wall-Clock Loss Is Purely Throughput on a Tiny Synthetic Task:** Under matched token budget (`409,600` tokens, 200 steps) in `results/cycle1_tier2_lm.json`, all Muon orthogonalizers achieve virtually identical validation loss (`3.0316`–`3.0439`, spread `0.008` across `muon_pe_triton`, `muon_pe_eager`, `muon_ps2/ps3_blackwell`, and `S1` Schatten-8 variants). The headline `-0.224` wall-clock validation loss improvement of `muon_ps3_blackwell` (`2.8669` vs `3.0910` for `muon_pe_eager`) comes **100% from running 237–240 steps instead of 186–188 steps** in 3.0 seconds on a synthetic 8-token modular-arithmetic motif stream (`batch_size=16, seq_len=128` = `2,048` tok/step on a 19.2M model, 2 seeds).
   - **Scale-up reality check:** At `2,048` tokens/step, optimizer orthogonalization accounts for ~30–40% of step time. In real 50M–120M+ LLM pretraining with realistic batch sizes (`32k–131k` tokens/step via gradient accumulation), forward+backward scales linearly with tokens/step while Muon orthogonalization is constant per step—so end-to-end tok/s gains will shrink substantially (typically single-digit %).
   - **Missing baseline in Tier-2:** `PS3_55 (eager)` was not run in Tier-2 (`tier2_wall_loss: null`). Against the included `PS2_55 (eager)` baseline (`2.9440 @ 149,840 tok/s`), `muon_ps3_blackwell` gains **`+8.4%` tok/s** and **`-0.077` wall-clock loss** (still a clean win, but far smaller than the `+27.7%` / `-0.224` headline vs `polar_express (eager)`).
   - **Uncompiled eager baselines:** All eager baselines ran without `torch.compile`, and `muon_pe_eager` / `muon_ps2_blackwell` / `muon_ps3_blackwell` all selected the upper boundary LR (`0.02` out of `{0.01, 0.02}`), meaning optimal LR was not bracketed.
3. **Apple Silicon M5 Max Port Is Incremental & Has an Unreported Regression:** `research/polar_opt/kernels_mlx.py` implements pure eager `float32` MLX matmuls (no custom Metal `SYRK`, not FP16/BF16), with only 1 warmup and 5 repeats (exhibiting large initial cold-run variance on `512×2048`: e.g., `jordan_ns5` samples `[2.91, 2.60, 1.41, 1.40, 1.32] ms`). Crucially, while `PS3_55` beats `PolarExpress` at `1024×4096` (`2.191 ms` vs `3.342 ms`, `1.52×`), it is **14% slower (`0.88×`) on square `2048×2048`** (`7.704 ms` vs `6.747 ms`) because dense `operand @ operand.T` pays full $2n^3$ GEMM cost across all 10 Gram squares. That negative square-matrix result is omitted from `RESEARCH_STATE.json`, `BACKLOG.md`, and `ledger.jsonl`.

---

## 2. Data-Consistency Audit Table

Cross-checked every reported metric in `state/RESEARCH_STATE.json`, `state/BACKLOG.md`, `state/ledger.jsonl`, and `reports/generate_infographic.py` against raw outputs in `results/harvested_spectra.json`, `results/cycle1_head_to_head.json`, `results/cycle1_tier2_lm.json`, `results/apple_silicon_cycle1.json`, `results/blackwell_kernels_cycle1.json`, and `results/polar_switch_blackwell_cycle1.json`.

| Claim / Field | Location | Reported Value | Raw JSON Ground Truth | Status |
| :--- | :--- | :--- | :--- | :--- |
| Leaderboard rows (all 8 methods × 5 metrics = 38 values) | `RESEARCH_STATE.json:60-69` | `tier1_ms`, `harvested_med_err`, `harvested_max_err`, `tier2_wall_loss`, `tier2_tps` | Exact match to `cycle1_head_to_head.json` (`tier1_summary.all` for `tier1_ms`; `tier1_summary.harvested_momentum` for errors) & `cycle1_tier2_lm.json` (`summary[budget=wall]`) | **PASS** (100% match to rounded precision; note mixed 117-case latency / 45-case error scopes) |
| Harvested LM momentum stats | `RESEARCH_STATE.json:37`, `BACKLOG.md:5`, `ledger.jsonl:2` | Eff-rank `3.19×`, right overlap `0.937`, max Δσ_max `31.82%` | `harvested_spectra.json`: `3.18744`, `0.93656`, `0.318175` | **PASS** |
| Pytest suite count | `RESEARCH_STATE.json:32` | `550` tests passing | `pytest --collect-only` collects exactly `550` tests | **PASS** |
| Tier-2 total runs & divergence | `RESEARCH_STATE.json:39-40`, `ledger.jsonl:4` | `72` runs, `0` div | `cycle1_tier2_lm.json`: `len(runs)=72`, `sum(diverged)=0` | **PASS** |
| `ps3_55_blackwell` speedup vs PE at `2048×8192` | `RESEARCH_STATE.json:30,38` | `2.61×` | `cycle1_head_to_head.json`: `2.6056×` (rounds to `2.61×`; `ps2` is `2.6091×`) | **PASS** |
| `ps3_55_blackwell` wall-clock val loss delta vs PE | `RESEARCH_STATE.json:39`, `BACKLOG.md:7` | `-0.224` (`2.8669` vs `3.0910`) | `2.866888 - 3.090979 = -0.224091` | **PASS** |
| **FP8 divergence cause & rate** | `RESEARCH_STATE.json:42,51`, `BACKLOG.md:7,13` | **"9% of Tier-1 adversarial cases" / "adversarial ill-conditioned spectra"** | `cycle1_head_to_head.json`: **`adversarial` = `0/32` (0.0%, `σ_max=1.12`)**; `harvested` = `0/45` (0%); `power_law` = `0/24` (0%, `σ_max=1.73`); **`uniform_mp` = `11/16` (68.75%, `σ_max=44,204.85`)**; `all` = `11/117` (`9.40%`) | **FAIL (Misdiagnosed)** |
| **Tier-2 `muon_ps3_blackwell` tok/s gain vs PE eager** | `RESEARCH_STATE.json:30,39`, `BACKLOG.md:7`, `ledger.jsonl:4` | **`+27.8%`** | `cycle1_tier2_lm.json` wall: `162448.31 / 127211.59 - 1 = +27.699%` → **`+27.7%`** (`ps2_blackwell` is `+28.3%`; `ps3` token-budget is `+29.9%`) | **MISMATCH (`+0.1 pp`)** |
| **Tier-1 117-case speedup vs PS eager** | `RESEARCH_STATE.json:31` | **`1.92×` (117-case Tier-1)** | `PS3_55 / ps3_bw = 0.581145 / 0.307082 = 1.8925×` (**`1.89×`**); `PS2_55 / ps2_bw = 1.9011×` (**`1.90×`**); `PS3_55 / ps2_bw = 1.9126×` (`1.91×`). None equals `1.92×`. | **MISMATCH** |
| **Blackwell PS speedup range vs PE** | `BACKLOG.md:7` | **`1.43–2.60×`** | `cycle1_head_to_head.json`: `1.41×` (Tier-1 117-case) and `1.88–2.61×` (LLM shapes); `polar_switch_blackwell_cycle1.json`: `1.76–2.55×`. Neither matches `1.43–2.60×`. | **MISMATCH** |
| **Apple M5 Max `PS3_55` speedup vs PE (`1024×4096`)** | `RESEARCH_STATE.json:18,41`, `BACKLOG.md:8` | **`1.53×`** (`2.191 ms` vs `3.342 ms`), `34%` lower error | `apple_silicon_cycle1.json`: `3.3415 / 2.19125 = 1.52493×` → **`1.52×`** (only `1.53×` via double-rounding). Error reduction `34.03%` (`34%`, PASS). Omits `2048×2048` `0.88×` regression (`7.704` vs `6.747 ms`). | **MISMATCH (double-round) + Omission** |
| **`triton_syrk` speedup range vs cuBLAS** | `RESEARCH_STATE.json:38`, `BACKLOG.md:7`, `ledger.jsonl:2` | **`1.69–2.02×`** | `blackwell_kernels_cycle1.json`: `1.125×` (`512×2048`), `1.667×` (`1024×4096 fp16`), `1.690×` (`1024×4096 bf16`), `1.634×` (`4×1024×4096 bf16`), up to `2.024×` (`4×1024×4096 fp16`). Full range is `1.13–2.02×` (`1.63–2.02×` for `N≥1024`). | **MISMATCH (cherry-picked floor)** |
| **`S1-k4+5-lo0.002_schatten8` max harvested error** | `RESEARCH_STATE.json:40` | `0.2800` ("best overall") | `cycle1_head_to_head.json`: `0.28002` is best among **stateless/non-drifting approximate** methods (`S1-k4+5-lo0.002_triton_fp16` warm-power got `0.2794`; `svd_polar_fp64` is `0.0000`). | **MINOR IMPRECISION** |
| **Infographic live output** | `reports/generate_infographic.py`, `reports/daily/2026-11-25.html` | Should display Cycle 1 KPIs, Pareto chart, excluded divergent methods, 4-model audit verdicts | Simulating `generate_infographic.py` on current `RESEARCH_STATE.json` outputs **`Pending`** for Speedup KPI, **`Awaiting paired latency and polar error measurements.`** (no chart/table), and only 2 stale reviewers (`Claude Opus 5.5`, `GPT 6.2 Sol`). On-disk `2026-11-25.html` is still Cycle 0 (`0` experiments, `0` tests). | **FAIL** |

---

## 3. Infographic Compliance & Clarity Audit (`reports/generate_infographic.py`)

### T3 `html_render` Styling & Layout Compliance: **PASS**
- **No `vh` units and no `height: 100%`** on `html` or `body`.
- **No background color, outer card, border, banner title, or side padding** on `html`, `body`, or `<main>`. Only `.kpis > div` uses `background: var(--card); padding: 16px; border-radius: var(--radius)`.
- **Theme CSS variables only:** Uses exclusively `:root` theme variables (`--foreground`, `--muted-foreground`, `--border`, `--card`, `--radius`, `--success`, `--info`, `--destructive`, `--primary`, `--chart-1`..`--chart-6`, `--font-sans`)—zero hardcoded hex/RGB colors.
- **Fixed chart height & fluid responsive width:** SVG uses fixed pixel height `height="214"` with `width="100%"`, `viewBox="0 0 330 214"`, and `@media (max-width: 580px)` mobile grid collapse.

### Functional & Schema Integrity Against Cycle 1 Data: **FAIL**
Running `generate_infographic.py` in memory against the current `state/RESEARCH_STATE.json` and `results/cycle1_head_to_head.json` exposes five defects:
1. **Pareto Chart & Benchmark Table Completely Missing:** `measurements()` (`lines 122–124`) ignores `tier1_ms` and `harvested_med_err` from `RESEARCH_STATE.json["leaderboard"]`, then `benchmarks()` (`line 143`) checks `data.get("summary", {})` instead of `data.get("tier1_summary", {})` in `cycle1_head_to_head.json`, and its fallback (`line 156`) checks `rows`/`results`/`leaderboard` instead of `llm_shapes`. Output: `"Awaiting paired latency and polar error measurements."`—with zero charts and zero mention of excluded divergent methods (`polar_express_fp8x2_bf16`).
2. **Speedup KPI Renders `"Pending"`:** `kpi_strip()` (`lines 72–75`) passes `best_speedup_vs_pe` and `best_speedup_vs_ps` through `number()`, which rejects the Cycle 1 descriptive strings in `RESEARCH_STATE.json`.
3. **Stale Reviewer Roster & Broken Audit Parsing:** `REVIEWERS = ("Claude Opus 5.5", "GPT 6.2 Sol")` (`line 12`) and KPI label `"Tests & Dual Audits"` (`line 79`) are outdated (vs. the 4-model Frontier Roster in `RESEARCH_PROGRAM.md`). Worse, `fleet_audits()` (`line 197`) matches `a.get("model") or a.get("reviewer")` against exact reviewer strings, while `RESEARCH_STATE.json` uses `"auditor": "Claude Opus 5.5 (high)"`. Even after all 4 audits complete, the infographic will silently keep printing `"Pending"`.
4. **No Visual Distinction for `PENDING_AUDIT` vs. `VERIFIED` vs. `BASELINE`:** `scatter()` does not encode `row["status"]`. Once wired to `RESEARCH_STATE.json`, unaudited `PENDING_AUDIT` entries would appear indistinguishable from verified wins.
5. **Stale On-Disk Daily Report:** `reports/daily/2026-11-25.html` still reflects the Cycle 0 initialization state (`0` experiments, `0` tests, `MacBook Pro: pending clone`).

---

## 4. Severity-Ranked Findings

### `HIGH`

- **[H1] Factual Misdiagnosis of `polar_express_fp8x2_bf16` Divergence (Steers Cycle 2 `T2-02` in Wrong Direction)**
  - **Evidence:** `RESEARCH_STATE.json` (lines 42, 51) and `BACKLOG.md` (lines 7, 13) state that `polar_express_fp8x2_bf16` diverged on "9% of Tier-1 adversarial cases" / "adversarial ill-conditioned spectra". In `results/cycle1_head_to_head.json`, `adversarial` had **0/32 (0.0%)** divergence (`σ_max = 1.12`) and `harvested_momentum` had **0/45 (0.0%)** divergence (`σ_max = 1.12`). All 11 failures occurred in **`uniform_mp`** (**11/16 = 68.75%**, `σ_max = 44,204.85`, reproduced directly across every `64×64`, `64×128`, `128×128`, `128×256`, `256×256`, and `uniform_singular_values/256×512` case; `power_law` also reached `σ_max = 1.73`).
  - **Root Cause:** On well-conditioned smaller-rank `uniform_mp` matrices, Step 1 lifts the entire bulk of singular values ($\sigma \approx 1/\sqrt{n}$) directly to $\approx 1.0$. At Step 2 (`fp8_steps=2`), $A = X X^\top \approx I$ across all $n$ directions simultaneously; per-tensor e4m3 FP8 quantization (±6.25% mantissa error) on `X`, `A`, and `B = A² + (b/c)A` interacts with cancellation in `a*X + B@X` (`+4.07X` vs `-2.32X`) to push `σ_max` outside Polar Express's narrow stability basin (`~1.08`), causing Steps 3–5 to explode.
  - **Action:** Correct the diagnosis across `RESEARCH_STATE.json`, `BACKLOG.md`, and `ledger.jsonl`. Furthermore, note that `polar_express_fp8x2_bf16` is **20–77% slower than FP16 `ps3_55_blackwell_fp16` on every LLM shape** (`12.15 ms` vs `10.15 ms` at `4096×4096`; `5.40 ms` vs `3.05 ms` at `2048×8192`)—do not waste Cycle 2 fixing a Pareto-dominated method.

- **[H2] `reports/generate_infographic.py` Is Functionally Broken Against Cycle 1 State & Results**
  - **Evidence:** Simulated execution against current `RESEARCH_STATE.json` and `cycle1_head_to_head.json` produces `"Pending"` for Best Speedup, `"Awaiting paired latency and polar error measurements."` (no chart, no table, no divergent-method exclusion note), and a broken 2-reviewer audit section that ignores `auditor` entries. Meanwhile `reports/daily/2026-11-25.html` is stale Cycle 0 output.
  - **Action:** Update `generate_infographic.py` to: (a) parse `tier1_summary["all"]` (and/or `leaderboard` keys `tier1_ms`/`harvested_med_err` with explicit scope & `PENDING_AUDIT` badge); (b) render numeric/string speedup KPIs cleanly; (c) update `REVIEWERS` to the 4-model Frontier Roster and parse `"auditor"` cleanly; (d) display excluded divergent methods (`polar_express_fp8x2_bf16`) prominently; and (e) add a regression test in `tests/test_infographic.py` that runs against real Cycle 1 schema shapes before regenerating `reports/daily/2026-11-25.html`.

### `MEDIUM`

- **[M1] Arithmetic, Rounding, and Cherry-Picking Discrepancies Across State/Backlog/Ledger**
  - **Evidence:** `+27.8%` vs actual `+27.7%` (`162448.31 / 127211.59 - 1`); `1.92×` 117-case Tier-1 vs actual `1.89×` (`PS3_55`) / `1.90×` (`PS2_55`); `1.53×` MLX vs actual `1.52×` (`3.3415 / 2.19125`); `1.43–2.60×` in `BACKLOG.md` matches neither benchmark file (`1.41×` Tier-1; `1.88–2.61×` H2H LLM shapes; `1.76–2.55×` `polar_switch_blackwell_cycle1.json`); `triton_syrk` `1.69–2.02×` omits `512×2048` (`1.13×`), `1024×4096 fp16` (`1.67×`), and `[4,1024,4096] bf16` (`1.63×`).
  - **Action:** Normalize all prose metrics directly from raw JSONs (ideally generated via a verification script) and explicitly report the Apple M5 Max `-14%` square-matrix (`2048×2048`) regression alongside the `+1.52×` rectangular (`1024×4096`) win.

- **[M2] Tier-2 LM Benchmark Framing & Experimental Caveats**
  - **Evidence:** In `cycle1_tier2_lm.json`: (1) Matched-token validation loss is statistically flat across all Muon orthogonalizers (`3.0316`–`3.0439`), so the `-0.224` wall-clock gain is purely step-count throughput on a tiny synthetic task (`2,048` tok/step, 3s horizon, 2 seeds); (2) `PS3_55 (eager)` was omitted from Tier-2 (vs `PS2_55 eager`, the throughput gain is `+8.4%` and loss delta is `-0.077`, not `+27.7%` / `-0.224`); (3) LR `{0.01, 0.02}` did not bracket the optimum (`0.02` won for nearly every Muon variant); (4) Eager baselines were uncompiled (no `torch.compile` control).
  - **Action:** Add an explicit caveat to Cycle 1 reporting separating **kernel speedup** from **end-to-end LLM pretraining speedup**, and fix all four methodology gaps in Cycle 2 `T3-01`.

- **[M3] Subtle `polar_express` Implementation Divergence Between `baselines.py` and `kernels_blackwell.py`**
  - **Evidence:** `baselines.polar_express` uses Amsel's `optimal_composition(1e-3, 10, cushion=0.02, safety_factor_eps=0.01)` (`step 0: (8.237, -23.158, 16.681)`) with **BF16** norm division, whereas `kernels_blackwell.polar_express_triton` uses `_PE_RAW / 1.01^k` (`step 0: (8.205, -22.902, 16.461)`) with **FP32** norm division. This contributes to `polar_express_triton_bf16` showing lower harvested error (`0.0893`) than eager `polar_express` (`0.0958`).
  - **Action:** Align coefficients and normalization precision between eager and Triton `polar_express` so speedups and numerical deltas isolate kernel behavior cleanly.

### `LOW`

- **[L1] Stale `README.md` & Unresolved `macbook_air` Blocker**
  - **Evidence:** `README.md` still reads "Cycle 0 / Cycle 1 Initializing", cites only dual review (`Claude Opus 5.5 + GPT 6.2 Sol`) instead of the 4-model Frontier Roster in `RESEARCH_PROGRAM.md`, and presents warm-started power scaling as the active Track 1 headline without noting the Schatten-8 pivot. `macbook_air` remains permanently `blocked` in `RESEARCH_STATE.json`.
  - **Action:** Update `README.md` once GATE-1 passes and either resolve or formally park `macbook_air` (one Apple Silicon host—M5 Max—is sufficient for MLX parity).

- **[L2] Apple M5 Max Benchmark Rigor**
  - **Evidence:** `kernels_mlx.py` tests `float32` only (Muon runs in BF16/FP16), with `warmups=1, repeats=5` causing 2× cold-start latency spikes in `512×2048` samples.
  - **Action:** In Cycle 2, add `bfloat16`/`float16`, increase to `warmups>=5, repeats>=20`, and report median + IQR.

---

## 5. Cycle 2 Roadmap Stress-Test & Recommended Priorities

### Stress-Testing the Proposed Cycle 2 Plan (`T1-04` + `T3-01` + `T2-02`)

`T1-04` and `T3-01` **are** the right two pillars for Cycle 2, **provided you avoid four concrete traps:**

1. **Trap 1 (`T1-04` Kernel Architecture Mismatch):** `BACKLOG.md` proposes to *"fuse `run_circuit` directly onto `_symprod_kernel` + `_launch` to match `PS3_55` kernel speed (`0.30 ms`)"*. **This does not work as stated for Cycle 1's `circuits.py`!** `circuits.py` uses **full-history FP32 linear combinations** (`L_j = w_I I + Σ_{i≤j} w_i B_i`) across all `k+1` buffers before every squaring, whereas `_symprod_kernel` computes a 2-input update `α(A @ B) + βC + γI` on FP16/BF16 buffers.
   - **Fix:** Constrain `T1-04`'s circuit search to **Polar-Switch-compatible 2-buffer or 3-buffer register-minimal updates** (`B[dst] = a * B[src]² + b * B[add] + c * I`) so the discovered Schatten-8 circuits execute *natively* on `_symprod_kernel` with zero extra combine kernels or FP32 buffer traffic.
2. **Trap 2 (`T1-04` Finite-Precision Safety of `u = 1.005`):** Schatten-8 ($\|R^2\|_F^{1/4} \ge \sigma_{\max}$) is a rigorous upper bound in **exact arithmetic**, NOT automatically in **FP16**! In `circuits.py`, `R = _syrk(W)` and `R2 = _square(R)` undergo FP16 rounding; on tiny/ill-conditioned spectra, FP16 rounding or underflow in `R²` can slightly underestimate $\|R^2\|_F^{1/4}$. With only a `0.5%` margin (`u = 1.005`), a tiny FP16 underestimate could break stability.
   - **Fix:** Empirically audit $\max(\sigma_{\max}^{\text{fp64}} / \hat{\sigma}_{S_8}^{\text{fp16}})$ across all 117 Tier-1 cases + dedicated FP16 underflow/cancellation fuzz tests, and design the circuit basin for `u = 1.015–1.02` (or inflate `upper` by a proven FP16 safety factor) rather than gambling on `1.005`.
3. **Trap 3 (`T1-04` Objective Must Target *Fewer Squarings*, Not Just Lower Polar Error):** Cycle 1 Tier-2 proved that lowering polar error below `PS3_55` (`0.2800` vs `0.3208` max error) gave **zero** matched-token LM loss gain (`3.0402` vs `3.0361`). Therefore, matching `PS3_55` speed (`5+5 = 10` squarings) with tighter error will not win Tier-2. The real prize of Schatten-8 pre-scaling (shrinking the effective lower bound by `~2–3×`) is **strictly shorter Polar-Switch circuits**—e.g., **`PS3-44` (8 squarings) or `PS3-34` (7 squarings) or a 0-restart `k=6/7` circuit**—that run **15–30% faster than `ps3_55_blackwell_fp16`** at equal (`PS3_55`-level) accuracy!
4. **Trap 4 (`T2-02` FP8 Amdahl Ceiling & Sunk-Cost Trap):**
   - Fixing `polar_express_fp8x2_bf16` is a dead end: even without divergence, it is **20–77% slower than FP16 `ps3_55_blackwell_fp16`** on every LLM shape.
   - Even `polar_switch_blackwell(..., fp8_sketch=True)` (already implemented and benchmarked in `results/polar_switch_blackwell_cycle1.json`) only accelerates `PS3_55` by **`1.016–1.038×`** on large shapes (`2048×8192`, `4096×4096`) and slows down `512×2048` by `-36%`, because the initial `X @ X.mT` Gram is only 6–12% of Polar Switch's FLOPs (Amdahl's law). Unless you pursue FP8 inside `_symprod_kernel` itself (high numerical risk), de-prioritize FP8 in Cycle 2.

### Recommended Cycle 2 Priority Order

1. **P0 (Gate-1 Remediation — Immediate):** Fix `reports/generate_infographic.py` + `tests/test_infographic.py`, correct the FP8 diagnosis & arithmetic/rounding discrepancies in `state/RESEARCH_STATE.json` / `state/BACKLOG.md` / `state/ledger.jsonl` / `README.md`, align `polar_express` coefficients/norm precision across eager and Triton, and regenerate `reports/daily/2026-11-25.html`.
2. **P1 (`T1-04` — Highest Scientific Leverage):** Search **Schatten-8 pre-scaled, `_symprod_kernel`-native 2/3-buffer Polar Switch circuits with strictly fewer squarings than `PS3_55`** (targeting `3+4`, `4+4`, or `4+5` with `u = 1.02` finite-precision-verified basin). Goal: strictly dominate `ps3_55_blackwell_fp16` on BOTH latency (`< 0.26 ms` Tier-1 / `< 2.6 ms` at `2048×8192`) and harvested polar accuracy (`<= 0.062` median, `<= 0.30` max) with `0/117` divergence.
3. **P2 (`T3-01` — Essential Credibility Gate):** Multi-seed (`>= 3` seeds), LR-bracketed (`>= 3` LRs) **50M–120M causal LM pretraining on real FineWeb-Edu / WikiText-103** on RTX 5080 over meaningful token horizons (`>= 20M–50M` tokens). Include **`PS3_55 (eager)` AND `torch.compile` eager baselines**, test realistic tokens/step (with gradient accumulation), and report **optimizer-step ms** alongside **end-to-end tok/s** and **matched-token vs. matched-wall-clock validation perplexity**.
4. **P3 (`T2-02` Refocused — Apple Silicon MLX Only):** Drop `polar_express_fp8` remediation (Pareto-dominated); optionally verify `fp8_sketch` on the 117-case Tier-1 suite for completeness. On M5 Max, upgrade `kernels_mlx.py` to `bfloat16`/`float16` with `>=5` warmups / `>=20` repeats and prototype a custom `mx.fast.metal_kernel` triangular `SYRK` to eliminate the `-14%` square-matrix regression.
5. **Defer (`T4-01`, `T5-01`):** Keep $\mu$P coordinate checks and NVFP4/2-bit PTQ queued for Cycle 3 until Cycle 2 delivers a clean, real-data, multi-seed verified win on Track 1 + Track 3.
