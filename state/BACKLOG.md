# Research Backlog

## Cycle 1 (Integrated — Pending 4-Model Adversarial Audit)
- [x] **T1-01**: Implemented verified late-2026 SOTA baselines (`svd_polar_fp64`, `jordan_ns5`, `polar_express`, `gram_ns`, `polar_switch PS2-55 & PS3-55`) (`bazzite`, GPT 6.2 Sol).
- [x] **T1-02**: Harvested real 19.2M Transformer momentum trajectories `(M_t, M_{t+1})` (median eff-rank `3.19×`, right overlap `0.937`, max `Δσ_max = 31.82%`) (`bazzite`, GPT 6.2 Sol).
- [x] **T1-03**: Warm-started spectral-norm estimator + rounding-aware polynomial circuits (`Claude Opus 5.5`); discovered step-11 multi-step drift vulnerability of heuristic warm-power scaling (`true/σ_hat = 1.191`) and pivoted to stateless rigorous **Schatten-8 upper-bound scaling (`scaling="schatten8"`)** (`0/72` Tier-2 divergence).
- [x] **T2-01**: Consumer Blackwell (`sm_120`) autotuned `triton_syrk` (`1.69–2.02×` vs cuBLAS), FP8-early PolarExpress (caught 9% adversarial divergence bug), and Blackwell Triton `PolarSwitch` `PS2_55`/`PS3_55` (`1.43–2.60×` vs PolarExpress, **+27.8% Tier-2 LM tok/s**, `-0.224` wall-clock val loss) (`bazzite`, Claude Sonnet 5.5 + Claude Fable 5.2).
- [x] **T0-01**: Responsive T3-themed Daily Infographic Engine + Apple Silicon M5 Max MLX/MPS benchmark (`PS3_55` MLX `2.191 ms`, `1.53×` faster than `PolarExpress` MLX) (`bazzite` + `MacBook Pro`, GPT 6.2 Astra).
- [ ] **GATE-1**: Unified Tier-1 + Tier-2 RTX 5080 verification & 4-Model Adversarial Audit (`Claude Opus 5.5`, `Claude Fable 5.2`, `GPT 6.2 Astra`, `Gemini 4 Neon` at `high`).

## Cycle 2 (Top Autonomous Priorities)
- [ ] **T1-04**: **Zero-Overshoot `u=1.005` Schatten-8 Polar Switch Circuits + Fused Triton `_symprod` Implementation** — because Schatten-8 is a rigorous upper bound (`σ_max / σ_hat <= 1.0`), eliminate the `[1, 1.15]` basin penalty and fuse `run_circuit` directly onto `_symprod_kernel` + `_launch` to match `PS3_55` kernel speed (`0.30 ms`) with tighter tail convergence.
- [ ] **T2-02**: Investigate safe FP8-early (`_scaled_mm`) step clamping / soft-coefficient schedules (fixing the 9% adversarial divergence found in Cycle 1 `polar_express_fp8`) and Apple Silicon MLX custom Metal/vectorized `Schatten-8` circuits (`MacBook Pro` + `bazzite`).
- [ ] **T3-01**: Multi-seed 50M–120M LM pretraining transfer on real FineWeb/WikiText tokens on RTX 5080 (moving beyond the synthetic 8-token motif stream).
- [ ] **T4-01**: Automated $\mu$P coordinate-check harness across widths ($128 \to 1024$) for hybrid Muon/AdamW variants.
- [ ] **T5-01**: Blackwell NVFP4 / 2-bit Micro-Rotated PTQ (adaptive butterfly/Hadamard rotations + Hessian rounding).
