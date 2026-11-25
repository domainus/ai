# Autonomous Verifiable AI Research Program

## Core Scientific Principles
1. **Programmatic Verification First:** Every hypothesis must map to a deterministic or tightly controlled verification harness measuring both **quality/accuracy** and **guardrail metrics** (worst-case singular value bounds, NaN/divergence rate, randomized fuzz shapes, OOD/multi-seed generalization).
2. **No Cherry-Picking / Anti-Overfitting:** Discoveries on Tier-1 synthetic/harvested matrices only promote to `VERIFIED` after passing adversarial stress tests AND Tier-2 end-to-end training checks across multiple random seeds. Negative or inconclusive results are logged honestly in `state/ledger.jsonl`.
3. **Mandatory Dual Cross-Model Adversarial Review:** Every cycle's implementation and claims must be reviewed independently by at least two distinct frontier model families (e.g., Claude Opus 5.5 + GPT 6.2 Sol), actively trying to break assumptions, spot timing bugs, or construct mathematical counterexamples.

## Active Research Tracks

### Track 1 (Flagship): Warm-Started Spectral Pre-Scaling & Ultra-Short Polar Circuits
- **Hypothesis:** In Muon optimization, consecutive momentum matrices $M_t = \beta M_{t-1} + (1-\beta) G_t$ have high top-singular-vector overlap. A 1–2 matvec warm-started power iteration + calibrated safety margin $\gamma$ yields a tight upper bound on $\sigma_{\max}(M_t)$, shrinking the effective condition interval from $[10^{-3}, 1]$ to $[10^{-2}, 1]$ and enabling 0-restart or compact polynomial circuits that beat Polar Express and Polar Switch in latency at equal or better orthogonality error.
- **Verification Gates:**
  - *Tier 1:* Exact `fp64` SVD polar error, singular value containment $[\sigma_{\min}, \sigma_{\max}]$, 0% divergence on adversarial spikes ($\kappa = 10^6$, orthogonal warm-start traps, rank-1 spikes), and real wall-clock GPU latency.
  - *Tier 2:* Multi-seed causal Transformer pretraining loss at matched LR sweeps and fixed wall-clock/token budgets.

### Track 2: Consumer Blackwell (`sm_120`) Triton SYRK, FP8-Early/BF16-Late & Apple Silicon MLX Kernels
- Autotuned Triton symmetric `XX^T` (`SYRK`) and fused polynomial kernels for RTX 5080 (`sm_120`).
- `torch.float8_e4m3fn` (`_scaled_mm`) early-step polar iterations transitioning to FP16/BF16 polishing.
- Apple Silicon (`MLX` / `MPS`) polar orthogonalization benchmarks and kernels.

### Track 3: Multi-Scale Real-Data LLM Pretraining Transfer
- Scale-up verification (20M -> 120M+ parameter causal Transformers on FineWeb/WikiText) checking wall-clock speedup and validation loss parity vs `PolarExpress` and `PS2-55`.

### Track 4: Layer-Adaptive Preconditioning & $\mu$P Transfer
- Automated coordinate-check verification across widths ($128 \to 256 \to 512 \to 1024$) for hybrid Muon/AdamW and spectral-scaled updates.

### Track 5: Blackwell NVFP4 & 2-Bit Micro-Rotated Post-Training Quantization
- Adaptive butterfly/block Hadamard rotations and Hessian-aware rounding for NVFP4/MXFP4/2-bit PTQ verified via perplexity and deterministic eval deltas.
