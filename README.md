# domainus/ai — Autonomous Verifiable AI Research Lab

An open, continuously operating autonomous AI research repository investigating **verifiable, programmatically grounded machine learning discoveries** across a multi-machine fleet (`bazzite` RTX 5080 Blackwell `sm_120`, Apple Silicon MacBook Pro & MacBook Air) with cross-model implementation and mandatory dual adversarial review (Claude + ChatGPT/Codex + Gemini Orchestrator).

## Flagship Program: Low-Precision Polar Orthogonalization & Hardware-Aware Muon Kernels

Muon-class optimizers rely on fast polynomial approximations of the matrix polar factor $U V^\top$ of 2D momentum tensors. Recent 2025–2026 advances—**Polar Express** (`arXiv:2505.16932`), **Gram Newton-Schulz**, and **Polar Switch** (`arXiv:2610.26742`)—still share open gaps:

1. **Conservative Frobenius Pre-Scaling:** Dividing by $\|M\|_F$ compresses real LLM momentum spectra by $\sqrt{r_{\text{eff}}} \approx 3\times\text{–}20\times$, forcing circuits to span $[10^{-3}, 1]$ with restarts. We investigate **warm-started spectral-norm upper-bound pre-scaling** with mathematical overshoot guards to unlock shorter / 0-restart circuits.
2. **Consumer Blackwell (`sm_120`) & Mixed Precision:** Existing kernels target H100/B200 QuACK. We develop autotuned Triton symmetric `SYRK` kernels and **FP8-early / BF16-late** mixed-precision polar iterations on RTX 5080 (`sm_120`).
3. **Apple Silicon (`MLX`/`MPS`) Parity:** Lightweight polar orthogonalization benchmarks and kernels for Apple Silicon unified memory.

## Repository Layout

- `research/polar_opt/` — Baselines, real momentum harvesting, spectral pre-scaling, polynomial circuit search, Blackwell Triton/FP8 kernels, Apple Silicon MLX/MPS kernels, Tier-1 & Tier-2 verification harnesses.
- `tests/` — Deterministic unit tests, adversarial spectral counterexamples, and kernel numerical-equivalence suites.
- `results/` — Machine-readable JSON benchmark records (no unverified claims).
- `audits/` — Independent cross-model adversarial review reports (`Claude Opus 5.5` + `GPT 6.2 Sol`).
- `reports/` — Automated daily HTML infographic generator & published daily reports.
- `state/` — Persistent research state (`RESEARCH_STATE.json`, `ledger.jsonl`, `BACKLOG.md`).

## Status

> **Cycle 0 / Cycle 1 Initializing** — Setting up verified baselines, RTX 5080 (`sm_120`) harness, fleet synchronization, and initial Cycle 1 discovery experiments. Verified leaderboards and audit links will populate here as experiments pass dual adversarial review.

## Research Roadmap

See [RESEARCH_PROGRAM.md](RESEARCH_PROGRAM.md) and [state/BACKLOG.md](state/BACKLOG.md) for the multi-track autonomous research queue covering Track 1 (Spectral-Prescaled Polar Circuits), Track 2 (Blackwell Triton/FP8 & MLX Kernels), Track 3 (Multi-Scale LM Pretraining Transfer), Track 4 (Hybrid Optimizer / $\mu$P Transfer), and Track 5 (Blackwell NVFP4 & 2-Bit Micro-Rotated PTQ).
