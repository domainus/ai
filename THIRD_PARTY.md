# Third-Party Credits and Reference Baselines

All third-party baselines implemented or compared in this repository are re-implemented cleanly for reproducible benchmarking and properly credited below:

1. **Muon Optimizer** (Keller Jordan et al.)
   - Repository: https://github.com/KellerJordan/Muon
   - Standard 5-step Newton-Schulz coefficients `(3.4445, -4.7750, 2.0315)`.

2. **The Polar Express** (Noah Amsel, David Persson, Christopher Musco, Robert M. Gower, ICLR 2026 / arXiv:2505.16932)
   - Repository: https://github.com/NoahAmsel/PolarExpress (MIT License)
   - Remez-optimal quintic coefficients and safety factor composition (`coeffs_list`).

3. **Gram Newton-Schulz** (Jack Zhang, Noah Amsel, Berlin Chen, Tri Dao, 2026)
   - Repository: https://github.com/Dao-AILab/gram-newton-schulz
   - Gram-space iteration formulation with restart schedule for rectangular matrices.

4. **Polar Switch** (Zhehang Du, Weijie Su, arXiv:2610.26742)
   - Repository: https://github.com/zhehangdu/polar-switch (MIT License)
   - Polynomial circuit formulation (`PS2-55`, `PS3-55`) using symmetric matrix product primitives.
