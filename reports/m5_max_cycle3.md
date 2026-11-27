# Apple M5 Max · Cycle 3

All three runs used clean source commit `7eacbb6516ef71d8ff19f63c723b861d3dd799fb` under `caffeinate -dimsu`. The baseline suite passed 82 tests. The final suite passed 101 tests with zero skips. Each of the 82 power snapshots recorded AC power, battery at 80% without charging, a closed lid, and no thermal or performance warning. The per-sample wall-clock check flagged zero of 5,140 timed calls. These are Tier-1 results.

At 4096×4096 FP16, Metal SS3_54 lowered median relative polar error from 0.1259 to 0.0539 compared with Metal PS3_55. Its median paired speed relative to PS3_55 was 0.89×. The dense BF16 SS3_35 implementation keeps the folded affine sum in FP32 until the BF16 buffer write to pass the coherent rank-one gate.

## Primitive ratios

Each cell gives the median / IQR width / minimum of the 20 within-block latency ratios. Values above 1 favor the denominator. Gram compares dense Gram with Metal SYRK. The other columns compare the dense affine expression, Metal SYRK with a separate epilogue (unfused), and Metal symprod (fused). All custom primitives used 64×64 tiles.

| Shape | Dtype | Dense Gram / SYRK | Dense affine / unfused | Dense affine / fused | Unfused / fused |
|---|---|---:|---:|---:|---:|
| 512×2048 | FP16 | 1.05 / 0.25 / 0.83 | 0.98 / 0.23 / 0.40 | 1.22 / 0.10 / 0.94 | 1.26 / 0.25 / 0.93 |
| 512×2048 | BF16 | 1.02 / 0.23 / 0.68 | 1.04 / 0.19 / 0.48 | 1.28 / 0.19 / 1.00 | 1.26 / 0.61 / 1.04 |
| 512×2048 | FP32 | 0.70 / 0.11 / 0.45 | 0.69 / 0.17 / 0.46 | 0.78 / 0.12 / 0.46 | 1.15 / 0.24 / 0.51 |
| 1024×4096 | FP16 | 1.37 / 0.27 / 0.92 | 1.33 / 0.18 / 0.89 | 1.72 / 0.37 / 1.21 | 1.38 / 0.24 / 0.81 |
| 1024×4096 | BF16 | 1.34 / 0.24 / 1.10 | 1.19 / 0.24 / 0.75 | 1.59 / 0.20 / 1.27 | 1.38 / 0.28 / 1.08 |
| 1024×4096 | FP32 | 0.69 / 0.06 / 0.58 | 0.74 / 0.12 / 0.59 | 0.79 / 0.08 / 0.64 | 1.09 / 0.20 / 0.84 |
| 2048×2048 | FP16 | 1.35 / 0.22 / 0.83 | 1.11 / 0.14 / 0.66 | 2.17 / 0.30 / 1.53 | 1.89 / 0.27 / 1.49 |
| 2048×2048 | BF16 | 1.26 / 0.20 / 1.03 | 1.17 / 0.19 / 0.82 | 2.15 / 0.35 / 1.76 | 1.93 / 0.35 / 1.52 |
| 2048×2048 | FP32 | 0.67 / 0.09 / 0.55 | 0.79 / 0.16 / 0.52 | 1.04 / 0.28 / 0.83 | 1.30 / 0.23 / 1.16 |
| 4096×4096 | FP16 | 1.63 / 0.15 / 1.43 | 1.25 / 0.11 / 1.10 | 2.99 / 0.48 / 2.45 | 2.45 / 0.23 / 1.69 |
| 4096×4096 | BF16 | 1.68 / 0.06 / 1.53 | 1.27 / 0.03 / 1.22 | 3.04 / 0.09 / 2.43 | 2.37 / 0.08 / 1.67 |
| 4096×4096 | FP32 | 0.69 / 0.02 / 0.65 | 0.76 / 0.02 / 0.71 | 0.99 / 0.05 / 0.87 | 1.31 / 0.04 / 1.12 |

## Interleaved end-to-end speedups

Each ratio divides dense latency by Metal latency in the same block. Each row pools 20 blocks for each of three seeds. BF16 SS3_54 and SS3_53 were excluded for all four shapes because those circuits lack BF16 verification. All Metal arms used the 64-tile kernel.

| Shape | Dtype | Method | Blocks | Median | P25 | P75 | IQR | Min | Max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512×2048 | FP16 | polar_express | 60 | 0.992 | 0.964 | 1.014 | 0.050 | 0.810 | 1.250 |
| 512×2048 | FP16 | PS3_55 | 60 | 1.257 | 1.226 | 1.289 | 0.062 | 1.065 | 1.763 |
| 512×2048 | FP16 | SS3_54 | 60 | 1.208 | 1.184 | 1.231 | 0.047 | 0.911 | 1.596 |
| 512×2048 | FP16 | SS3_53 | 60 | 1.196 | 1.169 | 1.205 | 0.037 | 0.841 | 1.460 |
| 512×2048 | FP16 | SS3_35 | 60 | 1.183 | 1.168 | 1.205 | 0.037 | 0.873 | 1.250 |
| 512×2048 | BF16 | polar_express | 60 | 0.983 | 0.960 | 1.009 | 0.049 | 0.847 | 1.149 |
| 512×2048 | BF16 | PS3_55 | 60 | 1.242 | 1.196 | 1.270 | 0.074 | 1.083 | 1.495 |
| 512×2048 | BF16 | SS3_35 | 60 | 1.158 | 1.139 | 1.179 | 0.039 | 1.018 | 1.335 |
| 1024×4096 | FP16 | polar_express | 60 | 1.164 | 1.151 | 1.174 | 0.023 | 1.070 | 1.207 |
| 1024×4096 | FP16 | PS3_55 | 60 | 1.662 | 1.632 | 1.689 | 0.058 | 1.554 | 1.749 |
| 1024×4096 | FP16 | SS3_54 | 60 | 1.539 | 1.502 | 1.564 | 0.062 | 1.418 | 1.700 |
| 1024×4096 | FP16 | SS3_53 | 60 | 1.452 | 1.419 | 1.487 | 0.068 | 1.363 | 1.556 |
| 1024×4096 | FP16 | SS3_35 | 60 | 1.455 | 1.418 | 1.492 | 0.074 | 1.328 | 1.652 |
| 1024×4096 | BF16 | polar_express | 60 | 1.162 | 1.153 | 1.179 | 0.025 | 1.120 | 1.238 |
| 1024×4096 | BF16 | PS3_55 | 60 | 1.662 | 1.612 | 1.687 | 0.075 | 1.535 | 1.741 |
| 1024×4096 | BF16 | SS3_35 | 60 | 1.489 | 1.464 | 1.514 | 0.051 | 1.365 | 1.632 |
| 2048×2048 | FP16 | polar_express | 60 | 1.188 | 1.182 | 1.196 | 0.014 | 1.165 | 1.211 |
| 2048×2048 | FP16 | PS3_55 | 60 | 2.489 | 2.464 | 2.520 | 0.055 | 2.405 | 2.599 |
| 2048×2048 | FP16 | SS3_54 | 60 | 2.080 | 2.067 | 2.092 | 0.025 | 2.013 | 2.155 |
| 2048×2048 | FP16 | SS3_53 | 60 | 1.962 | 1.945 | 1.976 | 0.031 | 1.915 | 2.013 |
| 2048×2048 | FP16 | SS3_35 | 60 | 1.998 | 1.974 | 2.020 | 0.046 | 1.944 | 2.077 |
| 2048×2048 | BF16 | polar_express | 60 | 1.183 | 1.175 | 1.190 | 0.015 | 1.145 | 1.218 |
| 2048×2048 | BF16 | PS3_55 | 60 | 2.461 | 2.444 | 2.487 | 0.043 | 2.397 | 2.540 |
| 2048×2048 | BF16 | SS3_35 | 60 | 1.977 | 1.959 | 1.993 | 0.034 | 1.930 | 2.036 |
| 4096×4096 | FP16 | polar_express | 60 | 1.280 | 1.251 | 1.307 | 0.056 | 1.140 | 1.414 |
| 4096×4096 | FP16 | PS3_55 | 60 | 2.484 | 2.328 | 2.543 | 0.214 | 2.133 | 2.713 |
| 4096×4096 | FP16 | SS3_54 | 60 | 2.139 | 2.099 | 2.173 | 0.074 | 1.949 | 2.289 |
| 4096×4096 | FP16 | SS3_53 | 60 | 2.037 | 1.962 | 2.062 | 0.101 | 1.791 | 2.674 |
| 4096×4096 | FP16 | SS3_35 | 60 | 2.078 | 2.006 | 2.109 | 0.103 | 1.893 | 2.293 |
| 4096×4096 | BF16 | polar_express | 60 | 1.254 | 1.248 | 1.269 | 0.020 | 1.176 | 1.384 |
| 4096×4096 | BF16 | PS3_55 | 60 | 2.505 | 2.481 | 2.521 | 0.041 | 2.369 | 2.652 |
| 4096×4096 | BF16 | SS3_35 | 60 | 2.127 | 2.107 | 2.145 | 0.038 | 1.917 | 2.276 |

## SS3 accuracy and speed against PS3_55

Error is the median relative Frobenius error against CPU FP64 SVD over the three input seeds. Each SS3 cell gives error / speed relative to PS3_55 in the same arm. Speed is the median of 60 paired PS3_55 / SS3 latency ratios. Values above 1 favor SS3. A dash marks a BF16 safety exclusion.

| Shape | Dtype | Arm | PS3_55 error | SS3_54 | SS3_53 | SS3_35 |
|---|---|---|---:|---:|---:|---:|
| 512×2048 | FP16 | dense | 0.0637 | 0.0372 / 0.86× | 0.0594 / 0.90× | 0.0599 / 0.92× |
| 512×2048 | FP16 | metal | 0.0636 | 0.0372 / 0.84× | 0.0594 / 0.85× | 0.0599 / 0.87× |
| 512×2048 | BF16 | dense | 0.0636 | — | — | 0.0610 / 0.93× |
| 512×2048 | BF16 | metal | 0.0632 | — | — | 0.0608 / 0.86× |
| 1024×4096 | FP16 | dense | 0.0649 | 0.0372 / 0.97× | 0.0603 / 1.02× | 0.0593 / 1.05× |
| 1024×4096 | FP16 | metal | 0.0647 | 0.0372 / 0.89× | 0.0603 / 0.89× | 0.0593 / 0.92× |
| 1024×4096 | BF16 | dense | 0.0645 | — | — | 0.0609 / 1.01× |
| 1024×4096 | BF16 | metal | 0.0625 | — | — | 0.0605 / 0.90× |
| 2048×2048 | FP16 | dense | 0.1148 | 0.0557 / 1.00× | 0.0770 / 1.06× | 0.0777 / 1.07× |
| 2048×2048 | FP16 | metal | 0.1145 | 0.0555 / 0.83× | 0.0770 / 0.83× | 0.0777 / 0.86× |
| 2048×2048 | BF16 | dense | 0.1158 | — | — | 0.0825 / 1.07× |
| 2048×2048 | BF16 | metal | 0.1140 | — | — | 0.0813 / 0.86× |
| 4096×4096 | FP16 | dense | 0.1260 | 0.0540 / 1.03× | 0.0762 / 1.11× | 0.0765 / 1.12× |
| 4096×4096 | FP16 | metal | 0.1259 | 0.0539 / 0.89× | 0.0762 / 0.91× | 0.0765 / 0.94× |
| 4096×4096 | BF16 | dense | 0.1275 | — | — | 0.0819 / 1.10× |
| 4096×4096 | BF16 | metal | 0.1267 | — | — | 0.0803 / 0.94× |

## FP32 dispatch

The FP32 run requested Metal on the command line. Every baseline method ran dense.

| Shape | Methods | Metal SYRK rows | Non-null SYRK tiles | Metal symprod rows | Divergences |
|---|---:|---:|---:|---:|---:|
| 1024×4096 | 5 | 0 | 0 | 0 | 0 |
| 2048×2048 | 5 | 0 | 0 | 0 | 0 |

Raw data and commands are in the [primitive](../results/m5_max_primitives_cycle3.json), [paired A/B](../results/m5_max_ab_ss3_cycle3.json), and [FP32 gate](../results/m5_max_fp32_gate_check_cycle3.json) files.
