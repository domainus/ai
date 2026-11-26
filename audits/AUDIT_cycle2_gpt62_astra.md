# Cycle 2 Tier-2 audit · real-text validity, statistics and telemetry

**Executive Verdict — PASS_WITH_CAVEATS**

The 100 raw runs support a narrower claim than “one Muon variant wins.” On the RTX 5080, PS3 eager, PS3 Blackwell, SS3_54 and SS3_53 reach lower test loss than eager Polar Express at equal **training compute time**. All four pass a two-sided paired t-test, including a Holm correction across the six comparisons to the reference in that budget. None shows a fixed-token test-loss advantage over eager Polar Express. Neither SS variant beats PS3 Blackwell in test loss under either budget. At fixed tokens, PS3 Blackwell is 0.72% faster than PS3 eager, but its test loss is 0.00454 higher at nominal p = 0.0282. The loss difference loses significance when all 15 Muon pairs are considered.

The common FP32 update path removes the Cycle 1 dtype mismatch at the parameter add. Across the three matched seeds, Blackwell’s throughput lead over eager Polar Express is 3.19% with FP32 updates and 3.17% with native updates. Most of the measured optimizer-time gap is in orthogonalization. Seed 5 weakens the fixed-wall Blackwell result. It stays in every primary estimate.

## Scope and integrity

Audited tree `877c2a8413e05d89f105360a550eb382e464ba70`. The five artifacts report its direct parent `9a23a6176726465cd65a0d55e959ef5b17462b62` and `git_dirty: false`. The harness, data builder and test file have identical contents at both commits. The recorded hardware is one RTX 5080, PyTorch 2.14.1+cu130 and CUDA 13.0. The “50m” model has 64,303,616 total parameters, including 38,547,968 outside the embedding. This was a CPU-only audit of existing GPU measurements. GPU dispatch tests were read as source.

The requested `.venv/bin/python research/polar_opt/real_text_data.py --verify` exited 0 with CUDA hidden. It checked the shard lengths, file SHA-256 values, token bounds, EOT counts, document-hash digests, duplicates and split intersections. I independently checked every recorded SHA-1 document ID against its split assignment and compared each artifact’s source, encoding, token counts and all three shard hashes against the manifest. All matched. The hash of the **manifest file itself** is `f469022a2c509c6d80f46f843e4c4dad76530f68d32ea5d52d65a036c8db6c4a`. The artifacts store the shard hashes. The splits contain FineWeb-Edu sample-10BT with GPT-2 tokenization. [Manifest](../artifacts/real_text/manifest.json#L17) · `research/polar_opt/real_text_data.py:94–117,164–193` · `research/polar_opt/harness_tier2_real.py:145–148`

| Split | Tokens | Documents | Shard SHA-256 |
|---|---:|---:|---|
| Train | 26,004,518 | 24,256 | `015ab83ceb742555c2398963a0bd1f3c5502dc52cf70a22bf173f1571326ca3a` |
| Validation | 2,002,330 | 1,804 | `371b64a537f7b1e306ea174a868b05932a254a98b7fbdbf4146648d93180b0b1` |
| Test | 2,000,081 | 1,960 | `4eb46f1424856849d2973923dbf3c886d37eb743cf22053fc7e1eea1f41ab92a` |

I rebuilt the 16-hex config hashes from artifact arguments, model config and shard digests. Every row matched. Distinct budgets and pilot settings explain the four hashes. Layout is a separate field in the run key. The AdamW pilot hash includes its CLI auxiliary LR of 0.002 even though AdamW does not use that parameter. The other pilots specify 0.001. Source for the hash and key is `research/polar_opt/harness_tier2_real.py:798–817,903–918`. Each file records Git provenance at lines 3–4.

| Artifact in `results/` | Raw run array starts at line | Data splits at line | Rows | Seeds | Recomputed run config hash |
|---|---:|---:|---:|---|---|
| cycle2_tier2_pilot_adamw.json | 205 | 92 | 5 | 0 | tokens `bdb2f6b3b68c5e82` |
| cycle2_tier2_pilot_muon.json | 476 | 98 | 16 | 0 | tokens `f8986fcdd901dc7b` |
| cycle2_tier2_pilot_ss3_53.json | 185 | 96 | 3 | 0 | tokens `f8986fcdd901dc7b` |
| cycle2_tier2_real.json | 1034 | 114 | 70 | 1–5 | tokens `c8d85b6e90bb3296`, wall `bc391caca4de9246` |
| cycle2_tier2_real_native.json | 231 | 98 | 6 | 1–3 | tokens `c8d85b6e90bb3296` |

All 100 rows have finite validation loss, test loss and throughput, and zero rows set `diverged`. In each of the ten main seed/budget blocks, all seven optimizers occur exactly once. Within a seed, every optimizer, budget and available layout has the same fingerprint pair. I also regenerated the stream fingerprints independently from the first three sets of NumPy window starts. At equal tokens the full stream is deterministic by construction. Wall runs share its prefix up to the shorter run. `research/polar_opt/harness_tier2_real.py:114–125,151–176,544–566`

| Seed | Initial-state fingerprint | Stream fingerprint |
|---:|---|---|
| 0 | `450226adb57b95ef` | `c799a807f454d68f` |
| 1 | `85659cbc0ffc79c1` | `b9f91c01f18ce738` |
| 2 | `eff774e6f234e5e7` | `f32172b52cfa7de3` |
| 3 | `48a8ed2b9025f468` | `e88efd98cfc4e5e9` |
| 4 | `5991a96f467739ab` | `d4afd0616d1db977` |
| 5 | `ed4aff32b452b660` | `01366092adfa364f` |

### Learning-rate separation

The pilot used 128 steps (4,194,304 tokens), seed 0 and 16 warmup steps. Its final row ended at 17:45:48. The main sweep’s first row began at 17:48:05 on the same date and timezone. It used held-out seeds 1–5 and only one LR per optimizer under both budgets. AdamW used 0.001, all Muons used 0.02 and their auxiliary AdamW used 0.001. The separate native ablation reused confirmatory seeds 1–3. Sorting raw pilot validation losses gives the following winners. For PE compiled and PS3 eager the main run inherited the family rate. The pilots contain no independent LR sweep for those two implementations. `results/cycle2_tier2_pilot_adamw.json:29`, `results/cycle2_tier2_pilot_muon.json:35`, `results/cycle2_tier2_pilot_ss3_53.json:33`, `results/cycle2_tier2_real.json:51,1034`

| Pilot optimizer | Best LR | Seed-0 validation loss | Test loss Δ vs PE eager | Speed ratio vs PE eager |
|---|---:|---:|---:|---:|
| adamw | 0.001 | 6.686017 | +0.527149 | 1.041936 |
| pe_eager | 0.02 | 6.154609 | +0.000000 | 1.000000 |
| ps3_blackwell | 0.02 | 6.161580 | +0.006994 | 1.027577 |
| ss3_34 | 0.02 | 6.197867 | +0.042061 | 1.029190 |
| ss3_54 | 0.02 | 6.159499 | +0.004981 | 1.027305 |
| ss3_53 | 0.02 | 6.161560 | +0.007185 | 1.027516 |

All pilot winners are interior to their grids. AdamW tested 0.0003, 0.0006, 0.001, 0.002 and 0.003. The main Muon pilot tested 0.005, 0.01, 0.02 and 0.04. SS3_53 tested the last three. SS3_34 occurs only in the pilot. Pilot contrasts have one paired observation, so SD and CI are undefined. AdamW and SS3_53 use the same seed-0 PE eager row from the separate Muon pilot as a descriptive reference. Pilot test loss is available for every LR, so the files establish held-out **seed** separation and consistent validation choices. They cannot establish blinding to the test shard or an undocumented model-selection decision.

## Independent statistical recomputation tables

I joined raw rows on seed within each budget and layout. For every pair A/B I computed the per-seed loss difference A − B and throughput ratio A/B, then the arithmetic mean, sample SD and interval `mean ± t(0.975, n−1) × SD / √n`. A negative difference favors A. A ratio above 1 favors A. Two-sided p-values test 0 for loss and 1 for the ratio. Timing inference uses one value per seed. It does not treat steps as replicates. All 16 published paired entries across the five JSONs reproduce in both metrics to absolute tolerance 1e−12, including the undefined pilot intervals. Nine contrasts per main budget cover the requested comparisons. The other seven complete the set of 15 Muon pairs. Optimizer names below omit the common `muon_` prefix.

### Fixed tokens · 10,485,760 tokens, 320 steps, five paired seeds

| A | B | Loss Δ mean | SD | 95% CI | p | Speed ratio mean | SD | 95% CI | p |
|---|---|---:|---:|---|---:|---:|---:|---|---:|
| adamw | pe_eager | 0.622417 | 0.033218 | [0.581171, 0.663663] | <0.0001 | 1.049601 | 0.000296 | [1.049234, 1.049969] | <0.0001 |
| pe_compiled | pe_eager | -0.003731 | 0.008693 | [-0.014524, 0.007062] | 0.3915 | 1.001946 | 0.000780 | [1.000977, 1.002914] | 0.0051 |
| ps3_eager | pe_eager | -0.004866 | 0.005755 | [-0.012011, 0.002280] | 0.1317 | 1.024454 | 0.000280 | [1.024107, 1.024802] | <0.0001 |
| ps3_blackwell | pe_eager | -0.000328 | 0.008305 | [-0.010639, 0.009983] | 0.9339 | 1.031806 | 0.000235 | [1.031514, 1.032098] | <0.0001 |
| ss3_54 | pe_eager | -0.003187 | 0.011313 | [-0.017234, 0.010859] | 0.5629 | 1.030963 | 0.000202 | [1.030712, 1.031213] | <0.0001 |
| ss3_53 | pe_eager | -0.007744 | 0.009889 | [-0.020023, 0.004535] | 0.1548 | 1.031681 | 0.000279 | [1.031334, 1.032028] | <0.0001 |
| ss3_54 | ps3_blackwell | -0.002859 | 0.008658 | [-0.013609, 0.007891] | 0.5012 | 0.999183 | 0.000183 | [0.998956, 0.999410] | 0.0006 |
| ss3_53 | ps3_blackwell | -0.007416 | 0.008885 | [-0.018449, 0.003616] | 0.1354 | 0.999879 | 0.000098 | [0.999758, 1.000000] | 0.0506 |
| ps3_blackwell | ps3_eager | 0.004538 | 0.003017 | [0.000792, 0.008283] | 0.0282 | 1.007176 | 0.000202 | [1.006925, 1.007427] | <0.0001 |
| ps3_eager | pe_compiled | -0.001134 | 0.004374 | [-0.006565, 0.004296] | 0.5930 | 1.022465 | 0.000822 | [1.021444, 1.023486] | <0.0001 |
| ps3_blackwell | pe_compiled | 0.003403 | 0.003175 | [-0.000539, 0.007345] | 0.0746 | 1.029802 | 0.000688 | [1.028949, 1.030656] | <0.0001 |
| ss3_54 | pe_compiled | 0.000544 | 0.009838 | [-0.011671, 0.012759] | 0.9076 | 1.028961 | 0.000662 | [1.028139, 1.029782] | <0.0001 |
| ss3_53 | pe_compiled | -0.004013 | 0.007742 | [-0.013627, 0.005600] | 0.3109 | 1.029678 | 0.000754 | [1.028741, 1.030614] | <0.0001 |
| ss3_54 | ps3_eager | 0.001678 | 0.007832 | [-0.008047, 0.011404] | 0.6568 | 1.006353 | 0.000325 | [1.005950, 1.006757] | <0.0001 |
| ss3_53 | ps3_eager | -0.002879 | 0.007525 | [-0.012222, 0.006465] | 0.4405 | 1.007054 | 0.000189 | [1.006820, 1.007289] | <0.0001 |
| ss3_53 | ss3_54 | -0.004557 | 0.006965 | [-0.013206, 0.004091] | 0.2173 | 1.000697 | 0.000273 | [1.000358, 1.001036] | 0.0047 |

For SS3_53 / PS3 Blackwell the upper throughput bound is 1.000000495 before rounding. The interval includes 1. All five alternatives to PE eager among Muons have speed ratios above 1 with nominal p < 0.05. PS3 eager has lower loss than PS3 Blackwell at nominal significance. That is the sole nominal loss difference among the 15 Muon pairs (Holm-adjusted p = 0.4231). AdamW is faster than PE eager but has substantially higher test loss.

### Fixed training compute time · 75 s, five paired seeds

| A | B | Loss Δ mean | SD | 95% CI | p | Speed ratio mean | SD | 95% CI | p |
|---|---|---:|---:|---|---:|---:|---:|---|---:|
| adamw | pe_eager | 0.641445 | 0.133321 | [0.475905, 0.806985] | 0.0004 | 1.050205 | 0.000250 | [1.049895, 1.050515] | <0.0001 |
| pe_compiled | pe_eager | -0.000848 | 0.005357 | [-0.007500, 0.005804] | 0.7414 | 1.001534 | 0.000304 | [1.001156, 1.001912] | 0.0004 |
| ps3_eager | pe_eager | -0.026037 | 0.003603 | [-0.030511, -0.021563] | <0.0001 | 1.024539 | 0.000143 | [1.024362, 1.024717] | <0.0001 |
| ps3_blackwell | pe_eager | -0.026162 | 0.012490 | [-0.041671, -0.010653] | 0.0094 | 1.027037 | 0.010154 | [1.014430, 1.039645] | 0.0040 |
| ss3_54 | pe_eager | -0.032860 | 0.005387 | [-0.039550, -0.026171] | 0.0002 | 1.030911 | 0.000107 | [1.030778, 1.031044] | <0.0001 |
| ss3_53 | pe_eager | -0.031440 | 0.007306 | [-0.040511, -0.022369] | 0.0007 | 1.030912 | 0.001218 | [1.029400, 1.032424] | <0.0001 |
| ss3_54 | ps3_blackwell | -0.006699 | 0.010746 | [-0.020042, 0.006645] | 0.2358 | 1.003851 | 0.010025 | [0.991404, 1.016299] | 0.4387 |
| ss3_53 | ps3_blackwell | -0.005279 | 0.017616 | [-0.027151, 0.016594] | 0.5395 | 1.003852 | 0.010137 | [0.991265, 1.016439] | 0.4433 |
| ps3_blackwell | ps3_eager | -0.000124 | 0.013951 | [-0.017447, 0.017198] | 0.9850 | 1.002438 | 0.009846 | [0.990212, 1.014663] | 0.6093 |
| ps3_eager | pe_compiled | -0.025190 | 0.003530 | [-0.029573, -0.020807] | <0.0001 | 1.022970 | 0.000245 | [1.022665, 1.023274] | <0.0001 |
| ps3_blackwell | pe_compiled | -0.025314 | 0.012160 | [-0.040413, -0.010215] | 0.0096 | 1.025464 | 0.010107 | [1.012914, 1.038014] | 0.0049 |
| ss3_54 | pe_compiled | -0.032013 | 0.003265 | [-0.036066, -0.027959] | <0.0001 | 1.029332 | 0.000210 | [1.029071, 1.029592] | <0.0001 |
| ss3_53 | pe_compiled | -0.030593 | 0.007630 | [-0.040066, -0.021119] | 0.0009 | 1.029333 | 0.001294 | [1.027726, 1.030940] | <0.0001 |
| ss3_54 | ps3_eager | -0.006823 | 0.004067 | [-0.011873, -0.001773] | 0.0199 | 1.006220 | 0.000092 | [1.006105, 1.006334] | <0.0001 |
| ss3_53 | ps3_eager | -0.005403 | 0.006339 | [-0.013274, 0.002468] | 0.1294 | 1.006220 | 0.001175 | [1.004762, 1.007679] | 0.0003 |
| ss3_53 | ss3_54 | 0.001420 | 0.008946 | [-0.009688, 0.012528] | 0.7406 | 1.000001 | 0.001197 | [0.998515, 1.001486] | 0.9992 |

The two PE implementations have indistinguishable wall-budget loss. PS3 eager, PS3 Blackwell and both SS variants each beat both PE implementations at nominal p < 0.05. In the family of six loss comparisons to PE eager, all four improvements survive Holm correction (largest adjusted p = 0.01885). In the broader family of 15 Muon loss comparisons, the PS3 Blackwell vs PE results have adjusted p = 0.0848. SS3_54 vs PS3 eager is nominally significant (p = 0.0199, Holm p = 0.1395 across 15). Neither SS method separates from PS3 Blackwell in wall loss or throughput. The direct Blackwell vs eager PS3 wall comparison also resolves neither outcome. These are short-run, same-corpus, per-seed t-inferences. A CI spanning zero does not establish equal losses.

### Native layout · fixed tokens, three paired seeds

| A | B | Loss Δ mean | SD | 95% CI | p | Speed ratio mean | SD | 95% CI | p |
|---|---|---:|---:|---|---:|---:|---:|---|---:|
| ps3_blackwell | pe_eager | 0.003850 | 0.005134 | [-0.008904, 0.016603] | 0.3236 | 1.031725 | 0.000655 | [1.030099, 1.033352] | 0.0001 |

### Layout ablation · same three seeds and fixed-token budget

Timings below are means across seeds of the per-run median milliseconds per step (319 steps timed per run, first step omitted). Speed uses every accepted step. Loss uses the same 1,048,576 test-token targets in each trial. `research/polar_opt/harness_tier2_real.py:433–445,493–509,622–631`; FP32 raw rows in `results/cycle2_tier2_real.json:1034`; all native rows in `results/cycle2_tier2_real_native.json:231`.

| Optimizer | Update layout | Polar ortho ms | Layout cast ms | Foreach update ms | Optimizer step ms | Tokens/s | Test loss |
|---|---|---:|---:|---:|---:|---:|---:|
| pe_eager | FP32 | 10.091168 | 0.184736 | 0.753536 | 13.147136 | 138563.228201 | 5.411616 |
| pe_eager | native | 10.087168 | 0.003605 | 0.705707 | 12.920149 | 138653.423415 | 5.405374 |
| ps3_blackwell | FP32 | 3.077760 | 0.002176 | 0.729163 | 5.922816 | 142989.796140 | 5.410817 |
| ps3_blackwell | native | 3.076181 | 0.002176 | 0.727168 | 5.927253 | 143052.225861 | 5.409223 |

With matched FP32 inputs to the parameter add, PE eager minus PS3 Blackwell spends 7.0134 ms more in orthogonalization, 0.1826 ms more in layout preparation and 0.0244 ms more in the foreach segment. The optimizer-step gap is 7.2243 ms. Segment medians need not add to the median total. On PE eager, FP32 preparation adds 0.1811 ms relative to native and the foreach segment also grows by 0.0478 ms. The native mixed-dtype add took less time in these measurements. Blackwell's cast segment is 0.002176 ms in both layouts. The Blackwell/PE speed ratio is 1.031946 [1.031475, 1.032418] for the same three FP32 pairs and 1.031725 [1.030099, 1.033352] for native. Their paired difference is +0.000221 [−0.001266, +0.001708], p = 0.588. The FP32-minus-native test-loss CIs are [−0.01528, +0.02776] for PE and [−0.00888, +0.01207] for Blackwell.

The common path explicitly prepares a contiguous FP32 update before unbinding it for `_foreach_add_`. The foreach timing includes weight decay and the add. A CUDA-specific test checks matching strides and dtype and counts fallback `aten::add_` calls on a tiny model. The production JSONs contain segment timings. They omit operator traces. They support removal of the dtype mismatch and persistence of the speed lead. Per-shape dispatch costs and the effects of internal kernel precision remain unresolved. All native runs followed the main sweep, so very small cross-layout timing shifts can include session/order effects. `research/polar_opt/harness_tier2_real.py:265–290`, `tests/test_tier2_real.py:138–153`, `research/polar_opt/baselines.py:167–169`, `research/polar_opt/tier2_candidates.py:22–31`, `results/cycle2_tier2_real.json:17881`, `results/cycle2_tier2_real_native.json:117`

## Fixed-wall integrity and telemetry

For all 35 rows, `train_seconds ≤ 75 < train_seconds + discarded_step_seconds`. Accepted compute ranged from 74.780928 to 74.960671 s. The rejected step took 0.224374–0.236562 s. Every reported overshoot was 0.0. I independently recomputed tokens as `steps × 32,768` and tokens/s as `tokens / train_seconds`. The minimum step count was 317.

| Optimizer | Accepted steps, seeds 1–5 | Mean test loss | Mean tokens/s |
|---|---|---:|---:|
| pe_eager | 317, 317, 317, 317, 317 | 5.415844 | 138,674 |
| ps3_blackwell | 327, 327, 327, 327, 320 | 5.389683 | 142,423 |
| ps3_eager | 325, 325, 325, 325, 325 | 5.389807 | 142,077 |
| pe_compiled | 317, 317, 317, 317, 317 | 5.414997 | 138,887 |
| ss3_54 | 327, 327, 327, 327, 327 | 5.382984 | 142,960 |
| ss3_53 | 327, 327, 326, 327, 327 | 5.384404 | 142,961 |
| adamw | 333, 333, 333, 333, 333 | 6.057289 | 145,636 |

The clock encloses synchronized forward, backward and optimizer work. Data preparation, host/device staging, parameter snapshots, gradient clearing, telemetry, evaluation and the rejected step are outside the accepted sum. The start-to-end telemetry timestamps alone span 75.625–75.818 s, before evaluation. The harness restores pre-step parameters before evaluating. Its mocked-clock test also checks the restored validation and test losses against an independently trained two-step model. `research/polar_opt/harness_tier2_real.py:574–605,615–630`, `tests/test_tier2_real.py:188–197`

Wall warmup is 6.25% of 75 s, or 4.6875 s. The harness sets LR before each step using accepted elapsed time. I evaluated the committed `lr_multiplier` on CPU. At 0, 2.34375, 4.6875, 39.84375 and 75 s it returns 0.05, 0.50, 1.00, 0.55 and 0.10. The 20-step setting determines the starting floor of 1/20. The wall peak is time-based, and step boundaries can straddle it. Token runs instead use a 20-step warmup and step-based decay. The JSON train curves contain loss and step indices, with no per-step timestamps or LR values for exact schedule replay. `research/polar_opt/harness_tier2_real.py:379–407,574–585,609–613`, `tests/test_tier2_real.py:157–168`, `results/cycle2_tier2_real.json:51`

### Seed 5 · PS3 Blackwell wall run

Seed 5 accepted 320 steps at 139,924 tok/s. Seeds 1–4 each accepted 327 at 143,009–143,073 tok/s. The loss rose to 5.408977 from 5.380009–5.389422 on those four seeds. Its median step was 229.32 ms versus 228.72–228.77 ms, but its accepted mean was 234.18 ms. Polar remained at 3.0761 ms versus 3.0746–3.0751 ms. The last rejected step took 228.70 ms, within the prior four runs’ 228.57–229.31 ms. The final discard cannot account for seven lost steps.

| Telemetry point | SM / memory MHz | Temp °C | Power W | GPU utilization | Memory used MiB |
|---|---|---:|---:|---:|---:|
| Start | 2827 / 14801 | 54 | 237.00 | 99% | 11672 |
| Step 100 | 2827 / 14801 | 64 | 276.15 | 99% | 11684 |
| Step 200 | 2827 / 14801 | 65 | 284.89 | 100% | 11672 |
| Step 300 | 2827 / 14801 | 64 | 284.92 | 100% | 11666 |
| End | 2827 / 14801 | 65 | 284.68 | 99% | 11666 |

All five readings report P1 and clock-event mask `0x400`, also seen on seeds 1–4. Host load (1/5/15-minute) went from 2.955/2.969/2.837 to 3.609/3.168/2.919 on a 16-logical-CPU host. Start to the first 100-step sample took 24.755 s, versus 23.073–23.118 s previously. The next two 100-step intervals were 23.084 and 23.094 s. This locates most of the delay in the first interval. The timestamps include untimed host work. These sparse samples cannot identify CPU contention, a transient GPU event or another process as the cause. Keep seed 5. As a sensitivity check only, omitting it changes PS3 Blackwell / PS3 eager wall loss from −0.000124 (p = 0.985) to −0.006245 (p = 0.0280), and the speed ratio from 1.002438 (p = 0.609) to 1.006840 (p < 0.0001). Raw Blackwell wall rows start at `results/cycle2_tier2_real.json:3920,7053,9941,14030,17163`; seed-5 measurements at lines 17172–17228 and 17364–17399.

## Severity-ranked findings

| Severity | Finding and consequence | Evidence |
|---|---|---|
| CRITICAL | None found. The corpus verified locally, raw cell counts and pairing were complete, and the stored comparison statistics reproduced. | The five raw-array locations in the integrity table and `research/polar_opt/real_text_data.py:164–193`. |
| HIGH | None found for the scoped equal-token and training-compute claims. The data do not establish a general SS-over-PS3 winner. | `results/cycle2_tier2_real.json:1034`, direct paired intervals above. |
| MEDIUM | “75 seconds wall clock” needs the accepted-compute qualifier. The saved overshoot field is a literal zero. The independent accepted-plus-discarded check supports the evaluated checkpoint, while telemetry proves that elapsed time exceeded 75 s. | `research/polar_opt/harness_tier2_real.py:586–605,615–625`; `results/cycle2_tier2_real.json:17178–17211`. |
| MEDIUM | Seed 5 materially changes the direct fixed-wall comparison to PS3 eager. Sparse sampling and per-run segment medians prevent attribution of the early slowdown. | `results/cycle2_tier2_real.json:17172–17228,17364–17399`; `research/polar_opt/harness_tier2_real.py:493–509,610–613`. |
| MEDIUM | Five seeds, a single RTX 5080 and one shared test prefix limit inference. Most fixed-token Muon CIs cross zero. The single nominal Muon loss difference fails the 15-pair correction. The test data also cover only the first 1,048,576 targets of the 2,000,081-token test split. | `results/cycle2_tier2_real.json:51,114,1034`; `research/polar_opt/harness_tier2_real.py:433–445,640–652`. |
| LOW | LR transfer was only partially calibrated. Two implementations inherit a family LR. The seed-0 pilots also used a shorter horizon and an uncompiled model, and recorded test scores before confirmation. The held-out run itself has only one LR per optimizer. | `results/cycle2_tier2_pilot_muon.json:35,476`; `results/cycle2_tier2_real.json:51`; `research/polar_opt/harness_tier2_real.py:617–629,678–686`. |
| LOW | Provenance is recorded per output file. The main invocation used resume. The hash excludes code revision, extra factory binding and runtime environment. The Git dirty flag excludes untracked files. All hashes and fingerprints match, but a previously appended row cannot independently attest its producing environment. | `results/cycle2_tier2_real.json:3,51`; `research/polar_opt/harness_tier2_real.py:475–484,798–817,920–934`; resume guard test at `tests/test_tier2_real.py:332–351`. |
| LOW | Corpus separation covers exact documents. The verifier checks ID-list integrity and disjointness. Near duplicates are outside that guarantee. The “leakage” test appends a hash without updating counts, so its exception can fire on the count check before it exercises cross-split intersection. The independent intersection and routing checks passed on the local corpus. | `research/polar_opt/real_text_data.py:113–118,181–192`; `tests/test_tier2_real.py:48–79`. |
