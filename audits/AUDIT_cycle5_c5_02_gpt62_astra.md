# C5-02 audit of per-seed iso-loss PTQ and Muon mitigations

**Verdict: FAIL. Hold the merge of `da148c296a4b8750fa9e67c0145e4f0789ecba54` into `main`.** The measurements support greater PTQ loss for Muon in each of the six primary cells of this five-seed experiment. The canonical promotion check accepts contradictory calibration, evaluation and mitigation records. Two published descriptions also conflict with the retained evidence. The hold applies to this revision. The analysis below identifies the measurements that can be carried forward.

GPT-6.2 Astra reviewed the three-file diff from `a8bd366ab482e414063945822cba886702cbd670` on the bazzite RTX 5080. The initial implementation was `05ea4ae37f1f564e1f804ac9e0c8c225e553a920`. The amended producer was `0355d16f258133b5fb7aec8e90fd223f781595e9`. References to the **driver** and **result** below give line numbers at the frozen target. [Driver](../research/quant_ptq/bench_c5_02_isoloss_ptq_mitigation.py) · [Result](../results/cycle5_c5_02_isoloss_ptq_mitigation.json) · [Tests](../tests/test_c5_02_isoloss_ptq_mitigation.py) · [Backlog brief](../state/BACKLOG.md#L55)

## Findings and merge conditions

| Severity | Finding and evidence | Required correction or limit |
|---|---|---|
| **Blocking B1** | The canonical validator accepts a different seeded calibration draw, a one-window baseline claiming 1,999,872 test targets, missing butterfly fit data, impossible static scales and contradictory training recipes. These claims can each be checked using data already in the JSON. The genuine artifact passes the corresponding independent checks. Driver 1550–1576, 1635–1642, 1663–1671, 1685–1695, 1711–1715, 1741–1747, 1784–1789. | Recompute the draw and coverage from the seed, split lengths and configuration. Check cooldown metadata against the protocol and validate the fit records, damping and scale domains. Regression-test the malformed records in the table below. |
| **Blocking B2** | Caveat 7 says the runs used less than one epoch. Two saved AdamW checkpoints consumed 1.190 and 1.109 training-set token equivalents. Seed 4 also trained past one equivalent before selecting its earlier checkpoint. Driver 461–469, 529–535, 1374–1378. Result 822, 1425, 1998, 87157. | Generate the statement from the selected and terminal step counts. Report draws with replacement, token equivalents and, when comparing compute, all steps actually run. |
| **Blocking B3** | The driver says the process first opened `test.bin` after Stage A. Its verification pass opened the file twice before the first Stage A split view. Seed-0 shakedowns also scored the same test split before the amendment. The aborted five-seed attempt has no Stage B scores. Driver 8–10, 126–140, 319–343, 1363–1370, 1894, 1917–1925, 1936–1940. | Correct the split-access and pilot chronology wherever they appear. Describe the test split as held out from the canonical selection rules. A claim of a previously unseen study-level holdout would require a fresh evaluation set. |
| **Hardening H1** | Some stage times, resume markers, scalar types, data descriptors, strings and nested keys lack checks. An explicit noncanonical validation request relaxes checks even when the payload marks itself canonical. Driver 1475–1540, 1611–1632, 1680–1701, 1820–1847. | Distinguish canonical certification at the call site, validate duplicated fields and types, and give noncanonical checks an unambiguous result. Details and controls appear below. |
| **Hardening H2** | The 6,000 matrix decode comparisons in the headline include 600 inherited from zero-alpha twins. Likewise, 20 of 200 hardware rows inherit a profiler count. Driver 1025–1033, 1245–1256. | Report unique executions alongside total rows. Each inherited value in the committed file equals its source. |
| **Hardening H3** | Resume checks checkpoint file presence, then can accept an existing Stage B unit by its recorded digest without reopening the checkpoint. The canonical run recorded zero resumed units. Driver 677–680, 1922–1925, 1944–1951. | Verify retained tensor digests before certifying a resumed run, including when the Stage B unit is cached. Test a resumed run after a checkpoint becomes unreadable. |
| **Caveat C1** | Validation stopping, extra tokens, the floor-LR phase and a fresh restart couple the comparison to the recipe. The pilot's “equal val NLL” comparison actually spans 5.235–5.414. Driver 126–140, 499–535, 1357–1370. | Limit causal and optimizer-wide claims. Describe the pilot as a sensitivity check at different losses. |
| **Caveat C2** | QAT trains on identity NVFP4 RTN. AWQ chooses alpha with weight-only validation. A live seed-1 sweep found that the Muon hardware winner changes from alpha 0.25 to 0.5. Driver 368–386, 740–750, 1001–1040. | Distinguish each tested intervention and deployment mode. A general mitigation failure or optimal hardware-alpha claim requires a different experiment. |
| **Caveat C3** | Five paired seeds share one corpus, one calibration draw and one set of test documents. The smallest two-sided sign-flip p-value is 0.0625. Driver 1060–1090, 1284–1290, 1378–1381. | Report the paired-t model, correction family and effect sizes. Restrict inference to this model size and training regime. |

## Provenance and retained checkpoints

The result file SHA-256 is `4944c2832056524f994a2d4931950983244ac20e547cc9e281d5e7e0defeddb8`. I hashed the 16 named Git blobs at the producer commit in module order. They give `424055b21b6f894ef850151501425839e5731e9919ac7399138d75788f7dc85b`. Independent reconstruction gave unit hash `37922272335f8cda` and run hash `519f50bec7c85095`. The original log and JSON agree on a clean producer, five new Stage A units, twenty new Stage B units and zero reused units. The result commit adds only the JSON to the producer tree.

| Shard | Raw tokens | File SHA-256, equal to manifest and pin |
|---|---:|---|
| Train | 26,004,518 | `015ab83ceb742555c2398963a0bd1f3c5502dc52cf70a22bf173f1571326ca3a` |
| Validation | 2,002,330 | `371b64a537f7b1e306ea174a868b05932a254a98b7fbdbf4146648d93180b0b1` |
| Test | 2,000,081 | `4eb46f1424856849d2973923dbf3c886d37eb743cf22053fc7e1eea1f41ab92a` |

All 20 retained files loaded with `weights_only=True`. The JSON pins a **tensor-content digest**. I reconstructed it on CPU over sorted names, NUL separators, shapes and contiguous FP32 tensor bytes. Each digest matched the selected Stage A value and both Stage B values. I also hashed each serialized file independently. The result contains no reference archive-byte hashes. The appendix records both complete digests. [Digest definition](../research/quant_ptq/bench_nvfp4_e2e.py#L404)

## Live GPU replay and tests

The targeted suite passed **43/43** on the RTX 5080. With `CUDA_VISIBLE_DEVICES=""` it passed **41** and skipped the two native-hardware tests at test lines 674 and 722. Both used the specified project virtual environment, `PYTHONPATH=.`, an isolated temporary directory and disabled pytest/bytecode caches in the worktree. GPU capability was (12,0), driver 615.78.08, PyTorch 2.14.1+cu130 and CUDA runtime 13.0.

In a fresh process I loaded the eight specified saved checkpoints, recreated all 128 train calibration windows and ran the unmodified Stage B evaluator with the committed protocol and numerics. Each row scored all 2,001,920 validation and 1,999,872 test targets in full windows, or inherited the corresponding identical zero-alpha row. For each checkpoint I compared the returned baseline fields, every grid field, alpha tables, rotation fits, per-layer arrays and summaries, method winners, hardware metadata, and before/after tensor digests directly to its committed record. Execution time was excluded. The evaluator uses BF16 forwards for calibration, FP64 Gram accumulation and Cholesky factors, then FP32 error feedback for GPTQ. RTN needs no Hessian to round weights. Native rows pack NVFP4, run the dispatcher-counted FP4 GEMM, then return BF16 outputs. [Hessian collection](../research/quant_ptq/micro_rotated_ptq.py#L625) · [GPTQ precision](../research/quant_ptq/micro_rotated_ptq.py#L470) · [Native scaled GEMM](../research/quant_ptq/nvfp4_inference_blackwell.py#L391)

| Seed | Arm | BF16 val / test NLL | Grid rows | Exact non-timing leaf values | Hardware rows | Decode matrix entries | Hardware minus FP32 emulation NLL, val / test |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | muon_ss3_54 | 5.355855915 / 5.392342555 | 37 | 6,316 | 12 | 360 | +0.000092121 / -0.000001358 |
| 1 | adamw | 5.354112845 / 5.387795102 | 37 | 6,351 | 12 | 360 | +0.000045921 / +0.000079103 |
| 1 | muon_adamw_cooldown | 5.382299902 / 5.419356255 | 25 | 4,444 | 8 | 240 | -0.000053470 / -0.000020513 |
| 1 | muon_qat_cooldown | 5.362176498 / 5.398812501 | 25 | 4,444 | 8 | 240 | +0.000044486 / -0.000080600 |
| 2 | muon_ss3_54 | 5.347835993 / 5.382843215 | 37 | 6,316 | 12 | 360 | -0.000034997 / +0.000001065 |
| 2 | adamw | 5.345434258 / 5.376100182 | 37 | 6,386 | 12 | 360 | -0.000000021 / -0.000060390 |
| 2 | muon_adamw_cooldown | 5.373594881 / 5.409464955 | 25 | 4,444 | 8 | 240 | -0.000069212 / -0.000040987 |
| 2 | muon_qat_cooldown | 5.354714644 / 5.389665215 | 25 | 4,444 | 8 | 240 | +0.000003924 / -0.000044277 |

All 43,145 non-timing values matched exactly across 248 grid rows. The rows cover all four formats, plain RTN, identity GPTQ, Hadamard GPTQ, fitted-butterfly GPTQ and both AWQ rotations wherever the protocol enables them. All 480 layer records matched, including stable rank and signal alignment, as did the per-method error alignment and relative output error arrays. GPU hardware rows reported 60 FP4 GEMMs per forward. All eight BF16 and eight emulation controls reported zero. For all 2,400 reported matrix decode entries, the unpacked hardware weights and the corresponding fake-quantized weights were equal when viewed as int32 bit patterns. Zero-alpha twins account for 180 of these entries. This leaves 2,220 fresh matrix comparisons and 74 fresh profiled hardware rows out of 80 reported. The hardware-versus-FP32 differences above use the dynamic Hadamard GPTQ control, the one method with an emulation row. The largest absolute replayed difference was 0.000092121 nats. The committed 20-checkpoint maximum was 0.000171324 against the 0.001-nat threshold. Decode equality is bitwise. The emulation comparison is an end-to-end NLL tolerance on the same quantized operands.

I also computed seed-1 Muon block-0 QKV geometry directly from the 65,536 calibration activation vectors and the NVFP4 RTN weight. An FP64 singular-value decomposition gave stable rank 148.371345944149. Direct projected signal and error energies gave signal alignment 1.180379451824, error alignment 1.012854024048, and relative output error 0.007765682592. Absolute differences from the committed values were 1.14×10⁻¹³, 4.45×10⁻¹⁶, zero and 1.74×10⁻¹⁸. This calculation used the activations directly, independently of the Gram-based geometry functions.

This replay independently executed the published evaluation code and did not retrain the checkpoints. Seeds 3–5 received file, arithmetic and historical log verification. Their full Stage B grid was outside the live replay sample.

## Protocol, split isolation and token exposure

The canonical Stage A evaluator closes over validation tokens. Its entry point requires exactly train and validation mappings. AdamW measures one trajectory at steps 400, 416, … and stops at the first nonpositive AdamW-minus-Muon validation gap, subject to the cap. It saves the measured state with the smallest absolute gap, breaking ties by earlier step. Seeds 3 and 4 select the preceding step. All five Stage B validation baselines confirm the selected weights, as expected from reuse. Driver 513–535, 582–600, 615–627, 669–675.

| Seed | Saved / last AdamW step | Validation looks | Floor steps in saved weights | Saved tokens | Token equivalents | Unique training input positions | Val gap | Test gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 720 / 720 | 21 | 256 | 23,592,960 | 0.907 | 59.57% | +0.001743 | +0.004547 |
| 2 | 944 / 944 | 35 | 480 | 30,932,992 | 1.190 | 69.59% | +0.002402 | +0.006743 |
| 3 | 880 / 896 | 32 | 416 | 28,835,840 | 1.109 | 66.86% | −0.001325 | +0.000952 |
| 4 | 784 / 800 | 26 | 320 | 25,690,112 | 0.988 | 62.64% | −0.000261 | +0.000958 |
| 5 | 672 / 672 | 18 | 208 | 22,020,096 | 0.847 | 57.10% | +0.003328 | +0.005116 |

Gaps are BF16 NLL in nats, Muon minus AdamW. Each Muon run saw 10,485,760 tokens, or 0.403 equivalents. Its unique coverage ranged from 33.04% to 33.35%. Each step draws 64 independent starts with replacement from 0 through N−513, inclusive. Each 513-token slice supplies 512 shifted targets. There is no sequential cursor or end-of-shard wrap. I regenerated all starts for each seed and measured the union of the 512-token input intervals. Higher exposure both repeats and adds positions. It can change generalization and geometry. These five pairs have no fixed-step or fixed-exposure control that separates this effect. [Batch stream](../research/polar_opt/harness_tier2_real.py#L151)

Seeds 2 and 3, the two selected above one token equivalent, have AdamW test-minus-validation BF16 gaps averaging 0.03087 nats, versus 0.03222 for seeds 1, 4 and 5. Their fitted-butterfly dynamic W4A4 test losses average 0.00809 versus 0.00659. For INT2 g32 the ordering reverses, 0.01708 versus 0.01881. These are descriptive groups of two and three selected trajectories. They give no uniform direction for an exposure effect.

The five validation gaps average +0.001177 (95% CI −0.001209 to +0.003564). The test gaps average +0.003663 (CI +0.000434 to +0.006892, paired-t p=0.03452, sign-flip p=0.0625). Every test gap is within the declared ±0.02 tolerance. Test minus validation gap is positive on all five seeds and averages +0.002486. Selecting the closest validation measurement conditions the validation gap on the target. The training seed, stopping time and extra floor-LR updates all affect the selected weights. Holding out test from this code removes direct test optimization. It does not identify how much of the small positive test gap comes from stopping, exposure or the two fixed document samples. The direction alone cannot establish optional-stopping bias. Significance of an exact-zero null is a separate question from meeting the 0.02 tolerance.

**Amendment chronology.** The stopped `05ea4ae` log records seed 1 at 640 steps (+0.0029 Muon-minus-AdamW, line 22). Seed 2 started the measured grid +0.5828 above Muon and ended +0.0925 above at step 1280 (lines 27 and 82–83). That is the end of the log. The diff to `0355d16` changes only the executable cap, its documentation and corresponding test expectations. The new cap doubled the measured grid from 56 to 136 possible looks. The new canonical log starts with zero cached units, records all five Stage A selections by line 158 and starts test scoring at line 159. Every selected and terminal step in this fresh run is below the original cap. Raising the cap did not allow additional steps on these realized trajectories. Restarting training did produce different trajectories. Report the stopped attempt as part of the design history, including the unmatched seed.

**Test exposure.** A Python file-open audit during construction of `Shards(verify=True)` observed two read opens per binary while `Shards.opened` was still empty. The verifier checks file hashes, token range and document boundaries, and verifies exact document-hash disjointness. These checks do not score any model. The val-only pilot probe uses only train calibration and validation NLL. Separately, `timing_seed0.log` lines 10–14 and `timing_seed0b.log` line 33 onward contain test scores. The latter JSON records four checkpoints and 124 grid rows on the same shards, ending Nov 26 at 23:26 MST, before the initial code commit at 00:02 and the amendment at 00:28 on Nov 27. Its seed is 0. This rules out a study-wide claim of unseen test documents. The logs establish that scores were available. They cannot establish whether a person used them to make a design decision. [Manifest verifier](../research/polar_opt/real_text_data.py#L164)

Hessian and per-channel moments come from the same 128 seeded, distinct train windows (65,536 input tokens). I reproduced all starts. Static scales use those train windows with the BF16 model. Each candidate alpha receives a weight-only validation score. The winner uses that score, including the zero-alpha case. `best_methods` also computes a retrospective test winner, but pooled and arm-specific family construction reads only `best_by_val`. I independently minimized the validation sums for every selected method in both families. I found no test NLL in training, stopping, Hessian or scale calibration, alpha selection or the choice of a primary contrast. Driver 685–708, 792–810, 950–962, 1007–1038, 1183–1220. [Calibration sampling](../research/quant_ptq/micro_rotated_ptq.py#L583) · [Static scales](../research/quant_ptq/bench_nvfp4_e2e.py#L363)

**Recipe sensitivity.** The backlog requests per-seed LR/step matching and spectral/QAT/AWQ mitigation. This implementation varies only the AdamW stopping step, fixes LR at 0.001 and uses an optimizer-switch cooldown as its indirect intervention on geometry. It measures stable rank and alignment. It supplies no experiment with an explicit spectral penalty. Seed-0 independent runs at fixed LR and 640 steps differed by 0.06295 validation nats, and at 472 steps by 0.07488. The retained floor-LR probe used identity GPTQ. At 848 steps it reported validation 5.4144 and quantization losses 0.001046 NVFP4 and 0.009581 INT2 g16. Fresh 544- and 608-step cosine runs reached validation 5.2975 and 5.2352 with corresponding losses 0.001003/0.010648 and 0.000816/0.010579. These are similar magnitudes at different losses. They cannot establish a floor-phase null effect at equal loss. Pilot evidence comes from `pilot_adamw.out`, `pilot_adamw2.out`, `pilot_hold_480.out` and `probe_hold.out` lines 1–7.

C4-01b fixed AdamW at 544 steps and LR 0.0015 after a seed-0 search. Only one of its three evaluation seeds matched. In C5-02 the primary INT2 g32 Muon loss is 0.05396 against 0.05372 previously. The AdamW loss rises from 0.00450 to 0.01812. For dynamic NVFP4 the corresponding changes are 0.01102 to 0.01127 and 0.00506 to 0.00719. The higher denominator explains the smaller *measured ratio*. Seeds, stopping, duration and LR changed together. Only a controlled comparison can attribute the change to LR. [C4-01b matched recipe and primary cells](AUDIT_cycle4_c4_01b_gpt62_astra.md#L53)

## Mitigation implementation and fairness

**AdamW cooldown.** The hook saves state just before zero-indexed step 256. Thus 256 Muon updates have completed. The snapshot includes cloned weights, cloned Muon momentum and a deep copy of the auxiliary AdamW state. Restoration reconstructs the optimizer, restores both states, then hands the same matrix parameters to a fresh AdamW while retaining the restored auxiliary optimizer. The stream advances exactly 256 draws before the next training batch. The common multiplier is 0.208456 at step 255, 0.205280 at 256 and 0.100027 at 319. The matrix base LR changes from 0.02 to 0.001 at the switch, and the update algorithm changes. The auxiliary LR schedule continues. The unit tests check uninterrupted continuation, stream offset, parameter identity and snapshot preservation. Driver 389–455, 477–495, 540–556. [Optimizer updates](../research/polar_opt/harness_tier2_real.py#L233) · [Branch tests](../tests/test_c5_02_isoloss_ptq_mitigation.py#L309)

The final ten training losses average 5.32218 for the AdamW cooldown versus 5.29346 for Muon. The BF16 test penalty is +0.026409 nats (95% CI +0.025902 to +0.026916). The intervention reduces primary PTQ loss in all six cells by 0.000207–0.005780 nats. It pays more baseline quality than it recovers under quantization. All six final NLL differences favor unchanged Muon, by 0.020629–0.026201 nats. A cold start on the hidden matrices during the final 64 low-LR steps is a plausible mechanism. Matrix moments, LR and optimizer change together. There is no separate experiment that identifies which accounts for the penalty. I found no skipped batch or reset of the auxiliary moments.

**QAT cooldown.** At step 272 the code installs a wrapper around each existing linear. It reads the original FP32 master weight. Both weight and input get identity-coordinate NVFP4 group-16 RTN. The detached quantization residual gives each fake-quantized operand the straight-through gradient. Quantization runs outside autocast, then the enclosing BF16 training autocast applies to the linear operation. The wrapper restores the original modules in a `finally` block before checkpoint evaluation and saving. Muon retains the parameter objects from before wrapping. An independent one-step tiny-model check found nonzero gradients, updates on all 10 original matrix objects and the same 10 restored modules. I repeated the binding check on the RTX 5080 using the saved seed-1 QAT model, a fresh SS3_54 optimizer and one real training batch at the step-272 stream offset. All 60 original matrix objects received nonzero gradients, changed value, and were restored in their original modules. This checks parameter binding on the real architecture. Driver 368–386, 557–575. [Training autocast](../research/polar_opt/harness_tier2_real.py#L412) · [Module replacement](../research/quant_ptq/bench_nvfp4_e2e.py#L347)

| Five-seed mean test ΔNLL | Plain RTN | Identity GPTQ | Hadamard GPTQ | Fitted butterfly GPTQ |
|---|---:|---:|---:|---:|
| Muon, NVFP4 weight only | 0.009315 | 0.002570 | 0.002231 | 0.002137 |
| QAT, NVFP4 weight only | **0.000116** | 0.001886 | 0.002024 | 0.001966 |
| QAT, native dynamic W4A4 | **0.007785** | 0.009483 | 0.011432 | 0.010383 |
| QAT, native static W4A4 | **0.007806** | 0.009383 | 0.011444 | 0.010289 |

Validation selected RTN for all three NVFP4 modes of QAT and test confirmed each ranking. QAT trains the precise identity RTN rounding pattern. GPTQ changes rounded values through error feedback. Both rotations change the coordinate system. This objective mismatch explains why a generic PTQ improvement need not carry over. On other formats QAT still chooses fitted butterfly GPTQ. Its BF16 test penalty is +0.006463 nats (CI +0.005914 to +0.007013). Even using its own best method, QAT gives +0.002982 nats in dynamic W4A4 final NLL relative to Muon's own best, despite a −0.003481 improvement in PTQ loss. The INT2 g32 final NLL difference is +0.000452 with within-family Holm p=0.3594. Evidence for an NVFP4-specific sensitivity improvement is stronger than evidence for a better final model.

**Activation-aware scaling.** With D diagonal, the experiment quantizes the transformed weight W D in rotated coordinates. It divides input channels by D, giving the transformed Hessian D⁻¹ H D⁻¹ before rotation. Live channels use centered log mean-absolute activation raised to alpha. Zero-energy channels keep scale one. Weight-only evaluation inverts the rotation and divides the effective weight by D. The hardware wrapper multiplies the FP32 input by the reciprocal before the native layer rotates and packs it. Static calibration applies the same reciprocal and rotation to train activations. Alpha zero reuses the unscaled rows. Both baseline arms receive the same search. Driver 710–750, 890–909, 916–941, 951–956, 1002–1039. [Native forward and accepted input types](../research/quant_ptq/nvfp4_inference_blackwell.py#L487)

The canonical weight-only search selects zero in 23/40 Muon and 15/40 AdamW cases. Nonzero scales can preserve weights better while worsening activation quantization, and the reverse is also possible. Only the weight-only winner receives a hardware score in the canonical table, apart from the always-available unscaled base method. The best hardware alpha among all four candidates cannot be inferred for every seed from these rows. Where a nonzero alpha was selected for NVFP4, all 10 such candidates improved hardware validation against their own zero-alpha twin on both dynamic and static W4A4. This establishes an improvement over zero for those comparisons. Other nonzero candidates might still do better. The canonical AWQ-versus-Muon primary test changes are −0.000363 nats in dynamic W4A4 and −0.000325 in static W4A4 (within-family Holm p=0.503 and 0.485). Treat these as results for this alpha-selection rule.

**Supplemental hardware validation sweep.** I re-quantized both seed-1 baseline checkpoints at all four alphas using fitted butterfly GPTQ. I scored each candidate on the full validation split under native dynamic and static W4A4. No test NLL entered this comparison. All four controls at the published weight-only choices reproduced the committed hardware validation NLL exactly.

| Seed 1 validation NLL | Alpha 0 | Alpha 0.25 | Alpha 0.5 | Alpha 0.75 | Weight-only choice | Hardware choice |
|---|---:|---:|---:|---:|---:|---:|
| Muon dynamic | 5.366707 | 5.366491 | **5.366260** | 5.366821 | 0.25 | 0.5 |
| Muon static | 5.366716 | 5.366411 | **5.366232** | 5.367067 | 0.25 | 0.5 |
| AdamW dynamic | **5.360404** | 5.360638 | 5.360633 | 5.360579 | 0 | 0 |
| AdamW static | **5.360355** | 5.360697 | 5.360556 | 5.360649 | 0 | 0 |

The proxy cost Muon 0.000231 dynamic and 0.000178 static validation nats on this checkpoint. It cost AdamW zero on the same grid. This establishes a real, small mode-selection handicap in one comparison. It cannot establish the hardware-tuned test effect across five seeds.

**Geometry.** Across five seeds the all-layer mean stable ranks for Muon, AdamW, AdamW cooldown and QAT are 172.49, 38.08, 172.02 and 172.46. The corresponding mean signal alignments are 1.153, 4.146, 1.165 and 1.159. On NVFP4 with the fitted butterfly, Muon has mean error alignment 0.1627 and relative output error 0.002159, against 0.1126 and 0.000577 for AdamW. Relative weight error is 0.01423 versus 0.01639. Thus unweighted weight error alone does not explain the sensitivity ordering. The identity relating output error, weight error, error alignment and signal alignment holds per layer. Its factors must be combined before averaging. These are descriptive diagnostics. The experiment contains no intervention that independently varies rank or alignment. Driver 752–784, 852–865, 914–915, 1258–1272.

## Independent statistics

I joined the 620 grid rows by seed, arm, format, mode and method. For each cell I reconstructed quantization loss from row NLL minus that checkpoint's BF16 NLL. I selected methods by summing validation NLLs and derived geometry contrasts from the raw per-layer entries. Neither calculation called the producer's analysis, statistics or summary functions. I tested both splits and endpoints throughout, including the secondary cells and the 48-cell corrections across contrasts. There are 173 secondary cells across eight contrasts, and six cells per contrast in each selected family. Driver 1060–1090, 1100–1138, 1190–1244, 1304–1348.

| Statistic | Independent coverage | Comparison |
|---|---:|---|
| Paired df=4 statistics | 692 all-cell + 192 primary + 192 own-best + 10 BF16 + 2 iso-loss + 40 layer = **1,128** | Every five-value input vector, mean, SD, SE, t, CI and p reproduced with the declared binary64 arithmetic |
| Sign flips | 1,128 × 32 sign assignments | All p-values exactly equal. All 192 headline copies also equal |
| Holm, within contrast | 692 all-cell + 384 selected-family adjustments | All exactly equal, across 96 families |
| Holm, across contrasts | 384 adjustments | All exactly equal, across eight 48-cell families |
| Derived test summaries | 192 CIs, 192 raw p, 192 sign-flip p, 192 within-family Holm and 192 across-contrast Holm | Every value reproduced |
| Independent library cross-check | SciPy 1.18.1 Student-t survival function and quantile, sample SD from Python statistics | Max absolute difference 2.85×10⁻¹⁴ for CI bounds, 7.11×10⁻¹⁵ for raw p, 1.00×10⁻¹⁵ for Holm, zero for sign flips |

The exact comparison used the published df=4 critical value and the stable closed-form tail probability. The library check used SciPy's own quantile and survival function. Their last-bit differences had no effect on a decision. Holm applies to the paired-t p-values. An exact two-sided sign-flip test cannot reach 0.05 with five nonzero pairs even before correction. It also assumes sign exchangeability under the null. All arms reuse the same validation documents for quality matching and method selection. A selected validation winner can look optimistic there. The pooled method also depends on all five seeds. These intervals condition on the selected rule and this single corpus. They omit variation from repeating method selection, sampling new documents, or drawing a new calibration set.

| Primary test cell | Muon ΔNLL | AdamW ΔNLL | Ratio | 95% CI for difference in ΔNLL | Holm p, six cells | Holm p, 48 cells | Final NLL difference |
|---|---:|---:|---:|---|---:|---:|---:|
| NVFP4 g16 weight only | .002137 | .000970 | 2.20 | [.000956, .001377] | .000312 | .003227 | +.004830 |
| NVFP4 dynamic W4A4 | .011267 | .007188 | 1.57 | [.003215, .004941] | .000312 | .005266 | +.007741 |
| NVFP4 static W4A4 | .011188 | .007081 | 1.58 | [.003298, .004915] | .000312 | .004401 | +.007770 |
| MXFP4 g32 weight only | .003675 | .002103 | 1.75 | [.001325, .001821] | .000245 | .002082 | +.005236 |
| INT2 g16 weight only | .033265 | .011572 | 2.87 | [.020001, .023385] | .0000223 | .000175 | +.025356 |
| INT2 g32 weight only | .053958 | .018116 | 2.98 | [.032615, .039069] | .0000329 | .000283 | +.039505 |

The pooled validation winner in all six cells is fitted butterfly GPTQ. Each ratio is the ratio of two five-seed means. Final quantized NLL includes the BF16 gap. For that separate endpoint, all six within-contrast adjusted p-values are below 0.05. Under the 48-cell family, only native static W4A4 and both INT2 cells pass 0.05. All six sensitivity differences pass either paired-t correction.

## Offline validator

I loaded the pinned JSON fresh, validated the unmodified control, then evaluated 66 independent, single-field data-quality cases in each mode. The controls passed in both modes. The altered copies stayed in memory. Counts exclude the repository pytest cases, which separately exercise early stopping, continuing after a crossing, parity above 0.001 nats after dependent arithmetic updates, and the documented limit of checking a self-consistent record offline. [Promotion-gate contract](../research/quant_ptq/bench_c5_02_isoloss_ptq_mitigation.py#L1793) · [Existing negative tests](../tests/test_c5_02_isoloss_ptq_mitigation.py#L801) · [Canonical parity regression](../tests/test_c5_02_isoloss_ptq_mitigation.py#L1048)

| Check family | Cases | Canonical rejected / accepted | Explicit noncanonical rejected / accepted |
|---|---:|---:|---:|
| Git, command, code/unit/run digests, pinned-verification flag, metadata NUL | 9 | 9 / 0 | 6 / 3 |
| Stage A split, stopping, selection, snapshot, epoch count | 10 | 10 / 0 | 10 / 0 |
| Stage B checkpoint, baseline, row arithmetic, geometry, AWQ, calibration split | 8 | 8 / 0 | 8 / 0 |
| Hardware GEMMs, emulation GEMMs, BF16 control, decode equality/count, emulation difference, static source | 7 | 7 / 0 | 7 / 0 |
| CI, raw t p, both selected-family Holm fields, sign flips, derived grid count, caveat text | 7 | 7 / 0 | 7 / 0 |
| NaN, +Inf, −Inf, extra top/row keys, boolean protocol seed, integer match flag | 7 | 7 / 0 | 7 / 0 |
| Additional internal contradictions or type/schema errors below | 18 | 0 / 18 | 0 / 18 |
| **Total** | **66** | **48 / 18** | **45 / 21** |

Each example below describes a separate minimal regression fixture against the unmodified artifact. “First checkpoint” means seed 1 Muon. In all cases the current validator accepted the fixture. The examples distinguish scientific contradictions from secondary metadata errors.

| Field or invariant | Observed accepted fixture | Correct validation |
|---|---|---|
| Calibration sample (driver 1663–1671) | First start 0 with the original calibration seed, in place of 346112 | Compare the ordered starts to the sample derived from seed, train length, sequence count and sequence length |
| Full test coverage (1685–1695), three cases | First checkpoint test baseline claims one window, or span [0,1], or two available split tokens. Each retains 1,999,872 scored targets | Reconstruct 3,906 full 512-token windows, span [0,1999873] and the 2,000,081-token source length. Apply the same rule to validation |
| Calibration schema (1663–1671) | Undeclared `units` member in the calibration record | Check the defined keys for this schema version |
| Fitted rotations (1679–1684, 1711–1715) | Empty `butterfly_fits` alongside adaptive GPTQ results | Require per-kind fit records for each fitted candidate and check step counts, layer counts and error reduction |
| Quantizer metadata (1740–1745, 1787–1789), two cases | Negative, reversed static-scale range on the first static row. Negative GPTQ damping on the first RTN row | Require finite, positive, ordered static scales. RTN has null damping. Validate GPTQ damping against the configured fallback |
| Cooldown recipes (1550–1572, 1635–1642), two cases | Seed 1 AdamW cooldown claims matrix LR 0.75. Seed 1 QAT cooldown claims no fake quantization | Derive optimizer, matrix LR and QAT mode from the locked branch recipe |
| Training loss (1570–1576) | The seed 1 Muon last loss is −1 despite a positive step-320 loss in its curve | Check nonnegative finite losses and equality when the final step is present in the curve |
| Durations and resume counts (1500–1525, 1617–1621, 1679–1684, 1741–1745), three cases | A negative Stage B duration, a negative first-row duration, or a resumed first checkpoint in a run with zero loaded units | Check component domains, equality of duplicated stage times, and the sum of per-unit resume flags |
| Data description (1838–1846), two cases | Unrecognized tokenizer encoding or embedded NUL in the data directory | Check canonical data identity and string validity for the whole data record |
| JSON scalar types (1624–1627, 1744–1747), two cases | Integer zero in a row's boolean scaling flag, or boolean true in the first Muon row's seed | Require JSON booleans for flags and integers for seed IDs |

Passing a noncanonical mode explicitly also accepted a dirty-tree record, an unrelated Python entry point and an unverified-shards flag on a payload still labeled canonical. These are expected relaxations for a genuine exploratory run. A caller seeking promotion must enforce canonical mode. The canonical flag and requested validation mode should agree when reporting certification. The unmodified canonical artifact has the expected schema for the tested fields, the exact seeded draw, correct coverage for every evaluated split, valid ranges, and all 510 required per-kind fit summaries. Every one of the 58 copied zero-alpha rows also agrees with its full source row apart from method/scaling labels. These checks can run offline.

The producer records a content hash for the selected weights. On resume, checking mere existence cannot verify those bytes if both a Stage A unit and its Stage B results are reused. A useful regression fixture retains both finished units, makes the retained checkpoint unreadable in a test directory, and requires failure or regeneration before a successful resumed result. This finding follows from the cache path. The audited canonical metadata records 25 newly computed units.

## Promotion decision

Keep the code/result merge and the C5-02 backlog signoff on hold until B1–B3 are corrected and independently rechecked. Preserve the original producer revision, artifact digest and raw measurement history when publishing a correction. If only reporting or validation changes, identify the code that postprocessed the original measurements separately from the code that produced them. A fresh training run would produce new weights and constitutes new evidence.

Carry forward the five within-tolerance test pairs, the larger Muon PTQ loss in all six primary cells for this recipe, the NVFP4 RTN benefit on the QAT arm, and the measured BF16 cost of each cooldown. The evidence does not warrant changing the default optimizer, claiming an optimizer-wide ratio, attributing the test gap to optional stopping, declaring W4A4-tuned AWQ ineffective, claiming an explicit spectral intervention, or describing this corpus as a previously unused test set. Keep the five-seed t assumptions and the hardware scope attached to the measured results.

## Appendix: retained file fingerprints

The implementation worktree retained the run evidence in `artifacts/c5_02_design_pilot/` and weights in `artifacts/c5_02_checkpoints/`. The historical canonical run log agrees at its printed precision with all five match summaries, all 20 BF16 validation/test pairs and all 562 independently evaluated grid rows. The other 58 rows reuse zero-alpha twins. Rounded log agreement covers seeds 3–5 as well. Full-precision live replay coverage is identified separately above.

| Retained evidence | SHA-256 |
|---|---|
| Stopped canonical log | `d877d6862b98b8d140aeb2436f0085ce18173ccada7252e0d140910e64c0d046` |
| Final canonical log | `cfde4b04bc416c04cb5d98c39f94ee59312b62c2f24bfe840acd35c065b843fe` |
| Full seed-0 shakedown JSON | `39f2d27e6661e89f7f3e7f4cef7bab7ed78441b73236e81bc92fb8c303421803` |
| Floor-LR probe output | `adf722e3ab295959d11bb6a0889d8c8e376d2c65f5a540656caecfb5ba44a2a5` |

Each archive is named `seed<seed>_<arm>.pt`. The rightmost column matches the committed JSON for all twenty records. The archive column identifies the exact files retained at audit time.

| Seed | Arm | Serialized file SHA-256 | Tensor-content SHA-256 |
|---:|---|---|---|
| 1 | muon_ss3_54 | `b35c20597182c7412684a5b6305640c189aab54f801fd92d0e3bb850142e8e79` | `2168a22db0419f4f233c822eab43409ac2c63f7fb428a9fe0a4655a6dd229292` |
| 1 | adamw | `3ecf004ef5b2b4a19ca1a04c8cc0c42fad9385a82d7714e63481efec0f6ea789` | `6290131afc34f95d5f7a283d3762f01df9e2955e539368f3fc63ec947423fe35` |
| 1 | muon_adamw_cooldown | `2c7ca42a2deae6e7346b7f8b7f19dc32d4f0fa6c3aa90c375f24f84b88fc9554` | `9e6b5ef7dfd6e00725c1ed80a6c540322ab8e0eacc1445569075c07d84ea9338` |
| 1 | muon_qat_cooldown | `fad4036564e4648170f9413e52557a5d95fe671819d0efb0f85a72339776cc81` | `bdedd4f7189559c82ac92ece294c9ceddfc48aefe0c3407fd001fd526610d377` |
| 2 | muon_ss3_54 | `d130b4afe00a229655945994054f40085520871737d2f7fce8d0fd29639e368f` | `6bccf401cffd23be1a267691b37dac693e784a51f37077742837569df0645090` |
| 2 | adamw | `4f12cc0e75ec05e2504dc710e63cc570889126cb55310ac51902b4510b4d2c1e` | `d283b650881206ae31f46c95c44e3ac130dbf0f02cec3d2caa787f4aa0d75f62` |
| 2 | muon_adamw_cooldown | `5480e366896f2235d11a2e5a8b3431d756ba017db0f26ffa553a0176a70e25d0` | `03f25f1982d412b6b73b21d1cd89826d325291eeccc2f846cc116f9a55c588f9` |
| 2 | muon_qat_cooldown | `f4ede7c0db2e41f9024d9b319850567813f8bddc545d099e5273047e17883185` | `f725b6890708d7c1f61d461774d8be3f07dc799aa711da2a0d1ff01b3e0e628c` |
| 3 | muon_ss3_54 | `8a0ce4465aea5f8f0a3698d14e059cf51ae29980726071bf5c916e912317f459` | `1b5f36c8274286a42cf64632ab3b066e0d5ed874f3c08af2d8debe731ea34eae` |
| 3 | adamw | `6a780324038dea1181d1220a55f677075d69cc0fe4f12160809bfef760f65eeb` | `81d3e9a73d9fb9c6fb3e6fb820be7b15751122b63285547d2427798a53fa0f34` |
| 3 | muon_adamw_cooldown | `219f3219774b53e629ca36c9085873c808e555b2732b32302fa15810c8f58a8b` | `83b87b296fc1211a68c55909c7cdb9c26015a7851385ed9a74dfa2f307edf25b` |
| 3 | muon_qat_cooldown | `264a6822e1f844994281ffcf6437d63f94590f3f2c7f35eff6926274299090c4` | `6cec64bb0d0e00b3c6f2830e0302c7211bf5cb13b2c9adff62c658f28cc89567` |
| 4 | muon_ss3_54 | `fd82b77f701a95beebf6269b8ebf8be39f2c7056a72b7f2f05050f3158e984ab` | `3ea853f9b1c046fdb90101bc527447bc2805466ca7ed55c2396bbaf181f775b3` |
| 4 | adamw | `4e32a57c89c23b6f04704deb76e98c5c3313c25a93fae71ac4260b0453e89116` | `602766b1c058826ca1bd83d778f19e3327f6a3af8be0fc966b3fd81600cb1067` |
| 4 | muon_adamw_cooldown | `d6ec5d14caf8653ee27dabb3cb9e1c097b1b1548212cd03a2f98b45a8786a02d` | `8cf33528caa91bf06d97058d418c7f850a29e48590b4542228fd7b8db83c0b67` |
| 4 | muon_qat_cooldown | `0438f80b3e1c67f6c5492aef5e24a499388abf1ed147098e5f17fe98847a15b5` | `9a71b785d58f7368a433e38dcc5992f22b2d743462438b5af478684de6112876` |
| 5 | muon_ss3_54 | `e05c3bfc33743d6f8b99aeb9b382a1e1756fe16d476688d321800a26c47a9ab7` | `d47d2f1efa30b7b1de912e0e1a6ba0bec39c64e1f5b4a6d18960a72e634772f9` |
| 5 | adamw | `b1e7e655c0103f2f9432a5318d967262a5ed2cab7b95c77ea1050250153a24bc` | `609133172d7d1c6ee9bf4cca7ac0fa536ba2b1c022189688f840528226ff7184` |
| 5 | muon_adamw_cooldown | `c61bedf038fb7eefde1558d147ea25c50b9442bed24df79671ddacfacf312487` | `3e86c6f288d7c5186b4bc3c02672873cb85afbcdd21bb31614e86626402f166a` |
| 5 | muon_qat_cooldown | `ad292a5c13edd1eaac790a02f150284e1ad5889297983acb8375b56a94602862` | `f688f0efa98c76d1ed99d7540297c1538b364177bb6e447a501ff60eae86d30d` |
