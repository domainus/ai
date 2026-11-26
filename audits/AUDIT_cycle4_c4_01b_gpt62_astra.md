# C4-01b independent audit · RTX 5080

**PASS_WITH_CAVEATS. Merge the benchmark and its result as a seed-0-calibrated experiment. Hold promotion of an optimizer, a three-seed iso-loss claim, and an optimizer-wide causal explanation.** All six primary cells show greater PTQ loss for Muon relative to its own BF16 baseline. The absolute quantized test NLL comparison is inconclusive in every primary cell. The original seed-2 pair supports the sensitivity result at closely matched quality. Independent retraining gives a second observation of the training recipe, with different weights.

## Findings in severity order

| Severity | Finding | Disposition |
|---|---|---|
| HIGH for promotion | Only one held-out seed meets the 0.02-nat validation target. Muon and AdamW also see different numbers of tokens. Pairing removes the BF16 offset from the sensitivity endpoint but cannot remove interactions between baseline quality, training budget, and quantization. | Retain the locked three pairs. Before promoting a quality-matched result, predeclare a validation-only matching procedure for each seed, independent test confirmation, and more seeds. |
| HIGH for exact replay | Both seed-2 retrains produced new checkpoint SHA-256s. The experiment records the six digests but saves no checkpoint tensors. Within-run restoration and across-run reproducibility are separate claims. | Accept the artifact arithmetic and direct hardware evidence. For exact mechanistic attribution, retain the original weights or replay training with verified deterministic numerics and saved weights. |
| MEDIUM | The resume key misses executable dependencies of the actual Muon update, including the SS3_54 wrapper and kernel. A changed implementation can reuse an old unit. | Invalidate the log across dependency changes, or include the transitive code and relevant runtime revisions in its key. The submitted run loaded zero units. |
| MEDIUM | The Stage A lock guards subscripting only. Normal dictionary reads can expose test tokens without incrementing the ledger. The present Stage A call path consumes only train and validation. | Supply an explicit train/validation view if the lock is meant to enforce isolation for future callers. |
| MEDIUM | Canonical validation checks many numerical relationships but accepts inconsistent provenance, pilot coverage, calibration locations, and layer identity. | Independently verify provenance for this result. Before treating the validator as a promotion gate, bind these fields to the actual protocol and inputs. |
| LOW | Three paired seeds give two degrees of freedom. The reported p-values depend on a normal model for paired differences. One calibration draw and one model size further limit transfer. | Report seed-level effects and separate inferential families. A two-sided sign-flip test with three exchangeable pairs has minimum attainable p = 0.25. |

No critical defect was identified in the recorded arithmetic or the FP4 execution path.

## 1. Provenance, split use, and guards

The audit branch began at `3f954e411666801bbfa879cf079cd063fe6e701c`. Its sole parent is `0659435bb221fb7c4f19394e52e616daf240306f`. The diff adds only `results/cycle4_c4_01b_isoloss_ptq.json`, 15,635 lines. The result SHA-256 is `c5a55219075bda530ecc58be3f9b8c8f5ad84afc881e3202b643d50e2e602ae3`.

| Input | Independently hashed bytes, matching source pin, artifact pin, and recorded split |
|---|---|
| train.bin | `015ab83ceb742555c2398963a0bd1f3c5502dc52cf70a22bf173f1571326ca3a` |
| val.bin | `371b64a537f7b1e306ea174a868b05932a254a98b7fbdbf4146648d93180b0b1` |
| test.bin | `4eb46f1424856849d2973923dbf3c886d37eb743cf22053fc7e1eea1f41ab92a` |

The JSON parser rejected injected NaN, both infinities and duplicate keys. The full payload passed strict parsing and a lossless `allow_nan=False` round trip. Schema was `cycle4_c4_01b.v1`. Metadata names the exact parent code SHA and records `git_dirty=false`. The six-file source digest recomputed to `979b5f473d1065236dffa7b36b3948edfc9caf9fd5dc3e8184e787b6aeaf75b6` and the protocol/config digest to `4cbd0322aafc2731`. The run reports 15 fresh units and zero resumed units.

**Actual data flow.** Initial manifest verification hashes all shards and creates read-only mappings. Stage A then binds only `ledger["train"]` and `ledger["val"]` inside `forbid("test")`. The search callback returns a validation NLL. It visits the largest Muon candidate first and brackets AdamW budgets on an 8-step grid. The pure selector takes the largest eligible Muon budget, then closest validation loss, fewer AdamW steps and lower LR. Replaying the nine recorded observations through the actual Stage A driver with identity-checked inputs reproduced every probe, bracket, selection and ledger count (train 1, val 1, test 0). Direct indexing raised under the lock. In separate probes, `.get("test")`, `.values()` and `.items()` returned the locked sentinel with the counter still zero.

The audit regenerated all 128 distinct, aligned train windows (65,536 tokens) from seed 20261126 and compared every start to the artifact. FP64 input second moments and GPTQ use those windows. Static activation scales use the same windows through the restored BF16 model in rotated coordinates. The butterfly fit takes weights only. Every candidate is evaluated on both held-out splits. Selection uses pooled validation NLL. The benchmark also records test rankings for confirmation. The full evaluations cover 2,001,920 validation targets and 1,999,872 test targets per row. All 150 rows have the same split counts as their own baseline.

**Output guards.** At the current canonical path, at its basename in another checkout, and at a temporary symlink alias, the parser rejected all 15 tested quick/protocol overrides per target. Quick mode defaults to its own filename. A non-default protocol without an explicit output was rejected. The device gate rejected CPU and simulated capabilities 9.0 and 10.0. It accepted 12.0 and 12.1, consistent with the implementation's minimum capability of 12.0. The recorded and live device were both RTX 5080, capability 12.0. These probes stopped before experiment execution and file creation at the target.

**Resume and validator probes.** Varying calibration seed caused a cache miss. A substituted source-byte change to each of the six modules in `code_sha256()` also caused a miss. Identical changes to `tier2_candidates`, `schatten_switch_blackwell`, `polar_switch_blackwell` and `real_text_data` all left a synthetic unit reusable. The key includes the device name and torch version but omits git revision, driver and Triton version. No checkout files were modified during the probes.

The production validator accepted the unmodified artifact and rejected a corrupted raw delta and a mismatched restoration hash. Independent one-field probes were accepted for dirty provenance, invalid git/code/config hashes, a test-labelled training or static-scale source, calibration starts `[-1]`, a missing pilot search trace, removal of the 0.001 full-budget AdamW trial, auxiliary LR 99, training token count 7, and a nonexistent diagnostic layer name. These illustrate the gap between internal consistency and independent evidence. The actual artifact contains the missing-LR trial, correct train inputs, expected auxiliary LR, regenerated starts and all 60 unique projection names.

## 2. Validation selection and paired statistics

All differences below use Muon minus AdamW. A positive **extra PTQ loss** means Muon loses more NLL when quantized. A negative **absolute difference** means the quantized Muon model has lower NLL. All intervals and losses use nats per token.

### Baseline transfer

| Seed | Muon val | AdamW val | Val difference | Muon test | AdamW test | Test difference |
|---|---:|---:|---:|---:|---:|---:|
| 0, pilot | 5.339254 | 5.325521 | +0.013732 | — | — | — |
| 1 | 5.356087 | 5.446575 | −0.090489 | 5.391238 | 5.478862 | −0.087624 |
| 2 | 5.346075 | 5.346474 | −0.000399 | 5.380844 | 5.375936 | +0.004908 |
| 3 | 5.346856 | 5.490772 | −0.143916 | 5.380075 | 5.522440 | −0.142365 |

AdamW uses 544 steps and 17,825,792 tokens at LR 0.0015. Muon uses 320 steps and 10,485,760 tokens at LR 0.02. AdamW uses 1.70 times as many tokens. On seed 0, the 528-, 536- and 544-step AdamW trials at the selected LR have validation NLLs 5.367608, 5.236636 and 5.325521. The last ten training losses average 5.289375, 5.166811 and 5.251845. The 544-step result is 0.088885 nats worse than the 536-step result. These are three fresh runs from the same initial weights and stream with warmups of 52, 53 and 54 steps and different cosine schedules. They do not measure deterioration between adjacent snapshots of one run. The data establish sensitivity to the training recipe. They do not identify its precise numerical cause. The pilot meets the specified seed-0 gate. The held-out gaps of 0.090489 and 0.143916 warrant a caveat and block a three-seed quality-matched interpretation.

### Independent arithmetic

The independent script uses the raw NLL and BF16 baseline to derive each delta, `exp(NLL)` for perplexity, and `100 × expm1(delta)` for the percentage change. It checks every baseline and both splits of every grid row. It independently computes pooled, optimizer-specific and checkpoint-specific rankings. For paired tests it uses the sample SD of the three differences, SE = SD/√3, t = mean/SE, and SciPy's two-sided Student-t survival function with df = 2. It independently implements Holm's sorted step-down procedure including the cumulative maximum. Results agree within absolute/relative tolerance 1e−12. The protocol critical value is 4.302652729911275. Inverting the df = 2 CDF gives 4.302652729749464. Switching constants moves any of the 25 test-comparison CI endpoints by at most 7.05e−12 nats.

The adjustments apply separately to the sensitivity and absolute endpoints. The intervals condition on the selected methods and the three observed seed pairs. They do not propagate uncertainty from the seed-0 calibration or from method selection across new seeds. Each primary cell chose adaptive butterfly B16 + GPTQ on pooled validation and confirmed it on pooled test. The FP32 emulation cell has one method and is outside the six-cell selection family.

| Format / mode | Muon ΔNLL | AdamW ΔNLL | Ratio | Extra loss | 95% CI | t | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NVFP4 g16 W4 | 0.001972 | 0.000374 | 5.28 | 0.001598 | 0.001282, 0.001914 | 21.74 | 0.002110 | 0.002110 |
| NVFP4 g16 W4A4 dynamic | 0.011022 | 0.005060 | 2.18 | 0.005962 | 0.005415, 0.006508 | 46.92 | 0.000454 | 0.001596 |
| NVFP4 g16 W4A4 static | 0.011018 | 0.005132 | 2.15 | 0.005886 | 0.005229, 0.006542 | 38.59 | 0.000671 | 0.001596 |
| MXFP4 g32 W4 | 0.003688 | 0.001070 | 3.45 | 0.002618 | 0.002461, 0.002774 | 71.90 | 0.000193 | 0.001160 |
| INT2 g16 W2 | 0.033131 | 0.003272 | 10.13 | 0.029859 | 0.027434, 0.032284 | 52.98 | 0.000356 | 0.001596 |
| INT2 g32 W2 | 0.053718 | 0.004504 | 11.93 | 0.049215 | 0.045431, 0.052999 | 55.96 | 0.000319 | 0.001596 |

**Absolute quantized test NLL.** The mean BF16 test offset is −0.075027 nats. It enters each absolute comparison additively. Each absolute 95% CI spans zero and all Holm-adjusted p-values equal 1.

| Format / mode | Muon NLL | AdamW NLL | Difference | 95% CI | t | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| NVFP4 g16 W4 | 5.386024 | 5.459453 | -0.073429 | -0.258626, 0.111768 | -1.706 | 0.230133 | 1.000000 |
| NVFP4 g16 W4A4 dynamic | 5.395074 | 5.464139 | -0.069066 | -0.254169, 0.116038 | -1.605 | 0.249625 | 1.000000 |
| NVFP4 g16 W4A4 static | 5.395070 | 5.464211 | -0.069141 | -0.254305, 0.116022 | -1.607 | 0.249371 | 1.000000 |
| MXFP4 g32 W4 | 5.387740 | 5.460149 | -0.072409 | -0.257464, 0.112645 | -1.684 | 0.234298 | 1.000000 |
| INT2 g16 W2 | 5.417183 | 5.462351 | -0.045168 | -0.228789, 0.138453 | -1.058 | 0.400823 | 1.000000 |
| INT2 g32 W2 | 5.437770 | 5.463583 | -0.025812 | -0.208075, 0.156450 | -0.609 | 0.604294 | 1.000000 |

**Sensitivity to optimizer-specific selection.** Muon keeps the adaptive butterfly method for all six cells, and test confirms all six. AdamW chooses plain GPTQ for NVFP4 fakequant and both hardware modes, Hadamard B16 + GPTQ for MXFP4 and INT2 group-16, and adaptive butterfly for INT2 group-32. AdamW's own test ranking confirms five of six. MXFP4 instead favors plain GPTQ on test. Selecting each optimizer's method on validation preserves all six signs and all six corrected results. In table order the extra losses are 0.001600, 0.006231, 0.006282, 0.002644, 0.030038 and 0.049215 nats, with adjusted p-values 0.002350, 0.003054, 0.003054, 0.000273, 0.001596 and 0.001596.

### All 25 test combinations

The secondary Holm family covers all 25 rows, including the emulation control. All 25 extra-loss differences are positive on test and significant under the paired-t model after this correction (adjusted p range 0.001553–0.012412). The same holds on validation (0.002897–0.011802). All 25 absolute NLL Holm p-values equal 1 on both splits. Even the absolute point estimates change direction for plain INT2 RTN on test (+0.057171 at group 16 and +0.169488 at group 32). The reproduction script checks the full t, p, confidence interval and adjusted p for **both** endpoints on each split. In the table, H means Hadamard B16 + GPTQ and B means adaptive butterfly B16 + GPTQ. W4 and W2 are weight-only fakequant with BF16 inference.

| Format / mode | Method | Extra loss | t | 95% CI | p | Holm p (25) | Absolute difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| NVFP4 g16 W4 | RTN | 0.007680 | 18.59 | 0.005902, 0.009458 | 0.002882 | 0.012412 | -0.067347 |
| NVFP4 g16 W4A4 dynamic | RTN | 0.012824 | 40.93 | 0.011476, 0.014172 | 0.000596 | 0.007157 | -0.062203 |
| NVFP4 g16 W4A4 static | RTN | 0.012780 | 126.85 | 0.012347, 0.013214 | 0.000062 | 0.001553 | -0.062247 |
| NVFP4 g16 W4 | GPTQ | 0.002188 | 59.57 | 0.002030, 0.002346 | 0.000282 | 0.005352 | -0.072839 |
| NVFP4 g16 W4A4 dynamic | GPTQ | 0.006961 | 20.54 | 0.005503, 0.008419 | 0.002362 | 0.012412 | -0.068066 |
| NVFP4 g16 W4A4 static | GPTQ | 0.007004 | 23.89 | 0.005742, 0.008265 | 0.001748 | 0.012412 | -0.068024 |
| NVFP4 g16 W4 | H | 0.001827 | 44.79 | 0.001652, 0.002003 | 0.000498 | 0.006475 | -0.073200 |
| NVFP4 g16 W4A4 dynamic | H | 0.006902 | 31.16 | 0.005949, 0.007855 | 0.001029 | 0.010286 | -0.068125 |
| NVFP4 g16 W4A4 static | H | 0.006906 | 23.97 | 0.005667, 0.008146 | 0.001735 | 0.012412 | -0.068121 |
| NVFP4 g16 W4A4 emulated | H | 0.006907 | 26.90 | 0.005802, 0.008011 | 0.001379 | 0.012412 | -0.068120 |
| NVFP4 g16 W4 | B | 0.001598 | 21.74 | 0.001282, 0.001914 | 0.002110 | 0.012412 | -0.073429 |
| NVFP4 g16 W4A4 dynamic | B | 0.005962 | 46.92 | 0.005415, 0.006508 | 0.000454 | 0.006354 | -0.069066 |
| NVFP4 g16 W4A4 static | B | 0.005886 | 38.59 | 0.005229, 0.006542 | 0.000671 | 0.007380 | -0.069141 |
| MXFP4 g32 W4 | RTN | 0.011190 | 19.27 | 0.008691, 0.013690 | 0.002683 | 0.012412 | -0.063837 |
| MXFP4 g32 W4 | GPTQ | 0.003393 | 26.23 | 0.002837, 0.003950 | 0.001451 | 0.012412 | -0.071634 |
| MXFP4 g32 W4 | H | 0.002766 | 109.92 | 0.002657, 0.002874 | 0.000083 | 0.001986 | -0.072261 |
| MXFP4 g32 W4 | B | 0.002618 | 71.90 | 0.002461, 0.002774 | 0.000193 | 0.004061 | -0.072409 |
| INT2 g16 W2 | RTN | 0.132198 | 23.14 | 0.107614, 0.156783 | 0.001863 | 0.012412 | 0.057171 |
| INT2 g16 W2 | GPTQ | 0.036579 | 78.37 | 0.034570, 0.038587 | 0.000163 | 0.003581 | -0.038449 |
| INT2 g16 W2 | H | 0.031670 | 51.88 | 0.029044, 0.034297 | 0.000371 | 0.006053 | -0.043357 |
| INT2 g16 W2 | B | 0.029859 | 52.98 | 0.027434, 0.032284 | 0.000356 | 0.006053 | -0.045168 |
| INT2 g32 W2 | RTN | 0.244515 | 49.60 | 0.223305, 0.265724 | 0.000406 | 0.006093 | 0.169488 |
| INT2 g32 W2 | GPTQ | 0.062070 | 71.75 | 0.058347, 0.065792 | 0.000194 | 0.004061 | -0.012958 |
| INT2 g32 W2 | H | 0.051811 | 98.76 | 0.049554, 0.054069 | 0.000103 | 0.002358 | -0.023216 |
| INT2 g32 W2 | B | 0.049215 | 55.96 | 0.045431, 0.052999 | 0.000319 | 0.005745 | -0.025812 |

### Distribution diagnostics

The independent reconstruction checked the 60 per-layer values per metric for each checkpoint. It recomputed all five metrics' means, medians, minima and maxima, both across layers and for each of five projection kinds. It then checked all 90 reported paired tests (five metrics × six scopes × three statistics), including the six input values for each. The five metrics are inputs at this stage. Reconstruction tests their aggregation. The accelerator replay measures the quantities again on new weights. These tests are descriptive. The table shows the per-checkpoint mean over all 60 layers, then the paired comparison over three seeds.

| Metric | Muon | AdamW | Difference | 95% CI | t | Unadjusted p |
|---|---:|---:|---:|---:|---:|---:|
| weight excess kurtosis | 0.4765 | 0.2191 | 0.257465 | 0.190796, 0.324133 | 16.616 | 0.003602 |
| weight max over rms | 5.8153 | 5.6745 | 0.140768 | -0.088589, 0.370126 | 2.641 | 0.118453 |
| activation excess kurtosis | 48.5007 | 63.8139 | -15.313194 | -128.196929, 97.570541 | -0.584 | 0.618495 |
| activation max over rms | 26.2378 | 26.4025 | -0.164706 | -9.639845, 9.310434 | -0.075 | 0.947187 |
| hessian diag max over mean | 8.3190 | 15.1566 | -6.837600 | -24.158050, 10.482851 | -1.699 | 0.231500 |

## 3. Training path and hardware

The requested three pytest files yielded **253 passed, 0 skipped, 0 failed in 120.27 s**. Nine warnings concerned CUDA sync debugging, an intentionally empty CUDA graph, and Triton AST deprecations. The run used Python 3.13, PyTorch 2.14.1+cu130, Triton 3.8, driver 615.78.08 and an RTX 5080.

| Operation | Comparison of `bench_isoloss_ptq.train_checkpoint` (lines 129–169) with `harness_tier2_real.run_trial` (lines 514 onward) |
|---|---|
| Model and initial state | Both construct the transformer on the target device and load the seeded CPU state. The experiment calls `initial_state` once per seed and shares it across optimizers. Each model enters train mode. The model has no dropout. |
| Batches | Both use `BatchStream(train, config, seed, device)` and fresh NumPy generators. Each step has 4 × 16 × 512 = 32,768 target tokens. The shared prefix spans 320 steps. The stored stream fingerprint covers the first three groups of starts. |
| Parameters and optimizer | Both call `build_optimizer`. AdamW uses fused CUDA AdamW, betas (0.9, 0.95), epsilon 1e−8, decay 0.01 on matrices including embedding, zero on norms. Muon applies decoupled matrix decay and Nesterov momentum to the 60 hidden matrices, with the SS3_54 FP16 orthogonalizer and FP32 update cast. Its auxiliary AdamW uses LR 0.001, decays embedding and exempts norms. AdamW alone has no auxiliary LR. |
| Learning rate | Both set the multiplier before reading the next batch. The experiment supplies `warmup=max(20,T//10)` (32/54) to the token-budget linear-warmup/cosine schedule with floor ratio 0.1. The standalone harness takes an explicit warmup and defaults to 50. |
| Loss and gradients | Both call the same `train_step` and `_loss`. CUDA BF16 autocast covers forward and cross entropy. Loss is divided across four microbatches before backward. There is one optimizer step and gradients are cleared with `set_to_none=True`. Neither path clips gradients. |
| Driver differences | The benchmark omits the harness's disposable compile warmup, timing instrumentation, optional compilation, wall-budget rollback and final held-out evaluation. It synchronizes loss via `.item()`, aborts on nonfinite losses and returns the trained model in eval mode. These are deliberate differences from the generic harness. |
| Determinism | The experiment disables TF32 and reduced-precision FP16/BF16 reductions. CPU initialization and the sampling stream are seeded. Global deterministic algorithms are off on this host, and the experiment does not pin all kernel choices. Independent checkpoint digests and prefix tests below quantify the practical consequence. |


### Checkpoint identity and hardware execution

Seed 2 reused the canonical preset, initial seed, stream seed, LR budgets, full train calibration and full held-out evaluation. The following are signed Muon-minus-AdamW baseline gaps.

| Seed-2 data | Muon val | AdamW val | Val gap | Muon test | AdamW test | Test gap |
|---|---:|---:|---:|---:|---:|---:|
| Artifact | 5.346075 | 5.346474 | -0.000399 | 5.380844 | 5.375936 | 0.004908 |
| Live retrain | 5.338893 | 5.440347 | -0.101454 | 5.372738 | 5.471328 | -0.098590 |

| Optimizer | Artifact checkpoint SHA-256 | Retrained SHA-256, identical before and after all 25 replayed rows |
|---|---|---|
| Muon | `ec60f31b983abac61ce0a0adb49e1514b4abd484da66060254c3fbb3fb184121` | `f9043dcbeb8f69fdeb48667a5899531f693f07ebaba6691672ab6a7151d39643` |
| AdamW | `460b2bd3cf873ca43ff69b152ce6cd0ce81609a5d24bb9c47865af760ab4af1b` | `d733123c1568443211e086317fc1bab02fd92aa6e63119bfe4be555d751db1ea` |

The artifact's training, pre-PTQ and post-PTQ hashes agree for each of its six checkpoints. The replay independently established within-run restoration of both retrained models. The hash covers the named persistent state tensors. The original driver stores hashes in JSON, releases the checkpoint after evaluation, and has no weights in its resume log. The original pair's 0.000399/0.004908 baseline gaps therefore apply only to the stored observations. Two additional 40-step prefixes per optimizer matched the artifact's initialization fingerprint `eff774e6f234e5e7`, stream fingerprint `f32172b52cfa7de3`, and first loss (10.9232826233). Repeats of each optimizer already had different post-update checkpoint hashes at step 1. At step 40, losses were 6.938287 and 6.928617 for Muon, and 7.458472 and 7.440907 for AdamW. This localizes measurable variation to training before PTQ. The specific source of that variation remains unresolved.

For each replayed checkpoint, a warmed 16 × 512 validation forward was profiled inside the active model path for baseline and every grid row. CPU dispatch counts and CUDA kernel counts were recorded independently. Complete test and validation NLLs were then measured outside the profiler.

| Forward mode | Profiled forwards across the two retrains | `aten::_scaled_mm_v2` per forward | SM120 block-scaled FP4 GEMMs per forward |
|---|---:|---:|---:|
| BF16 baseline | 2 | 0 | 0 |
| W4A16 / W2A16 fakequant | 32 | 0 | 0 |
| NVFP4 W4A4 dynamic | 8 | 60 | 60 |
| NVFP4 W4A4 static | 8 | 60 | 60 |
| W4A4 FP32 emulation | 2 | 0 | 0 |

The CUDA events identify CUTLASS `sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16` kernels. The count matches 12 blocks × 5 projections. For each of four NVFP4 methods on each retrain, the packed hardware weight decoded to the rotated fakequant FP32 tensor in all 60 integer bit-pattern comparisons (480/480 matrices). All 24 corresponding recorded checks also report 60/60. Both replayed full-split pairs met the 1e−3-nat hardware/emulation bound for dynamic Hadamard GPTQ. Signed hardware-minus-emulator NLL differences (validation, test) were (−0.000028421, −0.000103761) for Muon and (−0.000031495, −0.000006157) for AdamW. The maximum over the original 12 split/optimizer/seed contrasts was 0.000143552 nats. This control tests whole-model NLL agreement for one method and mode. It does not establish elementwise output equality.

## 4. Why the loss gap survives on hardware

The first two numeric columns below use the **recorded, closely matched** pair. The last two use the fresh retrains. All numbers come from full test evaluations. The method is held fixed at the original pooled validation choice for both optimizers and both runs.

| Primary cell | Artifact Muon / AdamW Δ | Artifact ratio | Live Muon / AdamW Δ | Live ratio |
|---|---:|---:|---:|---:|
| NVFP4 g16 W4 | 0.002103 / 0.000359 | 5.85 | 0.002063 / 0.000423 | 4.88 |
| NVFP4 g16 W4A4 dynamic | 0.011179 / 0.005227 | 2.14 | 0.011048 / 0.004657 | 2.37 |
| NVFP4 g16 W4A4 static | 0.010958 / 0.005072 | 2.16 | 0.011208 / 0.004659 | 2.41 |
| MXFP4 g32 W4 | 0.003714 / 0.001025 | 3.62 | 0.003736 / 0.001018 | 3.67 |
| INT2 g16 W2 | 0.032765 / 0.003124 | 10.49 | 0.033225 / 0.002481 | 13.39 |
| INT2 g32 W2 | 0.053602 / 0.005088 | 10.54 | 0.054062 / 0.004135 | 13.07 |

### Separating scalar magnitude from geometry

For each quantized layer, let `W` be its master weight, `E = W − Q` its error after the inverse rotation, and `H = XᵀX/N` its uncentered input second moment from training tokens. Let `d` be the layer's input width. Define alignment for any matrix B as `a(B) = tr(BHBᵀ) / (||B||²_F · tr(H)/d)`. A value of 1 is the isotropic reference. The exact per-layer relationship is

```text
relative output error = tr(EHEᵀ) / tr(WHWᵀ)
                      = (||E||²_F / ||W||²_F) × a(E) / a(W).
```

The live probes reconstructed both sides for all 1,920 layer/format/method measurements (2 optimizers × 16 weight-only rows × 60 matrices) and checked their means against the grid. The largest discrepancy in the 32 grid means was below 4.47e−12 in absolute terms (1.81e−10 relative maximum). The benchmark forms E in FP32 before contraction. The independent probe subtracts in FP64. Values below are equal-weight layer means. The identity holds before aggregation.

| Optimizer | Mean Frobenius norm | Mean stable rank | Signal alignment, own H | Signal alignment, other H |
|---|---:|---:|---:|---:|
| muon_ss3_54 | 46.9752 | 172.3000 | 1.1481 | 1.0565 |
| adamw | 16.5003 | 23.3355 | 8.2969 | 2.6099 |

Stable rank is `||W||²_F / σ₁(W)²`. The top eight eigenvectors of each layer's own H captured **1.48% of Muon vs 14.81% of AdamW weight energy** and carried **39.83% vs 68.17% of output signal energy**. These directions account for 42.13% vs 45.65% of total input energy. Every one of the 60 Muon matrices had higher stable rank than its paired AdamW matrix, and every AdamW matrix placed more weight energy in its own eight leading input directions. This separates the orientation of learned signal from the marginal distribution of input energy. The other-H column is a counterfactual that holds W fixed and uses the other optimizer's matching layer covariance.

| Format / rounding | Unweighted error M / A | Signal alignment M / A | Error alignment M / A | Relative output error M / A |
|---|---:|---:|---:|---:|
| NVFP4 g16 RTN | 0.00902 / 0.00907 | 1.14805 / 8.29686 | 1.03532 / 1.07080 | 0.00885 / 0.00160 |
| NVFP4 g16 butterfly GPTQ | 0.01423 / 0.01691 | 1.14805 / 8.29686 | 0.16369 / 0.06126 | 0.00218 / 0.00019 |
| MXFP4 g32 RTN | 0.01332 / 0.01355 | 1.14805 / 8.29686 | 1.07539 / 1.37499 | 0.01333 / 0.00293 |
| MXFP4 g32 butterfly GPTQ | 0.02069 / 0.02576 | 1.14805 / 8.29686 | 0.16463 / 0.06415 | 0.00320 / 0.00030 |
| INT2 g16 RTN | 0.10773 / 0.10617 | 1.14805 / 8.29686 | 0.99410 / 1.02282 | 0.10277 / 0.01845 |
| INT2 g16 butterfly GPTQ | 0.18205 / 0.21001 | 1.14805 / 8.29686 | 0.16626 / 0.06557 | 0.02825 / 0.00260 |
| INT2 g32 RTN | 0.15644 / 0.15321 | 1.14805 / 8.29686 | 0.99647 / 1.02213 | 0.14917 / 0.02671 |
| INT2 g32 butterfly GPTQ | 0.26640 / 0.30975 | 1.14805 / 8.29686 | 0.16554 / 0.06669 | 0.04130 / 0.00395 |

**Quantizer control.** Scaling the actual master tensors, quantizing, and dividing the result back out tests whether large weight magnitudes alone explain the difference. RTN covered all four formats and all 120 matrices. Identity-rotation GPTQ additionally covered NVFP4 and INT2 g32 at factors 0.5 and 2. NVFP4 and MXFP4 RTN were exactly equivariant at both powers of two on the tested weights, as was GPTQ for both tested formats. INT2 g16 RTN showed tiny finite-precision changes. Across all tested power-of-two cases, the largest squared discrepancy `||Q(cW)/c − Q(W)||²_F / ||W||²_F` was 1.01e−15 and the squared-error ratio differed from 1 by at most 2.58e−10. At factor 1.7, the table gives the range over individual matrices of squared error after inverse scaling divided by the original squared error. MXFP4 uses a power-of-two shared exponent, so arbitrary scaling can move it between grids. The asymmetric INT2 formats also round their scale and minimum to FP16.

| Format, RTN at 1.7× | Min. relative error ratio | Max. relative error ratio |
|---|---:|---:|
| NVFP4 g16 | 0.999991 | 1.000019 |
| MXFP4 g32 | 0.948093 | 1.063562 |
| INT2 g16 | 0.999976 | 1.000024 |
| INT2 g32 | 0.999968 | 1.000019 |

**GPTQ intervention.** In identity coordinates, replacing H by the identity made GPTQ equal RTN for every tested weight in NVFP4 and INT2 g32. The next table scores every quantizer against the authentic H. A fixed, symmetric channel permutation preserves the eigenvalues of the H supplied to GPTQ while breaking the association between input channels and their observed covariance. Applying this intervention to both optimizers distinguishes the effect of error redistribution from the scalar quantization grid.

| Format | Optimizer | RTN | GPTQ, own H | GPTQ, permuted H | RTN / native GPTQ |
|---|---:|---:|---:|---:|---:|
| NVFP4 g16 | muon_ss3_54 | 0.008855 | 0.002530 | 0.014228 | 3.50 |
| NVFP4 g16 | adamw | 0.001603 | 0.000224 | 0.002999 | 7.15 |
| INT2 g32 | muon_ss3_54 | 0.149174 | 0.047233 | 0.265360 | 3.16 |
| INT2 g32 | adamw | 0.026712 | 0.004734 | 0.058203 | 5.64 |

**Mechanistic verdict.** In the live pair, NVFP4 RTN has nearly equal relative weight errors (0.00902 Muon, 0.00907 AdamW). With butterfly GPTQ, AdamW actually has the larger unweighted error (0.01691 vs 0.01423), yet its relative output error is 0.000190 vs Muon's 0.002179. AdamW puts more useful signal in the leading input directions, and GPTQ pushes more of its rounding error into directions with little input energy. The two effects reinforce one another. Plain scalar weight magnitude, inaccurate hardware decode and a fakequant-only execution path do not account for these observations.

For the original matched seed-2 pair, the stored relative output error with butterfly GPTQ is 0.002181 vs 0.000296 for NVFP4 and 0.041301 vs 0.006076 for INT2 g32 (Muon vs AdamW). On both the original and live pairs the relative sensitivity ratio narrows on hardware W4A4, where quantizing activations adds another source of error. A local linear-layer proxy explains why the same size weight perturbation can matter more. Whole-model test NLL also depends on nonlinear interactions. The live validation gap of 0.101454 nats also prevents attributing the original matched pair's exact effect size to a single cause. The evidence supports a genuine geometry-dependent sensitivity difference in this experiment.

**Disposition.** Retain the measured PTQ sensitivity gap for these locked recipes and record the hardware results. Reserve exact-checkpoint mechanistic attribution for a run that preserves the original tensors. Promotion of a general optimizer preference would also require additional independently matched seeds and a decision criterion that covers absolute deployed loss.


## Reproduction

Run from the repository root in an isolated worktree at this audit commit (its parent is the result commit), the pinned shards available at the recorded path, the project virtual environment, and exclusive access to the RTX 5080. The independent arithmetic uses SciPy. The full live replay evaluates 25 rows plus baseline on both complete held-out splits for each retrained model. The appendix contains the exact standalone probes so the audit can be reproduced after scratch cleanup. The script loader extracts them from this report into a private scratch directory. Training can yield different weights and timing on subsequent runs.

```bash
set -eu
test "$(git rev-parse HEAD:results/cycle4_c4_01b_isoloss_ptq.json)" = "$(git rev-parse 3f954e4:results/cycle4_c4_01b_isoloss_ptq.json)"
git diff --exit-code 3f954e411666801bbfa879cf079cd063fe6e701c -- research tests results
git log -2 --format=%H 3f954e411666801bbfa879cf079cd063fe6e701c
git diff --stat 0659435bb221fb7c4f19394e52e616daf240306f 3f954e411666801bbfa879cf079cd063fe6e701c
sha256sum /home/ryan/Documents/projects/ai/artifacts/real_text/{train,val,test}.bin
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
install -d -m 0700 "$HOME/tmp"
task_tmp=$(mktemp -d "$HOME/tmp/pi-isoloss-audit.XXXXXX")
trap 'if [ -d "$task_tmp" ]; then rm -r -- "$task_tmp"; fi' EXIT
export TMPDIR="$task_tmp" TMP="$task_tmp" TEMP="$task_tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export PATH="/home/ryan/Documents/projects/ai/.venv/bin:$PATH"
python - <<'PY'
import os, re
from pathlib import Path
f = Path("audits/AUDIT_cycle4_c4_01b_gpt62_astra.md")
files = re.findall(r"^#### `([a-z]+\.py)`\n\n```python\n(.*?)\n```$", f.read_text(), re.M | re.S)
assert len(files) == 8
for name, src in files:
    (Path(os.environ["TMPDIR"]) / name).write_text(src + "\n")
PY
PYTEST_ADDOPTS='-p no:cacheprovider' PYTHONPATH=. /home/ryan/Documents/projects/ai/.venv/bin/pytest tests/test_isoloss_ptq.py tests/test_micro_rotated_ptq.py tests/test_nvfp4_inference.py -q
python "$TMPDIR/arithmetic.py"
python "$TMPDIR/guards.py"
python -u "$TMPDIR/train.py"
python -u "$TMPDIR/replay.py"
OPENBLAS_NUM_THREADS=2 python "$TMPDIR/spectra.py"
OPENBLAS_NUM_THREADS=2 python "$TMPDIR/geometry.py"
python -u "$TMPDIR/prefix.py"
python "$TMPDIR/assess.py"
```

To rerun the entire author experiment separately, use the same environment and a fresh scratch directory, then run `python research/quant_ptq/bench_isoloss_ptq.py --data-dir /home/ryan/Documents/projects/ai/artifacts/real_text --output "$TMPDIR/c4-01b-reproduction.json"`. This audit retrained seed 2 and checked the complete six-checkpoint artifact independently. Training for seeds 0, 1 and 3 is represented by the stored observations.


<details>
<summary>Executable audit probes</summary>

#### `arithmetic.py`

```python
import json, math, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t

P = Path("results/cycle4_c4_01b_isoloss_ptq.json")


def reject(x):
    raise ValueError(x)


def unique(pairs):
    d = dict(pairs)
    if len(d) != len(pairs):
        reject("duplicate")
    return d


def strict(s):
    return json.loads(s, parse_constant=reject, object_pairs_hook=unique)


for bad in ["NaN", "Infinity", "-Infinity", '{"a":1,"a":2}']:
    try:
        strict(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(bad)
p = strict(P.read_text())
assert strict(json.dumps(p, allow_nan=False)) == p
assert (
    p["schema_version"] == "cycle4_c4_01b.v1"
    and p["meta"]["git_sha"] == "0659435bb221fb7c4f19394e52e616daf240306f"
    and p["meta"]["git_dirty"] is False
)
A = p["analysis"]
cp = p["stage_b"]["checkpoints"]
opts = ("muon_ss3_54", "adamw")


def ck(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def block(x):
    return dict(values=list(x), mean=st.mean(x), sd=st.stdev(x))


def checkblock(raw, b):
    for k, v in b.items():
        ck(raw[k], v)


old = 4.302652729911275
exact = math.sqrt(2 * 0.95**2 / (1 - 0.95**2))
ck(t.ppf(0.975, 2), exact)
ck(A["t_quantile_0p975_df2"], old)
ck(A["t_quantile_0p975_df2_closed_form"], exact)
ci_gap = []


def paired(x):
    mu = st.mean(x)
    sd = st.stdev(x)
    se = sd / math.sqrt(3)
    z = mu / se
    ci_gap.append((old - exact) * se)
    return dict(
        diffs=x,
        n=3,
        df=2,
        mean=mu,
        sd=sd,
        se=se,
        t=z,
        ci95=[mu - old * se, mu + old * se],
        p_two_sided=2 * t.sf(abs(z), 2),
    )


def holm(ps):
    order = np.argsort(ps)
    return np.minimum(1, np.maximum.accumulate(np.array(ps)[order] * np.arange(len(ps), 0, -1)))[
        np.argsort(order)
    ]


rs = {}
ckpts = {}
groups = defaultdict(list)
dg = {}
splits = ("val", "test")


def combo(r):
    return tuple(r[k] for k in ("format", "method", "mode"))


def best(pool, s):
    scores = defaultdict(lambda: defaultdict(list))
    for r in pool:
        f, m, v = combo(r)
        scores[(f, v)][m].append(r[s + "_nll"])
    return {fv: min(meth, key=lambda m: st.mean(meth[m])) for fv, meth in scores.items()}


def checkbest(pool, obj):
    v, y = [best(pool, s) for s in splits]
    for (f, mode), m in v.items():
        a = obj[f + "/" + mode]
        assert (m, y[f, mode], m == y[f, mode]) == (
            a["best_by_val"],
            a["best_by_test"],
            a["confirmed_on_test"],
        )


for c in cp:
    seed, o = c["seed"], c["optimizer"]
    ckpts[seed, o] = c
    assert c["checkpoint_sha256"] == c["checkpoint_sha256_after_ptq"] == c["training"]["checkpoint_sha256"]
    checkbest(c["grid"], c["best"])
    for s, n in zip(splits, (2001920, 1999872)):
        base = c["baseline_" + s]
        ck(base["ppl"], math.exp(base["nll"]))
        assert base["tokens"] == n
    for r in c["grid"]:
        key = combo(r)
        assert (seed, o, key) not in rs
        z = {}
        for s in splits:
            ref = c["baseline_" + s]
            nll = r[s + "_nll"]
            d = nll - ref["nll"]
            for field, val in [
                ("ppl", math.exp(nll)),
                ("delta_nll", d),
                ("relative_delta_ppl_pct", 100 * math.expm1(d)),
                ("nll", nll),
            ]:
                ck(r[s + "_" + field], val)
                z[s, field] = val
            assert ref["tokens"] == r[s + "_tokens"]
        rs[seed, o, key] = z
    groups[o] += c["grid"]
    groups["pooled"] += c["grid"]
    layers = c["diagnostics"]["per_layer"]
    kinds = ("qkv", "out", "gate", "up", "down")
    assert {r["name"] for r in layers} == {f"blocks.{i}.{k}" for i in range(12) for k in kinds}
    for field, obj in c["diagnostics"]["summary"].items():
        for scope in ("all", *kinds):
            xs = [r[field] for r in layers if scope == "all" or r["kind"] == scope]
            raw = obj["all"] if scope == "all" else obj["by_kind"][scope]
            for stat, fn in [("mean", st.mean), ("median", st.median), ("min", min), ("max", max)]:
                v = fn(xs)
                ck(raw[stat], v)
                dg[seed, o, field, scope, stat] = v
for k, v in groups.items():
    checkbest(v, A["method_selection"][k])
for s in splits:
    x = [
        ckpts[i, opts[0]]["baseline_" + s]["nll"] - ckpts[i, opts[1]]["baseline_" + s]["nll"]
        for i in (1, 2, 3)
    ]
    checkblock(A["baseline"][s + "_gap_muon_minus_adamw"], paired(x))


def values(key, o, s, field):
    return [rs[i, o, key][s, field] for i in (1, 2, 3)]


def diffs(x, y):
    return [a - b for a, b in zip(x, y)]


def calc(km, ka, s, field):
    return paired(diffs(values(km, opts[0], s, field), values(ka, opts[1], s, field)))


keys = [combo(r) for r in cp[0]["grid"]]
assert len(keys) == 25 and len(rs) == 150
assert keys == [combo(c) for c in A["comparisons"]]
prim = [(f, m, v) for (f, v), m in best(groups["pooled"], "val").items() if "emulated" not in v]
assert len(prim) == 6 and prim == [combo(c) for c in A["primary"]]
for name, ks, rows, ss in [
    ("all", keys, A["comparisons"], splits),
    ("primary", prim, A["primary"], ["test"]),
]:
    for s in ss:
        for fld, stat in [("delta_nll", "delta_nll_diff"), ("nll", "abs_nll_diff")]:
            stats = [calc(k, k, s, fld) for k in ks]
            adj = holm([v["p_two_sided"] for v in stats])
            for k, row, z, hp in zip(ks, rows, stats, adj):
                raw = row[s] if name == "all" else row
                for o, tag in zip(opts, ["muon", "adamw"]):
                    for f in ["nll", "delta_nll", "relative_delta_ppl_pct"]:
                        checkblock(raw[tag + "_" + f], block(values(k, o, s, f)))
                checkblock(raw[stat], z)
                ck(raw[stat]["p_holm_all_combinations" if name == "all" else "p_holm"], hp)
                if name == "primary":
                    assert raw[stat]["significant_holm_0p05"] == (hp < 0.05)
            print(name, s, fld, "Holm <.05:", sum(adj < 0.05), "range", min(adj), max(adj))
for f, scopes in A["diagnostics_paired"].items():
    for sc, stats in scopes.items():
        for stat, z in stats.items():
            m, a = ([dg[i, o, f, sc, stat] for i in (1, 2, 3)] for o in opts)
            checkblock(z, dict(muon=m, adamw=a, **paired(diffs(m, a))))
            if (sc, stat) == ("all", "mean"):
                print("DIAG", f, st.mean(m), st.mean(a), z["mean"], z["ci95"], z["t"], z["p_two_sided"])
for k in prim:
    f, m, v = k
    alts = [(f, best(groups[o], "val")[f, v], v) for o in opts]
    print("SELECT", f, v, [a[1] for a in alts], [best(groups[o], "test")[f, v] for o in opts])
    print("SEED2", f, v, *[values(k, o, "test", "delta_nll")[1] for o in opts])
sens = [
    calc(
        (f, best(groups[opts[0]], "val")[f, v], v),
        (f, best(groups[opts[1]], "val")[f, v], v),
        "test",
        "delta_nll",
    )
    for f, m, v in prim
]
print("SENS", [(z["mean"], pv) for z, pv in zip(sens, holm([z["p_two_sided"] for z in sens]))])
print("CI_MAX_DIFF", max(ci_gap))
for k, r in zip(keys, A["comparisons"]):
    z = r["test"]["delta_nll_diff"]
    a = r["test"]["abs_nll_diff"]
    print(
        "ROW",
        *k,
        *[
            f"{n:.6g}"
            for n in (
                z["mean"],
                z["t"],
                *z["ci95"],
                z["p_two_sided"],
                z["p_holm_all_combinations"],
                a["mean"],
                a["p_holm_all_combinations"],
            )
        ],
    )
for r in A["primary"]:
    for stat in ["delta_nll_diff", "abs_nll_diff"]:
        z = r[stat]
        print(
            "PRIMARY",
            r["format"],
            r["mode"],
            stat,
            *[f"{n:.7g}" for n in (z["mean"], z["t"], *z["ci95"], z["p_two_sided"], z["p_holm"])],
        )

pilot = p["stage_a_pilot"]
mu = sorted((r for r in pilot["rows"] if r["optimizer"] == opts[0]), key=lambda r: -r["steps"])
ad = [r for r in pilot["rows"] if r["optimizer"] == opts[1] and r["steps"] <= 768]
pairs = [(m, r) for m in mu for r in ad if abs(m["val_nll"] - r["val_nll"]) <= 0.02]
m, w = min(
    pairs,
    key=lambda pair: (
        -pair[0]["steps"],
        abs(pair[0]["val_nll"] - pair[1]["val_nll"]),
        pair[1]["steps"],
        pair[1]["lr"],
    ),
)
select = pilot["selection"]
ck(
    [m["steps"], w["steps"], w["lr"], abs(m["val_nll"] - w["val_nll"])],
    [select["t_muon"], select["t_adamw"], select["adamw_lr"], select["abs_val_gap"]],
)
parity = A["hardware_vs_emulation"]
obs = []
for row in parity["rows"]:
    seed, o = row["seed"], row["optimizer"]
    pair = ("nvfp4_g16", "hadamard_b16_gptq")
    hw = rs[seed, o, (*pair, "w4a4_hw_dynamic")]
    em = rs[seed, o, (*pair, "w4a4_emulated_fp32_dynamic")]
    for s in splits:
        v = hw[s, "nll"] - em[s, "nll"]
        ck(v, row[s + "_nll_hw_minus_emulated"])
        obs.append(abs(v))
assert len(obs) == 12 and max(obs) <= 1e-3 and parity["passed"]
ck(max(obs), parity["max_abs_nll_difference"])
```

#### `guards.py`

```python
from train import *
import copy, hashlib, io, contextlib
from dataclasses import asdict
from unittest.mock import patch

d = data()
cfg = b.h.PRESETS["50m"]
prot = p["protocol"]
meta = p["meta"]
for s, v in b.q.EXPECTED_SHA256.items():
    assert (
        hashlib.file_digest(open(Path(p["data"]["directory"]) / (s + ".bin"), "rb"), "sha256").hexdigest()
        == v
        == p["data"]["splits"][s]["sha256"]
        == p["data"]["expected_sha256"][s]
    )
b.validate_results(p, canonical=True)
env = meta["environment"]


def conf(code, protocol):
    x = dict(
        schema_version=p["schema_version"],
        config=asdict(cfg),
        protocol=protocol,
        data_sha256={k: v["sha256"] for k, v in p["data"]["splits"].items()},
        code_sha256=code,
        torch=env["torch"],
        device=env["device"],
    )
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:16]


base = b.code_sha256()
assert base == meta["code_sha256"] and conf(base, prot) == meta["config_hash"]
assert b.build_protocol(b.parse_args([])) == prot
real = Path.read_bytes
key = ("pilot", 0, b.ADAMW, 544, 0.0015)
path = D / "probe.units"
u = b.UnitLog(path, conf(base, prot), False)
u.cached(key, lambda: {"x": 1})
for f in (
    "bench_isoloss_ptq",
    "micro_rotated_ptq",
    "nvfp4_inference_blackwell",
    "bench_nvfp4_e2e",
    "harness_tier2_real",
    "harness_tier2_lm",
    "tier2_candidates",
    "schatten_switch_blackwell",
    "polar_switch_blackwell",
    "real_text_data",
):
    with patch.object(Path, "read_bytes", lambda path: real(path) + (b"#audit" if path.stem == f else b"")):
        keyhash = conf(b.code_sha256(), prot)
    r = b.UnitLog(path, keyhash, True).cached(key, lambda: {"x": 2})
    print("DEPENDENCY", f, "resumed", r["resumed_from_unit_log"])
    assert r["resumed_from_unit_log"] == (
        f in ("tier2_candidates", "schatten_switch_blackwell", "polar_switch_blackwell", "real_text_data")
    )
assert not b.UnitLog(path, conf(base, dict(prot, calibration_seed=3)), True).cached(key, lambda: {"x": 2})[
    "resumed_from_unit_log"
]
target = Path("/home/ryan/Documents/projects/ai/results") / b.CANONICAL_OUTPUT.name
alias = D / "alias.json"
alias.unlink(missing_ok=True)
alias.symlink_to(b.CANONICAL_OUTPUT)
overs = [
    ["--quick"],
    ["--preset", "tiny"],
    ["--pilot-seed", "4"],
    ["--seeds", "2", "3", "4"],
    ["--tolerance", ".03"],
    ["--t-muon-candidates", "64"],
    ["--adamw-lrs", ".002"],
    ["--adamw-max-steps", "16"],
    ["--adamw-step-quantum", "4"],
    ["--calibration-sequences", "8"],
    ["--calibration-seed", "1"],
    ["--max-eval-windows", "16"],
    ["--butterfly-steps", "1"],
    ["--device", "cpu"],
    ["--no-verify"],
]
for out in (target, b.CANONICAL_OUTPUT, alias):
    for override in overs:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                b.parse_args(["--output", str(out), *override])
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError(override)
    for cap in ((9, 0), (10, 0), (12, 0), (12, 1)):
        with patch.object(torch.cuda, "get_device_capability", return_value=cap):
            try:
                b.require_blackwell(torch.device("cuda"), out)
            except RuntimeError:
                assert cap < (12, 0)
            else:
                assert cap >= (12, 0)
    try:
        b.require_blackwell(torch.device("cpu"), out)
    except RuntimeError:
        pass
    else:
        raise AssertionError(out)
assert b.parse_args(["--quick"]).output == b.QUICK_OUTPUT
with contextlib.redirect_stderr(io.StringIO()):
    try:
        b.parse_args(["--max-eval-windows", "1"])
    except SystemExit:
        pass
    else:
        raise AssertionError()
ledger = b.SplitLedger(dict(train=1, val=2, test=3))
with ledger.forbid("test"):
    try:
        ledger["test"]
    except RuntimeError:
        pass
    else:
        raise AssertionError()
    assert (
        ledger.get("test") == 3
        and 3 in ledger.values()
        and ("test", 3) in ledger.items()
        and ledger.reads["test"] == 0
    )
d.splits = b.SplitLedger(d.splits)
rows = {(r["optimizer"], r["steps"], r["lr"]): r for r in p["stage_a_pilot"]["rows"]}
tr, val = d.splits["train"], d.splits["val"]


def fake(o, c, x, device, *, seed, steps, lr, init):
    assert x is tr and seed == 0
    r = rows[o, steps, lr]
    return r["val_nll"], r.copy()


def evaluate(n, x, *args):
    import math

    assert x is val
    return dict(nll=n, ppl=math.exp(n), tokens=2001920)


with patch.object(b, "train_checkpoint", fake), patch.object(b.q, "evaluate_split", evaluate):
    replay = b.stage_a_pilot(
        cfg, d, torch.device("cpu"), prot, b.UnitLog(None, "fresh", False), lambda *a: None
    )
assert [r["val_nll"] for r in replay["rows"]] == [r["val_nll"] for r in p["stage_a_pilot"]["rows"]]
assert (
    replay["selection"] == p["stage_a_pilot"]["selection"]
    and replay["search"] == p["stage_a_pilot"]["search"]
    and replay["split_reads"] == dict(train=1, val=1, test=0)
)
for label, path, value in [
    ("dirty", ["meta", "git_dirty"], True),
    ("git", ["meta", "git_sha"], "garbage"),
    ("code", ["meta", "code_sha256"], "garbage"),
    ("hash", ["meta", "config_hash"], "stale"),
    ("train label", ["data", "training_split"], "test"),
    ("starts", ["stage_b", "calibration", "starts"], [-1]),
    ("trace", ["stage_a_pilot", "search"], []),
    ("aux LR", ["stage_b", "checkpoints", 0, "training", "auxiliary_lr"], 99),
    ("decoy layer", ["stage_b", "checkpoints", 0, "diagnostics", "per_layer", 0, "name"], "unknown"),
    ("tokens", ["stage_b", "checkpoints", 0, "training", "tokens"], 7),
    ("static provenance", ["stage_b", "checkpoints", 0, "grid", 2, "static_scale_calibration"], "test"),
    (
        "stage A full budgets",
        ["stage_a_pilot", "rows"],
        [r for r in p["stage_a_pilot"]["rows"] if r["lr"] != 0.001],
    ),
    ("bad raw delta", ["stage_b", "checkpoints", 0, "grid", 0, "test_delta_nll"], 9),
    ("bad restore", ["stage_b", "checkpoints", 0, "checkpoint_sha256_after_ptq"], "0" * 64),
]:
    x = copy.deepcopy(p)
    node = x
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    try:
        b.validate_results(x, canonical=True)
    except ValueError:
        accepted = False
    else:
        accepted = True
    print("VALIDATOR", label, "accepted", accepted)
    assert accepted == (label not in ("bad raw delta", "bad restore"))
```

#### `train.py`

```python
import os, json, torch
from pathlib import Path
from research.quant_ptq import bench_isoloss_ptq as b

D = Path(os.environ["TMPDIR"])
p = json.loads(b.CANONICAL_OUTPUT.read_text())


def setup():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def data():
    return b.q.load_data(Path(p["data"]["directory"]))


if __name__ == "__main__":
    setup()
    cfg = b.h.PRESETS["50m"]
    d = data()
    init = b.h.initial_state(cfg, 2)
    for c in p["stage_b"]["checkpoints"]:
        if c["seed"] != 2:
            continue
        r = c["training"]
        o = c["optimizer"]
        m, t = b.train_checkpoint(
            o, cfg, d.train, torch.device("cuda"), seed=2, steps=r["steps"], lr=r["lr"], init=init
        )
        print(
            o,
            t["checkpoint_sha256"],
            t["train_seconds"],
            t["last_train_loss"],
            r["last_train_loss"],
            flush=True,
        )
        torch.save(m.state_dict(), D / (o + ".pt"))
        del m
```

#### `replay.py`

```python
from train import *
import contextlib
from unittest.mock import patch
from collections import Counter
import numpy as np
from torch.profiler import profile, ProfilerActivity

setup()
dev = torch.device("cuda")
cfg = b.h.PRESETS["50m"]
d = data()
cal = b.q.calibration_windows(d, 128, 512, 20261126)
assert {k: v for k, v in cal.items() if k != "array"} == p["stage_b"]["calibration"]
profile_counts = []
for c in p["stage_b"]["checkpoints"]:
    if c["seed"] != 2:
        continue
    o = c["optimizer"]
    with torch.device(dev):
        m = b.Transformer(cfg)
    m.load_state_dict(torch.load(D / (o + ".pt"), map_location=dev))
    m.eval()
    sha = b.e2e._checkpoint_sha256(m)
    print("START", o, sha, flush=True)
    orig = b.q.evaluate_split
    collect = b.collect_calibration_statistics
    expected = [("baseline", "", "baseline")] + [(r["format"], r["method"], r["mode"]) for r in c["grid"]]
    mats = {}
    moments = []

    def capture(*args):
        H, mom = collect(*args)
        mats.update({k: (lin.weight.detach().clone(), H[k]) for k, lin in b.q.quantized_linears(m).items()})
        torch.save({k: v.cpu() for k, v in H.items()}, D / (o + ".H.pt"))
        return H, mom

    i = 0

    def evaluate(model, tokens, seq, batch, device, windows):
        global i
        label = expected[i // 2]
        i += 1
        if i % 2:
            if label[2].endswith("fakequant"):
                for n, lin in b.q.quantized_linears(m).items():
                    w, h = mats[n]
                    w = w.double()
                    e = w - lin.weight.detach().double()
                    tr = h.diag().mean()
                    wf = w.square().sum()
                    ef = e.square().sum()
                    sig = ((w @ h) * w).sum()
                    err = ((e @ h) * e).sum()
                    moments.append(
                        [
                            *label,
                            n,
                            *torch.stack([ef / wf, sig / (wf * tr), err / (ef * tr), err / sig]).tolist(),
                        ]
                    )
            x = torch.tensor(np.array(tokens[: batch * seq]).reshape(batch, seq).astype("int64"), device=dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(x)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    model(x)
                    torch.cuda.synchronize()
            ops = sum(e.count for e in prof.key_averages() if e.key == "aten::_scaled_mm_v2")
            kernels = Counter(
                e.name
                for e in prof.events()
                if e.device_type == torch.autograd.DeviceType.CUDA
                and "sm120" in e.name.lower()
                and ("gemm" in e.name.lower() or "blockscaled" in e.name.lower())
            )
            profile_counts.append((o, label, ops, dict(kernels)))
            print("PROFILE", o, label, ops, sum(kernels.values()), flush=True)
        return orig(model, tokens, seq, batch, device, windows)

    with (
        patch.object(b.q, "evaluate_split", evaluate),
        patch.object(b, "collect_calibration_statistics", capture),
    ):
        out = b.checkpoint_study(m, cfg, d, dev, cal, p["protocol"], lambda *x: None, o)
    (D / (o + ".moments.json")).write_text(json.dumps(moments))
    (D / (o + ".study.json")).write_text(json.dumps(out, allow_nan=False))
    (D / "profiles.json").write_text(json.dumps(profile_counts))
    print("END", o, out["checkpoint_sha256_after_ptq"], flush=True)
    del out, m
```

#### `spectra.py`

```python
from train import *
import numpy as np
from scipy.linalg import eigh

rows = []
for o in b.OPTIMIZERS:
    state = torch.load(D / (o + ".pt"), map_location="cpu")
    hs = torch.load(D / (o + ".H.pt"), map_location="cpu")
    for n, x in hs.items():
        h = x.numpy()
        w = state[n + ".weight"].numpy().astype(np.float64)
        vals, v = eigh(h, subset_by_index=[len(h) - 8, len(h) - 1])
        e = (w @ v) ** 2
        fro = (w * w).sum()
        signal = ((w @ h) * w).sum()
        sr = w.T @ w if w.shape[0] >= w.shape[1] else w @ w.T
        lmax = eigh(sr, eigvals_only=True, subset_by_index=[len(sr) - 1] * 2)[0]
        rows.append(
            [
                o,
                n,
                np.trace(h) ** 2 / (h * h).sum(),
                vals.sum() / np.trace(h),
                e.sum() / fro,
                (e * vals).sum() / signal,
                fro**0.5,
                fro / lmax,
            ]
        )
    print(
        o,
        "H participation rank, H top8 energy, W energy on H top8, signal from H top8, norm, W stable rank",
        np.mean([r[2:] for r in rows if r[0] == o], axis=0),
        flush=True,
    )
(D / "spectrum.json").write_text(json.dumps(rows))
```

#### `geometry.py`

```python
from train import *
from collections import defaultdict
import numpy as np
from scipy.linalg import eigh

setup()
q = b.q
W = {o: torch.load(D / (o + ".pt"), map_location="cpu") for o in b.OPTIMIZERS}
H = {o: torch.load(D / (o + ".H.pt"), map_location="cpu") for o in b.OPTIMIZERS}
records = []
metrics = []


def trh(x, h):
    return (x.double() @ h * x.double()).sum((-2, -1))


for oi, o in enumerate(b.OPTIMIZERS):
    other = b.OPTIMIZERS[1 - oi]
    for kind in q.LINEAR_KINDS:
        names = [f"blocks.{i}.{kind}" for i in range(12)]
        w = torch.stack([W[o][n + ".weight"] for n in names]).cuda()
        h = torch.stack([H[o][n] for n in names]).cuda()
        alt = torch.stack([H[other][n] for n in names]).cuda()
        wf = w.double().square().sum((-2, -1))

        def align(x, H):
            return trh(x, H) / (x.double().square().sum((-2, -1)) * H.diagonal(dim1=-2, dim2=-1).mean(-1))

        cross, native = align(w, alt).tolist(), align(w, h).tolist()
        for i, n in enumerate(names):
            a = w[i].cpu().numpy().astype("float64")
            small = a @ a.T if a.shape[0] < a.shape[1] else a.T @ a
            largest = eigh(small, eigvals_only=True, subset_by_index=[len(small) - 1] * 2)[0]
            records.append([o, n, wf[i].sqrt().item(), wf[i].item() / largest, cross[i], native[i]])
        for name, fmt in q.FORMATS.items():
            r = q.quantize_rtn(w, fmt)

            def flat(x):
                return x.double().flatten(1).square().sum(1)

            for scale in (0.5, 2.0, 1.7):
                scaled = q.quantize_rtn(w * scale, fmt) / scale
                diff = flat(r - scaled) / wf
                err = flat(w - scaled) / flat(w - r)
                metrics.append(
                    [o, kind, name, "rtn", scale, diff.max().item(), err.min().item(), err.max().item()]
                )
            if name not in ("nvfp4_g16", "int2_asym_g32"):
                continue
            g = q.gptq_quantize(w, h, fmt)[0]
            for scale in (0.5, 2.0):
                gs = q.gptq_quantize(w * scale, h, fmt)[0] / scale
                diff = flat(g - gs) / wf
                err = flat(w - gs) / flat(w - g)
                metrics.append(
                    [o, kind, name, "gptq", scale, diff.max().item(), err.min().item(), err.max().item()]
                )
            ident = torch.eye(w.shape[-1], dtype=torch.float64, device=w.device).expand_as(h)
            assert torch.equal(q.gptq_quantize(w, ident, fmt)[0], r)
            # Same eigenvalues; a fixed channel permutation breaks the covariance's native orientation.
            order = torch.randperm(w.shape[-1], generator=torch.Generator().manual_seed(412)).to(w.device)
            shuffled = h[:, order][:, :, order]
            wrong = q.gptq_quantize(w, shuffled, fmt)[0]
            for tag, z in [("RTN", r), ("GPTQ", g), ("permuted-H GPTQ", wrong)]:
                rel = trh(w - z, h) / trh(w, h)
                ce = trh(w - z, alt) / trh(w, alt)
                metrics.append(
                    [
                        o,
                        kind,
                        name,
                        tag,
                        "output",
                        rel.mean().item(),
                        ce.mean().item(),
                        align(w - z, h).mean().item(),
                    ]
                )
    print("finished", o, flush=True)
(D / "geom.json").write_text(json.dumps(dict(records=records, metrics=metrics), allow_nan=False))
```

#### `prefix.py`

```python
from train import *
from unittest.mock import patch

setup()
cfg = b.h.PRESETS["50m"]
dev = torch.device("cuda")
d = data()
seed = 2
print("flags", torch.are_deterministic_algorithms_enabled(), torch.get_num_threads())


class Enough(Exception):
    pass


for o in b.OPTIMIZERS:
    row = next(c["training"] for c in p["stage_b"]["checkpoints"] if c["optimizer"] == o and c["seed"] == 2)
    for run in range(2):
        init = b.h.initial_state(cfg, 2)
        raw = b.h.train_step
        trace = []
        digests = {}

        def step(m, batch, opt):
            loss = raw(m, batch, opt)
            trace.append(loss.item())
            if len(trace) in (1, 10, 40):
                digests[len(trace)] = b.e2e._checkpoint_sha256(m)
            if len(trace) == 40:
                raise Enough()
            return loss

        try:
            with patch.object(b.h, "train_step", step):
                b.train_checkpoint(o, cfg, d.train, dev, seed=2, steps=row["steps"], lr=row["lr"], init=init)
        except Enough:
            pass
        print(
            o,
            run,
            b.h.state_fingerprint(init),
            b.h.stream_fingerprint(d.train, cfg, 2),
            digests,
            [(i, trace[i - 1], dict(row["train_curve"])[i]) for i in (1, 10, 40)],
            flush=True,
        )
```

#### `assess.py`

```python
from train import *
from collections import Counter
import numpy as np


def read(n):
    return json.loads((D / n).read_text())


pr = read("profiles.json")
assert len(pr) == 52
names = set()
for o, lab, ops, kernels in pr:
    expect = 60 if lab[2] in b.HW_MODES else 0
    assert ops == sum(kernels.values()) == expect
    names.update(kernels)
print("KERNELS", sorted(names))
print("COUNTS", Counter((r[1][2], r[2]) for r in pr))
for o in b.OPTIMIZERS:
    orig = next(c for c in p["stage_b"]["checkpoints"] if c["seed"] == 2 and c["optimizer"] == o)
    c = read(o + ".study.json")
    rows = c["grid"]
    m = read(o + ".moments.json")
    assert c["checkpoint_sha256"] == c["checkpoint_sha256_after_ptq"]
    assert len(m) == 16 * 60
    print("SHA", o, c["checkpoint_sha256"], orig["checkpoint_sha256"])
    print("BASE", o, c["baseline_val"]["nll"], c["baseline_test"]["nll"])
    for r in rows:
        if r["format"] == b.HW_FORMAT and r["mode"].endswith("fakequant"):
            assert (
                r["hw_weights_checked_matrices"] == 60 and r["hw_weights_decode_to_fakequant_weights"] is True
            )
        if not r["mode"].endswith("fakequant"):
            continue
        selected = np.array([x[4:] for x in m if x[:3] == [r["format"], r["method"], r["mode"]]])
        assert selected.shape == (60, 4)
        u, s, e, y = selected.T
        np.testing.assert_allclose(u * e / s, y, rtol=1e-12)
        # The benchmark forms E in float32 before the FP64 contraction. The independent probe subtracts in FP64.
        np.testing.assert_allclose(
            y.mean(), r["calibration_relative_output_error_mean"], rtol=2 * np.finfo(np.float32).eps
        )
        print("MOMENT", o, r["format"], r["method"], *selected.mean(axis=0))
    for a in p["analysis"]["primary"]:
        r = next(r for r in rows if all(r[k] == a[k] for k in ("format", "method", "mode")))
        print("PRIMARY", o, r["format"], r["mode"], r["test_delta_nll"])
    hw, em = [
        next(r for r in rows if r["method"] == b.EMULATED_METHOD and r["mode"] == mode)
        for mode in ("w4a4_hw_dynamic", b.EMULATED_MODE)
    ]
    vs = [hw[s + "_nll"] - em[s + "_nll"] for s in ("val", "test")]
    print("PARITY", o, *vs)
    assert max(map(abs, vs)) <= 1e-3
g = read("geom.json")
for o in b.OPTIMIZERS:
    print("GEOMETRY", o, np.mean([r[2:] for r in g["records"] if r[0] == o], axis=0))
    for f in b.q.FORMATS:
        for method in ("rtn", "gptq"):
            z = [r for r in g["metrics"] if r[0] == o and r[2] == f and r[3] == method and r[4] in (0.5, 2)]
            if z:
                print("SCALE", o, f, method, max(r[5] for r in z), min(r[6] for r in z), max(r[7] for r in z))
        z = [r for r in g["metrics"] if r[0] == o and r[2] == f and r[4] == 1.7]
        print("SCALE1.7", o, f, max(r[5] for r in z), min(r[6] for r in z), max(r[7] for r in z))
        for tag in ("RTN", "GPTQ", "permuted-H GPTQ"):
            z = [r[5:] for r in g["metrics"] if r[0] == o and r[2] == f and r[3] == tag and r[4] == "output"]
            if z:
                print("SHUNT", o, f, tag, *np.mean(z, axis=0))
```

</details>
