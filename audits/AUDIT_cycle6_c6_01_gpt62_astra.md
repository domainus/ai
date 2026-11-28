# C6-01 independent RTX 5080 audit

## Post-Remediation v2 Re-Check — 2026-11-27 MST

**FAIL — PROMOTION HELD.** The fresh synthetic v2 run and the preserved v1 evidence both verify on the RTX 5080. Four of the seven findings still have reproducible certification gaps. A separate provenance check also accepted a change to the reported statistical inference. The examples below use altered review copies. The published measurements passed their byte and evidence checks.

### What still certifies

| Finding | Accepted by the validator | Correction needed for closure |
|---|---|---|
| **H1 · high** | Kept the real opening at 19:03:38 and the real end at 20:10:33. Set the run start to 19:03:35, stages A, B and C to one second each, study duration to 4,017.48 seconds and total duration to 4,018 seconds. The opening now follows only three seconds of apparent work. The three sequential Stage A training records alone require **320.214 seconds**. Full external verification still returned `certified_canonical=True`. | Require each pre-freeze stage to contain at least the sum of its known sequential training times. Apply the corresponding timing bounds to the opening. The genuine log authenticates the opening time but supplies no stage durations. |
| **H2 · medium residual** | Changed every Part A/B seed-1 initialization fingerprint to `0000000000000000`. Sibling agreement passed. CPU regeneration from the producer's preset and seed gave `85659cbc0ffc79c1`. All three original fingerprints matched independent regeneration. | Check one expected initialization fingerprint per preset and seed, then compare the siblings to it. The equivalent stream check already works. |
| **M1 · medium** | Changed only the seed-1 Muon identity activation's maximum group-of-16 crest from **3.24794 to 5**. Also tried stable-rank extrema of **0.5** and **513** separately. Each payload certified with the original analysis and summary. A combined copy with these edits and the false initialization passed external verification. | For 16 values, the maximum absolute value divided by RMS cannot exceed 4. The reported layer average obeys the same upper bound. Stable rank of a nonzero matrix lies between 1 and its smaller dimension, which is at most 512 for the 50m projections. Enforce these bounds with numerical tolerance. |
| **M2 · medium residual** | With the genuine reference, a +0.1-nat change to the seed-1 C5 Hadamard GPTQ static test row failed as required. In a private read-only filesystem view, supplied the same change to the C5 reference file seen by the validator. Recomputed NLL-derived fields, reproduction, analysis, summary and caveats. The unchanged validation code reported **0.0** across the comparisons and certified the result against all real checkpoints and the real log. | Load the reference from the producer's Git blob or verify the reference file against its independent, trusted digest. The current check derives the expected digest and the expected scores from the same checkout file. |
| **Additional provenance finding · medium** | The AST comparison rejected a direct edit of `SIGN_FLIP_MIN_P`. A following augmented assignment that divided the same constant by two passed `_matches_current_summary_code("1b4ea1c…")`. A fresh process using those source bytes reported a two-sided sign-flip floor of **0.125** and certified the recomputed v1 payload, including the changed caveat, against the real external evidence. Three seeds permit eight sign assignments and a minimum two-sided probability of **0.25**. | Bind historical recomputation to the producer's source or a reviewed, byte-pinned compatibility implementation. The current AST extractor considers top-level assignments and function arguments/bodies. It omits later augmented assignments, import bindings and decorators. |

The last two cases require a changed validator checkout or reference read. I confirmed each in a new process with one read-only file override in a private mount namespace. The Git objects and retained evidence came from the actual repository. The H1, H2 and M1 cases used the unmodified validator and its original reference. [Timing](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L1900) · [Paired controls](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L1962) · [Metric bounds](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L2063) · [Reference source](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L1206) · [AST comparison](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L1520)

### Seven-finding disposition and rejection tests

| ID | Disposition | Checks on the remediated revision |
|---|---|---|
| H1 | **Partial** | Rejected training on test, a 1900 opening, an opening before Stage C finished, an opening after test scoring, a timestamp without a time zone, malformed text, missing prior-entry fields and a future prior entry. The direct external verifier rejected a one-second discrepancy from the v1 log. The private v2 log also rejected a different configuration hash. The coordinated stage-duration case above remains. |
| H2 | **Partial** | Rejected one conflicting step-272 snapshot, one conflicting initialization, one conflicting stream, a consistent false stream across all seed-1 siblings and a Part C branch pointing to different weights. On live v2, changing either the model or optimizer snapshot digest of one step-30 synthetic sibling failed. Putting a digest on the fresh Part C optimizer also failed. The v2 producer measures restored weights and optimizer state before continuation. The historical v1 record contains the weight digest only. |
| H3 | **Closed** | For v1, swapped two candidate scores and selected the corresponding new winner across its method rows while preserving the winning NLL and historical freeze digest. The validator rejected it at `HISTORICAL_V1_AWQ_SELECTIONS_SHA256`. For v2, changed the winner, a losing candidate score and the effective-weight digest in three separate trials. Each first failed against the frozen AWQ list, then against the selection digest after rebuilding the list, then against the opening after rehashing. After updating the opening and recomputing all derived fields, JSON-only validation passed and the original log rejected each trial. |
| M1 | **Partial** | The seed-1 Muon emulation row recorded 0 FP4 GEMMs, 60 quantizers and 60 amax kernels on v1, and 0/10/10 on synthetic v2. All four CUPTI fields rejected negative integers, either Boolean and deleted keys. The FP4, quantizer and amax fields rejected wrong positive counts. Each of the five W4A16 module/dispatch fields and its CUPTI field rejected zero, false and deletion. Inverted extrema, out-of-range per-kind means, outlier share above 1, negative training loss, step 10,000, unsorted points and inconsistent step-time totals all failed. The two physical bounds above remain open. |
| M2 | **Partial** | All 42 genuine comparator NLLs reproduced at exactly 0.0 nats of difference. A consistent C6-only shift from 4.649378357089854 to 4.749378357089854 failed at the named comparator. In private checkpoint copies, a one-byte change failed the file hash. A changed tensor with a correct updated file hash failed the tensor hash. The verifier listed an extra `.pt` under `unrecorded_checkpoint_files` and returned a receipt. The reference-file substitution above remains open. |
| M3 | **Closed** | Confirmed the menu against the stored methods. The 50m Muon arm had 15 choices, including two cross-mix composites. The five QAT arms had 3 or 4 choices and lacked both composites. The module and v2 caveats direct training-effect claims to same-method contrasts. Both 120m arms had the historical winning cross-full method. |
| L1 | **Closed** | The module and v2 caveats specify zero pre-freeze test scoring or selection. The canonical caveat names the earlier manifest hash and memory map of `test.bin`. The live synthetic caveat names the earlier generation and hashing of synthetic bytes. Both distinguish the scoring view and explain v1 and v2 AWQ coverage. I checked canonical v2 wording through the formatter using the historical menu and a v2-shaped selection record. |

### Lineage, pins and live receipts

The single-parent edges were verified from Git objects. The reviewed chain is `1c7b938 → 1b4ea1c → 6ad5722 → e2aebdf → bac28fe`. The remediation commit is `bac28fe6f6ff9b7ade0234ab22b051f999794692`. Its entire diff from `e2aebdf` comprises the C6 driver and its test file. The audit restored the earlier report from `1afd9cd354320842bbefdf3713c9ced078f25ebf`.

| Object | Verified SHA-256 |
|---|---|
| Canonical C6 result | `c490a415568244bd1e2489d0004adc025d856e92452cb976c05efc2dcdf66bf9` |
| Rotation module | `8b8f22183c7263c1a90976bb292f579a42b705b0ecd50fb7da898ce1eb7fbb21` |
| Canonical test-access log | `c5ff0f76eb55c1b1d738dfc6acb45b5c949633ba7dc67604cfb27ec18dc89307` |

**Historical v1.** Called both `verify_external_evidence` and `validate_results(..., canonical=True, verify_external=True)` against the retained C6 checkpoints, C5 checkpoints and C6 evidence directory under the main checkout. Both file and tensor digests matched for 24 C6 and 3 C5 checkpoints, totaling **9,079,770,735 bytes**. The inventory contained zero extra checkpoints and one matching log entry. The receipt reported `certified_canonical=True` and `awq_in_contemporaneous_freeze=False`. The C5 reference on this host also matched the producer's Git blob, SHA-256 `e0e435ef2ea8598df22bace95ab76fd10c43b2ff394bf635ab8d1548d8bcad82`.

**Fresh v2.** Ran the driver with `--synthetic-shakedown` on the NVIDIA GeForce RTX 5080, compute capability 12.0. An independent reload passed `validate_results(..., canonical=False, verify_external=True)`. It checked **27 synthetic checkpoints, 15,241,683 bytes**, zero extra checkpoints and one private log entry. The receipt reported `shakedown=True` and `awq_in_contemporaneous_freeze=True`. The pre-test digest `f742f1eec739179f1d57a7ef054496a9f0e2b8cdb0943ee5463fe543285674fc` covered 36 AWQ choices. The synthetic ledger counted 549 validation and 477 test scorings. The standard and cross-mix parity sweeps passed 87 and 48 cases respectively. The synthetic run recorded a dirty worktree because the requested v1 report had already been restored. The driver and imported code matched the review revision. The canonical log retained its original digest.

**Live regression suite.** All **464 tests passed, 0 failed, 0 skipped**, in **759.63 seconds** on the RTX 5080. Pytest reported 10 warnings, comprising seven Triton AST deprecations, two empty CUDA graphs in capture-refusal tests and one CUDA synchronization-debug warning. The requested command was

```sh
PYTHONPATH=. /home/ryan/Documents/projects/ai/.venv/bin/pytest \
  tests/test_c6_01_wide_rotated_qat.py tests/test_nvfp4_inference.py \
  tests/test_c5_03_nvfp4_120m_fusion.py tests/test_c5_02_isoloss_ptq_mitigation.py -q
```

## Original v1 audit

**FAIL — PROMOTION HELD.** The measured result supports keeping the existing training default. The 120m dual cooldown cost 4.31× as much per step. It improved full-width Hadamard GPTQ at equal tokens, while the best deployment comparison did not reach the predeclared significance threshold. The canonical validator certifies records that contradict the experiment's split rules and its paired controls. The freeze digest also omits every AWQ alpha. Correct the certification before promoting it as evidence for another run. The reviewed result has a separate, correct byte-level pin.

## Ranked findings and merge decision

| Severity | Finding on the reviewed revision | Remediation |
|---|---|---|
| **HIGH** | Canonical certification accepted a payload declaring `test` as the training split, with all published summaries and caveats unchanged. It also accepted a test-access timestamp in 1900. The real ledger and log agree with the producer's train/validation selection order. | Validate the complete split-role map and the opening timestamp against the run interval. Bind the opening to the freeze and compare the external log when it is available. |
| **HIGH** | Changing just one step-272 branch hash or one same-seed batch-stream fingerprint still returned `certified_canonical=True`. The 50m comparisons depend on common weights and batches. Both contradictions are detectable from sibling records alone. | Require common non-null branch hashes for all step-272 arms of a seed. Check same-seed initialization and stream fingerprints, format and expected values. Record a digest for the optimizer snapshot if certifying its contents. |
| **HIGH** | A different winning AWQ alpha passed canonical certification with the original freeze SHA-256. The freeze covers the wide family, transfer arm and per-arm method names. The per-checkpoint AWQ choices and candidate scores live outside that object. The producer correctly retains the selected candidate for test scoring. | Include each selected alpha, its validation scores and the selected effective-weight digest in the pre-test freeze record. Record its full digest in the opening log. Label this experiment's historical coverage precisely. |
| **MEDIUM** | Canonical certification accepted negative CUPTI counts on an emulation row, an activation maximum below its minimum and a training curve with a negative loss at step 10,000. The FP4 hardware dispatch and kernel counts do have effective rejection tests. | Check counter types and ranges, emulation invariants, summary bounds, train-curve step order, range and losses. |
| **MEDIUM** | The C5-03 reference itself is pinned, but certification accepted a 0.1-nat discrepancy in a remeasured comparator after recomputing the derived fields. New C6 file and tensor digests also pass on hex format alone. They require the external files to verify. | For a gate that asserts exact reproduction, require zero across the comparator rows. Pair offline validation with the committed result pin and an evidence-directory digest check. |
| **MEDIUM** | The 50m Muon arm could select the winning cross-full butterfly GPTQ method. The QAT arms could not. Their best-vs-best contrast measures the available pipelines and cannot isolate the effect of QAT. The 120m token-matched comparison offers the same selected method to both arms. | Use same-method contrasts for claims about training. Include the winning PTQ method on both sides in a future balanced 50m study. |
| **LOW** | The real-data manifest verifier hashes and memory-maps `test.bin` before the experimental freeze. `splits_opened_before_freeze` and the external log count the scoring interface. Earlier cycles also evaluated the same test shard. | Describe the guarantee as no pre-freeze test **scoring or selection** in this run. |

Merge the validator and regression-test fixes before lifting the promotion hold. Preserve the original artifact, producer commit and one-opening log. Most checks can be applied to these retained measurements without training or rescoring. A digest generated today cannot establish a contemporaneous AWQ freeze. New runs should write that digest before test access.

The relevant checks are in the [selection digest](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L297), [freeze and test opening](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L927), [split and training validators](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L1646), [row validator](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L1723) and [manifest verifier](../research/polar_opt/real_text_data.py#L164).

## Provenance and split evidence

The four additions relative to `origin/main` are the driver, rotation module, canonical JSON and test file. No base file changed. The single-parent chain was:

```text
1c7b9385fb23a81c043677ce823259c6bd2eb32a  origin/main
1b4ea1c17ecc3f6b0e1232693a13ccca25ad4d5d  producer
6ad572285edc07725f4b9ef10b73fa827c5e171d  result and test pin
e2aebdfa29ccd13bb5dbf9ab522f22150e49c0e5  reviewed tip
```

| Object | Audit SHA-256 |
|---|---|
| Canonical result | `c490a415568244bd1e2489d0004adc025d856e92452cb976c05efc2dcdf66bf9` |
| Rotation module | `8b8f22183c7263c1a90976bb292f579a42b705b0ecd50fb7da898ce1eb7fbb21` |
| Benchmark driver | `3c33d9a092116823b95a5a8e52e91c8d447effcec8c5f185beda8cf3e4eaddd1` |
| Test file at reviewed tip | `ee5ef2bbfa9807e1432d4922411a726140a4717b4b5dd989df872355b12fe494` |
| Producer stdout | `a5a811a27a2cc1ad7348c8a41250d48fe7acbb868f4f3d55e05dd5e1d475beef` |
| Producer stderr | `4b9b902cc2760490b74f876d96ff37d08bb61cbc6004c86a6a35800413b07221` |
| External test-access log | `c5ff0f76eb55c1b1d738dfc6acb45b5c949633ba7dc67604cfb27ec18dc89307` |

I rebuilt the code digest from the 19 named Git blobs at the producer revision and recomputed the configuration digest from the schema, code, commit, numerics-related environment fields, protocol and shard digests. They give `86ab792cc21c469ee45a217302d3c6ec48533f6f85cfda3bd2a557a681598310` and `7123af71d2e91500`. The driver and rotation source bytes match the producer blobs. The tip changes tests only. A producer-SHA-only mutation now expects rejection at the earlier access-log guard. Two consistent producer substitutions exercise the import-closure guard. A separate code-hash mutation exercises the blob check. The relabeling probe now exercises both explicit canonical mode and inferred mode.

I streamed every checkpoint file and independently hashed its sorted state-dict names, FP32 shapes and tensor bytes. All 24 C6 files (7,596,634,404 bytes) and all three separately stored C5 files (1,483,136,331 bytes) matched both digests. The recorded files accounted for every checkpoint in the two directories.

I recomputed the seeded 128 nonoverlapping train windows, all validation argmins and all 36 AWQ selections. Counting the raw grid and the extra AWQ trials gave 549 validation and 477 test evaluations. The recorded split order before the freeze was train, validation. Producer stdout line 197 records the freeze. All 27 per-checkpoint test summaries follow it. The single external log entry matches the embedded entry and names the producer, configuration and opening at 19:03:38 MST on November 27. The sorted-JSON selection digest is `0dbdf496844ebb2ce3c95f3f25dec10012b47fe42d29b666630f8add2e1f7232`. The access log contains no earlier entry. These are application-level records. The audit's later reads used the published choices and left that log intact.

## Math, optimizer and calibration

With row-vector map Q, applying the same Q to inputs and weights preserves the unquantized product. Quantizing the two transformed operands introduces error. The hardware emulation shares the quantizer and tests the GEMM separately. The eager operand checks provide the independent check on the quantizer.

The full transform uses orthonormal DCT-IV and normalized Sylvester Hadamard factors. The fitted butterfly uses 2 × 2 orthogonal reflections and reverses stage order for the inverse. I measured the full FP64 Gram matrix, inverse round trip and product residual on each of the three requested wide families at all four widths, including newly fitted butterfly angles. Absolute maxima were 2.48e-14 for the Gram residual, 7.99e-15 for inversion and 4.41e-13 for the product. Maximum dense-QAT versus PTQ map difference on the two trainable wide families was 2.33e-15.

| Width | Groups of 16 | Full factors (inner block, power of two, odd DCT size) |
|---:|---:|---|
| 512 | 32 | 128, 4, 1 |
| 768 | 48 | 128, 2, 3 |
| 1408 | 88 | 128, 1, 11 |
| 2048 | 128 | 128, 16, 1 |

The STE differentiates both rotated operands. Its detached term supplies the quantized forward value. For upstream gradient G, the input gradient is `(G Wq) Qᵀ` and the weight gradient is `(Gᵀ Xq) Qᵀ`. RTN uses one dynamic tensor scale per flattened microbatch for activations, one per linear for weights, and 16-value block scales. AWQ applies reciprocal input scaling and corresponding weight and Hessian scaling. Static scales use the mapped BF16 calibration activations. QAT uses a BF16 autocast GEMM. GPTQ is a separate final weight conversion. The dual arm accumulates half the BF16 CE gradient and half the W4A4 CE gradient on the same inputs and takes one optimizer step. Its logged training loss is the W4A4 component.

At 50m, the snapshot precedes step index 272. Cooldown restores weights, Muon momentum and auxiliary AdamW state, advances the same seed's stream and runs indices 272–319. The 32-step warmup reaches 1.0 at index 31. The cosine multiplier is 0.1602885683 at 272 and 0.1000267727 at 319. I regenerated all three initialization and stream fingerprints and checked the common stored branch hash within each seed. At 120m, both continuations load the same pinned weights and start fresh optimizer state. Both use indices 640–735, 3,145,728 extra tokens and fixed learning rates 0.002 and 0.0001. This control isolates QAT from the extra tokens and optimizer reset. It does not test continuing the old optimizer state.

I replayed seed 1’s first 120m optimizer update after the fork, using the pinned 640-step weights, the advanced batch stream and a new optimizer for each arm. At recorded step 641, BF16 CE was 4.2978901863098145 and the QAT arm’s logged W4A4 CE was 4.328338623046875. Both matched exactly. This checks the branch entry. The audit used the retained endpoints for subsequent scoring and did not retrain the full trajectories.

## Independent statistical checks

I recomputed the per-seed statistics from the grid without calling the producer's analysis function. All 918 mean/SD/value triples, 94 activation shares and 140 contrast endpoints agreed. For each endpoint I checked the paired differences, standard error, t statistic, two-sided SciPy t probability, 95% interval and all eight sign assignments. I recomputed the Holm step-down adjustments for both probability families within AB or C, split and endpoint. There are 24 AB and 11 C contrasts. With three seeds, df = 2 and the minimum two-sided sign-flip p is 0.25. A nonsignificant comparison cannot establish equivalence.

| Published question, dynamic W4A4 test NLL | Mean paired difference, nats | Holm t p |
|---|---:|---:|
| 50m best wide PTQ vs block-16 Hadamard GPTQ | −0.00173430 | 0.02245 |
| 50m best wide PTQ vs AWQ butterfly | −0.0002157 | 0.3606 |
| 50m wide dual cooldown vs Muon, each arm's best method | +0.0012259 | 0.1301 |
| 50m wide dual cooldown vs Muon, same full-width RTN | −0.0045916 | 0.0109 |
| 120m QAT vs token-matched BF16 continuation, same winning cross-full method | −0.00071244 | 0.15188 |
| 120m QAT vs token-matched BF16 continuation, same full-width Hadamard GPTQ | −0.00125238 | 0.03480 |

For the 50m Muon probe, the activation share of the test NLL penalty ranged from 0.7874 to 0.8253 for GPTQ methods. RTN shares were 0.5169–0.5366. Identity, block-16 Hadamard and full-width Hadamard gave mean excess kurtosis 49.51078, 3.80099 and 0.27897. Relative squared rounding error was 0.008487, 0.009384 and 0.009083, respectively. Lower kurtosis did not reduce every error metric monotonically. I also remeasured those three transforms live on all three Muon checkpoints using independent activation hooks, per-layer FP64 moment sums and RTN error accumulators. The maximum absolute difference over the 18 per-seed kurtosis/error measurements was 1.43e-14.

| QAT arm | Mean of paired median step-time ratios | Mean tokens/s |
|---|---:|---:|
| 50m identity cooldown | 2.21655× | 51,037 |
| 50m identity full | 2.39482× | 46,264 |
| 50m block-16 cooldown | 2.96185× | 39,577 |
| 50m wide cooldown | 3.43908× | 34,100 |
| 50m wide dual cooldown | 3.74783× | 30,227 |
| 120m wide dual cooldown | 4.30899× | 15,644 |

These ratios use each seed's stored wall times from the matching step interval. The 120m control averaged 66,015 tokens/s. The synthetic development study selected the 0.5 dual weight and omitted rotated full-duration training. The driver comments document that process and the decision to drop full-vocabulary distillation on the 16 GB GPU. Performance of the omitted arms on real text remains unknown. Ten of the twelve **Part A probe** AWQ choices reached the 0.5 grid boundary. The three 120m preserved comparator choices were 0.25 in both C5 and C6. Wide inference adds unfused FP32 matrix products. No inference-latency or inference-throughput claim follows from these training timings.

## Validator stress tests

I applied the mutations to in-memory copies of the committed payload and called the canonical validator. It accepted the original. The separate committed-file SHA-256 test distinguishes every changed byte. The high findings concern the rules for certification when a new artifact is produced.

| Outcome | Independent cases |
|---|---|
| Rejected | Dirty producer, wrong code digest, wrong access configuration, wrong freeze digest, wrong validation count, early scoring-map declaration, wrong preserved checkpoint digest, 59 FP4 dispatches, 59 FP4 CUDA kernels, modified C5 reference, edited caveat, edited Holm result, NaN, Infinity, invalid calibration start. |
| Rejected | Canonical destination through a symlink alias, parent traversal, alternate filename. The shipped temporary-directory tests also exercise symlinked directory components and the final destination. |
| Certified | Test declared as training source, impossible test-opening date, changed new checkpoint tensor digest, changed new checkpoint file digest, inconsistent sibling branch hash, inconsistent batch-stream fingerprint. |
| Certified | Impossible training-curve point, inverted activation extrema, negative emulation CUPTI counts, different AWQ winner with identical freeze digest, 0.1-nat C5 comparator discrepancy with consistent derived arithmetic. |

The unchanged analysis and original source agree on 42 published C5 comparator values across three seeds, both splits, BF16 and three modes for each of two PTQ methods. Their maximum absolute difference is exactly 0.0. The canonical C6 grid records 282 hardware rows including 114 wide rows, 8,460/8,460 exact weight-matrix decodes and 54 hardware/emulation NLL comparisons with maximum absolute difference 0.000242313277691 nats.

The independent log-to-artifact check matched all 27 validation baselines, 423 validation grid deltas, 27 test baselines and 141 dynamic test NLLs at the log's four-decimal precision. The C5 result file had SHA-256 `e0e435ef2ea8598df22bace95ab76fd10c43b2ff394bf635ab8d1548d8bcad82`.

## Regression suite

The requested four-module command ran on the RTX 5080 in the audit worktree, using the shared virtual environment, `PYTHONPATH=.` and private temporary/cache directories. **452 passed, 0 failed, 0 skipped in 871.01 seconds.** The 84 warnings comprised 81 Triton AST deprecations, one CUDA synchronization-debug warning and two empty-graph warnings from capture-refusal tests.

```sh
PYTHONPATH=. /home/ryan/Documents/projects/ai/.venv/bin/pytest \
  tests/test_c6_01_wide_rotated_qat.py tests/test_nvfp4_inference.py \
  tests/test_c5_03_nvfp4_120m_fusion.py tests/test_c5_02_isoloss_ptq_mitigation.py -q
```

## Live GPU replay

NVIDIA GeForce RTX 5080 (sm_120), driver 615.78.08, PyTorch 2.14.1+cu130, Triton 3.8.0. I loaded the saved weights, rebuilt Hessians and quantized candidates from the seeded training calibration windows, and used the frozen alphas. A separate scoring loop ran BF16 autocast, summed cross-entropy and accumulated the sums in FP64 over 512-token windows. Each row below covers 2,001,920 validation tokens and 1,999,872 test tokens. The GPU suite and model replay ran in separate processes.

Of 60 full-split NLL measurements, 58 have a published value. The independent loop divided the FP64 sum by the token count on the host. It matched 49 exactly and differed by one ULP (8.881784197001252e-16 nats) on nine validation NLLs. I rebuilt those candidates and rescored the affected splits with the producer’s GPU normalization. All nine matched exactly. The tables use the repeated measurements for those nine entries. Across the 58 published values, including all 42 C5-03 comparators, the final maximum absolute difference was **0.0 nats**. The extra 50m cross-butterfly emulation provides an additional direct comparison. All 720/720 independently rebuilt weight matrices matched the FP4 decoder in their int32 representation. All 12 effective-weight digests matched the published rows.

All 18 profiled hardware forwards had 60 scaled-MM dispatches, 60 CUPTI FP4 GEMMs, 60 quantization kernels and zero BF16 block-linear slots. Dynamic modes launched 60 amax kernels. Static modes launched zero. The unquantized head and wide FP32 transforms still execute. Across 6 paired full-split hardware/emulation measurements, the largest absolute NLL difference was 0.00012960668349926152 nats (limit 0.001).

Both live operand sweeps passed bitwise, with 279/279 standard cases and 48/48 cross-mix cases. In eight extra single-layer trials (block-64 and full Hadamard at all four widths), the maximum relative Frobenius difference between hardware and FP32 emulation was 4.27241e-5. Relative differences between the BF16-autocast training surrogate and native FP4 were 0.00305–0.00363. The training proxy includes BF16 operand rounding and a different GEMM implementation. The 0.001-nat end-to-end parity criterion applies to native hardware versus FP32 emulation.

### Seed 1 training-arm replay

Method names below identify the final weight conversion. The 120m QAT arm uses the dual cooldown transferred from 50m.

| Size and training arm | Method | Execution | Validation NLL | Test NLL |
|---|---|---|---:|---:|
| 50m `muon_ss3_54` | `cross_full_butterfly_b16_gptq` | FP4 | 5.368441659288333 | 5.404208061454605 |
| 50m `muon_ss3_54` | `cross_full_butterfly_b16_gptq` | FP32 emulation | 5.368498444618167 | 5.404119734756774 |
| 50m `qat_wide_dual_cooldown` | `awq_w4a4_adaptive_butterfly_b16_gptq` | FP4 | 5.369585754316481 | 5.405069914220604 |
| 50m `qat_wide_dual_cooldown` | `kron_hadamard_full_gptq` | FP4 | 5.369925471157064 | 5.405493111227087 |
| 50m `qat_wide_dual_cooldown` | `kron_hadamard_full_gptq` | FP32 emulation | 5.369971075814093 | 5.405523470103649 |
| 120m `qat_cooldown` | `cross_full_butterfly_b16_gptq` | FP4 | 4.611547521313133 | 4.629276732206955 |
| 120m `qat_cooldown` | `kron_hadamard_full_gptq` | FP4 | 4.611792732199744 | 4.6292438040741635 |
| 120m `qat_cooldown` | `kron_hadamard_full_gptq` | FP32 emulation | 4.611687796804911 | 4.629114197390664 |
| 120m `bf16_continue` | `cross_full_butterfly_b16_gptq` | FP4 | 4.612423731908773 | 4.629694168590852 |

### All preserved C5-03 comparators

Each cell gives validation / test NLL. All 42 values below were measured again from the three pinned C5 checkpoints.

| Method | Execution | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| bf16 | bf16 | 4.615477041454266 / 4.633469397021878 | 4.613257640341053 / 4.62647873770562 | 4.60027274890324 / 4.612908237113503 |
| hadamard_b16_gptq | w4a16_fakequant | 4.618506357432021 / 4.636576955280606 | 4.616428486953306 / 4.629824566096449 | 4.60355430515221 / 4.616137575382949 |
| hadamard_b16_gptq | w4a4_hw_dynamic | 4.631390991113374 / 4.649785047668832 | 4.6298605579854275 / 4.643124870440927 | 4.616818412917349 / 4.629782787543715 |
| hadamard_b16_gptq | w4a4_hw_static | 4.631179945182312 / 4.649378357089854 | 4.6298477592370695 / 4.6433019244969955 | 4.616743461433273 / 4.629705426635586 |
| awq_w4a4_adaptive_butterfly_b16_gptq | w4a16_fakequant | 4.6185370672084485 / 4.636500559766294 | 4.6164164184609335 / 4.629670131896254 | 4.60342205711033 / 4.616017578933645 |
| awq_w4a4_adaptive_butterfly_b16_gptq | w4a4_hw_dynamic | 4.6299635167621895 / 4.648274206346081 | 4.6283467714743844 / 4.641667692220583 | 4.61551200593524 / 4.628540260023907 |
| awq_w4a4_adaptive_butterfly_b16_gptq | w4a4_hw_static | 4.629919431642498 / 4.648153039167553 | 4.628405688546807 / 4.641933831079642 | 4.6154766282766975 / 4.628312775982507 |
