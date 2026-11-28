# C6-01 independent RTX 5080 audit

## Post-Remediation v4 Signoff

**PASS — HOLD LIFTED** for `fc17583db3f636c0d90d0a2e69677fa3b07708e3` on the NVIDIA GeForce RTX 5080 (sm_120). Both genuine results passed external verification. The validator rejected all 18 timing and geometry challenges on preserved v1 and all 19 on fresh v2 at the expected checks.

| v3 finding | Live verification on v4 |
|---|---|
| H1 · Phase-one timing | With the real log entry and end time retained, v1 rejected an opening at 8,845 s against 9,347.227 s of recorded work. Fresh v2 rejected 98 s against 118.839 s. A separate v2 copy with timing components scaled to replay the earlier 103 s / 124.620 s intervals rejected the exact 21.620 s shortfall. All three failed the unit phase-one bound. Both schemas also rejected a single unit omitting non-AWQ preparation, an AWQ search shorter than its winner’s preparation, negative/null/absent W4A16 quantization time, a hardware row carrying quantization time, negative/null GPTQ damping, and damping on RTN or hardware rows. |
| M1 · Activation statistics, geometry and fit | Both rejected an identity crest maximum of 3.99 for the seed-1 Muon unit. The measured share capped it at 3.665792 on v1 and 3.365072 on v2. Each rejected independent upper-bound violations for the down projection and the median, and a crest below three times the outlier share. Each also rejected signal alignment one above the maximum dimension, a false relative error reduction, and an improved-layer count one above the layer count. |
| Producer trust | On v1, the external log check returned before the historical producer ran. A changed opening stopped at the log on both schemas. The namespace loader and the public v1 validator independently rejected `6ad5722`, which has the same driver bytes as the approved commit. A one-byte change to the approved driver failed its SHA-256 pin before execution. The approved commit `1b4ea1c` matched the current summary code and its driver hashed to `3c33d9a092116823b95a5a8e52e91c8d447effcec8c5f185beda8cf3e4eaddd1`. Fresh v2 completed with both historical producer functions set to raise on any call. |
| H2 · CUDA RNG | In each validation process, a cold call to `_expected_init_fingerprint("shakedown", 1)` returned `e378fc17b2f3e94d`. Both the CPU and RTX 5080 RNG state tensors matched their seed-617 snapshots byte for byte. |

| External verification receipt | Preserved canonical v1 | Fresh CLI shakedown v2 |
|---|---|---|
| Checkpoint file and tensor hashes | 24 C6 + 3 retained C5, 9,079,770,735 bytes | 27 synthetic, 15,241,683 bytes |
| External evidence | One matching log entry, zero extra checkpoints. All 42 C5 comparator NLLs reproduced exactly. AWQ selection pin is retrospective. | One matching log entry, zero extra checkpoints. The opening binds the AWQ choices to selection digest `f742f1eec739179f1d57a7ef054496a9f0e2b8cdb0943ee5463fe543285674fc`. |
| Parity cases recorded (kernel / cross-mix) | 279 / 48 passed | 87 / 48 passed |
| Result SHA-256 | `c490a415568244bd1e2489d0004adc025d856e92452cb976c05efc2dcdf66bf9` | `ce1a05f5d41760d921ef4c7f08335b8306c14e9c98b45800c931e0490e2b7bee` |

The v2 CLI recorded a clean checkout. A separate interpreter reloaded its JSON with `canonical=False, verify_external=True`. The v1 check used `canonical=True, verify_external=True`, the retained C6 checkpoint and log directories, and `default_baseline_dir()`.

The four requested pytest modules completed on this GPU with **467 passed, 0 failed, 0 skipped** in **694.53 s**. Pytest reported 10 warnings (seven Triton deprecations, two empty CUDA graphs and one synchronization-debug warning).

All six single-parent edges matched the requested chain from `origin/main` at `1c7b938` to `fc17583`. The final remediation diff touched only the C6 driver and its tests. The canonical result, rotation module and canonical access log matched all three v3 SHA-256 pins before and after verification. The prior report below is preserved from `6f9dfd3`.

## Post-Remediation v3 Re-Check — 2026-11-27 MST

**FAIL — PROMOTION HELD.** The genuine canonical result and a fresh RTX 5080 shakedown both passed external verification. All five v2 examples now fail. Two independent consistency checks still fail to protect certification. The v1 validator accepted a timeline that opened test scoring at least **502.227 seconds too early** and, in a separate trial, a physically impossible activation statistic. Both contradictions also passed on fresh v2 data. These findings concern what the validator accepts. The pinned measurement file passed.

### Findings on ef6edfa

| Finding | Disposition | Evidence |
|---|---|---|
| H1 · split and timing | **Open, high** | The three-second opening now fails. The phase-one lower bound still omits recorded, sequential quantization time. The real external log authenticated the opening in the contradictory timeline. |
| H2 · paired controls | **Closed for fingerprint verification** | The validator rejected all-zero seed-1 fingerprints across all six Part A/B arms on both schemas. It also rejected a false fingerprint on the synthetic Part C baseline. The canonical Part C baseline is a retained C5 checkpoint. |
| H3 · AWQ freeze | **Closed with the historical v1 limitation** | v1 uses the retrospective AWQ digest. Fresh v2 included all 36 AWQ choices in the digest written to its private opening log. The suite exercises changed winners, weights, freeze digests and external log mismatches. |
| M1 · physical bounds | **Open, medium** | The three reported single-metric violations now fail. The validator still checks crest and outlier share independently even though the two measurements constrain each other. |
| M2 · C5 reference | **Closed** | Both the checkout file and producer blob matched the trusted SHA-256. A shifted on-disk reference failed its pin even with exact internal reproduction and consistent derived fields. |
| M3 · unequal menus | **Closed** | The 50m Muon arm had 15 methods. The five QAT arms had 3 or 4. The fresh v2 caveat identifies both cross-mix methods and directs training-effect claims to comparisons using the same method. |
| L1 · held-out wording | **Closed** | The fresh caveat distinguishes hashing synthetic test data from scoring it. The canonical v2 formatter documents the earlier manifest read. The ledger reported 549 validation scorings before freeze and 477 subsequent test scorings. |
| AST / producer provenance | **Historical 0.125 finding closed for the pinned producer. Separate trust issue identified by source review.** | All seven source fixtures failed the public validator. Independent recomputation from the genuine producer returned 0.25. The new loader also executes a driver selected by the payload before checking external evidence. |

### Two remaining numerical contradictions

**H1.** The producer times candidate construction before entering the validation scoring loop. For each non-AWQ method, its W4A16 row records this duration once in `quantize_seconds`. That interval is disjoint from row validation times and AWQ search times. The validator sums only the latter two. On seed 1’s 50m Muon unit, those sum to 490.701 seconds. The non-AWQ candidates took another 40.470 seconds. The published phase took 557.232 seconds. A copy declaring just 490.701 seconds passed full external verification, despite a lower bound of 531.171 seconds. On fresh v2 the corresponding accepted duration was 6.906 seconds against a 9.260-second lower bound. [Candidate timer](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L678) · [Producer sequencing](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L854) · [Unit timing check](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L2376)

The effect reaches the opening itself. I kept the original training, quantization, AWQ and scoring measurements, the real opening, and the real end. I recomputed the unit, stage and run durations using the validator’s lower bounds. Both payloads passed with their respective original checkpoints and logs.

| Result | Declared start to opening | Minimum from sequential recorded work | Shortfall |
|---|---:|---:|---:|
| Canonical v1 | 8,845.000 s | 9,347.227 s | 502.227 s |
| Fresh v2 | 103.000 s | 124.620 s | 21.620 s |

Count each non-AWQ candidate’s recorded preparation once in the phase-one lower bound. For AWQ, the search timer already includes candidate preparation. Its duration must cover the recorded preparation of the winning candidate. Enforce finite, nonnegative candidate times where the producer records them. On the two genuine results, all 27 unit durations per schema exceeded the corrected lower bound. The smallest margins were 12.761 seconds on v1 and 0.062 seconds on v2. All 36 AWQ searches per schema covered their winning candidate’s preparation.

**M1.** Let `f` be the fraction of groups whose crest exceeds 3. For groups of 16, every crest is at most 4. A layer’s mean crest is therefore at most `3 × (1 − f) + 4 × f = 3 + f`. Across the same layers, the largest mean crest is at most `3 + max(f)`. This bound also holds for groups with zero RMS. I changed only the maximum layer mean crest for the seed-1 identity rotation to **3.99**. Its genuine v1 maximum was 3.247940, and the largest possible value given the recorded shares was **3.665792**. Fresh v2 had a maximum possible value of **3.363200**. Both validators accepted 3.99 with external verification. All 21 genuine seed/family summaries per schema obeyed this bound. The narrowest margins were 0.418 on v1 and 0.538 on v2. Check the bound for each probe family with a numerical tolerance. [Crest and outlier measurement](../research/quant_ptq/rotated_qat.py#L396) · [Summary checks](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L2367)

### Negative checks and provenance

- **H1.** On both schemas, the validator rejected three one-second stages, 4,017.48 seconds of study time, 4,018 seconds total, and an opening three seconds after the declared start. Each of four separate stage reductions to 0.01 seconds also failed. Reducing either unit phase to 0.0001 seconds failed its own check. Genuine Stage A training alone took 320.214 seconds on v1 and 20.029 seconds on fresh v2.
- **H2.** Sibling agreement on sixteen zeroes failed independent initial-weight regeneration. The genuine seed-1 fingerprints were `85659cbc0ffc79c1` for 50m and `e378fc17b2f3e94d` for the shakedown. A separate low-severity side effect appeared on a first call to the new helper. CPU RNG state was restored, but the CUDA generator changed from seed 617 to seed 1. The helper saves CPU state and calls an initializer that seeds every device. This matters to callers that continue stochastic GPU work in the same process. [Fingerprint helper](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L2118) · [Initializer](../research/polar_opt/harness_tier2_real.py#L114)
- **M1.** Both schemas rejected a group-of-16 crest maximum of 5 and a stable-rank minimum of 0.5. v1 rejected a stable-rank maximum of 513 at width 512. Fresh v2 rejected 65 at width 64.
- **M2.** A fresh process saw a single read-only C5 reference override. The seed-1 static Hadamard GPTQ test NLL changed from 4.649378357089854 to 4.749378357089854. I recomputed the row’s perplexity and deltas, the C6 reference, reproduction, analysis, summary and caveats. Maximum reproduction difference was still zero. With the real evidence directories supplied and external verification enabled, the validator rejected the shifted SHA-256 `a769be9a960a50a0220ad47cb7c26e061e06115d3c4aeb60190de11c1df894ac`. The checkout and genuine producer blob each independently hashed to `e0e435ef2ea8598df22bace95ab76fd10c43b2ff394bf635ab8d1548d8bcad82`. [Reference validation](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L1249)
- **AST and producer recomputation.** Each source variant ran in a fresh Python process with one read-only driver override, a v1 payload recomputed at 0.125, and the real external evidence paths. The augmented division, identity decorator on `analyze`, conditional assignment, import alias, function declaring the global, annotated assignment and changed return annotation each failed the summary-code check. The loaded floor was 0.125 in the three assignment cases and 0.25 in the other four. These calls stopped before checkpoint reads. Separately, the genuine producer blob recomputed `analysis.sign_flip_min_p` as 0.25 and disagreed with the 0.125 caveat. Its summary matched because that summary does not expose the floor. [Source comparison](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L1684) · [Producer comparisons](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L2631)

**Producer trust.** The historical review used the independently identified producer `1b4ea1c17ecc3f6b0e1232693a13ccca25ad4d5d`. In the public validation path, the payload supplies `meta.git_sha`. The validator checks the local commit, schema, recorded code hash and selected definitions. It then executes the whole Git driver as Python in the current process. The external log comparison happens later. A private Python module namespace shares the process permissions. If payloads may name unreviewed local commits, require an independently approved producer digest before executing a blob, or use a reviewed local recomputation implementation. This is a source-review finding. The live checks used only the real producer and established rejection of the incorrect floor. [Git object check](../research/quant_ptq/bench_c5_03_nvfp4_120m_fusion.py#L1765) · [Loader](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L1670) · [Verification order](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L2594)

### Lineage, immutable files and GPU receipts

I verified all five single-parent edges from Git objects in this order, `1c7b938 → 1b4ea1c → 6ad5722 → e2aebdf → bac28fe → ef6edfa`. The last commit matched `origin/exp/c6-01-remediate-v3` at `ef6edfaeeb4c8acfb6df7342e7b0f67c34a80b80`. Its diff from `bac28fe` contains only the C6 driver and its test file. The prior audit text below comes from `3a3c3f1b822a8428bdf5ebe39c1384e4e8b82796`.

| Object | Verified SHA-256 |
|---|---|
| Canonical C6 result | `c490a415568244bd1e2489d0004adc025d856e92452cb976c05efc2dcdf66bf9` |
| Rotation module | `8b8f22183c7263c1a90976bb292f579a42b705b0ecd50fb7da898ce1eb7fbb21` |
| Canonical external test-access log | `c5ff0f76eb55c1b1d738dfc6acb45b5c949633ba7dc67604cfb27ec18dc89307` |

| Live check on NVIDIA GeForce RTX 5080 (sm_120) | Result |
|---|---|
| Preserved v1, canonical and external | 24/24 C6 and 3/3 C5 checkpoints verified by file and tensor digest, 9,079,770,735 bytes, zero extra checkpoints, one matching log entry. All 42 C5 comparator rows reproduced at 0.0 nats difference. `awq_in_contemporaneous_freeze=False`. |
| Fresh CLI `--synthetic-shakedown`, independently reloaded with external verification | 27/27 synthetic checkpoints, 15,241,683 bytes, zero extra checkpoints, one private log entry. `cycle6_c6_01.v2`, `awq_in_contemporaneous_freeze=True`. The run passed 87 standard and 48 cross-mix parity cases. The selection digest was `148effc0164363bf3516dce7eeb5f13611630f2226361cc31ecd03c4442b443d`. The shakedown recorded a dirty worktree because the v2 audit report had already been restored. |
| Four requested pytest files | **465 passed, 0 failed, 0 skipped** in **732.80 s**. Pytest reported 10 warnings (seven Triton deprecations, two empty CUDA graph warnings, one synchronization-debug warning). |

The suite ran with the project virtual environment, `PYTHONPATH=.` and a private pytest cache. The four files were `tests/test_c6_01_wide_rotated_qat.py`, `tests/test_nvfp4_inference.py`, `tests/test_c5_03_nvfp4_120m_fusion.py` and `tests/test_c5_02_isoloss_ptq_mitigation.py`. I rehashed the canonical result, rotation module and external log after the live checks. All three still matched the pins above.

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
