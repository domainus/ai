# Cycle 4 C4-01a audit of the 120m FineWeb-Edu transfer

**Verdict — PASS_WITH_CAVEATS.** Merge the two experiment commits. The 15 fixed-token and 12 fixed-compute runs support closing the three-seed 120m backlog target at 20,971,520 tokens and a 300-second compute budget. Keep **`ps3_blackwell` as the conservative 120m default**. Both Schatten variants beat compiled Polar Express on throughput at both budgets. Against PS3, the extra throughput was 0.08–0.18%. Neither variant established a test-loss advantage over PS3.

**State promotion.** Record the measured budgets, zero divergences and zero *accepted-compute* overshoot. The fixed-token stage includes AdamW. The wall stage compares the four Muon variants. C4-01a supplies no 300-second AdamW result. The experiment supports a protocol-specific speed claim. It leaves optimizer promotion, equivalence of model quality, and the best long-budget AdamW LR open. This review changes only the audit document.

## Ranked findings

| Severity | Finding | Consequence |
|---|---|---|
| MEDIUM | The canonical validator checks substantial internal consistency but accepts coordinated changes to code/configuration hashes or seed assignments. It also accepts an unrelated summary commit and some summary metadata changes. | A fresh promotion needs an independent commit, file-hash and fingerprint check. All three matched for these artifacts. |
| MEDIUM | Fixed LRs, three seeds and incomplete order balance limit generalization. PE ran first in all three fixed-token blocks. SS3 vs PS3 loss differences have wide intervals. | Retain the measured rates and intervals. They do not establish a new training default or a tuned upper bound for AdamW. |
| MEDIUM | Two interruptions can make a resume log unreadable. A torn final line is ignored in memory but remains on disk for the next append. | Ordinary completion and one-interruption recovery passed. Repair the tail before appending in a future driver revision. Both archived stages loaded zero rows. |
| LOW | The wall artifact stores one discarded duration. The validator accepts any positive replacement that crosses the deadline. No post-training weights or independent raw timestamps for the discarded step were archived. | The 12 records pass the deadline arithmetic. Direct bitwise rollback evidence comes from the live probe below. |
| LOW | Wall epoch fraction is reported, with no ceiling. The compute budget excludes setup, batches, snapshots, evaluation and the rejected step. | Describe the result as a 300-second accepted model-plus-optimizer compute budget. The observed row start-to-end intervals were 319–320 seconds. |

No critical or high finding invalidates the committed observations.

## Gate 1 — commits, strict JSON and provenance

The isolated checkout started clean. Its first three commits on the requested audit branch were as follows.

| Revision | Full SHA | Diff from parent |
|---|---|---|
| Results | `302e26d694a37fd2f39475d0ca8b69b220c6127b` | Only the three canonical JSONs listed below |
| Implementation and run provenance | `714f5c1c5fce1f9c432188abdb7d34062c1bff6f` | Only `research/polar_opt/bench_c4_01a_120m_long.py` and `tests/test_c4_01a_120m_long.py` |
| Base | `94d73807e1e13f50258fa83ddf1782420c25a1cd` | C4-01b post-merge hardening |

All three files passed a parser with a rejecting `parse_constant` callback and a duplicate-key check at every object. Each tree round-tripped through `json.dumps(..., allow_nan=False)`. The recorded Git SHA in all three files is the implementation commit above, and every recorded dirty flag is false. No resume logs or weights entered the results diff.

| File in `results/` | SHA-256 over file bytes | Recomputed stage configuration hash |
|---|---|---|
| `cycle4_c4_01a_120m_tokens_21m.json` | `69f42e669380f29106dc39498e50deee84c54ca75fe1fa95ca27a01faa570b1a` | `dcd72411ed6a30d9` |
| `cycle4_c4_01a_120m_wall_300s.json` | `eec2272b10e5912b6b99607d01b432123d049e11bc834a563ce8ea1650f088b6` | `c5cd2286055c60ee` |
| `cycle4_c4_01a_120m_summary.json` | `22bebb7e9dfbc34d0b890a0c5cbfa507397019feb9ed1f91da878e2f46a0604f` | Binds both stage file digests |

I reconstructed the inner harness hash from the model configuration, split hashes, budget, optimizer settings, schedule, compile setting and eval count. I separately serialized the outer object with sorted keys, using the protocol, inner hash, schema, quick flag, code digest, Git SHA and recorded Torch, Triton, CUDA, device and capability. Both truncated hashes matched every row and summary source entry. Both source file digests matched the summary.

The code SHA-256 was `31f22cf9dab5488471b70eed1d208badbda49b7299a80668bb02e1aa0e321a01`. I reproduced it from the ordered file bytes and perturbed each input in memory. Each perturbation changed the digest. The inputs are the driver, `harness_tier2_real`, `harness_tier2_lm`, `real_text_data`, `tier2_candidates`, `baselines`, `circuits`, `kernels_blackwell`, `polar_switch_blackwell`, `schatten_switch_blackwell` and `micro_rotated_ptq`. This covers the active training and kernel adapters and the transitive modules added in C4-01b M1. Both versions include Triton and CUDA in the configuration hash. The list is explicit. For example, `circuits` imports `spectral_scaling` for other candidates, and perturbing that module leaves this digest unchanged. The pinned tracked Git tree supplies the broader source identity. The source digest excludes external toolchain binaries.

I read the original `/home/ryan/Documents/projects/ai/artifacts/real_text` shards byte-for-byte. All three hashes matched the manifest, the two run files, the summary data and `micro_rotated_ptq.EXPECTED_SHA256`.

| Split | uint16 tokens | Bytes | SHA-256 |
|---|---:|---:|---|
| train | 26,004,518 | 52,009,036 | `015ab83ceb742555c2398963a0bd1f3c5502dc52cf70a22bf173f1571326ca3a` |
| val | 2,002,330 | 4,004,660 | `371b64a537f7b1e306ea174a868b05932a254a98b7fbdbf4146648d93180b0b1` |
| test | 2,000,081 | 4,000,162 | `4eb46f1424856849d2973923dbf3c886d37eb743cf22053fc7e1eea1f41ab92a` |

## Gate 2 — independent numerical replay

The preset has 123,587,328 parameters, including 84,934,656 Muon matrix parameters. It uses width 768, 12 layers, 12 heads, hidden size 2,048 and sequence length 512. Micro-batch 8 and accumulation 8 give 32,768 tokens per step. I grouped raw rows by budget, optimizer and seed. I summed the positive per-step durations in harness order and recovered each recorded accepted duration exactly. Tokens equaled steps × 32,768. Dividing by accepted time reproduced throughput. All 27 rows had finite results and zero divergences. Each split evaluation used 488 complete 8 × 512 batches, or 1,998,848 scored targets of the requested 2,000,000. The 1,152-target remainder is common to every run.

I regenerated the full named initial-weight fingerprints on CPU at the 120m preset. I also regenerated the harness fingerprints of the first three batches' start offsets from the verified training shard. These matched across optimizers and budgets by seed. The stream fingerprint samples a prefix. The generator and seed determine the remaining offsets.

| Seed | Initial state | Batch stream |
|---:|---|---|
| 1 | `74a0bd19147b39be` | `b9f91c01f18ce738` |
| 2 | `3b55f66d5f4d4ac5` | `f32172b52cfa7de3` |
| 3 | `3ebcf8ac9ea46e96` | `e88efd98cfc4e5e9` |

The table shows the independently computed three-seed means. I checked all 63 metric cells against both stage and summary means, plus the individual values and sample SDs in the summary. Loss is mean next-token negative log likelihood in nats.

| Stage | Optimizer | Steps | Tokens | Epoch fraction | Accepted s | Tokens/s | Val loss | Test loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tokens | adamw | 640.000000 | 20971520.000000 | 0.806457 | 236.522744 | 88665.976320 | 5.087769 | 5.112360 |
| tokens | pe_compiled | 640.000000 | 20971520.000000 | 0.806457 | 250.735734 | 83639.976321 | 4.626116 | 4.640991 |
| tokens | ps3_blackwell | 640.000000 | 20971520.000000 | 0.806457 | 243.814337 | 86014.301981 | 4.632507 | 4.646928 |
| tokens | ss3_54 | 640.000000 | 20971520.000000 | 0.806457 | 243.612206 | 86085.670161 | 4.629442 | 4.644636 |
| tokens | ss3_53 | 640.000000 | 20971520.000000 | 0.806457 | 243.367599 | 86172.194673 | 4.630795 | 4.646210 |
| wall | pe_compiled | 765.000000 | 25067520.000000 | 0.963968 | 299.706817 | 83640.140363 | 4.491943 | 4.502895 |
| wall | ps3_blackwell | 787.333333 | 25799338.666667 | 0.992110 | 299.762683 | 86065.872872 | 4.471966 | 4.482182 |
| wall | ss3_54 | 788.000000 | 25821184.000000 | 0.992950 | 299.767126 | 86137.477266 | 4.470691 | 4.480756 |
| wall | ss3_53 | 789.000000 | 25853952.000000 | 0.994210 | 299.872586 | 86216.457485 | 4.473403 | 4.483813 |

For every planned pair I formed three candidate-minus-reference differences, or three candidate/reference ratios. The ratio null was 1. I used a one-sample SciPy Student-t test on the paired, null-centered values, with sample SD, SE = SD/√3, and df = 2. For the two-sided 95% interval I inverted the Student-t CDF at 0.975 (4.30265272975). I rebuilt Holm adjustments by sorting the raw probabilities, multiplying by the remaining group size, taking the cumulative maximum and clipping at 1. The per-seed values, mean, SD, SE, t, both CI endpoints, raw p, and all applicable Holm p-values matched at absolute/relative tolerance 1e−12. Appendix A contains every contrast.

Family sizes were 3 vs PE, 2 vs PS3 and 4 vs AdamW for each budget/metric combination that exists. The same-reference groups had 18, 12 and 12 comparisons, respectively. The all-Muon group contained 30. The 12 fixed-token AdamW contrasts carry no all-Muon adjustment.

The four Muon variants reduced fixed-token test loss by 0.465–0.471 nats versus the specified AdamW, with AdamW-reference joint p ≤ 0.00598. AdamW processed more tokens per second. At fixed compute PS3 improved test loss by 0.0207 nats versus PE (family p = 0.0138, all-Muon p = 0.0643). Against PS3, both SS throughput ratios passed the all-30 correction at fixed tokens (SS3_54 p = 0.0366, SS3_53 p = 0.0280). At fixed time the corresponding p-values were 0.1885 and 0.0792. The correction scope matters for these claims.

### Canonical validator probes

All three unmodified files passed `validate_results(..., canonical=True)`, including a summary reload with the on-disk source-hash checks. The following probes used independent copies.

| Change | Observed response |
|---|---|
| Dirty stage, malformed stage Git SHA, altered test shard hash, mismatched single row hash | Rejected |
| One wrong seed or one wrong initialization fingerprint | Rejected |
| Wall overshoot of 0.1 s, deleted discard on an early stop, discard too short to cross the deadline | Rejected |
| Changed summary source digest, reloading canonical sources from disk | Rejected |
| Valid-format alternate stage commit, arbitrary code digest, or arbitrary configuration hash copied into all rows | Accepted as a standalone stage |
| Swap the labels of entire seed blocks 1 and 2, retaining their original fingerprints | Accepted as a standalone stage |
| Nonexistent order position, CPU text in the recorded kernel description, or discard duration changed to 87 s | Accepted as a standalone stage |
| Unrelated summary Git SHA, or summary model width changed to 123 with original sources | Accepted even with the ordinary canonical summary file reload |
| Supply the two validated source objects directly and change the summary's recorded file digest | Accepted in that overload. The ordinary file-reloading path detects it. |

The production validator checks pairing and consistency. Independent reconstruction supplies the stronger provenance evidence above. The historical wall rows cannot prove exact rollback state from their scalar durations alone.

## Gate 3 — load, schedule and interpretation

**Clocks and host load.** Across all 27 rows I checked 228 start, interior and end telemetry snapshots, including all 174 interior samples (every 100 accepted steps). Every snapshot was P1 with memory clock 14,801 MHz. Fixed-token interior SM clocks were 2,827–2,835 MHz, utilization 98–100%, temperature 60–69 °C and power 296.44–306.61 W. All 84 wall interior SM samples were 2,827 MHz, with utilization 95–100%, temperature 61–69 °C and power 296.25–307.13 W. Of all snapshots, 26 reported active reason `0x4` (software power cap) and 202 reported `0x400` (reliability clock limit, confirmed by the live driver). No sampled mask indicated thermal slowdown. The clocks were constrained but stable at the sample times. Sampling does not exclude short transients.

On the 16-logical-CPU host the 1/5/15-minute start/end loads ranged from 2.28–3.50 / 2.49–3.00 / 2.57–2.94 during tokens and 2.25–3.61 / 2.46–3.04 / 2.55–2.84 during wall. No sampled load spike approached host capacity. Timestamps were monotone and runs did not overlap. Appendix B gives the row-level evidence.

**Order and limiter.** Replaying `Random(f'{seed}-{stage}').shuffle` reproduced all 27 positions. PE was always first in the fixed-token stage and last in two of three wall blocks. After centering each row's rate on that optimizer's three-seed mean, position-average residuals were within ±0.005% in both stages. This small diagnostic cannot separate an optimizer effect from a position it always occupied. The stage-level timing decomposition supports the large PE gap. Across fixed-token rows the means of the recorded median forward/backward times were 365.10–365.33 ms. PE orthogonalization took 19.44 ms, PS3 took 9.06 ms, and SS3_54/53 took 8.82/8.44 ms. The roughly 11 ms saved against PE explains the roughly 3% gain on a 380–392 ms step. The much smaller PS3-to-SS3 gaps warrant correspondingly narrower claims.

**Wall LR.** In `harness_tier2_real.lr_multiplier`, a positive fraction and positive warmup-step count select the wall branch. Before 18.75 accepted seconds the multiplier is `max(1/50, elapsed/18.75)`, capped at 1. Then cosine decay runs over the remaining 281.25 s toward 0.1. The first timed step uses 0.02 of base LR, i.e. 0.0004 for the matrices and 0.00002 for auxiliary AdamW. The 0.375-second floor affected exactly the first update in every archived wall row. Replaying actual step-start times gave 48 warmup updates for PE and 50 for each faster method. Elapsed time sets the schedule, so faster methods get slightly more updates within it. There is no extra 50-step ramp on top of the wall ramp. Three untimed compile-warmup steps use a disposable stream and optimizer, then the model reloads its original weights.

**Training exposure.** The token stage enforces a count below one train-shard length, at 0.806457 of the shard. The wall stage stops on time alone. It peaked at 789 steps, 25,853,952 tokens and fraction 0.994210. A faster future run could pass one shard length. Sampling uses random windows with replacement. Replaying all start offsets and taking the union of their target positions gave 55.23–55.45% actual distinct training positions for fixed tokens and 61.77–63.03% for fixed time. Epoch fraction measures sampled volume.

**LR choice.** C3-01 first used a seed-0, 128-step validation pilot. PS3 at 0.02 won among 0.005, 0.01, 0.02, 0.04. AdamW at 0.001 won among 0.0003, 0.0006, 0.001, 0.002. PE had a single 0.02 confirmation. C4-01a transferred the rates, with 0.001 auxiliary LR for Muon. This preserves a shared protocol across orthogonalizers. C4-01a did not search validation loss at the longer horizon. C4-01b selected AdamW 0.0015 for 544 steps at the **50m** preset in an iso-loss calibration against 320 Muon steps. That is a different objective and model. Neither study identifies the optimal 120m AdamW LR at 640 steps.

**100 s versus 300 s.** At 100 s, SS3_54 minus PS3 test-loss differences by seed were −0.005011, −0.005287 and −0.006094 nats (262 steps each). Mean −0.005464, SD 0.000563, raw p = 0.00352. The C3 audit's 0.0352 was Holm across ten wall test-loss pairs. Its joint p was 0.08096. At 300 s the differences were −0.006978, −0.006100 and **+0.008802**, mean −0.001425, SD 0.008868, CI [−0.023455, +0.020604], raw p = 0.80688 and all three adjusted p-values = 1. SS3_54 took 788/788/788 accepted steps, PS3 took 787/788/787. Seed 3 reversed the sign. The 100 s and 300 s protocols also used different warmup floors (1/20 and 1/50), warmup durations (6.25 and 18.75 s), and eval prefixes (1,048,576 and 1,998,848 targets). These are separate runs, not evaluations along a common checkpoint trajectory. Three seeds cannot identify a cause for the reversal or establish a budget interaction.

**Canonical device.** The non-quick parser defaults to CUDA. With CUDA availability forced false, the quick parser selected CPU while canonical training raised the sm_120+ error before loading data. Explicit CPU is noncanonical and the reserved-filename guard rejects it. The training guard accepts capability ≥ (12, 0). Both recorded environments name an RTX 5080 at (12, 0), PyTorch 2.14.1+cu130, CUDA 13.0, Triton 3.8.0 and driver 615.78.08.

## Gate 4 — live RTX 5080 checks

**Test suite.** In the isolated checkout, `PYTHONPATH=. /home/ryan/Documents/projects/ai/.venv/bin/pytest tests/test_c4_01a_120m_long.py tests/test_tier2_real.py -q -rs` gave **47 passed, 1 skipped** in 343.94 s. The 241 warnings were Triton Python-AST deprecations. The skip was the expected missing default shard directory in this worktree. Before collecting that test in a second pytest invocation, I pointed `real_text_data.DEFAULT_DIR` at the original read-only artifact. The unmodified on-disk shard test passed in 0.30 s. All 48 distinct tests therefore ran and passed.

**Kernel dispatch and numerics.** I profiled one warmed factory call per method on the same FP32 CUDA tensor of shape (2, 128, 64). All four results were finite, nonzero and pairwise distinct. I observed the actual variant and dtype at the candidate adapter and device kernels in the profiler.

| Factory | CUDA events | Observed execution |
|---|---:|---|
| pe_compiled | 32 | Torch-compiled Polar Express callable, with Inductor fused Triton events |
| ps3_blackwell | 16 | PS3_55, FP16, including `_syrk_kernel` and `_symprod_kernel` |
| ss3_54 | 16 | SS3_54, FP16, including `_syrk_norm_kernel`, `_square_norm_kernel` and `_fold_and_row0_axpy_kernel` |
| ss3_53 | 15 | SS3_53, FP16, with the Schatten norm/fold kernels and one fewer event on this shape |

The live PyTorch CUDA matmul and cuDNN TF32 flags were both false, matmul precision was highest, and both FP16 and BF16 reduced-precision reduction flags were false. Canonical metadata agrees. Those are backend settings. The forward uses BF16 autocast, and the three named Blackwell adapters explicitly use FP16. These short dispatch probes do not supply throughput estimates for the 120m model.

**Bitwise deadline rollback.** I ran the production wall loop with the tiny model, verified real shards, SS3_54 and a 0.4-second budget. At each actual synchronized step end, I read the clock and captured every persistent model tensor as CPU bytes. It accepted 35 steps in 0.39402214996516705 s. The next step took 0.011701222974807024 s and crossed the deadline. All 16 tensors changed during that step. After rollback, all 16 tensors (13,206,784 bytes) matched the preceding accepted state exactly. The reported accepted overshoot was 0.0 s. The harness restores parameters for immediate evaluation, then discards the optimizer. This probe does not establish an optimizer-state checkpoint or bitwise reproduction of the historical 120m endpoints.

**Resume.** I ran the normal quick CUDA CLI on verified real shards with two Muon optimizers and seed 2. Starting with one complete row and a torn suffix, it reused that row exactly, ran the other row once, and removed the JSONL after the final write. The fresh row matched the original run's seed, protocol, step/token counts and both evaluation losses exactly in this probe. Replaying both complete rows called training zero times, reproduced all rows and derived fields exactly, and removed the log. Timing equality applies to rows read from the log.

For a second-interruption probe I supplied a torn final line to a three-optimizer quick run. The production writer completed two real rows before I interrupted the third. The first new row was concatenated to the old torn bytes. With a valid row following it, the malformed line was now internal. Retrying `--resume` raised `JSONDecodeError` at line 1, column 8. The direct cause is in `load_rows` and the append in `run_stage`. Truncating to the last complete record before the first resumed append would preserve later recovery. The two archived canonical stages report 15/12 freshly run rows and zero rows resumed.

The audit replay used short CUDA runs plus the archived per-step timings. It did not repeat the full 27-run training campaign.

**Filename and clean-tree checks.** I exercised the production parser against scratch targets. Quick mode, a subset of optimizers, altered seeds, token count, seconds, LR warmup, a `subdir/../` alias, a symlink to a canonical filename, and the other stage's canonical name all failed with CLI exit 2. An alternate Git index gave the real checkout a tracked dirty status for the process only. The production training entry point rejected the canonical wall target before reading data or starting training. No reserved target was written. The tracked cleanliness check uses `git status --porcelain --untracked-files=no`, so untracked files alone do not activate it.

## Appendix A — all 42 paired contrasts

Rate differences are tokens/s. Loss differences are nats. Ratios are dimensionless. CIs are marginal two-sided 95% intervals. The final three columns are the Holm probabilities for the metric family, same-reference group and all 30 Muon contrasts.

| Stage | Candidate vs reference | Metric | Mean | CI low | CI high | t | Raw p | Family | Reference | All Muon |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tokens | ps3_blackwell vs pe_compiled | rate ratio | 1.028388 | 1.02608 | 1.030696 | 52.92148 | 0.0003568647 | 0.0007326117 | 0.002746175 | 0.006780429 |
| tokens | ps3_blackwell vs pe_compiled | rate diff | 2374.326 | 2186.377 | 2562.274 | 54.35476 | 0.0003383019 | 0.0006865437 | 0.002746175 | 0.006766038 |
| tokens | ps3_blackwell vs pe_compiled | loss diff | 0.00593757 | -0.01500543 | 0.02688057 | 1.219849 | 0.3468456 | 1 | 1 | 1 |
| tokens | ss3_54 vs pe_compiled | rate ratio | 1.029241 | 1.027105 | 1.031377 | 58.89814 | 0.0002881437 | 0.0007326117 | 0.002746175 | 0.006051018 |
| tokens | ss3_54 vs pe_compiled | rate diff | 2445.694 | 2272.282 | 2619.106 | 60.68199 | 0.0002714586 | 0.0006865437 | 0.002746175 | 0.005972088 |
| tokens | ss3_54 vs pe_compiled | loss diff | 0.003645154 | -0.01178389 | 0.0190742 | 1.016514 | 0.4163459 | 1 | 1 | 1 |
| tokens | ss3_53 vs pe_compiled | rate ratio | 1.030276 | 1.02824 | 1.032312 | 63.97999 | 0.0002442039 | 0.0007326117 | 0.002746175 | 0.005616689 |
| tokens | ss3_53 vs pe_compiled | rate diff | 2532.218 | 2367.37 | 2697.067 | 66.09247 | 0.0002288479 | 0.0006865437 | 0.002746175 | 0.005492349 |
| tokens | ss3_53 vs pe_compiled | loss diff | 0.005219491 | -0.01708102 | 0.02752001 | 1.007047 | 0.4199471 | 1 | 1 | 1 |
| tokens | ss3_54 vs ps3_blackwell | rate ratio | 1.00083 | 1.000659 | 1.001001 | 20.86399 | 0.002289348 | 0.003109818 | 0.02288194 | 0.0366111 |
| tokens | ss3_54 vs ps3_blackwell | rate diff | 71.36818 | 56.65408 | 86.08228 | 20.86927 | 0.002288194 | 0.00310688 | 0.02288194 | 0.0366111 |
| tokens | ss3_54 vs ps3_blackwell | loss diff | -0.002292416 | -0.01295092 | 0.008366087 | -0.9254087 | 0.4524479 | 0.9048957 | 1 | 1 |
| tokens | ss3_53 vs ps3_blackwell | rate ratio | 1.001836 | 1.001524 | 1.002147 | 25.33032 | 0.001554909 | 0.003109818 | 0.01864128 | 0.02796192 |
| tokens | ss3_53 vs ps3_blackwell | rate diff | 157.8927 | 131.0855 | 184.6999 | 25.34232 | 0.00155344 | 0.00310688 | 0.01864128 | 0.02796192 |
| tokens | ss3_53 vs ps3_blackwell | loss diff | -0.0007180791 | -0.01713389 | 0.01569774 | -0.1882115 | 0.8680775 | 0.9048957 | 1 | 1 |
| tokens | pe_compiled vs adamw | rate ratio | 0.9433153 | 0.9412675 | 0.9453631 | -119.1007 | 7.048971e-05 | 7.048971e-05 | 0.0004226227 | — |
| tokens | pe_compiled vs adamw | rate diff | -5026 | -5207.502 | -4844.498 | -119.1451 | 7.043712e-05 | 7.043712e-05 | 0.0004226227 | — |
| tokens | pe_compiled vs adamw | loss diff | -0.4713692 | -0.5696864 | -0.3730521 | -20.62853 | 0.002341725 | 0.005971653 | 0.005971653 | — |
| tokens | ps3_blackwell vs adamw | rate ratio | 0.9700937 | 0.9700128 | 0.9701745 | -1591.879 | 3.946205e-07 | 1.578482e-06 | 4.735446e-06 | — |
| tokens | ps3_blackwell vs adamw | rate diff | -2651.674 | -2658.998 | -2644.351 | -1557.872 | 4.120372e-07 | 1.648149e-06 | 4.735446e-06 | — |
| tokens | ps3_blackwell vs adamw | loss diff | -0.4654317 | -0.5453397 | -0.3855236 | -25.0612 | 0.001588403 | 0.005971653 | 0.005971653 | — |
| tokens | ss3_54 vs adamw | rate ratio | 0.9708986 | 0.9708064 | 0.9709907 | -1358.854 | 5.415692e-07 | 1.624708e-06 | 5.409089e-06 | — |
| tokens | ss3_54 vs adamw | rate diff | -2580.306 | -2588.471 | -2572.141 | -1359.683 | 5.409089e-07 | 1.648149e-06 | 5.409089e-06 | — |
| tokens | ss3_54 vs adamw | loss diff | -0.4677241 | -0.5506563 | -0.3847918 | -24.26625 | 0.001693909 | 0.005971653 | 0.005971653 | — |
| tokens | ss3_53 vs adamw | rate ratio | 0.9718744 | 0.9716107 | 0.9721382 | -458.8224 | 4.750154e-06 | 9.500309e-06 | 3.800123e-05 | — |
| tokens | ss3_53 vs adamw | rate diff | -2493.782 | -2517.309 | -2470.254 | -456.0524 | 4.808033e-06 | 9.616066e-06 | 3.800123e-05 | — |
| tokens | ss3_53 vs adamw | loss diff | -0.4661497 | -0.5437326 | -0.3885669 | -25.85211 | 0.001492913 | 0.005971653 | 0.005971653 | — |
| wall | ps3_blackwell vs pe_compiled | rate ratio | 1.029002 | 1.02821 | 1.029794 | 157.4957 | 4.03122e-05 | 4.03122e-05 | 0.0005518116 | 0.001024793 |
| wall | ps3_blackwell vs pe_compiled | rate diff | 2425.733 | 2360.205 | 2491.26 | 159.278 | 3.941511e-05 | 3.941511e-05 | 0.0005518116 | 0.001024793 |
| wall | ps3_blackwell vs pe_compiled | loss diff | -0.02071332 | -0.02677291 | -0.01465373 | -14.70763 | 0.004591088 | 0.01377326 | 0.02754653 | 0.06427523 |
| wall | ss3_54 vs pe_compiled | rate ratio | 1.029858 | 1.029529 | 1.030188 | 389.8371 | 6.580052e-06 | 1.31601e-05 | 9.955842e-05 | 0.0001776614 |
| wall | ss3_54 vs pe_compiled | rate diff | 2497.337 | 2470.533 | 2524.141 | 400.8842 | 6.222401e-06 | 1.24448e-05 | 9.955842e-05 | 0.0001742272 |
| wall | ss3_54 vs pe_compiled | loss diff | -0.02213849 | -0.03850084 | -0.005776145 | -5.821551 | 0.02826193 | 0.05652386 | 0.1413096 | 0.2543574 |
| wall | ss3_53 vs pe_compiled | rate ratio | 1.030802 | 1.030602 | 1.031003 | 659.8056 | 2.297029e-06 | 6.891088e-06 | 3.90495e-05 | 6.661385e-05 |
| wall | ss3_53 vs pe_compiled | rate diff | 2576.317 | 2560.295 | 2592.339 | 691.8489 | 2.089181e-06 | 6.267543e-06 | 3.760526e-05 | 6.267543e-05 |
| wall | ss3_53 vs pe_compiled | loss diff | -0.0190824 | -0.03881759 | 0.0006527909 | -4.160332 | 0.0532065 | 0.05652386 | 0.212826 | 0.425652 |
| wall | ss3_54 vs ps3_blackwell | rate ratio | 1.000832 | 1.000357 | 1.001307 | 7.534851 | 0.01716159 | 0.01716159 | 0.1028308 | 0.1885231 |
| wall | ss3_54 vs ps3_blackwell | rate diff | 71.60439 | 30.74417 | 112.4646 | 7.540067 | 0.01713846 | 0.01713846 | 0.1028308 | 0.1885231 |
| wall | ss3_54 vs ps3_blackwell | loss diff | -0.001425173 | -0.02345471 | 0.02060437 | -0.2783546 | 0.8068788 | 1 | 1 | 1 |
| wall | ss3_53 vs ps3_blackwell | rate ratio | 1.00175 | 1.001159 | 1.002341 | 12.7371 | 0.006107531 | 0.01221506 | 0.04873592 | 0.07919587 |
| wall | ss3_53 vs ps3_blackwell | rate diff | 150.5846 | 99.78179 | 201.3874 | 12.75349 | 0.00609199 | 0.01218398 | 0.04873592 | 0.07919587 |
| wall | ss3_53 vs ps3_blackwell | loss diff | 0.001630923 | -0.02416331 | 0.02742515 | 0.272049 | 0.8110958 | 1 | 1 | 1 |

## Appendix B — telemetry and rollback by row

All times below are local on 2026-11-26 (UTC−07). The first column gives the seed and the one-based position within that seed. Ranges include start/end snapshots. Interior ranges, which describe steady training more closely, are in Gate 3. Load is the host's one-minute load average.

| Stage, seed.position | Optimizer | Start–end | SM MHz | °C | W | Load start→end |
|---|---|---|---:|---:|---:|---:|
| tokens 1.1 | pe_compiled | 10:21:25–10:25:54 | 2827–2880 | 41–68 | 216.82–305.16 | 3.14→2.53 |
| tokens 1.2 | ss3_53 | 10:26:07–10:30:29 | 2827–2865 | 51–67 | 218.88–303.35 | 3.50→3.24 |
| tokens 1.3 | ps3_blackwell | 10:30:30–10:34:53 | 2827–2865 | 52–68 | 227.62–304.39 | 3.12→2.83 |
| tokens 1.4 | ss3_54 | 10:34:54–10:39:16 | 2827–2857 | 52–68 | 230.34–304.07 | 2.86→2.82 |
| tokens 1.5 | adamw | 10:39:18–10:43:33 | 2827–2865 | 52–68 | 221.18–303.73 | 2.73→2.75 |
| tokens 2.1 | pe_compiled | 10:43:35–10:48:04 | 2827–2850 | 52–68 | 244.24–305.46 | 2.68→3.04 |
| tokens 2.2 | adamw | 10:48:06–10:52:21 | 2827–2865 | 53–68 | 239.33–304.44 | 2.89→3.11 |
| tokens 2.3 | ps3_blackwell | 10:52:22–10:56:44 | 2827–2842 | 53–69 | 237.14–303.7 | 3.02→2.70 |
| tokens 2.4 | ss3_53 | 10:56:46–11:01:08 | 2827–2857 | 53–67 | 232.34–304.24 | 2.91→2.89 |
| tokens 2.5 | ss3_54 | 11:01:09–11:05:31 | 2827–2857 | 53–67 | 229.03–304.52 | 2.78→2.41 |
| tokens 3.1 | pe_compiled | 11:05:34–11:10:03 | 2827–2835 | 53–67 | 233.68–306.61 | 2.72→2.30 |
| tokens 3.2 | adamw | 11:10:04–11:14:19 | 2827–2842 | 53–68 | 236.23–303.92 | 2.55→2.67 |
| tokens 3.3 | ss3_54 | 11:14:21–11:18:43 | 2827–2850 | 53–68 | 246.96–304.39 | 2.54→2.40 |
| tokens 3.4 | ss3_53 | 11:18:44–11:23:06 | 2827–2865 | 53–67 | 250.32–304.91 | 2.28→2.62 |
| tokens 3.5 | ps3_blackwell | 11:23:07–11:27:30 | 2827–2835 | 53–68 | 224.21–304.78 | 2.44→2.56 |
| wall 1.1 | ps3_blackwell | 11:27:55–11:33:15 | 2827–2872 | 49–68 | 211.56–304.44 | 2.31→2.78 |
| wall 1.2 | ss3_53 | 11:33:23–11:38:43 | 2827–2865 | 52–68 | 217.54–304.5 | 2.98→2.60 |
| wall 1.3 | ss3_54 | 11:38:44–11:44:04 | 2827–2857 | 53–69 | 222.73–304.37 | 3.05→2.94 |
| wall 1.4 | pe_compiled | 11:44:06–11:49:25 | 2827–2865 | 52–69 | 219.97–306.44 | 2.75→3.61 |
| wall 2.1 | ps3_blackwell | 11:49:28–11:54:48 | 2827–2850 | 53–68 | 238.2–304.45 | 3.28→2.77 |
| wall 2.2 | pe_compiled | 11:54:49–12:00:09 | 2827–2835 | 53–69 | 252.61–306.46 | 2.63→3.01 |
| wall 2.3 | ss3_53 | 12:00:10–12:05:30 | 2827–2857 | 53–68 | 248.15–304.59 | 3.17→2.67 |
| wall 2.4 | ss3_54 | 12:05:32–12:10:51 | 2827–2857 | 53–68 | 234.06–303.88 | 2.59→2.37 |
| wall 3.1 | ss3_54 | 12:10:54–12:16:13 | 2827–2857 | 52–69 | 235.32–305.02 | 2.27→2.35 |
| wall 3.2 | ps3_blackwell | 12:16:15–12:21:34 | 2827–2835 | 53–68 | 225.76–305.1 | 2.49→2.25 |
| wall 3.3 | ss3_53 | 12:21:36–12:26:56 | 2827–2835 | 53–68 | 244.75–304.47 | 2.40→2.35 |
| wall 3.4 | pe_compiled | 12:26:57–12:32:17 | 2827–2835 | 53–68 | 229.52–307.13 | 2.72→2.35 |

The following are the stored binary64 values printed at round-trip precision. For each row I recovered the accepted time from the sum of every retained duration. In all 12 rows the accepted total was at most 300 s and adding the one discarded duration crossed 300 s. The actual measured compute including that attempt crossed by 0.033912–0.333392 s. The model accepted 0.0 s of overshoot on every row.

| Seed | Optimizer | Accepted steps | Accepted seconds | Discarded step seconds |
|---:|---|---:|---:|---:|
| 1 | ps3_blackwell | 787 | 299.6814246685244 | 0.38086283300071955 |
| 1 | ss3_53 | 789 | 299.8614220942836 | 0.380199080100283 |
| 1 | ss3_54 | 788 | 299.7717277640477 | 0.38034159201197326 |
| 1 | pe_compiled | 765 | 299.67019948665984 | 0.39156123297289014 |
| 2 | ps3_blackwell | 788 | 299.9534824709408 | 0.37990929116494954 |
| 2 | pe_compiled | 765 | 299.7445800860878 | 0.39183786790817976 |
| 2 | ss3_53 | 789 | 299.8894701753743 | 0.3801709490362555 |
| 2 | ss3_54 | 788 | 299.76957419747487 | 0.37959575396962464 |
| 3 | ss3_54 | 788 | 299.76007626764476 | 0.3797261680010706 |
| 3 | ps3_blackwell | 787 | 299.65314229158685 | 0.3807699850294739 |
| 3 | ss3_53 | 789 | 299.8668650542386 | 0.37985471705906093 |
| 3 | pe_compiled | 765 | 299.7056719942484 | 0.3917979020625353 |


**Repository evidence.** [C4-01a driver](../research/polar_opt/bench_c4_01a_120m_long.py) · [training harness](../research/polar_opt/harness_tier2_real.py) · [fixed-token rows](../results/cycle4_c4_01a_120m_tokens_21m.json) · [wall rows](../results/cycle4_c4_01a_120m_wall_300s.json) · [summary](../results/cycle4_c4_01a_120m_summary.json) · [C3-01 comparison](AUDIT_cycle3_c3_01_gpt62_astra.md) · [C4-01b hardening](../research/quant_ptq/bench_isoloss_ptq.py)
