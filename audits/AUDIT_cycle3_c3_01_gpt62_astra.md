# C3-01 audit · 120m FineWeb-Edu transfer and wide-layer crossover

**Verdict — PASS_WITH_CAVEATS.** Merge the benchmark and six artifacts as evidence for the measured 256-step and 100-second budgets. Keep C3-01’s longer-budget backlog target open unless a revised scope is recorded. The graph benchmark supports SS3_53’s speed advantage across the measured widths and SS3_54’s advantage at 1536 and 2048. In the 120m training runs, all three Blackwell candidates delivered about 2.9–3.0% more tokens/s than compiled PE. The incremental end-to-end differences between the switches and PS3 Blackwell were below 0.1% in the mean. The loss results do not justify replacing the training default.

| Severity | Finding | Merge disposition |
|---|---|---|
| CRITICAL | None found. The recorded raw samples, trial pairing and summary arithmetic support the scoped result. | Accept the measurements. |
| HIGH | Scope of closure. The backlog says ≥20M tokens and gives a longer target using the ambiguous wording “`--seconds >= 300` steps.” Each token trial used 8.39M tokens and each wall trial had 100 s of accepted compute. | Merge as a short-budget result. Closing the original target needs longer runs or an explicit scope revision. |
| MEDIUM | Statistical promotion. Three seeds yield family-dependent loss inferences. The wall LR schedule depends on elapsed compute. Equal step counts remove the extra-token explanation for SS3_54 but do not isolate optimizer quality. | Report adjusted p-values, paired uncertainty, the budget definition and the seed counts with any promotion claim. |
| MEDIUM | Artifact overwrite. A successful non-quick subset run without an explicit output path overwrites the canonical full result. | Published data are intact. Use a separate output path for reruns. Add a default-path guard and CPU-only regression test before routine automated reuse. |
| LOW | Clock order, allocator history and host load limit precision. Exact ties can duplicate an interpolated crossing. | Keep the clocks separate, qualify w512 timing tails, and report measured brackets. Test tie semantics when updating the helper. |

### Candidate decisions

- **SS3_53** merits promotion for wide-layer orthogonalization experiments. Its graph and eager median speedups over PS3 Blackwell exceed one on all seven rows. At 120m the training orthogonalizer saves about 0.33 ms on a roughly 380 ms step. The observed end-to-end gain over PS3 Blackwell is about 0.09%. Keep its training-quality advantage provisional.
- **SS3_54** merits promotion as a wide-layer candidate from width 1536 on this grid. Treat 1024 as a timing tie on this evidence. At 120m the fused proxy is near parity, actual training orthogonalization is slightly slower than PS3 Blackwell, and the wall test-loss signal deserves an independent replication.
- **PS3_55_blackwell** (`ps3_blackwell` in Part A, `muon_ps3_blackwell` in training) is the conservative 120m Blackwell choice. Retain it as the reference for evaluating switches. It has a clear throughput gain over compiled PE in both budgets. Fixed-token loss intervals allow small gains or regressions. A claim of loss equivalence would need a specified tolerance and adequate precision.

No source, test or JSON repair is required to merge the pinned artifacts with those limits. The CLI guard and exact-tie tests are follow-up hardening. A claim of general training superiority or full backlog completion needs additional evidence. [Cycle 3 backlog](../state/BACKLOG.md#L16)

## Provenance and CPU verification

Audited result commit `4011c9ca2de855bf8062cd8f0baddf52586f79d3`. Its direct parent is code commit `02c9a3419e37f8651364382871a3db8935094974`, co-authored by Claude Fable 5.2. Their diff contains exactly the six listed JSON files. I decoded all six with non-finite constants and duplicate keys rejected, then serialized each with `allow_nan=False`. All six record that exact parent and `git_dirty=false`. The harness checks tracked files for dirtiness. Part A records an RTX 5080 with 64 MiB L2. This audit used CPU computation and source inspection throughout.

All five training JSONs match the previously byte-verified FineWeb-Edu sample-10BT / GPT-2 records for source, encoding, token counts and digests. The sixth JSON times Gaussian tensors and has no shards. The harness calls `ShardData` with verification enabled. The shard binaries are absent from this audit worktree, so the live check here establishes metadata agreement. The prior audit supplies the independent file-byte check. [Recorded file-byte verification](AUDIT_cycle2_gpt62_astra.md#L13) · [Shard verifier](../research/polar_opt/real_text_data.py#L164)

| Split | Tokens | SHA-256 |
|---|---:|---|
| Train | 26,004,518 | `015ab83ceb742555c2398963a0bd1f3c5502dc52cf70a22bf173f1571326ca3a` |
| Validation | 2,002,330 | `371b64a537f7b1e306ea174a868b05932a254a98b7fbdbf4146648d93180b0b1` |
| Test | 2,000,081 | `4eb46f1424856849d2973923dbf3c886d37eb743cf22053fc7e1eea1f41ab92a` |

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /home/ryan/Documents/projects/ai/.venv/bin/pytest tests/test_wide_crossover.py tests/test_tier2_real.py -q
```

**20 passed, 9 skipped, 0 failed** in 22.55 s. Eight tests require CUDA (three wide, five tier-2). One requires real shards in the worktree. The CPU suite exercised Williams balance, preset shapes, interpolation, pairing and deadline rollback. The stored GPU samples and source provide the evidence for device-side checks. [Wide tests](../tests/test_wide_crossover.py) · [Harness tests](../tests/test_tier2_real.py)

## Timing methodology

- **Balance.** Seven methods × 14 blocks × five samples, separately on each clock and each shape row. I counted the recorded permutations. Each method occupies each position twice. Each of the 42 ordered, distinct method pairs occurs twice within blocks. This balance concerns block slots. Repeated samples within a slot share a capture. Block boundaries and drift across widths or clock passes are outside the design.
- **Timing and precision.** Before each timed call the same stream zeros a 268,435,456-byte uint8 buffer (4 × 67,108,864). The start event follows the zero and the end event follows the call. This puts the flush outside the measured interval and applies cache pressure. The experiment does not measure resulting cache residency. Both reduced-precision accumulation flags are assigned false before factories run and are recorded false. PE uses BF16 work tensors and the PS/SS methods use FP16. The proxy measures native orthogonalizer output, excluding FP32 layout conversion.
- **Graph lifetime.** The initial pass warms all implementations and verifies one capture per method. Each measured graph slot then captures with zero additional side-stream warmups, runs one untimed replay, synchronizes, and times five flushed replays. It synchronizes again, clones the final outputs, releases graph and static output references, and calls `empty_cache()` before the eager comparison. This bounds live graph memory at width 2048. Emptying unused allocator blocks cannot reset clocks, compilation caches, all live allocations or thermal history. A slot includes capture, validation and allocator activity outside the timer. These costs matter for fresh graphs and are excluded from the latency estimate.
- **Pass order.** Graph always precedes eager at a given width. Eager queues the full schedule, then synchronizes and reads the event pairs. Eager launch gaps can enter the device interval. The two passes have different synchronization, allocator and temperature histories. Williams ordering supports within-clock method comparisons. Small graph–eager differences cannot establish a causal graph benefit. A reversed-order independent session would test such a claim.
- **Verification.** All 49 method/shape records report finite outputs, unchanged inputs, 15 verified captures (one preliminary plus 14 timed slots) and exactly 0.0 maximum graph–eager difference. That totals 735 captures, 6,860 stored timings and 98 medians. The source compares finite outputs through integer views of matching element size using `torch.equal`. A zero in the JSON follows bitwise equality, including signed zero. The check samples the last replay of each slot, and the JSON retains the first per-group detail and maximum across captures. Input checks compare FP32 values against CPU copies after per-method warming and at the end of timing. Transient mutations between checkpoints would be invisible.
- **Shapes.** Grid width 768 uses hidden size 2176. Both 120m variants use 2048. The unfused 120m row combines Q/K/V/O as `(48,768,768)`. I instantiated the actual 120m Transformer and Muon optimizer on the meta device and read its group order as `(12,2304,768)` fused QKV, `(12,768,768)` output, `(24,2048,768)` gate/up and `(12,768,2048)` down. These cover 84,934,656 matrix elements in a 123,587,328-parameter model. Only the fused row has the shapes the optimizer invokes. Its graph times agree with the training polar segment to within 1.5% for the five common methods at both budgets. Input spectra, interleaved momentum/layout work, cache history, output lifetimes and event boundaries differ.
- **Interpolation.** Independent recomputation reproduced both SS3_54 estimates, 1020.86 graph and 955.69 eager, between measured widths 768 and 1024. The graph ratio at 1024 is 1.00005 [0.99971, 1.00039]. Both clocks first show an interval wholly above one at 1536. The 120m rows are excluded by the grid flag. The helper interpolates linearly in log width and log speedup and supports multiple crossings. CPU probes confirmed empty input, a plateau at one and two strict crossings behave as expected. A below→exactly-one→below touch emits the same width twice, while above→one→above emits no crossing. Duplicate widths would collapse in the dictionary. Nonpositive inputs invalidate the logarithms. These cases do not occur in the recorded grid. Treat the estimates as descriptive interpolation.
- **CLI.** A CPU probe stubbed the GPU run and intercepted filesystem writes. Bare `--quick` selects widths 64/128, two layers, two blocks and two iterations and resolves to `cycle3_c3_01_wide_crossover_quick.json`. Two blocks give partial Williams coverage. The existing quick test supplies its own output path. Bare `--widths 64` still resolves to the full canonical filename. An explicit quick output can also target it. `write_text` replaces an existing file directly. Requiring an explicit destination for noncanonical settings would protect a checked-in full run.

[Benchmark](../research/polar_opt/bench_wide_crossover.py) · [Capture and scheduling helpers](../research/polar_opt/bench_cuda_graph.py#L44) · [Training optimizer](../research/polar_opt/harness_tier2_real.py#L232) · [Wide raw timings](../results/cycle3_c3_01_wide_crossover.json)

## Part A · Latency and speedup

Each method/clock/row median uses 70 event samples. Recomputed medians agree within 0.0000051 ms and both stored reference ratios within 0.000004× across all entries. The source rounds individual samples to five decimal places after computing the original summaries. Speedup is reference median / method median. Brackets give pointwise 95% percentile bootstrap intervals. I drew 20,000 sets of 14 whole blocks with replacement (NumPy PCG64, seed 301, one shared 20,000 × 14 draw matrix). Each draw preserves the five consecutive samples and uses matching block indices across methods. Intervals describe timing variation within this session. Hardware, process, width order and seed were fixed.

### CUDA graph replay

| Row | Method | Median ms | × vs PS3 Blackwell [95% CI] | × vs PE compiled [95% CI] |
|---|---|---|---|---|
| w512 | pe_eager | 11.20766 | 0.32339 [0.32304, 0.32355] | 0.96291 [0.96260, 0.96325] |
| w512 | pe_compiled | 10.79194 | 0.33585 [0.33552, 0.33596] | 1.00000 [1.00000, 1.00000] |
| w512 | ps3_eager | 6.61606 | 0.54783 [0.54691, 0.54833] | 1.63117 [1.62915, 1.63246] |
| w512 | ps3_blackwell | 3.62445 | 1.00000 [1.00000, 1.00000] | 2.97754 [2.97655, 2.98049] |
| w512 | ss3_54 | 3.71098 | 0.97668 [0.97572, 0.97790] | 2.90811 [2.90731, 2.91133] |
| w512 | ss3_53 | 3.55738 | 1.01885 [1.01784, 1.02003] | 3.03368 [3.03194, 3.03718] |
| w512 | ss3_35 | 3.50208 | 1.03494 [1.03390, 1.03630] | 3.08158 [3.07948, 3.08519] |
| w768 | pe_eager | 25.58771 | 0.41676 [0.41669, 0.41691] | 0.87841 [0.87832, 0.87854] |
| w768 | pe_compiled | 22.47662 | 0.47445 [0.47436, 0.47460] | 1.00000 [1.00000, 1.00000] |
| w768 | ps3_eager | 22.59046 | 0.47206 [0.47199, 0.47219] | 0.99496 [0.99472, 0.99522] |
| w768 | ps3_blackwell | 10.66394 | 1.00000 [1.00000, 1.00000] | 2.10772 [2.10704, 2.10809] |
| w768 | ss3_54 | 10.71206 | 0.99551 [0.99513, 0.99584] | 2.09825 [2.09756, 2.09875] |
| w768 | ss3_53 | 10.26765 | 1.03860 [1.03838, 1.03890] | 2.18907 [2.18846, 2.18961] |
| w768 | ss3_35 | 10.15603 | 1.05001 [1.04969, 1.05053] | 2.21313 [2.21238, 2.21375] |
| w768_120m | pe_eager | 24.62823 | 0.42314 [0.42280, 0.42358] | 0.88420 [0.88384, 0.88514] |
| w768_120m | pe_compiled | 21.77636 | 0.47856 [0.47817, 0.47879] | 1.00000 [1.00000, 1.00000] |
| w768_120m | ps3_eager | 22.23309 | 0.46873 [0.46834, 0.46897] | 0.97946 [0.97880, 0.97989] |
| w768_120m | ps3_blackwell | 10.42125 | 1.00000 [1.00000, 1.00000] | 2.08961 [2.08861, 2.09132] |
| w768_120m | ss3_54 | 10.48064 | 0.99433 [0.99341, 0.99531] | 2.07777 [2.07689, 2.07955] |
| w768_120m | ss3_53 | 10.02240 | 1.03980 [1.03797, 1.04048] | 2.17277 [2.16996, 2.17426] |
| w768_120m | ss3_35 | 9.91334 | 1.05124 [1.05006, 1.05220] | 2.19667 [2.19449, 2.19886] |
| w768_120m_harness | pe_eager | 22.13859 | 0.41501 [0.41459, 0.41533] | 0.87901 [0.87882, 0.87915] |
| w768_120m_harness | pe_compiled | 19.46010 | 0.47214 [0.47164, 0.47250] | 1.00000 [1.00000, 1.00000] |
| w768_120m_harness | ps3_eager | 16.38092 | 0.56089 [0.55984, 0.56164] | 1.18797 [1.18600, 1.18893] |
| w768_120m_harness | ps3_blackwell | 9.18784 | 1.00000 [1.00000, 1.00000] | 2.11803 [2.11638, 2.12027] |
| w768_120m_harness | ss3_54 | 9.17863 | 1.00100 [0.99944, 1.00206] | 2.12015 [2.11793, 2.12203] |
| w768_120m_harness | ss3_53 | 8.83353 | 1.04011 [1.03840, 1.04149] | 2.20298 [2.20080, 2.20505] |
| w768_120m_harness | ss3_35 | 8.74701 | 1.05040 [1.04916, 1.05204] | 2.22477 [2.22281, 2.22711] |
| w1024 | pe_eager | 54.67802 | 0.38532 [0.38522, 0.38545] | 0.89135 [0.89116, 0.89148] |
| w1024 | pe_compiled | 48.73728 | 0.43229 [0.43217, 0.43242] | 1.00000 [1.00000, 1.00000] |
| w1024 | ps3_eager | 47.11906 | 0.44714 [0.44700, 0.44726] | 1.03434 [1.03405, 1.03452] |
| w1024 | ps3_blackwell | 21.06880 | 1.00000 [1.00000, 1.00000] | 2.31324 [2.31255, 2.31389] |
| w1024 | ss3_54 | 21.06778 | 1.00005 [0.99971, 1.00039] | 2.31336 [2.31279, 2.31374] |
| w1024 | ss3_53 | 20.35609 | 1.03501 [1.03391, 1.03578] | 2.39424 [2.39158, 2.39592] |
| w1024 | ss3_35 | 20.01716 | 1.05254 [1.05223, 1.05287] | 2.43477 [2.43408, 2.43527] |
| w1536 | pe_eager | 162.93069 | 0.40619 [0.40537, 0.40672] | 0.92047 [0.91885, 0.92194] |
| w1536 | pe_compiled | 149.97243 | 0.44128 [0.44066, 0.44187] | 1.00000 [1.00000, 1.00000] |
| w1536 | ps3_eager | 135.21510 | 0.48944 [0.48862, 0.49000] | 1.10914 [1.10749, 1.11070] |
| w1536 | ps3_blackwell | 66.18012 | 1.00000 [1.00000, 1.00000] | 2.26613 [2.26310, 2.26932] |
| w1536 | ss3_54 | 65.35220 | 1.01267 [1.01168, 1.01395] | 2.29483 [2.29186, 2.29845] |
| w1536 | ss3_53 | 62.50906 | 1.05873 [1.05787, 1.06008] | 2.39921 [2.39639, 2.40251] |
| w1536 | ss3_35 | 61.99800 | 1.06746 [1.06636, 1.06891] | 2.41899 [2.41569, 2.42330] |
| w2048 | pe_eager | 360.96535 | 0.40877 [0.40841, 0.40924] | 0.93428 [0.93356, 0.93534] |
| w2048 | pe_compiled | 337.24179 | 0.43752 [0.43702, 0.43791] | 1.00000 [1.00000, 1.00000] |
| w2048 | ps3_eager | 290.63680 | 0.50768 [0.50720, 0.50802] | 1.16035 [1.15919, 1.16157] |
| w2048 | ps3_blackwell | 147.55146 | 1.00000 [1.00000, 1.00000] | 2.28559 [2.28358, 2.28820] |
| w2048 | ss3_54 | 144.72909 | 1.01950 [1.01828, 1.02026] | 2.33016 [2.32787, 2.33233] |
| w2048 | ss3_53 | 138.29528 | 1.06693 [1.06631, 1.06754] | 2.43856 [2.43658, 2.44114] |
| w2048 | ss3_35 | 137.45664 | 1.07344 [1.07286, 1.07403] | 2.45344 [2.45145, 2.45603] |

### Eager CUDA events

| Row | Method | Median ms | × vs PS3 Blackwell [95% CI] | × vs PE compiled [95% CI] |
|---|---|---|---|---|
| w512 | pe_eager | 11.24301 | 0.32297 [0.32263, 0.32316] | 0.96286 [0.96214, 0.96311] |
| w512 | pe_compiled | 10.82542 | 0.33542 [0.33519, 0.33560] | 1.00000 [1.00000, 1.00000] |
| w512 | ps3_eager | 6.61912 | 0.54858 [0.54760, 0.54902] | 1.63548 [1.63337, 1.63629] |
| w512 | ps3_blackwell | 3.63110 | 1.00000 [1.00000, 1.00000] | 2.98131 [2.97976, 2.98335] |
| w512 | ss3_54 | 3.72712 | 0.97424 [0.97363, 0.97515] | 2.90450 [2.90412, 2.90742] |
| w512 | ss3_53 | 3.57171 | 1.01663 [1.01575, 1.01707] | 3.03088 [3.02836, 3.03151] |
| w512 | ss3_35 | 3.51523 | 1.03296 [1.03203, 1.03368] | 3.07958 [3.07802, 3.08161] |
| w768 | pe_eager | 25.66349 | 0.41691 [0.41667, 0.41705] | 0.87940 [0.87906, 0.87955] |
| w768 | pe_compiled | 22.56845 | 0.47408 [0.47384, 0.47429] | 1.00000 [1.00000, 1.00000] |
| w768 | ps3_eager | 22.68507 | 0.47165 [0.47138, 0.47186] | 0.99486 [0.99448, 0.99519] |
| w768 | ps3_blackwell | 10.69936 | 1.00000 [1.00000, 1.00000] | 2.10933 [2.10843, 2.11041] |
| w768 | ss3_54 | 10.75507 | 0.99482 [0.99410, 0.99556] | 2.09840 [2.09727, 2.09954] |
| w768 | ss3_53 | 10.30861 | 1.03791 [1.03735, 1.03856] | 2.18928 [2.18845, 2.19030] |
| w768 | ss3_35 | 10.19238 | 1.04974 [1.04902, 1.05041] | 2.21425 [2.21320, 2.21529] |
| w768_120m | pe_eager | 24.69683 | 0.42296 [0.42284, 0.42306] | 0.88455 [0.88433, 0.88463] |
| w768_120m | pe_compiled | 21.84550 | 0.47817 [0.47804, 0.47833] | 1.00000 [1.00000, 1.00000] |
| w768_120m | ps3_eager | 22.30765 | 0.46826 [0.46814, 0.46836] | 0.97928 [0.97893, 0.97954] |
| w768_120m | ps3_blackwell | 10.44582 | 1.00000 [1.00000, 1.00000] | 2.09132 [2.09059, 2.09187] |
| w768_120m | ss3_54 | 10.49906 | 0.99493 [0.99465, 0.99528] | 2.08071 [2.07996, 2.08136] |
| w768_120m | ss3_53 | 10.05281 | 1.03909 [1.03854, 1.03930] | 2.17307 [2.17193, 2.17342] |
| w768_120m | ss3_35 | 9.93997 | 1.05089 [1.05054, 1.05119] | 2.19774 [2.19693, 2.19822] |
| w768_120m_harness | pe_eager | 22.18906 | 0.41479 [0.41468, 0.41503] | 0.87974 [0.87964, 0.87985] |
| w768_120m_harness | pe_compiled | 19.52051 | 0.47149 [0.47138, 0.47174] | 1.00000 [1.00000, 1.00000] |
| w768_120m_harness | ps3_eager | 16.42496 | 0.56035 [0.56016, 0.56060] | 1.18847 [1.18802, 1.18865] |
| w768_120m_harness | ps3_blackwell | 9.20369 | 1.00000 [1.00000, 1.00000] | 2.12094 [2.11980, 2.12141] |
| w768_120m_harness | ss3_54 | 9.20576 | 0.99978 [0.99938, 1.00033] | 2.12047 [2.11959, 2.12116] |
| w768_120m_harness | ss3_53 | 8.85760 | 1.03907 [1.03879, 1.03963] | 2.20381 [2.20307, 2.20432] |
| w768_120m_harness | ss3_35 | 8.77158 | 1.04926 [1.04901, 1.05010] | 2.22543 [2.22505, 2.22644] |
| w1024 | pe_eager | 54.74099 | 0.38604 [0.38530, 0.38836] | 0.89150 [0.89140, 0.89163] |
| w1024 | pe_compiled | 48.80179 | 0.43302 [0.43220, 0.43563] | 1.00000 [1.00000, 1.00000] |
| w1024 | ps3_eager | 47.17324 | 0.44797 [0.44714, 0.45063] | 1.03452 [1.03435, 1.03472] |
| w1024 | ps3_blackwell | 21.13209 | 1.00000 [1.00000, 1.00000] | 2.30937 [2.29554, 2.31375] |
| w1024 | ss3_54 | 21.09748 | 1.00164 [0.99720, 1.00757] | 2.31316 [2.30297, 2.31377] |
| w1024 | ss3_53 | 20.41549 | 1.03510 [1.03308, 1.04125] | 2.39043 [2.38872, 2.39096] |
| w1024 | ss3_35 | 20.04431 | 1.05427 [1.05235, 1.06050] | 2.43470 [2.43391, 2.43517] |
| w1536 | pe_eager | 163.44985 | 0.40624 [0.40557, 0.40702] | 0.92029 [0.91934, 0.92213] |
| w1536 | pe_compiled | 150.42169 | 0.44142 [0.44080, 0.44181] | 1.00000 [1.00000, 1.00000] |
| w1536 | ps3_eager | 135.38828 | 0.49043 [0.48990, 0.49082] | 1.11104 [1.10989, 1.11228] |
| w1536 | ps3_blackwell | 66.39914 | 1.00000 [1.00000, 1.00000] | 2.26542 [2.26340, 2.26860] |
| w1536 | ss3_54 | 65.60205 | 1.01215 [1.01107, 1.01265] | 2.29294 [2.29120, 2.29555] |
| w1536 | ss3_53 | 62.71079 | 1.05882 [1.05777, 1.06075] | 2.39866 [2.39622, 2.40412] |
| w1536 | ss3_35 | 62.25561 | 1.06656 [1.06535, 1.06847] | 2.41619 [2.41370, 2.42136] |
| w2048 | pe_eager | 361.49300 | 0.40853 [0.40841, 0.40874] | 0.93406 [0.93372, 0.93454] |
| w2048 | pe_compiled | 337.65581 | 0.43737 [0.43729, 0.43749] | 1.00000 [1.00000, 1.00000] |
| w2048 | ps3_eager | 290.80254 | 0.50783 [0.50755, 0.50815] | 1.16112 [1.16040, 1.16177] |
| w2048 | ps3_blackwell | 147.67933 | 1.00000 [1.00000, 1.00000] | 2.28641 [2.28576, 2.28683] |
| w2048 | ss3_54 | 144.85709 | 1.01948 [1.01909, 1.01971] | 2.33096 [2.33008, 2.33143] |
| w2048 | ss3_53 | 138.30861 | 1.06775 [1.06750, 1.06797] | 2.44132 [2.44053, 2.44175] |
| w2048 | ss3_35 | 137.49246 | 1.07409 [1.07354, 1.07431] | 2.45581 [2.45444, 2.45625] |

### w512 sample distribution

Tukey outliers fall outside [Q1 − 1.5 IQR, Q3 + 1.5 IQR]. Counts use five-decimal stored samples. Block range is the smallest and largest of the 14 five-sample medians, in ms.

| Method | Clock | Min | Q1 | Median | Q3 | IQR | Max | Outliers low/high | Block range |
|---|---|---|---|---|---|---|---|---|---|
| pe_eager | cuda_graph | 11.1749 | 11.1995 | 11.2077 | 11.2138 | 0.0143 | 11.2517 | 1/3 | 11.1852–11.2179 |
| pe_eager | eager_event | 11.2216 | 11.2396 | 11.2430 | 11.2928 | 0.0532 | 25.0620 | 0/12 | 11.2339–11.3777 |
| pe_compiled | cuda_graph | 10.7735 | 10.7889 | 10.7919 | 10.7950 | 0.0061 | 10.8022 | 5/0 | 10.7878–10.7981 |
| pe_compiled | eager_event | 10.8145 | 10.8230 | 10.8254 | 10.8288 | 0.0058 | 11.1125 | 0/1 | 10.8206–10.8293 |
| ps3_eager | cuda_graph | 6.5956 | 6.6099 | 6.6161 | 6.6263 | 0.0164 | 7.2207 | 0/10 | 6.6038–6.6847 |
| ps3_eager | eager_event | 6.6034 | 6.6137 | 6.6191 | 6.6612 | 0.0475 | 7.1766 | 0/11 | 6.6075–6.7885 |
| ps3_blackwell | cuda_graph | 3.6126 | 3.6209 | 3.6245 | 3.6270 | 0.0061 | 3.8251 | 0/4 | 3.6188–3.8162 |
| ps3_blackwell | eager_event | 3.6204 | 3.6270 | 3.6311 | 3.6336 | 0.0066 | 3.6419 | 0/0 | 3.6209–3.6372 |
| ss3_54 | cuda_graph | 3.6956 | 3.7051 | 3.7110 | 3.7145 | 0.0094 | 3.9301 | 0/5 | 3.6987–3.8957 |
| ss3_54 | eager_event | 3.7069 | 3.7212 | 3.7271 | 3.7294 | 0.0082 | 4.0528 | 1/2 | 3.7192–3.7294 |
| ss3_53 | cuda_graph | 3.5410 | 3.5533 | 3.5574 | 3.5615 | 0.0082 | 3.7304 | 1/1 | 3.5451–3.5635 |
| ss3_53 | eager_event | 3.5635 | 3.5697 | 3.5717 | 3.5761 | 0.0064 | 3.9956 | 0/5 | 3.5680–3.7931 |
| ss3_35 | cuda_graph | 3.4816 | 3.4980 | 3.5021 | 3.5041 | 0.0061 | 3.7493 | 5/3 | 3.4847–3.6803 |
| ss3_35 | eager_event | 3.5062 | 3.5123 | 3.5152 | 3.5185 | 0.0061 | 3.9467 | 0/1 | 3.5103–3.5215 |

The one-minute host load rose from 6.57 to 8.67 on 16 logical CPUs during w512 and remained elevated at the start of w768. A load average counts runnable and uninterruptible tasks. It does not measure GPU occupancy or prove host saturation. The timings do show tails. PE eager reached 25.062 ms against an 11.243 ms eager median. Even graph timings have slot-level excursions, including a 3.816 ms block median for PS3 Blackwell against its 3.6245 ms pooled median. Deleting each whole block in turn leaves w512 SS3_53 vs Blackwell at 1.01843–1.01900 graph and 1.01633–1.01663 eager. For SS3_54 those ranges are 0.97628–0.97709 and 0.97406–0.97444. The ranking is robust for these recorded medians. Retain w512 with a host-load caveat. An idle-host repeat matters for tail latency, tiny differences or extrapolation to another session. The existing evidence suffices for the directional w512 comparison.

### Transfer to the training timer

Each training cell is the median of the run-level polar_ortho_ms medians. Pilots pool the measured LRs for that optimizer. A dash means the optimizer was absent. All values are ms. The training timer sums the groups within a step, then takes the median over accepted steps after step one.

| Method | Unfused graph | Fused graph | Fused eager | Pilot | Fixed tokens | Fixed wall |
|---|---|---|---|---|---|---|
| pe_compiled | 21.7764 | 19.4601 | 19.5205 | 19.4085 | 19.4300 | 19.4255 |
| ps3_blackwell | 10.4213 | 9.1878 | 9.2037 | 9.0596 | 9.0624 | 9.0603 |
| ps3_eager | 22.2331 | 16.3809 | 16.4250 | — | 16.3574 | 16.3086 |
| ss3_53 | 10.0224 | 8.8335 | 8.8576 | — | 8.7351 | 8.7364 |
| ss3_54 | 10.4806 | 9.1786 | 9.2058 | — | 9.0870 | 9.0849 |

## Part B · Seed-0 pilot

Every pilot accepted 128 steps (4,194,304 tokens). AdamW and the PS3 Blackwell Muon pilot each compared four LRs. Selection uses validation loss.

| Optimizer | LR | Validation loss | Test loss | Median step ms |
|---|---|---|---|---|
| adamw | 0.0003 | 6.783088 | 6.818475 | 369.3166 |
| adamw | 0.0006 | 6.919793 | 6.954639 | 368.8198 |
| adamw | 0.001 | 6.606958 | 6.639108 | 369.3558 |
| adamw | 0.002 | 6.707936 | 6.740904 | 369.3175 |
| ps3_blackwell | 0.005 | 6.163218 | 6.202695 | 380.8287 |
| ps3_blackwell | 0.01 | 6.117003 | 6.156536 | 380.4354 |
| ps3_blackwell | 0.02 | 6.075765 | 6.113101 | 380.8366 |
| ps3_blackwell | 0.04 | 6.108757 | 6.146634 | 380.8804 |
| pe_compiled | 0.02 | 6.072853 | 6.110736 | 391.4522 |

AdamW selected 0.001. Its curve worsens from 0.0003 to 0.0006 before it improves at 0.001 and worsens again at 0.002. An interior single-seed minimum supports this grid choice, with limited evidence for an optimum. Muon selected 0.02 on PS3 Blackwell. PE and both SS candidates inherit this LR. PE ran once at 0.02. Its stored median was 391.452175 ms over 127 timed steps after the first. The artifact omits individual step durations. Using that median gives 256 × 0.391452175 = 100.21176 s, rounded to 100 s. PE accepted 255 steps on every wall seed and 256 on every token seed. Integer rounding and the last-step rejection make this outcome consistent with the calibration.

## Part C · Independent seed pairing and budget checks

| Seed | Initial weights fingerprint | Batch-start fingerprint | Matching pilot or confirmation runs |
|---|---|---|---|
| 0 | 7401cff643fe5a57 | c799a807f454d68f | 9 |
| 1 | 74a0bd19147b39be | b9f91c01f18ce738 | 11 |
| 2 | 3b55f66d5f4d4ac5 | f32172b52cfa7de3 | 11 |
| 3 | 3ebcf8ac9ea46e96 | e88efd98cfc4e5e9 | 11 |

I regenerated the initial CPU state and recomputed the first three batches’ start-offset digest from the shard length for each seed. The stored 64-bit SHA-256 prefixes match exactly across methods and budgets. The harness reloads the same seed state after compilation and restarts the seeded stream. This pairs the examples through the last shared step. The fingerprint itself covers three steps of offsets. Training consumes a different-length prefix when accepted step counts differ.

### 8,388,608 fixed tokens

| Method | Seed | Steps | Train s | Tokens/s | Val loss | Test loss |
|---|---|---|---|---|---|---|
| adamw | 1 | 256 | 94.51024 | 88758.72 | 6.267132 | 6.296240 |
| adamw | 2 | 256 | 94.53888 | 88731.83 | 6.059060 | 6.085163 |
| adamw | 3 | 256 | 94.54517 | 88725.93 | 6.440919 | 6.466695 |
| pe_compiled | 1 | 256 | 100.34636 | 83596.53 | 5.444771 | 5.464784 |
| pe_compiled | 2 | 256 | 100.31732 | 83620.73 | 5.461209 | 5.478748 |
| pe_compiled | 3 | 256 | 100.37433 | 83573.24 | 5.440043 | 5.458134 |
| ps3_blackwell | 1 | 256 | 97.52579 | 86014.25 | 5.445104 | 5.464755 |
| ps3_blackwell | 2 | 256 | 97.50153 | 86035.65 | 5.460119 | 5.478610 |
| ps3_blackwell | 3 | 256 | 97.52656 | 86013.58 | 5.445011 | 5.463758 |
| ps3_eager | 1 | 256 | 99.38332 | 84406.60 | 5.447864 | 5.467180 |
| ps3_eager | 2 | 256 | 99.41014 | 84383.83 | 5.454396 | 5.473179 |
| ps3_eager | 3 | 256 | 99.41112 | 84383.00 | 5.455367 | 5.474741 |
| ss3_53 | 1 | 256 | 97.44607 | 86084.62 | 5.442907 | 5.463576 |
| ss3_53 | 2 | 256 | 97.42813 | 86100.47 | 5.447434 | 5.465432 |
| ss3_53 | 3 | 256 | 97.40712 | 86119.05 | 5.448473 | 5.468525 |
| ss3_54 | 1 | 256 | 97.39955 | 86125.74 | 5.442435 | 5.462761 |
| ss3_54 | 2 | 256 | 97.53662 | 86004.70 | 5.450102 | 5.467688 |
| ss3_54 | 3 | 256 | 97.51436 | 86024.34 | 5.442386 | 5.461681 |

### 100 s accepted training compute

| Method | Seed | Steps | Train s | Tokens/s | Val loss | Test loss |
|---|---|---|---|---|---|---|
| pe_compiled | 1 | 255 | 99.85702 | 83678.04 | 5.463782 | 5.483603 |
| pe_compiled | 2 | 255 | 99.87067 | 83666.61 | 5.472290 | 5.491422 |
| pe_compiled | 3 | 255 | 99.87684 | 83661.44 | 5.455575 | 5.474903 |
| ps3_blackwell | 1 | 262 | 99.71070 | 86101.25 | 5.442485 | 5.461064 |
| ps3_blackwell | 2 | 262 | 99.71427 | 86098.17 | 5.437267 | 5.454476 |
| ps3_blackwell | 3 | 262 | 99.69302 | 86116.52 | 5.441401 | 5.459071 |
| ps3_eager | 1 | 257 | 99.75296 | 84422.32 | 5.451816 | 5.472248 |
| ps3_eager | 2 | 257 | 99.67504 | 84488.32 | 5.446664 | 5.462979 |
| ps3_eager | 3 | 257 | 99.70038 | 84466.84 | 5.461669 | 5.481031 |
| ss3_53 | 1 | 263 | 99.98472 | 86193.01 | 5.434151 | 5.451563 |
| ss3_53 | 2 | 263 | 99.99883 | 86180.85 | 5.427319 | 5.443156 |
| ss3_53 | 3 | 262 | 99.63920 | 86163.04 | 5.427642 | 5.445634 |
| ss3_54 | 1 | 262 | 99.73373 | 86081.37 | 5.436416 | 5.456054 |
| ss3_54 | 2 | 262 | 99.72242 | 86091.13 | 5.431035 | 5.449189 |
| ss3_54 | 3 | 262 | 99.71081 | 86101.16 | 5.434017 | 5.452976 |

All 18 token trials and all 15 wall trials reported finite final losses and zero divergences. Every token run accepted 256 × 32,768 tokens. All 15 wall runs have max(0, train_seconds − 100) = 0.00000 s. Adding the discarded step exceeds 100 in every case. The overshoot field is hard-coded to zero by the harness. The independent check uses train_seconds. The harness restores model weights after the first over-budget step. The 100 s includes synchronized training steps. Batch preparation, snapshots, evaluation, warmup and the discarded step sit outside the accepted budget.

## Training means by method

Each confirmatory row averages three runs with equal seed weight. The earlier per-seed table retains their dispersion. I checked stored pilot and confirmatory grid means against trial records, and token/wall PE-reference means and intervals against the independent paired computation.

| Budget | Method | Mean steps | Mean train s | Mean tokens/s | Mean val loss | Mean test loss |
|---|---|---|---|---|---|---|
| tokens | adamw | 256.00 | 94.5314 | 88738.83 | 6.255704 | 6.282699 |
| tokens | pe_compiled | 256.00 | 100.3460 | 83596.84 | 5.448675 | 5.467222 |
| tokens | ps3_blackwell | 256.00 | 97.5180 | 86021.16 | 5.450078 | 5.469041 |
| tokens | ps3_eager | 256.00 | 99.4015 | 84391.14 | 5.452542 | 5.471700 |
| tokens | ss3_53 | 256.00 | 97.4271 | 86101.38 | 5.446271 | 5.465844 |
| tokens | ss3_54 | 256.00 | 97.4835 | 86051.59 | 5.444974 | 5.464043 |
| wall | pe_compiled | 255.00 | 99.8682 | 83668.70 | 5.463882 | 5.483310 |
| wall | ps3_blackwell | 262.00 | 99.7060 | 86105.32 | 5.440384 | 5.458204 |
| wall | ps3_eager | 257.00 | 99.7095 | 84459.16 | 5.453383 | 5.472086 |
| wall | ss3_53 | 262.67 | 99.8743 | 86178.96 | 5.429704 | 5.446785 |
| wall | ss3_54 | 262.00 | 99.7223 | 86091.22 | 5.433823 | 5.452740 |

## Interpretation of the wall-budget result

Across all optimizer pairs, no fixed-token test-loss contrast passes Holm correction at 0.05. At fixed wall, SS3_54 vs PS3 Blackwell is the only test-loss pair to pass that correction. Neither switch’s test-loss comparison with compiled PE survives. Throughput gains over compiled PE survive even the joint correction for all three Blackwell candidates under both budgets.

**SS3_54 vs PS3 Blackwell.** Both accepted 262 steps on every seed and saw the same batch prefix. SS3_54’s throughput ratio was 0.999836. Extra steps cannot explain its lower test loss. The mean paired difference was −0.005464, and the three differences had SD 0.000563. With fixed tokens the mean was similar (−0.004998) but the SD was 0.005131. The small raw wall p-value reflects much tighter agreement across these three seeds. All three validation differences also favored SS3_54. The test and validation results survive separate ten-pair Holm corrections (0.03520 and 0.03954), and fall short in the joint 30-test family (0.08096 and 0.08700). Both losses were evaluated on the same three trained model pairs. Their agreement provides supporting evidence, with no additional seed replication.

**SS3_53 vs PS3 Blackwell.** SS3_53 accepted 263, 263 and 262 steps against 262 each (mean 262.67 vs 262.00). The throughput ratio was 1.000855. Test loss differences were −0.009501, −0.011320 and −0.013436. Seed 3 has the largest gain with equal steps. Validation follows the same pattern. The two longer runs consumed one extra 32,768-token batch each. Their final losses combine that batch with optimizer and LR-trajectory effects. The JSON provides final validation/test losses only, so those components cannot be apportioned numerically. At fixed tokens the same third seed reverses sign on both losses. The paired wall test SD was 0.001969 versus 0.009141 at fixed tokens. Mean wall test loss improved by 0.011419 with raw p = 0.00977, all-pairs Holm p = 0.08793, and joint p = 0.20517. The narrower six-contrast test-loss family yields 0.04885. Choosing that family after inspecting results would overstate confirmation.

**What transfers.** Against compiled PE, the Blackwell candidates got about seven or eight extra steps in 100 s. That gives them more tokens as well as different updates. Against PS3 Blackwell, the equal-step SS3_54 result and the equal-step SS3_53 seed show why step count alone is inadequate. The wall harness schedules LR against elapsed accepted compute with a 6.25% warmup fraction. The token harness schedules by step with a 20-step warmup. Matching final step counts cannot match the entire wall-mode LR trajectory. All methods shared the same 1,048,576-token evaluation prefixes on each held-out shard. The data support specific observed seed-paired loss differences. They do not identify separate causal quality, schedule and step-count contributions, establish loss equivalence, or show wider-model training transfer.

The paired t inference assumes approximately normal differences across independent seeds. Three observations give little leverage to check that assumption. A distribution-free two-sided paired sign test with all three differences in the same direction gives p = 0.25. Before promoting either switch on training quality, use fresh seeds, predeclare the loss endpoint and contrast family, and compare validation/test losses at common step checkpoints under a common LR schedule. Keep the wall-budget comparison as a separate training outcome.

[AdamW pilot](../results/cycle3_c3_01_120m_pilot_adamw.json) · [Muon pilot](../results/cycle3_c3_01_120m_pilot_muon.json) · [PE calibration](../results/cycle3_c3_01_120m_pilot_pe_compiled.json) · [Fixed-token trials](../results/cycle3_c3_01_120m_real_tokens.json) · [Fixed-wall trials](../results/cycle3_c3_01_120m_real_wall.json) · [Training clock and rollback](../research/polar_opt/harness_tier2_real.py#L571)

## Part C · Paired contrasts and multiplicity

Each contrast uses the three within-seed differences (or within-seed throughput ratios). Positive speed changes and negative loss changes favor the candidate. CI = mean ± 4.30265273 × sample SD / √3. The two-sided paired t-test uses df = 2 and null 0 for loss or 1 for throughput ratio. Throughput uses tokens / train_seconds. Holm p adjusts over every optimizer pair for one endpoint and budget, including AdamW at fixed tokens (15 token pairs, 10 wall pairs). Joint p also adjusts over all pairs and all three endpoints in that budget (45 or 30 hypotheses). Both use ordered p-values, descending multipliers and cumulative maxima. The CIs are unadjusted. The seed is the replicate. All seeds reuse the same evaluation shards.

### Fixed tokens

**Test loss difference**

| Candidate vs reference | Mean [95% CI] | Raw p | Holm p | Joint p |
|---|---|---|---|---|
| ps3_blackwell vs pe_compiled | 0.001819 [-0.006368, 0.010006] | 0.439927 | 1 | 1 |
| ps3_eager vs pe_compiled | 0.004478 [-0.023428, 0.032385] | 0.56128 | 1 | 1 |
| ss3_54 vs pe_compiled | -0.003179 [-0.021492, 0.015134] | 0.533032 | 1 | 1 |
| ss3_53 vs pe_compiled | -0.001378 [-0.030825, 0.028070] | 0.859094 | 1 | 1 |
| adamw vs pe_compiled | 0.815477 [0.314802, 1.316152] | 0.0197602 | 0.262638 | 0.560294 |
| ps3_eager vs ps3_blackwell | 0.002659 [-0.017735, 0.023052] | 0.631286 | 1 | 1 |
| ss3_54 vs ps3_blackwell | -0.004998 [-0.017744, 0.007748] | 0.233627 | 1 | 1 |
| ss3_53 vs ps3_blackwell | -0.003197 [-0.025903, 0.019510] | 0.606263 | 1 | 1 |
| adamw vs ps3_blackwell | 0.813658 [0.319830, 1.307485] | 0.0193225 | 0.262638 | 0.560294 |
| ss3_54 vs ps3_eager | -0.007657 [-0.019358, 0.004044] | 0.106393 | 0.957537 | 1 |
| ss3_53 vs ps3_eager | -0.005856 [-0.011060, -0.000652] | 0.0401138 | 0.401138 | 0.84239 |
| adamw vs ps3_eager | 0.810999 [0.337454, 1.284544] | 0.017923 | 0.262638 | 0.560294 |
| ss3_53 vs ss3_54 | 0.001801 [-0.009698, 0.013300] | 0.569812 | 1 | 1 |
| adamw vs ss3_54 | 0.818656 [0.336251, 1.301061] | 0.0182446 | 0.262638 | 0.560294 |
| adamw vs ss3_53 | 0.816855 [0.345578, 1.288132] | 0.0175092 | 0.262638 | 0.560294 |

**Validation loss difference**

| Candidate vs reference | Mean [95% CI] | Raw p | Holm p | Joint p |
|---|---|---|---|---|
| ps3_blackwell vs pe_compiled | 0.001403 [-0.006466, 0.009273] | 0.523122 | 1 | 1 |
| ps3_eager vs pe_compiled | 0.003868 [-0.023679, 0.031414] | 0.607162 | 1 | 1 |
| ss3_54 vs pe_compiled | -0.003700 [-0.020661, 0.013260] | 0.446992 | 1 | 1 |
| ss3_53 vs pe_compiled | -0.002403 [-0.030008, 0.025201] | 0.743959 | 1 | 1 |
| adamw vs pe_compiled | 0.807029 [0.305360, 1.308698] | 0.0202413 | 0.252414 | 0.560294 |
| ps3_eager vs ps3_blackwell | 0.002464 [-0.017517, 0.022446] | 0.64868 | 1 | 1 |
| ss3_54 vs ps3_blackwell | -0.005103 [-0.015673, 0.005466] | 0.173352 | 1 | 1 |
| ss3_53 vs ps3_blackwell | -0.003806 [-0.024159, 0.016546] | 0.505433 | 1 | 1 |
| adamw vs ps3_blackwell | 0.805626 [0.311306, 1.299946] | 0.0197365 | 0.252414 | 0.560294 |
| ss3_54 vs ps3_eager | -0.007568 [-0.019299, 0.004164] | 0.10899 | 0.980911 | 1 |
| ss3_53 vs ps3_eager | -0.006271 [-0.009099, -0.003443] | 0.0108069 | 0.162104 | 0.356629 |
| adamw vs ps3_eager | 0.803161 [0.328807, 1.277516] | 0.0183257 | 0.252414 | 0.560294 |
| ss3_53 vs ss3_54 | 0.001297 [-0.009722, 0.012316] | 0.662883 | 1 | 1 |
| adamw vs ss3_54 | 0.810729 [0.325919, 1.295540] | 0.0187739 | 0.252414 | 0.560294 |
| adamw vs ss3_53 | 0.809432 [0.335360, 1.283505] | 0.0180296 | 0.252414 | 0.560294 |

**Tokens/s ratio**

| Candidate vs reference | Mean [95% CI] | Raw p | Holm p | Joint p |
|---|---|---|---|---|
| ps3_blackwell vs pe_compiled | 1.029000 [1.028567, 1.029433] | 1.20313e-05 | 0.000156407 | 0.000517346 |
| ps3_eager vs pe_compiled | 1.009502 [1.008693, 1.010311] | 0.000391258 | 0.00176832 | 0.0133028 |
| ss3_54 vs pe_compiled | 1.029364 [1.027195, 1.031534] | 0.00029472 | 0.00176832 | 0.0106099 |
| ss3_53 vs pe_compiled | 1.029960 [1.028871, 1.031048] | 7.12973e-05 | 0.000641676 | 0.0027806 |
| adamw vs pe_compiled | 1.061509 [1.060668, 1.062351] | 1.0103e-05 | 0.000141443 | 0.000444534 |
| ps3_eager vs ps3_blackwell | 0.981051 [0.980419, 0.981683] | 6.01167e-05 | 0.000628757 | 0.00240467 |
| ss3_54 vs ps3_blackwell | 1.000354 [0.998239, 1.002469] | 0.546347 | 0.774798 | 1 |
| ss3_53 vs ps3_blackwell | 1.000933 [1.000296, 1.001569] | 0.0242727 | 0.0728182 | 0.560294 |
| adamw vs ps3_blackwell | 1.031593 [1.030875, 1.032311] | 2.79176e-05 | 0.000335011 | 0.00117254 |
| ss3_54 vs ps3_eager | 1.019676 [1.018157, 1.021194] | 0.00032154 | 0.00176832 | 0.0112539 |
| ss3_53 vs ps3_eager | 1.020266 [1.019388, 1.021143] | 0.00010119 | 0.000809516 | 0.0038452 |
| adamw vs ps3_eager | 1.051518 [1.051400, 1.051637] | 2.86331e-07 | 4.29497e-06 | 1.28849e-05 |
| ss3_53 vs ss3_54 | 1.000579 [0.998306, 1.002852] | 0.387399 | 0.774798 | 1 |
| adamw vs ss3_54 | 1.031228 [1.029765, 1.032692] | 0.000118577 | 0.000830039 | 0.00438735 |
| adamw vs ss3_53 | 1.030632 [1.029635, 1.031628] | 5.71597e-05 | 0.000628757 | 0.00234355 |

Restricting the test-loss family to the 7 PE-reference and SS-vs-Blackwell contrasts gives an adjusted p for SS3_54 vs Blackwell of 1 and for SS3_53 vs Blackwell of 1.

### Fixed wall

**Test loss difference**

| Candidate vs reference | Mean [95% CI] | Raw p | Holm p | Joint p |
|---|---|---|---|---|
| ps3_blackwell vs pe_compiled | -0.025106 [-0.051905, 0.001694] | 0.0563949 | 0.222268 | 0.501872 |
| ps3_eager vs pe_compiled | -0.011223 [-0.054164, 0.031717] | 0.377598 | 0.377598 | 0.742868 |
| ss3_54 vs pe_compiled | -0.030570 [-0.056615, -0.004525] | 0.0370447 | 0.222268 | 0.501872 |
| ss3_53 vs pe_compiled | -0.036525 [-0.062017, -0.011033] | 0.0253174 | 0.177222 | 0.43605 |
| ps3_eager vs ps3_blackwell | 0.013882 [-0.003813, 0.031577] | 0.0776758 | 0.222268 | 0.501872 |
| ss3_54 vs ps3_blackwell | -0.005464 [-0.006863, -0.004066] | 0.00352004 | 0.0352004 | 0.0809609 |
| ss3_53 vs ps3_blackwell | -0.011419 [-0.016312, -0.006527] | 0.00976996 | 0.0879296 | 0.205169 |
| ss3_54 vs ps3_eager | -0.019346 [-0.038318, -0.000375] | 0.0482177 | 0.222268 | 0.501872 |
| ss3_53 vs ps3_eager | -0.025302 [-0.047047, -0.003557] | 0.0376589 | 0.222268 | 0.501872 |
| ss3_53 vs ss3_54 | -0.005955 [-0.009501, -0.002409] | 0.0186191 | 0.148953 | 0.372382 |

**Validation loss difference**

| Candidate vs reference | Mean [95% CI] | Raw p | Holm p | Joint p |
|---|---|---|---|---|
| ps3_blackwell vs pe_compiled | -0.023498 [-0.049824, 0.002828] | 0.0616001 | 0.250418 | 0.501872 |
| ps3_eager vs pe_compiled | -0.010499 [-0.050024, 0.029025] | 0.371434 | 0.371434 | 0.742868 |
| ss3_54 vs pe_compiled | -0.030059 [-0.055202, -0.004917] | 0.035774 | 0.250418 | 0.501872 |
| ss3_53 vs pe_compiled | -0.034178 [-0.057492, -0.010864] | 0.024225 | 0.197423 | 0.43605 |
| ps3_eager vs ps3_blackwell | 0.012999 [-0.002639, 0.028637] | 0.0700625 | 0.250418 | 0.501872 |
| ss3_54 vs ps3_blackwell | -0.006561 [-0.008342, -0.004781] | 0.00395444 | 0.0395444 | 0.0869977 |
| ss3_53 vs ps3_blackwell | -0.010680 [-0.017600, -0.003760] | 0.0219359 | 0.197423 | 0.416782 |
| ss3_54 vs ps3_eager | -0.019560 [-0.036970, -0.002151] | 0.0402264 | 0.250418 | 0.501872 |
| ss3_53 vs ps3_eager | -0.023679 [-0.046038, -0.001320] | 0.0449412 | 0.250418 | 0.501872 |
| ss3_53 vs ss3_54 | -0.004119 [-0.009297, 0.001060] | 0.0758042 | 0.250418 | 0.501872 |

**Tokens/s ratio**

| Candidate vs reference | Mean [95% CI] | Raw p | Holm p | Joint p |
|---|---|---|---|---|
| ps3_blackwell vs pe_compiled | 1.029122 [1.028625, 1.029619] | 1.57429e-05 | 0.000141687 | 0.000456545 |
| ps3_eager vs pe_compiled | 1.009448 [1.008233, 1.010662] | 0.000890861 | 0.00356344 | 0.0213807 |
| ss3_54 vs pe_compiled | 1.028954 [1.028404, 1.029504] | 1.94881e-05 | 0.000155905 | 0.000545666 |
| ss3_53 vs pe_compiled | 1.030002 [1.029785, 1.030220] | 2.83698e-06 | 2.83698e-05 | 8.51093e-05 |
| ps3_eager vs ps3_blackwell | 0.980882 [0.979883, 0.981881] | 0.000147515 | 0.000885088 | 0.00383538 |
| ss3_54 vs ps3_blackwell | 0.999836 [0.999648, 1.000024] | 0.0643796 | 0.077718 | 0.501872 |
| ss3_53 vs ps3_blackwell | 1.000855 [1.000165, 1.001546] | 0.0334581 | 0.077718 | 0.501872 |
| ss3_54 vs ps3_eager | 1.019324 [1.018476, 1.020171] | 0.0001039 | 0.000727298 | 0.00280529 |
| ss3_53 vs ps3_eager | 1.020363 [1.019046, 1.021680] | 0.000225877 | 0.00112938 | 0.00564691 |
| ss3_53 vs ss3_54 | 1.001019 [1.000299, 1.001739] | 0.025906 | 0.077718 | 0.43605 |

Restricting the test-loss family to the 6 PE-reference and SS-vs-Blackwell contrasts gives an adjusted p for SS3_54 vs Blackwell of 0.0211202 and for SS3_53 vs Blackwell of 0.0488498.

### Step count and loss within seed

| Budget | Candidate vs PS3 Blackwell | Seed | Δ steps | Tokens/s ratio | Δ validation | Δ test |
|---|---|---|---|---|---|---|
| tokens | ss3_54 | 1 | 0 | 1.001296 | -0.002669 | -0.001994 |
| tokens | ss3_54 | 2 | 0 | 0.999640 | -0.010016 | -0.010922 |
| tokens | ss3_54 | 3 | 0 | 1.000125 | -0.002625 | -0.002077 |
| tokens | ss3_53 | 1 | 0 | 1.000818 | -0.002197 | -0.001179 |
| tokens | ss3_53 | 2 | 0 | 1.000753 | -0.012685 | -0.013178 |
| tokens | ss3_53 | 3 | 0 | 1.001226 | +0.003462 | +0.004766 |
| wall | ss3_54 | 1 | 0 | 0.999769 | -0.006069 | -0.005011 |
| wall | ss3_54 | 2 | 0 | 0.999918 | -0.006232 | -0.005287 |
| wall | ss3_54 | 3 | 0 | 0.999822 | -0.007384 | -0.006094 |
| wall | ss3_53 | 1 | 1 | 1.001066 | -0.008334 | -0.009501 |
| wall | ss3_53 | 2 | 1 | 1.000960 | -0.009948 | -0.011320 |
| wall | ss3_53 | 3 | 0 | 1.000540 | -0.013759 | -0.013436 |
