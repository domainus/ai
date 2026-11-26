# Cycle 4 C4-03 audit on the GeForce RTX 5080

**Verdict — PASS_WITH_CAVEATS.** Merge the benchmark and its four recorded artifacts as evidence for the tested protocols. Promote the wide-model throughput result and the negative common-checkpoint loss replication with the qualifications below. Promote long-K sequential tensor-core accumulation as the cause of the measured Gram drift. Retain ChunkedGram as an experimental mitigation. Withhold promotion of a general bit-exact MMA rule. The proposed rule explains the measured long-K diagonal bias, but fresh normal FP16 inputs give counterexamples.

## Ranked findings

| Severity | Finding | Disposition |
|---|---|---|
| HIGH | Canonical summary validation with explicit source objects can accept checkpoint statistics that differ from the data in the cited, correctly hashed file. The default disk-backed check rejected the same document. | Use the disk-backed check on these immutable blobs. Bind explicit objects to the parsed bytes whose digest was checked before relying on that overload for promotion. |
| HIGH | The proposed MMA rule disagrees on all-positive normal FP16 and BF16 diagonals, signed off-diagonals and FP32 exponent-limit cases. The product drop threshold also depends on the accumulator’s binade. | Promote the measured long-K drift mechanism and its diagonal predictions. Withhold unrestricted bit-exactness for full Gram matrices. |
| MEDIUM | A standalone canonical stage can assert a different code digest and still pass internal consistency checks. Drift validation also accepts an impossible minimum singular value for a raw cell that contributes no minimum to the headline. | Authenticate the source closure against the producing Git object and inspect raw scientific invariants. Both are independently checked below for the committed data. |
| MEDIUM | Model depth, hidden width, head configuration and parameter count change along the wide grid. SS3_54’s throughput advantage over PS3 at width 1536 misses the across-width correction. | Name the width, comparator and metric for each transfer claim. Use the fixed batch and training schedule when quoting these results. |
| MEDIUM | Identical checkpoint schedules can give different GPU training trajectories with the recorded numerical settings. The controls differ before evaluation. | Preserve the paired seed analysis. Scope bitwise replay to deterministic settings, where the checkpoint control passed. |
| LOW | Five checkpoint seeds include the three C3-01 seeds. The 384-step fixed-token LR schedule differs from the C3-01 wall schedule. | Record a negative common-checkpoint replication. The confidence intervals still allow a small loss benefit. |
| LOW | The small-shape mitigation timing includes Python dispatch and patch installation. Endpoint telemetry samples leave the within-run clock and power history unknown. | Keep production kernels as shipped. The long-K speedup and normal-shape bit identity have separate evidence below. |

## Provenance and integrity

The audit began at `59486d91078caca0d1bbacc72b4b34311f323e88` on the requested audit branch. Its parent is the producing code commit `eb31cc5d39df996d0792f05077b92c8cb7ea7623`. The parent of that commit is `e4eebfcfc4e35d941d628523bbce20a8172abef2`. The first diff adds only `research/polar_opt/bench_c4_03_wide_and_gram_drift.py` (1,749 lines) and `tests/test_c4_03_wide_and_gram_drift.py` (780 lines). The second adds only the four JSON files below. No resume logs or model checkpoint files occur in either diff. Git reports identical contents for both switch kernels, the common Blackwell kernels and the real-text Tier-2 harness across the full range.

A separate parser rejected duplicate keys and non-finite constants. All four files passed, including a serialize/parse equality check with `allow_nan=False`. All four report the full producing SHA above and a clean tracked tree. I traversed static research imports from the driver and hashed all 14 modules from that Git object, in sorted module-name order, with the name and NUL separator before each file. Worktree bytes, recorded module lists and all three stage code digests agreed. The code SHA-256 is `b10c2868c5e90daf070ee1627bf7c3082da381a21a5a544509461f7618029008`.

| File under `results/cycle4_c4_03_` prefix | Independent run hash | File SHA-256 |
|---|---|---|
| wide.json | `9509c98922418cf3` | `6247f32c5b8eccced8d57a068f6b754d73e3b0ef5b43fbd6d8267e49cc870c6d` |
| ckpt.json | `a477809b256e553c` | `6d6b92f03af9a577aa65066f335d1df38cd4a9f6b51fb361493195c98ae8b1b3` |
| drift.json | `487bc951a27d9451` | `a759166b68442ecc588ac8a122b887ed4f48c46189c15cbd55550ee8929db306` |
| summary.json | Aggregation of the three inputs | `ad7100c18388a0629bf9f5bf7c1769afbdde6327725ef80a465cabfc3f9d0396` |

The first three file digests match the summary source records. I independently rebuilt each 16-hex run hash from sorted-key JSON over schema, stage, quick flag, protocol, shard digests or null, code digest, Git SHA and the five environment fields. The recorded environment is PyTorch 2.14.1+cu130, CUDA 13.0, Triton 3.8.0, driver 615.78.08 and device capability 12.0. TF32 and both reduced-precision matmul reduction options were disabled.

Byte-level SHA-256 of the real shards matched `research.quant_ptq.micro_rotated_ptq.EXPECTED_SHA256` and both training artifacts.

| Shard | SHA-256 |
|---|---|
| train.bin | `015ab83ceb742555c2398963a0bd1f3c5502dc52cf70a22bf173f1571326ca3a` |
| val.bin | `371b64a537f7b1e306ea174a868b05932a254a98b7fbdbf4146648d93180b0b1` |
| test.bin | `4eb46f1424856849d2973923dbf3c886d37eb743cf22053fc7e1eea1f41ab92a` |

## Training replay

I replayed the statistics from all 51 recorded trials. I also regenerated the full initial-parameter fingerprints on CPU and the stream fingerprints from the independent NumPy window starts for all 14 configuration/seed blocks. Every optimizer matched its block. Each fingerprint changed across seeds within each configuration. Every row matched the seeded shuffle, its block position and chronological start/end order. Neither training stage resumed. The artifacts report zero divergences in 36 wide trials and zero in 15 checkpoint trials.

The independent calculation rebuilt per-step segment medians (first step excluded), total timed optimizer medians, total training time, tokens/s and median polar time divided by median step time. All reported per-optimizer means, sample SDs, metric vectors and checkpoint means agreed. Each contrast used the arithmetic mean of the per-seed ratios or differences. I checked all 75 wide and 18 checkpoint contrasts against SciPy, including the seed vectors, sample SD, SE, null, t statistic, marginal two-sided 95% interval and raw p-value. The 97.5th percentiles were 4.30265273 for df 2 and 2.77644511 for df 4. The comparison tolerance was 2e−9 relative plus 2e−13 absolute.

Independent Holm adjustment sorted raw p-values and took the capped running maximum of remaining-family-size times p. Wide has 15 within-configuration families of five and five joint metric families of 15. Checkpoint has six singleton run-metric families and four three-checkpoint loss families. Its five joint families have sizes 2, 2, 2, 6 and 6. Every adjusted p-value agreed. Joint adjustment is per metric. Confidence intervals in this report are marginal.

### Wide-model transfer

The table gives the paired candidate/reference throughput ratio. The last column is the joint Holm p-value across all 15 throughput contrasts.

| Width | Candidate / reference | Tokens/s ratio | 95% CI | Joint p |
|---:|---|---:|---|---:|
| 1536 | SS3_54 / compiled PE | 1.11993 | 1.11884–1.12101 | 0.0000616 |
| 1536 | SS3_53 / compiled PE | 1.12410 | 1.12319–1.12502 | 0.0000442 |
| 1536 | PS3 / compiled PE | 1.11651 | 1.11388–1.11913 | 0.000302 |
| 1536 | SS3_54 / PS3 | 1.00306 | 1.00023–1.00590 | 0.12993 |
| 1536 | SS3_53 / PS3 | 1.00680 | 1.00474–1.00887 | 0.02468 |
| 2048 | SS3_54 / compiled PE | 1.15084 | 1.14874–1.15293 | 0.000136 |
| 2048 | SS3_53 / compiled PE | 1.15525 | 1.15279–1.15772 | 0.000163 |
| 2048 | PS3 / compiled PE | 1.14585 | 1.14180–1.14991 | 0.000417 |
| 2048 | SS3_54 / PS3 | 1.00435 | 1.00259–1.00610 | 0.03471 |
| 2048 | SS3_53 / PS3 | 1.00820 | 1.00615–1.01026 | 0.02025 |

At widths 1536 and 2048, SS3_54 used 96.90% and 96.45% of PS3's polar time. SS3_53 used 92.67% and 92.48%. All four polar-time comparisons survived joint correction. Mean polar fractions for compiled PE, PS3, SS3_54 and SS3_53 were 17.39%, 8.21%, 7.98%, 7.66% at width 1536 and 21.30%, 10.20%, 9.87%, 9.51% at width 2048. No wide validation or test loss comparison passed joint correction. The smallest joint p-values were 0.323 and 0.110 respectively.

Every wide trial used 64 × 16,384 tokens, with 262,144 evaluation tokens per split. The width-768 anchor halves gradient accumulation from eight to four at the same microbatch size of eight and sequence length 512. This doubles the frequency of orthogonalization per token relative to the 120m checkpoint protocol. The mean PS3 polar fraction was 4.59% at the wide anchor and 2.38% at 120m. Depth fell from 12 to 6 to 4 over the wide grid. Total parameters rose from 123.6M to 247.2M to 303.8M. These measurements establish transfer to the two observed architectures under the fixed batch. A universal width crossover or a width-only scaling law would need additional controls.

### Five-seed checkpoint replication

Each optimizer trained once per seed. Steps 128, 256 and 384 belong to the same trajectory. Each evaluation scored 1,998,848 tokens from each split. At step 384 the trial had trained on 12,582,912 tokens, or 0.483874 of the training shard. The table uses SS3_54 minus PS3 loss. All six joint p-values are 1.000.

| Step | Split | Paired difference | 95% CI | Raw p |
|---:|---|---:|---|---:|
| 128 | Validation | −0.001949 | −0.011424 to +0.007525 | 0.5984 |
| 128 | Test | −0.002301 | −0.012229 to +0.007627 | 0.5550 |
| 256 | Validation | −0.000346 | −0.007829 to +0.007137 | 0.9041 |
| 256 | Test | +0.000190 | −0.007617 to +0.007998 | 0.9493 |
| 384 | Validation | −0.001257 | −0.008998 to +0.006483 | 0.6754 |
| 384 | Test | −0.000728 | −0.007906 to +0.006449 | 0.7921 |

All six SS3_53 loss contrasts also had joint p = 1. The earlier C3-01 three-seed wall result was −0.005464 test loss, with a wall-time LR schedule and 262 accepted steps. That effect size lies inside all three new SS3_54 test intervals. The nine optimizer/seed pairs from seeds 1–3 have exactly the same init and stream fingerprints in the two studies. Seeds 4 and 5 add independent seed blocks. Record that the advantage did not replicate at these common checkpoints. Equivalence and superiority both remain unestablished.

### Hardware telemetry and memory

| Stage | SM MHz | Memory MHz | GPU °C | Power W | GPU utilization | Host load range at 1 / 5 / 15 min |
|---|---|---:|---|---|---|---|
| Wide | 2835–2872 | 14801 | 43–66 | 116.01–296.07 | 0–100% | 2.19–2.76 / 2.30–2.50 / 2.25–2.36 |
| Checkpoint | 2835–2865 | 14801 | 45–68 | 201.11–299.20 | 0–100% | 2.19–3.05 / 2.50–2.85 / 2.43–2.70 |

These ranges use all 72 wide and 30 checkpoint run-endpoint samples. Peak allocated/reserved MiB were 5404.05/6668 for width 768, 7178.74/8474 for 1536, 7865.34/8800 for 2048 and 5403.93/6668 for 120m. Throughput includes all timed training steps and excludes evaluation. The warmup and idle intervals affect endpoint telemetry.

### Does intermediate evaluation perturb training?

The driver calls `model.eval()`, evaluates fixed validation and test prefixes under `torch.no_grad()`, then calls `model.train()`. Evaluation uses disjoint deterministic windows and has no training-stream or optimizer argument. The model uses RMSNorm and causal attention with zero dropout. The training stream owns its seeded NumPy generator.

I instrumented the actual checkpoint loop, with real shards, compiled training forward, SS3_54, seed 13 and three disposable warmup steps. I checked every evaluation boundary for changes to model weights, Muon momentum, auxiliary AdamW state and the Torch CPU/CUDA, NumPy and Python RNG states. All boundary comparisons agreed. Every training step saw training mode, every evaluated module saw evaluation mode, and each run returned in training mode.

A first six-step 120m comparison yielded different final weights and losses with evaluations at [6] and [2, 4, 6]. To test causation I ran the schedules in A–B–B–A order with identical initial state and seeds. At the default settings, all six pairwise comparisons already differed in sampled weights at the first disposable warmup step. The two A runs also differed at the first timed update, before either schedule could evaluate. The A–A test losses were 8.201058 and 8.200894. The B–B values were 8.200502 and 8.200680. A second A–B–B–A series on the tiny model ran the actual 128/256/384 cadence. Its six comparisons first differed before step 128 as well. These controls expose ordinary run-to-run variation on the GPU.

With `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, the compiled 120m six-step A–B–B–A series was bit-identical across all four runs. Every training loss matched, along with the final full weight and optimizer-state hashes and RNG states. Final validation/test losses were 8.199599742889404 and 8.201811790466309. The final SHA-256 values were `1cb449173d95dd79bb744554e31ea1aa56ad5e4af671e1f09f579374c666c8bc` for weights and `9e1d7a034f1bd5fecdbd0d788886c337f5b797484a22fc697ff06346490fb0ed` for optimizer tensors. The short 120m probe used 8,192 evaluation tokens per split. The 384-step tiny probe used 2,048. The evidence supports evaluation neutrality. Exact trajectory replay at the canonical GPU settings and equal timing after cache or thermal transitions remain unverified. The audit replayed all published statistics and ran these controlled trajectories. It did not retrain the entire 51-run grid.

## Live tensor-core arithmetic

The independent C oracle implemented the claimed group rule without calling the Python model. It used binary64 for exact products of the input formats, power-of-two scaling and integer-valued truncation. The aligned group sum fits well within the binary64 integer precision. I compared it with the repository’s arbitrary-precision integer implementation and directly launched both Triton JIT kernels with explicit block, BK, warp and stage parameters. This bypassed the Python plan cache when switching configurations. The generated PTX used `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` and its BF16 form.

For each 16-term group, let E be the floor of the base-2 logarithm of the largest magnitude among the accumulator and products. The proposed model keeps multiples of q = 2^(E−25), truncating each signed term toward zero. The group sums those terms, then truncates the result to 24 significant bits. Positive diagonal terms repeatedly lose low bits as E grows. Dividing the columns into short chunks resets that growing accumulator. This rule predicts the direction and magnitude of the measured long-K diagonal defect.

Fresh seeds 113 and 997 produced 5 × 509 mixed-sign matrices, including cancellation, FP16 subnormals and a masked tail. I compared every entry, including off-diagonals, under all 11 launch configurations in both FP16 and BF16. The raw FP32 bits matched the oracle. The fused Gram store matched the oracle rounded to its destination dtype. Scaling the operands by 2¹² and undoing 2²⁴ on the raw result preserved the bits on every comparison. The integer model also agreed with the separate C oracle on all 100 seed/dtype/entry predictions before configuration repeats.

### Where exact agreement ends

| BF16 off-diagonal input | Integer model | RTX 5080 raw FP32 | Fused BF16 store |
|---|---:|---:|---:|
| One product 2⁻¹³³ × 2¹⁰ | 2⁻¹²³ | 2⁻¹²³ | 2⁻¹²³ |
| One product 2⁻⁷⁰ × 2⁻⁷⁰ | 2⁻¹⁴⁰ | 2⁻¹⁴⁰ | 0 |
| One product 2⁻⁸⁰ × 2⁻⁸⁰ | 2⁻¹⁶⁰ | 0 | 0 |
| 256 products of 2⁻⁷⁶ × 2⁻⁷⁷ | 2⁻¹⁴⁵ | 2⁻¹⁴⁹ | 0 |
| One product 2⁷⁰ × 2⁷⁰ | 2¹⁴⁰ | +∞ | +∞ |

Each result repeated across all 11 configurations. The fourth row gives a counterexample even when the final mathematical result fits FP32 exactly. A last-store cast alone cannot repair the discrepancy. The model omits FP32 exponent-boundary behavior between MMA instructions. The first two rows also show the distinction between preserving a BF16 input subnormal, producing an FP32 subnormal and rounding the fused output to BF16. Empirical agreement in the normalized training range supports a causal account of these kernels on this device. It does not specify every Blackwell tensor-core arithmetic operation.

The frequently quoted threshold “less than 2⁻²⁵ of the accumulator” needs a binade qualification. With accumulator 1.5 and 16 products of +2⁻²⁵, hardware produced 1.5000004768371582. The corresponding negative products gave 1.4999995231628418. Each product is smaller than 1.5 × 2⁻²⁵. Halving their magnitudes yielded exactly 1.5 in both directions. The binade-based q predicts these four results.

### Finite normal counterexample

**The proposed model fails even on positive diagonal entries.** Take one FP16 row of length 16, initially zero. Set the following zero-based positions to the exact values shown.

| Position | Value |
|---:|---:|
| 6 | 0.00162506103515625 |
| 11 | 0.19140625 |
| 14 | 0.005443572998046875 |
| 15 | 0.00036072731018066406 |

The raw FP32 diagonal from `_syrk_kernel` was **0.03666875511407852** (`0x1.2c63f4p-5`) on all 11 configurations. The separate C oracle and the submitted `mma_accumulate(row, row)` both gave **0.03666875138878822** (`0x1.2c63f2p-5`). The difference is exactly one FP32 ULP. All nonzero operands and products are normal, and the row norm is below one. This case requires neither cancellation nor FP32 underflow. A five-nonzero BF16 diagonal likewise gave `0x1.627972p-5` in hardware and `0x1.627970p-5` in the model on all configurations. The raw FP32 result is the discriminating output. Rounding it to a low-precision Gram buffer can conceal a one-ULP discrepancy.

Three additional tests characterized this limit.

- With fresh seed 419, nine rows, K = 2033 and magnitudes spread over powers of two, the nominal rule matched 39/45 unique FP16 entries and 41/45 BF16. All nine diagonals happened to agree in both sets. At a Frobenius norm below one, 7/45 FP16 and 4/45 BF16 entries still disagreed with predictions for the newly rounded operands. On the original inputs, changing group size to 8 or 32 or guard bits to 1 or 3 yielded at most 5/45 matches for either dtype.
- With 4,096 independent two-row, 16-column matrices per dtype, seed 927, normal random mantissas in [1, 2) and exponents −12 through −3, the nominal rule missed 873/4096 FP16 off-diagonals and 402/4096 BF16. On the first row’s diagonal it missed 718/4096 and 208/4096 respectively. A 12,000-case control restricted to signed power-of-two operands had zero off-diagonal mismatches in either dtype. Tests dominated by powers of two can miss the defect.
- The discrepancy can survive the fused low-precision store. With seed 719 and the same normal-mantissa input distribution, five of the first 32,768 FP16 off-diagonals rounded differently. For the first witness the rounded prediction was −0.00005161762237548828 and the actual fused store was −0.00005155801773071289. A BF16 witness gave −0.000002637505531311035 versus −0.000002652406692504883. Both witnesses repeated on all 11 configurations.

The published prediction counts remain reproducible for their recorded inputs. The four-value witness rules out an unrestricted bit-exact model of even the FP16 SYRK diagonal. The experiment identifies the long sequential MMA accumulation as the source of the large-K error. The precise internal arithmetic for arbitrary products remains open.

### Drift onset and ceilings

I independently evaluated the circuit tables over [0, 1] and refined all detected local maxima with a bounded scalar search. The ceilings were 1.080953842 for PS3_55, 1.079825890 for SS3_54 and 1.087545088 for SS3_53. The largest change from the committed grid was 1.68e−8. The independent pass agreed with all 96 onset records, including the first crossing, previous grid point, monotonicity thereafter, last sigma and log-K interpolation (relative tolerance 1e−4 for the refined ceilings). All 1,308 production and mitigation measurements had finite ordered singular values, nonnegative error and positive latency.

Each entry below gives the exponent of the first production crossing, in the order **own design ceiling / common 1.08**. A dash means there was no crossing on that family's grid. The table takes the largest output singular value across the available seeds at each K. Some first crossings recede at a later K. The width-16 grid ends at 2²⁴, and the width-1 grid starts at 2²⁰.

| Circuit / dtype | 4 × K | K × 2 | 1 × K | 16 × K |
|---|---:|---:|---:|---:|
| ps3_55_fp16 | 23 / 23 | — / 10 | 25 / 25 | — / — |
| ps3_55_bf16 | 23 / 23 | 24 / 24 | 25 / 25 | — / — |
| ss3_54_fp16 | 25 / 25 | 25 / 25 | — / — | — / — |
| ss3_54_bf16 | 25 / 25 | 25 / 25 | 24 / 24 | 24 / 24 |
| ss3_53_fp16 | 24 / 10 | 25 / 25 | 25 / 25 | — / — |
| ss3_53_bf16 | 23 / 10 | 25 / 10 | 22 / 22 | — / — |

The 109 input/seed cases gave 654 circuit/dtype measurements. Production exceeded the own-circuit ceiling in 52 and the common ceiling in 64. Mitigation exceeded its own ceiling in **0/654** and the common ceiling in **10/654**. The worst mitigated sigma was 1.087453669. It cleared the corresponding design ceiling by only 0.000091419. Chunking activated in 402 cells. This is a finite swept grid. The Schatten BF16 paths were called with `allow_unsafe_bf16=True`. These measurements give no blanket production BF16 qualification.

Across the 44 normalized Gram cases with 2¹⁶ ≤ K ≤ 2²², the mean of relative diagonal bias divided by K was 5.34307e−9 per product (individual range 4.15207e−9 to 5.71568e−9). This statistic is an empirical slope over a stated interval. The 4 × 2²⁵ FP16 value was 0.15636334, or 4.65999e−9 per product at that K. The FP16 operand had 52.0% nonzero subnormals in that case. BF16 had none and still gave a bias of 0.15635707. The corpus records 90/90 invariant Gram comparisons after a power-of-two rescale, 90/90 agreements between the fused store and rounded raw Gram, 260/260 diagonal predictions, 60/60 exact short entries and 160/160 targeted MMA probe agreements. The recorded 11/11 configuration agreement held on each of three shapes.

### Opus L6 reproduction and mitigation

I regenerated seed-0 FP32 inputs on the RTX 5080, ran the three production circuits in FP16 with the fixed 32 × 64 SYRK plan, and independently formed the short-side output Gram in FP64 in blocks of 2¹⁹ columns. Its largest eigenvalue gives the squared largest singular value. I timed three synchronized warm repetitions of each full circuit. The prototype used the specified strict K > 65,536 gate, 4,096-column strided chunks and FP64 summation of partial Grams.

| Shape at K = 2²⁵ | Circuit | Production σmax | Chunked σmax | Production ms | Chunked ms | Speedup |
|---|---|---:|---:|---:|---:|---:|
| 4 × K | ss3_54 | 1.133285 | 1.036842 | 174.62 | 6.02 | 29.00× |
| 4 × K | ps3_55 | 1.177945 | 0.928481 | 175.52 | 5.96 | 29.43× |
| 4 × K | ss3_53 | 1.073228 | 1.011916 | 175.74 | 6.00 | 29.28× |
| K × 2 | ss3_54 | 1.150927 | 0.997439 | 658.73 | 7.82 | 84.19× |
| K × 2 | ps3_55 | 0.977293 | 0.922767 | 656.23 | 7.79 | 84.26× |
| K × 2 | ss3_53 | 1.186664 | 1.012848 | 657.55 | 7.81 | 84.14× |
| 1 × K | ss3_54 | 1.042557 | 0.958751 | 169.86 | 4.49 | 37.83× |
| 1 × K | ps3_55 | 1.108653 | 1.016177 | 167.53 | 4.48 | 37.42× |
| 1 × K | ss3_53 | 1.128790 | 0.914466 | 169.38 | 4.49 | 37.71× |

Fresh runs of the remaining Opus seeds gave production SS3_54 ranges 1.130277–1.134546 for 4 × K and 1.150916–1.151037 for K × 2. The 4 × K PS3_55 range was 1.177945–1.178654. The 1 × K SS3_54 value was 1.042557. These reproduce the earlier L6 measurements at its stated precision/tolerance. For each of the three shapes I also launched all 11 SYRK configurations on the *same* normalized seed-0 operand and compared raw FP32 output bits and fused FP16 store bits. All 33 configuration/shape comparisons agreed with the reference outputs.

On 4 × K, K × 2 and 1 × K, the independent scalar oracle exactly predicted the first long FP32 diagonal as 0.21090516448020935, 0.42184674739837646 and 0.8436035513877869 respectively. The observed mean relative diagonal deficits were 0.156363, 0.156470 and 0.156396. Chunking reduced them to 1.8474e−5, 1.9137e−5 and 1.9756e−5. On fresh seed 419 at seven K values from 2¹⁶ to 2²², the width-4 mean relative bias per product was 5.646e−9 in FP16 and 4.936e−9 in BF16. These controls fed the same rounded operand to both accumulation methods. The committed 42 full-circuit latency ratios at K = 2²⁵ spanned 29.11×–85.31×.

### Normal-shape cost

The gate took the production path for all three circuits on 512 × 2048, 1024 × 4096, 2048 × 8192 and 4096 × 4096. Fresh seed 717 gave identical output **bits** for all 12 comparisons, with zero chunk launches. In a separate 20-repetition timing replay, the largest median ratio was 1.0612 at 512 × 2048, SS3_53. The published ratio there was 1.0586, or 0.201836 → 0.213666 ms. At 4096², the replay ratios were 1.00025–1.00038.

I randomized the order of three timing modes over 100 repetitions at 512 × 2048. The middle mode installed the patch outside the timed interval. The last installed and removed it inside each interval, as in the published normal-shape benchmark.

| Circuit | Baseline µs | Installed outside timer µs | Installed inside timer µs |
|---|---:|---:|---:|
| PS3_55 | 217.01 | 217.76 | 223.39 |
| SS3_54 | 212.56 | 216.34 | 222.65 |
| SS3_53 | 207.10 | 213.58 | 218.15 |

The empty install/remove context alone took 1.55 µs per call over 10,000 repetitions. The installed wrappers add dispatch and shape checks even when chunking is inactive. The original normal-shape timer also charged patch installation and removal to each call and always measured baseline first. The measurements support a host-side overhead contribution. They do not identify a fixed penalty for a future native implementation. Keep the process-wide monkeypatch scoped to this benchmark.

## Canonical guards, source binding and recovery

All four committed documents passed `validate_results(..., canonical=True)`. The summary also passed with all three original source objects supplied explicitly. I then tested altered copies in memory and filenames in the private scratch directory.

| Probe | Observed result |
|---|---|
| Quick run aimed at a reserved name, 63-step wide override, path containing `../`, symlink to a reserved name, or checkpoint stage aimed at the wide name | All five CLI parses exited with status 2. |
| Canonical stage started with a genuinely dirty alternate Git index | Rejected before training. The test used a scratch copy of the index. |
| Canonical start with simulated capability 9.0 | Rejected before training. All numerical GPU probes used actual capability 12.0. |
| Altered `order_in_block`, or an incorrect orthogonalizer description with an otherwise consistent run digest | Rejected by the plan and CUDA orthogonalizer checks. |
| Altered intermediate checkpoint loss alone, or altered production maximum singular value alone | Rejected due to inconsistent perplexity or derived onset. |
| Changed summary Git SHA or a source-record Git SHA | Rejected with both source-loading interfaces. |
| Changed a stage code digest consistently with its run and row digests | Each of wide, checkpoint and drift passed standalone canonical validation. These are self-consistency checks. The independent Git-object closure digest authenticates the committed stages in this audit. |
| Impossible raw minimum singular value or wrong summary prose for the direction of a contrast | Accepted. The published raw cells passed an independent minimum/maximum check, and the published direction agrees with the paired arithmetic. |
| Explicit checkpoint source objects differed from the hashed on-disk checkpoint document | A summary containing the altered values passed the explicit-source interface. Disk-backed validation of that same summary rejected the intermediate loss mismatch. |

The last case affects a numerical claim. The explicit overload hashes a path, then derives statistics from the supplied object. Agreement of its selected metadata fields does not establish that the two documents contain the same measurements. Reading the canonical source bytes once, checking their digest and deriving from that parsed object would bind those operations. The unconditional authenticity of arbitrary stage files is a separate question from this interface defect.

I also interrupted two real quick checkpoint CLI invocations immediately after each had persisted one new complete row and a torn tail. Each process exited with the injected code 72. The third invocation used `--resume`, preserved every field of both complete rows, ran exactly one remaining trial, returned seeds 1, 2 and 3 once each and removed the JSONL. This independent CPU probe used the real verified shards. The requested GPU pytest suite separately exercises double-tail repair on the quick CUDA path.

## Live regression tests

I ran `tests/test_c4_03_wide_and_gram_drift.py`, `tests/test_schatten_switch.py` and `tests/test_polar_switch_blackwell.py` with `PYTHONPATH=.`, the project venv’s pytest and `-q` on the RTX 5080. Pytest kept its cache and scratch files under the task directory. The first run reported **347 passed, 6 skipped, 0 failed** in 1,146.16 s. The 637 warnings covered a Triton AST deprecation, a cuSOLVER SVD fallback and PyTorch synchronization debug mode.

All six skips came from two parameterized historical Schatten Tier-1 tests. The isolated worktree lacks their 214 MiB trajectory fixture. I reran those six cases with only the fixture path redirected to the existing read-only `artifacts/harvested_trajectories.pt` in the main checkout (SHA-256 `310ec381a740f2887413c15813848d7e711c3f39685af67f2d5f1967b1cc30ec`). That run reported **6 passed** in 97.08 s. Across the two invocations, all **353 distinct collected cases passed**.

## Merge and research-state disposition

| Decision | Scope |
|---|---|
| Merge | Accept the two C4-03 commits and these measured artifacts with this audit. Use the disk-backed canonical summary check and the independently authenticated source and shard digests for this record. |
| Promote wide throughput transfer | On the 64-step, 16,384-token/step width-1536 and width-2048 configurations, both Schatten variants beat compiled PE. SS3_53 also beats PS3 on end-to-end throughput at both widths after joint adjustment. SS3_54 beats PS3 on polar time at both widths and on jointly adjusted end-to-end throughput at width 2048. |
| Promote negative replication | Five fixed-trajectory seed blocks under the 384-step schedule give no adjusted SS3_54 loss advantage over PS3 at 128, 256 or 384 steps. Keep the earlier wall-budget observation scoped to its original experiment. |
| Promote drift diagnosis | On this RTX 5080 and these kernels, sequential low-precision-input, FP32 MMA accumulation causes the large normalized long-K Gram deficit. The proposed model exactly matches the recorded long diagonals and the three long diagonals independently replayed here. Withhold a general bit-exact rule. A four-term, normal FP16 diagonal falsifies it. |
| Retain experimental mitigation | ChunkedGram passes the own-circuit ceilings on all 654 audited cells, accelerates the pathological long-K cases and preserves bits on all 12 ordinary FP16 circuit/shape pairs. The benchmark and the BF16 override support experimental characterization. Production adoption requires a separate implementation decision. |

These are proposed entries for `state/RESEARCH_STATE.json`. This audit delivers a report-only commit.
