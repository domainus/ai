# Cycle 5 C5-03 independent RTX 5080 audit

**FAIL — HOLD PROMOTION.** The saved result has the expected digest, and the seed-1 loss measurements reproduce exactly. The canonical validator also certifies an impossible training calibration draw. The draw is a condition of the experiment and can be checked from data already in the result. Correct that check and the contradictory metadata checks below before relying on this certification for promotion.

| Audit identity | Value |
|---|---|
| Auditor | GPT-6.2 Astra, independent review of the Opus implementation |
| Reviewed revision | `726b20c9590aba911361e300e7d2079c7bb55cfa` |
| Producer | `38cef483071d649fc74161b5d00359c7c5ad0927` |
| C4-02 baseline | `08fc1d082495fabd3946769abc81dbc0a14ba7f0` |
| Runtime | RTX 5080, sm_120, driver 615.78.08, torch 2.14.1+cu130, Triton 3.8.0 |
| Scope | Four Part A shapes, four graph ablations, 120m Part B, full validation and test for selected seed-1 candidates, and offline verification across all three seeds |

## Findings by severity

1. **High, promotion blocker.** Canonical validation checks the number of calibration starts but accepts 128 copies of −1. The committed starts match the seeded train-only draw. Certify the actual draw and token count, then add a negative regression test. [Part C validator](../research/quant_ptq/bench_c5_03_nvfp4_120m_fusion.py#L1946)
2. **Medium, evidence validation.** Four more internally contradictory records passed canonical certification, including a negative kernel duration and a false cooperative-launch flag. The recorded benchmark evidence can support stricter checks. The validator table below gives all five accepted cases and fourteen rejected controls. [Part A](../research/quant_ptq/bench_c5_03_nvfp4_120m_fusion.py#L1866) · [Part B](../research/quant_ptq/bench_c5_03_nvfp4_120m_fusion.py#L1906) · [Part C](../research/quant_ptq/bench_c5_03_nvfp4_120m_fusion.py#L1932)
3. **Medium, public API.** Supplying a populated workspace to `launch_quant` while leaving its public count argument at the default silently returns incorrect operands. The signature permits this call. The high-level API supplies the count correctly. Validate or derive the count for dynamic calls and specify the size, dtype, device and contiguity contract for caller-owned buffers. This ambiguity also exists in C4-02. [Public launcher](../research/quant_ptq/nvfp4_inference_blackwell.py#L390)
4. **Measurement limit.** Rotated graph latency still exceeds BF16 at 1024 × 512 × 1408. Static-scale differences change sign or disappear across shapes and sessions. The 120m graph speedups reproduced, while eager timings were more variable between runs. The previous producer had already scored the same test set before the canonical rerun. Retain both runs in the study history.
5. **Low, test precision.** The driver uses `torch.equal` on floating-point outputs for its “bitwise” flags. Opposite signed zeroes compare equal under that predicate. Its operand comparisons already use integer views. Use an integer view for BF16 outputs and logits as well. The independent byte-level replay below checks the stronger claim. [Layer check](../research/quant_ptq/bench_c5_03_nvfp4_120m_fusion.py#L664) · [Logit check](../research/quant_ptq/bench_c5_03_nvfp4_120m_fusion.py#L829)

## Kernel correctness and PDL

The direct E2M1 test used 262,144 FP32 bit patterns, including 1,083 NaNs. It combined a seeded uniform uint32 draw with explicit signed zeros, subnormals, quiet and signaling payloads of both signs, infinities and every positive and negative tie with its adjacent floats. An independent CPU oracle applied the seven tie comparisons and copied the IEEE sign bit. Both the native instruction and the ladder produced zero mismatched packed bytes across 131,072 pairs.

The end-to-end stress matrix compared packed bytes, padded E4M3 scale bytes and FP32 global-scale bits with `quantize_activation_ref`. It used FP32, FP16 and BF16, all three rotations, dynamic scale and two static scales, all four PDL/native switch combinations, and aligned and offset stride-two inputs at 19 × 80. The static unit-scale tie blocks included a magnitude-six anchor. Special values originated as integer payloads in each input dtype. Small and large blocks exercised E4M3 underflow and saturation. There were **864/864 exact comparisons** across the tie, special, underflow/saturation and random families. A further **72/72** public two-pass checks matched, including dirty packed/scale buffers, NaN-filled workspaces and untouched workspace tails.

**PDL survives the cached C launcher and graph capture.** The live Triton direct launcher was active. The producer metadata and direct launch head both had PDL false, and the producer PTX contained `griddepcontrol.launch_dependents`. The enabled consumer had PDL true in both places. Triton passes it to `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`. I queried the actual graph via `cuGraphGetEdges` for every rotation. Each two-node graph had exactly one edge. With PDL enabled its (type, from_port, to_port) was (1, 1, 0), a programmatic dependency. With PDL disabled it was (0, 0, 0), the default dependency.

Line-annotated sm_120a SASS maps the producer signal to `PREEXIT` and the consumer wait to `ACQBULK`. The partial workspace load follows the wait in PTX and SASS. I also found `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C` in the native conversion. Loads of input activations and rotation tables can precede the wait. Pass 1 only reads those operands. It writes the partial maxima, which pass 2 reads after the wait. A graph dependency permits overlap. Its mere presence does not measure how much overlap the GPU scheduled.

For each rotation, PDL setting and eager/graph path I issued 512 dynamic calls at 257 × 1408. The seven inputs ranged from all zero to random values scaled by 1e12. I changed the live PDL option after graph capture. All 6,144 calls matched the three reference operands byte for byte. A separate test queued 64 groups of seven adjacent dynamic calls per rotation at 193 × 512, with no comparison or tensor copy between calls within a group. All 1,344 results matched. These executions exposed no memory-visibility failure. The shape cache for pass 1 is independent of both switches. The pass-2 key includes the switches, and the compiled-launch key includes constants, input strides and pointer alignment. [Kernel and launcher](../research/quant_ptq/nvfp4_inference_blackwell.py#L233) · [Plan caches](../research/quant_ptq/nvfp4_inference_blackwell.py#L348)

In the public split-launch test I negated the strided input after pass 1. Its tensor maximum stayed the same, and pass 2 had to read the new values. Passing the count returned by pass 1 gave exact results for all rotations. The public consumer’s PDL metadata was false. Omitting the count gave scale 1.0 in every case. The correct identity, Hadamard and butterfly scales were 0.0014008142, 0.0014962533 and 0.0013731758. All calls used adequately sized contiguous output buffers.

## Baseline and latency replay

The baseline-source test matched all 12 decorated or plain function bodies to Git blob `08fc1d0:research/quant_ptq/nvfp4_inference_blackwell.py`, allowing only the C4-02 name prefix. I also compared the full `NVFP4Linear` class, input checker and scaled GEMM against the baseline. They are textually identical. Each variant shares packed weights and GEMM between implementations. The old quantizer keeps its compiled launcher cache and its original allocation of 512 partials plus one global scale. The wrapper adds a child-module lookup and omits the old eager dtype/device check. Those small host-side differences can affect close eager timings. Graph capture removes both from replay.

I kept the canonical timing protocol on the four requested shapes and the 120m model. Each CUDA-event sample followed a 256 MiB flush. Methods were warmed, interleaved in Williams order and synchronized as in the producer. Part A had 20 rounds × 10 samples per method and mode. The 11-method ablation had 44 rounds × 10. Part B had 38 rounds × 10. These are within-session measurements on one RTX 5080. All 48 Part A operand/output checks and all 24 ablation equality checks passed.

Variant order in the next table is identity, Hadamard, butterfly, static-scale Hadamard. Each speed ratio is the C4-02 median divided by the C5-03 median.

| M × N × K | Mode | BF16 µs | C5-03 µs | C4-02 / C5-03 |
|---|---|---|---|---|
| 1024 × 512 × 1408 | graph | 22.84 | 20.99 / 23.04 / 25.09 / 20.89 | 1.185 / 1.173 / 1.152 / 1.083 |
| 1024 × 512 × 1408 | eager | 23.01 | 21.74 / 24.24 / 26.08 / 20.84 | 1.162 / 1.124 / 1.136 / 1.113 |
| 16 × 4096 × 4096 | graph | 45.92 | 28.58 / 28.57 / 30.65 / 26.88 | 1.014 / 1.057 / 1.052 / 1.018 |
| 16 × 4096 × 4096 | eager | 46.02 | 28.96 / 29.01 / 30.80 / 26.77 | 1.006 / 1.050 / 1.057 / 1.029 |
| 1024 × 4096 × 4096 | graph | 341.76 | 82.74 / 85.95 / 90.21 / 81.47 | 1.040 / 1.045 / 1.073 / 1.000 |
| 1024 × 4096 × 4096 | eager | 341.96 | 83.16 / 86.28 / 90.92 / 81.50 | 1.037 / 1.042 / 1.070 / 0.995 |
| 2048 × 8192 × 8192 | graph | 2380.80 | 435.33 / 441.55 / 452.61 / 420.02 | 1.008 / 1.024 / 1.051 / 1.003 |
| 2048 × 8192 × 8192 | eager | 2379.16 | 433.87 / 441.75 / 452.22 / 419.18 | 1.010 / 1.024 / 1.052 / 1.005 |

At 1024 × 512 × 1408, graph Hadamard and butterfly both still trailed BF16. The 0.205 µs median difference between BF16 and Hadamard is small. The committed small-batch static-scale tie was a 1.018× ratio on replay. Static-scale throughput at 1024 × 4096 × 4096 was effectively level in graphs and slightly slower in eager. The actual GEMM limits gains at large prefill. Separate CUPTI measurements gave about 355 µs for the C5-03 Hadamard FP4 GEMM within a 441.55 µs graph. Quantizer kernel durations can overlap and should not be summed into elapsed time.

**Graph ablations, median µs**

| M × N × K | Rotation | C5-03 | PDL off | Ladder conversion | Cooperative single launch | C4-02 |
|---|---|---|---|---|---|---|
| 1024 × 512 × 1408 | none | 21.62 | 22.73 | 22.76 | 24.82 | 24.99 |
| 1024 × 512 × 1408 | had_b16 | 23.80 | 24.77 | 24.76 | 28.77 | 26.82 |
| 16 × 4096 × 4096 | none | 28.60 | 28.76 | 28.78 | 31.03 | 28.78 |
| 16 × 4096 × 4096 | had_b16 | 28.73 | 29.39 | 28.78 | 32.67 | 30.20 |
| 1024 × 4096 × 4096 | none | 82.57 | 82.95 | 85.60 | 88.88 | 86.12 |
| 1024 × 4096 × 4096 | had_b16 | 86.02 | 88.45 | 88.36 | 95.52 | 89.97 |
| 2048 × 8192 × 8192 | none | 435.20 | 437.30 | 437.25 | 459.11 | 438.89 |
| 2048 × 8192 × 8192 | had_b16 | 441.75 | 443.39 | 450.05 | 483.33 | 452.20 |

The cooperative kernel used 168 CTAs on 84 SMs. It reproduced all 1,000 changing-input replays on each canonical ablation shape and 32/32 on each additional prefill shape. All four barrier counters finished at zero. The cooperative variant was slower than C5-03 on all eight replay comparisons.

**120m full forward, 4,096 tokens per call**

The four eager C4-02/C5-03 logit comparisons passed. All 60 block linears used NVFP4. The following medians are milliseconds. The brackets give the C5-03 minimum and maximum round means.

| Variant | Mode | C4-02 | C5-03 [range] | C4-02 / C5-03 |
|---|---|---|---|---|
| none | eager | 8.924 | 8.867 [8.648, 10.213] | 1.0064 |
| none | graph | 8.575 | 8.372 [8.350, 8.486] | 1.0242 |
| had_b16 | eager | 9.086 | 8.887 [8.846, 9.586] | 1.0224 |
| had_b16 | graph | 8.956 | 8.749 [8.718, 8.885] | 1.0237 |
| bfly_b16 | eager | 9.651 | 9.251 [9.209, 10.097] | 1.0432 |
| bfly_b16 | graph | 9.524 | 9.098 [9.067, 9.218] | 1.0469 |
| had_b16_static_gs | eager | 8.703 | 8.564 [8.514, 10.012] | 1.0162 |
| had_b16_static_gs | graph | 8.581 | 8.429 [8.388, 8.738] | 1.0181 |

The baseline medians were eager 12.991 ms, compiled 10.861 ms, eager_graph 12.879 ms.

An additional replay captured one C4-02/C5-03 pair at a time in a shared graph pool. I tested two input values, reversed replay order, and read each output before reusing the pool. Comparisons used int16 views of complete BF16 outputs. All 96 layer and 24 full-model comparisons passed, including eager against eager and each graph against eager. The isolated 120m replay reproduced the four graph speed ratios within 0.002 of the committed ratios. Absolute eager times varied more. For example, C5-03 eager Hadamard was 8.887 ms here and 11.066 ms in the full canonical session.

## Seed 1 checkpoint replay

I loaded the saved weights and regenerated Hessians, channel statistics, rotations, GPTQ encodings and train-calibrated scales. A separate script drove those repository primitives and the full-split scorer. It selected alpha from four live validation scores before mapping test. This checks execution from the checkpoint and shares the implementation of the model, calibration and NLL primitives with the producer. Seeds 2 and 3 received file, tensor, log and statistical checks.

Each validation score covers 2,001,920 next-token predictions. Each test score covers 1,999,872. All 18 stored baseline, hardware and Hadamard emulation NLLs compared below matched the persisted binary64 values exactly. The four butterfly emulation values are new audit measurements. The two additional validation-only AWQ candidates also matched their selection records.

| Method | Mode | Validation NLL | Test NLL |
|---|---|---|---|
| BF16 | baseline | 4.615477041454 | 4.633469397022 |
| Butterfly GPTQ | dynamic | 4.630612429938 | 4.648580491573 |
| Butterfly GPTQ | static_bf16cal | 4.630293361244 | 4.648461113572 |
| Butterfly GPTQ | static_seqcal | 4.630316222110 | 4.648405881277 |
| Butterfly GPTQ | emulated | 4.630584685089 | 4.648520382746 |
| Butterfly AWQ + GPTQ | dynamic | 4.629963516762 | 4.648274206346 |
| Butterfly AWQ + GPTQ | static_bf16cal | 4.629919431642 | 4.648153039168 |
| Butterfly AWQ + GPTQ | static_seqcal | 4.629901569884 | 4.648134191526 |
| Butterfly AWQ + GPTQ | emulated | 4.629925858090 | 4.648209582581 |
| Hadamard GPTQ | dynamic | 4.631390991113 | 4.649785047669 |
| Hadamard GPTQ | emulated | 4.631265358303 | 4.649821873268 |

The four butterfly dynamic validation NLLs for alpha 0, 0.25, 0.50 and 0.75 were 4.630612429938, 4.629963516762, 4.630181308659, 4.630838215138. Alpha 0.25 won. All four butterfly fits and the additional Hadamard candidate decoded 60/60 weight matrices exactly under the int32 view. Each measured hardware mode dispatched 60 `aten::_scaled_mm_v2` calls on the probe forward and had 60 NVFP4 modules, zero BF16 block linears and zero emulation modules. Emulation had 60 emulated block linears and zero FP4 dispatches. BF16 had zero FP4 dispatches. The largest butterfly hardware versus emulated NLL gap was 0.000064624 nats. For the separately measured Hadamard pair it was 0.000125633 nats. Both are below the declared 0.001-nat threshold.

## Canonical result and checkpoint provenance

The result SHA-256 is `9ed6c9fd3879566f78e943f5aa54657b1f0e1c27e543e3e93f11103413da14ce`. The unmodified payload returned `certified_canonical=True` for ABC. I independently hashed each file and its sorted state dictionary on CPU. The tensor digest covers the UTF-8 key, a NUL, the FP32 array shape string and contiguous FP32 bytes. All six digests match the supplied pins and the corresponding training records.

| Seed | Checkpoint file SHA-256 | Tensor SHA-256 |
|---|---|---|
| 1 | `448b0297dfd632839fcdcae4fd530fdc070d785da5a5bf35d7e886377a5ab29b` | `ad9dc6a00b80e5b82b14eb01927ea0cd7b1c64825adcf7e3c4e016f566d8c7f9` |
| 2 | `0b02ffee244aaa6f811740759c166df20a4f3541ef60f0b7f223a98567aee3ef` | `82158f688be42b5acb46d9a33997c06317943da24db7ee2c11f796f935e338c4` |
| 3 | `057e78c783ff966cc368b1292cd4514ab6f2f9d06032dc9334f77e72b6154856` | `d705cbc5ea2a95296ecd5a899ef76b9ed06cff8a1bd04a75a20c5e354c9d1b38` |

**Two runs used the same holdout.** The superseded run at `4736807` finished at 10:19 MST. The canonical run at `38cef48` started at 10:31 MST on November 27. The intervening source diff adds host-side launch-plan caching and changes the driver’s runtime estimate. Both logs agree with their respective JSONs for all 84 printed validation deltas, 84 test deltas and six AWQ selection lines at printed precision. In both logs, the three Phase 1 completions are at lines 250, 341 and 432. The first printed test delta is at line 439.

All three checkpoint tensor digests changed on retraining. The largest absolute grid NLL differences between runs were 0.009366, 0.003867 and 0.005494 nats for seeds 1–3. Seed 2 selected butterfly AWQ alpha 0.50 in the earlier run and 0.25 in the canonical run. Both selected butterfly AWQ for hardware dynamic on validation. Its mean test delta was +0.015125 and +0.015209 nats respectively. The superseded 120m eager Hadamard result was a C5-03 slowdown (0.9635× C4-02 speed). The canonical ratio was 1.0030×. The logged values support a complete rerun. They establish prior access to test scores and do not establish anyone’s motive for the rerun.

## Statistical and split checks

I recomputed the nine paired contrasts on each split from raw per-seed NLLs. The independent calculation used the sample SD, Student’s t survival function with two degrees of freedom, the 0.975 quantile for the CI, enumeration of all eight sign patterns and sorted step-down Holm adjustment within each family of nine. The largest absolute differences from the JSON were 1.43e-14 for t, 6.74e-14 for CI endpoints, 1.12e-16 for unadjusted p and 2.23e-16 for adjusted p. Both sign-flip columns matched exactly. The two-sided floor is 2/8 = 0.25. Every Holm-adjusted sign-flip p is 1.

| Contrast, in protocol order | Validation t-test p, Holm | Test t-test p, Holm | Raw sign-flip p, val / test |
|---|---:|---:|---:|
| Hadamard vs identity GPTQ | .145574, .436723 | .107444, .322332 | .25 / .25 |
| Butterfly vs Hadamard GPTQ | .004621, .027724 | .008712, .052274 | .25 / .25 |
| GPTQ vs RTN | .000457, .003655 | .000247, .001978 | .25 / .25 |
| Hadamard GPTQ vs Hadamard RTN | .001971, .013797 | .002410, .016869 | .25 / .25 |
| AWQ vs plain Hadamard GPTQ | .047253, .189014 | .036167, .144669 | .25 / .25 |
| AWQ vs plain butterfly GPTQ | .031999, .159994 | .012471, .062354 | .25 / .25 |
| Sequential vs BF16 static calibration | .724957, .724957 | .477939, .955878 | .75 / .50 |
| Dynamic vs BF16 static calibration | .230983, .461967 | .606346, .955878 | .25 / .75 |
| W4A4 dynamic vs W4A16 | .000151, .001355 | .000099, .000889 | .25 / .25 |

The seed-20261126 NumPy draw reproduced all 128 distinct training starts and 65,536 calibration inputs. The manifest verifier reads the three binaries at startup. The scoring code maps train and validation in Phase 1, freezes across all seeds, then maps test. Per seed, validation has one baseline, 20 base-method rows, one emulation and 12 extra AWQ scorings. Test has one baseline, 20 base-method rows, one emulation and eight selected AWQ rows. All six alphas were nonzero. This gives 102 validation scorings before the freeze and 90 test scorings after it. The validation NLL sums select butterfly AWQ in all four modes. Rebuilding the selection object from the six alpha records and those winners gives `6f8e77fe477deca62651c39b0f5f7a5b969d7ce1f226f5aabea0293ff3ed69cf`, exactly the frozen digest. The digest includes the recorded AWQ search durations, so replaying identical choices at a different speed yields a different digest.

## Validator and output controls

Each of 19 independent negative data checks changed one field of an in-memory copy or temporarily substituted one value in memory. The writer was never called. Fourteen controls rejected the input.

| Rejected input | Independent checks |
|---|---:|
| Top summary mismatch, GPU median inconsistent with samples | 2 |
| Unknown Git commit, wrong source digest, dirty canonical flag | 3 |
| Phase 1 split order, scoring ledger, frozen-selection digest | 3 |
| False AWQ inheritance, non-minimum selected alpha | 2 |
| FP4 dispatch in emulation, contradictory HW/emulation parity summary | 2 |
| Wrong Part A plan, missing Part B NVFP4 layer | 2 |

Five inconsistent records individually received canonical certification. The first is sufficient to prevent using this validator as the calibration promotion gate.

| Accepted inconsistency | Check available to the validator |
|---|---|
| All 128 recorded calibration starts were −1 | Regenerate the sorted sample from the pinned train length, sequence length, seed and count. Check the token total. |
| 120m Part B recorded zero quantized linear parameters | Derive 84,934,656 parameters from the model configuration. |
| A CUDA kernel profile recorded a negative duration | Require finite nonnegative microseconds and the expected kernel roles. |
| The cooperative ablation recorded its cooperative launch flag as false | Require the launch flag and a positive grid consistent with its recorded capacity. |
| A plain GPTQ row carried AWQ alpha 0.75 | Require a null alpha on every non-AWQ row. |

Seven independent parser checks returned exit code 2. They covered a reserved basename, a symlink to that basename, an abbreviated option, explicit default device and parts, a window limit, and a canonical run targeting another path. A quick run targeting an ordinary temporary basename parsed as noncanonical. The repository suite also checks the dirty-tree and device gates. With this report untracked, the Git helper reported a clean tree. After staging the report, the live host guard rejected the actual tracked-dirty checkout before any benchmark or write. The output writer uses a predictable sibling staging name and the canonical comparison resolves paths. The path checks do not establish the safety of every staging path or ancestor. Use an exclusively created same-directory temporary file, and bind the destination to the intended checkout before replacement. No reserved output was written during these checks. [Output parser](../research/quant_ptq/bench_c5_03_nvfp4_120m_fusion.py#L2133) · [Git state](../research/polar_opt/harness_tier2_real.py#L477) · [Writer](../research/quant_ptq/bench_isoloss_ptq.py#L1143)

## Reproduction and coverage

I used the requested shared virtual environment with `PYTHONPATH=.` in the isolated worktree. Compiler caches, probes and replay outputs went into a private temporary directory. Training checkpoints, shard files and both producer logs were opened read-only. The host had 16 logical CPUs and a starting load average of 2.82. Desktop GPU clients remained present. GPU benchmarks ran sequentially.

```bash
PYTHONPATH=. /home/ryan/Documents/projects/ai/.venv/bin/pytest -q tests/test_nvfp4_inference.py tests/test_c5_03_nvfp4_120m_fusion.py
PYTHONPATH=. /home/ryan/Documents/projects/ai/.venv/bin/pytest -q tests/test_c5_02_isoloss_ptq_mitigation.py -k test_committed_canonical_artifact
```

The first command passed **249 tests** in 394.71 seconds with 10 warnings. The second passed **3 tests**, with 97 deselected, in 23.66 seconds. The live timing calls used the canonical seed and timing settings with `part_a_shape`, `ablation_shape` and the 120m entry to `part_b`. Checkpoint reconstruction used `Study.candidate`, `calibrate`, `Study.seqcal` and `Study.measure` after loading seed 1. The original result still has the pinned digest. The scope of the live evidence is the specified GPU, toolchain, shapes, model and checkpoint.
