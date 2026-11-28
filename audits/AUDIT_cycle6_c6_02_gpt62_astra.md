# C6-02 independent RTX 5080 audit

**Verdict: FAIL.** The omitted compiled BF16 baseline reverses the 50m Pareto conclusion when the comparison includes the previous study’s default compiled BF16 path. The published graph-only comparison is correctly labeled. On the same RTX 5080, full butterfly took 13.242 ms and compiled BF16 took 11.766 ms. Compiled BF16 also had lower NLL on the three real checkpoints. At 120m full butterfly retained its latency advantage at 10.101 ms versus 10.771 ms for compiled BF16. At w2048_6l the fused path is fast, but the two full-mix families materially change the logits and there is no trained NLL point for that preset.

Dynamic Hadamard closed the small-shape BF16 gap. The native fused path removed one multiplication kernel per layer. The archived artifact and all three inputs matched their pins, and all 729 tests passed. The failed verdict applies to accepting the broader frontier and drop-in compatibility claims.

Reviewed 2026-11-28 on Bazzite, RTX 5080 (sm_120). Artifact commit `a716333`. Producer commit `47cdedd`.

## Severity-ranked findings

1. **HIGH — The 50m frontier depends on omitting the primary BF16 reference.** The cross-full butterfly takes 12.54% longer than compiled BF16, with worse trained test NLL. All three 50m wide paths are slower than compiled BF16. At 120m the butterfly takes 6.22% less time, with worse NLL. Add compiled BF16 to the same timing order and use measured trained quality for its Pareto point. Report the two model conclusions separately.
2. **HIGH — Concurrent submissions can silently use the wrong scales.** The same cached `LtPlan` mutates a shared descriptor at `nvfp4_small_shape_blackwell.py:825`. Two host threads, each with its own CUDA stream and retained tensors, produced incorrect matrices. Serializing submission removed the errors. Isolate the mutable descriptor per concurrent submitter or protect both pointer updates and the matmul submission as one operation. Require repeated two-thread equality with a serial reference.
3. **HIGH — A successful butterfly state-dict load can corrupt ordinary inference.** The non-persistent packed coefficients only rebuild at construction. After a standard load from a second layer with different angles, the native forward disagreed at 766 of 768 outputs while public activation quantization still matched the loaded layer. Rebuild the packed table after loading. Require agreement on both sides of the dispatch threshold and after a checkpoint round trip.
4. **HIGH — Full-mix w2048 fidelity needs a shape-specific qualification.** The benchmark records the differences, but “near-0 parity” does not describe relative logits errors of 0.209 and 0.253. The first down projection changes packed FP4 bytes and later blocks amplify the difference. Use the reference cross-mix for exact-mode execution at affected shapes, or qualify the fused mode against trained w2048 quality before assigning it an NLL.
5. **MEDIUM — The module-wide dtype conversion changes cross-mix constants.** Casting an already constructed full-mix wrapper to BF16 also casts its FP32 matrices. At K=1408 this raised the maximum mix error from 4.77e-7 to 0.01022. Preserve FP32 coefficient buffers across dtype conversion and check conversion before and after wrapper construction.
6. **MEDIUM — The result validator accepts contradictory scientific evidence.** It accepts an empty raw direct trace, a failed heuristic parity flag, negative time and impossible logits counts. Validate raw and derived evidence together. The committed digest, independent evidence checks and live traces support the original artifact.
7. **MEDIUM — Workspace fallback and plan lifetime need an explicit public contract.** The layer falls back for a nonzero-workspace plan. The exported direct helper invokes the plan regardless, passing null workspace and size zero. All measured native heuristics needed zero workspace. Guard the direct helper or supply the required workspace, and bound or dispose of native resources for dynamic shape sets.
8. **LOW — Mutable tuning state and write-once publication assume a single owner.** Cached tile plans ignore later override changes. Option contexts use a process-global dictionary. The writer's atomic rename can replace an existing regular destination after separate run admission. Make tuning overrides local or invalidate plans on change, and use no-replace final publication if multiple producers may run.

## Same-session compiled BF16 comparison

GPU median in milliseconds, followed by the range of round means in brackets. Lower is faster.

| Forward method | 50m | 120m | w2048_6l |
|---|---:|---:|---:|
| Compiled BF16 | 11.765802 [11.664, 12.264] | 10.771410 [10.734, 10.790] | 16.532454 [16.529, 16.632] |
| BF16 CUDA graph | 14.481390 [14.456, 14.828] | 12.839829 [12.723, 12.882] | 17.549080 [17.467, 17.873] |
| C6 Hadamard b16 | 10.657070 [10.636, 10.787] | 8.284803 [8.256, 8.371] | 8.005096 [7.972, 8.358] |
| C6 fused Hadamard b64 | 12.646648 [12.603, 13.188] | 9.291939 [9.289, 9.405] | 9.004742 [9.001, 9.089] |
| C6 fused full Kron | 12.486928 [12.444, 13.052] | 9.410306 [9.394, 9.492] | 9.520984 [9.498, 9.677] |
| C6 fused full butterfly | 13.241598 [13.199, 13.598] | 10.101339 [10.073, 10.176] | 10.035414 [9.956, 10.423] |

Wall medians for compiled BF16 versus full butterfly were 11.979 vs 13.518 ms (50m), 11.000 vs 10.327 ms (120m), 16.924 vs 10.138 ms (w2048_6l). The GPU and wall rankings agree. The corresponding C5-03 published compiled BF16 medians were 11.751, 10.794 and 16.696 ms.

The uncompiled BF16 graph captures launch submission. Inductor also fuses operations. In separate CUPTI profiles of 50m and 120m, each uncompiled BF16 graph launched 375 kernels, including 73 GEMMs. Each compiled path launched 134 kernels, also including 73 GEMMs. The 50m full-butterfly path launched 60 each of cross-mix, amax, quantization and FP4 GEMM. Its largest remaining BF16 GEMM kernel name accounted for about 3.95 ms across the forward, and the layer-normalization kernel for 1.85 ms. The saved FP4 GEMM time competes with this remaining BF16 work and the per-layer rotation cost. All measured paths produced finite logits. The relative L2 differences between compiled and graphed BF16 logits were 0.01137, 0.01329 and 0.02006, respectively. Trained compiled-model NLL was therefore measured separately.

## Single-layer spot checks

GPU times below are microseconds. Each cell lists median [min, max] round means. Native C6 forces delegation off.

| M, N, K | Rotation | BF16 | C5 | Default C6 | Native C6 |
|---|---|---:|---:|---:|---:|
| 1024, 512, 1408 | none | 23.51 [22.75, 23.91] | 22.74 [21.72, 23.36] | 21.40 [20.19, 22.11] | 21.25 [20.07, 22.22] |
| 1024, 512, 1408 | had_b16 | 23.51 [22.75, 23.91] | 24.71 [23.77, 24.96] | 21.92 [20.33, 22.34] | 21.93 [20.36, 22.42] |
| 1024, 512, 1408 | bfly_b16 | 23.51 [22.75, 23.91] | 26.71 [25.47, 27.03] | 24.51 [23.54, 25.20] | 24.37 [23.05, 24.90] |
| 2048, 8192, 8192 | none | 2369.02 [2364.87, 2425.91] | 432.28 [430.70, 434.59] | 430.95 [429.77, 432.84] | 440.56 [438.34, 442.57] |
| 2048, 8192, 8192 | had_b16 | 2369.02 [2364.87, 2425.91] | 439.29 [438.03, 440.52] | 438.37 [436.33, 461.00] | 437.76 [436.33, 439.71] |
| 2048, 8192, 8192 | bfly_b16 | 2369.02 [2364.87, 2425.91] | 450.15 [449.13, 452.00] | 450.36 [449.13, 452.00] | 477.89 [476.73, 498.69] |

At the target small shape, dynamic Hadamard beats BF16, and the butterfly median still trails it. The latter ranges overlap. At the large shape, native butterfly costs 6.2% more time than C5, and native identity costs 1.9% more. Default C6 delegates both. The small native Hadamard advantage over C5 at the large shape is within the observed spread. All quantized spot-check outputs matched C5 bitwise. The retained 31/32 claim is the canonical grid count. This audit retimed two of the eight grid shapes.

## Trained checkpoints and test NLL

Each cell below uses the full C6-01 test split. The C6-01 column gives the archived three-seed mean. For compiled BF16 its reference is the archived BF16 baseline. Each rebuilt GPTQ candidate also matched the archived effective-weight digest.

| Preset | Forward | Seed 1 | Seed 2 | Seed 3 | Live mean | C6-01 mean |
|---|---|---:|---:|---:|---:|---:|
| 50m | compiled BF16 | 5.3936286222 | 5.3765352794 | 5.3759397540 | 5.3820345519 | 5.3820324911 |
| 50m | cross_full_butterfly_b16 | 5.4042080615 | 5.3869049001 | 5.3861837050 | 5.3924322222 | 5.3924322222 |
| 50m | kron_hadamard_full | 5.4052130389 | 5.3876423853 | 5.3873352995 | 5.3933969079 | 5.3933969079 |
| 50m | hadamard_b64 | 5.4053324393 | 5.3882967738 | 5.3879717323 | 5.3938669818 | 5.3938669818 |
| 120m | compiled BF16 | 4.6334646033 | 4.6264826242 | 4.6128836625 | 4.6242769633 | 4.6242854573 |
| 120m | cross_full_butterfly_b16 | 4.6478281566 | 4.6410651302 | 4.6275134074 | 4.6388022314 | 4.6388022444 |
| 120m | kron_hadamard_full | 4.6488627235 | 4.6417477110 | 4.6281219496 | 4.6395774614 | 4.6395774535 |

The 120m C6-01 baseline grid contains full Kron and full butterfly. It has no b64 GPTQ test score to transfer. The w2048_6l model has no C6-01 trained Pareto cell.

For the following unequal scores I also rescored the original C6-01 hardware path using the freshly rebuilt weights. Values are test NLLs.

| Model | Seed | Family | Archived | Fast | Rerun original |
|---|---:|---|---:|---:|---:|
| 120m | 1 | cross_full_butterfly_b16 | 4.647828187810661 | 4.647828156558661 | 4.647828187810661 |
| 120m | 1 | kron_hadamard_full | 4.648862786007367 | 4.648862723503367 | 4.648862786007367 |
| 120m | 3 | cross_full_butterfly_b16 | 4.627513415253108 | 4.627513407440107 | 4.627513415253108 |
| 120m | 3 | kron_hadamard_full | 4.628121863617631 | 4.628121949560631 | 4.628121863617631 |

Over these checkpoints, nine of nine 50m fused scores and two of six 120m fused scores reproduce exactly. The largest absolute 120m NLL difference was 0.000000085943. This supports numerical transfer at the reported precision for these two trained cells.

The reconstructed frontier below substitutes the live compiled BF16 and measured dynamic-method timings and qualities where available. Other narrow-method entries retain their pinned canonical timings and NLLs. The direct 50m dominance comparison uses only live timings and scores. The names run from lowest latency to lowest NLL.

- 50m — hadamard_b16_gptq_static, gptq, adaptive_butterfly_b16_gptq, compiled BF16.
- 120m — hadamard_b16_gptq_static, kron_hadamard_full_gptq, cross_full_butterfly_b16_gptq, compiled BF16.

## Where the wide logits diverge

The audit reproduced the random-weight logits comparisons independently. All three 50m wide families matched C6-01 across 412,090,368 logits each. All three 120m families matched across 206,045,184 logits each. The canonical mixed-input hooks had found 3,327 FP32 mismatches for each 120m full mix. A single exact final output does not establish equivalence on different tokens or on the last partial batch of a test set.

At w2048_6l, b64 remained exact across 103,022,592 logits. Full Kron differed at 102,127,503 logits with relative L2 error 0.209078. Full butterfly differed at 102,294,865 with relative L2 error 0.253306. These are finite outputs. A relative L2 error of 0.25 is substantial even when every individual mix error begins near FP32 rounding precision.

I instrumented the first five quantized linears on the actual quantized forward in block 0. The four projections with K=2048 produced identical mixed values and identical quantized operands for both families. The down projection has K=5632. At that projection the Triton two-half FMA reduction and the PyTorch/cuBLAS reference give slightly different FP32 results. With the very same activation input, weights and scales, the difference crosses quantization thresholds.

| First down projection, M=2048, K=5632 | Mixed FP32 mismatches | Max absolute difference | Differing packed FP4 bytes | Differing block scales | Tensor scale on both paths |
|---|---:|---:|---:|---:|---:|
| Full Kron | 8,448,528 | 9.536743e-7 | 3 | 0 | 0.000936902361 |
| Full butterfly | 8,449,721 | 9.536743e-7 | 4 | 0 | 0.001093562692 |

Those three or four packed-byte differences feed the first residual output. As the resulting activations pass through later projections and quantizers, more values change. The measured relative L2 errors at the six block outputs were:

| Family | Block 1 | Block 2 | Block 3 | Block 4 | Block 5 | Block 6 |
|---|---:|---:|---:|---:|---:|---:|
| Full Kron | 0.000138 | 0.028007 | 0.081769 | 0.137790 | 0.177348 | 0.208081 |
| Full butterfly | 0.000129 | 0.060188 | 0.137051 | 0.186336 | 0.222916 | 0.252368 |

The bitwise scale comparisons rule out a shared-scale change at the first divergence. The larger final errors follow successive nonlinear quantization decisions. “One ULP” around a typical mixed value does not bound the error in the logits, and cancellation can make the local ULP count larger. Use the original cross-mix where exact compatibility is needed, or qualify each fused shape against a stated task-level quality tolerance.

## Quantizer, cuBLASLt and CUDA graphs

**Alignment and bounds.** At `nvfp4_small_shape_blackwell.py:520`, vector I/O requires unit K stride. For both the 16-byte and 8-byte paths, the predicate tests the base address and the row stride in bytes. The row-stride test is redundant only when M=1. A valid group contains 16 values by the public K-multiple-of-16 contract. For 16-bit inputs the vector loads consume exactly 32 bytes, and for FP32 they consume 64 bytes. A group offset preserves alignment. Pass 1 masks out surplus tiles and rows. Pass 2 also masks padded groups and rows. The packed coefficient buffer is contiguous with four FP32 values per coefficient pair, so each vector starts on another 16-byte boundary. Both passes mask the coefficient groups separately. The generated `sm_120` PTX I inspected contains `setp.ne.b32`, two zero initializations and `@p ld.global.v2.b64`. A false predicate suppresses the memory operation itself. This argument applies to valid storage-backed tensor views and the shipped tile settings.

**Launch contract.** `launch_amax` returns the actual pass-1 program count. Dynamic `launch_quant` checks a 1-D contiguous workspace on the same device, its FP32 dtype and minimum capacity, and an exact built-in integer equal to that shape and rotation’s C6 count. Static mode requires zero and no workspace. Both public launchers use C6 plans even when the layer dispatcher delegates. The suite exercised short and noncontiguous buffers, invalid counts, dirty buffers and padded scale storage. The returned count does not identify the source activation. A caller who changes input, rotation or workspace contents must rerun pass 1.

**Direct GEMM.** The selected cuBLASLt heuristic uses FP32 compute, device pointer mode and a 1 MiB search limit. `LtPlan.__call__` updates both scale pointers and passes `torch.cuda.current_stream(out.device).cuda_stream` to `cublasLtMatmul`. The graph captures the device addresses for each submission. Eager warm-up is required for a new plan or kernel specialization. In graph use, keep external inputs, weights, scales and result storage valid for the replay lifetime. PyTorch manages allocations made inside its capture pool. Raw pointer submission alone does not extend the life of a tensor supplied by another stream. Use normal PyTorch stream ordering and allocator-lifetime discipline when sharing tensors across streams.

For isolated Hadamard layers at both spot-check shapes, CUPTI found one amax, one quantization and one FP4 GEMM on the direct path. C5 additionally launched one scale-product kernel. The separate eager-forward sync check found zero D2H copies, zero host syncs inside the recorded regions and no sync-debug errors across eight layer/wrapper configurations. Changing inputs produced 32/32 bit-exact graph replays. The full Transformer still contains unrelated multiplies and some delegated paths. The eliminated multiplication concerns the per-layer direct NVFP4 GEMM.

**Resource lifetime.** The cache is keyed by M, N, K, output dtype and device index. Each distinct plan holds a raw matmul descriptor and four layouts. Preferences are destroyed on the successful construction path. The implementation supplies no teardown for descriptors/layouts or per-device handles and no limit on this process-global shape cache. Finite, warmed benchmark shapes bound this cost. A service accepting arbitrary M can grow the cache over its entire lifetime. Normalize index-less device aliases, provide scoped ownership or a bounded cache, and clean up partial plan creation. Keep handle/layout reuse separate from mutable per-submission state.


**Independent quantizer cases.** Seed 502, M,K in `(1,16), (3,48), (5,144), (129,80), (4097,32)`, each tested with 5 valid view layouts and FP32/BF16/FP16. The layouts vary row padding, a one-element storage offset and K strides 1 or 2. Each storage ended exactly at the last valid element. For the 75 views, `_vec_ok` selected 128-bit loads 25 times, 64-bit loads eight times, and scalar loads 42 times. I compared packed codes, all padded block-scale bytes and the FP32 tensor-scale bits with the eager reference for all three rotations. Separate inputs isolated +0, −0, NaN, +Inf, −Inf, the smallest positive/negative subnormal, smallest normal and largest finite value of each dtype. The combined result was **306 comparisons, 0 mismatches**. K=1,15,17,127 raised on all three dtypes. With M=0 and K=48, all three returned packed shape `(0,24)`, zero block-scale bytes and global scale 1.0.

For a separate valid allocation-edge check, I placed a contiguous 3×48 view at the end of a 40 MiB device allocation. CUDA driver `cuMemGetAddressRange_v2` returned success and zero bytes between the storage end and the reported allocation boundary in all three dtypes. Each selected 128-bit I/O, completed after synchronization and matched the reference. This tests actual allocation endpoints. Protected neighboring pages and a compute-sanitizer run were outside this measurement.

**Scalar products, fallback, shapes and state.** At M=6, N=128, K=256, the following are the FP32 bit patterns written by the quantizer for alpha and independently returned by `aten::mul`. The signed decimal interpretation is `view(torch.int32)`. In every row, direct GEMM and `torch._scaled_mm_v2` on the same quantized operands returned bitwise-identical BF16, FP16 and FP32 outputs, including their nonfinite outputs.

| Activation scale × weight scale | Fused / PyTorch bits |
|---|---:|
| +0 × 1 | 0 / 0 |
| −0 × 1 | −2147483648 / −2147483648 |
| NaN × 1 | 2147483647 / 2147483647 |
| +Inf × 1 | 2139095040 / 2139095040 |
| 0 × +Inf | 2147483647 / 2147483647 |
| 1e30 × 1e30 | 2139095040 / 2139095040 |
| 1e−30 × 1e−20 | 0 / 0 |
| 1e−20 × 1e−20 | 71362 / 71362 |
| 1 × 1 | 1065353216 / 1065353216 |

A `(2,3,256)` input returned `(2,3,128)` bitwise equal to C5 for each output dtype. An empty leading dimension returned `(0,3,128)`. Bias is outside both layers’ APIs. A supplied bias produced `TypeError: NVFP4Linear.__init__() got an unexpected keyword argument 'bias'`. All 72 canonical direct GEMM comparisons passed. To exercise the otherwise unobserved fallback branch, I temporarily set the cached plan’s declared workspace to 4096 bytes, then restored it. The layer matched C5 in all three dtypes. CUPTI reported `[amax, quant, scale_product, fp4_gemm]` for the fallback. This is branch coverage with a simulated requirement. The public helper lacks the layer’s `plan.direct` branch and the selected heuristic returned zero workspace on the live grid.

On a warmed, non-default capture stream, eight input-changing replays interleaved with other eager calls matched the reference 8/8. Serial eager CPU/CUPTI traces gave `C5: aten::mul=True [amax, quant, scale_product, fp4_gemm]` and `C6: aten::mul=False [amax, quant, fp4_gemm]`. Consecutive cache queries with identical keys returned the same plan.

For the concurrent test, each host thread owned a stream, distinct activation/weight scales and a live serial reference. Both used M=32, N=128, K=256 and BF16 output, synchronized warm-up, a barrier before each ordinary direct call, and retained outputs through stream completion. The first 100 calls per thread yielded 99/100 correct per thread. A fresh process alternated unprotected and protected submissions twice. Unprotected counts were `[499,497]` and `[499,497]` out of 500 per thread. With a lock around host submission they were `[500,500]` both times. Driver execution remained asynchronous. Correct stream binding alone cannot protect shared descriptor mutations between CPU threads.

For a fresh adaptive butterfly layer at K=2048, C5 and C6 outputs matched bitwise at M=1023,1024,1025. Delegation was `False, True, True`. A separate checkpoint test constructed two K=256, N=128 layers with independently drawn angles, converted one using `from_layer`, then called `load_state_dict(other.state_dict())`. PyTorch reported `<All keys matched successfully>`. The new layer initially matched its source, and its public `quantize_input` still matched the loaded reference after the load. Its native forward disagreed in 766/768 elements. The delegated path reads the persistent C5-layout tables. The native path instead reads the stale `(16,G,4)` buffer. The same loaded K=2048 layer produced 130,831/130,944 differing outputs at M=1023 (native), then 0/131,072 at M=1024 (delegated). A second test applied `.bfloat16()` to a constructed full cross-mix wrapper at K=1408. Both of its own coefficient buffers changed dtype to BF16. The maximum absolute mix difference grew from 4.7683716e−7 to 0.010219097 on the same input.

At M=17, K=256, identity, the cached pass-1 tile was `(64,1,1)`. Installing a valid `(32,1,1,512)` override updated the tile selector, yet the cached launch still returned `(64,1,1)`. The override was removed after the probe. Both pass caches depend on mutable global tile tables outside the cache key. The benchmark's `options` context also saves, clears and restores a shared dictionary. Serial benchmark capture is covered. Independent simultaneous tuning contexts need isolated settings.

## Provenance and dispatch history

I fetched `origin` and branched from `a716333d6c692ebc35476fd61398def8289cbfbf`. The committed result passed `validate_results(..., canonical=True, output=CANONICAL_OUTPUT)`. Its SHA-256 is `ceff189c3298905b355705dab89532d3a80bbeaf7343a7d27c804865f83949a2`. The recorded producer is `47cdedd8ff03027f39dd9e225279626824a0aeff` with `git_dirty=false`. The retained log also reports a clean start. These observations establish source and artifact identity. The historical working tree itself is unavailable for direct inspection.

All seven `SUMMARY_MODULES` and all 20 modules in the driver’s static transitive import closure match the producer blobs byte for byte. Hashing each ordered module name, a NUL byte and its Git blob yields `b6f3c2a863a2d06d1d13b84b86b671ca8668701dcbac038e3af77f42e1b142c0`, exactly the recorded digest. I also compared these three inputs with their Git bytes at `88b473a`, `a4ebc54` and `47cdedd`.

| Input | SHA-256 in all four trees |
|---|---|
| `research/quant_ptq/nvfp4_inference_blackwell.py` | `2b3a2af2615061d952216fa1691fdbd2f659961a5706f555c76bf35593053192` |
| `results/cycle5_c5_03_nvfp4_120m_fusion.json` | `e0e435ef2ea8598df22bace95ab76fd10c43b2ff394bf635ab8d1548d8bcad82` |
| `results/cycle6_c6_01_wide_rotated_qat.json` | `c490a415568244bd1e2489d0004adc025d856e92452cb976c05efc2dcdf66bf9` |

| Retained evidence file | Measured SHA-256 | Check |
|---|---|---|
| `rejected_canonical_a4ebc54.json` | `33fabff660fea8da98c7fea98ff96975c88d098c826d5de9d642275e797efbb3` | Matches constant |
| `canonical_a4ebc54.log` | `f58cbf162b8d867e08c2f874cbae49a553afc2ccf3f906085b8b7cd1b33de1bd` | Matches constant. Ends with `ValueError: part_c 50m: kernel launches per forward`, exit 1 |
| `canonical_47cdedd.log` | `1a232f0c0c3f3898b1927b04556c4854c188c4ee3217d27a687335285cbed4a2` | No corresponding constant. Reports the canonical write and exit 0 |

The rejected run completed the timed grid. The next commit fixed the Part C launch count and also changed the model under test. Previously every rotation delegated above `M*K > 2^23`. The new rules delegate identity from `2^23` inclusive and butterfly from `2^21` inclusive if `K >= 2048`. Hadamard stays on C6. The code comments cite RTX 5080 measurements on overlapping shapes. The four C5-relative regressions listed in the rejected summary are precisely the four delegated cells in the accepted summary. Two large cells already delegated before the change. The new summary explicitly excludes delegated cells from the regression count. Both runs beat BF16 in 31/32 cells and miss butterfly at 1024 × 512 × 1408. At 2048 × 8192 × 8192 the accepted ablation reports native butterfly at 478.82 µs and C5 at 451.59 µs. All three fused full-butterfly model medians moved by less than 0.04 ms between attempts.

The switch is disclosed, performance-informed routing after the first attempt. The evidence supports the selected dispatcher on the measured shapes. A zero-regression count cannot establish a uniformly faster native quantizer or predict the crossover on unseen workloads. Freeze thresholds before evaluating disjoint shapes and publish the underlying sweep if making that broader claim.

## Offline validation and artifact handling

The canonical SHA-256 is the identity check. The offline validator also checks the recorded data for consistency. The following probes used fresh in-memory copies. Each copy went through the full canonical validator with the checkout’s canonical path. None wrote to the results directory.

| Changed evidence | Observed validator result |
|---|---|
| First Part B heuristic row reports failed bit equality and negative graph time | Accepted |
| Direct Part B trace reports no raw kernels and includes `aten::mul` in its CPU operators, while its separate kernel-kind list stays intact | Accepted |
| Large ablation row has an empty delegation map | Accepted |
| Ordinary Part A shape has an empty delegation map | Rejected, `delegation record` |
| Wide Kron row has negative relative logits error and a logits population of one with over 100 million reported mismatches | Accepted |
| 50m C6 Hadamard CUPTI summary has zero FP4 GEMMs | Rejected, `kernel launches per forward` |
| 50m Pareto frontier is empty | Rejected, `pareto.50m.frontier: lengths differ` |
| Producer dirty flag is true | Rejected, `a canonical result needs a 40-hex git_sha and a clean tree (git_dirty False)` |
| Source-identical results commit `a716333` as the producer with its corresponding run configuration digest | Accepted |

The validator verifies the committed code under the claimed producer. Several commits can have those bytes. The published file hash independently distinguishes this artifact. The history constant includes two file hashes, but offline validation never reads the retained artifacts. The canonical tests pin the published result separately. Cross-check raw kernel names with classifications and CPU operators, enforce positive durations and heuristic integrity, validate both delegation maps, and check population counts, error bounds and zero/parity relationships. Add an external receipt check for the historical artifacts when they are available. Size the claims of the offline mode to the evidence it actually reads.

The guarded writer traverses parent components relative to a verified directory descriptor using `O_NOFOLLOW`, checks the leaf, creates an exclusive temporary file, reopens it with `O_NOFOLLOW`, verifies round-trip JSON, and renames with both directory descriptors. The suite exercises symlinked directories and leaves, traversal, reserved aliases and outputs outside the checkout on private fixtures. The write-once check happens at run admission. In a private mock repository, two successive guarded publications of ordinary data to the same regular destination left `{"trial": 2}`. Thus atomic replacement is verified, while two admitted producers can still overwrite each other. Use atomic no-replace publication if concurrent canonical producers are supported.


I checked the unmodified artifact against several of the missing predicates separately. All four raw Part B traces classify to their recorded kernel lists. Both C6 traces omit CPU `aten::mul` and both C5/PyTorch traces contain it. All seven returned algorithm configurations report exact output, zero workspace and positive graph time (12.371–12.482 µs). All 236 Part A per-kernel records have positive finite duration and agree with the raw kernel name classification. Both ablation maps match the dispatch rules. All 274 timed method records have matching counts for GPU rounds, event-sample rows and stored means.

## Reproduction protocol

Python and pytest came from `/home/ryan/Documents/projects/ai/.venv/bin`. All application imports came from the audit checkout through `PYTHONPATH=.`. The only CUDA device was the RTX 5080, selected with `CUDA_VISIBLE_DEVICES=0`. The initial load average was 1.75 on 16 logical CPUs. Desktop applications were resident on the card. Audit GPU jobs ran serially. Temporary files, compiler caches and pytest outputs were private to this run.

For Part C, use the C6-02 driver as `b`, call `b.c503.set_numerics()`, and initialize each `Transformer(b.PART_C_CONFIGS[name])` on CUDA exactly as in `part_c_model`, with seed 1, BF16 weights, eval mode, and random token IDs. Batches are 16, 8 and 4 at 512 tokens. Generate per-layer angles with the device generator seeded to 6. Construct the dynamic Hadamard model with `NVFP4Linear.from_weight` and `FastNVFP4Linear.from_layer`. For each wide family, use `_wide_modules` and install `FastCrossInput(FastNVFP4Linear.from_layer(layer), cross)`. Compare each model’s logits with the reference wrapper holding the same quantized layer. Keep every model alive, warm the compiled BF16 model three times under inference mode, and capture the five uncompiled models into one shared pool. The following is the timing core. All methods compute full logits.

```python
compiled = torch.compile(bf16_model)
with torch.inference_mode():
    for _ in range(3):
        compiled(tokens)
    functions = {"compiled": lambda: compiled(tokens)}
    pool = torch.cuda.graph_pool_handle()
    for name, model in graph_models.items():
        functions[name] = b.c503.capture_shared(lambda m=model: m(tokens), pool)
    flush = torch.empty(4 * torch.cuda.get_device_properties(0).L2_cache_size,
                        dtype=torch.uint8, device="cuda")
    records, orders = b.e2e.williams_timing(functions, squares=2, iters=10,
                                             flush=flush, wall_iters=2)
```

The six-way comparison gives 12 balanced rounds, 120 flushed event samples and 24 wall samples per method per model. GPU medians and ranges use the 12 round means. The cache flush precedes the start event. Compilation, construction, profiling and parity checks are outside the timed interval. Compiled BF16 uses default Inductor settings, matching C5-03. The five graph paths replay one at a time on the same stream and discard their outputs during timing. Weights are random in this timing protocol. Checkpoint scores below use trained weights. The timing study covers full-batch forward execution on one machine and date.

For the layer spot checks, use `b._inputs(device, M, N, K, 1)` and `b._layers`. Capture BF16 `x @ W.T` plus C5, default C6 and C6 with `delegate_large=False` for each of identity, Hadamard and butterfly. Check every quantized output bitwise against its C5 partner. Run the same event protocol over these ten methods, giving 20 rounds and 200 event samples per method. Profile captured graphs in a separate untimed pass. The 4× L2 flush applies to each individual event measurement in both experiments.

For the checkpoint comparison, read the canonical C6-01 units `ab/muon_ss3_54` and `c/c5_03_baseline` for seeds 1–3. Use `c503.load_checkpoint` to verify file and tensor hashes. Verify the data manifest, regenerate the 128 train calibration sequences at seed 20261126, gather the C6-01 Hessians and build the published GPTQ rotations with `Study.candidate` (150 butterfly steps, AWQ alpha 0). Compare the resulting effective weight digest with the matching fake-quant row. Wrap the identical encoded hardware layers in `FastCrossInput` and evaluate the dynamic method with the canonical `q.evaluate_split` and microbatch. The test file SHA-256 is `4eb46f1424856849d2973923dbf3c886d37eb743cf22053fc7e1eea1f41ab92a`. Each score covers 3,906 windows, or 1,999,872 targets. When the fused score differs from the archived score, repeat the full scoring with the original hardware wrapper as a control. The trained BF16 comparator uses the same checkpoint cast to BF16 and the default compiled forward under inference mode, with the same FP32 cross-entropy scoring.


The runtime was PyTorch 2.14.1+cu130, Triton 3.8.0, CUDA 13.0, cuDNN 92400, driver 615.78.08 and Python 3.13.12. The flush buffer held 268,435,456 bytes (256 MiB). TF32 and reduced-precision BF16/FP16 reductions were disabled, as in both producer scripts.

```bash
PYTHONPATH=. /home/ryan/Documents/projects/ai/.venv/bin/pytest tests/test_c6_02_nvfp4_small_shape_fusion.py tests/test_nvfp4_inference.py tests/test_c5_03_nvfp4_120m_fusion.py tests/test_c5_02_isoloss_ptq_mitigation.py tests/test_c6_01_wide_rotated_qat.py -q
```

Exact pytest footer: `729 passed, 85 warnings in 1161.72s (0:19:21)`. Zero failures and zero skips. Warnings included the documented sync-debug limitation, empty graph warnings in capture-refusal tests, and a Triton AST deprecation. The canonical result independently records 300 parity cases, comprising 288 activation cases, nine launcher cases and three packed-table cases.

## Source locations

- Quantizer vector loads, sizing checks and partial-count contract. [C6-02 kernels and launchers](../research/quant_ptq/nvfp4_small_shape_blackwell.py#L279)
- cuBLASLt plan submission, cache and layer routing. [C6-02 direct GEMM](../research/quant_ptq/nvfp4_small_shape_blackwell.py#L763)
- Timing workload and Pareto mapping. [C6-02 Part C](../research/quant_ptq/bench_c6_02_nvfp4_small_shape_fusion.py#L526)
- Compiled BF16 comparator in the previous study. [C5-03 Part B](../research/quant_ptq/bench_c5_03_nvfp4_120m_fusion.py#L873)
- Trained GPTQ reconstruction and reference hardware wrapper. [C6-01 study](../research/quant_ptq/bench_c6_01_wide_rotated_qat.py#L683)
- Offline consistency checks. [C6-02 validator](../research/quant_ptq/bench_c6_02_nvfp4_small_shape_fusion.py#L1201)

## Post-Remediation v2 Re-Check

**Verdict: FAIL.** The compiled BF16 baseline, model-specific Pareto conclusions and trained scores are supported on the RTX 5080. The concurrency, state-loading, FP32 conversion, workspace and publication fixes passed their live checks. Three residual findings remain.

1. **Material timing caveat.** Two fresh sessions measured a 1.154× dynamic Hadamard speedup over C5 at `(1024, 512, 1408)`. The canonical 1.846× comes from long events concentrated in the C5 rounds. The artifact should identify that cell and the elevated 50m absolute latencies explicitly.
2. **Offline evidence checks.** All five original probes now fail validation. Fresh records with incomplete timing means, negative launch counts, contradictory logits errors or a Boolean seed still pass. Require consistency at those fields before lifting the validator hold.
3. **Exact-wrapper compatibility.** Exact mode passes for the two complete wide NVFP4 models. It changes the autocast context of a general inner module, which can change its output dtype and values.

### Provenance and scope

The canonical file at result commit `3e461419518b30781056d597ac56dd814cc5d91d` has SHA-256 `91deb1d2adc3eab3d26bf848ec4c6193a54178bfb145b92770264113d348583d`. It records schema `cycle6_c6_02.v2`, producer `411eaf01381c83115887c910d267cd560859ed90`, and `git_dirty=false`. All seven `SUMMARY_MODULES` have identical bytes at both commits and in this checkout. The producer’s 20-module import closure exactly matches the payload. Hashing the ordered module names, NUL separators and producer blobs gives `e101bd74f5417a070c786bc58f29d30e07da2b0e32a0fcee770f26a0c39abd12`. The result commit adds the result and changes its test pins.

| Pinned input | SHA-256 verified in the checkout and in all three Git trees¹ |
|---|---|
| `nvfp4_inference_blackwell.py` | `2b3a2af2615061d952216fa1691fdbd2f659961a5706f555c76bf35593053192` |
| `cycle5_c5_03_nvfp4_120m_fusion.json` | `e0e435ef2ea8598df22bace95ab76fd10c43b2ff394bf635ab8d1548d8bcad82` |
| `cycle6_c6_01_wide_rotated_qat.json` | `c490a415568244bd1e2489d0004adc025d856e92452cb976c05efc2dcdf66bf9` |

¹ The v1 result commit `a716333`, v2 producer, and v2 result. I also compared their bytes directly. In the remediation worktree’s `artifacts/c6_02_run_evidence`, `verify_superseded_receipts()` verified the rejected v1 JSON, superseded v1 JSON, and both corresponding logs. The fifth check verified the superseded result blob in Git. All five returned `verified`.

### Finding 1 · Compiled BF16 and the desktop-load burst

The canonical comparison includes default Inductor BF16 in the same 19-method Williams design as the graphed methods. I verified the orders, all 76 round means, 760 flushed CUDA events and 152 wall observations **per method per preset**. The graph path replays an eager BF16 forward. The compiled path also fuses elementwise operations. The profile runs separately from timing.

At the target Part A shape `(1024, 512, 1408)`, the choice of baseline changes the speedup. The canonical dynamic C6 Hadamard time of **20.635 µs** gives **1.122× versus BF16** at 23.150 µs, and **1.846× versus C5 Hadamard** at 38.086 µs. The second comparison is materially distorted. The C5 round means span 23.110–59.696 µs, with an IQR of 20.601 µs. Of 440 individual C5 events, 25 exceeded 100 µs, one in each of 25 distinct rounds. The largest event took 380.384 µs. Those 25 rounds form a majority of the 44 observations used for the median. C6 had six such rounds, BF16 had 18. Median individual event times were 24.544 µs for C5 and 20.480 µs for C6. These event medians diagnose the stalls. The published estimator remains the median of round means.

| Same target, dynamic Hadamard | C5 µs | C6 µs | C5 / C6 |
|---|---:|---:|---:|
| Canonical v2 main grid | 38.086 | 20.635 | 1.846× |
| Canonical v2 separate ablation | 24.318 | 20.686 | 1.176× |
| Retained first attempt | 23.968 | 20.744 | 1.155× |
| Retained v1 | 23.931 | 20.592 | 1.162× |
| Original independent ten-method session | 24.71 | 21.92 | 1.127× |
| Fresh v2 session 1, canonical target protocol | 23.770 | 20.594 | 1.154× |
| Fresh v2 session 2, canonical target protocol | 23.760 | 20.590 | 1.154× |

Both fresh sessions passed all operand and output checks. Each repeated the 11-method Williams protocol with 44 rounds and 440 events per method. Their C5 Hadamard IQRs were 0.482 and 0.398 µs. The corresponding fresh C6 speedups against BF16 both rounded to 1.114×. Thus the comparison near 1.11× depends on the chosen baseline.

The canonical start snapshot showed 22% utilization, P8 and 405 MHz memory. Telemetry has no timestamped samples at the individual stalls. Attributing them to a particular desktop process would need another measurement. The raw events establish the asymmetric slowdown without that attribution. The artifact’s generic 1–2% spread caveat understates it. Preserve the raw measurements and label the 1.846× ratio as an affected session. Use the separately measured ratio and its protocol when describing expected speed. The in-session count of 31/32 wins against BF16 and the identity of the trailing butterfly cell are correctly reported.

At 50m the canonical compiled BF16, dynamic Hadamard and fused full butterfly medians were 12.626, 11.499 and 14.219 ms. These exceed the original independent medians by 7.31%, 7.90% and 7.38%, respectively. Their canonical minimum round means were 11.711, 10.681 and 13.236 ms. The shift affects absolute latency. The dominance comparison survives it. Compiled BF16 has lower latency and lower measured NLL than each of the three wide 50m methods. At 120m full Kron and full butterfly stay on the primary frontier, with a latency–quality tradeoff. Full butterfly takes 10.100 ms versus 10.775 ms for compiled BF16, a 6.26% latency reduction. The 120m C6-01 grid has no trained b64 quality point.

The 50m graph-only frontier includes full butterfly. The primary frontier excludes it and includes both BF16 methods because their recorded NLLs differ by only 0.00000161 in favor of the slower graph point. Treat that membership as exact arithmetic on recorded means. The data support no practical quality distinction at that scale.

#### Independent model timings and profiles

I repeated the six-method protocol in the original audit using v2 code, seed 1 and the three canonical batches. The methods shared one Williams session per preset with 12 balanced rounds, 120 flushed CUDA events and 24 wall observations each. Compilation and CUPTI collection were outside timing. Values below are fresh median round means in milliseconds.

| Forward | 50m | 120m | w2048_6l |
|---|---:|---:|---:|
| Compiled BF16 | 11.726 | 10.777 | 16.538 |
| BF16 CUDA graph | 14.492 | 12.871 | 17.522 |
| C6 dynamic Hadamard b16 | 10.675 | 8.299 | 8.031 |
| C6 fused Hadamard b64 | 12.681 | 9.265 | 9.016 |
| C6 fused full Kron | 12.554 | 9.422 | 9.512 |
| C6 fused full butterfly | 13.350 | 10.067 | 10.034 |

At 50m the compiled round means ranged from 11.718 to 11.829 ms, full Kron from 12.544 to 12.639 ms and full butterfly from 13.337 to 13.440 ms. At 120m compiled BF16 ranged from 10.759 to 11.265 ms and full butterfly from 10.032 to 10.483 ms. Wall medians preserve the same wide-versus-compiled ordering. The fresh six-method run and the canonical 19-method run answer the same dominance question with different session lengths.

| Preset | CUPTI launches, compiled / graph | GEMMs per forward on each | Compiled vs graph logits, relative L2 | Max absolute error |
|---|---:|---:|---:|---:|
| 50m | 134 / 375 | 73 | 0.01136806 | 0.033203125 |
| 120m | 134 / 375 | 73 | 0.01328954 | 0.046875 |
| w2048_6l | 68 / 189 | 37 | 0.02006280 | 0.119140625 |

The independently profiled launch counts and freshly computed errors match all three canonical records.

#### Six trained checkpoints

I rescored the three C6-01 50m Muon checkpoints and the three C5-03 120m baseline checkpoints via the v2 compiled BF16 evaluation path. The loader checked each checkpoint’s file and tensor hashes. Each forward used the complete SHA-256-verified test split, 3,906 windows and 1,999,872 targets per seed. Each of the six live NLLs matched the corresponding canonical float exactly.

| Preset | Seed 1 | Seed 2 | Seed 3 | Live mean |
|---|---:|---:|---:|---:|
| 50m | 5.3936286222 | 5.3765366057 | 5.3759370624 | **5.3820340967** |
| 120m | 4.6334629811 | 4.6264842420 | 4.6128800969 | **4.6242757733** |

### Findings 2, 3, 5, 7 and 8 · Runtime contracts

**2. Concurrent direct GEMM.** I ran two Python threads at M=32, N=128, K=256, with separate CUDA streams, distinct activation and weight scale buffers, one warmed plan and a barrier immediately before every call. I retained each output through stream completion and compared its BF16 bits with a serial public-API reference. Both 500-call repeats returned 500/500 correct on each thread. No external submission lock was used. The plan’s own lock covers both pointer updates and host submission.

**2 and 7. Eviction and teardown.** Eviction drops the cache’s reference. A worker retained the oldest plan while another worker inserted 256 new, real cuBLASLt shapes. The cache stayed within its normal 256-entry bound. The second worker then cleared all 256 resident plans. The evicted plan stayed open and produced 500/500 exact outputs after the clear. Explicit clearing closes the plans still resident in the cache and shares the submission lock. All 100 simultaneous clear/submit trials happened to submit first and returned the correct output. I then forced both orders. Clearing between lookup and submission gave the documented closed-plan exception. Clearing while submission held the lock waited until release, and the submitted matrix was bit-exact. Callers that require uninterrupted service should clear the cache after their submitters stop.

**3. State loading and captured graphs.** I used two butterfly checkpoints with different weights and angles, saved a nested state dict to an in-memory file, then loaded it with `torch.load(weights_only=True)` and the default `load_state_dict`. Each graph captured the old layer and first replayed against its old reference. After the load, I replayed each captured graph and compared its output with the new C5 reference. The packed FP32 buffer kept its address. All four comparisons had zero bit mismatches. The count of output elements that changed from the old replay confirms that the graph read new state.

| K | M | Dispatch | Changed BF16 outputs |
|---:|---:|---|---:|
| 2048 | 1023 | native | 130,815 |
| 2048 | 1024 | delegated | 130,963 |
| 256 | 6 | native | 768 |
| 256 | 37 | native | 4,733 |

These results cover a same-device, same-shape, in-place load with `assign=False`.

**5. FP32 mix constants.** At K=1408 I exercised `.bfloat16()`, `.half()`, `.to(dtype=torch.float16)`, and a CPU/CUDA round trip on both sides of wrapper construction. In all eight cases the values and FP32 dtype of `h` and `mix` were preserved, and mixed tensors and layer outputs matched bitwise. The suite also covered `.to("cuda", torch.float16)` and inspected CPU placement. Substituting an FP16 buffer for either matrix made fused `mixed()` raise the intended error. Exact mode uses the reference matrices directly.

**7. Workspace and partial initialization.** The suite checked that index-less `torch.device("cuda")` and explicit `cuda:0` return the same plan. With a simulated 4,096-byte requirement, the direct plan rejected a missing workspace, 4,095 bytes, the wrong dtype, a strided view and a CPU buffer. A valid contiguous CUDA byte buffer succeeded. The public helper allocated a suitable workspace and matched the public GEMM reference bitwise. The real selected algorithms required zero bytes. I independently injected failures at the third layout creation and at heuristic selection, tracing native create and destroy calls by pointer. Cleanup released exactly one descriptor and two layouts in the first case, and one descriptor, four layouts and one preference in the second. Repeated close was idempotent, and the suite confirmed that a later call to a closed plan raises.

**8. Tuning and publication.** The live tests changed pass-1 and pass-2 tile overrides after warming the original shape, checked that the resolved plans changed immediately, verified bit-exact quantization, then restored the original plans. Opposite `pdl` choices in simultaneous threads and nested `vector_io` overrides remained independent and restored their defaults. In a private mock results directory, the first canonical write succeeded. The second raised `FileExistsError`. The original contents survived and the test found no temporary file. These checks close the original tuning and write-once findings.

### Finding 4 · Exact mode and w2048_6l

The live suite verified both the mixed FP32 values and the wrapped NVFP4 output bitwise for the w2048_6l down projection at K=5632, M=2048, N=2048, for full Kron and full butterfly. I also evaluated both complete six-block models at batch 4 and sequence length 512 using the producer’s random weights, tokens and angles. Switching all full-mix wrappers to `exact=True` gave **zero differences across 103,022,592 logits per family** against their C6-01 hardware wrappers. In the default fused mode, full Kron differed at 102,127,503 logits (relative L2 0.209078) and full butterfly at 102,294,865 (0.253306). The v2 result calls these paths latency-only and assigns neither a trained NLL nor a Pareto cell. The “near-0 parity” phrase is gone from the result and its driver. An old test comment still uses it for small-input mixed values.

A separate compatibility issue remains for callers that pass an autocast-sensitive inner module. The reference `CrossInput.forward` disables autocast only for the mix, then calls the layer in its caller’s precision context. `FastCrossInput.forward(exact=True)` disables autocast for both. Under CUDA BF16 autocast at K=1408, the same FP32 `nn.Linear` returned BF16 through the reference and FP32 through the exact wrapper, with a maximum absolute output difference of 0.0095694. The NVFP4 hardware and repository emulation controls both returned bitwise-equal BF16. To preserve the public wrapper’s behavior, compute the mix inside the disabled-autocast block and call the inner module after leaving it.

### Finding 6 · Offline validation

The original five contradictory records now fail full canonical validation on separate in-memory copies. They were the failed heuristic equality with negative time, the empty direct kernel trace with a CPU scale multiply, the empty large-ablation delegation map, impossible wide logits counts with negative error, and the source-identical result commit substituted for its producer. For the last check I supplied `3e46141` and recomputed the run configuration hash. The validator rejected the commit because its tree already contains the result.

Fresh controls also rejected a wrong trained token count, substitution of another seed’s checkpoint file, a negative compiled raw event, swapped primary and graph-only frontiers, an omitted study history, and a changed receipt digest in the published history. I independently recomputed both frontiers with pairwise dominance. Both match the canonical lists for both presets.

The following contradictions still passed the full validator. These tests used copies in memory. The original canonical file passed independently.

| Accepted evidence | Consequence and required check |
|---|---|
| The 50m compiled GPU record retained 76 ten-event rows but only its first round mean. Updating statistics and dependent summaries yielded a one-round median of 12.652883 ms. The measured 76-round median is 12.625928 ms. | The validator compares means with `zip` and checks the reported length against itself. Require the protocol round count for raw rows, GPU means and the stored count. Derive statistics from the complete raw array. |
| The compiled CUPTI profile recorded −1 launch of kind `other`. | The compiled-profile rules check GEMMs and excluded kinds. Validate every launch count as finite and nonnegative, and compatible with the three profiled repetitions. |
| The compiled logits record reported zero maximum absolute error and positive relative L2 error on the same logits. A separate probe gave the compiled entry in `logits_rel_diff_vs_bf16` a negative relative error. | Enforce the zero-error relationship on the same tensor pair. Validate the keys and finite nonnegative values of the full relative-error table. |
| The trained row identified the first seed with JSON `true`. | Python compares `True` equal to `1`. Require integer seed identifiers before looking up checkpoint records. |

All 277 timing records in the pinned result have matching raw and reported round counts. I recomputed their means and medians from the events. The canonical logits and launch records pass the additional checks above. The SHA-256 identifies this file. It does not validate a new measurement submitted to the same scientific checks.

### Verification environment

2026-11-28, NVIDIA GeForce RTX 5080, compute capability 12.0. All audit GPU jobs ran serially. I used the supplied project venv, `PYTHONPATH=.`, the isolated checkout and a read-only mount of the main project for checkpoint, data and evidence access. Pytest, captures, scratch files and benchmark outputs used private temporary storage. The timing repetitions covered the affected target shape and six full-model methods on each preset. The complete archived grid was checked offline.

```bash
PYTHONPATH=. /home/ryan/Documents/projects/ai/.venv/bin/pytest tests/test_c6_02_nvfp4_small_shape_fusion.py tests/test_nvfp4_inference.py tests/test_c5_03_nvfp4_120m_fusion.py tests/test_c5_02_isoloss_ptq_mitigation.py tests/test_c6_01_wide_rotated_qat.py -q
```

Exact live footer: `780 passed, 85 warnings in 905.83s (0:15:05)`. **780 passed, 0 failed, 0 skipped.** Warnings cover PyTorch sync-debug limits, deliberately empty graphs in capture-refusal tests, and a Triton AST deprecation.

Implementation references include the [timing validator](../research/quant_ptq/bench_c6_02_nvfp4_small_shape_fusion.py#L1412), [Part C evidence checks](../research/quant_ptq/bench_c6_02_nvfp4_small_shape_fusion.py#L1554), [plan ownership and cache teardown](../research/quant_ptq/nvfp4_small_shape_blackwell.py#L980), [exact wrapper](../research/quant_ptq/nvfp4_small_shape_blackwell.py#L1229) and [reference wrapper](../research/quant_ptq/rotated_qat.py#L273).
