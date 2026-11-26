# C4-02 audit on the RTX 5080

**Verdict: FAIL against the full audit criteria.** The native W4A4 path works on the benchmark models. All 204 repository tests passed. The committed artifact cannot support the requested independent recomputation of its original medians. Two additional GPU probes found silent numerical errors outside the benchmark workload. The output guard also permits a partial run in one checkout to replace the canonical result in another.

Audited snapshot `5951dd22c9f40d6382263d63d4b2ad9885f04747`, November 26, 2026. Its only change from `34064164e96011674d058d1a78ddd28ad8a8407d` is the result JSON. The JSON records that exact parent SHA and `git_dirty: false`. The producer checks tracked files using `--untracked-files=no`. The artifact SHA-256 is `28667da258ceba31b96acb78b4296c160717734e7867301af524e60f9b1a76ef`. It parsed with duplicate-key and non-finite-constant rejection and serialized with `allow_nan=False`.

Live environment: GeForce RTX 5080 (sm_120), driver 615.78.08, CUDA 13.0, PyTorch 2.14.1+cu130, Triton 3.8.0 and Python 3.13.12.

## Severity-ranked findings

| Severity | Finding and measured consequence | Action |
|---|---|---|
| HIGH | The original A/B measurements are unavailable. At lines 86–89 of `bench_nvfp4_e2e.py`, `_stats` discards its input. The 366 GPU/wall distributions store six summary fields each. The 69,960 underlying GPU event durations, 6,996 GPU round means and 13,404 individual wall timings cannot be recovered from those fields. Every downstream formula checked out. | Persist timings with round, order, method and execution mode. Recompute summaries from those records before publication. |
| MEDIUM | Finite FP32 inputs can produce NaN hardware outputs. At a tensor maximum of `2**-149` or `2**-140`, the dynamic global scale rounds to zero. A zero group then computes `0/0` and stores E4M3 NaN. On a 17×96 input with just one zero group and 48 output features, all 48 outputs of that row were NaN. The eager encoder agreed bit for bit. | Define underflow behavior jointly for the reference and Triton implementation. Test both decoded values and native GEMM finiteness. |
| MEDIUM | The public `quantize_activation` accepts same-shaped strided butterfly tables and silently misreads them. At lines 93–106 of `nvfp4_inference_blackwell.py`, the kernel assumes dense FP32 tables. The check at line 246 inspects only the first table's shape. With logically identical tables of stride (64,16,2), 810 of 816 packed bytes and 91 scale bytes changed. The module's table factory supplies dense tensors. | Validate both tables' shape, strides, dtype and device, or support their strides. Add an equal-values, different-layout test. |
| MEDIUM | The canonical-output guard applies to the current checkout's resolved path. It correctly rejected quick, partial and unverified runs targeting that path, including a symlink alias. From this audit checkout, the parser accepted `--parts A --output /home/ryan/Documents/projects/ai/results/cycle4_c4_02_nvfp4_e2e.json`. A writer probe replaced an identically named *scratch copy* with a marker. The main checkout was never a write target. Default canonical arguments also explicitly permit replacement in the current checkout. | Reserve the canonical result filename for canonical runs across checkout paths if the file needs that protection. Check device capability too if the name promises an RTX 5080 result. |
| LOW | The standalone quantization stage can consume another rotation's amax partials. At lines 180–188 of `bench_nvfp4_e2e.py`, all three stage pairs share one workspace. The Williams order does not pair a quant stage with its own amax. The isolated stages also run cold on preallocated buffers. For graph Hadamard, their median sums exceed measured end-to-end latency by 5.3–37.7%. | Precompute a separate workspace for each quant stage. Use an in-forward trace for an additive breakdown. The end-to-end calls already compute their own partials. |
| LOW | The recorded `hw_weights_decode_to_fakequant_weights` flag checks only block zero of each projection kind. The comparison at lines 458–460 of `bench_nvfp4_e2e.py` covers 5 of 60 matrices per method, and `torch.equal` treats signed zeros as equal. | Check every encoded matrix against the returned GPTQ values in rotated coordinates. Compare integer views for a literal bitwise claim. |
| LOW | Part C retrains its own checkpoint. The full fresh 320-step run gave baseline NLL 5.396820 versus 5.394250 in the artifact. Relative method effects were close. The reported “best” method is selected from test NLL among five fixed candidates. | Preserve a final checkpoint fingerprint or checkpoint for exact replay. Use validation for method selection if the result will guide deployment. |

## Kernel, GEMM and graph evidence

The tested entry point is named `quantize_activation` in this revision. I crossed all seven M values (1, 17, 127, 128, 129, 255, 257) with all seven K values (64, 80, 96, 144, 512, 1408, 4096). Each shape used three rotations, dynamic and fixed 0.0078125 global scales, and 14 input cases. Those cases covered random BF16, transposed FP32, an offset stride-two view, zeros, alternating signs at 1e-6, 1e4 and 65504, four subnormal patterns spanning FP16/BF16/FP32, NaN, +Inf and −Inf.

* **4,116/4,116 encodings** matched in packed bytes, all swizzled scale bytes and the FP32 global-scale bits. The oracle applied the eager FP32 rotation, then `q.nvfp4_encode`, `q.pack_e2m1` and `q.to_blocked`. For static scaling it recomputed each block's E4M3 scale from the supplied scalar and used `q.e2m1_codes`. Non-finite encodings agreed with the reference. A dynamic NaN maximum selects global scale 1, while an infinite maximum can give global scale infinity and destroy finite blocks.
* **294/294 dirty-buffer launches** reproduced the oracle. Packed buffers started at 0xA5, swizzled buffers at 0xFF and amax workspaces at NaN. An independently calculated index for every padded slot verified it contained zero. **210/210 native GEMMs** on the five hardware-compatible K values matched the clean oracle output bit for bit. K=80 and 144 are valid for encoding. The public GEMM rejected both with `CUBLAS_STATUS_NOT_SUPPORTED` on this GPU.
* **Zeroing K padding matters on this hardware.** After a correct encode, I changed only padded scale bytes to 0x7F. At K=96 the GEMM changed in all 42 M/rotation/scale combinations. At M=17, N=48, overwriting only padded columns of valid rows made **816/816** outputs NaN. Overwriting only padded rows changed zero outputs. Finite column poisons 0x38 and 0x7E changed zero outputs in that probe. The normal kernel overwrote the initial poison before GEMM in every case.
* **GPTQ:** Three 48×160 weight matrices per rotation, FP64 Hessians from 513 inputs, seed 402. All nine encodings decoded to the same FP32 bits as both the returned fake-quant matrix and a second GPTQ call with recording disabled. All nine module outputs matched public `F.scaled_mm` bit for bit. Against FP32 multiplication of decoded operands, relative Frobenius error was 0.00160–0.00178 with BF16 outputs.
* **Synchronization and replay:** Six warmed configurations, five forwards each. Scoped CPU/CUDA profiler traces showed zero D2H copies, host synchronization calls or scalar extractions. Every forward issued one quant kernel, one SM120 block-scaled GEMM and a scale-multiply kernel. Dynamic mode added one amax kernel. All **36** graph replays matched fresh eager execution and the eager encoder plus public GEMM in BF16 bits after overwrites with random, zero, ±65504, 1e-6-scale, 1e4-scale and restored inputs. This covers fixed shapes after warmup.

## Timing arithmetic and interpretation

I checked all published A/B results in their original execution mode, including top-level copies. Part A supplied 78 method rows and 48 stage groups. I recomputed end-to-end ms, wall ms, effective TFLOP/s, BF16 speedup and percentage of the measured prequantized GEMM throughput from the stored medians. For the stages I verified the matching amax and quant medians, GEMM-only median and measured end-to-end total. Part B supplied 21 method rows. I recomputed wall ms, GPU ms, tokens/s and all three BF16 speedup ratios. All comparisons agreed within 1e-12 relative/absolute tolerance. The throughput percentage is `100 × GEMM-only ms / end-to-end ms`; it uses the measured prequantized path on the same GPU.

| Part A shape (M,N,K) | Graph Hadamard ms | Speedup vs graph BF16 | Measured GEMM throughput retained |
|---|---:|---:|---:|
| 16,4096,4096 | 0.028877 | 1.593× | 85.82% |
| 128,4096,4096 | 0.030720 | 1.803× | 80.67% |
| 1024,512,1408 | 0.026624 | 0.858× | 52.82% |
| 1024,2048,768 | 0.022938 | 1.692× | 66.27% |
| 1024,4096,4096 | 0.089322 | 3.815× | 74.88% |
| 2048,8192,8192 | 0.449163 | 5.278× | 81.82% |

Across each complete design, every method appears equally often in each position and immediately follows every other method equally often within rows. Canonical Part A uses 14 eager methods (28 rounds), then 13 graph methods (52 rounds) because the eager PyTorch reference fails graph capture at all six shapes. Part B uses seven methods and 28 rounds. Each GPU round averages ten event pairs. Part A's wall statistics pool two wall samples per round and Part B uses one. Eager and graph are separate timing passes in A, so cross-pass comparisons can include clock or thermal drift. Eager CUDA event intervals can also include time when the GPU waits for the host to enqueue work.

The RTX 5080 reports 67,108,864 bytes of L2. The benchmark writes a 268,435,456-byte buffer before each sample on the same stream, outside the timed interval. This is a cold-cache workload. Cache occupancy and eviction were not measured with hardware counters. Stage inputs in a real forward can be hot. Event resolution and host overhead matter most for short calls.

| Part B model | Parameters in the five quantized projection kinds | Graph Hadamard vs BF16 graph | Graph Hadamard vs compiled BF16 |
|---|---:|---:|---:|
| 50m | 59.93% | 1.253× | 1.015× |
| 120m | 68.72% | 1.441× | 1.211× |
| w2048_6l | 74.95% | 2.088× | 1.976× |

Embedding, RMSNorm, attention and the tied output head stay on the BF16 model path. Part B measures full fixed-length random-token forwards, without a KV cache or token sampling. The 1.5% 50m margin against compiled BF16 is specific to these measurements. Part A's smallest M is 16.

## Evaluation, data and execution

I hashed the local read-only shards and compared them with the manifest, the pins in `micro_rotated_ptq.py` and the committed JSON. The verified dataset is FineWeb-Edu `sample-10BT`, GPT-2 tokenization. The manifest check also verifies document-hash disjointness between splits.

| Local shard | Tokens | SHA-256 |
|---|---:|---|
| train.bin | 26,004,518 | `015ab83ceb742555c2398963a0bd1f3c5502dc52cf70a22bf173f1571326ca3a` |
| val.bin | 2,002,330 | `371b64a537f7b1e306ea174a868b05932a254a98b7fbdbf4146648d93180b0b1` |
| test.bin | 2,000,081 | `4eb46f1424856849d2973923dbf3c886d37eb743cf22053fc7e1eea1f41ab92a` |

For the full Part C rerun I independently regenerated all 128 starts using NumPy seed 20261126 and reconstructed each 512-token window directly from train.bin. The live inputs to the Hessian collector and **all five** static calibration calls matched these 65,536 tokens element for element. All 60 returned Hessians were FP64. The collector promotes inputs to double before computing the Gram matrix. Training and warmup draw from the train stream. The harness evaluates validation and test after training. Test data enters the PTQ experiment for scoring; the butterfly fit uses the weights. These checks support train-only fitting and calibration. Ranking methods afterwards uses the reported test scores.

Every evaluation row scored 3,906 complete windows, or 1,999,872 next-token targets. The harness uses a 2,000,081-token file and leaves a 208-token tail beyond the last input/target span. I independently recomputed all 16 rows' PPL, ΔNLL, absolute ΔPPL and relative ΔPPL from the recorded NLLs, including all top-level copies. The relative change equals `100 × expm1(method NLL − baseline NLL)`. The artifact contains aggregate NLLs. It does not contain individual window losses.

| Method | Canonical dynamic hardware ΔNLL | Canonical dynamic ΔPPL | Canonical static ΔPPL |
|---|---:|---:|---:|
| RTN | +0.018564 | +1.874% | +1.875% |
| GPTQ | +0.011531 | +1.160% | +1.184% |
| Hadamard RTN | +0.018713 | +1.889% | +1.889% |
| Hadamard GPTQ | +0.012266 | +1.234% | +1.243% |
| Adaptive butterfly GPTQ | +0.011130 | +1.119% | +1.146% |

I profiled the actual first evaluation batch of the baseline and every Part C row. Each of the ten hardware rows issued **60 `aten::_scaled_mm_v2` calls** and launched the SM120 `bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1` kernel. The module calls the same operator as public `F.scaled_mm` directly. The baseline, the five W4A16 rows and the explicit FP32 emulation row issued zero such calls. The full fresh evaluation retained adaptive butterfly GPTQ as the lowest dynamic NLL (+1.12% PPL against its own baseline). Separate test runs shared the GPU during retraining; no inference performance number here comes from that retraining run.

## Independent timing replay

I repeated all six Part A shapes and all three Part B models with the canonical square and iteration counts. Instrumentation retained each measured `Event.elapsed_time` value before the benchmark averaged it, plus every sample passed to `_stats`. The timing run completed after the other substantial audit GPU jobs. I grouped the 69,960 raw durations by Williams order, independently formed 6,996 round means, and recalculated median, mean, min, max and interquartile range across all 366 GPU/wall distributions. All 1,830 summary statistics matched the fresh producer within 1e-12 relative/absolute tolerance. All fresh A/B derived fields also agreed. The six graph Hadamard speedups were 1.586×, 1.806×, 0.860×, 1.688×, 3.825× and 5.288×. Against compiled BF16 the three full-model speedups were 1.014×, 1.216× and 1.978×. These fresh observations reproduce the main performance conclusions. Historical raw timings remain unavailable.

## Reproduction

Run from the audited snapshot. The original test command returned **204 passed, 32 warnings in 225.59 s** (zero skips). Warnings concerned experimental sync detection, the intentionally refused first graph capture, and a Triton Python deprecation.

```bash
set -eu
umask 077
audit_tmp=$(mktemp -d "$HOME/tmp/pi-c402-replay.XXXXXX")
trap 'if [ -d "$audit_tmp" ]; then rm -r -- "$audit_tmp"; fi' EXIT
export TMPDIR="$audit_tmp" TMP="$audit_tmp" TEMP="$audit_tmp"
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider'
export TRITON_CACHE_DIR="$audit_tmp/triton" TORCHINDUCTOR_CACHE_DIR="$audit_tmp/inductor"
py=/home/ryan/Documents/projects/ai/.venv/bin/python
data=/home/ryan/Documents/projects/ai/artifacts/real_text
sha256sum "$data"/{train,val,test}.bin results/cycle4_c4_02_nvfp4_e2e.json
/home/ryan/Documents/projects/ai/.venv/bin/pytest tests/test_nvfp4_inference.py tests/test_micro_rotated_ptq.py -q
"$py" research/quant_ptq/bench_nvfp4_e2e.py --data-dir "$data" --output "$audit_tmp/repeat.json"
jq '.part_a.shapes[0].eager.bf16_cublas.gpu | keys' results/cycle4_c4_02_nvfp4_e2e.json
```

The following focused GPU reproducer checks underflow, strided tables and the distinction between dirty destination buffers and deliberate corruption after quantization. Use the same environment as above.

```bash
"$py" - <<'PY'
import torch
from research.quant_ptq import nvfp4_inference_blackwell as nb
torch.manual_seed(42)
d = "cuda"
angles = torch.randn(6, 4, 8, device=d)
x = torch.randn(17, 96, device=d)
tables = (angles.cos(), angles.sin())
views = tuple(torch.stack((t, torch.zeros_like(t)), -1)[..., 0] for t in tables)
assert all(torch.equal(a, b) for a, b in zip(tables, views))
def encode(t):
    return nb.quantize_activation(x, "adaptive_butterfly_b16", t)
a, b = encode(tables), encode(views)
print("strided table differences", (a.packed.view(torch.uint8) != b.packed.view(torch.uint8)).sum().item(),
      (a.scale.view(torch.uint8) != b.scale.view(torch.uint8)).sum().item())

w = nb.NVFP4Linear.from_weight(torch.ones(48, 96, device=d))
tiny = torch.full_like(x, 2**-149)
tiny[0, :16] = 0
print("scale and NaN outputs", nb.quantize_activation(tiny).global_scale.item(), w(tiny).isnan().sum().item())

buf = torch.full((17, 48), 0xa5, dtype=torch.uint8, device=d)
scale = torch.full((128 * 8,), 0xff, dtype=torch.uint8, device=d)
workspace = torch.full((nb.MAX_PARTIALS,), float("nan"), device=d)
gs = torch.empty((), device=d)
n = nb.launch_amax(x, workspace)
nb.launch_quant(x, buf, scale, gs, workspace=workspace, n_part=n)
ref = nb.quantize_activation_ref(x)
assert torch.equal(scale, ref.scale.view(torch.uint8))
def gemm():
    act = nb.NVFP4Activation(buf.view(torch.float4_e2m1fn_x2), scale.view(torch.float8_e4m3fn), gs)
    return nb.nvfp4_scaled_mm_public(act, w.w_packed, w.w_scale, w.w_global)
clean = gemm()
assert torch.equal(clean, nb.nvfp4_scaled_mm_public(ref, w.w_packed, w.w_scale, w.w_global))
r, c = torch.arange(17, device=d)[:, None], torch.arange(6, 8, device=d)[None, :]
offset = (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
assert (scale[offset] == 0).all()
scale[offset] = 0x7f
print("NaNs after column poison", gemm().isnan().sum().item(), "of", clean.numel())
PY
```

To inspect the guard safely, the following example **only parses** the two paths. The audit exercised the writer separately on disposable data with a stubbed experiment.

```bash
"$py" - <<'PY'
from research.quant_ptq import bench_nvfp4_e2e as b
from pathlib import Path
for target in (b.CANONICAL_OUTPUT, Path("/home/ryan/Documents/projects/ai/results") / b.CANONICAL_OUTPUT.name):
    try:
        print("accepted", b.parse_args(["--parts", "A", "--output", str(target)]).output)
    except SystemExit as exc:
        print("rejected", target, exc.code)
PY
```
