"""Triton-accelerated Polar Switch and Gram Newton-Schulz for NVIDIA Blackwell (RTX 5080, sm_120).

Both circuits keep the arithmetic of the eager references in ``baselines.py``: FP16 (or BF16)
operands, FP32 accumulation, and one rounding per matrix operation. The work is split as:

* ``X @ X.mT``             Triton SYRK from ``kernels_blackwell`` (upper-triangle tiles only).
* ``a*(A @ B) + b*C + c*I`` one fused symmetric Triton kernel. It covers every Polar Switch buffer
                           update and every Gram-space product of Gram Newton-Schulz.
* ``Q @ X``                cuBLAS. The last product writes FP32 directly, without a cast pass.

``fp8_sketch=True`` is **experimental and not promoted**: it computes only the first Gram
matrix from FP8 (e4m3) operands and rescales it to the exact trace. Although trace rescaling
bounds ``sigma_max``, quantization noise breaks commutation between ``poly(A_sketch)`` and
``X @ X.mT``, increasing median Tier-1 polar error by +18-23% and causing large rotation
errors on rank-deficient inputs (40-161% on rank-1).

Inputs have shape ``(..., rows, columns)``. Tall, wide, square, and odd sizes (including ``min==1``)
are supported.

Run the benchmark:
    PYTHONPATH=. .venv/bin/python research/polar_opt/polar_switch_blackwell.py --bench
"""

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from torch import Tensor
from triton import knobs
from triton.runtime import driver

from research.polar_opt.baselines import POLAR_EXPRESS_COEFFICIENTS, _SWITCH_TABLES, _take_steps
from research.polar_opt.kernels_blackwell import (
    _CONFIGS,
    _fp8_quant,
    _store_sym,
    _syrk_kernel,
    _tri_tile,
)

_WORK_DTYPES = (torch.float16, torch.bfloat16)
_EPS = 1e-7  # the eager baselines divide by ||G||_F + 1e-7


# restore_value: the output can alias C, so autotune trials must not accumulate into it.
@triton.autotune(configs=_CONFIGS, key=["N"], restore_value=["o_ptr"])
@triton.jit
def _symprod_kernel(
    a_ptr, b_ptr, c_ptr, o_ptr, N, alpha, beta, gamma,
    BLOCK: tl.constexpr, BK: tl.constexpr,
):
    """out = alpha*(A @ B) + beta*C + gamma*I for contiguous (batch, N, N) matrices.

    A, B, and C must be symmetric, and A @ B must be symmetric in exact arithmetic (A and B
    commute). Only upper-triangle tiles are computed, then mirrored, so the output is bitwise
    symmetric. The output can alias C but not A or B.
    """
    offset = tl.program_id(1).to(tl.int64) * N * N
    i, j = _tri_tile(tl.program_id(0), tl.cdiv(N, BLOCK))
    rm = i * BLOCK + tl.arange(0, BLOCK)
    rn = j * BLOCK + tl.arange(0, BLOCK)
    rk = tl.arange(0, BK)
    l_ptrs = a_ptr + offset + rm[:, None].to(tl.int64) * N + rk[None, :]
    r_ptrs = b_ptr + offset + rk[:, None].to(tl.int64) * N + rn[None, :]
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BK)):
        kmask = rk < N - k * BK
        left = tl.load(l_ptrs, mask=(rm[:, None] < N) & kmask[None, :], other=0.0)
        right = tl.load(r_ptrs, mask=kmask[:, None] & (rn[None, :] < N), other=0.0)
        acc = tl.dot(left, right, acc)
        l_ptrs += BK
        r_ptrs += BK * N
    tile = (rm[:, None] < N) & (rn[None, :] < N)
    c = tl.load(c_ptr + offset + rm[:, None].to(tl.int64) * N + rn[None, :], mask=tile, other=0.0)
    out = alpha * acc + beta * c.to(tl.float32) + gamma * (rm[:, None] == rn[None, :]).to(tl.float32)
    _store_sym(o_ptr + offset, out, i, j, N, BLOCK)


# A one-config autotune gives this kernel the same launch interface as the tuned kernels.
@triton.autotune(configs=[triton.Config({"BLOCK": 4096}, num_warps=4)], key=[])
@triton.jit
def _normalize_kernel(x_ptr, norm_ptr, o_ptr, size, eps, BLOCK: tl.constexpr):
    """out = x / (norm[matrix] + eps), rounded once to the output dtype.

    ``size`` is the element count of one matrix.
    """
    matrix = tl.program_id(1).to(tl.int64)
    local = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = local < size
    x = tl.load(x_ptr + matrix * size + local, mask=mask, other=0.0).to(tl.float32)
    y = x / (tl.load(norm_ptr + matrix) + eps)
    tl.store(o_ptr + matrix * size + local, y.to(o_ptr.dtype.element_ty), mask=mask)


_PLANS: dict = {}


def _autotune_key(kernel, args) -> tuple:
    nargs = dict(zip(kernel.arg_names, args))
    key = [nargs[k] for k in kernel.keys if k in nargs]
    for arg in nargs.values():
        if hasattr(arg, "dtype"):
            key.append(str(arg.dtype))
    return tuple(key)


def _config_of(kernel, args) -> dict:
    """Keyword arguments of the config the autotuner uses for ``args`` (after one run)."""
    if len(kernel.configs) > 1:
        return kernel.cache.get(_autotune_key(kernel, args), kernel.best_config).kwargs
    return kernel.configs[0].kwargs


def _c_launcher(compiled):
    """``(launch, head)`` to call Triton's C launcher without its Python wrappers, or None.

    ``CompiledKernel.__getitem__`` and ``CudaLauncher.__call__`` cost about 3 us per launch. For
    a kernel without scratch memory, sanitizer, or tensor descriptors they only read the current
    stream and pass constants. The argument layout is the one of Triton 3.8. Another version,
    or a kernel that needs the wrappers, keeps the public path.
    """
    try:
        launcher = compiled.run
        plain = (triton.__version__.startswith("3.8.") and type(launcher).__name__ == "CudaLauncher"
                 and launcher.launch is driver.active.utils.launch and not launcher.gsan_enabled
                 and not launcher.global_scratch_size and not launcher.profile_scratch_size)
        if not plain:
            return None
        # launch(grid x, y, z, stream, *head, kernel arguments): no launch metadata, no hooks,
        # and no global or profile scratch.
        return launcher.launch, (compiled.function, launcher.launch_cooperative_grid, launcher.launch_pdl,
                                 compiled.packed_metadata, None, None, None, None, None,
                                 launcher.arg_annotations, launcher.kernel_signature)
    except AttributeError:
        return None


def _launch(kernel, grid, *args) -> dict:
    """Launch an autotuned kernel on ``grid(config)`` programs and return ``config``.

    ``config`` holds the keyword arguments of the autotuned configuration (for example the tile
    ``BLOCK``), so a caller can size what the kernel wrote.

    The first call for a specialization runs Triton's autotuner. Later calls reuse its
    CompiledKernel and, when possible, call Triton's C launcher directly (``_c_launcher``).
    This skips the Python argument binding that otherwise dominates small matrices. The key
    holds everything Triton specializes on: integer values, dtypes, and 16-byte pointer
    alignment. Float arguments never specialize. The C launcher still receives the tensors, so
    it checks every pointer. Launch hooks (profilers) use the public path.
    """
    if torch.compiler.is_compiling():
        kernel[grid](*args)
        return _config_of(kernel, args)
    key = (kernel, args[0].device, *[
        (a.dtype, a.data_ptr() % 16) if isinstance(a, Tensor) else None if isinstance(a, float) else a
        for a in args
    ])
    plan = _PLANS.get(key)
    if plan is None:
        if (
            len(kernel.configs) > 1
            and torch.cuda.is_current_stream_capturing()
            and _autotune_key(kernel, args) not in kernel.cache
        ):
            raise RuntimeError(
                "Warm up Triton kernels eagerly once per (circuit, shape, dtype, device) "
                "before CUDA-graph capture"
            )
        compiled = kernel[grid](*args)
        config = _config_of(kernel, args)
        if compiled is not None and hasattr(compiled, "__getitem__"):
            constants = tuple(config[n] for n in kernel.arg_names if n in config)
            _PLANS[key] = compiled, config, constants, _c_launcher(compiled)
        return config
    compiled, config, constants, direct = plan
    x, y, z = grid(config)
    hooks = knobs.runtime
    if direct is None or hooks.launch_enter_hook.calls or hooks.launch_exit_hook.calls:
        compiled[x, y, z](*args, *constants)
    else:
        launch, head = direct
        active = driver.active
        launch(x, y, z, active.get_current_stream(active.get_current_device()), *head, (*args, *constants))
    return config


def _triangle_grid(n: int, batch: int):
    def grid(config):
        tiles = -(-n // config["BLOCK"])
        return tiles * (tiles + 1) // 2, batch, 1

    return grid


def _prepare(G: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Return X = G / (||G||_F + 1e-7) in ``dtype`` as (batch, rows, columns), and ||G||_F.

    The norm and the division are FP32, as in the eager ``_normalized_fp16``.
    """
    if dtype not in _WORK_DTYPES:
        raise TypeError(f"Working dtype must be float16 or bfloat16, received {dtype}")
    if G.ndim < 2 or not G.is_cuda or not G.is_floating_point():
        raise ValueError("Expected a floating-point CUDA tensor of shape (..., rows, columns)")
    X = G.reshape(-1, *G.shape[-2:]).float().contiguous()
    norm = torch.linalg.vector_norm(X, dim=(-2, -1))
    out = torch.empty(X.shape, dtype=dtype, device=X.device)
    size = X.shape[-2] * X.shape[-1]
    _launch(_normalize_kernel, lambda c: (-(-size // c["BLOCK"]), len(X), 1),
            X, norm, out, size, _EPS)
    return out, norm


def _gram(X: Tensor, out: Tensor, sketch_norm: Tensor | None = None) -> None:
    """Write the Gram matrix of the short side of ``X`` (batch, rows, columns) into ``out``.

    With ``sketch_norm`` (the ||G||_F that normalized X), use an **experimental** FP8 sketch:
    quantize X to e4m3, accumulate the Gram matrix in FP32, and rescale it to the exact trace
    ||X||_F^2. Although trace rescaling keeps sigma_max bounded, ``poly(A_sketch)`` does not
    commute with the true ``X @ X.mT``, increasing median Tier-1 error by +18-23% and rotating
    left singular vectors on rank-deficient matrices (do not promote over exact FP16 Gram).
    """
    target = out
    if sketch_norm is not None:
        X = _fp8_quant(X)[0]
        out = torch.empty(target.shape, dtype=torch.float32, device=X.device)
    view = X.mT if X.shape[-2] > X.shape[-1] else X
    batch, n, k = view.shape
    _launch(_syrk_kernel, _triangle_grid(n, batch), view, out, n, k, *view.stride())
    if sketch_norm is not None:
        energy = torch.where(
            torch.isfinite(sketch_norm),
            (sketch_norm / (sketch_norm + _EPS)).square(),
            torch.zeros_like(sketch_norm),
        )
        trace = out.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min_(torch.finfo(torch.float32).tiny)
        torch.mul(out, (energy / trace)[:, None, None], out=target)


def _symprod(A: Tensor, B: Tensor, C: Tensor, out: Tensor, alpha: float, beta: float,
             gamma: float = 0.0) -> None:
    """out = alpha*(A @ B) + beta*C + gamma*I. See ``_symprod_kernel`` for the requirements."""
    if out is A or out is B:
        raise ValueError("The output can alias only the additive operand")
    batch, n, _ = A.shape
    _launch(_symprod_kernel, _triangle_grid(n, batch), A, B, C, out, n,
            float(alpha), float(beta), float(gamma))


def _apply(Q: Tensor, X: Tensor, beta: float, out_dtype: torch.dtype) -> Tensor:
    """Return beta*X + Q @ X for wide X, or beta*X + X @ Q for tall X (cuBLAS).

    cuBLAS can write FP32 from FP16/BF16 operands. Other output dtypes need a cast.
    """
    left, right = (X, Q) if X.shape[-2] > X.shape[-1] else (Q, X)
    direct = out_dtype if out_dtype in (X.dtype, torch.float32) else X.dtype
    if beta:
        out = torch.baddbmm(X, left, right, beta=beta, out_dtype=direct)
    else:
        out = torch.bmm(left, right, out_dtype=direct)
    return out.to(out_dtype)


@torch.compiler.disable
@torch.no_grad()
def polar_switch_blackwell(G: Tensor, variant: str = "PS2_55", *,
                           dtype: torch.dtype = torch.float16, fp8_sketch: bool = False) -> Tensor:
    """Du and Su's Polar Switch circuit with Triton SYRK and fused buffer updates.

    Matches ``baselines.polar_switch`` up to rounding when ``dtype`` is float16. Each buffer
    update is one kernel: FP16/BF16 operands, FP32 accumulation, one rounding. The result has
    the dtype and shape of ``G``.
    """
    variant = variant.upper().replace("-", "_")
    if variant not in _SWITCH_TABLES:
        raise ValueError(f"Unsupported Polar Switch variant {variant!r}")
    if G.numel() == 0:
        return G.clone()
    table, buffers = _SWITCH_TABLES[variant]
    X, norm = _prepare(G, dtype)
    n = min(X.shape[-2:])
    work = [X.new_empty(len(X), n, n) for _ in range(buffers)]
    for phase, (residual, iterations) in enumerate(table):
        _gram(X, work[0], norm if fp8_sketch and phase == 0 else None)
        for dst, src, add, a, b, c in iterations:
            _symprod(work[src], work[src], work[add], work[dst], a, b, c)
        last = phase == len(table) - 1
        X = _apply(work[dst], X, residual, G.dtype if last else dtype)
    return X.reshape(G.shape)


@torch.compiler.disable
@torch.no_grad()
def gram_ns_blackwell(G: Tensor, steps: int = 5, reset_iterations=(2,), *,
                      dtype: torch.dtype = torch.float16, fp8_sketch: bool = False) -> Tensor:
    """Dao's Gram Newton-Schulz with Triton SYRK and fused symmetric Gram-space products.

    Matches ``baselines.gram_ns`` up to rounding when ``dtype`` is float16: the same recurrence,
    the same restarts, and direct quintic steps for square inputs.

    Use bfloat16 only for well-conditioned inputs. The Gram-space recurrence needs the FP16
    mantissa: in BF16 the largest output singular value reaches 1.3 to 1.8 on geometric and
    power-law spectra, against 1.15 in FP16. Polar Switch stays bounded in BF16.
    """
    coefficients = _take_steps(POLAR_EXPRESS_COEFFICIENTS, steps)
    if G.numel() == 0:
        return G.clone()
    X, norm = _prepare(G, dtype)
    sketch = norm if fp8_sketch else None
    rows, columns = X.shape[-2:]
    n = min(rows, columns)
    R, Z = X.new_empty(len(X), n, n), X.new_empty(len(X), n, n)
    last = len(coefficients) - 1
    if rows == columns:
        for i, (a, b, c) in enumerate(coefficients):
            _gram(X, R, sketch if i == 0 else None)
            _symprod(R, R, R, Z, c, b)
            X = _apply(Z, X, a, G.dtype if i == last else dtype)
    elif coefficients:
        resets = set(reset_iterations) - {0}
        Q, T = torch.empty_like(R), torch.empty_like(R)
        _gram(X, R, sketch)
        for i, (a, b, c) in enumerate(coefficients):
            if i in resets:
                X = _apply(Q, X, 0.0, dtype)
                _gram(X, R)
            _symprod(R, R, R, Z, c, b)
            if i == 0 or i in resets:
                Q.copy_(Z).diagonal(dim1=-2, dim2=-1).add_(a)
            else:
                _symprod(Q, Z, Q, T, 1.0, a)
                Q, T = T, Q
            if i < last and i + 1 not in resets:
                _symprod(R, Z, R, T, 1.0, a)
                _symprod(Z, T, T, R, 1.0, a)
        X = _apply(Q, X, 0.0, G.dtype)
    return X.to(G.dtype).reshape(G.shape)


# --------------------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------------------
BENCH_SHAPES = [(512, 2048), (1024, 4096), (2048, 8192), (4096, 4096), (4, 1024, 4096)]


def _methods():
    """Return name -> (callable, theoretical_cost name, eager baseline name)."""
    from research.polar_opt.baselines import gram_ns, polar_express, polar_switch
    from research.polar_opt.kernels_blackwell import polar_express_triton

    bf16, fp16 = torch.bfloat16, torch.float16
    methods = {
        "polar_express eager": (polar_express, "polar_express", None),
        "polar_express triton": (lambda G: polar_express_triton(G, 5, dtype=bf16), "polar_express",
                                 "polar_express eager"),
        "gram_ns eager": (gram_ns, "gram_ns", None),
    }
    for variant in ("PS2_55", "PS3_55"):
        methods[f"{variant} eager"] = (lambda G, v=variant: polar_switch(G, v), variant, None)
    for label, options in {
        "blackwell fp16": dict(dtype=fp16),
        "blackwell bf16": dict(dtype=bf16),
        "blackwell fp8-sketch+fp16": dict(dtype=fp16, fp8_sketch=True),
        "blackwell fp8-sketch+bf16": dict(dtype=bf16, fp8_sketch=True),
    }.items():
        for variant in ("PS2_55", "PS3_55"):
            methods[f"{variant} {label}"] = (
                lambda G, v=variant, o=options: polar_switch_blackwell(G, v, **o), variant,
                f"{variant} eager")
        methods[f"gram_ns {label}"] = (lambda G, o=options: gram_ns_blackwell(G, **o), "gram_ns",
                                       "gram_ns eager")
    return methods


def _stats(samples):
    q1, median, q3 = np.quantile(samples, [0.25, 0.5, 0.75])
    return {"median_ms": float(median), "iqr_ms": float(q3 - q1), "min_ms": float(min(samples))}


def _time(calls: dict, G: Tensor, flush: Tensor, warmup: int, iters: int, seed: int) -> dict:
    """Time every method with a flushed L2 cache, in a new random order each round.

    ``gpu``: CUDA events around the call. The flush runs first on the stream and the host does
    not wait for it; note that small-shape event intervals still reflect host Python launch
    latency unless captured inside a CUDA graph.
    ``wall``: host clock between two synchronizes, after an L2 flush; includes launch overhead.
    """
    order = random.Random(seed)
    names = list(calls)
    for name in names:
        for _ in range(warmup):
            calls[name](G)
    torch.cuda.synchronize()
    events = {name: [] for name in names}
    wall = {name: [] for name in names}
    for _ in range(iters):
        order.shuffle(names)
        for name in names:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            flush.zero_()
            start.record()
            out = calls[name](G)
            end.record()
            events[name].append((start, end))
            del out
        for name in names:
            flush.zero_()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            out = calls[name](G)
            torch.cuda.synchronize()
            wall[name].append((time.perf_counter() - begin) * 1e3)
            del out
    torch.cuda.synchronize()
    return {name: {"gpu": _stats([s.elapsed_time(e) for s, e in events[name]]),
                   "wall": _stats(wall[name])} for name in names}


def _quality(Q: Tensor, P: Tensor, singular_values: bool = False) -> dict:
    """Frobenius error relative to the FP64 SVD polar factor, and distance from orthogonality."""
    Q = Q.double().reshape(-1, *Q.shape[-2:])
    P = P.reshape(Q.shape)
    if not torch.isfinite(Q).all():
        return {"finite": False}
    extra = {"sigma_max": torch.linalg.svdvals(Q).max().item()} if singular_values else {}
    error = (Q - P).norm(dim=(-2, -1)) / P.norm(dim=(-2, -1))
    W = Q if Q.shape[-2] <= Q.shape[-1] else Q.mT
    eye = torch.eye(W.shape[-2], dtype=torch.float64, device=W.device)
    orthogonality = (W @ W.mT - eye).norm(dim=(-2, -1)) / W.shape[-2] ** 0.5
    return {"finite": True, "rel_fro_err_mean": error.mean().item(),
            "rel_fro_err_max": error.max().item(), "orth_err_rms": orthogonality.mean().item(),
            **extra}


def _add_speedups(rows: dict) -> None:
    for row in rows.values():
        for clock in ("gpu", "wall"):
            def ratio(name, clock=clock):
                return rows[name][clock]["median_ms"] / row[clock]["median_ms"] if name else None

            row[clock]["speedup_vs_own_eager"] = ratio(row["eager_baseline"])
            row[clock]["speedup_vs_polar_express_eager"] = ratio("polar_express eager")
            row[clock]["speedup_vs_polar_express_triton"] = ratio("polar_express triton")


def bench(output: Path, iters: int, warmup: int, seed: int) -> None:
    from research.polar_opt.baselines import svd_polar_fp64, theoretical_cost
    from research.polar_opt.harness_tier1 import synthetic_cases

    methods = _methods()
    properties = torch.cuda.get_device_properties(0)
    flush = torch.empty(4 * properties.L2_cache_size, dtype=torch.uint8, device="cuda")
    results = {
        "env": {
            "gpu": properties.name, "capability": list(torch.cuda.get_device_capability()),
            "sm_count": properties.multi_processor_count, "l2_cache_bytes": properties.L2_cache_size,
            "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            # Host CPU load affects launch-bound host clocks and small-shape event intervals.
            "host_cpus": os.cpu_count(), "host_load_start": os.getloadavg()[0],
        },
        "settings": {
            "iters": iters, "warmup": warmup, "seed": seed, "input": "torch.randn, float32",
            "flush_bytes": flush.numel(), "order": "random per round, all methods interleaved",
            "gpu_clock": "CUDA events; L2 flush queued before the start event; launches pre-queued",
            "wall_clock": "host perf_counter between synchronizes, after an L2 flush; includes launch overhead",
            "tflops": "theoretical_cost FLOPs / gpu median; dense = full GEMMs, symmetric = one triangle",
            "error": "||Q - P||_F / ||P||_F against baselines.svd_polar_fp64",
        },
        "shapes": [], "spectrum_suite": [],
    }
    print(f"{properties.name} sm_{properties.major}{properties.minor} torch {torch.__version__} "
          f"triton {triton.__version__}")
    for shape in BENCH_SHAPES:
        torch.manual_seed(seed)
        G = torch.randn(*shape, device="cuda")
        began = time.time()
        P = svd_polar_fp64(G)
        print(f"\n{shape}: FP64 SVD reference in {time.time() - began:.1f} s", flush=True)
        rows = {}
        for name, (call, cost_name, baseline) in methods.items():
            cost = theoretical_cost(cost_name, tuple(shape))
            rows[name] = {"eager_baseline": baseline, "flops_dense": cost.flops,
                          "flops_symmetric": cost.symmetric_flops, **_quality(call(G), P)}
        timings = _time({name: m[0] for name, m in methods.items()}, G, flush, warmup, iters, seed)
        for name, row in rows.items():
            row.update(timings[name])
            seconds = row["gpu"]["median_ms"] * 1e-3
            row["tflops_dense"] = row["flops_dense"] / seconds / 1e12
            row["tflops_symmetric"] = row["flops_symmetric"] / seconds / 1e12
        _add_speedups(rows)
        results["shapes"].append({"shape": list(shape), "methods": rows})
        for name, row in rows.items():
            gpu, wall = row["gpu"], row["wall"]
            own = f"{gpu['speedup_vs_own_eager']:.2f}x" if gpu["speedup_vs_own_eager"] else "  -  "
            print(f"  {name:<34} gpu {gpu['median_ms']:8.4f} ms (IQR {gpu['iqr_ms']:.4f})  "
                  f"wall {wall['median_ms']:8.4f} ms (IQR {wall['iqr_ms']:.4f})  "
                  f"{row['tflops_dense']:6.1f} TFLOPS  err {row['rel_fro_err_mean']:.4f}  "
                  f"vs eager {own}  vs PE {gpu['speedup_vs_polar_express_eager']:.2f}x", flush=True)
        del G, P

    # Accuracy on non-Gaussian spectra: the place where FP8 or BF16 can fail.
    print("\nSpectrum suite (relative Frobenius error to the FP64 SVD polar factor)")
    for case in synthetic_cases([512], [1, 4], seed):
        G = case.matrix.cuda()
        P = svd_polar_fp64(G)
        errors = {name: _quality(call(G), P, singular_values=True)
                  for name, (call, _, _) in methods.items()}
        results["spectrum_suite"].append({"name": case.name, "shape": list(G.shape), "methods": errors})
        shown = ("PS2_55 eager", "PS2_55 blackwell fp16", "PS2_55 blackwell fp8-sketch+fp16",
                 "PS2_55 blackwell fp8-sketch+bf16", "gram_ns eager", "gram_ns blackwell fp16",
                 "gram_ns blackwell fp8-sketch+fp16")
        print(f"  {case.name:<34}" + " ".join(
            f"{errors[n]['rel_fro_err_mean']:.4f}" if errors[n]["finite"] else "nonfin" for n in shown))
    suite = [(name, m) for case in results["spectrum_suite"] for name, m in case["methods"].items()]
    results["spectrum_suite_nonfinite"] = sum(not m["finite"] for _, m in suite)
    results["spectrum_suite_sigma_max"] = {
        name: max(m["sigma_max"] for other, m in suite if other == name and m["finite"])
        for name in methods}
    print("  largest singular value per method: " + ", ".join(
        f"{name} {value:.3f}" for name, value in results["spectrum_suite_sigma_max"].items()))

    results["env"]["host_load_end"] = os.getloadavg()[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"\nWrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", action="store_true", help="run the RTX 5080 benchmark")
    parser.add_argument("--output", type=Path, default=Path("results/polar_switch_blackwell_cycle1.json"))
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.bench:
        parser.error("Nothing to do. Pass --bench.")
    bench(args.output, args.iters, args.warmup, args.seed)


if __name__ == "__main__":
    main()
