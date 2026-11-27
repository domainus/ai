"""Cycle 5 C5-01: native K > 65536 chunked SYRK and the exact sm_120 Tensor-Core MMA arithmetic.

One run writes one strict JSON file (``allow_nan=False``) with five sections:

* ``zero_regression``: at ``K <= SYRK_CHUNK_GATE`` the native C5-01 kernels against the
  *unmodified* pre-C5-01 modules, imported from the git object ``LEGACY_BASE``: output bits
  (int16/int32 views), profiled kernel launches and host synchronizations. Both sides use the
  native autotuner's configurations (``share_autotune``).
* ``timing``: native C5-01 against the un-gated control (``ungated``: the gate is lifted, so
  every SYRK is the single pre-C5-01 launch). Both variants share the compiled kernels, so the
  comparison isolates the gate. Per round: a 4 x L2 (268 MB) flush, both variants in the
  Williams order of the round (AB, BA, AB, ...), CUDA events (``gpu``) and the host clock
  between two synchronizes (``wall``), for CUDA-graph replay and for eager calls. A graph cell's
  ``outputs_bitwise_equal`` is taken after both graphs replay, against a fresh eager call. Every
  raw sample is stored; ``validate_results`` recomputes medians, ratios and paired 95% CIs and
  checks the identity of every recorded cell against the protocol.
* ``long_k``: invariants of the chunked path (bitwise symmetry, no host synchronization,
  CUDA-graph replay equal to eager, gate boundaries, tall/batched/degenerate inputs,
  ``sigma_hat`` and ``scaling="frobenius"``, bitwise equality with the audited C4-03
  ``ChunkedGram``), Opus L6 at ``K = 2^25`` (native against the unchunked control: sigma_max
  and speedup from raw samples), and the full C4-03 drift grid against each circuit's own
  design ceiling.
* ``mma``: ``blackwell_mma_model.characterize`` (witnesses, exponent limits, lane permutations,
  alternative hypotheses, randomized stress suite, all 11 ``_CONFIGS``).
* ``derived``: headline numbers, recomputed by ``validate_results``.

``cycle5_c5_01*.json`` is reserved in every checkout and through symlinks: only the canonical
protocol, on an sm_120+ device and a clean git tree, may write it. Both guards run before any
kernel launches.

Run (alone on the GPU, from a clean commit):
    PYTHONPATH=. .venv/bin/python research/polar_opt/bench_c5_01_syrk_and_mma.py
Smoke test:
    PYTHONPATH=. .venv/bin/python research/polar_opt/bench_c5_01_syrk_and_mma.py --quick --output /tmp/c5.json
"""

from __future__ import annotations

import argparse
import ast
import atexit
import copy
import fnmatch
import functools
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from statistics import median

import torch
from torch import Tensor

from research.polar_opt import bench_c4_03_wide_and_gram_drift as c3
from research.polar_opt import blackwell_mma_model as mm
from research.polar_opt import harness_tier2_real as h
from research.polar_opt import kernels_blackwell as kbw
from research.polar_opt import polar_switch_blackwell as psb
from research.polar_opt import schatten_switch_blackwell as ssb
from research.polar_opt.bench_c4_01a_120m_long import (
    _assert_close, _is_hex, _require, set_numerics, write_json_atomic)
from research.quant_ptq import micro_rotated_ptq as q

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SCHEMA_VERSION = "cycle5_c5_01.v2"
TASK = ("Cycle 5 C5-01: zero-regression native K > 65536 chunked SYRK in the production Blackwell kernels and "
        "the exact sm_120 Tensor-Core MMA rounding model on the RTX 5080")
CANONICAL_OUTPUT = RESULTS / "cycle5_c5_01_syrk_and_mma.json"
QUICK_OUTPUT = RESULTS / "c5_01_quick.json"
RESERVED_PATTERN = "cycle5_c5_01*.json"
LEGACY_BASE = "289e6d75fc341ce6476867147a000da3d4704969"  # the last commit before C5-01
LEGACY_MODULES = ("kernels_blackwell", "polar_switch_blackwell", "schatten_switch_blackwell")
REQUIRED_CODE = ("bench_c5_01_syrk_and_mma", "blackwell_mma_model", "kernels_blackwell", "polar_switch_blackwell",
                 "schatten_switch_blackwell", "bench_c4_03_wide_and_gram_drift", "baselines")
KERNELS = ("ps3_55", "ss3_54", "ss3_53")
LAUNCHES = {"ps3_55": 16, "ss3_54": 16, "ss3_53": 15}  # pre-C5-01 profiled launches at 512 x 2048
NORMAL_SHAPES = [[512, 2048], [1024, 4096], [2048, 8192], [4096, 4096], [4, 1024, 4096]]
OPUS_SHAPES = [[4, 1 << 25], [1 << 25, 2], [1, 1 << 25]]

CANONICAL = {
    "device": "cuda", "legacy_base": LEGACY_BASE, "gate": kbw.SYRK_CHUNK_GATE, "chunk": kbw.SYRK_CHUNK_SIZE,
    "max_bytes": kbw.SYRK_CHUNK_MAX_BYTES, "kernels": list(KERNELS), "dtypes": ["fp16", "bf16"],
    "zero_regression": {
        "shapes": [*NORMAL_SHAPES, [2048, 512], [3, 257, 1031], [1, 4096], [4096, 1], [3, 1, 300],
                   [8, 65535], [8, 65536], [65536, 3]],
        "seeds": [0, 1], "launch_shapes": NORMAL_SHAPES, "launch_expected": LAUNCHES},
    "timing": {"shapes": NORMAL_SHAPES, "kernels": list(KERNELS), "dtype": "fp16", "rounds": 40, "warmup": 3,
               "flush_l2_multiple": 4, "modes": ["graph", "eager"], "seed": 0},
    "long_k": {
        "boundaries": [65535, 65536, 65537, 69632, 69633, 100000], "boundary_rows": 4,
        "graph_shapes": [[4, 131072], [4, 1 << 20]], "symmetry_shapes": [[4, 131072], [3, 5, 100000], [100000, 6]],
        "shapes": [[100000, 4], [3, 4, 100000], [3, 100000, 4], [2, 3, 200000]],
        "opus_shapes": OPUS_SHAPES, "opus_reps": 4, "opus_seed": 0,  # even: AB, BA, AB, BA
        "opus_control_pin": c3.PIN_CONFIG,
        "grid_families": c3.CANONICAL["drift"]["families"], "ceiling": c3.DESIGN_CEILING},
    "mma": {"stress_seed": 2026, "stress_dots": 4096, "scalar_crosscheck": 512, "chain_dots": 512,
            "chains": [[32, 16], [64, 32], [256, 64], [1024, 128], [4096, 64], [16384, 128], [16384, 16]],
            "syrk_shapes": [[9, 2033], [33, 300], [5, 16384], [64, 129]], "hypothesis_dots": 2048},
}
QUICK = {
    **CANONICAL,
    "zero_regression": {"shapes": [[256, 1024], [2, 64, 256], [300, 70], [1, 512], [4, 65536]], "seeds": [0],
                        "launch_shapes": [[512, 2048]], "launch_expected": LAUNCHES},
    "timing": {**CANONICAL["timing"], "shapes": [[256, 1024]], "rounds": 4, "warmup": 1},
    "long_k": {**CANONICAL["long_k"], "boundaries": [65536, 65537, 69633], "graph_shapes": [[4, 131072]],
               "symmetry_shapes": [[4, 131072], [70000, 3]], "shapes": [[70000, 4], [2, 3, 70000]],
               "opus_shapes": [[4, 1 << 18]], "opus_reps": 1,
               "grid_families": {"wide4": {"n": 4, "orientation": "wide", "k_exponents": [12, 17], "seeds": [0]}}},
    "mma": {"stress_seed": 7, "stress_dots": 256, "scalar_crosscheck": 32, "chain_dots": 32,
            "chains": [[64, 32], [1024, 128]], "syrk_shapes": [[9, 300]], "hypothesis_dots": 64},
}


# ======================================================================================
# Guards and provenance
# ======================================================================================
def reserved(path: Path) -> bool:
    """True for a ``cycle5_c5_01*.json`` name, in any checkout or through a symlink."""
    path = Path(path)
    return any(fnmatch.fnmatchcase(name, RESERVED_PATTERN) for name in (path.name, path.resolve().name))


def require_canonical_host(device: torch.device, git: dict, output: Path, canonical: bool) -> None:
    """A canonical run (or a reserved filename) needs an sm_120+ CUDA device and a clean, committed tree."""
    if not (canonical or reserved(output)):
        return
    if device.type != "cuda" or not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(device)) < (12, 0):
        raise RuntimeError(f"{Path(output).name} requires a Blackwell (sm_120+) CUDA device")
    if git.get("git_dirty") is not False or not _is_hex(git.get("git_sha"), 40):
        raise RuntimeError("A canonical run needs a clean, committed git tree (git_dirty must be False)")


def _module_path(name: str) -> Path:
    return ROOT.joinpath(*name.split(".")).with_suffix(".py")


@functools.cache
def code_modules() -> tuple[str, ...]:
    """Every ``research.*`` module this driver imports, directly or transitively (static ``ast`` closure)."""
    found, todo = set(), [__name__ if __name__.startswith("research.") else "research.polar_opt.bench_c5_01_syrk_and_mma"]
    while todo:
        name = todo.pop()
        if name in found or not _module_path(name).is_file():
            continue
        found.add(name)
        for node in ast.walk(ast.parse(_module_path(name).read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("research"):
                todo += [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
            elif isinstance(node, ast.Import):
                todo += [alias.name for alias in node.names if alias.name.startswith("research")]
    return tuple(sorted(found))


def code_sha256() -> str:
    digest = hashlib.sha256()
    for name in code_modules():
        digest.update(name.encode() + b"\0" + _module_path(name).read_bytes())
    return digest.hexdigest()


ENV_KEYS = ("torch", "triton", "cuda", "driver", "device", "capability")


def run_config_hash(quick: bool, protocol: dict, code: str, git_sha: str | None, env: dict) -> str:
    payload = {"schema": SCHEMA_VERSION, "quick": quick, "protocol": protocol, "code_sha256": code,
               "git_sha": git_sha, "env": {k: env.get(k) for k in ENV_KEYS}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ======================================================================================
# Variants: legacy modules (pre-C5-01) and the un-gated control
# ======================================================================================
def _private_tmp() -> Path:
    base = Path.home() / "tmp"
    base.mkdir(mode=0o700, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="c5_01_legacy.", dir=base))


@functools.cache
def load_legacy(base: str = LEGACY_BASE) -> dict:
    """Import the three Blackwell modules exactly as committed at ``base`` (``git show``), as a separate package.

    Only the ``research.polar_opt.<module>`` imports among the three are renamed; every other byte
    is the committed source (``blob`` records the git object id). The directory is removed at exit.
    """
    root = _private_tmp()
    atexit.register(shutil.rmtree, root, True)
    package = f"c5_legacy_{base[:12]}"
    (root / package).mkdir()
    (root / package / "__init__.py").write_text("")
    blobs = {}
    for name in LEGACY_MODULES:
        path = f"research/polar_opt/{name}.py"
        source = subprocess.run(["git", "-C", str(ROOT), "show", f"{base}:{path}"], capture_output=True, text=True,
                                check=True).stdout
        blobs[name] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{base}:{path}"], capture_output=True,
                                     text=True, check=True).stdout.strip()
        for other in LEGACY_MODULES:
            source = source.replace(f"research.polar_opt.{other}", f"{package}.{other}")
        (root / package / f"{name}.py").write_text(source)
    sys.path.insert(0, str(root))
    try:
        modules = {name: importlib.import_module(f"{package}.{name}") for name in LEGACY_MODULES}
    finally:
        sys.path.remove(str(root))
    return {"modules": modules, "blobs": blobs, "base": base}


def share_autotune(legacy: dict) -> None:
    """Give the legacy kernels the configurations the native autotuners chose (same keys).

    The Schatten circuits sum one norm partial per tile, so their bits depend on the tile size the
    autotuner picks (a pre-C5-01 property). A comparison is bitwise only for equal choices; the
    native call of every key runs first, so the legacy autotuners never benchmark on their own.
    """
    from triton.runtime.autotuner import Autotuner

    for name in LEGACY_MODULES:
        native, old = importlib.import_module(f"research.polar_opt.{name}"), legacy["modules"][name]
        for attr, kernel in vars(native).items():
            twin = getattr(old, attr, None)
            if isinstance(kernel, Autotuner) and isinstance(twin, Autotuner):
                twin.cache.update(kernel.cache)


@contextmanager
def ungated():
    """The control: lift the gate in all three modules, so every SYRK is the single pre-C5-01 launch."""
    saved = (kbw.SYRK_CHUNK_GATE, psb.SYRK_CHUNK_GATE, ssb.SYRK_CHUNK_GATE)
    kbw.SYRK_CHUNK_GATE = psb.SYRK_CHUNK_GATE = ssb.SYRK_CHUNK_GATE = 1 << 60
    try:
        yield
    finally:
        kbw.SYRK_CHUNK_GATE, psb.SYRK_CHUNK_GATE, ssb.SYRK_CHUNK_GATE = saved


def circuit(kernel: str, dtype: str, modules: dict | None = None):
    """``f(G) -> Q`` of a production orthogonalizer, from the native modules or from ``modules`` (legacy)."""
    work = mm.DTYPES[dtype]
    ps = (modules or {}).get("polar_switch_blackwell", psb)
    ss = (modules or {}).get("schatten_switch_blackwell", ssb)
    if kernel == "ps3_55":
        return lambda G: ps.polar_switch_blackwell(G, "PS3_55", dtype=work)
    if kernel == "gram_ns":
        return lambda G: ps.gram_ns_blackwell(G, dtype=work)
    if kernel == "triton_syrk":
        kb = (modules or {}).get("kernels_blackwell", kbw)
        # the Gram matrix of the short side, as the circuits form it
        return lambda G: kb.triton_syrk((G / G.norm()).to(work).transpose(-2, -1).contiguous()
                                        if G.shape[-2] > G.shape[-1] else (G / G.norm()).to(work))
    return lambda G: ss.schatten_switch_blackwell(G, kernel.upper(), dtype=work, allow_unsafe_bf16=True)


def _bits(x: Tensor) -> Tensor:
    view = {torch.float16: torch.int16, torch.bfloat16: torch.int16, torch.float32: torch.int32,
            torch.float64: torch.int64}[x.dtype]
    return x.contiguous().view(view)


def bit_equal(a: Tensor, b: Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(_bits(a), _bits(b)))


def _input(shape, seed: int, device: torch.device) -> Tensor:
    return torch.randn(*shape, device=device, generator=torch.Generator(device).manual_seed(seed))


def profiled_launches(call, G: Tensor) -> int:
    call(G)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        call(G)
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
               and not e.name.startswith(("Memcpy", "Memset")))


def sync_free(call, G: Tensor) -> bool:
    """True when the warm call runs under ``torch.cuda.set_sync_debug_mode("error")``."""
    call(G)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        call(G)
        return True
    except RuntimeError:
        return False
    finally:
        torch.cuda.set_sync_debug_mode(previous)
        torch.cuda.synchronize()


def graph_check(call, shape, seed: int, device: torch.device) -> dict:
    """Capture after an eager warm-up, replay on new data, compare with an eager call on that data."""
    static = _input(shape, seed, device)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            call(static)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call(static)
    static.copy_(_input(shape, seed + 1, device) * 3.0)
    graph.replay()
    torch.cuda.synchronize()
    eager = call(static)
    diff = (out.double() - eager.double()).abs().max().item()
    return {"bitwise_equal": bit_equal(out, eager), "max_abs_diff": diff, "finite": bool(torch.isfinite(out).all())}


# ======================================================================================
# Section 1: zero regression at K <= gate (native vs the committed pre-C5-01 modules)
# ======================================================================================
def zero_regression(protocol: dict, device: torch.device, log) -> dict:
    legacy = load_legacy(protocol["legacy_base"])
    spec = protocol["zero_regression"]
    rows = []
    for shape in spec["shapes"]:
        for seed in spec["seeds"]:
            G = _input(shape, seed, device)
            for kernel in (*protocol["kernels"], "gram_ns", "triton_syrk"):
                for dtype in protocol["dtypes"]:
                    native = circuit(kernel, dtype)(G)
                    share_autotune(legacy)
                    old = circuit(kernel, dtype, legacy["modules"])(G)
                    rows.append({"shape": shape, "seed": seed, "kernel": kernel, "dtype": dtype,
                                 "k": max(shape[-2:]), "bitwise_equal": bit_equal(native, old),
                                 "finite": bool(torch.isfinite(native).all())})
            log(f"[zero-regression] {shape} seed {seed}: "
                f"{sum(r['bitwise_equal'] for r in rows if r['shape'] == shape and r['seed'] == seed)} bitwise")
    launches = []
    for shape in spec["launch_shapes"]:
        G = _input(shape, 0, device)
        for kernel in protocol["kernels"]:
            native, old = circuit(kernel, "fp16"), circuit(kernel, "fp16", legacy["modules"])
            native_launches = profiled_launches(native, G)
            share_autotune(legacy)
            launches.append({"shape": shape, "kernel": kernel, "native": native_launches,
                             "legacy": profiled_launches(old, G), "native_sync_free": sync_free(native, G),
                             "legacy_sync_free": sync_free(old, G)})
    return {"legacy_base": legacy["base"], "legacy_blobs": legacy["blobs"], "rows": rows, "launches": launches}


# ======================================================================================
# Section 2: timing (native vs the un-gated control), every raw sample
# ======================================================================================
VARIANTS = ("native", "control")
WILLIAMS = (("native", "control"), ("control", "native"))  # the Williams square for two variants


def williams_order(round_index: int) -> list[str]:
    return list(WILLIAMS[round_index % 2])


def _context(variant: str):
    return ungated() if variant == "control" else nullcontext()


def _sample(run, flush: Tensor, variant: str) -> dict:
    """One flushed measurement. The variant's context is entered and left outside the timed interval."""
    with _context(variant):
        flush.zero_()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - begin) * 1e3
    return {"gpu_ms": start.elapsed_time(end), "wall_ms": wall}


def _graph(call, G: Tensor, variant: str):
    with _context(variant):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                call(G)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call(G)
    torch.cuda.synchronize()
    return graph, out


def graphs_bitwise_equal(graphs: dict, eager: Tensor) -> bool:
    """Replay every captured graph, synchronize, then compare their outputs with each other and ``eager``.

    ``graphs`` maps a variant to ``(CUDAGraph, output)``. Capture records the work without running
    it, so the output buffers hold nothing meaningful until the first replay.
    """
    for graph, _ in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    outputs = [out for _, out in graphs.values()]
    return all(bit_equal(out, eager) for out in outputs)


def timing(protocol: dict, device: torch.device, log) -> dict:
    spec = protocol["timing"]
    properties = torch.cuda.get_device_properties(device)
    flush = torch.empty(spec["flush_l2_multiple"] * properties.L2_cache_size, dtype=torch.uint8, device=device)
    cells = []
    for shape in spec["shapes"]:
        G = _input(shape, spec["seed"], device)
        for kernel in spec["kernels"]:
            call = circuit(kernel, spec["dtype"])
            for mode in spec["modes"]:
                if mode == "graph":
                    graphs = {v: _graph(call, G, v) for v in VARIANTS}
                    runs = {v: graphs[v][0].replay for v in VARIANTS}
                    equal = graphs_bitwise_equal(graphs, call(G))  # both replays equal a fresh eager call
                else:
                    runs = {v: functools.partial(call, G) for v in VARIANTS}
                    with ungated():
                        control = call(G)
                    equal = bit_equal(call(G), control)
                for _ in range(spec["warmup"]):
                    for v in VARIANTS:
                        _sample(runs[v], flush, v)
                rounds = []
                for r in range(spec["rounds"]):
                    order = williams_order(r)
                    rounds.append({"round": r, "order": order, **{v: _sample(runs[v], flush, v) for v in order}})
                cell = {"shape": shape, "kernel": kernel, "dtype": spec["dtype"], "mode": mode,
                        "outputs_bitwise_equal": equal, "rounds": rounds}
                cell["summary"] = timing_summary(rounds)
                cells.append(cell)
                log(f"[timing] {shape} {kernel} {mode}: gpu {cell['summary']['gpu']['median_ratio']:.4f} "
                    f"wall {cell['summary']['wall']['median_ratio']:.4f} (native / control)")
                if mode == "graph":
                    del graphs
    return {"flush_bytes": flush.numel(), "l2_cache_bytes": properties.L2_cache_size, "cells": cells}


def paired_ci(ratios: list[float]) -> dict:
    """Geometric-mean ratio with a paired 95% t interval on ``log`` ratios (df = n - 1)."""
    from scipy import stats

    logs = [math.log(r) for r in ratios]
    n = len(logs)
    mean = sum(logs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in logs) / (n - 1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "geomean_ratio": math.exp(mean), "ci95_low": math.exp(mean - half),
            "ci95_high": math.exp(mean + half)}


def timing_summary(rounds: list[dict]) -> dict:
    out = {}
    for clock in ("gpu", "wall"):
        key = f"{clock}_ms"
        native = [r["native"][key] for r in rounds]
        control = [r["control"][key] for r in rounds]
        out[clock] = {"native_median_ms": median(native), "control_median_ms": median(control),
                      "median_ratio": median(native) / median(control),
                      "paired": paired_ci([a / b for a, b in zip(native, control)])}
    return out


# ======================================================================================
# Section 3: long K
# ======================================================================================
def _chunked_reference(call, G: Tensor) -> Tensor:
    """The audited C4-03 prototype (``ChunkedGram.patched``) on the same input."""
    mitigation = c3.ChunkedGram(kbw.SYRK_CHUNK_SIZE, kbw.SYRK_CHUNK_GATE)
    with mitigation.patched():
        return call(G)


def _quality(Q: Tensor, G: Tensor) -> dict:
    A = c3.gram64(c3._short(G))
    values, vectors = torch.linalg.eigh(A)
    return c3.polar_quality(Q, G, (vectors / values.clamp_min(1e-300).sqrt()) @ vectors.mT)


def _syrk_checks(protocol: dict, device: torch.device) -> list[dict]:
    rows = []
    for shape in protocol["long_k"]["symmetry_shapes"]:
        G = _input(shape, 3, device)
        for dtype in protocol["dtypes"]:
            X = (G / G.norm()).to(mm.DTYPES[dtype])
            if X.shape[-2] > X.shape[-1]:
                view = X.mT if X.ndim == 3 else X.mT[None]
            else:
                view = X if X.ndim == 3 else X[None]
            batch, n, k = view.shape
            exact = torch.stack([c3.gram64(view[b]) for b in range(batch)])
            for out_dtype in (mm.DTYPES[dtype], torch.float32):
                out = torch.empty(batch, n, n, dtype=out_dtype, device=device)
                kbw._syrk_chunked(view, out)
                api = kbw.triton_syrk(X if X.shape[-2] <= X.shape[-1] else X.mT.contiguous(), out_dtype)
                diag = 1.0 - out.double().diagonal(dim1=-2, dim2=-1) / exact.diagonal(dim1=-2, dim2=-1)
                rows.append({"shape": shape, "dtype": dtype, "out_dtype": str(out_dtype).replace("torch.", ""),
                             "view_strides": list(view.stride()), "k": k,
                             "symmetric": bit_equal(out, out.mT.contiguous()),
                             "api_equals_helper": bit_equal(api.reshape(out.shape), out),
                             "sync_free": sync_free(lambda _: kbw._syrk_chunked(view, out), None),
                             "diag_bias_max": diag.abs().max().item()})
    return rows


def _boundaries(protocol: dict, device: torch.device, legacy: dict) -> list[dict]:
    rows = []
    gate = protocol["gate"]
    for k in protocol["long_k"]["boundaries"]:
        for orientation in ("wide", "tall"):
            shape = [protocol["long_k"]["boundary_rows"], k] if orientation == "wide" else [k, 3]
            G = _input(shape, k % 97, device)
            for kernel in protocol["kernels"]:
                for dtype in protocol["dtypes"]:
                    call = circuit(kernel, dtype)
                    native = call(G)
                    if k <= gate:
                        share_autotune(legacy)
                        reference = circuit(kernel, dtype, legacy["modules"])(G)
                    else:
                        reference = _chunked_reference(call, G)
                    with ungated():
                        control = call(G)
                    rows.append({"k": k, "shape": shape, "kernel": kernel, "dtype": dtype, "chunked": k > gate,
                                 "reference": "legacy" if k <= gate else "c4_03_chunked_gram",
                                 "bitwise_equal_reference": bit_equal(native, reference),
                                 "bitwise_equal_control": bit_equal(native, control),
                                 "sigma_max": _quality(native, G)["sigma_max"]})
    return rows


def _input_case_specs(protocol: dict) -> list[tuple]:
    """(name, shape, special, options) of every ``_inputs_cases`` input."""
    cases = [(f"shape {s}", s, None, {}) for s in protocol["long_k"]["shapes"]]
    k = protocol["long_k"]["shapes"][0][0]
    return cases + [("mixed batch: normal, zero, tiny, NaN, inf", [5, 4, k], "mixed", {}),
                    ("sigma_hat", [2, 4, k], None, {"sigma_hat": 0.9}),
                    ("scaling frobenius", [2, 4, k], None, {"scaling": "frobenius"})]


def _input_case_cells(protocol: dict) -> list[tuple]:
    """(case, shape, kernel, dtype) of every ``_inputs_cases`` row, in order."""
    return [(name, shape, kernel, dtype) for name, shape, _, options in _input_case_specs(protocol)
            for kernel in protocol["kernels"] if not (options and kernel == "ps3_55") for dtype in protocol["dtypes"]]


def _inputs_cases(protocol: dict, device: torch.device) -> list[dict]:
    """Tall, batched, degenerate (zero, tiny, non-finite) and explicit-scale inputs above the gate."""
    rows = []
    for name, shape, special, options in _input_case_specs(protocol):
        G = _input(shape, 11, device)
        if special == "mixed":
            G[1] = 0.0
            G[2] *= 1e-17  # ||G||_F about 6e-15, below FRO_FLOOR
            G[3, 0, 5] = math.nan
            G[4, 1, 7] = math.inf
        for kernel in protocol["kernels"]:
            if options and kernel == "ps3_55":
                continue
            for dtype in protocol["dtypes"]:
                if options:
                    call = (lambda x, kn=kernel, d=dtype: ssb.schatten_switch_blackwell(
                        x, kn.upper(), dtype=mm.DTYPES[d], allow_unsafe_bf16=True, **options))
                else:
                    call = circuit(kernel, dtype)
                native = call(G)
                reference = _chunked_reference(call, G)
                finite = torch.isfinite(native).flatten(-2).all(-1).reshape(-1).tolist()
                record = {"case": name, "shape": shape, "kernel": kernel, "dtype": dtype,
                          "bitwise_equal_c4_03": bit_equal(native, reference), "finite_per_matrix": finite,
                          "sync_free": sync_free(call, G) if "sigma_hat" not in options else None}
                if special == "mixed":
                    flat = native.reshape(-1, *native.shape[-2:])
                    record["degenerate_zero"] = [bool((flat[i] == 0).all()) for i in range(1, 5)] \
                        if kernel != "ps3_55" else None
                    record["normal_quality"] = _quality(native[0], G[0])
                rows.append(record)
    return rows


def _graphs(protocol: dict, device: torch.device) -> list[dict]:
    rows = []
    for shape in protocol["long_k"]["graph_shapes"]:
        for kernel in protocol["kernels"]:
            for dtype in protocol["dtypes"]:
                result = graph_check(circuit(kernel, dtype), shape, 21, device)
                rows.append({"shape": shape, "kernel": kernel, "dtype": dtype, **result})
        torch.cuda.empty_cache()
    return rows


def _timed_pair(call, G: Tensor, reps: int, pin: dict) -> tuple[dict, list[dict]]:
    """Outputs and interleaved synchronized samples (Williams order) of native and the pinned control."""
    @contextmanager
    def control_context():
        with ungated(), c3.pinned_syrk(pin):
            yield

    contexts = {"native": nullcontext, "control": control_context}
    outputs = {}
    for variant in VARIANTS:  # warm-up (compilation, autotuning) outside the samples
        with contexts[variant]():
            outputs[variant] = call(G)
        torch.cuda.synchronize()
    samples = []
    for r in range(reps):
        row = {"round": r, "order": williams_order(r)}
        for variant in row["order"]:
            with contexts[variant]():
                torch.cuda.synchronize()
                begin = time.perf_counter()
                out = call(G)
                torch.cuda.synchronize()
                row[variant] = {"wall_ms": (time.perf_counter() - begin) * 1e3}
                del out
        samples.append(row)
    return outputs, samples


def opus_l6(protocol: dict, device: torch.device, log) -> list[dict]:
    spec = protocol["long_k"]
    rows = []
    design = c3.design_ceilings(list(protocol["kernels"]))
    for shape in spec["opus_shapes"]:
        G = _input(shape, spec["opus_seed"], device)
        for kernel in protocol["kernels"]:
            for dtype in protocol["dtypes"]:
                call = circuit(kernel, dtype)
                outputs, samples = _timed_pair(call, G, spec["opus_reps"], spec["opus_control_pin"])
                native, control = (_quality(outputs[v], G) for v in VARIANTS)
                reference = _chunked_reference(call, G)
                row = {"shape": shape, "kernel": kernel, "dtype": dtype, "design_ceiling": design[kernel],
                       "native": native, "control": control, "samples": samples,
                       "bitwise_equal_c4_03": bit_equal(outputs["native"], reference)}
                row["speedup"] = speedup(samples)
                row["speedup_paired"] = paired_speedup(samples)
                rows.append(row)
                log(f"[opus-l6] {shape} {kernel} {dtype}: sigma_max {control['sigma_max']:.4f} -> "
                    f"{native['sigma_max']:.4f}, speedup {row['speedup']:.1f}x")
                del outputs, reference
        del G
        torch.cuda.empty_cache()
    return rows


def speedup(samples: list[dict]) -> float:
    return median(s["control"]["wall_ms"] for s in samples) / median(s["native"]["wall_ms"] for s in samples)


def paired_speedup(samples: list[dict]) -> dict:
    """``paired_ci`` of the per-round control / native ratios: every sample changes it."""
    return paired_ci([s["control"]["wall_ms"] / s["native"]["wall_ms"] for s in samples])


def ceiling_grid(protocol: dict, device: torch.device, log) -> list[dict]:
    """The full C4-03 drift grid with the native kernels: sigma_max against the own design ceiling."""
    spec = protocol["long_k"]
    design = c3.design_ceilings(list(protocol["kernels"]))
    rows = []
    for family, fam in spec["grid_families"].items():
        for exponent in fam["k_exponents"]:
            for seed in fam["seeds"]:
                n, k = fam["n"], 1 << exponent
                shape = [n, k] if fam["orientation"] == "wide" else [k, n]
                G = _input(shape, seed, device)
                A = c3.gram64(c3._short(G))
                values, vectors = torch.linalg.eigh(A)
                inverse_root = (vectors / values.sqrt()) @ vectors.mT
                cells = {}
                for kernel in protocol["kernels"]:
                    for dtype in protocol["dtypes"]:
                        call = circuit(kernel, dtype)
                        Q = call(G)
                        quality = c3.polar_quality(Q, G, inverse_root)
                        cells[f"{kernel}_{dtype}"] = {**quality, "design_ceiling": design[kernel],
                                                      "bitwise_equal_c4_03": bit_equal(Q, _chunked_reference(call, G))}
                        del Q
                rows.append({"family": family, "n": n, "k": k, "k_exponent": exponent, "seed": seed, "shape": shape,
                             "chunked": k > protocol["gate"], "cells": cells})
                worst = max((c["sigma_max"] or math.inf) - c["design_ceiling"] for c in cells.values())
                log(f"[grid] {family} 2^{exponent} seed {seed}: worst sigma_max - own ceiling {worst:+.5f}")
                del G, A
                torch.cuda.empty_cache()
    return rows


def long_k(protocol: dict, device: torch.device, log) -> dict:
    legacy = load_legacy(protocol["legacy_base"])
    long_shape = protocol["long_k"]["graph_shapes"][0]
    G = _input(long_shape, 5, device)
    launches = {kernel: profiled_launches(circuit(kernel, "fp16"), G) for kernel in protocol["kernels"]}
    sync = {kernel: sync_free(circuit(kernel, "fp16"), G) for kernel in protocol["kernels"]}
    out = {"launches_at_graph_shape": {"shape": long_shape, "native": launches, "sync_free": sync}}
    out["syrk"] = _syrk_checks(protocol, device)
    log("[long-k] syrk checks done")
    out["boundaries"] = _boundaries(protocol, device, legacy)
    log("[long-k] gate boundaries done")
    out["inputs"] = _inputs_cases(protocol, device)
    log("[long-k] input cases done")
    out["graphs"] = _graphs(protocol, device)
    log("[long-k] CUDA graphs done")
    out["opus_l6"] = opus_l6(protocol, device, log)
    out["grid"] = ceiling_grid(protocol, device, log)
    return out


# ======================================================================================
# Derived numbers (recomputed by validate_results)
# ======================================================================================
def _mma_totals(mma: dict) -> dict:
    def total(records):
        return [sum(r["bit_matches"] for r in records), sum(r["cases"] for r in records)]

    stress = mma["stress"]
    return {
        "witnesses": [sum(w["model_equals_hardware"] and w["configs_equal_hardware"] for w in mma["witnesses"]),
                      len(mma["witnesses"])],
        "witness_permutations": [sum(w["permutations_equal_hardware"] for w in mma["witnesses"]),
                                 sum(w["permutations"] for w in mma["witnesses"])],
        "exponent_table": [sum(r["model_equals_hardware"] for r in mma["exponent_table"]), len(mma["exponent_table"])],
        "edge_cases": [sum(r["model_equals_hardware"] and r["vectorized_equals_hardware"] for r in mma["edge_cases"]),
                       len(mma["edge_cases"])],
        "stress_single": total([r for r in stress if r["kind"] == "single"]),
        "stress_chains": total([r for r in stress if r["kind"] == "chain"]),
        "scalar_crosscheck": [sum(r["scalar_crosscheck"][0] for r in stress if r["scalar_crosscheck"]),
                              sum(r["scalar_crosscheck"][1] for r in stress if r["scalar_crosscheck"])],
        "syrk_configs": total(mma["syrk_configs"]),
        "syrk_fused_store": [sum(r["fused_store_matches"] for r in mma["syrk_configs"]),
                             sum(r["cases"] for r in mma["syrk_configs"])],
        "audit_seeds": total(mma["audit_seeds"]),
        "audit_seed719_fused_store": [sum(r["fused_store_matches"] for r in mma["audit_seeds"] if "fused_store_matches" in r),
                                      sum(r["cases"] for r in mma["audit_seeds"] if "fused_store_matches" in r)],
        "all_dot_products": total([*stress, *mma["syrk_configs"], *mma["audit_seeds"]]),
        "mismatch_records": sum(len(r["mismatches"]) for r in [*stress, *mma["syrk_configs"], *mma["audit_seeds"]]),
        "hypotheses": {f"{r['hypothesis']}/{r['dtype']}": [r["bit_matches"], r["cases"], r["witness_equals_hardware"]]
                       for r in mma["hypotheses"]},
        "ptx": sorted({op for ops in mma["ptx_mma"].values() for op in ops}
                      | {op for r in mma["syrk_configs"] for op in r["ptx_mma"]}),
    }


def derive(payload: dict) -> dict:
    zr, tm, lk = payload["zero_regression"], payload["timing"], payload["long_k"]
    gpu = {f"{c['mode']}:{c['kernel']}:{'x'.join(map(str, c['shape']))}": c["summary"] for c in tm["cells"]}

    def worst(mode, clock):
        values = [s[clock]["median_ratio"] for key, s in gpu.items() if key.startswith(mode + ":")]
        return {"min": min(values), "max": max(values)} if values else None

    grid = [(r, name, cell) for r in lk["grid"] for name, cell in r["cells"].items()]
    excess = [math.inf if c["sigma_max"] is None else c["sigma_max"] - c["design_ceiling"] for _, _, c in grid]
    opus = lk["opus_l6"]
    return {
        "zero_regression": {
            "cells": len(zr["rows"]), "bitwise_equal": sum(r["bitwise_equal"] for r in zr["rows"]),
            "launches_equal": sum(r["native"] == r["legacy"] for r in zr["launches"]),
            "launch_cells": len(zr["launches"]),
            "launches_at_512x2048": {r["kernel"]: r["native"] for r in zr["launches"] if r["shape"] == [512, 2048]},
            "sync_free": sum(r["native_sync_free"] for r in zr["launches"])},
        "timing": {"cells": len(tm["cells"]), "outputs_bitwise_equal": sum(c["outputs_bitwise_equal"] for c in tm["cells"]),
                   "graph_gpu_ratio": worst("graph", "gpu"), "graph_wall_ratio": worst("graph", "wall"),
                   "eager_gpu_ratio": worst("eager", "gpu"), "eager_wall_ratio": worst("eager", "wall")},
        "long_k": {
            "syrk_symmetric": [sum(r["symmetric"] for r in lk["syrk"]), len(lk["syrk"])],
            "syrk_sync_free": [sum(r["sync_free"] for r in lk["syrk"]), len(lk["syrk"])],
            "syrk_diag_bias_max": max(r["diag_bias_max"] for r in lk["syrk"]),
            "boundaries_bitwise_reference": [sum(r["bitwise_equal_reference"] for r in lk["boundaries"]),
                                             len(lk["boundaries"])],
            "boundaries_unchunked_equal_control": [
                sum(r["bitwise_equal_control"] for r in lk["boundaries"] if not r["chunked"]),
                sum(not r["chunked"] for r in lk["boundaries"])],
            "inputs_bitwise_c4_03": [sum(r["bitwise_equal_c4_03"] for r in lk["inputs"]), len(lk["inputs"])],
            "inputs_sync_free": [sum(bool(r["sync_free"]) for r in lk["inputs"] if r["sync_free"] is not None),
                                 sum(r["sync_free"] is not None for r in lk["inputs"])],
            "graphs_bitwise": [sum(r["bitwise_equal"] for r in lk["graphs"]), len(lk["graphs"])],
            "graphs_max_abs_diff": max(r["max_abs_diff"] for r in lk["graphs"]),
            "opus_l6_bitwise_c4_03": [sum(r["bitwise_equal_c4_03"] for r in opus), len(opus)],
            "opus_l6_speedup": {"min": min(r["speedup"] for r in opus), "max": max(r["speedup"] for r in opus)},
            "opus_l6_speedup_geomean": {"min": min(r["speedup_paired"]["geomean_ratio"] for r in opus),
                                        "max": max(r["speedup_paired"]["geomean_ratio"] for r in opus)},
            "opus_l6_native_sigma_max": max(r["native"]["sigma_max"] for r in opus),
            "opus_l6_control_sigma_max": max(r["control"]["sigma_max"] for r in opus),
            "opus_l6_native_above_design": sum(r["native"]["sigma_max"] > r["design_ceiling"] for r in opus),
            "grid_cells": len(grid), "grid_chunked_cells": sum(r["chunked"] for r, _, _ in grid),
            "grid_bitwise_c4_03": sum(c["bitwise_equal_c4_03"] for _, _, c in grid),
            "grid_above_design_ceiling": sum(e > 0 for e in excess),
            "grid_above_common_ceiling": sum(c["sigma_max"] is None or c["sigma_max"] > lk_ceiling(payload)
                                             for _, _, c in grid),
            "grid_max_excess_over_design": None if not excess or math.isinf(max(excess)) else max(excess),
            "grid_sigma_max_worst": (None if not grid or any(c["sigma_max"] is None for _, _, c in grid)
                                     else max(c["sigma_max"] for _, _, c in grid))},
        "mma": _mma_totals(payload["mma"]),
    }


def lk_ceiling(payload: dict) -> float:
    return payload["protocol"]["long_k"]["ceiling"]


# ======================================================================================
# Run
# ======================================================================================
def run(args) -> dict:
    device = torch.device(args.protocol["device"])
    git = h.git_info()
    require_canonical_host(device, git, args.output, args.canonical)  # before any kernel launch
    if device.type != "cuda":
        raise RuntimeError("C5-01 runs the Blackwell Triton kernels and needs a CUDA device")
    log = (lambda *a: None) if args.silent else (lambda *a: print(*a, flush=True))
    numerics = set_numerics(device)
    env = q.environment(device)
    protocol = copy.deepcopy(args.protocol)
    code = code_sha256()
    cfg_hash = run_config_hash(args.quick, protocol, code, git["git_sha"], env)
    started = time.perf_counter()
    log(f"[c5-01] git {git['git_sha']} dirty={git['git_dirty']} {env.get('device')} sm_"
        f"{''.join(map(str, env.get('capability', [])))} hash {cfg_hash} canonical={args.canonical}")
    meta = {**git, "command": [sys.executable, *sys.argv], "environment": env, "numerics": numerics,
            "code_sha256": code, "code_modules": list(code_modules()),
            "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "loadavg_start": list(os.getloadavg()),
            "gpu_telemetry": {"start": h.gpu_telemetry(device)}}
    random.seed(0)
    payload = {"schema_version": SCHEMA_VERSION, "task": TASK, "quick": args.quick, "canonical": args.canonical,
               "meta": meta, "protocol": protocol, "run_config_hash": cfg_hash}
    sections = {"zero_regression": zero_regression, "timing": timing, "long_k": long_k,
                "mma": lambda p, d, lg: mm.characterize(p["mma"])}
    meta["section_seconds"] = {}
    for name, section in sections.items():
        began = time.perf_counter()
        payload[name] = section(protocol, device, log)
        meta["section_seconds"][name] = time.perf_counter() - began
        log(f"[c5-01] {name} done in {meta['section_seconds'][name]:.0f} s")
    meta.update({"timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "loadavg_end": list(os.getloadavg()),
                 "total_seconds": time.perf_counter() - started})
    meta["gpu_telemetry"]["end"] = h.gpu_telemetry(device)
    payload = json.loads(json.dumps(payload, allow_nan=False))
    payload["derived"] = derive(payload)
    payload = json.loads(json.dumps(payload, allow_nan=False))
    validate_results(payload, canonical=args.canonical)
    return payload


# ======================================================================================
# Validation
# ======================================================================================
def _validate_timing(payload: dict) -> None:
    spec = payload["protocol"]["timing"]
    cells = payload["timing"]["cells"]
    expected = [(s, k, m) for s in spec["shapes"] for k in spec["kernels"] for m in spec["modes"]]
    _require([(c["shape"], c["kernel"], c["mode"]) for c in cells] == expected, "timing cells differ from the protocol")
    _require(payload["timing"]["flush_bytes"] == spec["flush_l2_multiple"] * payload["timing"]["l2_cache_bytes"],
             "the L2 flush must be flush_l2_multiple x L2")
    for c in cells:
        where = f"timing {c['mode']} {c['kernel']} {c['shape']}"
        _require(c["dtype"] == spec["dtype"], f"{where}: dtype differs from the protocol")
        _require(isinstance(c["outputs_bitwise_equal"], bool), f"{where}: outputs_bitwise_equal must be a bool")
        _require(len(c["rounds"]) == spec["rounds"], f"{where}: round count differs")
        for i, r in enumerate(c["rounds"]):
            _require(r["round"] == i and r["order"] == williams_order(i), f"{where}: round {i} order differs")
            for v in VARIANTS:
                _require(r[v]["gpu_ms"] > 0 and r[v]["wall_ms"] > 0, f"{where}: non-positive sample")
        _assert_close(c["summary"], timing_summary(c["rounds"]), f"{where}.summary", 1e-9)


def _validate_long_k(payload: dict) -> None:
    lk, protocol = payload["long_k"], payload["protocol"]
    spec = protocol["long_k"]
    design = c3.design_ceilings(list(protocol["kernels"]))
    kernels, dtypes = protocol["kernels"], protocol["dtypes"]
    out_dtypes = {d: str(mm.DTYPES[d]).replace("torch.", "") for d in dtypes}
    identities = {
        "syrk": ([(r["shape"], r["dtype"], r["out_dtype"]) for r in lk["syrk"]],
                 [(s, d, o) for s in spec["symmetry_shapes"] for d in dtypes for o in (out_dtypes[d], "float32")]),
        "boundaries": ([(r["k"], r["shape"], r["kernel"], r["dtype"], r["chunked"], r["reference"])
                        for r in lk["boundaries"]],
                       [(k, [spec["boundary_rows"], k] if o == "wide" else [k, 3], kn, d, k > protocol["gate"],
                         "c4_03_chunked_gram" if k > protocol["gate"] else "legacy")
                        for k in spec["boundaries"] for o in ("wide", "tall") for kn in kernels for d in dtypes]),
        "inputs": ([(r["case"], r["shape"], r["kernel"], r["dtype"]) for r in lk["inputs"]],
                   _input_case_cells(protocol)),
        "graphs": ([(r["shape"], r["kernel"], r["dtype"]) for r in lk["graphs"]],
                   [(s, kn, d) for s in spec["graph_shapes"] for kn in kernels for d in dtypes]),
        "opus_l6": ([(r["shape"], r["kernel"], r["dtype"]) for r in lk["opus_l6"]],
                    [(s, kn, d) for s in spec["opus_shapes"] for kn in kernels for d in dtypes]),
    }
    for name, (recorded, expected) in identities.items():
        _require(recorded == expected, f"long_k {name} rows differ from the protocol")
    for r in lk["opus_l6"]:
        where = f"opus_l6 {r['shape']} {r['kernel']} {r['dtype']}"
        _require(len(r["samples"]) == spec["opus_reps"], f"{where}: sample count differs")
        for i, s in enumerate(r["samples"]):
            _require(s["round"] == i and s["order"] == williams_order(i), f"{where}: sample order differs")
            _require(all(s[v]["wall_ms"] > 0 for v in VARIANTS), f"{where}: non-positive sample")
        _assert_close(r["speedup"], speedup(r["samples"]), f"{where}.speedup", 1e-9)
        _assert_close(r["speedup_paired"], paired_speedup(r["samples"]), f"{where}.speedup_paired", 1e-9)
        _assert_close(r["design_ceiling"], design[r["kernel"]], f"{where}.design_ceiling", 1e-12)
    expected = [(f, e, s) for f, fam in spec["grid_families"].items() for e in fam["k_exponents"] for s in fam["seeds"]]
    _require([(r["family"], r["k_exponent"], r["seed"]) for r in lk["grid"]] == expected,
             "grid rows differ from the protocol families")
    names = {f"{k}_{d}" for k in protocol["kernels"] for d in protocol["dtypes"]}
    for r in lk["grid"]:
        fam = spec["grid_families"][r["family"]]
        shape = [fam["n"], r["k"]] if fam["orientation"] == "wide" else [r["k"], fam["n"]]
        _require(r["n"] == fam["n"] and r["shape"] == shape, f"grid {r['family']} 2^{r['k_exponent']}: shape differs")
        _require(set(r["cells"]) == names, f"grid {r['family']} 2^{r['k_exponent']}: kernel cells differ")
        _require(r["chunked"] == (r["k"] > protocol["gate"]) and r["k"] == 1 << r["k_exponent"], "grid chunk flag")
        for name, c in r["cells"].items():
            _assert_close(c["design_ceiling"], design[name.rsplit("_", 1)[0]], f"grid {name}.design_ceiling", 1e-12)
            if c["finite"]:
                _require(0 <= c["sigma_min"] <= c["sigma_max"] and c["rel_fro_err"] >= 0,
                         f"grid {name}: singular values out of order")


def _fp32_word(text: str) -> int:
    """Decode a hex/inf FP32 string from ``_hex`` into its 32-bit IEEE-754 word (rejects NaN)."""
    if not isinstance(text, str):
        raise ValueError(f"expected string FP32 value, got {text!r}")
    if text == "inf":
        value = math.inf
    elif text == "-inf":
        value = -math.inf
    else:
        value = float.fromhex(text)
    try:
        packed = struct.pack("<f", value)
    except OverflowError as exc:
        raise ValueError(f"FP32 overflow in {text!r}") from exc
    _require(not math.isnan(value) and struct.unpack("<f", packed)[0] == value,
             f"expected exact finite or infinite FP32 value, got {text!r}")
    return struct.unpack("<I", packed)[0]


def _validate_mma(payload: dict) -> None:
    from research.polar_opt.kernels_blackwell import _CONFIGS

    mma, spec = payload["mma"], payload["protocol"]["mma"]
    expected_witnesses = [
        ("Astra FP16 4-term diagonal", "fp16", math.perm(mm.GROUP, len(mm.ASTRA_FP16_WITNESS)),
         mm.ASTRA_FP16_HARDWARE),
        ("BF16 5-term diagonal (reconstructed)", "bf16", math.perm(mm.GROUP, len(mm.BF16_WITNESS)),
         mm.ASTRA_BF16_HARDWARE),
    ]
    _require([(w["name"], w["dtype"], w["permutations"]) for w in mma["witnesses"]]
             == [(n, dt, perms) for n, dt, perms, _ in expected_witnesses],
             "witnesses differ from the protocol")
    for w, (_, _, _, expected_audit) in zip(mma["witnesses"], expected_witnesses):
        audit_word = _fp32_word(w["audit_hardware"])
        _require(audit_word == _fp32_word(expected_audit), f"witness {w['name']}: audit_hardware differs")
        _require(len(w["syrk_configs_hardware"]) == len(_CONFIGS), f"witness {w['name']}: config count differs")
        expected_model_eq = (_fp32_word(w["sm120_model"]) == _fp32_word(w["probe_hardware"]) == audit_word)
        _require(w["model_equals_hardware"] is expected_model_eq,
                 f"witness {w['name']}: model_equals_hardware differs from recorded words")
        expected_cfg_eq = all(_fp32_word(v) == audit_word for v in w["syrk_configs_hardware"])
        _require(w["configs_equal_hardware"] is expected_cfg_eq,
                 f"witness {w['name']}: configs_equal_hardware differs from recorded words")
        _require(0 <= w["permutations_equal_hardware"] <= w["permutations"],
                 f"witness {w['name']}: permutation count")

    _require([r["name"] for r in mma["exponent_table"]] == [c["name"] for c in mm.ASTRA_EXPONENT_TABLE],
             "exponent_table rows differ from the protocol")
    for r, case in zip(mma["exponent_table"], mm.ASTRA_EXPONENT_TABLE):
        audit_word = _fp32_word(r["audit_hardware"])
        _require(audit_word == _fp32_word(mm._hex(case["hardware"])),
                 f"exponent_table {r['name']}: audit_hardware differs")
        expected_eq = (_fp32_word(r["sm120_model"]) == _fp32_word(r["probe_hardware"]) == audit_word)
        _require(r["model_equals_hardware"] is expected_eq,
                 f"exponent_table {r['name']}: model_equals_hardware differs from recorded words")

    _require([(r["name"], r["dtype"]) for r in mma["edge_cases"]]
             == [(name, dtype) for name, dtype, *_ in mm.EDGE_CASES],
             "edge_cases differ from the protocol")
    for r in mma["edge_cases"]:
        bits_text = r.get("hardware_bits")
        _require(isinstance(bits_text, str) and bits_text.startswith("0x") and _is_hex(bits_text[2:], 8),
                 f"edge_case {r['name']}: hardware_bits format")
        hw_word = int(bits_text, 16)
        if r["hardware"] == "nan":
            _require((hw_word & 0x7F800000) == 0x7F800000 and (hw_word & 0x007FFFFF) != 0,
                     f"edge_case {r['name']}: non-NaN hardware_bits")
            expected_eq = (r["sm120_model"] == "nan" and hw_word == mm.NAN_BITS)
        else:
            _require(hw_word == _fp32_word(r["hardware"]),
                     f"edge_case {r['name']}: hardware_bits differs from hardware")
            expected_eq = (r["sm120_model"] != "nan" and _fp32_word(r["sm120_model"]) == hw_word)
        _require(r["model_equals_hardware"] is expected_eq,
                 f"edge_case {r['name']}: model_equals_hardware differs from recorded words")

    for r in [*mma["stress"], *mma["syrk_configs"], *mma["audit_seeds"]]:
        _require(0 <= r["bit_matches"] <= r["cases"] and _is_hex(r["hardware_sha256"], 64), f"{r['category']}: counts")
        _require(len(r["mismatches"]) == min(4, r["cases"] - r["bit_matches"]), f"{r['category']}: mismatch records")

    expected_stress = []
    for dt in mm.DTYPES:
        for cat in mm.STRESS_CATEGORIES:
            expected_stress.append((f"{cat}/{dt}", dt, "single", mm.GROUP, mm.GROUP, spec["stress_dots"]))
        for k, bk in spec["chains"]:
            expected_stress.append((f"chain_k{k}_bk{bk}/{dt}", dt, "chain", k, bk, spec["chain_dots"]))
    _require([(r["category"], r["dtype"], r["kind"], r["k"], r["bk"], r["cases"]) for r in mma["stress"]]
             == expected_stress, "stress rows differ from the protocol")
    _require(all(isinstance(r["scalar_crosscheck"], list) and len(r["scalar_crosscheck"]) == 2
                 and 0 <= r["scalar_crosscheck"][0] <= r["scalar_crosscheck"][1] == spec["scalar_crosscheck"]
                 for r in mma["stress"] if r["kind"] == "single"), "scalar_crosscheck differs")

    cfg_records = [mm.config_record(c) for c in _CONFIGS]
    expected_syrk = [(f"syrk_{n}x{k}/{dt}", dt, [n, k], cfg, n * (n + 1) // 2)
                     for dt in mm.DTYPES for n, k in spec["syrk_shapes"] for cfg in cfg_records]
    _require([(r["category"], r["dtype"], r["shape"], r["config"], r["cases"]) for r in mma["syrk_configs"]]
             == expected_syrk, "syrk_configs differ from the protocol")
    _require(all(0 <= r["fused_store_matches"] <= r["cases"] for r in mma["syrk_configs"]),
             "syrk_configs fused_store_matches")

    expected_audit = []
    for dt in mm.DTYPES:
        for label in ("seed419_9x2033", "seed419_9x2033_unit_frobenius"):
            for cfg in cfg_records:
                expected_audit.append((f"{label}/{dt}", dt, cfg, 45))
        for cfg in cfg_records:
            for part in ("offdiag", "diag0"):
                expected_audit.append((f"seed927_4096x2x16_{part}/{dt}", dt, cfg, 4096))
        for cfg in cfg_records:
            expected_audit.append((f"seed719_32768x2x16_offdiag/{dt}", dt, cfg, 32768))
    _require([(r["category"], r["dtype"], r["config"], r["cases"]) for r in mma["audit_seeds"]]
             == expected_audit, "audit_seeds differ from the protocol")
    _require(all(0 <= r["fused_store_matches"] <= r["cases"]
                 for r in mma["audit_seeds"] if r["category"].startswith("seed719_")),
             "audit_seeds fused_store_matches")

    _require([(r["hypothesis"], r["dtype"], r["cases"]) for r in mma["hypotheses"]]
             == [(name, dt, spec["hypothesis_dots"]) for dt in mm.DTYPES for name in mm.HYPOTHESES],
             "hypotheses differ")


def validate_results(payload: dict, *, canonical: bool | None = None) -> None:
    """Raise ValueError unless ``payload`` is a complete, internally consistent C5-01 result."""
    canonical = payload.get("canonical") if canonical is None else canonical
    _require(payload.get("schema_version") == SCHEMA_VERSION, "schema_version differs")
    meta, protocol = payload["meta"], payload["protocol"]
    _require(isinstance(meta.get("git_dirty"), bool), "git_dirty must be recorded")
    _require(_is_hex(meta.get("code_sha256"), 64), "code_sha256 must be 64-hex")
    _require(all(any(m.endswith("." + needed) for m in meta["code_modules"]) for needed in REQUIRED_CODE),
             "code_modules does not cover the driver, the model and the kernels")
    env = meta["environment"]
    _require(all(env.get(k) is not None for k in ENV_KEYS), f"environment needs {ENV_KEYS}")
    _require(_is_hex(payload.get("run_config_hash"), 16)
             and payload["run_config_hash"] == run_config_hash(payload["quick"], protocol, meta["code_sha256"],
                                                               meta["git_sha"], env),
             "run_config_hash differs from the recomputed digest")
    numerics = meta["numerics"]
    _require(numerics["allow_tf32"] is False and numerics["allow_fp16_reduced_precision_reduction"] is False
             and numerics["allow_bf16_reduced_precision_reduction"] is False, "full-precision numerics are required")
    _require(protocol["gate"] == kbw.SYRK_CHUNK_GATE and protocol["chunk"] == kbw.SYRK_CHUNK_SIZE
             and protocol["max_bytes"] == kbw.SYRK_CHUNK_MAX_BYTES, "gate constants differ from kernels_blackwell")
    _require(protocol["legacy_base"] == LEGACY_BASE == payload["zero_regression"]["legacy_base"], "legacy base differs")
    if canonical:
        _require(payload["canonical"] is True and payload["quick"] is False, "a canonical result cannot be --quick")
        _require(_is_hex(meta.get("git_sha"), 40) and meta["git_dirty"] is False,
                 "canonical results need a 40-hex git_sha and a clean tree")
        _require(list(env["capability"]) >= [12, 0], "a canonical result needs an sm_120+ CUDA device")
        _require(protocol == CANONICAL, "protocol differs from the canonical protocol")
        modules = meta["code_modules"]
        _require(len(set(modules)) == len(modules) and all(m.startswith("research.") for m in modules)
                 and modules == list(code_modules()), "code_modules differ from the static import closure")
    zr = payload["zero_regression"]
    spec = protocol["zero_regression"]
    expected = [(s, seed, k, d) for s in spec["shapes"] for seed in spec["seeds"]
                for k in (*protocol["kernels"], "gram_ns", "triton_syrk") for d in protocol["dtypes"]]
    _require([(r["shape"], r["seed"], r["kernel"], r["dtype"]) for r in zr["rows"]] == expected,
             "zero_regression rows differ from the protocol")
    _require(all(r["k"] <= protocol["gate"] for r in zr["rows"]), "zero_regression covers K <= gate only")
    _validate_timing(payload)
    _validate_long_k(payload)
    _validate_mma(payload)
    _assert_close(payload["derived"], json.loads(json.dumps(derive(payload), allow_nan=False)), "derived", 1e-12)
    if canonical:
        d = payload["derived"]
        _require(d["zero_regression"]["bitwise_equal"] == d["zero_regression"]["cells"] > 0, "a K <= gate cell differs")
        _require(d["zero_regression"]["launches_equal"] == d["zero_regression"]["launch_cells"]
                 and d["zero_regression"]["launches_at_512x2048"] == spec["launch_expected"], "launch counts differ")
        _require(d["long_k"]["grid_above_design_ceiling"] == 0, "a chunked cell exceeds its own design ceiling")
        _require(d["timing"]["outputs_bitwise_equal"] == d["timing"]["cells"], "a timing cell's outputs differ")
        _require(d["long_k"]["opus_l6_native_above_design"] == 0, "an Opus L6 cell exceeds its own design ceiling")
        m = d["mma"]
        _require(m["all_dot_products"][0] == m["all_dot_products"][1] > 0, "promotion: mma all_dot_products")
        for key in ("witnesses", "witness_permutations", "exponent_table", "edge_cases", "syrk_fused_store",
                    "audit_seed719_fused_store", "scalar_crosscheck"):
            _require(m[key][0] == m[key][1], f"promotion: mma {key}")
        for key in ("syrk_symmetric", "syrk_sync_free", "graphs_bitwise", "boundaries_bitwise_reference",
                    "inputs_bitwise_c4_03", "opus_l6_bitwise_c4_03"):
            _require(d["long_k"][key][0] == d["long_k"][key][1], f"promotion: long_k {key}")


# ======================================================================================
# CLI
# ======================================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="small grids and few rounds (never canonical)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args(argv)
    args.canonical = not args.quick
    if args.output is None:
        args.output = QUICK_OUTPUT if args.quick else CANONICAL_OUTPUT
    if reserved(args.output) and (args.quick or args.output.resolve().name != CANONICAL_OUTPUT.name):
        parser.error(f"Refusing to write the reserved {args.output.name} from a --quick run; "
                     f"choose a path outside {RESERVED_PATTERN}")
    if args.canonical and (args.output.is_symlink() or args.output.resolve() != CANONICAL_OUTPUT.resolve()):
        parser.error(f"The canonical protocol writes {CANONICAL_OUTPUT} of this checkout (no symlink); "
                     "use --quick for other paths")
    args.protocol = copy.deepcopy(QUICK if args.quick else CANONICAL)
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    payload = write_json_atomic(args.output, run(args))
    if not args.silent:
        print(json.dumps(payload["derived"], indent=1))
        print(f"Wrote {args.output}")
    return payload


if __name__ == "__main__":
    main()
