"""Small-shape NVFP4 W4A4 fast path for Blackwell (sm_120; Cycle 6 C6-02).

A drop-in, bit-exact replacement for the activation quantizer and the GEMM call of
``nvfp4_inference_blackwell`` (C5-03), plus a fused cross-block mix for the C6-01 wide rotations. The
C5-03 module stays unchanged: the C5-03 and C6-01 validators pin its bytes. This module makes the same
operands (``packed``, swizzled ``scale``, FP32 ``global``) and the same GEMM output bits.

Quantizer (``quantize_activation``, ``launch_amax``, ``launch_quant``, ``FastNVFP4Linear``)
    One program element is one NVFP4 group of 16 inputs, and each thread holds its 16 values as 16 FP32
    registers (a tuple of tensors). The block-16 Hadamard and butterfly stages are then register
    arithmetic in the operation order of the eager reference, with no ``permute``/``split``/``join``
    layout conversions. Contiguous inputs load with 128-bit ``ld.global.v2.b64`` (inline PTX; 64-bit or
    per-element loads when the alignment or strides do not allow it), and the 8 code bytes of a group
    leave with one 64-bit store. Tiles are 1-D over (rows x groups), so none runs half empty when K / 16
    is not a power of two (K = 1408 has 88 groups). ``FastNVFP4Linear`` keeps the butterfly coefficients
    as (16, G, 4) ``(cos, sin)`` pairs, so a warp loads them coalesced, 128 bits per two pairs.
    Dynamic mode keeps the two launches of C5-03 (deterministic partial maxima, no atomics, PDL), with
    one pass-1 tile per program where possible (at most ``PASS1_PROGRAMS`` = 512 partials).
    Large inputs are DRAM bound; where the C5-03 kernels are as fast or faster there (``delegated``:
    ``none`` from 2^23 elements, the butterfly from 2^21 elements with K >= 2048, never ``hadamard_b16``),
    ``quantize_activation`` and ``FastNVFP4Linear`` hand them to ``nvfp4_inference_blackwell``.

GEMM (``nvfp4_scaled_mm_lt``)
    ``torch._scaled_mm_v2`` multiplies the two FP32 tensor scales in a separate elementwise kernel
    (``aten::mul``, about 1.6 us under a CUDA graph) and passes the product to cuBLASLt as a device
    ``alpha``. Here the quantizer writes ``alpha = global_scale * w_global`` (one rounded FP32 product,
    the same value) next to the tensor scale, and the GEMM calls ``cublasLtMatmul`` directly (ctypes)
    with the matmul descriptor, layouts, 1 MiB workspace limit and first heuristic algorithm that
    PyTorch uses. The output equals ``nvfp4_scaled_mm_public`` bit for bit (checked by the tests and the
    C6-02 benchmark). In ``FastNVFP4Linear``, plans whose heuristic needs a workspace fall back to
    ``_scaled_mm_v2``; ``nvfp4_scaled_mm_lt`` allocates the workspace instead. A per-plan lock makes the
    scale-pointer update and the submission one host section (concurrent threads on their own streams).
    Plans are an LRU cache of ``LT_PLAN_CACHE_MAX`` entries (``clear_lt_plan_cache`` closes them).

Cross-block mix (``FastCrossInput``)
    ``rotated_qat.CrossInput`` runs the C6-01 cross-block mix as ``x.float()`` and two batched cuBLAS
    products (50-110 us per layer at 1024 rows). One Triton kernel reads the input directly and computes
    the same two contractions with the two-half FMA order of the cuBLAS ``gemmSN`` kernels. That order
    gives the same FP32 bits wherever cuBLAS uses it (measured); elsewhere results differ by rounding,
    which can change FP4 codes (a latency path, not exact; ``exact=True`` uses the reference mix).

Tuning state: ``options_scope`` overrides ``OPTIONS`` per context (``contextvars``); ``TILES`` overrides
take effect on the next call (the launch-plan caches are keyed on the resolved tile).

Numerics are those of ``nvfp4_inference_blackwell`` (IEEE ``div_rn``, no FMA contraction, native
``cvt.rn.satfinite.e2m1x2.f32`` with NaN mapped to a signed zero, E4M3 via ``cvt.rn.satfinite``).

Every launch runs on the current stream without a host synchronization, so the modules are CUDA-graph
capturable. Warm up each shape eagerly once before capture (the first call compiles the kernels and
builds the cuBLASLt plan).
"""

from __future__ import annotations

import collections
import contextlib
import contextvars
import ctypes
import functools
import threading
from pathlib import Path
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from torch import Tensor, nn

from research.quant_ptq import micro_rotated_ptq as q
from research.quant_ptq import nvfp4_inference_blackwell as nb
from research.quant_ptq.nvfp4_inference_blackwell import NVFP4Activation, scale_layout

ROTATIONS = nb.ROTATIONS
MAX_PARTIALS = nb.MAX_PARTIALS  # capacity of a ``launch_amax`` workspace (pass 1 writes at most PASS1_PROGRAMS)
PASS1_PROGRAMS = 512
# Switches for the benchmark ablation (read at launch time; a CUDA graph keeps the setting of its capture):
# PDL between the passes, 128/64-bit vector loads and 64-bit code stores, the direct cuBLASLt GEMM with the
# fused alpha, the rotated FP32 stash of the dynamic butterfly (off: measured slower), and the delegation of
# large inputs to C5-03.
OPTIONS = {"pdl": True, "vector_io": True, "lt_gemm": True, "stash": False, "delegate_large": True}
# Scoped overrides of ``OPTIONS`` (``options_scope``): a ``ContextVar``, so concurrent threads or tasks that tune
# different settings never see each other's changes. ``OPTIONS`` itself stays the process-wide default.
_OPTION_OVERRIDES: contextvars.ContextVar[dict | None] = contextvars.ContextVar("c6_02_options", default=None)


def _option(name: str):
    """The setting ``name`` of the current context: the innermost ``options_scope`` override, else ``OPTIONS``."""
    overrides = _OPTION_OVERRIDES.get()
    if overrides is not None and name in overrides:
        return overrides[name]
    return OPTIONS[name]


@contextlib.contextmanager
def options_scope(**changes):
    """Override ``OPTIONS`` keys in this context only (nested scopes merge; a CUDA graph keeps its capture's)."""
    unknown = set(changes) - set(OPTIONS)
    if unknown:
        raise KeyError(f"unknown options {sorted(unknown)}")
    token = _OPTION_OVERRIDES.set({**(_OPTION_OVERRIDES.get() or {}), **changes})
    try:
        yield
    finally:
        _OPTION_OVERRIDES.reset(token)

# Large inputs are DRAM bound. Measured on the RTX 5080 (CUDA graphs, 4x L2 flush; C5-03 time / C6-02 time):
# hadamard_b16 1.08-1.14 at 8-12M elements and 1.00 at 16.8M; none 1.00 at 8-12M and 0.98 at 16.8M; the
# butterfly 1.01-1.05 below 2M elements and at K = 1408 (1.05 at 8192 x 1408), but 0.98-1.02 at K >= 2048 from
# 2M elements and 0.94 at 2048 x 8192, where the C5-03 2D tiles share the coefficients of a group across 16-32
# rows. From these sizes on, ``quantize_activation`` and ``FastNVFP4Linear`` hand the input to
# ``nvfp4_inference_blackwell`` unchanged (same bits); ``launch_amax`` / ``launch_quant`` never delegate.
# These limits were set after a first measured attempt, from sweeps on overlapping shapes: they are
# performance-informed routing for the measured shapes, not a predicted crossover for unseen workloads.
DELEGATE_FROM = {"none": 1 << 23, "hadamard_b16": None, "adaptive_butterfly_b16": 1 << 21}
DELEGATE_MIN_K = {"none": 0, "hadamard_b16": 0, "adaptive_butterfly_b16": 2048}
STASH_MAX_BYTES = 16 << 20  # largest FP32 rotated stash (M * K * 4 bytes) of the butterfly dynamic path


# ======================================================================================
# Tiles (chosen from sweeps on the RTX 5080 at the eight C5-03 shapes)
# ======================================================================================
# (M, K, rotation) -> tile override, for tuning sweeps (empty otherwise). The launch-plan caches are keyed on
# the resolved tile, so a change here takes effect on the next call.
TILES = {"pass1": {}, "pass2": {}}


def _pass1_tile(M: int, K: int, rotation: str) -> tuple[int, int, int, int]:
    """(groups per tile, rows per strip, warps, most programs) of pass 1.

    Pass 1 is latency bound: one tile per program where possible (up to 512 programs, so up to 512
    partials), with 4 groups per thread for the butterfly and 2 otherwise for memory-level parallelism.
    """
    found = TILES["pass1"].get((M, K, rotation))
    if found:
        return found
    per_thread = 4 if rotation == "adaptive_butterfly_b16" else 2
    want = -(-M * (K // 16) // 512)
    block = min(max(1 << max(want - 1, 0).bit_length(), 32 * per_thread), 1024)
    return block, 1, max(1, min(8, block // (32 * per_thread))), 512


def _pass2_tile(M: int, K: int, rotation: str) -> tuple[int, int, int]:
    """(groups per program, rows per strip, warps) of pass 2 (or of the static single pass): one group per thread."""
    found = TILES["pass2"].get((M, K, rotation))
    if found:
        return found
    return 128, 1, 4


# ======================================================================================
# Triton kernels
# ======================================================================================
@triton.jit
def _nanmax(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _u16_to_f32(h, DT: tl.constexpr):
    """FP32 value of the 16-bit pattern ``h`` (uint64 holding 0..65535) of a BF16 or FP16 input."""
    if DT == tl.bfloat16:
        return (h.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    else:
        return h.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _load16(x_ptr, off, mask, stride_k, VEC: tl.constexpr):
    """The 16 inputs of each group as FP32 (0 where masked). ``off`` is the element offset of input 0.

    ``VEC`` (``stride_k == 1``): 1 = groups start 8-byte aligned, 64-bit loads; 2 = 16-byte aligned,
    128-bit ``ld.global.v2.b64`` loads (``_ld128``); 0 = any strides, one load per input.
    """
    DT: tl.constexpr = x_ptr.dtype.element_ty
    if VEC > 0:
        p = (x_ptr + off).to(tl.pointer_type(tl.uint64), bitcast=True)
        if DT == tl.float32:
            if VEC == 2:
                u0, u1 = _ld128(p.to(tl.int64), mask)
                u2, u3 = _ld128((p + 2).to(tl.int64), mask)
                u4, u5 = _ld128((p + 4).to(tl.int64), mask)
                u6, u7 = _ld128((p + 6).to(tl.int64), mask)
            else:
                u0 = tl.load(p + 0, mask=mask, other=0)
                u1 = tl.load(p + 1, mask=mask, other=0)
                u2 = tl.load(p + 2, mask=mask, other=0)
                u3 = tl.load(p + 3, mask=mask, other=0)
                u4 = tl.load(p + 4, mask=mask, other=0)
                u5 = tl.load(p + 5, mask=mask, other=0)
                u6 = tl.load(p + 6, mask=mask, other=0)
                u7 = tl.load(p + 7, mask=mask, other=0)
            lo: tl.constexpr = 0xFFFFFFFF
            return ((u0 & lo).to(tl.uint32).to(tl.float32, bitcast=True),
                    (u0 >> 32).to(tl.uint32).to(tl.float32, bitcast=True),
                    (u1 & lo).to(tl.uint32).to(tl.float32, bitcast=True),
                    (u1 >> 32).to(tl.uint32).to(tl.float32, bitcast=True),
                    (u2 & lo).to(tl.uint32).to(tl.float32, bitcast=True),
                    (u2 >> 32).to(tl.uint32).to(tl.float32, bitcast=True),
                    (u3 & lo).to(tl.uint32).to(tl.float32, bitcast=True),
                    (u3 >> 32).to(tl.uint32).to(tl.float32, bitcast=True),
                    (u4 & lo).to(tl.uint32).to(tl.float32, bitcast=True),
                    (u4 >> 32).to(tl.uint32).to(tl.float32, bitcast=True),
                    (u5 & lo).to(tl.uint32).to(tl.float32, bitcast=True),
                    (u5 >> 32).to(tl.uint32).to(tl.float32, bitcast=True),
                    (u6 & lo).to(tl.uint32).to(tl.float32, bitcast=True),
                    (u6 >> 32).to(tl.uint32).to(tl.float32, bitcast=True),
                    (u7 & lo).to(tl.uint32).to(tl.float32, bitcast=True),
                    (u7 >> 32).to(tl.uint32).to(tl.float32, bitcast=True))
        else:
            if VEC == 2:
                u0, u1 = _ld128(p.to(tl.int64), mask)
                u2, u3 = _ld128((p + 2).to(tl.int64), mask)
            else:
                u0 = tl.load(p + 0, mask=mask, other=0)
                u1 = tl.load(p + 1, mask=mask, other=0)
                u2 = tl.load(p + 2, mask=mask, other=0)
                u3 = tl.load(p + 3, mask=mask, other=0)
            m: tl.constexpr = 0xFFFF
            return (_u16_to_f32(u0 & m, DT), _u16_to_f32((u0 >> 16) & m, DT),
                    _u16_to_f32((u0 >> 32) & m, DT), _u16_to_f32(u0 >> 48, DT),
                    _u16_to_f32(u1 & m, DT), _u16_to_f32((u1 >> 16) & m, DT),
                    _u16_to_f32((u1 >> 32) & m, DT), _u16_to_f32(u1 >> 48, DT),
                    _u16_to_f32(u2 & m, DT), _u16_to_f32((u2 >> 16) & m, DT),
                    _u16_to_f32((u2 >> 32) & m, DT), _u16_to_f32(u2 >> 48, DT),
                    _u16_to_f32(u3 & m, DT), _u16_to_f32((u3 >> 16) & m, DT),
                    _u16_to_f32((u3 >> 32) & m, DT), _u16_to_f32(u3 >> 48, DT))
    else:
        p = x_ptr + off
        return (tl.load(p, mask=mask, other=0.0).to(tl.float32),
                tl.load(p + stride_k, mask=mask, other=0.0).to(tl.float32),
                tl.load(p + 2 * stride_k, mask=mask, other=0.0).to(tl.float32),
                tl.load(p + 3 * stride_k, mask=mask, other=0.0).to(tl.float32),
                tl.load(p + 4 * stride_k, mask=mask, other=0.0).to(tl.float32),
                tl.load(p + 5 * stride_k, mask=mask, other=0.0).to(tl.float32),
                tl.load(p + 6 * stride_k, mask=mask, other=0.0).to(tl.float32),
                tl.load(p + 7 * stride_k, mask=mask, other=0.0).to(tl.float32),
                tl.load(p + 8 * stride_k, mask=mask, other=0.0).to(tl.float32),
                tl.load(p + 9 * stride_k, mask=mask, other=0.0).to(tl.float32),
                tl.load(p + 10 * stride_k, mask=mask, other=0.0).to(tl.float32),
                tl.load(p + 11 * stride_k, mask=mask, other=0.0).to(tl.float32),
                tl.load(p + 12 * stride_k, mask=mask, other=0.0).to(tl.float32),
                tl.load(p + 13 * stride_k, mask=mask, other=0.0).to(tl.float32),
                tl.load(p + 14 * stride_k, mask=mask, other=0.0).to(tl.float32),
                tl.load(p + 15 * stride_k, mask=mask, other=0.0).to(tl.float32))


@triton.jit
def _angles(cos_ptr, sin_ptr, goff, sj, amask, ROT: tl.constexpr, PACKED: tl.constexpr):
    """The 64 butterfly coefficients of each group (stage s: cos at 16 s + p, sin at 16 s + 8 + p).

    Coefficient j = 8 s + p of group g is at ``g * sg + j * sj`` (``goff = g * sg``): (sg, sj) = (32, 1)
    for the (G, 4, 8) tables of the public API; (1, G) for the (32, G) layout of ``FastNVFP4Linear``,
    whose loads coalesce across the consecutive groups of a warp. ``PACKED``: ``cos_ptr`` holds (cos, sin)
    FP32 pairs as (16, G, 4) = (c_2q, s_2q, c_2q+1, s_2q+1), one 128-bit load per two pairs. All 64 are loaded up front (better latency hiding
    than per-stage loads) and reused for every row of a strip. Other rotations get a placeholder.
    """
    if ROT == 2:
        if PACKED:
            cs = cos_ptr.to(tl.pointer_type(tl.uint64), bitcast=True)
            a0, b0 = _ld128((cs + (goff + 0 * sj) * 2).to(tl.int64), amask)
            a1, b1 = _ld128((cs + (goff + 1 * sj) * 2).to(tl.int64), amask)
            a2, b2 = _ld128((cs + (goff + 2 * sj) * 2).to(tl.int64), amask)
            a3, b3 = _ld128((cs + (goff + 3 * sj) * 2).to(tl.int64), amask)
            a4, b4 = _ld128((cs + (goff + 4 * sj) * 2).to(tl.int64), amask)
            a5, b5 = _ld128((cs + (goff + 5 * sj) * 2).to(tl.int64), amask)
            a6, b6 = _ld128((cs + (goff + 6 * sj) * 2).to(tl.int64), amask)
            a7, b7 = _ld128((cs + (goff + 7 * sj) * 2).to(tl.int64), amask)
            a8, b8 = _ld128((cs + (goff + 8 * sj) * 2).to(tl.int64), amask)
            a9, b9 = _ld128((cs + (goff + 9 * sj) * 2).to(tl.int64), amask)
            a10, b10 = _ld128((cs + (goff + 10 * sj) * 2).to(tl.int64), amask)
            a11, b11 = _ld128((cs + (goff + 11 * sj) * 2).to(tl.int64), amask)
            a12, b12 = _ld128((cs + (goff + 12 * sj) * 2).to(tl.int64), amask)
            a13, b13 = _ld128((cs + (goff + 13 * sj) * 2).to(tl.int64), amask)
            a14, b14 = _ld128((cs + (goff + 14 * sj) * 2).to(tl.int64), amask)
            a15, b15 = _ld128((cs + (goff + 15 * sj) * 2).to(tl.int64), amask)
            return (_lo(a0), _lo(b0), _lo(a1), _lo(b1), _lo(a2), _lo(b2), _lo(a3), _lo(b3),
                    _hi(a0), _hi(b0), _hi(a1), _hi(b1), _hi(a2), _hi(b2), _hi(a3), _hi(b3),
                    _lo(a4), _lo(b4), _lo(a5), _lo(b5), _lo(a6), _lo(b6), _lo(a7), _lo(b7),
                    _hi(a4), _hi(b4), _hi(a5), _hi(b5), _hi(a6), _hi(b6), _hi(a7), _hi(b7),
                    _lo(a8), _lo(b8), _lo(a9), _lo(b9), _lo(a10), _lo(b10), _lo(a11), _lo(b11),
                    _hi(a8), _hi(b8), _hi(a9), _hi(b9), _hi(a10), _hi(b10), _hi(a11), _hi(b11),
                    _lo(a12), _lo(b12), _lo(a13), _lo(b13), _lo(a14), _lo(b14), _lo(a15), _lo(b15),
                    _hi(a12), _hi(b12), _hi(a13), _hi(b13), _hi(a14), _hi(b14), _hi(a15), _hi(b15))
        else:
            c0 = _coef8(cos_ptr, goff, sj, 0, amask)
            s0 = _coef8(sin_ptr, goff, sj, 0, amask)
            c1 = _coef8(cos_ptr, goff, sj, 8, amask)
            s1 = _coef8(sin_ptr, goff, sj, 8, amask)
            c2 = _coef8(cos_ptr, goff, sj, 16, amask)
            s2 = _coef8(sin_ptr, goff, sj, 16, amask)
            c3 = _coef8(cos_ptr, goff, sj, 24, amask)
            s3 = _coef8(sin_ptr, goff, sj, 24, amask)
            return (c0[0], c0[1], c0[2], c0[3], c0[4], c0[5], c0[6], c0[7],
                    s0[0], s0[1], s0[2], s0[3], s0[4], s0[5], s0[6], s0[7],
                    c1[0], c1[1], c1[2], c1[3], c1[4], c1[5], c1[6], c1[7],
                    s1[0], s1[1], s1[2], s1[3], s1[4], s1[5], s1[6], s1[7],
                    c2[0], c2[1], c2[2], c2[3], c2[4], c2[5], c2[6], c2[7],
                    s2[0], s2[1], s2[2], s2[3], s2[4], s2[5], s2[6], s2[7],
                    c3[0], c3[1], c3[2], c3[3], c3[4], c3[5], c3[6], c3[7],
                    s3[0], s3[1], s3[2], s3[3], s3[4], s3[5], s3[6], s3[7])
    else:
        return (goff,)


@triton.jit
def _coef8(ptr, goff, sj, j0: tl.constexpr, amask):
    """Coefficients j0 .. j0 + 7 of each group from a (G, 4, 8) or (32, G) table (see ``_angles``)."""
    return (tl.load(ptr + goff + j0 * sj, mask=amask, other=0.0),
            tl.load(ptr + goff + (j0 + 1) * sj, mask=amask, other=0.0),
            tl.load(ptr + goff + (j0 + 2) * sj, mask=amask, other=0.0),
            tl.load(ptr + goff + (j0 + 3) * sj, mask=amask, other=0.0),
            tl.load(ptr + goff + (j0 + 4) * sj, mask=amask, other=0.0),
            tl.load(ptr + goff + (j0 + 5) * sj, mask=amask, other=0.0),
            tl.load(ptr + goff + (j0 + 6) * sj, mask=amask, other=0.0),
            tl.load(ptr + goff + (j0 + 7) * sj, mask=amask, other=0.0))


@triton.jit
def _ld128(addr, mask):
    """Two 64-bit words from one 16-byte aligned ``ld.global.v2.b64`` at byte address ``addr`` (0 where masked)."""
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.b32 p, $3, 0; mov.b64 $0, 0; mov.b64 $1, 0; @p ld.global.v2.b64 {$0, $1}, [$2]; }",
        "=l,=l,l,r", [addr, mask.to(tl.int32)], dtype=(tl.uint64, tl.uint64), is_pure=False, pack=1)


@triton.jit
def _lo(u):
    return (u & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)


@triton.jit
def _hi(u):
    return (u >> 32).to(tl.uint32).to(tl.float32, bitcast=True)


@triton.jit
def _rotate16(v, ang, ROT: tl.constexpr):
    """Block-16 rotation of the 16 FP32 columns ``v`` (a tuple), with the operations of the eager reference.

    ROT 1: the four ``fwht`` stages (a + d, a - d) and the scale 16 ** -0.5 = 0.25. ROT 2: the four
    ``Butterfly._stage`` reflections (c a + s d, s a - c d). Stage s pairs i and i + 2^s; its pair index
    is p = (i // 2^(s+1)) 2^s + i % 2^s, with cos at ``ang[16 s + p]`` and sin at ``ang[16 s + 8 + p]``
    (``_angles``: all 64 loaded up front, which hides their latency better than per-stage loads).
    """
    if ROT == 1:
        v = (v[0] + v[1], v[0] - v[1], v[2] + v[3], v[2] - v[3],
             v[4] + v[5], v[4] - v[5], v[6] + v[7], v[6] - v[7],
             v[8] + v[9], v[8] - v[9], v[10] + v[11], v[10] - v[11],
             v[12] + v[13], v[12] - v[13], v[14] + v[15], v[14] - v[15])
        v = (v[0] + v[2], v[1] + v[3], v[0] - v[2], v[1] - v[3],
             v[4] + v[6], v[5] + v[7], v[4] - v[6], v[5] - v[7],
             v[8] + v[10], v[9] + v[11], v[8] - v[10], v[9] - v[11],
             v[12] + v[14], v[13] + v[15], v[12] - v[14], v[13] - v[15])
        v = (v[0] + v[4], v[1] + v[5], v[2] + v[6], v[3] + v[7],
             v[0] - v[4], v[1] - v[5], v[2] - v[6], v[3] - v[7],
             v[8] + v[12], v[9] + v[13], v[10] + v[14], v[11] + v[15],
             v[8] - v[12], v[9] - v[13], v[10] - v[14], v[11] - v[15])
        v = (v[0] + v[8], v[1] + v[9], v[2] + v[10], v[3] + v[11],
             v[4] + v[12], v[5] + v[13], v[6] + v[14], v[7] + v[15],
             v[0] - v[8], v[1] - v[9], v[2] - v[10], v[3] - v[11],
             v[4] - v[12], v[5] - v[13], v[6] - v[14], v[7] - v[15])
        v = (v[0] * 0.25, v[1] * 0.25, v[2] * 0.25, v[3] * 0.25,
             v[4] * 0.25, v[5] * 0.25, v[6] * 0.25, v[7] * 0.25,
             v[8] * 0.25, v[9] * 0.25, v[10] * 0.25, v[11] * 0.25,
             v[12] * 0.25, v[13] * 0.25, v[14] * 0.25, v[15] * 0.25)
    elif ROT == 2:
        v = (ang[0] * v[0] + ang[8] * v[1], ang[8] * v[0] - ang[0] * v[1],
             ang[1] * v[2] + ang[9] * v[3], ang[9] * v[2] - ang[1] * v[3],
             ang[2] * v[4] + ang[10] * v[5], ang[10] * v[4] - ang[2] * v[5],
             ang[3] * v[6] + ang[11] * v[7], ang[11] * v[6] - ang[3] * v[7],
             ang[4] * v[8] + ang[12] * v[9], ang[12] * v[8] - ang[4] * v[9],
             ang[5] * v[10] + ang[13] * v[11], ang[13] * v[10] - ang[5] * v[11],
             ang[6] * v[12] + ang[14] * v[13], ang[14] * v[12] - ang[6] * v[13],
             ang[7] * v[14] + ang[15] * v[15], ang[15] * v[14] - ang[7] * v[15])
        v = (ang[16] * v[0] + ang[24] * v[2], ang[17] * v[1] + ang[25] * v[3],
             ang[24] * v[0] - ang[16] * v[2], ang[25] * v[1] - ang[17] * v[3],
             ang[18] * v[4] + ang[26] * v[6], ang[19] * v[5] + ang[27] * v[7],
             ang[26] * v[4] - ang[18] * v[6], ang[27] * v[5] - ang[19] * v[7],
             ang[20] * v[8] + ang[28] * v[10], ang[21] * v[9] + ang[29] * v[11],
             ang[28] * v[8] - ang[20] * v[10], ang[29] * v[9] - ang[21] * v[11],
             ang[22] * v[12] + ang[30] * v[14], ang[23] * v[13] + ang[31] * v[15],
             ang[30] * v[12] - ang[22] * v[14], ang[31] * v[13] - ang[23] * v[15])
        v = (ang[32] * v[0] + ang[40] * v[4], ang[33] * v[1] + ang[41] * v[5],
             ang[34] * v[2] + ang[42] * v[6], ang[35] * v[3] + ang[43] * v[7],
             ang[40] * v[0] - ang[32] * v[4], ang[41] * v[1] - ang[33] * v[5],
             ang[42] * v[2] - ang[34] * v[6], ang[43] * v[3] - ang[35] * v[7],
             ang[36] * v[8] + ang[44] * v[12], ang[37] * v[9] + ang[45] * v[13],
             ang[38] * v[10] + ang[46] * v[14], ang[39] * v[11] + ang[47] * v[15],
             ang[44] * v[8] - ang[36] * v[12], ang[45] * v[9] - ang[37] * v[13],
             ang[46] * v[10] - ang[38] * v[14], ang[47] * v[11] - ang[39] * v[15])
        v = (ang[48] * v[0] + ang[56] * v[8], ang[49] * v[1] + ang[57] * v[9],
             ang[50] * v[2] + ang[58] * v[10], ang[51] * v[3] + ang[59] * v[11],
             ang[52] * v[4] + ang[60] * v[12], ang[53] * v[5] + ang[61] * v[13],
             ang[54] * v[6] + ang[62] * v[14], ang[55] * v[7] + ang[63] * v[15],
             ang[56] * v[0] - ang[48] * v[8], ang[57] * v[1] - ang[49] * v[9],
             ang[58] * v[2] - ang[50] * v[10], ang[59] * v[3] - ang[51] * v[11],
             ang[60] * v[4] - ang[52] * v[12], ang[61] * v[5] - ang[53] * v[13],
             ang[62] * v[6] - ang[54] * v[14], ang[63] * v[7] - ang[55] * v[15])
    return v


@triton.jit
def _amax16(v):
    """NaN-propagating max |v_j| over the 16 columns (a balanced tree; max is exact in any order)."""
    a0 = _nanmax(_nanmax(tl.abs(v[0]), tl.abs(v[1])), _nanmax(tl.abs(v[2]), tl.abs(v[3])))
    a1 = _nanmax(_nanmax(tl.abs(v[4]), tl.abs(v[5])), _nanmax(tl.abs(v[6]), tl.abs(v[7])))
    a2 = _nanmax(_nanmax(tl.abs(v[8]), tl.abs(v[9])), _nanmax(tl.abs(v[10]), tl.abs(v[11])))
    a3 = _nanmax(_nanmax(tl.abs(v[12]), tl.abs(v[13])), _nanmax(tl.abs(v[14]), tl.abs(v[15])))
    return _nanmax(_nanmax(a0, a1), _nanmax(a2, a3))


@triton.jit
def _signed_zero_if_nan(y):
    bits = y.to(tl.int32, bitcast=True)
    return tl.where(y == y, y, ((bits >> 31) << 31).to(tl.float32, bitcast=True))


@triton.jit
def _e2m1x2(lo, hi):
    """Packed E2M1 byte ``code(hi) << 4 | code(lo)`` (uint64), as ``nvfp4_inference_blackwell._e2m1x2``."""
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t; }", "=r,r,r",
        [_signed_zero_if_nan(lo), _signed_zero_if_nan(hi)], dtype=tl.uint32, is_pure=True, pack=1)
    return packed.to(tl.uint64)


@triton.jit
def _global_scale(amax):
    raw_gs = tl.div_rn(amax, 2688.0)  # amax / (6 * 448)
    return tl.where(raw_gs > 0, raw_gs, 1.0)  # 0, underflowing subnormal, or NaN -> 1


@triton.jit
def _amax_kernel(x_ptr, part_ptr, cos_ptr, sin_ptr, st_ptr, M, G, stride_m, stride_k, n_index, n_tiles, sg, sj,
                 ROT: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr, VEC: tl.constexpr,
                 STASH: tl.constexpr, PACKED: tl.constexpr):
    """Pass 1: partial[program] = NaN-propagating max |R x| over a strided set of tiles.

    A tile is ``BLOCK`` (strip, group) elements in row-major order; a strip is ``ROWS`` rows, and each
    thread walks the rows of its strip with the coefficients of its group in registers. ``STASH`` also
    stores the rotated FP32 values as 16 planes of M * G floats at ``st_ptr`` (exact bits), so pass 2 need not
    rotate again.
    """
    tl.extra.cuda.gdc_launch_dependents()  # pass 2 may start now (a no-op without PDL)
    pid = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for t in range(pid, n_tiles, tl.num_programs(0)):
        idx = t * BLOCK + tl.arange(0, BLOCK)
        strip = idx // G
        g = idx - strip * G
        inside = idx < n_index
        ang = _angles(cos_ptr, sin_ptr, g * sg, sj, inside, ROT, PACKED)
        for i in tl.static_range(ROWS):
            r = strip * ROWS + i
            valid = inside & (r < M)
            off = r.to(tl.int64) * stride_m + (g * 16).to(tl.int64) * stride_k
            v = _rotate16(_load16(x_ptr, off, valid, stride_k, VEC), ang, ROT)
            acc = _nanmax(acc, tl.reduce(_amax16(v), None, _nanmax))
            if STASH:  # 16 planes of M * G floats: plane j holds input j of every group (coalesced stores)
                _store_planes(st_ptr, r.to(tl.int64) * G + g, M * G, v, valid)
    tl.store(part_ptr + pid, acc)


@triton.jit
def _store_planes(st_ptr, e, NG, v, mask):
    for j in tl.static_range(16):
        tl.store(st_ptr + j * NG + e, v[j], mask=mask)


@triton.jit
def _load_planes(st_ptr, e, NG, mask):
    p = st_ptr + e
    return (tl.load(p, mask=mask, other=0.0), tl.load(p + NG, mask=mask, other=0.0),
            tl.load(p + 2 * NG, mask=mask, other=0.0), tl.load(p + 3 * NG, mask=mask, other=0.0),
            tl.load(p + 4 * NG, mask=mask, other=0.0), tl.load(p + 5 * NG, mask=mask, other=0.0),
            tl.load(p + 6 * NG, mask=mask, other=0.0), tl.load(p + 7 * NG, mask=mask, other=0.0),
            tl.load(p + 8 * NG, mask=mask, other=0.0), tl.load(p + 9 * NG, mask=mask, other=0.0),
            tl.load(p + 10 * NG, mask=mask, other=0.0), tl.load(p + 11 * NG, mask=mask, other=0.0),
            tl.load(p + 12 * NG, mask=mask, other=0.0), tl.load(p + 13 * NG, mask=mask, other=0.0),
            tl.load(p + 14 * NG, mask=mask, other=0.0), tl.load(p + 15 * NG, mask=mask, other=0.0))


@triton.jit
def _quant_kernel(x_ptr, q_ptr, s_ptr, gs_ptr, part_ptr, cos_ptr, sin_ptr, wg_ptr, alpha_ptr, M, G, pc,
                  stride_m, stride_k, pc4, n_part, n_index, sg, sj, ROT: tl.constexpr, DYNAMIC: tl.constexpr,
                  BLOCK: tl.constexpr, ROWS: tl.constexpr, PBLOCK: tl.constexpr, PDL: tl.constexpr,
                  VEC: tl.constexpr, Q64: tl.constexpr, ALPHA: tl.constexpr, PLANES: tl.constexpr,
                  PACKED: tl.constexpr):
    """Pass 2 (or the only pass with a static scale) over the padded (pr, pc) scale matrix.

    ``BLOCK`` (strip, column) elements per program, ``ROWS`` rows per strip (a divisor of 128, so strips
    tile the padded rows). Writes codes and block scales of every real group, the zero scales of the
    padding, the tensor scale (dynamic, program 0) and, with ``ALPHA``, the GEMM alpha ``gs * w_global``.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    strip = idx // pc
    g = idx - strip * pc
    inside = idx < n_index
    ang = _angles(cos_ptr, sin_ptr, g * sg, sj, inside & (g < G), ROT, PACKED)
    for i in tl.static_range(ROWS):
        r = strip * ROWS + i
        valid = inside & (r < M) & (g < G)
        if PLANES:  # the rotated stash of pass 1 (x_ptr), 16 planes of M * G floats; ROT is 0
            if PDL and i == 0:  # the stash is complete only after pass 1
                tl.extra.cuda.gdc_wait()
            v = _load_planes(x_ptr, r.to(tl.int64) * G + g, M * G, valid)
        else:
            off = r.to(tl.int64) * stride_m + (g * 16).to(tl.int64) * stride_k
            v = _rotate16(_load16(x_ptr, off, valid, stride_k, VEC), ang, ROT)
        bamax = _amax16(v)
        if i == 0:
            if DYNAMIC:
                if PDL:  # everything above only reads x and the tables; the partials are complete after the wait
                    tl.extra.cuda.gdc_wait()
                p = tl.arange(0, PBLOCK)
                gs = _global_scale(tl.reduce(tl.load(part_ptr + p, mask=p < n_part, other=0.0), None, _nanmax))
                if pid == 0:
                    tl.store(gs_ptr, gs)
            else:
                gs = tl.load(gs_ptr)
            if ALPHA:
                if pid == 0:
                    tl.store(alpha_ptr, gs * tl.load(wg_ptr))
        _store_group(v, bamax, gs, r, g, valid, inside, q_ptr, s_ptr, G, pc4, Q64)


@triton.jit
def _store_group(v, bamax, gs, r, g, valid, inside, q_ptr, s_ptr, G, pc4, Q64: tl.constexpr):
    """Block scale, E2M1 codes (one 64-bit word per group) and the swizzled scale byte of each group."""
    # E4M3 block scale: clamp(amax / (6 gs), max=448) -> e4m3 (round to nearest even).
    block = tl.minimum(tl.div_rn(bamax, 6.0 * gs), 448.0, propagate_nan=tl.PropagateNan.ALL)
    block8 = block.to(tl.float8e4nv)
    scale = block8.to(tl.float32) * gs
    safe = tl.where(scale > 0, scale, 1.0)
    word = _e2m1x2(tl.div_rn(v[0], safe), tl.div_rn(v[1], safe))
    word |= _e2m1x2(tl.div_rn(v[2], safe), tl.div_rn(v[3], safe)) << 8
    word |= _e2m1x2(tl.div_rn(v[4], safe), tl.div_rn(v[5], safe)) << 16
    word |= _e2m1x2(tl.div_rn(v[6], safe), tl.div_rn(v[7], safe)) << 24
    word |= _e2m1x2(tl.div_rn(v[8], safe), tl.div_rn(v[9], safe)) << 32
    word |= _e2m1x2(tl.div_rn(v[10], safe), tl.div_rn(v[11], safe)) << 40
    word |= _e2m1x2(tl.div_rn(v[12], safe), tl.div_rn(v[13], safe)) << 48
    word |= _e2m1x2(tl.div_rn(v[14], safe), tl.div_rn(v[15], safe)) << 56
    qoff = r.to(tl.int64) * G + g  # in 8-byte words: a packed row has K / 2 = 8 G bytes
    if Q64:
        tl.store(q_ptr.to(tl.pointer_type(tl.uint64), bitcast=True) + qoff, word, mask=valid)
    else:
        for i in tl.static_range(8):
            tl.store(q_ptr + qoff * 8 + i, ((word >> (8 * i)) & 0xFF).to(tl.uint8), mask=valid)
    # SWIZZLE_32_4_4: scale (r, c) -> ((r // 128) * pc4 + c // 4) * 512 + (r % 32) * 16 + (r % 128 // 32) * 4 + c % 4.
    bits = tl.where(valid, block8.to(tl.uint8, bitcast=True), 0).to(tl.float8e4nv, bitcast=True)
    ri = r % 128
    soff = ((r // 128).to(tl.int64) * pc4 + g // 4) * 512 + (ri % 32) * 16 + (ri // 32) * 4 + g % 4
    tl.store(s_ptr + soff, bits, mask=inside)


# ======================================================================================
# Launch
# ======================================================================================
def _vec_ok(x: Tensor) -> int:
    """The ``VEC`` load width of ``_load16``: 2 (128-bit) or 1 (64-bit) when the K stride is 1 and every
    group (the first one and every row) is 16- or 8-byte aligned; 0 (one load per input) otherwise."""
    if not _option("vector_io") or x.stride(1) != 1:
        return 0
    row = x.stride(0) * x.element_size()
    for width, align in ((2, 16), (1, 8)):
        if x.data_ptr() % align == 0 and (x.shape[0] == 1 or row % align == 0):
            return width
    return 0


def _public_tables(tables):
    """(cos, sin, sg, sj, packed) of the checked (G, 4, 8) tables of the public API (see ``_angles``)."""
    return None if tables is None else (*tables, 32, 1, False)


def packed_tables(cos: Tensor, sin: Tensor) -> Tensor:
    """(16, G, 4) FP32 ``(cos_2q, sin_2q, cos_2q+1, sin_2q+1)`` of (G, 4, 8) butterfly tables (coefficient
    j = 8 stage + pair): the coalesced 128-bit layout of ``FastNVFP4Linear``."""
    G = cos.shape[0]
    pairs = torch.stack((cos.reshape(G, 32), sin.reshape(G, 32)), -1)  # (G, 32, 2)
    return pairs.reshape(G, 16, 4).transpose(0, 1).contiguous()


def _stash(M: int, K: int, rotation: str) -> bool:
    """Dynamic butterfly: pass 1 stores R x so pass 2 does not repeat the 64-coefficient rotation."""
    return rotation == "adaptive_butterfly_b16" and M * K * 4 <= STASH_MAX_BYTES


def _pass1(M: int, K: int, rot: str) -> tuple:
    """(grid, n_index, n_tiles, BLOCK, ROWS, warps) of pass 1 (cached per shape and resolved tile)."""
    return _pass1_plan(M, K, tuple(_pass1_tile(M, K, rot)))


@functools.cache
def _pass1_plan(M: int, K: int, tile: tuple) -> tuple:
    block, rows, warps, most = tile
    n_index = -(-M // rows) * (K // 16)
    n_tiles = -(-n_index // block)
    programs = max(1, min(n_tiles, most, PASS1_PROGRAMS))
    return (programs, 1, 1), n_index, n_tiles, block, rows, warps


def _pass2(M: int, K: int, rot: str, dynamic: bool, n_part: int) -> tuple:
    """(grid, pc, pc4, n_index, BLOCK, ROWS, PBLOCK, warps) of pass 2 (cached per shape, setting and resolved tile).

    The grid covers the padded scale matrix: pr / ROWS strips (pr is a multiple of 128) times pc columns.
    """
    return _pass2_plan(M, K, tuple(_pass2_tile(M, K, rot)), dynamic, n_part)


@functools.cache
def _pass2_plan(M: int, K: int, tile: tuple, dynamic: bool, n_part: int) -> tuple:
    pr, pc = scale_layout(M, K)
    block, rows, warps = tile
    pblock = 1 << max(n_part - 1, 0).bit_length() if dynamic else 1  # next power of 2
    n_index = pr // rows * pc
    return (max(1, -(-n_index // block)), 1, 1), pc, pc // 4, n_index, block, rows, pblock, warps


def _n_partials(M: int, K: int, rot: str) -> int:
    """Pass-1 programs (= partials): one per tile, at most ``PASS1_PROGRAMS``."""
    return _pass1(M, K, rot)[0][0]


def _amax(x, M, K, workspace, rot, tables, stash: Tensor | None = None) -> int:
    grid, n_index, n_tiles, block, rows, warps = _pass1(M, K, rot)
    cos, sin, sg, sj, packed = tables if tables is not None else (workspace, workspace, 0, 0, False)
    vec = _vec_ok(x)
    nb._launch(_amax_kernel, grid, (x, workspace, cos, sin, workspace if stash is None else stash),
               (M, K // 16, x.stride(0), x.stride(1), n_index, n_tiles, sg, sj),
               (ROTATIONS[rot], block, rows, vec, stash is not None, packed), warps)
    return grid[0]


def _quant(x, M, K, packed_u8, scale, gs, workspace, n_part, rot, tables, pdl: bool = False,
           w_global: Tensor | None = None, alpha: Tensor | None = None, planes: bool = False) -> None:
    """Pass 2 of ``rot``'s plan; ``planes``: ``x`` is the rotated FP32 plane stash of pass 1 (no rotation)."""
    dynamic = workspace is not None
    pdl = pdl and dynamic and _option("pdl")
    grid, pc, pc4, n_index, block, rows, pblock, warps = _pass2(M, K, rot, dynamic, n_part if dynamic else 0)
    rot_id = 0 if planes else ROTATIONS[rot]
    ws = workspace if dynamic else gs
    cos, sin, sg, sj, packed = tables if tables is not None else (ws, ws, 0, 0, False)
    fused = alpha is not None
    vec = 0 if planes else _vec_ok(x)
    q64 = _option("vector_io") and packed_u8.data_ptr() % 8 == 0
    nb._launch(_quant_kernel, grid, (x, packed_u8, scale, gs, ws, cos, sin, w_global if fused else gs,
                                     alpha if fused else gs),
               (M, K // 16, pc, x.stride(0), x.stride(1), pc4, n_part, n_index, sg, sj),
               (rot_id, dynamic, block, rows, pblock, pdl, vec, q64, fused, planes, packed), warps, pdl)


def quantizer_plan(M: int, K: int, rotation: str, static: bool = False) -> dict:
    """The launches that ``quantize_activation`` makes for an (M, K) input (for result records)."""
    if delegated(M, K, rotation):
        return {"delegated_to": "nvfp4_inference_blackwell", **nb.quantizer_plan(M, K, rotation, static)}
    grid, pc, pc4, n_index, block, rows, pblock, warps = _pass2(M, K, rotation, not static,
                                                               0 if static else _n_partials(M, K, rotation))
    plan = {"pass2_tile": [block, rows, warps], "pass2_programs": grid[0]}
    if static:
        return {"launches": 1, **plan}
    g1, _, n_tiles, block1, rb1, warps1 = _pass1(M, K, rotation)
    return {"launches": 2, "pass1_tile": [block1, rb1, warps1], "pass1_programs": g1[0],
            "pass1_tiles": n_tiles, **plan, "pdl": _option("pdl"), "vector_io": _option("vector_io")}


def launch_amax(x: Tensor, workspace: Tensor, rotation: str = "none", tables=None) -> int:
    """Pass 1 of the dynamic mode into ``workspace``. Returns the number of partials ``n``.

    The contract of ``nvfp4_inference_blackwell.launch_amax`` with this module's partial count:
    ``workspace`` is a caller-owned 1-D contiguous ``torch.float32`` tensor on ``x.device`` with at least
    ``n`` elements (``n <= PASS1_PROGRAMS <= MAX_PARTIALS``). The kernel writes ``workspace[:n]`` only.
    """
    M, K, tables = nb._check(x, rotation, tables)
    nb._check_buffer(workspace, "workspace", x, (torch.float32,), flat=True, min_numel=_n_partials(M, K, rotation))
    return _amax(x, M, K, workspace, rotation, _public_tables(tables))


def launch_quant(x: Tensor, packed: Tensor, scale: Tensor, global_scale: Tensor, *, rotation: str = "none",
                 tables=None, workspace: Tensor | None = None, n_part: int = 0) -> None:
    """Pass 2 (dynamic) or the static single pass, with the buffer and exact ``n_part`` contract of
    ``nvfp4_inference_blackwell.launch_quant`` (``n_part`` must equal ``launch_amax``'s count of this
    module; any other count raises ``ValueError``). Launches without PDL."""
    M, K, tables = nb._check(x, rotation, tables)
    pr, pc = scale_layout(M, K)
    nb._check_buffer(packed, "packed", x, (torch.uint8, torch.float4_e2m1fn_x2), shape=(M, K // 2))
    nb._check_buffer(scale, "scale", x, (torch.uint8, torch.float8_e4m3fn), numel=pr * pc)
    nb._check_buffer(global_scale, "global_scale", x, (torch.float32,), numel=1)
    if type(n_part) is not int:
        raise ValueError(f"n_part must be an int, not {type(n_part).__name__}")
    if workspace is not None:
        expected_n = _n_partials(M, K, rotation)
        nb._check_buffer(workspace, "workspace", x, (torch.float32,), flat=True, min_numel=expected_n)
        if n_part != expected_n:
            raise ValueError(
                f"dynamic launch_quant needs n_part == {expected_n} "
                f"(the count returned by launch_amax for ({M}, {K}, {rotation!r})), not {n_part}")
    elif n_part != 0:
        raise ValueError(f"static launch_quant (no workspace) takes n_part = 0, not {n_part}")
    _quant(x, M, K, packed.view(torch.uint8), scale.view(torch.float8_e4m3fn), global_scale, workspace, n_part,
           rotation, _public_tables(tables))


class _Quantized(NamedTuple):
    act: NVFP4Activation
    alpha: Tensor | None  # () float32 ``global_scale * w_global`` when requested


def _quantize(x: Tensor, M: int, K: int, rotation: str, tables, global_scale: Tensor | None,
              w_global: Tensor | None = None) -> _Quantized:
    pr, pc = scale_layout(M, K)
    packed = torch.empty((M, K // 2), dtype=torch.float4_e2m1fn_x2, device=x.device)
    scale = torch.empty(pr * pc, dtype=torch.float8_e4m3fn, device=x.device)
    packed_u8 = packed.view(torch.uint8)
    if global_scale is not None:
        alpha = torch.empty((), dtype=torch.float32, device=x.device) if w_global is not None else None
        _quant(x, M, K, packed_u8, scale, global_scale, None, 0, rotation, tables, w_global=w_global, alpha=alpha)
        return _Quantized(NVFP4Activation(packed, scale, global_scale), alpha)
    n = _n_partials(M, K, rotation)
    workspace = torch.empty(n + 2, dtype=torch.float32, device=x.device)  # partials, tensor scale, alpha
    gs = workspace[n]
    alpha = workspace[n + 1] if w_global is not None else None
    if _option("stash") and _stash(M, K, rotation):
        # Pass 1 also writes R x (FP32 planes); pass 2 quantizes them without rotating. Pass 2 reads what
        # pass 1 writes, so under PDL it waits before its first load (only its launch overlaps pass 1).
        stash = torch.empty((16 * M, K // 16), dtype=torch.float32, device=x.device)  # 16 planes (M, G)
        _amax(x, M, K, workspace, rotation, tables, stash)
        _quant(stash, M, K, packed_u8, scale, gs, workspace, n, rotation, None, pdl=True, w_global=w_global,
               alpha=alpha, planes=True)
        return _Quantized(NVFP4Activation(packed, scale, gs), alpha)
    _amax(x, M, K, workspace, rotation, tables)
    # Pass 1 is the launch right before pass 2 and only reads x, so pass 2 may overlap it (PDL).
    _quant(x, M, K, packed_u8, scale, gs, workspace, n, rotation, tables, pdl=True, w_global=w_global, alpha=alpha)
    return _Quantized(NVFP4Activation(packed, scale, gs), alpha)


def quantize_activation(x: Tensor, rotation: str = "none", tables=None,
                        global_scale: Tensor | None = None) -> NVFP4Activation:
    """``F.scaled_mm``-ready NVFP4 operands of ``x`` (M, K), bit-identical to ``nb.quantize_activation``."""
    M, K, tables = nb._check(x, rotation, tables)
    if global_scale is not None:
        global_scale = global_scale.reshape(())
        if global_scale.dtype != torch.float32 or global_scale.device != x.device:
            raise ValueError("A static global scale must be a float32 tensor on the device of x")
    if delegated(M, K, rotation):
        return nb._quantize(x, M, K, rotation, tables, global_scale)
    return _quantize(x, M, K, rotation, _public_tables(tables), global_scale).act


def delegated(M: int, K: int, rotation: str) -> bool:
    """True when an (M, K) activation of ``rotation`` goes through the C5-03 quantizer and ``torch._scaled_mm_v2``."""
    limit = DELEGATE_FROM[rotation]
    return _option("delegate_large") and limit is not None and M * K >= limit and K >= DELEGATE_MIN_K[rotation]


# ======================================================================================
# cuBLASLt GEMM with a device alpha (no elementwise scale product)
# ======================================================================================
_LT_LIB = None
_LT_CACHE_LOCK = threading.RLock()  # guards the three process-wide caches below
_LT_HANDLES: dict = {}
_LT_PLANS: collections.OrderedDict = collections.OrderedDict()
_LT_ZERO: dict = {}
LT_PLAN_CACHE_MAX = 256  # cached (M, N, K, out dtype, device) plans; the least recently used is dropped beyond it
LT_WORKSPACE_LIMIT = 1 << 20  # the limit PyTorch's scaled_mm passes to the heuristic (selects the same algorithm)
# cuBLASLt enums (cublasLt.h / library_types.h of CUDA 13)
_COMPUTE_32F, _R_32F, _R_16F, _R_16BF, _R_4F_E2M1 = 68, 0, 2, 14, 33
_DESC_POINTER_MODE, _DESC_TRANSA, _DESC_TRANSB = 2, 3, 4
_DESC_A_SCALE_POINTER, _DESC_B_SCALE_POINTER, _DESC_FAST_ACCUM = 17, 18, 25
_DESC_A_SCALE_MODE, _DESC_B_SCALE_MODE = 31, 32
_SCALE_VEC16_UE4M3, _POINTER_MODE_DEVICE, _OP_N, _OP_T = 1, 1, 0, 1
_PREF_MAX_WORKSPACE_BYTES = 1
_ALGO_CONFIG = {"algo_id": 0, "tile_id": 1, "splitk_num": 2, "reduction_scheme": 3, "stages_id": 6,
                "cluster_shape_id": 8}
_OUT_TYPES = {torch.bfloat16: _R_16BF, torch.float16: _R_16F, torch.float32: _R_32F}


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _Heuristic(ctypes.Structure):
    _fields_ = [("algo", _Algo), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]


def _lt():
    """The cuBLASLt library that PyTorch loaded (``nvidia/cu13/lib``), with the argument types we call."""
    global _LT_LIB
    with _LT_CACHE_LOCK:
        if _LT_LIB is None:
            import nvidia  # the CUDA wheels of this PyTorch build
            lib = ctypes.CDLL(str(Path(next(iter(nvidia.__path__))) / "cu13" / "lib" / "libcublasLt.so.13"))
            vp = ctypes.c_void_p
            lib.cublasLtMatmul.argtypes = [vp] * 14 + [ctypes.c_size_t, vp]
            lib.cublasLtMatmul.restype = ctypes.c_int
            lib.cublasLtMatmulDescSetAttribute.argtypes = [vp, ctypes.c_int, vp, ctypes.c_size_t]
            lib.cublasLtMatmulDescSetAttribute.restype = ctypes.c_int
            for name in ("cublasLtMatmulDescDestroy", "cublasLtMatrixLayoutDestroy", "cublasLtMatmulPreferenceDestroy"):
                getattr(lib, name).argtypes = [vp]
                getattr(lib, name).restype = ctypes.c_int
            _LT_LIB = lib
        return _LT_LIB


def _ck(status: int, what: str) -> None:
    if status != 0:
        raise RuntimeError(f"cuBLASLt {what} failed with status {status}")


def _set(desc, attr: int, value, ctype) -> None:
    v = ctype(value)
    _ck(_lt().cublasLtMatmulDescSetAttribute(desc, attr, ctypes.byref(v), ctypes.sizeof(v)), f"attribute {attr}")


class LtPlan:
    """Matmul descriptor, layouts and algorithm of one (M, N, K, out dtype) NVFP4 GEMM ``out = alpha x W^T``.

    The descriptor is PyTorch's (``CUBLASLT_LOG_LEVEL=5`` trace of ``torch._scaled_mm_v2``): FP32 compute
    and scale, ``A = W`` (K x N, transposed), ``B = x`` (K x M), VEC16_UE4M3 block scales, device pointer
    mode, no fast accumulation, and the first heuristic result under a 1 MiB workspace limit.

    Thread safety: the two block-scale pointers live in the shared descriptor, so ``__call__`` sets them and
    submits ``cublasLtMatmul`` under one per-plan lock. The lock covers host submission only; the GEMM still
    runs asynchronously on the caller's current stream, and a CUDA graph replay does not touch it.
    Lifetime: ``close`` (idempotent) destroys the descriptor and layouts; a dropped plan closes itself.
    """

    def __init__(self, M: int, N: int, K: int, out_dtype: torch.dtype, device: torch.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up the NVFP4 cuBLASLt GEMM eagerly once per shape before CUDA-graph capture")
        lt = _lt()
        self._lock = threading.Lock()
        self.M, self.N, self.K, self.out_dtype = M, N, K, out_dtype
        self.device_index = _device_index(device)
        self.desc, self.layouts = None, []
        pref = None
        try:
            self.handle = _handle(self.device_index)
            desc = ctypes.c_void_p()
            _ck(lt.cublasLtMatmulDescCreate(ctypes.byref(desc), _COMPUTE_32F, _R_32F), "MatmulDescCreate")
            self.desc = desc
            _set(desc, _DESC_TRANSA, _OP_T, ctypes.c_int32)
            _set(desc, _DESC_TRANSB, _OP_N, ctypes.c_int32)
            _set(desc, _DESC_FAST_ACCUM, 0, ctypes.c_int8)
            _set(desc, _DESC_POINTER_MODE, _POINTER_MODE_DEVICE, ctypes.c_int32)
            _set(desc, _DESC_A_SCALE_MODE, _SCALE_VEC16_UE4M3, ctypes.c_int32)
            _set(desc, _DESC_B_SCALE_MODE, _SCALE_VEC16_UE4M3, ctypes.c_int32)
            probe = torch.empty(256, dtype=torch.uint8, device=_cuda(self.device_index))  # non-null scale pointers
            _set(desc, _DESC_A_SCALE_POINTER, probe.data_ptr(), ctypes.c_void_p)
            _set(desc, _DESC_B_SCALE_POINTER, probe.data_ptr(), ctypes.c_void_p)
            out = _OUT_TYPES[out_dtype]
            for dtype, rows, cols in ((_R_4F_E2M1, K, N), (_R_4F_E2M1, K, M), (out, N, M), (out, N, M)):
                layout = ctypes.c_void_p()
                _ck(lt.cublasLtMatrixLayoutCreate(ctypes.byref(layout), dtype, ctypes.c_uint64(rows),
                                                  ctypes.c_uint64(cols), ctypes.c_int64(rows)), "MatrixLayoutCreate")
                self.layouts.append(layout)
            pref = ctypes.c_void_p()
            _ck(lt.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "MatmulPreferenceCreate")
            limit = ctypes.c_uint64(LT_WORKSPACE_LIMIT)
            _ck(lt.cublasLtMatmulPreferenceSetAttribute(pref, _PREF_MAX_WORKSPACE_BYTES, ctypes.byref(limit), 8),
                "MatmulPreferenceSetAttribute")
            self.heuristic = _Heuristic()
            found = ctypes.c_int()
            _ck(lt.cublasLtMatmulAlgoGetHeuristic(self.handle, desc, *self.layouts, pref, 1,
                                                  ctypes.byref(self.heuristic), ctypes.byref(found)), "AlgoGetHeuristic")
            if found.value < 1:
                raise RuntimeError(f"cuBLASLt has no NVFP4 algorithm for {(M, N, K)}")
        except BaseException:
            self.close()
            raise
        finally:
            if pref is not None and pref.value:
                lt.cublasLtMatmulPreferenceDestroy(pref)
        self.workspace_bytes = int(self.heuristic.workspaceSize)
        self.algo = ctypes.byref(self.heuristic.algo)
        self.config = {}
        for name, attr in _ALGO_CONFIG.items():
            value, written = ctypes.c_int32(), ctypes.c_size_t()
            lt.cublasLtMatmulAlgoConfigGetAttribute(self.algo, attr, ctypes.byref(value), 4, ctypes.byref(written))
            self.config[name] = value.value
        self.zero = _zero(self.device_index)

    @property
    def direct(self) -> bool:
        """The algorithm needs no workspace (every one seen on sm_120). ``FastNVFP4Linear`` falls back to
        ``_scaled_mm_v2`` otherwise; ``nvfp4_scaled_mm_lt`` allocates the workspace."""
        return self.workspace_bytes == 0

    @property
    def closed(self) -> bool:
        return self.desc is None

    def record(self) -> dict:
        return {"mnk": [self.M, self.N, self.K], "out_dtype": str(self.out_dtype), "workspace_bytes": self.workspace_bytes,
                "algo": dict(self.config), "direct": self.direct}

    def __call__(self, a: NVFP4Activation, w_packed: Tensor, w_scale: Tensor, alpha: Tensor, out: Tensor,
                 workspace: Tensor | None = None) -> Tensor:
        """Submit ``out = alpha a W^T`` on the current stream of ``out.device``.

        A plan with ``workspace_bytes > 0`` needs a contiguous CUDA ``uint8`` ``workspace`` of at least that
        many bytes on that device (kept alive by the caller until the GEMM completes); ``ValueError`` otherwise.
        """
        size = 0
        if self.workspace_bytes > 0:
            if (not isinstance(workspace, Tensor) or not workspace.is_cuda or workspace.dtype != torch.uint8
                    or not workspace.is_contiguous() or workspace.device != out.device
                    or workspace.nbytes < self.workspace_bytes):
                raise ValueError(f"this cuBLASLt plan needs a contiguous CUDA uint8 workspace of at least "
                                 f"{self.workspace_bytes} bytes on {out.device}")
            size = self.workspace_bytes
        lt = _lt()
        stream = torch.cuda.current_stream(out.device).cuda_stream
        with self._lock:  # the scale pointers and the submission that reads them are one host section
            if self.desc is None:
                raise RuntimeError("this cuBLASLt plan is closed")
            _set(self.desc, _DESC_A_SCALE_POINTER, w_scale.data_ptr(), ctypes.c_void_p)
            _set(self.desc, _DESC_B_SCALE_POINTER, a.scale.data_ptr(), ctypes.c_void_p)
            _ck(lt.cublasLtMatmul(self.handle, self.desc, alpha.data_ptr(), w_packed.data_ptr(), self.layouts[0],
                                  a.packed.data_ptr(), self.layouts[1], self.zero.data_ptr(), out.data_ptr(),
                                  self.layouts[2], out.data_ptr(), self.layouts[3], self.algo,
                                  workspace.data_ptr() if size else None, size, stream), "Matmul")
        return out

    def close(self) -> None:
        """Destroy the descriptor and layouts (idempotent; a later call raises). CUDA graphs captured
        earlier keep working: the descriptor is read at submission only."""
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            lt = _LT_LIB
            if lt is not None:
                for layout in self.layouts:
                    lt.cublasLtMatrixLayoutDestroy(layout)
                if self.desc is not None:
                    lt.cublasLtMatmulDescDestroy(self.desc)
            self.layouts, self.desc = [], None

    def __del__(self):
        try:
            self.close()
        except Exception:  # interpreter shutdown
            pass


def _device_index(device) -> int:
    """The CUDA device index of ``device`` (an index-less ``cuda`` is the current device)."""
    device = torch.device(device) if not isinstance(device, int) else torch.device("cuda", device)
    if device.type != "cuda":
        raise ValueError(f"the NVFP4 cuBLASLt GEMM needs a CUDA device, not {device}")
    return torch.cuda.current_device() if device.index is None else device.index


def _cuda(index: int) -> torch.device:
    return torch.device("cuda", index)


def _handle(index: int):
    """The per-device cuBLASLt handle (one per device for the process lifetime; holds no per-call state)."""
    with _LT_CACHE_LOCK:
        handle = _LT_HANDLES.get(index)
        if handle is None:
            handle = ctypes.c_void_p()
            with torch.cuda.device(index):
                _ck(_lt().cublasLtCreate(ctypes.byref(handle)), "Create")
            _LT_HANDLES[index] = handle
        return handle


def _zero(index: int) -> Tensor:
    with _LT_CACHE_LOCK:
        if index not in _LT_ZERO:
            _LT_ZERO[index] = torch.zeros((), dtype=torch.float32, device=_cuda(index))  # beta (device pointer mode)
        return _LT_ZERO[index]


def lt_plan(M: int, N: int, K: int, out_dtype: torch.dtype, device) -> LtPlan:
    """The cached plan of (M, N, K, out dtype, device index): an LRU of at most ``LT_PLAN_CACHE_MAX`` plans.

    An evicted plan is not closed here (a caller may still hold it); it closes when its last reference goes.
    """
    key = (M, N, K, out_dtype, _device_index(device))
    with _LT_CACHE_LOCK:
        plan = _LT_PLANS.get(key)
        if plan is not None:
            _LT_PLANS.move_to_end(key)
            return plan
        plan = _LT_PLANS[key] = LtPlan(M, N, K, out_dtype, _cuda(key[-1]))
        while len(_LT_PLANS) > LT_PLAN_CACHE_MAX:
            _LT_PLANS.popitem(last=False)
        return plan


def clear_lt_plan_cache() -> int:
    """Close and drop every cached plan (teardown; a closed plan refuses later calls). Returns the count."""
    with _LT_CACHE_LOCK:
        plans = list(_LT_PLANS.values())
        _LT_PLANS.clear()
    for plan in plans:
        plan.close()
    return len(plans)


def nvfp4_scaled_mm_lt(a: NVFP4Activation, w_packed: Tensor, w_scale: Tensor, alpha: Tensor,
                       out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """``a @ W^T`` through cuBLASLt with the device scalar ``alpha = a.global_scale * w_global``.

    Equals ``nb.nvfp4_scaled_mm_public(a, w_packed, w_scale, w_global, out_dtype)`` bit for bit when
    ``alpha`` holds that FP32 product. A plan that needs a workspace gets one from the caching allocator
    (stream-ordered, so it stays valid until the GEMM on this stream completes).
    """
    M, N, K = a.packed.shape[0], w_packed.shape[0], a.packed.shape[1] * 2
    device = a.packed.device
    out = torch.empty((M, N), dtype=out_dtype, device=device)
    plan = lt_plan(M, N, K, out_dtype, device)
    workspace = None if plan.direct else torch.empty(plan.workspace_bytes, dtype=torch.uint8, device=device)
    return plan(a, w_packed, w_scale, alpha, out, workspace)


# ======================================================================================
# Module
# ======================================================================================
class FastNVFP4Linear(nb.NVFP4Linear):
    """``nb.NVFP4Linear`` (same state dict and outputs) on the C6-02 quantizer and GEMM.

    A butterfly layer also keeps its coefficients in the (16, G, 4) layout of ``packed_tables`` (a
    non-persistent buffer derived from ``cos`` and ``sin``). ``load_state_dict`` rebuilds it after the load,
    in place when the shape and device are unchanged (so earlier CUDA-graph captures read the new values).
    """

    _FP32_BUFFERS = (*nb.NVFP4Linear._FP32_BUFFERS, "cos_sin")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_transposed()

    def _add_transposed(self) -> None:
        if self.rotation != "adaptive_butterfly_b16":
            return
        packed = packed_tables(self.cos, self.sin)
        current = self._buffers.get("cos_sin")
        if current is not None and current.shape == packed.shape and current.device == packed.device:
            with torch.no_grad():
                current.copy_(packed)
        else:
            self.register_buffer("cos_sin", packed, persistent=False)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._add_transposed()  # cos_sin is not in the state dict: derive it from the loaded tables

    @property
    def _fast_tables(self):
        if self.rotation != "adaptive_butterfly_b16":
            return None
        return self.cos_sin, self.cos_sin, 1, self.in_features // 16, True

    @classmethod
    def from_layer(cls, layer: nb.NVFP4Linear) -> "FastNVFP4Linear":
        """A fast layer that shares the buffers of an existing ``NVFP4Linear``."""
        fast = cls.__new__(cls)
        nn.Module.__init__(fast)
        fast.in_features, fast.out_features = layer.in_features, layer.out_features
        fast.rotation, fast.out_dtype = layer.rotation, layer.out_dtype
        for name, buf in layer._buffers.items():
            fast.register_buffer(name, buf)
        fast._add_transposed()
        return fast

    def quantize_input(self, x2: Tensor) -> NVFP4Activation:
        return quantize_activation(x2, self.rotation, self.tables(), self.act_global_scale)

    def forward(self, x: Tensor) -> Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        M = x2.shape[0]
        if M == 0:
            return x.new_empty(*lead, self.out_features, dtype=self.out_dtype)
        if not x2.is_cuda or x2.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("NVFP4Linear needs a CUDA float32, bfloat16 or float16 input")
        if delegated(M, self.in_features, self.rotation):
            return nb.NVFP4Linear.forward(self, x)
        plan = lt_plan(M, self.out_features, self.in_features, self.out_dtype, x2.device) \
            if _option("lt_gemm") and self.out_dtype in _OUT_TYPES else None
        fused = plan is not None and plan.direct
        a, alpha = _quantize(x2, M, self.in_features, self.rotation, self._fast_tables, self.act_global_scale,
                             self.w_global if fused else None)
        if fused:
            out = plan(a, self.w_packed, self.w_scale, alpha,
                       torch.empty((M, self.out_features), dtype=self.out_dtype, device=x2.device))
        else:
            out = nb.nvfp4_scaled_mm(a, self.w_packed, self.w_scale, self.w_global, self.out_dtype)
        return out.reshape(*lead, self.out_features)


# ======================================================================================
# Fused cross-block mix (the PyTorch part of the C6-01 wide rotations)
# ======================================================================================
@triton.jit
def _group_hadamard(xrow, h_ptr, a, t, j, rmask, stride_k, GH: tl.constexpr, BM: tl.constexpr):
    """z[a, t] = sum_c H[a, c] x[j, c, t] of block j: two FMA chains over the halves of c, then one add."""
    B: tl.constexpr = GH * 16
    if GH == 1:
        return tl.load(xrow + (j * B + t) * stride_k, mask=rmask, other=0.0).to(tl.float32)[:, None]
    HG: tl.constexpr = (GH + 1) // 2
    z0 = tl.full((BM * 16, GH), -0.0, tl.float32)
    z1 = tl.full((BM * 16, GH), -0.0, tl.float32)
    for c in tl.static_range(GH):
        xc = tl.load(xrow + (j * B + c * 16 + t) * stride_k, mask=rmask, other=0.0).to(tl.float32)
        hc = tl.load(h_ptr + a * GH + c)
        if c < HG:
            z0 = tl.fma(hc[None, :], xc[:, None], z0)
        else:
            z1 = tl.fma(hc[None, :], xc[:, None], z1)
    return z0 + z1


@triton.jit
def _cross_mix_kernel(x_ptr, h_ptr, m_ptr, o_ptr, M, stride_m, stride_k, MIXN: tl.constexpr, MP: tl.constexpr,
                      GH: tl.constexpr, HAS_MIX: tl.constexpr, BM: tl.constexpr):
    """FP32 ``CrossMix.forward(x.float())`` of ``BM`` rows: lanes are (row, position t in the group).

    Stage 1 mixes the GH groups of each block with ``H_GH``; stage 2 (``HAS_MIX``) mixes the MIXN blocks
    with ``mix``, MP output blocks per program (axis 1 of the grid). Each contraction is two FMA chains (first and second half of the summed index, each
    started from -0.0 so a lone product keeps its sign) added at the end: the order of the cuBLAS
    ``gemmSN`` kernels that ``torch.matmul`` uses for these small products on the RTX 5080 (measured;
    cuBLAS picks other kernels for some shapes, so parity is checked, not assumed).
    """
    pid = tl.program_id(0)
    lane = tl.arange(0, BM * 16)
    r = pid * BM + lane // 16
    t = lane % 16
    rmask = r < M
    a = tl.arange(0, GH)
    B: tl.constexpr = GH * 16
    K: tl.constexpr = MIXN * B
    xrow = x_ptr + r.to(tl.int64) * stride_m
    orow = o_ptr + r.to(tl.int64) * K
    if HAS_MIX:  # output blocks i of this program: a chunk of MP (stage 1 is recomputed per chunk)
        HM: tl.constexpr = (MIXN + 1) // 2
        i = tl.program_id(1) * MP + tl.arange(0, MP)
        out0 = tl.full((BM * 16, MP, GH), -0.0, tl.float32)
        out1 = tl.full((BM * 16, MP, GH), -0.0, tl.float32)
        for j in tl.static_range(MIXN):
            z = _group_hadamard(xrow, h_ptr, a, t, j, rmask, stride_k, GH, BM)
            mj = tl.load(m_ptr + i * MIXN + j, mask=i < MIXN, other=0.0)
            if j < HM:
                out0 = tl.fma(mj[None, :, None], z[:, None, :], out0)
            else:
                out1 = tl.fma(mj[None, :, None], z[:, None, :], out1)
        tl.store(orow[:, None, None] + i[None, :, None] * B + a[None, None, :] * 16 + t[:, None, None], out0 + out1,
                 mask=rmask[:, None, None] & (i < MIXN)[None, :, None])
    else:
        for j in tl.static_range(MIXN):
            z = _group_hadamard(xrow, h_ptr, a, t, j, rmask, stride_k, GH, BM)
            tl.store(orow[:, None] + j * B + a[None, :] * 16 + t[:, None], z, mask=rmask[:, None])


CROSS_TILE = (2, 2)  # (rows per program, warps): swept on the RTX 5080 at K = 512 .. 2048
CROSS_MAX_OUT_BLOCKS = 16  # output blocks per program of the block mix (registers: 2 x 32 x 16 x 8 per program)


class FastCrossInput(nn.Module):
    """``rotated_qat.CrossInput`` with the cross-block mix in one Triton kernel (BF16/FP16/FP32 input, FP32 out).

    ``forward(x) = layer(C x)`` like ``CrossInput``. The fused mix reads the input directly (no ``x.float()``
    copy) and replaces the two batched cuBLAS products. Its FP32 output equals ``cross.forward(x.float())``
    bit for bit only where cuBLAS uses the two-half ``gemmSN`` order (every b64 shape measured, the full mix
    at M = 1024, K <= 2048); elsewhere single values differ by FP32 rounding (about 1e-6 at K = 5632).
    Such a difference can move a value across an FP4 rounding threshold, and later layers amplify the
    changed codes: the full mix of the w2048_6l model differs from ``CrossInput`` by 0.21-0.25 relative L2
    in its logits. The fused mode is therefore a latency path that must be qualified per shape.
    ``exact=True`` runs ``cross.forward(x.float())`` instead (bit-identical to ``CrossInput`` at any shape).

    ``h`` and ``mix`` stay FP32 under module dtype conversion (``.bfloat16()``, ``.half()``, ``.to(dtype)``)
    and follow device moves.
    """

    _FP32_BUFFERS = ("h", "mix")

    def __init__(self, layer: nn.Module, cross, exact: bool = False):
        super().__init__()
        if cross.identity:
            raise ValueError("FastCrossInput needs a mixing CrossMix (block > 16 or a block mix)")
        self.layer, self.cross, self.exact = layer, cross, bool(exact)
        self.n, self.groups, self.blocks = cross.n, cross.block // 16, cross.n // cross.block
        # ``rotated_qat.hadamard_matrix``: the FP64 Sylvester matrix of ``fwht``, rounded to FP32 as ``CrossMix._matrix``
        device = next(iter(layer.buffers()), torch.empty(0)).device  # the mix matrices follow the layer
        h = q.fwht(torch.eye(self.groups, dtype=torch.float64), self.groups)
        self.register_buffer("h", h.float().contiguous().to(device), persistent=False)
        self.register_buffer("mix", None if cross._mix is None else cross._mix.float().contiguous().to(device),
                             persistent=False)

    def _apply(self, fn, recurse=True):
        # Like ``nb.NVFP4Linear``: hold the FP32 matrices out of ``fn`` and only follow its device.
        held = {n: self._buffers.pop(n) for n in self._FP32_BUFFERS if self._buffers.get(n) is not None}
        try:
            super()._apply(fn, recurse)
        finally:
            for name, value in held.items():
                self._buffers[name] = value.to(fn(value).device)
        return self

    def mixed(self, x: Tensor) -> Tensor:
        """FP32 ``C x`` of ``x`` (..., n) as a contiguous (rows, n) tensor."""
        x2 = x.reshape(-1, self.n)
        if not x2.is_cuda or x2.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("FastCrossInput needs a CUDA float32, bfloat16 or float16 input")
        M = x2.shape[0]
        if self.exact:
            with torch.autocast(device_type="cuda", enabled=False):
                return self.cross.forward(x2.float()).reshape(M, self.n).contiguous()
        if self.h.dtype != torch.float32 or (self.mix is not None and self.mix.dtype != torch.float32):
            raise RuntimeError("FastCrossInput mix matrices must be float32")
        out = torch.empty((M, self.n), dtype=torch.float32, device=x2.device)
        if M == 0:
            return out
        rows, warps = CROSS_TILE
        mix = self.mix if self.mix is not None else self.h
        chunk = min(1 << max(self.blocks - 1, 0).bit_length(), CROSS_MAX_OUT_BLOCKS)
        chunks = -(-self.blocks // chunk) if self.mix is not None else 1
        nb._launch(_cross_mix_kernel, (-(-M // rows), chunks, 1), (x2, self.h, mix, out), (M, x2.stride(0), x2.stride(1)),
                   (self.blocks, chunk, self.groups, self.mix is not None, rows), warps)
        return out

    def forward(self, x: Tensor) -> Tensor:
        if self.exact:  # the expression of ``rotated_qat.CrossInput.forward``
            with torch.autocast(device_type=x.device.type, enabled=False):
                return self.layer(self.cross.forward(x.float()))
        return self.layer(self.mixed(x).reshape(*x.shape[:-1], self.n))


__all__ = ["FastNVFP4Linear", "FastCrossInput", "CROSS_TILE", "quantize_activation", "launch_amax", "launch_quant", "quantizer_plan",
           "nvfp4_scaled_mm_lt", "lt_plan", "clear_lt_plan_cache", "packed_tables", "LtPlan", "ROTATIONS", "MAX_PARTIALS",
           "PASS1_PROGRAMS", "OPTIONS", "options_scope", "TILES", "delegated", "DELEGATE_FROM", "DELEGATE_MIN_K",
           "LT_WORKSPACE_LIMIT", "LT_PLAN_CACHE_MAX"]
