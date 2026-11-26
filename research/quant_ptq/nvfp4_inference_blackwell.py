"""Fused NVFP4 activation quantization and W4A4 linear layers for Blackwell (sm_120, Cycle 4 C4-02).

One Triton kernel turns BF16/FP16/FP32 activations ``x`` (M, K) into the three operands that
``F.scaled_mm`` needs for an NVFP4 GEMM:

* ``packed``  (M, K // 2) ``torch.float4_e2m1fn_x2``. Element 2i is in the low nibble.
* ``scale``   flat ``float8_e4m3fn`` block scales (one per 16 inputs), written directly in the cuBLAS
              ``SWIZZLE_32_4_4`` layout of ``micro_rotated_ptq.to_blocked``. The kernel also writes
              the zero slots of the padding (rows up to ``ceil(M / 128) * 128``, scale columns up to
              ``ceil(K / 64) * 4``), so no memset and no separate swizzle pass run.
* ``global``  FP32 tensor scale ``amax / (6 * 448)`` (1 when ``amax`` is 0 or NaN).

Optionally the kernel first applies a block-16 rotation in registers (``hadamard_b16`` or
``adaptive_butterfly_b16`` with per-block angles (K // 16, 4, 8)). The rotated activation is never
written to memory.

Global scale modes
    dynamic  Two launches. Pass 1 writes one NaN-propagating amax per program (no atomics, so the
             result is deterministic). Pass 2 reduces these partials in every program (at most
             ``MAX_PARTIALS`` values, read from L2), then quantizes, packs and swizzles.
    static   One launch. The caller gives a pre-calibrated FP32 ``global_scale`` tensor. Values
             above the calibrated range saturate (block scale 448, element 6).

Numerics: the kernel matches ``micro_rotated_ptq.nvfp4_encode`` (plus ``pack_e2m1`` and
``to_blocked``) bit for bit, including NaN and Inf inputs and rotations. To get this:
* divisions use ``tl.div_rn`` (IEEE), as the PyTorch tensor divisions of the reference do;
* the kernels compile with ``enable_fp_fusion=False``, so ``c * a + s * d`` of the butterfly is
  two rounded products and one rounded sum, as in eager PyTorch (no FMA contraction);
* E2M1 rounding uses the same seven midpoint comparisons as ``e2m1_codes`` (ties to the even
  code). The PTX ``cvt.rn.satfinite.e2m1x2.f32`` instruction is not used: it has the same
  round-to-nearest-even rule for finite values but maps NaN differently;
* the E4M3 block scale uses Triton's FP32 -> ``float8e4nv`` conversion (``cvt.rn.satfinite``)
  after a NaN-propagating clamp to 448. That is the reference ``clamp(max=448).to(e4m3fn)``.

Launch overhead: after the first call for a specialization, kernels go through Triton's C
launcher directly (``polar_switch_blackwell._c_launcher``). Every launch, buffer and GEMM runs on
the current stream without a host synchronization, so ``NVFP4Linear`` is CUDA-graph capturable.
Warm up each shape eagerly once before capture (the first call compiles).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor, nn
from triton import knobs
from triton.runtime import driver

from research.polar_opt.polar_switch_blackwell import _c_launcher
from research.quant_ptq import micro_rotated_ptq as q

ROTATIONS = {"none": 0, "hadamard_b16": 1, "adaptive_butterfly_b16": 2}
MAX_PARTIALS = 512  # pass-1 programs; pass 2 reduces this many values per program


def _tile(M: int, rotation: str) -> tuple[int, int, int]:
    """(rows, 16-input groups, warps) per program. Rows must divide 128 (the scale-row padding).

    From a sweep on the RTX 5080 with a flushed L2 (M in 16..8192, K in 512..8192): narrow
    8-row tiles suit the add-only rotations; the butterfly, with its per-group angle loads,
    prefers 32 x 4 once M fills the GPU.
    """
    if rotation == "adaptive_butterfly_b16":
        return (8, 4, 4) if M < 128 else (32, 4, 4)
    return (8, 8, 4) if M < 128 else (8, 16, 4)


# ======================================================================================
# Triton kernels
# ======================================================================================
@triton.jit
def _nanmax(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _pairs(v, R: tl.constexpr, S: tl.constexpr):
    """(R, 16) -> the two halves (R, 8 // S, S) of every stage-S pair (i, i + S) in each block."""
    w = tl.permute(tl.reshape(v, (R, 8 // S, 2, S)), (0, 1, 3, 2))
    return tl.split(w)


@triton.jit
def _unpairs(a, d, R: tl.constexpr, S: tl.constexpr):
    return tl.reshape(tl.permute(tl.join(a, d), (0, 1, 3, 2)), (R, 16))


@triton.jit
def _butterfly_stage(v, cos_ptr, sin_ptr, groups, gmask, s: tl.constexpr, BM: tl.constexpr,
                     BG: tl.constexpr):
    """Stage ``s``: (a, d) -> (c a + s d, s a - c d) with angles (G, 4, 8) loaded for this tile."""
    R: tl.constexpr = BM * BG
    S: tl.constexpr = 1 << s
    offs = groups[:, None] * 32 + s * 8 + tl.arange(0, 8)[None, :]
    c = tl.load(cos_ptr + offs, mask=gmask[:, None], other=0.0)
    sn = tl.load(sin_ptr + offs, mask=gmask[:, None], other=0.0)
    c = tl.reshape(tl.broadcast_to(c[None, :, :], (BM, BG, 8)), (R, 8 // S, S))
    sn = tl.reshape(tl.broadcast_to(sn[None, :, :], (BM, BG, 8)), (R, 8 // S, S))
    a, d = _pairs(v, R, S)
    return _unpairs(c * a + sn * d, sn * a - c * d, R, S)


@triton.jit
def _load_rotated(x_ptr, cos_ptr, sin_ptr, rows, g0, M, K, G, stride_m, stride_k,
                  ROT: tl.constexpr, BM: tl.constexpr, BG: tl.constexpr):
    """FP32 tile (BM * BG, 16) of rotated activations; out-of-range inputs are 0."""
    R: tl.constexpr = BM * BG
    cols = g0 * 16 + tl.arange(0, BG * 16)
    mask = (rows[:, None] < M) & (cols[None, :] < K)
    ptrs = x_ptr + rows[:, None].to(tl.int64) * stride_m + cols[None, :].to(tl.int64) * stride_k
    v = tl.reshape(tl.load(ptrs, mask=mask, other=0.0).to(tl.float32), (R, 16))
    if ROT == 1:  # orthonormal Walsh-Hadamard (Sylvester order), the add sequence of ``fwht``
        for s in tl.static_range(4):
            a, d = _pairs(v, R, 1 << s)
            v = _unpairs(a + d, a - d, R, 1 << s)
        v = v * 0.25
    elif ROT == 2:  # adaptive butterfly: four stages of 2x2 reflections, ``Butterfly.forward``
        groups = g0 + tl.arange(0, BG)
        gmask = groups < G
        for s in tl.static_range(4):
            v = _butterfly_stage(v, cos_ptr, sin_ptr, groups, gmask, s, BM, BG)
    return v


@triton.jit
def _amax_kernel(x_ptr, part_ptr, cos_ptr, sin_ptr, M, K, G, stride_m, stride_k, n_gt, n_tiles,
                 ROT: tl.constexpr, BM: tl.constexpr, BG: tl.constexpr):
    """Pass 1: partial[program] = NaN-propagating max |R x| over a strided set of tiles."""
    pid = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for t in range(pid, n_tiles, tl.num_programs(0)):
        rows = (t // n_gt) * BM + tl.arange(0, BM)
        v = _load_rotated(x_ptr, cos_ptr, sin_ptr, rows, (t % n_gt) * BG, M, K, G, stride_m, stride_k,
                          ROT, BM, BG)
        acc = _nanmax(acc, tl.reduce(tl.abs(v), None, _nanmax))
    tl.store(part_ptr + pid, acc)


@triton.jit
def _quant_kernel(x_ptr, q_ptr, s_ptr, gs_ptr, part_ptr, cos_ptr, sin_ptr, M, K, G, stride_m, stride_k,
                  pc4, n_part, ROT: tl.constexpr, DYNAMIC: tl.constexpr, BM: tl.constexpr, BG: tl.constexpr,
                  PBLOCK: tl.constexpr):
    """Pass 2 (or the only pass with a static scale): quantize, pack and swizzle one tile.

    The grid covers the padded scale matrix (ceil(M / 128) * 128 rows, ``4 * pc4`` columns).
    """
    pid_m, pid_g = tl.program_id(0), tl.program_id(1)
    if DYNAMIC:
        p = tl.arange(0, PBLOCK)
        amax = tl.reduce(tl.load(part_ptr + p, mask=p < n_part, other=0.0), None, _nanmax)
        gs = tl.where(amax > 0, tl.div_rn(amax, 2688.0), 1.0)  # amax / (6 * 448); NaN or 0 -> 1
        if (pid_m == 0) & (pid_g == 0):
            tl.store(gs_ptr, gs)
    else:
        gs = tl.load(gs_ptr)
    rows = pid_m * BM + tl.arange(0, BM)
    groups = pid_g * BG + tl.arange(0, BG)
    v = _load_rotated(x_ptr, cos_ptr, sin_ptr, rows, pid_g * BG, M, K, G, stride_m, stride_k, ROT, BM, BG)
    v = tl.reshape(v, (BM, BG, 16))

    # E4M3 block scale: clamp(amax / (6 gs), max=448) -> e4m3 (round to nearest even).
    bamax = tl.reduce(tl.abs(v), 2, _nanmax)
    block = tl.minimum(tl.div_rn(bamax, 6.0 * gs), 448.0, propagate_nan=tl.PropagateNan.ALL)
    block8 = block.to(tl.float8e4nv)
    scale = block8.to(tl.float32) * gs
    safe = tl.where(scale > 0, scale, 1.0)
    y = tl.div_rn(v, safe[:, :, None])

    # E2M1 codes: the comparisons of ``e2m1_codes`` (ties to even), sign from the IEEE sign bit.
    a = tl.abs(y)
    code = ((a > 0.25).to(tl.uint8) + (a >= 0.75).to(tl.uint8) + (a > 1.25).to(tl.uint8)
            + (a >= 1.75).to(tl.uint8) + (a > 2.5).to(tl.uint8) + (a >= 3.5).to(tl.uint8)
            + (a > 5.0).to(tl.uint8))
    sign = ((y.to(tl.int32, bitcast=True) >> 31) & 1).to(tl.uint8)
    code = code | (sign << 3)
    lo, hi = tl.split(tl.reshape(code, (BM, BG, 8, 2)))
    packed = tl.reshape(lo | (hi << 4), (BM, BG * 8))
    pcols = pid_g * BG * 8 + tl.arange(0, BG * 8)
    half = K // 2
    tl.store(q_ptr + rows[:, None].to(tl.int64) * half + pcols[None, :], packed,
             mask=(rows[:, None] < M) & (pcols[None, :] < half))

    # SWIZZLE_32_4_4: scale (r, c) -> ((r // 128) * pc4 + c // 4) * 512 + (r % 32) * 16 + (r % 128 // 32) * 4 + c % 4.
    valid = (rows[:, None] < M) & (groups[None, :] < G)
    bits = tl.where(valid, block8.to(tl.uint8, bitcast=True), 0).to(tl.float8e4nv, bitcast=True)
    ri = rows % 128
    offs = (((rows // 128).to(tl.int64) * pc4)[:, None] + (groups // 4)[None, :]) * 512 \
        + ((ri % 32) * 16 + (ri // 32) * 4)[:, None] + (groups % 4)[None, :]
    tl.store(s_ptr + offs, bits, mask=groups[None, :] < pc4 * 4)


# ======================================================================================
# Launch
# ======================================================================================
_PLANS: dict = {}


def _launch(kernel, grid: tuple[int, int, int], tensors: tuple, scalars: tuple, constants: tuple,
            num_warps: int) -> None:
    """Launch ``kernel(*tensors, *scalars, *constants)`` with FP contraction disabled.

    The first call per specialization compiles; later calls use Triton's C launcher. The plan key
    holds what Triton specializes on: dtypes, 16-byte alignment and integer values.
    """
    key = (kernel, tensors[0].device, constants, num_warps, *[(t.dtype, t.data_ptr() & 15) for t in tensors],
           *scalars)
    plan = _PLANS.get(key)
    if plan is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up NVFP4 kernels eagerly once per shape before CUDA-graph capture")
        compiled = kernel[grid](*tensors, *scalars, *constants, num_warps=num_warps, enable_fp_fusion=False)
        _PLANS[key] = compiled, _c_launcher(compiled)
        return
    compiled, direct = plan
    hooks = knobs.runtime
    if direct is None or hooks.launch_enter_hook.calls or hooks.launch_exit_hook.calls:
        compiled[grid](*tensors, *scalars, *constants)
    else:
        launch, head = direct
        active = driver.active
        launch(*grid, active.get_current_stream(active.get_current_device()), *head,
               (*tensors, *scalars, *constants))


class NVFP4Activation(NamedTuple):
    packed: Tensor        # (M, K // 2) float4_e2m1fn_x2
    scale: Tensor         # (pr * pc,) float8_e4m3fn in SWIZZLE_32_4_4 layout
    global_scale: Tensor  # () float32


def scale_layout(M: int, K: int) -> tuple[int, int]:
    """Padded (rows, columns) of the swizzled block-scale matrix of an (M, K) operand."""
    return -(-M // 128) * 128, -(-(K // 16) // 4) * 4


def butterfly_tables(angles: Tensor) -> tuple[Tensor, Tensor]:
    """FP32 (cos, sin) of butterfly angles (K // 16, 4, 8), computed as ``Butterfly._stage`` does."""
    angles = angles.float().contiguous()
    return angles.cos(), angles.sin()


def _check(x: Tensor, rotation: str, tables) -> tuple[int, int]:
    if x.ndim != 2 or not x.is_cuda or x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Expected a CUDA (M, K) float32, bfloat16 or float16 tensor")
    M, K = x.shape
    if K % 16:
        raise ValueError(f"K = {K} is not a multiple of the NVFP4 block size 16")
    if rotation not in ROTATIONS:
        raise ValueError(f"Unknown rotation {rotation!r}; expected one of {tuple(ROTATIONS)}")
    if rotation == "adaptive_butterfly_b16":
        if tables is None or tables[0].shape != (K // 16, 4, 8):
            raise ValueError("adaptive_butterfly_b16 needs (cos, sin) tables of shape (K // 16, 4, 8)")
    return M, K


def _amax(x, M, K, workspace, rot, tables) -> int:
    G = K // 16
    bm, bg, warps = _tile(M, rot)
    n_gt = -(-G // bg)
    n_tiles = -(-M // bm) * n_gt
    programs = max(1, min(n_tiles, MAX_PARTIALS))
    cos, sin = tables if tables is not None else (workspace, workspace)
    _launch(_amax_kernel, (programs, 1, 1), (x, workspace, cos, sin),
            (M, K, G, x.stride(0), x.stride(1), n_gt, max(n_tiles, 1)), (ROTATIONS[rot], bm, bg), warps)
    return programs


def _quant(x, M, K, packed_u8, scale, gs, workspace, n_part, rot, tables) -> None:
    pr, pc = scale_layout(M, K)
    bm, bg, warps = _tile(M, rot)
    dynamic = workspace is not None
    ws = workspace if dynamic else gs
    cos, sin = tables if tables is not None else (ws, ws)
    _launch(_quant_kernel, (pr // bm, -(-pc // bg), 1), (x, packed_u8, scale, gs, ws, cos, sin),
            (M, K, K // 16, x.stride(0), x.stride(1), pc // 4, n_part),
            (ROTATIONS[rot], dynamic, bm, bg, MAX_PARTIALS), warps)


def launch_amax(x: Tensor, workspace: Tensor, rotation: str = "none", tables=None) -> int:
    """Pass 1 into ``workspace`` (>= MAX_PARTIALS + 1 float32). Returns the number of partials."""
    M, K = _check(x, rotation, tables)
    return _amax(x, M, K, workspace, rotation, tables)


def launch_quant(x: Tensor, packed: Tensor, scale: Tensor, global_scale: Tensor, *, rotation: str = "none",
                 tables=None, workspace: Tensor | None = None, n_part: int = 0) -> None:
    """Pass 2 or the static single pass, into ``packed`` (uint8 view) and ``scale`` (float8_e4m3fn).

    With ``workspace`` (dynamic) the kernel reduces its first ``n_part`` partials and writes the
    tensor scale to ``global_scale``; without it, ``global_scale`` is read (static).
    """
    M, K = _check(x, rotation, tables)
    _quant(x, M, K, packed.view(torch.uint8), scale.view(torch.float8_e4m3fn), global_scale, workspace, n_part,
           rotation, tables)


def _quantize(x: Tensor, M: int, K: int, rotation: str, tables, global_scale: Tensor | None) -> NVFP4Activation:
    pr, pc = scale_layout(M, K)
    packed = torch.empty((M, K // 2), dtype=torch.float4_e2m1fn_x2, device=x.device)
    scale = torch.empty(pr * pc, dtype=torch.float8_e4m3fn, device=x.device)
    packed_u8 = packed.view(torch.uint8)
    if global_scale is None:
        workspace = torch.empty(MAX_PARTIALS + 1, dtype=torch.float32, device=x.device)
        gs = workspace[MAX_PARTIALS]
        n_part = _amax(x, M, K, workspace, rotation, tables)
        _quant(x, M, K, packed_u8, scale, gs, workspace, n_part, rotation, tables)
    else:
        gs = global_scale
        _quant(x, M, K, packed_u8, scale, gs, None, 0, rotation, tables)
    return NVFP4Activation(packed, scale, gs)


def quantize_activation(x: Tensor, rotation: str = "none", tables=None,
                        global_scale: Tensor | None = None) -> NVFP4Activation:
    """``F.scaled_mm``-ready NVFP4 operands of ``x`` (M, K). Static mode when ``global_scale`` is given."""
    M, K = _check(x, rotation, tables)
    if global_scale is not None:
        global_scale = global_scale.reshape(())
        if global_scale.dtype != torch.float32 or global_scale.device != x.device:
            raise ValueError("A static global scale must be a float32 tensor on the device of x")
    return _quantize(x, M, K, rotation, tables, global_scale)


# ======================================================================================
# PyTorch references (eager)
# ======================================================================================
def make_rotation(rotation: str, angles: Tensor | None = None) -> q.Rotation:
    if rotation == "none":
        return q.Rotation()
    if rotation == "hadamard_b16":
        return q.BlockHadamard(16)
    if rotation == "adaptive_butterfly_b16":
        return q.Butterfly(16, angles.float())
    raise ValueError(f"Unknown rotation {rotation!r}")


def nvfp4_encode_ref(W: Tensor, global_scale: Tensor | None = None) -> dict:
    """``micro_rotated_ptq.nvfp4_encode``, or the same with a given (static) FP32 tensor scale."""
    if global_scale is None:
        return q.nvfp4_encode(W)
    rows, cols = W.shape
    gs = global_scale.float().reshape(1, 1, 1)
    grouped = W.float().reshape(rows, cols // 16, 16)
    block = q.to_e4m3(grouped.abs().amax(-1, keepdim=True) / (q.E2M1_MAX * gs))
    codes, values = q.quantize_with(grouped, block * gs, None, q.FORMATS["nvfp4_g16"])
    return {"codes": codes.reshape(rows, cols), "block_scale": block.squeeze(-1).to(torch.float8_e4m3fn),
            "global_scale": gs.reshape(()), "values": values.reshape(rows, cols)}


def quantize_activation_ref(x: Tensor, rotation: str = "none", angles: Tensor | None = None,
                            global_scale: Tensor | None = None) -> NVFP4Activation:
    """Eager reference: rotate in FP32, ``nvfp4_encode``, ``pack_e2m1`` and ``to_blocked``."""
    enc = nvfp4_encode_ref(make_rotation(rotation, angles).forward(x.float()), global_scale)
    return NVFP4Activation(q.pack_e2m1(enc["codes"]), q.to_blocked(enc["block_scale"]),
                           enc["global_scale"].float())


def dequantize(act: NVFP4Activation, M: int, K: int) -> Tensor:
    """FP32 values of NVFP4 operands (undoes the swizzle)."""
    pr, pc = scale_layout(M, K)
    flat = act.scale.view(torch.uint8)
    # to_blocked(index matrix)[p] is the row-major index of the scale stored at flat position p.
    perm = q.to_blocked(torch.arange(pr * pc, device=flat.device).view(pr, pc))
    logical = torch.empty_like(flat)
    logical[perm] = flat
    block = logical.view(pr, pc)[:M, :K // 16].view(torch.float8_e4m3fn).float()
    codes = q.unpack_e2m1(act.packed)
    return q.e2m1_decode(codes) * (block * act.global_scale.float()).repeat_interleave(16, dim=-1)


# ======================================================================================
# GEMM and module
# ======================================================================================
_SWIZZLE = [int(F.SwizzleType.SWIZZLE_32_4_4.value), int(F.SwizzleType.NO_SWIZZLE.value)]
_RECIPE = [int(F.ScalingType.BlockWise1x16.value), int(F.ScalingType.TensorWise.value)]


def nvfp4_scaled_mm(a: NVFP4Activation, w_packed: Tensor, w_scale: Tensor, w_global: Tensor,
                    out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """``a @ W^T`` with BlockWise1x16 + TensorWise scales, both operands ``SWIZZLE_32_4_4``.

    Calls ``torch._scaled_mm_v2`` with the integer recipe lists that ``F.scaled_mm`` would build,
    which saves the wrapper's per-call list conversions (same kernel, same result).
    """
    return torch._scaled_mm_v2(a.packed, w_packed.t(), [a.scale, a.global_scale], _RECIPE, _SWIZZLE,
                               [w_scale, w_global], _RECIPE, _SWIZZLE, None, out_dtype, (), False)


def nvfp4_scaled_mm_public(a: NVFP4Activation, w_packed: Tensor, w_scale: Tensor, w_global: Tensor,
                           out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """The same GEMM through the public ``F.scaled_mm`` API (reference for the direct call)."""
    swz = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]
    recipe = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
    return F.scaled_mm(a.packed, w_packed.t(), [a.scale, a.global_scale], recipe, [w_scale, w_global], recipe,
                       swizzle_a=swz, swizzle_b=swz, output_dtype=out_dtype)


class NVFP4Linear(nn.Module):
    """W4A4 linear layer: offline NVFP4 weights, online fused activation quantization, native FP4 GEMM.

    Stores the weights of the rotated layer, ``Q(W R^T)``, so ``forward(x) = Q(x R^T) Q(W R^T)^T``.
    ``act_global_scale`` (FP32 scalar) selects the static one-pass activation path.
    """

    def __init__(self, w_packed: Tensor, w_scale: Tensor, w_global: Tensor, *, in_features: int,
                 out_features: int, rotation: str = "none", angles: Tensor | None = None,
                 act_global_scale: Tensor | None = None, out_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        if rotation not in ROTATIONS:
            raise ValueError(f"Unknown rotation {rotation!r}")
        if in_features % 32:
            raise ValueError("scaled_mm needs K to be a multiple of 32 (16-byte FP4 rows)")
        self.in_features, self.out_features = in_features, out_features
        self.rotation, self.out_dtype = rotation, out_dtype
        # FP4/FP8 count as floating point, so ``.to(dtype)`` would cast them: store their bytes.
        self.register_buffer("w_packed_bytes", w_packed.view(torch.uint8).reshape(out_features, in_features // 2))
        self.register_buffer("w_scale_bytes", w_scale.reshape(-1).view(torch.uint8))
        self.register_buffer("w_global", w_global.float().reshape(()))
        self.register_buffer("angles", None if angles is None else angles.float().contiguous())
        if rotation == "adaptive_butterfly_b16":
            if angles is None:
                raise ValueError("adaptive_butterfly_b16 needs angles (K // 16, 4, 8)")
            cos, sin = butterfly_tables(angles)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)
        self.register_buffer("act_global_scale",
                             None if act_global_scale is None else act_global_scale.float().reshape(()).clone())

    @classmethod
    def from_encoding(cls, codes: Tensor, block_scale: Tensor, global_scale: Tensor, **kwargs) -> "NVFP4Linear":
        """From E2M1 codes (N, K), E4M3 block scales (N, K // 16) and the FP32 tensor scale of ``W R^T``."""
        N, K = codes.shape
        return cls(q.pack_e2m1(codes), q.to_blocked(block_scale), global_scale, in_features=K, out_features=N,
                   **kwargs)

    @classmethod
    def from_weight(cls, W: Tensor, rotation: str = "none", angles: Tensor | None = None, **kwargs) -> "NVFP4Linear":
        """Round-to-nearest NVFP4 of the rotated weight ``W R^T`` (``W`` is (N, K))."""
        enc = q.nvfp4_encode(make_rotation(rotation, angles).forward(W.detach().float()))
        return cls.from_encoding(enc["codes"], enc["block_scale"], enc["global_scale"], rotation=rotation,
                                 angles=angles, **kwargs)

    def tables(self):
        return (self.cos, self.sin) if self.rotation == "adaptive_butterfly_b16" else None

    @property
    def _tables(self):
        return (self.cos, self.sin) if self.rotation == "adaptive_butterfly_b16" else None

    @property
    def w_packed(self) -> Tensor:
        return self.w_packed_bytes.view(torch.float4_e2m1fn_x2)

    @property
    def w_scale(self) -> Tensor:
        return self.w_scale_bytes.view(torch.float8_e4m3fn)

    _FP32_BUFFERS = ("w_global", "act_global_scale", "angles", "cos", "sin")

    def _apply(self, fn, recurse=True):
        # ``model.to(torch.bfloat16)`` must not cast the FP32 scales and rotation tables: hold them
        # out of ``fn`` and only follow the device of the (never dtype-cast) uint8 weight bytes.
        held = {n: self._buffers.pop(n) for n in self._FP32_BUFFERS if self._buffers.get(n) is not None}
        try:
            super()._apply(fn, recurse)
        finally:
            for name, value in held.items():
                self._buffers[name] = value.to(self.w_packed_bytes.device)
        return self

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
        a = _quantize(x2, M, self.in_features, self.rotation, self._tables, self.act_global_scale)
        out = nvfp4_scaled_mm(a, self.w_packed, self.w_scale, self.w_global, self.out_dtype)
        return out.reshape(*lead, self.out_features)

    def extra_repr(self) -> str:
        mode = "static" if self.act_global_scale is not None else "dynamic"
        return f"in={self.in_features}, out={self.out_features}, rotation={self.rotation}, act_scale={mode}"


def dequantized_weight(layer: NVFP4Linear) -> Tensor:
    """FP32 ``Q(W R^T)`` of a layer (rotated coordinates)."""
    act = NVFP4Activation(layer.w_packed, layer.w_scale, layer.w_global)
    return dequantize(act, layer.out_features, layer.in_features)


__all__ = ["NVFP4Activation", "NVFP4Linear", "quantize_activation", "quantize_activation_ref", "launch_amax",
           "launch_quant", "nvfp4_scaled_mm", "nvfp4_scaled_mm_public", "dequantize", "dequantized_weight", "nvfp4_encode_ref",
           "butterfly_tables", "scale_layout", "ROTATIONS", "MAX_PARTIALS"]
