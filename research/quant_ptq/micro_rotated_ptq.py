"""Micro-rotated post-training quantization: NVFP4, MXFP4 and 2-bit weights (Cycle 3, T5-01).

The module quantizes the weights of every Transformer linear layer (attention QKV and output,
MLP gate, up and down). Embeddings, norms and the tied LM head stay in BF16/FP32. Activations
are not quantized ("weight-only" PTQ), and evaluation uses dequantized weights ("fake quant").

Formats (groups run along the input dimension, which is the GEMM contraction dimension)
    nvfp4_g16      FP4 E2M1 elements, one float8_e4m3fn scale per 16 weights, one FP32 scale per
                   tensor. The tensor scale is amax / (6 * 448), so the largest block maps to the
                   E4M3 maximum 448 and no block scale overflows. 4.5 bits + 32 / numel.
    mxfp4_g32      FP4 E2M1 elements, one E8M0 power-of-two scale per 32 weights (OCP MX v1.0:
                   2^(floor(log2 amax) - 2)). 4.25 bits.
    int2_asym_g16  4 uniform levels per 16 weights, FP16 scale and FP16 zero (group minimum).
    int2_asym_g32  Same with 32 weights per group. 4.0 and 3.0 bits.

Micro-rotations (orthogonal R; the layer computes (x R^T)(W R^T)^T = x W^T)
    hadamard_b16 / hadamard_b64   block-diagonal Walsh-Hadamard (fast butterfly transform)
    rand_hadamard_b64             R = H D, with a seeded random sign vector D per layer
    adaptive_butterfly_b16        per-block product of 4 butterfly stages of 2x2 orthogonal
                                  reflections [[c, s], [s, -c]]. At angle pi/4 every stage is the
                                  Hadamard stage, so the start point is exactly hadamard_b16.
                                  Adam on the angles minimizes the weight quantization error of
                                  the target format (fixed-code gradient). Data is not used.
                                  A layer keeps the fitted angles only if the exact quantizer
                                  error is lower than at the start point.

Rounding
    rtn    round to nearest in each group.
    gptq   GPTQ (Frantar et al. 2023): column-sequential error feedback through the upper
           Cholesky factor of H^-1, with lazy 128-column block updates. LDLQ (Chee et al. 2023)
           is the same algorithm. H = X^T X / N + lambda * mean(diag) * I, with lambda = 0.01 and
           a x10 per-layer fallback when Cholesky fails. In rotated coordinates H_R = R H R^T.
           Hessians come from one forward pass of the full-precision model ("parallel" GPTQ,
           not layer-by-layer sequential); calibration tokens come only from ``train.bin``.

Run (full Cycle 3 experiment, about 10 minutes on an RTX 5080):
    PYTHONPATH=. .venv/bin/python research/quant_ptq/micro_rotated_ptq.py \\
        --output results/cycle3_t5_01_nvfp4_ptq.json
Smoke run:
    PYTHONPATH=. .venv/bin/python research/quant_ptq/micro_rotated_ptq.py --quick --output /tmp/q.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from research.polar_opt import harness_tier2_real as h
from research.polar_opt.harness_tier2_lm import Transformer

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "cycle3_t5_01.v1"
EXPECTED_SHA256 = {
    "train": "015ab83ceb742555c2398963a0bd1f3c5502dc52cf70a22bf173f1571326ca3a",
    "val": "371b64a537f7b1e306ea174a868b05932a254a98b7fbdbf4146648d93180b0b1",
    "test": "4eb46f1424856849d2973923dbf3c886d37eb743cf22053fc7e1eea1f41ab92a",
}

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E2M1_MAX = 6.0
E4M3_MAX = 448.0
E8M0_MIN_EXP, E8M0_MAX_EXP = -127, 127
LINEAR_KINDS = ("qkv", "out", "gate", "up", "down")


def default_data_dir() -> Path:
    """``REAL_TEXT_DIR``, else ``artifacts/real_text`` here, else in the main checkout of this worktree."""
    if os.environ.get("REAL_TEXT_DIR"):
        return Path(os.environ["REAL_TEXT_DIR"])
    local = ROOT / "artifacts" / "real_text"
    if local.exists():
        return local
    try:
        common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return Path(common).parent / "artifacts" / "real_text"
    except (OSError, subprocess.SubprocessError):
        return local


# ======================================================================================
# Formats
# ======================================================================================
@dataclass(frozen=True)
class QuantFormat:
    name: str
    kind: str  # "nvfp4" | "mxfp4" | "int2"
    group_size: int
    element_bits: int
    scale_bits: int
    zero_bits: int = 0
    tensor_scale_bits: int = 0
    metadata: str = ""

    @property
    def levels(self) -> int:
        return 1 << self.element_bits

    def bits_per_weight(self, numel: int | None = None) -> float:
        """Storage bits per weight: elements + group scale/zero + the tensor scale over ``numel``."""
        bits = self.element_bits + (self.scale_bits + self.zero_bits) / self.group_size
        return bits + (self.tensor_scale_bits / numel if numel else 0.0)


FORMATS = {
    "nvfp4_g16": QuantFormat("nvfp4_g16", "nvfp4", 16, 4, 8, 0, 32,
                             "E2M1 codes; float8_e4m3fn scale per 16; FP32 scale per tensor"),
    "mxfp4_g32": QuantFormat("mxfp4_g32", "mxfp4", 32, 4, 8, 0, 0,
                             "E2M1 codes; E8M0 power-of-two scale per 32 (OCP MX v1.0 floor rule)"),
    "int2_asym_g16": QuantFormat("int2_asym_g16", "int2", 16, 2, 16, 16, 0,
                                 "uint2 codes; FP16 scale and FP16 zero (minimum) per 16"),
    "int2_asym_g32": QuantFormat("int2_asym_g32", "int2", 32, 2, 16, 16, 0,
                                 "uint2 codes; FP16 scale and FP16 zero (minimum) per 32"),
}


# ======================================================================================
# Element codecs
# ======================================================================================
def _ste(x: Tensor, fn) -> Tensor:
    """``fn(x)`` in the forward pass; identity gradient when ``x`` needs a gradient."""
    y = fn(x.detach())
    return x + (y - x).detach() if x.requires_grad else y


def to_e4m3(x: Tensor) -> Tensor:
    """Round to float8_e4m3fn (round to nearest even), saturate at 448, return in ``x.dtype``."""
    return x.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).to(x.dtype)


FP16_MAX = float(torch.finfo(torch.float16).max)


def to_fp16(x: Tensor) -> Tensor:
    return x.clamp(-FP16_MAX, FP16_MAX).to(torch.float16).to(x.dtype)


def e2m1_codes(x: Tensor) -> Tensor:
    """4-bit E2M1 codes (bit 3 = sign, bits 0-2 = magnitude index into ``E2M1_VALUES``).

    Round to nearest; a tie goes to the even magnitude index (even mantissa), as in IEEE
    round-to-nearest-even. Magnitudes above 6 saturate to 6.
    """
    a = x.detach().abs()
    index = torch.zeros(x.shape, dtype=torch.uint8, device=x.device)
    for k, midpoint in enumerate(_E2M1_MIDPOINTS):
        # Tie between index k and k + 1 goes to the even one: down for even k, up for odd k.
        index += (a >= midpoint if k % 2 else a > midpoint).to(torch.uint8)
    return index | (torch.signbit(x.detach()).to(torch.uint8) << 3)


def e2m1_decode(codes: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    magnitude = torch.tensor(E2M1_VALUES, dtype=dtype, device=codes.device)
    table = torch.cat((magnitude, -magnitude))
    return table[codes.long()]


def pack_e2m1(codes: Tensor) -> Tensor:
    """Pack pairs of 4-bit codes into ``torch.float4_e2m1fn_x2`` (element 2i in the low nibble)."""
    if codes.shape[-1] % 2:
        raise ValueError("The last dimension must be even to pack FP4 pairs")
    codes = codes.to(torch.uint8)
    packed = (codes[..., 1::2] << 4) | codes[..., ::2]
    return packed.view(torch.float4_e2m1fn_x2)


def unpack_e2m1(packed: Tensor) -> Tensor:
    raw = packed.view(torch.uint8)
    return torch.stack((raw & 0xF, raw >> 4), dim=-1).flatten(-2)


# ======================================================================================
# Group quantizers
# ======================================================================================
def nvfp4_global_scale(W: Tensor) -> Tensor:
    """FP32 per-tensor scale amax / (6 * 448) over the last two dimensions, shape (..., 1, 1)."""
    amax = W.detach().abs().amax(dim=(-2, -1), keepdim=True).float()
    # Tensor divisor: CUDA divides by a Python scalar via its reciprocal, which is not exact IEEE division.
    return torch.where(amax > 0, amax / torch.full_like(amax, E2M1_MAX * E4M3_MAX), torch.ones_like(amax))


def mxfp4_exponent(amax: Tensor) -> Tensor:
    """OCP MX shared exponent floor(log2(amax)) - 2, clamped to the E8M0 range. Exact via frexp."""
    _, exponent = torch.frexp(amax.float())  # amax = m * 2^e with m in [0.5, 1)
    exponent = torch.where(amax > 0, exponent - 1 - 2, torch.full_like(exponent, E8M0_MIN_EXP))
    return exponent.clamp(E8M0_MIN_EXP, E8M0_MAX_EXP)


def group_params(block: Tensor, fmt: QuantFormat, global_scale: Tensor | None = None):
    """Scale (and INT2 zero) of each group along the last dimension, shape (..., 1).

    The NVFP4 scale returned is the effective one, ``e4m3_block_scale * global_scale``.
    Scale rounding uses a straight-through gradient, so an STE fit can differentiate it.
    """
    if fmt.kind == "nvfp4":
        if global_scale is None:
            raise ValueError("NVFP4 needs the per-tensor global scale")
        amax = block.abs().amax(-1, keepdim=True)
        return _ste(amax / (E2M1_MAX * global_scale), to_e4m3) * global_scale, None
    if fmt.kind == "mxfp4":
        amax = block.detach().abs().amax(-1, keepdim=True)
        return torch.exp2(mxfp4_exponent(amax).to(block.dtype)), None
    if fmt.kind == "int2":
        low, high = block.amin(-1, keepdim=True), block.amax(-1, keepdim=True)
        return _ste((high - low) / (fmt.levels - 1), to_fp16), _ste(low, to_fp16)
    raise ValueError(f"Unknown format kind {fmt.kind!r}")


def quantize_with(x: Tensor, scale: Tensor, zero: Tensor | None, fmt: QuantFormat) -> tuple[Tensor, Tensor]:
    """(codes, dequantized values) of ``x`` for given group parameters. Codes carry no gradient."""
    safe = torch.where(scale > 0, scale, torch.ones_like(scale)).detach()
    if fmt.kind in ("nvfp4", "mxfp4"):
        codes = e2m1_codes(x.detach() / safe)
        return codes, e2m1_decode(codes, x.dtype) * scale  # a zero scale gives exact zeros
    q = torch.round((x.detach() - zero.detach()) / safe).clamp(0, fmt.levels - 1)
    return q.to(torch.uint8), q * scale + zero


def quantize_rtn(W: Tensor, fmt: QuantFormat, global_scale: Tensor | None = None,
                 return_codes: bool = False):
    """Round-to-nearest group quantization along the last dimension. Returns dequantized weights."""
    *lead, rows, cols = W.shape
    g = fmt.group_size
    if cols % g:
        raise ValueError(f"{cols} columns are not a multiple of group size {g}")
    if fmt.kind == "nvfp4" and global_scale is None:
        global_scale = nvfp4_global_scale(W)
    grouped = W.reshape(*lead, rows, cols // g, g)
    gs = global_scale.unsqueeze(-1) if global_scale is not None else None
    scale, zero = group_params(grouped, fmt, gs)
    codes, values = quantize_with(grouped, scale, zero, fmt)
    values = values.reshape(W.shape)
    return (values, codes.reshape(W.shape)) if return_codes else values


def nvfp4_encode(W: Tensor) -> dict:
    """Hardware NVFP4 tensors of a 2D ``W``: E2M1 codes, E4M3 block scales and the FP32 tensor scale."""
    rows, cols = W.shape
    W = W.float()
    gs = nvfp4_global_scale(W)
    grouped = W.reshape(rows, cols // 16, 16)
    block = to_e4m3(grouped.abs().amax(-1, keepdim=True) / (E2M1_MAX * gs.unsqueeze(-1)))
    scale = block * gs.unsqueeze(-1)
    codes, values = quantize_with(grouped, scale, None, FORMATS["nvfp4_g16"])
    return {"codes": codes.reshape(rows, cols), "block_scale": block.squeeze(-1).to(torch.float8_e4m3fn),
            "global_scale": gs.reshape(()), "values": values.reshape(rows, cols)}


def mxfp4_encode(W: Tensor) -> dict:
    """Hardware MXFP4 tensors of a 2D ``W``: E2M1 codes and biased E8M0 block exponents."""
    rows, cols = W.shape
    grouped = W.float().reshape(rows, cols // 32, 32)
    exponent = mxfp4_exponent(grouped.abs().amax(-1, keepdim=True))
    codes, values = quantize_with(grouped, torch.exp2(exponent.float()), None, FORMATS["mxfp4_g32"])
    e8m0 = (exponent.squeeze(-1) + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    return {"codes": codes.reshape(rows, cols), "block_scale": e8m0, "values": values.reshape(rows, cols)}


def nvfp4_decode(encoded: dict) -> Tensor:
    codes, block = encoded["codes"], encoded["block_scale"].float()
    scale = (block * encoded["global_scale"]).repeat_interleave(16, dim=-1)
    return e2m1_decode(codes) * scale


# ======================================================================================
# Micro-rotations. ``forward(x) = x @ R^T`` and ``inverse(z) = z @ R`` along the last dimension.
# ======================================================================================
def fwht(x: Tensor, block: int) -> Tensor:
    """Orthonormal block-diagonal Walsh-Hadamard transform (Sylvester order) along the last dim."""
    n = x.shape[-1]
    if block & (block - 1) or n % block:
        raise ValueError(f"Block {block} must be a power of two dividing {n}")
    y = x
    step = 1
    while step < block:
        y = y.reshape(*x.shape[:-1], -1, block // (2 * step), 2, step)
        a, c = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + c, a - c), dim=-2)
        step *= 2
    return y.reshape(x.shape) * (block ** -0.5)


class Rotation:
    name = "identity"
    block = 1
    parameter_bits_per_input = 0.0  # storage per input channel, beyond the quantized weights

    def forward(self, x: Tensor) -> Tensor:
        return x

    def inverse(self, z: Tensor) -> Tensor:
        return z

    def rotate_hessian(self, H: Tensor) -> Tensor:
        """R H R^T for a symmetric H."""
        return self.forward(self.forward(H).mT)

    def matrix(self, n: int, dtype=torch.float64, device=None) -> Tensor:
        """Dense R for an unbatched rotation (rows of forward(I) are the columns of R)."""
        return self.forward(torch.eye(n, dtype=dtype, device=device)).mT


class BlockHadamard(Rotation):
    def __init__(self, block: int):
        self.block, self.name = block, f"hadamard_b{block}"

    def forward(self, x):
        return fwht(x, self.block)

    inverse = forward  # H is symmetric and orthonormal


class SignHadamard(Rotation):
    """R = H D. ``signs`` has shape (*lead, n) with entries +-1."""

    parameter_bits_per_input = 1.0

    def __init__(self, block: int, signs: Tensor):
        self.block, self.name = block, f"rand_hadamard_b{block}"
        self.signs = signs.unsqueeze(-2)

    def forward(self, x):
        return fwht(x * self.signs.to(x.dtype), self.block)

    def inverse(self, z):
        return fwht(z, self.block) * self.signs.to(z.dtype)


def random_signs(shape, seed: int, device=None) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, shape, generator=generator) * 2 - 1).float().to(device)


class Butterfly(Rotation):
    """Block-diagonal butterfly of 2x2 orthogonal reflections [[c, s], [s, -c]].

    ``angles`` has shape (*lead, n // block, log2(block), block // 2). Stage ``s`` pairs
    coordinates i and i + 2^s of each block. Every stage is symmetric and orthogonal, so the
    inverse applies the stages in reverse order. All angles pi/4 give ``fwht(x, block)``.
    """

    parameter_bits_per_input = None  # set from the angle count below

    def __init__(self, block: int, angles: Tensor):
        self.block, self.name, self.angles = block, f"adaptive_butterfly_b{block}", angles
        self.stages = int(math.log2(block))
        # FP16 angles: stages * (block / 2) per block of ``block`` inputs.
        self.parameter_bits_per_input = 16.0 * self.stages / 2

    @staticmethod
    def initial_angles(lead, n: int, block: int, device=None, dtype=torch.float32) -> Tensor:
        return torch.full((*lead, n // block, int(math.log2(block)), block // 2), math.pi / 4, device=device,
                          dtype=dtype)

    def _stage(self, x: Tensor, s: int) -> Tensor:
        b, step = self.block, 1 << s
        theta = self.angles[..., s, :].to(x.dtype).unsqueeze(-3)
        theta = theta.reshape(*theta.shape[:-1], b // (2 * step), step)
        c, sn = theta.cos(), theta.sin()
        y = x.reshape(*x.shape[:-1], -1, b // (2 * step), 2, step)
        a, d = y[..., 0, :], y[..., 1, :]
        return torch.stack((c * a + sn * d, sn * a - c * d), dim=-2).reshape(x.shape)

    def forward(self, x):
        for s in range(self.stages):
            x = self._stage(x, s)
        return x

    def inverse(self, z):
        for s in reversed(range(self.stages)):
            z = self._stage(z, s)
        return z


def fit_butterfly(W: Tensor, fmt: QuantFormat, *, block: int = 16, steps: int = 150, lr: float = 0.01,
                  check_every: int = 25) -> tuple[Butterfly, dict]:
    """Fit butterfly angles per layer of a stack ``W`` (L, rows, cols) to the weight quantization error.

    Adam minimizes ||Q(W R^T) - W R^T||^2 with the codes held fixed in each step (the gradient
    flows through W R^T and the group scales). Every ``check_every`` steps the exact quantizer
    error is measured, and each layer keeps its best angles. The start point is Hadamard, so a
    layer never ends worse than ``hadamard_b{block}``. No data is used.
    """
    W = W.detach().float()
    lead, cols = W.shape[:-2], W.shape[-1]

    def exact_error(angles):
        with torch.no_grad():
            Z = Butterfly(block, angles).forward(W)
            return (quantize_rtn(Z, fmt) - Z).pow(2).sum(dim=(-2, -1))

    theta = Butterfly.initial_angles(lead, cols, block, W.device).requires_grad_(True)
    best, best_error = theta.detach().clone(), exact_error(theta.detach())
    start_error = best_error.clone()
    optimizer = torch.optim.Adam([theta], lr=lr)
    for step in range(1, steps + 1):
        Z = Butterfly(block, theta).forward(W)
        gs = nvfp4_global_scale(Z) if fmt.kind == "nvfp4" else None
        loss = (quantize_rtn(Z, fmt, gs) - Z).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % check_every == 0 or step == steps:
            error = exact_error(theta.detach())
            better = error < best_error
            best[better] = theta.detach()[better]
            best_error = torch.where(better, error, best_error)
    improved = best_error < start_error
    info = {"steps": steps, "lr": lr, "layers": int(improved.numel()), "layers_improved": int(improved.sum()),
            "weight_sq_error_hadamard_start": start_error.sum().item(),
            "weight_sq_error_fitted": best_error.sum().item(),
            "relative_error_reduction": 1.0 - best_error.sum().item() / max(start_error.sum().item(), 1e-30)}
    return Butterfly(block, best), info


ROTATION_FAMILIES = ("identity", "hadamard_b16", "hadamard_b64", "rand_hadamard_b64", "adaptive_butterfly_b16")


def make_rotation(family: str, W: Tensor | None = None, fmt: QuantFormat | None = None, *, seed: int = 0,
                  butterfly_steps: int = 150) -> tuple[Rotation, dict]:
    """Rotation for a weight stack ``W`` (L, rows, cols). Seeded and fitted rotations are per layer."""
    if family == "identity":
        return Rotation(), {}
    if family.startswith("hadamard_b"):
        return BlockHadamard(int(family.removeprefix("hadamard_b"))), {}
    if family.startswith("rand_hadamard_b"):
        L, cols = W.shape[0], W.shape[-1]
        signs = torch.stack([random_signs((cols,), seed + i, W.device) for i in range(L)])
        return SignHadamard(int(family.removeprefix("rand_hadamard_b")), signs), {"seeds": [seed + i for i in range(L)]}
    if family.startswith("adaptive_butterfly_b"):
        return fit_butterfly(W, fmt, block=int(family.removeprefix("adaptive_butterfly_b")), steps=butterfly_steps)
    raise ValueError(f"Unknown rotation family {family!r}")


# ======================================================================================
# GPTQ / LDLQ
# ======================================================================================
def gptq_quantize(W: Tensor, H: Tensor, fmt: QuantFormat, *, block_size: int = 128, damp: float = 0.01,
                  max_damp_tries: int = 8, return_encoding: bool = False) -> tuple[Tensor, dict]:
    """GPTQ of a weight stack ``W`` (L, rows, cols) with input Hessians ``H`` (L, cols, cols).

    Column j is rounded with its group's parameters, and its scaled error
    (w_j - q_j) / U_jj is fed into the later columns through row j of U, the upper Cholesky
    factor of H^-1. Group parameters are fitted when the sweep reaches the first column of a
    group, from the error-updated weights. With H = I this is exactly RTN.

    ``return_encoding`` (NVFP4 only) adds ``info["encoding"]``: the E2M1 codes, the E4M3 block
    scales and the FP32 tensor scales, so that ``codes * block * global`` is exactly ``Q``.
    """
    if W.ndim != 3 or H.ndim != 3:
        raise ValueError("Expected W (L, rows, cols) and H (L, cols, cols)")
    g = fmt.group_size
    if block_size % g or W.shape[-1] % g:
        raise ValueError("Block size and column count must be multiples of the group size")
    W = W.detach().float().clone()
    L, rows, cols = W.shape
    H = H.detach().double().clone()
    diag = H.diagonal(dim1=-2, dim2=-1)
    dead = diag <= 0
    diag[dead] = 1.0  # no input energy: any value is exact; keep the weight and round it like RTN
    mean_diag = diag.mean(-1)
    eye = torch.eye(cols, dtype=H.dtype, device=H.device)
    lam = torch.full((L,), damp, dtype=H.dtype, device=H.device)
    for _ in range(max_damp_tries):
        chol, info = torch.linalg.cholesky_ex(H + (lam * mean_diag)[:, None, None] * eye)
        if not (info > 0).any():
            break
        lam = torch.where(info > 0, lam * 10, lam)
    else:
        raise RuntimeError("Cholesky failed after the damping fallback")
    U, info = torch.linalg.cholesky_ex(torch.cholesky_inverse(chol), upper=True)
    if (info > 0).any():
        raise RuntimeError("Cholesky of H^-1 failed")
    U = U.float()
    gs = nvfp4_global_scale(W) if fmt.kind == "nvfp4" else None
    Q = torch.zeros_like(W)
    encode = return_encoding and fmt.kind == "nvfp4"
    if return_encoding and not encode:
        raise ValueError("return_encoding needs the NVFP4 format")
    if encode:
        codes = torch.zeros(W.shape, dtype=torch.uint8, device=W.device)
        blocks = torch.zeros(L, rows, cols // g, dtype=W.dtype, device=W.device)
    scale = zero = None
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W1 = W[:, :, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        U1 = U[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            if i % g == 0:
                scale, zero = group_params(W1[:, :, i:i + g], fmt, gs)
                if encode:  # the E4M3 factor of ``scale``, by the expression of ``group_params``
                    amax = W1[:, :, i:i + g].abs().amax(-1, keepdim=True)
                    blocks[:, :, (i1 + i) // g] = to_e4m3(amax / (E2M1_MAX * gs)).squeeze(-1)
            w = W1[:, :, i:i + 1]
            c, q = quantize_with(w, scale, zero, fmt)
            Q[:, :, i1 + i:i1 + i + 1] = q
            if encode:
                codes[:, :, i1 + i:i1 + i + 1] = c
            err = (w - q) / U1[:, i, i].view(L, 1, 1)
            W1[:, :, i:] -= err * U1[:, i:i + 1, i:]
            E1[:, :, i:i + 1] = err
        W[:, :, i2:] -= E1 @ U[:, i1:i2, i2:]
    info = {"damping_lambda": lam.tolist(), "dead_columns": int(dead.sum())}
    if encode:
        info["encoding"] = {"codes": codes, "block_scale": blocks.to(torch.float8_e4m3fn),
                            "global_scale": gs.reshape(L)}
    return Q, info


def output_error(W: Tensor, W_hat: Tensor, H: Tensor) -> Tensor:
    """tr((W - W_hat) H (W - W_hat)^T) per layer: ||X (W - W_hat)^T||_F^2 / N for H = X^T X / N."""
    E = (W - W_hat).double()
    return ((E @ H.double()) * E).sum(dim=(-2, -1))


# ======================================================================================
# Statistics
# ======================================================================================
def excess_kurtosis(x: Tensor) -> float:
    x = x.detach().double().flatten()
    x = x - x.mean()
    m2 = x.pow(2).mean()
    return (x.pow(4).mean() / m2.pow(2) - 3.0).item() if m2 > 0 else 0.0


def max_over_rms(x: Tensor) -> float:
    x = x.detach().double()
    rms = x.pow(2).mean().sqrt()
    return (x.abs().max() / rms).item() if rms > 0 else 0.0


def group_max_over_rms(W: Tensor, group: int = 16) -> float:
    grouped = W.detach().double().reshape(*W.shape[:-1], -1, group)
    rms = grouped.pow(2).mean(-1).sqrt().clamp_min(1e-30)
    return (grouped.abs().amax(-1) / rms).mean().item()


def stable_rank(W: Tensor) -> float:
    W = W.detach().double()
    return (W.pow(2).sum() / torch.linalg.matrix_norm(W, ord=2).pow(2)).item()


def _summary(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "min": float(np.min(values)), "max": float(np.max(values))}


# ======================================================================================
# Data
# ======================================================================================
def load_data(directory: Path, *, verify: bool = True, expected: dict | None = EXPECTED_SHA256) -> h.ShardData:
    """Read-only shards. With ``verify``, every file's SHA-256 is checked against the manifest and
    the manifest against ``expected``."""
    data = h.ShardData(Path(directory), verify=verify)
    if expected:
        found = {k: v["sha256"] for k, v in data.manifest["splits"].items()}
        if any(found.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Shard SHA-256 values differ from the Cycle 3 pins: {found}")
    return data


def calibration_windows(data: h.ShardData, sequences: int, seq_len: int, seed: int) -> dict:
    """Disjoint calibration windows from ``train.bin`` only (non-overlapping, seeded)."""
    train = data.splits["train"]
    available = len(train) // seq_len
    if sequences > available:
        raise ValueError("Not enough training tokens for the calibration set")
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.choice(available, size=sequences, replace=False)) * seq_len
    tokens = np.stack([np.asarray(train[s:s + seq_len], dtype=np.int64) for s in starts])
    return {"split": "train", "seed": seed, "sequences": sequences, "seq_len": seq_len,
            "tokens": int(tokens.size), "starts": [int(s) for s in starts], "array": tokens}


def contiguous_windows(tokens: np.ndarray, sequences: int, seq_len: int) -> np.ndarray:
    return np.asarray(tokens[:sequences * seq_len], dtype=np.int64).reshape(sequences, seq_len)


@torch.no_grad()
def evaluate_split(model, tokens: np.ndarray, seq_len: int, batch: int, device: torch.device,
                   max_windows: int | None = None) -> dict:
    """Mean next-token NLL over disjoint ``seq_len`` windows covering the split from token 0."""
    windows = (len(tokens) - 1) // seq_len
    if max_windows is not None:
        windows = min(windows, max_windows)
    offsets = np.arange(seq_len + 1)
    total = torch.zeros((), dtype=torch.float64, device=device)
    for first in range(0, windows, batch):
        starts = np.arange(first, min(first + batch, windows)) * seq_len
        chunk = torch.from_numpy(tokens[starts[:, None] + offsets].astype(np.int64)).to(device)
        total += h._loss(model, chunk[:, :-1], chunk[:, 1:], reduction="sum").double()
    count = windows * seq_len
    nll = (total / count).item()
    return {"nll": nll, "ppl": math.exp(nll), "tokens": count, "windows": windows,
            "span": [0, count + 1], "split_tokens": int(len(tokens))}


def quantized_linears(model) -> dict[str, torch.nn.Linear]:
    return {f"blocks.{i}.{kind}": getattr(block, kind) for i, block in enumerate(model.blocks)
            for kind in LINEAR_KINDS}


@torch.no_grad()
def collect_hessians(model, windows: np.ndarray, batch: int, device: torch.device,
                     *, autocast: bool | None = None) -> dict[str, Tensor]:
    """H = X^T X / N (FP64, strictly PSD) of the input of every quantized linear, from forward hooks."""
    linears = quantized_linears(model)
    sums = {name: torch.zeros(m.in_features, m.in_features, dtype=torch.float64, device=device)
            for name, m in linears.items()}
    count = {"tokens": 0}
    use_autocast = (device.type == "cuda") if autocast is None else autocast

    def hook(name):
        def accumulate(module, inputs):
            with torch.autocast(device_type=device.type, enabled=False):
                x = inputs[0].detach().reshape(-1, module.in_features).double()
                sums[name].add_(x.T @ x)
        return accumulate

    handles = [m.register_forward_pre_hook(hook(name)) for name, m in linears.items()]
    try:
        model.eval()
        for first in range(0, len(windows), batch):
            x = torch.from_numpy(windows[first:first + batch]).to(device)
            count["tokens"] += x.numel()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
                model(x)
    finally:
        for handle in handles:
            handle.remove()
    return {name: 0.5 * (s + s.T) / count["tokens"] for name, s in sums.items()}


# ======================================================================================
# Methods
# ======================================================================================
METHODS = ("rtn", "hadamard_b16_rtn", "hadamard_b64_rtn", "rand_hadamard_b64_rtn", "adaptive_butterfly_b16_rtn",
           "gptq", "hadamard_b16_gptq", "hadamard_b64_gptq", "rand_hadamard_b64_gptq", "adaptive_butterfly_b16_gptq")


def parse_method(method: str) -> tuple[str, str]:
    if method in ("rtn", "gptq"):
        return "identity", method
    family, _, rounding = method.rpartition("_")
    if family not in ROTATION_FAMILIES or rounding not in ("rtn", "gptq"):
        raise ValueError(f"Unknown method {method!r}")
    return family, rounding


def quantize_stack(W: Tensor, H: Tensor | None, fmt: QuantFormat, rotation: Rotation, rounding: str):
    """Effective weights Q(W R^T) R of a stack, rounded by RTN or by GPTQ on H_R = R H R^T."""
    Wr = rotation.forward(W)
    info = {}
    if rounding == "gptq":
        Qr, info = gptq_quantize(Wr, rotation.rotate_hessian(H), fmt)
    else:
        Qr = quantize_rtn(Wr, fmt)
    return rotation.inverse(Qr), info


# ======================================================================================
# Synthetic checks
# ======================================================================================
def correlated_inputs(tokens: int, n: int, condition: float, seed: int, device=None,
                      dtype=torch.float64) -> Tensor:
    """Gaussian X with covariance Q diag(logspace(0, -log10 kappa)) Q^T (kappa = ``condition``)."""
    g = torch.Generator().manual_seed(seed)
    Qm, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    eig = torch.logspace(0, -math.log10(condition), n, dtype=torch.float64)
    X = torch.randn(tokens, n, generator=g, dtype=torch.float64) * eig.sqrt() @ Qm.T
    return X.to(device=device, dtype=dtype)


def rotation_checks(device: torch.device, sizes=(512, 1408), seed: int = 0) -> dict:
    """Orthogonality ||R^T R - I||_F and the forward equivalence of every rotation family."""
    out = []
    for n in sizes:
        g = torch.Generator().manual_seed(seed + n)
        X = torch.randn(256, n, generator=g, dtype=torch.float64)
        W = torch.randn(128, n, generator=g, dtype=torch.float64)
        angles = (torch.rand(n // 16, 4, 8, generator=g, dtype=torch.float64) * 2 * math.pi)
        rotations = {"hadamard_b16": BlockHadamard(16), "hadamard_b64": BlockHadamard(64),
                     "rand_hadamard_b64": SignHadamard(64, random_signs((n,), seed)),
                     "adaptive_butterfly_b16 (random angles)": Butterfly(16, angles)}
        for name, rot in rotations.items():
            row = {"family": name, "n": n}
            for dtype, label in ((torch.float64, "fp64"), (torch.float32, "fp32")):
                x, w = X.to(device, dtype), W.to(device, dtype)
                if isinstance(rot, SignHadamard):
                    rot = SignHadamard(64, rot.signs.squeeze(-2).to(device))
                elif isinstance(rot, Butterfly):
                    rot = Butterfly(16, rot.angles.to(device))
                R = rot.matrix(n, dtype=dtype, device=device)
                row[f"orthogonality_fro_{label}"] = torch.linalg.matrix_norm(R.T @ R - torch.eye(n, dtype=dtype, device=device)).item()
                y = x @ w.T
                yr = rot.forward(x) @ rot.forward(w).T
                row[f"equivalence_rel_{label}"] = ((y - yr).norm() / y.norm()).item()
                row[f"inverse_rel_{label}"] = ((rot.inverse(rot.forward(w)) - w).norm() / w.norm()).item()
                Hm = x.T @ x / x.shape[0]
                row[f"hessian_identity_rel_{label}"] = ((rot.rotate_hessian(Hm) - R @ Hm @ R.T).norm() / Hm.norm()).item()
            out.append(row)
    passed = all(r["orthogonality_fro_fp64"] <= 1e-6 and r["orthogonality_fro_fp32"] <= 1e-4
                 and r["equivalence_rel_fp32"] < 1e-5 and r["equivalence_rel_fp64"] < 1e-12 for r in out)
    return {"rows": out, "passed": passed,
            "criteria": "||R^T R - I||_F <= 1e-6 (FP64); forward equivalence < 1e-5 (FP32), < 1e-12 (FP64)"}


def gptq_synthetic_check(device: torch.device, *, rows: int = 128, cols: int = 256, tokens: int = 4096,
                         condition: float = 1e3, seed: int = 0) -> dict:
    """GPTQ versus RTN output error ||X (W - W_hat)^T||_F^2 on correlated Gaussian inputs."""
    X = correlated_inputs(tokens, cols, condition, seed, device, torch.float32)
    g = torch.Generator().manual_seed(seed + 1)
    W = (torch.randn(1, rows, cols, generator=g) * 0.02).to(device)
    H = (X.double().T @ X.double() / tokens).unsqueeze(0)
    results = []
    for name, fmt in FORMATS.items():
        rtn = output_error(W, quantize_rtn(W, fmt), H).item()
        gptq = output_error(W, gptq_quantize(W, H, fmt)[0], H).item()
        rot = BlockHadamard(64)
        rot_gptq = output_error(W, quantize_stack(W, H, fmt, rot, "gptq")[0], H).item()
        results.append({"format": name, "rtn": rtn, "gptq": gptq, "hadamard_b64_gptq": rot_gptq,
                        "gptq_over_rtn": gptq / rtn, "gptq_strictly_better": gptq < rtn})
    return {"rows": rows, "cols": cols, "tokens": tokens, "condition": condition, "seed": seed,
            "results": results, "passed": all(r["gptq_strictly_better"] for r in results)}


def synthetic_stress_test(device: torch.device, *, rows: int = 512, cols: int = 1024, tokens: int = 16384,
                          outlier_channels: int = 8, outlier_gain: float = 25.0, condition: float = 1e4,
                          seed: int = 0, butterfly_steps: int = 150) -> dict:
    """Heavy-tailed weights (Student-t, df 3) with spiked input channels and kappa = 1e4 inputs.

    A spiked input channel puts one huge value in every group that contains it. The group scale
    follows that value, and the other weights of the group round to zero ("zero collapse").
    Rotations spread the spike over a block; GPTQ feeds the rounding error into later columns.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(rows, cols, generator=g, dtype=torch.float64)
    chi = torch.randn(rows, cols, 3, generator=g, dtype=torch.float64).pow(2).sum(-1) / 3.0
    W = t / chi.sqrt() * 0.02  # Student-t with 3 degrees of freedom
    spikes = torch.randperm(cols, generator=g)[:outlier_channels]
    W[:, spikes] *= outlier_gain
    W = W.float().unsqueeze(0).to(device)
    X = correlated_inputs(tokens, cols, condition, seed + 1, device, torch.float64)
    H = (X.T @ X / tokens).unsqueeze(0)
    eig = torch.linalg.eigvalsh(H[0])
    reference = output_error(W, torch.zeros_like(W), H).item()
    stats = {"weight_excess_kurtosis": excess_kurtosis(W), "weight_max_over_rms": max_over_rms(W),
             "hessian_condition": (eig.max() / eig.min()).item(), "spiked_channels": spikes.tolist(),
             "rows": rows, "cols": cols, "tokens": tokens, "outlier_gain": outlier_gain, "seed": seed}
    rows_out = []
    for fmt_name, fmt in FORMATS.items():
        cache = {}
        for method in METHODS:
            family, rounding = parse_method(method)
            if family not in cache:
                cache[family] = make_rotation(family, W, fmt, seed=seed + 100, butterfly_steps=butterfly_steps)[0]
            rot = cache[family]
            W_eff, _ = quantize_stack(W, H, fmt, rot, rounding)
            Wr = rot.forward(W)
            Qr = rot.forward(W_eff)
            nonzero = Wr != 0
            rows_out.append({
                "format": fmt_name, "method": method,
                "relative_output_error": output_error(W, W_eff, H).item() / reference,
                "relative_weight_error": ((W_eff - W).pow(2).sum() / W.pow(2).sum()).item(),
                "zero_collapse_fraction": ((Qr.abs() < 1e-12) & nonzero).float().sum().item() / max(1, nonzero.sum().item()),
                "rotated_excess_kurtosis": excess_kurtosis(Wr), "rotated_max_over_rms": max_over_rms(Wr),
            })
    by = {(r["format"], r["method"]): r for r in rows_out}
    recovery = {f: {"rtn": by[(f, "rtn")]["relative_output_error"],
                    "best_method": min((m for m in METHODS), key=lambda m: by[(f, m)]["relative_output_error"]),
                    "best": min(by[(f, m)]["relative_output_error"] for m in METHODS),
                    "rand_hadamard_b64_gptq": by[(f, "rand_hadamard_b64_gptq")]["relative_output_error"]}
                for f in FORMATS}
    for f, r in recovery.items():
        r["rtn_over_best"] = r["rtn"] / r["best"]
    return {"setup": stats, "rows": rows_out, "recovery": recovery}


# ======================================================================================
# Native Blackwell FP4 GEMM probe
# ======================================================================================
def to_blocked(scales: Tensor) -> Tensor:
    """cuBLAS 128x4 block-scale swizzle (``SWIZZLE_32_4_4``) of a (rows, cols) scale matrix."""
    rows, cols = scales.shape
    pr, pc = -(-rows // 128) * 128, -(-cols // 4) * 4
    if (pr, pc) != (rows, cols):
        padded = torch.zeros((pr, pc), dtype=torch.uint8, device=scales.device)
        padded[:rows, :cols] = scales.view(torch.uint8)
        scales = padded.view(scales.dtype)
    blocks = scales.view(pr // 128, 128, pc // 4, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _nvfp4_mm(a: dict, b: dict):
    """Native NVFP4 GEMM a @ b^T in BF16 from encoded operands, with both tensor scales."""
    A4, B4 = pack_e2m1(a["codes"]), pack_e2m1(b["codes"])
    sa, sb = to_blocked(a["block_scale"]), to_blocked(b["block_scale"])
    ga, gb = a["global_scale"].float(), b["global_scale"].float()
    if hasattr(F, "scaled_mm"):
        swz = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]
        recipe = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
        return lambda: F.scaled_mm(A4, B4.t(), [sa, ga], recipe, [sb, gb], recipe, swizzle_a=swz, swizzle_b=swz,
                                   output_dtype=torch.bfloat16)
    scale = ga * gb
    return lambda: torch._scaled_mm(A4, B4.t(), sa, sb, out_dtype=torch.bfloat16) * scale


def _williams_latency(functions: dict, rounds: int, iterations: int, warmup: int = 5,
                      flush: Tensor | None = None) -> dict:
    """CUDA-event milliseconds per call. Two treatments, Williams (AB/BA) balanced round order."""
    names = list(functions)
    samples = {name: [] for name in names}
    for r in range(rounds):
        for name in (names if r % 2 == 0 else names[::-1]):
            fn = functions[name]
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            pairs = []
            for _ in range(iterations):
                if flush is not None:
                    flush.zero_()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                pairs.append((start, end))
            torch.cuda.synchronize()
            samples[name].append(sum(s.elapsed_time(e) for s, e in pairs) / iterations)
    return samples


def probe_native_fp4_gemm(device: torch.device, *, shapes=((1024, 4096), (2048, 8192)), rounds: int = 10,
                          iterations: int = 10, seed: int = 0) -> dict:
    """Is native FP4 GEMM available through ``scaled_mm`` here, is it right, and how fast is it?"""
    result = {"torch": torch.__version__, "float4_e2m1fn_x2_dtype": hasattr(torch, "float4_e2m1fn_x2"),
              "float8_e8m0fnu_dtype": hasattr(torch, "float8_e8m0fnu"),
              "scaled_mm_v2_api": hasattr(F, "scaled_mm")}
    codes = torch.arange(16, dtype=torch.uint8).repeat(4)
    result["pack_unpack_roundtrip"] = bool(torch.equal(unpack_e2m1(pack_e2m1(codes)), codes))
    if device.type != "cuda":
        result["status"] = "skipped: CUDA not available"
        return result
    result["device"] = torch.cuda.get_device_name(device)
    result["capability"] = list(torch.cuda.get_device_capability(device))
    g = torch.Generator(device=device).manual_seed(seed)
    M, N, K = 256, 512, 1024
    A = torch.randn(M, K, device=device, generator=g)
    B = torch.randn(N, K, device=device, generator=g)
    # NVFP4: compare against FP32 products of the same dequantized operands.
    ea, eb = nvfp4_encode(A), nvfp4_encode(B)
    reference = nvfp4_decode(ea) @ nvfp4_decode(eb).T
    try:
        out = _nvfp4_mm(ea, eb)()
        torch.cuda.synchronize()
        rel = ((out.float() - reference).norm() / reference.norm()).item()
        result["nvfp4"] = {"supported": True, "shape_mnk": [M, N, K], "relative_error_vs_fp32_dequant": rel,
                           "verified": rel < 1e-2, "api": "F.scaled_mm BlockWise1x16 + TensorWise"
                           if hasattr(F, "scaled_mm") else "torch._scaled_mm",
                           "note": "Output is BF16, so ~2^-9 relative rounding per element is expected."}
    except Exception as error:  # noqa: BLE001 - record the exact failure
        result["nvfp4"] = {"supported": False, "error": f"{type(error).__name__}: {error}"}
    # MXFP4 through the same API.
    ma, mb = mxfp4_encode(A), mxfp4_encode(B)
    mx_reference = ma["values"] @ mb["values"].T
    attempts = {}
    for api in ("F.scaled_mm BlockWise1x32", "torch._scaled_mm"):
        try:
            A4, B4 = pack_e2m1(ma["codes"]), pack_e2m1(mb["codes"])
            sa, sb = to_blocked(ma["block_scale"]), to_blocked(mb["block_scale"])
            if api.startswith("F."):
                out = F.scaled_mm(A4, B4.t(), sa, F.ScalingType.BlockWise1x32, sb, F.ScalingType.BlockWise1x32,
                                  swizzle_a=F.SwizzleType.SWIZZLE_32_4_4, swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                                  output_dtype=torch.bfloat16)
            else:
                out = torch._scaled_mm(A4, B4.t(), sa, sb, out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            attempts[api] = {"supported": True,
                             "relative_error": ((out.float() - mx_reference).norm() / mx_reference.norm()).item()}
        except Exception as error:  # noqa: BLE001
            attempts[api] = {"supported": False, "error": f"{type(error).__name__}: {str(error).splitlines()[0]}"}
    result["mxfp4"] = attempts
    if not result["nvfp4"].get("verified"):
        result["status"] = "native NVFP4 GEMM not available or not verified; no latency is reported"
        return result
    flush = torch.empty(4 * torch.cuda.get_device_properties(device).L2_cache_size,
                        dtype=torch.uint8, device=device)
    latency = []
    for M, K in shapes:
        N = K
        x = torch.randn(M, K, device=device, generator=g, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=device, generator=g, dtype=torch.bfloat16)
        ex, ew = nvfp4_encode(x.float()), nvfp4_encode(w.float())
        fp4 = _nvfp4_mm(ex, ew)
        check = nvfp4_decode(ex) @ nvfp4_decode(ew).T
        shape_error = ((fp4().float() - check).norm() / check.norm()).item()
        del check
        functions = {"bf16_cublas": lambda x=x, w=w: x @ w.T, "nvfp4_scaled_mm": fp4}
        samples = _williams_latency(functions, rounds, iterations, flush=flush)
        flops = 2.0 * M * N * K
        row = {"m": M, "n": N, "k": K, "rounds": rounds, "iterations_per_round": iterations,
               "l2_flush_bytes": flush.numel(),
               "design": "Williams AB/BA balanced rounds; CUDA events; 4x L2 flush before every sample; "
                         "operands pre-quantized (GEMM only; excludes online activation quantization and rotation)",
               "nvfp4_relative_error_vs_fp32_dequant": shape_error}
        for name, values in samples.items():
            row[name] = {"median_ms": median(values), "min_ms": min(values), "max_ms": max(values),
                         "tflops_median": flops / (median(values) * 1e-3) / 1e12, "samples_ms": values}
        ratios = [a / b for a, b in zip(samples["bf16_cublas"], samples["nvfp4_scaled_mm"])]
        row["speedup_bf16_over_nvfp4"] = {"median": median(ratios), "min": min(ratios), "max": max(ratios)}
        latency.append(row)
        del x, w, ex, ew, fp4, functions
    del flush
    result["latency"] = latency
    result["status"] = "native NVFP4 GEMM verified on this device"
    return result


# ======================================================================================
# LM pipeline
# ======================================================================================
MODEL_SPECS = {
    "muon_ss3_54": {"optimizer": "muon_ss3_54", "lr": 0.02, "auxiliary_lr": 0.001},
    "adamw": {"optimizer": "adamw", "lr": 0.001, "auxiliary_lr": None},
}


def _factory(name: str, device: torch.device):
    if name == "adamw":
        return None, "fused AdamW"
    if device.type == "cuda":
        from research.polar_opt import tier2_candidates

        return (lambda: tier2_candidates.ss3_54_blackwell_fp16), "schatten_switch_blackwell SS3_54 FP16 (Triton)"
    from research.polar_opt.baselines import polar_express

    return (lambda: polar_express), "polar_express eager BF16 (CPU stand-in: the SS3_54 kernel needs CUDA)"


def train_model(name: str, config: h.ModelConfig, data: h.ShardData, device: torch.device, *, steps: int,
                seed: int, warmup_steps: int, eval_tokens: int):
    spec = MODEL_SPECS[name]
    factory, orthogonalizer = _factory(spec["optimizer"], device)
    with torch.device(device):
        model = Transformer(config)
    init = h.initial_state(config, seed)
    row = h.run_trial(config, data, device=device, seed=seed, optimizer_name=spec["optimizer"], factory=factory,
                      lr=spec["lr"], auxiliary_lr=spec["auxiliary_lr"] or 2e-3, tokens=steps * config.step_tokens,
                      lr_warmup_steps=warmup_steps, eval_tokens=eval_tokens, init=init,
                      reusable_model=(model, model), log_every=10, telemetry_every=100)
    row["orthogonalizer"] = orthogonalizer
    row["harness_eval_note"] = "val/test loss here are the harness's first-eval_tokens check, not the full test.bin"
    model.eval()
    return model, row


def _stack(names: list[str], tensors: dict[str, Tensor]) -> Tensor:
    return torch.stack([tensors[n] for n in names])


def ptq_study(model, config: h.ModelConfig, data: h.ShardData, device: torch.device, *, calibration: dict,
              val_windows: np.ndarray, eval_batch: int, max_eval_windows: int | None, butterfly_steps: int,
              rotation_seed: int, formats=tuple(FORMATS), methods=METHODS, log=print) -> dict:
    t0 = time.perf_counter()
    linears = quantized_linears(model)
    test = data.splits["test"]
    baseline = evaluate_split(model, test, config.sequence_length, eval_batch, device, max_eval_windows)
    log(f"  baseline test NLL {baseline['nll']:.4f} PPL {baseline['ppl']:.2f} on {baseline['tokens']:,} tokens")
    H_cal = collect_hessians(model, calibration["array"], eval_batch, device)
    H_val = collect_hessians(model, val_windows, eval_batch, device)
    kinds = {kind: [f"blocks.{i}.{kind}" for i in range(config.layers)] for kind in LINEAR_KINDS}
    weights = {n: m.weight.detach().float().clone() for n, m in linears.items()}
    W = {k: _stack(v, weights) for k, v in kinds.items()}
    Hc = {k: _stack(v, H_cal) for k, v in kinds.items()}
    Hv = {k: _stack(v, H_val) for k, v in kinds.items()}
    del H_cal, H_val

    total_params = sum(p.numel() for p in model.parameters())
    quantized_params = sum(m.weight.numel() for m in linears.values())

    # Pre-quantization weight statistics.
    per_layer = []
    for name, w in weights.items():
        per_layer.append({"name": name, "shape": list(w.shape), "excess_kurtosis": excess_kurtosis(w),
                          "max_over_rms": max_over_rms(w), "group16_max_over_rms": group_max_over_rms(w),
                          "stable_rank": stable_rank(w)})
    weight_stats = {
        "excess_kurtosis": _summary([r["excess_kurtosis"] for r in per_layer]),
        "max_over_rms": _summary([r["max_over_rms"] for r in per_layer]),
        "group16_max_over_rms": _summary([r["group16_max_over_rms"] for r in per_layer]),
        "stable_rank": _summary([r["stable_rank"] for r in per_layer]),
        "by_kind": {k: {"excess_kurtosis_mean": float(np.mean([r["excess_kurtosis"] for r in per_layer
                                                               if r["name"].endswith("." + k)])),
                        "max_over_rms_mean": float(np.mean([r["max_over_rms"] for r in per_layer
                                                            if r["name"].endswith("." + k)])),
                        "stable_rank_mean": float(np.mean([r["stable_rank"] for r in per_layer
                                                           if r["name"].endswith("." + k)]))} for k in LINEAR_KINDS},
        "per_layer": per_layer,
    }

    # Rotations: one per (format, kind, family); fitted butterflies are format specific.
    rotations, rotation_info = {}, {}
    for fmt_name in formats:
        fmt = FORMATS[fmt_name]
        for kind in LINEAR_KINDS:
            for family in {parse_method(m)[0] for m in methods}:
                key = (fmt_name, kind, family)
                seed = rotation_seed + 1000 * LINEAR_KINDS.index(kind)
                if family != "adaptive_butterfly_b16":
                    shared = (None, kind, family)
                    if shared not in rotations:
                        rotations[shared] = make_rotation(family, W[kind], fmt, seed=seed)[0]
                    rotations[key] = rotations[shared]
                else:
                    rotations[key], info = make_rotation(family, W[kind], fmt, butterfly_steps=butterfly_steps)
                    rotation_info[f"{fmt_name}/{kind}"] = info
    log(f"  rotations ready ({time.perf_counter() - t0:.0f}s)")

    stats_format = "nvfp4_g16" if "nvfp4_g16" in formats else formats[0]

    def rotation_summary(family: str, fmt_name: str = stats_format) -> dict:
        kurt, mor, gmor, hdiag = [], [], [], []
        for kind in LINEAR_KINDS:
            rot = rotations[(fmt_name, kind, family)]
            Wr = rot.forward(W[kind])
            Hr = rot.rotate_hessian(Hc[kind])
            for i in range(config.layers):
                kurt.append(excess_kurtosis(Wr[i]))
                mor.append(max_over_rms(Wr[i]))
                gmor.append(group_max_over_rms(Wr[i]))
                d = Hr[i].diagonal()
                hdiag.append((d.max() / d.mean()).item())
        return {"weight_excess_kurtosis": _summary(kurt), "weight_max_over_rms": _summary(mor),
                "weight_group16_max_over_rms": _summary(gmor), "input_energy_max_over_mean": _summary(hdiag)}

    rotation_stats = {family: rotation_summary(family) for family in ROTATION_FAMILIES
                      if (formats[0], LINEAR_KINDS[0], family) in rotations}
    rotation_stats["note"] = ("Computed on W R^T (and R H R^T for input energy). The fitted butterfly is the one "
                              "fitted for " + stats_format + ". Stable rank is unchanged by an orthogonal R.")

    def apply(eff: dict[str, Tensor] | None):
        with torch.no_grad():
            for kind, names in kinds.items():
                source = eff[kind] if eff is not None else W[kind]
                for i, name in enumerate(names):
                    linears[name].weight.copy_(source[i])

    grid = []
    for fmt_name in formats:
        fmt = FORMATS[fmt_name]
        for method in methods:
            t1 = time.perf_counter()
            family, rounding = parse_method(method)
            eff, damping, cal_rel, val_rel, cal_mse, val_mse, bits_num = {}, [], [], [], [], [], 0.0
            for kind in LINEAR_KINDS:
                rot = rotations[(fmt_name, kind, family)]
                eff[kind], info = quantize_stack(W[kind], Hc[kind], fmt, rot, rounding)
                damping += info.get("damping_lambda", [])
                ec, ev = output_error(W[kind], eff[kind], Hc[kind]), output_error(W[kind], eff[kind], Hv[kind])
                rc = output_error(W[kind], torch.zeros_like(W[kind]), Hc[kind])
                rv = output_error(W[kind], torch.zeros_like(W[kind]), Hv[kind])
                cal_rel += (ec / rc).tolist()
                val_rel += (ev / rv).tolist()
                rows_out = W[kind].shape[1]
                cal_mse += (ec / rows_out).tolist()
                val_mse += (ev / rows_out).tolist()
                numel = W[kind][0].numel()
                extra = (rot.parameter_bits_per_input or 0.0) / rows_out  # per weight
                bits_num += config.layers * numel * (fmt.bits_per_weight(numel) + extra)
            apply(eff)
            result = evaluate_split(model, test, config.sequence_length, eval_batch, device, max_eval_windows)
            apply(None)
            if result["tokens"] != baseline["tokens"]:
                raise RuntimeError("Evaluation token count changed between runs")
            grid.append({
                "format": fmt_name, "method": method, "rotation": family, "rounding": rounding,
                "test_nll": result["nll"], "test_ppl": result["ppl"], "test_tokens": result["tokens"],
                "delta_nll": result["nll"] - baseline["nll"], "delta_ppl": result["ppl"] - baseline["ppl"],
                "relative_delta_ppl_pct": 100.0 * (result["ppl"] / baseline["ppl"] - 1.0),
                "calibration_relative_output_error_mean": float(np.mean(cal_rel)),
                "val_relative_output_error_mean": float(np.mean(val_rel)),
                "calibration_layer_output_mse_mean": float(np.mean(cal_mse)),
                "val_layer_output_mse_mean": float(np.mean(val_mse)),
                "effective_bits_per_weight": bits_num / quantized_params,
                "format_bits_per_weight": fmt.bits_per_weight(),
                "gptq_damping_max": max(damping) if damping else None,
                "seconds": time.perf_counter() - t1,
            })
            r = grid[-1]
            log(f"  {fmt_name:<14} {method:<28} PPL {r['test_ppl']:9.2f} dNLL {r['delta_nll']:+.4f} "
                f"cal err {r['calibration_relative_output_error_mean']:.4f} ({r['seconds']:.1f}s)")
    return {
        "baseline_test": baseline,
        "parameters": {"total": total_params, "quantized": quantized_params,
                       "quantized_fraction": quantized_params / total_params,
                       "kept_high_precision": "token embedding (tied LM head) and all RMSNorm weights"},
        "weight_stats": weight_stats, "rotation_stats": rotation_stats, "butterfly_fits": rotation_info,
        "grid": grid, "seconds": time.perf_counter() - t0,
    }


# ======================================================================================
# Results
# ======================================================================================
REQUIRED_GRID_KEYS = ("format", "method", "test_nll", "test_ppl", "test_tokens", "delta_nll", "delta_ppl",
                      "relative_delta_ppl_pct", "calibration_relative_output_error_mean",
                      "val_relative_output_error_mean", "effective_bits_per_weight")


def validate_results(payload: dict, *, require_models=tuple(MODEL_SPECS)) -> None:
    """Raise if a result is incomplete, leaks evaluation data or contains a non-finite number."""
    json.dumps(payload, allow_nan=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected schema version")
    for key in ("meta", "data", "formats", "rotation_checks", "gptq_synthetic_check", "native_fp4_gemm_probe",
                "synthetic_stress", "models", "summary"):
        if key not in payload:
            raise ValueError(f"Missing top-level key {key!r}")
    meta = payload["meta"]
    for key in ("git_sha", "git_dirty", "command", "environment", "gpu_telemetry", "loadavg_start", "loadavg_end"):
        if key not in meta:
            raise ValueError(f"Missing meta key {key!r}")
    data = payload["data"]
    if data["calibration"]["split"] != "train" or data["evaluation_split"] != "test":
        raise ValueError("Calibration must use train.bin and evaluation test.bin")
    for name in require_models:
        model = payload["models"][name]
        grid = model["study"]["grid"]
        tokens = model["study"]["baseline_test"]["tokens"]
        if not grid:
            raise ValueError(f"{name}: empty grid")
        for row in grid:
            missing = [k for k in REQUIRED_GRID_KEYS if k not in row]
            if missing or row["test_tokens"] != tokens:
                raise ValueError(f"{name}: bad grid row {row.get('format')}/{row.get('method')}: {missing}")


def summarize(models: dict) -> dict:
    tables = {}
    for name, entry in models.items():
        study = entry["study"]
        table = {}
        for row in study["grid"]:
            table.setdefault(row["format"], {})[row["method"]] = {
                k: row[k] for k in ("test_nll", "test_ppl", "delta_nll", "delta_ppl", "relative_delta_ppl_pct",
                                    "effective_bits_per_weight")}
        best = {f: min(t, key=lambda m: t[m]["test_nll"]) for f, t in table.items()}
        tables[name] = {"baseline_test_nll": study["baseline_test"]["nll"],
                        "baseline_test_ppl": study["baseline_test"]["ppl"],
                        "test_tokens": study["baseline_test"]["tokens"], "table": table, "best_method": best}
    comparison = []
    if {"muon_ss3_54", "adamw"} <= tables.keys():
        a, b = tables["muon_ss3_54"]["table"], tables["adamw"]["table"]
        for f in a:
            for m in a[f]:
                if m in b.get(f, {}):
                    comparison.append({"format": f, "method": m,
                                       "muon_delta_nll": a[f][m]["delta_nll"], "adamw_delta_nll": b[f][m]["delta_nll"],
                                       "muon_minus_adamw_delta_nll": a[f][m]["delta_nll"] - b[f][m]["delta_nll"],
                                       "muon_test_ppl": a[f][m]["test_ppl"], "adamw_test_ppl": b[f][m]["test_ppl"]})
    return {"per_model": tables, "muon_vs_adamw_quantizability": comparison,
            "note": "delta = quantized minus the same model in BF16 on the same held-out test tokens. "
                    "A negative muon_minus_adamw_delta_nll means the Muon model lost less. "
                    "Caveat: compares iso-token (320-step, single-seed) checkpoints with unequal baseline PPL "
                    "(Muon vs undertrained AdamW), not iso-loss quality-matched checkpoints."}


def environment(device: torch.device) -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "host_cpus": os.cpu_count(),
           "numpy": np.__version__,
           "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "OMP_NUM_THREADS")}}
    try:
        import triton

        env["triton"] = triton.__version__
    except ImportError:
        env["triton"] = None
    if device.type == "cuda":
        env["device"] = torch.cuda.get_device_name(device)
        env["capability"] = list(torch.cuda.get_device_capability(device))
        try:
            env["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            env["driver"] = None
    return env


def run_experiment(args) -> dict:
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    quick = args.quick
    log = (lambda *a, **k: None) if args.silent else (lambda *a, **k: print(*a, **k, flush=True))
    started = time.perf_counter()
    meta = {**h.git_info(), "command": [sys.executable, *sys.argv], "args": {k: str(v) for k, v in vars(args).items()},
            "environment": environment(device), "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "loadavg_start": list(os.getloadavg()), "gpu_telemetry": {"start": h.gpu_telemetry(device)}}
    log(f"git {meta['git_sha']} dirty={meta['git_dirty']} device={meta['environment'].get('device', 'cpu')}")
    data = load_data(args.data_dir, verify=not args.no_verify, expected=None if args.no_verify else EXPECTED_SHA256)
    config = h.PRESETS[args.preset]
    torch.manual_seed(args.seed)

    payload = {"schema_version": SCHEMA_VERSION, "task": "Cycle 3 T5-01: NVFP4 / MXFP4 / INT2 micro-rotated PTQ",
               "quick": quick, "meta": meta}
    payload["formats"] = {name: {**asdict(fmt), "bits_per_weight": fmt.bits_per_weight(),
                                 "bits_per_weight_512x512": fmt.bits_per_weight(512 * 512)}
                          for name, fmt in FORMATS.items()}

    t = time.perf_counter()
    payload["rotation_checks"] = rotation_checks(device, sizes=(128, 192) if quick else (512, 1408))
    log(f"rotation checks passed={payload['rotation_checks']['passed']} ({time.perf_counter() - t:.1f}s)")
    t = time.perf_counter()
    payload["gptq_synthetic_check"] = gptq_synthetic_check(device, **({"rows": 32, "cols": 128, "tokens": 1024}
                                                                       if quick else {}))
    log(f"GPTQ synthetic check passed={payload['gptq_synthetic_check']['passed']} ({time.perf_counter() - t:.1f}s)")
    t = time.perf_counter()
    payload["native_fp4_gemm_probe"] = probe_native_fp4_gemm(
        device, **({"shapes": ((256, 512),), "rounds": 2, "iterations": 3} if quick else {}))
    log(f"native FP4 probe: {payload['native_fp4_gemm_probe'].get('status')} ({time.perf_counter() - t:.1f}s)")
    t = time.perf_counter()
    payload["synthetic_stress"] = synthetic_stress_test(
        device, **({"rows": 64, "cols": 128, "tokens": 2048, "butterfly_steps": 10} if quick
                   else {"butterfly_steps": args.butterfly_steps}))
    log(f"synthetic stress test ({time.perf_counter() - t:.1f}s): "
        + ", ".join(f"{f} rtn/best={r['rtn_over_best']:.1f}" for f, r in payload["synthetic_stress"]["recovery"].items()))

    calibration = calibration_windows(data, args.calibration_sequences, config.sequence_length, args.calibration_seed)
    val_windows = contiguous_windows(data.splits["val"], args.val_check_sequences, config.sequence_length)
    payload["data"] = {
        **data.summary(), "directory": str(args.data_dir), "sha256_verified": not args.no_verify,
        "expected_sha256": EXPECTED_SHA256 if not args.no_verify else None,
        "calibration": {k: v for k, v in calibration.items() if k != "array"},
        "val_check": {"split": "val", "sequences": args.val_check_sequences, "seq_len": config.sequence_length,
                      "span": [0, args.val_check_sequences * config.sequence_length],
                      "use": "layer output error report only; no selection or tuning"},
        "evaluation_split": "test", "training_split": "train",
        "rotation_tuning_data": "none (adaptive butterfly fits weights only)",
    }
    models = {}
    for name in MODEL_SPECS:
        t = time.perf_counter()
        log(f"[{name}] training {args.steps} steps from seed {args.seed}")
        model, row = train_model(name, config, data, device, steps=args.steps, seed=args.seed,
                                 warmup_steps=args.warmup_steps, eval_tokens=args.harness_eval_tokens)
        log(f"[{name}] trained in {time.perf_counter() - t:.0f}s: last loss {row['last_train_loss']:.4f}")
        study = ptq_study(model, config, data, device, calibration=calibration, val_windows=val_windows,
                          eval_batch=config.micro_batch, max_eval_windows=args.max_eval_windows,
                          butterfly_steps=args.butterfly_steps, rotation_seed=args.seed * 7919, log=log)
        models[name] = {"spec": MODEL_SPECS[name], "training": row, "study": study,
                        "preset": args.preset, "config": asdict(config)}
        meta["gpu_telemetry"][f"after_{name}"] = h.gpu_telemetry(device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload["models"] = models
    payload["summary"] = summarize(models)
    meta["gpu_telemetry"]["end"] = h.gpu_telemetry(device)
    meta["loadavg_end"] = list(os.getloadavg())
    meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["total_seconds"] = time.perf_counter() - started
    validate_results(payload)
    return payload


CANONICAL_OUTPUT = ROOT / "results" / "cycle3_t5_01_nvfp4_ptq.json"
QUICK_OUTPUT = ROOT / "results" / "cycle3_t5_01_nvfp4_ptq_quick.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preset", default="50m", choices=sorted(h.PRESETS))
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--calibration-sequences", type=int, default=128)
    parser.add_argument("--calibration-seed", type=int, default=20261126)
    parser.add_argument("--val-check-sequences", type=int, default=32)
    parser.add_argument("--max-eval-windows", type=int, default=None, help="limit test windows (default: all)")
    parser.add_argument("--harness-eval-tokens", type=int, default=1 << 20)
    parser.add_argument("--butterfly-steps", type=int, default=150)
    parser.add_argument("--no-verify", action="store_true", help="skip SHA-256 checks (synthetic shards only)")
    parser.add_argument("--quick", action="store_true", help="tiny model, few steps, short evaluation")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _is_canonical_args(args: argparse.Namespace) -> bool:
    return (
        not args.quick
        and not args.no_verify
        and args.preset == "50m"
        and args.steps == 320
        and args.warmup_steps == 32
        and args.seed == 1
        and args.calibration_sequences == 128
        and args.calibration_seed == 20261126
        and args.val_check_sequences == 32
        and args.max_eval_windows is None
        and args.harness_eval_tokens == (1 << 20)
        and args.butterfly_steps == 150
    )


def _resolve_output(args: argparse.Namespace) -> Path:
    canonical = _is_canonical_args(args)
    if args.output is None:
        if args.quick:
            return QUICK_OUTPUT
        if canonical:
            return CANONICAL_OUTPUT
        raise ValueError(
            "Non-canonical run requires an explicit --output path to avoid "
            f"overwriting {CANONICAL_OUTPUT.name}"
        )
    if args.output.resolve() == CANONICAL_OUTPUT.resolve() and not canonical:
        raise ValueError(
            f"Refusing to overwrite canonical {CANONICAL_OUTPUT.name} from a --quick or "
            "non-canonical run; choose a different --output path"
        )
    return args.output


def parse_args(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.data_dir is None:
        args.data_dir = default_data_dir()
    if args.quick:
        args.preset, args.steps, args.warmup_steps = "tiny", 6, 2
        args.calibration_sequences, args.val_check_sequences = 8, 4
        args.max_eval_windows = args.max_eval_windows or 64
        args.harness_eval_tokens, args.butterfly_steps = 4096, 10
    try:
        args.output = _resolve_output(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    payload = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(args.output)
    if not args.silent:
        print(f"Wrote {args.output}")
    return payload


if __name__ == "__main__":
    main()
