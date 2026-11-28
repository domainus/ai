"""Wide orthogonal rotations and rotated-space NVFP4 W4A4 fake quantization (Cycle 6 C6-01).

C5-02 / C5-03 rotate the input of a quantized linear inside one NVFP4 group of 16 (``hadamard_b16``,
``adaptive_butterfly_b16``). This module adds rotations that also mix across groups, and the
straight-through (STE) training wrapper that quantizes in those coordinates.

Rotation families. Every family is ``R = R16 . C``: a cross-block mix ``C`` and a block-16 rotation
``R16``. ``forward(x) = x R^T`` and ``inverse(z) = z R`` along the last dimension, as in
``micro_rotated_ptq``.

    family                      C (``CrossMix``)                     R16
    identity                    -                                    -
    hadamard_b16                -                                    Hadamard 16
    adaptive_butterfly_b16      -                                    fitted butterfly 16
    hadamard_b64                (I (x) H_4) (x) I_16                 Hadamard 16   (= block-64 Walsh-Hadamard)
    kron_hadamard_full          (A_m (x) H_{B/16}) (x) I_16          Hadamard 16   (= A_m (x) H_B over all K inputs)
    cross_b64_butterfly_b16     (I (x) H_4) (x) I_16                 fitted butterfly 16
    cross_full_butterfly_b16    (A_m (x) H_{B/16}) (x) I_16          fitted butterfly 16

``kron_hadamard_full`` factors the width as ``K = m B`` with ``B = min(128, largest power of two that
divides K)``. ``A_m = DCT4_odd (x) H_pow2`` for ``m = odd * pow2``. For a power-of-two ``K`` it is the
full-width Sylvester Walsh-Hadamard transform. For other widths (1408 = 11 x 128, 768 = 6 x 128) the
odd factor uses the orthonormal DCT-IV matrix, which is symmetric, so ``R`` stays a symmetric
involution. No Hadamard matrix of odd order exists.

Hardware path. ``C`` runs in FP32 PyTorch (``CrossInput``); ``R16`` runs in registers inside the fused
Triton NVFP4 quantizer of ``nvfp4_inference_blackwell``. ``C`` is one or two small matrix products and
is not fused into the kernel.

Training path. ``RotatedFakeQuantLinear`` computes ``Q(x R^T) Q(W R^T)^T`` with round-to-nearest NVFP4
and identity gradients through both quantizers. With the identity rotation it is exactly
``bench_c5_02_isoloss_ptq_mitigation.FakeQuantLinear``. Rotated families use dense matrix products
(``WideHadamard.dense``), so the rotation adds little to the cost of the fake quantizer itself.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from research.quant_ptq import bench_isoloss_ptq as c4
from research.quant_ptq import bench_nvfp4_e2e as e2e
from research.quant_ptq import micro_rotated_ptq as q
from research.quant_ptq import nvfp4_inference_blackwell as nb

FMT = q.FORMATS["nvfp4_g16"]
GROUP = 16
FULL_BLOCK_CAP = 128
AWQ_PREFIX = "awq_w4a4_"

# family -> (cross-block mix, block-16 rotation)
FAMILIES = {
    "identity": (None, "identity"),
    "hadamard_b16": (None, "hadamard_b16"),
    "adaptive_butterfly_b16": (None, "adaptive_butterfly_b16"),
    "hadamard_b64": ("b64", "hadamard_b16"),
    "kron_hadamard_full": ("full", "hadamard_b16"),
    "cross_b64_butterfly_b16": ("b64", "adaptive_butterfly_b16"),
    "cross_full_butterfly_b16": ("full", "adaptive_butterfly_b16"),
}
WIDE_FAMILIES = tuple(f for f, (cross, _) in FAMILIES.items() if cross is not None)
QAT_FAMILIES = tuple(f for f, (_, inner) in FAMILIES.items() if inner != "adaptive_butterfly_b16")  # no fitted angles
KERNEL_ROTATION = {"identity": "none", "hadamard_b16": "hadamard_b16",
                   "adaptive_butterfly_b16": "adaptive_butterfly_b16"}


# ======================================================================================
# Method names
# ======================================================================================
def method_name(family: str, rounding: str, awq: bool = False) -> str:
    if family not in FAMILIES or rounding not in ("rtn", "gptq"):
        raise ValueError(f"Unknown family or rounding: {family!r}, {rounding!r}")
    base = rounding if family == "identity" else f"{family}_{rounding}"
    return AWQ_PREFIX + base if awq else base


def parse_method(method: str) -> tuple[str, str, bool]:
    """(rotation family, rounding, activation-aware scaling) of a method name."""
    awq = method.startswith(AWQ_PREFIX)
    base = method.removeprefix(AWQ_PREFIX)
    if base in ("rtn", "gptq"):
        return "identity", base, awq
    family, _, rounding = base.rpartition("_")
    if family not in FAMILIES or family == "identity" or rounding not in ("rtn", "gptq"):
        raise ValueError(f"Unknown method {method!r}")
    return family, rounding, awq


def is_wide(method: str) -> bool:
    return parse_method(method)[0] in WIDE_FAMILIES


def kernel_rotation(family: str) -> str:
    """The in-kernel block-16 rotation of ``nvfp4_inference_blackwell`` that the family uses."""
    return KERNEL_ROTATION[FAMILIES[family][1]]


# ======================================================================================
# Orthogonal factors
# ======================================================================================
def hadamard_matrix(n: int, dtype=torch.float64) -> Tensor:
    """Orthonormal Sylvester Walsh-Hadamard matrix (symmetric), the matrix of ``q.fwht(., n)``."""
    return q.fwht(torch.eye(n, dtype=dtype), n)


def dct4_matrix(m: int, dtype=torch.float64) -> Tensor:
    """Orthonormal DCT-IV matrix sqrt(2 / m) cos(pi (2j + 1)(2k + 1) / (4 m)): symmetric and involutory."""
    index = 2 * torch.arange(m, dtype=torch.float64) + 1
    return (math.sqrt(2.0 / m) * torch.cos(math.pi * torch.outer(index, index) / (4 * m))).to(dtype)


def full_factors(n: int) -> tuple[int, int, int]:
    """(block B, pow2, odd) with n = odd * pow2 * B and B = min(128, largest power of two dividing n)."""
    block = min(FULL_BLOCK_CAP, n & -n)
    if block < GROUP:
        raise ValueError(f"{n} inputs are not a multiple of the NVFP4 group size {GROUP}")
    m = n // block
    pow2 = m & -m
    return block, pow2, m // pow2


class CrossMix(q.Rotation):
    """``(A (x) H_{block/16}) (x) I_16``: mixes NVFP4 groups, keeps the position inside a group.

    ``block`` inputs form one Hadamard block; ``mix`` (m x m, symmetric orthogonal, or None for the
    identity) mixes the ``m = n / block`` blocks. The map is a symmetric involution, so
    ``inverse = forward``.
    """

    def __init__(self, n: int, block: int, mix: Tensor | None = None, name: str = "cross"):
        if block & (block - 1) or block < GROUP or n % block:
            raise ValueError(f"Block {block} must be a power of two >= {GROUP} that divides {n}")
        if mix is not None and tuple(mix.shape) != (n // block, n // block):
            raise ValueError("mix must be (n / block, n / block)")
        self.n, self.block, self.name = n, block, name
        self._mix = None if mix is None else mix.double().cpu()
        self._cache: dict = {}

    @property
    def identity(self) -> bool:
        return self.block == GROUP and self._mix is None

    def _matrix(self, kind: str, like: Tensor) -> Tensor:
        key = (kind, like.dtype, like.device)
        found = self._cache.get(key)
        if found is None:
            source = {"h_group": lambda: hadamard_matrix(self.block // GROUP),
                      "h_block": lambda: hadamard_matrix(self.block), "mix": lambda: self._mix}[kind]()
            found = self._cache[key] = source.to(device=like.device, dtype=like.dtype)
        return found

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.n:
            raise ValueError(f"Expected {self.n} inputs, got {x.shape[-1]}")
        if self.identity:
            return x
        lead, groups = x.shape[:-1], self.block // GROUP
        y = x.reshape(*lead, self.n // self.block, groups, GROUP)
        if groups > 1:
            y = self._matrix("h_group", x) @ y
        if self._mix is not None:
            y = self._matrix("mix", x) @ y.reshape(*lead, self.n // self.block, self.block)
        return y.reshape(x.shape)

    inverse = forward


class WideHadamard(q.Rotation):
    """``R = A (x) H_block``: the cross-block mix followed by the block-16 Hadamard of the NVFP4 kernel."""

    def __init__(self, cross: CrossMix, name: str):
        self.cross, self.name, self.block = cross, name, cross.n if cross._mix is not None else cross.block

    def forward(self, x: Tensor) -> Tensor:
        return q.fwht(self.cross.forward(x), GROUP)

    def inverse(self, z: Tensor) -> Tensor:
        return self.cross.forward(q.fwht(z, GROUP))

    def dense(self, x: Tensor) -> Tensor:
        """The same map as two matrix products (differentiable and fast: the QAT path)."""
        c = self.cross
        y = x.reshape(*x.shape[:-1], c.n // c.block, c.block) @ c._matrix("h_block", x)
        if c._mix is not None:
            y = c._matrix("mix", x) @ y
        return y.reshape(x.shape)


class CrossButterfly(q.Rotation):
    """The cross-block mix followed by a fitted block-16 butterfly (angles per layer of a stack)."""

    def __init__(self, cross: CrossMix, butterfly: q.Butterfly, name: str):
        self.cross, self.butterfly, self.name, self.block = cross, butterfly, name, cross.block

    @property
    def angles(self) -> Tensor:
        return self.butterfly.angles

    def forward(self, x: Tensor) -> Tensor:
        return self.butterfly.forward(self.cross.forward(x))

    def inverse(self, z: Tensor) -> Tensor:
        return self.cross.forward(self.butterfly.inverse(z))


def make_cross(kind: str | None, n: int) -> CrossMix | None:
    if kind is None:
        return None
    if kind == "b64":
        return CrossMix(n, 64, None, "cross_b64")
    if kind == "full":
        block, pow2, odd = full_factors(n)
        m = odd * pow2
        mix = None if m == 1 else torch.kron(dct4_matrix(odd), hadamard_matrix(pow2))
        return CrossMix(n, block, mix, "cross_full")
    raise ValueError(f"Unknown cross-block mix {kind!r}")


def make_rotation(family: str, W: Tensor | None = None, n: int | None = None, *,
                  butterfly_steps: int = 150) -> tuple[q.Rotation, dict]:
    """Rotation of ``family`` for ``n`` inputs. Butterfly families fit angles to the weight stack ``W`` (L, N, n)."""
    cross_kind, inner = FAMILIES[family]
    n = W.shape[-1] if n is None else n
    cross = make_cross(cross_kind, n)
    if inner == "identity":
        return q.Rotation(), {}
    if inner == "hadamard_b16":
        return (q.BlockHadamard(GROUP) if cross is None else WideHadamard(cross, family)), {}
    if W is None:
        raise ValueError(f"{family} fits butterfly angles and needs the weight stack")
    mixed = W if cross is None else cross.forward(W.detach().float())
    butterfly, fit = q.fit_butterfly(mixed, FMT, block=GROUP, steps=butterfly_steps)
    return (butterfly if cross is None else CrossButterfly(cross, butterfly, family)), fit


def cross_of(rotation: q.Rotation) -> CrossMix | None:
    """The PyTorch (not in-kernel) part of a rotation, or None."""
    cross = getattr(rotation, "cross", None)
    return None if cross is None or cross.identity else cross


def layer_rotation(rotation: q.Rotation, i: int) -> q.Rotation:
    """The rotation of layer ``i`` of a stack (fitted angles are per layer)."""
    if isinstance(rotation, CrossButterfly):
        return CrossButterfly(rotation.cross, q.Butterfly(GROUP, rotation.angles[i]), rotation.name)
    return e2e._layer_rotation(rotation, i)


def rotation_to(rotation: q.Rotation, device) -> q.Rotation:
    """``rotation`` with its fitted angles on ``device`` (matrices of a ``CrossMix`` follow their input)."""
    if isinstance(rotation, CrossButterfly):
        return CrossButterfly(rotation.cross, q.Butterfly(GROUP, rotation.angles.to(device)), rotation.name)
    if isinstance(rotation, q.Butterfly):
        return q.Butterfly(GROUP, rotation.angles.to(device))
    return rotation


class CrossInput(nn.Module):
    """Applies the FP32 cross-block mix, then calls the hardware (or emulated) NVFP4 layer."""

    def __init__(self, layer: nn.Module, cross: CrossMix):
        super().__init__()
        self.layer, self.cross = layer, cross

    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):  # autocast would run the mix in BF16
            mixed = self.cross.forward(x.float())
        return self.layer(mixed)


# ======================================================================================
# Rotated-space STE fake quantization
# ======================================================================================
def qat_rotation(family: str, n: int) -> q.Rotation:
    """The rotation of a QAT family for ``n`` inputs, with a ``dense`` (matrix product) path when it rotates."""
    if family not in QAT_FAMILIES:
        raise ValueError(f"QAT supports the parameter-free families {QAT_FAMILIES}, not {family!r}")
    if family == "identity":
        return q.Rotation()
    return WideHadamard(make_cross(FAMILIES[family][0], n) or CrossMix(n, GROUP), family)


class RotatedFakeQuantLinear(nn.Module):
    """``Q(x R^T) Q(W R^T)^T`` with NVFP4 round-to-nearest and straight-through gradients.

    The weight uses groups of 16 along the rotated input dimension and one FP32 tensor scale; the
    input uses groups of 16 and the dynamic tensor scale of the forward batch, as the hardware
    dynamic mode does. The fake-quantized FP32 operands go to the BF16 autocast GEMM of the training
    step. ``active = False`` gives the plain BF16 linear (the BF16 pass of the dual loss).
    """

    def __init__(self, linear: nn.Linear, rotation: q.Rotation):
        super().__init__()
        self.linear, self.rotation, self.active = linear, rotation, True
        self._rotate = getattr(rotation, "dense", rotation.forward)

    def forward(self, x: Tensor) -> Tensor:
        w = self.linear.weight
        if not self.active:
            return F.linear(x, w)
        with torch.autocast(device_type=x.device.type, enabled=False):
            wr = self._rotate(w.float())
            wq = wr + (q.quantize_rtn(wr.detach(), FMT) - wr.detach())
            xr = self._rotate(x.reshape(-1, x.shape[-1]).float())
            xq = (xr + (q.quantize_rtn(xr.detach(), FMT) - xr.detach())).reshape(x.shape)
        return F.linear(xq, wq)


@contextmanager
def fake_quant(model, family: str):
    """Wrap every quantized linear of ``model`` in a ``RotatedFakeQuantLinear``; restore on exit."""
    wrappers = {name: RotatedFakeQuantLinear(linear, qat_rotation(family, linear.in_features))
                for name, linear in q.quantized_linears(model).items()}
    previous = e2e._swap(model, wrappers)
    try:
        yield wrappers
    finally:
        e2e._swap(model, previous)


def set_active(wrappers: dict, active: bool) -> None:
    for wrapper in wrappers.values():
        wrapper.active = active


def _cross_entropy(model, x: Tensor, y: Tensor) -> Tensor:
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        return F.cross_entropy(model(x).flatten(0, 1), y.flatten())  # FP32 under autocast, as ``h._loss``


def train_step(model, batch: tuple[Tensor, Tensor], optimizer, wrappers: dict | None = None,
               bf16_weight: float | None = None) -> Tensor:
    """One optimizer step over the ``grad_accum`` micro-batches of ``batch``. Returns the mean loss.

    ``bf16_weight`` None: the loss of the model as wrapped (plain or fake-quantized), as ``h.train_step``.
    ``bf16_weight = a`` (the dual loss): per micro-batch one BF16 pass and one W4A4 pass,
    ``a CE(bf16) + (1 - a) CE(w4a4)``. The two passes are differentiated one after the other, so one
    graph is alive at a time. The returned value is then the W4A4 cross entropy.
    """
    inputs, targets = batch
    total = torch.zeros((), device=inputs.device)
    for x, y in zip(inputs, targets):
        if bf16_weight is None:
            loss = _cross_entropy(model, x, y) / len(inputs)
            loss.backward()
            total += loss.detach()
            continue
        set_active(wrappers, False)
        try:
            (bf16_weight * _cross_entropy(model, x, y) / len(inputs)).backward()
        finally:
            set_active(wrappers, True)
        ce_q = _cross_entropy(model, x, y)
        ((1.0 - bf16_weight) * ce_q / len(inputs)).backward()
        total += ce_q.detach() / len(inputs)
    optimizer.step()
    return total


# ======================================================================================
# Activation statistics in rotated coordinates
# ======================================================================================
ACTIVATION_METRICS = ("excess_kurtosis", "max_over_rms", "group16_max_over_rms", "group16_outlier_share",
                      "nvfp4_relative_sq_error")
OUTLIER_CREST = 3.0  # a group of 16 whose max |z| exceeds 3 x its RMS (a 16-sample Gaussian group averages ~2.1)


@torch.no_grad()
def rotated_activation_statistics(model, maps: dict[str, dict[str, q.Rotation]], windows: np.ndarray, batch: int,
                                  device: torch.device) -> dict[str, dict[str, dict]]:
    """Statistics of ``z = map(x)`` for the input ``x`` of every quantized linear, per map family.

    ``maps[family][layer name]`` is the input map (rotation, with activation-aware scaling if any).
    Per layer: excess kurtosis and max / RMS of all elements (FP64 streaming moments), the mean
    max / RMS inside a group of 16 (the crest factor that the E4M3 block scale sees), the share of
    groups with a crest factor above ``OUTLIER_CREST``, and the relative squared error of dynamic
    NVFP4 round-to-nearest of ``z`` (one tensor scale per forward batch, as the hardware).
    """
    linears = q.quantized_linears(model)
    acc = {family: {name: {"moments": c4.Moments(device), "crest": torch.zeros((), dtype=torch.float64, device=device),
                           "outliers": torch.zeros((), dtype=torch.float64, device=device), "groups": 0,
                           "error": torch.zeros((), dtype=torch.float64, device=device),
                           "energy": torch.zeros((), dtype=torch.float64, device=device)}
                    for name in linears} for family in maps}

    def hook(name):
        def record(module, inputs):
            with torch.autocast(device_type=device.type, enabled=False):
                x = inputs[0].detach().reshape(-1, module.in_features).float()
                for family, per_layer in maps.items():
                    z = per_layer[name].forward(x)
                    a = acc[family][name]
                    a["moments"].update(z)
                    grouped = z.double().reshape(z.shape[0], -1, GROUP)
                    rms = grouped.square().mean(-1).sqrt()
                    crest = torch.where(rms > 0, grouped.abs().amax(-1) / rms.clamp_min(1e-300), torch.zeros_like(rms))
                    a["crest"] += crest.sum()
                    a["outliers"] += (crest > OUTLIER_CREST).sum()
                    a["groups"] += crest.numel()
                    a["error"] += (q.quantize_rtn(z, FMT) - z).double().square().sum()
                    a["energy"] += z.double().square().sum()
        return record

    handles = [m.register_forward_pre_hook(hook(name)) for name, m in linears.items()]
    try:
        for first in range(0, len(windows), batch):
            x = torch.from_numpy(np.asarray(windows[first:first + batch], dtype=np.int64)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model(x)
    finally:
        for handle in handles:
            handle.remove()
    out = {}
    for family, per_layer in acc.items():
        out[family] = {}
        for name, a in per_layer.items():
            moments = a["moments"].result()
            energy = a["energy"].item()
            out[family][name] = {
                "excess_kurtosis": moments["excess_kurtosis"], "max_over_rms": moments["max_over_rms"],
                "group16_max_over_rms": a["crest"].item() / a["groups"],
                "group16_outlier_share": a["outliers"].item() / a["groups"],
                "nvfp4_relative_sq_error": a["error"].item() / energy if energy > 0 else 0.0}
    return out


# ======================================================================================
# Hardware evidence: CUPTI kernel launches of one forward pass
# ======================================================================================
GEMM_KERNEL_MARKERS = ("gemm", "cutlass", "nvjet", "cublas")
FP4_KERNEL_MARKERS = ("e2m1", "nvfp4", "fp4")
CUPTI_KEYS = ("fp4_gemm_kernels", "non_fp4_gemm_kernels", "nvfp4_amax_kernels", "nvfp4_quant_kernels")


def classify_kernel(name: str) -> str | None:
    low = name.lower()
    if name == nb._amax_kernel.fn.__name__:
        return "nvfp4_amax_kernels"
    if name == nb._quant_kernel.fn.__name__:
        return "nvfp4_quant_kernels"
    if any(marker in low for marker in GEMM_KERNEL_MARKERS):
        return "fp4_gemm_kernels" if any(marker in low for marker in FP4_KERNEL_MARKERS) else "non_fp4_gemm_kernels"
    return None


def prime_cupti(device=None) -> None:
    """Start CUPTI for this process. Call before any CPU-only ``torch.profiler`` session.

    Kineto initializes once per process, with or without GPU support, from the activities of the
    first profiler session. After a CPU-only first session no later session records a CUDA kernel.
    """
    x = torch.zeros(8, device=device or "cuda")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]):
        x.add_(1.0)
        torch.cuda.synchronize()


@torch.no_grad()
def forward_profile(model, tokens: Tensor) -> dict:
    """One profiled BF16-autocast forward: ``aten::_scaled_mm_v2`` dispatches and CUPTI kernel launches.

    Returns ``{"scaled_mm_dispatches": n, "cupti": {FP4 GEMMs, other GEMMs, NVFP4 amax and quantizer
    kernels}}``. Raises when CUPTI recorded no kernel at all (see ``prime_cupti``): a count of zero must
    mean "no such kernel ran", never "the profiler was blind".
    """
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        with torch.autocast(device_type=tokens.device.type, dtype=torch.bfloat16):
            model(tokens)
        torch.cuda.synchronize()
    counts, kernels = dict.fromkeys(CUPTI_KEYS, 0), 0
    for event in profile.events():
        if event.device_type.name == "CUDA":
            kernels += 1
            kind = classify_kernel(event.name)
            if kind is not None:
                counts[kind] += 1
    if kernels == 0:
        raise RuntimeError("CUPTI recorded no CUDA kernel: a CPU-only profiler session ran first in this process "
                           "(call rotated_qat.prime_cupti() before it)")
    dispatches = sum(e.count for e in profile.key_averages() if e.key == "aten::_scaled_mm_v2")
    return {"scaled_mm_dispatches": dispatches, "cupti": counts}


# ======================================================================================
# Synthetic shards for the shakedown (never real text)
# ======================================================================================
SYNTHETIC_SOURCE = "synthetic-markov"
SYNTHETIC_ENCODING = "synthetic-uint16"


def write_synthetic_shards(directory: Path, *, seed: int = 0, vocab: int = 512, branching: int = 6,
                           document_tokens: int = 400, documents: dict[str, int] | None = None) -> dict:
    """Seeded first-order Markov shards (train / val / test from disjoint random streams) and their manifest.

    Every token has ``branching`` successors with probabilities proportional to 1 / rank, so a model
    can reach about 1.5 nats. Token 0 starts each document. The manifest has the layout that
    ``bench_c5_02_isoloss_ptq_mitigation.Shards(verify=False)`` reads.
    """
    documents = documents or {"train": 1200, "val": 120, "test": 120}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table_rng = np.random.default_rng([seed, 0])
    successors = np.stack([table_rng.choice(np.arange(1, vocab), size=branching, replace=False) for _ in range(vocab)])
    weights = 1.0 / np.arange(1, branching + 1)
    weights /= weights.sum()
    splits = {}
    for index, (name, count) in enumerate(documents.items()):
        rng = np.random.default_rng([seed, index + 1])
        tokens = np.zeros((count, document_tokens), dtype=np.uint16)
        state = rng.integers(1, vocab, size=count)
        for t in range(1, document_tokens):
            state = successors[state, rng.choice(branching, size=count, p=weights)]
            tokens[:, t] = state
        data = tokens.reshape(-1).astype("<u2").tobytes()
        (directory / f"{name}.bin").write_bytes(data)
        splits[name] = {"file": f"{name}.bin", "tokens": count * document_tokens, "documents": count,
                        "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {"source": {"name": SYNTHETIC_SOURCE, "seed": seed, "vocab": vocab, "branching": branching,
                           "document_tokens": document_tokens},
                "encoding": SYNTHETIC_ENCODING, "dtype": "uint16", "splits": splits}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


__all__ = ["FAMILIES", "WIDE_FAMILIES", "QAT_FAMILIES", "CrossMix", "WideHadamard", "CrossButterfly", "CrossInput",
           "RotatedFakeQuantLinear", "make_rotation", "make_cross", "cross_of", "layer_rotation", "rotation_to",
           "method_name", "parse_method", "is_wide", "kernel_rotation", "hadamard_matrix", "dct4_matrix",
           "qat_rotation",
           "full_factors", "fake_quant", "set_active", "train_step", "rotated_activation_statistics",
           "forward_profile", "prime_cupti", "classify_kernel", "write_synthetic_shards", "ACTIVATION_METRICS", "CUPTI_KEYS"]
