"""Stateful-safe and stateless candidate adapters for Tier-2 language-model pretraining."""

from __future__ import annotations

import torch
from torch import Tensor

from research.polar_opt.circuits import SpectralPrescaledPolar
from research.polar_opt.kernels_blackwell import polar_express_triton
from research.polar_opt.polar_switch_blackwell import polar_switch_blackwell
from research.polar_opt.schatten_switch_blackwell import schatten_switch_blackwell


def pe_triton_bf16(G: Tensor) -> Tensor:
    return polar_express_triton(G, steps=5, dtype=torch.bfloat16)


def ps2_blackwell_fp16(G: Tensor) -> Tensor:
    return polar_switch_blackwell(G, "PS2_55", dtype=torch.float16)


def ps3_blackwell_fp16(G: Tensor) -> Tensor:
    return polar_switch_blackwell(G, "PS3_55", dtype=torch.float16)


def ss3_54_blackwell_fp16(G: Tensor) -> Tensor:
    return schatten_switch_blackwell(G, "SS3_54", dtype=torch.float16)


def ss3_53_blackwell_fp16(G: Tensor) -> Tensor:
    return schatten_switch_blackwell(G, "SS3_53", dtype=torch.float16)


def ss3_35_blackwell_fp16(G: Tensor) -> Tensor:
    return schatten_switch_blackwell(G, "SS3_35", dtype=torch.float16)


def ss3_34_blackwell_fp16(G: Tensor) -> Tensor:
    return schatten_switch_blackwell(G, "SS3_34", dtype=torch.float16)


class ShapeKeyedSpectralPolar:
    """Adapter for ``harness_tier2_lm.Muon`` supporting stateless Schatten-8 or warm spectral scaling."""

    def __init__(
        self,
        circuit: str,
        *,
        dtype: torch.dtype = torch.float16,
        use_triton_syrk: bool = True,
        scaling: str = "schatten8",
    ):
        self.polar = SpectralPrescaledPolar(
            circuit, dtype=dtype, use_triton_syrk=use_triton_syrk, scaling=scaling
        )

    def __call__(self, G: Tensor) -> Tensor:
        return self.polar(G, key=tuple(G.shape))
