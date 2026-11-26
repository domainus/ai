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


def ss3_35_blackwell_bf16(G: Tensor) -> Tensor:
    return schatten_switch_blackwell(G, "SS3_35", dtype=torch.bfloat16)


class HybridPhasePolar:
    """Use ``warmup_fn`` for the first ``warmup_steps`` optimizer steps, then ``fast_fn``.

    The phase advances on ``on_step_end()``, which ``Muon.step`` calls once per optimizer step
    (not once per same-shape group). ``set_warmup(True)`` is for the untimed compile warm-up of
    ``run_trial``: each call then also runs ``fast_fn`` (so its autotune and plan caches are warm
    before timed step 0) and returns the ``warmup_fn`` result, and ``on_step_end`` does not
    advance the phase. The harness builds a fresh instance for the timed run.
    """

    def __init__(self, warmup_fn, fast_fn, warmup_steps: int = 40):
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.warmup_fn, self.fast_fn, self.warmup_steps = warmup_fn, fast_fn, warmup_steps
        self.step = 0
        self._warming = False

    def set_warmup(self, enabled: bool) -> None:
        self._warming = bool(enabled)

    def on_step_end(self) -> None:
        if not self._warming:
            self.step += 1

    def __call__(self, G: Tensor) -> Tensor:
        if self._warming:
            self.fast_fn(G)
            return self.warmup_fn(G)
        return self.warmup_fn(G) if self.step < self.warmup_steps else self.fast_fn(G)


def hybrid_ps3_to_ss3_34_w40() -> HybridPhasePolar:
    return HybridPhasePolar(ps3_blackwell_fp16, ss3_34_blackwell_fp16, warmup_steps=40)


def hybrid_ss3_54_to_ss3_34_w40() -> HybridPhasePolar:
    return HybridPhasePolar(ss3_54_blackwell_fp16, ss3_34_blackwell_fp16, warmup_steps=40)


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
