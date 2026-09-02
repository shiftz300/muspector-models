"""Function-preserving channel duplication for a finite-history fuzz core.

Unequal outgoing splits break the duplicate channels' gradient symmetry without
adding output noise or changing the initial mathematical function. No layer,
history length, sample clock, control semantics or audio level is changed.
"""
from __future__ import annotations

import torch

from .multirate_fuzz import MultirateFuzz


WIDE_GEOMETRY = {"audio_width": 32, "audio_blocks": 10, "controller_width": 16, "controller_blocks": 10}


def split_inputs(value):
    result = value.repeat_interleave(2, dim=1)
    shape = [1, result.shape[1], *[1 for _ in range(value.ndim - 2)]]
    # Exactly complementary binary fractions: unequal but sum exactly to one.
    splits = value.new_tensor([.375, .625]).repeat(value.shape[1]).reshape(shape)
    return result * splits


@torch.no_grad()
def widen(source):
    if (source.audio.width != 16 or source.controller.width != 8
            or source.fast_count != 10 or source.slow_count != 10):
        raise ValueError("only audited16/8-width ten-block source may be widened")
    target = MultirateFuzz(**WIDE_GEOMETRY).to(next(source.parameters()).device)
    transformed = {}
    for name, value in source.state_dict().items():
        if name in ("audio.input.weight", "controller.input.weight") or name.startswith("audio.current_controls."):
            transformed[name] = value.repeat_interleave(2, dim=0)
        elif name.startswith("controller.gate_bias."):
            transformed[name] = value.repeat_interleave(2, dim=0)
        elif name == "audio.output.weight":
            transformed[name] = split_inputs(value)
        elif name.endswith(".weight") and any(segment in name for segment in (".convolutions.", ".projections.", ".slow_controls.", "controller.output.")):
            transformed[name] = split_inputs(value).repeat_interleave(2, dim=0)
        else:
            raise ValueError(f"unsupported widening parameter: {name}")
    target.load_state_dict(transformed, strict=True)
    return target.eval()
