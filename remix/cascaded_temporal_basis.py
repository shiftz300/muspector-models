"""Training-only derivative coordinates for the first4 taps in each array."""
import torch

from .cascaded_multirate_fuzz import CascadedMultirateFuzz
from .dfz_temporal_basis import TemporalBasisConv


INDICES = (0, 1, 2, 3, 10, 11, 12, 13)


def attach(model):
    if type(model) is not CascadedMultirateFuzz or model.audio.blocks_per_stage != 10 or len(model.audio.convolutions) != 20:
        raise ValueError("reviewed two-array model required")
    if any(type(model.audio.convolutions[index]) is not torch.nn.Conv1d for index in INDICES):
        raise ValueError("native convolutions required; do not attach twice")
    for index in INDICES:
        model.audio.convolutions[index] = TemporalBasisConv(model.audio.convolutions[index])
    return model


def materialize(model):
    prefixes = [f"audio.convolutions.{index}." for index in INDICES]
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
             if not any(name.startswith(prefix) for prefix in prefixes)}
    for index, prefix in zip(INDICES, prefixes):
        layer = model.audio.convolutions[index]
        if type(layer) is not TemporalBasisConv:
            raise ValueError("all reviewed coordinate layers required")
        state[prefix+"weight"] = layer.effective_weight().detach().cpu().clone()
    return state
