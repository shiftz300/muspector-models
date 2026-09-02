"""Training-only temporal tap preconditioning; ordinary Conv1d at inference.

An exact zero delta starts from existing float32 weights. DC, first difference
and second difference span every three-tap kernel. Larger derivative-coordinate
steps can learn sharp fuzz transitions without filtering or scaling input/output
audio. Materialization removes all training-only buffers and parameters.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalBasisConv(nn.Module):
    def __init__(self, source, derivative_scale=8.):
        super().__init__()
        if (not isinstance(source, nn.Conv1d) or source.kernel_size != (3,) or source.bias is not None
                or source.groups != 1 or source.stride != (1,) or source.padding != (0,)):
            raise ValueError("bias-free unpadded unit-stride three-tap convolution required")
        if derivative_scale != 8.:
            raise ValueError("fixed derivative-coordinate scale8 required")
        self.dilation = source.dilation
        self.register_buffer("base_weight", source.weight.detach().clone())
        self.register_buffer("basis", source.weight.new_tensor([[1., 1., 1.], [-8., 0., 8.], [8., -16., 8.]]))
        self.coefficients = nn.Parameter(torch.zeros_like(source.weight))

    def effective_weight(self):
        return self.base_weight + self.coefficients @ self.basis

    def forward(self, value):
        return F.conv1d(value, self.effective_weight(), dilation=self.dilation)


def attach_temporal_basis(model):
    """Change optimization coordinates for only the first4 fast convolutions."""
    if len(model.audio.convolutions) != 10:
        raise ValueError("audited ten-block fast network required")
    for index in range(4):
        if isinstance(model.audio.convolutions[index], TemporalBasisConv):
            raise ValueError("basis already attached")
    for index in range(4):
        model.audio.convolutions[index] = TemporalBasisConv(model.audio.convolutions[index])
    return model


def materialized_state_dict(model):
    names = [f"audio.convolutions.{index}." for index in range(4)]
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
             if not any(name.startswith(prefix) for prefix in names)}
    for index, prefix in enumerate(names):
        layer = model.audio.convolutions[index]
        if not isinstance(layer, TemporalBasisConv):
            raise ValueError("four expected training-only coordinate layers required")
        state[prefix + "weight"] = layer.effective_weight().detach().cpu().clone()
    return state
