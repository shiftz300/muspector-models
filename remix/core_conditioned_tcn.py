"""Independent finite-history residual conditioned on a frozen core's output.

This experimental class is not registered in the accepted runtime loader.
The second audio input is the core prediction, never the measured Wet target.
"""
from __future__ import annotations

import torch


class CoreConditionedTCN(torch.nn.Module):
    def __init__(self, width=16, blocks=12):
        super().__init__()
        self.width = width
        self.dilations = [2 ** index for index in range(blocks)]
        self.input = torch.nn.Conv1d(2, width, 1, bias=False)
        self.convolutions = torch.nn.ModuleList(
            torch.nn.Conv1d(width, 2 * width, 3, dilation=value, bias=False)
            for value in self.dilations)
        self.conditions = torch.nn.ModuleList(torch.nn.Linear(2, width) for _ in self.dilations)
        self.projections = torch.nn.ModuleList(
            torch.nn.Conv1d(width, width, 1, bias=False) for _ in self.dilations)
        self.output = torch.nn.Conv1d(width, 1, 1, bias=False)
        torch.nn.init.zeros_(self.output.weight)

    @property
    def receptive_field(self):
        return 1 + 2 * sum(self.dilations)

    def forward(self, dry, original, controls, state=None):
        if not torch.jit.is_tracing():
            if dry.ndim != 2 or original.shape != dry.shape:
                raise ValueError("core-conditioned residual needs matching Dry/core arrays")
            if state is not None and len(state) != len(self.dilations):
                raise ValueError("wrong core-conditioned TCN state count")
        condition = controls[:, None, :].expand(-1, dry.shape[1], -1) if controls.ndim == 2 else controls
        value = self.input(torch.stack((dry * 21.4, original), 1))
        states = []
        for index, (dilation, convolution, condition_layer, projection) in enumerate(zip(
                self.dilations, self.convolutions, self.conditions, self.projections)):
            old = (value.new_zeros(value.shape[0], self.width, 2 * dilation) if state is None else
                   torch.cat(state[index], 2).squeeze(0).reshape(value.shape[0], self.width, 2 * dilation))
            extended = torch.cat((old, value), 2)
            activation, gate = convolution(extended).split(self.width, 1)
            update = torch.tanh(activation) * torch.sigmoid(gate + condition_layer(condition).transpose(1, 2))
            value = value + projection(update) / len(self.dilations) ** .5
            history = extended[:, :, -2 * dilation:].reshape(value.shape[0], -1)
            states.append((history[:, :self.width * dilation].unsqueeze(0),
                           history[:, self.width * dilation:].unsqueeze(0)))
        return self.output(value).squeeze(1), tuple(states)


class CoreConditionedResidual(torch.nn.Module):
    """Compute causal core output first, then its output-conditioned residual."""
    def __init__(self, base, tcn):
        super().__init__()
        self.base, self.tcn = base.eval().requires_grad_(False), tcn
        self.control_count = base.control_count
        base_widths = getattr(base, "export_state_widths", [layer.hidden_size for layer in base.modules()
                                                           if isinstance(layer, torch.nn.LSTM)])
        self.base_states = len(base_widths)
        self.export_state_widths = [*base_widths, *[tcn.width * value for value in tcn.dilations]]

    def forward(self, dry, controls, state=None):
        if state is not None and len(state) != len(self.export_state_widths):
            raise ValueError("wrong core-conditioned model state count")
        original, base_state = self.base(dry, controls, None if state is None else state[:self.base_states])
        residual, tcn_state = self.tcn(dry, original, controls, None if state is None else state[self.base_states:])
        return original + residual, (*base_state, *tcn_state)
