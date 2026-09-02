"""Experimental zero-origin conditional activation for the existing two rates.

This adds a learned conditional threshold inside each audio nonlinearity:
tanh(activation+bias)-tanh(bias). Unlike an uncentered feature offset it cannot
emit audio from zero fast history, even with nonzero control/slow history.
All new heads start at zero; an existing core's initial function is unchanged.
No input/output preprocessing, extra state, gain stage or limiter is added.
"""
from __future__ import annotations

import math

import torch

from .multirate_fuzz import ModulatedAudioGCN, MultirateFuzz, _history, _state_pair


class CenteredModulatedAudioGCN(ModulatedAudioGCN):
    def __init__(self, width=32, blocks=10, controller_width=16):
        super().__init__(width, blocks, controller_width)
        self.activation_current = torch.nn.ModuleList(torch.nn.Linear(2, width) for _ in self.dilations)
        self.activation_slow = torch.nn.ModuleList(torch.nn.Linear(controller_width, width, bias=False) for _ in self.dilations)
        for module in (*self.activation_current, *self.activation_slow):
            torch.nn.init.zeros_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, dry, controls, held_sequence, block_indices, state=None):
        value, states = self.input(dry[:, None] * 21.4), []
        for index, (dilation, convolution, projection, current, slow, threshold_current, threshold_slow) in enumerate(zip(
                self.dilations, self.convolutions, self.projections, self.current_controls, self.slow_controls,
                self.activation_current, self.activation_slow)):
            old = _history(value, None if state is None else state[index], self.width, dilation)
            extended = torch.cat((old, value), 2)
            activation, gate = convolution(extended).split(self.width, 1)
            modulation = slow(held_sequence)
            threshold = threshold_slow(held_sequence)
            if block_indices.ndim == 1:
                modulation = modulation[:, block_indices]
                threshold = threshold[:, block_indices]
            else:
                modulation = torch.gather(modulation, 1, block_indices[..., None].expand(-1, -1, 2 * self.width))
                threshold = torch.gather(threshold, 1, block_indices[..., None].expand(-1, -1, self.width))
            modulation = modulation + current(controls)
            threshold = (threshold + threshold_current(controls)).transpose(1, 2)
            scale, bias = modulation.transpose(1, 2).split(self.width, 1)
            centered = torch.tanh(activation + threshold) - torch.tanh(threshold)
            update = (1 + .5 * torch.tanh(scale)) * centered * torch.sigmoid(gate + bias)
            value = value + projection(update) / math.sqrt(len(self.dilations))
            states.append(_state_pair(extended[:, :, -2 * dilation:], self.width, dilation))
        return self.output(value).squeeze(1), tuple(states)


class CenteredMultirateFuzz(MultirateFuzz):
    def __init__(self, audio_width=32, audio_blocks=10, controller_width=16, controller_blocks=10):
        super().__init__(audio_width, audio_blocks, controller_width, controller_blocks)
        self.audio = CenteredModulatedAudioGCN(audio_width, audio_blocks, controller_width)

    @classmethod
    def from_base(cls, source):
        if type(source) is not MultirateFuzz:
            raise ValueError("plain independent core required for zero-head initialization")
        result = cls(source.audio.width, source.fast_count, source.controller.width, source.slow_count)
        result.to(next(source.parameters()).device)
        state = result.state_dict()
        source_state = source.state_dict()
        if not set(source_state).issubset(state):
            raise ValueError("source includes nonmaterialized or unsupported weights")
        state.update({name: value.detach().clone() for name, value in source_state.items()})
        result.load_state_dict(state, strict=True)
        return result.eval()
