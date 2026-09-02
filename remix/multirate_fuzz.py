"""Standalone causal two-rate fuzz model; experimental, not an admitted schema.

The audio path never consumes the current block's future. Completed 64-frame
blocks drive a low-rate finite-history controller; an explicit integer clock
and real pending samples survive arbitrary callback boundaries. Both paths
are finite convolutional networks, with no original LSTM/core prediction.
"""
from __future__ import annotations

import math
import torch


BLOCK = 64


def _history(value, state, width, dilation):
    if state is None:
        return value.new_zeros(value.shape[0], width, 2 * dilation)
    return torch.cat(state, 2).squeeze(0).reshape(value.shape[0], width, 2 * dilation)


def _state_pair(history, width, dilation):
    flat = history.reshape(history.shape[0], 2 * width * dilation)
    return flat[:, :width * dilation].unsqueeze(0), flat[:, width * dilation:].unsqueeze(0)


class FiniteController(torch.nn.Module):
    def __init__(self, width=8, blocks=10):
        super().__init__()
        self.width = width
        self.dilations = [2 ** index for index in range(blocks)]
        self.input = torch.nn.Conv1d(8, width, 1, bias=False)
        self.convolutions = torch.nn.ModuleList(torch.nn.Conv1d(width, 2 * width, 3, dilation=d, bias=False)
                                                for d in self.dilations)
        self.gate_bias = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros(width)) for _ in self.dilations)
        self.projections = torch.nn.ModuleList(torch.nn.Conv1d(width, width, 1, bias=False) for _ in self.dilations)
        self.output = torch.nn.Conv1d(width, width, 1, bias=False)

    @property
    def receptive_field(self):
        return 1 + 2 * sum(self.dilations)

    def forward(self, descriptors, state=None, advance=None):
        value, states = self.input(descriptors.transpose(1, 2)), []
        if advance is None:
            advance = descriptors.shape[1]
        for index, (dilation, convolution, bias, projection) in enumerate(zip(
                self.dilations, self.convolutions, self.gate_bias, self.projections)):
            old = _history(value, None if state is None else state[index], self.width, dilation)
            extended = torch.cat((old, value), 2)
            activation, gate = convolution(extended).split(self.width, 1)
            value = value + projection(torch.tanh(activation) * torch.sigmoid(gate + bias[None, :, None])) / math.sqrt(len(self.dilations))
            # Only real completed descriptors advance state. The extra shadow
            # descriptor keeps Conv valid when a callback completes zero blocks.
            history = extended[:, :, advance:advance + 2 * dilation]
            states.append(_state_pair(history, self.width, dilation))
        return self.output(value).transpose(1, 2), tuple(states)


class ModulatedAudioGCN(torch.nn.Module):
    def __init__(self, width=16, blocks=10, controller_width=8):
        super().__init__()
        self.width = width
        self.dilations = [2 ** index for index in range(blocks)]
        self.input = torch.nn.Conv1d(1, width, 1, bias=False)
        self.convolutions = torch.nn.ModuleList(torch.nn.Conv1d(width, 2 * width, 3, dilation=d, bias=False)
                                                for d in self.dilations)
        self.projections = torch.nn.ModuleList(torch.nn.Conv1d(width, width, 1, bias=False) for _ in self.dilations)
        self.current_controls = torch.nn.ModuleList(torch.nn.Linear(2, 2 * width) for _ in self.dilations)
        self.slow_controls = torch.nn.ModuleList(torch.nn.Linear(controller_width, 2 * width, bias=False)
                                                for _ in self.dilations)
        self.output = torch.nn.Conv1d(width, 1, 1, bias=False)
        torch.nn.init.zeros_(self.output.weight)

    @property
    def receptive_field(self):
        return 1 + 2 * sum(self.dilations)

    def forward(self, dry, controls, held_sequence, block_indices, state=None):
        value, states = self.input(dry[:, None] * 21.4), []
        for index, (dilation, convolution, projection, current, slow) in enumerate(zip(
                self.dilations, self.convolutions, self.projections, self.current_controls, self.slow_controls)):
            old = _history(value, None if state is None else state[index], self.width, dilation)
            extended = torch.cat((old, value), 2)
            activation, gate = convolution(extended).split(self.width, 1)
            # Project once per low-rate token before gathering sample controls.
            slow_values = slow(held_sequence)
            if block_indices.ndim == 1:
                modulation = slow_values[:, block_indices] + current(controls)
            else:
                modulation = torch.gather(slow_values, 1, block_indices[..., None].expand(-1, -1, 2 * self.width)) + current(controls)
            scale, bias = modulation.transpose(1, 2).split(self.width, 1)
            update = (1 + .5 * torch.tanh(scale)) * torch.tanh(activation) * torch.sigmoid(gate + bias)
            value = value + projection(update) / math.sqrt(len(self.dilations))
            states.append(_state_pair(extended[:, :, -2 * dilation:], self.width, dilation))
        return self.output(value).squeeze(1), tuple(states)


class MultirateFuzz(torch.nn.Module):
    """Flat tensor state: fast pairs, slow pairs, pending Dry, held, int phase."""

    control_count = 2
    sample_rate = 48_000

    def __init__(self, audio_width=16, audio_blocks=10, controller_width=8, controller_blocks=10):
        super().__init__()
        self.audio = ModulatedAudioGCN(audio_width, audio_blocks, controller_width)
        self.controller = FiniteController(controller_width, controller_blocks)
        self.fast_count, self.slow_count = audio_blocks, controller_blocks
        self.state_widths = [*[audio_width * d for d in self.audio.dilations],
                             *[controller_width * d for d in self.controller.dilations]]

    def initial_state(self, dry):
        return (*[dry.new_zeros(1, dry.shape[0], width) for width in self.state_widths for _ in range(2)],
                dry.new_zeros(dry.shape[0], BLOCK), dry.new_zeros(dry.shape[0], self.controller.width),
                torch.zeros((), dtype=torch.int64, device=dry.device))

    @staticmethod
    def descriptors(block_audio, block_controls):
        value = block_audio * 21.4
        return torch.stack((value.mean(-1), value.abs().mean(-1), value.square().mean(-1),
                            value.abs().amax(-1), value.amax(-1), value.amin(-1),
                            block_controls[..., -1, 0], block_controls[..., -1, 1]), -1)

    def training_held(self, complete_dry, static_controls):
        """All real descriptors from file0; each audio block uses its predecessor.

        Computing later descriptors in parallel does not expose their values:
        the finite controller is causal and the fast window gathers only held
        indices at or before that sample's block. No learned state is cached.
        """
        if complete_dry.shape[1] % BLOCK:
            raise ValueError("training held sequence requires complete audited64-frame blocks")
        blocks = complete_dry.reshape(complete_dry.shape[0], -1, BLOCK)
        controls = static_controls[:, None, None].expand(-1, blocks.shape[1], BLOCK, -1)
        features = self.descriptors(blocks, controls)
        slow, _ = self.controller(features)
        return torch.cat((slow.new_zeros(slow.shape[0], 1, slow.shape[2]), slow), 1)

    def forward(self, dry, controls, state=None):
        if not torch.jit.is_tracing():
            if dry.ndim != 2 or dry.shape[1] < 1 or dry.dtype != torch.float32:
                raise ValueError("nonempty batch/time float32 Dry required")
            if controls.ndim not in (2, 3) or controls.shape[0] != dry.shape[0] or controls.shape[-1] != 2:
                raise ValueError("two normalized controls required")
            if controls.ndim == 3 and controls.shape[1] != dry.shape[1]:
                raise ValueError("dynamic control clock differs from Dry")
            if not torch.isfinite(dry).all() or not torch.isfinite(controls).all() or (controls < 0).any() or (controls > 1).any():
                raise ValueError("finite audio and normalized controls required")
        if controls.ndim == 2:
            controls = controls[:, None].expand(-1, dry.shape[1], -1)
        if state is None:
            state = self.initial_state(dry)
        if not torch.jit.is_tracing():
            if len(state) != 2 * len(self.state_widths) + 3:
                raise ValueError("wrong multirate state count")
            if state[-1].ndim != 0 or state[-1].dtype != torch.int64 or not 0 <= int(state[-1]) < BLOCK:
                raise ValueError("one shared integer sample clock in0..63 required")
        pairs = tuple((state[2 * index], state[2 * index + 1]) for index in range(len(self.state_widths)))
        pending, held, phase = state[-3:]
        extended = torch.cat((pending[:, :phase], dry), 1)
        extended_controls = torch.cat((controls.new_zeros(controls.shape[0], phase, 2), controls), 1)
        completed = (phase + dry.shape[1]) // BLOCK
        boundary = completed * BLOCK
        # One shadow zero block is always ignored by the clock and state. It
        # avoids a traced Python/If branch and supports callbacks of one sample.
        block_audio = torch.cat((extended[:, :boundary], dry.new_zeros(dry.shape[0], BLOCK)), 1).reshape(dry.shape[0], -1, BLOCK)
        block_controls = torch.cat((extended_controls[:, :boundary], controls.new_zeros(controls.shape[0], BLOCK, 2)), 1).reshape(dry.shape[0], -1, BLOCK, 2)
        descriptors = self.descriptors(block_audio, block_controls)
        slow, slow_state = self.controller(descriptors, pairs[self.fast_count:], advance=completed)
        held_sequence = torch.cat((held[:, None], slow), 1)
        indices = (torch.arange(dry.shape[1], device=dry.device, dtype=torch.int64) + phase) // BLOCK
        result, fast_state = self.audio(dry, controls, held_sequence, indices, pairs[:self.fast_count])
        next_pending = torch.cat((extended[:, boundary:], dry.new_zeros(dry.shape[0], BLOCK)), 1)[:, :BLOCK]
        next_held = held_sequence[:, completed]
        next_phase = (phase + dry.shape[1]) % BLOCK
        return result, (*[value for pair in (*fast_state, *slow_state) for value in pair], next_pending, next_held, next_phase)
