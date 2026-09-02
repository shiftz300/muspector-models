"""Independent two-array causal audio core with raw-input mixin and skip heads.

NAM's documented residual/skip separation motivates the topology; this is our
own conditional zero-origin model, not an import of NAM weights or its runtime.
Both sample-clock and finite low-rate controller contracts are inherited.
"""
from __future__ import annotations

import math

import torch

from .multirate_fuzz import MultirateFuzz, _history, _state_pair


ARCHITECTURE = "cascaded-zero-origin-causal-multirate"
GEOMETRY = {"stage_widths": [16, 8], "blocks_per_stage": 10, "skip_width": 8,
            "controller_width": 8, "controller_blocks": 10}


class CascadedAudio(torch.nn.Module):
    def __init__(self, stage_widths=(16, 8), blocks_per_stage=10, skip_width=8, controller_width=8):
        super().__init__()
        self.stage_widths, self.blocks_per_stage = tuple(stage_widths), blocks_per_stage
        self.width, self.skip_width = stage_widths[0], skip_width
        self.dilations = [2**index for _ in stage_widths for index in range(blocks_per_stage)]
        self.widths = [width for width in stage_widths for _ in range(blocks_per_stage)]
        self.rechannels = torch.nn.ModuleList(torch.nn.Conv1d(a, b, 1, bias=False)
                                              for a, b in zip((1, *stage_widths[:-1]), stage_widths))
        self.convolutions = torch.nn.ModuleList(torch.nn.Conv1d(width, width, 3, dilation=dilation, bias=False)
                                               for width, dilation in zip(self.widths, self.dilations))
        self.raw_mixins = torch.nn.ModuleList(torch.nn.Conv1d(1, width, 1, bias=False) for width in self.widths)
        self.projections = torch.nn.ModuleList(torch.nn.Conv1d(width, width, 1, bias=False) for width in self.widths)
        self.skip_heads = torch.nn.ModuleList(torch.nn.Conv1d(width, skip_width, 1, bias=False) for width in self.widths)
        self.current_controls = torch.nn.ModuleList(torch.nn.Linear(2, 2*width) for width in self.widths)
        self.slow_controls = torch.nn.ModuleList(torch.nn.Linear(controller_width, 2*width, bias=False) for width in self.widths)
        self.output = torch.nn.Conv1d(skip_width, 1, 1, bias=False)
        torch.nn.init.zeros_(self.output.weight)

    @property
    def receptive_field(self):
        return 1+2*sum(self.dilations)

    def forward(self, dry, controls, held_sequence, block_indices, state=None):
        raw = dry[:, None]*21.4
        value, states, skips = raw, [], []
        for index, (width, dilation, convolution, mixin, projection, head, current, slow) in enumerate(zip(
                self.widths, self.dilations, self.convolutions, self.raw_mixins, self.projections,
                self.skip_heads, self.current_controls, self.slow_controls)):
            if index % self.blocks_per_stage == 0:
                value = self.rechannels[index//self.blocks_per_stage](value)
            old = _history(value, None if state is None else state[index], width, dilation)
            extended = torch.cat((old, value), 2)
            activation = convolution(extended)+mixin(raw)
            modulation = slow(held_sequence)
            if block_indices.ndim == 1:
                modulation = modulation[:, block_indices]
            else:
                modulation = torch.gather(modulation, 1, block_indices[..., None].expand(-1, -1, 2*width))
            scale, threshold = (modulation+current(controls)).transpose(1, 2).split(width, 1)
            update = (1+.5*torch.tanh(scale))*(torch.tanh(activation+threshold)-torch.tanh(threshold))
            value = value+projection(update)/math.sqrt(self.blocks_per_stage)
            skips.append(head(update)/math.sqrt(len(self.dilations)))
            states.append(_state_pair(extended[:, :, -2*dilation:], width, dilation))
        accumulated = skips[0]
        for skip in skips[1:]:
            accumulated = accumulated+skip
        return self.output(accumulated).squeeze(1), tuple(states)


class CascadedMultirateFuzz(MultirateFuzz):
    def __init__(self, stage_widths=(16, 8), blocks_per_stage=10, skip_width=8, controller_width=8, controller_blocks=10):
        if not stage_widths or min(stage_widths) < 1 or min(blocks_per_stage, skip_width, controller_width, controller_blocks) < 1:
            raise ValueError("positive cascaded geometry required")
        super().__init__(stage_widths[0], blocks_per_stage, controller_width, controller_blocks)
        self.audio = CascadedAudio(stage_widths, blocks_per_stage, skip_width, controller_width)
        self.fast_count = len(self.audio.dilations)
        self.state_widths = [*[width*dilation for width, dilation in zip(self.audio.widths, self.audio.dilations)],
                             *[controller_width*dilation for dilation in self.controller.dilations]]
