"""Fixed convex composition of zero-preserving DFZ recurrent predictors."""
from __future__ import annotations

import math
import torch


class StableCompositeEnsemble(torch.nn.Module):
    def __init__(self, members, weights):
        super().__init__()
        if not members or len(members) != len(weights):
            raise ValueError('one positive weight is required per member')
        if any(not math.isfinite(w) or w <= 0 for w in weights) or not math.isclose(sum(weights), 1., abs_tol=1e-7, rel_tol=0.):
            raise ValueError('fixed convex weights must sum to one')
        if any(member.control_count != 2 for member in members):
            raise ValueError('DFZ ensemble requires two normalized controls')
        semantics = [getattr(member, 'base', member).inverted_controls for member in members]
        if any(value != semantics[0] for value in semantics):
            raise ValueError('ensemble members disagree on control semantics')
        self.members = torch.nn.ModuleList(members)
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.control_count = 2
        self.export_state_widths = []
        self.member_state_counts = []
        for member in members:
            widths = getattr(member, 'export_state_widths', [layer.hidden_size for layer in member.modules() if isinstance(layer, torch.nn.LSTM)])
            self.export_state_widths.extend(widths)
            self.member_state_counts.append(len(widths))
        self.layers = len(self.export_state_widths)

    @property
    def state_floats_per_mono_stream(self):
        return sum(self.export_state_widths) * 2

    def forward(self, dry, controls, state=None):
        if state is not None and len(state) != self.layers:
            raise ValueError('wrong composite ensemble state count')
        values, states, offset = [], [], 0
        for index, (member, count) in enumerate(zip(self.members, self.member_state_counts)):
            previous = None if state is None else state[offset:offset + count]
            value, current = member(dry, controls, previous)
            values.append(value * self.weights[index])
            states.extend(current)
            offset += count
        return torch.stack(values).sum(0), tuple(states)
