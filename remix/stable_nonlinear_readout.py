"""Bias-free nonlinear readout of a frozen stable recurrent state."""
from __future__ import annotations
import torch


class NonlinearReadout(torch.nn.Module):
    def __init__(self, hidden_size=64, width=16):
        super().__init__()
        self.input = torch.nn.Parameter(torch.randn(9, hidden_size, width) / hidden_size**.5)
        self.output = torch.nn.Parameter(torch.zeros(9, width))
        self.linear = torch.nn.Parameter(torch.zeros(9, hidden_size))

    def at_training_knots(self, hidden, controls):
        """Exact fast path for the audited 3x3 training-control grid only."""
        ij=(controls*2).round().long();index=ij[:,0]*3+ij[:,1]
        nonlinear=torch.tanh(torch.bmm(hidden,self.input[index])*8)
        return (nonlinear*self.output[index][:,None]).sum(-1)+(hidden*self.linear[index][:,None]).sum(-1)

    def forward(self, hidden, controls):
        condition = controls[:, None, :] if controls.ndim == 2 else controls
        knots = hidden.new_tensor((0., .5, 1.))
        basis = (1 - (condition[..., None] - knots).abs() * 2).clamp_min(0)
        coefficient = (basis[..., 0, :, None] * basis[..., 1, None, :]).flatten(-2)
        # All operations on audio state are bias-free: arbitrary control changes
        # cannot produce audio from an exact zero recurrent state.
        branches = torch.stack([
            torch.tanh(hidden @ self.input[i] * 8) @ self.output[i]
            + hidden @ self.linear[i] for i in range(9)
        ], -1)
        return (branches * coefficient).sum(-1)


class StableNonlinearReadout(torch.nn.Module):
    def __init__(self, base, readout):
        super().__init__()
        self.base, self.readout = base, readout
        self.control_count, self.layers = base.control_count, base.layers
        self.export_state_widths = [layer.hidden_size for layer in base.rnn_layers]

    @property
    def state_floats_per_mono_stream(self):
        return self.base.state_floats_per_mono_stream

    def forward(self, dry, controls, state=None):
        hidden, state = self.base.encode(dry, controls, state)
        return self.base.output_layer(hidden).squeeze(-1) + self.readout(hidden, controls), state
