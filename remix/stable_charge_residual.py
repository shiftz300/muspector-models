"""Input-driven, zero-origin charge states and an additive DFZ residual readout."""
from __future__ import annotations

import math
import torch


class ChargeBank(torch.nn.Module):
    """Fixed leaky states, expressed with a standard exportable ReLU RNN.

    q[t] = a*q[t-1] + (1-a)*p[t], where p=tanh(100*Dry)^2.
    All coefficients, inputs and reachable states are nonnegative, so ReLU
    is exactly identity. No target signal, input-clip gain or future is used.
    Direct constant coefficients avoid sigmoid drift in very slow LSTM gates.
    """
    def __init__(self, milliseconds=(1., 5., 20., 100., 300.)):
        super().__init__()
        if not milliseconds or any(not math.isfinite(t) or t <= 0 for t in milliseconds):
            raise ValueError('positive finite charge time constants required')
        self.milliseconds = tuple(milliseconds)
        self.width = len(milliseconds)
        self.cell = torch.nn.RNN(1, self.width, nonlinearity='relu', batch_first=True)
        with torch.no_grad():
            for parameter in self.cell.parameters():
                parameter.zero_()
            a = torch.exp(-1 / (torch.tensor(milliseconds, dtype=torch.float32) * 48))
            self.cell.weight_hh_l0.copy_(torch.diag(a))
            self.cell.weight_ih_l0[:, 0].copy_(1-a)
        self.requires_grad_(False)

    def forward(self, dry, state=None):
        power = torch.tanh(dry * 100).square()
        charge, hidden = self.cell(power[..., None], None if state is None else state[0])
        # Keep the public (h,c) pair without interpreting c as a gate. The zero
        # multiplication also preserves this otherwise-unused graph input.
        dummy = torch.zeros_like(hidden) if state is None else state[1] * 0.
        # Adjacent charge differences encode attack/recovery at distinct scales.
        features = torch.cat((power[..., None], charge[..., :-1] - charge[..., 1:], charge[..., -1:]), -1)
        return features, (hidden, dummy)


class ChargeReadout(torch.nn.Module):
    def __init__(self, width=6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(9, width))

    def at_training_knots(self, features, controls):
        indices = (controls * 2).round().long()
        weights = self.weight[indices[:, 0] * 3 + indices[:, 1]]
        return (features * weights[:, None]).sum(-1)

    def forward(self, features, controls):
        controls = controls[:, None] if controls.ndim == 2 else controls
        knots = features.new_tensor((0., .5, 1.))
        basis = (1 - (controls[..., None] - knots).abs() * 2).clamp_min(0)
        coefficients = (basis[..., 0, :, None] * basis[..., 1, None, :]).flatten(-2)
        return (features * (coefficients @ self.weight)).sum(-1)


class StableChargeResidual(torch.nn.Module):
    def __init__(self, base, bank, readout):
        super().__init__()
        self.base, self.bank, self.readout = base, bank, readout
        self.control_count = base.control_count
        widths = getattr(base, 'export_state_widths', [layer.hidden_size for layer in base.modules() if isinstance(layer, torch.nn.LSTM)])
        self.export_state_widths = [*widths, bank.width]
        self.layers = len(self.export_state_widths)

    @property
    def state_floats_per_mono_stream(self):
        return sum(self.export_state_widths) * 2

    def forward(self, dry, controls, state=None):
        if state is not None and len(state) != self.layers:
            raise ValueError('wrong charge residual state count')
        original, next_base = self.base(dry, controls, None if state is None else state[:-1])
        features, next_charge = self.bank(dry, None if state is None else state[-1])
        return original + self.readout(features, controls), (*next_base, next_charge)
