"""Experimental zero-origin nonlinear readout of short causal audio history.

The states contain only past Dry and the frozen core's output.
Nothing is estimated from target Wet at inference, and no gain is fitted to a
clip. This experiment deliberately does not register a runtime schema.
"""
from __future__ import annotations

import math
import torch


LAGS = (0, 1, 2, 4, 8, 16, 32, 64)
FEATURES = 2 * len(LAGS)


class CausalTapBank(torch.nn.Module):
    """Exactly 64 real preceding samples of each input, not learned state."""

    def __init__(self):
        super().__init__()
        self.width = max(LAGS)

    def forward(self, dry, original, state=None):
        if not torch.jit.is_tracing() and (dry.ndim != 2 or original.shape != dry.shape):
            raise ValueError("aligned batch/time Dry and frozen core output required")
        old_dry, old_original = ((dry.new_zeros(dry.shape[0], self.width), dry.new_zeros(dry.shape[0], self.width))
                                 if state is None else (state[0].squeeze(0), state[1].squeeze(0)))
        x, p = torch.cat((old_dry, dry), 1), torch.cat((old_original, original), 1)
        columns = []
        for lag in LAGS:
            columns.extend((x[:, self.width - lag:self.width - lag + dry.shape[1]] * 21.4,
                            p[:, self.width - lag:self.width - lag + dry.shape[1]]))
        return torch.stack(columns, -1), (x[:, -self.width:].unsqueeze(0), p[:, -self.width:].unsqueeze(0))

    @torch.inference_mode()
    def cpu_features(self, dry, original):
        """Full causal recording, with padding only before sample zero."""
        if dry.ndim != 1 or original.shape != dry.shape or dry.device.type != "cpu":
            raise ValueError("complete aligned one-dimensional CPU arrays required")
        return self(dry[None], original[None])[0][0]


class DynamicCornerReadout(torch.nn.Module):
    """Nine small continuous-interpolated MLPs, each with f(z)-f(0) origin."""

    def __init__(self, rms=None, width=32):
        super().__init__()
        self.width = width
        self.register_buffer("rms", torch.ones(FEATURES) if rms is None else rms.detach().clone())
        if self.rms.shape != (FEATURES,) or not torch.isfinite(self.rms).all() or (self.rms <= 0).any():
            raise ValueError("finite positive fit-only RMS scale required")
        self.first = torch.nn.Parameter(torch.randn(9, FEATURES, width) / math.sqrt(FEATURES))
        self.first_bias = torch.nn.Parameter(torch.zeros(9, width))
        self.second = torch.nn.Parameter(torch.randn(9, width, width) / math.sqrt(width))
        self.second_bias = torch.nn.Parameter(torch.zeros(9, width))
        self.output = torch.nn.Parameter(torch.zeros(9, width))

    def at_training_knots(self, features, controls):
        ids = (controls * 2).round().long()
        corner = ids[:, 0] * 3 + ids[:, 1]
        value = features / self.rms
        first, second = self.first[corner], self.second[corner]
        b1, b2 = self.first_bias[corner, None], self.second_bias[corner, None]
        hidden = torch.tanh(torch.bmm(torch.tanh(torch.bmm(value, first) + b1), second) + b2)
        origin = torch.tanh(torch.bmm(torch.tanh(b1), second) + b2)
        return ((hidden - origin) * self.output[corner, None]).sum(-1) * .1

    def forward(self, features, controls):
        condition = controls[:, None] if controls.ndim == 2 else controls
        knots = features.new_tensor((0., .5, 1.))
        basis = (1 - (condition[..., None] - knots).abs() * 2).clamp_min(0)
        coefficients = (basis[..., 0, :, None] * basis[..., 1, None, :]).flatten(-2)
        value = features / self.rms
        branches = []
        for index in range(9):
            hidden = torch.tanh(torch.tanh(value @ self.first[index] + self.first_bias[index])
                                @ self.second[index] + self.second_bias[index])
            origin = torch.tanh(torch.tanh(self.first_bias[index]) @ self.second[index] + self.second_bias[index])
            branches.append(((hidden - origin) * self.output[index]).sum(-1) * .1)
        return (torch.stack(branches, -1) * coefficients).sum(-1)


class DynamicReadoutResidual(torch.nn.Module):
    def __init__(self, base, readout):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.bank, self.readout = CausalTapBank(), readout
        self.control_count = self.base.control_count
        widths = getattr(base, "export_state_widths", [layer.hidden_size for layer in base.modules()
                                                       if isinstance(layer, torch.nn.LSTM)])
        self.export_state_widths = [*widths, self.bank.width]

    def forward(self, dry, controls, state=None):
        if state is not None and len(state) != len(self.export_state_widths):
            raise ValueError("wrong dynamic-readout state count")
        original, next_base = self.base(dry, controls, None if state is None else state[:-1])
        features, next_bank = self.bank(dry, original, None if state is None else state[-1])
        return original + self.readout(features, controls), (*next_base, next_bank)
