"""Independent standard-operator runtime for stable conditioned ASRNN effects."""

from __future__ import annotations

from pathlib import Path
import math

import torch


class StableEffectRenderer(torch.nn.Module):
    """Deep conditioned LSTM with explicit state and configurable controls."""

    def __init__(
        self,
        control_count: int,
        hidden_size: int = 8,
        layers: int = 4,
        input_coef: float = 21.4,
        inverted_controls: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        if control_count <= 0 or hidden_size <= 0 or layers <= 0 or input_coef <= 0.0:
            raise ValueError("stable effect runtime dimensions must be positive")
        if any(index < 0 or index >= control_count for index in inverted_controls):
            raise ValueError("stable effect inverted control index is outside the control set")
        self.control_count = control_count
        self.hidden_size = hidden_size
        self.layers = layers
        self.input_coef = float(input_coef)
        self.inverted_controls = tuple(inverted_controls)
        recurrent = [torch.nn.LSTM(1 + control_count, hidden_size, batch_first=True)]
        recurrent.extend(
            torch.nn.LSTM(hidden_size + control_count, hidden_size, batch_first=True)
            for _ in range(1, layers)
        )
        self.rnn_layers = torch.nn.ModuleList(recurrent)
        self.output_layer = torch.nn.Linear(hidden_size, 1, bias=False)

    @property
    def state_floats_per_mono_stream(self) -> int:
        return self.layers * self.hidden_size * 2

    def _official_controls(self, controls: torch.Tensor, frames: int) -> torch.Tensor:
        if controls.ndim == 2:
            if controls.shape[1] != self.control_count:
                raise ValueError("stable effect controls have the wrong width")
            controls = controls.unsqueeze(1).expand(-1, frames, -1)
        elif controls.ndim != 3 or controls.shape[1:] != (frames, self.control_count):
            raise ValueError(
                "stable effect controls must be [batch,controls] or "
                "[batch,time,controls]"
            )
        physical = controls.clone()
        for index in self.inverted_controls:
            physical[..., index] = 1.0 - physical[..., index]
        return physical.mul(2.0).sub(1.0)

    def encode(
        self,
        dry: torch.Tensor,
        controls: torch.Tensor,
        state: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        if not torch.jit.is_tracing():
            if dry.ndim != 2:
                raise ValueError("stable effect renderer expects dry [batch,time]")
            if controls.shape[0] != dry.shape[0]:
                raise ValueError("stable effect control batch differs from audio")
            if not torch.isfinite(dry).all() or not torch.isfinite(controls).all():
                raise ValueError("stable effect inputs must be finite")
            if torch.any(controls < 0.0) or torch.any(controls > 1.0):
                raise ValueError("stable effect controls must be normalized")
            if state is not None and len(state) != self.layers:
                raise ValueError("stable effect recurrent state has the wrong layer count")
        condition = self._official_controls(controls, dry.shape[1])
        value = dry.unsqueeze(-1) * self.input_coef
        previous = (None,) * self.layers if state is None else state
        next_states = []
        for layer, layer_state in zip(self.rnn_layers, previous):
            value, next_state = layer(torch.cat((value, condition), dim=2), layer_state)
            next_states.append(next_state)
        return value, tuple(next_states)

    def forward(
        self,
        dry: torch.Tensor,
        controls: torch.Tensor,
        state: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        value, next_states = self.encode(dry, controls, state)
        rendered = self.output_layer(value).squeeze(-1)
        if not torch.jit.is_tracing() and not torch.isfinite(rendered).all():
            raise ValueError("stable effect renderer produced non-finite audio")
        return rendered, next_states


class StableEffectEnsemble(torch.nn.Module):
    """Fixed convex average of causal predictors, with separate flattened states.

    Coefficients are model parameters fixed before evaluation, never inferred
    from an input clip's peak/loudness or adapted after seeing its Wet target.
    """

    def __init__(self, members: list[StableEffectRenderer], weights: list[float]) -> None:
        super().__init__()
        if not members or len(members) != len(weights):
            raise ValueError("ensemble requires one weight per member")
        if any(not math.isfinite(weight) or weight <= 0 for weight in weights) or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError("ensemble weights must be positive and sum to one")
        self.control_count = members[0].control_count
        if any(member.control_count != self.control_count or member.inverted_controls != members[0].inverted_controls for member in members):
            raise ValueError("ensemble members disagree on control semantics")
        self.members = torch.nn.ModuleList(members)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.layers = sum(member.layers for member in members)

    @property
    def state_floats_per_mono_stream(self) -> int:
        return sum(member.state_floats_per_mono_stream for member in self.members)

    def forward(self, dry: torch.Tensor, controls: torch.Tensor, state=None):
        if state is not None and len(state) != self.layers:
            raise ValueError("ensemble recurrent state has the wrong layer count")
        outputs, states, offset = [], [], 0
        for member, weight in zip(self.members, self.weights):
            previous = None if state is None else state[offset:offset + member.layers]
            rendered, next_state = member(dry, controls, previous)
            outputs.append(rendered * weight)
            states.extend(next_state)
            offset += member.layers
        return torch.stack(outputs).sum(0), tuple(states)


class ConditionedReadoutEffect(torch.nn.Module):
    """Stable core with a bias-free, continuous control-conditioned readout.

    The nine readout vectors are trained model parameters, not target-derived
    per-clip gains. Bilinear interpolation is convex over the full control box.
    """
    def __init__(self, base: StableEffectRenderer, weights: torch.Tensor):
        super().__init__()
        if base.control_count != 2 or weights.shape != (3, 3, base.hidden_size):
            raise ValueError('conditioned readout requires a 3x3 two-control grid')
        if not torch.isfinite(weights).all():
            raise ValueError('non-finite readout weights')
        self.base = base
        self.readout = torch.nn.Parameter(weights.detach().clone())
        self.control_count, self.layers, self.hidden_size = 2, base.layers, base.hidden_size

    @property
    def state_floats_per_mono_stream(self):
        return self.base.state_floats_per_mono_stream

    def forward(self, dry, controls, state=None):
        hidden, next_state = self.base.encode(dry, controls, state)
        condition = controls[:, None, :] if controls.ndim == 2 else controls
        knots = torch.tensor((0., .5, 1.), device=controls.device, dtype=controls.dtype)
        basis = (1. - (condition[..., None] - knots).abs() * 2.).clamp_min(0.)
        coefficients = (basis[..., 0, :, None] * basis[..., 1, None, :]).flatten(-2)
        weight = coefficients @ self.readout.flatten(0, 1)
        output = (hidden * weight).sum(-1)
        if not torch.jit.is_tracing() and not torch.isfinite(output).all():
            raise ValueError('conditioned readout produced non-finite audio')
        return output, next_state


class StableEffectFIRResidual(torch.nn.Module):
    """Stable recurrent core plus a learned zero-preserving causal FIR branch."""
    def __init__(self, base: StableEffectRenderer, filters: torch.Tensor):
        super().__init__()
        if base.control_count != 2 or filters.ndim != 3 or filters.shape[:2] != (9, 6) or filters.shape[2] % 2 != 1 or filters.shape[2] < 3:
            raise ValueError('DFZ FIR residual requires nine six-basis odd-length filters')
        if not torch.isfinite(filters).all():raise ValueError('non-finite FIR weights')
        self.base=base;self.filters=torch.nn.Parameter(filters.detach().clone())
        self.control_count=2;self.layers=base.layers+1;self.taps=filters.shape[2]
        self.export_state_widths=[layer.hidden_size for layer in base.rnn_layers]+[(self.taps-1)//2]

    @property
    def state_floats_per_mono_stream(self):return self.base.state_floats_per_mono_stream+self.taps-1

    @staticmethod
    def basis(dry):
        return torch.stack((dry,torch.tanh(dry*2),torch.tanh(dry*8),torch.tanh(dry*32),torch.tanh(dry*128),torch.tanh(dry*512)),1)

    def forward(self,dry,controls,state=None):
        if state is not None and len(state)!=self.layers:raise ValueError('wrong FIR residual state count')
        baseline,base_state=self.base(dry,controls,None if state is None else state[:-1])
        # The last state pair stores two halves of raw Dry history, not gates.
        history=dry.new_zeros(dry.shape[0],self.taps-1) if state is None else torch.cat((state[-1][0].squeeze(0),state[-1][1].squeeze(0)),1)
        extended=torch.cat((history,dry),1)
        residual=torch.nn.functional.conv1d(self.basis(extended),self.filters.flip(-1)).transpose(1,2)
        condition=controls[:,None,:] if controls.ndim==2 else controls
        knots=torch.tensor((0.,.5,1.),device=controls.device,dtype=controls.dtype)
        basis=(1-(condition[...,None]-knots).abs()*2).clamp_min(0)
        coefficient=(basis[...,0,:,None]*basis[...,1,None,:]).flatten(-2)
        output=baseline+(residual*coefficient).sum(-1)
        next_history=extended[:,-(self.taps-1):];half=(self.taps-1)//2
        next_state=(*base_state,(next_history[:,:half].unsqueeze(0),next_history[:,half:].unsqueeze(0)))
        if not torch.jit.is_tracing() and not torch.isfinite(output).all():raise ValueError('non-finite FIR output')
        return output,next_state


def stable_effect_from_payload(payload: dict) -> StableEffectRenderer | StableEffectEnsemble:
    if payload.get('schema') == 10 and payload.get('architecture') == 'stable-dfz-charge-residual':
        from .stable_charge_residual import ChargeBank, ChargeReadout, StableChargeResidual
        if payload.get('device') != 'dfz' or payload.get('sample_rate') != 48000 or payload['base'].get('schema') not in (2, 7, 8) or payload['base'].get('device') != 'dfz':
            raise ValueError('incompatible DFZ charge residual')
        bank=ChargeBank(payload['charge_milliseconds']);readout=ChargeReadout(bank.width+1)
        readout.load_state_dict(payload['readout_state_dict'],strict=True)
        return StableChargeResidual(stable_effect_from_payload(payload['base']),bank,readout).eval()
    if payload.get('schema') == 9 and payload.get('architecture') == 'stable-dfz-composite-ensemble':
        from .stable_composite_ensemble import StableCompositeEnsemble
        if payload.get('device') != 'dfz' or payload.get('sample_rate') != 48000 or not payload.get('members'):
            raise ValueError('incompatible DFZ composite ensemble')
        if any(member.get('schema') not in (2, 7, 8) or member.get('device') != 'dfz' or member.get('sample_rate') != 48000 for member in payload['members']):
            raise ValueError('composite ensemble requires audited same-device member types')
        return StableCompositeEnsemble([stable_effect_from_payload(member) for member in payload['members']], payload['weights']).eval()
    if payload.get('schema') == 8 and payload.get('architecture') == 'stable-skip-nonlinear-readout':
        from .stable_nonlinear_readout import NonlinearReadout
        from .stable_skip_readout import StableSkipReadout
        if payload.get('device') != 'dfz' or payload.get('sample_rate') != 48000 or payload['base'].get('schema') != 2 or payload['base'].get('device') != 'dfz':raise ValueError('incompatible skip readout checkpoint')
        base=stable_effect_from_payload(payload['base']);readout=NonlinearReadout(1+base.layers*base.hidden_size,payload['readout_width'])
        readout.load_state_dict(payload['readout_state_dict'],strict=True)
        return StableSkipReadout(base,readout).eval()
    if payload.get('schema') == 7 and payload.get('architecture') == 'stable-nonlinear-readout':
        from .stable_nonlinear_readout import NonlinearReadout, StableNonlinearReadout
        if payload.get('device') != 'dfz' or payload.get('sample_rate') != 48000 or payload['base'].get('schema') != 2 or payload['base'].get('device') != 'dfz':raise ValueError('incompatible nonlinear readout checkpoint')
        base = stable_effect_from_payload(payload['base'])
        readout = NonlinearReadout(base.hidden_size, payload['readout_width'])
        readout.load_state_dict(payload['readout_state_dict'], strict=True)
        return StableNonlinearReadout(base, readout).eval()
    if payload.get('schema') == 6 and payload.get('architecture') == 'stable-zero-gated-tcn-residual':
        from .stable_tcn_residual import ZeroGatedTCN,StableTCNResidual
        if payload.get('device') != 'dfz' or payload.get('sample_rate') != 48000 or payload['base'].get('schema') != 2 or payload['base'].get('device') != 'dfz':raise ValueError('incompatible TCN residual checkpoint')
        tcn=ZeroGatedTCN(payload['tcn_width'],payload['tcn_blocks']);tcn.load_state_dict(payload['tcn_state_dict'],strict=True)
        return StableTCNResidual(stable_effect_from_payload(payload['base']),tcn).eval()
    if payload.get('schema') == 5 and payload.get('architecture') == 'stable-causal-fir-residual':
        if payload.get('device') != 'dfz' or payload.get('sample_rate') != 48000 or payload['base'].get('schema') != 2 or payload['base'].get('device') != 'dfz':raise ValueError('incompatible FIR residual checkpoint')
        return StableEffectFIRResidual(stable_effect_from_payload(payload['base']),payload['filters']).eval()
    if payload.get('schema') == 4 and payload.get('architecture') == 'stable-conditioned-readout':
        if payload.get('device') != 'dfz' or payload.get('sample_rate') != 48000 or payload['base'].get('schema') != 2 or payload['base'].get('device') != 'dfz':
            raise ValueError('incompatible conditioned readout checkpoint')
        return ConditionedReadoutEffect(stable_effect_from_payload(payload['base']), payload['readout']).eval()
    if payload.get("schema") == 3 and payload.get("architecture") == "stable-conditioned-lstm-ensemble":
        if payload.get("sample_rate") != 48000:
            raise ValueError("incompatible ensemble sample rate")
        members = payload["members"]
        if any(member.get("schema") != 2 or member.get("device") != payload["device"] for member in members):
            raise ValueError("ensemble requires same-device standard stable members")
        return StableEffectEnsemble([stable_effect_from_payload(member) for member in members], payload["weights"]).eval()
    if (
        payload.get("schema") != 2
        or payload.get("sample_rate") != 48_000
        or payload.get("architecture") != "stable-conditioned-lstm"
    ):
        raise ValueError("incompatible stable effect checkpoint")
    model = StableEffectRenderer(
        control_count=int(payload["control_count"]),
        hidden_size=int(payload["hidden_size"]),
        layers=int(payload["layers"]),
        input_coef=float(payload["input_coef"]),
        inverted_controls=tuple(int(value) for value in payload["inverted_controls"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def load_stable_effect(path: Path) -> tuple[StableEffectRenderer | StableEffectEnsemble, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return stable_effect_from_payload(payload), payload
