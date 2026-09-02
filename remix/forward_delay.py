"""Hybrid physical/learned forward renderer for Delay research."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from .forward_drive import FORWARD_RATE, _latin_hypercube, _segment
from .pedalboard_renderer import render_pedalboard_chain
from .quality import validate_render
from .render import render_chain
from .spec import ChainSpec, Delay


DelayForwardDomain = Literal["reference", "alternate", "stress", "challenge", "pedalboard"]


class DelayForwardRenderer(torch.nn.Module):
    """Exact echo geometry with a tiny learned causal repeat-path filter."""

    def __init__(self, fir_taps: int = 64) -> None:
        super().__init__()
        if fir_taps < 8:
            raise ValueError("Delay repeat FIR needs at least eight taps")
        self.fir_taps = fir_taps
        initial_cutoff = 5_500.0
        normalized = (initial_cutoff - 1_500.0) / (12_000.0 - 1_500.0)
        self.cutoff_logit = torch.nn.Parameter(
            torch.tensor(math.log(normalized / (1.0 - normalized)), dtype=torch.float32)
        )
        self.blend_logits = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def learned_parameters(self) -> dict[str, float | list[float]]:
        with torch.no_grad():
            return {
                "cutoff_hz": float(self._cutoff()),
                "basis_weights": torch.softmax(self.blend_logits, dim=0).cpu().tolist(),
                "fir_taps": self.fir_taps,
            }

    def _cutoff(self) -> torch.Tensor:
        return 1_500.0 + 10_500.0 * torch.sigmoid(self.cutoff_logit)

    def _basis_kernels(self, dry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cutoff = self._cutoff()
        coefficient = torch.exp(
            torch.as_tensor(-2.0 * math.pi / FORWARD_RATE, device=dry.device, dtype=dry.dtype)
            * cutoff
        )
        lag = torch.arange(self.fir_taps, device=dry.device, dtype=dry.dtype)
        first_order = (1.0 - coefficient) * torch.pow(coefficient, lag)
        second_order = (lag + 1.0) * torch.square(1.0 - coefficient) * torch.pow(
            coefficient, lag
        )
        # Truncation is negligible at these cutoffs; normalize DC exactly so
        # the learned coloration cannot introduce a hidden level control.
        first_order = first_order / first_order.sum().clamp_min(1.0e-8)
        second_order = second_order / second_order.sum().clamp_min(1.0e-8)
        return first_order, second_order

    def _repeat_path_with_history(
        self,
        dry: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        if history.shape != (dry.shape[0], self.fir_taps - 1):
            raise ValueError("Delay FIR history shape is incompatible")
        first_order, second_order = self._basis_kernels(dry)
        context = torch.cat((history, dry), dim=1).unsqueeze(1)

        def causal_fir(kernel: torch.Tensor) -> torch.Tensor:
            return functional.conv1d(
                context,
                kernel.flip(0).view(1, 1, -1),
            ).squeeze(1)

        weights = torch.softmax(self.blend_logits, dim=0)
        return (
            dry * weights[0]
            + causal_fir(first_order) * weights[1]
            + causal_fir(second_order) * weights[2]
        )

    def _repeat_path(self, dry: torch.Tensor) -> torch.Tensor:
        history = torch.zeros(
            (dry.shape[0], self.fir_taps - 1),
            device=dry.device,
            dtype=dry.dtype,
        )
        return self._repeat_path_with_history(dry, history)

    def forward(self, dry: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        if dry.ndim != 2:
            raise ValueError(f"expected dry [batch,time], got {tuple(dry.shape)}")
        if controls.ndim != 2 or controls.shape != (dry.shape[0], 3):
            raise ValueError(
                f"expected controls [batch,3] for {dry.shape[0]} examples, "
                f"got {tuple(controls.shape)}"
            )
        if not torch.isfinite(dry).all() or not torch.isfinite(controls).all():
            raise ValueError("Delay forward input contains non-finite values")
        if torch.any(controls < 0.0) or torch.any(controls > 1.0):
            raise ValueError("Delay forward controls must be normalized to [0,1]")

        time_ms = 40.0 * torch.pow(25.0, controls[:, 0])
        delay_samples = torch.round(time_ms * (FORWARD_RATE / 1_000.0)).long()
        feedback = controls[:, 1] * 0.9
        mix = controls[:, 2:3] * 0.7
        repeat = self._repeat_path(dry)
        wet = torch.zeros_like(dry)
        positions = torch.arange(dry.shape[1], device=dry.device).unsqueeze(0)
        maximum_echoes = (dry.shape[1] - 1) // round(40.0 * FORWARD_RATE / 1_000.0)
        for echo in range(1, maximum_echoes + 1):
            source_positions = positions - echo * delay_samples.unsqueeze(1)
            active = source_positions >= 0
            shifted = torch.gather(repeat, 1, source_positions.clamp_min(0))
            amplitude = torch.pow(feedback, echo - 1).unsqueeze(1)
            wet = wet + torch.where(active, shifted * amplitude, 0.0)
        rendered = dry * (1.0 - mix) + wet * mix
        if not torch.isfinite(rendered).all():
            raise ValueError("Delay forward renderer produced non-finite audio")
        return rendered


@dataclass(frozen=True)
class DelayStreamState:
    dry_history: torch.Tensor
    delay_line: torch.Tensor
    controls: torch.Tensor


def stream_delay_block(
    model: DelayForwardRenderer,
    dry: torch.Tensor,
    controls: torch.Tensor,
    state: DelayStreamState | None = None,
) -> tuple[torch.Tensor, DelayStreamState]:
    """Render a block no longer than the physical delay and carry exact state."""

    if dry.ndim != 2 or controls.shape != (dry.shape[0], 3):
        raise ValueError("streaming Delay expects dry [batch,time] and controls [batch,3]")
    if not torch.isfinite(dry).all() or not torch.isfinite(controls).all():
        raise ValueError("streaming Delay input contains non-finite values")
    if torch.any(controls < 0.0) or torch.any(controls > 1.0):
        raise ValueError("streaming Delay controls must be normalized to [0,1]")
    time_ms = 40.0 * torch.pow(25.0, controls[:, 0])
    delay_samples = torch.round(time_ms * (FORWARD_RATE / 1_000.0)).long()
    if not torch.all(delay_samples == delay_samples[0]):
        raise ValueError("a streaming Delay batch must share one Time setting")
    delay = int(delay_samples[0])
    if dry.shape[1] > delay:
        raise ValueError(f"stream block {dry.shape[1]} exceeds physical delay {delay}")
    if state is None:
        history = torch.zeros(
            (dry.shape[0], model.fir_taps - 1),
            device=dry.device,
            dtype=dry.dtype,
        )
        delay_line = torch.zeros(
            (dry.shape[0], delay),
            device=dry.device,
            dtype=dry.dtype,
        )
    else:
        if state.delay_line.shape != (dry.shape[0], delay):
            raise ValueError("streaming Delay state has the wrong delay-line geometry")
        if not torch.equal(state.controls, controls):
            raise ValueError("Delay controls cannot change while reusing stream state")
        history = state.dry_history
        delay_line = state.delay_line
    repeat = model._repeat_path_with_history(dry, history)
    delayed = delay_line[:, : dry.shape[1]]
    feedback = controls[:, 1:2] * 0.9
    mix = controls[:, 2:3] * 0.7
    write = repeat + feedback * delayed
    next_delay_line = torch.cat((delay_line[:, dry.shape[1] :], write), dim=1)
    context = torch.cat((history, dry), dim=1)
    next_state = DelayStreamState(
        dry_history=context[:, -(model.fir_taps - 1) :],
        delay_line=next_delay_line,
        controls=controls.clone(),
    )
    rendered = dry * (1.0 - mix) + delayed * mix
    return rendered, next_state


def normalized_delay_controls(effect: Delay) -> np.ndarray:
    effect.validate()
    return np.asarray(
        (
            math.log(effect.time_ms / 40.0) / math.log(25.0),
            effect.feedback / 0.9,
            effect.mix / 0.7,
        ),
        dtype=np.float32,
    )


def controls_to_delay(controls: Sequence[float]) -> Delay:
    if len(controls) != 3:
        raise ValueError(f"expected three Delay controls, got {len(controls)}")
    values = tuple(float(value) for value in controls)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"normalized Delay controls are invalid: {values}")
    return Delay(40.0 * 25.0 ** values[0], values[1] * 0.9, values[2] * 0.7)


class DelayForwardDataset(Dataset):
    """Offline Delay pairs with exact physical controls and source isolation."""

    def __init__(
        self,
        dry_paths: Sequence[Path],
        samples: int,
        frames: int,
        *,
        seed: int,
        domains: Sequence[DelayForwardDomain],
        input_peak_ranges: Sequence[tuple[float, float]] = ((0.04, 0.08), (0.08, 0.24), (0.24, 0.5)),
    ) -> None:
        if not dry_paths or samples <= 0 or frames <= FORWARD_RATE or not domains:
            raise ValueError("Delay dataset needs sources, domains, and >1 second segments")
        ranges = tuple(input_peak_ranges)
        if not ranges or any(
            len(values) != 2
            or not all(math.isfinite(value) for value in values)
            or not 0.0 < values[0] <= values[1] <= 1.0
            for values in ranges
        ):
            raise ValueError("Delay input peak ranges are invalid")
        self.dry_paths = tuple(dry_paths)
        self.samples = samples
        self.frames = frames
        self.seed = seed
        self.domains = tuple(domains)
        self.input_peak_ranges = ranges
        self.controls = _latin_hypercube(samples, 3, seed ^ 0x444C4159)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rng = random.Random(self.seed + index * 104_729)
        dry = _segment(self.dry_paths[index % len(self.dry_paths)], self.frames, rng)
        peak = max(float(np.max(np.abs(dry))), 1.0e-5)
        target_peak = rng.uniform(*self.input_peak_ranges[index % len(self.input_peak_ranges)])
        dry = (dry * (target_peak / peak)).astype(np.float32)
        controls = self.controls[index]
        effect = controls_to_delay(controls)
        domain = self.domains[index % len(self.domains)]
        spec = ChainSpec((effect,))
        if domain == "pedalboard":
            wet = render_pedalboard_chain(dry, spec, FORWARD_RATE)
        else:
            wet = render_chain(dry, spec, FORWARD_RATE, domain)
        validate_render(dry, wet)
        return {
            "dry": torch.from_numpy(dry.copy()),
            "wet": torch.from_numpy(wet.copy()),
            "controls": torch.from_numpy(controls.copy()),
            "domain": domain,
        }
