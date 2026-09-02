"""Compact paired estimator for pairwise order and normalized controls."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Estimate:
    order_logits: torch.Tensor
    control_logits: torch.Tensor


class PairedEstimator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "correlation_lags",
            torch.unique(torch.logspace(0.9, 2.31, 32).round().to(torch.int64)),
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 24, 5, stride=2, padding=2),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU(),
            torch.nn.Conv2d(24, 48, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 96, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(),
            # Retain coarse frequency/time position. Global pooling erased the
            # echo/tail structure and local nonlinear interactions that carry
            # most ordering evidence.
            torch.nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(96 * 4 * 8 + 64 + len(self.correlation_lags), 192),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
        )
        self.order = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(192, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
            )
            for _ in range(3)
        )
        self.controls = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(192, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 3),
            )
            for _ in range(3)
        )

    @staticmethod
    def frontend(audio: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(1_024, device=audio.device)
        spectrum = torch.stft(
            audio,
            n_fft=1_024,
            hop_length=512,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        magnitude = torch.log1p(spectrum.abs())
        return torch.nn.functional.adaptive_avg_pool2d(magnitude.unsqueeze(1), (96, 216))

    def forward(self, dry: torch.Tensor, wet: torch.Tensor) -> Estimate:
        value = self.encode(dry, wet)
        order = torch.cat(tuple(head(value) for head in self.order), dim=1)
        controls = torch.cat(tuple(head(value) for head in self.controls), dim=1)
        return Estimate(order, controls)

    def encode(self, dry: torch.Tensor, wet: torch.Tensor) -> torch.Tensor:
        return self.shared(self.pre_shared_features(dry, wet))

    def pre_shared_features(self, dry: torch.Tensor, wet: torch.Tensor) -> torch.Tensor:
        """Return the frozen paired frontend before task-specific compression."""

        dry_features = self.frontend(dry)
        wet_features = self.frontend(wet)
        value = torch.cat((dry_features, wet_features, wet_features - dry_features), dim=1)
        value = torch.cat((self.encoder(value).flatten(1), self.physics(dry, wet)), dim=1)
        return value

    def physics(self, dry: torch.Tensor, wet: torch.Tensor) -> torch.Tensor:
        """Expose delay correlation and reverb-envelope evidence explicitly."""

        dry_energy = torch.nn.functional.adaptive_avg_pool1d(
            dry.square().unsqueeze(1), 64
        ).squeeze(1)
        wet_energy = torch.nn.functional.adaptive_avg_pool1d(
            wet.square().unsqueeze(1), 64
        ).squeeze(1)
        envelope_delta = torch.log1p(wet_energy * 10_000.0) - torch.log1p(
            dry_energy * 10_000.0
        )

        dry_small = torch.nn.functional.adaptive_avg_pool1d(dry.unsqueeze(1), 1_024).squeeze(1)
        wet_small = torch.nn.functional.adaptive_avg_pool1d(wet.unsqueeze(1), 1_024).squeeze(1)
        correlations = []
        for lag_tensor in self.correlation_lags:
            lag = int(lag_tensor)
            left = dry_small[:, :-lag]
            right = wet_small[:, lag:]
            numerator = (left * right).mean(dim=1)
            denominator = torch.sqrt(
                left.square().mean(dim=1) * right.square().mean(dim=1)
            ).clamp_min(1.0e-6)
            correlations.append(numerator / denominator)
        return torch.cat((envelope_delta, torch.stack(correlations, dim=1)), dim=1)


def masked_loss(estimate: Estimate, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    order = torch.nn.functional.binary_cross_entropy_with_logits(
        estimate.order_logits, batch["order"], reduction="none"
    )
    order = (order * batch["order_mask"]).sum() / batch["order_mask"].sum().clamp_min(1.0)
    control_mask = batch.get("control_train_mask", batch["control_mask"])
    controls = torch.nn.functional.smooth_l1_loss(
        torch.sigmoid(estimate.control_logits), batch["controls"], reduction="none", beta=0.05
    )
    controls = (controls * control_mask).sum() / control_mask.sum().clamp_min(1.0)
    total = order + controls
    return total, {"order": float(order.detach()), "controls": float(controls.detach())}
