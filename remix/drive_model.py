"""Specialized Drive-control estimator over paired time-frequency evidence."""

from __future__ import annotations

import torch

from .model import Estimate

IMAGE_SIZE = (32, 32)
QUANTILES = 33
STATISTICS = QUANTILES * 3


@torch.no_grad()
def drive_features(
    dry: torch.Tensor, wet: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep spectral position plus nonlinear amplitude-distribution changes."""

    if dry.ndim != 2 or wet.shape != dry.shape:
        raise ValueError(f"expected matching [batch,time] audio, got {dry.shape}/{wet.shape}")
    window = torch.hann_window(2_048, device=dry.device, dtype=dry.dtype)
    images = []
    for audio in (dry, wet):
        magnitude = torch.stft(
            audio,
            n_fft=2_048,
            hop_length=1_024,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        ).abs()
        images.append(
            torch.nn.functional.adaptive_avg_pool2d(
                torch.log1p(magnitude).unsqueeze(1), IMAGE_SIZE
            )
        )
    image = torch.cat((images[0], images[1], images[1] - images[0]), dim=1)
    levels = torch.linspace(0.0, 1.0, QUANTILES, device=dry.device, dtype=dry.dtype)
    statistics = torch.cat(
        tuple(
            torch.quantile(value.abs(), levels, dim=1).transpose(0, 1)
            for value in (dry, wet, wet - dry)
        ),
        dim=1,
    )
    scale = statistics[:, QUANTILES : 2 * QUANTILES].amax(dim=1, keepdim=True)
    statistics = torch.log1p(statistics / scale.clamp_min(1.0e-6) * 100.0)
    return image, statistics


class DriveControlEstimator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 24, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.GELU(),
            torch.nn.Conv2d(24, 48, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(48),
            torch.nn.GELU(),
            torch.nn.Conv2d(48, 96, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(96),
            torch.nn.GELU(),
        )
        self.network = torch.nn.Sequential(
            torch.nn.Linear(96 * 4 * 4 + STATISTICS, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 3),
        )

    def forward(self, image: torch.Tensor, statistics: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((self.encoder(image).flatten(1), statistics), dim=1))


def apply_drive_estimate(
    estimate: Estimate,
    dry: torch.Tensor,
    wet: torch.Tensor,
    model: DriveControlEstimator,
) -> Estimate:
    controls = estimate.control_logits.clone()
    image, statistics = drive_features(dry, wet)
    controls[:, :3] = model(image, statistics)
    return Estimate(estimate.order_logits, controls)
