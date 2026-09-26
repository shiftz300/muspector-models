"""Content-invariant Wet-only tone encoder for EGDB-PG Amp+cab presets."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .egdb_pg_amp_data import RATE


class WetToneEncoder(nn.Module):
    def __init__(self, embedding_size: int = 64) -> None:
        super().__init__()
        if embedding_size < 16:
            raise ValueError("tone embedding is too small")
        self.embedding_size = embedding_size
        self.n_fft = 1024
        self.hop = 512
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.network = nn.Sequential(
            nn.Conv1d(2, 16, 9, stride=2, padding=4), nn.GELU(),
            nn.Conv1d(16, 24, 9, stride=2, padding=4), nn.GELU(),
            nn.Conv1d(24, 40, 9, stride=2, padding=4), nn.GELU(),
            nn.Conv1d(40, 64, 9, stride=2, padding=4), nn.GELU(),
            # Preserve coarse frequency location; pooling frequency to one bin
            # makes different cabinets collapse to the same representation.
            nn.AvgPool1d(2, 2), nn.Flatten(),
            nn.Linear(64 * 16, embedding_size),
        )

    def forward(self, wet: torch.Tensor) -> torch.Tensor:
        if wet.ndim != 2 or not torch.isfinite(wet).all():
            raise ValueError("tone encoder expects finite Wet [batch,time]")
        magnitude = torch.stft(
            wet, self.n_fft, self.hop, window=self.window,
            center=True, pad_mode="constant", return_complex=True,
        ).abs()
        db = torch.log1p(magnitude)
        global_mean = db.mean(dim=(-2, -1), keepdim=True)
        global_deviation = db.std(dim=(-2, -1), keepdim=True).clamp_min(1.0e-4)
        normalized = (db - global_mean) / global_deviation
        frequency_mean = normalized.mean(2)
        frequency_deviation = normalized.std(2)
        embedding = self.network(torch.stack((frequency_mean, frequency_deviation), dim=1))
        return F.normalize(embedding, dim=1)

    def manifest(self) -> dict:
        return {
            "schema": 1,
            "architecture": "wet-log-spectrum-content-invariant-tone-encoder",
            "sample_rate": RATE,
            "embedding_size": self.embedding_size,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "minimum_observation_seconds": 3.0,
            "input": "Wet audio only",
            "profile_id_input": False,
            "gain_category_input": False,
            "clean_or_oracle_input": False,
            "graph_order_input": False,
            "neighbor_effect_input": False,
        }
