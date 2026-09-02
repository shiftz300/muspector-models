"""Typed, unadmitted loader for the separate cascaded DFZ experiment."""
from pathlib import Path
import json

import torch

from .cascaded_multirate_fuzz import ARCHITECTURE, GEOMETRY, CascadedMultirateFuzz
from .precheck_multirate_fuzz import sha256


def model_from_payload(payload):
    if (payload.get("experimental_schema") != 1 or payload.get("architecture") != ARCHITECTURE
            or payload.get("geometry") != GEOMETRY or payload.get("device") != "dfz"
            or payload.get("sample_rate") != 48000):
        raise ValueError("exact cascaded DFZ float32 schema required")
    state = payload.get("state_dict")
    model = CascadedMultirateFuzz(**GEOMETRY)
    if not isinstance(state, dict) or set(state) != set(model.state_dict()):
        raise ValueError("complete cascaded state dictionary required")
    for name, reference in model.state_dict().items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.device.type != "cpu" or value.shape != reference.shape
                or not torch.isfinite(value).all()):
            raise ValueError(f"invalid source float32 tensor: {name}")
    model.load_state_dict(state, strict=True)
    return model.eval()


def load_candidate(checkpoint):
    checkpoint = Path(checkpoint)
    metrics = json.loads(checkpoint.with_name("metrics.json").read_text())
    if (metrics.get("architecture") != ARCHITECTURE or metrics.get("geometry") != GEOMETRY
            or metrics.get("completed_steps") != metrics.get("requested_steps")
            or not metrics.get("source_and_audio_reverified")
            or metrics.get("checkpoint_sha256") != sha256(checkpoint)):
        raise ValueError("complete same-checkpoint training evidence required")
    for name, digest in metrics["source_sha256"].items():
        if Path(name).name != name or sha256(Path(__file__).parent/name) != digest:
            raise ValueError("cascaded training implementation changed")
    return model_from_payload(torch.load(checkpoint, map_location="cpu", weights_only=True)), metrics
