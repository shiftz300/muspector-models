"""Float32 CPU-only ONNX runtime for offline stable-effect experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime
import torch


class OnnxStableEffect(torch.nn.Module):
    def __init__(self, checkpoint: Path, graph: Path) -> None:
        super().__init__()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("checkpoint_sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError("ONNX graph does not match checkpoint provenance")
        self.control_count = int(metadata["control_count"])
        self.state_widths = tuple(int(value) for value in metadata["state_widths"].split(","))
        self.layers = len(self.state_widths)
        self.register_parameter("cpu_marker", torch.nn.Parameter(torch.zeros(()), requires_grad=False))

    @property
    def state_floats_per_mono_stream(self) -> int:
        return 2 * sum(self.state_widths)

    def forward(self, dry: torch.Tensor, controls: torch.Tensor, state=None):
        if dry.device.type != "cpu" or controls.device.type != "cpu":
            raise ValueError("offline ONNX stable runtime is CPU-only")
        if dry.ndim != 2 or dry.shape[1] == 0 or not torch.isfinite(dry).all():
            raise ValueError("invalid ONNX Dry input")
        if controls.ndim == 2:
            controls = controls[:, None, :].expand(-1, dry.shape[1], -1)
        if controls.shape != (dry.shape[0], dry.shape[1], self.control_count):
            raise ValueError("invalid ONNX controls shape")
        if not torch.isfinite(controls).all() or torch.any(controls < 0) or torch.any(controls > 1):
            raise ValueError("invalid ONNX normalized controls")
        if state is None:
            state = tuple((torch.zeros(1, dry.shape[0], width), torch.zeros(1, dry.shape[0], width)) for width in self.state_widths)
        if len(state) != self.layers:
            raise ValueError("invalid ONNX state layer count")
        feeds = {"dry": np.ascontiguousarray(dry.detach().numpy(), dtype=np.float32),
                 "controls": np.ascontiguousarray(controls.detach().numpy(), dtype=np.float32)}
        for index, ((hidden, cell), width) in enumerate(zip(state, self.state_widths)):
            if hidden.shape != (1, dry.shape[0], width) or cell.shape != hidden.shape:
                raise ValueError("invalid ONNX state shape")
            feeds[f"h{index}"] = hidden.detach().numpy()
            feeds[f"c{index}"] = cell.detach().numpy()
        outputs = self.session.run(None, feeds)
        if any(not np.isfinite(value).all() for value in outputs):
            raise ValueError("ONNX stable runtime produced non-finite output/state")
        return torch.from_numpy(outputs[0]), tuple((torch.from_numpy(outputs[1 + 2*i]), torch.from_numpy(outputs[2 + 2*i])) for i in range(self.layers))


def load_onnx_stable_effect(checkpoint: Path, graph: Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return OnnxStableEffect(checkpoint, graph).eval(), payload
