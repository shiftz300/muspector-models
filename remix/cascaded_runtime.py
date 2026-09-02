"""Reviewed graph boundary for cascaded experiments, not production admission."""
import math

import onnx
import torch

from .audit_multirate_fuzz import CheckedMultirate
from .cascaded_multirate_fuzz import ARCHITECTURE, GEOMETRY, CascadedMultirateFuzz
from .precheck_multirate_fuzz import OrtMultirate


def structure_bound(model):
    """Validate the canonical module graph before computing a real-arithmetic bound."""
    reference = CascadedMultirateFuzz(**GEOMETRY)
    if type(model) is not CascadedMultirateFuzz:
        raise ValueError("reviewed cascaded class required")
    actual_modules, expected_modules = dict(model.named_modules()), dict(reference.named_modules())
    if set(actual_modules) != set(expected_modules):
        raise ValueError("module graph differs")
    for name, expected in expected_modules.items():
        actual = actual_modules[name]
        if type(actual) is not type(expected):
            raise ValueError("module type differs")
        if isinstance(expected, torch.nn.Conv1d):
            if any(getattr(actual, key) != getattr(expected, key) for key in
                   ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "padding_mode")):
                raise ValueError("convolution geometry differs")
        if isinstance(expected, torch.nn.Linear):
            if actual.in_features != expected.in_features or actual.out_features != expected.out_features:
                raise ValueError("control projection geometry differs")
        if isinstance(expected, (torch.nn.Linear, torch.nn.Conv1d)) and (actual.bias is None) != (expected.bias is None):
            raise ValueError("bias/zero-origin policy differs")
    if (model.state_widths != reference.state_widths or model.fast_count != 20 or model.slow_count != 10
            or model.audio.dilations != reference.audio.dilations or model.audio.widths != reference.audio.widths
            or model.audio.stage_widths != reference.audio.stage_widths or model.audio.blocks_per_stage != 10
            or model.controller.dilations != reference.controller.dilations or model.sample_rate != 48000
            or model.control_count != 2):
        raise ValueError("causal clock/state geometry differs")
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("finite float32 weights required")

    def norm(layer):
        weight = layer.weight.detach().double()
        return float(weight.reshape(weight.shape[0], -1).abs().sum(1).max())

    # Each zero-origin tanh difference is bounded by2; its multiplier by1.5.
    # Unlike a final residual-only readout, the audio head sees bounded updates.
    skip_bound = sum(3*norm(layer)/math.sqrt(20) for layer in model.audio.skip_heads)
    output_bound = skip_bound*norm(model.audio.output)
    value_bound, fast_bounds = 21.4, []
    for index, layer in enumerate(model.audio.projections):
        if index%10 == 0:
            value_bound *= norm(model.audio.rechannels[index//10])
        value_bound += 3*norm(layer)/math.sqrt(10)
        fast_bounds.append(value_bound)
    slow_bound = norm(model.controller.input)*21.4**2
    slow_bound += sum(norm(layer)/math.sqrt(10) for layer in model.controller.projections)
    slow_bound *= norm(model.controller.output)
    if not all(math.isfinite(value) and 0 <= value < 1e30 for value in (output_bound, slow_bound, *fast_bounds)):
        raise ValueError("conservative finite bounds exceed representable budget")
    return {"architecture": ARCHITECTURE, "geometry": GEOMETRY, "fast_update_bound": 3.,
            "learned_recurrent_matrices": False, "audio_receptive_field_frames": 4093,
            "controller_receptive_field_blocks": 2047, "controller_block_frames": 64,
            "zero_audio_tail_expiry_frames": 4092, "state_tensor_bytes_mono": 262184,
            "conservative_output_bound_for_abs_dry_le1": output_bound,
            "conservative_slow_held_bound_for_abs_dry_le1": slow_bound,
            "bound_scope": "real-arithmetic BIBO at zero initial state and abs(Dry)<=1; not fidelity, a limiter or arbitrary-input floating-point safety"}


def validate_graph(graph, checkpoint_digest):
    metadata = {entry.key: entry.value for entry in graph.metadata_props}
    if any(metadata.get(key) != value for key, value in
           {"architecture": ARCHITECTURE, "checkpoint_sha256": checkpoint_digest, "sample_rate": "48000",
            "precision": "float32", "trained": "true", "admitted": "false"}.items()):
        raise ValueError("same-checkpoint trained cascaded graph required")

    def tensor(value):
        if value.data_location == onnx.TensorProto.EXTERNAL or value.external_data:
            raise ValueError("external weights forbidden")
        kind = onnx.TensorProto.DataType.Name(value.data_type)
        if ("FLOAT" in kind or kind == "DOUBLE") and kind != "FLOAT":
            raise ValueError("non-float32 constants/weights forbidden")

    if graph.functions or graph.graph.sparse_initializer:
        raise ValueError("unreviewed local functions/sparse weights forbidden")
    for value in graph.graph.initializer:
        tensor(value)
    for node in graph.graph.node:
        if node.domain not in ("", "ai.onnx") or "quantize" in node.op_type.lower() or node.op_type.startswith("QLinear"):
            raise ValueError("custom or quantized operators forbidden")
        for attribute in node.attribute:
            if node.op_type == "Cast" and attribute.name == "to" and attribute.i in (
                    onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16, onnx.TensorProto.DOUBLE):
                raise ValueError("non-float32 arithmetic forbidden")
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensor(attribute.t)
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS, onnx.AttributeProto.TENSORS):
                raise ValueError("unreviewed nested graph/constants forbidden")
    widths = [width*2**index for width in (16, 8, 8) for index in range(10)]
    signature = {"dry": ("batch", "frames"), "controls": ("batch", "frames", 2)}
    signature.update({name: (1, "batch", width) for index, width in enumerate(widths) for name in (f"h{index}", f"c{index}")})
    signature.update(pending=("batch", 64), held=("batch", 8), phase=())
    output_signature = {"rendered": ("batch", "frames"), **{"next_"+name: shape for name, shape in signature.items()
                                                            if name not in ("dry", "controls")}}
    for actual, expected in ((graph.graph.input, signature), (graph.graph.output, output_signature)):
        if [value.name for value in actual] != list(expected):
            raise ValueError("graph ordered signature differs")
        for value in actual:
            tensor_type = value.type.tensor_type
            dtype = onnx.TensorProto.INT64 if value.name in ("phase", "next_phase") else onnx.TensorProto.FLOAT
            if tensor_type.elem_type != dtype or len(tensor_type.shape.dim) != len(expected[value.name]):
                raise ValueError("graph float32/integer-clock type differs")
            for axis, (dim, size) in enumerate(zip(tensor_type.shape.dim, expected[value.name])):
                # ONNX's dynamic Slice cannot infer this fixed64 output width.
                # CheckedMultirate validates the actual returned tensor on every call.
                if value.name == "next_pending" and axis == 1 and dim.dim_param:
                    continue
                if (isinstance(size, str) and not dim.dim_param) or (isinstance(size, int) and dim.dim_value != size):
                    raise ValueError("graph dynamic/static dimension differs")
    onnx.checker.check_model(graph, full_check=True)


def graph_runtime(path, model, checkpoint_digest):
    structure_bound(model)
    graph = onnx.load(path, load_external_data=False)
    validate_graph(graph, checkpoint_digest)
    runtime = OrtMultirate(path, model)
    states = [name for index in range(30) for name in (f"h{index}", f"c{index}")]+["pending", "held", "phase"]
    if runtime.names != ["dry", "controls", *states]:
        raise ValueError("graph input/state signature differs")
    if [value.name for value in runtime.session.get_outputs()] != ["rendered", *["next_"+name for name in states]]:
        raise ValueError("graph output/state signature differs")
    return CheckedMultirate(runtime, model)
