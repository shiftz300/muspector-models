import copy
from pathlib import Path
import tempfile
import unittest

import onnx
import torch

from .cascaded_multirate_fuzz import ARCHITECTURE, GEOMETRY, CascadedMultirateFuzz
from .cascaded_runtime import graph_runtime, structure_bound, validate_graph
from .precheck_multirate_fuzz import export, streamed


class CascadedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(1014)
        cls.model = CascadedMultirateFuzz(**GEOMETRY).eval()
        with torch.no_grad():
            cls.model.audio.output.weight.normal_(0, .05)
        cls.directory = tempfile.TemporaryDirectory(prefix="muspector-cascaded-test-")
        cls.path = Path(cls.directory.name)/"synthetic.onnx"
        export(cls.model, cls.path)
        cls.graph = onnx.load(cls.path)
        metadata = {entry.key: entry.value for entry in cls.graph.metadata_props}
        metadata.update(architecture=ARCHITECTURE, checkpoint_sha256="a"*64, trained="true", admitted="false")
        del cls.graph.metadata_props[:]
        for key, value in metadata.items():
            entry = cls.graph.metadata_props.add()
            entry.key, entry.value = key, value
        onnx.save(cls.graph, cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_reviewed_structure_bound_and_no_gain_patch(self):
        evidence = structure_bound(self.model)
        self.assertEqual(evidence["audio_receptive_field_frames"], 4093)
        self.assertEqual(evidence["state_tensor_bytes_mono"], 262184)
        self.assertGreater(evidence["conservative_output_bound_for_abs_dry_le1"], 0)
        bad = copy.deepcopy(self.model)
        bad.audio.convolutions[3].padding = (1,)
        with self.assertRaises(ValueError):
            structure_bound(bad)
        bad = copy.deepcopy(self.model)
        bad.audio.raw_mixins[0].bias = torch.nn.Parameter(torch.zeros(16))
        with self.assertRaises(ValueError):
            structure_bound(bad)
        with self.assertRaises(ValueError):
            structure_bound(copy.deepcopy(self.model).double())

    def test_graph_identity_and_precision_fail_closed(self):
        validate_graph(self.graph, "a"*64)
        with self.assertRaises(ValueError):
            validate_graph(self.graph, "b"*64)
        for kind in ("external", "double", "phase", "static", "quantization"):
            bad = copy.deepcopy(self.graph)
            if kind == "external":
                bad.graph.initializer[0].data_location = onnx.TensorProto.EXTERNAL
            elif kind == "double":
                bad.graph.initializer[0].data_type = onnx.TensorProto.DOUBLE
            elif kind == "phase":
                bad.graph.input[-1].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            elif kind == "static":
                dim = bad.graph.input[0].type.tensor_type.shape.dim[1]
                dim.ClearField("dim_param")
                dim.dim_value = 257
            else:
                bad.graph.node[0].op_type = "DequantizeLinear"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_graph(bad, "a"*64)

    def test_actual_cpu_ort_irregular_callbacks_and_input_integrity(self):
        runtime = graph_runtime(self.path, self.model, "a"*64)
        dry, controls = torch.randn(2, 1033)*.03, torch.rand(2, 1033, 2)
        before = dry.clone(), controls.clone()
        with torch.inference_mode():
            reference = self.model(dry, controls)[0]
            whole = runtime(dry, controls)[0]
            chunks = streamed(runtime, dry, controls, [1, 17, 63, 257])[0]
        self.assertLess(float((reference-whole).abs().max()), 2e-6)
        self.assertTrue(torch.equal(whole, chunks))
        self.assertTrue(torch.equal(dry, before[0]) and torch.equal(controls, before[1]))


if __name__ == "__main__":
    unittest.main()
