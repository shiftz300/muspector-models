import json
import unittest

import numpy as np

from remix.proteus import ProteusRuntime


def zero_model(skip=1):
    return {
        "model_data": {
            "model": "SimpleRNN", "input_size": 1, "skip": skip,
            "output_size": 1, "unit_type": "LSTM", "num_layers": 1,
            "hidden_size": 40, "bias_fl": True,
        },
        "state_dict": {
            "rec.weight_ih_l0": np.zeros((160, 1)).tolist(),
            "rec.weight_hh_l0": np.zeros((160, 40)).tolist(),
            "rec.bias_ih_l0": np.zeros(160).tolist(),
            "rec.bias_hh_l0": np.zeros(160).tolist(),
            "lin.weight": np.zeros((1, 40)).tolist(),
            "lin.bias": [0.0],
        },
    }


class ProteusTests(unittest.TestCase):
    def test_zero_network_is_bit_exact_skip(self):
        runtime = ProteusRuntime.bytes(json.dumps(zero_model()).encode())
        source = np.linspace(-0.5, 0.5, 101, dtype=np.float32)
        before = source.copy()
        rendered = runtime.render(source, chunk=17)
        np.testing.assert_array_equal(rendered, source)
        np.testing.assert_array_equal(source, before)

    def test_rejects_knob_model_contract(self):
        model = zero_model(skip=1)
        model["model_data"]["input_size"] = 2
        with self.assertRaises(ValueError):
            ProteusRuntime(model)


if __name__ == "__main__":
    unittest.main()

