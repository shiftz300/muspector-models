import copy
import unittest

from .train_wide_multirate_fuzz import STEPS, learning_rate, verify_candidate_runtime


class WideTrainingPreconditions(unittest.TestCase):
    def evidence(self):
        return {"runtime_passed": False,
                "real_audio_cases": [{"parity_max": 1e-6, "onnx_stream_max": 0., "narrow_wide_max": 2.1e-6} for _ in range(54)],
                "dynamic_probes": [{"parity_max": 1e-6, "onnx_stream_max": 0., "cpu_stream_max": 1e-7,
                                    "phase_exact": True, "inputs_unchanged": True} for _ in range(3)],
                "safety": {backend: {"zero_peak": 0., "expired_tail_peak": 0., "future_prefix_error": 0., "quiet_peak": 1e-4}
                           for backend in ("cpu", "onnx")},
                "single_thread_rtf_including_validation": {backend: {"full": [.2] * 3, "stream1024": [.2] * 3}
                                                           for backend in ("cpu", "onnx")}}

    def test_new_training_init_does_not_claim_equivalent_replacement(self):
        data = self.evidence()
        before = copy.deepcopy(data)
        verify_candidate_runtime(data)
        self.assertEqual(data, before)
        self.assertFalse(data["runtime_passed"])

    def test_candidate_parity_is_never_relaxed(self):
        data = self.evidence()
        data["real_audio_cases"][0]["parity_max"] = 2.01e-6
        with self.assertRaises(ValueError):
            verify_candidate_runtime(data)

    def test_safety_and_timing_remain_required(self):
        for key, value in (("zero_peak", 1e-12), ("quiet_peak", .00101), ("expired_tail_peak", 1e-12), ("future_prefix_error", 1e-12)):
            data = self.evidence()
            data["safety"]["onnx"][key] = value
            with self.assertRaises(ValueError):
                verify_candidate_runtime(data)
        data = self.evidence()
        data["single_thread_rtf_including_validation"]["onnx"]["stream1024"][1] = 1.
        with self.assertRaises(ValueError):
            verify_candidate_runtime(data)

    def test_incomplete_or_mutating_replay_rejected(self):
        data = self.evidence()
        data["real_audio_cases"].pop()
        with self.assertRaises(ValueError):
            verify_candidate_runtime(data)
        data = self.evidence()
        data["dynamic_probes"][0]["inputs_unchanged"] = False
        with self.assertRaises(ValueError):
            verify_candidate_runtime(data)

    def test_fixed_learning_rate_schedule(self):
        self.assertAlmostEqual(learning_rate(250), 1e-3)
        self.assertAlmostEqual(learning_rate(4000), 1e-3)
        self.assertAlmostEqual(learning_rate(STEPS), 3e-5)
        with self.assertRaises(ValueError):
            learning_rate(0)


if __name__ == "__main__":
    unittest.main()
