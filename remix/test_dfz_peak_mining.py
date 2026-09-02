"""Offline synthetic CPU tests; no corpus/cache/GPU or audio-device access."""
import copy
import json
import unittest
from unittest.mock import patch

import torch

from .dfz_dynamic_readout import CausalTapBank, DynamicCornerReadout
from .dfz_peak_mining import FitPeakMiner, REFRESH_STEPS, mining_resource_bounds, module_sha256


class ProbeReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(.25))

    def at_training_knots(self, features, controls):
        return self.gain * features[..., 0]


def fixtures(frames=1_601):
    fit, cal, rows = [], [], []
    for a in (0, 50, 100):
        for b in (0, 50, 100):
            for take in range(1, 33):
                path = f"dfz/train/{a},{b},{take}.wav"
                (cal if take % 5 == 0 else fit).append({"path": path, "sha256": "a" * 64})
                if take % 5:
                    dry, original = torch.zeros(frames), torch.zeros(frames)
                    dry[12], dry[1_500], dry[1_300] = 20., .2, -.3
                    original[1_400] = .1
                    rows.append({"path": path, "dry": dry, "original": original,
                                 "controls": torch.tensor([a, b], dtype=torch.float32) / 100,
                                 "wet": torch.zeros(frames)})
    signature = {"source_sha256": "b" * 64, "frames_per_recording": frames,
                 "partitions": {"fit": fit, "calibration": cal}}
    return signature, rows


def mark_step(optimizer, readout, step):
    # Synthetic optimizer bookkeeping only; no optimization/training is run.
    for parameter in readout.parameters():
        optimizer.state[parameter]["step"] = torch.tensor(float(step))


class FitPeakMiningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(991)
        self.signature, self.rows = fixtures()
        self.model = ProbeReadout()
        self.optimizer = torch.optim.AdamW(self.model.parameters())

    def miner(self):
        return FitPeakMiner(self.signature, "c" * 64)

    def test_current_peak_can_move_outside_frozen_events_without_wet_features(self):
        before = module_sha256(self.model)
        result = self.miner().refresh(self.rows, self.model, self.optimizer, 0, block_frames=257)
        event = result.events[0]
        self.assertEqual((event.positive_index, event.negative_index), (1_500, 1_300))
        self.assertNotEqual(event.positive_index, 1_400)
        self.assertEqual(result.readout_sha256, before)
        self.assertEqual(module_sha256(self.model), before)
        self.assertTrue(self.model.training)
        self.assertTrue(self.model.gain.requires_grad)
        json.dumps(result.metadata(), allow_nan=False)
        self.assertNotIn("wet", result.metadata())
        self.assertEqual(len(result.events), 234)
        for row in self.rows:
            row["wet"][1_200] = 2.
        changed = self.miner().refresh(self.rows, self.model, self.optimizer, 0, block_frames=257)
        self.assertEqual(changed.events[0].positive_index, event.positive_index)
        self.assertEqual(changed.events[0].negative_index, event.negative_index)
        self.assertNotEqual(changed.metrics["squared_error_sum"], result.metrics["squared_error_sum"])
        self.assertEqual(changed.events[0].wet_positive_index, 1_200)

    def test_actual_nonlinear_fast_stream_matches_complete_prediction(self):
        model = DynamicCornerReadout(width=4)
        with torch.no_grad():
            model.output.normal_(std=.2)
            model.first_bias.normal_(std=.2)
            model.second_bias.normal_(std=.2)
        optimizer = torch.optim.AdamW(model.parameters())
        result = self.miner().refresh(self.rows, model, optimizer, 0, block_frames=71)
        row = self.rows[0]
        with torch.inference_mode():
            feature = CausalTapBank().cpu_features(row["dry"], row["original"])
            prediction = row["original"] + model(feature[None], row["controls"][None])[0]
        event = result.events[0]
        self.assertEqual(event.positive_index, int(prediction[1_024:].argmax()) + 1_024)
        self.assertEqual(event.negative_index, int(prediction[1_024:].argmin()) + 1_024)
        self.assertAlmostEqual(event.positive_value, float(prediction[1_024:].max()), delta=2e-6)
        body = prediction[1_024:].double()
        expected_error = float((body - row["wet"][1_024:].double()).square().sum())
        self.assertAlmostEqual(event.squared_error_sum, expected_error, delta=2e-6)

    def test_order_calibration_controls_and_nonfinite_inputs_are_rejected(self):
        swapped = self.rows.copy()
        swapped[0], swapped[1] = swapped[1], swapped[0]
        cal = self.rows.copy()
        cal[0] = {**cal[0], "path": self.signature["partitions"]["calibration"][0]["path"]}
        controls = self.rows.copy()
        controls[0] = {**controls[0], "controls": torch.tensor([1., 1.])}
        nonfinite = self.rows.copy()
        nonfinite[0] = {**nonfinite[0], "dry": torch.full_like(nonfinite[0]["dry"], torch.nan)}
        for rows in (swapped, cal, controls, nonfinite, self.rows[:-1]):
            with self.assertRaises(ValueError):
                self.miner().refresh(rows, self.model, self.optimizer, 0)
        wrong_signature = copy.deepcopy(self.signature)
        wrong_signature["partitions"]["fit"][0]["path"] = "dfz/train/0,0,5.wav"
        with self.assertRaises(ValueError):
            FitPeakMiner(wrong_signature, "c" * 64)

    def test_streamed_full_fit_metrics_match_direct_arrays_with_same_burnin(self):
        errors, energies, peaks = [], [], []
        for index, row in enumerate(self.rows):
            row["wet"][12] = 100.  # Excluded metric sample; must not leak in.
            row["wet"][1_300] = -.1 - index * .0001
            row["wet"][1_500] = .08 + index * .0002
            prediction = row["original"] + (row["dry"] * 21.4) * .25
            body, wet = prediction[1_024:].double(), row["wet"][1_024:].double()
            errors.append(float((body - wet).square().sum()))
            energies.append(float(wet.square().sum()))
            peaks.append(float((body.abs().max() - wet.abs().max()).abs()))
        result = self.miner().refresh(self.rows, self.model, self.optimizer, 0, block_frames=257)
        self.assertAlmostEqual(result.metrics["squared_error_sum"], sum(errors), places=8)
        self.assertAlmostEqual(result.metrics["wet_energy_sum"], sum(energies), places=8)
        self.assertAlmostEqual(result.metrics["global_esr"], sum(errors) / sum(energies), places=8)
        expected_p95 = float(torch.quantile(torch.tensor(peaks, dtype=torch.float64), .95))
        self.assertAlmostEqual(result.metrics["absolute_peak_error_p95"], expected_p95, places=8)
        self.assertEqual(result.events[0].wet_negative_index, 1_300)
        self.assertEqual(result.events[0].wet_positive_index, 1_500)
        self.assertTrue(all(e.scored_frames == 577 for e in result.events))

    def test_optimizer_completed_update_and_refresh_schedule_are_strict(self):
        miner = self.miner()
        with self.assertRaises(ValueError):
            miner.refresh(self.rows, self.model, self.optimizer, 250)
        miner.refresh(self.rows, self.model, self.optimizer, 0)
        for step in (0, 249, 500, 4_000):
            with self.assertRaises(ValueError):
                miner.refresh(self.rows, self.model, self.optimizer, step)
        with self.assertRaises(ValueError):
            miner.refresh(self.rows, self.model, self.optimizer, 250)
        mark_step(self.optimizer, self.model, 249)
        with self.assertRaises(ValueError):
            miner.refresh(self.rows, self.model, self.optimizer, 250)
        mark_step(self.optimizer, self.model, 250)
        miner.refresh(self.rows, self.model, self.optimizer, 250)

    def test_positive_negative_rotation_survives_refresh_and_audit_is_terminal(self):
        miner = self.miner()
        first = miner.refresh(self.rows, self.model, self.optimizer, 0)
        path = self.rows[0]["path"]
        self.assertEqual(miner.next_event(path), (1_500, "positive"))
        for step in REFRESH_STEPS[1:]:
            mark_step(self.optimizer, self.model, step)
            miner.refresh(self.rows, self.model, self.optimizer, step)
        self.assertEqual(miner.next_event(path), (1_300, "negative"))
        before = miner.active_readout_sha256
        mark_step(self.optimizer, self.model, 4_000)
        audit = miner.audit(self.rows, self.model, self.optimizer, 4_000)
        self.assertEqual(audit.purpose, "audit")
        self.assertEqual(miner.active_readout_sha256, before)
        self.assertEqual(first.readout_sha256, before)
        with self.assertRaises(ValueError):
            miner.next_event(path)
        with self.assertRaises(ValueError):
            miner.audit(self.rows, self.model, self.optimizer, 4_000)

    def test_ties_pick_earliest_scored_sample_and_burnin_still_has_real_state(self):
        for row in self.rows:
            row["dry"].zero_()
            row["original"].zero_()
        result = self.miner().refresh(self.rows, self.model, self.optimizer, 0, block_frames=511)
        self.assertTrue(all(e.positive_index == 1_024 and e.negative_index == 1_024 for e in result.events))
        self.rows[0]["dry"][1_023] = .2
        class LagOne(ProbeReadout):
            def at_training_knots(self, features, controls):
                return self.gain * features[..., 2]
        model = LagOne()
        optimizer = torch.optim.AdamW(model.parameters())
        result = self.miner().refresh(self.rows, model, optimizer, 0, block_frames=511)
        self.assertEqual(result.events[0].positive_index, 1_024)
        self.assertGreater(result.events[0].positive_value, 0.)

    def test_live_weight_mutation_during_mining_fails_closed(self):
        miner = self.miner()
        original = CausalTapBank.forward
        changed = False
        def mutate(bank, *args, **kwargs):
            nonlocal changed
            if not changed:
                with torch.no_grad():
                    self.model.gain.add_(.1)
                changed = True
            return original(bank, *args, **kwargs)
        with patch.object(CausalTapBank, "forward", mutate):
            with self.assertRaisesRegex(ValueError, "changed while"):
                miner.refresh(self.rows, self.model, self.optimizer, 0)
        self.assertIsNone(miner.active_readout_sha256)

    def test_logical_memory_and_work_counts_are_bounded(self):
        bound = mining_resource_bounds()
        self.assertEqual(bound["new_full_prediction_cache_bytes"], 0)
        self.assertEqual(bound["frames_per_refresh"], 33_696_000)
        self.assertEqual(bound["block_forward_calls_per_refresh"], 936)
        self.assertEqual(bound["maximum_readout_snapshot_tensor_bytes"], 58_816)
        self.assertEqual(bound["scalar_current_event_payload_bytes_without_python_or_paths"], 5_616)


if __name__ == "__main__":
    unittest.main()
