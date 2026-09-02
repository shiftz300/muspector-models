import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from remix.net import SpectralNet
from remix.runtime import CleanRuntime


class RuntimeTests(unittest.TestCase):
    def checkpoint(self, root: Path) -> Path:
        torch.manual_seed(1)
        model = SpectralNet(channels=4).eval()
        torch.nn.init.normal_(model.head.weight, std=0.01)
        torch.nn.init.normal_(model.head.bias, std=0.01)
        path = root / "clean.pt"
        torch.save({
            "schema": 1,
            "architecture": "complex-stft",
            "sample_rate": 48_000,
            "channels": 4,
            "n_fft": model.n_fft,
            "hop": model.hop,
            "state_dict": model.state_dict(),
        }, path)
        return path

    def test_bounded_chunks_match_shared_scale_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = CleanRuntime(self.checkpoint(Path(temporary)), core=8_192, halo=4_096, threads=1)
            source = (np.random.default_rng(2).standard_normal(44_417) * 0.05).astype(np.float32)
            before = source.copy()
            scale = runtime.scale(source)
            with torch.inference_mode():
                expected_scale = torch.stft(
                    torch.from_numpy(source)[None],
                    runtime.model.n_fft,
                    runtime.model.hop,
                    window=runtime.model.window,
                    return_complex=True,
                ).abs().mean((1, 2), keepdim=True)
                reference = runtime.model.scaled(torch.from_numpy(source)[None], scale)[0].numpy()
            restored = runtime.render(source)
            self.assertEqual(float(scale), float(expected_scale))
            self.assertLess(float(np.max(np.abs(restored - reference))), 2e-5)
            self.assertEqual(restored.shape, source.shape)
            np.testing.assert_array_equal(source, before)

    def test_exact_bypass_and_invalid_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self.checkpoint(Path(temporary))
            runtime = CleanRuntime(checkpoint, core=8_192, halo=4_096, threads=1)
            source = (np.random.default_rng(3).standard_normal(20_000) * 0.05).astype(np.float32)
            np.testing.assert_array_equal(runtime.render(source, strength=0), source)
            with self.assertRaises(ValueError):
                CleanRuntime(checkpoint, threads=3)

    def test_checkpoint_declares_custom_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dilations = ((1, 1), (2, 4), (4, 16), (8, 64))
            model = SpectralNet(channels=4, dilations=dilations).eval()
            path = root / "reverb.pt"
            torch.save({
                "schema": 1,
                "architecture": "complex-stft",
                "sample_rate": 48_000,
                "channels": 4,
                "n_fft": model.n_fft,
                "hop": model.hop,
                "dilations": [list(value) for value in dilations],
                "state_dict": model.state_dict(),
            }, path)
            self.assertEqual(model.minimum_halo, 11_136)
            with self.assertRaises(ValueError):
                CleanRuntime(path, core=8_192, halo=4_096, threads=1)
            runtime = CleanRuntime(path, core=8_192, halo=11_264, threads=1)
            self.assertEqual(runtime.model.dilations, dilations)


if __name__ == "__main__":
    unittest.main()
