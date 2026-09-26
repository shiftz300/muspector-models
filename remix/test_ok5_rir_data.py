import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from .ok5_rir_data import EXPECTED_LICENSE, load_ok5_rir


class OK5RIRDataTest(unittest.TestCase):
    def test_loader_uses_one_physical_channel_and_normalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "room.sofa"
            with h5py.File(path, "w") as handle:
                handle.attrs["Conventions"] = "SOFA"
                handle.attrs["SOFAConventions"] = "SingleRoomSRIR"
                handle.attrs["License"] = EXPECTED_LICENSE
                handle.attrs["Title"] = "room"
                handle.create_dataset("Data.SamplingRate", data=np.asarray([48_000.0]))
                values = np.zeros((1, 6, 48_100), dtype=np.float64)
                values[0, 0, 50] = 1.0
                values[0, 1, 50] = 100.0
                handle.create_dataset("Data.IR", data=values)
            impulse = load_ok5_rir(path, 0)
            self.assertEqual(impulse.shape, (48_000,))
            self.assertAlmostEqual(float(np.sum(np.square(impulse))), 1.0)
            self.assertEqual(float(impulse[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
