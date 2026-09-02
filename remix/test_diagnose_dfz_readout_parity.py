import unittest

import numpy as np
import torch

from .diagnose_dfz_readout_parity import columns, decision, feature_bank, partition, sampling, solve


class ReadoutParityTests(unittest.TestCase):
    def test_only_original_fit_partition(self):
        rows = [{'path': f'corpus/dfz/train/{a},{b},{i}.wav'}
                for a in (0, 50, 100) for b in (0, 50, 100)
                for i in range(1, 33) if i % 5]
        train, held = partition(rows)
        self.assertEqual((len(train), len(held)), (180, 54))
        self.assertFalse({r['path'] for r in train} & {r['path'] for r in held})
        for replacement in ('corpus/dfz/eval/0,0,1.wav', 'corpus/dfz/train/0,0,5.wav'):
            with self.assertRaises(ValueError):
                partition([{'path': replacement}, *rows[1:]])
        with self.assertRaises(ValueError):
            partition([rows[1], *rows[1:]])

    def test_zero_and_matched_parity_basis(self):
        torch.manual_seed(1017)
        h, projection = torch.randn(30, 64)*.03, torch.randn(64, 16)*.1
        plus, minus = feature_bank(h, projection), feature_bank(-h, projection)
        self.assertTrue(torch.equal(plus[:, :160], -minus[:, :160]))
        self.assertTrue(torch.equal(plus[:, 160:], minus[:, 160:]))
        self.assertEqual(float(feature_bank(torch.zeros_like(h), projection).abs().max()), 0.)
        self.assertEqual(len(columns('odd_even')), len(columns('odd_cubic')))

    def test_sampling_same_mass_and_no_warmup(self):
        wet = np.linspace(-.2, .3, 4096)
        ids, weight = sampling(wet, -wet)
        self.assertGreaterEqual(ids.min(), 1024)
        self.assertLess(ids.max(), len(wet))
        self.assertAlmostEqual(float(weight[:2048].sum()), .5)
        self.assertAlmostEqual(float(weight[2048:].sum()), .5)

    def test_solve_even_signal_and_keep_controls_separate(self):
        rng = np.random.default_rng(7)
        odd = rng.normal(size=(4000, 80))*.1
        x = np.c_[odd, odd**3, odd**2]
        y = .2*x[:, 165]-.1*x[:, 162]
        xx, xy = x.T @ x, x.T @ y
        w = solve(xx, xy, 'odd_even')
        self.assertLess(np.mean((x@w-y)**2), np.mean(y**2)*.001)
        self.assertEqual(float(abs(w[80:160]).max()), 0.)
        self.assertEqual(w.dtype, np.float32)

    def test_investment_gate_rejects_peak_only_or_cubic_tie(self):
        def rows(error, energy=1.):
            return [{'peak_error': error, 'error_energy': energy, 'wet_energy': 10., 'blend': c}
                    for c in (0, 50, 100) for _ in range(6)]
        arms = {'unchanged': rows(.03), 'odd': rows(.03), 'odd_cubic': rows(.03), 'odd_even': rows(.02)}
        self.assertTrue(decision(arms)[1]['continue_to_full_fit'])
        arms['odd_even'] = rows(.02, 1.1)
        self.assertFalse(decision(arms)[1]['continue_to_full_fit'])
        arms['odd_even'] = rows(.03)
        self.assertFalse(decision(arms)[1]['continue_to_full_fit'])


if __name__ == '__main__':
    unittest.main()
