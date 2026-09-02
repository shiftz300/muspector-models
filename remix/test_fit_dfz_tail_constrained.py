import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy import sparse
import torch

from .fit_dfz_tail_constrained import FrozenFeatures, solve_problem, solve_tail, tail_mean, tail_problem, whiten_problem
from .stable_nonlinear_readout import NonlinearReadout


class TailConstraintTests(unittest.TestCase):
    def test_status_and_original_coordinate_residuals_are_all_required(self):
        for status, x, y in ((7, 0., 0.), (1, 1., 0.), (1, 0., 1.), (1, float('nan'), 0.)):
            with self.subTest(status=status, x=x, y=y):
                info = SimpleNamespace(status_val=status, status='mock', iter=1, prim_res=0., dual_res=0., run_time=0.)
                result = SimpleNamespace(info=info, x=np.array([x]), y=np.array([y]))
                with patch('remix.fit_dfz_tail_constrained.osqp.OSQP') as solver:
                    solver.return_value.solve.return_value = result
                    original, _ = solve_problem(sparse.eye(1, format='csc'), np.zeros(1),
                                                sparse.eye(1, format='csc'), np.zeros(1), np.zeros(1), 1, width=1)
                self.assertIsNone(original)

    def test_phase_one_witness_is_transformed_before_warm_start(self):
        info = SimpleNamespace(status_val=1, status='solved', iter=1, prim_res=0., dual_res=0., run_time=0.)
        result = SimpleNamespace(info=info, x=np.array([1.]), y=np.array([0.]))
        with patch('remix.fit_dfz_tail_constrained.osqp.OSQP') as solver:
            solver.return_value.solve.return_value = result
            original, audit = solve_problem(sparse.csc_matrix([[4.]]), np.array([-2.]),
                                            sparse.csc_matrix([[1.]]), np.array([0.]), np.array([1.]),
                                            1, width=1, original_warm_start=np.array([3.]))
        # P=4 gives transform=1/2, so original x=3 maps to solver x=6.
        np.testing.assert_allclose(solver.return_value.warm_start.call_args.kwargs['x'], [6.])
        np.testing.assert_allclose(original, [.5])
        self.assertTrue(audit['phase_one_primal_warm_start'])

    def test_tail_bound_dominates_original_percentile(self):
        rng = np.random.default_rng(1017)
        for n in (20, 54, 60, 180, 234):
            for _ in range(10):
                values = rng.random(n)
                self.assertGreaterEqual(tail_mean(values), np.quantile(values, .95))

    def test_sparse_qp_and_actual_peak_bounds(self):
        equations = [(np.eye(1), np.array([.05]))]
        rows = [{'corner': 0, 'blend': 0, 'peak': .25, 'sign': 1.,
                 'anchor_x': np.ones(1), 'anchor_p': .2,
                 'cuts': {0: (np.ones(1), .2), 1: (-np.ones(1), -.2)}} for _ in range(20)]
        weight, bound, _, audit = solve_tail(equations, rows, width=1)
        self.assertIsNotNone(weight)
        self.assertLessEqual(abs(.2+weight[0, 0]-.25), .02)
        self.assertLessEqual(tail_mean(bound), .02)
        self.assertLess(audit['independent_primal_residual'], 2e-7)

    def test_all_files_participate_in_tail_budget(self):
        equations = [(np.eye(1), np.zeros(1))]
        rows = [{'corner': 0, 'blend': 0, 'peak': .25 if i < 10 else .75, 'sign': 1.,
                 'anchor_x': np.ones(1), 'anchor_p': 0., 'cuts': {0: (np.ones(1), 0.)}} for i in range(20)]
        weight, _, _, _ = solve_tail(equations, rows, width=1)
        self.assertIsNone(weight)

    def test_whitening_preserves_objective_and_constraints(self):
        equations = [(np.array([[1., .999], [.999, 1.]]), np.array([.02, .01]))]
        rows = [{'corner': 0, 'blend': 0, 'peak': .25, 'sign': 1., 'anchor_x': np.array([1., .5]),
                 'anchor_p': .2, 'cuts': {0: (np.array([1., .5]), .2)}} for _ in range(20)]
        p, q, a, _, _, _, _, _ = tail_problem(equations, rows, width=2)
        r, pw, qw, aw = whiten_problem(p, q, a, 1, 2)
        v = np.random.default_rng(1017).normal(size=len(q)); original = r@v
        self.assertAlmostEqual(float(.5*v@pw@v+qw@v), float(.5*original@p@original+q@original), places=10)
        np.testing.assert_allclose(aw@v, a@original, atol=1e-12)
        np.testing.assert_allclose(pw[:2, :2].toarray(), np.eye(2), atol=1e-12)

    def test_batch_features_keep_first_scored_and_final_sample(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('coeff', torch.linspace(.01, .64, 64))
                self.output_layer = torch.nn.Linear(64, 1, bias=False)
            def encode(self, dry, controls, state=None):
                return dry[..., None]*self.coeff*(1+controls[:, None, :1]), None
        torch.manual_seed(1017)
        model = SimpleNamespace(base=Encoder(), readout=NonlinearReadout())
        rng = np.random.default_rng(1017)
        pairs = [(rng.normal(size=144000).astype(np.float32)*.03, np.zeros(144000, np.float32), np.array(c, np.float32))
                 for c in ((0., 0.), (.5, 1.))]
        with patch('remix.fit_dfz_tail_constrained.read_effect_pair', side_effect=pairs):
            actual = FrozenFeatures(model, torch.device('cpu')).batch([{'path': 'a'}, {'path': 'b'}])
        with torch.inference_mode():
            for (x, p, _, corner), (dry, _, control) in zip(actual, pairs):
                self.assertEqual(x.shape, (142976, 80))
                for score_index in (0, 3071, 3072, 142975):
                    h, _ = model.base.encode(torch.tensor([[dry[score_index+1024]]]), torch.from_numpy(control)[None])
                    expected = torch.cat((h[0]*8, torch.tanh(h[0]@model.readout.input[corner]*8)), -1)[0]
                    np.testing.assert_allclose(x[score_index], expected.numpy(), atol=1e-7)
                    value = model.base.output_layer(h).squeeze(-1)+model.readout(h, torch.from_numpy(control)[None])
                    self.assertAlmostEqual(float(p[score_index]), float(value[0, 0]), places=7)


if __name__ == '__main__':
    unittest.main()
