"""Small offline contracts for the full-chain development handoff."""
import unittest

from .evaluate_forward_chain import _audit_status, _original_order_chain
from .order_search import decode_controls
from .spec import ChainSpec


class FullChainAdmissionTests(unittest.TestCase):
    def test_transfer_requires_its_own_acceptance(self):
        common = dict(numerical_passed=True, immutable=True, uses_transfer=True, candidate_mode=False)
        self.assertEqual(_audit_status(**common, transfer_passed=True), 'passed')
        for state in (False, None):
            self.assertEqual(_audit_status(**common, transfer_passed=state), 'failed')

    def test_candidate_or_mutation_cannot_pass(self):
        common = dict(numerical_passed=True, immutable=True, uses_transfer=True,
                      transfer_passed=True, candidate_mode=False)
        for field, value in (('numerical_passed', False), ('immutable', False), ('candidate_mode', True)):
            self.assertEqual(_audit_status(**{**common, field: value}), 'failed')

    def test_original_order_comparison_preserves_all_controls(self):
        controls = [.3, .4, .5, .6, .7, .2, .5, .8, .4]
        selected = ChainSpec(decode_controls(('drive', 'delay', 'reverb'), controls))
        report = {'chain': selected.document(), 'normalized_controls': controls,
                  'transfer': {'original_selected': ['reverb', 'drive', 'delay']}}
        before = _original_order_chain(report)
        self.assertEqual(before.topology, ('reverb', 'drive', 'delay'))
        self.assertEqual({v.kind: v for v in before.effects}, {v.kind: v for v in selected.effects})
        self.assertEqual(report['normalized_controls'], controls)


if __name__ == '__main__':
    unittest.main()
