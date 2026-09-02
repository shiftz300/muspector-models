"""Local contracts for nonstationarity features and frozen relation decisions."""
import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import torch
import soundfile
from .order_temporal_features import temporal_order_features
from .order_cascade_features import cascade_order_features
from .calibrate_relation_order import matrices,evaluate
from .evaluate_order_transfer import measure,exact_improvement_five_points
from .order_bank_fit import control_variants,needs_fit,raw_proposal,fit_topology_options,apply_bank_fit
from .order_transfer_runtime import TransferRemixerRuntime,file_hash,assert_file_hashes


class TemporalContracts(unittest.TestCase):
    def test_packager_rejects_lowered_reported_baseline(self):
        from .package_order_bank import checked_evidence
        source=Path(__file__).parent/'runs/order-bankfit-bound-phase11'
        if not (source/'blend-development.json').exists():self.skipTest('bound development artifact not present')
        provenance=json.loads((source/'bank-calibrate-real.manifest.json').read_text())['signature']['source_sha256']
        if not all(Path(path).exists() for path in provenance):self.skipTest('original research artifacts were compacted after package admission')
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)
            for split in ('calibrate','valid'):
                for domain in ('real','reference','alternate','stress','pedalboard'):
                    for name in (f'bank-{split}-{domain}.json',f'bank-{split}-{domain}.manifest.json',f'blend-{split}-{domain}.json'):
                        (target/name).symlink_to(source/name)
            calibration=json.loads((source/'blend-calibration.json').read_text())
            development=json.loads((source/'blend-development.json').read_text())
            calibration['baseline']['real']['exact']=0.
            (target/'blend-calibration.json').write_text(json.dumps(calibration))
            development['calibration_sha256']=file_hash(target/'blend-calibration.json')
            (target/'blend-development.json').write_text(json.dumps(development))
            with self.assertRaisesRegex(ValueError,'reported baselines differ'):
                checked_evidence(target)

    def test_default_rejects_unpackaged_order_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'explicit candidate mode'):
                TransferRemixerRuntime(Path(folder))

    def test_package_file_loader_preserves_analysis_levels_and_files(self):
        runtime=TransferRemixerRuntime.__new__(TransferRemixerRuntime)
        runtime.packaged=True;runtime.sample_rate=44100;runtime.analysis_seconds=1
        with tempfile.TemporaryDirectory() as folder:
            dry=Path(folder)/'dry.wav';wet=Path(folder)/'wet.wav'
            original=np.array([.1,-.2,.15],dtype=np.float32)
            soundfile.write(dry,original,44100,subtype='FLOAT');soundfile.write(wet,original*.3,44100,subtype='FLOAT')
            hashes={str(path):file_hash(path) for path in (dry,wet)}
            x,y,metadata=runtime.load_pair(dry,wet)
            np.testing.assert_array_equal(x[:3],original);np.testing.assert_array_equal(y[:3],original*.3)
            self.assertEqual(metadata['analysis_level_policy'],'preserve-input-levels')
            runtime.packaged=False
            x,y,metadata=runtime.load_pair(dry,wet)
            self.assertEqual(float(abs(x).max()),1.)
            assert_file_hashes(hashes)

    def test_packaged_frontend_preserves_levels_legacy_keeps_its_normalization(self):
        runtime=TransferRemixerRuntime.__new__(TransferRemixerRuntime)
        dry=np.array([.1,-.2,.15],dtype=np.float32);wet=dry*.3
        runtime.packaged=True
        clean,affected=runtime._prepare_analysis_pair(dry,wet)
        np.testing.assert_array_equal(clean,dry);np.testing.assert_array_equal(affected,wet)
        self.assertFalse(np.shares_memory(clean,dry));self.assertFalse(np.shares_memory(affected,wet))
        runtime.packaged=False
        clean,affected=runtime._prepare_analysis_pair(dry,wet)
        self.assertEqual(float(abs(clean).max()),1.)
        np.testing.assert_array_equal(dry,np.array([.1,-.2,.15],dtype=np.float32))

    def test_package_batch_geometry_is_fixed_and_padding_never_uses_other_audio(self):
        runtime=TransferRemixerRuntime.__new__(TransferRemixerRuntime)
        runtime.packaged=True;runtime.target=torch.device('cpu')
        dry=np.array([.1,-.2,.15],dtype=np.float32);wet=dry*.3
        clean,affected=runtime._model_analysis_batch(dry,wet)
        self.assertEqual(tuple(clean.shape),(8,3))
        np.testing.assert_array_equal(clean[0].numpy(),dry);np.testing.assert_array_equal(affected[0].numpy(),wet)
        self.assertEqual(int(torch.count_nonzero(clean[1:])),0)
        self.assertEqual(int(torch.count_nonzero(affected[1:])),0)
        runtime.packaged=False
        self.assertEqual(tuple(runtime._model_analysis_batch(dry,wet)[0].shape),(1,3))

    def test_integrity_guard_rereads_actual_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'artifact';path.write_bytes(b'original')
            expected={str(path):file_hash(path)};assert_file_hashes(expected)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                assert_file_hashes(expected)

    def test_bank_bounds_and_rejected_proposal_ignore_truth(self):
        controls=np.array([0.,.2,.4,.6,.8,1.,.3,.5,.7])
        original=controls.copy()
        for _,values in control_variants(controls,('drive','delay','reverb')):
            self.assertTrue(((0<=values)&(values<=1)).all())
            self.assertLessEqual(float(np.max(abs(values-original))),.1000000001)
            np.testing.assert_array_equal(values[[1,2,3,4,6,7]],original[[1,2,3,4,6,7]])
        np.testing.assert_array_equal(controls,original)
        row={'active':['delay','drive'],'baseline':['delay','drive'],'baseline_error':.2,
             'logits':[0.,0.,0.],'probabilities':{'frozen':[.9,.5,.5]},
             'errors':[{'topology':['delay','drive'],'error':.18},{'topology':['drive','delay'],'error':.3}],
             'truth':[0,0,0],'mask':[0,0,0]}
        self.assertTrue(needs_fit(row,('frozen',1.,0.)))
        selected=raw_proposal(row,('frozen',1.,0.));row['truth']=[1,1,1];row['mask']=[1,1,1]
        self.assertEqual(selected,raw_proposal(row,('frozen',1.,0.)))

    def test_parallel_clean_fit_is_bounded_and_diagnostic_only(self):
        rng=np.random.default_rng(32);x=rng.normal(0,.02,4096).astype(np.float32);y=np.tanh(x*2).astype(np.float32)
        before=(x.copy(),y.copy());controls=np.full(9,.5)
        fit=fit_topology_options(x,y,('drive','delay'),controls,menus=('parallel-clean',))['parallel-clean']
        self.assertIn(fit['parallel_clean_fraction'],(0.,.1,.25));self.assertTrue(fit['scoring_only']);self.assertFalse(fit['gain_applied_to_output'])
        np.testing.assert_array_equal(fit['normalized_controls'],controls)
        self.assertAlmostEqual(sum(fit['renderer_weights']),1.)
        np.testing.assert_array_equal(x,before[0]);np.testing.assert_array_equal(y,before[1])

    def test_bank_fit_is_bound_to_the_fitted_topology(self):
        row={'active':['delay','drive'],'baseline':['delay','drive'],'baseline_error':.2,
             'logits':[0.,0.,0.],'probabilities':{'frozen':[.9,.5,.5]},
             'errors':[{'topology':['delay','drive'],'error':.18},{'topology':['drive','delay'],'error':.3}],
             'bank_fit':{'bounded-knobs':{'fit_topology':['delay','drive'],'error':.01}}}
        with self.assertRaisesRegex(ValueError,'different topology'):
            apply_bank_fit(row,('frozen',1.,0.),'bounded-knobs')
        row['bank_fit']['bounded-knobs']['fit_topology']=['drive','delay']
        self.assertEqual(apply_bank_fit(row,('frozen',1.,0.),'bounded-knobs')['errors'][1]['error'],.01)
        self.assertEqual(row['errors'][1]['error'],.3)

    def test_five_point_gate_uses_exact_counts_not_tolerance(self):
        baseline={'examples':320,'exact':238/320}
        self.assertTrue(exact_improvement_five_points({'examples':320,'exact':254/320},baseline))
        self.assertFalse(exact_improvement_five_points({'examples':320,'exact':253/320},baseline))
        baseline={'examples':122,'exact':63/122}
        self.assertTrue(exact_improvement_five_points({'examples':122,'exact':70/122},baseline))
        self.assertFalse(exact_improvement_five_points({'examples':122,'exact':69/122},baseline))

    def test_quiet_short_read_only_and_deterministic(self):
        x=np.zeros(4096,dtype=np.float32);y=x.copy()
        a=temporal_order_features(x,y)
        np.testing.assert_array_equal(a,temporal_order_features(x,y))
        self.assertEqual(a.shape,(2040,));self.assertTrue(np.isfinite(a).all())
        np.testing.assert_array_equal(x,np.zeros(4096,dtype=np.float32))
        np.testing.assert_array_equal(y,x)

    def test_dynamic_read_only(self):
        rng=np.random.default_rng(18);x=rng.normal(0,.1,16384).astype(np.float32)
        y=.7*x+.2*np.roll(x,320);before=(x.copy(),y.copy())
        value=temporal_order_features(x,y)
        self.assertTrue(np.isfinite(value).all())
        np.testing.assert_array_equal(x,before[0]);np.testing.assert_array_equal(y,before[1])

    def test_cascade_diagnostics_are_read_only_and_finite(self):
        rng=np.random.default_rng(19);x=rng.normal(0,.1,16384).astype(np.float32)
        y=np.tanh(3*x)+.2*np.roll(x,800);before=(x.copy(),y.copy())
        features=cascade_order_features(x,y)
        self.assertEqual(features.shape,(416,));self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(x,before[0]);np.testing.assert_array_equal(y,before[1])
        self.assertTrue(np.isfinite(cascade_order_features(np.zeros(4096),np.zeros(4096))).all())

    def test_cascade_features_do_not_invent_lti_order_information(self):
        from scipy.signal import lfilter
        rng=np.random.default_rng(20);x=rng.normal(0,.15,16384)
        echo=np.r_[1.,np.zeros(299),.35]
        room=np.exp(-np.arange(180)/35)*rng.normal(size=180)*.03;room[0]=1.
        first=lfilter(room,[1.],lfilter(echo,[1.],x))
        second=lfilter(echo,[1.],lfilter(room,[1.],x))
        np.testing.assert_allclose(first,second,rtol=0,atol=1e-14)
        np.testing.assert_allclose(cascade_order_features(x,first),cascade_order_features(x,second),rtol=0,atol=1e-5)
        pre=lfilter(echo,[1.],np.tanh(x*4))
        post=np.tanh(lfilter(echo,[1.],x)*4)
        self.assertGreater(np.linalg.norm(cascade_order_features(x,pre)-cascade_order_features(x,post)),.1)

    def test_relation_evaluator_matches_existing_decision(self):
        from .spec import order_targets
        import itertools
        rng=np.random.default_rng(55);rows=[]
        for active in (('delay','drive'),('delay','reverb'),('drive','reverb'),('delay','drive','reverb')):
            tops=list(itertools.permutations(active))
            for _ in range(30):
                truth,mask=order_targets(tops[int(rng.integers(len(tops)))])
                errors=[{'topology':top,'error':float(rng.uniform(.1,1))} for top in tops]
                rows.append({'truth':truth,'mask':mask,'active':active,'baseline':tops[0],
                             'baseline_error':errors[0]['error'],'errors':errors,
                             'logits':rng.normal(size=3).tolist(),
                             'probabilities':{'a':rng.uniform(.01,.99,3).tolist()}})
        expected=measure(rows,'a',1.,0.)
        actual=evaluate({'x':matrices(rows)},('a','a','a'))['x']
        self.assertEqual(expected,actual)


if __name__=='__main__':unittest.main()
