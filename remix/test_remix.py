from __future__ import annotations

import os
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile
import torch

from remix.asrnn_data import audit_asrnn, read_rat_pair
from remix.asrnn_effects import audit_effect, read_effect_pair
from remix.audit_asrnn_phase7_groups import coverage_calibration_files, near_performance_pairs
from remix.capture_manifest import sha256 as capture_sha256
from remix.capture_manifest import validate_manifest
from remix.capture_pack import audit_drive_plan, validate_source_registry
from remix.capture_kit import (
    SAMPLE_RATE as CAPTURE_SAMPLE_RATE,
    build_capture_program,
    drive_capture_plan,
    drive_pilot_session,
    plan_coverage,
    prepare_capture_record,
    smoke_report as capture_smoke_report,
    write_capture_program,
)
from remix.data import _split
from remix.differentiable import render as differentiable_render
from remix.delay_model import FEATURES as DELAY_FEATURES
from remix.delay_model import delay_features
from remix.drive_model import IMAGE_SIZE, STATISTICS, drive_features
from remix.drive_adapter import DriveDeviceAdapter
from remix.rat_direct import RatDirectRenderer
from remix.stable_rat import StableRatRenderer
from remix.stable_effect import StableEffectRenderer, StableEffectEnsemble, ConditionedReadoutEffect, StableEffectFIRResidual
from remix.paired_transfer_features import paired_transfer_features
from remix.train_effect_inverse_features import metrics as semantic_inverse_metrics
from remix.evaluate_order_transfer import decision as transfer_order_decision
from remix.order_interaction_features import order_interaction_features
from remix.order_search import convex_renderer_fit
from remix.stable_tcn_residual import ZeroGatedTCN, StableTCNResidual
from remix.stable_effect_adapter import StableEffectResidualAdapter
from remix.centered_stable_effect import CenteredStableEffectRenderer
from remix.train_asrnn_phase7 import _group_cvar, _partition as phase7_partition
from remix.evaluate_phase7_guitarset_ood import _render as phase7_ood_render
from remix.summarize_asrnn_inverse import control_decisions
from remix.summarize_asrnn_phase7 import summarize as summarize_phase7
from remix.screen_asrnn_capacity import selection_key as capacity_selection_key
from remix.benchmark_stable_effect import benchmark as benchmark_stable_effect
from remix.promote_asrnn_capacity import validate_evidence as validate_capacity_evidence
from remix.export_stable_effect import export_graph as export_stable_graph
from remix.onnx_stable_effect import OnnxStableEffect
from remix.forward_delay import (
    DelayForwardRenderer,
    controls_to_delay,
    normalized_delay_controls,
    stream_delay_block,
)
from remix.forward_drive import DriveForwardRenderer, forward_loss
from remix.forward_chain import ForwardChainRuntime
from remix.forward_reverb import ReverbDeviceProfile, stream_reverb_block
from remix.inference import load_analysis_pair, parse_effects
from remix.physics import deconvolution_delay_controls, deconvolution_delay_hint
from remix.order_identifiability import perceptual_order_targets
from remix.order_search import decode_controls, gain_aligned_error, rank_topologies
from remix.pedalboard_renderer import (
    MAX_REVERB_DECAY_SECONDS,
    MIN_REVERB_DECAY_SECONDS,
    render_pedalboard_chain,
    reverb_room_size,
)
from remix.quality import (
    analysis_pair,
    checked_audio,
    checked_sample_rate,
    contract_manifest,
)
from remix.render import render_chain
from remix.reverb_model import FEATURES, reverb_features
from remix.spec import (
    ChainSpec,
    Delay,
    Drive,
    Reverb,
    control_targets,
    identifiable_order_targets,
    order_targets,
    ranked_topologies,
)
from remix.train_capture_adapter import capture_examples


class RemixContractTests(unittest.TestCase):
    def test_charge_bank_is_causal_bounded_zero_origin_and_exportable(self) -> None:
        from remix.stable_charge_residual import ChargeBank,ChargeReadout
        from remix.stable_effect import stable_effect_from_payload
        from remix.train_dfz_transients import project
        torch.manual_seed(948);bank=ChargeBank();dry=torch.randn(1,263)*.02
        with torch.no_grad():
            features,state=bank(dry);first,initial=bank(dry[:,:17]);second,_=bank(dry[:,17:],initial)
            torch.testing.assert_close(features,torch.cat((first,second),1),atol=2e-6,rtol=0)
            self.assertLessEqual(float(features.abs().max()),1.)
            self.assertEqual(float(bank(torch.zeros_like(dry))[0].abs().max()),0.)
            alpha=bank.cell.weight_hh_l0.diag()
            self.assertTrue(torch.all((alpha>0)&(alpha<1)).item())
            torch.testing.assert_close(bank.cell.weight_hh_l0,torch.diag(alpha),atol=0,rtol=0)
            torch.testing.assert_close(alpha+bank.cell.weight_ih_l0[:,0],torch.ones_like(alpha),atol=0,rtol=0)
        base=StableEffectRenderer(2,4,2);project(base,mirror_gates=True)
        source={'schema':2,'sample_rate':48000,'architecture':'stable-conditioned-lstm','device':'dfz','control_count':2,'hidden_size':4,'layers':2,'input_coef':base.input_coef,'inverted_controls':[],'state_dict':base.state_dict()}
        readout=ChargeReadout();torch.nn.init.normal_(readout.weight,std=.01)
        payload={'schema':10,'sample_rate':48000,'architecture':'stable-dfz-charge-residual','device':'dfz','base':source,'charge_milliseconds':bank.milliseconds,'readout_state_dict':readout.state_dict()}
        model=stable_effect_from_payload(payload);controls=torch.rand(1,263,2)
        with torch.no_grad():
            whole,_=model(dry,controls);first,state=model(dry[:,:17],controls[:,:17]);second,_=model(dry[:,17:],controls[:,17:],state)
            torch.testing.assert_close(whole,torch.cat((first,second),1),atol=2e-6,rtol=0)
            self.assertEqual(float(model(torch.zeros_like(dry),controls)[0].abs().max()),0.)
        with tempfile.TemporaryDirectory(prefix='muspector-charge-onnx-') as directory:
            checkpoint=Path(directory)/'model.pt';graph=Path(directory)/'model.onnx';torch.save(payload,checkpoint)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');export_stable_graph(checkpoint,graph)
            runtime=OnnxStableEffect(checkpoint,graph);actual,_=runtime(dry,controls)
            torch.testing.assert_close(whole,actual,atol=2e-6,rtol=0)
            long_dry=torch.randn(1,144000)*.02;static=torch.full((1,2),.5)
            with torch.no_grad():expected,_=model(long_dry,static)
            actual,_=runtime(long_dry,static)
            torch.testing.assert_close(expected,actual,atol=2e-6,rtol=0)

    def test_composite_ensemble_preserves_dynamic_zero_state_and_onnx(self) -> None:
        from remix.stable_effect import stable_effect_from_payload
        from remix.stable_nonlinear_readout import NonlinearReadout
        from remix.train_dfz_transients import project
        torch.manual_seed(941)
        base=StableEffectRenderer(2,4,2);project(base,mirror_gates=True)
        source={'schema':2,'sample_rate':48000,'architecture':'stable-conditioned-lstm','device':'dfz','control_count':2,'hidden_size':4,'layers':2,'input_coef':base.input_coef,'inverted_controls':[],'state_dict':base.state_dict()}
        readout=NonlinearReadout(4,3);torch.nn.init.normal_(readout.output,std=.01)
        nonlinear={'schema':7,'sample_rate':48000,'architecture':'stable-nonlinear-readout','device':'dfz','base':source,'readout_width':3,'readout_state_dict':readout.state_dict()}
        payload={'schema':9,'sample_rate':48000,'architecture':'stable-dfz-composite-ensemble','device':'dfz','members':[source,nonlinear],'weights':[.4,.6]}
        model=stable_effect_from_payload(payload);dry=torch.randn(1,263)*.02;controls=torch.rand(1,263,2)
        with torch.no_grad():
            full,_=model(dry,controls);first,state=model(dry[:,:17],controls[:,:17]);second,_=model(dry[:,17:],controls[:,17:],state)
            self.assertEqual(model.state_floats_per_mono_stream,32)
            torch.testing.assert_close(full,torch.cat((first,second),1),atol=2e-6,rtol=0)
            zero,_=model(torch.zeros_like(dry),controls);self.assertEqual(float(zero.abs().max()),0.)
            expected=sum(weight*member(dry,controls)[0] for weight,member in zip(payload['weights'],model.members))
            torch.testing.assert_close(full,expected,atol=1e-7,rtol=0)
        for weights in ([.2,.7],[-.1,1.1],[float('nan'),.6]):
            with self.assertRaises(ValueError):stable_effect_from_payload(dict(payload,weights=weights))
        with tempfile.TemporaryDirectory(prefix='muspector-composite-onnx-') as directory:
            checkpoint=Path(directory)/'model.pt';graph=Path(directory)/'model.onnx';torch.save(payload,checkpoint)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');export_stable_graph(checkpoint,graph)
            runtime=OnnxStableEffect(checkpoint,graph);actual,_=runtime(dry,controls)
            torch.testing.assert_close(full,actual,atol=2e-6,rtol=0)

    def test_skip_readout_matches_frozen_core_and_exports_dynamic_state(self) -> None:
        from remix.stable_skip_readout import StableSkipReadout,encode_skip
        from remix.stable_nonlinear_readout import NonlinearReadout
        from remix.train_dfz_transients import project
        torch.manual_seed(934);base=StableEffectRenderer(2,4,2);project(base,mirror_gates=True)
        readout=NonlinearReadout(9,3);model=StableSkipReadout(base,readout).eval()
        dry=torch.randn(1,263)*.02;controls=torch.rand(1,263,2)
        with torch.no_grad():
            expected,_=base(dry,controls);actual,_=model(dry,controls)
            torch.testing.assert_close(expected,actual,atol=0,rtol=0)
            torch.nn.init.normal_(readout.output,std=.01)
            full,_=model(dry,controls);first,state=model(dry[:,:17],controls[:,:17]);second,_=model(dry[:,17:],controls[:,17:],state)
            torch.testing.assert_close(full,torch.cat((first,second),1),atol=2e-6,rtol=0)
            zero,_=model(torch.zeros_like(dry),controls);self.assertEqual(float(zero.abs().max()),0.)
        with tempfile.TemporaryDirectory(prefix='muspector-skip-onnx-') as directory:
            checkpoint=Path(directory)/'model.pt';graph=Path(directory)/'model.onnx'
            payload={'schema':2,'sample_rate':48000,'architecture':'stable-conditioned-lstm','device':'dfz','control_count':2,'hidden_size':4,'layers':2,'input_coef':base.input_coef,'inverted_controls':[],'state_dict':base.state_dict()}
            torch.save({'schema':8,'sample_rate':48000,'architecture':'stable-skip-nonlinear-readout','device':'dfz','base':payload,'readout_width':3,'readout_state_dict':readout.state_dict()},checkpoint)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');export_stable_graph(checkpoint,graph)
            runtime=OnnxStableEffect(checkpoint,graph);actual,_=runtime(dry,controls)
            torch.testing.assert_close(full,actual,atol=2e-6,rtol=0)

    def test_semantic_runtime_requires_replay_provenance_and_rejects_quiet_wet(self) -> None:
        from unittest.mock import patch
        from remix.semantic_effect_runtime import SemanticEffectRuntime
        from remix.effect_admission import validate_inverse_teacher,forward_teacher_evidence
        import hashlib,json
        with tempfile.TemporaryDirectory(prefix='muspector-semantic-admission-') as directory:
            root=Path(directory);checkpoint=root/'fake.pt';checkpoint.write_bytes(b'local-test-checkpoint')
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();evidence=root/'metrics.json';evidence.write_text('{}')
            acceptance=root/'acceptance.json';report={'accepted':True,'device':'cs3','renderer_sha256':digest,'inverse_evidence_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()}
            acceptance.write_text(json.dumps(report));cardpath=root/'model-card.json'
            card={'accepted':True,'runtime_replay_verified':True,'runtime_replay_compute':'cpu','acceptance_sha256':hashlib.sha256(acceptance.read_bytes()).hexdigest(),'renderer':{'sha256':digest}}
            cardpath.write_text(json.dumps(card))
            with patch('remix.semantic_effect_runtime.load_stable_effect',return_value=(StableEffectRenderer(1,4,1),{'device':'cs3'})):
                runtime=SemanticEffectRuntime(acceptance,checkpoint)
                with self.assertRaisesRegex(ValueError,'Wet too quiet'):runtime.infer(np.full(144000,.02,dtype=np.float32),np.zeros(144000,dtype=np.float32))
                with self.assertRaisesRegex(ValueError,'backend'):SemanticEffectRuntime(acceptance,checkpoint,'mps')
                card['runtime_replay_verified']=False;cardpath.write_text(json.dumps(card))
                with self.assertRaisesRegex(ValueError,'replay is not accepted'):SemanticEffectRuntime(acceptance,checkpoint)
                card['runtime_replay_verified']=True;card['acceptance_sha256']='stale';cardpath.write_text(json.dumps(card))
                with self.assertRaisesRegex(ValueError,'provenance changed'):SemanticEffectRuntime(acceptance,checkpoint)
            self.assertEqual(validate_inverse_teacher({'device':'cs3','checkpoint_sha256':digest},checkpoint,{'device':'cs3'}),digest)
            with self.assertRaisesRegex(ValueError,'teacher changed'):validate_inverse_teacher({'device':'cs3','checkpoint_sha256':'wrong'},checkpoint,{'device':'cs3'})
            with self.assertRaisesRegex(ValueError,'device mismatch'):validate_inverse_teacher({'device':'cs3','checkpoint_sha256':digest},checkpoint,{'device':'dfz'})
            self.assertFalse(forward_teacher_evidence(checkpoint)['passed'])

    def test_full_prefix_training_preserves_zero_input_gate_constraints(self) -> None:
        from remix.train_dfz_transients import project,combine_mirrored_gate_gradients
        base=StableEffectRenderer(2,hidden_size=4,layers=2)
        norms=project(base,mirror_gates=True)
        self.assertLessEqual(max(norms),.995001)
        for layer in base.rnn_layers:
            torch.testing.assert_close(layer.weight_hh_l0[:4],-layer.weight_hh_l0[4:8],atol=0,rtol=0)
            torch.testing.assert_close(layer.weight_ih_l0[:4,-2:],-layer.weight_ih_l0[4:8,-2:],atol=0,rtol=0)
            for parameter in layer.parameters():parameter.grad=torch.ones_like(parameter)
            layer.weight_hh_l0.grad[:4].fill_(2)
        combine_mirrored_gate_gradients(base)
        for layer in base.rnn_layers:
            torch.testing.assert_close(layer.weight_hh_l0.grad[4:8],-torch.ones_like(layer.weight_hh_l0.grad[4:8]),atol=0,rtol=0)
            self.assertEqual(float(layer.weight_hh_l0.grad[:4].abs().max()),0.)
            self.assertEqual(float(layer.weight_ih_l0.grad[:4,:-2].min()),1.)
        with torch.no_grad():
            output,_=base(torch.zeros(2,257),torch.rand(2,257,2))
            self.assertEqual(float(output.abs().max()),0.)

    def test_nonlinear_readout_knots_zero_stream_and_onnx(self) -> None:
        from remix.stable_nonlinear_readout import NonlinearReadout,StableNonlinearReadout
        torch.manual_seed(929)
        base=StableEffectRenderer(2,hidden_size=4,layers=1).eval()
        with torch.no_grad():
            layer=base.rnn_layers[0];layer.bias_ih_l0.zero_();layer.bias_hh_l0.zero_();layer.weight_ih_l0[8:12,-2:].zero_()
        readout=NonlinearReadout(4,3);torch.nn.init.normal_(readout.output,std=.01)
        model=StableNonlinearReadout(base,readout).eval()
        dry=torch.randn(1,263)*.02;controls=torch.rand(1,263,2)
        with torch.no_grad():
            hidden=torch.randn(9,17,4)*.01;knots=torch.cartesian_prod(torch.tensor([0.,.5,1.]),torch.tensor([0.,.5,1.]))
            torch.testing.assert_close(readout(hidden,knots),readout.at_training_knots(hidden,knots),atol=1e-7,rtol=0)
            full,_=model(dry,controls);one,state=model(dry[:,:17],controls[:,:17]);two,_=model(dry[:,17:],controls[:,17:],state)
            torch.testing.assert_close(full,torch.cat((one,two),1),atol=2e-6,rtol=0)
            zero,_=model(torch.zeros_like(dry),controls);self.assertEqual(float(zero.abs().max()),0.)
        with tempfile.TemporaryDirectory(prefix='muspector-nonlinear-onnx-') as directory:
            checkpoint=Path(directory)/'model.pt';graph=Path(directory)/'model.onnx'
            payload={'schema':2,'sample_rate':48000,'architecture':'stable-conditioned-lstm','device':'dfz','control_count':2,'hidden_size':4,'layers':1,'input_coef':base.input_coef,'inverted_controls':[],'state_dict':base.state_dict()}
            torch.save({'schema':7,'sample_rate':48000,'architecture':'stable-nonlinear-readout','device':'dfz','base':payload,'readout_width':3,'readout_state_dict':readout.state_dict()},checkpoint)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');export_stable_graph(checkpoint,graph)
            runtime=OnnxStableEffect(checkpoint,graph);actual,_=runtime(dry,controls)
            torch.testing.assert_close(full,actual,atol=2e-6,rtol=0)

    def test_zero_gated_tcn_dynamic_state_and_onnx(self) -> None:
        torch.manual_seed(925)
        base=StableEffectRenderer(2,hidden_size=4,layers=1).eval()
        with torch.no_grad():
            layer=base.rnn_layers[0];layer.bias_ih_l0.zero_();layer.bias_hh_l0.zero_();layer.weight_ih_l0[8:12,-2:].zero_()
        tcn=ZeroGatedTCN(width=4,blocks=3)
        torch.nn.init.normal_(tcn.output.weight,std=.01)
        model=StableTCNResidual(base,tcn).eval()
        dry=torch.randn(1,263)*.02;controls=torch.rand(1,263,2)
        with torch.no_grad():
            full,_=model(dry,controls)
            one,state=model(dry[:,:17],controls[:,:17]);two,_=model(dry[:,17:],controls[:,17:],state)
            torch.testing.assert_close(full,torch.cat((one,two),1),atol=2e-6,rtol=0)
            zero,_=model(torch.zeros_like(dry),controls);self.assertEqual(float(zero.abs().max()),0.)
        with tempfile.TemporaryDirectory(prefix='muspector-tcn-onnx-') as directory:
            checkpoint=Path(directory)/'model.pt';graph=Path(directory)/'model.onnx'
            payload={'schema':2,'sample_rate':48000,'architecture':'stable-conditioned-lstm','device':'dfz','control_count':2,'hidden_size':4,'layers':1,'input_coef':base.input_coef,'inverted_controls':[],'state_dict':base.state_dict()}
            torch.save({'schema':6,'sample_rate':48000,'architecture':'stable-zero-gated-tcn-residual','device':'dfz','base':payload,'tcn_width':4,'tcn_blocks':3,'tcn_state_dict':tcn.state_dict()},checkpoint)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');export_stable_graph(checkpoint,graph)
            runtime=OnnxStableEffect(checkpoint,graph);actual,_=runtime(dry,controls)
            torch.testing.assert_close(full,actual,atol=2e-6,rtol=0)
            self.assertEqual(runtime.state_widths,(4,4,8,16))

    def test_stable_fir_residual_causality_stream_and_onnx(self) -> None:
        torch.manual_seed(922)
        base=StableEffectRenderer(2,hidden_size=4,layers=2).eval()
        with torch.no_grad():
            for layer in base.rnn_layers:
                layer.bias_ih_l0.zero_();layer.bias_hh_l0.zero_();layer.weight_ih_l0[8:12,-2:].zero_()
        filters=torch.zeros(9,6,5);filters[0,0,2]=.1
        model=StableEffectFIRResidual(base,filters).eval()
        dry=torch.randn(2,517)*.02;controls=torch.zeros(2,2)
        with torch.no_grad():
            original,_=base(dry,controls);full,_=model(dry,controls)
            expected=torch.nn.functional.pad(dry[:,:-2],(2,0))*.1
            torch.testing.assert_close(full-original,expected,atol=1e-7,rtol=0)
            one,state=model(dry[:,:7],controls);two,_=model(dry[:,7:],controls,state)
            torch.testing.assert_close(full,torch.cat((one,two),1),atol=2e-6,rtol=0)
            silent,_=model(torch.zeros_like(dry),torch.rand(2,517,2))
            self.assertEqual(float(silent.abs().max()),0.)
        with tempfile.TemporaryDirectory(prefix='muspector-fir-onnx-') as directory:
            checkpoint=Path(directory)/'model.pt';graph=Path(directory)/'model.onnx'
            payload={'schema':2,'sample_rate':48000,'architecture':'stable-conditioned-lstm','device':'dfz','control_count':2,'hidden_size':4,'layers':2,'input_coef':base.input_coef,'inverted_controls':[],'state_dict':base.state_dict()}
            torch.save({'schema':5,'sample_rate':48000,'architecture':'stable-causal-fir-residual','device':'dfz','base':payload,'filters':filters},checkpoint)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');export_stable_graph(checkpoint,graph)
            runtime=OnnxStableEffect(checkpoint,graph)
            actual,_=runtime(dry,controls)
            torch.testing.assert_close(full,actual,atol=2e-6,rtol=0)
            self.assertEqual(runtime.state_widths,(4,4,2))

    def test_nonlinear_order_features_and_convex_renderer_fit(self) -> None:
        rng=np.random.default_rng(921)
        dry=rng.normal(0,.03,12000).astype(np.float32)
        wet=np.tanh(dry*3)
        copies=dry.copy(),wet.copy()
        features=order_interaction_features(dry,wet)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features.shape,order_interaction_features(np.zeros(12000),np.zeros(12000)).shape)
        np.testing.assert_array_equal(dry,copies[0]);np.testing.assert_array_equal(wet,copies[1])
        first=dry;second=np.roll(dry,100);target=.3*first+.7*second
        error,weights=convex_renderer_fit([first,second],target)
        self.assertLess(error,1e-12)
        self.assertAlmostEqual(sum(weights),1.,places=12)
        self.assertTrue(all(v>=0 for v in weights))

    def test_transfer_features_are_readonly_finite_and_label_free(self) -> None:
        rng = np.random.default_rng(914)
        dry = rng.normal(0, .02, 12000).astype(np.float32)
        wet = np.tanh(dry * 2)
        before = dry.copy(), wet.copy()
        first = paired_transfer_features(dry, wet)
        np.testing.assert_array_equal(first, paired_transfer_features(dry, wet))
        np.testing.assert_array_equal(dry, before[0])
        np.testing.assert_array_equal(wet, before[1])
        self.assertTrue(np.isfinite(first).all())
        for rate in (44100, 48000, 96000):
            quiet = paired_transfer_features(np.zeros(64), np.zeros(64), rate)
            self.assertEqual(quiet.shape, first.shape)
            self.assertTrue(np.isfinite(quiet).all())
        with self.assertRaises(ValueError):
            paired_transfer_features(np.array([np.nan]), np.zeros(1))

    def test_inverse_semantics_cannot_hide_failed_audible_knobs(self) -> None:
        truth = np.zeros((20, 3)); predicted = np.zeros_like(truth)
        predicted[:, 1] = .3
        audible = np.ones(20, dtype=bool); audible[0] = False
        report = semantic_inverse_metrics(predicted, truth, audible, 'rat')
        self.assertFalse(report['semantic_passed'])
        self.assertEqual(report['per_control']['tone']['examples'], 19)
        self.assertAlmostEqual(report['per_control']['tone']['all_sample_mae'], .3)
        truth=np.zeros((100,1)); prediction=np.zeros_like(truth); prediction[-6:]=np.float32(.15)
        self.assertTrue(semantic_inverse_metrics(prediction,truth,np.ones(100,dtype=bool),'cs3')['semantic_passed'])
        prediction[-6:]=.150001
        self.assertFalse(semantic_inverse_metrics(prediction,truth,np.ones(100,dtype=bool),'cs3')['semantic_passed'])

    def test_transfer_order_decision_ignores_truth_and_protects_residual(self) -> None:
        row = {'baseline': ['drive', 'delay'], 'active': ['delay', 'drive'],
               'logits': [0., 0., 0.], 'probabilities': {'head': [.001, .5, .5]},
               'errors': [{'topology': ['drive', 'delay'], 'error': .1},
                          {'topology': ['delay', 'drive'], 'error': .2}],
               'truth': [0., 0., 0.], 'mask': [1., 0., 0.]}
        self.assertEqual(transfer_order_decision(row, 'head', 1., 0.), ('drive', 'delay'))
        row['errors'][1]['error'] = .09
        choice = transfer_order_decision(row, 'head', 1., 0.)
        self.assertEqual(choice, ('delay', 'drive'))
        row['truth'] = [1., 1., 1.]; row['mask'] = [0., 0., 0.]
        self.assertEqual(transfer_order_decision(row, 'head', 1., 0.), choice)

    def test_conditioned_readout_dynamic_stream_zero_and_knots(self) -> None:
        torch.manual_seed(914)
        base = StableEffectRenderer(2, hidden_size=4, layers=2).eval()
        with torch.no_grad():
            for layer in base.rnn_layers:
                layer.bias_ih_l0.zero_(); layer.bias_hh_l0.zero_()
                layer.weight_ih_l0[8:12, -2:].zero_()
        weights = torch.randn(3, 3, 4) * .1
        model = ConditionedReadoutEffect(base, weights).eval()
        dry = torch.randn(2, 917) * .03
        controls = torch.rand(2, 917, 2)
        with torch.no_grad():
            full, _ = model(dry, controls)
            first, state = model(dry[:, :317], controls[:, :317])
            last, _ = model(dry[:, 317:], controls[:, 317:], state)
            torch.testing.assert_close(full, torch.cat((first, last), 1), atol=2e-6, rtol=0)
            silence, _ = model(torch.zeros_like(dry), controls)
            self.assertEqual(float(silence.abs().max()), 0.)
            for i in range(3):
                for j in range(3):
                    c = torch.tensor([[i/2, j/2]]).expand(2, -1)
                    hidden, _ = base.encode(dry, c)
                    value, _ = model(dry, c)
                    torch.testing.assert_close(value, (hidden * weights[i, j]).sum(-1))

    def test_stable_onnx_dynamic_geometry_parity_and_provenance(self) -> None:
        torch.manual_seed(19)
        model = StableEffectRenderer(1, hidden_size=4, layers=2).eval()
        with torch.no_grad():
            for layer in model.rnn_layers:
                layer.bias_ih_l0.zero_()
                layer.bias_hh_l0.zero_()
                layer.weight_ih_l0[8:12, -1].zero_()
        with tempfile.TemporaryDirectory(prefix="muspector-stable-onnx-") as temporary:
            path = Path(temporary)
            checkpoint, graph = path / "test.pt", path / "test.onnx"
            payload = {"schema": 2, "sample_rate": 48000, "architecture": "stable-conditioned-lstm",
                       "device": "cs3", "control_count": 1, "hidden_size": 4, "layers": 2,
                       "input_coef": 21.4, "inverted_controls": [], "state_dict": model.state_dict()}
            torch.save(payload, checkpoint)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                export_stable_graph(checkpoint, graph)
            runtime = OnnxStableEffect(checkpoint, graph)
            with torch.inference_mode():
                for batch in (1, 3):
                    dry, controls = torch.randn(batch, 139) * 0.02, torch.rand(batch, 139, 1)
                    target, _ = model(dry, controls)
                    actual, _ = runtime(dry, controls)
                    torch.testing.assert_close(target, actual, rtol=0, atol=2e-6)
                    first, state = runtime(dry[:, :17], controls[:, :17])
                    second, _ = runtime(dry[:, 17:], controls[:, 17:], state)
                    torch.testing.assert_close(torch.cat((first, second), 1), actual, rtol=0, atol=2e-6)
                    silence, _ = runtime(torch.zeros_like(dry), controls)
                    self.assertEqual(float(silence.abs().max()), 0.0)
            other = path / "other.pt"
            torch.save(payload, other)
            with self.assertRaisesRegex(ValueError, "provenance"):
                OnnxStableEffect(other, graph)

    def test_fixed_stable_ensemble_preserves_independent_stream_state(self) -> None:
        torch.manual_seed(971)
        members = [StableEffectRenderer(1, hidden_size=4, layers=1), StableEffectRenderer(1, hidden_size=8, layers=2)]
        model = StableEffectEnsemble(members, [0.5, 0.5]).eval()
        dry, controls = torch.randn(2, 173) * 0.03, torch.rand(2, 173, 1)
        with torch.inference_mode():
            whole, state = model(dry, controls)
            expected = 0.5 * members[0](dry, controls)[0] + 0.5 * members[1](dry, controls)[0]
            torch.testing.assert_close(whole, expected, atol=0, rtol=0)
            first, carried = model(dry[:, :29], controls[:, :29])
            second, _ = model(dry[:, 29:], controls[:, 29:], carried)
            torch.testing.assert_close(torch.cat((first, second), 1), whole, atol=2e-6, rtol=0)
        self.assertEqual(len(state), 3)
        self.assertEqual(model.state_floats_per_mono_stream, 40)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            StableEffectEnsemble(members, [0.5, 0.6])
        with self.assertRaisesRegex(ValueError, "sum to one"):
            StableEffectEnsemble(members, [-0.1, 1.1])
        with self.assertRaisesRegex(ValueError, "layer count"):
            model(dry, controls, state[:1])

    def test_capacity_promotion_requires_complete_and_same_checkpoint_evidence(self) -> None:
        row = {"checkpoint_sha256": "candidate"}
        incomplete = {"results": [row], "expected_candidates": 2, "screen_complete": False}
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_capacity_evidence(incomplete, {}, {}, {}, "candidate")
        complete = {**incomplete, "expected_candidates": 1, "screen_complete": True}
        with self.assertRaisesRegex(ValueError, "different checkpoint bytes"):
            validate_capacity_evidence(complete, {"checkpoint_sha256": "other"}, {}, {}, "candidate")

    def test_capacity_selection_does_not_trade_away_an_admission_gate(self) -> None:
        admitted = {"passes_selection_gate": True, "absolute_peak_error_p95": 0.015,
                    "worst_attack_absolute_peak_error_p95": 0.025, "global_esr": 0.01}
        rejected = {**admitted, "passes_selection_gate": False, "absolute_peak_error_p95": 0.005,
                    "worst_attack_absolute_peak_error_p95": 0.04}
        self.assertLess(capacity_selection_key(admitted), capacity_selection_key(rejected))

    def test_generic_stable_benchmark_keeps_dynamic_silence_and_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="muspector-effect-benchmark-") as temporary:
            model = StableEffectRenderer(1, hidden_size=4, layers=1)
            for parameter in model.parameters():
                torch.nn.init.zeros_(parameter)
            checkpoint = Path(temporary) / "test.pt"
            torch.save({"schema": 2, "sample_rate": 48000, "architecture": "stable-conditioned-lstm",
                        "device": "cs3", "control_count": 1, "hidden_size": 4, "layers": 1,
                        "input_coef": 21.4, "inverted_controls": [], "state_dict": model.state_dict()}, checkpoint)
            report = benchmark_stable_effect(checkpoint, iterations=1)
            self.assertTrue(report["additional_safety_passed"])
            self.assertEqual(report["dynamic_control_silence_peak"], 0.0)
            self.assertFalse(report["physical_audio_devices_used"])

    def test_generic_asrnn_effect_admission_keeps_take_id_out_of_controls(self) -> None:
        with tempfile.TemporaryDirectory(prefix="muspector-asrnn-effects-") as temporary:
            root = Path(temporary)
            time = np.arange(48_000, dtype=np.float32) / 48_000.0
            dry = 0.04 * np.sin(2.0 * np.pi * 137.0 * time)
            wet = np.tanh(dry * 2.0)
            for device, names in (
                ("dfz", ("50,100,7.wav", "50,100,8.wav")),
                ("cs3", ("70,7.wav", "70,8.wav")),
            ):
                for split, name in zip(("train", "eval"), names):
                    directory = root / device / split
                    directory.mkdir(parents=True)
                    offset = 0 if split == "train" else 19
                    soundfile.write(
                        directory / name,
                        np.stack((np.roll(dry, offset), np.roll(wet, offset)), axis=1),
                        48_000,
                        subtype="FLOAT",
                    )
                report = audit_effect(root, device)
                self.assertTrue(report["passed"])
                self.assertEqual(report["official_train_eval_exact_dry_duplicates"], 0)
            _, _, dfz_controls = read_effect_pair(root / "dfz/train/50,100,7.wav", "dfz")
            _, _, cs3_controls = read_effect_pair(root / "cs3/train/70,7.wav", "cs3")
            np.testing.assert_allclose(dfz_controls, (0.5, 1.0))
            np.testing.assert_allclose(cs3_controls, (0.7,))

    def test_stable_effect_runtime_supports_variable_control_width(self) -> None:
        for controls in (1, 2):
            model = StableEffectRenderer(controls, hidden_size=4, layers=2).eval()
            for parameter in model.parameters():
                torch.nn.init.zeros_(parameter)
            dry = torch.randn(2, 191) * 0.04
            settings = torch.rand(2, controls)
            whole, _ = model(dry, settings)
            first, state = model(dry[:, :73], settings)
            second, _ = model(dry[:, 73:], settings, state)
            torch.testing.assert_close(whole, torch.cat((first, second), dim=1))
            silent, _ = model(torch.zeros_like(dry), settings)
            self.assertEqual(float(silent.detach().abs().max()), 0.0)

    def test_stable_effect_adapter_is_zero_safe_and_streamable(self) -> None:
        adapter = StableEffectResidualAdapter(control_count=1, hidden_size=6).eval()
        dry = torch.randn(2, 1_117) * 0.04
        base = torch.tanh(dry * 2.0)
        controls = torch.tensor(((0.2,), (0.8,)))
        whole, _ = adapter(dry, base, controls)
        first, state = adapter(dry[:, :319], base[:, :319], controls)
        second, _ = adapter(dry[:, 319:], base[:, 319:], controls, state)
        torch.testing.assert_close(whole, torch.cat((first, second), dim=1))
        silent, _ = adapter(torch.zeros_like(dry), torch.zeros_like(base), controls)
        self.assertEqual(float(silent.detach().abs().max()), 0.0)

    def test_centered_stable_effect_cancels_zero_and_projects_recurrence(self) -> None:
        base = StableEffectRenderer(1, hidden_size=4, layers=2)
        centered = CenteredStableEffectRenderer(base).eval()
        with torch.no_grad():
            for layer in base.rnn_layers:
                layer.weight_hh_l0[8:12].fill_(0.5)
        norms = centered.project_candidate_recurrence(0.9)
        self.assertTrue(all(norm <= 0.900001 for norm in norms))
        dry = torch.randn(2, 379) * 0.04
        controls = torch.tensor(((0.2,), (0.8,)))
        whole, _ = centered(dry, controls)
        first, state = centered(dry[:, :113], controls)
        second, _ = centered(dry[:, 113:], controls, state)
        torch.testing.assert_close(whole, torch.cat((first, second), dim=1))
        silence, _ = centered(torch.zeros_like(dry), controls)
        self.assertEqual(float(silence.detach().abs().max()), 0.0)

    def test_phase7_calibration_is_balanced_and_group_cvar_targets_tail(self) -> None:
        rows = {}
        for attack in (0, 10):
            for index in range(4):
                rows[f"{attack},{index}.wav"] = {
                    "attack": attack,
                    "features": np.asarray(
                        (index, index**2, index % 2, index / 3, index / 4, index / 5),
                        dtype=np.float64,
                    ),
                }
        selected = coverage_calibration_files(rows, count_per_control=2)
        self.assertEqual(len(selected), 4)
        self.assertEqual(
            {attack: sum(name.startswith(f"{attack},") for name in selected) for attack in (0, 10)},
            {0: 2, 10: 2},
        )
        losses = torch.tensor((1.0, 2.0, 3.0, 4.0))
        self.assertAlmostEqual(float(_group_cvar(losses)), 3.25)

    def test_phase7_near_performance_audit_does_not_rely_on_exact_hashes(self) -> None:
        rng = np.random.default_rng(71)
        envelope = rng.random(100)
        wave = rng.normal(size=3_000)
        rows = {
            "first": {"envelope": envelope, "wave_probe": wave},
            "near": {"envelope": envelope * 1.01, "wave_probe": wave * 0.99 + 1.0e-6},
            "other": {"envelope": rng.random(100), "wave_probe": rng.normal(size=3_000)},
        }
        pairs = near_performance_pairs(rows)
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0]["first"], pairs[0]["second"]), ("first", "near"))

    def test_phase7_partition_requires_near_performance_audit(self) -> None:
        paths = [Path("first.wav"), Path("second.wav")]
        audit = {"fit_files": ["first.wav"], "calibration_files": ["second.wav"]}
        with self.assertRaises(ValueError):
            phase7_partition(paths, audit)
        audit["fit_calibration_near_performance_pairs"] = 1
        with self.assertRaises(ValueError):
            phase7_partition(paths, audit)
        audit["fit_calibration_near_performance_pairs"] = 0
        fit, calibration = phase7_partition(paths, audit)
        self.assertEqual((fit, calibration), ([paths[0]], [paths[1]]))

    def test_phase7_ood_parity_probe_detects_lost_state(self) -> None:
        class Accumulator(torch.nn.Module):
            def __init__(self, lose_state: bool):
                super().__init__()
                self.reference = torch.nn.Parameter(torch.zeros(()))
                self.lose_state = lose_state

            def forward(self, dry, controls, state=None):
                if state is None or self.lose_state:
                    state = torch.zeros(dry.shape[0], 1, device=dry.device)
                output = dry.cumsum(1) + state
                return output, output[:, -1:]

        dry = np.ones((1, 512), dtype=np.float32) * 0.01
        _, good_error, _ = phase7_ood_render(Accumulator(False), dry, 0.5)
        _, bad_error, _ = phase7_ood_render(Accumulator(True), dry, 0.5)
        self.assertLess(good_error, 2.0e-6)
        self.assertGreater(bad_error, 0.1)

    def test_phase7_summary_rejects_incomplete_or_mismatched_evidence(self) -> None:
        training = {
            "admitted_for_development_challenge": False,
            "checkpoint_sha256": "candidate",
        }
        source = {"passed": True, "archive_md5": "archive"}
        ood = {
            "checkpoint_sha256": "candidate",
            "archive_md5": "archive",
            "candidate_diagnostic_only": True,
        }
        cases = [
            (None, None, "requires OOD"),
            ({}, ood, "rejected calibration"),
            (None, {**ood, "checkpoint_sha256": "other"}, "different checkpoints"),
            (None, {**ood, "archive_md5": "other"}, "different archives"),
            (None, {**ood, "candidate_diagnostic_only": False}, "diagnostic-only label"),
        ]
        for challenge, external, message in cases:
            with self.subTest(message=message):
                with patch(
                    "remix.summarize_asrnn_phase7._read",
                    side_effect=[training, {}, {}, source, challenge, external],
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        summarize_phase7(Path("unused-phase7-test-fixture"))

    def test_rat_inverse_control_gates_keep_tone_failure_visible(self) -> None:
        report = {
            "official_eval": {
                "per_control": {
                    "volume": {"mae_normalized": 0.02, "p95_normalized": 0.05}
                },
                "audible_distortion_tone": {
                    "distortion": {
                        "mae_normalized": 0.03,
                        "p95_normalized": 0.10,
                    },
                    "tone": {"mae_normalized": 0.09, "p95_normalized": 0.28},
                },
            }
        }
        decisions = control_decisions(report)
        self.assertTrue(decisions["volume"]["passed"])
        self.assertTrue(decisions["distortion"]["passed"])
        self.assertFalse(decisions["tone"]["passed"])
        self.assertEqual(
            decisions["distortion"]["population"],
            "audible official-eval examples only",
        )

    def test_stable_rat_runtime_maps_tone_and_carries_state(self) -> None:
        model = StableRatRenderer(hidden_size=4, layers=2).eval()
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        controls = torch.tensor(((0.2, 0.7, 0.4),))
        official = model._official_controls(controls, 3)
        torch.testing.assert_close(
            official[0, 0], torch.tensor((-0.6, -0.4, -0.2))
        )
        dry = torch.randn(1, 127) * 0.04
        whole, _ = model(dry, controls)
        first, state = model(dry[:, :31], controls)
        second, _ = model(dry[:, 31:], controls, state)
        torch.testing.assert_close(whole, torch.cat((first, second), dim=1))
        dynamic = controls.unsqueeze(1).expand(-1, 127, -1)
        silent, _ = model(torch.zeros_like(dry), dynamic)
        self.assertEqual(float(silent.detach().abs().max()), 0.0)

    def test_rat_direct_renderer_is_streamable_and_silent_by_construction(self) -> None:
        model = RatDirectRenderer(hidden_size=8, layers=2).eval()
        torch.manual_seed(17)
        dry = torch.randn(2, 1_113) * 0.04
        controls = torch.tensor(((0.2, 0.7, 0.4), (0.9, 0.1, 0.0)))
        whole, _ = model(dry, controls)
        first, state = model(dry[:, :317], controls)
        second, _ = model(dry[:, 317:], controls, state)
        torch.testing.assert_close(whole, torch.cat((first, second), dim=1))
        self.assertEqual(float(whole[1].detach().abs().max()), 0.0)
        silent, _ = model(torch.zeros_like(dry), controls)
        self.assertEqual(float(silent.detach().abs().max()), 0.0)

    def test_asrnn_rat_admission_preserves_stereo_dry_wet_and_controls(self) -> None:
        with tempfile.TemporaryDirectory(prefix="muspector-asrnn-") as temporary:
            root = Path(temporary) / "RAT"
            time = np.arange(48_000, dtype=np.float32) / 48_000.0
            dry = 0.05 * np.sin(2.0 * np.pi * 110.0 * time)
            wet = np.tanh(dry * 3.0) * 0.8
            for split, name in (("train", "20,30,40.wav"), ("eval", "20,30,40.wav")):
                directory = root / split
                directory.mkdir(parents=True)
                split_dry = dry if split == "train" else np.roll(dry, 17)
                split_wet = wet if split == "train" else np.roll(wet, 17)
                soundfile.write(
                    directory / name,
                    np.stack((split_dry, split_wet), axis=1),
                    48_000,
                    subtype="FLOAT",
                )
            report = audit_asrnn(Path(temporary))
            self.assertTrue(report["passed"])
            self.assertEqual(report["files"], 2)
            self.assertEqual(report["official_train_eval_shared_control_settings"], 1)
            self.assertEqual(report["official_train_eval_exact_dry_duplicates"], 0)
            loaded_dry, loaded_wet, controls = read_rat_pair(root / "train/20,30,40.wav")
            np.testing.assert_array_equal(loaded_dry, dry)
            np.testing.assert_allclose(loaded_wet, wet, rtol=0.0, atol=1.0e-7)
            np.testing.assert_allclose(controls, (0.2, 0.7, 0.4), rtol=0.0, atol=1.0e-7)

    def test_zero_drive_adapter_is_exactly_neutral_and_streamable(self) -> None:
        adapter = DriveDeviceAdapter(hidden_size=8)
        torch.nn.init.zeros_(adapter.output.weight)
        with tempfile.TemporaryDirectory(prefix="muspector-drive-adapter-") as temporary:
            checkpoint = Path(temporary) / "adapter.pt"
            torch.save(
                {
                    "schema": 1,
                    "sample_rate": 48_000,
                    "hidden_size": adapter.hidden_size,
                    "state_dict": adapter.state_dict(),
                },
                checkpoint,
            )
            base = ForwardChainRuntime()
            calibrated = ForwardChainRuntime(drive_adapter=checkpoint)
            rng = np.random.default_rng(20270130)
            dry = (rng.standard_normal(6_173) * 0.04).astype(np.float32)
            spec = ChainSpec((Drive(13.0, 0.61, -4.0), Delay(80.0, 0.3, 0.2)))
            expected = base.render(dry, spec)
            self.assertTrue(np.array_equal(calibrated.render(dry, spec), expected))
            streamed = calibrated.stream(dry, spec, 1_024)
            self.assertLessEqual(float(np.max(np.abs(streamed - expected))), 2.0e-6)
            silence = calibrated.stream(np.zeros_like(dry), spec)
            self.assertEqual(float(np.max(np.abs(silence))), 0.0)

    def test_forward_chain_preserves_order_bypass_and_stream_state(self) -> None:
        frames = 9_113
        rng = np.random.default_rng(20261230)
        dry = (rng.standard_normal(frames) * 0.04).astype(np.float32)
        profile = ReverbDeviceProfile(
            sample_rate=48_000,
            mean_ir=np.pad(np.asarray([1.0], dtype=np.float32), (0, 383_999)),
            basis_ir=np.zeros((1, 384_000), dtype=np.float32),
            decay_grid=np.asarray([0.2], dtype=np.float64),
            damping_grid=np.asarray([0.0], dtype=np.float64),
            coefficient_grid=np.zeros((1, 1, 1), dtype=np.float32),
            source_domain="unit-test",
            calibration_hash="unit-test",
        )
        runtime = ForwardChainRuntime(reverb_profile=profile)
        self.assertTrue(np.array_equal(runtime.render(dry, ChainSpec(())), dry))
        spec = ChainSpec(
            (
                Delay(61.0, 0.44, 0.31),
                Drive(8.0, 0.62, -4.0),
                Reverb(1.2, 0.4, 0.2),
            )
        )
        complete = runtime.render(dry, spec)
        streamed = runtime.stream(dry, spec, 1_024)
        self.assertLessEqual(float(np.max(np.abs(complete - streamed))), 2.0e-6)
        reverse = runtime.render(dry, ChainSpec(tuple(reversed(spec.effects))))
        self.assertGreater(float(np.mean(np.abs(complete - reverse))), 1.0e-5)
        silence = runtime.stream(np.zeros(frames, dtype=np.float32), spec)
        self.assertEqual(float(np.max(np.abs(silence))), 0.0)

    def test_hardware_capture_manifest_enforces_alignment_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="muspector-remix-capture-") as temporary:
            root = Path(temporary)
            clean = root / "clean.wav"
            wet = root / "wet.wav"
            audio = np.linspace(-0.2, 0.2, 4_800, dtype=np.float32)
            soundfile.write(clean, audio, 48_000, subtype="FLOAT")
            soundfile.write(wet, audio * 0.8, 48_000, subtype="FLOAT")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "records": [
                            {
                                "id": "pedal-a-session-1-setting-1",
                                "split": "train",
                                "device_id": "pedal-a",
                                "session_id": "session-1",
                                "player_id": "player-1",
                                "clean": clean.name,
                                "wet": wet.name,
                                "clean_sha256": capture_sha256(clean),
                                "wet_sha256": capture_sha256(wet),
                                "measured_latency_samples": 37,
                                "latency_compensated": True,
                                "raw_program": clean.name,
                                "raw_program_sha256": capture_sha256(clean),
                                "raw_latency_capture": clean.name,
                                "raw_latency_capture_sha256": capture_sha256(clean),
                                "raw_wet_capture": wet.name,
                                "raw_wet_capture_sha256": capture_sha256(wet),
                                "capture_quality": {
                                    "passed": True,
                                    "sample_rate": 48_000,
                                    "frames": 4_800,
                                    "channels": 1,
                                    "measured_latency_samples": 37,
                                    "clipped_samples": 0,
                                    "dropout_blocks": 0,
                                    "automatic_normalization": False,
                                },
                                "chain": ChainSpec(
                                    (Drive(12.0, 0.5, -3.0),)
                                ).document(),
                            }
                        ],
                    }
                )
            )
            report = validate_manifest(manifest)
            self.assertEqual(report["records"], 1)
            self.assertTrue(report["session_disjoint"])
            invalid_quality = json.loads(manifest.read_text())
            invalid_quality["records"][0]["capture_quality"]["passed"] = False
            manifest.write_text(json.dumps(invalid_quality))
            with self.assertRaises(ValueError):
                validate_manifest(manifest)
            invalid_quality["records"][0]["capture_quality"]["passed"] = True
            invalid_quality["records"][0]["split"] = "unknown"
            manifest.write_text(json.dumps(invalid_quality))
            with self.assertRaises(ValueError):
                validate_manifest(manifest)

    def test_drive_capture_plan_is_deterministic_and_split_local(self) -> None:
        first = drive_capture_plan(device_id="drive-a", player_id="player-a")
        second = drive_capture_plan(device_id="drive-a", player_id="player-a")
        self.assertEqual(first, second)
        self.assertEqual(first["sample_rate"], CAPTURE_SAMPLE_RATE)
        self.assertEqual(first["training_records"], 192)
        self.assertEqual(len(first["sessions"]), 12)
        self.assertEqual(len(set(first["source_programs"].values())), 4)
        pilot = drive_pilot_session(first, "train-session-01")
        self.assertEqual(pilot["training_takes"], 16)
        self.assertEqual(len(pilot["takes"]), 18)
        self.assertEqual(pilot["takes"][0]["phase"], "before")
        self.assertEqual(pilot["takes"][-1]["phase"], "after")
        coverage = plan_coverage(first)
        for split in ("train", "calibrate", "valid", "locked-final"):
            for axis in ("gain", "tone", "level"):
                self.assertTrue(coverage[split]["axes"][axis]["strata_complete"])
        session_splits = {}
        for record in first["records"]:
            previous = session_splits.setdefault(record["session_id"], record["split"])
            self.assertEqual(previous, record["split"])
        readiness = audit_drive_plan(first)
        self.assertTrue(readiness["passed"])
        self.assertTrue(readiness["replay_program_disjoint"])

    def test_capture_plan_rejects_replay_program_split_leakage(self) -> None:
        plan = drive_capture_plan(device_id="drive-a", player_id="player-a")
        plan["records"][144]["source_program_id"] = plan["records"][0][
            "source_program_id"
        ]
        report = audit_drive_plan(plan)
        self.assertFalse(report["passed"])
        self.assertIn(
            "replay-program-split-leakage",
            {issue["code"] for issue in report["issues"]},
        )

    def test_source_registry_blocks_unverified_public_training(self) -> None:
        registry = validate_source_registry(
            Path(__file__).with_name("data_sources.json"),
            workspace=Path(__file__).parent.parent,
        )
        self.assertTrue(registry["passed"])
        sources = {entry["id"]: entry for entry in registry["entries"]}
        self.assertEqual(sources["pod-set"]["admission"], "metadata-only")
        self.assertFalse(sources["pod-set"]["local"])
        self.assertEqual(registry["audio_files_opened"], 0)

    def test_development_manifest_never_opens_locked_final_audio(self) -> None:
        with tempfile.TemporaryDirectory(prefix="muspector-remix-locked-") as temporary:
            root = Path(temporary)
            clean = root / "clean.wav"
            wet = root / "wet.wav"
            audio = np.linspace(-0.2, 0.2, 4_800, dtype=np.float32)
            soundfile.write(clean, audio, 48_000, subtype="FLOAT")
            soundfile.write(wet, audio * 0.8, 48_000, subtype="FLOAT")

            def record(identifier: str, split: str, digest_prefix: str) -> dict:
                is_locked = split == "locked-final"
                clean_hash = digest_prefix * 64 if is_locked else capture_sha256(clean)
                wet_hash = ("f" if is_locked else "") * 64 if is_locked else capture_sha256(wet)
                program_hash = ("e" * 64) if is_locked else capture_sha256(clean)
                return {
                    "id": identifier,
                    "split": split,
                    "device_id": "pedal-a",
                    "session_id": f"{split}-session-1",
                    "player_id": "player-1",
                    "clean": "missing-locked-clean.wav" if is_locked else clean.name,
                    "wet": "missing-locked-wet.wav" if is_locked else wet.name,
                    "clean_sha256": clean_hash,
                    "wet_sha256": wet_hash,
                    "raw_program": "missing-locked-program.wav" if is_locked else clean.name,
                    "raw_program_sha256": program_hash,
                    "raw_latency_capture": (
                        "missing-locked-loopback.wav" if is_locked else clean.name
                    ),
                    "raw_latency_capture_sha256": program_hash,
                    "raw_wet_capture": "missing-locked-raw.wav" if is_locked else wet.name,
                    "raw_wet_capture_sha256": wet_hash,
                    "measured_latency_samples": 37,
                    "latency_compensated": True,
                    "capture_quality": {
                        "passed": True,
                        "sample_rate": 48_000,
                        "frames": 4_800,
                        "channels": 1,
                        "measured_latency_samples": 37,
                        "clipped_samples": 0,
                        "dropout_blocks": 0,
                        "automatic_normalization": False,
                    },
                    "chain": ChainSpec((Drive(12.0, 0.5, -3.0),)).document(),
                }

            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "records": [
                            record("train-1", "train", "d"),
                            record("locked-1", "locked-final", "d"),
                        ],
                    }
                )
            )
            report = validate_manifest(
                manifest,
                excluded_splits=frozenset({"locked-final"}),
            )
            self.assertEqual(report["records"], 2)
            self.assertEqual(report["records_inspected"], 1)
            self.assertEqual(report["excluded_splits"], ["locked-final"])

            class NeutralBase(torch.nn.Module):
                def forward(self, dry, controls, state=None):
                    return dry, state

            examples = capture_examples(
                manifest,
                NeutralBase(),
                "train",
                frames=1_024,
                windows_per_record=1,
            )
            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0]["record_id"], "train-1")
            with self.assertRaises(ValueError):
                capture_examples(
                    manifest,
                    NeutralBase(),
                    "locked-final",
                    frames=1_024,
                    windows_per_record=1,
                )

    def test_capture_kit_recovers_latency_and_rejects_clipping(self) -> None:
        report = capture_smoke_report()
        self.assertEqual(
            report["latency"]["measured_samples"],
            report["latency"]["injected_samples"],
        )
        self.assertGreater(report["latency"]["confidence"], 0.99)
        self.assertEqual(report["alignment"]["max_absolute_error"], 0.0)
        self.assertTrue(report["quality"]["passed"])
        self.assertTrue(report["negative_gate"]["clipped_capture_rejected"])
        self.assertTrue(report["drift_gate"]["stable_anchor_passed"])
        self.assertTrue(report["drift_gate"]["changed_anchor_rejected"])

    def test_capture_kit_preserves_raw_files_and_emits_valid_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="muspector-capture-kit-") as temporary:
            root = Path(temporary)
            frames = 2 * CAPTURE_SAMPLE_RATE
            time = np.arange(frames, dtype=np.float32) / CAPTURE_SAMPLE_RATE
            source_audio = (0.12 * np.sin(2.0 * np.pi * 110.0 * time)).astype(np.float32)
            source_path = root / "source.wav"
            program_path = root / "raw-program.wav"
            metadata_path = root / "raw-program.json"
            latency_path = root / "raw-loopback.wav"
            wet_raw_path = root / "raw-wet.wav"
            soundfile.write(source_path, source_audio, CAPTURE_SAMPLE_RATE, subtype="FLOAT")
            write_capture_program(source_path, program_path, metadata_path)
            program, rate = soundfile.read(program_path, dtype="float32")
            self.assertEqual(rate, CAPTURE_SAMPLE_RATE)
            latency = 211

            def delayed(value: np.ndarray) -> np.ndarray:
                result = np.zeros_like(value)
                result[latency:] = value[:-latency]
                return result

            soundfile.write(
                latency_path,
                delayed(program * np.float32(0.75)),
                rate,
                subtype="FLOAT",
            )
            drive = Drive(12.0, 0.6, -12.0)
            effected = render_chain(program, ChainSpec((drive,)), rate, "alternate")
            soundfile.write(wet_raw_path, delayed(effected), rate, subtype="FLOAT")
            raw_before = {
                path: path.read_bytes()
                for path in (program_path, latency_path, wet_raw_path)
            }
            manifest_path = root / "manifest.json"
            result = prepare_capture_record(
                program_path=program_path,
                program_metadata_path=metadata_path,
                latency_capture_path=latency_path,
                wet_raw_path=wet_raw_path,
                clean_output_path=root / "aligned-clean.wav",
                wet_output_path=root / "aligned-wet.wav",
                manifest_path=manifest_path,
                capture_id="drive-a-setting-1",
                split="train",
                device_id="drive-a",
                session_id="train-session-01",
                player_id="player-a",
                drive=drive,
                rights="synthetic-smoke",
                source_kind="synthetic-smoke",
            )
            self.assertEqual(result["validation"]["records"], 1)
            record = result["manifest"]["records"][0]
            self.assertEqual(record["measured_latency_samples"], latency)
            self.assertTrue(record["capture_quality"]["passed"])
            second_drive = Drive(18.0, 0.25, -12.0)
            second_raw = root / "raw-wet-2.wav"
            second_effected = render_chain(
                program,
                ChainSpec((second_drive,)),
                rate,
                "alternate",
            )
            soundfile.write(second_raw, delayed(second_effected), rate, subtype="FLOAT")
            appended = prepare_capture_record(
                program_path=program_path,
                program_metadata_path=metadata_path,
                latency_capture_path=latency_path,
                wet_raw_path=second_raw,
                clean_output_path=root / "aligned-clean-2.wav",
                wet_output_path=root / "aligned-wet-2.wav",
                manifest_path=manifest_path,
                capture_id="drive-a-setting-2",
                split="train",
                device_id="drive-a",
                session_id="train-session-01",
                player_id="player-a",
                drive=second_drive,
                rights="synthetic-smoke",
                source_kind="synthetic-smoke",
                append=True,
            )
            self.assertEqual(appended["validation"]["records"], 2)
            self.assertEqual(len(json.loads(manifest_path.read_text())["records"]), 2)
            rejected_raw = root / "raw-wet-rejected.wav"
            soundfile.write(rejected_raw, delayed(second_effected), rate, subtype="FLOAT")
            rejected_clean = root / "aligned-clean-rejected.wav"
            rejected_wet = root / "aligned-wet-rejected.wav"
            with self.assertRaises(ValueError):
                prepare_capture_record(
                    program_path=program_path,
                    program_metadata_path=metadata_path,
                    latency_capture_path=latency_path,
                    wet_raw_path=rejected_raw,
                    clean_output_path=rejected_clean,
                    wet_output_path=rejected_wet,
                    manifest_path=manifest_path,
                    capture_id="drive-a-setting-rejected",
                    split="valid",
                    device_id="drive-a",
                    session_id="train-session-01",
                    player_id="player-a",
                    drive=second_drive,
                    rights="synthetic-smoke",
                    source_kind="synthetic-smoke",
                    append=True,
                )
            self.assertFalse(rejected_clean.exists())
            self.assertFalse(rejected_wet.exists())
            self.assertEqual(len(json.loads(manifest_path.read_text())["records"]), 2)
            for path, contents in raw_before.items():
                self.assertEqual(path.read_bytes(), contents)

    def test_forward_drive_is_causal_and_chunk_state_is_exact(self) -> None:
        torch.manual_seed(29)
        model = DriveForwardRenderer(hidden_size=8).eval()
        dry = torch.randn((1, 2_048)) * 0.05
        controls = torch.tensor(((0.4, 0.7, 0.5),), dtype=torch.float32)
        complete, _ = model(dry, controls)
        first, state = model(dry[:, :1_024], controls)
        second, _ = model(dry[:, 1_024:], controls, state)
        np.testing.assert_allclose(
            torch.cat((first, second), dim=1).detach().numpy(),
            complete.detach().numpy(),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        changed = dry.clone()
        changed[:, 1_024:] *= -3.0
        changed_output, _ = model(changed, controls)
        np.testing.assert_array_equal(
            changed_output[:, :1_024].detach().numpy(),
            complete[:, :1_024].detach().numpy(),
        )
        silence, _ = model(torch.zeros_like(dry), controls)
        np.testing.assert_array_equal(silence.detach().numpy(), np.zeros_like(dry.numpy()))

    def test_forward_drive_multicomponent_loss_is_finite_and_trainable(self) -> None:
        torch.manual_seed(31)
        model = DriveForwardRenderer(hidden_size=8)
        dry = torch.randn((2, 2_048)) * 0.04
        target = torch.tanh(dry * 3.0) * 0.5
        controls = torch.tensor(((0.2, 0.3, 0.4), (0.8, 0.7, 0.6)))
        prediction, _ = model(dry, controls)
        loss, parts = forward_loss(prediction, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(parts),
            {
                "esr",
                "preemphasis_esr",
                "spectral",
                "normalized_l1",
                "relative_peak",
                "amplitude_overshoot",
            },
        )
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in model.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )

    def test_forward_drive_level_is_an_exact_final_physical_gain(self) -> None:
        torch.manual_seed(37)
        model = DriveForwardRenderer(hidden_size=8).eval()
        dry = torch.randn((1, 2_048)) * 0.05
        low = torch.tensor(((0.6, 0.3, 0.0),), dtype=torch.float32)
        high = torch.tensor(((0.6, 0.3, 1.0),), dtype=torch.float32)
        low_output, _ = model(dry, low)
        high_output, _ = model(dry, high)
        expected_ratio = 10.0 ** (30.0 / 20.0)
        np.testing.assert_allclose(
            high_output.detach().numpy(),
            low_output.detach().numpy() * expected_ratio,
            rtol=2.0e-6,
            atol=2.0e-7,
        )
        with self.assertRaisesRegex(ValueError, "normalized"):
            model(dry, torch.tensor(((1.01, 0.3, 0.5),), dtype=torch.float32))

    def test_forward_delay_preserves_physical_echo_geometry_and_bypass(self) -> None:
        model = DelayForwardRenderer(fir_taps=16).eval()
        with torch.no_grad():
            model.blend_logits.copy_(torch.tensor((20.0, -20.0, -20.0)))
        dry = torch.zeros((1, 5_000), dtype=torch.float32)
        dry[:, 0] = 1.0
        controls = torch.tensor(((0.0, 0.5, 1.0),), dtype=torch.float32)
        rendered = model(dry, controls)
        self.assertAlmostEqual(float(rendered[0, 0].detach()), 0.3, places=6)
        self.assertAlmostEqual(float(rendered[0, 1_920].detach()), 0.7, places=6)
        self.assertAlmostEqual(float(rendered[0, 3_840].detach()), 0.7 * 0.45, places=6)
        self.assertEqual(int(torch.count_nonzero(rendered[:, 1:1_920])), 0)
        bypass = model(dry, torch.tensor(((1.0, 1.0, 0.0),), dtype=torch.float32))
        np.testing.assert_array_equal(bypass.detach().numpy(), dry.numpy())
        silence = model(torch.zeros_like(dry), controls)
        np.testing.assert_array_equal(silence.detach().numpy(), np.zeros_like(dry.numpy()))

    def test_forward_delay_controls_round_trip_and_filter_is_trainable(self) -> None:
        effect = Delay(40.0, 0.45, 0.3)
        controls = normalized_delay_controls(effect)
        restored = controls_to_delay(controls)
        self.assertAlmostEqual(restored.time_ms, effect.time_ms, places=4)
        self.assertAlmostEqual(restored.feedback, effect.feedback, places=6)
        self.assertAlmostEqual(restored.mix, effect.mix, places=6)
        model = DelayForwardRenderer(fir_taps=16)
        dry = torch.randn((1, 5_000), dtype=torch.float32) * 0.05
        control_tensor = torch.from_numpy(controls).unsqueeze(0)
        prediction = model(dry, control_tensor)
        target = torch.from_numpy(
            render_chain(
                dry.squeeze(0).numpy(),
                ChainSpec((effect,)),
                48_000,
                "alternate",
            )
        ).unsqueeze(0)
        loss, _ = forward_loss(prediction, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.cutoff_logit.grad.abs()), 0.0)
        self.assertGreater(float(model.blend_logits.grad.abs().sum()), 0.0)

    def test_forward_delay_streaming_state_matches_complete_render(self) -> None:
        torch.manual_seed(41)
        model = DelayForwardRenderer(fir_taps=16).eval()
        dry = torch.randn((1, 20_123), dtype=torch.float32) * 0.05
        controls = torch.tensor(((0.19, 0.71, 0.58),), dtype=torch.float32)
        complete = model(dry, controls)
        state = None
        pieces = []
        for start in range(0, dry.shape[1], 509):
            piece, state = stream_delay_block(
                model,
                dry[:, start : start + 509],
                controls,
                state,
            )
            pieces.append(piece)
        chunked = torch.cat(pieces, dim=1)
        np.testing.assert_allclose(
            chunked.detach().numpy(),
            complete.detach().numpy(),
            rtol=2.0e-6,
            atol=2.0e-7,
        )

    def test_forward_reverb_profile_preserves_bypass_silence_and_streaming(self) -> None:
        sample_rate = 100
        frames = sample_rate * 8
        mean = np.zeros(frames, dtype=np.float32)
        mean[[3, 17, 41]] = (0.5, -0.2, 0.1)
        basis = np.zeros((2, frames), dtype=np.float32)
        basis[0, [5, 23]] = (0.2, 0.08)
        basis[1, [9, 37]] = (-0.15, 0.06)
        profile = ReverbDeviceProfile(
            sample_rate,
            mean,
            basis,
            np.asarray((0.2, 8.0)),
            np.asarray((0.0, 1.0)),
            np.asarray(
                (
                    ((0.1, -0.1), (0.2, -0.2)),
                    ((0.3, -0.3), (0.4, -0.4)),
                ),
                dtype=np.float32,
            ),
            "test",
            "test-calibration",
        )
        dry = np.random.default_rng(43).standard_normal(256).astype(np.float32) * 0.03
        effect = Reverb(2.0, 0.4, 0.5)
        complete = profile.render(dry, effect)
        state = None
        pieces = []
        for start in range(0, len(dry), 31):
            piece, state = stream_reverb_block(profile, dry[start : start + 31], effect, state)
            pieces.append(piece)
        np.testing.assert_allclose(np.concatenate(pieces), complete, rtol=1.0e-6, atol=1.0e-7)
        np.testing.assert_array_equal(profile.render(dry, Reverb(8.0, 1.0, 0.0)), dry)
        np.testing.assert_array_equal(
            profile.render(np.zeros_like(dry), Reverb(8.0, 1.0, 0.7)),
            np.zeros_like(dry),
        )

    def test_inference_input_is_read_only_and_uses_analysis_copies(self) -> None:
        root = Path("/tmp")
        identifier = f"muspector-remix-{os.getpid()}"
        clean = root / f"{identifier}-clean.wav"
        wet = root / f"{identifier}-wet.wav"
        frames = 48_000
        time = np.arange(frames, dtype=np.float32) / 48_000.0
        stereo = np.stack(
            (0.1 * np.sin(2.0 * np.pi * 220.0 * time), 0.05 * np.sin(2.0 * np.pi * 330.0 * time)),
            axis=1,
        ).astype(np.float32)
        soundfile.write(clean, stereo, 48_000, subtype="FLOAT")
        soundfile.write(wet, stereo * 0.8, 48_000, subtype="FLOAT")
        before = (clean.read_bytes(), wet.read_bytes())
        try:
            dry_copy, wet_copy, report = load_analysis_pair(
                clean, wet, 44_100, 5, 0.0
            )
            self.assertEqual(dry_copy.shape, (220_500,))
            self.assertEqual(wet_copy.shape, dry_copy.shape)
            self.assertEqual(report["clean"]["channels"], 2)
            self.assertEqual(report["analysis_sample_rate"], 44_100)
            self.assertEqual((clean.read_bytes(), wet.read_bytes()), before)
            self.assertEqual(parse_effects("drive, delay"), ("drive", "delay"))
            with self.assertRaises(ValueError):
                parse_effects("drive,drive")
        finally:
            clean.unlink(missing_ok=True)
            wet.unlink(missing_ok=True)

    def test_dry_source_split_ignores_wav_suffix(self) -> None:
        self.assertEqual(_split("strat_bridge_pick12.wav"), "valid")
        self.assertEqual(_split("strat_bridge_pick13.wav"), "calibrate")
        self.assertEqual(_split("strat_bridge_pick25.wav"), "calibrate")

    def test_pairwise_targets_rank_the_true_topology(self) -> None:
        topology = ("reverb", "drive", "delay")
        targets, mask = order_targets(topology)
        self.assertEqual(mask, (1.0, 1.0, 1.0))
        logits = tuple(8.0 if target else -8.0 for target in targets)
        self.assertEqual(ranked_topologies(topology, logits)[0][0], topology)

    def test_order_search_decodes_controls_and_finds_reference_chain(self) -> None:
        topology = ("drive", "delay")
        controls = np.asarray((0.4, 0.6, 0.5, 0.3, 0.4, 0.3, 0.0, 0.0, 0.0))
        effects = decode_controls(topology, controls)
        audio = np.zeros(22_050, dtype=np.float32)
        audio[100:200] = 0.1
        wet = render_chain(audio, ChainSpec(effects), 44_100)
        ranked = rank_topologies(audio, wet, topology, controls)
        self.assertEqual(ranked[0][0], topology)
        self.assertLess(gain_aligned_error(wet, wet), 1.0e-12)

    def test_controls_are_normalized_and_masked(self) -> None:
        spec = ChainSpec((Drive(15.0, 0.25, -3.0), Delay(200.0, 0.45, 0.35)))
        controls, mask = control_targets(spec)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in controls))
        self.assertEqual(mask, (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0))

    def test_adjacent_linear_effects_form_an_order_equivalence_class(self) -> None:
        _, adjacent = identifiable_order_targets(("drive", "delay", "reverb"))
        _, separated = identifiable_order_targets(("delay", "drive", "reverb"))
        _, pair = identifiable_order_targets(("delay", "reverb"))
        self.assertEqual(adjacent, (1.0, 1.0, 0.0))
        self.assertEqual(separated, (1.0, 1.0, 1.0))
        self.assertEqual(pair, (0.0, 0.0, 0.0))

    def test_near_bypass_effects_form_a_perceptual_equivalence_class(self) -> None:
        audio = np.linspace(-0.1, 0.1, 22_050, dtype=np.float32)
        spec = ChainSpec((Drive(12.0, 0.5, -3.0), Delay(80.0, 0.4, 0.0)))
        wet = render_chain(audio, spec, 44_100)
        _, mask = perceptual_order_targets(
            audio,
            wet,
            spec,
            lambda value, candidate: render_chain(value, candidate, 44_100),
        )
        self.assertEqual(mask[0], 0.0)

    def test_renderer_is_deterministic_and_order_sensitive(self) -> None:
        rng = np.random.default_rng(7)
        audio = (rng.standard_normal(22_050) * 0.03).astype(np.float32)
        effects = (Drive(12.0, 0.6, -3.0), Delay(80.0, 0.5, 0.3), Reverb(0.3, 0.4, 0.2))
        forward = render_chain(audio, ChainSpec(effects), 44_100)
        repeated = render_chain(audio, ChainSpec(effects), 44_100)
        reverse = render_chain(audio, ChainSpec(tuple(reversed(effects))), 44_100)
        np.testing.assert_array_equal(forward, repeated)
        self.assertGreater(float(np.mean(np.abs(forward - reverse))), 1.0e-5)

    def test_empty_chain_is_bit_exact_and_stereo_safe(self) -> None:
        rng = np.random.default_rng(17)
        audio = (rng.standard_normal((4_096, 2)) * 0.1).astype(np.float32)
        bypass = render_chain(audio, ChainSpec(()), 48_000)
        np.testing.assert_array_equal(bypass, audio)
        rendered = render_chain(
            audio,
            ChainSpec((Delay(80.0, 0.3, 0.2),)),
            48_000,
        )
        self.assertEqual(rendered.shape, audio.shape)
        self.assertEqual(rendered.dtype, np.float32)

    def test_quality_contract_rejects_corruption_and_uses_joint_analysis_gain(self) -> None:
        with self.assertRaises(ValueError):
            checked_audio(np.asarray((0.0, np.nan), dtype=np.float32))
        with self.assertRaises(ValueError):
            checked_sample_rate(0)
        dry = np.asarray((0.25, -0.5), dtype=np.float32)
        wet = np.asarray((0.5, -1.0), dtype=np.float32)
        scaled_dry, scaled_wet = analysis_pair(dry, wet)
        np.testing.assert_array_equal(scaled_dry, dry)
        np.testing.assert_array_equal(scaled_wet, wet)
        self.assertEqual(scaled_wet[0] / scaled_dry[0], 2.0)
        manifest = contract_manifest()
        self.assertFalse(manifest["automatic_normalization"])
        self.assertFalse(manifest["lossy_reencoding"])

    def test_alternate_renderer_preserves_contract_but_changes_domain(self) -> None:
        audio = np.zeros(22_050, dtype=np.float32)
        audio[100] = 0.2
        spec = ChainSpec((Drive(18.0, 0.4, -4.0), Delay(60.0, 0.4, 0.3)))
        reference = render_chain(audio, spec, 44_100, "reference")
        alternate = render_chain(audio, spec, 44_100, "alternate")
        self.assertEqual(reference.shape, alternate.shape)
        self.assertTrue(np.isfinite(alternate).all())
        self.assertGreater(float(np.mean(np.abs(reference - alternate))), 1.0e-6)

    def test_stress_renderer_is_a_distinct_locked_domain(self) -> None:
        audio = np.zeros(22_050, dtype=np.float32)
        audio[100] = 0.2
        spec = ChainSpec((Drive(18.0, 0.4, -4.0), Delay(60.0, 0.4, 0.3)))
        reference = render_chain(audio, spec, 44_100, "reference")
        stress = render_chain(audio, spec, 44_100, "stress")
        self.assertTrue(np.isfinite(stress).all())
        self.assertGreater(float(np.mean(np.abs(reference - stress))), 1.0e-6)
        challenge = render_chain(audio, spec, 44_100, "challenge")
        self.assertTrue(np.isfinite(challenge).all())
        self.assertGreater(float(np.mean(np.abs(stress - challenge))), 1.0e-6)

    def test_pedalboard_renderer_is_deterministic_and_calibrated(self) -> None:
        self.assertGreaterEqual(reverb_room_size(MIN_REVERB_DECAY_SECONDS), 0.0)
        self.assertLessEqual(reverb_room_size(MAX_REVERB_DECAY_SECONDS), 1.0)
        audio = np.zeros(44_100, dtype=np.float32)
        audio[100] = 0.2
        spec = ChainSpec(
            (Drive(12.0, 0.6, -3.0), Delay(80.0, 0.4, 0.3), Reverb(1.2, 0.5, 0.2))
        )
        first = render_pedalboard_chain(audio, spec, 44_100)
        second = render_pedalboard_chain(audio, spec, 44_100)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, audio.shape)
        self.assertTrue(np.isfinite(first).all())
        stereo = np.stack((audio, audio * 0.5), axis=1)
        stereo_rendered = render_pedalboard_chain(stereo, spec, 44_100)
        self.assertEqual(stereo_rendered.shape, stereo.shape)
        self.assertFalse(np.array_equal(stereo_rendered[:, 0], stereo_rendered[:, 1]))

    def test_differentiable_renderer_propagates_control_gradients(self) -> None:
        dry = torch.linspace(-0.1, 0.1, 8_192)
        controls = torch.full((9,), 0.5, requires_grad=True)
        topology = torch.tensor((0, 1, -1))
        output = differentiable_render(dry, controls, topology, renderer=0)
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(controls.grad)
        self.assertTrue(torch.isfinite(controls.grad).all())
        self.assertGreater(float(controls.grad.abs().sum()), 0.0)

    def test_deconvolution_recovers_delay_time(self) -> None:
        generator = torch.Generator().manual_seed(9)
        dry = torch.randn((1, 44_100), generator=generator) * 0.02
        delay = round(0.240 * 44_100)
        wet = dry * 0.7
        wet[:, delay:] += dry[:, :-delay] * 0.3
        normalized = float(deconvolution_delay_hint(dry, wet)[0])
        estimated_ms = 40.0 * 25.0**normalized
        self.assertLess(abs(estimated_ms - 240.0), 2.0)

    def test_deconvolution_recovers_delay_feedback_and_mix(self) -> None:
        generator = np.random.default_rng(11)
        dry = (generator.standard_normal(5 * 44_100) * 0.02).astype(np.float32)
        spec = ChainSpec((Delay(240.0, 0.45, 0.3),))
        for renderer in ("reference", "alternate"):
            wet = render_chain(dry, spec, 44_100, renderer)
            feedback, mix = deconvolution_delay_controls(
                torch.from_numpy(dry)[None], torch.from_numpy(wet)[None]
            )
            self.assertLess(abs(float(feedback[0]) * 0.9 - 0.45), 0.03)
            self.assertLess(abs(float(mix[0]) * 0.7 - 0.3), 0.06)

    def test_reverb_features_are_fixed_and_finite(self) -> None:
        generator = np.random.default_rng(13)
        dry = (generator.standard_normal(5 * 44_100) * 0.02).astype(np.float32)
        wet = render_chain(dry, ChainSpec((Reverb(1.2, 0.4, 0.3),)), 44_100)
        value = reverb_features(
            torch.from_numpy(dry)[None], torch.from_numpy(wet)[None]
        )
        self.assertEqual(value.shape, (1, FEATURES))
        self.assertTrue(torch.isfinite(value).all())

    def test_deconvolution_features_are_finite_for_silence(self) -> None:
        silence = torch.zeros((1, 5 * 44_100))
        feedback, mix = deconvolution_delay_controls(silence, silence)
        self.assertTrue(torch.isfinite(feedback).all())
        self.assertTrue(torch.isfinite(mix).all())
        self.assertTrue(torch.isfinite(reverb_features(silence, silence)).all())
        image, statistics = drive_features(silence, silence)
        self.assertEqual(image.shape[1:], (3, *IMAGE_SIZE))
        self.assertEqual(statistics.shape, (1, STATISTICS))
        self.assertTrue(torch.isfinite(image).all())
        self.assertTrue(torch.isfinite(statistics).all())
        self.assertEqual(delay_features(silence, silence).shape, (1, DELAY_FEATURES))


if __name__ == "__main__":
    unittest.main()
