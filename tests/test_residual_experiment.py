"""Real CPU residual/CNN tests on synthetic data; no scientific performance claims."""

from contextlib import redirect_stdout
from dataclasses import replace
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import lfilter
import torch

from gait import residual_experiment as rexp
from gait.cross_validation import export_results, make_splits


def synthetic_records():
    rng = np.random.default_rng(81)
    records = []
    for label in (0, 1):
        for subject in range(1, 9):
            identity = f"Ga{'Pt' if label else 'Co'}{subject:02d}"
            x = lfilter([1.], [1., -.7], rng.normal(size=(256,18)), axis=0).astype(np.float32)
            x[:,0] = 0
            records.append(dict(subject=identity,label=label,recording=identity+"_01",
                                path=identity+"_01.txt",signal=x))
    return records


class ResidualTests(unittest.TestCase):
    def test_epoch_recovery_after_destroyed_runtime(self):
        from gait.durable_storage import DurableMirror,DirectoryBackend
        from gait.torch_recovery import TorchRecovery
        import shutil
        records50=rexp.downsample(synthetic_records())
        subjects={r['subject'] for r in records50}
        ref=rexp.fit_reference(records50,subjects)
        records=rexp.transform(records50,ref,(11,))
        train=[r for r in records if r['subject'].endswith(('01','02','03','04'))]
        val=[r for r in records if r['subject'].endswith(('05','06'))]
        cfg=rexp.ResidualProtocol(window_samples=32,encoder_epochs=2,head_epochs=2,batch_size=16)
        candidate=rexp.candidate_search(1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); local=root/'runtime'; remote=root/'cloud'
            mirror=DurableMirror(local,DirectoryBackend(remote)); mirror.verify()
            recovery=TorchRecovery(local/'trial',mirror)
            original=recovery.save
            def disconnect(payload):
                original(payload)
                if payload['phase']=='head' and payload['epoch']==1: raise RuntimeError('simulated VM reset')
            with patch.object(recovery,'save',side_effect=disconnect),self.assertRaisesRegex(RuntimeError,'VM reset'):
                rexp.train_trial(train,val,candidate,cfg,(11,),42,torch.device('cpu'),recovery=recovery)
            shutil.rmtree(local)
            fresh=root/'new-runtime'; recovered=DurableMirror(fresh,DirectoryBackend(remote)); recovered.verify(); recovered.restore()
            state,trial=rexp.train_trial(train,val,candidate,cfg,(11,),42,torch.device('cpu'),recovery=TorchRecovery(fresh/'trial',recovered))
            self.assertEqual(len(trial['history']),2); self.assertEqual(len(trial['head_history']),2)
            final=TorchRecovery(fresh/'trial',recovered).load()
            self.assertEqual(final['phase'],'complete')
            self.assertEqual(final['trial'],trial)
            for name in state: torch.testing.assert_close(state[name],final['best_state'][name])

    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_reference_training_control_only_and_no_cross_record_histories(self):
        records = rexp.downsample(synthetic_records())
        training = {r["subject"] for r in records[:4]} | {records[-1]["subject"]}
        reference = rexp.fit_reference(records,training)
        self.assertEqual(len(reference['training_control_subjects']),4)
        altered = copy.deepcopy(records)
        for r in altered:
            if r['subject'] not in training or r['label'] == 1:
                r['signal'] *= 1e5
        self.assertEqual(reference,rexp.fit_reference(altered,training))
        # Reversing record iteration cannot create inter-record prediction histories.
        reversed_ref = rexp.fit_reference(list(reversed(records)),training)
        for order in reference['coefficients']:
            np.testing.assert_allclose(reference['coefficients'][order],reversed_ref['coefficients'][order],atol=1e-12)
        # Duplicating a control recording within an identity leaves equal-record means unchanged.
        duplicated = records + [{**records[0], 'recording':records[0]['recording']+'_copy'}]
        ref2 = rexp.fit_reference(duplicated,training)
        np.testing.assert_allclose(reference['scale'],ref2['scale'],atol=1e-12)
        self.assertEqual(reference['constant_channels'],[0])

    def test_filter_equation_order_layout_boundary_and_linear_amplitude(self):
        records50 = rexp.downsample(synthetic_records())
        reference = rexp.fit_reference(records50,{r['subject'] for r in records50 if not r['label']})
        single = rexp.transform(records50,reference,(11,))
        multi = rexp.transform(records50,reference,(5,11,21))
        self.assertEqual(single[0]['signal'].shape,(128-rexp.WARMUP,18))
        bank = multi[0]['signal'].reshape(-1,18,3)
        np.testing.assert_array_equal(single[0]['signal'],bank[:,:,1])
        x = records50[0]['signal'][:,1]/reference['scale'][1]
        a = np.array(reference['coefficients']['11'][1])
        expected = x-lfilter(np.r_[0.,-a[1:]],[1.],x)
        np.testing.assert_allclose(single[0]['signal'][:,1],expected[rexp.WARMUP:],atol=1e-6)
        changed = [{**records50[0], 'signal':2*records50[0]['signal']}]
        np.testing.assert_allclose(rexp.transform(changed,reference,(11,))[0]['signal'],2*single[0]['signal'])
        # Constant sensor yields finite outputs; no per-window energy normalization.
        self.assertTrue(np.isfinite(multi[0]['signal']).all())

    def test_models_real_forward_backward_variable_lengths(self):
        counts = []
        for k in (1,3):
            model = rexp.ResidualStatsCNN(k)
            counts.append(sum(p.numel() for p in model.parameters()))
            for length in (32,100,1000):
                model.zero_grad(set_to_none=True)
                x = torch.randn(2,length,18*k,requires_grad=True)
                logits = model(x)
                self.assertEqual(tuple(logits.shape),(2,2))
                logits.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(counts[1]-counts[0],72)  # Only sensor-fusion inputs differ.
        self.assertLess(max(counts),10000)

    def test_search_and_invalid_config(self):
        self.assertEqual(rexp.candidate_search(),rexp.candidate_search())
        for change in (dict(window_samples=20),dict(training_seeds=(42,42)),dict(ridge=0),dict(encoder_epochs=0)):
            with self.assertRaises(ValueError):
                rexp.validate(replace(rexp.ResidualProtocol(),**change),rexp.candidate_search())

    def test_full_real_cpu_two_stage_resume_oof_reports_and_export(self):
        records = synthetic_records()
        cfg = rexp.ResidualProtocol(outer_folds=2,inner_fraction=.5,training_seeds=(42,123),
                                    window_samples=32,encoder_epochs=1,head_epochs=1,
                                    batch_size=16,bootstrap_samples=2)
        candidates = rexp.candidate_search(2)
        result = {}
        calls = []
        real_train = rexp.train_trial

        def spy(training,validation,*args,**kwargs):
            calls.append(({r['subject'] for r in training},{r['subject'] for r in validation}))
            return real_train(training,validation,*args,**kwargs)

        with patch.object(rexp,'train_trial',side_effect=spy),redirect_stdout(io.StringIO()):
            rexp.run(records,[],{},cfg,candidates,progress=result,max_new_tasks=1,device='cpu')
            self.assertFalse(result['complete'])
            result = json.loads(json.dumps(result))
            rexp.run(records,[],{},cfg,candidates,progress=result,max_new_tasks=None,device='cpu')
            self.assertTrue(result['complete'])
            n = len(calls)
            rexp.run(records,[],{},cfg,candidates,progress=result,device='cpu')
            self.assertEqual(len(calls),n)
        self.assertEqual(len(result['runs']),8)
        for index,run in enumerate(result['runs']):
            split = result['splits'][run['fold']-1]
            self.assertTrue(set(run['reference']['training_control_subjects']) <= set(split['train']))
            for training,validation in calls[2*index:2*index+2]:
                self.assertEqual(training,set(split['train']))
                self.assertEqual(validation,set(split['validation']))
                self.assertFalse((training|validation) & set(split['test']))
            self.assertEqual(run['selected_candidate'],int(np.argmin([t['inner_metrics']['log_loss'] for t in run['trials']])))
            for t in run['trials']:
                self.assertEqual(len(t['history']),1)
                self.assertEqual(len(t['head_history']),1)
                self.assertIn(t['best_head_epoch'],(0,1))
        report = rexp.reports(result,{})
        self.assertEqual(len(report['participant_errors']),16*2*2)
        self.assertEqual(len(report['seed_stability']),16*2)
        result['diagnostics'] = report
        json.dumps(result,allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(export_results(result,Path(directory)/'test.zip').is_file())
        changed = copy.deepcopy(candidates)
        changed[0]['lr'] *= 2
        with self.assertRaisesRegex(ValueError,'Resume rejected'):
            rexp.run(records,[],{},cfg,changed,progress=result,device='cpu')
        altered = copy.deepcopy(result)
        altered['runs'][0]['test_participants'][0]['probability'] = .111111
        with self.assertRaisesRegex(ValueError,'predictions and metrics'):
            rexp.run(records,[],{},cfg,candidates,progress=altered,device='cpu')

    def test_full_recording_embedding_aggregation_equal_record_weights(self):
        records50 = rexp.downsample(synthetic_records())
        ref = rexp.fit_reference(records50,{r['subject'] for r in records50})
        transformed = rexp.transform(records50,ref,(11,))
        first = transformed[0]
        second = {**first,'recording':'second','signal':first['signal'][:50]*2}
        model = rexp.ResidualStatsCNN(1).eval()
        emb,meta = rexp.participant_embeddings(model,[first,second],{first['subject']},torch.device('cpu'))
        with torch.no_grad():
            expected = (model.encode(torch.from_numpy(first['signal'])[None])+model.encode(torch.from_numpy(second['signal'])[None]))/2
        torch.testing.assert_close(emb,expected)
        self.assertEqual(meta[0]['recordings'],2)


if __name__ == '__main__':
    unittest.main()