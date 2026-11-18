"""Optional real TensorFlow tests; no surrogate models and no research results."""
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
HAS_TF=importlib.util.find_spec("tensorflow") is not None
if HAS_TF:
    import numpy as np
    from gait import tf_notebook_support as s
    tf,keras=s.tf,s.keras
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass


@unittest.skipUnless(HAS_TF,"Optional TensorFlow/Keras test environment not installed")
class TensorflowSafetyTests(unittest.TestCase):
    def cfg(self,mode="last"):
        return dict(channels=16,kernels=(3,5,7),se_ratio=4,embedding_dim=16,
                    spat_drop=0.,drop=0.,l2=1e-4,fine_tune=mode,lr_mil=1e-4)

    def test_mil_finite_masked_weights_and_padding_invariance_training_and_eval(self):
        for mode in ("frozen","last"):
            with self.subTest(mode=mode):
                cfg=self.cfg(mode)
                keras.utils.set_random_seed(51)
                model=s.build_mil_model(s.build_window_classifier(cfg,32),cfg,32)
                x=np.random.default_rng(9).normal(size=(2,3,32,18)).astype(np.float32)
                mask=np.array([[True,False,False],[True,True,False]])
                changed=x.copy(); changed[~mask]=np.nan
                p=model([x,mask],training=False)
                np.testing.assert_allclose(p,model([changed,mask],training=False),atol=1e-6)
                inspector=keras.Model(model.inputs,model.get_layer("window_attention_remask").output)
                weights=np.asarray(inspector([changed,mask],training=False))
                np.testing.assert_array_equal(weights[~mask],0)
                np.testing.assert_allclose(weights.sum(1),1,atol=1e-6)
                # Extra padded windows do not enter trainable BN or change real outputs.
                padded=np.pad(x,((0,0),(0,4),(0,0),(0,0)),constant_values=999)
                padded_mask=np.pad(mask,((0,0),(0,4)),constant_values=False)
                encoder=model.get_layer('shared_window_encoder')
                # Remove stochastic noise only for the equality assertion; BN stays trainable.
                encoder.encoder.get_layer('input_gaussian_noise').stddev=0.
                original=encoder([x,mask],training=True)
                expanded=encoder([padded,padded_mask],training=True)
                np.testing.assert_allclose(original,expanded[:,:3],atol=1e-5)
                with self.assertRaises(tf.errors.InvalidArgumentError):
                    model([x,np.zeros_like(mask)],training=False)

    def test_safe_serialization_including_optimizer_and_mil_gradient(self):
        for family,cfg in (("LP_ResMIL",self.cfg()),("A",dict(channels=16,kernels=(7,5,3),se_ratio=4,spat_drop=0.,drop=0.,l2=1e-4)),("factorial",{})):
            report=s.smoke_check(family,cfg)
            self.assertTrue(report['safe_model_optimizer_roundtrip'])

    def tiny_data(self):
        rng=np.random.default_rng(37)
        records=[]
        for label in (0,1):
            for i in range(8):
                records.append(dict(file=f'Ga{"Pt" if label else "Co"}{i+1:02d}_01.txt',
                    subject=f'Ga{"Pt" if label else "Co"}{i+1:02d}',label=label,study='Ga',
                    signal=rng.normal(size=(160,18)).astype(np.float32)))
        return records

    def test_weights_and_bounded_mil_batches(self):
        records=self.tiny_data(); features={i:r['signal'] for i,r in enumerate(records)}
        records[1]['subject']=records[0]['subject']
        batch=s.Batches(records,[0,1,2,8],features,'window',win=32,stride=16,batch=5,balanced=True)
        mass={}
        for (i,_),weight in zip(batch.pairs,batch.weights):
            mass[records[i]['subject']]=mass.get(records[i]['subject'],0)+float(weight)
        np.testing.assert_allclose(list(mass.values()),list(mass.values())[0])
        batch=s.Batches(records,[0,8],{0:features[0][:90],8:features[8]},'mil',win=32,stride=16,batch=2)
        (values,mask),_,_=batch[0]
        self.assertEqual(values.shape[:2],mask.shape)
        self.assertFalse(mask[0].all())

    def test_epoch_interrupt_resume_optimizer_and_counters(self):
        records=self.tiny_data(); feats={i:r['signal'] for i,r in enumerate(records)}
        train=s.Batches(records,[0,1,8,9],feats,'window',win=32,stride=16,batch=8,shuffle=True)
        val=s.Batches(records,[2,10],feats,'window',win=32,stride=16,batch=8)
        def model():
            m=s.build_lpgnet(channels=4,kernels=(3,3,3),spat_drop=0.,drop=0.,noise=0.)
            s.compile_model(m,1e-3,0.)
            return m
        with tempfile.TemporaryDirectory() as d:
            original=s.atomic_json
            def interrupt(path,payload):
                original(path,payload)
                if Path(path).name=='state.json' and payload['next_epoch']==1:
                    raise RuntimeError('simulated disconnection')
            with patch.object(s,'atomic_json',side_effect=interrupt),self.assertRaisesRegex(RuntimeError,'simulated'):
                s.safe_fit(model(),train,val,Path(d),3,10,2,1e-7)
            saved=json.loads((Path(d)/'state.json').read_text())
            self.assertEqual(saved['next_epoch'],1)
            m,h,best=s.safe_fit(model(),train,val,Path(d),3,10,2,1e-7)
            self.assertEqual(len(h['loss']),3)
            state=json.loads((Path(d)/'state.json').read_text())
            latest=keras.models.load_model(Path(d)/state['last'],safe_mode=True)
            self.assertEqual(int(latest.optimizer.iterations),3*len(train))
            self.assertTrue(state['complete'])
            with patch.object(keras.Model,'train_on_batch',side_effect=AssertionError('should not train')):
                _,history,_=s.safe_fit(model(),train,val,Path(d),3,10,2,1e-7)
            self.assertEqual(history,h)

    def test_atomic_plan_and_task_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            store=s.RunStore(Path(d)/'run',dict(seed=42))
            store.save(['outer',0],dict(rows=[1,2]))
            self.assertEqual(store.load(['outer',0]),dict(rows=[1,2]))
            with self.assertRaisesRegex(ValueError,'plan mismatch'):
                s.RunStore(Path(d)/'run',dict(seed=1))
            path=store.folder(['outer',0])/'complete.json'
            obj=json.loads(path.read_text()); obj['payload']['rows']=[9]; path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,'Corrupt'):
                store.load(['outer',0])

    def test_checkpoint_bundle_restore_and_unsafe_path_rejection(self):
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run'
            store=s.RunStore(root,dict(seed=42)); store.save(['a'],dict(done=True))
            bundle=s.export_checkpoint_bundle(root)
            restored=s.restore_checkpoint_bundle(bundle,Path(d)/'restored')
            self.assertEqual(s.RunStore(restored,dict(seed=42)).load(['a']),dict(done=True))
            with self.assertRaisesRegex(ValueError,'empty'):
                s.restore_checkpoint_bundle(bundle,root)
            bad=Path(d)/'bad.zip'
            with zipfile.ZipFile(bad,'w') as archive:
                archive.writestr('run_plan.json','{}'); archive.writestr('../escape.txt','bad')
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                s.restore_checkpoint_bundle(bad,Path(d)/'bad-target')

    def test_nested_core_lp_isolation_and_real_single_task_resume(self):
        records=self.tiny_data()
        with tempfile.TemporaryDirectory() as d:
            ns=dict(records=records,CHECKPOINT_DIR=Path(d)/'run',SEED=1,FOLD_SEED=0,SPLIT_SEED=9137,SEARCH_SEED=3,
                N_FOLDS=2,N_INNER=2,VAL_FRAC=.5,LP_ORDER=3,WIN=32,STRIDE=16,BATCH_WIN=16,BATCH_REC=4,
                LR_WIN=1e-3,LR_REC=1e-3,LABEL_SMOOTH=.1,MAX_EPOCHS_WIN=1,MAX_EPOCHS_REC=1,
                PATIENCE=2,RLR_PATIENCE=1,MIN_LR=1e-7,TRAIN_SEEDS=[42],N_TRIALS=1,THRESHOLD=.5,
                SPACE_A=dict(channels=[4],kernels=[(3,3,3)],se_ratio=[4],spat_drop=[0.],drop=[0.],l2=[1e-4]))
            runner=s.NotebookExperiment(ns,'A'); calls=[]; original=runner.features
            def spy(train,evaluate):
                calls.append((set(train),set(evaluate)))
                return original(train,evaluate)
            with patch.object(runner,'features',side_effect=spy),redirect_stdout(io.StringIO()):
                output=runner.nested(max_tasks=1)
            self.assertFalse(output['complete']); self.assertEqual(len(output['selections']),1)
            held=set(runner.folds[0]['test'])
            self.assertTrue(all(not train&held for train,_ in calls))
            # All fitting phases exclude evaluation; final refit still excludes outer test.
            self.assertTrue(all(not a&b for a,b in calls))
            with patch.object(runner,'train',side_effect=AssertionError('completed task trained twice')),redirect_stdout(io.StringIO()):
                again=runner.nested(max_tasks=0)
            self.assertEqual(again['all_oof_rows'],output['all_oof_rows'])

    def test_factorial_all_conditions_real_fit_resume_and_reporting(self):
        # Import only function definitions from the notebook's metrics cell, never
        # execute setup/download/training cells during this unit test.
        import ast
        from sklearn.metrics import (roc_auc_score,accuracy_score,balanced_accuracy_score,
                                    f1_score,confusion_matrix,log_loss)
        import pandas as pd
        records=self.tiny_data()
        metric_ns=dict(np=np,pd=pd,roc_auc_score=roc_auc_score,accuracy_score=accuracy_score,
            balanced_accuracy_score=balanced_accuracy_score,f1_score=f1_score,
            confusion_matrix=confusion_matrix,log_loss=log_loss,THRESHOLD=.5,N_BOOT=2)
        path=Path(__file__).resolve().parents[1]/'notebooks'/'LPGNet_Exact_Replication_and_Factorial_Leakage_Study_RestartSafe.ipynb'
        document=json.loads(path.read_text(encoding='utf-8'))
        text=next(''.join(c['source']) for c in document['cells'] if 'def metric_dict' in ''.join(c['source']))
        definitions=ast.Module(body=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)],type_ignores=[])
        exec(compile(definitions,'<metric function definitions>','exec'),metric_ns)
        metric_ns['clinical']={r['subject']:dict(study='Ga',diagnosis='PD' if r['label'] else 'Control',
            hy=2. if r['label'] else np.nan,hy_label='2' if r['label'] else 'Not applicable',hy_status='verified',age=65,gender=1) for r in records}
        with tempfile.TemporaryDirectory() as d:
            ns=dict(metric_ns,records=records,CHECKPOINT_DIR=Path(d)/'run',SEED=1,FOLD_SEED=0,
                N_FOLDS=2,N_INNER=2,VAL_FRAC=.5,LP_ORDER=3,WIN=32,STRIDE=16,BATCH_WIN=32,BATCH_REC=8,
                LR_WIN=1e-3,LR_REC=1e-3,LABEL_SMOOTH=.1,MAX_EPOCHS_WIN=1,MAX_EPOCHS_REC=1,
                PATIENCE=2,RLR_PATIENCE=1,MIN_LR=1e-7,THRESHOLD=.5,N_BOOT=2)
            runner=s.NotebookExperiment(ns,'factorial')
            with redirect_stdout(io.StringIO()): result=runner.factorial()
            self.assertTrue(result['complete']); self.assertEqual(len(result['fold_rows']),8)
            ns.update(s.unpack_result(ns,result,'factorial'))
            ns.update(s.summary_tables(ns,'factorial'))
            ns.update(s.error_reports(ns,'factorial'))
            paired=s.paired_tables(ns,'factorial')
            self.assertEqual(len(paired),40)
            with patch.object(runner,'train',side_effect=AssertionError('must not retrain')):
                replay=runner.factorial()
            self.assertEqual(result,replay)


if __name__=='__main__': unittest.main()