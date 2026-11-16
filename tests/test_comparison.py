"""Synthetic selection tests and real CPU CNN tests; no research scores."""

from collections import Counter
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from gait import comparison as cmp
from gait import diagnostics as diag
from gait import official_data as data
from gait.cross_validation import Protocol, export_results
from gait.comparison_models import CompactCNN, temporal_pool


def synthetic_records():
    rng = np.random.default_rng(19)
    records = []
    for label in (0, 1):
        for i in range(12):
            subject = f"Ga{'Pt' if label else 'Co'}{i+1:02d}"
            x = rng.normal(size=(24, 18)).astype(np.float32)
            records.append(dict(subject=subject, label=label, study="Ga", recording=subject+"_01",
                                path=subject+"_01.txt", samples=24, signal=x,
                                force_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
    return records


class SyntheticModel(nn.Module):
    """Explicit test double for selection plumbing, not a Mamba implementation."""
    def __init__(self, candidate):
        super().__init__()
        self.head = nn.Linear(18, 2)

    def forward(self, x):
        return self.head(x.mean(1))


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def plan(self):
        cfg = Protocol(seq_len=8, outer_folds=3, inner_fraction=.25,
                       training_seeds=(42, 123), max_epochs=1, bootstrap_samples=3)
        candidates = cmp.sample_candidates(2)
        for trials in candidates.values():
            for i, trial in enumerate(trials):
                trial.update(seq_len=8 if i == 0 else 12,
                             normalization="training" if i == 0 else "recording")
        return cfg, candidates

    def test_search_matched_shared_dimensions_and_reproducible(self):
        first = cmp.sample_candidates(8)
        self.assertEqual(first, cmp.sample_candidates(8))
        cmp.validate_search(Protocol(training_seeds=(42, 123, 2026)), first)
        for a, b in zip(first["mamba"], first["cnn"]):
            for field in ("seq_len", "normalization", "dropout", "pooling", "lr", "weight_decay"):
                self.assertEqual(a[field], b[field])
        for family in first:
            self.assertEqual(len({json.dumps({k:v for k,v in c.items() if k != 'name'}, sort_keys=True)
                                  for c in first[family]}), 8)

    def test_real_cnn_shapes_gradients_and_receptive_field(self):
        for pooling in ("max", "mean_std"):
            model = CompactCNN(width=32, layers=4, kernel=7, pooling=pooling)
            x = torch.randn(2, 200, 18, requires_grad=True)
            y = model(x)
            self.assertEqual(tuple(y.shape), (2, 2))
            self.assertEqual(model.receptive_field, 181)
            y.square().sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        constant = torch.ones(2, 5, 4, requires_grad=True)
        temporal_pool(constant, "mean_std").sum().backward()
        self.assertTrue(torch.isfinite(constant.grad).all())

    def test_curation_excludes_entire_ambiguous_participants(self):
        inventory, signals = [], {}
        for subject, walk, value in [("GaPt01", 1, 1), ("GaPt02", 1, 1), ("GaPt01", 2, 2),
                                     ("JuCo01", 1, 3), ("JuCo01", 2, 3), ("SiPt01", 1, 4)]:
            name = f"{subject}_{walk:02d}.txt"
            inventory.append(dict(path=name, subject=subject, force_sha256=str(value), samples=24))
            signals[name] = np.full((24, 18), value, dtype=np.float32)
        records, audit, provenance = data.curate_inventory(inventory, signals, 8)
        self.assertEqual(provenance["excluded_subjects"], ["GaPt01", "GaPt02"])
        self.assertEqual(Counter(r["status"] for r in audit),
                         dict(participant_excluded_identity_ambiguity=3, duplicate_excluded=1, included=2))
        self.assertEqual(len(records), 2)

    def test_clinical_formats_cross_check_and_no_id_guessing(self):
        txt = b'ID Study Group Subjnum Gender Age Height Weight HoehnYahr\nGaPt01 Ga 1 1 1 70 1.7 70 2.5\nGaCo01 Ga 2 1 2 65 1.6 60 NaN\nJuc010 Ju 2 10 1 66 170 60 NaN\n'
        header = ['ID','Study','Group','Subjnum','Gender','Age','Height','Weight','HoehnYahr']
        rows = [header, ['GaPt01','Ga','PD','1','male','70','1.7','70','2.5'],
                ['GaCo01','Ga','CO','1','female','65','1.6','60','-']]
        html = ('<table>'+''.join('<tr>'+''.join('<td>'+c+'</td>' for c in row)+'</tr>' for row in rows)+'</table>').encode()
        clinical, issues = data.parse_clinical(txt, html)
        self.assertEqual(clinical['GaPt01']['hy'], 2.5)
        self.assertIsNone(clinical['GaCo01']['hy'])
        self.assertNotIn('JuCo10', clinical)
        self.assertEqual(issues[0]['raw_id'], 'Juc010')
        clinical, _ = data.parse_clinical(txt, html.replace(b'2.5', b'3.0'))
        self.assertIsNone(clinical['GaPt01']['hy'])
        self.assertEqual(clinical['GaPt01']['hy_status'], 'clinical_formats_disagree')

    def test_training_and_recording_normalization_isolation(self):
        cfg, candidates = self.plan()
        records = synthetic_records()
        split = cmp.cv.make_splits(records, cfg)[0]
        for c in candidates['mamba']:
            before, _ = cmp.prepare_datasets(records, split, c)
            altered = copy.deepcopy(records)
            for r in altered:
                if r['subject'] in split['test']:
                    r['signal'] *= 1000
            after, _ = cmp.prepare_datasets(altered, split, c)
            torch.testing.assert_close(before['train'][0][0], after['train'][0][0], rtol=0, atol=0)
            torch.testing.assert_close(before['validation'][0][0], after['validation'][0][0], rtol=0, atol=0)

    def test_synthetic_matched_selection_resume_diagnostics_and_export(self):
        cfg, candidates = self.plan()
        records = synthetic_records()
        result = {}
        calls = []
        fit = cmp.fit_trial

        def spy(train, validation, *args):
            calls.append(({r['subject'] for r in train.windows}, {r['subject'] for r in validation.windows}))
            return fit(train, validation, *args)

        with patch.object(cmp.cv, 'require_cuda', return_value=torch.device('cpu')), \
             patch.object(cmp, 'build_candidate', side_effect=SyntheticModel), \
             patch.object(cmp, 'fit_trial', side_effect=spy), redirect_stdout(io.StringIO()):
            cmp.run_comparison(records, [], {}, cfg, candidates, progress=result, max_new_tasks=2)
            self.assertFalse(result['complete'])
            self.assertEqual(len(result['runs']), 2)
            # JSON roundtrip emulates an explicitly exported/resumed session.
            result = json.loads(json.dumps(result))
            cmp.run_comparison(records, [], {}, cfg, candidates, progress=result)
            self.assertEqual(len(result['runs']), 12)
            count = len(calls)
            cmp.run_comparison(records, [], {}, cfg, candidates, progress=result)
            self.assertEqual(count, len(calls))
        self.assertTrue(result['complete'])
        for index, run in enumerate(result['runs']):
            split = result['splits'][run['fold']-1]
            for train, validation in calls[index*2:index*2+2]:
                self.assertEqual(train, set(split['train']))
                self.assertEqual(validation, set(split['validation']))
                self.assertFalse((train | validation) & set(split['test']))
            self.assertEqual(run['selected_candidate'], int(np.argmin([t['inner_metrics']['log_loss'] for t in run['trials']])))
        report = diag.build_diagnostics(result, {})
        self.assertEqual(len(report['participant_errors']), 24*2*2)
        self.assertEqual(len(report['seed_stability']), 24*2)
        self.assertTrue(all(r['hy'] is None for r in report['participant_errors']))
        self.assertTrue(all(r['hy_status'] == 'unmatched' for r in report['participant_errors']))
        self.assertEqual(len(report['paired_comparison']), 6)
        self.assertTrue(all(r['n_participants'] == 24 for r in report['paired_comparison']))
        result['diagnostics'] = report
        with tempfile.TemporaryDirectory() as directory:
            export_results(result, Path(directory)/'test.zip')
        changed = copy.deepcopy(candidates)
        changed['cnn'][0]['lr'] *= 2
        with self.assertRaisesRegex(ValueError, 'Resume rejected'):
            cmp.run_comparison(records, [], {}, cfg, changed, progress=result)
        tampered = copy.deepcopy(result)
        tampered['runs'][0]['selected_candidate'] = 1-tampered['runs'][0]['selected_candidate']
        with self.assertRaisesRegex(ValueError, 'selection'):
            cmp.run_comparison(records, [], {}, cfg, candidates, progress=tampered)
        with patch.object(cmp, 'source_hashes', return_value={'different': 'revision'}):
            with self.assertRaisesRegex(ValueError, 'Resume rejected'):
                cmp.run_comparison(records, [], {}, cfg, candidates, progress=result)
        json.dumps(result, allow_nan=False)

    def test_subgroups_handle_single_class_and_unknown_severity(self):
        rows = [dict(family='mamba', seed=42, subject=f'GaPt{i:02d}', label=1, correct=i%2 == 0,
                     nll=.3, probability=.7 if i%2 == 0 else .3, study='Ga') for i in range(1, 5)]
        table = diag.subgroup_table(rows, ['study'])
        self.assertEqual(table[0]['n'], 4)
        self.assertEqual(table[0]['sensitivity'], .5)
        self.assertIsNone(table[0]['roc_auc'])
        self.assertIsNone(table[0]['specificity'])
        with self.assertRaisesRegex(ValueError, 'Repeated participants'):
            diag.subgroup_table(rows+rows, ['study'])


if __name__ == '__main__':
    unittest.main()