"""Offline standard-library tests; never import historical training scripts."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import audit_splits as splits
import inspect_checkpoint as checkpoints


ROOT = Path(__file__).resolve().parents[1]


class IdentityTests(unittest.TestCase):
    def test_multiple_recordings_and_chunks_keep_full_participant(self):
        names = ['GaCo02_01.txt', 'GaCo02_02.txt', 'GaCo02_01_chunk_003.txt',
                 'folder/GaCo02_02_segment_4.txt', r'data\GaCo02_01_0.txt']
        self.assertEqual({splits.subject_id(name) for name in names}, {'GaCo02'})
        self.assertEqual(splits.recording_identity(names[2]), ('GaCo02', 'GaCo02_01.txt'))

    def test_study_and_class_are_part_of_identity(self):
        self.assertEqual(len({splits.subject_id(name) for name in
                              ['GaCo01_01.txt', 'JuCo01_01.txt', 'SiCo01_01.txt',
                               'GaPt01_01.txt']}), 4)

    def test_invalid_filenames_are_explicit_errors(self):
        for value in ['', None, 'Co01_01.txt', 'GaCo001_01.txt', 'GaCo01.txt',
                      'GaCo01_001.txt', 'other_GaCo01_01.txt', 'XaCo01_01.txt',
                      'GaCo01_01.csv', 'GaCo01_01.txt/']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                splits.subject_id(value)


class ManifestAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.provenance = splits.load_manifest()
        cls.result = splits.audit(cls.records)

    def test_official_fixture_digest_and_counts(self):
        self.assertEqual(self.provenance['sha256'], splits.MANIFEST_SHA256)
        self.assertEqual(self.provenance['bytes'], 24573)
        self.assertEqual(self.result['sensor_files'], 306)
        self.assertEqual(self.result['participant_ids'], 165)
        self.assertEqual(self.result['participant_ids_with_multiple_recordings'], 54)

    def test_default_does_not_access_network(self):
        with mock.patch.object(splits, 'urlopen', side_effect=AssertionError('network forbidden')):
            splits.load_manifest()

    def test_explicit_download_uses_only_official_manifest(self):
        response = io.BytesIO(splits.DEFAULT_MANIFEST.read_bytes())
        with mock.patch.object(splits, 'urlopen', return_value=response) as request:
            records, provenance = splits.load_manifest(download=True)
        request.assert_called_once_with(splits.MANIFEST_URL, timeout=30)
        self.assertEqual(records, self.records)
        self.assertTrue(provenance['matches_bundled_official_sha256'])

    def test_malformed_empty_and_duplicate_manifests(self):
        line = 'a' * 64 + ' GaCo01_01.txt\n'
        for content in ['', 'not a checksum', line + line, 'a' * 64 + ' README.md\n']:
            with self.subTest(content=content), self.assertRaises(ValueError):
                splits.parse_manifest(content)

    def test_manifest_binary_marker_and_auxiliary_rows(self):
        content = 'a' * 64 + ' *GaCo01_01.txt\n' + 'b' * 64 + ' format.txt\n'
        records = splits.parse_manifest(content)
        self.assertEqual([row['filename'] for row in records], ['GaCo01_01.txt'])

    def test_replays_match_reported_counterexamples(self):
        expected = {
            'ElMaachi': (92, 9, ['GaCo02', 'GaCo06', 'GaCo09', 'GaCo14']),
            'Naimi_Ga_only': (38, 3, ['GaCo06', 'GaCo07', 'GaCo17']),
        }
        for label, (size, validation, overlap) in expected.items():
            with self.subTest(label=label):
                entry = self.result['control_replays'][label]
                replay = entry['recording_boundary_replay']
                self.assertEqual(entry['input_control_records'], size)
                self.assertEqual(replay['validation_records'], validation)
                self.assertEqual(replay['overlapping_subject_ids'], overlap)
                self.assertEqual(entry['grouped_counterexample']['overlapping_subject_count'], 0)
        el = self.result['control_replays']['ElMaachi']['recording_boundary_replay']
        na = self.result['control_replays']['Naimi_Ga_only']['recording_boundary_replay']
        self.assertIn('GaCo02_01.txt', el['validation_filenames'])
        self.assertIn('GaCo02_02.txt', el['training_filenames'])
        self.assertIn('GaCo06_02.txt', na['validation_filenames'])
        self.assertIn('GaCo06_01.txt', na['training_filenames'])
        self.assertFalse(self.result['seed_is_original_author_seed'])

    def test_grouped_disjointness_and_coverage_for_all_ten_folds(self):
        for prefix in ('', 'Ga'):
            files = [r['filename'] for r in self.records
                     if r['subject_id'][2:4] == 'Co' and r['filename'].startswith(prefix)]
            validation_union = set()
            for fold in range(10):
                train, validation = splits.grouped_control_fold(files, fold=fold)
                self.assertFalse({splits.subject_id(f) for f in train} &
                                 {splits.subject_id(f) for f in validation})
                self.assertFalse(set(train) & set(validation))
                self.assertEqual(set(train) | set(validation), set(files))
                self.assertFalse(validation_union & set(validation))
                validation_union.update(validation)
            self.assertEqual(validation_union, set(files))

    def test_grouping_keeps_chunks_together(self):
        files = [f'GaCo{i:02d}_01_chunk_{chunk}.txt' for i in range(1, 11) for chunk in range(3)]
        train, validation = splits.grouped_control_fold(files)
        self.assertEqual(len(validation), 3)
        self.assertEqual(splits.describe_split(train, validation)['overlapping_subject_count'], 0)

    def test_replay_preserves_published_boundary_array_plus_one(self):
        files = [f'GaCo{i:02d}_01.txt' for i in range(1, 20)]
        train, validation = splits.replay_control_fold(files)
        self.assertEqual(len(validation), 2)  # int((19+1)/10), not int(19/10).
        self.assertEqual(set(train) | set(validation), set(files))
        with self.assertRaisesRegex(ValueError, 'beyond'):
            splits.replay_control_fold(files, fold=9)  # Retain, do not repair, formula bounds.

    def test_invalid_split_inputs(self):
        for files, kwargs in [([], {}), (['GaCo01_01.txt'] * 2, {}),
                              (['GaPt01_01.txt'], {}), (['GaCo01_01.txt'], {'fold': -1}),
                              (['GaCo01_01.txt'], {'folds': 1}), (['GaCo01_01.txt'], {})]:
            with self.subTest(files=files, kwargs=kwargs), self.assertRaises(ValueError):
                splits.replay_control_fold(files, **kwargs)
        with self.assertRaisesRegex(ValueError, 'one subject'):
            splits.grouped_control_fold(['GaCo01_01.txt'])

    def test_cli_json_only(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(splits.main([]), 0)
        self.assertEqual(json.loads(output.getvalue())['sensor_files'], 306)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        # All temporary writes stay inside this artifact, including on Windows.
        self.temporary = tempfile.TemporaryDirectory(prefix='.audit-test-', dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'fixture.zip'

    def make_archive(self, payload, member='archive/data.pkl'):
        with zipfile.ZipFile(self.path, 'w') as archive:
            archive.writestr(member, payload)
        return self.path

    def test_global_reduce_tokens_never_execute(self):
        # This stream would invoke os.system if unpickled. It must remain inert.
        path = self.make_archive(b"cos\nsystem\n(S'echo SHOULD_NOT_RUN'\ntR.")
        with mock.patch('os.system', side_effect=AssertionError('executed')) as system:
            with mock.patch('pickle.load', side_effect=AssertionError('unpickled')):
                with mock.patch('pickle.loads', side_effect=AssertionError('unpickled')):
                    result = checkpoints.inspect_checkpoint(path)
        system.assert_not_called()
        self.assertFalse(result['pickle_executed'])
        self.assertEqual(result['global_tokens'][0]['token'], 'os system')
        self.assertEqual(result['opcode_counts']['REDUCE'], 1)

    def test_stack_global_is_reported_as_unresolved(self):
        path = self.make_archive(b'\x80\x04\x8c\x08builtins\x8c\x05print\x93.')
        result = checkpoints.inspect_checkpoint(path)
        self.assertEqual(result['unresolved_stack_globals'], 1)
        self.assertEqual(result['global_tokens'], [])

    def test_missing_and_non_zip_files(self):
        with self.assertRaisesRegex(ValueError, 'ZIP'):
            checkpoints.inspect_checkpoint(self.path)
        self.path.write_bytes(b'not a ZIP archive')
        with self.assertRaisesRegex(ValueError, 'ZIP'):
            checkpoints.inspect_checkpoint(self.path)

    def test_missing_or_ambiguous_pickle_member(self):
        self.make_archive(b'N.', 'not_pickle.txt')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            checkpoints.inspect_checkpoint(self.path)
        with zipfile.ZipFile(self.path, 'w') as archive:
            archive.writestr('a/data.pkl', b'N.')
            archive.writestr('b/data.pkl', b'N.')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            checkpoints.inspect_checkpoint(self.path)

    def test_malformed_truncated_and_trailing_pickle(self):
        for payload in [b'', b'\x00', b'\x80\x02N', b'N.trailing']:
            with self.subTest(payload=payload):
                self.make_archive(payload)
                with self.assertRaises(ValueError):
                    checkpoints.inspect_checkpoint(self.path)

    def test_size_limits(self):
        self.make_archive(b'\x80\x02N.')
        with self.assertRaisesRegex(ValueError, 'size limit'):
            checkpoints.inspect_checkpoint(self.path, max_pickle_bytes=1)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            checkpoints.inspect_checkpoint(self.path, max_pickle_bytes=0)

    def test_cli_error_is_json_and_nonzero(self):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            self.assertEqual(checkpoints.main([str(self.path)]), 2)
        self.assertEqual(output.getvalue(), '')
        self.assertIn('error', json.loads(error.getvalue()))


class FrozenResultsTests(unittest.TestCase):
    def test_original_result_bytes(self):
        expected = {
            'mamba_report.json': '163243356612ff8735fbc236bb8cdc5a3ab3afd50c442e543939768879d0f138',
            'lstm_report.csv': '1aec44d7e135520c8538ecee8ce96d511ed99b1ac571c4fd5066e4a4603d15bb',
            'gru_report.csv': 'be03f55111063fbc9288f9da932d1dcc2db3a84dd3504c7cdc6848e65b660b87',
        }
        for filename, digest in expected.items():
            with self.subTest(filename=filename):
                self.assertEqual(hashlib.sha256((ROOT / 'results' / filename).read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
