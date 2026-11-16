"""CPU synthetic protocol tests; never substitute these scores for CUDA results."""

from contextlib import redirect_stdout
from dataclasses import replace
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch import nn

from gait import cross_validation as cv


class SyntheticClassifier(nn.Module):
    """Trainable linear test double, NOT a Mamba backend."""
    def __init__(self, **kwargs):
        super().__init__()
        self.head = nn.Linear(kwargs["input_dim"], 2)

    def forward(self, x):
        return self.head(x.mean(1))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, seed=1, n=12):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        force = np.random.default_rng(seed).normal(size=(n, 18))
        force[:, 0] = 7
        np.savetxt(path, np.column_stack([np.arange(n)*0.01, force]))
        return path

    def data(self):
        for label in ("Co", "Pt"):
            for s in range(1, 13):
                self.write(f"Ga{label}{s:02d}_01.txt", s + (100 if label == "Pt" else 0))
        return cv.load_originals(self.root, seq_len=4)

    def test_exact_duplicates_removed_without_changing_support(self):
        first = self.write("GaCo01_01.txt")
        (self.root / "GaCo01_01 (1).txt").write_bytes(first.read_bytes())
        # Numerically identical force signal with a different time origin/text encoding.
        matrix = np.loadtxt(first)
        matrix[:, 0] += 10
        np.savetxt(self.root / "GaCo01_02.txt", matrix, fmt="%.17g")
        records, audit = cv.load_originals(self.root, 4)
        self.assertEqual(len(records), 1)
        self.assertEqual(sum(r["status"] == "duplicate_excluded" for r in audit), 2)
        self.assertEqual(len(cv.make_windows(records, 4)), 3)

    def test_cross_participant_duplicate_and_conflicting_filename_rejected(self):
        first = self.write("GaCo01_01.txt")
        other = self.root / "GaPt01_01.txt"
        other.write_bytes(first.read_bytes())
        with self.assertRaisesRegex(ValueError, "different participants"):
            cv.load_originals(self.root, 4)
        other.unlink()
        self.write("sub/GaCo01_01.txt", seed=9)
        with self.assertRaisesRegex(ValueError, "Conflicting contents"):
            cv.load_originals(self.root, 4)

    def test_prepared_nonfinite_and_irregular_inputs_rejected(self):
        path = self.write("GaCo01_01_chunk_1.txt")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            cv.load_originals(self.root, 4)
        path.unlink()
        path = self.write("GaCo01_01.txt")
        matrix = np.loadtxt(path)
        matrix[2, 1] = np.nan
        np.savetxt(path, matrix)
        with self.assertRaisesRegex(ValueError, "finite"):
            cv.load_originals(self.root, 4)
        path = self.write("GaCo01_01.txt")
        matrix = np.loadtxt(path)
        matrix[2, 0] = matrix[1, 0]
        np.savetxt(path, matrix)
        with self.assertRaisesRegex(ValueError, "time axis"):
            cv.load_originals(self.root, 4)

    def test_short_exclusion_nonoverlap_and_tail_policy(self):
        self.write("GaCo01_01.txt", n=3)
        self.write("GaPt01_01.txt", n=11, seed=3)
        records, audit = cv.load_originals(self.root, 4)
        self.assertEqual(len(records), 1)
        self.assertIn("short_excluded", [r["status"] for r in audit])
        windows = cv.make_windows(records, 4)
        self.assertEqual([(w["start"], w["stop"]) for w in windows], [(0, 4), (4, 8)])
        self.assertEqual(len(cv.make_windows(records, 4, 0.5)), 4)

    def test_tenfold_split_disjoint_exhaustive_and_reproducible(self):
        records, _ = self.data()
        cfg = cv.Protocol(seq_len=4)
        splits = cv.make_splits(records, cfg)
        self.assertEqual(splits, cv.make_splits(list(reversed(records)), cfg))
        tests = []
        for split in splits:
            a, b, c = (set(split[k]) for k in ("train", "validation", "test"))
            self.assertFalse(a & b or a & c or b & c)
            self.assertEqual(len(a | b | c), 24)
            tests.extend(c)
        self.assertEqual(len(tests), len(set(tests)))
        self.assertEqual(len(tests), 24)

    def test_normalization_and_weights_use_only_inner_training(self):
        records, _ = self.data()
        subjects = {"GaCo01", "GaCo02", "GaPt01"}
        mean, std = cv.fit_normalizer(records, subjects)
        train = np.concatenate([r["signal"] for r in records if r["subject"] in subjects])
        np.testing.assert_allclose(mean, train.astype(float).mean(0), atol=1e-6)
        expected_std = train.astype(float).std(0)
        expected_std[0] = 1
        np.testing.assert_allclose(std, expected_std, atol=1e-6)
        windows = cv.make_windows(records, 4)
        ds = cv.WindowDataset(records, windows, subjects, mean, std)
        torch.testing.assert_close(cv.class_weights(ds), torch.tensor([0.75, 1.5]))
        for r in records:
            if r["subject"] not in subjects:
                r["signal"][:] = 1e6
        mean2, std2 = cv.fit_normalizer(records, subjects)
        np.testing.assert_array_equal(mean, mean2)
        np.testing.assert_array_equal(std, std2)
        self.assertEqual(tuple(ds[0][0].shape), (4, 18))
        self.assertTrue(torch.equal(ds[0][0][:, 0], torch.zeros(4)))

    def test_equal_recording_aggregation_and_fixed_threshold(self):
        rows = [dict(subject="A", recording="long", label=0, probability=0.875)]*9
        rows += [dict(subject="A", recording="short", label=0, probability=0.125)]
        p = cv.aggregate_predictions(rows)[0]
        self.assertAlmostEqual(p["probability"], 0.5)
        self.assertEqual(p["prediction"], 1)  # predefined tie rule
        self.assertEqual((p["recordings"], p["windows"]), (2, 10))

    def test_bootstrap_participant_unit_reproducibility_and_bounds(self):
        rows = [dict(subject=str(i), label=i % 2, probability=0.8 if i % 2 else 0.2)
                for i in range(20)]
        first = cv.confidence_intervals(rows, 20, 8)
        self.assertEqual(first, cv.confidence_intervals(rows, 20, 8))
        for score in first.values():
            self.assertEqual(score, dict(estimate=1.0, lower=1.0, upper=1.0))
        with self.assertRaisesRegex(ValueError, "one prediction"):
            cv.confidence_intervals(rows+rows, 20)

    def test_invalid_configs_fail_early(self):
        for change in (dict(overlap=1), dict(outer_folds=1), dict(training_seeds=(42, 42)),
                       dict(seq_len=0), dict(max_epochs=0), dict(min_lr=float("nan"))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                cv.validate_protocol(replace(cv.Protocol(), **change), cv.DEFAULT_CANDIDATES)
        with self.assertRaises(ValueError):
            cv.validate_protocol(cv.Protocol(), [dict(name="bad", lr=0, weight_decay=0)])

    def test_complete_synthetic_selection_two_seeds_and_exports(self):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old)
        records, audit = self.data()
        cfg = cv.Protocol(seq_len=4, outer_folds=3, inner_fraction=0.25,
                          training_seeds=(11, 29), max_epochs=2, bootstrap_samples=5)
        candidates = cv.DEFAULT_CANDIDATES[:2]
        observed = []
        original_fit = cv.fit_candidate

        def spy(train, validation, candidate, config, seed, device, **kwargs):
            observed.append(({w["subject"] for w in train.windows},
                             {w["subject"] for w in validation.windows}))
            return original_fit(train, validation, candidate, config, seed, device, **kwargs)

        before = list(self.root.iterdir())
        progress = {}
        with patch.object(cv, "require_cuda", return_value=torch.device("cpu")), \
                patch.object(cv, "build_model", side_effect=SyntheticClassifier), \
                patch.object(cv, "fit_candidate", side_effect=spy), redirect_stdout(io.StringIO()):
            result = cv.run_nested_cv(records, audit, cfg, candidates, progress=progress, verbose=False)
        self.assertIs(result, progress)
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["runs"]), 6)
        self.assertEqual(len(observed), 12)
        self.assertEqual(before, list(self.root.iterdir()))  # No automatic exports.
        for index, run in enumerate(result["runs"]):
            split = result["splits"][run["fold"]-1]
            for train, validation in observed[2*index:2*index+2]:
                self.assertEqual(train, set(split["train"]))
                self.assertEqual(validation, set(split["validation"]))
                self.assertFalse((train | validation) & set(split["test"]))
            losses = [t["inner_metrics"]["log_loss"] for t in run["trials"]]
            self.assertEqual(run["selected_candidate"], int(np.argmin(losses)))
            self.assertTrue(all(len(t["history"]) == 2 for t in run["trials"]))
        for seed in cfg.training_seeds:
            summary = result["summaries"][str(seed)]
            self.assertEqual(summary["participant_metrics"]["n"], 24)
        json.dumps(result, allow_nan=False)
        destination = self.root / "explicit.zip"
        cv.export_results(result, destination)
        with zipfile.ZipFile(destination) as z:
            self.assertTrue(json.loads(z.read("results.json"))["complete"])
        with self.assertRaises(FileExistsError):
            cv.export_results(result, destination)

    def test_test_data_cannot_change_inner_checkpoint(self):
        """Perturb outer-test signals; fixed inner data must train identically."""
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old)
        records, _ = self.data()
        cfg = cv.Protocol(seq_len=4, outer_folds=3, max_epochs=1)
        split = cv.make_splits(records, cfg)[0]

        def fit(data):
            mean, std = cv.fit_normalizer(data, set(split["train"]))
            windows = cv.make_windows(data, 4)
            training = cv.WindowDataset(data, windows, set(split["train"]), mean, std)
            validation = cv.WindowDataset(data, windows, set(split["validation"]), mean, std)
            with patch.object(cv, "build_model", side_effect=SyntheticClassifier):
                return cv.fit_candidate(training, validation, cv.DEFAULT_CANDIDATES[0], cfg,
                                        7, torch.device("cpu"), verbose=False)

        state1, trial1 = fit(records)
        altered = copy.deepcopy(records)
        for r in altered:
            if r["subject"] in set(split["test"]):
                r["signal"][:] = 999
        state2, trial2 = fit(altered)
        for name in state1:
            torch.testing.assert_close(state1[name], state2[name], rtol=0, atol=0)
        self.assertEqual(trial1["inner_metrics"], trial2["inner_metrics"])

    def test_no_cpu_backend_for_real_research(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CPU substitute"):
                cv.require_cuda()


if __name__ == "__main__":
    unittest.main()