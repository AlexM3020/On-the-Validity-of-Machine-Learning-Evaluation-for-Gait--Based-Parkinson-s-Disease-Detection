"""Offline integrity checks for retained manuscript assets and archives."""

import csv
import hashlib
import json
from pathlib import Path
import unittest
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class PaperArtifactTests(unittest.TestCase):
    def test_anonymous_citation_metadata(self):
        text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn("cff-version: 1.2.0", text)
        self.assertIn("name: Anonymous", text)
        self.assertIn("type: unpublished", text)
        self.assertIn("under review at Transactions on Machine Learning Research", text)
        self.assertNotIn("repository-code:", text)
        self.assertNotIn("email:", text)

    def test_lstm_reports_match_manuscript(self):
        expected = {
            "classification_report_recording.csv": (0.925, 0.9249530956848031, 80.0),
            "classification_report_participant.csv": (0.8307692307692308, 0.8304839177050131, 390.0),
        }
        for name, (accuracy, macro_f1, support) in expected.items():
            with (ROOT / "paper_assets/lstm" / name).open(newline="", encoding="utf-8") as handle:
                rows = {row[""]: row for row in csv.DictReader(handle)}
            self.assertAlmostEqual(float(rows["accuracy"]["precision"]), accuracy)
            self.assertAlmostEqual(float(rows["macro avg"]["f1-score"]), macro_f1)
            self.assertEqual(float(rows["macro avg"]["support"]), support)
        for name in ("accuracy_curve_11.png", "loss_curve_11.png"):
            blob = (ROOT / "paper_assets/lstm" / name).read_bytes()
            self.assertTrue(blob.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(len(blob), 100_000)

    def test_factorial_histories_complete_and_finite(self):
        path = ROOT / "paper_assets/lpgnet/recording_loss_histories.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        histories = payload["histories"]
        self.assertEqual(payload["n_condition_fold_histories"], 40)
        self.assertEqual(len(histories), 40)
        keys = {(h["condition"], h["fold"]) for h in histories}
        self.assertEqual(len(keys), 40)
        for condition in payload["conditions"]:
            self.assertEqual(sum(h["condition"] == condition for h in histories), 10)
        self.assertEqual(len({h["plan_signature"] for h in histories}), 1)
        self.assertEqual(len(payload["source_bundle_sha256"]), 64)
        for history in histories:
            self.assertEqual(len(history["loss"]), len(history["val_loss"]))
            self.assertGreater(len(history["loss"]), 0)
            self.assertTrue(np.isfinite(history["loss"]).all())
            self.assertTrue(np.isfinite(history["val_loss"]).all())
            self.assertTrue((np.asarray(history["loss"]) >= 0).all())
            self.assertTrue((np.asarray(history["val_loss"]) >= 0).all())

    def test_retained_paired_export_has_complete_coverage(self):
        path = ROOT / "paper_assets/lpgnet/paired_comparisons.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({(r["comparison"], r["metric"]) for r in rows}), 40)
        self.assertTrue(all(int(r["participants"]) == 165 for r in rows))

    def test_immutable_archives_and_core_source_hashes(self):
        direct_path = ROOT / "notebooks/pd-mamba-cnn-results/comparison_progress_20261116T204946297832Z.zip"
        with zipfile.ZipFile(direct_path) as archive:
            direct = json.loads(archive.read("results.json"))
        self.assertTrue(direct["complete"])
        self.assertEqual(len(direct["runs"]), 60)
        for name, expected in direct["source_hashes"].items():
            actual = hashlib.sha256((ROOT / "gait" / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)
        factorial_path = ROOT / "notebooks/results_lpgnet_replication/reports.zip"
        with zipfile.ZipFile(factorial_path) as archive:
            plan = json.loads(archive.read("reports/run_plan.json"))
        actual = hashlib.sha256((ROOT / "gait/tf_notebook_support.py").read_bytes()).hexdigest()
        self.assertEqual(actual, plan["source_hash"])
        self.assertEqual(plan["configuration"]["RUN_PROFILE"], "full")
        self.assertEqual(plan["configuration"]["SEED"], 1)
        self.assertEqual(plan["configuration"]["FOLD_SEED"], 0)


if __name__ == "__main__":
    unittest.main()
