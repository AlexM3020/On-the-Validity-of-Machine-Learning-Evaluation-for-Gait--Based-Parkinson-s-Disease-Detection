"""Synthetic end-to-end tests, not paper experiments."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from gait.experiment import main


class ExperimentTests(unittest.TestCase):
    def test_train_evaluate_and_provenance_guards(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            rng = np.random.default_rng(42)
            for label in ("Co", "Pt"):
                for subject in range(1, 11):
                    for recording in (1, 2):
                        np.savetxt(data / f"Ga{label}{subject:02d}_{recording:02d}.txt",
                                   rng.normal(size=(9, 19)))
            for name in ("lstm", "gru"):
                output = root / name
                with redirect_stdout(io.StringIO()):
                    main(["--data-dir", str(data), "--output-dir", str(output),
                          "--model", name, "--device", "cpu", "--epochs", "2",
                          "--seq-len", "4", "--hidden-dim", "4", "--batch-size", "7",
                          "--test-fraction", "0.2"])
                for filename in ("best_model.pt", "history.json", "loss_curve.png",
                                 "validation_report.json", "test_report.json", "split_manifest.json"):
                    self.assertTrue((output / filename).is_file(), filename)
                manifest = json.loads((output / "split_manifest.json").read_text())
                groups = [set(part["subjects"]) for part in manifest.values()]
                self.assertTrue(all(left.isdisjoint(right) for i, left in enumerate(groups)
                                    for right in groups[i + 1:]))
                original = json.loads((output / "test_predictions.json").read_text())
                evaluation = root / f"{name}-evaluation"
                with redirect_stdout(io.StringIO()):
                    main(["--data-dir", str(data), "--output-dir", str(evaluation),
                          "--checkpoint", str(output / "best_model.pt"), "--partition", "test",
                          "--device", "cpu"])
                self.assertEqual(original, json.loads((evaluation / "test_predictions.json").read_text()))
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(["--data-dir", str(data), "--output-dir", str(output)])
            changed = data / "GaCo01_01.txt"
            changed.write_text(changed.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Dataset content or split"):
                main(["--data-dir", str(data), "--output-dir", str(root / "changed"),
                      "--checkpoint", str(root / "gru" / "best_model.pt"), "--device", "cpu"])


if __name__ == "__main__":
    unittest.main()