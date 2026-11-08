"""Synthetic runner tests: no real Mamba CUDA execution or research results.

Only tests patch device resolution and model construction to use a trainable
CPU linear test double. Production has no CPU execution bypass.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from gait.experiment import main, parser, resolve_device


class FakeClassifier(nn.Module):
    """Synthetic masked-mean/linear classifier; not a Mamba implementation."""

    def __init__(self, input_dim, hidden_dim, num_layers, d_state, d_conv, expand, dropout):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, values, lengths):
        lengths = lengths.to(values.device)
        valid = torch.arange(values.shape[1], device=values.device)[None, :] < lengths[:, None]
        pooled = values.masked_fill(~valid[..., None], 0).sum(1) / lengths[:, None]
        return self.linear(pooled)


class ExperimentTests(unittest.TestCase):
    def test_synthetic_cpu_double_train_evaluate_and_provenance_guards(self):
        """Exercise runner plumbing only; all temporary scores are synthetic."""
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
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
            output = root / "synthetic-test-only"
            with patch("gait.experiment.resolve_device", return_value="cpu") as device_mock, \
                    patch("gait.experiment.build_model", side_effect=FakeClassifier) as model_mock:
                with redirect_stdout(io.StringIO()):
                    main(["--data-dir", str(data), "--output-dir", str(output),
                          "--epochs", "2",
                          "--seq-len", "4", "--hidden-dim", "4", "--batch-size", "7",
                          "--test-fraction", "0.2"])
                device_mock.assert_called_once_with("auto")
                model_mock.assert_called_once_with(18, 4, 2, 32, 32, 2, 0.0)
                for filename in ("best_model.pt", "history.json", "loss_curve.png",
                                 "validation_report.json", "test_report.json", "split_manifest.json",
                                 "configuration.json", "validation_predictions.json", "test_predictions.json"):
                    self.assertTrue((output / filename).is_file(), filename)
                configuration = json.loads((output / "configuration.json").read_text())
                self.assertEqual(configuration["model"], "mamba")
                self.assertEqual(configuration["device"], "cpu")  # Patched test only.
                history = json.loads((output / "history.json").read_text())
                self.assertEqual([row["epoch"] for row in history], [1, 2])
                self.assertTrue(all(np.isfinite(row["train_loss"]) and np.isfinite(row["validation_loss"])
                                    for row in history))
                manifest = json.loads((output / "split_manifest.json").read_text())
                groups = [set(part["subjects"]) for part in manifest.values()]
                self.assertTrue(all(left.isdisjoint(right) for i, left in enumerate(groups)
                                    for right in groups[i + 1:]))
                original = json.loads((output / "test_predictions.json").read_text())
                evaluation = root / "synthetic-evaluation-only"
                with redirect_stdout(io.StringIO()):
                    main(["--data-dir", str(data), "--output-dir", str(evaluation),
                          "--checkpoint", str(output / "best_model.pt"), "--partition", "test"])
                self.assertEqual(original, json.loads((evaluation / "test_predictions.json").read_text()))
                self.assertFalse((evaluation / "history.json").exists())
                self.assertFalse((evaluation / "best_model.pt").exists())
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(["--data-dir", str(data), "--output-dir", str(output)])
                self.assertEqual(model_mock.call_count, 2)

                checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["configuration"]["model"], "mamba")
                checkpoint["configuration"]["model"] = "unsupported"
                invalid_checkpoint = root / "invalid-checkpoint.pt"
                torch.save(checkpoint, invalid_checkpoint)
                with self.assertRaisesRegex(ValueError, "Mamba configuration"):
                    main(["--data-dir", str(data), "--output-dir", str(root / "invalid"),
                          "--checkpoint", str(invalid_checkpoint)])
                self.assertFalse((root / "invalid").exists())
                self.assertEqual(model_mock.call_count, 2)

                changed = data / "GaCo01_01.txt"
                changed.write_text(changed.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "Dataset content or split"):
                    main(["--data-dir", str(data), "--output-dir", str(root / "changed"),
                          "--checkpoint", str(output / "best_model.pt")])
                self.assertFalse((root / "changed").exists())
                self.assertEqual(model_mock.call_count, 2)

    def test_production_cpu_rejected_before_data_model_checkpoint_or_output(self):
        self.assert_rejected_device("cpu", "CPU execution is not supported")

    def test_production_auto_and_cuda_reject_unavailable_hardware(self):
        # Simulate absence only, never pretend a CPU test has CUDA hardware.
        with patch("gait.experiment.torch.cuda.is_available", return_value=False):
            for requested in ("auto", "cuda"):
                with self.subTest(device=requested):
                    self.assert_rejected_device(requested, "CUDA is unavailable")

    def assert_rejected_device(self, requested, message):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "must-not-exist"
            stderr = io.StringIO()
            with patch("gait.experiment.build_model") as model_mock, \
                    patch("gait.experiment.discover_recordings") as data_mock, \
                    patch("gait.experiment.torch.load") as load_mock, \
                    redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main(["--data-dir", str(root / "missing-data"), "--output-dir", str(output),
                      "--checkpoint", str(root / "missing-checkpoint.pt"), "--device", requested])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(message, stderr.getvalue())
            model_mock.assert_not_called()
            data_mock.assert_not_called()
            load_mock.assert_not_called()
            self.assertFalse(output.exists())

    def test_device_helper_rejects_unsupported_device(self):
        with self.assertRaisesRegex(ValueError, "auto or cuda"):
            resolve_device("unsupported")

    def test_cli_is_mamba_only_without_model_selector(self):
        help_text = parser().format_help()
        self.assertIn("Mamba", help_text)
        self.assertNotIn("--model", help_text)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            parser().parse_args(["--data-dir", "unused", "--output-dir", "unused",
                                 "--model", "other"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()