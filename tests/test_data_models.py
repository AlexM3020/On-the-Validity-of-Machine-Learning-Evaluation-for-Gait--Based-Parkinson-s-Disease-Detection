"""Synthetic CPU and architecture tests for the reconstructed implementation.

Fake Mamba tests check wiring only, not CUDA kernels or scientific results.
Optional dependencies are not installed by this suite. Missing dependencies
produce explicit skips; no test claims to reproduce historical training.
"""

import builtins
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
if HAS_NUMPY:
    import numpy as np
if HAS_TORCH:
    import torch
    from torch import nn
if HAS_NUMPY and HAS_TORCH:
    from gait.data import GaitDataset, Recording, discover_recordings, split_recordings
if HAS_TORCH:
    from gait.models import MambaClassifier, RecurrentClassifier, build_model, wide_mamba_forward


class PackageTests(unittest.TestCase):
    def test_package_import_has_no_heavy_dependency_imports(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"numpy", "torch", "sklearn", "mamba_ssm"}:
                raise AssertionError(f"Unexpected package dependency import: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            importlib.reload(importlib.import_module("gait"))


@unittest.skipUnless(HAS_NUMPY and HAS_TORCH, "Data tests require numpy and torch")
class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name="GaCo01_01.txt", rows=10, columns=19):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = np.column_stack([
            np.arange(rows, dtype=np.float64) * (column + 1) + column * 10
            for column in range(columns)
        ])
        # Constant sensor and very different time values detect column mistakes.
        if columns > 1:
            raw[:, 1] = 7
        raw[:, 0] = np.arange(rows, dtype=np.float64) ** 2 + 1000
        np.savetxt(path, raw)
        return path, raw

    def test_known_normalization_channels_and_window_count(self):
        path, raw = self.write(rows=10)
        record = Recording(path, "GaCo01", 0)
        dataset = GaitDataset([record], seq_len=4)
        self.assertEqual(dataset.recordings, [record])
        self.assertEqual(len(dataset), 2)  # Two rows discarded, not padded.
        self.assertEqual([(w.start, w.valid_length) for w in dataset.windows], [(0, 4), (4, 4)])
        self.assertTrue(all(w.record == record for w in dataset.windows))
        selected = raw[:, 1:19]
        std = selected.std(axis=0)
        expected = (selected - selected.mean(axis=0)) / np.where(std == 0, 1, std)
        for index, start in enumerate((0, 4)):
            values, label, length = dataset[index]
            self.assertEqual(tuple(values.shape), (4, 18))
            self.assertEqual(values.dtype, torch.float32)
            self.assertEqual(label.dtype, torch.long)
            self.assertEqual(length.dtype, torch.long)
            self.assertEqual(tuple(label.shape), ())
            self.assertEqual(tuple(length.shape), ())
            self.assertEqual((label.item(), length.item()), (0, 4))
            np.testing.assert_allclose(values.numpy(), expected[start:start + 4], atol=1e-6)
            self.assertTrue(torch.equal(values[:, 0], torch.zeros(4)))
        values, _, _ = dataset[0]
        values.fill_(123)
        self.assertFalse(torch.any(dataset[0][0] == 123).item())

    def test_short_padding_and_exact_length(self):
        path, raw = self.write(name="JuPt03_01.txt", rows=3)
        dataset = GaitDataset([Recording(path, "JuPt03", 1)], seq_len=5, sensor_columns=(2, 18))
        values, label, length = dataset[0]
        self.assertEqual(len(dataset), 1)
        self.assertEqual((label.item(), length.item()), (1, 3))
        self.assertEqual(tuple(values.shape), (5, 2))
        np.testing.assert_allclose(values[:3].numpy(), (raw[:, (2, 18)] - raw[:, (2, 18)].mean(0)) / raw[:, (2, 18)].std(0), atol=1e-6)
        self.assertTrue(torch.equal(values[3:], torch.zeros(2, 2)))
        exact = GaitDataset([Recording(path, "JuPt03", 1)], seq_len=3)
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0][2].item(), 3)

    def test_default_sequence_length(self):
        path, _ = self.write(rows=1)
        values, _, length = GaitDataset([Recording(path, "GaCo01", 0)])[0]
        self.assertEqual(tuple(values.shape), (2000, 18))
        self.assertEqual(length.item(), 1)
        self.assertTrue(torch.equal(values, torch.zeros_like(values)))

    def test_invalid_matrix_and_nonfinite_values(self):
        for columns in (1, 18):
            with self.subTest(columns=columns):
                path, _ = self.write(rows=3, columns=columns)
                with self.assertRaisesRegex(ValueError, "shape"):
                    GaitDataset([Recording(path, "GaCo01", 0)])
        for value in (np.nan, np.inf, -np.inf):
            for column in (0, 2):
                with self.subTest(value=value, column=column):
                    path, raw = self.write(rows=3)
                    raw[1, column] = value
                    np.savetxt(path, raw)
                    with self.assertRaisesRegex(ValueError, "Nonfinite"):
                        GaitDataset([Recording(path, "GaCo01", 0)])
        path.write_text("1 2 3\n1 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cannot read"):
            GaitDataset([Recording(path, "GaCo01", 0)])
        path, raw = self.write(rows=3)
        raw[:, 2] = (1e308, -1e308, 1e308)
        np.savetxt(path, raw)
        with self.assertRaisesRegex(ValueError, "Nonfinite channel normalization"):
            GaitDataset([Recording(path, "GaCo01", 0)])

    def test_invalid_dataset_config(self):
        for seq_len in (0, -1, 2.5, True):
            with self.subTest(seq_len=seq_len), self.assertRaises(ValueError):
                GaitDataset([], seq_len=seq_len)
        for columns in ((), (-1,), (1, 1), (True,), (1.5,)):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                GaitDataset([], sensor_columns=columns)

    def test_discovery_strict_names_full_study_and_labels(self):
        names = {
            "nested/GaCo01_01.txt": ("GaCo01", 0),
            "GaCo01_02.txt": ("GaCo01", 0),
            "GaCo01_01_chunk_1.txt": ("GaCo01", 0),
            "JuCo01_01.txt": ("JuCo01", 0),
            "SiPt02_01.txt": ("SiPt02", 1),
            "GaPt03.txt": ("GaPt03", 1),
        }
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for name in ("README.txt", "GaCo1_01.txt", "GaCo001_01.txt", "GaCo01_notes.txt", "GaCo01_01_chunk_x.txt", "GaCo01_01_chunk_1_extra.txt", "OtherCo01_01.txt"):
            (self.root / name).touch()
        found = discover_recordings(self.root)
        self.assertEqual(len(found), len(names))
        self.assertEqual(
            {str(r.path.relative_to(self.root)).replace("\\", "/"): (r.subject, r.label) for r in found}, names
        )

    def test_discovery_empty_and_duplicate_basename(self):
        (self.root / "README.txt").touch()
        with self.assertRaisesRegex(ValueError, "No sensor recordings"):
            discover_recordings(self.root)
        with self.assertRaises(FileNotFoundError):
            discover_recordings(self.root / "missing")
        for directory in ("first", "second"):
            path = self.root / directory
            path.mkdir()
            (path / "GaCo01_01.txt").touch()
        with self.assertRaisesRegex(ValueError, "Duplicate recording filename"):
            discover_recordings(self.root)

    @unittest.skipUnless(HAS_SKLEARN, "Subject split tests require scikit-learn")
    def test_stratified_split_repeats_chunks_and_test_determinism(self):
        records = []
        for study in ("Ga", "Ju"):
            for group, label in (("Co", 0), ("Pt", 1)):
                for number in range(1, 11):
                    subject = f"{study}{group}{number:02d}"
                    for suffix in ("_01", "_02", "_01_chunk_1"):
                        records.append(Recording(self.root / f"{subject}{suffix}.txt", subject, label))
        for test_fraction, sizes in ((0.0, {"train": 32, "validation": 8}), (0.2, {"train": 24, "validation": 8, "test": 8})):
            with self.subTest(test_fraction=test_fraction):
                parts = split_recordings(records, test_fraction=test_fraction, seed=17)
                self.assertEqual(parts, split_recordings(reversed(records), test_fraction=test_fraction, seed=17))
                subject_sets = {key: {r.subject for r in group} for key, group in parts.items()}
                self.assertEqual({key: len(ids) for key, ids in subject_sets.items()}, sizes)
                for key, group in parts.items():
                    self.assertEqual(len(group), 3 * sizes[key])
                    self.assertEqual(sum(r.label == 1 for r in group), len(group) // 2)
                    for other, ids in subject_sets.items():
                        if key != other:
                            self.assertTrue(subject_sets[key].isdisjoint(ids))
                self.assertCountEqual([r for group in parts.values() for r in group], records)

    @unittest.skipUnless(HAS_SKLEARN, "Subject split tests require scikit-learn")
    def test_split_invalid_input_and_insufficient_strata(self):
        record = Recording(self.root / "GaCo01_01.txt", "GaCo01", 0)
        for options in ({"validation_fraction": 0}, {"validation_fraction": float("nan")}, {"test_fraction": -0.1}, {"test_fraction": 0.8}, {"seed": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                split_recordings([record], **options)
        with self.assertRaisesRegex(ValueError, "empty"):
            split_recordings([])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            split_recordings([record, Recording(self.root / "GaCo01_02.txt", "GaCo01", 1)])
        with self.assertRaisesRegex(ValueError, "Cannot stratify subjects"):
            split_recordings([record, Recording(self.root / "GaPt01_01.txt", "GaPt01", 1)])


@unittest.skipUnless(HAS_TORCH, "Model tests require torch")
class ModelTests(unittest.TestCase):
    def test_cpu_recurrent_forward_backward_batch_one(self):
        for name in ("lstm", "gru"):
            with self.subTest(name=name):
                model = build_model(name, input_dim=3, hidden_dim=4, num_layers=2)
                self.assertIsInstance(model, RecurrentClassifier)
                x = torch.randn(1, 7, 3, requires_grad=True)
                logits = model(x, torch.tensor([5]))
                self.assertEqual(tuple(logits.shape), (1, 2))
                nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all().item())
                self.assertTrue(torch.equal(x.grad[:, 5:], torch.zeros_like(x.grad[:, 5:])))
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()))

    def test_recurrent_padding_invariance_unsorted_lengths(self):
        for name in ("lstm", "gru"):
            with self.subTest(name=name):
                model = build_model(name, 3, hidden_dim=4, num_layers=2, dropout=0).eval()
                x = torch.randn(2, 8, 3)
                lengths = torch.tensor([3, 6])
                changed = x.clone()
                changed[0, 3:] = 1000
                changed[1, 6:] = -1000
                with torch.no_grad():
                    actual = model(x, lengths)
                    torch.testing.assert_close(actual, model(changed, lengths))
                    for index, length in enumerate(lengths.tolist()):
                        torch.testing.assert_close(actual[index:index + 1], model(x[index:index + 1, :length], torch.tensor([length])))
                    torch.testing.assert_close(model(x), model(x, torch.tensor([8, 8])))

    def test_lazy_mamba_import_and_clear_missing_backend_error(self):
        original_import = builtins.__import__
        attempted = []

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] == "mamba_ssm":
                attempted.append(name)
                raise ImportError("Backend intentionally unavailable in this unit test")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            # Execute a fresh module without replacing live classifier classes.
            # This makes the test independent of unittest method execution order.
            spec = importlib.util.spec_from_file_location(
                "isolated_gait_models", Path(__file__).resolve().parents[1] / "gait" / "models.py"
            )
            isolated = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(isolated)
            for name in ("lstm", "gru"):
                isolated.build_model(name, 3, hidden_dim=4)(torch.randn(1, 3, 3), torch.tensor([2]))
            self.assertEqual(attempted, [])
            with self.assertRaisesRegex(RuntimeError, "Linux/CUDA"):
                isolated.build_model("mamba", 3)
            self.assertEqual(attempted, ["mamba_ssm"])

    def test_mamba_architecture_wiring_only_with_explicit_fake(self):
        calls = []

        class FakeMamba(nn.Module):
            """Causal identity transform for architecture tests, not a backend."""

            def __init__(self, **config):
                super().__init__()
                calls.append(config)

            def forward(self, x):
                return x

        fake = types.ModuleType("mamba_ssm")
        fake.Mamba = FakeMamba
        with patch.dict(sys.modules, {"mamba_ssm": fake}):
            model = build_model("mamba", 3, hidden_dim=4, d_conv=4).eval()
        self.assertEqual(calls, [{"d_model": 4, "d_state": 32, "d_conv": 4, "expand": 2}] * 2)
        self.assertEqual(model.d_conv, 4)
        self.assertEqual(len(model.mamba_layers), 2)
        self.assertEqual(len(model.norms), 2)
        self.assertEqual((model.input_proj.in_features, model.cls_head.out_features), (3, 2))
        x = torch.randn(2, 7, 3, requires_grad=True)
        lengths = torch.tensor([2, 5])
        hidden = model.norm_in(model.input_proj(x))
        for norm in model.norms:
            hidden = norm(hidden + hidden)
        expected = torch.stack([hidden[index, :length].amax(0) for index, length in enumerate(lengths.tolist())])
        logits = model(x, lengths)
        torch.testing.assert_close(logits, model.cls_head(expected))
        changed = x.detach().clone()
        changed[0, 2:] = 1e4
        changed[1, 5:] = -1e4
        torch.testing.assert_close(logits, model(changed, lengths))
        for index, length in enumerate(lengths.tolist()):
            torch.testing.assert_close(logits[index:index + 1], model(x[index:index + 1, :length], torch.tensor([length])))
        logits.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertTrue(torch.equal(x.grad[0, 2:], torch.zeros_like(x.grad[0, 2:])))

    def test_mamba_default_wide_configuration_without_fake_forward(self):
        calls = []

        class ConfigurationOnlyMamba(nn.Module):
            """Records construction only; cannot execute a Mamba forward."""

            def __init__(self, **config):
                super().__init__()
                calls.append(config)

            def forward(self, hidden):
                raise AssertionError("Configuration-only fake must not be executed")

        fake = types.ModuleType("mamba_ssm")
        fake.Mamba = ConfigurationOnlyMamba
        with patch.dict(sys.modules, {"mamba_ssm": fake}):
            model = build_model("mamba", 3, hidden_dim=4)
        self.assertEqual(calls, [{"d_model": 4, "d_state": 32, "d_conv": 32, "expand": 2}] * 2)
        self.assertEqual(model.d_conv, 32)
        self.assertEqual(len(model.mamba_layers), 2)

    def test_wide_mamba_synthetic_scan_api_contract_not_real_cuda(self):
        """CPU tensor/gradient/causality contract only, not CUDA scan validation."""
        torch.manual_seed(17)
        blocks = []
        scan_calls = []

        class SyntheticMamba(nn.Module):
            """Actual parameter layout with an intentionally forbidden forward."""

            def __init__(self, d_model, d_state, d_conv, expand):
                super().__init__()
                inner = expand * d_model
                self.dt_rank = 2
                self.d_state = d_state
                self.in_proj = nn.Linear(d_model, 2 * inner, bias=False)
                self.conv1d = nn.Conv1d(
                    inner, inner, d_conv, groups=inner, padding=d_conv - 1,
                )
                self.act = nn.SiLU()
                self.x_proj = nn.Linear(inner, self.dt_rank + 2 * d_state, bias=False)
                self.dt_proj = nn.Linear(self.dt_rank, inner)
                with torch.no_grad():
                    self.dt_proj.bias.fill_(0.75)
                self.A_log = nn.Parameter(torch.arange(1, d_state + 1).float().log().repeat(inner, 1))
                self.D = nn.Parameter(torch.ones(inner))
                self.out_proj = nn.Linear(inner, d_model, bias=False)
                blocks.append(self)

            def forward(self, hidden):
                raise AssertionError("Wide path must bypass the optimized block forward")

        def fake_selective_scan(x, dt, a, state_b, state_c, d, *, z, delta_bias, delta_softplus):
            # This fake checks the kernel boundary, not selective-scan math.
            block = blocks[len(scan_calls) % len(blocks)]
            batch, inner, time = x.shape
            for value in (x, dt, z):
                self.assertEqual(tuple(value.shape), (batch, inner, time))
                self.assertTrue(value.is_contiguous())
            for value in (state_b, state_c):
                self.assertEqual(tuple(value.shape), (batch, block.d_state, time))
                self.assertTrue(value.is_contiguous())
            self.assertEqual(tuple(a.shape), (inner, block.d_state))
            self.assertEqual(tuple(d.shape), (inner,))
            self.assertEqual(tuple(delta_bias.shape), (inner,))
            self.assertTrue(delta_softplus)
            for value in (a, d, delta_bias):
                self.assertEqual(value.dtype, torch.float32)
                self.assertTrue(value.is_contiguous())
            torch.testing.assert_close(a, -block.A_log.float().exp())
            torch.testing.assert_close(d, block.D.float())
            torch.testing.assert_close(delta_bias, block.dt_proj.bias.float())
            projected = block.x_proj(x.transpose(1, 2))
            expected_dt, expected_b, expected_c = projected.split(
                (block.dt_rank, block.d_state, block.d_state), dim=-1,
            )
            torch.testing.assert_close(dt, nn.functional.linear(expected_dt, block.dt_proj.weight).transpose(1, 2))
            torch.testing.assert_close(state_b, expected_b.transpose(1, 2))
            torch.testing.assert_close(state_c, expected_c.transpose(1, 2))
            scan_calls.append(time)
            return x * nn.functional.silu(z)

        fake = types.ModuleType("mamba_ssm")
        fake.Mamba = SyntheticMamba
        fake_ops = types.ModuleType("mamba_ssm.ops")
        fake_scan = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
        fake_scan.selective_scan_fn = fake_selective_scan
        with patch.dict(sys.modules, {
            "mamba_ssm": fake, "mamba_ssm.ops": fake_ops,
            "mamba_ssm.ops.selective_scan_interface": fake_scan,
        }):
            model = build_model("mamba", 3, hidden_dim=4, d_state=3).eval()
            self.assertEqual(model.d_conv, 32)
            self.assertEqual(len(blocks), 2)
            x = torch.randn(2, 7, 3, requires_grad=True)
            lengths = torch.tensor([2, 5])
            logits = model(x, lengths)
            self.assertEqual(tuple(logits.shape), (2, 2))
            logits.square().sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all().item())
            self.assertGreater(x.grad.abs().sum().item(), 0)
            for block in blocks:
                for layer in (block.in_proj, block.conv1d, block.out_proj):
                    self.assertIsNotNone(layer.weight.grad)
                    self.assertTrue(torch.isfinite(layer.weight.grad).all().item())
                    self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
            changed = x.detach().clone()
            changed[0, 2:] = 1000
            changed[1, 5:] = -1000
            with torch.no_grad():
                torch.testing.assert_close(logits, model(changed, lengths))
                for index, length in enumerate(lengths.tolist()):
                    torch.testing.assert_close(
                        logits[index:index + 1],
                        model(x[index:index + 1, :length], torch.tensor([length])),
                    )
                hidden = torch.randn(2, 40, 4)
                wide = wide_mamba_forward(blocks[0], hidden)
                self.assertEqual(tuple(wide.shape), (2, 40, 4))
                changed_hidden = hidden.clone()
                changed_hidden[:, 35:] = 1000
                # Direct calls use block zero; reset the fake's call-order marker.
                scan_calls.clear()
                changed_wide = wide_mamba_forward(blocks[0], changed_hidden)
                torch.testing.assert_close(wide[:, :35], changed_wide[:, :35])

    def test_model_config_and_lengths_validation(self):
        for name in ("unknown", "", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_model(name, 3)
        for config in ({"input_dim": 0}, {"hidden_dim": -1}, {"num_layers": True}, {"d_state": 0}, {"d_conv": 2.5}, {"expand": 0}, {"dropout": float("nan")}, {"dropout": 1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                build_model("gru", **({"input_dim": 3} | config))
        model = build_model("GRU", 3, hidden_dim=4)
        for lengths in (torch.tensor([0]), torch.tensor([5]), torch.tensor([2.0]), torch.tensor([[2]]), torch.tensor([2, 2])):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                model(torch.randn(1, 4, 3), lengths)
        for x in (torch.randn(4, 3), torch.randn(1, 4, 2), torch.empty(1, 0, 3), torch.ones(1, 4, 3, dtype=torch.long)):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                model(x, torch.tensor([1]))


if __name__ == "__main__":
    unittest.main()
