"""Static release checks; no notebook cells or training are executed here."""

import ast
import json
from pathlib import Path
import re
import subprocess
import unittest


class NotebookTests(unittest.TestCase):
    def test_lpr_notebook_clean_pinned_extensions_only(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/lpr-extensions.ipynb").read_text(encoding="utf-8"))
        code = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                ast.parse(source)
                code.append(source)
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        full = "\n".join(code)
        self.assertIn("RUN_EXPERIMENT = False", full)
        self.assertIn("SAVE_RESULTS=False", full)
        self.assertNotIn("plt.savefig", full)
        self.assertNotIn("INSTALL_MAMBA", full)
        self.assertIn("stale =", full)
        revision = re.search(r"SOURCE_REV = '([a-f0-9]{40})'", full).group(1)
        if (root / ".git").exists():
            for name in ("__init__.py", "models.py", "cross_validation.py", "official_data.py", "diagnostics.py", "residual_experiment.py"):
                expected = subprocess.run(["git", "-C", str(root), "show", revision+":gait/"+name], capture_output=True, check=True).stdout
                actual = (root / "gait" / name).read_bytes()
                self.assertEqual(actual.replace(b"\r\n", b"\n"), expected.replace(b"\r\n", b"\n"))

    def test_comparison_notebook_clean_syntax_safety_and_source_pin(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/pd-mamba-comparison.ipynb").read_text(encoding="utf-8"))
        code = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                ast.parse(source)
                code.append(source)
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        full = "\n".join(code)
        self.assertIn("RUN_EXPERIMENT = False", full)
        self.assertIn("SAVE_RESULTS = False", full)
        self.assertIn("TRAINING_SEEDS = (42, 123, 2026)", full)
        self.assertIn("CANDIDATES_PER_FAMILY = 8", full)
        self.assertNotIn("plt.savefig", full)
        revision = re.search(r"SOURCE_REV = '([a-f0-9]{40})'", full).group(1)
        if (root / ".git").exists():
            for name in ("models.py", "cross_validation.py", "comparison_models.py", "comparison.py", "official_data.py", "diagnostics.py"):
                expected = subprocess.run(["git", "-C", str(root), "show", revision+":gait/"+name], capture_output=True, check=True).stdout
                actual = (root / "gait" / name).read_bytes()
                self.assertEqual(actual.replace(b"\r\n", b"\n"), expected.replace(b"\r\n", b"\n"))

    def test_valid_clean_python_cells_and_pinned_implementation(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/pd-mamba-comparison.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["language_info"]["name"], "python")
        identifiers = []
        code = []
        for cell in notebook["cells"]:
            identifiers.append(cell["id"])
            # VS Code exposes virtual language metadata but serializes ordinary
            # Jupyter cells using cell_type and the notebook language_info.
            self.assertIn(cell["cell_type"], ("code", "markdown"))
            if "language" in cell["metadata"]:
                self.assertEqual(cell["metadata"]["language"], "python" if cell["cell_type"] == "code" else "markdown")
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                ast.parse(source)
                code.append(source)
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        self.assertEqual(len(identifiers), len(set(identifiers)))
        full_code = "\n".join(code)
        self.assertIn("RUN_EXPERIMENT = False", full_code)
        self.assertIn("SAVE_RESULTS = False", full_code)
        self.assertNotIn("plt.savefig", full_code)
        revision = re.search(r"SOURCE_REV = '([a-f0-9]{40})'", full_code).group(1)
        # Git-distributed notebook must not accidentally pin an older engine.
        if (root / ".git").exists():
            for filename in ("gait/models.py", "gait/cross_validation.py"):
                expected = subprocess.run(["git", "-C", str(root), "show", revision+":"+filename],
                                          check=True, capture_output=True).stdout
                actual = (root / filename).read_bytes()
                self.assertEqual(actual.replace(b"\r\n", b"\n"), expected.replace(b"\r\n", b"\n"))


if __name__ == "__main__":
    unittest.main()