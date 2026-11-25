"""Read-only notebook schema/source contract checks; never execute notebook cells."""
import ast
import json
from pathlib import Path
import re
import subprocess
import unittest


NAMES=("LPGNet_Exact_Replication_and_Factorial_Leakage_Study_RestartSafe",
       "LPGNet_Variant_A_Fully_Nested_and_Diagnostics","LP_ResMIL_Fully_Nested_Evaluation")


class TensorflowNotebookArtifactTests(unittest.TestCase):
    def test_all_notebooks_clean_pinned_and_guarded(self):
        root=Path(__file__).resolve().parents[1]
        for name in NAMES:
            with self.subTest(notebook=name):
                n=json.loads((root/'notebooks'/(name+'.ipynb')).read_text(encoding='utf-8'))
                ids=[c['id'] for c in n['cells']]
                self.assertEqual(len(ids),len(set(ids)))
                code=[]
                for c in n['cells']:
                    if c['cell_type']=='code':
                        source=''.join(c['source']); ast.parse(source); code.append(source)
                        self.assertEqual(c['outputs'],[]); self.assertIsNone(c['execution_count'])
                text='\n'.join(code)
                self.assertIn('RUN_EXPERIMENT=False',text)
                self.assertIn('MAX_NEW_TASKS=1',text)
                self.assertIn('if EXPERIMENT_COMPLETE:',text)
                self.assertIn('connect_storage',text)
                self.assertIn('DURABLE_MIRROR',text)
                expected_profile='full' if name==NAMES[0] else 'efficient'
                self.assertIn(f"RUN_PROFILE='{expected_profile}'",text)
                self.assertIn("STORAGE_BACKEND='colab_drive'",text)
                self.assertIn('REQUIRE_DURABLE_STORAGE=True',text)
                self.assertNotIn('layers.Lambda',text)
                self.assertNotIn('safe_mode=False',text)
                revision=re.search(r"SOURCE_REV = '([a-f0-9]{40})'",text).group(1)
                if (root/'.git').exists():
                    for file in ('__init__.py','official_data.py','tf_notebook_support.py','durable_storage.py'):
                        original=subprocess.run(['git','-C',str(root),'show',revision+':gait/'+file],capture_output=True,check=True).stdout
                        self.assertEqual((root/'gait'/file).read_bytes().replace(b'\r\n',b'\n'),original.replace(b'\r\n',b'\n'))
                self.assertIn("COHORT_POLICY='official_all'" if name==NAMES[0] else "COHORT_POLICY='curated'",text)
                self.assertIn('On-the-Validity-of-Machine-Learning-Evaluation-for-Gait--Based-Parkinson-s-Disease-Detection.git',text)
                self.assertNotIn('Subject-Independent-Evaluation-of-Mamba-for-Parkinson-s-Disease-Detection-from-Gait-Signals.git',text)
                if name==NAMES[0]:
                    self.assertIn('N_BOOT=1000',text)
                    self.assertIn('BOOTSTRAP_SEED=2028',text)
                    self.assertIn('PAIRED_BOOTSTRAP_SEED=2028',text)
                    self.assertIn('default_rng(BOOTSTRAP_SEED)',text)
                    self.assertNotIn('default_rng(2026)',text)
                    self.assertIn("'N_BOOT','BOOTSTRAP_SEED','PAIRED_BOOTSTRAP_SEED'",text)


if __name__=='__main__': unittest.main()