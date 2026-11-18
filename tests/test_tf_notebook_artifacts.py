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
                self.assertIn('restore_checkpoint_bundle',text)
                self.assertNotIn('layers.Lambda',text)
                self.assertNotIn('safe_mode=False',text)
                revision=re.search(r"SOURCE_REV = '([a-f0-9]{40})'",text).group(1)
                if (root/'.git').exists():
                    for file in ('__init__.py','official_data.py','tf_notebook_support.py'):
                        original=subprocess.run(['git','-C',str(root),'show',revision+':gait/'+file],capture_output=True,check=True).stdout
                        self.assertEqual((root/'gait'/file).read_bytes().replace(b'\r\n',b'\n'),original.replace(b'\r\n',b'\n'))
                self.assertIn("COHORT_POLICY='official_all'" if name==NAMES[0] else "COHORT_POLICY='curated'",text)


if __name__=='__main__': unittest.main()