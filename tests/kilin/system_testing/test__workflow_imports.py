"""Keep optional notebook and model dependencies outside workflow imports."""

# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0103:invalid-name

import subprocess
import sys
import unittest


class TestWorkflowImports(unittest.TestCase):
    def test_import_without_notebooks_models_or_qt(self):
        script = """
import sys
from importlib.abc import MetaPathFinder

class BlockOptional(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ImportError('Blocked optional dependency: ' + fullname)

blocked = {'nbformat', 'nbclient', 'jupyter_client', 'ipykernel', 'IPython',
           'mistune', 'torch', 'mlx', 'PySide6'}
sys.meta_path.insert(0, BlockOptional())
from kilin.system_testing import analysis, batch, generation
assert callable(batch.run_batch)
assert callable(generation.generate_reference)
assert '&lt;script&gt;' in analysis._answer_html('<script>test</script>')
assert not any(name.split('.')[0] in blocked for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True,
            text=True, timeout=30, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
