"""Guard package ownership and private shared-service boundaries.

Every registered executable tool must be independently packaged, with a
validated local configuration and no cross-tool implementation imports.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

from qachat.tools._registry import TOOL_REGISTRY

_ROOT = Path(__file__).resolve().parents[3] / "qachat" / "tools"
_REGISTERED = frozenset(TOOL_REGISTRY)


class TestToolPackageArchitecture(unittest.TestCase):
    """Reject flat executable modules and cross-tool dependencies."""

    def test_each_registered_tool_has_local_resources(self) -> None:
        """Require the standard tool layout and documented JSON keys.

        """

        for name in sorted(_REGISTERED):
            with self.subTest(tool=name):
                folder = _ROOT / name
                self.assertTrue((folder / "__init__.py").is_file())
                self.assertTrue((folder / "tool.py").is_file())
                self.assertTrue((folder / "_configuration.py").is_file())
                self.assertFalse((_ROOT / f"{name}.py").exists())
                data = json.loads((folder / "defaults.json").read_text())
                readme = (folder / "README.md").read_text()
                self.assertIn("SETTINGS", data)
                for section, values in data.items():
                    self.assertIn(f"`{section}`", readme)
                    self.assertIsInstance(values, dict)
                    for key in values:
                        self.assertIn(f"`{key}`", readme)

    def test_executable_tools_do_not_import_each_other(self) -> None:
        """Keep reusable algorithms in private shared modules.

        """

        for name in sorted(_REGISTERED):
            for file in (_ROOT / name).rglob("*.py"):
                with self.subTest(tool=name, module=file.name):
                    tree = ast.parse(file.read_text())
                    for node in ast.walk(tree):
                        imported = ""
                        if isinstance(node, ast.ImportFrom):
                            imported = node.module or ""
                        elif isinstance(node, ast.Import):
                            imported = " ".join(
                                alias.name for alias in node.names
                            )
                        for other in _REGISTERED - {name}:
                            prefix = f"qachat.tools.{other}"
                            self.assertNotIn(prefix, imported)

    def test_private_shared_modules_do_not_import_tools(self) -> None:
        """Prevent circular dependencies from shared infrastructure.

        """

        for file in (_ROOT / "_shared").glob("*.py"):
            with self.subTest(module=file.name):
                tree = ast.parse(file.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        imported = node.module or ""
                        for name in _REGISTERED:
                            self.assertFalse(
                                imported.startswith(f"qachat.tools.{name}")
                            )


if __name__ == "__main__":
    unittest.main()
