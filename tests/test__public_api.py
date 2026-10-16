"""Unit tests for KíLín's public Python API.

Tests verify that every library module is importable and declares a valid
public interface, including nested packages and GUI components.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

import kilin


class TestPublicAPI(unittest.TestCase):
    """Test imports and explicit exports across the source hierarchy."""

    @staticmethod
    def _module_names() -> list[str]:
        """Discover all library modules without importing them."""

        root = Path(kilin.__file__).resolve().parent
        names = ["kilin"]
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or path.name == "__main__.py":
                continue
            relative = path.relative_to(root).with_suffix("")
            parts = relative.parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            name = ".".join(("kilin", *parts))
            if name not in names:
                names.append(name)
        return names

    def test_all_modules_import_and_export_valid_names(self) -> None:
        """Require importable modules with complete valid __all__ lists."""

        names = self._module_names()
        script = """
import importlib
import json
import sys

results = {}
for name in json.loads(sys.stdin.read()):
    try:
        module = importlib.import_module(name)
        if not hasattr(module, "__all__"):
            raise AssertionError("missing __all__")
        exports = module.__all__
        if not isinstance(exports, (list, tuple)):
            raise AssertionError("__all__ must be a list or tuple")
        if any(not isinstance(item, str) or not item.isidentifier()
               for item in exports):
            raise AssertionError("invalid __all__ entry")
        if len(exports) != len(set(exports)):
            raise AssertionError("duplicate __all__ entries")
        missing = [item for item in exports if not hasattr(module, item)]
        if missing:
            raise AssertionError("missing exports: " + repr(missing))
        results[name] = None
    except Exception as error:
        results[name] = type(error).__name__ + ": " + str(error)
print("KILIN_API_RESULTS=" + json.dumps(results))
"""
        process = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(names),
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(
            process.returncode,
            0,
            f"API subprocess failed:\n{process.stderr}",
        )
        prefix = "KILIN_API_RESULTS="
        lines = [
            line
            for line in process.stdout.splitlines()
            if line.startswith(prefix)
        ]
        self.assertTrue(lines, f"No API results:\n{process.stdout}")
        results = json.loads(lines[-1][len(prefix) :])
        self.assertEqual(set(results), set(names))
        for name, error in results.items():
            with self.subTest(module=name):
                if (
                    name.startswith("kilin.gui.")
                    and error == (
                        "ModuleNotFoundError: "
                        "No module named 'PySide6'"
                    )
                ):
                    self.skipTest("PySide6 is not installed")

                self.assertIsNone(error, error)

    def test_root_exports(self) -> None:
        """Verify that root package exports resolve at runtime."""

        for name in kilin.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(kilin, name))

    def test_model_defaults_reexports(self) -> None:
        """Preserve public access to the shared model defaults."""

        module = importlib.import_module("kilin.model_defaults")
        for name in module.__all__:
            with self.subTest(name=name):
                self.assertIs(getattr(kilin, name), getattr(module, name))


if __name__ == "__main__":
    unittest.main()
