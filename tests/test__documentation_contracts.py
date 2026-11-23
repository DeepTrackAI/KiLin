"""Guard meaningful documentation contracts against future drift.

Documentation checks parse source without importing optional backends, Qt,
or tools. Examples are syntax-checked only; they may need a real workspace,
model, or runtime context and must not be executed by this test.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import ast
import doctest
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestDocumentationContracts(unittest.TestCase):
    """Check example syntax and the desktop bridge's public API overview."""

    def test_module_example_syntax(self) -> None:
        """Compile Python examples without running their side effects."""

        failures = []
        parser = doctest.DocTestParser()
        for path in sorted((ROOT / "kilin").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docstring = ast.get_docstring(tree) or ""
            try:
                examples = parser.get_examples(docstring)
                for example in examples:
                    compile(example.source, str(path), "single")
            except (SyntaxError, ValueError) as error:
                failures.append(f"{path.relative_to(ROOT)}: {error}")
        self.assertEqual(failures, [])

    def test_gui_class_methods_are_documented(self) -> None:
        """List all public action methods, but not properties, as methods."""

        failures = []
        for relative, name in (
            ("kilin/gui/engine.py", "ChatEngine"),
            ("kilin/gui/window.py", "ChatBridge"),
        ):
            path = ROOT / relative
            tree = ast.parse(path.read_text(encoding="utf-8"))
            cls = next(
                item for item in tree.body
                if isinstance(item, ast.ClassDef) and item.name == name
            )
            docstring = ast.get_docstring(cls) or ""
            methods = docstring.partition("Methods\n-------")[2]
            methods = methods.partition("Notes\n-----")[0]
            for member in cls.body:
                if not isinstance(
                    member, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) or member.name.startswith("_"):
                    continue
                if any(
                    isinstance(dec, ast.Name) and dec.id == "property"
                    for dec in member.decorator_list
                ):
                    continue
                pattern = rf"(?m)^\s*{re.escape(member.name)}\("
                if re.search(pattern, methods) is None:
                    failures.append(f"{relative}: {member.name}()")
        self.assertEqual(failures, [])

    def test_chat_bridge_signals_are_documented(self) -> None:
        """Account for every Qt signal and its argument signature."""

        path = ROOT / "kilin/gui/window.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(
            item for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == "ChatBridge"
        )
        attributes = (ast.get_docstring(cls) or "").partition(
            "Attributes\n----------"
        )[2].partition("Methods\n-------")[0]
        failures = []
        for member in cls.body:
            if not isinstance(member, ast.Assign):
                continue
            value = member.value
            if not isinstance(value, ast.Call):
                continue
            if not isinstance(value.func, ast.Name):
                continue
            if value.func.id != "Signal":
                continue
            for target in member.targets:
                if not isinstance(target, ast.Name):
                    continue
                pattern = rf"(?m)^\s*{re.escape(target.id)}:\s*Signal\("
                if re.search(pattern, attributes) is None:
                    failures.append(target.id)
        self.assertEqual(failures, [])
