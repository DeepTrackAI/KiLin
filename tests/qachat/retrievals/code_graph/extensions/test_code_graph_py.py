"""Test Python code-graph extraction and AST relationship detection.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import ast
import unittest

from qachat.retrievals.code_graph import code_graph as cg
from qachat.retrievals.code_graph.extensions import code_graph_py as py
from tests.qachat.retrievals._support import source_file


class TestPythonCodeGraphExtension(unittest.TestCase):
    """Test Python AST relationship heuristics without source execution."""

    def test_syntax_error_yields_empty_results(self) -> None:
        """Malformed files produce no speculative relationship records.

        """
        f = source_file("bad.py", "def invalid(: pass")
        self.assertEqual(py.extract_code_relations(f, {}), set())
        self.assertEqual(py.extract_module_constants(f), {})

    def test_simple_uppercase_module_constants(self) -> None:
        """Only top-level simple uppercase assignments become known names.

        """
        f = source_file(
            "pkg/options.py",
            "MODEL_CHOICES = 1\nMAX_TOKENS: int = 5\n"
            "lower = 2\nA, B = 2, 3\nobj.NAME = 4\n"
            "def f():\n    INNER = 2\n",
        )
        self.assertEqual(
            py.extract_module_constants(f),
            {"MODEL_CHOICES": "pkg.options", "MAX_TOKENS": "pkg.options"},
        )

    def test_import_inheritance_async_and_nested_calls(self) -> None:
        """Distinguish lexical scopes while recording supported relations.

        """
        f = source_file(
            "pkg/module.py",
            "import os\nfrom math import sqrt\n"
            "class Example(Base):\n"
            "    def run(self):\n        return helper(sqrt(4))\n"
            "async def perform():\n    return helper()\n",
        )
        known = {"Base": {"pkg.base.Base"},
                 "helper": {"pkg.helper"}}
        result = py.extract_code_relations(f, known)
        self.assertIn(cg.CodeRelation("pkg.module", "imports", "os"),
                      result)
        self.assertIn(cg.CodeRelation("pkg.module", "imports", "math.sqrt"),
                      result)
        self.assertIn(cg.CodeRelation("pkg.module.Example", "inherits",
                                      "pkg.base.Base"), result)
        self.assertIn(cg.CodeRelation("pkg.module.Example.run", "calls",
                                      "pkg.helper"), result)
        self.assertIn(cg.CodeRelation("pkg.module.perform", "calls",
                                      "pkg.helper"), result)

    def test_uppercase_reference_resolution(self) -> None:
        """Reference unique module constants from another module only.

        """
        f = source_file(
            "pkg/use.py",
            "def run():\n    return MODEL_CHOICES + UNRESOLVED\n",
        )
        known = {"MODEL_CHOICES": {"pkg.options"},
                 "UNRESOLVED": {"pkg.first", "pkg.second"}}
        relations = py.extract_code_relations(f, known)
        self.assertIn(cg.CodeRelation("pkg.use.run", "references",
                                      "pkg.options"), relations)
        self.assertFalse(any(r.relation == "references" and
                             r.target == "UNRESOLVED" for r in relations))

    def test_reference_helper_name_attribute_and_dynamic(self) -> None:
        """Recognize static dotted names and skip dynamic expressions.

        """
        names = [
            ("call", "call"),
            ("thing.attr", "thing.attr"),
            ("factory().method", "method"),
            ("[1, 2]", None),
        ]
        for expression, expected in names:
            with self.subTest(expression=expression):
                node = ast.parse(expression, mode="eval").body
                self.assertEqual(py._reference(node), expected)
