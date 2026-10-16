"""Test MATLAB code-graph extraction and relationship detection.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.retrievals.code_graph import code_graph as cg
from kilin.retrievals.code_graph.extensions import code_graph_m as matlab
from tests.kilin.retrievals._support import source_file


class TestMatlabCodeGraphExtension(unittest.TestCase):
    """Verify MATLAB relation extraction with realistic source patterns."""

    def test_class_inheritance_calls_and_control_keywords(self) -> None:
        """Detect methods and omit MATLAB control constructs from call hints.

        """
        f = source_file(
            "pkg/Matrix.m",
            "classdef Matrix < Base\n"
            "function out = transform(obj)\n"
            "if(flag)\n    helper(1);\nend\n"
            "function out = other(obj)\n"
            "for(i)\n    transform(obj);\nend\n",
        )
        known = {"Base": {"pkg.Base"},
                 "transform": {"pkg.Matrix.transform"}}
        relations = matlab.extract_code_relations(f, known)
        self.assertIn(cg.CodeRelation("pkg.Matrix.Matrix", "inherits",
                                      "pkg.Base"), relations)
        self.assertIn(cg.CodeRelation("pkg.Matrix.Matrix.transform", "calls",
                                      "helper"), relations)
        self.assertIn(cg.CodeRelation("pkg.Matrix.Matrix.other", "calls",
                                      "pkg.Matrix.transform"), relations)
        self.assertFalse(any(r.target.lower() in ("if", "for")
                             for r in relations))

    def test_simple_functions_without_class(self) -> None:
        """Use the module name when a MATLAB class declaration is absent.

        """
        f = source_file("fit.m", "function out = fit(x)\nhelper(x);\n")
        result = matlab.extract_code_relations(f, {})
        self.assertIn(cg.CodeRelation("fit.fit", "calls", "helper"), result)

    def test_no_functions_produces_no_relations(self) -> None:
        """Non-function MATLAB content should not invent relationships.

        """
        f = source_file("notes.m", "x = 1;\n")
        self.assertEqual(matlab.extract_code_relations(f, {}), set())
