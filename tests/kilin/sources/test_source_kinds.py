"""Test the shared names and groups of code passage kinds."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.sources import source_kinds as kinds
from kilin.sources.extensions import (
    sources_js,
    sources_m,
    sources_py,
    sources_r,
)
from kilin.sources.sources import SourceFile

SAMPLES = (
    (
        sources_py,
        "a.py",
        '"""Doc."""\n\ndef f():\n    pass\n\nclass C:\n'
        "    x = 1\n\n    def m(self):\n        pass\n",
    ),
    (
        sources_m,
        "C.m",
        "classdef C\n  methods\n    function obj = C()\n    end\n"
        "    function m(obj)\n    end\n  end\nend\n",
    ),
    (sources_m, "f.m", "function f()\nend\n"),
    (
        sources_js,
        "a.js",
        "const x = 1;\nfunction f() {}\nclass C {\n  m() {}\n}\n",
    ),
    (
        sources_r,
        "a.R",
        "x <- 1\nf <- function() {\n  1\n}\n"
        'C <- setRefClass("C",\n  methods = list(\n'
        "    m = function() {\n      1\n    }\n  )\n)\n",
    ),
)


def _kinds(unit) -> set[str]:
    return {unit.kind} | {
        kind for child in unit.children for kind in _kinds(child)
    }


class TestSourceKinds(unittest.TestCase):
    """Keep saved kind values and give each group its meaning."""

    def test_values_are_the_saved_strings(self) -> None:
        self.assertEqual(kinds.PYTHON_FUNCTION, "python_function")
        self.assertEqual(kinds.MATLAB_CONSTRUCTOR, "matlab_constructor")
        self.assertEqual(kinds.JS_METHOD, "js_method")
        self.assertEqual(kinds.R_SCRIPT, "r_script")

    def test_methods_and_constructors_are_callables(self) -> None:
        self.assertLessEqual(kinds.METHOD_KINDS, kinds.CALLABLE_KINDS)
        for kind in (
            kinds.MATLAB_METHOD,
            kinds.MATLAB_CONSTRUCTOR,
            kinds.JS_FUNCTION,
            kinds.R_METHOD,
        ):
            with self.subTest(kind=kind):
                self.assertIn(kind, kinds.CALLABLE_KINDS)
        self.assertNotIn(kinds.PYTHON_FUNCTION, kinds.METHOD_KINDS)
        self.assertTrue(kinds.CLASS_KINDS.isdisjoint(kinds.CALLABLE_KINDS))

    def test_every_kind_a_reader_produces_is_named(self) -> None:
        named = {
            value
            for name, value in vars(kinds).items()
            if name.isupper() and isinstance(value, str)
        }
        for reader, path, text in SAMPLES:
            units = reader.extract_source_units(SourceFile(Path(path), text))
            produced = {kind for unit in units for kind in _kinds(unit)}
            # MATLAB class sections ("matlab_methods_block") are named by
            # their keyword.
            produced = {k for k in produced if not k.endswith("_block")}
            with self.subTest(reader=reader.__name__):
                self.assertLessEqual(produced, named)

    def test_code_prefixes_cover_every_code_kind(self) -> None:
        for kind in kinds.CALLABLE_KINDS | kinds.CLASS_KINDS:
            with self.subTest(kind=kind):
                self.assertTrue(kind.startswith(kinds.CODE_KIND_PREFIXES))


if __name__ == "__main__":
    unittest.main()
