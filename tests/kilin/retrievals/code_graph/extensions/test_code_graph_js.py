"""Test JavaScript code-graph extraction."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.retrievals.code_graph import code_graph as cg
from kilin.retrievals.code_graph.extensions import code_graph_js as js
from kilin.sources import extract_source_units
from tests.kilin.retrievals._support import source_file

SCRIPT = """// helper("x") in a comment is not a call
function helper(a) {
  return a;
}
class Base {}
class Store extends Base {
  save(name) {
    console.log("helper(1)");
    if (name) {
      return helper(name);
    }
  }
}
init();
function init() {
  new Store().save(1);
}
"""


class TestJavaScriptCodeGraphExtension(unittest.TestCase):
    """Link functions and methods to the plain calls in their bodies."""

    def setUp(self) -> None:
        files = [source_file("web/app.js", SCRIPT)]
        graph = cg.build_code_graph(files, extract_source_units(files))
        self.relations = {
            relation
            for relation in graph.relations
            if relation.relation != "contains"
        }

    def test_calls_inheritance_and_top_level_code(self) -> None:
        self.assertEqual(
            self.relations,
            {
                cg.CodeRelation(
                    "web.app.Store.save", "calls", "web.app.helper"
                ),
                cg.CodeRelation("web.app.Store", "inherits", "web.app.Base"),
                cg.CodeRelation("web.app", "calls", "web.app.init"),
                cg.CodeRelation("web.app.init", "calls", "web.app.Store"),
            },
        )

    def test_comments_strings_keywords_and_methods_are_not_calls(self) -> None:
        targets = {relation.target for relation in self.relations}
        for name in ("if", "log", "save", "console"):
            with self.subTest(name=name):
                self.assertFalse(
                    any(target.endswith(name) for target in targets)
                )

    def test_extension_alone(self) -> None:
        relations = js.extract_code_relations(
            source_file("a.js", "function a() {\n  b();\n}\n"), {}
        )
        self.assertEqual(relations, {cg.CodeRelation("a.a", "calls", "b")})


if __name__ == "__main__":
    unittest.main()
