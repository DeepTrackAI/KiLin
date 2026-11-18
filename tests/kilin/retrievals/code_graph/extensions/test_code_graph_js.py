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


    def test_one_line_function_does_not_own_module_call(self) -> None:
        text = (
            "function first() { return 1; }\n"
            "second();\n"
            "function second() { return 2; }\n"
        )
        file = source_file("mini.js", text)
        graph = cg.build_code_graph([file], extract_source_units([file]))
        self.assertIn(
            cg.CodeRelation("mini", "calls", "mini.second"),
            graph.relations,
        )
        self.assertNotIn(
            cg.CodeRelation("mini.first", "calls", "mini.second"),
            graph.relations,
        )

    def test_regex_literals_do_not_invent_calls(self) -> None:
        text = (
            "const pattern = /helper(foo)/;\n"
            "function helper(x) { return x; }\n"
            "function real() { return /helper(bar)/.test(1); }\n"
        )
        file = source_file("mini.js", text)
        graph = cg.build_code_graph([file], extract_source_units([file]))
        self.assertFalse(
            any(
                relation.relation == "calls"
                and relation.target.endswith("helper")
                for relation in graph.relations
            )
        )

    def test_nested_object_signature_is_not_a_call(self) -> None:
        text = (
            "class Handler {\n"
            "  run() {\n"
            "    const item = {\n"
            "      fake() { return helper(); }\n"
            "    };\n"
            "    return item;\n"
            "  }\n"
            "}\n"
            "function helper() {}\n"
        )
        file = source_file("mini.js", text)
        graph = cg.build_code_graph([file], extract_source_units([file]))
        self.assertIn(
            cg.CodeRelation("mini.Handler.run", "calls", "mini.helper"),
            graph.relations,
        )
        self.assertFalse(
            any(
                relation.relation == "calls"
                and relation.target.endswith("fake")
                for relation in graph.relations
            )
        )
        self.assertFalse(
            any(entity.name.endswith("Handler.fake")
                for entity in graph.entities)
        )


if __name__ == "__main__":
    unittest.main()
