"""Test R code-graph extraction."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.retrievals.code_graph import code_graph as cg
from kilin.retrievals.code_graph.extensions import code_graph_r as r
from kilin.sources import extract_source_units
from tests.kilin.retrievals._support import source_file

SCRIPT = """library(stats)
source("helpers.R")

# helper(1) in a comment is not a call
helper <- function(a) {
  a
}

Base <- setRefClass("Base", fields = list(value = "numeric"))

Store <- setRefClass(
  "Store",
  contains = "Base",
  methods = list(
    save = function(x) {
      message("helper(1)")
      if (x > 0) helper(x) else stats::median(x)
    }
  )
)

run <- function() {
  store <- Store$new(value = 1)
  store$save(2)
  helper(3)
}
"""


class TestRCodeGraphExtension(unittest.TestCase):
    """Link functions and methods to the plain calls in their bodies."""

    def setUp(self) -> None:
        files = [
            source_file("analysis/main.R", SCRIPT),
            source_file("analysis/helpers.R", "util <- function() 1\n"),
        ]
        graph = cg.build_code_graph(files, extract_source_units(files))
        self.relations = {
            relation
            for relation in graph.relations
            if relation.relation != "contains"
        }

    def test_calls_inheritance_and_loaded_code(self) -> None:
        main = "analysis.main"
        for expected in (
            cg.CodeRelation(main, "imports", "stats"),
            cg.CodeRelation(main, "imports", "analysis.helpers"),
            cg.CodeRelation(f"{main}.Store", "inherits", f"{main}.Base"),
            cg.CodeRelation(f"{main}.Store.save", "calls", f"{main}.helper"),
            cg.CodeRelation(f"{main}.run", "calls", f"{main}.helper"),
        ):
            with self.subTest(relation=expected):
                self.assertIn(expected, self.relations)

    def test_comments_strings_keywords_and_object_calls_are_skipped(
        self,
    ) -> None:
        targets = {
            relation.target
            for relation in self.relations
            if relation.relation == "calls"
        }
        for name in ("if", "function", "new", "save", "median", "source"):
            with self.subTest(name=name):
                self.assertFalse(
                    any(target.endswith(name) for target in targets)
                )
        calls_in_comment = {
            relation.source
            for relation in self.relations
            if relation.target.endswith("helper")
        }
        self.assertNotIn("analysis.main", calls_in_comment)

    def test_comment_and_string_loader_text_does_not_import(
        self,
    ) -> None:
        text = (
            '# library(fakepkg)\n'
            '# source("phantom.R")\n'
            'message("library(fakequoted)")\n'
            'message("source(\\"quoted.R\\")")\n'
            'value <- "note # is text"\n'
            'library(realpkg)\n'
            'require("also.real")\n'
            'requireNamespace("thirdpkg")\n'
            'source("helpers.R")\n'
        )
        files = [
            source_file('analysis/main.R', text),
            source_file('analysis/helpers.R', 'helper <- function() 1\n'),
        ]
        graph = cg.build_code_graph(files, extract_source_units(files))
        imports = {
            relation.target
            for relation in graph.relations
            if relation.source == 'analysis.main'
            and relation.relation == 'imports'
        }
        self.assertEqual(
            imports,
            {'realpkg', 'also.real', 'thirdpkg', 'analysis.helpers'},
        )

    def test_commented_inheritance_cannot_override_real_base(
        self,
    ) -> None:
        source = (
            'Base <- setRefClass("Base")\n'
            'Child <- setRefClass("Child",\n'
            '  # contains = "Wrong",\n'
            '  note = "contains = \\"Quoted\\"",\n'
            '  contains = "Base")\n'
        )
        files = [source_file('class.R', source)]
        graph = cg.build_code_graph(files, extract_source_units(files))
        inherits = {
            relation
            for relation in graph.relations
            if relation.relation == 'inherits'
        }
        self.assertEqual(
            inherits,
            {cg.CodeRelation('class.Child', 'inherits', 'class.Base')},
        )

    def test_commented_inheritance_alone_adds_no_edge(self) -> None:
        source = (
            'Child <- setRefClass("Child",\n'
            '  # contains = "Wrong"\n'
            '  fields = list(x = "numeric"))\n'
        )
        files = [source_file('class.R', source)]
        graph = cg.build_code_graph(files, extract_source_units(files))
        self.assertFalse(
            any(
                relation.relation == 'inherits'
                for relation in graph.relations
            )
        )

    def test_r6_inherit_is_recognized_outside_comments(self) -> None:
        source = (
            'Base <- R6::R6Class("Base")\n'
            'Child <- R6::R6Class("Child",\n'
            '  # inherit = Fake,\n'
            '  inherit = Base)\n'
        )
        files = [source_file('r6.R', source)]
        graph = cg.build_code_graph(files, extract_source_units(files))
        self.assertIn(
            cg.CodeRelation('r6.Child', 'inherits', 'r6.Base'),
            graph.relations,
        )
        self.assertFalse(
            any(
                relation.target == 'Fake'
                for relation in graph.relations
            )
        )

    def test_extension_alone(self) -> None:
        relations = r.extract_code_relations(
            source_file("a.R", "a <- function() {\n  b()\n}\n"), {}
        )
        self.assertEqual(relations, {cg.CodeRelation("a.a", "calls", "b")})


if __name__ == "__main__":
    unittest.main()
