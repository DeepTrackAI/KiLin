"""Unit tests for R declaration boundaries.

R is never run and no R installation is required.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.sources.extensions import sources_r
from kilin.sources.sources import SourceFile

SOURCE = """# Analysis helpers.
library(stats)

#' Double a number.
#' @param x A number.
double_it <- function(x) {
  # "}" in a comment is not a bracket
  label <- "}"
  2 * x
}

square <- function(x) x^2

clean.data = function(values)
{
  values[values > 0]
}

`odd name` <- function() NULL

Store <- setRefClass("Store",
  fields = list(items = "list"),
  methods = list(
    # Keeps one more item.
    add = function(x) {
      items[[length(items) + 1]] <<- x
    }
  )
)

setClass("Shape", representation("VIRTUAL"))
setMethod("area", "Circle", function(shape) {
  pi * shape@r^2
})
result <- double_it(3)
"""


def _units(text: str = SOURCE):
    return sources_r.extract_source_units(SourceFile(Path("a.R"), text))


class TestRExtraction(unittest.TestCase):
    """Verify declarations, attached comments and the code between."""

    def test_declarations_and_context_cover_the_file(self) -> None:
        units = _units()
        self.assertEqual(
            [(unit.kind, unit.symbol) for unit in units],
            [
                ("r_script", None),
                ("r_function", "double_it"),
                ("r_function", "square"),
                ("r_function", "clean.data"),
                ("r_function", "odd name"),
                ("r_class", "Store"),
                ("r_class", "Shape"),
                ("r_method", "Circle.area"),
                ("r_script", None),
            ],
        )
        covered = [
            line
            for unit in units
            for line in range(unit.start_line, unit.end_line + 1)
        ]
        lines = SOURCE.splitlines()
        blank = {
            number
            for number, line in enumerate(lines, start=1)
            if not line.strip()
        }
        self.assertEqual(set(covered) | blank, set(range(1, len(lines) + 1)))

    def test_roxygen_comment_stays_with_its_function(self) -> None:
        function = _units()[1]
        self.assertTrue(function.text.startswith("#' Double a number."))
        self.assertTrue(function.text.rstrip().endswith("}"))

    def test_brackets_in_comments_and_strings_are_ignored(self) -> None:
        function = _units()[1]
        self.assertEqual((function.start_line, function.end_line), (4, 10))

    def test_one_line_and_next_line_bodies(self) -> None:
        units = {unit.symbol: unit for unit in _units()}
        self.assertEqual(units["square"].end_line, units["square"].start_line)
        self.assertTrue(units["clean.data"].text.rstrip().endswith("}"))

    def test_comments_between_signature_and_body_stay_in_function(
        self,
    ) -> None:
        text = (
            'continued <- function(x)\n'
            '# Why the body begins below.\n'
            "#' An additional note.\n"
            '{\n'
            '  x + 1\n'
            '}\n'
            'after <- function() 2\n'
        )
        units = _units(text)
        self.assertEqual(
            [(unit.kind, unit.symbol) for unit in units],
            [
                ('r_function', 'continued'),
                ('r_function', 'after'),
            ],
        )
        self.assertEqual((units[0].start_line, units[0].end_line), (1, 6))
        self.assertIn('# Why the body begins below.', units[0].text)
        self.assertIn('x + 1', units[0].text)

    def test_reference_class_methods_are_children(self) -> None:
        store = next(unit for unit in _units() if unit.symbol == "Store")
        self.assertEqual(
            [(child.kind, child.symbol) for child in store.children],
            [
                ("r_class_context", None),
                ("r_method", "Store.add"),
                ("r_class_context", None),
            ],
        )
        method = store.children[1]
        self.assertTrue(method.text.lstrip().startswith("# Keeps"))

    def test_empty_file(self) -> None:
        self.assertEqual(_units(""), [])


if __name__ == "__main__":
    unittest.main()
