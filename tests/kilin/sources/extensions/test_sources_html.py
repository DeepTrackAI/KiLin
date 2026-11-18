"""Unit tests for reading HTML as page text and as code.

Pages are parsed with the standard library; no script is run and no page
is rendered.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from kilin.sources import extract_source_units, read_repository_files
from kilin.sources.extensions import sources_html
from kilin.sources.sources import SourceFile

PAGE = """<!doctype html>
<html>
<head>
  <title>Study notes</title>
  <style>p { color: red; }</style>
</head>
<body>
<nav>Home | About</nav>
<section id="aims">
  <h2>Aims &amp; scope</h2>
  <p>We test
    <b>graph</b> measures.<br>On two cohorts.</p>
  <ul><li><p>First cohort</p></li><li>Second cohort</li></ul>
</section>
<section class="results wide">
  <h2>Results</h2>
  <table><tr><th>Group</th><th>N</th></tr>
  <tr><td>A</td><td>12</td></tr></table>
  <pre>x = 1
# not a heading</pre>
  <p># Also not a heading</p>
  <button onclick="save()">Save</button>
</section>
<script src="app.js"></script>
<script>console.log("hidden");</script>
</body>
</html>
"""


class TestPageText(unittest.TestCase):
    """Keep what a reader sees, as Markdown-style lines."""

    def setUp(self) -> None:
        self.text = sources_html.page_text(PAGE)

    def test_title_headings_lists_and_tables(self) -> None:
        for line in (
            "# Study notes",
            "## Aims & scope",
            "- First cohort",
            "- Second cohort",
            "Group | N",
            "A | 12",
        ):
            with self.subTest(line=line):
                self.assertIn(line, self.text.splitlines())

    def test_source_line_ends_are_spaces_and_br_is_a_new_line(self) -> None:
        self.assertIn("We test graph measures.\nOn two cohorts.", self.text)

    def test_scripts_styles_and_tags_are_left_out(self) -> None:
        for hidden in ("color: red", "hidden", "<", "app.js"):
            with self.subTest(hidden=hidden):
                self.assertNotIn(hidden, self.text)

    def test_text_starting_with_a_hash_is_not_a_heading(self) -> None:
        self.assertIn("```\nx = 1\n# not a heading\n```", self.text)
        self.assertIn("\\# Also not a heading", self.text)

    def test_hidden_svg_title_is_not_the_page_title(self) -> None:
        """An icon's accessibility title is not a document heading."""

        html = (
            "<svg><title>Hidden icon</title></svg>"
            "<p>Visible paragraph</p>"
        )
        self.assertEqual(
            sources_html.page_text(html), "Visible paragraph\n"
        )

    def test_head_without_an_end_tag_keeps_the_body(self) -> None:
        text = sources_html.page_text(
            "<html><head><title>T</title><body><p>Body text</p>"
        )
        self.assertEqual(text, "# T\n\nBody text\n")

    def test_sections_follow_the_headings(self) -> None:
        units = sources_html.extract_source_units(
            SourceFile(Path("notes.html"), self.text)
        )
        self.assertEqual(
            [unit.section for unit in units],
            ["Study notes", "Aims & scope", "Results"],
        )


def _section(attributes: str, lines: int) -> str:
    """Return a section of the given number of lines."""

    body = "".join(f"  <p>Line {number}</p>\n" for number in range(lines - 2))
    return f"<section {attributes}>\n{body}</section>\n"


CODE = (
    "<!doctype html>\n<html>\n<head>\n<title>App</title>\n</head>\n"
    + _section('id="settings"', 12)
    + '<script src="app.js"></script>\n'
    + _section('class="results wide"', 12)
    + "</html>\n"
)


class TestCodeUnits(unittest.TestCase):
    """Keep HTML as written and label its main parts."""

    def setUp(self) -> None:
        [self.unit] = sources_html.extract_code_units(
            SourceFile(Path("web/index.html"), CODE)
        )

    def test_one_unit_keeps_the_whole_file(self) -> None:
        self.assertEqual(self.unit.kind, "html_file")
        self.assertEqual(self.unit.text, CODE)
        self.assertEqual(
            "".join(child.text for child in self.unit.children), CODE
        )

    def test_parts_are_labelled_and_short_ones_join_the_one_before(
        self,
    ) -> None:
        self.assertEqual(
            [
                (child.section, child.start_line, child.end_line)
                for child in self.unit.children
            ],
            [
                # The short <head> joins the lines before it.
                ("head", 1, 5),
                # The one-line <script> joins the settings section.
                ("section#settings", 6, 18),
                ("section.results", 19, 31),
            ],
        )

    def test_labels_use_id_class_or_source(self) -> None:
        for tag, attributes, label in (
            ("section", ' id="a" class="b"', "section#a"),
            ("nav", ' class="  main wide"', "nav.main"),
            ("script", ' src="app.js"', "script src=app.js"),
            ("style", "", "style"),
        ):
            with self.subTest(label=label):
                self.assertEqual(
                    sources_html._part_label(tag, attributes), label
                )

    def test_a_file_without_parts_has_no_children(self) -> None:
        [unit] = sources_html.extract_code_units(
            SourceFile(Path("part.html"), "<p>Hello</p>\n")
        )
        self.assertEqual(unit.children, ())


class TestReadingSwitch(unittest.TestCase):
    """Read page text by default and the file as written as code."""

    def test_as_code_keeps_the_file_as_written(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "index.html").write_text(PAGE, encoding="utf-8")
            report = {"repository": folder, "selected_files": ["index.html"]}
            [page] = read_repository_files(report)
            [code] = read_repository_files(report, as_code=True)
            self.assertEqual(page.text, sources_html.page_text(PAGE))
            self.assertEqual(code.text, PAGE)
            [unit] = extract_source_units([code], as_code=True)
            self.assertEqual(unit.kind, "html_file")
            units = extract_source_units([page])
            self.assertEqual(units[0].kind, "markdown_section")

    def test_declared_charset_is_used_when_not_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "old.html").write_bytes(
                '<meta charset="windows-1252"><p>Café</p>'.encode("cp1252")
            )
            report = {"repository": folder, "selected_files": ["old.html"]}
            [page] = read_repository_files(report)
            self.assertEqual(page.text, "Café\n")


if __name__ == "__main__":
    unittest.main()
