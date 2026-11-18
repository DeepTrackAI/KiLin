"""Unit tests for splitting JSON at its top-level keys."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kilin.chunking import chunk_source_units
from kilin.sources import extract_source_units, read_repository_files
from kilin.sources.extensions import sources_json
from kilin.sources.sources import SourceFile

SOURCE = """{
  "models": {
    "chat": "Qwen",
    "note": "a \\"quoted\\" text, with a comma"
  },
  "extensions": [
    ".py", {"nested": "key"}
  ],
  "a": 1, "b": 2
}
"""


def _units(text: str):
    return sources_json.extract_source_units(SourceFile(Path("x.json"), text))


class TestJsonExtraction(unittest.TestCase):
    """Keep JSON as written and label the parts of a large object."""

    def test_parts_start_at_top_level_keys(self) -> None:
        [unit] = _units(SOURCE)
        self.assertEqual(unit.kind, "json_file")
        self.assertEqual(
            [(child.section, child.start_line) for child in unit.children],
            [(None, 1), ('"models"', 2), ('"extensions"', 6), ('"a"', 9)],
        )
        self.assertEqual(
            "".join(child.text for child in unit.children), SOURCE
        )

    def test_one_line_or_an_array_stays_one_unit(self) -> None:
        for text in ('{"a": 1, "b": 2}', "[1, 2]\n", "not json\n"):
            with self.subTest(text=text):
                [unit] = _units(text)
                self.assertEqual(unit.children, ())

    def test_large_file_becomes_one_passage_per_key(self) -> None:
        values = ",\n".join(f'    "item_{n}": {n}' for n in range(40))
        text = (
            "{\n"
            f'  "first": {{\n{values}\n  }},\n'
            f'  "second": {{\n{values}\n  }}\n'
            "}\n"
        )
        chunks = chunk_source_units(
            _units(text),
            count_tokens=lambda part: len(part.split()),
            target_tokens=60,
            max_tokens=100,
            overlap_tokens=0,
        )
        sections = [chunk.section for chunk in chunks]
        self.assertIn('"first"', sections)
        self.assertIn('"second"', sections)

    def test_a_file_with_an_error_is_still_split(self) -> None:
        text = '{\n  "a": [1, 2,\n  "b": 3\n'
        [unit] = _units(text)
        self.assertEqual(
            [child.section for child in unit.children], [None, '"a"']
        )

    def test_unicode_separators_preserve_key_coordinates(self) -> None:
        """Attribute literal Unicode separators to the preceding value."""

        text = (
            '{\n  "alpha": "first\u2028second\u2029third",\n'
            '  "beta": 2,\n  "gamma": 3\n}\n'
        )
        self.assertEqual(
            json.loads(text)["alpha"], "first\u2028second\u2029third"
        )
        [unit] = _units(text)
        self.assertEqual(
            [
                (child.section, child.start_line, child.end_line)
                for child in unit.children
            ],
            [
                (None, 1, 1),
                ('"alpha"', 2, 4),
                ('"beta"', 5, 5),
                ('"gamma"', 6, 7),
            ],
        )
        self.assertEqual(
            unit.children[1].text,
            '  "alpha": "first\u2028second\u2029third",\n',
        )
        self.assertEqual(unit.children[2].text, '  "beta": 2,\n')
        self.assertEqual("".join(c.text for c in unit.children), text)
        chunks = chunk_source_units(
            [unit],
            count_tokens=len,
            target_tokens=30,
            max_tokens=50,
            overlap_tokens=0,
        )
        self.assertEqual(
            [
                (c.section, c.start_line, c.end_line)
                for c in chunks if c.section is not None
            ],
            [
                ('"alpha"', 2, 4),
                ('"beta"', 5, 5),
                ('"gamma"', 6, 7),
            ],
        )

    def test_source_reader_preserves_unicode_section_lines(self) -> None:
        """Keep coordinates correct after real UTF-8 source reading."""

        text = '{\n  "alpha": "A\u0085B",\n  "beta": 2\n}\n'
        with TemporaryDirectory() as folder:
            file = Path(folder, "settings.json")
            file.write_text(text, encoding="utf-8")
            report = {
                "repository": folder,
                "selected_files": ["settings.json"],
            }
            [source] = read_repository_files(report)
            [unit] = extract_source_units([source])
        self.assertEqual(json.loads(source.text)["alpha"], "A\u0085B")
        self.assertEqual(
            [(c.section, c.start_line, c.end_line) for c in unit.children],
            [(None, 1, 1), ('"alpha"', 2, 3), ('"beta"', 4, 5)],
        )
        self.assertEqual(unit.children[2].text, '  "beta": 2\n}\n')

    def test_escaped_separators_and_crlf_coordinates(self) -> None:
        """Count source lines, not escaped characters or newline bytes."""

        text = (
            '{\r\n  "alpha": "escaped \\u2028",\r\n'
            '  "beta": 2\r\n}\r\n'
        )
        self.assertEqual(json.loads(text)["alpha"], "escaped \u2028")
        [unit] = _units(text)
        self.assertEqual(
            [(c.section, c.start_line, c.end_line) for c in unit.children],
            [(None, 1, 1), ('"alpha"', 2, 2), ('"beta"', 3, 4)],
        )
        self.assertEqual("".join(c.text for c in unit.children), text)

    def test_empty_file(self) -> None:
        self.assertEqual(_units(""), [])


if __name__ == "__main__":
    unittest.main()
