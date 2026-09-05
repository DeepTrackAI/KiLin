"""Tests for token-bounded source chunking."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.chunking import chunk_source_units
from qachat.source import SourceUnit


class TestChunking(unittest.TestCase):

    def test_chunk_source_units(self) -> None:
        small = SourceUnit(
            path=Path("small.py"),
            text="one two\n",
            start_line=4,
            end_line=4,
            kind="python_function",
            symbol="small",
        )
        chunks = chunk_source_units(
            [small],
            count_tokens=lambda text: len(text.split()),
            target_tokens=2,
            max_tokens=3,
            overlap_tokens=1,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, small.text)
        self.assertEqual(chunks[0].start_line, 4)
        self.assertEqual(chunks[0].symbol, "small")

        long = SourceUnit(
            path=Path("long.py"),
            text="one two\nthree four\nfive six\n",
            start_line=10,
            end_line=12,
            kind="python_function",
            symbol="long",
        )
        chunks = chunk_source_units(
            [long],
            count_tokens=lambda text: len(text.split()),
            target_tokens=2,
            max_tokens=3,
            overlap_tokens=1,
        )
        self.assertEqual([chunk.start_line for chunk in chunks], [10, 11, 12])
        self.assertTrue(all(len(chunk.text.split()) <= 3 for chunk in chunks))
        self.assertEqual(len({chunk.identifier for chunk in chunks}), 3)

        child = SourceUnit(
            path=Path("class.py"),
            text="small child\n",
            start_line=2,
            end_line=2,
            kind="python_method",
            symbol="Example.method",
        )
        parent = SourceUnit(
            path=Path("class.py"),
            text="class Example:\n    small child\n",
            start_line=1,
            end_line=2,
            kind="python_class",
            symbol="Example",
            children=(child,),
        )
        chunks = chunk_source_units(
            [parent],
            count_tokens=lambda text: len(text.split()),
            target_tokens=2,
            max_tokens=2,
            overlap_tokens=0,
        )
        self.assertEqual(
            [chunk.symbol for chunk in chunks], ["Example.method"]
        )

        invalid_settings = (
            {"target_tokens": 0},
            {"target_tokens": 3, "max_tokens": 2},
            {"target_tokens": 2, "overlap_tokens": 2},
            {"target_tokens": True},
        )
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    chunk_source_units(
                        [small],
                        count_tokens=lambda text: len(text),
                        **settings,
                    )

        with self.assertRaisesRegex(ValueError, "count_tokens"):
            chunk_source_units(
                [small],
                count_tokens=lambda text: -1,
            )


if __name__ == "__main__":
    unittest.main()
