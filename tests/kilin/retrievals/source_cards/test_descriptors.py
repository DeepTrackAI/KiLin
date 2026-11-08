"""Verify generic descriptors retain source identity without classifying it.

The source inventory deliberately contains files without extracted content.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.chunking import Chunk
from kilin.retrievals.source_cards import describe_sources


def _chunk(identifier: str, path: str) -> Chunk:
    """Build one source chunk without loading a file or model."""
    return Chunk(identifier, Path(path), "Example", 1, 1,
                 "text_paragraph", None, None)


class TestDescribeSources(unittest.TestCase):

    def test_mixed_formats_and_chunkless_source(self):
        paths = (
            Path("studies/paper.PDF"), Path("notes.md"),
            Path("src/module.py"), Path("certificate.pdf"),
            Path("photo.png"), Path("readme.txt"),
        )
        sources = describe_sources(
            paths,
            (_chunk("c1", "studies/paper.PDF"),
             _chunk("c2", "src/module.py"),
             _chunk("c3", "notes.md")),
            titles={Path("studies/paper.PDF"): "A study"},
        )
        self.assertEqual(tuple(sources), paths)
        self.assertEqual(sources[paths[0]].title, "A study")
        self.assertEqual(sources[paths[0]].suffix, ".pdf")
        self.assertEqual(sources[paths[2]].chunk_ids, ("c2",))
        self.assertEqual(sources[paths[3]].chunk_ids, ())
        self.assertEqual(sources[paths[4]].chunk_ids, ())
        self.assertEqual(sources[paths[5]].title, "readme")

    def test_preserves_input_chunk_order(self):
        source = describe_sources(
            (Path("a.txt"),),
            (_chunk("later", "a.txt"), _chunk("earlier", "a.txt")),
        )[Path("a.txt")]
        self.assertEqual(source.chunk_ids, ("later", "earlier"))

    def test_empty_selected_collection_is_valid(self):
        self.assertEqual(describe_sources((), ()), {})

    def test_unsupported_paths_and_duplicate_files(self):
        for paths in (
            (Path("a.txt"), Path("a.txt")),
            (Path("."),), (Path("../escape"),),
            (Path("C:/outside"),),
        ):
            with self.subTest(paths=paths):
                with self.assertRaises(ValueError):
                    describe_sources(paths, ())

    def test_unselected_chunk_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "unselected"):
            describe_sources((Path("a.txt"),), (_chunk("c", "b.txt"),))

    def test_duplicate_chunk_ids_across_files_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            describe_sources(
                (Path("a.txt"), Path("b.txt")),
                (_chunk("same", "a.txt"), _chunk("same", "b.txt")),
            )

    def test_title_lookup_must_be_for_known_source(self):
        with self.assertRaisesRegex(ValueError, "unselected"):
            describe_sources(
                (Path("a.txt"),), (),
                titles={Path("b.txt"): "Wrong"},
            )
        with self.assertRaisesRegex(ValueError, "title"):
            describe_sources(
                (Path("a.txt"),), (), titles={Path("a.txt"): " "},
            )


if __name__ == "__main__":
    unittest.main()
