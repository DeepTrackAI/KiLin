"""Tests for deterministic, token-bounded source chunking.

The tests exercise the public chunking interface with deterministic token
counters. They check exact text, token limits, line boundaries, overlap,
provenance, identifiers, progress callbacks, and failure behavior.

Regressions verify that incomplete child coverage cannot lose substantive
text and that distinct provenance fields produce distinct identifiers.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from itertools import product
import unittest
from unittest.mock import Mock

from kilin.chunking import Chunk, chunk_source_units
from kilin.sources.sources import SourceUnit


def make_unit(
    text: str = "one two\n",
    *,
    path: str = "src/example.py",
    start_line: int = 1,
    end_line: int = 1,
    kind: str = "python_function",
    symbol: str | None = "example",
    section: str | None = None,
    cell: int | None = None,
    children: tuple[SourceUnit, ...] = (),
) -> SourceUnit:
    """Construct a source unit with explicit, overridable provenance.

    Parameters
    ----------
    text: str, optional
        Original source text.
    path: str, optional
        Repository-relative POSIX path.
    start_line: int, optional
        Inclusive first source line.
    end_line: int, optional
        Inclusive last source line.
    kind: str, optional
        Structural source-unit kind.
    symbol: str | None, optional
        Associated symbol name.
    section: str | None, optional
        Associated document section.
    cell: int | None, optional
        One-based notebook cell number.
    children: tuple[SourceUnit, ...], optional
        Natural subunits for hierarchical splitting.

    Returns
    -------
    SourceUnit
        New immutable source unit.

    """

    return SourceUnit(
        path=Path(path),
        text=text,
        start_line=start_line,
        end_line=end_line,
        kind=kind,
        symbol=symbol,
        section=section,
        cell=cell,
        children=children,
    )


class TestChunkContract(unittest.TestCase):
    """Verify chunk structure, identity, and immutable metadata."""

    def test_empty_input_returns_no_chunks(self) -> None:
        """Empty input produces no chunks or tokenizer calls."""

        counter = Mock(return_value=1)
        self.assertEqual(chunk_source_units([], count_tokens=counter), [])
        counter.assert_not_called()

    def test_blank_units_are_omitted(self) -> None:
        """Whitespace-only units do not become retrievable chunks."""

        counter = Mock(side_effect=len)
        units = [
            make_unit(""),
            make_unit(" \t\n"),
            make_unit("\n\n"),
        ]
        self.assertEqual(chunk_source_units(units, count_tokens=counter), [])
        counter.assert_not_called()

    def test_default_settings_keep_a_small_unit_whole(self) -> None:
        """Default token budgets accept an ordinary small unit."""

        unit = make_unit()
        chunks = chunk_source_units([unit], count_tokens=len)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, unit.text)

    def test_whole_unit_preserves_all_provenance(self) -> None:
        """An unsplit chunk retains every public source coordinate."""

        unit = make_unit(
            "αβγ\n",
            path="notes/review.ipynb",
            start_line=8,
            end_line=8,
            kind="notebook_markdown",
            symbol="Introduction",
            section="Methods",
            cell=4,
        )
        chunk, = chunk_source_units([unit], count_tokens=len)
        self.assertIsInstance(chunk, Chunk)
        for field in (
            "path", "text", "start_line", "end_line", "kind",
            "symbol", "section", "cell",
        ):
            with self.subTest(field=field):
                self.assertEqual(
                    getattr(chunk, field), getattr(unit, field)
                )

    def test_chunks_are_frozen_and_slotted(self) -> None:
        """Callers cannot reassign or attach fields to a chunk."""

        chunk, = chunk_source_units(
            [make_unit()], count_tokens=len
        )
        with self.assertRaises(FrozenInstanceError):
            chunk.text = "changed"
        with self.assertRaises((AttributeError, TypeError)):
            chunk.extra_field = "untracked"

    def test_unsplit_units_preserve_order(self) -> None:
        """Top-level units retain their caller-supplied order."""

        units = [
            make_unit("first", path="z.py"),
            make_unit("second", path="a.py"),
            make_unit("third", path="m.py"),
        ]
        chunks = chunk_source_units(units, count_tokens=len)
        self.assertEqual(
            [(item.path, item.text) for item in chunks],
            [(unit.path, unit.text) for unit in units],
        )

    def test_colons_in_metadata_must_not_alias_chunk_identity(self) -> None:
        """Metadata separators must not cause identifier aliasing."""

        first = make_unit(
            "text", kind="markdown_section",
            symbol="alpha:beta", section="gamma",
        )
        second = make_unit(
            "text", kind="markdown_section",
            symbol="alpha", section="beta:gamma",
        )
        first_chunk, = chunk_source_units([first], count_tokens=len)
        second_chunk, = chunk_source_units([second], count_tokens=len)
        self.assertNotEqual(first_chunk.identifier, second_chunk.identifier)

    def test_optional_provenance_values_remain_distinct(self) -> None:
        """An absent symbol or section is distinct from an empty one."""

        for field in ("symbol", "section"):
            with self.subTest(field=field):
                original = make_unit("abc", symbol="x", section="y")
                absent, = chunk_source_units(
                    [replace(original, **{field: None})],
                    count_tokens=len,
                )
                empty, = chunk_source_units(
                    [replace(original, **{field: ""})],
                    count_tokens=len,
                )
                self.assertNotEqual(absent.identifier, empty.identifier)

    def test_unit_above_target_but_within_max_stays_whole(self) -> None:
        """The target guides splits but does not cap intact units."""

        unit = make_unit("abcdef")
        chunks = chunk_source_units(
            [unit], count_tokens=len,
            target_tokens=3, max_tokens=6, overlap_tokens=1,
        )
        self.assertEqual([chunk.text for chunk in chunks], ["abcdef"])

    def test_identifiers_are_stable_and_well_formed(self) -> None:
        """Identical input gives reproducible hexadecimal identifiers."""

        units = [make_unit("abc"), make_unit("def")]
        one = chunk_source_units(units, count_tokens=len)
        two = chunk_source_units(units, count_tokens=len)
        self.assertEqual(
            [chunk.identifier for chunk in one],
            [chunk.identifier for chunk in two],
        )
        for chunk in one:
            self.assertRegex(chunk.identifier, r"\A[0-9a-f]{16}\Z")

    def test_identifiers_include_text_and_provenance(self) -> None:
        """Changing any identity field produces a different identifier."""

        original = make_unit(
            "abc", path="a.py", start_line=2, end_line=2,
            kind="code", symbol="name", section="section", cell=3,
        )
        baseline, = chunk_source_units([original], count_tokens=len)
        changes = (
            {"text": "abd"},
            {"path": "b.py"},
            {"start_line": 1},
            {"end_line": 3},
            {"kind": "text"},
            {"symbol": "other"},
            {"section": "other"},
            {"cell": 4},
        )
        for change in changes:
            with self.subTest(change=change):
                values = {
                    key: Path(value) if key == "path" else value
                    for key, value in change.items()
                }
                updated = replace(original, **values)
                result, = chunk_source_units([updated], count_tokens=len)
                self.assertNotEqual(result.identifier, baseline.identifier)


class TestLineSplitting(unittest.TestCase):
    """Verify multiline boundaries, overlap, and line coordinates."""

    def test_multiple_lines_split_without_overlap(self) -> None:
        """Nonoverlapping chunks retain exact source lines and text."""

        unit = make_unit(
            "ab\ncd\nef\n", start_line=10, end_line=12,
        )
        chunks = chunk_source_units(
            [unit], count_tokens=len,
            target_tokens=3, max_tokens=3, overlap_tokens=0,
        )
        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["ab\n", "cd\n", "ef\n"],
        )
        self.assertEqual(
            [(chunk.start_line, chunk.end_line) for chunk in chunks],
            [(10, 10), (11, 11), (12, 12)],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), unit.text)

    def test_split_prefers_complete_line_groups(self) -> None:
        """Whole source lines are grouped up to the target budget."""

        unit = make_unit("ab\ncd\nef\n", end_line=3)
        chunks = chunk_source_units(
            [unit], count_tokens=len,
            target_tokens=6, max_tokens=6, overlap_tokens=0,
        )
        self.assertEqual(
            [chunk.text for chunk in chunks], ["ab\ncd\n", "ef\n"]
        )
        self.assertEqual(
            [(chunk.start_line, chunk.end_line) for chunk in chunks],
            [(1, 2), (3, 3)],
        )

    def test_whole_line_overlap_repeats_source_lines(self) -> None:
        """Overlap repeats complete lines with correct inclusive ranges."""

        unit = make_unit("A\nB\nC\nD\n", end_line=4, start_line=7)
        chunks = chunk_source_units(
            [unit], count_tokens=len,
            target_tokens=4, max_tokens=4, overlap_tokens=2,
        )
        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["A\nB\n", "B\nC\n", "C\nD\n"],
        )
        self.assertEqual(
            [(chunk.start_line, chunk.end_line) for chunk in chunks],
            [(7, 8), (8, 9), (9, 10)],
        )
        self.assertTrue(all(len(c.text) <= 4 for c in chunks))

    def test_line_overlap_search_rejects_oversized_suffix(self) -> None:
        """Only a suffix within the overlap budget may be repeated."""

        unit = make_unit("A\nB\nC\nD\nE\nF\n", end_line=6)
        chunks = chunk_source_units(
            [unit], count_tokens=len,
            target_tokens=6, max_tokens=6, overlap_tokens=2,
        )
        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["A\nB\nC\n", "C\nD\nE\n", "E\nF\n"],
        )
        self.assertTrue(all(len(chunk.text) <= 6 for chunk in chunks))

    def test_word_count_tokenizer_respects_budget(self) -> None:
        """Chunking also works with a word-based token-count callback."""

        unit = make_unit(
            "one two\nthree four\nfive six\n",
            start_line=10, end_line=12,
        )
        chunks = chunk_source_units(
            [unit], count_tokens=lambda text: len(text.split()),
            target_tokens=2, max_tokens=3, overlap_tokens=1,
        )
        self.assertEqual([c.start_line for c in chunks], [10, 11, 12])
        self.assertEqual(
            [c.text for c in chunks],
            ["one two\n", "three four\n", "five six\n"],
        )
        self.assertTrue(all(len(c.text.split()) <= 3 for c in chunks))

    def test_nonfitting_overlap_does_not_prevent_progress(self) -> None:
        """Lines exceeding the overlap allowance are not repeated."""

        unit = make_unit("abcd\nefgh\nijkl\n", end_line=3)
        chunks = chunk_source_units(
            [unit], count_tokens=len,
            target_tokens=5, max_tokens=5, overlap_tokens=2,
        )
        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["abcd\n", "efgh\n", "ijkl\n"],
        )

    def test_final_chunk_may_exceed_target_but_not_max(self) -> None:
        """A fitting remainder is kept intact even above the target."""

        unit = make_unit("a\nb\ncc\n", end_line=3)
        chunks = chunk_source_units(
            [unit], count_tokens=len,
            target_tokens=2, max_tokens=5, overlap_tokens=0,
        )
        self.assertEqual([chunk.text for chunk in chunks], ["a\n", "b\ncc\n"])
        self.assertTrue(all(len(chunk.text) <= 5 for chunk in chunks))

    def test_oversized_line_among_other_lines_is_split(self) -> None:
        """Character splitting remains available within multiline units."""

        unit = make_unit(
            "abc\n123456789\nxy\n", start_line=20, end_line=22,
            section="Text", symbol="long_function",
        )
        chunks = chunk_source_units(
            [unit], count_tokens=len,
            target_tokens=4, max_tokens=5, overlap_tokens=0,
        )
        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["abc\n", "1234", "5678", "9\n", "xy\n"],
        )
        self.assertEqual(
            [(chunk.start_line, chunk.end_line) for chunk in chunks],
            [(20, 20), (21, 21), (21, 21), (21, 21), (22, 22)],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), unit.text)
        self.assertTrue(all(chunk.symbol == unit.symbol for chunk in chunks))
        self.assertTrue(all(chunk.section == unit.section for chunk in chunks))

    def test_many_line_sizes_preserve_content_without_overlap(self) -> None:
        """Small exhaustive cases preserve nonblank content and budgets.

        The chunker may omit whitespace-only fragments, so the invariant
        compares substantive source text rather than literal newlines.

        """

        for sizes in product(range(1, 5), repeat=3):
            text = "".join(
                chr(65 + index) * size + "\n"
                for index, size in enumerate(sizes)
            )
            unit = make_unit(text, end_line=3)
            for target in range(1, 5):
                for maximum in range(target, 6):
                    with self.subTest(
                        sizes=sizes, target=target, maximum=maximum,
                    ):
                        chunks = chunk_source_units(
                            [unit], count_tokens=len,
                            target_tokens=target,
                            max_tokens=maximum,
                            overlap_tokens=0,
                        )
                        observed = "".join(
                            "".join(chunk.text for chunk in chunks).split()
                        )
                        expected = "".join(text.split())
                        self.assertEqual(observed, expected)
                        self.assertTrue(
                            all(len(chunk.text) <= maximum for chunk in chunks)
                        )

    def test_mixed_newline_styles_retain_original_text(self) -> None:
        """CRLF and LF delimiters survive line-boundary splitting."""

        unit = make_unit("A\r\nB\nC\r\n", end_line=3)
        chunks = chunk_source_units(
            [unit], count_tokens=len,
            target_tokens=3, max_tokens=3, overlap_tokens=0,
        )
        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["A\r\n", "B\n", "C\r\n"],
        )


class TestCharacterSplitting(unittest.TestCase):
    """Verify long-line fragmentation and character-based overlap."""

    def test_long_line_split_with_no_overlap(self) -> None:
        """Long lines preserve all characters when overlap is disabled."""

        unit = make_unit("abcdefg", start_line=8, end_line=8)
        chunks = chunk_source_units(
            [unit], count_tokens=len,
            target_tokens=3, max_tokens=3, overlap_tokens=0,
        )
        self.assertEqual([chunk.text for chunk in chunks],
                         ["abc", "def", "g"])
        self.assertEqual("".join(c.text for c in chunks), unit.text)
        self.assertTrue(all(c.start_line == 8 for c in chunks))
        self.assertTrue(all(c.end_line == 8 for c in chunks))

    def test_long_line_overlap_and_unique_part_ids(self) -> None:
        """Repeated substrings use character offsets for distinct IDs."""

        unit = make_unit("aaaaaaa", start_line=8, end_line=8)
        chunks = chunk_source_units(
            [unit], count_tokens=len,
            target_tokens=3, max_tokens=3, overlap_tokens=1,
        )
        self.assertEqual([chunk.text for chunk in chunks],
                         ["aaa", "aaa", "aaa"])
        self.assertEqual(len({c.identifier for c in chunks}), 3)
        repeated = chunk_source_units(
            [unit], count_tokens=len,
            target_tokens=3, max_tokens=3, overlap_tokens=1,
        )
        self.assertEqual(
            [c.identifier for c in chunks],
            [c.identifier for c in repeated],
        )

    def test_unicode_line_split_at_character_boundaries(self) -> None:
        """Character budgets split Unicode without breaking characters."""

        unit = make_unit("αβγδεζ", path="notes.txt", kind="text")
        chunks = chunk_source_units(
            [unit], count_tokens=len,
            target_tokens=2, max_tokens=2, overlap_tokens=1,
        )
        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["αβ", "βγ", "γδ", "δε", "εζ"],
        )

    def test_one_character_exceeding_target_raises(self) -> None:
        """Unsplittable oversized characters produce an explicit error."""

        unit = make_unit("abc")
        counter = lambda text: 2 if text else 0
        with self.assertRaisesRegex(ValueError, "one character"):
            chunk_source_units(
                [unit], count_tokens=counter,
                target_tokens=1, max_tokens=1, overlap_tokens=0,
            )


class TestHierarchicalSplitting(unittest.TestCase):
    """Check structural child handling and nested source provenance."""

    def test_small_parent_is_kept_even_with_children(self) -> None:
        """Natural parent units take precedence when within max size."""

        child = make_unit("child\n", symbol="Class.method")
        parent = make_unit("parent child\n", children=(child,))
        chunks = chunk_source_units(
            [parent], count_tokens=len,
            target_tokens=6, max_tokens=13, overlap_tokens=0,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, parent.text)
        self.assertEqual(chunks[0].symbol, parent.symbol)

    def test_large_parent_prefers_child_units(self) -> None:
        """Oversized parents are represented by their smaller children."""

        first = make_unit(
            "alpha\n", start_line=1, end_line=1,
            symbol="Class.first", kind="python_method",
        )
        second = make_unit(
            "bravo\n", start_line=2, end_line=2,
            symbol="Class.second", kind="python_method",
        )
        parent = make_unit(
            first.text + second.text, end_line=2,
            symbol="Class", kind="python_class",
            children=(first, second),
        )
        chunks = chunk_source_units(
            [parent], count_tokens=len,
            target_tokens=6, max_tokens=6, overlap_tokens=0,
        )
        self.assertEqual([c.text for c in chunks],
                         [first.text, second.text])
        self.assertEqual([c.symbol for c in chunks],
                         [first.symbol, second.symbol])
        self.assertEqual([c.kind for c in chunks],
                         [first.kind, second.kind])
        self.assertEqual([c.start_line for c in chunks], [1, 2])

    def test_nested_oversized_children_are_split_recursively(self) -> None:
        """Hierarchy recursion reaches grandchildren when needed."""

        grandchild = make_unit(
            "abc\n", start_line=3, end_line=3,
            kind="python_method", symbol="Class.method",
        )
        child = make_unit(
            "abc\n", start_line=3, end_line=3,
            kind="python_class", symbol="Class",
            children=(grandchild,),
        )
        parent = make_unit(
            "abc\n", start_line=3, end_line=3,
            kind="python_module", symbol=None,
            children=(child,),
        )
        chunks = chunk_source_units(
            [parent], count_tokens=len,
            target_tokens=2, max_tokens=3, overlap_tokens=0,
        )
        self.assertEqual([c.text for c in chunks], ["ab", "c\n"])
        self.assertTrue(all(c.symbol == "Class.method" for c in chunks))
        self.assertTrue(all(c.kind == "python_method" for c in chunks))

    def test_blank_children_do_not_generate_chunks(self) -> None:
        """Whitespace-only children are skipped during recursion."""

        blank = make_unit("\n", start_line=1, end_line=1)
        content = make_unit("abcd", start_line=2, end_line=2)
        parent = make_unit(
            "\nabcd", end_line=2, children=(blank, content),
        )
        chunks = chunk_source_units(
            [parent], count_tokens=len,
            target_tokens=3, max_tokens=4, overlap_tokens=0,
        )
        self.assertEqual([c.text for c in chunks], ["abcd"])

    def test_incomplete_children_must_not_discard_parent_text(self) -> None:
        """Keep uncovered parent text when splitting a hierarchy."""

        child = make_unit(
            "    work()\n", start_line=2, end_line=2,
            kind="python_method", symbol="Class.work",
        )
        parent = make_unit(
            "class Class:\n    work()\n", start_line=1, end_line=2,
            kind="python_class", symbol="Class", children=(child,),
        )
        chunks = chunk_source_units(
            [parent], count_tokens=len,
            target_tokens=6, max_tokens=10, overlap_tokens=0,
        )
        expected = "".join(parent.text.split())
        observed = "".join("".join(c.text for c in chunks).split())
        self.assertEqual(
            observed,
            expected,
            "Uncovered parent content must survive hierarchical splitting.",
        )

    def test_missing_child_text_at_any_position_uses_parent(self) -> None:
        """Preserve missing text before, between, or after children."""

        cases = (
            ("header\nfirst\nsecond\n", 2, 3),
            ("first\nmiddle\nsecond\n", 1, 3),
            ("first\nsecond\nfooter\n", 1, 2),
        )
        for text, first_line, second_line in cases:
            with self.subTest(text=text):
                parent = make_unit(
                    text, end_line=3,
                    children=(
                        make_unit("first\n", start_line=first_line,
                                  end_line=first_line),
                        make_unit("second\n", start_line=second_line,
                                  end_line=second_line),
                    ),
                )
                chunks = chunk_source_units(
                    [parent], count_tokens=len,
                    target_tokens=7, max_tokens=9, overlap_tokens=0,
                )
                self.assertEqual(
                    "".join(chunk.text for chunk in chunks), text
                )

    def test_out_of_order_children_use_complete_parent(self) -> None:
        """Children in a different source order cannot reorder the text."""

        first = make_unit("first\n", start_line=1, end_line=1)
        second = make_unit("second\n", start_line=2, end_line=2)
        parent = make_unit(
            "first\nsecond\n", end_line=2,
            children=(second, first),
        )
        chunks = chunk_source_units(
            [parent], count_tokens=len,
            target_tokens=7, max_tokens=7, overlap_tokens=0,
        )
        self.assertEqual("".join(c.text for c in chunks), parent.text)

    def test_duplicate_children_use_complete_parent(self) -> None:
        """Repeated child ranges must not duplicate or omit source text."""

        first = make_unit("first\n", start_line=1, end_line=1)
        parent = make_unit(
            "first\nsecond\n", end_line=2,
            children=(first, first),
        )
        chunks = chunk_source_units(
            [parent], count_tokens=len,
            target_tokens=7, max_tokens=7, overlap_tokens=0,
        )
        self.assertEqual("".join(c.text for c in chunks), parent.text)

    def test_mismatched_child_source_uses_complete_parent(self) -> None:
        """Child provenance from another file must not replace source text."""

        child = make_unit("body\n", path="other.py")
        parent = make_unit(
            "body\nheader\n", end_line=2,
            children=(child,),
        )
        chunks = chunk_source_units(
            [parent], count_tokens=len,
            target_tokens=7, max_tokens=7, overlap_tokens=0,
        )
        self.assertEqual("".join(c.text for c in chunks), parent.text)
        self.assertTrue(all(c.path == parent.path for c in chunks))

    def test_empty_child_uses_complete_parent(self) -> None:
        """Empty children cannot establish valid structural coverage."""

        parent = make_unit(
            "body\nheader\n", end_line=2,
            children=(make_unit(""), make_unit("body\n")),
        )
        chunks = chunk_source_units(
            [parent], count_tokens=len,
            target_tokens=7, max_tokens=7, overlap_tokens=0,
        )
        self.assertEqual("".join(c.text for c in chunks), parent.text)

    def test_blank_gaps_keep_structural_child_provenance(self) -> None:
        """Whitespace-only gaps do not require abandoning hierarchy."""

        first = make_unit("alpha\n", symbol="Class.first")
        second = make_unit("beta\n", start_line=3,
                           end_line=3, symbol="Class.second")
        parent = make_unit(
            "alpha\n\nbeta\n", end_line=3,
            children=(first, second),
        )
        chunks = chunk_source_units(
            [parent], count_tokens=len,
            target_tokens=6, max_tokens=6, overlap_tokens=0,
        )
        self.assertEqual([c.text for c in chunks],
                         [first.text, second.text])
        self.assertEqual([c.symbol for c in chunks],
                         [first.symbol, second.symbol])

    def test_nested_incomplete_children_use_complete_child(self) -> None:
        """A nested gap is preserved even if ancestor coverage is complete."""

        grandchild = make_unit("work()\n", symbol="Class.work")
        child = make_unit(
            "header\nwork()\n", end_line=2,
            symbol="Class", children=(grandchild,),
        )
        parent = make_unit(
            child.text, end_line=2, kind="python_module",
            symbol=None, children=(child,),
        )
        chunks = chunk_source_units(
            [parent], count_tokens=len,
            target_tokens=7, max_tokens=7, overlap_tokens=0,
        )
        self.assertEqual("".join(c.text for c in chunks), parent.text)
        self.assertTrue(all(c.symbol == "Class" for c in chunks))



class TestChunkValidation(unittest.TestCase):
    """Check budget, tokenizer, and callback input validation."""

    def test_invalid_size_settings_raise_value_error(self) -> None:
        """Reject invalid numeric types, ranges, and overlapping budgets."""

        cases = (
            {"target_tokens": 0},
            {"target_tokens": -1},
            {"max_tokens": 0},
            {"target_tokens": 4, "max_tokens": 3},
            {"overlap_tokens": -1},
            {"target_tokens": 3, "overlap_tokens": 3},
            {"target_tokens": 3, "overlap_tokens": 4},
            {"target_tokens": True},
            {"target_tokens": 2.5},
            {"target_tokens": "5"},
            {"target_tokens": None},
            {"max_tokens": False},
            {"max_tokens": 2.5},
            {"max_tokens": "5"},
            {"max_tokens": None},
            {"overlap_tokens": True},
            {"overlap_tokens": 0.5},
            {"overlap_tokens": "1"},
            {"overlap_tokens": None},
        )
        for settings in cases:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    chunk_source_units(
                        [make_unit()], count_tokens=len, **settings,
                    )

    def test_valid_size_boundaries_are_accepted(self) -> None:
        """Minimum target, equal maximum, and zero overlap are valid."""

        chunks = chunk_source_units(
            [make_unit("a")], count_tokens=len,
            target_tokens=1, max_tokens=1, overlap_tokens=0,
        )
        self.assertEqual([c.text for c in chunks], ["a"])

    def test_invalid_token_counter_results_raise_value_error(self) -> None:
        """Token counts must be exact nonnegative Python integers."""

        class IntSubclass(int):
            """Integer subtype for testing strict return-type validation."""

        for result in (-1, True, False, 1.5, "1", None, IntSubclass(1)):
            with self.subTest(result=result):
                with self.assertRaisesRegex(ValueError, "count_tokens"):
                    chunk_source_units(
                        [make_unit("abc")],
                        count_tokens=lambda text: result,
                    )

    def test_zero_token_counter_result_is_valid(self) -> None:
        """The current contract accepts zero for nonempty source text."""

        chunks = chunk_source_units(
            [make_unit("abc")], count_tokens=lambda text: 0,
        )
        self.assertEqual([c.text for c in chunks], ["abc"])

    def test_token_counter_exception_propagates(self) -> None:
        """Tokenizer exceptions are not caught or disguised."""

        def failing_counter(text: str) -> int:
            """Raise to simulate a tokenizer failure."""

            raise RuntimeError("tokenizer unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            chunk_source_units(
                [make_unit("abc")], count_tokens=failing_counter,
            )

    def test_invalid_progress_callback_rejected(self) -> None:
        """Noncallable progress callbacks are rejected even without units."""

        for callback in (False, 7, "not callable"):
            with self.subTest(callback=callback):
                with self.assertRaisesRegex(ValueError, "progress_callback"):
                    chunk_source_units(
                        [], count_tokens=len,
                        progress_callback=callback,
                    )

    def test_count_tokens_is_cached_within_each_call(self) -> None:
        """Identical text is counted once per call, not globally."""

        counter = Mock(side_effect=len)
        units = [
            make_unit("shared", path="one.py"),
            make_unit("shared", path="two.py"),
        ]
        chunk_source_units(units, count_tokens=counter)
        counter.assert_called_once_with("shared")
        chunk_source_units(units, count_tokens=counter)
        self.assertEqual(counter.call_count, 2)

    def test_progress_reports_every_top_level_unit(self) -> None:
        """Progress includes skipped blanks and counts completed units."""

        events: list[tuple[int, int]] = []
        units = [
            make_unit("abc"),
            make_unit("\n"),
            make_unit("abcdef", path="long.py"),
        ]
        chunks = chunk_source_units(
            units, count_tokens=len,
            target_tokens=3, max_tokens=3, overlap_tokens=0,
            progress_callback=lambda current, total:
                events.append((current, total)),
        )
        self.assertEqual(events, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual([c.text for c in chunks],
                         ["abc", "abc", "def"])

    def test_no_progress_for_empty_input(self) -> None:
        """The callback is not invoked when there are no source units."""

        callback = Mock()
        self.assertEqual(
            chunk_source_units(
                [], count_tokens=len, progress_callback=callback,
            ),
            [],
        )
        callback.assert_not_called()

    def test_progress_callback_exception_propagates(self) -> None:
        """A failing callback stops processing without being swallowed."""

        callback = Mock(side_effect=RuntimeError("progress failed"))
        counter = Mock(side_effect=len)
        units = [make_unit("first"), make_unit("second")]
        with self.assertRaisesRegex(RuntimeError, "progress failed"):
            chunk_source_units(
                units, count_tokens=counter,
                progress_callback=callback,
            )
        callback.assert_called_once_with(1, 2)
        counter.assert_called_once_with("first")

    def test_token_failure_does_not_report_completed_unit(self) -> None:
        """No progress notification is emitted for an unfinished unit."""

        callback = Mock()
        with self.assertRaisesRegex(ValueError, "count_tokens"):
            chunk_source_units(
                [make_unit("abc")], count_tokens=lambda text: -1,
                progress_callback=callback,
            )
        callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
