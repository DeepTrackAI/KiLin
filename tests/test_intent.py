"""Tests for deterministic repository-question planning."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.chunking import Chunk
from qachat.intent import plan_question


def _chunk(
    identifier: str,
    path: str,
    *,
    symbol: str | None = None,
) -> Chunk:
    """Create a small chunk for planner tests.

    Parameters
    ----------
    identifier: str
        Test identifier.
    path: str
        Relative source path.
    symbol: str | None, optional
        Optional source symbol.

    Returns
    -------
    Chunk
        Test chunk.

    """

    return Chunk(
        identifier=identifier,
        path=Path(path),
        text="source",
        start_line=1,
        end_line=1,
        kind="python_function",
        symbol=symbol,
        section=None,
    )


class TestIntent(unittest.TestCase):

    def test_plan_question(self) -> None:
        chunks = [
            _chunk("one", "qachat/runtime.py", symbol="choose_backend"),
            _chunk("two", "tests/runtime.py", symbol="test_backend"),
            _chunk("three", "src/Example.m", symbol="Example.compute"),
        ]
        cases = (
            (
                "How does backend selection work?",
                "general",
                ("semantic", "lexical"),
            ),
            (
                "Explain qachat/runtime.py",
                "file",
                (Path("qachat/runtime.py"),),
            ),
            (
                "What does choose_backend return?",
                "symbol",
                ("choose_backend",),
            ),
            (
                "What calls Example.compute?",
                "relationship",
                ("Example.compute",),
            ),
            (
                "Explain the whole repository architecture",
                "architecture",
                ("semantic", "lexical", "metadata", "graph"),
            ),
            (
                "How many functions are there?",
                "exact",
                ("graph", "metadata"),
            ),
        )

        for question, expected_kind, expected in cases:
            with self.subTest(question=question):
                plan = plan_question(question, chunks)
                self.assertEqual(plan.kind, expected_kind)
                combined = (
                    plan.matched_paths or plan.matched_symbols or plan.methods
                )
                self.assertEqual(combined, expected)
                self.assertTrue(plan.reason)

        duplicate_names = plan_question("Explain runtime.py", chunks)
        self.assertEqual(
            duplicate_names.matched_paths,
            (Path("qachat/runtime.py"), Path("tests/runtime.py")),
        )

        for invalid in ("", "   ", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    plan_question(invalid, chunks)

        with self.assertRaises(ValueError):
            plan_question("Question", "not chunks")


if __name__ == "__main__":
    unittest.main()
