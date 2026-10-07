"""Check whole-document reading: target choice, budget, and order."""

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.chunking import Chunk
from qachat.retrievals import (
    CodeEntity,
    CodeGraph,
    HybridResult,
    estimate_tokens,
    read_document,
)

_EMPTY_GRAPH = CodeGraph(entities=(), relations=())


def _chunk(path: str, number: int, text: str, symbol=None) -> Chunk:
    return Chunk(
        identifier=f"{path}:{number}",
        path=Path(path),
        text=text,
        start_line=number * 10,
        end_line=number * 10 + 9,
        kind="text",
        symbol=symbol,
        section=None,
    )


def _ranked(*chunks: Chunk) -> list[HybridResult]:
    return [
        HybridResult(score=1.0 / (rank + 60), chunk=chunk, channels=("x",))
        for rank, chunk in enumerate(chunks, start=1)
    ]


class TestDocumentReading(unittest.TestCase):
    """One target is read whole, or its most relevant parts within budget."""

    def setUp(self) -> None:
        self.paper = [
            _chunk("papers/patil25deep.pdf", n, f"DIVA part {n}. " * 20)
            for n in range(1, 6)
        ]
        self.other = [
            _chunk("papers/mijalkov17braph.pdf", n, f"BRAPH part {n}. " * 20)
            for n in range(1, 4)
        ]
        self.chunks = self.paper + self.other

    def test_dominant_file_is_read_whole_in_source_order(self) -> None:
        ranked = _ranked(self.paper[3], self.other[0], self.paper[1])
        selection = read_document(
            "check the DIVA paper entirely",
            self.chunks,
            _EMPTY_GRAPH,
            ranked,
            max_tokens=10_000,
        )
        self.assertEqual(selection.path, Path("papers/patil25deep.pdf"))
        self.assertEqual(selection.reason, "search")
        self.assertTrue(selection.complete)
        self.assertEqual(list(selection.chunks), self.paper)

    def test_named_file_wins_over_search_results(self) -> None:
        ranked = _ranked(self.paper[0], self.paper[1], self.other[2])
        selection = read_document(
            "read mijalkov17braph from start to end",
            self.chunks,
            _EMPTY_GRAPH,
            ranked,
            max_tokens=10_000,
        )
        self.assertEqual(selection.path, Path("papers/mijalkov17braph.pdf"))
        self.assertEqual(selection.reason, "named_file")
        self.assertEqual(list(selection.chunks), self.other)

    def test_long_document_keeps_most_relevant_parts_in_order(self) -> None:
        size = estimate_tokens(self.paper[0].text)
        ranked = _ranked(self.paper[4], self.paper[1], self.paper[0])
        selection = read_document(
            "which language does the DIVA paper use",
            self.chunks,
            _EMPTY_GRAPH,
            ranked,
            max_tokens=2 * size,
        )
        self.assertFalse(selection.complete)
        self.assertEqual(selection.total_chunks, 5)
        # The two best-ranked parts, in the document's own order.
        self.assertEqual(
            list(selection.chunks), [self.paper[1], self.paper[4]]
        )
        self.assertEqual(selection.estimated_tokens, 2 * size)

    def test_named_function_is_read_alone(self) -> None:
        fit = [
            _chunk("model.py", 1, "def fit(x):", symbol="fit"),
            _chunk("model.py", 2, "    return x", symbol="fit"),
        ]
        other = _chunk("model.py", 3, "def predict(x):", symbol="predict")
        graph = CodeGraph(
            entities=(
                CodeEntity("fit", "python_function", Path("model.py")),
                CodeEntity("predict", "python_function", Path("model.py")),
            ),
            relations=(),
        )
        selection = read_document(
            "review the whole fit function",
            [*fit, other],
            graph,
            _ranked(other),
            max_tokens=10_000,
        )
        self.assertEqual(selection.reason, "named_function")
        self.assertEqual(selection.functions, ("fit",))
        self.assertEqual(list(selection.chunks), fit)

    def test_papers_named_by_author_or_author_year(self) -> None:
        from qachat.retrievals.document_reading import named_paths

        papers = [
            Path(f"repo/{name}.pdf")
            for name in ("yao25connecting", "patil25deep", "mijalkov17braph")
        ]

        def named(question):
            return [path.stem for path in named_paths(
                question, papers, authors=True
            )]

        self.assertEqual(named("a very short summary on yao25"),
                         ["yao25connecting"])
        self.assertEqual(named("what does Yao's paper say"),
                         ["yao25connecting"])
        self.assertEqual(named("turn patil25 into a briefing"),
                         ["patil25deep"])
        self.assertEqual(named("can you summarize the paper"), [])
        self.assertEqual(named("the 2025 papers"), [])

    def test_nothing_to_read(self) -> None:
        self.assertIsNone(
            read_document("q", [], _EMPTY_GRAPH, [], max_tokens=1_000)
        )
        self.assertIsNone(
            read_document("q", self.chunks, _EMPTY_GRAPH, [], max_tokens=1)
        )


if __name__ == "__main__":
    unittest.main()
