"""Check paper_overview: the paper is settled first, then summarized."""

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.retrievals import HybridResult
from qachat.tools import EarlierTurn, ToolRequest
from qachat.tools.paper_overview import OverviewSettings, run
from tests.qachat.tools.helpers import chunk, context


def _papers():
    return [
        chunk(f"repo/{name}.pdf", [f"{name} title.", "Abstract text."])
        for name in ("tiunn26braph2", "yao25connecting", "patil25deep")
    ]


class TestPaperOverview(unittest.TestCase):
    """Short summaries never guess which paper is meant."""

    def summarize(self, question, earlier=(), choose=None, router=None):
        papers = _papers()
        written = []

        def write(system, prompt, tokens):
            written.append(prompt)
            return "It studies networks [1]."

        built, _ = context(
            papers,
            write=write,
            router=router,
            rank=lambda text: [
                HybridResult(score=0.1, chunk=c, channels=("x",))
                for c in papers
            ],
            choose=choose,
        )
        request = ToolRequest(question, question, earlier=tuple(earlier))
        return run(OverviewSettings(), request, built), written

    def test_a_named_paper_is_summarized_from_its_own_passages(self) -> None:
        result, written = self.summarize("a short summary of yao25")
        self.assertEqual(result.summary["paper"], "repo/yao25connecting.pdf")
        self.assertIn("Passages of yao25connecting.pdf", written[0])
        self.assertNotIn("tiunn26braph2 title", written[0])
        self.assertTrue(result.streamed)

    def test_another_paper_in_any_wording_is_asked(self) -> None:
        earlier = [
            EarlierTurn(1, "turn Tiunn's paper into a briefing",
                        (Path("repo/tiunn26braph2.pdf"),) * 5)
        ]
        for question in (
            "i want another short summary for another paper",
            "now do a different study",
            "and the next one?",
        ):
            with self.subTest(question=question):
                asked = []
                result, _ = self.summarize(
                    question,
                    earlier,
                    choose=lambda q, options: asked.append(options) or 0,
                    # The router model judges the wording.
                    router=lambda system, prompt, tokens: "DIFFERENT",
                )
                options, = asked
                self.assertFalse(any("tiunn26braph2" in o for o in options))
                self.assertNotEqual(
                    result.summary["paper"], "repo/tiunn26braph2.pdf"
                )

    def test_the_same_paper_is_used_without_asking(self) -> None:
        earlier = [
            EarlierTurn(1, "turn Tiunn's paper into a briefing",
                        (Path("repo/tiunn26braph2.pdf"),) * 5)
        ]
        asked = []
        result, _ = self.summarize(
            "now a short summary of it",
            earlier,
            choose=lambda q, options: asked.append(options) or 0,
            router=lambda system, prompt, tokens: "SAME",
        )
        self.assertEqual(asked, [])
        self.assertEqual(result.summary["paper"], "repo/tiunn26braph2.pdf")

    def test_cancel_asks_the_user_to_name_the_paper(self) -> None:
        result, written = self.summarize(
            "summarize one of the papers", choose=lambda q, o: None
        )
        self.assertEqual(written, [])
        self.assertIn("Tell me which one you mean", result.answer)


if __name__ == "__main__":
    unittest.main()
