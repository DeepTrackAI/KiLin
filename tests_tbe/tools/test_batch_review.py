"""Check batch_review: parts in order, checked notes, streamed answer."""

from __future__ import annotations

import unittest

from qachat.retrievals import HybridResult
from qachat.tools import ToolRequest
from qachat.tools.batch_review import (
    ReviewSettings,
    check_citations,
    clean_notes,
    plan_parts,
    run,
)
from tests.qachat.tools.helpers import chunk, context


def _ranked(*chunks):
    return [
        HybridResult(score=1 / (rank + 60), chunk=item, channels=("x",))
        for rank, item in enumerate(chunks, start=1)
    ]


class TestBatchReview(unittest.TestCase):
    """The model takes notes per part; Python keeps them honest."""

    def setUp(self) -> None:
        # Five passages of about 250 tokens each in one paper.
        self.paper = [
            chunk("repo/smith2020.pdf", [f"Part {n}. " + "x" * 1000], start=n)
            for n in range(1, 6)
        ]
        self.other = [chunk("repo/other.md", ["unrelated"])]

    def test_parts_keep_order_and_budget(self) -> None:
        self.assertEqual(plan_parts(self.paper, 600), [[0, 1], [2, 3], [4]])

    def test_notes_must_cite_passages_of_their_part(self) -> None:
        reply = (
            "- Uses Python 3.11 [2]\n- Elsewhere [9]\n- No citation\n"
            "* Data from 2019 [1, 2]"
        )
        self.assertEqual(
            clean_notes(reply, [1, 2]),
            ["- Uses Python 3.11 [2]", "- Data from 2019 [1, 2]"],
        )
        self.assertEqual(clean_notes("NOTHING", [1]), [])

    def test_invalid_citations_are_reported(self) -> None:
        self.assertEqual(check_citations("A [1] and [3, 7].", 5), [7])

    def test_reads_every_part_and_streams_the_answer(self) -> None:
        prompts = []

        def model(system, prompt, tokens):
            prompts.append(prompt)
            return "- The code is written in Python [3]" if "[3]" in (
                prompt
            ) else "NOTHING"

        written = []

        def write(system, prompt, tokens):
            written.append(prompt)
            return "It uses Python [3]."

        built, events = context(
            [*self.paper, *self.other],
            model=model,
            write=write,
            rank=lambda text: _ranked(self.paper[2], self.paper[0]),
        )
        request = ToolRequest(
            "which language does it use?", "which language", "Be brief."
        )
        result = run(ReviewSettings(part_tokens=1000), request, built)
        self.assertTrue(result.streamed)
        self.assertEqual(result.answer, "It uses Python [3].")
        self.assertEqual(len(prompts), 2)  # 5 passages in 2 parts.
        self.assertIn("- The code is written in Python [3]", written[0])
        self.assertIn("Part 2:\nNOTHING", written[0])
        self.assertTrue(written[0].endswith("Be brief."))
        self.assertEqual(len(result.evidence), 5)
        self.assertEqual(result.summary["passages_read"], 5)
        self.assertEqual(result.summary["parts_without_notes"], [2])
        self.assertEqual(result.summary["invalid_citations"], [])
        self.assertIn(
            ("done", "Read 5 of 5 passages of smith2020.pdf"),
            [event[:2] for event in events],
        )
        # Progress says the whole paper is read, part by part.
        self.assertIn(
            ("progress", "Reading the whole paper (part 2 of 2): "
             "smith2020.pdf"),
            [event[:2] for event in events],
        )

    def test_quick_reads_the_most_relevant_part_only(self) -> None:
        built, events = context(
            self.paper,
            model=lambda system, prompt, tokens: "NOTHING",
            write=lambda system, prompt, tokens: "Not stated.",
            rank=lambda text: _ranked(self.paper[4], self.paper[1]),
            confirm=lambda plan: "quick",
        )
        settings = ReviewSettings(part_tokens=600, large_batches=0)
        result = run(settings, ToolRequest("q", "q"), built)
        self.assertEqual(result.summary["choice"], "quick")
        self.assertIn(
            "Reading the most relevant passages: smith2020.pdf",
            [message for _, message, _ in events],
        )
        self.assertEqual(result.summary["passages_read"], 2)
        self.assertIn("Read 2 of 5 passages", result.answer)

    def test_cancel_makes_no_model_calls(self) -> None:
        def model(system, prompt, tokens):
            raise AssertionError("no model call expected")

        built, _ = context(
            self.paper,
            model=model,
            write=model,
            rank=lambda text: _ranked(self.paper[0]),
            confirm=lambda plan: "cancel",
        )
        settings = ReviewSettings(part_tokens=600, large_batches=0)
        result = run(settings, ToolRequest("q", "q"), built)
        self.assertIn("Cancelled before reading", result.answer)
        self.assertEqual(result.evidence, ())


    def test_an_unnamed_document_is_asked_not_guessed(self) -> None:
        asked = []

        def choose(question, options):
            asked.append((question, options))
            return None  # The user cancels.

        built, _ = context(
            self.paper + self.other,
            rank=lambda text: _ranked(self.paper[0], self.other[0]),
            choose=choose,
        )
        question = "read the whole thing and list the software"
        result = run(ReviewSettings(), ToolRequest(question, question), built)
        (text, options), = asked
        self.assertEqual(text, "Which document should I read in full?")
        self.assertEqual(
            options[0], "smith2020.pdf (best match for the question)"
        )
        self.assertIn("Tell me which one you mean", result.answer)
        # A named document is read without asking.
        asked.clear()
        built, _ = context(
            self.paper + self.other,
            rank=lambda text: _ranked(self.paper[0]),
            confirm=lambda plan: "cancel",
            choose=choose,
        )
        question = "read Smith's paper in full"
        run(ReviewSettings(), ToolRequest(question, question), built)
        self.assertEqual(asked, [])

    def test_quick_reads_only_passages_that_mention_the_term(self) -> None:
        paper = [
            chunk("repo/smith2020.pdf", [f"Part {n}. " + "x" * 1000],
                  start=n)
            for n in range(1, 5)
        ] + [chunk("repo/smith2020.pdf", ["We analysed it with XYZ 2."],
                   start=9)]
        plans, read = [], []

        def model(system, prompt, tokens):
            read.append(prompt)
            return "- Uses XYZ 2 [5]"

        built, _ = context(
            paper,
            router=lambda system, prompt, tokens: (
                '{"terms": ["XYZ2"], "document": null}'
            ),
            model=model,
            write=lambda system, prompt, tokens: "Yes, with XYZ 2 [5].",
            rank=lambda text: _ranked(paper[0]),
            confirm=lambda plan: plans.append(plan) or "quick",
        )
        question = "read Smith's paper: does it use xyz2 to analyse data?"
        settings = ReviewSettings(confirmation="always")
        result = run(settings, ToolRequest(question, question), built)
        self.assertEqual(plans[0].quick_label, "Exact matches only")
        self.assertEqual(plans[0].quick_note, "1 passage mentions “XYZ2”")
        self.assertEqual(len(read), 1)  # One part: the matching passage.
        self.assertIn("We analysed it with XYZ 2.", read[0])
        self.assertNotIn("Part 1.", read[0])
        self.assertIn("those that mention XYZ2", result.answer)

    def test_quick_without_mentions_answers_without_the_model(self) -> None:
        def model(system, prompt, tokens):
            raise AssertionError("no model call expected")

        built, _ = context(
            self.paper,
            router=lambda system, prompt, tokens: (
                '{"terms": ["XYZ2"], "document": null}'
            ),
            model=model,
            write=model,
            rank=lambda text: _ranked(self.paper[0]),
            confirm=lambda plan: "quick",
        )
        question = "read Smith's paper: does it use xyz2?"
        settings = ReviewSettings(confirmation="always")
        result = run(settings, ToolRequest(question, question), built)
        self.assertIn("No passage of smith2020.pdf mentions “XYZ2”",
                      result.answer)

if __name__ == "__main__":
    unittest.main()
