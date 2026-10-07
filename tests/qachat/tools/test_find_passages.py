"""Check find_passages, and that term search uses the question's words."""

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.tools import ToolRequest
from qachat.tools.find_passages import PassageSettings, run, split_sentences
from qachat.tools.scan_and_list import (
    ScanSettings,
    collect_mentions,
    scan_mentions,
)
from tests.qachat.tools.helpers import chunk, context

_ONE = [
    "Graph theory measures describe how brain regions connect.",
    "We used MATLAB for all analyses in this study.",
    "Network analysis showed hubs in the default mode network.",
]
_TWO = [
    "The weather was sunny during data collection days.",
    "Small-world networks balance segregation and integration.",
]


class TestFindPassages(unittest.TestCase):
    """The model picks sentence numbers; Python copies the sentences."""

    def chunks(self):
        return [chunk("repo/one.md", _ONE), chunk("repo/two.md", _TWO)]

    def test_overlapping_passages_give_each_sentence_once(self) -> None:
        overlap = chunk("repo/one.md", _ONE[1:], start=2)
        sentences = split_sentences([chunk("repo/one.md", _ONE), overlap])
        self.assertEqual(len(sentences), 3)
        self.assertEqual([s.line for s in sentences], [1, 2, 3])

    def test_every_sentence_is_read_and_copied_word_for_word(self) -> None:
        prompts = []

        def model(system, prompt, tokens):
            prompts.append(prompt)
            # Sentences 1, 3, 5 are about networks; 99 does not exist.
            return "1, 3, 5, 99"

        built, _ = context(
            self.chunks(),
            router=lambda system, prompt, tokens: (
                '{"topic": "network analysis", "document": null}'
            ),
            model=model,
        )
        question = "list the sentences that talk about network analysis"
        result = run(PassageSettings(), ToolRequest(question, question), built)
        self.assertEqual(len(prompts), 1)  # One batch holds all five.
        self.assertIn("Found **3 sentences**", result.answer)
        self.assertIn(f"“{_ONE[0]}” (line 1)", result.answer)
        self.assertIn(f"“{_TWO[1]}” (line 2)", result.answer)
        self.assertNotIn("MATLAB", result.answer)
        self.assertEqual(result.summary["read"], 5)
        self.assertEqual(len(result.evidence), 2)

    def test_a_request_for_files_lists_matching_files(self) -> None:
        prompts = []

        def model(system, prompt, tokens):
            prompts.append(prompt)
            return "1, 3"  # Both sentences from one.md.

        built, _ = context(
            self.chunks(),
            router=lambda system, prompt, tokens: (
                '{"topic": "braph2 as an analysis tool", "document": null}'
            ),
            model=model,
        )
        question = (
            "i want a list of files that are using braph2 as analysis tool"
        )
        result = run(PassageSettings(), ToolRequest(question, question), built)
        # The judgement follows the user's request, not only the topic.
        self.assertIn(f"The user asked: {question}", prompts[0])
        self.assertIn("**1 of 2 files** match", result.answer)
        self.assertIn("- **one.md** (2 sentences)", result.answer)
        self.assertNotIn("two.md", result.answer)

    def test_topic_falls_back_to_the_question(self) -> None:
        built, _ = context(
            self.chunks(),
            router=lambda system, prompt, tokens: "not json",
            model=lambda system, prompt, tokens: "NONE",
        )
        question = "which sentences are talking about small-world networks?"
        result = run(PassageSettings(), ToolRequest(question, question), built)
        self.assertEqual(result.summary["topic"], "small-world networks")
        self.assertIn("No sentence in scope", result.answer)


class TestTermsFromTheQuestion(unittest.TestCase):
    """Exact-term search uses the question's words, and spelling variants."""

    def test_earlier_terms_are_not_added(self) -> None:
        chunks = [chunk("repo/a.md", ["BRAPH2 does network analysis."])]
        built, _ = context(
            chunks,
            router=lambda system, prompt, tokens: (
                '{"terms": ["network analysis", "BRAPH2"], "document": null}'
            ),
        )
        request = ToolRequest(
            "list every mention of network analysis",
            "list every mention of network analysis what is BRAPH2",
        )
        found = collect_mentions(
            ScanSettings(), request, built, tool="scan_and_list"
        )
        self.assertEqual(found.scan.terms, ("network analysis",))

    def test_spelling_variants_are_one_term(self) -> None:
        chunks = [
            chunk(
                "repo/a.md",
                ["BRAPH2 and BRAPH 2 and braph-2 are one tool.",
                 "BRAPH 20 is a different number."],
            )
        ]
        scan = scan_mentions(chunks, ["braph2"])
        self.assertEqual(scan.term_counts(), {"braph2": 3})
        self.assertIn(Path("repo/a.md"), scan.counts())


if __name__ == "__main__":
    unittest.main()
