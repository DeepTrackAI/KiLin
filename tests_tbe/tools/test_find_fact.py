"""Check find_fact: read everything (or quickly), answer from notes."""

from __future__ import annotations

import unittest

from qachat.retrievals import HybridResult
from qachat.tools import ToolRequest
from qachat.tools.find_fact import FactSettings, run
from tests.qachat.tools.helpers import chunk, context


def _files():
    return [
        chunk("repo/a.pdf", ["Methods.", "We trained for 50 epochs."]),
        chunk("repo/a.pdf", ["Results were good."], start=3),
        chunk("repo/b.pdf", ["Another study with 20 epochs."]),
    ]


class TestFindFact(unittest.TestCase):
    """The user always chooses between reading everything and quick."""

    def ask(self, choice, question="how many epochs do the papers use?"):
        files = _files()
        plans, notes_prompts, answers = [], [], []

        def model(system, prompt, tokens):
            notes_prompts.append(prompt)
            return "- 50 epochs [2]" if "a.pdf" in prompt else "- 20 [3]"

        def write(system, prompt, tokens):
            answers.append(prompt)
            return "a.pdf used 50 epochs [2]; b.pdf used 20 [4]."

        built, _ = context(
            files,
            model=model,
            write=write,
            rank=lambda text: [
                HybridResult(score=0.1, chunk=files[0], channels=("x",))
            ],
            confirm=lambda plan: plans.append(plan) or choice,
        )
        result = run(FactSettings(), ToolRequest(question, question), built)
        return result, plans, notes_prompts, answers

    def test_it_always_asks_with_a_time_estimate(self) -> None:
        result, plans, notes, _ = self.ask("cancel")
        self.assertEqual(len(plans), 1)
        self.assertGreater(plans[0].estimated_seconds, 0)
        self.assertEqual(plans[0].quick_label, "Quick answer")
        self.assertIn("1 best-ranked passage;", plans[0].quick_note)
        self.assertEqual(notes, [])
        self.assertIn("Cancelled", result.answer)

    def test_continue_reads_every_file_and_answers_per_file(self) -> None:
        result, _, notes, answers = self.ask("continue")
        self.assertEqual(len(notes), 2)  # One part per file.
        self.assertIn("a.pdf:\n- 50 epochs [2]", answers[0])
        self.assertIn("b.pdf:\n", answers[0])
        self.assertEqual(result.summary["passages_read"], 3)
        # Only cited passages are evidence, with their numbers kept.
        self.assertEqual([e["citation"] for e in result.evidence], [2])

    def test_quick_answers_from_ranked_passages_only(self) -> None:
        result, _, notes, answers = self.ask("quick")
        self.assertEqual(notes, [])
        self.assertIn("[1] Methods.", answers[0])
        self.assertNotIn("Another study", answers[0])
        self.assertIn("Quick answer from the 1 best-ranked", result.answer)


if __name__ == "__main__":
    unittest.main()
