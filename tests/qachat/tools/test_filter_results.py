"""Check filter_results: judge the items of an earlier list."""

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.tools import ToolRequest
from qachat.tools.contract import EarlierTurn
from qachat.tools.filter_results import FilterSettings, run
from tests.qachat.tools.helpers import context


def _listed() -> EarlierTurn:
    return EarlierTurn(
        2,
        "list every mention of XYZ in that paper",
        (Path("a.pdf"), Path("a.pdf"), Path("b.pdf")),
        (
            "We analysed the data with XYZ 2.",
            "Doe, A. XYZ: a framework. 2020.",
            "XYZ 2 computes graph measures.",
        ),
        (10, 40, 3),
    )


class TestFilterResults(unittest.TestCase):
    """Each listed item is judged; nothing is searched again."""

    def ask(self, reply, earlier, settings=FilterSettings()):
        prompts = []

        def model(system, prompt, tokens):
            prompts.append(prompt)
            return reply

        built, _ = context(model=model)
        question = "among those, which describe using the software?"
        request = ToolRequest(question, question, earlier=earlier)
        return run(settings, request, built), prompts

    def test_matching_items_are_listed_with_locations(self) -> None:
        result, prompts = self.ask("1, 3", (_listed(),))
        self.assertIn("1. We analysed the data with XYZ 2.", prompts[0])
        self.assertIn("**2 of 3** items from turn 2 match.", result.answer)
        self.assertIn("a.pdf, line 10:", result.answer)
        self.assertIn("b.pdf, line 3:", result.answer)
        self.assertNotIn("Doe, A.", result.answer)
        self.assertEqual(
            [(e["citation"], e["start_line"]) for e in result.evidence],
            [(1, 10), (2, 3)],
        )

    def test_numbers_outside_the_batch_are_ignored(self) -> None:
        result, _ = self.ask("2, 7, 0", (_listed(),))
        self.assertIn("**1 of 3**", result.answer)
        self.assertEqual(result.summary["matched"], 1)

    def test_small_batches_judge_every_item(self) -> None:
        settings = FilterSettings(batch_tokens=500, item_characters=100)
        result, prompts = self.ask("NONE", (_listed(),), settings)
        self.assertGreaterEqual(len(prompts), 1)
        self.assertIn("None of the items matches.", result.answer)

    def test_the_related_list_wins_over_a_newer_unlinked_turn(self) -> None:
        newer = EarlierTurn(
            3, "why is it used?", (Path("c.pdf"),), ("Other text.",), (1,),
            linked=False,
        )
        result, prompts = self.ask("1", (newer, _listed()))
        self.assertIn("We analysed the data with XYZ 2.", prompts[0])
        self.assertNotIn("Other text.", prompts[0])
        self.assertIn("items from turn 2", result.answer)

    def test_without_an_earlier_list_it_explains(self) -> None:
        empty = EarlierTurn(1, "hello")
        result, prompts = self.ask("1", (empty,))
        self.assertEqual(prompts, [])
        self.assertIn("couldn't find an earlier answer", result.answer)
        self.assertEqual(result.evidence, ())


if __name__ == "__main__":
    unittest.main()
