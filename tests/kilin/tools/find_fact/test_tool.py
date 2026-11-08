"""Test the find_fact tool package without live model backends.

Check configuration ownership, public registry entries, settings validation,
and deterministic execution behavior.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from kilin.retrievals.strategies.retrieval_hybrid import HybridResult
from kilin.tools import ToolRequest
from kilin.tools import find_fact as tool
from kilin.tools.find_fact.tool import _cited
from kilin.tools._shared.planning import PLAN_CHOICE_QUICK

from tests.kilin.tools._contract_support import (
    assert_packaged_defaults,
    assert_registry_entry,
)
from tests.kilin.tools._shared._support import chunk


def _quick_context(passages, terms):
    """Return tool services that choose the quick answer."""

    reply = json.dumps({"terms": list(terms)})
    return SimpleNamespace(
        chunks=passages,
        ask_router=Mock(return_value=reply),
        rank=Mock(return_value=[
            HybridResult(1.0, passage, ("semantic",))
            for passage in reversed(passages)
        ]),
        retrieval_count=2,
        confirm=Mock(return_value=PLAN_CHOICE_QUICK),
        write=Mock(return_value=(
            'It is "implemented in MATLAB with C++" [1].'
        )),
        progress=Mock(),
        done=Mock(),
        trace=SimpleNamespace(emit=Mock(), include_content=True),
        trace_turn_id="test",
        model_id="fake",
    )


class TestFindFact(unittest.TestCase):
    """Verify independent configuration and offline functionality."""

    def test_registry_uses_package_entry_points(self) -> None:
        """Resolve parser and runner to the packaged implementation.

        """

        assert_registry_entry(
            self, "find_fact", tool
        )

    def test_packaged_defaults_and_project_type_overrides(self) -> None:
        """Use the tool-local JSON when settings are not supplied.

        """

        assert_packaged_defaults(
            self, tool, tool.FactSettings,
        )

    def test_invalid_project_type_settings(self) -> None:
        """Reject out-of-range, mistyped and unknown overrides.

        """

        invalid_settings = (
            {'part_tokens': 15},
            {'max_model_calls': 0},
            {"unknown": True},
        )
        for values in invalid_settings:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_fact_prompts_distinguish_quick_and_full(self) -> None:
        """Distinguish ranked passages from complete-document notes.

        """

        _, full = tool.fact_prompt(
            "Which parameter?", [("paper.md", ["[3] value 12"])],
            None,
        )
        _, quick = tool.fact_prompt(
            "Which parameter?", [("paper.md", ["[3] value 12"])],
            "Be brief", quick=True,
        )
        self.assertIn("Notes by document", full)
        self.assertIn("Passages by document", quick)
        self.assertIn("Be brief", quick)

    def test_evidence_uses_only_valid_bounded_citations(self) -> None:
        """Reject oversized ranges without admitting partial evidence.

        """

        self.assertEqual(_cited("Found [1, 3-9]", 3), {1})
        self.assertEqual(_cited("Found [1-99999999]", 2), set())
        self.assertEqual(_cited("Found [1–2]", 2), {1, 2})

    def test_invalid_ranges_are_reported_in_tool_summary(self) -> None:
        """Preserve numeric diagnostics and report invalid range parts.

        """

        passages = [
            chunk("a", "An observation.", path="a.md"),
            chunk("b", "Another observation.", path="a.md"),
        ]
        context = _quick_context(passages, ["unmatched phrase"])
        context.write.return_value = "Supported [1, 2-9] and [7]."
        result = tool.run(
            tool.FactSettings(),
            ToolRequest("Which evidence?", "Which evidence?"),
            context,
        )
        self.assertEqual(result.summary["invalid_citations"], [7])
        self.assertEqual(
            result.summary["invalid_citation_groups"], ["2-9"]
        )
        self.assertEqual([e["citation"] for e in result.evidence], [1])

    def test_quick_answer_ranks_when_no_passage_has_the_terms(self) -> None:
        """Answer from ranked passages when the exact terms are absent.

        """

        passages = [
            chunk("a", "Introduction to brain networks.", path="a.md"),
            chunk("b", "Graph measures and statistics.", path="a.md"),
            chunk("c", "Implemented in MATLAB with C++ parts.", path="a.md"),
        ]
        question = "Which languages does metadata table C6 list?"
        context = _quick_context(passages, ["metadata table C6"])

        result = tool.run(
            tool.FactSettings(), ToolRequest(question, question), context
        )

        context.write.assert_called_once()
        prompt = context.write.call_args.args[1]
        self.assertIn("Passages by document", prompt)
        # Ranked passages are numbered 1, 2... in reading order.
        self.assertIn("[1] Graph measures", prompt)
        self.assertIn("[2] Implemented in MATLAB", prompt)
        self.assertNotIn("Introduction to brain networks", prompt)
        # The quote is in passage [2], so the citation moves there.
        self.assertIn('C++" [2].', result.answer)
        self.assertEqual(
            result.summary["quote_citations"], {"moved": 1, "unplaced": 0}
        )
        self.assertIn("best-ranked passages", result.answer)
        self.assertIn("“metadata table C6”", result.answer)
        self.assertEqual(
            result.summary["terms_without_hits"], ["metadata table C6"]
        )
        self.assertEqual(
            [(item["citation"], item["text"]) for item in result.evidence],
            [(2, "Implemented in MATLAB with C++ parts.")],
        )
        plan = context.confirm.call_args.args[0]
        self.assertEqual(plan.quick_label, "Quick answer")
        self.assertIn("No passage mentions", plan.quick_note)


if __name__ == "__main__":
    unittest.main()
