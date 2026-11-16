"""Test exclusion of rejected turns from routing and answer history.

These tests use stable one-based turn numbers and no model inference.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest
from unittest.mock import patch

from kilin import answering
from kilin.project_type import (
    HISTORY_POLICY_RECENT,
    HISTORY_POLICY_RELEVANT,
    project_type,
)
from tests.kilin._rag_support import _turn


class TestDislikedHistorySelection(unittest.TestCase):
    def setUp(self):
        self.turns = [
            _turn("Earlier useful topic", "Useful answer"),
            _turn("Rejected private topic", "REJECTED ANSWER"),
            _turn("Later useful topic", "Recent answer"),
        ]
        self.excluded = frozenset((2,))

    def test_recent_window_counts_eligible_turns_without_renumbering(self):
        decision = answering.RouteDecision("x", False)
        self.assertEqual(answering.select_history(
            3, HISTORY_POLICY_RECENT, 2, decision,
            excluded_turn_ids=self.excluded,
        ), [1, 3])
        self.assertEqual(answering.select_history(
            3, HISTORY_POLICY_RELEVANT, 2, decision,
            excluded_turn_ids=self.excluded,
        ), [1, 3])
        self.assertEqual(answering.select_history(
            3, HISTORY_POLICY_RELEVANT, 2,
            answering.RouteDecision("x", True),
            excluded_turn_ids=frozenset((3,)),
        ), [2])
        self.assertEqual(answering.select_history(
            3, HISTORY_POLICY_RECENT, 2, decision,
            excluded_turn_ids=frozenset((1, 2, 3)),
        ), [])

    def test_history_omits_whole_pair_without_changing_saved_turns(self):
        messages = answering.history_messages(
            self.turns, [1, 2, 3], excluded_turn_ids=self.excluded
        )
        contents = [item["content"] for item in messages]
        self.assertEqual(contents, [
            "Earlier useful topic", "Useful answer",
            "Later useful topic", "Recent answer",
        ])
        self.assertEqual(self.turns[1].answer, "REJECTED ANSWER")
        self.assertEqual(len(self.turns), 3)

    def test_search_expansion_does_not_reuse_rejected_question(self):
        decision = answering.RouteDecision(
            "x", False, related_turn_ids=(1, 2), refers_back=True
        )
        query = answering.search_text(
            "Tell me more", self.turns, decision,
            excluded_turn_ids=self.excluded,
        )
        self.assertIn("Earlier useful topic", query)
        self.assertNotIn("Rejected private topic", query)

    def test_router_omits_rejected_summaries_and_related_numbers(self):
        config = project_type("literature")
        reply = (
            '{"intent": "question_answering", "standalone": false, '
            '"related_turns": [1, 2, 3]}'
        )
        with patch.object(answering, "side_reply", return_value=reply) as rt:
            result = answering.route_question(
                object(), config, self.turns, "What did you say earlier?",
                excluded_turn_ids=self.excluded,
            )
        prompt = rt.call_args.args[2]
        self.assertIn("Turn 1:", prompt)
        self.assertIn("Turn 3:", prompt)
        self.assertNotIn("Turn 2:", prompt)
        self.assertNotIn("REJECTED ANSWER", prompt)
        self.assertNotIn("Rejected private topic", prompt)
        self.assertEqual(result.related_turn_ids, (1, 3))
        self.assertIn((2, "turn ignores the conversation"),
                      result.dropped_turns)

    def test_router_reuse_fallback_skips_last_rejected_turn(self):
        config = project_type("literature")
        reply = (
            '{"intent": "conversation_followup", "standalone": false, '
            '"related_turns": []}'
        )
        with patch.object(answering, "side_reply", return_value=reply):
            decision = answering.route_question(
                object(), config, self.turns, "Continue that answer",
                excluded_turn_ids=frozenset((3,)),
            )
        # The reuse fallback selects the latest eligible earlier turn.
        self.assertEqual(decision.intent, "conversation_followup")
        self.assertEqual(decision.related_turn_ids, (2,))

    def test_router_failure_without_eligible_turns_is_standalone(self):
        with patch.object(
            answering, "side_reply", side_effect=RuntimeError("offline")
        ):
            decision = answering.route_question(
                object(), project_type("literature"), self.turns,
                "Continue that answer",
                excluded_turn_ids=frozenset((1, 2, 3)),
            )
        self.assertTrue(decision.standalone)
        self.assertEqual(decision.related_turn_ids, ())


if __name__ == "__main__":
    unittest.main()
