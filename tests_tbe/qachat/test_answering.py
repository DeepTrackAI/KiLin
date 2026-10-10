"""Check intent routing, history selection, and turn representations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qachat.answering import (
    RouteDecision,
    history_messages,
    refers_back,
    route_question,
    search_text,
    select_history,
    turn_representation,
)
from qachat.project_type import project_type
from qachat.rag import Evidence


def _turn(question: str, answer: str, citations=(1,)):
    evidence = tuple(
        Evidence(number, Path(f"src/file{number}.py"), 1, 2, "fit", "text")
        for number in citations
    )
    return SimpleNamespace(
        question=question, answer=answer, evidence=evidence, parameters={}
    )


class TestRouting(unittest.TestCase):
    """The router's reply is validated; configuration decides behavior."""

    def setUp(self) -> None:
        self.kind = project_type("code")
        self.turns = [_turn(f"q{n}", f"a{n} [1].") for n in range(1, 5)]

    def _route(self, reply: str) -> RouteDecision:
        with patch("qachat.answering.side_reply", return_value=reply):
            return route_question(object(), self.kind, self.turns, "next?")

    def test_valid_reply_is_used(self) -> None:
        decision = self._route(
            'Sure: {"intent": "code_generation", "standalone": false, '
            '"related_turns": [1, 9, 2, 2]}'
        )
        self.assertEqual(decision.intent, "code_generation")
        self.assertEqual(decision.related_turn_ids, (1, 2))
        self.assertTrue(decision.refers_back)
        self.assertIsNone(decision.fallback)

    def test_unknown_intent_and_bad_json_fall_back(self) -> None:
        unknown = self._route('{"intent": "poetry", "standalone": true}')
        self.assertEqual(unknown.intent, self.kind.default_intent)
        self.assertIn("poetry", unknown.fallback)
        broken = self._route("REUSE")
        self.assertEqual(broken.intent, self.kind.default_intent)
        self.assertEqual(broken.related_turn_ids, ())

    def test_small_talk_shortcut_skips_the_router_model(self) -> None:
        reply = '{"intent": "code_explanation", "standalone": true}'
        with patch(
            "qachat.answering.side_reply", return_value=reply
        ) as router:
            decision = route_question(
                object(), self.kind, self.turns, "How are you today?"
            )
            mixed = route_question(
                object(), self.kind, self.turns, "hi, what does fit do?"
            )
        self.assertEqual(decision.intent, "small_talk")
        self.assertIsNotNone(decision.shortcut)
        self.assertIsNone(mixed.shortcut)
        router.assert_called_once()

    def test_router_errors_do_not_stop_answers(self) -> None:
        with patch("qachat.answering.side_reply", side_effect=OSError("x")):
            decision = route_question(object(), self.kind, self.turns, "q")
        self.assertEqual(decision.intent, self.kind.default_intent)
        self.assertIn("router failed", decision.fallback)


class TestRelatedTurnChecks(unittest.TestCase):
    """Related turns need a sign in the question; search uses user words."""

    def setUp(self) -> None:
        self.kind = project_type("literature")
        greeting = _turn("how are you", "Fine, thanks.", ())
        greeting.parameters = {"intent": "small_talk"}
        self.turns = [
            greeting,
            _turn("what is NOVA", "NOVA analyses infrared spectra [1]."),
        ]

    def _route(self, question: str, related: str) -> tuple:
        reply = (
            '{"intent": "question_answering", "standalone": false, '
            f'"related_turns": {related}}}'
        )
        with patch(
            "qachat.answering.side_reply", return_value=reply
        ) as router:
            decision = route_question(
                object(), self.kind, self.turns, question
            )
        return decision, router.call_args.args[2]

    def test_question_naming_its_own_topic_drops_unrelated_turns(
        self,
    ) -> None:
        decision, prompt = self._route("what is RFE", "[1, 2]")
        self.assertEqual(decision.related_turn_ids, ())
        self.assertTrue(decision.standalone)
        self.assertEqual(
            [number for number, _ in decision.dropped_turns], [1, 2]
        )
        self.assertIn("ignores the conversation", decision.dropped_turns[0][1])
        # Small talk is not offered to the router at all.
        self.assertNotIn("Turn 1:", prompt)
        self.assertIn("what is NOVA", prompt)
        self.assertEqual(
            search_text("what is RFE", self.turns, decision), "what is RFE"
        )

    def test_shared_term_keeps_a_turn_without_extending_the_search(
        self,
    ) -> None:
        decision, _ = self._route("what are NOVA's limitations", "[2]")
        self.assertEqual(decision.related_turn_ids, (2,))
        self.assertFalse(decision.refers_back)
        self.assertEqual(
            search_text("what are NOVA's limitations", self.turns, decision),
            "what are NOVA's limitations",
        )

    def test_pointing_back_adds_the_related_question_to_the_search(
        self,
    ) -> None:
        decision, _ = self._route("how do I install it?", "[2]")
        self.assertEqual(decision.related_turn_ids, (2,))
        self.assertEqual(
            search_text("how do I install it?", self.turns, decision),
            "how do I install it? what is NOVA",
        )

    def test_refers_back_cues(self) -> None:
        for question in (
            "how do I install it?",
            "explain the second point",
            "what do you think so far?",
            "and for MATLAB?",
            "Compare the methods you mentioned",
            "can you compare the two methods we discussed?",
            "why?",
        ):
            self.assertTrue(refers_back(question), question)
        for question in (
            "what is RFE",
            "why NOV A not NOVA?",
            "can you explain recursive feature elimination",
        ):
            self.assertFalse(refers_back(question), question)


class TestHistory(unittest.TestCase):
    """Recent turns plus related older turns, as question and answer only."""

    def test_selection_policies(self) -> None:
        related = RouteDecision("x", False, related_turn_ids=(2, 9))
        standalone = RouteDecision("x", True)
        self.assertEqual(
            select_history(10, "relevant", 2, related), [2, 9, 10]
        )
        # A standalone question still sees the previous turn.
        self.assertEqual(
            select_history(10, "relevant", 2, standalone), [10]
        )
        self.assertEqual(select_history(10, "recent", 2, standalone), [9, 10])
        self.assertEqual(select_history(10, "none", 2, related), [])
        self.assertEqual(select_history(0, "relevant", 2, related), [])

    def test_history_has_no_evidence_and_neutral_citations(self) -> None:
        turns = [_turn("What is fit?", "<think>hm</think>It trains [1].")]
        self.assertEqual(
            history_messages(turns, [1]),
            [
                {"role": "user", "content": "What is fit?"},
                {"role": "assistant", "content": "It trains [source]."},
            ],
        )

    def test_representation_lists_cited_sources_and_first_sentence(
        self,
    ) -> None:
        turn = _turn("Explain fit", "Fit trains the model [2]. More.", (1, 2))
        representation = turn_representation(turn, 7)
        self.assertEqual(representation["turn_id"], 7)
        self.assertEqual(representation["sources"], ["file2.py:fit"])
        self.assertEqual(representation["summary"], "Fit trains the model.")


if __name__ == "__main__":
    unittest.main()
