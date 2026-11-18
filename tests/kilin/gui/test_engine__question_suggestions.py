"""Test the event that sends suggested questions to the page.

The engine sends the project's questions and welcome headings just before
``opened``. A failure must not stop the project from opening: the page then
keeps the project type's fixed suggestions and welcome prompts.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from kilin.gui.engine import ChatEngine
from kilin.question_suggestions import Suggestion

PROJECT = SimpleNamespace(root="/projects/demo")


def _engine(session: object) -> SimpleNamespace:
    """Return the engine state the event needs, recording sent events."""

    events: list[dict[str, object]] = []
    return SimpleNamespace(
        session=session,
        active_trace=Mock(),
        events=events,
        _send=lambda event, **data: events.append({"event": event, **data}),
    )


class TestQuestionSuggestionsEvent(unittest.TestCase):
    """Send questions, or an empty list after a failure."""

    def test_questions_are_sent_with_their_intents(self) -> None:
        """Send each question with the intent it is meant to reach."""

        session = Mock()
        session.question_suggestions.return_value = [
            Suggestion("paper_overview", "Give me a short summary of a.pdf.")
        ]
        session.welcome_prompts.return_value = ["What about sleep?"]
        engine = _engine(session)
        ChatEngine._send_question_suggestions(engine, PROJECT)
        self.assertEqual(
            engine.events,
            [
                {
                    "event": "question_suggestions",
                    "project": "/projects/demo",
                    "questions": [
                        {
                            "intent": "paper_overview",
                            "question": "Give me a short summary of a.pdf.",
                        }
                    ],
                    "welcome": ["What about sleep?"],
                }
            ],
        )

    def test_failure_sends_no_questions_and_is_traced(self) -> None:
        """Trace a failure and let the page keep its fixed suggestions."""

        session = Mock()
        session.question_suggestions.side_effect = RuntimeError("no model")
        engine = _engine(session)
        ChatEngine._send_question_suggestions(engine, PROJECT)
        self.assertEqual(
            engine.events,
            [
                {
                    "event": "question_suggestions",
                    "project": "/projects/demo",
                    "questions": [],
                    "welcome": [],
                }
            ],
        )
        name = engine.active_trace.emit.call_args.args[0]
        self.assertEqual(name, "gui.question_suggestions.failed")


if __name__ == "__main__":
    unittest.main()
