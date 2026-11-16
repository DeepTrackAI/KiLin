"""Unit tests for routing, side replies, and conversation history.

Tests use packaged project types and deterministic fake model sessions. They
exercise observable routing contracts and malformed-input recovery without
loading models or accessing the network.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kilin import answering
from kilin.chat import ASSISTANT_ROLE, SYSTEM_ROLE, USER_ROLE
from kilin.project_type import (
    HISTORY_POLICY_NONE,
    HISTORY_POLICY_RECENT,
    HISTORY_POLICY_RELEVANT,
    project_type,
)
from kilin.rag import Evidence


def _turn(
    question: str,
    answer: str,
    citations: tuple[int, ...] = (1,),
    *,
    intent: str | None = None,
) -> SimpleNamespace:
    """Construct an earlier turn without a model or retrieval index.

    Parameters
    ----------
    question: str
        User question retained for the conversation.
    answer: str
        Model answer, possibly containing numbered citations.
    citations: tuple[int, ...], optional
        Evidence citation numbers. Defaults to ``(1,)``.
    intent: str | None, optional
        Stored intent identifier. Defaults to ``None``.

    Returns
    -------
    SimpleNamespace
        Minimal turn with question, answer, evidence, and parameters.

    """

    evidence = tuple(
        Evidence(number, Path(f"src/file{number}.py"), 1, 2, "fit", "text")
        for number in citations
    )
    parameters = {"intent": intent} if intent is not None else {}
    return SimpleNamespace(
        question=question,
        answer=answer,
        evidence=evidence,
        parameters=parameters,
    )


class _FakeChat:
    """Simulate side replies while retaining mutable conversation state.

    Attributes
    ----------
    messages: list[dict[str, str]]
        In-memory chat history, including the original system instruction.
    thinking: bool
        Current generation setting.
    temperature: float
        Current sampling temperature.
    response: str
        Text returned by the fake reply operation.
    reply_error: Exception | None
        Error raised instead of generating a reply, when supplied.
    observations: list[dict[str, object]]
        State and arguments captured at the time of each reply.

    """

    def __init__(
        self,
        *,
        response: str = "<think>draft</think>Final reply.",
        reply_error: Exception | None = None,
    ) -> None:
        """Initialize a fake chat with nondefault generation settings.

        Parameters
        ----------
        response: str, optional
            Generated text. Defaults to a reply with a thinking block.
        reply_error: Exception | None, optional
            Exception raised by ``reply``. Defaults to ``None``.

        """

        self.messages = [
            {"role": SYSTEM_ROLE, "content": "Original instructions"},
            {"role": USER_ROLE, "content": "Earlier question"},
            {"role": ASSISTANT_ROLE, "content": "Earlier answer"},
        ]
        self.thinking = True
        self.temperature = 0.6
        self.response = response
        self.reply_error = reply_error
        self.observations: list[dict[str, object]] = []

    def restore_history(self, messages: list[dict[str, str]]) -> None:
        """Copy replacement messages to the mutable conversation state.

        Parameters
        ----------
        messages: list[dict[str, str]]
            Replacement chat messages.

        """

        self.messages = [dict(message) for message in messages]

    def configure_generation(self, settings: dict[str, object]) -> None:
        """Apply the generation settings supplied by ``side_reply``.

        Parameters
        ----------
        settings: dict[str, object]
            Settings containing thinking and temperature.

        """

        self.thinking = bool(settings["thinking"])
        self.temperature = float(settings["temperature"])

    def reply(
        self,
        message: str,
        *,
        max_new_tokens: int | None = None,
        token_callback: Callable[[str], None] | None = None,
        language_context: tuple[str, ...] = (),
    ) -> str:
        """Capture the side-request state and simulate a model response.

        Parameters
        ----------
        message: str
            The side question passed to the model.
        max_new_tokens: int | None, optional
            Explicit generation length. Defaults to ``None``.
        token_callback: object, optional
            Receives generated text when supplied. Defaults to ``None``.
        language_context: tuple[str, ...], optional
            Original user messages for script policy. Defaults to ``()``.

        Returns
        -------
        str
            Configured fake response.

        """

        self.observations.append(
            {
                "message": message,
                "max_new_tokens": max_new_tokens,
                "thinking": self.thinking,
                "temperature": self.temperature,
                "history": [dict(item) for item in self.messages],
                "token_callback": token_callback,
                "language_context": language_context,
            }
        )
        self.messages.append({"role": USER_ROLE, "content": message})
        if self.reply_error is not None:
            raise self.reply_error
        self.messages.append(
            {"role": ASSISTANT_ROLE, "content": self.response}
        )
        if token_callback is not None:
            token_callback(self.response)
        return self.response


class TestPackagedAnsweringDefaults(unittest.TestCase):
    """Test that routing uses the validated packaged answering settings."""

    def test_configured_limits_match_runtime_constants(self) -> None:
        """Keep numerical routing limits consistent with the JSON values."""

        path = Path(answering.__file__).with_name("kilin_defaults.json")
        config = json.loads(path.read_text(encoding="utf-8"))["ANSWERING"]
        names = {
            "MAX_RELATED_TURNS": answering.MAX_RELATED_TURNS,
            "MAX_LISTED_TURNS": answering.MAX_LISTED_TURNS,
            "ROUTER_MAX_TOKENS": answering._ROUTER_MAX_TOKENS,
            "SUMMARY_MAX_TOKENS": answering._SUMMARY_MAX_TOKENS,
            "SUMMARY_ANSWER_MAX_CHARACTERS": (
                answering._SUMMARY_ANSWER_MAX_CHARACTERS
            ),
            "REPRESENTATION_MAX_CHARACTERS": (
                answering._REPRESENTATION_MAX_CHARACTERS
            ),
            "REPRESENTATION_MAX_SOURCES": (
                answering._REPRESENTATION_MAX_SOURCES
            ),
            "SHORT_QUESTION_MAX_WORDS": answering._SHORT_QUESTION_MAX_WORDS,
        }
        for name, current in names.items():
            with self.subTest(name=name):
                self.assertEqual(current, config[name])
                self.assertGreater(current, 0)

    def test_configured_language_heuristics_match_runtime(self) -> None:
        """Compile exactly the stored regex and use the stored vocabulary."""

        path = Path(answering.__file__).with_name("kilin_defaults.json")
        config = json.loads(path.read_text(encoding="utf-8"))["ANSWERING"]
        self.assertEqual(
            answering._REFERS_BACK.pattern, config["REFERS_BACK_PATTERN"]
        )
        self.assertEqual(
            answering._COMMON_WORDS, frozenset(config["COMMON_WORDS"])
        )
        self.assertEqual(
            answering._REFERS_BACK.flags & re.I, re.I
        )
        self.assertTrue(answering.refers_back("Frob quux?"))
        self.assertFalse(answering.refers_back("Frob quux baz?"))

    def test_section_has_only_documented_answering_keys(self) -> None:
        """Prevent unreviewed default keys or accidental format additions."""

        path = Path(answering.__file__).with_name("kilin_defaults.json")
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(config["ANSWERING"]),
            {
                "MAX_RELATED_TURNS",
                "MAX_LISTED_TURNS",
                "ROUTER_MAX_TOKENS",
                "SUMMARY_MAX_TOKENS",
                "SUMMARY_ANSWER_MAX_CHARACTERS",
                "REPRESENTATION_MAX_CHARACTERS",
                "REPRESENTATION_MAX_SOURCES",
                "SHORT_QUESTION_MAX_WORDS",
                "REFERS_BACK_PATTERN",
                "COMMON_WORDS",
            },
        )
        self.assertNotIn("EMBEDDING", config)
        self.assertNotIn("DEFAULT_QUERY_FORMAT", config)


class TestStripThinking(unittest.TestCase):
    """Test extraction and removal of model-generated thinking text."""

    def test_extracts_complete_and_template_opened_thinking(self) -> None:
        """Read thinking separately from answer text in common formats."""

        self.assertEqual(
            answering.extract_thinking(
                "<think>First</think>Body<think>Second</think>"
            ),
            "First\n\nSecond",
        )
        self.assertEqual(
            answering.extract_thinking("Template reasoning</think>Answer"),
            "Template reasoning",
        )
        self.assertEqual(
            answering.extract_thinking("Implicit</think><think>More</think>"),
            "Implicit\n\nMore",
        )

    def test_extracts_open_thinking_without_inventing_answer(self) -> None:
        """Keep an unfinished thought without retaining ordinary text."""

        self.assertEqual(
            answering.extract_thinking("Text<think>unfinished"),
            "unfinished",
        )
        self.assertEqual(answering.extract_thinking("Plain answer"), "")
        self.assertEqual(answering.extract_thinking(""), "")

    def test_removes_complete_thinking_blocks(self) -> None:
        """Remove multiple reasoning blocks without removing answer text."""

        self.assertEqual(
            answering.strip_thinking(
                "<think>reason 1</think>Answer. <think>reason 2</think>Done."
            ),
            "Answer. Done.",
        )

    def test_handles_template_opened_thinking(self) -> None:
        """Discard text before a closing tag without an opening tag."""

        self.assertEqual(
            answering.strip_thinking("model reasoning</think> Result"),
            "Result",
        )

    def test_discards_unclosed_thinking_and_strips_whitespace(self) -> None:
        """Do not leak text following an unfinished thinking opener."""

        self.assertEqual(answering.strip_thinking("Text <think>draft"), "Text")
        self.assertEqual(answering.strip_thinking("  Answer  "), "Answer")


class TestSideReply(unittest.TestCase):
    """Test restoration of model state after temporary side requests."""

    def test_side_reply_restores_history_and_generation_settings(self) -> None:
        """Use greedy nonthinking generation and restore the caller state."""

        chat = _FakeChat()
        original = [dict(message) for message in chat.messages]

        result = answering.side_reply(
            chat, "Side instructions", "Classify this", max_new_tokens=23
        )

        self.assertEqual(result, "Final reply.")
        self.assertEqual(chat.messages, original)
        self.assertTrue(chat.thinking)
        self.assertEqual(chat.temperature, 0.6)
        self.assertEqual(
            chat.observations,
            [
                {
                    "message": "Classify this",
                    "max_new_tokens": 23,
                    "thinking": False,
                    "temperature": 0.0,
                    "history": [
                        {"role": SYSTEM_ROLE, "content": "Side instructions"}
                    ],
                    "token_callback": None,
                    "language_context": (),
                }
            ],
        )

    def test_tool_side_reply_inherits_generation_settings(self) -> None:
        """Inherit thinking/temperature and cap the tool output budget."""

        chat = _FakeChat()
        chat.max_new_tokens = 64
        original = [dict(message) for message in chat.messages]
        result = answering.side_reply(
            chat, "Tool system", "Tool question", max_new_tokens=250,
            inherit_settings=True,
        )
        self.assertEqual(result, "Final reply.")
        self.assertEqual(chat.messages, original)
        observation = chat.observations[0]
        self.assertTrue(observation["thinking"])
        self.assertEqual(observation["temperature"], 0.6)
        self.assertEqual(observation["max_new_tokens"], 64)
        # Router calls retain their existing deterministic behavior.
        answering.side_reply(
            chat, "Router", "Choose", max_new_tokens=30,
        )
        self.assertFalse(chat.observations[1]["thinking"])
        self.assertEqual(chat.observations[1]["temperature"], 0.0)

    def test_visible_tool_reply_preserves_thinking_and_stream(self) -> None:
        """Return the raw tool answer without altering the streamed text."""

        chat = _FakeChat()
        history = [dict(message) for message in chat.messages]
        streamed: list[str] = []

        answer = answering.side_reply(
            chat, "Tool", "Summarize", max_new_tokens=120,
            token_callback=streamed.append, inherit_settings=True,
            preserve_thinking=True,
        )

        self.assertEqual(answer, "<think>draft</think>Final reply.")
        self.assertEqual(streamed, [answer])
        self.assertEqual(chat.messages, history)
        self.assertTrue(chat.thinking)
        self.assertEqual(chat.temperature, 0.6)
        self.assertTrue(chat.observations[0]["thinking"])

    def test_internal_tool_reply_strips_thinking_by_default(self) -> None:
        """Do not expose thinking from tool calls used internally."""

        chat = _FakeChat()
        answer = answering.side_reply(
            chat, "Tool", "Inspect", max_new_tokens=120,
            inherit_settings=True,
        )

        self.assertEqual(answer, "Final reply.")
        self.assertTrue(chat.observations[0]["thinking"])

    def test_tool_side_reply_inherits_token_budget_by_default(self) -> None:
        """No tool cap: use the active user-selected output allowance."""

        chat = _FakeChat()
        chat.max_new_tokens = 4096
        original = [dict(message) for message in chat.messages]
        result = answering.side_reply(
            chat, "Tool system", "Check the quotation",
            max_new_tokens=None, inherit_settings=True,
        )
        self.assertEqual(result, "Final reply.")
        self.assertEqual(chat.messages, original)
        observation = chat.observations[0]
        self.assertEqual(observation["max_new_tokens"], 4096)
        self.assertTrue(observation["thinking"])
        self.assertEqual(observation["temperature"], 0.6)

        # Thinking bypasses a local cap, while respecting the user maximum.
        answering.side_reply(
            chat, "Tool system", "Check the quotation",
            max_new_tokens=2048, inherit_settings=True,
        )
        self.assertEqual(chat.observations[1]["max_new_tokens"], 4096)
        answering.side_reply(
            chat, "Tool system", "Check the quotation",
            max_new_tokens=8192, inherit_settings=True,
        )
        self.assertEqual(chat.observations[2]["max_new_tokens"], 4096)

    def test_tool_side_reply_restores_history_after_failure(self) -> None:
        """An inherited tool reply cannot corrupt the conversation."""

        chat = _FakeChat(reply_error=RuntimeError("model failed"))
        original = [dict(message) for message in chat.messages]
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            answering.side_reply(
                chat, "Tool", "Question", max_new_tokens=120,
                inherit_settings=True,
            )
        self.assertEqual(chat.messages, original)
        self.assertTrue(chat.thinking)
        self.assertEqual(chat.temperature, 0.6)

    def test_side_reply_passes_token_callback(self) -> None:
        """Forward streaming output without retaining side messages."""

        chat = _FakeChat(response="Streamed reply")
        received: list[str] = []
        original = [dict(message) for message in chat.messages]

        callback = received.append
        result = answering.side_reply(
            chat,
            "Side instructions",
            "Side question",
            max_new_tokens=9,
            token_callback=callback,
        )

        self.assertEqual(result, "Streamed reply")
        self.assertEqual(received, ["Streamed reply"])
        self.assertIs(chat.observations[0]["token_callback"], callback)
        self.assertEqual(chat.messages, original)

    def test_reply_exception_restores_history_and_settings(self) -> None:
        """Restore the original state even after partial reply mutation."""

        chat = _FakeChat(reply_error=RuntimeError("generation failed"))
        original = [dict(message) for message in chat.messages]

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            answering.side_reply(
                chat, "Side instructions", "Question", max_new_tokens=12
            )

        self.assertEqual(chat.messages, original)
        self.assertTrue(chat.thinking)
        self.assertEqual(chat.temperature, 0.6)

    def test_token_callback_exception_restores_history(self) -> None:
        """Do not leak an in-progress side reply after callback failure."""

        chat = _FakeChat(response="Streamed reply")
        original = [dict(message) for message in chat.messages]

        def failing_callback(_text: str) -> None:
            """Represent a callback that rejects generated output."""

            raise RuntimeError("stream callback failed")

        with self.assertRaisesRegex(RuntimeError, "stream callback failed"):
            answering.side_reply(
                chat,
                "Side instructions",
                "Side question",
                max_new_tokens=8,
                token_callback=failing_callback,
            )

        self.assertEqual(chat.messages, original)
        self.assertTrue(chat.thinking)
        self.assertEqual(chat.temperature, 0.6)

    def test_repeated_side_replies_never_accumulate_history(self) -> None:
        """Begin each independent side request with only its own system."""

        chat = _FakeChat(response="Result")
        original = [dict(message) for message in chat.messages]

        for number in (1, 2):
            answer = answering.side_reply(
                chat, f"System {number}", "Question", max_new_tokens=8
            )
            self.assertEqual(answer, "Result")
            self.assertEqual(chat.messages, original)

        self.assertEqual(len(chat.observations), 2)
        self.assertEqual(
            [item["history"][0]["content"] for item in chat.observations],
            ["System 1", "System 2"],
        )

    def test_setting_configuration_exception_restores_history(self) -> None:
        """Restore history when configuring the side request fails."""

        chat = _FakeChat()
        original = [dict(message) for message in chat.messages]
        original_configure = chat.configure_generation
        failure = RuntimeError("invalid generation settings")
        calls = 0

        def configure(settings: dict[str, object]) -> None:
            """Fail the first configuration, then allow restoration."""

            nonlocal calls
            calls += 1
            if calls == 1:
                raise failure
            original_configure(settings)

        with patch.object(chat, "configure_generation", side_effect=configure):
            with self.assertRaisesRegex(RuntimeError, "invalid generation"):
                answering.side_reply(
                    chat, "System", "Question", max_new_tokens=8
                )

        self.assertEqual(chat.messages, original)
        self.assertEqual(chat.observations, [])
        self.assertEqual(calls, 2)


class TestRouting(unittest.TestCase):
    """Test router output validation and deterministic fallback choices."""

    def setUp(self) -> None:
        """Create code-project turns without loading a language model."""

        self.config = project_type("code")
        self.turns = [_turn(f"q{n}", f"a{n} [1].") for n in range(1, 5)]

    def _route(
        self, reply: str, question: str = "next?"
    ) -> answering.RouteDecision:
        """Return a decision using deterministic mocked router output.

        Parameters
        ----------
        reply: str
            Model-produced router response.
        question: str, optional
            User question being classified. Defaults to ``'next?'``.

        Returns
        -------
        answering.RouteDecision
            Checked router decision.

        """

        with patch.object(answering, "side_reply", return_value=reply):
            return answering.route_question(
                object(), self.config, self.turns, question
            )

    def test_valid_reply_is_used(self) -> None:
        """Use a valid intent and deduplicate bounded turn references."""

        decision = self._route(
            'Sure: {"intent": "code_generation", "standalone": false, '
            '"related_turns": [1, 9, 2, 2]}'
        )
        self.assertEqual(decision.intent, "code_generation")
        self.assertEqual(decision.related_turn_ids, (1, 2))
        self.assertTrue(decision.refers_back)
        self.assertFalse(decision.standalone)
        self.assertIsNone(decision.fallback)
        self.assertIn("code_generation", decision.raw)

    def test_unknown_intent_falls_back(self) -> None:
        """Preserve the default intent and explain an unknown identifier."""

        decision = self._route('{"intent": "poetry", "standalone": true}')
        self.assertEqual(decision.intent, self.config.default_intent)
        self.assertIn("poetry", decision.fallback)
        self.assertTrue(decision.standalone)

    def test_non_string_intents_fall_back_without_raising(self) -> None:
        """Reject non-string model intent IDs without failing routing."""

        for value in ([], {}, ["code_explanation"], 3, False, None):
            with self.subTest(value=value):
                reply = json.dumps(
                    {"intent": value, "standalone": True}
                )
                decision = self._route(reply)
                self.assertEqual(decision.intent, self.config.default_intent)
                self.assertEqual(decision.related_turn_ids, ())
                self.assertEqual(
                    decision.fallback, "intent was not a string"
                )

    def test_non_list_related_turns_fall_back_without_raising(self) -> None:
        """Reject scalar and mapping related-turn fields as invalid JSON."""

        for value in (3, "2", {"turn": 2}, True, 1.5):
            with self.subTest(value=value):
                reply = json.dumps(
                    {
                        "intent": "code_generation",
                        "standalone": False,
                        "related_turns": value,
                    }
                )
                decision = self._route(reply)
                self.assertEqual(decision.intent, self.config.default_intent)
                self.assertEqual(decision.related_turn_ids, ())
                self.assertEqual(
                    decision.fallback, "related_turns was not a list"
                )

    def test_null_related_turns_is_treated_as_absent(self) -> None:
        """Allow null in the optional related-turn field for compatibility."""

        reply = (
            '{"intent": "code_generation", "standalone": false, '
            '"related_turns": null}'
        )
        decision = self._route(reply)
        self.assertEqual(decision.intent, "code_generation")
        self.assertEqual(decision.related_turn_ids, ())
        self.assertIsNone(decision.fallback)

    def test_non_boolean_standalone_uses_safe_default(self) -> None:
        """Use position-based standalone defaults for malformed flags."""

        reply = (
            '{"intent": "code_generation", "standalone": "false"}'
        )
        decision = self._route(reply)
        self.assertFalse(decision.standalone)
        self.assertEqual(
            decision.fallback, "standalone was not a boolean"
        )

    def test_invalid_json_falls_back_without_related_turns(self) -> None:
        """Use the default intent for non-JSON or non-object replies."""

        for reply in ("REUSE", "[1, 2]", '{"intent": missing}'):
            with self.subTest(reply=reply):
                decision = self._route(reply)
                self.assertEqual(decision.intent, self.config.default_intent)
                self.assertEqual(decision.related_turn_ids, ())
                self.assertEqual(decision.fallback, "reply was not JSON")

    def test_invalid_related_turn_ids_are_excluded(self) -> None:
        """Reject booleans, strings, duplicates, and out-of-range numbers."""

        decision = self._route(
            '{"intent": "code_generation", "standalone": false, '
            '"related_turns": [false, true, -1, 0, "2", 2, 2, 99]}'
        )
        self.assertEqual(decision.related_turn_ids, (2,))
        self.assertIsNone(decision.fallback)

    def test_router_uses_newest_three_related_turns(self) -> None:
        """Bound related turns by the configured maximum and recency."""

        decision = self._route(
            '{"intent": "code_explanation", "standalone": false, '
            '"related_turns": [4, 1, 3, 2]}'
        )
        self.assertEqual(
            decision.related_turn_ids,
            (2, 3, 4),
        )
        self.assertEqual(
            len(decision.related_turn_ids), answering.MAX_RELATED_TURNS
        )

    def test_missing_standalone_defaults_by_conversation_position(
        self,
    ) -> None:
        """A missing standalone flag is true only without earlier turns."""

        raw = '{"intent": "code_explanation"}'
        decision = self._route(raw)
        self.assertFalse(decision.standalone)

        with patch.object(answering, "side_reply", return_value=raw):
            first = answering.route_question(
                object(), self.config, [], "Explain the repository"
            )
        self.assertTrue(first.standalone)

    def test_related_ids_override_a_true_standalone_flag(self) -> None:
        """Treat explicit related turns as a nonstandalone decision."""

        decision = self._route(
            '{"intent": "code_explanation", "standalone": true, '
            '"related_turns": [2]}'
        )
        self.assertEqual(decision.related_turn_ids, (2,))
        self.assertFalse(decision.standalone)

    def test_small_talk_shortcut_skips_the_router_model(self) -> None:
        """Only whole-message greetings skip classification."""

        reply = '{"intent": "code_explanation", "standalone": true}'
        with patch.object(
            answering, "side_reply", return_value=reply
        ) as router:
            decision = answering.route_question(
                object(), self.config, self.turns, "How are you today?"
            )
            mixed = answering.route_question(
                object(), self.config, self.turns, "hi, what does fit do?"
            )

        self.assertEqual(decision.intent, "small_talk")
        self.assertTrue(decision.standalone)
        self.assertIsNotNone(decision.shortcut)
        self.assertIsNone(mixed.shortcut)
        router.assert_called_once()

    def test_follow_up_tools_need_an_earlier_answer(self) -> None:
        """Replace an answer-inspecting tool for a new question on a file."""

        literature = project_type("literature")
        reply = '{"intent": "source_context", "standalone": true}'
        cases = (
            ("In Figure 1G of smith20.pdf, what does 79.7 represent?",
             literature.default_intent),
            ("Show the sentence you cited from smith20.pdf.",
             "source_context"),
            ("Show the context of the first cited passage.",
             "source_context"),
        )
        for question, intent in cases:
            with self.subTest(question=question):
                with patch.object(answering, "side_reply",
                                  return_value=reply):
                    decision = answering.route_question(
                        object(), literature, self.turns, question
                    )
                self.assertEqual(decision.intent, intent)
                if intent == literature.default_intent:
                    self.assertIn("names a file", decision.fallback)

    def test_follow_up_tools_are_rerouted_without_them(self) -> None:
        """Ask the router again without tools that need an earlier answer."""

        literature = project_type("literature")
        replies = [
            '{"intent": "source_context", "standalone": true}',
            '{"intent": "topic_listing", "standalone": true}',
        ]
        question = (
            "Which passages of smith20.pdf explain how its interface works?"
        )
        with patch.object(
            answering, "side_reply", side_effect=replies
        ) as reply:
            decision = answering.route_question(
                object(), literature, self.turns, question
            )

        self.assertEqual(decision.intent, "topic_listing")
        self.assertIn("source_context replaced by topic_listing",
                      decision.fallback)
        first, second = (call.args[2] for call in reply.call_args_list)
        self.assertIn("- source_context:", first)
        self.assertNotIn("- source_context:", second)
        self.assertIn("- topic_listing:", second)

    def test_wording_cues_set_the_intent(self) -> None:
        """Let clear wording overrule the router's intent."""

        literature = project_type("literature")
        cases = (
            ("Give me an overview of the BRAPH 2 paper.", "research_briefing",
             "paper_overview"),
            ("Prepare a research briefing for smith20.pdf.",
             "research_briefing", "research_briefing"),
            ("Summarize its main contributions and limitations.",
             "question_answering", "question_answering"),
            ("How does BRAPH 2 differ from the earlier BRAPH framework?",
             "question_answering", "compare_sources"),
            ("Write a structured report comparing the methods of a.pdf "
             "and b.pdf.", "report_generation", "report_generation"),
            ("Of the BRAPH mentions you listed, which describe software "
             "features?", "mention_classification", "refine_results"),
            ("Find new passages discussing interfaces, not just the "
             "earlier matches.", "refine_results", "topic_listing"),
        )
        for question, routed, expected in cases:
            with self.subTest(question=question):
                reply = json.dumps({"intent": routed, "standalone": True})
                with patch.object(answering, "side_reply",
                                  return_value=reply) as call:
                    decision = answering.route_question(
                        object(), literature, self.turns, question
                    )
                self.assertEqual(decision.intent, expected)
                self.assertEqual(call.call_count, 1)
                if routed != expected:
                    self.assertIn("set by the question's wording",
                                  decision.fallback)

    def test_required_words_rule_out_an_intent(self) -> None:
        """Ask again without an intent whose required words are absent."""

        literature = project_type("literature")
        replies = [
            '{"intent": "reference_check", "standalone": true}',
            '{"intent": "question_answering", "standalone": true}',
        ]
        question = "Did the original paper a.pdf support plug-ins?"
        with patch.object(
            answering, "side_reply", side_effect=replies
        ) as reply:
            decision = answering.route_question(
                object(), literature, self.turns, question
            )

        self.assertEqual(decision.intent, "question_answering")
        self.assertIn("lacks its required words", decision.fallback)
        self.assertNotIn("- reference_check:",
                         reply.call_args_list[1].args[2])
        with patch.object(
            answering, "side_reply",
            return_value='{"intent": "reference_check", "standalone": true}',
        ):
            kept = answering.route_question(
                object(), literature, self.turns,
                "Check the references of a.pdf.",
            )
        self.assertEqual(kept.intent, "reference_check")

    def test_router_errors_do_not_stop_answers(self) -> None:
        """Use a fallback when generation of the route raises."""

        with patch.object(
            answering, "side_reply", side_effect=OSError("offline")
        ):
            decision = answering.route_question(
                object(), self.config, self.turns, "why?"
            )

        self.assertEqual(decision.intent, self.config.default_intent)
        self.assertIn("router failed: offline", decision.fallback)
        self.assertTrue(decision.refers_back)
        self.assertFalse(decision.standalone)
        self.assertEqual(decision.related_turn_ids, ())

    def test_router_failure_on_first_turn_is_standalone(self) -> None:
        """Avoid attributing related turns to a failed first routing call."""

        with patch.object(
            answering, "side_reply", side_effect=RuntimeError("failure")
        ):
            decision = answering.route_question(
                object(), self.config, [], "What is fit?"
            )

        self.assertTrue(decision.standalone)
        self.assertEqual(decision.related_turn_ids, ())
        self.assertEqual(decision.intent, self.config.default_intent)


class TestRepeatedQuestions(unittest.TestCase):
    """Test that the router is told when a question repeats a turn."""

    def setUp(self) -> None:
        """Create a conversation whose second of three turns used a tool."""

        self.config = project_type("code")
        self.turns = [
            _turn("Explain the fit function", "Fit trains [1]. More."),
            _turn("List every MRI scanner", "Two scanners [1]. More."),
            _turn("Explain the predict function", "It predicts [1]. More."),
        ]
        self.turns[1].parameters = {"intent": "code_generation"}

    def _route(
        self,
        question: str,
        excluded: frozenset[int] = frozenset(),
    ) -> tuple[answering.RouteDecision, str]:
        """Route a question and return the router's user prompt.

        Parameters
        ----------
        question: str
            User question being classified.
        excluded: frozenset[int], optional
            Turns excluded from routing. Defaults to none.

        Returns
        -------
        tuple[answering.RouteDecision, str]
            The decision, and the prompt the router model received.

        """

        reply = '{"intent": "debugging", "standalone": true}'
        with patch.object(
            answering, "side_reply", return_value=reply
        ) as side:
            decision = answering.route_question(
                object(), self.config, self.turns, question,
                excluded_turn_ids=excluded,
            )
        return decision, side.call_args.args[2]

    def test_router_is_told_about_a_repeat_and_decides(self) -> None:
        """Name the turn and its intent; keep the router's choice."""

        for question in (
            "List every MRI scanner",
            "MRI scanner: list every?",
        ):
            with self.subTest(question=question):
                decision, prompt = self._route(question)
                self.assertIn(
                    "repeats turn 2, which was answered as code_generation.",
                    prompt,
                )
                self.assertEqual(decision.repeated_turn, 2)
                self.assertEqual(decision.intent, "debugging")

    def test_repeat_of_the_last_turn_only_names_it(self) -> None:
        """Without the user, the note never suggests another intent."""

        del self.turns[2]
        decision, prompt = self._route("List every MRI scanner")
        self.assertIn(
            "repeats turn 2, which was answered as code_generation.", prompt
        )
        self.assertNotIn("different", prompt.split("New question:")[1])
        self.assertEqual(decision.repeated_turn, 2)

    def test_user_chooses_how_to_answer_a_back_to_back_repeat(self) -> None:
        """Same way reuses the saved route; another way reroutes."""

        del self.turns[2]
        self.turns[1].parameters = {
            "intent": "project_inventory",
            "standalone": True,
            "related_turn_ids": [],
        }
        reply = '{"intent": "debugging", "standalone": true}'
        for choice, intent, asked in (
            (0, "project_inventory", False),
            (1, "debugging", True),
        ):
            with self.subTest(choice=choice):
                choose = Mock(return_value=choice)
                with patch.object(
                    answering, "side_reply", return_value=reply
                ) as side:
                    decision = answering.route_question(
                        object(), self.config, self.turns,
                        "List every MRI scanner", choose=choose,
                    )
                question, options = choose.call_args.args
                self.assertIn("right after the last answer", question)
                self.assertEqual(
                    options,
                    ("Same way again (Project inventory)", "Try another way"),
                )
                self.assertEqual(decision.intent, intent)
                self.assertEqual(decision.repeated_turn, 2)
                self.assertEqual(decision.another_way, asked)
                self.assertEqual(side.called, asked)
                if asked:
                    intents = side.call_args.args[2].split("Earlier")[0]
                    self.assertNotIn("- project_inventory:", intents)
                    # Reuse intents would answer from the same passages.
                    self.assertNotIn("- conversation_followup:", intents)
                    self.assertNotIn("- rephrase_previous:", intents)
                    self.assertIn("- debugging:", intents)

    def test_cancelled_or_unusable_choice_leaves_it_to_the_router(
        self,
    ) -> None:
        """Route with the repeat note when the user cancels."""

        del self.turns[2]
        self.turns[1].parameters = {
            "intent": "project_inventory",
            "standalone": True,
            "related_turn_ids": [],
        }
        choose = Mock(return_value=None)
        reply = '{"intent": "debugging", "standalone": true}'
        with patch.object(answering, "side_reply", return_value=reply):
            decision = answering.route_question(
                object(), self.config, self.turns,
                "List every MRI scanner", choose=choose,
            )
        self.assertEqual(decision.intent, "debugging")

        # Without a saved route there is nothing to repeat, so no question.
        self.turns[1].parameters = {}
        choose.reset_mock()
        with patch.object(answering, "side_reply", return_value=reply):
            answering.route_question(
                object(), self.config, self.turns,
                "List every MRI scanner", choose=choose,
            )
        choose.assert_not_called()

    def test_earlier_repeats_are_not_asked_about(self) -> None:
        """Only a question asked twice in a row asks the user."""

        choose = Mock(return_value=0)
        reply = '{"intent": "debugging", "standalone": true}'
        with patch.object(answering, "side_reply", return_value=reply):
            answering.route_question(
                object(), self.config, self.turns,
                "List every MRI scanner", choose=choose,
            )
        choose.assert_not_called()

    def test_back_to_back_repeat_counts_even_when_it_points_back(
        self,
    ) -> None:
        """Right after the same question, "they" means the same thing."""

        self.turns[2].question = "Which dataset did they use?"
        decision, prompt = self._route("which dataset did they use")
        self.assertEqual(decision.repeated_turn, 3)
        self.assertIn("repeats turn 3", prompt)

    def test_singular_and_plural_are_the_same_question(self) -> None:
        """"scanners" and "scanner" are the same key word."""

        decision, _ = self._route("List every MRI scanners")
        self.assertEqual(decision.repeated_turn, 2)
        self.assertEqual(answering._singular("studies"), "study")
        for word in ("analysis", "process", "corpus", "gas"):
            with self.subTest(word=word):
                self.assertEqual(answering._singular(word), word)

    def test_new_questions_are_not_repeats(self) -> None:
        """A changed name, an added word, or pointing back later is new."""

        self.turns[1].question = "List every MRI scanner it uses"
        for question in (
            "List every CT scanner",
            "List every MRI scanner model",
            "List every MRI scanner it uses",
        ):
            with self.subTest(question=question):
                decision, prompt = self._route(question)
                self.assertIsNone(decision.repeated_turn)
                self.assertNotIn("repeats turn", prompt)

    def test_excluded_turns_are_not_matched(self) -> None:
        """A disliked answer is never named as the repeated turn."""

        decision, prompt = self._route(
            "List every MRI scanner", frozenset({2})
        )
        self.assertIsNone(decision.repeated_turn)
        self.assertNotIn("repeats turn", prompt)


class TestRouterPrompt(unittest.TestCase):
    """Test the router prompt's scope and conversation representation."""

    def test_lists_configured_intents_and_question(self) -> None:
        """Show supported intent descriptions and the current question."""

        config = project_type("literature")
        with patch.object(
            answering, "side_reply", return_value="not JSON"
        ) as router:
            answering.route_question(
                object(), config, [], "What does MRI measure?"
            )

        system, prompt = router.call_args.args[1:3]
        self.assertIn("literature collection", system)
        self.assertIn("small_talk:", prompt)
        self.assertIn("question_answering:", prompt)
        self.assertIn("New question: What does MRI measure?", prompt)
        self.assertIn('"related_turns"', prompt)
        self.assertIn("Earlier turns:\nnone", prompt)
        self.assertIn(
            'Example: {"intent": "question_answering", '
            '"standalone": true}',
            prompt,
        )
        self.assertNotIn('"related_turns": [2]}', prompt)
        self.assertEqual(router.call_args.kwargs, {"max_new_tokens": 64})

    def test_limits_listed_turns_to_latest_thirty(self) -> None:
        """Keep the router prompt bounded even with a long conversation."""

        config = project_type("literature")
        turns = [_turn(f"Topic_{n}", "Response") for n in range(1, 36)]
        with patch.object(
            answering, "side_reply", return_value="not JSON"
        ) as router:
            answering.route_question(
                object(), config, turns, "How does it work?"
            )

        prompt = router.call_args.args[2]
        self.assertNotIn("Turn 5:", prompt)
        self.assertIn("Turn 6:", prompt)
        self.assertIn("Turn 35:", prompt)
        self.assertEqual(
            prompt.count(" | Sources: "), answering.MAX_LISTED_TURNS
        )

    def test_excludes_small_talk_from_router_context(self) -> None:
        """Do not show conversation-ignoring turns to the router."""

        config = project_type("literature")
        turns = [
            _turn("Hello", "Hi", intent="small_talk"),
            _turn("What is NOVA?", "A model."),
        ]
        with patch.object(
            answering, "side_reply", return_value="not JSON"
        ) as router:
            answering.route_question(
                object(), config, turns, "What is RFE?"
            )

        prompt = router.call_args.args[2]
        self.assertNotIn("Turn 1:", prompt)
        self.assertIn("Turn 2: Q: What is NOVA?", prompt)

    def test_uses_saved_turn_representation_when_available(self) -> None:
        """Prefer a saved summary over rebuilding it from current text."""

        config = project_type("code")
        turn = _turn("Newer question", "Newer answer")
        turn.parameters["representation"] = {
            "turn_id": 99,
            "question": "Saved question",
            "sources": ["saved.py:method"],
            "summary": "Saved summary.",
        }
        with patch.object(
            answering, "side_reply", return_value="not JSON"
        ) as router:
            answering.route_question(
                object(), config, [turn], "What happens next?"
            )

        prompt = router.call_args.args[2]
        self.assertIn("Turn 1: Q: Saved question", prompt)
        self.assertIn("Sources: saved.py:method", prompt)
        self.assertIn("A: Saved summary.", prompt)
        self.assertNotIn("Newer question", prompt)

    def test_rebuilds_malformed_saved_turn_representations(self) -> None:
        """Rebuild a turn when saved summary fields are incorrectly typed."""

        valid = {
            "question": "Saved question",
            "sources": ["saved.py:fit"],
            "summary": "Saved summary.",
        }
        malformed = (
            {**valid, "question": ["wrong type"]},
            {**valid, "question": "  "},
            {**valid, "sources": 42},
            {**valid, "sources": ["saved.py:fit", 17]},
            {**valid, "summary": {"text": "wrong type"}},
            {**valid, "question": "Q" * 201},
            {**valid, "summary": "S" * 201},
            {**valid, "sources": ["f.py"] * 6},
        )
        config = project_type("code")
        for stored in malformed:
            with self.subTest(stored=stored):
                turn = _turn("Original question", "Original answer.")
                turn.parameters["representation"] = stored
                with patch.object(
                    answering, "side_reply", return_value="not JSON"
                ) as router:
                    answering.route_question(
                        object(), config, [turn], "What happens next?"
                    )
                prompt = router.call_args.args[2]
                self.assertIn("Turn 1: Q: Original question", prompt)
                self.assertIn("A: Original answer.", prompt)
                self.assertNotIn("Saved summary.", prompt)
                self.assertIs(turn.parameters["representation"], stored)

    def test_missing_saved_fields_rebuild_representation(self) -> None:
        """Missing sources and summary must not crash prompt formatting."""

        config = project_type("code")
        turn = _turn("Original question", "Original answer")
        turn.parameters["representation"] = {"question": "Saved"}
        with patch.object(
            answering, "side_reply", return_value="not JSON"
        ) as router:
            answering.route_question(
                object(), config, [turn], "What happens next?"
            )
        self.assertIn(
            "Turn 1: Q: Original question", router.call_args.args[2]
        )

    def test_invalid_turn_parameters_do_not_break_routing(self) -> None:
        """Treat malformed persisted turn metadata as nonrestrictive."""

        config = project_type("code")
        turn = _turn("Original question", "Original answer")
        for parameters in (["invalid metadata"], {"intent": []}):
            with self.subTest(parameters=parameters):
                turn.parameters = parameters
                with patch.object(
                    answering, "side_reply", return_value="not JSON"
                ) as router:
                    decision = answering.route_question(
                        object(), config, [turn], "What happens next?"
                    )
                self.assertEqual(decision.intent, config.default_intent)
                self.assertIn(
                    "Turn 1: Q: Original question",
                    router.call_args.args[2],
                )

    def test_includes_example_for_reuse_intent(self) -> None:
        """Use a structural example without mislabeling user wording."""

        config = project_type("code")
        turns = [_turn("How is it loaded?", "With a loader.")]
        with patch.object(
            answering, "side_reply", return_value="not JSON"
        ) as router:
            answering.route_question(
                object(), config, turns, "What happens next?"
            )

        prompt = router.call_args.args[2]
        self.assertIn("- conversation_followup:", prompt)
        self.assertIn(
            'Example for intent "conversation_followup" referring to turn 1: '
            '{"intent": "conversation_followup", "standalone": false, '
            '"related_turns": [1]}',
            prompt,
        )
        self.assertNotIn('Example: "just in two sentences"', prompt)
        self.assertIn("- rephrase_previous:", prompt)


class TestRelatedTurnChecks(unittest.TestCase):
    """Test related-turn acceptance and search-query construction."""

    def setUp(self) -> None:
        """Prepare a small-talk turn and an evidence-bearing turn."""

        self.config = project_type("literature")
        self.turns = [
            _turn("how are you", "Fine, thanks.", (), intent="small_talk"),
            _turn("what is NOVA", "NOVA analyses infrared spectra [1]."),
        ]

    def _route(
        self, question: str, related: str, intent: str = "question_answering"
    ) -> tuple[answering.RouteDecision, str]:
        """Return a checked decision and captured router prompt.

        Parameters
        ----------
        question: str
            Question to route.
        related: str
            JSON list of related turn numbers.
        intent: str, optional
            Intent in the mocked reply. Defaults to question answering.

        Returns
        -------
        tuple[answering.RouteDecision, str]
            Checked routing result and user prompt.

        """

        reply = (
            f'{{"intent": "{intent}", "standalone": false, '
            f'"related_turns": {related}}}'
        )
        with patch.object(
            answering, "side_reply", return_value=reply
        ) as router:
            decision = answering.route_question(
                object(), self.config, self.turns, question
            )
        return decision, router.call_args.args[2]

    def test_question_naming_its_own_topic_drops_unrelated_turns(
        self,
    ) -> None:
        """Discard unrelated turns despite the model's related labels."""

        decision, prompt = self._route("what is RFE", "[1, 2]")
        self.assertEqual(decision.related_turn_ids, ())
        self.assertTrue(decision.standalone)
        self.assertEqual(
            [number for number, _ in decision.dropped_turns], [1, 2]
        )
        self.assertIn("ignores the conversation", decision.dropped_turns[0][1])
        self.assertNotIn("Turn 1:", prompt)
        self.assertIn("what is NOVA", prompt)
        self.assertEqual(
            answering.search_text("what is RFE", self.turns, decision),
            "what is RFE",
        )

    def test_shared_term_keeps_a_turn_without_extending_search(
        self,
    ) -> None:
        """Keep a topical reference but do not rewrite standalone search."""

        question = "what are NOVA's limitations"
        decision, _ = self._route(question, "[2]")
        self.assertEqual(decision.related_turn_ids, (2,))
        self.assertFalse(decision.refers_back)
        self.assertEqual(
            answering.search_text(question, self.turns, decision), question
        )

    def test_pointing_back_adds_related_questions_to_search(self) -> None:
        """Expand deictic queries using earlier user wording only."""

        question = "how do I install it?"
        decision, _ = self._route(question, "[2]")
        self.assertEqual(decision.related_turn_ids, (2,))
        self.assertTrue(decision.refers_back)
        self.assertEqual(
            answering.search_text(question, self.turns, decision),
            "how do I install it? what is NOVA",
        )

    def test_ignored_conversation_turn_is_never_related(self) -> None:
        """Reject small talk even if a question explicitly refers back."""

        decision, _ = self._route("What did you say earlier?", "[1]")
        self.assertEqual(decision.related_turn_ids, ())
        self.assertEqual(
            decision.dropped_turns,
            ((1, "turn ignores the conversation"),),
        )

    def test_reuse_intent_falls_back_to_latest_non_small_talk(self) -> None:
        """Reuse the last meaningful turn when no related ID is retained."""

        decision, _ = self._route(
            "Just in two sentences", "[]", "rephrase_previous"
        )
        self.assertEqual(decision.related_turn_ids, (2,))
        self.assertFalse(decision.standalone)

    def test_followup_tool_falls_back_to_latest_non_small_talk(self) -> None:
        """Resolve follow-up tools against an earlier meaningful answer."""

        decision, _ = self._route(
            "Show that source", "[]", "source_context"
        )
        self.assertEqual(decision.related_turn_ids, (2,))
        self.assertFalse(decision.standalone)

    def test_reuse_without_meaningful_turn_has_no_related_id(self) -> None:
        """Never reuse a small-talk-only conversation as evidence."""

        self.turns = self.turns[:1]
        decision, _ = self._route(
            "Explain the earlier answer", "[]", "conversation_followup"
        )
        self.assertEqual(decision.related_turn_ids, ())

    def test_related_turn_with_unknown_intent_remains_eligible(self) -> None:
        """Do not exclude older turns with unknown stored intent IDs."""

        self.turns[1].parameters = {"intent": "old_intent"}
        decision, _ = self._route("explain NOVA", "[2]")
        self.assertEqual(decision.related_turn_ids, (2,))

    def test_refers_back_cues(self) -> None:
        """Recognize pronouns, references, ellipses, and short questions."""

        for question in (
            "how do I install it?",
            "explain the second point",
            "what do you think so far?",
            "and for MATLAB?",
            "Compare the methods you mentioned",
            "can you compare the two methods we discussed?",
            "why?",
        ):
            with self.subTest(question=question):
                self.assertTrue(answering.refers_back(question))

    def test_questions_without_context_cues_do_not_refer_back(self) -> None:
        """Do not classify independent, named questions as follow-ups."""

        for question in (
            "what is RFE",
            "why NOV A not NOVA?",
            "can you explain recursive feature elimination",
        ):
            with self.subTest(question=question):
                self.assertFalse(answering.refers_back(question))


class TestSearchText(unittest.TestCase):
    """Test user-authored retrieval queries for contextual follow-ups."""

    def setUp(self) -> None:
        """Prepare non-model previous turns for search construction."""

        self.turns = [
            _turn("  Topic alpha  ", "An answer about alpha."),
            _turn("Topic   beta", "An answer about beta."),
            _turn("Topic gamma", "An answer about gamma."),
        ]

    def test_unrelated_or_standalone_query_is_unchanged(self) -> None:
        """Return the original query when no expansion is requested."""

        query = "  How does beta work?  "
        decisions = (
            answering.RouteDecision("x", True, related_turn_ids=(2,)),
            answering.RouteDecision("x", False, related_turn_ids=()),
        )
        for decision in decisions:
            with self.subTest(decision=decision):
                self.assertEqual(
                    answering.search_text(query, self.turns, decision),
                    query,
                )

    def test_expands_with_multiple_questions_not_answers(self) -> None:
        """Retain only selected user questions in their supplied order."""

        decision = answering.RouteDecision(
            "x", False, related_turn_ids=(1, 3), refers_back=True
        )
        self.assertEqual(
            answering.search_text("  And why?  ", self.turns, decision),
            "And why? Topic alpha Topic gamma",
        )

    def test_drops_out_of_range_numbers_and_blank_questions(self) -> None:
        """Avoid selecting missing turns or adding empty query segments."""

        self.turns[0].question = "  "
        decision = answering.RouteDecision(
            "x", False, related_turn_ids=(-1, 1, 4, 2), refers_back=True
        )
        self.assertEqual(
            answering.search_text("Why?", self.turns, decision),
            "Why? Topic beta",
        )


class TestHistory(unittest.TestCase):
    """Test history policies, citation handling, and compact summaries."""

    def test_selection_policies(self) -> None:
        """Combine recent and related turns according to intent policy."""

        related = answering.RouteDecision("x", False, (2, 9))
        standalone = answering.RouteDecision("x", True)
        self.assertEqual(
            answering.select_history(
                10, HISTORY_POLICY_RELEVANT, 2, related
            ),
            [2, 9, 10],
        )
        self.assertEqual(
            answering.select_history(
                10, HISTORY_POLICY_RELEVANT, 2, standalone
            ),
            [10],
        )
        self.assertEqual(
            answering.select_history(
                10, HISTORY_POLICY_RECENT, 2, standalone
            ),
            [9, 10],
        )
        self.assertEqual(
            answering.select_history(
                10, HISTORY_POLICY_NONE, 2, related
            ),
            [],
        )
        self.assertEqual(
            answering.select_history(
                0, HISTORY_POLICY_RELEVANT, 2, related
            ),
            [],
        )

    def test_recent_history_respects_requested_window(self) -> None:
        """Keep a chronological window, capped by conversation length."""

        decision = answering.RouteDecision("x", False)
        self.assertEqual(
            answering.select_history(3, HISTORY_POLICY_RECENT, 2, decision),
            [2, 3],
        )
        self.assertEqual(
            answering.select_history(3, HISTORY_POLICY_RECENT, 6, decision),
            [1, 2, 3],
        )

    def test_relevant_history_is_sorted_and_deduplicated(self) -> None:
        """Merge older related turns with the recent history in order."""

        decision = answering.RouteDecision(
            "x", False, related_turn_ids=(7, 2, 9, 2)
        )
        self.assertEqual(
            answering.select_history(10, HISTORY_POLICY_RELEVANT, 2, decision),
            [2, 7, 9, 10],
        )

    def test_history_has_no_evidence_and_neutral_citations(self) -> None:
        """Include questions and cleaned answers without source evidence."""

        turns = [
            _turn("What is fit?", "<think>hm</think>It trains [1]."),
        ]
        self.assertEqual(
            answering.history_messages(turns, [1]),
            [
                {"role": USER_ROLE, "content": "What is fit?"},
                {"role": ASSISTANT_ROLE, "content": "It trains [source]."},
            ],
        )

    def test_history_preserves_selected_turn_order(self) -> None:
        """Emit only the requested turns as user/assistant pairs."""

        turns = [_turn("Q1", "A1"), _turn("Q2", "A2"), _turn("Q3", "A3")]
        self.assertEqual(
            answering.history_messages(turns, [1, 3]),
            [
                {"role": USER_ROLE, "content": "Q1"},
                {"role": ASSISTANT_ROLE, "content": "A1"},
                {"role": USER_ROLE, "content": "Q3"},
                {"role": ASSISTANT_ROLE, "content": "A3"},
            ],
        )
        self.assertEqual(answering.history_messages(turns, []), [])

    def test_history_neutralizes_multiple_citation_formats(self) -> None:
        """Replace numbered groups without changing unrelated brackets."""

        turns = [
            _turn("Q", "Result [1, 2] and [3] with [source] and [abc].")
        ]
        messages = answering.history_messages(turns, [1])
        self.assertEqual(
            messages[-1]["content"],
            "Result [source] and [source] with [source] and [abc].",
        )

    def test_representation_lists_cited_sources_and_first_sentence(
        self,
    ) -> None:
        """List cited evidence and summarize the first answer sentence."""

        turn = _turn("Explain fit", "Fit trains the model [2]. More.", (1, 2))
        representation = answering.turn_representation(turn, 7)
        self.assertEqual(representation["turn_id"], 7)
        self.assertEqual(representation["question"], "Explain fit")
        self.assertEqual(representation["sources"], ["file2.py:fit"])
        self.assertEqual(representation["summary"], "Fit trains the model.")

    def test_representation_uses_a_given_summary(self) -> None:
        """Replace the first sentence with a model-written summary."""

        turn = _turn("Explain fit", "Fit trains [1]. More.")
        representation = answering.turn_representation(
            turn, 1, summary="S" * 250
        )
        self.assertEqual(representation["summary"], "S" * 200)

    def test_representation_handles_foreign_platform_paths(self) -> None:
        """Read POSIX and Windows evidence labels on every host platform."""

        for raw in (
            "src/module.py",
            r"src\module.py",
            r"C:\repo\src\module.py",
            r"\\server\share\src\module.py",
        ):
            with self.subTest(path=raw):
                turn = _turn("Q", "Answer [1].", (1,))
                turn.evidence = (
                    Evidence(1, Path(raw), 1, 2, "fit", "text"),
                )
                representation = answering.turn_representation(turn, 1)
                self.assertEqual(
                    representation["sources"], ["module.py:fit"]
                )

    def test_representation_without_citations_lists_all_sources(self) -> None:
        """List all distinct evidence labels if no citation was used."""

        turn = _turn("Q", "An uncited answer.", (1, 2, 2, 3))
        representation = answering.turn_representation(turn, 1)
        self.assertEqual(
            representation["sources"],
            ["file1.py:fit", "file2.py:fit", "file3.py:fit"],
        )

    def test_representation_limits_question_summary_and_sources(self) -> None:
        """Bound compact representations without altering the stored turn."""

        question = "Q" * 250
        answer = "A" * 250 + ". An additional sentence."
        turn = _turn(question, answer, tuple(range(1, 10)))
        representation = answering.turn_representation(turn, 2)
        self.assertEqual(len(representation["question"]), 200)
        self.assertEqual(len(representation["summary"]), 200)
        self.assertEqual(len(representation["sources"]), 5)
        self.assertEqual(turn.question, question)
        self.assertEqual(turn.answer, answer)

    def test_representation_ignores_citations_inside_thinking(self) -> None:
        """Only cited evidence in the visible answer selects sources."""

        turn = _turn(
            "Explain fit", "<think>Inspect [1]</think>Result [2].", (1, 2)
        )
        representation = answering.turn_representation(turn, 4)
        self.assertEqual(representation["sources"], ["file2.py:fit"])
        self.assertEqual(representation["summary"], "Result.")

    def test_representation_strips_thinking_and_collapses_spaces(self) -> None:
        """Exclude thinking text and normalize whitespace in summaries."""

        turn = _turn(
            "  Explain   NOVA\n briefly  ",
            "<think>reason</think> First   sentence. Second sentence.",
        )
        representation = answering.turn_representation(turn, 3)
        self.assertEqual(representation["question"], "Explain NOVA briefly")
        self.assertEqual(representation["summary"], "First sentence.")


if __name__ == "__main__":
    unittest.main()


class TestAnswerSummary(unittest.TestCase):
    """Test the router model's one-sentence answer summary."""

    def test_long_answer_is_summarized_without_citations(self) -> None:
        """Send the visible answer only and collapse the reply."""

        chat = _FakeChat(response="  Fit trains\n the model.  ")
        summary = answering.summarize_answer(
            chat, "Explain fit",
            "<think>Inspect [1]</think>Status line [1]. Fit trains it [2].",
        )
        self.assertEqual(summary, "Fit trains the model.")
        prompt = chat.observations[0]["message"]
        self.assertIn("Status line. Fit trains it.", prompt)
        self.assertNotIn("Inspect", prompt)
        self.assertEqual(
            chat.observations[0]["max_new_tokens"],
            answering._SUMMARY_MAX_TOKENS,
        )
        self.assertEqual(chat.messages[0]["content"], "Original instructions")

    def test_one_sentence_answer_needs_no_model(self) -> None:
        """Keep the first sentence when it is the whole answer."""

        chat = _FakeChat()
        self.assertIsNone(
            answering.summarize_answer(chat, "Q", "Only one sentence [1].")
        )
        self.assertEqual(chat.observations, [])

    def test_failed_or_empty_summaries_are_none(self) -> None:
        """A failed call never stops saving the turn."""

        for chat in (
            _FakeChat(reply_error=RuntimeError("out of memory")),
            _FakeChat(response="<think>unfinished"),
            _FakeChat(response="   "),
        ):
            with self.subTest(response=chat.response):
                self.assertIsNone(
                    answering.summarize_answer(chat, "Q", "One. Two.")
                )
