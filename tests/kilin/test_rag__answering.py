"""Test RAG answer orchestration, intent selection, and tool routing.

Answer processing uses controlled offline model substitutes.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, Mock, patch, sentinel
from kilin import rag
from kilin.answering import RouteDecision
from kilin.plugins import list_tools
from kilin.project_type import (
    HISTORY_POLICY_NONE,
    RETRIEVAL_MODE_DOCUMENT,
    RETRIEVAL_MODE_NONE,
    RETRIEVAL_MODE_REUSE,
    RETRIEVAL_MODE_SEARCH,
    project_type,
)
from kilin.retrievals import DocumentSelection
from kilin.tools import ToolResult, ToolUse
from kilin.tools._registry import TOOL_REGISTRY
from kilin.trace_logging import NullTraceLogger, TraceConfig, TraceLogger
from tests.kilin.retrievals._support import chunk, ranked

from tests.kilin._rag_support import (
    _answer_patches,
    _evidence,
    _intent,
    _session,
    _turn,
    _type_config,
)


class TestRAGAnswerOrchestration(unittest.TestCase):
    """Exercise the configured answer modes and their saved turn records."""

    def test_search_answer_tracks_phases_and_persists_turn(self) -> None:
        """Route, retrieve, answer, and save a search-mode turn."""
        session = _session()
        session.chunks = [chunk("a", "text")]
        phases = []
        tokens = Mock()
        with _answer_patches(session, _intent()) as mocks:
            answer = session.ask(
                "  Explain the function  ",
                retrieval_count=2,
                retrieval_query="  focused search  ",
                answer_guidance="Be brief.",
                token_callback=tokens,
                phase_callback=lambda phase, details: phases.append(
                    (phase, details)
                ),
                use_router=False,
            )
        self.assertEqual(answer, "model answer")
        self.assertEqual(len(session.turns), 1)
        turn = session.turns[0]
        self.assertEqual(turn.question, "Explain the function")
        self.assertEqual(turn.evidence, (_evidence(),))
        self.assertEqual(turn.parameters["retrieval_count"], 2)
        self.assertEqual(turn.parameters["retrieval_query"],
                         "focused search")
        self.assertEqual(turn.parameters["answer_guidance"], "Be brief.")
        self.assertEqual(turn.parameters["retrieval_mode"], "search")
        self.assertEqual(turn.parameters["representation"]["turn_id"], 1)
        self.assertEqual(
            [phase for phase, _ in phases],
            [
                "routed", "retrieving", "evidence_ready", "generating",
                "saving", "completed",
            ],
        )
        self.assertEqual(phases[-1][1]["evidence_count"], 1)
        self.assertEqual(mocks.retrieve.call_args.args[:2],
                         ("focused search", 2))
        self.assertIn("[1] src/a.py:1-3", session.chat_session.reply.call_args
                      .args[0])
        self.assertIn("Be brief.", session.chat_session.reply.call_args
                      .args[0])
        self.assertIn("token_callback",
                      session.chat_session.reply.call_args.kwargs)
        mocks.save.assert_called_once()
        tokens.assert_not_called()

    def test_streamed_and_saved_citations_are_identical(self) -> None:
        """Link worded prose citations without rewriting streamed code."""

        session = _session()
        raw = (
            "See [sources 1 and 2] and "
            "`[source 1]`.\n"
            "```python\nvalue = '[source 1]'\n```\n"
            "Then [source 1]."
        )
        emitted: list[str] = []

        def reply(_prompt: str, **options: object) -> str:
            """Simulate a model yielding one character at a time."""

            callback = options["token_callback"]
            for char in raw:
                callback(char)
            return raw

        session.chat_session.reply.side_effect = reply
        session.chat_session.last_finish_reason = "stop"
        with _answer_patches(session, _intent()):
            answer = session.ask(
                "Explain the cited code", use_router=False,
                token_callback=emitted.append,
            )
        self.assertEqual(answer, rag._numbered_citations(raw))
        self.assertEqual("".join(emitted), answer)
        self.assertEqual(session.turns[-1].answer, answer)

    def test_model_summary_comes_from_the_core(self) -> None:
        """Keep the model label with the saved turn, not in the GUI."""
        session = _session()
        session.chat_session.temperature = 0.9
        session.chat_session.max_new_tokens = 8192
        with _answer_patches(session, _intent()):
            session.ask("Question", use_router=False)
        self.assertEqual(
            session.turns[-1].parameters["model_summary"],
            "Model: chat-model (thinking off, T=0.9, 8 192 max tokens)",
        )

    def test_routed_turn_saves_the_router_summary(self) -> None:
        """Save the router model's summary only when routing is on."""
        session = _session()
        with _answer_patches(session, _intent()), patch.object(
            rag, "summarize_answer", return_value="Model summary."
        ) as summarize:
            session.ask("Question", use_router=True)
            session.ask("Question two", use_router=False)
        summarize.assert_called_once_with(
            session.chat_session, "Question", "model answer"
        )
        routed, unrouted = (
            turn.parameters["representation"] for turn in session.turns
        )
        self.assertEqual(routed["summary"], "Model summary.")
        self.assertEqual(routed["summary_model"], "chat-model")
        self.assertEqual(unrouted["summary"], "model answer")
        self.assertNotIn("summary_model", unrouted)

    def test_choice_dialog_reaches_routing_and_the_page(self) -> None:
        """Show the routing question on the page before waiting for it."""
        session = _session()
        choose = Mock(return_value=0)
        phases = []
        with _answer_patches(session, _intent()) as mocks:
            session.ask(
                "Question", use_router=True, choice_callback=choose,
                phase_callback=lambda phase, details: phases.append(
                    (phase, details)
                ),
            )
            ask_user = mocks.route.call_args.kwargs["choose"]
            self.assertEqual(ask_user("How?", ("Same", "Other")), 0)
        choose.assert_called_once_with("How?", ("Same", "Other"))
        self.assertEqual(phases[-1][0], "processing")
        self.assertEqual(
            phases[-1][1]["choice"],
            {
                "question": "How?", "options": ["Same", "Other"],
                "allow_cancel": False,
            },
        )

    def test_another_way_tells_the_answer_the_last_one_missed(self) -> None:
        """Add the hint only after the user chose "Try another way"."""
        for another_way in (True, False):
            with self.subTest(another_way=another_way):
                session = _session()
                decision = RouteDecision(
                    "test_intent", True, repeated_turn=3,
                    another_way=another_way,
                )
                with _answer_patches(session, _intent(), decision):
                    session.ask(
                        "Question", use_router=False,
                        answer_guidance="Be brief.",
                    )
                prompt = session.chat_session.reply.call_args.args[0]
                guidance = session.turns[-1].parameters["answer_guidance"]
                self.assertIn("Be brief.", prompt)
                self.assertEqual(
                    "right after turn 3 and chose another way" in prompt,
                    another_way,
                )
                self.assertEqual("chose another way" in guidance, another_way)

    def test_small_talk_keeps_the_first_sentence(self) -> None:
        """Skip the summary for turns the router never lists."""
        session = _session()
        small_talk = _intent(RETRIEVAL_MODE_NONE, policy=HISTORY_POLICY_NONE)
        with _answer_patches(session, small_talk), patch.object(
            rag, "summarize_answer"
        ) as summarize:
            session.ask("Hello there, how are you", use_router=True)
        summarize.assert_not_called()

    def test_search_reads_further_when_passages_do_not_answer(self) -> None:
        """Ask for one more page of results after the router says no."""
        session = _session()
        first = (_evidence(),)
        wider = (_evidence(), _evidence(start=20))
        with _answer_patches(session, _intent()) as mocks, patch.object(
            rag, "summarize_answer", return_value=None
        ), patch.object(
            session, "_ask_check", return_value="No."
        ) as ask:
            mocks.retrieve.side_effect = [first, wider]
            session.ask("Which software?", retrieval_count=2,
                        use_router=True)
        self.assertEqual(
            [call.args[1] for call in mocks.retrieve.call_args_list], [2, 4]
        )
        self.assertEqual(ask.call_count, rag._EXTRA_SEARCHES)
        turn = session.turns[-1]
        self.assertEqual(len(turn.evidence), 2)
        self.assertEqual(turn.parameters["extra_searches"], 1)

    def test_source_first_does_not_retry_a_fixed_evidence_budget(
        self,
    ) -> None:
        """No redundant check or embedding when limits fix the result."""

        session = _session()
        intent = replace(_intent(), retrieval_strategy="source_first")
        with _answer_patches(session, intent) as mocks, patch.object(
            session, "_ask_check", return_value="no"
        ) as ask, patch.object(rag, "summarize_answer", return_value=None):
            session.ask("Which software?", use_router=True)
        mocks.retrieve.assert_called_once()
        ask.assert_not_called()
        self.assertNotIn(
            "extra_searches", session.turns[-1].parameters
        )

    def test_oversize_check_and_zero_searches_skip_model_calls(self) -> None:
        """Avoid model calls that cannot usefully expand evidence."""

        for zero, oversized in ((True, False), (False, True)):
            with self.subTest(zero=zero, oversized=oversized):
                session = _session()
                evidence = _evidence(
                    text="X" * 20000 if oversized else "X"
                )
                with _answer_patches(session, _intent()) as mocks, \
                        patch.object(session, "_ask_check") as ask, \
                        patch.object(rag, "summarize_answer",
                                     return_value=None), \
                        patch.object(rag, "_EXTRA_SEARCHES",
                                     0 if zero else 1):
                    mocks.retrieve.return_value = (evidence,)
                    session.ask("Which software?", use_router=True)
                mocks.retrieve.assert_called_once()
                ask.assert_not_called()

    def test_wider_search_respects_the_global_result_cap(self) -> None:
        """Expansion never requests more than the configured count cap."""

        session = _session()
        with _answer_patches(session, _intent()) as mocks, patch.object(
            session, "_ask_check", return_value="no"
        ), patch.object(rag, "summarize_answer", return_value=None), \
                patch.object(rag, "_MAX_EXPANDED_RESULTS", 3):
            mocks.retrieve.side_effect = [
                (_evidence(),),
                (_evidence(), _evidence(start=10)),
            ]
            session.ask("Which software?", retrieval_count=2,
                        use_router=True)
        self.assertEqual(
            [call.args[1] for call in mocks.retrieve.call_args_list],
            [2, 3],
        )

    def test_check_asks_the_answering_model_without_thinking(self) -> None:
        """Judge the passages with the user's model, thinking off."""
        session = _session()
        session.chat_session.thinking = True
        with patch.object(
            rag, "side_reply", return_value="yes"
        ) as side:
            reply = session._ask_check("system", "prompt", 8)
        self.assertEqual(reply, "yes")
        side.assert_called_once_with(
            session.chat_session, "system", "prompt", max_new_tokens=8
        )

    def test_search_keeps_passages_that_answer_or_without_router(
        self,
    ) -> None:
        """Search once when the router says yes, or routing is off."""
        for use_router, reply in ((True, "Yes"), (False, "No")):
            with self.subTest(use_router=use_router):
                session = _session()
                with _answer_patches(session, _intent()) as mocks, \
                        patch.object(rag, "summarize_answer",
                                     return_value=None), \
                        patch.object(session, "_ask_check",
                                     return_value=reply):
                    session.ask("Which software?", use_router=use_router)
                mocks.retrieve.assert_called_once()
                self.assertNotIn(
                    "extra_searches", session.turns[-1].parameters
                )

    def test_search_again_stops_when_nothing_more_is_found(self) -> None:
        """Keep the first passages when a wider search adds none."""
        session = _session()
        with _answer_patches(session, _intent()) as mocks, patch.object(
            rag, "summarize_answer", return_value=None
        ), patch.object(session, "_ask_check", return_value="no"):
            session.ask("Which software?", use_router=True)
        self.assertEqual(mocks.retrieve.call_count, 2)
        self.assertEqual(session.turns[-1].evidence, (_evidence(),))

    def test_search_adds_related_evidence_and_history(self) -> None:
        """Combine related evidence with the selected conversation history."""
        session = _session()
        session.turns = [
            _turn("Prior", "Prior answer [1]", (_evidence(start=20),))
        ]
        decision = RouteDecision(
            "test_intent", False, related_turn_ids=(1,), refers_back=True
        )
        with _answer_patches(
            session, _intent(related=True), decision
        ) as mocks:
            session.ask("And what about that?", use_router=False)
        turn = session.turns[-1]
        self.assertEqual([item.citation for item in turn.evidence], [1, 2])
        self.assertEqual([item.start_line for item in turn.evidence],
                         [1, 20])
        self.assertEqual(turn.parameters["related_evidence_count"], 1)
        self.assertEqual(turn.parameters["evidence_from_turns"], [1])
        self.assertEqual(turn.parameters["history_turn_ids"], [1])
        history = session.chat_session.restore_history.call_args.args[0]
        self.assertEqual([message["role"] for message in history],
                         ["system", "user", "assistant"])
        self.assertEqual(mocks.retrieve.call_count, 1)

    def test_search_uses_document_scope_candidates(self) -> None:
        """Pass detected document candidates to the retrieval operation."""
        session = _session()
        session.chunks = [
            chunk("a", path="one.py"),
            chunk("b", path="one.py"),
            chunk("c", path="two.py"),
        ]
        with _answer_patches(session, _intent()) as mocks:
            with patch.object(
                rag, "document_scope", return_value=(
                    (Path("one.py"),), "explicit"
                ),
            ) as scope:
                session.ask("Question", use_router=False)
        self.assertEqual(scope.call_args.args[2],
                         [Path("one.py"), Path("two.py")])
        self.assertEqual(mocks.retrieve.call_args.kwargs["scope"],
                         (Path("one.py"),))

    def test_document_answer_keeps_document_metadata(self) -> None:
        """Retain selected-document provenance and answer parameters."""
        session = _session()
        sample = chunk("a", "first", path="paper.md")
        selection = DocumentSelection(
            Path("paper.md"), (), "named_file", (sample,), 1, 2, True
        )
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_DOCUMENT)
        ) as mocks:
            mocks.read.return_value = selection
            session.ask("Read paper", use_router=False)
        turn = session.turns[-1]
        self.assertEqual(turn.evidence[0].path, Path("paper.md"))
        self.assertEqual(turn.parameters["document"]["path"], "paper.md")
        self.assertEqual(turn.parameters["document"]["max_tokens"], 1200)
        mocks.read.assert_called_once()
        self.assertEqual(mocks.read.call_args.args[:2],
                         ("Read paper", 1200))
        mocks.retrieve.assert_not_called()

    def test_document_not_found_produces_no_evidence(self) -> None:
        """Handle missing documents without inventing evidence."""
        session = _session()
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_DOCUMENT)
        ) as mocks:
            session.ask("Read missing", use_router=False)
        self.assertEqual(session.turns[-1].evidence, ())
        self.assertNotIn("document", session.turns[-1].parameters)
        mocks.read.assert_called_once()

    def test_none_mode_skips_retrieval(self) -> None:
        """Generate an answer without retrieval in no-retrieval mode."""
        session = _session()
        phases = []
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_NONE)
        ) as mocks:
            session.ask("Hello", use_router=False,
                        phase_callback=lambda phase, data: phases.append(
                            (phase, data)
                        ))
        mocks.retrieve.assert_not_called()
        mocks.read.assert_not_called()
        self.assertEqual(session.turns[-1].evidence, ())
        self.assertEqual(
            [phase for phase, data in phases if data.get("skipped")],
            [rag.ANSWER_PHASE_EVIDENCE_READY],
        )
        self.assertEqual(session.chat_session.reply.call_args.args[0],
                         "Hello")

    def test_reuse_mode_combines_prior_turns_without_search(self) -> None:
        """Reuse referenced evidence without invoking a new search."""
        session = _session()
        session.turns = [
            _turn("One", evidence=(_evidence(start=10),)),
            _turn("Two", evidence=(_evidence(start=20),)),
        ]
        decision = RouteDecision(
            "test_intent", False, related_turn_ids=(1, 2)
        )
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_REUSE), decision
        ) as mocks:
            session.ask("Reuse", use_router=False)
        turn = session.turns[-1]
        self.assertEqual([item.start_line for item in turn.evidence],
                         [10, 20])
        self.assertTrue(turn.parameters["evidence_reused"])
        self.assertEqual(turn.parameters["evidence_from_turns"], [1, 2])
        mocks.retrieve.assert_not_called()

    def test_reuse_evidence_override_uses_last_turn(self) -> None:
        """Honor explicit reuse of evidence from the previous answer."""
        session = _session()
        session.turns = [_turn(evidence=(_evidence(),))]
        with _answer_patches(session, _intent()) as mocks:
            session.ask("Reuse last", reuse_evidence=True,
                        use_router=False)
        self.assertEqual(session.turns[-1].parameters["retrieval_mode"],
                         RETRIEVAL_MODE_REUSE)
        mocks.retrieve.assert_not_called()

    def test_reuse_without_matching_evidence_falls_back_to_search(
        self,
    ) -> None:
        """Search when requested earlier evidence cannot be reused."""
        session = _session()
        session.turns = [_turn(evidence=())]
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_REUSE)
        ) as mocks:
            session.ask("Follow up", use_router=False)
        self.assertEqual(session.turns[-1].parameters["retrieval_mode"],
                         RETRIEVAL_MODE_SEARCH)
        mocks.retrieve.assert_called_once()

    def test_intent_guidance_overrides_caller_guidance(self) -> None:
        """Give configured intent guidance precedence over caller text."""
        session = _session()
        with _answer_patches(
            session, _intent(guidance="Configured guidance")
        ):
            session.ask("Question", answer_guidance="Caller guidance",
                        use_router=False)
        prompt = session.chat_session.reply.call_args.args[0]
        self.assertIn("Configured guidance", prompt)
        self.assertNotIn("Caller guidance", prompt)
        self.assertEqual(session.turns[-1].parameters["answer_guidance"],
                         "Configured guidance")

    def test_project_guidance_is_applied_without_gui(self) -> None:
        """Use project guidance automatically in direct API calls."""
        session = _session()
        intent = _intent()
        with _answer_patches(session, intent):
            with patch.object(
                rag, "project_type",
                return_value=_type_config(intent, guidance="Core instruction"),
            ):
                session.ask("Question", use_router=False)
        prompt = session.chat_session.reply.call_args.args[0]
        self.assertIn("Core instruction", prompt)
        self.assertEqual(
            session.turns[-1].parameters["answer_guidance"],
            "Core instruction",
        )

    def test_caller_guidance_overrides_project_default(self) -> None:
        """Apply explicit caller guidance instead of the config's default."""
        session = _session()
        intent = _intent()
        with _answer_patches(session, intent):
            with patch.object(
                rag, "project_type",
                return_value=_type_config(intent, guidance="Core instruction"),
            ):
                session.ask(
                    "Question", answer_guidance="Caller instruction",
                    use_router=False,
                )
        prompt = session.chat_session.reply.call_args.args[0]
        self.assertIn("Caller instruction", prompt)
        self.assertNotIn("Core instruction", prompt)
        self.assertEqual(
            session.turns[-1].parameters["answer_guidance"],
            "Caller instruction",
        )

    def test_explicit_empty_guidance_disables_project_default(self) -> None:
        """Allow callers to suppress project guidance without a GUI."""
        session = _session()
        intent = _intent()
        with _answer_patches(session, intent):
            with patch.object(
                rag, "project_type",
                return_value=_type_config(intent, guidance="Core instruction"),
            ):
                session.ask(
                    "Question", answer_guidance="", use_router=False
                )
        prompt = session.chat_session.reply.call_args.args[0]
        self.assertNotIn("Core instruction", prompt)
        self.assertEqual(
            session.turns[-1].parameters["answer_guidance"], ""
        )

    def test_intent_guidance_overrides_empty_caller_guidance(self) -> None:
        """Keep intent guidance authoritative even for empty overrides."""
        session = _session()
        intent = _intent(guidance="Intent instruction")
        with _answer_patches(session, intent):
            with patch.object(
                rag, "project_type",
                return_value=_type_config(intent, guidance="Core instruction"),
            ):
                session.ask(
                    "Question", answer_guidance="", use_router=False
                )
        prompt = session.chat_session.reply.call_args.args[0]
        self.assertIn("Intent instruction", prompt)
        self.assertNotIn("Core instruction", prompt)

    def test_tool_receives_resolved_project_guidance(self) -> None:
        """Send the same resolved guidance to tool-written answers."""
        session = _session()
        intent = _intent(tool=ToolUse("test_tool", {}))
        with _answer_patches(session, intent) as mocks:
            with patch.object(
                rag, "project_type",
                return_value=_type_config(intent, guidance="Core instruction"),
            ):
                session.ask("Question", use_router=False)
        self.assertEqual(
            mocks.run.call_args.kwargs["answer_guidance"],
            "Core instruction",
        )
        self.assertEqual(
            session.turns[-1].parameters["answer_guidance"],
            "Core instruction",
        )

    def test_invalid_citations_are_flagged_without_rewriting_answer(
        self,
    ) -> None:
        """Record nonexistent citation IDs without changing model text."""
        session = _session()
        answer = "Supported [1], invented [5, 9], repeated [5]."
        session.chat_session.reply.return_value = answer
        with _answer_patches(session, _intent()):
            self.assertEqual(
                session.ask("Question", use_router=False), answer
            )
        self.assertEqual(session.turns[-1].answer, answer)
        self.assertEqual(
            session.turns[-1].parameters["invalid_citations"], [5, 9]
        )
        session.chat_session.reply.assert_called_once()

    def test_valid_and_missing_citations_need_no_metadata(self) -> None:
        """Do not add warnings when the answer cites only its evidence."""
        for answer in ("Supported [1]", "No numbered citations"):
            with self.subTest(answer=answer):
                session = _session()
                session.chat_session.reply.return_value = answer
                with _answer_patches(session, _intent()):
                    session.ask("Question", use_router=False)
                self.assertNotIn(
                    "invalid_citations", session.turns[-1].parameters
                )

    def test_tool_citations_are_checked_against_tool_evidence(self) -> None:
        """Validate tool citations against their returned evidence IDs."""
        session = _session()
        intent = _intent(tool=ToolUse("test_tool", {}))
        with _answer_patches(session, intent) as mocks:
            mocks.run.return_value = ToolResult(
                "Tool result [2] [99]", (asdict(_evidence(2)),),
                {"name": "test_tool"},
            )
            session.ask("Question", use_router=False)
        self.assertEqual(
            session.turns[-1].parameters["invalid_citations"], [99]
        )

    def test_generation_settings_and_finish_reason_are_recorded(self) -> None:
        """Persist effective generation settings and valid finish reasons."""
        session = _session()
        session.chat_session.last_finish_reason = "length"
        with _answer_patches(session, _intent()) as mocks:
            session.ask(
                "Question", use_router=False,
                generation_settings={"temperature": 0.5},
            )
        self.assertEqual(
            session.chat_session.configure_generation.call_args.args[0],
            {"temperature": 0.5},
        )
        self.assertIn("trace_turn_id",
                      session.chat_session.configure_generation.call_args
                      .kwargs)
        self.assertEqual(session.turns[-1].parameters["finish_reason"],
                         "length")
        mocks.save.assert_called_once()
        session.turns.clear()
        session.chat_session.last_finish_reason = "unknown"
        with _answer_patches(session, _intent()):
            session.ask("Again", use_router=False)
        self.assertNotIn("finish_reason", session.turns[-1].parameters)

    def test_tool_answer_bypasses_chat_and_streams_once(self) -> None:
        """Use the selected tool instead of an answering model."""
        session = _session()
        use = ToolUse("project_inventory", object())
        tokens = Mock()
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_NONE, tool=use)
        ) as mocks:
            answer = session.ask("Inventory", use_router=False,
                                 token_callback=tokens)
        self.assertEqual(answer, "tool answer")
        session.chat_session.reply.assert_not_called()
        self.assertEqual(session.turns[-1].parameters["tool"],
                         {"name": "tool"})
        self.assertEqual(session.turns[-1].evidence, (_evidence(),))
        tokens.assert_called_once_with("tool answer")
        forwarded = mocks.run.call_args.kwargs["token_callback"]
        self.assertTrue(callable(forwarded))
        # A wrapper tracks the first output and formats worded citations.
        self.assertIsNot(forwarded, tokens)

    def test_tool_worded_citations_match_stream_and_saved_answer(
        self,
    ) -> None:
        """Apply the same citation policy to a tool-written response."""

        session = _session()
        use = ToolUse("project_inventory", object())
        raw = "Tool [source 1]. Literal `[source 2]` stays."
        pieces: list[str] = []

        def run(*args: object, **kwargs: object) -> ToolResult:
            """Simulate a tool sending its answer in partial fragments."""

            callback = kwargs["token_callback"]
            for fragment in (raw[:8], raw[8:15], raw[15:]):
                callback(fragment)
            return ToolResult(
                raw, (asdict(_evidence()),), {"name": "tool"},
                streamed=True,
            )

        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_NONE, tool=use)
        ) as mocks:
            mocks.run.side_effect = run
            answer = session.ask(
                "Inventory", use_router=False, token_callback=pieces.append
            )
        self.assertEqual(answer, rag._numbered_citations(raw))
        self.assertEqual("".join(pieces), answer)
        self.assertEqual(session.turns[-1].answer, answer)

    def test_tool_that_streamed_does_not_repeat_answer(self) -> None:
        """Avoid replaying an answer already streamed by the tool."""
        session = _session()
        use = ToolUse("project_inventory", object())
        tokens = Mock()
        with _answer_patches(
            session, _intent(RETRIEVAL_MODE_NONE, tool=use)
        ) as mocks:
            mocks.run.return_value = ToolResult(
                "streamed", (), {}, streamed=True
            )
            session.ask("Inventory", token_callback=tokens,
                        use_router=False)
        tokens.assert_not_called()
        self.assertEqual(session.turns[-1].answer, "streamed")

    def test_tool_writer_streams_and_saves_thinking(self) -> None:
        """Keep a user-visible tool answer intact through commitment."""

        session = _session()
        session.chat_session.thinking = True
        raw = "<think>Reading the paper</think>Overview [1]."
        tokens = Mock()
        intent = _intent(
            RETRIEVAL_MODE_NONE,
            tool=ToolUse("paper_overview", object()),
        )

        def model_reply(_prompt: str, **options: object) -> str:
            """Simulate streaming the complete answering-model output."""

            callback = options.get("token_callback")
            if callback is not None:
                callback(raw)
            return raw

        def tool(_use: object, request: object, context: object) -> ToolResult:
            """Write an answer through the real tool context adapter."""

            answer = context.write("Summarize", request.question, 100)
            return ToolResult(
                answer, (asdict(_evidence()),),
                {"name": "paper_overview"}, streamed=True,
            )

        session.chat_session.reply.side_effect = model_reply
        with _answer_patches(session, intent) as mocks:
            mocks.run.side_effect = lambda *args, **kwargs: (
                rag.RAGSession._run_tool(session, *args, **kwargs)
            )
            with patch.object(rag, "run_tool", side_effect=tool):
                answer = session.ask(
                    "Summarize the paper", use_router=False,
                    token_callback=tokens,
                )

        self.assertEqual(answer, raw)
        self.assertEqual(session.turns[-1].answer, raw)
        self.assertNotIn("invalid_citations", session.turns[-1].parameters)
        tokens.assert_called_once_with(raw)
        session.chat_session.reply.assert_called_once()
        self.assertTrue(session.chat_session.thinking)

    def test_tool_ask_model_thinking_is_shown_without_parser_leak(
        self,
    ) -> None:
        """Aggregate internal thoughts but parse only clean model results."""

        session = _session()
        session.chat_session.thinking = True
        tokens = Mock()
        intent = _intent(
            RETRIEVAL_MODE_NONE,
            tool=ToolUse("filter_results", object()),
        )
        session.chat_session.reply.side_effect = [
            "<think>Check [99]</think>1, 2",
            "Reason about scope</think>3",
        ]

        def tool(_use: object, request: object, context: object) -> ToolResult:
            """Validate the replies that reach a real tool parser."""

            self.assertEqual(context.ask_model("System", "First", 20),
                             "1, 2")
            self.assertEqual(context.ask_model("System", "Second", 20),
                             "3")
            return ToolResult(
                "Found [1].", (asdict(_evidence()),),
                {"name": "filter_results"},
            )

        with _answer_patches(session, intent) as mocks:
            mocks.run.side_effect = lambda *args, **kwargs: (
                rag.RAGSession._run_tool(session, *args, **kwargs)
            )
            with patch.object(rag, "run_tool", side_effect=tool):
                answer = session.ask(
                    "Filter items", use_router=False,
                    token_callback=tokens,
                )

        self.assertEqual(
            answer,
            "<think>Model call 1:\nCheck [99]\n\n</think>\n\n"
            "<think>Model call 2:\nReason about scope\n\n</think>"
            "\n\nFound [1].",
        )
        streamed = "".join(call.args[0] for call in tokens.call_args_list)
        self.assertEqual(streamed, answer)
        self.assertEqual(rag.strip_thinking(streamed), "Found [1].")
        self.assertEqual(session.turns[-1].answer, answer)
        self.assertNotIn("invalid_citations", session.turns[-1].parameters)
        self.assertEqual(session.chat_session.reply.call_count, 2)
        self.assertTrue(session.chat_session.thinking)

    def test_tool_ask_model_thinking_and_writer_stream_together(
        self,
    ) -> None:
        """Preserve thoughts from intermediate and final model calls."""

        session = _session()
        tokens = Mock()
        session.chat_session.reply.side_effect = [
            "<think>Plan</think>selected",
            "<think>Compose</think>Final [1].",
        ]

        def tool(_use: object, request: object, context: object) -> ToolResult:
            """Use both adapters without duplicating streamed thoughts."""

            self.assertEqual(context.ask_model("s", "q", 20), "selected")
            answer = context.write("s", "q", 20)
            return ToolResult(
                answer, (asdict(_evidence()),),
                {"name": "paper_overview"}, streamed=True,
            )

        def reply(_prompt: str, **options: object) -> str:
            """Emit raw writer tokens as the real chat session would."""

            response = next(responses)
            callback = options.get("token_callback")
            if callback is not None:
                callback(response)
            return response

        responses = iter(session.chat_session.reply.side_effect)
        session.chat_session.reply.side_effect = reply
        with patch.object(rag, "run_tool", side_effect=tool):
            result = session._run_tool(
                ToolUse("paper_overview", object()), "Overview",
                RouteDecision("paper_overview", True),
                trace_turn_id=None, report_phase=Mock(),
                token_callback=tokens,
            )
        expected = (
            "<think>Model call 1:\nPlan\n\n</think>\n\n"
            "<think>Compose</think>Final [1]."
        )
        self.assertTrue(result.streamed)
        self.assertEqual(result.answer, expected)
        self.assertEqual(
            "".join(call.args[0] for call in tokens.call_args_list),
            expected,
        )

    def test_generation_failure_does_not_append_turn(self) -> None:
        """Avoid appending a turn when generation raises an error."""
        session = _session()
        session.chat_session.reply.side_effect = RuntimeError("model failed")
        with _answer_patches(session, _intent()) as mocks:
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                session.ask("Question", use_router=False)
        self.assertEqual(session.turns, [])
        mocks.save.assert_not_called()

    def test_save_failure_retains_only_committed_turns(self) -> None:
        """Exclude a generated answer when its checkpoint cannot be saved."""
        session = _session()
        with _answer_patches(session, _intent()) as mocks:
            mocks.save.side_effect = OSError("disk full")
            with self.assertRaisesRegex(OSError, "disk full"):
                session.ask("Question", use_router=False)
        self.assertEqual(session.turns, [])
        self.assertIsNone(session._checkpoint_id)
        session.chat_session.restore_history.assert_called()

    def test_phase_callback_exception_propagates(self) -> None:
        """Propagate exceptions raised by the interface phase callback."""
        session = _session()
        def fail(phase: str, details: dict) -> None:
            """Simulate a GUI callback failure."""
            raise RuntimeError("callback failure")
        with _answer_patches(session, _intent()):
            with self.assertRaisesRegex(RuntimeError, "callback failure"):
                session.ask("Question", use_router=False,
                            phase_callback=fail)
        self.assertEqual(session.turns, [])


class TestRAGIntentThinking(unittest.TestCase):
    """Exercise intent policy with real reply orchestration and fake models."""

    def _chat(self, *, thinking: bool) -> rag.ChatSession:
        """Create a chat with real settings and a deterministic tokenizer.

        Parameters
        ----------
        thinking: bool
            Configured setting that intent overrides must not mutate.

        Returns
        -------
        kilin.ChatSession
            Session whose model generation must be patched by the test.

        """

        with patch.object(rag.ChatSession, "_load"):
            chat = rag.ChatSession(
                "chat-model", backend="torch", thinking=thinking,
                max_new_tokens=16, block_unrequested_cjk=False,
            )
        self.addCleanup(chat.close)
        chat._model = object()
        chat._tokenizer = Mock()
        chat._tokenizer.apply_chat_template.return_value = [1, 2, 3]
        chat._tokenizer.encode.return_value = [4]
        return chat

    def test_intent_policy_propagates_without_reconfiguration(self) -> None:
        """Apply generic intent policies without a small-talk special case."""

        for configured in (True, False):
            for override in (None, True, False):
                with self.subTest(configured=configured, override=override):
                    session = _session()
                    session.chat_session.thinking = configured
                    intent = replace(_intent(), thinking=override)
                    with _answer_patches(session, intent) as mocks:
                        session.ask("Question", use_router=False)
                    options = session.chat_session.reply.call_args.kwargs
                    if override is None:
                        self.assertNotIn("thinking", options)
                    else:
                        self.assertIs(options["thinking"], override)
                    expected = (
                        configured if override is None else override
                    )
                    parameters = session.turns[-1].parameters
                    self.assertIs(parameters["thinking"], configured)
                    self.assertIs(parameters["thinking_effective"], expected)
                    self.assertIn(
                        "thinking on" if expected else "thinking off",
                        parameters["model_summary"],
                    )
                    self.assertIs(session.chat_session.thinking, configured)
                    configure = session.chat_session.configure_generation
                    configure.assert_not_called()
                    session.chat_session.reply.assert_called_once()
                    mocks.route.assert_called_once()

    def test_intent_override_has_precedence_over_generation_update(
        self,
    ) -> None:
        """Retain explicit session updates while overriding only the answer."""

        for configured, updated, override in (
            (False, True, False), (True, False, True), (False, True, None),
        ):
            with self.subTest(updated=updated, override=override):
                session = _session()
                chat = self._chat(thinking=configured)
                session.chat_session = chat
                intent = replace(_intent(), thinking=override)
                with (
                    _answer_patches(session, intent),
                    patch.object(chat, "_torch_reply", return_value="Answer"),
                ):
                    session.ask(
                        "Question", use_router=False,
                        generation_settings={"thinking": updated},
                    )
                expected = updated if override is None else override
                self.assertIs(chat.thinking, updated)
                self.assertIs(
                    chat._tokenizer.apply_chat_template.call_args.kwargs[
                        "enable_thinking"
                    ], expected,
                )
                parameters = session.turns[-1].parameters
                self.assertIs(parameters["thinking"], updated)
                self.assertIs(parameters["thinking_effective"], expected)

    def test_small_talk_then_normal_answer_uses_configured_thinking(
        self,
    ) -> None:
        """Packaged small talk affects only its answer on the same model."""

        for type_id in ("code", "literature"):
            with self.subTest(type_id=type_id):
                config = rag.project_type(type_id)
                session = _session()
                session.project.profile.type_id = type_id
                chat = self._chat(thinking=True)
                session.chat_session = chat
                small_talk = config.intents["small_talk"]
                normal = config.intents[config.default_intent]
                with patch.object(
                    chat, "_torch_reply", return_value="Answer"
                ) as generate:
                    with (
                        _answer_patches(session, small_talk) as mocks,
                        patch.object(rag, "project_type", return_value=config),
                    ):
                        mocks.route.side_effect = lambda *args: (
                            rag.RAGSession._route(session, *args)
                        )
                        session.ask("Hello")
                    with _answer_patches(session, normal):
                        session.ask("Explain the sources", use_router=False)
                calls = chat._tokenizer.apply_chat_template.call_args_list
                self.assertEqual(
                    [call.kwargs["enable_thinking"] for call in calls],
                    [False, True],
                )
                self.assertEqual(generate.call_count, 2)
                self.assertIs(session.chat_session, chat)
                self.assertTrue(chat.thinking)
                self.assertEqual(
                    [turn.parameters["thinking"] for turn in session.turns],
                    [True, True],
                )
                self.assertEqual(
                    [turn.parameters["thinking_effective"]
                     for turn in session.turns],
                    [False, True],
                )

    def test_failed_small_talk_does_not_disable_following_answer(
        self,
    ) -> None:
        """A failed answer leaves configured thinking available to retries."""

        session = _session()
        chat = self._chat(thinking=True)
        session.chat_session = chat
        with (
            _answer_patches(session, replace(_intent(), thinking=False)),
            patch.object(
                chat, "_torch_reply", side_effect=RuntimeError("failed")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                session.ask("First", use_router=False)
        self.assertTrue(chat.thinking)
        self.assertEqual(session.turns, [])
        with (
            _answer_patches(session, _intent()),
            patch.object(chat, "_torch_reply", return_value="Answer"),
        ):
            session.ask("Second", use_router=False)
        self.assertTrue(
            chat._tokenizer.apply_chat_template.call_args.kwargs[
                "enable_thinking"
            ]
        )
        self.assertEqual(len(session.turns), 1)

    def test_tool_answer_has_no_direct_model_thinking_metadata(self) -> None:
        """Do not claim one effective policy for tool-owned model calls."""

        session = _session()
        intent = _intent(
            RETRIEVAL_MODE_NONE,
            tool=ToolUse("project_inventory", object()),
        )
        with _answer_patches(session, intent):
            session.ask("Inventory", use_router=False)
        session.chat_session.reply.assert_not_called()
        session.chat_session.configure_generation.assert_not_called()
        self.assertNotIn("thinking_effective", session.turns[-1].parameters)

    def test_saved_small_talk_restores_configured_not_effective_thinking(
        self,
    ) -> None:
        """Round-trip real checkpoints with tracing both enabled and off."""

        for tracing in (False, True):
            with self.subTest(tracing=tracing), TemporaryDirectory() as root:
                session = _session(Path(root))
                chat = self._chat(thinking=True)
                session.chat_session = chat
                trace = (
                    TraceLogger(
                        TraceConfig(
                            enabled=True, output_path=session.conversation_path
                        ),
                        default_directory=Path(root),
                    ) if tracing else NullTraceLogger()
                )
                session.trace = trace
                chat.trace = trace
                try:
                    with (
                        _answer_patches(
                            session, replace(_intent(), thinking=False)
                        ) as mocks,
                        patch.object(
                            chat, "_torch_reply", return_value="Answer"
                        ),
                    ):
                        mocks.save.side_effect = lambda **options: (
                            rag.RAGSession._save_conversation(
                                session, **options
                            )
                        )
                        session.ask("Hello", use_router=False)
                    _, turns, _ = session._load_conversation()
                    self.assertIs(turns[0].parameters["thinking"], True)
                    self.assertIs(
                        turns[0].parameters["thinking_effective"], False
                    )
                    chat.configure_generation({"thinking": False})
                    session.switch_conversation(
                        session.conversation_path, trace=trace
                    )
                    self.assertTrue(chat.thinking)
                    self.assertIs(session.chat_session, chat)
                    self.assertEqual(len(session.turns), 1)
                finally:
                    session.close()
                    trace.close()


class TestRAGRoutingAndTools(unittest.TestCase):
    """Test intent routing, prior-turn references, and tool adapters."""

    def test_routing_off_uses_default_without_loading_router(self) -> None:
        """Select the default intent without loading a routing model."""
        session = _session()
        config = _type_config(_intent())
        session._router_chat = Mock()
        report = Mock()
        decision = session._route("Question", config, False, "turn", report)
        self.assertEqual(decision.intent, "test_intent")
        self.assertTrue(decision.standalone)
        self.assertEqual(decision.fallback, "routing off")
        session._router_chat.assert_not_called()
        report.assert_not_called()
        session.turns = [_turn()]
        self.assertFalse(session._route(
            "Followup", config, False, "turn", report
        ).standalone)

    def test_writing_without_router_uses_grounded_default(self) -> None:
        """Use evidence-based writing when intent routing is disabled.

        """

        config = project_type("writing")
        session = _session()
        session._router_chat = Mock()
        report = Mock()
        decision = session._route(
            "Draft a methods section", config, False, "turn", report
        )
        self.assertEqual(decision.intent, "grounded_writing")
        self.assertEqual(decision.fallback, "routing off")
        self.assertEqual(
            config.intents[decision.intent].retrieval_mode, "search"
        )
        session._router_chat.assert_not_called()
        report.assert_not_called()

    def test_routing_on_records_classification_and_fallback(self) -> None:
        """Trace route decisions and recover from invalid classifications."""
        session = _session()
        config = _type_config(_intent())
        session.trace = MagicMock()
        session.trace.include_content = True
        decision = RouteDecision(
            "test_intent", True, raw="raw reply", fallback="bad JSON",
            dropped_turns=((2, "invalid"),),
        )
        session._router_chat = Mock(return_value=sentinel.router)
        report = Mock()
        with patch.object(rag, "route_question", return_value=decision) as rt:
            self.assertIs(session._route(
                "Question", config, True, "turn", report
            ), decision)
        rt.assert_called_once_with(
            sentinel.router, config, session.turns, "Question"
        )
        report.assert_called_once_with(
            rag.ANSWER_PHASE_ROUTING, message="Identifying intent"
        )
        event = [
            call for call in session.trace.emit.call_args_list
            if call.args[0] == "intent.classification.decided"
        ][0]
        self.assertEqual(event.kwargs["data"]["raw"], "raw reply")
        self.assertEqual(event.kwargs["data"]["dropped_turns"],
                         [{"turn_id": 2, "reason": "invalid"}])

    def test_earlier_turns_walks_related_chain_and_marks_last(self) -> None:
        """Follow related-turn links and identify the latest turn."""
        session = _session()
        session.turns = [
            _turn("First", evidence=(_evidence(start=1),)),
            _turn("Second", parameters={"related_turn_ids": [1]}),
            _turn("Third"),
        ]
        decision = RouteDecision(
            "test_intent", False, related_turn_ids=(2,)
        )
        earlier = session._earlier_turns(decision)
        self.assertEqual([item.number for item in earlier], [3, 2, 1])
        self.assertEqual([item.linked for item in earlier],
                         [False, True, True])
        self.assertEqual(earlier[-1].paths, (Path("src/a.py"),))
        self.assertEqual(earlier[-1].texts, ("source text",))
        self.assertEqual(earlier[-1].lines, (1,))

    def test_earlier_turns_excludes_model_thinking(self) -> None:
        """Give tool follow-ups only the earlier answer, not its thoughts."""

        session = _session()
        first = replace(
            _turn("Question"),
            answer="<think>Private [99]</think>Result [1].",
        )
        session.turns = [first]
        earlier = session._earlier_turns(
            RouteDecision("test_intent", False, refers_back=True)
        )
        self.assertEqual(earlier[0].answer, "Result [1].")

    def test_earlier_turns_handles_back_reference_without_links(self) -> None:
        """Resolve backward references when explicit links are missing."""
        session = _session()
        session.turns = [_turn("First")]
        self.assertEqual(session._earlier_turns(
            RouteDecision("test_intent", False)
        ), ())
        earlier = session._earlier_turns(
            RouteDecision("test_intent", False, refers_back=True)
        )
        self.assertEqual(len(earlier), 1)
        self.assertTrue(earlier[0].linked)

    def test_earlier_turns_rejects_invalid_links_and_caps_length(self) -> None:
        """Reject invalid earlier-turn links and cap the history depth."""
        session = _session()
        session.turns = [
            _turn(str(number), parameters={"related_turn_ids": [
                number - 1, "bad", 999
            ]})
            for number in range(1, 15)
        ]
        decision = RouteDecision(
            "test_intent", False, related_turn_ids=(14, 0, 100)
        )
        earlier = session._earlier_turns(decision)
        self.assertEqual(len(earlier), rag._MAX_EARLIER_TURNS)
        self.assertEqual(earlier[0].number, 14)
        self.assertEqual(earlier[-1].number, 9)

    def test_run_tool_builds_context_and_preserves_callbacks(self) -> None:
        """Pass required context and callbacks to the selected tool."""
        session = _session()
        session.chunks = [chunk("one")]
        session._rank_everything = Mock(return_value=[ranked(chunk("one"))])
        session._ask_router = Mock(return_value="router")
        session._earlier_turns = Mock(return_value=())
        session._source_files = Mock(return_value={Path("a.py"): 2})
        session._unsearchable_sources = Mock(return_value=[])
        use = ToolUse("project_inventory", object())
        decision = RouteDecision("test_intent", True)
        phases = Mock()
        callback = Mock()
        writer = Mock()
        choose = Mock()
        tokens = Mock()
        result = ToolResult("result", (), {"ok": True})
        with patch.object(rag, "run_tool", return_value=result) as runner:
            actual = session._run_tool(
                use, "Inventory", decision, trace_turn_id="t",
                report_phase=phases, confirm_callback=callback,
                pdf_writer=writer, choice_callback=choose,
                token_callback=tokens, answer_guidance="Short",
            )
        self.assertIs(actual, result)
        request = runner.call_args.args[1]
        context = runner.call_args.args[2]
        self.assertEqual(request.question, "Inventory")
        self.assertEqual(request.guidance, "Short")
        self.assertIs(context.confirm, callback)
        self.assertIs(context.write_pdf, writer)
        self.assertIs(context.choose, choose)
        # Cards are read through the session, without a model call.
        self.assertEqual(context.card.__func__, rag.RAGSession._card_fields)
        self.assertEqual(context.model_id, "chat-model")
        self.assertEqual(context.rank("topic"), [ranked(chunk("one"))])
        self.assertEqual(context.source_files(), {Path("a.py"): 2})
        self.assertEqual(context.unsearchable([]), [])
        context.progress("step", current=1)
        context.done("done")
        self.assertEqual(phases.call_args_list[0].args[0],
                         rag.ANSWER_PHASE_PROCESSING)
        self.assertEqual(phases.call_args_list[1].args[0],
                         rag.ANSWER_PHASE_EVIDENCE_READY)
        self.assertEqual(phases.call_args_list[1].kwargs["tool"],
                         use.name)

    def test_tool_runs_with_project_settings_and_is_traced(self) -> None:
        """Saved tool settings apply; each run is recorded with them."""
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "profile.ini").write_text(
                "[tool.project_inventory]\nmax_listed = 7\n",
                encoding="utf-8",
            )
            session = _session(root)
            session._earlier_turns = Mock(return_value=())
            trace = MagicMock()
            session.trace = trace
            settings = TOOL_REGISTRY["project_inventory"].parse_settings(
                {}, "test"
            )
            use = ToolUse("project_inventory", settings)
            result = ToolResult("result", (), {"ok": True})
            with patch.object(rag, "run_tool", return_value=result) as runner:
                session._run_tool(
                    use, "Inventory", RouteDecision("test_intent", True),
                    trace_turn_id="t", report_phase=Mock(),
                )
        ran = runner.call_args.args[0]
        self.assertEqual(ran.settings.max_listed, 7)
        identity = next(
            info.identity() for info in list_tools()
            if info.name == use.name and info.available
        )
        self.assertEqual(identity["name"], "project_inventory")
        self.assertEqual(identity["origin"], "distribution")
        self.assertEqual(identity["version"], TOOL_REGISTRY[use.name].version)
        self.assertEqual(identity["api_version"],
                         TOOL_REGISTRY[use.name].api_version)
        self.assertRegex(identity["sha256"], r"^[0-9a-f]{64}$")
        trace.span.assert_any_call(
            "tool.run",
            component="tool",
            data={
                "name": "project_inventory",
                "plugin": identity,
                "settings": asdict(ran.settings),
                # The tool's own fixed values, from its defaults.json.
                "internal": {"MODEL_CALLS": {"THINKING": "inherit"}},
            },
            turn_id="t",
        )

    def test_tool_side_question_and_writer_use_isolated_reply(self) -> None:
        """Keep auxiliary tool replies separate from conversation history."""
        session = _session()
        session._rank_everything = Mock(return_value=[])
        use = ToolUse("project_inventory", object())
        tokens = Mock()
        with patch.object(rag, "run_tool") as runner, \
            patch.object(rag, "side_reply", return_value="reply") as side:
            session._run_tool(
                use, "Question", RouteDecision("test_intent", True),
                trace_turn_id=None, report_phase=Mock(),
                token_callback=tokens,
            )
            context = runner.call_args.args[2]
            self.assertEqual(context.ask_model("s", "q", None), "reply")
            self.assertEqual(context.ask_model("s", "q", 5), "reply")
            self.assertEqual(context.write("s", "q", 6), "reply")
        self.assertEqual(side.call_args_list[0].kwargs,
                         {"max_new_tokens": None,
                          "language_context": ("Question",),
                          "inherit_settings": True,
                          "preserve_thinking": True,
                          "require_complete": True,
                          "thinking": None})
        self.assertEqual(side.call_args_list[1].kwargs,
                         {"max_new_tokens": 5,
                          "language_context": ("Question",),
                          "inherit_settings": True,
                          "preserve_thinking": True,
                          "require_complete": True,
                          "thinking": None})
        self.assertEqual(side.call_args_list[2].kwargs,
                         {"max_new_tokens": 6,
                          "token_callback": tokens,
                          "language_context": ("Question",),
                          "inherit_settings": True,
                          "preserve_thinking": True,
                          "require_complete": True})

    def test_tool_writer_preserves_thinking_only_for_visible_answer(
        self,
    ) -> None:
        """Keep raw output from write but strip internal model replies."""

        session = _session()
        session.chat_session.thinking = True
        raw = "<think>Reasoning</think>Overview [1]."
        session.chat_session.reply.return_value = raw
        with patch.object(rag, "run_tool") as runner:
            session._run_tool(
                ToolUse("paper_overview", object()), "Summarize",
                RouteDecision("paper_overview", True),
                trace_turn_id=None, report_phase=Mock(),
            )
            context = runner.call_args.args[2]
            self.assertEqual(context.ask_model("s", "q", 80),
                             "Overview [1].")
            self.assertEqual(context.write("s", "q", 80), raw)
        self.assertEqual(session.chat_session.reply.call_count, 2)
        session.chat_session.restore_history.assert_called()


if __name__ == "__main__":
    unittest.main()
