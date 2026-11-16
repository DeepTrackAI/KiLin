"""Verify core answer activity persistence without the desktop interface.

The RAG state machine runs against offline fixtures and actual conversation
checkpoint writes; no models are loaded and tracing may be disabled.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kilin import rag
from kilin._model_completion import GenerationCancelled
from kilin.conversation_format import conversation_state
from kilin.project_type import RETRIEVAL_MODE_NONE
from kilin.tools import ToolUse
from kilin.trace_logging import TraceConfig, TraceLogger
from tests.kilin._rag_support import (
    _answer_patches,
    _intent,
    _session,
)


class TestRAGAnswerTimeline(unittest.TestCase):
    """Ensure timelines belong to successful, persisted conversation turns."""

    def setUp(self) -> None:
        """Create an isolated session without model initialization."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.session = _session(self.root)
        self.session.conversation_path.parent.mkdir(parents=True)

    def _answer(self, question: str, intent: object) -> str:
        """Run real answering and persistence with mocked model calls."""
        with _answer_patches(self.session, intent) as mocks:
            mocks.save.side_effect = lambda **options: (
                rag.RAGSession._save_conversation(self.session, **options)
            )
            return self.session.ask(question, use_router=False)

    def _stored_turn(self) -> dict[str, object]:
        """Read the first turn from the authoritative saved checkpoint."""
        data = json.loads(
            self.session.conversation_path.read_text(encoding="utf-8")
        )
        return conversation_state(data)["turns"][0]

    def test_script_answer_has_timeline_without_callback_or_trace(
        self,
    ) -> None:
        """Save a timeline even without GUI events or diagnostic tracing."""
        self.assertFalse(self.session.trace.enabled)
        self.assertEqual(self._answer("Question", _intent()), "model answer")
        timeline = self.session.turns[-1].timeline
        self.assertIsNotNone(timeline)
        self.assertGreaterEqual(timeline.elapsed_ms, 0)
        self.assertEqual(
            sum(step.ms for step in timeline.steps), timeline.elapsed_ms
        )
        self.assertIn("retrieving", [step.phase for step in timeline.steps])
        self.assertIn("generating", [step.phase for step in timeline.steps])
        self.assertNotIn("completed", [
            step.phase for step in timeline.steps
        ])
        self.assertEqual(timeline.steps[-1].phase, "saving")
        self.assertEqual(self._stored_turn()["timeline"]["elapsed_ms"],
                         timeline.elapsed_ms)
        _, restored, _ = self.session._load_conversation()
        self.assertEqual(restored[-1].timeline, timeline)
        self.assertEqual(restored[-1].answer, "model answer")

    def test_timeline_is_persisted_in_same_traced_checkpoint(self) -> None:
        """Preserve core timing inside a traced conversation's resume state."""
        logger = TraceLogger(
            TraceConfig(enabled=True,
                        output_path=self.session.conversation_path),
            default_directory=self.root,
        )
        self.addCleanup(logger.close)
        self.session.trace = logger
        self._answer("Question with trace", _intent())
        self.assertIsNotNone(self.session.turns[0].timeline)
        self.assertEqual(
            self._stored_turn()["timeline"]["steps"][0]["phase"],
            "routed",
        )
        _, restored, _ = self.session._load_conversation()
        self.assertEqual(restored, self.session.turns)

    def test_live_activities_match_turn_and_diagnostic_trace(self) -> None:
        """Emit, trace, and save exactly the same ordered activity list."""
        from kilin.trace_logging import TRACE_EVENT_ANSWER_ACTIVITY

        logger = TraceLogger(
            TraceConfig(enabled=True,
                        output_path=self.session.conversation_path),
            default_directory=self.root,
        )
        self.addCleanup(logger.close)
        self.session.trace = logger
        events: list[dict[str, object]] = []
        phases: list[tuple[str, dict[str, object]]] = []
        with _answer_patches(
            self.session, _intent(RETRIEVAL_MODE_NONE)
        ) as mocks:
            mocks.save.side_effect = lambda **options: (
                rag.RAGSession._save_conversation(self.session, **options)
            )
            self.session.ask(
                "Check the references", use_router=False,
                activity_callback=events.append,
                phase_callback=lambda phase, details: phases.append(
                    (phase, details)
                ),
            )
        routed = next(
            details for phase, details in phases
            if phase == rag.ANSWER_PHASE_ROUTED
        )
        self.assertNotIn("tool_name", routed)
        steps = self.session.turns[-1].timeline.steps
        self.assertEqual(
            steps[0].label, "Test intent identified (test_intent)"
        )
        self.assertEqual(
            [(item["phase"], item["label"]) for item in events],
            [(step.phase, step.label) for step in steps],
        )
        self.assertEqual(
            [item["index"] for item in events], list(range(len(steps)))
        )
        self.assertIn("No earlier conversation needed", [
            step.label for step in steps
        ])
        self.assertIn("No retrieval needed", [
            step.label for step in steps
        ])
        traced = [event["data"] for event in logger._events
                  if event["event_type"] == TRACE_EVENT_ANSWER_ACTIVITY]
        self.assertEqual(traced, events)
        # Assert the events are present on disk, not only in memory.
        payload = json.loads(
            self.session.conversation_path.read_text(encoding="utf-8")
        )
        stages = (payload["sessions"]["05_conversations"]
                  ["turns"]["turn_0001"]["stages"])
        operations = (
            value for stage in stages.values()
            for value in stage["operations"].values()
        )
        disk_events = sorted((
            item["details"] for item in operations
            if "index" in item.get("details", {})
            and "started_ms" in item["details"]
        ), key=lambda item: item["index"])
        self.assertEqual(disk_events, events)
        self.assertEqual(
            [(item["phase"], item["label"]) for item in traced],
            [(step.phase, step.label) for step in steps],
        )
        self.assertEqual(
            self._stored_turn()["timeline"]["steps"],
            [{"phase": item.phase, "label": item.label, "ms": item.ms}
             for item in steps],
        )

    def test_activities_are_reported_with_diagnostics_disabled(self) -> None:
        """Core event delivery must not depend on optional trace writes."""
        observed: list[dict[str, object]] = []
        with _answer_patches(
            self.session, _intent(RETRIEVAL_MODE_NONE)
        ) as mocks:
            mocks.save.side_effect = lambda **options: (
                rag.RAGSession._save_conversation(self.session, **options)
            )
            self.session.ask(
                "Check references", use_router=False,
                activity_callback=observed.append,
            )
        self.assertFalse(self.session.trace.enabled)
        self.assertEqual(
            [event["label"] for event in observed],
            [step.label for step in self.session.turns[-1].timeline.steps],
        )

    def test_first_stream_piece_has_a_core_recorded_boundary(self) -> None:
        """Keep prompt reading and first output in the same activity stream."""
        observed: list[dict[str, object]] = []
        chunks: list[str] = []
        with _answer_patches(
            self.session, _intent(RETRIEVAL_MODE_NONE)
        ) as mocks:
            def reply(*args, **kwargs):
                """Simulate an already requested streaming backend."""
                callback = kwargs["token_callback"]
                callback("Part one")
                callback("Part two")
                return "Part onePart two"

            self.session.chat_session.reply.side_effect = reply
            mocks.save.side_effect = lambda **options: (
                rag.RAGSession._save_conversation(self.session, **options)
            )
            self.session.ask(
                "Explain", use_router=False,
                token_callback=chunks.append,
                activity_callback=observed.append,
            )
        self.assertEqual(chunks, ["Part one", "Part two"])
        labels = [event["label"] for event in observed]
        self.assertIn("Reading the conversation", labels)
        self.assertIn("Generating the answer", labels)
        self.assertLess(
            labels.index("Reading the conversation"),
            labels.index("Generating the answer"),
        )
        self.assertEqual(
            labels,
            [step.label for step in self.session.turns[0].timeline.steps],
        )

    def test_each_answer_keeps_its_own_timeline(self) -> None:
        """Keep separate timelines when answering multiple questions."""
        self._answer("First", _intent())
        self._answer("Second", _intent(RETRIEVAL_MODE_NONE))
        saved = conversation_state(json.loads(
            self.session.conversation_path.read_text(encoding="utf-8")
        ))["turns"]
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0]["timeline"]["steps"][0]["phase"],
                         "routed")
        self.assertEqual(saved[1]["timeline"]["steps"][0]["phase"],
                         "routed")
        self.assertEqual(
            [turn.timeline for turn in self.session.turns],
            [turn.timeline for turn in self.session._load_conversation()[1]],
        )

    def test_tool_answer_also_has_timeline(self) -> None:
        """Keep timing when a tool, rather than ChatSession, writes a turn."""
        tool = ToolUse("project_inventory", object())
        self._answer("Use the tool", _intent(RETRIEVAL_MODE_NONE, tool=tool))
        self.assertIsNotNone(self.session.turns[0].timeline)
        self.assertEqual(
            self._stored_turn()["timeline"]["elapsed_ms"],
            self.session.turns[0].timeline.elapsed_ms,
        )
        self.session.chat_session.reply.assert_not_called()

    def test_tool_name_is_reported_live_and_saved(self) -> None:
        """Use the selected tool in phases, live events and saved timing."""
        intent = _intent(
            RETRIEVAL_MODE_NONE,
            tool=ToolUse("project_inventory", object()),
        )
        observed: list[dict[str, object]] = []
        phases: list[tuple[str, dict[str, object]]] = []
        with _answer_patches(self.session, intent) as mocks:
            mocks.save.side_effect = lambda **options: (
                rag.RAGSession._save_conversation(self.session, **options)
            )
            self.session.ask(
                "Inventory", use_router=False,
                activity_callback=observed.append,
                phase_callback=lambda phase, details: phases.append(
                    (phase, details)
                ),
            )
        routed = next(
            details for phase, details in phases
            if phase == rag.ANSWER_PHASE_ROUTED
        )
        self.assertEqual(routed["intent"], "test_intent")
        self.assertEqual(routed["tool_name"], "project_inventory")
        expected = (
            "Test intent identified (test_intent using project_inventory)"
        )
        self.assertEqual(observed[0]["label"], expected)
        self.assertEqual(self.session.turns[0].timeline.steps[0].label,
                         expected)
        self.assertEqual(self._stored_turn()["timeline"]["steps"][0]
                         ["label"], expected)

    def test_tool_processing_messages_are_timed_in_order(self) -> None:
        """Record detailed tool steps carried through existing callbacks."""
        intent = _intent(
            RETRIEVAL_MODE_NONE,
            tool=ToolUse("project_inventory", object()),
        )
        with _answer_patches(self.session, intent) as mocks:
            result = mocks.run.return_value

            def process(*args, **kwargs):
                """Announce two progress steps without invoking a model."""
                report = kwargs["report_phase"]
                report(rag.ANSWER_PHASE_PROCESSING, message="Scanning files")
                report(rag.ANSWER_PHASE_PROCESSING, message="Counting files")
                return result

            mocks.run.side_effect = process
            mocks.save.side_effect = lambda **options: (
                rag.RAGSession._save_conversation(self.session, **options)
            )
            self.session.ask("Inventory", use_router=False)
        steps = self.session.turns[0].timeline.steps
        self.assertEqual(
            [step.label for step in steps if step.phase == "processing"],
            ["Scanning files", "Counting files"],
        )
        self.assertEqual(
            sum(step.ms for step in steps),
            self.session.turns[0].timeline.elapsed_ms,
        )

    def test_absent_timeline_remains_a_readable_conversation_turn(self):
        """Keep untimed turns readable without migrating GUI sidecars."""
        turn = rag.ConversationTurn("now", "Q", "A", (), {})
        payload = rag._turn_to_data(turn)
        payload.pop("timeline")
        self.assertIsNone(rag._turn_from_data(payload).timeline)
        payload["timeline"] = {"elapsed_ms": 1, "steps": []}
        with self.assertRaisesRegex(ValueError, "durations do not add up"):
            rag._turn_from_data(payload)

    def test_cancelled_generation_commits_no_timing_or_turn(self) -> None:
        """Never publish a timeline for an interrupted answer."""
        self.session.chat_session.reply.side_effect = GenerationCancelled(
            "Stop requested."
        )
        with _answer_patches(self.session, _intent()):
            with self.assertRaises(GenerationCancelled):
                self.session.ask("Stop", use_router=False)
        self.assertFalse(self.session.turns)
        self.assertFalse(self.session.conversation_path.exists())

    def test_failed_checkpoint_commits_no_timing_or_turn(self) -> None:
        """Do not append a timeline when the same atomic save fails."""
        with _answer_patches(self.session, _intent()) as mocks:
            mocks.save.side_effect = OSError("checkpoint failed")
            with self.assertRaisesRegex(OSError, "checkpoint failed"):
                self.session.ask("Cannot save", use_router=False)
        self.assertFalse(self.session.turns)
        self.assertFalse(self.session.conversation_path.exists())
