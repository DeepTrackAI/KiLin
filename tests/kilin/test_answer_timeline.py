"""Test core-owned answer timeline recording and stored-data validation.

Use a controlled monotonic clock; no RAG project or model is required.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

import kilin
from kilin.answer_timeline import (
    AnswerTimeline,
    AnswerTimelineStep,
    _AnswerTimelineRecorder,
    _timeline_from_data,
    _timeline_to_data,
)


class TestAnswerTimelineRecorder(unittest.TestCase):
    """Verify stable transitions and non-overlapping monotonic durations."""

    def test_phase_durations_are_measured_once_and_add_up(self) -> None:
        """Merge repeated updates while retaining the saving event."""
        now = [10.0]
        recorder = _AnswerTimelineRecorder(clock=lambda: now[0])
        now[0] = 10.125
        recorder.record("routing", {"message": "Identifying intent"})
        now[0] = 10.3
        recorder.record("routing", {"message": "Identifying intent"})
        now[0] = 11.0
        recorder.record("retrieving", {"message": "Semantic search"})
        now[0] = 11.25
        recorder.record("retrieving", {"message": "Lexical search"})
        now[0] = 11.7
        recorder.record("saving", {"message": "Saving conversation"})
        now[0] = 12.0
        timeline = recorder.finish()
        self.assertEqual(timeline.elapsed_ms, 2000)
        self.assertEqual(
            [(step.phase, step.label, step.ms)
             for step in timeline.steps],
            [
                ("routing", "Identifying intent", 1000),
                ("retrieving", "Semantic search", 250),
                ("retrieving", "Lexical search", 450),
                ("saving", "Saving conversation", 300),
            ],
        )
        now[0] = 100.0
        recorder.record("completed", {"message": "Answer ready"})
        self.assertIs(recorder.finish(), timeline)

    def test_routed_phase_emits_intent_and_history_activities(self) -> None:
        """Use the same detailed semantics shown in the original live GUI."""
        now = [10.0]
        recorder = _AnswerTimelineRecorder(clock=lambda: now[0])
        routed = recorder.record("routed", {
            "intent": "reference_check",
            "tool_name": "reference_check",
            "history_turn_ids": [],
            "related_turn_ids": [],
        })
        self.assertEqual([item["label"] for item in routed], [
            "Reference check identified "
            "(reference_check using reference_check)",
            "No earlier conversation needed",
        ])
        self.assertEqual([item["index"] for item in routed], [0, 1])
        self.assertEqual([item["started_ms"] for item in routed], [0, 0])
        now[0] += 1.0
        skipped = recorder.record("evidence_ready", {
            "message": "No retrieval needed", "skipped": True,
        })
        self.assertEqual(skipped[0]["label"], "No retrieval needed")
        now[0] += 0.5
        steps = recorder.finish().steps
        self.assertEqual([item.label for item in steps], [
            *(event["label"] for event in routed),
            skipped[0]["label"],
        ])

    def test_memory_retrieval_activity_has_its_own_timed_span(self) -> None:
        """Measure memory search separately from sources and generation."""
        now = [5.0]
        recorder = _AnswerTimelineRecorder(clock=lambda: now[0])
        recorder.record("memory_retrieving", {
            "message": "Retrieving conversation memories",
        })
        now[0] += 0.225
        recorder.record("memory_ready", {
            "message": "3 conversation memories retrieved from 2 turns",
        })
        now[0] += 0.050
        recorder.record("generating", {"message": "Generating answer"})
        now[0] += 0.400
        steps = recorder.finish().steps
        self.assertEqual(
            [(step.phase, step.label, step.ms) for step in steps],
            [
                ("memory_retrieving", "Retrieving conversation memories",
                 225),
                ("memory_ready",
                 "3 conversation memories retrieved from 2 turns", 50),
                ("generating", "Generating answer", 400),
            ],
        )

    def test_reference_check_reports_every_activity_in_order(self) -> None:
        """Keep the detailed tool progression with no GUI skip policy."""
        now = [1.0]
        recorder = _AnswerTimelineRecorder(clock=lambda: now[0])
        emitted: list[dict[str, object]] = []
        sequence = (
            ("routing", {"message": "Identifying intent"}),
            ("routed", {"intent": "reference_check",
                        "tool_name": "reference_check",
                        "history_turn_ids": [], "related_turn_ids": []}),
            ("evidence_ready", {"message": "No retrieval needed",
                                "skipped": True}),
            ("processing", {"message": "Finding the paper"}),
            ("processing", {"message":
                            "Identifying the project's files"}),
            ("processing", {"message": "Reading references and citations"}),
        )
        for phase, details in sequence:
            emitted.extend(recorder.record(phase, details))
            now[0] += 0.25
        saved = recorder.finish()
        self.assertEqual([event["label"] for event in emitted], [
            "Identifying intent", "Reference check identified "
            "(reference_check using reference_check)",
            "No earlier conversation needed", "No retrieval needed",
            "Finding the paper", "Identifying the project's files",
            "Reading references and citations",
        ])
        self.assertEqual(
            [(event["phase"], event["label"]) for event in emitted],
            [(step.phase, step.label) for step in saved.steps],
        )
        self.assertEqual(
            [event["index"] for event in emitted],
            list(range(len(saved.steps))),
        )

    def test_history_selection_has_one_core_description(self) -> None:
        """Describe recent and related turns without browser inference."""
        recorder = _AnswerTimelineRecorder(clock=lambda: 1.0)
        records = recorder.record("routed", {
            "intent": "question_answering",
            "history_turn_ids": [2, 3, 5],
            "related_turn_ids": [2, 5],
        })
        self.assertEqual(
            records[0]["label"],
            "Question answering identified (question_answering)",
        )
        self.assertEqual(records[1]["label"], (
            "Including the last 1 turn and turns 2, 5 "
            "from the conversation"
        ))

    def test_routed_phase_handles_absent_optional_names(self) -> None:
        """Retain the fallback label without inventing intent or tool IDs."""
        for details, expected in (
            ({}, "Intent identified"),
            ({"intent": "test_intent", "tool_name": None},
             "Test intent identified (test_intent)"),
            ({"intent": "test_intent", "tool_name": ""},
             "Test intent identified (test_intent)"),
        ):
            with self.subTest(details=details):
                recorder = _AnswerTimelineRecorder(clock=lambda: 1.0)
                routed = recorder.record("routed", details)
                self.assertEqual(routed[0]["label"], expected)

    def test_no_phase_notifications_still_produces_timing(self) -> None:
        """Measure work when routing and retrieval produce no callbacks."""
        now = [1.0]
        recorder = _AnswerTimelineRecorder(clock=lambda: now[0])
        now[0] = 1.001
        timeline = recorder.finish()
        self.assertEqual(timeline.elapsed_ms, 1)
        self.assertEqual(timeline.steps[0].phase, "preparing")
        self.assertEqual(timeline.steps[0].ms, 1)

    def test_rounding_does_not_create_negative_durations(self) -> None:
        """Clamp rounded boundaries even for a nonmonotonic test clock."""
        now = [1.0]
        recorder = _AnswerTimelineRecorder(clock=lambda: now[0])
        now[0] = 2.0
        recorder.record("routing", {})
        now[0] = 1.5
        recorder.record("retrieving", {})
        now[0] = 2.5
        timeline = recorder.finish()
        self.assertEqual(
            sum(step.ms for step in timeline.steps), timeline.elapsed_ms
        )
        self.assertTrue(all(step.ms >= 0 for step in timeline.steps))


class TestAnswerTimelinePersistence(unittest.TestCase):
    """Keep serialized records compact, validated and GUI-readable."""

    def test_public_api_and_json_roundtrip(self) -> None:
        """Expose immutable records and store steps as JSON arrays."""
        self.assertIs(kilin.AnswerTimeline, AnswerTimeline)
        self.assertIs(kilin.AnswerTimelineStep, AnswerTimelineStep)
        timeline = AnswerTimeline(4, (
            AnswerTimelineStep("retrieving", "Search", 3),
            AnswerTimelineStep("generating", "Answer", 1),
        ))
        data = _timeline_to_data(timeline)
        self.assertEqual(data, {
            "elapsed_ms": 4,
            "steps": [
                {"phase": "retrieving", "label": "Search", "ms": 3},
                {"phase": "generating", "label": "Answer", "ms": 1},
            ],
        })
        self.assertEqual(_timeline_from_data(data), timeline)
        self.assertIsNone(_timeline_to_data(None))
        self.assertIsNone(_timeline_from_data(None))
        with self.assertRaises(FrozenInstanceError):
            timeline.elapsed_ms = 40
        with self.assertRaises(FrozenInstanceError):
            timeline.steps[0].ms = 9

    def test_bad_timelines_are_rejected_not_silently_reinterpreted(
        self,
    ) -> None:
        """Reject negative, boolean, mismatched, or unstructured timing."""
        valid = {
            "elapsed_ms": 1,
            "steps": [{"phase": "routing", "label": "Intent", "ms": 1}],
        }
        invalid = [
            [],
            {"elapsed_ms": True, "steps": []},
            {"elapsed_ms": -1, "steps": []},
            {"elapsed_ms": 0, "steps": "bad"},
            {"elapsed_ms": 1, "steps": [3]},
            {"elapsed_ms": 1, "steps": [{"label": "x", "ms": 1}]},
            {"elapsed_ms": 1, "steps": [{"phase": "x", "ms": 1}]},
            {"elapsed_ms": 1, "steps": [
                {"phase": "x", "label": "x", "ms": True}
            ]},
            {"elapsed_ms": 1, "steps": [
                {"phase": "x", "label": "x", "ms": -1}
            ]},
            {**valid, "elapsed_ms": 2},
        ]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _timeline_from_data(value)
