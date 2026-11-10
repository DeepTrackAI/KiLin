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
        """Ignore duplicate notifications and checkpoint-only phases."""
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
                ("preparing", "Preparing answer", 125),
                ("routing", "Identifying intent", 875),
                ("retrieving", "Semantic search", 250),
                ("retrieving", "Lexical search", 750),
            ],
        )
        now[0] = 100.0
        recorder.record("completed", {"message": "Answer ready"})
        self.assertIs(recorder.finish(), timeline)

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
