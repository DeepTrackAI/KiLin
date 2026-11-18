"""Preserve evidence previews and core-owned timelines in the GUI.

Tests use lightweight event and turn objects, without Qt or models.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from kilin.gui.engine import (
    ChatEngine, _reference_context, _turn_payloads
)
from kilin.answer_timeline import AnswerTimeline, AnswerTimelineStep
from kilin.rag import ConversationTurn, Evidence


class TestReferencePreview(unittest.TestCase):
    """Source previews use saved evidence and show nearby lines."""

    def test_text_preview_highlights_only_the_cited_lines(self) -> None:
        """Highlight exactly the cited lines in a text preview."""
        preview = _reference_context(
            "first\nbefore\nmemory\ncapacity\nafter\n",
            "memory\ncapacity",
            3,
            4,
            False,
        )

        self.assertEqual(preview["start_line"], 3)
        self.assertEqual(preview["end_line"], 4)
        self.assertEqual(
            [
                line["number"]
                for line in preview["lines"]
                if line["highlighted"]
            ],
            [3, 4],
        )
        self.assertEqual(preview["lines"][1]["text"], "before")

    def test_pdf_preview_finds_the_matching_page(self) -> None:
        """Keep saved PDF page and line highlighting together."""
        preview = _reference_context(
            "First page\n\fSecond page\nmemory\ncapacity\nlast\n",
            "memory\ncapacity",
            2,
            3,
            True,
        )

        self.assertEqual(preview["page"], 2)
        self.assertEqual(
            [
                line["number"]
                for line in preview["lines"]
                if line["highlighted"]
            ],
            [2, 3],
        )

    def test_engine_opens_only_saved_evidence(self) -> None:
        """Reject unknown citations and fall back to saved evidence."""
        with TemporaryDirectory() as temporary:
            source = Path(temporary).resolve()
            file = source / "notes.md"
            file.write_text(
                "before\nmemory\ncapacity\nafter\n", encoding="utf-8"
            )
            evidence = Evidence(
                1,
                Path("notes.md"),
                2,
                3,
                None,
                "memory\ncapacity",
            )
            events: list[dict[str, object]] = []
            engine = ChatEngine(source, events.append)
            engine.session = SimpleNamespace(
                project=SimpleNamespace(source=source),
                turns=[SimpleNamespace(evidence=(evidence,))],
                _reads_as_code=lambda: False,
            )

            engine.preview_reference(0, 1)
            self.assertEqual(events[-1]["event"], "reference_preview")
            self.assertEqual(events[-1]["path"], "notes.md")
            engine.preview_reference(0, 2)
            self.assertEqual(events[-1]["event"], "reference_error")

            file.unlink()
            engine.preview_reference(0, 1)
            self.assertIn("saved passage", events[-1]["note"])
            self.assertTrue(
                all(line["highlighted"] for line in events[-1]["lines"])
            )


class TestAnswerTimelines(unittest.TestCase):
    """Receive recorded core timelines without GUI-side persistence."""

    def test_saved_timelines_return_with_their_turns(self) -> None:
        """Serialize the timeline already attached to each completed turn."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            timeline = AnswerTimeline(4200, (
                AnswerTimelineStep("routing", "Identifying intent", 1200),
                AnswerTimelineStep("generating", "Generating answer", 3000),
            ))
            first = ConversationTurn(
                "2026-10-07T10:00:00+00:00", "First?", "Yes.",
                (), {"usage": {"input_tokens": 1, "output_tokens": 2}},
                timeline,
            )
            second = ConversationTurn(
                "2026-10-07T10:05:00+00:00", "Second?", "No.",
                (), {}, None,
            )
            payloads = json.loads(json.dumps(_turn_payloads([
                first, second,
            ])))
            self.assertEqual(payloads[0]["timeline"], {
                "elapsed_ms": 4200,
                "steps": [
                    {"phase": "routing", "label": "Identifying intent",
                     "ms": 1200},
                    {"phase": "generating", "label": "Generating answer",
                     "ms": 3000},
                ],
            })
            self.assertEqual(payloads[0]["question"], "First?")
            self.assertEqual(payloads[0]["parameters"], first.parameters)
            self.assertIsNone(payloads[1]["timeline"])
            self.assertFalse(
                (root / "logs" / "answer_timelines.json").exists()
            )

    def test_without_timing_turns_keep_other_fields(self) -> None:
        """Preserve a valid answer lacking timing with a null timeline."""
        turn = ConversationTurn("t", "Q", "A", (), {"x": 1})
        self.assertEqual(_turn_payloads([turn]), [{
            "timestamp": "t", "question": "Q", "answer": "A",
            "evidence": (), "parameters": {"x": 1}, "timeline": None,
            "conversation_turn_uuid": turn.conversation_turn_uuid,
        }])
