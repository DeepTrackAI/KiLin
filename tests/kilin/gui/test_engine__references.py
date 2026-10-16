"""Preserve saved evidence previews and answer timelines in the GUI.

Tests use lightweight event and turn objects, without Qt or models.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kilin.gui.engine import (
    ChatEngine, _reference_context, _turn_payloads
)
from kilin.rag import Evidence


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
            file.write_text("before\nmemory\ncapacity\nafter\n")
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
    """A resumed chat shows how long each answer took, as it did live."""

    def test_saved_timelines_return_with_their_turns(self) -> None:
        """Restore valid timing data from the persisted turn timeline."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            engine = ChatEngine(root, lambda event: None)
            first = SimpleNamespace(timestamp="2026-10-07T10:00:00+00:00")
            second = SimpleNamespace(timestamp="2026-10-07T10:05:00+00:00")
            engine.session = SimpleNamespace(
                project=SimpleNamespace(root=root), turns=[first, second]
            )
            timeline = {
                "elapsed_ms": 4200,
                "steps": [{"label": "Identifying intent", "ms": 1200}],
            }
            engine.save_answer_timeline(first.timestamp, timeline)
            # A malformed timeline from the page is ignored.
            engine.save_answer_timeline(second.timestamp, {"steps": "x"})

            with patch(
                "kilin.gui.engine.asdict",
                side_effect=lambda turn: {"timestamp": turn.timestamp},
            ):
                payloads = _turn_payloads([first, second], root)

            self.assertEqual(payloads[0]["timeline"], timeline)
            self.assertNotIn("timeline", payloads[1])
            self.assertTrue((root / "logs" / "answer_timelines.json").exists())

    def test_without_saved_timelines_turns_are_unchanged(self) -> None:
        """Leave turn payloads unchanged when no timeline exists."""
        with TemporaryDirectory() as folder:
            engine = ChatEngine(Path(folder), lambda event: None)
            engine.session = SimpleNamespace(
                project=SimpleNamespace(root=Path(folder)), turns=[]
            )
            turn = SimpleNamespace(timestamp="t")
            with patch(
                "kilin.gui.engine.asdict",
                side_effect=lambda turn: {"timestamp": turn.timestamp},
            ):
                self.assertEqual(
                    _turn_payloads([turn], Path(folder)),
                    [{"timestamp": "t"}],
                )
