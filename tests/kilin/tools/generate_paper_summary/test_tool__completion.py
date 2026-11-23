"""Reject incomplete or invalid reports before creating output artifacts.

Use one deterministic source and a minimal template to exercise the real
report publication boundary without loading models or reading PDFs.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kilin._model_completion import IncompleteModelResponse
from kilin.tools import ToolRequest
from kilin.tools.generate_paper_summary import tool
from kilin.tools.generate_paper_summary._report_template import (
    ReportTemplate,
    TemplateSection,
)
from tests.kilin.tools._shared._support import chunk


class TestReportCompletion(unittest.TestCase):
    """Publish only reports which have usable notes and pass validation."""

    def setUp(self) -> None:
        """Build a minimal report context with no inference dependency."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.context = SimpleNamespace(
            project_root=self.root, progress=Mock(), done=Mock(),
            trace=SimpleNamespace(emit=Mock()), trace_turn_id="t",
            model_id="fake", write_pdf=None,
        )
        self.template = ReportTemplate(
            "test", "Test Report", "", (TemplateSection(
                "Summary", words=(3, 30),
            ),),
        )
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(
            tool, "choose_template", return_value=self.template
        ))
        self.stack.enter_context(patch.object(
            tool, "choose_paper", return_value=(
                Path("paper.md"), [chunk("1", "Supported observation.")],
                "named", False,
            ),
        ))
        self.stack.enter_context(patch.object(
            tool, "confirm_plan", return_value=("continue", False)
        ))
        self.notes = self.stack.enter_context(patch.object(
            tool, "take_notes", return_value=[(1, ["Observation [1]"])]
        ))
        self.stack.enter_context(patch.object(
            tool, "reference_entries", return_value=[]
        ))
        self.section = self.stack.enter_context(patch.object(
            tool, "_write_section",
            return_value=("A supported report contains useful results.", []),
        ))
        self.save = self.stack.enter_context(patch.object(
            tool, "save_output_file", return_value=Path("outputs/test.md")
        ))
        self.pdf = self.stack.enter_context(patch.object(
            tool, "_write_pdf", return_value=(None, "")
        ))

    def _run(self):
        """Run the real publication path with the prepared template."""

        return tool.run(
            tool.SummarySettings(), ToolRequest("Summary", "Summary"),
            self.context,
        )

    def test_invalid_report_never_publishes_files(self) -> None:
        """Do not save invalid Markdown or invoke the PDF writer."""

        self.section.return_value = ("", ["Section remained empty"])
        result = self._run()
        self.assertEqual(result.summary["status"], "incomplete")
        self.assertFalse(result.summary["valid"])
        self.assertTrue(result.summary["problems"])
        self.assertIsNone(result.summary["saved"])
        self.assertIsNone(result.summary["saved_pdf"])
        self.assertEqual(result.evidence, ())
        self.assertIn("No report file was saved", result.answer)
        self.assertNotIn("Done:", result.answer)
        self.save.assert_not_called()
        self.pdf.assert_not_called()

    def test_valid_report_keeps_existing_publication_behavior(self) -> None:
        """Preserve successful report writing and source evidence."""

        result = self._run()
        self.assertTrue(result.summary["valid"])
        self.assertEqual(result.summary["saved"], "outputs/test.md")
        self.assertIn("Done:", result.answer)
        self.assertEqual(len(result.evidence), 1)
        self.save.assert_called_once()
        self.pdf.assert_called_once()

    def test_no_usable_notes_stops_before_section_writing(self) -> None:
        """Do not synthesize a scientific report from an empty note set."""

        self.notes.return_value = [(1, [])]
        with self.assertRaises(IncompleteModelResponse) as failure:
            self._run()
        self.assertEqual(failure.exception.reason, "no_notes")
        self.section.assert_not_called()
        self.save.assert_not_called()
        self.pdf.assert_not_called()

    def test_incomplete_reading_never_reaches_publication(self) -> None:
        """Stop on the typed failure rather than consume partial notes."""

        self.notes.side_effect = IncompleteModelResponse(
            "Truncated notes", reason="token_limit", finish_reason="length"
        )
        with self.assertRaises(IncompleteModelResponse):
            self._run()
        self.section.assert_not_called()
        self.save.assert_not_called()
        self.pdf.assert_not_called()


if __name__ == "__main__":
    unittest.main()
