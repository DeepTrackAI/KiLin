"""Keep independently completed report sections after model failures.

Exercise real Markdown publication and existing validation, without a model,
PDF renderer or network. A partial report never replaces a completed report.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from kilin._model_completion import IncompleteModelResponse
from kilin.tools._shared.document_notes import take_notes
from kilin.tools._shared.outputs import save_output_file
from kilin.tools.generate_paper_summary import tool
from kilin.tools.generate_paper_summary._report_template import (
    ReportTemplate,
    TemplateSection,
)
from tests.kilin.tools._shared._support import chunk
from tests.kilin.tools.generate_paper_summary import test_tool__completion


def _failure():
    """Simulate an exhausted call after the adapter's one recovery attempt."""

    return IncompleteModelResponse(
        "Output-token limit exhausted.", reason="token_limit",
        finish_reason="length", raw_text="TRUNCATED UNSAFE TEXT",
    )


class TestReportRecovery(unittest.TestCase):
    """Separate safe partial artifacts from successful complete reports."""

    def setUp(self):
        """Reuse the strict-publication fixture without duplicating tests."""

        self.section_writer = tool._write_section
        test_tool__completion.TestReportCompletion.setUp(self)
        self.save.side_effect = save_output_file
        self.template = ReportTemplate(
            "test", "Test Report", "", tuple(
                TemplateSection(name, words=(4, 40))
                for name in ("First", "Second", "Third")
            ),
        )
        self.stack.enter_context(patch.object(
            tool, "choose_template", return_value=self.template
        ))

    def _run(self):
        """Invoke the actual report runner with the prepared fixture."""

        return test_tool__completion.TestReportCompletion._run(self)

    def test_middle_section_failure_keeps_later_sections(self):
        self.section.side_effect = [
            ("First completed section has useful information.", []),
            _failure(),
            ("Third completed section has other information.", []),
        ]
        output = self.root / "outputs"
        output.mkdir()
        complete = output / "paper_summary.md"
        complete.write_text(
            "Keep the earlier complete report.", encoding="utf-8",
        )
        pdf = output / "paper_summary.pdf"
        pdf.write_bytes(b"Existing PDF")
        result = self._run()
        self.assertEqual(self.section.call_count, 3)
        self.assertEqual(result.summary["status"], "partial")
        self.assertFalse(result.summary["valid"])
        self.assertIn("_summary_partial.md", result.summary["saved"])
        text = (self.root / result.summary["saved"]).read_text(
            encoding="utf-8",
        )
        self.assertIn("First completed", text)
        self.assertIn("Third completed", text)
        self.assertIn("Not completed:", text)
        self.assertNotIn("TRUNCATED UNSAFE TEXT", text)
        self.assertNotIn("Done:", result.answer)
        self.assertEqual(
            complete.read_text(
                encoding="utf-8",
            ), "Keep the earlier complete report."
        )
        self.assertEqual(pdf.read_bytes(), b"Existing PDF")
        self.pdf.assert_not_called()
        earlier = self.section.call_args_list[2].args[4]
        self.assertIn("First", [name for name, _ in earlier])
        self.assertNotIn("Second", [name for name, _ in earlier])

    def test_all_sections_failed_does_not_publish_an_empty_report(self):
        self.section.side_effect = _failure()
        result = self._run()
        self.assertEqual(self.section.call_count, 3)
        self.assertEqual(result.summary["status"], "incomplete")
        self.assertEqual(len(result.summary["incomplete_items"]), 3)
        self.assertIsNone(result.summary["saved"])
        self.save.assert_not_called()
        self.pdf.assert_not_called()

    def test_invalid_section_is_not_rescued_by_another_failure(self):
        self.section.side_effect = [
            ("Invalid", []), _failure(),
            ("Third completed section has enough words.", []),
        ]
        result = self._run()
        self.assertEqual(result.summary["status"], "partial")
        failures = result.summary["incomplete_items"]
        self.assertEqual({f["item"] for f in failures}, {"First", "Second"})
        text = (self.root / result.summary["saved"]).read_text(
            encoding="utf-8",
        )
        self.assertNotIn("\nInvalid\n", text)
        self.assertIn("Third completed", text)
        self.assertIn("Not completed:", text)

    def test_failed_reading_omits_notes_and_continues(self):
        passages = [chunk("1", "Failed input."),
                    chunk("2", "Verified later observation.", line=10)]
        self.stack.enter_context(patch.object(
            tool, "choose_paper", return_value=(
            Path("paper.md"), passages, "named", False,
        )))
        self.stack.enter_context(patch.object(
            tool, "plan_parts", return_value=[[0], [1]]
        ))
        self.notes.side_effect = take_notes
        self.context.ask_model = Mock(side_effect=[
            _failure(), "- Verified later observation [2]",
        ])
        self.context.trace.include_content = False
        result = self._run()
        self.assertEqual(result.summary["status"], "partial")
        self.assertEqual(result.summary["passages_read"], 1)
        self.assertEqual(self.context.ask_model.call_count, 2)
        for call in self.section.call_args_list:
            self.assertTrue(call.kwargs["incomplete_reading"])
            self.assertEqual(call.args[3],
                             [(2, ["- Verified later observation [2]"])])
        self.pdf.assert_not_called()

    def test_all_reading_failed_does_not_synthesize_sections(self):
        self.notes.side_effect = take_notes
        self.context.ask_model = Mock(side_effect=_failure())
        result = self._run()
        self.assertEqual(result.summary["status"], "incomplete")
        self.section.assert_not_called()
        self.save.assert_not_called()
        self.assertEqual(result.evidence, ())

    def test_failed_repair_does_not_stop_next_section(self):
        self.section.side_effect = self.section_writer
        self.context.ask_model = Mock(side_effect=[
            "Short", _failure(),
            "Second section supplies sufficient completed content [1].",
            "Third section supplies further completed content [1].",
        ])
        result = self._run()
        self.assertEqual(self.context.ask_model.call_count, 4)
        self.assertEqual(result.summary["status"], "partial")
        text = (self.root / result.summary["saved"]).read_text(
            encoding="utf-8",
        )
        self.assertIn("Second section", text)
        self.assertIn("Third section", text)
        self.assertNotIn("\nShort\n", text)
        self.assertNotIn("TRUNCATED", text)

    def test_headline_failure_does_not_stop_sections(self):
        with patch.object(tool, "_write_headline", side_effect=_failure()):
            result = self._run()
        self.assertEqual(self.section.call_count, 3)
        self.assertEqual(result.summary["status"], "partial")
        self.assertEqual(result.summary["incomplete_items"][0]["item"],
                         "Headline")
        self.assertTrue((self.root / result.summary["saved"]).is_file())


if __name__ == "__main__":
    unittest.main()
