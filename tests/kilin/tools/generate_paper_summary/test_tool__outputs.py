"""Test how the report tool invokes an optional PDF output service.

Markdown remains the primary report artifact. PDF output is requested only
when the caller provides a renderer, and renderer errors must not discard
the already saved Markdown report.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from kilin.tools.generate_paper_summary.tool import _write_pdf


class TestReportOutputs(unittest.TestCase):
    """Exercise the PDF callback contract without Qt or a language model."""

    def setUp(self) -> None:
        """Prepare a representative saved Markdown report.

        """

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.outputs = self.root / 'outputs'
        self.outputs.mkdir()
        self.report = self.outputs / 'report.md'
        self.report.write_text('# Report\n', encoding='utf-8')
        self.template = SimpleNamespace(title='Research briefing')
        self.writer = Mock(return_value=1)

        def render(markdown: str, path: Path, *, label: str) -> int:
            """Mimic a renderer writing to its assigned staging file.

            """

            path.write_bytes(b'%PDF-1.4\nexample\n')
            return self.writer.return_value

        self.writer.side_effect = render
        self.context = SimpleNamespace(
            project_root=self.root,
            write_pdf=self.writer,
            trace=SimpleNamespace(emit=Mock()),
            trace_turn_id='example-turn',
        )

    def test_missing_pdf_writer_leaves_markdown_unchanged(self) -> None:
        """Support core sessions that do not provide a GUI renderer.

        """

        self.context.write_pdf = None
        result = _write_pdf(
            '# Report\n', self.report, self.template, self.context
        )
        self.assertEqual(result, (None, None))
        self.assertEqual(self.report.read_text(encoding="utf-8"), '# Report\n')
        self.assertEqual(list(self.outputs.iterdir()), [self.report])
        self.writer.assert_not_called()
        self.context.trace.emit.assert_not_called()

    def test_renderer_receives_output_and_report_title(self) -> None:
        """Write a distinct PDF path and record the generated artifact.

        """

        pdf, note = _write_pdf(
            '# Report\n', self.report, self.template, self.context
        )
        self.assertIsNone(note)
        self.assertEqual(pdf, Path('outputs/report.pdf'))
        self.writer.assert_called_once()
        self.assertEqual(self.writer.call_args.args[0], '# Report\n')
        staging = self.writer.call_args.args[1]
        self.assertEqual(staging.parent.resolve(), self.outputs.resolve())
        self.assertEqual(staging.suffix, '.pdf')
        self.assertNotEqual(staging.name, 'report.pdf')
        self.assertFalse(staging.exists())
        self.assertEqual(
            self.writer.call_args.kwargs, {'label': 'Research briefing'}
        )
        self.assertEqual(
            (self.outputs / 'report.pdf').read_bytes(),
            b'%PDF-1.4\nexample\n',
        )
        self.context.trace.emit.assert_called_once_with(
            'tool.generate_paper_summary.pdf',
            component='tool',
            data={'file': 'outputs/report.pdf', 'pages': 1},
            turn_id='example-turn',
        )
        self.assertEqual(self.report.read_text(encoding="utf-8"), '# Report\n')

    def test_existing_pdf_is_not_overwritten(self) -> None:
        """Choose another output name when the PDF already exists.

        """

        old_pdf = self.outputs / 'report.pdf'
        old_pdf.write_text('existing pdf', encoding='utf-8')
        pdf, _ = _write_pdf(
            '# Report\n', self.report, self.template, self.context
        )
        self.assertIsNotNone(pdf)
        self.assertNotEqual(pdf, Path('outputs/report.pdf'))
        self.assertNotEqual(
            self.writer.call_args.args[1], old_pdf
        )
        self.assertEqual(old_pdf.read_text(encoding="utf-8"), 'existing pdf')
        self.assertEqual(
            (self.root / pdf).read_bytes(), b'%PDF-1.4\nexample\n'
        )

    def test_dangling_pdf_link_is_skipped(self) -> None:
        """The final PDF never follows an existing dangling symlink.

        """

        outside = self.root / 'outside.pdf'
        linked = self.outputs / 'report.pdf'
        try:
            linked.symlink_to(outside)
        except (NotImplementedError, OSError):
            self.skipTest('Symbolic links are unavailable')

        pdf, note = _write_pdf(
            '# Report\n', self.report, self.template, self.context
        )
        self.assertIsNone(note)
        self.assertEqual(pdf, Path('outputs/report-2.pdf'))
        self.assertFalse(outside.exists())
        self.assertTrue(linked.is_symlink())
        self.assertEqual(
            (self.root / pdf).read_bytes(), b'%PDF-1.4\nexample\n'
        )

    def test_empty_renderer_output_is_not_published(self) -> None:
        """Treat a successful callback with no PDF bytes as a failure.

        """

        self.writer.side_effect = None
        pdf, note = _write_pdf(
            '# Report\n', self.report, self.template, self.context
        )
        self.assertIsNone(pdf)
        self.assertIn('empty file', note)
        self.assertEqual(list(self.outputs.iterdir()), [self.report])
        self.context.trace.emit.assert_not_called()

    def test_overflow_is_reported_without_discarding_pdf(self) -> None:
        """Keep the PDF artifact when one-page fitting does not succeed.

        """

        self.writer.return_value = 3
        pdf, note = _write_pdf(
            '# Report\n', self.report, self.template, self.context
        )
        self.assertEqual(pdf, Path('outputs/report.pdf'))
        self.assertIn('needed 3 pages', note)
        self.assertEqual(self.report.read_text(encoding="utf-8"), '# Report\n')

    def test_renderer_error_preserves_saved_markdown(self) -> None:
        """Surface PDF failures without losing the primary artifact.

        """

        self.writer.side_effect = RuntimeError('renderer unavailable')
        pdf, note = _write_pdf(
            '# Report\n', self.report, self.template, self.context
        )
        self.assertIsNone(pdf)
        self.assertIn('renderer unavailable', note)
        self.assertTrue(self.report.is_file())
        self.context.trace.emit.assert_not_called()
        self.assertEqual(list(self.outputs.iterdir()), [self.report])


if __name__ == '__main__':
    unittest.main()
