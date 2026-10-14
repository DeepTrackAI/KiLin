"""Test pure report formatting independently of optional PDF backends.

Verify safe HTML generation, supported Markdown syntax, report styling,
metadata titles and the absence of a Qt dependency at import time.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from qachat.tools.generate_paper_summary import _report_format


class TestReportFormat(unittest.TestCase):
    """Verify generated HTML without invoking the Qt PDF writer."""

    def test_escapes_source_markup_and_label(self) -> None:
        """Do not treat report content as arbitrary HTML.

        """

        report = '# <script>\n\nA & B <img src="x">.'
        html = _report_format.markdown_to_html(
            report, label='<em title="x">Briefing</em>'
        )
        self.assertIn('&lt;script&gt;', html)
        self.assertIn('A &amp; B &lt;img src="x"&gt;.', html)
        self.assertIn('&lt;em title=&quot;x&quot;&gt;', html)
        self.assertNotIn('<script>', html)
        self.assertNotIn('<img ', html)
        self.assertNotIn('<em ', html)

    def test_headings_paragraphs_and_inline_emphasis(self) -> None:
        """Preserve the supported report Markdown constructs.

        """

        markdown = (
            '# A report\n\n## Findings\nFirst line\nsecond line.\n\n'
            '**Important** *detail* and `code`.\n---\n'
        )
        html = _report_format.markdown_to_html(markdown)
        self.assertIn('<h1>A report</h1>', html)
        self.assertIn('<p class="section">Findings</p>', html)
        self.assertIn('<p>First line second line.</p>', html)
        self.assertIn('<b>Important</b>', html)
        self.assertIn('<i>detail</i>', html)
        self.assertIn('<code>code</code>', html)
        self.assertNotIn('<hr', html)

    def test_unordered_and_numbered_lists(self) -> None:
        """Render supported list markers and keep items escaped.

        """

        markdown = (
            '- First\n* Second\n• Third & fourth\n\n'
            '1. Alpha\n2) Beta < gamma\n'
        )
        html = _report_format.markdown_to_html(markdown)
        self.assertIn(
            '<ul><li>First</li><li>Second</li>'
            '<li>Third &amp; fourth</li></ul>',
            html,
        )
        self.assertIn(
            '<ol><li>Alpha</li><li>Beta &lt; gamma</li></ol>',
            html,
        )

    def test_note_paragraph_and_font_scaling(self) -> None:
        """Apply note and body styles with caller-selected sizing.

        """

        html = _report_format.markdown_to_html(
            '*Source note*', font_size=8.0, label='Experiment'
        )
        self.assertIn('<p class="note"><i>Source note</i></p>', html)
        self.assertIn('font-size: 8.0px', html)
        self.assertIn('font-size: 16.0px', html)
        self.assertIn('<p class="label">Experiment</p>', html)

    def test_empty_report_is_still_an_html_document(self) -> None:
        """Produce a usable HTML shell even without report text.

        """

        html = _report_format.markdown_to_html('')
        self.assertTrue(html.startswith('<html><head><style>'))
        self.assertTrue(html.endswith('</body></html>'))
        self.assertNotIn('class="label"', html)

    def test_title_uses_first_level_one_heading(self) -> None:
        """Match prior PDF metadata title selection.

        """

        self.assertEqual(
            _report_format.report_title('## Subtitle\n# Real title\n# Later'),
            'Real title',
        )
        self.assertEqual(
            _report_format.report_title('## Subtitle\nText'),
            'QAChat report',
        )
        self.assertEqual(
            _report_format.report_title('prefix # Not a heading'),
            'QAChat report',
        )

    def test_formatter_import_does_not_require_qt(self) -> None:
        """Import the standalone implementation with Qt unavailable.

        """

        file = Path(_report_format.__file__)
        spec = importlib.util.spec_from_file_location(
            '_isolated_report_format', file
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'PySide6': None}):
            spec.loader.exec_module(module)
        self.assertIn('<h1>Independent</h1>',
                      module.markdown_to_html('# Independent'))


if __name__ == '__main__':
    unittest.main()
