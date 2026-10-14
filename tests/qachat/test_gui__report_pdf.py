"""Test the private Qt PDF adapter without loading an actual GUI.

Fake Qt page-layout objects exercise renderer options, metadata and fitting
when PySide6 is absent; a separate optional integration test checks a real
PDF when the GUI dependency is installed.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qachat.gui import _report_pdf


class _FakePageSize:
    """Represent a Qt page-size object for deterministic adapter tests."""

    PageSizeId = SimpleNamespace(A4='A4')

    def __init__(self, value: str) -> None:
        """Remember which paper format was requested.

        Parameters
        ----------
        value: str
            Requested page-size identifier.

        """

        self.value = value


class _FakePdfWriter:
    """Record Qt PDF settings and printed document state for assertions."""

    instances: list[_FakePdfWriter] = []

    def __init__(self, filename: str) -> None:
        """Track the PDF destination and creation.

        Parameters
        ----------
        filename: str
            Destination of the simulated PDF.

        """

        self.filename = filename
        self.properties: dict[str, object] = {}
        self.printed_document: object | None = None
        self.instances.append(self)

    def setResolution(self, value: int) -> None:
        """Record the print resolution.

        Parameters
        ----------
        value: int
            Points per inch.

        """

        self.properties['resolution'] = value

    def setPageSize(self, value: object) -> None:
        """Record the requested page size.

        Parameters
        ----------
        value: object
            Fake Qt A4 page-size object.

        """

        self.properties['page_size'] = value

    def setPageMargins(self, margins: object, unit: object) -> None:
        """Record printable margins.

        Parameters
        ----------
        margins: object
            Margin object.
        unit: object
            Unit of measure.

        """

        self.properties['margins'] = (margins, unit)

    def setTitle(self, title: str) -> None:
        """Record the title embedded in PDF metadata.

        Parameters
        ----------
        title: str
            Report title.

        """

        self.properties['title'] = title

    def setCreator(self, creator: str) -> None:
        """Record the originating application.

        Parameters
        ----------
        creator: str
            PDF creator name.

        """

        self.properties['creator'] = creator

    def pageLayout(self) -> object:
        """Return a fake printable rectangle.

        Returns
        -------
        object
            Object providing a printable rectangle in points.

        """

        area = SimpleNamespace(width=lambda: 550, height=lambda: 780)
        return SimpleNamespace(paintRectPoints=lambda: area)


class _FakeTextDocument:
    """Track font sizes and provide a predictable synthetic page count."""

    instances: list[_FakeTextDocument] = []

    def __init__(self) -> None:
        """Keep each attempted page-fitting document for inspection."""

        self.html = ''
        self.instances.append(self)

    def setDocumentMargin(self, margin: int) -> None:
        """Check that the renderer suppresses extra margins.

        Parameters
        ----------
        margin: int
            Document margin value.

        """

        if margin != 0:
            raise AssertionError('Expected zero document margin')

    def setPageSize(self, page: object) -> None:
        """Accept the fake printable page geometry.

        Parameters
        ----------
        page: object
            Renderable area.

        """

        self.page = page

    def setHtml(self, html: str) -> None:
        """Capture the generated report HTML.

        Parameters
        ----------
        html: str
            Render-ready HTML string.

        """

        self.html = html

    def pageCount(self) -> int:
        """Require smaller fonts to fit a report on one page.

        Returns
        -------
        int
            Two pages above 9 pt, otherwise one page.

        """

        size = re.search(r'font-size: (\d+\.\d+)px', self.html)
        return 2 if float(size.group(1)) > 9.0 else 1

    def print_(self, writer: _FakePdfWriter) -> None:
        """Record the final document handed to Qt.

        Parameters
        ----------
        writer: _FakePdfWriter
            Simulated PDF destination.

        """

        writer.printed_document = self


class TestReportPdfAdapter(unittest.TestCase):
    """Verify the GUI adapter is optional and uses the shared formatter."""

    def setUp(self) -> None:
        """Prepare fake Qt classes without importing PySide6.

        """

        _FakePdfWriter.instances.clear()
        _FakeTextDocument.instances.clear()
        qtcore = ModuleType('PySide6.QtCore')
        qtgui = ModuleType('PySide6.QtGui')
        qtcore.QMarginsF = lambda *args: args
        qtcore.QSizeF = lambda *args: args
        qtgui.QPageLayout = SimpleNamespace(
            Unit=SimpleNamespace(Millimeter='mm')
        )
        qtgui.QPageSize = _FakePageSize
        qtgui.QPdfWriter = _FakePdfWriter
        qtgui.QTextDocument = _FakeTextDocument
        self.modules = {
            'PySide6': ModuleType('PySide6'),
            'PySide6.QtCore': qtcore,
            'PySide6.QtGui': qtgui,
        }

    def test_pdf_writer_uses_a4_and_fits_the_page(self) -> None:
        """Shrink the font and print exactly the last attempted document.

        """

        destination = Path('/tmp/report.pdf')
        with patch.dict(sys.modules, self.modules), patch.object(
            _report_pdf, '_application'
        ) as application:
            pages = _report_pdf.write_report_pdf(
                '# My report\n\n## Facts\nData',
                destination,
                font_size=10.5,
                min_font_size=7.5,
                label='Research briefing',
            )
        application.assert_called_once()
        self.assertEqual(pages, 1)
        self.assertEqual(len(_FakeTextDocument.instances), 4)
        writer = _FakePdfWriter.instances[0]
        self.assertEqual(writer.filename, str(destination))
        self.assertEqual(writer.properties['resolution'], 72)
        self.assertEqual(writer.properties['page_size'].value, 'A4')
        self.assertEqual(writer.properties['title'], 'My report')
        self.assertEqual(writer.properties['creator'], 'QAChat')
        self.assertEqual(writer.properties['margins'][1], 'mm')
        self.assertIs(writer.printed_document,
                      _FakeTextDocument.instances[-1])
        self.assertIn('Research briefing', writer.printed_document.html)
        self.assertIn('font-size: 9.0px', writer.printed_document.html)

    def test_reports_excess_pages_at_the_minimum_font(self) -> None:
        """Report overflow instead of claiming that an unfit page fits.

        """

        with patch.dict(sys.modules, self.modules), patch.object(
            _report_pdf, '_application'
        ):
            pages = _report_pdf.write_report_pdf(
                'Text', Path('/tmp/overflow.pdf'),
                font_size=10.5, min_font_size=10.0,
            )
        self.assertEqual(pages, 2)
        self.assertEqual(len(_FakeTextDocument.instances), 2)
        self.assertEqual(
            _FakePdfWriter.instances[0].properties['title'],
            'QAChat report',
        )

    def test_import_has_no_qt_startup_side_effect(self) -> None:
        """Import the renderer module even if PySide6 is unavailable.

        """

        spec = importlib.util.spec_from_file_location(
            '_isolated_gui_report_pdf', Path(_report_pdf.__file__)
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'PySide6': None}):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.write_report_pdf))

    def test_offscreen_application_is_created_only_when_needed(self) -> None:
        """Avoid repeated QApplication creation or GUI window startup.

        """

        application = Mock(return_value=object())
        qtwidgets = ModuleType('PySide6.QtWidgets')
        qtwidgets.QApplication = application
        gui_application = SimpleNamespace(instance=Mock(return_value=None))
        self.modules['PySide6.QtWidgets'] = qtwidgets
        self.modules['PySide6.QtGui'].QGuiApplication = gui_application
        with patch.dict(sys.modules, self.modules), patch.object(
            _report_pdf, '_APPLICATION', []
        ) as holders:
            _report_pdf._application()
            _report_pdf._application()
            self.assertEqual(len(holders), 1)
        application.assert_called_once_with(
            ['qachat-pdf', '-platform', 'offscreen']
        )

    def test_existing_application_is_reused(self) -> None:
        """Never create a second QApplication when one already exists.

        """

        application = Mock()
        qtwidgets = ModuleType('PySide6.QtWidgets')
        qtwidgets.QApplication = application
        gui_application = SimpleNamespace(
            instance=Mock(return_value=object())
        )
        self.modules['PySide6.QtWidgets'] = qtwidgets
        self.modules['PySide6.QtGui'].QGuiApplication = gui_application
        with patch.dict(sys.modules, self.modules), patch.object(
            _report_pdf, '_APPLICATION', []
        ) as holders:
            _report_pdf._application()
            self.assertEqual(holders, [])
        application.assert_not_called()


@unittest.skipUnless(
    importlib.util.find_spec('PySide6') is not None,
    'PySide6 is not installed',
)
class TestReportPdfQtIntegration(unittest.TestCase):
    """Exercise a real PDF export when the optional GUI library exists."""

    def test_writes_readable_pdf_with_qt(self) -> None:
        """Write an A4 report PDF and verify its metadata and page count.

        """

        from pypdf import PdfReader

        with TemporaryDirectory() as temporary:
            file = Path(temporary) / 'report.pdf'
            pages = _report_pdf.write_report_pdf(
                '# Real PDF\n\n## Section\nSome text.',
                file,
                label='Technical summary',
            )
            self.assertTrue(file.is_file())
            self.assertGreater(file.stat().st_size, 0)
            self.assertEqual(len(PdfReader(file).pages), pages)
            self.assertEqual(PdfReader(file).metadata.title, 'Real PDF')


if __name__ == '__main__':
    unittest.main()
