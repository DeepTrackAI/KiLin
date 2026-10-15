"""Render formatted report Markdown to PDF through optional Qt.

This private GUI adapter receives report text from the core through an
explicit callback. Report parsing, escaping, styling and title extraction
belong to ``generate_paper_summary._report_format``, which has no Qt
imports. The adapter uses PySide6 only when ``write_report_pdf`` is called,
so importing the QAChat core does not require the graphical interface.

Key Features
------------

- **Optional Qt backend**

    Use ``QTextDocument`` and ``QPdfWriter`` without adding core dependencies.

- **Page fitting**

    Reduce the body font in half-point steps up to a configured minimum.

Module Structure
----------------

- ``write_report_pdf``

    Write the formatted report as an A4 PDF and return the page count.

Examples
--------
>>> from qachat.gui._report_pdf import write_report_pdf
>>> callable(write_report_pdf)
True

"""

from __future__ import annotations

from pathlib import Path

from qachat.tools.generate_paper_summary._report_format import (
    markdown_to_html,
    report_title,
)

__all__ = ["write_report_pdf"]

_APPLICATION: list[object] = []


def _application() -> None:
    """Create an offscreen Qt application when none exists.

    A full ``QApplication`` (not only a ``QGuiApplication``) is created, so
    widgets can still be made later in the same process.

    """

    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    if QGuiApplication.instance() is None and not _APPLICATION:
        _APPLICATION.append(
            QApplication(["qachat-pdf", "-platform", "offscreen"])
        )


def write_report_pdf(
    markdown: str,
    path: Path,
    *,
    max_pages: int = 1,
    font_size: float = 10.5,
    min_font_size: float = 7.5,
    label: str = "",
) -> int:
    """Write a report as an A4 PDF, shrinking the font to fit.

    Parameters
    ----------
    markdown: str
        The report.
    path: pathlib.Path
        Caller-owned PDF staging file. The renderer may replace its contents;
        the calling tool publishes it separately without overwriting files.
    max_pages: int, optional
        Pages the report should fit on. Defaults to ``1``.
    font_size: float, optional
        Starting body font size in points. Defaults to ``10.5``.
    min_font_size: float, optional
        Smallest body font size tried. Defaults to ``7.5``.
    label: str, optional
        Small label above the headline. Defaults to ``""``.

    Returns
    -------
    int
        Pages written; more than ``max_pages`` when the report does not fit
        even at ``min_font_size``.

    """

    _application()
    from PySide6.QtCore import QMarginsF, QSizeF
    from PySide6.QtGui import (
        QPageLayout,
        QPageSize,
        QPdfWriter,
        QTextDocument,
    )

    writer = QPdfWriter(str(path))
    writer.setResolution(72)  # One unit is one point, as in the layout.
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setPageMargins(
        QMarginsF(16, 14, 16, 14), QPageLayout.Unit.Millimeter
    )
    writer.setTitle(report_title(markdown))
    writer.setCreator("QAChat")
    area = writer.pageLayout().paintRectPoints()
    size = font_size
    while True:
        document = QTextDocument()
        document.setDocumentMargin(0)
        document.setPageSize(QSizeF(area.width(), area.height()))
        document.setHtml(
            markdown_to_html(markdown, font_size=size, label=label)
        )
        pages = document.pageCount()
        if pages <= max_pages or size - 0.5 < min_font_size:
            break
        size -= 0.5
    document.print_(writer)
    return pages
