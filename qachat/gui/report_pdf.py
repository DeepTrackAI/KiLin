"""Write generated reports as one-page PDFs with Qt.

Reports such as research briefings are saved as Markdown by the core tools,
which work without a GUI. The desktop app also has Qt, which can lay out
styled text and write PDF without further dependencies, so the engine gives
tools ``write_report_pdf`` and every report also gets a PDF next to its
Markdown.

The Markdown that tools write is simple (a ``#`` headline, ``##`` sections,
paragraphs, lists, bold and italic), so it is converted to HTML here, styled
for print, and laid out on A4. When the text does not fit the page limit,
the font size is reduced in small steps down to a readable minimum; the
result reports how many pages were needed.

Key Features
------------

- **No new dependencies**: Qt's ``QTextDocument`` and ``QPdfWriter``.
- **Fits one page**: the font shrinks until the report fits, within limits.
- **Works without a window**: an offscreen Qt application is created when
  none exists (the engine process has no window).

Module Structure
----------------

- ``markdown_to_html``
- ``write_report_pdf``

"""

from __future__ import annotations

import html
from pathlib import Path
import re
from typing import Final

__all__ = [
    "markdown_to_html",
    "write_report_pdf",
]

_ACCENT: Final[str] = "#334e43"
_MUTED: Final[str] = "#5f6b66"
_ITEM = re.compile(r"^\s*(?:[-*•]|(\d+)[.)])\s+(.*)$")
_APPLICATION: list[object] = []


def _inline(text: str) -> str:
    """Escape text and apply bold, italic, and code markup."""

    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(
        r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text
    )
    return text


def markdown_to_html(
    markdown: str, *, font_size: float = 10.5, label: str = ""
) -> str:
    """Convert a report's Markdown to styled HTML for print.

    Parameters
    ----------
    markdown: str
        Report with a ``#`` headline, ``##`` sections, paragraphs, and
        lists.
    font_size: float, optional
        Body font size in points; headings scale with it. Defaults to
        ``10.5``.
    label: str, optional
        Small capital label above the headline, such as "Research
        Briefing". Defaults to ``""`` (none).

    Returns
    -------
    str
        A complete HTML document. Text is escaped, so Markdown cannot
        inject markup.

    """

    blocks: list[str] = []
    paragraph: list[str] = []
    items: list[str] = []
    ordered = False

    def flush() -> None:
        nonlocal ordered
        if paragraph:
            text = " ".join(paragraph)
            note = text.startswith("*") and text.endswith("*")
            css = ' class="note"' if note else ""
            blocks.append(f"<p{css}>{_inline(text)}</p>")
            paragraph.clear()
        if items:
            tag = "ol" if ordered else "ul"
            body = "".join(f"<li>{_inline(item)}</li>" for item in items)
            blocks.append(f"<{tag}>{body}</{tag}>")
            items.clear()
            ordered = False

    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped.startswith("## "):
            flush()
            # A styled paragraph: Qt sizes <h2> itself, ignoring CSS.
            text = _inline(stripped[3:].strip())
            blocks.append(f'<p class="section">{text}</p>')
        elif stripped.startswith("# "):
            flush()
            blocks.append(f"<h1>{_inline(stripped[2:].strip())}</h1>")
        elif stripped == "---":
            flush()
        elif match := _ITEM.match(line):
            if paragraph:
                flush()
            ordered = ordered or match.group(1) is not None
            items.append(match.group(2))
        else:
            if items:
                flush()
            paragraph.append(stripped)
    flush()
    # Pixels, not points: the page is laid out at 72 units per inch, and
    # Qt would scale points by the screen's resolution.
    size = f"{font_size:.1f}px"
    style = f"""
body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
       font-size: {size}; color: #17202a; }}
h1 {{ font-size: {font_size * 2.0:.1f}px; font-weight: 700;
      margin: 0 0 {font_size * 0.4:.1f}px 0; }}
p.section {{ font-size: {font_size * 0.95:.1f}px; font-weight: 700;
      color: {_ACCENT}; text-transform: uppercase;
      margin: {font_size * 0.9:.1f}px 0 {font_size * 0.2:.1f}px 0; }}
p.label {{ font-size: {font_size * 0.85:.1f}px; font-weight: 700;
      color: {_ACCENT}; text-transform: uppercase;
      margin: 0 0 {font_size * 0.3:.1f}px 0; }}
p {{ margin: 0 0 {font_size * 0.45:.1f}px 0; line-height: 125%; }}
p.note {{ color: {_MUTED}; font-size: {font_size * 0.85:.1f}px; }}
ul, ol {{ margin: 0 0 {font_size * 0.45:.1f}px 0; }}
li {{ margin: 0 0 {font_size * 0.2:.1f}px 0; }}
code {{ font-family: Menlo, monospace; }}
"""
    if label:
        blocks.insert(0, f'<p class="label">{html.escape(label)}</p>')
    return (
        f"<html><head><style>{style}</style></head><body>"
        + "".join(blocks)
        + "</body></html>"
    )


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
        PDF file to write (replaced if it exists; callers choose a free
        name).
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
    writer.setTitle(_title(markdown))
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


def _title(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "QAChat report"
