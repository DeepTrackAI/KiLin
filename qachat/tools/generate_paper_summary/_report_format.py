"""Format report Markdown into safe HTML for print rendering.

Report content and structure belong to ``generate_paper_summary`` rather
than to a particular presentation backend. This module has no Qt dependency;
the optional desktop PDF adapter imports it when rendering a report.

The supported Markdown is deliberately limited to headings, paragraphs,
lists, emphasis, inline code and rules, matching generated report templates.

Key Features
------------

- **Safe text**

    Escape source content before inserting the supported inline markup.

- **Report styling**

    Convert report sections into styled HTML with adjustable body size.

Module Structure
----------------

- ``markdown_to_html``

    Convert the supported report Markdown into complete print-ready HTML.

- ``report_title``

    Extract the report title for document metadata.

Examples
--------
>>> from qachat.tools.generate_paper_summary._report_format import (
...     markdown_to_html,
... )
>>> "Example" in markdown_to_html("# Example")
True

"""

from __future__ import annotations

import html
import re
from typing import Final

__all__ = ["markdown_to_html", "report_title"]

_ACCENT: Final[str] = "#334e43"
_MUTED: Final[str] = "#5f6b66"
_ITEM = re.compile(r"^\s*(?:[-*•]|(\d+)[.)])\s+(.*)$")
def _inline(text: str) -> str:
    """Escape text and apply bold, italic, and code markup.

    Parameters
    ----------
    text: str
        Text to analyze or transform.

    Returns
    -------
    str
        HTML-escaped text with supported inline formatting.

    """

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
        """Append pending Markdown paragraphs and lists as HTML blocks.

        """

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
            # Qt sizes <h2> itself, ignoring CSS, so style a paragraph.
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


def report_title(markdown: str) -> str:
    """Extract the first Markdown level-one heading as the report title.

    Parameters
    ----------
    markdown: str
        Report Markdown whose top-level heading is inspected.

    Returns
    -------
    str
        First top-level heading, or ``"QAChat report"`` if absent.

    """

    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "QAChat report"
