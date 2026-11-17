"""Read HTML files as page text, or as code in code projects.

A saved web page, an exported report or a documentation page is read like
a Word file: the text a reader sees, with headings as Markdown ``#`` lines,
list items as ``-`` lines and each table row as one line with its cells
separated by `` | ``. Scripts, styles and other markup are left out, and
extraction reuses the Markdown heading sections, so a passage is cited with
its heading. Line numbers then refer to the extracted text.

In a project that reads its files as code (``code_graph`` set to true),
the HTML is kept as written, tags and scripts included, because the
markup is what a developer asks about. A large file is split where its
main parts start, such as ``<head>``, ``<body>``, each ``<section>``,
``<script>`` and ``<style>``, and each part is labelled, for example
``section#settings`` or ``script src=app.js``. A part of a few lines,
such as one ``<script src>`` tag, joins the part before it. Line
numbers then refer to the file itself. No script is ever run.

Key Features
------------

- **Page text**

    Keep headings, paragraphs, lists, tables and preformatted text in
    reading order; leave out scripts, styles and markup.

- **Code**

    Keep the file as written and label its main parts.

Module Structure
----------------

- ``read_source_file``

    Read the page text of one HTML file.

- ``extract_source_units``

    Return Markdown-style heading sections of the page text.

- ``extract_code_units``

    Return one HTML file as written, split at its main parts.

- ``page_text``

    The text a reader sees in an HTML document, as Markdown-style lines.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import kilin
>>> report = kilin.scan_repository("/path/to/site", extensions=[".html"])
>>> pages = kilin.extract_source_units(kilin.read_repository_files(report))
>>> code = kilin.extract_source_units(
...     kilin.read_repository_files(report, as_code=True), as_code=True
... )

"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Final

from kilin.sources.extensions import sources_md
from kilin.sources.sources import (
    SourceFile,
    SourceUnit,
    _source_unit,
)

__all__ = [
    "read_source_file",
    "extract_source_units",
    "extract_code_units",
    "page_text",
]

# Elements whose content a reader never sees as text.
_HIDDEN: Final[frozenset[str]] = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
        "iframe",
        "object",
        "head",
    }
)
# Elements that start a new block of text.
_BLOCKS: Final[frozenset[str]] = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "dd",
        "details",
        "dialog",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "header",
        "hr",
        "main",
        "nav",
        "ol",
        "p",
        "section",
        "summary",
        "table",
        "ul",
    }
)
_HEADINGS: Final[dict[str, int]] = {
    f"h{level}": level for level in range(1, 7)
}
# Body lines that Markdown would read as a heading or a code fence.
_MARKDOWN_MARKUP: Final[re.Pattern[str]] = re.compile(r"^(?:#|```|~~~)")
_CELL_SEPARATOR: Final[str] = " | "
# Marks a <br> in collected text; line ends in the source are only spaces.
_LINE_BREAK: Final[str] = "\x00"
_CHARSET: Final[re.Pattern[bytes]] = re.compile(
    rb"<meta[^>]+charset\s*=\s*[\"']?([\w-]+)", re.IGNORECASE
)
# The parts of a page a large file is split at when it is read as code.
_PART: Final[re.Pattern[str]] = re.compile(
    r"^\s*<(head|body|header|nav|main|section|article|aside|footer|form"
    r"|dialog|template|script|style)\b([^>]*)",
    re.IGNORECASE,
)
# Shorter parts join the part before them.
_MIN_PART_LINES: Final[int] = 10
_ATTRIBUTE: Final[str] = r"\b{}\s*=\s*[\"']([^\"']+)[\"']"


class _PageText(HTMLParser):
    """Collect the visible text of an HTML document as Markdown lines."""

    def __init__(self) -> None:
        """Start with no text."""

        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.title: list[str] = []
        self.words: list[str] = []
        self.prefix = ""
        # Open hidden elements, innermost last.
        self.hidden: list[str] = []
        self.in_title = False
        self.preformatted = 0
        self.cells: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        """Open an element.

        Parameters
        ----------
        tag: str
            Lowercase element name.
        attrs: list[tuple[str, str | None]]
            Element attributes; not used.

        """

        if tag == "title":
            self.in_title = True
        if tag == "body" and "head" in self.hidden:
            # The end of <head> may be left out.
            self.hidden = [name for name in self.hidden if name != "head"]
        if tag in _HIDDEN:
            self.hidden.append(tag)
        if self.hidden:
            return
        if tag in _HEADINGS:
            self._end_block()
            self.prefix = "#" * _HEADINGS[tag] + " "
        elif tag == "li":
            self._end_block()
            self.prefix = "- "
        elif tag == "tr":
            self._end_block()
            self.cells = []
        elif tag in ("td", "th"):
            self._end_cell()
        elif tag == "pre":
            self._end_block()
            self.preformatted += 1
        elif tag == "br":
            self._end_line()
        elif tag in _BLOCKS:
            self._end_block()

    def handle_endtag(self, tag: str) -> None:
        """Close an element.

        Parameters
        ----------
        tag: str
            Lowercase element name.

        """

        if tag == "title":
            self.in_title = False
        if tag in _HIDDEN:
            if tag in self.hidden:
                del self.hidden[
                    len(self.hidden) - 1 - self.hidden[::-1].index(tag)
                ]
            return
        if self.hidden:
            return
        if tag in ("td", "th"):
            self._end_cell()
        elif tag == "tr":
            self._end_cell()
            if self.cells:
                self.blocks.append(_CELL_SEPARATOR.join(self.cells))
            self.cells = None
        elif tag == "pre" and self.preformatted:
            self.preformatted -= 1
            text = "".join(self.words).strip("\n")
            self.words = []
            if text.strip():
                self.blocks.append(f"```\n{text}\n```")
        elif tag in _HEADINGS or tag == "li" or tag in _BLOCKS:
            self._end_block()

    def handle_data(self, data: str) -> None:
        """Keep visible text.

        Parameters
        ----------
        data: str
            Text between tags, with character references converted.

        """

        if self.in_title:
            self.title.append(data)
        if not self.hidden:
            self.words.append(data)

    def text(self) -> str:
        """Return the collected text.

        Returns
        -------
        str
            Blocks separated by blank lines, with the page title first when
            the page has no top-level heading.

        """

        self._end_cell()
        self._end_block()
        title = " ".join("".join(self.title).split())
        blocks = list(self.blocks)
        if title and not any(block.startswith("# ") for block in blocks):
            blocks.insert(0, f"# {title}")
        return "\n\n".join(blocks) + "\n" if blocks else ""

    def _end_cell(self) -> None:
        """Finish the current table cell, if any."""

        if self.cells is None:
            return
        cell = " ".join("".join(self.words).split())
        self.words = []
        if cell:
            self.cells.append(cell)

    def _end_line(self) -> None:
        """Finish a line inside the current block (for ``<br>``)."""

        if self.preformatted:
            self.words.append("\n")
        elif self.cells is None:
            self.words.append(_LINE_BREAK)

    def _end_block(self) -> None:
        """Finish the current block of text."""

        if self.preformatted or self.cells is not None:
            return
        lines = [
            " ".join(line.split())
            for line in "".join(self.words).split(_LINE_BREAK)
        ]
        self.words = []
        lines = [line for line in lines if line]
        # A list item or heading whose text is in a nested block keeps its
        # prefix until that text arrives.
        if not lines:
            return
        # A heading keeps its "#"; any other block is escaped, prefix included
        # ("- # 1" cannot read as a heading, so escaping it is harmless).
        first = (
            _escape(lines[0]) if not self.prefix else self.prefix + lines[0]
        )
        self.blocks.append(
            "\n".join([first, *(_escape(line) for line in lines[1:])])
        )
        self.prefix = ""


def _escape(line: str) -> str:
    """Keep a text line from reading as a Markdown heading or code fence.

    Parameters
    ----------
    line: str
        One line of page text.

    Returns
    -------
    str
        The line, with a backslash before a leading ``#`` or fence.

    """

    if _MARKDOWN_MARKUP.match(line):
        return f"\\{line}"
    return line


def page_text(html: str) -> str:
    """The text a reader sees in an HTML document, as Markdown-style lines.

    Parameters
    ----------
    html: str
        HTML source.

    Returns
    -------
    str
        Headings as ``#`` lines, list items as ``-`` lines, table rows with
        cells separated by `` | `` and preformatted text in code fences,
        with blank lines between blocks; empty when the page has no text.

    """

    parser = _PageText()
    parser.feed(html)
    parser.close()
    return parser.text()


def _decode(data: bytes, relative_path: Path) -> str:
    """Decode an HTML file as UTF-8 or as the charset it declares.

    Parameters
    ----------
    data: bytes
        File contents.
    relative_path: pathlib.Path
        Repository-relative path, for the error message.

    Returns
    -------
    str
        Decoded text.

    Raises
    ------
    ValueError
        If the text is neither UTF-8 nor in a declared, known charset.

    """

    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    declared = _CHARSET.search(data[:4096])
    if declared is not None:
        try:
            return data.decode(declared.group(1).decode("ascii"))
        except (LookupError, UnicodeDecodeError):
            pass
    raise ValueError(
        f"Selected file is not UTF-8 text: {relative_path.as_posix()}"
    )


def read_source_file(target: Path, relative_path: Path) -> SourceFile:
    """Read the page text of one HTML file.

    Parameters
    ----------
    target: pathlib.Path
        Resolved on-disk file path to read.
    relative_path: pathlib.Path
        Repository-relative path stored in the returned source record.

    Returns
    -------
    SourceFile
        The page text, as returned by ``page_text``.

    Raises
    ------
    ValueError
        If the file is neither UTF-8 nor in the charset it declares.
    OSError
        If the file cannot be read.

    """

    html = _decode(target.read_bytes(), relative_path)
    return SourceFile(relative_path, page_text(html))


def extract_source_units(source_file: SourceFile) -> list[SourceUnit]:
    """Return Markdown-style heading sections of the page text.

    Parameters
    ----------
    source_file: SourceFile
        Page text from ``read_source_file``.

    Returns
    -------
    list[SourceUnit]
        Natural units in reading order.

    """

    return sources_md.extract_source_units(source_file)


def _part_label(tag: str, attributes: str) -> str:
    """Name one main part of a page, such as ``section#settings``.

    Parameters
    ----------
    tag: str
        Element name.
    attributes: str
        Text after the element name in its opening tag.

    Returns
    -------
    str
        The element name with its id, else its first class, else the file
        a script or style loads.

    """

    tag = tag.lower()
    identifier = re.search(_ATTRIBUTE.format("id"), attributes, re.I)
    if identifier is not None:
        return f"{tag}#{identifier.group(1)}"
    classes = re.search(_ATTRIBUTE.format("class"), attributes, re.I)
    if classes is not None and classes.group(1).split():
        return f"{tag}.{classes.group(1).split()[0]}"
    loaded = re.search(_ATTRIBUTE.format("(?:src|href)"), attributes, re.I)
    if loaded is not None:
        return f"{tag} src={loaded.group(1)}"
    return tag


def extract_code_units(source_file: SourceFile) -> list[SourceUnit]:
    """Return one HTML file as written, split at its main parts.

    Parameters
    ----------
    source_file: SourceFile
        HTML source read as written.

    Returns
    -------
    list[SourceUnit]
        One unit for the whole file. Its children start at each main part,
        such as ``<head>``, ``<section id="settings">`` or ``<script>``; a
        part shorter than ten lines joins the part before it. The children
        are used only when the file is too large for one passage.

    """

    lines = source_file.text.splitlines(keepends=True)
    if not lines:
        return []
    starts = [
        (number, _part_label(match.group(1), match.group(2)))
        for number, line in enumerate(lines, start=1)
        if (match := _PART.match(line)) is not None
    ]
    # Each part as [start, end, label]; the text before the first part has
    # no label. A part of a few lines, such as one <script src> tag, joins
    # the part before it, so passages are not a line long.
    parts: list[list[int | str | None]] = []
    first = starts[0][0] if starts else len(lines) + 1
    if first > 1:
        parts.append([1, first - 1, None])
    for index, (start, label) in enumerate(starts):
        end = (
            starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        )
        if parts and end - start + 1 < _MIN_PART_LINES:
            parts[-1][1] = end
            # The text before the first part takes the first label.
            parts[-1][2] = parts[-1][2] or label
        else:
            parts.append([start, end, label])
    children = tuple(
        unit
        for start, end, label in parts
        if (
            unit := _source_unit(
                source_file.path,
                lines,
                int(start),
                int(end),
                kind="html_context" if label is None else "html_element",
                section=None if label is None else str(label),
            )
        )
        is not None
    )
    whole = _source_unit(
        source_file.path,
        lines,
        1,
        len(lines),
        kind="html_file",
        children=children if len(children) > 1 else (),
    )
    return [] if whole is None else [whole]
