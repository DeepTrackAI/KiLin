"""Link an HTML page to the scripts it loads and the functions it calls.

In a project that reads HTML as code, a page's ``<script src="...">``
tags become ``imports`` of those script files, and leading direct calls
in inline event handlers, such as ``onclick="save()"``, become ``calls``, so a
question about a JavaScript function also finds the page that uses it.
Handlers attached from JavaScript (``addEventListener``) are found by the
JavaScript extension instead. Actual HTML tags are distinguished from
comments, script contents and inert templates. The page is never rendered
or run. This hook does not analyze arbitrary JavaScript handler expressions.

Key Features
------------

- **Loaded scripts**

    Link the page to the local script files it loads.

- **Inline handlers**

    Link the page to the functions its ``on...`` attributes call.

Module Structure
----------------

- ``extract_code_relations``

    Return the scripts one HTML page loads and the functions it calls.

Examples
--------
>>> import kilin
>>> report = kilin.scan_repository("/path/to/app", extensions=[".html", ".js"])
>>> files = kilin.read_repository_files(report, as_code=True)
>>> graph = kilin.build_code_graph(
...     files, kilin.extract_source_units(files, as_code=True)
... )

"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import PurePosixPath
import posixpath
import re
from typing import Final

from kilin.retrievals.code_graph.code_graph import (
    CodeRelation,
    _module_name,
    _resolve_name,
)
from kilin.sources.sources import SourceFile

__all__ = [
    "extract_code_relations",
]

# Recognize only the leading direct call in an inline event handler.
_HANDLER_CALL: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:return\s+)?([A-Za-z_$][\w$]*)\s*\("
)
# External script references are not local repository paths.
_EXTERNAL: Final[re.Pattern[str]] = re.compile(r"^(?:[a-z][\w+.-]*:|//)", re.I)
_INERT: Final[frozenset[str]] = frozenset({"template", "noscript"})


class _MarkupLinks(HTMLParser):
    """Collect real HTML script tags and inline event-handler attributes.

    HTMLParser distinguishes actual tags from comments and script text.
    Template contents are inert and therefore do not load scripts or
    attach event handlers until separately instantiated.

    """

    def __init__(self) -> None:
        """Initialize an empty collection of links."""

        super().__init__(convert_charrefs=True)
        self.scripts: list[str] = []
        self.handlers: list[str] = []
        self.inert: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        """Collect links from a non-inert element.

        Parameters
        ----------
        tag: str
            Lowercase HTML element name.
        attrs: list[tuple[str, str | None]]
            Parsed attributes, with character references resolved.

        """

        if tag in _INERT:
            self.inert.append(tag)
            return
        if self.inert:
            return
        for name, value in attrs:
            if value is None:
                continue
            if tag == "script" and name == "src":
                self.scripts.append(value)
            elif name.startswith("on") and name[2:].isalpha():
                match = _HANDLER_CALL.match(value)
                if match is not None:
                    self.handlers.append(match.group(1))

    def handle_endtag(self, tag: str) -> None:
        """Close an inert element without processing its descendants.

        Parameters
        ----------
        tag: str
            Lowercase HTML element name.

        """

        if tag in self.inert:
            del self.inert[len(self.inert) - 1 - self.inert[::-1].index(tag)]


def extract_code_relations(
    source_file: SourceFile,
    known: dict[str, set[str]],
) -> set[CodeRelation]:
    """Return the scripts one HTML page loads and the functions it calls.

    Parameters
    ----------
    source_file: SourceFile
        HTML source and its repository path.
    known: dict[str, set[str]]
        Known code entities used to resolve function names.

    Returns
    -------
    set[CodeRelation]
        ``imports`` of local script files, named as the graph names them,
        and ``calls`` of the functions inline handlers call.

    """

    relations: set[CodeRelation] = set()
    page = _module_name(source_file.path)
    folder = PurePosixPath(source_file.path.as_posix()).parent
    markup = _MarkupLinks()
    markup.feed(source_file.text)
    markup.close()
    for location in markup.scripts:
        location = location.split("?", 1)[0].split("#", 1)[0]
        if not location or _EXTERNAL.match(location):
            continue
        path = posixpath.normpath(
            location.lstrip("/")
            if location.startswith("/")
            else (folder / location).as_posix()
        )
        if path.startswith(".."):
            continue
        relations.add(
            CodeRelation(page, "imports", _module_name(PurePosixPath(path)))
        )
    for handler in markup.handlers:
        relations.add(
            CodeRelation(page, "calls", _resolve_name(handler, known))
        )
    return relations
