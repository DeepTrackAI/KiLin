"""Link an HTML page to the scripts it loads and the functions it calls.

In a project that reads HTML as code, a page's ``<script src="...">``
tags become ``imports`` of those script files, and inline event handlers
such as ``onclick="save()"`` become ``calls`` of those functions, so a
question about a JavaScript function also finds the page that uses it.
Handlers attached from JavaScript (``addEventListener``) are found by the
JavaScript extension instead. HTML read as page text has no tags, so this
hook then finds nothing. The page is never rendered or run.

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

_SCRIPT: Final[re.Pattern[str]] = re.compile(
    r"<script\b[^>]*\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE
)
_HANDLER: Final[re.Pattern[str]] = re.compile(
    r"\bon[a-z]+\s*=\s*[\"']\s*(?:return\s+)?([A-Za-z_$][\w$]*)\s*\(",
    re.IGNORECASE,
)
# A script from another site or a packaged resource is not in the project.
_EXTERNAL: Final[re.Pattern[str]] = re.compile(r"^(?:[a-z][\w+.-]*:|//)", re.I)


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
    for script in _SCRIPT.finditer(source_file.text):
        location = script.group(1).split("?", 1)[0].split("#", 1)[0]
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
    for handler in _HANDLER.finditer(source_file.text):
        relations.add(
            CodeRelation(page, "calls", _resolve_name(handler.group(1), known))
        )
    return relations
