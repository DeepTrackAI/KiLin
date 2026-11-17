"""Extract lightweight R relationships using textual patterns.

This extension finds which functions an R function, class or S4 method
calls, which class a class extends, and which packages and files a script
loads. It reuses the units of the R reader, so a call belongs to the
function or class method it appears in. Comments and strings are skipped.
It uses regular expressions and known entity names, not an R parser:
calls through an object (``store$add()``), into a package
(``stats::median()``) or built at run time with ``do.call`` are not
followed. The source is never run.

Key Features
------------

- **Calls**

    Link each function, class or method to the plain function calls in it.

- **Inheritance**

    Link a class to the class it extends, through ``contains`` or
    ``inherit``.

- **Loaded code**

    Link a file to the packages it loads with ``library`` or ``require``
    and the files it runs with ``source``.

Module Structure
----------------

- ``extract_code_relations``

    Return code relationships found in one R source file.

Examples
--------
>>> import kilin
>>> report = kilin.scan_repository("/path/to/repository", extensions=[".R"])
>>> files = kilin.read_repository_files(report)
>>> graph = kilin.build_code_graph(files, kilin.extract_source_units(files))

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
from kilin.sources.extensions import sources_r
from kilin.sources.sources import SourceFile, SourceUnit

__all__ = [
    "extract_code_relations",
]

# Comments and string literals, removed before looking for calls.
_SKIPPED: Final[re.Pattern[str]] = re.compile(
    r"#[^\n]*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'"
)
# A plain call, not one through an object ("x$f(", "x@f(") or into a
# package ("pkg::f(").
_CALL: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w.$@:])([A-Za-z.][\w.]*)\s*\("
)
_INHERITS: Final[re.Pattern[str]] = re.compile(
    r"\bcontains\s*=\s*(?:c\s*\(\s*)?[\"']([\w.]+)[\"']"
    r"|\binherit\s*=\s*([A-Za-z.][\w.]*)"
)
_LIBRARY: Final[re.Pattern[str]] = re.compile(
    r"\b(?:library|require|requireNamespace)\s*\(\s*[\"']?([A-Za-z][\w.]*)"
)
_SOURCE: Final[re.Pattern[str]] = re.compile(
    r"\bsource\s*\(\s*[\"']([^\"']+\.[Rr])[\"']"
)
# Statements and loaders followed by "(", which are not calls.
_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "if",
        "for",
        "while",
        "switch",
        "repeat",
        "function",
        "return",
        "library",
        "require",
        "requireNamespace",
        "source",
    }
)


def extract_code_relations(
    source_file: SourceFile,
    known: dict[str, set[str]],
) -> set[CodeRelation]:
    """Return code relationships found in one R source file.

    Parameters
    ----------
    source_file: SourceFile
        R source and its repository path.
    known: dict[str, set[str]]
        Known code entities used to resolve short names.

    Returns
    -------
    set[CodeRelation]
        Calls, inheritance and loaded packages and files, found without
        running code.

    Notes
    -----
    Regular-expression matches are structural hints, not a full parse. A
    ``source`` path is read relative to the file's folder, as when the
    script is run from there, or else to the repository root.

    """

    relations: set[CodeRelation] = set()
    module = _module_name(source_file.path)
    for package in _LIBRARY.finditer(source_file.text):
        relations.add(CodeRelation(module, "imports", package.group(1)))
    for script in _SOURCE.finditer(source_file.text):
        target = _sourced_module(source_file.path, script.group(1), known)
        if target is not None:
            relations.add(CodeRelation(module, "imports", target))
    # Each unit with the graph name of the code around it: a class's own
    # lines belong to the class, other code between declarations to the
    # file.
    pending: list[tuple[SourceUnit, str]] = [
        (unit, module) for unit in sources_r.extract_source_units(source_file)
    ]
    while pending:
        unit, around = pending.pop()
        # Graph names: module.function, module.Class, module.Class.method.
        owner = around if unit.symbol is None else f"{module}.{unit.symbol}"
        if unit.kind == "r_class":
            base = _INHERITS.search(unit.text)
            if base is not None:
                name = base.group(1) or base.group(2)
                relations.add(
                    CodeRelation(owner, "inherits", _resolve_name(name, known))
                )
            if unit.children:
                pending.extend((child, owner) for child in unit.children)
                continue
        for call in _CALL.finditer(_SKIPPED.sub(" ", unit.text)):
            target = call.group(1)
            if target in _KEYWORDS:
                continue
            relations.add(
                CodeRelation(
                    owner,
                    "calls",
                    _resolve_name(target, known, owner, module),
                )
            )
    return relations


def _sourced_module(
    path: object, location: str, known: dict[str, set[str]]
) -> str | None:
    """Return the graph name of a file run with ``source``.

    Parameters
    ----------
    path: pathlib.Path
        Repository path of the file that runs it.
    location: str
        Path written in ``source("...")``.
    known: dict[str, set[str]]
        Known code entities, used to tell which reading names a file.

    Returns
    -------
    str | None
        The module name next to the file, or from the repository root when
        only that one is known; None for a path outside the repository.

    """

    folder = PurePosixPath(str(path).replace("\\", "/")).parent
    readings = []
    for base in (folder, PurePosixPath(".")):
        normalized = posixpath.normpath((base / location).as_posix())
        if not normalized.startswith(".."):
            readings.append(_module_name(PurePosixPath(normalized)))
    for reading in readings:
        if reading in known.get(reading.rsplit(".", 1)[-1], set()):
            return reading
    return readings[0] if readings else None
