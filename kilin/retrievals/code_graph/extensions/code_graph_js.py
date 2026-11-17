"""Extract lightweight JavaScript relationships using textual patterns.

This extension finds which functions a JavaScript function or method calls
and which class a class extends. It reuses the units of the JavaScript
reader, so calls belong to the functions, methods or module context in
which they appear. Shared lexical masking excludes comments, strings,
templates and likely regex literals. Patterns are not a JavaScript parser:
calls through objects (``store.save()``), callbacks passed by name, and
code constructed at runtime are not followed. No source is executed.

Key Features
------------

- **Calls**

    Link each function or method to the plain function calls in its body.

- **Inheritance**

    Link a class to the class it extends.

Module Structure
----------------

- ``extract_code_relations``

    Return code relationships found in one JavaScript source file.

Examples
--------
>>> import kilin
>>> report = kilin.scan_repository("/path/to/repository", extensions=[".js"])
>>> files = kilin.read_repository_files(report)
>>> graph = kilin.build_code_graph(files, kilin.extract_source_units(files))

"""

from __future__ import annotations

import re
from typing import Final

from kilin.retrievals.code_graph.code_graph import (
    CodeRelation,
    _module_name,
    _resolve_name,
)
from kilin.sources import source_kinds as kinds
from kilin.sources.extensions import sources_js
from kilin.sources.sources import SourceFile, SourceUnit

__all__ = [
    "extract_code_relations",
]

# A plain call, not one through an object such as "console.log(".
_CALL: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w$.#])([A-Za-z_$][\w$]*)\s*\("
)
_EXTENDS: Final[re.Pattern[str]] = re.compile(
    r"\bclass\s+[A-Za-z_$][\w$]*\s+extends\s+([A-Za-z_$][\w$.]*)"
)
# Statements and operators followed by "(", which are not calls.
_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "if",
        "for",
        "while",
        "switch",
        "catch",
        "function",
        "return",
        "typeof",
        "await",
        "new",
        "super",
        "with",
        "import",
        "async",
        "yield",
        "delete",
        "void",
        "in",
        "of",
        "instanceof",
    }
)


def _opens_declaration_body(text: str, start: int) -> bool:
    """Distinguish a method signature from an ordinary call.

    Parameters
    ----------
    text: str
        Lexically masked source-unit text.
    start: int
        Offset immediately after a candidate call's opening parenthesis.

    Returns
    -------
    bool
        Whether its matching parenthesis precedes a declaration body.

    """

    depth = 1
    position = start
    while position < len(text) and depth:
        if text[position] == "(":
            depth += 1
        elif text[position] == ")":
            depth -= 1
        position += 1
    if depth:
        return False
    while position < len(text) and text[position].isspace():
        position += 1
    return position < len(text) and text[position] == "{"


def extract_code_relations(
    source_file: SourceFile,
    known: dict[str, set[str]],
) -> set[CodeRelation]:
    """Return code relationships found in one JavaScript source file.

    Parameters
    ----------
    source_file: SourceFile
        JavaScript source and its repository path.
    known: dict[str, set[str]]
        Known code entities used to resolve short names.

    Returns
    -------
    set[CodeRelation]
        Calls and inheritance relationships found without executing code.

    Notes
    -----
    Regular-expression matches are structural hints, not a full parse.
    A function calling itself is not recorded, because its declaration
    line looks like a call.

    """

    relations: set[CodeRelation] = set()
    module = _module_name(source_file.path)
    masked_lines = sources_js._mask_non_code(source_file.text).splitlines(
        keepends=True
    )
    # Each unit with the graph name of the code around it: a class line
    # or field belongs to its class, other code between declarations to
    # the file.
    pending: list[tuple[SourceUnit, str]] = [
        (unit, module) for unit in sources_js.extract_source_units(source_file)
    ]
    while pending:
        unit, around = pending.pop()
        # Graph names: module.function, module.Class, module.Class.method.
        owner = around if unit.symbol is None else f"{module}.{unit.symbol}"
        text = "".join(masked_lines[unit.start_line - 1 : unit.end_line])
        if unit.kind == kinds.JS_CLASS:
            base = _EXTENDS.search(text)
            if base is not None:
                relations.add(
                    CodeRelation(
                        owner, "inherits", _resolve_name(base.group(1), known)
                    )
                )
            if unit.children:
                pending.extend((child, owner) for child in unit.children)
            # A class declaration is not a call. Only recognized class
            # context children can contribute initializer calls.
            continue
        own_name = owner.rsplit(".", 1)[-1]
        for call in _CALL.finditer(text):
            target = call.group(1)
            if target in _KEYWORDS or target == own_name:
                continue
            if _opens_declaration_body(text, call.end()):
                continue
            relations.add(
                CodeRelation(
                    owner,
                    "calls",
                    _resolve_name(target, known, owner, module),
                )
            )
    return relations
