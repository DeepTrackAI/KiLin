"""Extract lightweight MATLAB relationships using textual patterns.

This extension recognizes selected inheritance and function-call forms in
MATLAB source, skipping comments and strings. A call written as
``py.module.function(...)`` is a call into Python and is linked to that
Python function. It uses regular expressions and known entity names, not a
MATLAB parser or interpreter. Complex declarations and nested constructs
can therefore be incomplete or ambiguous. The source is never executed by
this hook.

Key Features
------------

- **Textual graph hints**

    Add recognizable MATLAB inheritance and call relationships to the
    structural graph.

- **Shared name resolution**

    Use repository entity names supplied by the graph builder.

Module Structure
----------------

- ``CALLS_INTO``

    Prefix MATLAB writes before a call into Python (``py``).

- ``extract_code_relations``

    Return code relationships found in one MATLAB source file.

Examples
--------
>>> import kilin
>>> report = kilin.scan_repository("/path/to/repository", extensions=[".m"])
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
from kilin.sources.sources import SourceFile

__all__ = [
    "CALLS_INTO",
    "extract_code_relations",
]

# MATLAB calls Python as py.module.function(...); the graph links such a
# call to the Python function.
CALLS_INTO: Final[dict[str, str]] = {"py": ".py"}

# Match only structural forms needed by the graph. This keeps the parser
# deterministic without attempting to interpret MATLAB at runtime.
_CLASS = re.compile(
    r"^\s*classdef(?:\s*\([^)]*\))?\s+([A-Za-z]\w*)"
    r"(?:\s*<\s*([A-Za-z]\w*))?",
    re.IGNORECASE | re.MULTILINE,
)
_FUNCTION = re.compile(
    r"^\s*function\s+(?:(?:\[[^]]*\]|[A-Za-z]\w*)\s*=\s*)?"
    r"([A-Za-z]\w*)\s*\(",
    re.IGNORECASE | re.MULTILINE,
)
# A call, possibly through a dotted name ("obj.save(", "py.mod.f(").
_CALL = re.compile(r"(?<![\w.])([A-Za-z]\w*(?:\.[A-Za-z]\w*)*)\s*\(")
# Comments and strings, removed before looking for calls. A quote right
# after a name, bracket or another quote is the transpose operator.
_SKIPPED = re.compile(
    r"^\s*%\{.*?^\s*%\}"
    r"|%[^\n]*"
    r'|"(?:[^"\n]|"")*"'
    r"|(?<![\w)\]}'.])'(?:[^'\n]|'')*'",
    re.DOTALL | re.MULTILINE,
)

# MATLAB control statements can resemble function calls in simple text
# matching, so exclude them from call relationships.
_CONTROL = {"if", "for", "while", "switch"}


def extract_code_relations(
    source_file: SourceFile,
    known: dict[str, set[str]],
) -> set[CodeRelation]:
    """Return code relationships found in one MATLAB source file.

    Parameters
    ----------
    source_file: SourceFile
        MATLAB source and its repository path.
    known: dict[str, set[str]]
        Known code entities used to resolve short names.

    Returns
    -------
    set[CodeRelation]
        Calls and inheritance relationships found without executing code.

    Notes
    -----
    Regular-expression matches are structural hints, not full MATLAB parsing.
    Comments and strings are skipped. Multiline declarations and nested
    definitions can be ambiguous or missed. ``py.module.function(...)`` is
    a call into Python (``CALLS_INTO``).

    """

    relations: set[CodeRelation] = set()
    module = _module_name(source_file.path)
    class_match = _CLASS.search(source_file.text)
    parent = module

    # A class becomes the parent entity for functions declared in the file.
    if class_match is not None:
        parent = f"{module}.{class_match.group(1)}"
        base = class_match.group(2)

        if base is not None:
            relations.add(
                CodeRelation(
                    parent,
                    "inherits",
                    _resolve_name(base, known),
                )
            )

    # Bound each function by the next function declaration so calls are
    # attached to the function that contains them.
    functions = list(_FUNCTION.finditer(source_file.text))
    for index, function_match in enumerate(functions):
        function = f"{parent}.{function_match.group(1)}"
        end = (
            functions[index + 1].start()
            if index + 1 < len(functions)
            else None
        )
        body = _SKIPPED.sub(" ", source_file.text[function_match.end() : end])

        for call_match in _CALL.finditer(body):
            parts = call_match.group(1).split(".")
            # A call through an object is matched by its last name; a call
            # into another language keeps its prefix.
            target = (
                call_match.group(1)
                if len(parts) > 1 and parts[0] in CALLS_INTO
                else parts[-1]
            )
            if target.lower() in _CONTROL:
                continue

            relations.add(
                CodeRelation(
                    function,
                    "calls",
                    _resolve_name(target, known, function, module),
                )
            )

    return relations
