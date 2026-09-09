"""Extract MATLAB calls and inheritance."""

from __future__ import annotations

import re

from qachat.retrievals.code_graph import (
    CodeRelation,
    _module_name,
    _resolve_name,
)
from qachat.sources.sources import SourceFile

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
_CALL = re.compile(r"\b([A-Za-z]\w*)\s*\(")
_CONTROL = {"if", "for", "while", "switch"}


def extract_code_relations(
    source_file: SourceFile,
    known: dict[str, set[str]],
) -> set[CodeRelation]:
    """Extract relationships without executing MATLAB source."""

    relations: set[CodeRelation] = set()
    module = _module_name(source_file.path)
    class_match = _CLASS.search(source_file.text)
    parent = module

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

    functions = list(_FUNCTION.finditer(source_file.text))
    for index, function_match in enumerate(functions):
        function = f"{parent}.{function_match.group(1)}"
        end = (
            functions[index + 1].start()
            if index + 1 < len(functions)
            else None
        )
        body = source_file.text[function_match.end() : end]
        for call_match in _CALL.finditer(body):
            target = call_match.group(1)
            if target.lower() in _CONTROL:
                continue
            relations.add(
                CodeRelation(
                    function,
                    "calls",
                    _resolve_name(target, known),
                )
            )

    return relations
