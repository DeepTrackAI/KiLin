"""Extract Python relationships for the shared code graph.

Functions
---------
extract_code_relations
    Find calls, imports, and inheritance without executing Python source.

"""

from __future__ import annotations

import ast

from qachat.retrievals.code_graph.code_graph import (
    CodeRelation,
    _module_name,
    _resolve_name,
)
from qachat.sources.sources import SourceFile


__all__ = [
    "extract_code_relations",
]


def extract_code_relations(
    source_file: SourceFile,
    known: dict[str, set[str]],
) -> set[CodeRelation]:
    """Return code relationships found in one Python source file.

    Parameters
    ----------
    source_file: SourceFile
        Python source and its repository path.
    known: dict[str, set[str]]
        Known code entities used to resolve short names.

    Returns
    -------
    set[CodeRelation]
        Calls, imports, and inheritance relationships found statically.

    """

    try:
        tree = ast.parse(source_file.text)
    except SyntaxError:
        # Invalid or incomplete source remains usable elsewhere in QAChat.
        # Code-graph extraction simply contributes no Python relationships.
        return set()

    visitor = _Visitor(_module_name(source_file.path), known)
    visitor.visit(tree)
    return visitor.relations


class _Visitor(ast.NodeVisitor):
    """Collect relationships while keeping the current code entity in scope."""

    def __init__(
        self: "_Visitor",
        module: str,
        known: dict[str, set[str]],
    ) -> None:
        self.parents = [module]
        self.known = known
        self.relations: set[CodeRelation] = set()

    def visit_Import(self: "_Visitor", node: ast.Import) -> None:
        """Record module imports from the current entity."""

        for alias in node.names:
            self._add("imports", alias.name)

    def visit_ImportFrom(
        self: "_Visitor",
        node: ast.ImportFrom,
    ) -> None:
        """Record imported names from the current entity."""

        module = node.module or ""
        for alias in node.names:
            target = ".".join(part for part in (module, alias.name) if part)
            self._add("imports", target)

    def visit_ClassDef(
        self: "_Visitor",
        node: ast.ClassDef,
    ) -> None:
        """Record inheritance and visit the class in its own scope."""

        name = f"{self.parents[-1]}.{node.name}"

        for base in node.bases:
            target = _reference(base)
            if target is not None:
                self.relations.add(
                    CodeRelation(
                        name,
                        "inherits",
                        _resolve_name(target, self.known),
                    )
                )

        self.parents.append(name)
        self.generic_visit(node)
        self.parents.pop()

    def visit_FunctionDef(
        self: "_Visitor",
        node: ast.FunctionDef,
    ) -> None:
        """Visit a function with that function as the current entity."""

        self._visit_function(node)

    def visit_AsyncFunctionDef(
        self: "_Visitor",
        node: ast.AsyncFunctionDef,
    ) -> None:
        """Visit an async function with that function as the current entity."""

        self._visit_function(node)

    def visit_Call(self: "_Visitor", node: ast.Call) -> None:
        """Record a call from the current code entity."""

        target = _reference(node.func)
        if target is not None:
            self._add("calls", _resolve_name(target, self.known))

        self.generic_visit(node)

    def _visit_function(
        self: "_Visitor",
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """Visit one function body while preserving its entity scope."""

        self.parents.append(f"{self.parents[-1]}.{node.name}")
        self.generic_visit(node)
        self.parents.pop()

    def _add(self: "_Visitor", relation: str, target: str) -> None:
        """Add one relationship from the current code entity."""

        self.relations.add(
            CodeRelation(
                self.parents[-1],
                relation,
                target,
            )
        )


def _reference(node: ast.AST) -> str | None:
    """Return a dotted name when an expression is a static reference."""

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        parent = _reference(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr

    return None
