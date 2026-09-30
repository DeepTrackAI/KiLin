"""Extract Python relationships for the shared code graph.

Functions
---------
extract_code_relations
    Find calls, imports, and inheritance without executing Python source.
extract_module_constants
    Find top-level SCREAMING_SNAKE_CASE constant names for name resolution.

"""

from __future__ import annotations

import ast
import re

from qachat.retrievals.code_graph.code_graph import (
    CodeRelation,
    _module_name,
    _resolve_name,
)
from qachat.sources.sources import SourceFile


__all__ = [
    "extract_code_relations",
    "extract_module_constants",
]

_CONSTANT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


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


def extract_module_constants(source_file: SourceFile) -> dict[str, str]:
    """Return top-level SCREAMING_SNAKE_CASE constant names for one file.

    A plain reference to a module-level constant (for example using
    ``MODEL_CHOICES`` without calling it) is otherwise invisible to
    ``extract_code_relations``: only calls, imports, and inheritance are
    tracked there. These constants deliberately are NOT registered as
    their own graph entities -- they stay merged into the whole-module
    chunk chunking already produces for them, so this only adds an extra
    name the shared resolver can look up, pointing at the defining module
    itself rather than a new entity.

    Parameters
    ----------
    source_file: SourceFile
        Python source and its repository path.

    Returns
    -------
    dict[str, str]
        Each constant's bare name mapped to the qualified module that
        defines it, e.g. ``{"MODEL_CHOICES": "gui.model_catalog"}``.

    """

    try:
        tree = ast.parse(source_file.text)
    except SyntaxError:
        return {}

    module = _module_name(source_file.path)
    constants: dict[str, str] = {}
    for node in tree.body:
        name = _module_constant_name(node)
        if name is not None and _CONSTANT_NAME.match(name):
            constants[name] = module
    return constants


def _module_constant_name(node: ast.AST) -> str | None:
    """Return a top-level assignment's target name, or None if not a simple one.

    Only a single, plain ``NAME = ...`` or ``NAME: annotation = ...`` target
    qualifies -- tuple unpacking, attribute targets, and subscript targets
    have no single unambiguous name to register.

    Parameters
    ----------
    node: ast.AST
        A top-level module statement.

    Returns
    -------
    str | None
        The assigned name, or None when the statement is not a simple
        top-level assignment.

    """

    if isinstance(node, ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return None

    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id

    return None


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

    def visit_Name(self: "_Visitor", node: ast.Name) -> None:
        """Record a plain reference to a known module-level constant.

        Restricted to SCREAMING_SNAKE_CASE identifiers that already resolve
        unambiguously through ``known`` (populated from every file's
        ``extract_module_constants`` before relations are extracted), so
        this stays conservative -- an ordinary local variable or parameter
        essentially never collides with that pattern and a registered
        constant name at once.

        """

        if isinstance(node.ctx, ast.Load) and _CONSTANT_NAME.match(node.id):
            candidates = self.known.get(node.id, set())
            if len(candidates) == 1:
                target = next(iter(candidates))
                if target != self.parents[-1]:
                    self._add("references", target)

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
