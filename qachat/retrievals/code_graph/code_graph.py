"""Build a common code graph from natural source units.

Classes
-------
CodeEntity
    One named element found in source code.
CodeRelation
    One directed relationship between code entities.
CodeGraph
    Entities and relationships extracted from source.

Functions
---------
build_code_graph
    Reuse source units and add language-specific relationships.

"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from qachat.sources.sources import SourceFile, SourceUnit


__all__ = [
    "CodeEntity",
    "CodeRelation",
    "CodeGraph",
    "build_code_graph",
]

_EXTENSION_PACKAGE = "qachat.retrievals.code_graph.extensions"


@dataclass(frozen=True, slots=True)
class CodeEntity:
    """One named source-code element.

    Attributes
    ----------
    name: str
        Stable qualified name.
    kind: str
        Source-unit kind.
    path: pathlib.Path
        File containing the entity.

    """

    name: str
    kind: str
    path: Path


@dataclass(frozen=True, slots=True)
class CodeRelation:
    """One directed relationship between code entities.

    Attributes
    ----------
    source: str
        Qualified source entity.
    relation: str
        Relationship type.
    target: str
        Qualified or external target entity.

    """

    source: str
    relation: str
    target: str


@dataclass(frozen=True, slots=True)
class CodeGraph:
    """Entities and deterministic code relationships.

    Attributes
    ----------
    entities: tuple[CodeEntity, ...]
        Named elements found in source units.
    relations: tuple[CodeRelation, ...]
        Structural and language-specific relationships.

    """

    entities: tuple[CodeEntity, ...]
    relations: tuple[CodeRelation, ...]


def build_code_graph(
    files: list[SourceFile],
    units: list[SourceUnit],
) -> CodeGraph:
    """Build a code graph without relying on directory roles.

    Parameters
    ----------
    files: list[SourceFile]
        Source files already converted to text.
    units: list[SourceUnit]
        Natural units extracted from those files.

    Returns
    -------
    CodeGraph
        Stable entities and relationships.

    """

    if not files or not units:
        raise ValueError("Source files and source units are required.")

    entities: set[CodeEntity] = set()
    relations: set[CodeRelation] = set()
    known: dict[str, set[str]] = {}

    for source_file in files:
        module = _module_name(source_file.path)
        entities.add(CodeEntity(module, "source_file", source_file.path))
        _remember_name(known, module)

    for unit in units:
        _add_unit(
            unit,
            _module_name(unit.path),
            entities,
            relations,
            known,
        )

    extensions: dict[Path, object] = {}
    for source_file in files:
        suffix = source_file.path.suffix.lower().lstrip(".")
        if not suffix:
            continue
        module_name = f"{_EXTENSION_PACKAGE}.code_graph_{suffix}"
        try:
            extension = import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name == module_name:
                continue
            raise
        extensions[source_file.path] = extension

        extract_constants = getattr(extension, "extract_module_constants", None)
        if extract_constants is not None:
            for short_name, module in extract_constants(source_file).items():
                known.setdefault(short_name, set()).add(module)

    for source_file in files:
        extension = extensions.get(source_file.path)
        if extension is None:
            continue
        suffix = source_file.path.suffix.lower().lstrip(".")

        extracted = extension.extract_code_relations(source_file, known)
        if not isinstance(extracted, set) or any(
            not isinstance(item, CodeRelation) for item in extracted
        ):
            raise RuntimeError(f"Invalid code-graph extension: {suffix}.")
        relations.update(extracted)

    return CodeGraph(
        tuple(sorted(entities, key=lambda item: item.name)),
        tuple(
            sorted(
                relations,
                key=lambda item: (
                    item.source,
                    item.relation,
                    item.target,
                ),
            )
        ),
    )


def _add_unit(
    unit: SourceUnit,
    parent: str,
    entities: set[CodeEntity],
    relations: set[CodeRelation],
    known: dict[str, set[str]],
) -> None:
    """Add one natural unit and its children."""

    name = _unit_name(unit, parent)
    entities.add(CodeEntity(name, unit.kind, unit.path))
    _remember_name(known, name)
    if name != parent:
        relations.add(CodeRelation(parent, "contains", name))

    for child in unit.children:
        _add_unit(child, name, entities, relations, known)


def _unit_name(unit: SourceUnit, parent: str) -> str:
    """Return a qualified name for one source unit."""

    if unit.symbol is None:
        return parent
    context_kinds = (
        "python_class_context",
        "python_module_context",
        "matlab_class_context",
        "matlab_file_context",
    )
    if unit.kind in context_kinds or unit.kind.endswith("_block"):
        return parent
    if unit.kind in {"matlab_class", "python_class"} and (
        unit.symbol == parent
    ):
        return f"{parent}.{unit.symbol}"
    if unit.symbol == parent or unit.symbol.startswith(f"{parent}."):
        return unit.symbol
    if unit.symbol.startswith(f"{parent.split('.')[-1]}."):
        prefix = parent.rsplit(".", 1)[0]
        return f"{prefix}.{unit.symbol}"
    return f"{parent}.{unit.symbol}"


def _module_name(path: Path) -> str:
    """Return a module name from a relative file path."""

    return ".".join(path.with_suffix("").parts)


def _remember_name(known: dict[str, set[str]], name: str) -> None:
    """Record a qualified name by its final component."""

    known.setdefault(name.split(".")[-1], set()).add(name)


def _resolve_name(name: str, known: dict[str, set[str]]) -> str:
    """Resolve an unambiguous local name."""

    candidates = known.get(name.split(".")[-1], set())
    if len(candidates) == 1:
        return next(iter(candidates))

    nested = {candidate for candidate in candidates if candidate != name}
    return next(iter(nested)) if len(nested) == 1 else name
