"""Build a source-oriented graph of code entities and relationships.

Structural source units provide entities and containment relationships.
Optional suffix-specific extensions add imports, inheritance, calls, and
other references without executing the source. Names are derived from
repository-relative paths and available symbols. Resolution is heuristic:
external references and ambiguous names need not map to a unique local
entity, and not every runtime dependency is represented.

Key Features
------------

- **Structural entities**

    Collect modules, classes, functions, and nested source-unit symbols.

- **Heuristic relations**

    Combine containment with relationships supplied by supported language
    extensions.

- **Stable ordering**

    Deduplicate records and return a reproducibly sorted graph for fixed input.

Module Structure
----------------

- ``CodeEntity``

    One named source-code element.

- ``CodeRelation``

    One directed relationship between code entities.

- ``CodeGraph``

    Entities and deterministic code relationships.

- ``build_code_graph``

    Build a code graph without relying on directory roles.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import kilin
>>> report = kilin.scan_repository("/path/to/repository", extensions=[".py"])
>>> files = kilin.read_repository_files(report)
>>> units = kilin.extract_source_units(files)
>>> graph = kilin.build_code_graph(files, units)

"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from kilin.sources import source_kinds as kinds
from kilin.sources.sources import SourceFile, SourceUnit

__all__ = [
    "CodeEntity",
    "CodeRelation",
    "CodeGraph",
    "build_code_graph",
]

_EXTENSION_PACKAGE = "kilin.retrievals.code_graph.extensions"
# File types written in the same language share their names: a notebook's
# functions are Python functions.
_SAME_LANGUAGE = {".ipynb": ".py"}


class _ReachableNames(dict[str, set[str]]):
    """Names one file can call, keyed by their last part.

    The mapping holds the names of the file's own language and of languages
    it calls by plain name. ``prefixed`` holds, for each prefix the language
    writes before a call into another one (MATLAB's ``py``), the names of
    that language.

    """

    __slots__ = ("prefixed",)

    def __init__(self) -> None:
        """Start with no names and no other languages.

        """

        super().__init__()
        self.prefixed: dict[str, dict[str, set[str]]] = {}


def _language(path: Path) -> str:
    """Return the file type that stands for a file's language.

    Parameters
    ----------
    path: pathlib.Path
        Repository-relative source path.

    Returns
    -------
    str
        Lowercase suffix, with notebooks counted as Python.

    """

    suffix = path.suffix.lower()
    return _SAME_LANGUAGE.get(suffix, suffix)


def _reachable_names(
    known: dict[str, dict[str, set[str]]],
    language: str,
    calls_into: dict[str, str],
) -> _ReachableNames:
    """Return the names a file of one language can call.

    Parameters
    ----------
    known: dict[str, dict[str, set[str]]]
        Known names of each language.
    language: str
        File type of the calling file's language.
    calls_into: dict[str, str]
        The extension's ``CALLS_INTO``: for each prefix written before a
        call into another language, that language's file type; an empty
        prefix means plain calls.

    Returns
    -------
    _ReachableNames
        Copies of the names, so resolving never changes ``known``.

    """

    names = _ReachableNames()
    plain = [language] + [
        other for prefix, other in calls_into.items() if not prefix
    ]
    for other in plain:
        for short_name, qualified in known.get(other, {}).items():
            names.setdefault(short_name, set()).update(qualified)
    for prefix, other in calls_into.items():
        if prefix:
            names.prefixed[prefix] = known.get(other, {})
    return names


@dataclass(frozen=True, slots=True)
class CodeEntity:
    """One named source-code element.

    Parameters
    ----------
    name: str
        Stable qualified name.
    kind: str
        Source-unit kind.
    path: pathlib.Path
        File containing the entity.

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

    Parameters
    ----------
    source: str
        Qualified source entity.
    relation: str
        Relationship type.
    target: str
        Qualified or external target entity.

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

    Parameters
    ----------
    entities: tuple[CodeEntity, ...]
        Named elements found in source units.
    relations: tuple[CodeRelation, ...]
        Structural and language-specific relationships.

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

    Raises
    ------
    ValueError
        If the source-file or source-unit input is empty.
    RuntimeError
        If a language hook returns an invalid relation collection.

    Notes
    -----
    Language hooks inspect text without executing it. Missing hooks leave
    structural entities/containment only. Relations may target external or
    unresolved names not represented by graph.entities. Fixed inputs produce
    sorted deduplicated records; this is not a complete runtime call graph.
    Entities are ordered by name, kind, and path because distinct source
    entities can share the same qualified name.

    A plain call is matched only to names of the caller's own language. A
    language hook that calls into another language declares it in
    ``CALLS_INTO``, a mapping from the prefix it writes before such a call
    to that language's file type: MATLAB's ``{"py": ".py"}`` links
    ``py.helpers.smooth(x)`` to the Python function ``helpers.smooth``, and
    HTML's ``{"": ".js"}`` lets plain handler calls reach JavaScript.

    """

    if not files or not units:
        raise ValueError("Source files and source units are required.")

    entities: set[CodeEntity] = set()
    relations: set[CodeRelation] = set()
    # Names are kept per language, so a call in one language is never
    # matched to a same-named function in another by accident.
    known: dict[str, dict[str, set[str]]] = {}

    for source_file in files:
        module = _module_name(source_file.path)
        entities.add(CodeEntity(module, "source_file", source_file.path))
        _remember_name(
            known.setdefault(_language(source_file.path), {}), module
        )

    for unit in units:
        _add_unit(
            unit,
            _module_name(unit.path),
            entities,
            relations,
            known.setdefault(_language(unit.path), {}),
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

        extract_constants = getattr(
            extension, "extract_module_constants", None
        )
        if extract_constants is not None:
            names = known.setdefault(_language(source_file.path), {})
            for short_name, module in extract_constants(source_file).items():
                names.setdefault(short_name, set()).add(module)

    for source_file in files:
        extension = extensions.get(source_file.path)
        if extension is None:
            continue
        suffix = source_file.path.suffix.lower().lstrip(".")

        reachable = _reachable_names(
            known,
            _language(source_file.path),
            getattr(extension, "CALLS_INTO", {}),
        )
        extracted = extension.extract_code_relations(source_file, reachable)
        if not isinstance(extracted, set) or any(
            not isinstance(item, CodeRelation) for item in extracted
        ):
            raise RuntimeError(f"Invalid code-graph extension: {suffix}.")
        relations.update(extracted)

    # Different source entities can share a name; all identity fields must
    # participate in sorting to keep graph fingerprints reproducible.
    return CodeGraph(
        tuple(
            sorted(
                entities,
                key=lambda item: (
                    item.name,
                    item.kind,
                    item.path.as_posix(),
                ),
            )
        ),
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
    """Add one natural unit and its children.

    Parameters
    ----------
    unit: SourceUnit
        Source unit whose symbol and children are incorporated into the graph.
    parent: str
        Qualified containing module or entity name.
    entities: set[CodeEntity]
        Mutable set accumulating deduplicated CodeEntity records.
    relations: set[CodeRelation]
        Mutable set accumulating deduplicated CodeRelation records.
    known: dict[str, set[str]]
        Mapping from recognized names to candidate qualified entity names.

    """

    name = _unit_name(unit, parent)
    entities.add(CodeEntity(name, unit.kind, unit.path))
    _remember_name(known, name)
    if name != parent:
        relations.add(CodeRelation(parent, "contains", name))

    for child in unit.children:
        _add_unit(child, name, entities, relations, known)


def _unit_name(unit: SourceUnit, parent: str) -> str:
    """Return a qualified name for one source unit.

    Parameters
    ----------
    unit: SourceUnit
        Source unit whose symbol and children are incorporated into the graph.
    parent: str
        Qualified containing module or entity name.

    Returns
    -------
    str
        A qualified name for one source unit.

    """

    if unit.symbol is None:
        return parent
    if unit.kind in kinds.CONTEXT_KINDS or unit.kind.endswith("_block"):
        return parent
    if unit.kind in kinds.CLASS_KINDS and (unit.symbol == parent):
        return f"{parent}.{unit.symbol}"
    if unit.symbol == parent or unit.symbol.startswith(f"{parent}."):
        return unit.symbol
    if unit.symbol.startswith(f"{parent.split('.')[-1]}."):
        prefix = parent.rsplit(".", 1)[0]
        return f"{prefix}.{unit.symbol}"
    return f"{parent}.{unit.symbol}"


def _module_name(path: Path) -> str:
    """Return a module name from a relative file path.

    Parameters
    ----------
    path: pathlib.Path
        Repository-relative source-file path.

    Returns
    -------
    str
        A module name from a relative file path.

    Notes
    -----
    The final suffix is removed and remaining path components are joined with
    periods. Package __init__ filenames are not collapsed to their parent
    package here.

    """

    return ".".join(path.with_suffix("").parts)


def _remember_name(known: dict[str, set[str]], name: str) -> None:
    """Record a qualified name by its final component.

    Parameters
    ----------
    known: dict[str, set[str]]
        Mapping from recognized names to candidate qualified entity names.
    name: str
        Name to register or resolve using the known-entity map.

    """

    known.setdefault(name.split(".")[-1], set()).add(name)


def _resolve_name(
    name: str,
    known: dict[str, set[str]],
    scope: str | None = None,
    module: str | None = None,
) -> str:
    """Resolve a referenced name to one known entity when it is clear.

    Parameters
    ----------
    name: str
        Name to register or resolve using the known-entity map.
    known: dict[str, set[str]]
        Mapping from recognized names to candidate qualified entity names.
    scope: str | None, optional
        Qualified name of the entity containing the reference, such as
        ``pkg.module.Class.method``. Defaults to ``None`` (no scope rule).
    module: str | None, optional
        Qualified name of the module containing the reference. Defaults to
        ``None`` (no scope rule).

    Returns
    -------
    str
        A unique known qualified name when resolvable, otherwise the original
        reference string.

    Notes
    -----
    A reference that starts with a prefix the language uses to call another
    one (``py.helpers.smooth`` in MATLAB) is resolved among that language's
    names only. A qualified cross-language reference must match its whole
    name or a unique qualified suffix; a same-leaf fallback is not enough.
    Otherwise candidates share the reference's last component,
    and a file named after its function counts as that function. One
    candidate is taken. Otherwise, in order:

    1. a dotted reference (``sources_py.extract_source_units``) takes the
       only candidate that is it or ends with it;
    2. with ``scope`` and ``module``, as Python looks names up: a plain
       name (``_candidate``) takes the only candidate defined at the top
       level of the same module, and a ``self.``/``cls.`` attribute the
       candidate defined in the innermost class enclosing the reference;
    3. the only candidate that differs from the reference itself.

    Anything still ambiguous is kept as written rather than guessed with
    runtime import or dispatch semantics.

    """

    first, _, rest = name.partition(".")
    other = getattr(known, "prefixed", {}).get(first) if rest else None
    if other is not None:
        if "." in rest:
            # A qualified cross-language call must not fall back to the
            # only function with the same leaf in an unrelated module.
            leaf = rest.rsplit(".", 1)[-1]
            candidates = {
                candidate
                for candidate in other.get(leaf, set())
                if f"{candidate}.{leaf}" not in other.get(leaf, set())
            }
            qualified = {
                candidate for candidate in candidates
                if candidate == rest or candidate.endswith(f".{rest}")
            }
            return next(iter(qualified)) if len(qualified) == 1 else name
        resolved = _resolve_name(rest, other)
        found = resolved in other.get(resolved.split(".")[-1], set())
        return resolved if found else name

    leaf = name.split(".")[-1]
    # A file named after the function it defines (MATLAB's helper.m): the
    # call means the function, not the file.
    candidates = {
        candidate
        for candidate in known.get(leaf, set())
        if f"{candidate}.{leaf}" not in known.get(leaf, set())
    }
    if len(candidates) == 1:
        return next(iter(candidates))

    parts = name.split(".")
    if len(parts) > 1 and parts[0] not in {"self", "cls"}:
        suffixed = {
            candidate
            for candidate in candidates
            if candidate == name or candidate.endswith(f".{name}")
        }
        if len(suffixed) == 1:
            return next(iter(suffixed))

    if scope and module:
        parents = {
            candidate: candidate.rsplit(".", 1)[0]
            for candidate in candidates
            if "." in candidate
        }
        if len(parts) == 1:
            # Class attributes are not visible to plain names in methods.
            top_level = [
                candidate
                for candidate, parent in parents.items()
                if parent == module
            ]
            if len(top_level) == 1:
                return top_level[0]
        elif len(parts) == 2 and parts[0] in {"self", "cls"}:
            members = [
                candidate
                for candidate, parent in parents.items()
                if parent.startswith(f"{module}.")
                and scope.startswith(f"{parent}.")
            ]
            if members:
                depth = max(candidate.count(".") for candidate in members)
                innermost = [c for c in members if c.count(".") == depth]
                if len(innermost) == 1:
                    return innermost[0]

    nested = {candidate for candidate in candidates if candidate != name}
    return next(iter(nested)) if len(nested) == 1 else name
