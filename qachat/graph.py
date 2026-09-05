"""Build deterministic repository facts from source code.

The graph reads source files and natural source units. It never depends on
tokenizer chunks, embedding models, or an LLM. Exact facts therefore remain
stable when retrieval settings change.

"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from qachat.source import SourceFile, SourceUnit, extract_source_units

Language = Literal["python", "matlab"]
NodeKind = Literal["module", "file", "class", "function", "method"]
RelationKind = Literal[
    "contains",
    "imports",
    "inherits",
    "calls",
    "tests",
    "references",
]
Certainty = Literal["exact", "ambiguous", "unresolved"]
Direction = Literal["incoming", "outgoing", "both"]

DEFINITION_KINDS = {"class", "function", "method"}
MATLAB_UNIT_KINDS: dict[str, NodeKind] = {
    "matlab_class": "class",
    "matlab_constructor": "method",
    "matlab_function": "function",
    "matlab_method": "method",
}
MATLAB_CLASS = re.compile(
    r"^\s*classdef(?:\s*\([^)]*\))?\s+"
    r"(?P<name>[A-Za-z]\w*)"
    r"(?:\s*<\s*(?P<bases>[^%]+))?"
)
MATLAB_IMPORT = re.compile(
    r"^\s*import\s+(?P<names>[A-Za-z]\w*(?:\.[A-Za-z*]\w*)*)"
)
MATLAB_CALL = re.compile(
    r"(?<![.\w])(?P<name>[A-Za-z]\w*(?:\.[A-Za-z]\w*)*)\s*\("
)
MATLAB_CONTROL_WORDS = {
    "arguments",
    "case",
    "catch",
    "classdef",
    "else",
    "elseif",
    "end",
    "enumeration",
    "events",
    "for",
    "function",
    "if",
    "methods",
    "otherwise",
    "parfor",
    "properties",
    "spmd",
    "switch",
    "try",
    "while",
}


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """Exact source range supporting one graph fact.

    Attributes
    ----------
    path: pathlib.Path
        File path relative to the repository root.
    start_line: int
        First one-based source line.
    end_line: int
        Last one-based source line.

    """

    path: Path
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One definition found in repository source.

    Attributes
    ----------
    identifier: str
        Stable identifier within one repository graph.
    name: str
        Short definition name.
    symbol: str
        Definition name relative to its file.
    qualified_name: str
        Name including its Python module or MATLAB file.
    kind: NodeKind
        Definition kind.
    language: Language
        Source language.
    provenance: SourceLocation
        Exact source range supporting the definition.

    """

    identifier: str
    name: str
    symbol: str
    qualified_name: str
    kind: NodeKind
    language: Language
    provenance: SourceLocation


@dataclass(frozen=True, slots=True)
class GraphRelation:
    """One directional relationship supported by source code.

    Attributes
    ----------
    source_identifier: str
        Identifier of the source node.
    target_identifier: str | None
        Resolved repository node, when one can be identified.
    target_name: str
        Name written in source code.
    kind: RelationKind
        Relationship kind.
    certainty: Certainty
        Whether the target is exact, ambiguous, or unresolved.
    reason: str
        Plain explanation of the certainty.
    provenance: SourceLocation
        Exact source range supporting the relationship.

    """

    source_identifier: str
    target_identifier: str | None
    target_name: str
    kind: RelationKind
    certainty: Certainty
    reason: str
    provenance: SourceLocation


@dataclass(frozen=True, slots=True)
class RepositoryGraph:
    """Definitions and relationships extracted from repository source.

    Attributes
    ----------
    nodes: tuple[GraphNode, ...]
        Definitions in deterministic source order.
    relations: tuple[GraphRelation, ...]
        Relationships in deterministic source order.

    """

    nodes: tuple[GraphNode, ...]
    relations: tuple[GraphRelation, ...]

    def list_nodes(
        self: RepositoryGraph,
        *,
        kind: NodeKind | None = None,
        language: Language | None = None,
    ) -> tuple[GraphNode, ...]:
        """Return nodes matching optional exact filters.

        Parameters
        ----------
        kind: NodeKind | None, optional
            Definition kind to keep.
        language: Language | None, optional
            Source language to keep.

        Returns
        -------
        tuple[GraphNode, ...]
            Matching nodes in deterministic order.

        """

        return tuple(
            node
            for node in self.nodes
            if (kind is None or node.kind == kind)
            and (language is None or node.language == language)
        )

    def count_nodes(
        self: RepositoryGraph,
        *,
        kind: NodeKind | None = None,
        language: Language | None = None,
    ) -> int:
        """Count nodes matching optional exact filters.

        Parameters
        ----------
        kind: NodeKind | None, optional
            Definition kind to count.
        language: Language | None, optional
            Source language to count.

        Returns
        -------
        int
            Exact number of matching definitions.

        """

        return len(self.list_nodes(kind=kind, language=language))

    def find_definitions(
        self: RepositoryGraph,
        name: str,
        *,
        kind: NodeKind | None = None,
        language: Language | None = None,
    ) -> tuple[GraphNode, ...]:
        """Find definitions by short, file-local, or qualified name.

        Parameters
        ----------
        name: str
            Exact name to find.
        kind: NodeKind | None, optional
            Definition kind to keep.
        language: Language | None, optional
            Source language to search.

        Returns
        -------
        tuple[GraphNode, ...]
            Matching class, function, and method definitions.

        """

        return tuple(
            node
            for node in self.nodes
            if node.kind in DEFINITION_KINDS
            and (kind is None or node.kind == kind)
            and (language is None or node.language == language)
            and name in {node.name, node.symbol, node.qualified_name}
        )

    def relations_for(
        self: RepositoryGraph,
        identifier: str,
        *,
        kind: RelationKind | None = None,
        direction: Direction = "both",
    ) -> tuple[GraphRelation, ...]:
        """Return relations connected to one node.

        Parameters
        ----------
        identifier: str
            Graph node identifier.
        kind: RelationKind | None, optional
            Relationship kind to keep.
        direction: Direction, optional
            Keep incoming, outgoing, or both directions.

        Returns
        -------
        tuple[GraphRelation, ...]
            Matching relations, including uncertain references.

        """

        return tuple(
            relation
            for relation in self.relations
            if (
                direction == "both"
                and (
                    relation.source_identifier == identifier
                    or relation.target_identifier == identifier
                )
                or direction == "outgoing"
                and relation.source_identifier == identifier
                or direction == "incoming"
                and relation.target_identifier == identifier
            )
            and (kind is None or relation.kind == kind)
        )

    def neighbors(
        self: RepositoryGraph,
        identifier: str,
        *,
        kind: RelationKind | None = None,
    ) -> tuple[GraphNode, ...]:
        """Return exactly resolved nodes connected to one node.

        Parameters
        ----------
        identifier: str
            Graph node identifier.
        kind: RelationKind | None, optional
            Relationship kind to keep.

        Returns
        -------
        tuple[GraphNode, ...]
            Connected repository definitions without duplicates.

        """

        neighbor_identifiers: set[str] = set()
        for relation in self.relations_for(identifier, kind=kind):
            if relation.certainty != "exact":
                continue
            if relation.source_identifier == identifier:
                target = relation.target_identifier
                if target is not None:
                    neighbor_identifiers.add(target)
            elif relation.target_identifier == identifier:
                neighbor_identifiers.add(relation.source_identifier)

        return tuple(
            node
            for node in self.nodes
            if node.identifier in neighbor_identifiers
        )


@dataclass(frozen=True, slots=True)
class _PythonSource:
    """Parsed Python source and its definition identifiers."""

    source_file: SourceFile
    tree: ast.Module
    module_node: GraphNode
    identifiers: dict[int, str]


class _GraphBuilder:
    """Collect nodes and relations before deterministic sorting."""

    def __init__(self: _GraphBuilder) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.relations: set[GraphRelation] = set()

    def add_node(self: _GraphBuilder, node: GraphNode) -> None:
        """Add one node, keeping the first identical identifier."""

        self.nodes.setdefault(node.identifier, node)

    def add_relation(self: _GraphBuilder, relation: GraphRelation) -> None:
        """Add one relation without duplicates."""

        self.relations.add(relation)

    def finish(self: _GraphBuilder) -> RepositoryGraph:
        """Return an immutable graph in deterministic source order."""

        nodes = tuple(sorted(self.nodes.values(), key=_node_sort_key))
        relations = tuple(sorted(self.relations, key=_relation_sort_key))
        return RepositoryGraph(nodes, relations)


def build_repository_graph(
    source_files: list[SourceFile],
    source_units: list[SourceUnit] | None = None,
) -> RepositoryGraph:
    """Build exact and explicitly uncertain repository relationships.

    Parameters
    ----------
    source_files: list[SourceFile]
        Repository source files. Only Python and MATLAB are inspected.
    source_units: list[SourceUnit] | None, optional
        Natural units for the same source files. They are extracted when
        omitted. MATLAB definition ranges come from these units when present.

    Returns
    -------
    RepositoryGraph
        Deterministic graph independent of chunks, embeddings, and an LLM.

    """

    ordered_files = sorted(source_files, key=lambda item: item.path.as_posix())
    units = (
        extract_source_units(ordered_files)
        if source_units is None
        else source_units
    )
    units_by_path = _units_by_path(units)
    builder = _GraphBuilder()
    python_sources: list[_PythonSource] = []
    matlab_files: list[SourceFile] = []

    for source_file in ordered_files:
        suffix = source_file.path.suffix.lower()
        if suffix == ".py":
            parsed = _add_python_nodes(builder, source_file)
            if parsed is not None:
                python_sources.append(parsed)
        elif suffix == ".m":
            _add_matlab_nodes(
                builder,
                source_file,
                units_by_path.get(source_file.path, ()),
            )
            matlab_files.append(source_file)

    index = _NodeIndex(tuple(builder.nodes.values()))
    for parsed in python_sources:
        _PythonRelationVisitor(builder, index, parsed).visit(parsed.tree)

    for source_file in matlab_files:
        _add_matlab_relations(builder, index, source_file)

    return builder.finish()


def _units_by_path(
    source_units: list[SourceUnit],
) -> dict[Path, tuple[SourceUnit, ...]]:
    """Group flattened source units by file path."""

    grouped: dict[Path, list[SourceUnit]] = {}
    stack = list(reversed(source_units))
    while stack:
        unit = stack.pop()
        grouped.setdefault(unit.path, []).append(unit)
        stack.extend(reversed(unit.children))

    return {path: tuple(units) for path, units in grouped.items()}


def _module_symbol(path: Path) -> str:
    """Return a dotted module-like name for a relative source path."""

    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or path.stem


def _identifier(
    language: Language,
    path: Path,
    kind: NodeKind,
    symbol: str,
    start_line: int,
) -> str:
    """Return a deterministic graph node identifier."""

    return f"{language}:{path.as_posix()}:{kind}:{symbol}:{start_line}"


def _new_node(
    *,
    language: Language,
    path: Path,
    kind: NodeKind,
    symbol: str,
    module_symbol: str,
    start_line: int,
    end_line: int,
) -> GraphNode:
    """Build one graph node from a source range."""

    name = symbol.rsplit(".", 1)[-1]
    qualified_name = (
        module_symbol
        if kind in {"module", "file"}
        else f"{module_symbol}.{symbol}"
    )
    return GraphNode(
        identifier=_identifier(language, path, kind, symbol, start_line),
        name=name,
        symbol=symbol,
        qualified_name=qualified_name,
        kind=kind,
        language=language,
        provenance=SourceLocation(path, start_line, end_line),
    )


def _node_sort_key(node: GraphNode) -> tuple[object, ...]:
    """Return deterministic source ordering for nodes."""

    return (
        node.provenance.path.as_posix(),
        node.provenance.start_line,
        node.provenance.end_line,
        node.kind,
        node.symbol,
    )


def _relation_sort_key(relation: GraphRelation) -> tuple[object, ...]:
    """Return deterministic source ordering for relationships."""

    return (
        relation.provenance.path.as_posix(),
        relation.provenance.start_line,
        relation.kind,
        relation.source_identifier,
        relation.target_name,
        relation.target_identifier or "",
        relation.certainty,
    )


def _add_python_nodes(
    builder: _GraphBuilder,
    source_file: SourceFile,
) -> _PythonSource | None:
    """Parse one Python file and add its definitions."""

    module_symbol = _module_symbol(source_file.path)
    line_count = max(1, len(source_file.text.splitlines()))
    module_node = _new_node(
        language="python",
        path=source_file.path,
        kind="module",
        symbol=module_symbol,
        module_symbol=module_symbol,
        start_line=1,
        end_line=line_count,
    )
    builder.add_node(module_node)

    try:
        tree = ast.parse(source_file.text)
    except SyntaxError:
        return None

    identifiers: dict[int, str] = {}
    _add_python_definition_nodes(
        builder,
        source_file.path,
        module_symbol,
        tree.body,
        parent_node=module_node,
        parent_symbol="",
        parent_kind="module",
        identifiers=identifiers,
    )
    return _PythonSource(source_file, tree, module_node, identifiers)


def _add_python_definition_nodes(
    builder: _GraphBuilder,
    path: Path,
    module_symbol: str,
    statements: list[ast.stmt],
    *,
    parent_node: GraphNode,
    parent_symbol: str,
    parent_kind: NodeKind,
    identifiers: dict[int, str],
) -> None:
    """Recursively add Python class, function, and method definitions."""

    for statement in statements:
        if isinstance(statement, ast.ClassDef):
            kind: NodeKind = "class"
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "method" if parent_kind == "class" else "function"
        else:
            continue

        symbol = (
            f"{parent_symbol}.{statement.name}"
            if parent_symbol
            else statement.name
        )
        end_line = getattr(statement, "end_lineno", statement.lineno)
        node = _new_node(
            language="python",
            path=path,
            kind=kind,
            symbol=symbol,
            module_symbol=module_symbol,
            start_line=statement.lineno,
            end_line=end_line,
        )
        builder.add_node(node)
        identifiers[id(statement)] = node.identifier
        builder.add_relation(
            GraphRelation(
                source_identifier=parent_node.identifier,
                target_identifier=node.identifier,
                target_name=node.symbol,
                kind="contains",
                certainty="exact",
                reason=(
                    "The definition is directly nested in this source scope."
                ),
                provenance=node.provenance,
            )
        )
        _add_python_definition_nodes(
            builder,
            path,
            module_symbol,
            statement.body,
            parent_node=node,
            parent_symbol=symbol,
            parent_kind=kind,
            identifiers=identifiers,
        )


class _NodeIndex:
    """Small lookup index used while resolving source relationships."""

    def __init__(self: _NodeIndex, nodes: tuple[GraphNode, ...]) -> None:
        self.by_identifier = {node.identifier: node for node in nodes}
        self.by_qualified: dict[str, list[GraphNode]] = {}
        self.by_name: dict[str, list[GraphNode]] = {}
        self.by_path: dict[Path, list[GraphNode]] = {}
        for node in nodes:
            self.by_qualified.setdefault(node.qualified_name, []).append(node)
            self.by_name.setdefault(node.name, []).append(node)
            self.by_path.setdefault(node.provenance.path, []).append(node)

    def qualified(self: _NodeIndex, name: str) -> tuple[GraphNode, ...]:
        """Return definitions with one exact qualified name."""

        return tuple(self.by_qualified.get(name, ()))


def _dotted_name(node: ast.AST) -> str | None:
    """Return a direct dotted Python name without evaluating code."""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _dotted_name(node.value)
        return f"{value}.{node.attr}" if value is not None else None
    return None


def _python_package(module: str, path: Path, level: int) -> str:
    """Resolve the package portion of a relative Python import."""

    if level == 0:
        return ""

    parts = module.split(".")
    package = parts if path.name == "__init__.py" else parts[:-1]
    remove = level - 1
    if remove >= len(package):
        return ""
    return ".".join(package[: len(package) - remove])


class _PythonRelationVisitor(ast.NodeVisitor):
    """Add statically resolvable relationships from one Python tree."""

    def __init__(
        self: _PythonRelationVisitor,
        builder: _GraphBuilder,
        index: _NodeIndex,
        parsed: _PythonSource,
    ) -> None:
        self.builder = builder
        self.index = index
        self.parsed = parsed
        self.source_identifiers = [parsed.module_node.identifier]
        self.symbols = [""]
        self.classes: list[str | None] = [None]
        self.aliases: list[dict[str, str]] = [
            self._top_level_aliases(parsed.tree)
        ]

    @property
    def source_identifier(self: _PythonRelationVisitor) -> str:
        """Return the current definition identifier."""

        return self.source_identifiers[-1]

    def _top_level_aliases(
        self: _PythonRelationVisitor,
        tree: ast.Module,
    ) -> dict[str, str]:
        """Collect aliases visible to Python function bodies."""

        aliases: dict[str, str] = {}
        for statement in tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                aliases.update(self._import_aliases(statement))
        return aliases

    def _import_aliases(
        self: _PythonRelationVisitor,
        node: ast.Import | ast.ImportFrom,
    ) -> dict[str, str]:
        """Return names bound by one Python import statement."""

        result: dict[str, str] = {}
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                result[bound] = alias.name if alias.asname else bound
            return result

        package = _python_package(
            self.parsed.module_node.symbol,
            self.parsed.source_file.path,
            node.level,
        )
        module = ".".join(part for part in (package, node.module) if part)
        for alias in node.names:
            if alias.name == "*":
                continue
            target = ".".join(part for part in (module, alias.name) if part)
            result[alias.asname or alias.name] = target
        return result

    def visit_Import(self: _PythonRelationVisitor, node: ast.Import) -> None:
        """Record direct Python module imports."""

        self.aliases[-1].update(self._import_aliases(node))
        for alias in node.names:
            self._add_named_relation("imports", alias.name, node.lineno)

    def visit_ImportFrom(
        self: _PythonRelationVisitor,
        node: ast.ImportFrom,
    ) -> None:
        """Record direct Python from-imports."""

        aliases = self._import_aliases(node)
        self.aliases[-1].update(aliases)
        for alias in node.names:
            if alias.name == "*":
                package = _python_package(
                    self.parsed.module_node.symbol,
                    self.parsed.source_file.path,
                    node.level,
                )
                target = ".".join(
                    part for part in (package, node.module or "") if part
                )
            else:
                target = aliases[alias.asname or alias.name]
            self._add_named_relation("imports", target, node.lineno)

    def visit_ClassDef(
        self: _PythonRelationVisitor,
        node: ast.ClassDef,
    ) -> None:
        """Record inheritance and visit a Python class body."""

        identifier = self.parsed.identifiers[id(node)]
        symbol = self.index.by_identifier[identifier].symbol
        aliases = dict(self.aliases[-1])
        for base in node.bases:
            name = _dotted_name(base)
            if name is not None:
                self._add_python_reference(
                    source_identifier=identifier,
                    kind="inherits",
                    written_name=name,
                    line=node.lineno,
                    aliases=aliases,
                    symbol=symbol,
                    class_symbol=symbol,
                    allow_unresolved=True,
                )

        self.source_identifiers.append(identifier)
        self.symbols.append(symbol)
        self.classes.append(symbol)
        self.aliases.append(aliases)
        for statement in node.body:
            self.visit(statement)
        self.aliases.pop()
        self.classes.pop()
        self.symbols.pop()
        self.source_identifiers.pop()

    def visit_FunctionDef(
        self: _PythonRelationVisitor,
        node: ast.FunctionDef,
    ) -> None:
        """Visit a synchronous Python function body."""

        self._visit_function(node)

    def visit_AsyncFunctionDef(
        self: _PythonRelationVisitor,
        node: ast.AsyncFunctionDef,
    ) -> None:
        """Visit an asynchronous Python function body."""

        self._visit_function(node)

    def _visit_function(
        self: _PythonRelationVisitor,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """Visit one Python function with its own local imports."""

        identifier = self.parsed.identifiers[id(node)]
        symbol = self.index.by_identifier[identifier].symbol
        self.source_identifiers.append(identifier)
        self.symbols.append(symbol)
        self.classes.append(self.classes[-1])
        self.aliases.append(dict(self.aliases[-1]))
        for statement in node.body:
            self.visit(statement)
        self.aliases.pop()
        self.classes.pop()
        self.symbols.pop()
        self.source_identifiers.pop()

    def visit_Call(self: _PythonRelationVisitor, node: ast.Call) -> None:
        """Record a call only when its target resolves from source."""

        name = _dotted_name(node.func)
        if name is not None:
            self._add_python_reference(
                source_identifier=self.source_identifier,
                kind="calls",
                written_name=name,
                line=node.lineno,
                aliases=self.aliases[-1],
                symbol=self.symbols[-1],
                class_symbol=self.classes[-1],
                allow_unresolved=False,
            )
        self.generic_visit(node)

    def _add_named_relation(
        self: _PythonRelationVisitor,
        kind: RelationKind,
        target_name: str,
        line: int,
    ) -> None:
        """Add an import whose statement itself is exact."""

        candidates = self.index.qualified(target_name)
        target = candidates[0].identifier if len(candidates) == 1 else None
        self.builder.add_relation(
            GraphRelation(
                source_identifier=self.source_identifier,
                target_identifier=target,
                target_name=target_name,
                kind=kind,
                certainty="exact" if len(candidates) <= 1 else "ambiguous",
                reason=(
                    "The import statement directly names this target."
                    if len(candidates) <= 1
                    else "Several selected definitions have this name."
                ),
                provenance=SourceLocation(
                    self.parsed.source_file.path,
                    line,
                    line,
                ),
            )
        )

    def _add_python_reference(
        self: _PythonRelationVisitor,
        *,
        source_identifier: str,
        kind: RelationKind,
        written_name: str,
        line: int,
        aliases: dict[str, str],
        symbol: str,
        class_symbol: str | None,
        allow_unresolved: bool,
    ) -> None:
        """Resolve and record one Python base or direct call expression."""

        candidates = _python_candidates(
            self.index,
            self.parsed.module_node.symbol,
            written_name,
            aliases,
            symbol,
            class_symbol,
        )
        if not candidates and not allow_unresolved:
            return

        location = SourceLocation(self.parsed.source_file.path, line, line)
        if len(candidates) == 1:
            target = candidates[0]
            relation = GraphRelation(
                source_identifier=source_identifier,
                target_identifier=target.identifier,
                target_name=written_name,
                kind=kind,
                certainty="exact",
                reason="The direct source name resolves to one definition.",
                provenance=location,
            )
            self.builder.add_relation(relation)
            if kind == "calls" and _is_test_node(
                self.index.by_identifier[source_identifier]
            ):
                self.builder.add_relation(
                    GraphRelation(
                        source_identifier=source_identifier,
                        target_identifier=target.identifier,
                        target_name=written_name,
                        kind="tests",
                        certainty="exact",
                        reason=(
                            "A test definition directly calls this target."
                        ),
                        provenance=location,
                    )
                )
            return

        if not candidates:
            self.builder.add_relation(
                GraphRelation(
                    source_identifier=source_identifier,
                    target_identifier=None,
                    target_name=written_name,
                    kind=kind,
                    certainty="unresolved",
                    reason=(
                        "The source names a base, but no selected definition "
                        "matches it."
                    ),
                    provenance=location,
                )
            )
            return

        for target in candidates:
            self.builder.add_relation(
                GraphRelation(
                    source_identifier=source_identifier,
                    target_identifier=target.identifier,
                    target_name=written_name,
                    kind=kind,
                    certainty="ambiguous",
                    reason=(
                        "Several selected definitions could match this name."
                    ),
                    provenance=location,
                )
            )


def _python_candidates(
    index: _NodeIndex,
    module: str,
    written_name: str,
    aliases: dict[str, str],
    symbol: str,
    class_symbol: str | None,
) -> tuple[GraphNode, ...]:
    """Return repository definitions safely named by a Python expression."""

    parts = written_name.split(".")
    names: list[str] = []
    if parts[0] in {"self", "cls"} and class_symbol is not None:
        tail = ".".join(parts[1:])
        names.append(f"{module}.{class_symbol}.{tail}")
    elif parts[0] in aliases:
        imported = aliases[parts[0]]
        tail = ".".join(parts[1:])
        names.append(".".join(part for part in (imported, tail) if part))
    else:
        if len(parts) > 1:
            names.extend((f"{module}.{written_name}", written_name))
        else:
            scope_parts = symbol.split(".")[:-1]
            while scope_parts:
                names.append(
                    f"{module}.{'.'.join(scope_parts)}.{written_name}"
                )
                scope_parts.pop()
            if class_symbol is not None:
                names.append(f"{module}.{class_symbol}.{written_name}")
            names.append(f"{module}.{written_name}")

    candidates: dict[str, GraphNode] = {}
    for name in names:
        for node in index.qualified(name):
            if node.kind in DEFINITION_KINDS:
                candidates[node.identifier] = node
        if candidates:
            break
    return tuple(sorted(candidates.values(), key=_node_sort_key))


def _is_test_node(node: GraphNode) -> bool:
    """Return whether a Python definition explicitly declares a test."""

    return node.kind in {"function", "method"} and node.name.startswith(
        "test_"
    )


def _add_matlab_nodes(
    builder: _GraphBuilder,
    source_file: SourceFile,
    source_units: tuple[SourceUnit, ...],
) -> None:
    """Add MATLAB definitions using natural source units when available."""

    module_symbol = _module_symbol(source_file.path)
    line_count = max(1, len(source_file.text.splitlines()))
    file_node = _new_node(
        language="matlab",
        path=source_file.path,
        kind="file",
        symbol=module_symbol,
        module_symbol=module_symbol,
        start_line=1,
        end_line=line_count,
    )
    builder.add_node(file_node)

    definition_units = [
        unit for unit in source_units if unit.kind in MATLAB_UNIT_KINDS
    ]
    if definition_units:
        for unit in definition_units:
            if unit.symbol is None:
                continue
            node = _new_node(
                language="matlab",
                path=source_file.path,
                kind=MATLAB_UNIT_KINDS[unit.kind],
                symbol=unit.symbol,
                module_symbol=module_symbol,
                start_line=unit.start_line,
                end_line=unit.end_line,
            )
            builder.add_node(node)
    else:
        _add_matlab_fallback_nodes(builder, source_file, module_symbol)

    _add_matlab_containment(builder, source_file.path, file_node)


def _matlab_function_name(code: str) -> str | None:
    """Return the declared name from one MATLAB function line."""

    stripped = code.lstrip()
    if not stripped.startswith("function"):
        return None
    declaration = stripped[len("function") :].strip()
    if "=" in declaration:
        declaration = declaration.split("=", 1)[1].strip()
    match = re.match(r"([A-Za-z]\w*)", declaration)
    return match.group(1) if match is not None else None


def _add_matlab_fallback_nodes(
    builder: _GraphBuilder,
    source_file: SourceFile,
    module_symbol: str,
) -> None:
    """Find basic MATLAB definitions when structured units are unavailable."""

    lines = source_file.text.splitlines()
    code_lines = _matlab_code_lines(lines)
    class_matches = (
        match
        for code in code_lines
        if (match := MATLAB_CLASS.match(code)) is not None
    )
    class_match = next(class_matches, None)
    class_name = class_match.group("name") if class_match else None

    declarations: list[tuple[int, str]] = []
    for line, code in enumerate(code_lines, 1):
        name = _matlab_function_name(code)
        if name is not None:
            declarations.append((line, name))

    if class_name is not None:
        class_line = next(
            line
            for line, code in enumerate(code_lines, 1)
            if MATLAB_CLASS.match(code) is not None
        )
        builder.add_node(
            _new_node(
                language="matlab",
                path=source_file.path,
                kind="class",
                symbol=class_name,
                module_symbol=module_symbol,
                start_line=class_line,
                end_line=max(class_line, len(lines)),
            )
        )

    for position, (line, name) in enumerate(declarations):
        end_line = (
            declarations[position + 1][0] - 1
            if position + 1 < len(declarations)
            else max(line, len(lines))
        )
        kind: NodeKind = "method" if class_name is not None else "function"
        symbol = f"{class_name}.{name}" if class_name is not None else name
        builder.add_node(
            _new_node(
                language="matlab",
                path=source_file.path,
                kind=kind,
                symbol=symbol,
                module_symbol=module_symbol,
                start_line=line,
                end_line=end_line,
            )
        )


def _add_matlab_containment(
    builder: _GraphBuilder,
    path: Path,
    file_node: GraphNode,
) -> None:
    """Connect one MATLAB file, its class, and its definitions."""

    nodes = [
        node
        for node in builder.nodes.values()
        if node.language == "matlab"
        and node.provenance.path == path
        and node.identifier != file_node.identifier
    ]
    classes = [node for node in nodes if node.kind == "class"]
    for node in nodes:
        parent = file_node
        if node.kind == "method":
            matching = [
                class_node
                for class_node in classes
                if node.symbol.startswith(class_node.symbol + ".")
            ]
            if matching:
                parent = matching[0]
        builder.add_relation(
            GraphRelation(
                source_identifier=parent.identifier,
                target_identifier=node.identifier,
                target_name=node.symbol,
                kind="contains",
                certainty="exact",
                reason="The definition belongs to this MATLAB source scope.",
                provenance=node.provenance,
            )
        )


def _matlab_code_lines(lines: list[str]) -> list[str]:
    """Remove MATLAB strings and comments while preserving line numbers."""

    result: list[str] = []
    in_block_comment = False
    for line in lines:
        code, in_block_comment = _matlab_code_line(line, in_block_comment)
        result.append(code)
    return result


def _matlab_code_line(line: str, in_block: bool) -> tuple[str, bool]:
    """Return visible MATLAB code from one physical source line."""

    stripped = line.lstrip()
    if in_block:
        if stripped.startswith("%}"):
            return "", False
        return "", True
    if stripped.startswith("%{"):
        return "", True

    result = list(line)
    quote: str | None = None
    index = 0
    while index < len(line):
        character = line[index]
        if quote is not None:
            result[index] = " "
            if character == quote:
                if index + 1 < len(line) and line[index + 1] == quote:
                    result[index + 1] = " "
                    index += 2
                    continue
                quote = None
            index += 1
            continue

        if character == "%":
            for position in range(index, len(result)):
                result[position] = " "
            break
        if character == '"':
            result[index] = " "
            quote = character
        elif character == "'" and _starts_matlab_string(line, index):
            result[index] = " "
            quote = character
        index += 1

    return "".join(result), False


def _starts_matlab_string(line: str, index: int) -> bool:
    """Distinguish a simple MATLAB character vector from transpose syntax."""

    previous = line[:index].rstrip()
    return not previous or not (
        previous[-1].isalnum() or previous[-1] in "_)]}'"
    )


def _add_matlab_relations(
    builder: _GraphBuilder,
    index: _NodeIndex,
    source_file: SourceFile,
) -> None:
    """Add MATLAB inheritance, imports, calls, and uncertain references."""

    code_lines = _matlab_code_lines(source_file.text.splitlines())
    path_nodes = index.by_path.get(source_file.path, [])
    file_node = next(node for node in path_nodes if node.kind == "file")
    for line_number, code in enumerate(code_lines, 1):
        if not code.strip():
            continue

        class_match = MATLAB_CLASS.match(code)
        if class_match is not None:
            class_node = _smallest_source_node(
                path_nodes,
                line_number,
                kinds={"class"},
            )
            if class_node is not None:
                bases = class_match.group("bases") or ""
                for base in bases.split("&"):
                    name = base.strip()
                    if name:
                        _add_matlab_reference(
                            builder,
                            index,
                            class_node,
                            "inherits",
                            name,
                            line_number,
                        )
            continue

        import_match = MATLAB_IMPORT.match(code)
        if import_match is not None:
            source = _matlab_owner(path_nodes, file_node, line_number)
            _add_matlab_import(
                builder,
                index,
                source,
                import_match.group("names"),
                line_number,
            )
            continue

        if _matlab_function_name(code) is not None:
            continue

        source = _matlab_owner(path_nodes, file_node, line_number)
        for match in MATLAB_CALL.finditer(code):
            name = match.group("name")
            if name.split(".", 1)[0].lower() in MATLAB_CONTROL_WORDS:
                continue
            _add_matlab_reference(
                builder,
                index,
                source,
                "calls",
                name,
                line_number,
            )


def _smallest_source_node(
    nodes: list[GraphNode],
    line: int,
    *,
    kinds: set[NodeKind] | None = None,
) -> GraphNode | None:
    """Return the narrowest source definition containing one line."""

    candidates = [
        node
        for node in nodes
        if node.provenance.start_line <= line <= node.provenance.end_line
        and (kinds is None or node.kind in kinds)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda node: (
            node.provenance.end_line - node.provenance.start_line,
            _node_sort_key(node),
        ),
    )


def _matlab_owner(
    nodes: list[GraphNode],
    file_node: GraphNode,
    line: int,
) -> GraphNode:
    """Return the narrowest MATLAB definition owning one source line."""

    return (
        _smallest_source_node(
            nodes,
            line,
            kinds={"class", "function", "method"},
        )
        or file_node
    )


def _matlab_candidates(
    index: _NodeIndex,
    source: GraphNode,
    written_name: str,
) -> tuple[GraphNode, ...]:
    """Return MATLAB definitions that may match one written name."""

    short_name = written_name.rsplit(".", 1)[-1]
    same_file = [
        node
        for node in index.by_path.get(source.provenance.path, [])
        if node.kind in DEFINITION_KINDS
        and (
            node.name == short_name
            or node.symbol == written_name
            or node.qualified_name == written_name
        )
    ]
    if same_file:
        return tuple(sorted(same_file, key=_node_sort_key))

    candidates = [
        node
        for node in index.by_name.get(short_name, [])
        if node.language == "matlab" and node.kind in DEFINITION_KINDS
    ]
    return tuple(sorted(candidates, key=_node_sort_key))


def _add_matlab_import(
    builder: _GraphBuilder,
    index: _NodeIndex,
    source: GraphNode,
    target_name: str,
    line: int,
) -> None:
    """Add an exact MATLAB import statement."""

    candidates = index.qualified(target_name)
    target = candidates[0].identifier if len(candidates) == 1 else None
    builder.add_relation(
        GraphRelation(
            source_identifier=source.identifier,
            target_identifier=target,
            target_name=target_name,
            kind="imports",
            certainty="exact" if len(candidates) <= 1 else "ambiguous",
            reason=(
                "The MATLAB import statement directly names this target."
                if len(candidates) <= 1
                else "Several selected definitions have this imported name."
            ),
            provenance=SourceLocation(source.provenance.path, line, line),
        )
    )


def _add_matlab_reference(
    builder: _GraphBuilder,
    index: _NodeIndex,
    source: GraphNode,
    kind: RelationKind,
    target_name: str,
    line: int,
) -> None:
    """Add a MATLAB relation without overstating path-based resolution."""

    candidates = _matlab_candidates(index, source, target_name)
    location = SourceLocation(source.provenance.path, line, line)
    if len(candidates) == 1:
        target = candidates[0]
        same_file = target.provenance.path == source.provenance.path
        builder.add_relation(
            GraphRelation(
                source_identifier=source.identifier,
                target_identifier=target.identifier,
                target_name=target_name,
                kind=kind,
                certainty="exact" if same_file else "ambiguous",
                reason=(
                    "The name resolves to one definition in the same MATLAB "
                    "file."
                    if same_file
                    else "One selected definition matches, but the MATLAB "
                    "path may resolve another definition."
                ),
                provenance=location,
            )
        )
        return

    if not candidates:
        builder.add_relation(
            GraphRelation(
                source_identifier=source.identifier,
                target_identifier=None,
                target_name=target_name,
                kind="references" if kind == "calls" else kind,
                certainty="unresolved",
                reason=(
                    "The name appears in source, but its MATLAB definition "
                    "is not selected."
                ),
                provenance=location,
            )
        )
        return

    for target in candidates:
        builder.add_relation(
            GraphRelation(
                source_identifier=source.identifier,
                target_identifier=target.identifier,
                target_name=target_name,
                kind=kind,
                certainty="ambiguous",
                reason=(
                    "Several selected MATLAB definitions share this name, "
                    "and path resolution is unknown."
                ),
                provenance=location,
            )
        )
