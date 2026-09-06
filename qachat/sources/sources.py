"""Read selected repository files and identify natural source units.

Files are treated as text, never imported or executed. Python uses the
standard-library syntax tree, MATLAB uses bounded declaration parsing,
Markdown uses headings, and notebooks use their code and Markdown cells.
Other supported text files remain whole.

"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re

MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
MATLAB_BLOCK = re.compile(
    r"^\s*(properties|methods|events|enumeration)\b",
    re.IGNORECASE,
)
MATLAB_CLASS = re.compile(
    r"^\s*classdef(?:\s*\([^)]*\))?\s+([A-Za-z]\w*)",
    re.IGNORECASE,
)
MATLAB_FUNCTION = re.compile(r"^\s*function\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One repository text file held in memory.

    Attributes
    ----------
    path: pathlib.Path
        File path relative to the repository root.
    text: str
        UTF-8 file content.

    """

    path: Path
    text: str


@dataclass(frozen=True, slots=True)
class SourceUnit:
    """One natural section of a source file.

    Attributes
    ----------
    path: pathlib.Path
        File path relative to the repository root.
    text: str
        Exact source text covered by this unit.
    start_line: int
        First one-based source line.
    end_line: int
        Last one-based source line.
    kind: str
        Unit type, such as ``python_function`` or ``markdown_section``.
    symbol: str | None
        Class, method, function, or section name when available.
    section: str | None
        Document section name when available.
    cell: int | None
        One-based notebook cell number when available.
    children: tuple[SourceUnit, ...]
        Smaller natural units used only when this unit is too large.

    """

    path: Path
    text: str
    start_line: int
    end_line: int
    kind: str
    symbol: str | None = None
    section: str | None = None
    cell: int | None = None
    children: tuple[SourceUnit, ...] = ()


def read_repository_files(report: dict[str, object]) -> list[SourceFile]:
    """Read the files selected by :func:`qachat.repository.scan_repository`.

    Paths are checked again immediately before reading. This prevents a
    changed or hand-written report from reading outside the repository.

    Parameters
    ----------
    report: dict[str, object]
        Repository scan report.

    Returns
    -------
    list[SourceFile]
        Selected UTF-8 files in stable path order.

    Raises
    ------
    ValueError
        If the report or a selected path is unsafe or not UTF-8 text.

    """

    try:
        repository_value = report["repository"]
        selected_value = report["selected_files"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Use a report returned by scan_repository."
        ) from error

    if not isinstance(repository_value, str):
        raise ValueError("The scan report has an invalid repository path.")

    if not isinstance(selected_value, list):
        raise ValueError("The scan report has an invalid selected file list.")

    root = Path(repository_value).resolve(strict=True)
    files: list[SourceFile] = []
    for relative_value in selected_value:
        if not isinstance(relative_value, str):
            raise ValueError("The scan report contains an invalid file path.")

        relative = PurePosixPath(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Selected files must stay inside the repository.")

        candidate = root.joinpath(*relative.parts)
        target = candidate.resolve(strict=True)
        if (
            candidate.is_symlink()
            or not target.is_relative_to(root)
            or not target.is_file()
        ):
            raise ValueError(
                "Selected files must be regular repository files."
            )

        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Selected file is not UTF-8 text: {relative.as_posix()}"
            ) from error

        files.append(SourceFile(Path(relative.as_posix()), text))

    return files


def extract_source_units(files: list[SourceFile]) -> list[SourceUnit]:
    """Identify natural source units without executing source code.

    Parameters
    ----------
    files: list[SourceFile]
        Repository text files.

    Returns
    -------
    list[SourceUnit]
        Natural units in the same order as the input files.

    """

    units: list[SourceUnit] = []
    for source_file in files:
        lines = source_file.text.splitlines(keepends=True)
        if not lines:
            continue

        suffix = source_file.path.suffix.lower()
        if suffix == ".py":
            units.extend(_python_units(source_file.path, lines))
        elif suffix == ".m":
            units.extend(_matlab_units(source_file.path, lines))
        elif suffix == ".md":
            units.extend(_markdown_units(source_file.path, lines))
        elif suffix == ".ipynb":
            units.extend(_notebook_units(source_file))
        else:
            unit = _source_unit(
                source_file.path,
                lines,
                1,
                len(lines),
                kind="file",
            )
            if unit is not None:
                units.append(unit)

    return units


def _source_unit(
    path: Path,
    lines: list[str],
    start_line: int,
    end_line: int,
    *,
    kind: str,
    symbol: str | None = None,
    section: str | None = None,
    cell: int | None = None,
    children: tuple[SourceUnit, ...] = (),
) -> SourceUnit | None:
    """Build a non-empty unit from inclusive one-based source lines.

    Parameters
    ----------
    path: pathlib.Path
        Relative source path.
    lines: list[str]
        Source lines with their endings.
    start_line: int
        First one-based line.
    end_line: int
        Last one-based line.
    kind: str
        Unit type.
    symbol: str | None, optional
        Symbol name.
    section: str | None, optional
        Section name.
    cell: int | None, optional
        One-based notebook cell number.
    children: tuple[SourceUnit, ...], optional
        Smaller natural units.

    Returns
    -------
    SourceUnit | None
        Source unit, or ``None`` when the range has no visible text.

    """

    if start_line > end_line:
        return None

    text = "".join(lines[start_line - 1 : end_line])
    if not text.strip():
        return None

    return SourceUnit(
        path=path,
        text=text,
        start_line=start_line,
        end_line=end_line,
        kind=kind,
        symbol=symbol,
        section=section,
        cell=cell,
        children=children,
    )


def _node_start(node: ast.AST) -> int:
    """Return the first line of a Python definition and its decorators.

    Parameters
    ----------
    node: ast.AST
        Function or class definition.

    Returns
    -------
    int
        First one-based source line.

    """

    decorators = getattr(node, "decorator_list", ())
    lines = [node.lineno]
    lines.extend(decorator.lineno for decorator in decorators)
    return min(lines)


def _node_end(node: ast.AST) -> int:
    """Return the final line of a Python definition.

    Parameters
    ----------
    node: ast.AST
        Function or class definition.

    Returns
    -------
    int
        Last one-based source line.

    Raises
    ------
    ValueError
        If Python does not report an ending line.

    """

    end_line = getattr(node, "end_lineno", None)
    if not isinstance(end_line, int):
        raise ValueError("Python did not report the definition's final line.")

    return end_line


def _python_class_unit(
    path: Path,
    lines: list[str],
    node: ast.ClassDef,
) -> SourceUnit:
    """Build one class unit with complete, non-overlapping child units.

    Parameters
    ----------
    path: pathlib.Path
        Relative source path.
    lines: list[str]
        Source lines with their endings.
    node: ast.ClassDef
        Parsed class definition.

    Returns
    -------
    SourceUnit
        Complete class unit.

    """

    start_line = _node_start(node)
    end_line = _node_end(node)
    methods = sorted(
        (
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        key=_node_start,
    )

    children: list[SourceUnit] = []
    cursor = start_line
    for method in methods:
        method_start = _node_start(method)
        context = _source_unit(
            path,
            lines,
            cursor,
            method_start - 1,
            kind="python_class_context",
            symbol=node.name,
        )
        if context is not None:
            children.append(context)

        method_unit = _source_unit(
            path,
            lines,
            method_start,
            _node_end(method),
            kind="python_method",
            symbol=f"{node.name}.{method.name}",
        )
        if method_unit is not None:
            children.append(method_unit)

        cursor = _node_end(method) + 1

    tail = _source_unit(
        path,
        lines,
        cursor,
        end_line,
        kind="python_class_context",
        symbol=node.name,
    )
    if tail is not None:
        children.append(tail)

    unit = _source_unit(
        path,
        lines,
        start_line,
        end_line,
        kind="python_class",
        symbol=node.name,
        children=tuple(children),
    )
    if unit is None:
        raise ValueError("A parsed Python class unexpectedly had no text.")

    return unit


def _python_units(path: Path, lines: list[str]) -> list[SourceUnit]:
    """Identify top-level Python definitions and surrounding context.

    Parameters
    ----------
    path: pathlib.Path
        Relative source path.
    lines: list[str]
        Source lines with their endings.

    Returns
    -------
    list[SourceUnit]
        Python units. Invalid Python remains one safe file unit.

    """

    source = "".join(lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        fallback = _source_unit(
            path,
            lines,
            1,
            len(lines),
            kind="python_file",
        )
        return [fallback] if fallback is not None else []

    definitions = sorted(
        (
            node
            for node in tree.body
            if isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            )
        ),
        key=_node_start,
    )

    units: list[SourceUnit] = []
    cursor = 1
    for node in definitions:
        start_line = _node_start(node)
        context = _source_unit(
            path,
            lines,
            cursor,
            start_line - 1,
            kind="python_module_context",
            symbol=path.stem,
        )
        if context is not None:
            units.append(context)

        if isinstance(node, ast.ClassDef):
            units.append(_python_class_unit(path, lines, node))
        else:
            function = _source_unit(
                path,
                lines,
                start_line,
                _node_end(node),
                kind="python_function",
                symbol=node.name,
            )
            if function is not None:
                units.append(function)

        cursor = _node_end(node) + 1

    tail = _source_unit(
        path,
        lines,
        cursor,
        len(lines),
        kind="python_module_context",
        symbol=path.stem,
    )
    if tail is not None:
        units.append(tail)

    return units


def _markdown_units(path: Path, lines: list[str]) -> list[SourceUnit]:
    """Identify Markdown preamble and heading sections.

    Parameters
    ----------
    path: pathlib.Path
        Relative source path.
    lines: list[str]
        Source lines with their endings.

    Returns
    -------
    list[SourceUnit]
        Markdown units.

    """

    headings: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines, 1):
        match = MARKDOWN_HEADING.match(line.rstrip("\r\n"))
        if match is not None:
            title = match.group(2).strip().rstrip("#").rstrip()
            headings.append((line_number, title))

    units: list[SourceUnit] = []
    first_heading = headings[0][0] if headings else len(lines) + 1
    preamble = _source_unit(
        path,
        lines,
        1,
        first_heading - 1,
        kind="markdown_preamble",
    )
    if preamble is not None:
        units.append(preamble)

    for index, (start_line, title) in enumerate(headings):
        end_line = (
            headings[index + 1][0] - 1
            if index + 1 < len(headings)
            else len(lines)
        )
        section = _source_unit(
            path,
            lines,
            start_line,
            end_line,
            kind="markdown_section",
            section=title,
        )
        if section is not None:
            units.append(section)

    return units


def _matlab_function_name(line: str) -> str | None:
    """Return the declared name from one MATLAB function line.

    Parameters
    ----------
    line: str
        MATLAB source line beginning with function.

    Returns
    -------
    str | None
        Declared function name, or None for an incomplete declaration.

    """

    signature = line.split("%", 1)[0].strip()
    signature = re.sub(
        r"^function\b",
        "",
        signature,
        flags=re.IGNORECASE,
    ).strip()
    if "=" in signature:
        signature = signature.split("=", 1)[1].strip()

    match = re.match(r"([A-Za-z]\w*)", signature)
    return match.group(1) if match is not None else None


def _matlab_markers(
    lines: list[str],
    *,
    class_name: str | None,
    first_line: int,
) -> list[tuple[int, str, str | None]]:
    """Find visible MATLAB blocks and function declarations.

    Parameters
    ----------
    lines: list[str]
        Source lines with their endings.
    class_name: str | None
        Enclosing class name, when the file defines a class.
    first_line: int
        First one-based line to inspect.

    Returns
    -------
    list[tuple[int, str, str | None]]
        Line, source-unit kind, and optional symbol for each marker.

    """

    markers: list[tuple[int, str, str | None]] = []
    for line_number in range(first_line, len(lines) + 1):
        line = lines[line_number - 1]
        if line.lstrip().startswith("%"):
            continue

        if MATLAB_FUNCTION.match(line) is not None:
            name = _matlab_function_name(line)
            if class_name is None:
                kind = "matlab_function"
                symbol = name
            else:
                kind = (
                    "matlab_constructor"
                    if name == class_name
                    else "matlab_method"
                )
                symbol = (
                    f"{class_name}.{name}" if name is not None else class_name
                )
            markers.append((line_number, kind, symbol))
            continue

        block = MATLAB_BLOCK.match(line)
        if block is not None:
            name = block.group(1).lower()
            markers.append(
                (
                    line_number,
                    f"matlab_{name}_block",
                    class_name,
                )
            )

    return markers


def _partition_matlab(
    path: Path,
    lines: list[str],
    *,
    start_line: int,
    end_line: int,
    markers: list[tuple[int, str, str | None]],
    context_kind: str,
    context_symbol: str | None,
) -> tuple[SourceUnit, ...]:
    """Partition a MATLAB range at visible structural markers.

    Parameters
    ----------
    path: pathlib.Path
        Relative source path.
    lines: list[str]
        Source lines with their endings.
    start_line: int
        First one-based line in the range.
    end_line: int
        Last one-based line in the range.
    markers: list[tuple[int, str, str | None]]
        Structural markers inside the range.
    context_kind: str
        Kind used for text outside a marked declaration.
    context_symbol: str | None
        Symbol used for surrounding context.

    Returns
    -------
    tuple[SourceUnit, ...]
        Complete, non-overlapping source units.

    """

    if not markers:
        context = _source_unit(
            path,
            lines,
            start_line,
            end_line,
            kind=context_kind,
            symbol=context_symbol,
        )
        return (context,) if context is not None else ()

    units: list[SourceUnit] = []
    cursor = start_line
    for index, (marker_line, kind, symbol) in enumerate(markers):
        context = _source_unit(
            path,
            lines,
            cursor,
            marker_line - 1,
            kind=context_kind,
            symbol=context_symbol,
        )
        if context is not None:
            units.append(context)

        next_line = (
            markers[index + 1][0] - 1 if index + 1 < len(markers) else end_line
        )
        marked = _source_unit(
            path,
            lines,
            marker_line,
            next_line,
            kind=kind,
            symbol=symbol,
        )
        if marked is not None:
            units.append(marked)
        cursor = next_line + 1

    return tuple(units)


def _matlab_units(path: Path, lines: list[str]) -> list[SourceUnit]:
    """Identify MATLAB classes, blocks, methods, functions, and scripts.

    Parameters
    ----------
    path: pathlib.Path
        Relative .m source path.
    lines: list[str]
        Source lines with their endings.

    Returns
    -------
    list[SourceUnit]
        MATLAB units. The source is inspected as text and never executed.

    """

    class_line: int | None = None
    class_name: str | None = None
    for line_number, line in enumerate(lines, start=1):
        if line.lstrip().startswith("%"):
            continue
        match = MATLAB_CLASS.match(line)
        if match is not None:
            class_line = line_number
            class_name = match.group(1)
            break

    if class_line is None or class_name is None:
        markers = _matlab_markers(
            lines,
            class_name=None,
            first_line=1,
        )
        return list(
            _partition_matlab(
                path,
                lines,
                start_line=1,
                end_line=len(lines),
                markers=markers,
                context_kind="matlab_script",
                context_symbol=path.stem,
            )
        )

    units: list[SourceUnit] = []
    preamble = _source_unit(
        path,
        lines,
        1,
        class_line - 1,
        kind="matlab_file_context",
        symbol=path.stem,
    )
    if preamble is not None:
        units.append(preamble)

    markers = _matlab_markers(
        lines,
        class_name=class_name,
        first_line=class_line + 1,
    )
    children = _partition_matlab(
        path,
        lines,
        start_line=class_line,
        end_line=len(lines),
        markers=markers,
        context_kind="matlab_class_context",
        context_symbol=class_name,
    )
    class_unit = _source_unit(
        path,
        lines,
        class_line,
        len(lines),
        kind="matlab_class",
        symbol=class_name,
        children=children,
    )
    if class_unit is not None:
        units.append(class_unit)

    return units


def _notebook_units(source_file: SourceFile) -> list[SourceUnit]:
    """Extract code and Markdown cells without stored outputs or metadata.

    Parameters
    ----------
    source_file: SourceFile
        UTF-8 notebook JSON.

    Returns
    -------
    list[SourceUnit]
        Non-empty code and Markdown cells with one-based cell provenance.

    Raises
    ------
    ValueError
        If the notebook is malformed.

    """

    try:
        notebook = json.loads(source_file.text)
        cells = notebook["cells"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(
            f"Invalid Jupyter notebook: {source_file.path.as_posix()}"
        ) from error

    if not isinstance(cells, list):
        raise ValueError(
            f"Invalid Jupyter notebook: {source_file.path.as_posix()}"
        )

    units: list[SourceUnit] = []
    for cell_number, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict):
            raise ValueError(
                f"Invalid Jupyter notebook: {source_file.path.as_posix()}"
            )

        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown"}:
            continue

        raw_source = cell.get("source", "")
        if isinstance(raw_source, list):
            if not all(isinstance(line, str) for line in raw_source):
                raise ValueError(
                    f"Invalid Jupyter notebook: "
                    f"{source_file.path.as_posix()}"
                )
            cell_text = "".join(raw_source)
        elif isinstance(raw_source, str):
            cell_text = raw_source
        else:
            raise ValueError(
                f"Invalid Jupyter notebook: {source_file.path.as_posix()}"
            )

        if not cell_text.strip():
            continue

        lines = cell_text.splitlines(keepends=True)
        if not lines:
            lines = [cell_text]

        section: str | None = None
        if cell_type == "markdown":
            for line in lines:
                heading = MARKDOWN_HEADING.match(line.rstrip("\r\n"))
                if heading is not None:
                    section = heading.group(2).strip().rstrip("#").rstrip()
                    break

        unit = _source_unit(
            source_file.path,
            lines,
            1,
            len(lines),
            kind=f"notebook_{cell_type}",
            section=section,
            cell=cell_number,
        )
        if unit is not None:
            units.append(unit)

    return units
