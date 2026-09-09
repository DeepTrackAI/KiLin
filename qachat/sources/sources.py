"""Read repository files and coordinate natural-unit extraction.

Classes
-------
SourceFile
    Text read or extracted from one selected repository file.
SourceUnit
    One natural, traceable part of a source file.

Functions
---------
read_repository_files
    Safely read files selected by qachat.repository.scan_repository.
extract_source_units
    Send known file types to automatically found extension modules.

Extension modules use the sources_<extension>.py naming convention.

"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path, PurePosixPath
from types import ModuleType


__all__ = [
    "SourceFile",
    "SourceUnit",
    "read_repository_files",
    "extract_source_units",
]

EXTENSION_PACKAGE = "qachat.sources.extensions"


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One repository file converted to text.

    Attributes
    ----------
    path: pathlib.Path
        File path relative to the repository root.
    text: str
        UTF-8 text or text extracted from a document.

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
        Unit type, such as python_function or markdown_section.
    symbol: str | None
        Class, method, function, or section name when available.
    section: str | None
        Document section or PDF page name when available.
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
    """Read files selected by qachat.repository.scan_repository.

    Paths are checked again immediately before reading. Known binary formats
    may provide their own reader; other files must be UTF-8 text.

    Parameters
    ----------
    report: dict[str, object]
        Repository scan report.

    Returns
    -------
    list[SourceFile]
        Selected files in stable path order.

    Raises
    ------
    ValueError
        If the report, a selected path, or selected file is invalid.

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

        relative_path = Path(relative.as_posix())
        extension = _extension_module(relative_path)
        reader = getattr(extension, "read_source_file", None)
        if reader is not None:
            source_file = reader(target, relative_path)
        else:
            source_file = _read_text_file(target, relative_path)

        if not isinstance(source_file, SourceFile):
            raise RuntimeError(
                f"Invalid source reader for {relative_path.suffix or 'file'}."
            )
        files.append(source_file)

    return files


def extract_source_units(files: list[SourceFile]) -> list[SourceUnit]:
    """Identify natural source units without executing source code.

    Parameters
    ----------
    files: list[SourceFile]
        Repository files converted to text.

    Returns
    -------
    list[SourceUnit]
        Natural units in the same order as the input files.

    """

    units: list[SourceUnit] = []
    for source_file in files:
        extension = _extension_module(source_file.path)
        extractor = getattr(extension, "extract_source_units", None)
        if extractor is not None:
            extracted = extractor(source_file)
            if not isinstance(extracted, list) or any(
                not isinstance(unit, SourceUnit) for unit in extracted
            ):
                raise RuntimeError(
                    f"Invalid source extractor for "
                    f"{source_file.path.suffix}."
                )
            units.extend(extracted)
            continue

        lines = source_file.text.splitlines(keepends=True)
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


def _extension_module(path: Path) -> ModuleType | None:
    """Find an extension module using the source file suffix."""

    suffix = path.suffix.lower().lstrip(".")
    if not suffix:
        return None

    module_name = f"{EXTENSION_PACKAGE}.sources_{suffix}"
    try:
        return import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            return None
        raise


def _read_text_file(target: Path, relative_path: Path) -> SourceFile:
    """Read one UTF-8 text file."""

    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Selected file is not UTF-8 text: {relative_path.as_posix()}"
        ) from error

    return SourceFile(relative_path, text)


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
    """Build a non-empty unit from inclusive one-based source lines."""

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
