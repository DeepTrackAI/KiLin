"""Read selected repository files and extract provenance-bearing text units.

The reader consumes a scan report and checks resolved file containment before
reading. Text files use UTF-8; format-specific readers may transform binary
sources such as PDFs into text. Extraction discovers a suffix-specific hook,
otherwise retaining the whole nonblank file as one source unit.

Paths identify source files relative to the scanned repository. Line ranges
are inclusive and one-based. Their coordinate system follows the extractor:
ordinary text uses file lines, notebooks use cell-local lines, and PDFs use
page-local lines with a page label in ``section``. These distinctions must be
preserved by callers presenting citations. No source code is executed.

Key Features
------------

- **Selected-file reading**

    Read the paths listed in a scan report and validate reader results.

- **Extension dispatch**

    Load hooks by file suffix, falling back only when the hook module itself is
    absent.

- **Structured units**

    Carry symbols, sections, cell numbers, child units, and line ranges into
    retrieval preparation.

- **Progress callbacks**

    Notify callers after each input file without suppressing their callback
    errors.

Module Structure
----------------

- ``SourceFile``

    One repository file converted to text.

- ``SourceUnit``

    One natural section of a source file.

- ``read_repository_files``

    Read files selected by qachat.repository.scan_repository.

- ``extract_source_units``

    Identify natural source units without executing source code.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import qachat
>>> report = qachat.scan_repository("/path/to/repository")
>>> files = qachat.read_repository_files(report)
>>> units = qachat.extract_source_units(files)

"""

from __future__ import annotations

from collections.abc import Callable
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

    Parameters
    ----------
    path: pathlib.Path
        File path relative to the repository root.
    text: str
        UTF-8 text or text extracted from a document.

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

    Parameters
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
    symbol: str | None, optional
        Class, method, function, or section name when available. Defaults to
        ``None``.
    section: str | None, optional
        Document section or PDF page name when available. Defaults to ``None``.
    cell: int | None, optional
        One-based notebook cell number when available. Defaults to ``None``.
    children: tuple[SourceUnit, ...], optional
        Smaller natural units used only when this unit is too large. Defaults
        to ``()``.

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

    Notes
    -----
    Frozen/slotted fields describe provenance; no text is executed. Line
    coordinates are file-local for ordinary text, cell-local for notebooks, and
    page-local for PDFs. Use cell and section to interpret those coordinates.
    Children can cover structural portions of a parent unit and are used when
    chunking oversized units.

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


def read_repository_files(
    report: dict[str, object],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[SourceFile]:
    """Read files selected by qachat.repository.scan_repository.

    Paths are checked again immediately before reading. Known binary formats
    may provide their own reader; other files must be UTF-8 text.

    Parameters
    ----------
    report: dict[str, object]
        Repository scan report.
    progress_callback: Callable[[int, int], None] | None, optional
        Called after each selected file is read with ``current`` and ``total``.
        Defaults to ``None``.

    Returns
    -------
    list[SourceFile]
        Selected files in stable path order.

    Raises
    ------
    ValueError
        If report structure, file containment, regular-file status, UTF-8
        decoding, or the optional callback is invalid.
    RuntimeError
        If an extension reader returns something other than SourceFile.
    OSError
        If a selected file cannot be resolved or read.
    ImportError
        If an existing reader extension cannot import a required dependency.

    Notes
    -----
    The reader follows the selected_files list; it does not rerun selection
    rules or reject report["complete"]=False. Resolved containment is checked,
    but filesystem changes can still occur between checking and opening. PDF
    and other extension-specific errors propagate. Progress callbacks are
    invoked after each completed file.

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

    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None.")

    root = Path(repository_value).resolve(strict=True)
    files: list[SourceFile] = []
    total = len(selected_value)
    for current, relative_value in enumerate(selected_value, start=1):
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
        if progress_callback is not None:
            progress_callback(current, total)

    return files


def extract_source_units(
    files: list[SourceFile],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[SourceUnit]:
    """Identify natural source units without executing source code.

    Parameters
    ----------
    files: list[SourceFile]
        Repository files converted to text.
    progress_callback: Callable[[int, int], None] | None, optional
        Called after each file is structurally extracted with ``current`` and
        ``total``. Defaults to ``None``.

    Returns
    -------
    list[SourceUnit]
        Natural units in the same order as the input files.

    Raises
    ------
    ValueError
        If a progress callback is not callable or an extension rejects its
        source content.
    RuntimeError
        If an extractor does not return a list containing only SourceUnit
        records.
    ImportError
        If an existing extension cannot import a required dependency.

    Notes
    -----
    Missing extension modules trigger a whole-nonblank-file fallback.
    ModuleNotFoundError for a dependency inside an existing extension is not
    treated as a missing hook. Source order is preserved and callback
    exceptions propagate.

    """

    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None.")

    units: list[SourceUnit] = []
    total = len(files)
    for current, source_file in enumerate(files, start=1):
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
            if progress_callback is not None:
                progress_callback(current, total)
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
        if progress_callback is not None:
            progress_callback(current, total)

    return units


def _extension_module(path: Path) -> ModuleType | None:
    """Find an extension module using the source file suffix.

    Parameters
    ----------
    path: pathlib.Path
        Repository-relative source path used for provenance or suffix
        selection.

    Returns
    -------
    ModuleType | None
        Imported suffix-specific module, or None when the file has no supported
        hook module.

    Notes
    -----
    Only ModuleNotFoundError naming the requested hook itself is suppressed;
    errors from its imports propagate.

    """

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
    """Read one UTF-8 text file.

    Parameters
    ----------
    target: pathlib.Path
        Resolved on-disk file path to read.
    relative_path: pathlib.Path
        Repository-relative path stored in the returned source record.

    Returns
    -------
    SourceFile
        SourceFile containing unchanged decoded UTF-8 text and the supplied
        relative path.

    Raises
    ------
    ValueError
        If bytes are not valid UTF-8 text.
    OSError
        If the file cannot be read.

    """

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
    """Build a non-empty unit from inclusive one-based source lines.

    Parameters
    ----------
    path: pathlib.Path
        Repository-relative source path used for provenance or suffix
        selection.
    lines: list[str]
        Source text split into ordered lines, retaining line endings where
        supplied.
    start_line: int
        Inclusive one-based first line in the supplied source coordinate
        system.
    end_line: int
        Inclusive one-based last line in the supplied source coordinate system.
    kind: str
        Structural label assigned by the format-specific extractor.
    symbol: str | None, optional
        Optional code-symbol name associated with this unit. Defaults to
        ``None``.
    section: str | None, optional
        Optional section heading or PDF page label. Defaults to ``None``.
    cell: int | None, optional
        Optional one-based notebook cell position; line ranges are then
        cell-local. Defaults to ``None``.
    children: tuple[SourceUnit, ...], optional
        Ordered nested source units used for structural subdivision. Defaults
        to ``()``.

    Returns
    -------
    SourceUnit | None
        Unit containing the requested inclusive line slice, or None when the
        slice is empty or whitespace-only.

    Notes
    -----
    The helper preserves supplied provenance and child objects. Bounds and
    structural labels are trusted rather than comprehensively validated.

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
