"""Load documents from a project's ``target/`` folder.

A project has two kinds of documents. ``sources/`` holds the knowledge base:
it is copied in, indexed, embedded, and searched. ``target/`` holds what the
user is working on (a paper to summarize, a manuscript, generated reports):
it is never indexed, so a report QAChat writes there can never come back as
a "source", and it is read only when a tool asks for it.

``load_target`` reads one target file with the same readers that index
sources (pdfminer.six layout text for PDFs, structure-aware units for
Markdown and code) and splits it into numbered passages with the same
chunker, so tools can cite passages exactly as they cite indexed evidence.

Key Features
------------

- **Folder**

    ``target_folder`` returns (and creates) ``<project>/target``;
    ``target_files`` lists readable files in it.

- **Choosing a file**

    ``find_target`` picks the file a question names, or the only file.

- **Loading**

    ``load_target`` returns a ``TargetDocument`` with passages.

Module Structure
----------------

- ``TARGET_FOLDER``
- ``TargetDocument``
- ``target_folder``
- ``target_files``
- ``find_target``
- ``load_target``
- ``save_target_file``
- ``free_target_path``

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from qachat.chunking import Chunk, chunk_source_units
from qachat.retrievals.document_reading import estimate_tokens, named_paths
from qachat.sources import extract_source_units, read_repository_files

__all__ = [
    "READABLE_SUFFIXES",
    "TARGET_FOLDER",
    "TargetDocument",
    "find_target",
    "free_target_path",
    "load_target",
    "save_target_file",
    "target_files",
    "target_folder",
]

# Folder name inside a project; never indexed.
TARGET_FOLDER: Final[str] = "target"
# File types load_target can read, as for indexed sources.
READABLE_SUFFIXES: Final[tuple[str, ...]] = (
    ".pdf",
    ".md",
    ".txt",
    ".py",
    ".m",
    ".ipynb",
)


@dataclass(frozen=True, slots=True)
class TargetDocument:
    """One target file, read and split into passages.

    Attributes
    ----------
    path: pathlib.Path
        Path relative to the project, such as ``target/paper.pdf``.
    passages: tuple[Chunk, ...]
        Passages in reading order; passage ``n`` is ``passages[n - 1]``.
    estimated_tokens: int
        Estimated tokens of all passages.

    """

    path: Path
    passages: tuple[Chunk, ...]
    estimated_tokens: int


def target_folder(project_root: Path) -> Path:
    """Return ``<project>/target``, creating it when missing.

    Parameters
    ----------
    project_root: pathlib.Path
        The project folder.

    Returns
    -------
    pathlib.Path
        The target folder.

    """

    folder = Path(project_root) / TARGET_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def target_files(project_root: Path) -> list[Path]:
    """List readable files in the target folder.

    Parameters
    ----------
    project_root: pathlib.Path
        The project folder.

    Returns
    -------
    list[pathlib.Path]
        Paths relative to the target folder, sorted, of files with a
        readable suffix (subfolders included; hidden files skipped).

    """

    folder = target_folder(project_root)
    return sorted(
        path.relative_to(folder)
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in READABLE_SUFFIXES
        and not any(part.startswith(".") for part in path.parts)
    )


def find_target(
    question: str, project_root: Path, *, suffixes: tuple[str, ...] = ()
) -> Path | None:
    """Pick the target file a question is about.

    Parameters
    ----------
    question: str
        The question.
    project_root: pathlib.Path
        The project folder.
    suffixes: tuple[str, ...], optional
        Only consider these file types (for example ``(".pdf",)``); empty
        allows every readable type. Defaults to ``()``.

    Returns
    -------
    pathlib.Path | None
        The file the question names (by path, name, or stem); otherwise the
        only candidate file; otherwise None.

    """

    files = [
        path
        for path in target_files(project_root)
        if not suffixes or path.suffix.lower() in suffixes
    ]
    named = named_paths(question, files)
    if len(named) == 1:
        return named[0]
    if not named and len(files) == 1:
        return files[0]
    return None


def load_target(project_root: Path, name: Path | str) -> TargetDocument:
    """Read one target file and split it into passages.

    Parameters
    ----------
    project_root: pathlib.Path
        The project folder.
    name: pathlib.Path | str
        File path relative to the target folder.

    Returns
    -------
    TargetDocument
        The passages, with paths such as ``target/paper.pdf``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist; the message lists the target files.
    ValueError
        If the file type cannot be read, the path leaves the target folder,
        or the file has no text (for example a scanned PDF).

    """

    folder = target_folder(project_root)
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Target files must stay inside the target folder.")
    if relative.suffix.lower() not in READABLE_SUFFIXES:
        raise ValueError(
            f"{relative.name} cannot be read; supported types: "
            f"{', '.join(READABLE_SUFFIXES)}."
        )
    if not (folder / relative).is_file():
        available = ", ".join(str(p) for p in target_files(project_root))
        raise FileNotFoundError(
            f"{TARGET_FOLDER}/{relative.as_posix()} does not exist. Files in "
            f"{TARGET_FOLDER}/: {available or 'none'}."
        )
    files = read_repository_files(
        {"repository": str(folder), "selected_files": [relative.as_posix()]}
    )
    units = extract_source_units(files)
    chunks = chunk_source_units(
        units, count_tokens=estimate_tokens, overlap_tokens=0
    )
    if not chunks:
        raise ValueError(
            f"{relative.name} has no readable text (a scanned PDF needs OCR "
            "first)."
        )
    shown = Path(TARGET_FOLDER) / relative
    passages = tuple(
        Chunk(
            identifier=f"{TARGET_FOLDER}:{chunk.identifier}",
            path=shown,
            text=chunk.text,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            kind=chunk.kind,
            symbol=chunk.symbol,
            section=chunk.section,
            cell=chunk.cell,
        )
        for chunk in chunks
    )
    return TargetDocument(
        path=shown,
        passages=passages,
        estimated_tokens=sum(estimate_tokens(c.text) for c in passages),
    )


def save_target_file(project_root: Path, name: str, text: str) -> Path:
    """Write a generated file into the target folder without overwriting.

    Parameters
    ----------
    project_root: pathlib.Path
        The project folder.
    name: str
        File name, such as ``paper_summary.md``.
    text: str
        Content.

    Returns
    -------
    pathlib.Path
        Path relative to the project. When the name is taken, ``-2``,
        ``-3``, ... is added before the suffix, so earlier files (including
        the user's own) are never replaced.

    """

    candidate = free_target_path(project_root, name)
    candidate.write_text(text, encoding="utf-8")
    return Path(TARGET_FOLDER) / candidate.name


def free_target_path(project_root: Path, name: str) -> Path:
    """Return a path in the target folder that no file uses yet.

    Parameters
    ----------
    project_root: pathlib.Path
        The project folder.
    name: str
        Wanted file name, such as ``paper_summary.pdf``.

    Returns
    -------
    pathlib.Path
        ``target/<name>``, or with ``-2``, ``-3``, ... before the suffix
        when that name is taken.

    """

    folder = target_folder(project_root)
    candidate = folder / name
    stem, suffix = candidate.stem, candidate.suffix
    number = 2
    while candidate.exists():
        candidate = folder / f"{stem}-{number}{suffix}"
        number += 1
    return candidate
