"""Read and write files in a project's ``outputs/`` folder.

A project has two kinds of documents. ``sources/`` holds the knowledge base:
it is copied in, indexed, embedded, and searched. ``outputs/`` holds what
QAChat writes (research briefings, reports) and anything the user adds to
work on: it is never indexed, so a report QAChat writes there can never
come back as a "source", and it is read only when a tool asks for it.

``load_output`` reads one file with the same readers that index sources
(pdfminer.six layout text for PDFs, structure-aware units for Markdown and
code) and splits it into numbered passages with the same chunker, so tools
can cite passages exactly as they cite indexed evidence.

Saving never replaces an existing file: a taken name gets a numbered
suffix. The check and the write are separate steps, so two writers saving
the same name at the same moment are not protected against each other.
Reading rejects absolute paths and ``..``; the shared source reader also
rejects symbolic links and files outside the folder.

Key Features
------------

- **Folder**

    Create ``<project>/outputs`` on first use without moving unrelated
    folders, and list the readable files in it.

- **Choosing a file**

    Pick the file a question names, or the only file.

- **Loading**

    Read one file into numbered passages that tools can cite.

- **Saving**

    Write a generated file under a name no file uses yet.

Module Structure
----------------

- ``OUTPUTS_FOLDER``

    Name of the outputs folder inside a project.

- ``READABLE_SUFFIXES``

    File types ``load_output`` can read.

- ``OutputDocument``

    One file in ``outputs/``, read and split into passages.

- ``output_folder``

    Return ``<project>/outputs``, creating it when missing.

- ``output_files``

    List readable files in the outputs folder.

- ``find_output``

    Pick the file in ``outputs/`` a question is about.

- ``load_output``

    Read one file in ``outputs/`` and split it into passages.

- ``save_output_file``

    Write a generated file into the outputs folder without overwriting.

- ``free_output_path``

    Return a path in the outputs folder that no file uses yet.

Examples
--------
>>> from pathlib import Path
>>> from qachat.tools._shared.outputs import load_output, save_output_file
>>> root = Path("/path/to/project")
>>> saved = save_output_file(root, "notes.md", "# Notes\n\nMRI results.\n")
>>> document = load_output(root, saved.name)
>>> len(document.passages)
1

The path is illustrative; saving creates ``outputs/`` in that project.

"""

from __future__ import annotations

from qachat.workspace_resources import OUTPUTS_FOLDER

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from qachat.chunking import Chunk, chunk_source_units
from qachat.retrievals.document_reading import estimate_tokens, named_paths
from qachat.sources import extract_source_units, read_repository_files

__all__ = [
    "OUTPUTS_FOLDER",
    "READABLE_SUFFIXES",
    "OutputDocument",
    "find_output",
    "free_output_path",
    "load_output",
    "output_files",
    "output_folder",
    "save_output_file",
]

# File types load_output can read, as for indexed sources.
READABLE_SUFFIXES: Final[tuple[str, ...]] = (
    ".pdf",
    ".md",
    ".txt",
    ".py",
    ".m",
    ".ipynb",
)


@dataclass(frozen=True, slots=True)
class OutputDocument:
    """One file in ``outputs/``, read and split into passages.

    Attributes
    ----------
    path: pathlib.Path
        Path relative to the project, such as ``outputs/paper.pdf``.
    passages: tuple[Chunk, ...]
        Passages in reading order; passage ``n`` is ``passages[n - 1]``.
    estimated_tokens: int
        Estimated tokens of all passages.

    """

    path: Path
    passages: tuple[Chunk, ...]
    estimated_tokens: int


def output_folder(project_root: Path) -> Path:
    """Return ``<project>/outputs``, creating it when missing.

    Parameters
    ----------
    project_root: pathlib.Path
        The project folder.

    Returns
    -------
    pathlib.Path
        Project-owned outputs folder, creating it when absent.

    """

    folder = Path(project_root) / OUTPUTS_FOLDER
    if folder.is_symlink():
        raise ValueError(f"Workspace output folder cannot be a link: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def output_files(project_root: Path) -> list[Path]:
    """List readable files in the outputs folder.

    Parameters
    ----------
    project_root: pathlib.Path
        The project folder.

    Returns
    -------
    list[pathlib.Path]
        Paths relative to the outputs folder, sorted, of files with a
        readable suffix (subfolders included; hidden files skipped).

    """

    folder = output_folder(project_root)
    return sorted(
        path.relative_to(folder)
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in READABLE_SUFFIXES
        and not any(part.startswith(".") for part in path.parts)
    )


def find_output(
    question: str, project_root: Path, *, suffixes: tuple[str, ...] = ()
) -> Path | None:
    """Pick the file in ``outputs/`` a question is about.

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
        for path in output_files(project_root)
        if not suffixes or path.suffix.lower() in suffixes
    ]
    named = named_paths(question, files)
    if len(named) == 1:
        return named[0]

    if not named and len(files) == 1:
        return files[0]

    return None


def load_output(project_root: Path, name: Path | str) -> OutputDocument:
    """Read one file in ``outputs/`` and split it into passages.

    Parameters
    ----------
    project_root: pathlib.Path
        The project folder.
    name: pathlib.Path | str
        File path relative to the outputs folder.

    Returns
    -------
    OutputDocument
        The passages, with paths such as ``outputs/paper.pdf``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist; the message lists the files there.
    ValueError
        If the file type cannot be read, the path leaves the outputs folder,
        or the file has no text (for example a scanned PDF).

    """

    folder = output_folder(project_root)
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Files must stay inside the outputs folder.")

    if relative.suffix.lower() not in READABLE_SUFFIXES:
        raise ValueError(
            f"{relative.name} cannot be read; supported types: "
            f"{', '.join(READABLE_SUFFIXES)}."
        )

    if not (folder / relative).is_file():
        available = ", ".join(
            str(path) for path in output_files(project_root)
        )
        raise FileNotFoundError(
            f"{OUTPUTS_FOLDER}/{relative.as_posix()} does not exist. Files "
            f"in {OUTPUTS_FOLDER}/: {available or 'none'}."
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

    shown = Path(OUTPUTS_FOLDER) / relative
    passages = tuple(
        Chunk(
            identifier=f"{OUTPUTS_FOLDER}:{chunk.identifier}",
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
    return OutputDocument(
        path=shown,
        passages=passages,
        estimated_tokens=sum(
            estimate_tokens(passage.text) for passage in passages
        ),
    )


def save_output_file(project_root: Path, name: str, text: str) -> Path:
    """Write a generated file into the outputs folder without overwriting.

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

    candidate = free_output_path(project_root, name)
    candidate.write_text(text, encoding="utf-8")
    return Path(OUTPUTS_FOLDER) / candidate.name


def free_output_path(project_root: Path, name: str) -> Path:
    """Return a path in the outputs folder that no file uses yet.

    Parameters
    ----------
    project_root: pathlib.Path
        The project folder.
    name: str
        Wanted file name, such as ``paper_summary.pdf``.

    Returns
    -------
    pathlib.Path
        ``outputs/<name>``, or with ``-2``, ``-3``, ... before the suffix
        when that name is taken.

    """

    folder = output_folder(project_root)
    candidate = folder / name
    stem, suffix = candidate.stem, candidate.suffix
    number = 2
    while candidate.exists():
        candidate = folder / f"{stem}-{number}{suffix}"
        number += 1
    return candidate
