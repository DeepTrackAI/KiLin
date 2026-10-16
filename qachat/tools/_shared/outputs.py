"""Read and write files in a project's ``outputs/`` folder.

A project has two kinds of documents. ``sources/`` holds the knowledge base:
it is copied in, indexed, embedded, and searched. ``outputs/`` holds what
KíLín writes (research briefings, reports) and anything the user adds to
work on: it is never indexed, so a report KíLín writes there can never
come back as a "source", and it is read only when a tool asks for it.

``load_output`` reads one file with the same readers that index sources
(pdfminer.six layout text for PDFs, structure-aware units for Markdown and
code) and splits it into numbered passages with the same chunker, so tools
can cite passages exactly as they cite indexed evidence.

Saving validates a single filename, writes to an exclusive staging file,
and publishes the completed content without replacing existing paths. A
taken name (including a symbolic link) gets a numbered suffix even when
two writers save concurrently. Reading rejects absolute paths and ``..``;
the shared source reader also rejects symbolic links and files outside
the folder.

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

- ``staged_output_file`` and ``publish_output_file``

    Stage generated content and publish it without replacing a file.

Examples
--------
>>> from pathlib import Path
>>> from kilin.tools._shared.outputs import load_output, save_output_file
>>> root = Path("/path/to/project")
>>> saved = save_output_file(root, "notes.md", "# Notes\n\nMRI results.\n")
>>> document = load_output(root, saved.name)
>>> len(document.passages)
1

The path is illustrative; saving creates ``outputs/`` in that project.

"""

from __future__ import annotations

from kilin.workspace_resources import OUTPUTS_FOLDER

from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Final, Iterator

from kilin.chunking import Chunk, chunk_source_units
from kilin.retrievals.document_reading import estimate_tokens, named_paths
from kilin.sources import extract_source_units, read_repository_files

__all__ = [
    "OUTPUTS_FOLDER",
    "READABLE_SUFFIXES",
    "OutputDocument",
    "find_output",
    "load_output",
    "output_files",
    "output_folder",
    "publish_output_file",
    "save_output_file",
    "staged_output_file",
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
    if folder.resolve() != Path(project_root).resolve() / OUTPUTS_FOLDER:
        raise ValueError(f"Workspace output folder cannot be a link: {folder}")
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


def _output_name(name: str) -> Path:
    """Validate an output basename independently of the host platform.

    Reject path separators, Windows drive or stream syntax, reserved device
    names, and names that Windows normalizes to a different spelling.

    Parameters
    ----------
    name: str
        Requested file name, without any directory components.

    Returns
    -------
    pathlib.Path
        Validated relative basename.

    Raises
    ------
    ValueError
        If the name cannot be used as a single output filename.

    """

    if not isinstance(name, str) or not name:
        raise ValueError("Output name must be a single filename.")
    if (
        name in (".", "..")
        or any(char in name for char in '/\\:*?"<>|\x00')
        or any(ord(char) < 32 for char in name)
        or name.endswith((" ", "."))
        or Path(name).name != name
    ):
        raise ValueError("Output name must be a single safe filename.")

    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    reserved.update(f"COM{number}" for number in "123456789¹²³")
    reserved.update(f"LPT{number}" for number in "123456789¹²³")
    if name.split(".")[0].upper() in reserved:
        raise ValueError("Output name is reserved on Windows.")

    return Path(name)


def _output_candidates(folder: Path, name: str) -> Iterator[Path]:
    """Yield the requested name followed by numbered alternatives.

    Parameters
    ----------
    folder: pathlib.Path
        Project-owned outputs directory.
    name: str
        Validated output basename.

    Yields
    ------
    pathlib.Path
        Candidate paths in the outputs directory.

    """

    relative = _output_name(name)
    yield folder / relative
    for number in count(2):
        yield folder / f"{relative.stem}-{number}{relative.suffix}"


@contextmanager
def staged_output_file(
    project_root: Path, *, suffix: str = ""
) -> Iterator[Path]:
    """Create and clean up a private staging file for generated output.

    The file is created exclusively in ``outputs/``. It is never published
    under its temporary name. Callers may overwrite the staging file, but
    they must use ``publish_output_file`` for the final destination.

    Parameters
    ----------
    project_root: pathlib.Path
        The project folder.
    suffix: str, optional
        Staging suffix required by the writer, such as ``".pdf"``.
        Defaults to ``""``.

    Yields
    ------
    pathlib.Path
        Caller-owned temporary file inside ``outputs/``.

    """

    _output_name("staging" + suffix)
    folder = output_folder(project_root)
    descriptor, name = tempfile.mkstemp(
        prefix=".kilin-output-", suffix=suffix, dir=folder
    )
    os.close(descriptor)
    staged = Path(name)
    try:
        yield staged
    finally:
        staged.unlink(missing_ok=True)


def _copy_output_exclusively(staged: Path, candidate: Path) -> bool:
    """Copy staged data without opening an occupied destination.

    This fallback supports filesystems that prohibit hard links. The
    destination is exclusively created, so no pre-existing content can be
    replaced, although a reader could observe a partial copy in progress.

    Parameters
    ----------
    staged: pathlib.Path
        Complete staged file.
    candidate: pathlib.Path
        Destination to create exclusively.

    Returns
    -------
    bool
        True when the file was created; False when its name is taken.

    """

    try:
        descriptor = os.open(
            candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError:
        return False

    try:
        with os.fdopen(descriptor, "wb") as destination:
            with staged.open("rb") as source:
                shutil.copyfileobj(source, destination)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise
    return True


def publish_output_file(
    project_root: Path, name: str, staged: Path
) -> Path:
    """Publish a completed staged output without replacing another file.

    Prefer a same-directory hard link, which atomically makes a completed
    file visible without replacing an existing name or dangling symlink.
    When hard links are unavailable, copy to an exclusively created file:
    that fallback still cannot overwrite, but is not an atomic publication.
    On collision, try the next numbered name.

    Parameters
    ----------
    project_root: pathlib.Path
        The project folder.
    name: str
        Desired output basename.
    staged: pathlib.Path
        Regular staging file created in the project's output directory.

    Returns
    -------
    pathlib.Path
        Published file path relative to the project.

    Raises
    ------
    ValueError
        If the name or staging-file location is unsafe.
    OSError
        If publication fails for reasons other than a name collision.

    """

    folder = output_folder(project_root)
    _output_name(name)
    temporary = Path(staged)
    if temporary.parent.resolve() != folder.resolve():
        raise ValueError("Staging file must belong to outputs/.")
    if not stat.S_ISREG(temporary.lstat().st_mode):
        raise ValueError("Staging output must be a regular file.")

    for candidate in _output_candidates(folder, name):
        try:
            os.link(temporary, candidate)
        except FileExistsError:
            continue
        except OSError:
            if not _copy_output_exclusively(temporary, candidate):
                continue
        return Path(OUTPUTS_FOLDER) / candidate.name

    raise RuntimeError("Could not select an output filename.")


def save_output_file(project_root: Path, name: str, text: str) -> Path:
    """Write generated text under a unique name without overwriting.

    Content is fully written to an exclusive staging file before its final
    name becomes visible. Concurrent saves receive different filenames.

    Parameters
    ----------
    project_root: pathlib.Path
        The project folder.
    name: str
        Output filename, such as ``paper_summary.md``.
    text: str
        UTF-8 content to save.

    Returns
    -------
    pathlib.Path
        Path relative to the project. A taken name receives ``-2``,
        ``-3``, etc., without replacing previous files or symlinks.

    """

    _output_name(name)
    with staged_output_file(project_root) as staged:
        staged.write_text(text, encoding="utf-8")
        return publish_output_file(project_root, name, staged)
