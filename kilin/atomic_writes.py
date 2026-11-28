"""Write local text files through atomic replacement or exclusive publication.

A uniquely created temporary file in the destination directory avoids
collisions and following an existing predictable temporary-file symlink.
atomic_write_text replaces an existing file; atomic_create_text instead
publishes a new entry and fails if any entry already exists. Failed writes
remove their temporary files without replacing a protected destination.

The caller remains responsible for creating the destination directory and
coordinating concurrent writers. Atomic replacement does not provide a
multi-file transaction or guarantee durability across sudden power loss.

Key Features
------------

- **Isolated temporary files**

    Use unpredictable sibling filenames for every write.

- **Atomic single-file replacement**

    Preserve the prior destination if writing or replacement fails.

Module Structure
----------------

- ``atomic_write_text``

    Write UTF-8 text through a temporary file and replace one destination.

- ``atomic_create_text``

    Publish a new UTF-8 file atomically without replacing any existing entry.

Examples
--------
>>> from pathlib import Path
>>> from kilin.atomic_writes import atomic_write_text
>>> path = Path("/path/to/existing/folder/record.json")
>>> atomic_write_text(path, "{}\\n")

"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic, sleep
from typing import Final

__all__ = ["atomic_write_text", "atomic_create_text"]

_WINDOWS_REPLACE_TIMEOUT: Final[float] = 1.0
_WINDOWS_REPLACE_POLL: Final[float] = 0.01


def _replace(temporary: Path, path: Path) -> None:
    """Retry briefly when Windows denies replacement of a closed file.

    Parameters
    ----------
    temporary: pathlib.Path
        Fully written, closed temporary file owned by the caller.
    path: pathlib.Path
        Destination to replace atomically.

    Notes
    -----
    Windows readers and file scanners can deny a rename even when all
    cooperating writers obey the application's file lease. Only Windows
    access-denied, sharing-violation and lock-violation errors are retried.
    The caller retains its existing lease and the same temporary file;
    persistent failures propagate without deleting the destination.

    """

    deadline = monotonic() + _WINDOWS_REPLACE_TIMEOUT
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise
            sleep(min(_WINDOWS_REPLACE_POLL, remaining))


def atomic_write_text(path: Path, content: str) -> None:
    """Replace one text file using a unique sibling temporary file.

    Parameters
    ----------
    path: pathlib.Path
        Destination file. Its parent directory must already exist.
    content: str
        Text to write using UTF-8 encoding.

    Raises
    ------
    OSError
        If creating, writing, or replacing the file fails.

    Notes
    -----
    The operating system creates a new exclusive temporary file, avoiding
    conflicts with pre-existing ``.tmp`` files and symbolic links. The
    temporary file is removed even if writing or replacement fails. This
    helper does not acquire a file lock or force data to stable storage.
    Transient Windows replacement denials are retried for at most one
    second; other errors and persistent denials propagate to the caller.

    """

    stream = NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(content)
        _replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_create_text(path: Path, content: str) -> None:
    """Publish complete text only when the destination does not exist.

    Parameters
    ----------
    path: pathlib.Path
        New file whose parent already exists.
    content: str
        UTF-8 text to publish.

    Raises
    ------
    FileExistsError
        If any destination entry already exists, including a symbolic link.
    OSError
        If writing or linking fails, including unsupported filesystems.

    Notes
    -----
    A same-directory hard link publishes the fully written temporary file
    without a check-then-replace race. Unsupported hard links fail closed;
    this helper never falls back to overwriting. It does not force data to
    stable storage or provide a multi-file transaction.

    """

    stream = NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(content)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
