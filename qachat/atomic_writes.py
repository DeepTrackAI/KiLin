"""Write local text files through isolated, atomic replacements.

A uniquely created temporary file in the destination directory avoids
collisions and following an existing predictable temporary-file symlink.
Each successful write replaces one file atomically. Failed writes remove the
temporary file without replacing the previous destination.

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

Examples
--------
>>> from pathlib import Path
>>> from qachat.atomic_writes import atomic_write_text
>>> path = Path("/path/to/existing/folder/record.json")
>>> atomic_write_text(path, "{}\\n")

"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

__all__ = ["atomic_write_text"]


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

    """

    stream = NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
