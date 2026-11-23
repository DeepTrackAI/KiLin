"""Persist rebuildable conversation-memory vectors and occurrence records.

The index is derived from active conversation checkpoints, project policy,
and authoritative feedback. Its checksum detects corruption, while a separate
file lease and atomic replacement protect concurrent writers. A corrupted
index can be rebuilt; no feedback or conversation checkpoint is modified.

Key Features
------------

- **Independent derived state**

    Keep memory vectors out of primary-source indexes and feedback storage.

- **Validated atomic persistence**

    Reject invalid or linked paths and checksum mismatches on read.

Module Structure
----------------

- ``read_memory_index``

    Read a validated index payload, or return None if absent or corrupt.

- ``write_memory_index``

    Atomically publish one checksum-protected index payload.

Examples
--------
>>> from kilin.conversation_memory.persistence import memory_index_path
>>> memory_index_path('/project').name
'conversation_memory.json'

"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Final

from kilin.atomic_writes import atomic_write_text
from kilin.conversation_ownership import conversation_write_ownership

__all__ = [
    "memory_index_path",
    "read_memory_index",
    "write_memory_index",
]

_FORMAT: Final[str] = "conversation-memory-index"
_MAX_BYTES: Final[int] = 128 * 1024 * 1024


def memory_index_path(project_directory: str | Path) -> Path:
    """Return the derived memory-index path for a workspace.

    Parameters
    ----------
    project_directory: str | pathlib.Path
        Project root, not its source directory.

    Returns
    -------
    pathlib.Path
        Project-owned index location.

    """

    return Path(project_directory).resolve() / "indexes" / (
        "conversation_memory.json"
    )


def _checked_path(path: Path) -> None:
    """Reject a linked or invalid index location before accessing it.

    Parameters
    ----------
    path: pathlib.Path
        Derived index filename.

    Raises
    ------
    ValueError
        If the index or its parent directory is a link or invalid.

    """

    directory = path.parent
    if directory.is_symlink() or (
        directory.exists() and not directory.is_dir()
    ):
        raise ValueError("The project indexes directory is invalid.")
    if path.is_symlink():
        raise ValueError("The conversation memory index cannot be a link.")


def _checksum(data: dict[str, object]) -> str:
    """Compute the canonical checksum of a derived payload.

    Parameters
    ----------
    data: dict[str, object]
        Index contents without their checksum.

    Returns
    -------
    str
        Lowercase SHA-256 digest of canonical UTF-8 JSON.

    """

    encoded = json.dumps(
        data, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def read_memory_index(path: Path) -> dict[str, object] | None:
    """Read a valid derived index or request reconstruction.

    Parameters
    ----------
    path: pathlib.Path
        Workspace-local memory index.

    Returns
    -------
    dict[str, object] | None
        Verified payload excluding the checksum, or None if missing,
        corrupt, or structurally unsupported.

    Raises
    ------
    ValueError
        If the filesystem path violates ownership constraints.

    """

    _checked_path(path)
    if not path.exists():
        return None
    if not path.is_file() or path.stat().st_size > _MAX_BYTES:
        return None
    try:
        content = path.read_bytes()
        if len(content) > _MAX_BYTES:
            return None
        loaded = json.loads(content)
        if not isinstance(loaded, dict):
            return None
        expected = loaded.pop("checksum", None)
        if (
            set(loaded) != {
                "format", "model_identity", "manifest", "policy_digest",
                "feedback_revision", "records", "vectors",
            }
            or loaded["format"] != _FORMAT
            or not isinstance(expected, str)
            or expected != _checksum(loaded)
        ):
            return None
        return loaded
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        return None


def write_memory_index(path: Path, data: dict[str, object]) -> None:
    """Atomically replace a derived index under project file ownership.

    Parameters
    ----------
    path: pathlib.Path
        Index location inside its owning project.
    data: dict[str, object]
        Validated derived state without format or checksum fields.

    Raises
    ------
    ValueError
        If the target is a link, the payload is invalid, or too large.
    OSError
        If directory creation or atomic persistence fails.

    """

    _checked_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {"format": _FORMAT, **data}
    payload = {**content, "checksum": _checksum(content)}
    rendered = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, indent=2,
        allow_nan=False,
    ) + "\n"
    if len(rendered.encode("utf-8")) > _MAX_BYTES:
        raise ValueError("The conversation memory index is too large.")
    with conversation_write_ownership(path):
        _checked_path(path)
        atomic_write_text(path, rendered)
