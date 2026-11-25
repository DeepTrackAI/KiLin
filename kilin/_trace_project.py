"""Coordinate project-wide numbering and shared project-trace persistence.

Conversation traces retain their exclusive, long-lived write leases. The
project trace and its small sequence counter instead use short-lived leases:
no process owns the entire workspace for the duration of a conversation.
The counter contains only allocation and recording-health metadata, not a
second copy of any diagnostic event.

Key Features
------------
- **Project-wide sequence allocation**

  Allocate positive, monotonically increasing numbers across logger runs
  and processes, including archived conversations.
- **Shared project trace**

  Publish one logger run's hierarchical sessions without overwriting the
  sessions of another writer.

Module Structure
----------------
- ``ProjectTraceSequence``

  Allocate and persist project-wide event numbers.
- ``project_trace_directory``

  Resolve the managed trace directory for a workspace trace.
- ``write_project_trace``

  Update the shared project trace under short-lived ownership.

"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import json
from pathlib import Path
from time import monotonic, sleep
from typing import Final

from kilin.atomic_writes import atomic_write_text
from kilin.conversation_ownership import (
    ConversationFileLock,
    ConversationLockedError,
)

__all__ = [
    "TRACE_FORMAT_VERSION",
    "ProjectTraceSequence",
    "project_trace_directory",
    "write_project_trace",
]

TRACE_FORMAT_VERSION: Final[int] = 1
_SEQUENCE_NAME: Final[str] = ".sequence.json"
_LOCK_TIMEOUT: Final[float] = 5.0
_LOCK_POLL: Final[float] = 0.01


def project_trace_directory(
    output_path: Path, project_directory: str | Path | None = None
) -> Path | None:
    """Find the common trace directory for a workspace-owned trace.

    Parameters
    ----------
    output_path: pathlib.Path
        Path of the trace to be written.
    project_directory: str | pathlib.Path | None, optional
        Explicit workspace root, including a reserved project that does
        not yet have its profile. Otherwise infer an existing workspace
        from its logs directory and profile.

    Returns
    -------
    pathlib.Path | None
        Common ``logs/traces`` directory, or None for a standalone logger.

    Raises
    ------
    ValueError
        If an explicit workspace does not own the trace, or a managed
        directory or trace is a symbolic link.

    """

    path = output_path.expanduser().absolute()
    if project_directory is None:
        root = next(
            (
                parent.parent
                for parent in path.parents
                if parent.name == "logs"
                and (parent.parent / "profile.ini").is_file()
            ),
            None,
        )
        if root is None:
            return None
    else:
        root = Path(project_directory).expanduser().resolve()
    logs = root / "logs"
    directory = logs / "traces"
    if logs.is_symlink() or directory.is_symlink() or path.is_symlink():
        raise ValueError("Project trace paths must not be symbolic links.")
    if not path.resolve().is_relative_to(logs.resolve()):
        raise ValueError("A project trace must stay inside project logs.")
    return directory.resolve()


@contextmanager
def _short_ownership(path: Path) -> Iterator[None]:
    """Acquire a short write lease without changing conversation ownership.

    Parameters
    ----------
    path: pathlib.Path
        Shared project metadata or project trace file.

    Yields
    ------
    None
        The caller may read and atomically replace the protected file.

    Raises
    ------
    OSError
        If ownership cannot be obtained within the bounded wait or the
        target is a symbolic link.

    Notes
    -----
    Neither this lease nor the sequence lease is held while another trace
    file is written. The persistent lock companion is never removed.

    """

    if path.is_symlink() or path.parent.is_symlink():
        raise OSError(f"Shared trace path cannot be a link: {path}")
    owner = ConversationFileLock(path)
    deadline = monotonic() + _LOCK_TIMEOUT
    while True:
        try:
            owner.acquire()
        except ConversationLockedError as error:
            if monotonic() >= deadline:
                raise OSError(
                    f"Timed out locking trace file: {path}"
                ) from error
            sleep(_LOCK_POLL)
        else:
            break
    try:
        if path.is_symlink():
            raise OSError(f"Shared trace path cannot be a link: {path}")
        yield
    finally:
        owner.release()


def _read_object(path: Path) -> dict[str, object]:
    """Read a JSON object without silently replacing malformed data.

    Parameters
    ----------
    path: pathlib.Path
        Existing regular JSON file.

    Returns
    -------
    dict[str, object]
        Decoded object.

    Raises
    ------
    OSError
        If the file is not readable or its JSON is not an object.

    """

    if path.is_symlink() or not path.is_file():
        raise OSError(f"Invalid project trace file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise OSError(f"Invalid project trace JSON: {path}") from error
    if not isinstance(value, dict):
        raise OSError(f"Invalid project trace JSON: {path}")
    return value


class ProjectTraceSequence:
    """Allocate a shared event number for a workspace trace.

    Parameters
    ----------
    directory: pathlib.Path
        Managed ``logs/traces`` directory of a single workspace.

    Methods
    -------
    next()
        Persist and return the next event number.
    mark_incomplete()
        Persist a recording-health flag after an optional trace failure.

    Notes
    -----
    The high-water mark is allocated before the event is published. Gaps
    may therefore identify failed or unflushed writes; allocated numbers
    are never reused. Counter corruption is not repaired automatically.
    The counter is small and does not contain event payloads.

    """

    def __init__(self, directory: Path) -> None:
        """Remember the shared metadata path without writing it."""

        self.path = directory / _SEQUENCE_NAME

    def _read(self) -> dict[str, object]:
        """Read the counter under ownership, or initialize a new one.

        Returns
        -------
        dict[str, object]
            Validated counter and recording-health metadata.

        Raises
        ------
        OSError
            If the counter is malformed or missing for a versioned project.

        """

        if not self.path.exists():
            # Never recycle numbers if the counter disappeared. Older,
            # unversioned diagnostic traces do not certify a sequence.
            logs = self.path.parent.parent
            for candidate in logs.rglob("*.json"):
                if candidate.name != "project.json" and not (
                    candidate.name.startswith("conversation_")
                ):
                    continue
                if candidate.is_symlink():
                    raise OSError(f"Linked trace file: {candidate}")
                payload = _read_object(candidate)
                if payload.get("sequence_scope") == "project":
                    raise OSError("Project trace sequence counter is missing.")
            return {
                "format_version": TRACE_FORMAT_VERSION,
                "last_sequence": 0,
                "incomplete": False,
            }
        state = _read_object(self.path)
        if (
            set(state) != {"format_version", "last_sequence", "incomplete"}
            or type(state["format_version"]) is not int
            or state["format_version"] != TRACE_FORMAT_VERSION
            or type(state["last_sequence"]) is not int
            or state["last_sequence"] < 0
            or type(state["incomplete"]) is not bool
        ):
            raise OSError("Invalid project trace sequence counter.")
        return state

    def next(self) -> int:
        """Reserve the next project-wide event number durably.

        Returns
        -------
        int
            Newly reserved positive sequence number.

        """

        with _short_ownership(self.path):
            state = self._read()
            number = int(state["last_sequence"]) + 1
            state["last_sequence"] = number
            atomic_write_text(self.path, json.dumps(state) + "\n")
            return number

    def mark_incomplete(self) -> None:
        """Record that at least one optional diagnostic write failed."""

        with _short_ownership(self.path):
            state = self._read()
            state["incomplete"] = True
            atomic_write_text(self.path, json.dumps(state) + "\n")


def write_project_trace(
    path: Path,
    trace_id: str,
    sessions: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    """Publish one run's hierarchical project-trace projection.

    Parameters
    ----------
    path: pathlib.Path
        Shared ``project.json`` trace path.
    trace_id: str
        Unique identifier of this logger run.
    sessions: mapping of str to object
        Hierarchical sessions projected from this run's events.
    metadata: mapping of str to object
        Trace header including the current run's event counts.

    Notes
    -----
    Stable run-specific session keys allow each writer to replace its own
    projection while preserving other writers' sessions. Read, validation,
    merge and atomic publication use one short lease. The project trace
    never contains a resumable conversation checkpoint.

    """

    with _short_ownership(path):
        previous = {}
        if path.exists():
            previous = _read_object(path)
            if (
                previous.get("trace_format_version") != TRACE_FORMAT_VERSION
                or previous.get("trace_scope") != "project"
                or previous.get("sequence_scope") != "project"
                or not isinstance(previous.get("sessions"), dict)
            ):
                raise OSError("Invalid shared project trace format.")
            saved = dict(previous["sessions"])
        else:
            saved = {}
        # Session keys have a numbered category prefix, so lexical sorting
        # retains the existing human-readable category order.
        for key, value in sessions.items():
            saved[f"{key}_{trace_id}"] = value
        payload = {
            **metadata,
            "trace_incomplete": bool(
                metadata.get("trace_incomplete")
                or previous.get("trace_incomplete")
            ),
            "sessions": dict(sorted(saved.items())),
        }
        atomic_write_text(
            path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )
