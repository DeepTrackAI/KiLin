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
- ``TRACE_FORMAT_VERSION`` and ``OPERATION_RECORDING_VERSION``

  Structural trace version and full command-recording contract version.
- ``ProjectTraceSequence``

  Allocate and persist project-wide event numbers.
- ``project_trace_directory``

  Resolve the managed trace directory for a workspace trace.
- ``write_project_trace``

  Update the shared project trace under short-lived ownership.
- ``project_trace_publication``

  Coordinate a configuration observation with a single publication lease.
- ``compact_project_configuration``

  Compare a new configuration with the latest persisted full observation.

Examples
--------
>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> from kilin._trace_project import ProjectTraceSequence
>>> with TemporaryDirectory() as directory:
...     counter = ProjectTraceSequence(Path(directory) / "logs" / "traces")
...     assert counter.next() == 1
...     assert counter.next() == 2

"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import json
from pathlib import Path
from time import monotonic, sleep
from typing import Final, TypeAlias

from kilin.atomic_writes import atomic_write_text
from kilin._trace_configuration import (
    CONFIGURATION_EVENT,
    CONFIGURATION_REFERENCE,
    configuration_bytes,
    configuration_reference,
)
from kilin.conversation_ownership import (
    ConversationFileLock,
    ConversationLockedError,
)

__all__ = [
    "TRACE_FORMAT_VERSION",
    "OPERATION_RECORDING_VERSION",
    "ProjectTraceSequence",
    "project_trace_directory",
    "ProjectTracePublisher",
    "compact_project_configuration",
    "project_trace_publication",
    "write_project_trace",
]

TRACE_FORMAT_VERSION: Final[int] = 1
# Mark fresh projects recorded with complete engine command provenance.
# Do not stamp existing projects as complete when they are reopened.
OPERATION_RECORDING_VERSION: Final[int] = 1
_SEQUENCE_NAME: Final[str] = ".sequence.json"
_LOCK_TIMEOUT: Final[float] = 5.0
_LOCK_POLL: Final[float] = 0.01

ProjectTracePublisher: TypeAlias = Callable[
    [str, Mapping[str, object], Mapping[str, object]], None
]


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
    Ordinary publication and sequence allocation take separate leases.
    Configuration observations may reserve a sequence while holding the
    project publication lease; no path holds the sequence lease while
    acquiring the project lease. The lock companion is never removed.

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


def _raise_scan_error(error: OSError) -> None:
    """Abort history discovery when a directory cannot be inspected.

    Parameters
    ----------
    error: OSError
        Directory traversal failure reported by ``Path.walk``.

    Raises
    ------
    OSError
        The original failure; inaccessible history is not absent history.

    """

    raise error


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
        """Remember the shared metadata path without writing it.

        Parameters
        ----------
        directory: pathlib.Path
            Managed trace directory of a single workspace.

        """

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

        Notes
        -----
        Missing-counter discovery recognizes project-numbered records under
        any supported JSON filename. Canonical trace names remain strict;
        malformed unrelated JSON is ignored. Unreadable or linked candidates
        and history directories cannot establish that history is absent, so
        they block initialization. Existing counters require no history scan.

        """

        if not self.path.exists():
            # Never recycle numbers if the counter disappeared. Older,
            # unversioned diagnostic traces do not certify a sequence.
            logs = self.path.parent.parent
            if logs.is_symlink():
                raise OSError(f"Linked trace directory: {logs}")
            for folder, directories, files in logs.walk(
                on_error=_raise_scan_error, follow_symlinks=False
            ):
                for name in directories + files:
                    candidate = folder / name
                    linked = candidate.is_symlink()
                    if linked and candidate.is_dir():
                        raise OSError(f"Linked trace directory: {candidate}")
                    if candidate.suffix.lower() != ".json":
                        continue
                    if linked:
                        raise OSError(f"Linked trace file: {candidate}")
                    if candidate.name == "project.json" or (
                        candidate.name.startswith("conversation_")
                    ):
                        payload = _read_object(candidate)
                    else:
                        if not candidate.is_file():
                            continue
                        try:
                            payload = json.loads(
                                candidate.read_text(encoding="utf-8")
                            )
                        except (ValueError, UnicodeError):
                            # Unrelated logs need not be trace objects.
                            continue
                        if not isinstance(payload, dict):
                            continue
                    if payload.get("sequence_scope") == "project":
                        raise OSError(
                            "Project trace sequence counter is missing."
                        )
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


def compact_project_configuration(
    previous: Mapping[str, object], configuration: dict, sequence: int,
) -> dict:
    """Compare a configuration with the latest published complete snapshot.

    Parameters
    ----------
    previous: mapping of str to object
        Shared project trace read under the current publication lease.
    configuration: dict
        Complete JSON-ready effective configuration for the new event.
    sequence: int
        Newly reserved sequence number of the current observation.

    Returns
    -------
    dict
        Original full configuration or a direct reference to the latest
        equal full configuration. Ineligible or ambiguous history is not
        used for compaction.

    Notes
    -----
    Inspect only project-level operation nodes, not arbitrary event payloads.
    Session names are not chronological across logger runs, so compare
    sequence numbers. No disk reads or writes are performed here.

    """

    latest = None
    seen = set()
    sessions = previous.get("sessions", {})
    if not isinstance(sessions, dict):
        return configuration
    for session in sessions.values():
        if (not isinstance(session, dict)
                or session.get("category") != "project"):
            continue
        operations = session.get("operations", {})
        if not isinstance(operations, dict):
            return configuration
        for node in operations.values():
            if not isinstance(node, dict):
                continue
            event = node.get("event")
            if (not isinstance(event, dict)
                    or event.get("event_type") != CONFIGURATION_EVENT):
                continue
            number = event.get("sequence")
            details = node.get("details")
            if (type(number) is not int or not 0 < number < sequence
                    or number in seen or not isinstance(details, dict)
                    or event.get("status") != "completed"
                    or "turn_id" in event):
                return configuration
            seen.add(number)
            if CONFIGURATION_REFERENCE in details:
                continue
            if latest is None or number > latest[0]:
                latest = (number, details)
    if latest is not None:
        try:
            encoded = configuration_bytes(configuration)
            if encoded == configuration_bytes(latest[1]):
                return configuration_reference(latest[0], encoded)
        except (ValueError, TypeError, RecursionError):
            # Keep full evidence rather than guessing when a configuration
            # cannot be compared. Inspection still validates the history.
            pass
    return configuration


@contextmanager
def project_trace_publication(
    path: Path,
) -> Iterator[tuple[dict, ProjectTracePublisher]]:
    """Read and publish a shared project trace under one ownership lease.

    Parameters
    ----------
    path: pathlib.Path
        Shared ``project.json`` trace path.

    Yields
    ------
    tuple of dict and ProjectTracePublisher
        Persisted trace (empty on first publication) and a publisher for one
        logger run. The publisher is used only within this context.

    Notes
    -----
    The configuration event reserves its sequence and chooses its payload
    inside this context. This prevents another configuration writer from
    publishing an intervening full snapshot. Other operations continue to
    allocate numbers before separately acquiring the publication lease.

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

        def publish(
            trace_id: str,
            sessions: Mapping[str, object],
            metadata: Mapping[str, object],
        ) -> None:
            """Merge this run and atomically replace the owned project trace.

            Parameters
            ----------
            trace_id: str
                Unique identifier of the publishing logger run.
            sessions: mapping of str to object
                Current hierarchical sessions of that run.
            metadata: mapping of str to object
                Trace header and current run's event counts.

            """

            saved = dict(previous.get("sessions", {}))
            # Preserve other runs, including their configuration snapshots.
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

        yield previous, publish


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

    with project_trace_publication(path) as (_, publish):
        publish(trace_id, sessions, metadata)
