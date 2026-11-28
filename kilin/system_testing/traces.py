"""Read recorded project events without opening a live project session.

Key Features
------------
- Decode only the diagnostic hierarchy, including compact span completions.
- Inspect active and archived conversations without silent discovery skips.
- Preserve raw event identity and report malformed records without repair.
- Expand verified configuration references for downstream consumers.

Module Structure
----------------
- ``TraceEvent`` and ``Recording`` describe the decoded recording.
- ``ValidationIssue`` describes a problem at a file or event location.
- ``read_recording`` loads the supported recording format.

Examples
--------
>>> from kilin.system_testing.traces import TraceEvent
>>> TraceEvent.__name__
'TraceEvent'

"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path, PureWindowsPath

from kilin._trace_configuration import (
    CONFIGURATION_EVENT,
    CONFIGURATION_REFERENCE,
    resolve_configuration_reference,
)
from kilin._trace_project import TRACE_FORMAT_VERSION
from kilin.conversation_format import conversation_state
from kilin.conversations import _validated_metadata
from kilin.conversation_memory.models import require_saved_turn_ids

__all__ = ["TraceEvent", "ValidationIssue", "Recording", "read_recording"]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One independently identifiable recording problem or limitation.

    Attributes
    ----------
    code: str
        Stable machine-readable issue category.
    message: str
        Explanation intended for the researcher.
    path: str | None
        Workspace-relative file location when available.
    sequence: int | None
        Original event number when available.

    """

    code: str
    message: str
    path: str | None = None
    sequence: int | None = None


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One original event, including an expanded span terminal event.

    Attributes
    ----------
    path: str
        Workspace-relative trace file, including archival placement.
    location: str
        Location within the diagnostic JSON hierarchy.
    metadata: dict
        Original event metadata, not the readable span summary.
    details: dict
        Input, result, or sampled progress payload. Valid compact project
        configuration observations expose their full resolved payload.
    exact_content: dict
        Separately retained artifact contents, if present.

    """

    path: str
    location: str
    metadata: dict
    details: dict
    exact_content: dict = field(default_factory=dict)

    @property
    def sequence(self) -> int:
        """Return the original workspace-wide allocation number."""
        return self.metadata["sequence"]

    @property
    def event_type(self) -> str:
        """Return the original event type, not a display name."""
        return self.metadata["event_type"]

    @property
    def span_id(self) -> str | None:
        """Return the event's span correlation identifier, if available."""
        return self.metadata.get("span_id")


@dataclass(slots=True)
class Recording:
    """Internal read-only snapshot of trace payloads and decoding issues.

    Attributes
    ----------
    root: pathlib.Path
        Original workspace root.
    events: list[TraceEvent]
        Decodable events in allocation order.
    states: dict[str, dict]
        Canonical conversation resume states by relative trace path.
    issues: list[ValidationIssue]
        Problems encountered during decoding.
    last_sequence: int | None
        Recorded counter high-water mark, if valid.

    """

    root: Path
    events: list[TraceEvent] = field(default_factory=list)
    states: dict[str, dict] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)
    last_sequence: int | None = None

    def issue(self, code: str, message: str, path: str | None = None,
              sequence: int | None = None) -> None:
        """Collect an issue without concealing other decodable records.

        Parameters
        ----------
        code: str
            Stable problem category.
        message: str
            Human-readable explanation.
        path: str | None, optional
            Workspace-relative file location.
        sequence: int | None, optional
            Event number if known.

        """
        self.issues.append(ValidationIssue(code, message, path, sequence))


def safe_path(root: Path, relative: str) -> Path:
    """Resolve a normalized relative path without following child links.

    Parameters
    ----------
    root: pathlib.Path
        Resolved workspace root.
    relative: str
        Untrusted workspace-relative POSIX path.

    Returns
    -------
    pathlib.Path
        Safe child path, which need not exist.

    """
    if (not isinstance(relative, str) or not relative
            or "\\" in relative or "\x00" in relative
            or PureWindowsPath(relative).drive
            or any(p in ("", ".", "..") for p in relative.split("/"))):
        raise ValueError("Expected a normalized workspace-relative path.")
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Linked workspace path: {relative}")
    return path


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    """Reject ambiguous JSON objects instead of using the last value.

    Parameters
    ----------
    pairs: list of tuple
        Ordered decoded JSON object members.

    Returns
    -------
    dict
        Unambiguous object.

    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON member: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    """Reject non-JSON numeric constants.

    Parameters
    ----------
    value: str
        Invalid numeric constant.

    """
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_json(root: Path, relative: str) -> dict:
    """Read one regular JSON object without modifying its file.

    Parameters
    ----------
    root: pathlib.Path
        Workspace root.
    relative: str
        Normalized relative file location.

    Returns
    -------
    dict
        Decoded object.

    """
    path = safe_path(root, relative)
    if not path.is_file():
        raise ValueError(f"Missing or nonregular file: {relative}")
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream, object_pairs_hook=_unique_object,
                           parse_constant=_invalid_constant)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")
    return result


def _object(value: object) -> dict:
    """Require an object at a known structural position.

    Parameters
    ----------
    value: object
        Decoded JSON value.

    Returns
    -------
    dict
        Checked object.

    """
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def _event(recording: Recording, path: str, location: str,
           metadata: object, details: object, exact: object) -> None:
    """Decode one event and retain its source location.

    Parameters
    ----------
    recording: Recording
        Snapshot receiving events and issues.
    path: str
        Trace file location.
    location: str
        Diagnostic hierarchy location.
    metadata: object
        Raw compact event metadata.
    details: object
        Raw event data.
    exact: object
        Raw separately retained content.

    """
    try:
        meta = dict(_object(metadata))
        seq = meta.get("sequence")
        if type(seq) is not int or seq < 1:
            raise ValueError("Invalid event sequence.")
        for key in ("event_type", "trace_id", "event_id", "component",
                    "status", "timestamp_utc"):
            if not isinstance(meta.get(key), str) or not meta[key]:
                raise ValueError(f"Missing event metadata: {key}")
        stamp = datetime.fromisoformat(meta["timestamp_utc"])
        if stamp.utcoffset() != timedelta(0):
            raise ValueError("Expected a UTC event timestamp.")
        elapsed = meta.get("elapsed_ms")
        if (type(elapsed) not in (int, float) or elapsed < 0
                or not math.isfinite(elapsed)):
            raise ValueError("Invalid event elapsed time.")
        for key in ("turn_id", "span_id", "parent_span_id"):
            if key in meta and (not isinstance(meta[key], str)
                                or not meta[key]):
                raise ValueError(f"Invalid event correlation: {key}")
        # The writer substitutes running/completed/failed for the original
        # start status. Preserve the original rather than the summary.
        if "started_status" in meta:
            status = meta.pop("started_status")
            if not isinstance(status, str) or not status:
                raise ValueError("Invalid original start status.")
            meta["status"] = status
        meta.pop("finished_at_utc", None)
        recording.events.append(TraceEvent(
            path, location, meta, _object(details), _object(exact)
        ))
    except (ValueError, TypeError, OverflowError) as error:
        recording.issue("event.invalid", f"{location}: {error}", path)


def _operations(recording: Recording, path: str, location: str,
                operations: object, turn_id: str | None = None) -> None:
    """Decode an operation mapping and its compact lifecycle terminals.

    Parameters
    ----------
    recording: Recording
        Snapshot receiving events and issues.
    path: str
        Trace file location.
    location: str
        Parent diagnostic hierarchy location.
    operations: object
        Operation mapping.
    turn_id: str | None, optional
        Enclosing turn identity, when in a turn stage.

    """
    for key, value in _object(operations).items():
        here = f"{location}/{key}"
        try:
            node = _object(value)
            meta = _object(node.get("event"))
            if meta.get("turn_id") != turn_id:
                raise ValueError("Event is in the wrong turn group.")
            _event(recording, path, here + "/event", meta,
                   node.get("details", {}), node.get("exact_content", {}))
            if "completion" in node:
                end = _object(node["completion"])
                start_type = meta.get("event_type")
                if (not isinstance(start_type, str)
                        or not start_type.endswith(".started")
                        or end.get("event_type") not in (
                            start_type[:-8] + ".completed",
                            start_type[:-8] + ".failed")
                        or any(end.get(k) != meta.get(k) for k in (
                            "span_id", "trace_id", "turn_id", "component",
                            "parent_span_id"))
                        or not meta.get("span_id")
                        or type(end.get("sequence")) is not int
                        or type(meta.get("sequence")) is not int
                        or end["sequence"] <= meta["sequence"]):
                    raise ValueError("Mismatched span completion.")
                _event(recording, path, here + "/completion", end,
                       node.get("result", {}), {})
            elif "started_status" in meta:
                recording.issue("span.incomplete", "Unfinished span.", path,
                                meta.get("sequence"))
        except (ValueError, TypeError) as error:
            recording.issue("hierarchy.invalid", f"{here}: {error}", path)


def _document(recording: Recording, path: str, payload: dict) -> None:
    """Read only documented diagnostic locations and canonical state.

    Parameters
    ----------
    recording: Recording
        Snapshot receiving events, states, and issues.
    path: str
        Trace file location.
    payload: dict
        Decoded trace document.

    """
    project = path == "logs/traces/project.json"
    if (type(payload.get("trace_format_version")) is not int
            or payload["trace_format_version"] != TRACE_FORMAT_VERSION
            or payload.get("sequence_scope") != "project"
            or payload.get("trace_scope") != (
                "project" if project else "conversation")):
        recording.issue("trace.format", "Unsupported trace format or scope.",
                        path)
    if payload.get("trace_incomplete") is not False:
        recording.issue("trace.incomplete", "Trace health is not complete.",
                        path)
    if not project:
        try:
            state = conversation_state(payload)
            _validated_metadata(state)
            require_saved_turn_ids(state.get("turns", []))
            recording.states[path] = state
        except (ValueError, TypeError) as error:
            recording.issue("conversation.invalid", str(error), path)
    for key, value in _object(payload.get("sessions")).items():
        here = f"sessions/{key}"
        try:
            session = _object(value)
            if project and "resume_state" in session:
                raise ValueError("Project trace contains conversation state.")
            _operations(recording, path, here + "/operations",
                        session.get("operations", {}))
            for name, raw in _object(session.get("turns", {})).items():
                turn = _object(raw)
                tid = turn.get("turn_id")
                if not isinstance(tid, str) or not tid:
                    raise ValueError("Missing diagnostic turn identity.")
                for stage, group in _object(turn.get("stages")).items():
                    _operations(
                        recording, path,
                        f"{here}/turns/{name}/stages/{stage}/operations",
                        _object(group).get("operations"), tid,
                    )
            checkpoints = _object(session.get("progress_checkpoints", {}))
            for name, raw in checkpoints.items():
                progress = _object(raw)
                keys = ("sequence", "timestamp_utc", "elapsed_ms",
                        "trace_id", "event_id", "event_type", "component",
                        "status", "turn_id", "span_id", "parent_span_id",
                        "started_elapsed_ms", "duration_ms")
                meta = {k: progress[k] for k in keys if k in progress}
                details = {k: v for k, v in progress.items() if k not in keys}
                _event(recording, path,
                       f"{here}/progress_checkpoints/{name}",
                       meta, details, {})
        except (ValueError, TypeError) as error:
            recording.issue("hierarchy.invalid", f"{here}: {error}", path)


def _trace_paths(recording: Recording) -> list[str]:
    """Discover every JSON candidate without silently skipping archives.

    Parameters
    ----------
    recording: Recording
        Workspace and issue collector.

    Returns
    -------
    list[str]
        Sorted relative trace paths, excluding the sequence counter.

    """
    root = recording.root
    directory = safe_path(root, "logs/traces")
    if not directory.is_dir():
        raise ValueError("Missing trace directory.")
    paths = []

    def on_error(error: OSError) -> None:
        """Report an inaccessible history directory.

        Parameters
        ----------
        error: OSError
            Directory traversal error.

        """
        recording.issue("trace.discovery", str(error), "logs/traces")

    for parent, dirs, files in os.walk(directory, onerror=on_error):
        dirs.sort()
        for name in sorted(dirs):
            child = Path(parent) / name
            if child.is_symlink():
                recording.issue("trace.discovery", "Linked trace directory.",
                                child.relative_to(root).as_posix())
                dirs.remove(name)
        for name in sorted(files):
            relative = (Path(parent) / name).relative_to(root).as_posix()
            if name.lower().endswith(".json") and relative != (
                "logs/traces/.sequence.json"
            ):
                paths.append(relative)
    return sorted(paths)


def _resolve_configurations(recording: Recording) -> None:
    """Expand compact configuration observations in chronological order.

    Parameters
    ----------
    recording: Recording
        Decoded recording sorted by project-wide sequence number. Invalid
        references contribute blocking issues; original files stay untouched.

    Notes
    -----
    Track the most recent full project configuration, not the most recent
    reference. Legacy full observations pass through unchanged. A reference
    must belong to the same project trace and cannot target a later event,
    a different event type, an earlier superseded snapshot, or another
    reference.

    """

    latest = None
    for index, event in enumerate(recording.events):
        if event.event_type != CONFIGURATION_EVENT:
            continue
        # A duplicate allocation cannot identify a unique reference target.
        duplicate = (
            (index > 0
             and recording.events[index - 1].sequence == event.sequence)
            or (index + 1 < len(recording.events)
                and recording.events[index + 1].sequence == event.sequence)
        )
        try:
            if CONFIGURATION_REFERENCE not in event.details:
                if event.path == "logs/traces/project.json":
                    if (not duplicate
                            and event.metadata.get("status") == "completed"
                            and "turn_id" not in event.metadata):
                        latest = (event.sequence, event.details)
                    else:
                        latest = None
                continue
            if duplicate:
                raise ValueError("Ambiguous configuration event sequence.")
            if (event.path != "logs/traces/project.json"
                    or event.metadata.get("status") != "completed"
                    or "turn_id" in event.metadata):
                raise ValueError(
                    "Reference is not a completed project observation."
                )
            details = resolve_configuration_reference(
                event.details, event.sequence, latest
            )
            recording.events[index] = replace(event, details=details)
        except (ValueError, TypeError, RecursionError) as error:
            recording.issue("configuration.reference", str(error),
                            event.path, event.sequence)


def read_recording(root: Path) -> Recording:
    """Read the project counter and all trace documents independently.

    Parameters
    ----------
    root: pathlib.Path
        Reference workspace, already resolved by the caller.

    Returns
    -------
    Recording
        Decoded events, canonical states, and structural issues. No project
        files or lock companions are created, updated, or repaired.

    """
    recording = Recording(root)
    counter = "logs/traces/.sequence.json"
    try:
        data = read_json(root, counter)
        last = data.get("last_sequence")
        if (type(data.get("format_version")) is not int
                or data["format_version"] != TRACE_FORMAT_VERSION
                or type(last) is not int or last < 1):
            raise ValueError("Invalid shared sequence counter.")
        recording.last_sequence = last
        if data.get("incomplete") is not False:
            recording.issue("counter.incomplete", "Recording is incomplete.",
                            counter)
    except (OSError, ValueError, RecursionError) as error:
        recording.issue("counter.invalid", str(error), counter)
    try:
        paths = _trace_paths(recording)
    except (OSError, ValueError) as error:
        recording.issue("trace.discovery", str(error), "logs/traces")
        return recording
    if "logs/traces/project.json" not in paths:
        recording.issue("trace.missing", "Missing shared project trace.")
    for path in paths:
        try:
            _document(recording, path, read_json(root, path))
        except (OSError, ValueError, TypeError, RecursionError) as error:
            recording.issue("trace.invalid", str(error), path)
    recording.events.sort(key=lambda event: event.sequence)
    _resolve_configurations(recording)
    return recording
