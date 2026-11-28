"""Validate a closed reference workspace without inference or mutation.

Key Features
------------
- Check sequence continuity, span identity, and recorded command boundaries.
- Verify retained sources against historical import and indexing records.
- Reconstruct conversation identities, checkpoints, and feedback state.
- Report detectable inconsistencies without repairing the reference.

Module Structure
----------------
- ``ValidationReport`` and ``InvalidRecordingError`` expose validation results.
- ``validate_recording`` checks the decoded history and retained workspace.

Examples
--------
>>> from kilin.system_testing import Experiment
>>> # report = Experiment(workspace).validate()
>>> # report.valid

"""

from __future__ import annotations

from configparser import ConfigParser, Error as ConfigError
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from kilin._trace_project import OPERATION_RECORDING_VERSION
from kilin.conversation_memory.feedback import FeedbackStore
from kilin.conversation_memory.models import (
    MessageFeedback,
    require_turn_uuid,
)
from kilin.project import open_project
from kilin.repository import _manifest_files
from kilin.workspace import _read_records

from .timeline import TimelineOperation, build_timeline, one
from .traces import (
    Recording, TraceEvent, ValidationIssue, read_json, safe_path,
)

__all__ = ["ValidationReport", "InvalidRecordingError", "validate_recording"]

_PROJECT = "logs/traces/project.json"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Result of inspecting a reference recording.

    Attributes
    ----------
    workspace: pathlib.Path
        Resolved reference workspace.
    issues: tuple[ValidationIssue, ...]
        Blocking recording errors, in deterministic discovery order.
    warnings: tuple[ValidationIssue, ...]
        Nonblocking limitations or environmental differences.
    event_count: int
        Number of successfully decoded original events.
    events: tuple[TraceEvent, ...]
        All decodable original events, including non-command diagnostics.
    operations: tuple[TimelineOperation, ...]
        Candidate command intervals, including unsuccessful commands. These
        are for inspection only when validation fails.

    """

    workspace: Path
    issues: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...]
    event_count: int
    events: tuple[TraceEvent, ...]
    operations: tuple[TimelineOperation, ...]

    @property
    def valid(self) -> bool:
        """Return whether no blocking issue was detected."""
        return not self.issues

    def raise_for_errors(self) -> None:
        """Reject invalid recordings before further processing."""
        if not self.valid:
            raise InvalidRecordingError(self)


class InvalidRecordingError(ValueError):
    """A reference cannot be used as a validated operation timeline.

    Attributes
    ----------
    report: ValidationReport
        Complete inspection result for diagnosing the rejection.

    """

    def __init__(self, report: ValidationReport) -> None:
        """Retain the validation report with a concise exception message.

        Parameters
        ----------
        report: ValidationReport
            Invalid reference inspection result.

        """
        self.report = report
        first = report.issues[0]
        super().__init__(f"Invalid recording ({len(report.issues)} issues): "
                         f"{first.code}: {first.message}")


def _require(condition: bool, message: str) -> None:
    """Require a recorded invariant without relying on Python assertions.

    Parameters
    ----------
    condition: bool
        Checked recording invariant.
    message: str
        Explanation if the check fails.

    """
    if not condition:
        raise ValueError(message)


def _text(value: object) -> str:
    """Require a nonempty recorded string identity.

    Parameters
    ----------
    value: object
        Untrusted recorded value.

    Returns
    -------
    str
        Nonempty identity.

    """
    if not isinstance(value, str) or not value:
        raise ValueError("Missing or invalid recorded string.")
    return value


def _strings(value: object) -> list[str]:
    """Require a list of distinct nonempty recorded strings.

    Parameters
    ----------
    value: object
        Untrusted recorded inventory.

    Returns
    -------
    list[str]
        Checked inventory in its original order.

    """
    if not isinstance(value, list):
        raise ValueError("Missing or invalid recorded list.")
    items = [_text(item) for item in value]
    _require(len(items) == len(set(items)), "Duplicate inventory entries.")
    return items


def _sequence_checks(recording: Recording) -> None:
    """Verify event numbers, identities, span links and logger lifetimes.

    Parameters
    ----------
    recording: Recording
        Snapshot receiving detected issues.

    """
    previous = 0
    identifiers = set()
    runs: dict[str, list[TraceEvent]] = {}
    spans: dict[str, TraceEvent] = {}
    finished: dict[str, TraceEvent] = {}
    for event in recording.events:
        seq = event.sequence
        meta = event.metadata
        if seq == previous:
            recording.issue("sequence.duplicate", "Repeated event sequence.",
                            event.path, seq)
        elif seq > previous + 1:
            recording.issue("sequence.gap",
                            f"Missing sequences {previous + 1}-{seq - 1}.",
                            event.path, seq)
        previous = seq
        if meta["event_id"] in identifiers:
            recording.issue("event.duplicate", "Repeated event identity.",
                            event.path, seq)
        identifiers.add(meta["event_id"])
        runs.setdefault(meta["trace_id"], []).append(event)
        if "started_elapsed_ms" in meta:
            span = event.span_id
            if event.event_type.endswith(".started"):
                if span is None or span in spans:
                    recording.issue("span.identity", "Invalid span identity.",
                                    event.path, seq)
                else:
                    spans[span] = event
            elif event.event_type.endswith((".completed", ".failed")):
                if span is None or span in finished:
                    recording.issue("span.identity", "Invalid span terminal.",
                                    event.path, seq)
                else:
                    finished[span] = event
    high = recording.last_sequence
    if high is not None and previous != high:
        recording.issue("sequence.counter", f"Last event {previous} differs "
                        f"from allocation counter {high}.")
    for span in sorted(set(spans) | set(finished)):
        start, end = spans.get(span), finished.get(span)
        if (start is None or end is None or start.sequence >= end.sequence
                or start.event_type[:-8] != end.event_type.rsplit(".", 1)[0]
                or start.path != end.path
                or start.metadata["trace_id"] != end.metadata["trace_id"]):
            event = start or end
            recording.issue("span.incomplete", "Missing or mismatched span.",
                            event.path, event.sequence)
    for event in recording.events:
        parent = event.metadata.get("parent_span_id")
        if parent is None:
            continue
        start, end = spans.get(parent), finished.get(parent)
        if (start is None or end is None or start.path != event.path
                or start.metadata["trace_id"] != event.metadata["trace_id"]
                or not start.sequence < event.sequence < end.sequence):
            recording.issue("span.parent", "Invalid parent span link.",
                            event.path, event.sequence)
    archived = {}
    for event in recording.events:
        if event.event_type == "gui.conversation.deleted" and (
                event.details.get("deleted_active") is True):
            path = event.details.get("archived_path")
            if isinstance(path, str):
                archived[path] = event.sequence
    latest = {}
    for run in runs.values():
        latest[run[-1].path] = max(latest.get(run[-1].path, 0),
                                   run[-1].sequence)
    for run in runs.values():
        first, last = run[0], run[-1]
        # Active deletion detaches the writer without appending a close to
        # the archived bytes. This is the only supported missing-close case.
        detached = (archived.get(last.path, -1) > last.sequence
                    and latest[last.path] == last.sequence)
        if (len({e.path for e in run}) != 1
                or first.event_type != "trace.started"
                or sum(e.event_type == "trace.started" for e in run) != 1
                or (last.event_type != "trace.closed" and not detached)
                or sum(e.event_type == "trace.closed" for e in run) != (
                    1 if last.event_type == "trace.closed" else 0)):
            recording.issue("trace.lifetime", "Unclosed or ambiguous logger "
                            "run.", first.path, first.sequence)


@dataclass(slots=True)
class _SourceHistory:
    """Accumulate actual imported copies and historical content identities."""

    root: Path
    files: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    origins: dict[str, tuple[str, str]] = field(default_factory=dict)
    hashes: dict[str, dict] = field(default_factory=dict)
    batches: dict[str, TraceEvent] = field(default_factory=dict)
    completed: set[str] = field(default_factory=set)
    copied: dict[str, set[str]] = field(default_factory=dict)

    def consume(self, event: TraceEvent) -> None:
        """Validate one source batch or historical manifest in order.

        Parameters
        ----------
        event: TraceEvent
            Original event whose payload may describe a source change.

        """
        kind, data = event.event_type, event.details
        if kind in ("workspace.prepare.started", "workspace.sync.started"):
            batch = _text(data.get("batch_id"))
            _require(batch not in self.batches, "Repeated source batch ID.")
            _require(not set(self.batches) - self.completed,
                     "Overlapping source batches.")
            self.batches[batch] = event
            self.copied[batch] = set()
            if kind == "workspace.prepare.started":
                _require(data.get("operation") in ("create", "import"),
                         "Unknown source preparation operation.")
                _require(data.get("replace_existing") is False,
                         "Reference imports must not replace sources.")
                _strings(data.get("source_paths"))
        elif kind in ("workspace.source.copied", "workspace.sync.file",
                      "workspace.prepare.completed",
                      "workspace.sync.completed", "workspace.sync.detached"):
            batch = _text(data.get("batch_id"))
            start = self.batches.get(batch)
            _require(start is not None and batch not in self.completed,
                     "Source event without an active batch.")
            if kind.endswith(".completed"):
                _require(kind[:-10] == start.event_type[:-8],
                         "Mismatched source batch completion.")
                if kind == "workspace.prepare.completed":
                    _require(data.get("operation") == start.details.get(
                        "operation"), "Mismatched preparation operation.")
                    # Copy records use resolved external identities, while
                    # the submitted paths may contain aliases. Never resolve
                    # original external paths during reference inspection.
                    _require(len(self.copied[batch]) <=
                             len(start.details["source_paths"]),
                             "More source copies than submitted paths.")
                    _require(type(data.get("source_count")) is int
                             and data["source_count"] == len(self.origins),
                             "Source count differs from recorded origins.")
                else:
                    changes = data.get("changes")
                    _require(isinstance(changes, dict), "Missing sync result.")
                    _require(set(_strings(changes.get("added"))) ==
                             self.copied[batch], "Sync result differs from "
                             "recorded imports.")
                    _require(not changes.get("changed") and not
                             changes.get("deleted") and not
                             changes.get("edited"), "Nonadditive source sync.")
                self.completed.add(batch)
            elif kind == "workspace.source.copied":
                _require(start.event_type == "workspace.prepare.started",
                         "Copy belongs to the wrong batch.")
                self._copy(data, batch)
            elif kind == "workspace.sync.file":
                _require(start.event_type == "workspace.sync.started",
                         "Sync copy belongs to the wrong batch.")
                self._sync(data, batch)
            else:
                raise ValueError("Source detachment is not additive.")
        elif kind in ("index.check.completed", "index.empty"):
            manifest = _manifest_files(data.get("source_manifest"))
            managed = {f"sources/{name}": record
                       for name, record in manifest.items()}
            _require(set(managed) == set(self.files),
                     "Index inventory differs from imported source files.")
            for path, record in managed.items():
                _require(path not in self.hashes or
                         self.hashes[path] == record,
                         f"Historical source content changed: {path}")
                self.hashes[path] = record

    def _copy(self, data: dict, batch: str) -> None:
        """Record actual file destinations, including renamed direct files.

        Parameters
        ----------
        data: dict
            Source copy payload.
        batch: str
            Active preparation batch identity.

        """
        source = _text(data.get("source"))
        destination = _text(data.get("workspace_path"))
        kind = data.get("kind")
        _require(kind in ("file", "folder"), "Invalid source kind.")
        _require(destination.startswith("sources/") and
                 len(destination.split("/")) == 2,
                 "Invalid managed source destination.")
        safe_path(self.root, destination)
        _require(source not in self.copied[batch], "Duplicate source copy.")
        self.copied[batch].add(source)
        previous = self.origins.get(source)
        _require(previous is None or previous == (destination, kind),
                 "Source destination or kind changed.")
        _require(not any(value[0] == destination and key != source
                         for key, value in self.origins.items()),
                 "Conflicting source ownership.")
        self.origins[source] = (destination, kind)
        copied = _strings(data.get("copied_files"))
        preserved = _strings(data.get("preserved_files"))
        _require(not set(copied) & set(preserved),
                 "File both copied and preserved.")
        selection = data.get("requested_files")
        requested = self.batches[batch].details.get("selected_files")
        if requested is not None:
            _require(isinstance(requested, dict) and
                     requested.get(source) == selection,
                     "Copy selection differs from batch request.")
        if selection is not None:
            _require(set(_strings(selection)) == set(copied + preserved),
                     "Copy result differs from exact selection.")
        if kind == "file":
            _require(len(copied + preserved) == 1 and
                     "/" not in (copied + preserved)[0],
                     "Direct import must identify exactly one file.")
        for relative in copied + preserved:
            safe_path(self.root, f"sources/{relative}")
            path = (destination if kind == "file" else
                    f"{destination}/{relative}")
            identity = (source, kind, relative)
            if relative in copied:
                _require(path not in self.files,
                         f"Historical source was overwritten: {path}")
                self.files[path] = identity
            else:
                _require(self.files.get(path) == identity,
                         f"Preserved source was never imported: {path}")

    def _sync(self, data: dict, batch: str) -> None:
        """Record an additive synchronization into an existing source.

        Parameters
        ----------
        data: dict
            Actual synchronization event payload.
        batch: str
            Active synchronization batch identity.

        """
        _require(data.get("action") == "import", "Nonadditive source sync.")
        path = _text(data.get("path"))
        original = _text(data.get("original"))
        relative = _text(data.get("relative"))
        origin = self.origins.get(original)
        _require(origin is not None, "Sync source has no recorded origin.")
        destination, kind = origin
        _require(kind == "folder" and data.get("kind") == kind
                 and path == f"{destination}/{relative}",
                 "Sync destination differs from original source ownership.")
        safe_path(self.root, path)
        _require(path not in self.files, "Sync overwrites a recorded source.")
        self.files[path] = (original, kind, relative)
        self.copied[batch].add(path)

    def finish(self, recording: Recording) -> None:
        """Compare final managed inventory and bytes with historical records.

        Parameters
        ----------
        recording: Recording
            Issue collector for mismatching final state.

        """
        if set(self.batches) != self.completed:
            recording.issue("batch.incomplete", "Unfinished source batch.")
        try:
            path = safe_path(self.root, "sources.json")
            if path.exists() or self.origins:
                raw = read_json(self.root, "sources.json")
                _require(type(raw.get("version")) is int,
                         "Invalid source manifest version.")
            records = _read_records(path)
            observed = {}
            origins = {}
            for record in records:
                source, name = record["original"], record["name"]
                kind = record.get("kind")
                destination = f"sources/{name}"
                origins[source] = (destination, kind)
                _require(not record.get("versions"),
                         "Reference contains replaced source versions.")
                for relative in record.get("files", []):
                    path = destination if kind == "file" else (
                        f"{destination}/{relative}")
                    observed[path] = (source, kind, relative)
            _require(origins == self.origins and observed == self.files,
                     "Source ownership manifest differs from recorded "
                     "imports.")
        except (OSError, ValueError, TypeError, KeyError) as error:
            recording.issue("source.inventory", str(error), "sources.json")
        found = set()
        try:
            directory = safe_path(self.root, "sources")
            _require(directory.is_dir(), "Missing managed source directory.")
            for parent, dirs, files in os.walk(directory, onerror=_raise_os):
                dirs.sort()
                for name in sorted(dirs + files):
                    path = Path(parent) / name
                    relative = path.relative_to(self.root).as_posix()
                    if path.is_symlink():
                        recording.issue("source.link", "Linked source path.",
                                        relative)
                        if name in dirs:
                            dirs.remove(name)
                    elif name in files:
                        found.add(relative)
                        if not path.is_file():
                            recording.issue("source.invalid", "Nonregular "
                                            "source file.", relative)
            for path in sorted(found - self.files.keys()):
                recording.issue("source.unrecorded", "Unrecorded source.",
                                path)
        except (OSError, ValueError) as error:
            recording.issue("source.discovery", str(error), "sources")
        for relative in sorted(self.files):
            try:
                path = safe_path(self.root, relative)
                _require(path.is_file(), "Missing retained source file.")
                expected = self.hashes.get(relative)
                _require(expected is not None,
                         "Source lacks a historical content manifest.")
                with path.open("rb") as stream:
                    before = os.fstat(stream.fileno())
                    digest = sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                    after = os.fstat(stream.fileno())
                fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns",
                          "st_ctime_ns")
                _require(all(getattr(before, field) == getattr(after, field)
                             for field in fields),
                         "Source changed during inspection.")
                _require(expected == {"sha256": digest.hexdigest(),
                                      "size_bytes": after.st_size},
                         "Retained source differs from recorded content.")
            except (OSError, ValueError) as error:
                recording.issue("source.content", str(error), relative)


def _raise_os(error: OSError) -> None:
    """Propagate a failed directory traversal.

    Parameters
    ----------
    error: OSError
        Directory scan error.

    """
    raise error


@dataclass(slots=True)
class _Conversation:
    """Reconstructed conversation identity and committed history."""

    path: str
    name: str
    parent: str | None = None
    origin_count: int | None = None
    origin_checkpoint: str | None = None
    turns: list[str] = field(default_factory=list)
    checkpoint: str | None = None
    deleted: bool = False


@dataclass(slots=True)
class _ConversationHistory:
    """Check conversation and feedback history against durable final state."""

    recording: Recording
    conversations: dict[str, _Conversation] = field(default_factory=dict)
    questions: dict[str, str] = field(default_factory=dict)
    checkpoints: set[str] = field(default_factory=set)
    ratings: dict[tuple[str, str], MessageFeedback] = field(
        default_factory=dict
    )
    feedback_revision: str = field(
        default_factory=lambda: sha256(b"").hexdigest()
    )
    active: str | None = None
    created: bool = False

    def _new(self, data: dict, *, parent: str | None = None) -> str:
        """Register a recorded new conversation and its inherited prefix.

        Parameters
        ----------
        data: dict
            Recorded conversation identity and lineage.
        parent: str | None, optional
            Expected parent of a branch.

        Returns
        -------
        str
            Newly registered conversation identifier.

        """
        identifier = _text(data.get("conversation_id"))
        _require(identifier not in self.conversations,
                 "Conversation identity reused for a new record.")
        path = _text(data.get("path"))
        safe_path(self.recording.root, path)
        _require(Path(path).parent.as_posix() == "logs/traces" and
                 Path(path).name.startswith("conversation_") and
                 path.endswith(".json"), "Invalid conversation path.")
        _require(not any(item.path == path for item in
                         self.conversations.values()),
                 "Conflicting conversation paths.")
        _require(data.get("parent_conversation_id") == parent,
                 "Unexpected conversation parent.")
        count = data.get("origin_turn_count")
        origin = data.get("origin_checkpoint_id")
        item = _Conversation(path, _text(data.get("name")), parent,
                             count, origin)
        if parent is not None:
            ancestor = self._known(parent)
            _require(type(count) is int and 0 <= count <= len(ancestor.turns)
                     and origin == ancestor.checkpoint,
                     "Branch origin does not match parent history.")
            item.turns = ancestor.turns[:count]
            if count == len(ancestor.turns):
                item.checkpoint = ancestor.checkpoint
        else:
            _require(count is None and origin is None,
                     "Root conversation has branch metadata.")
        self.conversations[identifier] = item
        return identifier

    def _known(self, identifier: object) -> _Conversation:
        """Resolve a live conversation identity at the current sequence.

        Parameters
        ----------
        identifier: object
            Recorded conversation ID.

        Returns
        -------
        _Conversation
            Live reconstructed conversation.

        """
        item = self.conversations.get(_text(identifier))
        _require(item is not None and not item.deleted,
                 "Command refers to an unknown or archived conversation.")
        return item

    def _check_record(self, data: dict) -> None:
        """Check recorded catalog metadata against reconstructed lineage.

        Parameters
        ----------
        data: dict
            One catalog, selection, or mutation result.

        """
        item = self._known(data.get("conversation_id"))
        _require(data.get("path") == item.path and data.get("name") ==
                 item.name and data.get("parent_conversation_id") ==
                 item.parent and data.get("origin_turn_count") ==
                 item.origin_count and data.get("origin_checkpoint_id") ==
                 item.origin_checkpoint, "Conversation metadata disagrees "
                 "with recorded history.")

    def consume(self, op: TimelineOperation) -> None:
        """Apply one command's recorded durable outcome to the history.

        Parameters
        ----------
        op: TimelineOperation
            Candidate recorded command, never executed by this validator.

        """
        kind, request = op.kind, op.request
        if kind in ("create_project", "open_project"):
            self._open(op)
        else:
            _require(self.active is not None, "Command without open project.")
            previous = self.active
            if kind == "close_project":
                _require(request.get("conversation_id") == self.active,
                         "Close refers to the wrong conversation.")
                self.active = None
            elif kind == "ask":
                self._question(op)
            elif kind in ("create_conversation", "reset_conversation"):
                _require(request.get("previous_conversation_id") == previous,
                         "Creation refers to the wrong active conversation.")
                result = one(op.events, "gui.conversation.create.result" if
                             kind == "create_conversation" else
                             "gui.conversation.reset.result").details
                if kind == "create_conversation":
                    _require(type(request.get("branch")) is bool and
                             type(result.get("reused")) is bool,
                             "Missing creation or reuse policy.")
                if result.get("reused") is True:
                    _require(request.get("branch") is False and
                             not self._known(result.get("conversation_id")
                                             ).turns,
                             "Only an empty nonbranch chat may be reused.")
                    self._check_record(result)
                    self.active = result["conversation_id"]
                    _require(self.active != previous, "Reused active chat.")
                else:
                    parent = previous if request.get("branch") else None
                    self.active = self._new(result, parent=parent)
                    if parent is not None:
                        count = request.get("turn_count")
                        expected = (len(self._known(parent).turns)
                                    if count is None else count)
                        _require(type(expected) is int and
                                 result.get("origin_turn_count") == expected,
                                 "Branch result differs from requested "
                                 "prefix.")
                self._selection(op, previous)
            elif kind == "select_conversation":
                target = _text(request.get("conversation_id"))
                self._known(target)
                self.active = target
                if previous == target:
                    unchanged = one(op.events,
                                    "gui.conversation.select.unchanged")
                    _require(unchanged.details.get("conversation_id") ==
                             target, "Invalid unchanged selection.")
                    _require(not op.matching("gui.conversation.selected"),
                             "No-op selection also switched conversations.")
                else:
                    self._selection(op, previous)
            elif kind == "rename_conversation":
                item = self._known(request.get("conversation_id"))
                result = one(op.events, "gui.conversation.renamed").details
                _require(result.get("conversation_id") ==
                         request.get("conversation_id") and
                         result.get("name") == " ".join(
                             _text(request.get("name")).split()),
                         "Rename result differs from requested name.")
                item.name = result["name"]
                self._check_record(result)
            elif kind == "delete_conversation":
                self._delete(op)
            elif kind in ("feedback", "memory_exclusion"):
                self._feedback(op)
            elif kind == "sync_memory":
                _require(request.get("trigger") == "manual",
                         "Expected manual memory synchronization.")
                one(op.events, "gui.conversation.memory.sync.result")
                one(op.events, "project.memory.sync.result")
        # Restoration is expected when an actual switch/open occurs, but
        # additional model-restoration diagnostics do not create new turns.
        for event in op.matching("conversation.restored"):
            _require(self.active is not None, "Restore without active chat.")
            self._restored(event)

    def _open(self, op: TimelineOperation) -> None:
        """Verify opening and the active catalog without creating state.

        Parameters
        ----------
        op: TimelineOperation
            Creation or resume command.

        """
        _require(self.active is None, "Project opened without closing first.")
        catalog = one(op.events, "gui.conversation.catalog.opened").details
        rows = catalog.get("conversations")
        _require(isinstance(rows, list), "Missing conversation catalog.")
        if op.kind == "create_project":
            _require(not self.created and len(rows) == 1,
                     "Reference must begin with one new conversation.")
            self.created = True
            self._new(rows[0])
            request = op.request["open"]
        else:
            _require(self.created, "Resume precedes creation.")
            request = op.request
        identifiers = [_text(row.get("conversation_id")) for row in rows]
        _require(len(set(identifiers)) == len(identifiers) and
                 set(identifiers) == {key for key, item in
                                      self.conversations.items()
                                      if not item.deleted},
                 "Catalog differs from reconstructed conversations.")
        for row in rows:
            self._check_record(row)
        self.active = _text(catalog.get("active_conversation_id"))
        self._known(self.active)
        _require(request.get("conversation_id") == self.active,
                 "Open request and catalog disagree.")
        if request.get("requested_conversation_id") is not None:
            _require(request["requested_conversation_id"] == self.active,
                     "Resume selected a different conversation.")
        one(op.events, "conversation.restored")

    def _selection(self, op: TimelineOperation, previous: str) -> None:
        """Check the selected identity and its restoration event.

        Parameters
        ----------
        op: TimelineOperation
            Command that switches the active conversation.
        previous: str
            Previously active conversation.

        """
        result = one(op.events, "gui.conversation.selected").details
        _require(result.get("previous_conversation_id") == previous and
                 result.get("conversation_id") == self.active,
                 "Selection result differs from requested transition.")
        self._check_record(result)
        one(op.events, "conversation.restored")

    def _restored(self, event: TraceEvent) -> None:
        """Check restored turns and checkpoint against reconstructed state.

        Parameters
        ----------
        event: TraceEvent
            Core conversation restoration diagnostic.

        """
        item = self._known(self.active)
        data = event.details
        _require(data.get("turn_uuids") == item.turns and
                 type(data.get("turn_count")) is int and
                 data["turn_count"] == len(item.turns) and
                 data.get("checkpoint_id") == item.checkpoint,
                 "Restored history differs from committed history.")
        # Archived files have their final location even for older events.
        state = self.recording.states.get(event.path, {})
        _require(state.get("conversation_id") == self.active,
                 "Restoration was written to a different conversation.")

    def _delete(self, op: TimelineOperation) -> None:
        """Verify archive identity and any replacement conversation.

        Parameters
        ----------
        op: TimelineOperation
            Recorded deletion command.

        """
        previous = self.active
        identifier = _text(op.request.get("conversation_id"))
        item = self._known(identifier)
        result = one(op.events, "gui.conversation.deleted").details
        _require(result.get("conversation_id") == identifier and
                 result.get("deleted_active") is (identifier == previous),
                 "Invalid deletion result.")
        archive = _text(result.get("archived_path"))
        safe_path(self.recording.root, archive)
        _require(Path(archive).parent.as_posix() == "logs/traces/.deleted",
                 "Invalid archive location.")
        item.path, item.deleted = archive, True
        replacement = result.get("replacement")
        _require(isinstance(replacement, dict) and
                 type(result.get("replacement_created")) is bool,
                 "Missing deletion replacement result.")
        if result["replacement_created"]:
            _require(all(c.deleted for c in self.conversations.values()),
                     "Unexpected replacement creation.")
            self.active = self._new(replacement)
        else:
            self._check_record(replacement)
            self.active = replacement["conversation_id"]
        if identifier != previous:
            _require(self.active == previous,
                     "Deleting an inactive chat changed selection.")
        else:
            self._selection(op, previous)

    def _question(self, op: TimelineOperation) -> None:
        """Validate the request, exact question, and its durable checkpoint.

        Parameters
        ----------
        op: TimelineOperation
            Question command and correlated core turn events.

        """
        request = op.request
        _require(request.get("conversation_id") == self.active,
                 "Question targets the wrong active conversation.")
        item = self._known(self.active)
        question = _text(request.get("question")).strip()
        _require(bool(question) and isinstance(request.get("title_question"),
                                               str), "Invalid question text.")
        _require(isinstance(request.get("generation_settings"), dict) and
                 isinstance(request.get("explicit_generation_settings"), dict)
                 and isinstance(request.get("profile_generation_settings"),
                                dict), "Missing question settings provenance.")
        start = one(op.events, "gui.question.request.started")
        resolved = one(op.events, "gui.question.settings.resolved")
        core = one(op.events, "turn.started")
        effective = one(op.events, "turn.settings.snapshot")
        _require(core.sequence < effective.sequence and
                 effective.metadata.get("turn_id") ==
                 core.metadata.get("turn_id") and
                 resolved.details.get("chat_model") ==
                 effective.details.get("chat_model") and
                 (resolved.details.get("retrieval_count") is None or
                  resolved.details["retrieval_count"] ==
                  effective.details.get("retrieval_count")),
                 "Resolved question settings disagree with core turn.")
        end = one(op.events, "turn.completed")
        committed = one(op.events, "state.checkpoint.committed")
        saved = one(op.events, "gui.question.saved")
        _require(start.sequence < resolved.sequence < core.sequence <
                 committed.sequence < end.sequence < saved.sequence and
                 core.metadata.get("parent_span_id") == start.span_id and
                 core.path == start.path == committed.path == saved.path and
                 core.metadata.get("turn_id") ==
                 committed.metadata.get("turn_id") and
                 saved.span_id == start.span_id,
                 "Uncorrelated question, turn, or checkpoint.")
        _require(self.recording.states.get(start.path, {}).get(
            "conversation_id") == self.active,
            "Question stored in another conversation trace.")
        _require(core.details.get("question") == question and
                 core.details.get("question_sha256") == sha256(
                     question.encode("utf-8")).hexdigest(),
                 "Core question differs from recorded request.")
        data = committed.details
        uid = require_turn_uuid(data.get("conversation_turn_uuid"))
        checkpoint = _text(data.get("checkpoint_id"))
        _require(uid not in self.questions and checkpoint not in
                 self.checkpoints and data.get("parent_checkpoint_id") ==
                 item.checkpoint and data.get("immutable_state_saved") is True
                 and type(data.get("turn_count")) is int and
                 data["turn_count"] == len(item.turns) + 1,
                 "Invalid checkpoint identity or lineage.")
        _require(all(saved.details.get(k) == data.get(k) for k in (
            "conversation_turn_uuid", "turn_count", "checkpoint_id")),
            "GUI save result disagrees with core checkpoint.")
        self.questions[uid] = question
        self.checkpoints.add(checkpoint)
        item.turns.append(uid)
        item.checkpoint = checkpoint
        _interactions(op, core)

    def _feedback(self, op: TimelineOperation) -> None:
        """Check feedback mutations against the referenced committed turn.

        Parameters
        ----------
        op: TimelineOperation
            Feedback or memory-exclusion command.

        """
        request = op.request
        item = self._known(request.get("conversation_id"))
        uid = require_turn_uuid(request.get("turn_uuid"))
        role = request.get("role")
        _require(uid in item.turns and role in ("user", "assistant"),
                 "Feedback references an unavailable turn or role.")
        result = one(op.events, "gui.conversation.feedback.saved" if
                     op.kind == "feedback" else
                     "gui.conversation.memory_exclusion.saved").details
        _require(all(result.get(k) == request.get(k) for k in (
            "conversation_id", "turn_uuid", "role")),
            "Feedback result targets another message.")
        revision = request.get("expected_revision")
        _require(revision is None or revision == self.feedback_revision,
                 "Feedback revision does not follow previous mutation.")
        key = uid, role
        before = self.ratings.get(key, MessageFeedback())
        values = asdict(before)
        fields = ("saved", "reaction") if op.kind == "feedback" else (
            "excluded",)
        for name in fields:
            if request.get(name) is not None:
                values[name] = request[name]
        rating = MessageFeedback(**values)
        _require(result.get("rating") == asdict(rating),
                 "Committed feedback differs from requested mutation.")
        self.ratings[key] = rating
        self.feedback_revision = _text(result.get("revision"))

    def generated_title(self, event: TraceEvent) -> None:
        """Apply a generated name as observed state, never as a command.

        Parameters
        ----------
        event: TraceEvent
            Generated title diagnostic.

        """
        identifier = event.details.get("conversation_id")
        item = self._known(identifier)
        _require(identifier == self.active and item.turns,
                 "Generated title without a committed active conversation.")
        item.name = _text(event.details.get("name"))

    def finish(self) -> None:
        """Compare archived and active states and final feedback to history."""
        recording = self.recording
        if self.active is not None:
            recording.issue("project.unclosed", "Close the reference project.")
        actual = {}
        copies = {}
        for path, state in recording.states.items():
            try:
                identifier = _text(state.get("conversation_id"))
                _require(identifier not in actual,
                         "Duplicate conversation ID.")
                actual[identifier] = path
                item = self.conversations.get(identifier)
                _require(item is not None, "Unrecorded conversation.")
                _require(item.path == path and item.name == state.get("name")
                         and item.parent == state.get("parent_conversation_id")
                         and item.origin_count == state.get(
                             "origin_turn_count")
                         and item.origin_checkpoint ==
                         state.get("origin_checkpoint_id"),
                         "Final conversation metadata differs from history.")
                turns = state.get("turns", [])
                _require([t["conversation_turn_uuid"] for t in turns] ==
                         item.turns and state.get("checkpoint_id") ==
                         item.checkpoint, "Final turn/checkpoint mismatch.")
                for turn in turns:
                    uid = turn["conversation_turn_uuid"]
                    _require(turn.get("question") == self.questions.get(uid)
                             and isinstance(turn.get("answer"), str)
                             and isinstance(turn.get("parameters"), dict)
                             and isinstance(turn.get("evidence"), list),
                             "Saved turn does not match recorded input.")
                    _require(uid not in copies or copies[uid] == turn,
                             "Inherited turn content differs across branches.")
                    copies[uid] = turn
                if item.checkpoint is not None:
                    checkpoint = state.get("checkpoint")
                    _require(isinstance(checkpoint, dict) and
                             checkpoint.get("checkpoint_id") == item.checkpoint
                             and type(checkpoint.get("turn_count")) is int
                             and checkpoint["turn_count"] == len(turns),
                             "Invalid saved checkpoint metadata.")
                    content = {"system_prompt": state.get("system_prompt"),
                               "turns": turns}
                    digest = sha256(json.dumps(content, sort_keys=True,
                                    ensure_ascii=False).encode("utf-8")
                                    ).hexdigest()
                    _require(checkpoint.get("conversation_sha256") == digest,
                             "Saved conversation content hash mismatch.")
                else:
                    _require(state.get("checkpoint") is None,
                             "Checkpoint exists without recorded identity.")
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                recording.issue("conversation.state", str(error), path)
        for identifier in sorted(set(self.conversations) - set(actual)):
            recording.issue("conversation.missing", f"Missing {identifier}.",
                            self.conversations[identifier].path)
        try:
            feedback = safe_path(recording.root, "memory/feedback.json")
            if feedback.exists():
                read_json(recording.root, "memory/feedback.json")
            snapshot = FeedbackStore(recording.root).load()
            observed = {(uid, role): rating for uid, values in
                        snapshot.entries.items() for role, rating in
                        values.items()}
            expected = {key: rating for key, rating in self.ratings.items()
                        if rating != MessageFeedback()}
            _require(observed == expected and snapshot.revision ==
                     self.feedback_revision,
                     "Final feedback differs from recorded mutations.")
        except (ValueError, OSError, TypeError) as error:
            recording.issue("feedback.state", str(error),
                            "memory/feedback.json")


def _interactions(op: TimelineOperation, core: TraceEvent) -> None:
    """Verify proposal/decision pairs and semantic user responses.

    Parameters
    ----------
    op: TimelineOperation
        Question with all recorded interaction events.
    core: TraceEvent
        Owning core turn start.

    """
    pending = {}
    seen = set()
    for event in op.events:
        prefix, _, suffix = event.event_type.rpartition(".")
        if prefix not in ("routing.choice", "tool.choice", "tool.plan"):
            continue
        if suffix not in ("proposed", "decided", "failed"):
            continue
        data = event.details
        identifier = _text(data.get("interaction_id"))
        _require(event.path == core.path and event.metadata.get("turn_id")
                 == core.metadata.get("turn_id"),
                 "Interaction is not correlated to the core turn.")
        if suffix == "proposed":
            _require(identifier not in seen and not pending,
                     "Repeated or overlapping interaction.")
            _require(type(data.get("asked")) is bool,
                     "Missing interaction callback policy.")
            seen.add(identifier)
            pending[identifier] = event
            continue
        proposal = pending.pop(identifier, None)
        _require(proposal is not None and proposal.event_type ==
                 prefix + ".proposed", "Interaction lacks matching proposal.")
        _require(suffix == "decided", "Interactive callback failed.")
        request = proposal.details
        asked = request["asked"]
        _require(data.get("asked") is asked, "Interaction policy changed.")
        if prefix == "tool.plan":
            choices = _strings(request.get("choices"))
            choice = data.get("choice")
            _require(choice in choices and choice in ("continue", "quick")
                     and data.get("policy") == request.get("policy") and
                     (data.get("response") == choice if asked else
                      choice == "continue" and data.get("response") is None),
                     "Invalid or cancelled plan choice.")
        else:
            _require(isinstance(request.get("question"), str) and
                     type(request.get("allow_cancel")) is bool and
                     isinstance(request.get("options"), list),
                     "Invalid recorded option proposal.")
            options = [_text(value) for value in request["options"]]
            index = data.get("index")
            if asked:
                _require(type(index) is int and 0 <= index < len(options)
                         and data.get("response") == index and
                         data.get("selected_option") == options[index] and
                         options.count(options[index]) == 1,
                         "Invalid, ambiguous or cancelled option choice.")
            else:
                _require(index is None and data.get("selected_option") is None
                         and data.get("response") is None,
                         "Automatic choice has a user response.")
    _require(not pending, "Interaction proposal has no terminal decision.")


def _differences(defaults: dict, requested: dict) -> dict:
    """Check the historical sparse-override classification.

    Parameters
    ----------
    defaults: dict
        Defaults actually displayed in the original environment.
    requested: dict
        Full submitted settings, not present-day defaults.

    Returns
    -------
    dict
        Recursively differing fields, treating lists as atomic.

    """
    changes = {}
    for key, value in requested.items():
        if isinstance(value, dict) and isinstance(defaults.get(key), dict):
            nested = _differences(defaults[key], value)
            if nested:
                changes[key] = nested
        elif key not in defaults or defaults[key] != value:
            changes[key] = value
    return changes


def _creation_checks(op: TimelineOperation) -> None:
    """Require the current complete-recording marker and settings provenance.

    Parameters
    ----------
    op: TimelineOperation
        Recorded creation command.

    """
    settings = one(op.events, "gui.project.creation.settings").details
    _require(type(settings.get("operation_recording_version")) is int and
             settings["operation_recording_version"] ==
             OPERATION_RECORDING_VERSION, "Unsupported recording version.")
    requested = settings.get("requested_settings")
    explicit = settings.get("explicit_settings")
    _require(isinstance(requested, dict) and isinstance(explicit, dict),
             "Creation settings provenance is unavailable.")
    if settings.get("provenance") == "explicit_request":
        _require(explicit == requested, "Explicit creation settings differ.")
    else:
        defaults = settings.get("displayed_defaults")
        _require(settings.get("provenance") == "displayed_defaults" and
                 isinstance(defaults, dict) and explicit ==
                 _differences(defaults, requested),
                 "Creation defaults and explicit overrides disagree.")


def _operation_checks(op: TimelineOperation) -> None:
    """Require indexing, settings and source outcome coverage per command.

    Parameters
    ----------
    op: TimelineOperation
        Supported command with its correlated diagnostics.

    """
    kind = op.kind
    needs_index = kind in ("create_project", "open_project", "sync_sources")
    if kind == "create_project":
        _creation_checks(op)
    if kind in ("create_project", "open_project"):
        config = one(op.events, "gui.project.configuration.effective").details
        _require(all(key in config for key in (
            "profile", "effort_levels", "router_models", "search_depth",
            "source_cards", "behavior", "tools")),
            "Missing effective project configuration.")
    if kind == "add_sources":
        result = one(op.events, "gui.sources.add.completed").details
        _strings(op.request.get("sources"))
        new = _strings(result.get("new_sources"))
        batches = op.matching("workspace.prepare.started")
        _require(len(batches) == bool(new) and
                 (not new or batches[0].details.get("operation") == "import"
                  and batches[0].details.get("source_paths") == new),
                 "Source addition lacks matching import batch.")
        decisions = op.request.get("decisions")
        _require(result.get("decisions") == decisions and
                 len(op.matching("workspace.sync.started")) == bool(decisions),
                 "Source addition lacks matching synchronization batch.")
        needs_index = bool(new or decisions)
    if kind in ("add_sources", "sync_sources") and needs_index:
        one(op.events, "gui.sources.refreshed")
    if needs_index:
        first = one(op.events, "indexing.started")
        last = one(op.events, "indexing.completed")
        scan = one(op.events, "source.scan.completed")
        result = one(op.events, "index.rebuilt", "index.reused",
                     "index.empty")
        _require(first.sequence < scan.sequence < result.sequence <
                 last.sequence and first.span_id == last.span_id and
                 scan.details.get("complete") is True,
                 "Incomplete or uncorrelated index preparation.")
        manifest = (result if result.event_type == "index.empty" else
                    one(op.events, "index.check.completed"))
        _require(scan.sequence < manifest.sequence <= result.sequence and
                 set(_strings(scan.details.get("selected_files"))) ==
                 set(_manifest_files(manifest.details.get("source_manifest"))),
                 "Source scan differs from historical content manifest.")
        for event in op.matching("workspace.prepare.completed",
                                 "workspace.sync.completed"):
            _require(event.sequence < first.sequence,
                     "Indexing precedes completion of source import.")
    if kind == "ask":
        request = op.request
        explicit = request.get("explicit_generation_settings")
        settings = request.get("generation_settings")
        profile = request.get("profile_generation_settings")
        _require(isinstance(settings, dict) and isinstance(explicit, dict)
                 and isinstance(profile, dict), "Invalid question settings.")
        provenance = request.get("generation_provenance")
        _require((provenance == "explicit_request" and explicit == settings)
                 or (provenance == "profile_difference" and explicit ==
                     _differences(profile, settings)),
                 "Question settings provenance does not match the request.")
        # Generation values can be persisted inside an answer request. Other
        # persistent setting mutations are not permitted in reference runs.
        for event in op.matching("project.configuration.requested"):
            changes = event.details.get("requested_settings")
            _require(isinstance(changes, dict) and set(changes) <= {
                "temperature", "thinking", "max_new_tokens"},
                "Unsupported configuration change inside question.")


def _configuration_checks(recording: Recording,
                          operations: tuple[TimelineOperation, ...]) -> None:
    """Check fixed recorded configuration and its saved profile.

    Parameters
    ----------
    recording: Recording
        Recording and issue collector.
    operations: tuple[TimelineOperation, ...]
        Candidate command boundaries.

    """
    profile = None
    configuration = None
    for event in recording.events:
        data = event.details
        kind = event.event_type
        try:
            if kind == "gui.project.configuration.effective":
                current = data.get("profile")
                _require(isinstance(current, dict),
                         "Missing effective profile.")
                if profile is not None:
                    _require(profile == current,
                             "Effective profile changed between openings.")
                profile = current
                fixed = {k: v for k, v in data.items() if k != "profile"}
                if isinstance(fixed.get("behavior"), dict):
                    fixed["behavior"] = {k: v for k, v in
                                         fixed["behavior"].items()
                                         if k != "project"}
                _require(configuration is None or configuration == fixed,
                         "Persistent settings changed between openings.")
                configuration = fixed
            elif kind == "project.configuration.saved":
                _require(profile is not None and data.get("before") == profile
                         and isinstance(data.get("effective"), dict),
                         "Saved configuration does not follow prior profile.")
                profile = data["effective"]
            elif kind in ("project.configuration.requested",
                          "project.settings.changed"):
                _require(any(op.kind == "ask" and op.start_sequence <
                             event.sequence < op.end_sequence
                             for op in operations),
                         "Persistent setting change outside a question.")
            elif kind == "gui.models.effort_levels.saved":
                _require(any(op.kind == "create_project" and
                             op.start_sequence < event.sequence <
                             op.end_sequence for op in operations),
                         "Effort settings changed after creation.")
            elif (kind in ("project.behavior.saved",
                           "project.tool_settings.saved",
                           "gui.models.change.requested",
                           "gui.models.change.applied",
                           "gui.source_cards.settings.saved") or
                  kind.startswith(("gui.folder.", "gui.sources.remove."))):
                raise ValueError("Unsupported persistent or resource change.")
            elif kind == "gui.project.configuration.snapshot.failed":
                raise ValueError("Project configuration snapshot failed.")
        except (ValueError, TypeError, KeyError) as error:
            recording.issue("configuration.history", str(error),
                            event.path, event.sequence)
    try:
        profile_path = safe_path(recording.root, "profile.ini")
        safe_path(recording.root, "sources")
        parser = ConfigParser(interpolation=None)
        with profile_path.open(encoding="utf-8") as stream:
            parser.read_file(stream)
        _require(parser.get("project", "source") == "sources",
                 "Reference must use its managed source directory.")
        project = open_project(recording.root)
        _require(project.profile.source == "sources" and
                 project.source == recording.root / "sources",
                 "Reference must use its managed source directory.")
        observed = json.loads(json.dumps(asdict(project.profile)))
        _require(profile is not None and observed == profile,
                 "Saved profile differs from the last recorded profile.")
        # Explicit behavior changes are not supported during reference
        # preparation. Its revision identifies the saved overrides, not
        # current package defaults, which are allowed to evolve.
        behavior = safe_path(recording.root, "project_behavior.json")
        _require(not behavior.exists() or behavior.is_file(),
                 "Invalid saved behavior file.")
        revision = sha256(behavior.read_bytes() if behavior.is_file()
                          else b"").hexdigest()
        _require(configuration is not None and
                 isinstance(configuration.get("behavior"), dict) and
                 configuration["behavior"].get("revision") == revision,
                 "Saved behavior overrides differ from recorded revision.")
        _saved_settings_match(parser, configuration)
        overrides = safe_path(recording.root, "source_card_sources.json")
        _require(not overrides.exists(),
                 "Per-source card overrides are not recorded replay inputs.")
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            ConfigError) as error:
        recording.issue("configuration.state", str(error), "profile.ini")


def _saved_settings_match(parser: ConfigParser, configuration: dict) -> None:
    """Compare persisted overrides to historical, not current, defaults.

    Parameters
    ----------
    parser: configparser.ConfigParser
        Saved project profile, read without interpolation.
    configuration: dict
        Last recorded effective settings apart from the core profile.

    """
    for backend in ("mlx", "torch"):
        levels = []
        for index in range(1, 6):
            value = parser.get(f"effort.{backend}", f"level{index}")
            name, separator, model = value.partition("|")
            _require(bool(separator), "Invalid saved effort level.")
            levels.append({"name": name.strip(), "model": model.strip()})
        _require(levels == configuration["effort_levels"][backend] and
                 parser.get("router", backend) ==
                 configuration["router_models"][backend],
                 "Saved effort or router settings differ from recording.")
    depths = [int(value) for value in parser.get("search", "depth").split(",")]
    _require(depths == configuration["search_depth"],
             "Saved search depth differs from recording.")
    tools = {entry["name"]: entry for entry in configuration["tools"]}
    saved = {section[5:]: dict(parser.items(section))
             for section in parser.sections() if section.startswith("tool.")}
    for name, entries in saved.items():
        _require(name in tools and all(
            key in tools[name]["settings"] and
            json.loads(value) == tools[name]["settings"][key]
            for key, value in entries.items()),
            "Saved tool overrides differ from recording.")
    for name, entry in tools.items():
        changed = _differences(entry["type_settings"], entry["settings"])
        _require(all(key in saved.get(name, {}) for key in changed),
                 "Recorded tool overrides are missing from the profile.")
    cards = configuration["source_cards"]
    policy = "source_cards"
    if parser.has_section(policy):
        for key in ("enabled", "kind", "automatic"):
            if parser.has_option(policy, key):
                value = (parser.get(policy, key) or None if key == "kind"
                         else parser.getboolean(policy, key))
                _require(value == cards["policy"][key],
                         "Saved source-card policy differs from recording.")
    section = "source_card_generation"
    for backend, settings in cards["models"].items():
        if parser.has_option(section, backend):
            _require(parser.get(section, backend) == settings["model_id"],
                     "Saved source-card model differs from recording.")
        for name, getter in (("thinking", parser.getboolean),
                             ("temperature", parser.getfloat),
                             ("max_input_tokens", parser.getint),
                             ("max_new_tokens", parser.getint)):
            if parser.has_option(section, name):
                _require(getter(section, name) == settings[name],
                         "Saved source-card settings differ from recording.")


def _environment_warnings(recording: Recording) -> list[ValidationIssue]:
    """Report runtime differences without importing model backends.

    Parameters
    ----------
    recording: Recording
        Original logger environments.

    Returns
    -------
    list[ValidationIssue]
        Distinct differences from the inspection environment.

    """
    differences = set()
    versions = {}
    for event in recording.events:
        if event.event_type != "trace.started":
            continue
        environment = event.details.get("environment")
        if not isinstance(environment, dict):
            recording.issue("trace.environment", "Missing logger environment.",
                            event.path, event.sequence)
            continue
        for key, current in (("python", sys.version),
                             ("platform", platform.platform())):
            recorded = environment.get(key)
            if not isinstance(recorded, str):
                recording.issue("trace.environment", f"Invalid {key}.",
                                event.path, event.sequence)
            elif recorded != current:
                differences.add(f"{key}: recorded {recorded}; local {current}")
        packages = environment.get("packages")
        if not isinstance(packages, dict):
            recording.issue("trace.environment", "Missing package versions.",
                            event.path, event.sequence)
            continue
        for name, recorded in packages.items():
            if recorded is not None and not isinstance(recorded, str):
                recording.issue("trace.environment",
                                "Invalid package version.",
                                event.path, event.sequence)
                continue
            if name not in versions:
                try:
                    versions[name] = version(name)
                except (PackageNotFoundError, ValueError):
                    versions[name] = None
            if recorded != versions[name]:
                differences.add(f"{name}: recorded {recorded}; "
                                f"local {versions[name]}")
    return [ValidationIssue("environment.difference", message)
            for message in sorted(differences)]


def validate_recording(recording: Recording) -> ValidationReport:
    """Collect structural, chronological and durable-state inconsistencies.

    Parameters
    ----------
    recording: Recording
        Decoded snapshot of an offline reference workspace.

    Returns
    -------
    ValidationReport
        Blocking issues, warnings and the candidate timeline. No models,
        GUI components, external source paths or generated indexes are opened.

    """
    _sequence_checks(recording)
    operations = build_timeline(recording)
    creations = [op for op in operations if op.kind == "create_project"]
    if (len(creations) != 1 or not operations
            or operations[0].kind != "create_project"):
        recording.issue("creation.missing", "Expected one initial creation.")
    sources = _SourceHistory(recording.root)
    for event in recording.events:
        try:
            sources.consume(event)
        except (ValueError, OSError, KeyError, TypeError, AttributeError) as e:
            recording.issue("source.history", str(e), event.path,
                            event.sequence)
    sources.finish(recording)
    history = _ConversationHistory(recording)
    titles = iter(e for e in recording.events
                  if e.event_type == "conversation.title.generated")
    title = next(titles, None)
    for i, op in enumerate(operations):
        try:
            _operation_checks(op)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            recording.issue("command.coverage", str(error),
                            sequence=op.start_sequence)
        try:
            if op.status in ("completed", "committed_with_error"):
                history.consume(op)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            recording.issue("conversation.history", str(error),
                            sequence=op.start_sequence)
        # Automatic naming is emitted just after the question span closes.
        # Do not borrow a later question's title for an earlier operation.
        next_start = (operations[i + 1].start_sequence
                      if i + 1 < len(operations) else float("inf"))
        generated = 0
        while title is not None and title.sequence < next_start:
            try:
                _require(op.kind == "ask" and
                         op.start_sequence < title.sequence and generated == 0,
                         "Unmatched generated title.")
                history.generated_title(title)
                generated += 1
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                recording.issue("conversation.title", str(error), title.path,
                                title.sequence)
            title = next(titles, None)
    if title is not None:
        recording.issue("conversation.title", "Unmatched generated title.",
                        title.path, title.sequence)
    history.finish()
    _configuration_checks(recording, operations)
    warnings = [ValidationIssue(
        "recording.limit", "Validation requires a closed, unchanged reference "
        "workspace. Unrecorded actions and edits restored before observation "
        "cannot always be detected. Model availability is not checked."
    )]
    warnings.extend(ValidationIssue(
        "relocation.inferred",
        "Historical sibling rename inferred from independent trace paths; "
        "explicit rename markers were not recorded.",
        sequence=op.start_sequence,
    ) for op in operations if op.request.get("relocation", {}).get(
        "provenance") == "inferred")
    warnings.extend(_environment_warnings(recording))
    return ValidationReport(recording.root, tuple(recording.issues),
                            tuple(warnings), len(recording.events),
                            tuple(recording.events), operations)
