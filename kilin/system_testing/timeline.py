"""Reconstruct explicit user operations from project-wide event order.

Key Features
------------
- Group creation, source batches, and question spans without guessing inputs.
- Keep intermediate diagnostics with the operation that caused them.
- Detect ambiguous or overlapping command boundaries.

Module Structure
----------------
- ``TimelineOperation`` describes one command and its recorded diagnostics.
- ``build_timeline`` groups supported command boundaries in allocation order.

Examples
--------
>>> from kilin.system_testing import Experiment
>>> # Experiment(workspace).timeline() returns ordered command records.

"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .traces import Recording, TraceEvent

__all__ = ["TimelineOperation", "build_timeline"]

_SPANS = {
    "gui.project.open": "open_project",
    "gui.question.request": "ask",
    "gui.conversation.create": "create_conversation",
    "gui.conversation.select": "select_conversation",
    "gui.conversation.rename": "rename_conversation",
    "gui.conversation.delete": "delete_conversation",
    "gui.conversation.reset": "reset_conversation",
    "gui.conversation.feedback": "feedback",
    "gui.conversation.memory_exclusion": "memory_exclusion",
    "gui.conversation.memory.sync": "sync_memory",
    "project.source_cards.regenerate": "regenerate_source_cards",
}
_PROJECT = "logs/traces/project.json"


@dataclass(frozen=True, slots=True)
class TimelineOperation:
    """A recorded command with its request, outcome, and diagnostic context.

    Attributes
    ----------
    kind: str
        Stable operation name, independent of GUI event display labels.
    start_sequence: int
        First event allocated for this operation.
    end_sequence: int
        Last event at the operation's recorded boundary.
    status: str
        Completed, failed, incomplete, or committed_with_error. The last
        status preserves a saved core turn despite a failed outer request.
    request: dict
        Recorded input, including provenance for creation and questions.
    events: tuple[TraceEvent, ...]
        All diagnostic events within the operation's sequence interval.
        Intermediate model and indexing events are not separate commands.

    """

    kind: str
    start_sequence: int
    end_sequence: int
    status: str
    request: dict
    events: tuple[TraceEvent, ...]

    def matching(self, *event_types: str) -> tuple[TraceEvent, ...]:
        """Return recorded events of the requested types in original order.

        Parameters
        ----------
        *event_types: str
            Exact original event types to select.

        Returns
        -------
        tuple[TraceEvent, ...]
            Matching diagnostics, including results and interaction records.

        """
        return tuple(e for e in self.events if e.event_type in event_types)


def one(events: tuple[TraceEvent, ...] | list[TraceEvent],
        *types: str) -> TraceEvent:
    """Require exactly one matching event rather than selecting arbitrarily.

    Parameters
    ----------
    events: sequence of TraceEvent
        Candidate events in an operation or recording.
    *types: str
        Alternative exact event types.

    Returns
    -------
    TraceEvent
        The sole matching event.

    """
    found = [event for event in events if event.event_type in types]
    if len(found) != 1:
        raise ValueError(f"Expected one of {types}, found {len(found)}.")
    return found[0]


def _batch_end(recording: Recording, start: TraceEvent,
               ends: dict[str, list[TraceEvent]]) -> TraceEvent | None:
    """Find a terminal source event by its recorded batch ID.

    Parameters
    ----------
    recording: Recording
        Issue collector.
    start: TraceEvent
        Batch start.
    ends: dict
        Candidate terminal events by event type.

    Returns
    -------
    TraceEvent | None
        Unambiguous matching terminal, if recorded.

    """
    batch = start.details.get("batch_id")
    found = [e for e in ends.get(start.event_type[:-8] + ".completed", [])
             if e.details.get("batch_id") == batch]
    if (not isinstance(batch, str) or not batch or len(found) != 1
            or found[0].sequence <= start.sequence):
        recording.issue("batch.incomplete", "Missing or ambiguous batch end.",
                        start.path, start.sequence)
        return None
    return found[0]


def build_timeline(recording: Recording) -> tuple[TimelineOperation, ...]:
    """Build supported command intervals from explicit recorded boundaries.

    Parameters
    ----------
    recording: Recording
        Ordered decoded events; receives grouping issues.

    Returns
    -------
    tuple[TimelineOperation, ...]
        Candidate chronological commands. A validator must check the recording
        and these intervals before exposing them for replay.

    Notes
    -----
    Initial import and first open constitute one creation command. Additional
    source imports and sync batches inside an add command belong to that add.
    A standalone synchronization ends after the GUI index refresh. Legacy
    diagnostics or arbitrary direct core calls are not promoted to commands.

    """
    events = recording.events
    sequences = [event.sequence for event in events]
    by_type: dict[str, list[TraceEvent]] = {}
    terminals: dict[tuple[str, str], list[TraceEvent]] = {}
    for event in events:
        by_type.setdefault(event.event_type, []).append(event)
        if event.span_id:
            terminals.setdefault((event.span_id, event.event_type), []).append(
                event
            )
    operations = []

    def add(kind: str, start: TraceEvent, end: TraceEvent | None,
            request: dict, status: str = "completed") -> None:
        """Retain the sequence interval and all its recorded diagnostics.

        Parameters
        ----------
        kind: str
            Command kind.
        start: TraceEvent
            Opening boundary.
        end: TraceEvent | None
            Closing boundary, if present.
        request: dict
            Original command data.
        status: str, optional
            Recorded outcome.

        """
        last = end.sequence if end is not None else start.sequence
        subset = tuple(events[bisect_left(sequences, start.sequence):
                              bisect_right(sequences, last)])
        if end is None:
            status = "incomplete"
        elif end.event_type.endswith(".failed"):
            status = "failed"
            if kind == "ask" and any(
                e.event_type == "state.checkpoint.committed" for e in subset
            ):
                status = "committed_with_error"
        operations.append(TimelineOperation(
            kind, start.sequence, last, status, request, subset
        ))
        if status != "completed":
            recording.issue("command." + status,
                            f"{kind}: {status.replace('_', ' ')}.",
                            start.path, start.sequence)

    create_batches = [e for e in by_type.get("workspace.prepare.started", [])
                      if e.details.get("operation") == "create"]
    markers = by_type.get("gui.project.creation.settings", [])
    for prefix, kind in _SPANS.items():
        for start in by_type.get(prefix + ".started", []):
            ends = [e for suffix in (".completed", ".failed")
                    for e in terminals.get(
                        (start.span_id, prefix + suffix), [])
                    if e.path == start.path and e.sequence > start.sequence]
            end = ends[0] if len(ends) == 1 else None
            if ((kind == "ask" and start.path == _PROJECT)
                    or (kind != "ask" and start.path != _PROJECT)):
                recording.issue("command.scope", "Command in wrong trace.",
                                start.path, start.sequence)
            if kind == "open_project":
                reason = start.details.get("reason")
                if reason == "create":
                    if len(create_batches) != 1 or len(markers) != 1:
                        recording.issue("creation.missing",
                                        "Ambiguous project creation.",
                                        start.path, start.sequence)
                        add("create_project", start, end, start.details)
                        continue
                    batch, marker = create_batches[0], markers[0]
                    completed = _batch_end(recording, batch, by_type)
                    if (completed is None or not batch.sequence <
                            completed.sequence < marker.sequence <
                            start.sequence):
                        recording.issue("creation.order",
                                        "Invalid creation event order.",
                                        start.path, start.sequence)
                    add("create_project", batch, end, {
                        "source_batch": batch.details,
                        "settings": marker.details,
                        "open": start.details,
                    })
                    continue
                if reason == "internal_reopen":
                    # Internal reopening must belong to a known user command.
                    continue
                if reason != "resume":
                    recording.issue("command.unsupported",
                                    "Unknown project opening reason.",
                                    start.path, start.sequence)
            add(kind, start, end, start.details)

    # The add command has explicit request/completion diagnostics but no span.
    pending = None
    add_events = [e for e in events if e.event_type in (
        "gui.sources.add.requested", "gui.sources.add.completed",
        "gui.sources.add.failed")]
    for event in add_events:
        if event.event_type.endswith(".requested"):
            if pending is not None:
                add("add_sources", pending, None, pending.details)
            pending = event
        elif pending is None:
            recording.issue("command.orphan", "Source addition has no "
                            "request.",
                            event.path, event.sequence)
        else:
            add("add_sources", pending, event, pending.details)
            pending = None
    if pending is not None:
        add("add_sources", pending, None, pending.details)

    for start in by_type.get("workspace.sync.started", []):
        end = _batch_end(recording, start, by_type)
        if any(op.kind == "add_sources" and op.start_sequence < start.sequence
               <= op.end_sequence for op in operations):
            continue
        # A subsequent refresh must follow this batch without crossing any
        # other explicit command or source batch. Never borrow a later refresh.
        next_bound = min((op.start_sequence for op in operations
                          if op.start_sequence > start.sequence),
                         default=float("inf"))
        next_bound = min(next_bound, min((e.sequence for e in events
                         if e.sequence > start.sequence and e.event_type in (
                             "workspace.prepare.started",
                             "workspace.sync.started", "gui.project.closed")),
                         default=float("inf")))
        refreshes = [e for e in by_type.get("gui.sources.refreshed", [])
                     if end is not None and end.sequence < e.sequence
                     < next_bound]
        add("sync_sources", start,
            refreshes[0] if len(refreshes) == 1 else None, start.details)
    for event in by_type.get("gui.project.closed", []):
        add("close_project", event, event, event.details)

    operations.sort(key=lambda op: op.start_sequence)
    previous = None
    for op in operations:
        if previous is not None and op.start_sequence <= previous.end_sequence:
            recording.issue("command.overlap", "Overlapping user commands.",
                            sequence=op.start_sequence)
        if previous is None or op.end_sequence > previous.end_sequence:
            previous = op
    # Known producer events must belong to reconstructed commands.
    boundaries = {
        "workspace.prepare.started": ("create_project", "add_sources"),
        "workspace.prepare.completed": ("create_project", "add_sources"),
        "workspace.source.copied": ("create_project", "add_sources"),
        "workspace.sync.started": ("add_sources", "sync_sources"),
        "workspace.sync.completed": ("add_sources", "sync_sources"),
        "workspace.sync.file": ("add_sources", "sync_sources"),
        "workspace.sync.detached": ("add_sources", "sync_sources"),
        "gui.sources.refreshed": ("add_sources", "sync_sources"),
        "gui.project.creation.settings": ("create_project",),
        "gui.project.open.started": ("create_project", "open_project"),
        "gui.conversation.catalog.opened": ("create_project", "open_project"),
        "turn.started": ("ask",),
        "state.checkpoint.committed": ("ask",),
        "gui.question.saved": ("ask",),
    }
    # Some GUI failures precede the core span, or occur during notification
    # after it. They still make the reference unsuitable for exact replay.
    failures = {
        "gui.question.failed", "gui.answer.cancelled",
        "gui.project.prepare.failed", "gui.project.resume.failed",
        "gui.sources.sync.failed", "gui.source_cards.regenerate.failed",
        "gui.conversation_create.failed", "gui.conversation_select.failed",
        "gui.conversation_rename.failed", "gui.conversation_delete.failed",
        "gui.conversation_reset.failed", "gui.command.failed",
    }
    cursor = 0
    for event in events:
        while (cursor < len(operations) and operations[cursor].end_sequence
               < event.sequence):
            cursor += 1
        op = (operations[cursor] if cursor < len(operations) and
              operations[cursor].start_sequence <= event.sequence else None)
        allowed = boundaries.get(event.event_type)
        if allowed is not None and (op is None or op.kind not in allowed):
            recording.issue("command.orphan", f"Orphan {event.event_type}.",
                            event.path, event.sequence)
        if event.event_type.startswith(("tool.choice.", "tool.plan.",
                                        "routing.choice.")) and (
                op is None or op.kind != "ask"):
            recording.issue("command.orphan", "Interaction outside question.",
                            event.path, event.sequence)
        if event.event_type in failures:
            recording.issue("command.failed", f"{event.event_type}.",
                            event.path, event.sequence)
    return tuple(operations)
