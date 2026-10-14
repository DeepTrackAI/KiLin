"""Extract performance measurements from existing QAChat conversation traces.

The helpers only read the numbered operation hierarchy stored by
``qachat.TraceLogger``. They intentionally discard prompts, generated text,
retrieved passages, file paths, token IDs, and arbitrary diagnostic fields.
No models are loaded and no QAChat project is modified.

Key Features
------------

- **Read-only extraction**

    Interpret persisted operation durations without changing trace files.

- **Turn summaries**

    Separate complete-turn, routing, retrieval, and generation timings.

- **Question comparisons**

    Summarize independent measurements from a repeatable question set.

Module Structure
----------------

- ``read_trace_operations``

    Read and validate safe operation measurements from a trace file.

- ``summarize_trace_turns``

    Calculate per-turn timings without double-counting nested operations.

- ``question_statistics``

    Group observed wall-clock times by question and trace configuration.

Examples
--------
>>> from assets.trace_benchmark import read_trace_operations
>>> rows = read_trace_operations("/path/to/conversation.json", "run_01")
>>> isinstance(rows, list)
True

"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import re
from statistics import median

__all__ = [
    "read_trace_operations",
    "summarize_trace_turns",
    "question_statistics",
]

_NUMBERED_OPERATION = re.compile(r"^\d+_(.+)$")
_TURN_NAME = re.compile(r"^turn_(\d+)$")

# Explicit known QAChat metrics prevent arbitrary diagnostic event names from
# appearing in exported timing-only reports. Extend this list deliberately
# when the core trace vocabulary gains another measured operation.
_TRACKED_EVENTS = frozenset({
    "session.ready",
    "model.embedding.load",
    "model.chat.load",
    "indexing",
    "source.read",
    "source.extract",
    "source.chunk",
    "index.code_graph.build",
    "index.load",
    "index.semantic.build",
    "index.lexical.build",
    "index.save",
    "embedding.calculate",
    "turn",
    "intent.classification",
    "history.selection",
    "retrieval.semantic",
    "retrieval.lexical",
    "retrieval.code",
    "retrieval.document",
    "llm.call",
    "llm.call.output",
})
_KNOWN_STATUSES = frozenset({"completed", "failed", "running", "started"})
_KNOWN_CATEGORIES = frozenset({
    "trace", "project", "sources", "indexing", "conversations", "runtime",
})
_KNOWN_STAGES = frozenset({
    "question", "embedding", "retrieval", "llm_input", "generation",
})
_NAME_BY_KEY = {
    name.replace(".", "_"): name
    for name in _TRACKED_EVENTS | _KNOWN_STAGES
}


def _operation_name(key: str) -> str:
    """Convert a persisted numbered key to a canonical event name.

    Parameters
    ----------
    key: str
        Persisted operation key with a numeric prefix.

    Returns
    -------
    str
        Canonical event/stage name, or an empty string for unknown keys.

    """

    match = _NUMBERED_OPERATION.fullmatch(key)
    return _NAME_BY_KEY.get(match.group(1), "") if match else ""


def _nonnegative(value: object) -> float | None:
    """Return a finite, nonnegative duration without coercing text or booleans.

    Parameters
    ----------
    value: object
        Numeric value from a saved trace operation.

    Returns
    -------
    float | None
        Valid numeric value, otherwise None.

    """

    if type(value) not in (float, int):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _operation_rows(
    operations: Mapping[str, object],
    *,
    label: str,
    category: str,
    stage: str,
    turn: int | None,
) -> list[dict[str, object]]:
    """Collect only permitted performance fields from readable operations.

    Parameters
    ----------
    operations: Mapping[str, object]
        Numbered operation nodes from one trace category or turn stage.
    label: str
        User-assigned run label with no filesystem location.
    category: str
        Trace session category.
    stage: str
        Turn stage name or empty text for initialization events.
    turn: int | None
        Persisted turn ordinal, if applicable.

    Returns
    -------
    list[dict[str, object]]
        Operation measurements, excluding all arbitrary diagnostic content.

    """

    rows: list[dict[str, object]] = []
    for key, operation in operations.items():
        if not isinstance(key, str) or not isinstance(operation, dict):
            continue
        event_name = _operation_name(key)
        event = operation.get("event")
        timing = operation.get("timing")
        details = operation.get("details")
        if event_name not in _TRACKED_EVENTS or not isinstance(event, dict):
            continue
        if not isinstance(timing, dict):
            timing = {}
        if not isinstance(details, dict):
            details = {}
        duration = _nonnegative(timing.get("operation_duration_ms"))
        if duration is None:
            continue
        status = event.get("status")
        if not isinstance(status, str) or status not in _KNOWN_STATUSES:
            status = "unknown"
        row: dict[str, object] = {
            "run": label,
            "category": category,
            "turn": turn,
            "stage": stage,
            "operation": event_name,
            "status": status,
            "duration_ms": duration,
        }
        if event_name == "llm.call.output":
            token_count = details.get("output_token_count")
            if type(token_count) is int and token_count >= 0:
                row["output_tokens"] = token_count
            reply_ms = _nonnegative(details.get("reply_duration_ms"))
            if reply_ms is not None:
                row["reply_ms"] = reply_ms
        rows.append(row)
    return rows


def read_trace_operations(
    path: str | Path, label: str
) -> list[dict[str, object]]:
    """Read timing-only operations from one QAChat conversation JSON file.

    Parameters
    ----------
    path: str | pathlib.Path
        Explicit path of a saved QAChat conversation trace.
    label: str
        Local, non-sensitive run label used to associate its measurements.

    Returns
    -------
    list[dict[str, object]]
        Measurements from top-level sessions and numbered conversation turns.

    Raises
    ------
    ValueError
        If the trace structure or label is invalid.
    OSError
        If the file cannot be read.
    json.JSONDecodeError
        If the file contains invalid JSON.

    Notes
    -----
    Only a known set of core operation names is extracted; no arbitrary
    diagnostic text, event details, or filesystem paths are returned.
    Incomplete spans have no duration and are omitted. Reopened sessions can
    have multiple top-level categories of the same type.

    """

    if not isinstance(label, str) or not label.strip():
        raise ValueError("A non-empty run label is required.")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("A QAChat trace must contain a JSON object.")
    sessions = payload.get("sessions")
    if not isinstance(sessions, dict):
        raise ValueError("The trace has no valid sessions mapping.")

    rows: list[dict[str, object]] = []
    for session in sessions.values():
        if not isinstance(session, dict):
            raise ValueError("The trace contains an invalid session.")
        category = session.get("category")
        if not isinstance(category, str) or category not in _KNOWN_CATEGORIES:
            raise ValueError("A trace session has an unknown category.")
        operations = session.get("operations", {})
        if not isinstance(operations, dict):
            raise ValueError("A trace session has invalid operations.")
        rows.extend(_operation_rows(
            operations,
            label=label,
            category=category,
            stage="",
            turn=None,
        ))
        if category != "conversations":
            continue
        turns = session.get("turns", {})
        if not isinstance(turns, dict):
            raise ValueError("Conversation turns must be a mapping.")
        for key, turn_data in turns.items():
            match = _TURN_NAME.fullmatch(str(key))
            if match is None or not isinstance(turn_data, dict):
                raise ValueError("The trace contains an invalid turn.")
            stages = turn_data.get("stages", {})
            if not isinstance(stages, dict):
                raise ValueError("Conversation stages must be a mapping.")
            for stage_key, stage_data in stages.items():
                if not isinstance(stage_data, dict):
                    raise ValueError("A conversation stage is invalid.")
                stage_ops = stage_data.get("operations", {})
                if not isinstance(stage_ops, dict):
                    raise ValueError("Stage operations must be a mapping.")
                stage = _operation_name(str(stage_key))
                if stage not in _KNOWN_STAGES:
                    raise ValueError("A conversation stage is unknown.")
                rows.extend(_operation_rows(
                    stage_ops,
                    label=label,
                    category=category,
                    stage=stage,
                    turn=int(match.group(1)),
                ))
    return rows


def summarize_trace_turns(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Summarize each recorded turn without adding nested span durations.

    Parameters
    ----------
    rows: Sequence[Mapping[str, object]]
        Safe operation records returned by ``read_trace_operations``.

    Returns
    -------
    list[dict[str, object]]
        One row per traced turn, with independent stage measurements.

    Notes
    -----
    The duration of ``turn`` already includes routing, retrieval, generation,
    persistence and trace overhead. Individual stages are reported without
    summing them into that duration. Missing stages remain None. Failed or
    running spans never contribute to completed timing statistics.

    """

    groups: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        turn = row.get("turn")
        label = row.get("run")
        if type(turn) is int and isinstance(label, str):
            groups[(label, turn)].append(row)

    output: list[dict[str, object]] = []
    for (label, turn), records in sorted(groups.items()):
        def values(event: str) -> list[float]:
            """Find completed durations of a named operation."""
            return [
                float(item["duration_ms"])
                for item in records
                if item.get("operation") == event
                and item.get("status") == "completed"
            ]

        def single(event: str) -> float | None:
            """Return the unique duration, or None for missing/duplicates."""
            durations = values(event)
            return durations[0] if len(durations) == 1 else None

        generations = values("llm.call.output")
        tokens = [
            int(item["output_tokens"])
            for item in records
            if item.get("operation") == "llm.call.output"
            and item.get("status") == "completed"
            and type(item.get("output_tokens")) is int
        ]
        generation_ms = sum(generations) if generations else None
        output_tokens = (
            sum(tokens) if len(tokens) == len(generations) else None
        )
        throughput = None
        if output_tokens is not None and generation_ms:
            throughput = output_tokens * 1000.0 / generation_ms
        output.append({
            "run": label,
            "turn": turn,
            "turn_ms": single("turn"),
            "routing_ms": single("intent.classification"),
            "semantic_ms": single("retrieval.semantic"),
            "lexical_ms": single("retrieval.lexical"),
            "code_ms": single("retrieval.code"),
            "generation_ms": generation_ms,
            "output_tokens": output_tokens,
            "output_tokens_per_second": throughput,
            "llm_calls": len(generations),
        })
    return output


def question_statistics(
    results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Calculate small-sample wall-clock summaries per standardized question.

    Parameters
    ----------
    results: Sequence[Mapping[str, object]]
        Benchmark records containing question_id, trace_enabled, wall_ms,
        and a completed status.

    Returns
    -------
    list[dict[str, object]]
        Sample count, median, minimum, and maximum in milliseconds.

    Notes
    -----
    The reported range is an observed spread, not a confidence interval.
    Failed questions are omitted rather than interpreted as zero latency.

    """

    groups: dict[tuple[str, bool], list[float]] = defaultdict(list)
    for record in results:
        name = record.get("question_id")
        enabled = record.get("trace_enabled")
        duration = _nonnegative(record.get("wall_ms"))
        if (
            isinstance(name, str)
            and type(enabled) is bool
            and record.get("status") == "completed"
            and duration is not None
        ):
            groups[(name, enabled)].append(duration)

    summaries: list[dict[str, object]] = []
    for (question_id, enabled), durations in sorted(groups.items()):
        summaries.append({
            "question_id": question_id,
            "trace_enabled": enabled,
            "repeats": len(durations),
            "median_ms": median(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
        })
    return summaries
