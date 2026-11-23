"""Record structured execution events and resumable conversation state.

``TraceLogger`` collects events and writes a grouped JSON representation using
replacement of a temporary file. Span contexts link start, completion, and
failure events. ``NullTraceLogger`` accepts the same calling pattern without
writing, while ``TraceTextStream`` converts diagnostic output into events.

Enabled tracing records diagnostic content and token identifiers whenever
they are available. Embedding-vector capture has a separate opt-in flag.
This is not a sanitization boundary: conversation resume state, exceptions,
and diagnostic text can contain questions, answers, source excerpts, paths,
and other sensitive information.
Artifacts returned by this implementation are inline JSON records rather than
separately stored artifact files.

Event writes are serialized within one logger. A stable companion-file lock
excludes simultaneous writers in the current or another process; this is not
a multi-file transaction or protection against nonparticipating programs.
I/O failure handling applies to event writes according to the configuration.
Direct conversation-state updates propagate write failures and restore their
previous in-memory state, preventing later events from committing an aborted
checkpoint.

Reopening an exact output path preserves previously saved readable turns and
conversation-level operations. New records start above the highest saved
number, including gaps in the existing numbering. Numbering is fixed for the
lifetime of each logger, so rewriting its accumulated events does not append
duplicate records. Turn identifiers group events within a logger run; reusing
an identifier in a later run creates a separate readable turn rather than
modifying a saved one. Event sequence numbers remain local to each run.

Key Features
------------

- **Canonical trace vocabulary**

    Provide canonical identifiers for failure policies and shared event types
    interpreted across KíLín components.

- **Structured spans**

    Associate events, timings, parent spans, and conversation-turn identifiers.

- **Grouped JSON output**

    Present lifecycle, preparation, conversation, and progress operations using
    the shared internal conversation-format contract.

- **Sequential trace reuse**

    Retain saved turns and conversation operations when reopening a file, with
    stable numbering for subsequent writes by the new logger.

- **Optional artifacts**

    Carry inline content and vectors when callers request them.

- **Ownership control**

    Distinguish caller-supplied sinks from loggers created by the factory and
    release the file lease only when the owning logger is closed.

Module Structure
----------------

- ``FAILURE_POLICY_*`` constants and ``FailurePolicy``

    Canonical identifiers controlling trace-write failure handling.

- ``TRACE_EVENT_*`` constants

    Canonical event identifiers interpreted across KíLín components.

- ``NullTraceLogger``

    No-op trace used when structured tracing is disabled.

- ``TraceConfig``

    Settings for one KíLín execution trace.

- ``TraceLogger``

    Collect ordered events and write a grouped conversation-trace JSON file.

- ``TraceSink``

    Type alias for a concrete or null trace logger.

- ``TraceTextStream``

    Turn ordinary backend text output into timestamped trace events.

- ``create_trace_logger``

    Return a trace and whether the caller owns its lifetime.

- ``trace_text_preview``

    Return three opening and three closing words for trace inspection.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import kilin
>>> logger = kilin.TraceLogger(
...     kilin.TraceConfig(enabled=True),
...     default_directory="/path/to/local/traces",
... )
>>> try:
...     with logger.span("example.operation", component="example"):
...         logger.emit("example.ready", component="example")
... finally:
...     logger.close()

Protect trace directories as potentially sensitive project data.

"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import sys
from threading import Lock
from time import perf_counter
import traceback
from typing import Final, Literal, TypeAlias
from uuid import uuid4
import warnings

from kilin.atomic_writes import atomic_write_text
from kilin.conversation_ownership import ConversationFileLock
from kilin.conversation_format import (
    CONVERSATION_RESUME_STATE_KEY,
    CONVERSATION_SESSION_CATEGORY,
    CONVERSATION_SESSION_KEY,
    CONVERSATION_SESSION_TITLE,
    conversation_state,
    new_conversation_session,
)

__all__ = [
    "FAILURE_POLICY_CHOICES",
    "FAILURE_POLICY_RAISE",
    "FAILURE_POLICY_WARN",
    "FailurePolicy",
    "TRACE_EVENT_ANSWER_ACTIVITY",
    "TRACE_EVENT_EMBEDDING_TOKENS_PREPARED",
    "TRACE_EVENT_LLM_CALL_INPUT",
    "TRACE_EVENT_LLM_CALL_OUTPUT",
    "TRACE_EVENT_LLM_CALL_OUTPUT_TOKENS",
    "TRACE_EVENT_PROGRESS_UPDATED",
    "NullTraceLogger",
    "TraceConfig",
    "TraceLogger",
    "TraceSink",
    "TraceTextStream",
    "create_trace_logger",
    "trace_text_preview",
]

FailurePolicy: TypeAlias = Literal["warn", "raise"]

FAILURE_POLICY_WARN: Final[FailurePolicy] = "warn"
FAILURE_POLICY_RAISE: Final[FailurePolicy] = "raise"
FAILURE_POLICY_CHOICES: Final[tuple[FailurePolicy, ...]] = (
    FAILURE_POLICY_WARN,
    FAILURE_POLICY_RAISE,
)

TRACE_EVENT_ANSWER_ACTIVITY: Final[str] = "answer.activity"
TRACE_EVENT_EMBEDDING_TOKENS_PREPARED: Final[str] = (
    "embedding.tokens.prepared"
)
TRACE_EVENT_LLM_CALL_INPUT: Final[str] = "llm.call.input"
TRACE_EVENT_LLM_CALL_OUTPUT: Final[str] = "llm.call.output"
TRACE_EVENT_LLM_CALL_OUTPUT_TOKENS: Final[str] = "llm.call.output.tokens"
TRACE_EVENT_PROGRESS_UPDATED: Final[str] = "progress.updated"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_EXACT_CONTENT_SECTIONS = {
    "chunks_artifact": (
        "01_retrieved_chunks",
        "Retrieved chunk previews supplied as evidence",
    ),
    "messages_artifact": (
        "01_structured_messages",
        "Structured messages before the tokenizer applies its chat template",
    ),
    "rendered_prompt_artifact": (
        "02_rendered_prompt",
        "Model-specific prompt produced by the tokenizer chat template",
    ),
    "input_token_ids_artifact": (
        "03_input_token_ids",
        "Integer token IDs passed directly to the generation model",
    ),
    "answer_artifact": (
        "01_generated_answer",
        "Exact text generated by the model",
    ),
}


@dataclass(frozen=True, slots=True)
class TraceConfig:
    """Settings for one KíLín execution trace.

    Parameters
    ----------
    enabled: bool, optional
        Whether events should be written. Defaults to ``False``.
    output_path: str | pathlib.Path | None, optional
        Exact conversation ``.json`` file or folder for the trace. ``None``
        uses the caller's project-specific default folder.
    include_embedding_vectors: bool, optional
        Whether callers are allowed to attach complete embedding vectors.
        Defaults to ``False``.
    failure_policy: FailurePolicy, optional
        Event-write I/O handling: warn disables later event writes and emits a
        warning, while raise converts the failure to RuntimeError. Direct
        conversation-state writes do not use this policy. Defaults to
        ``'warn'``.

    Attributes
    ----------
    enabled: bool
        Whether events should be written.
    output_path: str | pathlib.Path | None
        Exact conversation ``.json`` file or folder for the trace. ``None``
        uses the caller's project-specific default folder.
    include_embedding_vectors: bool
        Whether callers are allowed to attach complete embedding vectors.
        This is the only optional diagnostic payload category.
    failure_policy: FailurePolicy
        Event-write I/O handling: warn disables later event writes and emits a
        warning, while raise converts the failure to RuntimeError. Direct
        conversation-state writes do not use this policy.

    Raises
    ------
    ValueError
        If Boolean fields, failure policy, or output-path type fail
        validation.

    Notes
    -----
    Configuration validates values before a logger is created; it does not
    create directories or prove that an output path is writable. The trace is
    not encrypted or automatically redacted.

    """

    enabled: bool = False
    output_path: str | Path | None = None
    include_embedding_vectors: bool = False
    failure_policy: FailurePolicy = FAILURE_POLICY_WARN

    def __post_init__(self) -> None:
        """Reject invalid trace settings before any file is opened.

        Raises
        ------
        ValueError
            If enabled/vector flags are not exact booleans, policy is
            unsupported, or output_path has an invalid type.

        """

        if type(self.enabled) is not bool:
            raise ValueError("trace enabled must be True or False.")
        if type(self.include_embedding_vectors) is not bool:
            raise ValueError(
                "include_embedding_vectors must be True or False."
            )
        if self.failure_policy not in FAILURE_POLICY_CHOICES:
            raise ValueError("trace failure_policy must be warn or raise.")
        if self.output_path is not None and not isinstance(
            self.output_path, (str, Path)
        ):
            raise ValueError("trace output_path must be a path or None.")


class TraceLogger:
    """Collect ordered events and write a grouped conversation-trace JSON file.

    Construction resolves the output path, creates its parent directory,
    optionally loads prior grouped state, and emits the initial event. Event
    mutation is guarded by this instance's lock; different instances or
    processes are excluded by an exclusive conversation-file lock.

    Parameters
    ----------
    config: TraceConfig
        Enabled TraceConfig controlling output location, optional embedding
        vectors, and event-write I/O policy.
    default_directory: str | pathlib.Path
        Fallback output directory when the configuration does not name an exact
        JSON file.

    Attributes
    ----------
    config: TraceConfig
        Configuration used to create this logger.
    output_path: pathlib.Path
        Resolved JSON output path, potentially an explicitly reused
        conversation file.
    readable_path: pathlib.Path
        Alias of output_path; there is no separate readable-output file.
    trace_id: str
        Random correlation identifier for the current logger instance.
    enabled: bool
        True for this concrete sink; it does not reflect the private
        closed/failed flags.
    include_content: bool
        Always True for a concrete logger. NullTraceLogger uses False so
        callers can avoid preparing artifacts while tracing is disabled.
    include_token_ids: bool
        Always True for a concrete logger. NullTraceLogger uses False so
        callers can avoid preparing token arrays while tracing is disabled.
    include_embedding_vectors: bool
        Configured embedding-vector flag, independent of tracing detail.

    Methods
    -------
    new_id(prefix)
        Return a short, readable correlation identifier.
    emit(...)
        Record one event and refresh the readable output when required.
    exception(...)
        Record one failure with its complete Python traceback.
    settings_changed(...)
        Record only settings whose effective values changed.
    span(...)
        Record operation start, completion, failure, and duration.
    artifact(name, value, *, text=False)
        Keep an exact payload inline so the conversation stays portable.
    update_conversation(data)
        Merge resumable conversation state into this same JSON document.
    reset_conversation_state()
        Clear retained conversation state and readable conversation records.
    close(*, emit_event=True)
        Finish the logger and release exclusive file ownership.

    Raises
    ------
    ValueError
        If configuration is disabled or an existing conversation-trace file
        cannot be read as expected.
    OSError
        If the output directory cannot be created. Event-write errors follow
        the configured policy.

    Notes
    -----
    Enabled tracing captures available detailed diagnostics, potentially
    including sensitive data. Artifacts are inline dictionaries.
    There is no append-only log or multi-file transaction. Opening a second
    writer for the same canonical path raises ConversationLockedError.
    Saved readable turns and conversation operations are retained;
    new records use numbers above the highest saved keys. Turn identifiers and
    span identifiers group events only within one logger run. Event sequence
    numbers and top-level event counts also describe the current run, not the
    full lifetime of the conversation file.

    """

    _turn_offset: int
    _conversation_operation_offset: int

    enabled = True

    def __init__(
        self,
        config: TraceConfig,
        *,
        default_directory: str | Path,
    ) -> None:
        """Create one trace file and record its first event.

        Parameters
        ----------
        config: TraceConfig
            Enabled TraceConfig controlling output location, optional vector
            recording, and event-write I/O policy.
        default_directory: str | pathlib.Path
            Fallback output directory when the configuration does not name an
            exact JSON file.

        Raises
        ------
        ValueError
            If config.enabled is false or an existing conversation-trace
            document is invalid.
        OSError
            If the output directory cannot be created.

        Notes
        -----
        The first event is written during construction. Exact JSON paths can be
        reused; output is not guaranteed to be a new file. Saved sessions
        provide a fixed numbering baseline before that first event is emitted.
        Exclusive ownership is acquired before reading saved state or writing
        the first event and released on construction failure. Resumable state
        is copied separately so updates cannot modify the saved snapshot.

        """

        if not config.enabled:
            raise ValueError("TraceLogger requires an enabled TraceConfig.")

        self.config = config
        self.trace_id = self.new_id("trace")
        self._started_at = perf_counter()
        self._sequence = 0
        self._lock = Lock()
        self._closed = False
        self._failed = False
        self._events: list[dict[str, object]] = []
        self._conversation: dict[str, object] = {}
        self._previous_sessions: dict[str, object] = {}
        self._last_readable_write = 0.0
        self._active_span_id: ContextVar[str | None] = ContextVar(
            f"kilin_active_span_{id(self)}",
            default=None,
        )
        self.output_path = _trace_path(
            config.output_path,
            Path(default_directory),
            self.trace_id,
        )
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
            mode=0o700,
        )
        self.readable_path = self.output_path
        self._file_lock = ConversationFileLock(self.output_path)
        self._file_lock.acquire()
        try:
            if self.output_path.is_file():
                try:
                    existing = json.loads(
                        self.output_path.read_text(encoding="utf-8")
                    )
                    # Validate before the constructor writes its first event.
                    # Never replace invalid records with empty state.
                    self._conversation = conversation_state(existing)
                    self._previous_sessions = existing["sessions"]
                except (
                    OSError, json.JSONDecodeError, TypeError, KeyError
                ) as error:
                    raise ValueError(
                        "The conversation trace is invalid."
                    ) from error
            # Keep prior trace state immutable during current writes.
            self._conversation = deepcopy(self._conversation)
            previous_conversation = self._previous_sessions.get(
                CONVERSATION_SESSION_KEY
            )
            if not isinstance(previous_conversation, Mapping):
                previous_conversation = {}
            self._turn_offset = _numbered_record_offset(
                previous_conversation.get("turns"),
                r"turn_([0-9]+)",
            )
            self._conversation_operation_offset = _numbered_record_offset(
                previous_conversation.get("operations"),
                r"([0-9]+)_.+",
            )
            _restrict_permissions(self.output_path, 0o600)
            self.emit(
                "trace.started",
                component="trace",
                data={
                    # Historical fixed marker, not a configurable setting.
                    "mode": "forensic",
                    "output_path": str(self.output_path),
                    "include_embedding_vectors": (
                        config.include_embedding_vectors
                    ),
                    "environment": {
                        "python": sys.version,
                        "python_executable": sys.executable,
                        "platform": platform.platform(),
                        "process_id": os.getpid(),
                        "packages": _package_versions(),
                    },
                },
            )

        except BaseException:
            self._file_lock.release()
            raise

    @property
    def include_content(self) -> bool:
        """Return True because enabled tracing always retains exact content.

        Returns
        -------
        bool
            True for a concrete active trace logger.

        """

        return True

    @property
    def include_token_ids(self) -> bool:
        """Return True because enabled tracing retains available token IDs.

        Returns
        -------
        bool
            True for a concrete active trace logger.

        """

        return True

    @property
    def include_embedding_vectors(self) -> bool:
        """Return whether full embedding vectors may be stored.

        Returns
        -------
        bool
            Whether full embedding vectors may be stored.

        """

        return self.config.include_embedding_vectors

    def new_id(self, prefix: str) -> str:
        """Return a short, readable correlation identifier.

        Parameters
        ----------
        prefix: str
            Human-readable prefix sanitized before appending a random
            correlation identifier.

        Returns
        -------
        str
            A short, readable correlation identifier.

        """

        clean = _safe_name(prefix) or "item"
        return f"{clean}_{uuid4().hex[:12]}"

    def emit(
        self,
        event_type: str,
        *,
        component: str,
        data: Mapping[str, object] | None = None,
        status: str = "completed",
        turn_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        started_elapsed_ms: float | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Record one event and refresh the readable output when required.

        Parameters
        ----------
        event_type: str
            Nonblank machine-readable operation/event name, conventionally
            dot-separated.
        component: str
            Nonblank emitting subsystem name recorded on the event.
        data: Mapping[str, object] | None, optional
            Optional event or conversation-state mapping. Values are converted
            through the implemented JSON conversion helper. Defaults to
            ``None``.
        status: str, optional
            Event status label; the logger does not restrict it to a fixed
            enumeration. Defaults to ``'completed'``.
        turn_id: str | None, optional
            Optional identifier connecting this event to a conversation turn.
            Defaults to ``None``.
        span_id: str | None, optional
            Optional identifier for the operation span associated with the
            event. Defaults to ``None``.
        parent_span_id: str | None, optional
            Optional parent operation identifier. Span contexts infer it from
            the current context when omitted. Defaults to ``None``.
        started_elapsed_ms: float | None, optional
            Optional operation-start offset from logger creation, in
            milliseconds. Defaults to ``None``.
        duration_ms: float | None, optional
            Optional operation duration, in milliseconds. Defaults to ``None``.

        Raises
        ------
        ValueError
            If event_type or component is blank or non-string.
        RuntimeError
            If an event write encounters OSError and failure_policy is raise.

        Notes
        -----
        Closed or failed loggers return before validation. Enabled tracing
        retains all supplied event types, including token diagnostics. Event
        sequencing is serialized within this logger. Progress and activity
        updates can be throttled; other events trigger a refresh. Arbitrary
        payload content is not automatically redacted.

        """

        if self._closed or self._failed:
            return
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be non-empty text.")
        if not isinstance(component, str) or not component.strip():
            raise ValueError("component must be non-empty text.")
        normalized_event_type = event_type.strip()
        with self._lock:
            # A queued writer must not write after close releases the lease.
            if self._closed or self._failed:
                return
            if span_id is None:
                span_id = self._active_span_id.get()
            self._sequence += 1
            event: dict[str, object] = {
                "sequence": self._sequence,
                "timestamp_utc": _utc_timestamp(),
                "elapsed_ms": _milliseconds(perf_counter() - self._started_at),
                "trace_id": self.trace_id,
                "event_id": self.new_id("event"),
                "event_type": normalized_event_type,
                "component": component.strip(),
                "status": status,
            }
            if turn_id is not None:
                event["turn_id"] = turn_id
            if span_id is not None:
                event["span_id"] = span_id
            if parent_span_id is not None:
                event["parent_span_id"] = parent_span_id
            if started_elapsed_ms is not None:
                event["started_elapsed_ms"] = round(
                    float(started_elapsed_ms),
                    3,
                )
            if duration_ms is not None:
                event["duration_ms"] = round(float(duration_ms), 3)
            event["data"] = _json_value(dict(data or {}))
            self._write_event(event)

    def exception(
        self,
        event_type: str,
        error: BaseException,
        *,
        component: str,
        data: Mapping[str, object] | None = None,
        turn_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        started_elapsed_ms: float | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Record one failure with its complete Python traceback.

        Parameters
        ----------
        event_type: str
            Nonblank machine-readable operation/event name, conventionally
            dot-separated.
        error: BaseException
            Exception whose type, message, and formatted traceback are recorded
            or handled.
        component: str
            Nonblank emitting subsystem name recorded on the event.
        data: Mapping[str, object] | None, optional
            Optional event or conversation-state mapping. Values are converted
            through the implemented JSON conversion helper. Defaults to
            ``None``.
        turn_id: str | None, optional
            Optional identifier connecting this event to a conversation turn.
            Defaults to ``None``.
        span_id: str | None, optional
            Optional identifier for the operation span associated with the
            event. Defaults to ``None``.
        parent_span_id: str | None, optional
            Optional parent operation identifier. Span contexts infer it from
            the current context when omitted. Defaults to ``None``.
        started_elapsed_ms: float | None, optional
            Optional operation-start offset from logger creation, in
            milliseconds. Defaults to ``None``.
        duration_ms: float | None, optional
            Optional operation duration, in milliseconds. Defaults to ``None``.

        Notes
        -----
        Formats the complete traceback and merges error data into the event.
        Tracebacks can reveal paths or user data. The
        exception is recorded, not re-raised by this method; span handles
        propagation separately.

        """

        details = dict(data or {})
        details["error_type"] = type(error).__name__
        details["message"] = str(error)
        details["traceback"] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        self.emit(
            event_type,
            component=component,
            data=details,
            status="failed",
            turn_id=turn_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            started_elapsed_ms=started_elapsed_ms,
            duration_ms=duration_ms,
        )

    def settings_changed(
        self,
        event_type: str,
        *,
        component: str,
        before: Mapping[str, object],
        after: Mapping[str, object],
        data: Mapping[str, object] | None = None,
        turn_id: str | None = None,
    ) -> bool:
        """Record only settings whose effective values changed.

        The compact ``changes`` object keeps configuration history easy to
        inspect while the surrounding snapshot events retain complete state.
        Each entry names the earlier and effective value explicitly.

        Parameters
        ----------
        event_type: str
            Nonblank machine-readable operation/event name, conventionally
            dot-separated.
        component: str
            Nonblank emitting subsystem name recorded on the event.
        before: Mapping[str, object]
            Effective settings before the requested change.
        after: Mapping[str, object]
            Effective settings after the requested change.
        data: Mapping[str, object] | None, optional
            Optional event or conversation-state mapping. Values are converted
            through the implemented JSON conversion helper. Defaults to
            ``None``.
        turn_id: str | None, optional
            Optional identifier connecting this event to a conversation turn.
            Defaults to ``None``.

        Returns
        -------
        bool
            True if normalized effective values differ, otherwise False.

        Notes
        -----
        A True result means a difference was found, not proof that a file write
        occurred: the sink may be closed/failed or event output may be disabled
        by its policy. Added and removed setting keys participate in
        comparison.

        """

        changes = {
            key: {
                "before": _json_value(before.get(key)),
                "after": _json_value(after.get(key)),
            }
            for key in sorted(before.keys() | after.keys())
            if _json_value(before.get(key)) != _json_value(after.get(key))
        }
        if not changes:
            return False
        details = dict(data or {})
        details["changes"] = changes
        self.emit(
            event_type,
            component=component,
            data=details,
            turn_id=turn_id,
        )
        return True

    @contextmanager
    def span(
        self,
        event_type: str,
        *,
        component: str,
        data: Mapping[str, object] | None = None,
        turn_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> Iterator[str]:
        """Record operation start, completion, failure, and duration.

        Parameters
        ----------
        event_type: str
            Nonblank machine-readable operation/event name, conventionally
            dot-separated.
        component: str
            Nonblank emitting subsystem name recorded on the event.
        data: Mapping[str, object] | None, optional
            Optional event or conversation-state mapping. Values are converted
            through the implemented JSON conversion helper. Defaults to
            ``None``.
        turn_id: str | None, optional
            Optional identifier connecting this event to a conversation turn.
            Defaults to ``None``.
        parent_span_id: str | None, optional
            Optional parent operation identifier. Span contexts infer it from
            the current context when omitted. Defaults to ``None``.

        Yields
        ------
        str
            Generated span identifier used by the start/completion/failure
            events.

        Notes
        -----
        Use as a context manager. Parent span identity is taken from a
        ContextVar when not supplied. BaseException from the managed body is
        recorded and re-raised; the prior active span is restored in a finally
        block. Trace errors can themselves propagate according to the sink
        behavior.

        """

        span_id = self.new_id("span")
        if parent_span_id is None:
            parent_span_id = self._active_span_id.get()
        started_at = perf_counter()
        started_elapsed_ms = _milliseconds(started_at - self._started_at)
        token = self._active_span_id.set(span_id)
        try:
            self.emit(
                f"{event_type}.started",
                component=component,
                data=data,
                status="started",
                turn_id=turn_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                started_elapsed_ms=started_elapsed_ms,
            )
            try:
                yield span_id
            except BaseException as error:
                self.exception(
                    f"{event_type}.failed",
                    error,
                    component=component,
                    turn_id=turn_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    started_elapsed_ms=started_elapsed_ms,
                    duration_ms=(perf_counter() - started_at) * 1000,
                )
                raise
            else:
                self.emit(
                    f"{event_type}.completed",
                    component=component,
                    status="completed",
                    turn_id=turn_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    started_elapsed_ms=started_elapsed_ms,
                    duration_ms=(perf_counter() - started_at) * 1000,
                )
        finally:
            self._active_span_id.reset(token)

    def artifact(
        self,
        name: str,
        value: object,
        *,
        text: bool = False,
    ) -> dict[str, object] | None:
        """Keep an exact payload inline so the conversation stays portable.

        Parameters
        ----------
        name: str
            Human-readable artifact or operation label normalized for the
            stored key.
        value: object
            Python value to convert or package as trace content.
        text: bool, optional
            Whether to store str(value) as a text artifact instead of using the
            JSON conversion helper. Defaults to ``False``.

        Returns
        -------
        dict[str, object] | None
            Inline dictionary containing name, format, and inline_content, or
            None after close/failure.

        Raises
        ------
        ValueError
            If sanitizing the artifact name leaves no safe character.

        Notes
        -----
        This method returns a payload; it does not write a separate artifact
        file or emit an event. It does not enforce include_content itself.
        Callers should avoid preparing artifacts when tracing is disabled
        and protect output as potentially sensitive data.

        """

        if self._closed or self._failed:
            return None
        safe_name = _safe_name(name)
        if not safe_name:
            raise ValueError("artifact name must contain a safe character.")
        content = str(value) if text else _json_value(value)
        return {
            "name": safe_name,
            "format": "text" if text else "json",
            "inline_content": content,
        }

    def update_conversation(self, data: Mapping[str, object]) -> None:
        """Merge resumable conversation state into this same JSON document.

        Parameters
        ----------
        data: Mapping[str, object]
            Optional event or conversation-state mapping. Values are converted
            through the implemented JSON conversion helper.

        Raises
        ------
        OSError
            If the direct state rewrite fails.

        Notes
        -----
        Merges converted fields into a candidate state under the event lock
        and writes it atomically. If serialization or writing fails, the
        previous in-memory state is restored so subsequent events cannot
        accidentally commit the failed checkpoint. Omitted fields are kept.
        Checkpoint writes bypass the optional event-write failure policy.

        """

        changes = {
            str(key): _json_value(value) for key, value in data.items()
        }
        with self._lock:
            if self._closed:
                raise RuntimeError("This trace logger is closed.")
            previous = self._conversation
            self._conversation = {**previous, **changes}
            try:
                self._write_readable_trace()
            except BaseException:
                self._conversation = previous
                raise

    def reset_conversation_state(self) -> None:
        """Clear retained state for an archived conversation.

        Raises
        ------
        RuntimeError
            If the trace logger has already been closed.

        Notes
        -----
        Removes resumable conversation state, saved readable turns and
        conversation-level operations, and conversation events accumulated by
        the current logger run. Non-conversation sessions and events remain so
        a reset can continue using the same loaded models and project context.

        This method changes only in-memory logger state. It does not write,
        archive, delete, or recreate the output file. A subsequent event or
        conversation-state update writes the new projection. Turn and
        conversation-operation numbering restart from one after the reset.

        """

        with self._lock:
            if self._closed:
                raise RuntimeError("This trace logger is closed.")
            self._conversation = {}
            self._previous_sessions.pop(CONVERSATION_SESSION_KEY, None)
            self._events = [
                event
                for event in self._events
                if not _is_conversation_event(event)
            ]
            self._turn_offset = 0
            self._conversation_operation_offset = 0

    def close(self, *, emit_event: bool = True) -> None:
        """Finish the logger and release exclusive conversation ownership.

        Parameters
        ----------
        emit_event: bool, optional
            Whether to write the final ``trace.closed`` event. Set False
            when detaching an already persisted conversation for archival.
            Defaults to True.

        Notes
        -----
        Releases the OS and process-local locks even when strict event-write
        handling raises. Repeated calls do not write or release again.
        Caller-owned sinks must be closed by their owner, not by RAGSession.

        """

        if self._closed:
            return
        try:
            if emit_event:
                self.emit("trace.closed", component="trace")
        finally:
            with self._lock:
                self._closed = True
            self._file_lock.release()

    def _write_event(self, event: dict[str, object]) -> None:
        """Append an event in memory and conditionally refresh the output file.

        Parameters
        ----------
        event: dict[str, object]
            Structured event dictionary used for storage or readable
            projection.

        Notes
        -----
        Progress updates and answer-activity starts share one bounded
        rewrite interval; other events rewrite immediately. The checkpoint
        flushes all queued events. OSError is passed to
        _handle_io_failure. The event remains in memory when writing fails.

        """

        try:
            self._events.append(event)
            now = perf_counter()
            event_type = str(event["event_type"])
            if (
                event_type not in (
                    TRACE_EVENT_PROGRESS_UPDATED,
                    TRACE_EVENT_ANSWER_ACTIVITY,
                )
                or now - self._last_readable_write >= 0.5
            ):
                self._write_readable_trace()
                self._last_readable_write = now
        except OSError as error:
            self._handle_io_failure(error)

    def _write_readable_trace(self) -> None:
        """Write a compact, named hierarchy for human inspection.

        Notes
        -----
        Projects current events using the fixed numbering offsets captured
        when the file was opened. Merging into a deep copy preserves the loaded
        baseline and prevents nested turn, operation, or resume-state mappings
        from being changed by a write. Repeated writes replace this run's
        projection without duplicating it or overwriting earlier runs.

        A uniquely created temporary JSON file replaces the target while
        the exclusive companion-file lock is held. This is not an
        immutable snapshot history or multi-file transaction. Other KíLín
        writers cannot open this path until this logger releases ownership.

        """

        progress_checkpoints: list[dict[str, object]] = []
        execution_trace: list[dict[str, object]] = []
        for event in self._events:
            if event.get("event_type") == TRACE_EVENT_PROGRESS_UPDATED:
                progress_checkpoints.append(_progress_checkpoint(event))
            else:
                execution_trace.append(self._readable_event(event))
        current_sessions = _readable_sessions(
            execution_trace,
            progress_checkpoints,
            turn_offset=self._turn_offset,
            conversation_operation_offset=self._conversation_operation_offset,
        )
        sessions = deepcopy(self._previous_sessions)
        for key, value in current_sessions.items():
            if (
                key == CONVERSATION_SESSION_KEY
                and isinstance(sessions.get(key), dict)
                and isinstance(value, dict)
            ):
                existing = sessions[key]
                for field in ("turns", "operations"):
                    current_items = value.get(field)
                    saved_items = existing.setdefault(field, {})
                    if isinstance(current_items, dict) and isinstance(
                        saved_items, dict
                    ):
                        saved_items.update(current_items)
                continue
            candidate = key
            number = 2
            while candidate in sessions:
                candidate = f"{key}_{number}"
                number += 1
            sessions[candidate] = value
        conversation_session = sessions.setdefault(
            CONVERSATION_SESSION_KEY,
            new_conversation_session(),
        )
        # A trace may contain only turn events or only conversation-level
        # operations. Persist both groups so every saved record is canonical.
        conversation_session.setdefault("turns", {})
        conversation_session.setdefault("operations", {})
        conversation_session[CONVERSATION_RESUME_STATE_KEY] = (
            self._conversation
        )
        sessions = dict(
            sorted(
                sessions.items(),
                key=lambda item: (
                    _SESSION_ORDER.get(
                        (
                            str(item[1].get("category"))
                            if isinstance(item[1], dict)
                            else "runtime"
                        ),
                        99,
                    ),
                    item[0],
                ),
            )
        )
        payload = {
            "trace_id": self.trace_id,
            "updated_at_utc": _utc_timestamp(),
            "event_count": len(self._events),
            "execution_event_count": len(execution_trace),
            "progress_checkpoint_count": len(progress_checkpoints),
            "sessions": sessions,
        }
        content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        atomic_write_text(self.readable_path, content)

    def _readable_event(
        self,
        event: dict[str, object],
    ) -> dict[str, object]:
        """Return one event, resolving only its essential exact artifacts.

        Parameters
        ----------
        event: dict[str, object]
            Structured event dictionary used for storage or readable
            projection.

        Returns
        -------
        dict[str, object]
            One event, resolving only its essential exact artifacts.

        """

        readable = dict(event)
        data = event.get("data")
        if not isinstance(data, Mapping):
            return readable
        artifact_fields = {
            "retrieval.context.expanded": ("chunks_artifact",),
            TRACE_EVENT_LLM_CALL_INPUT: (
                "messages_artifact",
                "rendered_prompt_artifact",
                "input_token_ids_artifact",
            ),
            TRACE_EVENT_LLM_CALL_OUTPUT: ("answer_artifact",),
        }.get(str(event.get("event_type")), ())
        readable_data = dict(data)
        resolved: dict[str, object] = {}
        for field in artifact_fields:
            content = self._artifact_content(data.get(field))
            if content is not None:
                key, title = _EXACT_CONTENT_SECTIONS[field]
                section: dict[str, object] = {
                    "title": title,
                    "value": content,
                }
                if field == "input_token_ids_artifact" and isinstance(
                    content, list
                ):
                    section["token_count"] = len(content)
                resolved[key] = section
                readable_data.pop(field, None)
        readable["data"] = readable_data
        if resolved:
            readable["artifact_content"] = resolved
        return readable

    def _artifact_content(self, reference: object) -> object:
        """Resolve an inline or supported legacy file artifact.

        Parameters
        ----------
        reference: object
            Inline artifact reference, or a legacy artifact-file reference
            constrained to the output directory.

        Returns
        -------
        object
            Artifact content, or None for an unsupported, missing, unreadable,
            or invalid legacy reference.

        Notes
        -----
        Inline content takes precedence. Legacy paths must resolve inside the
        trace output directory. This is a compatibility read, not arbitrary
        remote artifact retrieval.

        """

        if not isinstance(reference, Mapping):
            return None
        if "inline_content" in reference:
            return reference["inline_content"]
        relative = reference.get("path")
        if not isinstance(relative, str):
            return None
        try:
            root = self.output_path.parent.resolve()
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                return None
            if reference.get("format") == "json":
                return json.loads(path.read_text(encoding="utf-8"))
            return path.read_text(encoding="utf-8")
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _handle_io_failure(self, error: OSError) -> None:
        """Apply the configured policy after trace storage fails.

        Parameters
        ----------
        error: OSError
            Exception whose type, message, and formatted traceback are recorded
            or handled.

        Raises
        ------
        RuntimeError
            If failure_policy is raise.

        Notes
        -----
        Marks the logger failed first. Warn policy emits a RuntimeWarning and
        returns; later emit calls become no-ops. Direct update_conversation
        calls do not pass through this handler.

        """

        self._failed = True
        if self.config.failure_policy == FAILURE_POLICY_RAISE:
            raise RuntimeError(
                f"Could not write KíLín trace: {error}"
            ) from error
        warnings.warn(
            f"KíLín trace disabled after a write failure: {error}",
            RuntimeWarning,
            stacklevel=3,
        )


class NullTraceLogger:
    """No-op trace used when structured tracing is disabled.

    Methods
    -------
    new_id(prefix)
        Return a usable correlation ID without storing it.
    emit(*args, **kwargs)
        Accept an event without writing it.
    exception(*args, **kwargs)
        Accept an exception without writing it.
    settings_changed(*args, **kwargs)
        Accept a settings comparison without writing it.
    span(*args, **kwargs)
        Preserve the span interface without writing events.
    artifact(*args, **kwargs)
        Return no artifact while tracing is disabled.
    update_conversation(data)
        Accept resumable state without writing it.
    reset_conversation_state()
        Keep resetting conversation state harmless while disabled.
    close()
        Keep closing a disabled trace harmless.

    Notes
    -----
    Content, token, vector, and enabled flags are false. Calls accept
    trace-shaped arguments without performing validation or file I/O.
    Correlation IDs and a span
    context are still available so callers need not branch on tracing state.

    """

    enabled = False
    include_content = False
    include_token_ids = False
    include_embedding_vectors = False
    output_path = None
    readable_path = None
    trace_id = "trace_disabled"

    def new_id(self, prefix: str) -> str:
        """Return a usable correlation ID without storing it.

        Parameters
        ----------
        prefix: str
            Human-readable prefix sanitized before appending a random
            correlation identifier.

        Returns
        -------
        str
            A usable correlation ID without storing it.

        """

        return f"{_safe_name(prefix) or 'item'}_{uuid4().hex[:12]}"

    def emit(self, *args: object, **kwargs: object) -> None:
        """Accept an event without writing it.

        Parameters
        ----------
        *args: object
            Positional arguments accepted for compatibility with the active
            trace interface; ignored by the null sink.
        **kwargs: object
            Keyword arguments accepted for compatibility with the active trace
            interface; ignored by the null sink.

        """

    def exception(self, *args: object, **kwargs: object) -> None:
        """Accept an exception without writing it.

        Parameters
        ----------
        *args: object
            Positional arguments accepted for compatibility with the active
            trace interface; ignored by the null sink.
        **kwargs: object
            Keyword arguments accepted for compatibility with the active trace
            interface; ignored by the null sink.

        """

    def settings_changed(self, *args: object, **kwargs: object) -> bool:
        """Accept a settings comparison without writing it.

        Parameters
        ----------
        *args: object
            Positional arguments accepted for compatibility with the active
            trace interface; ignored by the null sink.
        **kwargs: object
            Keyword arguments accepted for compatibility with the active trace
            interface; ignored by the null sink.

        Returns
        -------
        bool
            False unconditionally; settings are not compared by the null
            implementation.

        """

        return False

    @contextmanager
    def span(self, *args: object, **kwargs: object) -> Iterator[str]:
        """Preserve the span interface without writing events.

        Parameters
        ----------
        *args: object
            Positional arguments accepted for compatibility with the active
            trace interface; ignored by the null sink.
        **kwargs: object
            Keyword arguments accepted for compatibility with the active trace
            interface; ignored by the null sink.

        Yields
        ------
        str
            Generated correlation identifier without any trace events.

        Notes
        -----
        Use as a context manager. Exceptions from the caller's body propagate
        normally; no trace is recorded.

        """

        yield self.new_id("span")

    def artifact(
        self, *args: object, **kwargs: object
    ) -> dict[str, object] | None:
        """Return no artifact while tracing is disabled.

        Parameters
        ----------
        *args: object
            Positional arguments accepted for compatibility with the active
            trace interface; ignored by the null sink.
        **kwargs: object
            Keyword arguments accepted for compatibility with the active trace
            interface; ignored by the null sink.

        Returns
        -------
        dict[str, object] | None
            None unconditionally; the supplied value is not serialized.

        """

        return None

    def update_conversation(self, data: Mapping[str, object]) -> None:
        """Accept resumable state without writing it.

        Parameters
        ----------
        data: Mapping[str, object]
            Optional event or conversation-state mapping. Values are converted
            through the implemented JSON conversion helper.

        """

    def reset_conversation_state(self) -> None:
        """Keep resetting conversation state harmless while disabled.

        Notes
        -----
        A null sink retains neither resumable conversation state nor readable
        trace records, so there is nothing to clear.

        """

    def close(self) -> None:
        """Keep closing a disabled trace harmless."""


TraceSink: TypeAlias = TraceLogger | NullTraceLogger


class TraceTextStream:
    """Turn ordinary backend text output into timestamped trace events.

    Parameters
    ----------
    trace: TraceSink | Callable
        Fixed sink or zero-argument callable returning the currently active
        sink.
    stream_name: str
        Diagnostic stream label included in backend.output events, such as
        stdout or stderr.

    Methods
    -------
    write(text)
        Record complete non-empty lines and retain any final fragment.
    flush()
        Record the remaining text fragment.

    Notes
    -----
    This is the small write/flush adapter used for redirected backend
    diagnostics, not a complete TextIOBase implementation. Lines are buffered
    until newline or flush; blank lines are ignored. Output can contain
    sensitive backend text.

    """

    def __init__(
        self,
        trace: TraceSink | Callable[[], TraceSink],
        *,
        stream_name: str,
    ) -> None:
        """Keep incomplete lines until a newline or explicit flush.

        A provider lets a long-running GUI switch output from its startup trace
        to the currently open project's trace.

        Parameters
        ----------
        trace: TraceSink | Callable
            Fixed sink or zero-argument callable returning the currently active
            sink.
        stream_name: str
            Diagnostic stream label included in backend.output events, such as
            stdout or stderr.

        """

        self._trace_provider = trace if callable(trace) else lambda: trace
        self.stream_name = stream_name
        self._buffer = ""

    def write(self, text: str) -> int:
        """Record complete non-empty lines and retain any final fragment.

        Parameters
        ----------
        text: str
            Text to buffer, preview, or package as specified by the operation.

        Returns
        -------
        int
            Number of characters accepted from text, including any buffered
            fragment.

        Notes
        -----
        Complete nonblank lines are emitted immediately; a trailing partial
        line remains buffered. Sink errors propagate to the caller.

        """

        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)
        return len(text)

    def flush(self) -> None:
        """Record the remaining text fragment."""

        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""

    def _emit(self, line: str) -> None:
        """Ignore blank output and trace one complete line.

        Parameters
        ----------
        line: str
            One diagnostic output line or final buffered fragment.

        Notes
        -----
        Obtains the current sink when a provider callable was supplied, then
        emits a backend.output event. Blank text is ignored.

        """

        if line.strip():
            self._trace_provider().emit(
                "runtime.output",
                component="runtime",
                data={"stream": self.stream_name, "message": line},
            )


def create_trace_logger(
    trace: TraceConfig | TraceSink | None,
    *,
    default_directory: str | Path,
) -> tuple[TraceSink, bool]:
    """Return a trace and whether the caller owns its lifetime.

    Parameters
    ----------
    trace: TraceConfig | TraceSink | None
        Trace configuration or existing sink. None produces a NullTraceLogger
        in the factory.
    default_directory: str | pathlib.Path
        Fallback output directory when the configuration does not name an exact
        JSON file.

    Returns
    -------
    tuple[TraceSink, bool]
        Pair (sink, owns_sink). Ownership is True only for a new enabled logger
        constructed from TraceConfig; existing and null sinks are not owned.

    Raises
    ------
    ValueError
        If trace is not None, TraceConfig, TraceLogger, or NullTraceLogger.

    Notes
    -----
    None selects a null sink at this factory layer. RAGSession changes its own
    None default to an enabled configuration with full diagnostic recording
    before calling this factory. Logger-construction errors propagate.

    """

    if trace is None:
        return NullTraceLogger(), False
    if isinstance(trace, (TraceLogger, NullTraceLogger)):
        return trace, False
    if not isinstance(trace, TraceConfig):
        raise ValueError(
            "trace must be TraceConfig, TraceLogger, NullTraceLogger, or None."
        )
    if not trace.enabled:
        return NullTraceLogger(), False
    return (
        TraceLogger(trace, default_directory=default_directory),
        True,
    )


def _trace_path(
    configured: str | Path | None,
    default_directory: Path,
    trace_id: str,
) -> Path:
    """Resolve a configured trace path or choose a timestamped JSON filename.

    Parameters
    ----------
    configured: str | pathlib.Path | None
        Configured exact JSON output file, output directory, or None.
    default_directory: pathlib.Path
        Fallback output directory when the configuration does not name an exact
        JSON file.
    trace_id: str
        Logger correlation identifier. This parameter is retained by the helper
        signature but is not used to construct the path.

    Returns
    -------
    pathlib.Path
        Output path after home-directory expansion. Relative paths remain
        relative to the working directory.

    Notes
    -----
    Directory-based names use a UTC timestamp with microseconds. The helper
    neither checks for collisions nor reserves the file. An exact .json path
    may already exist. The trace_id argument does not participate in naming.

    """

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    name = f"conversation_{timestamp}.json"
    if configured is None:
        return default_directory.expanduser() / name
    path = Path(configured).expanduser()
    if path.suffix.lower() == ".json":
        return path
    return path / name


def _safe_name(value: str) -> str:
    """Return a portable single path component.

    Parameters
    ----------
    value: str
        Python value to convert or package as trace content.

    Returns
    -------
    str
        A portable single path component.

    Notes
    -----
    Sanitizes to the configured safe character set and strips leading/trailing
    punctuation. The result can be empty; callers decide whether to reject it
    or use a fallback.

    """

    return _SAFE_NAME.sub("-", value).strip("-._")


def _utc_timestamp() -> str:
    """Return one readable UTC timestamp with millisecond precision.

    Returns
    -------
    str
        One readable UTC timestamp with millisecond precision.

    """

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _milliseconds(seconds: float) -> float:
    """Round a duration without converting it to text.

    Parameters
    ----------
    seconds: float
        Duration in seconds to round for the stored millisecond value.

    Returns
    -------
    float
        Seconds multiplied by 1000 and rounded to three decimal places.

    """

    return round(seconds * 1000, 3)


def trace_text_preview(text: str) -> str:
    """Return three opening and three closing words for trace inspection.

    Parameters
    ----------
    text: str
        Text to buffer, preview, or package as specified by the operation.

    Returns
    -------
    str
        Three opening and three closing words for trace inspection.

    Notes
    -----
    Whitespace is normalized. Text with at most six words is returned in full;
    longer text keeps the first three and last three words separated by an
    ellipsis. This is a display preview, not the exact source text.

    """

    words = text.split()
    if len(words) <= 6:
        return " ".join(words)
    return " ".join([*words[:3], "…", *words[-3:]])


def _progress_checkpoint(event: Mapping[str, object]) -> dict[str, object]:
    """Flatten one sampled progress event for compact inspection.

    Parameters
    ----------
    event: Mapping[str, object]
        Structured event dictionary used for storage or readable projection.

    Returns
    -------
    dict[str, object]
        Compact checkpoint with selected event and progress fields for readable
        output.

    """

    data = event.get("data")
    details = dict(data) if isinstance(data, Mapping) else {}
    return {
        "sequence": event.get("sequence"),
        "timestamp_utc": event.get("timestamp_utc"),
        **details,
    }


_SESSION_TITLES = {
    "trace": "Trace setup and lifecycle",
    "project": "Project and workspace setup",
    "sources": "Reading, extracting, and chunking sources",
    "indexing": "Loading models, embedding, and saving indexes",
    CONVERSATION_SESSION_CATEGORY: CONVERSATION_SESSION_TITLE,
    "runtime": "Runtime output and errors",
}

_SESSION_ORDER = {
    "trace": 1,
    "project": 2,
    "sources": 3,
    "indexing": 4,
    CONVERSATION_SESSION_CATEGORY: 5,
    "runtime": 6,
}

_TURN_STAGE_TITLES = {
    "question": "Question and effective settings",
    "embedding": "Embedding the question",
    "retrieval": "Retrieving and ranking evidence",
    "llm_input": "Exact input for one LLM call",
    "generation": "One LLM call, exact output, and saved checkpoint",
}


def _numbered_record_offset(records: object, pattern: str) -> int:
    """Find the highest saved number in a readable record mapping.

    Parameters
    ----------
    records: object
        Decoded turn or operation mapping. A missing or non-mapping value
        contributes no reserved numbers.
    pattern: str
        Internal regular expression matching an entire record key. Its first
        capture group must contain the decimal number, for example
        ``r"turn_([0-9]+)"`` for readable turns.

    Returns
    -------
    int
        Highest matching number, or ``0`` when no numbered keys are present.
        Non-string and non-matching keys are ignored but remain in the saved
        mapping; this helper does not modify or validate their contents.

    Notes
    -----
    Use the highest number rather than the number of records: saved traces may
    contain gaps, legacy names, or keys wider than the usual padding. The
    result is captured once per logger so repeated writes use a fixed offset.

    """

    if not isinstance(records, Mapping):
        return 0
    numbers = (
        int(match.group(1))
        for key in records
        if isinstance(key, str)
        if (match := re.fullmatch(pattern, key)) is not None
    )
    return max(numbers, default=0)


def _is_conversation_event(event: Mapping[str, object]) -> bool:
    """Return whether an event belongs to the active conversation.

    Parameters
    ----------
    event: Mapping[str, object]
        Structured trace event to classify.

    Returns
    -------
    bool
        ``True`` for turn-associated events and conversation-level events.

    Notes
    -----
    A string ``turn_id`` is authoritative because turn events can use event
    names routed to several readable stages. Events without a turn identifier
    are conversation-specific only when their event-name prefix routes to the
    conversation session.

    """

    if isinstance(event.get("turn_id"), str):
        return True
    return _event_session(str(event.get("event_type", ""))) == (
        CONVERSATION_SESSION_CATEGORY
    )


def _readable_sessions(
    events: list[dict[str, object]],
    progress: list[dict[str, object]],
    *,
    turn_offset: int = 0,
    conversation_operation_offset: int = 0,
) -> dict[str, object]:
    """Group compact events under names that remain useful when folded.

    Parameters
    ----------
    events: list[dict[str, object]]
        Ordered non-progress trace events for the current logger run.
    progress: list[dict[str, object]]
        Sampled progress-checkpoint dictionaries to place beside their
        operations.
    turn_offset: int, optional
        Highest saved readable-turn number. New turns start at the next number.
        Defaults to ``0`` for a fresh trace.
    conversation_operation_offset: int, optional
        Highest saved conversation-level operation number. Applies only to
        conversation events without a turn identifier, not to operations inside
        a turn or another session category. Defaults to ``0``.

    Returns
    -------
    dict[str, object]
        Ordered mapping of readable session categories, operations, numbered
        turns/stages, and progress checkpoints.

    Notes
    -----
    Offsets are supplied from a fixed saved-session baseline. The helper
    builds only this run's projection; its caller merges that projection with
    saved sessions. Turn identifiers are grouped in first-seen order within
    this run. Repeated or interleaved events use their own assigned number
    when naming the turn, even after another turn begins.

    """

    sessions: dict[str, dict[str, object]] = {}
    span_nodes: dict[str, dict[str, object]] = {}
    turn_numbers: dict[str, int] = {}
    turn_count = turn_offset
    for event in events:
        operation_offset = 0
        event_type = str(event.get("event_type", "event"))
        turn_id = event.get("turn_id")
        if isinstance(turn_id, str):
            session = _session_bucket(sessions, CONVERSATION_SESSION_CATEGORY)
            turns = session.setdefault("turns", {})
            if not isinstance(turns, dict):
                continue
            if turn_id not in turn_numbers:
                turn_count += 1
                turn_numbers[turn_id] = turn_count
            turn_number = turn_numbers[turn_id]
            turn_key = f"turn_{turn_number:04d}"
            turn = turns.setdefault(
                turn_key,
                {
                    "title": _turn_title(event, turn_number),
                    "turn_id": turn_id,
                    "stages": {},
                },
            )
            if event_type == "turn.started":
                turn["title"] = _turn_title(event, turn_number)
            stages = turn["stages"]
            if not isinstance(stages, dict):
                continue
            stage_name = _turn_stage(event_type)
            stage_key = _numbered_key(
                list(_TURN_STAGE_TITLES).index(stage_name) + 1,
                stage_name,
            )
            stage = stages.setdefault(
                stage_key,
                {
                    "title": _TURN_STAGE_TITLES[stage_name],
                    "operations": {},
                },
            )
            operations = stage["operations"]
        else:
            category = _event_session(event_type)
            session = _session_bucket(sessions, category)
            operations = session.setdefault("operations", {})
            if category == CONVERSATION_SESSION_CATEGORY:
                operation_offset = conversation_operation_offset
        if isinstance(operations, dict):
            _append_readable_operation(
                operations,
                event,
                span_nodes,
                offset=operation_offset,
            )

    for session in sessions.values():
        turns = session.get("turns")
        if not isinstance(turns, dict):
            continue
        for turn in turns.values():
            if not isinstance(turn, dict):
                continue
            stages = turn.get("stages")
            if isinstance(stages, dict):
                turn["stages"] = dict(sorted(stages.items()))

    for checkpoint in progress:
        category = _progress_session(str(checkpoint.get("stage", "")))
        session = _session_bucket(sessions, category)
        checkpoints = session.setdefault("progress_checkpoints", {})
        if not isinstance(checkpoints, dict):
            continue
        number = len(checkpoints) + 1
        percent = checkpoint.get("percent")
        label = (
            f"{percent:g}_percent" if isinstance(percent, float) else "update"
        )
        checkpoints[_numbered_key(number, label)] = checkpoint

    return dict(
        sorted(
            sessions.items(),
            key=lambda item: _SESSION_ORDER[item[1]["category"]],
        )
    )


def _session_bucket(
    sessions: dict[str, dict[str, object]],
    category: str,
) -> dict[str, object]:
    """Return one consistently named top-level session.

    Parameters
    ----------
    sessions: dict[str, dict[str, object]]
        Mutable readable-session mapping indexed by ordered category names.
    category: str
        Recognized top-level trace category used by the session-order and title
        tables.

    Returns
    -------
    dict[str, object]
        One consistently named top-level session.

    Notes
    -----
    Mutates sessions by creating the category bucket when absent. Category must
    exist in the configured title/order tables.

    """

    key = _numbered_key(_SESSION_ORDER[category], category)
    return sessions.setdefault(
        key,
        {
            "title": _SESSION_TITLES[category],
            "category": category,
        },
    )


def _append_readable_operation(
    operations: dict[str, object],
    event: dict[str, object],
    span_nodes: dict[str, dict[str, object]],
    *,
    offset: int = 0,
) -> None:
    """Add one compact event, merging span start and finish records.

    Parameters
    ----------
    operations: dict[str, object]
        Readable operation mapping whose keys already identify stored entries.
    event: dict[str, object]
        Structured event dictionary used for storage or readable projection.
    span_nodes: dict[str, dict[str, object]]
        Mutable mapping from span identifiers to readable operations, used to
        merge lifecycle events within the current logger run.
    offset: int, optional
        Number of the last operation reserved by an earlier logger run in this
        group. Defaults to ``0``. The supplied operations contain only records
        from the current run.

    Notes
    -----
    Mutates operations and span_nodes. A recognized span start creates a node;
    completion/failure updates the existing node rather than appending a full
    duplicate lifecycle record.

    """

    event_type = str(event.get("event_type", "event"))
    span_id = event.get("span_id")
    lifecycle = event_type.rsplit(".", 1)[-1]
    if isinstance(span_id, str) and lifecycle in {
        "started",
        "completed",
        "failed",
    }:
        base = event_type.rsplit(".", 1)[0]
        node = span_nodes.get(span_id)
        if lifecycle == "started" and node is None:
            node = _compact_event(event)
            key = _unique_operation_key(operations, base, offset=offset)
            operations[key] = node
            span_nodes[span_id] = node
        if lifecycle == "started" and node is not None:
            summary = node.get("event")
            if isinstance(summary, dict):
                summary["status"] = "running"
            return
        summary = node.get("event") if isinstance(node, dict) else None
        if node is not None and isinstance(summary, dict):
            summary["finished_at_utc"] = event.get("timestamp_utc")
            summary["status"] = lifecycle
            if "duration_ms" in event:
                timing = node.setdefault("timing", {})
                if isinstance(timing, dict):
                    timing["operation_duration_ms"] = event["duration_ms"]
            data = event.get("data")
            if isinstance(data, Mapping) and data:
                node["result"] = data
            return

    key = _unique_operation_key(operations, event_type, offset=offset)
    operations[key] = _compact_event(event)


def _compact_event(event: Mapping[str, object]) -> dict[str, object]:
    """Remove repeated transport metadata from a readable event.

    Parameters
    ----------
    event: Mapping[str, object]
        Structured event dictionary used for storage or readable projection.

    Returns
    -------
    dict[str, object]
        Readable dictionary of event status/identity, timing, nonredundant
        details, and resolved artifact content.

    """

    compact: dict[str, object] = {
        "event": {
            "status": event.get("status"),
            "sequence": event.get("sequence"),
            "timestamp_utc": event.get("timestamp_utc"),
        },
    }
    summary = compact["event"]
    if isinstance(summary, dict):
        for field in ("span_id", "parent_span_id"):
            if field in event:
                summary[field] = event[field]
    timing: dict[str, object] = {}
    if "duration_ms" in event:
        timing["operation_duration_ms"] = event["duration_ms"]
    if timing:
        compact["timing"] = timing
    data = event.get("data")
    if isinstance(data, Mapping) and data:
        clean_data = dict(data)
        operation_duration = event.get("duration_ms")
        if isinstance(operation_duration, (int, float)):
            for field, value in tuple(clean_data.items()):
                if (
                    field.endswith("_duration_ms")
                    and isinstance(value, (int, float))
                    and round(float(value), 3)
                    == round(float(operation_duration), 3)
                ):
                    clean_data.pop(field)
        artifact_content = event.get("artifact_content")
        if isinstance(artifact_content, Mapping):
            for name in artifact_content:
                clean_data.pop(f"{name}_artifact", None)
        if clean_data:
            compact["details"] = clean_data
    if "artifact_content" in event:
        compact["exact_content"] = event["artifact_content"]
    return compact


def _unique_operation_key(
    operations: Mapping[str, object],
    event_type: str,
    *,
    offset: int = 0,
) -> str:
    """Return a numbered operation name that is readable when collapsed.

    Parameters
    ----------
    operations: Mapping[str, object]
        Sequentially generated operation records for the current logger run.
        Previously saved records are not included in this mapping.
    event_type: str
        Nonblank machine-readable operation/event name, conventionally
        dot-separated.
    offset: int, optional
        Highest operation number reserved by a previously saved group.
        Defaults to ``0``.

    Returns
    -------
    str
        Next operation key after the reserved offset and current-run records.

    """

    number = offset + len(operations) + 1
    return _numbered_key(number, event_type.replace(".", "_"))


def _numbered_key(number: int, name: str) -> str:
    """Create a stable JSON key with a visible order and title.

    Parameters
    ----------
    number: int
        One-based display order or turn number supplied by the grouping code.
    name: str
        Human-readable artifact or operation label normalized for the stored
        key.

    Returns
    -------
    str
        Two-digit-minimum order prefix followed by a lowercase sanitized title,
        using item when no title characters remain.

    """

    clean = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"{number:02d}_{clean or 'item'}"


def _turn_title(event: Mapping[str, object], number: int) -> str:
    """Name a conversation turn from its exact question when available.

    Parameters
    ----------
    event: Mapping[str, object]
        Structured event dictionary used for storage or readable projection.
    number: int
        One-based display order or turn number supplied by the grouping code.

    Returns
    -------
    str
        Human-readable title derived from event question text when present,
        otherwise a numbered turn label.

    """

    data = event.get("data")
    question = data.get("question") if isinstance(data, Mapping) else None
    if isinstance(question, str) and question.strip():
        clean = " ".join(question.split())
        return f"Turn {number}: {clean[:100]}"
    return f"Turn {number}"


def _event_session(event_type: str) -> str:
    """Choose the human session containing one non-turn event.

    Parameters
    ----------
    event_type: str
        Nonblank machine-readable operation/event name, conventionally
        dot-separated.

    Returns
    -------
    str
        Recognized category name selected from the event-type prefixes and
        fallback policy.

    """

    if event_type.startswith(("project.", "workspace.", "gui.project")):
        return "project"
    if event_type.startswith(("source.", "document.")):
        return "sources"
    if event_type.startswith(("index.", "embedding.", "model.")):
        return "indexing"
    if event_type.startswith("conversation."):
        return CONVERSATION_SESSION_CATEGORY
    if event_type.startswith(("trace.", "gui.trace")):
        return "trace"
    return "runtime"


def _turn_stage(event_type: str) -> str:
    """Choose a nested conversation phase for one turn event.

    Parameters
    ----------
    event_type: str
        Nonblank machine-readable operation/event name, conventionally
        dot-separated.

    Returns
    -------
    str
        Recognized conversation-stage name selected from the event-type routing
        rules.

    """

    if event_type.startswith(("embedding.", "query.")):
        return "embedding"
    if event_type.startswith(
        ("retrieval.", "ranking.", "search.", "evidence.")
    ):
        return "retrieval"
    if event_type == TRACE_EVENT_LLM_CALL_INPUT:
        return "llm_input"
    if event_type.startswith("llm.call") or event_type == (
        "state.checkpoint.committed"
    ):
        return "generation"
    return "question"


def _progress_session(stage: str) -> str:
    """Place sampled progress beside the operation it describes.

    Parameters
    ----------
    stage: str
        Preparation stage label mapped to a readable session category.

    Returns
    -------
    str
        Top-level category associated with the supplied preparation stage.

    """

    if any(word in stage for word in ("read", "extract", "chunk", "source")):
        return "sources"
    if any(word in stage for word in ("embed", "index", "saving")):
        return "indexing"
    return "project"


def _json_value(value: object) -> object:
    """Convert supported Python objects into trace-serializable values.

    Parameters
    ----------
    value: object
        Python value to convert or package as trace content.

    Returns
    -------
    object
        Converted scalar, mapping, list, or string representation accepted by
        the logger's JSON-writing path.

    Notes
    -----
    Dataclasses and paths are converted, sequences become lists, sets are
    ordered by string representation, and other objects fall back to str.
    Non-finite floats are not rejected; arbitrary object string representations
    need not be deterministic. This conversion is not redaction or strict
    portable JSON validation.

    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=str)
    return str(value)


def _restrict_permissions(path: Path, mode: int) -> None:
    """Apply private local permissions where the platform supports them.

    Parameters
    ----------
    path: pathlib.Path
        Filesystem path whose permissions should be restricted.
    mode: int
        POSIX-style permission mask passed to chmod, for example 0o600.

    Notes
    -----
    OSError from chmod is ignored, allowing tracing to proceed on platforms
    that cannot apply the requested permission mode. Private permissions are
    therefore best-effort, not a guaranteed protection boundary.

    """

    try:
        path.chmod(mode)
    except OSError:
        pass


def _package_versions() -> dict[str, str | None]:
    """Return versions of packages that can affect a KíLín trace.

    Returns
    -------
    dict[str, str | None]
        Versions of packages that can affect a KíLín trace.

    """

    names = (
        "kilin",
        "huggingface-hub",
        "numpy",
        "pypdf",
        "mlx-lm",
        "mlx-embeddings",
        "torch",
        "transformers",
        "PySide6",
    )
    resolved: dict[str, str | None] = {}
    for name in names:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = None
    return resolved
