"""Coordinate retrieval, chat generation, and resumable project conversations.

``RAGSession`` opens a configured project, loads an embedding session, reuses
or builds source chunks and retrieval indexes, then loads a chat session.
For each question it combines semantic, lexical, and code-based rankings,
expands explicitly requested context, and formats evidence into the model's
user prompt. Returned ``Evidence`` records identify the supplied excerpts;
the module does not independently verify the generated answer or citations.

Conversation turns retain questions, answers, evidence, and effective turn
parameters. Persistence uses a resumable state in a conversation JSON record
or in the supplied trace sink. The session has side effects: initialization
may create files, rebuild indexes, load models, and download missing model
files according to the profile. ``trace=None`` creates enabled forensic
tracing in this class, unlike the low-level chat and embedding classes.

History and expanded evidence have no automatic summarization or shared
prompt-token budget. A retrieval count limits the initial fused ranking, not
necessarily the final expanded evidence set. Explicit cleanup releases owned
models and trace resources.

Key Features
------------

- **Project orchestration**

    Connect scanning, source extraction, token-aware chunking, indexing,
    retrieval, and chat generation.

- **Persistent indexes**

    Compare selected-source manifests and configuration metadata before reusing
    saved retrieval data.

- **Inspectable evidence**

    Retain chunk provenance and ranking information alongside each answer.

- **Conversation state**

    Restore or switch saved conversations while maintaining explicit
    model-session ownership.

- **Progress and tracing**

    Emit stage callbacks and structured events, with content retention
    controlled by trace settings.

Module Structure
----------------

- ``ConversationTurn``

    One retained question, generated answer, and supplied evidence.

- ``Evidence``

    One source chunk supplied to the chat model.

- ``ProgressCallback``

    Callback type receiving a ``ProgressEvent``.

- ``ProgressEvent``

    One structured progress update emitted while opening a RAG session.

- ``RAGSession``

    A loaded project, reusable indexes, models, and conversation.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import qachat
>>> session = qachat.RAGSession(
...     "/path/to/configured/workspace",
...     trace=qachat.TraceConfig(enabled=False),
... )
>>> try:
...     answer = session.ask("How does this repository select files?")
... finally:
...     session.close()

Disabling tracing does not disable conversation checkpoint persistence.
The workspace and model dependencies must already be suitable for its profile.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from qachat.chat import ChatSession
from qachat.chunking import Chunk, chunk_source_units
from qachat.embeddings import EmbeddingSession
from qachat.project import QAProject, open_project
from qachat.query_formats import (
    LITERATURE_REPOSITORY_INSTRUCTION,
    SOFTWARE_REPOSITORY_INSTRUCTION,
    format_qwen3_query,
)
from qachat.repository import (
    build_repository_manifest,
    compare_repository_manifests,
    scan_repository,
)
from qachat.retrievals import (
    CodeGraph,
    PersistentIndexes,
    build_code_graph,
    build_lexical_index,
    build_semantic_index,
    code_search,
    expand_retrieval_context,
    hybrid_search,
    lexical_search,
    load_indexes,
    plan_retrieval,
    prepare_persistent_indexes,
    save_indexes,
    semantic_search,
)
from qachat.runtime import SelectedBackend, choose_backend
from qachat.sources import extract_source_units, read_repository_files
from qachat.trace_logging import (
    NullTraceLogger,
    TraceConfig,
    TraceSink,
    create_trace_logger,
    trace_text_preview,
)

__all__ = [
    "ConversationTurn",
    "Evidence",
    "ProgressCallback",
    "ProgressEvent",
    "RAGSession",
]

_INDEX_METADATA_VERSION = 1
_EMBEDDING_PROGRESS_BATCH_SIZE = 64
_PROGRESS_MIN_INTERVAL_SECONDS = 0.15
_PROGRESS_TRACE_PERCENT_STEP = 5.0
_SYSTEM_PROMPT = (
    "Answer only from the supplied project evidence. Put source numbers such "
    "as [1] and [2] directly after supported claims. Do not write a separate "
    "reference list or copy bibliographic text. Say when evidence "
    "is insufficient."
)


@dataclass(frozen=True, slots=True)
class Evidence:
    """One source chunk supplied to the chat model.

    Parameters
    ----------
    citation: int
        One-based citation number used in the answer prompt.
    path: pathlib.Path
        Source path relative to the configured source folder.
    start_line: int
        One-based source line range. For a notebook cell (``cell`` is not
        None), these are local to that cell -- every cell starts at line 1 --
        not a position in the underlying ``.ipynb`` file, which has no
        meaningful line numbering a Jupyter reader would recognize.
    end_line: int
        One-based source line range. For a notebook cell (``cell`` is not
        None), these are local to that cell -- every cell starts at line 1 --
        not a position in the underlying ``.ipynb`` file, which has no
        meaningful line numbering a Jupyter reader would recognize.
    symbol: str | None
        Function, class, or section name when available.
    text: str
        Exact retrieved source text.
    cell: int | None, optional
        One-based notebook cell number, when this evidence came from a
        ``.ipynb`` source. None for every other source type. Required to tell
        two notebook cells apart when they happen to share the same local
        start_line/end_line range. Defaults to ``None``.

    Attributes
    ----------
    citation: int
        One-based citation number used in the answer prompt.
    path: pathlib.Path
        Source path relative to the configured source folder.
    start_line: int
        One-based source line range. For a notebook cell (``cell`` is not
        None), these are local to that cell -- every cell starts at line 1 --
        not a position in the underlying ``.ipynb`` file, which has no
        meaningful line numbering a Jupyter reader would recognize.
    end_line: int
        One-based source line range. For a notebook cell (``cell`` is not
        None), these are local to that cell -- every cell starts at line 1 --
        not a position in the underlying ``.ipynb`` file, which has no
        meaningful line numbering a Jupyter reader would recognize.
    symbol: str | None
        Function, class, or section name when available.
    text: str
        Exact retrieved source text.
    cell: int | None
        One-based notebook cell number, when this evidence came from a
        ``.ipynb`` source. None for every other source type. Required to tell
        two notebook cells apart when they happen to share the same local
        start_line/end_line range.

    Notes
    -----
    Line numbers refer to extractor coordinates. Notebook lines are cell-local;
    PDF lines are page-local. This record stores cell but has no section/page
    field, so not all source-unit page provenance survives into Evidence. The
    model prompt currently includes path/line/symbol but not the notebook cell
    identifier.

    """

    citation: int
    path: Path
    start_line: int
    end_line: int
    symbol: str | None
    text: str
    cell: int | None = None


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One structured progress update emitted while opening a RAG session.

    Parameters
    ----------
    stage: str
        Stable machine-readable stage name for notebook or GUI clients.
    percent: float
        Approximate overall session-preparation progress from 0 to 100.
    message: str
        Short human-readable description of the current work.
    current: int | None, optional
        Optional item counts when a stage has a meaningful count. Defaults to
        ``None``.
    total: int | None, optional
        Optional item counts when a stage has a meaningful count. Defaults to
        ``None``.
    remaining: int | None, optional
        Number of items still remaining in the current counted stage. Defaults
        to ``None``.
    elapsed_seconds: float | None, optional
        Elapsed time for the current counted stage. Defaults to ``None``.
    eta_seconds: float | None, optional
        Estimated seconds remaining for the current counted stage. This is an
        estimate based on the average throughput observed so far. Defaults to
        ``None``.

    Attributes
    ----------
    stage: str
        Stable machine-readable stage name for notebook or GUI clients.
    percent: float
        Approximate overall session-preparation progress from 0 to 100.
    message: str
        Short human-readable description of the current work.
    current: int | None
        Optional item counts when a stage has a meaningful count.
    total: int | None
        Optional item counts when a stage has a meaningful count.
    remaining: int | None
        Number of items still remaining in the current counted stage.
    elapsed_seconds: float | None
        Elapsed time for the current counted stage.
    eta_seconds: float | None
        Estimated seconds remaining for the current counted stage. This is an
        estimate based on the average throughput observed so far.

    """

    stage: str
    percent: float
    message: str
    current: int | None = None
    total: int | None = None
    remaining: int | None = None
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None


ProgressCallback = Callable[[ProgressEvent], None]
AnswerPhaseCallback = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One retained question, generated answer, and supplied evidence.

    The record stores what was supplied and generated; it does not certify
    answer correctness or citation grounding.

    Parameters
    ----------
    timestamp: str
        UTC time in ISO 8601 form.
    question: str
        Original user question.
    answer: str
        Chat-model response.
    evidence: tuple[Evidence, ...]
        Source chunks supplied for this answer.
    parameters: dict[str, object]
        Exact project and model settings used for this turn.

    Attributes
    ----------
    timestamp: str
        UTC time in ISO 8601 form.
    question: str
        Original user question.
    answer: str
        Chat-model response.
    evidence: tuple[Evidence, ...]
        Source chunks supplied for this answer.
    parameters: dict[str, object]
        Exact project and model settings used for this turn.

    Notes
    -----
    The dataclass is frozen, but its parameters dictionary is still mutable.
    Parameter fields describe effective application settings, not every
    backend-internal generation option.

    """

    timestamp: str
    question: str
    answer: str
    evidence: tuple[Evidence, ...]
    parameters: dict[str, object]


class RAGSession:
    """A loaded project, reusable indexes, models, and conversation.

    Parameters
    ----------
    project_directory: str | pathlib.Path
        QAChat project folder containing profile.ini.
    progress_callback: ProgressCallback | None, optional
        Optional callback receiving ProgressEvent records during preparation.
        Callback exceptions propagate. Defaults to ``None``.
    trace: TraceConfig | TraceSink | None, optional
        Trace configuration or caller-owned sink. None requests a new enabled
        forensic logger. TraceConfig(enabled=False) disables tracing but not
        conversation persistence. Defaults to ``None``.
    conversation_path: str | pathlib.Path | None, optional
        Optional conversation_*.json path under project logs. Relative paths
        start at the project root; None chooses a new path under logs/traces.
        When tracing is enabled, writes go through the sink, so configure its
        output path consistently with this path. Defaults to ``None``.

    Attributes
    ----------
    project: QAProject
        Resolved project paths and settings.
    backend: "mlx" | "torch"
        Selected model backend.
    index_status: str
        Either reused or rebuilt for this opening.
    turns: list[ConversationTurn]
        Previous and current persisted interactions.
    conversation_path: pathlib.Path
        Path used when loading conversation state and when saving without an
        enabled trace sink.
    trace: TraceSink
        Active structured trace sink, potentially caller-owned.
    chat_session: ChatSession
        Loaded chat model and its in-memory formatted conversation.
    embedding_session: EmbeddingSession
        Loaded embedding model and tokenizer.
    chunks: list[Chunk]
        Chunks used by the current retrieval indexes.
    source_changes: dict[str, object]
        Selected-source manifest comparison, including configuration-change
        information when provided.

    Methods
    -------
    ask(...)
        Retrieve evidence, generate an answer, and save the turn.
    switch_conversation(conversation_path, *, trace)
        Load a conversation without reloading models or indexes.
    reset()
        Archive the active conversation and clear it from memory.
    close()
        Release both models while keeping indexes and logs on disk.

    Raises
    ------
    ValueError
        If the callback, profile, conversation state, source/index data, or a
        validated model setting is invalid.
    OSError
        If required project/source/state files cannot be read or written.
    ImportError
        If a required backend/reader dependency is unavailable.

    Notes
    -----
    Construction performs I/O and loads models. A rebuilt index reads selected
    source contents and writes index/manifest metadata. Forensic traces and
    resumable state can contain complete source excerpts and conversations. No
    automatic evidence-token budget, answer validation, or history
    summarization is supplied. Close explicitly to release owned resources.

    """

    project: QAProject
    backend: SelectedBackend
    index_status: str
    turns: list[ConversationTurn]

    def __init__(
        self: RAGSession,
        project_directory: str | Path,
        *,
        progress_callback: ProgressCallback | None = None,
        trace: TraceConfig | TraceSink | None = None,
        conversation_path: str | Path | None = None,
    ) -> None:
        """Open a project, its models, indexes, and saved conversation.

        Parameters
        ----------
        project_directory: str | pathlib.Path
            QAChat project folder containing profile.ini.
        progress_callback: ProgressCallback | None, optional
            Optional callback receiving ProgressEvent records during
            preparation. Callback exceptions propagate. Defaults to ``None``.
        trace: TraceConfig | TraceSink | None, optional
            Trace configuration or caller-owned sink. None requests a new
            enabled forensic logger. TraceConfig(enabled=False) disables
            tracing but not conversation persistence. Defaults to ``None``.
        conversation_path: str | pathlib.Path | None, optional
            Optional conversation_*.json path under project logs. Relative
            paths start at the project root; None chooses a new path under
            logs/traces. When tracing is enabled, writes go through the sink,
            so configure its output path consistently with this path. Defaults
            to ``None``.

        Raises
        ------
        ValueError
            If a non-None progress callback is not callable; profile, source,
            index, and model validation errors also propagate.

        Notes
        -----
        Attempts to close partially constructed model sessions after a
        preparation exception. Trace construction occurs before model loading.
        Heavy dependencies and model downloads are governed by the selected
        profile.

        """

        session_started_at = perf_counter()
        if progress_callback is not None and not callable(progress_callback):
            raise ValueError("progress_callback must be callable or None.")

        self.progress_callback = progress_callback
        self._progress_last_emit: dict[str, float] = {}
        self._progress_last_trace_percent: dict[str, float] = {}
        self.project = open_project(project_directory)
        self.conversation_path = self._resolve_conversation_path(
            conversation_path
        )
        if trace is None:
            trace = TraceConfig(enabled=True, mode="forensic")
        self.trace, self._owns_trace = create_trace_logger(
            trace,
            default_directory=self.conversation_path.parent,
        )
        self.trace.emit(
            "session.started",
            component="rag",
            data={
                "project_directory": str(self.project.root),
                "source_directory": str(self.project.source),
                "conversation_path": str(self.conversation_path),
                "profile": asdict(self.project.profile),
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                },
            },
            status="started",
        )
        self._emit_progress(
            "opening_project",
            0.0,
            "Opening QAChat project...",
        )
        profile = self.project.profile
        self.backend = choose_backend(profile.backend)
        embedding_model = profile.embedding_model(self.backend)
        chat_model = profile.chat_model(self.backend)
        self.trace.emit(
            "backend.selected",
            component="rag",
            data={
                "requested": profile.backend,
                "resolved": self.backend,
                "embedding_model": embedding_model,
                "chat_model": chat_model,
            },
        )

        embedding_session: EmbeddingSession | None = None
        chat_session: ChatSession | None = None
        try:
            self._emit_progress(
                "loading_embedding_model",
                5.0,
                f"Loading embedding model: {embedding_model}",
            )
            embedding_session = EmbeddingSession(
                embedding_model,
                backend=self.backend,
                allow_download=profile.allow_download,
                trace=self.trace,
            )
            with self.trace.span(
                "indexing",
                component="rag",
                data={"project_directory": str(self.project.root)},
            ):
                chunks, indexes, status, source_changes = (
                    self._prepare_retrieval(embedding_session)
                )
            self._emit_progress(
                "loading_chat_model",
                90.0,
                f"Loading chat model: {chat_model}",
            )
            chat_session = ChatSession(
                chat_model,
                backend=self.backend,
                thinking=profile.thinking,
                temperature=profile.temperature,
                max_new_tokens=profile.max_new_tokens,
                system_prompt=_SYSTEM_PROMPT,
                allow_download=profile.allow_download,
                trace=self.trace,
            )
            self._emit_progress(
                "loading_conversation",
                97.0,
                "Loading saved conversation...",
            )
            messages, turns, checkpoint_id = self._load_conversation()
            self._checkpoint_id = checkpoint_id
            self.trace.emit(
                "conversation.restored",
                component="rag",
                data={
                    "message_count": len(messages),
                    "turn_count": len(turns),
                    "path": str(self.conversation_path),
                    "checkpoint_id": checkpoint_id,
                },
            )
            if messages:
                chat_session.restore_history(messages)
            if turns:
                previous = turns[-1].parameters
                chat_session.configure_generation(
                    {
                        key: previous[key]
                        for key in (
                            "temperature",
                            "thinking",
                            "max_new_tokens",
                        )
                        if key in previous
                    }
                )
        except Exception as error:
            self.trace.exception(
                "session.open.failed",
                error,
                component="rag",
                duration_ms=(perf_counter() - session_started_at) * 1000,
            )
            if chat_session is not None:
                chat_session.close()
            if embedding_session is not None:
                embedding_session.close()
            if self._owns_trace:
                self.trace.close()
            raise

        self.embedding_session = embedding_session
        self.chat_session = chat_session
        self.chunks = chunks
        self.indexes = indexes
        self.index_status = status
        self.source_changes = source_changes
        self.turns = turns
        self._closed = False
        self.trace.emit(
            "session.ready",
            component="rag",
            data={
                "index_status": self.index_status,
                "chunk_count": len(self.chunks),
                "turn_count": len(self.turns),
            },
            duration_ms=(perf_counter() - session_started_at) * 1000,
        )
        self._emit_progress(
            "ready",
            100.0,
            f"QAChat ready. Index {self.index_status}.",
        )

    def _emit_progress(
        self: RAGSession,
        stage: str,
        percent: float,
        message: str,
        *,
        current: int | None = None,
        total: int | None = None,
        remaining: int | None = None,
        elapsed_seconds: float | None = None,
        eta_seconds: float | None = None,
    ) -> None:
        """Send one structured progress event when a callback is configured.

        Parameters
        ----------
        stage: str
            Machine-readable preparation stage name.
        percent: float
            Approximate overall preparation percentage passed to progress
            consumers.
        message: str
            Human-readable progress description.
        current: int | None, optional
            Completed item count for this stage, when meaningful. Defaults to
            ``None``.
        total: int | None, optional
            Total item count for this stage, when meaningful. Defaults to
            ``None``.
        remaining: int | None, optional
            Items remaining in the current stage, when meaningful. Defaults to
            ``None``.
        elapsed_seconds: float | None, optional
            Elapsed duration of the current counted stage. Defaults to
            ``None``.
        eta_seconds: float | None, optional
            Estimated remaining stage duration from observed average
            throughput, not a completion guarantee. Defaults to ``None``.

        Notes
        -----
        Trace progress is reduced to configured percentage checkpoints while
        callbacks receive the emitted ProgressEvent. This reports estimates,
        not guaranteed completion or elapsed limits.

        """

        event = ProgressEvent(
            stage=stage,
            percent=float(percent),
            message=message,
            current=current,
            total=total,
            remaining=remaining,
            elapsed_seconds=elapsed_seconds,
            eta_seconds=eta_seconds,
        )
        trace = getattr(self, "trace", NullTraceLogger())
        last_trace_percent = self._progress_last_trace_percent.get(stage)
        completed_count = (
            current is not None and total is not None and current == total
        )
        if (
            last_trace_percent is None
            or percent - last_trace_percent >= _PROGRESS_TRACE_PERCENT_STEP
            or completed_count
        ):
            trace.emit(
                "progress.updated",
                component="rag",
                data=asdict(event),
            )
            self._progress_last_trace_percent[stage] = float(percent)
        callback = getattr(self, "progress_callback", None)
        if callback is not None:
            callback(event)

    def _emit_count_progress(
        self: RAGSession,
        stage: str,
        start_percent: float,
        end_percent: float,
        label: str,
        current: int,
        total: int,
        started_at: float,
    ) -> None:
        """Emit throttled counted progress with elapsed time and ETA.

        Parameters
        ----------
        stage: str
            Machine-readable preparation stage name.
        start_percent: float
            Overall progress value at the start of this counted stage.
        end_percent: float
            Overall progress value assigned to completion of this counted
            stage.
        label: str
            Human-readable stage description used to compose the progress
            message.
        current: int
            Completed item count for this stage, when meaningful.
        total: int
            Total item count for this stage, when meaningful.
        started_at: float
            perf_counter timestamp captured at the beginning of the stage.

        Raises
        ------
        ValueError
            If counts do not satisfy 0 <= current <= total.

        Notes
        -----
        Updates between endpoints may be throttled by elapsed time. Remaining
        time uses average throughput and is unavailable before a positive
        count; callback failures are not swallowed.

        """

        if total < 0 or current < 0 or current > total:
            raise ValueError(
                "Progress counts must satisfy 0 <= current <= total."
            )

        now = perf_counter()
        last = self._progress_last_emit.get(stage, 0.0)
        if (
            current not in {0, total}
            and now - last < _PROGRESS_MIN_INTERVAL_SECONDS
        ):
            return
        self._progress_last_emit[stage] = now

        fraction = 1.0 if total == 0 else current / total
        percent = start_percent + (end_percent - start_percent) * fraction
        remaining = max(0, total - current)
        elapsed = max(0.0, now - started_at)
        eta: float | None = None
        if current > 0 and elapsed > 0.0:
            eta = 0.0 if remaining == 0 else elapsed * remaining / current

        self._emit_progress(
            stage,
            percent,
            label,
            current=current,
            total=total,
            remaining=remaining,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
        )

    def use_chat_model(self: RAGSession, model_id: str) -> None:
        """Switch the generation model while preserving conversation state.

        The embedding model and retrieval indexes are left unchanged. The old
        model is released before the replacement is loaded to avoid holding
        two large language models in memory at once.
        """

        if self._closed:
            raise RuntimeError("This RAG session is closed.")
        if self.backend != "mlx":
            raise RuntimeError("GUI effort presets currently require MLX.")
        if self.chat_session.model_id == model_id:
            return
        old = self.chat_session
        old_model_id = old.model_id
        messages = [dict(message) for message in old.messages]
        settings = {
            "thinking": old.thinking,
            "temperature": old.temperature,
            "max_new_tokens": old.max_new_tokens,
        }
        old.close()
        try:
            replacement = ChatSession(
                model_id,
                backend=self.backend,
                system_prompt=_SYSTEM_PROMPT,
                allow_download=self.project.profile.allow_download,
                trace=self.trace,
                **settings,
            )
        except Exception:
            self.chat_session = ChatSession(
                old_model_id,
                backend=self.backend,
                system_prompt=_SYSTEM_PROMPT,
                allow_download=self.project.profile.allow_download,
                trace=self.trace,
                **settings,
            )
            self.chat_session.restore_history(messages)
            raise
        replacement.restore_history(messages)
        self.chat_session = replacement
        self.trace.emit(
            "chat.model.switched",
            component="rag",
            data={"from": old_model_id, "to": model_id},
        )

    def ask(
        self: RAGSession,
        question: str,
        *,
        retrieval_count: int | None = None,
        generation_settings: dict[str, object] | None = None,
        phase_callback: AnswerPhaseCallback | None = None,
    ) -> str:
        """Retrieve evidence, generate an answer, and save the turn.

        The initial retrieval count applies to each ranking and fusion stage;
        explicit file/function context may expand the final evidence.
        Generation settings supplied here persist on the loaded chat session
        after validation.

        Parameters
        ----------
        question: str
            User question about the opened project.
        retrieval_count: int | None, optional
            Optional evidence count for this question only. Defaults to
            ``None``.
        generation_settings: dict[str, object] | None, optional
            Optional temperature, thinking, and answer length for this and
            later questions, until changed again. Defaults to ``None``.
        phase_callback: AnswerPhaseCallback | None, optional
            Optional callback receiving (phase_name, details_dict) for answer
            phases. Its exceptions propagate; it is not an asynchronous
            notification queue. Defaults to ``None``.

        Returns
        -------
        str
            Grounded chat-model answer.

        Raises
        ------
        RuntimeError
            If the session is closed.
        ValueError
            If the question is empty.

        Notes
        -----
        The method appends chat history and a ConversationTurn before
        checkpoint writing finishes. A persistence or late phase-callback
        failure does not roll back that state. Evidence is supplied in the
        prompt but generated claims/citations are not checked. Full chat
        history and all expanded evidence are used without an aggregate
        token-budget guard.

        """

        if self._closed:
            raise RuntimeError("This RAG session is closed.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Enter a non-empty question.")
        if retrieval_count is not None and (
            type(retrieval_count) is not int or retrieval_count < 1
        ):
            raise ValueError(
                "retrieval_count must be a positive whole number."
            )
        if phase_callback is not None and not callable(phase_callback):
            raise ValueError("phase_callback must be callable or None.")

        def report_phase(phase: str, **details: object) -> None:
            """Notify an interface without coupling retrieval to GUI code.

            Parameters
            ----------
            phase: str
                Machine-readable answer phase reported to the caller.
            **details: object
                Additional phase-specific event fields passed to the callback.

            Notes
            -----
            The nested helper calls the closed-over callback only when
            supplied. It does not catch callback exceptions.

            """

            if phase_callback is not None:
                phase_callback(phase, details)

        question = question.strip()
        trace = getattr(self, "trace", NullTraceLogger())
        turn_id = trace.new_id(f"turn-{len(self.turns) + 1:04d}")
        history_roles = [
            str(message.get("role", ""))
            for message in self.chat_session.messages
        ]
        conversation_state = {
            "is_new_chat": len(self.turns) == 0,
            "previous_turn_count": len(self.turns),
            "history_message_count": len(self.chat_session.messages),
            "history_roles": history_roles,
            "prior_conversation_included": any(
                role in {"user", "assistant"} for role in history_roles
            ),
        }
        question_data: dict[str, object] = {
            "question_sha256": sha256(question.encode("utf-8")).hexdigest(),
            "question_characters": len(question),
            "retrieval_count_override": retrieval_count,
            "generation_settings_override": generation_settings,
            "conversation_state": conversation_state,
        }
        if trace.include_content:
            question_data["question"] = question
        with trace.span(
            "turn",
            component="rag",
            data=question_data,
            turn_id=turn_id,
        ):
            if generation_settings is not None:
                before = {
                    "temperature": self.chat_session.temperature,
                    "thinking": self.chat_session.thinking,
                    "max_new_tokens": self.chat_session.max_new_tokens,
                }
                self.chat_session.configure_generation(
                    generation_settings,
                    trace_turn_id=turn_id,
                )
                trace.emit(
                    "turn.generation_settings.resolved",
                    component="rag",
                    data={
                        "before": before,
                        "effective": {
                            "temperature": self.chat_session.temperature,
                            "thinking": self.chat_session.thinking,
                            "max_new_tokens": (
                                self.chat_session.max_new_tokens
                            ),
                        },
                    },
                    turn_id=turn_id,
                )

            effective = self._parameters(retrieval_count)
            previous_retrieval_count = (
                self.turns[-1].parameters.get(
                    "retrieval_count",
                    self.project.profile.retrieval_count,
                )
                if self.turns
                else self.project.profile.retrieval_count
            )
            trace.settings_changed(
                "retrieval.settings.changed",
                component="rag",
                before={"retrieval_count": previous_retrieval_count},
                after={"retrieval_count": effective["retrieval_count"]},
                data={"scope": "conversation_turn"},
                turn_id=turn_id,
            )
            trace.emit(
                "turn.settings.snapshot",
                component="rag",
                data=effective,
                turn_id=turn_id,
            )
            report_phase("retrieving", message="Retrieving evidence")
            evidence = self._retrieve(
                question,
                retrieval_count,
                trace_turn_id=turn_id,
                status_callback=lambda message: report_phase(
                    "retrieving", message=message
                ),
            )
            report_phase(
                "evidence_ready",
                message=f"{len(evidence)} references found",
                evidence_count=len(evidence),
            )
            prompt = _answer_prompt(question, evidence)
            report_phase(
                "generating",
                message="Generating answer",
                evidence_count=len(evidence),
            )
            answer = self.chat_session.reply(
                prompt,
                trace_turn_id=turn_id,
            )
            turn = ConversationTurn(
                timestamp=datetime.now(timezone.utc).isoformat(),
                question=question,
                answer=answer,
                evidence=evidence,
                parameters=effective,
            )
            self.turns.append(turn)
            report_phase("saving", message="Saving conversation")
            self._save_conversation(trace_turn_id=turn_id)
            report_phase(
                "completed",
                message="Answer ready",
                evidence_count=len(evidence),
            )
            return answer

    def switch_conversation(
        self,
        conversation_path: str | Path,
        *,
        trace: TraceSink,
    ) -> list[ConversationTurn]:
        """Load a conversation without reloading models or indexes.

        Parameters
        ----------
        conversation_path: str | pathlib.Path
            Conversation JSON path beneath the project logs tree, using a
            conversation_*.json filename. Relative paths start at the project
            root.
        trace: TraceSink
            Caller-owned trace sink associated with the target conversation.
            Its output path should agree with the conversation path when
            persistence is shared.

        Returns
        -------
        list[ConversationTurn]
            New list of the loaded ConversationTurn records.

        Raises
        ------
        RuntimeError
            If the session has already been closed.
        ValueError
            If the path, saved state, restored history, or saved generation
            settings are invalid.
        OSError
            If saved state or trace writes cannot be accessed.

        Notes
        -----
        Loaded models and indexes are reused. The trace sink is replaced and
        made caller-owned before later state loading/validation; failures are
        not transactional. Generation settings come from the last saved turn
        when available, otherwise the profile defaults.

        """

        if self._closed:
            raise RuntimeError("This RAG session is closed.")
        if self._owns_trace:
            self.trace.close()
        self.trace = trace
        self._owns_trace = False
        self.chat_session.trace = trace
        self.embedding_session.trace = trace
        self.conversation_path = self._resolve_conversation_path(
            conversation_path
        )
        messages, turns, checkpoint_id = self._load_conversation()
        self.chat_session.reset()
        if messages:
            self.chat_session.restore_history(messages)
        defaults = {
            "temperature": self.project.profile.temperature,
            "thinking": self.project.profile.thinking,
            "max_new_tokens": self.project.profile.max_new_tokens,
        }
        if turns:
            defaults.update(
                {
                    key: turns[-1].parameters[key]
                    for key in defaults
                    if key in turns[-1].parameters
                }
            )
        self.chat_session.configure_generation(defaults)
        self.turns = turns
        self._checkpoint_id = checkpoint_id
        self.trace.emit(
            "conversation.restored",
            component="rag",
            data={
                "message_count": len(messages),
                "turn_count": len(turns),
                "path": str(self.conversation_path),
                "checkpoint_id": checkpoint_id,
                "models_reused": True,
                "indexes_reused": True,
            },
        )
        return list(self.turns)

    def reset(self: RAGSession) -> Path | None:
        """Archive the active conversation and clear it from memory.

        Returns
        -------
        pathlib.Path | None
            Renamed conversation log, or None when no active log exists.

        Raises
        ------
        RuntimeError
            If the session is closed.
        OSError
            If the existing conversation cannot be archived.

        Notes
        -----
        Archives the selected conversation file when present, clears in-memory
        turns/checkpoint identity, and resets the chat history to its system
        instruction. It does not reload models or clear an enabled trace sink's
        retained resume-state dictionary.

        """

        if self._closed:
            raise RuntimeError("This RAG session is closed.")
        archived_checkpoint_id = getattr(self, "_checkpoint_id", None)
        archived_path = self._archive_conversation()
        self.chat_session.reset()
        self.turns = []
        self._checkpoint_id = None
        getattr(self, "trace", NullTraceLogger()).emit(
            "conversation.reset",
            component="rag",
            data={
                "archived_path": (
                    str(archived_path) if archived_path is not None else None
                ),
                "archived_checkpoint_id": archived_checkpoint_id,
            },
        )
        return archived_path

    def close(self: RAGSession) -> None:
        """Release both models while keeping indexes and logs on disk.

        Notes
        -----
        The closed flag is set in a finally block and an owned trace is closed
        there. Caller-owned sinks are not closed. Backend cleanup errors
        propagate; this is not a guarantee that every allocation is released if
        earlier cleanup fails.

        """

        if self._closed:
            return
        trace = getattr(self, "trace", NullTraceLogger())
        try:
            self.chat_session.close()
            self.embedding_session.close()
            trace.emit("session.closed", component="rag")
        finally:
            self._closed = True
            if getattr(self, "_owns_trace", False):
                trace.close()

    def _prepare_source(
        self: RAGSession,
        embedding_session: EmbeddingSession,
        report: dict[str, object] | None = None,
    ) -> tuple[list[Chunk], CodeGraph]:
        """Read configured source and produce chunks and its code graph.

        Parameters
        ----------
        embedding_session: EmbeddingSession
            Loaded embedding session used for tokenizer-aware chunking and/or
            index/query embedding.
        report: dict[str, object] | None, optional
            Optional existing scan report. None causes a scan using the project
            profile. Defaults to ``None``.

        Returns
        -------
        tuple[list[Chunk], CodeGraph]
            Pair (chunks, code_graph) derived from the selected source files.

        Notes
        -----
        Reads selected content, extracts units, counts tokens through the
        embedding tokenizer, and builds the heuristic code graph. A supplied
        incomplete scan is not rejected here.

        """

        profile = self.project.profile
        if report is None:
            report = scan_repository(
                self.project.source,
                extensions=profile.extensions,
                max_file_bytes=profile.max_file_bytes,
            )
        selected_count = len(report.get("selected_files", []))

        reading_started = perf_counter()
        self._emit_count_progress(
            "reading_source",
            35.0,
            42.0,
            "Reading source files",
            0,
            selected_count,
            reading_started,
        )
        with self.trace.span(
            "source.read",
            component="rag",
            data={"selected_file_count": selected_count},
        ):
            source_files = read_repository_files(
                report,
                progress_callback=(
                    lambda current, total: self._emit_count_progress(
                        "reading_source",
                        35.0,
                        42.0,
                        "Reading source files",
                        current,
                        total,
                        reading_started,
                    )
                ),
            )
        self.trace.emit(
            "source.read.summary",
            component="rag",
            data={
                "file_count": len(source_files),
                "files": [
                    {
                        "path": file.path,
                        "characters": len(file.text),
                        "sha256": sha256(
                            file.text.encode("utf-8")
                        ).hexdigest(),
                    }
                    for file in source_files
                ],
            },
        )

        extracting_started = perf_counter()
        source_file_count = len(source_files)
        self._emit_count_progress(
            "extracting_source",
            43.0,
            49.0,
            "Extracting source structure",
            0,
            source_file_count,
            extracting_started,
        )
        with self.trace.span(
            "source.extract",
            component="rag",
            data={"file_count": source_file_count},
        ):
            units = extract_source_units(
                source_files,
                progress_callback=(
                    lambda current, total: self._emit_count_progress(
                        "extracting_source",
                        43.0,
                        49.0,
                        "Extracting source structure",
                        current,
                        total,
                        extracting_started,
                    )
                ),
            )

        chunking_started = perf_counter()
        unit_count = len(units)
        self._emit_count_progress(
            "chunking_source",
            50.0,
            55.0,
            "Chunking source units",
            0,
            unit_count,
            chunking_started,
        )
        chunk_settings = {
            "target_tokens": profile.target_tokens,
            "max_tokens": profile.max_tokens,
            "overlap_tokens": profile.overlap_tokens,
        }
        with self.trace.span(
            "source.chunk",
            component="rag",
            data={"unit_count": unit_count, **chunk_settings},
        ):
            chunks = chunk_source_units(
                units,
                count_tokens=embedding_session.count_tokens,
                target_tokens=profile.target_tokens,
                max_tokens=profile.max_tokens,
                overlap_tokens=profile.overlap_tokens,
                progress_callback=(
                    lambda current, total: self._emit_count_progress(
                        "chunking_source",
                        50.0,
                        55.0,
                        "Chunking source units",
                        current,
                        total,
                        chunking_started,
                    )
                ),
            )
        self.trace.emit(
            "source.chunk.summary",
            component="rag",
            data={
                "chunk_count": len(chunks),
                "settings": chunk_settings,
            },
        )
        self._emit_progress(
            "building_code_graph",
            56.0,
            f"Building code graph from {len(chunks)} chunks...",
            current=len(chunks),
            total=len(chunks),
        )
        with self.trace.span(
            "index.code_graph.build",
            component="rag",
            data={"chunk_count": len(chunks)},
        ):
            code_graph = build_code_graph(source_files, units)
        return chunks, code_graph

    def _prepare_retrieval(
        self: RAGSession,
        embedding_session: EmbeddingSession,
    ) -> tuple[
        list[Chunk],
        PersistentIndexes,
        str,
        dict[str, object],
    ]:
        """Reuse indexes when source hashes and index settings still match.

        Parameters
        ----------
        embedding_session: EmbeddingSession
            Loaded embedding session used for tokenizer-aware chunking and/or
            index/query embedding.

        Returns
        -------
        tuple[list[Chunk], PersistentIndexes, str, dict[str, object]]
            Tuple (chunks, indexes, index_status, source_changes), where status
            is reused or rebuilt.

        Notes
        -----
        A selected-source manifest and stored configuration metadata control
        reuse. Existing vectors are loaded only if the stored bundle is
        accepted. Source completeness is not checked here; index and metadata
        files are written separately rather than as one transaction.

        """

        profile = self.project.profile
        self._emit_progress(
            "scanning_source",
            15.0,
            "Scanning selected project files...",
        )
        report = scan_repository(
            self.project.source,
            extensions=profile.extensions,
            max_file_bytes=profile.max_file_bytes,
        )
        report_data: dict[str, object] = {
            "repository": report.get("repository"),
            "complete": report.get("complete"),
            "selected_file_count": len(report.get("selected_files", [])),
            "selected_files": report.get("selected_files", []),
            "entries": report.get("entries", []),
        }
        self.trace.emit(
            "source.scan.completed",
            component="rag",
            data=report_data,
        )
        selected_count = len(report.get("selected_files", []))
        self._emit_progress(
            "checking_index",
            22.0,
            f"Checking {selected_count} files against the saved index...",
            current=selected_count,
            total=selected_count,
        )
        current_manifest = build_repository_manifest(report)
        index_path = self.project.indexes / "retrieval.json"
        metadata_path = self.project.indexes / "settings.json"
        manifest_path = self.project.indexes / "source_manifest.json"
        expected_metadata = self._index_metadata()

        saved_manifest: dict[str, object] | None = None
        saved_metadata: dict[str, object] | None = None
        if manifest_path.is_file():
            try:
                value = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    saved_manifest = value
            except (OSError, json.JSONDecodeError):
                saved_manifest = None

        try:
            changes = compare_repository_manifests(
                saved_manifest,
                current_manifest,
            )
        except ValueError:
            saved_manifest = None
            changes = compare_repository_manifests(None, current_manifest)

        if metadata_path.is_file():
            try:
                value = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    saved_metadata = value
            except (OSError, json.JSONDecodeError):
                saved_metadata = None

        configuration_changed = saved_metadata != expected_metadata
        changes["configuration_changed"] = configuration_changed
        self.trace.emit(
            "index.check.completed",
            component="rag",
            data={
                "index_path": str(index_path),
                "metadata_path": str(metadata_path),
                "manifest_path": str(manifest_path),
                "expected_metadata": expected_metadata,
                "saved_metadata": saved_metadata,
                "changes": changes,
                "reusable": (
                    index_path.is_file()
                    and saved_metadata is not None
                    and saved_manifest is not None
                    and not configuration_changed
                    and bool(changes["matches"])
                ),
            },
        )
        self._emit_progress(
            "validating_index",
            28.0,
            "Validating saved index metadata and source manifest...",
        )

        if (
            index_path.is_file()
            and saved_metadata is not None
            and saved_manifest is not None
            and not configuration_changed
            and bool(changes["matches"])
        ):
            try:
                self._emit_progress(
                    "loading_index",
                    32.0,
                    "Source unchanged; loading the saved retrieval index...",
                )
                with self.trace.span(
                    "index.load",
                    component="rag",
                    data={"index_path": str(index_path)},
                ):
                    indexes = load_indexes(index_path)
                self.trace.emit(
                    "index.reused",
                    component="rag",
                    data={
                        "index_path": str(index_path),
                        "chunk_count": len(indexes.semantic.chunks),
                        "fingerprint": indexes.fingerprint,
                    },
                )
                self._emit_progress(
                    "index_ready",
                    85.0,
                    "Saved retrieval index loaded.",
                )
                return (
                    list(indexes.semantic.chunks),
                    indexes,
                    "reused",
                    changes,
                )
            except (
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                self.trace.exception(
                    "index.load.failed",
                    error,
                    component="rag",
                    data={"index_path": str(index_path)},
                )

        chunks, code_graph = self._prepare_source(
            embedding_session,
            report=report,
        )
        embedding_started = perf_counter()
        chunk_count = len(chunks)
        self._emit_count_progress(
            "building_semantic_index",
            58.0,
            74.0,
            "Embedding chunks",
            0,
            chunk_count,
            embedding_started,
        )
        with self.trace.span(
            "index.semantic.build",
            component="rag",
            data={
                "chunk_count": chunk_count,
                "batch_size": _EMBEDDING_PROGRESS_BATCH_SIZE,
            },
        ):
            semantic = build_semantic_index(
                chunks,
                embedding_session,
                batch_size=_EMBEDDING_PROGRESS_BATCH_SIZE,
                progress_callback=(
                    lambda current, total: self._emit_count_progress(
                        "building_semantic_index",
                        58.0,
                        74.0,
                        "Embedding chunks",
                        current,
                        total,
                        embedding_started,
                    )
                ),
            )
        self._emit_progress(
            "building_lexical_index",
            75.0,
            (
                "Semantic index complete; building lexical index for "
                f"{len(chunks)} chunks..."
            ),
            current=len(chunks),
            total=len(chunks),
        )
        with self.trace.span(
            "index.lexical.build",
            component="rag",
            data={"chunk_count": len(chunks)},
        ):
            lexical = build_lexical_index(chunks)
        indexes = prepare_persistent_indexes(
            semantic,
            lexical,
            code_graph,
        )
        self._emit_progress(
            "saving_index",
            82.0,
            "Saving retrieval index and manifests...",
        )
        with self.trace.span(
            "index.save",
            component="rag",
            data={
                "index_path": str(index_path),
                "metadata_path": str(metadata_path),
                "manifest_path": str(manifest_path),
            },
        ):
            save_indexes(indexes, index_path)
            _write_json(metadata_path, expected_metadata)
            _write_json(manifest_path, current_manifest)
        self.trace.emit(
            "index.rebuilt",
            component="rag",
            data={
                "index_path": str(index_path),
                "metadata_path": str(metadata_path),
                "manifest_path": str(manifest_path),
                "fingerprint": indexes.fingerprint,
                "chunk_count": len(chunks),
                "changes": changes,
            },
        )
        self._emit_progress(
            "index_ready",
            85.0,
            "Retrieval index saved.",
        )
        return chunks, indexes, "rebuilt", changes

    def _retrieve(
        self: RAGSession,
        question: str,
        retrieval_count: int | None = None,
        *,
        trace_turn_id: str | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> tuple[Evidence, ...]:
        """Run hybrid retrieval and deterministic context expansion.

        Parameters
        ----------
        question: str
            User question supplied to query formatting or evidence retrieval.
        retrieval_count: int | None, optional
            Optional positive initial retrieval-result count; None uses the
            project profile. Explicit context expansion may produce more
            evidence. Defaults to ``None``.
        trace_turn_id: str | None, optional
            Optional identifier associating trace events with one conversation
            turn. Defaults to ``None``.

        Returns
        -------
        tuple[Evidence, ...]
            Evidence tuple numbered from one after rank fusion and context
            expansion.

        Notes
        -----
        Semantic search receives the configured query format, while lexical and
        code searches use the raw question. Explicit context expansion is not
        capped by retrieval_count. Trace details use previews; the evidence
        records keep full chunk text.

        """

        def report(message: str) -> None:
            if status_callback is not None:
                status_callback(message)

        count = retrieval_count or self.project.profile.retrieval_count
        report("Formatting the search query")
        query_text = self._format_query(question)
        trace = getattr(self, "trace", NullTraceLogger())
        query_data: dict[str, object] = {
            "query_format": self.project.profile.query_format,
            "retrieval_count": count,
            "query_sha256": sha256(query_text.encode("utf-8")).hexdigest(),
        }
        if trace.include_content:
            query_data["question"] = question
            query_data["formatted_query"] = query_text
        trace.emit(
            "retrieval.query.formatted",
            component="retrieval",
            data=query_data,
            turn_id=trace_turn_id,
        )
        with trace.span(
            "retrieval.semantic",
            component="retrieval",
            turn_id=trace_turn_id,
        ):
            report("Embedding the query and running semantic search")
            semantic = semantic_search(
                self.indexes.semantic,
                query_text,
                self.embedding_session,
                count=count,
                trace_turn_id=trace_turn_id,
            )
        trace.emit(
            "retrieval.semantic.ranked",
            component="retrieval",
            data=_ranked_trace_data(
                semantic,
                include_text=trace.include_content,
            ),
            turn_id=trace_turn_id,
        )
        report(f"Semantic search ranked {len(semantic)} results")
        with trace.span(
            "retrieval.lexical",
            component="retrieval",
            turn_id=trace_turn_id,
        ):
            report("Running lexical search")
            lexical = lexical_search(
                self.indexes.lexical,
                question,
                count=count,
            )
        trace.emit(
            "retrieval.lexical.ranked",
            component="retrieval",
            data=_ranked_trace_data(
                lexical,
                include_text=trace.include_content,
            ),
            turn_id=trace_turn_id,
        )
        report(f"Lexical search ranked {len(lexical)} results")
        with trace.span(
            "retrieval.code",
            component="retrieval",
            turn_id=trace_turn_id,
        ):
            report(
                "Searching the code graph"
                if self.project.profile.kind == "code"
                else "Checking structured relationships"
            )
            code = code_search(
                self.indexes.code_graph,
                self.chunks,
                question,
                count=count,
            )
        trace.emit(
            "retrieval.code.ranked",
            component="retrieval",
            data=_ranked_trace_data(
                code,
                include_text=trace.include_content,
            ),
            turn_id=trace_turn_id,
        )
        report(
            f"Code graph search ranked {len(code)} results"
            if self.project.profile.kind == "code"
            else f"Structured relationship search found {len(code)} results"
        )
        report("Combining and ranking search results")
        combined = hybrid_search(
            semantic=semantic,
            lexical=lexical,
            code=code,
            count=count,
        )
        trace.emit(
            "retrieval.hybrid.ranked",
            component="retrieval",
            data=_ranked_trace_data(
                combined,
                include_text=trace.include_content,
            ),
            turn_id=trace_turn_id,
        )
        report(f"Hybrid ranking kept {len(combined)} results")
        plan = plan_retrieval(question, self.indexes.code_graph)
        plan_data = asdict(plan)
        expansion_requested = any(
            bool(plan_data.get(name))
            for name in (
                "paths",
                "functions",
                "relationship",
                "related_functions",
            )
        )
        trace.emit(
            "retrieval.context_expansion.planned",
            component="retrieval",
            data={
                "input": {
                    "question_sha256": sha256(
                        question.encode("utf-8")
                    ).hexdigest(),
                    "hybrid_result_count": len(combined),
                },
                "decision": (
                    "expand_related_context"
                    if expansion_requested
                    else "keep_ranked_context"
                ),
                "reason": {"signals": plan_data.get("signals", [])},
                "output": plan_data,
            },
            turn_id=trace_turn_id,
        )
        report("Expanding related context")
        expanded = expand_retrieval_context(
            combined,
            self.chunks,
            self.indexes.code_graph,
            plan,
        )
        expansion_data: dict[str, object] = {
            "rrf_count": len(combined),
            "expanded_count": len(expanded),
            "chunks": [
                _chunk_trace_data(
                    chunk,
                    include_text=trace.include_content,
                )
                for chunk in expanded
            ],
        }
        if trace.include_content:
            expansion_data["chunks_artifact"] = trace.artifact(
                f"{trace_turn_id or 'turn'}-evidence",
                expansion_data.pop("chunks"),
            )
        trace.emit(
            "retrieval.context.expanded",
            component="retrieval",
            data=expansion_data,
            turn_id=trace_turn_id,
        )
        evidence = tuple(
            Evidence(
                citation=index,
                path=chunk.path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                symbol=chunk.symbol,
                text=chunk.text,
                cell=chunk.cell,
            )
            for index, chunk in enumerate(expanded, start=1)
        )
        trace.emit(
            "evidence.finalized",
            component="retrieval",
            data={
                "count": len(evidence),
                "citations": [
                    {
                        "citation": item.citation,
                        "path": item.path,
                        "start_line": item.start_line,
                        "end_line": item.end_line,
                        "symbol": item.symbol,
                        "text_sha256": sha256(
                            item.text.encode("utf-8")
                        ).hexdigest(),
                    }
                    for item in evidence
                ],
            },
            turn_id=trace_turn_id,
        )
        return evidence

    def _format_query(self: RAGSession, question: str) -> str:
        """Apply the embedding model's configured query format.

        Parameters
        ----------
        question: str
            User question supplied to query formatting or evidence retrieval.

        Returns
        -------
        str
            Unmodified question for plain format, or instruction-prefixed
            software/literature query for qwen3.

        Notes
        -----
        Formatting is for the embedding model, not the chat template.

        """

        profile = self.project.profile
        if profile.query_format == "plain":
            return question
        instruction = (
            SOFTWARE_REPOSITORY_INSTRUCTION
            if profile.kind == "code"
            else LITERATURE_REPOSITORY_INSTRUCTION
        )
        return format_qwen3_query(question, instruction)

    def _index_metadata(self: RAGSession) -> dict[str, object]:
        """Return settings that determine reusable index compatibility.

        Returns
        -------
        dict[str, object]
            Settings that determine reusable index compatibility.

        Notes
        -----
        These fields identify the configured embedding/model/chunking choices,
        not the resolved model commit or implementation/library versions. The
        caller separately compares a selected-source manifest.

        """

        profile = self.project.profile
        return {
            "format_version": _INDEX_METADATA_VERSION,
            "project_kind": profile.kind,
            "backend": self.backend,
            "embedding_model": profile.embedding_model(self.backend),
            "query_format": profile.query_format,
            "extensions": list(profile.extensions),
            "max_file_bytes": profile.max_file_bytes,
            "target_tokens": profile.target_tokens,
            "max_tokens": profile.max_tokens,
            "overlap_tokens": profile.overlap_tokens,
        }

    def _parameters(
        self: RAGSession, retrieval_count: int | None = None
    ) -> dict[str, object]:
        """Return the complete settings snapshot recorded for one turn.

        Parameters
        ----------
        retrieval_count: int | None, optional
            Optional positive initial retrieval-result count; None uses the
            project profile. Explicit context expansion may produce more
            evidence. Defaults to ``None``.

        Returns
        -------
        dict[str, object]
            The complete settings snapshot recorded for one turn.

        Notes
        -----
        Returns configured project/model metadata and live chat generation
        settings. A retrieval-count argument records an answer-specific
        override; callers omitting it record the profile count.

        """

        profile = self.project.profile
        return {
            **self._index_metadata(),
            "project_kind": profile.kind,
            "source": str(self.project.source),
            "backend_preference": profile.backend,
            "allow_download": profile.allow_download,
            "chat_model": profile.chat_model(self.backend),
            "retrieval_count": retrieval_count or profile.retrieval_count,
            "thinking": self.chat_session.thinking,
            "temperature": self.chat_session.temperature,
            "max_new_tokens": self.chat_session.max_new_tokens,
            "system_prompt": _SYSTEM_PROMPT,
        }

    def _load_conversation(
        self: RAGSession,
    ) -> tuple[list[dict[str, str]], list[ConversationTurn], str | None]:
        """Load saved chat messages and human-readable turns.

        Returns
        -------
        tuple[list[dict[str, str]], list[ConversationTurn], str | None]
            Tuple (chat_messages, turns, checkpoint_id), with empty
            history/turns and None when the file is absent.

        Raises
        ------
        ValueError
            If saved JSON or recognized conversation state is invalid.
        OSError
            If saved state cannot be read.

        Notes
        -----
        Recognizes legacy and grouped trace conversation records. It does not
        validate that saved messages, turns, evidence, profile, and indexes are
        mutually consistent.

        """

        path = self.conversation_path
        if not path.is_file():
            return [], [], None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            conversation = _conversation_state(payload)
            turns = [
                _turn_from_data(item) for item in conversation.get("turns", [])
            ]
            legacy_messages = conversation.get("chat_messages")
            if isinstance(legacy_messages, list):
                messages = [
                    {
                        "role": str(message["role"]),
                        "content": str(message["content"]),
                    }
                    for message in legacy_messages
                ]
            else:
                system_prompt = str(
                    conversation.get("system_prompt", _SYSTEM_PROMPT)
                )
                messages = [{"role": "system", "content": system_prompt}]
                for turn in turns:
                    messages.extend(
                        (
                            {
                                "role": "user",
                                "content": _answer_prompt(
                                    turn.question,
                                    turn.evidence,
                                ),
                            },
                            {"role": "assistant", "content": turn.answer},
                        )
                    )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("The saved conversation is invalid.") from error
        checkpoint_id = conversation.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            checkpoint_id = None
        return messages, turns, checkpoint_id

    def _save_conversation(
        self,
        *,
        trace_turn_id: str | None = None,
    ) -> None:
        """Save exact chat state and inspectable interaction records.

        Parameters
        ----------
        trace_turn_id: str | None, optional
            Optional identifier associating trace events with one conversation
            turn. Defaults to ``None``.

        Notes
        -----
        Builds a hash-identified resumable snapshot. With an enabled trace
        sink, state is written through update_conversation; otherwise it is
        written to conversation_path. The hash labels content but this
        implementation does not store a separate immutable historical snapshot
        for every checkpoint.

        """

        conversation = {
            "system_prompt": _SYSTEM_PROMPT,
            "turns": [_turn_to_data(turn) for turn in self.turns],
        }
        parent_checkpoint_id = getattr(self, "_checkpoint_id", None)
        checkpoint_snapshot = {
            "project": {
                "root": str(self.project.root),
                "source": str(self.project.source),
                "index_fingerprint": self.indexes.fingerprint,
                "index_status": self.index_status,
            },
            "effective_settings": self._parameters(),
            "models": {
                "backend": self.backend,
                "chat_model_id": self.chat_session.model_id,
                "chat_model_source": getattr(
                    self.chat_session,
                    "model_source",
                    None,
                ),
                "embedding_model_id": self.embedding_session.model_id,
                "embedding_model_source": getattr(
                    self.embedding_session,
                    "model_source",
                    None,
                ),
            },
            "conversation_sha256": sha256(
                json.dumps(
                    conversation,
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
            "turn_count": len(self.turns),
        }
        checkpoint_id = _resume_checkpoint_id(checkpoint_snapshot)
        checkpoint_state = {
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "conversation_sha256": checkpoint_snapshot["conversation_sha256"],
            "index_fingerprint": self.indexes.fingerprint,
            "turn_count": len(self.turns),
        }

        path = self.conversation_path
        saved_state = {
            **conversation,
            "checkpoint_id": checkpoint_id,
            "checkpoint": checkpoint_state,
        }
        if self.trace.enabled:
            self.trace.update_conversation(saved_state)
        else:
            payload: dict[str, object] = {"sessions": {}}
            if path.is_file():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
            stored = _conversation_state(payload, create=True)
            stored.update(saved_state)
            _write_json(path, payload)
        self.trace.emit(
            "state.checkpoint.committed",
            component="rag",
            data={
                "checkpoint_id": checkpoint_id,
                "parent_checkpoint_id": parent_checkpoint_id,
                "immutable_state_saved": True,
                "conversation_path": str(path),
                "turn_count": len(self.turns),
                "index_fingerprint": self.indexes.fingerprint,
            },
            turn_id=trace_turn_id,
        )
        self._checkpoint_id = checkpoint_id

    def _archive_conversation(self: RAGSession) -> Path | None:
        """Rename the active conversation log with a readable UTC time.

        Returns
        -------
        pathlib.Path | None
            Archive path after renaming an existing conversation, or None if
            there was no file to archive.

        Notes
        -----
        Only the configured conversation_path is moved. It does not relocate an
        independently configured trace output file.

        """

        path = self.conversation_path
        if not path.is_file():
            return None
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H-%M-%S-%fZ"
        )
        archived_path = path.with_name(f"conversation_{timestamp}.json")
        path.replace(archived_path)
        return archived_path

    def _resolve_conversation_path(
        self,
        conversation_path: str | Path | None,
    ) -> Path:
        """Keep saved conversation state inside this project's logs folder.

        Parameters
        ----------
        conversation_path: str | pathlib.Path | None
            Conversation JSON path beneath the project logs tree, using a
            conversation_*.json filename. Relative paths start at the project
            root.

        Returns
        -------
        pathlib.Path
            Resolved JSON path beneath the project logs directory, without
            creating a conversation record.

        Raises
        ------
        ValueError
            If the resolved path leaves project logs, lacks the required
            conversation_*.json naming, or fails the implemented folder-link
            check.

        Notes
        -----
        None allocates a timestamped name beneath logs/traces. Path resolution
        is not a multi-process filename reservation.

        """

        if conversation_path is None:
            directory = self.project.logs / "traces"
            timestamp = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H-%M-%S"
            )
            candidate = directory / f"conversation_{timestamp}.json"
            number = 2
            while candidate.exists():
                candidate = directory / (
                    f"conversation_{timestamp}_{number}.json"
                )
                number += 1
        else:
            candidate = Path(conversation_path).expanduser()
            if not candidate.is_absolute():
                candidate = self.project.root / candidate
        candidate = candidate.resolve()
        logs = self.project.logs.resolve()
        if not candidate.is_relative_to(logs):
            raise ValueError(
                "Conversation state must stay inside project logs."
            )
        if (
            candidate.suffix.lower() != ".json"
            or not candidate.name.startswith("conversation_")
        ):
            raise ValueError(
                "Conversation state must use conversation_<timestamp>.json."
            )
        if candidate.parent.is_symlink():
            raise ValueError("Conversation folder must not be a link.")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate


def _conversation_state(
    payload: dict[str, object], *, create: bool = False
) -> dict[str, object]:
    """Locate resumable state in the chronologically later chat session.

    Parameters
    ----------
    payload: dict[str, object]
        Decoded conversation/trace JSON document.
    create: bool, optional
        Whether missing grouped conversation-state containers may be inserted
        into payload. Defaults to ``False``.

    Returns
    -------
    dict[str, object]
        Mutable conversation/resume-state dictionary within payload.

    Raises
    ------
    ValueError
        If no valid supported state exists and creation is disabled, or the
        enclosing structure is invalid.

    Notes
    -----
    Creation modifies the supplied payload in place. Legacy top-level state is
    recognized before grouped sessions.

    """

    legacy = payload.get("conversation")
    if isinstance(legacy, dict):
        return legacy
    sessions = (
        payload.setdefault("sessions", {})
        if create
        else payload.get("sessions")
    )
    if not isinstance(sessions, dict):
        raise ValueError("The saved conversation is invalid.")
    for session in sessions.values():
        if (
            isinstance(session, dict)
            and session.get("category") == "conversations"
        ):
            state = (
                session.setdefault("resume_state", {})
                if create
                else session.get("resume_state", session.get("conversation"))
            )
            if isinstance(state, dict):
                return state
    if create:
        session = {
            "title": "Conversation turns",
            "category": "conversations",
            "turns": {},
            "operations": {},
            "resume_state": {},
        }
        sessions["05_conversations"] = session
        return session["resume_state"]
    raise ValueError("The saved conversation is invalid.")


def _answer_prompt(
    question: str,
    evidence: tuple[Evidence, ...],
) -> str:
    """Combine one question with numbered, traceable evidence.

    Parameters
    ----------
    question: str
        User question supplied to query formatting or evidence retrieval.
    evidence: tuple[Evidence, ...]
        Ordered retrieved chunks with citation numbers and source provenance.

    Returns
    -------
    str
        Chat user-message text containing numbered evidence blocks and the
        question.

    Notes
    -----
    Evidence text is inserted without semantic validation. The current label
    includes path, line range, and optional symbol, but not notebook cell or
    PDF page metadata. This is prompt construction, not instruction isolation
    for untrusted source text.

    """

    sections = [f"Question:\n{question}", "Evidence:"]
    for item in evidence:
        location = f"{item.path}:{item.start_line}-{item.end_line}"
        symbol = f" ({item.symbol})" if item.symbol else ""
        sections.append(f"[{item.citation}] {location}{symbol}\n{item.text}")
    sections.append(
        "Answer using the evidence above. Put [1], [2], and so on immediately "
        "after the claims they support. Do not repeat the source text or "
        "write a bibliography."
    )
    return "\n\n".join(sections)


def _turn_to_data(turn: ConversationTurn) -> dict[str, object]:
    """Convert one conversation turn to JSON-safe data.

    Parameters
    ----------
    turn: ConversationTurn
        One completed conversation turn with question, answer, evidence, and
        parameters.

    Returns
    -------
    dict[str, object]
        Dictionary of turn values and serialized evidence paths suitable for
        JSON persistence.

    """

    return {
        "timestamp": turn.timestamp,
        "question": turn.question,
        "answer": turn.answer,
        "evidence": [
            {
                **asdict(item),
                "path": item.path.as_posix(),
            }
            for item in turn.evidence
        ],
        "parameters": turn.parameters,
    }


def _turn_from_data(data: dict[str, object]) -> ConversationTurn:
    """Restore one conversation turn from JSON data.

    Parameters
    ----------
    data: dict[str, object]
        Dictionary decoded from or prepared for persisted conversation JSON.

    Returns
    -------
    ConversationTurn
        Reconstructed ConversationTurn with evidence records and copied
        parameters.

    Raises
    ------
    ValueError
        If evidence is not a list, parameters is not a dictionary, or numeric
        conversion fails.
    KeyError
        If required turn/evidence fields are absent.
    TypeError
        If a saved value cannot be converted as expected.

    Notes
    -----
    This is structural reconstruction and coercion, not validation of the
    original answer or current source content.

    """

    evidence_data = data["evidence"]
    if not isinstance(evidence_data, list):
        raise ValueError("Saved evidence must be a list.")
    evidence = tuple(
        Evidence(
            citation=int(item["citation"]),
            path=Path(str(item["path"])),
            start_line=int(item["start_line"]),
            end_line=int(item["end_line"]),
            symbol=(
                str(item["symbol"]) if item["symbol"] is not None else None
            ),
            text=str(item["text"]),
            cell=(
                int(item["cell"])
                if item.get("cell") is not None
                else None
            ),
        )
        for item in evidence_data
    )
    parameters = data["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("Saved parameters must be a mapping.")
    return ConversationTurn(
        timestamp=str(data["timestamp"]),
        question=str(data["question"]),
        answer=str(data["answer"]),
        evidence=evidence,
        parameters=parameters,
    )


def _resume_checkpoint_id(snapshot: dict[str, object]) -> str:
    """Return a stable identity for one complete restorable session state.

    Parameters
    ----------
    snapshot: dict[str, object]
        Resumable state dictionary whose canonical JSON serialization is
        hashed.

    Returns
    -------
    str
        A stable identity for one complete restorable session state.

    Notes
    -----
    The identifier hashes sorted JSON of the supplied snapshot. It is a content
    label, not a signature or a guarantee that a historical snapshot remains
    stored.

    """

    encoded = json.dumps(
        snapshot,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"checkpoint_{sha256(encoded).hexdigest()}"


def _write_json(path: Path, data: dict[str, object]) -> None:
    """Write readable JSON atomically.

    Parameters
    ----------
    path: pathlib.Path
        Destination JSON path.
    data: dict[str, object]
        Dictionary decoded from or prepared for persisted conversation JSON.

    Notes
    -----
    Serializes through a temporary file then replaces the destination. No
    inter-process lock or multi-file transaction is provided.

    """

    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _chunk_trace_data(
    chunk: Chunk,
    *,
    include_text: bool,
) -> dict[str, object]:
    """Return chunk identity and an optional compact text preview.

    Parameters
    ----------
    chunk: Chunk
        Source chunk whose provenance and optional trace preview are recorded.
    include_text: bool
        Whether to include a shortened text preview in trace data.

    Returns
    -------
    dict[str, object]
        Chunk identity and an optional compact text preview.

    Notes
    -----
    The include_text flag adds the trace_text_preview result, not the complete
    source chunk.

    """

    data: dict[str, object] = {
        "identifier": chunk.identifier,
        "path": chunk.path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "kind": chunk.kind,
        "symbol": chunk.symbol,
        "section": chunk.section,
        "cell": chunk.cell,
        "text_sha256": sha256(chunk.text.encode("utf-8")).hexdigest(),
        "text_characters": len(chunk.text),
    }
    if include_text:
        data["text_preview"] = trace_text_preview(chunk.text)
    return data


def _ranked_trace_data(
    results: object,
    *,
    include_text: bool = False,
) -> dict[str, object]:
    """Return ranked chunks, adding exact text only in forensic traces.

    Parameters
    ----------
    results: object
        Iterable of ranked result objects with chunk and score attributes, plus
        optional ranking metadata.
    include_text: bool, optional
        Whether to include a shortened text preview in trace data. Defaults to
        ``False``.

    Returns
    -------
    dict[str, object]
        Ranked chunks, adding exact text only in forensic traces.

    Notes
    -----
    Ranks are one-based in the supplied iteration order. Optional channel,
    entity, and distance fields are included only when the result object
    exposes them.

    """

    rows: list[dict[str, object]] = []
    for rank, result in enumerate(results, start=1):
        row: dict[str, object] = {
            "rank": rank,
            "score": getattr(result, "score", None),
            "chunk": _chunk_trace_data(
                result.chunk,
                include_text=include_text,
            ),
        }
        for name in ("entity", "distance", "channels"):
            if hasattr(result, name):
                row[name] = getattr(result, name)
        rows.append(row)
    return {"count": len(rows), "results": rows}
