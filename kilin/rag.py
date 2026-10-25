"""Coordinate retrieval, chat generation, and resumable project conversations.

``RAGSession`` opens a configured project, loads an embedding session, reuses
or builds source chunks and retrieval indexes, then loads a chat session.
For each question it combines semantic, lexical, and code-based rankings,
expands explicitly requested context, and formats evidence into the model's
user prompt. Returned ``Evidence`` records identify the supplied excerpts;
the module checks citation identifiers against the supplied evidence
without attempting to judge whether claims are scientifically supported.

Before retrieval, a router chat model (by default the answering model)
assigns each question one of the project type's configured intents. The
intent's configuration, not the router, then selects the history policy,
the retrieval mode (search, one document, reused evidence, or none), and
optionally a tool that writes the answer itself. An unusable router reply
falls back to the project type's default intent.

Conversation turns retain questions, answers, evidence, and effective turn
parameters. One authoritative conversation JSON path stores resumable state.
When tracing is enabled for a RAG session, its trace output is required to use
that same path; tracing therefore cannot silently redirect conversation
persistence. The session has side effects: initialization may create files,
rebuild indexes, load models, and download missing model files according to the
profile. ``trace=None`` enables full tracing here, unlike
the low-level chat and embedding classes.

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

    Restore or switch saved conversations around one authoritative JSON path
    using the shared internal conversation-format contract and a single-writer
    file lease.

- **Trace alignment**

    Bind RAG tracing to the active conversation record so enabling tracing does
    not change where resumable conversation state is stored.

- **Answer-phase vocabulary**

    Expose canonical phase identifiers for interfaces observing answer
    retrieval, evidence preparation, generation, saving, and completion.

- **Progress and tracing**

    Emit stage callbacks and complete diagnostic events when tracing is
    enabled, with embedding vectors independently opt-in.

- **Intent routing and tools**

    Route each question to a configured intent that chooses history,
    retrieval mode, and an optional tool that writes the answer.

- **Incremental indexing**

    Reuse the extracted text, chunks, and vectors of unchanged files so only
    new or changed files are read and embedded again.

- **Token usage**

    Count the prompt and generated tokens of the answering and router models
    for each answer.

Module Structure
----------------

- ``ANSWER_PHASE_*`` constants and ``AnswerPhase``

    Canonical machine-readable phases emitted while answering a question.

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
>>> import kilin
>>> session = kilin.RAGSession(
...     "/path/to/configured/workspace",
...     trace=kilin.TraceConfig(enabled=False),
... )
>>> try:
...     answer = session.ask("How does this repository select files?")
... finally:
...     session.close()

Disabling tracing does not disable conversation checkpoint persistence.
An exclusive companion-file lock protects each active conversation path while
it is open, including when tracing is disabled. The workspace and model
dependencies must already be suitable for its profile.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path, PureWindowsPath
import platform
import re
import sys
from time import perf_counter
from typing import Final, Literal, TypeAlias

from kilin.atomic_writes import atomic_write_text
from kilin.defaults import (
    load_defaults_section,
    require_default_int,
    require_default_number,
)
from kilin.answering import (
    MACHINE_STATE_ANSWERING,
    RouteDecision,
    history_messages,
    route_question,
    search_text as answering_search_text,
    select_history,
    side_reply,
    turn_representation,
)
from kilin.chat import (
    ASSISTANT_ROLE,
    SYSTEM_ROLE,
    USER_ROLE,
    ChatSession,
)
from kilin.chunking import Chunk, chunk_source_units
from kilin.embeddings import EmbeddingSession
from kilin.conversation_format import conversation_state
from kilin.conversation_ownership import (
    ConversationFileLock,
    conversation_write_ownership,
)
from kilin.project import QAProject, open_project
from kilin.project_layout import source_scan_exclusions
from kilin.workspace import include_explicit_imports
from kilin.project_type import (
    HISTORY_POLICY_RECENT,
    RETRIEVAL_MODE_DOCUMENT,
    RETRIEVAL_MODE_REUSE,
    RETRIEVAL_MODE_SEARCH,
    project_type,
)
from kilin.query_formats import (
    QUERY_FORMAT_PLAIN,
    format_qwen3_query,
)
from kilin.repository import (
    build_repository_manifest,
    compare_repository_manifests,
    scan_repository,
)
from kilin.retrievals.model_identity import embedding_model_identity
from kilin.retrievals import (
    CodeGraph,
    DocumentSelection,
    HybridResult,
    LexicalIndex,
    PersistentIndexes,
    SemanticIndex,
    build_code_graph,
    build_lexical_index,
    build_semantic_index,
    code_search,
    expand_retrieval_context,
    hybrid_search,
    lexical_search,
    load_indexes,
    plan_context_expansion,
    prepare_persistent_indexes,
    read_document,
    save_indexes,
    semantic_search,
)
from kilin.retrievals.context_expansion import (
    SOURCE_SEARCH_DEPTH,
    demote_documentation,
    demote_test_results,
    names_code,
    prefer_source_results,
)
from kilin.runtime import BACKEND_MLX, SelectedBackend, choose_backend
from kilin.sources import (
    SourceFile,
    extract_source_units,
    read_repository_files,
)
from kilin.sources.extensions.sources_pdf import PDF_TEXT_EXTRACTOR
from kilin.tools import (
    EarlierTurn,
    ToolContext,
    ToolPlan,
    ToolRequest,
    ToolResult,
    ToolUse,
    run_tool,
)
from kilin.tools._shared.choose_document import document_scope
from kilin.trace_logging import (
    TRACE_EVENT_PROGRESS_UPDATED,
    NullTraceLogger,
    TraceConfig,
    TraceLogger,
    TraceSink,
    create_trace_logger,
    trace_text_preview,
)

__all__ = [
    "ANSWER_PHASE_CHOICES",
    "ANSWER_PHASE_COMPLETED",
    "ANSWER_PHASE_EVIDENCE_READY",
    "ANSWER_PHASE_GENERATING",
    "ANSWER_PHASE_PROCESSING",
    "ANSWER_PHASE_RETRIEVING",
    "ANSWER_PHASE_ROUTED",
    "ANSWER_PHASE_ROUTING",
    "ANSWER_PHASE_SAVING",
    "AnswerPhase",
    "ConversationTurn",
    "Evidence",
    "ProgressCallback",
    "ProgressEvent",
    "RAGSession",
]

AnswerPhase: TypeAlias = Literal[
    "routing",
    "routed",
    "retrieving",
    "evidence_ready",
    "generating",
    "saving",
    "completed",
    "processing",
]
ANSWER_PHASE_ROUTING: Final[AnswerPhase] = "routing"
ANSWER_PHASE_ROUTED: Final[AnswerPhase] = "routed"
ANSWER_PHASE_RETRIEVING: Final[AnswerPhase] = "retrieving"
ANSWER_PHASE_EVIDENCE_READY: Final[AnswerPhase] = "evidence_ready"
ANSWER_PHASE_GENERATING: Final[AnswerPhase] = "generating"
ANSWER_PHASE_SAVING: Final[AnswerPhase] = "saving"
ANSWER_PHASE_COMPLETED: Final[AnswerPhase] = "completed"
# A tool's own steps, such as "Classifying batch 3 of 12".
ANSWER_PHASE_PROCESSING: Final[AnswerPhase] = "processing"
ANSWER_PHASE_CHOICES: Final[tuple[AnswerPhase, ...]] = (
    ANSWER_PHASE_ROUTING,
    ANSWER_PHASE_ROUTED,
    ANSWER_PHASE_RETRIEVING,
    ANSWER_PHASE_EVIDENCE_READY,
    ANSWER_PHASE_GENERATING,
    ANSWER_PHASE_SAVING,
    ANSWER_PHASE_COMPLETED,
    ANSWER_PHASE_PROCESSING,
)

# Limits for evidence reuse and progress reporting are package defaults.
_RAG_DEFAULTS = load_defaults_section("RAG")
_EMBEDDING_PROGRESS_BATCH_SIZE: Final[int] = require_default_int(
    _RAG_DEFAULTS, "EMBEDDING_PROGRESS_BATCH_SIZE", "RAG"
)
_PROGRESS_MIN_INTERVAL_SECONDS: Final[float] = require_default_number(
    _RAG_DEFAULTS, "PROGRESS_MIN_INTERVAL_SECONDS", "RAG"
)
_PROGRESS_TRACE_PERCENT_STEP: Final[float] = require_default_number(
    _RAG_DEFAULTS, "PROGRESS_TRACE_PERCENT_STEP", "RAG"
)
if _PROGRESS_TRACE_PERCENT_STEP > 100:
    raise ValueError("RAG.PROGRESS_TRACE_PERCENT_STEP cannot exceed 100.")
_SYSTEM_PROMPT = (
    "Answer only from the supplied project evidence. Put source numbers such "
    "as [1] and [2] directly after supported claims. Do not write a separate "
    "reference list or copy bibliographic text. Say when evidence "
    "is insufficient."
)
# File in the index folder caching each source file's extracted text with the
# SHA-256 of its bytes, so unchanged files (slow PDFs above all) are not read
# again when sources are re-indexed.
_TEXT_CACHE: Final[str] = "source_text.json"
# How indexed text is built from passages (``_retrieval_text``); a change
# rebuilds saved indexes.
RETRIEVAL_TEXT_FORMAT: Final[str] = "code-path-and-module-summary-v1"
# Longest module summary put in front of a code passage for searching.
_SUMMARY_MAX_CHARACTERS: Final[int] = 200
# Opening of a module docstring, after optional comment lines.
_MODULE_DOCSTRING: Final[re.Pattern[str]] = re.compile(
    r"\A(?:[ \t]*(?:#[^\n]*)?\n)*[ \t]*[rRuU]?(\"\"\"|\'\'\')"
)
# Most items a reused-evidence answer keeps; oldest evidence is dropped.
_MAX_REUSED_EVIDENCE: Final[int] = require_default_int(
    _RAG_DEFAULTS, "MAX_REUSED_EVIDENCE", "RAG"
)
# Most items earlier related turns add to freshly retrieved evidence.
_MAX_RELATED_EVIDENCE: Final[int] = require_default_int(
    _RAG_DEFAULTS, "MAX_RELATED_EVIDENCE", "RAG"
)
# Items used from a related answer that cited none of its evidence.
_UNCITED_RELATED_EVIDENCE: Final[int] = require_default_int(
    _RAG_DEFAULTS, "UNCITED_RELATED_EVIDENCE", "RAG"
)
# Earlier turns made available when resolving references in tool requests.
_MAX_EARLIER_TURNS: Final[int] = require_default_int(
    _RAG_DEFAULTS, "MAX_EARLIER_TURNS", "RAG"
)
# A citation group in an answer, such as "[2]" or "[1, 3]".
_CITATION_GROUP: Final[re.Pattern[str]] = re.compile(
    r"\[(\d+(?:\s*,\s*\d+)*)\]"
)
# The separator between numbers inside one citation group.
_CITATION_SEPARATOR: Final[re.Pattern[str]] = re.compile(r"\s*,\s*")


def _invalid_citations(
    answer: str, evidence: Sequence[Evidence]
) -> tuple[int | str, ...]:
    """Identify answer citation numbers absent from the supplied evidence.

    Parameters
    ----------
    answer: str
        Generated answer to inspect without changing it.
    evidence: Sequence[Evidence]
        Evidence shown for the current answer, including tool results.

    Returns
    -------
    tuple[int | str, ...]
        Invalid identifiers, in first-occurrence order without duplicates.
        Extremely long identifiers stay strings to avoid conversion limits.
        An answer without citation groups produces an empty tuple.

    Notes
    -----
    This inexpensive syntax check does not verify that cited evidence
    supports the accompanying claim.

    """

    if "[" not in answer:
        return ()
    available = {item.citation for item in evidence}
    invalid: dict[int | str, None] = {}
    for group in _CITATION_GROUP.findall(answer):
        for text in _CITATION_SEPARATOR.split(group):
            # Python limits integer conversion of enormous digit strings.
            # Such groups cannot refer to the small evidence identifiers.
            normalized = text.lstrip("0") or "0"
            number: int | str = (
                normalized if len(normalized) > 15 else int(normalized)
            )
            if number not in available:
                invalid[number] = None
    return tuple(invalid)


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
AnswerPhaseCallback = Callable[[AnswerPhase, dict[str, object]], None]


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
        KíLín project folder containing profile.ini.
    progress_callback: ProgressCallback | None, optional
        Optional callback receiving ProgressEvent records during preparation.
        Callback exceptions propagate. Defaults to ``None``.
    trace: TraceConfig | TraceSink | None, optional
        Trace configuration or caller-owned sink. None requests a new enabled
        logger with complete diagnostic recording. TraceConfig(enabled=False)
        disables tracing but not conversation persistence. Defaults to
        ``None``.
    conversation_path: str | pathlib.Path | None, optional
        Optional conversation_*.json path under project logs. Relative paths
        start at the project root; None chooses a new path under logs/traces.
        When tracing is enabled, writes go through the sink, so configure its
        output path consistently with this path. Defaults to ``None``.
    router_model: str | None, optional
        Chat model that classifies each question's intent. None uses the
        profile's chat model. Defaults to ``None``.

    Attributes
    ----------
    project: QAProject
        Resolved project paths and settings.
    backend: "mlx" | "torch"
        Selected model backend.
    index_status: str
        Either ``reused``, ``rebuilt``, or ``empty`` for this opening.
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
    router_model: str
        Chat model that classifies intents; it is loaded separately only when
        it differs from the answering model.

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
    refresh_sources(project)
        Update the indexes after sources were added, changed, or removed.
    use_chat_model(model_id)
        Switch the answering model while preserving conversation state.
    use_router_model(model_id)
        Change the router model, releasing the previous one if separate.
    router_reply(system, prompt, *, max_new_tokens)
        Ask the router model a side question, such as a chat title.
    usage_mark()
        Remember the token totals of the loaded model sessions.
    usage_since(mark)
        Return the tokens read and generated since ``usage_mark``.

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
    Construction acquires exclusive ownership of the conversation file and
    releases it on failure or close. A rebuilt index reads selected source
    contents and writes index/manifest metadata. Detailed traces and
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
        router_model: str | None = None,
    ) -> None:
        """Open a project, its models, indexes, and saved conversation.

        Parameters
        ----------
        project_directory: str | pathlib.Path
            KíLín project folder containing profile.ini.
        progress_callback: ProgressCallback | None, optional
            Optional callback receiving ProgressEvent records during
            preparation. Callback exceptions propagate. Defaults to ``None``.
        trace: TraceConfig | TraceSink | None, optional
            Trace configuration or caller-owned sink. ``None`` requests a new
            enabled logger recording full diagnostic information. An enabled
            TraceConfig without an output path is bound there automatically.
            An enabled configuration or sink with a different output path is
            rejected. ``TraceConfig(enabled=False)`` disables tracing but not
            conversation persistence. Defaults to ``None``.
        conversation_path: str | pathlib.Path | None, optional
            Optional ``conversation_*.json`` path under project logs. Relative
            paths start at the project root; ``None`` chooses a new path under
            ``logs/traces``. This path is authoritative for resumable state and
            enabled RAG tracing. Defaults to ``None``.
        router_model: str | None, optional
            Chat model that classifies each question's intent. It stays
            loaded beside the answering model, or is shared with it when
            both are the same model. Defaults to ``None``, which uses the
            profile's chat model.

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

        self._conversation_ownership: ConversationFileLock | None = None
        self._owns_trace = False
        try:
            self._initialize(
                project_directory,
                progress_callback=progress_callback,
                trace=trace,
                conversation_path=conversation_path,
                router_model=router_model,
            )
        except BaseException as error:
            ownership = self._conversation_ownership
            try:
                if getattr(self, "_owns_trace", False):
                    try:
                        self.trace.close()
                    except BaseException as cleanup_error:
                        error.add_note(
                            f"Trace cleanup failed: {cleanup_error!r}"
                        )
            finally:
                if ownership is not None:
                    try:
                        ownership.release()
                    except BaseException as cleanup_error:
                        error.add_note(
                            f"Conversation lock cleanup failed: "
                            f"{cleanup_error!r}"
                        )
            raise

    def _initialize(
        self,
        project_directory: str | Path,
        *,
        progress_callback: ProgressCallback | None,
        trace: TraceConfig | TraceSink | None,
        conversation_path: str | Path | None,
        router_model: str | None,
    ) -> None:
        """Prepare the owned trace, conversation history, indexes and models.

        Parameters
        ----------
        project_directory: str | pathlib.Path
            Existing project workspace to open.
        progress_callback: ProgressCallback | None
            Optional preparation-progress observer.
        trace: TraceConfig | TraceSink | None
            Trace configuration or a caller-owned logger.
        conversation_path: str | pathlib.Path | None
            Path of the conversation record, or None for a new one.
        router_model: str | None
            Optional separate model identifier used to route questions.

        Notes
        -----
        ``__init__`` is responsible for releasing the acquired conversation
        ownership and an owned logger if any preparation step raises, even
        before model construction begins.

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
        self.trace, self._owns_trace = _conversation_trace(
            trace,
            self.conversation_path,
        )
        if not self.trace.enabled:
            ownership = ConversationFileLock(self.conversation_path)
            ownership.acquire()
            self._conversation_ownership = ownership
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
            "Opening KíLín project...",
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
                system_prompt=project_type(profile.type_id).system_prompt,
                allow_download=profile.allow_download,
                trace=self.trace,
                block_unrequested_cjk=profile.block_unrequested_cjk,
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
        except BaseException as error:
            try:
                self.trace.exception(
                    "session.open.failed",
                    error,
                    component="rag",
                    duration_ms=(perf_counter() - session_started_at) * 1000,
                )
            except BaseException as cleanup_error:
                error.add_note(
                    f"Failure tracing initialization: {cleanup_error!r}"
                )
            # Close both partially created models even if one close fails.
            # Preserve the initialization failure as the primary exception.
            for model in (chat_session, embedding_session):
                if model is None:
                    continue
                try:
                    model.close()
                except BaseException as cleanup_error:
                    error.add_note(
                        f"Model cleanup failed: {cleanup_error!r}"
                    )
            # __init__ releases the owned trace and conversation lock on
            # every construction failure, including this one.
            raise

        self.embedding_session = embedding_session
        self.chat_session = chat_session
        self.chunks = chunks
        self.indexes = indexes
        self.index_status = status
        self.source_changes = source_changes
        self.turns = turns
        self.router_model = router_model or profile.chat_model(self.backend)
        self._router_session: ChatSession | None = None
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
            f"KíLín ready. Index {self.index_status}.",
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
                TRACE_EVENT_PROGRESS_UPDATED,
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

    def ask(
        self: RAGSession,
        question: str,
        *,
        retrieval_count: int | None = None,
        generation_settings: dict[str, object] | None = None,
        phase_callback: AnswerPhaseCallback | None = None,
        answer_guidance: str | None = None,
        token_callback: Callable[[str], None] | None = None,
        retrieval_query: str | None = None,
        reuse_evidence: bool = False,
        use_router: bool = True,
        confirm_callback: Callable[[ToolPlan], str] | None = None,
        pdf_writer: Callable[..., int] | None = None,
        choice_callback: (
            Callable[[str, tuple[str, ...]], int | None] | None
        ) = None,
    ) -> str:
        """Retrieve evidence, generate an answer, and save the turn.

        The initial retrieval count applies to each ranking and fusion stage;
        explicit file/function context may expand the final evidence.
        Generation settings supplied here persist on the loaded chat session
        after validation.

        Each question is first routed to one of the project type's intents.
        The intent decides which earlier turns are sent as history and
        whether evidence is searched, read from one document, reused from
        earlier turns, or skipped. An intent with a tool lets the tool write
        the answer instead of the answering model.

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
            phases. Exceptions during processing propagate. An exception from
            the final completed-phase notification is recorded in the trace
            without invalidating a committed answer, unless strict trace
            logging itself fails. Defaults to ``None``.
        answer_guidance: str | None, optional
            Answer instruction for this question. Intent guidance takes
            precedence, followed by explicit caller guidance and the project
            type's default. ``None`` inherits the project default; an empty
            string disables that default unless the intent has guidance.
            The resolved instruction is saved with the turn.
        token_callback: Callable[[str], None] | None, optional
            Receives answer text pieces as they are generated, for live
            display; see ``ChatSession.reply``. Defaults to ``None``.
        retrieval_query: str | None, optional
            Text to search with instead of the question, such as a follow-up
            rewritten to stand alone. The question itself is still answered.
            Saved with the turn. Defaults to ``None``.
        reuse_evidence: bool, optional
            Answer from the previous turn's evidence without searching, for a
            follow-up about the same sources. Saved with the turn. Defaults to
            ``False``.
        use_router: bool, optional
            Classify the question's intent with the router model. When False,
            the project type's default intent is used. Defaults to ``True``.
        confirm_callback: Callable[[ToolPlan], str] | None, optional
            Receives the plan of an expensive tool run (such as
            ``batch_classify``) when its confirmation policy asks, and
            returns a ``PlanChoice``: ``"continue"``, ``"quick"``, or
            ``"cancel"``. None runs every plan without asking. Defaults to
            ``None``.
        pdf_writer: Callable[..., int] | None, optional
            ``pdf_writer(markdown, path, label=...)`` writes a report as a
            PDF and returns its page count; tools that save reports also
            save a PDF when it is given (the desktop app passes a Qt
            writer). Defaults to ``None``.
        choice_callback: Callable[..., int | None] | None, optional
            ``choice_callback(question, options)`` asks the user to pick one
            option and returns its index, or None to cancel; tools use it
            when they cannot tell what the user means (for example which
            paper to summarize). None lets tools fall back to their best
            guess. Defaults to ``None``.

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
            Also if an optional argument has the wrong type or value, or
            ``reuse_evidence`` is True before any answered turn.

        Notes
        -----
        A generated turn is committed to memory only after its resumable
        checkpoint has been written. If persistence fails, the previous
        chat-model history is restored. The final completed-phase callback
        runs after the turn span closes; its failure is traced separately and
        does not invalidate the answer. Other callback errors and explicit
        strict diagnostic-write failures can still propagate after commit.
        Citation identifiers are checked once after generation against
        this turn's evidence. Invalid identifiers are recorded without
        changing the answer or retrying generation. The content of claims
        is not checked. History and evidence have no aggregate token guard.

        """

        if self._closed:
            raise RuntimeError("This RAG session is closed.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Enter a non-empty question.")
        if answer_guidance is not None and not isinstance(
            answer_guidance, str
        ):
            raise ValueError("answer_guidance must be text or None.")
        if retrieval_count is not None and (
            type(retrieval_count) is not int or retrieval_count < 1
        ):
            raise ValueError(
                "retrieval_count must be a positive whole number."
            )
        if confirm_callback is not None and not callable(confirm_callback):
            raise ValueError("confirm_callback must be callable or None.")
        if phase_callback is not None and not callable(phase_callback):
            raise ValueError("phase_callback must be callable or None.")
        if retrieval_query is not None and (
            not isinstance(retrieval_query, str) or not retrieval_query.strip()
        ):
            raise ValueError("retrieval_query must be non-empty text or None.")
        if type(reuse_evidence) is not bool:
            raise ValueError("reuse_evidence must be True or False.")
        if reuse_evidence and not self.turns:
            raise ValueError("There is no previous answer to reuse.")

        def report_phase(phase: AnswerPhase, **details: object) -> None:
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
        # Keep the committed model history available if saving fails after
        # generation temporarily changes its message list.
        committed_messages = [
            dict(message) for message in self.chat_session.messages
        ]
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
                role in {USER_ROLE, ASSISTANT_ROLE} for role in history_roles
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
        usage = self.usage_mark()
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

            # Machine state "answering": route, select history, retrieve,
            # then answer. Configuration, not the router, decides behavior.
            config = project_type(self.project.profile.type_id)
            decision = self._route(
                question, config, use_router, turn_id, report_phase
            )
            intent = config.intents[decision.intent]
            # Resolve once, including for tool-written answers. An explicit
            # empty string disables the project default, not intent guidance.
            if intent.answer_guidance:
                resolved_guidance = intent.answer_guidance
            elif answer_guidance is not None:
                resolved_guidance = answer_guidance
            else:
                resolved_guidance = config.answer_guidance
            if resolved_guidance or answer_guidance is not None:
                effective["answer_guidance"] = resolved_guidance
            mode = intent.retrieval_mode
            if reuse_evidence:
                mode = RETRIEVAL_MODE_REUSE

            with trace.span(
                "history.selection",
                component="rag",
                data={
                    "policy": intent.history_policy,
                    "recent_turns": config.recent_turns,
                },
                turn_id=turn_id,
            ):
                history_ids = select_history(
                    len(self.turns),
                    intent.history_policy,
                    config.recent_turns,
                    decision,
                )
            recent_ids = select_history(
                len(self.turns),
                HISTORY_POLICY_RECENT,
                config.recent_turns,
                decision,
            )
            trace.emit(
                "history.selection.result",
                component="rag",
                data={
                    "recent_turn_ids": [
                        i for i in recent_ids if i in history_ids
                    ],
                    "related_turn_ids": list(decision.related_turn_ids),
                    "final_turn_ids": history_ids,
                },
                turn_id=turn_id,
            )

            evidence: tuple[Evidence, ...] = ()
            source_turns: list[int] = []
            if mode == RETRIEVAL_MODE_REUSE:
                # Reuse the related turns' evidence, or the last turn's
                # when the router related none.
                candidates = list(decision.related_turn_ids) or [
                    len(self.turns)
                ]
                source_turns = [
                    number
                    for number in candidates
                    if 0 < number <= len(self.turns)
                    and self.turns[number - 1].evidence
                ]
                if not source_turns:
                    # Nothing to reuse, so search instead.
                    mode = RETRIEVAL_MODE_SEARCH

            search_text = None
            if mode in (RETRIEVAL_MODE_SEARCH, RETRIEVAL_MODE_DOCUMENT):
                if retrieval_query is not None:
                    search_text = retrieval_query.strip()
                else:
                    # The user's own words; a question that points back
                    # also carries its related turns' questions.
                    search_text = answering_search_text(
                        question, self.turns, decision
                    )

            effective.update(
                {
                    "machine_state": MACHINE_STATE_ANSWERING,
                    "intent": intent.intent_id,
                    "standalone": decision.standalone,
                    "related_turn_ids": list(decision.related_turn_ids),
                    "history_turn_ids": history_ids,
                    "retrieval_mode": mode,
                }
            )
            report_phase(
                ANSWER_PHASE_ROUTED,
                intent=intent.intent_id,
                history_turn_ids=history_ids,
                related_turn_ids=list(decision.related_turn_ids),
                retrieval_mode=mode,
            )
            trace.emit(
                "retrieval.plan",
                component="rag",
                data={
                    "mode": mode,
                    "strategy": intent.retrieval_strategy,
                    "intent": intent.intent_id,
                    "query_extended": search_text not in (None, question),
                    "prefer_source_code": intent.prefer_source_code,
                    "source_search_depth": (
                        SOURCE_SEARCH_DEPTH
                        if intent.prefer_source_code
                        else None
                    ),
                },
                turn_id=turn_id,
            )

            retrieval_started = perf_counter()
            if mode == RETRIEVAL_MODE_REUSE:
                evidence = _merge_evidence(
                    [
                        self.turns[number - 1].evidence
                        for number in source_turns
                    ]
                )
                effective["evidence_reused"] = True
                effective["evidence_from_turns"] = source_turns
                report_phase(
                    ANSWER_PHASE_EVIDENCE_READY,
                    message=f"Reusing {len(evidence)} previous references",
                    evidence_count=len(evidence),
                    reused=True,
                )
            elif mode == RETRIEVAL_MODE_SEARCH:
                if search_text != question:
                    effective["retrieval_query"] = search_text
                report_phase(
                    ANSWER_PHASE_RETRIEVING,
                    message="Retrieving evidence",
                )
                # A document the question names, or the one the earlier
                # turns discussed, limits the search (see ``_retrieve``).
                # The candidates are the indexed files in index order.
                scope, _ = document_scope(
                    ToolRequest(
                        question,
                        search_text,
                        earlier=self._earlier_turns(decision),
                    ),
                    self._ask_router,
                    list(dict.fromkeys(chunk.path for chunk in self.chunks)),
                )
                evidence = self._retrieve(
                    search_text,
                    retrieval_count,
                    trace_turn_id=turn_id,
                    status_callback=lambda message: report_phase(
                        ANSWER_PHASE_RETRIEVING, message=message
                    ),
                    scope=scope,
                    prefer_source_code=intent.prefer_source_code,
                )
                found = len(evidence)
                related_turns = [
                    number
                    for number in decision.related_turn_ids
                    if 0 < number <= len(self.turns)
                ]
                if intent.include_related_evidence and related_turns:
                    # Follow-ups keep the sources their earlier answers
                    # cited, whatever intent the router chose.
                    evidence = _add_related_evidence(
                        evidence,
                        [self.turns[number - 1] for number in related_turns],
                    )
                    effective["related_evidence_count"] = len(evidence) - found
                    effective["evidence_from_turns"] = related_turns
                report_phase(
                    ANSWER_PHASE_EVIDENCE_READY,
                    message=f"{found} references found",
                    evidence_count=len(evidence),
                    related_count=len(evidence) - found,
                )
            elif mode == RETRIEVAL_MODE_DOCUMENT:
                if search_text != question:
                    effective["retrieval_query"] = search_text
                report_phase(
                    ANSWER_PHASE_RETRIEVING,
                    message="Finding the document to read",
                )
                selection = self._read_document(
                    search_text,
                    intent.max_tokens,
                    retrieval_count,
                    trace_turn_id=turn_id,
                )
                if selection is None:
                    evidence = ()
                else:
                    evidence = _document_evidence(selection)
                    effective["document"] = {
                        **_document_data(selection),
                        "max_tokens": intent.max_tokens,
                    }
                report_phase(
                    ANSWER_PHASE_EVIDENCE_READY,
                    message=_document_message(selection),
                    evidence_count=len(evidence),
                    document=effective.get("document"),
                )
            else:
                report_phase(
                    ANSWER_PHASE_EVIDENCE_READY,
                    message="No retrieval needed",
                    evidence_count=0,
                    skipped=True,
                )

            trace.emit(
                "retrieval.result",
                component="rag",
                data={
                    "mode": mode,
                    "strategy": intent.retrieval_strategy,
                    "evidence_count": len(evidence),
                    "duration_ms": round(
                        (perf_counter() - retrieval_started) * 1000, 3
                    ),
                },
                turn_id=turn_id,
            )

            if intent.tool is not None:
                # The tool, not the answering model, writes the answer.
                result = self._run_tool(
                    intent.tool,
                    question,
                    decision,
                    trace_turn_id=turn_id,
                    report_phase=report_phase,
                    confirm_callback=confirm_callback,
                    pdf_writer=pdf_writer,
                    choice_callback=choice_callback,
                    token_callback=token_callback,
                    answer_guidance=resolved_guidance,
                )
                answer = result.answer
                evidence = tuple(Evidence(**item) for item in result.evidence)
                effective["tool"] = result.summary
                if token_callback is not None and not result.streamed:
                    token_callback(answer)
            else:
                system_prompt = (
                    f"{config.system_prompt}\n\n{intent.system_prompt}"
                )
                # The model sees only the selected turns (questions and
                # answers, without their evidence), so the history is
                # rebuilt for every question.
                self.chat_session.restore_history(
                    [
                        {"role": SYSTEM_ROLE, "content": system_prompt},
                        *history_messages(self.turns, history_ids),
                    ]
                )
                prompt = _answer_prompt(
                    question, evidence, resolved_guidance
                )
                trace.emit(
                    "generation.input.prepared",
                    component="rag",
                    data={
                        "machine_state": MACHINE_STATE_ANSWERING,
                        "project_type_id": config.id,
                        "intent": intent.intent_id,
                        "history_turn_ids": history_ids,
                        "history_message_count": 2 * len(history_ids),
                        "retrieval_mode": mode,
                        "retrieval_strategy": intent.retrieval_strategy,
                        "evidence_count": len(evidence),
                        "prompt_characters": len(prompt) + len(system_prompt),
                    },
                    turn_id=turn_id,
                )
                report_phase(
                    ANSWER_PHASE_GENERATING,
                    message="Generating answer",
                    evidence_count=len(evidence),
                )
                reply_options: dict[str, object] = {"trace_turn_id": turn_id}
                if token_callback is not None:
                    reply_options["token_callback"] = token_callback
                answer = self.chat_session.reply(prompt, **reply_options)

                # "length" marks an answer cut off by the token limit. Other
                # values, or a session without the attribute, record nothing.
                finish_reason = getattr(
                    self.chat_session, "last_finish_reason", None
                )
                if finish_reason in ("length", "stop"):
                    effective["finish_reason"] = finish_reason

            # One fast, deterministic check; do not modify or retry the
            # answer even when the model cites unavailable evidence.
            invalid = _invalid_citations(answer, evidence)
            if invalid:
                # The existing checkpoint records the flag in turn metadata;
                # avoid an additional trace event on the answer path.
                effective["invalid_citations"] = list(invalid)

            # Tokens the models read and generated for this answer.
            effective["usage"] = self.usage_since(usage)
            turn = ConversationTurn(
                timestamp=datetime.now(timezone.utc).isoformat(),
                question=question,
                answer=answer,
                evidence=evidence,
                parameters=effective,
            )
            # A compact summary lets the router find this turn later.
            representation = turn_representation(turn, len(self.turns) + 1)
            effective["representation"] = representation
            trace.emit(
                "conversation.turn.representation",
                component="rag",
                data=representation,
                turn_id=turn_id,
            )
            report_phase(
                ANSWER_PHASE_SAVING,
                message="Saving conversation",
            )
            try:
                self._save_conversation(
                    pending_turn=turn,
                    trace_turn_id=turn_id,
                )
            except Exception:
                # A checkpoint event can fail *after* persistence succeeds.
                # In that case _save_conversation already committed the turn;
                # only a genuine pre-commit failure restores model history.
                if not self.turns or self.turns[-1] is not turn:
                    self.chat_session.restore_history(committed_messages)
                raise
        # Completion notifications are observers, not part of the committed
        # turn. Close the successful turn span before invoking the callback so
        # a presentation failure cannot be mistaken for a generation failure.
        try:
            report_phase(
                ANSWER_PHASE_COMPLETED,
                message="Answer ready",
                evidence_count=len(evidence),
            )
        except Exception as error:
            trace.exception(
                "turn.notification.failed",
                error,
                component="rag",
                data={"phase": ANSWER_PHASE_COMPLETED},
                turn_id=turn_id,
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
            When enabled, its output path must be the target conversation path.

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
        Loaded models and indexes are reused. The target file's exclusive
        ownership is validated or acquired before releasing the old path.
        A failed acquisition or JSON read leaves the active session unchanged.
        Later model/history reconfiguration is not fully transactional.
        Generation settings come from the last saved turn or profile defaults.

        """

        if self._closed:
            raise RuntimeError("This RAG session is closed.")
        target_path = self._resolve_conversation_path(conversation_path)
        _check_conversation_trace(trace, target_path)
        old_ownership = getattr(self, "_conversation_ownership", None)
        target_ownership: ConversationFileLock | None = None
        if not trace.enabled:
            if (
                target_path == self.conversation_path
                and old_ownership is not None
                and old_ownership.holds(target_path)
            ):
                target_ownership = old_ownership
            else:
                target_ownership = ConversationFileLock(target_path)
                target_ownership.acquire()

        try:
            # Acquire and validate the destination before changing any active
            # session references or releasing the previous conversation.
            messages, turns, checkpoint_id = self._load_conversation(
                path=target_path
            )
        except BaseException:
            if (
                target_ownership is not None
                and target_ownership is not old_ownership
            ):
                target_ownership.release()
            raise

        previous_trace = self.trace
        previous_owns_trace = self._owns_trace
        if previous_owns_trace and trace is not previous_trace:
            previous_trace.close()
        if (
            old_ownership is not None
            and old_ownership is not target_ownership
        ):
            old_ownership.release()

        self.trace = trace
        self._owns_trace = (
            previous_owns_trace and trace is previous_trace
        )
        self._conversation_ownership = target_ownership
        self.chat_session.trace = trace
        self.embedding_session.trace = trace
        self.conversation_path = target_path
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
        instruction. An enabled trace then discards conversation-specific
        retained state before recording the reset in a fresh active file.
        Loaded models, indexes, and non-conversation trace context are reused.

        """

        if self._closed:
            raise RuntimeError("This RAG session is closed.")
        archived_checkpoint_id = getattr(self, "_checkpoint_id", None)
        archived_path = self._archive_conversation()
        self.chat_session.reset()
        self.turns = []
        self._checkpoint_id = None
        trace = getattr(self, "trace", NullTraceLogger())
        trace.reset_conversation_state()
        trace.emit(
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

    def _close_model_sessions(self: RAGSession) -> list[BaseException]:
        """Attempt to close every model owned by this RAG session.

        Returns
        -------
        list[BaseException]
            Exceptions from individual model cleanup operations, in model
            close order. All available models are attempted independently.

        Notes
        -----
        The separate router is detached even when its cleanup raises.
        Models are not closed during an active model replacement here; this
        helper is used only during final session shutdown.

        """

        errors: list[BaseException] = []
        for attribute in (
            "chat_session",
            "_router_session",
            "embedding_session",
        ):
            model = getattr(self, attribute, None)
            if model is None:
                continue
            if attribute == "_router_session":
                self._router_session = None
            try:
                model.close()
            except BaseException as error:
                errors.append(error)
        return errors

    def close(self: RAGSession) -> None:
        """Attempt all owned cleanup and release conversation ownership.

        Raises
        ------
        BaseException
            The first cleanup error, after attempting remaining model,
            trace, and ownership cleanup. Further errors are attached as
            exception notes when supported by Python.

        Notes
        -----
        The closed flag is set even when backend cleanup raises. An owned
        trace releases its file lease; caller-owned traces remain open and
        must be closed by their owner. A tracing-disabled session releases
        its own conversation lease. Failure of one cleanup step never skips
        another owned resource.

        """

        if self._closed:
            return
        self._closed = True
        trace = getattr(self, "trace", NullTraceLogger())
        errors = self._close_model_sessions()
        try:
            if errors:
                trace.emit(
                    "session.closed",
                    component="rag",
                    status="failed",
                    data={"model_cleanup_errors": len(errors)},
                )
            else:
                trace.emit("session.closed", component="rag")
        except BaseException as error:
            errors.append(error)
        try:
            if getattr(self, "_owns_trace", False):
                trace.close()
        except BaseException as error:
            errors.append(error)
        finally:
            ownership = getattr(self, "_conversation_ownership", None)
            if ownership is not None:
                self._conversation_ownership = None
                try:
                    ownership.release()
                except BaseException as error:
                    errors.append(error)
        if errors:
            for error in errors[1:]:
                errors[0].add_note(f"Additional cleanup failure: {error!r}")
            raise errors[0]

    def refresh_sources(self: RAGSession, project: QAProject) -> None:
        """Update the indexes after sources were added, changed, or removed.

        The loaded models and the conversation stay; only the indexes are
        prepared again, and files that did not change keep their chunks and
        vectors, so only new or changed files are embedded.

        Parameters
        ----------
        project: QAProject
            The project reopened after its sources changed.

        Notes
        -----
        The new project replaces the old one before indexing starts, so a
        failure while indexing leaves the new project with the previous
        chunks and indexes.

        """

        self.project = project
        with self.trace.span(
            "indexing",
            component="rag",
            data={"project_directory": str(project.root), "refresh": True},
        ):
            chunks, indexes, status, source_changes = self._prepare_retrieval(
                self.embedding_session
            )
        self.chunks = chunks
        self.indexes = indexes
        self.index_status = status
        self.source_changes = source_changes

    def use_chat_model(self: RAGSession, model_id: str) -> None:
        """Switch the answering model while preserving conversation state.

        The embedding model and retrieval indexes are left unchanged. The old
        model is released before the replacement is loaded, except the router
        model, which stays loaded for routing and is reused when it becomes
        the answering model again. Generation settings and chat history move
        to the new model.

        Parameters
        ----------
        model_id: str
            Hugging Face model ID or local model directory of the new
            answering model. The current model ID returns without change.

        Raises
        ------
        RuntimeError
            If the session is closed or the backend is not MLX.
        Exception
            Any error loading the new model is raised again after the old
            model is restored (reloaded if it had been released).

        """

        if self._closed:
            raise RuntimeError("This RAG session is closed.")
        if self.backend != BACKEND_MLX:
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
        # Release the old model first so two large models are never loaded
        # together, unless it is also the router, which must stay loaded.
        keep_old = old_model_id == getattr(self, "router_model", None)
        if not keep_old:
            old.close()

        router = getattr(self, "_router_session", None)
        try:
            if router is not None and router.model_id == model_id:
                replacement = router
                replacement.configure_generation(settings)
            else:
                replacement = self._new_chat_session(model_id, **settings)
        except Exception:
            # Leave a working answering model behind before re-raising.
            if keep_old:
                self.chat_session = old
            else:
                self.chat_session = self._new_chat_session(
                    old_model_id, **settings
                )
                self.chat_session.restore_history(messages)
            raise

        # The router session now answers, so it is no longer kept apart.
        if replacement is router:
            self._router_session = None
        if keep_old:
            # The router model stays loaded for the next question.
            self._router_session = old
        replacement.restore_history(messages)
        self.chat_session = replacement
        self.trace.emit(
            "chat.model.switched",
            component="rag",
            data={"from": old_model_id, "to": model_id},
        )

    def use_router_model(self: RAGSession, model_id: str) -> None:
        """Change the router model, releasing the previous one if separate.

        Parameters
        ----------
        model_id: str
            Chat model that classifies intents from the next question on.
            It is loaded when first needed, or shared with the answering
            model when both are the same.

        Raises
        ------
        RuntimeError
            If the session is closed.

        """

        if self._closed:
            raise RuntimeError("This RAG session is closed.")
        if model_id == self.router_model:
            return

        # A router shared with the answering model is not in
        # ``_router_session``, so the answering model is never closed here.
        if self._router_session is not None:
            self._router_session.close()
            self._router_session = None
        self.trace.emit(
            "router.model.changed",
            component="rag",
            data={"from": self.router_model, "to": model_id},
        )
        self.router_model = model_id

    def router_reply(
        self: RAGSession, system: str, prompt: str, *, max_new_tokens: int
    ) -> str:
        """Ask the router model a side question, such as a chat title.

        Parameters
        ----------
        system: str
            System instruction for the side question only.
        prompt: str
            The side question.
        max_new_tokens: int
            Maximum length of the reply.

        Returns
        -------
        str
            The reply. The answering model's conversation is not changed.

        """

        return side_reply(
            self._router_chat(),
            system,
            prompt,
            max_new_tokens=max_new_tokens,
        )

    def usage_mark(self: RAGSession) -> dict[int, tuple[int, int]]:
        """Remember the token totals of the loaded model sessions.

        Returns
        -------
        dict[int, tuple[int, int]]
            Prompt and generated token totals per model session, for
            ``usage_since``.

        """

        return {
            id(chat): (chat.input_token_total, chat.output_token_total)
            for chat in self._model_sessions()
        }

    def usage_since(
        self: RAGSession, mark: dict[int, tuple[int, int]]
    ) -> dict[str, int]:
        """Return the tokens read and generated since ``usage_mark``.

        Parameters
        ----------
        mark: dict[int, tuple[int, int]]
            Totals from ``usage_mark``. A session loaded since then counts
            from zero.

        Returns
        -------
        dict[str, int]
            ``input_tokens`` (prompts the models read: questions, evidence,
            instructions, and history) and ``output_tokens`` (text they
            generated), over the answer model, the router, and any tool
            calls. Safe to call while an answer is being generated.

        """

        read = written = 0
        for chat in self._model_sessions():
            before_read, before_written = mark.get(id(chat), (0, 0))
            read += max(0, chat.input_token_total - before_read)
            written += max(0, chat.output_token_total - before_written)
        return {"input_tokens": read, "output_tokens": written}

    def _prepare_source(
        self: RAGSession,
        embedding_session: EmbeddingSession,
        report: dict[str, object] | None = None,
        cached_chunks: dict[Path, list[Chunk]] | None = None,
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
        cached_chunks: dict[pathlib.Path, list[Chunk]] | None, optional
            Chunks of unchanged files from the saved index, by path; these
            files are not chunked again. Defaults to ``None`` (chunk every
            file).

        Returns
        -------
        tuple[list[Chunk], CodeGraph]
            Pair (chunks, code_graph) derived from the selected source files.

        Notes
        -----
        Reads selected content, extracts units, counts tokens through the
        embedding tokenizer, and builds the heuristic code graph. A supplied
        incomplete scan is not rejected here.

        Extracted text is cached in ``_TEXT_CACHE`` under the index folder
        with the SHA-256 of each file's bytes; a file whose bytes are
        unchanged reuses its text instead of being read again. Every selected
        file is still hashed, and every file's text is still extracted into
        units because the code graph needs all of them.

        """

        profile = self.project.profile
        if report is None:
            report = scan_repository(
                self.project.source,
                extensions=profile.extensions,
                max_file_bytes=profile.max_file_bytes,
                exclude_paths=source_scan_exclusions(
                    self.project.root, self.project.source
                ),
            )
            report = include_explicit_imports(self.project, report)
        cached_chunks = cached_chunks or {}
        selected = list(report.get("selected_files", []))
        use_code_graph = project_type(profile.type_id).code_graph is not False

        # Text read earlier is reused when the file's bytes are unchanged
        # (same SHA-256), so only new or changed files are read again;
        # reading is slow for PDFs. A file with the same name but other
        # content has another hash and is read again.
        texts = _read_text_cache(self.project.indexes)
        hashes = {
            name: sha256((self.project.source / name).read_bytes()).hexdigest()
            for name in selected
        }
        reused = {
            name: SourceFile(Path(name), texts[name]["text"])
            for name in selected
            if texts.get(name, {}).get("sha256") == hashes[name]
        }
        # Only files without reusable text are passed on to the reader.
        report = {
            **report,
            "selected_files": [
                name for name in selected if name not in reused
            ],
        }
        selected_count = len(report["selected_files"])

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
            source_files = (
                []
                if not selected_count
                else read_repository_files(
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
            )
        # Rewriting the cache from the selected files drops removed files.
        read = {source.path.as_posix(): source for source in source_files}
        _write_text_cache(
            self.project.indexes,
            {
                name: {
                    "sha256": hashes[name],
                    "text": (reused.get(name) or read[name]).text,
                }
                for name in selected
                if name in reused or name in read
            },
        )
        source_files = [
            reused.get(name) or read[name]
            for name in selected
            if name in reused or name in read
        ]
        self.trace.emit(
            "source.read.summary",
            component="rag",
            data={
                "reused_file_count": len(reused),
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
            units = (
                []
                if not source_files
                else extract_source_units(
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
            )

        # Files with chunks from the saved index are not chunked again.
        changed_units = [
            unit for unit in units if unit.path not in cached_chunks
        ]
        chunking_started = perf_counter()
        unit_count = len(changed_units)
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
                changed_units,
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

        # Merge reused and new chunks in the scan's file order.
        fresh_by_path: dict[Path, list[Chunk]] = {}
        for chunk in chunks:
            fresh_by_path.setdefault(chunk.path, []).append(chunk)
        chunks = [
            chunk
            for name in selected
            for chunk in cached_chunks.get(
                Path(name), fresh_by_path.get(Path(name), [])
            )
        ]
        self.trace.emit(
            "source.chunk.summary",
            component="rag",
            data={
                "chunk_count": len(chunks),
                "settings": chunk_settings,
                "reused_file_count": len(cached_chunks),
                "chunked_unit_count": unit_count,
            },
        )
        self._emit_progress(
            "building_code_graph",
            56.0,
            (
                f"Building source relationships from {len(chunks)} chunks..."
                if use_code_graph and source_files
                else "No source relationships to build; skipping graph"
            ),
            current=len(chunks),
            total=len(chunks),
        )
        with self.trace.span(
            "index.code_graph.build",
            component="rag",
            data={"chunk_count": len(chunks)},
        ):
            if not use_code_graph or not source_files:
                # A project type without a code graph, or a project without
                # files, gets an empty graph; retrieval then skips it.
                code_graph = CodeGraph((), ())
            else:
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
            is ``reused``, ``rebuilt``, or ``empty``.

        Raises
        ------
        ValueError
            If the embedding model is unavailable, the saved index model
            identity differs, or cached embedding vectors are invalid.

        Notes
        -----
        A selected-source manifest and stored configuration metadata control
        reuse. Existing vectors are loaded only if the stored bundle is
        accepted. Source completeness is not checked here; index and metadata
        files are written separately rather than as one transaction.
        Empty collections keep a current inventory in memory but write no
        retrieval bundle. Search is skipped until sources yield chunks.

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
            exclude_paths=source_scan_exclusions(
                self.project.root, self.project.source
            ),
        )
        report = include_explicit_imports(self.project, report)
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

        if not selected_count:
            changes["configuration_changed"] = (
                saved_metadata is None
                or any(
                    saved_metadata.get(key) != value
                    for key, value in expected_metadata.items()
                )
            )
            return self._empty_retrieval(changes, current_manifest)

        # Bind the saved vectors to the actual loaded model, not merely to
        # its configured ID or directory path. Cache identity on the loaded
        # session so source refreshes do not rehash large local weight files.
        model_source = getattr(embedding_session, "model_source", None)
        if not isinstance(model_source, str):
            raise ValueError("The embedding model source is unavailable.")
        model_identity = getattr(
            embedding_session, "_retrieval_model_identity", None
        )
        if model_identity is None:
            model_identity = embedding_model_identity(
                embedding_session.model_id, model_source
            )
            embedding_session._retrieval_model_identity = model_identity
        expected_metadata["embedding_model_source"] = model_source
        expected_metadata["embedding_model_identity"] = model_identity

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
                    if indexes.embedding_model_identity != model_identity:
                        raise ValueError(
                            "Saved index embedding model identity differs."
                        )
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
                self._current_source_manifest = current_manifest
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

        # With the same configuration, unchanged files keep their saved
        # chunks and vectors, so only new or changed files are embedded.
        # Any problem with the saved index means embedding everything.
        cached_chunks: dict[Path, list[Chunk]] = {}
        cached_vectors: dict[str, tuple[Chunk, tuple[float, ...]]] = {}
        if not configuration_changed and saved_manifest is not None:
            try:
                saved = load_indexes(index_path)
                if saved.embedding_model_identity != model_identity:
                    raise ValueError(
                        "Saved index embedding model identity differs."
                    )
                vectors = saved.semantic.embeddings
                old_chunks = saved.semantic.chunks
                # Saved vectors are mixed with new ones only when each chunk
                # has one, all have the same nonzero size, and all values
                # are finite.
                dimensions = {len(vector) for vector in vectors}
                if (
                    len(vectors) != len(old_chunks)
                    or len(dimensions) != 1
                    or 0 in dimensions
                    or any(
                        not isfinite(value)
                        for vector in vectors
                        for value in vector
                    )
                ):
                    raise ValueError("Saved embedding vectors are invalid.")

                changed_paths = set(changes["added"]) | set(
                    changes["modified"]
                )
                unchanged = set(report["selected_files"]) - changed_paths
                cached_chunks = {Path(path): [] for path in unchanged}
                for chunk, vector in zip(old_chunks, vectors):
                    if chunk.path in cached_chunks:
                        cached_chunks[chunk.path].append(chunk)
                        cached_vectors[chunk.identifier] = (chunk, vector)
            except (OSError, ValueError, TypeError, KeyError):
                cached_chunks = {}
                cached_vectors = {}

        chunks, code_graph = self._prepare_source(
            embedding_session,
            report=report,
            cached_chunks=cached_chunks,
        )
        if not chunks:
            return self._empty_retrieval(changes, current_manifest)
        # A chunk is embedded unless an identical saved chunk has a vector.
        pending_chunks = [
            chunk
            for chunk in chunks
            if cached_vectors.get(chunk.identifier, (None,))[0] != chunk
        ]
        embedding_started = perf_counter()
        chunk_count = len(pending_chunks)
        self.trace.emit(
            "index.incremental.plan",
            component="rag",
            data={
                "reused_chunks": len(chunks) - chunk_count,
                "embedded_chunks": chunk_count,
                "added": changes["added"],
                "modified": changes["modified"],
                "removed": changes["removed"],
                "configuration_changed": configuration_changed,
            },
        )
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
            # Code passages are embedded with their file path and module
            # summary in front, so a function is found by what its module
            # is about even when its own name and code do not say it.
            summaries = _module_summaries(chunks)
            fresh_semantic = (
                build_semantic_index(
                    pending_chunks,
                    embedding_session,
                    texts=[
                        _retrieval_text(chunk, summaries)
                        for chunk in pending_chunks
                    ],
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
                if pending_chunks
                else SemanticIndex((), ())
            )

        # Join new and saved vectors in the final chunk order.
        fresh_vectors = {
            chunk.identifier: vector
            for chunk, vector in zip(
                fresh_semantic.chunks, fresh_semantic.embeddings
            )
        }
        semantic = SemanticIndex(
            tuple(chunks),
            tuple(
                (
                    fresh_vectors[chunk.identifier]
                    if chunk.identifier in fresh_vectors
                    else cached_vectors[chunk.identifier][1]
                )
                for chunk in chunks
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
            lexical = build_lexical_index(
                chunks,
                texts=[_retrieval_text(chunk, summaries) for chunk in chunks],
            )
        indexes = prepare_persistent_indexes(
            semantic,
            lexical,
            code_graph,
            embedding_model_identity=model_identity,
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
        self._current_source_manifest = current_manifest
        return chunks, indexes, "rebuilt", changes

    def _empty_retrieval(
        self,
        changes: dict[str, object],
        manifest: dict[str, object],
    ) -> tuple[list[Chunk], PersistentIndexes, str, dict[str, object]]:
        """Represent an empty source collection without persisting an index.

        Parameters
        ----------
        changes: dict[str, object]
            Source and configuration changes detected by the current scan.
        manifest: dict[str, object]
            Current source inventory, retained in memory for tool coverage.

        Returns
        -------
        tuple[list[Chunk], PersistentIndexes, str, dict[str, object]]
            Empty chunks, an in-memory retrieval container, the status
            ``empty``, and the supplied change report.

        Notes
        -----
        Persistence still requires real chunks. Old on-disk caches, when
        present, are left untouched and are not consulted for this session's
        evidence. Adding sources runs normal indexing on the next refresh.

        """

        self._current_source_manifest = manifest
        indexes = PersistentIndexes(
            fingerprint="",
            semantic=SemanticIndex((), ()),
            lexical=LexicalIndex((), (), {}),
            code_graph=CodeGraph((), ()),
        )
        self.trace.emit(
            "index.empty", component="rag",
            data={"chunk_count": 0, "changes": changes},
        )
        self._emit_progress(
            "index_ready", 85.0,
            "No source passages to index; ready without retrieval evidence.",
        )
        return [], indexes, "empty", changes

    def _retrieve(
        self: RAGSession,
        question: str,
        retrieval_count: int | None = None,
        *,
        trace_turn_id: str | None = None,
        status_callback: Callable[[str], None] | None = None,
        scope: Sequence[Path] = (),
        prefer_source_code: bool = False,
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
        status_callback: Callable[[str], None] | None, optional
            Receives a short message before and after each search step, for
            progress display. Defaults to ``None``.
        scope: Sequence[pathlib.Path], optional
            Documents the question is limited to, from ``document_scope``.
            When it holds exactly one path, only that document's passages
            are kept (if it has any); otherwise every document is searched.
            Defaults to ``()``.
        prefer_source_code: bool, optional
            Let each channel rank ``SOURCE_SEARCH_DEPTH`` times deeper and
            keep its best source-code results (``prefer_source_results``),
            then rank documentation and notebook passages after source code
            before keeping the top results, unless the question asks about
            documentation (``demote_documentation``). Defaults to ``False``.

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

        A one-document scope ranks every chunk again (see
        ``_rank_everything``) so that document's best passages are found even
        when they are outside the project-wide top results.

        """

        def report(message: str) -> None:
            """Pass one retrieval-progress message to the configured callback.

            Parameters
            ----------
            message: str
                Text describing the retrieval step. Nothing is sent when the
                session has no status callback.

            """

            if status_callback is not None:
                status_callback(message)

        if not self.chunks:
            report("No indexed source passages; skipping retrieval")
            return ()
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
        # When the intent prefers source code, each channel ranks deeper and
        # keeps its best source-code results (``prefer_source_results``).
        depth = count * SOURCE_SEARCH_DEPTH if prefer_source_code else count
        depth = max(depth, count)

        def keep(results: list) -> list:
            """Keep a channel's results for fusion.

            Parameters
            ----------
            results: list
                One channel's results, ranked to ``depth``.

            Returns
            -------
            list
                Its best source-code results and overall best few when the
                intent prefers source code; otherwise the results.

            """

            if depth == count:
                return results
            return prefer_source_results(results, question, count)

        with trace.span(
            "retrieval.semantic",
            component="retrieval",
            turn_id=trace_turn_id,
        ):
            report("Embedding the query and running semantic search")
            semantic = keep(
                semantic_search(
                    self.indexes.semantic,
                    query_text,
                    self.embedding_session,
                    count=depth,
                    trace_turn_id=trace_turn_id,
                )
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
            lexical = keep(
                lexical_search(
                    self.indexes.lexical,
                    question,
                    count=depth,
                )
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
            if not self.indexes.code_graph.entities:
                code = []
                report(
                    "No source relationships indexed; skipping graph search"
                )
            elif not names_code(question, self.indexes.code_graph):
                # The graph channel follows named code; matching the words
                # of a prose question to identifiers only adds noise.
                code = []
                report("No code named in the question; skipping graph search")
            else:
                report("Searching relationships between source entities")
                code = keep(
                    code_search(
                        self.indexes.code_graph,
                        self.chunks,
                        question,
                        count=depth,
                    )
                )
                report(
                    "Relationship search ranked "
                    f"{len(code)} results"
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
        report("Combining and ranking search results")
        # Fuse every candidate, then rank test code after source code (unless
        # the question is about tests) and, when the intent prefers source
        # code, documentation before tests, before keeping the top results.
        candidates = hybrid_search(
            semantic=semantic,
            lexical=lexical,
            code=code,
            count=max(count, len(semantic) + len(lexical) + len(code)),
        )
        if prefer_source_code:
            candidates = demote_documentation(candidates, question)
        combined = demote_test_results(candidates, question)[:count]
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

        # A question about one document ("in this paper", "in smith2020")
        # ranks that document's passages only.
        named = list(scope)
        if len(named) == 1:
            focused = [
                result
                for result in self._rank_everything(question, trace_turn_id)
                if result.chunk.path == named[0]
            ][:count]
            if focused:
                combined = focused
                report(f"Kept the passages of {named[0].name}")
                trace.emit(
                    "retrieval.named_file.focused",
                    component="retrieval",
                    data={"path": str(named[0]), "count": len(focused)},
                    turn_id=trace_turn_id,
                )

        # Every fused candidate, not only the kept top results, may lead to
        # code: retrieved documentation, module headers, and tests.
        plan = plan_context_expansion(
            question, self.indexes.code_graph, candidates
        )
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
        # Related passages compete for a fixed number of places, by the
        # similarity of their embedding to the query's.
        relevance = (
            self._passage_relevance(query_text)
            if plan.related_functions
            else None
        )
        expanded = expand_retrieval_context(
            combined,
            self.chunks,
            self.indexes.code_graph,
            plan,
            relevance,
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

    def _run_tool(
        self: RAGSession,
        use: ToolUse,
        question: str,
        decision: RouteDecision,
        *,
        trace_turn_id: str | None,
        report_phase: Callable[..., None],
        confirm_callback: Callable[[ToolPlan], str] | None = None,
        pdf_writer: Callable[..., int] | None = None,
        choice_callback: (
            Callable[[str, tuple[str, ...]], int | None] | None
        ) = None,
        token_callback: Callable[[str], None] | None = None,
        answer_guidance: str | None = None,
    ) -> ToolResult:
        """Run an intent's tool with this session's services.

        Parameters
        ----------
        use: ToolUse
            The intent's tool and settings.
        question: str
            The question.
        decision: RouteDecision
            The checked routing decision; a question that points back also
            gives the tool its related turns' questions as user words.
        trace_turn_id: str | None
            Turn identifier for trace events.
        report_phase: Callable[..., None]
            Progress reporter; tool steps become ``processing`` phases and
            the tool's result an ``evidence_ready`` phase.
        confirm_callback: Callable[[ToolPlan], str] | None, optional
            Asked to confirm an expensive plan. Defaults to ``None``.
        pdf_writer: Callable[..., int] | None, optional
            Writes reports as PDF. Defaults to ``None``.
        choice_callback: Callable[..., int | None] | None, optional
            ``choice_callback(question, options)`` asks the user to pick one
            option (for example which paper to summarize). Defaults to
            ``None``.
        token_callback: Callable[[str], None] | None, optional
            Receives an answer the tool writes with the model, as it is
            generated. Defaults to ``None``.
        answer_guidance: str | None, optional
            Answer guidance for model-written answers. Defaults to ``None``.

        Returns
        -------
        ToolResult
            The tool's answer, evidence fields, and summary.

        """

        trace = getattr(self, "trace", NullTraceLogger())

        def ask(chat: object) -> Callable[[str, str, int], str]:
            """Adapt a chat session to the tools' side-question callback.

            Parameters
            ----------
            chat: object
                Chat session used to answer an isolated side question.

            Returns
            -------
            Callable[[str, str, int], str]
                Callback accepting a system instruction, prompt, and token
                budget. Calls delegate to ``side_reply`` on the supplied
                chat session.

            """

            return lambda system, prompt, tokens: side_reply(
                chat, system, prompt, max_new_tokens=tokens
            )

        # ``ask_model`` answers silently; ``write`` streams a tool's final
        # answer to the caller through ``token_callback``.
        context = ToolContext(
            project_root=Path(self.project.root),
            chunks=self.chunks,
            graph=self.indexes.code_graph,
            rank=lambda text: self._rank_everything(text, trace_turn_id),
            retrieval_count=self.project.profile.retrieval_count,
            unsearchable=self._unsearchable_sources,
            source_files=self._source_files,
            ask_router=self._ask_router,
            ask_model=ask(self.chat_session),
            write=lambda system, prompt, tokens: side_reply(
                self.chat_session,
                system,
                prompt,
                max_new_tokens=tokens,
                token_callback=token_callback,
            ),
            model_id=str(getattr(self.chat_session, "model_id", "")),
            progress=lambda message, **details: report_phase(
                ANSWER_PHASE_PROCESSING, message=message, **details
            ),
            done=lambda message, **details: report_phase(
                ANSWER_PHASE_EVIDENCE_READY,
                message=message,
                tool=use.name,
                **details,
            ),
            confirm=confirm_callback,
            trace=trace,
            trace_turn_id=trace_turn_id,
            write_pdf=pdf_writer,
            choose=choice_callback,
        )

        request = ToolRequest(
            question=question,
            text=answering_search_text(question, self.turns, decision),
            guidance=answer_guidance,
            earlier=self._earlier_turns(decision),
        )
        return run_tool(use, request, context)

    def _earlier_turns(
        self: RAGSession, decision: RouteDecision
    ) -> tuple[EarlierTurn, ...]:
        """Return the turns a question points back to, newest first.

        The router's related turns are followed through the turns they
        were related to in turn, so "turn this into a briefing" after
        "summarize that" still reaches "walk me through Smith's paper". A
        question that points back without related turns uses the last
        turn. The last turn is always included (``linked=False`` when the
        router did not relate it), because "that paper" usually means the
        paper just discussed even when the router relates an older turn on
        the same topic.

        Parameters
        ----------
        decision: RouteDecision
            The checked routing decision for the new question.

        Returns
        -------
        tuple[EarlierTurn, ...]
            At most ``_MAX_EARLIER_TURNS`` turns with their questions,
            answers, and evidence; empty when the question points back to
            nothing.

        """

        pending = list(decision.related_turn_ids)
        if not pending and decision.refers_back and self.turns:
            pending = [len(self.turns)]

        found: set[int] = set()
        while pending:
            number = pending.pop()
            if number in found or not 1 <= number <= len(self.turns):
                continue
            found.add(number)
            related = self.turns[number - 1].parameters.get("related_turn_ids")
            if isinstance(related, list):
                pending.extend(
                    related_number
                    for related_number in related
                    if isinstance(related_number, int)
                )

        linked = set(found)
        if found:
            found.add(len(self.turns))

        earlier: list[EarlierTurn] = []
        for number in sorted(found, reverse=True)[:_MAX_EARLIER_TURNS]:
            turn = self.turns[number - 1]
            earlier.append(
                EarlierTurn(
                    number=number,
                    question=turn.question,
                    answer=turn.answer,
                    paths=tuple(Path(item.path) for item in turn.evidence),
                    texts=tuple(item.text for item in turn.evidence),
                    lines=tuple(item.start_line for item in turn.evidence),
                    linked=number in linked,
                )
            )
        return tuple(earlier)

    def _source_files(self: RAGSession) -> dict[Path, int]:
        """Return selected source files and sizes from the source manifest.

        Prefer the current in-memory scan, including an empty collection,
        over an older saved manifest. A missing or unreadable manifest gives
        an empty mapping; tools use this for inventories and coverage reports.

        Returns
        -------
        dict[pathlib.Path, int]
            Manifest source paths mapped to their file sizes in bytes, or an
            empty mapping if the manifest is unavailable or unreadable.

        """

        manifest = self.project.indexes / "source_manifest.json"
        try:
            current = getattr(self, "_current_source_manifest", None)
            if current is None:
                current = json.loads(manifest.read_text(encoding="utf-8"))
            files = current["files"]
            return {
                Path(name): int(entry.get("size_bytes", 0))
                for name, entry in files.items()
            }
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return {}

    def _unsearchable_sources(
        self: RAGSession, indexed: list[Path]
    ) -> list[Path]:
        """Return selected source files that are not among ``indexed``.

        Parameters
        ----------
        indexed: list[pathlib.Path]
            Paths that have chunks in the index.

        Returns
        -------
        list[pathlib.Path]
            Selected files without indexed text, such as scanned PDFs, in
            manifest order.

        """

        known = set(indexed)
        return [path for path in self._source_files() if path not in known]

    def _rank_everything(
        self: RAGSession, text: str, trace_turn_id: str | None = None
    ) -> list[HybridResult]:
        """Rank every chunk for a text with the fused search channels.

        Parameters
        ----------
        text: str
            Text to rank chunks against.
        trace_turn_id: str | None, optional
            Turn identifier for trace events. Defaults to ``None``.

        Returns
        -------
        list[HybridResult]
            All chunks, best first, fused from semantic, lexical, and (when
            the graph has entities) code-graph rankings.

        """

        if not self.chunks:
            return []
        everything = len(self.chunks)
        semantic = semantic_search(
            self.indexes.semantic,
            self._format_query(text),
            self.embedding_session,
            count=everything,
            trace_turn_id=trace_turn_id,
        )
        lexical = lexical_search(self.indexes.lexical, text, count=everything)
        code = (
            code_search(
                self.indexes.code_graph,
                self.chunks,
                text,
                count=everything,
            )
            if self.indexes.code_graph.entities
            else []
        )
        return hybrid_search(
            semantic=semantic, lexical=lexical, code=code, count=everything
        )

    def _read_document(
        self: RAGSession,
        question: str,
        max_tokens: int,
        retrieval_count: int | None = None,
        *,
        trace_turn_id: str | None = None,
    ) -> DocumentSelection | None:
        """Choose one document for a question and select its chunks.

        Parameters
        ----------
        question: str
            Search text: the question, with earlier questions it refers to.
        max_tokens: int
            Evidence budget in estimated tokens.
        retrieval_count: int | None, optional
            How many leading search results decide the dominant file when
            the question names none; None uses the project profile.
            Defaults to ``None``.
        trace_turn_id: str | None, optional
            Turn identifier for trace events. Defaults to ``None``.

        Returns
        -------
        DocumentSelection | None
            The document read and its selected chunks, or None when the
            index is empty.

        Notes
        -----
        Every chunk is ranked (semantic, lexical, and code graph, fused),
        so a document longer than the budget keeps its parts most relevant
        to the question. See ``kilin.retrievals.document_reading``.

        """

        trace = getattr(self, "trace", NullTraceLogger())
        everything = len(self.chunks)
        if not everything:
            return None
        with trace.span(
            "retrieval.document",
            component="retrieval",
            data={"max_tokens": max_tokens},
            turn_id=trace_turn_id,
        ):
            ranked = self._rank_everything(question, trace_turn_id)
            selection = read_document(
                question,
                self.chunks,
                self.indexes.code_graph,
                ranked,
                max_tokens=max_tokens,
                top=retrieval_count or self.project.profile.retrieval_count,
            )
        trace.emit(
            "retrieval.document.selected",
            component="retrieval",
            data=(
                {"path": None}
                if selection is None
                else _document_data(selection)
            ),
            turn_id=trace_turn_id,
        )
        return selection

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
        if profile.query_format == QUERY_FORMAT_PLAIN:
            return question
        instruction = project_type(profile.type_id).retrieval_instruction
        return format_qwen3_query(question, instruction)

    def _system_prompt(self: RAGSession) -> str:
        """Return the project type's base system prompt.

        Each answer adds its intent's instruction to this prompt.

        Returns
        -------
        str
            Base instruction configured for the current project's type.

        """

        return project_type(self.project.profile.type_id).system_prompt

    def _route(
        self: RAGSession,
        question: str,
        config: object,
        use_router: bool,
        turn_id: str,
        report_phase: Callable[..., None],
    ) -> RouteDecision:
        """Classify a question's intent and trace the resulting decision.

        When routing is disabled, select the project type's default intent
        without loading or invoking the router model.

        Parameters
        ----------
        question: str
            User question to classify.
        config: object
            Loaded project type with its intents and default intent.
        use_router: bool
            Whether to call the router instead of selecting the default intent.
        turn_id: str
            Correlation identifier for trace events generated by this turn.
        report_phase: Callable[..., None]
            Callback notifying the frontend when intent classification starts.

        Returns
        -------
        RouteDecision
            Chosen intent, classification, fallback, and history metadata.

        """

        trace = getattr(self, "trace", NullTraceLogger())
        started = perf_counter()
        if not use_router:
            decision = RouteDecision(
                intent=config.default_intent,
                standalone=not self.turns,
                fallback="routing off",
            )
        else:
            report_phase(ANSWER_PHASE_ROUTING, message="Identifying intent")
            trace.emit(
                "intent.input.prepared",
                component="rag",
                data={
                    "machine_state": MACHINE_STATE_ANSWERING,
                    "project_type_id": config.id,
                    "intents": list(config.intents),
                    "turn_count": len(self.turns),
                    "router_model": self.router_model,
                },
                turn_id=turn_id,
            )
            with trace.span(
                "intent.classification",
                component="rag",
                data={"router_model": self.router_model},
                turn_id=turn_id,
            ):
                decision = route_question(
                    self._router_chat(), config, self.turns, question
                )
        intent = config.intents[decision.intent]
        data: dict[str, object] = {
            "classification": {
                "intent": decision.intent,
                "standalone": decision.standalone,
                "related_turn_ids": list(decision.related_turn_ids),
                "refers_back": decision.refers_back,
            },
            "dropped_turns": [
                {"turn_id": number, "reason": reason}
                for number, reason in decision.dropped_turns
            ],
            "resolved": {
                "history_policy": intent.history_policy,
                "retrieval_mode": intent.retrieval_mode,
                "retrieval_strategy": intent.retrieval_strategy,
            },
            "fallback": decision.fallback,
            "shortcut": decision.shortcut,
            "duration_ms": round((perf_counter() - started) * 1000, 3),
        }
        if trace.include_content:
            data["raw"] = decision.raw
        trace.emit(
            "intent.classification.decided",
            component="rag",
            data=data,
            turn_id=turn_id,
        )
        return decision

    def _model_sessions(self: RAGSession) -> list[ChatSession]:
        """The answer model's session and, when separate, the router's.

        Returns
        -------
        list[ChatSession]
            Loaded answer and distinct router chat sessions that expose token
            counters.

        """

        sessions = [self.chat_session]
        router = getattr(self, "_router_session", None)
        if router is not None and router is not self.chat_session:
            sessions.append(router)
        return [
            chat for chat in sessions if hasattr(chat, "input_token_total")
        ]

    def _router_chat(self: RAGSession) -> ChatSession:
        """Return the router model's session, loading it once if needed.

        Returns
        -------
        ChatSession
            The existing answering session if it also routes, otherwise the
            lazily loaded routing session.

        """

        if self.chat_session.model_id == self.router_model:
            return self.chat_session

        if self._router_session is None:
            self._router_session = self._new_chat_session(self.router_model)
        return self._router_session

    def _ask_router(
        self: RAGSession, system: str, prompt: str, max_new_tokens: int
    ) -> str:
        """Ask the router model a side question, in the tools' call form.

        Tools and ``document_scope`` call ``ask(system, prompt, tokens)``
        with the token limit as a positional argument. The router is
        looked up on each call, so it is loaded only when a tool asks.

        Parameters
        ----------
        system: str
            System instruction for the side question only.
        prompt: str
            The side question.
        max_new_tokens: int
            Maximum length of the reply.

        Returns
        -------
        str
            The router model's reply.

        """

        return self.router_reply(system, prompt, max_new_tokens=max_new_tokens)

    def _new_chat_session(
        self: RAGSession, model_id: str, **settings: object
    ) -> ChatSession:
        """Load a chat model with this project's system prompt and trace.

        Parameters
        ----------
        model_id: str
            Hugging Face model ID or local model directory.
        **settings: object
            Optional ``thinking``, ``temperature``, and ``max_new_tokens``
            passed to ``ChatSession``.

        Returns
        -------
        ChatSession
            The loaded chat session, without conversation history.

        """

        return ChatSession(
            model_id,
            backend=self.backend,
            system_prompt=self._system_prompt(),
            allow_download=self.project.profile.allow_download,
            trace=self.trace,
            block_unrequested_cjk=(
                self.project.profile.block_unrequested_cjk
            ),
            **settings,
        )

    def _passage_relevance(
        self: RAGSession, query_text: str
    ) -> Callable[[Chunk], float]:
        """Return a scorer of indexed passages by similarity to a query.

        Parameters
        ----------
        query_text: str
            Query formatted for the embedding model.

        Returns
        -------
        Callable[[Chunk], float]
            Cosine similarity between a passage's saved embedding and the
            query's (both vectors are unit length); minus infinity for a
            passage without an embedding.

        """

        query = self.embedding_session.embed_query(query_text)
        vectors = dict(
            zip(
                (chunk.identifier for chunk in self.indexes.semantic.chunks),
                self.indexes.semantic.embeddings,
                strict=True,
            )
        )

        def score(chunk: Chunk) -> float:
            """Score one passage against the query.

            Parameters
            ----------
            chunk: Chunk
                Indexed passage.

            Returns
            -------
            float
                Cosine similarity, or minus infinity without an embedding.

            """

            vector = vectors.get(chunk.identifier)
            if vector is None:
                return float("-inf")
            return float(sum(a * b for a, b in zip(query, vector)))

        return score

    def _index_metadata(self: RAGSession) -> dict[str, object]:
        """Return settings that determine reusable index compatibility.

        Returns
        -------
        dict[str, object]
            Settings that determine reusable index compatibility.

        Notes
        -----
        These fields identify configured embedding and chunking choices. The
        retrieval preparation step additionally records the loaded model's
        identity and source. Implementation/library versions are not tracked;
        a selected-source manifest is compared separately.

        """

        profile = self.project.profile
        metadata: dict[str, object] = {
            "project_type_id": profile.type_id,
            "backend": self.backend,
            "embedding_model": profile.embedding_model(self.backend),
            "query_format": profile.query_format,
            "extensions": list(profile.extensions),
            "max_file_bytes": profile.max_file_bytes,
            "target_tokens": profile.target_tokens,
            "max_tokens": profile.max_tokens,
            "overlap_tokens": profile.overlap_tokens,
        }
        # Recorded only when off, so existing indexes stay valid by default.
        if project_type(profile.type_id).code_graph is False:
            metadata["code_graph"] = False
        # Indexes built from plain chunk text are rebuilt once.
        metadata["retrieval_text"] = RETRIEVAL_TEXT_FORMAT
        # Projects that read PDFs re-index when the PDF text extractor
        # changes; other projects are unaffected.
        if ".pdf" in profile.extensions:
            metadata["pdf_text_extractor"] = PDF_TEXT_EXTRACTOR
        return metadata

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
            "project_type_id": profile.type_id,
            "source": str(self.project.source),
            "backend_preference": profile.backend,
            "allow_download": profile.allow_download,
            "chat_model": profile.chat_model(self.backend),
            "retrieval_count": retrieval_count or profile.retrieval_count,
            "thinking": self.chat_session.thinking,
            "temperature": self.chat_session.temperature,
            "max_new_tokens": self.chat_session.max_new_tokens,
            "system_prompt": self._system_prompt(),
        }

    def _load_conversation(
        self: RAGSession,
        *,
        path: Path | None = None,
    ) -> tuple[list[dict[str, str]], list[ConversationTurn], str | None]:
        """Load saved chat messages and human-readable turns.

        Parameters
        ----------
        path: pathlib.Path | None, optional
            Read a staged target conversation without first switching the
            active path. None reads the current conversation.

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
        Reads only the canonical grouped conversation record. The caller
        holds file ownership while restoring or staging a switch. This does
        not validate that saved turns, evidence, profile, and indexes are
        mutually consistent.

        """

        path = self.conversation_path if path is None else path
        if not path.is_file():
            return [], [], None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            conversation = conversation_state(payload)
            turns = [
                _turn_from_data(item) for item in conversation.get("turns", [])
            ]
            system_prompt = str(
                conversation.get("system_prompt", _SYSTEM_PROMPT)
            )
            messages = [{"role": SYSTEM_ROLE, "content": system_prompt}]
            # Earlier evidence stays with its turn; history is Q/A only.
            messages.extend(
                history_messages(turns, list(range(1, len(turns) + 1)))
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
        pending_turn: ConversationTurn | None = None,
        trace_turn_id: str | None = None,
    ) -> None:
        """Save exact chat state and inspectable interaction records.

        Parameters
        ----------
        pending_turn: ConversationTurn | None, optional
            New turn to save with the already committed conversation. It is
            appended to the in-memory list only after the checkpoint write
            succeeds. ``None`` saves the existing state unchanged.
        trace_turn_id: str | None, optional
            Optional identifier associating trace events with one conversation
            turn. Defaults to ``None``.

        Notes
        -----
        Builds a hash-identified resumable snapshot. With an enabled trace
        sink, state is written through ``update_conversation`` to the same file
        as ``conversation_path``; otherwise it is written directly to that
        path under an exclusive file lease. A failed write leaves the committed
        turn list and checkpoint ID unchanged. The hash labels content but does
        not store a separate immutable snapshot for each checkpoint.

        """

        turns = self.turns
        if pending_turn is not None:
            turns = [*turns, pending_turn]
        conversation = {
            "system_prompt": self._system_prompt(),
            "turns": [_turn_to_data(turn) for turn in turns],
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
            "turn_count": len(turns),
        }
        checkpoint_id = _resume_checkpoint_id(checkpoint_snapshot)
        checkpoint_state = {
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "conversation_sha256": checkpoint_snapshot["conversation_sha256"],
            "index_fingerprint": self.indexes.fingerprint,
            "turn_count": len(turns),
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
            with conversation_write_ownership(
                path, ownership=self._active_conversation_lock()
            ):
                payload: dict[str, object] = {"sessions": {}}
                if path.is_file():
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(loaded, dict):
                        raise ValueError("The saved conversation is invalid.")
                    payload = loaded
                stored = conversation_state(payload, create=True)
                stored.update(saved_state)
                _write_json(path, payload)
        # The checkpoint write is the commit boundary. Diagnostic events
        # may fail afterward without undoing successfully persisted data.
        if pending_turn is not None:
            self.turns.append(pending_turn)
        self._checkpoint_id = checkpoint_id
        self.trace.emit(
            "state.checkpoint.committed",
            component="rag",
            data={
                "checkpoint_id": checkpoint_id,
                "parent_checkpoint_id": parent_checkpoint_id,
                "immutable_state_saved": True,
                "conversation_path": str(path),
                "turn_count": len(turns),
                "index_fingerprint": self.indexes.fingerprint,
            },
            turn_id=trace_turn_id,
        )

    def _active_conversation_lock(self) -> ConversationFileLock | None:
        """Return the owner of the current conversation, when available.

        Returns
        -------
        ConversationFileLock | None
            Trace logger's held file lock or the RAG session's direct-write
            lock. ``None`` is possible in minimal test fixtures.

        Notes
        -----
        This is an internal borrowing interface. Callers must not release
        the returned owner and must not retain it after switching sessions.

        """

        trace = getattr(self, "trace", None)
        if isinstance(trace, TraceLogger):
            return trace._file_lock
        return getattr(self, "_conversation_ownership", None)

    def _archive_conversation(self: RAGSession) -> Path | None:
        """Rename the active conversation log with a readable UTC time.

        Returns
        -------
        pathlib.Path | None
            Archive path after renaming an existing conversation, or None if
            there was no file to archive.

        Notes
        -----
        The configured conversation path is authoritative for both resumable
        state and any enabled RAG trace. Reset semantics for an active logger
        that references an archived path are handled separately by ``reset``.

        """

        path = self.conversation_path
        with conversation_write_ownership(
            path, ownership=self._active_conversation_lock()
        ):
            if not path.is_file():
                return None
            timestamp = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H-%M-%S-%fZ"
            )
            archived_path = path.with_name(
                f"conversation_{timestamp}.json"
            )
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
        None allocates a timestamped name beneath logs/traces. Filename
        selection is not a reservation; exclusive ownership must be acquired
        before the resulting record is read or modified.

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


def _conversation_trace(
    trace: TraceConfig | TraceSink | None,
    conversation_path: Path,
) -> tuple[TraceSink, bool]:
    """Create or validate the trace bound to one conversation record.

    Parameters
    ----------
    trace: TraceConfig | TraceSink | None
        Trace request for the RAG session. ``None`` selects an enabled trace
        with the packaged default mode. An enabled configuration without an
        output path is bound to the conversation record automatically.
    conversation_path: pathlib.Path
        Authoritative conversation JSON path used for resumable state.

    Returns
    -------
    tuple[TraceSink, bool]
        Trace sink and whether the caller owns its lifetime.

    Raises
    ------
    ValueError
        If an enabled configuration or existing sink targets a different path,
        or if ``trace`` is not supported by the trace factory.

    Notes
    -----
    The stricter one-file rule belongs to ``RAGSession`` rather than the
    generic trace logger. Disabled tracing remains independent of the
    conversation path.

    """

    if trace is None:
        trace = TraceConfig(
            enabled=True,
            output_path=conversation_path,
        )
    elif isinstance(trace, TraceConfig) and trace.enabled:
        if trace.output_path is None:
            trace = replace(trace, output_path=conversation_path)
        else:
            configured = Path(trace.output_path).expanduser().resolve()
            if configured != conversation_path:
                raise ValueError(
                    "Enabled RAG tracing must use the conversation path."
                )

    sink, owns_sink = create_trace_logger(
        trace,
        default_directory=conversation_path.parent,
    )
    _check_conversation_trace(sink, conversation_path)
    return sink, owns_sink


def _check_conversation_trace(
    trace: TraceSink,
    conversation_path: Path,
) -> None:
    """Require an enabled RAG trace to use the conversation record.

    Parameters
    ----------
    trace: TraceSink
        Existing trace sink to validate. Disabled sinks need no output path.
    conversation_path: pathlib.Path
        Authoritative conversation JSON path used for resumable state.

    Raises
    ------
    ValueError
        If an enabled sink has no output path or resolves to another file.

    """

    if not trace.enabled:
        return
    output_path = trace.output_path
    if output_path is None:
        raise ValueError(
            "Enabled RAG tracing requires an exact conversation path."
        )
    resolved = Path(output_path).expanduser().resolve()
    if resolved != conversation_path:
        raise ValueError("Enabled RAG tracing must use the conversation path.")
    if isinstance(trace, TraceLogger) and not trace._file_lock.holds(
        conversation_path
    ):
        raise RuntimeError("The conversation trace logger has no file lock.")


def _read_text_cache(indexes: Path) -> dict[str, dict[str, str]]:
    """Read cached extracted source text, ignoring invalid cache entries.

    Parameters
    ----------
    indexes: pathlib.Path
        Folder containing the source-text cache JSON file.

    Returns
    -------
    dict[str, dict[str, str]]
        Cached entries containing ``sha256`` and ``text`` strings, keyed by
        source path. Missing, unreadable, or invalid records are omitted.

    """

    try:
        data = json.loads((indexes / _TEXT_CACHE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {
        str(name): entry
        for name, entry in (data.items() if isinstance(data, dict) else ())
        if isinstance(entry, dict)
        and isinstance(entry.get("sha256"), str)
        and isinstance(entry.get("text"), str)
    }


def _write_text_cache(
    indexes: Path, entries: dict[str, dict[str, str]]
) -> None:
    """Save the current source-text cache with an atomic file replacement.

    Parameters
    ----------
    indexes: pathlib.Path
        Index folder in which the cache is stored; created if necessary.
    entries: dict[str, dict[str, str]]
        Current source paths and extracted text metadata. Deleted source
        files are absent from this mapping and so drop out of the cache.

    """

    indexes.mkdir(parents=True, exist_ok=True)
    path = indexes / _TEXT_CACHE
    atomic_write_text(path, json.dumps(entries))


def _document_evidence(selection: DocumentSelection) -> tuple[Evidence, ...]:
    """Number a document's selected chunks in source order.

    Parameters
    ----------
    selection: DocumentSelection
        Document selection whose chunks are included as answer evidence.

    Returns
    -------
    tuple[Evidence, ...]
        Evidence records retaining chunk provenance and citations starting
        from one, in the order supplied by the selection.

    """

    return tuple(
        Evidence(
            citation=number,
            path=chunk.path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            symbol=chunk.symbol,
            text=chunk.text,
            cell=chunk.cell,
        )
        for number, chunk in enumerate(selection.chunks, start=1)
    )


def _document_data(selection: DocumentSelection) -> dict[str, object]:
    """Describe a document selection for the turn parameters and trace.

    Parameters
    ----------
    selection: DocumentSelection
        The document read and its selected chunks.

    Returns
    -------
    dict[str, object]
        JSON-ready ``path``, ``functions``, ``reason``, ``total_chunks``,
        ``selected_chunks``, ``estimated_tokens``, and ``complete``.

    """

    return {
        "path": selection.path.as_posix(),
        "functions": list(selection.functions),
        "reason": selection.reason,
        "total_chunks": selection.total_chunks,
        "selected_chunks": len(selection.chunks),
        "estimated_tokens": selection.estimated_tokens,
        "complete": selection.complete,
    }


def _document_message(selection: DocumentSelection | None) -> str:
    """Summarize a document selection for the progress display.

    Parameters
    ----------
    selection: DocumentSelection | None
        Chosen document or ``None`` if no document was selected.

    Returns
    -------
    str
        Description of the selected file and how many chunks will be read,
        or a message explaining that no document was found.

    """

    if selection is None:
        return "No document found to read"
    name = selection.path.name
    if selection.functions:
        name = f"{', '.join(selection.functions)} in {name}"
    if selection.complete:
        return f"Reading all {selection.total_chunks} parts of {name}"
    return (
        f"Reading the {len(selection.chunks)} most relevant of "
        f"{selection.total_chunks} parts of {name}"
    )


def _add_related_evidence(
    fresh: tuple[Evidence, ...],
    turns: list[ConversationTurn],
) -> tuple[Evidence, ...]:
    """Append the evidence earlier related turns relied on.

    Parameters
    ----------
    fresh: tuple[Evidence, ...]
        Evidence just retrieved for the new question; it keeps its numbers.
    turns: list[ConversationTurn]
        Earlier turns the router marked as related, oldest first.

    Returns
    -------
    tuple[Evidence, ...]
        Fresh evidence followed by up to ``_MAX_RELATED_EVIDENCE`` related
        items, renumbered after it. Items a related answer cited come
        first, newest turn first; an answer without citations contributes
        its first three items. Chunks already present are skipped.

    """

    seen = {(str(item.path), item.start_line, item.end_line) for item in fresh}
    extra: list[Evidence] = []
    for turn in reversed(turns):
        cited = {
            int(number)
            for group in _CITATION_GROUP.findall(str(turn.answer))
            for number in _CITATION_SEPARATOR.split(group)
        }
        chosen = [item for item in turn.evidence if item.citation in cited]
        if not cited:
            chosen = list(turn.evidence[:_UNCITED_RELATED_EVIDENCE])
        for item in chosen:
            key = (str(item.path), item.start_line, item.end_line)
            if key not in seen and len(extra) < _MAX_RELATED_EVIDENCE:
                seen.add(key)
                extra.append(item)
    start = len(fresh) + 1
    return fresh + tuple(
        replace(item, citation=number)
        for number, item in enumerate(extra, start)
    )


def _merge_evidence(
    groups: list[tuple[Evidence, ...]],
) -> tuple[Evidence, ...]:
    """Combine earlier turns' evidence and renumber the citations.

    Each turn numbered its evidence from 1, so the combined list is
    renumbered in conversation order; a repeated chunk keeps its first
    position. At most ``_MAX_REUSED_EVIDENCE`` items are kept, dropping the
    oldest.

    Parameters
    ----------
    groups: list[tuple[Evidence, ...]]
        Evidence from earlier turns in conversation order. Each turn's
        citations start at one.

    Returns
    -------
    tuple[Evidence, ...]
        Deduplicated evidence renumbered from one after applying the limit.

    """

    seen: set[tuple[str, int, int]] = set()
    ordered: list[Evidence] = []
    for group in groups:
        for item in group:
            key = (str(item.path), item.start_line, item.end_line)
            if key not in seen:
                seen.add(key)
                ordered.append(item)
    kept = ordered[-_MAX_REUSED_EVIDENCE:]
    return tuple(
        replace(item, citation=number) for number, item in enumerate(kept, 1)
    )


def _answer_prompt(
    question: str,
    evidence: tuple[Evidence, ...],
    guidance: object = None,
) -> str:
    """Combine one question with numbered, traceable evidence.

    Parameters
    ----------
    question: str
        User question supplied to query formatting or evidence retrieval.
    evidence: tuple[Evidence, ...]
        Ordered retrieved chunks with citation numbers and source provenance.
    guidance: object, optional
        Extra answer instruction; only a nonempty string is appended.
        Defaults to ``None``.

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

    if not evidence:
        # Without evidence (for example small talk) the message is sent as
        # written; an instruction here was echoed back by the model.
        if isinstance(guidance, str) and guidance.strip():
            return f"{question}\n\n{guidance.strip()}"
        return question
    sections = [f"Question:\n{question}", "Evidence:"]
    for item in evidence:
        # Render evidence from either host with POSIX separators.
        location = (
            f"{PureWindowsPath(str(item.path)).as_posix()}:"
            f"{item.start_line}-{item.end_line}"
        )
        symbol = f" ({item.symbol})" if item.symbol else ""
        sections.append(f"[{item.citation}] {location}{symbol}\n{item.text}")
    sections.append(
        "Answer using the evidence above. Put [1], [2], and so on immediately "
        "after the claims they support. Do not repeat the source text or "
        "write a bibliography."
    )
    if isinstance(guidance, str) and guidance.strip():
        sections[-1] += " " + guidance.strip()
    # The question again, last, so its own length or format wins.
    sections.append(
        "Answer this question as asked, including any length or format it "
        f"requests:\n{question}"
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
            cell=(int(item["cell"]) if item.get("cell") is not None else None),
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

    atomic_write_text(
        path, json.dumps(data, indent=2, sort_keys=True) + "\n"
    )


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
    """Return ranked chunks, adding exact text when tracing is enabled.

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
        Ranked chunks, adding exact text when tracing is enabled.

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


def _module_summaries(chunks: Sequence[Chunk]) -> dict[Path, str]:
    """Return the first docstring line of each Python module.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        All indexed passages.

    Returns
    -------
    dict[pathlib.Path, str]
        For each file whose first passage is Python module context that
        starts with a docstring (after optional comment lines), the
        docstring's first nonblank line (at most
        ``_SUMMARY_MAX_CHARACTERS`` characters).

    """

    summaries: dict[Path, str] = {}
    for chunk in chunks:
        if (
            chunk.kind != "python_module_context"
            or chunk.start_line != 1
            or chunk.path in summaries
        ):
            continue
        # A long docstring can continue in the next passage, so the text is
        # matched rather than parsed.
        match = _MODULE_DOCSTRING.match(chunk.text)
        if match is None:
            continue
        body = chunk.text[match.end():].split(match.group(1), 1)[0]
        lines = [line.strip() for line in body.splitlines()]
        first = next((line for line in lines if line), "")
        if first:
            summaries[chunk.path] = first[:_SUMMARY_MAX_CHARACTERS]
    return summaries


def _retrieval_text(chunk: Chunk, summaries: dict[Path, str]) -> str:
    """Return the text indexed for one passage.

    Parameters
    ----------
    chunk: Chunk
        Indexed passage.
    summaries: dict[pathlib.Path, str]
        First docstring lines of Python modules (``_module_summaries``).

    Returns
    -------
    str
        For Python and MATLAB code, the file path, then the module summary
        when there is one, then the passage text; for other passages, the
        passage text.

    Notes
    -----
    Only searching sees this text: evidence, citations, and the answering
    model keep the passage text itself.

    """

    if not str(chunk.kind).startswith(("python_", "matlab_")):
        return chunk.text
    header = [chunk.path.as_posix()]
    summary = summaries.get(chunk.path)
    if summary and chunk.kind != "python_module_context":
        header.append(summary)
    return "\n".join([*header, "", chunk.text])
