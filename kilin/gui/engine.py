"""Run repository and model operations behind the desktop interface.

``ChatEngine`` converts frontend requests into synchronous KíLín operations
and emits structured status, progress, result, and error events. The module's
entry point reads newline-delimited JSON commands from standard input and
writes newline-delimited JSON events to standard output. Backend diagnostic
output is redirected into trace events so it does not corrupt that protocol.

A workspace is a managed directory containing copied sources, configuration,
indexes, and conversations. It is not an operating-system execution sandbox.
The engine can download a catalog model when explicitly requested and can
load models according to project settings. The process entry point enables
complete diagnostic tracing, which can retain conversation and source
content.

Commands are processed sequentially. Cancellation in the desktop window
terminates the engine process rather than interrupting a model call in place.

Key Features
------------

- **Workspace preparation**

    Preview sources, copy selected files, and create or resume configured
    projects.

- **Conversation management**

    Reuse loaded sessions when switching or branching stored conversations.

- **Reference previews**

    Locate evidence in current source text where possible, otherwise display
    saved excerpts.

- **Event protocol**

    Report operations through an injected event sink or the JSON-line process
    protocol.

Module Structure
----------------

- ``ChatEngine``

    Coordinate synchronous backend operations for the desktop frontend.

- ``main``

    Read one JSON command per line and emit one JSON event per line.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import kilin.gui.engine
>>> from pathlib import Path
>>> events = []
>>> engine = kilin.gui.engine.ChatEngine(
...     Path("/path/to/workspaces"), events.append,
... )
>>> engine.close()

This constructs the coordinator without loading a model. Use the desktop
interface or the documented engine methods to open a project.

"""

from __future__ import annotations

from configparser import ConfigParser, Error as ConfigError
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
import json
from pathlib import Path
import math
from queue import Queue
import re
import sys
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Callable, Final

from kilin._model_completion import GenerationCancelled
from kilin.atomic_writes import atomic_write_text
from kilin.gui._paths import default_workspace_directory
from kilin.gui.preferences import (
    default_appearance,
    default_backend,
    default_effort_levels,
    default_embedding_model,
    default_query_format,
    default_router_model,
    default_search_depth,
    required_models,
    validate_effort_levels,
    validate_model_id,
    validate_search_depth,
    write_defaults,
)
from kilin.gui._report_pdf import write_report_pdf
from kilin.gui.model_settings import (
    DEFAULT_LEVEL,
    LEVEL_COUNT,
    PROFILE_FILE,
    read_effort_levels,
    read_router_model,
    read_search_depth,
    save_project_settings,
    write_effort_levels,
    write_router_model,
    write_search_depth,
)
from kilin.gui.model_catalog import (
    MODEL_CHOICES,
    cached_model_roles,
    downloaded_models,
    installed_backends,
    model_choices,
    thinking_switches,
)
from kilin import RAGSession, prepare_sources_workspace
from kilin.workspace import (
    add_project_sources,
    default_workspace_path,
    find_source_changes,
    preview_project_sources,
    remove_workspace_sources,
    sync_project_sources,
)
from kilin.tool_settings import project_tool_settings, save_tool_settings
from kilin.workspace_resources import PROJECT_SOURCES_DIRECTORY
from kilin.project_files import (
    add_project_files,
    list_project_files,
    remove_project_files,
    replace_project_file,
)
from kilin.conversation_format import conversation_state
from kilin.conversation_ownership import ConversationFileLock
from kilin.conversations import ConversationInfo, ConversationStore
from kilin.huggingface import download_model
from kilin.project_type import load_project_types, project_type
from kilin.project import (
    PROJECT_TYPE_IDS,
    ProjectAlreadyExistsError,
    WorkspaceOccupiedError,
    QAProject,
    configure_project,
    default_project_profile,
    delete_project,
    open_project,
)
from kilin.repository import DEFAULT_EXTENSIONS, scan_repository
from kilin.repository_reason_codes import (
    SELECTION_REASON_CODES,
    SELECTION_REASON_EMPTY_FILE,
    SELECTION_REASON_FILE_TOO_LARGE,
)
from kilin.runtime import (
    BACKEND_AUTO,
    BACKEND_MLX,
    BACKEND_TORCH,
    choose_backend,
)
from kilin.sources import read_repository_files
from kilin.tools import (
    PLAN_CHOICE_CANCEL,
    PLAN_CHOICE_CHOICES,
    PlanChoice,
    ToolPlan,
)
from kilin.trace_logging import (
    FAILURE_POLICY_WARN,
    NullTraceLogger,
    TraceConfig,
    TraceLogger,
    TraceSink,
    TraceTextStream,
)

__all__ = [
    "ChatEngine",
    "main",
]

ROOT = Path(__file__).resolve().parents[2]
# The engine adds this phase to RAGSession's answer phases while it switches
# the chat model before a question; the page shows it as its own step.
ENGINE_PHASE_LOADING_MODEL: Final[str] = "loading_model"
# Settings the index records: changing any of them rebuilds the index.
_INDEXING_SETTINGS: Final[frozenset[str]] = frozenset(
    ("extensions", "max_file_bytes", "target_tokens", "max_tokens",
     "overlap_tokens")
)
EventSink = Callable[[dict[str, object]], None]


class _AnswerStream:
    """Batch generated text into a few frontend events per second.

    Parameters
    ----------
    send: Callable[..., None]
        Callback publishing accumulated text to the frontend.
    interval: float, optional
        Minimum interval between progress notifications in seconds.
        Defaults to ``0.08``.

    """

    def __init__(
        self, send: Callable[..., None], interval: float = 0.08
    ) -> None:
        """Initialize buffered answer text and its notification interval.

        Parameters
        ----------
        send: Callable[..., None]
            Callback used to emit streamed text events.
        interval: float, optional
            Minimum time in seconds between streamed updates. Defaults
            to 0.08.

        """

        self._send = send
        self._interval = interval
        self._pieces: list[str] = []
        self._last = 0.0

    def add(self, text: str) -> None:
        """Buffer new answer text and emit it after the time threshold.

        Parameters
        ----------
        text: str
            Newly generated assistant text to append to the stream.

        """

        self._pieces.append(text)
        if perf_counter() - self._last >= self._interval:
            self.flush()

    def flush(self) -> None:
        """Emit buffered answer text and update the last-flush time."""

        if self._pieces:
            self._send("answer_delta", text="".join(self._pieces))
            self._pieces.clear()
        self._last = perf_counter()


class ChatEngine:
    """Coordinate synchronous backend operations for the desktop frontend.

    Parameters
    ----------
    sandbox: pathlib.Path
        Parent directory for default workspaces and saved-project discovery.
        This name does not imply execution isolation.
    emit: EventSink
        Callable receiving one structured event dictionary at a time.
    trace_enabled: bool, optional
        Whether activated conversations use detailed tracing. The process entry
        point passes True even though the constructor default is False.
    read_command: Callable[[], dict | None] | None, optional
        Reads the next frontend command while an answer waits for the user
        to confirm a tool plan; None runs plans without asking.

    Attributes
    ----------
    session: RAGSession | None
        Currently loaded project/model coordinator, or None before
        activation/after close.
    conversation_store: ConversationStore | None
        Registry of named conversations for the current project.
    active_conversation: ConversationInfo | None
        Metadata for the active conversation.
    project_trace: TraceSink
        Active forensic or null sink.
    active_trace: TraceSink
        Property exposing the current project trace to redirected backend
        output.

    Methods
    -------

    **Startup and model configuration.**

    startup()
        Suggest the most recently used valid sandbox project.
    readiness()
        Report runtime packages and the preferred models' availability.
    use_installed_models()
        Use installed models in place of missing preferred ones.
    install_required_models()
        Download missing GUI models with live byte progress and timing.
    settings(type_id, workspace)
        Emit saved or default settings and cached model IDs.
    download(model_id)
        Download only a model explicitly chosen from the GUI catalog.
    apply_model_settings(request)
        Apply confirmed model edits and retain the active chat.
    save_answer_settings(settings)
        Save answer settings changed outside a question, e.g. Thinking.

    **Project discovery and sources.**

    list_saved_projects(workspace)
        Report valid projects of all registered types, newest first.
    find_saved_project(type_id, workspace)
        Refresh Resume for the selected type and workspace folder.
    resume_project(directory, conversation_id=None)
        Reopen a saved project and optionally restore its exact chat.
    preview(sources, extensions, max_file_bytes, workspace)
        List eligible files before copying any source.
    open_project(sources, type_id, workspace, ...)
        Create and activate a new, independently owned workspace.
    check_sources(sources)
        Tell the user what adding these sources would copy.
    add_project_sources(sources, decisions)
        Add confirmed sources and restore the active chat.
    remove_project_sources(paths)
        Delete managed source copies through the core and reindex.
    sync_sources(decisions)
        Apply the user's choice for each changed source, then reindex.
    edit_folder(folder, operation)
        Delegate editing outputs/ and templates/ to the core API.
    rename_project(name)
        Rename the active workspace folder, then reopen the same chat.
    delete_project(directory)
        Move a closed project into the deleted folder beside it.
    apply_indexing_settings(settings)
        Save the open project's indexing settings and rebuild its index.

    **Answering and evidence.**

    ask(question, retrieval_count=None, ...)
        Emit progress, saved answer, and evidence events.
    cancel_answer()
        Stop the running answer at its next generated token.
    save_answer_timeline(timestamp, timeline)
        Keep how long an answer took, as the page showed it.
    preview_reference(turn_index, citation)
        Show nearby source lines for one saved answer citation.

    **Tools.**

    tool_settings()
        Send the tools the open project can use, with their settings.
    save_tool_settings(name, settings)
        Save one tool's settings for the open project.

    **Conversations.**

    create_conversation(branch=False, ...)
        Create an empty conversation or branch the selected conversation.
    select_conversation(conversation_id)
        Switch conversations while retaining loaded project resources.
    rename_conversation(conversation_id, name)
        Rename one conversation without changing its saved state.
    delete_conversation(conversation_id)
        Archive one conversation and select a remaining conversation.
    reset()
        Create and activate a new empty conversation for the open project.

    **Lifecycle.**

    close()
        Release the loaded backend session.

    Notes
    -----
    Most public action methods catch operation failures and emit error events
    rather than raising them to the frontend. Errors in the event sink or
    error-reporting path can still propagate. Calls are sequential in the
    child-process protocol, not independently scheduled jobs.

    """

    def __init__(
        self,
        sandbox: Path,
        emit: EventSink,
        *,
        trace_enabled: bool = False,
        read_command: Callable[[], dict[str, object] | None] | None = None,
    ) -> None:
        """Keep one session and send structured updates to the window.

        Parameters
        ----------
        sandbox: pathlib.Path
            Default parent folder for local project workspaces.
        emit: EventSink
            Callback that forwards an event to the web page.
        trace_enabled: bool, optional
            Whether each opened project receives its own detailed trace.
            Defaults to ``False``.
        read_command: Callable[[], dict | None] | None, optional
            Returns the next frontend command, or None when input ended.
            Used only while a tool plan waits for the user's choice.
            Defaults to ``None`` (plans run without asking).

        Raises
        ------
        ValueError
            If trace_enabled is not an exact Boolean.

        """

        self.sandbox = sandbox
        self.emit = emit
        if type(trace_enabled) is not bool:
            raise ValueError("trace_enabled must be True or False.")
        self.trace_enabled = trace_enabled
        self.read_command = read_command
        # Set from the input thread by Stop; checked by every model call.
        self._answer_stop = Event()
        self._answering = False
        self.project_trace: TraceSink = NullTraceLogger()
        self.session: RAGSession | None = None
        self.conversation_store: ConversationStore | None = None
        self.active_conversation: ConversationInfo | None = None

    @property
    def active_trace(self) -> TraceSink:
        """Return the active project trace or a harmless no-op trace.

        Returns
        -------
        TraceSink
            The active project trace or a harmless no-op trace.

        """

        return self.project_trace

    def _start_project_trace(
        self,
        project_directory: Path,
        conversation: ConversationInfo | None = None,
    ) -> None:
        """Open the single trace file owned by one conversation.

        Parameters
        ----------
        project_directory: pathlib.Path
            Resolved managed-project directory associated with the conversation
            trace.
        conversation: ConversationInfo | None, optional
            Registered conversation metadata for the state to activate or
            trace. Defaults to ``None``.

        Raises
        ------
        RuntimeError
            If tracing is enabled without a conversation and opened store.
        ValueError
            If the stored conversation trace is invalid.

        Notes
        -----
        With tracing disabled, this selects a null sink. Otherwise it opens a
        logger using full diagnostic recording at the conversation state path
        and emits trace_ready.
        The path comes from the store, keeping GUI trace and resume state
        aligned.

        """

        if not self.trace_enabled:
            self.project_trace = NullTraceLogger()
            return
        if conversation is None or self.conversation_store is None:
            raise RuntimeError("Conversation store is not open.")
        trace_path = self.conversation_store.state_path(conversation)
        scope = "conversation"
        self.project_trace = TraceLogger(
            TraceConfig(
                enabled=True,
                output_path=trace_path,
                failure_policy=FAILURE_POLICY_WARN,
            ),
            default_directory=trace_path.parent,
        )
        self.project_trace.emit(
            f"{scope}.trace.started",
            component="gui_engine",
            data={
                "scope": scope,
                "entrypoint": "gui",
                "project_directory": str(project_directory),
                "conversation": (
                    asdict(conversation) if conversation is not None else None
                ),
            },
        )
        self._send(
            "trace_ready",
            trace_path=str(self.project_trace.readable_path),
            scope=scope,
            conversation_id=(
                conversation.conversation_id
                if conversation is not None
                else None
            ),
        )

    def _close_project_trace(self) -> None:
        """Finish the current project trace without closing the GUI trace."""

        if self.project_trace.enabled:
            self.project_trace.close()
        self.project_trace = NullTraceLogger()

    def _detach_project_trace(self) -> None:
        """Close an archived trace and release its file lock without writing.

        GUI conversation deletion archives a trace whose accepted events and
        resume state have already been written. Closing without a final event
        avoids rewriting the moved JSON and releases the exclusive file lock.

        """

        if isinstance(self.project_trace, TraceLogger):
            self.project_trace.close(emit_event=False)
        self.project_trace = NullTraceLogger()

    def _store_ownership(
        self, conversation_id: str
    ) -> ConversationFileLock | None:
        """Borrow the active trace or RAG file lease for store mutations.

        Parameters
        ----------
        conversation_id: str
            Conversation being modified by the store.

        Returns
        -------
        ConversationFileLock | None
            Held ownership of the currently active file, or None when the
            operation targets another conversation.

        Notes
        -----
        The store validates borrowed ownership and never releases it. This
        permits GUI metadata changes and archival while RAG or a trace logger
        holds the active conversation's exclusive lock. The trace itself can
        own the lease before RAG model initialization finishes.

        """

        if (
            self.active_conversation is None
            or conversation_id != self.active_conversation.conversation_id
        ):
            return None
        if isinstance(self.project_trace, TraceLogger):
            return self.project_trace._file_lock
        if self.session is not None:
            return self.session._active_conversation_lock()
        return None

    def _store_state_writer(
        self, conversation_id: str
    ) -> Callable[[Mapping[str, object]], None] | None:
        """Return the current trace writer for active GUI metadata.

        Parameters
        ----------
        conversation_id: str
            Registered conversation whose metadata will be changed.

        Returns
        -------
        Callable[[Mapping[str, object]], None] | None
            The active trace state writer, or None for a nonactive
            conversation or when tracing is disabled.

        Notes
        -----
        The store invokes this writer once, under the borrowed trace
        ownership token, instead of separately replacing the JSON.

        """

        if (
            self.active_conversation is not None
            and conversation_id == self.active_conversation.conversation_id
            and isinstance(self.active_trace, TraceLogger)
        ):
            return self.active_trace.update_conversation
        return None

    def _send(self, event: str, **data: object) -> None:
        """Pass one named event dictionary to the configured frontend sink.

        Parameters
        ----------
        event: str
            Frontend event name stored under the event key.
        **data: object
            Additional fields merged into the emitted frontend event.

        Notes
        -----
        The event dictionary is constructed synchronously. Sink exceptions are
        not suppressed.

        """

        self.emit({"event": event, **data})

    def _report_error(
        self,
        event_type: str,
        error: BaseException,
        message: str,
        **event_fields: object,
    ) -> None:
        """Trace one backend failure and show a short GUI message.

        Parameters
        ----------
        event_type: str
            Structured trace event name describing the failed engine operation.
        error: BaseException
            Caught backend exception to record before reporting the frontend
            error.
        message: str
            User-facing explanation included in the error event.
        **event_fields: object
            Optional operation-specific recovery information for the GUI.

        Notes
        -----
        Records the traceback through the active sink and sends an error event
        containing the readable trace path when available. A failure while
        tracing or emitting this error can propagate.

        """

        trace = self.active_trace
        trace.exception(
            event_type,
            error,
            component="gui_engine",
        )
        self._send(
            "error",
            message=message,
            trace_path=(
                str(trace.readable_path)
                if trace.readable_path is not None
                else None
            ),
            **event_fields,
        )

    def startup(self) -> None:
        """Suggest the most recently used valid sandbox project."""

        project = self._find_saved_project(None, "")
        if project is not None:
            folder, type_id = project
            self._send("startup", project=str(folder), type_id=type_id)
        else:
            self._send("startup", project=None)

    def readiness(self) -> None:
        """Report runtime packages and the preferred models' availability.

        Notes
        -----
        The required models are those for the saved preferred backend,
        resolved by platform policy only when the preference is ``auto``.
        An unsupported explicit backend is reported as unavailable without
        choosing a replacement backend. For each missing model, the closest
        installed model of the same role may be offered as a replacement;
        ``can_use_installed`` also requires a usable backend runtime.

        """

        try:
            try:
                backend = _setup_backend()
                unavailable_reason = ""
            except ValueError as error:
                # Preserve an explicit unsupported choice for a useful
                # readiness report; never substitute another backend.
                backend = default_backend()
                if backend not in (BACKEND_MLX, BACKEND_TORCH):
                    raise
                unavailable_reason = str(error)
            packages = installed_backends()
            environment_ready = (
                packages.get(backend, False) and not unavailable_reason
            )
            entries = downloaded_models()
            cached = {str(item["model_id"]) for item in entries}
            required = required_models(backend)
            missing = [
                model_id for model_id in required if model_id not in cached
            ]
            replacements = _installed_replacements(
                missing, cached_model_roles(entries), backend
            )
            self._send(
                "readiness",
                backend=backend,
                environment_ready=environment_ready,
                unavailable_reason=unavailable_reason,
                required_models=required,
                cached_models=sorted(cached),
                missing_models=missing,
                replacements=replacements,
                can_use_installed=environment_ready
                and bool(missing) and len(replacements) == len(missing),
                ready=environment_ready and not missing,
            )
        except Exception as error:
            self._report_error("gui.readiness.failed", error, str(error))

    def use_installed_models(self) -> None:
        """Use installed models in place of missing preferred ones.

        Notes
        -----
        Each missing model is replaced by the installed model of the same
        role (language or embedding) and preferred backend whose size is
        closest,
        in the effort levels, router, and embedding preferences. The
        change is saved in the app-wide preferences and reported with
        ``installed_models_used``; readiness is then checked again.

        """

        try:
            backend = _setup_backend()
            entries = downloaded_models()
            cached = {str(item["model_id"]) for item in entries}
            missing = [
                model_id
                for model_id in required_models(backend)
                if model_id not in cached
            ]
            chosen = _installed_replacements(
                missing, cached_model_roles(entries), backend
            )
            levels = [
                {**level, "model": chosen.get(level["model"], level["model"])}
                for level in default_effort_levels(backend)
            ]
            router = default_router_model(backend)
            embedding = default_embedding_model(backend)
            saved = write_defaults(
                {
                    "effort_levels": {backend: levels},
                    "router_models": {backend: chosen.get(router, router)},
                    "embedding_models": {
                        backend: chosen.get(embedding, embedding)
                    },
                }
            )
            self.active_trace.emit(
                "gui.preferences.installed_models_used",
                component="gui_engine",
                data={"replacements": chosen},
            )
            self._send(
                "installed_models_used",
                replacements=chosen,
                preferences=saved,
                **_preference_payload(),
            )
            self.readiness()
        except Exception as error:
            self._report_error(
                "gui.preferences.failed",
                error,
                f"Could not switch to installed models: {error}",
            )

    def install_required_models(self) -> None:
        """Download missing models for the selected setup backend.

        Notes
        -----
        An explicit backend choice is never replaced by ``auto``. If its
        runtime is unavailable, installation fails before downloading.

        """

        started = perf_counter()
        try:
            backend = _setup_backend()
            if not installed_backends().get(backend, False):
                raise RuntimeError(
                    f"The {backend.upper()} runtime is not installed. Restart "
                    "KíLín through "
                    f"`uv run kilin --gui --backend {backend}`."
                )
            cached = {str(item["model_id"]) for item in downloaded_models()}
            missing = [
                model_id
                for model_id in required_models(backend)
                if model_id not in cached
            ]
            total = len(missing)
            if not missing:
                self._send("installation_complete", elapsed_seconds=0.0)
                self.readiness()
                return
            self._send("installation_started", total=total, backend=backend)
            for index, model_id in enumerate(missing, start=1):
                model_seconds = self._download_with_progress(
                    model_id,
                    progress_event="installation_progress",
                    index=index,
                    total=total,
                    overall_started=started,
                )
                self._send(
                    "installation_model_complete",
                    index=index,
                    total=total,
                    model_id=model_id,
                    model_seconds=model_seconds,
                    elapsed_seconds=perf_counter() - started,
                )
            self._send(
                "installation_complete",
                elapsed_seconds=perf_counter() - started,
            )
            self.readiness()
        except Exception as error:
            self._report_error(
                "gui.installation.failed",
                error,
                f"Setup could not finish: {error}",
            )

    def _download_with_progress(
        self,
        model_id: str,
        *,
        progress_event: str,
        index: int = 1,
        total: int = 1,
        overall_started: float | None = None,
    ) -> float:
        """Download one model while emitting heartbeat and byte progress.

        Parameters
        ----------
        model_id: str
            Model identifier to download.
        progress_event: str
            Frontend event name for progress updates.
        index: int, optional
            One-based position of this model in the download sequence. Defaults
            to ``1``.
        total: int, optional
            Total number of models in the download sequence. Defaults to ``1``.
        overall_started: float | None, optional
            Monotonic start time for the full download operation. Defaults to
            ``None``.

        Returns
        -------
        float
            Elapsed time for the completed model download.

        """

        model_started = perf_counter()
        overall_started = (
            model_started if overall_started is None else overall_started
        )
        stopped = Event()
        progress_lock = Lock()
        completed = 0
        expected = 0

        def record_progress(current: int, byte_total: int) -> None:
            """Record monotonic download totals under the progress lock.

            Parameters
            ----------
            current: int
                Bytes transferred according to the latest callback.
            byte_total: int
                Current reported total download size in bytes.

            """

            nonlocal completed, expected
            with progress_lock:
                completed = max(completed, current)
                expected = max(expected, byte_total)

        def send_progress() -> None:
            """Send current download speed and progress to the frontend."""

            with progress_lock:
                current = completed
                byte_total = expected
            now = perf_counter()
            model_elapsed = max(0.0, now - model_started)
            rate = (
                current / model_elapsed if current and model_elapsed else 0.0
            )
            eta = (
                max(0.0, byte_total - current) / rate
                if byte_total > current and rate > 0
                else (0.0 if byte_total and current >= byte_total else None)
            )
            self._send(
                progress_event,
                index=index,
                total=total,
                model_id=model_id,
                elapsed_seconds=max(0.0, now - overall_started),
                model_elapsed_seconds=model_elapsed,
                completed_bytes=current,
                total_bytes=byte_total,
                bytes_per_second=rate,
                eta_seconds=eta,
            )

        def heartbeat() -> None:
            """Publish periodic download progress until the event is set."""

            while not stopped.wait(0.5):
                send_progress()

        send_progress()
        ticker = Thread(
            target=heartbeat,
            name="kilin-download-progress",
            daemon=True,
        )
        ticker.start()
        try:
            download_model(model_id, progress_callback=record_progress)
        finally:
            stopped.set()
            ticker.join(timeout=1.0)
        with progress_lock:
            if expected:
                completed = expected
        send_progress()
        return perf_counter() - model_started

    def list_saved_projects(self, workspace: str) -> None:
        """Report valid projects of all registered types, newest first.

        Parameters
        ----------
        workspace: str
            Workspace containing saved projects.

        """

        self._send(
            "projects",
            projects=[
                {
                    "project": str(folder),
                    "name": folder.name,
                    "type_id": type_id,
                    "updated_at": updated_at,
                }
                for updated_at, folder, type_id
                in self._saved_projects(workspace)
            ],
        )

    def find_saved_project(self, type_id: str, workspace: str) -> None:
        """Refresh Resume for the selected type and workspace folder.

        A chosen workspace is checked directly. Without one, the most recent
        valid sandbox project of the selected type is suggested.

        Parameters
        ----------
        type_id: str
            Requested registered project-type identifier. Discovery helpers
            also accept None to search all registered types.
        workspace: str
            Explicit workspace path string. An empty string requests discovery
            or default allocation under the engine workspace root.

        """

        project = self._find_saved_project(type_id or None, workspace)
        self._send(
            "saved_project",
            project=str(project[0]) if project is not None else None,
            type_id=type_id,
            workspace=workspace,
        )

    def _find_saved_project(
        self, type_id: str | None, workspace: str
    ) -> tuple[Path, str] | None:
        """Find a current project, ignoring deleted or incomplete folders.

        Parameters
        ----------
        type_id: str | None
            Requested registered project-type identifier. Discovery helpers
            also accept None to search all registered types.
        workspace: str
            Explicit workspace path string. An empty string requests discovery
            or default allocation under the engine workspace root.

        Returns
        -------
        tuple[pathlib.Path, str] | None
            Pair (workspace_path, project_type_id) for the most recently used
            valid candidate, or None.

        Notes
        -----
        Discovery inspects profile/source existence and profile/conversation
        modification times without loading a model. Invalid candidate profiles
        are skipped; the whole project schema is checked later when opening.

        """

        projects = self._saved_projects(workspace, type_id)
        if not projects:
            return None
        _, folder, saved_type_id = max(
            projects, key=lambda item: (item[0], str(item[1]))
        )
        return folder, saved_type_id

    def _saved_projects(
        self, workspace: str, type_id: str | None = None
    ) -> list[tuple[float, Path, str]]:
        """Return valid saved projects ordered from newest to oldest.

        Parameters
        ----------
        workspace: str
            Workspace containing saved projects.
        type_id: str | None, optional
            Project-type identifier to filter by. Defaults to ``None``.

        Returns
        -------
        list[tuple[float, Path, str]]
            valid saved projects ordered from newest to oldest.

        """

        if type_id is not None and type_id not in PROJECT_TYPE_IDS:
            return []
        if workspace.strip():
            folders = [Path(workspace).expanduser()]
        elif self.sandbox.is_dir():
            folders = list(self.sandbox.iterdir())
        else:
            return []

        projects: list[tuple[float, Path, str]] = []
        for folder in folders:
            if not folder.is_dir() or folder.is_symlink():
                continue
            profile_path = folder / "profile.ini"
            if not profile_path.is_file() or profile_path.is_symlink():
                continue
            parser = ConfigParser()
            try:
                parser.read(profile_path, encoding="utf-8")
                saved_type_id = parser.get("project", "type_id")
                source_text = parser.get("project", "source")
                if saved_type_id not in PROJECT_TYPE_IDS:
                    continue
                if type_id is not None and saved_type_id != type_id:
                    continue
                if not source_text.strip():
                    continue
                source = Path(source_text).expanduser()
                source_path = (
                    source if source.is_absolute() else folder / source
                )
                if not source_path.is_dir():
                    continue
                chat_records = tuple(
                    (folder / "logs" / "traces").glob("conversation_*.json")
                )
                last_chat = max(
                    (
                        record.stat().st_mtime
                        for record in chat_records
                        if record.is_file() and not record.is_symlink()
                    ),
                    default=0,
                )
                recent = max(profile_path.stat().st_mtime, last_chat)
                projects.append((recent, folder, saved_type_id))
            except (ConfigError, OSError, ValueError):
                continue
        return sorted(
            projects,
            key=lambda item: (item[0], str(item[1])),
            reverse=True,
        )

    def resume_project(
        self, directory: str, conversation_id: str | None = None
    ) -> None:
        """Reopen a saved project and optionally restore its exact chat.

        Parameters
        ----------
        directory: str
            Workspace containing an existing profile.ini.
        conversation_id: str | None, optional
            Saved conversation to restore. If omitted, select the most
            recently updated conversation. Defaults to ``None``.

        """

        previous_released = False
        try:
            project = open_project(directory)
            # Once closing starts, the previous UI cannot assume its
            # session remains usable if a subsequent load fails.
            previous_released = True
            self.close()
            conversation = self._open_conversation_store(
                project, conversation_id
            )
            self._start_project_trace(project.root, conversation)
            self._activate_project(project, conversation)
            self._report_source_changes(project)
        except Exception as error:
            self._report_error(
                "gui.project.resume.failed",
                error,
                f"Could not resume the project: {error}",
                resume_released=previous_released,
            )
            # An activation error can leave a partially opened session.
            # Release it rather than retaining a project the GUI cannot use.
            if previous_released:
                self.close()
            elif self.session is None:
                self._close_project_trace()

    def _report_source_changes(self, project: QAProject) -> None:
        """Tell the frontend which originals changed since their import.

        Parameters
        ----------
        project: QAProject
            Reopened project whose copies are compared with their originals.

        Notes
        -----
        Sends ``source_changes`` only when a file can be imported, updated
        or removed, so the user can choose per file. Copies the user edited
        are listed but always kept. A failed check is traced but not shown:
        the project is already open and usable.

        """

        try:
            changes = find_source_changes(project)
        except Exception as error:
            self.active_trace.exception(
                "gui.sources.changes.failed", error, component="gui_engine"
            )
            return
        self.active_trace.emit(
            "gui.sources.changes.found", component="gui_engine", data=changes
        )
        if changes["added"] or changes["changed"] or changes["deleted"]:
            self._send("source_changes", **changes)

    def settings(self, type_id: str, workspace: str) -> None:
        """Emit saved or default settings and cached model IDs.

        Parameters
        ----------
        type_id: str
            Registered project-type identifier.
        workspace: str
            Existing workspace path, or blank for new-project defaults.

        Notes
        -----
        Sends a ``settings`` event; no value is returned to the caller.

        """

        try:
            # Before the page knows the types, use the first one in the file.
            type_id = type_id or next(iter(load_project_types()))
            profile_path = Path(workspace).expanduser() / "profile.ini"
            profile = (
                open_project(profile_path.parent).profile
                if workspace.strip() and profile_path.is_file()
                else replace(
                    default_project_profile(type_id),
                    # Generation comes from the core project profile;
                    # the GUI selects only application-specific defaults.
                    chat_mlx=str(
                        default_effort_levels(BACKEND_MLX)[DEFAULT_LEVEL][
                            "model"
                        ]
                    ),
                    chat_torch=str(
                        default_effort_levels(BACKEND_TORCH)[DEFAULT_LEVEL][
                            "model"
                        ]
                    ),
                    embedding_mlx=default_embedding_model(BACKEND_MLX),
                    embedding_torch=default_embedding_model(BACKEND_TORCH),
                    query_format=default_query_format(),
                )
            )
            entries = downloaded_models()
            cached = {str(item["model_id"]) for item in entries}
            self._send(
                "settings",
                profile=asdict(profile),
                effort_levels=_effort_levels_by_backend(
                    profile_path.parent if profile_path.is_file() else None,
                    profile,
                ),
                default_effort_levels={
                    backend: default_effort_levels(backend)
                    for backend in (BACKEND_MLX, BACKEND_TORCH)
                },
                search_depth=(
                    read_search_depth(profile_path.parent)
                    if profile_path.is_file()
                    else default_search_depth()
                ),
                appearance=default_appearance(),
                # The backend chosen in KíLín setup for new projects.
                backend_preference=default_backend(),
                preferences=_preference_payload(),
                default_search_depth=default_search_depth(),
                router_models=_router_models_by_backend(
                    profile_path.parent if profile_path.is_file() else None
                ),
                default_embedding_models={
                    backend: default_embedding_model(backend)
                    for backend in (BACKEND_MLX, BACKEND_TORCH)
                },
                default_router_models={
                    backend: default_router_model(backend)
                    for backend in (BACKEND_MLX, BACKEND_TORCH)
                },
                project_types=[
                    {
                        "type_id": item.id,
                        "label": item.display_name,
                        "intents": {
                            intent_id: intent_id.replace("_", " ").capitalize()
                            for intent_id in item.intents
                        },
                        "welcome_prompts": list(item.welcome_prompts),
                        "suggestions": list(item.suggestions),
                    }
                    for item in load_project_types().values()
                ],
                backend=choose_backend(BACKEND_AUTO),
                installed_backends=installed_backends(),
                supported_extensions=list(DEFAULT_EXTENSIONS),
                selection_reason_codes=dict(SELECTION_REASON_CODES),
                cached_models=sorted(cached),
                model_roles=cached_model_roles(entries),
                thinking_switches=thinking_switches(entries),
                model_choices=model_choices(cached),
                workspace_files=(
                    list_project_files(open_project(profile_path.parent))
                    if profile_path.is_file()
                    else []
                ),
                workspace=(
                    str(profile_path.parent.resolve())
                    if profile_path.is_file()
                    else ""
                ),
            )
        except Exception as error:
            self._report_error(
                "gui.settings.failed",
                error,
                str(error),
            )

    def download(self, model_id: str) -> None:
        """Download only a model explicitly chosen from the GUI catalog.

        Download progress is shown as an ongoing operation. The existing Stop
        control can interrupt its child process if the user changes their mind.

        Parameters
        ----------
        model_id: str
            One of the model IDs offered in the GUI's model panel.

        Notes
        -----
        Custom Hugging Face IDs and existing local paths are accepted. This can
        contact Hugging Face and write to its cache. Success emits refreshed
        model status; download/validation failures become error events.

        """

        try:
            model_id = validate_model_id(model_id)
            self._send("model_download_started", model_id=model_id)
            elapsed = self._download_with_progress(
                model_id,
                progress_event="model_download_progress",
            )
            self._send(
                "model_downloaded",
                model_id=model_id,
                elapsed_seconds=elapsed,
                thinking_switches=thinking_switches(downloaded_models()),
            )
        except Exception as error:
            self._report_error(
                "gui.model_download.failed",
                error,
                f"Could not download {model_id}: {error}",
            )

    def apply_model_settings(self, request: dict[str, object]) -> None:
        """Apply confirmed model edits and retain the active chat.

        Parameters
        ----------
        request: dict[str, object]
            Request parameters supplied by the frontend.

        """

        root = None
        old_profile = None
        old_levels = None
        old_router = None
        old_depth = None
        old_backend = None
        conversation_id = None
        profile_text = None
        changed_profile = False
        changed_runtime = False
        levels = None
        try:
            if request.get("effort_levels") is not None:
                levels = validate_effort_levels(request["effort_levels"])
                # The middle level's model is the project's chat model.
                chat_model = str(levels[DEFAULT_LEVEL]["model"])
            else:
                chat_model = validate_model_id(request["chat_model"])
            depth = (
                validate_search_depth(request["search_depth"])
                if request.get("search_depth") is not None
                else None
            )
            embedding_model = validate_model_id(request["embedding_model"])
            router_model = (
                validate_model_id(request["router_model"])
                if request.get("router_model")
                else None
            )
            query_format = request.get("query_format", "qwen3")
            if query_format not in {"qwen3", "plain"}:
                raise ValueError("Choose a supported embedding query format.")
            ids = {
                chat_model,
                embedding_model,
                *(str(level["model"]) for level in levels or ()),
                *((router_model,) if router_model else ()),
            }
            entries = downloaded_models()
            cached = {str(item["model_id"]) for item in entries}
            missing = sorted(
                model
                for model in ids
                if model not in cached
                and not Path(model).expanduser().is_dir()
            )
            if missing and request.get("download") is not True:
                raise ValueError(
                    "Confirm the model downloads before applying."
                )
            if self.session is not None:
                root = self.session.project.root
                old_profile = self.session.project.profile
                old_backend = self.session.backend
                old_levels = read_effort_levels(root, self.session.backend)
                old_router = read_router_model(root, self.session.backend)
                profile_text = (root / "profile.ini").read_text(
                    encoding="utf-8"
                )
                conversation_id = self.active_conversation.conversation_id
            self.active_trace.emit(
                "gui.models.change.requested",
                component="gui_engine",
                data={
                    "before": asdict(old_profile) if old_profile else None,
                    "before_levels": old_levels,
                    "chat_model": chat_model,
                    "embedding_model": embedding_model,
                    "query_format": query_format,
                    "effort_levels": levels,
                    "downloads": missing,
                },
            )
            for model_id in missing:
                self.active_trace.emit(
                    "gui.model.download.started",
                    component="gui_engine",
                    data={"model_id": model_id},
                )
                self._send("model_download_started", model_id=model_id)
                elapsed = self._download_with_progress(
                    model_id, progress_event="model_download_progress"
                )
                self.active_trace.emit(
                    "gui.model.download.completed",
                    component="gui_engine",
                    data={"model_id": model_id},
                )
                self._send(
                    "model_downloaded",
                    model_id=model_id,
                    elapsed_seconds=elapsed,
                    thinking_switches=thinking_switches(downloaded_models()),
                )
            if root is None:
                # No project is open: these become the defaults new
                # projects start from (user preferences over package defaults).
                backend = request.get("backend")
                if backend not in (BACKEND_MLX, BACKEND_TORCH):
                    backend = _setup_backend()
                changes: dict[str, object] = {
                    "embedding_models": {backend: embedding_model},
                    "query_format": query_format,
                }
                if levels is not None:
                    changes["effort_levels"] = {backend: levels}
                if depth is not None:
                    changes["search_depth"] = depth
                if router_model is not None:
                    changes["router_models"] = {backend: router_model}
                write_defaults(changes)
            rebuilt = False
            if root is not None:
                backend = self.session.backend
                rebuilt = (
                    embedding_model != old_profile.embedding_model(backend)
                    or query_format != old_profile.query_format
                )
                settings = {
                    f"chat_{backend}": chat_model,
                    f"embedding_{backend}": embedding_model,
                    "query_format": query_format,
                }
                self._send(
                    "status",
                    message=(
                        "Rebuilding the embedding index…"
                        if rebuilt
                        else "Applying model settings…"
                    ),
                )
                # Explicitly confirm the change in the trace before releasing
                # the models; the replacement uses the same conversation file.
                self.active_trace.emit(
                    "gui.models.change.confirmed",
                    component="gui_engine",
                    data={**settings, "reindex_required": rebuilt},
                )
                # RAGSession switches chat models in place only on MLX;
                # other backends load the new model by reopening.
                reopen = rebuilt or (
                    chat_model != old_profile.chat_model(backend)
                    and backend != BACKEND_MLX
                )
                if not reopen and chat_model != old_profile.chat_model(
                    backend
                ):
                    self.session.use_chat_model(chat_model)
                    changed_runtime = True
                project = configure_project(
                    root, settings, trace=self.active_trace
                )
                changed_profile = True
                if levels is not None:
                    write_effort_levels(root, backend, levels)
                if depth is not None:
                    old_depth = read_search_depth(root)
                    write_search_depth(root, depth)
                if router_model is not None:
                    write_router_model(root, backend, router_model)
                if reopen:
                    self.close()
                    self._reopen_project_conversation(project, conversation_id)
                else:
                    self.session.project = project
                    if router_model is not None:
                        self.session.use_router_model(router_model)
                self.active_trace.emit(
                    "gui.models.change.applied",
                    component="gui_engine",
                    data={
                        **settings,
                        "effort_levels": levels,
                        "search_depth": depth,
                        "reindexed": rebuilt,
                    },
                )
            profile = (
                self.session.project.profile
                if self.session is not None
                else None
            )
            self._send(
                "models_applied",
                preferences=_preference_payload(),
                effort_levels=(
                    _effort_levels_by_backend(
                        self.session.project.root, profile
                    )
                    if profile is not None
                    else None
                ),
                router_models=(
                    _router_models_by_backend(self.session.project.root)
                    if profile is not None
                    else None
                ),
                search_depth=(
                    read_search_depth(self.session.project.root)
                    if profile is not None
                    else None
                ),
                profile=asdict(profile) if profile is not None else None,
                cached_models=sorted(cached | set(missing)),
                model_roles=cached_model_roles(downloaded_models()),
                thinking_switches=thinking_switches(downloaded_models()),
                reindexed=rebuilt,
            )
        except Exception as error:
            if root is not None and (changed_profile or changed_runtime):
                try:
                    (root / "profile.ini").write_text(
                        profile_text, encoding="utf-8"
                    )
                    write_effort_levels(root, old_backend, old_levels)
                    write_router_model(root, old_backend, old_router)
                    if old_depth is not None:
                        write_search_depth(root, old_depth)
                    self.close()
                    self._reopen_project_conversation(
                        open_project(root), conversation_id
                    )
                except Exception:
                    self._send(
                        "backend_failed",
                        message="Reload failed. Reopen the project to retry.",
                    )
            self._report_error(
                "gui.models.change.failed",
                error,
                f"Could not apply model settings: {error}",
            )

    def preview(
        self,
        sources: list[str],
        extensions: list[str],
        max_file_bytes: int,
        workspace: str = "",
    ) -> None:
        """List eligible files before copying any source.

        The preview gives the user a chance to exclude individual files. No
        workspace files are changed by this operation.

        Parameters
        ----------
        sources: list[str]
            Files or folders the user selected.
        extensions: list[str]
            File types the new project copies: its type's own and any the
            user added. Folder files of other types are listed as skipped,
            so they cannot be ticked.
        max_file_bytes: int
            Largest file that may be copied.
        workspace: str, optional
            Chosen workspace path, used to avoid scanning itself. Defaults to
            ``''``.

        Notes
        -----
        Builds selectable and skipped source-entry lists without copying files
        or loading models. Oversized files are reported separately for frontend
        selection/limit adjustment. Directory selection uses the repository
        scanner. Direct files bypass name and suffix filters while retaining
        size limits. Errors become frontend events.

        """

        try:
            roots = []
            allowed_extensions = tuple(dict.fromkeys(extensions))
            destination = _workspace_path(sources, self.sandbox, workspace)
            for value in sources:
                original = Path(value).expanduser()
                if original.is_symlink():
                    raise ValueError(f"Do not add a symbolic link: {original}")
                source = original.resolve(strict=True)
                if source.is_file():
                    size = source.stat().st_size
                    if size == 0:
                        reason = "empty file"
                        reason_code = SELECTION_REASON_EMPTY_FILE
                    elif size > max_file_bytes:
                        reason = "larger than the file-size limit"
                        reason_code = SELECTION_REASON_FILE_TOO_LARGE
                    else:
                        reason = None
                        reason_code = None
                    files = [] if reason else [source.name]
                    skipped = (
                        []
                        if reason is None
                        else [
                            {
                                "path": source.name,
                                "reason": reason,
                                "reason_code": reason_code,
                                "size_bytes": size,
                            }
                        ]
                    )
                else:
                    exclusions = (
                        (destination.relative_to(source).as_posix(),)
                        if destination.is_relative_to(source)
                        else ()
                    )
                    report = scan_repository(
                        source,
                        extensions=allowed_extensions,
                        max_file_bytes=max_file_bytes,
                        exclude_paths=exclusions,
                    )
                    if not report["complete"]:
                        raise ValueError(
                            f"Could not scan every file in {source}"
                        )
                    files = report["selected_files"]
                    skipped = [
                        {
                            "path": row["path"],
                            "reason": row["reason"],
                            "reason_code": row["reason_code"],
                            "size_bytes": row["size_bytes"],
                        }
                        for row in report["entries"]
                        if row["kind"] == "file"
                        and row["status"] != "selected"
                    ]
                roots.append(
                    {"path": str(source), "files": files, "skipped": skipped}
                )
            self._send("preview", roots=roots)
        except Exception as error:
            self._report_error(
                "gui.source_preview.failed",
                error,
                f"Could not inspect the selected files: {error}",
            )

    def open_project(
        self,
        sources: list[str],
        type_id: str,
        workspace: str,
        selected_files: dict[str, list[str]] | None = None,
        settings: dict[str, object] | None = None,
    ) -> None:
        """Create and activate a new, independently owned workspace.

        Creation is exclusive and cannot resume an existing project. Empty
        source lists require an explicit nonexistent or empty destination.

        Parameters
        ----------
        sources: list[str]
            Source files and folders to copy into the workspace.
        type_id: str
            Registered project-type identifier.
        workspace: str
            Workspace folder; required without sources. With sources, a blank
            value derives a name beneath the configured workspace parent.
        selected_files: dict[str, list[str]] | None, optional
            Checked file paths within each source folder. Defaults to ``None``.
        settings: dict[str, object] | None, optional
            Preparation settings saved in profile.ini. Defaults to ``None``.

        Notes
        -----
        The core reserves the final destination and completes creation before
        the current session is closed or any new conversation/trace is created.
        Default collisions receive a fresh name. Explicit collisions emit
        workspace_conflict without modifying the destination or current
        session; the frontend may request a separate resume operation.

        """

        switched = False
        try:
            if type_id not in PROJECT_TYPE_IDS:
                raise ValueError("Choose a project type.")
            settings = dict(settings or {})
            created_levels = settings.pop("_gui_effort_levels", None)
            created_depth = settings.pop("_gui_search_depth", None)
            created_routers = settings.pop("_gui_router_models", None)
            if created_routers is not None:
                if not isinstance(created_routers, dict) or any(
                    backend not in {BACKEND_MLX, BACKEND_TORCH}
                    for backend in created_routers
                ):
                    raise ValueError("Invalid intent models.")
                created_routers = {
                    backend: validate_model_id(model)
                    for backend, model in created_routers.items()
                    if model
                }
            if created_depth is not None:
                created_depth = validate_search_depth(created_depth)
            if created_levels is not None:
                if not isinstance(created_levels, dict) or any(
                    backend not in {BACKEND_MLX, BACKEND_TORCH}
                    for backend in created_levels
                ):
                    raise ValueError("Invalid effort levels.")
                created_levels = {
                    backend: validate_effort_levels(levels)
                    for backend, levels in created_levels.items()
                }
            if not sources and not workspace.strip():
                raise ValueError(
                    "Choose a workspace folder for a project without sources."
                )
            if selected_files is not None:
                # Resolve path aliases before matching checked files.
                selected_files = {
                    str(Path(path).expanduser().resolve(strict=True)): files
                    for path, files in selected_files.items()
                }
                missing = [
                    source
                    for source in sources
                    if str(Path(source).expanduser().resolve(strict=True))
                    not in selected_files
                ]
                if missing:
                    raise ValueError("Choose files for every added source.")
            # Creation must succeed before touching the current session or
            # creating conversation/trace files in the destination.
            self._send("status", message="Preparing the new project...")
            options: dict[str, object] = {}
            if selected_files is not None:
                options["selected_files"] = selected_files
            if settings:
                options["settings"] = settings
            if not workspace.strip():
                options["workspace_root"] = self.sandbox
            project = prepare_sources_workspace(
                sources, type_id, workspace if workspace.strip() else None,
                **options,
            )
            # The profile is now published. Stop must preserve it, even
            # if Qt has not yet processed the later opened event.
            self._send("project_created", workspace=str(project.root))
            if created_depth is not None:
                write_search_depth(project.root, created_depth)
            if created_levels:
                for backend, levels in created_levels.items():
                    write_effort_levels(project.root, backend, levels)
            for backend, model in (created_routers or {}).items():
                write_router_model(project.root, backend, model)
            self.close()
            switched = True
            conversation = self._open_conversation_store(project)
            self._start_project_trace(project.root, conversation)
            if created_levels:
                self.active_trace.emit(
                    "gui.models.effort_levels.saved",
                    component="gui_engine",
                    data=created_levels,
                )
            self._activate_project(project, conversation)
        except WorkspaceOccupiedError as error:
            # Do not append even an error trace to the protected project.
            # Resuming requires a separate explicit frontend command.
            self._send(
                "workspace_conflict",
                workspace=str(error.directory),
                existing_project=isinstance(error, ProjectAlreadyExistsError),
                message=str(error),
            )
        except Exception as error:
            message = f"Could not prepare the project: {error}"
            if switched:
                self._report_error(
                    "gui.project.prepare.failed", error, message
                )
            else:
                self._send("error", message=message, trace_path=None)
            if self.session is None:
                self._close_project_trace()

    def _open_conversation_store(
        self,
        project: QAProject,
        conversation_id: str | None = None,
    ) -> ConversationInfo:
        """Open a conversation store and select the requested conversation.

        Parameters
        ----------
        project: QAProject
            Opened project whose profile, source, index, and log paths are
            ready for activation.
        conversation_id: str | None, optional
            Existing conversation to activate. If omitted, select the most
            recently updated conversation. Defaults to ``None``.

        Returns
        -------
        ConversationInfo
            Selected conversation, also assigned as the active conversation.

        Raises
        ------
        ValueError
            If the explicitly requested conversation does not exist.

        """

        store = ConversationStore(project.root)
        conversation = store.ensure()
        if conversation_id is not None:
            conversation = store.select(conversation_id)
        self.conversation_store = store
        self.active_conversation = conversation
        return conversation

    def _activate_project(
        self,
        project: QAProject,
        conversation: ConversationInfo,
    ) -> None:
        """Load models and indexes, then report the active workspace.

        Parameters
        ----------
        project: QAProject
            Opened project whose profile, source, index, and log paths are
            ready for activation.
        conversation: ConversationInfo
            Registered conversation metadata for the state to activate or
            trace.

        Raises
        ------
        RuntimeError
            If no conversation store is open.

        Notes
        -----
        Constructs RAGSession with the active sink and exact conversation path,
        then emits opened and conversation metadata. Model loading,
        source/index preparation, and state validation can raise to the public
        action wrapper.

        """

        options: dict[str, object] = {
            "progress_callback": lambda progress: self._send(
                "progress", **asdict(progress)
            )
        }
        options["trace"] = self.project_trace
        if self.conversation_store is None:
            raise RuntimeError("Conversation store is not open.")
        # The project keeps every setting from now on, so it resumes as it
        # was set up even when the defaults or preferences change.
        if save_project_settings(project.root):
            self.active_trace.emit(
                "gui.project.settings.saved",
                component="gui_engine",
                data={"path": str(project.root / PROFILE_FILE)},
            )
        options["conversation_path"] = self.conversation_store.state_path(
            conversation
        )
        session = RAGSession(project.root, **options)
        session.use_router_model(
            read_router_model(project.root, session.backend)
        )
        self.session = session
        self._use_project_answer_settings()
        self._announce_project(project, conversation)

    def _announce_project(
        self, project: QAProject, conversation: ConversationInfo
    ) -> None:
        """Send the ``opened`` event for the prepared core project.

        Parameters
        ----------
        project: QAProject
            Current project and its associated metadata.
        conversation: ConversationInfo
            Conversation metadata included in the frontend event.

        """

        session = self.session
        self._send(
            "opened",
            project=str(project.root),
            source=str(project.source),
            type_id=project.profile.type_id,
            profile=asdict(project.profile),
            effort_levels=_effort_levels_by_backend(
                project.root, project.profile
            ),
            router_models=_router_models_by_backend(project.root),
            search_depth=read_search_depth(project.root),
            preferences=_preference_payload(),
            usage=self._usage_totals(),
            index_status=session.index_status,
            turns=_turn_payloads(session.turns, project.root),
            workspace_files=list_project_files(project),
            retrieval_count=(
                session.turns[-1].parameters.get(
                    "retrieval_count", project.profile.retrieval_count
                )
                if session.turns
                else project.profile.retrieval_count
            ),
            temperature=session.chat_session.temperature,
            thinking=session.chat_session.thinking,
            max_new_tokens=session.chat_session.max_new_tokens,
            active_conversation_id=conversation.conversation_id,
            conversations=[
                {
                    **asdict(item),
                    "empty": self.conversation_store.is_empty(
                        item.conversation_id
                    ),
                }
                for item in self.conversation_store.conversations()
            ],
        )

    def _use_project_answer_settings(self) -> None:
        """Answer every chat with the project's saved answer settings.

        Temperature, thinking, and answer length belong to the project, not
        to a chat: switching chats keeps them, and switching projects loads
        the other project's. Saved values are never substituted.

        """

        profile = self.session.project.profile
        self.session.chat_session.configure_generation(
            {
                "temperature": profile.temperature,
                "thinking": profile.thinking,
                "max_new_tokens": profile.max_new_tokens,
            }
        )

    def _save_answer_settings(self, settings: dict[str, object]) -> None:
        """Save changed answer settings to the project's profile.ini.

        Parameters
        ----------
        settings: dict[str, object]
            Settings used to configure the operation.

        """

        profile = self.session.project.profile
        changes = {
            key: settings[key]
            for key in ("temperature", "thinking", "max_new_tokens")
            if key in settings and settings[key] != getattr(profile, key)
        }
        if not changes:
            return
        self.session.project = configure_project(
            self.session.project.root,
            {
                key: "none" if value is None else value
                for key, value in changes.items()
            },
            trace=self.active_trace,
        )

    def save_answer_settings(
        self,
        settings: object,
        request_id: int | None = None,
        project: str | None = None,
    ) -> None:
        """Persist answer settings and acknowledge the requesting project.

        A correlated GUI request carries a request ID and project path. The
        acknowledgement describes the saved profile, never merely the
        requested values. When saving fails, reread ``profile.ini`` and
        reconfigure the live chat before declaring the values verified.

        Parameters
        ----------
        settings: object
            Mapping with temperature, thinking, or max_new_tokens. Other
            keys are ignored, as in the existing command contract.
        request_id: int | None, optional
            Positive identifier used to match the GUI acknowledgement.
            None retains the uncorrelated caller's event contract.
        project: str | None, optional
            Canonical project path attached to a correlated GUI request.
            None retains the existing caller's project selection.

        """

        event_project = project or ""
        try:
            if request_id is not None and (
                type(request_id) is not int or request_id < 1
            ):
                raise ValueError("Invalid answer-settings request ID.")
            if self.session is None:
                raise RuntimeError("Open a project first.")
            actual_project = str(self.session.project.root)
            if project is not None and project != actual_project:
                raise RuntimeError("The active project has changed.")
            event_project = actual_project
            if not isinstance(settings, dict):
                raise ValueError("Answer settings must be an object.")
            self._save_answer_settings(settings)
            self._use_project_answer_settings()
            profile = self.session.project.profile
            response = {
                "temperature": profile.temperature,
                "thinking": profile.thinking,
                "max_new_tokens": profile.max_new_tokens,
            }
            if request_id is not None:
                response.update(
                    request_id=request_id, project=event_project
                )
            self._send("answer_settings_saved", **response)
        except Exception as error:
            if request_id is None:
                self._report_error(
                    "gui.answer_settings.failed",
                    error,
                    f"Could not save the answer settings: {error}",
                )
                return
            # configure_project may have replaced profile.ini before a
            # trace or live-session update failed. The disk state, not the
            # old in-memory profile, is authoritative after such a failure.
            verified = False
            response = {}
            if (
                self.session is not None
                and (
                    project is None
                    or project == str(self.session.project.root)
                )
            ):
                try:
                    self.session.project = open_project(
                        self.session.project.root
                    )
                    self._use_project_answer_settings()
                    profile = self.session.project.profile
                    response = {
                        "temperature": profile.temperature,
                        "thinking": profile.thinking,
                        "max_new_tokens": profile.max_new_tokens,
                    }
                    verified = True
                except Exception as recovery_error:
                    self.active_trace.exception(
                        "gui.answer_settings.reconcile_failed",
                        recovery_error,
                        component="gui_engine",
                    )
            self.active_trace.exception(
                "gui.answer_settings.failed",
                error,
                component="gui_engine",
            )
            self._send(
                "answer_settings_failed",
                request_id=request_id,
                project=event_project,
                verified=verified,
                message=f"Could not save the answer settings: {error}",
                **response,
            )

    def tool_settings(self) -> None:
        """Send the tools the open project can use, with their settings.

        Notes
        -----
        Emits ``tool_settings`` with one record per tool, as
        ``project_tool_settings`` describes them.

        """

        try:
            if self.session is None:
                raise RuntimeError("Open a project first.")
            self._send(
                "tool_settings",
                tools=project_tool_settings(self.session.project.root),
            )
        except Exception as error:
            self._report_error(
                "gui.tool_settings.failed",
                error,
                f"Could not read the tool settings: {error}",
            )

    def save_tool_settings(self, name: str, settings: object) -> None:
        """Save one tool's settings for the open project.

        Parameters
        ----------
        name: str
            Tool the project type uses.
        settings: object
            Mapping of the tool's changeable settings to new values.

        Notes
        -----
        Emits ``tool_settings`` with every tool and ``tool`` set to ``name``:
        with ``saved`` True, or with ``error`` when the tool rejects a value,
        so the page shows the reason next to the tool. The next tool run
        uses the saved values; nothing is reloaded.

        """

        try:
            if self.session is None:
                raise RuntimeError("Open a project first.")
            if not isinstance(settings, dict):
                raise ValueError("Tool settings must be an object.")
            root = self.session.project.root
            try:
                tools = save_tool_settings(
                    root, name, settings, trace=self.active_trace
                )
            except ValueError as error:
                self._send(
                    "tool_settings",
                    tools=project_tool_settings(root),
                    tool=name,
                    error=str(error),
                )
                return
            self._send("tool_settings", tools=tools, tool=name, saved=True)
        except Exception as error:
            self._report_error(
                "gui.tool_settings.failed",
                error,
                f"Could not save the tool settings: {error}",
            )

    def check_sources(self, sources: list[str]) -> None:
        """Tell the user what adding these sources would copy.

        Parameters
        ----------
        sources: list[str]
            Files and folders the user dropped or chose.

        Notes
        -----
        Sends ``sources_checked`` with the files that would be added, those
        left out for their type or size, and, for sources already in the
        project, which of their copies changed since they were copied;
        nothing is copied. The user then confirms with
        ``add_project_sources``.

        """

        try:
            if self.session is None:
                raise RuntimeError("Open a project first.")
            if (
                not sources
                or not isinstance(sources, list)
                or not all(
                    isinstance(path, str) and path.strip() for path in sources
                )
            ):
                raise ValueError("Choose source files or folders.")
            project = self.session.project
            plan = preview_project_sources(project, sources)
            changes = {"added": [], "changed": [], "deleted": [], "edited": []}
            existing = [
                item["name"] for item in plan["add"]
                if item["state"] == "existing"
            ]
            if existing:
                found = find_source_changes(project)
                changes = {
                    group: [
                        path for path in paths
                        if any(
                            path == f"{PROJECT_SOURCES_DIRECTORY}/{name}"
                            or path.startswith(
                                f"{PROJECT_SOURCES_DIRECTORY}/{name}/"
                            )
                            for name in existing
                        )
                    ]
                    for group, paths in found.items()
                }
            self._send(
                "sources_checked",
                sources=sources,
                add=plan["add"],
                files=plan["files"],
                skipped_folders=[Path(path).name for path in plan["skipped"]],
                changes=changes,
            )
        except Exception as error:
            self._report_error(
                "gui.sources.check.failed",
                error,
                f"Could not check the sources: {error}",
            )

    def add_project_sources(
        self, sources: list[str], decisions: object = None
    ) -> None:
        """Add confirmed sources and restore the active chat.

        Parameters
        ----------
        sources: list[str]
            Source paths approved for addition to the active project.
        decisions: object, optional
            For sources already in the project, sync actions keyed by the
            ``sources/`` paths reported by ``check_sources``. Defaults to
            ``None``: copies already in the project stay as they are.

        Notes
        -----
        Only sources not yet in the project are copied; folders without any
        file to copy are skipped. The index is refreshed once at the end.

        """

        root = None
        conversation_id = None
        try:
            if self.session is None or self.active_conversation is None:
                raise RuntimeError("Open a project first.")
            if (
                not sources
                or not isinstance(sources, list)
                or not all(
                    isinstance(path, str) and path.strip() for path in sources
                )
            ):
                raise ValueError("Choose source files or folders.")
            root = self.session.project.root
            conversation_id = self.active_conversation.conversation_id
            self.active_trace.emit(
                "gui.sources.add.requested",
                component="gui_engine",
                data={"sources": sources},
            )
            self._send(
                "status", message="Adding sources and updating the index…"
            )
            if decisions is not None and not (
                isinstance(decisions, dict)
                and all(
                    isinstance(path, str) and isinstance(action, str)
                    for path, action in decisions.items()
                )
            ):
                raise ValueError("Source decisions must map paths to actions.")
            # A folder without a single file of the project's types cannot
            # be copied, and a source already in the project changes only
            # through the user's choices; the check told the user both.
            plan = preview_project_sources(self.session.project, sources)
            new = [
                value
                for value, item in zip(plan["sources"], plan["add"])
                if item["state"] != "existing"
            ]
            if new:
                add_project_sources(root, new, trace=self.active_trace)
            if decisions:
                done = sync_project_sources(
                    open_project(root), decisions=decisions
                )
                self.active_trace.emit(
                    "gui.sources.synced", component="gui_engine", data=done
                )
            if new or decisions:
                self._refresh_sources(open_project(root))
            else:
                self._announce_project(
                    self.session.project, self.active_conversation
                )
        except Exception as error:
            # A failed copy may have written some files: index what is there,
            # so an old in-memory index is never used.
            if root is not None and self.session is not None:
                try:
                    self._refresh_sources(open_project(root))
                except Exception:
                    pass
            if root is not None and self.session is None:
                try:
                    self._reopen_project_conversation(
                        open_project(root), conversation_id
                    )
                except Exception:
                    self.close()
                    self._send(
                        "backend_failed",
                        message=(
                            "Reopen the project to finish indexing. "
                            "Your saved conversations are preserved."
                        ),
                    )
            self._report_error(
                "gui.sources.add.failed",
                error,
                f"Could not finish adding sources: {error}",
            )

    def remove_project_sources(self, paths: list[str]) -> None:
        """Delete managed source copies through the core and reindex.

        Parameters
        ----------
        paths: list[str]
            Workspace-relative source snapshots to remove.

        """

        root = None
        conversation_id = None
        try:
            if self.session is None or self.active_conversation is None:
                raise RuntimeError("Open a project first.")
            root = self.session.project.root
            conversation_id = self.active_conversation.conversation_id
            self.active_trace.emit(
                "gui.sources.remove.requested",
                component="gui_engine",
                data={"paths": paths},
            )
            self._send(
                "status", message="Removing sources and updating the index…"
            )
            remove_workspace_sources(self.session.project, paths)
            self._refresh_sources(open_project(root))
        except Exception as error:
            # Refresh after a partial deletion to match the files on disk.
            if root is not None and self.session is not None:
                try:
                    self._refresh_sources(open_project(root))
                except Exception:
                    pass
            if root is not None and self.session is None:
                try:
                    self._reopen_project_conversation(
                        open_project(root), conversation_id
                    )
                except Exception:
                    self.close()
                    self._send(
                        "backend_failed",
                        message=(
                            "Reopen the project to finish indexing. "
                            "Your saved conversations are preserved."
                        ),
                    )
            self._report_error(
                "gui.sources.remove.failed",
                error,
                f"Could not remove sources: {error}",
            )

    def sync_sources(self, decisions: object = None) -> None:
        """Apply the user's choice for each changed source, then reindex.

        Parameters
        ----------
        decisions: object, optional
            Mapping of ``sources/`` paths from ``source_changes`` to
            ``import``, ``replace``, ``keep_both``, ``delete`` or ``keep``.
            Unlisted files are left untouched. Defaults to ``None``, which
            updates changed copies, removes deleted ones and imports nothing.

        Notes
        -----
        Copies the user edited in the project are kept unless a decision
        names them. The core checks every decision before changing files.
        The index is refreshed even if a file fails, so it matches the
        files on disk.

        """

        try:
            if self.session is None:
                raise RuntimeError("Open a project first.")
            if decisions is not None and not (
                isinstance(decisions, dict)
                and all(
                    isinstance(path, str) and isinstance(action, str)
                    for path, action in decisions.items()
                )
            ):
                raise ValueError("Source decisions must map paths to actions.")
            project = self.session.project
            self._send(
                "status", message="Updating sources and the index…"
            )
            try:
                done = sync_project_sources(project, decisions=decisions)
                self.active_trace.emit(
                    "gui.sources.synced", component="gui_engine", data=done
                )
            finally:
                self._refresh_sources(open_project(project.root))
        except Exception as error:
            self._report_error(
                "gui.sources.sync.failed",
                error,
                f"Could not update sources: {error}",
            )

    def edit_folder(
        self, folder: str, operation: str, **arguments: object
    ) -> None:
        """Delegate editing outputs/ and templates/ to the core API.

        Parameters
        ----------
        folder: str
            User-editable project folder.
        operation: str
            One of ``add``, ``replace`` or ``remove``.
        **arguments: object
            File paths and optional replacement policy from the GUI.

        """

        try:
            if self.session is None:
                raise RuntimeError("Open a project first.")
            project = self.session.project
            if operation == "add":
                changed = add_project_files(
                    project,
                    folder,
                    _path_list(arguments.get("paths")),
                    replace=arguments.get("replace") is True,
                )
            elif operation == "replace":
                path = arguments.get("path")
                source = arguments.get("source")
                if not isinstance(path, str) or not isinstance(source, str):
                    raise ValueError("Choose a file and its replacement.")
                if not path.startswith(folder + "/"):
                    raise ValueError(f"Only files in {folder}/ can be edited.")
                changed = [replace_project_file(project, path, source)]
            elif operation == "remove":
                values = _path_list(arguments.get("paths"))
                if any(
                    not value.startswith(folder + "/") for value in values
                ):
                    raise ValueError(
                        f"Only files in {folder}/ can be deleted."
                    )
                changed = remove_project_files(project, values)
            else:
                raise ValueError(f"Unknown operation: {operation}.")
            self.active_trace.emit(
                f"gui.folder.{operation}",
                component="gui_engine",
                data={"folder": folder, "paths": changed},
            )
            self._send(
                "folder_changed",
                folder=folder,
                operation=operation,
                paths=changed,
                workspace_files=list_project_files(project),
            )
        except Exception as error:
            self._report_error(
                "gui.folder.edit.failed",
                error,
                f"Could not change {folder}/: {error}",
            )

    def _refresh_sources(self, project: QAProject) -> None:
        """Re-index changed sources, keeping the models and the chat.

        Only new or changed files are chunked and embedded; the models
        stay loaded, so adding or removing a file does not reload them.

        Parameters
        ----------
        project: QAProject
            Current project and its associated metadata.

        """

        self.session.refresh_sources(project)
        self.active_trace.emit(
            "gui.sources.refreshed",
            component="gui_engine",
            data={
                "index_status": self.session.index_status,
                "changes": self.session.source_changes,
            },
        )
        self._announce_project(project, self.active_conversation)

    def _reopen_project_conversation(
        self, project: QAProject, conversation_id: str
    ) -> None:
        """Discover records before restoring a moved or updated project.

        Parameters
        ----------
        project: QAProject
            Current project and its associated metadata.
        conversation_id: str
            Identifier of the saved conversation.

        """

        self._close_project_trace()
        store = ConversationStore(project.root)
        store.ensure()
        conversation = store.select(conversation_id)
        self.conversation_store = store
        self.active_conversation = conversation
        self._start_project_trace(project.root, conversation)
        self._activate_project(project, conversation)

    def _name_active_conversation(
        self, question: str, answer: str = ""
    ) -> None:
        """Generate and persist a short title from the first exchange.

        The answer, when given, lets the model resolve abbreviations and
        jargon instead of guessing what they stand for.

        Parameters
        ----------
        question: str
            Question or request to analyze.
        answer: str, optional
            Answer text to inspect. Defaults to ``''``.

        """

        if (
            self.session is None
            or self.conversation_store is None
            or self.active_conversation is None
        ):
            return
        excerpt = _answer_excerpt(answer, 1200)
        prompt = f"Question: {question}\n"
        if excerpt:
            prompt += f"Answer: {excerpt}\n"
        # The router model writes titles; it stays loaded, so no switch.
        title = self.session.router_reply(
            "Create a concise conversation title. Use three to six words. "
            "Describe the topic, not the user's request. Keep abbreviations "
            "and technical terms as the user wrote them; expand one only as "
            "the answer defines it, never by guessing. Do not use citations, "
            "brackets, labels, quotation marks, or punctuation. Return only "
            "the title.",
            f"{prompt}Title:",
            max_new_tokens=32,
            language_context=(question,),
        )
        title = title.splitlines()[0] if title.splitlines() else ""
        title = re.sub(r"^(?:title\s*:\s*)", "", title, flags=re.I)
        title = re.sub(r"^\[[0-9, ]+\]\s*", "", title)
        title = " ".join(title.strip().strip("\"'").split())
        title = re.sub(
            r"^help me understand\s+", "Understanding ", title, flags=re.I
        )
        fallback = " ".join(question.strip().rstrip("?.!").split())
        fallback = re.sub(
            r"^help me understand\s+",
            "Understanding ",
            fallback,
            flags=re.I,
        )
        fallback = re.sub(
            r"^what (?:is|are)\s+", "Understanding ", fallback, flags=re.I
        )
        # A reply longer than a title (a sentence) would be cut mid-phrase;
        # the question makes a better title then.
        if (
            not title
            or len(title.split()) > 6
            or re.search(r"\bunderstand(?:ing)? me\b", title, re.I)
        ):
            title = fallback
        title = " ".join(title.split()[:6])[:80]
        if not title:
            return
        renamed = self.conversation_store.rename(
            self.active_conversation.conversation_id,
            title,
            ownership=self._store_ownership(
                self.active_conversation.conversation_id
            ),
            state_writer=self._store_state_writer(
                self.active_conversation.conversation_id
            ),
        )
        self.active_conversation = renamed
        self._send_conversations("conversations")

    def ask(
        self,
        question: str,
        retrieval_count: int | None = None,
        generation_settings: dict[str, object] | None = None,
        effort: dict[str, int] | str | None = None,
    ) -> None:
        """Ask the loaded project and emit its saved answer and evidence.

        Generation settings accompanying a question are also persisted to
        profile.ini. Retrieval effort does not rebuild the project indexes.

        Parameters
        ----------
        question: str
            Question about the active project.
        retrieval_count: int | None, optional
            Number of evidence results to use for this answer. Defaults to
            ``None``.
        generation_settings: dict[str, object] | None, optional
            Temperature, thinking, and answer length for this answer. Defaults
            to ``None``.
        effort: dict | str | None, optional
            Effort for this answer: ``{"model": i, "search_depth": j}``
            picks the model of effort level i and search-depth step j
            (zero-based; the step is the number of passages retrieved). A
            level name such as ``"lagom"`` uses the same index for both.
            Defaults to ``None``.

        Notes
        -----
        Requires an active session. Progress phases and the final serialized
        turn are emitted as events; the method does not return the answer
        string. Model/state failures become error events, potentially after
        in-memory history has advanced.

        """

        try:
            if self.session is None:
                raise RuntimeError("Open a project before asking a question.")
            generation_settings = dict(generation_settings or {})
            title_question = str(
                generation_settings.pop("_title_question", question)
            )
            # Answer settings belong to the project; every chat uses them.
            self._save_answer_settings(generation_settings)
            backend = self.session.backend
            levels = read_effort_levels(self.session.project.root, backend)
            indexes = _effort_indexes(effort)
            is_mlx = self.session.chat_session.backend == BACKEND_MLX
            # Name a new chat after its first answer, from question and answer.
            name_after_answer = (
                indexes is not None
                and not self.session.turns
                and self.conversation_store is not None
                and self.active_conversation is not None
                and self.active_conversation.name.startswith("Conversation ")
            )
            if indexes is not None:
                model_index, depth_index = indexes
                depth = read_search_depth(self.session.project.root)
                retrieval_count = depth[depth_index]
                model_id = str(levels[model_index]["model"])
                # RAGSession switches chat models in place only on MLX.
                if is_mlx:
                    if model_id != self.session.chat_session.model_id:
                        # Loading can take seconds; show it as its own stage.
                        self._send(
                            "answer_phase",
                            phase=ENGINE_PHASE_LOADING_MODEL,
                            model_id=model_id,
                        )
                    self.session.use_chat_model(model_id)
            self._send("status", message="Finding evidence and answering…")
            options: dict[str, object] = {}
            if retrieval_count is not None:
                options["retrieval_count"] = retrieval_count
            if generation_settings:
                options["generation_settings"] = generation_settings
            # The RAG core resolves project-type and intent guidance.
            stream = _AnswerStream(self._send)
            options["token_callback"] = stream.add
            options["phase_callback"] = lambda phase, details: self._send(
                "answer_phase",
                phase=phase,
                **details,
            )
            if self.read_command is not None:
                options["confirm_callback"] = self._confirm_plan
                options["choice_callback"] = self._choose_option
            # Reports such as research briefings also get a one-page PDF.
            options["pdf_writer"] = write_report_pdf
            options["stop_requested"] = self._answer_stop.is_set
            self._answer_stop.clear()
            self._answering = True
            try:
                with self._live_usage():
                    self.session.ask(question, **options)
            finally:
                self._answering = False
                self._answer_stop.clear()
            stream.flush()
            if (
                self.conversation_store is not None
                and self.active_conversation is not None
            ):
                self.active_conversation = self.conversation_store.touch(
                    self.active_conversation.conversation_id,
                    ownership=self._store_ownership(
                        self.active_conversation.conversation_id
                    ),
                    state_writer=self._store_state_writer(
                        self.active_conversation.conversation_id
                    ),
                )
            # An answer can add files (a tool's report in outputs/, the
            # conversation log), so the workspace tree is refreshed too.
            self._send(
                "answered",
                turn=asdict(self.session.turns[-1]),
                usage=self._usage_totals(),
                workspace_files=list_project_files(self.session.project),
            )
        except GenerationCancelled:
            # Stop: no turn is saved, and models and project stay loaded.
            self.active_trace.emit(
                "gui.answer.cancelled",
                component="gui_engine",
                data={"reason": "Stopped by the user."},
            )
            self._send("answer_cancelled", usage=self._usage_totals())
            return
        except Exception as error:
            self._report_error(
                "gui.question.failed",
                error,
                str(error),
            )
            return
        if name_after_answer:
            try:
                self._name_active_conversation(
                    title_question, self.session.turns[-1].answer
                )
            except Exception as error:
                # The answer is saved; a failed title keeps the default name.
                self.active_trace.emit(
                    "gui.conversation.title.failed",
                    component="gui_engine",
                    data={"error": str(error)},
                )

    def cancel_answer(self) -> None:
        """Stop the running answer at its next generated token.

        Notes
        -----
        Called from the input thread, so it works while ``ask`` is busy.
        Without a running answer it does nothing. The answer ends with an
        ``answer_cancelled`` event; no turn is saved and loaded models stay
        in memory.

        """

        if self._answering:
            self._answer_stop.set()

    @contextmanager
    def _live_usage(self) -> Iterator[None]:
        """Send the tokens used by the running answer twice a second.

        Yields
        ------
        None
            Control while periodic progress reporting runs.

        Notes
        -----
        A background thread reads the model sessions' token totals, which
        grow while a reply streams, and sends ``usage_progress`` events
        with the tokens read and generated so far for this answer.

        """

        session = self.session
        mark = session.usage_mark()
        stopped = Event()

        def tick() -> None:
            """Periodically publish token usage while the answer is active."""

            while not stopped.wait(0.5):
                self._send("usage_progress", **session.usage_since(mark))

        ticker = Thread(target=tick, name="kilin-usage", daemon=True)
        ticker.start()
        try:
            yield
        finally:
            stopped.set()
            ticker.join(timeout=1.0)

    def save_answer_timeline(self, timestamp: str, timeline: object) -> None:
        """Keep how long an answer took, as the page showed it.

        Parameters
        ----------
        timestamp: str
            The answered turn's timestamp, which identifies it.
        timeline: object
            ``elapsed_ms`` and ``steps`` (each a ``label`` and ``ms``) from
            the page; anything else is ignored.

        Notes
        -----
        The timing is presentation only: it is not part of the
        conversation record, and a missing or unreadable file only means
        resumed answers show their tokens without the time.

        """

        if self.session is None or not timestamp:
            return

        if not _valid_timeline(timeline):
            return

        root = self.session.project.root
        timelines = _read_answer_timelines(root)
        timelines[timestamp] = timeline
        path = root / "logs" / _ANSWER_TIMELINES
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(timelines))

    def _usage_totals(self) -> dict[str, dict[str, int]]:
        """Tokens used by the open chat and by the whole project.

        Returns
        -------
        dict[str, dict[str, int]]
            ``chat`` and ``project``, each with ``input_tokens``,
            ``output_tokens``, and ``answers``. The project counts every
            saved chat, deleted chats included; a turn that branched chats
            share is counted once.

        """

        def add(total: dict[str, int], parameters: object) -> None:
            """Accumulate token counts and answer count from a usage record.

            Parameters
            ----------
            total: dict[str, int]
                Mutable token counters accumulated across records.
            parameters: object
                Model usage record, expected to contain a usage dictionary.

            """

            usage = (
                parameters.get("usage")
                if isinstance(parameters, dict)
                else None
            )
            if isinstance(usage, dict):
                total["input_tokens"] += int(usage.get("input_tokens", 0))
                total["output_tokens"] += int(usage.get("output_tokens", 0))
                total["answers"] += 1

        empty = {"input_tokens": 0, "output_tokens": 0, "answers": 0}
        chat = dict(empty)
        project = dict(empty)
        if self.session is None:
            return {"chat": chat, "project": project}
        for turn in self.session.turns:
            add(chat, getattr(turn, "parameters", None))
        seen: set[tuple[str, str]] = set()
        traces = self.session.project.root / "logs" / "traces"
        for path in sorted(traces.rglob("conversation_*.json")):
            try:
                state = conversation_state(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError):
                continue  # An unreadable file is left out of the total.
            for turn in state.get("turns", []) or []:
                if not isinstance(turn, dict):
                    continue
                key = (str(turn.get("timestamp")), str(turn.get("question")))
                if key not in seen:
                    seen.add(key)
                    add(project, turn.get("parameters"))
        return {"chat": chat, "project": project}

    def _confirm_plan(self, plan: ToolPlan) -> PlanChoice:
        """Wait for the user's choice on a tool plan.

        The page shows the plan from the ``processing`` answer phase that
        precedes this call and answers with a ``plan_choice`` command.

        Parameters
        ----------
        plan: ToolPlan
            The plan the page is showing.

        Returns
        -------
        PlanChoice
            ``continue``, ``quick``, or ``cancel``. Ended input cancels.

        Notes
        -----
        Commands are sequential, so other commands cannot run meanwhile;
        each is answered with an error event marked ``pending_choice``
        and the wait continues. The page must retain the controls.

        """

        while True:
            command = self.read_command()
            if command is None:
                return PLAN_CHOICE_CANCEL
            if command.get("action") == "cancel_answer":
                raise GenerationCancelled("Stopped by the user.")
            if command.get("action") == "plan_choice":
                choice = command.get("choice")
                if choice in PLAN_CHOICE_CHOICES:
                    return choice
                return PLAN_CHOICE_CANCEL
            self._send(
                "error",
                message="Choose Continue, Quick, or Cancel for the plan "
                "first.",
                pending_choice=True,
            )

    def _choose_option(
        self, question: str, options: tuple[str, ...]
    ) -> int | None:
        """Wait for the user to pick one of a tool's options.

        The page shows the options from the ``processing`` answer phase
        that precedes this call and answers with a ``plan_choice`` command
        whose choice is ``option:<index>`` or ``cancel``. Rejected
        unrelated commands emit errors marked ``pending_choice``;
        they must not dismiss the question or abort the answer.

        Parameters
        ----------
        question: str
            The question the page is showing.
        options: tuple[str, ...]
            The options the page is showing.

        Returns
        -------
        int | None
            The chosen index; None for cancel, an invalid answer, or ended
            input.

        """

        while True:
            command = self.read_command()
            if command is None:
                return None
            if command.get("action") == "cancel_answer":
                raise GenerationCancelled("Stopped by the user.")
            if command.get("action") == "plan_choice":
                choice = str(command.get("choice") or "")
                number = choice.removeprefix("option:")
                if choice.startswith("option:") and number.isdigit():
                    index = int(number)
                    return index if index < len(options) else None
                return None
            self._send(
                "error",
                message=f"Answer the question first: {question}",
                pending_choice=True,
            )

    def preview_reference(self, turn_index: int, citation: int) -> None:
        """Show nearby source lines for one saved answer citation.

        Only a citation already attached to the active conversation can be
        opened. The reader rechecks that the file stays inside the copied
        project source folder.

        Parameters
        ----------
        turn_index: int
            Zero-based position of a stored turn in the current RAG session.
        citation: int
            One-based evidence citation number within the selected turn.

        Notes
        -----
        Attempts to read current source context, falling back to saved evidence
        when unavailable. Notebook previews always retain saved cell-local
        excerpts. A source file modified since retrieval may no longer match
        the saved line range.

        """

        if self.session is None:
            self._send(
                "reference_error",
                turn_index=turn_index,
                citation=citation,
                message="Open a project first.",
            )
            return
        if (
            type(turn_index) is not int
            or not 0 <= turn_index < len(self.session.turns)
            or type(citation) is not int
        ):
            self._send(
                "reference_error",
                turn_index=turn_index,
                citation=citation,
                message="Reference not found.",
            )
            return

        turn = self.session.turns[turn_index]
        evidence = next(
            (item for item in turn.evidence if item.citation == citation),
            None,
        )
        if evidence is None:
            self._send(
                "reference_error",
                turn_index=turn_index,
                citation=citation,
                message="Reference not found.",
            )
            return

        try:
            report = {
                "repository": str(self.session.project.source),
                "selected_files": [evidence.path.as_posix()],
            }
            source = read_repository_files(report)[0]
            preview = _reference_context(
                source.text,
                evidence.text,
                evidence.start_line,
                evidence.end_line,
                evidence.path.suffix.lower() == ".pdf",
                evidence.cell,
            )
        except (OSError, ValueError, IndexError):
            preview = _saved_reference_context(
                evidence.text,
                evidence.start_line,
                "Source file unavailable; showing the saved passage.",
            )
        self._send(
            "reference_preview",
            turn_index=turn_index,
            citation=citation,
            path=evidence.path.as_posix(),
            **preview,
        )

    def create_conversation(
        self, *, branch: bool = False, turn_count: int | None = None
    ) -> None:
        """Create an empty conversation or branch the selected conversation.

        Parameters
        ----------
        branch: bool, optional
            Whether to copy resumable state from the active conversation
            instead of creating an empty conversation. Defaults to ``False``.
        turn_count: int | None, optional
            Number of the active conversation's first turns a branch keeps.
            None keeps every turn. Defaults to ``None``.

        Notes
        -----
        Creates a new or branched store record, then activates it while reusing
        loaded models. A new empty conversation reopens an unused empty one
        instead, if any, so empty files do not pile up. The previous
        conversation remains available. Errors are reported through the
        frontend error channel.

        """

        try:
            if (
                self.session is None
                or self.conversation_store is None
                or self.active_conversation is None
            ):
                raise RuntimeError("Open a project first.")
            unused = None if branch else self._unused_empty_conversation()
            if unused is not None:
                self.conversation_store.select(unused.conversation_id)
                self._switch_with_store_rollback(unused)
                return
            parent_id = (
                self.active_conversation.conversation_id if branch else None
            )
            conversation = self.conversation_store.create(
                branch_from=parent_id,
                turn_count=turn_count,
                ownership=(
                    self._store_ownership(parent_id)
                    if parent_id is not None
                    else None
                ),
            )
            self._switch_with_store_rollback(conversation)
        except Exception as error:
            self._report_error(
                "gui.conversation_create.failed",
                error,
                str(error),
            )

    def _unused_empty_conversation(self) -> ConversationInfo | None:
        """Find an empty conversation other than the active one.

        Returns
        -------
        ConversationInfo | None
            The first such conversation in creation order, or None. Empty
            conversations are not listed, so one left unused is reopened
            rather than a new file created.

        """

        active = self.active_conversation.conversation_id
        for item in self.conversation_store.conversations():
            if item.conversation_id == active:
                continue
            try:
                if self.conversation_store.is_empty(item.conversation_id):
                    return item
            except ValueError:
                continue
        return None

    def select_conversation(self, conversation_id: str) -> None:
        """Switch conversations while retaining loaded project resources.

        Parameters
        ----------
        conversation_id: str
            Identifier of a conversation registered in the current project
            store.

        Notes
        -----
        Operation errors are caught and reported through an error event.
        A rejected target-file lock restores the previous store selection and
        keeps the existing live session active.

        """

        try:
            if self.conversation_store is None:
                raise RuntimeError("Open a project first.")
            conversation = self.conversation_store.select(conversation_id)
            if (
                self.active_conversation is not None
                and conversation.conversation_id
                == self.active_conversation.conversation_id
            ):
                return
            self._switch_with_store_rollback(conversation)
        except Exception as error:
            self._report_error(
                "gui.conversation_select.failed",
                error,
                str(error),
            )

    def rename_conversation(self, conversation_id: str, name: str) -> None:
        """Rename one conversation without changing its saved state.

        Parameters
        ----------
        conversation_id: str
            Identifier of a conversation registered in the current project
            store.
        name: str
            Requested display name, normalized and validated by
            ConversationStore.

        Notes
        -----
        Operation errors are caught and reported through an error event. The
        method is synchronous; event-sink or error-reporting failures may still
        propagate.

        """

        try:
            if self.conversation_store is None:
                raise RuntimeError("Open a project first.")
            renamed = self.conversation_store.rename(
                conversation_id,
                name,
                ownership=self._store_ownership(conversation_id),
                state_writer=self._store_state_writer(conversation_id),
            )
            if (
                self.active_conversation is not None
                and renamed.conversation_id
                == self.active_conversation.conversation_id
            ):
                self.active_conversation = renamed
            self.active_trace.emit(
                "conversation.renamed",
                component="gui_engine",
                data=asdict(renamed),
            )
            self._send_conversations("conversations")
        except Exception as error:
            self._report_error(
                "gui.conversation_rename.failed",
                error,
                str(error),
            )

    def delete_conversation(self, conversation_id: str) -> None:
        """Archive one conversation and select a remaining conversation.

        The frontend is notified before trace cleanup so it can remove the
        row immediately. An active diagnostic trace is already durable after
        every accepted event, so deletion detaches it instead of appending a
        closing event that would rewrite a potentially large JSON document.

        Parameters
        ----------
        conversation_id: str
            Identifier of the saved conversation.

        """

        try:
            if (
                self.session is None
                or self.conversation_store is None
                or self.active_conversation is None
            ):
                raise RuntimeError("Open a project first.")
            deleting_active = (
                conversation_id == self.active_conversation.conversation_id
            )
            self._send(
                "conversation_delete_started",
                conversation_id=conversation_id,
            )
            replacement, archived_path = self.conversation_store.archive(
                conversation_id,
                ownership=self._store_ownership(conversation_id),
            )
            if deleting_active:
                self._detach_project_trace()
                self._switch_conversation(
                    replacement, close_current_trace=False
                )
            else:
                self._send_conversations("conversations")
            self.active_trace.emit(
                "conversation.deleted",
                component="gui_engine",
                data={
                    "conversation_id": conversation_id,
                    "archived_path": str(archived_path),
                },
            )
        except Exception as error:
            if self.conversation_store is not None:
                try:
                    self._send_conversations("conversations")
                except Exception:
                    pass
            self._report_error(
                "gui.conversation_delete.failed", error, str(error)
            )

    def rename_project(self, name: str) -> None:
        """Rename the active workspace folder, then reopen the same chat.

        Parameters
        ----------
        name: str
            Filename or project name to use.

        """

        source = destination = None
        conversation_id = None
        moved = False
        try:
            if self.session is None or self.active_conversation is None:
                raise RuntimeError("Open a project first.")
            clean = name.strip()
            if (
                not clean
                or clean in {".", ".."}
                or Path(clean).name != clean
                or "/" in clean
                or "\\" in clean
            ):
                raise ValueError("Use a single folder name without / or \\.")
            source = self.session.project.root
            destination = source.with_name(clean)
            if destination == source:
                return
            if destination.exists():
                raise ValueError(f"A folder named {clean!r} already exists.")
            conversation_id = self.active_conversation.conversation_id
            self.close()
            source.rename(destination)
            moved = True
            project = open_project(destination)
            self._reopen_project_conversation(project, conversation_id)
        except Exception as error:
            if source is not None and self.session is None:
                try:
                    self._close_project_trace()
                    if moved:
                        destination.rename(source)
                    self._reopen_project_conversation(
                        open_project(source), conversation_id
                    )
                except Exception:
                    self.close()
                    self._send(
                        "backend_failed",
                        message=(
                            "Could not reopen the project after renaming. "
                            "Reopen it from the saved projects list."
                        ),
                    )
            self._report_error(
                "gui.project.rename.failed",
                error,
                f"Could not rename the project: {error}",
            )

    def delete_project(self, directory: str) -> None:
        """Move a closed project into the deleted folder beside it.

        Sends the updated projects list, without the deleted project.

        Parameters
        ----------
        directory: str
            Folder containing the project's profile.ini.

        """

        try:
            folder = Path(directory).expanduser()
            if (
                self.session is not None
                and self.session.project.root == folder.resolve()
            ):
                raise RuntimeError("Close the project before deleting it.")
            delete_project(folder)
            self.list_saved_projects("")
        except Exception as error:
            self._report_error(
                "gui.project.delete.failed",
                error,
                f"Could not delete the project: {error}",
            )

    def apply_indexing_settings(self, settings: object) -> None:
        """Save the open project's indexing settings and rebuild its index.

        The index records these settings, so reopening the project with new
        values rebuilds it. On failure, restore a changed profile atomically
        and reopen the previous conversation if the session was replaced.

        Parameters
        ----------
        settings: object
            Changed values among ``extensions``, ``max_file_bytes``,
            ``target_tokens``, ``max_tokens`` and ``overlap_tokens``.

        Notes
        -----
        ``configure_project`` can fail after publishing ``profile.ini``.
        Recovery therefore compares the saved profile to the current file,
        not only whether configuration returned successfully. Index files
        are revalidated by reopening; they are not rolled back atomically.

        """

        root = None
        profile_text = None
        conversation_id = None
        reopen_started = False
        try:
            if self.session is None:
                raise RuntimeError("Open a project before changing indexing.")
            if (
                not isinstance(settings, dict)
                or not settings
                or not set(settings) <= _INDEXING_SETTINGS
            ):
                raise ValueError("Invalid indexing settings.")
            root = self.session.project.root
            profile_text = (root / PROFILE_FILE).read_text(encoding="utf-8")
            conversation_id = self.active_conversation.conversation_id
            self._send("status", message="Rebuilding the index…")
            # Validation precedes the core's atomic profile replacement.
            project = configure_project(
                root, settings, trace=self.active_trace
            )
            reopen_started = True
            self.close()
            self._reopen_project_conversation(project, conversation_id)
            self._send(
                "indexing_applied",
                profile=asdict(self.session.project.profile),
            )
        except Exception as error:
            recovery_error = None
            if root is not None and profile_text is not None:
                try:
                    path = root / PROFILE_FILE
                    # A post-write trace failure also needs a rollback.
                    if path.read_text(encoding="utf-8") != profile_text:
                        atomic_write_text(path, profile_text)
                    if reopen_started:
                        self.close()
                        self._reopen_project_conversation(
                            open_project(root), conversation_id
                        )
                except Exception as failure:
                    recovery_error = failure
            message = f"Could not apply indexing settings: {error}"
            if recovery_error is not None:
                try:
                    self.close()
                except Exception as close_error:
                    recovery_error = RuntimeError(
                        f"{recovery_error}; closing failed: {close_error}"
                    )
                    advice = "Restart KíLín, inspect profile.ini and reopen."
                else:
                    advice = "Inspect profile.ini and reopen the project."
                self._send(
                    "backend_failed",
                    message=(
                        "Could not complete indexing recovery. "
                        + advice
                    ),
                )
                message += f"; recovery failed: {recovery_error}. {advice}"
            self._report_error(
                "gui.indexing.change.failed",
                error,
                message,
            )

    def _switch_with_store_rollback(
        self,
        conversation: ConversationInfo,
    ) -> None:
        """Switch the active conversation or restore store selection.

        Parameters
        ----------
        conversation: ConversationInfo
            Target already selected or created by ConversationStore.

        Notes
        -----
        The store may change its active ID before the engine can acquire the
        target file. A rejected acquisition leaves the live session unchanged;
        restore the previous store selection in that case. If the session
        already switched, retain its new selection instead.

        """

        previous = self.active_conversation
        try:
            self._switch_conversation(conversation)
        except BaseException:
            if (
                previous is not None
                and self.active_conversation is previous
                and self.conversation_store is not None
            ):
                self.conversation_store.select(previous.conversation_id)
            raise

    def _switch_conversation(
        self,
        conversation: ConversationInfo,
        *,
        close_current_trace: bool = True,
    ) -> None:
        """Move the live session and trace to one conversation file.

        Parameters
        ----------
        conversation: ConversationInfo
            Registered conversation metadata for the state to activate or
            trace.
        close_current_trace: bool, optional
            Whether to append a closing event to the old trace after a
            successful switch. Deletion passes False after it already closed
            and detached its archived trace. Defaults to True.

        Raises
        ------
        RuntimeError
            If a loaded session or conversation store is unavailable.

        Notes
        -----
        Acquires the target trace before releasing the previous one. If the
        target is already owned or invalid, the old logger is retained. The
        loaded model/history reconfiguration is not a multi-step transaction.

        """

        if self.session is None or self.conversation_store is None:
            raise RuntimeError("Open a project first.")
        previous_trace = self.project_trace
        try:
            # Acquiring the target first leaves the active writer untouched
            # if the conversation is locked by another KíLín process.
            self._start_project_trace(
                self.session.project.root, conversation
            )
            turns = self.session.switch_conversation(
                self.conversation_store.state_path(conversation),
                trace=self.project_trace,
            )
        except BaseException:
            if self.project_trace is not previous_trace:
                if isinstance(self.project_trace, TraceLogger):
                    self.project_trace.close(emit_event=False)
                self.project_trace = previous_trace
            raise
        if close_current_trace and previous_trace is not self.project_trace:
            previous_trace.close()
        self.active_conversation = conversation
        self._use_project_answer_settings()
        self.active_trace.emit(
            "conversation.selected",
            component="gui_engine",
            data=asdict(conversation),
        )
        self._send_conversations(
            "conversation_opened",
            turns=_turn_payloads(turns, self.session.project.root),
            retrieval_count=(
                turns[-1].parameters.get(
                    "retrieval_count",
                    self.session.project.profile.retrieval_count,
                )
                if turns
                else self.session.project.profile.retrieval_count
            ),
            temperature=self.session.chat_session.temperature,
            thinking=self.session.chat_session.thinking,
            max_new_tokens=self.session.chat_session.max_new_tokens,
            usage=self._usage_totals(),
        )

    def _send_conversations(self, event: str, **data: object) -> None:
        """Send the catalog, active identity and project files together.

        Parameters
        ----------
        event: str
            Frontend event name stored under the event key.
        **data: object
            Additional fields merged into the emitted frontend event.

        Raises
        ------
        RuntimeError
            If no conversation store or active conversation is available.

        """

        if self.conversation_store is None or self.active_conversation is None:
            raise RuntimeError("Conversation store is not open.")
        # Creating, branching or deleting a chat adds or removes its file,
        # so the workspace tree is refreshed too.
        if self.session is not None:
            data.setdefault(
                "workspace_files", list_project_files(self.session.project)
            )
        self._send(
            event,
            active_conversation_id=(self.active_conversation.conversation_id),
            conversations=[
                {
                    **asdict(item),
                    "empty": self.conversation_store.is_empty(
                        item.conversation_id
                    ),
                }
                for item in self.conversation_store.conversations()
            ],
            **data,
        )

    def reset(self) -> None:
        """Create and activate a new empty conversation for the open project.

        Notes
        -----
        This GUI action does not call RAGSession.reset or archive/delete the
        old conversation. It creates a new registered record and emits reset
        with archived=None. The old conversation remains selectable.

        """

        try:
            if self.session is None or self.conversation_store is None:
                raise RuntimeError("Open a project first.")
            conversation = self.conversation_store.create()
            self._switch_with_store_rollback(conversation)
            self._send("reset", archived=None)
        except Exception as error:
            self._report_error(
                "gui.conversation_reset.failed",
                error,
                str(error),
            )

    def close(self) -> None:
        """Release the loaded backend session.

        Notes
        -----
        Closes the loaded RAG session and active trace, then clears
        conversation references. Backend cleanup exceptions are not caught
        here.

        """

        if self.session is not None:
            self.session.close()
            self.session = None
        self._close_project_trace()
        self.conversation_store = None
        self.active_conversation = None


def _reference_context(
    text: str,
    excerpt: str,
    start_line: int,
    end_line: int,
    pdf: bool,
    cell: int | None = None,
) -> dict[str, object]:
    """Find the retrieved text and return a small, numbered line window.

    PDF line numbers restart on each extracted page. Matching the saved excerpt
    also tells us which page to show without changing KíLín's stored evidence
    format.

    A notebook cell (``cell`` is not None) is never searched for in the raw
    ``.ipynb`` file: its JSON encoding has no line numbering a person would
    recognize, and ``start_line``/``end_line`` are local to that cell (every
    cell starts at line 1), so slicing the raw file's lines would show
    unrelated JSON structure or the wrong cell entirely. The saved excerpt is
    shown directly instead.

    Parameters
    ----------
    text: str
        Current extracted source text, with form-feed-separated pages for a
        PDF.
    excerpt: str
        Exact excerpt saved with the original evidence record.
    start_line: int
        Inclusive one-based first evidence line in its source coordinate
        system.
    end_line: int
        Inclusive one-based last evidence line in its source coordinate system.
    pdf: bool
        Whether source text represents page-separated PDF extraction.
    cell: int | None, optional
        Optional one-based notebook cell number. Notebook previews use the
        saved excerpt and cell-local line numbers. Defaults to ``None``.

    Returns
    -------
    dict[str, object]
        Dictionary describing numbered surrounding lines, highlighted evidence,
        optional page, and an explanatory note.

    Notes
    -----
    For PDFs, the first page containing the saved excerpt is used. Plain text
    falls back to the saved line range when it remains within the current file,
    even if the excerpt no longer matches. Notebook inputs use saved excerpts
    to preserve cell-local coordinates.

    """

    if cell is not None:
        return _saved_reference_context(
            excerpt,
            start_line,
            f"Notebook cell {cell}; line numbers are local to this cell, "
            "not the file.",
        )

    needle = excerpt.strip("\r\n")
    pages = text.split("\f") if pdf else [text]
    for page_number, page in enumerate(pages, start=1):
        position = page.find(needle) if needle else -1
        if position < 0:
            continue
        first = page[:position].count("\n") + 1
        last = first + needle.count("\n")
        return _numbered_reference_lines(
            page.splitlines(),
            first,
            last,
            page_number if pdf else None,
            "",
        )

    # Plain text keeps stable file-wide line numbers even if a reader has
    # normalized whitespace since the index was built.
    if not pdf and 1 <= start_line <= end_line <= len(pages[0].splitlines()):
        return _numbered_reference_lines(
            pages[0].splitlines(),
            start_line,
            end_line,
            None,
            "",
        )
    return _saved_reference_context(
        excerpt,
        start_line,
        "Showing the saved passage; surrounding file lines are unavailable.",
    )


def _numbered_reference_lines(
    lines: list[str],
    start_line: int,
    end_line: int,
    page: int | None,
    note: str,
) -> dict[str, object]:
    """Keep a little context around the cited range.

    Parameters
    ----------
    lines: list[str]
        Current source-file or PDF-page text split into lines.
    start_line: int
        Inclusive one-based first evidence line in its source coordinate
        system.
    end_line: int
        Inclusive one-based last evidence line in its source coordinate system.
    page: int | None
        Optional one-based PDF page number displayed with the preview.
    note: str
        Explanation shown with the preview when original source context is
        unavailable or uses special coordinates.

    Returns
    -------
    dict[str, object]
        Preview dictionary containing page, start_line, end_line, note, and
        numbered lines with highlighted flags.

    Notes
    -----
    Includes up to five surrounding lines on either side, clipped to the
    current text bounds.

    """

    first = max(1, start_line - 5)
    last = min(len(lines), end_line + 5)
    return {
        "page": page,
        "start_line": start_line,
        "end_line": end_line,
        "note": note,
        "lines": [
            {
                "number": number,
                "text": lines[number - 1],
                "highlighted": start_line <= number <= end_line,
            }
            for number in range(first, last + 1)
        ],
    }


def _saved_reference_context(
    excerpt: str,
    start_line: int,
    note: str,
) -> dict[str, object]:
    """Preserve a usable preview when the copied source cannot be read.

    Parameters
    ----------
    excerpt: str
        Exact excerpt saved with the original evidence record.
    start_line: int
        Inclusive one-based first evidence line in its source coordinate
        system.
    note: str
        Explanation shown with the preview when original source context is
        unavailable or uses special coordinates.

    Returns
    -------
    dict[str, object]
        Preview of the saved excerpt with line numbering beginning at
        start_line and the supplied explanatory note.

    """

    lines = excerpt.splitlines() or [excerpt]
    return {
        "page": None,
        "start_line": start_line,
        "end_line": start_line + len(lines) - 1,
        "note": note,
        "lines": [
            {
                "number": start_line + offset,
                "text": text,
                "highlighted": True,
            }
            for offset, text in enumerate(lines)
        ],
    }


def _model_size(model_id: str) -> float:
    """Parameter count in billions from a name such as "Qwen3-8B" (or 8).

    Parameters
    ----------
    model_id: str
        Model identifier from which to infer size.

    Returns
    -------
    float
        Parameter count in billions, defaulting to 8 for unknown names.

    """

    match = re.search(r"(\d+(?:\.\d+)?)B\b", model_id)
    return float(match.group(1)) if match else 8.0


def _setup_backend() -> str:
    """Resolve new-project setup from the saved backend preference.

    Returns
    -------
    str
        The selected backend, using platform policy only for ``auto``.

    Raises
    ------
    ValueError
        If an explicitly selected backend is unsupported on this platform.

    Notes
    -----
    The preference is read at the start of each setup operation. In
    particular, an explicit PyTorch choice must not become MLX merely
    because the automatic policy would select MLX.

    """

    return choose_backend(default_backend())


def _installed_replacements(
    missing: list[str], roles: dict[str, str], backend: str
) -> dict[str, str]:
    """Pick, for each missing model, the closest installed model.

    The replacement has the same role (language or embedding), suits the
    backend (MLX models have "mlx" in their name), and is closest in size.
    Missing models without such a replacement are left out.

    Parameters
    ----------
    missing: list[str]
        Identifiers of models that are not available locally.
    roles: dict[str, str]
        Model purposes associated with the missing identifiers.
    backend: str
        Model backend for which settings are requested.

    Returns
    -------
    dict[str, str]
        Mapping from missing model IDs to installed alternatives.

    """

    presets = {choice.model_id: choice.role for choice in MODEL_CHOICES}
    chosen: dict[str, str] = {}
    for model_id in missing:
        role = presets.get(model_id) or (
            "embedding" if "embed" in model_id.lower() else "chat"
        )
        candidates = [
            candidate
            for candidate, candidate_role in roles.items()
            if candidate_role == role
            and ("mlx" in candidate.lower()) == (backend == BACKEND_MLX)
        ]
        if candidates:
            size = _model_size(model_id)
            chosen[model_id] = min(
                candidates,
                key=lambda candidate: (
                    abs(math.log(_model_size(candidate) / size)),
                    candidate,
                ),
            )
    return chosen


def _preference_payload() -> dict[str, object]:
    """App-wide preferences for the page, by backend.

    Returns
    -------
    dict[str, object]
        Default GUI preferences organized by backend.

    """

    backends = (BACKEND_MLX, BACKEND_TORCH)
    return {
        "effort_levels": {b: default_effort_levels(b) for b in backends},
        "router_models": {b: default_router_model(b) for b in backends},
        "embedding_models": {b: default_embedding_model(b) for b in backends},
        "search_depth": default_search_depth(),
        "query_format": default_query_format(),
        "appearance": default_appearance(),
    }


# How long each answer took, as the page showed it, by turn timestamp; kept
# per project so resumed and branched chats show it again.
_ANSWER_TIMELINES: Final[str] = "answer_timelines.json"


def _read_answer_timelines(root: Path) -> dict[str, dict]:
    """Return the saved answer timings of a project, by turn timestamp.

    Parameters
    ----------
    root: pathlib.Path
        Project folder.

    Returns
    -------
    dict[str, dict]
        Timings saved by ``ChatEngine.save_answer_timeline``; empty when
        none are saved or the file cannot be read.

    """

    try:
        data = json.loads(
            (root / "logs" / _ANSWER_TIMELINES).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}

    return data if isinstance(data, dict) else {}


def _turn_payloads(turns: list, root: Path) -> list[dict]:
    """Return turns for the page, each with its saved timing if any.

    Parameters
    ----------
    turns: list[ConversationTurn]
        Turns of the open conversation.
    root: pathlib.Path
        Project folder holding the saved timings.

    Returns
    -------
    list[dict]
        The turns as dictionaries; a turn whose timing was saved has it
        under ``timeline``.

    """

    timelines = _read_answer_timelines(root)
    payloads = []
    for turn in turns:
        payload = asdict(turn)
        timeline = timelines.get(getattr(turn, "timestamp", ""))
        if _valid_timeline(timeline):
            payload["timeline"] = timeline
        payloads.append(payload)

    return payloads


def _valid_timeline(timeline: object) -> bool:
    """Whether a timeline from the page has the expected shape.

    Parameters
    ----------
    timeline: object
        Timeline data received from the frontend.

    Returns
    -------
    bool
        True if the timeline has the expected structure.

    """

    if not isinstance(timeline, dict):
        return False

    steps = timeline.get("steps")
    return (
        isinstance(timeline.get("elapsed_ms"), int)
        and isinstance(steps, list)
        and all(
            isinstance(step, dict)
            and isinstance(step.get("label"), str)
            and isinstance(step.get("ms"), int)
            for step in steps
        )
    )


def _path_list(value: object) -> list[str]:
    """Return a non-empty list of path strings, or raise ValueError.

    Parameters
    ----------
    value: object
        Value to validate or convert.

    Returns
    -------
    list[str]
        Nonempty list of file path strings.

    Raises
    ------
    ValueError
        If the input is not a nonempty list of path strings.

    """

    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError("Choose files or folders.")
    return value


_LEVEL_KEYS = ("light", "instant", "lagom", "thorough", "deep")


def _answer_excerpt(answer: str, limit: int) -> str:
    """Return answer text without thinking or citation markers.

    Parameters
    ----------
    answer: str
        Answer text to inspect.
    limit: int
        Maximum number of items or characters to include.

    Returns
    -------
    str
        answer text without thinking or citation markers.

    """

    text = re.sub(r"<think>.*?</think>", "", str(answer), flags=re.S)
    text = re.sub(r"\s*\[[0-9, ]+\]", "", text)
    return text.strip()[:limit]


def _router_models_by_backend(root: Path | None) -> dict[str, str]:
    """Return a project's router model (or the preferences) by backend.

    Parameters
    ----------
    root: Path | None
        Project or workspace root directory.

    Returns
    -------
    dict[str, str]
        a project's router model (or the preferences) by backend.

    """

    if root is None:
        return _preference_payload()["router_models"]
    return {
        backend: read_router_model(root, backend)
        for backend in (BACKEND_MLX, BACKEND_TORCH)
    }


def _effort_levels_by_backend(
    root: Path | None, profile: object
) -> dict[str, list[dict[str, object]]]:
    """Return a project's effort levels (or the preferences) by backend.

    Parameters
    ----------
    root: Path | None
        Project or workspace root directory.
    profile: object
        Current project profile, when available.

    Returns
    -------
    dict[str, list[dict[str, object]]]
        a project's effort levels (or the preferences) by backend.

    """

    if root is None:
        return _preference_payload()["effort_levels"]
    return {
        backend: read_effort_levels(root, backend)
        for backend in (BACKEND_MLX, BACKEND_TORCH)
    }


def _effort_indexes(effort: object) -> tuple[int, int] | None:
    """Read the effort level and search-depth step chosen for one answer.

    Parameters
    ----------
    effort: object
        Requested answer effort selection.

    Returns
    -------
    tuple[int, int] | None
        Zero-based effort and search-depth positions, or None.

    Raises
    ------
    ValueError
        If the supplied effort selection is invalid or out of range.

    """

    if effort is None:
        return None
    if isinstance(effort, str):
        if effort in _LEVEL_KEYS:
            index = _LEVEL_KEYS.index(effort)
            return index, index
        try:
            effort = json.loads(effort)
        except ValueError:
            effort = None
    if isinstance(effort, dict):
        # "retrieval" is the name earlier pages used for search depth.
        depth = effort.get("search_depth", effort.get("retrieval"))
        indexes = (effort.get("model"), depth)
        if all(
            type(index) is int and 0 <= index < LEVEL_COUNT
            for index in indexes
        ):
            return indexes
    raise ValueError("Choose an effort level and a search depth.")


def _workspace_path(sources: list[str], sandbox: Path, chosen: str) -> Path:
    """Suggest a preview location without reserving or reusing a workspace.

    Parameters
    ----------
    sources: list[str]
        Original source-path strings supplied by the frontend.
    sandbox: pathlib.Path
        Parent directory for default managed workspaces; it is not an execution
        sandbox.
    chosen: str
        Explicit workspace path string, or an empty string to derive one from
        the first source.

    Returns
    -------
    pathlib.Path
        Resolved explicit destination, or a deterministic source-name/path-hash
        destination beneath sandbox.

    Notes
    -----
    Deriving a default uses the core naming helper and first supplied source.
    Creation may select a numbered sibling. Preview does not reserve a path.

    """

    if chosen.strip():
        return Path(chosen).expanduser().resolve()
    return default_workspace_path(sources, sandbox)


def main() -> None:
    """Read one JSON command per line and emit one JSON event per line.

    Notes
    -----
    Reads one JSON object per input line and dispatches known actions
    sequentially. Unknown actions and ordinary exceptions are emitted as error
    events. Standard output is reserved for JSON; backend stdout/stderr are
    redirected into trace events. The engine is closed when input ends or the
    loop exits.

    """

    protocol_output = sys.stdout
    protocol_lock = Lock()

    def emit(data: dict[str, object]) -> None:
        """Write one frontend event as a flushed JSON line.

        Parameters
        ----------
        data: dict[str, object]
            Event dictionary serialized to the original protocol output stream.

        Notes
        -----
        This closure bypasses redirected diagnostic stdout so protocol framing
        is preserved. JSON conversion uses string fallback for unsupported
        objects.

        """

        with protocol_lock:
            protocol_output.write(json.dumps(data, default=str) + "\n")
            protocol_output.flush()

    sandbox = default_workspace_directory()
    # A reader thread passes stdin lines on through this queue (None ends
    # input), so Stop reaches a busy answer at once. One iterator over the
    # queue lets a tool plan waiting for the user's choice read the next
    # command from the same stream.
    commands: Queue[str | None] = Queue()

    def queued_lines() -> Iterator[str]:
        """Yield queued input lines until input ends.

        Yields
        ------
        str
            One raw command line.

        """

        while (line := commands.get()) is not None:
            yield line

    lines = queued_lines()

    def read_command() -> dict[str, object] | None:
        """Return the next JSON command, or None when input ended.

        Returns
        -------
        dict[str, object] | None
            the next JSON command, or None when input ended.

        """

        for line in lines:
            try:
                command = json.loads(line)
            except ValueError:
                emit({"event": "error", "message": "Unreadable command."})
                continue
            if isinstance(command, dict):
                return command
        return None

    engine = ChatEngine(
        sandbox, emit, trace_enabled=True, read_command=read_command
    )

    def read_input() -> None:
        """Queue stdin lines; apply Stop at once, even during an answer.

        Notes
        -----
        ``cancel_answer`` is also queued: a wait for a plan choice ends on
        it, and the main loop ignores it.

        """

        for line in sys.stdin:
            try:
                command = json.loads(line)
            except ValueError:
                command = None
            if (
                isinstance(command, dict)
                and command.get("action") == "cancel_answer"
            ):
                engine.cancel_answer()
            commands.put(line)
        commands.put(None)

    Thread(target=read_input, name="kilin-input", daemon=True).start()
    standard_output = TraceTextStream(
        lambda: engine.active_trace,
        stream_name="backend.stdout",
    )
    standard_error = TraceTextStream(
        lambda: engine.active_trace,
        stream_name="backend.stderr",
    )
    try:
        for line in lines:
            try:
                command = json.loads(line)
                if not isinstance(command, dict):
                    raise ValueError("A GUI command must be an object.")
                with (
                    redirect_stdout(standard_output),
                    redirect_stderr(standard_error),
                ):
                    action = command.get("action")
                    if action == "cancel_answer":
                        # Applied by the input thread when it arrived.
                        pass
                    elif action == "open":
                        engine.open_project(
                            command["sources"],
                            command["type_id"],
                            command["workspace"],
                            command.get("selected_files"),
                            command.get("settings"),
                        )
                    elif action == "settings":
                        engine.settings(
                            command["type_id"], command["workspace"]
                        )
                    elif action == "apply_models":
                        engine.apply_model_settings(command["settings"])
                    elif action == "download_model":
                        engine.download(command["model_id"])
                    elif action == "startup":
                        engine.startup()
                    elif action == "readiness":
                        engine.readiness()
                    elif action == "install_required_models":
                        engine.install_required_models()
                    elif action == "use_installed_models":
                        engine.use_installed_models()
                    elif action == "saved_project":
                        engine.find_saved_project(
                            command["type_id"], command["workspace"]
                        )
                    elif action == "saved_projects":
                        engine.list_saved_projects(command["workspace"])
                    elif action == "resume":
                        engine.resume_project(
                            command["project"], command.get("conversation_id")
                        )
                    elif action == "close":
                        engine.close()
                    elif action == "preview":
                        engine.preview(
                            command["sources"],
                            command["extensions"],
                            command["max_file_bytes"],
                            command.get("workspace", ""),
                        )
                    elif action == "preview_reference":
                        engine.preview_reference(
                            command["turn_index"],
                            command["citation"],
                        )
                    elif action == "ask":
                        engine.ask(
                            command["question"],
                            command.get("retrieval_count"),
                            command.get("generation_settings"),
                            command.get("effort"),
                        )
                    elif action == "new_conversation":
                        engine.create_conversation()
                    elif action == "branch_conversation":
                        engine.create_conversation(
                            branch=True, turn_count=command.get("turn_count")
                        )
                    elif action == "select_conversation":
                        engine.select_conversation(command["conversation_id"])
                    elif action == "rename_conversation":
                        engine.rename_conversation(
                            command["conversation_id"],
                            command["name"],
                        )
                    elif action == "delete_conversation":
                        engine.delete_conversation(command["conversation_id"])
                    elif action == "check_sources":
                        engine.check_sources(command["sources"])
                    elif action == "add_project_sources":
                        engine.add_project_sources(
                            command["sources"], command.get("decisions")
                        )
                    elif action == "remove_project_sources":
                        engine.remove_project_sources(command["paths"])
                    elif action == "sync_sources":
                        engine.sync_sources(command.get("decisions"))
                    elif action == "edit_folder":
                        arguments = dict(command)
                        del arguments["action"]
                        engine.edit_folder(
                            str(arguments.pop("folder", "")),
                            str(arguments.pop("operation", "")),
                            **arguments,
                        )
                    elif action == "rename_project":
                        engine.rename_project(command["name"])
                    elif action == "delete_project":
                        engine.delete_project(command["project"])
                    elif action == "apply_indexing":
                        engine.apply_indexing_settings(command["settings"])
                    elif action == "save_answer_settings":
                        engine.save_answer_settings(
                            command.get("settings"),
                            command.get("request_id"),
                            command.get("project"),
                        )
                    elif action == "tool_settings":
                        engine.tool_settings()
                    elif action == "save_tool_settings":
                        engine.save_tool_settings(
                            command["tool"], command.get("settings")
                        )
                    elif action == "save_answer_timeline":
                        engine.save_answer_timeline(
                            str(command.get("timestamp", "")),
                            command.get("timeline"),
                        )
                    elif action == "reset":
                        engine.reset()
                    elif action == "plan_choice":
                        # Only meaningful while a plan waits; ignore late
                        # clicks after the answer finished.
                        pass
                    else:
                        emit({"event": "error", "message": "Unknown command."})
                standard_output.flush()
                standard_error.flush()
            except Exception as error:
                engine.active_trace.exception(
                    "gui.command.failed",
                    error,
                    component="gui_engine",
                )
                emit(
                    {
                        "event": "error",
                        "message": f"The background task failed: {error}",
                    }
                )
    finally:
        standard_output.flush()
        standard_error.flush()
        engine.close()


if __name__ == "__main__":
    main()
