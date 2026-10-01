"""Run repository and model operations behind the desktop interface.

``ChatEngine`` converts frontend requests into synchronous QAChat operations
and emits structured status, progress, result, and error events. The module's
entry point reads newline-delimited JSON commands from standard input and
writes newline-delimited JSON events to standard output. Backend diagnostic
output is redirected into trace events so it does not corrupt that protocol.

A workspace is a managed directory containing copied sources, configuration,
indexes, and conversations. It is not an operating-system execution sandbox.
The engine can download a catalog model when explicitly requested and can
load models according to project settings. The process entry point enables
forensic tracing, which can retain conversation and source content.

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
>>> import qachat.gui.engine
>>> from pathlib import Path
>>> events = []
>>> engine = qachat.gui.engine.ChatEngine(
...     Path("/path/to/workspaces"), events.append,
... )
>>> engine.close()

This constructs the coordinator without loading a model. Use the desktop
interface or the documented engine methods to open a project.

"""

from __future__ import annotations

from configparser import ConfigParser, Error as ConfigError
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Callable

from qachat.gui.model_catalog import (
    MODEL_CHOICES,
    installed_backends,
    model_choices,
)
from qachat import RAGSession, prepare_sources_workspace
from qachat.conversations import ConversationInfo, ConversationStore
from qachat.huggingface import available_models, download_model
from qachat.project import (
    PROJECT_KIND_CODE,
    PROJECT_KIND_CHOICES,
    QAProject,
    default_project_profile,
    open_project,
)
from qachat.repository import (
    BLOCKED_SUFFIXES,
    DEFAULT_EXTENSIONS,
    SKIP_FILES,
    scan_repository,
)
from qachat.runtime import BACKEND_AUTO, choose_backend
from qachat.sources import read_repository_files
from qachat.trace_logging import (
    NullTraceLogger,
    TraceConfig,
    TraceLogger,
    TraceSink,
    TraceTextStream,
)

ROOT = Path(__file__).resolve().parents[2]
EventSink = Callable[[dict[str, object]], None]


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
        Whether activated conversations use forensic tracing. The process entry
        point passes True even though the constructor default is False.

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
    startup()
        Suggest the most recently used valid sandbox project.
    find_saved_project(kind, workspace)
        Refresh Resume for the selected type and workspace folder.
    resume_project(directory)
        Reopen a saved project without copying source files.
    settings(kind, workspace)
        Return saved or default settings and cached model IDs.
    download(model_id)
        Download only a model explicitly chosen from the GUI catalog.
    preview(sources, extensions, max_file_bytes, workspace='')
        List eligible files before copying any source.
    open_project(sources, kind, workspace, selected_files=None, settings=None)
        Add selected sources and open one QAChat workspace.
    ask(question, retrieval_count=None, generation_settings=None)
        Ask the loaded project and return its saved answer and evidence.
    preview_reference(turn_index, citation)
        Show nearby source lines for one saved answer citation.
    create_conversation(*, branch=False)
        Create an empty conversation or branch the selected conversation.
    select_conversation(conversation_id)
        Switch conversations while retaining loaded project resources.
    rename_conversation(conversation_id, name)
        Rename one conversation without changing its saved state.
    reset()
        Create and activate a new empty conversation for the open project.
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
    ) -> None:
        """Keep one session and send structured updates to the window.

        Parameters
        ----------
        sandbox: pathlib.Path
            Default parent folder for local project workspaces.
        emit: EventSink
            Callback that forwards an event to the web page.
        trace_enabled: bool, optional
            Whether each opened project receives its own forensic trace.
            Defaults to ``False``.

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
        forensic logger at the conversation state path and emits trace_ready.
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
                mode=TRACE_MODE_FORENSIC,
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
        """Finish the current project trace without closing the GUI trace.


        """

        if self.project_trace.enabled:
            self.project_trace.close()
        self.project_trace = NullTraceLogger()

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
        )

    def startup(self) -> None:
        """Suggest the most recently used valid sandbox project.


        """

        project = self._find_saved_project(None, "")
        if project is not None:
            folder, kind = project
            self._send("startup", project=str(folder), kind=kind)
        else:
            self._send("startup", project=None)

    def find_saved_project(self, kind: str, workspace: str) -> None:
        """Refresh Resume for the selected type and workspace folder.

        A chosen workspace is checked directly. Without one, the most recent
        valid sandbox project of the selected type is suggested.

        Parameters
        ----------
        kind: str
            Requested project category: code or literature. Discovery helpers
            also accept None to search either kind.
        workspace: str
            Explicit workspace path string. An empty string requests discovery
            or default allocation under the engine workspace root.

        """

        project = self._find_saved_project(kind, workspace)
        self._send(
            "saved_project",
            project=str(project[0]) if project is not None else None,
            kind=kind,
            workspace=workspace,
        )

    def _find_saved_project(
        self, kind: str | None, workspace: str
    ) -> tuple[Path, str] | None:
        """Find a current project, ignoring deleted or incomplete folders.

        Parameters
        ----------
        kind: str | None
            Requested project category: code or literature. Discovery helpers
            also accept None to search either kind.
        workspace: str
            Explicit workspace path string. An empty string requests discovery
            or default allocation under the engine workspace root.

        Returns
        -------
        tuple[pathlib.Path, str] | None
            Pair (workspace_path, project_kind) for the most recently used
            valid candidate, or None.

        Notes
        -----
        Discovery inspects profile/source existence and profile/conversation
        modification times without loading a model. Invalid candidate profiles
        are skipped; the whole project schema is checked later when opening.

        """

        if kind is not None and kind not in PROJECT_KIND_CHOICES:
            return None
        if workspace.strip():
            folders = [Path(workspace).expanduser()]
        elif self.sandbox.is_dir():
            folders = list(self.sandbox.iterdir())
        else:
            return None

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
                saved_kind = parser.get("project", "kind")
                source_text = parser.get("project", "source")
                if saved_kind not in PROJECT_KIND_CHOICES:
                    continue
                if kind is not None and saved_kind != kind:
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
                projects.append((recent, folder, saved_kind))
            except (ConfigError, OSError, ValueError):
                continue
        if not projects:
            return None
        _, folder, saved_kind = max(
            projects, key=lambda item: (item[0], str(item[1]))
        )
        return folder, saved_kind

    def resume_project(self, directory: str) -> None:
        """Reopen a saved project without copying source files.

        Parameters
        ----------
        directory: str
            Workspace containing an existing profile.ini.

        """

        try:
            project = open_project(directory)
            self.close()
            conversation = self._open_conversation_store(project)
            self._start_project_trace(project.root, conversation)
            self._activate_project(project, conversation)
        except Exception as error:
            self._report_error(
                "gui.project.resume.failed",
                error,
                f"Could not resume the project: {error}",
            )
            if self.session is None:
                self._close_project_trace()

    def settings(self, kind: str, workspace: str) -> None:
        """Return saved or default settings and cached model IDs.

        Parameters
        ----------
        kind: str
            Code or literature project.
        workspace: str
            Existing workspace path, or blank for new-project defaults.

        """

        try:
            profile_path = Path(workspace).expanduser() / "profile.ini"
            profile = (
                open_project(profile_path.parent).profile
                if workspace.strip() and profile_path.is_file()
                else default_project_profile(kind)
            )
            cached = {str(item["model_id"]) for item in available_models()}
            self._send(
                "settings",
                profile=asdict(profile),
                backend=choose_backend(BACKEND_AUTO),
                installed_backends=installed_backends(),
                supported_extensions=list(DEFAULT_EXTENSIONS),
                cached_models=sorted(cached),
                model_choices=model_choices(cached),
                workspace_files=(
                    _workspace_files(profile_path.parent)
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
        Only model IDs in MODEL_CHOICES are accepted. This explicit action can
        contact Hugging Face and write to its cache. Success emits refreshed
        model status; download/validation failures become error events.

        """

        try:
            if model_id not in {choice.model_id for choice in MODEL_CHOICES}:
                raise ValueError("Choose a suggested model to download.")
            self._send("model_download_started", model_id=model_id)
            download_model(model_id)
            self._send("model_downloaded", model_id=model_id)
        except Exception as error:
            self._report_error(
                "gui.model_download.failed",
                error,
                f"Could not download {model_id}: {error}",
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
            Additional text file types requested by the user.
        max_file_bytes: int
            Largest file that may be copied.
        workspace: str, optional
            Chosen workspace path, used to avoid scanning itself. Defaults to
            ``''``.

        Notes
        -----
        Builds selectable and skipped source-entry lists without copying files
        or loading models. Oversized files are reported separately for frontend
        selection/limit adjustment. Directory selection uses scan_repository;
        direct files have explicit protected-name checks. Failures become error
        events.

        """

        try:
            roots = []
            allowed_extensions = tuple(
                dict.fromkeys((*DEFAULT_EXTENSIONS, *extensions))
            )
            destination = _workspace_path(sources, self.sandbox, workspace)
            for value in sources:
                original = Path(value).expanduser()
                if original.is_symlink():
                    raise ValueError(f"Do not add a symbolic link: {original}")
                source = original.resolve(strict=True)
                if source.is_file():
                    size = source.stat().st_size
                    name = source.name.lower()
                    if name in SKIP_FILES or name.startswith(".env."):
                        reason = "potentially secret or generated file"
                    elif source.suffix.lower() in BLOCKED_SUFFIXES:
                        reason = "model weights or potentially secret file"
                    elif not name.endswith(allowed_extensions):
                        reason = "file type not selected"
                    elif size == 0:
                        reason = "empty file"
                    elif size > max_file_bytes:
                        reason = "larger than the file-size limit"
                    else:
                        reason = None
                    files = [] if reason else [source.name]
                    skipped = (
                        []
                        if reason is None
                        else [
                            {
                                "path": source.name,
                                "reason": reason,
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
        kind: str,
        workspace: str,
        selected_files: dict[str, list[str]] | None = None,
        settings: dict[str, object] | None = None,
    ) -> None:
        """Add selected sources and open one QAChat workspace.

        The selected source snapshot is prepared before models are loaded. An
        empty source list can reopen a managed workspace unchanged.

        Parameters
        ----------
        sources: list[str]
            Source files and folders to copy into the workspace.
        kind: str
            Code or literature project.
        workspace: str
            Workspace folder, or blank to use the sandbox.
        selected_files: dict[str, list[str]] | None, optional
            Checked file paths within each source folder. Defaults to ``None``.
        settings: dict[str, object] | None, optional
            Preparation settings saved in profile.ini. Defaults to ``None``.

        Notes
        -----
        Closes any current session before preparation. Selected file suffixes
        can extend the saved extension list; copying uses the cumulative
        multi-source workspace API. With neither sources nor workspace, the
        implementation uses a development example repository path that is not
        included in this archive. Errors become frontend events.

        """

        try:
            if kind not in PROJECT_KIND_CHOICES:
                raise ValueError("Choose code or literature.")
            if not sources and not workspace.strip():
                example = (
                    "code_repo"
                    if kind == PROJECT_KIND_CODE
                    else "literature_repo"
                )
                sources = [str(ROOT / "example_repos" / example)]
                selected_files = None
            if not workspace.strip():
                workspace = str(_workspace_path(sources, self.sandbox, ""))
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
            self.close()
            project_directory = Path(workspace).expanduser().resolve()
            conversation: ConversationInfo | None = None
            if self.trace_enabled:
                store = ConversationStore(project_directory)
                conversation = store.ensure()
                self.conversation_store = store
                self.active_conversation = conversation
                self._start_project_trace(project_directory, conversation)
            self._send("status", message="Preparing the project…")
            options: dict[str, object] = {}
            if selected_files is not None:
                options["selected_files"] = selected_files
            if settings:
                settings = dict(settings)
                configured = settings.get("extensions", ())
                if selected_files and isinstance(configured, (list, tuple)):
                    # A checked file may use a type absent from an old profile.
                    known = tuple(
                        dict.fromkeys((*DEFAULT_EXTENSIONS, *configured))
                    )
                    selected_types = [
                        extension
                        for paths in selected_files.values()
                        for path in paths
                        for extension in known
                        if Path(path).name.lower().endswith(extension)
                    ]
                    settings["extensions"] = list(
                        dict.fromkeys((*configured, *selected_types))
                    )
                options["settings"] = settings
            if self.project_trace.enabled:
                options["trace"] = self.project_trace
            project = prepare_sources_workspace(
                sources,
                kind,
                workspace,
                **options,
            )
            if conversation is None:
                conversation = self._open_conversation_store(project)
            self._activate_project(project, conversation)
        except Exception as error:
            self._report_error(
                "gui.project.prepare.failed",
                error,
                f"Could not prepare the project: {error}",
            )
            if self.session is None:
                self._close_project_trace()

    def _open_conversation_store(
        self,
        project: QAProject,
    ) -> ConversationInfo:
        """Scan project conversation files and select the latest one.

        Parameters
        ----------
        project: QAProject
            Opened project whose profile, source, index, and log paths are
            ready for activation.

        Returns
        -------
        ConversationInfo
            The store's selected initial ConversationInfo, also assigned as the
            active conversation.

        """

        store = ConversationStore(project.root)
        conversation = store.ensure()
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
        options["conversation_path"] = self.conversation_store.state_path(
            conversation
        )
        session = RAGSession(project.root, **options)
        self.session = session
        self._send(
            "opened",
            project=str(project.root),
            source=str(project.source),
            kind=project.profile.kind,
            index_status=session.index_status,
            turns=[asdict(turn) for turn in session.turns],
            workspace_files=_workspace_files(project.root),
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
                asdict(item)
                for item in self.conversation_store.conversations()
            ],
        )

    def ask(
        self,
        question: str,
        retrieval_count: int | None = None,
        generation_settings: dict[str, object] | None = None,
    ) -> None:
        """Ask the loaded project and return its saved answer and evidence.

        Conversation settings affect the next answer without rebuilding the
        project indexes or editing profile.ini.

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
            self._send("status", message="Finding evidence and answering…")
            options: dict[str, object] = {}
            if retrieval_count is not None:
                options["retrieval_count"] = retrieval_count
            if generation_settings is not None:
                options["generation_settings"] = generation_settings
            options["phase_callback"] = lambda phase, details: self._send(
                "answer_phase",
                phase=phase,
                **details,
            )
            self.session.ask(question, **options)
            if (
                self.conversation_store is not None
                and self.active_conversation is not None
            ):
                self.active_conversation = self.conversation_store.touch(
                    self.active_conversation.conversation_id
                )
                self.active_trace.update_conversation(
                    asdict(self.active_conversation)
                )
            self._send("answered", turn=asdict(self.session.turns[-1]))
        except Exception as error:
            self._report_error(
                "gui.question.failed",
                error,
                str(error),
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

    def create_conversation(self, *, branch: bool = False) -> None:
        """Create an empty conversation or branch the selected conversation.

        Parameters
        ----------
        branch: bool, optional
            Whether to copy resumable state from the active conversation
            instead of creating an empty conversation. Defaults to ``False``.

        Notes
        -----
        Creates a new or branched store record, then activates it while reusing
        loaded models. The previous conversation remains available. Errors are
        reported through the frontend error channel.

        """

        try:
            if (
                self.session is None
                or self.conversation_store is None
                or self.active_conversation is None
            ):
                raise RuntimeError("Open a project first.")
            parent_id = (
                self.active_conversation.conversation_id if branch else None
            )
            conversation = self.conversation_store.create(
                branch_from=parent_id
            )
            self._switch_conversation(conversation)
        except Exception as error:
            self._report_error(
                "gui.conversation_create.failed",
                error,
                str(error),
            )

    def select_conversation(self, conversation_id: str) -> None:
        """Switch conversations while retaining loaded project resources.

        Parameters
        ----------
        conversation_id: str
            Identifier of a conversation registered in the current project
            store.

        Notes
        -----
        Operation errors are caught and reported through an error event. The
        method is synchronous; event-sink or error-reporting failures may still
        propagate.

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
            self._switch_conversation(conversation)
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
            renamed = self.conversation_store.rename(conversation_id, name)
            if (
                self.active_conversation is not None
                and renamed.conversation_id
                == self.active_conversation.conversation_id
            ):
                self.active_conversation = renamed
                self.active_trace.update_conversation(asdict(renamed))
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

    def _switch_conversation(self, conversation: ConversationInfo) -> None:
        """Move the live session and trace to one conversation file.

        Parameters
        ----------
        conversation: ConversationInfo
            Registered conversation metadata for the state to activate or
            trace.

        Raises
        ------
        RuntimeError
            If a loaded session or conversation store is unavailable.

        Notes
        -----
        Closes the previous trace, opens the target trace, and switches RAG
        history/settings while reusing models and indexes. This sequence is not
        rolled back as one transaction on failure.

        """

        if self.session is None or self.conversation_store is None:
            raise RuntimeError("Open a project first.")
        self._close_project_trace()
        self._start_project_trace(self.session.project.root, conversation)
        turns = self.session.switch_conversation(
            self.conversation_store.state_path(conversation),
            trace=self.project_trace,
        )
        self.active_conversation = conversation
        self.active_trace.emit(
            "conversation.selected",
            component="gui_engine",
            data=asdict(conversation),
        )
        self._send_conversations(
            "conversation_opened",
            turns=[asdict(turn) for turn in turns],
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
        )

    def _send_conversations(self, event: str, **data: object) -> None:
        """Send the catalog and active identity in one consistent shape.

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
        self._send(
            event,
            active_conversation_id=(self.active_conversation.conversation_id),
            conversations=[
                asdict(item)
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
            self._switch_conversation(conversation)
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
    also tells us which page to show without changing QAChat's stored evidence
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


def _workspace_files(root: Path) -> list[str]:
    """List saved workspace files for the sidebar.

    Parameters
    ----------
    root: pathlib.Path
        Managed project directory whose sources, indexes, and logs are listed.

    Returns
    -------
    list[str]
        Mapping of managed workspace areas to their relative file listings.

    Notes
    -----
    The listing is capped by the implementation entry limit. It is a UI
    inventory, not a selection-policy or content-integrity check.

    """

    files = ["profile.ini"]
    for folder in (root / "sources", root / "indexes", root / "logs"):
        if folder.is_dir():
            files.extend(
                path.relative_to(root).as_posix()
                for path in folder.rglob("*")
                if path.is_file()
            )
    return sorted(files)[:10_000]


def _workspace_path(sources: list[str], sandbox: Path, chosen: str) -> Path:
    """Resolve the same workspace location for preview and preparation.

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
    Deriving a default uses the first supplied source. The returned name does
    not establish isolation or ownership by itself.

    """

    if chosen.strip():
        return Path(chosen).expanduser().resolve()
    source = Path(sources[0]).expanduser().resolve()
    digest = sha256(str(source).encode()).hexdigest()[:8]
    return (sandbox / f"{source.stem}-{digest}").resolve()


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

        protocol_output.write(json.dumps(data, default=str) + "\n")
        protocol_output.flush()

    sandbox = ROOT / "sandbox"
    engine = ChatEngine(sandbox, emit, trace_enabled=True)
    standard_output = TraceTextStream(
        lambda: engine.active_trace,
        stream_name="backend.stdout",
    )
    standard_error = TraceTextStream(
        lambda: engine.active_trace,
        stream_name="backend.stderr",
    )
    try:
        for line in sys.stdin:
            try:
                command = json.loads(line)
                if not isinstance(command, dict):
                    raise ValueError("A GUI command must be an object.")
                with (
                    redirect_stdout(standard_output),
                    redirect_stderr(standard_error),
                ):
                    action = command.get("action")
                    if action == "open":
                        engine.open_project(
                            command["sources"],
                            command["kind"],
                            command["workspace"],
                            command.get("selected_files"),
                            command.get("settings"),
                        )
                    elif action == "settings":
                        engine.settings(command["kind"], command["workspace"])
                    elif action == "download_model":
                        engine.download(command["model_id"])
                    elif action == "startup":
                        engine.startup()
                    elif action == "saved_project":
                        engine.find_saved_project(
                            command["kind"], command["workspace"]
                        )
                    elif action == "resume":
                        engine.resume_project(command["project"])
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
                        )
                    elif action == "new_conversation":
                        engine.create_conversation()
                    elif action == "branch_conversation":
                        engine.create_conversation(branch=True)
                    elif action == "select_conversation":
                        engine.select_conversation(command["conversation_id"])
                    elif action == "rename_conversation":
                        engine.rename_conversation(
                            command["conversation_id"],
                            command["name"],
                        )
                    elif action == "reset":
                        engine.reset()
                    else:
                        emit({"event": "error", "message": "Unknown command."})
                standard_output.flush()
                standard_error.flush()
            except (KeyError, TypeError, ValueError) as error:
                engine.active_trace.exception(
                    "gui.command.invalid",
                    error,
                    component="gui_engine",
                )
                emit({"event": "error", "message": str(error)})
    finally:
        standard_output.flush()
        standard_error.flush()
        engine.close()


if __name__ == "__main__":
    main()
