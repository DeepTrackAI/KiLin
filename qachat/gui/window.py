"""Host QAChat's local web interface in a Qt desktop window.

``ChatWindow`` connects a local HTML page to ``ChatBridge`` through Qt's web
channel and routes bridge commands to a separate Python engine process.
``SourceDropView`` forwards local drag-and-drop paths to the page. The engine
owns repository access, model loading, generation, and conversation storage.

The page's local-content access to remote URLs is disabled. This setting is
not a network prohibition for the engine, which may download model files.
Stopping a task kills the engine process, discards queued commands, and
starts a fresh process; it is not a resumable model-generation pause.

Key Features
------------

- **Qt/web bridge**

    Translate JavaScript slot calls into structured engine commands and relay
    event signals back to the page.

- **Local source selection**

    Support native file dialogs and drag-and-drop of local paths.

- **Process lifecycle**

    Buffer JSON-line responses and manage launch, stop, restart, and shutdown.

Module Structure
----------------

- ``ChatBridge``

    Expose Qt signals and slots connecting the page to backend commands.

- ``SourceDropView``

    Accept only local files and folders dropped onto the window.

- ``ChatWindow``

    Display the web page while owning one stoppable backend process.

Examples
--------
Start the window through the application entry point rather than constructing
Qt widgets without an application:

>>> import qachat.gui.app
>>> exit_code = qachat.gui.app.main(["qachat"])

"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from PySide6.QtCore import QObject, QProcess, QUrl, Signal, Slot
from PySide6.QtGui import (
    QCloseEvent,
    QCursor,
    QDesktopServices,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
)
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMenu, QWidget

ROOT = Path(__file__).resolve().parents[2]


class ChatBridge(QObject):
    """Expose Qt signals and slots connecting the page to backend commands.

    Attributes
    ----------
    event: Signal(str)
        Serialized engine event sent back to the web page.
    pathsSelected: Signal(list)
        Local source paths chosen using native dialogs.
    workspaceSelected: Signal(str)
        Selected workspace directory.
    commandRequested: Signal(dict)
        Backend command handed to the window process controller.
    stopRequested: Signal()
        Request to terminate the active engine process.

    Methods
    -------
    chooseSources()
        Offer files and folders through one browse control.
    chooseFolder()
        Show the native folder picker on the GUI thread.
    chooseFiles()
        Add one or more individual files.
    chooseWorkspace()
        Choose the folder in which QAChat stores its project.
    requestSettings(kind, workspace)
        Read the existing profile or new-project defaults.
    downloadModel(model_id)
        Download one suggested model only after the user chooses it.
    previewSources(sources_json, extensions_json, max_file_bytes, workspace)
        Scan selected roots without copying them.
    openProject(sources_json, kind, workspace, selected_json, settings_json)
        Request project preparation in the backend process.
    askQuestion(question, effort, settings_json)
        Request one answer with the selected GUI effort preset.
    previewReference(turn_index, citation)
        Request source context for one saved answer citation.
    resetConversation()
        Archive the current conversation in the backend process.
    newConversation()
        Create and open a separate empty conversation.
    branchConversation()
        Branch the active conversation at its current saved checkpoint.
    selectConversation(conversation_id)
        Open a saved conversation without reloading project resources.
    renameConversation(conversation_id, name)
        Rename a saved conversation in the project catalog.
    deleteConversation(conversation_id)
        Archive a conversation and remove it from the project catalog.
    listSavedProjects(workspace)
        List valid saved code and literature projects.
    resumeProject(directory)
        Open a saved project without changing its source snapshot.
    resumeProjectConversation(directory, conversation_id)
        Reopen a saved project at one exact conversation.
    closeProject()
        Release models when the user chooses a different project.
    stopOperation()
        Terminate the active backend process immediately.

    Notes
    -----
    Slots preserve JavaScript-facing camelCase names. They primarily decode
    JSON and emit commands; model and filesystem policy is enforced in the
    engine. Invalid JSON may propagate from decoding rather than becoming a
    frontend error here.

    """

    event = Signal(str)
    pathsSelected = Signal(list)
    attachmentsSelected = Signal(list)
    workspaceSelected = Signal(str)
    commandRequested = Signal(dict)
    stopRequested = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        """Keep only the currently opened project path on the UI side."""

        super().__init__(parent)
        self.project_root: Path | None = None

    @Slot()
    def chooseSources(self) -> None:
        """Offer files and folders through one browse control.


        """

        menu = QMenu()
        folder = menu.addAction("Choose a folder…")
        files = menu.addAction("Choose files…")
        chosen = menu.exec(QCursor.pos())
        if chosen == folder:
            self.chooseFolder()
        elif chosen == files:
            self.chooseFiles()

    @Slot()
    def chooseFolder(self) -> None:
        """Show the native folder picker on the GUI thread.


        """

        path = QFileDialog.getExistingDirectory(None, "Choose project folder")
        if path:
            self.pathsSelected.emit([path])

    @Slot()
    def chooseFiles(self) -> None:
        """Add one or more individual files.


        """

        paths, _ = QFileDialog.getOpenFileNames(None, "Choose source files")
        if paths:
            self.pathsSelected.emit(paths)

    @Slot()
    def chooseAttachmentFolder(self) -> None:
        """Choose one folder for the current chat message."""

        path = QFileDialog.getExistingDirectory(None, "Attach folder")
        if path:
            self.attachmentsSelected.emit([path])

    @Slot()
    def chooseAttachmentFiles(self) -> None:
        """Choose one or more files for the current chat message."""

        paths, _ = QFileDialog.getOpenFileNames(None, "Attach files")
        if paths:
            self.attachmentsSelected.emit(paths)

    @Slot()
    def chooseWorkspace(self) -> None:
        """Choose the folder in which QAChat stores its project.


        """

        path = QFileDialog.getExistingDirectory(
            None, "Choose QAChat workspace"
        )
        if path:
            self.workspaceSelected.emit(path)

    @Slot(str, str)
    def requestSettings(self, kind: str, workspace: str) -> None:
        """Read the existing profile or new-project defaults.

        ``workspace`` is blank when the user is preparing a new project.

        Parameters
        ----------
        kind: str
            Project category string forwarded to the backend engine.
        workspace: str
            Workspace path string, possibly empty when the engine should
            discover or allocate a default.

        """

        self.commandRequested.emit(
            {"action": "settings", "kind": kind, "workspace": workspace}
        )

    @Slot(str)
    def downloadModel(self, model_id: str) -> None:
        """Download one suggested model only after the user chooses it.

        Parameters
        ----------
        model_id: str
            Catalog model identifier forwarded for explicit download.

        """

        self.commandRequested.emit(
            {"action": "download_model", "model_id": model_id}
        )

    @Slot(str, str, int, str)
    def previewSources(
        self,
        sources_json: str,
        extensions_json: str,
        max_file_bytes: int,
        workspace: str,
    ) -> None:
        """Scan selected roots without copying them.

        The source and extension lists arrive as JSON from the web page.

        Parameters
        ----------
        sources_json: str
            JSON-encoded list of source-path strings from the web page.
        extensions_json: str
            JSON-encoded list of literal source filename suffixes.
        max_file_bytes: int
            File-size selection limit in bytes forwarded to source preview.
        workspace: str
            Workspace path string, possibly empty when the engine should
            discover or allocate a default.

        Raises
        ------
        json.JSONDecodeError
            If a supplied JSON string cannot be decoded.

        Notes
        -----
        Decoding and signal emission occur in the GUI process. The engine
        performs the operation later. The openProject slot separately emits an
        error for a decoded source value that is not a list of strings.

        """

        self.commandRequested.emit(
            {
                "action": "preview",
                "sources": json.loads(sources_json),
                "extensions": json.loads(extensions_json),
                "max_file_bytes": max_file_bytes,
                "workspace": workspace,
            }
        )

    @Slot(str, str, str, str, str)
    def openProject(
        self,
        sources_json: str,
        kind: str,
        workspace: str,
        selected_json: str,
        settings_json: str,
    ) -> None:
        """Request project preparation in the backend process.

        Files are checked before they are copied into the workspace.

        Parameters
        ----------
        sources_json: str
            JSON-encoded list of source-path strings from the web page.
        kind: str
            Project category string forwarded to the backend engine.
        workspace: str
            Workspace path string, possibly empty when the engine should
            discover or allocate a default.
        selected_json: str
            JSON-encoded mapping from original source paths to selected
            relative files.
        settings_json: str
            JSON-encoded profile or generation-settings dictionary for the
            requested operation.

        Raises
        ------
        json.JSONDecodeError
            If a supplied JSON string cannot be decoded.

        Notes
        -----
        Decoding and signal emission occur in the GUI process. The engine
        performs the operation later. The openProject slot separately emits an
        error for a decoded source value that is not a list of strings.

        """

        sources = json.loads(sources_json)
        if not isinstance(sources, list) or not all(
            isinstance(path, str) for path in sources
        ):
            self.event.emit(
                json.dumps({"event": "error", "message": "Invalid sources."})
            )
            return
        self.commandRequested.emit(
            {
                "action": "open",
                "sources": sources,
                "kind": kind,
                "workspace": workspace,
                "selected_files": json.loads(selected_json),
                "settings": json.loads(settings_json),
            }
        )

    @Slot(str, str, str)
    def askQuestion(
        self, question: str, effort: str, settings_json: str
    ) -> None:
        """Request one answer with the current conversation settings.

        These settings affect generation but do not change profile.ini or the
        saved retrieval indexes.

        Parameters
        ----------
        question: str
            User question forwarded to the loaded RAG session.
        effort: str
            Light, instant, lagom, thorough, or deep GUI effort preset.
        settings_json: str
            JSON-encoded profile or generation-settings dictionary for the
            requested operation.

        Raises
        ------
        json.JSONDecodeError
            If a supplied JSON string cannot be decoded.

        Notes
        -----
        Decoding and signal emission occur in the GUI process. The engine
        performs the operation later. The openProject slot separately emits an
        error for a decoded source value that is not a list of strings.

        """

        self.commandRequested.emit(
            {
                "action": "ask",
                "question": question,
                "effort": effort,
                "generation_settings": json.loads(settings_json),
            }
        )

    @Slot(int, int)
    def previewReference(self, turn_index: int, citation: int) -> None:
        """Request source context for one saved answer citation.

        The GUI sends numbers rather than a file path, so the backend can
        resolve only evidence attached to the open project.

        Parameters
        ----------
        turn_index: int
            Zero-based saved turn position used for a reference preview.
        citation: int
            One-based evidence number within the selected turn.

        """

        self.commandRequested.emit(
            {
                "action": "preview_reference",
                "turn_index": turn_index,
                "citation": citation,
            }
        )

    @Slot()
    def resetConversation(self) -> None:
        """Archive the current conversation in the backend process.

        Notes
        -----
        The engine reset action creates a new conversation rather than erasing
        the previous one.

        """

        self.commandRequested.emit({"action": "reset"})

    @Slot()
    def newConversation(self) -> None:
        """Create and open a separate empty conversation.


        """

        self.commandRequested.emit({"action": "new_conversation"})

    @Slot()
    def branchConversation(self) -> None:
        """Branch the active conversation at its current saved checkpoint.


        """

        self.commandRequested.emit({"action": "branch_conversation"})

    @Slot(str)
    def selectConversation(self, conversation_id: str) -> None:
        """Open a saved conversation without reloading project resources.

        Parameters
        ----------
        conversation_id: str
            Registered conversation identifier forwarded to the engine.

        """

        self.commandRequested.emit(
            {
                "action": "select_conversation",
                "conversation_id": conversation_id,
            }
        )

    @Slot(str, str)
    def renameConversation(self, conversation_id: str, name: str) -> None:
        """Rename a saved conversation in the project catalog.

        Parameters
        ----------
        conversation_id: str
            Registered conversation identifier forwarded to the engine.
        name: str
            Requested conversation display name, validated by the backend
            store.

        """

        self.commandRequested.emit(
            {
                "action": "rename_conversation",
                "conversation_id": conversation_id,
                "name": name,
            }
        )

    @Slot(str)
    def deleteConversation(self, conversation_id: str) -> None:
        """Archive a conversation and remove it from the visible catalog."""

        self.commandRequested.emit(
            {
                "action": "delete_conversation",
                "conversation_id": conversation_id,
            }
        )

    @Slot(str)
    def renameProject(self, name: str) -> None:
        """Ask the backend to rename the active project directory."""

        self.commandRequested.emit({"action": "rename_project", "name": name})

    @Slot(str)
    def openWorkspaceFile(self, relative_path: str) -> None:
        """Open one validated project file with its desktop application."""

        if self.project_root is None:
            return
        try:
            root = self.project_root.resolve(strict=True)
            path = (root / relative_path).resolve(strict=True)
            if not path.is_file() or not path.is_relative_to(root):
                raise ValueError("File is outside the open project.")
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        except (OSError, ValueError):
            self.event.emit(
                json.dumps(
                    {
                        "event": "error",
                        "message": "Could not open that project file.",
                    }
                )
            )

    @Slot(str, str)
    def findSavedProject(self, kind: str, workspace: str) -> None:
        """Refresh Resume for the selected project type and workspace.

        Parameters
        ----------
        kind: str
            Project category string forwarded to the backend engine.
        workspace: str
            Workspace path string, possibly empty when the engine should
            discover or allocate a default.

        """

        self.commandRequested.emit(
            {
                "action": "saved_project",
                "kind": kind,
                "workspace": workspace,
            }
        )

    @Slot(str)
    def listSavedProjects(self, workspace: str) -> None:
        """List all valid saved projects without loading their models."""

        self.commandRequested.emit(
            {"action": "saved_projects", "workspace": workspace}
        )

    @Slot(str)
    def resumeProject(self, directory: str) -> None:
        """Open a saved project without changing its source snapshot.

        Parameters
        ----------
        directory: str
            Folder containing the saved QAChat profile.ini.

        """

        self.commandRequested.emit({"action": "resume", "project": directory})

    @Slot(str, str)
    def resumeProjectConversation(
        self,
        directory: str,
        conversation_id: str,
    ) -> None:
        """Reopen a project at the exact conversation active before a stop."""

        self.commandRequested.emit(
            {
                "action": "resume",
                "project": directory,
                "conversation_id": conversation_id,
            }
        )

    @Slot()
    def closeProject(self) -> None:
        """Release models when the user chooses a different project.


        """

        self.commandRequested.emit({"action": "close"})

    @Slot()
    def stopOperation(self) -> None:
        """Terminate the active backend process immediately.

        Notes
        -----
        Signals the window to kill the engine process. It does not gracefully
        cancel a single generation while retaining loaded model state.

        """

        self.stopRequested.emit()


class SourceDropView(QWebEngineView):
    """Accept only local files and folders dropped onto the window.

    Parameters
    ----------
    parent: QWidget | None, optional
        Optional parent widget passed to the Qt base constructor. Defaults to
        ``None``.

    Methods
    -------
    dragEnterEvent(event)
        Accept a drag only when every item is a local path.
    dragMoveEvent(event)
        Keep the native drop target active over the whole view.
    dropEvent(event)
        Send dropped paths through the same source-selection flow.

    Notes
    -----
    Accepts local URL paths from native drops and delegates unsupported
    drag/drop events to the base QWebEngineView implementation.

    """

    pathsDropped = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Enable native drops before the web page is loaded.

        Parameters
        ----------
        parent: QWidget | None, optional
            Optional parent widget passed to the Qt base constructor. Defaults
            to ``None``.

        """

        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept a drag only when every item is a local path.

        Parameters
        ----------
        event: QDragEnterEvent
            Qt event delivered by the corresponding widget event handler.

        """

        if event.mimeData().hasUrls() and all(
            url.isLocalFile() for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """Keep the native drop target active over the whole view.

        Parameters
        ----------
        event: QDragMoveEvent
            Qt event delivered by the corresponding widget event handler.

        """

        if event.mimeData().hasUrls() and all(
            url.isLocalFile() for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        """Send dropped paths through the same source-selection flow.

        Parameters
        ----------
        event: QDropEvent
            Qt event delivered by the corresponding widget event handler.

        """

        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        if paths:
            self.pathsDropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class ChatWindow(QMainWindow):
    """Display the web page while owning one stoppable backend process.

    The backend writes JSON events to stdout. The window keeps the GUI
    responsive while model loading, indexing, or answering takes time.

    Methods
    -------
    closeEvent(event)
        Stop only this window's backend child process on exit.

    Notes
    -----
    Requires an existing QApplication. The web channel exposes the bridge as
    "qachat". The child process uses the current Python executable and handles
    blocking source/model operations. Local-page remote URL access is disabled,
    but backend network access is governed separately by model operations.

    """

    def __init__(self) -> None:
        """Create the bridge, backend process, and local web view.


        """

        super().__init__()
        self.setWindowTitle("QAChat")
        self.resize(1180, 820)
        self.setMinimumSize(760, 560)
        self._closing = False
        self._stopping = False
        self._pending: list[dict[str, object]] = []
        self._stdout = bytearray()

        self.bridge = ChatBridge(self)
        self.bridge.commandRequested.connect(self._send_command)
        self.bridge.stopRequested.connect(self._stop_operation)

        # Model work belongs in a child process so Stop can interrupt it.
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.started.connect(self._send_pending)
        self.process.readyReadStandardOutput.connect(self._read_events)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self._start_engine()

        self.view = SourceDropView(self)
        self.view.pathsDropped.connect(self.bridge.pathsSelected)
        settings = self.view.settings()
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls,
            False,
        )
        self.channel = QWebChannel(self.view.page())
        self.channel.registerObject("qachat", self.bridge)
        self.view.page().setWebChannel(self.channel)
        self.setCentralWidget(self.view)
        page = Path(__file__).resolve().parent / "web" / "index.html"
        self.view.load(QUrl.fromLocalFile(str(page)))

    def _start_engine(self) -> None:
        """Start the child that owns models, indexes, and chat history.


        """

        self._stdout.clear()
        self.process.start(sys.executable, ["-m", "qachat.gui.engine"])

    def _send_command(self, command: dict[str, object]) -> None:
        """Queue a command until the child process is ready to receive it.

        Parameters
        ----------
        command: dict[str, object]
            Structured command dictionary queued for newline-delimited JSON
            transport.

        """

        self._pending.append(command)
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self._start_engine()
        else:
            self._send_pending()

    def _send_pending(self) -> None:
        """Write queued commands as one JSON object per line.


        """

        if self.process.state() != QProcess.ProcessState.Running:
            return
        while self._pending:
            command = self._pending.pop(0)
            self.process.write((json.dumps(command) + "\n").encode())

    def _read_events(self) -> None:
        """Forward complete JSON lines without exposing backend stdout.

        Notes
        -----
        Buffers incomplete stdout lines until newline, parses complete JSON
        events, and ignores malformed lines. Events arriving during a stop are
        discarded.

        """

        self._stdout.extend(bytes(self.process.readAllStandardOutput()))
        while b"\n" in self._stdout:
            line, _, rest = self._stdout.partition(b"\n")
            self._stdout = bytearray(rest)
            try:
                event = json.loads(line)
                if (
                    not self._stopping
                    and isinstance(event, dict)
                    and "event" in event
                ):
                    if event["event"] == "opened" and isinstance(
                        event.get("project"), str
                    ):
                        self.bridge.project_root = Path(event["project"])
                    self.bridge.event.emit(json.dumps(event))
            except (UnicodeError, ValueError):
                continue

    def _stop_operation(self) -> None:
        """Interrupt the child immediately and discard queued commands.

        Notes
        -----
        Clears pending commands and terminates the backend with QProcess.kill.
        On completion a fresh engine is started; previously loaded models do
        not remain available.

        """

        if self._stopping:
            return
        self._stopping = True
        self._pending.clear()
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self._process_finished()
        else:
            self.process.kill()

    def _process_finished(self, *_args) -> None:
        """Restart after Stop, or report an unexpected backend exit.

        Parameters
        ----------
        *_args: object
            Qt finished-signal positional arguments, ignored by this handler.

        Notes
        -----
        Suppresses notifications during application close. A requested stop
        emits stopped and restarts the engine; an unexpected exit emits
        backend_failed and requires project reopening.

        """

        if self._closing:
            return
        if self._stopping:
            self._stopping = False
            self._start_engine()
            self.bridge.event.emit(json.dumps({"event": "stopped"}))
        else:
            self.bridge.event.emit(
                json.dumps(
                    {
                        "event": "backend_failed",
                        "message": "Backend stopped. Reopen the project.",
                    }
                )
            )

    def _process_error(self, _error) -> None:
        """Forward a child-process failure to the visible interface.

        Parameters
        ----------
        _error: object
            Qt process-error signal value; the handler uses process.errorString
            instead.

        """

        if not self._closing and not self._stopping:
            self.bridge.event.emit(
                json.dumps(
                    {"event": "error", "message": self.process.errorString()}
                )
            )

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop only this window's backend child process on exit.

        Parameters
        ----------
        event: QCloseEvent
            Qt event delivered by the corresponding widget event handler.

        Notes
        -----
        Marks shutdown, kills a running child, waits up to the existing 3000 ms
        bound, and delegates the event to QMainWindow. This is process
        termination, not a graceful checkpoint request.

        """

        self._closing = True
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.process.waitForFinished(3000)
        super().closeEvent(event)
