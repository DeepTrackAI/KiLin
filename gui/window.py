"""Host the local web interface without mixing GUI and model execution.

Classes
-------
ChatBridge
    Pass user actions to the backend and events back to JavaScript.
SourceDropView
    Accept local files and folders dropped onto the window.
ChatWindow
    Own the web view and one stoppable backend child process.

"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from PySide6.QtCore import QObject, QProcess, QUrl, Signal, Slot
from PySide6.QtGui import (
    QCloseEvent,
    QCursor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
)
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMenu, QWidget

ROOT = Path(__file__).resolve().parents[1]


class ChatBridge(QObject):
    """Expose user actions and backend events to the local web page.

    The page never calls QAChat models directly. It sends small commands
    through this bridge to the child process that owns the active session.
    """

    event = Signal(str)
    pathsSelected = Signal(list)
    workspaceSelected = Signal(str)
    commandRequested = Signal(dict)
    stopRequested = Signal()

    @Slot()
    def chooseSources(self) -> None:
        """Offer files and folders through one browse control."""

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
        """Show the native folder picker on the GUI thread."""

        path = QFileDialog.getExistingDirectory(None, "Choose project folder")
        if path:
            self.pathsSelected.emit([path])

    @Slot()
    def chooseFiles(self) -> None:
        """Add one or more individual files."""

        paths, _ = QFileDialog.getOpenFileNames(None, "Choose source files")
        if paths:
            self.pathsSelected.emit(paths)

    @Slot()
    def chooseWorkspace(self) -> None:
        """Choose the folder in which QAChat stores its project."""

        path = QFileDialog.getExistingDirectory(
            None, "Choose QAChat workspace"
        )
        if path:
            self.workspaceSelected.emit(path)

    @Slot(str, str)
    def requestSettings(self, kind: str, workspace: str) -> None:
        """Read the existing profile or new-project defaults.

        ``workspace`` is blank when the user is preparing a new project.
        """

        self.commandRequested.emit(
            {"action": "settings", "kind": kind, "workspace": workspace}
        )

    @Slot(str)
    def downloadModel(self, model_id: str) -> None:
        """Download one suggested model only after the user chooses it."""

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

    @Slot(str, int, str)
    def askQuestion(
        self, question: str, retrieval_count: int, settings_json: str
    ) -> None:
        """Request one answer with the current conversation settings.

        These settings affect generation but do not change profile.ini or
        the saved retrieval indexes.
        """

        self.commandRequested.emit(
            {
                "action": "ask",
                "question": question,
                "retrieval_count": retrieval_count,
                "generation_settings": json.loads(settings_json),
            }
        )

    @Slot(int, int)
    def previewReference(self, turn_index: int, citation: int) -> None:
        """Request source context for one saved answer citation.

        The GUI sends numbers rather than a file path, so the backend can
        resolve only evidence attached to the open project.
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
        """Archive the current conversation in the backend process."""

        self.commandRequested.emit({"action": "reset"})

    @Slot(str, str)
    def findSavedProject(self, kind: str, workspace: str) -> None:
        """Refresh Resume for the selected project type and workspace."""

        self.commandRequested.emit(
            {
                "action": "saved_project",
                "kind": kind,
                "workspace": workspace,
            }
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

    @Slot()
    def closeProject(self) -> None:
        """Release models when the user chooses a different project."""

        self.commandRequested.emit({"action": "close"})

    @Slot()
    def stopOperation(self) -> None:
        """Terminate the active backend process immediately."""

        self.stopRequested.emit()


class SourceDropView(QWebEngineView):
    """Accept only local files and folders dropped onto the window."""

    pathsDropped = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Enable native drops before the web page is loaded."""

        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept a drag only when every item is a local path."""

        if event.mimeData().hasUrls() and all(
            url.isLocalFile() for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """Keep the native drop target active over the whole view."""

        if event.mimeData().hasUrls() and all(
            url.isLocalFile() for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        """Send dropped paths through the same source-selection flow."""

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
    """

    def __init__(self) -> None:
        """Create the bridge, backend process, and local web view."""

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

        sandbox = ROOT / "sandbox"
        sandbox.mkdir(exist_ok=True)
        # Model work belongs in a child process so Stop can interrupt it.
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.setStandardErrorFile(str(sandbox / "gui-engine.log"))
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
        page = ROOT / "gui" / "web" / "index.html"
        self.view.load(QUrl.fromLocalFile(str(page)))

    def _start_engine(self) -> None:
        """Start the child that owns models, indexes, and chat history."""

        self._stdout.clear()
        self.process.start(sys.executable, ["-m", "gui.engine"])

    def _send_command(self, command: dict[str, object]) -> None:
        """Queue a command until the child process is ready to receive it."""

        self._pending.append(command)
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self._start_engine()
        else:
            self._send_pending()

    def _send_pending(self) -> None:
        """Write queued commands as one JSON object per line."""

        if self.process.state() != QProcess.ProcessState.Running:
            return
        while self._pending:
            command = self._pending.pop(0)
            self.process.write((json.dumps(command) + "\n").encode())

    def _read_events(self) -> None:
        """Forward complete JSON lines without exposing backend stdout."""

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
                    self.bridge.event.emit(json.dumps(event))
            except (UnicodeError, ValueError):
                continue

    def _stop_operation(self) -> None:
        """Interrupt the child immediately and discard queued commands."""

        if self._stopping:
            return
        self._stopping = True
        self._pending.clear()
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self._process_finished()
        else:
            self.process.kill()

    def _process_finished(self, *_args) -> None:
        """Restart after Stop, or report an unexpected backend exit."""

        if self._closing:
            return
        if self._stopping:
            self._stopping = False
            self.bridge.event.emit(json.dumps({"event": "stopped"}))
            self._start_engine()
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
        """Forward a child-process failure to the visible interface."""

        if not self._closing and not self._stopping:
            self.bridge.event.emit(
                json.dumps(
                    {"event": "error", "message": self.process.errorString()}
                )
            )

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop only this window's backend child process on exit."""

        self._closing = True
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.process.waitForFinished(3000)
        super().closeEvent(event)
