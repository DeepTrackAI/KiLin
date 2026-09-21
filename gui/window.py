"""Qt shell hosting QAChat's local JavaScript interface."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from PySide6.QtCore import QObject, QProcess, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QFileDialog, QMainWindow

ROOT = Path(__file__).resolve().parents[1]


class ChatBridge(QObject):
    """Expose only user actions and backend events to the web page."""

    event = Signal(str)
    folderSelected = Signal(str)
    commandRequested = Signal(dict)
    stopRequested = Signal()

    @Slot()
    def chooseFolder(self) -> None:
        """Show the native folder picker on the GUI thread."""

        path = QFileDialog.getExistingDirectory(None, "Choose project folder")
        if path:
            self.folderSelected.emit(path)

    @Slot(str, str)
    def openProject(self, source: str, kind: str) -> None:
        """Request project preparation in the backend process."""

        self.commandRequested.emit(
            {"action": "open", "source": source, "kind": kind}
        )

    @Slot(str)
    def askQuestion(self, question: str) -> None:
        """Request one grounded answer in the backend process."""

        self.commandRequested.emit({"action": "ask", "question": question})

    @Slot()
    def resetConversation(self) -> None:
        """Archive the current conversation in the backend process."""

        self.commandRequested.emit({"action": "reset"})

    @Slot()
    def stopOperation(self) -> None:
        """Terminate the active backend process immediately."""

        self.stopRequested.emit()


class ChatWindow(QMainWindow):
    """Display JS while owning exactly one stoppable backend process."""

    def __init__(self) -> None:
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
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.setStandardErrorFile(str(sandbox / "gui-engine.log"))
        self.process.started.connect(self._send_pending)
        self.process.readyReadStandardOutput.connect(self._read_events)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self._start_engine()

        self.view = QWebEngineView(self)
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
        self._stdout.clear()
        self.process.start(sys.executable, ["-m", "gui.engine"])

    def _send_command(self, command: dict[str, object]) -> None:
        self._pending.append(command)
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self._start_engine()
        else:
            self._send_pending()

    def _send_pending(self) -> None:
        if self.process.state() != QProcess.ProcessState.Running:
            return
        while self._pending:
            command = self._pending.pop(0)
            self.process.write((json.dumps(command) + "\n").encode())

    def _read_events(self) -> None:
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
        if self._stopping:
            return
        self._stopping = True
        self._pending.clear()
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self._process_finished()
        else:
            self.process.kill()

    def _process_finished(self, *_args) -> None:
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
        if not self._closing and not self._stopping:
            self.bridge.event.emit(
                json.dumps(
                    {"event": "error", "message": self.process.errorString()}
                )
            )

    def closeEvent(self, event) -> None:
        """Stop only this window's backend child process on exit."""

        self._closing = True
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.process.waitForFinished(3000)
        super().closeEvent(event)
