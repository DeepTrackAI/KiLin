"""Host KíLín's local web interface in a Qt desktop window.

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

>>> import kilin.gui.app
>>> exit_code = kilin.gui.app.main(["kilin"])

"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from PySide6.QtCore import (
    QObject,
    QProcess,
    QRectF,
    Qt,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QCloseEvent,
    QCursor,
    QDesktopServices,
    QDragEnterEvent,
    QDragMoveEvent,
    QDragLeaveEvent,
    QDropEvent,
    QGuiApplication,
    QIcon,
)
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMenu, QWidget

from kilin.conversation_format import conversation_state
from kilin.gui._paths import (
    default_workspace_directory,
    saved_workspace_folder,
    standard_workspace_directory,
)
from kilin.gui.preferences import default_appearance, write_defaults

__all__ = [
    "ChatBridge",
    "SourceDropView",
    "ChatWindow",
    "saved_conversation_turns",
]

ROOT = Path(__file__).resolve().parents[2]
# How long Stop waits for the engine to end an answer before restarting it,
# as a last resort for an engine that no longer responds.
CANCEL_GRACE_MS = 30_000
ICONS = Path(__file__).resolve().parent / "icons"


class ChatBridge(QObject):
    """Expose Qt signals and slots connecting the page to backend commands.

    Attributes
    ----------
    event: Signal(str)
        Serialized engine event sent back to the web page.
    pathsSelected: Signal(list)
        Local source paths chosen using native dialogs.
    folderFilesSelected: Signal(str, str, list)
        Editable folder (outputs or templates), purpose ("add" or the path
        to replace), and the chosen files.
    projectsFolderChosen: Signal(str)
        Folder chosen to hold the projects, or "" for the standard one.
    commandRequested: Signal(dict)
        Backend command handed to the window process controller.
    stopRequested: Signal()
        Request to terminate the active engine process.
    answerCancelRequested: Signal()
        Stop was pressed for a running answer.

    Methods
    -------
    chooseSources()
        Offer files and folders through one browse control.
    chooseFolder()
        Show the native folder picker on the GUI thread.
    chooseFiles()
        Add one or more individual files.
    chooseProjectsFolder(), useDefaultProjectsFolder()
        Choose the workspace folder that holds the projects, or go back to
        the standard one.
    requestProjectsFolder()
        Report the workspace folder in use.
    requestSettings(type_id, workspace)
        Read the existing profile or new-project defaults.
    downloadModel(model_id)
        Download one suggested model only after the user chooses it.
    previewSources(sources_json, extensions_json, max_file_bytes, workspace)
        Scan selected roots without copying them.
    openProject(sources_json, type_id, workspace, selected_json, settings_json)
        Request project preparation in the backend process.
    askQuestion(question, effort, settings_json)
        Request one answer with the selected GUI effort preset.
    previewReference(turn_index, citation)
        Request source context for one saved answer citation.
    resetConversation()
        Archive the current conversation in the backend process.
    newConversation()
        Create and open a separate empty conversation.
    branchConversation(turn_count)
        Branch the active conversation, keeping its first turns.
    selectConversation(conversation_id)
        Open a saved conversation without reloading project resources.
    viewConversation(conversation_id)
        Show a saved conversation read-only while an answer is generated.
    renameConversation(conversation_id, name)
        Rename a saved conversation in the project catalog.
    deleteConversation(conversation_id)
        Archive a conversation and remove it from the project catalog.
    listSavedProjects(workspace)
        List valid saved projects of all registered types.
    resumeProject(directory)
        Open a saved project without changing its source snapshot.
    resumeProjectConversation(directory, conversation_id)
        Reopen a saved project at one exact conversation.
    closeProject()
        Release models when the user chooses a different project.
    cancelAnswer()
        Stop the running answer while the engine and models stay loaded.
    requestToolSettings(), saveToolSettings(name, settings_json)
        List the open project's tools, or save one tool's settings.
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
    folderFilesSelected = Signal(str, str, list)
    commandRequested = Signal(dict)
    stopRequested = Signal()
    answerCancelRequested = Signal()
    sourceDropTargetChanged = Signal(str)
    sourceDragChanged = Signal(bool)
    appearanceChanged = Signal(str)
    projectsFolderChosen = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Keep only the currently opened project path on the UI side.

        Parameters
        ----------
        parent: QObject | None, optional
            Optional parent Qt widget. Defaults to ``None``.

        """

        super().__init__(parent)
        self.project_root: Path | None = None

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

    @Slot(str, str)
    def chooseFolderFiles(self, folder: str, purpose: str) -> None:
        """Choose files for ``outputs/`` or ``templates/``.

        ``purpose`` is ``"add"`` (any number of files) or a path to replace
        (one file); folder and purpose are echoed back with the paths.

        Parameters
        ----------
        folder: str
            Name of the workspace folder whose files are requested.
        purpose: str
            Reason for choosing the files, supplied to the page.

        """

        if purpose == "add":
            paths, _ = QFileDialog.getOpenFileNames(None, f"Add to {folder}")
        else:
            path, _ = QFileDialog.getOpenFileName(None, "Replace with")
            paths = [path] if path else []
        if paths:
            self.folderFilesSelected.emit(folder, purpose, paths)

    @Slot(str, list)
    def folderDropped(self, folder: str, paths: list) -> None:
        """Offer files dropped on ``outputs`` or ``templates`` to the page.

        Parameters
        ----------
        folder: str
            Target workspace folder for the incoming files.
        paths: list
            Local file paths that were dropped on the folder.

        """

        self.folderFilesSelected.emit(folder, "add", paths)

    @Slot(str)
    def editFolder(self, request_json: str) -> None:
        """Add, replace, or delete files in outputs or templates.

        Parameters
        ----------
        request_json: str
            JSON command specifying the folder and requested edit.

        """

        request = json.loads(request_json)
        self.commandRequested.emit({**request, "action": "edit_folder"})

    @Slot()
    def chooseProjectsFolder(self) -> None:
        """Choose the folder where KíLín keeps its projects."""

        path = QFileDialog.getExistingDirectory(
            None, "Choose where KíLín keeps projects"
        )
        if path:
            self.projectsFolderChosen.emit(path)

    @Slot()
    def useDefaultProjectsFolder(self) -> None:
        """Go back to the standard projects folder."""

        self.projectsFolderChosen.emit("")

    @Slot()
    def requestProjectsFolder(self) -> None:
        """Report the projects folder in use and the one saved in Settings.

        Notes
        -----
        Sends ``projects_folder`` with ``folder`` (in use), ``standard``
        (used without a choice), ``saved`` (the choice, or ""), and
        ``missing``: the saved folder is gone or not mounted, so the
        standard one is used.

        """

        try:
            folder = str(default_workspace_directory())
            standard = str(standard_workspace_directory())
        except ValueError as error:
            self.event.emit(
                json.dumps({"event": "error", "message": str(error)})
            )
            return
        saved = saved_workspace_folder()
        self.event.emit(
            json.dumps(
                {
                    "event": "projects_folder",
                    "folder": folder,
                    "standard": standard,
                    "saved": saved,
                    "missing": bool(saved)
                    and not Path(saved).expanduser().is_dir(),
                }
            )
        )

    @Slot(str, str)
    def requestSettings(self, type_id: str, workspace: str) -> None:
        """Read the existing profile or new-project defaults.

        ``workspace`` is blank when the user is preparing a new project.

        Parameters
        ----------
        type_id: str
            Project category string forwarded to the backend engine.
        workspace: str
            Workspace path string, possibly empty when the engine should
            discover or allocate a default.

        """

        self.commandRequested.emit(
            {"action": "settings", "type_id": type_id, "workspace": workspace}
        )

    @Slot()
    def requestReadiness(self) -> None:
        """Ask the backend to check runtime and required model availability."""

        self.commandRequested.emit({"action": "readiness"})

    @Slot()
    def installRequiredModels(self) -> None:
        """Ask the backend to download all missing default models."""

        self.commandRequested.emit({"action": "install_required_models"})

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
        type_id: str,
        workspace: str,
        selected_json: str,
        settings_json: str,
    ) -> None:
        """Request exclusive new-project creation in the backend process.

        An occupied explicit destination produces workspace_conflict, not an
        implicit reopen. resumeProject is the separate existing-project action.

        Parameters
        ----------
        sources_json: str
            JSON-encoded list of source-path strings from the web page.
        type_id: str
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
                "type_id": type_id,
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

    @Slot()
    def useInstalledModels(self) -> None:
        """Use installed models in place of missing preferred ones."""

        self.commandRequested.emit({"action": "use_installed_models"})

    @Slot(str)
    def choosePlan(self, choice: str) -> None:
        """Answer a tool plan that waits for the user's choice.

        Parameters
        ----------
        choice: str
            ``continue``, ``quick``, or ``cancel`` for a plan, or
            ``option:<index>`` for a tool's question; the engine treats any
            other value as cancel.

        Notes
        -----
        The engine reads this command while the answer that proposed the
        plan is still running.

        """

        self.commandRequested.emit({"action": "plan_choice", "choice": choice})

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
        """Create and open a separate empty conversation."""

        self.commandRequested.emit({"action": "new_conversation"})

    @Slot(int)
    def branchConversation(self, turn_count: int) -> None:
        """Branch the active conversation, keeping its first turns.

        Parameters
        ----------
        turn_count: int
            Number of saved turns the branch keeps.

        """

        self.commandRequested.emit(
            {"action": "branch_conversation", "turn_count": turn_count}
        )

    @Slot(str)
    def viewConversation(self, conversation_id: str) -> None:
        """Show a saved conversation read-only while an answer is generated.

        Parameters
        ----------
        conversation_id: str
            Saved conversation to read.

        Notes
        -----
        The engine is busy answering, so the conversation file is read here,
        without the engine. Emits ``conversation_viewed`` with its saved
        turns, or with ``turns`` None when it cannot be read.

        """

        turns = (
            None if self.project_root is None
            else saved_conversation_turns(self.project_root, conversation_id)
        )
        # Not an ``error`` event, which would end the running answer.
        self.event.emit(
            json.dumps(
                {
                    "event": "conversation_viewed",
                    "conversation_id": conversation_id,
                    "turns": turns,
                }
            )
        )

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

    @Slot(str, str)
    def saveAnswerTimeline(self, timestamp: str, timeline_json: str) -> None:
        """Keep how long an answer took, so a resumed chat shows it again.

        Parameters
        ----------
        timestamp: str
            The answered turn's timestamp, which identifies it.
        timeline_json: str
            JSON object with ``elapsed_ms`` and ``steps`` (each a ``label``
            and ``ms``), as the page displayed them.

        """

        self.commandRequested.emit(
            {
                "action": "save_answer_timeline",
                "timestamp": timestamp,
                "timeline": json.loads(timeline_json),
            }
        )

    @Slot(str)
    def deleteConversation(self, conversation_id: str) -> None:
        """Archive a conversation and remove it from the visible catalog.

        Parameters
        ----------
        conversation_id: str
            Identifier of the saved conversation.

        """

        self.commandRequested.emit(
            {
                "action": "delete_conversation",
                "conversation_id": conversation_id,
            }
        )

    @Slot(str)
    def applyModelSettings(self, settings_json: str) -> None:
        """Send the JSON-decoded model settings to the backend.

        Parameters
        ----------
        settings_json: str
            JSON object containing the selected model settings.

        """

        self.commandRequested.emit(
            {"action": "apply_models", "settings": json.loads(settings_json)}
        )

    @Slot(str)
    def setAppearance(self, mode: str) -> None:
        """Report the page's color mode: system, light, or dark.

        Parameters
        ----------
        mode: str
            Appearance or operating mode selected by the page.

        """

        self.appearanceChanged.emit(mode)

    @Slot(str)
    def setSourceDropTarget(self, bounds_json: str) -> None:
        """Align the native drop region with the visible sources tree.

        Parameters
        ----------
        bounds_json: str
            JSON bounds supplied by the web page.

        """

        self.sourceDropTargetChanged.emit(bounds_json)

    @Slot(str)
    def checkProjectSources(self, sources_json: str) -> None:
        """Ask what adding these sources would copy, before adding.

        Parameters
        ----------
        sources_json: str
            JSON-encoded source paths dropped or chosen in the page.

        """

        self.commandRequested.emit(
            {
                "action": "check_sources",
                "sources": json.loads(sources_json),
            }
        )

    @Slot(str)
    def saveAnswerSettings(self, settings_json: str) -> None:
        """Save answer settings to the open project right away.

        Parameters
        ----------
        settings_json: str
            JSON object with any of temperature, thinking, and
            max_new_tokens.

        """

        self.commandRequested.emit(
            {
                "action": "save_answer_settings",
                "settings": json.loads(settings_json),
            }
        )

    @Slot()
    def requestToolSettings(self) -> None:
        """Ask for the open project's tools and their settings."""

        self.commandRequested.emit({"action": "tool_settings"})

    @Slot(str, str)
    def saveToolSettings(self, name: str, settings_json: str) -> None:
        """Save one tool's settings for the open project.

        Parameters
        ----------
        name: str
            Tool the project type uses.
        settings_json: str
            JSON object of the tool's changeable settings.

        """

        self.commandRequested.emit(
            {
                "action": "save_tool_settings",
                "tool": name,
                "settings": json.loads(settings_json),
            }
        )

    @Slot(str, str)
    def addProjectSources(
        self, sources_json: str, decisions_json: str
    ) -> None:
        """Import sources after the user confirmed the check.

        Parameters
        ----------
        sources_json: str
            JSON-encoded source paths selected in the page.
        decisions_json: str
            JSON object of sync actions for copies already in the project.

        """

        self.commandRequested.emit(
            {
                "action": "add_project_sources",
                "sources": json.loads(sources_json),
                "decisions": json.loads(decisions_json),
            }
        )

    @Slot(str)
    def removeProjectSources(self, paths_json: str) -> None:
        """Delete confirmed workspace source copies and reindex.

        Parameters
        ----------
        paths_json: str
            JSON-encoded paths selected in the page.

        """

        self.commandRequested.emit(
            {
                "action": "remove_project_sources",
                "paths": json.loads(paths_json),
            }
        )

    @Slot(str)
    def syncProjectSources(self, decisions_json: str) -> None:
        """Apply the user's choice for each changed source and reindex.

        Parameters
        ----------
        decisions_json: str
            JSON object mapping ``sources/`` paths to sync actions.

        """

        self.commandRequested.emit(
            {
                "action": "sync_sources",
                "decisions": json.loads(decisions_json),
            }
        )

    @Slot(str)
    def renameProject(self, name: str) -> None:
        """Ask the backend to rename the active project directory.

        Parameters
        ----------
        name: str
            Filename or project name to use.

        """

        self.commandRequested.emit({"action": "rename_project", "name": name})

    @Slot(str)
    def openWorkspaceFile(self, relative_path: str) -> None:
        """Open one validated project file with its desktop application.

        Parameters
        ----------
        relative_path: str
            Path relative to the active workspace.

        """

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

    @Slot(str)
    def copyText(self, text: str) -> None:
        """Put text on the system clipboard for the page's Copy buttons.

        Qt WebEngine does not let the page write to the clipboard itself.

        Parameters
        ----------
        text: str
            Text to copy.

        """

        QGuiApplication.clipboard().setText(text)

    @Slot(str, str)
    def findSavedProject(self, type_id: str, workspace: str) -> None:
        """Refresh Resume for the selected project type and workspace.

        Parameters
        ----------
        type_id: str
            Project category string forwarded to the backend engine.
        workspace: str
            Workspace path string, possibly empty when the engine should
            discover or allocate a default.

        """

        self.commandRequested.emit(
            {
                "action": "saved_project",
                "type_id": type_id,
                "workspace": workspace,
            }
        )

    @Slot(str)
    def listSavedProjects(self, workspace: str) -> None:
        """List all valid saved projects without loading their models.

        Parameters
        ----------
        workspace: str
            Workspace containing saved projects.

        """

        self.commandRequested.emit(
            {"action": "saved_projects", "workspace": workspace}
        )

    @Slot(str)
    def resumeProject(self, directory: str) -> None:
        """Open a saved project without changing its source snapshot.

        Parameters
        ----------
        directory: str
            Folder containing the saved KíLín profile.ini.

        """

        self.commandRequested.emit({"action": "resume", "project": directory})

    @Slot(str, str)
    def resumeProjectConversation(
        self,
        directory: str,
        conversation_id: str,
    ) -> None:
        """Reopen a project at the exact conversation active before a stop.

        Parameters
        ----------
        directory: str
            Directory in which to find or create the file.
        conversation_id: str
            Identifier of the saved conversation.

        """

        self.commandRequested.emit(
            {
                "action": "resume",
                "project": directory,
                "conversation_id": conversation_id,
            }
        )

    @Slot()
    def closeProject(self) -> None:
        """Release models when the user chooses a different project."""

        self.commandRequested.emit({"action": "close"})

    @Slot(str)
    def openExternalLink(self, url: str) -> None:
        """Open an approved HTTPS link only after the user clicks it.

        Parameters
        ----------
        url: str
            HTTPS publication or bibliographic search URL from a report.

        Notes
        -----
        No link is fetched by the embedded web view or by the backend.
        Explicit clicks open the system browser, outside KíLín.

        """

        target = QUrl(url)
        if (
            target.isValid() and target.scheme() == "https"
            and target.host() and not target.userName()
            and not target.password()
        ):
            QDesktopServices.openUrl(target)

    @Slot()
    def cancelAnswer(self) -> None:
        """Stop the running answer while the engine and models stay loaded.

        Notes
        -----
        Sends ``cancel_answer``, which the engine applies at the model's next
        token. The window restarts the engine only if no answer event follows
        within ``CANCEL_GRACE_MS``.

        """

        self.commandRequested.emit({"action": "cancel_answer"})
        self.answerCancelRequested.emit()

    @Slot()
    def stopOperation(self) -> None:
        """Terminate the active backend process immediately.

        Notes
        -----
        Signals the window to kill the engine process. It does not gracefully
        cancel a single generation while retaining loaded model state.

        """

        self.stopRequested.emit()


def saved_conversation_turns(
    root: Path, conversation_id: str
) -> list[dict[str, object]] | None:
    """Read one saved conversation's turns, as the page shows them.

    Parameters
    ----------
    root: pathlib.Path
        Open project folder.
    conversation_id: str
        Saved conversation to read.

    Returns
    -------
    list[dict[str, object]] | None
        Its saved turns, each with its saved timing under ``timeline`` when
        there is one, or None when no readable conversation has that ID.

    Notes
    -----
    Reads files only, so it never waits for the busy engine. Symbolic links
    and unreadable files are skipped, as conversation discovery does.

    """

    try:
        timelines = json.loads(
            (root / "logs" / "answer_timelines.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError):
        timelines = {}
    if not isinstance(timelines, dict):
        timelines = {}
    for path in sorted((root / "logs" / "traces").glob("conversation_*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            state = conversation_state(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError):
            continue
        turns = state.get("turns", [])
        if state.get("conversation_id") != conversation_id:
            continue
        if not isinstance(turns, list):
            return None
        shown = []
        for turn in turns:
            if not isinstance(turn, dict):
                return None
            timeline = timelines.get(turn.get("timestamp"))
            shown.append(
                {**turn, "timeline": timeline}
                if isinstance(timeline, dict) else turn
            )
        return shown
    return None


class SourceDropView(QWebEngineView):
    """Accept local files only inside the page's visible drop targets.

    The sources folder (or the new-project drop zone) adds sources; the
    workspace's ``outputs`` and ``templates`` folders receive copies.

    Parameters
    ----------
    parent: QWidget | None, optional
        Owning Qt widget, if any. Defaults to ``None``.

    """

    pathsDropped = Signal(list)
    folderDropped = Signal(str, list)
    dragChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Enable native file drops and initialize the drop-target regions.

        Parameters
        ----------
        parent: QWidget | None, optional
            Optional parent Qt widget. Defaults to ``None``.

        """

        super().__init__(parent)
        self.setAcceptDrops(True)
        self.drop_target = QRectF()
        self.folder_targets: list[tuple[str, QRectF]] = []

    @Slot(str)
    def set_drop_target(self, bounds_json: str) -> None:
        """Update the native drop regions from page-provided JSON bounds.

        Parameters
        ----------
        bounds_json: str
            JSON bounds supplied by the web page.

        """

        bounds = json.loads(bounds_json)

        def rect(item: dict) -> QRectF:
            """Convert an object with x, y, width, and height into a rectangle.

            Parameters
            ----------
            item: dict
                Dictionary containing the rectangle coordinates.

            Returns
            -------
            QRectF
                Rectangle specified by the supplied page bounds.

            """

            return QRectF(
                *[
                    float(item.get(key, 0))
                    for key in ("x", "y", "width", "height")
                ]
            )

        self.drop_target = rect(bounds)
        self.folder_targets = [
            (str(item.get("name")), rect(item))
            for item in bounds.get("folders", [])
            if isinstance(item, dict)
        ]

    def _target(self, event) -> str | None:
        """``"sources"``, a folder name, or None at the event position.

        Parameters
        ----------
        event: object
            Qt drag, drop, or window event.

        Returns
        -------
        str | None
            Target folder name, ``"sources"``, or None.

        """

        position = event.position()
        if self.drop_target.contains(position):
            return "sources"
        for name, rect in self.folder_targets:
            if rect.contains(position):
                return name
        return None

    def _local_drag(self, event) -> bool:
        """Check whether the drag contains only local file URLs.

        Parameters
        ----------
        event: object
            Qt drag, drop, or window event.

        Returns
        -------
        bool
            True if every dragged URL refers to a local file.

        """

        urls = event.mimeData().urls()
        return bool(urls) and all(url.isLocalFile() for url in urls)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        # Drags inside the page (a workspace file onto sources) are the
        # page's own.
        """Accept a local file drag when the page has an active drop zone.

        Parameters
        ----------
        event: QDragEnterEvent
            Qt drag, drop, or window event.

        """

        if not event.mimeData().hasUrls():
            super().dragEnterEvent(event)
            return
        # Accept entry to continue receiving movement; the final drop is
        # restricted to the target, even when the drag entered elsewhere.
        if self._local_drag(event) and (
            not self.drop_target.isEmpty() or self.folder_targets
        ):
            self.dragChanged.emit(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """Accept a moving local file drag only above a drop target.

        Parameters
        ----------
        event: QDragMoveEvent
            Qt drag, drop, or window event.

        """

        if not event.mimeData().hasUrls():
            super().dragMoveEvent(event)
            return
        if self._local_drag(event) and self._target(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        """Clear the drag indicator when a dragged item leaves the view.

        Parameters
        ----------
        event: QDragLeaveEvent
            Qt drag, drop, or window event.

        """

        self.dragChanged.emit(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        """Dispatch local files to the matching drop-target signal.

        Parameters
        ----------
        event: QDropEvent
            Qt drag, drop, or window event.

        """

        self.dragChanged.emit(False)
        if not event.mimeData().hasUrls():
            super().dropEvent(event)
            return
        target = self._target(event) if self._local_drag(event) else None
        if target is None:
            event.ignore()
            return
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        if target == "sources":
            self.pathsDropped.emit(paths)
        else:
            self.folderDropped.emit(target, paths)
        event.acceptProposedAction()


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
    "kilin". The child process uses the current Python executable and handles
    blocking source/model operations. Local-page remote URL access is disabled,
    but backend network access is governed separately by model operations.

    """

    def __init__(self) -> None:
        """Create the bridge, backend process, and local web view."""

        super().__init__()
        self.setWindowTitle("KíLín")
        self.resize(1180, 820)
        self.setMinimumSize(760, 560)
        self._closing = False
        self._stopping = False
        # Restarting the engine so that it uses a new projects folder.
        self._switching_folder = False
        self._pending: list[dict[str, object]] = []
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._recovery_event: dict[str, object] | None = None

        self.bridge = ChatBridge(self)
        self.bridge.commandRequested.connect(self._send_command)
        self.bridge.stopRequested.connect(self._stop_operation)
        # A Stop the engine does not answer in time restarts it instead.
        self._cancel_timer = QTimer(self)
        self._cancel_timer.setSingleShot(True)
        self._cancel_timer.setInterval(CANCEL_GRACE_MS)
        self._cancel_timer.timeout.connect(self._stop_operation)
        self.bridge.answerCancelRequested.connect(self._cancel_timer.start)

        # The saved appearance until the page reports its choice; "system"
        # follows the OS.
        self._appearance = default_appearance()
        self.bridge.appearanceChanged.connect(self._set_appearance)
        self.bridge.projectsFolderChosen.connect(self._set_projects_folder)
        QGuiApplication.styleHints().colorSchemeChanged.connect(
            lambda _scheme: self._update_icon()
        )
        self._update_icon()

        # Model work belongs in a child process so Stop can interrupt it.
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.started.connect(self._send_pending)
        self.process.started.connect(self._process_started)
        self.process.readyReadStandardOutput.connect(self._read_events)
        self.process.readyReadStandardError.connect(self._read_diagnostics)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self._start_engine()

        self.view = SourceDropView(self)
        self.view.pathsDropped.connect(self.bridge.pathsSelected)
        self.view.folderDropped.connect(self.bridge.folderDropped)
        self.bridge.sourceDropTargetChanged.connect(self.view.set_drop_target)
        self.view.dragChanged.connect(self.bridge.sourceDragChanged)
        settings = self.view.settings()
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls,
            False,
        )
        self.channel = QWebChannel(self.view.page())
        self.channel.registerObject("kilin", self.bridge)
        self.view.page().setWebChannel(self.channel)
        self.setCentralWidget(self.view)
        page = Path(__file__).resolve().parent / "web" / "index.html"
        self.view.load(QUrl.fromLocalFile(str(page)))

    def _set_appearance(self, mode: str) -> None:
        """Remember the page's color mode and refresh the app icon.

        The choice is saved in the app-wide preferences.

        Parameters
        ----------
        mode: str
            Appearance or operating mode selected by the page.

        """

        self._appearance = (
            mode if mode in ("system", "light", "dark") else "system"
        )
        try:
            write_defaults({"appearance": self._appearance})
        except OSError:
            pass  # An unwritable preferences file keeps today's choice.
        self._update_icon()

    def _set_projects_folder(self, folder: str) -> None:
        """Save the projects folder and restart the engine to use it.

        The engine reads the folder when it starts. The page offers this
        only on the start screen while idle, so no project or answer is
        interrupted.

        Parameters
        ----------
        folder: str
            Chosen absolute folder, or "" for the standard folder.

        """

        try:
            write_defaults({"workspace_folder": folder})
        except (OSError, ValueError) as error:
            self.bridge.event.emit(
                json.dumps(
                    {
                        "event": "error",
                        "message": (
                            f"Could not save the projects folder: {error}"
                        ),
                    }
                )
            )
            return
        self._switching_folder = True
        self._pending.clear()
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self._process_finished()
        else:
            self.process.kill()

    def _update_icon(self) -> None:
        """Show the light or dark logo in the window and Dock.

        Missing icon files leave Qt's default icon in place.

        """

        dark = self._appearance == "dark" or (
            self._appearance == "system"
            and QGuiApplication.styleHints().colorScheme()
            == Qt.ColorScheme.Dark
        )
        path = ICONS / ("kilin-dark.png" if dark else "kilin-light.png")
        if not path.is_file():
            return
        icon = QIcon(str(path))
        self.setWindowIcon(icon)
        QGuiApplication.setWindowIcon(icon)

    def _start_engine(self) -> None:
        """Start the child that owns models, indexes, and chat history."""

        self._stdout.clear()
        self._stderr.clear()
        self.process.start(sys.executable, ["-m", "kilin.gui.engine"])

    def _process_started(self) -> None:
        """Report a successful automatic restart after an unexpected exit."""

        if self._recovery_event is not None:
            self.bridge.event.emit(json.dumps(self._recovery_event))
            self._recovery_event = None

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
        """Write queued commands as one JSON object per line."""

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
                    if event["event"] in {
                        "answered", "answer_cancelled", "error"
                    }:
                        # The answer ended, so a pending Stop needs no
                        # restart.
                        self._cancel_timer.stop()
                    if event["event"] == "opened" and isinstance(
                        event.get("project"), str
                    ):
                        self.bridge.project_root = Path(event["project"])
                    self.bridge.event.emit(json.dumps(event))
            except (UnicodeError, ValueError):
                continue

    def _read_diagnostics(self) -> None:
        """Drain and retain a bounded tail of backend diagnostic output."""

        self._stderr.extend(bytes(self.process.readAllStandardError()))
        if len(self._stderr) > 65_536:
            self._stderr = self._stderr[-65_536:]

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
        self._cancel_timer.stop()
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
        if self._switching_folder:
            self._switching_folder = False
            self._start_engine()
            self.bridge.requestProjectsFolder()
            return
        if self._stopping:
            self._stopping = False
            self._start_engine()
            self.bridge.event.emit(json.dumps({"event": "stopped"}))
        else:
            exit_code = int(_args[0]) if _args else self.process.exitCode()
            exit_status = _args[1] if len(_args) > 1 else None
            crashed = exit_status == QProcess.ExitStatus.CrashExit
            diagnostic_lines = (
                self._stderr.decode("utf-8", errors="replace")
                .strip()
                .splitlines()
            )
            diagnostic = (
                diagnostic_lines[-1][-1000:] if diagnostic_lines else ""
            )
            reason = (
                "The background worker crashed"
                if crashed
                else f"The background worker exited with code {exit_code}"
            )
            self._recovery_event = {
                "event": "backend_restarted",
                "message": (
                    f"{reason}, so KíLín restarted it. Downloaded model "
                    "files were kept."
                ),
                "exit_code": exit_code,
                "diagnostic": diagnostic,
            }
            self._start_engine()

    def _process_error(self, _error) -> None:
        """Forward a child-process failure to the visible interface.

        Parameters
        ----------
        _error: object
            Qt process-error signal value; the handler uses process.errorString
            instead.

        """

        if not (self._closing or self._stopping or self._switching_folder):
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
