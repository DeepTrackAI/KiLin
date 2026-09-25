"""Keep one QAChat session in a stoppable desktop child process.

Classes
-------
ChatEngine
    Discover saved projects and translate GUI requests into QAChat calls.

"""

from __future__ import annotations

from configparser import ConfigParser, Error as ConfigError
from contextlib import redirect_stdout
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import traceback
from typing import Callable

from qachat.gui.model_catalog import (
    MODEL_CHOICES,
    installed_backends,
    model_choices,
)
from qachat import RAGSession, prepare_sources_workspace
from qachat.huggingface import available_models, download_model
from qachat.project import QAProject, default_project_profile, open_project
from qachat.repository import (
    BLOCKED_SUFFIXES,
    DEFAULT_EXTENSIONS,
    SKIP_FILES,
    scan_repository,
)
from qachat.runtime import choose_backend
from qachat.sources import read_repository_files

ROOT = Path(__file__).resolve().parents[2]
EventSink = Callable[[dict[str, object]], None]


class ChatEngine:
    """Translate GUI actions into public QAChat project operations."""

    def __init__(self, sandbox: Path, emit: EventSink) -> None:
        """Keep one session and send structured updates to the window.

        Parameters
        ----------
        sandbox: pathlib.Path
            Default parent folder for local project workspaces.
        emit: EventSink
            Callback that forwards an event to the web page.

        """

        self.sandbox = sandbox
        self.emit = emit
        self.session: RAGSession | None = None

    def _send(self, event: str, **data: object) -> None:
        self.emit({"event": event, **data})

    def startup(self) -> None:
        """Suggest the most recently used valid sandbox project."""

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
        """Find a current project, ignoring deleted or incomplete folders."""

        if kind is not None and kind not in {"code", "literature"}:
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
                if saved_kind not in {"code", "literature"}:
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
                conversation = folder / "logs" / "conversation.json"
                last_chat = (
                    conversation.stat().st_mtime
                    if conversation.is_file()
                    else 0
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
            self._activate_project(project)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            self._send(
                "error",
                message=f"Could not resume the project: {error}",
                log_path=str(self.sandbox / "gui-engine.log"),
            )

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
                backend=choose_backend("auto"),
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
            self._send("error", message=str(error))

    def download(self, model_id: str) -> None:
        """Download only a model explicitly chosen from the GUI catalog.

        Download progress is shown as an ongoing operation. The existing Stop
        control can interrupt its child process if the user changes their mind.

        Parameters
        ----------
        model_id: str
            One of the model IDs offered in the GUI's model panel.

        """

        try:
            if model_id not in {choice.model_id for choice in MODEL_CHOICES}:
                raise ValueError("Choose a suggested model to download.")
            self._send("model_download_started", model_id=model_id)
            download_model(model_id)
            self._send("model_downloaded", model_id=model_id)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            self._send(
                "error",
                message=f"Could not download {model_id}: {error}",
                log_path=str(self.sandbox / "gui-engine.log"),
            )

    def preview(
        self,
        sources: list[str],
        extensions: list[str],
        max_file_bytes: int,
        workspace: str = "",
    ) -> None:
        """List eligible files before copying any source.

        The preview gives the user a chance to exclude individual files.
        No workspace files are changed by this operation.

        Parameters
        ----------
        sources: list[str]
            Files or folders the user selected.
        extensions: list[str]
            Additional text file types requested by the user.
        max_file_bytes: int
            Largest file that may be copied.
        workspace: str, optional
            Chosen workspace path, used to avoid scanning itself.

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
            traceback.print_exc(file=sys.stderr)
            self._send(
                "error",
                message=f"Could not inspect the selected files: {error}",
                log_path=str(self.sandbox / "gui-engine.log"),
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

        The selected source snapshot is prepared before models are loaded.
        An empty source list can reopen a managed workspace unchanged.

        Parameters
        ----------
        sources: list[str]
            Source files and folders to copy into the workspace.
        kind: str
            Code or literature project.
        workspace: str
            Workspace folder, or blank to use the sandbox.
        selected_files: dict[str, list[str]] | None, optional
            Checked file paths within each source folder.
        settings: dict[str, object] | None, optional
            Preparation settings saved in profile.ini.

        """

        try:
            if kind not in {"code", "literature"}:
                raise ValueError("Choose code or literature.")
            if not sources and not workspace.strip():
                example = "code_repo" if kind == "code" else "literature_repo"
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
            project = prepare_sources_workspace(
                sources, kind, workspace, **options
            )
            self._activate_project(project)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            self._send(
                "error",
                message=f"Could not prepare the project: {error}",
                log_path=str(self.sandbox / "gui-engine.log"),
            )

    def _activate_project(self, project: QAProject) -> None:
        """Load models and indexes, then report the active workspace."""

        session = RAGSession(
            project.root,
            progress_callback=lambda progress: self._send(
                "progress", **asdict(progress)
            ),
        )
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
            Number of evidence results to use for this answer.
        generation_settings: dict[str, object] | None, optional
            Temperature, thinking, and answer length for this answer.

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
            self.session.ask(question, **options)
            self._send("answered", turn=asdict(self.session.turns[-1]))
        except Exception as error:
            self._send("error", message=str(error))

    def preview_reference(self, turn_index: int, citation: int) -> None:
        """Show nearby source lines for one saved answer citation.

        Only a citation already attached to the active conversation can be
        opened. The reader rechecks that the file stays inside the copied
        project source folder.
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

    def reset(self) -> None:
        """Archive conversation while keeping the index and models."""

        try:
            if self.session is None:
                raise RuntimeError("Open a project first.")
            archived = self.session.reset()
            self._send("reset", archived=str(archived) if archived else None)
        except Exception as error:
            self._send("error", message=str(error))

    def close(self) -> None:
        """Release the loaded backend session."""

        if self.session is not None:
            self.session.close()
            self.session = None


def _reference_context(
    text: str,
    excerpt: str,
    start_line: int,
    end_line: int,
    pdf: bool,
) -> dict[str, object]:
    """Find the retrieved text and return a small, numbered line window.

    PDF line numbers restart on each extracted page. Matching the saved
    excerpt also tells us which page to show without changing QAChat's
    stored evidence format.
    """

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
    """Keep a little context around the cited range."""

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
    """Preserve a usable preview when the copied source cannot be read."""

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
    """List saved workspace files for the sidebar."""

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
    """Resolve the same workspace location for preview and preparation."""

    if chosen.strip():
        return Path(chosen).expanduser().resolve()
    source = Path(sources[0]).expanduser().resolve()
    digest = sha256(str(source).encode()).hexdigest()[:8]
    return (sandbox / f"{source.stem}-{digest}").resolve()


def main() -> None:
    """Read one JSON command per line and emit one JSON event per line."""

    protocol_output = sys.stdout

    def emit(data: dict[str, object]) -> None:
        protocol_output.write(json.dumps(data, default=str) + "\n")
        protocol_output.flush()

    engine = ChatEngine(ROOT / "sandbox", emit)
    try:
        for line in sys.stdin:
            try:
                command = json.loads(line)
                if not isinstance(command, dict):
                    raise ValueError("A GUI command must be an object.")
                with redirect_stdout(sys.stderr):
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
                    elif action == "reset":
                        engine.reset()
                    else:
                        emit({"event": "error", "message": "Unknown command."})
            except (KeyError, TypeError, ValueError) as error:
                emit({"event": "error", "message": str(error)})
    finally:
        engine.close()


if __name__ == "__main__":
    main()
