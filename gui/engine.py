"""Keep one QAChat session in a stoppable desktop child process."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Callable

from qachat import RAGSession, prepare_sources_workspace

ROOT = Path(__file__).resolve().parents[1]
EventSink = Callable[[dict[str, object]], None]


class ChatEngine:
    """Translate a few GUI commands into public QAChat calls."""

    def __init__(self, sandbox: Path, emit: EventSink) -> None:
        self.sandbox = sandbox
        self.emit = emit
        self.session: RAGSession | None = None

    def _send(self, event: str, **data: object) -> None:
        self.emit({"event": event, **data})

    def open_project(self, sources: list[str], kind: str, workspace: str) -> None:
        """Add selected sources and open or resume one QAChat workspace."""

        try:
            if kind not in {"code", "literature"}:
                raise ValueError("Choose code or literature.")
            if not sources and not workspace.strip():
                example = "code_repo" if kind == "code" else "literature_repo"
                sources = [str(ROOT / "example_repos" / example)]
            if not workspace.strip():
                source = Path(sources[0]).expanduser().resolve()
                digest = sha256(str(source).encode()).hexdigest()[:8]
                workspace = str(self.sandbox / f"{source.stem}-{digest}")
            self.close()
            self._send("status", message="Preparing the project…")
            project = prepare_sources_workspace(sources, kind, workspace)
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
                kind=kind,
                index_status=session.index_status,
                turns=[asdict(turn) for turn in session.turns],
            )
        except Exception as error:
            self._send("error", message=str(error))

    def ask(self, question: str) -> None:
        """Ask the loaded project and return its saved answer and evidence."""

        try:
            if self.session is None:
                raise RuntimeError("Open a project before asking a question.")
            self._send("status", message="Finding evidence and answering…")
            self.session.ask(question)
            self._send("answered", turn=asdict(self.session.turns[-1]))
        except Exception as error:
            self._send("error", message=str(error))

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
                            command["sources"], command["kind"], command["workspace"]
                        )
                    elif action == "ask":
                        engine.ask(command["question"])
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
