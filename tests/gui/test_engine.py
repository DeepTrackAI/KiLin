"""Check the GUI engine boundary without loading model weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from gui.engine import ChatEngine


@dataclass
class SavedTurn:
    """Small conversation record for the engine test."""

    question: str
    answer: str
    evidence: tuple[()] = ()


class TestChatEngine(unittest.TestCase):
    """The child process delegates each operation to QAChat."""

    @patch("gui.engine.RAGSession")
    @patch("gui.engine.prepare_project_workspace")
    def test_open_ask_and_reset(self, prepare, rag_session) -> None:
        prepare.return_value.root = Path("/sandbox/example")
        prepare.return_value.source = Path("/sandbox/example/repository")
        session = MagicMock()
        session.index_status = "reused"
        session.turns = [SavedTurn("Old question", "Old answer")]
        rag_session.return_value = session
        events: list[dict[str, object]] = []
        engine = ChatEngine(Path("/sandbox"), events.append)

        engine.open_project("/original", "code")
        session.turns.append(SavedTurn("New question", "New answer"))
        engine.ask("New question")
        engine.reset()
        engine.close()

        prepare.assert_called_once_with("/original", "code", Path("/sandbox"))
        session.ask.assert_called_once_with("New question")
        session.reset.assert_called_once_with()
        session.close.assert_called_once_with()
        self.assertEqual(events[1]["event"], "opened")
        self.assertEqual(events[2]["event"], "status")
        self.assertEqual(events[3]["turn"]["answer"], "New answer")
        self.assertEqual(events[-1]["event"], "reset")

    @patch("gui.engine.RAGSession")
    @patch("gui.engine.prepare_project_workspace")
    def test_blank_source_uses_bundled_example(
        self, prepare, rag_session
    ) -> None:
        prepare.return_value.root = Path("/sandbox/example")
        prepare.return_value.source = Path("/sandbox/example/documents")
        rag_session.return_value.turns = []
        rag_session.return_value.index_status = "rebuilt"
        engine = ChatEngine(Path("/project/sandbox"), lambda event: None)

        engine.open_project("", "literature")

        prepare.assert_called_once_with(
            str(
                Path(__file__).resolve().parents[2]
                / "example_repos"
                / "literature_repo"
            ),
            "literature",
            Path("/project/sandbox"),
        )


if __name__ == "__main__":
    unittest.main()
