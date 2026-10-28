"""Characterize GUI engine workflows without loading model weights.

The fake session exposes the project and conversation fields used by the GUI.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY, MagicMock, patch
import unittest

from kilin.gui.engine import ChatEngine
from kilin.gui.model_catalog import MODEL_CHOICES
from kilin.workspace import prepare_sources_workspace


@dataclass
class _SavedTurn:
    """Represent the minimum fields of a persisted turn."""

    question: str
    answer: str
    evidence: tuple = ()
    parameters: dict[str, object] = field(default_factory=dict)
    timestamp: str = "2026-10-07T10:00:00+00:00"


class TestEngineWorkflow(unittest.TestCase):
    """Verify project and answer operations at the GUI boundary."""

    def setUp(self) -> None:
        """Prepare disposable project files and mocked model sessions."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def model_session(self):
        """Provide real profile values without loading model weights."""

        source = self.root / "source.py"
        source.write_text("def example():\n    return 1\n", encoding="utf-8")
        project = prepare_sources_workspace(
            [source], "code", self.root / "project"
        )
        session = MagicMock()
        session.project = project
        session.backend = "mlx"
        session.index_status = "reused"
        session.turns = []
        session.chat_session.backend = "mlx"
        session.chat_session.temperature = 0.0
        session.chat_session.thinking = False
        session.chat_session.max_new_tokens = 2048
        session.switch_conversation.return_value = []
        return session

    def assert_question_forwarded(self, session, question, **options):
        """Verify exact answer options and live callback handoff."""
        session.ask.assert_called_once_with(
            question,
            token_callback=ANY,
            phase_callback=ANY,
            pdf_writer=ANY,
            **options,
        )
        options = session.ask.call_args.kwargs
        self.assertTrue(callable(options["token_callback"]))
        self.assertTrue(callable(options["phase_callback"]))

    @patch("kilin.gui.engine.download_model")
    def test_download_accepts_explicit_catalog_and_custom_models(
        self, download
        ) -> None:
        """Accept valid cached or custom model IDs, but reject invalid IDs."""
        events: list[dict[str, object]] = []
        engine = ChatEngine(Path("/sandbox"), events.append)

        engine.download(MODEL_CHOICES[0].model_id)

        download.assert_called_once_with(
            MODEL_CHOICES[0].model_id, progress_callback=ANY
        )
        self.assertTrue(
            callable(download.call_args.kwargs["progress_callback"])
        )
        self.assertEqual(events[0]["event"], "model_download_started")
        self.assertEqual(events[-1]["event"], "model_downloaded")
        self.assertIn(
            "model_download_progress", [event["event"] for event in events]
        )

        engine.download("someone/custom-model")
        self.assertEqual(download.call_count, 2)
        self.assertEqual(download.call_args.args, ("someone/custom-model",))

        engine.download("invalid model id")
        self.assertEqual(download.call_count, 2)
        self.assertEqual(events[-1]["event"], "error")

    @patch("kilin.gui.engine.RAGSession")
    @patch("kilin.gui.engine.prepare_sources_workspace")
    def test_open_ask_and_reset(self, prepare, rag_session) -> None:
        """Delegate project creation, answering, and reset to the core."""
        session = self.model_session()
        prepare.return_value = session.project
        session.turns = [_SavedTurn("Old question", "Old answer")]
        rag_session.return_value = session
        events: list[dict[str, object]] = []
        engine = ChatEngine(self.root, events.append)

        engine.open_project(["/original"], "code", str(session.project.root))
        old_conversation = engine.active_conversation
        session.turns.append(_SavedTurn("New question", "New answer"))
        engine.ask("New question")
        engine.reset()
        engine.close()

        prepare.assert_called_once_with(
            ["/original"], "code", str(session.project.root)
        )
        self.assert_question_forwarded(session, "New question")
        session.reset.assert_not_called()
        session.switch_conversation.assert_called_once()
        self.assertNotEqual(
            engine.active_conversation, old_conversation
        )
        session.close.assert_called_once_with()
        self.assertIn("opened", [event["event"] for event in events])
        answered = next(
            event for event in events if event["event"] == "answered"
        )
        self.assertEqual(answered["turn"]["answer"], "New answer")
        self.assertEqual(events[-1]["event"], "reset")

    @patch("kilin.gui.engine.RAGSession")
    @patch("kilin.gui.engine.prepare_sources_workspace")
    def test_blank_source_requires_workspace_without_using_examples(
        self, prepare, rag_session
        ) -> None:
        """Reject a source-free request without its explicit destination."""

        events: list[dict[str, object]] = []
        engine = ChatEngine(self.root, events.append)

        engine.open_project([], "literature", "", selected_files={})

        prepare.assert_not_called()
        rag_session.assert_not_called()
        self.assertEqual(events[-1]["event"], "error")
        self.assertIn("Choose a workspace folder", events[-1]["message"])
        self.assertIsNone(engine.session)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_evidence_count_is_passed_for_one_question(self) -> None:
        """Forward the requested retrieval count to a single answer."""
        events: list[dict[str, object]] = []
        engine = ChatEngine(Path("/sandbox"), events.append)
        session = self.model_session()
        session.turns = [_SavedTurn("Question", "Answer")]
        engine.session = session

        engine.ask("Question", retrieval_count=8)

        self.assert_question_forwarded(session, "Question", retrieval_count=8)
        self.assertEqual(events[-1]["event"], "answered")

    def test_generation_settings_are_passed_with_the_question(self) -> None:
        """Chat controls affect answers without editing the project profile."""

        events: list[dict[str, object]] = []
        engine = ChatEngine(Path("/sandbox"), events.append)
        session = self.model_session()
        session.turns = [_SavedTurn("Question", "Answer")]
        engine.session = session
        settings = {
            "temperature": 0.4,
            "thinking": True,
            "max_new_tokens": 128,
        }

        engine.ask("Question", 6, settings)

        self.assert_question_forwarded(
            session,
            "Question", retrieval_count=6, generation_settings=settings
        )

    @patch("kilin.gui.engine.RAGSession")
    @patch("kilin.gui.engine.open_project")
    def test_resume_does_not_copy_sources(
        self, open_saved, rag_session
        ) -> None:
        """Startup restores the existing project without source preparation."""

        session = self.model_session()
        project = session.project
        open_saved.return_value = project
        rag_session.return_value = session
        events: list[dict[str, object]] = []
        engine = ChatEngine(self.root, events.append)

        engine.resume_project(str(project.root))

        open_saved.assert_called_once_with(str(project.root))
        rag_session.assert_called_once()
        self.assertEqual(events[-1]["event"], "opened")

    @patch("kilin.gui.engine.RAGSession")
    def test_chat_events_refresh_the_workspace_tree(self, rag):
        """Creating or deleting a chat sends the updated project files."""

        session = self.model_session()
        rag.return_value = session
        events = []
        engine = ChatEngine(self.root, events.append)
        self.addCleanup(engine.close)
        engine.resume_project(str(session.project.root))
        old = engine.active_conversation
        root = session.project.root

        def shown(path: Path) -> bool:
            """Whether the latest chat event lists this file."""
            latest = [
                event for event in events
                if event["event"] in {"conversation_opened", "conversations"}
            ][-1]
            return path.relative_to(root).as_posix() in latest[
                "workspace_files"
            ]

        engine.create_conversation()
        new = engine.conversation_store.state_path(engine.active_conversation)
        self.assertTrue(shown(new))
        path = engine.conversation_store.state_path(old)
        self.assertTrue(shown(path))

        engine.delete_conversation(old.conversation_id)
        self.assertFalse(any(event["event"] == "error" for event in events))
        self.assertFalse(path.exists())
        self.assertFalse(shown(path))

    @patch("kilin.gui.engine.RAGSession")
    def test_deleting_active_chat_preserves_trace_and_switches(self, rag):
        """Deletion archives durable content without a closing rewrite."""

        session = self.model_session()
        rag.return_value = session
        events = []
        engine = ChatEngine(self.root, events.append, trace_enabled=True)
        self.addCleanup(engine.close)
        engine.resume_project(str(session.project.root))
        self.assertFalse(any(event["event"] == "error" for event in events))
        # A second saved chat avoids coupling this test to the clock's
        # filename resolution when the final chat is archived.
        old = engine.active_conversation
        engine.create_conversation()
        engine.select_conversation(old.conversation_id)
        session.switch_conversation.reset_mock()
        path = engine.conversation_store.state_path(old)
        engine.active_trace.emit(
            "turn.started", component="rag", turn_id="retained",
            data={"question": "Keep this historical trace"},
        )
        before = path.read_bytes()
        events.clear()

        engine.delete_conversation(old.conversation_id)

        self.assertEqual(events[0]["event"], "conversation_delete_started")
        self.assertFalse(any(event["event"] == "error" for event in events))
        # The last chat's replacement may reuse a timestamped filename;
        # its identity and content must still be entirely new.
        if path.exists():
            self.assertNotIn(
                "Keep this historical trace", path.read_text(encoding="utf-8")
            )
        archived = list((path.parent / "deleted").glob("*.json"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_bytes(), before)
        self.assertNotEqual(
            engine.active_conversation.conversation_id, old.conversation_id
        )
        session.switch_conversation.assert_called_once()
        session.close.assert_not_called()
        self.assertTrue(
            any(event["event"] == "conversation_opened" for event in events)
        )
