"""Exercise GUI project rename and source-import recovery.

Use real project and conversation files with a fake RAG session.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name



from dataclasses import dataclass, field
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


from qachat.conversation_format import conversation_state
from qachat.conversations import ConversationStore
from qachat.gui.engine import ChatEngine
from qachat.project import open_project

from qachat.workspace import prepare_sources_workspace


@dataclass
class Turn:
    question: str
    answer: str
    parameters: dict = field(default_factory=dict)
    evidence: tuple = ()


class FakeSession:
    """Use real workspace/history files without loading model weights."""

    def __init__(self, directory, *, conversation_path, **_options):
        """Initialize a fake RAG session from saved project state."""
        self.project = open_project(directory)
        self.conversation_path = conversation_path
        self.turns = [
            Turn(**turn)
            for turn in conversation_state(
                json.loads(conversation_path.read_text())
            ).get("turns", [])
        ]
        self.index_status = "rebuilt"
        self.chat_session = SimpleNamespace(
            temperature=0.1,
            thinking=False,
            max_new_tokens=self.project.profile.max_new_tokens,
            configure_generation=self.configure,
        )
        self.closed = False
        self.backend = "mlx"
        self.router_model = None

    def use_router_model(self, model_id):
        """Record the router selected for the fake session."""
        self.router_model = model_id

    def refresh_sources(self, project):
        """Index the copied sources again: one stand-in chunk per file."""

        self.project = project
        self.refreshed = getattr(self, "refreshed", 0) + 1
        self.source_changes = {}
        self.chunks = [
            SimpleNamespace(path=path.relative_to(project.source))
            for path in project.source.rglob("*")
            if path.is_file()
        ]

    def configure(self, settings):
        """Apply generation settings to the fake chat session."""
        for key, value in settings.items():
            setattr(self.chat_session, key, value)

    def close(self):
        """Record that the fake session has closed."""
        self.closed = True


class TestEngineProjectRecovery(unittest.TestCase):
    """Keep active conversations valid across project changes."""
    def setUp(self):
        """Prepare disposable project files and mocked model sessions."""
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        source = self.root / "original.txt"
        source.write_text("Original source text.")
        self.project = prepare_sources_workspace(
            [source], "literature", self.root / "project"
        )
        store = ConversationStore(self.project.root)
        self.active = store.ensure()
        self.history_path = store.state_path(self.active)
        payload = json.loads(self.history_path.read_text())
        conversation_state(payload)["turns"] = [
            {
                "question": "Original question",
                "answer": "Original answer.",
            }
        ]
        self.history_path.write_text(json.dumps(payload))
        # Ensure the selected chat is not the most recently created one.
        store.create()
        store.select(self.active.conversation_id)
        self.events = []
        self.engine = ChatEngine(self.root, self.events.append)
        self.addCleanup(self.engine.close)
        patcher = patch("qachat.gui.engine.RAGSession", FakeSession)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.engine.conversation_store = store
        self.engine.active_conversation = self.active
        self.engine._activate_project(self.project, self.active)
        self.events.clear()

    def assert_chat_preserved(self, expected_root):
        """Check that current project identity and history survived."""
        self.assertEqual(self.engine.session.project.root, expected_root)
        self.assertEqual(
            self.engine.active_conversation.conversation_id,
            self.active.conversation_id,
        )
        self.assertEqual(
            self.engine.session.turns[0].answer, "Original answer."
        )
        self.assertEqual(
            len(self.engine.conversation_store.conversations()), 2
        )

    def test_rename_discovers_and_restores_selected_chat(self):
        """Keep the selected conversation after a workspace rename."""
        self.engine.trace_enabled = True
        old_session = self.engine.session
        self.engine.rename_project("renamed")
        self.assertTrue(old_session.closed)
        self.assertFalse(self.project.root.exists())
        self.assert_chat_preserved(self.root / "renamed")
        self.assertEqual(self.events[-1]["event"], "opened")
        self.assertTrue(self.engine.session.conversation_path.is_file())

    def test_rename_collision_keeps_current_session(self):
        """Reject conflicting names without closing the current session."""
        (self.root / "occupied").mkdir()
        old_session = self.engine.session
        self.engine.rename_project("occupied")
        self.assertIs(self.engine.session, old_session)
        self.assertFalse(old_session.closed)
        self.assert_chat_preserved(self.project.root)
        self.assertEqual(self.events[-1]["event"], "error")

    def test_failed_reopen_rolls_back_rename(self):
        """Roll back a rename when reactivation fails."""
        activate = self.engine._activate_project
        calls = 0

        def fail_once(*args):
            """Simulate one failed project reopening."""
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("Failed loading model")
            return activate(*args)

        with patch.object(self.engine, "_activate_project", fail_once):
            self.engine.rename_project("renamed")
        self.assertFalse((self.root / "renamed").exists())
        self.assert_chat_preserved(self.project.root)
        self.assertEqual(self.events[-1]["event"], "error")

    def test_add_file_and_folder_preserves_sources_and_chat(self):
        """Add new sources without reloading the active chat."""
        file = self.root / "new.txt"
        file.write_text("New source text.")
        folder = self.root / "papers"
        folder.mkdir()
        (folder / "paper.md").write_text("# New paper")
        old_session = self.engine.session
        self.engine.add_project_sources([str(file), str(folder)])
        # The models stay loaded; only the indexes are updated.
        self.assertIs(self.engine.session, old_session)
        self.assertFalse(old_session.closed)
        indexed = {chunk.path.name for chunk in old_session.chunks}
        self.assertTrue({"new.txt", "paper.md"} <= indexed, indexed)
        self.assert_chat_preserved(self.project.root)
        sources = self.project.root / "sources"
        self.assertEqual((sources / "new.txt").read_text(), "New source text.")
        self.assertTrue((sources / "papers" / "paper.md").is_file())
        self.assertTrue((sources / "original.txt").is_file())
        self.assertEqual(file.read_text(), "New source text.")
        self.assertEqual(self.events[-1]["event"], "opened")

    def test_failed_import_restores_usable_chat(self):
        """Keep the current conversation accessible after import failure."""
        self.engine.add_project_sources([str(self.root / "missing.txt")])
        self.assert_chat_preserved(self.project.root)
        self.assertEqual(self.events[-1]["event"], "error")
