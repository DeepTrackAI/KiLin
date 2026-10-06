"""Regression coverage for source additions, renaming, and complete answers."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from qachat.chat import ChatSession
from qachat.conversation_format import conversation_state
from qachat.conversations import ConversationStore
from qachat.gui.engine import ChatEngine
from qachat.project import open_project
from qachat.trace_logging import NullTraceLogger
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

    def configure(self, settings):
        for key, value in settings.items():
            setattr(self.chat_session, key, value)

    def close(self):
        self.closed = True


class TestProjectPolish(unittest.TestCase):
    def setUp(self):
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
        self.engine.trace_enabled = True
        old_session = self.engine.session
        self.engine.rename_project("renamed")
        self.assertTrue(old_session.closed)
        self.assertFalse(self.project.root.exists())
        self.assert_chat_preserved(self.root / "renamed")
        self.assertEqual(self.events[-1]["event"], "opened")
        self.assertTrue(self.engine.session.conversation_path.is_file())

    def test_rename_collision_keeps_current_session(self):
        (self.root / "occupied").mkdir()
        old_session = self.engine.session
        self.engine.rename_project("occupied")
        self.assertIs(self.engine.session, old_session)
        self.assertFalse(old_session.closed)
        self.assert_chat_preserved(self.project.root)
        self.assertEqual(self.events[-1]["event"], "error")

    def test_failed_reopen_rolls_back_rename(self):
        activate = self.engine._activate_project
        calls = 0

        def fail_once(*args):
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
        file = self.root / "new.txt"
        file.write_text("New source text.")
        folder = self.root / "papers"
        folder.mkdir()
        (folder / "paper.md").write_text("# New paper")
        old_session = self.engine.session
        self.engine.add_project_sources([str(file), str(folder)])
        self.assertTrue(old_session.closed)
        self.assert_chat_preserved(self.project.root)
        sources = self.project.root / "sources"
        self.assertEqual((sources / "new.txt").read_text(), "New source text.")
        self.assertTrue((sources / "papers" / "paper.md").is_file())
        self.assertTrue((sources / "original.txt").is_file())
        self.assertEqual(file.read_text(), "New source text.")
        self.assertEqual(self.events[-1]["event"], "opened")

    def test_failed_import_restores_usable_chat(self):
        self.engine.add_project_sources([str(self.root / "missing.txt")])
        self.assert_chat_preserved(self.project.root)
        self.assertEqual(self.events[-1]["event"], "error")

    def test_upgrade_legacy_budget_preserves_other_settings(self):
        self.engine.session.chat_session.max_new_tokens = 256
        self.engine._upgrade_answer_budget()
        self.assertEqual(self.engine.session.chat_session.max_new_tokens, 2048)
        self.engine.session.chat_session.max_new_tokens = 512
        self.engine._upgrade_answer_budget()
        self.assertEqual(self.engine.session.chat_session.max_new_tokens, 512)


class TestIndexPersistence(unittest.TestCase):
    def test_import_rebuilds_saves_and_reuses_index(self):
        embeddings = SimpleNamespace(
            count_tokens=lambda text: len(text.split()),
            embed_documents=lambda texts: [[1.0, 0.0] for text in texts],
            _clear_backend_cache=lambda: None,
            close=lambda: None,
        )
        with (
            TemporaryDirectory() as temporary,
            patch("qachat.rag.EmbeddingSession", return_value=embeddings),
            patch.object(ChatSession, "_load"),
        ):
            root = Path(temporary).resolve()
            first, second = root / "one.txt", root / "two.txt"
            first.write_text("The original source explains memory capacity.")
            second.write_text("The added source explains feature elimination.")
            project = prepare_sources_workspace(
                [first], "literature", root / "workspace"
            )
            events = []
            engine = ChatEngine(root, events.append)
            try:
                engine.resume_project(str(project.root))
                self.assertIsNotNone(engine.session, events)
                self.assertEqual(engine.session.index_status, "rebuilt")
                first_fingerprint = engine.session.indexes.fingerprint
                engine.add_project_sources([str(second)])
                self.assertEqual(engine.session.index_status, "rebuilt")
                self.assertNotEqual(
                    engine.session.indexes.fingerprint, first_fingerprint
                )
                paths = {str(chunk.path) for chunk in engine.session.chunks}
                self.assertEqual(paths, {"one.txt", "two.txt"})
                for name in (
                    "retrieval.json",
                    "settings.json",
                    "source_manifest.json",
                ):
                    self.assertTrue((project.indexes / name).is_file())
                engine.close()
                engine.resume_project(str(project.root))
                self.assertEqual(engine.session.index_status, "reused")
                self.assertEqual(
                    {str(chunk.path) for chunk in engine.session.chunks}, paths
                )
            finally:
                engine.close()


class TestGenerationFinish(unittest.TestCase):
    def test_mlx_preserves_entire_answer_and_finish_reason(self):
        for reason in ("length", "stop"):
            with self.subTest(reason=reason):
                session = ChatSession.__new__(ChatSession)
                session._model = object()
                session._tokenizer = object()
                session.temperature = 0.1
                model_api = SimpleNamespace(
                    stream_generate=MagicMock(
                        return_value=iter(
                            [
                                SimpleNamespace(
                                    text="First ",
                                    generation_tokens=1,
                                    finish_reason=None,
                                ),
                                SimpleNamespace(
                                    text="last.",
                                    generation_tokens=2,
                                    finish_reason=reason,
                                ),
                            ]
                        )
                    )
                )
                sampler = SimpleNamespace(make_sampler=MagicMock())
                with patch(
                    "qachat.chat.import_module",
                    side_effect=[
                        model_api,
                        sampler,
                    ],
                ):
                    pieces = []
                    answer = session._mlx_reply(
                        [1, 2], 2048, token_callback=pieces.append
                    )
                self.assertEqual(answer, "First last.")
                self.assertEqual(pieces, ["First ", "last."])
                self.assertEqual(session.last_finish_reason, reason)

    def test_torch_eos_at_limit_is_complete(self):
        for token, expected in [(2, "stop"), (99, "length")]:
            with self.subTest(token=token):
                session = ChatSession.__new__(ChatSession)
                session._model = MagicMock()
                session._model.generation_config.eos_token_id = [2, 3]
                output = session._model.generate.return_value
                output.__getitem__.return_value.tolist.return_value = [token]
                session._tokenizer = MagicMock(eos_token_id=2)
                session._tokenizer.decode.return_value = "Answer"
                session.temperature = 0.1
                session.device = "cpu"
                session.trace = NullTraceLogger()
                with patch(
                    "qachat.chat.import_module", return_value=MagicMock()
                ):
                    self.assertEqual(session._torch_reply([1], 1), "Answer")
                self.assertEqual(session.last_finish_reason, expected)


if __name__ == "__main__":
    unittest.main()
