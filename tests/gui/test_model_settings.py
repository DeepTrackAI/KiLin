"""Model edits preserve chats and request only authorized downloads."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qachat.gui.engine import ChatEngine
from qachat.gui.model_settings import (
    default_effort_levels,
    read_effort_levels,
    write_effort_levels,
)
from qachat.project import open_project
from qachat.workspace import prepare_sources_workspace


class TestModelSettings(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        source = self.root / "one.txt"
        source.write_text("Original document about memory.")
        self.project = prepare_sources_workspace(
            [source], "literature", self.root / "project"
        )
        self.embedded = []
        self.counted = []

        def embed(texts):
            self.embedded.extend(texts)
            return [[1.0, float(len(text))] for text in texts]

        def count(text):
            self.counted.append(text)
            return len(text.split())

        embeddings = SimpleNamespace(
            count_tokens=count,
            embed_documents=embed,
            _clear_backend_cache=lambda: None,
            close=lambda: None,
        )
        for target, options in (
            ("qachat.rag.EmbeddingSession", {"return_value": embeddings}),
            ("qachat.chat.ChatSession._load", {}),
        ):
            patcher = patch(target, **options)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.events = []
        self.engine = ChatEngine(
            self.root, self.events.append, trace_enabled=True
        )
        self.addCleanup(self.engine.close)
        self.engine.resume_project(str(self.project.root))
        self.assertIsNotNone(self.engine.session, self.events)
        self.embedded.clear()
        self.counted.clear()
        self.backend = self.engine.session.backend
        self.request = {
            "chat_model": self.project.profile.chat_model(self.backend),
            "embedding_model": self.project.profile.embedding_model(
                self.backend
            ),
            "query_format": self.project.profile.query_format,
            "download": False,
        }
        self.cached = patch(
            "qachat.gui.engine.available_models",
            return_value=[
                {"model_id": value}
                for value in {
                    self.request["chat_model"],
                    self.request["embedding_model"],
                    "example/new-chat",
                    "example/new-embedding",
                    *(
                        level["model"]
                        for level in default_effort_levels(self.backend)
                    ),
                }
            ],
        )
        self.cached.start()
        self.addCleanup(self.cached.stop)

    def trace_text(self):
        return self.engine.active_trace.readable_path.read_text()

    def test_chat_change_does_not_reindex_and_is_traced(self):
        self.request["chat_model"] = "example/new-chat"
        self.engine.apply_model_settings(self.request)
        self.assertEqual(self.events[-1]["event"], "models_applied")
        self.assertFalse(self.events[-1]["reindexed"])
        self.assertEqual(
            self.engine.session.chat_session.model_id, "example/new-chat"
        )
        self.assertEqual(self.embedded, [])
        self.assertEqual(self.counted, [])
        self.assertEqual(
            open_project(self.project.root).profile.chat_model(self.backend),
            "example/new-chat",
        )
        self.assertIn("gui_models_change_applied", self.trace_text())
        self.assertIn("chat_model_switched", self.trace_text())

    def test_embedding_change_forces_rebuild_and_keeps_chat(self):
        conversation = self.engine.active_conversation.conversation_id
        self.request["embedding_model"] = "example/new-embedding"
        self.engine.apply_model_settings(self.request)
        self.assertEqual(self.events[-1]["event"], "models_applied")
        self.assertTrue(self.events[-1]["reindexed"])
        self.assertEqual(len(self.embedded), 1)
        self.assertEqual(
            self.engine.active_conversation.conversation_id, conversation
        )
        self.assertIn("example/new-embedding", self.trace_text())
        self.assertIn("gui_models_change_confirmed", self.trace_text())

    def test_download_requires_consent_and_persists_custom_slot(self):
        levels = default_effort_levels(self.backend)
        levels[0]["model"] = "example/uncached"
        self.request["effort_levels"] = levels
        with patch.object(
            self.engine, "_download_with_progress", return_value=0.1
        ) as download:
            self.engine.apply_model_settings(self.request)
            self.assertEqual(self.events[-1]["event"], "error")
            download.assert_not_called()
            self.request["download"] = True
            self.engine.apply_model_settings(self.request)
            self.assertEqual(self.events[-1]["event"], "models_applied")
            download.assert_called_once()
        self.assertEqual(
            read_effort_levels(self.project.root, self.backend)[0]["model"],
            "example/uncached",
        )
        self.assertIn("gui_model_download_completed", self.trace_text())

    def test_failed_model_load_keeps_profile_and_session(self):
        self.request["chat_model"] = "example/new-chat"
        original_profile = (self.project.root / "profile.ini").read_text()
        with patch.object(
            self.engine.session,
            "use_chat_model",
            side_effect=ValueError("incompatible model"),
        ):
            self.engine.apply_model_settings(self.request)
        self.assertEqual(self.events[-1]["event"], "error")
        self.assertEqual(
            (self.project.root / "profile.ini").read_text(), original_profile
        )
        self.assertIsNotNone(self.engine.session)

    def test_incremental_add_modify_delete_matches_full_index(self):
        first = self.engine.session.indexes
        new_file = self.root / "new.txt"
        new_file.write_text("Added document about feature elimination.")
        self.engine.add_project_sources([str(new_file)])
        self.assertEqual(self.events[-1]["event"], "opened")
        self.assertEqual(self.embedded, [new_file.read_text()])
        self.assertNotIn("Original document about memory.", self.counted)
        original_vector = dict(
            zip(
                (c.identifier for c in first.semantic.chunks),
                first.semantic.embeddings,
            )
        )
        for chunk, vector in zip(
            self.engine.session.chunks,
            self.engine.session.indexes.semantic.embeddings,
        ):
            if chunk.identifier in original_vector:
                self.assertEqual(vector, original_vector[chunk.identifier])
        self.assertIn("workspace_source_copied", self.trace_text())
        self.assertIn("index_incremental_plan", self.trace_text())
        self.embedded.clear()
        new_file.write_text("Updated document about feature elimination.")
        self.engine.add_project_sources([str(new_file)])
        self.assertEqual(self.embedded, [new_file.read_text()])
        (self.project.source / "new.txt").unlink()
        self.embedded.clear()
        self.engine.resume_project(str(self.project.root))
        self.assertEqual(self.embedded, [])
        incremental = self.engine.session.indexes
        (self.project.indexes / "settings.json").unlink()
        self.engine.resume_project(str(self.project.root))
        self.assertEqual(self.engine.session.indexes, incremental)

    def test_remove_source_deletes_copy_and_reindexes(self):
        new_file = self.root / "new.txt"
        new_file.write_text("Added document about feature elimination.")
        self.engine.add_project_sources([str(new_file)])
        copy = self.project.source / "new.txt"
        self.assertTrue(copy.is_file())
        self.assertIn(
            Path("new.txt"),
            {chunk.path for chunk in self.engine.session.chunks},
        )
        self.embedded.clear()
        self.engine.remove_project_sources(["sources/new.txt"])
        self.assertEqual(self.events[-1]["event"], "opened")
        self.assertFalse(copy.exists())
        self.assertTrue(new_file.is_file())
        self.assertEqual(self.embedded, [])
        self.assertNotIn(
            Path("new.txt"),
            {chunk.path for chunk in self.engine.session.chunks},
        )
        self.assertIn("gui_sources_remove_requested", self.trace_text())

    def test_remove_source_rejects_paths_outside_sources(self):
        for path in ("sources", "profile.ini", "sources/../profile.ini"):
            with self.subTest(path=path):
                self.engine.remove_project_sources([path])
                self.assertEqual(self.events[-1]["event"], "error")
                self.assertTrue((self.project.root / "profile.ini").is_file())
                self.assertTrue(self.project.source.is_dir())
        self.assertIsNotNone(self.engine.session)

    def test_effort_levels_file_holds_only_customized_levels(self):
        light = "Qwen/Qwen3-0.6B-MLX-4bit"
        root = self.project.root
        legacy = root / "model_choices.json"
        legacy.write_text(
            '{"retired/preset": "example/old", "%s": "example/new"}' % light
        )
        # Earlier per-preset replacements migrate; retired presets are ignored.
        levels = read_effort_levels(root, "mlx")
        self.assertEqual(
            [level["model"] for level in levels[:2]], ["example/new"] * 2
        )
        write_effort_levels(root, "mlx", default_effort_levels("mlx"))
        self.assertFalse(legacy.exists())
        self.assertFalse((root / "effort_levels.json").exists())
        levels[0]["name"] = "Quick"
        write_effort_levels(root, "mlx", levels)
        self.assertEqual(read_effort_levels(root, "mlx")[0]["name"], "Quick")
        self.assertEqual(
            read_effort_levels(root, "torch"), default_effort_levels("torch")
        )

    def test_effort_uses_saved_model_and_reference_levels(self):
        levels = default_effort_levels(self.backend)
        levels[0]["model"] = "example/new-chat"
        levels[3]["retrieval"] = 7
        self.request["effort_levels"] = levels
        self.engine.apply_model_settings(self.request)
        self.assertEqual(self.events[-1]["event"], "models_applied")
        self.engine.session.chat_session.backend = "mlx"
        self.engine.session.turns = [SimpleNamespace()]
        with (
            patch.object(self.engine.session, "use_chat_model") as use,
            patch.object(
                self.engine.session,
                "ask",
                side_effect=RuntimeError("stop before generation"),
            ) as ask,
        ):
            self.engine.ask("test", effort={"model": 0, "retrieval": 3})
        use.assert_called_once_with("example/new-chat")
        self.assertEqual(ask.call_args.kwargs["retrieval_count"], 7)

    def test_new_chat_is_named_after_its_first_answer(self):
        from qachat.rag import ConversationTurn

        session = self.engine.session
        session.chat_session.backend = "mlx"
        session.turns = []
        order = []

        def answer(question, **options):
            session.turns.append(
                ConversationTurn(
                    "now", question, "RFE is Recursive...", (), {}
                )
            )

        def name(question, model, answer_text=""):
            order.append(("name", question, model, answer_text))

        sent = self.engine._send
        with (
            patch.object(session, "use_chat_model"),
            patch.object(session, "ask", side_effect=answer),
            patch.object(
                self.engine, "_name_active_conversation", side_effect=name
            ),
            patch.object(
                self.engine,
                "_send",
                side_effect=lambda event, **data: (
                    order.append((event,)),
                    sent(event, **data),
                ),
            ),
        ):
            self.engine.ask("what is RFE", effort="light")
        events = [entry[0] for entry in order]
        self.assertLess(events.index("answered"), events.index("name"))
        _, question, model, answer_text = order[events.index("name")]
        self.assertEqual(question, "what is RFE")
        self.assertEqual(answer_text, "RFE is Recursive...")
        # Light answers with 0.6B, so the title uses the Lagom model.
        self.assertEqual(model, self.project.profile.chat_model("mlx"))

    def test_answer_text_is_streamed_before_the_saved_turn(self):
        from qachat.rag import ConversationTurn

        session = self.engine.session
        session.turns = [SimpleNamespace()]

        def answer(question, **options):
            for piece in ("Hello ", "streamed ", "world."):
                options["token_callback"](piece)
            session.turns.append(
                ConversationTurn(
                    "now", question, "Hello streamed world.", (), {}
                )
            )

        with patch.object(session, "ask", side_effect=answer):
            self.engine.ask("question")
        names = [event["event"] for event in self.events]
        deltas = [
            e["text"] for e in self.events if e["event"] == "answer_delta"
        ]
        self.assertEqual("".join(deltas), "Hello streamed world.")
        self.assertLess(names.index("answer_delta"), names.index("answered"))

    def _follow_up(self, decision, answering_model=None):
        """Ask a follow-up after one answered turn; return search calls."""

        from qachat.rag import ConversationTurn, Evidence

        session = self.engine.session
        evidence = (
            Evidence(1, Path("one.txt"), 1, 1, None, "Memory capacity."),
        )
        session.turns = [
            ConversationTurn(
                "t", "What is memory capacity?", "It is.", evidence, {}
            )
        ]
        if answering_model:
            session.chat_session.model_id = answering_model
        searches = []

        def retrieve(text, count=None, **options):
            searches.append(text)
            return evidence

        with (
            patch.object(session, "use_chat_model"),
            patch.object(session, "_retrieve", side_effect=retrieve),
            # The fixture's stub embedding model cannot be saved.
            patch.object(session, "_save_conversation"),
            patch.object(
                session.chat_session, "reply", return_value="Answer [1]."
            ),
            patch.object(
                self.engine, "_side_reply", return_value=decision
            ) as side,
        ):
            self.engine.ask("And how is it measured?")
        self.assertEqual(self.events[-1]["event"], "answered", self.events)
        return searches, side, self.events[-1]["turn"]["parameters"]

    def test_follow_up_can_reuse_previous_references(self):
        searches, side, parameters = self._follow_up("REUSE")
        side.assert_called_once()
        self.assertEqual(searches, [])
        self.assertTrue(parameters["evidence_reused"])
        phases = [e.get("phase") for e in self.events]
        self.assertIn("identifying_intent", phases)
        self.assertIn("gui_intent_decided", self.trace_text())

    def test_follow_up_can_search_with_a_rewritten_query(self):
        searches, _, parameters = self._follow_up(
            "SEARCH: how memory capacity is measured"
        )
        self.assertEqual(searches, ["how memory capacity is measured"])
        self.assertEqual(
            parameters["retrieval_query"], "how memory capacity is measured"
        )

    def test_small_models_skip_intent_and_search_as_asked(self):
        searches, side, parameters = self._follow_up(
            "REUSE", answering_model="Qwen/Qwen3-0.6B-MLX-4bit"
        )
        side.assert_not_called()
        self.assertEqual(searches, ["And how is it measured?"])
        self.assertNotIn("evidence_reused", parameters)


if __name__ == "__main__":
    unittest.main()
