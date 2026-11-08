"""Exercise GUI model transitions, source refresh, and answer streaming.

Real project/index files are used, with mocked local model sessions.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kilin.gui.engine import ChatEngine
from kilin.gui.model_settings import read_effort_levels
from kilin.gui.preferences import default_effort_levels
from kilin.project import open_project
from kilin.project_files import list_project_files
from kilin.workspace import prepare_sources_workspace


class TestEngineModelTransitions(unittest.TestCase):
    """Maintain project settings and index state through GUI changes."""

    def setUp(self):
        """Prepare disposable project files and mocked model sessions."""
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        source = self.root / "one.txt"
        source.write_text("Original document about memory.", encoding="utf-8")
        user_preferences = patch.dict(
            "os.environ",
            {"KILIN_DEFAULTS_FILE": str(self.root / "preferences.json")},
        )
        user_preferences.start()
        self.addCleanup(user_preferences.stop)
        self.project = prepare_sources_workspace(
            [source], "literature", self.root / "project"
        )
        self.embedded = []
        self.counted = []

        def embed(texts):
            """Record fake embeddings for each indexed text."""
            self.embedded.extend(texts)
            return [[1.0, float(len(text))] for text in texts]

        def count(text):
            """Record token-count requests without a tokenizer."""
            self.counted.append(text)
            return len(text.split())

        def embedding_session(model_id: str, **_options):
            """Keep fake index identities tied to configured model IDs."""

            return SimpleNamespace(
                model_source=str(self.root),
                model_id=model_id,
                _retrieval_model_identity=f"test:{model_id}",
                count_tokens=count,
                embed_documents=embed,
                _clear_backend_cache=lambda: None,
                close=lambda: None,
            )
        backend = patch("kilin.rag.choose_backend", return_value="mlx")
        backend.start()
        self.addCleanup(backend.stop)
        chat_backend = patch(
            "kilin.chat._choose_backend", return_value="mlx"
        )
        chat_backend.start()
        self.addCleanup(chat_backend.stop)
        for target, options in (
            (
                "kilin.rag.EmbeddingSession",
                {"side_effect": embedding_session},
            ),
            ("kilin.chat.ChatSession._load", {}),
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
            "kilin.gui.engine.downloaded_models",
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
        """Return the active trace text for assertions."""
        return self.engine.active_trace.readable_path.read_text(
            encoding="utf-8"
        )

    def test_source_card_settings_save_without_rebuilding_indexes(self):
        """Persist card settings without changing chunk or answer models."""
        from kilin.model_settings import (
            read_source_card_generation_settings,
        )
        from kilin.source_card_policy import read_source_card_policy
        from kilin.project_type import project_type

        session = self.engine.session
        indexes = session.indexes
        chat = session.chat_session
        settings = self.engine._source_card_settings(
            self.project.root, "literature"
        )
        settings["policy"]["automatic"] = False
        settings["models"]["mlx"]["model_id"] = "example/card-mlx"
        settings["models"]["torch"]["model_id"] = "example/card-torch"
        with patch.object(session, "refresh_source_cards") as refresh:
            self.engine.save_source_card_settings(settings)
        refresh.assert_called_once()
        self.assertEqual(self.events[-1]["event"], "source_cards_applied")
        self.assertIs(session.indexes, indexes)
        self.assertIs(session.chat_session, chat)
        self.assertEqual(
            read_source_card_generation_settings(
                self.project.root, "mlx"
            ).model_id, "example/card-mlx",
        )
        self.assertEqual(
            read_source_card_generation_settings(
                self.project.root, "torch"
            ).model_id, "example/card-torch",
        )
        self.assertFalse(read_source_card_policy(
            self.project.root, project_type("literature")
        ).automatic)
        self.assertEqual(
            self.events[-1]["source_card_summary"]["total"],
            len(session.source_cards.records),
        )

    def test_invalid_card_settings_do_not_modify_profile(self):
        """Reject an inconsistent request without writing settings."""
        path = self.project.root / "profile.ini"
        original = path.read_bytes()
        settings = self.engine._source_card_settings(
            self.project.root, "literature"
        )
        settings["models"]["mlx"]["max_input_tokens"] = -5
        with patch.object(self.engine.session,
                          "refresh_source_cards") as refresh:
            self.engine.save_source_card_settings(settings)
        refresh.assert_not_called()
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.events[-1]["event"], "error")

    def test_failed_card_refresh_rolls_back_project_configuration(self):
        """Restore original core settings after a refresh failure."""
        path = self.project.root / "profile.ini"
        original = path.read_bytes()
        settings = self.engine._source_card_settings(
            self.project.root, "literature"
        )
        settings["policy"]["automatic"] = False
        with patch.object(
            self.engine.session, "refresh_source_cards",
            side_effect=[OSError("card refresh failed"), None],
        ) as refresh:
            self.engine.save_source_card_settings(settings)
        self.assertEqual(refresh.call_count, 2)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.events[-1]["event"], "error")

    def test_card_regeneration_uses_existing_session(self):
        """Regenerate cards without rebuilding other index channels."""
        session = self.engine.session
        index = session.indexes
        with (
            patch.object(session, "regenerate_source_cards") as regenerate,
            patch.object(session, "refresh_sources") as full_refresh,
        ):
            self.engine.regenerate_source_cards()
        regenerate.assert_called_once()
        full_refresh.assert_not_called()
        self.assertIs(session.indexes, index)
        self.assertEqual(
            self.events[-1]["event"], "source_cards_regenerated"
        )

    def test_chat_change_does_not_reindex_and_is_traced(self):
        """Switch the chat model without recalculating embeddings."""
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
        """Rebuild indexes while preserving the selected conversation."""
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
        """Require download confirmation before persisting model edits."""
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
        """Keep the original session and profile after a failed switch."""
        self.request["chat_model"] = "example/new-chat"
        original_profile = (self.project.root / "profile.ini").read_text(
            encoding="utf-8"
        )
        with patch.object(
            self.engine.session,
            "use_chat_model",
            side_effect=ValueError("incompatible model"),
        ):
            self.engine.apply_model_settings(self.request)
        self.assertEqual(self.events[-1]["event"], "error")
        self.assertEqual(
            (self.project.root / "profile.ini").read_text(encoding="utf-8"),
            original_profile,
        )
        self.assertIsNotNone(self.engine.session)

    def test_incremental_add_preserves_snapshot_and_index(self):
        """Preserve an existing snapshot when the original file changes."""
        first = self.engine.session.indexes
        new_file = self.root / "new.txt"
        new_file.write_text(
            "Added document about feature elimination.", encoding="utf-8"
        )
        self.engine.add_project_sources([str(new_file)])
        self.assertEqual(self.events[-1]["event"], "opened")
        # Paper passages are embedded after their search header: the title
        # (here the file name) and the file name.
        self.assertEqual(
            self.embedded,
            ["new\nnew.txt\n\n" + new_file.read_text(encoding="utf-8")],
        )
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
        new_file.write_text(
            "Updated document about feature elimination.", encoding="utf-8"
        )
        self.engine.add_project_sources([str(new_file)])
        self.assertEqual(self.embedded, [])
        self.assertEqual(
            (self.project.source / "new.txt").read_text(encoding="utf-8"),
            "Added document about feature elimination.",
        )
        (self.project.source / "new.txt").unlink()
        self.embedded.clear()
        self.engine.resume_project(str(self.project.root))
        self.assertEqual(self.embedded, [])
        incremental = self.engine.session.indexes
        (self.project.indexes / "settings.json").unlink()
        self.engine.resume_project(str(self.project.root))
        self.assertEqual(self.engine.session.indexes, incremental)

    def test_only_new_or_changed_files_are_read_again(self):
        """Avoid rereading unchanged source snapshots during refresh."""
        import kilin.rag as rag

        reads = []
        original = rag.read_repository_files

        def counting(report, **options):
            """Record the file names read during incremental retrieval."""
            reads.append(list(report["selected_files"]))
            return original(report, **options)

        new_file = self.root / "new.txt"
        new_file.write_text(
            "Added document about feature elimination.", encoding="utf-8"
        )
        with patch.object(rag, "read_repository_files", counting):
            self.engine.add_project_sources([str(new_file)])
            # The unchanged file's text comes from the cache.
            self.assertEqual(reads[-1], ["new.txt"])
            # Same name, other content: read again and indexed again.
            copy = self.project.source / "one.txt"
            copy.write_text(
                "Rewritten document about attention.", encoding="utf-8"
            )
            self.engine.resume_project(str(self.project.root))
            self.assertEqual(reads[-1], ["one.txt"])
        texts = " ".join(chunk.text for chunk in self.engine.session.chunks)
        self.assertIn("Rewritten document about attention.", texts)
        self.assertNotIn("Original document about memory.", texts)

    def test_remove_source_deletes_copy_and_reindexes(self):
        """Delete only managed source copies and refresh the index."""
        new_file = self.root / "new.txt"
        new_file.write_text(
            "Added document about feature elimination.", encoding="utf-8"
        )
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

    def test_check_lists_files_to_add_and_left_out(self):
        """The check before adding copies nothing and explains the rest."""
        folder = self.root / "notes"
        folder.mkdir()
        (folder / "kept.txt").write_text(
            "Added note about attention.", encoding="utf-8"
        )
        (folder / "data.json").write_text("{}", encoding="utf-8")
        (folder / ".DS_Store").write_text("finder", encoding="utf-8")
        code = self.root / "ops"
        code.mkdir()
        (code / "edges.py").write_text("x = 1\n", encoding="utf-8")
        note = self.root / "note.json"
        note.write_text("{}", encoding="utf-8")
        sources = [str(folder), str(code), str(note)]
        self.events.clear()
        self.engine.check_sources(sources)
        self.assertEqual(
            self.events,
            [{
                "event": "sources_checked",
                "sources": sources,
                "add": [
                    {
                        "name": "notes", "kind": "folder", "files": 1,
                        "state": "new",
                    },
                    {
                        "name": "note.json", "kind": "file", "files": 1,
                        "state": "new",
                    },
                ],
                "files": [
                    {
                        "path": "notes/data.json",
                        "reason_code": "file_type_not_selected",
                    },
                    {
                        "path": "ops/edges.py",
                        "reason_code": "file_type_not_selected",
                    },
                ],
                "skipped_folders": ["ops"],
                "changes": {
                    "added": [], "changed": [], "deleted": [], "edited": [],
                },
            }],
        )
        self.assertFalse((self.project.source / "notes").exists())

    def test_check_recognises_sources_already_in_the_project(self):
        """Same original: unchanged or changed; same name: renamed."""
        note = self.root / "note.txt"
        note.write_text("First note about attention.", encoding="utf-8")
        os.utime(note, ns=(10**18, 10**18))
        self.engine.add_project_sources([str(note)])
        copy = self.project.source / "note.txt"
        self.assertTrue(copy.is_file())

        def check(path: Path) -> dict:
            self.events.clear()
            self.engine.check_sources([str(path)])
            return self.events[-1]

        unchanged = check(note)
        self.assertEqual(unchanged["add"][0]["state"], "existing")
        self.assertEqual(unchanged["changes"]["changed"], [])
        other = self.root / "elsewhere" / "note.txt"
        other.parent.mkdir()
        other.write_text("A different note.", encoding="utf-8")
        self.assertEqual(check(other)["add"][0]["state"], "renamed")

        note.write_text("Updated note about attention.", encoding="utf-8")
        os.utime(note, ns=(2 * 10**18, 2 * 10**18))
        self.assertEqual(
            check(note)["changes"]["changed"], ["sources/note.txt"]
        )
        # Adding again without a choice keeps the old copy.
        self.engine.add_project_sources([str(note)])
        self.assertIn("First", copy.read_text(encoding="utf-8"))
        self.engine.add_project_sources(
            [str(note)], {"sources/note.txt": "replace"}
        )
        self.assertIn("Updated", copy.read_text(encoding="utf-8"))
        self.engine.add_project_sources([str(note)], ["not", "a", "dict"])
        self.assertEqual(self.events[-1]["event"], "error")

    def test_check_rejects_unimportable_direct_files(self):
        """An invalid direct file is never announced as one to add."""

        empty = self.root / "empty.txt"
        empty.write_bytes(b"")
        large = self.root / "large.txt"
        large.write_bytes(
            b"x" * (self.project.profile.max_file_bytes + 1)
        )
        paths = [str(empty), str(large)]
        self.events.clear()
        self.engine.check_sources(paths)
        checked = self.events[-1]
        self.assertEqual(checked["event"], "sources_checked")
        self.assertEqual(checked["add"], [])
        self.assertEqual(checked["files"], [
            {"path": "empty.txt", "reason_code": "empty_file"},
            {"path": "large.txt", "reason_code": "file_too_large"},
        ])
        self.assertFalse((self.project.source / "empty.txt").exists())
        self.assertFalse((self.project.source / "large.txt").exists())

        for invalid in (self.root / "missing.txt",):
            with self.subTest(invalid=invalid):
                self.events.clear()
                self.engine.check_sources([str(invalid)])
                self.assertEqual(self.events[-1]["event"], "error")

    def test_check_allocates_duplicate_names_before_confirmation(self):
        """The preview reports the names used by a mixed source batch."""

        from hashlib import sha256

        a = self.root / "a" / "same.txt"
        b = self.root / "b" / "same.txt"
        for path in (a, b):
            path.parent.mkdir()
            path.write_text(str(path), encoding="utf-8")
        self.events.clear()
        self.engine.check_sources([str(a), str(b)])
        plan = self.events[-1]
        self.assertEqual(plan["event"], "sources_checked")
        self.assertEqual(plan["add"][0]["state"], "new")
        self.assertEqual(plan["add"][1]["state"], "renamed")
        self.assertEqual(
            plan["add"][1]["name"],
            f"same-{sha256(str(b).encode()).hexdigest()[:8]}.txt",
        )
        self.engine.add_project_sources([str(a), str(b)])
        self.assertTrue((self.project.source / "same.txt").is_file())
        self.assertTrue(
            (self.project.source / plan["add"][1]["name"]).is_file()
        )

    def test_folder_without_project_files_is_skipped(self):
        """Adding skips a folder with nothing to copy instead of failing."""
        code = self.root / "ops"
        code.mkdir()
        (code / "edges.py").write_text("x = 1\n", encoding="utf-8")
        note = self.root / "note.txt"
        note.write_text("Another note about attention.", encoding="utf-8")
        for sources in ([code, note], [code]):
            with self.subTest(sources=len(sources)):
                self.events.clear()
                self.engine.add_project_sources(
                    [str(path) for path in sources]
                )
                names = [event["event"] for event in self.events]
                self.assertNotIn("error", names)
                self.assertIn("opened", names)
        self.assertTrue((self.project.source / "note.txt").is_file())
        self.assertFalse((self.project.source / "ops").exists())

    def test_remove_source_rejects_paths_outside_sources(self):
        """Reject source-removal requests targeting project metadata."""
        for path in ("sources", "profile.ini", "sources/../profile.ini"):
            with self.subTest(path=path):
                self.engine.remove_project_sources([path])
                self.assertEqual(self.events[-1]["event"], "error")
                self.assertTrue((self.project.root / "profile.ini").is_file())
                self.assertTrue(self.project.source.is_dir())
        self.assertIsNotNone(self.engine.session)

    def test_answer_text_is_streamed_before_the_saved_turn(self):
        """Deliver text increments before publishing the completed turn."""
        from kilin.rag import ConversationTurn

        session = self.engine.session
        session.turns = [SimpleNamespace()]

        def answer(question, **options):
            """Produce a deterministic answer through the fake callback."""
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

    def test_every_answer_refreshes_the_workspace_tree(self):
        """Send the current project files with each answer, tool or not."""
        from kilin.rag import ConversationTurn

        session = self.engine.session
        session.turns = []

        def answer(question, **options):
            """Save a plain answer that used no tool."""
            session.turns.append(
                ConversationTurn("now", question, "Plain answer.", (), {})
            )

        with patch.object(session, "ask", side_effect=answer):
            self.engine.ask("question")
        answered = [e for e in self.events if e["event"] == "answered"]
        self.assertEqual(len(answered), 1)
        self.assertEqual(
            answered[0]["workspace_files"],
            list_project_files(session.project),
        )

    def test_new_chat_is_named_after_its_first_answer(self):
        """Name a new conversation only after its answer is available."""
        from kilin.rag import ConversationTurn

        session = self.engine.session
        session.chat_session.backend = "mlx"
        session.turns = []
        order = []

        def answer(question, **options):
            """Produce a deterministic answer through the fake callback."""
            session.turns.append(
                ConversationTurn(
                    "now", question, "RFE is Recursive...", (), {}
                )
            )

        def name(question, answer_text=""):
            """Record the order of generated conversation names."""
            order.append(("name", question, answer_text))

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
        _, question, answer_text = order[events.index("name")]
        self.assertEqual(question, "what is RFE")
        self.assertEqual(answer_text, "RFE is Recursive...")

    def test_router_model_is_saved_and_applied(self):
        """Persist the router model and update the active session."""
        from kilin.gui.model_settings import (
            default_router_model,
            read_router_model,
        )

        root = self.project.root
        self.request["router_model"] = "example/new-chat"
        self.engine.apply_model_settings(self.request)
        self.assertEqual(self.events[-1]["event"], "models_applied")
        self.assertEqual(self.engine.session.router_model, "example/new-chat")
        self.assertEqual(
            read_router_model(root, self.backend), "example/new-chat"
        )
        self.assertEqual(
            self.events[-1]["router_models"][self.backend], "example/new-chat"
        )
        # Restoring the default is saved like any other choice.
        self.request["router_model"] = default_router_model(self.backend)
        self.engine.apply_model_settings(self.request)
        self.assertEqual(
            read_router_model(root, self.backend),
            default_router_model(self.backend),
        )
        self.assertEqual(
            self.engine.session.router_model,
            default_router_model(self.backend),
        )
