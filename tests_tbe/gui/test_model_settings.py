"""Model edits preserve chats and request only authorized downloads."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qachat.gui.engine import ChatEngine
from qachat.gui.model_settings import (
    read_effort_levels,
    write_effort_levels,
)
from qachat.gui.preferences import default_effort_levels
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

    def test_only_new_or_changed_files_are_read_again(self):
        import qachat.rag as rag

        reads = []
        original = rag.read_repository_files

        def counting(report, **options):
            reads.append(list(report["selected_files"]))
            return original(report, **options)

        new_file = self.root / "new.txt"
        new_file.write_text("Added document about feature elimination.")
        with patch.object(rag, "read_repository_files", counting):
            self.engine.add_project_sources([str(new_file)])
            # The unchanged file's text comes from the cache.
            self.assertEqual(reads[-1], ["new.txt"])
            # Same name, other content: read again and indexed again.
            copy = self.project.source / "one.txt"
            copy.write_text("Rewritten document about attention.")
            self.engine.resume_project(str(self.project.root))
            self.assertEqual(reads[-1], ["one.txt"])
        texts = " ".join(chunk.text for chunk in self.engine.session.chunks)
        self.assertIn("Rewritten document about attention.", texts)
        self.assertNotIn("Original document about memory.", texts)

    def test_outputs_file_can_become_a_source(self):
        summary = self.project.root / "outputs" / "paper_summary.md"
        summary.parent.mkdir(exist_ok=True)
        summary.write_text("# Briefing\n\nA summary about memory.")
        self.engine.add_project_sources([str(summary)])
        self.assertEqual(self.events[-1]["event"], "opened")
        self.assertTrue(
            (self.project.source / "paper_summary.md").is_file()
        )
        # The rest of the workspace still cannot become a source.
        self.engine.add_project_sources([str(self.project.root / "logs")])
        self.assertEqual(self.events[-1]["event"], "error")
        self.assertIn("own source", self.events[-1]["message"])

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

    def _as_earlier_version(self):
        """Remove the GUI sections, as a project of an earlier version."""

        from configparser import ConfigParser

        path = self.project.root / "profile.ini"
        profile = ConfigParser(interpolation=None)
        profile.read(path)
        for section in ("effort.mlx", "effort.torch", "search", "router"):
            profile.remove_section(section)
        with path.open("w") as stream:
            profile.write(stream)

    def test_project_keeps_every_setting_in_profile_ini(self):
        import json

        from qachat.gui.model_settings import (
            read_router_model,
            read_search_depth,
            save_project_settings,
        )
        from qachat.gui.preferences import defaults_file, write_defaults

        root = self.project.root
        # Opening the project saved all settings in profile.ini.
        text = (root / "profile.ini").read_text()
        for section in ("[effort.mlx]", "[effort.torch]", "[search]",
                        "[router]"):
            self.assertIn(section, text)
        light = default_effort_levels("mlx")[0]
        self.assertIn(f"level1 = {light['name']} | {light['model']}", text)
        self.assertIn("depth = 5, 5, 10, 10, 15", text)
        self.assertFalse((root / "effort_levels.json").exists())
        self.assertFalse(save_project_settings(root))
        # The core library still reads the profile.
        self.assertEqual(open_project(root).profile.kind, "literature")
        # Later changes to the defaults do not reach an existing project.
        backup = defaults_file().read_text()
        self.addCleanup(defaults_file().write_text, backup)
        levels = default_effort_levels("mlx")
        levels[4]["model"] = "example/other"
        write_defaults({"effort_levels": {"mlx": levels}})
        self.assertNotEqual(read_effort_levels(root, "mlx"), levels)
        # A project of an earlier version moves effort_levels.json in.
        self._as_earlier_version()
        old = default_effort_levels("mlx")
        old[0]["name"] = "Quick"
        (root / "effort_levels.json").write_text(
            json.dumps(
                {
                    "mlx": old,
                    "search_depth": [3, 3, 6, 6, 9],
                    "router": {"mlx": "example/router"},
                }
            )
        )
        self.assertTrue(save_project_settings(root))
        self.assertFalse((root / "effort_levels.json").exists())
        self.assertEqual(read_effort_levels(root, "mlx")[0]["name"], "Quick")
        self.assertEqual(read_search_depth(root), [3, 3, 6, 6, 9])
        self.assertEqual(read_router_model(root, "mlx"), "example/router")

    def test_settings_without_a_project_become_the_defaults(self):
        from qachat.gui.preferences import defaults_file

        backup = defaults_file().read_text()
        self.addCleanup(defaults_file().write_text, backup)
        self.engine.close()
        levels = default_effort_levels(self.backend)
        levels[0]["model"] = "example/new-chat"
        self.engine.apply_model_settings(
            {**self.request, "effort_levels": levels, "backend": self.backend}
        )
        self.assertEqual(self.events[-1]["event"], "models_applied")
        self.assertEqual(default_effort_levels(self.backend), levels)

    def test_answer_settings_belong_to_the_project(self):
        from qachat.project import open_project

        def ask(temperature, thinking=False):
            with patch.object(
                self.engine.session,
                "ask",
                side_effect=RuntimeError("stop before generation"),
            ):
                self.engine.ask(
                    "test",
                    generation_settings={
                        "temperature": temperature,
                        "thinking": thinking,
                        "max_new_tokens": 300,
                    },
                )

        first = self.engine.active_conversation.conversation_id
        ask(0.4, thinking=True)
        profile = open_project(self.project.root).profile
        self.assertEqual(
            (profile.temperature, profile.thinking, profile.max_new_tokens),
            (0.4, True, 300),
        )
        # A new chat and a switch back keep the project's settings.
        self.engine.create_conversation()
        chat = self.engine.session.chat_session
        self.assertEqual((chat.temperature, chat.thinking), (0.4, True))
        ask(0.7)
        self.engine.select_conversation(first)
        chat = self.engine.session.chat_session
        self.assertEqual((chat.temperature, chat.thinking), (0.7, False))
        self.assertEqual(self.events[-1]["temperature"], 0.7)

    def test_tools_get_the_chain_of_earlier_turns(self):
        from qachat.answering import RouteDecision
        from qachat.rag import ConversationTurn, Evidence

        def turn(question, related, path=None):
            evidence = (
                (Evidence(1, path, 1, 2, None, "text", None),)
                if path
                else ()
            )
            return ConversationTurn(
                "now", question, "A", evidence,
                {"related_turn_ids": related},
            )

        session = self.engine.session
        session.turns = [
            turn("hi", []),
            turn("what is memory capacity", [], "m.pdf"),
            turn("walk me through Park's paper", [], "t.pdf"),
            turn("turn that into a summary report", [2, 3], "t.pdf"),
        ]
        earlier = session._earlier_turns(
            RouteDecision(intent="x", standalone=False, related_turn_ids=(4,))
        )
        self.assertEqual([t.number for t in earlier], [4, 3, 2])
        self.assertEqual(earlier[1].question, "walk me through Park's paper")
        self.assertEqual(earlier[1].paths, (Path("t.pdf"),))
        # Pointing back without related turns means the last turn.
        pointing = session._earlier_turns(
            RouteDecision(intent="x", standalone=True, refers_back=True)
        )
        self.assertEqual([t.number for t in pointing], [4, 3, 2])
        # An older turn on the same topic: the last turn is still included,
        # unlinked, so "that paper" can mean the paper just discussed.
        older = session._earlier_turns(
            RouteDecision(intent="x", standalone=False, related_turn_ids=(2,))
        )
        self.assertEqual([t.number for t in older], [4, 2])
        self.assertEqual([t.linked for t in older], [False, True])
        # A standalone question does not point back at all.
        alone = session._earlier_turns(
            RouteDecision(intent="x", standalone=True)
        )
        self.assertEqual(alone, ())

    def test_tool_choice_waits_for_an_option(self):
        commands = iter(
            [
                {"action": "new_conversation"},
                {"action": "plan_choice", "choice": "option:1"},
                {"action": "plan_choice", "choice": "option:7"},
                {"action": "plan_choice", "choice": "cancel"},
            ]
        )
        self.engine.read_command = lambda: next(commands, None)
        options = ("a.pdf", "b.pdf")
        self.assertEqual(self.engine._choose_option("Which?", options), 1)
        self.assertEqual(self.events[-1]["event"], "error")
        self.assertIsNone(self.engine._choose_option("Which?", options))
        self.assertIsNone(self.engine._choose_option("Which?", options))
        self.assertIsNone(self.engine._choose_option("Which?", options))

    def test_token_usage_per_answer_chat_and_project(self):
        import json

        from qachat.rag import ConversationTurn

        session = self.engine.session
        chat = session.chat_session
        chat.input_token_total, chat.output_token_total = 100, 10
        mark = session.usage_mark()
        chat.input_token_total += 2500
        chat.output_token_total += 300
        self.assertEqual(
            session.usage_since(mark),
            {"input_tokens": 2500, "output_tokens": 300},
        )

        def turn(question, read, written):
            return ConversationTurn(
                timestamp=f"2026-10-07T{question}",
                question=question,
                answer="A",
                evidence=(),
                parameters={
                    "usage": {"input_tokens": read, "output_tokens": written}
                },
            )

        session.turns = [turn("one", 2000, 200)]
        # Another chat, deleted later, shares the first turn (a branch).
        deleted = self.project.root / "logs" / "traces" / "deleted"
        deleted.mkdir(parents=True, exist_ok=True)
        (deleted / "conversation_old.json").write_text(
            json.dumps(
                {
                    "conversation": {
                        "turns": [
                            {
                                "timestamp": "2026-10-07Tone",
                                "question": "one",
                                "parameters": {
                                    "usage": {
                                        "input_tokens": 2000,
                                        "output_tokens": 200,
                                    }
                                },
                            },
                            {
                                "timestamp": "2026-10-07Ttwo",
                                "question": "two",
                                "parameters": {
                                    "usage": {
                                        "input_tokens": 500,
                                        "output_tokens": 50,
                                    }
                                },
                            },
                        ]
                    }
                }
            )
        )
        totals = self.engine._usage_totals()
        self.assertEqual(
            totals["chat"],
            {"input_tokens": 2000, "output_tokens": 200, "answers": 1},
        )
        self.assertEqual(
            totals["project"],
            {"input_tokens": 2500, "output_tokens": 250, "answers": 2},
        )

    def test_folder_edits_never_reindex(self):
        root = self.project.root
        session = self.engine.session
        indexes = {
            path: path.stat().st_mtime_ns
            for path in (root / "indexes").rglob("*")
            if path.is_file()
        }
        paper = self.root / "paper.md"
        paper.write_text("# Draft\n\nFirst version.")
        other = self.root / "other.md"
        other.write_text("# Draft\n\nSecond version.")
        self.engine.edit_folder("outputs", "add", paths=[str(paper)])
        self.assertEqual(self.events[-1]["event"], "folder_changed")
        self.assertIn("outputs/paper.md", self.events[-1]["workspace_files"])
        # A taken name is kept and the copy numbered, unless replacing.
        self.engine.edit_folder("outputs", "add", paths=[str(paper)])
        self.assertEqual(self.events[-1]["paths"], ["outputs/paper-2.md"])
        self.engine.edit_folder(
            "outputs", "replace", path="outputs/paper.md", source=str(other)
        )
        self.assertIn(
            "Second", (root / "outputs" / "paper.md").read_text()
        )
        self.engine.edit_folder(
            "outputs",
            "remove",
            paths=["outputs/paper.md", "outputs/paper-2.md"],
        )
        self.assertFalse((root / "outputs" / "paper.md").exists())
        # Templates are edited the same way.
        self.engine.edit_folder("templates", "add", paths=[str(other)])
        self.assertTrue((root / "templates" / "other.md").is_file())
        # Paths outside the folder, and other folders, are refused.
        self.engine.edit_folder("outputs", "remove", paths=["sources/one.txt"])
        self.assertEqual(self.events[-1]["event"], "error")
        self.engine.edit_folder("sources", "remove", paths=["sources/one.txt"])
        self.assertEqual(self.events[-1]["event"], "error")
        self.assertTrue((root / "sources" / "one.txt").exists())
        # The session stayed open and nothing was embedded or rewritten.
        self.assertIs(self.engine.session, session)
        self.assertEqual(self.embedded, [])
        self.assertEqual(
            {
                path: path.stat().st_mtime_ns
                for path in (root / "indexes").rglob("*")
                if path.is_file()
            },
            indexes,
        )

    def test_earlier_model_choices_migrate(self):
        from qachat.gui.model_settings import save_project_settings

        light = "Qwen/Qwen3-0.6B-MLX-4bit"
        root = self.project.root
        self._as_earlier_version()
        legacy = root / "model_choices.json"
        legacy.write_text(
            '{"retired/preset": "example/old", "%s": "example/new"}' % light
        )
        # Earlier per-preset replacements migrate; retired presets are ignored.
        levels = read_effort_levels(root, "mlx")
        self.assertEqual(levels[0]["model"], "example/new")
        self.assertEqual(levels[1:], default_effort_levels("mlx")[1:])
        save_project_settings(root)
        self.assertFalse(legacy.exists())
        self.assertEqual(read_effort_levels(root, "mlx"), levels)
        levels[0]["name"] = "Quick"
        write_effort_levels(root, "mlx", levels)
        self.assertEqual(read_effort_levels(root, "mlx")[0]["name"], "Quick")
        with self.assertRaisesRegex(ValueError, "'\\|'"):
            write_effort_levels(
                root, "mlx", [{**level, "name": "A|B"} for level in levels]
            )

    def test_effort_uses_saved_model_and_search_depth(self):
        levels = default_effort_levels(self.backend)
        levels[0]["model"] = "example/new-chat"
        self.request["effort_levels"] = levels
        self.request["search_depth"] = [5, 5, 7, 10, 15]
        self.engine.apply_model_settings(self.request)
        self.assertEqual(self.events[-1]["event"], "models_applied")
        self.assertEqual(self.events[-1]["search_depth"], [5, 5, 7, 10, 15])
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
            self.engine.ask("test", effort={"model": 0, "search_depth": 2})
        use.assert_called_once_with("example/new-chat")
        self.assertEqual(ask.call_args.kwargs["retrieval_count"], 7)

    def test_search_depth_defaults_and_migration(self):
        import json

        from qachat.gui.model_settings import (
            read_search_depth,
            write_search_depth,
        )
        from qachat.gui.preferences import (
            default_search_depth,
            validate_search_depth,
        )

        root = self.project.root
        self._as_earlier_version()
        self.assertEqual(read_search_depth(None), [5, 5, 10, 10, 15])
        self.assertEqual(read_search_depth(root), default_search_depth())
        with self.assertRaisesRegex(ValueError, "Search depth"):
            validate_search_depth([5, 5, 0, 10, 15])
        write_search_depth(root, [1, 2, 3, 4, 5])
        self.assertEqual(read_search_depth(root), [1, 2, 3, 4, 5])
        self._as_earlier_version()

        def old_file(counts):
            levels = [
                {**level, "retrieval": count}
                for level, count in zip(
                    default_effort_levels("mlx"), counts
                )
            ]
            (root / "effort_levels.json").write_text(
                json.dumps({"mlx": levels})
            )

        # Earlier versions' default counts become the new defaults ...
        old_file([5, 20, 20, 50, 50])
        self.assertEqual(read_search_depth(root), [5, 5, 10, 10, 15])
        # ... and customized counts are kept as search depth.
        old_file([3, 6, 9, 12, 15])
        self.assertEqual(read_search_depth(root), [3, 6, 9, 12, 15])
        self.assertEqual(
            read_effort_levels(root, "mlx"), default_effort_levels("mlx")
        )

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

        def name(question, answer_text=""):
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

    def _follow_up(self, router_reply):
        """Ask a follow-up after one answered turn with a fixed router reply.

        Returns the searched texts, the history the model saw, and the saved
        turn parameters.
        """

        from qachat.rag import ConversationTurn, Evidence

        session = self.engine.session
        evidence = (
            Evidence(1, Path("one.txt"), 1, 1, None, "Memory capacity."),
        )
        session.turns = [
            ConversationTurn(
                "t",
                "What is memory capacity?",
                "It is a measure [1].",
                evidence,
                {},
            )
        ]
        searches = []
        seen = {}

        def retrieve(text, count=None, **options):
            searches.append(text)
            return evidence

        def reply(prompt, **options):
            seen["history"] = [dict(m) for m in session.chat_session.messages]
            seen["prompt"] = prompt
            return "Answer [1]."

        with (
            patch.object(session, "use_chat_model"),
            patch.object(session, "_retrieve", side_effect=retrieve),
            # The fixture's stub embedding model cannot be saved.
            patch.object(session, "_save_conversation"),
            patch.object(session.chat_session, "reply", side_effect=reply),
            patch(
                "qachat.answering.side_reply", return_value=router_reply
            ) as router,
        ):
            self.engine.ask("And how is it measured?")
        self.assertEqual(self.events[-1]["event"], "answered", self.events)
        seen["router"] = router
        return searches, seen, self.events[-1]["turn"]["parameters"]

    def test_follow_up_can_reuse_previous_references(self):
        searches, seen, parameters = self._follow_up(
            '{"intent": "rephrase_previous", "standalone": false, '
            '"related_turns": [1]}'
        )
        seen["router"].assert_called_once()
        self.assertEqual(searches, [])
        self.assertTrue(parameters["evidence_reused"])
        self.assertEqual(parameters["intent"], "rephrase_previous")
        self.assertEqual(parameters["machine_state"], "answering")
        phases = [e.get("phase") for e in self.events]
        self.assertIn("routing", phases)
        trace = self.trace_text()
        for name in (
            "intent_classification",
            "intent_classification_decided",
            "history_selection_result",
            "retrieval_plan",
            "conversation_turn_representation",
        ):
            self.assertIn(name, trace)

    def test_follow_up_searches_with_own_words_and_q_a_history_only(self):
        searches, seen, parameters = self._follow_up(
            '{"intent": "question_answering", "standalone": false, '
            '"related_turns": [1]}'
        )
        # "it" points back, so the related turn's question joins the search.
        self.assertEqual(
            searches, ["And how is it measured? What is memory capacity?"]
        )
        self.assertEqual(parameters["history_turn_ids"], [1])
        # Earlier evidence and citation markers stay out of the history.
        history = seen["history"]
        self.assertEqual(
            [m["content"] for m in history[1:]],
            ["What is memory capacity?", "It is a measure [source]."],
        )
        self.assertNotIn("Memory capacity.", str(history))
        self.assertIn("Answer the question directly", history[0]["content"])
        self.assertIn("representation", parameters)

    def test_unusable_router_reply_falls_back_to_default_intent(self):
        searches, seen, parameters = self._follow_up("I think it is REUSE")
        self.assertEqual(parameters["intent"], "question_answering")
        self.assertEqual(searches, ["And how is it measured?"])

    def test_standalone_question_sees_only_the_previous_turn(self):
        searches, seen, parameters = self._follow_up(
            '{"intent": "compare_sources", "standalone": true, '
            '"related_turns": []}'
        )
        # The previous turn's question and answer, never its evidence.
        self.assertEqual(parameters["history_turn_ids"], [1])
        self.assertEqual(len(seen["history"]), 3)
        self.assertIn("Compare the sources", seen["history"][0]["content"])
        # The search still uses only the question as typed.
        self.assertEqual(searches, ["And how is it measured?"])

    def test_follow_up_reuses_evidence_from_several_turns(self):
        from qachat.rag import ConversationTurn, Evidence

        session = self.engine.session
        first = (Evidence(1, Path("a.txt"), 1, 1, None, "Alpha."),)
        third = (
            Evidence(1, Path("c.txt"), 1, 1, None, "Gamma."),
            Evidence(2, Path("a.txt"), 1, 1, None, "Alpha."),
        )
        session.turns = [
            ConversationTurn("t", "About A?", "A [1].", first, {}),
            ConversationTurn("t", "About B?", "B.", (), {}),
            ConversationTurn("t", "About C?", "C [1].", third, {}),
        ]
        seen = {}

        def reply(prompt, **options):
            seen["prompt"] = prompt
            return "Both [1] [2]."

        with (
            patch.object(session, "_retrieve") as retrieve,
            patch.object(session, "_save_conversation"),
            patch.object(session.chat_session, "reply", side_effect=reply),
            patch(
                "qachat.answering.side_reply",
                return_value='{"intent": "conversation_followup", '
                '"standalone": false, "related_turns": [1, 3]}',
            ),
        ):
            self.engine.ask("How do A and C relate?")
        retrieve.assert_not_called()
        turn = self.events[-1]["turn"]
        self.assertEqual(turn["parameters"]["evidence_from_turns"], [1, 3])
        # Citations are renumbered; the repeated chunk is kept once.
        self.assertEqual(
            [(e["citation"], e["text"]) for e in turn["evidence"]],
            [(1, "Alpha."), (2, "Gamma.")],
        )
        self.assertEqual(turn["parameters"]["history_turn_ids"], [1, 2, 3])

    def test_small_talk_skips_retrieval(self):
        session = self.engine.session
        session.turns = []
        with (
            patch.object(session, "_retrieve") as retrieve,
            patch.object(session, "_save_conversation"),
            patch.object(session.chat_session, "reply", return_value="Hello!"),
            patch(
                "qachat.answering.side_reply",
                return_value='{"intent": "small_talk", "standalone": true}',
            ),
        ):
            self.engine.ask("hi there")
        retrieve.assert_not_called()
        parameters = self.events[-1]["turn"]["parameters"]
        self.assertEqual(parameters["retrieval_mode"], "none")
        self.assertEqual(list(self.events[-1]["turn"]["evidence"]), [])

    def test_document_intent_reads_in_parts_and_streams(self):
        from qachat.chunking import Chunk

        session = self.engine.session
        session.turns = []
        kind = session.project.profile.kind
        intent = "file_reading" if kind == "code" else "document_reading"
        session.chunks = [
            Chunk(f"p:{n}", Path("paper.pdf"), f"Part {n} " + "x" * 9000,
                  n, n, "text", None, None)
            for n in (1, 2, 3)
        ]
        def model(chat, system, prompt, **options):
            if "route questions" in system:
                return f'{{"intent": "{intent}", "standalone": true}}'
            if "take notes" in system:
                return "- Written in Python [2]" if "[2]" in prompt else (
                    "NOTHING"
                )
            options["token_callback"]("Python [2].")
            return "Python [2]."

        with (
            patch.object(session, "_save_conversation"),
            patch.object(session, "_rank_everything", return_value=[]),
            patch.object(session.chat_session, "reply") as reply,
            patch("qachat.answering.side_reply", side_effect=model),
            patch("qachat.rag.side_reply", side_effect=model),
        ):
            self.engine.ask("read paper.pdf: which language?")
        reply.assert_not_called()
        turn = self.events[-1]["turn"]
        self.assertEqual(turn["answer"], "Python [2].")
        self.assertEqual(len(turn["evidence"]), 3)
        tool = turn["parameters"]["tool"]
        self.assertEqual(tool["name"], "batch_review")
        self.assertEqual(tool["parts_read"], 2)
        self.assertEqual(tool["reason"], "named_file")
        deltas = [e for e in self.events if e["event"] == "answer_delta"]
        self.assertEqual("".join(e["text"] for e in deltas), "Python [2].")

    def test_mention_listing_scans_without_the_answering_model(self):
        from qachat.chunking import Chunk

        session = self.engine.session
        session.turns = []
        lines = "XYZ 2 is used.\nWe extend XYZ\n2 here.\n"
        session.chunks = [
            Chunk("c1", Path("notes.md"), lines, 1, 3, "text", None, None),
            Chunk("c2", Path("other.md"), "No match.\n", 1, 1, "text",
                  None, None),
        ]
        replies = iter(
            [
                '{"intent": "mention_listing", "standalone": true}',
                '{"terms": ["XYZ 2"]}',
            ]
        )
        with (
            patch.object(session, "_retrieve") as retrieve,
            patch.object(session, "_save_conversation"),
            patch.object(session.chat_session, "reply") as reply,
            patch(
                "qachat.answering.side_reply",
                side_effect=lambda *a, **k: next(replies),
            ),
            patch(
                "qachat.rag.side_reply",
                side_effect=lambda *a, **k: next(replies),
            ),
        ):
            self.engine.ask("list all mentions of XYZ 2")
        retrieve.assert_not_called()
        reply.assert_not_called()
        turn = self.events[-1]["turn"]
        self.assertIn("Found **2 mentions** of “XYZ 2”", turn["answer"])
        self.assertEqual(
            [(e["citation"], e["start_line"]) for e in turn["evidence"]],
            [(1, 1), (2, 1)],
        )
        tool = turn["parameters"]["tool"]
        self.assertEqual(tool["terms"], ["XYZ 2"])
        self.assertEqual(tool["terms_from"], "router")
        self.assertEqual(tool["total_mentions"], 2)
        self.assertEqual(tool["files_scanned"], 2)
        self.assertIn("tool_scan_and_list_result", self.trace_text())

        # A term that names a file in the question becomes the scope.
        replies = iter(
            [
                '{"intent": "mention_listing", "standalone": true}',
                '{"terms": ["XYZ 2", "other"], "document": null}',
            ]
        )
        session.chunks.append(
            Chunk("c3", Path("other.md"), "XYZ 2 once.\n", 2, 2, "text",
                  None, None)
        )
        session.turns = []
        with (
            patch.object(session, "_save_conversation"),
            patch.object(session.chat_session, "reply") as reply,
            patch(
                "qachat.answering.side_reply",
                side_effect=lambda *a, **k: next(replies),
            ),
            patch(
                "qachat.rag.side_reply",
                side_effect=lambda *a, **k: next(replies),
            ),
        ):
            self.engine.ask("where is XYZ 2 mentioned in other")
        reply.assert_not_called()
        tool = self.events[-1]["turn"]["parameters"]["tool"]
        self.assertEqual(tool["terms"], ["XYZ 2"])
        self.assertEqual(tool["terms_as_scope"], ["other"])
        self.assertEqual(tool["scope"], ["other.md"])
        self.assertEqual(tool["total_mentions"], 1)

    def _classify(self, choice, label_reply):
        """Ask a classification question; the user answers the plan."""

        from qachat.chunking import Chunk

        session = self.engine.session
        session.turns = []
        session.chunks = [
            Chunk("c1", Path("a.md"), "Our Python code.\nPython [3].\n",
                  1, 2, "text", None, None),
        ]
        commands = [{"action": "preview"}, {"action": "plan_choice",
                                            "choice": choice}]
        self.engine.read_command = lambda: commands.pop(0)
        prompts = []

        def router(chat, system, prompt, **options):
            if "route questions" in system:
                return (
                    '{"intent": "mention_classification", '
                    '"standalone": true}'
                )
            if "pick search terms" in system:
                return ('{"terms": ["Python"], "document": null, '
                        '"categories": ["own code", "citation"]}')
            prompts.append(prompt)
            return label_reply(len(prompts))

        with (
            patch.object(session, "_save_conversation"),
            patch.object(session.chat_session, "reply") as reply,
            patch("qachat.answering.side_reply", side_effect=router),
            patch("qachat.rag.side_reply", side_effect=router),
            patch(
                "qachat.tools.planning.needs_confirmation",
                return_value=True,
            ),
        ):
            self.engine.ask("which Python mentions are own code or citations?")
        reply.assert_not_called()
        return self.events[-1]["turn"], prompts

    def test_classification_plan_continue_with_retry(self):
        def labels(call):
            if call == 1:  # Mention 2 is missing; it is retried alone.
                return '[{"id": 1, "category": "own code", "reason": "ours"}]'
            return '[{"id": 2, "category": "citation", "reason": "ref"}]'

        turn, prompts = self._classify("continue", labels)
        plans = [
            e for e in self.events
            if e.get("phase") == "processing" and e.get("plan")
        ]
        self.assertEqual(plans[0]["plan"]["items"], 2)
        self.assertIn("Categories: own code, citation, unclear",
                      plans[0]["plan"]["details"])
        # The unrelated command during the wait was refused, not run.
        self.assertIn("Choose Continue", str(self.events))
        self.assertEqual(len(prompts), 2)
        self.assertIn("[2] a.md, line 2", prompts[1])
        self.assertNotIn("[1] a.md", prompts[1])
        tool = turn["parameters"]["tool"]
        self.assertEqual(tool["choice"], "continue")
        self.assertEqual(tool["counts"],
                         {"own code": 1, "citation": 1, "unclear": 0})
        self.assertEqual(tool["not_classified"], 0)
        self.assertEqual(tool["model_calls"], 2)
        self.assertIn("Classified **2 of 2 mentions**", turn["answer"])
        self.assertEqual(len(turn["evidence"]), 2)
        trace = self.trace_text()
        for name in ("tool_plan_proposed", "tool_plan_decided",
                     "tool_batch_result",
                     "tool_batch_classify_result"):
            self.assertIn(name, trace)

    def test_classification_cancel_makes_no_model_calls(self):
        turn, prompts = self._classify("cancel", lambda call: "[]")
        self.assertEqual(prompts, [])
        self.assertIn("Cancelled before classifying", turn["answer"])
        self.assertEqual(turn["parameters"]["tool"]["choice"], "cancel")
        self.assertFalse(turn["evidence"])

    def test_classification_failed_labels_are_reported(self):
        turn, prompts = self._classify("quick", lambda call: "nonsense")
        tool = turn["parameters"]["tool"]
        self.assertEqual(tool["choice"], "quick")
        self.assertEqual(tool["not_classified"], 2)
        self.assertIn("Quick sample", turn["answer"])
        self.assertIn("- not classified: 2", turn["answer"])

    def test_router_model_is_saved_and_applied(self):
        from qachat.gui.model_settings import (
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

    def test_follow_up_search_keeps_the_earlier_cited_evidence(self):
        from qachat.rag import ConversationTurn, Evidence

        session = self.engine.session
        earlier = (
            Evidence(1, Path("nova.pdf"), 1, 2, None, "We propose NOVA."),
            Evidence(2, Path("other.pdf"), 1, 2, None, "Unrelated."),
        )
        session.turns = [
            ConversationTurn(
                "t", "What is infrared?", "NOVA uses it [1].", earlier, {}
            )
        ]
        fresh = (
            Evidence(1, Path("new.pdf"), 1, 1, None, "Fresh."),
            Evidence(2, Path("nova.pdf"), 1, 2, None, "We propose NOVA."),
        )
        with (
            patch.object(session, "use_chat_model"),
            patch.object(session, "_retrieve", return_value=fresh),
            patch.object(session, "_save_conversation"),
            patch.object(session.chat_session, "reply", return_value="A [1]."),
            patch(
                "qachat.answering.side_reply",
                return_value='{"intent": "question_answering", '
                '"standalone": false, "related_turns": [1]}',
            ),
        ):
            self.engine.ask("why NOV A not NOVA?")
        turn = self.events[-1]["turn"]
        # The cited chunk was already found again, so nothing is duplicated;
        # the uncited earlier chunk is not added.
        self.assertEqual(
            [(e["citation"], str(e["path"])) for e in turn["evidence"]],
            [(1, "new.pdf"), (2, "nova.pdf")],
        )
        self.assertEqual(turn["parameters"]["related_evidence_count"], 0)

        session.turns = [
            ConversationTurn(
                "t", "What is infrared?", "NOVA uses it [1].", earlier, {}
            )
        ]
        only_new = (Evidence(1, Path("new.pdf"), 1, 1, None, "Fresh."),)
        with (
            patch.object(session, "use_chat_model"),
            patch.object(session, "_retrieve", return_value=only_new),
            patch.object(session, "_save_conversation"),
            patch.object(session.chat_session, "reply", return_value="A [2]."),
            patch(
                "qachat.answering.side_reply",
                return_value='{"intent": "question_answering", '
                '"standalone": false, "related_turns": [1]}',
            ),
        ):
            self.engine.ask("why NOV A not NOVA?")
        turn = self.events[-1]["turn"]
        self.assertEqual(
            [(e["citation"], str(e["path"])) for e in turn["evidence"]],
            [(1, "new.pdf"), (2, "nova.pdf")],
        )
        self.assertEqual(turn["parameters"]["related_evidence_count"], 1)
        self.assertEqual(turn["parameters"]["evidence_from_turns"], [1])


if __name__ == "__main__":
    unittest.main()
