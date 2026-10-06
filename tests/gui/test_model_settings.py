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

    def test_standalone_question_gets_no_history(self):
        searches, seen, parameters = self._follow_up(
            '{"intent": "compare_sources", "standalone": true, '
            '"related_turns": []}'
        )
        self.assertEqual(parameters["history_turn_ids"], [])
        self.assertEqual(len(seen["history"]), 1)
        self.assertIn("Compare the sources", seen["history"][0]["content"])
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

    def test_document_intent_reads_the_whole_file(self):
        from qachat.chunking import Chunk
        from qachat.retrievals import DocumentSelection

        session = self.engine.session
        session.turns = []
        kind = session.project.profile.kind
        intent = "file_reading" if kind == "code" else "document_reading"
        parts = tuple(
            Chunk(f"p:{n}", Path("paper.pdf"), f"Part {n}.", n, n, "text",
                  None, None)
            for n in (1, 2, 3)
        )
        selection = DocumentSelection(
            path=Path("paper.pdf"),
            functions=(),
            reason="search",
            chunks=parts,
            total_chunks=3,
            estimated_tokens=6,
            complete=True,
        )
        with (
            patch.object(session, "_retrieve") as retrieve,
            patch.object(
                session, "_read_document", return_value=selection
            ) as read,
            patch.object(session, "_save_conversation"),
            patch.object(session.chat_session, "reply", return_value="M [2]."),
            patch(
                "qachat.answering.side_reply",
                return_value=f'{{"intent": "{intent}", "standalone": true}}',
            ),
        ):
            self.engine.ask("check the paper entirely for its language")
        retrieve.assert_not_called()
        self.assertEqual(read.call_args.args[1], 30_000)
        turn = self.events[-1]["turn"]
        self.assertEqual(
            [(e["citation"], e["text"]) for e in turn["evidence"]],
            [(1, "Part 1."), (2, "Part 2."), (3, "Part 3.")],
        )
        document = turn["parameters"]["document"]
        self.assertEqual(document["path"], "paper.pdf")
        self.assertTrue(document["complete"])
        ready = [
            e for e in self.events if e.get("phase") == "evidence_ready"
        ]
        self.assertEqual(
            ready[-1]["message"], "Reading all 3 parts of paper.pdf"
        )

    def test_mention_listing_scans_without_the_answering_model(self):
        from qachat.chunking import Chunk

        session = self.engine.session
        session.turns = []
        lines = "BRAPH 2 is used.\nWe extend BRAPH\n2 here.\n"
        session.chunks = [
            Chunk("c1", Path("notes.md"), lines, 1, 3, "text", None, None),
            Chunk("c2", Path("other.md"), "No match.\n", 1, 1, "text",
                  None, None),
        ]
        replies = iter(
            [
                '{"intent": "mention_listing", "standalone": true}',
                '{"terms": ["BRAPH 2"]}',
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
            self.engine.ask("list all mentions of BRAPH 2")
        retrieve.assert_not_called()
        reply.assert_not_called()
        turn = self.events[-1]["turn"]
        self.assertIn("Found **2 mentions** of “BRAPH 2”", turn["answer"])
        self.assertEqual(
            [(e["citation"], e["start_line"]) for e in turn["evidence"]],
            [(1, 1), (2, 1)],
        )
        workflow = turn["parameters"]["workflow"]
        self.assertEqual(workflow["terms"], ["BRAPH 2"])
        self.assertEqual(workflow["terms_from"], "router")
        self.assertEqual(workflow["total_mentions"], 2)
        self.assertEqual(workflow["files_scanned"], 2)
        self.assertIn("workflow_scan_and_list_result", self.trace_text())

        # A term that names a file in the question becomes the scope.
        replies = iter(
            [
                '{"intent": "mention_listing", "standalone": true}',
                '{"terms": ["BRAPH 2", "other"], "document": null}',
            ]
        )
        session.chunks.append(
            Chunk("c3", Path("other.md"), "BRAPH 2 once.\n", 2, 2, "text",
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
            self.engine.ask("where is BRAPH 2 mentioned in other")
        reply.assert_not_called()
        workflow = self.events[-1]["turn"]["parameters"]["workflow"]
        self.assertEqual(workflow["terms"], ["BRAPH 2"])
        self.assertEqual(workflow["terms_as_scope"], ["other"])
        self.assertEqual(workflow["scope"], ["other.md"])
        self.assertEqual(workflow["total_mentions"], 1)

    def _classify(self, choice, label_reply):
        """Ask a classification question; the user answers the plan."""

        from qachat.chunking import Chunk

        session = self.engine.session
        session.turns = []
        session.chunks = [
            Chunk("c1", Path("a.md"), "Our Python code.\nPython [3].\n",
                  1, 2, "text", None, None),
        ]
        commands = [{"action": "preview"}, {"action": "workflow_choice",
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
            patch("qachat.rag.needs_confirmation", return_value=True),
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
        workflow = turn["parameters"]["workflow"]
        self.assertEqual(workflow["choice"], "continue")
        self.assertEqual(workflow["counts"],
                         {"own code": 1, "citation": 1, "unclear": 0})
        self.assertEqual(workflow["not_classified"], 0)
        self.assertEqual(workflow["model_calls"], 2)
        self.assertIn("Classified **2 of 2 mentions**", turn["answer"])
        self.assertEqual(len(turn["evidence"]), 2)
        trace = self.trace_text()
        for name in ("workflow_plan_proposed", "workflow_plan_decided",
                     "workflow_batch_result",
                     "workflow_batch_classify_result"):
            self.assertIn(name, trace)

    def test_classification_cancel_makes_no_model_calls(self):
        turn, prompts = self._classify("cancel", lambda call: "[]")
        self.assertEqual(prompts, [])
        self.assertIn("Cancelled before classifying", turn["answer"])
        self.assertEqual(turn["parameters"]["workflow"]["choice"], "cancel")
        self.assertFalse(turn["evidence"])

    def test_classification_failed_labels_are_reported(self):
        turn, prompts = self._classify("quick", lambda call: "nonsense")
        workflow = turn["parameters"]["workflow"]
        self.assertEqual(workflow["choice"], "quick")
        self.assertEqual(workflow["not_classified"], 2)
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
        # Restoring the default leaves no saved entry behind.
        self.request["router_model"] = default_router_model(self.backend)
        self.engine.apply_model_settings(self.request)
        self.assertFalse((root / "effort_levels.json").exists())
        self.assertEqual(
            self.engine.session.router_model,
            default_router_model(self.backend),
        )

    def test_follow_up_search_keeps_the_earlier_cited_evidence(self):
        from qachat.rag import ConversationTurn, Evidence

        session = self.engine.session
        earlier = (
            Evidence(1, Path("diva.pdf"), 1, 2, None, "We propose DIVA."),
            Evidence(2, Path("other.pdf"), 1, 2, None, "Unrelated."),
        )
        session.turns = [
            ConversationTurn(
                "t", "What is Raman?", "DIVA uses it [1].", earlier, {}
            )
        ]
        fresh = (
            Evidence(1, Path("new.pdf"), 1, 1, None, "Fresh."),
            Evidence(2, Path("diva.pdf"), 1, 2, None, "We propose DIVA."),
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
            self.engine.ask("why DIV A not DIVA?")
        turn = self.events[-1]["turn"]
        # The cited chunk was already found again, so nothing is duplicated;
        # the uncited earlier chunk is not added.
        self.assertEqual(
            [(e["citation"], str(e["path"])) for e in turn["evidence"]],
            [(1, "new.pdf"), (2, "diva.pdf")],
        )
        self.assertEqual(turn["parameters"]["related_evidence_count"], 0)

        session.turns = [
            ConversationTurn(
                "t", "What is Raman?", "DIVA uses it [1].", earlier, {}
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
            self.engine.ask("why DIV A not DIVA?")
        turn = self.events[-1]["turn"]
        self.assertEqual(
            [(e["citation"], str(e["path"])) for e in turn["evidence"]],
            [(1, "new.pdf"), (2, "diva.pdf")],
        )
        self.assertEqual(turn["parameters"]["related_evidence_count"], 1)
        self.assertEqual(turn["parameters"]["evidence_from_turns"], [1])


if __name__ == "__main__":
    unittest.main()
