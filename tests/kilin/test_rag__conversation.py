"""Test RAG conversation lifecycle and persistence.

All state and filesystem interactions use isolated fixtures.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch, sentinel
from kilin import rag
from kilin.conversation_format import (
    CONVERSATION_SESSION_KEY,
    new_conversation_session,
)
from kilin.runtime import BACKEND_MLX
from kilin.trace_logging import NullTraceLogger, TraceConfig
from tests.kilin.retrievals._support import chunk

from tests.kilin._rag_support import (
    _evidence,
    _session,
    _turn,
)


class TestRAGLifecycle(unittest.TestCase):
    """Test model ownership, switching, refresh, and token bookkeeping."""

    def test_close_releases_sessions_once(self) -> None:
        """Close model resources exactly once per RAG session."""
        session = _session()
        session.close()
        session.close()
        self.assertTrue(session._closed)
        session.chat_session.close.assert_called_once_with()
        session.embedding_session.close.assert_called_once_with()

    def test_close_releases_distinct_router_and_owned_trace(self) -> None:
        """Release a separate router and owned trace on shutdown."""
        session = _session()
        router = MagicMock()
        session._router_session = router
        session._owns_trace = True
        session.trace = MagicMock()
        session.close()
        self.assertIsNone(session._router_session)
        router.close.assert_called_once()
        session.trace.close.assert_called_once()

    def test_close_marks_closed_even_if_model_cleanup_fails(self) -> None:
        """Mark the session closed despite cleanup exceptions."""
        session = _session()
        session.chat_session.close.side_effect = OSError("close failed")
        session.trace = MagicMock()
        session._owns_trace = True
        with self.assertRaisesRegex(OSError, "close failed"):
            session.close()
        self.assertTrue(session._closed)
        session.trace.close.assert_called_once()
        session.embedding_session.close.assert_called_once()

    def test_refresh_sources_replaces_only_retrieval_state(self) -> None:
        """Refresh indexed sources without reloading chat resources."""
        session = _session()
        previous_chat = session.chat_session
        new_project = SimpleNamespace(root=Path("/new"))
        values = ([chunk("new")], sentinel.indexes, "reused", {"a": 1})
        session._prepare_retrieval = Mock(return_value=values)
        session._prepare_source_cards = Mock(
            return_value=(sentinel.cards, sentinel.card_vectors)
        )
        session.refresh_sources(new_project)
        session._prepare_source_cards.assert_called_once_with(
            values[0], sentinel.indexes, session.embedding_session,
        )
        self.assertIs(session.source_cards, sentinel.cards)
        self.assertIs(session.source_card_vectors, sentinel.card_vectors)
        self.assertIs(session.project, new_project)
        self.assertEqual(session.chunks, values[0])
        self.assertIs(session.indexes, sentinel.indexes)
        self.assertEqual(session.index_status, "reused")
        self.assertEqual(session.source_changes, {"a": 1})
        self.assertIs(session.chat_session, previous_chat)

    def test_card_refresh_uses_existing_indexes(self) -> None:
        """Reconcile cards without rescanning or embedding source chunks."""
        session = _session()
        session._prepare_retrieval = Mock()
        session._prepare_source_cards = Mock(return_value=(
            sentinel.cards, sentinel.vectors,
        ))
        session.refresh_source_cards()
        session._prepare_retrieval.assert_not_called()
        session._prepare_source_cards.assert_called_once_with(
            session.chunks, session.indexes, session.embedding_session,
        )
        self.assertIs(session.source_cards, sentinel.cards)

    def test_card_generation_guard_preserves_history_and_parameters(
        self,
    ) -> None:
        """Unload both interactive models and restore conversation state."""
        session = _session()
        original = session.chat_session
        original.messages = [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "previous question"},
        ]
        original.thinking = True
        original.temperature = 0.2
        original.max_new_tokens = 432
        original.input_token_total = 123
        original.output_token_total = 45
        router = MagicMock()
        session._router_session = router
        replacement = MagicMock()
        session._new_chat_session = Mock(return_value=replacement)
        with session._card_generation_guard():
            original.close.assert_called_once()
            router.close.assert_called_once()
            self.assertIsNone(session.chat_session)
            self.assertIsNone(session._router_session)
        session._new_chat_session.assert_called_once_with(
            original.model_id, thinking=True, temperature=0.2,
            max_new_tokens=432,
        )
        replacement.restore_history.assert_called_once_with(
            original.messages
        )
        self.assertEqual(replacement.input_token_total, 123)
        self.assertEqual(replacement.output_token_total, 45)
        self.assertIs(session.chat_session, replacement)
        self.assertFalse(session._closed)

    def test_card_generation_restores_chat_when_generation_fails(self):
        """Return to the interactive model even after inference raises."""
        session = _session()
        previous = session.chat_session
        replacement = MagicMock()
        session._new_chat_session = Mock(return_value=replacement)
        with self.assertRaisesRegex(ValueError, "model problem"):
            with session._card_generation_guard():
                raise ValueError("model problem")
        previous.close.assert_called_once()
        self.assertIs(session.chat_session, replacement)
        self.assertFalse(session._closed)

    def test_card_release_trace_failure_restores_answering_session(self):
        """Trace failures must not strand a temporarily unloaded model."""
        session = _session()
        previous = session.chat_session
        session.trace = MagicMock()
        session.trace.emit.side_effect = [OSError("bad trace"), None]
        restored = MagicMock()
        session._new_chat_session = Mock(return_value=restored)
        with self.assertRaisesRegex(OSError, "bad trace"):
            with session._card_generation_guard():
                self.fail("Trace failure must precede model generation")
        previous.close.assert_called_once()
        self.assertIs(session.chat_session, restored)
        self.assertFalse(session._closed)

    def test_failed_card_model_restoration_closes_session(self):
        """Prevent use of an interactive model that failed to reload."""
        session = _session()
        session._new_chat_session = Mock(
            side_effect=OSError("model unavailable")
        )
        with self.assertRaisesRegex(RuntimeError, "Reopen"):
            with session._card_generation_guard():
                pass
        self.assertTrue(session._closed)
        session.embedding_session.close.assert_called_once()

    def test_refresh_failure_keeps_old_index_but_new_project(self) -> None:
        """Preserve the existing index after a failed source refresh."""
        session = _session()
        original_index = session.indexes
        project = SimpleNamespace(root=Path("/new"))
        session._prepare_retrieval = Mock(side_effect=OSError("bad scan"))
        with self.assertRaisesRegex(OSError, "bad scan"):
            session.refresh_sources(project)
        self.assertIs(session.project, project)
        self.assertIs(session.indexes, original_index)

    def test_close_attempts_every_resource_after_multiple_failures(
        self,
    ) -> None:
        """Report the first failure after releasing all owned resources."""

        session = _session()
        session._router_session = MagicMock()
        session.chat_session.close.side_effect = OSError("chat failure")
        session._router_session.close.side_effect = ValueError(
            "router failure"
        )
        session.embedding_session.close.side_effect = RuntimeError(
            "embedding failure"
        )
        session.trace = MagicMock()
        session._owns_trace = True
        session.trace.emit.side_effect = OSError("trace event failure")
        session.trace.close.side_effect = OSError("trace close failure")
        owner = MagicMock()
        session._conversation_ownership = owner
        with self.assertRaisesRegex(OSError, "chat failure") as caught:
            session.close()
        self.assertTrue(session._closed)
        self.assertIsNone(session._router_session)
        self.assertEqual(session.chat_session.close.call_count, 1)
        self.assertEqual(session.embedding_session.close.call_count, 1)
        self.assertEqual(session.trace.close.call_count, 1)
        self.assertIsNone(session._conversation_ownership)
        owner.release.assert_called_once_with()
        self.assertIn("router failure", str(caught.exception.__notes__))
        self.assertIn("embedding failure", str(caught.exception.__notes__))
        session.close()
        self.assertEqual(session.chat_session.close.call_count, 1)

    def test_close_releases_direct_ownership_after_trace_failure(
        self,
    ) -> None:
        """Release direct-write ownership even if emitting close fails."""

        session = _session()
        owner = MagicMock()
        session._conversation_ownership = owner
        session.trace = MagicMock()
        session.trace.emit.side_effect = OSError("event write failed")
        with self.assertRaisesRegex(OSError, "event write failed"):
            session.close()
        owner.release.assert_called_once_with()
        session.embedding_session.close.assert_called_once_with()

    def test_switch_chat_model_rejects_closed_or_wrong_backend(self) -> None:
        """Reject unsupported model switches and closed sessions."""
        session = _session()
        session._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.use_chat_model("other")
        session._closed = False
        session.backend = "torch"
        with self.assertRaisesRegex(RuntimeError, "MLX"):
            session.use_chat_model("other")

    def test_switch_chat_model_noop_and_new_session(self) -> None:
        """Reuse the current model or create a replacement as required."""
        session = _session()
        session.router_model = "different-router"
        original = session.chat_session
        session.use_chat_model(original.model_id)
        original.close.assert_not_called()
        replacement = MagicMock()
        replacement.model_id = "other"
        session._new_chat_session = Mock(return_value=replacement)
        session.use_chat_model("other")
        original.close.assert_called_once()
        self.assertIs(session.chat_session, replacement)
        self.assertIsNone(session._router_session)
        self.assertEqual(
            replacement.restore_history.call_args.args[0],
            [{"role": "system", "content": "system"}],
        )
        self.assertEqual(
            session._new_chat_session.call_args.kwargs,
            {"thinking": False, "temperature": 0.2,
             "max_new_tokens": 128},
        )

    def test_switch_chat_model_reuses_loaded_router(self) -> None:
        """Reuse a loaded router when it already matches the chat model."""
        session = _session()
        session.router_model = "different-router"
        original = session.chat_session
        router = MagicMock()
        router.model_id = "router"
        session._router_session = router
        session.use_chat_model("router")
        self.assertIs(session.chat_session, router)
        self.assertIsNone(session._router_session)
        router.configure_generation.assert_called_once()
        original.close.assert_called_once()

    def test_switch_chat_model_preserves_old_router_when_shared(self) -> None:
        """Avoid closing a router still shared with another session."""
        session = _session()
        original = session.chat_session
        session.router_model = original.model_id
        replacement = MagicMock()
        session._new_chat_session = Mock(return_value=replacement)
        session.use_chat_model("other")
        original.close.assert_not_called()
        self.assertIs(session._router_session, original)
        self.assertIs(session.chat_session, replacement)

    def test_switch_chat_model_rolls_back_failed_new_load(self) -> None:
        """Keep the previous chat model if loading its replacement fails."""
        session = _session()
        session.router_model = "different-router"
        original = session.chat_session
        restored = MagicMock()
        session._new_chat_session = Mock(
            side_effect=[OSError("cannot load"), restored]
        )
        with self.assertRaisesRegex(OSError, "cannot load"):
            session.use_chat_model("missing")
        original.close.assert_called_once()
        self.assertIs(session.chat_session, restored)
        restored.restore_history.assert_called_once_with(
            [{"role": "system", "content": "system"}]
        )

    def test_switch_chat_model_keeps_shared_router_on_failure(self) -> None:
        """Preserve shared router ownership after failed model loading."""
        session = _session()
        original = session.chat_session
        session.router_model = original.model_id
        session._new_chat_session = Mock(side_effect=OSError("failed"))
        with self.assertRaises(OSError):
            session.use_chat_model("other")
        self.assertIs(session.chat_session, original)
        original.close.assert_not_called()

    def test_router_model_change_closes_only_separate_session(self) -> None:
        """Close the old router only when it owns a separate session."""
        session = _session()
        router = MagicMock()
        session._router_session = router
        session.use_router_model("other")
        router.close.assert_called_once()
        self.assertIsNone(session._router_session)
        self.assertEqual(session.router_model, "other")
        session.use_router_model("other")
        router.close.assert_called_once()
        session._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.use_router_model("again")

    def test_router_session_is_shared_or_loaded_lazily(self) -> None:
        """Share an existing chat model or load a router on demand."""
        session = _session()
        self.assertIs(session._router_chat(), session.chat_session)
        session.router_model = "router"
        router = MagicMock()
        session._new_chat_session = Mock(return_value=router)
        self.assertIs(session._router_chat(), router)
        self.assertIs(session._router_chat(), router)
        session._new_chat_session.assert_called_once_with("router")

    def test_router_reply_delegates_to_side_reply(self) -> None:
        """Use the router session for classification-only replies."""
        session = _session()
        with patch.object(rag, "side_reply", return_value="yes") as side:
            self.assertEqual(session.router_reply("system", "question",
                                                  max_new_tokens=12), "yes")
            self.assertEqual(session._ask_router("s", "q", 9), "yes")
        self.assertEqual(side.call_args_list[0].kwargs,
                         {"max_new_tokens": 12, "language_context": ()})
        self.assertEqual(side.call_args_list[1].kwargs,
                         {"max_new_tokens": 9, "language_context": ()})

    def test_new_chat_session_uses_project_backend_and_trace(self) -> None:
        """Construct new chat models with the project runtime and trace."""
        session = _session()
        with patch.object(rag, "ChatSession") as constructor:
            self.assertIs(
                session._new_chat_session("model", temperature=0.6),
                constructor.return_value,
            )
        self.assertEqual(constructor.call_args.args, ("model",))
        self.assertEqual(constructor.call_args.kwargs["backend"],
                         BACKEND_MLX)
        self.assertEqual(constructor.call_args.kwargs["temperature"], 0.6)
        self.assertFalse(constructor.call_args.kwargs["allow_download"])
        self.assertTrue(
            constructor.call_args.kwargs["block_unrequested_cjk"]
        )

    def test_usage_counters_include_distinct_router_only(self) -> None:
        """Avoid double-counting token usage for a shared router."""
        session = _session()
        session._router_session = session.chat_session
        self.assertEqual(len(session.usage_mark()), 1)
        router = MagicMock()
        router.input_token_total = 3
        router.output_token_total = 2
        session._router_session = router
        mark = session.usage_mark()
        self.assertEqual(len(mark), 2)
        session.chat_session.input_token_total = 15
        session.chat_session.output_token_total = 9
        router.input_token_total = 8
        router.output_token_total = 4
        self.assertEqual(
            session.usage_since(mark),
            {"input_tokens": 10, "output_tokens": 6},
        )
        self.assertEqual(
            session.usage_since({}),
            {"input_tokens": 23, "output_tokens": 13},
        )
        self.assertEqual(
            session.usage_since({id(session.chat_session): (100, 100)}),
            {"input_tokens": 8, "output_tokens": 4},
        )

    def test_model_sessions_ignores_missing_counters(self) -> None:
        """Tolerate model sessions without optional usage counters."""
        session = _session()
        session._router_session = object()
        self.assertEqual(session._model_sessions(), [session.chat_session])

    def test_index_metadata_and_turn_parameters(self) -> None:
        """Expose index provenance and effective turn parameters."""
        session = _session()
        metadata = session._index_metadata()
        self.assertEqual(metadata["embedding_model"], "embedding-model")
        self.assertEqual(metadata["extensions"], [".py"])
        self.assertNotIn("pdf_text_extractor", metadata)
        self.assertNotIn("format_version", metadata)
        parameters = session._parameters(8)
        self.assertEqual(parameters["retrieval_count"], 8)
        self.assertEqual(parameters["chat_model"], "chat-model")
        session.chat_session.model_id = "switched-27b"
        self.assertEqual(
            session._parameters()["chat_model"], "switched-27b"
        )
        self.assertEqual(
            session.project.profile.chat_model("mlx"), "chat-model"
        )
        self.assertEqual(parameters["temperature"], 0.2)
        self.assertEqual(session._parameters()["retrieval_count"], 5)
        session.project.profile.extensions = (".pdf",)
        self.assertEqual(
            session._index_metadata()["pdf_text_extractor"],
            rag.PDF_TEXT_EXTRACTOR,
        )
        with patch.object(rag, "project_type") as lookup:
            lookup.return_value.code_graph = False
            self.assertFalse(session._index_metadata()["code_graph"])

    def test_html_is_read_as_code_only_when_code_graph_is_true(self) -> None:
        """Record how HTML is read, so changing it indexes HTML again."""
        session = _session()
        self.assertNotIn("html_as_code", session._index_metadata())
        session.project.profile.extensions = (".html",)
        for setting, as_code in (
            (True, True),
            ("auto", False),
            (False, False),
        ):
            with self.subTest(code_graph=setting), patch.object(
                rag, "project_type"
            ) as lookup:
                lookup.return_value.code_graph = setting
                self.assertIs(session._reads_as_code(), as_code)
                self.assertIs(
                    session._index_metadata()["html_as_code"], as_code
                )


    def test_json_source_format_reindexes_only_json_profiles(self) -> None:
        """Distinguish legacy JSON chunks without changing other indexes."""

        session = _session()
        self.assertNotIn("json_source_units", session._index_metadata())
        session.project.profile.extensions = (".json",)
        metadata = session._index_metadata()
        self.assertEqual(
            metadata["json_source_units"], rag.JSON_SOURCE_UNITS_FORMAT
        )
        # Older indexes have the same configured options, but no version.
        legacy = dict(metadata)
        legacy.pop("json_source_units")
        self.assertNotEqual(legacy, metadata)
        # The version is stable for repeated reads of the same profile.
        self.assertEqual(session._index_metadata(), metadata)
        session.project.profile.extensions = (".py", ".md")
        self.assertNotIn("json_source_units", session._index_metadata())


class TestRAGConversationPersistence(unittest.TestCase):
    """Verify resumable turns, checkpoint identities, and path containment."""

    def setUp(self) -> None:
        """Create isolated state and arrange test-specific fixtures."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.session = _session(self.root)
        self.session.conversation_path.parent.mkdir(parents=True)
        self.addCleanup(self._release_conversation)

    def _release_conversation(self) -> None:
        """Release a conversation lock a switch acquired, before cleanup.

        Windows cannot delete the temporary folder while a lock file is
        open.

        """

        ownership = getattr(self.session, "_conversation_ownership", None)
        if ownership is not None:
            ownership.release()

    def test_absent_conversation_has_empty_state(self) -> None:
        """Treat a missing conversation file as a new conversation."""
        self.assertEqual(self.session._load_conversation(), ([], [], None))

    def test_save_and_reload_grouped_conversation_without_trace(self) -> None:
        """Round-trip saved turns and checkpoints without active tracing."""
        session = self.session
        session.turns = [
            _turn("Question", "Answer [1]", (_evidence(),))
        ]
        session._save_conversation(trace_turn_id="turn-1")
        first = session._checkpoint_id
        self.assertTrue(first.startswith("checkpoint_"))
        payload = json.loads(
            session.conversation_path.read_text(encoding="utf-8")
        )
        saved = rag.conversation_state(payload)
        self.assertEqual(saved["checkpoint_id"], first)
        self.assertEqual(saved["checkpoint"]["turn_count"], 1)
        self.assertIsNone(saved["checkpoint"]["parent_checkpoint_id"])
        messages, turns, checkpoint = session._load_conversation()
        self.assertEqual(turns, session.turns)
        self.assertEqual(checkpoint, first)
        self.assertEqual([message["role"] for message in messages],
                         ["system", "user", "assistant"])
        session.turns.append(_turn("Second", "Answer"))
        session._save_conversation()
        payload = json.loads(
            session.conversation_path.read_text(encoding="utf-8")
        )
        saved = rag.conversation_state(payload)
        self.assertEqual(saved["checkpoint"]["parent_checkpoint_id"],
                         first)
        self.assertNotEqual(saved["checkpoint_id"], first)

    def test_save_conversation_with_trace_uses_same_record(self) -> None:
        """Write resumable state into the enabled trace conversation file."""
        session = self.session
        session.trace = MagicMock()
        session.trace.enabled = True
        session.turns = [_turn(evidence=(_evidence(),))]
        session._save_conversation()
        session.trace.update_conversation.assert_called_once()
        state = session.trace.update_conversation.call_args.args[0]
        self.assertEqual(state["checkpoint_id"], session._checkpoint_id)
        self.assertEqual(state["turns"][0]["evidence"][0]["path"],
                         "src/a.py")
        self.assertFalse(session.conversation_path.exists())

    def test_failed_untraced_checkpoint_preserves_previous_turns(
        self,
    ) -> None:
        """Keep existing disk and memory state when direct JSON write fails."""
        session = self.session
        session.turns = [_turn("Saved question", "Saved answer")]
        session._save_conversation()
        first = session._checkpoint_id
        before = session.conversation_path.read_bytes()
        pending = _turn("Pending question", "Pending answer")
        with patch.object(
            rag, "_write_json", side_effect=OSError("out of space")
        ):
            with self.assertRaisesRegex(OSError, "out of space"):
                session._save_conversation(pending_turn=pending)
        self.assertEqual(len(session.turns), 1)
        self.assertEqual(session._checkpoint_id, first)
        self.assertEqual(session.conversation_path.read_bytes(), before)
        session._save_conversation(pending_turn=pending)
        state = rag.conversation_state(
            json.loads(session.conversation_path.read_text(encoding="utf-8"))
        )
        self.assertEqual(len(session.turns), 2)
        self.assertEqual(state["checkpoint"]["parent_checkpoint_id"], first)
        self.assertEqual(state["turns"][1]["question"], "Pending question")

    def test_load_canonical_turns_reconstructs_model_history(self) -> None:
        """Restore chat messages and turns from canonical saved records."""
        session = self.session
        data = {"sessions": {CONVERSATION_SESSION_KEY:
                             new_conversation_session({
            "system_prompt": "Saved system prompt",
            "turns": [rag._turn_to_data(_turn())],
            "checkpoint_id": "checkpoint_previous",
        })}}
        session.conversation_path.write_text(
            json.dumps(data), encoding="utf-8"
        )
        messages, turns, checkpoint = session._load_conversation()
        self.assertEqual(messages[0]["content"], "Saved system prompt")
        self.assertEqual([message["role"] for message in messages],
                         ["system", "user", "assistant"])
        self.assertEqual(len(turns), 1)
        self.assertEqual(checkpoint, "checkpoint_previous")

    def test_turn_uuid_survives_serialization_and_reload(self) -> None:
        """Persist a stable UUID independent of duplicate answer content."""

        first = _turn("Same question", "Same answer")
        second = _turn("Same question", "Same answer")
        self.assertNotEqual(first.conversation_turn_uuid,
                            second.conversation_turn_uuid)
        serialized = rag._turn_to_data(first)
        self.assertEqual(serialized["conversation_turn_uuid"],
                         first.conversation_turn_uuid)
        self.assertEqual(rag._turn_from_data(serialized), first)
        self.session._save_conversation(pending_turn=first)
        messages, turns, _ = self.session._load_conversation()
        self.assertEqual(len(messages), 3)
        self.assertEqual(turns[0].conversation_turn_uuid,
                         first.conversation_turn_uuid)

    def test_missing_or_duplicate_uuids_rejected_without_modifying_file(
        self,
    ) -> None:
        """Unsupported prerelease records are never assigned hidden IDs."""

        current = rag._turn_to_data(_turn())
        invalid = [
            [{key: value for key, value in current.items()
              if key != "conversation_turn_uuid"}],
            [current, current],
            [{**current, "conversation_turn_uuid": "bad"}],
        ]
        for items in invalid:
            with self.subTest(items=items):
                payload = {"sessions": {CONVERSATION_SESSION_KEY:
                           new_conversation_session({"turns": items})}}
                self.session.conversation_path.write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                before = self.session.conversation_path.read_bytes()
                with self.assertRaises(ValueError):
                    self.session._load_conversation()
                self.assertEqual(
                    self.session.conversation_path.read_bytes(), before
                )

    def test_load_conversation_rejects_corrupt_records(self) -> None:
        """Reject corrupt and obsolete conversation representations."""
        session = self.session
        session.conversation_path.write_text("not JSON", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid"):
            session._load_conversation()
        corrupt = {"sessions": {CONVERSATION_SESSION_KEY:
                                 new_conversation_session({
            "turns": [{"timestamp": "x", "evidence": "bad"}],
        })}}
        session.conversation_path.write_text(
            json.dumps(corrupt), encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            session._load_conversation()
        obsolete = {"sessions": {CONVERSATION_SESSION_KEY:
                                  new_conversation_session({
            "turns": [], "chat_messages": [{"role": "user"}],
        })}}
        session.conversation_path.write_text(
            json.dumps(obsolete), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "invalid"):
            session._load_conversation()
        for record in ({"conversation": {"turns": []}}, [], None):
            session.conversation_path.write_text(
                json.dumps(record), encoding="utf-8"
            )
            with self.subTest(record=record):
                with self.assertRaisesRegex(ValueError, "invalid"):
                    session._load_conversation()

    def test_load_conversation_ignores_invalid_checkpoint(self) -> None:
        """Ignore invalid checkpoint identifiers while restoring turns."""
        session = self.session
        session.conversation_path.write_text(json.dumps({
            "sessions": {CONVERSATION_SESSION_KEY:
                         new_conversation_session({
                "turns": [], "checkpoint_id": 100,
            })},
        }), encoding="utf-8")
        self.assertIsNone(session._load_conversation()[2])

    def test_save_preserves_existing_unrelated_sessions(self) -> None:
        """Keep unrelated trace sessions when saving without tracing."""
        session = self.session
        session.conversation_path.write_text(json.dumps({
            "sessions": {"custom": {"title": "Keep"}},
        }), encoding="utf-8")
        session._save_conversation()
        payload = json.loads(
            session.conversation_path.read_text(encoding="utf-8")
        )
        self.assertEqual(payload["sessions"]["custom"],
                         {"title": "Keep"})

    def test_untraced_save_does_not_convert_obsolete_state(self) -> None:
        """Refuse to overwrite obsolete conversation data implicitly."""
        session = self.session
        session.conversation_path.write_text(json.dumps({
            "conversation": {"turns": [{"question": "Do not replace"}]},
        }), encoding="utf-8")
        before = session.conversation_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "invalid"):
            session._save_conversation()
        self.assertEqual(session.conversation_path.read_bytes(), before)

    def test_untraced_save_preserves_invalid_grouped_state(self) -> None:
        """Leave invalid grouped records unchanged after save failure."""
        session = self.session
        bad_group = new_conversation_session({
            "turns": [], "checkpoint_id": "old",
        })
        bad_group["conversation"] = {"turns": [{"question": "Old"}]}
        session.conversation_path.write_text(json.dumps({
            "sessions": {CONVERSATION_SESSION_KEY: bad_group},
        }), encoding="utf-8")
        before = session.conversation_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "invalid"):
            session._save_conversation()
        self.assertEqual(session.conversation_path.read_bytes(), before)

    def test_write_json_is_atomic_and_replaces_existing_data(self) -> None:
        """Replace JSON atomically without leftover temporary files."""
        path = self.root / "payload.json"
        rag._write_json(path, {"a": 1})
        rag._write_json(path, {"a": 2})
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8")), {"a": 2}
        )
        self.assertFalse((self.root / "payload.json.tmp").exists())

    def test_resolve_conversation_path_validation(self) -> None:
        """Enforce project containment and supported conversation paths."""
        session = self.session
        expected = session._resolve_conversation_path(
            "logs/traces/conversation_custom.json"
        )
        self.assertEqual(expected, (
            self.root / "logs/traces/conversation_custom.json"
        ).resolve())
        self.assertEqual(
            session._resolve_conversation_path(expected), expected
        )
        for name in (
            "other/conversation_bad.json",
            "logs/traces/not_conversation.json",
            "logs/traces/conversation_bad.txt",
            "../conversation_escape.json",
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    session._resolve_conversation_path(name)

    def test_resolve_conversation_allocates_unique_default_names(self) -> None:
        """Allocate distinct conversation names when none are specified."""
        session = self.session
        first = session._resolve_conversation_path(None)
        first.write_text("{}", encoding="utf-8")
        second = session._resolve_conversation_path(None)
        self.assertNotEqual(first, second)
        self.assertTrue(second.name.startswith("conversation_"))
        self.assertTrue(second.parent.is_dir())

    def test_resolve_rejects_symlink_escape(self) -> None:
        """Reject conversation paths escaping through symbolic links."""
        session = self.session
        outside = self.root / "elsewhere"
        outside.mkdir()
        link = self.root / "logs" / "outside"
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside project logs"):
            session._resolve_conversation_path(
                "logs/outside/conversation_escape.json"
            )

    def test_archive_conversation_moves_existing_file(self) -> None:
        """Archive an existing conversation without destroying its file."""
        session = self.session
        self.assertIsNone(session._archive_conversation())
        session.conversation_path.write_text("original", encoding="utf-8")
        archive = session._archive_conversation()
        self.assertTrue(archive.is_file())
        self.assertEqual(archive.read_text(encoding="utf-8"), "original")
        self.assertFalse(session.conversation_path.exists())
        self.assertTrue(archive.name.startswith("conversation_"))

    def test_reset_archives_turns_and_preserves_models(self) -> None:
        """Reset conversation state while retaining loaded model sessions."""
        session = self.session
        session.trace = MagicMock()
        session.turns = [_turn()]
        session._checkpoint_id = "old-checkpoint"
        session.conversation_path.write_text("old", encoding="utf-8")
        previous_chat = session.chat_session
        archived = session.reset()
        self.assertTrue(archived.exists())
        self.assertEqual(session.turns, [])
        self.assertIsNone(session._checkpoint_id)
        self.assertIs(session.chat_session, previous_chat)
        session.chat_session.reset.assert_called_once()
        session.trace.reset_conversation_state.assert_called_once()
        session._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.reset()

    def test_switch_conversation_reuses_models_and_restores_settings(
        self,
    ) -> None:
        """Restore another conversation without reloading model resources."""
        session = self.session
        target = session._resolve_conversation_path(
            "logs/traces/conversation_second.json"
        )
        earlier = _turn(parameters={
            "temperature": 0.7, "thinking": True,
            "max_new_tokens": 64,
        })
        target.write_text(json.dumps({
            "sessions": {CONVERSATION_SESSION_KEY:
                         new_conversation_session({
                "turns": [rag._turn_to_data(earlier)],
                "checkpoint_id": "restored",
            })},
        }), encoding="utf-8")
        old_chat = session.chat_session
        turns = session.switch_conversation(target, trace=NullTraceLogger())
        self.assertEqual(turns, [earlier])
        self.assertIs(session.chat_session, old_chat)
        self.assertEqual(session._checkpoint_id, "restored")
        self.assertEqual(session.conversation_path, target)
        self.assertEqual(
            session.chat_session.configure_generation.call_args.args[0],
            {"temperature": 0.7, "thinking": True,
             "max_new_tokens": 64},
        )
        self.assertEqual(session.chat_session.trace, session.trace)
        self.assertEqual(session.embedding_session.trace, session.trace)

    def test_switch_conversation_uses_profile_defaults_without_turns(
        self,
    ) -> None:
        """Use profile generation defaults for a conversation with no turns."""
        session = self.session
        session._owns_trace = True
        old_trace = MagicMock()
        old_trace.enabled = False
        session.trace = old_trace
        target = session._resolve_conversation_path(
            "logs/traces/conversation_empty.json"
        )
        session.switch_conversation(target, trace=NullTraceLogger())
        old_trace.close.assert_called_once()
        self.assertEqual(
            session.chat_session.configure_generation.call_args.args[0],
            {"temperature": 0.2, "thinking": False,
             "max_new_tokens": 128},
        )

    def test_switch_rejects_closed_session_and_mismatched_trace(self) -> None:
        """Reject switching after close or with an unrelated trace."""
        session = self.session
        session._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.switch_conversation(
                "logs/traces/conversation_second.json",
                trace=NullTraceLogger(),
            )
        session._closed = False
        bad = MagicMock()
        bad.enabled = True
        bad.output_path = self.root / "logs/traces/conversation_wrong.json"
        with self.assertRaisesRegex(ValueError, "conversation path"):
            session.switch_conversation(
                "logs/traces/conversation_second.json", trace=bad
            )

    def test_trace_configuration_is_bound_to_conversation_path(self) -> None:
        """Bind newly created RAG traces to the conversation JSON path."""
        path = self.session.conversation_path.resolve()
        disabled, owned = rag._conversation_trace(
            TraceConfig(enabled=False), path
        )
        self.assertFalse(disabled.enabled)
        self.assertFalse(owned)
        enabled, owned = rag._conversation_trace(None, path)
        self.assertTrue(enabled.enabled)
        self.assertTrue(owned)
        self.assertEqual(enabled.output_path.resolve(), path)
        self.assertTrue(enabled.include_content)
        self.assertTrue(enabled.include_token_ids)
        enabled.close()
        configured, owned = rag._conversation_trace(
            TraceConfig(enabled=True),
            path.with_name("conversation_configured.json"),
        )
        self.assertTrue(owned)
        self.assertTrue(configured.include_content)
        self.assertTrue(configured.include_token_ids)
        configured.close()
        with self.assertRaisesRegex(ValueError, "conversation path"):
            rag._conversation_trace(
                TraceConfig(enabled=True, output_path=self.root / "other"),
                path,
            )

    def test_trace_sink_must_match_exact_path_when_enabled(self) -> None:
        """Reject enabled trace sinks pointing outside the conversation."""
        path = self.session.conversation_path.resolve()
        disabled = NullTraceLogger()
        rag._check_conversation_trace(disabled, path)
        bad = MagicMock()
        bad.enabled = True
        bad.output_path = None
        with self.assertRaisesRegex(ValueError, "exact"):
            rag._check_conversation_trace(bad, path)
        bad.output_path = self.root / "other.json"
        with self.assertRaisesRegex(ValueError, "conversation path"):
            rag._check_conversation_trace(bad, path)
        bad.output_path = path
        rag._check_conversation_trace(bad, path)

    def test_source_text_cache_discards_invalid_entries(self) -> None:
        """Ignore malformed cached source text and retain valid records."""
        indexes = self.root / "indexes"
        self.assertEqual(rag._read_text_cache(indexes), {})
        rag._write_text_cache(indexes, {
            "good.py": {"sha256": "hash", "text": "source"},
        })
        self.assertEqual(rag._read_text_cache(indexes)["good.py"]["text"],
                         "source")
        cache = indexes / rag._TEXT_CACHE
        cache.write_text(json.dumps({
            "good": {"sha256": "hash", "text": "source"},
            "bad": {"sha256": 1, "text": "source"},
            "missing": {"sha256": "hash"},
            "null": None,
        }), encoding="utf-8")
        self.assertEqual(list(rag._read_text_cache(indexes)), ["good"])
        cache.write_text("not JSON", encoding="utf-8")
        self.assertEqual(rag._read_text_cache(indexes), {})
        cache.write_text("[]", encoding="utf-8")
        self.assertEqual(rag._read_text_cache(indexes), {})


if __name__ == "__main__":
    unittest.main()
