"""Test real RAG orchestration with independent conversation memory.

The answering path uses offline embedding and chat doubles and real saved
project conversations. Current source passages retain documentary citations.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin import rag
from kilin.answering import RouteDecision
from kilin.conversation_format import conversation_state
from kilin.conversation_memory.feedback import FeedbackStore
from kilin.conversation_memory.persistence import memory_index_path
from kilin.conversations import ConversationStore
from kilin.project_type import project_type
from tests.kilin._rag_support import (
    _answer_patches, _evidence, _session, _turn,
)
from tests.kilin.test__conversation_memory_index import FakeEmbedding


class TestRAGConversationMemory(unittest.TestCase):
    def setUp(self):
        holder = TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name).resolve()
        (self.root / "indexes").mkdir()
        self.store = ConversationStore(self.root)
        earlier = self.store.create()
        history = _turn(
            "What is our preferred foldon terminology?",
            "We previously preferred foldon modules.",
            parameters={"intent": "writing"},
        )
        path = self.store.state_path(earlier)
        data = json.loads(path.read_text(encoding="utf-8"))
        conversation_state(data)["turns"] = [rag._turn_to_data(history)]
        path.write_text(json.dumps(data), encoding="utf-8")
        current = self.store.create()
        self.session = _session(self.root)
        self.session.conversation_path = self.store.state_path(current)
        self.session.project.profile.type_id = "writing"
        self.embedding = FakeEmbedding()
        self.session.embedding_session.embed_documents.side_effect = (
            self.embedding.embed_documents
        )
        self.session.embedding_session.embed_query.side_effect = (
            self.embedding.embed_query
        )
        self.config = project_type("writing")
        self.intent = self.config.intents["writing"]
        self.earlier_id = earlier.conversation_id
        self.earlier_uuid = history.conversation_turn_uuid

    def test_writing_memory_only_independent_of_sources(self):
        FeedbackStore(self.root).update(
            self.earlier_id, self.earlier_uuid,
            "assistant", saved=True, reaction="liked",
        )
        with patch.object(
            self.session, "_behavior_config", return_value=self.config
        ), _answer_patches(
            self.session, self.intent, RouteDecision("writing", True)
        ):
            self.session.ask("Recall our foldon terminology")
        prompt = self.session.chat_session.reply.call_args.args[0]
        self.assertIn("CONVERSATION MEMORY", prompt)
        self.assertIn("foldon modules", prompt)
        self.assertIn("CURRENT REQUEST", prompt)
        self.assertNotIn("\n\nPROJECT SOURCE EVIDENCE", prompt)
        self.assertNotIn("Evidence:", prompt)
        self.assertEqual(len(self.embedding.queries), 1)
        self.assertEqual(len(self.embedding.embedded), 2)
        self.assertTrue(memory_index_path(self.root).exists())
        system = self.session.chat_session.restore_history.call_args.args[0]
        self.assertIn("PROVENANCE", system[0]["content"])
        stored = self.session.turns[-1]
        references = stored.parameters["memory_references"]
        self.assertEqual(len(references), 2)
        self.assertEqual([item["label"] for item in references],
                         ["M1", "M2"])
        self.assertEqual(
            {item["role"] for item in references},
            {"user", "assistant"},
        )
        self.assertTrue(all(item["turn_uuid"] == self.earlier_uuid
                            for item in references))
        self.assertTrue(all(item["origins"] == [{
            "conversation_id": self.earlier_id, "turn_number": 1,
        }] for item in references))
        self.assertTrue(all(
            set(item) == {"label", "turn_uuid", "role", "origins"}
            for item in references
        ))
        self.assertEqual(stored.evidence, ())

    def test_references_do_not_copy_source_message_text(self):
        """Retain navigational metadata without persisting previews."""

        self.assertFalse(hasattr(rag, "_memory_reference_preview"))
        FeedbackStore(self.root).update(
            self.earlier_id, self.earlier_uuid,
            "assistant", saved=True, reaction="liked",
        )
        with patch.object(
            self.session, "_behavior_config", return_value=self.config
        ), _answer_patches(
            self.session, self.intent, RouteDecision("writing", True)
        ):
            self.session.ask("Recall our foldon terminology")
        for item in self.session.turns[-1].parameters[
            "memory_references"
        ]:
            self.assertNotIn("preview", item)
            self.assertNotIn("summary", item)
            self.assertEqual(item["turn_uuid"], self.earlier_uuid)

    def test_memory_retrieval_timeline_and_trace_provenance(self):
        """Expose counts and timing without recording remembered text."""
        from unittest.mock import MagicMock

        FeedbackStore(self.root).update(
            self.earlier_id, self.earlier_uuid,
            "assistant", saved=True, reaction="liked",
        )
        observed: list[dict[str, object]] = []
        phases: list[tuple[str, dict[str, object]]] = []
        trace = MagicMock()
        trace.include_content = False
        trace.new_id.side_effect = lambda name: name
        self.session.trace = trace
        with patch.object(
            self.session, "_behavior_config", return_value=self.config
        ), _answer_patches(
            self.session, self.intent, RouteDecision("writing", True)
        ):
            self.session.ask(
                "Recall our foldon terminology",
                activity_callback=observed.append,
                phase_callback=lambda phase, details: phases.append(
                    (phase, details)
                ),
            )
        steps = self.session.turns[-1].timeline.steps
        self.assertEqual(
            [(event["phase"], event["label"]) for event in observed],
            [(step.phase, step.label) for step in steps],
        )
        self.assertEqual(
            [event["index"] for event in observed],
            list(range(len(steps))),
        )
        retrieval = next(
            step for step in steps if step.phase == "memory_retrieving"
        )
        self.assertEqual(retrieval.label, "Retrieving conversation memories")
        result = next(
            step for step in steps if step.phase == "memory_ready"
        )
        self.assertEqual(
            result.label, "2 conversation memories retrieved from 1 turn"
        )
        self.assertGreaterEqual(retrieval.ms, 0)
        self.assertEqual(
            sum(step.ms for step in steps),
            self.session.turns[-1].timeline.elapsed_ms,
        )
        details = next(
            data for phase, data in phases if phase == "memory_ready"
        )
        self.assertEqual(details["memory_hit_count"], 2)
        self.assertEqual(details["memory_turn_count"], 1)
        self.assertGreaterEqual(details["duration_ms"], 0)
        recorded = [
            call.kwargs["data"]
            for call in trace.emit.call_args_list
            if call.args[0] == "conversation_memory.retrieved"
        ]
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]["memory_hit_count"], 2)
        self.assertEqual(recorded[0]["memory_turn_count"], 1)
        self.assertGreaterEqual(recorded[0]["duration_ms"], 0)
        selections = recorded[0]["selected_memories"]
        self.assertEqual(len(selections), 2)
        self.assertEqual({item["role"] for item in selections},
                         {"assistant", "user"})
        self.assertTrue(all(item["turn_uuid"] == self.earlier_uuid
                            for item in selections))
        self.assertTrue(all(item["origins"][0]["conversation_id"]
                            == self.earlier_id for item in selections))
        self.assertTrue(all(isinstance(item["score"], float)
                            for item in selections))
        self.assertNotIn("foldon modules", str(recorded[0]))

    def test_off_or_intent_read_false_does_not_embed(self):
        disabled = replace(
            self.config,
            conversation_memory=replace(
                self.config.conversation_memory, enabled=False,
            ),
        )
        with patch.object(
            self.session, "_behavior_config", return_value=disabled
        ), _answer_patches(
            self.session, self.intent, RouteDecision("writing", True)
        ):
            self.session.ask("Recall our foldon terminology")
        self.assertFalse(memory_index_path(self.root).exists())
        self.assertEqual(self.embedding.embedded, [])
        self.assertEqual(self.embedding.queries, [])
        self.assertNotIn(
            "CONVERSATION MEMORY",
            self.session.chat_session.reply.call_args.args[0],
        )
        self.assertNotIn(
            "memory_references", self.session.turns[-1].parameters
        )

    def test_no_memory_hits_still_reports_retrieval_completion(self):
        """Make a zero-hit memory search visible without false references."""
        phases: list[tuple[str, dict[str, object]]] = []
        with patch.object(
            self.session, "_behavior_config", return_value=self.config
        ), patch.object(
            rag, "search_memory", return_value=()
        ), _answer_patches(
            self.session, self.intent, RouteDecision("writing", True)
        ):
            self.session.ask(
                "An unrelated question",
                phase_callback=lambda phase, details: phases.append(
                    (phase, details)
                ),
            )
        ready = [details for phase, details in phases
                 if phase == "memory_ready"]
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0]["memory_hit_count"], 0)
        self.assertEqual(ready[0]["memory_turn_count"], 0)
        self.assertIn("0 conversation memories retrieved", [
            step.label for step in self.session.turns[-1].timeline.steps
        ])
        self.assertNotIn(
            "memory_references", self.session.turns[-1].parameters
        )

    def test_disabled_memory_adds_no_retrieval_timeline_steps(self):
        """Leave disabled projects' timeline and prompt paths unchanged."""
        disabled = replace(
            self.config,
            conversation_memory=replace(
                self.config.conversation_memory, enabled=False,
            ),
        )
        with patch.object(
            self.session, "_behavior_config", return_value=disabled
        ), _answer_patches(
            self.session, self.intent, RouteDecision("writing", True)
        ):
            self.session.ask("Recall our foldon terminology")
        self.assertFalse(any(
            step.phase.startswith("memory_")
            for step in self.session.turns[-1].timeline.steps
        ))

    def test_intent_memory_read_disabled_avoids_embeddings(self):
        config = replace(
            self.config,
            intents={
                **self.config.intents,
                "writing": replace(self.intent, memory_read=False),
            },
        )
        intent = config.intents["writing"]
        with patch.object(
            self.session, "_behavior_config", return_value=config
        ), _answer_patches(
            self.session, intent, RouteDecision("writing", True)
        ):
            self.session.ask("Recall our foldon terminology")
        self.assertEqual(self.embedding.embedded, [])
        self.assertEqual(self.embedding.queries, [])
        self.assertFalse(memory_index_path(self.root).exists())

    def test_memory_qwen_query_differs_from_source_instruction(self):
        self.session.project.profile.query_format = "qwen3"
        with patch.object(
            self.session, "_behavior_config", return_value=self.config
        ):
            memory_query = self.session._format_memory_query("Foldon?")
            source_query = self.session._format_query("Foldon?")
        self.assertIn("earlier researcher questions", memory_query)
        self.assertNotEqual(memory_query, source_query)

    def test_source_and_memory_prompt_citations_are_separate(self):
        source_intent = replace(self.intent, retrieval_mode="search")
        config = replace(
            self.config,
            intents={**self.config.intents, "writing": source_intent},
        )
        with patch.object(
            self.session, "_behavior_config", return_value=config
        ), _answer_patches(
            self.session, source_intent, RouteDecision("writing", True)
        ):
            self.session.ask("Explain foldon terminology")
        prompt = self.session.chat_session.reply.call_args.args[0]
        self.assertIn("PROJECT SOURCE EVIDENCE", prompt)
        self.assertIn("CONVERSATION MEMORY", prompt)
        self.assertIn("Memory M1", prompt)
        self.assertIn("[1]", prompt)
        self.assertNotIn("[M1]", prompt)
        committed = self.session.turns[-1]
        self.assertEqual(len(committed.evidence), 1)
        self.assertEqual(committed.evidence[0], _evidence())
        self.assertEqual(committed.parameters["intent"], "writing")

    def test_reviewing_role_boundary_remains_in_system(self):
        reviewing = project_type("reviewing")
        self.config = reviewing
        self.intent = reviewing.intents["question_answering"]
        self.session.project.profile.type_id = "reviewing"
        with patch.object(
            self.session, "_behavior_config", return_value=reviewing
        ), _answer_patches(
            self.session, self.intent,
            RouteDecision("question_answering", True),
        ):
            self.session.ask("Find factual evidence for foldons")
        system = self.session.chat_session.restore_history.call_args.args[0]
        self.assertIn("researcher", system[0]["content"].lower())
        self.assertIn("NOT a referee", system[0]["content"])
        self.assertNotIn("CONVERSATION MEMORY",
                         self.session.chat_session.reply.call_args.args[0])
