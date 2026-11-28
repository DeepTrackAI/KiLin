"""Exercise a chronologically recorded offline project through ChatEngine.

Use real project files, the real RAG turn/checkpoint and switch methods,
ordinary tool callbacks, and deterministic model doubles. No Qt or model
weights are required.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from collections import deque
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from kilin import rag
from kilin._trace_project import OPERATION_RECORDING_VERSION
from kilin.answering import RouteDecision
from kilin.conversation_format import conversation_state
from kilin.conversation_memory.feedback import FeedbackStore
from kilin.gui import engine as gui_engine
from kilin.gui.engine import ChatEngine
from kilin.project import open_project
from kilin.project_type import RETRIEVAL_MODE_NONE
from kilin.tools import ToolResult, ToolUse
from kilin.tools._shared.planning import (
    CONFIRMATION_ALWAYS,
    ToolPlan,
    ask_choice,
    confirm_plan,
)
from tests.kilin._rag_support import _answer_patches, _intent, _session


def _nodes(path):
    """Inspect only diagnostic hierarchy, never details or saved state."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    for session in payload["sessions"].values():
        yield from session.get("operations", {}).values()
        for turn in session.get("turns", {}).values():
            for stage in turn["stages"].values():
                yield from stage["operations"].values()
        for progress in session.get("progress_checkpoints", {}).values():
            yield {"event": progress}


class TestEngineRecording(unittest.TestCase):
    """Verify replay inputs and outcomes without a replay implementation."""

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workspace = self.root / "recorded"
        self.events = []
        self.commands = deque()
        self.engine = ChatEngine(
            self.root, self.events.append, trace_enabled=True,
            read_command=self.commands.popleft,
        )
        patcher = patch("kilin.gui.engine.RAGSession", self._activate)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.engine.close)
        source = self.root / "source.txt"
        source.write_text("Original evidence.\n", encoding="utf-8")
        self.engine.open_project(
            [str(source)], "writing", str(self.workspace)
        )
        self.assertEqual(self.events[-1]["event"], "opened", self.events)
        self.traces = self.workspace / "logs" / "traces"

    def _activate(self, directory, *, conversation_path, trace,
                  project_trace, **options):
        """Replace model/index initialization, keeping real turn operations."""
        session = _session(Path(directory))
        session.project = open_project(directory)
        session.conversation_path = conversation_path
        session.trace = trace
        session.project_trace = project_trace
        session._conversation_ownership = None
        session.chat_session.backend = "mlx"
        session.embedding_session.trace = trace
        session.chat_session.trace = trace
        session.question_suggestions = Mock(return_value=[])
        session.welcome_prompts = Mock(return_value=[])
        session.router_reply = Mock(return_value="Generated title")
        session._router_chat = Mock(return_value=session.chat_session)

        def configure(settings, **kwargs):
            for name, value in settings.items():
                setattr(session.chat_session, name, value)

        session.chat_session.configure_generation.side_effect = configure
        session.use_chat_model = Mock(side_effect=lambda model: setattr(
            session.chat_session, "model_id", model
        ))
        session.embedding_session.embed_documents.side_effect = (
            lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]
        )
        messages, turns, checkpoint = session._load_conversation()
        session.turns = turns
        session._checkpoint_id = checkpoint
        session.chat_session.messages = messages
        session.trace.emit(
            "conversation.restored", component="rag",
            data={
                "turn_uuids": [turn.conversation_turn_uuid for turn in turns],
                "turn_count": len(turns), "checkpoint_id": checkpoint,
            },
        )
        return session

    def project_details(self, event):
        return [node.get("details", {}) for node in sorted(
            _nodes(self.traces / "project.json"),
            key=lambda node: node["event"]["sequence"],
        ) if node["event"]["event_type"] == event]

    def _ask(self, question, *, interactive=False, tool=False, **options):
        """Use the real question and save boundaries with fake inference."""
        session = self.engine.session
        use = ToolUse("project_inventory", object()) if tool else None
        intent = _intent(RETRIEVAL_MODE_NONE, tool=use)
        decision = RouteDecision(intent.intent_id, True)

        def route(*args, **kwargs):
            if interactive:
                self.assertEqual(kwargs["choose"](
                    "How should I answer?", ("Same", "Other")
                ), 1)
            return decision

        def run(use, request, context):
            plan = ToolPlan(
                "project_inventory", "Inspect sources", "All files",
                2, 2, 2, 2.0, 1, 1.0,
            )
            self.assertEqual(confirm_plan(
                plan, CONFIRMATION_ALWAYS, 1, context
            ), ("quick", True))
            self.assertEqual(ask_choice(
                "Which source?", ("alpha", "beta"), context
            ), 0)
            return ToolResult("Tool answer", (), {"name": use.name})

        with _answer_patches(session, intent, decision) as mocks:
            mocks.route.side_effect = route
            mocks.save.side_effect = lambda **kw: (
                rag.RAGSession._save_conversation(session, **kw)
            )
            if tool:
                mocks.run.side_effect = lambda *args, **kw: (
                    rag.RAGSession._run_tool(session, *args, **kw)
                )
            with patch.object(
                rag, "run_tool", side_effect=run
            ), patch.object(rag, "summarize_answer", return_value="Summary"):
                self.engine.ask(question, **options)
        self.assertFalse(any(event["event"] == "error"
                             for event in self.events), self.events)
        return session.turns[-1]

    def test_project_rename_records_both_sides_of_move(self):
        """Correlate successful rename events across released trace leases."""
        turn = self._ask("Keep this question")
        original = self.workspace
        renamed = original.with_name("renamed")
        active = self.engine.active_conversation.conversation_id
        self.engine.rename_project(renamed.name)
        self.assertFalse(original.exists())
        self.assertEqual(self.engine.session.project.root, renamed)
        self.assertEqual(self.engine.active_conversation.conversation_id,
                         active)
        self.assertEqual(self.engine.session.turns[0].conversation_turn_uuid,
                         turn.conversation_turn_uuid)
        self.engine.close()
        nodes = sorted(_nodes(renamed / "logs" / "traces" / "project.json"),
                       key=lambda node: node["event"]["sequence"])
        selected = [node for node in nodes if node["event"]["event_type"]
                    in ("gui.project.rename.requested", "gui.project.closed",
                        "gui.project.open.started",
                        "gui.project.rename.completed")]
        self.assertEqual([node["event"]["event_type"] for node in selected], [
            "gui.project.open.started", "gui.project.rename.requested",
            "gui.project.closed", "gui.project.open.started",
            "gui.project.rename.completed", "gui.project.closed",
        ])
        request = selected[1]["details"]
        self.assertEqual(request, selected[4]["details"])
        self.assertTrue(request["rename_id"])
        self.assertEqual(request["source_directory"], str(original))
        self.assertEqual(request["destination_directory"], str(renamed))
        self.assertEqual(request["conversation_id"], active)
        self.assertEqual(selected[3]["details"]["reason"], "internal_reopen")

    def test_failed_project_rename_has_no_success_marker(self):
        """A recovered failed move must not appear as a completed rename."""
        original = self.workspace
        renamed = original.with_name("renamed")
        activate = self.engine._activate_project
        calls = []

        def fail_once(*args):
            """Fail activation at the new path, then allow recovery."""
            calls.append(args)
            if len(calls) == 1:
                raise RuntimeError("Failed loading model")
            return activate(*args)

        with patch.object(self.engine, "_activate_project", fail_once):
            self.engine.rename_project(renamed.name)
        self.assertTrue(original.is_dir())
        self.assertFalse(renamed.exists())
        self.assertEqual(self.engine.session.project.root, original)
        self.engine.close()
        self.assertEqual(len(self.project_details(
            "gui.project.rename.requested")), 1)
        self.assertEqual(self.project_details("gui.project.rename.completed"),
                         [])
        self.assertEqual(len(self.project_details(
            "gui.project.rename.failed")), 1)

    def test_complete_lifecycle_and_interactions(self):
        """Preserve identity, selections, and feedback across an archive."""
        engine = self.engine
        parent = engine.active_conversation
        original_path = engine.conversation_store.state_path(parent)
        profile = engine.session.project.profile
        self.commands.extend([
            {"action": "plan_choice", "choice": "option:1"},
            {"action": "plan_choice", "choice": "quick"},
            {"action": "plan_choice", "choice": "option:0"},
        ])
        turn = self._ask(
            "Inspect the source", interactive=True, tool=True,
            effort={"model": 1, "search_depth": 2},
            generation_settings={
                "temperature": profile.temperature,
                "thinking": profile.thinking,
                "max_new_tokens": profile.max_new_tokens,
                "_title_question": "Inspect",
            },
        )
        self.assertEqual(len(self.commands), 0)
        uuid = turn.conversation_turn_uuid
        self.assertEqual(engine.active_conversation.name, "Generated title")
        engine.create_conversation(branch=True, turn_count=1)
        branch = engine.active_conversation
        self.assertEqual(engine.session.turns[0].conversation_turn_uuid, uuid)
        engine.rename_conversation(branch.conversation_id, "My branch")
        request = {
            "project": str(self.workspace),
            "conversation_id": branch.conversation_id,
            "turn_uuid": uuid, "role": "assistant", "saved": True,
            "reaction": "liked", "request_id": 1,
            "revision": FeedbackStore(self.workspace).load().revision,
        }
        engine.save_message_feedback(request)
        rating = FeedbackStore(self.workspace).load()
        self.assertTrue(rating.get(uuid, "assistant").saved)
        engine.set_memory_exclusion({
            key: request[key] for key in (
                "project", "conversation_id", "turn_uuid", "role",
            )
        } | {"excluded": True, "revision": rating.revision,
             "request_id": 2})
        engine.sync_memory_index(str(self.workspace))
        self.assertTrue(FeedbackStore(self.workspace).load().get(
            uuid, "assistant"
        ).excluded)
        engine.select_conversation(parent.conversation_id)
        engine.select_conversation(parent.conversation_id)
        engine.create_conversation()
        empty = engine.active_conversation
        engine.select_conversation(parent.conversation_id)
        engine.create_conversation()
        self.assertEqual(engine.active_conversation, empty)
        engine.reset()
        self.assertNotEqual(engine.active_conversation, empty)
        engine.delete_conversation(branch.conversation_id)
        engine.select_conversation(parent.conversation_id)
        self.assertEqual(engine.session.turns[0].conversation_turn_uuid, uuid)
        engine.close()
        engine.resume_project(str(self.workspace), parent.conversation_id)
        self.assertEqual(engine.session.turns[0].conversation_turn_uuid, uuid)
        engine.close()
        self.assertFalse(any(event["event"] == "error"
                             for event in self.events), self.events)

        creation = self.project_details("gui.project.creation.settings")[0]
        self.assertEqual(creation["operation_recording_version"],
                         OPERATION_RECORDING_VERSION)
        catalog = self.project_details("gui.conversation.catalog.opened")
        self.assertEqual(len(catalog), 2)
        self.assertEqual(catalog[0]["active_conversation_id"],
                         parent.conversation_id)
        creates = self.project_details("gui.conversation.create.result")
        self.assertEqual([item["reused"] for item in creates],
                         [False, False, True])
        self.assertEqual(creates[0]["parent_conversation_id"],
                         parent.conversation_id)
        self.assertEqual(creates[0]["origin_turn_count"], 1)
        self.assertEqual(creates[2]["conversation_id"], empty.conversation_id)
        self.assertEqual(self.project_details(
            "gui.conversation.renamed"
        )[0]["name"], "My branch")
        self.assertEqual(len(self.project_details(
            "gui.conversation.select.unchanged"
        )), 1)
        self.assertEqual(len(self.project_details(
            "gui.conversation.reset.result"
        )), 1)
        deleted = self.project_details("gui.conversation.deleted")[0]
        self.assertEqual(deleted["conversation_id"], branch.conversation_id)
        self.assertFalse(deleted["deleted_active"])
        archived = self.workspace / deleted["archived_path"]
        saved = conversation_state(json.loads(archived.read_text()))
        self.assertEqual(saved["turns"][0]["conversation_turn_uuid"], uuid)
        self.assertEqual(self.project_details(
            "gui.conversation.feedback.saved"
        )[0]["rating"], {"saved": True, "reaction": "liked",
                          "excluded": False})
        self.assertTrue(self.project_details(
            "gui.conversation.memory_exclusion.saved"
        )[0]["rating"]["excluded"])
        self.assertEqual(len(self.project_details(
            "project.memory.sync.result"
        )), 1)
        operations = {node["event"]["event_type"]: node
                      for node in _nodes(original_path)}
        request = operations["gui.question.request.started"]
        self.assertEqual(request["details"]["effort"],
                         {"model": 1, "search_depth": 2})
        self.assertEqual(request["details"]["explicit_generation_settings"],
                         {})
        self.assertEqual(request["details"]["title_question"], "Inspect")
        core = operations["turn.started"]
        self.assertEqual(core["event"]["parent_span_id"],
                         request["event"]["span_id"])
        self.assertEqual(operations["state.checkpoint.committed"]["details"]
                         ["conversation_turn_uuid"], uuid)
        self.assertEqual(operations["gui.question.saved"]["details"]
                         ["conversation_turn_uuid"], uuid)
        for prefix, label in (("routing.choice", "Other"),
                              ("tool.choice", "alpha")):
            proposed = operations[prefix + ".proposed"]["details"]
            decided = operations[prefix + ".decided"]["details"]
            self.assertEqual(proposed["interaction_id"],
                             decided["interaction_id"])
            self.assertEqual(decided["selected_option"], label)
            self.assertTrue(decided["asked"])
        plan = operations["tool.plan.decided"]["details"]
        self.assertEqual(plan["choice"], "quick")
        self.assertEqual(plan["interaction_id"], operations[
            "tool.plan.proposed"
        ]["details"]["interaction_id"])
        # Every event, including terminal span metadata, survives exactly
        # once across all conversation and project files and archives.
        sequences = []
        for path in [self.traces / "project.json",
                     *self.traces.rglob("conversation_*.json")]:
            for node in _nodes(path):
                sequences.append(node["event"]["sequence"])
                if "completion" in node:
                    sequences.append(node["completion"]["sequence"])
        high = json.loads((self.traces / ".sequence.json").read_text())
        self.assertEqual(sorted(sequences),
                         list(range(1, high["last_sequence"] + 1)))
        self.assertFalse(high["incomplete"])

    def test_command_input_eof_closes_completed_project_recording(self):
        """Close the real coordinator through the engine's EOF path."""
        self._ask("Hello")
        self._ask("Another question")
        project_path = self.traces / "project.json"
        conversation_path = self.engine.conversation_store.state_path(
            self.engine.active_conversation
        )
        state = conversation_state(json.loads(conversation_path.read_text()))
        self.assertEqual(len(state["turns"]), 2)
        # No explicit close command: the parent ends its write channel.
        with (
            patch.object(gui_engine, "ChatEngine", return_value=self.engine),
            patch.object(gui_engine, "default_workspace_directory",
                         return_value=self.root),
            patch("sys.stdin", io.StringIO()),
            patch("sys.stdout", io.StringIO()),
        ):
            gui_engine.main()
        self.assertIsNone(self.engine.session)
        self.assertIsNone(self.engine.active_conversation)
        events = []
        for path in (project_path, conversation_path):
            nodes = list(_nodes(path))
            events.extend(node["event"] for node in nodes)
            events.extend(node["completion"] for node in nodes
                          if "completion" in node)
            self.assertEqual(sum(e["event_type"] == "trace.started"
                                 for e in (n["event"] for n in nodes)),
                             sum(e["event_type"] == "trace.closed"
                                 for e in (n["event"] for n in nodes)))
        events.sort(key=lambda event: event["sequence"])
        self.assertEqual([e["event_type"] for e in events[-3:]], [
            "gui.project.closed", "trace.closed", "trace.closed",
        ])
        counter = json.loads((self.traces / ".sequence.json").read_text())
        self.assertFalse(counter["incomplete"])
        self.assertEqual([e["sequence"] for e in events],
                         list(range(1, counter["last_sequence"] + 1)))
        self.assertEqual(conversation_state(json.loads(
            conversation_path.read_text()
        )), state)
        before = [p.read_bytes() for p in (project_path, conversation_path)]
        self.engine.close()
        self.assertEqual(before, [p.read_bytes() for p in (
            project_path, conversation_path,
        )])

    def test_delete_last_creates_and_records_replacement(self):
        old = self.engine.active_conversation
        path = self.engine.conversation_store.state_path(old)
        before = path.read_bytes()
        self.engine.delete_conversation(old.conversation_id)
        replacement = self.engine.active_conversation
        self.assertNotEqual(old.conversation_id, replacement.conversation_id)
        result = self.project_details("gui.conversation.deleted")[0]
        self.assertTrue(result["replacement_created"])
        self.assertTrue(result["deleted_active"])
        self.assertEqual(result["replacement"]["conversation_id"],
                         replacement.conversation_id)
        self.assertEqual(
            (self.workspace / result["archived_path"]).read_bytes(), before
        )

    def test_failed_selection_is_not_a_completed_command(self):
        original = self.engine.active_conversation
        self.engine.select_conversation("missing")
        self.assertEqual(self.engine.active_conversation, original)
        node = next(node for node in _nodes(self.traces / "project.json")
                    if node["event"]["event_type"] ==
                    "gui.conversation.select.started")
        self.assertEqual(node["completion"]["event_type"],
                         "gui.conversation.select.failed")
        self.assertEqual(self.project_details("gui.conversation.selected"), [])


if __name__ == "__main__":
    unittest.main()
