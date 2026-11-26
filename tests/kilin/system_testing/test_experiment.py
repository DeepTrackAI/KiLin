"""Exercise real recorded histories and reject inconsistent references.

Use the existing deterministic ChatEngine fixture with real source scanning,
chunk preparation, index persistence and reuse. No model weights, model
backend, or Qt application are necessary.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

import json
from shutil import copytree
import unittest
from unittest.mock import patch

from kilin.project import open_project
from kilin.system_testing import Experiment, InvalidRecordingError
from kilin.trace_logging import TraceLogger
from tests.kilin.gui import test_engine__recording as recording_tests


def _snapshot(root):
    """Inventory all paths and contents without relying on access times."""
    return {
        p.relative_to(root).as_posix(): (p.read_bytes() if p.is_file()
                                        else None)
        for p in root.rglob("*")
    }


class TestExperiment(unittest.TestCase):
    """Validate producer output rather than a hand-invented trace schema."""

    def setUp(self):
        fixture = recording_tests.TestEngineRecording(
            "test_complete_lifecycle_and_interactions"
        )
        self.addCleanup(fixture.doCleanups)
        activate = fixture._activate

        def with_indexes(directory, **kwargs):
            session = activate(directory, **kwargs)
            # Only the model identity and tokenizer are substituted. Keep
            # real source reading, chunking, hashing and index persistence.
            embedding = session.embedding_session
            embedding._retrieval_model_identity = "hf:fixture"
            embedding.count_tokens.side_effect = (
                lambda text: len(text.split())
            )
            with session._project_embedding_diagnostics(), (
                session.project_trace.span("indexing", component="rag")
            ):
                (session.chunks, session.indexes, session.index_status,
                 session.source_changes) = session._prepare_retrieval(
                    embedding
                )
            session.source_cards, session.source_card_vectors = (
                session._prepare_source_cards(
                    session.chunks, session.indexes, embedding
                )
            )
            return session

        fixture._activate = with_indexes
        fixture.setUp()
        self.fixture = fixture
        self.root = fixture.workspace
        self.engine = fixture.engine
        self.experiment = Experiment(self.root)

    def assertValid(self, experiment=None):
        report = (experiment or self.experiment).validate()
        self.assertTrue(report.valid, report.issues)
        return report

    def assertInvalid(self, code, experiment=None):
        experiment = experiment or self.experiment
        report = experiment.validate()
        self.assertFalse(report.valid)
        self.assertIn(code, {issue.code for issue in report.issues},
                      report.issues)
        with self.assertRaises(InvalidRecordingError) as raised:
            experiment.timeline()
        self.assertFalse(raised.exception.report.valid)
        return report

    def test_complete_reference_and_read_only_inspection(self):
        self.fixture.test_complete_lifecycle_and_interactions()
        before = _snapshot(self.root)
        report = self.assertValid()
        self.assertEqual(report.event_count, len(report.events))
        self.assertEqual([e.sequence for e in report.events],
                         list(range(1, report.event_count + 1)))
        operations = self.experiment.timeline()
        self.assertEqual([op.kind for op in operations], [
            "create_project", "ask", "create_conversation",
            "rename_conversation", "feedback", "memory_exclusion",
            "sync_memory", "select_conversation", "select_conversation",
            "create_conversation", "select_conversation",
            "create_conversation", "reset_conversation",
            "delete_conversation", "select_conversation", "close_project",
            "open_project", "close_project",
        ])
        ask = operations[1]
        self.assertEqual(ask.request["question"], "Inspect the source")
        self.assertEqual(ask.matching("tool.choice.decided")[0]
                         .details["selected_option"], "alpha")
        self.assertEqual(ask.matching("tool.plan.decided")[0]
                         .details["choice"], "quick")
        self.assertEqual(len(operations[0].matching("index.rebuilt")), 1)
        self.assertEqual(len(operations[-2].matching("index.reused")), 1)
        self.assertEqual(before, _snapshot(self.root))

    def test_additive_source_batches_and_active_archive(self):
        folder = self.fixture.root / "folder"
        folder.mkdir()
        (folder / "first.txt").write_text("First source", encoding="utf-8")
        (folder / "later.txt").write_text("Later source", encoding="utf-8")
        self.engine.add_project_sources(
            [str(folder)], selected_files={str(folder): ["first.txt"]}
        )
        self.engine.sync_sources({"sources/folder/later.txt": "import"})
        self.engine.add_project_sources([str(folder)])
        self.engine.delete_conversation(
            self.engine.active_conversation.conversation_id
        )
        self.engine.close()
        # The external originals are no longer required for validation.
        (folder / "first.txt").unlink()
        (folder / "later.txt").unlink()
        (self.fixture.root / "source.txt").unlink()
        report = self.assertValid()
        self.assertEqual([op.kind for op in report.operations], [
            "create_project", "add_sources", "sync_sources", "add_sources",
            "delete_conversation", "close_project",
        ])
        counts = [len(e.details["source_manifest"]["files"])
                  for e in report.events
                  if e.event_type == "index.check.completed"]
        self.assertEqual(counts, [1, 2, 3])
        archived = self.root / report.operations[-2].matching(
            "gui.conversation.deleted"
        )[0].details["archived_path"]
        self.assertTrue(archived.is_file())

    def test_blank_reference(self):
        self.engine.close()
        root = self.fixture.root / "blank"
        self.engine.open_project([], "writing", str(root))
        self.assertEqual(self.fixture.events[-1]["event"], "opened")
        self.engine.close()
        report = self.assertValid(Experiment(root))
        self.assertEqual(len(report.operations[0].matching("index.empty")), 1)

    def test_partial_branch_new_question_and_generation_overrides(self):
        self.fixture._ask("First question")
        self.fixture._ask("Second question", generation_settings={
            "temperature": 0.3, "thinking": False, "max_new_tokens": 128,
        })
        self.engine.create_conversation(branch=True, turn_count=1)
        self.engine.rename_conversation(
            self.engine.active_conversation.conversation_id, "Branch  name"
        )
        self.fixture._ask("Branch question")
        self.engine.close()
        report = self.assertValid()
        asks = [op for op in report.operations if op.kind == "ask"]
        self.assertEqual(len(asks), 3)
        self.assertEqual(asks[1].request["explicit_generation_settings"], {
            "temperature": 0.3, "thinking": False, "max_new_tokens": 128,
        })

    def _rename(self, name):
        """Use the ordinary engine command and follow the moved fixture."""
        destination = self.root.with_name(name)
        self.engine.rename_project(name)
        self.assertEqual(self.engine.session.project.root, destination)
        self.root = destination
        self.experiment = Experiment(destination)

    def test_recorded_consecutive_renames(self):
        """Recognize distinct marker pairs, even with no intervening ask."""
        self.fixture._ask("First question")
        self.engine.create_conversation()
        self.fixture._ask("Second question")
        original = self.root
        active = self.engine.active_conversation.conversation_id
        self._rename("first_move")
        middle = self.root
        self._rename("second_move")
        self.engine.rename_project(self.root.name)  # No-op.
        self.engine.close()
        before = _snapshot(self.root)
        report = self.assertValid()
        self.assertFalse(original.exists())
        self.assertFalse(middle.exists())
        openings = [op for op in report.operations
                    if op.kind == "open_project"]
        self.assertEqual(len(openings), 2)
        self.assertEqual([op.request["conversation_id"] for op in openings],
                         [active, active])
        self.assertEqual([op.request["requested_conversation_id"]
                          for op in openings], [active, active])
        moves = [op.request["relocation"] for op in openings]
        self.assertEqual([m["provenance"] for m in moves],
                         ["recorded", "recorded"])
        self.assertEqual([m["source_directory"] for m in moves],
                         [str(original), str(middle)])
        self.assertEqual([m["destination_directory"] for m in moves],
                         [str(middle), str(self.root)])
        self.assertNotEqual(moves[0]["rename_id"], moves[1]["rename_id"])
        self.assertNotIn("relocation.inferred", {w.code for w in
                                                 report.warnings})
        self.assertEqual(before, _snapshot(self.root))

    def _rename_without_markers(self, *, completion_only=False):
        """Simulate an older producer or a missing completion marker."""
        emit = TraceLogger.emit

        def legacy(logger, event_type, **options):
            if event_type == "gui.project.rename.completed" or (
                    not completion_only and
                    event_type == "gui.project.rename.requested"):
                return None
            return emit(logger, event_type, **options)

        with patch.object(TraceLogger, "emit", legacy):
            self._rename("renamed")
        self.engine.close()

    def test_legacy_rename_is_inferred_with_warning(self):
        """Accept an already-recorded move only with independent paths."""
        self.fixture._ask("Before legacy rename")
        self._rename_without_markers()
        report = self.assertValid()
        reopening = next(op for op in report.operations
                         if op.kind == "open_project")
        self.assertEqual(reopening.request["relocation"]["provenance"],
                         "inferred")
        self.assertIn("relocation.inferred", {w.code for w in report.warnings})
        self.assertFalse(any(e.event_type.startswith("gui.project.rename.")
                             for e in report.events))

    def test_incomplete_rename_markers_do_not_use_legacy_fallback(self):
        self._rename_without_markers(completion_only=True)
        self.assertInvalid("relocation.history")

    def test_mismatched_rename_markers(self):
        self._rename("renamed")
        self.engine.close()
        self._change(self.root / "logs/traces/project.json",
                     "gui.project.rename.completed", lambda node: node[
                         "details"].update(rename_id="different"))
        self.assertInvalid("relocation.history")

    def test_legacy_rename_requires_independent_path_corroboration(self):
        self._rename_without_markers()
        self._change(self.root / "logs/traces/project.json",
                     "gui.project.configuration.effective", lambda node: node[
                         "details"]["behavior"].update(
                             project=str(self.root.with_name("other"))))
        self.assertInvalid("relocation.history")

    def test_unrelated_same_directory_internal_reopen_is_rejected(self):
        active = self.engine.active_conversation.conversation_id
        self.engine.close()
        self.engine._reopen_project_conversation(open_project(self.root),
                                                 active)
        self.engine.close()
        self.assertInvalid("relocation.history")

    def test_failed_project_rename_is_not_replayable(self):
        activate = self.engine._activate_project
        calls = []

        def fail_once(*args):
            calls.append(args)
            if len(calls) == 1:
                raise RuntimeError("Failed to reopen after move")
            return activate(*args)

        with patch.object(self.engine, "_activate_project", fail_once):
            self.engine.rename_project("renamed")
        self.engine.close()
        self.assertInvalid("command.failed")
        self.assertInvalid("relocation.history")

    def test_relocated_reference_and_environment_difference(self):
        self.engine.close()
        root = self.fixture.root / "relocated"
        copytree(self.root, root)
        path = root / "logs/traces/project.json"
        self._change(path, "trace.started", lambda node: node["details"][
            "environment"
        ].update(platform="Other platform"))
        report = self.assertValid(Experiment(root))
        self.assertIn("environment.difference", {w.code for w in
                                                  report.warnings})

    def test_no_creation_or_modification_for_missing_reference(self):
        root = self.fixture.root / "missing"
        self.assertInvalid("counter.invalid", Experiment(root))
        self.assertFalse(root.exists())

    def test_unclosed_reference_is_invalid(self):
        self.assertInvalid("project.unclosed")

    def test_missing_counter_and_recording_health(self):
        self.engine.close()
        path = self.root / "logs/traces/.sequence.json"
        original = path.read_bytes()
        path.unlink()
        self.assertInvalid("counter.invalid")
        data = json.loads(original)
        data["incomplete"] = True
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertInvalid("counter.incomplete")

    def test_invalid_sequence_and_version(self):
        self.engine.close()
        path = self.root / "logs/traces/project.json"
        original = path.read_bytes()
        self._change(path, "trace.started", lambda node: node["event"].update(
            sequence=2
        ))
        report = self.assertInvalid("sequence.duplicate")
        self.assertIn("sequence.gap", {i.code for i in report.issues})
        path.write_bytes(original)
        self._change(path, "gui.project.creation.settings", lambda node:
                     node["details"].update(operation_recording_version=0))
        self.assertInvalid("command.coverage")

    def test_missing_archived_trace_and_malformed_json(self):
        self.engine.delete_conversation(
            self.engine.active_conversation.conversation_id
        )
        self.engine.close()
        archive = next((self.root / "logs/traces/.deleted").glob("*.json"))
        archive.unlink()
        self.assertInvalid("conversation.missing")
        archive.write_text('{"duplicate": 1, "duplicate": 2}',
                           encoding="utf-8")
        self.assertInvalid("trace.invalid")

    def test_historical_source_tampering_and_unrecorded_file(self):
        self.engine.close()
        (self.root / "sources/source.txt").write_text("Changed evidence.")
        (self.root / "sources/unrecorded.txt").write_text("Extra file.")
        report = self.assertInvalid("source.content")
        self.assertIn("source.unrecorded", {i.code for i in report.issues})

    def test_manifest_corruption_does_not_borrow_future_files(self):
        self.engine.close()
        path = self.root / "logs/traces/project.json"
        self._change(path, "index.check.completed", lambda node: node[
            "details"
        ]["source_manifest"]["files"].update({
            "future.txt": {"sha256": "a" * 64, "size_bytes": 1}
        }))
        self.assertInvalid("source.history")

    def test_invalid_saved_conversation_content(self):
        self.fixture._ask("Question")
        self.engine.close()
        path = next((self.root / "logs/traces").glob("conversation_*.json"))
        data = json.loads(path.read_text(encoding="utf-8"))
        state = next(s["resume_state"] for s in data["sessions"].values()
                     if "resume_state" in s)
        state["turns"][0]["answer"] = "Altered"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertInvalid("conversation.state")

    def test_invalid_interaction_label(self):
        self.fixture.commands.append({"action": "plan_choice",
                                      "choice": "option:1"})
        self.fixture._ask("Question", interactive=True)
        self.engine.close()
        path = next((self.root / "logs/traces").glob("conversation_*.json"))
        self._change(path, "routing.choice.decided", lambda node:
                     node["details"].update(selected_option="Same"))
        self.assertInvalid("conversation.history")

    def test_persistent_profile_change(self):
        self.engine.close()
        path = self.root / "profile.ini"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("depth = 5, 8, 10, 15, 20",
                                     "depth = 1, 2, 3, 4, 5"),
                        encoding="utf-8")
        self.assertInvalid("configuration.state")

    def test_external_source_profile_is_rejected_without_opening_it(self):
        self.engine.close()
        path = self.root / "profile.ini"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("source = sources", "source = /other"),
                        encoding="utf-8")
        with patch("kilin.system_testing.validation.open_project") as opener:
            self.assertInvalid("configuration.state")
        opener.assert_not_called()

    def test_invalid_command_and_no_timeline(self):
        self.engine.select_conversation("missing")
        self.engine.close()
        report = self.assertInvalid("command.failed")
        self.assertIn("failed", [op.status for op in report.operations])

    def test_failure_before_source_card_regeneration_span(self):
        # Writing projects have source cards disabled. The error is emitted
        # before the core regeneration span could start.
        self.engine.regenerate_source_cards()
        self.engine.close()
        report = self.assertInvalid("command.failed")
        self.assertNotIn("regenerate_source_cards", [
            op.kind for op in report.operations
        ])

    def test_committed_answer_after_notification_failure(self):
        send = self.engine._send

        def fail_answered(event, **kwargs):
            if event == "answered":
                raise RuntimeError("Presentation failed")
            return send(event, **kwargs)

        with patch.object(self.engine, "_send", side_effect=fail_answered):
            # The fixture expects a successful presentation; the injected
            # error is deliberately reported after the core saves the turn.
            with self.assertRaises(AssertionError):
                self.fixture._ask("Question")
        self.assertEqual(len(self.engine.session.turns), 1)
        self.engine.close()
        report = self.assertInvalid("command.committed_with_error")
        ask = next(op for op in report.operations if op.kind == "ask")
        self.assertEqual(ask.status, "committed_with_error")
        self.assertEqual(len(ask.matching("state.checkpoint.committed")), 1)
        self.assertNotIn("conversation.state", {i.code for i in report.issues})

    def test_linked_source_is_rejected(self):
        self.engine.close()
        path = self.root / "sources/source.txt"
        path.unlink()
        try:
            path.symlink_to(self.fixture.root / "source.txt")
        except OSError:
            self.skipTest("Symbolic links are unavailable.")
        self.assertInvalid("source.link")

    def _change(self, path, event_type, change):
        data = json.loads(path.read_text(encoding="utf-8"))

        def visit(value):
            if isinstance(value, dict):
                if value.get("event", {}).get("event_type") == event_type:
                    change(value)
                    return True
                return any(visit(child) for child in value.values())
            if isinstance(value, list):
                return any(visit(child) for child in value)
            return False

        self.assertTrue(visit(data), event_type)
        path.write_text(json.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
