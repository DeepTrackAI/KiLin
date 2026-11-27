"""Exercise replay with ordinary traces and deterministic model doubles.

The fixtures retain real source imports, index persistence, conversation
operations, and committed answers. Replay regenerates answers rather than
copying the saved original state.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from contextlib import chdir
from copy import deepcopy
import re
import unittest
from unittest.mock import patch

from kilin import project, rag
from kilin.answering import RouteDecision
from kilin.conversation_memory.feedback import FeedbackStore
from kilin.project_type import RETRIEVAL_MODE_NONE
from kilin.system_testing import Experiment, InvalidRecordingError, ReplayError
from kilin.tools import ToolResult, ToolUse
from kilin.tools._shared.planning import (
    CONFIRMATION_ALWAYS, ToolPlan, ask_choice, confirm_plan,
)
from tests.kilin._rag_support import _answer_patches, _intent
from tests.kilin.system_testing import test_experiment as experiment_tests


class TestReplay(unittest.TestCase):
    """Reproduce actual engine recordings, including user decisions."""

    def setUp(self):
        self.original = experiment_tests.TestExperiment()
        self.addCleanup(self.original.doCleanups)
        self.original.setUp()
        self.fixture = self.original.fixture
        self.engine = self.fixture.engine
        self.experiment = self.original.experiment
        self.target = self.experiment.workspace.with_name("recorded_replay")
        self.seen = []

    def _model(self, mode="normal"):
        real_ask = rag.RAGSession.ask

        def ask(session, question, **options):
            self.seen.append((question, sorted(
                p.relative_to(session.project.source).as_posix()
                for p in session.project.source.rglob("*") if p.is_file()
            )))
            interactive = question == "Inspect the source"
            use = (ToolUse("project_inventory", object())
                   if interactive else None)
            intent = _intent(RETRIEVAL_MODE_NONE, tool=use)
            decision = RouteDecision(intent.intent_id, True)

            def route(*args, **kwargs):
                if interactive and mode != "missing":
                    label = "Other" if mode != "changed" else "Unknown"
                    self.assertEqual(kwargs["choose"](
                        "How should I answer?", (label, "Same")
                    ), 0)
                elif mode == "extra":
                    kwargs["choose"]("Unexpected?", ("Yes", "No"))
                return decision

            def run(use, request, context):
                plan = ToolPlan(
                    "project_inventory", "Inspect sources",
                    "Changed scope" if mode == "plan" else "All files",
                    2, 2, 2, 2.0, 1, 1.0,
                )
                self.assertEqual(confirm_plan(
                    plan, CONFIRMATION_ALWAYS, 1, context
                ), ("quick", True))
                self.assertEqual(ask_choice(
                    "Which source?", ("beta", "alpha"), context
                ), 1)
                return ToolResult("Replayed answer", (), {"name": use.name})

            with _answer_patches(session, intent, decision) as mocks:
                mocks.route.side_effect = route
                mocks.save.side_effect = lambda **kw: (
                    rag.RAGSession._save_conversation(session, **kw)
                )
                if use:
                    mocks.run.side_effect = lambda *args, **kw: (
                        rag.RAGSession._run_tool(session, *args, **kw)
                    )
                with patch.object(rag, "run_tool", side_effect=run), (
                    patch.object(rag, "summarize_answer", return_value="New")
                ):
                    return real_ask(session, question, **options)

        return patch.object(rag.RAGSession, "ask", ask)

    def test_complete_replay_with_reordered_options(self):
        self.fixture.test_complete_lifecycle_and_interactions()
        before = experiment_tests._snapshot(self.experiment.workspace)
        reference = self.experiment.validate()
        self.assertTrue(reference.valid, reference.issues)
        progress = []
        with self._model():
            result = self.experiment.replay(
                progress=lambda i, total, op: progress.append((i, total))
            )
        self.assertEqual(result.workspace, self.target)
        self.assertTrue(result.validation.valid, result.validation.issues)
        self.assertEqual(result.operation_count, len(reference.operations))
        self.assertEqual(progress, [(i + 1, len(reference.operations))
                                    for i in range(len(reference.operations))])
        self.assertEqual(len(result.turn_uuids), 1)
        uuid, replayed = next(iter(result.turn_uuids.items()))
        self.assertNotEqual(uuid, replayed)
        self.assertTrue(FeedbackStore(self.target).load().get(
            replayed, "assistant"
        ).excluded)
        self.assertEqual([d.field for d in result.differences
                          if d.field.startswith("choice.")],
                         ["choice.options", "choice.options"])
        self.assertEqual(before, experiment_tests._snapshot(
            self.experiment.workspace
        ))

    def test_rename_replays_close_and_reopen_in_fixed_directory(self):
        """Restore the selected chat without moving the replay directory."""
        first = self.fixture._ask("Before moving")
        active = self.engine.active_conversation.conversation_id
        original = self.experiment.workspace
        moved = original.with_name("renamed_original")
        self.engine.rename_project(moved.name)
        self.fixture._ask("After moving")
        self.engine.close()
        experiment = Experiment(moved)
        before = experiment_tests._snapshot(moved)
        report = experiment.validate()
        self.assertTrue(report.valid, report.issues)
        with self._model():
            result = experiment.replay(self.target)
        self.assertTrue(result.validation.valid, result.validation.issues)
        self.assertEqual(result.operation_count, len(report.operations))
        self.assertEqual(result.workspace, self.target)
        self.assertFalse(original.exists())
        self.assertEqual([op.kind for op in result.validation.operations], [
            "create_project", "ask", "close_project", "open_project",
            "ask", "close_project",
        ])
        opening = result.validation.operations[3]
        self.assertEqual(opening.request["conversation_id"],
                         result.conversation_ids[active])
        restored = opening.matching("conversation.restored")[0]
        self.assertEqual(restored.details["turn_uuids"], [
            result.turn_uuids[first.conversation_turn_uuid]
        ])
        changes = [d for d in result.differences
                   if d.field == "project.relocation"]
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].replayed["directory"], str(self.target))
        comparison = experiment.compare(self.target)
        self.assertTrue(comparison.complete, comparison.data["alignment"])
        row = next(op for op in comparison.data["operations"]
                   if "relocation" in op)
        self.assertFalse(row["relocation"]["physical_rename_replayed"])
        self.assertIn("gui.project.rename.requested",
                      comparison.data["expected_event_differences"])
        self.assertIn("gui.project.rename.completed",
                      comparison.data["expected_event_differences"])
        self.assertEqual(row["relocation"]["original"]["provenance"],
                         "recorded")
        self.assertIn("replay directory retained", comparison.markdown())
        self.assertEqual(before, experiment_tests._snapshot(moved))

    def test_sources_are_imported_in_order_from_retained_copies(self):
        self.fixture._ask("First")
        folder = self.fixture.root / "folder"
        folder.mkdir()
        (folder / "first.txt").write_text("First file", encoding="utf-8")
        (folder / "later.txt").write_text("Later file", encoding="utf-8")
        self.engine.add_project_sources(
            [str(folder)], selected_files={str(folder): ["first.txt"]}
        )
        self.fixture._ask("Second")
        self.engine.create_conversation()
        self.engine.sync_sources({"sources/folder/later.txt": "import"})
        self.fixture._ask("Third")
        self.engine.close()
        for p in (folder / "first.txt", folder / "later.txt",
                  self.fixture.root / "source.txt"):
            p.unlink()
        before = experiment_tests._snapshot(self.experiment.workspace)
        with self._model():
            result = self.experiment.replay(self.target)
        self.assertEqual(self.seen, [
            ("First", ["source.txt"]),
            ("Second", ["folder/first.txt", "source.txt"]),
            ("Third", ["folder/first.txt", "folder/later.txt", "source.txt"]),
        ])
        self.assertEqual(len(result.turn_uuids), 3)
        self.assertEqual(before, experiment_tests._snapshot(
            self.experiment.workspace
        ))

    def test_extra_interaction_stops_and_retains_partial_workspace(self):
        self.fixture._ask("First")
        self.fixture._ask("Second")
        self.engine.close()
        before = experiment_tests._snapshot(self.experiment.workspace)
        with self._model("extra"), self.assertRaises(ReplayError) as caught:
            self.experiment.replay()
        self.assertIn("Unexpected interaction", str(caught.exception))
        self.assertEqual(caught.exception.completed_operations, 1)
        self.assertEqual(caught.exception.workspace, self.target)
        self.assertTrue(self.target.is_dir())
        self.assertEqual([q for q, _ in self.seen], ["First"])
        self.assertEqual(before, experiment_tests._snapshot(
            self.experiment.workspace
        ))

    def test_changed_choice_stops(self):
        self.fixture.test_complete_lifecycle_and_interactions()
        with self._model("changed"), self.assertRaises(ReplayError) as caught:
            self.experiment.replay()
        self.assertIn("missing or ambiguous", str(caught.exception))
        self.assertEqual(caught.exception.completed_operations, 1)

    def test_changed_plan_scope_stops(self):
        self.fixture.test_complete_lifecycle_and_interactions()
        with self._model("plan"), self.assertRaises(ReplayError) as caught:
            self.experiment.replay()
        self.assertIn("Tool plan scope or meaning changed",
                      str(caught.exception))
        self.assertEqual(caught.exception.completed_operations, 1)

    def test_current_defaults_and_explicit_overrides(self):
        # Use the actual creation form's partial nested backend mapping.
        self.engine.close()
        root = self.fixture.root / "settings_original"
        self.engine.settings("writing", "")
        defaults = self.engine._creation_defaults[1]
        settings = {key: deepcopy(defaults[key]) for key in (
            "backend", "allow_download", "temperature", "thinking",
            "max_new_tokens", "_gui_search_depth", "_gui_router_models",
            "_gui_source_cards", "_gui_effort_levels",
        )}
        settings["_gui_effort_levels"].pop("torch")
        settings["_gui_router_models"].pop("torch")
        settings["_gui_search_depth"] = [1, 2, 3, 4, 5]
        settings["_gui_full_snapshot"] = True
        self.engine.open_project([], "writing", str(root), settings=settings)
        self.assertEqual(self.fixture.events[-1]["event"], "opened")
        profile = self.engine.session.project.profile
        self.fixture._ask("Inherited", generation_settings={
            "temperature": profile.temperature,
            "thinking": profile.thinking,
            "max_new_tokens": profile.max_new_tokens,
            "_title_question": "Inherited",
        })
        self.fixture._ask("Explicit", generation_settings={
            "temperature": 0.13,
            "thinking": profile.thinking,
            "max_new_tokens": profile.max_new_tokens,
            "_title_question": "Explicit",
        })
        self.engine.close()
        original = Experiment(root)
        report = original.validate()
        self.assertTrue(report.valid, report.issues)
        old_template = project._default_profile_text

        def current_template(*args, **kwargs):
            return re.sub(r"(?m)^temperature = .*?$", "temperature = 0.91",
                          old_template(*args, **kwargs))

        with patch.object(project, "_default_profile_text",
                          side_effect=current_template), self._model():
            result = original.replay()
        creation = result.validation.operations[0]
        settings = creation.request["settings"]
        self.assertEqual(set(settings["requested_settings"]), {
            "backend", "allow_download", "temperature", "thinking",
            "max_new_tokens", "_gui_search_depth", "_gui_router_models",
            "_gui_source_cards", "_gui_effort_levels",
        })
        self.assertEqual(set(settings["requested_settings"][
            "_gui_effort_levels"
        ]), {"mlx"})
        self.assertEqual(settings["requested_settings"]["temperature"], 0.91)
        self.assertEqual(settings["requested_settings"]["_gui_search_depth"],
                         [1, 2, 3, 4, 5])
        asks = [op for op in result.validation.operations if op.kind == "ask"]
        self.assertEqual([op.request["generation_settings"]["temperature"]
                          for op in asks], [0.91, 0.13])
        self.assertIn("configuration.profile.temperature",
                      [d.field for d in result.differences])

    def test_collision_adjusted_names_and_partial_branch(self):
        self.fixture._ask("Before")
        folder = self.fixture.root / "second"
        folder.mkdir()
        source = folder / "source.txt"
        source.write_text("Another source", encoding="utf-8")
        self.engine.add_project_sources([str(source)])
        self.fixture._ask("After")
        self.engine.create_conversation(branch=True, turn_count=1)
        self.fixture._ask("Branch")
        self.engine.close()
        names = sorted(p.name for p in
                       (self.experiment.workspace / "sources").iterdir())
        self.assertEqual(len(names), 2)
        source.unlink()
        with self._model():
            result = self.experiment.replay()
        self.assertEqual(self.seen, [("Before", ["source.txt"]),
                                    ("After", names), ("Branch", names)])
        self.assertEqual(len(result.turn_uuids), 3)

    def test_empty_creation_and_sync_noop(self):
        self.engine.close()
        root = self.fixture.root / "blank_original"
        self.engine.open_project([], "writing", str(root))
        self.engine.sync_sources(None)
        self.engine.close()
        result = Experiment(root).replay()
        self.assertEqual(result.workspace.name, "blank_replay")
        self.assertEqual(result.operation_count, 3)
        self.assertEqual(result.turn_uuids, {})

    def test_missing_interaction_stops(self):
        self.fixture.commands.append({"action": "plan_choice",
                                      "choice": "option:1"})
        self.fixture._ask("Inspect the source", interactive=True)
        self.engine.close()
        with self._model("missing"), self.assertRaises(ReplayError) as caught:
            self.experiment.replay()
        self.assertEqual(caught.exception.completed_operations, 1)

    def test_existing_source_noop_and_active_archive(self):
        self.original.test_additive_source_batches_and_active_archive()
        result = self.experiment.replay()
        self.assertTrue(result.validation.valid, result.validation.issues)
        self.assertEqual(len(result.conversation_ids), 2)
        self.assertEqual(len(list((self.target / "logs/traces/.deleted")
                                  .glob("*.json"))), 1)

    def test_manual_regeneration_without_selected_cards(self):
        self.engine.close()
        root = self.fixture.root / "cards_original"
        self.engine.settings("writing", "")
        cards = deepcopy(
            self.engine._creation_defaults[1]["_gui_source_cards"]
        )
        cards["policy"].update(enabled=True, automatic=False)
        self.engine.open_project([], "writing", str(root), settings={
            "_gui_source_cards": cards,
        })
        self.assertEqual(self.fixture.events[-1]["event"], "opened")
        self.engine.regenerate_source_cards()
        self.assertEqual(self.fixture.events[-1]["event"],
                         "source_cards_regenerated")
        self.engine.close()
        result = Experiment(root).replay()
        self.assertEqual([op.kind for op in result.validation.operations], [
            "create_project", "regenerate_source_cards", "close_project",
        ])

    def test_symlink_destination_is_not_followed(self):
        self.engine.close()
        try:
            self.target.symlink_to(self.experiment.workspace,
                                   target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Symbolic links unavailable: {error}")
        with self.assertRaises(ValueError):
            self.experiment.replay()
        self.assertTrue(self.target.is_symlink())

    def test_working_directory_cannot_override_inherited_defaults(self):
        self.engine.close()
        root = self.fixture.root / "gui_original"
        self.engine.settings("writing", "")
        temperature = self.engine._creation_defaults[1]["temperature"]
        self.engine.open_project([], "writing", str(root), settings={
            "temperature": temperature, "_gui_full_snapshot": True,
        })
        self.assertEqual(self.fixture.events[-1]["event"], "opened")
        self.engine.close()
        reference = Experiment(root)
        self.assertTrue(reference.validate().valid)
        before = experiment_tests._snapshot(root)
        with chdir(root), self.assertRaises(ReplayError) as caught:
            reference.replay()
        self.assertIn("directory without profile.ini", str(caught.exception))
        self.assertEqual(caught.exception.completed_operations, 0)
        self.assertFalse(root.with_name("gui_replay").exists())
        self.assertEqual(before, experiment_tests._snapshot(root))

    def test_destination_protection_and_invalid_reference(self):
        self.engine.close()
        for target in (self.experiment.workspace,
                       self.experiment.workspace / "nested"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.experiment.replay(target)
        self.target.mkdir()
        sentinel = self.target / "keep.txt"
        sentinel.write_text("Keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.experiment.replay()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "Keep")
        other = self.target.with_name("other_replay")
        Experiment(self.target)  # Construction itself must not mutate it.
        with self.assertRaises(InvalidRecordingError):
            Experiment(self.target).replay(other)
        self.assertFalse(other.exists())


if __name__ == "__main__":
    unittest.main()
