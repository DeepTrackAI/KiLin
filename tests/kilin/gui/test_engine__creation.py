"""Keep existing projects and live sessions intact on creation conflicts.

These tests use real workspace/conversation files and mocked model loading.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import kilin
from kilin.gui.engine import ChatEngine
from tests.kilin.gui.test_engine__projects import FakeSession


class TestEngineCreation(unittest.TestCase):
    """Validate create/resume/import separation before any session mutation."""

    def setUp(self):
        """Activate a real project with a traced fake-model conversation."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "note.txt"
        self.source.write_text("Original evidence.", encoding="utf-8")
        self.project = kilin.prepare_sources_workspace(
            [self.source], "literature", self.root / "existing"
        )
        self.events = []
        patcher = patch("kilin.gui.engine.RAGSession", FakeSession)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.engine = ChatEngine(self.root, self.events.append,
                                 trace_enabled=True)
        self.engine.resume_project(str(self.project.root))
        self.addCleanup(self.engine.close)
        self.assertIsNotNone(self.engine.session)
        self.events.clear()

    def snapshot(self, root):
        """Read exact bytes and modification times of the protected tree.

        Parameters
        ----------
        root: pathlib.Path
            Protected workspace.

        Returns
        -------
        dict
            Snapshot of each entry.

        """
        return {
            path.relative_to(root).as_posix(): (
                path.stat().st_mtime_ns,
                path.read_bytes() if path.is_file() else None,
            )
            for path in [root, *root.rglob("*")]
            # A Windows process may hold these advisory lock files with
            # exclusive access. The test protects persisted user data.
            if not (path.is_file() and path.suffix == ".lock")
        }

    def test_explicit_conflict_leaves_session_and_trace_bytes_unchanged(self):
        """Reject before closing, configuring, or writing conversations."""
        session = self.engine.session
        conversation = self.engine.active_conversation
        before = self.snapshot(self.project.root)
        with patch("kilin.gui.engine.ConversationStore") as store:
            self.engine.open_project(
                [str(self.source)], "literature", str(self.project.root),
                settings={"temperature": 0.7, "_gui_search_depth": [1] * 5},
            )
        store.assert_not_called()
        self.assertIs(self.engine.session, session)
        self.assertFalse(session.closed)
        self.assertIs(self.engine.active_conversation, conversation)
        self.assertEqual(self.snapshot(self.project.root), before)
        conflict = self.events[-1]
        self.assertEqual(conflict["event"], "workspace_conflict")
        self.assertTrue(conflict["existing_project"])
        self.assertEqual(conflict["workspace"], str(self.project.root))

    def test_start_screen_choices_go_into_the_new_project(self):
        """The intent model and search depth chosen before creation."""
        target = self.root / "chosen"
        self.engine.open_project(
            [str(self.source)], "literature", str(target),
            settings={
                "_gui_search_depth": [3, 4, 5, 6, 7],
                "_gui_router_models": {"mlx": "Qwen/Qwen3-1.7B-MLX-4bit"},
            },
        )
        self.assertEqual(self.events[-1]["event"], "opened")
        # Reported before indexing, so a Stop then can remove it again.
        names = [event["event"] for event in self.events]
        created = names.index("project_created")
        self.assertLess(created, names.index("opened"))
        self.assertEqual(
            self.events[created]["workspace"], str(target.resolve())
        )
        profile = (target / "profile.ini").read_text(encoding="utf-8")
        self.assertIn("depth = 3, 4, 5, 6, 7", profile)
        self.assertIn("mlx = Qwen/Qwen3-1.7B-MLX-4bit", profile)

    def test_source_card_preferences_precede_first_indexing(self):
        """Save card choices before a new project session is constructed."""
        from kilin.gui import engine as gui_engine
        from kilin.model_settings import (
            read_source_card_generation_settings,
        )
        from kilin.source_card_policy import read_source_card_policy
        from kilin.project_type import project_type

        target = self.root / "cards-before-index"
        settings = self.engine._source_card_settings(None, "literature")
        settings["policy"]["automatic"] = False
        settings["models"]["mlx"]["model_id"] = "example/card-mlx"
        settings["models"]["torch"]["model_id"] = "example/card-torch"
        seen = []

        class CheckingSession(FakeSession):
            """Observe persisted settings before RAG session startup."""

            def __init__(self, directory, **options):
                """Assert profile writes precede any model work."""
                path = Path(directory)
                policy = read_source_card_policy(
                    path, project_type("literature")
                )
                self_assert = (policy.enabled, policy.automatic)
                seen.append(self_assert)
                seen.append(read_source_card_generation_settings(
                    path, "mlx"
                ).model_id)
                super().__init__(directory, **options)

        with patch.object(gui_engine, "RAGSession", CheckingSession):
            self.engine.open_project(
                [str(self.source)], "literature", str(target),
                settings={"_gui_source_cards": settings},
            )
        self.assertEqual(self.events[-1]["event"], "opened")
        self.assertEqual(seen, [(True, False), "example/card-mlx"])
        self.assertEqual(
            read_source_card_generation_settings(target, "torch").model_id,
            "example/card-torch",
        )

    def test_invalid_new_card_preferences_leave_current_project(self):
        """Reject malformed card settings before creating a workspace."""
        settings = self.engine._source_card_settings(None, "literature")
        settings["models"]["torch"]["max_input_tokens"] = 0
        original = self.engine.session
        target = self.root / "invalid-new-card-settings"
        self.engine.open_project(
            [str(self.source)], "literature", str(target),
            settings={"_gui_source_cards": settings},
        )
        self.assertEqual(self.events[-1]["event"], "error")
        self.assertIs(self.engine.session, original)
        self.assertFalse(target.exists())

    def test_gui_router_preference_is_saved_for_new_projects(self):
        """Seed the project from user preferences, not the core fallback."""

        from kilin.gui.preferences import write_defaults

        settings = self.root / "preferences.json"
        target = self.root / "custom-router"
        with patch.dict(
            os.environ, {"KILIN_DEFAULTS_FILE": str(settings)}
        ):
            write_defaults({
                "router_models": {"mlx": "example/preferred-router"},
            })
            self.engine.open_project(
                [str(self.source)], "literature", str(target)
            )
        self.assertEqual(self.events[-1]["event"], "opened")
        self.assertEqual(
            kilin.read_router_model(target, "mlx"),
            "example/preferred-router",
        )

    def test_partial_destination_is_not_resumed_or_modified(self):
        """Unrelated contents do not become an existing-project candidate."""
        target = self.root / "partial"
        target.mkdir()
        (target / "keep.txt").write_text("preserve", encoding="utf-8")
        before = self.snapshot(target)
        session = self.engine.session
        self.engine.open_project([], "code", str(target))
        self.assertEqual(self.events[-1]["event"], "workspace_conflict")
        self.assertFalse(self.events[-1]["existing_project"])
        self.assertIs(self.engine.session, session)
        self.assertFalse(session.closed)
        self.assertEqual(self.snapshot(target), before)

    def test_creation_failure_keeps_the_previous_chat_available(self):
        """A failed new source copy must not close the existing session."""
        session = self.engine.session
        before = self.snapshot(self.project.root)
        target = self.root / "failed"
        with patch("kilin.workspace._copy_file", side_effect=OSError("full")):
            self.engine.open_project([str(self.source)], "literature",
                                     str(target))
        self.assertEqual(self.events[-1]["event"], "error")
        self.assertIs(self.engine.session, session)
        self.assertFalse(session.closed)
        self.assertEqual(self.snapshot(self.project.root), before)
        self.assertFalse(target.exists())

    def test_default_creation_delegates_unique_allocation_to_core(self):
        """Activate each allocated path instead of the preview name."""
        self.engine.open_project([str(self.source)], "literature", "")
        self.assertEqual(self.events[-1]["event"], "opened")
        first = self.engine.session.project
        source_bytes = (first.source / self.source.name).read_bytes()
        self.engine.open_project([str(self.source)], "literature", "")
        self.assertEqual(self.events[-1]["event"], "opened")
        second = self.engine.session.project
        self.assertNotEqual(first.root, second.root)
        self.assertEqual(second.root.name, first.root.name + "-2")
        self.assertEqual(self.events[-1]["project"], str(second.root))
        self.assertEqual((first.source / self.source.name).read_bytes(),
                         source_bytes)

    def test_empty_default_name_is_allocated_in_the_core(self):
        """A missing name never resumes or changes an occupied project."""
        existing = kilin.prepare_sources_workspace(
            [], "code", self.root / "new-project"
        )
        before = self.snapshot(existing.root)
        (self.root / "new-project-2").mkdir()
        self.engine.open_project(
            [], "code", "", settings={"temperature": 0.4}
        )
        self.assertEqual(self.events[-1]["event"], "opened")
        first = self.engine.session.project
        self.assertEqual(first.root, self.root / "new-project-3")
        self.assertEqual(first.profile.temperature, 0.4)
        self.assertEqual(list(first.source.iterdir()), [])
        self.engine.open_project([], "code", "")
        self.assertEqual(self.events[-1]["event"], "opened")
        self.assertEqual(
            self.engine.session.project.root,
            self.root / "new-project-4",
        )
        self.assertEqual(self.snapshot(existing.root), before)

    def test_no_trace_or_conversation_is_created_before_core_commit(self):
        """Late GUI state cannot make an empty destination appear occupied."""
        target = self.root / "empty"
        target.mkdir()
        original = kilin.prepare_sources_workspace

        def create(*args, **kwargs):
            self.assertEqual(list(target.iterdir()), [])
            return original(*args, **kwargs)

        with patch("kilin.gui.engine.prepare_sources_workspace",
                   side_effect=create):
            self.engine.open_project([], "literature", str(target))
        self.assertEqual(self.events[-1]["event"], "opened")
        self.assertTrue((target / "profile.ini").is_file())
        self.assertEqual(list((target / "sources").iterdir()), [])
        self.assertTrue(self.engine.active_conversation)


if __name__ == "__main__":
    unittest.main()
