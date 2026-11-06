"""Verify project-specific failures when a saved project is reopened.

Reopening a project does not run the new-project model-readiness gate.
If the saved project cannot be activated, the engine reports whether the
previous session was released so that the GUI can show an accurate state.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kilin.gui import engine as gui_engine


class TestEngineResumeRecovery(unittest.TestCase):
    """Separate rejected saved profiles from failed project switches."""

    def setUp(self) -> None:
        """Create an engine without Qt or real model dependencies."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.events: list[dict[str, object]] = []
        self.engine = gui_engine.ChatEngine(self.root, self.events.append)
        self.old_session = SimpleNamespace(close=Mock())
        self.engine.session = self.old_session
        self.project = SimpleNamespace(root=self.root)

    def test_invalid_profile_preserves_current_session(self) -> None:
        """Fail before the old project closes, without losing its session."""

        with patch.object(
            gui_engine, "open_project", side_effect=ValueError("profile")
        ):
            self.engine.resume_project(str(self.root))
        self.assertIs(self.engine.session, self.old_session)
        self.old_session.close.assert_not_called()
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0]["event"], "error")
        self.assertIs(self.events[0]["resume_released"], False)
        self.assertIn("profile", self.events[0]["message"])
        self.engine.close()

    def test_missing_saved_model_releases_old_session(self) -> None:
        """A failing saved model leaves no stale active project."""

        self._resume_with_activation_error("saved model is unavailable")
        self.old_session.close.assert_called_once_with()
        self.assertIsNone(self.engine.session)
        self.assertIsNone(self.engine.conversation_store)
        self.assertIsNone(self.engine.active_conversation)
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0]["event"], "error")
        self.assertIs(self.events[0]["resume_released"], True)
        self.assertIn("saved model is unavailable",
                      self.events[0]["message"])

    def test_partial_activation_is_closed_after_failure(self) -> None:
        """Close any replacement session constructed before an error."""

        replacement = SimpleNamespace(close=Mock())
        self._resume_with_activation_error("bad embedding", replacement)
        self.old_session.close.assert_called_once_with()
        replacement.close.assert_called_once_with()
        self.assertIsNone(self.engine.session)
        self.assertIs(self.events[0]["resume_released"], True)

    def test_successful_resume_uses_the_saved_project_directly(
        self,
    ) -> None:
        """Do not run new-project readiness before saved project activation."""

        conversation = SimpleNamespace(conversation_id="saved")
        with patch.object(
            gui_engine, "open_project", return_value=self.project
        ) as open_saved, patch.object(
            self.engine, "readiness",
            side_effect=AssertionError("global setup must not run"),
        ), patch.object(
            self.engine, "_open_conversation_store",
            return_value=conversation,
        ), patch.object(
            self.engine, "_start_project_trace"
        ), patch.object(
            self.engine, "_activate_project"
        ) as activate, patch.object(
            self.engine, "_report_source_changes"
        ):
            self.engine.resume_project(str(self.root))
        open_saved.assert_called_once_with(str(self.root))
        activate.assert_called_once_with(self.project, conversation)
        self.old_session.close.assert_called_once_with()
        self.assertEqual(self.events, [])

    def _resume_with_activation_error(
        self, message: str, replacement: object | None = None
    ) -> None:
        """Fail activation after the prior session has been closed.

        Parameters
        ----------
        message: str
            Simulated model-load error.
        replacement: object | None, optional
            Partially constructed session to release, if any.

        """

        def fail(*_args: object) -> None:
            """Simulate a failed session activation."""

            self.engine.session = replacement
            raise RuntimeError(message)

        with patch.object(
            gui_engine, "open_project", return_value=self.project
        ), patch.object(
            self.engine, "_open_conversation_store",
            return_value=SimpleNamespace(conversation_id="chat"),
        ), patch.object(
            self.engine, "_start_project_trace"
        ), patch.object(
            self.engine, "_activate_project", side_effect=fail
        ):
            self.engine.resume_project(str(self.root))


if __name__ == "__main__":
    unittest.main()
