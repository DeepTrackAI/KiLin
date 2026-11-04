"""Test GUI delegation of generation defaults to the KíLín core.

The GUI is tested without starting Qt or loading pretrained models.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kilin.gui import engine as gui_engine
from kilin.gui.preferences import built_in_defaults
from kilin.project import default_project_profile


class TestGUIGenerationSettings(unittest.TestCase):
    """Keep generation defaults and saved settings core-owned."""

    def test_gui_defaults_do_not_define_generation_policy(self) -> None:
        """Restrict the GUI defaults file to application preferences."""
        self.assertNotIn("answers", built_in_defaults())

    def test_existing_token_limit_is_not_substituted(self) -> None:
        """Respect an explicitly saved 256-token project setting."""
        profile = SimpleNamespace(
            temperature=0.3, thinking=False, max_new_tokens=256
        )
        chat = SimpleNamespace(configure_generation=Mock())
        engine = gui_engine.ChatEngine.__new__(gui_engine.ChatEngine)
        engine.session = SimpleNamespace(
            project=SimpleNamespace(profile=profile),
            chat_session=chat,
        )
        engine._use_project_answer_settings()
        chat.configure_generation.assert_called_once_with(
            {
                "temperature": 0.3,
                "thinking": False,
                "max_new_tokens": 256,
            }
        )

    def test_new_project_settings_use_core_generation_defaults(self) -> None:
        """Use the core profile values when no workspace exists yet."""
        events: list[dict[str, object]] = []
        engine = gui_engine.ChatEngine.__new__(gui_engine.ChatEngine)
        engine._send = lambda event, **payload: events.append(
            {"event": event, **payload}
        )
        engine._report_error = Mock()
        with patch.object(
            gui_engine, "downloaded_models", return_value=[]
        ), patch.object(
            gui_engine, "installed_backends", return_value=[]
        ), patch.object(
            gui_engine, "choose_backend", return_value="mlx"
        ):
            engine.settings("code", "")
        engine._report_error.assert_not_called()
        self.assertEqual(len(events), 1)
        profile = events[0]["profile"]
        core = default_project_profile("code")
        self.assertEqual(profile["thinking"], core.thinking)
        self.assertEqual(profile["temperature"], core.temperature)
        self.assertEqual(
            profile["max_new_tokens"], core.max_new_tokens
        )
        self.assertTrue(core.thinking)
        self.assertEqual(core.max_new_tokens, 8192)
