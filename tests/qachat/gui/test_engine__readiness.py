"""Verify GUI model readiness and user-owned substitution settings.

Use fake installed-model catalog entries, without loading model weights or Qt.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin.gui.engine import ChatEngine, _installed_replacements
from kilin.gui.preferences import required_models


class TestEngineReadiness(unittest.TestCase):
    """Keep installed-model suggestions independent of bundled defaults."""

    def test_missing_models_have_role_compatible_replacements(self) -> None:
        """Never substitute a chat model for an embedding model."""

        roles = {
            "Qwen/Qwen3-4B-MLX-4bit": "chat",
            "mlx-community/Qwen3-32B-4bit": "chat",
            "Qwen/Qwen3-8B": "chat",
            "mlx-community/Qwen3-Embedding-4B-4bit": "embedding",
        }
        chosen = _installed_replacements(
            [
                "Qwen/Qwen3-8B-MLX-4bit",
                "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
            ],
            roles,
            "mlx",
        )
        self.assertEqual(
            chosen,
            {
                "Qwen/Qwen3-8B-MLX-4bit": "Qwen/Qwen3-4B-MLX-4bit",
                "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ": (
                    "mlx-community/Qwen3-Embedding-4B-4bit"
                ),
            },
        )

    def test_readiness_can_use_installed_models_without_package_writes(
        self,
    ) -> None:
        """Save replacements in user preferences, not shipped defaults."""

        with TemporaryDirectory() as temporary:
            home = Path(temporary).resolve()
            preference_file = home / "preferences.json"
            events: list[dict[str, object]] = []
            engine = ChatEngine(home / "sandbox", events.append)
            missing_model = "Qwen/Qwen3-8B-MLX-4bit"
            replacement = "example/Qwen3-8B-MLX-4bit"
            with patch.dict(
                os.environ,
                {"KILIN_DEFAULTS_FILE": str(preference_file)},
            ):
                required = required_models("mlx")
                self.assertIn(missing_model, required)
                cached = [
                    {"model_id": value}
                    for value in required
                    if value != missing_model
                ]
                cached.append({"model_id": replacement})
                roles = {value["model_id"]: "chat" for value in cached}
                for value in required:
                    if "Embedding" in value:
                        roles[value] = "embedding"
                with (
                    patch(
                        "kilin.gui.engine.choose_backend",
                        return_value="mlx",
                    ),
                    patch(
                        "kilin.gui.engine.installed_backends",
                        return_value={"mlx": True},
                    ),
                    patch(
                        "kilin.gui.engine.available_models",
                        return_value=cached,
                    ),
                    patch(
                        "kilin.gui.engine.cached_model_roles",
                        return_value=roles,
                    ),
                ):
                    engine.readiness()
                    offer = events[-1]
                    self.assertEqual(offer["event"], "readiness")
                    self.assertEqual(offer["missing_models"], [missing_model])
                    self.assertTrue(offer["can_use_installed"])
                    engine.use_installed_models()
                    used = [
                        event for event in events
                        if event["event"] == "installed_models_used"
                    ]
                    self.assertEqual(len(used), 1)
                    self.assertEqual(
                        used[0]["replacements"],
                        {missing_model: replacement},
                    )
                    self.assertEqual(events[-1]["event"], "readiness")
                    self.assertTrue(events[-1]["ready"])
                    self.assertNotIn(missing_model, required_models("mlx"))
            self.assertTrue(preference_file.is_file())


if __name__ == "__main__":
    unittest.main()
