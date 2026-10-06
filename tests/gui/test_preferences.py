"""Check the app-wide desktop preferences and installed-model choices."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qachat.gui.engine import ChatEngine, _installed_replacements
from qachat.gui.model_settings import default_effort_levels
from qachat.gui.preferences import (
    preferences_file,
    preferred_appearance,
    preferred_effort_levels,
    preferred_search_depth,
    read_preferences,
    required_models,
    write_preferences,
)


class TestPreferences(unittest.TestCase):
    """Only customized choices are saved; defaults stay in code."""

    def setUp(self) -> None:
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "preferences.json"

    def test_tests_never_use_the_real_file(self) -> None:
        self.assertIn("qachat-tests-", str(preferences_file()))
        self.assertIn("QACHAT_PREFERENCES_FILE", os.environ)

    def test_only_customized_values_are_written(self) -> None:
        self.assertEqual(read_preferences(self.path), {})
        write_preferences({"appearance": "dark"}, self.path)
        write_preferences({"search_depth": [3, 3, 6, 6, 9]}, self.path)
        self.assertEqual(preferred_appearance(self.path), "dark")
        self.assertEqual(preferred_search_depth(self.path), [3, 3, 6, 6, 9])
        write_preferences(
            {"appearance": "system", "search_depth": [5, 5, 10, 10, 15]},
            self.path,
        )
        self.assertFalse(self.path.exists())

    def test_required_models_follow_the_preferences(self) -> None:
        levels = default_effort_levels("mlx")
        defaults = required_models("mlx", self.path)
        self.assertIn(levels[0]["model"], defaults)
        levels = [{**level, "model": "a/Qwen3-4B-MLX"} for level in levels]
        write_preferences({"effort_levels": {"mlx": levels}}, self.path)
        self.assertEqual(
            preferred_effort_levels("mlx", self.path)[0]["model"],
            "a/Qwen3-4B-MLX",
        )
        self.assertIn("a/Qwen3-4B-MLX", required_models("mlx", self.path))
        self.assertNotIn(
            default_effort_levels("mlx")[0]["model"],
            required_models("mlx", self.path),
        )

    def test_invalid_values_are_rejected_and_ignored(self) -> None:
        with self.assertRaisesRegex(ValueError, "appearance"):
            write_preferences({"appearance": "pink"}, self.path)
        self.path.write_text('{"search_depth": [0], "appearance": "dark"}')
        self.assertEqual(read_preferences(self.path), {"appearance": "dark"})

    def test_closest_installed_model_replaces_a_missing_one(self) -> None:
        roles = {
            "Qwen/Qwen3-4B-MLX-4bit": "chat",
            "mlx-community/Qwen3-32B-4bit": "chat",
            "Qwen/Qwen3-8B": "chat",  # PyTorch, not MLX.
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


class TestDefaults(unittest.TestCase):
    """Every desktop default comes from qachat/gui/defaults.json."""

    def test_defaults_file_is_the_single_source(self) -> None:
        import json

        from qachat.gui.preferences import (
            DEFAULTS_FILE,
            default_answer_settings,
            default_embedding_model,
            default_router_model,
            default_search_depth,
        )

        data = json.loads(DEFAULTS_FILE.read_text(encoding="utf-8"))
        self.assertEqual(
            default_effort_levels("mlx"), data["effort_levels"]["mlx"]
        )
        self.assertEqual(default_search_depth(), data["search_depth"])
        self.assertEqual(
            default_router_model("torch"), data["router_models"]["torch"]
        )
        self.assertEqual(
            default_embedding_model("mlx"), data["embedding_models"]["mlx"]
        )
        self.assertEqual(
            default_answer_settings()["max_new_tokens"],
            data["answers"]["max_new_tokens"],
        )

    def test_mistakes_in_the_defaults_file_name_the_setting(self) -> None:
        import json

        from qachat.gui.preferences import DEFAULTS_FILE, built_in_defaults

        data = json.loads(DEFAULTS_FILE.read_text(encoding="utf-8"))
        with TemporaryDirectory() as folder:
            broken = dict(data, search_depth=[5, 5, 0, 10, 15])
            path = Path(folder) / "broken.json"
            path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "'search_depth'"):
                built_in_defaults(path)
            missing = {k: v for k, v in data.items() if k != "answers"}
            path = Path(folder) / "missing.json"
            path.write_text(json.dumps(missing))
            with self.assertRaisesRegex(ValueError, "'answers'"):
                built_in_defaults(path)


class TestUseInstalledModels(unittest.TestCase):
    """The startup card can switch to installed models."""

    def test_readiness_offers_and_applies_installed_models(self) -> None:
        events: list[dict[str, object]] = []
        engine = ChatEngine(Path("/sandbox"), events.append)
        required = required_models("mlx")
        cached = [
            {"model_id": model_id, "path": ""}
            for model_id in required
            if "8B" not in model_id
        ] + [{"model_id": "Qwen/Qwen3-4B-MLX-4bit", "path": ""}]
        roles = {item["model_id"]: "chat" for item in cached}
        roles.update(
            {m: "embedding" for m in required if "Embedding" in m}
        )
        with (
            patch("qachat.gui.engine.choose_backend", return_value="mlx"),
            patch(
                "qachat.gui.engine.installed_backends",
                return_value={"mlx": True},
            ),
            patch("qachat.gui.engine.available_models", return_value=cached),
            patch(
                "qachat.gui.engine.cached_model_roles", return_value=roles
            ),
        ):
            engine.readiness()
            offer = events[-1]
            self.assertEqual(offer["event"], "readiness")
            self.assertFalse(offer["ready"])
            self.assertTrue(offer["can_use_installed"])
            engine.use_installed_models()
        used = [e for e in events if e["event"] == "installed_models_used"]
        # 14B (1.75x) is closer to the missing 8B than 4B (2x).
        self.assertEqual(
            set(used[0]["replacements"].values()),
            {"mlx-community/Qwen3-14B-4bit"},
        )
        self.assertTrue(events[-1]["ready"])
        self.assertNotIn("Qwen/Qwen3-8B-MLX-4bit", required_models("mlx"))
        write_preferences(
            {
                "effort_levels": {"mlx": default_effort_levels("mlx")},
                "router_models": {"mlx": "Qwen/Qwen3-8B-MLX-4bit"},
            }
        )


if __name__ == "__main__":
    unittest.main()
