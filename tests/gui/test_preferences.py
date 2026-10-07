"""Check the desktop defaults file and installed-model choices."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qachat.gui.engine import ChatEngine, _installed_replacements
from qachat.gui.preferences import (
    DEFAULTS_FILE,
    built_in_defaults,
    default_effort_levels,
    default_search_depth,
    defaults_file,
    required_models,
    write_defaults,
)


class TestWriteDefaults(unittest.TestCase):
    """App-wide choices are saved in the one defaults file."""

    def setUp(self) -> None:
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "defaults.json"
        shutil.copyfile(DEFAULTS_FILE, self.path)

    def test_tests_never_use_the_real_file(self) -> None:
        self.assertIn("qachat-tests-", str(defaults_file()))
        self.assertIn("QACHAT_DEFAULTS_FILE", os.environ)

    def test_changes_are_saved_in_the_same_style(self) -> None:
        write_defaults({"appearance": "dark"}, self.path)
        write_defaults({"search_depth": [3, 3, 6, 6, 9]}, self.path)
        saved = built_in_defaults(self.path)
        self.assertEqual(saved["appearance"], "dark")
        self.assertEqual(saved["search_depth"], [3, 3, 6, 6, 9])
        text = self.path.read_text()
        # Short lists and levels stay on one line, as the file is kept.
        self.assertIn('"search_depth": [3, 3, 6, 6, 9]', text)
        self.assertIn('{"name": "Light", "model": ', text)
        # Other settings are untouched.
        original = json.loads(DEFAULTS_FILE.read_text())
        self.assertEqual(json.loads(text)["answers"], original["answers"])

    def test_required_models_follow_the_defaults(self) -> None:
        levels = [
            {**level, "model": "a/Qwen3-4B-MLX"}
            for level in default_effort_levels("mlx")
        ]
        write_defaults({"effort_levels": {"mlx": levels}})
        try:
            self.assertIn("a/Qwen3-4B-MLX", required_models("mlx"))
        finally:
            write_defaults(
                {"effort_levels": {"mlx": built_in_defaults(
                    DEFAULTS_FILE)["effort_levels"]["mlx"]}}
            )

    def test_invalid_values_are_rejected(self) -> None:
        before = self.path.read_text()
        with self.assertRaisesRegex(ValueError, "appearance"):
            write_defaults({"appearance": "pink"}, self.path)
        with self.assertRaisesRegex(ValueError, "Unknown default"):
            write_defaults({"colour": "red"}, self.path)
        self.assertEqual(self.path.read_text(), before)

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
            default_answer_settings,
            default_embedding_model,
            default_router_model,
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
        original = built_in_defaults(DEFAULTS_FILE)
        write_defaults(
            {
                "effort_levels": original["effort_levels"],
                "router_models": original["router_models"],
                "embedding_models": original["embedding_models"],
            }
        )


if __name__ == "__main__":
    unittest.main()
