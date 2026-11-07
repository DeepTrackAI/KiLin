"""Verify GUI model readiness and user-owned substitution settings.

Use fake installed-model catalog entries, without loading model weights or Qt.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin.gui.engine import ChatEngine, _installed_replacements
from kilin.gui.preferences import (
    default_embedding_model,
    default_effort_levels,
    default_router_model,
    required_models,
    write_defaults,
)


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
                        "kilin.gui.engine.downloaded_models",
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


class TestPreferredBackendSetup(unittest.TestCase):
    """Keep explicit backend choices authoritative for every setup action."""

    def setUp(self) -> None:
        """Isolate persistent preferences without loading a model."""

        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        settings = patch.dict(
            os.environ,
            {"KILIN_DEFAULTS_FILE": str(self.root / "preferences.json")},
        )
        settings.start()
        self.addCleanup(settings.stop)
        self.events: list[dict[str, object]] = []
        self.engine = ChatEngine(self.root / "sandbox", self.events.append)
        self.addCleanup(self.engine.close)

    def _platform(self, system: str, machine: str) -> ExitStack:
        """Simulate a platform without importing native inference packages."""

        stack = ExitStack()
        stack.enter_context(patch(
            "kilin.runtime.platform.system", return_value=system,
        ))
        stack.enter_context(patch(
            "kilin.runtime.platform.machine", return_value=machine,
        ))
        return stack

    def test_readiness_respects_auto_and_explicit_backends(self) -> None:
        """An installed other backend must not override the preference."""

        cases = (
            ("auto", "Darwin", "arm64", "mlx"),
            ("auto", "Linux", "x86_64", "torch"),
            ("mlx", "Darwin", "arm64", "mlx"),
            ("torch", "Darwin", "arm64", "torch"),
            ("torch", "Linux", "x86_64", "torch"),
        )
        for preference, system, machine, selected in cases:
            with self.subTest(preference=preference, system=system):
                write_defaults({"backend": preference})
                cached = [
                    {"model_id": name} for name in required_models(selected)
                ]
                self.events.clear()
                with (
                    self._platform(system, machine),
                    patch(
                        "kilin.gui.engine.installed_backends",
                        return_value={"mlx": True, "torch": True},
                    ),
                    patch(
                        "kilin.gui.engine.downloaded_models",
                        return_value=cached,
                    ),
                ):
                    self.engine.readiness()
                event = self.events[-1]
                self.assertEqual(event["event"], "readiness")
                self.assertEqual(event["backend"], selected)
                self.assertEqual(
                    event["required_models"], required_models(selected)
                )
                self.assertEqual(event["missing_models"], [])
                self.assertTrue(event["ready"])

    def test_changed_preference_refreshes_same_engine(self) -> None:
        """Recheck current preferences for every independent operation."""

        with (
            self._platform("Darwin", "arm64"),
            patch(
                "kilin.gui.engine.installed_backends",
                return_value={"mlx": True, "torch": True},
            ),
            patch(
                "kilin.gui.engine.downloaded_models",
                return_value=[],
            ),
        ):
            write_defaults({"backend": "torch"})
            self.engine.readiness()
            self.assertEqual(self.events[-1]["backend"], "torch")
            self.assertEqual(
                self.events[-1]["missing_models"], required_models("torch")
            )
            write_defaults({"backend": "mlx"})
            self.engine.readiness()
            self.assertEqual(self.events[-1]["backend"], "mlx")
            self.assertEqual(
                self.events[-1]["missing_models"], required_models("mlx")
            )

    def test_explicit_torch_missing_runtime_never_falls_back(self) -> None:
        """An installed MLX runtime does not make PyTorch ready."""

        write_defaults({"backend": "torch"})
        cached = [{"model_id": name} for name in required_models("torch")]
        with (
            self._platform("Darwin", "arm64"),
            patch(
                "kilin.gui.engine.installed_backends",
                return_value={"mlx": True, "torch": False},
            ),
            patch(
                "kilin.gui.engine.downloaded_models",
                return_value=cached,
            ),
            patch.object(self.engine, "_download_with_progress") as download,
        ):
            self.engine.readiness()
            self.assertEqual(self.events[-1]["backend"], "torch")
            self.assertFalse(self.events[-1]["environment_ready"])
            self.assertFalse(self.events[-1]["ready"])
            self.engine.install_required_models()
            self.assertEqual(self.events[-1]["event"], "error")
            self.assertIn("TORCH runtime", self.events[-1]["message"])
            download.assert_not_called()

    def test_missing_both_runtimes_is_not_reported_ready(self) -> None:
        """A complete cache cannot replace unavailable backend packages."""

        for preference, system, machine, selected in (
            ("auto", "Linux", "x86_64", "torch"),
            ("mlx", "Darwin", "arm64", "mlx"),
        ):
            with self.subTest(preference=preference):
                write_defaults({"backend": preference})
                cached = [
                    {"model_id": name} for name in required_models(selected)
                ]
                with (
                    self._platform(system, machine),
                    patch(
                        "kilin.gui.engine.installed_backends",
                        return_value={"mlx": False, "torch": False},
                    ),
                    patch(
                        "kilin.gui.engine.downloaded_models",
                        return_value=cached,
                    ),
                ):
                    self.engine.readiness()
                event = self.events[-1]
                self.assertEqual(event["backend"], selected)
                self.assertFalse(event["ready"])
                self.assertFalse(event["environment_ready"])
                self.assertFalse(event["can_use_installed"])
                self.assertEqual(event["missing_models"], [])

    def test_explicit_unsupported_mlx_is_reported_not_switched(self) -> None:
        """An explicit MLX request remains unavailable off Apple Silicon."""

        write_defaults({"backend": "mlx"})
        cached = [{"model_id": name} for name in required_models("mlx")]
        with (
            self._platform("Linux", "x86_64"),
            patch(
                "kilin.gui.engine.installed_backends",
                return_value={"mlx": True, "torch": True},
            ),
            patch(
                "kilin.gui.engine.downloaded_models",
                return_value=cached,
            ),
            patch.object(self.engine, "_download_with_progress") as download,
        ):
            self.engine.readiness()
            event = self.events[-1]
            self.assertEqual(event["backend"], "mlx")
            self.assertFalse(event["environment_ready"])
            self.assertFalse(event["ready"])
            self.assertIn("Apple Silicon", event["unavailable_reason"])
            self.engine.install_required_models()
            self.assertEqual(self.events[-1]["event"], "error")
            self.assertIn("Apple Silicon", self.events[-1]["message"])
            download.assert_not_called()
            self.engine.use_installed_models()
            self.assertEqual(self.events[-1]["event"], "error")
            self.assertIn("Apple Silicon", self.events[-1]["message"])

    def test_substitution_only_changes_selected_backend(self) -> None:
        """Do not overwrite MLX defaults when substituting PyTorch models."""

        write_defaults({"backend": "torch"})
        selected = "Qwen/Qwen3-4B"
        alternative = "example/Qwen3-4B"
        cached = [
            {"model_id": model}
            for model in required_models("torch") if model != selected
        ] + [{"model_id": alternative}]
        original_mlx = default_effort_levels("mlx")
        with (
            self._platform("Darwin", "arm64"),
            patch(
                "kilin.gui.engine.downloaded_models",
                return_value=cached,
            ),
            patch(
                "kilin.gui.engine.cached_model_roles",
                return_value={alternative: "chat"},
            ),
            patch(
                "kilin.gui.engine.installed_backends",
                return_value={"mlx": True, "torch": True},
            ),
        ):
            self.engine.use_installed_models()
        event = next(
            event for event in self.events
            if event["event"] == "installed_models_used"
        )
        self.assertEqual(event["replacements"], {selected: alternative})
        self.assertEqual(default_router_model("torch"), alternative)
        self.assertIn(
            alternative,
            [level["model"] for level in default_effort_levels("torch")],
        )
        self.assertEqual(default_effort_levels("mlx"), original_mlx)
        self.assertEqual(self.events[-1]["backend"], "torch")

    def test_new_project_model_edits_use_preferred_backend(self) -> None:
        """Apply new-project model defaults without implicit auto fallback."""

        write_defaults({"backend": "torch"})
        original_mlx = default_embedding_model("mlx")
        for requested in (None, "auto"):
            with self.subTest(requested=requested):
                embedding = (
                    "example/Custom-Embed-1B" if requested is None
                    else "example/Second-Embed-1B"
                )
                request = {
                    "backend": requested,
                    "chat_model": "Qwen/Qwen3-4B",
                    "embedding_model": embedding,
                    "query_format": "qwen3",
                }
                cached = [
                    {"model_id": "Qwen/Qwen3-4B"},
                    {"model_id": embedding},
                ]
                with (
                    self._platform("Darwin", "arm64"),
                    patch(
                        "kilin.gui.engine.downloaded_models",
                        return_value=cached,
                    ),
                    patch(
                        "kilin.gui.engine.cached_model_roles",
                        return_value={},
                    ),
                    patch(
                        "kilin.gui.engine.thinking_switches",
                        return_value={},
                    ),
                ):
                    self.engine.apply_model_settings(request)
                self.assertEqual(self.events[-1]["event"], "models_applied")
                self.assertEqual(default_embedding_model("torch"), embedding)
                self.assertEqual(default_embedding_model("mlx"), original_mlx)

    def test_install_downloads_only_explicit_backend_models(self) -> None:
        """On Apple Silicon, PyTorch preference downloads PyTorch IDs."""

        write_defaults({"backend": "torch"})
        missing = required_models("torch")[:2]
        cached = [
            {"model_id": value}
            for value in required_models("torch") if value not in missing
        ]
        observed: list[str] = []

        def download(model_id: str, **_kwargs: object) -> float:
            """Simulate completion without accessing the network."""

            observed.append(model_id)
            cached.append({"model_id": model_id})
            return 0.01

        with (
            self._platform("Darwin", "arm64"),
            patch(
                "kilin.gui.engine.installed_backends",
                return_value={"mlx": True, "torch": True},
            ),
            patch(
                "kilin.gui.engine.downloaded_models",
                side_effect=lambda: list(cached),
            ),
            patch.object(
                self.engine, "_download_with_progress", side_effect=download
            ),
        ):
            self.engine.install_required_models()
        self.assertEqual(observed, missing)
        started = next(
            event for event in self.events
            if event["event"] == "installation_started"
        )
        self.assertEqual(started["backend"], "torch")
        self.assertEqual(started["models"], missing)
        self.assertEqual(self.events[-1]["event"], "readiness")
        self.assertTrue(self.events[-1]["ready"])


if __name__ == "__main__":
    unittest.main()
