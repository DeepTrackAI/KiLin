"""Verify core-owned router configuration without GUI or loaded models.

Use temporary project settings to exercise precedence, legacy input,
validation, and atomic persistence independently of desktop preferences.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from configparser import ConfigParser
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import kilin
from kilin import model_settings
from kilin.gui import model_settings as gui_settings
from kilin.gui import preferences
from kilin.runtime import BACKEND_MLX, BACKEND_TORCH


class TestRouterSettings(unittest.TestCase):
    """Check the authoritative core router-selection contract."""

    def setUp(self) -> None:
        """Create a disposable project directory and profile."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profile = self.root / "profile.ini"
        self.profile.write_text(
            "[project]\ntype_id = code\n\n[models]\n"
            "chat_mlx = example/answer\n",
            encoding="utf-8",
        )

    def test_core_defaults_match_shipped_gui_defaults(self) -> None:
        """Keep initial router choices aligned across both interfaces."""

        packaged = preferences.built_in_defaults()["router_models"]
        for backend in (BACKEND_MLX, BACKEND_TORCH):
            with self.subTest(backend=backend):
                self.assertEqual(
                    model_settings.default_router_model(backend),
                    packaged[backend],
                )
                self.assertEqual(
                    model_settings.read_router_model(None, backend),
                    packaged[backend],
                )
                self.assertEqual(
                    model_settings.read_router_model(self.root, backend),
                    packaged[backend],
                )
        self.assertNotIn("[router]", self.profile.read_text())

    def test_saved_router_precedes_override_default_and_legacy(self) -> None:
        """Saved router IDs are authoritative for core and GUI users."""

        (self.root / "effort_levels.json").write_text(
            json.dumps({"router": {BACKEND_MLX: "example/legacy"}}),
            encoding="utf-8",
        )
        self.assertEqual(
            model_settings.read_router_model(self.root, BACKEND_MLX),
            "example/legacy",
        )
        model_settings.write_router_model(
            self.root, BACKEND_MLX, " example/saved "
        )
        for reader in (model_settings.read_router_model,
                       gui_settings.read_router_model,
                       kilin.read_router_model):
            self.assertEqual(
                reader(
                    self.root, BACKEND_MLX,
                    default_model="example/application",
                ),
                "example/saved",
            )
        self.assertEqual(
            model_settings.read_router_model(self.root, BACKEND_TORCH),
            model_settings.default_router_model(BACKEND_TORCH),
        )

    def test_gui_preference_initializes_missing_saved_router(self) -> None:
        """An initial preference does not override a saved model."""

        self.assertEqual(
            model_settings.read_router_model(
                self.root, BACKEND_TORCH,
                default_model="example/gui-preference",
            ),
            "example/gui-preference",
        )
        model_settings.write_router_model(
            self.root, BACKEND_TORCH, "example/project"
        )
        self.assertEqual(
            model_settings.read_router_model(
                self.root, BACKEND_TORCH,
                default_model="example/gui-preference",
            ),
            "example/project",
        )

    def test_atomic_write_preserves_unrelated_options(self) -> None:
        """Editing the router must not change other project options."""

        with patch(
            "kilin.model_settings.atomic_write_text",
            wraps=model_settings.atomic_write_text,
        ) as writer:
            kilin.write_router_model(
                self.root, BACKEND_MLX, "example/independent-router"
            )
        writer.assert_called_once()
        parser = ConfigParser(interpolation=None)
        parser.read(self.profile, encoding="utf-8")
        self.assertEqual(
            parser.get("models", "chat_mlx"), "example/answer"
        )
        self.assertEqual(
            parser.get("router", "mlx"), "example/independent-router"
        )

    def test_invalid_configuration_is_rejected_without_writes(self) -> None:
        """Reject malformed saved values and unsupported backends."""

        before = self.profile.read_bytes()
        for backend, model in (
            ("auto", "example/model"),
            ("invalid", "example/model"),
            (BACKEND_MLX, ""),
            (BACKEND_TORCH, "not a valid id !"),
        ):
            with self.subTest(backend=backend, model=model):
                with self.assertRaises(ValueError):
                    model_settings.write_router_model(
                        self.root, backend, model
                    )
                self.assertEqual(self.profile.read_bytes(), before)
        with self.assertRaises(ValueError):
            model_settings.read_router_model(self.root, "auto")
        self.profile.write_text(
            self.profile.read_text(encoding="utf-8")
            + "\n[router]\nmlx = \n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            model_settings.read_router_model(self.root, BACKEND_MLX)

    def test_legacy_configuration_does_not_modify_project(self) -> None:
        """Read legacy router settings without performing migration."""

        old = self.root / "effort_levels.json"
        old.write_text(
            json.dumps({"router": {BACKEND_TORCH: "example/old"}}),
            encoding="utf-8",
        )
        profile_before = self.profile.read_bytes()
        self.assertEqual(
            model_settings.read_router_model(self.root, BACKEND_TORCH),
            "example/old",
        )
        self.assertEqual(self.profile.read_bytes(), profile_before)
        self.assertTrue(old.exists())
        old.write_text("{invalid", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            model_settings.read_router_model(self.root, BACKEND_TORCH)


if __name__ == "__main__":
    unittest.main()
