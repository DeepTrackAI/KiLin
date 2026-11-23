"""Verify core-owned source-card model and inference settings.

Exercise default independence, project persistence, invalid input rejection,
and deterministic settings fingerprints without initializing an inference
backend or importing the optional desktop interface.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import replace
from math import inf, nan
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin import model_settings
from kilin.model_defaults import (
    DEFAULT_SOURCE_CARD_MODEL_MLX,
    DEFAULT_SOURCE_CARD_MODEL_TORCH,
)
from kilin.runtime import BACKEND_MLX, BACKEND_TORCH


class TestSourceCardModelSettings(unittest.TestCase):

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.file = self.root / "profile.ini"
        self.file.write_text(
            "[project]\ntype_id = literature\n\n"
            "[router]\nmlx = example/router\n\n"
            "[generation]\ntemperature = 0.5\n",
            encoding="utf-8",
        )

    def test_packaged_defaults_do_not_load_or_write(self):
        before = self.file.read_bytes()
        for backend, expected in (
            (BACKEND_MLX, DEFAULT_SOURCE_CARD_MODEL_MLX),
            (BACKEND_TORCH, DEFAULT_SOURCE_CARD_MODEL_TORCH),
        ):
            with self.subTest(backend=backend):
                settings = model_settings.read_source_card_generation_settings(
                    self.root, backend
                )
                self.assertEqual(settings.model_id, expected)
                self.assertFalse(settings.thinking)
                self.assertEqual(settings.temperature, 0.0)
                self.assertEqual(settings.max_input_tokens, 5000)
                self.assertEqual(settings.max_new_tokens, 1200)
                self.assertEqual(
                    settings,
                    model_settings.default_source_card_generation_settings(
                        backend
                    ),
                )
        self.assertEqual(self.file.read_bytes(), before)
        self.assertEqual(
            model_settings.default_source_card_generation_settings(
                BACKEND_MLX
            ).model_id,
            "Qwen/Qwen3-8B-MLX-4bit",
        )
        self.assertEqual(
            model_settings.default_source_card_generation_settings(
                BACKEND_TORCH
            ).model_id,
            "Qwen/Qwen3-8B",
        )

    def test_core_settings_are_independent_of_router_and_answering(self):
        settings = model_settings.read_source_card_generation_settings(
            self.root, BACKEND_MLX
        )
        self.assertNotEqual(
            settings.model_id,
            model_settings.read_router_model(self.root, BACKEND_MLX),
        )
        selected = replace(
            settings, model_id=" example/card-model ", thinking=True,
            temperature=0.2, max_input_tokens=4200, max_new_tokens=700,
        )
        model_settings.write_source_card_generation_settings(
            self.root, BACKEND_MLX, selected
        )
        saved = model_settings.read_source_card_generation_settings(
            self.root, BACKEND_MLX
        )
        self.assertEqual(saved.model_id, "example/card-model")
        self.assertEqual(saved.temperature, 0.2)
        self.assertTrue(saved.thinking)
        self.assertEqual(saved.max_input_tokens, 4200)
        self.assertEqual(saved.max_new_tokens, 700)
        self.assertEqual(
            model_settings.read_router_model(self.root, BACKEND_MLX),
            "example/router",
        )
        self.assertEqual(
            model_settings.read_source_card_generation_settings(
                self.root, BACKEND_TORCH
            ).model_id,
            DEFAULT_SOURCE_CARD_MODEL_TORCH,
        )
        parsed = ConfigParser(interpolation=None)
        parsed.read(self.file, encoding="utf-8")
        self.assertEqual(parsed.get("generation", "temperature"), "0.5")
        self.assertEqual(parsed.get("project", "type_id"), "literature")

    def test_writing_second_backend_preserves_first(self):
        mlx = replace(
            model_settings.default_source_card_generation_settings(
                BACKEND_MLX
            ), model_id="example/mlx",
        )
        model_settings.write_source_card_generation_settings(
            self.root, BACKEND_MLX, mlx
        )
        torch = replace(
            model_settings.default_source_card_generation_settings(
                BACKEND_TORCH
            ), model_id="example/torch",
        )
        model_settings.write_source_card_generation_settings(
            self.root, BACKEND_TORCH, torch
        )
        self.assertEqual(
            model_settings.read_source_card_generation_settings(
                self.root, BACKEND_MLX
            ).model_id,
            "example/mlx",
        )
        self.assertEqual(
            model_settings.read_source_card_generation_settings(
                self.root, BACKEND_TORCH
            ).model_id,
            "example/torch",
        )

    def test_invalid_parameters_cannot_change_profile(self):
        before = self.file.read_bytes()
        valid = model_settings.default_source_card_generation_settings(
            BACKEND_TORCH
        )
        invalid = (
            {"model_id": "not valid !"},
            {"thinking": 1},
            {"temperature": True},
            {"temperature": -0.01},
            {"temperature": nan},
            {"temperature": inf},
            {"temperature": 10 ** 400},
            {"max_input_tokens": True},
            {"max_input_tokens": 0},
            {"max_new_tokens": -2},
            {"max_new_tokens": 200001},
            {"max_new_tokens": 1.5},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(valid, **changes)
        for backend in ("auto", "invalid"):
            with self.assertRaises(ValueError):
                model_settings.write_source_card_generation_settings(
                    self.root, backend, valid
                )
        with self.assertRaises(ValueError):
            model_settings.write_source_card_generation_settings(
                self.root, BACKEND_TORCH, None
            )
        self.assertEqual(self.file.read_bytes(), before)

    def test_invalid_saved_settings_are_rejected(self):
        original = self.file.read_text(encoding="utf-8")
        for name, value in (
            ("thinking", "possibly"),
            ("temperature", "nan"),
            ("temperature", "-1"),
            ("max_input_tokens", "1.5"),
            ("max_new_tokens", "0"),
            ("mlx", "bad id !"),
        ):
            with self.subTest(name=name, value=value):
                self.file.write_text(
                    original
                    + f"\n[source_card_generation]\n{name} = {value}\n",
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    model_settings.read_source_card_generation_settings(
                        self.root, BACKEND_MLX
                    )
        self.file.write_text(original, encoding="utf-8")

    def test_atomic_persistence_and_failed_write(self):
        settings = model_settings.default_source_card_generation_settings(
            BACKEND_MLX
        )
        original = self.file.read_bytes()
        with patch(
            "kilin.model_settings.atomic_write_text",
            side_effect=OSError("refused"),
        ):
            with self.assertRaises(OSError):
                model_settings.write_source_card_generation_settings(
                    self.root, BACKEND_MLX, settings
                )
        self.assertEqual(self.file.read_bytes(), original)
        with patch(
            "kilin.model_settings.atomic_write_text",
            wraps=model_settings.atomic_write_text,
        ) as writer:
            model_settings.write_source_card_generation_settings(
                self.root, BACKEND_MLX, settings
            )
        writer.assert_called_once()

    def test_fingerprint_ignores_model_id_not_parameters(self):
        original = model_settings.default_source_card_generation_settings(
            BACKEND_TORCH
        )
        changed_model = replace(original, model_id="example/new-model")
        changed_tokens = replace(original, max_new_tokens=1234)
        self.assertEqual(
            original.fingerprint(), changed_model.fingerprint()
        )
        self.assertNotEqual(
            original.fingerprint(), changed_tokens.fingerprint()
        )
        self.assertEqual(len(original.fingerprint()), 64)


if __name__ == "__main__":
    unittest.main()
