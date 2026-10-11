"""Unit tests for QAChat's shared model defaults.

Tests verify the canonical model roles, default model identifiers, and
model identifier structure without loading or downloading any models.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
import unittest

from qachat import model_defaults


class TestModelRoles(unittest.TestCase):
    """Test the canonical model-purpose identifiers."""

    def test_model_role_choices(self) -> None:
        """Expose the two supported model roles in a stable order."""

        self.assertEqual(
            model_defaults.MODEL_ROLE_CHOICES,
            (
                model_defaults.MODEL_ROLE_CHAT,
                model_defaults.MODEL_ROLE_EMBEDDING,
            ),
        )
        self.assertEqual(len(set(model_defaults.MODEL_ROLE_CHOICES)), 2)


class TestDefaultModels(unittest.TestCase):
    """Test the shared default model identifiers."""

    def test_default_models_match_packaged_json(self) -> None:
        """Expose exactly the model IDs configured in the packaged JSON."""

        config_path = Path(model_defaults.__file__).with_name(
            "qachat_defaults.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_names = (
            "DEFAULT_CHAT_MODEL_MLX",
            "DEFAULT_CHAT_MODEL_TORCH",
            "DEFAULT_EMBEDDING_MODEL_MLX",
            "DEFAULT_EMBEDDING_MODEL_TORCH",
        )
        configured_models = config["DEFAULT_MODELS"]
        self.assertEqual(set(configured_models), set(expected_names))
        for name in expected_names:
            with self.subTest(name=name):
                self.assertEqual(
                    getattr(model_defaults, name), configured_models[name]
                )

    def test_default_model_identifiers(self) -> None:
        """Provide nonempty Hugging Face model identifiers."""

        defaults = (
            model_defaults.DEFAULT_CHAT_MODEL_MLX,
            model_defaults.DEFAULT_CHAT_MODEL_TORCH,
            model_defaults.DEFAULT_EMBEDDING_MODEL_MLX,
            model_defaults.DEFAULT_EMBEDDING_MODEL_TORCH,
        )

        for model_id in defaults:
            with self.subTest(model_id=model_id):
                self.assertIsInstance(model_id, str)
                self.assertTrue(model_id.strip())
                self.assertEqual(len(model_id.split("/")), 2)

if __name__ == "__main__":
    unittest.main()
