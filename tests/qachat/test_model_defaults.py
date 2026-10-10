"""Unit tests for QAChat's shared model defaults.

Tests verify the canonical model roles, default model identifiers, and
model identifier structure without loading or downloading any models.

"""

from __future__ import annotations

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
