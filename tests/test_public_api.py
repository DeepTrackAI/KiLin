"""Unit tests for QAChat's public Python API.

Tests verify declared exports and selected package-level compatibility
imports without exercising the behavior of individual modules.

"""

from __future__ import annotations

import unittest

import qachat
from qachat import huggingface, model_defaults


class TestPublicAPI(unittest.TestCase):
    """Test declared exports and package-level public imports."""

    def test_package_exports_exist(self) -> None:
        """Ensure all names declared by the package are available."""

        for name in qachat.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(qachat, name))

    def test_module_exports_exist(self) -> None:
        """Ensure declared exports exist in each covered source module."""

        modules = (huggingface, model_defaults)
        for module in modules:
            for name in module.__all__:
                with self.subTest(module=module.__name__, name=name):
                    self.assertTrue(hasattr(module, name))

    def test_model_defaults_package_exports(self) -> None:
        """Preserve the intended package-level model default imports."""

        names = (
            "MODEL_ROLE_CHAT",
            "MODEL_ROLE_EMBEDDING",
            "MODEL_ROLE_CHOICES",
            "ModelRole",
            "DEFAULT_CHAT_MODEL_MLX",
            "DEFAULT_CHAT_MODEL_TORCH",
            "DEFAULT_EMBEDDING_MODEL_MLX",
            "DEFAULT_EMBEDDING_MODEL_TORCH",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(qachat, name),
                    getattr(model_defaults, name),
                )


if __name__ == "__main__":
    unittest.main()
