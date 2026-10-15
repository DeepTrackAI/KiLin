"""Verify catalog model choices and cache-role classification.

These checks do not load models or start a Qt application.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest
from unittest.mock import patch

from qachat.gui.model_catalog import (
    MODEL_CHOICES,
    installed_backends,
    model_choices,
)
from qachat.project import default_project_profile


class TestModelCatalog(unittest.TestCase):
    """Suggested choices must agree with QAChat's project defaults."""

    @patch("qachat.gui.model_catalog.version")
    def test_backend_packages_are_checked_separately(
        self, get_version
    ) -> None:
        """Check each backend independently of the other package group."""
        from importlib.metadata import PackageNotFoundError

        def installed(package: str) -> str:
            """Return a synthetic backend package version."""
            if package == "torch":
                raise PackageNotFoundError(package)
            return "1.0"

        get_version.side_effect = installed

        self.assertEqual(installed_backends(), {"mlx": True, "torch": False})

    def test_default_models_have_distinct_roles(self) -> None:
        """Keep catalog role assignments aligned with core defaults."""
        profile = default_project_profile("code")
        offered = {
            (choice.backend, choice.role): choice for choice in MODEL_CHOICES
        }

        for backend in ("mlx", "torch"):
            self.assertEqual(
                offered[(backend, "chat")].model_id,
                profile.chat_model(backend),
            )
            self.assertEqual(
                offered[(backend, "embedding")].model_id,
                profile.embedding_model(backend),
            )
            self.assertEqual(
                offered[(backend, "embedding")].query_format,
                profile.query_format,
            )

    def test_unknown_cached_model_is_not_mislabeled(self) -> None:
        """Do not present unknown cache entries as suggested models."""
        cached = {MODEL_CHOICES[0].model_id, "someone/unknown-model"}

        choices = model_choices(cached)

        self.assertEqual(len(choices), len(MODEL_CHOICES))
        self.assertTrue(choices[0]["cached"])
        self.assertNotIn(
            "someone/unknown-model", {choice["model_id"] for choice in choices}
        )


if __name__ == "__main__":
    unittest.main()
