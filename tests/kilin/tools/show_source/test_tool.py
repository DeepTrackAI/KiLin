"""Test the show_source tool package without live model backends.

Check configuration ownership, public registry entries, settings validation,
and deterministic execution behavior.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.tools import show_source as tool

from tests.kilin.tools._contract_support import (
    assert_packaged_defaults,
    assert_registry_entry,
)


class TestShowSource(unittest.TestCase):
    """Verify independent configuration and offline functionality."""

    def test_registry_uses_package_entry_points(self) -> None:
        """Resolve parser and runner to the packaged implementation.

        """

        assert_registry_entry(
            self, "show_source", tool
        )

    def test_packaged_defaults_and_project_type_overrides(self) -> None:
        """Use the tool-local JSON when settings are not supplied.

        """

        assert_packaged_defaults(
            self, tool, tool.SourceSettings,
        )

    def test_invalid_project_type_settings(self) -> None:
        """Reject out-of-range, mistyped and unknown overrides.

        """

        invalid_settings = (
            {'max_passages': 0},
            {'context_sentences': 11},
            {"unknown": True},
        )
        for values in invalid_settings:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_choose_linked_evidence(self) -> None:
        """Prefer linked evidence over unlinked recent evidence.

        """

        from pathlib import Path
        from kilin.tools._contract import EarlierTurn
        recent = EarlierTurn(
            3, "recent", paths=(Path("a.md"),),
            texts=("a",), linked=False,
        )
        linked = EarlierTurn(
            2, "linked", paths=(Path("b.md"),),
            texts=("b",), linked=True,
        )
        self.assertEqual(tool.newest_listed([recent, linked]), linked)
        self.assertIsNone(tool.newest_listed([]))

    def test_explanation_prompt_includes_exact_text(self) -> None:
        """Preserve the cited passage and its source label.

        """

        system, prompt = tool.explain_prompt(
            "Explain this", [(1, "paper.md", ">>Important<<")]
        )
        self.assertIn("Use only the text shown", system)
        self.assertIn("paper.md", prompt)
        self.assertIn(">>Important<<", prompt)


if __name__ == "__main__":
    unittest.main()
