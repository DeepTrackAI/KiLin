"""Test the find_fact tool package without live model backends.

Check configuration ownership, public registry entries, settings validation,
and deterministic execution behavior.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.tools import find_fact as tool

from tests.kilin.tools._contract_support import (
    assert_packaged_defaults,
    assert_registry_entry,
)


class TestFindFact(unittest.TestCase):
    """Verify independent configuration and offline functionality."""

    def test_registry_uses_package_entry_points(self) -> None:
        """Resolve parser and runner to the packaged implementation.

        """

        assert_registry_entry(
            self, "find_fact", tool
        )

    def test_packaged_defaults_and_project_type_overrides(self) -> None:
        """Use the tool-local JSON when settings are not supplied.

        """

        assert_packaged_defaults(
            self, tool, tool.FactSettings,
        )

    def test_invalid_project_type_settings(self) -> None:
        """Reject out-of-range, mistyped and unknown overrides.

        """

        invalid_settings = (
            {'part_tokens': 15},
            {'max_model_calls': 0},
            {"unknown": True},
        )
        for values in invalid_settings:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_fact_prompts_distinguish_quick_and_full(self) -> None:
        """Distinguish ranked passages from complete-document notes.

        """

        _, full = tool.fact_prompt(
            "Which parameter?", [("paper.md", ["[3] value 12"])],
            None,
        )
        _, quick = tool.fact_prompt(
            "Which parameter?", [("paper.md", ["[3] value 12"])],
            "Be brief", quick=True,
        )
        self.assertIn("Notes by document", full)
        self.assertIn("Passages by document", quick)
        self.assertIn("Be brief", quick)


if __name__ == "__main__":
    unittest.main()
