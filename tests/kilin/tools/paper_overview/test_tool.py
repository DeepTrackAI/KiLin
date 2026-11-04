"""Test the paper_overview tool package without live model backends.

Check configuration ownership, public registry entries, settings validation,
and deterministic execution behavior.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.tools import paper_overview as tool

from tests.kilin.tools._contract_support import (
    assert_packaged_defaults,
    assert_registry_entry,
)


class TestPaperOverview(unittest.TestCase):
    """Verify independent configuration and offline functionality."""

    def test_registry_uses_package_entry_points(self) -> None:
        """Resolve parser and runner to the packaged implementation.

        """

        assert_registry_entry(
            self, "paper_overview", tool
        )

    def test_packaged_defaults_and_project_type_overrides(self) -> None:
        """Use the tool-local JSON when settings are not supplied.

        """

        assert_packaged_defaults(
            self, tool, tool.OverviewSettings,
        )

    def test_invalid_project_type_settings(self) -> None:
        """Reject out-of-range, mistyped and unknown overrides.

        """

        invalid_settings = (
            {'max_tokens': 0},
            {'answer_tokens': 30},
            {"unknown": True},
        )
        for values in invalid_settings:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_overview_prompt_preserves_evidence_and_guidance(self) -> None:
        """Include numbered passages and requested length constraints.

        """

        from tests.kilin.tools._shared._support import chunk
        _, prompt = tool.overview_prompt(
            "Summarize findings", "paper.pdf",
            [(3, chunk("a", "A result of interest."))],
            "Two sentences",
        )
        self.assertIn("[3] A result", prompt)
        self.assertIn("Two sentences", prompt)


if __name__ == "__main__":
    unittest.main()
