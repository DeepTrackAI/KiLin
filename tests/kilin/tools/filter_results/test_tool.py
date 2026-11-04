"""Test the filter_results tool package without live model backends.

Check configuration ownership, public registry entries, settings validation,
and deterministic execution behavior.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.tools import filter_results as tool

from tests.kilin.tools._contract_support import (
    assert_packaged_defaults,
    assert_registry_entry,
)


class TestFilterResults(unittest.TestCase):
    """Verify independent configuration and offline functionality."""

    def test_registry_uses_package_entry_points(self) -> None:
        """Resolve parser and runner to the packaged implementation.

        """

        assert_registry_entry(
            self, "filter_results", tool
        )

    def test_packaged_defaults_and_project_type_overrides(self) -> None:
        """Use the tool-local JSON when settings are not supplied.

        """

        assert_packaged_defaults(
            self, tool, tool.FilterSettings,
        )

    def test_invalid_project_type_settings(self) -> None:
        """Reject out-of-range, mistyped and unknown overrides.

        """

        invalid_settings = (
            {'batch_tokens': 0},
            {'item_characters': 2},
            {"unknown": True},
        )
        for values in invalid_settings:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_no_earlier_list_returns_actionable_result(self) -> None:
        """Return a message rather than invoke a language model.

        """

        from unittest.mock import Mock
        from kilin.tools._contract import ToolRequest
        context = Mock()
        answer = tool.run(
            tool.FilterSettings(), ToolRequest("which of those?", "..."),
            context,
        )
        self.assertIn("earlier answer", answer.answer)
        self.assertFalse(answer.evidence)
        context.ask_model.assert_not_called()

    def test_filter_prompt_preserves_item_numbers(self) -> None:
        """Format only the supplied items for classification.

        """

        system, prompt = tool.filter_prompt(
            "keep methods", [(2, "experiment"), (5, "method")]
        )
        self.assertIn("item numbers only", system)
        self.assertIn("2. experiment", prompt)
        self.assertIn("5. method", prompt)


if __name__ == "__main__":
    unittest.main()
