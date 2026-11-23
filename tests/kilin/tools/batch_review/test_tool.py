"""Test the batch_review tool package without live model backends.

Check configuration ownership, public registry entries, settings validation,
and deterministic execution behavior.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.tools import batch_review as tool

from tests.kilin.tools._contract_support import (
    assert_packaged_defaults,
    assert_registry_entry,
)


class TestBatchReview(unittest.TestCase):
    """Verify independent configuration and offline functionality."""

    def test_registry_uses_package_entry_points(self) -> None:
        """Resolve parser and runner to the packaged implementation.

        """

        assert_registry_entry(
            self, "batch_review", tool
        )

    def test_packaged_defaults_and_project_type_overrides(self) -> None:
        """Use the tool-local JSON when settings are not supplied.

        """

        assert_packaged_defaults(
            self, tool, tool.ReviewSettings,
        )

    def test_invalid_project_type_settings(self) -> None:
        """Reject out-of-range, mistyped and unknown overrides.

        """

        invalid_settings = (
            {'part_tokens': 20},
            {'confirmation': 'invalid'},
            {"unknown": True},
        )
        for values in invalid_settings:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_answer_prompt_requires_citations(self) -> None:
        """Build a grounded answer prompt without calling any model.

        """

        system, question = tool.answer_prompt(
            "What happened?", "paper.pdf", [(1, ["[4] evidence"])],
            1, "Short answer",
        )
        self.assertIn("cite passage numbers", system)
        self.assertIn("[4] evidence", question)
        self.assertIn("Short answer", question)


if __name__ == "__main__":
    unittest.main()
