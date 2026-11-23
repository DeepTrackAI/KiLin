"""Test the batch_classify tool package without live model backends.

Check configuration ownership, public registry entries, settings validation,
and deterministic execution behavior.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.tools import batch_classify as tool

from tests.kilin.tools._contract_support import (
    assert_packaged_defaults,
    assert_registry_entry,
)


class TestBatchClassify(unittest.TestCase):
    """Verify independent configuration and offline functionality."""

    def test_registry_uses_package_entry_points(self) -> None:
        """Resolve parser and runner to the packaged implementation.

        """

        assert_registry_entry(
            self, "batch_classify", tool
        )

    def test_packaged_defaults_and_project_type_overrides(self) -> None:
        """Use the tool-local JSON when settings are not supplied.

        """

        assert_packaged_defaults(
            self, tool, tool.ClassifySettings,
        )

    def test_invalid_project_type_settings(self) -> None:
        """Reject out-of-range, mistyped and unknown overrides.

        """

        invalid_settings = (
            {'batch_tokens': 100},
            {'categories': ['only_one']},
            {"unknown": True},
        )
        for values in invalid_settings:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_category_validation(self) -> None:
        """Reject malformed categories and preserve valid label order.

        """

        self.assertEqual(
            tool.validated_categories(
                '{"categories": ["method", "result"]}'
            ),
            ("method", "result"),
        )
        self.assertEqual(tool.validated_categories("broken JSON"), ())
        self.assertEqual(
            tool.validated_categories('{"categories": ["only"]}'), ()
        )

    def test_label_validation(self) -> None:
        """Discard labels outside the batch or allowed categories.

        """

        reply = (
            '[{"id": 1, "category": "method", "reason": "yes"},'
            '{"id": 3, "category": "method"}]'
        )
        parsed = tool.parse_labels(reply, [1, 2], ["method", "result"])
        self.assertEqual(set(parsed), {1})
        self.assertEqual(parsed[1].category, "method")


if __name__ == "__main__":
    unittest.main()
