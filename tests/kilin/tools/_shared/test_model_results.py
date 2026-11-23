"""Test explicit negative selections and rejection of malformed results.

These checks verify the parsing contract, not the model's relevance
judgement. They require neither model weights nor indexed project files.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin._model_completion import IncompleteModelResponse
from kilin.tools._shared.model_results import selected_numbers


class TestSelectedNumbers(unittest.TestCase):
    """Only an explicit NONE is a valid zero-match result."""

    def test_valid_numbers_are_sorted_and_bounded(self) -> None:
        """Preserve deduplication and ignoring extra out-of-batch numbers."""

        self.assertEqual(selected_numbers("3, 1, 3, 99", {1, 2, 3}), [1, 3])
        self.assertEqual(selected_numbers("1\n2", {1, 2, 3}), [1, 2])
        self.assertEqual(selected_numbers(" NONE ", {1, 2, 3}), [])

    def test_empty_or_malformed_is_not_none(self) -> None:
        """Reject empty, prose, unclosed, and wholly out-of-range replies."""

        for reply in ("", "  ", "I need to think", "1,", "99", "1 or 2"):
            with self.subTest(reply=reply):
                with self.assertRaises(IncompleteModelResponse) as failure:
                    selected_numbers(reply, {1, 2})
                self.assertEqual(failure.exception.reason, "invalid_output")


if __name__ == "__main__":
    unittest.main()
