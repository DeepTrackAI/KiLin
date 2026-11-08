"""Test bounded passage citation parsing across document tools.

Ranges, malformed markers and large numbers are deterministic and require
no model backend or source files.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.tools._shared.citations import inspect_citations


class TestCitationValidation(unittest.TestCase):
    """Check the shared citation parser's strict evidence boundaries."""

    def test_single_numbers_lists_and_ranges(self) -> None:
        """Accept mixed formats without duplicating passage numbers.

        """

        report = inspect_citations("Evidence [2], [1, 3-5], [4–6]", 6)
        self.assertEqual(report.valid, frozenset(range(1, 7)))
        self.assertEqual(report.invalid_numbers, ())
        self.assertEqual(report.invalid_groups, ())

    def test_mixed_valid_number_and_invalid_range(self) -> None:
        """Retain independent references without accepting invalid ranges.

        """

        report = inspect_citations("Evidence [1, 3-9] and [2]", 3)
        self.assertEqual(report.valid, frozenset({1, 2}))
        self.assertEqual(report.invalid_numbers, ())
        self.assertEqual(report.invalid_groups, ("3-9",))

    def test_missing_individual_numbers_remain_numeric(self) -> None:
        """Keep the established numeric invalid-citations contract.

        """

        report = inspect_citations("[0] and [4, 9] and [9]", 3)
        self.assertEqual(report.valid, frozenset())
        self.assertEqual(report.invalid_numbers, (0, 4, 9))
        self.assertEqual(report.invalid_groups, ())

    def test_oversized_reversed_and_zero_ranges(self) -> None:
        """Never partially admit an out-of-range or reversed span.

        """

        report = inspect_citations(
            "[1-1000000000] [9-2] [0-2] [1-101]", 500
        )
        self.assertEqual(report.valid, frozenset())
        self.assertEqual(
            report.invalid_groups,
            ("1-1000000000", "9-2", "0-2", "1-101"),
        )

    def test_malformed_parts_and_unrelated_brackets(self) -> None:
        """Report numeric citation syntax errors but ignore prose tags.

        """

        report = inspect_citations(
            "[ref] [1, nope] [2--3] [3;4] [4,]", 5
        )
        self.assertEqual(report.valid, frozenset({1, 4}))
        self.assertEqual(
            report.invalid_groups, ("nope", "2--3", "3;4", "(empty)")
        )

    def test_enormous_identifiers_have_bounded_diagnostics(self) -> None:
        """Avoid large integer conversion and diagnostic allocation.

        """

        huge = "9" * 12000
        report = inspect_citations(f"[{huge}] [1-{huge}]", 2)
        self.assertEqual(report.valid, frozenset())
        self.assertEqual(report.invalid_numbers, ())
        self.assertEqual(len(report.invalid_groups), 2)
        self.assertTrue(all(len(part) <= 83 for part in
                            report.invalid_groups))

    def test_invalid_passage_count_is_rejected(self) -> None:
        """Keep the parser independent of invalid caller configuration.

        """

        for count in (-1, True, 1.0):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    inspect_citations("[1]", count)


if __name__ == "__main__":
    unittest.main()
