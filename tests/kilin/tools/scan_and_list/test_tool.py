"""Test the scan_and_list tool package without live model backends.

Check configuration ownership, public registry entries, settings validation,
and deterministic execution behavior.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.tools import scan_and_list as tool

from tests.kilin.tools._contract_support import (
    assert_packaged_defaults,
    assert_registry_entry,
)


class TestScanAndList(unittest.TestCase):
    """Verify independent configuration and offline functionality."""

    def test_registry_uses_package_entry_points(self) -> None:
        """Resolve parser and runner to the packaged implementation.

        """

        assert_registry_entry(
            self, "scan_and_list", tool
        )

    def test_packaged_defaults_and_project_type_overrides(self) -> None:
        """Use the tool-local JSON when settings are not supplied.

        """

        assert_packaged_defaults(
            self, tool, tool.ScanSettings,
        )

    def test_invalid_project_type_settings(self) -> None:
        """Reject out-of-range, mistyped and unknown overrides.

        """

        invalid_settings = (
            {'context_lines': -1},
            {'max_listed': 0},
            {"unknown": True},
        )
        for values in invalid_settings:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_overlap_and_multiline_mentions(self) -> None:
        """Count one occurrence across overlapping indexed chunks.

        """

        from tests.kilin.tools._shared._support import chunk
        first = chunk("a", "Before XYZ\n2 there", line=1)
        overlap = chunk("b", "2 there\nthen XYZ 2", line=2)
        scan = tool.scan_mentions([first, overlap], ["XYZ 2"])
        self.assertEqual(len(scan.mentions), 2)
        self.assertEqual(scan.term_counts(), {"XYZ 2": 2})
        self.assertEqual(scan.mentions[0].line, 1)
        self.assertEqual(scan.mentions[1].line, 3)

    def test_snippets_and_source_coverage(self) -> None:
        """Keep concrete source locations in deterministic evidence.

        """

        from tests.kilin.tools._shared._support import chunk
        scan = tool.scan_mentions(
            [chunk("a", "ABC and other evidence")], ["ABC"]
        )
        self.assertEqual(scan.counts(), {scan.files_scanned[0]: 1})
        self.assertIn("**ABC**", tool.mention_report(scan, limit=1))
        self.assertEqual(tool.mention_evidence(scan, 1)[0]["citation"], 1)


if __name__ == "__main__":
    unittest.main()
