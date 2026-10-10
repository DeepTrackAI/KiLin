"""Tests for offering long lists as a Markdown file."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from qachat.tools import ToolRequest
from qachat.tools.scan_and_list import ScanSettings, run
from tests.qachat.tools.helpers import chunk, context


class TestLongLists(unittest.TestCase):
    """More than 20 hits: save them all as a file, or show 20."""

    def setUp(self) -> None:
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        lines = [f"Line {n} uses MRI." for n in range(1, 31)]
        self.chunks = [chunk("repo/a.md", lines)]

    def scan(self, answer):
        """Run a scan where the user picks ``answer``; return the questions."""

        asked = []

        def choose(question, options):
            asked.append((question, options))
            return answer

        built, _ = context(
            self.chunks,
            router=lambda system, prompt, tokens: (
                '{"terms": ["MRI"], "document": null}'
            ),
            root=self.root,
            choose=choose,
        )
        question = "list every mention of MRI"
        result = run(ScanSettings(), ToolRequest(question, question), built)
        return result, asked

    def test_saving_keeps_the_counts_in_the_chat(self) -> None:
        result, asked = self.scan(0)
        (question, options), = asked
        self.assertEqual(
            question,
            "Found 30 mentions of “MRI”. Save the full list as a Markdown "
            "file?",
        )
        self.assertIn("Save all 30", options[0])
        saved = self.root / result.summary["saved"]
        items = re.findall(r"^\d+\. ", saved.read_text(), flags=re.M)
        self.assertEqual(len(items), 30)
        self.assertNotIn("[1]", saved.read_text())  # No passage numbers.
        self.assertIn("Found **30 mentions**", result.answer)
        self.assertNotIn("**Where**", result.answer)
        self.assertIn("outputs/mentions_mri.md", result.answer)

    def test_showing_lists_the_first_twenty(self) -> None:
        result, _ = self.scan(1)
        self.assertNotIn("saved", result.summary)
        self.assertIn("Listed the first 20 of 30", result.answer)
        self.assertEqual(len(result.evidence), 20)


if __name__ == "__main__":
    unittest.main()
