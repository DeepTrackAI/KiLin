"""Check qachat_help: answers come from QAChat's own documentation."""

from __future__ import annotations

import unittest

from qachat.tools import ToolRequest
from qachat.tools.qachat_help import HelpSettings, documentation, run
from tests.qachat.tools.helpers import context


class TestQAChatHelp(unittest.TestCase):
    """The model sees documentation sections, never project sources."""

    def test_reindexing_question_gets_the_indexing_section(self) -> None:
        prompts = []

        def write(system, prompt, tokens):
            prompts.append(prompt)
            return "Only added or modified files are embedded [1]."

        built, events = context(write=write)
        question = "will adding a file re-index everything?"
        result = run(HelpSettings(), ToolRequest(question, question), built)
        self.assertTrue(result.streamed)
        self.assertIn("added or modified files", " ".join(prompts[0].split()))
        self.assertTrue(result.evidence)
        self.assertTrue(
            all(
                str(item["path"]).startswith("QAChat documentation/")
                for item in result.evidence
            )
        )
        self.assertEqual(events[-1][0], "done")

    def test_documentation_is_split_into_sections(self) -> None:
        sections = documentation()
        paths = {chunk.path.as_posix() for chunk in sections}
        self.assertIn("QAChat documentation/qachat/gui/README.md", paths)
        self.assertGreater(len(sections), 5)


if __name__ == "__main__":
    unittest.main()
