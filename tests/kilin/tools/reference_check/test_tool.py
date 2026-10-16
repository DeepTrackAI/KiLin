"""Test the reference_check tool package without live model backends.

Check configuration ownership, the registry entry and intent, settings
validation, and a full model-free run on a small paper and project.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kilin.project_type import project_type
from kilin.tools import reference_check as tool
from kilin.tools._contract import ToolRequest
from kilin.tools._registry import TOOL_REFERENCE_CHECK, TOOL_REGISTRY
from kilin.tools.reference_check import _configuration as config
from tests.kilin.tools._shared._support import chunk

PAPER = """Introduction
Brain networks can be analysed with graph theory [1]. Software such as
BRAPH implements these measures [2, 3]. Deep learning helps too [4].
Methods
We followed the protocol of [2].
References
1. Sporns, O. The human connectome: a structural description. PLoS
Comput. Biol. 1, e42 (2005).
2. Mijalkov, M. et al. BRAPH: A graph theory software for the analysis of
brain connectivity. PLoS ONE 12, e0178798 (2017).
3. Chang, Y.-W. et al. BRAPH 2: a flexible, open-source framework for
network analyses in neurosciences. bioRxiv (2025).
4. LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. Nature 521,
436–444 (2015).
5. Never, C. A reference the text does not cite at all. Cell 1, 2 (2020).
"""
BRAPH = """BRAPH: A graph theory software for the analysis of brain
connectivity
Mite Mijalkov, Ehsan Kakaei, Joana B. Pereira
Abstract. Graph theory ...
"""
# Another version of the cited preprint: same title and server, another
# first author, so it is only probable.
BRAPH2 = """bioRxiv preprint
BRAPH 2: A flexible, open-source framework for network analyses
in neurosciences
Iok-ui Tiunn, Blanca Zufiria-Gerboles
"""


class TestReferenceCheckPackage(unittest.TestCase):
    """Configuration, registry and intent."""

    def test_registry_uses_package_entry_points(self) -> None:
        entry = TOOL_REGISTRY[TOOL_REFERENCE_CHECK]
        self.assertIs(entry.parse_settings, tool.parse_settings)
        self.assertIs(entry.run, tool.run)
        self.assertIsNone(entry.workspace_resources)
        self.assertFalse(entry.follows_up)

    def test_packaged_defaults_are_documented(self) -> None:
        package = Path(tool.__file__).parent
        data = json.loads((package / "defaults.json").read_text())
        readme = (package / "README.md").read_text()
        for section, values in data.items():
            with self.subTest(section=section):
                self.assertIn(f"`{section}`", readme)
            for key in values:
                with self.subTest(key=f"{section}.{key}"):
                    self.assertIn(f"`{key}`", readme)
        defaults = tool.ReferenceCheckSettings()
        self.assertEqual(defaults, tool.parse_settings({}, "example"))
        self.assertEqual(
            {f.name for f in fields(defaults)},
            {name.lower() for name in data["SETTINGS"]},
        )
        self.assertEqual(
            config.REFERENCE_LAYOUT.max_characters,
            data["REFERENCES"]["REFERENCE_CHARACTERS"],
        )
        self.assertEqual(
            config.MATCH_RULES.common_words,
            frozenset(data["MATCHING"]["COMMON_WORDS"]),
        )

    def test_invalid_project_type_settings(self) -> None:
        for values in ({"max_listed": 0}, {"unknown": 1}):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_literature_intent_uses_the_tool(self) -> None:
        literature = project_type("literature")
        intent = literature.intents["reference_check"]
        self.assertEqual(intent.tool.name, TOOL_REFERENCE_CHECK)


class TestReferenceCheckRun(unittest.TestCase):
    """A full run on a small Markdown project."""

    def setUp(self) -> None:
        self.folder = TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.chunks = [
            chunk("p1", PAPER, path="paper.md"),
            chunk("b1", BRAPH, path="braph.md"),
            chunk("c1", BRAPH2, path="braph2.md"),
        ]

    def tearDown(self) -> None:
        self.folder.cleanup()

    def _run(self, chosen: object) -> tuple[object, SimpleNamespace]:
        context = SimpleNamespace(
            project_root=self.root, chunks=self.chunks,
            progress=Mock(), done=Mock(), choose=None,
        )
        request = ToolRequest(
            "check the references of paper.md",
            "check the references of paper.md",
        )
        with patch(
            "kilin.tools.reference_check.tool.choose_document",
            return_value=chosen,
        ):
            return tool.run(
                tool.ReferenceCheckSettings(max_listed=1), request, context
            ), context

    def test_report_and_answer(self) -> None:
        result, context = self._run((Path("paper.md"), "named", False))
        summary = result.summary
        self.assertEqual(summary["references"], 5)
        self.assertEqual(summary["numbers_printed"], 5)
        self.assertEqual(
            (summary["found"], summary["probable"], summary["missing"]),
            (1, 1, 3),
        )
        self.assertEqual(summary["citation_style"], "brackets")
        self.assertEqual(summary["uncited"], [5])
        self.assertIn("[2] Mijalkov", result.answer)
        self.assertIn("→ braph.md", result.answer)
        self.assertIn("another version", result.answer)
        # One missing reference is listed, the others counted.
        self.assertIn("… and 2 more", result.answer)
        self.assertIn("Never cited in the text: [5]", result.answer)
        context.done.assert_called_once()

        report = (self.root / summary["report"]).read_text()
        self.assertTrue(report.startswith("# Reference check: paper.md"))
        section = report[report.index("### [2]"):report.index("### [3]")]
        self.assertIn(
            "braph.md (found: first page and first author)", section
        )
        self.assertIn("Software such as BRAPH implements", section)
        self.assertIn("We followed the protocol of [2].", section)
        self.assertIn("not found in the text", report)

    def test_cancelled_choice(self) -> None:
        result, context = self._run("Name the paper to check.")
        self.assertEqual(result.answer, "Name the paper to check.")
        self.assertEqual(result.summary, {"name": "reference_check"})
        context.done.assert_called_once()

    def test_paper_without_reference_list(self) -> None:
        self.chunks[0] = chunk("p1", "Just text [1] here.\n", path="paper.md")
        result, _ = self._run((Path("paper.md"), "named", True))
        self.assertIn("no reference list", result.answer)
        self.assertIsNone(result.summary["report"])
        self.assertIn("name the paper if you meant another", result.answer)
        self.assertFalse((self.root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
