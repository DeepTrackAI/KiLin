"""Validate the local technical documentation without external services.

The checks keep Markdown targets, editable SVG diagrams, and the module
ownership matrix synchronized with the repository. They use only the Python
standard library, so they also run in the lightweight documentation job.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import unittest
from urllib.parse import unquote
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SVG_NS = "{http://www.w3.org/2000/svg}"
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
# Paths in HTML inside Markdown: the README logo's <img src> and srcset.
HTML_PATTERN = re.compile(r"\b(?:src|srcset|href)=\"([^\"]+)\"")
MODULE_PATTERN = re.compile(r"\| \[`(kilin/[^`]+\.py)`\]\(")


class TestDocumentationAssets(unittest.TestCase):
    """Check repository-local references and maintained inventories."""

    def test_markdown_targets_exist(self) -> None:
        """Resolve local paths without requiring a network connection."""

        documents = [
            ROOT / "README.md",
            *sorted(DOCS.glob("*.md")),
            *sorted((ROOT / "kilin").rglob("README.md")),
        ]
        errors: list[str] = []
        for document in documents:
            text = document.read_text(encoding="utf-8")
            for target in (
                *LINK_PATTERN.findall(text),
                *HTML_PATTERN.findall(text),
            ):
                if target.startswith(("https://", "http://", "mailto:")):
                    continue
                path = unquote(target.split("#", 1)[0])
                if not path:
                    continue
                if not (document.parent / path).exists():
                    name = document.relative_to(ROOT).as_posix()
                    errors.append(f"{name}: {target}")
        self.assertEqual(errors, [], "Invalid local Markdown targets")

    def test_svg_figures_are_valid_and_documented(self) -> None:
        """Require accessible, self-contained SVGs used by owner pages."""

        figures = sorted((DOCS / "figures").glob("*.svg"))
        self.assertTrue(figures, "No architecture diagrams were found")
        owner_pages = [
            path.read_text(encoding="utf-8")
            for path in DOCS.glob("*.md")
            if path.name != "diagrams.md"
        ]
        for figure in figures:
            with self.subTest(figure=figure.name):
                diagram = ElementTree.parse(figure).getroot()
                self.assertEqual(diagram.tag, f"{SVG_NS}svg")
                for tag in ("title", "desc"):
                    description = diagram.find(f"{SVG_NS}{tag}")
                    self.assertIsNotNone(description)
                    self.assertTrue("".join(description.itertext()).strip())
                reference = f"figures/{figure.name}"
                self.assertTrue(
                    any(reference in page for page in owner_pages),
                    f"No owner document embeds {reference}",
                )

    def test_module_map_covers_every_production_module(self) -> None:
        """Assign exactly one primary documentation owner per module."""

        expected = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "kilin").rglob("*.py")
        }
        inventory = MODULE_PATTERN.findall(
            (DOCS / "module_map.md").read_text(encoding="utf-8")
        )
        counts = Counter(inventory)
        self.assertEqual(set(counts), expected)
        self.assertEqual(
            sorted(name for name, count in counts.items() if count != 1),
            [],
        )


if __name__ == "__main__":
    unittest.main()
