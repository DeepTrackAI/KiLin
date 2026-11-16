"""Check that the tutorial notebooks remain runnable source examples.

Tests load notebook JSON as data and do not launch models, download files,
or execute project workspaces. Runtime examples are also verified manually
with controlled stand-in services during tutorial revisions.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
LINK_PATTERN = re.compile(r"\]\(([^)]+)\)")


def notebooks() -> list[Path]:
    """Return the current numbered tutorial notebooks in order.

    Returns
    -------
    list[pathlib.Path]
        Tutorial notebooks ordered by their numeric prefixes.

    """

    return sorted(
        TUTORIALS.glob("[0-9]*_*.ipynb"),
        key=lambda path: int(path.name.split("_", 1)[0]),
    )


class TestTutorials(unittest.TestCase):
    """Detect broken notebook structure and stale source references."""

    def test_numbered_series_and_guide(self) -> None:
        items = notebooks()
        self.assertEqual(
            [int(item.name.split("_", 1)[0]) for item in items],
            list(range(10, 200, 10)),
        )
        self.assertTrue((TUTORIALS / "README.md").is_file())
        for file in items:
            with self.subTest(file=file.name):
                document = json.loads(file.read_text(encoding="utf-8"))
                introduction = "".join(document["cells"][0]["source"])
                self.assertIn("[tutorial guide](./README.md)", introduction)

    def test_numbered_notebook_cell_ids_are_unique(self) -> None:
        """Require valid, unique IDs in all numbered notebook cells."""

        for file in notebooks():
            document = json.loads(file.read_text(encoding="utf-8"))
            identifiers = []
            for index, cell in enumerate(document["cells"]):
                with self.subTest(file=file.name, cell=index):
                    identifier = cell.get("id")
                    self.assertIsInstance(identifier, str)
                    self.assertRegex(identifier, r"^[A-Za-z0-9_-]{1,64}$")
                    identifiers.append(identifier)
            self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_guide_and_notebook_navigation_match_series(self) -> None:
        """Keep tutorial titles, index links, and navigation synchronized."""

        items = notebooks()
        guide = (TUTORIALS / "README.md").read_text(
            encoding="utf-8"
        )
        for index, path in enumerate(items):
            with self.subTest(file=path.name):
                number = int(path.name.split("_", 1)[0])
                notebook = json.loads(path.read_text(encoding="utf-8"))
                first = "".join(notebook["cells"][0]["source"])
                last = "".join(notebook["cells"][-1]["source"])
                self.assertTrue(first.startswith(f"# Tutorial {number} "))
                self.assertIn(f"./{path.name}", guide)
                if index:
                    self.assertIn(f"./{items[index - 1].name}", last)
                    self.assertIn("**Previous:**", last)
                else:
                    self.assertNotIn("**Previous:**", last)
                if index < len(items) - 1:
                    self.assertIn(f"./{items[index + 1].name}", last)
                    self.assertIn("**Next:**", last)
                else:
                    self.assertNotIn("**Next:**", last)

    def test_code_cells_compile_and_outputs_are_empty(self) -> None:
        for file in notebooks():
            document = json.loads(file.read_text(encoding="utf-8"))
            for index, cell in enumerate(document["cells"]):
                with self.subTest(file=file.name, cell=index):
                    if cell["cell_type"] != "code":
                        continue
                    code = "".join(cell["source"])
                    compile(code, f"{file.name}:cell{index}", "exec")
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertFalse(cell.get("outputs"))

    def test_static_local_links_exist(self) -> None:
        for file in [*notebooks(), TUTORIALS / "README.md"]:
            if file.suffix == ".ipynb":
                document = json.loads(file.read_text(encoding="utf-8"))
                content = "\n".join(
                    "".join(cell["source"])
                    for cell in document["cells"]
                    if cell["cell_type"] == "markdown"
                )
            else:
                content = file.read_text(encoding="utf-8")
            for link in LINK_PATTERN.findall(content):
                if link.startswith(("http:", "https:", "#")):
                    continue
                target = link.split("#", 1)[0]
                if not target or target.startswith("../sandbox/"):
                    continue
                with self.subTest(file=file.name, link=link):
                    self.assertTrue((file.parent / target).exists())

    def test_source_card_tutorial_is_offline_and_provenance_based(
        self,
    ) -> None:
        """Keep card examples independent of downloaded inference models."""

        path = TUTORIALS / "130_source_cards.ipynb"
        document = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in document["cells"]
            if cell["cell_type"] == "code"
        )
        for name in (
            "SourceDescriptor", "parse_card", "save_source_cards",
            "source_first_search", "chunk_ids",
        ):
            with self.subTest(name=name):
                self.assertIn(name, code)
        self.assertNotIn("SourceCardGenerator(", code)
        self.assertNotIn("download_model(", code)

    def test_capstone_bundled_corpus_and_model_gate(self) -> None:
        """Keep the final tutorial offline-first and its data complete."""

        path = TUTORIALS / "190_end_to_end_research_assistant.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for name in (
            "prepare_project_workspace", "read_repository_files",
            "chunk_source_units", "build_lexical_index", "lexical_search",
            "source_coverage", "sha256", "TraceConfig",
        ):
            with self.subTest(api=name):
                self.assertIn(name, code)
        self.assertIn("RUN_REAL_MODEL = False", code)
        self.assertIn("allow_download", code)
        self.assertNotIn("download_model(", code)

        source_dir = TUTORIALS / "data" / "190_research_pilot"
        expected = {
            "study_plan.md", "automated_tracking.md",
            "manual_tracking.md", "interpretation.md",
        }
        self.assertEqual(
            {path.name for path in source_dir.iterdir()}, expected
        )
        for file in sorted(source_dir.glob("*.md")):
            with self.subTest(source=file.name):
                self.assertIn(
                    "Fictional teaching dataset",
                    file.read_text(encoding="utf-8"),
                )

    def test_optional_article_corpus_paths(self) -> None:
        for number in (150, 180):
            file = next(TUTORIALS.glob(f"{number}_*.ipynb"))
            document = json.loads(file.read_text(encoding="utf-8"))
            code = "\n".join(
                "".join(cell["source"])
                for cell in document["cells"]
                if cell["cell_type"] == "code"
            )
            with self.subTest(file=file.name):
                self.assertIn('root / "_si_article_"', code)
                self.assertIn('root / "tests" / "fixtures"', code)
                self.assertNotIn('root / "example_repos"', code)

    def test_conversation_memory_tutorial_uses_real_local_contracts(
        self,
    ) -> None:
        """Keep the memory tutorial complete without model downloads."""

        notebook = json.loads((
            TUTORIALS / "170_conversation_memory.ipynb"
        ).read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for api in (
            "ConversationStore", "ConversationTurn", "TraceLogger",
            "FeedbackStore", "synchronize_memory", "search_memory",
            "render_memory_hits", "recall_topic", "model_identity",
            'reaction="disliked"', "excluded=True", "excluded=False",
        ):
            with self.subTest(api=api):
                self.assertIn(api, code)
        self.assertNotIn("download_model(", code)
        self.assertNotIn("ChatSession(", code)
        self.assertNotIn("RAGSession(", code)
        self.assertTrue((TUTORIALS / "180_tool_execution_flow.svg").is_file())
        self.assertFalse(
            (TUTORIALS / "170_tool_execution_flow.svg").exists()
        )
        self.assertFalse(
            (TUTORIALS / "data" / "180_research_pilot").exists()
        )


if __name__ == "__main__":
    unittest.main()
