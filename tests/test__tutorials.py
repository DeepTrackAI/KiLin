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
            list(range(10, 161, 10)),
        )
        self.assertTrue((TUTORIALS / "README.md").is_file())
        for file in items:
            with self.subTest(file=file.name):
                document = json.loads(file.read_text(encoding="utf-8"))
                introduction = "".join(document["cells"][0]["source"])
                self.assertIn("[tutorial guide](./README.md)", introduction)

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

    def test_optional_article_corpus_paths(self) -> None:
        for number in (140, 160):
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


if __name__ == "__main__":
    unittest.main()
