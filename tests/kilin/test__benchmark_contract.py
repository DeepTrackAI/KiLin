"""Test the KíLín benchmark notebooks and offline grading contracts.

The tests verify independent benchmark helpers, gold-file structure,
notebook syntax, and current project-workspace calls without downloading
benchmark datasets or loading language models.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.common import (
    abstains,
    citation_coverage,
    citation_validity,
    extract_cited_indices,
    format_query,
    gold_evidence_indices,
)
from kilin import PROJECT_TYPE_ID_CODE
from kilin.project import (
    configure_project,
    prepare_project_workspace,
    synchronize_project_workspace,
)


_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARKS = _ROOT / "benchmarks"


class TestBenchmarkHelpers(unittest.TestCase):
    """Check citation, prompt-formatting and abstention helpers."""

    def test_grouped_citations_are_counted(self) -> None:
        """Recognize the same comma-grouped syntax accepted by KíLín."""

        self.assertEqual(
            extract_cited_indices("One [1, 2], then [2] and [3]."),
            [1, 2, 2, 3],
        )
        self.assertEqual(
            citation_validity("One [1, 3]", evidence_count=2),
            (False, [3]),
        )
        self.assertEqual(
            citation_validity("One [1, 2]", evidence_count=2),
            (True, []),
        )

    def test_source_location_and_citation_coverage(self) -> None:
        """Honor notebook cell and source-line boundaries."""

        class Evidence:
            """Minimal provenance needed by the benchmark helpers."""

            def __init__(self, citation: int, cell: int) -> None:
                """Attach one citation index to a notebook cell."""

                self.citation = citation
                self.path = Path("tutorials/example.ipynb")
                self.start_line = 1
                self.end_line = 8
                self.cell = cell

        evidence = [Evidence(1, 3), Evidence(2, 4)]
        gold = ("tutorials/example.ipynb#4:2-5",)
        self.assertEqual(gold_evidence_indices(evidence, gold), [2])
        self.assertEqual(citation_coverage("Found [1, 2]", evidence, gold), 1)
        self.assertEqual(citation_coverage("Found [1]", evidence, gold), 0)
        self.assertIsNone(
            citation_coverage("No sources", evidence, ("other.py:1",))
        )

    def test_abstention_is_only_a_heuristic(self) -> None:
        """Count final abstention phrasing, not an incidental mention."""

        self.assertTrue(abstains("Insufficient evidence."))
        self.assertFalse(
            abstains(
                "The prompt says insufficient evidence, but the "
                "repository documents this function clearly."
            )
        )

    def test_benchmark_query_uses_public_formatter(self) -> None:
        """Retain the Qwen3 instruction/query contract."""

        self.assertTrue(format_query("Find this function").startswith(
            "Instruct: Given a coding problem description, "
        ))
        self.assertIn("\nQuery: Find this function", format_query(
            "Find this function"
        ))


class TestBenchmarkArtifacts(unittest.TestCase):
    """Check benchmark files for migration and provenance regressions."""

    def test_notebook_cells_compile_and_have_no_saved_outputs(self) -> None:
        """Keep notebooks source-only, style-compliant and importable."""

        notebooks = sorted((_BENCHMARKS / "notebooks").glob("*.ipynb"))
        self.assertEqual(len(notebooks), 7)
        for path in notebooks:
            with self.subTest(notebook=path.name):
                data = json.loads(path.read_text(encoding="utf-8"))
                for index, cell in enumerate(data["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    source = "".join(cell["source"])
                    compile(source, f"{path.name}:cell{index}", "exec")
                    self.assertNotIn("from qachat", source)
                    self.assertNotIn("import qachat", source)
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    for line in source.splitlines():
                        self.assertLessEqual(len(line), 79)

    def test_kilin_gold_has_valid_source_locations(self) -> None:
        """Reject nonexistent files and impossible source line ranges."""

        gold = _BENCHMARKS / "gold" / "dogfood_kilin_code.json"
        data = json.loads(gold.read_text(encoding="utf-8"))
        tasks = data["tasks"]
        self.assertEqual(len(tasks), 22)
        self.assertEqual(
            sum(task["requires_abstention"] for task in tasks), 5
        )
        for task in tasks:
            with self.subTest(task=task["task_id"]):
                self.assertTrue(task["question"].strip())
                for spec in task["gold_citations"]:
                    location, _, span = spec.rpartition(":")
                    if "#" in location:
                        name, cell_number = location.rsplit("#", 1)
                    else:
                        name, cell_number = location, None
                    source_file = _ROOT / name
                    self.assertTrue(source_file.is_file())
                    if cell_number is not None:
                        notebook = json.loads(source_file.read_text())
                        text = "".join(
                            notebook["cells"][int(cell_number) - 1]["source"]
                        )
                    else:
                        text = source_file.read_text(encoding="utf-8")
                    first, _, last = span.partition("-")
                    start = int(first)
                    end = int(last) if last else start
                    self.assertGreaterEqual(start, 1)
                    self.assertGreaterEqual(end, start)
                    self.assertLessEqual(end, len(text.splitlines()))

    def test_project_profile_is_synchronized_not_recreated(self) -> None:
        """Keep a configured workspace when increasing source limits."""

        with TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "external"
            source.mkdir()
            (source / "small.py").write_text("value = 1\n")
            (source / "large.py").write_text("x = 1\n" * 200_000)
            (source / "config.json").write_text('{"valid": true}\n')
            project = prepare_project_workspace(
                source, PROJECT_TYPE_ID_CODE, base / "workspaces"
            )
            self.assertFalse((project.source / "config.json").exists())
            project = configure_project(
                project.root,
                {
                    "max_file_bytes": 5 * 1024 * 1024,
                    "extensions": ".py, .ipynb, .m, .md, .json",
                },
            )
            updated = synchronize_project_workspace(source, project.root)
            self.assertEqual(updated.root, project.root)
            self.assertTrue((updated.source / "large.py").is_file())
            self.assertTrue((updated.source / "config.json").is_file())


if __name__ == "__main__":
    unittest.main()
