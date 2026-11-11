"""Test the article benchmark helpers, notebooks and gold files.

The tests verify the scoring helpers in ``_si_article_/benchmarks``,
gold-file structure, notebook syntax, and the project-workspace calls the
notebooks rely on, without downloading benchmark datasets or loading
language models. They are skipped in a copy without the article folder.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "_si_article_" / "benchmarks"

if not (BENCHMARKS / "common.py").is_file():
    raise unittest.SkipTest("The article benchmarks are not in this copy.")

# pylint: disable=C0413:wrong-import-position
from _si_article_.benchmarks.common import (  # noqa: E402
    abstains,
    citation_coverage,
    citation_validity,
    extract_cited_indices,
    final_answer,
    format_query,
    gold_evidence_indices,
    open_or_initialize,
    verbatim_overlap_chars,
)
from kilin.project import (  # noqa: E402
    configure_project,
    prepare_project_workspace,
    synchronize_project_workspace,
)

# These fixtures intentionally benchmark the packaged code-repository type.
CODE_BENCHMARK_TYPE = "code"

GOLD_SETS = {
    "dogfood_kilin_code.json": 22,
    "dogfood_deeptrack_code.json": 20,
}


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

    def test_reasoning_is_not_scored(self) -> None:
        """Read citations and overlap from the final answer only."""

        reply = (
            "<think>Evidence [7] has the list [0, 1, 2] and the text "
            "repeated verbatim across many characters.</think>\n"
            "The function is defined in the module [1]."
        )
        self.assertEqual(
            final_answer(reply), "The function is defined in the module [1]."
        )
        self.assertEqual(final_answer("Opened by the template</think> A"), "A")
        self.assertEqual(extract_cited_indices(reply), [1])
        self.assertEqual(
            citation_validity(reply, evidence_count=1), (True, [])
        )

        class Excerpt:
            """Evidence text needed by the overlap check."""

            text = "repeated verbatim across many characters"

        self.assertLess(verbatim_overlap_chars(reply, [Excerpt()]), 20)

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

        notebooks = sorted((BENCHMARKS / "notebooks").glob("*.ipynb"))
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

        gold = BENCHMARKS / "gold" / "dogfood_kilin_code.json"
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
                    source_file = ROOT / name
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

    def test_notebooks_use_the_article_folder(self) -> None:
        """Read and write benchmark files under _si_article_/benchmarks."""

        for path in sorted((BENCHMARKS / "notebooks").glob("*.ipynb")):
            with self.subTest(notebook=path.name):
                data = json.loads(path.read_text(encoding="utf-8"))
                source = "\n".join(
                    "".join(cell["source"])
                    for cell in data["cells"]
                    if cell["cell_type"] == "code"
                )
                self.assertIn(
                    'benchmarks_dir = root / "_si_article_"', source
                )
                self.assertNotIn('root / "benchmarks"', source)
                self.assertNotIn("initialize_project(", source)

    def test_readme_lists_every_notebook(self) -> None:
        """Keep the README inventory in step with the notebook files."""

        readme = (BENCHMARKS / "README.md").read_text(encoding="utf-8")
        listed = set(re.findall(r"^\| `([^`]+\.ipynb)` \|", readme, re.M))
        present = {
            path.name for path in (BENCHMARKS / "notebooks").glob("*.ipynb")
        }
        self.assertEqual(listed, present)

    def test_gold_sets_are_consistent(self) -> None:
        """Give each task a unique id and citations only when answerable."""

        for name, count in GOLD_SETS.items():
            data = json.loads(
                (BENCHMARKS / "gold" / name).read_text(encoding="utf-8")
            )
            tasks = data["tasks"]
            self.assertEqual(len(tasks), count, name)
            ids = [task["task_id"] for task in tasks]
            self.assertEqual(len(ids), len(set(ids)), name)
            for task in tasks:
                with self.subTest(gold=name, task=task["task_id"]):
                    self.assertTrue(task["question"].strip())
                    if task["requires_abstention"]:
                        self.assertEqual(task["gold_citations"], [])
                    else:
                        self.assertTrue(task["gold_citations"])
                        self.assertTrue(task["fact_checklist"])

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
                source, CODE_BENCHMARK_TYPE, base / "workspaces"
            )
            self.assertTrue((project.source / "config.json").is_file())
            self.assertFalse((project.source / "large.py").exists())
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


class TestOpenOrInitialize(unittest.TestCase):
    """Check how benchmark notebooks open their sandbox projects."""

    def test_creates_a_project_in_a_new_folder(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder) / "benchmark"
            project = open_or_initialize(root, CODE_BENCHMARK_TYPE)
            self.assertTrue((root / "profile.ini").is_file())
            self.assertEqual(project.root.resolve(), root.resolve())

    def test_reopens_an_existing_project_with_its_settings(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder) / "benchmark"
            project = open_or_initialize(root, CODE_BENCHMARK_TYPE)
            configure_project(project.root, {"extensions": ".py"})
            project.source.mkdir(parents=True, exist_ok=True)
            (project.source / "kept.py").write_text("value = 1\n")
            reopened = open_or_initialize(root, CODE_BENCHMARK_TYPE)
            self.assertEqual(reopened.profile.extensions, (".py",))
            self.assertTrue((reopened.source / "kept.py").is_file())

    def test_recreates_a_folder_left_by_an_interrupted_run(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder) / "benchmark"
            (root / "repository").mkdir(parents=True)
            (root / "repository" / "stale.py").write_text("value = 1\n")
            project = open_or_initialize(root, CODE_BENCHMARK_TYPE)
            self.assertTrue((root / "profile.ini").is_file())
            self.assertFalse((project.source / "stale.py").exists())


if __name__ == "__main__":
    unittest.main()
