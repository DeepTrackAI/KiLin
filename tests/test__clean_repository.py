"""Test the safe source ZIP policy in the repository-cleanup notebook.

The notebook is loaded as data: preview, deletion, and ZIP-export cells
are never executed during test initialization.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "assets" / "clean_repository.ipynb"


def load_export_helpers() -> dict[str, object]:
    """Load only the safe definitions from the maintenance notebook.

    Returns
    -------
    dict[str, object]
        Namespace containing the cleanup and ZIP-export functions.

    """

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "_clean_repository_tests"}
    for index in (1, 5, 6, 8, 15, 16, 18):
        source = "".join(notebook["cells"][index]["source"])
        exec(compile(source, f"{NOTEBOOK}:cell-{index}", "exec"),
             namespace)
    return namespace


class TestCleanRepository(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.helpers = load_export_helpers()

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "checkout"
        self.root.mkdir()
        article = self.root / "_si_article_"
        for name in (
            "benchmarks/notebooks",
            "benchmarks/gold",
            "benchmarks/sandbox",
            "benchmarks/results",
            "code_repo",
            "literature_repo",
            "code_analysis",
            "reference_analysis",
        ):
            (article / name).mkdir(parents=True)
        (article / "benchmarks/common.py").write_text(
            "# Benchmark source\n", encoding="utf-8"
        )
        (article / "benchmarks/gold/gold.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (article / "benchmarks/notebooks/run.ipynb").write_text(
            "{}\n", encoding="utf-8"
        )
        (article / "code_analysis/article.tex").write_text(
            "Article draft", encoding="utf-8"
        )
        (article / "reference_analysis/.gitkeep").touch()
        (article / "benchmarks/sandbox/.gitkeep").touch()
        (article / "benchmarks/results/.gitkeep").touch()
        (article / "code_repo/private.py").write_text(
            "SECRET", encoding="utf-8"
        )
        (article / "literature_repo/private.pdf").write_bytes(b"PDF")
        (article / "benchmarks/results/results.json").write_text(
            "CONFIDENTIAL", encoding="utf-8"
        )
        (article / "benchmarks/sandbox/workspace.db").write_text(
            "PRIVATE", encoding="utf-8"
        )
        self.selection = {"_si_article_/": True, "sandbox/": True}

    def git(self, *args: str) -> None:
        subprocess.run(
            ("git", "-C", str(self.root), *args),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def track_sources(self) -> None:
        self.git("init", "-q")
        self.git("add", "-f", "_si_article_")

    def test_default_selection_uses_si_article(self) -> None:
        choices = self.helpers["ZIP_TOP_LEVEL_SELECTION"]
        self.assertTrue(choices["_si_article_/"])
        self.assertNotIn("benchmarks/", choices)

    def test_source_policy_excludes_local_repositories(self) -> None:
        allowed = self.helpers["should_export"]
        self.assertTrue(allowed("_si_article_/code_analysis/article.tex"))
        self.assertTrue(allowed("_si_article_/benchmarks/common.py"))
        self.assertTrue(
            allowed("_si_article_/reference_analysis/.gitkeep")
        )
        for relative in (
            "_si_article_/code_repo/private.py",
            "_si_article_/code_repo/.gitkeep",
            "_si_article_/literature_repo/private.pdf",
            "_si_article_/benchmarks/results/results.json",
            "_si_article_/benchmarks/sandbox/workspace.db",
            "_si_article_/reference_analysis/unfinished.md",
        ):
            with self.subTest(relative=relative):
                self.assertFalse(allowed(relative))

    def test_required_directories_include_empty_placeholders(self) -> None:
        selected = self.helpers["resolve_top_level_selection"]
        files, folders = selected(self.root, self.selection)
        self.assertFalse(files)
        self.assertIn("_si_article_/", folders)
        for relative in self.helpers["SI_ARTICLE_REQUIRED_DIRECTORIES"]:
            self.assertIn(relative, folders)

    def test_symlinked_checkout_root_uses_canonical_paths(self) -> None:
        """Accept safe source files when the checkout path is an alias.

        """

        alias = self.root.parent / "checkout_alias"
        try:
            alias.symlink_to(self.root, target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("Cannot create directory symlinks")
        self.track_sources()

        selected = self.helpers["resolve_top_level_selection"]
        files, directories = selected(alias, self.selection)
        self.assertFalse(files)
        self.assertIn("_si_article_/", directories)

        source_files, untracked = self.helpers["list_source_files"](
            alias, self.selection
        )
        self.assertIn(
            "_si_article_/code_analysis/article.tex", source_files
        )
        self.assertFalse(untracked)

        output = self.root.parent / "alias-export.zip"
        self.helpers["export_source_zip"](
            alias, output, source_files, self.selection
        )
        import zipfile

        with zipfile.ZipFile(output) as archive:
            self.assertIn(
                "_si_article_/code_analysis/article.tex",
                archive.namelist(),
            )
            self.assertEqual(archive.testzip(), None)

    def test_missing_article_directory_rejects_export(self) -> None:
        missing = self.root / "_si_article_/reference_analysis"
        shutil.rmtree(missing)
        selected = self.helpers["resolve_top_level_selection"]
        with self.assertRaisesRegex(RuntimeError, "SI article folder"):
            selected(self.root, self.selection)

    def test_symlinked_article_directory_rejects_export(self) -> None:
        article = self.root / "_si_article_"
        link = article / "code_repo"
        shutil.rmtree(link)
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("Cannot create directory symlinks")
        selected = self.helpers["resolve_top_level_selection"]
        with self.assertRaisesRegex(RuntimeError, "SI article folder"):
            selected(self.root, self.selection)

    def test_tracked_sources_export_and_untracked_are_reported(self) -> None:
        self.track_sources()
        article = self.root / "_si_article_"
        (article / "code_analysis/new.tex").write_text(
            "UNTRACKED", encoding="utf-8"
        )
        (article / "code_repo/untracked.py").write_text(
            "PRIVATE", encoding="utf-8"
        )
        source, untracked = self.helpers["list_source_files"](
            self.root, self.selection
        )
        self.assertIn("_si_article_/code_analysis/article.tex", source)
        self.assertIn("_si_article_/reference_analysis/.gitkeep", source)
        self.assertNotIn("_si_article_/code_repo/private.py", source)
        self.assertNotIn("_si_article_/literature_repo/private.pdf", source)
        self.assertIn("_si_article_/code_analysis/new.tex", untracked)
        self.assertNotIn("_si_article_/code_repo/untracked.py", untracked)

    def test_missing_tracked_reference_placeholder_rejected(self) -> None:
        self.git("init", "-q")
        self.git("add", "-f", "_si_article_/code_analysis")
        with self.assertRaisesRegex(RuntimeError, "reference-analysis"):
            self.helpers["list_source_files"](
                self.root, self.selection
            )

    def test_export_zip_has_correct_structure_and_no_private_data(self):
        self.track_sources()
        files, _ = self.helpers["list_source_files"](
            self.root, self.selection
        )
        output = self.root.parent / "export.zip"
        self.helpers["export_source_zip"](
            self.root, output, files, self.selection
        )
        import zipfile

        with zipfile.ZipFile(output) as archive:
            entries = set(archive.namelist())
            self.assertIn("_si_article_/benchmarks/common.py", entries)
            self.assertIn("_si_article_/code_analysis/article.tex", entries)
            self.assertIn(
                "_si_article_/reference_analysis/.gitkeep", entries
            )
            for folder in self.helpers["SI_ARTICLE_EMPTY_DIRECTORIES"]:
                self.assertIn(folder, entries)
                self.assertFalse(any(
                    item.startswith(folder) and item != folder
                    for item in entries
                ))
            self.assertFalse(any(
                item.startswith("benchmarks/") for item in entries
            ))
            self.assertFalse(any(
                item.startswith("_si_article_/reference_analysis/")
                and item != "_si_article_/reference_analysis/.gitkeep"
                and item != "_si_article_/reference_analysis/"
                for item in entries
            ))
            self.assertEqual(archive.testzip(), None)

    def test_tracked_corpora_stay_in_git_but_not_in_zip(self) -> None:
        """Retain populated corpora while exporting only their directories."""

        import zipfile

        self.track_sources()
        corpora = (
            "_si_article_/code_repo/private.py",
            "_si_article_/literature_repo/private.pdf",
        )
        tracked_before = self.helpers["git_tracked_paths"](self.root)
        snapshots = {
            relative: (self.root / relative).read_bytes()
            for relative in corpora
        }
        for relative in corpora:
            with self.subTest(relative=relative):
                self.assertIn(relative, tracked_before)
                self.assertFalse(
                    (self.root / relative).parent.joinpath(".gitkeep")
                    .exists()
                )
        directories, files = self.helpers["find_temporary_paths"](self.root)
        for candidate in (*directories, *files):
            for folder in ("code_repo", "literature_repo"):
                protected = self.root / "_si_article_" / folder
                self.assertFalse(candidate.is_relative_to(protected))
        sources, _ = self.helpers["list_source_files"](
            self.root, self.selection
        )
        output = self.root.parent / "tracked-corpora.zip"
        self.helpers["export_source_zip"](
            self.root, output, sources, self.selection
        )
        with zipfile.ZipFile(output) as archive:
            entries = archive.namelist()
            for folder in ("code_repo", "literature_repo"):
                prefix = f"_si_article_/{folder}/"
                self.assertEqual(
                    [name for name in entries if name.startswith(prefix)],
                    [prefix],
                )
        self.assertEqual(
            self.helpers["git_tracked_paths"](self.root), tracked_before
        )
        for relative, content in snapshots.items():
            with self.subTest(relative=relative):
                self.assertEqual((self.root / relative).read_bytes(), content)

    def test_repository_tutorial_excludes_article_sources(self) -> None:
        notebook = ROOT / "tutorials/40_repository_files.ipynb"
        content = json.loads(notebook.read_text(encoding="utf-8"))
        exclusions = None
        for cell in content["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            for statement in tree.body:
                if not isinstance(statement, ast.Assign):
                    continue
                if any(
                    isinstance(target, ast.Name)
                    and target.id == "EXCLUDE_PATHS"
                    for target in statement.targets
                ):
                    exclusions = ast.literal_eval(statement.value)
        self.assertIsNotNone(exclusions)
        self.assertIn("_si_article_", exclusions)
        self.assertIn("benchmarks", exclusions)

    def test_cleanup_never_enters_local_repository_directories(self):
        self.track_sources()
        (self.root / "_si_article_/code_repo/.DS_Store").touch()
        (self.root / "_si_article_/literature_repo/.DS_Store").touch()
        (self.root / "_si_article_/code_analysis/.DS_Store").touch()
        _, removable = self.helpers["find_temporary_paths"](self.root)
        relative = {path.relative_to(self.root).as_posix()
                    for path in removable}
        self.assertIn("_si_article_/code_analysis/.DS_Store", relative)
        self.assertNotIn("_si_article_/code_repo/.DS_Store", relative)
        self.assertNotIn("_si_article_/literature_repo/.DS_Store", relative)


if __name__ == "__main__":
    unittest.main()
