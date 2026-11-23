"""Protect relocated article scripts and evaluation corpus boundaries.

The tests exercise command-line startup without loading models and export
small, temporary Git histories to check that evaluation material does not
enter the default indexed corpus. Existing article results are read-only.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from _si_article_.code_analysis import code_example, paper_numbers


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "_si_article_" / "code_analysis"


class _StopBeforeModels(Exception):
    """Stop a CLI test immediately before workspace or model creation."""


class TestCodeAnalysis(unittest.TestCase):
    def test_paths_resolve_from_scripts(self) -> None:
        """Keep questions, results and workspace rooted in the checkout."""

        self.assertEqual(code_example.REPOSITORY_ROOT, ROOT)
        self.assertEqual(paper_numbers.REPOSITORY_ROOT, ROOT)
        self.assertEqual(
            code_example.QUESTIONS_FILE, ANALYSIS / "code_questions.json"
        )
        self.assertEqual(
            code_example.OUTPUT_DIRECTORY, ANALYSIS / "results"
        )
        self.assertEqual(
            paper_numbers.RESULTS_DIRECTORY, ANALYSIS / "results"
        )
        self.assertEqual(
            paper_numbers.WORKSPACE_DIRECTORY,
            ROOT / "sandbox" / "paper_numbers",
        )
        payload = json.loads(
            code_example.QUESTIONS_FILE.read_text(encoding="utf-8")
        )
        self.assertTrue(payload["questions"])

    def test_help_from_outside_repository(self) -> None:
        """Support direct script execution from an unrelated directory."""

        with TemporaryDirectory() as directory:
            for name in ("code_example.py", "paper_numbers.py"):
                with self.subTest(script=name):
                    result = subprocess.run(
                        [sys.executable, str(ANALYSIS / name), "--help"],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                        encoding="utf-8",
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("--ref", result.stdout)
                    self.assertIn("--allow-download", result.stdout)

    def test_module_help(self) -> None:
        """Support module execution for both namespace-package scripts."""

        for name in ("code_example", "paper_numbers"):
            with self.subTest(module=name):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        f"_si_article_.code_analysis.{name}",
                        "--help",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                    encoding="utf-8",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--ref", result.stdout)

    def test_import_and_help_do_not_load_models(self) -> None:
        """Keep inspection independent of backends and model downloads."""

        code = (
            "import contextlib, io, sys\n"
            "from _si_article_.code_analysis import "
            "code_example, paper_numbers\n"
            "for module in (code_example, paper_numbers):\n"
            "    with contextlib.redirect_stdout(io.StringIO()):\n"
            "        try:\n"
            "            module.main(['--help'])\n"
            "        except SystemExit as error:\n"
            "            assert error.code == 0\n"
            "blocked = ('torch', 'mlx', 'transformers', 'PySide6', "
            "'huggingface_hub')\n"
            "assert not any(name.split('.')[0] in blocked "
            "for name in sys.modules), sorted(sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            encoding="utf-8",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_cli_exclusions(self) -> None:
        """Both measurement entry points exclude evaluation material."""

        for module in (code_example, paper_numbers):
            with self.subTest(module=module.__name__):
                with (
                    TemporaryDirectory() as directory,
                    patch.dict(os.environ),
                    patch.object(
                        paper_numbers,
                        "RESULTS_DIRECTORY",
                        Path(directory) / "results",
                    ),
                    patch.object(module, "describe_environment",
                                 return_value={}),
                    patch.object(
                        module,
                        "prepare_code_project",
                        side_effect=_StopBeforeModels,
                    ) as prepare,
                    self.assertRaises(_StopBeforeModels),
                ):
                    module.main([])
                prepare.assert_called_once_with(
                    "HEAD", paper_numbers.DEFAULT_EXCLUDES,
                    allow_download=False,
                )
                self.assertIn(
                    "_si_article_", prepare.call_args.args[1]
                )

    def test_include_examples_is_explicit_opt_in(self) -> None:
        """Retain the exploratory override without weakening defaults."""

        with (
            TemporaryDirectory() as directory,
            patch.dict(os.environ),
            patch.object(
                paper_numbers, "RESULTS_DIRECTORY", Path(directory)
            ),
            patch.object(paper_numbers, "describe_environment",
                         return_value={}),
            patch.object(
                paper_numbers,
                "prepare_code_project",
                side_effect=_StopBeforeModels,
            ) as prepare,
            self.assertRaises(_StopBeforeModels),
        ):
            paper_numbers.main(["--include-examples"])
        prepare.assert_called_once_with("HEAD", (), allow_download=False)
        with redirect_stdout(StringIO()) as output:
            with self.assertRaises(SystemExit):
                paper_numbers.main(["--help"])
        self.assertIn(
            "evaluation material", " ".join(output.getvalue().split())
        )


@unittest.skipUnless(
    shutil.which("git") and shutil.which("tar"),
    "Git and tar are required for revision-export integration tests",
)
class TestCodeAnalysisExport(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "checkout"
        self.root.mkdir()
        self.files = {
            "kilin/example.py": "def example():\n    return 1\n",
            "docs/api.md": "Useful API documentation\n",
            "README.md": "Source repository\n",
            "_si_article_/code_analysis/code_questions.json": "SECRET_Q",
            "_si_article_/code_analysis/results/answer.json": "SECRET_A",
            "_si_article_/code_repo/large.py": "CORPUS_CODE",
            "_si_article_/literature_repo/article.txt": "CORPUS_TEXT",
            "_si_article_/benchmarks/gold/questions.json": "GOLD",
            "assets/check.py": "MEASUREMENT",
            "benchmarks/questions.json": "LEGACY_GOLD",
            "paper/answers.json": "LEGACY_ANSWER",
            "example_repos/package.py": "EXAMPLE_CORPUS",
        }
        for relative, content in self.files.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self.git("init", "-q")
        self.git("add", ".")
        self.git(
            "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "-qm", "Test corpus",
        )
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.destination = self.root.parent / "export"
        self.destination.mkdir()
        self.patcher = patch.object(
            paper_numbers, "REPOSITORY_ROOT", self.root
        )
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def git(self, *args: str) -> str:
        """Run Git within the isolated test checkout.

        Parameters
        ----------
        *args: str
            Arguments passed after the checkout selection.

        Returns
        -------
        str
            Standard output of the successful command.

        """

        result = subprocess.run(
            ["git", "-C", str(self.root), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
            encoding="utf-8",
        )
        return result.stdout

    def test_default_export_excludes_evaluation_data(self) -> None:
        """Exclude tracked questions, results and corpora, not API docs."""

        result = paper_numbers.export_revision("HEAD", self.destination)
        exported = {
            path.relative_to(self.destination).as_posix()
            for path in self.destination.rglob("*") if path.is_file()
        }
        self.assertEqual(
            exported, {"kilin/example.py", "docs/api.md", "README.md"}
        )
        self.assertEqual(result["commit"], self.commit)
        self.assertEqual(result["ref"], "HEAD")
        self.assertEqual(
            result["excluded"], list(paper_numbers.DEFAULT_EXCLUDES)
        )
        for relative, content in self.files.items():
            with self.subTest(relative=relative):
                self.assertEqual(
                    (self.root / relative).read_text(encoding="utf-8"),
                    content,
                )

    def test_explicit_empty_exclusions_include_tracked_corpora(self) -> None:
        """Keep deliberate broad exports available for exploratory runs."""

        result = paper_numbers.export_revision(
            "HEAD", self.destination, excludes=()
        )
        exported = {
            path.relative_to(self.destination).as_posix()
            for path in self.destination.rglob("*") if path.is_file()
        }
        self.assertEqual(exported, set(self.files))
        self.assertEqual(result["excluded"], [])

    def test_export_uses_pinned_commit_not_working_files(self) -> None:
        """Exclude uncommitted changes and untracked source files."""

        (self.root / "kilin/example.py").write_text(
            "UNCOMMITTED", encoding="utf-8"
        )
        (self.root / "kilin/untracked.py").write_text(
            "UNTRACKED", encoding="utf-8"
        )
        result = paper_numbers.export_revision(
            self.commit, self.destination
        )
        self.assertEqual(result["commit"], self.commit)
        self.assertEqual(
            (self.destination / "kilin/example.py").read_text(
                encoding="utf-8"
            ),
            self.files["kilin/example.py"],
        )
        self.assertFalse((self.destination / "kilin/untracked.py").exists())

    def test_unknown_revision_has_no_export_side_effects(self) -> None:
        """Fail before writing destination files for an unknown revision."""

        with self.assertRaisesRegex(RuntimeError, "Unknown git revision"):
            paper_numbers.export_revision(
                "does-not-exist", self.destination
            )
        self.assertEqual(list(self.destination.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
