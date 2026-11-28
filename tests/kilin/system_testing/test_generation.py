"""Exercise shared recipe execution through the recorded engine."""

# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from kilin.system_testing import inspect_receipt, receipt_path
from kilin.system_testing.generation import (
    Recorder, generate_reference, reference_overview,
)
from tests.kilin.system_testing import test_replay


class TestGeneration(unittest.TestCase):
    """The common recipe runner owns publication and failure boundaries."""

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workspaces = self.root / "workspaces"
        self.workspaces.mkdir()
        self.original = self.workspaces / "recipe_original"
        self.staged = self.root / "_recordings" / "recipe_recording"
        self.settings = {
            "backend": "mlx", "allow_download": False,
            "temperature": 0.0, "thinking": False,
            "max_new_tokens": 128,
        }

    def generate(self, record, *, operations=None, coverage=None):
        return generate_reference(
            self.root, "recipe", "writing", self.settings,
            operations or ["create_project", "ask", "close_project"],
            record, coverage or (lambda recorder, report: None),
        )

    def test_recipe_publishes_valid_reference(self):
        fixture = test_replay.TestReplay()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.engine.close()

        def record(recorder):
            recorder.open()
            recorder.ask("Q01", "What is saved?", terms=("unseen-term",))

        with fixture._model():
            recorder, report = self.generate(record)
        self.assertTrue(report.valid, report.issues)
        self.assertEqual(recorder.directory, self.original)
        self.assertFalse(self.staged.exists())
        self.assertEqual(report.workspace, self.original)
        receipt = inspect_receipt(self.original, report)
        self.assertEqual(receipt["status"], "unchanged")
        self.assertTrue(any("unseen-term" in row for row in
                            receipt["coverage"]))
        self.assertIn("What is saved?", reference_overview(recorder, report))

    def test_existing_paths_refuse_before_engine_construction(self):
        for path in (self.original, self.workspaces / "recipe_replay",
                     self.staged, receipt_path(self.original)):
            with self.subTest(path=path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("protected", encoding="utf-8")
                with patch("kilin.system_testing.generation.Recorder") as r:
                    with self.assertRaises(FileExistsError):
                        self.generate(Mock())
                    r.assert_not_called()
                self.assertEqual(path.read_text(encoding="utf-8"),
                                 "protected")
                path.unlink()

    def test_failure_closes_engine_and_preserves_staging(self):
        def fail(recorder):
            self.staged.mkdir()
            (self.staged / "partial.txt").write_text("partial",
                                                     encoding="utf-8")
            raise RuntimeError("Recipe failed")

        with patch("kilin.system_testing.generation.Recorder") as r:
            with self.assertRaisesRegex(RuntimeError, "Recipe failed"):
                self.generate(fail)
            r.return_value.engine.close.assert_called_once_with()
        self.assertFalse(self.original.exists())
        self.assertFalse(receipt_path(self.original).exists())
        self.assertTrue((self.staged / "partial.txt").is_file())

    def test_rejects_invalid_stem(self):
        for stem in ("", "..", "a/b", "a\\b", "a:b", "a_original"):
            with self.subTest(stem=stem):
                with self.assertRaises(ValueError):
                    generate_reference(self.root, stem, "writing",
                                       self.settings, [], Mock(), Mock())
        self.assertFalse((self.root / "_recordings").exists())

    def test_interactions_require_explicit_unique_local_approval(self):
        with patch("kilin.gui.engine.ChatEngine"):
            recorder = Recorder(self.staged, self.staged, "writing",
                                self.settings)
        recorder.plans = {"local": "quick"}
        recorder.prompt = ("plan", {"tool": "local"})
        self.assertEqual(recorder.read_command(), {
            "action": "plan_choice", "choice": "quick"})
        recorder.prompt = ("plan", {"tool": "local", "operation": "web"})
        with self.assertRaisesRegex(RuntimeError, "manual review"):
            recorder.read_command()
        recorder.options = {"Which?": "A"}
        recorder.prompt = ("choice", {"question": "Which?",
                                      "options": ["A1", "A2"]})
        with self.assertRaisesRegex(RuntimeError, "Unapproved choice"):
            recorder.read_command()


if __name__ == "__main__":
    unittest.main()
