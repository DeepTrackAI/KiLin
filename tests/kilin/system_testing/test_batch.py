"""Test the opt-in clean replay policy without loading models."""

# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0103:invalid-name
# pylint: disable=protected-access

from importlib.resources import files
from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kilin.system_testing.batch import NOTEBOOKS, MARKER, _compare
from kilin.system_testing.batch import _notebook, run_batch
from kilin.system_testing import Experiment
from kilin.system_testing.replay_management import _archive


class TestReplayPolicy(unittest.TestCase):
    """A fresh run preserves old recordings and never replaces originals."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / "workspaces" / "010_original"
        self.original.mkdir(parents=True)
        self.original_file = self.original / "original.txt"
        self.original_file.write_text("reference", encoding="utf-8")
        self.replay = self.original.with_name("010_replay")
        self.replay.mkdir()
        self.previous = self.replay / "previous.txt"
        self.previous.write_text("previous", encoding="utf-8")
        self.experiment = Mock()
        event = SimpleNamespace(sequence=1, path="trace.json", metadata={},
                                details={}, exact_content={})
        operation = SimpleNamespace(kind="close_project", end_sequence=1)
        report = Mock(workspace=self.original.resolve(), valid=True,
                      event_count=1, events=(event,), operations=(operation,))
        self.experiment.validate.return_value = report

        def fresh(destination, progress):
            self.assertFalse(destination.exists())
            destination.mkdir()
            (destination / "new.txt").write_text("fresh", encoding="utf-8")
            (destination / "logs").mkdir()
            return SimpleNamespace(operation_count=1, validation=report)

        self.experiment.replay.side_effect = fresh

    def invoke(self, policy, *, replay=True):
        row = dict(replay_status="not requested")
        with patch.object(Experiment, "validate", self.experiment.validate), (
                patch.object(Experiment, "replay", self.experiment.replay)):
            _compare(self.original, row, replay, False, False, False, False,
                     {}, policy, self.root)
        self.assertEqual(self.original_file.read_text(encoding="utf-8"),
                         "reference")
        return row

    def test_archive_and_fresh(self):
        row = self.invoke("archive")
        archive = Path(row["archived_replay"])
        self.assertEqual(row["replay_status"], "completed")
        self.assertEqual((archive / "previous.txt").read_text(
            encoding="utf-8"), "previous")
        self.assertTrue((self.replay / "new.txt").is_file())
        self.assertTrue(archive.is_relative_to(self.root / "_replay_archive"))
        second = self.invoke("archive")
        self.assertNotEqual(second["archived_replay"], str(archive))
        self.assertTrue((archive / "previous.txt").is_file())

    def test_retain(self):
        self.assertEqual(self.invoke("retain")["replay_status"], "retained")
        self.experiment.replay.assert_not_called()
        self.assertTrue(self.previous.is_file())

    def test_refuse_reuse(self):
        self.assertEqual(self.invoke("error")["replay_status"], "error")
        self.experiment.replay.assert_not_called()
        self.assertTrue(self.previous.is_file())

    def test_disabled_replay(self):
        self.assertEqual(self.invoke("archive", replay=False)["replay_status"],
                         "not requested")
        self.experiment.replay.assert_not_called()
        self.assertTrue(self.previous.is_file())

    def test_invalid_original(self):
        self.experiment.validate.return_value.raise_for_errors.side_effect = (
            ValueError("Invalid original"))
        row = self.invoke("archive")
        self.assertEqual(row["replay_status"], "error")
        self.assertNotIn("archived_replay", row)
        self.assertTrue(self.previous.is_file())

    def test_failure_preserves_previous(self):
        def fail(destination, progress):
            destination.mkdir()
            (destination / "partial.txt").touch()
            raise RuntimeError("Replay failed")

        self.experiment.replay.side_effect = fail
        row = self.invoke("archive")
        self.assertEqual(row["replay_status"], "error")
        self.assertTrue((Path(row["archived_replay"]) /
                         "previous.txt").is_file())
        self.assertTrue((self.replay / "partial.txt").is_file())

    def test_archive_original_refused(self):
        with self.assertRaises(ValueError):
            _archive(self.original, self.root / "_replay_archive")
        self.assertTrue(self.original_file.is_file())


class TestPackagedNotebooks(unittest.TestCase):
    """Use package-owned templates independently of the calling checkout."""

    def test_resources_exist_and_contain_one_marker(self):
        resources = files("kilin.system_testing").joinpath("_notebooks")
        for _, name, _ in NOTEBOOKS:
            with self.subTest(name=name):
                resource = resources.joinpath(name)
                self.assertTrue(resource.is_file())
                data = json.loads(resource.read_text(encoding="utf-8"))
                cells = [c for c in data["cells"]
                         if c["cell_type"] == "code"]
                markers = sum(line == MARKER for c in cells
                              for line in "".join(c["source"]).splitlines())
                self.assertEqual(markers, 1)
                self.assertTrue(all(not c["outputs"] and
                                    c["execution_count"] is None
                                    for c in cells))

    @unittest.skipUnless(find_spec("nbformat"), "nbformat is optional")
    def test_generate_without_local_templates(self):
        import nbformat

        with TemporaryDirectory() as name:
            root = Path(name)
            original = root / "workspaces" / "café's_original"
            original.mkdir(parents=True)

            def generate(refresh=True):
                reports, rows = run_batch(
                    [original], root, run_replay=False,
                    run_comparison=False, save_reports=False,
                    overwrite_reports=False, save_visual_analysis=False,
                    run_notebooks=False, refresh_notebooks=refresh,
                )
                self.assertEqual(reports, {})
                self.assertEqual(rows[0]["status"], "NOT ANALYZED")
                return rows[0]["notebooks"]

            created = generate()
            self.assertEqual(len(created), 2)
            for kind, template, suffix in NOTEBOOKS:
                path = original.with_name("café's" + suffix)
                self.assertEqual(created[kind]["status"], "created")
                nb = nbformat.read(path, as_version=4)
                nbformat.validate(nb)
                self.assertEqual(nb.metadata["kilin_system_tests"], {
                    "generated": True, "reference": original.name,
                    "template": template,
                })
                code = "\n".join(c.source for c in nb.cells
                                 if c.cell_type == "code")
                self.assertIn(f"REFERENCE = Path({original.name!r})", code)
                compile(code, str(path), "exec")
                path.write_text("retained local artifact", encoding="utf-8")
                # Obsolete local templates must not be read or executed.
                (root / template).write_text("invalid local notebook",
                                             encoding="utf-8")
            retained = generate(refresh=False)
            for kind, _, suffix in NOTEBOOKS:
                path = original.with_name("café's" + suffix)
                self.assertEqual(retained[kind]["status"], "retained")
                self.assertEqual(path.read_text(encoding="utf-8"),
                                 "retained local artifact")
            refreshed = generate()
            for kind, template, suffix in NOTEBOOKS:
                self.assertEqual(refreshed[kind]["status"], "refreshed")
                path = original.with_name("café's" + suffix)
                nbformat.validate(nbformat.read(path, as_version=4))
                self.assertEqual((root / template).read_text(
                    encoding="utf-8"), "invalid local notebook")

    @unittest.skipUnless(find_spec("nbformat"), "nbformat is optional")
    def test_missing_resource_does_not_fall_back_to_local_template(self):
        with self.assertRaisesRegex(FileNotFoundError, "Missing packaged"):
            _notebook("unavailable.ipynb", Path("example_original"))
