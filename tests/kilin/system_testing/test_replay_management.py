"""Exercise managed replay and receipts using recorded engine histories.

Use deterministic inference substitutes, real workspace persistence and
independent validation. Old reference and replay bytes must be preserved.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from kilin.system_testing import (
    Experiment, execution_provenance, inspect_receipt, publish_receipt,
)
from tests.kilin.system_testing import test_replay, test_experiment


class TestReplayManagement(unittest.TestCase):
    def setUp(self):
        fixture = test_replay.TestReplay()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        fixture.fixture._ask("Reference question")
        fixture.engine.close()
        self.root = fixture.experiment.workspace.with_name("010_original")
        fixture.experiment.workspace.rename(self.root)
        self.fixture = fixture
        self.experiment = Experiment(self.root)
        self.target = self.root.with_name("010_replay")
        self.archive = self.root.parent.parent / (self.root.parent.name +
                                                  "_archive")
        self.addCleanup(self._clean_archive)
        self.report = self.experiment.validate()
        self.assertTrue(self.report.valid, self.report.issues)
        self.before = test_experiment._snapshot(self.root)

    def _clean_archive(self):
        if self.archive.exists():
            from shutil import rmtree
            rmtree(self.archive)

    def run_action(self, policy="retain"):
        with self.fixture._model():
            return self.experiment.run_replay(
                existing_replay=policy, archive_directory=self.archive)

    def test_fresh_retain_archive_and_provenance(self):
        first = self.run_action()
        self.assertEqual(first.status, "completed", first.error)
        self.assertEqual(first.completed_operations,
                         len(self.report.operations))
        provenance = first.provenance
        self.assertEqual(provenance["status"], "recorded", provenance)
        self.assertEqual(provenance["recorded"]["reference"]["event_count"],
                         self.report.event_count)
        self.assertEqual(len(provenance["recorded"]["environment"][
            "package_sha256"]), 64)
        prior = test_experiment._snapshot(self.target)
        with patch.object(self.experiment, "replay") as replay:
            retained = self.run_action()
            replay.assert_not_called()
        self.assertEqual(retained.status, "retained")
        self.assertEqual(retained.provenance, provenance)
        self.assertEqual(prior, test_experiment._snapshot(self.target))
        refused = self.run_action("error")
        self.assertEqual(refused.status, "error")
        self.assertEqual(prior, test_experiment._snapshot(self.target))
        second = self.run_action("archive")
        self.assertEqual(second.status, "completed", second.error)
        self.assertEqual(prior, test_experiment._snapshot(
            second.archived_workspace))
        self.assertEqual(self.before, test_experiment._snapshot(self.root))
        data = self.experiment.compare().data
        self.assertEqual(data["alignment"]["status"], "complete")
        self.assertEqual(data["replay_execution"],
                         {**second.provenance, "reference_match": "exact"})

    def test_invalid_original_preserves_previous(self):
        first = self.run_action()
        self.assertEqual(first.status, "completed", first.error)
        prior = test_experiment._snapshot(self.target)
        with patch.object(self.experiment, "validate") as validate:
            validate.return_value.raise_for_errors.side_effect = ValueError(
                "Invalid reference")
            result = self.run_action("archive")
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.archived_workspace)
        self.assertEqual(prior, test_experiment._snapshot(self.target))
        self.assertEqual(self.before, test_experiment._snapshot(self.root))

    def test_failed_fresh_replay_preserves_archive_and_partial(self):
        first = self.run_action()
        self.assertEqual(first.status, "completed", first.error)
        prior = test_experiment._snapshot(self.target)

        def fail(destination, **kwargs):
            destination.mkdir()
            (destination / "partial.txt").write_text("Partial",
                                                       encoding="utf-8")
            raise RuntimeError("Execution failed")

        with patch.object(self.experiment, "replay", side_effect=fail):
            result = self.run_action("archive")
        self.assertEqual(result.status, "error")
        self.assertEqual(prior, test_experiment._snapshot(
            result.archived_workspace))
        self.assertTrue((self.target / "partial.txt").is_file())
        self.assertEqual(self.before, test_experiment._snapshot(self.root))

    def test_reference_receipt_and_gui_inspection(self):
        receipt = publish_receipt(self.root, self.report, ["REVIEW: test"])
        original = receipt.read_bytes()
        self.assertEqual(inspect_receipt(self.root, self.report)["status"],
                         "unchanged")
        first = self.run_action()
        self.assertEqual(first.status, "completed", first.error)
        self.fixture.engine.resume_project(str(self.root))
        self.fixture.engine.close()
        report = self.experiment.validate()
        self.assertTrue(report.valid, report.issues)
        self.assertEqual(inspect_receipt(self.root, report)["status"],
                         "inspection_only")
        data = self.experiment.compare().data
        self.assertEqual(data["reference_publication"]["status"],
                         "inspection_only")
        self.assertEqual(data["inspection"]["status"], "inspection_only")
        self.assertEqual(data["alignment"]["status"], "diverged")
        self.assertEqual(data["replay_execution"]["reference_match"],
                         "prefix")
        self.assertEqual(receipt.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            publish_receipt(self.root, report, [])
        receipt.write_text('{"format_version":1,"format_version":1}',
                           encoding="utf-8")
        self.assertEqual(inspect_receipt(self.root, report)["status"],
                         "error")

    def test_missing_and_changed_execution_receipt(self):
        with self.fixture._model():
            ordinary = self.experiment.replay()
        report = ordinary.validation
        self.assertEqual(execution_provenance(self.target, report),
                         {"status": "unavailable"})
        fresh = self.run_action("archive")
        self.assertEqual(fresh.status, "completed", fresh.error)
        path = self.target / "logs/system_testing_execution.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["replayed"]["prefix_sha256"] = "0" * 64
        path.write_text(json.dumps(data), encoding="utf-8")
        result = execution_provenance(self.target, fresh.result.validation)
        self.assertEqual(result["status"], "changed")

    def test_invalid_policy_and_archive_destination(self):
        with self.assertRaises(ValueError):
            self.experiment.run_replay(existing_replay="replace")
        first = self.run_action()
        self.assertEqual(first.status, "completed", first.error)
        prior = test_experiment._snapshot(self.target)
        result = self.experiment.run_replay(
            existing_replay="archive", archive_directory=self.root.parent)
        self.assertEqual(result.status, "error")
        self.assertEqual(prior, test_experiment._snapshot(self.target))


if __name__ == "__main__":
    unittest.main()
