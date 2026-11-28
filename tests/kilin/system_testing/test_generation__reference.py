"""Test publication fingerprints and inspection-only history extensions."""

# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0103:invalid-name
# pylint: disable=protected-access

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from kilin.system_testing.generation import check_output
from kilin.system_testing.reference import (
    inspect_receipt, publish_receipt, receipt_path,
)
from kilin.system_testing._inspection import inspection_suffix, note_inspection


def recording(workspace, kinds, *, valid=True):
    """Construct a minimal validated-report double for policy tests."""
    ops, events = [], []
    for i, kind in enumerate(kinds, 1):
        request = {"reason": "resume"} if kind == "open_project" else {}
        ops.append(SimpleNamespace(kind=kind, request=request,
                                   status="completed", start_sequence=i,
                                   end_sequence=i))
        events.append(SimpleNamespace(sequence=i, event_type=kind,
                                      path="project.json", metadata={},
                                      details={}, exact_content=None))

    def check():
        if not valid:
            raise ValueError("Invalid recording")

    return SimpleNamespace(workspace=workspace.resolve(), valid=valid,
                           event_count=len(events), events=events,
                           operations=ops, raise_for_errors=check)


class TestReferenceTools(unittest.TestCase):
    """Preserve the published prefix while allowing recorded inspection."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.original = Path(self.temp.name) / "010_original"
        for name in ("sources", "templates", "outputs"):
            (self.original / name).mkdir(parents=True)
        self.source = self.original / "sources" / "test.txt"
        self.source.write_text("reference", encoding="utf-8")
        self.base = ["create_project", "ask", "close_project"]
        self.tail = ["open_project", "select_conversation", "close_project"]
        self.report = recording(self.original, self.base)

    def test_missing_receipt(self):
        self.assertEqual(inspect_receipt(self.original, self.report),
                         {"status": "unavailable"})

    def test_receipt_and_append(self):
        target = publish_receipt(self.original, self.report, ["REVIEW: test"])
        self.assertEqual(target, receipt_path(self.original))
        before = target.read_bytes()
        result = inspect_receipt(self.original, self.report)
        self.assertEqual(result["status"], "unchanged")
        self.assertEqual(result["coverage"], ["REVIEW: test"])
        extended = recording(self.original, self.base + self.tail * 2)
        result = inspect_receipt(self.original, extended)
        self.assertEqual(result["status"], "inspection_only")
        self.assertEqual(len(result["operations"]), 6)
        self.assertEqual(target.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            publish_receipt(self.original, extended, [])
        self.assertEqual(target.read_bytes(), before)

    def test_changed_artifact(self):
        publish_receipt(self.original, self.report, [])
        self.source.write_text("changed", encoding="utf-8")
        result = inspect_receipt(self.original, self.report)
        self.assertEqual(result["status"], "changed")
        self.assertEqual(result["artifacts"], ["sources/test.txt"])

    def test_changed_trace(self):
        publish_receipt(self.original, self.report, [])
        self.report.events[1].details["question"] = "changed"
        self.assertEqual(inspect_receipt(self.original, self.report)["status"],
                         "changed")

    def test_substantive_append(self):
        publish_receipt(self.original, self.report, [])
        report = recording(self.original, self.base +
                           ["open_project", "ask", "close_project"])
        self.assertEqual(inspect_receipt(self.original, report)["status"],
                         "extended")
        self.assertIsNone(inspection_suffix(report, 3))

    def test_invalid_and_incomplete_tails(self):
        for tail, valid in [(self.tail, False), (self.tail[:-1], True),
                            (["select_conversation"], True),
                            (["open_project", "add_sources",
                              "close_project"], True)]:
            with self.subTest(tail=tail, valid=valid):
                report = recording(self.original, self.base + tail,
                                   valid=valid)
                self.assertIsNone(inspection_suffix(report, 3))
        report = recording(self.original, self.base + self.tail)
        report.operations[-1].status = "failed"
        self.assertIsNone(inspection_suffix(report, 3))

    def test_relocation_not_inspection(self):
        report = recording(self.original, self.base + self.tail)
        report.operations[3].request["relocation"] = {}
        self.assertIsNone(inspection_suffix(report, 3))

    def test_source_change_notice(self):
        report = recording(self.original, self.base + self.tail)
        notice = SimpleNamespace(sequence=4,
                                 event_type="gui.sources.changes.found",
                                 details=dict(added=[], changed=[],
                                              deleted=[], edited=[]))
        report.events.append(notice)
        self.assertIsNotNone(inspection_suffix(report, 3))
        notice.details["added"] = ["extra.txt"]
        self.assertIsNone(inspection_suffix(report, 3))
        notice.details = {}
        self.assertIsNone(inspection_suffix(report, 3))

    def test_strict_alignment_preserved(self):
        report = recording(self.original, self.base + self.tail)
        alignment = dict(status="diverged", matched_operations=3,
                         reason="Recorded command counts differ.",
                         original_unmatched=[{"sequence": i}
                                             for i in (4, 5, 6)],
                         replayed_unmatched=[])
        data = dict(alignment=alignment,
                    original=dict(valid=True, operations=self.base+self.tail,
                                  event_count=6),
                    replayed=dict(valid=True, operations=self.base))
        self.assertEqual(note_inspection(data, report)["status"],
                         "inspection_only")
        self.assertEqual(alignment["status"], "diverged")
        data["original"]["event_count"] = 7
        self.assertIsNone(note_inspection(data, report))
        data["original"]["event_count"] = 6
        alignment["replayed_unmatched"] = [{"sequence": 4}]
        self.assertIsNone(note_inspection(data, report))
        alignment["replayed_unmatched"] = []
        alignment["reason"] = "Different command."
        self.assertIsNone(note_inspection(data, report))

    def test_wrong_workspace_and_invalid_receipt(self):
        self.report.workspace = self.original.parent
        with self.assertRaises(ValueError):
            publish_receipt(self.original, self.report, [])
        receipt_path(self.original).write_text("{", encoding="utf-8")
        self.assertEqual(inspect_receipt(self.original, self.report)["status"],
                         "error")

    def test_output_coverage(self):
        expected = "outputs/paper_summary.md"
        (self.original / expected).write_text("report", encoding="utf-8")
        summary = dict(name="generate_paper_summary", saved=expected,
                       valid=True, template="popular_science_summary")

        def check(result):
            return check_output(
                self.original, result, label="Q02",
                tool="generate_paper_summary", field="saved",
                expected=expected, template="popular_science_summary",
                require_valid=True)

        self.assertFalse(any(s.startswith("REVIEW:") for s in check(summary)))
        for change in (dict(valid=False), dict(status="incomplete"),
                       dict(name="writing_check"), dict(saved=None),
                       dict(template="other")):
            with self.subTest(change=change):
                self.assertTrue(check({**summary, **change})[0].startswith(
                    "REVIEW:"))
        (self.original / expected).write_text("", encoding="utf-8")
        self.assertTrue(check(summary)[0].startswith("REVIEW:"))
