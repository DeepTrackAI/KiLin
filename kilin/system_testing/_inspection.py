"""Identify additional closed reference inspection sessions.

Key Features
------------
- Recognize only validated open, select and close operation suffixes.
- Retain strict alignment and distinguish subsequent substantive changes.

Module Structure
----------------
- ``inspection_suffix`` validates a closed tail of view-only commands.
- ``note_inspection`` annotates a strictly divergent comparison.

Examples
--------
>>> from kilin.system_testing import Experiment
>>> # Experiment("010_original").compare().data.get("inspection")

"""

from .validation import ValidationReport

__all__ = []


def inspection_suffix(report: ValidationReport,
                      count: int) -> list[dict] | None:
    """Recognize a nonempty closed tail of ordinary inspection commands.

    Parameters
    ----------
    report: ValidationReport
        Closed recording to inspect.
    count: int
        Number of commands in the matched or published prefix.

    Notes
    -----
    Callers must validate the entire recording first. No substantive command
    or incomplete session qualifies. Selection may change the viewed chat
    but not its saved turns. The original operation history is not trimmed.

    """
    operations = report.operations
    if not report.valid or not 0 < count < len(operations):
        return None
    if operations[count - 1].kind != "close_project":
        return None
    # Source-change notices can be outside the open-command span.
    for event in report.events:
        if (event.sequence > operations[count - 1].end_sequence and
                event.event_type == "gui.sources.changes.found" and
                any(event.details.get(key) != [] for key in
                    ("added", "changed", "deleted", "edited"))):
            return None
    opened = False
    tail = []
    for op in operations[count:]:
        if op.status != "completed":
            return None
        if op.kind == "open_project":
            if (opened or op.request.get("reason") != "resume" or
                    "relocation" in op.request):
                return None
            opened = True
        elif op.kind == "select_conversation":
            if not opened:
                return None
        elif op.kind == "close_project":
            if not opened:
                return None
            opened = False
        else:
            return None
        tail.append({"kind": op.kind, "sequence": op.start_sequence,
                     "end_sequence": op.end_sequence})
    return None if opened else tail


def note_inspection(data: dict, report: ValidationReport) -> dict | None:
    """Describe an inspection-only tail absent from an existing replay.

    Parameters
    ----------
    data: dict
        Independently aligned comparison data.
    report: ValidationReport
        Validation of the same closed original snapshot.

    Notes
    -----
    Require independently valid recordings and exact prefix alignment. Keep
    the strict comparison status and unmatched operations unchanged. A
    substantive extension, failed command or replay-side mismatch is never
    downgraded to inspection-only.

    """
    alignment = data["alignment"]
    if (not report.valid or not data["original"]["valid"] or
            not data["replayed"]["valid"] or
            alignment["status"] != "diverged" or
            alignment["reason"] != "Recorded command counts differ." or
            alignment["replayed_unmatched"]):
        return None
    count = alignment["matched_operations"]
    if (report.event_count != data["original"]["event_count"] or
            len(report.operations) != len(data["original"]["operations"]) or
            count != len(data["replayed"]["operations"])):
        return None
    tail = inspection_suffix(report, count)
    if not tail:
        return None
    if [op["sequence"] for op in tail] != [
            op["sequence"] for op in alignment["original_unmatched"]]:
        return None
    return {"status": "inspection_only", "operations": tail,
            "message": "The existing replay matches the recorded prefix. "
                       "The original has additional closed inspection "
                       "sessions. A fresh replay includes these sessions. "
                       "Strict alignment remains divergent."}


def relocation_event_differences(data: dict, before: ValidationReport,
                                 after: ValidationReport) -> list[str]:
    """Identify fully explained event-count deltas for normalized renames.

    Parameters
    ----------
    data: dict
        Complete independently aligned comparison.
    before: ValidationReport
        Validated original recording.
    after: ValidationReport
        Validated replay recording.

    Notes
    -----
    Only classify deltas when all counted extra events are accounted for
    by recorded rename markers or empty checks immediately after a mapped
    reopening. Other source checks or unmatched commands are not ignored.

    """
    if (not before.valid or not after.valid or
            data["alignment"]["status"] != "complete"):
        return []
    expected = {"gui.project.rename.requested": 0,
                "gui.project.rename.completed": 0,
                "gui.sources.changes.found": 0}
    for i, row in enumerate(data["operations"]):
        move = row.get("relocation")
        if not move or move["physical_rename_replayed"]:
            continue
        if move["original"]["provenance"] == "recorded":
            expected["gui.project.rename.requested"] -= 1
            expected["gui.project.rename.completed"] -= 1
        a, b = before.operations[i], after.operations[i]
        # A successful opening may be followed by a source check outside
        # its command span. Only an explicitly empty diagnostic qualifies.
        counts = []
        for report, op in ((before, a), (after, b)):
            end = (report.operations[i + 1].start_sequence
                   if i + 1 < len(report.operations) else
                   report.event_count + 1)
            notices = [e for e in report.events if
                       op.start_sequence <= e.sequence < end and
                       e.event_type == "gui.sources.changes.found"]
            if any(any(e.details.get(k) != [] for k in
                       ("added", "changed", "deleted", "edited"))
                   for e in notices):
                return []
            counts.append(len(notices))
        expected["gui.sources.changes.found"] += counts[1] - counts[0]
    old = data["original"]["event_counts"]
    new = data["replayed"]["event_counts"]
    return [kind for kind, delta in expected.items() if delta and
            new.get(kind, 0) - old.get(kind, 0) == delta]
