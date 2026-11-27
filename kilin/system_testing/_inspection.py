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
