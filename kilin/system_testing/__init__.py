"""Expose the read-only KíLín system-testing inspection API.

Key Features
------------
- Reference workspace validation and chronological operation inspection.
- Model-independent, read-only checks of current-format recordings.

Module Structure
----------------
- ``Experiment``: reference experiment entry point.
- ``TraceEvent``, ``TimelineOperation``: decoded history records.
- ``ValidationIssue``, ``ValidationReport``, ``InvalidRecordingError``:
  structured diagnostics and rejection of invalid reference recordings.

Examples
--------
>>> from kilin.system_testing import Experiment
>>> # Experiment("workspaces/010_original").validate()

"""

from .experiment import Experiment
from .timeline import TimelineOperation
from .traces import TraceEvent, ValidationIssue
from .validation import InvalidRecordingError, ValidationReport

__all__ = [
    "Experiment",
    "TraceEvent",
    "TimelineOperation",
    "ValidationIssue",
    "ValidationReport",
    "InvalidRecordingError",
]
