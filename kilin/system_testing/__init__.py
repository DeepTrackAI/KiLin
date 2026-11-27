"""Expose the KíLín system-testing inspection, replay and comparison API.

Key Features
------------
- Reference workspace validation and chronological operation inspection.
- Model-independent, read-only checks of current-format recordings.
- Independent reproduction of validated workspace histories.
- Independent quantitative comparison of original and replayed workspaces.
- Recipe-driven reference generation and reproducible batch reporting.

Module Structure
----------------
- ``Experiment``: reference experiment entry point.
- ``TraceEvent``, ``TimelineOperation``: decoded history records.
- ``ValidationIssue``, ``ValidationReport``, ``InvalidRecordingError``:
  structured diagnostics and rejection of invalid reference recordings.
- ``ReplayResult``, ``ReplayDifference``, ``ReplayError``: replay outcomes.
- ``Comparison``: structured observations and saved reports.
- ``ManagedReplay``: retained or fresh replay outcome and provenance.
- Reference receipt publication and inspection functions.
- ``generation``, ``batch``, ``analysis``: optional system-testing
  workflows, imported explicitly when needed.

Examples
--------
>>> from kilin.system_testing import Experiment
>>> # Experiment("workspaces/010_original").validate()

"""

from .comparison import Comparison
from .experiment import Experiment
from .replay import ReplayDifference, ReplayError, ReplayResult
from .reference import inspect_receipt, publish_receipt, receipt_path
from .replay_management import ManagedReplay, execution_provenance
from .timeline import TimelineOperation
from .traces import TraceEvent, ValidationIssue
from .validation import InvalidRecordingError, ValidationReport

__all__ = [
    "Experiment",
    "Comparison",
    "TraceEvent",
    "TimelineOperation",
    "ValidationIssue",
    "ValidationReport",
    "InvalidRecordingError",
    "ReplayDifference",
    "ReplayError",
    "ReplayResult",
    "ManagedReplay",
    "receipt_path",
    "publish_receipt",
    "inspect_receipt",
    "execution_provenance",
]
