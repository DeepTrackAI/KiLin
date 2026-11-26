"""Represent one immutable reference workspace for system testing.

Key Features
------------
- Inspect reference recordings without loading a model or starting the GUI.
- Validate on demand and reconstruct supported chronological commands.
- Never modify, repair, or migrate the reference workspace.

Module Structure
----------------
- ``Experiment`` provides the public entry point for reference inspection.

Examples
--------
>>> from pathlib import Path
>>> from kilin.system_testing import Experiment
>>> experiment = Experiment(Path("workspaces/010_original"))
>>> # report = experiment.validate()
>>> # if report.valid:
>>> #     operations = experiment.timeline()

"""

from pathlib import Path

from .timeline import TimelineOperation
from .traces import read_recording
from .validation import ValidationReport, validate_recording

__all__ = ["Experiment"]


class Experiment:
    """Read-only system-testing reference experiment.

    Parameters
    ----------
    workspace: str | pathlib.Path
        Reference project directory. The path is resolved without creating
        missing files. Validation is explicit, not performed on construction.

    Attributes
    ----------
    workspace: pathlib.Path
        Resolved reference workspace path.

    Notes
    -----
    Each inspection reads a fresh snapshot. Close the original project and
    keep its files unchanged during inspection. No writer lock is acquired;
    acquiring one could modify the reference. Replay and comparison are not
    implemented by this interface.

    """

    def __init__(self, workspace: str | Path) -> None:
        """Store the original workspace location without opening it.

        Parameters
        ----------
        workspace: str | pathlib.Path
            Reference project directory.

        """
        self.workspace = Path(workspace).expanduser().resolve()

    def validate(self) -> ValidationReport:
        """Inspect the current reference and collect detectable issues.

        Returns
        -------
        ValidationReport
            Complete inspection result with blocking issues and limitations.

        """
        return validate_recording(read_recording(self.workspace))

    def timeline(self) -> tuple[TimelineOperation, ...]:
        """Return chronological operations only for a valid reference.

        Returns
        -------
        tuple[TimelineOperation, ...]
            Chronologically ordered supported commands with diagnostics.

        Raises
        ------
        InvalidRecordingError
            If validation detects a missing or inconsistent recording.

        """
        report = self.validate()
        report.raise_for_errors()
        return report.operations
