"""Manage complete fresh replays without replacing reference recordings.

Key Features
------------
- Keep, archive or reject an existing replay through an explicit policy.
- Preserve old replays and partial failures without copy-and-delete fallback.
- Record actual replay execution provenance separately from project traces.

Module Structure
----------------
- ``ManagedReplay`` reports the requested action and any resulting error.
- ``run_replay`` applies the lifecycle policy around ordinary replay.
- ``execution_provenance`` reads the optional historical execution receipt.

Examples
--------
>>> from kilin.system_testing import Experiment
>>> # Experiment("workspaces/010_original").run_replay(
>>> #     existing_replay="archive")

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
from tempfile import NamedTemporaryFile, mkdtemp
from typing import TYPE_CHECKING

import kilin

from .reference import _prefix_digest
from .replay import ProgressCallback, ReplayError, ReplayResult
from .traces import read_json, safe_path
from .validation import ValidationReport

if TYPE_CHECKING:
    from .experiment import Experiment

__all__ = ["ManagedReplay", "run_replay", "execution_provenance"]

_RECEIPT = "logs/system_testing_execution.json"


@dataclass(frozen=True, slots=True)
class ManagedReplay:
    """One managed replay action, with preservation and provenance details.

    Attributes
    ----------
    workspace: pathlib.Path
        Designated sibling replay workspace.
    status: str
        ``completed``, ``retained`` or ``error``.
    archived_workspace: pathlib.Path or None
        Preserved previous replay, if moved.
    result: ReplayResult or None
        Successful new replay, never fabricated for a retained recording.
    error: str or None
        Actual failed action, if any.
    provenance: dict
        Execution receipt or explicit unavailable/error information.
    completed_operations: int or None
        Reproduced commands, if established by the actual execution.
    sequence: int or None
        Original command at a replay failure, if known.

    """

    workspace: Path
    status: str
    archived_workspace: Path | None
    result: ReplayResult | None
    error: str | None
    provenance: dict
    completed_operations: int | None = None
    sequence: int | None = None


def _environment() -> dict:
    """Identify the executing code and runtime, without guessing a revision."""
    source = Path(kilin.__file__).resolve().parent
    digest = sha256()
    # Include package resources as well as Python modules. A checkout hash
    # alone cannot identify uncommitted code or modified packaged defaults.
    for path in sorted(source.rglob("*")):
        if (not path.is_file() or "__pycache__" in path.parts or
                path.suffix in (".pyc", ".pyo") or path.name == ".DS_Store"):
            continue
        digest.update(path.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path.read_bytes()).digest())
    result = {"kilin_version": kilin.__version__,
              "package_sha256": digest.hexdigest(),
              "python": platform.python_version(),
              "platform": platform.platform(), "machine": platform.machine(),
              "git_revision": None, "git_dirty": None}
    # A wheel installed inside some unrelated Git project is not its source.
    parent = source.parent
    if (parent / ".git").exists() and (parent / "pyproject.toml").is_file():
        try:
            command = ["git", "-C", str(parent)]
            result["git_revision"] = subprocess.check_output(
                command + ["rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL, timeout=5).strip()
            result["git_dirty"] = bool(subprocess.check_output(
                command + ["status", "--porcelain", "--", "kilin"],
                text=True, stderr=subprocess.DEVNULL, timeout=5).strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def _fingerprint(report: ValidationReport) -> dict:
    """Identify the exact validated command/event prefix of a recording.

    Parameters
    ----------
    report: ValidationReport
        Closed recording snapshot.

    """
    report.raise_for_errors()
    return {"event_count": report.event_count,
            "operation_count": len(report.operations),
            "prefix_sha256": _prefix_digest(report.events,
                                            report.event_count)}


def _archive(destination: Path, directory: Path) -> Path:
    """Move a previous replay into a reserved, separate archive directory.

    Parameters
    ----------
    destination: pathlib.Path
        Existing regular workspace with a ``_replay`` suffix.
    directory: pathlib.Path
        Archive root outside the workspace discovery directory.

    """
    if (not destination.name.endswith("_replay") or
            destination.is_symlink() or not destination.is_dir()):
        raise ValueError(f"Expected a real replay directory: {destination}")
    if (directory.is_symlink() or
            (directory.exists() and not directory.is_dir())):
        raise ValueError(f"Nonregular replay archive: {directory}")
    directory = directory.resolve()
    if (directory.is_relative_to(destination.parent) or
            destination.parent.is_relative_to(directory)):
        raise ValueError("Replay archive must be separate from workspaces.")
    directory.mkdir(parents=True, exist_ok=True)
    folder = Path(mkdtemp(prefix=destination.name + "-", dir=directory))
    target = folder / destination.name
    try:
        destination.rename(target)
    except OSError:
        folder.rmdir()
        raise
    return target


def _write_receipt(workspace: Path, data: dict) -> None:
    """Publish provenance exclusively after a successfully closed replay.

    Parameters
    ----------
    workspace: pathlib.Path
        Newly generated replay workspace.
    data: dict
        Execution provenance bound to its validated event prefix.

    """
    target = safe_path(workspace, _RECEIPT)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                dir=target.parent) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, allow_nan=False,
                      indent=2)
            stream.write("\n")
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def execution_provenance(workspace: Path, report: ValidationReport) -> dict:
    """Read an optional historical replay receipt and verify its trace prefix.

    Parameters
    ----------
    workspace: pathlib.Path
        Closed replay workspace.
    report: ValidationReport
        Fresh validation of this workspace.

    Returns
    -------
    dict
        Recorded execution with verified prefix, or explicit unavailable,
        unverified, changed or error status. Missing older receipts are
        never replaced with information about the inspection environment.

    """
    workspace = Path(workspace)
    try:
        path = safe_path(workspace, _RECEIPT)
        if not path.exists():
            return {"status": "unavailable"}
        data = read_json(workspace, _RECEIPT)
        if (type(data["format_version"]) is not int or
                data["format_version"] != 1 or
                data["status"] != "completed"):
            raise ValueError("Invalid execution receipt format.")
        if report.workspace != workspace.resolve():
            raise ValueError("Validation describes another workspace.")
        prefix = data["replayed"]
        count = prefix["event_count"]
        operations = prefix["operation_count"]
        if (type(count) is not int or count < 1 or
                type(operations) is not int or operations < 1):
            raise ValueError("Invalid replay prefix counts.")
        if not report.valid:
            return {"status": "not_validated", "recorded": data}
        if (count > report.event_count or
                _prefix_digest(report.events, count) != prefix["prefix_sha256"]
                or operations > len(report.operations) or
                report.operations[operations - 1].kind != "close_project" or
                report.operations[operations - 1].end_sequence > count or
                (operations < len(report.operations) and
                 report.operations[operations].start_sequence <= count)):
            return {"status": "changed", "recorded": data}
        return {"status": "recorded", "recorded": data,
                "additional_events": report.event_count - count}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return {"status": "error", "message": str(error)}


def reference_execution_match(provenance: dict,
                              report: ValidationReport) -> str:
    """Compare a recorded replay input to the current reference prefix.

    Parameters
    ----------
    provenance: dict
        Independently verified execution receipt observation.
    report: ValidationReport
        Current reference recording, which may have been extended.

    """
    if provenance.get("status") != "recorded" or not report.valid:
        return "unavailable"
    try:
        prefix = provenance["recorded"]["reference"]
        count, operations = prefix["event_count"], prefix["operation_count"]
        if (type(count) is not int or count < 1 or
                type(operations) is not int or operations < 1):
            return "error"
        if (count > report.event_count or operations > len(report.operations)
                or report.operations[operations - 1].kind != "close_project"
                or report.operations[operations - 1].end_sequence > count or
                (operations < len(report.operations) and
                 report.operations[operations].start_sequence <= count) or
                _prefix_digest(report.events, count) !=
                prefix["prefix_sha256"]):
            return "changed"
        if (count == report.event_count and
                operations == len(report.operations)):
            return "exact"
        return "prefix"
    except (ValueError, KeyError, TypeError, AttributeError):
        return "error"


def run_replay(experiment: Experiment, destination: str | Path | None = None,
               *, existing_replay: str = "retain",
               archive_directory: str | Path | None = None,
               progress: ProgressCallback | None = None) -> ManagedReplay:
    """Apply an explicit existing-replay policy and record actual execution.

    Parameters
    ----------
    experiment: Experiment
        Closed original workspace; validated before any archive or replay.
    destination: str or pathlib.Path or None, optional
        Separate sibling replay directory. Defaults to the usual suffix.
    existing_replay: str, optional
        ``retain`` (default), ``archive`` or ``error``. Archiving preserves
        the old workspace even if execution of the new replay fails.
    archive_directory: str or pathlib.Path or None, optional
        Archive root separate from workspace discovery. Defaults to
        ``_replay_archive`` beside the workspace discovery directory.
    progress: ProgressCallback or None, optional
        Forwarded to ordinary replay when a fresh execution is requested.

    Returns
    -------
    ManagedReplay
        Action outcome, optional archived path, replay result and provenance.
        Operational failures are represented explicitly; invalid policy or
        destination arguments raise ValueError.

    Notes
    -----
    The original and replay must be closed and have no concurrent writers.
    The low-level ``Experiment.replay`` still requires an absent destination.
    A failed replay remains available. Only a successful fresh replay is
    given a persistent execution receipt; failed attempts retain their
    actual execution context in the returned action record.

    """
    from .comparison import _replay_path

    if existing_replay not in ("retain", "archive", "error"):
        raise ValueError(
            "Choose retain, archive, or error for existing replay.")
    target = _replay_path(experiment.workspace, destination)
    archived = None
    provenance = {"status": "not_executed"}
    completed = sequence = None
    try:
        report = experiment.validate()
        report.raise_for_errors()
        if target.exists():
            if existing_replay == "error":
                raise FileExistsError(f"Replay already exists: {target}")
            if existing_replay == "retain":
                from .experiment import Experiment

                retained = Experiment(target).validate()
                return ManagedReplay(target, "retained", None, None, None,
                                     execution_provenance(target, retained))
        provenance = {"format_version": 1, "status": "running",
                      "reference": _fingerprint(report),
                      "environment": _environment(),
                      "started_at_utc": datetime.now(timezone.utc).isoformat()}
        if target.exists():
            directory = (Path(archive_directory).expanduser() if
                         archive_directory is not None else
                         target.parent.parent / "_replay_archive")
            archived = _archive(target, directory)
        result = experiment.replay(target, progress=progress)
        completed = result.operation_count
        provenance.update(status="completed", replayed=_fingerprint(
            result.validation), finished_at_utc=datetime.now(
                timezone.utc).isoformat())
        try:
            _write_receipt(target, provenance)
        except (OSError, ValueError, TypeError) as error:
            return ManagedReplay(target, "completed", archived, result, None,
                                 {"status": "error", "message": str(error),
                                  "recorded": provenance}, completed)
        return ManagedReplay(target, "completed", archived, result, None,
                             {"status": "recorded", "recorded": provenance,
                              "additional_events": 0}, completed)
    except Exception as error:
        if isinstance(error, ReplayError):
            completed, sequence = error.completed_operations, error.sequence
        if provenance["status"] == "running":
            provenance.update(status="error", finished_at_utc=datetime.now(
                timezone.utc).isoformat())
        return ManagedReplay(target, "error", archived, None, str(error),
                             provenance, completed, sequence)
