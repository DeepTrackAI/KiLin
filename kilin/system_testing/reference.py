"""Record and inspect reference publication without freezing GUI history.

Key Features
------------
- An optional receipt fingerprints the published trace prefix and artifacts.
- Closed inspection sessions may extend a reference without rewriting it.
- Strict KíLín comparison remains authoritative; benign tails are annotated.

Module Structure
----------------
- Publication receipt creation and verification.
- Publication receipt integrity and trace-prefix fingerprints.

Examples
--------
>>> receipt_path(Path('workspaces/010_hello_world_original')).name
'010_hello_world_reference.json'

"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from ._inspection import inspection_suffix
from .traces import TraceEvent, read_json
from .validation import ValidationReport

__all__ = ["receipt_path", "publish_receipt", "inspect_receipt"]


def receipt_path(original: Path) -> Path:
    """Return the optional publication receipt beside a reference.

    Parameters
    ----------
    original: pathlib.Path
        Reference directory named with the ``_original`` suffix.

    """
    original = Path(original)
    if not original.name.endswith("_original"):
        raise ValueError("Expected a reference workspace name.")
    return original.with_name(
        original.name.removesuffix("_original") + "_reference.json")


def _prefix_digest(events: tuple[TraceEvent, ...], count: int) -> str:
    """Hash exact decoded event payloads, not mutable trace containers.

    Parameters
    ----------
    events: tuple[TraceEvent, ...]
        Decoded chronological trace events.
    count: int
        Number of initial events to fingerprint.

    """
    prefix = [e for e in events if e.sequence <= count]
    if [e.sequence for e in prefix] != list(range(1, count + 1)):
        raise ValueError("Published trace prefix is not contiguous.")
    digest = sha256()
    for event in prefix:
        payload = {"path": event.path, "metadata": event.metadata,
                   "details": event.details,
                   "exact_content": event.exact_content}
        digest.update(json.dumps(payload, ensure_ascii=False,
                                 sort_keys=True, allow_nan=False,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _artifacts(original: Path) -> dict[str, str]:
    """Fingerprint managed source, template and output files only.

    Parameters
    ----------
    original: pathlib.Path
        Closed reference workspace, with regular managed directories.

    """
    files = {}
    for folder in ("sources", "templates", "outputs"):
        root = original / folder
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"Expected a real artifact directory: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Linked artifact: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"Nonregular artifact: {path}")
            # Finder and similar desktop metadata are not experiment data.
            if path.name == ".DS_Store":
                continue
            digest = sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            files[path.relative_to(original).as_posix()] = digest.hexdigest()
    return files


def publish_receipt(original: Path, report: ValidationReport,
                    coverage: list[str]) -> Path:
    """Publish a new receipt only after successful reference validation.

    Parameters
    ----------
    original: pathlib.Path
        Published, closed reference workspace.
    report: ValidationReport
        Fresh validation of the same reference. The caller must ensure no
        intervening writes or concurrent GUI access.
    coverage: list[str]
        Generator observations, including explicit missing coverage.

    Notes
    -----
    The receipt describes the reference at publication. It is never updated
    by a replay or ordinary GUI inspection. A preexisting receipt is not
    overwritten, and a failed publication leaves the closed reference intact.

    """
    report.raise_for_errors()
    original = Path(original)
    if original.is_symlink():
        raise ValueError("Reference must not be a symbolic link.")
    if (not isinstance(coverage, list) or
            any(not isinstance(item, str) for item in coverage)):
        raise ValueError("Coverage must be a list of strings.")
    if report.workspace != original.resolve():
        raise ValueError("Receipt must describe this validated reference.")
    target = receipt_path(original)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    data = {"format_version": 1, "reference": original.name,
            "event_count": report.event_count,
            "operation_count": len(report.operations),
            "prefix_sha256": _prefix_digest(report.events,
                                            report.event_count),
            "artifacts": _artifacts(original), "coverage": list(coverage)}
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                dir=target.parent) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2,
                      allow_nan=False)
            stream.write("\n")
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def inspect_receipt(original: Path, report: ValidationReport) -> dict:
    """Describe preservation or extension of an optional published baseline.

    Parameters
    ----------
    original: pathlib.Path
        Closed original workspace.
    report: ValidationReport
        Fresh validation of this same workspace.

    Notes
    -----
    Missing receipts are normal for manual and older references. A changed
    prefix or artifact is a review finding, not a replacement for validation.
    The original can intentionally be extended through recorded commands.

    """
    original = Path(original)
    if not original.name.endswith("_original"):
        return {"status": "unavailable"}
    path = receipt_path(original)
    if not path.exists() and not path.is_symlink():
        return {"status": "unavailable"}
    try:
        if original.is_symlink() or path.is_symlink() or not path.is_file():
            raise ValueError(f"Nonregular receipt: {path}")
        data = read_json(path.parent, path.name)
        if (type(data["format_version"]) is not int or
                data["format_version"] != 1 or
                data["reference"] != original.name):
            raise ValueError("Receipt does not identify this reference.")
        count, operations = data["event_count"], data["operation_count"]
        if (type(count) is not int or type(operations) is not int or
                count < 1 or operations < 1):
            raise ValueError("Invalid publication counts.")
        if (not isinstance(data["coverage"], list) or
                any(not isinstance(item, str) for item in data["coverage"])):
            raise ValueError("Invalid publication coverage.")
        if not report.valid:
            return {"status": "not_validated"}
        if report.workspace != original.resolve():
            raise ValueError("Validation report describes another reference.")
        result = {"status": "unchanged", "published_events": count,
                  "published_operations": operations,
                  "additional_events": report.event_count - count,
                  "coverage": data["coverage"]}
        if count > report.event_count or (
                _prefix_digest(report.events, count) != data["prefix_sha256"]
        ):
            return {**result, "status": "changed",
                    "message": "Published trace prefix differs."}
        if not isinstance(data["artifacts"], dict):
            raise ValueError("Invalid published artifact inventory.")
        current = _artifacts(original)
        changed = sorted(k for k in current.keys() | data["artifacts"].keys()
                         if current.get(k) != data["artifacts"].get(k))
        if changed:
            return {**result, "status": "changed", "artifacts": changed,
                    "message": "Published managed artifacts differ."}
        # A receipt must describe a closed command prefix, not a count
        # chosen from inside a later inspection session.
        if (operations > len(report.operations) or
                report.operations[operations - 1].kind != "close_project" or
                report.operations[operations - 1].end_sequence > count or
                (operations < len(report.operations) and
                 report.operations[operations].start_sequence <= count)):
            return {**result, "status": "changed",
                    "message": "Published command boundary differs."}
        if len(report.operations) == operations:
            if report.event_count != count:
                result.update(status="extended",
                              message="Additional diagnostic events.")
        elif len(report.operations) > operations:
            tail = inspection_suffix(report, operations)
            if tail:
                result.update(status="inspection_only", operations=tail)
            else:
                result.update(status="extended",
                              message="Additional recorded commands.")
        else:
            result.update(status="changed",
                          message="Published operation count decreased.")
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return {"status": "error", "message": str(error)}
