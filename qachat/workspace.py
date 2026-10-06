"""Copy selected local sources into a managed multi-source workspace.

``prepare_sources_workspace`` accepts local files or directories, prepares a
project profile, copies selected content into ``sources``, and records source
origins in ``sources.json``. Source names are disambiguated when needed.
Directory inputs use repository scanning; individual-file inputs have a
separate extension-and-size check.

This API is cumulative: adding or reselecting sources copies their current
selected files but does not remove old copies merely because the originals
were deleted or deselected. It differs from the single-source mirroring API
in ``qachat.project``. A workspace organizes local files; it is not an
execution sandbox or a complete defense against hostile filesystem changes.
Preparation can leave earlier changes in place if a later copy fails.

Key Features
------------

- **Multiple origins**

    Keep source-origin records and allocate noncolliding names within one
    sources directory.

- **Selected copying**

    Use profile selection rules and optional per-source selections for
    directory inputs.

- **Profile integration**

    Initialize a managed project and apply supported configuration updates.

- **Trace integration**

    Emit workspace preparation and source-copy events through the supplied
    sink.

Module Structure
----------------

- ``prepare_sources_workspace``

    Add sources to one project without changing their original files.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import qachat
>>> project = qachat.prepare_sources_workspace(
...     ["/path/to/source_repository"], qachat.PROJECT_KIND_CODE,
...     "/path/to/new_workspace",
... )

This writes a workspace copy; it does not alter the source repository.

"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Mapping, Sequence

from qachat.project import (
    PROJECT_KIND_CHOICES,
    ProjectKind,
    QAProject,
    configure_project,
    initialize_project,
    open_project,
)
from qachat.repository import scan_repository
from qachat.trace_logging import NullTraceLogger, TraceSink

__all__ = ["prepare_sources_workspace"]


def prepare_sources_workspace(
    source_paths: Sequence[str | Path],
    kind: ProjectKind,
    workspace_directory: str | Path,
    *,
    selected_files: Mapping[str, Sequence[str]] | None = None,
    settings: dict[str, object] | None = None,
    trace: TraceSink | None = None,
) -> QAProject:
    """Add sources to one project without changing their original files.

    Adding sources is cumulative. Re-adding a source updates its selected files
    without deleting older snapshots. Pass an empty list to reopen an existing
    workspace unchanged. ``selected_files`` limits each folder to checked
    relative file paths. ``settings`` is validated before copying.

    Parameters
    ----------
    source_paths: list[str | pathlib.Path] | tuple
        Local file/directory paths. Runtime validation accepts a list or tuple,
        although the annotation is Sequence. Entries are resolved,
        deduplicated, and checked against the destination workspace.
    kind: ProjectKind
        Project category, exactly code or literature.
    workspace_directory: str | pathlib.Path
        Destination managed workspace, new/empty or already containing a
        compatible profile.ini. Relative paths use the current working
        directory.
    selected_files: Mapping[str, Sequence[str]] | None, optional
        Optional mapping from resolved original source-path strings to relative
        selected files. A directory selection must be a nonempty subset of the
        scan report; None uses the scan selection. Defaults to ``None``.
    settings: dict[str, object] | None, optional
        Optional supported profile updates applied through configure_project
        before copying. Defaults to ``None``.
    trace: TraceSink | None, optional
        Optional caller-owned sink receiving preparation and copy events; None
        uses a null sink. Defaults to ``None``.

    Returns
    -------
    QAProject
        QAProject reopened from the prepared workspace after the selected copy
        operations.

    Raises
    ------
    ValueError
        If kinds, paths, workspace ownership/profile, source selections, or
        source-origin records fail their implemented checks.
    NotADirectoryError
        If an existing workspace is not a directory.
    OSError
        If sources cannot be resolved/read or workspace content cannot be
        created/copied/written.

    Notes
    -----
    Copying is cumulative: removed or deselected original files are not deleted
    from old workspace copies. Directory sources must have complete scans;
    direct file sources use a narrower extension/size check. Files, profile,
    and source-origin records are not committed as one transaction. A workspace
    is not an execution sandbox.

    """

    started_at = perf_counter()
    active_trace = trace if trace is not None else NullTraceLogger()
    active_trace.emit(
        "workspace.prepare.started",
        component="workspace",
        data={
            "source_paths": [str(path) for path in source_paths],
            "kind": kind,
            "workspace_directory": str(workspace_directory),
            "selected_files": selected_files,
            "settings": settings,
        },
        status="started",
    )
    if kind not in PROJECT_KIND_CHOICES:
        raise ValueError("kind must be code or literature.")
    if not isinstance(source_paths, (list, tuple)):
        raise ValueError("Choose source files or folders.")
    if not str(workspace_directory).strip():
        raise ValueError("Choose a workspace folder.")

    workspace = Path(workspace_directory).expanduser().resolve()
    if workspace in {Path.home().resolve(), Path(workspace.anchor)}:
        raise ValueError(
            "Choose a project workspace, not a home or drive root."
        )
    if workspace.exists() and not workspace.is_dir():
        raise NotADirectoryError(f"Choose a workspace folder: {workspace}")
    if workspace.is_dir() and not (workspace / "profile.ini").is_file():
        if any(workspace.iterdir()) and not _contains_only_trace_output(
            workspace
        ):
            raise ValueError("Choose an empty folder or a QAChat workspace.")

    chosen: list[Path] = []
    for value in source_paths:
        source = Path(value).expanduser()
        if source.is_symlink():
            raise ValueError(f"Do not add a symbolic link: {source}")
        source = source.resolve(strict=True)
        if not source.is_file() and not source.is_dir():
            raise ValueError(f"Choose a regular file or folder: {source}")
        if source == workspace or source.is_relative_to(workspace):
            raise ValueError("Do not add the workspace as its own source.")
        if source not in chosen:
            chosen.append(source)

    if not chosen and not (workspace / "profile.ini").is_file():
        raise ValueError("Add a source before creating a workspace.")
    workspace.mkdir(parents=True, exist_ok=True)
    source_root = workspace / "sources"
    if source_root.is_symlink():
        raise ValueError("The workspace sources folder must not be a link.")
    source_root.mkdir(exist_ok=True)
    project = initialize_project(workspace, kind)
    if project.profile.kind != kind:
        raise ValueError(
            "Workspace project type does not match the selection."
        )
    if project.source != source_root.resolve():
        raise ValueError(
            "This project does not use a managed sources/ folder."
        )
    if settings:
        project = configure_project(
            workspace,
            settings,
            trace=active_trace,
        )

    manifest_path = workspace / "sources.json"
    records = _read_records(manifest_path)
    used = {item["name"] for item in records}
    for source in chosen:
        original = str(source)
        record = next(
            (item for item in records if item["original"] == original),
            None,
        )
        if record is None:
            name = _unique_name(source, used, source_root)
            record = {"original": original, "name": name}
        destination = source_root / record["name"]
        selection = (
            selected_files.get(original)
            if selected_files is not None
            else None
        )
        copied = _copy_source(
            source, destination, project, workspace, selection
        )
        active_trace.emit(
            "workspace.source.copied",
            component="workspace",
            data={
                "source": str(source),
                "destination": str(destination),
                "requested_files": selection,
                "copied_files": copied,
            },
        )
        record["files"] = copied
        if record not in records:
            records.append(record)
            used.add(record["name"])
        _write_records(manifest_path, records)
    prepared = open_project(workspace)
    active_trace.emit(
        "workspace.prepare.completed",
        component="workspace",
        data={
            "project_directory": str(prepared.root),
            "source_directory": str(prepared.source),
            "sources_manifest": str(manifest_path),
            "source_count": len(records),
        },
        duration_ms=(perf_counter() - started_at) * 1000,
    )
    return prepared


def _contains_only_trace_output(workspace: Path) -> bool:
    """Allow a new project trace to exist before workspace preparation.

    Parameters
    ----------
    workspace: pathlib.Path
        Resolved managed-workspace path excluded from source copies when
        applicable.

    Returns
    -------
    bool
        True if the existing workspace contains only the permitted logs/traces
        preparation structure.

    Notes
    -----
    Used to allow an early-created trace directory before the project profile
    is initialized.

    """

    entries = list(workspace.iterdir())
    if len(entries) != 1 or entries[0].name != "logs":
        return False
    logs = entries[0]
    if logs.is_symlink() or not logs.is_dir():
        return False
    log_entries = list(logs.iterdir())
    return (
        len(log_entries) == 1
        and log_entries[0].name == "traces"
        and log_entries[0].is_dir()
        and not log_entries[0].is_symlink()
    )


def _unique_name(source: Path, used: set[str], root: Path) -> str:
    """Keep the original name unless another source already owns it.

    Parameters
    ----------
    source: pathlib.Path
        Original local source file or directory.
    used: set[str]
        Source names already reserved in the source-origin manifest.
    root: pathlib.Path
        Managed sources directory in which a name must not collide.

    Returns
    -------
    str
        Original basename when unused, otherwise a basename incorporating a
        deterministic source-path hash.

    Raises
    ------
    ValueError
        If the hashed alternative is also occupied or reserved.

    """

    if source.name not in used and not (root / source.name).exists():
        return source.name
    digest = sha256(str(source).encode()).hexdigest()[:8]
    name = f"{source.stem}-{digest}{source.suffix}"
    if name in used or (root / name).exists():
        raise ValueError(f"Could not choose a distinct source name: {source}")
    return name


def _copy_source(
    source: Path,
    destination: Path,
    project: QAProject,
    workspace: Path,
    selected_files: Sequence[str] | None,
) -> list[str]:
    """Copy only files selected by the workspace profile.

    Parameters
    ----------
    source: pathlib.Path
        Original local source file or directory.
    destination: pathlib.Path
        Managed destination path for the selected source or file.
    project: QAProject
        Resolved project whose profile provides extension and size-selection
        rules.
    workspace: pathlib.Path
        Resolved managed-workspace path excluded from source copies when
        applicable.
    selected_files: Sequence[str] | None
        Optional mapping from resolved original source-path strings to relative
        selected files. A directory selection must be a nonempty subset of the
        scan report; None uses the scan selection.

    Returns
    -------
    list[str]
        Relative filenames copied from a directory, or the copied source
        filename for a single-file input.

    Raises
    ------
    ValueError
        If a source file violates extension/size rules, a directory scan is
        incomplete, or an explicit selection is empty or outside the eligible
        scan set.
    OSError
        If scanning or copying fails.

    Notes
    -----
    The directory path applies scan_repository rules and excludes a nested
    workspace. Direct files are not passed through all protected-name rules.
    Existing destination files omitted from this copy operation are left in
    place.

    """

    profile = project.profile
    if source.is_file():
        if selected_files is not None and list(selected_files) != [
            source.name
        ]:
            raise ValueError(f"Invalid file selection: {source}")
        name = source.name.lower()
        size = source.stat().st_size
        if not any(name.endswith(ext) for ext in profile.extensions):
            raise ValueError(
                f"File type not selected in profile.ini: {source}"
            )
        if not 0 < size <= profile.max_file_bytes:
            raise ValueError(
                f"File is empty or exceeds the size limit: {source}"
            )
        _copy_file(source, destination)
        return [source.name]

    excludes: tuple[str, ...] = ()
    if workspace.is_relative_to(source):
        excludes = (workspace.relative_to(source).as_posix(),)
    report = scan_repository(
        source,
        extensions=profile.extensions,
        exclude_paths=excludes,
        max_file_bytes=profile.max_file_bytes,
    )
    if not report["complete"]:
        raise ValueError(f"Could not completely scan source folder: {source}")
    selected = report["selected_files"]
    if selected_files is not None:
        requested = set(selected_files)
        if not requested or not requested.issubset(set(selected)):
            raise ValueError(f"Invalid file selection: {source}")
        selected = [path for path in selected if path in requested]
    if not selected:
        raise ValueError(f"No selected files in source folder: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for relative in selected:
        _copy_file(source / str(relative), destination / str(relative))
    return [str(path) for path in selected]


def _copy_file(source: Path, destination: Path) -> None:
    """Replace one snapshot only after its new bytes are complete.

    Parameters
    ----------
    source: pathlib.Path
        Original local source file or directory.
    destination: pathlib.Path
        Managed destination path for the selected source or file.

    Raises
    ------
    ValueError
        If the destination or one of its parent directories is a symbolic link.
    OSError
        If creating directories, copying metadata/content, or replacing the
        target fails.

    Notes
    -----
    The file is copied through a predictable temporary sibling and then
    replaced. These checks are not a race-free security boundary and do not
    validate every possible temporary-path interference.

    """

    if destination.is_symlink():
        raise ValueError(f"Do not copy through a symbolic link: {destination}")
    for parent in destination.parents:
        if parent.is_symlink():
            raise ValueError(f"Do not copy through a symbolic link: {parent}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _read_records(path: Path) -> list[dict[str, str]]:
    """Validate persisted source ownership before using destination names.

    Parameters
    ----------
    path: pathlib.Path
        Source-origin JSON manifest path.

    Returns
    -------
    list[dict[str, str]]
        List of source-origin record dictionaries, or an empty list when the
        manifest is missing.

    Raises
    ------
    ValueError
        If the manifest format, record structure, original path, or stored
        source name is invalid.
    OSError
        If the file cannot be read.

    Notes
    -----
    Stored records may also contain a files list; the existing return
    annotation is narrower than that actual JSON structure. This helper does
    not validate every copied-file entry.

    """

    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Invalid workspace source list.")
    records = payload.get("sources")
    if not isinstance(records, list):
        raise ValueError("Invalid workspace source list.")
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Invalid workspace source entry.")
        original, name = item.get("original"), item.get("name")
        if not isinstance(original, str) or not isinstance(name, str):
            raise ValueError("Invalid workspace source entry.")
        if name in {"", ".", ".."} or Path(name).name != name:
            raise ValueError("Invalid workspace source name.")
    return records


def _write_records(path: Path, records: list[dict[str, str]]) -> None:
    """Save ownership after each successful source copy.

    Parameters
    ----------
    path: pathlib.Path
        Source-origin JSON manifest path.
    records: list[dict[str, str]]
        Source-origin records with original and name strings and, when written
        by preparation, a list of copied files.

    Notes
    -----
    Writes the source-origin format version and supplied records. No
    multi-writer coordination is provided; records are not checked against
    destination contents here.

    """

    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"version": 1, "sources": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
