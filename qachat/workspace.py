"""Copy selected local sources into a managed multi-source workspace.

``prepare_sources_workspace`` accepts local files or directories, prepares a
project profile, creates its shared core directories and selected tool
resources, copies content into ``sources``, and records source origins in
``sources.json``. Imported snapshots are project-owned: later
imports preserve existing files unless replacement is explicitly requested.
Source names are disambiguated when needed.
Directory inputs use conservative repository scanning. Individual-file
inputs bypass scan extension and protected-name filters, while retaining
size and filesystem checks.

This API is cumulative: adding or reselecting sources copies their current
selected files but does not remove old copies merely because the originals
were deleted or deselected. It differs from the single-source mirroring API
in ``qachat.project``. A workspace organizes local files; it is not an
execution sandbox or a complete defense against hostile filesystem changes.
First-time initialization commits profile.ini last and rolls back newly
created entries on failure. Later cumulative imports can leave earlier
successful copies in place if a subsequent source fails.

Key Features
------------

- **Multiple origins**

    Keep source-origin records and allocate noncolliding names within one
    sources directory.

- **Selected copying**

    Use profile selection rules and optional per-source selections for
    directory inputs. Preserve existing snapshot files by default; require
    ``replace_existing=True`` to overwrite them.

- **Profile integration**

    Initialize a managed project and apply supported configuration updates.

- **Trace integration**

    Emit workspace preparation and source-copy events through the supplied
    sink.

Module Structure
----------------

- ``prepare_sources_workspace``

    Add sources to one project without changing their original files.

- ``remove_workspace_sources``

    Delete managed source copies and update provenance, never originals.

- ``include_explicit_imports``

    Preserve explicitly imported files during RAG source scanning.

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
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import (
    Callable, Final, Mapping, NotRequired, Sequence, TypedDict, cast,
)

from qachat.project import (
    PROJECT_KIND_CHOICES,
    ProjectKind,
    QAProject,
    _preparing_project,
    open_project,
)
from qachat.repository import scan_repository
from qachat.trace_logging import NullTraceLogger, TraceSink

__all__ = [
    "prepare_sources_workspace",
    "remove_workspace_sources",
    "include_explicit_imports",
]

# Workspace folders whose files may be copied into the sources: what QAChat
# wrote (qachat.tools._shared.outputs) and the report templates.
_COPYABLE_WORKSPACE_FOLDERS: Final[tuple[str, ...]] = ("outputs", "templates")


class _SourceRecord(TypedDict):
    """Persisted ownership and imported-file inventory of one source.

    """

    original: str
    name: str
    files: NotRequired[list[str]]


def prepare_sources_workspace(
    source_paths: Sequence[str | Path],
    kind: ProjectKind,
    workspace_directory: str | Path,
    *,
    selected_files: Mapping[str, Sequence[str]] | None = None,
    replace_existing: bool = False,
    settings: dict[str, object] | None = None,
    trace: TraceSink | None = None,
) -> QAProject:
    """Add sources to one project without changing their original files.

    Adding sources is cumulative. Re-adding a source imports newly selected
    files without deleting or overwriting existing snapshots. To replace
    selected snapshot files explicitly, pass ``replace_existing=True``.
    Pass an empty list to reopen an existing workspace unchanged. When
    provided, ``selected_files`` must identify every supplied source, so
    unmatched or missing keys cannot broaden the import. ``settings`` is
    validated before copying.

    Parameters
    ----------
    source_paths: list[str | pathlib.Path] | tuple
        Local file/directory paths. Runtime validation accepts a list or tuple,
        although the annotation is Sequence. Entries are resolved,
        deduplicated, and checked against the destination workspace.
    kind: ProjectKind
        Project type ID from ``qachat/project_types/``.
    workspace_directory: str | pathlib.Path
        Destination managed workspace, new/empty or already containing a
        compatible profile.ini. Relative paths use the current working
        directory.
    selected_files: Mapping[str, Sequence[str]] | None, optional
        Optional mapping from source paths to selected relative filenames.
        Keys are normalized and must match all supplied source paths exactly.
        Each directory selection must be a nonempty subset of the scan report;
        None uses the scan selection. Defaults to ``None``.
    replace_existing: bool, optional
        Whether to overwrite already imported files selected again. Defaults
        to ``False``, preserving project-owned snapshots even when their
        originals or the local copies have changed. Missing files and newly
        selected files are still copied.
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
        If kinds, paths, workspace ownership/profile, source selections,
        replacement policy, or source-origin records fail their checks.
    NotADirectoryError
        If an existing workspace is not a directory.
    OSError
        If sources cannot be resolved/read or workspace content cannot be
        created/copied/written.

    Notes
    -----
    Copying is cumulative: removed or deselected original files are not deleted
    from old workspace copies. Existing snapshots are preserved by default;
    a replacement must be explicitly requested. Directory sources must have
    complete scans; direct files use a narrower extension/size check.
    Files, profile, and source-origin records are not committed as one
    transaction. A workspace is not an execution sandbox.

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
            "replace_existing": replace_existing,
            "settings": settings,
        },
        status="started",
    )
    if kind not in PROJECT_KIND_CHOICES:
        raise ValueError(
            f"kind must be one of: {', '.join(PROJECT_KIND_CHOICES)}."
        )
    if not isinstance(source_paths, (list, tuple)):
        raise ValueError("Choose source files or folders.")
    if type(replace_existing) is not bool:
        raise ValueError("replace_existing must be true or false.")
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
        # A file QAChat wrote (outputs/) or a template may become a source;
        # the rest of the workspace (sources, indexes, logs) may not.
        inside = source == workspace or source.is_relative_to(workspace)
        if inside and not any(
            source.is_relative_to(workspace / folder)
            and source != workspace / folder
            for folder in _COPYABLE_WORKSPACE_FOLDERS
        ):
            raise ValueError("Do not add the workspace as its own source.")
        if source not in chosen:
            chosen.append(source)

    selections: dict[str, Sequence[str]] | None = None
    if selected_files is not None:
        if not isinstance(selected_files, Mapping):
            raise ValueError("Source selections must be a path mapping.")
        selections = {}
        for key, files in selected_files.items():
            if not isinstance(key, (str, Path)) or not str(key).strip():
                raise ValueError("Invalid source selection path.")
            original = str(Path(key).expanduser().resolve())
            if original in selections:
                raise ValueError(f"Duplicate source selection: {original}")
            if not isinstance(files, (list, tuple)) or not all(
                isinstance(file, str) for file in files
            ):
                raise ValueError(f"Invalid file selection: {original}")
            selections[original] = files
        expected = {str(source) for source in chosen}
        if selections.keys() != expected:
            missing = sorted(expected - selections.keys())
            unexpected = sorted(selections.keys() - expected)
            raise ValueError(
                "Source selection paths must match requested sources; "
                f"missing: {missing}; unexpected: {unexpected}."
            )

    if not chosen and not (workspace / "profile.ini").is_file():
        raise ValueError("Add a source before creating a workspace.")
    source_root = workspace / "sources"
    if source_root.is_symlink():
        raise ValueError("The workspace sources folder must not be a link.")
    with _preparing_project(
        workspace, kind, "sources", settings=settings,
        trace=active_trace, validate_source=True,
    ) as (project, journal):
        if project.source != source_root.resolve(strict=True):
            raise ValueError(
                "This project does not use a managed sources/ folder."
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
                record = _SourceRecord(original=original, name=name)
            destination = source_root / record["name"]
            selection = (
                selections[original] if selections is not None else None
            )
            copied, preserved = _copy_source(
                source,
                destination,
                project,
                workspace,
                selection,
                replace_existing=replace_existing,
                on_create=journal.note if journal is not None else None,
            )
            active_trace.emit(
                "workspace.source.copied",
                component="workspace",
                data={
                    "source": str(source),
                    "destination": str(destination),
                    "requested_files": selection,
                    "copied_files": copied,
                    "preserved_files": preserved,
                },
            )
            if source.is_file():
                record["files"] = copied + preserved
            else:
                inventory = set(record.get("files", [])) | set(copied)
                record["files"] = sorted(
                    relative
                    for relative in inventory
                    if _is_regular_snapshot(destination, relative)
                )
            if record not in records:
                records.append(record)
                used.add(record["name"])
            if journal is not None:
                journal.note(manifest_path)
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
    *,
    replace_existing: bool = False,
    on_create: Callable[[Path], None] | None = None,
) -> tuple[list[str], list[str]]:
    """Copy eligible files while protecting existing snapshots.

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
        Optional selected filenames relative to a directory or the original
        filename for a direct source file. None uses the scan selection.
    replace_existing: bool, optional
        Replace existing regular snapshot files when True. Defaults to False.
    on_create: Callable[[pathlib.Path], None] | None, optional
        Record newly created snapshot paths for first-creation rollback.

    Returns
    -------
    tuple[list[str], list[str]]
        Original-relative filenames copied and preserved, respectively.

    Raises
    ------
    ValueError
        If a source file violates size rules, a directory scan is
        incomplete, or an explicit selection is empty or outside the eligible
        scan set.
    OSError
        If scanning or copying fails.

    Notes
    -----
    The directory path applies scan_repository rules and excludes a nested
    workspace. Explicitly selected files bypass extension and protected-name
    selection filters but retain size and containment safeguards.
    Files omitted from this copy operation are left in place, as are files
    selected again without an explicit replacement request.

    """

    profile = project.profile
    if source.is_file():
        if selected_files is not None and list(selected_files) != [
            source.name
        ]:
            raise ValueError(f"Invalid file selection: {source}")
        size = source.stat().st_size
        if not 0 < size <= profile.max_file_bytes:
            raise ValueError(
                f"File is empty or exceeds the size limit: {source}"
            )
        if _copy_file(
            source, destination, replace_existing=replace_existing,
            on_create=on_create,
        ):
            return [source.name], []
        return [], [source.name]

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
    if on_create is not None:
        on_create(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    preserved: list[str] = []
    for relative in selected:
        path = str(relative)
        if _copy_file(
            source / path,
            destination / path,
            replace_existing=replace_existing,
            on_create=on_create,
        ):
            copied.append(path)
        else:
            preserved.append(path)
    return copied, preserved


def _copy_file(
    source: Path,
    destination: Path,
    *,
    replace_existing: bool = False,
    on_create: Callable[[Path], None] | None = None,
) -> bool:
    """Create one snapshot, preserving an existing file by default.

    Parameters
    ----------
    source: pathlib.Path
        Original local source file or directory.
    destination: pathlib.Path
        Managed destination path for the selected source or file.
    replace_existing: bool, optional
        Explicitly allow replacement of an existing regular file. Defaults
        to False.
    on_create: Callable[[pathlib.Path], None] | None, optional
        Record a prospective new destination before writing its contents.

    Returns
    -------
    bool
        True if copied, False if an existing snapshot was preserved.

    Raises
    ------
    ValueError
        If the destination or one of its parents is a symbolic link, or an
        existing destination is not a regular file.
    OSError
        If creating directories, copying metadata/content, or replacing the
        target fails.

    Notes
    -----
    Existing regular files are not overwritten without explicit permission.
    New bytes are copied into a randomly named temporary file in the
    destination directory before atomic replacement. The temporary file is
    removed if copying or replacement fails. This is not a general defense
    against concurrent filesystem mutation by another process.

    """

    if destination.is_symlink():
        raise ValueError(f"Do not copy through a symbolic link: {destination}")
    for parent in destination.parents:
        if parent.is_symlink():
            raise ValueError(f"Do not copy through a symbolic link: {parent}")
    if destination.exists():
        if not destination.is_file():
            raise ValueError(
                f"Snapshot destination is not a regular file: {destination}"
            )
        if not replace_existing:
            return False
    if on_create is not None:
        on_create(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(stream.name)
    try:
        with stream:
            with source.open("rb") as original:
                shutil.copyfileobj(original, stream)
        shutil.copystat(source, temporary)
        temporary.replace(destination)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _is_regular_snapshot(destination: Path, relative: str) -> bool:
    """Check whether a tracked imported file still exists without links.

    Parameters
    ----------
    destination: pathlib.Path
        Original source's destination file or directory in the workspace.
    relative: str
        Name of a previously imported snapshot relative to the destination.

    Returns
    -------
    bool
        True if the tracked path is a regular file reached without symlinks.

    """

    path = destination / relative
    if path.is_symlink() or not path.is_file():
        return False
    parent = path.parent
    while parent != destination:
        if parent.is_symlink():
            return False
        parent = parent.parent
    return True


def _read_records(path: Path) -> list[_SourceRecord]:
    """Validate persisted source ownership before using destination names.

    Parameters
    ----------
    path: pathlib.Path
        Source-origin JSON manifest path.

    Returns
    -------
    list[_SourceRecord]
        Validated ownership and inventory records, or an empty list if missing.

    Raises
    ------
    ValueError
        If the manifest format, fields, inventory, or ownership is invalid.
    OSError
        If the file cannot be read.

    Notes
    -----
    A missing ``files`` list is accepted for older manifest records. Present
    inventories must contain unique, normalized relative paths.

    """

    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Invalid workspace source list.")
    records = payload.get("sources")
    if not isinstance(records, list):
        raise ValueError("Invalid workspace source list.")
    seen_names: set[str] = set()
    seen_originals: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Invalid workspace source entry.")
        original, name = item.get("original"), item.get("name")
        if not isinstance(original, str) or not isinstance(name, str):
            raise ValueError("Invalid workspace source entry.")
        if (
            name in {"", ".", ".."}
            or Path(name).name != name
            or "\\" in name
            or PureWindowsPath(name).drive
        ):
            raise ValueError("Invalid workspace source name.")
        if not original.strip():
            raise ValueError("Invalid workspace source entry.")
        if original in seen_originals or name in seen_names:
            raise ValueError("Duplicate or conflicting workspace sources.")
        seen_originals.add(original)
        seen_names.add(name)
        files = item.get("files", [])
        if not isinstance(files, list):
            raise ValueError("Invalid workspace source files inventory.")
        seen_files: set[str] = set()
        for relative in files:
            if not isinstance(relative, str) or not relative:
                raise ValueError("Invalid workspace source files inventory.")
            normalized = PurePosixPath(relative)
            if (
                normalized.is_absolute()
                or not normalized.parts
                or ".." in normalized.parts
                or PureWindowsPath(relative).drive
                or "\\" in relative
                or normalized.as_posix() != relative
                or relative in seen_files
            ):
                raise ValueError("Invalid workspace source files inventory.")
            seen_files.add(relative)
    return cast("list[_SourceRecord]", records)


def _write_records(path: Path, records: list[_SourceRecord]) -> None:
    """Save ownership after each successful source copy.

    Parameters
    ----------
    path: pathlib.Path
        Source-origin JSON manifest path.
    records: list[_SourceRecord]
        Source-origin records with original and name strings and, when written
        by preparation, a list of copied files.

    Notes
    -----
    Uses a newly created temporary file in the same directory and replaces
    the manifest after writing it completely. Temporary files are cleaned up
    on errors. There is no multi-writer coordination.

    """

    stream = NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(
                json.dumps({"version": 1, "sources": records}, indent=2)
                + "\n"
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def remove_workspace_sources(
    project: QAProject, paths: Sequence[str]
) -> list[str]:
    """Delete managed source snapshots, never the original source files.

    Parameters
    ----------
    project: QAProject
        Opened project backed by the managed ``sources/`` directory.
    paths: Sequence[str]
        Workspace-relative files or folders strictly beneath ``sources/``.

    Returns
    -------
    list[str]
        Workspace-relative paths removed, in the requested order.

    Raises
    ------
    ValueError
        If the project references external sources, an input path is unsafe,
        or a requested snapshot does not exist.

    Notes
    -----
    All paths are checked before deletion. Source-origin records are updated
    after deletion, including when an I/O error interrupts a batch. Index
    refresh is intentionally left to the caller.

    """

    if not isinstance(project, QAProject):
        raise TypeError("Expected an opened QAProject.")
    root = project.root.resolve(strict=True)
    managed = root / "sources"
    if (
        managed.is_symlink()
        or not managed.is_dir()
        or project.source.resolve(strict=True) != managed
    ):
        raise ValueError("Only managed sources/ snapshots can be removed.")
    if not isinstance(paths, (list, tuple)) or not paths:
        raise ValueError("Choose source files or folders to delete.")

    targets: list[Path] = []
    for value in paths:
        if not isinstance(value, str) or not value:
            raise ValueError("Choose a workspace-relative source path.")
        relative = PurePosixPath(value)
        if (
            relative.is_absolute()
            or PureWindowsPath(value).drive
            or "\\" in value
            or any(part in ("", ".", "..") for part in value.split("/"))
            or len(relative.parts) < 2
            or relative.parts[0] != "sources"
        ):
            raise ValueError(
                f"Only items inside sources/ can be removed: {value}"
            )
        target = root.joinpath(*relative.parts)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"Do not remove symbolic links: {value}")
        if not target.exists():
            raise ValueError(f"Source not found: {value}")
        targets.append(target)
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate source paths.")
    for index, target in enumerate(targets):
        if any(
            target.is_relative_to(other)
            for j, other in enumerate(targets) if index != j
        ):
            raise ValueError("Cannot remove a folder and its child together.")

    manifest = root / "sources.json"
    records = _read_records(manifest)
    try:
        for target in targets:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
    finally:
        if manifest.exists():
            retained: list[_SourceRecord] = []
            for record in records:
                destination = managed / record["name"]
                if destination.is_symlink() or not destination.exists():
                    continue
                if destination.is_file():
                    record["files"] = [Path(record["original"]).name]
                elif destination.is_dir():
                    record["files"] = [
                        item for item in record.get("files", [])
                        if _is_regular_snapshot(destination, item)
                    ]
                else:
                    continue
                retained.append(record)
            _write_records(manifest, retained)
    return [target.relative_to(root).as_posix() for target in targets]


def include_explicit_imports(
    project: QAProject, report: dict[str, object]
) -> dict[str, object]:
    """Keep directly imported source files eligible for indexing.

    Parameters
    ----------
    project: QAProject
        Opened project whose managed source manifest records explicit imports.
    report: dict[str, object]
        Ordinary conservative repository scan of the project's source root.

    Returns
    -------
    dict[str, object]
        Scan report that additionally selects safe, explicitly imported files
        irrespective of their filename suffix or protected-name pattern.

    Notes
    -----
    Only direct-file imports recorded in ``sources.json`` are added. A folder
    imported from a repository retains the repository's ordinary selection
    policy. The file-size limit and symlink safeguards still apply, and an
    unsupported source format may fail during subsequent text extraction.

    """

    root = project.root.resolve(strict=True)
    managed = root / "sources"
    if (
        project.source.resolve(strict=True) != managed
        or not (root / "sources.json").is_file()
    ):
        return report
    records = _read_records(root / "sources.json")
    rows = report.get("entries")
    if not isinstance(rows, list):
        raise ValueError("Invalid project source scan report.")
    indexed: dict[str, dict[str, object]] = {}
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("Invalid project source scan entry.")
        relative = item.get("path")
        if isinstance(relative, str):
            indexed[relative] = dict(item)
    selected = set(report.get("selected_files", []))
    for record in records:
        name = record["name"]
        snapshot = managed / name
        if snapshot.is_symlink() or not snapshot.is_file():
            continue
        size = snapshot.stat().st_size
        if not 0 < size <= project.profile.max_file_bytes:
            continue
        indexed[name] = {
            "path": name,
            "kind": "file",
            "size_bytes": size,
            "status": "selected",
            "reason": "explicitly imported file",
        }
        selected.add(name)
    rows = [indexed[name] for name in sorted(indexed)]
    return {
        **report,
        "entries": rows,
        "selected_files": sorted(selected),
        "selected_bytes": sum(
            int(row["size_bytes"]) for row in rows
            if row["status"] == "selected"
        ),
    }
