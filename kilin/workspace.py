"""Copy selected local sources into a managed multi-source workspace.

``prepare_sources_workspace`` exclusively creates a new project, copies
selected local inputs into ``sources``, and records origins in
``sources.json``, with each copy's size and modification time at import.
An explicit destination must be nonexistent or empty. Default names are
allocated without reusing occupied folders, even for identical sources.
``add_project_sources`` deliberately imports into a saved project, preserving
existing snapshots unless replacement is explicitly requested. It never
creates or reconfigures the project. Source names are disambiguated as needed.
Directory inputs use conservative repository scanning. Individual-file
inputs bypass scan extension and protected-name filters, while retaining
size and filesystem checks.

The explicit import API is cumulative: reselecting sources copies their current
selected files but does not remove old copies merely because the originals
were deleted or deselected. It differs from the single-source mirroring API
in ``kilin.project``. A workspace organizes local files; it is not an
execution sandbox or a complete defense against hostile filesystem changes.
First-time initialization commits profile.ini last and rolls back newly
created entries on failure. Later cumulative imports can leave earlier
successful copies in place if a subsequent source fails.

Key Features
------------

- **Multiple origins**

    Keep source-origin records and allocate noncolliding names within one
    sources directory.

- **Copy stamps**

    Record each copy's size and modification time when it is imported, for
    fast comparison with its original and for detecting edited copies.
    Matching metadata is treated as unchanged, not as proof of equal bytes.

- **Change detection and per-file synchronization**

    Recursively detect eligible new files, changed originals, and deletions.
    A caller may import, replace, keep, delete, or preserve both versions for
    individual files. Metadata-only changes avoid rewriting file contents.
    Restored originals of detached copies require explicit reconciliation.

- **Separately retrievable versions**

    Record earlier snapshots under ``sources/.kilin_versions`` with distinct
    paths, so accepted versions are independently represented in retrieval
    indexes and cited without ambiguity.

- **Selected copying**

    Use profile selection rules and optional per-source selections for
    directory inputs. Preserve existing snapshot files by default; require
    ``replace_existing=True`` to overwrite them. New and updated copies
    obey the saved project file-size limit.

- **Profile integration**

    Apply validated settings only during new-project creation. Existing
    settings change only through the separate configure_project API.

- **Trace integration**

    Emit workspace preparation and source-copy events through the supplied
    sink.

Module Structure
----------------

- ``prepare_sources_workspace``

    Create an independent managed project without changing original sources.

- ``add_project_sources``

    Deliberately add sources to an existing managed project.

- ``preview_project_sources``

    Plan eligible source imports and destinations without copying files.

- ``default_workspace_path``

    Suggest a base name without reserving or establishing ownership.

- ``remove_workspace_sources``

    Delete managed source copies and update provenance, never originals.

- ``find_source_changes``

    List eligible added files and copies whose originals changed or vanished.

- ``sync_project_sources``

    Apply individual changes, optionally retaining both source versions.

- ``list_source_versions``

    List the separate indexed-source identities of retained versions.

- ``include_explicit_imports``

    Preserve explicitly imported files and tracked historical versions during
    RAG source scanning.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import kilin
>>> project = kilin.prepare_sources_workspace(
...     ["/path/to/source_repository"], kilin.PROJECT_TYPE_IDS[0],
...     "/path/to/new_workspace",
... )

This writes a workspace copy; it does not alter the source repository.

"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from hashlib import sha256
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import (
    Callable, Final, Literal, Mapping, NotRequired, Sequence, TypeAlias,
    TypedDict, cast,
)

from kilin._project_creation import check_available, destination_path
from kilin.project import (
    project_type_ids,
    ProjectTypeId,
    QAProject,
    _preparing_project,
    open_project,
)
from kilin.repository import scan_repository
from kilin.repository_reason_codes import (
    SELECTION_REASON_EMPTY_FILE,
    SELECTION_REASON_FILE_TOO_LARGE,
    SELECTION_REASON_FILE_TYPE_NOT_SELECTED,
)
from kilin._trace_project import ProjectTraceSequence
from kilin.conversation_ownership import ConversationFileLock
from kilin.trace_logging import (
    NullTraceLogger, TraceConfig, TraceLogger, TraceSink,
    _project_trace_logger,
)
from kilin.workspace_resources import PROJECT_SOURCES_DIRECTORY

__all__ = [
    "prepare_sources_workspace",
    "add_project_sources",
    "preview_project_sources",
    "default_workspace_path",
    "remove_workspace_sources",
    "find_source_changes",
    "sync_project_sources",
    "SourceChanges",
    "SourceAction",
    "SourceVersion",
    "list_source_versions",
    "include_explicit_imports",
]

# Workspace folders whose files may be copied into the sources: what KíLín
# wrote (kilin.tools._shared.outputs) and the report templates.
_COPYABLE_WORKSPACE_FOLDERS: Final[tuple[str, ...]] = ("outputs", "templates")
_VERSION_DIRECTORY: Final[str] = ".kilin_versions"
SourceAction: TypeAlias = Literal[
    "import", "replace", "keep", "keep_both", "delete"
]


class _CopyStamp(TypedDict):
    """Size and modification time of one source copy when imported.

    Attributes
    ----------
    size: int
        File size in bytes when the managed copy was imported.
    mtime_ns: int
        Modification time in nanoseconds at that import.

    """

    size: int
    mtime_ns: int


class SourceVersion(TypedDict):
    """One retained and separately indexable source version.

    Attributes
    ----------
    source: str
        Original workspace-relative path, for example ``sources/a/main.py``.
    version: str
        Unique workspace-relative path of the preserved source version.
    sha256: str
        Content hash recorded when the version was preserved.

    """

    source: str
    version: str
    sha256: str


class _StoredVersion(TypedDict):
    """Persisted path and digest for one retained source version.

    Attributes
    ----------
    file: str
        Relative filename within the corresponding imported source.
    path: str
        Saved version path relative to the managed ``sources/`` directory.
    sha256: str
        Lowercase SHA-256 digest of the retained source snapshot.

    """

    file: str
    path: str
    sha256: str


class SourceChanges(TypedDict):
    """Classify differences between imported copies and their original sources.

    The four lists contain workspace-relative paths. ``edited`` copies are
    retained even when their originals changed or disappeared.

    Attributes
    ----------
    added: list[str]
        Newly discovered files that can be imported from tracked folders.
    changed: list[str]
        Existing copies whose original files have changed.
    deleted: list[str]
        Tracked copies whose originals no longer exist.
    edited: list[str]
        Managed copies edited locally and therefore protected from replacement.

    """

    added: list[str]
    changed: list[str]
    deleted: list[str]
    edited: list[str]


class _SourceRecord(TypedDict):
    """Persist ownership and the imported-file inventory for one source.

    Attributes
    ----------
    original: str
        Recorded path of the external source selected for import.
    name: str
        Unique name allocated inside the managed ``sources/`` directory.
    files: list[str], optional
        Imported files relative to the source directory; absent in old records.
    stamps: dict[str, _CopyStamp], optional
        Copy-time file metadata keyed by imported relative filename.
    kind: "file" | "folder", optional
        Source type; absent in older manifests.
    versions: list[_StoredVersion], optional
        Versions retained from earlier imports, with paths and digests.

    """

    original: str
    name: str
    files: NotRequired[list[str]]
    stamps: NotRequired[dict[str, _CopyStamp]]
    kind: NotRequired[Literal["file", "folder"]]
    versions: NotRequired[list[_StoredVersion]]


def default_workspace_path(
    source_paths: Sequence[str | Path], workspace_root: str | Path
) -> Path:
    """Suggest a deterministic base name without reserving or reusing it.

    Parameters
    ----------
    source_paths: Sequence[str | pathlib.Path]
        Sources whose first entry supplies the name. An empty collection
        uses ``new-project`` as the preferred name.
    workspace_root: str | pathlib.Path
        Parent directory for automatically allocated workspaces.

    Returns
    -------
    pathlib.Path
        Preferred source-name/path-hash destination. Creation may choose a
        numbered sibling when this path is occupied.

    Raises
    ------
    ValueError
        If the parent directory is missing.

    Notes
    -----
    This helper has no filesystem side effects and confers no ownership.
    prepare_sources_workspace reserves the final location exclusively.

    """

    if not isinstance(source_paths, (list, tuple)):
        raise ValueError("Sources must be a list or tuple.")
    if not str(workspace_root).strip():
        raise ValueError("Choose a workspace parent folder.")
    parent = Path(workspace_root).expanduser().resolve()
    if not source_paths:
        return parent / "new-project"
    source = Path(source_paths[0]).expanduser().resolve()
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", source.stem).strip("-._")
    name = name or "project"
    digest = sha256(str(source).encode("utf-8")).hexdigest()[:8]
    return parent / f"{name}-{digest}"


def prepare_sources_workspace(
    source_paths: Sequence[str | Path],
    type_id: ProjectTypeId,
    workspace_directory: str | Path | None = None,
    *,
    workspace_root: str | Path | None = None,
    selected_files: Mapping[str, Sequence[str]] | None = None,
    settings: dict[str, object] | None = None,
    trace: TraceConfig | TraceSink | None = None,
) -> QAProject:
    """Create a new managed workspace without reusing an existing project.

    Parameters
    ----------
    source_paths: Sequence[str | pathlib.Path]
        List or tuple of files or folders to copy. An empty collection creates
        a source-free project, named ``new-project`` when allocated.
    type_id: ProjectTypeId
        Registered project type.
    workspace_directory: str | pathlib.Path | None, optional
        Explicit nonexistent or empty destination. Defaults to None, requiring
        workspace_root for automatic name allocation.
    workspace_root: str | pathlib.Path | None, optional
        Parent for a default destination: the first source determines its
        name, or ``new-project`` when no sources are supplied. On collision,
        append -2, -3, and so on, even for existing empty folders. Cannot
        be combined with workspace_directory. Defaults to None.
    selected_files: Mapping[str, Sequence[str]] | None, optional
        Exact per-source selections. Defaults to None, using scan selections.
    settings: dict[str, object] | None, optional
        Validated profile overrides for this new project. Defaults to None.
    trace: TraceConfig | TraceSink | None, optional
        Caller-owned diagnostic sink, or a configuration that creates an
        owned project trace after exclusive reservation. Defaults to None.
        No event is emitted before the destination is claimed and inputs
        are validated. A configuration must not specify an output path.

    Returns
    -------
    QAProject
        New project with managed sources/ and normal resource directories.

    Raises
    ------
    ProjectAlreadyExistsError
        If an explicit destination already contains a profile entry.
    WorkspaceOccupiedError
        If an explicit destination contains other content or is reserved.
    ValueError
        If paths, settings, project type, or selections are invalid.
    OSError
        If exclusive creation, copying, or profile publication fails.
    RuntimeError
        If an owned trace fails under the strict diagnostic policy.

    Notes
    -----
    Creation never resumes a project, applies settings to an existing profile,
    or permits overwriting a destination. Use open_project for read-only
    opening and add_project_sources for intentional cumulative imports.
    profile.ini is published last; failures roll back this attempt's entries.
    Owned completion diagnostics and trace closure precede publication so
    strict diagnostic failures also roll back. Caller-owned diagnostics
    complete after publication, and this function never closes their sink.
    Copying provides organization, not an execution sandbox.

    """

    allocate = workspace_directory is None
    if allocate:
        if workspace_root is None:
            raise ValueError("Choose a workspace folder or parent folder.")
        destination = default_workspace_path(source_paths, workspace_root)
    else:
        if workspace_root is not None:
            raise ValueError("Choose a workspace or its parent, not both.")
        destination = destination_path(workspace_directory)
        check_available(destination)
    return _prepare_sources_workspace(
        source_paths, type_id, destination, selected_files=selected_files,
        settings=settings, trace=trace, allocate=allocate,
    )


def add_project_sources(
    project_directory: str | Path,
    source_paths: Sequence[str | Path],
    *,
    selected_files: Mapping[str, Sequence[str]] | None = None,
    replace_existing: bool = False,
    trace: TraceSink | None = None,
) -> QAProject:
    """Deliberately import sources into an existing managed workspace.

    Parameters
    ----------
    project_directory: str | pathlib.Path
        Existing project with a managed sources/ directory. Its saved profile
        is authoritative; this operation never creates or reconfigures it.
    source_paths: Sequence[str | pathlib.Path]
        List or tuple of files or directories. Empty collections are allowed.
    selected_files: Mapping[str, Sequence[str]] | None, optional
        Exact selections for each source. Defaults to None, using scan rules.
    replace_existing: bool, optional
        Replace reselected snapshot files only when explicitly requested.
        Defaults to False, preserving existing project-owned copies.
    trace: TraceSink | None, optional
        Caller-owned diagnostic sink. Defaults to None.

    Returns
    -------
    QAProject
        Existing project after cumulative import and provenance updates.

    Raises
    ------
    FileNotFoundError
        If the destination has no existing project profile.
    ValueError
        If the profile, sources, or selection rules are invalid.
    OSError
        If reading, copying, or updating provenance fails.

    Notes
    -----
    This explicit update is not a multi-file transaction. Successful earlier
    copies may remain if a later source fails. Deleted or deselected originals
    never remove existing snapshots. configure_project remains the separate
    operation for changing saved profile settings.

    """

    project = open_project(project_directory)
    return _prepare_sources_workspace(
        source_paths, project.profile.type_id, project.root,
        selected_files=selected_files, replace_existing=replace_existing,
        trace=trace, existing=project,
    )



def preview_project_sources(
    project: QAProject, source_paths: Sequence[str | Path]
) -> dict[str, list]:
    """Preview cumulative source imports without modifying the workspace.

    Parameters
    ----------
    project: QAProject
        Opened project with a managed ``sources/`` directory.
    source_paths: Sequence[str | pathlib.Path]
        Candidate direct files and folders, in user-selected order.

    Returns
    -------
    dict[str, list]
        ``add`` describes eligible sources and their allocated destination
        names; ``sources`` retains corresponding input values. ``files``
        lists excluded files and selection reasons; ``skipped`` lists folders
        without eligible files. Existing sources use state ``existing``.

    Raises
    ------
    ValueError
        If a source or destination violates import rules, or a directory
        cannot be scanned completely.
    OSError
        If a selected path cannot be inspected.

    Notes
    -----
    The plan shares destination allocation and profile limits with the core
    importer, but does not reserve paths. Copying revalidates these rules.
    Explicit files bypass directory extension and protected-name filters.

    """

    root = _managed_root(project)
    if root is None:
        raise ValueError("Only managed sources/ snapshots can be imported.")
    if not isinstance(source_paths, (list, tuple)):
        raise ValueError("Choose source files or folders.")
    records = _read_records(root / "sources.json")
    imported = {item["original"]: item["name"] for item in records}
    used = set(imported.values())
    shown = {
        SELECTION_REASON_FILE_TYPE_NOT_SELECTED,
        SELECTION_REASON_FILE_TOO_LARGE,
        SELECTION_REASON_EMPTY_FILE,
    }
    plan: dict[str, list] = {
        "add": [], "sources": [], "files": [], "skipped": []
    }
    seen: set[Path] = set()
    for value in source_paths:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError("Choose source files or folders.")
        requested = Path(value).expanduser()
        if requested.is_symlink():
            raise ValueError(f"Do not add a symbolic link: {requested}")
        source = requested.resolve(strict=True)
        if not source.is_file() and not source.is_dir():
            raise ValueError(f"Choose a regular file or folder: {source}")
        if source in seen:
            continue
        seen.add(source)
        if (source == root or source.is_relative_to(root)) and not any(
            source.is_relative_to(root / folder)
            and source != root / folder
            for folder in _COPYABLE_WORKSPACE_FOLDERS
        ):
            raise ValueError("Do not add the workspace as its own source.")
        if source.is_file():
            size = source.stat().st_size
            if not _eligible_file_size(source, project.profile.max_file_bytes):
                plan["files"].append({
                    "path": source.name,
                    "reason_code": (
                        SELECTION_REASON_EMPTY_FILE if size == 0
                        else SELECTION_REASON_FILE_TOO_LARGE
                    ),
                })
                continue
            selected = 1
            kind = "file"
        else:
            excluded = (
                (root.relative_to(source).as_posix(),)
                if root.is_relative_to(source) else ()
            )
            report = scan_repository(
                source,
                extensions=project.profile.extensions,
                max_file_bytes=project.profile.max_file_bytes,
                exclude_paths=excluded,
            )
            if not report["complete"]:
                raise ValueError(
                    f"Could not completely scan source folder: {source}"
                )
            plan["files"].extend({
                "path": f"{source.name}/{row['path']}",
                "reason_code": row["reason_code"],
            } for row in report["entries"] if (
                row["kind"] == "file" and row["reason_code"] in shown
            ))
            selected = len(report["selected_files"])
            if not selected:
                plan["skipped"].append(str(value))
                continue
            kind = "folder"
        name = imported.get(str(source))
        if name is None:
            name = _unique_name(
                source, used, root / PROJECT_SOURCES_DIRECTORY
            )
            state = "renamed" if name != source.name else "new"
            used.add(name)
        else:
            state = "existing"
        plan["add"].append({
            "name": name, "kind": kind, "files": selected,
            "state": state,
        })
        plan["sources"].append(str(value))
    return plan


def _prepare_sources_workspace(
    source_paths: Sequence[str | Path],
    type_id: ProjectTypeId,
    workspace_directory: str | Path,
    *,
    selected_files: Mapping[str, Sequence[str]] | None = None,
    replace_existing: bool = False,
    settings: dict[str, object] | None = None,
    trace: TraceConfig | TraceSink | None = None,
    existing: QAProject | None = None,
    allocate: bool = False,
) -> QAProject:
    """Share validated source copying between explicit create and import.

    Parameters
    ----------
    source_paths: Sequence[str | pathlib.Path]
        Original source inputs.
    type_id: ProjectTypeId
        Registered project type, derived from the saved profile for imports.
    workspace_directory: str | pathlib.Path
        Explicit project destination or preferred allocation path.
    selected_files: Mapping[str, Sequence[str]] | None, optional
        Per-source selections. Defaults to None.
    replace_existing: bool, optional
        Explicit snapshot replacement for imports. Defaults to False.
    settings: dict[str, object] | None, optional
        New-profile overrides only. Defaults to None.
    trace: TraceConfig | TraceSink | None, optional
        Caller-owned diagnostic sink, or a configuration that creates an
        owned project trace after exclusive reservation. Defaults to None.
    existing: QAProject | None, optional
        Explicitly opened destination for an import. Defaults to None.
    allocate: bool, optional
        Reserve a fresh default sibling for creation. Defaults to False.

    Returns
    -------
    QAProject
        Created or deliberately updated project.

    """

    started_at = perf_counter()
    active_trace = (
        trace if trace is not None and not isinstance(trace, TraceConfig)
        else NullTraceLogger()
    )
    owned_trace = None
    owned_creation = False
    if isinstance(trace, TraceConfig) and trace.output_path is not None:
        raise ValueError("Workspace creation allocates its own trace path.")
    if type_id not in project_type_ids():
        raise ValueError(
            f"type_id must be one of: {', '.join(project_type_ids())}."
        )
    if not isinstance(source_paths, (list, tuple)):
        raise ValueError("Choose source files or folders.")
    if type(replace_existing) is not bool:
        raise ValueError("replace_existing must be true or false.")
    if not str(workspace_directory).strip():
        raise ValueError("Choose a workspace folder.")

    workspace = (
        Path(workspace_directory).expanduser().resolve()
        if existing is not None
        else destination_path(
            workspace_directory, allow_occupied_link=allocate
        )
    )
    if workspace in {Path.home().resolve(), Path(workspace.anchor)}:
        raise ValueError(
            "Choose a project workspace, not a home or drive root."
        )
    if not allocate and workspace.exists() and not workspace.is_dir():
        raise NotADirectoryError(f"Choose a workspace folder: {workspace}")
    chosen: list[Path] = []
    for value in source_paths:
        source = Path(value).expanduser()
        if source.is_symlink():
            raise ValueError(f"Do not add a symbolic link: {source}")
        source = source.resolve(strict=True)
        if not source.is_file() and not source.is_dir():
            raise ValueError(f"Choose a regular file or folder: {source}")
        # A file KíLín wrote (outputs/) or a template may become a source;
        # the rest of the workspace (sources, indexes, logs) may not.
        inside = source == workspace or source.is_relative_to(workspace)
        if not allocate and inside and not any(
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

    source_root = workspace / PROJECT_SOURCES_DIRECTORY
    if not allocate and source_root.is_symlink():
        raise ValueError("The workspace sources folder must not be a link.")
    preparation = (
        nullcontext((existing, None))
        if existing is not None
        else _preparing_project(
            workspace, type_id, settings=settings, allocate=allocate,
        )
    )

    def complete(prepared: QAProject) -> None:
        """Record successful source preparation for the resolved project.

        Parameters
        ----------
        prepared: QAProject
            Provisional owned creation or validated existing project.

        """

        active_trace.emit(
            "workspace.prepare.completed",
            component="workspace",
            data={
                "batch_id": batch_id,
                "operation": "import" if existing is not None else "create",
                "project_directory": str(prepared.root),
                "source_directory": str(prepared.source),
                "sources_manifest": str(manifest_path),
                "source_count": len(records),
            },
            duration_ms=(perf_counter() - started_at) * 1000,
        )

    try:
        with preparation as (project, journal):
            workspace = project.root
            source_root = project.source
            # A suggested path can be occupied; only the claimed root defines
            # the creation boundary. Never inspect a skipped project's sources.
            if allocate and any(
                source == workspace or source.is_relative_to(workspace)
                for source in chosen
            ):
                raise ValueError("Do not add the workspace as its own source.")
            if isinstance(trace, TraceConfig):
                if trace.enabled and journal is not None:
                    directory = project.logs / "traces"
                    paths = (
                        directory / "project.json",
                        ProjectTraceSequence(directory).path,
                    )
                    for path in paths:
                        journal.note(path)
                        journal.note(ConversationFileLock(path).lock_path)
                owned_trace = _project_trace_logger(trace, workspace)
                active_trace = owned_trace
                owned_creation = (
                    existing is None and isinstance(owned_trace, TraceLogger)
                )
            batch_id = active_trace.new_id("source_batch")
            active_trace.emit(
                "workspace.prepare.started",
                component="workspace",
                data={
                    "batch_id": batch_id,
                    "operation": (
                        "import" if existing is not None else "create"
                    ),
                    "source_paths": [str(path) for path in source_paths],
                    "type_id": type_id,
                    "workspace_directory": str(workspace),
                    "selected_files": selected_files,
                    "replace_existing": replace_existing,
                    "settings": settings,
                    "effective_profile": asdict(project.profile),
                },
                status="started",
            )
            managed = (workspace / PROJECT_SOURCES_DIRECTORY).resolve(
                strict=True
            )
            if project.source != managed:
                raise ValueError(
                    "project.source must use the managed sources/ folder."
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
                    record = _SourceRecord(
                        original=original, name=name,
                        kind="file" if source.is_file() else "folder",
                    )
                else:
                    record.setdefault(
                        "kind", "file" if source.is_file() else "folder"
                    )
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
                if source.is_file():
                    record["files"] = copied + preserved
                else:
                    inventory = set(record.get("files", [])) | set(copied)
                    record["files"] = sorted(
                        relative
                        for relative in inventory
                        if _is_regular_snapshot(destination, relative)
                    )
                _stamp_copies(record, destination, copied)
                if record not in records:
                    records.append(record)
                    used.add(record["name"])
                if journal is not None:
                    journal.note(manifest_path)
                _write_records(manifest_path, records)
                active_trace.emit(
                    "workspace.source.copied",
                    component="workspace",
                    data={
                        "batch_id": batch_id,
                        "source": str(source),
                        "kind": record["kind"],
                        "workspace_path": destination.relative_to(
                            workspace
                        ).as_posix(),
                        "destination": str(destination),
                        "requested_files": selection,
                        "copied_files": copied,
                        "preserved_files": preserved,
                    },
                )
            if owned_creation:
                # Finalize the owned diagnostics while all their files are
                # still journaled. A strict failure must prevent publication.
                complete(project)
                owned_trace.close()
        prepared = open_project(workspace)
        if not owned_creation:
            # Borrowed sinks cannot be rolled back with a new workspace;
            # report completion only after the profile is published.
            complete(prepared)
            if isinstance(owned_trace, TraceLogger):
                owned_trace.close()
        return prepared
    except BaseException as error:
        if isinstance(owned_trace, TraceLogger):
            # Rollback may already have removed every trace file. Release
            # resources without writing, preserving the initiating failure.
            try:
                owned_trace.close(emit_event=False)
            except BaseException as cleanup_error:
                error.add_note(
                    f"Project trace cleanup also failed: {cleanup_error!r}"
                )
        raise


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

    if (
        source.name not in used
        and source.name != _VERSION_DIRECTORY
        and not (root / source.name).exists()
    ):
        return source.name
    digest = sha256(str(source).encode()).hexdigest()[:8]
    name = f"{source.stem}-{digest}{source.suffix}"
    if name in used or (root / name).exists():
        raise ValueError(f"Could not choose a distinct source name: {source}")
    return name



def _eligible_file_size(source: Path, max_file_bytes: int) -> bool:
    """Check the size policy shared by source imports and synchronization.

    Parameters
    ----------
    source: pathlib.Path
        Existing regular source file.
    max_file_bytes: int
        Maximum size saved in the project profile.

    Returns
    -------
    bool
        Whether the source is nonempty and within the size limit.

    """

    return 0 < source.stat().st_size <= max_file_bytes


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
        if not _eligible_file_size(source, profile.max_file_bytes):
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


def _copy_stamp(path: Path) -> _CopyStamp:
    """Read the size and modification time of one source copy.

    Parameters
    ----------
    path: pathlib.Path
        Regular file inside the managed ``sources/`` folder.

    Returns
    -------
    _CopyStamp
        File size in bytes and modification time in nanoseconds.

    """

    status = path.stat()
    return _CopyStamp(size=status.st_size, mtime_ns=status.st_mtime_ns)


def _stamp_copies(
    record: _SourceRecord, destination: Path, copied: Sequence[str]
) -> None:
    """Keep one stamp per tracked file, renewing those just copied.

    Parameters
    ----------
    record: _SourceRecord
        Source record whose ``files`` inventory is already up to date.
    destination: pathlib.Path
        The source's copy in ``sources/``: a file for a direct-file source,
        otherwise a folder.
    copied: Sequence[str]
        Files written by this import; their stamps are renewed.

    Notes
    -----
    Files kept from an earlier import keep their stamps. Files without one,
    from manifests written before stamps existed, are stamped as they are
    now. Stamps of files no longer tracked are dropped.

    """

    stamps = record.get("stamps", {})
    renewed: dict[str, _CopyStamp] = {}
    for relative in record.get("files", []):
        path = destination if destination.is_file() else destination / relative
        if relative in copied or relative not in stamps:
            renewed[relative] = _copy_stamp(path)
        else:
            renewed[relative] = stamps[relative]
    record["stamps"] = renewed


def _safe_relative(value: str) -> bool:
    """Check that an untrusted manifest path is a normalized POSIX path.

    Parameters
    ----------
    value: str
        Workspace-relative path from the persisted source manifest.

    Returns
    -------
    bool
        Whether the path has safe components and no drive or traversal.

    """

    if not value or "\\" in value or "\0" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and not PureWindowsPath(value).drive
        and all(part not in ("", ".", "..") for part in value.split("/"))
        and path.as_posix() == value
    )


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
    Missing ``files`` and ``stamps`` are accepted for older manifest
    records. Present inventories must contain unique, normalized relative
    paths, and every stamp must belong to a tracked file.

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
    seen_versions: set[str] = set()
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
        kind = item.get("kind")
        if kind is not None and kind not in ("file", "folder"):
            raise ValueError("Invalid workspace source kind.")
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
        stamps = item.get("stamps", {})
        if not isinstance(stamps, dict) or not set(stamps) <= seen_files:
            raise ValueError("Invalid workspace source copy stamps.")
        for stamp in stamps.values():
            if (
                not isinstance(stamp, dict)
                or set(stamp) != {"size", "mtime_ns"}
                or not all(
                    type(value) is int and value >= 0
                    for value in stamp.values()
                )
            ):
                raise ValueError("Invalid workspace source copy stamps.")
        versions = item.get("versions", [])
        if not isinstance(versions, list):
            raise ValueError("Invalid workspace source versions.")
        for version in versions:
            if not isinstance(version, dict) or set(version) != {
                "file", "path", "sha256"
            }:
                raise ValueError("Invalid workspace source versions.")
            relative, stored, digest = (
                version["file"], version["path"], version["sha256"]
            )
            if not isinstance(relative, str) or not _safe_relative(relative):
                raise ValueError("Invalid workspace source version file.")
            prefix = f"{_VERSION_DIRECTORY}/{name}/"
            if (
                not isinstance(stored, str)
                or not _safe_relative(stored)
                or not stored.startswith(prefix)
                or stored in seen_versions
            ):
                raise ValueError("Invalid workspace source version path.")
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError("Invalid workspace source version digest.")
            seen_versions.add(stored)
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
    managed = root / PROJECT_SOURCES_DIRECTORY
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
            or relative.parts[0] != PROJECT_SOURCES_DIRECTORY
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
        if target == managed / _VERSION_DIRECTORY:
            raise ValueError("Remove retained versions individually.")
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
                if destination.is_symlink():
                    record["files"] = []
                elif destination.is_file():
                    record["files"] = [
                        Path(record["original"]).name
                        if Path(record["original"]).name in
                        record.get("files", []) else ""
                    ]
                    record["files"] = [
                        value for value in record["files"] if value
                    ]
                elif destination.is_dir():
                    record["files"] = [
                        item for item in record.get("files", [])
                        if _is_regular_snapshot(destination, item)
                    ]
                else:
                    record["files"] = []
                if "versions" in record:
                    record["versions"] = [
                        version for version in record["versions"]
                        if _is_regular_snapshot(managed, version["path"])
                    ]
                if not record["files"] and not record.get("versions"):
                    continue
                if "stamps" in record:
                    record["stamps"] = {
                        relative: stamp
                        for relative, stamp in record["stamps"].items()
                        if relative in record["files"]
                    }
                retained.append(record)
            _write_records(manifest, retained)
    return [target.relative_to(root).as_posix() for target in targets]


def find_source_changes(project: QAProject) -> SourceChanges:
    """Find added originals and differences from imported source copies.

    Parameters
    ----------
    project: QAProject
        Opened project with managed ``sources/`` snapshots.

    Returns
    -------
    SourceChanges
        Sorted workspace-relative paths categorized as ``added``,
        ``changed``, ``deleted``, or ``edited``.

    Notes
    -----
    Original directories are scanned recursively using the project profile.
    Empty and excluded directories contribute no new files. Missing
    previously tracked originals, including renamed paths, are deletions;
    their new locations are not inferred. A matching size and modification
    time is treated as unchanged without hashing. Detection never writes.

    """

    root = _managed_root(project)
    changes = _empty_changes()
    if root is None:
        return changes
    records = _read_records(root / "sources.json")
    for _record, group, _source, copy, _relative in _differences(
        project, root, records
    ):
        changes[group].append(copy.relative_to(root).as_posix())
    for paths in changes.values():
        paths.sort()
    return changes


def list_source_versions(project: QAProject) -> list[SourceVersion]:
    """List retained source versions with their distinct retrieval paths.

    Parameters
    ----------
    project: QAProject
        Project whose retained copies are being inspected.

    Returns
    -------
    list[SourceVersion]
        Existing regular versions with workspace-relative source, version,
        and saved content digest. Missing or linked files are omitted.

    Notes
    -----
    Version files are separately selectable for retrieval indexing; the
    repository scanner's extension, size, and extraction constraints still
    determine whether a particular version is searchable.

    """

    root = _managed_root(project)
    if root is None:
        return []
    versions: list[SourceVersion] = []
    managed = root / PROJECT_SOURCES_DIRECTORY
    for record in _read_records(root / "sources.json"):
        copy = f"{PROJECT_SOURCES_DIRECTORY}/{record['name']}"
        for item in record.get("versions", []):
            if not _is_regular_snapshot(managed, item["path"]):
                continue
            versions.append(SourceVersion(
                source=copy if record.get("kind") == "file"
                else f"{copy}/{item['file']}",
                version=f"{PROJECT_SOURCES_DIRECTORY}/{item['path']}",
                sha256=item["sha256"],
            ))
    return sorted(versions, key=lambda item: item["version"])


def sync_project_sources(
    project: QAProject,
    *,
    decisions: Mapping[str, SourceAction] | None = None,
    trace: TraceSink | None = None,
) -> SourceChanges:
    """Synchronize selected files and optionally retain indexed versions.

    Parameters
    ----------
    project: QAProject
        Opened project with managed ``sources/`` snapshots.
    decisions: Mapping[str, SourceAction] | None, optional
        Actions keyed by workspace-relative paths from
        ``find_source_changes``. ``import`` accepts new originals;
        ``replace`` updates existing copies; ``keep`` leaves a changed copy
        alone or detaches a deleted original; ``keep_both`` saves the
        previous copy as a separately indexed version before replacement;
        ``delete`` removes a missing original's copy. Unlisted paths are
        untouched. None replaces tracked changes and deletes missing
        copies, while preserving edited and detached copies.
    trace: TraceSink | None, optional
        Caller-owned project diagnostic sink. Defaults to None.

    Returns
    -------
    SourceChanges
        Paths imported, replaced, or removed, plus unchanged edited copies.
        Retained version paths can be read with ``list_source_versions``.

    Raises
    ------
    ValueError
        If a decision is invalid, refers to a no-longer-pending file, or
        would overwrite a user-owned snapshot without explicit permission.
    OSError
        If reading, copying, or saving the source inventory fails.

    Notes
    -----
    Rechecks pending changes before acting and validates all requested
    decisions before writing. Edits to a workspace copy are never discarded
    by the default policy. A restored original with a detached copy needs
    an explicit replacement or ``keep_both`` choice. The latter requires
    an existing original and archives the old copy under
    ``sources/.kilin_versions``. The archive
    is tracked and exposed for indexing as an independent file, not as an
    untracked backup. Identical bytes trigger a metadata-only update rather
    than an unnecessary duplicate. Updated files obey the saved file-size
    limit, as do new imports and retrieval indexing.
    Index refresh remains the caller's responsibility and should happen
    once after the full batch. Multi-file updates are not transactional.

    """

    root = _managed_root(project)
    done = _empty_changes()
    if root is None:
        if decisions:
            raise ValueError("Only managed sources can be synchronized.")
        return done
    manifest = root / "sources.json"
    records = _read_records(manifest)
    pending = _differences(project, root, records)
    changes_by_path = {
        copy.relative_to(root).as_posix(): (
            record, group, source, copy, relative
        )
        for record, group, source, copy, relative in pending
    }
    if decisions is not None:
        if not isinstance(decisions, Mapping):
            raise ValueError("Source decisions must map paths to actions.")
        for path, action in decisions.items():
            if not isinstance(path, str) or path not in changes_by_path:
                raise ValueError(
                    f"Source is not pending synchronization: {path}"
                )
            group = changes_by_path[path][1]
            allowed = {
                "added": ("import", "keep"),
                "changed": ("replace", "keep", "keep_both"),
                "deleted": ("delete", "keep"),
                "edited": ("replace", "keep", "keep_both", "delete"),
            }[group]
            if not isinstance(action, str) or action not in allowed:
                raise ValueError(
                    f"Invalid action for {group} source: {action}"
                )
            source = changes_by_path[path][2]
            if group == "edited" and action in ("replace", "keep_both"):
                if not _regular_original(source):
                    raise ValueError(
                        f"Cannot replace a copy of a deleted source: {path}"
                    )
            if group == "edited" and action == "delete":
                if _regular_original(source):
                    raise ValueError(
                        f"Cannot delete a copy of an existing source: {path}"
                    )
            if action == "import" and (
                changes_by_path[path][3].exists()
                or changes_by_path[path][3].is_symlink()
            ):
                raise ValueError(
                    f"New source conflicts with a workspace copy: {path}"
                )
    for record, group, source, copy, relative in pending:
        path = copy.relative_to(root).as_posix()
        action = (
            decisions.get(path, "keep") if decisions is not None else
            "replace" if group == "changed" and relative in
            record.get("files", []) else "keep"
        )
        if action in ("import", "replace", "keep_both") and (
            _regular_original(source)
            and not _eligible_file_size(
                source, project.profile.max_file_bytes
            )
        ):
            raise ValueError(
                f"Source is empty or exceeds the file-size limit: {source}"
            )
    active_trace = trace if trace is not None else NullTraceLogger()
    batch_id = active_trace.new_id("source_sync")
    active_trace.emit(
        "workspace.sync.started", component="workspace", status="started",
        data={"batch_id": batch_id, "decisions": decisions},
    )
    modified = False
    try:
        for record, group, source, copy, relative in pending:
            path = copy.relative_to(root).as_posix()
            action = (
                decisions.get(path, "keep") if decisions is not None else
                {
                    "added": "keep", "changed": (
                        "replace" if relative in record.get("files", [])
                        else "keep"
                    ),
                    "deleted": "delete", "edited": "keep",
                }[group]
            )
            if action == "keep":
                if decisions is not None and group == "deleted":
                    _untrack_file(record, relative)
                    modified = True
                    active_trace.emit(
                        "workspace.sync.detached", component="workspace",
                        data={"batch_id": batch_id, "path": path},
                    )
                elif group == "edited" and (
                    decisions is None or path in decisions
                ):
                    done["edited"].append(path)
                continue
            if action == "import":
                if (
                    not _regular_original(source)
                    or _linked_source_parent(
                        Path(record["original"]), source
                    )
                ):
                    raise ValueError(
                        f"The newly selected file is no longer safe: {source}"
                    )
                size = source.stat().st_size
                if not 0 < size <= project.profile.max_file_bytes:
                    raise ValueError(
                        f"New file is empty or exceeds the limit: {source}"
                    )
                if copy.exists() or copy.is_symlink():
                    raise ValueError(
                        f"New source conflicts with a workspace copy: {path}"
                    )
                _copy_file(source, copy)
                record["files"] = sorted(
                    set(record.get("files", [])) | {relative}
                )
                record.setdefault("stamps", {})[relative] = _copy_stamp(copy)
                done["added"].append(path)
                modified = True
            elif action in ("replace", "keep_both"):
                if (
                    not _regular_original(source)
                    or _linked_source_parent(
                        Path(record["original"]), source
                    )
                ):
                    raise FileNotFoundError(
                        f"Source disappeared during synchronization: {source}"
                    )
                identical = _same_contents(source, copy)
                if action == "keep_both" and not identical:
                    retained = _archive_source_version(
                        root, record, relative, copy
                    )
                    record.setdefault("versions", []).append(retained)
                    modified = True
                if identical:
                    shutil.copystat(source, copy)
                else:
                    _copy_file(source, copy, replace_existing=True)
                record["files"] = sorted(
                    set(record.get("files", [])) | {relative}
                )
                record.setdefault("stamps", {})[relative] = _copy_stamp(copy)
                done["changed"].append(path)
                modified = True
            elif action == "delete":
                if _regular_original(source) or source.is_symlink():
                    raise ValueError(
                        f"The original source is available again: {source}"
                    )
                copy.unlink()
                _untrack_file(record, relative)
                folder = root / PROJECT_SOURCES_DIRECTORY / record["name"]
                parent = copy.parent
                while (
                    parent != folder and parent.is_relative_to(folder)
                    and not any(parent.iterdir())
                ):
                    parent.rmdir()
                    parent = parent.parent
                done["deleted"].append(path)
                modified = True
            active_trace.emit(
                "workspace.sync.file", component="workspace",
                data={
                    "batch_id": batch_id, "action": action,
                    "path": path, "source": str(source),
                    "original": record["original"],
                    "relative": relative, "kind": record.get("kind"),
                },
            )
    finally:
        if modified:
            _write_records(manifest, records)
    for paths in done.values():
        paths.sort()
    active_trace.emit(
        "workspace.sync.completed", component="workspace",
        data={"batch_id": batch_id, "changes": done},
    )
    return done


def _empty_changes() -> SourceChanges:
    """Create distinct mutable lists for source change categories.

    Returns
    -------
    SourceChanges
        Empty grouped source changes.

    """

    return SourceChanges(added=[], changed=[], deleted=[], edited=[])


def _untrack_file(record: _SourceRecord, relative: str) -> None:
    """Detach one file from an original without removing its copy.

    Parameters
    ----------
    record: _SourceRecord
        Mutable persisted source inventory.
    relative: str
        Previously tracked filename relative to the original.

    """

    record["files"] = [
        item for item in record.get("files", []) if item != relative
    ]
    record.get("stamps", {}).pop(relative, None)


def _regular_original(source: Path) -> bool:
    """Check for an existing regular original without following a link.

    Parameters
    ----------
    source: pathlib.Path
        Original source file to examine.

    Returns
    -------
    bool
        Whether the source path itself is an ordinary regular file.

    """

    try:
        return stat.S_ISREG(source.lstat().st_mode)
    except (FileNotFoundError, NotADirectoryError):
        return False


def _archive_source_version(
    root: Path, record: _SourceRecord, relative: str, copy: Path
) -> _StoredVersion:
    """Store one historical snapshot at a unique, indexable source path.

    Parameters
    ----------
    root: pathlib.Path
        Managed project root.
    record: _SourceRecord
        Original source record owning the version.
    relative: str
        Filename relative to the original directory.
    copy: pathlib.Path
        Existing workspace snapshot to preserve.

    Returns
    -------
    _StoredVersion
        Source-relative archived version path and its content digest.

    Notes
    -----
    A content digest distinguishes versions with identical names. A
    sequence number also permits later restorations of earlier contents.
    Each archive is an independent file under a reserved namespace, not
    a replacement for the current file.

    """

    identity = _hash_file(copy)
    original = PurePosixPath(relative)
    count = sum(
        item["file"] == relative for item in record.get("versions", [])
    )
    while True:
        count += 1
        name = (
            f"{original.stem}__v{count:04d}__{identity[:12]}"
            f"{original.suffix}"
        )
        archived = (
            PurePosixPath(_VERSION_DIRECTORY) / record["name"] /
            original.parent / name
        ).as_posix()
        destination = root / PROJECT_SOURCES_DIRECTORY / archived
        if not destination.exists() and not destination.is_symlink():
            break
    _copy_file(copy, destination)
    if _hash_file(destination) != identity:
        destination.unlink()
        raise OSError("The snapshot changed while creating its version.")
    return _StoredVersion(file=relative, path=archived, sha256=identity)


def _hash_file(path: Path) -> str:
    """Hash a potentially large source copy using bounded reads.

    Parameters
    ----------
    path: pathlib.Path
        Existing file whose content is fingerprinted.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest of the file bytes.

    """

    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_contents(source: Path, copy: Path) -> bool:
    """Compare file contents only after source metadata has changed.

    Parameters
    ----------
    source: pathlib.Path
        Original file that differs from its imported metadata stamp.
    copy: pathlib.Path
        Unedited workspace copy of that original.

    Returns
    -------
    bool
        True if the files have identical content, regardless of timestamps.

    Notes
    -----
    Different sizes avoid hashing. Equal-sized files are hashed in bounded
    blocks to avoid loading their contents into memory. This comparison is
    not performed when a file's size and modification time still match the
    saved stamp.

    """

    if source.stat().st_size != copy.stat().st_size:
        return False
    return _hash_file(source) == _hash_file(copy)


def _managed_root(project: QAProject) -> Path | None:
    """Return the project root when its sources are a managed folder.

    Parameters
    ----------
    project: QAProject
        Opened project.

    Returns
    -------
    pathlib.Path | None
        Resolved project root, or None for external or linked sources.

    """

    if not isinstance(project, QAProject):
        raise TypeError("Expected an opened QAProject.")
    root = project.root.resolve(strict=True)
    managed = root / PROJECT_SOURCES_DIRECTORY
    if (
        managed.is_symlink()
        or not managed.is_dir()
        or project.source.resolve(strict=True) != managed
    ):
        return None
    return root


def _linked_source_parent(original: Path, source: Path) -> bool:
    """Check whether a tracked source passes through a linked subfolder.

    Parameters
    ----------
    original: pathlib.Path
        Root of the imported source, or the directly imported file.
    source: pathlib.Path
        Tracked original file beneath that root.

    Returns
    -------
    bool
        True if a parent of the tracked file, below the source root, is a
        symbolic link. The direct-file case has no such parents.

    """

    parent = source.parent
    while source != original and parent != original:
        if parent.is_symlink():
            return True
        parent = parent.parent
    return False


def _differences(
    project: QAProject, root: Path, records: list[_SourceRecord]
) -> list[
    tuple[
        _SourceRecord,
        Literal["added", "changed", "deleted", "edited"],
        Path,
        Path,
        str,
    ]
]:
    """Compare tracked files and discover eligible new originals recursively.

    Parameters
    ----------
    project: QAProject
        Project whose saved profile determines new-file eligibility.
    root: pathlib.Path
        Resolved root containing the managed ``sources/`` directory.
    records: list[_SourceRecord]
        Validated source ownership records.

    Returns
    -------
    list[tuple]
        Source record, change category, original, copy, and relative name.

    Notes
    -----
    Previously imported files are compared by metadata, while new paths
    are discovered with the same recursive selection rules used at import.
    An inaccessible original aborts scanning rather than suggesting a bulk
    deletion; symlink paths are never followed. When a detached copy has
    a restored eligible original, the pair is reported as changed so a
    caller can explicitly reconcile it without an import collision.

    """

    found = []
    managed = root / PROJECT_SOURCES_DIRECTORY
    for record in records:
        original = Path(record["original"])
        destination = managed / record["name"]
        if original.is_symlink() or destination.is_symlink():
            continue
        kind = record.get("kind")
        if kind is None:
            kind = (
                "file" if destination.is_file() and
                record.get("files") == [original.name] else "folder"
            )
        if kind == "file":
            pairs = [
                (original, destination, original.name)
                for relative in record.get("files", [])
                if relative == original.name and destination.is_file()
            ]
        elif destination.is_dir():
            pairs = [
                (original / relative, destination / relative, relative)
                for relative in record.get("files", [])
                if _is_regular_snapshot(destination, relative)
            ]
        else:
            pairs = []
        for source, copy, relative in pairs:
            stamp = record.get("stamps", {}).get(relative)
            if stamp is None or _linked_source_parent(original, source):
                continue
            try:
                status = source.lstat()
            except (FileNotFoundError, NotADirectoryError):
                group = "deleted"
            else:
                if stat.S_ISLNK(status.st_mode):
                    continue
                if not stat.S_ISREG(status.st_mode):
                    group = "deleted"
                elif (
                    status.st_size == stamp["size"]
                    and status.st_mtime_ns == stamp["mtime_ns"]
                ):
                    continue
                else:
                    group = "changed"
            if _copy_stamp(copy) != stamp:
                group = "edited"
            found.append((record, group, source, copy, relative))

        # Discovery is deliberately independent of archive storage and does
        # not turn inaccessible source folders into an empty selection.
        if kind == "file":
            if (
                not record.get("files") and _regular_original(original)
                and _eligible_file_size(
                    original, project.profile.max_file_bytes
                )
            ):
                group = (
                    "changed" if _is_regular_snapshot(
                        managed, record["name"]
                    ) else "added"
                )
                found.append((
                    record, group, original, destination, original.name
                ))
        elif original.is_dir() and not original.is_symlink():
            excluded = (
                (root.relative_to(original).as_posix(),)
                if root.is_relative_to(original) else ()
            )
            report = scan_repository(
                original,
                extensions=project.profile.extensions,
                max_file_bytes=project.profile.max_file_bytes,
                exclude_paths=excluded,
            )
            if not report["complete"]:
                raise OSError(
                    f"Could not completely scan source folder: {original}"
                )
            tracked = set(record.get("files", []))
            for relative in report["selected_files"]:
                if relative not in tracked:
                    group = (
                        "changed" if _is_regular_snapshot(
                            destination, relative
                        ) else "added"
                    )
                    found.append((
                        record, group, original / relative,
                        destination / relative, relative
                    ))
    return found


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

    Raises
    ------
    ValueError
        If the source manifest or scan entries are malformed.
    OSError
        If an existing source manifest or snapshot cannot be read.

    Notes
    -----
    Direct-file imports and recorded retained versions are selected outside
    normal extension and protected-name rules. Folder imports retain the
    ordinary scanner policy. Unrecorded files in the reserved version folder
    are never selected. File-size and symlink restrictions still apply, and
    an unsupported format may fail during text extraction.

    """

    root = project.root.resolve(strict=True)
    managed = root / PROJECT_SOURCES_DIRECTORY
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
        if isinstance(relative, str) and not relative.startswith(
            f"{_VERSION_DIRECTORY}/"
        ):
            indexed[relative] = dict(item)
    selected = {
        name for name in report.get("selected_files", [])
        if not name.startswith(f"{_VERSION_DIRECTORY}/")
    }
    for record in records:
        name = record["name"]
        snapshot = managed / name
        if not snapshot.is_symlink() and snapshot.is_file():
            size = snapshot.stat().st_size
            if 0 < size <= project.profile.max_file_bytes:
                indexed[name] = {
                    "path": name,
                    "kind": "file",
                    "size_bytes": size,
                    "status": "selected",
                    "reason": "explicitly imported file",
                }
                selected.add(name)
        # Only recorded historical snapshots are added. The reserved
        # version directory is excluded from ordinary recursive scanning.
        for version in record.get("versions", []):
            path = version["path"]
            if not _is_regular_snapshot(managed, path):
                continue
            size = (managed / path).stat().st_size
            if not 0 < size <= project.profile.max_file_bytes:
                continue
            indexed[path] = {
                "path": path,
                "kind": "file",
                "size_bytes": size,
                "status": "selected",
                "reason": "retained source version",
            }
            selected.add(path)
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
