"""Read project profiles and prepare a managed single-source workspace.

``QAProject`` groups an existing project directory, its ``profile.ini``
profile, selected source directory, and managed index/log directories.
Initialization creates a default profile only when one is missing. Opening a
project validates its profile and source location and creates index/log
directories as needed; it is not a strictly read-only operation.

``prepare_project_workspace`` mirrors selected files from one original
repository into a managed workspace and compares source manifests to update
that copy. This is distinct from the cumulative multi-source copying API in
``qachat.workspace``. Neither API provides process or filesystem sandboxing.
Configuration helpers validate settings through the INI reader before writing
the revised profile.

Key Features
------------

- **Typed project records**

    Represent profile settings and resolved workspace paths with dataclasses.

- **Validated profiles**

    Read backend, model, query-format, scan, chunk, retrieval, and generation
    settings.

- **Managed source copy**

    Track and verify selected source and snapshot content using SHA-256.

- **Explicit configuration**

    Update supported profile options while preserving other INI fields.

Module Structure
----------------

- ``PROJECT_KIND_CODE`` and ``PROJECT_KIND_LITERATURE``

    Canonical string values for the two supported project categories.

- ``PROJECT_KIND_CHOICES``

    Ordered collection of the supported project-kind values.

- ``ProjectProfile``

    Settings loaded from one project's profile.ini.

- ``QAProject``

    Resolved paths and settings for one QAChat project.

- ``initialize_project``

    Create a missing project profile, then open the project.

- ``open_project``

    Open a configured project and prepare its output folders.

- ``prepare_project_workspace``

    Mirror one source folder into a managed QAChat workspace.

- ``default_project_profile``

    Return the settings a new project of this kind would receive.

- ``configure_project``

    Validate and save settings without changing source ownership.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
Open an existing workspace:

>>> import qachat
>>> project = qachat.open_project("/path/to/workspace")

Access submodule-only configuration helpers through their actual namespace:

>>> import qachat.project
>>> profile = qachat.project.default_project_profile(
...     qachat.PROJECT_KIND_CODE
... )

"""

from __future__ import annotations

from collections.abc import Mapping
from configparser import ConfigParser
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import StringIO
import json
from math import isfinite
from pathlib import Path
import re
import shutil
from typing import Final, TypeAlias, cast

from qachat.atomic_writes import atomic_write_text
from qachat.model_defaults import (
    DEFAULT_CHAT_MODEL_MLX,
    DEFAULT_CHAT_MODEL_TORCH,
    DEFAULT_EMBEDDING_MODEL_MLX,
    DEFAULT_EMBEDDING_MODEL_TORCH,
)
from qachat.project_type import load_project_types
from qachat.query_formats import (
    QUERY_FORMAT_CHOICES,
    QUERY_FORMAT_QWEN3,
    QueryFormat,
)
from qachat.repository import (
    build_repository_manifest,
    compare_repository_manifests,
    normalize_extensions,
    scan_repository,
)
from qachat.runtime import (
    BACKEND_AUTO,
    BACKEND_CHOICES,
    BACKEND_MLX,
    Backend,
    SelectedBackend,
)
from qachat.trace_logging import NullTraceLogger, TraceSink

__all__ = [
    "PROJECT_KIND_CODE",
    "PROJECT_KIND_LITERATURE",
    "PROJECT_KIND_CHOICES",
    "ProjectProfile",
    "QAProject",
    "initialize_project",
    "open_project",
    "prepare_project_workspace",
    "default_project_profile",
    "configure_project",
]

# Project types, their default files, and their instructions come from
# qachat/project_types/; "code" and "literature" are always present.
ProjectKind: TypeAlias = str
PROJECT_KIND_CODE: Final[ProjectKind] = "code"
PROJECT_KIND_LITERATURE: Final[ProjectKind] = "literature"


def _check_built_in_kinds(types: Mapping[str, object]) -> tuple[str, ...]:
    """Return the configured project kinds, requiring the built-in ones.

    Parameters
    ----------
    types: Mapping[str, object]
        Project types loaded from ``qachat/project_types/``, in file order.

    Returns
    -------
    tuple[str, ...]
        Project kind identifiers in file order.

    Raises
    ------
    ValueError
        If ``code`` or ``literature`` is missing.

    """

    missing = [
        kind
        for kind in (PROJECT_KIND_CODE, PROJECT_KIND_LITERATURE)
        if kind not in types
    ]
    if missing:
        raise ValueError(
            "qachat/project_types/ must keep the built-in types: "
            f"{', '.join(missing)}."
        )

    return tuple(types)


# Loading fails at import when a project type file is malformed or a
# built-in type is missing, so no project opens with a partial set.
PROJECT_KIND_CHOICES: Final[tuple[ProjectKind, ...]] = _check_built_in_kinds(
    load_project_types()
)

_PROJECT_EXTENSIONS: Final[dict[ProjectKind, str]] = {
    kind: ", ".join(item.extensions)
    for kind, item in load_project_types().items()
}
_DEFAULT_FILE_LIMITS: Final[dict[ProjectKind, int]] = {
    kind: item.max_file_bytes for kind, item in load_project_types().items()
}
# Accepted kinds as listed in error messages.
_KIND_CHOICE_TEXT: Final[str] = ", ".join(PROJECT_KIND_CHOICES)
_DEFAULT_PROFILE = """[project]
kind = {kind}
source = {source}
extensions = {extensions}
max_file_bytes = {max_file_bytes}

[models]
backend = {backend}
allow_download = true
chat_mlx = {chat_mlx}
chat_torch = {chat_torch}
embedding_mlx = {embedding_mlx}
embedding_torch = {embedding_torch}
query_format = {query_format}

[chunking]
target_tokens = 300
max_tokens = 400
overlap_tokens = 40

[retrieval]
count = 5

[generation]
thinking = false
temperature = 0.0
max_new_tokens = 256
"""

_EDITABLE_SETTINGS = {
    "extensions": "project",
    "max_file_bytes": "project",
    "backend": "models",
    "allow_download": "models",
    "chat_mlx": "models",
    "chat_torch": "models",
    "embedding_mlx": "models",
    "embedding_torch": "models",
    "query_format": "models",
    "target_tokens": "chunking",
    "max_tokens": "chunking",
    "overlap_tokens": "chunking",
    "retrieval_count": "retrieval",
    "thinking": "generation",
    "temperature": "generation",
    "max_new_tokens": "generation",
}


def _default_profile_text(
    kind: ProjectKind,
    source: str,
    extensions: str,
) -> str:
    """Format the built-in profile with shared core defaults.

    Parameters
    ----------
    kind: ProjectKind
        Project category written to the profile.
    source: str
        Source path written to the profile.
    extensions: str
        Comma-separated source extensions written to the profile.

    Returns
    -------
    str
        Complete profile text using the current shared backend, model, query,
        chunking, retrieval, and generation defaults.

    """

    return _DEFAULT_PROFILE.format(
        kind=kind,
        source=source,
        extensions=extensions,
        max_file_bytes=_DEFAULT_FILE_LIMITS[kind],
        backend=BACKEND_AUTO,
        chat_mlx=DEFAULT_CHAT_MODEL_MLX,
        chat_torch=DEFAULT_CHAT_MODEL_TORCH,
        embedding_mlx=DEFAULT_EMBEDDING_MODEL_MLX,
        embedding_torch=DEFAULT_EMBEDDING_MODEL_TORCH,
        query_format=QUERY_FORMAT_QWEN3,
    )


def default_project_profile(kind: ProjectKind) -> ProjectProfile:
    """Return the settings a new project of this kind would receive.

    Parameters
    ----------
    kind: ProjectKind
        Project category, using ``PROJECT_KIND_CODE`` or
        ``PROJECT_KIND_LITERATURE``.

    Returns
    -------
    ProjectProfile
        The settings a new project of this kind would receive.

    Raises
    ------
    ValueError
        If ``kind`` is not one of ``PROJECT_KIND_CHOICES``.

    Notes
    -----
    This parses the built-in profile template without touching a project
    directory or loading models.

    """

    if kind not in PROJECT_KIND_CHOICES:
        raise ValueError(f"kind must be one of: {_KIND_CHOICE_TEXT}.")
    extensions = _PROJECT_EXTENSIONS[kind]
    parser = ConfigParser()
    parser.read_string(_default_profile_text(kind, "sources", extensions))
    return _read_profile(parser)


def configure_project(
    directory: str | Path,
    settings: dict[str, object],
    *,
    trace: TraceSink | None = None,
) -> QAProject:
    """Validate and save settings without changing source ownership.

    Parameters
    ----------
    directory: str | pathlib.Path
        Local project directory, with relative paths interpreted from the
        current working directory.
    settings: dict[str, object]
        Dictionary of supported profile options to update. Unknown names are
        rejected; values are converted into INI text and then revalidated.
    trace: TraceSink | None, optional
        Optional caller-owned trace sink for configuration-change events. None
        disables those events through a null sink. Defaults to ``None``.

    Returns
    -------
    QAProject
        Reopened QAProject containing the validated updated profile.

    Raises
    ------
    ValueError
        If a setting key or resulting profile value is invalid.
    OSError
        If the profile or managed directories cannot be read or written.

    Notes
    -----
    The profile is validated before replacing profile.ini. A trace error after
    writing does not roll back the file. This updates configuration only; it
    does not reconfigure a loaded RAGSession or rebuild indexes in place.

    """

    active_trace = trace if trace is not None else NullTraceLogger()
    project = open_project(directory)
    before = asdict(project.profile)
    active_trace.emit(
        "project.configuration.requested",
        component="project",
        data={
            "project_directory": str(project.root),
            "requested_settings": settings,
            "before": before,
        },
    )
    parser = ConfigParser()
    parser.read(project.root / "profile.ini", encoding="utf-8")
    for key, value in settings.items():
        section = _EDITABLE_SETTINGS.get(key)
        if section is None:
            raise ValueError(f"Unknown project setting: {key}")
        option = "count" if key == "retrieval_count" else key
        if key == "extensions" and isinstance(value, (list, tuple)):
            value = ", ".join(value)
        parser.set(section, option, str(value))
    _read_profile(parser)
    path = project.root / "profile.ini"
    stream = StringIO()
    parser.write(stream)
    atomic_write_text(path, stream.getvalue())
    configured = open_project(project.root)
    effective = asdict(configured.profile)
    active_trace.emit(
        "project.configuration.saved",
        component="project",
        data={
            "profile_path": str(path),
            "before": before,
            "effective": effective,
        },
    )
    active_trace.settings_changed(
        "project.settings.changed",
        component="project",
        before=before,
        after=effective,
        data={"profile_path": str(path)},
    )
    return configured


@dataclass(frozen=True, slots=True)
class ProjectProfile:
    """Settings loaded from one project's profile.ini.

    Parameters
    ----------
    kind: ProjectKind
        Type of material in the project.
    source: str
        Source folder, relative to the project or absolute.
    extensions: tuple[str, ...]
        File extensions included during scanning.
    max_file_bytes: int
        Largest source file read by QAChat.
    backend: "auto" | "mlx" | "torch"
        Preferred calculation backend.
    allow_download: bool
        Whether Hugging Face may download missing model files.
    chat_mlx: str
        Chat model IDs for the two backends.
    chat_torch: str
        Chat model IDs for the two backends.
    embedding_mlx: str
        Embedding model IDs for the two backends.
    embedding_torch: str
        Embedding model IDs for the two backends.
    query_format: "plain" | "qwen3"
        Query format expected by the embedding model.
    target_tokens: int
        Chunk size settings.
    max_tokens: int
        Chunk size settings.
    overlap_tokens: int
        Chunk size settings.
    retrieval_count: int
        Number of results requested from each retrieval stage.
    thinking: bool
        Whether compatible chat templates enable thinking.
    temperature: float
        Chat generation temperature.
    max_new_tokens: int | None
        Optional maximum generated tokens per answer.

    Attributes
    ----------
    kind: ProjectKind
        Type of material in the project.
    source: str
        Source folder, relative to the project or absolute.
    extensions: tuple[str, ...]
        File extensions included during scanning.
    max_file_bytes: int
        Largest source file read by QAChat.
    backend: "auto" | "mlx" | "torch"
        Preferred calculation backend.
    allow_download: bool
        Whether Hugging Face may download missing model files.
    chat_mlx: str
        Chat model IDs for the two backends.
    chat_torch: str
        Chat model IDs for the two backends.
    embedding_mlx: str
        Embedding model IDs for the two backends.
    embedding_torch: str
        Embedding model IDs for the two backends.
    query_format: "plain" | "qwen3"
        Query format expected by the embedding model.
    target_tokens: int
        Chunk size settings.
    max_tokens: int
        Chunk size settings.
    overlap_tokens: int
        Chunk size settings.
    retrieval_count: int
        Number of results requested from each retrieval stage.
    thinking: bool
        Whether compatible chat templates enable thinking.
    temperature: float
        Chat generation temperature.
    max_new_tokens: int | None
        Optional maximum generated tokens per answer.

    Methods
    -------
    chat_model(backend)
        Return the chat model configured for one selected backend.
    embedding_model(backend)
        Return the embedding model configured for one selected backend.

    Notes
    -----
    Values normally come from the validated INI reader. Direct dataclass
    construction is not validated. Model accessors return the configured MLX
    value only for backend="mlx" and the PyTorch value otherwise. Changes
    require constructing a new profile; the dataclass fields are frozen.

    """

    kind: ProjectKind
    source: str
    extensions: tuple[str, ...]
    max_file_bytes: int
    backend: Backend
    allow_download: bool
    chat_mlx: str
    chat_torch: str
    embedding_mlx: str
    embedding_torch: str
    query_format: QueryFormat
    target_tokens: int
    max_tokens: int
    overlap_tokens: int
    retrieval_count: int
    thinking: bool
    temperature: float
    max_new_tokens: int | None

    def chat_model(self, backend: SelectedBackend) -> str:
        """Return the chat model configured for one selected backend.

        Parameters
        ----------
        backend: SelectedBackend
            Already selected "mlx" or "torch" backend; this accessor does not
            validate the value.

        Returns
        -------
        str
            The chat model configured for one selected backend.

        """

        return self.chat_mlx if backend == BACKEND_MLX else self.chat_torch

    def embedding_model(self, backend: SelectedBackend) -> str:
        """Return the embedding model configured for one selected backend.

        Parameters
        ----------
        backend: SelectedBackend
            Already selected "mlx" or "torch" backend; this accessor does not
            validate the value.

        Returns
        -------
        str
            The embedding model configured for one selected backend.

        """

        return (
            self.embedding_mlx
            if backend == BACKEND_MLX
            else self.embedding_torch
        )


@dataclass(frozen=True, slots=True)
class QAProject:
    """Resolved paths and settings for one QAChat project.

    Parameters
    ----------
    root: pathlib.Path
        Project folder containing profile.ini.
    source: pathlib.Path
        Folder containing source code or literature documents.
    indexes: pathlib.Path
        Folder containing reusable retrieval indexes.
    logs: pathlib.Path
        Folder containing persistent conversation records.
    profile: ProjectProfile
        Parsed project settings.

    Attributes
    ----------
    root: pathlib.Path
        Project folder containing profile.ini.
    source: pathlib.Path
        Folder containing source code or literature documents.
    indexes: pathlib.Path
        Folder containing reusable retrieval indexes.
    logs: pathlib.Path
        Folder containing persistent conversation records.
    profile: ProjectProfile
        Parsed project settings.

    """

    root: Path
    source: Path
    indexes: Path
    logs: Path
    profile: ProjectProfile


def initialize_project(
    directory: str | Path,
    kind: ProjectKind,
) -> QAProject:
    """Create a missing project profile, then open the project.

    Parameters
    ----------
    directory: str | pathlib.Path
        Existing project, repository, or document folder.
    kind: ProjectKind
        Type of source material in the project.

    Returns
    -------
    QAProject
        Initialized and opened project.

    Raises
    ------
    FileNotFoundError
        If the selected directory does not exist.
    NotADirectoryError
        If the selected path is not a directory.
    ValueError
        If kind is invalid or conflicts with an existing profile.

    Notes
    -----
    Requires an existing directory. A missing profile.ini is initialized using
    the selected kind; an existing profile is preserved and checked for kind
    compatibility. Opening the resulting project creates indexes/logs
    directories. No model is loaded.

    """

    if kind not in PROJECT_KIND_CHOICES:
        raise ValueError(f"kind must be one of: {_KIND_CHOICE_TEXT}.")
    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Choose a project folder: {root}")

    profile_path = root / "profile.ini"
    if not profile_path.exists():
        source_name = (
            "repository" if kind == PROJECT_KIND_CODE else "documents"
        )
        if (root / "sources").is_dir():
            source = "sources"
        else:
            source = source_name if (root / source_name).is_dir() else "."

        extensions = _PROJECT_EXTENSIONS[kind]
        profile_path.write_text(
            _default_profile_text(kind, source, extensions),
            encoding="utf-8",
        )

    project = open_project(root)
    if project.profile.kind != kind:
        raise ValueError(
            "PROJECT_KIND conflicts with the existing profile.ini."
        )
    return project


def prepare_project_workspace(
    source_directory: str | Path,
    kind: ProjectKind,
    workspace_root: str | Path,
) -> QAProject:
    """Mirror one source folder into a managed QAChat workspace.

    The original source is never modified.  A normal workspace is named
    ``project_<source-name>`` and contains the copied source folder beside
    ``indexes/``, ``logs/``, ``profile.ini``, and ``source_snapshot.json``.
    Only files selected by the project's repository rules are mirrored.

    Reopening the same source verifies the managed copy against its saved
    SHA-256 manifest. If the copy differs, it is rebuilt from the original;
    otherwise only added, modified, and removed source files are synchronized.
    If another source folder with the same basename already owns the preferred
    workspace name, a short path hash is appended to avoid a collision.

    Parameters
    ----------
    source_directory: str | pathlib.Path
        Existing source code or literature folder.
    kind: ProjectKind
        Type of source material in the project.
    workspace_root: str | pathlib.Path
        QAChat sandbox folder that contains managed project workspaces.

    Returns
    -------
    QAProject
        Project whose source points to the synchronized sandbox snapshot.

    Raises
    ------
    ValueError
        If kind is unsupported, source/workspace locations conflict, an
        existing managed profile is inconsistent, or a unique workspace
        location cannot be selected.
    NotADirectoryError
        If the original source is not a directory.
    OSError
        If scanning, hashing, copying, deleting, or writing metadata fails.

    Notes
    -----
    This mirrors one original repository using its selected-source manifest. It
    is not the cumulative multi-source API in qachat.workspace. The current
    implementation requires complete source and snapshot scans and verifies
    copied content on each reopen. This adds disk I/O proportional to selected
    source size. Synchronization is not atomic across snapshot and manifest,
    and source copying is organization, not execution sandboxing.

    """

    if kind not in PROJECT_KIND_CHOICES:
        raise ValueError(f"kind must be one of: {_KIND_CHOICE_TEXT}.")

    source = Path(source_directory).expanduser().resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(f"Choose a source folder: {source}")

    sandbox = Path(workspace_root).expanduser()
    sandbox.mkdir(parents=True, exist_ok=True)
    sandbox = sandbox.resolve(strict=True)
    if sandbox == source:
        raise ValueError("The workspace folder must differ from the source.")

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", source.name).strip("-._")
    if not safe_name:
        safe_name = "project"

    preferred_root = sandbox / f"project_{safe_name}"
    root = _select_workspace_root(preferred_root, source)
    if root.is_symlink():
        raise ValueError("Managed workspace root must not be a symbolic link.")
    root.mkdir(parents=True, exist_ok=True)

    # Keep the source snapshot as a sibling of indexes/ and logs/.  Preserve
    # the original folder name so the sandbox structure is easy to inspect.
    source_name = source.name
    snapshot = root / source_name
    if snapshot.is_symlink():
        raise ValueError("Managed source copy must not be a symbolic link.")
    snapshot.mkdir(parents=True, exist_ok=True)

    profile_path = root / "profile.ini"
    if not profile_path.exists():

        extensions = _PROJECT_EXTENSIONS[kind]
        profile_path.write_text(
            _default_profile_text(kind, source_name, extensions),
            encoding="utf-8",
        )

    project = open_project(root)
    if project.profile.kind != kind:
        raise ValueError(
            "PROJECT_KIND conflicts with the existing workspace profile.ini."
        )
    if project.source != snapshot.resolve():
        raise ValueError(
            "Managed workspace profile.ini must keep project.source set to "
            f"{source_name}."
        )

    excludes: tuple[str, ...] = ()
    if sandbox.is_relative_to(source):
        excludes = (sandbox.relative_to(source).as_posix(),)

    report = scan_repository(
        source,
        extensions=project.profile.extensions,
        exclude_paths=excludes,
        max_file_bytes=project.profile.max_file_bytes,
    )
    current_manifest = build_repository_manifest(report)
    snapshot_manifest_path = root / "source_snapshot.json"
    saved_manifest = _read_json_if_possible(snapshot_manifest_path)
    try:
        changes = compare_repository_manifests(
            saved_manifest,
            current_manifest,
        )
    except ValueError:
        saved_manifest = None
        changes = compare_repository_manifests(None, current_manifest)

    # The saved source manifest alone cannot detect edits to the managed
    # copy. Verify selected copied bytes before using the incremental diff.
    # A divergent snapshot is rebuilt, not synchronized on top of untrusted
    # content (including unexpected selected files and symbolic links).
    if saved_manifest is not None and not _snapshot_matches_manifest(
        snapshot, project.profile, saved_manifest
    ):
        saved_manifest = None
        changes = compare_repository_manifests(None, current_manifest)

    if not bool(changes["matches"]):
        if saved_manifest is None:
            shutil.rmtree(snapshot)
            snapshot.mkdir(parents=True, exist_ok=True)

        for relative in changes["removed"]:
            target = snapshot / str(relative)
            if target.is_file() or target.is_symlink():
                target.unlink()

        for relative in [*changes["added"], *changes["modified"]]:
            source_path = source / str(relative)
            target = snapshot / str(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)

        _remove_empty_directories(snapshot)

    # Always record ownership, including the zero-file case.  This lets the
    # next open distinguish two different source folders with the same name.
    manifest_payload = {
        **current_manifest,
        "source_directory": str(source),
    }
    _write_json(snapshot_manifest_path, manifest_payload)

    return open_project(root)


def _snapshot_matches_manifest(
    snapshot: Path,
    profile: ProjectProfile,
    manifest: dict[str, object],
) -> bool:
    """Verify that selected copied files still match their saved hashes.

    Parameters
    ----------
    snapshot: pathlib.Path
        Managed copy of the original source directory.
    profile: ProjectProfile
        Profile defining the literal extensions and file-size limit.
    manifest: dict[str, object]
        Previously validated source manifest used to construct the copy.

    Returns
    -------
    bool
        True when the selected copied content exactly matches the manifest.

    Raises
    ------
    ValueError
        If the snapshot cannot be scanned completely.
    OSError
        If files cannot be read or change while their hashes are computed.

    Notes
    -----
    Selected files are rehashed on each workspace preparation, including
    when the source manifest is unchanged. Unselected files are not part of
    this integrity contract. A mismatched copy is rebuilt by the caller.

    """

    report = scan_repository(
        snapshot,
        extensions=profile.extensions,
        max_file_bytes=profile.max_file_bytes,
    )
    copied_manifest = build_repository_manifest(report)
    return bool(compare_repository_manifests(manifest, copied_manifest)[
        "matches"
    ])


def _select_workspace_root(preferred_root: Path, source: Path) -> Path:
    """Choose a readable workspace name without a source collision.

    Parameters
    ----------
    preferred_root: pathlib.Path
        Preferred managed-workspace directory beneath the supplied workspace
        root.
    source: pathlib.Path
        Resolved original source directory whose identity is recorded in a
        workspace manifest.

    Returns
    -------
    pathlib.Path
        Preferred root when compatible with the original source, otherwise a
        source-path-hashed collision root.

    Raises
    ------
    ValueError
        If the deterministic collision root already belongs to another recorded
        source.

    Notes
    -----
    Ownership is inferred from stored source_directory metadata. Missing or
    malformed metadata is not a general proof that an existing directory is
    safe to replace.

    """

    if not preferred_root.exists():
        return preferred_root

    metadata = _read_json_if_possible(preferred_root / "source_snapshot.json")
    if metadata is not None and metadata.get("source_directory") == str(
        source
    ):
        return preferred_root

    identity = sha256(str(source).encode("utf-8")).hexdigest()[:8]
    collision_root = preferred_root.with_name(
        f"{preferred_root.name}-{identity}"
    )
    collision_metadata = _read_json_if_possible(
        collision_root / "source_snapshot.json"
    )
    if (
        collision_root.exists()
        and collision_metadata is not None
        and collision_metadata.get("source_directory") != str(source)
    ):
        raise ValueError(
            "Could not choose a unique QAChat workspace for this source."
        )
    return collision_root


def open_project(directory: str | Path) -> QAProject:
    """Open a configured project and prepare its output folders.

    Parameters
    ----------
    directory: str | pathlib.Path
        Folder containing profile.ini.

    Returns
    -------
    QAProject
        Resolved project paths and checked settings.

    Raises
    ------
    FileNotFoundError
        If the project, profile, or configured source folder is missing.
    NotADirectoryError
        If the project or source path is not a folder.
    ValueError
        If a profile setting is missing or invalid.

    Notes
    -----
    The root and configured source must already be directories and profile.ini
    must exist. A profile may name an absolute source outside the project. The
    function creates indexes and logs directories before returning; it does not
    load models.

    """

    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Choose a project folder: {root}")

    profile_path = root / "profile.ini"
    if not profile_path.is_file():
        raise FileNotFoundError(f"Missing project profile: {profile_path}")

    parser = ConfigParser()
    parser.read(profile_path, encoding="utf-8")
    profile = _read_profile(parser)
    source = Path(profile.source).expanduser()
    if not source.is_absolute():
        source = root / source
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(f"Choose a source folder: {source}")

    indexes = root / "indexes"
    logs = root / "logs"
    indexes.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    return QAProject(
        root=root,
        source=source,
        indexes=indexes,
        logs=logs,
        profile=profile,
    )


def _read_json_if_possible(path: Path) -> dict[str, object] | None:
    """Return one JSON object, or ``None`` when it is missing or invalid.

    Parameters
    ----------
    path: pathlib.Path
        Local JSON path to read or replace.

    Returns
    -------
    dict[str, object] | None
        Decoded dictionary when an existing file can be read as a JSON object,
        otherwise None.

    Notes
    -----
    This helper deliberately treats missing, unreadable, and malformed JSON as
    an unavailable baseline.

    """

    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, value: dict[str, object]) -> None:
    """Atomically write one readable JSON object.

    Parameters
    ----------
    path: pathlib.Path
        Local JSON path to read or replace.
    value: dict[str, object]
        Dictionary serialized into the JSON file.

    Notes
    -----
    The implementation writes through a temporary file and replaces the target.
    It does not lock against other writers or commit related files as a group.

    """

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True) + "\n"
    )


def _remove_empty_directories(root: Path) -> None:
    """Remove empty snapshot directories left after deleted source files.

    Parameters
    ----------
    root: pathlib.Path
        Directory whose empty descendant directories should be removed.

    Notes
    -----
    Directories are attempted deepest first. Nonempty or otherwise unremovable
    directories are left in place; OSError is suppressed.

    """

    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass


def _read_profile(parser: ConfigParser) -> ProjectProfile:
    """Read and validate every supported project setting.

    Parameters
    ----------
    parser: ConfigParser
        ConfigParser containing the profile sections and values.

    Returns
    -------
    ProjectProfile
        ProjectProfile with parsed source, backend, model, chunking, retrieval,
        and generation fields.

    Raises
    ------
    ValueError
        If required profile values are missing, unsupported, malformed, or
        violate the implemented numeric bounds.

    Notes
    -----
    Temperatures must be finite and nonnegative. Literal extension grammar
    is shared with the repository scanner to reject invalid settings early.

    """

    kind_value = _required(parser, "project", "kind")
    if kind_value not in PROJECT_KIND_CHOICES:
        raise ValueError(f"project.kind must be one of: {_KIND_CHOICE_TEXT}.")
    kind = cast("ProjectKind", kind_value)

    extensions = tuple(
        part.strip()
        for part in _required(
            parser,
            "project",
            "extensions",
        ).split(",")
        if part.strip()
    )
    if not extensions or any(
        not extension.startswith(".") for extension in extensions
    ):
        raise ValueError("project.extensions must contain values like .py.")
    try:
        extensions = normalize_extensions(extensions)
    except ValueError as error:
        raise ValueError(
            f"project.extensions must contain literal suffixes: {error}"
        ) from error

    backend_value = _required(parser, "models", "backend")
    if backend_value not in BACKEND_CHOICES:
        raise ValueError("models.backend must be auto, mlx, or torch.")
    backend = cast("Backend", backend_value)

    query_value = _required(parser, "models", "query_format")
    if query_value not in QUERY_FORMAT_CHOICES:
        raise ValueError("models.query_format must be plain or qwen3.")
    query_format = cast("QueryFormat", query_value)

    target_tokens = _positive_integer(
        parser,
        "chunking",
        "target_tokens",
    )
    max_tokens = _positive_integer(parser, "chunking", "max_tokens")
    overlap_tokens = _non_negative_integer(
        parser,
        "chunking",
        "overlap_tokens",
    )
    if max_tokens < target_tokens:
        raise ValueError("chunking.max_tokens must reach target_tokens.")
    if overlap_tokens >= target_tokens:
        raise ValueError(
            "chunking.overlap_tokens must be below target_tokens."
        )

    temperature = _float(parser, "generation", "temperature")
    if not isfinite(temperature):
        raise ValueError("generation.temperature must be finite.")
    if temperature < 0:
        raise ValueError("generation.temperature cannot be negative.")

    return ProjectProfile(
        kind=kind,
        source=_required(parser, "project", "source"),
        extensions=extensions,
        max_file_bytes=_positive_integer(
            parser,
            "project",
            "max_file_bytes",
        ),
        backend=backend,
        allow_download=_boolean(
            parser,
            "models",
            "allow_download",
        ),
        chat_mlx=_required(parser, "models", "chat_mlx"),
        chat_torch=_required(parser, "models", "chat_torch"),
        embedding_mlx=_required(parser, "models", "embedding_mlx"),
        embedding_torch=_required(parser, "models", "embedding_torch"),
        query_format=query_format,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        retrieval_count=_positive_integer(
            parser,
            "retrieval",
            "count",
        ),
        thinking=_boolean(parser, "generation", "thinking"),
        temperature=temperature,
        max_new_tokens=_optional_integer(
            parser,
            "generation",
            "max_new_tokens",
        ),
    )


def _required(parser: ConfigParser, section: str, option: str) -> str:
    """Return one required, non-empty profile value.

    Parameters
    ----------
    parser: ConfigParser
        ConfigParser containing the profile sections and values.
    section: str
        INI section containing the requested option.
    option: str
        Name of the option within that INI section.

    Returns
    -------
    str
        One required, non-empty profile value.

    Raises
    ------
    ValueError
        If the section/option is missing or its stripped text is empty.

    """

    if not parser.has_option(section, option):
        raise ValueError(f"Missing profile setting: {section}.{option}.")
    value = parser.get(section, option).strip()
    if not value:
        raise ValueError(f"Profile setting {section}.{option} is empty.")
    return value


def _positive_integer(
    parser: ConfigParser,
    section: str,
    option: str,
) -> int:
    """Return one positive whole-number profile value.

    Parameters
    ----------
    parser: ConfigParser
        ConfigParser containing the profile sections and values.
    section: str
        INI section containing the requested option.
    option: str
        Name of the option within that INI section.

    Returns
    -------
    int
        One positive whole-number profile value.

    Raises
    ------
    ValueError
        If the option is not an integer or is not positive.

    """

    value = _integer(parser, section, option)
    if value < 1:
        raise ValueError(f"{section}.{option} must be positive.")
    return value


def _non_negative_integer(
    parser: ConfigParser,
    section: str,
    option: str,
) -> int:
    """Return one non-negative whole-number profile value.

    Parameters
    ----------
    parser: ConfigParser
        ConfigParser containing the profile sections and values.
    section: str
        INI section containing the requested option.
    option: str
        Name of the option within that INI section.

    Returns
    -------
    int
        One non-negative whole-number profile value.

    Raises
    ------
    ValueError
        If the option is not an integer or is negative.

    """

    value = _integer(parser, section, option)
    if value < 0:
        raise ValueError(f"{section}.{option} cannot be negative.")
    return value


def _integer(parser: ConfigParser, section: str, option: str) -> int:
    """Return one whole-number profile value.

    Parameters
    ----------
    parser: ConfigParser
        ConfigParser containing the profile sections and values.
    section: str
        INI section containing the requested option.
    option: str
        Name of the option within that INI section.

    Returns
    -------
    int
        One whole-number profile value.

    Raises
    ------
    ValueError
        If the option is missing, empty, or not a valid integer.

    """

    value = _required(parser, section, option)
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"{section}.{option} must be a whole number."
        ) from error


def _optional_integer(
    parser: ConfigParser,
    section: str,
    option: str,
) -> int | None:
    """Return a positive integer or None from one profile value.

    Parameters
    ----------
    parser: ConfigParser
        ConfigParser containing the profile sections and values.
    section: str
        INI section containing the requested option.
    option: str
        Name of the option within that INI section.

    Returns
    -------
    int | None
        A positive integer or None from one profile value.

    Raises
    ------
    ValueError
        If the option is neither the supported none spelling nor a positive
        integer.

    """

    value = _required(parser, section, option)
    if value.casefold() == "none":
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(
            f"{section}.{option} must be positive or none."
        ) from error
    if parsed < 1:
        raise ValueError(f"{section}.{option} must be positive or none.")
    return parsed


def _boolean(parser: ConfigParser, section: str, option: str) -> bool:
    """Return one Boolean profile value.

    Parameters
    ----------
    parser: ConfigParser
        ConfigParser containing the profile sections and values.
    section: str
        INI section containing the requested option.
    option: str
        Name of the option within that INI section.

    Returns
    -------
    bool
        One Boolean profile value.

    Raises
    ------
    ValueError
        If the option is missing or is not a supported ConfigParser Boolean
        spelling.

    """

    _required(parser, section, option)
    try:
        return parser.getboolean(section, option)
    except ValueError as error:
        raise ValueError(
            f"{section}.{option} must be true or false."
        ) from error


def _float(parser: ConfigParser, section: str, option: str) -> float:
    """Return one numeric profile value.

    Parameters
    ----------
    parser: ConfigParser
        ConfigParser containing the profile sections and values.
    section: str
        INI section containing the requested option.
    option: str
        Name of the option within that INI section.

    Returns
    -------
    float
        One numeric profile value.

    Raises
    ------
    ValueError
        If the required option cannot be converted to float.

    Notes
    -----
    Conversion alone permits non-finite values; ``_read_profile`` rejects
    them after parsing.

    """

    value = _required(parser, section, option)
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{section}.{option} must be a number.") from error
