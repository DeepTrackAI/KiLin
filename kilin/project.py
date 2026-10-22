"""Create and open project workspaces independently of the GUI.

``QAProject`` groups an existing project directory, its ``profile.ini``
profile, selected source directory, and managed index/log directories.
Initialization prepares sources and required resources first, then commits
a missing profile atomically as its completion marker, using
``PROJECT_PROFILE`` in the packaged defaults. Opening a project validates its
profile and source location and creates the shared indexes, logs, outputs,
and templates folders as needed. A new project receives only the packaged
resources declared by the tools selected by its project type.

``prepare_project_workspace`` mirrors selected files from one original
repository into a managed workspace and compares source manifests to update
that copy. This is distinct from the cumulative multi-source copying API in
``kilin.workspace``. Neither API provides process or filesystem sandboxing.
Configuration helpers validate settings through the INI reader before writing
the revised profile.

Key Features
------------

- **Typed project records**

    Represent profile settings and resolved workspace paths with dataclasses.

- **Validated profiles**

    Read backend, model, query-format, scan, chunk, retrieval, and generation
    settings.

- **GUI-independent workspace layout**

    Create shared folders and install selected tool resources in the core.

- **Managed source copy**

    Track and verify selected source and snapshot content using SHA-256.

- **Explicit configuration**

    Update supported profile options while preserving other INI fields.

Module Structure
----------------

- ``PROJECT_TYPE_ID_CODE`` and ``PROJECT_TYPE_ID_LITERATURE``

    String identifiers for the standard packaged project types.

- ``PROJECT_TYPE_IDS``

    All packaged type IDs in deterministic filename order, not an enum.

- ``ProjectTypeId``

    Open string alias for a registered project-type identifier.

- ``ProjectProfile``

    Settings loaded from one project's profile.ini.

- ``QAProject``

    Resolved paths and settings for one KíLín project.

- ``initialize_project``

    Create a missing profile and install selected tool resources.

- ``open_project``

    Open a project and ensure its shared workspace folders exist.

- ``prepare_project_workspace``

    Mirror one source folder into a managed KíLín workspace.

- ``default_project_profile``

    Return the default profile for one registered project type.

- ``configure_project``

    Validate and save settings without changing source ownership.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
Open an existing workspace:

>>> import kilin
>>> project = kilin.open_project("/path/to/workspace")

Access submodule-only configuration helpers through their actual namespace:

>>> import kilin.project
>>> profile = kilin.project.default_project_profile(
...     kilin.PROJECT_TYPE_ID_CODE
... )

"""

from __future__ import annotations

from collections.abc import Mapping
from configparser import ConfigParser
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import StringIO
import json
from math import isfinite
from pathlib import Path
import re
import shutil
from typing import Final, Iterator, cast

from kilin.atomic_writes import atomic_write_text
from kilin.defaults import (
    load_defaults_section,
    require_default_bool,
    require_default_int,
    require_default_nonnegative_int,
    require_default_number,
)
from kilin.model_defaults import (
    DEFAULT_CHAT_MODEL_MLX,
    DEFAULT_CHAT_MODEL_TORCH,
    DEFAULT_EMBEDDING_MODEL_MLX,
    DEFAULT_EMBEDDING_MODEL_TORCH,
)
from kilin.project_type import (
    ProjectTypeId,
    load_project_types,
    project_type,
)
from kilin.project_layout import (
    ensure_project_folders,
    install_project_resources,
    plan_project_layout,
)
from kilin.query_formats import (
    QUERY_FORMAT_CHOICES,
    QUERY_FORMAT_QWEN3,
    QueryFormat,
)
from kilin.repository import (
    build_repository_manifest,
    compare_repository_manifests,
    normalize_extensions,
    scan_repository,
)
from kilin.runtime import (
    BACKEND_AUTO,
    BACKEND_CHOICES,
    BACKEND_MLX,
    Backend,
    SelectedBackend,
)
from kilin.trace_logging import NullTraceLogger, TraceSink

__all__ = [
    "PROJECT_TYPE_ID_CODE",
    "PROJECT_TYPE_ID_LITERATURE",
    "PROJECT_TYPE_IDS",
    "ProjectTypeId",
    "ProjectProfile",
    "QAProject",
    "initialize_project",
    "open_project",
    "prepare_project_workspace",
    "default_project_profile",
    "configure_project",
]

# Standard identifiers do not require their definitions to be installed.
PROJECT_TYPE_ID_CODE: Final[ProjectTypeId] = "code"
PROJECT_TYPE_ID_LITERATURE: Final[ProjectTypeId] = "literature"

# Loading rejects an empty or invalid registry before projects can open.
PROJECT_TYPE_IDS: Final[tuple[ProjectTypeId, ...]] = tuple(
    load_project_types()
)

_PROJECT_EXTENSIONS: Final[dict[ProjectTypeId, str]] = {
    type_id: ", ".join(item.extensions)
    for type_id, item in load_project_types().items()
}
_DEFAULT_FILE_LIMITS: Final[dict[ProjectTypeId, int]] = {
    type_id: item.max_file_bytes
    for type_id, item in load_project_types().items()
}
# Registered type identifiers as listed in error messages.
_TYPE_ID_CHOICE_TEXT: Final[str] = ", ".join(PROJECT_TYPE_IDS)

# These defaults are used only when generating a missing project profile;
# existing ``profile.ini`` values always take precedence.
_PROFILE_DEFAULTS = load_defaults_section("PROJECT_PROFILE")
_PROFILE_ALLOW_DOWNLOAD: Final[bool] = require_default_bool(
    _PROFILE_DEFAULTS, "ALLOW_DOWNLOAD", "PROJECT_PROFILE"
)
_PROFILE_TARGET_TOKENS: Final[int] = require_default_int(
    _PROFILE_DEFAULTS, "TARGET_TOKENS", "PROJECT_PROFILE"
)
_PROFILE_MAX_TOKENS: Final[int] = require_default_int(
    _PROFILE_DEFAULTS, "MAX_TOKENS", "PROJECT_PROFILE"
)
_PROFILE_OVERLAP_TOKENS: Final[int] = require_default_nonnegative_int(
    _PROFILE_DEFAULTS, "OVERLAP_TOKENS", "PROJECT_PROFILE"
)
_PROFILE_RETRIEVAL_COUNT: Final[int] = require_default_int(
    _PROFILE_DEFAULTS, "RETRIEVAL_COUNT", "PROJECT_PROFILE"
)
_PROFILE_THINKING: Final[bool] = require_default_bool(
    _PROFILE_DEFAULTS, "THINKING", "PROJECT_PROFILE"
)
_PROFILE_TEMPERATURE: Final[float] = require_default_number(
    _PROFILE_DEFAULTS, "TEMPERATURE", "PROJECT_PROFILE", allow_zero=True
)
_PROFILE_MAX_NEW_TOKENS: Final[int] = require_default_int(
    _PROFILE_DEFAULTS, "MAX_NEW_TOKENS", "PROJECT_PROFILE"
)
if _PROFILE_MAX_TOKENS < _PROFILE_TARGET_TOKENS:
    raise ValueError(
        "PROJECT_PROFILE.MAX_TOKENS must reach TARGET_TOKENS."
    )
if _PROFILE_OVERLAP_TOKENS >= _PROFILE_TARGET_TOKENS:
    raise ValueError(
        "PROJECT_PROFILE.OVERLAP_TOKENS must be below TARGET_TOKENS."
    )

# Keep INI formatting and ordering unchanged for generated profiles.
_DEFAULT_PROFILE = """[project]
type_id = {type_id}
source = {source}
extensions = {extensions}
max_file_bytes = {max_file_bytes}

[models]
backend = {backend}
allow_download = {allow_download}
chat_mlx = {chat_mlx}
chat_torch = {chat_torch}
embedding_mlx = {embedding_mlx}
embedding_torch = {embedding_torch}
query_format = {query_format}

[chunking]
target_tokens = {target_tokens}
max_tokens = {max_tokens}
overlap_tokens = {overlap_tokens}

[retrieval]
count = {retrieval_count}

[generation]
thinking = {thinking}
temperature = {temperature}
max_new_tokens = {max_new_tokens}
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
    type_id: ProjectTypeId,
    source: str,
    extensions: str,
) -> str:
    """Format the built-in profile with shared core defaults.

    Parameters
    ----------
    type_id: ProjectTypeId
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
        type_id=type_id,
        source=source,
        extensions=extensions,
        max_file_bytes=_DEFAULT_FILE_LIMITS[type_id],
        allow_download=str(_PROFILE_ALLOW_DOWNLOAD).lower(),
        target_tokens=_PROFILE_TARGET_TOKENS,
        max_tokens=_PROFILE_MAX_TOKENS,
        overlap_tokens=_PROFILE_OVERLAP_TOKENS,
        retrieval_count=_PROFILE_RETRIEVAL_COUNT,
        thinking=str(_PROFILE_THINKING).lower(),
        temperature=_PROFILE_TEMPERATURE,
        max_new_tokens=_PROFILE_MAX_NEW_TOKENS,
        backend=BACKEND_AUTO,
        chat_mlx=DEFAULT_CHAT_MODEL_MLX,
        chat_torch=DEFAULT_CHAT_MODEL_TORCH,
        embedding_mlx=DEFAULT_EMBEDDING_MODEL_MLX,
        embedding_torch=DEFAULT_EMBEDDING_MODEL_TORCH,
        query_format=QUERY_FORMAT_QWEN3,
    )


def default_project_profile(type_id: ProjectTypeId) -> ProjectProfile:
    """Return the default profile for one registered project type.

    Parameters
    ----------
    type_id: ProjectTypeId
        Registered project-type identifier from ``PROJECT_TYPE_IDS``.

    Returns
    -------
    ProjectProfile
        The settings a new project of the selected type would receive.

    Raises
    ------
    ValueError
        If ``type_id`` is not one of ``PROJECT_TYPE_IDS``.

    Notes
    -----
    This parses the built-in profile template without touching a project
    directory or loading models.

    """

    if type_id not in PROJECT_TYPE_IDS:
        raise ValueError(f"type_id must be one of: {_TYPE_ID_CHOICE_TEXT}.")
    extensions = _PROJECT_EXTENSIONS[type_id]
    parser = ConfigParser()
    parser.read_string(_default_profile_text(type_id, "sources", extensions))
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
    type_id: ProjectTypeId
        Registered project-type identifier.
    source: str
        Source folder, relative to the project or absolute.
    extensions: tuple[str, ...]
        File extensions included during scanning.
    max_file_bytes: int
        Largest source file read by KíLín.
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
    type_id: ProjectTypeId
        Registered project-type identifier.
    source: str
        Source folder, relative to the project or absolute.
    extensions: tuple[str, ...]
        File extensions included during scanning.
    max_file_bytes: int
        Largest source file read by KíLín.
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

    type_id: ProjectTypeId
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
    """Resolved paths and settings for one KíLín project.

    Parameters
    ----------
    root: pathlib.Path
        Project folder containing profile.ini.
    source: pathlib.Path
        Folder containing source material for the selected project type.
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
        Folder containing source material for the selected project type.
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
    type_id: ProjectTypeId,
) -> QAProject:
    """Create a project profile and its required tool resources.

    Parameters
    ----------
    directory: str | pathlib.Path
        Existing project, repository, or document folder.
    type_id: ProjectTypeId
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
        If type_id is invalid or conflicts with an existing profile.

    Notes
    -----
    Requires an existing directory. A missing profile.ini is initialized using
    the selected type; an existing profile is preserved and checked for a
    matching type ID. A new profile prefers an existing ``sources/``, then
    the configured ``source_directory``, then the project root.
    New projects receive only the template files required
    by their type; existing user templates are preserved. Opening creates
    indexes, logs, outputs and templates directories. No model is loaded.

    """

    if type_id not in PROJECT_TYPE_IDS:
        raise ValueError(f"type_id must be one of: {_TYPE_ID_CHOICE_TEXT}.")
    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Choose a project folder: {root}")

    source_name = project_type(type_id).source_directory
    if (root / "sources").is_dir():
        source = "sources"
    else:
        source = source_name if (root / source_name).is_dir() else "."
    return _initialize_profile(root, type_id, source)


class _InitializationJournal:
    """Track only paths absent before a new project's initialization.

    Parameters
    ----------
    root: pathlib.Path
        Destination workspace. Existing user-owned files are not recorded.

    Notes
    -----
    Rollback removes newly written files and then attempts to remove empty
    directories. It does not delete existing files or recursively remove
    arbitrary user content, and cannot defend against concurrent writers.

    """

    def __init__(self, root: Path) -> None:
        """Initialize an empty collection of paths created by this attempt.

        Parameters
        ----------
        root: pathlib.Path
            Root that bounds tracked paths.

        """

        self.root = root
        self.created: set[Path] = set()

    def note(self, path: Path) -> None:
        """Record a prospective path and any absent workspace parents.

        Parameters
        ----------
        path: pathlib.Path
            File or directory that an initialization step may create.

        """

        current = path
        while current == self.root or current.is_relative_to(self.root):
            if current.exists() or current.is_symlink():
                break
            self.created.add(current)
            if current == self.root:
                break
            current = current.parent

    def rollback(self) -> None:
        """Remove created files and empty folders, preserving other data.

        Notes
        -----
        Failed cleanup is best-effort. User-owned or concurrently created
        entries in previously empty directories prevent their removal.

        """

        for path in sorted(
            self.created, key=lambda item: len(item.parts), reverse=True
        ):
            try:
                if path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            except OSError:
                continue


def _prepared_profile(
    type_id: ProjectTypeId,
    source: str,
    settings: dict[str, object] | None,
) -> tuple[str, ProjectProfile]:
    """Validate and render the complete profile before filesystem writes.

    Parameters
    ----------
    type_id: ProjectTypeId
        Validated project-type identifier.
    source: str
        Initial source folder recorded in the profile.
    settings: dict[str, object] | None
        Optional overrides accepted by ``configure_project``.

    Returns
    -------
    tuple[str, ProjectProfile]
        Final profile text and parsed settings.

    """

    text = _default_profile_text(type_id, source, _PROJECT_EXTENSIONS[type_id])
    parser = ConfigParser()
    parser.read_string(text)
    if settings:
        for key, value in settings.items():
            section = _EDITABLE_SETTINGS.get(key)
            if section is None:
                raise ValueError(f"Unknown project setting: {key}")
            option = "count" if key == "retrieval_count" else key
            if key == "extensions" and isinstance(value, (list, tuple)):
                value = ", ".join(value)
            parser.set(section, option, str(value))
        stream = StringIO()
        parser.write(stream)
        text = stream.getvalue()
    return text, _read_profile(parser)


@contextmanager
def _preparing_project(
    root: Path,
    type_id: ProjectTypeId,
    source: str,
    *,
    settings: dict[str, object] | None = None,
    trace: TraceSink | None = None,
    validate_source: bool = False,
) -> Iterator[tuple[QAProject, _InitializationJournal | None]]:
    """Initialize or reopen a project, committing a new profile last.

    Parameters
    ----------
    root: pathlib.Path
        Destination project directory.
    type_id: ProjectTypeId
        Expected project type.
    source: str
        Relative source location for a new project.
    settings: dict[str, object] | None, optional
        Profile overrides applied before first copying. Defaults to None.
    trace: TraceSink | None, optional
        Optional diagnostic sink for existing-profile configuration changes.
    validate_source: bool, optional
        Require an existing profile to use the specified managed source
        folder. Defaults to False for general project opening.

    Yields
    ------
    tuple[QAProject, _InitializationJournal | None]
        A project view and optional journal for source-copy operations.

    Notes
    -----
    For new projects, ``profile.ini`` is written atomically only when the
    caller finishes all workspace and resource operations successfully.
    Existing projects are opened without reinstalling user-deleted files.

    """

    profile_path = root / "profile.ini"
    if profile_path.exists() or profile_path.is_symlink():
        project = open_project(root)
        if project.profile.type_id != type_id:
            raise ValueError(
                "PROJECT_TYPE_ID conflicts with the existing profile.ini."
            )
        expected = Path(source).expanduser()
        if not expected.is_absolute():
            expected = root / expected
        if (
            validate_source
            and project.source != expected.resolve(strict=True)
        ):
            raise ValueError("project.source conflicts with profile.ini.")
        if settings:
            project = configure_project(root, settings, trace=trace)
        yield project, None
        return

    text, profile = _prepared_profile(type_id, source, settings)
    layout = plan_project_layout(type_id)
    journal = _InitializationJournal(root)
    journal.note(root)
    for name in ("indexes", "logs", "outputs", "templates"):
        journal.note(root / name)
    for resource in layout.resources:
        journal.note(root / resource.destination)
    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = root / source_path
    if source_path != root:
        if source_path.is_relative_to(root):
            journal.note(source_path)
    try:
        root.mkdir(parents=True, exist_ok=True)
        if source_path.is_relative_to(root) and source_path != root:
            if source_path.is_symlink():
                raise ValueError("Managed source directory cannot be a link.")
            source_path.mkdir(parents=True, exist_ok=True)
        ensure_project_folders(root)
        install_project_resources(root, layout)
        provisional = QAProject(
            root=root,
            source=source_path.resolve(strict=True),
            indexes=root / "indexes",
            logs=root / "logs",
            profile=profile,
        )
        yield provisional, journal
        journal.note(profile_path)
        atomic_write_text(profile_path, text)
    except BaseException:
        journal.rollback()
        raise


def _initialize_profile(
    root: Path, type_id: ProjectTypeId, source: str
) -> QAProject:
    """Create a project profile after successful resource initialization.

    Parameters
    ----------
    root: pathlib.Path
        Workspace root.
    type_id: ProjectTypeId
        Expected project type.
    source: str
        Source folder within a new profile.

    Returns
    -------
    QAProject
        Opened and validated project.

    """

    with _preparing_project(root, type_id, source):
        pass
    return open_project(root)


def prepare_project_workspace(
    source_directory: str | Path,
    type_id: ProjectTypeId,
    workspace_root: str | Path,
) -> QAProject:
    """Mirror one source folder into a managed KíLín workspace.

    The original source is never modified. A normal workspace is named
    ``project_<source-name>`` and contains the copied source folder beside
    ``profile.ini``, the shared workspace folders, and
    ``source_snapshot.json``.
    Only files selected by the project's repository rules are mirrored.

    Reopening the same source verifies the managed copy against its saved
    SHA-256 manifest. If the copy differs, it is rebuilt from the original;
    otherwise only added, modified, and removed source files are synchronized.
    If another source folder with the same basename already owns the preferred
    workspace name, a short path hash is appended to avoid a collision.

    Parameters
    ----------
    source_directory: str | pathlib.Path
        Existing folder containing project source material.
    type_id: ProjectTypeId
        Type of source material in the project.
    workspace_root: str | pathlib.Path
        KíLín sandbox folder that contains managed project workspaces.

    Returns
    -------
    QAProject
        Project whose source points to the synchronized sandbox snapshot.

    Raises
    ------
    ValueError
        If type_id is unsupported, source/workspace locations conflict, an
        existing managed profile is inconsistent, or a unique workspace
        location cannot be selected.
    NotADirectoryError
        If the original source is not a directory.
    OSError
        If scanning, hashing, copying, deleting, or writing metadata fails.

    Notes
    -----
    This mirrors one original repository using its selected-source manifest. It
    is not the cumulative multi-source API in kilin.workspace. The current
    implementation requires complete source and snapshot scans and verifies
    copied content on each reopen. This adds disk I/O proportional to selected
    source size. Reopening and synchronization are not atomic across snapshot
    and manifest; initial project creation commits its profile last.
    Source copying is organization, not execution sandboxing.

    """

    if type_id not in PROJECT_TYPE_IDS:
        raise ValueError(f"type_id must be one of: {_TYPE_ID_CHOICE_TEXT}.")

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
    # A new profile is committed only after the complete source snapshot and
    # its provenance manifest have been prepared successfully.
    source_name = source.name
    snapshot = root / source_name
    if snapshot.is_symlink():
        raise ValueError("Managed source copy must not be a symbolic link.")
    if not (root / "profile.ini").exists() and snapshot.is_dir():
        if any(snapshot.iterdir()):
            raise ValueError(
                "Uninitialized workspace has an existing source snapshot."
            )
    with _preparing_project(
        root, type_id, source_name, validate_source=True
    ) as (
        project, journal
    ):
        if project.source != snapshot.resolve(strict=True):
            raise ValueError(
                "Managed workspace profile.ini must keep project.source "
                f"set to {source_name}."
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
                if journal is not None:
                    journal.note(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target)

            _remove_empty_directories(snapshot)

        # Always record ownership, including the zero-file case.  This lets the
        # next open distinguish different sources with the same basename.
        manifest_payload = {
            **current_manifest,
            "source_directory": str(source),
        }
        if journal is not None:
            journal.note(snapshot_manifest_path)
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
            "Could not choose a unique KíLín workspace for this source."
        )
    return collision_root


def open_project(directory: str | Path) -> QAProject:
    """Open a configured project and prepare its shared folders.

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
    function creates indexes, logs, outputs and templates directories before
    returning, without reinstalling packaged templates; it does not load
    models.

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

    ensure_project_folders(root)
    indexes = root / "indexes"
    logs = root / "logs"
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

    type_id_value = _required(parser, "project", "type_id")
    if type_id_value not in PROJECT_TYPE_IDS:
        raise ValueError(
            f"project.type_id must be one of: {_TYPE_ID_CHOICE_TEXT}."
        )
    type_id = cast("ProjectTypeId", type_id_value)

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
        type_id=type_id,
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
