"""Open one configurable QAChat project folder.

Classes
-------
ProjectProfile
    Models, source selection, chunking, retrieval, and generation settings.
QAProject
    Resolved source, index, and log paths for one project.

Functions
---------
initialize_project
    Create a missing profile.ini, then open the project.
open_project
    Read profile.ini and prepare the project output folders.
default_project_profile
    Return validated settings for a new project.
configure_project
    Validate and save editable profile settings.

"""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from typing import Literal, TypeAlias, cast

from qachat.repository import (
    build_repository_manifest,
    compare_repository_manifests,
    scan_repository,
)
from qachat.runtime import Backend, SelectedBackend
from qachat.trace_logging import NullTraceLogger, TraceSink

__all__ = [
    "ProjectProfile",
    "QAProject",
    "initialize_project",
    "open_project",
    "prepare_project_workspace",
    "default_project_profile",
    "configure_project",
]

ProjectKind: TypeAlias = Literal["code", "literature"]
QueryFormat: TypeAlias = Literal["plain", "qwen3"]
# Literature PDFs need more space than typical code files.
_DEFAULT_FILE_LIMITS = {"code": 1_048_576, "literature": 20 * 1024 * 1024}
_DEFAULT_PROFILE = """[project]
kind = {kind}
source = {source}
extensions = {extensions}
max_file_bytes = {max_file_bytes}

[models]
backend = auto
allow_download = true
chat_mlx = Qwen/Qwen3-0.6B-MLX-4bit
chat_torch = Qwen/Qwen3-0.6B
embedding_mlx = mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
embedding_torch = Qwen/Qwen3-Embedding-0.6B
query_format = qwen3

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

def default_project_profile(kind: ProjectKind) -> ProjectProfile:
    """Return the settings a new project of this kind would receive."""

    if kind not in {"code", "literature"}:
        raise ValueError("kind must be code or literature.")
    extensions = (
        ".py, .ipynb, .m, .md" if kind == "code" else ".pdf, .md, .txt"
    )
    parser = ConfigParser()
    parser.read_string(
        _DEFAULT_PROFILE.format(
            kind=kind,
            source="sources",
            extensions=extensions,
            max_file_bytes=_DEFAULT_FILE_LIMITS[kind],
        )
    )
    return _read_profile(parser)


def configure_project(
    directory: str | Path,
    settings: dict[str, object],
    *,
    trace: TraceSink | None = None,
) -> QAProject:
    """Validate and save settings without changing source ownership."""

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
    temporary = path.with_suffix(".ini.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        parser.write(stream)
    temporary.replace(path)
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

    Attributes
    ----------
    kind: "code" | "literature"
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
    chat_mlx, chat_torch: str
        Chat model IDs for the two backends.
    embedding_mlx, embedding_torch: str
        Embedding model IDs for the two backends.
    query_format: "plain" | "qwen3"
        Query format expected by the embedding model.
    target_tokens, max_tokens, overlap_tokens: int
        Chunk size settings.
    retrieval_count: int
        Number of results requested from each retrieval stage.
    thinking: bool
        Whether compatible chat templates enable thinking.
    temperature: float
        Chat generation temperature.
    max_new_tokens: int | None
        Optional maximum generated tokens per answer.

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
        """Return the chat model configured for one selected backend."""

        return self.chat_mlx if backend == "mlx" else self.chat_torch

    def embedding_model(self, backend: SelectedBackend) -> str:
        """Return the embedding model configured for one selected backend."""

        return self.embedding_mlx if backend == "mlx" else self.embedding_torch


@dataclass(frozen=True, slots=True)
class QAProject:
    """Resolved paths and settings for one QAChat project.

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
    kind: "code" | "literature"
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

    """

    if kind not in {"code", "literature"}:
        raise ValueError("kind must be code or literature.")
    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Choose a project folder: {root}")

    profile_path = root / "profile.ini"
    if not profile_path.exists():
        source_name = "repository" if kind == "code" else "documents"
        if (root / "sources").is_dir():
            source = "sources"
        else:
            source = source_name if (root / source_name).is_dir() else "."

        extensions = (
            ".py, .ipynb, .m, .md"
            if kind == "code"
            else ".pdf, .md, .txt"
        )
        profile_path.write_text(
            _DEFAULT_PROFILE.format(
                kind=kind,
                source=source,
                extensions=extensions,
                max_file_bytes=_DEFAULT_FILE_LIMITS[kind],
            ),
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

    Reopening the same source compares a saved SHA-256 manifest and copies
    only files that were added or modified while deleting files that were
    removed.  If another source folder with the same basename already owns
    the preferred workspace name, a short path hash is appended to avoid a
    collision.

    Parameters
    ----------
    source_directory: str | pathlib.Path
        Existing source code or literature folder.
    kind: "code" | "literature"
        Type of source material in the project.
    workspace_root: str | pathlib.Path
        QAChat sandbox folder that contains managed project workspaces.

    Returns
    -------
    QAProject
        Project whose source points to the synchronized sandbox snapshot.

    """

    if kind not in {"code", "literature"}:
        raise ValueError("kind must be code or literature.")

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
    root.mkdir(parents=True, exist_ok=True)

    # Keep the source snapshot as a sibling of indexes/ and logs/.  Preserve
    # the original folder name so the sandbox structure is easy to inspect.
    source_name = source.name
    snapshot = root / source_name
    snapshot.mkdir(parents=True, exist_ok=True)

    profile_path = root / "profile.ini"
    if not profile_path.exists():

        extensions = (
            ".py, .ipynb, .m, .md"
            if kind == "code"
            else ".pdf, .md, .txt"
        )
        profile_path.write_text(
            _DEFAULT_PROFILE.format(
                kind=kind,
                source=source_name,
                extensions=extensions,
                max_file_bytes=_DEFAULT_FILE_LIMITS[kind],
            ),
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


def _select_workspace_root(preferred_root: Path, source: Path) -> Path:
    """Choose a readable workspace name without a source collision."""

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
    """Return one JSON object, or ``None`` when it is missing or invalid."""

    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, value: dict[str, object]) -> None:
    """Atomically write one readable JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _remove_empty_directories(root: Path) -> None:
    """Remove empty snapshot directories left after deleted source files."""

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
    """Read and validate every supported project setting."""

    kind_value = _required(parser, "project", "kind")
    if kind_value not in {"code", "literature"}:
        raise ValueError("project.kind must be code or literature.")
    kind = cast("ProjectKind", kind_value)

    extensions = tuple(
        part.strip().lower()
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

    backend_value = _required(parser, "models", "backend")
    if backend_value not in {"auto", "mlx", "torch"}:
        raise ValueError("models.backend must be auto, mlx, or torch.")
    backend = cast("Backend", backend_value)

    query_value = _required(parser, "models", "query_format")
    if query_value not in {"plain", "qwen3"}:
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
    """Return one required, non-empty profile value."""

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
    """Return one positive whole-number profile value."""

    value = _integer(parser, section, option)
    if value < 1:
        raise ValueError(f"{section}.{option} must be positive.")
    return value


def _non_negative_integer(
    parser: ConfigParser,
    section: str,
    option: str,
) -> int:
    """Return one non-negative whole-number profile value."""

    value = _integer(parser, section, option)
    if value < 0:
        raise ValueError(f"{section}.{option} cannot be negative.")
    return value


def _integer(parser: ConfigParser, section: str, option: str) -> int:
    """Return one whole-number profile value."""

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
    """Return a positive integer or None from one profile value."""

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
    """Return one Boolean profile value."""

    _required(parser, section, option)
    try:
        return parser.getboolean(section, option)
    except ValueError as error:
        raise ValueError(
            f"{section}.{option} must be true or false."
        ) from error


def _float(parser: ConfigParser, section: str, option: str) -> float:
    """Return one numeric profile value."""

    value = _required(parser, section, option)
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{section}.{option} must be a number.") from error
