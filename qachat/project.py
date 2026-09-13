"""Open one configurable QAChat project folder.

Classes
-------
ProjectProfile
    Models, source selection, chunking, retrieval, and generation settings.
QAProject
    Resolved source, index, and log paths for one project.

Functions
---------
open_project
    Read profile.ini and prepare the project output folders.

"""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from qachat.runtime import Backend, SelectedBackend


__all__ = [
    "ProjectProfile",
    "QAProject",
    "open_project",
]

ProjectKind: TypeAlias = Literal["code", "literature"]
QueryFormat: TypeAlias = Literal["plain", "qwen3"]


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
