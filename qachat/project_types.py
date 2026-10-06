"""Load project types and their instructions from ``project_types.json``.

A project type (for example a code repository or a literature collection)
decides which files a new project reads by default and which instructions
the models receive. Editing the JSON file changes those instructions without
code changes; restart QAChat afterwards. Saved conversations keep the system
prompt they were answered with.

Key Features
------------

- **One file per installation**

    ``qachat/project_types.json`` lists every type. The built-in ``code`` and
    ``literature`` types must stay; more can be added.

- **Validated on load**

    A malformed file raises ``ValueError`` naming the type and field.

Module Structure
----------------

- ``ProjectType``

    One validated type.

- ``load_project_types``

    Read and validate the file, cached for the process.

- ``project_type``

    Look up one type by its ID.

Examples
--------
>>> from qachat.project_types import project_type
>>> project_type("code").label
'Code repository'

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re

__all__ = [
    "PROJECT_TYPES_FILE",
    "ProjectType",
    "load_project_types",
    "project_type",
]

PROJECT_TYPES_FILE = Path(__file__).with_name("project_types.json")
_BUILT_IN = ("code", "literature")
_KIND_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,31}")


@dataclass(frozen=True, slots=True)
class ProjectType:
    """One project type from ``project_types.json``.

    Attributes
    ----------
    kind: str
        Identifier written to ``profile.ini``.
    label: str
        Name shown when choosing a type.
    extensions: tuple[str, ...]
        File suffixes a new project reads by default.
    max_file_bytes: int
        Default size limit for one source file.
    retrieval_instruction: str
        Task description added to Qwen3-format embedding queries.
    system_prompt: str
        Instructions given to the chat model for every answer.
    answer_guidance: str
        Extra instruction the desktop app adds to each question; empty for
        none. Notebooks do not add it.
    welcome_prompts: tuple[str, ...]
        Headings the desktop app shows in an empty chat.
    suggestions: tuple[str, ...]
        Example requests the desktop app shows in an empty chat.

    """

    kind: str
    label: str
    extensions: tuple[str, ...]
    max_file_bytes: int
    retrieval_instruction: str
    system_prompt: str
    answer_guidance: str = ""
    welcome_prompts: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()


def _text(kind: str, data: dict, field: str, required: bool = True) -> str:
    value = data.get(field, None if required else "")
    if not isinstance(value, str) or (required and not value.strip()):
        raise ValueError(f"Project type {kind!r} needs text for {field!r}.")
    return value.strip()


def _texts(kind: str, data: dict, field: str) -> tuple[str, ...]:
    value = data.get(field, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(
            f"Project type {kind!r} needs a list of text for {field!r}."
        )
    return tuple(item.strip() for item in value)


def _parse(kind: object, data: object) -> ProjectType:
    if not isinstance(kind, str) or not _KIND_PATTERN.fullmatch(kind):
        raise ValueError(
            f"Project type ID {kind!r} must be lowercase letters, digits, "
            "hyphens, or underscores, starting with a letter."
        )
    if not isinstance(data, dict):
        raise ValueError(f"Project type {kind!r} must be an object.")
    extensions = _texts(kind, data, "extensions")
    if not extensions or not all(
        extension.startswith(".") and len(extension) > 1
        for extension in extensions
    ):
        raise ValueError(
            f"Project type {kind!r} needs file suffixes such as '.pdf'."
        )
    max_file_bytes = data.get("max_file_bytes")
    if type(max_file_bytes) is not int or max_file_bytes < 1:
        raise ValueError(
            f"Project type {kind!r} needs a positive whole number for "
            "'max_file_bytes'."
        )
    return ProjectType(
        kind=kind,
        label=_text(kind, data, "label"),
        extensions=tuple(extension.lower() for extension in extensions),
        max_file_bytes=max_file_bytes,
        retrieval_instruction=_text(kind, data, "retrieval_instruction"),
        system_prompt=_text(kind, data, "system_prompt"),
        answer_guidance=_text(kind, data, "answer_guidance", required=False),
        welcome_prompts=_texts(kind, data, "welcome_prompts"),
        suggestions=_texts(kind, data, "suggestions"),
    )


@lru_cache(maxsize=None)
def load_project_types(
    path: Path = PROJECT_TYPES_FILE,
) -> dict[str, ProjectType]:
    """Read and validate every project type, in file order.

    Parameters
    ----------
    path: pathlib.Path, optional
        JSON file to read. Defaults to the packaged ``project_types.json``.

    Returns
    -------
    dict[str, ProjectType]
        Types keyed by ID. The result is cached for the process.

    Raises
    ------
    ValueError
        If the file is not valid JSON, a type is malformed, or a built-in
        type is missing.

    """

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object of project types.")
    types = {kind: _parse(kind, data) for kind, data in value.items()}
    missing = [kind for kind in _BUILT_IN if kind not in types]
    if missing:
        raise ValueError(
            f"{path} must keep the built-in types: {', '.join(missing)}."
        )
    return types


def project_type(kind: str) -> ProjectType:
    """Return one project type by ID.

    Raises
    ------
    ValueError
        If no type has that ID.

    """

    types = load_project_types()
    if kind not in types:
        raise ValueError(
            f"Unknown project type {kind!r}. Choose one of: "
            f"{', '.join(types)}."
        )
    return types[kind]
