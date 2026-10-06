"""Persist editable effort levels alongside a GUI workspace.

Each backend has five effort levels. A level has a display name, the chat
model it answers with, and how many references it retrieves. The composer
picks a model level and a reference level independently. The file is
written only when a project differs from the defaults.
"""

import json
from pathlib import Path

from huggingface_hub.utils import validate_repo_id

from qachat.gui.model_catalog import (
    DEEP_CHAT_MODEL_MLX,
    LIGHT_CHAT_MODEL_MLX,
)
from qachat.model_defaults import (
    DEFAULT_CHAT_MODEL_MLX,
    DEFAULT_CHAT_MODEL_TORCH,
)
from qachat.runtime import BACKEND_MLX

EFFORT_FILE = "effort_levels.json"
# Earlier GUI versions stored per-preset model replacements here.
LEGACY_FILE = "model_choices.json"
LEVEL_COUNT = 5
# The level whose model is also the project's default chat model.
DEFAULT_LEVEL = 2
MAX_LEVEL_NAME = 24
MAX_RETRIEVAL = 200

_NAMES = ("Light", "Instant", "Lagom", "Thorough", "Deep")
_RETRIEVAL = (5, 20, 20, 50, 50)


def validate_model_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Enter a Hugging Face model ID.")
    value = value.strip()
    if not Path(value).expanduser().is_dir():
        validate_repo_id(value)
    return value


def default_effort_levels(backend: str) -> list[dict[str, object]]:
    """Return the built-in levels for a backend."""

    if backend == BACKEND_MLX:
        models = (
            LIGHT_CHAT_MODEL_MLX,
            LIGHT_CHAT_MODEL_MLX,
            DEFAULT_CHAT_MODEL_MLX,
            DEFAULT_CHAT_MODEL_MLX,
            DEEP_CHAT_MODEL_MLX,
        )
    else:
        # Other backends cannot switch chat models per question.
        models = (DEFAULT_CHAT_MODEL_TORCH,) * LEVEL_COUNT
    return [
        {"name": name, "model": model, "retrieval": count}
        for name, model, count in zip(_NAMES, models, _RETRIEVAL)
    ]


def validate_effort_levels(value: object) -> list[dict[str, object]]:
    """Check five levels of name, model ID, and reference count."""

    if not isinstance(value, list) or len(value) != LEVEL_COUNT:
        raise ValueError(f"Expected {LEVEL_COUNT} effort levels.")
    levels = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Invalid effort level.")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Give every effort level a name.")
        name = " ".join(name.split())
        if len(name) > MAX_LEVEL_NAME:
            raise ValueError(
                f"Effort level names can have at most {MAX_LEVEL_NAME} "
                "characters."
            )
        retrieval = item.get("retrieval")
        if (
            type(retrieval) is not int
            or not 1 <= retrieval <= MAX_RETRIEVAL
        ):
            raise ValueError(
                f"Reference counts must be whole numbers from 1 to "
                f"{MAX_RETRIEVAL}."
            )
        levels.append(
            {
                "name": name,
                "model": validate_model_id(item.get("model")),
                "retrieval": retrieval,
            }
        )
    return levels


def _read_json(path: Path) -> object:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_effort_levels(
    root: Path | None,
    backend: str,
    chat_model: str | None = None,
) -> list[dict[str, object]]:
    """Return a project's levels, falling back to defaults.

    Parameters
    ----------
    root: pathlib.Path | None
        Project workspace, or None for a project not created yet.
    backend: str
        Computation backend whose levels are wanted.
    chat_model: str | None, optional
        The profile's chat model, used only to migrate projects saved before
        effort levels existed. Defaults to ``None``.

    """

    levels = default_effort_levels(backend)
    if root is None:
        return levels
    saved = _read_json(root / EFFORT_FILE)
    if isinstance(saved, dict) and backend in saved:
        return validate_effort_levels(saved[backend])

    # Migrate the earlier per-preset replacements, ignoring retired presets.
    legacy = _read_json(root / LEGACY_FILE)
    replacements = legacy if isinstance(legacy, dict) else {}
    for level in levels:
        replacement = replacements.get(level["model"])
        if isinstance(replacement, str) and replacement.strip():
            level["model"] = validate_model_id(replacement)
    if chat_model:
        # The middle levels used to follow the profile's chat model; other
        # backends use that one model throughout.
        targets = (
            (DEFAULT_LEVEL, DEFAULT_LEVEL + 1)
            if backend == BACKEND_MLX
            else range(LEVEL_COUNT)
        )
        for index in targets:
            levels[index]["model"] = chat_model
    return levels


def write_effort_levels(
    root: Path, backend: str, levels: list[dict[str, object]]
) -> None:
    """Save a backend's levels, or drop them when they match defaults."""

    levels = validate_effort_levels(levels)
    path = root / EFFORT_FILE
    saved = _read_json(path)
    saved = dict(saved) if isinstance(saved, dict) else {}
    if levels == default_effort_levels(backend):
        saved.pop(backend, None)
    else:
        saved[backend] = levels
    if saved:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(saved, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    else:
        path.unlink(missing_ok=True)
    # Migrated into effort levels; the old file is no longer read.
    (root / LEGACY_FILE).unlink(missing_ok=True)
