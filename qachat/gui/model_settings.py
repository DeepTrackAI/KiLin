"""Persist editable effort levels and search depth with a GUI workspace.

Each backend has five effort levels; a level has a display name and the
chat model it answers with. Search depth is a separate vector of five
steps, each the number of passages retrieved for an answer. The composer
picks an effort level and a search-depth step independently. The file is
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
# The same file keeps a customized intent router model per backend.
ROUTER_KEY = "router"
# Earlier GUI versions stored per-preset model replacements here.
LEGACY_FILE = "model_choices.json"
LEVEL_COUNT = 5
# The level whose model is also the project's default chat model.
DEFAULT_LEVEL = 2
MAX_LEVEL_NAME = 24
# Search depth: passages retrieved at each of five steps, project-wide.
SEARCH_DEPTH_KEY = "search_depth"
DEFAULT_SEARCH_DEPTH = (5, 5, 10, 10, 15)
MAX_SEARCH_DEPTH = 200

_NAMES = ("Light", "Instant", "Lagom", "Thorough", "Deep")
# Per-level reference counts of earlier versions, before search depth.
_LEGACY_RETRIEVAL = (5, 20, 20, 50, 50)


def validate_model_id(value: str) -> str:
    """Check a Hugging Face model ID or an existing local model folder.

    Parameters
    ----------
    value: str
        Model ID such as ``Qwen/Qwen3-8B-MLX-4bit`` or a local path.

    Returns
    -------
    str
        The value without surrounding space.

    Raises
    ------
    ValueError
        If the value is empty or not a valid repository ID.

    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Enter a Hugging Face model ID.")
    value = value.strip()
    if not Path(value).expanduser().is_dir():
        validate_repo_id(value)
    return value


def default_effort_levels(backend: str) -> list[dict[str, object]]:
    """Return the built-in levels for a backend.

    Parameters
    ----------
    backend: str
        Computation backend, ``mlx`` or ``torch``.

    Returns
    -------
    list[dict[str, object]]
        Five levels of ``name`` and ``model``.

    """

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
        {"name": name, "model": model} for name, model in zip(_NAMES, models)
    ]


def validate_effort_levels(value: object) -> list[dict[str, object]]:
    """Check five levels of name and model ID.

    Parameters
    ----------
    value: object
        Levels as received from the page or read from the file.

    Returns
    -------
    list[dict[str, object]]
        Levels with names collapsed to single spaces. A ``retrieval``
        count saved by earlier versions is ignored here; see
        ``read_search_depth``.

    Raises
    ------
    ValueError
        If there are not five levels or a field is invalid.

    """

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
        levels.append(
            {"name": name, "model": validate_model_id(item.get("model"))}
        )
    return levels


def validate_search_depth(value: object) -> list[int]:
    """Check five search-depth steps.

    Parameters
    ----------
    value: object
        Steps as received from the page or read from the file.

    Returns
    -------
    list[int]
        Five whole numbers of passages, each from 1 to
        ``MAX_SEARCH_DEPTH``.

    Raises
    ------
    ValueError
        If there are not five steps or a step is out of range.

    """

    if not isinstance(value, list) or len(value) != LEVEL_COUNT:
        raise ValueError(f"Expected {LEVEL_COUNT} search-depth steps.")
    if not all(
        type(step) is int and 1 <= step <= MAX_SEARCH_DEPTH for step in value
    ):
        raise ValueError(
            "Search depth steps must be whole numbers from 1 to "
            f"{MAX_SEARCH_DEPTH}."
        )
    return list(value)


def read_search_depth(root: Path | None) -> list[int]:
    """Return a project's search-depth steps, or the defaults.

    Parameters
    ----------
    root: pathlib.Path | None
        Project workspace, or None for a project not created yet.

    Returns
    -------
    list[int]
        Five steps. Projects saved with per-level reference counts keep
        their customized counts as steps; the old defaults become the new
        defaults.

    """

    saved = _read_json(root / EFFORT_FILE) if root is not None else None
    saved = saved if isinstance(saved, dict) else {}
    if SEARCH_DEPTH_KEY in saved:
        return validate_search_depth(saved[SEARCH_DEPTH_KEY])
    return _legacy_depth(saved) or list(DEFAULT_SEARCH_DEPTH)


def write_search_depth(root: Path, steps: list[int]) -> None:
    """Save search-depth steps, or drop them when they are the defaults.

    Parameters
    ----------
    root: pathlib.Path
        Project workspace.
    steps: list[int]
        Five steps to validate and save.

    """

    steps = validate_search_depth(steps)
    path = root / EFFORT_FILE
    saved = _read_json(path)
    saved = dict(saved) if isinstance(saved, dict) else {}
    if steps == list(DEFAULT_SEARCH_DEPTH):
        saved.pop(SEARCH_DEPTH_KEY, None)
    else:
        saved[SEARCH_DEPTH_KEY] = steps
    _write_json(path, saved)


def _legacy_depth(saved: dict) -> list[int] | None:
    """Customized per-level counts saved by earlier versions, if any."""

    for key, levels in saved.items():
        if key in (ROUTER_KEY, SEARCH_DEPTH_KEY) or not isinstance(
            levels, list
        ):
            continue
        counts = [
            level.get("retrieval") for level in levels
            if isinstance(level, dict)
        ]
        if (
            len(counts) == LEVEL_COUNT
            and all(type(c) is int and 1 <= c <= MAX_SEARCH_DEPTH
                    for c in counts)
            and tuple(counts) != _LEGACY_RETRIEVAL
        ):
            return counts
    return None


def default_router_model(backend: str) -> str:
    """Return the built-in intent router model for a backend.

    Parameters
    ----------
    backend: str
        Computation backend, ``mlx`` or ``torch``.

    Returns
    -------
    str
        The 8B default chat model of that backend.

    """

    if backend == BACKEND_MLX:
        return DEFAULT_CHAT_MODEL_MLX
    return DEFAULT_CHAT_MODEL_TORCH


def read_router_model(root: Path | None, backend: str) -> str:
    """Return a project's intent router model, or the default.

    Parameters
    ----------
    root: pathlib.Path | None
        Project workspace, or None for a project not created yet.
    backend: str
        Computation backend whose router model is wanted.

    Returns
    -------
    str
        The saved model ID, or ``default_router_model(backend)``.

    """

    saved = _read_json(root / EFFORT_FILE) if root is not None else None
    routers = saved.get(ROUTER_KEY) if isinstance(saved, dict) else None
    if isinstance(routers, dict) and isinstance(routers.get(backend), str):
        return validate_model_id(routers[backend])
    return default_router_model(backend)


def write_router_model(root: Path, backend: str, model_id: str) -> None:
    """Save a backend's router model, or drop it when it is the default.

    Parameters
    ----------
    root: pathlib.Path
        Project workspace.
    backend: str
        Computation backend the router model belongs to.
    model_id: str
        Model ID to validate and save.

    """

    model_id = validate_model_id(model_id)
    path = root / EFFORT_FILE
    saved = _read_json(path)
    saved = dict(saved) if isinstance(saved, dict) else {}
    routers = dict(saved.get(ROUTER_KEY) or {})
    if model_id == default_router_model(backend):
        routers.pop(backend, None)
    else:
        routers[backend] = model_id
    if routers:
        saved[ROUTER_KEY] = routers
    else:
        saved.pop(ROUTER_KEY, None)
    _write_json(path, saved)


def _write_json(path: Path, value: dict) -> None:
    if value:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    else:
        path.unlink(missing_ok=True)


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
    """Save a backend's levels, or drop them when they match defaults.

    Parameters
    ----------
    root: pathlib.Path
        Project workspace.
    backend: str
        Computation backend the levels belong to.
    levels: list[dict[str, object]]
        Five levels to validate and save.

    Notes
    -----
    The file is removed when no backend differs from its defaults, and the
    earlier ``model_choices.json`` is removed once migrated.

    """

    levels = validate_effort_levels(levels)
    path = root / EFFORT_FILE
    saved = _read_json(path)
    saved = dict(saved) if isinstance(saved, dict) else {}
    legacy = _legacy_depth(saved)
    if legacy and SEARCH_DEPTH_KEY not in saved:
        # Keep customized counts of earlier versions as search depth.
        saved[SEARCH_DEPTH_KEY] = legacy
    if levels == default_effort_levels(backend):
        saved.pop(backend, None)
    else:
        saved[backend] = levels
    _write_json(path, saved)
    # Migrated into effort levels; the old file is no longer read.
    (root / LEGACY_FILE).unlink(missing_ok=True)
