"""Persist editable effort levels and search depth with a GUI workspace.

Each backend has five effort levels; a level has a display name and the
chat model it answers with. Search depth is a separate vector of five
steps, each the number of passages retrieved for an answer. The composer
picks an effort level and a search-depth step independently. The file is
written only when a project differs from the defaults.

The defaults themselves, and the validation shared with the app-wide
preferences, live in ``qachat.gui.preferences`` (``defaults.json``); they
are imported here so existing imports keep working.

"""

import json
from pathlib import Path

from qachat.gui.preferences import (
    LEVEL_COUNT,
    MAX_LEVEL_NAME,
    MAX_SEARCH_DEPTH,
    default_effort_levels,
    default_router_model,
    default_search_depth,
    validate_effort_levels,
    validate_model_id,
    validate_search_depth,
)
from qachat.runtime import BACKEND_MLX

__all__ = [
    "DEFAULT_LEVEL",
    "EFFORT_FILE",
    "LEVEL_COUNT",
    "MAX_LEVEL_NAME",
    "MAX_SEARCH_DEPTH",
    "ROUTER_KEY",
    "SEARCH_DEPTH_KEY",
    "default_effort_levels",
    "default_router_model",
    "default_search_depth",
    "read_effort_levels",
    "read_router_model",
    "read_search_depth",
    "validate_effort_levels",
    "validate_model_id",
    "validate_search_depth",
    "write_effort_levels",
    "write_router_model",
    "write_search_depth",
]

EFFORT_FILE = "effort_levels.json"
# The same file keeps a customized intent router model per backend.
ROUTER_KEY = "router"
# Earlier GUI versions stored per-preset model replacements here.
LEGACY_FILE = "model_choices.json"
# The level whose model is also the project's default chat model.
DEFAULT_LEVEL = 2
# Search depth: passages retrieved at each of five steps, project-wide.
SEARCH_DEPTH_KEY = "search_depth"
# Per-level reference counts of earlier versions, before search depth.
_LEGACY_RETRIEVAL = (5, 20, 20, 50, 50)


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
    return _legacy_depth(saved) or default_search_depth()


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
    if steps == default_search_depth():
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
