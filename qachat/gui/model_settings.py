"""Keep a project's effort levels, search depth, and router in profile.ini.

Each backend has five effort levels; a level has a display name and the
chat model it answers with. Search depth is a separate vector of five
steps, each the number of passages retrieved for an answer. The composer
picks an effort level and a search-depth step independently. The intent
router model is chosen per backend.

Every project keeps all its settings, changed or not, in its own
``profile.ini``, next to the core settings, so it resumes exactly as it was
set up even when ``qachat/gui/defaults.json`` changes later:

.. code-block:: ini

    [effort.mlx]
    level1 = Light | Qwen/Qwen3-0.6B-MLX-4bit
    ...
    level5 = Deep | Qwen/Qwen3-14B-MLX-4bit

    [search]
    depth = 5, 5, 10, 10, 15

    [router]
    mlx = Qwen/Qwen3-8B-MLX-4bit
    torch = Qwen/Qwen3-8B

``save_project_settings`` writes them when a project is created or first
opened, from the defaults, or from the ``effort_levels.json`` and
``model_choices.json`` files earlier versions kept, which it then removes.

The defaults and the shared validation live in
``qachat.gui.preferences``.

Key Features
------------

- **Project-local settings**

    Read and update effort levels, search-depth steps, and router models in
    the project's ``profile.ini``.

- **Legacy migration**

    Import compatible settings from earlier project files when they exist.

Module Structure
----------------

- ``read_effort_levels``, ``write_effort_levels``

    Read and write the five backend-specific effort levels.

- ``read_search_depth``, ``write_search_depth``

    Read and write the five retrieval-depth steps.

- ``read_router_model``, ``write_router_model``

    Read and write the project router model.

- ``save_project_settings``

    Write missing settings or migrate compatible legacy values.

Examples
--------
>>> from qachat.gui.model_settings import read_search_depth
>>> len(read_search_depth(None))
5

"""

from configparser import ConfigParser
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
from qachat.runtime import BACKEND_MLX, BACKEND_TORCH

__all__ = [
    "DEFAULT_LEVEL",
    "LEVEL_COUNT",
    "MAX_LEVEL_NAME",
    "MAX_SEARCH_DEPTH",
    "PROFILE_FILE",
    "read_effort_levels",
    "read_router_model",
    "read_search_depth",
    "save_project_settings",
    "write_effort_levels",
    "write_router_model",
    "write_search_depth",
]

# The project's settings file, shared with the core library.
PROFILE_FILE = "profile.ini"
# The level whose model is also the project's default chat model.
DEFAULT_LEVEL = 2
# Sections and keys in profile.ini.
_EFFORT_SECTION = "effort.{backend}"
_SEARCH_SECTION = "search"
_ROUTER_SECTION = "router"
_SEPARATOR = " | "
# Files of earlier versions, migrated into profile.ini and removed.
_LEGACY_EFFORT_FILE = "effort_levels.json"
_LEGACY_CHOICES_FILE = "model_choices.json"
# Per-level reference counts of earlier versions, before search depth.
_LEGACY_RETRIEVAL = (5, 20, 20, 50, 50)


def read_effort_levels(
    root: Path | None, backend: str
) -> list[dict[str, object]]:
    """Return a project's levels for one backend.

    Parameters
    ----------
    root: pathlib.Path | None
        Project workspace, or None for a project not created yet.
    backend: str
        Computation backend whose levels are wanted.

    Returns
    -------
    list[dict[str, object]]
        Five ``{"name", "model"}`` levels from ``profile.ini``; for a
        project that has not saved them yet, from an earlier version's
        files, otherwise the defaults.

    """

    if root is None:
        return default_effort_levels(backend)
    profile = _read(root)
    section = _EFFORT_SECTION.format(backend=backend)
    if profile.has_section(section):
        levels = []
        for index in range(1, LEVEL_COUNT + 1):
            name, _, model = profile.get(
                section, f"level{index}", fallback=""
            ).partition(_SEPARATOR.strip())
            levels.append({"name": name.strip(), "model": model.strip()})
        return validate_effort_levels(levels)
    legacy = _legacy(root)
    if backend in legacy:
        return validate_effort_levels(
            [
                {"name": level["name"], "model": level["model"]}
                for level in legacy[backend]
            ]
        )
    levels = default_effort_levels(backend)
    choices = _read_json(root / _LEGACY_CHOICES_FILE)
    replacements = choices if isinstance(choices, dict) else {}
    for level in levels:
        replacement = replacements.get(level["model"])
        if isinstance(replacement, str) and replacement.strip():
            level["model"] = validate_model_id(replacement)
    return levels


def write_effort_levels(
    root: Path, backend: str, levels: list[dict[str, object]]
) -> None:
    """Save a project's levels for one backend in ``profile.ini``.

    Parameters
    ----------
    root: pathlib.Path
        Project workspace.
    backend: str
        Computation backend the levels belong to.
    levels: list[dict[str, object]]
        Five levels to validate and save. A name may not contain ``|``.

    Raises
    ------
    ValueError
        If a level is invalid or a name contains the separator ``|``.

    """

    levels = validate_effort_levels(levels)
    if any(_SEPARATOR.strip() in str(level["name"]) for level in levels):
        raise ValueError("Effort level names cannot contain '|'.")
    profile = _read(root)
    section = _EFFORT_SECTION.format(backend=backend)
    if not profile.has_section(section):
        profile.add_section(section)
    for index, level in enumerate(levels, start=1):
        profile.set(
            section,
            f"level{index}",
            f"{level['name']}{_SEPARATOR}{level['model']}",
        )
    _write(root, profile)


def read_search_depth(root: Path | None) -> list[int]:
    """Return a project's search-depth steps.

    Parameters
    ----------
    root: pathlib.Path | None
        Project workspace, or None for a project not created yet.

    Returns
    -------
    list[int]
        Five steps from ``profile.ini``; for a project that has not saved
        them yet, from an earlier version's files, otherwise the defaults.

    Raises
    ------
    ValueError
        If saved search-depth values cannot be parsed or validated.

    """

    if root is None:
        return default_search_depth()
    profile = _read(root)
    if profile.has_option(_SEARCH_SECTION, "depth"):
        try:
            steps = [
                int(part)
                for part in profile.get(_SEARCH_SECTION, "depth").split(",")
            ]
        except ValueError as error:
            raise ValueError(
                "Search depth steps must be whole numbers."
            ) from error
        return validate_search_depth(steps)
    legacy = _legacy(root)
    if "search_depth" in legacy:
        return validate_search_depth(legacy["search_depth"])
    return _legacy_depth(legacy) or default_search_depth()


def write_search_depth(root: Path, steps: list[int]) -> None:
    """Save a project's search-depth steps in ``profile.ini``.

    Parameters
    ----------
    root: pathlib.Path
        Project workspace.
    steps: list[int]
        Five steps to validate and save.

    """

    steps = validate_search_depth(steps)
    profile = _read(root)
    if not profile.has_section(_SEARCH_SECTION):
        profile.add_section(_SEARCH_SECTION)
    profile.set(_SEARCH_SECTION, "depth", ", ".join(map(str, steps)))
    _write(root, profile)


def read_router_model(root: Path | None, backend: str) -> str:
    """Return a project's intent router model for one backend.

    Parameters
    ----------
    root: pathlib.Path | None
        Project workspace, or None for a project not created yet.
    backend: str
        Computation backend whose router model is wanted.

    Returns
    -------
    str
        The saved model ID; for a project that has not saved one yet, an
        earlier version's choice, otherwise the default.

    """

    if root is None:
        return default_router_model(backend)
    profile = _read(root)
    if profile.has_option(_ROUTER_SECTION, backend):
        return validate_model_id(profile.get(_ROUTER_SECTION, backend))
    routers = _legacy(root).get("router")
    if isinstance(routers, dict) and isinstance(routers.get(backend), str):
        return validate_model_id(routers[backend])
    return default_router_model(backend)


def write_router_model(root: Path, backend: str, model_id: str) -> None:
    """Save a project's intent router model for one backend.

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
    profile = _read(root)
    if not profile.has_section(_ROUTER_SECTION):
        profile.add_section(_ROUTER_SECTION)
    profile.set(_ROUTER_SECTION, backend, model_id)
    _write(root, profile)


def save_project_settings(root: Path) -> bool:
    """Write every setting a project's ``profile.ini`` does not keep yet.

    Parameters
    ----------
    root: pathlib.Path
        Project workspace.

    Returns
    -------
    bool
        Whether anything was written. Saved values are never changed;
        missing ones come from an earlier version's ``effort_levels.json``
        and ``model_choices.json`` (then removed) or the defaults, so the
        project no longer follows later changes to the defaults.

    """

    profile = _read(root)
    changed = False
    for backend in (BACKEND_MLX, BACKEND_TORCH):
        if not profile.has_section(_EFFORT_SECTION.format(backend=backend)):
            levels = read_effort_levels(root, backend)
            write_effort_levels(root, backend, levels)
            changed = True
        if not profile.has_option(_ROUTER_SECTION, backend):
            write_router_model(root, backend, read_router_model(root, backend))
            changed = True
    if not profile.has_option(_SEARCH_SECTION, "depth"):
        write_search_depth(root, read_search_depth(root))
        changed = True
    for name in (_LEGACY_EFFORT_FILE, _LEGACY_CHOICES_FILE):
        if (root / name).exists():
            (root / name).unlink()
            changed = True
    return changed


def _read(root: Path) -> ConfigParser:
    # No interpolation: model IDs and level names are taken literally.
    """Read the project profile without interpolating its values.

    Parameters
    ----------
    root: Path
        Directory holding the project profile.

    Returns
    -------
    ConfigParser
        Parser loaded from the current project profile.

    """

    profile = ConfigParser(interpolation=None)
    profile.read(Path(root) / PROFILE_FILE, encoding="utf-8")
    return profile


def _write(root: Path, profile: ConfigParser) -> None:
    """Write the project profile using an atomic replacement.

    Parameters
    ----------
    root: Path
        Project or workspace root directory.
    profile: ConfigParser
        Profile contents to write to disk.

    """

    path = Path(root) / PROFILE_FILE
    temporary = path.with_suffix(".ini.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        profile.write(stream)
    temporary.replace(path)


def _read_json(path: Path) -> object:
    """Read a JSON file, returning None if the file is absent.

    Parameters
    ----------
    path: Path
        Optional JSON file to parse.

    Returns
    -------
    object
        Parsed JSON object, or None when the file does not exist.

    Raises
    ------
    json.JSONDecodeError
        If the file does not contain valid JSON.

    """

    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _legacy(root: Path) -> dict:
    """The ``effort_levels.json`` an earlier version kept, if any.

    Parameters
    ----------
    root: Path
        Project or workspace root directory.

    Returns
    -------
    dict
        Earlier-version effort settings, or an empty mapping.

    """

    saved = _read_json(root / _LEGACY_EFFORT_FILE)
    return saved if isinstance(saved, dict) else {}


def _legacy_depth(saved: dict) -> list[int] | None:
    """Customized per-level counts saved by even earlier versions.

    Parameters
    ----------
    saved: dict
        Settings loaded from an earlier profile format.

    Returns
    -------
    list[int] | None
        Earlier per-level retrieval counts, or None when unavailable.

    """

    for key, levels in saved.items():
        if key in ("router", "search_depth") or not isinstance(levels, list):
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
