"""Default settings and app-wide preferences of the desktop app.

Every default setting of the desktop app is in one file,
``qachat/gui/defaults.json``: the appearance, the five effort levels (names
and models) per backend, the search depth, the intent router and embedding
models per backend, the embedding query format, and the answer settings.
Edit that file to change what everyone starts with; restart QAChat after.

Each user's own changes are kept apart, in ``qachat/gui/preferences.json``
(git-ignored), and only where they differ from the defaults. New projects
start from the defaults plus these preferences; each project then keeps
its own copy (``profile.ini`` and ``effort_levels.json``).

The core library has its own defaults for notebooks
(``qachat.model_defaults``); ``defaults.json`` decides what the desktop
app uses.

Key Features
------------

- **One place for defaults**: ``defaults.json``, validated on first use; a
  mistake names the setting.
- **Personal overrides**: ``read_preferences`` and ``write_preferences``
  keep only values that differ from the defaults.
- **Shared validation**: ``validate_model_id``, ``validate_effort_levels``,
  and ``validate_search_depth`` also check project settings.
- **Required models**: ``required_models`` lists what the preferences use,
  so the first-run check asks only for those.

Module Structure
----------------

- ``DEFAULTS_FILE`` and ``built_in_defaults``
- ``default_effort_levels``, ``default_search_depth``,
  ``default_router_model``, ``default_embedding_model``, and
  ``default_answer_settings``
- ``preferences_file``, ``read_preferences``, and ``write_preferences``
- ``preferred_*`` and ``required_models``

"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Final

from huggingface_hub.utils import validate_repo_id

from qachat.runtime import BACKEND_MLX, BACKEND_TORCH

__all__ = [
    "APPEARANCE_CHOICES",
    "DEFAULTS_FILE",
    "LEVEL_COUNT",
    "MAX_LEVEL_NAME",
    "MAX_SEARCH_DEPTH",
    "PREFERENCES_ENVIRONMENT",
    "PREFERENCES_FILE",
    "built_in_defaults",
    "default_answer_settings",
    "default_effort_levels",
    "default_embedding_model",
    "default_router_model",
    "default_search_depth",
    "preferences_file",
    "preferred_appearance",
    "preferred_effort_levels",
    "preferred_embedding_model",
    "preferred_query_format",
    "preferred_router_model",
    "preferred_search_depth",
    "read_preferences",
    "required_models",
    "validate_effort_levels",
    "validate_model_id",
    "validate_search_depth",
    "write_preferences",
]

# The shipped defaults: edit this file to change what everyone starts with.
DEFAULTS_FILE: Final[Path] = Path(__file__).with_name("defaults.json")
# Each user's own changes, written by the app.
PREFERENCES_FILE: Final[Path] = Path(__file__).with_name("preferences.json")
# Tests and special installs can point QAChat at another preferences file.
PREFERENCES_ENVIRONMENT: Final[str] = "QACHAT_PREFERENCES_FILE"
# The page's appearance choices; "system" follows the operating system.
APPEARANCE_CHOICES: Final[tuple[str, ...]] = ("system", "light", "dark")
LEVEL_COUNT: Final[int] = 5
MAX_LEVEL_NAME: Final[int] = 24
MAX_SEARCH_DEPTH: Final[int] = 200
_BACKENDS: Final[tuple[str, ...]] = (BACKEND_MLX, BACKEND_TORCH)
_QUERY_FORMATS: Final[tuple[str, ...]] = ("qwen3", "plain")


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


def validate_effort_levels(value: object) -> list[dict[str, object]]:
    """Check five levels of name and model ID.

    Parameters
    ----------
    value: object
        Levels as received from the page or read from a file.

    Returns
    -------
    list[dict[str, object]]
        Levels with names collapsed to single spaces. A ``retrieval``
        count saved by earlier versions is ignored; see
        ``qachat.gui.model_settings.read_search_depth``.

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
        Steps as received from the page or read from a file.

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


def _validate_answers(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("must be an object.")
    temperature = value.get("temperature")
    thinking = value.get("thinking")
    tokens = value.get("max_new_tokens")
    if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be from 0 to 2.")
    if type(thinking) is not bool:
        raise ValueError("thinking must be true or false.")
    if tokens is not None and (type(tokens) is not int or tokens < 1):
        raise ValueError("max_new_tokens must be a positive number or null.")
    return {
        "temperature": float(temperature),
        "thinking": thinking,
        "max_new_tokens": tokens,
    }


def _per_backend(check: Callable[[object], object]) -> Callable:
    def checked(value: object) -> dict[str, object]:
        if not isinstance(value, dict) or set(value) != set(_BACKENDS):
            raise ValueError("give exactly one value for mlx and torch.")
        return {backend: check(value[backend]) for backend in _BACKENDS}

    return checked


def _choice(options: tuple[str, ...]) -> Callable:
    def checked(value: object) -> str:
        if value not in options:
            raise ValueError(f"choose one of: {', '.join(options)}.")
        return str(value)

    return checked


@lru_cache(maxsize=None)
def built_in_defaults(path: Path = DEFAULTS_FILE) -> dict[str, object]:
    """Read and validate the shipped defaults.

    Parameters
    ----------
    path: pathlib.Path, optional
        Defaults file. Defaults to ``DEFAULTS_FILE``.

    Returns
    -------
    dict[str, object]
        Every default setting; cached for the process (restart QAChat after
        editing the file).

    Raises
    ------
    ValueError
        If the file is not valid JSON or a setting is missing or invalid;
        the message names the setting.

    """

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    checks = {
        "appearance": _choice(APPEARANCE_CHOICES),
        "effort_levels": _per_backend(validate_effort_levels),
        "search_depth": validate_search_depth,
        "router_models": _per_backend(validate_model_id),
        "embedding_models": _per_backend(validate_model_id),
        "query_format": _choice(_QUERY_FORMATS),
        "answers": _validate_answers,
    }
    defaults: dict[str, object] = {}
    for name, check in checks.items():
        if name not in data:
            raise ValueError(f"{path} needs the setting {name!r}.")
        try:
            defaults[name] = check(data[name])
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path} setting {name!r}: {error}") from error
    return defaults


def default_effort_levels(backend: str) -> list[dict[str, object]]:
    """Return the default effort levels for a backend (a fresh copy).

    Parameters
    ----------
    backend: str
        Computation backend, ``mlx`` or ``torch``.

    Returns
    -------
    list[dict[str, object]]
        Five levels of ``name`` and ``model`` from ``defaults.json``.

    """

    return [
        dict(level) for level in built_in_defaults()["effort_levels"][backend]
    ]


def default_search_depth() -> list[int]:
    """Return the default search-depth steps (a fresh copy)."""

    return list(built_in_defaults()["search_depth"])


def default_router_model(backend: str) -> str:
    """Return the default intent router model for a backend."""

    return str(built_in_defaults()["router_models"][backend])


def default_embedding_model(backend: str) -> str:
    """Return the default embedding model for a backend."""

    return str(built_in_defaults()["embedding_models"][backend])


def default_answer_settings() -> dict[str, object]:
    """Return the default temperature, thinking, and longest answer."""

    return dict(built_in_defaults()["answers"])


def preferences_file() -> Path:
    """Return the preferences file location.

    Returns
    -------
    pathlib.Path
        ``$QACHAT_PREFERENCES_FILE`` when set (tests use it), otherwise
        ``qachat/gui/preferences.json``.

    """

    override = os.environ.get(PREFERENCES_ENVIRONMENT)
    return Path(override) if override else PREFERENCES_FILE


def read_preferences(path: Path | None = None) -> dict[str, object]:
    """Read the user's saved preferences, keeping only valid entries.

    Parameters
    ----------
    path: pathlib.Path | None, optional
        Preferences file. Defaults to ``preferences_file()``.

    Returns
    -------
    dict[str, object]
        Valid saved entries; empty when the file is missing or unreadable.

    """

    try:
        value = json.loads(
            Path(path or preferences_file()).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}
    if not isinstance(value, dict):
        return {}
    clean: dict[str, object] = {}
    if value.get("appearance") in APPEARANCE_CHOICES:
        clean["appearance"] = value["appearance"]
    if value.get("query_format") in _QUERY_FORMATS:
        clean["query_format"] = value["query_format"]
    try:
        if "search_depth" in value:
            clean["search_depth"] = validate_search_depth(
                value["search_depth"]
            )
    except ValueError:
        pass
    for key, check in (
        ("effort_levels", validate_effort_levels),
        ("router_models", validate_model_id),
        ("embedding_models", validate_model_id),
    ):
        entries = value.get(key)
        if not isinstance(entries, dict):
            continue
        kept = {}
        for backend in _BACKENDS:
            if backend in entries:
                try:
                    kept[backend] = check(entries[backend])
                except (ValueError, TypeError):
                    continue
        if kept:
            clean[key] = kept
    return clean


def write_preferences(
    changes: dict[str, object], path: Path | None = None
) -> dict[str, object]:
    """Merge changes into the preferences, dropping default values.

    Parameters
    ----------
    changes: dict[str, object]
        Any of ``appearance``, ``query_format``, ``search_depth``, and
        ``effort_levels``, ``router_models``, or ``embedding_models`` as
        ``{backend: value}``.
    path: pathlib.Path | None, optional
        Preferences file. Defaults to ``preferences_file()``.

    Returns
    -------
    dict[str, object]
        The preferences now saved; the file is removed when none differ
        from the defaults.

    Raises
    ------
    ValueError
        If a value is invalid; nothing is written then.

    """

    path = Path(path or preferences_file())
    defaults = built_in_defaults()
    saved = read_preferences(path)
    for key, options in (
        ("appearance", APPEARANCE_CHOICES),
        ("query_format", _QUERY_FORMATS),
    ):
        if key in changes:
            if changes[key] not in options:
                raise ValueError(
                    f"Choose a supported {key.replace('_', ' ')}."
                )
            saved[key] = changes[key]
            if saved[key] == defaults[key]:
                saved.pop(key)
    if "search_depth" in changes:
        depth = validate_search_depth(changes["search_depth"])
        saved["search_depth"] = depth
        if depth == defaults["search_depth"]:
            saved.pop("search_depth")
    for key, check in (
        ("effort_levels", validate_effort_levels),
        ("router_models", validate_model_id),
        ("embedding_models", validate_model_id),
    ):
        entries = changes.get(key)
        if not isinstance(entries, dict):
            continue
        kept = dict(saved.get(key) or {})
        for backend, value in entries.items():
            if backend not in _BACKENDS:
                raise ValueError(f"Unknown backend {backend!r}.")
            value = check(value)
            if value == defaults[key][backend]:
                kept.pop(backend, None)
            else:
                kept[backend] = value
        if kept:
            saved[key] = kept
        else:
            saved.pop(key, None)
    if saved:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(saved, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    else:
        path.unlink(missing_ok=True)
    return saved


def _preferred(key: str, path: Path | None) -> object:
    return read_preferences(path).get(key)


def preferred_effort_levels(
    backend: str, path: Path | None = None
) -> list[dict[str, object]]:
    """Return the effort levels new projects start from."""

    levels = (_preferred("effort_levels", path) or {}).get(backend)
    return levels or default_effort_levels(backend)


def preferred_search_depth(path: Path | None = None) -> list[int]:
    """Return the search depth new projects start from."""

    return _preferred("search_depth", path) or default_search_depth()


def preferred_router_model(backend: str, path: Path | None = None) -> str:
    """Return the intent router model new projects start from."""

    routers = _preferred("router_models", path) or {}
    return routers.get(backend) or default_router_model(backend)


def preferred_embedding_model(
    backend: str, path: Path | None = None
) -> str:
    """Return the embedding model new projects start from."""

    models = _preferred("embedding_models", path) or {}
    return models.get(backend) or default_embedding_model(backend)


def preferred_query_format(path: Path | None = None) -> str:
    """Return the embedding query format new projects start from."""

    return str(
        _preferred("query_format", path)
        or built_in_defaults()["query_format"]
    )


def preferred_appearance(path: Path | None = None) -> str:
    """Return the appearance: system, light, or dark."""

    return str(
        _preferred("appearance", path) or built_in_defaults()["appearance"]
    )


def required_models(backend: str, path: Path | None = None) -> list[str]:
    """List the models the preferences use, for the first-run check.

    Parameters
    ----------
    backend: str
        Computation backend.
    path: pathlib.Path | None, optional
        Preferences file. Defaults to ``preferences_file()``.

    Returns
    -------
    list[str]
        The effort levels' models, the router model, and the embedding
        model, without repeats, in that order.

    """

    models = [
        str(level["model"])
        for level in preferred_effort_levels(backend, path)
    ]
    models.append(preferred_router_model(backend, path))
    models.append(preferred_embedding_model(backend, path))
    return list(dict.fromkeys(models))
