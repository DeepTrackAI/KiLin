"""Validate GUI defaults and persist user-owned preference overrides.

The package ships immutable ``kilin/gui/defaults.json``. Application-wide
changes are written to a user-owned ``preferences.json`` and merged with the
shipped defaults when initializing new projects. Project-specific settings
remain authoritative in each workspace's ``profile.ini``.

Key Features
------------

- **Immutable packaged defaults**

    Validate the complete shipped configuration independently of user edits.

- **Sparse user overrides**

    Persist only differences from packaged defaults. Missing user preferences
    are normal; invalid overrides fail explicitly instead of being discarded.

- **Existing project precedence**

    Default values affect new projects, not saved project profiles.

Module Structure
----------------

- ``built_in_defaults`` and ``app_defaults``

    Load validated packaged defaults and merged application settings.

- ``write_defaults``

    Validate and atomically save application-wide changes to user storage.

- ``default_effort_levels``, ``default_search_depth``, and related helpers

    Supply the effective settings used to create a new project.

Examples
--------
>>> from kilin.gui.preferences import default_search_depth
>>> len(default_search_depth())
5

"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import json
from pathlib import Path
from typing import Final

from huggingface_hub.utils import validate_repo_id

from kilin.atomic_writes import atomic_write_text
from kilin.gui._paths import (
    PREFERENCES_ENVIRONMENT,
    preferences_path,
)
from kilin.runtime import BACKEND_MLX, BACKEND_TORCH

__all__ = [
    "APPEARANCE_CHOICES",
    "DEFAULTS_ENVIRONMENT",
    "DEFAULTS_FILE",
    "LEVEL_COUNT",
    "MAX_LEVEL_NAME",
    "MAX_SEARCH_DEPTH",
    "app_defaults",
    "built_in_defaults",
    "default_appearance",
    "default_effort_levels",
    "default_embedding_model",
    "default_query_format",
    "default_router_model",
    "default_search_depth",
    "defaults_file",
    "preferences_file",
    "required_models",
    "validate_effort_levels",
    "validate_model_id",
    "validate_search_depth",
    "write_defaults",
]

# Shipped configuration is read-only and has no runtime write path.
DEFAULTS_FILE: Final[Path] = Path(__file__).with_name("defaults.json")
# An explicit override selects a writable user-preferences file.
DEFAULTS_ENVIRONMENT: Final[str] = PREFERENCES_ENVIRONMENT
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
        ``kilin.gui.model_settings.read_search_depth``.

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


def _per_backend(check: Callable[[object], object]) -> Callable:
    """Create a validator for per-backend settings dictionaries.

    Parameters
    ----------
    check: Callable[[object], object]
        Validation function for each backend-specific entry.

    Returns
    -------
    Callable
        Validator for dictionaries with one entry per backend.

    """

    def checked(value: object) -> dict[str, object]:
        """Validate each configured backend using the supplied validator.

        Parameters
        ----------
        value: object
            Value to validate or convert.

        Returns
        -------
        dict[str, object]
            Validated values indexed by the two supported backends.

        Raises
        ------
        ValueError
            If the mapping omits a backend or contains extra keys.

        """

        if not isinstance(value, dict) or set(value) != set(_BACKENDS):
            raise ValueError("give exactly one value for mlx and torch.")
        return {backend: check(value[backend]) for backend in _BACKENDS}

    return checked


def _choice(options: tuple[str, ...]) -> Callable:
    """Create a validator accepting exactly the given string choices.

    Parameters
    ----------
    options: tuple[str, ...]
        Permitted string choices for the returned validator.

    Returns
    -------
    Callable
        Validator that permits only the supplied choices.

    """

    def checked(value: object) -> str:
        """Return an allowed choice or reject an unsupported value.

        Parameters
        ----------
        value: object
            Value to validate or convert.

        Returns
        -------
        str
            The accepted choice as a string.

        Raises
        ------
        ValueError
            If the value is not one of the supported choices.

        """

        if value not in options:
            raise ValueError(f"choose one of: {', '.join(options)}.")
        return str(value)

    return checked


# Validated complete packaged defaults, cached by modification time and size.
_CACHE: dict[Path, tuple[int, int, dict[str, object]]] = {}


def defaults_file() -> Path:
    """Return the immutable packaged GUI defaults file.

    Returns
    -------
    pathlib.Path
        Shipped ``kilin/gui/defaults.json``.

    """

    return DEFAULTS_FILE


def preferences_file() -> Path:
    """Return the file used for writable application-wide overrides.

    Returns
    -------
    pathlib.Path
        ``KILIN_DEFAULTS_FILE`` when explicitly supplied, or a
        platform-specific user configuration path.

    """

    return preferences_path()


def _checks() -> dict[str, Callable[[object], object]]:
    """Map each application preference name to its validator.

    Returns
    -------
    dict[str, Callable[[object], object]]
        Mapping of setting names to validation functions.

    """

    return {
        "appearance": _choice(APPEARANCE_CHOICES),
        "effort_levels": _per_backend(validate_effort_levels),
        "search_depth": validate_search_depth,
        "router_models": _per_backend(validate_model_id),
        "embedding_models": _per_backend(validate_model_id),
        "query_format": _choice(_QUERY_FORMATS),
    }


def built_in_defaults(path: Path | None = None) -> dict[str, object]:
    """Read and validate the complete, immutable packaged defaults.

    Parameters
    ----------
    path: pathlib.Path | None, optional
        Complete defaults file. By default, use the shipped configuration.
        An explicit path supports controlled validation without writing.

    Returns
    -------
    dict[str, object]
        Fresh copy of all validated built-in settings.

    Raises
    ------
    ValueError
        If JSON, a required setting, or a configured value is invalid.

    """

    path = Path(path) if path is not None else defaults_file()
    stat = path.stat()
    cached = _CACHE.get(path)
    if (
        cached is not None
        and cached[0] == stat.st_mtime_ns
        and cached[1] == stat.st_size
    ):
        return deepcopy(cached[2])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    defaults: dict[str, object] = {}
    for name, check in _checks().items():
        if name not in data:
            raise ValueError(f"{path} needs the setting {name!r}.")
        try:
            defaults[name] = check(data[name])
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path} setting {name!r}: {error}") from error
    _CACHE[path] = (stat.st_mtime_ns, stat.st_size, defaults)
    return deepcopy(defaults)


def _apply_changes(
    values: dict[str, object], changes: dict[str, object]
) -> None:
    """Validate and merge settings, including individual backend entries.

    Parameters
    ----------
    values: dict[str, object]
        Complete settings mapping modified after successful validation.
    changes: dict[str, object]
        Sparse top-level or backend-specific changes to apply.

    Raises
    ------
    ValueError
        If a setting name, backend, type, or value is unsupported.

    """

    checks = _checks()
    backend_checks = {
        "effort_levels": validate_effort_levels,
        "router_models": validate_model_id,
        "embedding_models": validate_model_id,
    }
    for key, value in changes.items():
        if key in backend_checks:
            if not isinstance(value, dict):
                raise ValueError(f"{key} must map backends to values.")
            for backend, entry in value.items():
                if backend not in _BACKENDS:
                    raise ValueError(f"Unknown backend {backend!r}.")
                try:
                    values[key][backend] = backend_checks[key](entry)
                except (ValueError, TypeError) as error:
                    raise ValueError(
                        f"{key} {backend!r}: {error}"
                    ) from error
        elif key in checks:
            try:
                values[key] = checks[key](value)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{key}: {error}") from error
        else:
            raise ValueError(f"Unknown default setting {key!r}.")


def _read_preferences(path: Path) -> dict[str, object]:
    """Read user overrides without creating or repairing their file.

    Parameters
    ----------
    path: pathlib.Path
        Location of the optional user-preferences JSON file.

    Returns
    -------
    dict[str, object]
        Sparse override mapping or an empty dictionary if no file exists.

    Raises
    ------
    ValueError
        If the file is malformed or contains invalid settings.

    """

    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def app_defaults(path: Path | None = None) -> dict[str, object]:
    """Combine immutable packaged defaults with validated user overrides.

    Parameters
    ----------
    path: pathlib.Path | None, optional
        Writable preferences file. Defaults to ``preferences_file()``.

    Returns
    -------
    dict[str, object]
        Complete effective settings for new projects and GUI presentation.

    Raises
    ------
    ValueError
        If saved preferences contain invalid names or values.

    """

    path = Path(path) if path is not None else preferences_file()
    values = built_in_defaults()
    try:
        _apply_changes(values, _read_preferences(path))
    except (ValueError, TypeError) as error:
        raise ValueError(f"{path}: {error}") from error
    return values


def _overrides(
    defaults: dict[str, object], values: dict[str, object]
) -> dict[str, object]:
    """Compute only settings that differ from the packaged defaults.

    Parameters
    ----------
    defaults: dict[str, object]
        Complete validated packaged configuration.
    values: dict[str, object]
        Complete effective user configuration.

    Returns
    -------
    dict[str, object]
        Sparse differences, including only changed backend values.

    """

    result: dict[str, object] = {}
    for key, value in values.items():
        original = defaults[key]
        if key in ("effort_levels", "router_models", "embedding_models"):
            changed = {
                backend: deepcopy(value[backend])
                for backend in _BACKENDS
                if value[backend] != original[backend]
            }
            if changed:
                result[key] = changed
        elif value != original:
            result[key] = deepcopy(value)
    return result


def write_defaults(
    changes: dict[str, object], path: Path | None = None
) -> dict[str, object]:
    """Atomically save sparse app-wide overrides outside the package.

    Parameters
    ----------
    changes: dict[str, object]
        Values or backend-specific values to update. Unchanged settings
        remain in effect; restoring a built-in value drops its override.
    path: pathlib.Path | None, optional
        Writable preference file, defaulting to ``preferences_file()``.

    Returns
    -------
    dict[str, object]
        Validated effective settings after the update.

    Raises
    ------
    ValueError
        If changes or existing preferences are invalid, or the destination
        refers to the immutable packaged defaults.
    OSError
        If the user preferences cannot be written.

    Notes
    -----
    The destination directory is created only for a write. No existing
    file is changed when validation fails. A write does not alter a
    saved project's ``profile.ini``.

    """

    path = Path(path) if path is not None else preferences_file()
    if path.resolve() == defaults_file().resolve():
        raise ValueError("Packaged GUI defaults are read-only.")
    effective = app_defaults(path)
    _apply_changes(effective, changes)
    if not changes:
        return effective
    delta = _overrides(built_in_defaults(), effective)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, _format(delta) + "\n")
    return effective


def _format(value: object, indent: int = 0) -> str:
    """JSON with short lists and objects on one line, as the file is kept.

    Parameters
    ----------
    value: object
        Value to validate or convert.
    indent: int, optional
        Indentation depth for serialized output. Defaults to ``0``.

    Returns
    -------
    str
        Formatted JSON string for the preferences file.

    """

    inline = json.dumps(value, ensure_ascii=False)
    nested = isinstance(value, dict) and any(
        isinstance(item, (dict, list)) for item in value.values()
    )
    if not isinstance(value, (dict, list)) or (
        not nested and len(inline) + indent <= 76
    ):
        return inline
    pad = " " * (indent + 2)
    if isinstance(value, dict):
        items = [
            f"{pad}{json.dumps(key)}: {_format(item, indent + 2)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    items = [f"{pad}{_format(item, indent + 2)}" for item in value]
    return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"


def default_effort_levels(backend: str) -> list[dict[str, object]]:
    """Return the default effort levels for a backend (a fresh copy).

    Parameters
    ----------
    backend: str
        Computation backend, ``mlx`` or ``torch``.

    Returns
    -------
    list[dict[str, object]]
        Five levels of ``name`` and ``model`` from effective preferences.

    """

    return [
        dict(level) for level in app_defaults()["effort_levels"][backend]
    ]


def default_search_depth() -> list[int]:
    """Return the default search-depth steps (a fresh copy).

    Returns
    -------
    list[int]
        the default search-depth steps (a fresh copy).

    """

    return list(app_defaults()["search_depth"])


def default_router_model(backend: str) -> str:
    """Return the default intent router model for a backend.

    Parameters
    ----------
    backend: str
        Model backend for which settings are requested.

    Returns
    -------
    str
        the default intent router model for a backend.

    """

    return str(app_defaults()["router_models"][backend])


def default_embedding_model(backend: str) -> str:
    """Return the default embedding model for a backend.

    Parameters
    ----------
    backend: str
        Model backend for which settings are requested.

    Returns
    -------
    str
        the default embedding model for a backend.

    """

    return str(app_defaults()["embedding_models"][backend])


def default_query_format() -> str:
    """Return the default embedding query format.

    Returns
    -------
    str
        the default embedding query format.

    """

    return str(app_defaults()["query_format"])


def default_appearance() -> str:
    """Return the appearance: system, light, or dark.

    Returns
    -------
    str
        the appearance: system, light, or dark.

    """

    return str(app_defaults()["appearance"])


def required_models(backend: str) -> list[str]:
    """List the models the defaults use, for the first-run check.

    Parameters
    ----------
    backend: str
        Computation backend.

    Returns
    -------
    list[str]
        The effort levels' models, the router model, and the embedding
        model, without repeats, in that order.

    """

    models = [str(level["model"]) for level in default_effort_levels(backend)]
    models.append(default_router_model(backend))
    models.append(default_embedding_model(backend))
    return list(dict.fromkeys(models))
