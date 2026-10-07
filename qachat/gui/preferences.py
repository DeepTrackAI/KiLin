"""Default settings of the desktop app, in one file.

Every default setting of the desktop app is in ``qachat/gui/defaults.json``:
the appearance, the five effort levels (names and models) per backend, the
search depth, the intent router and embedding models per backend, the
embedding query format, and the answer settings. Edit that file to change
what new projects start with; restart QAChat after. The app writes to it
only for app-wide choices: the appearance, "Use installed models", and
settings edited while no project is open.

Each project then keeps all of its own settings in its ``profile.ini``, so
a project never changes when the defaults change. The core library has its
own defaults for notebooks (``qachat.model_defaults``).

Key Features
------------

- **One place for defaults**: ``defaults.json``, validated when read; a
  mistake names the setting.
- **Shared validation**: ``validate_model_id``, ``validate_effort_levels``,
  and ``validate_search_depth`` also check project settings.
- **Required models**: ``required_models`` lists what the defaults use, so
  the first-run check asks only for those.

Module Structure
----------------

- ``DEFAULTS_FILE``, ``defaults_file``, ``built_in_defaults``, and
  ``write_defaults``
- ``default_effort_levels``, ``default_search_depth``,
  ``default_router_model``, ``default_embedding_model``,
  ``default_query_format``, ``default_appearance``, and
  ``default_answer_settings``
- ``required_models``

"""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
from typing import Final

from huggingface_hub.utils import validate_repo_id

from qachat.runtime import BACKEND_MLX, BACKEND_TORCH

__all__ = [
    "APPEARANCE_CHOICES",
    "DEFAULTS_ENVIRONMENT",
    "DEFAULTS_FILE",
    "LEVEL_COUNT",
    "MAX_LEVEL_NAME",
    "MAX_SEARCH_DEPTH",
    "built_in_defaults",
    "default_answer_settings",
    "default_appearance",
    "default_effort_levels",
    "default_embedding_model",
    "default_query_format",
    "default_router_model",
    "default_search_depth",
    "defaults_file",
    "required_models",
    "validate_effort_levels",
    "validate_model_id",
    "validate_search_depth",
    "write_defaults",
]

# The defaults: edit this file to change what new projects start with.
DEFAULTS_FILE: Final[Path] = Path(__file__).with_name("defaults.json")
# Tests point QAChat at a copy, so they never change the real defaults.
DEFAULTS_ENVIRONMENT: Final[str] = "QACHAT_DEFAULTS_FILE"
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


# Validated contents per file, with the modification time they were read at.
_CACHE: dict[Path, tuple[float, dict[str, object]]] = {}


def defaults_file() -> Path:
    """Return the defaults file location.

    Returns
    -------
    pathlib.Path
        ``$QACHAT_DEFAULTS_FILE`` when set (tests use a copy), otherwise
        ``qachat/gui/defaults.json``.

    """

    override = os.environ.get(DEFAULTS_ENVIRONMENT)
    return Path(override) if override else DEFAULTS_FILE


def _checks() -> dict[str, Callable[[object], object]]:
    return {
        "appearance": _choice(APPEARANCE_CHOICES),
        "effort_levels": _per_backend(validate_effort_levels),
        "search_depth": validate_search_depth,
        "router_models": _per_backend(validate_model_id),
        "embedding_models": _per_backend(validate_model_id),
        "query_format": _choice(_QUERY_FORMATS),
        "answers": _validate_answers,
    }


def built_in_defaults(path: Path | None = None) -> dict[str, object]:
    """Read and validate the defaults.

    Parameters
    ----------
    path: pathlib.Path | None, optional
        Defaults file. Defaults to ``defaults_file()``.

    Returns
    -------
    dict[str, object]
        Every default setting. The file is read again when it changed, so
        edits apply to the next new project.

    Raises
    ------
    ValueError
        If the file is not valid JSON or a setting is missing or invalid;
        the message names the setting.

    """

    path = Path(path or defaults_file())
    modified = path.stat().st_mtime
    cached = _CACHE.get(path)
    if cached is not None and cached[0] == modified:
        return cached[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    defaults: dict[str, object] = {}
    for name, check in _checks().items():
        if name not in data:
            raise ValueError(f"{path} needs the setting {name!r}.")
        try:
            defaults[name] = check(data[name])
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path} setting {name!r}: {error}") from error
    _CACHE[path] = (modified, defaults)
    return defaults


def write_defaults(
    changes: dict[str, object], path: Path | None = None
) -> dict[str, object]:
    """Save app-wide choices into the defaults file.

    Parameters
    ----------
    changes: dict[str, object]
        Any of ``appearance``, ``query_format``, ``search_depth``, and
        ``effort_levels``, ``router_models``, or ``embedding_models`` as
        ``{backend: value}``.
    path: pathlib.Path | None, optional
        Defaults file. Defaults to ``defaults_file()``.

    Returns
    -------
    dict[str, object]
        The defaults now saved.

    Raises
    ------
    ValueError
        If a value is invalid; nothing is written then.

    """

    path = Path(path or defaults_file())
    saved = json.loads(json.dumps(built_in_defaults(path)))
    checks = _checks()
    for key, value in changes.items():
        if key in ("effort_levels", "router_models", "embedding_models"):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must map backends to values.")
            for backend, entry in value.items():
                if backend not in _BACKENDS:
                    raise ValueError(f"Unknown backend {backend!r}.")
                check = {
                    "effort_levels": validate_effort_levels,
                }.get(key, validate_model_id)
                saved[key][backend] = check(entry)
        elif key in ("appearance", "query_format", "search_depth"):
            try:
                saved[key] = checks[key](value)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{key}: {error}") from error
        else:
            raise ValueError(f"Unknown default setting {key!r}.")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(_format(saved) + "\n", encoding="utf-8")
    temporary.replace(path)
    _CACHE.pop(path, None)
    return built_in_defaults(path)


def _format(value: object, indent: int = 0) -> str:
    """JSON with short lists and objects on one line, as the file is kept."""

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


def default_query_format() -> str:
    """Return the default embedding query format."""

    return str(built_in_defaults()["query_format"])


def default_appearance() -> str:
    """Return the appearance: system, light, or dark."""

    return str(built_in_defaults()["appearance"])


def default_answer_settings() -> dict[str, object]:
    """Return the default temperature, thinking, and longest answer."""

    return dict(built_in_defaults()["answers"])


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
