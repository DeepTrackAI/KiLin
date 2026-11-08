"""Resolve and persist auxiliary model settings in the KíLín core.

The router model is configured per project and backend in ``profile.ini``.
The core reads this selection without importing the GUI, so Python clients
and desktop sessions use the same router. An explicit session-level model
selection remains an override, not a persistent configuration change.

The GUI may use a different application-wide preference when first saving
settings for a project. Once saved, the project value is authoritative.
Older ``effort_levels.json`` router selections are read without mutating the
workspace; the GUI's existing migration workflow owns their removal.

Key Features
------------

- **Shared model validation**

    Validate repository IDs or existing local model directories.

- **Core router configuration**

    Read and atomically write per-backend router selections in ``profile.ini``.

Module Structure
----------------

- ``validate_model_id``

    Validate a model identifier without loading any model.

- ``default_router_model``

    Return the packaged default router for a selected backend.

- ``read_router_model`` and ``write_router_model``

    Read and update persistent project router settings.

Examples
--------
>>> from kilin.model_settings import default_router_model
>>> bool(default_router_model("mlx"))
True

"""

from __future__ import annotations

from configparser import ConfigParser
from io import StringIO
import json
from pathlib import Path
from typing import Final

from huggingface_hub.utils import validate_repo_id

from kilin.atomic_writes import atomic_write_text
from kilin.model_defaults import (
    DEFAULT_ROUTER_MODEL_MLX,
    DEFAULT_ROUTER_MODEL_TORCH,
)
from kilin.runtime import BACKEND_MLX, BACKEND_TORCH

__all__ = [
    "default_router_model",
    "read_router_model",
    "validate_model_id",
    "write_router_model",
]

_ROUTER_SECTION: Final[str] = "router"
_PROFILE_FILE: Final[str] = "profile.ini"
_LEGACY_FILE: Final[str] = "effort_levels.json"


def validate_model_id(value: object) -> str:
    """Validate a Hugging Face model ID or an existing local model folder.

    Parameters
    ----------
    value: object
        Model identifier or existing local directory path.

    Returns
    -------
    str
        Identifier without surrounding whitespace.

    Raises
    ------
    ValueError
        If the identifier is empty or is not a valid repository ID.

    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Enter a Hugging Face model ID.")
    model_id = value.strip()
    if not Path(model_id).expanduser().is_dir():
        validate_repo_id(model_id)
    return model_id


def default_router_model(backend: str) -> str:
    """Return the core default router model for a selected backend.

    Parameters
    ----------
    backend: str
        ``"mlx"`` or ``"torch"``; ``"auto"`` is not resolved here.

    Returns
    -------
    str
        The packaged default model identifier for the backend.

    Raises
    ------
    ValueError
        If the backend is not a selected backend.

    """

    if backend == BACKEND_MLX:
        return DEFAULT_ROUTER_MODEL_MLX
    if backend == BACKEND_TORCH:
        return DEFAULT_ROUTER_MODEL_TORCH
    raise ValueError("Router backend must be 'mlx' or 'torch'.")


def read_router_model(
    root: Path | None,
    backend: str,
    *,
    default_model: str | None = None,
) -> str:
    """Read the router model from project settings without changing files.

    Parameters
    ----------
    root: pathlib.Path | None
        Project directory, or ``None`` for a project not yet created.
    backend: str
        Selected calculation backend, ``"mlx"`` or ``"torch"``.
    default_model: str | None, optional
        Initial choice when neither current nor legacy settings specify a
        router. GUI project setup may provide its user preference; core
        callers default to the packaged model. Defaults to ``None``.

    Returns
    -------
    str
        Validated saved router, legacy router, or default in that order.

    Raises
    ------
    ValueError
        If the backend or a present model identifier is invalid.
    json.JSONDecodeError
        If the legacy router settings contain malformed JSON.

    Notes
    -----
    This reader does not migrate or write project files. Consequently a
    model selection made through the GUI is respected by a core-only
    session without first initializing a GUI engine.

    """

    default = default_router_model(backend)
    if root is not None:
        directory = Path(root)
        profile = ConfigParser(interpolation=None)
        profile.read(directory / _PROFILE_FILE, encoding="utf-8")
        if profile.has_option(_ROUTER_SECTION, backend):
            return validate_model_id(
                profile.get(_ROUTER_SECTION, backend)
            )
        legacy_path = directory / _LEGACY_FILE
        if legacy_path.is_file():
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
            if isinstance(legacy, dict):
                routers = legacy.get("router")
                if isinstance(routers, dict) and isinstance(
                    routers.get(backend), str
                ):
                    return validate_model_id(routers[backend])
    return (
        validate_model_id(default_model)
        if default_model is not None
        else default
    )


def write_router_model(root: Path, backend: str, model_id: str) -> None:
    """Atomically save a project's router model in ``profile.ini``.

    Parameters
    ----------
    root: pathlib.Path
        Project directory.
    backend: str
        Selected calculation backend, ``"mlx"`` or ``"torch"``.
    model_id: str
        Model ID to validate and save.

    Raises
    ------
    ValueError
        If the backend or model ID is invalid.

    Notes
    -----
    All other profile sections and options are retained. The model is not
    loaded, and an already running session is not reconfigured.

    """

    default_router_model(backend)
    model_id = validate_model_id(model_id)
    profile = ConfigParser(interpolation=None)
    path = Path(root) / _PROFILE_FILE
    profile.read(path, encoding="utf-8")
    if not profile.has_section(_ROUTER_SECTION):
        profile.add_section(_ROUTER_SECTION)
    profile.set(_ROUTER_SECTION, backend, model_id)
    stream = StringIO()
    profile.write(stream)
    atomic_write_text(path, stream.getvalue())
