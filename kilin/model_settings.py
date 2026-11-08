"""Resolve and persist auxiliary model settings in the KíLín core.

Auxiliary models are configured per project and backend in ``profile.ini``.
The router model and source-card generation model have distinct settings.
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

- **Source-card generation configuration**

    Read and write a dedicated generation model and bounded parameters.
    This is not affected by interactive answering or router settings.

Module Structure
----------------

- ``validate_model_id``

    Validate a model identifier without loading any model.

- ``default_router_model``

    Return the packaged default router for a selected backend.

- ``read_router_model`` and ``write_router_model``

    Read and update persistent project router settings.

- ``SourceCardGenerationSettings``

    Validated, immutable settings for dedicated card inference.

- ``read_source_card_generation_settings`` and
  ``write_source_card_generation_settings``

    Resolve and persist independent per-project settings.

Examples
--------
>>> from kilin.model_settings import default_router_model
>>> bool(default_router_model("mlx"))
True

"""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
import json
from math import isfinite
from pathlib import Path
from typing import Final

from huggingface_hub.utils import validate_repo_id

from kilin.atomic_writes import atomic_write_text
from kilin.defaults import (
    load_defaults_section,
    require_default_bool,
    require_default_int,
    require_default_number,
)
from kilin.model_defaults import (
    DEFAULT_ROUTER_MODEL_MLX,
    DEFAULT_ROUTER_MODEL_TORCH,
    DEFAULT_SOURCE_CARD_MODEL_MLX,
    DEFAULT_SOURCE_CARD_MODEL_TORCH,
)
from kilin.runtime import BACKEND_MLX, BACKEND_TORCH

__all__ = [
    "SourceCardGenerationSettings",
    "default_router_model",
    "default_source_card_generation_settings",
    "read_source_card_generation_settings",
    "write_source_card_generation_settings",
    "read_router_model",
    "validate_model_id",
    "write_router_model",
]

_ROUTER_SECTION: Final[str] = "router"
_PROFILE_FILE: Final[str] = "profile.ini"
_LEGACY_FILE: Final[str] = "effort_levels.json"
_SOURCE_CARD_SECTION: Final[str] = "source_card_generation"
_MAX_SOURCE_CARD_TOKENS: Final[int] = 200_000
_SOURCE_CARD_DEFAULTS = load_defaults_section("SOURCE_CARD_GENERATION")
_DEFAULT_CARD_THINKING: Final[bool] = require_default_bool(
    _SOURCE_CARD_DEFAULTS, "THINKING", "SOURCE_CARD_GENERATION"
)
_DEFAULT_CARD_TEMPERATURE: Final[float] = require_default_number(
    _SOURCE_CARD_DEFAULTS, "TEMPERATURE", "SOURCE_CARD_GENERATION",
    allow_zero=True,
)
_DEFAULT_CARD_INPUT: Final[int] = require_default_int(
    _SOURCE_CARD_DEFAULTS, "MAX_INPUT_TOKENS", "SOURCE_CARD_GENERATION"
)
_DEFAULT_CARD_OUTPUT: Final[int] = require_default_int(
    _SOURCE_CARD_DEFAULTS, "MAX_NEW_TOKENS", "SOURCE_CARD_GENERATION"
)


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


@dataclass(frozen=True, slots=True)
class SourceCardGenerationSettings:
    """Independent model and inference parameters for source-card creation.

    Parameters
    ----------
    model_id: str
        Model for the selected calculation backend.
    thinking: bool
        Whether a compatible template enables thinking.
    temperature: float
        Finite, nonnegative sampling temperature; zero is greedy.
    max_input_tokens: int
        Maximum estimated card-generation input budget. Passage selection
        will enforce this value when card kinds are implemented in PR6.
    max_new_tokens: int
        Strict positive limit on generated tokens for each card.

    Notes
    -----
    These are indexing settings, not answering-session settings. Changing
    them must not mutate router or answering configurations. Valid cards
    can be reused across changes to this model or its parameters.

    """

    model_id: str
    thinking: bool
    temperature: float
    max_input_tokens: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        """Reject invalid model identifiers or generation parameters.

        Raises
        ------
        ValueError
            If a configured value violates the indexing contract.

        """

        validated_model = validate_model_id(self.model_id)
        object.__setattr__(self, "model_id", validated_model)
        if type(self.thinking) is not bool:
            raise ValueError("Source-card thinking must be a boolean.")
        if type(self.temperature) not in (int, float):
            raise ValueError("Source-card temperature must be numeric.")
        try:
            temperature = float(self.temperature)
        except OverflowError as error:
            raise ValueError("Source-card temperature must be finite.") \
                from error
        if not isfinite(temperature) or temperature < 0:
            raise ValueError(
                "Source-card temperature must be finite and nonnegative."
            )
        object.__setattr__(self, "temperature", temperature)
        for name in ("max_input_tokens", "max_new_tokens"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= (
                _MAX_SOURCE_CARD_TOKENS
            ):
                raise ValueError(
                    f"Source-card {name} must be between 1 and "
                    f"{_MAX_SOURCE_CARD_TOKENS}."
                )

    def fingerprint(self) -> str:
        """Fingerprint inference parameters independently of model identity.

        Returns
        -------
        str
            Stable SHA-256 for generation settings and prompt budgets.

        Notes
        -----
        The separate model ID is supplied to source-card cache planning.
        This fingerprint allows failed attempts to be retried if generation
        settings change without rebuilding otherwise valid cards.

        """

        data = {
            "thinking": self.thinking,
            "temperature": float(self.temperature),
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens,
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return sha256(serialized.encode("utf-8")).hexdigest()


def default_source_card_generation_settings(
    backend: str,
) -> SourceCardGenerationSettings:
    """Return the independent packaged source-card generation defaults.

    Parameters
    ----------
    backend: str
        Selected backend, either ``mlx`` or ``torch``.

    Returns
    -------
    SourceCardGenerationSettings
        Qwen3-8B model selection and the packaged inference parameters.

    Raises
    ------
    ValueError
        If the backend is not explicitly selected.

    """

    if backend == BACKEND_MLX:
        model_id = DEFAULT_SOURCE_CARD_MODEL_MLX
    elif backend == BACKEND_TORCH:
        model_id = DEFAULT_SOURCE_CARD_MODEL_TORCH
    else:
        raise ValueError("Source-card backend must be 'mlx' or 'torch'.")
    return SourceCardGenerationSettings(
        model_id=model_id, thinking=_DEFAULT_CARD_THINKING,
        temperature=_DEFAULT_CARD_TEMPERATURE,
        max_input_tokens=_DEFAULT_CARD_INPUT,
        max_new_tokens=_DEFAULT_CARD_OUTPUT,
    )


def read_source_card_generation_settings(
    root: Path | None, backend: str,
) -> SourceCardGenerationSettings:
    """Resolve source-card generation settings without modifying the project.

    Parameters
    ----------
    root: pathlib.Path | None
        Project directory, or ``None`` for packaged defaults.
    backend: str
        Explicitly selected calculation backend.

    Returns
    -------
    SourceCardGenerationSettings
        Validated project overrides with packaged defaults for absent keys.

    Raises
    ------
    ValueError
        If saved settings or the selected backend are invalid.

    Notes
    -----
    Reading this section neither loads a model nor changes profile.ini.
    Source-card values never inherit the router or answering settings.

    """

    defaults = default_source_card_generation_settings(backend)
    if root is None:
        return defaults
    profile = ConfigParser(interpolation=None)
    profile.read(Path(root) / _PROFILE_FILE, encoding="utf-8")
    section = _SOURCE_CARD_SECTION
    if not profile.has_section(section):
        return defaults
    model_id = profile.get(
        section, backend, fallback=defaults.model_id
    )
    thinking = profile.getboolean(
        section, "thinking", fallback=defaults.thinking
    )
    temperature = profile.getfloat(
        section, "temperature", fallback=defaults.temperature
    )
    max_input_tokens = profile.getint(
        section, "max_input_tokens", fallback=defaults.max_input_tokens
    )
    max_new_tokens = profile.getint(
        section, "max_new_tokens", fallback=defaults.max_new_tokens
    )
    return SourceCardGenerationSettings(
        model_id=validate_model_id(model_id), thinking=thinking,
        temperature=temperature, max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
    )


def write_source_card_generation_settings(
    root: Path, backend: str, settings: SourceCardGenerationSettings,
) -> None:
    """Atomically save independent source-card generation settings.

    Parameters
    ----------
    root: pathlib.Path
        Existing or new project directory.
    backend: str
        Calculation backend whose model identifier is updated.
    settings: SourceCardGenerationSettings
        Selected model and shared generation parameters to persist.

    Raises
    ------
    ValueError
        If any supplied setting or backend is invalid.

    Notes
    -----
    The other backend's model remains unchanged, as do router, models,
    and unrelated profile sections. No model is loaded when saving.

    """

    default_source_card_generation_settings(backend)
    if not isinstance(settings, SourceCardGenerationSettings):
        raise ValueError("Expected SourceCardGenerationSettings.")
    # Validate even if a caller circumvented the immutable dataclass.
    settings.__post_init__()
    profile = ConfigParser(interpolation=None)
    path = Path(root) / _PROFILE_FILE
    profile.read(path, encoding="utf-8")
    if not profile.has_section(_SOURCE_CARD_SECTION):
        profile.add_section(_SOURCE_CARD_SECTION)
    options = {
        backend: validate_model_id(settings.model_id),
        "thinking": str(settings.thinking).lower(),
        "temperature": str(float(settings.temperature)),
        "max_input_tokens": str(settings.max_input_tokens),
        "max_new_tokens": str(settings.max_new_tokens),
    }
    for key, setting in options.items():
        profile.set(_SOURCE_CARD_SECTION, key, setting)
    stream = StringIO()
    profile.write(stream)
    atomic_write_text(path, stream.getvalue())
