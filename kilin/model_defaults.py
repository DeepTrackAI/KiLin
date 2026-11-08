"""Define shared model purposes and default model identifiers.

This module owns the small set of model-related values shared by KíLín core
configuration and interface layers. The constants describe default starting
models and model purposes; they are not an allowlist of compatible models, a
cache inventory, or a runtime compatibility check.

Keeping these defaults outside the GUI prevents project configuration and
interface presets from drifting while preserving the dependency direction from
core KíLín functionality to optional interfaces. Default model IDs are read
from the packaged ``kilin_defaults.json`` when this module is imported.
The shared reader validates their strings; the backend is not selected here.

Key Features
------------

- **Model-purpose vocabulary**

    Provide canonical identifiers for chat and embedding model purposes.

- **Shared default models**

    Define default chat, router, and embedding models for both backends.

Module Structure
----------------

- ``MODEL_ROLE_CHAT`` and ``MODEL_ROLE_EMBEDDING``

    Canonical string values for the two model purposes used by KíLín.

- ``MODEL_ROLE_CHOICES``

    Ordered collection of the supported model-purpose values.

- ``DEFAULT_CHAT_MODEL_MLX`` and ``DEFAULT_CHAT_MODEL_TORCH``

    Default chat-model IDs for the two calculation backends.

- ``DEFAULT_ROUTER_MODEL_MLX`` and ``DEFAULT_ROUTER_MODEL_TORCH``

    Default independent router-model IDs for the two backends.

- ``DEFAULT_EMBEDDING_MODEL_MLX`` and ``DEFAULT_EMBEDDING_MODEL_TORCH``

    Default embedding-model IDs for the two calculation backends.

Examples
--------
>>> import kilin
>>> model_id = kilin.DEFAULT_CHAT_MODEL_MLX
>>> role = kilin.MODEL_ROLE_CHAT

Notes
-----
These identifiers are defaults only. Callers and project profiles may continue
to provide other compatible Hugging Face model IDs.

"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

from kilin.defaults import (
    load_defaults_section,
    require_default_string,
)

__all__ = [
    "MODEL_ROLE_CHAT",
    "MODEL_ROLE_EMBEDDING",
    "MODEL_ROLE_CHOICES",
    "ModelRole",
    "DEFAULT_CHAT_MODEL_MLX",
    "DEFAULT_CHAT_MODEL_TORCH",
    "DEFAULT_EMBEDDING_MODEL_MLX",
    "DEFAULT_EMBEDDING_MODEL_TORCH",
    "DEFAULT_ROUTER_MODEL_MLX",
    "DEFAULT_ROUTER_MODEL_TORCH",
]

ModelRole: TypeAlias = Literal["chat", "embedding"]

MODEL_ROLE_CHAT: Final[ModelRole] = "chat"
MODEL_ROLE_EMBEDDING: Final[ModelRole] = "embedding"
MODEL_ROLE_CHOICES: Final[tuple[ModelRole, ...]] = (
    MODEL_ROLE_CHAT,
    MODEL_ROLE_EMBEDDING,
)

# The shared reader caches the packaged JSON and validates this section.
_MODEL_DEFAULTS = load_defaults_section("DEFAULT_MODELS")

DEFAULT_CHAT_MODEL_MLX: Final[str] = require_default_string(
    _MODEL_DEFAULTS, "DEFAULT_CHAT_MODEL_MLX", "DEFAULT_MODELS"
)
DEFAULT_CHAT_MODEL_TORCH: Final[str] = require_default_string(
    _MODEL_DEFAULTS, "DEFAULT_CHAT_MODEL_TORCH", "DEFAULT_MODELS"
)
DEFAULT_ROUTER_MODEL_MLX: Final[str] = require_default_string(
    _MODEL_DEFAULTS, "DEFAULT_ROUTER_MODEL_MLX", "DEFAULT_MODELS"
)
DEFAULT_ROUTER_MODEL_TORCH: Final[str] = require_default_string(
    _MODEL_DEFAULTS, "DEFAULT_ROUTER_MODEL_TORCH", "DEFAULT_MODELS"
)
DEFAULT_EMBEDDING_MODEL_MLX: Final[str] = require_default_string(
    _MODEL_DEFAULTS, "DEFAULT_EMBEDDING_MODEL_MLX", "DEFAULT_MODELS"
)
DEFAULT_EMBEDDING_MODEL_TORCH: Final[str] = require_default_string(
    _MODEL_DEFAULTS, "DEFAULT_EMBEDDING_MODEL_TORCH", "DEFAULT_MODELS"
)
