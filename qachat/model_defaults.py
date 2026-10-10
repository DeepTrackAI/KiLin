"""Define shared model purposes and default model identifiers.

This module owns the small set of model-related values shared by QAChat core
configuration and interface layers. The constants describe default starting
models and model purposes; they are not an allowlist of compatible models, a
cache inventory, or a runtime compatibility check.

Keeping these defaults outside the GUI prevents project configuration and
interface presets from drifting while preserving the dependency direction from
core QAChat functionality to optional interfaces. Default model IDs are read
from the packaged ``qachat_defaults.json`` when this module is imported.

Key Features
------------

- **Model-purpose vocabulary**

    Provide canonical identifiers for chat and embedding model purposes.

- **Shared default models**

    Define the four model IDs used by new project profiles and GUI presets.

Module Structure
----------------

- ``MODEL_ROLE_CHAT`` and ``MODEL_ROLE_EMBEDDING``

    Canonical string values for the two model purposes used by QAChat.

- ``MODEL_ROLE_CHOICES``

    Ordered collection of the supported model-purpose values.

- ``DEFAULT_CHAT_MODEL_MLX`` and ``DEFAULT_CHAT_MODEL_TORCH``

    Default chat-model IDs for the two calculation backends.

- ``DEFAULT_EMBEDDING_MODEL_MLX`` and ``DEFAULT_EMBEDDING_MODEL_TORCH``

    Default embedding-model IDs for the two calculation backends.

Examples
--------
>>> import qachat
>>> model_id = qachat.DEFAULT_CHAT_MODEL_MLX
>>> role = qachat.MODEL_ROLE_CHAT

Notes
-----
These identifiers are defaults only. Callers and project profiles may continue
to provide other compatible Hugging Face model IDs.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal, TypeAlias

__all__ = [
    "MODEL_ROLE_CHAT",
    "MODEL_ROLE_EMBEDDING",
    "MODEL_ROLE_CHOICES",
    "ModelRole",
    "DEFAULT_CHAT_MODEL_MLX",
    "DEFAULT_CHAT_MODEL_TORCH",
    "DEFAULT_EMBEDDING_MODEL_MLX",
    "DEFAULT_EMBEDDING_MODEL_TORCH",
]

ModelRole: TypeAlias = Literal["chat", "embedding"]

MODEL_ROLE_CHAT: Final[ModelRole] = "chat"
MODEL_ROLE_EMBEDDING: Final[ModelRole] = "embedding"
MODEL_ROLE_CHOICES: Final[tuple[ModelRole, ...]] = (
    MODEL_ROLE_CHAT,
    MODEL_ROLE_EMBEDDING,
)

# Load packaged model identifiers once at import time.
_MODEL_DEFAULTS: dict[str, str] = json.loads(
    Path(__file__).with_name("qachat_defaults.json").read_text(
        encoding="utf-8"
    )
)["DEFAULT_MODELS"]

DEFAULT_CHAT_MODEL_MLX: Final[str] = _MODEL_DEFAULTS["DEFAULT_CHAT_MODEL_MLX"]
DEFAULT_CHAT_MODEL_TORCH: Final[str] = _MODEL_DEFAULTS[
    "DEFAULT_CHAT_MODEL_TORCH"
]
DEFAULT_EMBEDDING_MODEL_MLX: Final[str] = _MODEL_DEFAULTS[
    "DEFAULT_EMBEDDING_MODEL_MLX"
]
DEFAULT_EMBEDDING_MODEL_TORCH: Final[str] = _MODEL_DEFAULTS[
    "DEFAULT_EMBEDDING_MODEL_TORCH"
]
