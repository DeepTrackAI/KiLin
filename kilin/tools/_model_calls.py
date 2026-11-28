"""Resolve tool-owned thinking policies for internal model requests.

Tools inherit the answering session unless their own ``defaults.json``
selects another policy. Machine-readable requests can opt out of thinking
explicitly, without classifying a request by its token budget.
This policy never controls router calls or user-visible ``write`` calls.

Key Features
------------

- **Tool ownership**

    Validate an explicit ``MODEL_CALLS`` section in each tool's defaults.

- **Reversible defaults**

    Select ``inherit``, ``on`` or ``off`` independently of the interface.

Module Structure
----------------

- ``ThinkingPolicy``

    The accepted inherit, on and off policy vocabulary.

- ``ModelCallSettings``

    Immutable, validated policy and thinking resolution.

- ``load_model_call_settings``

    Read and cache one tool's policy from its packaged configuration.

Examples
--------
>>> from kilin.tools._model_calls import ModelCallSettings
>>> ModelCallSettings().thinking_override() is None
True
>>> ModelCallSettings(thinking="off").thinking_override()
False

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypeAlias

from kilin.tools._defaults import (
    load_tool_defaults,
    require_keys,
    require_section,
    require_string,
)

__all__ = [
    "ModelCallSettings",
    "ThinkingPolicy",
    "load_model_call_settings",
    "load_optional_model_call_settings",
]

ThinkingPolicy: TypeAlias = Literal["inherit", "on", "off"]


@dataclass(frozen=True, slots=True)
class ModelCallSettings:
    """Control thinking for a tool's internal answering-model requests.

    Attributes
    ----------
    thinking: ThinkingPolicy
        ``inherit`` follows the active session. ``on`` or ``off`` overrides
        internal requests explicitly, independently of their token budgets.
        Defaults to ``inherit``. User-visible ``write`` calls are unaffected.

    """

    thinking: ThinkingPolicy = "inherit"

    def __post_init__(self) -> None:
        """Reject unsupported thinking policies.

        Raises
        ------
        ValueError
            If the policy is not ``inherit``, ``on`` or ``off``.

        """

        if not isinstance(self.thinking, str) or self.thinking not in {
            "inherit", "on", "off"
        }:
            raise ValueError("thinking must be 'inherit', 'on' or 'off'.")

    def thinking_override(self) -> bool | None:
        """Resolve the policy without inspecting the output budget.

        Returns
        -------
        bool | None
            Forced thinking state, or ``None`` to inherit the session.

        """

        return None if self.thinking == "inherit" else self.thinking == "on"


@lru_cache(maxsize=None)
def load_model_call_settings(path: Path) -> ModelCallSettings:
    """Load a tool's internal generation policy without repeated I/O.

    Parameters
    ----------
    path: pathlib.Path
        Path to the tool-owned ``defaults.json``.

    Returns
    -------
    ModelCallSettings
        Validated immutable policy. Restart the application or notebook
        kernel after changing packaged defaults.

    Raises
    ------
    ValueError
        If the section is missing or contains unknown keys or values.
    OSError
        If the configuration cannot be read.

    """

    where = f"{path.parent.name}.MODEL_CALLS"
    data = require_section(load_tool_defaults(path), "MODEL_CALLS", where)
    require_keys(data, {"THINKING"}, where)
    return ModelCallSettings(thinking=require_string(data, "THINKING", where))


@lru_cache(maxsize=None)
def load_optional_model_call_settings(path: Path) -> ModelCallSettings:
    """Resolve a tool policy, inheriting when no policy is declared.

    Parameters
    ----------
    path: pathlib.Path
        Path to the tool-owned ``defaults.json``.

    Returns
    -------
    ModelCallSettings
        Validated declared policy, or the session-inheritance policy.

    Raises
    ------
    ValueError
        If a supplied defaults document or policy is invalid.
    OSError
        If a supplied defaults file cannot be read.

    """

    if not path.exists():
        return ModelCallSettings()
    if "MODEL_CALLS" not in load_tool_defaults(path):
        return ModelCallSettings()
    return load_model_call_settings(path)
