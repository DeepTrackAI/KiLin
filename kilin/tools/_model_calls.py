"""Resolve tool-owned thinking policies for internal model requests.

Tools inherit the answering session unless their own ``defaults.json``
selects another policy. Short, bounded machine-readable requests can opt
out of thinking without changing the model, temperature or token budget.
This policy never controls router calls or user-visible ``write`` calls.

Key Features
------------

- **Tool ownership**

    Validate an explicit ``MODEL_CALLS`` section in each tool's defaults.

- **Reversible defaults**

    Select ``inherit``, ``on`` or ``off`` independently of the interface.

- **Bounded requests**

    Optionally override short replies by their requested output budget,
    not by the length or wording of the user's question.

Module Structure
----------------

- ``ThinkingPolicy``

    The accepted inherit, on and off policy vocabulary.

- ``ModelCallSettings``

    Immutable, validated policy and per-request thinking resolution.

- ``load_model_call_settings``

    Read and cache one tool's policy from its packaged configuration.

Examples
--------
>>> from kilin.tools._model_calls import ModelCallSettings
>>> ModelCallSettings().thinking_override(120) is None
True
>>> policy = ModelCallSettings(
...     short_reply_thinking="off", short_reply_max_tokens=512
... )
>>> policy.thinking_override(120)
False

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypeAlias

from kilin.tools._defaults import (
    load_tool_defaults,
    require_int,
    require_keys,
    require_section,
    require_string,
)

__all__ = [
    "ModelCallSettings",
    "ThinkingPolicy",
    "load_model_call_settings",
]

ThinkingPolicy: TypeAlias = Literal["inherit", "on", "off"]


@dataclass(frozen=True, slots=True)
class ModelCallSettings:
    """Control thinking for a tool's internal answering-model requests.

    Attributes
    ----------
    thinking: ThinkingPolicy
        ``inherit`` follows the session, subject to the short-reply rule.
        ``on`` or ``off`` overrides every internal request and takes
        precedence over that rule. Defaults to ``inherit``.
    short_reply_thinking: ThinkingPolicy
        Policy for requests within the short-reply bound. ``inherit``
        disables the override. Defaults to ``inherit``.
    short_reply_max_tokens: int
        Inclusive bound on an explicitly requested output-token limit.
        Zero disables the short-reply rule. Requests with no explicit
        limit are never classified as short. Defaults to zero.

    """

    thinking: ThinkingPolicy = "inherit"
    short_reply_thinking: ThinkingPolicy = "inherit"
    short_reply_max_tokens: int = 0

    def __post_init__(self) -> None:
        """Reject unsupported policies and ambiguous numeric bounds.

        Raises
        ------
        ValueError
            If a policy is unknown or the bound is not an integer in
            ``[0, 100000]``. Booleans are not accepted as integers.

        """

        for name in ("thinking", "short_reply_thinking"):
            value = getattr(self, name)
            if not isinstance(value, str) or value not in {
                "inherit", "on", "off"
            }:
                raise ValueError(
                    f"{name} must be 'inherit', 'on' or 'off'."
                )
        if (
            type(self.short_reply_max_tokens) is not int
            or not 0 <= self.short_reply_max_tokens <= 100000
        ):
            raise ValueError(
                "short_reply_max_tokens must be an integer from 0 to 100000."
            )

    def thinking_override(self, tokens: int | None) -> bool | None:
        """Resolve an internal request without changing session settings.

        Parameters
        ----------
        tokens: int | None
            Requested output budget, before the session ceiling is applied.
            ``None`` means the caller inherits the session token limit.

        Returns
        -------
        bool | None
            Forced thinking state, or ``None`` to inherit the session.

        Raises
        ------
        ValueError
            If an explicit token budget is not a positive integer.

        """

        if tokens is not None and (type(tokens) is not int or tokens < 1):
            raise ValueError("tokens must be a positive integer or None.")
        policy = self.thinking
        if (
            policy == "inherit" and tokens is not None
            and tokens <= self.short_reply_max_tokens
        ):
            policy = self.short_reply_thinking
        return None if policy == "inherit" else policy == "on"


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
    require_keys(data, {
        "THINKING", "SHORT_REPLY_THINKING", "SHORT_REPLY_MAX_TOKENS"
    }, where)
    return ModelCallSettings(
        thinking=require_string(data, "THINKING", where),
        short_reply_thinking=require_string(
            data, "SHORT_REPLY_THINKING", where
        ),
        short_reply_max_tokens=require_int(
            data, "SHORT_REPLY_MAX_TOKENS", where, 0, 100000
        ),
    )
