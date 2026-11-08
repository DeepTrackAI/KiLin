"""Run source-card generation with an independent, project-selected model.

Source-card generation is an indexing activity, not a conversation turn. The
model, thinking mode, temperature, and output cap are loaded from core-owned
project settings, independently of both the answering and routing sessions.
A generator loads its model only for the first requested card, reuses the
session for subsequent cards, and releases it at the end of the batch.

This module does not choose source kinds, compose scientific-paper prompts,
or determine which cards need rebuilding. The card-kind implementation will
provide the prompts in PR6; the indexing coordinator will call this adapter
only for pending sources in PR7. No model is loaded during index searching.

Key Features
------------

- **Lazy, isolated inference**

    Open a separate chat session only when a generation call is made.

- **Complete responses**

    Reject incomplete replies, reusing the established side-reply checks.

- **Explicit lifecycle**

    A context manager releases the loaded model even if generation fails.

Module Structure
----------------

- ``SourceCardGenerator``

    Run one or several prompted generations with validated project settings.

Examples
--------
>>> from kilin.model_settings import default_source_card_generation_settings
>>> from kilin.retrievals.source_cards.generation import SourceCardGenerator
>>> settings = default_source_card_generation_settings("torch")
>>> generator = SourceCardGenerator(settings, backend="torch",
...                                 allow_download=False)
>>> generator.close()  # No model was loaded.

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from kilin.answering import side_reply
from kilin.chat import ChatSession
from kilin.model_settings import SourceCardGenerationSettings
from kilin.runtime import BACKEND_MLX, BACKEND_TORCH
from kilin.trace_logging import TraceSink

__all__ = ["SourceCardGenerator"]


class SourceCardGenerator:
    """Manage an isolated model session for a source-card generation batch.

    Parameters
    ----------
    settings: SourceCardGenerationSettings
        Validated model and generation settings from the KíLín core.
    backend: str
        Selected backend, ``mlx`` or ``torch``; ``auto`` is not accepted.
    allow_download: bool
        Project's existing model-download policy. No separate setting may
        override a project-wide restriction on downloading models.
    trace: TraceSink | None, optional
        Optional caller-owned trace sink. Defaults to ``None``.
    session_factory: Callable[..., ChatSession] | None, optional
        Optional session construction adapter for deterministic tests. The
        default is ``ChatSession``. Defaults to ``None``.

    Notes
    -----
    Construction and entering the context do not load a model. The first
    ``generate`` call loads a dedicated session with an empty history.
    Consecutive calls reuse it, but every request's conversation is reset
    by ``side_reply``. Exiting the context releases the session, even if a
    generation raises. The generator cannot be reused after ``close``.

    The input-token budget is metadata for passage selection in PR6, not
    a tokenizer-based check in this adapter. The output limit is enforced
    through ``ChatSession`` and completion validation.

    """

    def __init__(
        self,
        settings: SourceCardGenerationSettings,
        *,
        backend: str,
        allow_download: bool,
        trace: TraceSink | None = None,
        session_factory: Callable[..., ChatSession] | None = None,
    ) -> None:
        """Validate settings without loading a language model.

        Parameters
        ----------
        settings: SourceCardGenerationSettings
            Configured model and inference parameters.
        backend: str
            Explicit selected calculation backend.
        allow_download: bool
            Effective project download policy.
        trace: TraceSink | None, optional
            Caller-owned trace sink. Defaults to ``None``.
        session_factory: Callable[..., ChatSession] | None, optional
            Model-session adapter for tests. Defaults to ``None``.

        Raises
        ------
        ValueError
            If configuration, backend, or download policy is invalid.

        """

        if not isinstance(settings, SourceCardGenerationSettings):
            raise ValueError("Expected SourceCardGenerationSettings.")
        settings.__post_init__()
        if backend not in (BACKEND_MLX, BACKEND_TORCH):
            raise ValueError("Source-card backend must be 'mlx' or 'torch'.")
        if type(allow_download) is not bool:
            raise ValueError("allow_download must be a boolean.")
        if session_factory is not None and not callable(session_factory):
            raise ValueError("session_factory must be callable.")
        self.settings = settings
        self.backend = backend
        self.allow_download = allow_download
        self.trace = trace
        self._session_factory = (
            ChatSession if session_factory is None else session_factory
        )
        self._session: ChatSession | None = None
        self._closed = False

    def __enter__(self) -> Self:
        """Enter a generation batch without loading its model.

        Returns
        -------
        SourceCardGenerator
            This generator, for use within a context manager.

        Raises
        ------
        RuntimeError
            If the generator has already been closed.

        """

        if self._closed:
            raise RuntimeError("Source-card generator is closed.")
        return self

    def __exit__(
        self, exception_type: object, exception: object, traceback: object
    ) -> None:
        """Release the generation model on normal or exceptional exit.

        Parameters
        ----------
        exception_type: object
            Context-manager exception type, if any.
        exception: object
            Context-manager exception value, if any.
        traceback: object
            Context-manager traceback, if any.

        """

        self.close()

    def generate(self, system: str, prompt: str) -> str:
        """Generate one complete card response without conversation state.

        Parameters
        ----------
        system: str
            Card-kind-owned system instructions.
        prompt: str
            Card-kind-owned prompt containing selected source passages.

        Returns
        -------
        str
            Complete model reply, with thinking content removed.

        Raises
        ------
        ValueError
            If either prompt is empty.
        RuntimeError
            If the generator was already closed.
        IncompleteModelResponse
            If output is empty, incomplete, or truncated.

        Notes
        -----
        Each model call has its own temporary system/user messages.
        Original source data must be treated as untrusted prompt content by
        the card-kind prompt builder in PR6.

        """

        if self._closed:
            raise RuntimeError("Source-card generator is closed.")
        if not isinstance(system, str) or not system.strip():
            raise ValueError("Source-card system prompt must be nonempty.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Source-card input prompt must be nonempty.")
        if self._session is None:
            self._session = self._session_factory(
                self.settings.model_id,
                backend=self.backend,
                thinking=self.settings.thinking,
                temperature=self.settings.temperature,
                max_new_tokens=self.settings.max_new_tokens,
                allow_download=self.allow_download,
                trace=self.trace,
                block_unrequested_cjk=False,
            )
        return side_reply(
            self._session,
            system,
            prompt,
            max_new_tokens=self.settings.max_new_tokens,
            inherit_settings=True,
            require_complete=True,
        )

    def close(self) -> None:
        """Close the loaded model and prevent further generation requests.

        Notes
        -----
        Calling close before any generation performs no model operations.
        Closing more than once is safe. The caller owns any trace sink.

        """

        self._closed = True
        session = self._session
        self._session = None
        if session is not None:
            session.close()
