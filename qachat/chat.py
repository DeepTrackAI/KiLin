"""Manage one local language-model chat session.

This module provides ``ChatSession``, the low-level QAChat abstraction for
loading one compatible language model and maintaining one in-memory
conversation. It resolves the calculation backend, loads the model and
tokenizer, formats chat messages, generates replies, records optional
execution traces, and releases model resources when the session is closed.

The implementation supports MLX on Apple Silicon and PyTorch on other
supported platforms. Models may be loaded from a local directory, from an
existing Hugging Face cache entry, or, when explicitly allowed, through the
normal Hugging Face download mechanism.

Conversation state is intentionally lightweight. Messages are stored as a
sequence of ``system``, ``user``, and ``assistant`` mappings. The model
remains loaded across replies, while generation-only settings can be changed
without reloading it.

Key Features
------------
- **Backend-aware model loading**

    Select MLX or PyTorch through the shared QAChat runtime utilities and load
    only the dependencies required by the selected backend.

- **Local-first model resolution**

    Accept local model directories directly and, when downloads are disabled,
    resolve Hugging Face model IDs through the existing shared cache.

- **In-memory conversation state**

    Keep one ordered chat history together with an optional system prompt and
    support resetting or restoring that history.

- **Configurable generation**

    Control thinking mode, sampling temperature, and reply length at session
    creation or update generation-only settings without reloading the model.

- **Context-aware reply limits**

    Use an explicit token limit when provided, otherwise derive the available
    generation budget from the model's reported context size.

- **Structured tracing**

    Emit model-loading, prompt, generation, timing, and optional content or
    token-ID trace data through a caller-provided ``TraceSink``.

- **Explicit resource release**

    Release model and tokenizer references, clear conversation history, run
    garbage collection, and clear backend caches when the session is closed.

Module Structure
----------------
Classes:

- ``ChatSession``

    Load one model, maintain one conversation, generate replies, and release
    the associated resources.

The remaining functions are private validation and bookkeeping helpers used
internally by ``ChatSession``.

Examples
--------
Create a session using the platform-default backend and send a message:

>>> chat = qachat.ChatSession("Qwen/Qwen3-4B")
>>> question = "Explain Brownian motion briefly."
>>> answer = chat.reply(question)
>>> print(answer)

Callers should invoke ``close()`` explicitly
when the loaded model is no longer needed:

>>> chat.close()

Use only locally available model files:

>>> chat = qachat.ChatSession(
...     "Qwen/Qwen3-4B",
...     allow_download=False,
... )

"""

from __future__ import annotations

import gc
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any, cast, TYPE_CHECKING, TypeAlias

from qachat.huggingface import cached_model_snapshot
from qachat.runtime import (
    Backend,
    SelectedBackend,
    choose_backend as _choose_backend,
    torch_device as _torch_device,
)
from qachat.trace_logging import NullTraceLogger, TraceSink

__all__ = [
    "ChatSession",
]

if TYPE_CHECKING:
    from mlx.nn import Module as MlxModel
    from mlx_lm.tokenizer_utils import TokenizerWrapper as MlxTokenizer
    from transformers import PreTrainedModel as TorchModel
    from transformers import PreTrainedTokenizerBase as TorchTokenizer
else:
    MlxModel = Any
    MlxTokenizer = Any
    TorchModel = Any
    TorchTokenizer = Any

Model: TypeAlias = MlxModel | TorchModel
Tokenizer: TypeAlias = MlxTokenizer | TorchTokenizer


class ChatSession:
    """A loaded model, its settings, and one in-memory conversation.

    ``ChatSession`` owns the runtime objects required for one conversational
    model session. Construction selects a backend, validates generation
    settings, initializes the message history, and immediately loads the model
    and tokenizer.

    Parameters
    ----------
    model_id: str
        Hugging Face model ID or path to a local model directory.
    backend: "auto" | "mlx" | "torch", optional
        Calculation backend. ``auto`` delegates platform selection to the
        shared QAChat runtime utilities.
    thinking: bool, optional
        Whether a compatible chat template should enable thinking output.
        Defaults to ``False``.
    temperature: float, optional
        Sampling temperature. ``0`` requests deterministic greedy generation;
        positive values enable sampling. Defaults to ``0``.
    max_new_tokens: int | None, optional
        Session-level maximum number of tokens generated for each reply.
        ``None`` uses the remaining model context unless a per-reply override
        is supplied.
    system_prompt: str | None, optional
        Optional system instruction placed first in the conversation history.
    allow_download: bool, optional
        Whether model loading may download missing Hugging Face files.
        ``False`` restricts model IDs to locally cached snapshots or local
        directories. Defaults to ``True``.
    revision: str | None, optional
        Optional Hugging Face branch, tag, or commit to load.
    trace: TraceSink | None, optional
        Trace sink receiving structured model-loading and generation events.
        ``None`` disables trace output through a ``NullTraceLogger``.

    Attributes
    ----------
    model_id: str
        Requested Hugging Face model ID or local model directory.
    model_source: str
        Resolved source passed to the backend loader.
    backend: "mlx" | "torch"
        Selected calculation backend.
    device: str
        Device used for generation, such as ``metal``, ``cuda``, ``mps``, or
        ``cpu``.
    thinking: bool
        Whether compatible templates enable thinking output.
    temperature: float
        Current sampling temperature.
    max_new_tokens: int | None
        Current session-level generation-length limit.
    messages: list[dict[str, str]]
        Ordered in-memory conversation history.
    trace: TraceSink
        Trace sink used by the session.

    Raises
    ------
    ValueError
        If ``model_id`` or any generation setting is invalid.
    RuntimeError
        If the selected backend cannot be used or the model cannot be loaded.
    ImportError
        If a dependency required by the selected backend is unavailable.

    Notes
    -----
    The session loads the model during construction. Call ``close()`` when the
    model is no longer needed so backend caches and Python references can be
    released explicitly.

    """

    model_id: str
    backend: SelectedBackend
    device: str
    thinking: bool
    temperature: float
    max_new_tokens: int | None
    messages: list[dict[str, str]]
    _system_prompt: str | None
    _model: Model | None
    _tokenizer: Tokenizer | None

    def __init__(
        self: ChatSession,
        model_id: str,
        *,
        backend: Backend = "auto",
        thinking: bool = False,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
        system_prompt: str | None = None,
        allow_download: bool = True,
        revision: str | None = None,
        trace: TraceSink | None = None,
    ) -> None:
        """Initialize and load one chat session.

        Validation is performed before model loading where possible. The
        initial history contains only the normalized system prompt when one is
        supplied.

        Raises
        ------
        ValueError
            If the model identifier or a session setting is invalid.
        RuntimeError
            If the selected backend is unavailable or model loading fails.
        ImportError
            If a dependency required by the selected backend is unavailable.

        """

        self.trace = trace if trace is not None else NullTraceLogger()
        self.backend = _choose_backend(backend)
        self.model_id = model_id
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError(
                "Enter a Hugging Face model ID or local model path."
            )
        self.thinking = _check_bool(thinking, "thinking")
        self.temperature = _check_temperature(temperature)
        self.max_new_tokens = _check_max_new_tokens(max_new_tokens)
        self._system_prompt = _check_system_prompt(system_prompt)
        self.messages = self._new_history()
        self._model = None
        self._tokenizer = None
        self.device = ""
        self._load(
            allow_download=_check_bool(allow_download, "allow_download"),
            revision=revision,
        )

    def _load(
        self: ChatSession,
        *,
        allow_download: bool,
        revision: str | None,
    ) -> None:
        """Resolve and load the selected model and tokenizer.

        Parameters
        ----------
        allow_download: bool
            Whether backend loaders may fetch missing Hugging Face files.
        revision: str | None
            Optional Hugging Face branch, tag, or commit.

        Raises
        ------
        RuntimeError
            If MLX is selected but Metal is unavailable.
        ImportError
            If a dependency required by the selected backend is unavailable.

        Notes
        -----
        Local directories are passed directly to the backend loader. When
        downloads are disabled and ``model_id`` is not a local directory, the
        corresponding cached Hugging Face snapshot is resolved first.

        """

        requested_revision = revision
        with self.trace.span(
            "model.chat.load",
            component="chat",
            data={
                "model_id": self.model_id,
                "backend": self.backend,
                "allow_download": allow_download,
                "requested_revision": requested_revision,
            },
        ):
            source_path = Path(self.model_id).expanduser()
            source = (
                str(source_path) if source_path.is_dir() else self.model_id
            )
            if not allow_download and not source_path.is_dir():
                source = str(cached_model_snapshot(source, revision=revision))
                revision = None
            self.model_source = source

            if self.backend == "mlx":
                mx = import_module("mlx.core")
                if not mx.metal.is_available():
                    raise RuntimeError("MLX Metal is not available.")
                mlx_lm = import_module("mlx_lm")
                loaded_model, loaded_tokenizer = mlx_lm.load(
                    source,
                    tokenizer_config={"trust_remote_code": False},
                    revision=revision,
                )
                self._model = loaded_model
                self._tokenizer = loaded_tokenizer
                self.device = "metal"
            else:
                torch = import_module("torch")
                transformers = import_module("transformers")
                local_only = not allow_download
                loaded_tokenizer = transformers.AutoTokenizer.from_pretrained(
                    source,
                    local_files_only=local_only,
                    revision=revision,
                    trust_remote_code=False,
                )
                loaded_model = (
                    transformers.AutoModelForCausalLM.from_pretrained(
                        source,
                        local_files_only=local_only,
                        revision=revision,
                        trust_remote_code=False,
                        dtype="auto",
                    )
                )
                self.device = _torch_device(
                    torch.cuda.is_available(),
                    torch.backends.mps.is_available(),
                )
                loaded_model.to(self.device)
                loaded_model.eval()
                self._model = loaded_model
                self._tokenizer = loaded_tokenizer

            self.trace.emit(
                "model.chat.resolved",
                component="chat",
                data={
                    "model_id": self.model_id,
                    "model_source": source,
                    "requested_revision": requested_revision,
                    "loaded_revision": revision,
                    "backend": self.backend,
                    "device": self.device,
                },
            )

    def configure_generation(
        self: ChatSession,
        settings: dict[str, object],
        *,
        trace_turn_id: str | None = None,
    ) -> None:
        """Update generation-only settings without reloading the model.

        Supported settings are ``temperature``, ``thinking``, and
        ``max_new_tokens``. All supplied values are validated before any
        setting is changed, so the update is applied atomically.

        Parameters
        ----------
        settings: dict[str, object]
            Generation settings to update. Omitted supported settings retain
            their current values.
        trace_turn_id: str | None, optional
            Optional trace identifier associating the settings change with a
            conversation turn.

        Raises
        ------
        ValueError
            If ``settings`` is not a dictionary, contains an unsupported key,
            or contains an invalid value.

        """

        if not isinstance(settings, dict):
            raise ValueError("Generation settings must be a dictionary.")
        unknown = settings.keys() - {
            "temperature",
            "thinking",
            "max_new_tokens",
        }
        if unknown:
            raise ValueError(
                f"Unknown generation setting: {sorted(unknown)[0]}"
            )
        before = {
            "temperature": self.temperature,
            "thinking": self.thinking,
            "max_new_tokens": self.max_new_tokens,
        }
        temperature = _check_temperature(
            settings.get("temperature", self.temperature)
        )
        thinking = _check_bool(
            settings.get("thinking", self.thinking), "thinking"
        )
        max_new_tokens = _check_max_new_tokens(
            settings.get("max_new_tokens", self.max_new_tokens)
        )
        self.temperature = temperature
        self.thinking = thinking
        self.max_new_tokens = max_new_tokens
        after = {
            "temperature": self.temperature,
            "thinking": self.thinking,
            "max_new_tokens": self.max_new_tokens,
        }
        self.trace.settings_changed(
            "generation.settings.changed",
            component="chat",
            before=before,
            after=after,
            turn_id=trace_turn_id,
        )

    def reply(
        self: ChatSession,
        message: str,
        *,
        max_new_tokens: int | None = None,
        trace_turn_id: str | None = None,
    ) -> str:
        """Generate, return, and remember one assistant reply.

        The new user message is appended to the existing history for prompt
        construction. The selected chat template converts the complete
        conversation into model tokens, after which MLX or PyTorch performs
        generation. The user message and generated assistant reply are stored
        only after generation succeeds.

        Parameters
        ----------
        message: str
            Non-empty user message.
        max_new_tokens: int | None, optional
            Per-reply generation-length override. ``None`` uses the session
            setting and, if that is also ``None``, the remaining model
            context.
        trace_turn_id: str | None, optional
            Optional trace identifier associating emitted events and artifacts
            with the current conversation turn.

        Returns
        -------
        str
            Generated assistant text with surrounding whitespace removed.

        Raises
        ------
        RuntimeError
            If the session is closed or the model returns no usable text.
        ValueError
            If ``message`` is empty, ``max_new_tokens`` is invalid, the model
            context size is unavailable when needed, or the conversation has
            exhausted the model context.

        """

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("This chat is closed. Create a new session.")

        if not isinstance(message, str) or not message.strip():
            raise ValueError("Enter a non-empty message.")

        reply_started_at = perf_counter()
        token_override = _check_max_new_tokens(max_new_tokens)
        messages = [
            *self.messages,
            {"role": "user", "content": message.strip()},
        ]
        rendered_prompt: object | None = None
        if self.trace.include_content:
            try:
                rendered_prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.thinking,
                )
            except (TypeError, ValueError):
                rendered_prompt = None

        tokens = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        token_budget = token_override or self.max_new_tokens
        if token_budget is None:
            token_budget = self._remaining_context(len(tokens))

        input_data: dict[str, object] = {
            "call_id": self.trace.new_id("llm-call"),
            "call_count_for_question": 1,
            "model_id": self.model_id,
            "model_source": getattr(self, "model_source", self.model_id),
            "backend": self.backend,
            "device": self.device,
            "history_message_count": len(self.messages),
            "current_user_message_count": 1,
            "message_count": len(messages),
            "roles": [item["role"] for item in messages],
            "input_token_count": len(tokens),
            "prompt_token_count": len(tokens),
            "thinking": self.thinking,
            "generation": {
                "temperature": self.temperature,
                "max_new_tokens": token_budget,
                "sampling": self.temperature > 0,
            },
            "chat_template": {
                "add_generation_prompt": True,
                "enable_thinking": self.thinking,
            },
        }
        artifact_stem = trace_turn_id or self.trace.new_id("chat-input")
        if self.trace.include_content:
            input_data["messages_artifact"] = self.trace.artifact(
                f"{artifact_stem}-messages",
                messages,
            )
            if rendered_prompt is not None:
                input_data["rendered_prompt_artifact"] = self.trace.artifact(
                    f"{artifact_stem}-rendered-prompt",
                    rendered_prompt,
                    text=True,
                )
        if self.trace.include_token_ids:
            input_data["input_token_ids_artifact"] = self.trace.artifact(
                f"{artifact_stem}-input-token-ids",
                list(tokens),
            )
        self.trace.emit(
            "llm.call.input",
            component="chat",
            data=input_data,
            turn_id=trace_turn_id,
        )

        self._last_generation_output_token_count = None
        self._last_generation_token_count_source = None
        generation_started_at = perf_counter()
        with self.trace.span(
            "llm.call",
            component="chat",
            data={
                "backend": self.backend,
                "model_id": self.model_id,
                "prompt_token_count": len(tokens),
                "max_new_tokens": token_budget,
                "temperature": self.temperature,
                "call_id": input_data["call_id"],
            },
            turn_id=trace_turn_id,
        ):
            if self.backend == "mlx":
                if self.trace.enabled:
                    answer = self._mlx_reply(
                        tokens,
                        token_budget,
                        trace_turn_id=trace_turn_id,
                    )
                else:
                    answer = self._mlx_reply(tokens, token_budget)
            elif self.trace.enabled:
                answer = self._torch_reply(
                    tokens,
                    token_budget,
                    trace_turn_id=trace_turn_id,
                )
            else:
                answer = self._torch_reply(tokens, token_budget)
        generation_duration_ms = (
            perf_counter() - generation_started_at
        ) * 1000

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("The model returned no text.")

        if (
            self.trace.enabled
            and self._last_generation_output_token_count is None
        ):
            self._last_generation_output_token_count = _text_token_count(
                self._tokenizer,
                answer,
            )
            self._last_generation_token_count_source = (
                "retokenized_decoded_text"
                if self._last_generation_output_token_count is not None
                else "unavailable"
            )
        answer = answer.strip()
        self.messages = [
            *messages,
            {"role": "assistant", "content": answer},
        ]
        output_data: dict[str, object] = {
            "call_id": input_data["call_id"],
            "call_count_for_question": 1,
            "model_id": self.model_id,
            "input_token_count": len(tokens),
            "output_token_count": (self._last_generation_output_token_count),
            "output_token_count_source": (
                self._last_generation_token_count_source
            ),
            "character_count": len(answer),
            "generation_duration_ms": round(generation_duration_ms, 3),
            "reply_duration_ms": round(
                (perf_counter() - reply_started_at) * 1000,
                3,
            ),
        }
        if self.trace.include_content:
            output_data["answer_artifact"] = self.trace.artifact(
                f"{artifact_stem}-answer",
                answer,
                text=True,
            )
        self.trace.emit(
            "llm.call.output",
            component="chat",
            data=output_data,
            turn_id=trace_turn_id,
            duration_ms=generation_duration_ms,
        )
        return answer

    def _remaining_context(
        self: ChatSession,
        prompt_tokens: int,
    ) -> int:
        """Return the model context still available for generation.

        Parameters
        ----------
        prompt_tokens: int
            Number of tokens occupied by the formatted prompt.

        Returns
        -------
        int
            Positive number of tokens available for generation.

        Raises
        ------
        ValueError
            If the model does not report a valid context size or the prompt
            already fills the available context.

        """

        if self.backend == "mlx":
            model = cast("MlxModel", self._model)
            context = model.args.max_position_embeddings
        else:
            model = cast("TorchModel", self._model)
            context = model.config.max_position_embeddings

        if not isinstance(context, int) or context < 1:
            raise ValueError(
                "This model does not report its context size. Set "
                "max_new_tokens when creating the session or sending a reply."
            )

        remaining = context - prompt_tokens

        if remaining < 1:
            raise ValueError("The conversation has filled the model context.")

        return remaining

    def _mlx_reply(
        self: ChatSession,
        tokens: list[int],
        token_budget: int,
        *,
        trace_turn_id: str | None = None,
    ) -> str:
        """Generate one decoded reply through MLX.

        Parameters
        ----------
        tokens: list[int]
            Tokenized and formatted conversation prompt.
        token_budget: int
            Maximum number of new tokens to generate.
        trace_turn_id: str | None, optional
            Reserved trace identifier for parity with backend generation
            helpers. The current MLX generation path does not use it directly.

        Returns
        -------
        str
            Decoded model output.

        """

        mlx_lm = import_module("mlx_lm")
        sampling = import_module("mlx_lm.sample_utils")
        sampler = sampling.make_sampler(temp=self.temperature)
        return mlx_lm.generate(
            self._model,
            self._tokenizer,
            prompt=tokens,
            max_tokens=token_budget,
            sampler=sampler,
            verbose=False,
        )

    def _torch_reply(
        self: ChatSession,
        tokens: list[int],
        token_budget: int,
        *,
        trace_turn_id: str | None = None,
    ) -> str:
        """Generate one decoded reply through PyTorch.

        Parameters
        ----------
        tokens: list[int]
            Tokenized and formatted conversation prompt.
        token_budget: int
            Maximum number of new tokens to generate.
        trace_turn_id: str | None, optional
            Optional trace identifier used when recording generated token IDs.

        Returns
        -------
        str
            Decoded model output with special tokens removed.

        """

        torch = import_module("torch")
        model = cast("TorchModel", self._model)
        tokenizer = cast("TorchTokenizer", self._tokenizer)
        inputs = torch.tensor(
            [tokens],
            dtype=torch.long,
            device=self.device,
        )
        options: dict[str, object] = {
            "input_ids": inputs,
            "attention_mask": torch.ones_like(inputs),
            "max_new_tokens": token_budget,
            "do_sample": self.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            options["temperature"] = self.temperature
        with torch.inference_mode():
            output = model.generate(**options)
        generated = output[0, len(tokens) :].tolist()
        self._last_generation_output_token_count = len(generated)
        self._last_generation_token_count_source = "generated_token_ids"
        data: dict[str, object] = {
            "input_token_count": len(tokens),
            "output_token_count": len(generated),
            "generated_token_count": len(generated),
        }
        if self.trace.include_token_ids:
            stem = trace_turn_id or self.trace.new_id("chat-output")
            data["output_token_ids_artifact"] = self.trace.artifact(
                f"{stem}-output-token-ids",
                generated,
            )
        self.trace.emit(
            "llm.call.output.tokens",
            component="chat",
            data=data,
            turn_id=trace_turn_id,
        )
        return tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

    def reset(self: ChatSession) -> None:
        """Reset the conversation while keeping the model loaded.

        The history is replaced with its initial state: an empty list when no
        system prompt is configured, or a single system message otherwise.
        Generation settings and loaded runtime objects are unchanged.

        """

        self.messages = self._new_history()

    def restore_history(
        self: ChatSession,
        messages: list[dict[str, str]],
    ) -> None:
        """Replace the in-memory conversation with validated messages.

        Parameters
        ----------
        messages: list[dict[str, str]]
            Saved messages in conversation order. Each item must contain a
            supported ``role`` and non-empty ``content``. A system message, if
            present, may only appear first.

        Raises
        ------
        ValueError
            If the history is not a list, an item is not a mapping, a role is
            unsupported or misplaced, or message content is empty.

        Notes
        -----
        Message content is stripped before being stored. The supplied list is
        not retained by reference.

        """

        if not isinstance(messages, list):
            raise ValueError("Saved chat history must be a list.")

        checked: list[dict[str, str]] = []
        for index, message in enumerate(messages):

            if not isinstance(message, dict):
                raise ValueError("Every saved chat message must be a mapping.")

            role = message.get("role")

            if role not in {"system", "user", "assistant"}:
                raise ValueError("A saved chat message has an invalid role.")
            if role == "system" and index != 0:
                raise ValueError("A system message can only appear first.")

            content = message.get("content")

            if not isinstance(content, str) or not content.strip():
                raise ValueError("A saved chat message has empty content.")

            checked.append(
                {
                    "role": role,
                    "content": content.strip(),
                }
            )

        self.messages = checked

    def close(self: ChatSession) -> None:
        """Release model resources and reset the conversation.

        Closing clears the model and tokenizer references, restores the initial
        history, runs Python garbage collection, and clears the active MLX,
        CUDA, or MPS cache when applicable.

        Repeated calls are safe. After closing, ``reply()`` raises
        ``RuntimeError``; create a new ``ChatSession`` to load the model again.

        """

        if self._model is None:
            return

        self._model = None
        self._tokenizer = None
        self.reset()
        gc.collect()
        if self.backend == "mlx":
            import_module("mlx.core").clear_cache()
        elif self.device == "cuda":
            import_module("torch").cuda.empty_cache()
        elif self.device == "mps":
            import_module("torch").mps.empty_cache()

    def _new_history(self: ChatSession) -> list[dict[str, str]]:
        """Create the initial conversation history.

        Returns
        -------
        list[dict[str, str]]
            Empty history when no system prompt is configured, otherwise a
            single normalized system message.

        """

        if self._system_prompt is None:
            return []

        return [
            {
                "role": "system",
                "content": self._system_prompt,
            }
        ]


def _check_bool(value: bool, label: str) -> bool:
    """Validate a Boolean setting.

    Parameters
    ----------
    value: bool
        Value to validate.
    label: str
        Setting name included in an error message.

    Returns
    -------
    bool
        Validated Boolean value.

    Raises
    ------
    ValueError
        If ``value`` is not a Boolean.

    """

    if not isinstance(value, bool):
        raise ValueError(f"{label} must be True or False.")

    return value


def _check_temperature(value: float) -> float:
    """Validate and normalize a sampling temperature.

    Parameters
    ----------
    value: float
        Numeric temperature to validate.

    Returns
    -------
    float
        Temperature converted to ``float``.

    Raises
    ------
    ValueError
        If ``value`` is Boolean, non-numeric, or negative.

    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("temperature must be zero or a positive number.")

    if value < 0:
        raise ValueError("temperature must be zero or a positive number.")

    return float(value)


def _text_token_count(tokenizer: object, text: str) -> int | None:
    """Count tokens in decoded text when the tokenizer permits it.

    Parameters
    ----------
    tokenizer: object
        Tokenizer-like object that may provide an ``encode`` method.
    text: str
        Decoded generated text to tokenize.

    Returns
    -------
    int | None
        Number of encoded tokens, or ``None`` when token counting is not
        supported or fails with a handled tokenizer error.

    Notes
    -----
    The helper first requests encoding without added special tokens. It falls
    back to a simpler ``encode(text)`` call for tokenizer implementations that
    do not accept ``add_special_tokens``.

    """

    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return None
    try:
        try:
            tokens = encode(text, add_special_tokens=False)
        except TypeError:
            tokens = encode(text)
    except (AttributeError, TypeError, ValueError):
        return None
    try:
        return len(tokens)
    except TypeError:
        return None


def _check_max_new_tokens(value: int | None) -> int | None:
    """Validate an optional generation-length limit.

    Parameters
    ----------
    value: int | None
        Token limit to validate.

    Returns
    -------
    int | None
        Validated positive integer or ``None``.

    Raises
    ------
    ValueError
        If ``value`` is neither ``None`` nor a positive non-Boolean integer.

    """

    if value is not None and (type(value) is not int or value < 1):
        raise ValueError("max_new_tokens must be a positive integer or None.")

    return value


def _check_system_prompt(value: str | None) -> str | None:
    """Validate and normalize an optional system instruction.

    Parameters
    ----------
    value: str | None
        System instruction to validate.

    Returns
    -------
    str | None
        Stripped non-empty instruction or ``None``.

    Raises
    ------
    ValueError
        If ``value`` is not ``None`` or non-empty text.

    """

    if value is None:
        return None

    if not isinstance(value, str) or not value.strip():
        raise ValueError("system_prompt must be non-empty text or None.")

    return value.strip()
