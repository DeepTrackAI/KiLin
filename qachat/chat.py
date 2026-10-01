"""Manage one loaded language model and its in-memory conversation.

``ChatSession`` selects a backend, resolves a local or Hugging Face model
source, loads a compatible model and tokenizer, and generates one reply at
a time. Construction loads the model immediately. Downloads are permitted
by default; ``allow_download=False`` requests local-only model resolution.

Messages are dictionaries with ``role`` and ``content`` fields. The string
constants ``SYSTEM_ROLE``, ``USER_ROLE``, and ``ASSISTANT_ROLE`` identify,
respectively, the optional conversation instruction, user messages, and
model replies. Initial history contains the constructor's system instruction
when supplied. Restored history may contain a system message only first.
The model's chat template interprets these roles; they are not an independent
access-control or instruction-enforcement mechanism.

Every reply renders the complete retained history. This layer does not
summarize, truncate, persist, or retrieve old messages. A caller-specified
generation limit takes precedence over the automatic remaining-context
calculation. Explicit limits are not checked against the context size here.
Tracing can retain prompts and answers according to the supplied sink.

Key Features
------------

- **Backend-specific loading**

    Use MLX or PyTorch while importing the selected model backend lazily.

- **Canonical roles**

    Use ordinary string-valued role constants in message construction and
    restored-history checks.

- **Conversation lifecycle**

    Generate replies, update generation settings, replace or reset history, and
    explicitly release loaded resources.

- **Optional tracing**

    Record model resolution, generation settings, timings, and opt-in content
    or token artifacts.

Module Structure
----------------

- ``SYSTEM_ROLE``

    String value ``"system"`` for the optional conversation-level instruction.

- ``USER_ROLE``

    String value ``"user"`` for user messages.

- ``ASSISTANT_ROLE``

    String value ``"assistant"`` for generated replies.

- ``ChatSession``

    A loaded model, its settings, and one in-memory conversation.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
Create a session from a model directory compatible with the selected backend:

>>> import qachat
>>> chat = qachat.ChatSession(
...     "/path/to/local/chat-model",
...     allow_download=False,
...     system_prompt="Explain scientific concepts clearly.",
...     max_new_tokens=256,
... )
>>> try:
...     answer = chat.reply("Explain Brownian motion briefly.")
... finally:
...     chat.close()

The path is illustrative and must identify an available compatible model.

"""

from __future__ import annotations

import gc
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any, cast, Literal, TYPE_CHECKING, TypeAlias

from qachat.huggingface import cached_model_snapshot
from qachat.runtime import (
    Backend,
    SelectedBackend,
    choose_backend as _choose_backend,
    torch_device as _torch_device,
)
from qachat.trace_logging import NullTraceLogger, TraceSink

__all__ = [
    "SYSTEM_ROLE",
    "USER_ROLE",
    "ASSISTANT_ROLE",
    "ChatSession",
]

SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"

Role: TypeAlias = Literal["system", "user", "assistant"]
_ROLES = {
    SYSTEM_ROLE,
    USER_ROLE,
    ASSISTANT_ROLE,
}

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
        shared QAChat runtime utilities. Defaults to ``'auto'``.
    thinking: bool, optional
        Whether a compatible chat template should enable thinking output.
        Defaults to ``False``.
    temperature: float, optional
        Sampling temperature. Zero selects greedy decoding; positive values
        enable sampling. This is not a cross-device reproducibility guarantee.
        The current validator does not reject NaN or positive infinity.
        Defaults to ``0.0``.
    max_new_tokens: int | None, optional
        Session-level maximum number of tokens generated for each reply.
        ``None`` uses the remaining model context unless a per-reply override
        is supplied. Defaults to ``None``.
    system_prompt: str | None, optional
        Optional nonblank system instruction, stripped and used as the first
        initial-history message. Reset restores this constructor value.
        Defaults to ``None``.
    allow_download: bool, optional
        Whether model loading may download missing Hugging Face files.
        ``False`` restricts model IDs to locally cached snapshots or local
        directories. Defaults to ``True``.
    revision: str | None, optional
        Optional Hugging Face branch, tag, or commit to load. Defaults to
        ``None``.
    trace: TraceSink | None, optional
        Caller-owned trace sink. None selects a NullTraceLogger. Content
        retention depends on the sink and its flags. Defaults to ``None``.

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
        Ordered in-memory conversation history using the canonical roles.
    trace: TraceSink
        Trace sink used by the session.

    Methods
    -------
    configure_generation(settings, *, trace_turn_id=None)
        Update generation settings after validating all supplied values.
    reply(message, *, max_new_tokens=None, trace_turn_id=None)
        Generate, return, and retain one assistant reply.
    reset()
        Reset the conversation while keeping the model loaded.
    restore_history(messages)
        Replace the in-memory conversation with validated messages.
    close()
        Release loaded references and request backend cache cleanup.

    Raises
    ------
    ValueError
        If backend selection, the model identifier, or a validated setting is
        invalid.
    ImportError
        If a selected backend dependency is missing.
    RuntimeError
        If MLX is selected but Metal is unavailable. Other backend runtime
        errors can also propagate.

    Notes
    -----
    Construction loads a model immediately. Close the session explicitly when
    finished; no context-manager protocol is provided. The constants
    SYSTEM_ROLE, USER_ROLE, and ASSISTANT_ROLE contain the strings expected by
    compatible chat templates. Template support is model-dependent, and roles
    do not create a separate security boundary. Backend and Hub errors may
    propagate without conversion to RuntimeError.

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

        Parameters
        ----------
        model_id: str
            Hugging Face model ID or path to a local model directory.
        backend: "auto" | "mlx" | "torch", optional
            Calculation backend. ``auto`` delegates platform selection to the
            shared QAChat runtime utilities. Defaults to ``'auto'``.
        thinking: bool, optional
            Whether a compatible chat template should enable thinking output.
            Defaults to ``False``.
        temperature: float, optional
            Sampling temperature. Zero selects greedy decoding; positive values
            enable sampling. This is not a cross-device reproducibility
            guarantee. The current validator does not reject NaN or positive
            infinity. Defaults to ``0.0``.
        max_new_tokens: int | None, optional
            Session-level maximum number of tokens generated for each reply.
            ``None`` uses the remaining model context unless a per-reply
            override is supplied. Defaults to ``None``.
        system_prompt: str | None, optional
            Optional nonblank system instruction, stripped and used as the
            first initial-history message. Reset restores this constructor
            value. Defaults to ``None``.
        allow_download: bool, optional
            Whether model loading may download missing Hugging Face files.
            ``False`` restricts model IDs to locally cached snapshots or local
            directories. Defaults to ``True``.
        revision: str | None, optional
            Optional Hugging Face branch, tag, or commit to load. Defaults to
            ``None``.
        trace: TraceSink | None, optional
            Caller-owned trace sink. None selects a NullTraceLogger. Content
            retention depends on the sink and its flags. Defaults to ``None``.

        Raises
        ------
        ValueError
            If backend selection, the model identifier, or a validated setting
            is invalid.
        ImportError
            If a selected backend dependency cannot be imported.
        RuntimeError
            If the MLX Metal check fails; loader-specific errors otherwise
            propagate.

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
        """Update generation settings after validating all supplied values.

        Only temperature, thinking, and max_new_tokens may change. Omitted
        settings retain their values. Validation completes before assignments,
        but the operation is not a thread-safety or trace-write transaction.

        Parameters
        ----------
        settings: dict[str, object]
            Generation settings to update. Omitted supported settings retain
            their current values.
        trace_turn_id: str | None, optional
            Optional trace identifier associating the settings change with a
            conversation turn. Defaults to ``None``.

        Raises
        ------
        ValueError
            If ``settings`` is not a dictionary, contains an unsupported key,
            or contains an invalid value.

        Notes
        -----
        The model and current history are unchanged. A trace failure after
        assignment does not restore the previous settings.

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
        """Generate, return, and retain one assistant reply.

        The complete retained history plus the stripped user message is
        rendered by the tokenizer chat template. After a nonblank answer is
        produced, the stripped user and assistant messages replace the stored
        history. Tokenization or generation failures before that point leave
        history unchanged.

        Parameters
        ----------
        message: str
            Non-empty user message.
        max_new_tokens: int | None, optional
            Per-reply generation-length override. ``None`` uses the session
            setting and, if that is also ``None``, the remaining model context.
            Defaults to ``None``.
        trace_turn_id: str | None, optional
            Optional trace identifier associating emitted events and artifacts
            with the current conversation turn. Defaults to ``None``.

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

        Notes
        -----
        A per-call positive token limit overrides the session limit. Only when
        both are None is the remaining context calculated. Explicit limits are
        not clamped to that context, and no history truncation or summarization
        occurs. Final output tracing happens after history is committed; an
        error there can raise even though the turn has been retained. The
        returned text is decoded model output, not a validated answer or a
        guarantee that thinking content was removed.

        """

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("This chat is closed. Create a new session.")

        if not isinstance(message, str) or not message.strip():
            raise ValueError("Enter a non-empty message.")

        reply_started_at = perf_counter()
        token_override = _check_max_new_tokens(max_new_tokens)
        messages = [
            *self.messages,
            {"role": USER_ROLE, "content": message.strip()},
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
            {"role": ASSISTANT_ROLE, "content": answer},
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
            If a present context size is not a positive integer, or no
            generation token remains.
        AttributeError
            If the selected model does not expose the expected context-size
            attribute.

        Notes
        -----
        The configured max_position_embeddings field is read directly. An
        absent model args/config field can raise AttributeError before the
        explicit ValueError checks.

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
            Defaults to ``None``.

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
            Defaults to ``None``.

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
            List of dictionaries containing role and nonblank content. Accepted
            role values are SYSTEM_ROLE, USER_ROLE, and ASSISTANT_ROLE; a
            system message can appear only at index zero. An empty list is
            allowed.

        Raises
        ------
        ValueError
            If the supplied list, message dictionaries, recognized role values,
            placement, or text content is invalid.
        TypeError
            If a supplied role is unhashable and therefore cannot be checked
            against the role set.

        Notes
        -----
        The method copies role/content only and strips content. Additional
        fields are discarded, and role alternation is not enforced. It does not
        insert or update the constructor system prompt; a later reset restores
        the original constructor instruction. Direct external mutations of the
        public messages list bypass these checks.

        """

        if not isinstance(messages, list):
            raise ValueError("Saved chat history must be a list.")

        checked: list[dict[str, str]] = []
        for index, message in enumerate(messages):

            if not isinstance(message, dict):
                raise ValueError("Every saved chat message must be a mapping.")

            role = message.get("role")

            if role not in _ROLES:
                raise ValueError("A saved chat message has an invalid role.")
            if role == SYSTEM_ROLE and index != 0:
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
        """Release loaded references and request backend cache cleanup.

        The first close with a loaded model clears model/tokenizer references,
        restores initial history (including the constructor system prompt),
        runs garbage collection, and clears the selected device cache where
        applicable. Later calls return immediately when no model remains.

        Notes
        -----
        This does not close a caller-owned trace sink or promise that every
        backend allocation is released. Backend cleanup errors propagate. Reply
        requires a newly loaded session after closing.

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
            single normalized ``SYSTEM_ROLE`` message.

        """

        if self._system_prompt is None:
            return []

        return [
            {
                "role": SYSTEM_ROLE,
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

    Notes
    -----
    The test rejects booleans, non-int/non-float values, and values below zero.
    It does not check finiteness, so NaN and positive infinity currently pass
    unchanged through float conversion.

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
    This is a best-effort count of re-encoded decoded text, not necessarily the
    number of generated token IDs. Encoding is first attempted without added
    special tokens; a TypeError triggers a call without that keyword. Handled
    AttributeError, TypeError, and ValueError failures return None; other
    exceptions propagate.

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
