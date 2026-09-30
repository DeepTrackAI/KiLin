"""Load a language model and keep one conversation in memory.

``ChatSession`` selects MLX or PyTorch, loads one compatible model, formats
messages, generates replies, and releases the model when finished.

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

    Parameters
    ----------
    model_id: str
        A Hugging Face model ID or local model directory.
    backend: "auto" | "mlx" | "torch", optional
        ``auto`` selects MLX on Apple Silicon and PyTorch elsewhere.
    thinking: bool, optional
        Ask a compatible chat template to enable thinking output.
    temperature: float, optional
        ``0`` uses repeatable greedy generation. A higher value samples.
    max_new_tokens: int | None, optional
        Optional generation length. ``None`` uses the remaining model context
        instead of an application-defined limit.
    system_prompt: str | None, optional
        Optional instruction kept at the beginning of the conversation.
    allow_download: bool, optional
        If ``True``, Hugging Face downloads missing files to its normal cache.
        If ``False``, loading only uses local files.
    revision: str | None, optional
        Optional Hugging Face branch, tag, or commit.
    trace: TraceSink | None, optional
        Structured execution trace shared with the calling session.

    Attributes
    ----------
    model_id: str
        Loaded Hugging Face model ID or local directory.
    backend: "mlx" | "torch"
        Selected calculation engine.
    device: str
        Device used by the selected engine.
    thinking: bool
        Whether compatible templates enable thinking output.
    temperature: float
        Sampling temperature.
    max_new_tokens: int | None
        Optional session-level reply length.
    messages: list[dict[str, str]]
        Conversation kept in memory.

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
        """Initialize settings, history, tokenizer, and model."""

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
        """Load only the selected engine and model.

        Parameters
        ----------
        allow_download: bool
            Whether Hugging Face may fetch missing files.
        revision: str | None
            Optional Hugging Face revision.

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

        Only temperature, thinking, and answer length may change here. All
        values are checked before any setting is applied.
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
        """Generate and remember one reply.

        Parameters
        ----------
        message: str
            New user message.
        max_new_tokens: int | None, optional
            Optional one-reply override. ``None`` uses the session setting or
            the remaining model context.

        Returns
        -------
        str
            Generated assistant text.

        Raises
        ------
        RuntimeError
            If the session is closed or the model returns no text.
        ValueError
            If the message or token setting is invalid.

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
        """Calculate the model's remaining context.

        Parameters
        ----------
        prompt_tokens: int
            Number of tokens already used by the prompt.

        Returns
        -------
        int
            Tokens available for generation.

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
        """Generate through MLX.

        Parameters
        ----------
        tokens: list[int]
            Formatted prompt tokens.
        token_budget: int
            Maximum tokens for this reply.

        Returns
        -------
        str
            Decoded reply.

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
        """Generate through PyTorch.

        Parameters
        ----------
        tokens: list[int]
            Formatted prompt tokens.
        token_budget: int
            Maximum tokens for this reply.

        Returns
        -------
        str
            Decoded reply.

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
        """Forget the conversation while keeping the model loaded."""

        self.messages = self._new_history()

    def restore_history(
        self: ChatSession,
        messages: list[dict[str, str]],
    ) -> None:
        """Replace the in-memory conversation with checked saved messages.

        Parameters
        ----------
        messages: list[dict[str, str]]
            Saved system, user, and assistant messages in conversation order.

        Raises
        ------
        ValueError
            If the saved conversation has an invalid message.

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
        """Release the model, tokenizer, conversation, and engine cache."""

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
        """Create empty history with the optional system instruction.

        Returns
        -------
        list[dict[str, str]]
            Initial conversation messages.

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
    """Check a Boolean setting.

    Parameters
    ----------
    value: bool
        Value to check.
    label: str
        Setting name used in an error.

    Returns
    -------
    bool
        Checked value.

    """

    if not isinstance(value, bool):
        raise ValueError(f"{label} must be True or False.")

    return value


def _check_temperature(value: float) -> float:
    """Check a sampling temperature.

    Parameters
    ----------
    value: float
        Temperature to check.

    Returns
    -------
    float
        Checked temperature.

    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("temperature must be zero or a positive number.")

    if value < 0:
        raise ValueError("temperature must be zero or a positive number.")

    return float(value)


def _text_token_count(tokenizer: object, text: str) -> int | None:
    """Count decoded output with the active tokenizer when necessary."""

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
    """Check an optional generation length without imposing an upper cap.

    Parameters
    ----------
    value: int | None
        Token count to check.

    Returns
    -------
    int | None
        Checked value.

    """

    if value is not None and (type(value) is not int or value < 1):
        raise ValueError("max_new_tokens must be a positive integer or None.")

    return value


def _check_system_prompt(value: str | None) -> str | None:
    """Check and normalize an optional system instruction.

    Parameters
    ----------
    value: str | None
        System instruction to check.

    Returns
    -------
    str | None
        Stripped instruction or ``None``.

    """

    if value is None:
        return None

    if not isinstance(value, str) or not value.strip():
        raise ValueError("system_prompt must be non-empty text or None.")

    return value.strip()
