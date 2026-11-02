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
A per-reply thinking override changes only the chat-template request, not
the configured session setting. Thinking control depends on the selected
model and its template. Tracing can retain prompts and answers according
to the supplied sink.

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

>>> import kilin
>>> chat = kilin.ChatSession(
...     "/path/to/local/chat-model",
...     allow_download=False,
...     system_prompt="Explain scientific concepts clearly.",
...     max_new_tokens=256,
... )
>>> try:
...     answer = chat.reply("Explain Brownian motion briefly.")
... finally:
...     chat.close()

The path is illustrative and must identify an available compatible model. It
can also be the model id.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import gc
from importlib import import_module
from math import isfinite
from pathlib import Path
import re
from time import perf_counter
from typing import Any, cast, Final, Literal, TYPE_CHECKING, TypeAlias

from kilin._model_completion import (
    GenerationCancelled,
    IncompleteModelResponse,
)
from kilin.defaults import (
    load_defaults_section,
    require_default_lowercase_words,
)
from kilin.huggingface import cached_model_snapshot
from kilin.runtime import (
    BACKEND_AUTO,
    BACKEND_MLX,
    Backend,
    SelectedBackend,
    choose_backend as _choose_backend,
    torch_device as _torch_device,
)
from kilin.trace_logging import (
    TRACE_EVENT_LLM_CALL_INPUT,
    TRACE_EVENT_LLM_CALL_OUTPUT,
    TRACE_EVENT_LLM_CALL_OUTPUT_TOKENS,
    NullTraceLogger,
    TraceSink,
)

__all__ = [
    "SYSTEM_ROLE",
    "USER_ROLE",
    "ASSISTANT_ROLE",
    "ChatSession",
]

Role: TypeAlias = Literal["system", "user", "assistant"]

SYSTEM_ROLE: Final[Role] = "system"
USER_ROLE: Final[Role] = "user"
ASSISTANT_ROLE: Final[Role] = "assistant"

_ROLES: Final[frozenset[Role]] = frozenset(
    (SYSTEM_ROLE, USER_ROLE, ASSISTANT_ROLE)
)

# Curated Unicode 17.0 Han, kana, Hangul, and Bopomofo ranges. Deliberately
# exclude general punctuation, fullwidth Latin, and unrelated Asian scripts.
# See docs/model_management.md for the coverage and tokenizer limitations.
_CJK: Final[re.Pattern[str]] = re.compile(
    "[\u1100-\u11ff\u2e80-\u2fdf\u3041-\u3096\u3099-\u309f"
    "\u30a1-\u30fa\u30fc-\u30ff\u3105-\u312f\u3131-\u318e"
    "\u31a0-\u31bf\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff"
    "\ua960-\ua97f\uac00-\ud7a3\ud7b0-\ud7ff\uf900-\ufaff"
    "\uff66-\uff9f\uffa0-\uffdc\U0001aff0-\U0001afff"
    "\U0001b000-\U0001b16f\U00020000-\U0002a6df"
    "\U0002a700-\U0002ee5f\U0002f800-\U0002fa1f"
    "\U00030000-\U0003347f]"
)
_CJK_REQUEST_WORDS: Final = require_default_lowercase_words(
    load_defaults_section("CHAT"), "CJK_REQUEST_WORDS", "CHAT"
)
_CJK_LANGUAGE = (
    r"(?:" + "|".join(map(re.escape, sorted(_CJK_REQUEST_WORDS))) + r")s?"
)
_CJK_REQUEST: Final[re.Pattern[str]] = re.compile(
    r"\b(?:in|into|using)\s+(?:(?:simplified|traditional|the)\s+)*"
    + _CJK_LANGUAGE + r"\b|"
    r"\b(?:use|include|show|write|print|quote|give|provide|output)\s+"
    r"(?:(?:me|a|the|some|its|original|simplified|traditional)\s+)*"
    + _CJK_LANGUAGE + r"\b|"
    r"\b" + _CJK_LANGUAGE
    + r"\s+(?:for|translation|equivalent|characters?|spelling|version)\b|"
    r"^\s*" + _CJK_LANGUAGE + r"(?:,?\s+(?:please|only))?[.!?]?\s*$",
    re.IGNORECASE,
)
_CJK_NEGATIVE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:do\s+not|don['\u2019]t|never|avoid|without|no|exclude|stop)\b"
    r"[^,.!?;\n]{0,80}\b" + _CJK_LANGUAGE + r"\b|"
    r"\b" + _CJK_LANGUAGE
    + r"(?:\s+\w+){0,3}\s+not\s+(?:needed|wanted|allowed|requested)\b",
    re.IGNORECASE,
)
# Recognize common explicit switches away from CJK without treating every
# occurrence of a language name as an instruction. This is not a parser.
_LATIN_LANGUAGE = (
    r"(?:english|french|german|spanish|italian|portuguese|swedish|"
    r"norwegian|danish|dutch|finnish|icelandic|polish|czech|turkish|"
    r"pinyin|romaji|romanized\s+\w+|latin(?:[- ]script|\s+letters)?)"
)
_NON_CJK_REQUEST: Final[re.Pattern[str]] = re.compile(
    r"\b(?:in|into|use|using)\s+(?:only\s+)?" + _LATIN_LANGUAGE
    + r"\b|\b" + _LATIN_LANGUAGE + r"\s+(?:only|please)\b",
    re.IGNORECASE,
)

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
        shared KíLín runtime utilities. Defaults to ``'auto'``.
    thinking: bool, optional
        Whether a compatible chat template should enable thinking output.
        Defaults to ``False``.
    temperature: float, optional
        Sampling temperature. Zero selects greedy decoding; positive values
        enable sampling. This is not a cross-device reproducibility guarantee.
        The temperature must be finite and nonnegative.
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
    block_unrequested_cjk: bool, optional
        Best-effort suppression of unsolicited Chinese, Japanese, and
        Korean script tokens, based on user-authored text. Defaults to
        ``True``.

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
        Configured thinking setting for compatible templates. A per-reply
        override does not change this value.
    temperature: float
        Current sampling temperature.
    max_new_tokens: int | None
        Current session-level generation-length limit.
    block_unrequested_cjk: bool
        Whether unrequested Chinese, Japanese, and Korean tokens are blocked.
    messages: list[dict[str, str]]
        Ordered in-memory conversation history using the canonical roles.
    input_token_total: int
        Prompt tokens this session has read, over all replies.
    output_token_total: int
        Tokens this session has generated, over all replies; it grows while
        a reply streams, so callers can show usage live.
    last_finish_reason: str | None
        Why the latest generation stopped, such as ``"stop"`` or
        ``"length"``. Set by the first reply; MLX can also report None.
    stop_requested: Callable[[], bool] | None
        Checked after every generated token; when it returns True the reply
        raises ``GenerationCancelled``. None, the default, never stops.
    trace: TraceSink
        Trace sink used by the session.

    Methods
    -------
    configure_generation(settings, *, trace_turn_id=None)
        Update generation settings after validating all supplied values.
    reply(message, *, language_context=None, **generation_options)
        Generate and retain a reply, with optional original-user context.
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

    With ``block_unrequested_cjk``, token suppression reduces unsolicited
    CJK output. Only original user messages authorize those scripts, by
    using them or positively requesting them. Later negative directives or
    explicit switches to English override earlier authorization. Merely
    discussing Japanese, or requesting Latin-script pinyin, does not grant
    permission. System instructions and assistant replies never grant it.
    Supply ``reply(language_context=...)`` when the prompt contains evidence
    or generated instructions; RAG and tool adapters do this automatically.

    Suppression is best-effort, not a language guarantee: script detection
    uses curated Unicode ranges and a deterministic request heuristic, and
    characters assembled from multiple byte-level tokens can escape the
    standalone-token mask. Ordinary punctuation and fullwidth Latin are not
    suppressed. Disable the setting for workflows requiring unrestricted
    multilingual quotations or tokenizer-specific handling.

    """

    model_id: str
    backend: SelectedBackend
    device: str
    thinking: bool
    temperature: float
    max_new_tokens: int | None
    block_unrequested_cjk: bool
    messages: list[dict[str, str]]
    _system_prompt: str | None
    _model: Model | None
    _tokenizer: Tokenizer | None

    # Token totals start at zero; replies add to them on the instance.
    input_token_total: int = 0
    output_token_total: int = 0
    # Set for one answer by its caller, so Stop can end any reply.
    stop_requested: Callable[[], bool] | None = None

    def __init__(
        self: ChatSession,
        model_id: str,
        *,
        backend: Backend = BACKEND_AUTO,
        thinking: bool = False,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
        system_prompt: str | None = None,
        allow_download: bool = True,
        revision: str | None = None,
        trace: TraceSink | None = None,
        block_unrequested_cjk: bool = True,
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
            shared KíLín runtime utilities. Defaults to ``'auto'``.
        thinking: bool, optional
            Whether a compatible chat template should enable thinking output.
            Defaults to ``False``.
        temperature: float, optional
            Sampling temperature. Zero selects greedy decoding; positive values
            enable sampling. This is not a cross-device reproducibility
            guarantee. The value must be finite and nonnegative.
            Defaults to ``0.0``.
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
        block_unrequested_cjk: bool, optional
            Best-effort suppression of unsolicited Chinese, Japanese, and
            Korean script tokens. Defaults to ``True``.

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
        self.block_unrequested_cjk = _check_bool(
            block_unrequested_cjk, "block_unrequested_cjk"
        )
        self._cjk_token_ids: tuple[int, ...] | None = None
        self._block_cjk = False
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

            if self.backend == BACKEND_MLX:
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
        thinking: bool | None = None,
        trace_turn_id: str | None = None,
        token_callback: Callable[[str], None] | None = None,
        language_context: Sequence[str] | None = None,
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
        thinking: bool | None, optional
            Per-reply thinking override passed to a compatible chat
            template. ``None`` inherits the session setting; ``True`` or
            ``False`` overrides it for this reply only. Does not change
            ``self.thinking``, including on failure. Defaults to ``None``.
        trace_turn_id: str | None, optional
            Optional trace identifier associating emitted events and artifacts
            with the current conversation turn. Defaults to ``None``.
        token_callback: Callable[[str], None] | None, optional
            Receives decoded text pieces as they are generated, for live
            display. MLX streams; PyTorch calls it once with the whole reply.
            Its exceptions propagate. Defaults to ``None``.
        language_context: Sequence[str] | None, optional
            Original user-authored messages, oldest first with the current
            question last, used only for CJK permission. Supply this when
            ``message`` contains retrieved or generated material. An empty
            sequence grants no permission. ``None`` uses user-role history
            plus ``message``, appropriate for direct chat. Defaults to
            ``None``. System and assistant messages are never inspected.

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
            Also raised when ``thinking`` is neither a bool nor None,
            ``token_callback`` is not callable, or
            ``language_context`` is not a sequence of strings.

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

        if token_callback is not None and not callable(token_callback):
            raise ValueError("token_callback must be callable or None.")

        effective_thinking = (
            self.thinking if thinking is None
            else _check_bool(thinking, "thinking")
        )

        if language_context is None:
            user_texts = [
                item["content"] for item in self.messages
                if item["role"] == USER_ROLE
            ] + [message]
        elif (
            isinstance(language_context, (str, bytes))
            or not isinstance(language_context, Sequence)
            or not all(isinstance(text, str) for text in language_context)
        ):
            raise ValueError("language_context must be a sequence of text.")
        else:
            user_texts = language_context

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
                    enable_thinking=effective_thinking,
                )
            except (TypeError, ValueError):
                rendered_prompt = None

        tokens = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=effective_thinking,
        )
        token_budget = token_override or self.max_new_tokens
        if token_budget is None:
            token_budget = self._remaining_context(len(tokens))

        self._block_cjk = (
            self.block_unrequested_cjk and not _allows_cjk(user_texts)
        )
        blocked = self._blocked_token_ids()
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
            "thinking": effective_thinking,
            "generation": {
                "temperature": self.temperature,
                "max_new_tokens": token_budget,
                "sampling": self.temperature > 0,
                "cjk_blocking_requested": self._block_cjk,
                "cjk_tokens_blocked": bool(blocked),
                "cjk_suppressed_token_count": len(blocked),
            },
            "chat_template": {
                "add_generation_prompt": True,
                "enable_thinking": effective_thinking,
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
            TRACE_EVENT_LLM_CALL_INPUT,
            component="chat",
            data=input_data,
            turn_id=trace_turn_id,
        )

        self._last_generation_output_token_count = None
        self._last_generation_token_count_source = None
        self.last_finish_reason = None
        self.input_token_total += len(tokens)
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
            if self.backend == BACKEND_MLX:
                if self.trace.enabled:
                    answer = self._mlx_reply(
                        tokens,
                        token_budget,
                        trace_turn_id=trace_turn_id,
                        token_callback=token_callback,
                    )
                elif token_callback is not None:
                    answer = self._mlx_reply(
                        tokens, token_budget, token_callback=token_callback
                    )
                else:
                    answer = self._mlx_reply(tokens, token_budget)
            else:
                if self.trace.enabled:
                    answer = self._torch_reply(
                        tokens,
                        token_budget,
                        trace_turn_id=trace_turn_id,
                    )
                else:
                    answer = self._torch_reply(tokens, token_budget)
                if token_callback is not None and isinstance(answer, str):
                    # PyTorch generation is not streamed, so the callback
                    # receives the whole reply at once.
                    token_callback(answer)
        generation_duration_ms = (
            perf_counter() - generation_started_at
        ) * 1000

        if not isinstance(answer, str) or not answer.strip():
            raise IncompleteModelResponse(
                "The model returned no text.", reason="empty_answer",
                finish_reason=self.last_finish_reason,
            )

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
            "finish_reason": self.last_finish_reason,
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
            TRACE_EVENT_LLM_CALL_OUTPUT,
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

        if self.backend == BACKEND_MLX:
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
        token_callback: Callable[[str], None] | None = None,
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
        token_callback: Callable[[str], None] | None, optional
            Receives each decoded text piece as it is generated. Defaults to
            ``None``.

        Returns
        -------
        str
            Decoded model output.

        Notes
        -----
        Updates ``output_token_total`` after each token and records the
        generated token count and ``last_finish_reason``.

        """

        mlx_lm = import_module("mlx_lm")
        sampling = import_module("mlx_lm.sample_utils")
        sampler = sampling.make_sampler(temp=self.temperature)

        options: dict[str, object] = {}
        blocked = self._blocked_token_ids()
        if blocked:
            options["logits_processors"] = [_mlx_token_blocker(blocked)]

        parts: list[str] = []
        generated = 0
        for response in mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=tokens,
            max_tokens=token_budget,
            sampler=sampler,
            **options,
        ):
            # Each response is one generated token, even when its text is
            # empty (part of a character still being decoded).
            generated += 1
            self.output_token_total += 1
            parts.append(response.text)
            self.last_finish_reason = response.finish_reason
            if token_callback is not None and response.text:
                token_callback(response.text)
            if self._stopped():
                raise GenerationCancelled("The answer was stopped.")

        self._last_generation_output_token_count = generated
        self._last_generation_token_count_source = "generated_tokens"
        return "".join(parts)

    def _stopped(self: ChatSession) -> bool:
        """Return whether the caller asked to stop the current reply.

        Returns
        -------
        bool
            True once ``stop_requested`` returns True.

        """

        return self.stop_requested is not None and bool(self.stop_requested())

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
        blocked = self._blocked_token_ids()
        if blocked:
            options["suppress_tokens"] = list(blocked)
        if self.stop_requested is not None:

            def stop(input_ids, scores, **kwargs):
                """End generation for every sequence once Stop is pressed.

                Parameters
                ----------
                input_ids: torch.Tensor
                    Prompt and generated token IDs, one row per sequence.
                scores: torch.Tensor
                    Next-token scores; not used.
                **kwargs: object
                    Further arguments from ``generate``; not used.

                Returns
                -------
                torch.Tensor
                    One Boolean per sequence, True once stopped.

                """

                return torch.full(
                    (input_ids.shape[0],), self._stopped(),
                    dtype=torch.bool, device=input_ids.device,
                )

            options["stopping_criteria"] = [stop]
        with torch.inference_mode():
            output = model.generate(**options)
        if self._stopped():
            raise GenerationCancelled("The answer was stopped.")
        generated = output[0, len(tokens) :].tolist()
        self.output_token_total += len(generated)

        # A reply cut by the token budget did not end with an end-of-sequence
        # token; the model's generation config may list several of them.
        eos = getattr(
            getattr(model, "generation_config", None), "eos_token_id", None
        )
        if eos is None:
            eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        ended_with_eos = bool(generated) and generated[-1] in eos_ids
        reached_budget = len(generated) >= token_budget
        self.last_finish_reason = (
            "length" if reached_budget and not ended_with_eos else "stop"
        )
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
            TRACE_EVENT_LLM_CALL_OUTPUT_TOKENS,
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
        self._cjk_token_ids = None
        self.reset()
        gc.collect()
        if self.backend == BACKEND_MLX:
            import_module("mlx.core").clear_cache()
        elif self.device == "cuda":
            import_module("torch").cuda.empty_cache()
        elif self.device == "mps":
            import_module("torch").mps.empty_cache()

    def _blocked_token_ids(self: ChatSession) -> tuple[int, ...]:
        """Return the token IDs the next reply must not generate.

        Returns
        -------
        tuple[int, ...]
            IDs of vocabulary tokens containing Chinese, Japanese, or Korean
            characters when blocking applies to the current input;
            otherwise an empty tuple.

        Notes
        -----
        The IDs are found once per loaded tokenizer by decoding every
        vocabulary entry. A tokenizer without a ``get_vocab`` mapping
        blocks nothing. Characters split across several byte-level tokens
        are not blocked. No assumption is made about how often a tokenizer
        uses that representation. Special tokens are never suppressed.

        """

        if not getattr(self, "_block_cjk", False):
            return ()
        if getattr(self, "_cjk_token_ids", None) is None:
            self._cjk_token_ids = _cjk_token_ids(self._tokenizer)
        return self._cjk_token_ids

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
        Finite, nonnegative temperature converted to ``float``.

    Raises
    ------
    ValueError
        If ``value`` is Boolean, nonnumeric, negative, or nonfinite, or
        cannot be represented as a finite float.

    """

    message = "Temperature must be a finite, nonnegative number."

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)

    try:
        temperature = float(value)
    except OverflowError:
        raise ValueError(message) from None

    if not isfinite(temperature) or temperature < 0:
        raise ValueError(message)

    return temperature


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


def _allows_cjk(user_messages: Sequence[str]) -> bool:
    """Resolve best-effort CJK permission from user-authored text only.

    Parameters
    ----------
    user_messages: Sequence[str]
        Original user messages in chronological order. Retrieved material,
        system instructions, and assistant output must not be included.

    Returns
    -------
    bool
        Whether CJK output was authorized. Later explicit directives
        supersede earlier ones. A negative or non-CJK directive overrides
        script characters quoted in the same message; an affirmative CJK
        directive after it can explicitly reauthorize output.

    Notes
    -----
    Ordinary mentions of a language are not requests to use its script.
    This deterministic English-directive heuristic is not a general natural
    language interpreter. Unspecified follow-ups retain the latest user
    decision, including authorization from actual script use.

    """

    allowed = False
    for text in user_messages:
        negative = list(_CJK_NEGATIVE.finditer(text))
        decisions = [(match.end(), False) for match in negative]
        decisions.extend(
            (match.end(), False) for match in _NON_CJK_REQUEST.finditer(text)
        )
        decisions.extend(
            (match.end(), True)
            for match in _CJK_REQUEST.finditer(text)
            if not any(
                item.start() <= match.start() < item.end()
                for item in negative
            )
        )
        if decisions:
            # On an exact tie the restrictive directive wins.
            allowed = max(decisions, key=lambda item: (item[0], not item[1]))[
                1
            ]
        elif _CJK.search(text):
            allowed = True
    return allowed


def _cjk_token_ids(tokenizer: object) -> tuple[int, ...]:
    """Find vocabulary tokens that contain Chinese, Japanese, or Korean text.

    Parameters
    ----------
    tokenizer: object
        Loaded tokenizer with ``get_vocab`` and ``decode``.

    Returns
    -------
    tuple[int, ...]
        Sorted token IDs whose decoded text contains such a character, or
        an empty tuple when the vocabulary is unavailable.

    """

    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocabulary = get_vocab() if callable(get_vocab) else None
    if not isinstance(vocabulary, dict):
        return ()
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return ()
    special = getattr(tokenizer, "all_special_ids", ())
    special = (
        set(special) if isinstance(special, (list, tuple, set)) else set()
    )
    token_ids = {
        value for value in vocabulary.values()
        if type(value) is int and value >= 0 and value not in special
    }
    found = []
    for token_id in sorted(token_ids):
        text = decode([token_id])
        if isinstance(text, str) and _CJK.search(text):
            found.append(token_id)
    return tuple(found)


def _mlx_token_blocker(
    token_ids: tuple[int, ...],
) -> Callable[[object, object], object]:
    """Return an MLX logits processor that blocks the given tokens.

    Parameters
    ----------
    token_ids: tuple[int, ...]
        Token IDs that must not be generated.

    Returns
    -------
    Callable[[object, object], object]
        Processor taking the tokens so far and the next-token logits, and
        returning the logits with the blocked tokens set to minus infinity.

    """

    mx = import_module("mlx.core")
    numpy = import_module("numpy")
    penalties: dict[tuple[int, str], object] = {}

    def block(tokens: object, logits: object) -> object:
        """Set the logits of blocked tokens to minus infinity.

        Parameters
        ----------
        tokens: object
            Tokens generated so far (unused).
        logits: object
            Next-token logits of shape ``(..., vocabulary)``.

        Returns
        -------
        object
            Logits with the blocked tokens excluded.

        """

        size = int(logits.shape[-1])  # type: ignore[attr-defined]
        dtype = logits.dtype  # type: ignore[attr-defined]
        key = (size, str(dtype))
        if key not in penalties:
            penalty = numpy.zeros(size, dtype=numpy.float32)
            inside = [token for token in token_ids if 0 <= token < size]
            penalty[inside] = -numpy.inf
            penalties[key] = mx.array(penalty, dtype=dtype)
        return logits + penalties[key]  # type: ignore[operator]

    return block
