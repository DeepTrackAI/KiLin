"""Unit tests for local chat sessions and generation helpers.

Tests isolate model loading and backend inference to verify conversation
state, validation, token accounting, tracing, and resource cleanup.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from kilin import chat
from kilin.trace_logging import (
    TRACE_EVENT_LLM_CALL_INPUT,
    TRACE_EVENT_LLM_CALL_OUTPUT,
    TRACE_EVENT_LLM_CALL_OUTPUT_TOKENS,
)


def _session(**settings: object) -> chat.ChatSession:
    """Create a loaded-looking chat without importing a model backend.

    The mock tokenizer provides three prompt tokens. Callers may replace
    its behavior or the model to exercise specific session paths.

    """

    with patch.object(chat.ChatSession, "_load"):
        session = chat.ChatSession(
            "group/model",
            backend="torch",
            **settings,
        )
    session._model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=32)
    )
    session._tokenizer = Mock()
    session._tokenizer.apply_chat_template.return_value = [4, 5, 6]
    session.device = "cpu"
    return session


class TestChatSessionInitialization(unittest.TestCase):
    """Test session defaults, constructor settings, and model loading."""

    def test_default_settings_and_initial_history(self) -> None:
        """Preserve the defaults and begin with an empty conversation."""

        with (
            patch.object(chat, "_choose_backend", return_value="torch")
            as choose,
            patch.object(chat.ChatSession, "_load") as load,
        ):
            session = chat.ChatSession("group/model")

        choose.assert_called_once_with("auto")
        load.assert_called_once_with(allow_download=True, revision=None)
        self.assertEqual(session.backend, "torch")
        self.assertEqual(session.model_id, "group/model")
        self.assertFalse(session.thinking)
        self.assertEqual(session.temperature, 0.0)
        self.assertIsNone(session.max_new_tokens)
        self.assertEqual(session.messages, [])
        self.assertEqual(session.input_token_total, 0)
        self.assertEqual(session.output_token_total, 0)
        self.assertIsNone(session._model)
        self.assertIsNone(session._tokenizer)
        self.assertIsInstance(session.trace, chat.NullTraceLogger)

    def test_normalizes_prompt_and_forwards_loader_options(self) -> None:
        """Retain a stripped system instruction and loading options."""

        trace = Mock()
        with patch.object(chat.ChatSession, "_load") as load:
            session = chat.ChatSession(
                "group/model",
                backend="torch",
                thinking=True,
                temperature=0.25,
                max_new_tokens=12,
                system_prompt="  Answer briefly.  ",
                allow_download=False,
                revision="abc123",
                trace=trace,
            )

        load.assert_called_once_with(
            allow_download=False, revision="abc123"
        )
        self.assertIs(session.trace, trace)
        self.assertTrue(session.thinking)
        self.assertEqual(session.temperature, 0.25)
        self.assertEqual(session.max_new_tokens, 12)
        self.assertEqual(
            session.messages,
            [{"role": chat.SYSTEM_ROLE, "content": "Answer briefly."}],
        )

    def test_rejects_invalid_model_ids_before_loading(self) -> None:
        """Reject empty and non-string identifiers without model I/O."""

        with patch.object(chat.ChatSession, "_load") as load:
            for model_id in ("", "  ", None, 17):
                with self.subTest(model_id=model_id):
                    with self.assertRaises(ValueError):
                        chat.ChatSession(model_id, backend="torch")
        load.assert_not_called()

    def test_rejects_invalid_constructor_settings(self) -> None:
        """Validate settings before attempting to load the model."""

        invalid = (
            {"thinking": "yes"},
            {"thinking": 1},
            {"temperature": -1},
            {"temperature": True},
            {"temperature": "0.5"},
            {"temperature": float("nan")},
            {"temperature": float("inf")},
            {"temperature": float("-inf")},
            {"temperature": 10**400},
            {"max_new_tokens": 0},
            {"max_new_tokens": True},
            {"max_new_tokens": 1.5},
            {"system_prompt": "  "},
            {"system_prompt": 5},
            {"allow_download": "no"},
        )
        with patch.object(chat.ChatSession, "_load") as load:
            for settings in invalid:
                with self.subTest(settings=settings):
                    with self.assertRaises(ValueError):
                        chat.ChatSession(
                            "group/model", backend="torch", **settings
                        )
        load.assert_not_called()

    def test_invalid_backend_prevents_loading(self) -> None:
        """Delegate invalid backend selection to the runtime policy."""

        with patch.object(chat.ChatSession, "_load") as load:
            with self.assertRaises(ValueError):
                chat.ChatSession("group/model", backend="invalid")
        load.assert_not_called()


class TestChatModelLoading(unittest.TestCase):
    """Test source resolution and lazy backend loading using fake modules."""

    @staticmethod
    def _torch_dependencies() -> tuple[Mock, Mock, Mock, Mock]:
        """Provide minimal PyTorch and Transformers loader interfaces."""

        torch = Mock()
        torch.cuda.is_available.return_value = False
        torch.backends.mps.is_available.return_value = False
        transformers = Mock()
        model = Mock()
        tokenizer = Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        transformers.AutoModelForCausalLM.from_pretrained.return_value = model
        return torch, transformers, model, tokenizer

    def test_torch_loads_remote_identifier_with_requested_revision(
        self,
    ) -> None:
        """Forward model source and online options to both loaders."""

        torch, transformers, model, tokenizer = self._torch_dependencies()
        modules = {"torch": torch, "transformers": transformers}
        with (
            patch.object(chat, "import_module", side_effect=modules.get),
            patch.object(chat, "_torch_device", return_value="cpu")
            as device,
        ):
            session = chat.ChatSession(
                "group/model",
                backend="torch",
                revision="main",
            )

        arguments = {
            "local_files_only": False,
            "revision": "main",
            "trust_remote_code": False,
        }
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "group/model", **arguments
        )
        model_loader = transformers.AutoModelForCausalLM.from_pretrained
        model_loader.assert_called_once_with(
            "group/model", dtype="auto", **arguments
        )
        device.assert_called_once_with(False, False)
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()
        self.assertIs(session._model, model)
        self.assertIs(session._tokenizer, tokenizer)
        self.assertEqual(session.model_source, "group/model")
        self.assertEqual(session.device, "cpu")

    def test_torch_resolves_cached_model_when_offline(self) -> None:
        """Resolve a Hub ID locally and remove its revision from loaders."""

        torch, transformers, _, _ = self._torch_dependencies()
        modules = {"torch": torch, "transformers": transformers}
        source = Path("/cache/snapshots/revision")
        with (
            patch.object(chat, "import_module", side_effect=modules.get),
            patch.object(
                chat, "cached_model_snapshot", return_value=source
            ) as snapshot,
        ):
            session = chat.ChatSession(
                "group/model",
                backend="torch",
                allow_download=False,
                revision="commit",
            )

        snapshot.assert_called_once_with("group/model", revision="commit")
        self.assertEqual(session.model_source, str(source))
        for loader in (
            transformers.AutoTokenizer.from_pretrained,
            transformers.AutoModelForCausalLM.from_pretrained,
        ):
            self.assertEqual(loader.call_args.args, (str(source),))
            self.assertTrue(loader.call_args.kwargs["local_files_only"])
            self.assertIsNone(loader.call_args.kwargs["revision"])
            self.assertFalse(loader.call_args.kwargs["trust_remote_code"])

    def test_local_directory_does_not_resolve_hub_snapshot(self) -> None:
        """Use a local directory directly even in offline mode."""

        torch, transformers, _, _ = self._torch_dependencies()
        modules = {"torch": torch, "transformers": transformers}
        with TemporaryDirectory() as directory:
            with (
                patch.object(
                    chat, "import_module", side_effect=modules.get
                ),
                patch.object(chat, "cached_model_snapshot") as snapshot,
            ):
                session = chat.ChatSession(
                    directory,
                    backend="torch",
                    allow_download=False,
                    revision="main",
                )
            snapshot.assert_not_called()
            self.assertEqual(session.model_source, directory)
            self.assertEqual(
                transformers.AutoTokenizer.from_pretrained.call_args.args,
                (directory,),
            )

    def test_mlx_loads_model_after_metal_check(self) -> None:
        """Load the MLX model lazily and report its Metal device."""

        mx = Mock()
        mx.metal.is_available.return_value = True
        mlx_lm = Mock()
        model, tokenizer = Mock(), Mock()
        mlx_lm.load.return_value = (model, tokenizer)
        modules = {"mlx.core": mx, "mlx_lm": mlx_lm}
        with (
            patch.object(chat, "_choose_backend", return_value="mlx"),
            patch.object(chat, "import_module", side_effect=modules.get),
        ):
            session = chat.ChatSession("group/model", backend="mlx")

        mx.metal.is_available.assert_called_once_with()
        mlx_lm.load.assert_called_once_with(
            "group/model",
            tokenizer_config={"trust_remote_code": False},
            revision=None,
        )
        self.assertEqual(session.device, "metal")
        self.assertIs(session._model, model)
        self.assertIs(session._tokenizer, tokenizer)

    def test_mlx_rejects_unavailable_metal_before_loading(self) -> None:
        """Fail explicitly without calling the model loader."""

        mx = Mock()
        mx.metal.is_available.return_value = False
        mlx_lm = Mock()
        modules = {"mlx.core": mx, "mlx_lm": mlx_lm}
        with (
            patch.object(chat, "_choose_backend", return_value="mlx"),
            patch.object(chat, "import_module", side_effect=modules.get),
        ):
            with self.assertRaisesRegex(RuntimeError, "Metal"):
                chat.ChatSession("group/model", backend="mlx")
        mlx_lm.load.assert_not_called()

    def test_missing_backend_dependency_propagates(self) -> None:
        """Do not suppress errors importing a selected model backend."""

        with patch.object(
            chat, "import_module", side_effect=ModuleNotFoundError("torch")
        ):
            with self.assertRaises(ModuleNotFoundError):
                chat.ChatSession("group/model", backend="torch")


class TestGenerationSettings(unittest.TestCase):
    """Test updating generation settings without reloading a model."""

    def test_updates_only_requested_settings(self) -> None:
        """Keep omitted settings, the loaded model, and history unchanged."""

        session = _session(system_prompt="Instruction", temperature=0.1)
        model = session._model
        history = session.messages.copy()
        session.configure_generation({"thinking": True, "max_new_tokens": 9})
        self.assertEqual(session.temperature, 0.1)
        self.assertTrue(session.thinking)
        self.assertEqual(session.max_new_tokens, 9)
        self.assertIs(session._model, model)
        self.assertEqual(session.messages, history)
        session.configure_generation({"max_new_tokens": None})
        self.assertIsNone(session.max_new_tokens)

    def test_invalid_updates_are_atomic(self) -> None:
        """Reject all changes when any updated setting is invalid."""

        session = _session(
            temperature=0.25, thinking=True, max_new_tokens=12
        )
        initial = (
            session.temperature, session.thinking, session.max_new_tokens
        )
        invalid = (
            None,
            [],
            {"backend": "mlx"},
            {"temperature": -1},
            {"temperature": float("nan")},
            {"temperature": float("inf")},
            {"temperature": float("-inf")},
            {"temperature": 10**400},
            {"thinking": "true"},
            {"max_new_tokens": 0},
            {"temperature": 0.75, "thinking": "true"},
            {"thinking": False, "max_new_tokens": -1},
        )
        for settings in invalid:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    session.configure_generation(settings)
                self.assertEqual(
                    (session.temperature, session.thinking,
                     session.max_new_tokens),
                    initial,
                )

    def test_settings_change_is_reported_with_turn_id(self) -> None:
        """Report previous and updated settings to the supplied sink."""

        trace = Mock()
        session = _session(trace=trace, max_new_tokens=5)
        session.configure_generation(
            {"temperature": 0.7}, trace_turn_id="turn-1"
        )
        trace.settings_changed.assert_called_once_with(
            "generation.settings.changed",
            component="chat",
            before={
                "temperature": 0.0,
                "thinking": False,
                "max_new_tokens": 5,
            },
            after={
                "temperature": 0.7,
                "thinking": False,
                "max_new_tokens": 5,
            },
            turn_id="turn-1",
        )

    def test_trace_error_does_not_rollback_settings(self) -> None:
        """Document settings committed before a tracing failure."""

        trace = Mock()
        trace.settings_changed.side_effect = OSError("trace unavailable")
        session = _session(trace=trace)
        with self.assertRaisesRegex(OSError, "trace unavailable"):
            session.configure_generation({"temperature": 0.5})
        self.assertEqual(session.temperature, 0.5)


class TestConversationHistory(unittest.TestCase):
    """Test initial, restored, and reset conversation histories."""

    def test_restore_normalizes_and_copies_history(self) -> None:
        """Strip content, omit extras, and keep an independent copy."""

        session = _session(system_prompt="Original")
        incoming = [
            {"role": chat.SYSTEM_ROLE, "content": "  New  ", "id": "1"},
            {"role": chat.USER_ROLE, "content": "  Hello ", "id": "2"},
            {"role": chat.ASSISTANT_ROLE, "content": " Hi  "},
        ]
        session.restore_history(incoming)
        self.assertEqual(
            session.messages,
            [
                {"role": chat.SYSTEM_ROLE, "content": "New"},
                {"role": chat.USER_ROLE, "content": "Hello"},
                {"role": chat.ASSISTANT_ROLE, "content": "Hi"},
            ],
        )
        incoming[1]["content"] = "Changed"
        self.assertEqual(session.messages[1]["content"], "Hello")
        self.assertIsNot(session.messages[1], incoming[1])

    def test_restore_accepts_empty_and_non_alternating_history(self) -> None:
        """Permit empty history and repeated non-system roles."""

        session = _session(system_prompt="Initial")
        session.restore_history([])
        self.assertEqual(session.messages, [])
        session.restore_history(
            [
                {"role": chat.USER_ROLE, "content": "one"},
                {"role": chat.USER_ROLE, "content": "two"},
            ]
        )
        self.assertEqual(len(session.messages), 2)

    def test_invalid_restoration_preserves_existing_history(self) -> None:
        """Reject malformed histories without committing partial changes."""

        session = _session(system_prompt="Initial")
        original = session.messages.copy()
        invalid = (
            None,
            {},
            "text",
            ["message"],
            [{}],
            [{"role": "tool", "content": "text"}],
            [{"role": chat.USER_ROLE, "content": ""}],
            [{"role": chat.USER_ROLE, "content": "  "}],
            [{"role": chat.USER_ROLE, "content": 5}],
            [
                {"role": chat.USER_ROLE, "content": "Valid"},
                {"role": chat.SYSTEM_ROLE, "content": "Late"},
            ],
        )
        for history in invalid:
            with self.subTest(history=history):
                with self.assertRaises(ValueError):
                    session.restore_history(history)
                self.assertEqual(session.messages, original)

    def test_unhashable_history_role_propagates_type_error(self) -> None:
        """Retain the current role-validation contract for list values."""

        session = _session()
        with self.assertRaises(TypeError):
            session.restore_history([{"role": [], "content": "text"}])

    def test_reset_restores_constructor_prompt_only(self) -> None:
        """Restore original instructions without changing model settings."""

        session = _session(system_prompt="  Original ", temperature=0.3)
        model = session._model
        session.restore_history(
            [{"role": chat.SYSTEM_ROLE, "content": "Replacement"}]
        )
        session.configure_generation({"thinking": True})
        session.reset()
        self.assertEqual(
            session.messages,
            [{"role": chat.SYSTEM_ROLE, "content": "Original"}],
        )
        self.assertIs(session._model, model)
        self.assertTrue(session.thinking)
        self.assertEqual(session.temperature, 0.3)


class TestChatReply(unittest.TestCase):
    """Test reply orchestration independently of backend generation."""

    def setUp(self) -> None:
        """Initialize a chat with a fake loaded model and tokenizer."""

        self.session = _session(system_prompt="Instruction")

    def test_successful_reply_updates_history(self) -> None:
        """Commit exactly one stripped user/assistant exchange."""

        with patch.object(
            self.session, "_torch_reply", return_value="  Answer.  "
        ) as generate:
            answer = self.session.reply("  Question?  ")

        self.assertEqual(answer, "Answer.")
        self.assertEqual(
            self.session.messages,
            [
                {"role": chat.SYSTEM_ROLE, "content": "Instruction"},
                {"role": chat.USER_ROLE, "content": "Question?"},
                {"role": chat.ASSISTANT_ROLE, "content": "Answer."},
            ],
        )
        self.assertEqual(self.session.input_token_total, 3)
        self.assertEqual(generate.call_args.args, ([4, 5, 6], 29))
        self.assertEqual(
            self.session._tokenizer.apply_chat_template.call_args.kwargs,
            {
                "tokenize": True,
                "return_dict": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )

    def test_follow_up_uses_retained_messages(self) -> None:
        """Provide prior history to the tokenizer on each new turn."""

        with patch.object(
            self.session, "_torch_reply", side_effect=["First", "Second"]
        ):
            self.session.reply("One")
            self.session.reply("Two")

        calls = self.session._tokenizer.apply_chat_template.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [item["role"] for item in calls[-1].args[0]],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(self.session.input_token_total, 6)
        self.assertEqual(len(self.session.messages), 5)

    def test_rejects_invalid_message_and_callback(self) -> None:
        """Reject invalid requests before tokenization or generation."""

        with patch.object(self.session, "_torch_reply") as generate:
            for message in ("", "  ", None, 7):
                with self.subTest(message=message):
                    with self.assertRaises(ValueError):
                        self.session.reply(message)
            with self.assertRaisesRegex(ValueError, "token_callback"):
                self.session.reply("Hello", token_callback=7)
        generate.assert_not_called()
        self.session._tokenizer.apply_chat_template.assert_not_called()
        self.assertEqual(len(self.session.messages), 1)

    def test_rejects_invalid_per_reply_token_limits(self) -> None:
        """Disallow invalid overrides without starting model generation."""

        with patch.object(self.session, "_torch_reply") as generate:
            for limit in (0, -1, True, 1.5, "3"):
                with self.subTest(limit=limit):
                    with self.assertRaises(ValueError):
                        self.session.reply("Hello", max_new_tokens=limit)
        generate.assert_not_called()
        self.assertEqual(len(self.session.messages), 1)

    def test_per_reply_limit_overrides_session_and_context(self) -> None:
        """Prioritize explicit limits without clamping to context size."""

        self.session.configure_generation({"max_new_tokens": 8})
        with patch.object(
            self.session, "_torch_reply", return_value="Answer"
        ) as generate:
            self.session.reply("Hello", max_new_tokens=50)
            self.assertEqual(generate.call_args.args, ([4, 5, 6], 50))
            self.session.reply("Again")
            self.assertEqual(generate.call_args.args, ([4, 5, 6], 8))

    def test_blank_or_non_string_answers_leave_history_unchanged(
        self,
    ) -> None:
        """Treat whitespace-only or non-text outputs as failed turns."""

        for answer in ("", "  ", None, 5):
            with self.subTest(answer=answer):
                with patch.object(
                    self.session, "_torch_reply", return_value=answer
                ):
                    with self.assertRaisesRegex(RuntimeError, "no text"):
                        self.session.reply("Hello")
                self.assertEqual(len(self.session.messages), 1)

    def test_generation_exception_does_not_commit_history(self) -> None:
        """Propagate inference failures while leaving history unchanged."""

        with patch.object(
            self.session,
            "_torch_reply",
            side_effect=RuntimeError("generation failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                self.session.reply("Hello")
        self.assertEqual(len(self.session.messages), 1)
        self.assertEqual(self.session.input_token_total, 3)

    def test_torch_callback_receives_complete_response(self) -> None:
        """Send the unstripped whole PyTorch reply in one callback."""

        callback = Mock()
        with patch.object(
            self.session, "_torch_reply", return_value="  Answer  "
        ):
            self.assertEqual(
                self.session.reply("Hello", token_callback=callback),
                "Answer",
            )
        callback.assert_called_once_with("  Answer  ")

    def test_callback_exception_does_not_commit_history(self) -> None:
        """Propagate callback failures before saving the completed turn."""

        callback = Mock(side_effect=RuntimeError("callback failed"))
        with patch.object(
            self.session, "_torch_reply", return_value="Answer"
        ):
            with self.assertRaisesRegex(RuntimeError, "callback failed"):
                self.session.reply("Hello", token_callback=callback)
        self.assertEqual(len(self.session.messages), 1)

    def test_mlx_dispatch_forwards_streaming_callback(self) -> None:
        """Dispatch to the streaming MLX interface when selected."""

        self.session.backend = "mlx"
        self.session._model = SimpleNamespace(
            args=SimpleNamespace(max_position_embeddings=32)
        )
        callback = Mock()
        with patch.object(
            self.session, "_mlx_reply", return_value="Streamed"
        ) as generate:
            answer = self.session.reply("Hello", token_callback=callback)
        self.assertEqual(answer, "Streamed")
        generate.assert_called_once_with(
            [4, 5, 6], 29, token_callback=callback
        )
        callback.assert_not_called()

    def test_mlx_dispatch_without_callback_or_trace(self) -> None:
        """Pass only tokens and budget for untraced MLX generation."""

        self.session.backend = "mlx"
        self.session._model = SimpleNamespace(
            args=SimpleNamespace(max_position_embeddings=32)
        )
        with patch.object(
            self.session, "_mlx_reply", return_value="Answer"
        ) as generate:
            self.assertEqual(self.session.reply("Hello"), "Answer")
        generate.assert_called_once_with([4, 5, 6], 29)

    def test_mlx_dispatch_includes_trace_turn(self) -> None:
        """Forward trace correlation and callbacks to MLX generation."""

        trace = Mock()
        trace.enabled = True
        trace.include_content = False
        trace.include_token_ids = False
        trace.span.side_effect = lambda *args, **kwargs: nullcontext()
        self.session.trace = trace
        self.session.backend = "mlx"
        self.session._model = SimpleNamespace(
            args=SimpleNamespace(max_position_embeddings=32)
        )
        callback = Mock()
        with patch.object(
            self.session, "_mlx_reply", return_value="Answer"
        ) as generate:
            self.assertEqual(
                self.session.reply(
                    "Hello", trace_turn_id="turn-1",
                    token_callback=callback,
                ),
                "Answer",
            )
        generate.assert_called_once_with(
            [4, 5, 6], 29, trace_turn_id="turn-1",
            token_callback=callback,
        )
        callback.assert_not_called()

    def test_closed_session_rejects_reply(self) -> None:
        """Fail before accessing tokenizer when resources are released."""

        self.session._model = None
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.session.reply("Hello")
        self.session._tokenizer.apply_chat_template.assert_not_called()


class TestChatContext(unittest.TestCase):
    """Test context-window calculations and token counting."""

    def test_remaining_context_for_both_backends(self) -> None:
        """Read the selected backend's context-size configuration."""

        session = _session()
        self.assertEqual(session._remaining_context(3), 29)
        session.backend = "mlx"
        session._model = SimpleNamespace(
            args=SimpleNamespace(max_position_embeddings=16)
        )
        self.assertEqual(session._remaining_context(3), 13)

    def test_rejects_invalid_and_exhausted_context(self) -> None:
        """Reject missing usable context and exhausted token budgets."""

        session = _session()
        for context in (None, 0, -1, "32", 2.5):
            with self.subTest(context=context):
                session._model.config.max_position_embeddings = context
                with self.assertRaisesRegex(ValueError, "context size"):
                    session._remaining_context(3)
        session._model.config.max_position_embeddings = 3
        with self.assertRaisesRegex(ValueError, "filled"):
            session._remaining_context(3)
        with self.assertRaisesRegex(ValueError, "filled"):
            session._remaining_context(4)

    def test_unavailable_context_attribute_propagates(self) -> None:
        """Expose missing backend context metadata to the caller."""

        session = _session()
        session._model = SimpleNamespace(config=SimpleNamespace())
        with self.assertRaises(AttributeError):
            session._remaining_context(3)

    def test_explicit_limit_avoids_context_lookup(self) -> None:
        """Do not require context metadata with an explicit token limit."""

        session = _session(max_new_tokens=4)
        session._model = SimpleNamespace(config=SimpleNamespace())
        with patch.object(
            session, "_torch_reply", return_value="Answer"
        ) as generate:
            session.reply("Hello")
        generate.assert_called_once_with([4, 5, 6], 4)

    def test_retokenizes_text_with_supported_encoders(self) -> None:
        """Count decoded tokens with both encoder call signatures."""

        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2]
        self.assertEqual(chat._text_token_count(tokenizer, "answer"), 2)
        tokenizer.encode.assert_called_once_with(
            "answer", add_special_tokens=False
        )
        tokenizer.encode.reset_mock()
        tokenizer.encode.side_effect = [TypeError("keyword"), [1, 2, 3]]
        self.assertEqual(chat._text_token_count(tokenizer, "answer"), 3)
        self.assertEqual(tokenizer.encode.call_count, 2)
        tokenizer.encode.assert_any_call("answer")

    def test_token_count_returns_none_when_unavailable(self) -> None:
        """Treat supported tokenizer failures as unknown counts."""

        self.assertIsNone(chat._text_token_count(object(), "answer"))
        tokenizer = Mock()
        tokenizer.encode.side_effect = ValueError("encode")
        self.assertIsNone(chat._text_token_count(tokenizer, "answer"))
        tokenizer.encode.side_effect = TypeError("bad")
        self.assertIsNone(chat._text_token_count(tokenizer, "answer"))


class TestTorchGeneration(unittest.TestCase):
    """Test PyTorch token generation using an isolated tensor interface."""

    @staticmethod
    def _torch_inputs(generated: list[int]) -> tuple[Mock, Mock, Mock]:
        """Create fake tensor operations, model output, and tokenizer."""

        torch = Mock()
        torch.long = "long"
        torch.inference_mode.return_value = nullcontext()
        tensor = object()
        mask = object()
        torch.tensor.return_value = tensor
        torch.ones_like.return_value = mask
        output = MagicMock()
        output.__getitem__.return_value.tolist.return_value = generated
        model = Mock()
        model.generate.return_value = output
        model.generation_config.eos_token_id = 99
        return torch, model, output

    def test_greedy_generation_tracks_tokens_and_stop_reason(self) -> None:
        """Decode new IDs, account for tokens, and recognize EOS."""

        session = _session()
        torch, model, _ = self._torch_inputs([7, 99])
        session._model = model
        session._tokenizer.eos_token_id = 99
        session._tokenizer.decode.return_value = "Answer"
        with patch.object(chat, "import_module", return_value=torch):
            answer = session._torch_reply([4, 5], 4)

        self.assertEqual(answer, "Answer")
        self.assertEqual(session.output_token_total, 2)
        self.assertEqual(session.last_finish_reason, "stop")
        self.assertEqual(session._last_generation_output_token_count, 2)
        self.assertEqual(
            session._last_generation_token_count_source,
            "generated_token_ids",
        )
        self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], 4)
        self.assertFalse(model.generate.call_args.kwargs["do_sample"])
        self.assertNotIn("temperature", model.generate.call_args.kwargs)
        torch.tensor.assert_called_once_with(
            [[4, 5]], dtype="long", device="cpu"
        )
        self.assertIs(
            model.generate.call_args.kwargs["input_ids"],
            torch.tensor.return_value,
        )
        self.assertIs(
            model.generate.call_args.kwargs["attention_mask"],
            torch.ones_like.return_value,
        )
        session._tokenizer.decode.assert_called_once_with(
            [7, 99], skip_special_tokens=True
        )

    def test_sampling_and_length_finish_reason(self) -> None:
        """Pass positive temperatures and detect budget exhaustion."""

        session = _session(temperature=0.8)
        torch, model, _ = self._torch_inputs([7, 8])
        session._model = model
        session._tokenizer.eos_token_id = 99
        with patch.object(chat, "import_module", return_value=torch):
            session._torch_reply([4], 2)
        options = model.generate.call_args.kwargs
        self.assertTrue(options["do_sample"])
        self.assertEqual(options["temperature"], 0.8)
        self.assertEqual(session.last_finish_reason, "length")
        self.assertEqual(session.output_token_total, 2)

    def test_multiple_eos_ids_are_recognized(self) -> None:
        """Accept sequence-valued EOS configuration from the model."""

        session = _session()
        torch, model, _ = self._torch_inputs([7, 88])
        model.generation_config.eos_token_id = [88, 99]
        session._model = model
        with patch.object(chat, "import_module", return_value=torch):
            session._torch_reply([4], 2)
        self.assertEqual(session.last_finish_reason, "stop")

    def test_falls_back_to_tokenizer_eos_when_model_has_none(self) -> None:
        """Use tokenizer EOS when generation configuration omits it."""

        session = _session()
        torch, model, _ = self._torch_inputs([7, 99])
        model.generation_config.eos_token_id = None
        session._model = model
        session._tokenizer.eos_token_id = 99
        with patch.object(chat, "import_module", return_value=torch):
            session._torch_reply([4], 2)
        self.assertEqual(session.last_finish_reason, "stop")

    def test_generation_failure_propagates(self) -> None:
        """Do not account for tokens when model.generate raises."""

        session = _session()
        torch, model, _ = self._torch_inputs([])
        model.generate.side_effect = RuntimeError("inference")
        session._model = model
        with patch.object(chat, "import_module", return_value=torch):
            with self.assertRaisesRegex(RuntimeError, "inference"):
                session._torch_reply([4], 3)
        self.assertEqual(session.output_token_total, 0)


class TestMlxGeneration(unittest.TestCase):
    """Test MLX streaming, callback delivery, and token accounting."""

    def test_streaming_counts_empty_decoded_fragments(self) -> None:
        """Count each generated token while emitting only text fragments."""

        session = _session(temperature=0.4)
        responses = [
            SimpleNamespace(text="A", finish_reason=None),
            SimpleNamespace(text="", finish_reason=None),
            SimpleNamespace(text="B", finish_reason="stop"),
        ]
        mlx_lm = Mock()
        mlx_lm.stream_generate.return_value = iter(responses)
        sampling = Mock()
        sampling.make_sampler.return_value = object()
        modules = {
            "mlx_lm": mlx_lm,
            "mlx_lm.sample_utils": sampling,
        }
        callback = Mock()
        with patch.object(
            chat, "import_module", side_effect=modules.get
        ):
            answer = session._mlx_reply(
                [4, 5], 5, token_callback=callback
            )

        self.assertEqual(answer, "AB")
        self.assertEqual(session.output_token_total, 3)
        self.assertEqual(session._last_generation_output_token_count, 3)
        self.assertEqual(
            session._last_generation_token_count_source,
            "generated_tokens",
        )
        self.assertEqual(session.last_finish_reason, "stop")
        sampling.make_sampler.assert_called_once_with(temp=0.4)
        self.assertEqual(
            mlx_lm.stream_generate.call_args.kwargs["prompt"], [4, 5]
        )
        self.assertEqual(
            mlx_lm.stream_generate.call_args.kwargs["max_tokens"], 5
        )
        self.assertEqual(
            [call.args[0] for call in callback.call_args_list],
            ["A", "B"],
        )

    def test_callback_error_interrupts_stream(self) -> None:
        """Propagate callback errors after accounting for the streamed ID."""

        session = _session()
        mlx_lm = Mock()
        mlx_lm.stream_generate.return_value = iter(
            [SimpleNamespace(text="A", finish_reason=None)]
        )
        sampling = Mock()
        modules = {
            "mlx_lm": mlx_lm,
            "mlx_lm.sample_utils": sampling,
        }
        callback = Mock(side_effect=RuntimeError("callback"))
        with patch.object(
            chat, "import_module", side_effect=modules.get
        ):
            with self.assertRaisesRegex(RuntimeError, "callback"):
                session._mlx_reply([4], 3, token_callback=callback)
        self.assertEqual(session.output_token_total, 1)


class TestChatTracing(unittest.TestCase):
    """Test trace metadata, artifacts, and failure ordering."""

    @staticmethod
    def _trace(*, content: bool = False, tokens: bool = False) -> Mock:
        """Create a fake sink with explicit artifact capabilities."""

        trace = Mock()
        trace.enabled = True
        trace.include_content = content
        trace.include_token_ids = tokens
        trace.span.side_effect = lambda *args, **kwargs: nullcontext()
        trace.new_id.side_effect = lambda prefix: prefix + "-1"
        trace.artifact.side_effect = (
            lambda name, value, **kwargs: {"name": name, "value": value}
        )
        return trace

    def test_enabled_trace_emits_input_and_output_metadata(self) -> None:
        """Forward call IDs and record retokenized output counts."""

        trace = self._trace()
        session = _session(trace=trace)
        session._tokenizer.encode.return_value = [7, 8]
        with patch.object(
            session, "_torch_reply", return_value="Answer"
        ) as generate:
            result = session.reply("Hello", trace_turn_id="turn-1")

        self.assertEqual(result, "Answer")
        generate.assert_called_once_with(
            [4, 5, 6], 29, trace_turn_id="turn-1"
        )
        self.assertEqual(trace.emit.call_count, 2)
        first, second = trace.emit.call_args_list
        self.assertEqual(first.args[0], TRACE_EVENT_LLM_CALL_INPUT)
        self.assertEqual(second.args[0], TRACE_EVENT_LLM_CALL_OUTPUT)
        self.assertEqual(first.kwargs["data"]["input_token_count"], 3)
        self.assertEqual(first.kwargs["data"]["message_count"], 1)
        self.assertEqual(first.kwargs["data"]["roles"], ["user"])
        self.assertEqual(first.kwargs["turn_id"], "turn-1")
        self.assertEqual(second.kwargs["data"]["output_token_count"], 2)
        self.assertEqual(
            second.kwargs["data"]["output_token_count_source"],
            "retokenized_decoded_text",
        )
        self.assertEqual(
            second.kwargs["data"]["call_id"],
            first.kwargs["data"]["call_id"],
        )
        trace.artifact.assert_not_called()

    def test_content_and_token_flags_create_artifacts(self) -> None:
        """Record structured input, rendered prompt, and answer content."""

        trace = self._trace(content=True, tokens=True)
        session = _session(trace=trace)
        session._tokenizer.apply_chat_template.side_effect = [
            "rendered prompt", [4, 5, 6]
        ]
        session._tokenizer.encode.return_value = [7]
        with patch.object(
            session, "_torch_reply", return_value="Answer"
        ):
            session.reply("Hello", trace_turn_id="turn-2")

        first, second = trace.emit.call_args_list
        input_data = first.kwargs["data"]
        output_data = second.kwargs["data"]
        self.assertIn("messages_artifact", input_data)
        self.assertIn("rendered_prompt_artifact", input_data)
        self.assertIn("input_token_ids_artifact", input_data)
        self.assertIn("answer_artifact", output_data)
        self.assertEqual(
            session._tokenizer.apply_chat_template.call_args_list[0]
            .kwargs["tokenize"],
            False,
        )
        self.assertEqual(
            session._tokenizer.apply_chat_template.call_args_list[1]
            .kwargs["tokenize"],
            True,
        )

    def test_rendered_prompt_error_does_not_block_generation(self) -> None:
        """Continue when optional prompt rendering is unsupported."""

        trace = self._trace(content=True)
        session = _session(trace=trace)
        session._tokenizer.apply_chat_template.side_effect = [
            TypeError("text rendering unsupported"), [4, 5, 6]
        ]
        with patch.object(
            session, "_torch_reply", return_value="Answer"
        ):
            self.assertEqual(session.reply("Hello"), "Answer")
        self.assertNotIn(
            "rendered_prompt_artifact", trace.emit.call_args_list[0]
            .kwargs["data"],
        )

    def test_output_trace_failure_preserves_committed_reply(self) -> None:
        """Document that final tracing happens after history is saved."""

        trace = self._trace()
        trace.emit.side_effect = [None, OSError("output trace failed")]
        session = _session(trace=trace)
        with patch.object(
            session, "_torch_reply", return_value="Answer"
        ):
            with self.assertRaisesRegex(OSError, "output trace failed"):
                session.reply("Hello")
        self.assertEqual(
            [item["role"] for item in session.messages],
            ["user", "assistant"],
        )

    def test_torch_generated_ids_are_traced_when_requested(self) -> None:
        """Record generated ID artifacts through the PyTorch helper."""

        trace = self._trace(tokens=True)
        session = _session(trace=trace)
        torch = Mock()
        torch.long = "long"
        torch.inference_mode.return_value = nullcontext()
        model = Mock()
        model.generation_config.eos_token_id = 99
        output = MagicMock()
        output.__getitem__.return_value.tolist.return_value = [7, 8]
        model.generate.return_value = output
        session._model = model
        with patch.object(chat, "import_module", return_value=torch):
            session._torch_reply([4], 2, trace_turn_id="turn-3")
        trace.emit.assert_called_once()
        args = trace.emit.call_args
        self.assertEqual(args.args[0], TRACE_EVENT_LLM_CALL_OUTPUT_TOKENS)
        self.assertEqual(args.kwargs["data"]["output_token_count"], 2)
        self.assertIn("output_token_ids_artifact", args.kwargs["data"])


class TestChatSessionClose(unittest.TestCase):
    """Test session shutdown and device-specific resource cleanup."""

    def test_close_clears_references_and_resets_history(self) -> None:
        """Clear model references while preserving initial instructions."""

        session = _session(system_prompt="Instruction")
        session.restore_history(
            [{"role": chat.USER_ROLE, "content": "Conversation"}]
        )
        with (
            patch.object(chat.gc, "collect") as collect,
            patch.object(chat, "import_module") as imports,
        ):
            session.close()
            session.close()

        collect.assert_called_once_with()
        imports.assert_not_called()
        self.assertIsNone(session._model)
        self.assertIsNone(session._tokenizer)
        self.assertEqual(
            session.messages,
            [{"role": chat.SYSTEM_ROLE, "content": "Instruction"}],
        )
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.reply("Hello")

    def test_device_specific_cleanup(self) -> None:
        """Clear caches only for supported accelerator devices."""

        for backend, device, method in (
            ("torch", "cpu", None),
            ("torch", "cuda", "cuda.empty_cache"),
            ("torch", "mps", "mps.empty_cache"),
            ("mlx", "metal", "clear_cache"),
        ):
            with self.subTest(backend=backend, device=device):
                session = _session()
                session.backend = backend
                session.device = device
                backend_module = Mock()
                with (
                    patch.object(
                        chat, "import_module", return_value=backend_module
                    ) as imported,
                    patch.object(chat.gc, "collect"),
                ):
                    session.close()
                if method is None:
                    imported.assert_not_called()
                else:
                    imported.assert_called_once_with(
                        "mlx.core" if backend == "mlx" else "torch"
                    )
                    target = backend_module
                    for part in method.split("."):
                        target = getattr(target, part)
                    target.assert_called_once_with()


class TestChatValidationHelpers(unittest.TestCase):
    """Test exact boundaries for the module's value validators."""

    def test_boolean_validation(self) -> None:
        """Allow only real Boolean objects."""

        self.assertTrue(chat._check_bool(True, "flag"))
        self.assertFalse(chat._check_bool(False, "flag"))
        for value in (0, 1, "true", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    chat._check_bool(value, "flag")

    def test_temperature_validation(self) -> None:
        """Accept only finite, nonnegative numerical temperatures."""

        valid = (
            (0, 0.0),
            (-0.0, 0.0),
            (1, 1.0),
            (0.25, 0.25),
            (float("1e-300"), float("1e-300")),
        )
        for value, expected in valid:
            with self.subTest(value=value):
                self.assertEqual(chat._check_temperature(value), expected)

        invalid = (
            -1,
            True,
            None,
            "0.5",
            1j,
            float("nan"),
            float("inf"),
            float("-inf"),
            10**400,
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    chat._check_temperature(value)

    def test_token_limit_validation(self) -> None:
        """Permit None or positive integral non-Boolean limits."""

        self.assertIsNone(chat._check_max_new_tokens(None))
        self.assertEqual(chat._check_max_new_tokens(1), 1)
        self.assertEqual(chat._check_max_new_tokens(10**6), 10**6)
        for value in (0, -1, True, 3.0, "3"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    chat._check_max_new_tokens(value)

    def test_system_prompt_validation(self) -> None:
        """Strip valid prompts and reject empty or non-text values."""

        self.assertIsNone(chat._check_system_prompt(None))
        self.assertEqual(chat._check_system_prompt("  Hi  "), "Hi")
        for value in ("", "  ", False, 2):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    chat._check_system_prompt(value)


if __name__ == "__main__":
    unittest.main()
