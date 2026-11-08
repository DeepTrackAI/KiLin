"""Verify independent and lazy source-card model generation.

A deterministic chat substitute exercises session reuse, validated complete
replies, failure propagation, and release without loading real model weights.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import Mock

from kilin._model_completion import IncompleteModelResponse
from kilin.model_settings import default_source_card_generation_settings
from kilin.retrievals.source_cards.generation import SourceCardGenerator


class FakeSession:
    instances = []
    response = '{"title": "Example"}'
    finish = "stop"

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.options = kwargs
        self.messages = []
        self.thinking = kwargs["thinking"]
        self.temperature = kwargs["temperature"]
        self.max_new_tokens = kwargs["max_new_tokens"]
        self.last_finish_reason = self.finish
        self.closed = False
        self.prompts = []
        self.instances.append(self)

    def restore_history(self, messages):
        self.messages = [dict(message) for message in messages]

    def configure_generation(self, settings):
        self.thinking = settings["thinking"]
        self.temperature = settings["temperature"]

    def reply(self, message, **options):
        self.prompts.append((message, options, list(self.messages)))
        self.messages.append({"role": "user", "content": message})
        self.messages.append({"role": "assistant", "content": self.response})
        return self.response

    def close(self):
        self.closed = True


class TestSourceCardGenerator(unittest.TestCase):

    def setUp(self):
        FakeSession.instances.clear()
        FakeSession.response = '{"title": "Example"}'
        FakeSession.finish = "stop"
        self.settings = default_source_card_generation_settings("torch")

    def new_generator(self, **changes):
        return SourceCardGenerator(
            replace(self.settings, **changes), backend="torch",
            allow_download=False, session_factory=FakeSession,
        )

    def test_lazy_generation_reuses_session_and_closes_on_exit(self):
        with self.new_generator() as generator:
            self.assertEqual(len(FakeSession.instances), 0)
            self.assertEqual(
                generator.generate("Summarize.", "First source."),
                FakeSession.response,
            )
            self.assertEqual(
                generator.generate("Summarize.", "Second source."),
                FakeSession.response,
            )
            self.assertEqual(len(FakeSession.instances), 1)
            session = FakeSession.instances[0]
            self.assertFalse(session.options["allow_download"])
            self.assertFalse(session.options["block_unrequested_cjk"])
            self.assertEqual(session.model_id, self.settings.model_id)
            self.assertEqual(session.options["backend"], "torch")
            self.assertEqual(session.messages, [])
            self.assertEqual(len(session.prompts), 2)
            self.assertEqual(
                session.prompts[0][2],
                [{"role": "system", "content": "Summarize."}],
            )
            self.assertEqual(
                session.prompts[0][1]["max_new_tokens"], 1200
            )
            self.assertFalse(session.closed)
        self.assertTrue(session.closed)
        generator.close()
        with self.assertRaises(RuntimeError):
            generator.generate("System", "Text")

    def test_configured_inference_parameters_are_inherited(self):
        with self.new_generator(
            thinking=True, temperature=0.4, max_new_tokens=512,
        ) as generator:
            generator.generate("System", "Text")
            session = FakeSession.instances[0]
            self.assertTrue(session.thinking)
            self.assertEqual(session.temperature, 0.4)
            self.assertEqual(
                session.prompts[0][1]["max_new_tokens"], 512
            )
            self.assertTrue(session.prompts[0][1]["language_context"] == ())

    def test_incomplete_response_releases_session_on_failure(self):
        FakeSession.response = "unfinished"
        FakeSession.finish = "length"
        with self.assertRaises(IncompleteModelResponse):
            with self.new_generator() as generator:
                generator.generate("System", "Text")
        self.assertTrue(FakeSession.instances[0].closed)

    def test_constructor_and_missing_prompts_are_model_free(self):
        for backend, allow, factory in (
            ("auto", True, FakeSession),
            ("invalid", True, FakeSession),
            ("torch", "yes", FakeSession),
            ("torch", False, "not callable"),
        ):
            with self.subTest(backend=backend, allow=allow):
                with self.assertRaises(ValueError):
                    SourceCardGenerator(
                        self.settings, backend=backend,
                        allow_download=allow, session_factory=factory,
                    )
        with self.new_generator() as generator:
            for system, prompt in (("", "Text"), ("System", "")):
                with self.assertRaises(ValueError):
                    generator.generate(system, prompt)
        self.assertEqual(len(FakeSession.instances), 0)

    def test_loading_error_never_falls_back_to_answer_model(self):
        factory = Mock(side_effect=RuntimeError("model unavailable"))
        generator = SourceCardGenerator(
            self.settings, backend="torch", allow_download=False,
            session_factory=factory,
        )
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            with generator:
                generator.generate("System", "Prompt")
        factory.assert_called_once()
        self.assertIsNone(generator._session)
        self.assertTrue(generator._closed)

    def test_context_never_loads_unused_model(self):
        with self.new_generator():
            pass
        self.assertEqual(FakeSession.instances, [])


if __name__ == "__main__":
    unittest.main()
