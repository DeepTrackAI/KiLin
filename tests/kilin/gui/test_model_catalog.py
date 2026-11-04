"""Verify catalog model choices, cache roles and the thinking check.

These checks do not load models or start a Qt application. The thinking
check renders small chat templates written to a temporary cache.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin.gui.model_catalog import (
    MODEL_CHOICES,
    downloaded_models,
    installed_backends,
    model_choices,
    thinking_switches,
)
from kilin.project import default_project_profile


class TestModelCatalog(unittest.TestCase):
    """Suggested choices must agree with KíLín's project defaults."""

    @patch("kilin.gui.model_catalog.version")
    def test_backend_packages_are_checked_separately(
        self, get_version
    ) -> None:
        """Check each backend independently of the other package group."""
        from importlib.metadata import PackageNotFoundError

        def installed(package: str) -> str:
            """Return a synthetic backend package version."""
            if package == "torch":
                raise PackageNotFoundError(package)
            return "1.0"

        get_version.side_effect = installed

        self.assertEqual(installed_backends(), {"mlx": True, "torch": False})

    def test_default_models_have_distinct_roles(self) -> None:
        """Keep catalog role assignments aligned with core defaults."""
        profile = default_project_profile("code")
        offered = {
            (choice.backend, choice.role): choice for choice in MODEL_CHOICES
        }

        for backend in ("mlx", "torch"):
            self.assertEqual(
                offered[(backend, "chat")].model_id,
                profile.chat_model(backend),
            )
            self.assertEqual(
                offered[(backend, "embedding")].model_id,
                profile.embedding_model(backend),
            )
            self.assertEqual(
                offered[(backend, "embedding")].query_format,
                profile.query_format,
            )

    def test_unknown_cached_model_is_not_mislabeled(self) -> None:
        """Do not present unknown cache entries as suggested models."""
        cached = {MODEL_CHOICES[0].model_id, "someone/unknown-model"}

        choices = model_choices(cached)

        self.assertEqual(len(choices), len(MODEL_CHOICES))
        self.assertTrue(choices[0]["cached"])
        self.assertNotIn(
            "someone/unknown-model", {choice["model_id"] for choice in choices}
        )



# Like Qwen3's template: an empty thought when thinking is switched off.
_SWITCH = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
    "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}"
    "<|im_start|>assistant\n{% if enable_thinking is defined and "
    "enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)
# Like an older template: thinking is never mentioned.
_NO_SWITCH = _SWITCH.split("{% if enable_thinking")[0] + "{% endif %}"


class TestThinkingModels(unittest.TestCase):
    """Say per model whether its own template reacts to enable_thinking."""

    def test_switch_is_checked_and_unknown_models_are_left_out(self) -> None:
        with TemporaryDirectory() as folder:
            entries = []

            def cache(model_id: str, files: dict[str, str]) -> None:
                snapshot = Path(folder, model_id, "snapshots", "abc")
                snapshot.mkdir(parents=True)
                for name, text in files.items():
                    (snapshot / name).write_text(text, encoding="utf-8")
                entries.append(
                    {"model_id": model_id, "path": str(Path(folder, model_id))}
                )

            config = "tokenizer_config.json"
            cache("a/switch", {config: json.dumps({"chat_template": _SWITCH})})
            cache("a/old", {config: json.dumps({"chat_template": _NO_SWITCH})})
            # chat_template.jinja wins over tokenizer_config.json.
            cache("a/jinja", {
                "chat_template.jinja": _SWITCH,
                config: json.dumps({"chat_template": _NO_SWITCH}),
            })
            # Several named templates: the "default" one is used.
            cache("a/named", {config: json.dumps({"chat_template": [
                {"name": "tool_use", "template": _NO_SWITCH},
                {"name": "default", "template": _SWITCH},
            ]})})
            # No template, a broken one, or one that raises: unknown.
            cache("a/none", {config: json.dumps({})})
            cache("a/broken", {"chat_template.jinja": "{% if %}"})
            cache("a/raises", {
                "chat_template.jinja": "{{ raise_exception('no') }}"
            })
            # A template using the tokenizer's date helper still renders.
            cache("a/dated", {
                "chat_template.jinja": "{{ strftime_now('%d %B %Y') }}"
                + _SWITCH
            })
            # A template that renders but ignores the switch is False; one
            # that is missing or fails is unknown, so it is not listed.
            self.assertEqual(thinking_switches(entries), {
                "a/dated": True, "a/jinja": True, "a/named": True,
                "a/old": False, "a/switch": True,
            })
            # The same files always give the same answer.
            self.assertEqual(
                thinking_switches(entries), thinking_switches(entries)
            )



class TestDownloadedModels(unittest.TestCase):
    """A stopped download is not a downloaded model."""

    def test_only_finished_snapshots_count(self) -> None:
        with TemporaryDirectory() as folder:
            entries = []

            def cache(model_id: str, files: dict[str, str]) -> None:
                snapshot = Path(folder, model_id, "snapshots", "abc")
                snapshot.mkdir(parents=True)
                for name, text in files.items():
                    (snapshot / name).write_text(text, encoding="utf-8")
                entries.append(
                    {"model_id": model_id, "path": str(Path(folder, model_id))}
                )

            index = json.dumps({"weight_map": {
                "a": "model-1.safetensors", "b": "model-2.safetensors",
            }})
            cache("a/whole", {"config.json": "{}", "model.safetensors": "w"})
            cache("a/shards", {
                "config.json": "{}", "model.safetensors.index.json": index,
                "model-1.safetensors": "w", "model-2.safetensors": "w",
            })
            # Stopped downloads: no weights yet, a missing shard, no config.
            cache("a/stopped", {"config.json": "{}", "README.md": "r"})
            cache("a/half", {
                "config.json": "{}", "model.safetensors.index.json": index,
                "model-1.safetensors": "w",
            })
            cache("a/noconfig", {"model.safetensors": "w"})
            with patch(
                "kilin.gui.model_catalog.available_models",
                return_value=entries,
            ):
                found = [item["model_id"] for item in downloaded_models()]
            self.assertEqual(found, ["a/whole", "a/shards"])


if __name__ == "__main__":
    unittest.main()
