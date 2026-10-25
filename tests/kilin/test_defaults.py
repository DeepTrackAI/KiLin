"""Test the shared loader for KíLín's packaged configuration defaults.

Tests use temporary JSON files and require no model, GUI, or network access.
They verify cached reads, defensive section copies, and strict type checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import inspect
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin import defaults


class TestPackagedDefaultsLoader(unittest.TestCase):
    """Test JSON shape validation and safe sharing of cached sections."""

    def setUp(self) -> None:
        """Create a disposable config path and an isolated reader cache."""

        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "kilin_defaults.json"
        self.path_patch = patch.object(defaults, "_DEFAULTS_PATH", self.path)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        defaults._packaged_defaults.cache_clear()
        self.addCleanup(defaults._packaged_defaults.cache_clear)

    def _write(self, payload: object) -> None:
        """Write JSON and clear the per-process loader cache.

        Parameters
        ----------
        payload: object
            JSON-compatible value used as the complete config file.

        """

        self.path.write_text(json.dumps(payload), encoding="utf-8")
        defaults._packaged_defaults.cache_clear()

    def test_caches_file_read_and_returns_independent_sections(self) -> None:
        """Load once while preventing callers from mutating cached lists."""

        self._write({"REPOSITORY": {"SKIP_FILES": ["first"]}})
        first = defaults.load_defaults_section("REPOSITORY")
        first["SKIP_FILES"].append("changed")
        self.path.write_text(
            '{"REPOSITORY": {"SKIP_FILES": ["second"]}}',
            encoding="utf-8",
        )
        second = defaults.load_defaults_section("REPOSITORY")

        self.assertEqual(second, {"SKIP_FILES": ["first"]})
        self.assertIsNot(first, second)
        self.assertIsNot(first["SKIP_FILES"], second["SKIP_FILES"])

    def test_missing_section_or_nonobject_section_is_rejected(self) -> None:
        """Fail when required configuration sections are unavailable."""

        for section in (None, [], "text"):
            with self.subTest(section=section):
                self._write({"ANSWERING": section})
                with self.assertRaisesRegex(ValueError, "ANSWERING"):
                    defaults.load_defaults_section("ANSWERING")

        self._write({"ANSWERING": {}})
        with self.assertRaisesRegex(ValueError, "REPOSITORY"):
            defaults.load_defaults_section("REPOSITORY")

    def test_bad_json_and_nonobject_root_are_rejected(self) -> None:
        """Reject malformed JSON and invalid root values explicitly."""

        self.path.write_text("{invalid", encoding="utf-8")
        defaults._packaged_defaults.cache_clear()
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            defaults.load_defaults_section("ANSWERING")

        for payload in ([], "text", None):
            with self.subTest(payload=payload):
                self._write(payload)
                with self.assertRaisesRegex(ValueError, "JSON object"):
                    defaults.load_defaults_section("ANSWERING")

    def test_missing_file_propagates_io_error(self) -> None:
        """Do not conceal missing packaged resources with fake defaults."""

        with self.assertRaises(FileNotFoundError):
            defaults.load_defaults_section("ANSWERING")


class TestDefaultValueValidation(unittest.TestCase):
    """Test typed validation used by modules owning default sections."""

    def test_integer_rejects_boolean_float_zero_and_missing_value(
        self,
    ) -> None:
        """Only positive integers are valid for numeric routing limits."""

        self.assertEqual(
            defaults.require_default_int({"MAX": 3}, "MAX", "ANSWERING"),
            3,
        )
        for value in (True, False, 0, -2, 1.5, "3", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "ANSWERING.MAX"
                ):
                    defaults.require_default_int(
                        {"MAX": value}, "MAX", "ANSWERING"
                    )
        with self.assertRaisesRegex(ValueError, "ANSWERING.MAX"):
            defaults.require_default_int({}, "MAX", "ANSWERING")

    def test_nonnegative_integer_accepts_zero_and_rejects_bad_types(
        self,
    ) -> None:
        """Allow a zero overlap or graph distance but never a boolean.

        """

        self.assertEqual(
            defaults.require_default_nonnegative_int(
                {"DISTANCE": 0}, "DISTANCE", "RETRIEVAL"
            ),
            0,
        )
        for value in (True, False, -1, 0.5, "0", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "RETRIEVAL.DISTANCE"
                ):
                    defaults.require_default_nonnegative_int(
                        {"DISTANCE": value}, "DISTANCE", "RETRIEVAL"
                    )

    def test_boolean_requires_an_actual_json_boolean(self) -> None:
        """Prevent numeric and string values from mimicking booleans.

        """

        for value in (True, False):
            self.assertIs(
                defaults.require_default_bool(
                    {"THINKING": value}, "THINKING", "PROJECT_PROFILE"
                ),
                value,
            )
        for value in (None, 0, 1, "false", "true", []):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "PROJECT_PROFILE.THINKING"
                ):
                    defaults.require_default_bool(
                        {"THINKING": value}, "THINKING", "PROJECT_PROFILE"
                    )

    def test_finite_numbers_distinguish_zero_and_positive_bounds(
        self,
    ) -> None:
        """Reject nonfinite and nonnumeric progress and temperature values.

        """

        self.assertEqual(
            defaults.require_default_number(
                {"TEMPERATURE": 0}, "TEMPERATURE", "PROJECT_PROFILE",
                allow_zero=True,
            ),
            0.0,
        )
        self.assertEqual(
            defaults.require_default_number(
                {"RATE": 0.15}, "RATE", "RAG"
            ),
            0.15,
        )
        for value in (True, False, None, "0", -0.5, float("nan"),
                      float("inf"), 10 ** 1000):
            with self.subTest(value=str(value)[:20]):
                with self.assertRaisesRegex(ValueError, "RAG.RATE"):
                    defaults.require_default_number(
                        {"RATE": value}, "RATE", "RAG"
                    )
        with self.assertRaisesRegex(ValueError, "RAG.RATE"):
            defaults.require_default_number({"RATE": 0}, "RATE", "RAG")

    def test_string_requires_nonempty_text(self) -> None:
        """Reject missing or blank model identifiers and regex strings."""

        self.assertEqual(
            defaults.require_default_string(
                {"MODEL": "example/model"}, "MODEL", "DEFAULT_MODELS"
            ),
            "example/model",
        )
        for value in (None, 1, "", " \t"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "DEFAULT_MODELS.MODEL"
                ):
                    defaults.require_default_string(
                        {"MODEL": value}, "MODEL", "DEFAULT_MODELS"
                    )

    def test_string_list_checks_container_and_items(self) -> None:
        """Return ordered immutable strings and reject malformed arrays."""

        self.assertEqual(
            defaults.require_default_strings(
                {"FILES": [".py", ".md"]}, "FILES", "REPOSITORY"
            ),
            (".py", ".md"),
        )
        for value in (None, [], [""], ["  "], [12], "one"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "REPOSITORY.FILES"
                ):
                    defaults.require_default_strings(
                        {"FILES": value}, "FILES", "REPOSITORY"
                    )

    def test_regex_is_compiled_with_configured_flags(self) -> None:
        """Preserve the regex semantics and report invalid syntax."""

        expression = defaults.require_default_pattern(
            {"CUE": r"\bhello\b"}, "CUE", "ANSWERING", flags=re.I
        )
        self.assertIsNotNone(expression.search("HELLO"))
        for value in ("(", "", False, 3):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "ANSWERING.CUE"
                ):
                    defaults.require_default_pattern(
                        {"CUE": value}, "CUE", "ANSWERING"
                    )

    def test_vocabulary_is_validated_and_immutable(self) -> None:
        """Validate lowercase common words independently of the router."""

        self.assertEqual(
            defaults.require_default_lowercase_words(
                {"WORDS": ["about", "after", "about"]},
                "WORDS",
                "ANSWERING",
            ),
            frozenset({"about", "after"}),
        )
        for value in ([], ["Word"], ["two words"], ["  word"], [9]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "ANSWERING.WORDS"
                ):
                    defaults.require_default_lowercase_words(
                        {"WORDS": value}, "WORDS", "ANSWERING"
                    )


class TestCoreDefaultIntegration(unittest.TestCase):
    """Verify configured defaults reach their intended core consumers."""

    def test_project_profile_receives_configured_creation_values(
        self,
    ) -> None:
        """Keep package settings separate from saved profile overrides.

        """

        from kilin import project

        settings = defaults.load_defaults_section("PROJECT_PROFILE")
        profile = project.default_project_profile("code")
        self.assertEqual(profile.allow_download, settings["ALLOW_DOWNLOAD"])
        self.assertEqual(profile.target_tokens, settings["TARGET_TOKENS"])
        self.assertEqual(profile.max_tokens, settings["MAX_TOKENS"])
        self.assertEqual(profile.overlap_tokens, settings["OVERLAP_TOKENS"])
        self.assertEqual(
            profile.retrieval_count, settings["RETRIEVAL_COUNT"]
        )
        self.assertEqual(profile.thinking, settings["THINKING"])
        self.assertEqual(profile.temperature, settings["TEMPERATURE"])
        self.assertEqual(
            profile.max_new_tokens, settings["MAX_NEW_TOKENS"]
        )
        # A query-format default has deliberately not been introduced.
        self.assertNotIn("DEFAULT_QUERY_FORMAT", settings)
        self.assertEqual(profile.query_format, "qwen3")

    def test_chunking_function_defaults_are_independent(self) -> None:
        """Use generic size defaults only when sizes are not supplied.

        """

        from kilin.chunking import chunk_source_units

        settings = defaults.load_defaults_section("CHUNKING")
        signature = inspect.signature(chunk_source_units)
        self.assertEqual(
            signature.parameters["target_tokens"].default,
            settings["TARGET_TOKENS"],
        )
        self.assertEqual(
            signature.parameters["max_tokens"].default,
            settings["MAX_TOKENS"],
        )
        self.assertEqual(
            signature.parameters["overlap_tokens"].default,
            settings["OVERLAP_TOKENS"],
        )

    def test_rag_limits_are_loaded_from_configuration(self) -> None:
        """Keep the RAG evidence and progress limits traceable to JSON.

        """

        from kilin import rag

        settings = defaults.load_defaults_section("RAG")
        values = {
            "EMBEDDING_PROGRESS_BATCH_SIZE":
                rag._EMBEDDING_PROGRESS_BATCH_SIZE,
            "PROGRESS_MIN_INTERVAL_SECONDS":
                rag._PROGRESS_MIN_INTERVAL_SECONDS,
            "PROGRESS_TRACE_PERCENT_STEP":
                rag._PROGRESS_TRACE_PERCENT_STEP,
            "MAX_REUSED_EVIDENCE": rag._MAX_REUSED_EVIDENCE,
            "MAX_RELATED_EVIDENCE": rag._MAX_RELATED_EVIDENCE,
            "UNCITED_RELATED_EVIDENCE": rag._UNCITED_RELATED_EVIDENCE,
            "MAX_EARLIER_TURNS": rag._MAX_EARLIER_TURNS,
        }
        self.assertEqual(values, settings)

    def test_project_type_fallbacks_use_packaged_values(self) -> None:
        """Keep omitted project-type fields linked to packaged defaults.

        """

        import importlib

        project_type_module = importlib.import_module("kilin.project_type")

        settings = defaults.load_defaults_section("PROJECT_TYPES")
        self.assertEqual(
            project_type_module.DOCUMENT_MAX_TOKENS,
            settings["DOCUMENT_MAX_TOKENS"]
        )
        # Only omitted JSON fields should use the fallback value.
        self.assertEqual(
            project_type_module._DEFAULT_RECENT_TURNS,
            settings["DEFAULT_RECENT_TURNS"],
        )

    def test_retrieval_function_defaults_and_heuristics(self) -> None:
        """Link retrieval algorithms and language heuristics to JSON.

        """

        from kilin.retrievals import context_expansion, document_reading
        from kilin.retrievals.strategies import (
            retrieval_code,
            retrieval_hybrid,
        )

        settings = defaults.load_defaults_section("RETRIEVAL")
        self.assertEqual(
            inspect.signature(retrieval_code.code_search)
            .parameters["max_distance"].default,
            settings["CODE_MAX_DISTANCE"],
        )
        self.assertEqual(
            inspect.signature(retrieval_hybrid.hybrid_search)
            .parameters["rank_constant"].default,
            settings["RRF_RANK_CONSTANT"],
        )
        self.assertEqual(
            document_reading._CHARACTERS_PER_TOKEN,
            settings["DOCUMENT_CHARACTERS_PER_TOKEN"],
        )
        self.assertEqual(
            document_reading._MIN_STEM_LENGTH,
            settings["MIN_DOCUMENT_STEM_LENGTH"],
        )
        self.assertEqual(
            context_expansion._RELATION_PHRASES,
            tuple(settings["RELATION_PHRASES"]),
        )
        for name in (
            "CALLEE_EXPANSION_DEPTH",
            "CALLEE_EXPANSION_MAX",
            "CLASS_METHOD_EXPANSION_MAX",
            "NAME_WORD_EXPANSION_MAX",
            "DOCUMENTATION_NAME_EXPANSION_MAX",
            "MODULE_FUNCTION_EXPANSION_MAX",
            "TEST_SUBJECT_EXPANSION_MAX",
            "RELATED_CHUNKS_MAX",
            "RELATED_PASSAGES_MAX",
            "SOURCE_SEARCH_OTHERS",
        ):
            with self.subTest(setting=name):
                self.assertEqual(
                    getattr(context_expansion, f"_{name}"), settings[name]
                )
        self.assertEqual(
            context_expansion.SOURCE_SEARCH_DEPTH,
            settings["SOURCE_SEARCH_DEPTH"],
        )


class TestDefaultDocumentation(unittest.TestCase):
    """Test that package-wide settings are accounted for in the guide."""

    def test_every_default_key_is_named_in_technical_guide(self) -> None:
        """Require a documented entry when another setting is introduced."""

        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (root / "kilin" / "kilin_defaults.json").read_text(
                encoding="utf-8"
            )
        )
        guide = (root / "docs" / "package_defaults.md").read_text(
            encoding="utf-8"
        )
        for section, settings in config.items():
            with self.subTest(section=section):
                self.assertIn(f"`{section}`", guide)
            for key in settings:
                with self.subTest(section=section, key=key):
                    self.assertIn(f"`{key}`", guide)



if __name__ == "__main__":
    unittest.main()
