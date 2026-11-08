"""Test project-type loading, validation, and packaged definitions.

Use small, synthetic JSON definitions to exercise schema contracts without
freezing the descriptions, prompts, or intent names in packaged types. The
packaged ``code`` and ``literature`` files are checked through the public
loader and their integration with project profile defaults.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin import project
from kilin.defaults import load_defaults_section
from kilin.project_type import (
    DOCUMENT_MAX_TOKENS,
    HISTORY_POLICY_CHOICES,
    RETRIEVAL_MODE_CHOICES,
    RETRIEVAL_STRATEGY_CHOICES,
    PROJECT_TYPES_DIR,
    Intent,
    ProjectType,
    load_project_types,
    project_type,
)
from tests.kilin._project_type_support import _type_data


class ProjectTypeFixture(unittest.TestCase):
    """Provide isolated directories and non-packaged JSON configurations."""

    def setUp(self) -> None:
        """Create a temporary root without changing the packaged cache.

        """

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _folder(
        self,
        name: str,
        definitions: dict[str, object] | None = None,
    ) -> Path:
        """Write an isolated collection of JSON project-type files.

        Parameters
        ----------
        name: str
            Unique directory name within the test's temporary root.
        definitions: dict[str, object] | None, optional
            Type IDs and JSON values; one synthetic definition by default.

        Returns
        -------
        pathlib.Path
            Directory accepted by ``load_project_types``.

        """

        directory = self.root / name
        directory.mkdir()
        if definitions is None:
            definitions = {"samples": _type_data()}
        for type_id, value in definitions.items():
            (directory / f"{type_id}.json").write_text(
                json.dumps(value), encoding="utf-8"
            )
        return directory

    def _load(
        self,
        name: str,
        data: dict[str, object],
    ) -> ProjectType:
        """Validate one synthetic definition through the public loader.

        Parameters
        ----------
        name: str
            Unique directory name for this validation case.
        data: dict[str, object]
            Candidate JSON definition.

        Returns
        -------
        ProjectType
            Loaded custom project type.

        """

        return load_project_types(self._folder(name, {"samples": data}))[
            "samples"
        ]


class TestPackagedProjectTypes(unittest.TestCase):
    """Check real JSON files without asserting their mutable prose."""

    def test_builtin_types_load_as_independent_json_files(self) -> None:
        """All packaged JSON files are parsed in deterministic file order."""

        paths = sorted(PROJECT_TYPES_DIR.glob("*.json"))
        self.assertGreaterEqual(len(paths), 2)
        types = load_project_types()
        self.assertEqual(list(types), [path.stem for path in paths])
        self.assertIn("code", types)
        self.assertIn("literature", types)
        self.assertEqual(
            project.PROJECT_TYPE_IDS, tuple(types)
        )
        for path in paths:
            with self.subTest(type_id=path.stem):
                raw = json.loads(path.read_text(encoding="utf-8"))
                definition = types[path.stem]
                self.assertIsInstance(definition, ProjectType)
                self.assertEqual(definition.id, path.stem)
                self.assertIs(project_type(path.stem), definition)
                self.assertEqual(definition.label, definition.display_name)
                self.assertEqual(
                    definition.extensions,
                    tuple(ext.lower() for ext in raw["extensions"]),
                )
                self.assertEqual(
                    definition.max_file_bytes, raw["max_file_bytes"]
                )
                self.assertEqual(
                    list(definition.intents), list(raw["intents"])
                )
                self.assertIn(definition.default_intent, definition.intents)
                self.assertTrue(definition.retrieval_instruction)
                self.assertTrue(definition.system_prompt)
                self.assertTrue(definition.description)
                self.assertTrue(definition.display_name)

    def test_packaged_intents_are_valid_and_usable(self) -> None:
        """Exercise parsed options for every current and future intent."""

        for type_id, definition in load_project_types().items():
            with self.subTest(type_id=type_id):
                self.assertTrue(definition.intents)
                self.assertGreater(definition.max_file_bytes, 0)
                self.assertGreaterEqual(definition.recent_turns, 0)
                self.assertTrue(definition.extensions)
                for intent_id, intent in definition.intents.items():
                    with self.subTest(type_id=type_id, intent=intent_id):
                        self.assertIsInstance(intent, Intent)
                        self.assertEqual(intent.intent_id, intent_id)
                        self.assertTrue(intent.description)
                        self.assertIn(
                            intent.history_policy,
                            HISTORY_POLICY_CHOICES,
                        )
                        self.assertIn(
                            intent.retrieval_mode,
                            RETRIEVAL_MODE_CHOICES,
                        )
                        self.assertIn(
                            intent.retrieval_strategy,
                            RETRIEVAL_STRATEGY_CHOICES,
                        )
                        if intent.tool is None:
                            self.assertTrue(intent.system_prompt)
                        else:
                            self.assertEqual(intent.retrieval_mode, "none")
                        for shortcut in intent.shortcuts:
                            self.assertIsInstance(shortcut, re.Pattern)
                            self.assertTrue(shortcut.flags & re.IGNORECASE)

    def test_only_packaged_small_talk_disables_thinking(self) -> None:
        """Other packaged intents inherit the configured session setting."""

        for type_id in ("code", "literature"):
            for intent_id, intent in project_type(type_id).intents.items():
                with self.subTest(type_id=type_id, intent_id=intent_id):
                    if intent_id == "small_talk":
                        self.assertIs(intent.thinking, False)
                    else:
                        self.assertIsNone(intent.thinking)

    def test_builtin_defaults_follow_current_json_configuration(self) -> None:
        """Project profiles adopt configured file suffixes and size limits."""

        for type_id in ("code", "literature"):
            with self.subTest(type_id=type_id):
                definition = project_type(type_id)
                profile = project.default_project_profile(type_id)
                self.assertEqual(profile.type_id, type_id)
                self.assertEqual(profile.extensions, definition.extensions)
                self.assertEqual(
                    profile.max_file_bytes, definition.max_file_bytes
                )
                self.assertGreater(profile.target_tokens, 0)
                self.assertLessEqual(
                    profile.target_tokens, profile.max_tokens
                )

    def test_every_packaged_file_rejects_unknown_fields(self) -> None:
        """Reject typos at every configuration level in every JSON file."""

        definitions = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(PROJECT_TYPES_DIR.glob("*.json"))
        }
        self.assertGreaterEqual(len(definitions), 2)
        count = 0
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for type_id, data in definitions.items():
                intent_id = next(iter(data["intents"]))
                locations = (
                    ("", ()),
                    (".capabilities", ("capabilities",)),
                    (".history", ("history",)),
                    (f".intents.{intent_id}", ("intents", intent_id)),
                    (
                        f".intents.{intent_id}.history",
                        ("intents", intent_id, "history"),
                    ),
                    (
                        f".intents.{intent_id}.retrieval",
                        ("intents", intent_id, "retrieval"),
                    ),
                )
                for location, keys in locations:
                    with self.subTest(type_id=type_id, location=location):
                        changed = deepcopy(definitions)
                        target = changed[type_id]
                        for key in keys:
                            target = target.setdefault(key, {})
                        target["unrecognized_setting"] = True
                        directory = root / f"case{count}"
                        directory.mkdir()
                        for name, content in changed.items():
                            (directory / f"{name}.json").write_text(
                                json.dumps(content), encoding="utf-8"
                            )
                        with self.assertRaisesRegex(
                            ValueError, "unknown settings"
                        ) as caught:
                            load_project_types(directory)
                        self.assertIn(
                            f"{type_id}{location}", str(caught.exception)
                        )
                        self.assertIn(
                            "unrecognized_setting", str(caught.exception)
                        )
                        count += 1
                tool_id = next(
                    (
                        name for name, intent in data["intents"].items()
                        if "tool" in intent
                    ),
                    None,
                )
                if tool_id is not None:
                    with self.subTest(type_id=type_id, location="tool"):
                        changed = deepcopy(definitions)
                        changed[type_id]["intents"][tool_id]["tool"][
                            "unrecognized_setting"
                        ] = True
                        directory = root / f"case{count}"
                        directory.mkdir()
                        for name, content in changed.items():
                            (directory / f"{name}.json").write_text(
                                json.dumps(content), encoding="utf-8"
                            )
                        with self.assertRaisesRegex(
                            ValueError, "unknown settings"
                        ) as caught:
                            load_project_types(directory)
                        self.assertIn(
                            f"{type_id}.intents.{tool_id}.tool",
                            str(caught.exception),
                        )
                        count += 1

    def test_packaged_files_accept_textual_and_intent_edits(self) -> None:
        """Modified descriptions and intent IDs preserve schema validity."""

        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for path in PROJECT_TYPES_DIR.glob("*.json"):
                raw = json.loads(path.read_text(encoding="utf-8"))
                raw["display_name"] = "A revised name"
                raw["description"] = "A revised description"
                raw["system_prompt"] = "A revised system prompt"
                raw["retrieval_instruction"] = "A revised instruction"
                raw["welcome_prompts"] = ["A new welcome prompt"]
                raw["suggestions"] = ["A new suggestion"]
                raw["intents"] = {
                    "new_intent": {
                        "description": "A replacement intent.",
                        "system_prompt": "Reply to the user.",
                    }
                }
                raw["default_intent"] = "new_intent"
                (folder / path.name).write_text(
                    json.dumps(raw), encoding="utf-8"
                )
            definitions = load_project_types(folder)
            self.assertIn("code", definitions)
            self.assertIn("literature", definitions)
            for definition in definitions.values():
                self.assertEqual(definition.default_intent, "new_intent")
                self.assertEqual(list(definition.intents), ["new_intent"])
                self.assertEqual(definition.display_name, "A revised name")


class TestProjectTypeLoading(ProjectTypeFixture):
    """Test file discovery, strict JSON, lookup, and process caching."""

    def test_discovery_order_and_custom_types(self) -> None:
        """A custom directory need not contain the two built-in types."""

        definitions = {"zeta": _type_data(), "alpha": _type_data()}
        types = load_project_types(self._folder("ordered", definitions))
        self.assertEqual(list(types), ["alpha", "zeta"])
        self.assertEqual(types["alpha"].id, "alpha")
        self.assertEqual(types["zeta"].id, "zeta")

    def test_missing_or_empty_directory_is_rejected(self) -> None:
        """Report the absence of JSON definitions explicitly."""

        for folder in (self.root / "missing", self.root / "empty"):
            with self.subTest(folder=folder):
                if folder.name == "empty":
                    folder.mkdir()
                with self.assertRaisesRegex(ValueError, "no project type"):
                    load_project_types(folder)

    def test_bad_json_and_non_object_roots(self) -> None:
        """Give descriptive errors for broken or non-object JSON input."""

        for name, content, message in (
            ("syntax", "{", "not valid JSON"),
            ("list", "[]", "one project type object"),
            ("null", "null", "one project type object"),
        ):
            with self.subTest(name=name):
                folder = self.root / name
                folder.mkdir()
                (folder / "samples.json").write_text(
                    content, encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, message):
                    load_project_types(folder)

    def test_duplicate_keys_are_rejected_at_every_json_level(self) -> None:
        """Reject duplicate IDs rather than silently taking the last."""

        raw = json.dumps(_type_data())
        inputs = (
            '{"display_name":"a","display_name":"b"}',
            raw.replace(
                '"max_file_bytes": 4096',
                '"max_file_bytes": 4096, "max_file_bytes": 4096',
            ),
        )
        for index, content in enumerate(inputs):
            with self.subTest(index=index):
                folder = self.root / f"duplicate{index}"
                folder.mkdir()
                (folder / "samples.json").write_text(
                    content, encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "Duplicate IDs"):
                    load_project_types(folder)

    def test_lookup_reports_unknown_project_type(self) -> None:
        """Include valid type identifiers in an unknown-type error."""

        with self.assertRaisesRegex(ValueError, "Unknown project type"):
            project_type("not_a_packaged_type")

    def test_cached_folder_uses_original_objects_until_cleared(self) -> None:
        """A known directory remains cached until explicitly cleared."""

        folder = self._folder("cached")
        original = load_project_types(folder)
        raw = _type_data()
        raw["display_name"] = "A new title"
        (folder / "samples.json").write_text(
            json.dumps(raw), encoding="utf-8"
        )
        self.assertIs(load_project_types(folder), original)
        self.assertEqual(original["samples"].display_name,
                         "Experimental type")
        try:
            load_project_types.cache_clear()
            refreshed = load_project_types(folder)
            self.assertEqual(refreshed["samples"].display_name,
                             "A new title")
        finally:
            load_project_types.cache_clear()

    def test_invalid_kind_ids_are_rejected(self) -> None:
        """Validate IDs from filenames using the restricted ID grammar."""

        for name in ("BadName", "2invalid", "too" + "x" * 33):
            with self.subTest(name=name):
                folder = self._folder(name, {name: _type_data()})
                with self.assertRaisesRegex(ValueError, "Project type ID"):
                    load_project_types(folder)

    def test_later_file_with_unknown_field_fails_entire_directory(
        self,
    ) -> None:
        """A valid first JSON file must not mask a broken later file."""

        valid = _type_data()
        broken = _type_data()
        broken["misspelled_description"] = "not a valid field"
        folder = self._folder(
            "laterunknown", {"alpha": valid, "zeta": broken}
        )
        with self.assertRaisesRegex(
            ValueError, "zeta.*unknown settings.*misspelled_description"
        ):
            load_project_types(folder)

    def test_valid_kind_id_boundaries(self) -> None:
        """Allow valid separators and the largest permitted identifier."""

        for index, type_id in enumerate(("x", "a-b_c2", "a" * 32)):
            with self.subTest(type_id=type_id):
                folder = self._folder(f"valid{index}",
                                      {type_id: _type_data()})
                self.assertEqual(
                    load_project_types(folder)[type_id].id, type_id
                )


class TestProjectTypeFields(ProjectTypeFixture):
    """Check project-type field types, defaults, and boundaries."""

    def test_private_parser_rejects_non_object_definitions(self) -> None:
        """Defensively validate direct calls to the JSON object parser."""

        with self.assertRaisesRegex(ValueError, "must be an object"):
            from kilin.project_type import _parse_project_type

            _parse_project_type("samples", [])

    def test_minimal_definition_uses_valid_defaults(self) -> None:
        """Load all implicit options for a type with one search intent."""

        loaded = self._load("minimal", _type_data())
        intent = loaded.intents["explain"]
        self.assertEqual(loaded.extensions, (".txt", ".md"))
        self.assertEqual(loaded.answer_guidance, "")
        self.assertEqual(loaded.welcome_prompts, ())
        self.assertEqual(loaded.suggestions, ())
        self.assertEqual(loaded.code_graph, "auto")
        self.assertEqual(
            loaded.recent_turns,
            load_defaults_section("PROJECT_TYPES")["DEFAULT_RECENT_TURNS"],
        )
        self.assertEqual(loaded.default_intent, "explain")
        self.assertEqual(intent.history_policy, "relevant")
        self.assertEqual(intent.retrieval_mode, "search")
        self.assertEqual(intent.retrieval_strategy, "hybrid")
        self.assertFalse(intent.include_related_evidence)
        self.assertEqual(intent.max_tokens, DOCUMENT_MAX_TOKENS)
        self.assertIsNone(intent.answer_guidance)
        self.assertIsNone(intent.tool)
        self.assertEqual(intent.shortcuts, ())
        with self.assertRaises(FrozenInstanceError):
            loaded.display_name = "other"
        with self.assertRaises(FrozenInstanceError):
            intent.description = "other"

    def test_unknown_project_and_nested_keys_are_rejected(self) -> None:
        """Reject misspelled fields without disallowing custom type IDs."""

        scenarios = (
            (("display_nam",), "samples"),
            (("capabilities", "code_grap"), "samples.capabilities"),
            (("history", "recent_turn"), "samples.history"),
        )
        for index, (keys, location) in enumerate(scenarios):
            with self.subTest(keys=keys):
                data = _type_data()
                target = data
                for key in keys[:-1]:
                    target = target.setdefault(key, {})
                target[keys[-1]] = 1
                with self.assertRaisesRegex(
                    ValueError, "unknown settings"
                ) as caught:
                    self._load(f"unknownproject{index}", data)
                self.assertIn(location, str(caught.exception))
                self.assertIn(keys[-1], str(caught.exception))

    def test_unknown_keys_are_reported_in_sorted_order(self) -> None:
        """Multiple typos produce a deterministic, actionable error."""

        data = _type_data()
        data["zeta_typo"] = 1
        data["alpha_typo"] = 2
        with self.assertRaisesRegex(
            ValueError, r"unknown settings: alpha_typo, zeta_typo"
        ):
            self._load("severalunknown", data)

    def test_optional_text_fields_are_stripped(self) -> None:
        """Normalize optional labels, guidance, and example strings."""

        data = _type_data()
        data.update({
            "display_name": "  A title  ",
            "answer_guidance": "  Respond briefly.  ",
            "welcome_prompts": [" First question ", " Second question "],
            "suggestions": [" Try this "],
        })
        item = self._load("stripped", data)
        self.assertEqual(item.label, "A title")
        self.assertEqual(item.answer_guidance, "Respond briefly.")
        self.assertEqual(item.welcome_prompts,
                         ("First question", "Second question"))
        self.assertEqual(item.suggestions, ("Try this",))

    def test_missing_required_and_invalid_text_fields(self) -> None:
        """Reject missing, blank, or non-string required descriptions."""

        fields = (
            "display_name", "description", "retrieval_instruction",
            "system_prompt",
        )
        counter = 0
        for field in fields:
            for value in (None, "  ", 3, [], True):
                with self.subTest(field=field, value=value):
                    data = _type_data()
                    data[field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        self._load(f"text{counter}", data)
                    counter += 1
        for field in fields:
            with self.subTest(missing=field):
                data = _type_data()
                del data[field]
                with self.assertRaisesRegex(ValueError, field):
                    self._load(f"missing{field}", data)

    def test_optional_lists_must_contain_nonblank_text(self) -> None:
        """Validate both optional display-list fields independently."""

        for index, (field, value) in enumerate(
            (field, value)
            for field in ("welcome_prompts", "suggestions")
            for value in ("text", [""], [17], [None], {})
        ):
            with self.subTest(field=field, value=value):
                data = _type_data()
                data[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    self._load(f"list{index}", data)

    def test_extension_validation(self) -> None:
        """Require a nonempty suffix list and preserve lowercase output."""

        for index, values in enumerate(
            ([], "py", ["py"], ["."], [".py", ""])
        ):
            with self.subTest(values=values):
                data = _type_data()
                data["extensions"] = values
                with self.assertRaisesRegex(ValueError, "extensions|suffix"):
                    self._load(f"suffix{index}", data)
        data = _type_data()
        data["extensions"] = [".PY", ".Md"]
        self.assertEqual(
            self._load("uppercase", data).extensions,
            (".py", ".md"),
        )

    def test_extensions_use_repository_scanner_grammar(
        self,
    ) -> None:
        """Reject wildcard patterns that the repository scanner rejects."""

        data = _type_data()
        data["extensions"] = [".py*"]
        with self.assertRaisesRegex(ValueError, "suffix|extensions"):
            self._load("wildcard", data)

    def test_normalized_extension_duplicates_are_removed(self) -> None:
        """Deduplicate equivalent suffixes using the scanner's grammar."""

        data = _type_data()
        data["extensions"] = [".PY", ".py", ".tar.GZ", ".TAR.gz"]
        self.assertEqual(
            self._load("deduplicated", data).extensions,
            (".py", ".tar.gz"),
        )

    def test_invalid_literal_extension_patterns(self) -> None:
        """Reject unsafe suffixes before scanning a project."""

        for index, extension in enumerate((
            ".py*", ".pdf?", ".py/file", ".py file", ".py\\ext",
            ".", ".py!", ".py;",
        )):
            with self.subTest(extension=extension):
                data = _type_data()
                data["extensions"] = [extension]
                with self.assertRaisesRegex(ValueError, "suffix"):
                    self._load(f"unsafe-suffix{index}", data)

    def test_positive_file_size_requires_integer_not_boolean(self) -> None:
        """Do not interpret booleans or numeric strings as byte counts."""

        for index, value in enumerate((None, 0, -5, True, 12.5, "4096")):
            with self.subTest(value=value):
                data = _type_data()
                data["max_file_bytes"] = value
                with self.assertRaisesRegex(ValueError, "max_file_bytes"):
                    self._load(f"bytes{index}", data)
        data = _type_data()
        data["max_file_bytes"] = 1
        self.assertEqual(self._load("onebyte", data).max_file_bytes, 1)

    def test_nested_fields_require_json_objects(self) -> None:
        """Reject malformed capabilities, history, and intents objects."""

        for index, (field, value) in enumerate(
            (field, value)
            for field in ("capabilities", "history", "intents")
            for value in ([], None, "text", 9)
        ):
            with self.subTest(field=field, value=value):
                data = _type_data()
                data[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    self._load(f"object{index}", data)

    def test_code_graph_accepts_true_false_auto_and_rejects_others(
        self,
    ) -> None:
        """Accept each canonical graph setting, excluding numeric aliases."""

        for index, value in enumerate((True, False, "auto")):
            data = _type_data()
            data["capabilities"] = {"code_graph": value}
            self.assertEqual(self._load(f"graphok{index}", data).code_graph,
                             value)
        for index, value in enumerate((1, 0, None, "yes", "AUTO", [])):
            with self.subTest(value=value):
                data = _type_data()
                data["capabilities"] = {"code_graph": value}
                with self.assertRaisesRegex(ValueError, "code_graph"):
                    self._load(f"graphbad{index}", data)

    def test_document_headers_accepts_booleans_only(self) -> None:
        """Default to no paper index and reject non-boolean settings."""

        self.assertFalse(
            self._load("paperdefault", _type_data()).document_headers
        )
        for index, value in enumerate((True, False)):
            data = _type_data()
            data["capabilities"] = {"document_headers": value}
            self.assertIs(
                self._load(f"paperok{index}", data).document_headers, value
            )
        for index, value in enumerate((1, 0, None, "true", [])):
            with self.subTest(value=value):
                data = _type_data()
                data["capabilities"] = {"document_headers": value}
                with self.assertRaisesRegex(ValueError, "document_headers"):
                    self._load(f"paperbad{index}", data)

    def test_papers_strategy_requires_document_headers(self) -> None:
        """Validate the papers strategy and its paper and passage limits."""

        data = _type_data()
        intent = next(iter(data["intents"].values()))
        intent["retrieval"] = {"mode": "search", "strategy": "papers"}
        with self.assertRaisesRegex(ValueError, "document_headers"):
            self._load("papersneedindex", data)
        data["capabilities"] = {"document_headers": True}
        loaded = next(iter(self._load("papersok", data).intents.values()))
        self.assertEqual(
            (loaded.retrieval_strategy, loaded.max_papers,
             loaded.passages_per_paper),
            ("papers", 3, 0),
        )
        for index, (key, value) in enumerate((
            ("max_papers", 0), ("max_papers", 51), ("max_papers", True),
            ("passages_per_paper", -1), ("passages_per_paper", 11),
            ("passages_per_paper", 1.5),
        )):
            with self.subTest(key=key, value=value):
                bad = deepcopy(data)
                next(iter(bad["intents"].values()))["retrieval"][key] = value
                with self.assertRaisesRegex(ValueError, key):
                    self._load(f"paperslimit{index}", bad)

    def test_source_first_strategy_is_generic_and_opt_in(self) -> None:
        """Allow source-first without paper indexing or stored card kinds."""

        data = _type_data()
        intent = next(iter(data["intents"].values()))
        intent["retrieval"] = {
            "mode": "search", "strategy": "source_first",
            "max_papers": 4, "passages_per_paper": 1,
        }
        config = self._load("sourcefirst", data)
        self.assertFalse(config.document_headers)
        self.assertFalse(config.source_cards_enabled)
        selected = next(iter(config.intents.values()))
        self.assertEqual(selected.retrieval_strategy, "source_first")
        self.assertEqual(selected.max_papers, 4)
        self.assertEqual(selected.passages_per_paper, 1)
        self.assertEqual(
            project_type("literature").intents[
                "question_answering"
            ].retrieval_strategy,
            "source_first",
        )

    def test_packaged_literature_type_indexes_papers(self) -> None:
        """Index papers for literature projects but not for code projects."""

        literature = project_type("literature")
        self.assertTrue(literature.document_headers)
        self.assertFalse(project_type("code").document_headers)
        for intent_id in ("question_answering", "compare_sources",
                          "report_generation"):
            with self.subTest(intent=intent_id):
                self.assertEqual(
                    literature.intents[intent_id].retrieval_strategy,
                    "source_first",
                )
        limits = literature.intents["compare_sources"].source_limits()
        self.assertEqual(
            (limits.max_sources, limits.passages_per_source,
             limits.max_passages, limits.max_tokens),
            (8, 2, 16, 9000),
        )

    def test_source_first_limits_are_per_intent_and_validated(self):
        data = _type_data()
        intent = next(iter(data["intents"].values()))
        intent["retrieval"] = {
            "mode": "search", "strategy": "source_first",
            "max_sources": 7, "passages_per_source": 3,
            "max_passages": 12, "evidence_tokens": 7500,
            "coverage": "exhaustive",
        }
        loaded = next(iter(self._load(
            "sourcebudgetok", data
        ).intents.values()))
        self.assertEqual(
            loaded.source_limits().max_sources, 7
        )
        self.assertEqual(loaded.source_limits().max_passages, 12)
        self.assertEqual(loaded.source_limits().max_tokens, 7500)
        self.assertEqual(loaded.source_limits().coverage, "exhaustive")
        invalid = (
            ("max_sources", True),
            ("passages_per_source", 0),
            ("max_passages", 201),
            ("evidence_tokens", 999),
            ("coverage", "all"),
        )
        for count, (key, value) in enumerate(invalid):
            with self.subTest(key=key, value=value):
                original = intent["retrieval"][key]
                intent["retrieval"][key] = value
                try:
                    with self.assertRaisesRegex(ValueError, key):
                        self._load(f"sourcebudgetbad{count}", data)
                finally:
                    intent["retrieval"][key] = original

    def test_source_first_limits_cannot_be_ignored_by_other_modes(self):
        data = _type_data()
        intent = next(iter(data["intents"].values()))
        for count, (mode, strategy) in enumerate((
            ("search", "hybrid"), ("document", "source_first"),
            ("none", "source_first"),
        )):
            with self.subTest(mode=mode, strategy=strategy):
                intent["retrieval"] = {
                    "mode": mode, "strategy": strategy,
                    "max_passages": 8,
                }
                with self.assertRaisesRegex(ValueError, "source-first"):
                    self._load(f"budgetmode{count}", data)

    def test_history_recent_turns_range(self) -> None:
        """Accept zero through one hundred turns, rejecting invalid values."""

        for index, value in enumerate((0, 1, 20, 21, 99, 100)):
            data = _type_data()
            data["history"] = {"recent_turns": value}
            self.assertEqual(
                self._load(f"recentok{index}", data).recent_turns,
                value,
            )
        for index, value in enumerate((-1, 101, "2", None, False, 2.5)):
            with self.subTest(value=value):
                data = _type_data()
                data["history"] = {"recent_turns": value}
                with self.assertRaisesRegex(ValueError, "recent_turns"):
                    self._load(f"recentbad{index}", data)

    def test_intents_and_default_intent_must_be_consistent(self) -> None:
        """Reject empty mappings and missing default-intent references."""

        for index, (field, value, error) in enumerate((
            ("intents", {}, "at least one intent"),
            ("default_intent", "missing", "default_intent"),
        )):
            data = _type_data()
            data[field] = value
            with self.assertRaisesRegex(ValueError, error):
                self._load(f"defaultbad{index}", data)
        data = _type_data()
        data["intents"]["second"] = {
            "description": "Answer another question.",
            "system_prompt": "Be concise.",
        }
        data["default_intent"] = "second"
        self.assertEqual(
            self._load("defaultsecond", data).default_intent,
            "second",
        )


class TestIntentFields(ProjectTypeFixture):
    """Validate independent intent contracts using synthetic definitions."""

    def _with_intent(self, **changes: object) -> dict[str, object]:
        """Return a fresh type whose only intent contains these fields.

        Parameters
        ----------
        **changes: object
            Intent overrides to exercise independently.

        Returns
        -------
        dict[str, object]
            Synthetic type with one modified intent.

        """

        data = _type_data()
        data["intents"]["explain"].update(changes)
        return data

    def test_omitted_thinking_defaults_to_inheritance(self) -> None:
        """Old JSON definitions acquire no explicit thinking policy."""

        intent = self._load("omittedthinking", _type_data()).intents[
            "explain"
        ]
        self.assertIsNone(intent.thinking)

    def test_thinking_accepts_only_boolean_or_null_values(self) -> None:
        """Preserve explicit true, false, and null without coercion."""

        for index, value in enumerate((True, False, None)):
            with self.subTest(value=value):
                data = self._with_intent(thinking=value)
                intent = self._load(f"thinking{index}", data).intents[
                    "explain"
                ]
                self.assertIs(intent.thinking, value)

    def test_thinking_rejects_other_json_types(self) -> None:
        """Reject strings and numbers, including bool-like zero and one."""

        invalid = ("false", "true", "null", "", 0, 1, 0.0, 1.0, [], {})
        for index, value in enumerate(invalid):
            with self.subTest(value=value):
                data = self._with_intent(thinking=value)
                with self.assertRaisesRegex(
                    ValueError, "thinking must be true, false, or null"
                ) as caught:
                    self._load(f"invalidthinking{index}", data)
                self.assertIn("samples.intents.explain", str(caught.exception))

    def test_tool_thinking_accepts_only_inheritance(self) -> None:
        """Do not imply control over internal model calls made by tools."""

        for index, options in enumerate(({}, {"thinking": None})):
            with self.subTest(options=options):
                data = self._with_intent(
                    tool={"name": "scan_and_list"}, **options
                )
                intent = self._load(f"toolthinking{index}", data).intents[
                    "explain"
                ]
                self.assertIsNone(intent.thinking)
                self.assertIsNotNone(intent.tool)
        for value in (True, False):
            with self.subTest(value=value):
                data = self._with_intent(
                    tool={"name": "scan_and_list"}, thinking=value
                )
                with self.assertRaisesRegex(
                    ValueError, "thinking must be null or omitted"
                ) as caught:
                    self._load(f"tooloverride{value}", data)
                self.assertIn("samples.intents.explain", str(caught.exception))

    def test_invalid_intent_identifiers_and_non_object_values(self) -> None:
        """Reject bad names and arrays where an intent object is needed."""

        for index, intent_id in enumerate(
            ("Invalid", "0number", "a" * 33, "has space")
        ):
            with self.subTest(intent_id=intent_id):
                data = _type_data()
                data["intents"] = {intent_id: data["intents"]["explain"]}
                with self.assertRaisesRegex(ValueError, "Intent ID"):
                    self._load(f"intentid{index}", data)
        data = _type_data()
        data["intents"] = {"explain": ["not", "an", "object"]}
        with self.assertRaisesRegex(ValueError, "must be an object"):
            self._load("intentobject", data)

    def test_intent_required_descriptions_and_instructions(self) -> None:
        """Require guidance for an LLM intent but not for a tool intent."""

        for index, field in enumerate(("description", "system_prompt")):
            for invalid in (None, "", 23):
                data = self._with_intent(**{field: invalid})
                with self.assertRaisesRegex(ValueError, field):
                    self._load(f"itext{index}{str(invalid)}", data)
        data = _type_data()
        del data["intents"]["explain"]["system_prompt"]
        with self.assertRaisesRegex(ValueError, "system_prompt"):
            self._load("noinstruction", data)

    def test_intent_default_and_explicit_history_policies(self) -> None:
        """Allow each policy and reject unknown policy or history shape."""

        for index, policy in enumerate(HISTORY_POLICY_CHOICES):
            data = self._with_intent(history={"policy": policy})
            item = self._load(f"policy{index}", data)
            self.assertEqual(item.intents["explain"].history_policy, policy)
        for index, bad in enumerate(("all", 1, True, [], None)):
            with self.subTest(bad=bad):
                data = self._with_intent(history={"policy": bad})
                with self.assertRaisesRegex(ValueError, "history policy"):
                    self._load(f"policybad{index}", data)
        with self.assertRaisesRegex(ValueError, "history"):
            self._load("historyarray", self._with_intent(history=[]))

    def test_retrieval_modes_and_strategy(self) -> None:
        """Accept supported search modes and the hybrid strategy."""

        for index, mode in enumerate(RETRIEVAL_MODE_CHOICES):
            data = self._with_intent(
                retrieval={"mode": mode, "strategy": "hybrid"}
            )
            item = self._load(f"mode{index}", data).intents["explain"]
            self.assertEqual(item.retrieval_mode, mode)
            self.assertEqual(item.retrieval_strategy, "hybrid")
        for field, invalid, error in (
            ("mode", "unknown", "retrieval mode"),
            ("strategy", "semantic", "retrieval strategy"),
        ):
            data = self._with_intent(retrieval={field: invalid})
            with self.assertRaisesRegex(ValueError, error):
                self._load(f"invalid{field}", data)
        with self.assertRaisesRegex(ValueError, "retrieval"):
            self._load("retrievalarray", self._with_intent(retrieval=[]))

    def test_related_evidence_requires_a_boolean(self) -> None:
        """Do not accept numeric truthiness for retrieval configuration."""

        for index, valid in enumerate((True, False)):
            data = self._with_intent(
                retrieval={"include_related_evidence": valid}
            )
            self.assertIs(
                self._load(f"relatedok{index}", data)
                .intents["explain"].include_related_evidence,
                valid,
            )
        for index, bad in enumerate(("yes", None, 0, 1, [])):
            data = self._with_intent(
                retrieval={"include_related_evidence": bad}
            )
            with self.assertRaisesRegex(ValueError,
                                        "include_related_evidence"):
                self._load(f"relatedbad{index}", data)

    def test_source_code_preference_requires_a_boolean(self) -> None:
        """Parse the optional preference for source code over docs."""

        default = self._load("preferdefault", self._with_intent())
        self.assertFalse(default.intents["explain"].prefer_source_code)
        data = self._with_intent(retrieval={"prefer_source_code": True})
        self.assertTrue(
            self._load("preferok", data)
            .intents["explain"].prefer_source_code
        )
        for index, bad in enumerate(("yes", None, 1)):
            data = self._with_intent(retrieval={"prefer_source_code": bad})
            with self.assertRaisesRegex(ValueError, "prefer_source_code"):
                self._load(f"preferbad{index}", data)

    def test_code_explanation_prefers_source_code(self) -> None:
        """Code explanation and debugging rank documentation later."""

        from kilin.project_type import project_type

        intents = project_type("code").intents
        self.assertTrue(intents["code_explanation"].prefer_source_code)
        self.assertTrue(intents["debugging"].prefer_source_code)
        self.assertFalse(intents["code_generation"].prefer_source_code)

    def test_document_token_budget_boundaries(self) -> None:
        """Enforce inclusive minimum and maximum evidence budgets."""

        for index, valid in enumerate((1000, 200000)):
            data = self._with_intent(retrieval={"max_tokens": valid})
            self.assertEqual(
                self._load(f"budgetok{index}", data)
                .intents["explain"].max_tokens,
                valid,
            )
        for index, bad in enumerate((0, 999, 200001, True, "30000", 1.5)):
            data = self._with_intent(retrieval={"max_tokens": bad})
            with self.assertRaisesRegex(ValueError, "max_tokens"):
                self._load(f"budgetbad{index}", data)

    def test_answer_guidance_and_shortcuts(self) -> None:
        """Preserve absent-versus-empty guidance and compile shortcuts."""

        data = self._with_intent(
            answer_guidance="  Be direct.  ",
            shortcuts=["hello", "YES|NO"],
        )
        intent = self._load("shortcutok", data).intents["explain"]
        self.assertEqual(intent.answer_guidance, "Be direct.")
        self.assertEqual(len(intent.shortcuts), 2)
        self.assertTrue(intent.shortcuts[0].fullmatch("HELLO"))
        self.assertTrue(intent.shortcuts[1].fullmatch("no"))
        data = self._with_intent(answer_guidance="  ")
        self.assertEqual(
            self._load("guidanceblank", data)
            .intents["explain"].answer_guidance,
            "",
        )
        for index, bad in enumerate((False, 1, [])):
            data = self._with_intent(answer_guidance=bad)
            with self.assertRaisesRegex(ValueError, "answer_guidance"):
                self._load(f"guidancebad{index}", data)
        for index, bad in enumerate(("hi", [""], [1])):
            data = self._with_intent(shortcuts=bad)
            with self.assertRaisesRegex(ValueError, "shortcuts"):
                self._load(f"shortcutsbad{index}", data)
        with self.assertRaisesRegex(ValueError, "regular expression"):
            self._load("regexbad", self._with_intent(shortcuts=["("]))

    def test_cues_and_requires_are_compiled_patterns(self) -> None:
        """Compile wording cues and required words case-insensitively."""

        data = self._with_intent(cues=[r"\boverview of\b"], requires=["cite"])
        intent = self._load("cuesok", data).intents["explain"]
        self.assertTrue(intent.cues[0].search("An OVERVIEW of it"))
        self.assertTrue(intent.requires[0].search("does it CITE x?"))
        plain = self._load("cuesnone", self._with_intent())
        self.assertEqual(plain.intents["explain"].cues, ())
        self.assertEqual(plain.intents["explain"].requires, ())
        for index, field in enumerate(("cues", "requires")):
            with self.assertRaisesRegex(ValueError, field):
                self._load(
                    f"cuesbad{index}", self._with_intent(**{field: ["("]})
                )

    def test_unknown_intent_and_nested_keys_are_rejected(self) -> None:
        """Reject unexpected intent, history, and retrieval properties."""

        scenarios = (
            (("system_promt",), "samples.intents.explain"),
            (("history", "polcy"), "samples.intents.explain.history"),
            (
                ("retrieval", "include_releated_evidence"),
                "samples.intents.explain.retrieval",
            ),
        )
        for index, (keys, location) in enumerate(scenarios):
            with self.subTest(keys=keys):
                data = _type_data()
                target = data["intents"]["explain"]
                for key in keys[:-1]:
                    target = target.setdefault(key, {})
                target[keys[-1]] = "ignored before validation"
                with self.assertRaisesRegex(
                    ValueError, "unknown settings"
                ) as caught:
                    self._load(f"unknownintent{index}", data)
                self.assertIn(location, str(caught.exception))
                self.assertIn(keys[-1], str(caught.exception))

    def test_tool_defaults_and_retrieval_incompatibility(self) -> None:
        """Tools default to no retrieval and permit absent instructions."""

        data = _type_data()
        data["intents"] = {
            "use_tool": {
                "description": "Find mentions.",
                "tool": {"name": "scan_and_list"},
            },
        }
        result = self._load("toolvalid", data).intents["use_tool"]
        self.assertEqual(result.retrieval_mode, "none")
        self.assertEqual(result.system_prompt, "")
        self.assertIsNotNone(result.tool)
        bad = deepcopy(data)
        bad["intents"]["use_tool"]["retrieval"] = {"mode": "search"}
        with self.assertRaisesRegex(ValueError, "replaces retrieval"):
            self._load("toolsearch", bad)
        bad = deepcopy(data)
        bad["intents"]["use_tool"]["tool"] = {"name": "not_a_tool"}
        with self.assertRaisesRegex(ValueError, "tool"):
            self._load("unknowntool", bad)

    def test_tool_configuration_validation_is_independent_of_json_intents(
        self,
    ) -> None:
        """Use synthetic tool fixtures instead of named packaged intents."""

        base = _type_data()
        base["intents"] = {
            "delegate": {
                "description": "Process matches.",
                "tool": {"name": "scan_and_list"},
            },
        }
        for index, (options, message) in enumerate((
            ({"context_lines": -1}, "context_lines"),
            ({"whole_words": "yes"}, "whole_words"),
            ({"unknown_option": 1}, "unknown"),
        )):
            data = deepcopy(base)
            data["intents"]["delegate"]["tool"].update(options)
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, message):
                    self._load(f"toolbad{index}", data)
        data = deepcopy(base)
        data["intents"]["delegate"]["tool"] = {
            "name": "batch_classify",
            "categories": ["one", "two"],
            "candidates": "mentions",
        }
        result = self._load("classifyvalid", data)
        self.assertEqual(
            result.intents["delegate"].tool.settings.categories,
            ("one", "two"),
        )
        data["intents"]["delegate"]["tool"]["categories"] = ["one"]
        with self.assertRaisesRegex(ValueError, "2-8"):
            self._load("classifybad", data)


if __name__ == "__main__":
    unittest.main()
