"""Test KíLín project profiles, configuration, and managed workspaces.

Exercise real profile files and disposable source directories without
loading models or writing to user projects. Cross-check that packaged
project-type definitions supply the defaults used by real projects, while
avoiding assumptions about mutable prompts and intent names.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import FrozenInstanceError, asdict
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from kilin import project
from kilin.defaults import load_defaults_section
from kilin.model_defaults import (
    DEFAULT_CHAT_MODEL_MLX,
    DEFAULT_CHAT_MODEL_TORCH,
    DEFAULT_EMBEDDING_MODEL_MLX,
    DEFAULT_EMBEDDING_MODEL_TORCH,
)
from kilin.project_type import load_project_types, project_type


class ProjectFixture(unittest.TestCase):
    """Provide isolated profiles and source directories for each case."""

    def setUp(self) -> None:
        """Create a temporary filesystem tree and select built-in types.

        """

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def _folder(self, name: str) -> Path:
        """Create one empty project or source folder.

        Parameters
        ----------
        name: str
            Folder name relative to the temporary root.

        Returns
        -------
        pathlib.Path
            Newly created directory.

        """

        directory = self.root / name
        directory.mkdir(parents=True)
        return directory

    def _profile(
        self,
        name: str = "configured",
        *,
        type_id: str = "code",
        source: str = ".",
    ) -> Path:
        """Write a profile using the production template and JSON defaults.

        Parameters
        ----------
        name: str, optional
            Unique directory name. Defaults to ``configured``.
        type_id: str, optional
            Existing built-in project type. Defaults to ``code``.
        source: str, optional
            Relative or absolute source path. Defaults to the project root.

        Returns
        -------
        pathlib.Path
            Directory containing the newly written profile.

        """

        directory = self._folder(name)
        definition = project_type(type_id)
        text = project._default_profile_text(
            type_id, source, ", ".join(definition.extensions)
        )
        (directory / "profile.ini").write_text(
            text, encoding="utf-8"
        )
        return directory

    def _edit(
        self,
        root: Path,
        section: str,
        option: str,
        value: str | None,
    ) -> None:
        """Replace or remove one INI setting in a fixture profile.

        Parameters
        ----------
        root: pathlib.Path
            Project directory containing ``profile.ini``.
        section: str
            Section containing the setting.
        option: str
            Setting key to replace or remove.
        value: str | None
            New setting text, or None to remove the setting.

        """

        path = root / "profile.ini"
        parser = ConfigParser()
        parser.read(path, encoding="utf-8")
        if value is None:
            parser.remove_option(section, option)
        else:
            parser.set(section, option, value)
        with path.open("w", encoding="utf-8") as stream:
            parser.write(stream)

    def _write(self, directory: Path, relative: str, content: str) -> Path:
        """Create one text file under a temporary source directory.

        Parameters
        ----------
        directory: pathlib.Path
            Source root under which the file is created.
        relative: str
            Relative source file path.
        content: str
            File contents.

        Returns
        -------
        pathlib.Path
            Newly written file.

        """

        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


class TestProjectTypeIds(ProjectFixture):
    """Check the mapping from type definitions to project defaults."""

    def test_registry_identifiers_are_not_a_builtin_allowlist(self) -> None:
        """Expose all configured types without requiring standard names."""

        current = load_project_types()
        self.assertEqual(project.PROJECT_TYPE_IDS, tuple(current))
        self.assertTrue(project.PROJECT_TYPE_IDS)
        self.assertFalse(hasattr(project, "_check_built_in_kinds"))
        self.assertEqual(project.PROJECT_TYPE_ID_CODE, "code")
        self.assertEqual(project.PROJECT_TYPE_ID_LITERATURE, "literature")

    def test_default_profiles_match_loaded_packaged_definitions(self) -> None:
        """File filters and size limits follow data, not hard-coded values."""

        for type_id, definition in load_project_types().items():
            with self.subTest(type_id=type_id):
                profile = project.default_project_profile(type_id)
                self.assertEqual(profile.type_id, type_id)
                self.assertEqual(profile.source, "sources")
                self.assertEqual(profile.extensions, definition.extensions)
                self.assertEqual(
                    profile.max_file_bytes, definition.max_file_bytes
                )
                self.assertEqual(profile.backend, "auto")
                self.assertEqual(profile.query_format, "qwen3")
                self.assertTrue(profile.allow_download)
                self.assertGreater(profile.retrieval_count, 0)
                self.assertGreater(profile.max_tokens, 0)
                self.assertLess(
                    profile.overlap_tokens, profile.target_tokens
                )

    def test_default_profiles_use_canonical_model_defaults(self) -> None:
        """Resolve chat and embedding settings for each selected backend."""

        for type_id in project.PROJECT_TYPE_IDS:
            with self.subTest(type_id=type_id):
                item = project.default_project_profile(type_id)
                self.assertEqual(item.chat_mlx, DEFAULT_CHAT_MODEL_MLX)
                self.assertEqual(item.chat_torch, DEFAULT_CHAT_MODEL_TORCH)
                self.assertEqual(
                    item.embedding_mlx, DEFAULT_EMBEDDING_MODEL_MLX
                )
                self.assertEqual(
                    item.embedding_torch, DEFAULT_EMBEDDING_MODEL_TORCH
                )
                self.assertEqual(item.chat_model("mlx"), item.chat_mlx)
                self.assertEqual(
                    item.chat_model("torch"), item.chat_torch
                )
                self.assertEqual(
                    item.embedding_model("mlx"), item.embedding_mlx
                )
                self.assertEqual(
                    item.embedding_model("torch"), item.embedding_torch
                )
                with self.assertRaises(FrozenInstanceError):
                    item.type_id = "other"

    def test_invalid_default_profile_kind_has_clear_error(self) -> None:
        """Reject unknown types before constructing a default profile."""

        with self.assertRaisesRegex(ValueError, "type_id must be one of"):
            project.default_project_profile("unknown")


class TestProjectInitialization(ProjectFixture):
    """Exercise creating and reopening project profiles in actual folders."""

    def test_initialize_both_types_without_modifying_source_files(
        self,
    ) -> None:
        """Initialize each built-in type and preserve existing data."""

        for type_id in ("code", "literature"):
            with self.subTest(type_id=type_id):
                directory = self._folder(type_id)
                original = self._write(
                    self.root, f"keep-{type_id}.txt", "keep this content"
                )
                result = project.initialize_project(directory, type_id)
                self.assertEqual(result.root, directory)
                self.assertEqual(result.source, directory / "sources")
                self.assertEqual(result.profile.type_id, type_id)
                self.assertEqual(
                    result.profile.extensions,
                    project_type(type_id).extensions,
                )
                self.assertEqual(
                    result.profile.max_file_bytes,
                    project_type(type_id).max_file_bytes,
                )
                self.assertTrue((directory / "profile.ini").is_file())
                self.assertTrue(result.indexes.is_dir())
                self.assertTrue(result.logs.is_dir())
                self.assertEqual(
                    original.read_text(encoding="utf-8"),
                    "keep this content",
                )

    def test_initialize_never_adopts_existing_source_folders(
        self,
    ) -> None:
        """Reject a folder that already holds source folders; adopt none."""

        for type_id, folder in (("code", "repository"),
                                ("literature", "documents")):
            with self.subTest(type_id=type_id):
                directory = self._folder(f"sources-{type_id}")
                (directory / folder).mkdir()
                (directory / "sources").mkdir()
                with self.assertRaises(project.WorkspaceOccupiedError):
                    project.initialize_project(directory, type_id)
                self.assertTrue((directory / folder).is_dir())
                self.assertTrue((directory / "sources").is_dir())
                self.assertFalse((directory / "profile.ini").exists())

    def test_initialize_creates_sources_for_every_type(self) -> None:
        """Every project type gets sources/, never a type-specific name."""

        for type_id in load_project_types():
            with self.subTest(type_id=type_id):
                directory = self.root / f"new-{type_id}"
                result = project.initialize_project(directory, type_id)
                self.assertEqual(result.source, directory / "sources")
                self.assertEqual(result.profile.source, "sources")
                self.assertEqual(
                    sorted(path.name for path in directory.iterdir()),
                    ["indexes", "logs", "outputs", "profile.ini",
                     "sources", "templates"],
                )

    def test_every_creation_path_uses_sources(self) -> None:
        """Core, mirror, and GUI creation give the same source folder."""

        from kilin.workspace import prepare_sources_workspace
        from kilin.workspace_resources import PROJECT_SOURCES_DIRECTORY

        original = self._folder("original")
        self._write(original, "notes.md", "text")
        for type_id in ("code", "literature"):
            with self.subTest(type_id=type_id):
                created = (
                    project.initialize_project(
                        self.root / f"core-{type_id}", type_id
                    ),
                    project.prepare_project_workspace(
                        original, type_id, self.root / f"mirror-{type_id}"
                    ),
                    prepare_sources_workspace(
                        [original], type_id, self.root / f"gui-{type_id}"
                    ),
                )
                for result in created:
                    self.assertEqual(
                        result.profile.source, PROJECT_SOURCES_DIRECTORY
                    )
                    self.assertEqual(
                        result.source, result.root / PROJECT_SOURCES_DIRECTORY
                    )

    def test_existing_source_folders_open_without_being_moved(
        self,
    ) -> None:
        """Older documents/ or repository/ projects stay as they are."""

        for type_id, folder in (
            ("literature", "documents"),
            ("code", "repository"),
            ("code", "sources"),
        ):
            with self.subTest(type_id=type_id, folder=folder):
                directory = self._profile(
                    f"saved-{type_id}-{folder}", type_id=type_id,
                    source=folder,
                )
                paper = self._write(directory / folder, "a.md", "text")
                opened = project.open_project(directory)
                self.assertEqual(opened.source, directory / folder)
                self.assertEqual(opened.profile.source, folder)
                # Opening never renames, moves, or adds a source folder.
                self.assertEqual(paper.read_text(encoding="utf-8"), "text")
                self.assertEqual(
                    sorted(path.name for path in directory.iterdir()),
                    sorted([folder, "profile.ini"]),
                )

    def test_initialize_is_idempotent_and_preserves_existing_profile(
        self,
    ) -> None:
        """Do not replace an existing valid profile during initialization."""

        directory = self._profile("existing")
        self._edit(directory, "retrieval", "count", "17")
        before = (directory / "profile.ini").read_bytes()
        with self.assertRaises(project.ProjectAlreadyExistsError):
            project.initialize_project(directory, "code")
        first = project.open_project(directory)
        second = project.open_project(directory)
        self.assertEqual(first.profile.retrieval_count, 17)
        self.assertEqual(first, second)
        self.assertEqual((directory / "profile.ini").read_bytes(), before)

    def test_initialize_rejects_unknown_or_conflicting_kinds(self) -> None:
        """Invalid or conflicting type IDs must not overwrite the profile."""

        directory = self._profile("wrong-type_id")
        before = (directory / "profile.ini").read_bytes()
        with self.assertRaisesRegex(ValueError, "type_id must be one of"):
            project.initialize_project(directory, "unregistered")
        with self.assertRaises(project.ProjectAlreadyExistsError):
            project.initialize_project(directory, "literature")
        self.assertEqual((directory / "profile.ini").read_bytes(), before)

    def test_initialize_rejects_missing_file_or_nondirectory(self) -> None:
        """Create missing destinations but reject paths to regular files."""

        created = project.initialize_project(self.root / "missing", "code")
        self.assertTrue((created.root / "profile.ini").is_file())
        file_path = self._write(self.root, "regular.txt", "data")
        with self.assertRaises(NotADirectoryError):
            project.initialize_project(file_path, "code")

    def test_open_project_requires_profile_and_valid_source(self) -> None:
        """Reject missing project paths and invalid configured sources."""

        with self.assertRaises(FileNotFoundError):
            project.open_project(self.root / "not_there")
        path = self._write(self.root, "not_a_folder.txt", "data")
        with self.assertRaises(NotADirectoryError):
            project.open_project(path)
        directory = self._folder("missingprofile")
        with self.assertRaisesRegex(FileNotFoundError, "profile"):
            project.open_project(directory)
        directory = self._profile("missingsource", source="missing")
        with self.assertRaises(FileNotFoundError):
            project.open_project(directory)
        directory = self._profile("file-as-source", source="some.txt")
        self._write(directory, "some.txt", "data")
        with self.assertRaises(NotADirectoryError):
            project.open_project(directory)

    def test_open_accepts_absolute_source_outside_project(self) -> None:
        """Resolve absolute source paths without replacing source content."""

        source = self._folder("external")
        directory = self._profile("absolute", source=str(source))
        self.assertEqual(project.open_project(directory).source, source)
        self.assertFalse((source / "profile.ini").exists())


class TestProjectProfileValidation(ProjectFixture):
    """Check every INI field and the important numeric boundary rules."""

    def test_default_template_round_trip(self) -> None:
        """Parse the canonical INI format without changing its contents."""

        for type_id in ("code", "literature"):
            with self.subTest(type_id=type_id):
                root = self._profile(f"roundtrip-{type_id}", type_id=type_id)
                parsed = project.open_project(root)
                default = project.default_project_profile(type_id)
                self.assertEqual(
                    asdict(parsed.profile),
                    {**asdict(default), "source": "."},
                )

    def test_missing_required_sections_and_options(self) -> None:
        """Identify missing INI keys using section-qualified errors."""

        for index, (section, option) in enumerate((
            ("project", "type_id"),
            ("project", "source"),
            ("project", "extensions"),
            ("project", "max_file_bytes"),
            ("models", "backend"),
            ("models", "query_format"),
            ("models", "allow_download"),
            ("models", "chat_torch"),
            ("models", "chat_mlx"),
            ("models", "embedding_torch"),
            ("models", "embedding_mlx"),
            ("chunking", "target_tokens"),
            ("chunking", "max_tokens"),
            ("chunking", "overlap_tokens"),
            ("retrieval", "count"),
            ("generation", "thinking"),
            ("generation", "temperature"),
            ("generation", "max_new_tokens"),
        )):
            with self.subTest(section=section, option=option):
                root = self._profile(f"missing{index}")
                self._edit(root, section, option, None)
                with self.assertRaisesRegex(
                    ValueError, rf"{section}\.{option}"
                ):
                    project.open_project(root)

    def test_empty_required_settings_are_rejected(self) -> None:
        """Reject blank settings instead of applying implicit values."""

        for index, (section, option) in enumerate((
            ("project", "source"),
            ("models", "chat_torch"),
            ("models", "backend"),
            ("generation", "temperature"),
        )):
            root = self._profile(f"blank{index}")
            self._edit(root, section, option, "   ")
            with self.assertRaisesRegex(ValueError, "empty"):
                project.open_project(root)

    def test_invalid_kind_backend_query_format_and_extensions(
        self,
    ) -> None:
        """Reject unsupported modes and malformed source suffix lists."""

        for index, (section, option, value, error) in enumerate((
            ("project", "type_id", "something_else", "project.type_id"),
            ("models", "backend", "cuda", "models.backend"),
            ("models", "query_format", "raw", "models.query_format"),
            ("project", "extensions", "py", "project.extensions"),
            ("project", "extensions", ",, ", "project.extensions"),
        )):
            root = self._profile(f"mode{index}")
            self._edit(root, section, option, value)
            with self.assertRaisesRegex(ValueError, error):
                project.open_project(root)

    def test_extensions_are_normalized_on_read(self) -> None:
        """Strip whitespace and lowercase each configured source suffix."""

        root = self._profile("extensions")
        self._edit(root, "project", "extensions", " .PY, .Md,, ")
        self.assertEqual(
            project.open_project(root).profile.extensions,
            (".py", ".md"),
        )

    def test_invalid_chunk_sizing_rules(self) -> None:
        """Require positive chunks and bounded overlap."""

        target = load_defaults_section("PROJECT_PROFILE")["TARGET_TOKENS"]
        cases = (
            ("target_tokens", "0", "positive"),
            ("max_tokens", "0", "positive"),
            ("overlap_tokens", "-1", "cannot be negative"),
            ("target_tokens", "invalid", "whole number"),
            ("max_tokens", "1", "reach target_tokens"),
            ("overlap_tokens", str(target), "below target_tokens"),
        )
        for index, (option, value, error) in enumerate(cases):
            with self.subTest(option=option, value=value):
                root = self._profile(f"chunk{index}")
                self._edit(root, "chunking", option, value)
                with self.assertRaisesRegex(ValueError, error):
                    project.open_project(root)

    def test_invalid_positive_integer_fields(self) -> None:
        """Reject zero, negative, or nonnumeric positive settings."""

        for index, (section, option, value, error) in enumerate((
            ("project", "max_file_bytes", "0", "positive"),
            ("project", "max_file_bytes", "NaN", "whole number"),
            ("retrieval", "count", "-1", "positive"),
            ("retrieval", "count", "1.5", "whole number"),
        )):
            root = self._profile(f"integer{index}")
            self._edit(root, section, option, value)
            with self.assertRaisesRegex(ValueError, error):
                project.open_project(root)

    def test_boolean_settings_use_configparser_spellings(self) -> None:
        """Accept canonical booleans and reject arbitrary text."""

        root = self._profile("booleans")
        self._edit(root, "models", "allow_download", "off")
        self._edit(root, "generation", "thinking", "yes")
        parsed = project.open_project(root).profile
        self.assertFalse(parsed.allow_download)
        self.assertTrue(parsed.thinking)
        for index, option in enumerate(("thinking", "allow_download")):
            root = self._profile(f"badbool{index}")
            section = "generation" if option == "thinking" else "models"
            self._edit(root, section, option, "possibly")
            with self.assertRaisesRegex(ValueError, "true or false"):
                project.open_project(root)

    def test_cjk_blocking_setting_is_optional(self) -> None:
        """New profiles write the flag; older profiles use the default."""

        root = self._profile("cjk")
        text = (root / "profile.ini").read_text(encoding="utf-8")
        self.assertIn("block_unrequested_cjk = true", text)
        self.assertTrue(
            project.open_project(root).profile.block_unrequested_cjk
        )
        self._edit(root, "generation", "block_unrequested_cjk", "false")
        self.assertFalse(
            project.open_project(root).profile.block_unrequested_cjk
        )
        self._edit(root, "generation", "block_unrequested_cjk", None)
        self.assertTrue(
            project.open_project(root).profile.block_unrequested_cjk
        )
        self._edit(root, "generation", "block_unrequested_cjk", "possibly")
        with self.assertRaisesRegex(ValueError, "true or false"):
            project.open_project(root)

    def test_generation_temperature_boundaries(self) -> None:
        """Allow zero/positive temperatures and reject negative/invalid."""

        for index, temperature in enumerate(("0", "0.125", "2")):
            root = self._profile(f"tempvalid{index}")
            self._edit(root, "generation", "temperature", temperature)
            self.assertEqual(
                project.open_project(root).profile.temperature,
                float(temperature),
            )
        for index, (value, error) in enumerate((
            ("-0.1", "cannot be negative"),
            ("oops", "must be a number"),
        )):
            root = self._profile(f"tempbad{index}")
            self._edit(root, "generation", "temperature", value)
            with self.assertRaisesRegex(ValueError, error):
                project.open_project(root)

    def test_nonfinite_temperature_is_rejected(self) -> None:
        """Reject NaN and infinity before initializing a chat backend."""

        for index, value in enumerate((
            "nan", "NaN", "inf", "+Infinity", "-inf",
        )):
            with self.subTest(value=value):
                root = self._profile(f"nonfinite-temperature{index}")
                self._edit(root, "generation", "temperature", value)
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    project.open_project(root)

    def test_invalid_extension_is_rejected_on_profile_read(self) -> None:
        """Validate literal suffixes before repository scanning."""

        for index, extension in enumerate((
            ".py*", ".pdf?", ".py/file", ".py file", ".py!",
        )):
            with self.subTest(extension=extension):
                root = self._profile(f"bad-extension{index}")
                self._edit(root, "project", "extensions", extension)
                with self.assertRaisesRegex(ValueError, "extensions"):
                    project.open_project(root)

    def test_profile_extensions_normalize_and_deduplicate(self) -> None:
        """Profiles use the same normalized suffixes as project types."""

        root = self._profile("duplicate-extensions")
        self._edit(root, "project", "extensions", ".PY, .py, .tar.GZ")
        self.assertEqual(
            project.open_project(root).profile.extensions,
            (".py", ".tar.gz"),
        )

    def test_max_new_tokens_accepts_none_or_positive_integer(self) -> None:
        """Parse optional generation limits without negative values."""

        for index, (raw, expected) in enumerate((
            ("none", None), ("NONE", None), ("32", 32),
        )):
            root = self._profile(f"tokensvalid{index}")
            self._edit(root, "generation", "max_new_tokens", raw)
            self.assertEqual(
                project.open_project(root).profile.max_new_tokens,
                expected,
            )
        for index, raw in enumerate(("0", "-1", "oops")):
            root = self._profile(f"tokensbad{index}")
            self._edit(root, "generation", "max_new_tokens", raw)
            with self.assertRaisesRegex(ValueError, "positive or none"):
                project.open_project(root)


class TestConfigureProject(ProjectFixture):
    """Test validated updates, preservation, tracing, and failed writes."""

    def test_configure_valid_settings_persists_and_preserves_source(
        self,
    ) -> None:
        """Round-trip editable settings without changing source ownership."""

        root = self._profile("configure")
        before = project.open_project(root)
        configured = project.configure_project(root, {
            "extensions": [".TXT", ".MD"],
            "max_file_bytes": 12345,
            "backend": "torch",
            "allow_download": False,
            "query_format": "plain",
            "target_tokens": 100,
            "max_tokens": 250,
            "overlap_tokens": 20,
            "retrieval_count": 9,
            "thinking": True,
            "temperature": 0.4,
            "max_new_tokens": None,
        })
        self.assertEqual(configured.source, before.source)
        self.assertEqual(configured.profile.extensions,
                         (".txt", ".md"))
        self.assertEqual(configured.profile.max_file_bytes, 12345)
        self.assertEqual(configured.profile.backend, "torch")
        self.assertFalse(configured.profile.allow_download)
        self.assertEqual(configured.profile.query_format, "plain")
        self.assertEqual(configured.profile.target_tokens, 100)
        self.assertEqual(configured.profile.max_tokens, 250)
        self.assertEqual(configured.profile.overlap_tokens, 20)
        self.assertEqual(configured.profile.retrieval_count, 9)
        self.assertTrue(configured.profile.thinking)
        self.assertEqual(configured.profile.temperature, 0.4)
        self.assertIsNone(configured.profile.max_new_tokens)
        self.assertEqual(project.open_project(root), configured)
        self.assertFalse((root / "profile.ini.tmp").exists())

    def test_configure_all_model_choices(self) -> None:
        """Persist backend-specific model IDs without rewriting other keys."""

        root = self._profile("models")
        overrides = {
            "chat_mlx": "local/chat-mlx",
            "chat_torch": "local/chat-torch",
            "embedding_mlx": "local/embed-mlx",
            "embedding_torch": "local/embed-torch",
        }
        expected = project.configure_project(root, overrides).profile
        for key, value in overrides.items():
            with self.subTest(key=key):
                self.assertEqual(getattr(expected, key), value)
        self.assertEqual(
            expected.chat_model("mlx"), "local/chat-mlx"
        )
        self.assertEqual(
            expected.embedding_model("torch"), "local/embed-torch"
        )

    def test_configure_rejects_unknown_settings_without_writing(
        self,
    ) -> None:
        """Refuse unexpected options, including source and project type."""

        root = self._profile("unknown")
        before = (root / "profile.ini").read_bytes()
        for key in ("source", "type_id", "not_a_setting"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "Unknown"):
                    project.configure_project(root, {key: "new"})
                self.assertEqual(
                    (root / "profile.ini").read_bytes(), before
                )

    def test_configure_invalid_settings_are_atomic(self) -> None:
        """Reject unsupported values before replacing a valid INI file."""

        root = self._profile("invalid-settings")
        before = (root / "profile.ini").read_bytes()
        for settings in (
            {"retrieval_count": 0},
            {"backend": "invalid"},
            {"target_tokens": 500, "max_tokens": 10},
            {"extensions": ["missing-dot"]},
            {"extensions": [".py*"]},
            {"temperature": -1},
            {"temperature": float("nan")},
            {"temperature": float("inf")},
            {"max_new_tokens": "unexpected"},
            {"thinking": "maybe"},
        ):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    project.configure_project(root, settings)
                self.assertEqual(
                    (root / "profile.ini").read_bytes(), before
                )

    def test_configure_empty_update_preserves_profile_values(self) -> None:
        """A no-op configuration remains valid and preserves settings."""

        root = self._profile("empty-update")
        before = project.open_project(root).profile
        updated = project.configure_project(root, {})
        self.assertEqual(updated.profile, before)

    def test_configure_emits_requested_saved_and_change_events(self) -> None:
        """Notify the caller-supplied trace sink of configuration updates."""

        root = self._profile("tracing")
        logger = MagicMock()
        configured = project.configure_project(
            root, {"retrieval_count": 8}, trace=logger
        )
        self.assertEqual(configured.profile.retrieval_count, 8)
        self.assertEqual(logger.emit.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in logger.emit.call_args_list],
            ["project.configuration.requested",
             "project.configuration.saved"],
        )
        logger.settings_changed.assert_called_once()
        self.assertEqual(
            logger.settings_changed.call_args.args[0],
            "project.settings.changed",
        )

    def test_configure_trace_errors_do_not_claim_atomic_rollback(
        self,
    ) -> None:
        """A late trace failure does not revert already saved settings."""

        root = self._profile("trace-failure")
        logger = MagicMock()
        logger.settings_changed.side_effect = RuntimeError("trace error")
        with self.assertRaisesRegex(RuntimeError, "trace error"):
            project.configure_project(
                root, {"retrieval_count": 12}, trace=logger
            )
        self.assertEqual(
            project.open_project(root).profile.retrieval_count, 12
        )

    def test_configure_failure_before_write_keeps_profile(
        self,
    ) -> None:
        """Validation failure emits no success events and leaves INI."""

        root = self._profile("invalid-trace")
        initial = (root / "profile.ini").read_bytes()
        logger = MagicMock()
        with self.assertRaises(ValueError):
            project.configure_project(
                root, {"max_file_bytes": "unknown"}, trace=logger
            )
        self.assertEqual(
            (root / "profile.ini").read_bytes(), initial
        )
        logger.settings_changed.assert_not_called()
        self.assertEqual(logger.emit.call_count, 1)


class TestProjectWorkspace(ProjectFixture):
    """Test managed copies, change detection, ownership, and conflicts."""

    def setUp(self) -> None:
        """Prepare a disposable source repository and workspace folder."""

        super().setUp()
        self.source = self._folder("original")
        self.sandbox = self._folder("sandbox")
        self.suffix = project_type("code").extensions[0]
        self.file = f"pkg/module{self.suffix}"
        self._write(self.source, self.file, "first version\n")
        self._write(self.source, "ignore.unselected_ext", "ignore\n")

    def _prepare(self, type_id: str = "code") -> project.QAProject:
        """Synchronize the test source into its disposable workspace.

        Parameters
        ----------
        type_id: str, optional
            Existing project type, defaulting to code.

        Returns
        -------
        QAProject
            Project opened after synchronizing the source snapshot.

        """

        if hasattr(self, "mirror"):
            return project.synchronize_project_workspace(
                self.source, self.mirror.root
            )
        self.mirror = project.prepare_project_workspace(
            self.source, type_id, self.sandbox
        )
        return self.mirror

    def test_initial_workspace_copies_only_selected_sources(self) -> None:
        """Do not edit originals or mirror files excluded by source rules."""

        result = self._prepare()
        self.assertTrue(result.root.is_relative_to(self.sandbox))
        self.assertEqual(result.profile.type_id, "code")
        self.assertEqual(result.profile.source, "sources")
        self.assertEqual(result.source, result.root / "sources")
        self.assertEqual(
            (result.source / self.file).read_text(encoding="utf-8"),
            "first version\n",
        )
        self.assertFalse(
            (result.source / "ignore.unselected_ext").exists()
        )
        self.assertTrue(result.indexes.is_dir())
        self.assertTrue(result.logs.is_dir())
        self.assertTrue((result.root / "source_snapshot.json").is_file())
        manifest = json.loads(
            (result.root / "source_snapshot.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["source_directory"], str(self.source))
        self.assertEqual((self.source / self.file).read_text(encoding="utf-8"),
                         "first version\n")

    def test_reopen_unchanged_workspace_preserves_identical_copy(self) -> None:
        """Do not replace the snapshot file when no source changed."""

        first = self._prepare()
        snapshot = first.source / self.file
        before = snapshot.stat()
        second = self._prepare()
        self.assertEqual(first.root, second.root)
        self.assertEqual(snapshot.stat().st_ino, before.st_ino)
        self.assertEqual(
            snapshot.read_text(encoding="utf-8"), "first version\n"
        )

    def test_added_modified_and_removed_files_are_synchronized(self) -> None:
        """Reflect all source changes while leaving originals intact."""

        first = self._prepare()
        extra = f"pkg/extra{self.suffix}"
        self._write(self.source, self.file, "modified first version\n")
        self._write(self.source, extra, "added version\n")
        second = self._prepare()
        self.assertEqual(second.root, first.root)
        self.assertEqual(
            (second.source / self.file).read_text(encoding="utf-8"),
            "modified first version\n",
        )
        self.assertEqual(
            (second.source / extra).read_text(encoding="utf-8"),
            "added version\n",
        )
        (self.source / self.file).unlink()
        third = self._prepare()
        self.assertFalse((third.source / self.file).exists())
        self.assertTrue((third.source / extra).is_file())

    def test_deletion_prunes_empty_snapshot_directories(self) -> None:
        """Remove nested directories once their last selected file goes."""

        file = f"unneeded/deep/nested{self.suffix}"
        self._write(self.source, file, "temporary\n")
        first = self._prepare()
        self.assertTrue((first.source / "unneeded/deep").is_dir())
        (self.source / file).unlink()
        refreshed = self._prepare()
        self.assertFalse((refreshed.source / "unneeded").exists())
        self.assertTrue((refreshed.source / "pkg").exists())

    def test_corrupt_manifest_recovers_from_source(self) -> None:
        """An unreadable baseline triggers a complete snapshot refresh."""

        first = self._prepare()
        self._write(first.source, "stale.txt", "stale\n")
        (first.root / "source_snapshot.json").write_text(
            "{broken", encoding="utf-8"
        )
        refreshed = self._prepare()
        self.assertFalse((refreshed.source / "stale.txt").exists())
        self.assertEqual(
            (refreshed.source / self.file).read_text(encoding="utf-8"),
            "first version\n",
        )

    def test_stale_manifest_structure_rebuilds_snapshot(self) -> None:
        """Recover from a readable but structurally invalid manifest."""

        initial = self._prepare()
        snapshot_path = initial.root / "source_snapshot.json"
        snapshot_path.write_text(
            json.dumps({"source_directory": str(self.source)}),
            encoding="utf-8",
        )
        result = self._prepare()
        self.assertEqual(
            (result.source / self.file).read_text(encoding="utf-8"),
            "first version\n",
        )

    def test_collision_of_identical_source_basenames(self) -> None:
        """Give independently owned source paths distinct workspace roots."""

        root_a = self._folder("origin-a")
        root_b = self._folder("origin-b")
        source_a = root_a / "same"
        source_b = root_b / "same"
        source_a.mkdir()
        source_b.mkdir()
        self._write(source_a, f"one{self.suffix}", "first\n")
        self._write(source_b, f"two{self.suffix}", "second\n")
        first = project.prepare_project_workspace(
            source_a, "code", self.sandbox
        )
        second = project.prepare_project_workspace(
            source_b, "code", self.sandbox
        )
        self.assertNotEqual(first.root, second.root)
        self.assertTrue((first.source / f"one{self.suffix}").is_file())
        self.assertTrue((second.source / f"two{self.suffix}").is_file())
        self.assertEqual(
            project.synchronize_project_workspace(
                source_b, second.root
            ).root,
            second.root,
        )

    def test_workspace_rejects_wrong_kind_without_rewriting_profile(
        self,
    ) -> None:
        """Do not replace a managed workspace's registered project type."""

        original = self._prepare()
        before = (original.root / "profile.ini").read_bytes()
        independent = project.prepare_project_workspace(
            self.source, "literature", self.sandbox
        )
        self.assertNotEqual(independent.root, original.root)
        self.assertEqual(independent.profile.type_id, "literature")
        self.assertEqual((original.root / "profile.ini").read_bytes(),
                         before)

    def test_older_mirror_keeps_its_source_folder(self) -> None:
        """A mirror copied into the original's name syncs in place."""

        first = self._prepare()
        older = first.root / self.source.name
        first.source.rename(older)
        self._edit(first.root, "project", "source", self.source.name)
        self._write(self.source, "pkg/extra" + self.suffix, "new\n")

        result = self._prepare()

        self.assertEqual(result.source, older)
        self.assertFalse((result.root / "sources").exists())
        self.assertEqual(
            (older / "pkg" / f"extra{self.suffix}").read_text(
                encoding="utf-8"
            ),
            "new\n",
        )

    def test_workspace_rejects_tampered_profile_source(self) -> None:
        """Reject profile sources that no longer point to the owned copy."""

        first = self._prepare()
        (first.root / "other").mkdir()
        self._edit(first.root, "project", "source", "other")
        with self.assertRaisesRegex(ValueError, "project.source"):
            self._prepare()

    def test_workspace_rejects_bad_kind_paths_and_same_source_root(
        self,
    ) -> None:
        """Prevent unsupported kinds and an equal source/workspace root."""

        with self.assertRaisesRegex(ValueError, "type_id must be one of"):
            self._prepare("missing")
        with self.assertRaises(ValueError):
            project.prepare_project_workspace(
                self.source, "code", self.source
            )
        with self.assertRaises(FileNotFoundError):
            project.prepare_project_workspace(
                self.root / "missing", "code", self.sandbox
            )
        regular = self._write(self.root, "not-a-directory", "data")
        with self.assertRaises(NotADirectoryError):
            project.prepare_project_workspace(
                regular, "code", self.sandbox
            )

    def test_workspace_folder_name_is_sanitized(self) -> None:
        """Build a safe root name from a source basename with spaces."""

        source = self._folder("a special name")
        self._write(source, f"note{self.suffix}", "content\n")
        result = project.prepare_project_workspace(
            source, "code", self.sandbox
        )
        self.assertEqual(result.root.name, "project_a-special-name")

    def test_workspace_inside_source_does_not_recursively_copy_itself(
        self,
    ) -> None:
        """Exclude workspace files when its root is within the sources."""

        root = self.source / "workspaces"
        result = project.prepare_project_workspace(
            self.source, "code", root
        )
        self.assertTrue((result.source / self.file).is_file())
        self.assertFalse((result.source / "workspaces").exists())

    def test_unchanged_manifest_verifies_snapshot_content(
        self,
    ) -> None:
        """Detect modifications to a workspace copy of unchanged sources."""

        first = self._prepare()
        (first.source / self.file).write_text(
            "modified inside workspace\n", encoding="utf-8"
        )
        refreshed = self._prepare()
        self.assertEqual(
            (refreshed.source / self.file).read_text(encoding="utf-8"),
            "first version\n",
        )

    def test_workspace_rejects_symlinked_managed_source(self) -> None:
        """Never scan or synchronize a substituted external source copy."""

        first = self._prepare()
        outside = self._folder("symlink-target")
        shutil.rmtree(first.source)
        try:
            first.source.symlink_to(outside, target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("Creating directory symlinks is unavailable")
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            self._prepare()
        self.assertEqual(list(outside.iterdir()), [])

    def test_workspace_rejects_symlinked_managed_root(self) -> None:
        """Keep workspace ownership rooted within the chosen sandbox."""

        first = self._prepare()
        target = self.root / "displaced-managed-project"
        first.root.rename(target)
        try:
            first.root.symlink_to(target, target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("Creating directory symlinks is unavailable")
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            project.synchronize_project_workspace(self.source, first.root)
        self.assertTrue((target / "source_snapshot.json").is_file())

    def test_snapshot_heals_missing_selected_file(self) -> None:
        """Recover a deleted copy even when its original is unchanged."""

        first = self._prepare()
        (first.source / self.file).unlink()
        refreshed = self._prepare()
        self.assertEqual(
            (refreshed.source / self.file).read_text(encoding="utf-8"),
            "first version\n",
        )

    def test_snapshot_heals_same_size_content_change(self) -> None:
        """Verify SHA-256, not only file existence or file size."""

        first = self._prepare()
        (first.source / self.file).write_text(
            "other version\n", encoding="utf-8"
        )
        refreshed = self._prepare()
        self.assertEqual(
            (refreshed.source / self.file).read_text(encoding="utf-8"),
            "first version\n",
        )

    def test_snapshot_heals_extra_selected_file(self) -> None:
        """Remove selected content not present in the original manifest."""

        first = self._prepare()
        unexpected = self._write(
            first.source, f"pkg/unexpected{self.suffix}", "new file\n"
        )
        self._prepare()
        self.assertFalse(unexpected.exists())

    def test_snapshot_integrity_checks_previous_state_before_sync(
        self,
    ) -> None:
        """Repair tampered content even when the original also changed."""

        first = self._prepare()
        (first.source / self.file).write_text(
            "tampered version\n", encoding="utf-8"
        )
        new = f"pkg/new{self.suffix}"
        self._write(self.source, new, "new original\n")
        refreshed = self._prepare()
        self.assertEqual(
            (refreshed.source / self.file).read_text(encoding="utf-8"),
            "first version\n",
        )
        self.assertEqual(
            (refreshed.source / new).read_text(encoding="utf-8"),
            "new original\n",
        )

    def test_snapshot_rebuild_does_not_follow_symlink(self) -> None:
        """Treat a selected symlink in the copy as changed content."""

        first = self._prepare()
        untouched = self._write(self.root, "outside.txt", "outside\n")
        target = first.source / self.file
        target.unlink()
        try:
            target.symlink_to(untouched)
        except (NotImplementedError, OSError):
            self.skipTest("Creating file symlinks is unavailable")
        refreshed = self._prepare()
        self.assertFalse((refreshed.source / self.file).is_symlink())
        self.assertEqual(
            (refreshed.source / self.file).read_text(encoding="utf-8"),
            "first version\n",
        )
        self.assertEqual(untouched.read_text(encoding="utf-8"), "outside\n")

    def test_snapshot_incomplete_integrity_scan_is_not_accepted(
        self,
    ) -> None:
        """Never accept an incomplete snapshot verification as valid."""

        first = self._prepare()
        actual_scan = project.scan_repository

        def scan_with_failed_snapshot(directory, **options):
            report = actual_scan(directory, **options)
            if Path(directory) == first.source:
                report["complete"] = False
            return report

        with patch.object(
            project, "scan_repository", side_effect=scan_with_failed_snapshot
        ):
            with self.assertRaisesRegex(ValueError, "incomplete scan"):
                self._prepare()

    def test_snapshot_copy_is_stable_when_selected_files_unchanged(
        self,
    ) -> None:
        """Avoid unnecessary copies after successful content verification."""

        first = self._prepare()
        with patch.object(
            project.shutil, "copy2", wraps=project.shutil.copy2
        ) as copied:
            second = self._prepare()
        self.assertEqual(first.root, second.root)
        copied.assert_not_called()

    def test_removed_copy_can_disappear_after_integrity_check(
        self,
    ) -> None:
        """Tolerate a concurrently removed copied file during cleanup."""

        first = self._prepare()
        target = first.source / self.file
        (self.source / self.file).unlink()
        actual_verify = project._snapshot_matches_manifest

        def delete_after_verification(*args):
            verified = actual_verify(*args)
            target.unlink()
            return verified

        with patch.object(
            project,
            "_snapshot_matches_manifest",
            side_effect=delete_after_verification,
        ):
            refreshed = self._prepare()
        self.assertEqual(refreshed.root, first.root)
        self.assertFalse(target.exists())

    def test_workspace_uses_safe_fallback_for_punctuation_only_name(
        self,
    ) -> None:
        """Generate a nonempty workspace name for punctuation source IDs."""

        # Not "...": Windows drops trailing dots from folder names.
        source = self._folder("!!!")
        self._write(source, f"example{self.suffix}", "source")
        result = project.prepare_project_workspace(
            source, "code", self.sandbox
        )
        self.assertEqual(result.root.name, "project_project")
        self.assertTrue((result.source / f"example{self.suffix}").exists())

    def test_source_deleted_when_copy_was_already_removed(self) -> None:
        """Allow a deleted file to be absent from the mirrored snapshot."""

        result = self._prepare()
        (self.source / self.file).unlink()
        (result.source / self.file).unlink()
        reopened = self._prepare()
        self.assertFalse((reopened.source / self.file).exists())

    def test_workspace_handles_empty_selected_source(self) -> None:
        """Record ownership even when no source files are selected."""

        (self.source / self.file).unlink()
        result = self._prepare()
        self.assertTrue((result.root / "source_snapshot.json").is_file())
        self.assertFalse((result.source / self.file).exists())
        self.assertEqual(result.root, self._prepare().root)


class TestWorkspaceHelpers(ProjectFixture):
    """Cover manifest helpers and collision decisions independently."""

    def test_read_json_returns_none_for_missing_corrupt_and_nonobject(
        self,
    ) -> None:
        """Treat invalid or unreadable snapshot records as unavailable."""

        path = self.root / "manifest.json"
        self.assertIsNone(project._read_json_if_possible(path))
        for text in ("{", "[]", '"a"'):
            path.write_text(text, encoding="utf-8")
            self.assertIsNone(project._read_json_if_possible(path))
        path.write_text('{"key": 1}', encoding="utf-8")
        self.assertEqual(project._read_json_if_possible(path), {"key": 1})
        with patch.object(Path, "read_text", side_effect=OSError("bad")):
            self.assertIsNone(project._read_json_if_possible(path))

    def test_write_json_is_sorted_atomic_and_creates_parents(self) -> None:
        """Write deterministic JSON without leaving the temporary file."""

        path = self.root / "nested" / "manifest.json"
        project._write_json(path, {"b": 2, "a": 1})
        self.assertEqual(
            path.read_text(encoding="utf-8"), '{\n  "a": 1,\n  "b": 2\n}\n'
        )
        self.assertFalse(path.with_suffix(".json.tmp").exists())
        project._write_json(path, {"different": True})
        self.assertEqual(project._read_json_if_possible(path),
                         {"different": True})

    def test_remove_empty_directories_preserves_nonempty_ones(self) -> None:
        """Prune only empty nested folders, ignoring expected OS errors."""

        root = self._folder("cleanup")
        (root / "empty" / "deep").mkdir(parents=True)
        self._write(root, "keep/file.txt", "content")
        project._remove_empty_directories(root)
        self.assertFalse((root / "empty").exists())
        self.assertTrue((root / "keep").exists())
        with patch.object(Path, "rmdir", side_effect=OSError("busy")):
            project._remove_empty_directories(root)
        self.assertTrue((root / "keep").exists())

    def test_select_workspace_root_for_new_and_known_owners(self) -> None:
        """A saved owner never turns repeated creation into a reopen."""

        source = self._folder("owner")
        parent = self._folder("projects")
        first = project.prepare_project_workspace(source, "code", parent)
        second = project.prepare_project_workspace(source, "code", parent)
        self.assertEqual(first.root.name, "project_owner")
        self.assertEqual(second.root.name, "project_owner-2")
        self.assertNotEqual(first.root, second.root)

    def test_select_workspace_root_avoids_unowned_preferred_root(
        self,
    ) -> None:
        """Even an empty or unknown default destination is skipped."""

        source = self._folder("owner")
        parent = self._folder("projects")
        occupied = parent / "project_owner"
        occupied.mkdir()
        marker = occupied / "keep.txt"
        marker.write_text("owned", encoding="utf-8")
        created = project.prepare_project_workspace(source, "code", parent)
        self.assertNotEqual(created.root, occupied)
        self.assertEqual(created.root.name, "project_owner-2")
        self.assertEqual(marker.read_text(encoding="utf-8"), "owned")

    def test_select_workspace_root_reuses_matching_collision_owner(
        self,
    ) -> None:
        """A matching collision owner also receives an independent sibling."""

        source = self._folder("owner")
        parent = self._folder("projects")
        first = project.prepare_project_workspace(source, "code", parent)
        second = project.prepare_project_workspace(source, "code", parent)
        third = project.prepare_project_workspace(source, "code", parent)
        self.assertEqual(second.root.name, first.root.name + "-2")
        self.assertEqual(third.root.name, first.root.name + "-3")
        self.assertNotEqual(third.root, second.root)


if __name__ == "__main__":
    unittest.main()
