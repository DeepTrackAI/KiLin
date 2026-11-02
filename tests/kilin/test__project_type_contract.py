"""Verify project-type identifiers across configuration and core boundaries.

Exercise clean prerelease schemas, relative source conventions, and an
isolated distribution containing no standard project-type definitions.
Models are never loaded and the package under test is not edited.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import kilin
from kilin import project
from kilin.project_type import (
    PROJECT_TYPES_DIR,
    ProjectType,
    load_project_types,
    project_type,
)
from tests.kilin.test_project_types import _type_data
from tests.kilin.test_rag import _session


class TestProjectTypeContract(unittest.TestCase):
    """Keep identifiers, profile fields, and metadata unambiguous."""

    def test_public_names_and_dataclass_fields(self) -> None:
        """Expose canonical identifiers without old Python API aliases."""

        for owner in (kilin, project):
            self.assertIs(owner.ProjectTypeId, str)
            for name in (
                "ProjectTypeId", "PROJECT_TYPE_ID_CODE",
                "PROJECT_TYPE_ID_LITERATURE", "PROJECT_TYPE_IDS",
            ):
                self.assertIn(name, owner.__all__)
            for old in (
                "ProjectKind", "PROJECT_KIND_CODE", "PROJECT_KIND_CHOICES",
                "PROJECT_KIND_LITERATURE",
            ):
                self.assertFalse(hasattr(owner, old))
                self.assertNotIn(old, owner.__all__)
        self.assertEqual(kilin.PROJECT_TYPE_IDS, tuple(load_project_types()))
        self.assertIn("id", {field.name for field in fields(ProjectType)})
        self.assertNotIn("kind", {field.name for field in fields(ProjectType)})
        profile = project.default_project_profile(kilin.PROJECT_TYPE_IDS[0])
        self.assertIn("type_id", asdict(profile))
        self.assertNotIn("kind", asdict(profile))
        self.assertFalse(hasattr(profile, "kind"))
        self.assertEqual(project_type(profile.type_id).id, profile.type_id)
        self.assertFalse(hasattr(project_type(profile.type_id), "kind"))

    def test_profile_and_metadata_use_the_new_identifiers(self) -> None:
        """Write new schemas and reject a profile with only the old key."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            created = kilin.prepare_sources_workspace([], "code", root)
            path = root / "profile.ini"
            text = path.read_text(encoding="utf-8")
            self.assertIn("type_id = code", text)
            self.assertNotIn("kind =", text)
            self.assertEqual(kilin.open_project(root), created)
            path.write_text(text.replace("type_id =", "kind ="),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "type_id"):
                kilin.open_project(root)
        session = _session()
        for record in (session._index_metadata(), session._parameters()):
            self.assertEqual(record["project_type_id"], "code")
            self.assertNotIn("project_kind", record)
            self.assertNotIn("project_type", record)

    def test_project_types_do_not_name_a_source_folder(self) -> None:
        """Types load without source_directory; new projects use sources/."""

        for type_id in load_project_types():
            with self.subTest(type_id=type_id):
                path = PROJECT_TYPES_DIR / f"{type_id}.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertNotIn("source_directory", data)
                self.assertFalse(
                    hasattr(project_type(type_id), "source_directory")
                )
        # The folder name is no longer a project-type setting.
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            data = _type_data()
            data["source_directory"] = "material"
            (folder / "samples.json").write_text(
                json.dumps(data), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "source_directory"):
                load_project_types(folder)

    def test_isolated_distribution_with_only_custom_types(self) -> None:
        """Import and use core and GUI with neither standard type present."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "kilin"
            shutil.copytree(
                Path(kilin.__file__).parent, package,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            shutil.copy2(
                Path(kilin.__file__).parent.parent / "pyproject.toml",
                root / "pyproject.toml",
            )
            directory = package / "project_types"
            for path in directory.glob("*.json"):
                path.unlink()
            for type_id in ("zeta", "alpha"):
                data = _type_data()
                (directory / f"{type_id}.json").write_text(
                    json.dumps(data), encoding="utf-8"
                )
            script = '''
import json
from pathlib import Path
import kilin
from kilin.gui.engine import ChatEngine
from kilin.project import configure_project
from kilin.rag import RAGSession
from kilin.runtime import BACKEND_TORCH

assert kilin.PROJECT_TYPE_IDS == ("alpha", "zeta")
assert kilin.PROJECT_TYPE_ID_CODE == "code"
for absent in ("code", "literature", "missing"):
    try:
        kilin.project_type(absent)
    except ValueError:
        pass
    else:
        raise AssertionError(absent)
root = Path("work").resolve()
root.mkdir()
adopted = root / "adopted"
adopted.mkdir()
first = kilin.initialize_project(adopted, "alpha")
assert first.source == (adopted / "sources").resolve()
assert kilin.open_project(adopted).source == first.source
preferred = root / "preferred"
preferred.mkdir()
assert kilin.initialize_project(preferred, "zeta").source == (
    preferred / "sources"
).resolve()
plain = root / "plain"
plain.mkdir()
assert kilin.initialize_project(plain, "alpha").source == plain / "sources"
empty = kilin.prepare_sources_workspace([], "alpha", root / "empty")
assert empty.profile.type_id == "alpha"
assert empty.source.name == "sources"
assert not list(empty.source.iterdir())
assert not list(empty.indexes.iterdir())
assert empty.profile.extensions == (".txt", ".md")
assert empty.profile.max_file_bytes == 4096
empty = configure_project(empty.root, {"max_file_bytes": 123})
file = root / "later.txt"
file.write_text("Later source.")
updated = kilin.add_project_sources(empty.root, [file])
assert updated.profile.max_file_bytes == 123
assert (updated.source / "later.txt").read_text() == "Later source."
coordinator = object.__new__(RAGSession)
coordinator.project = updated
coordinator.backend = BACKEND_TORCH
assert coordinator._index_metadata()["project_type_id"] == "alpha"
assert coordinator._system_prompt() == (
    kilin.project_type("alpha").system_prompt
)

events = []
engine = ChatEngine(root, events.append)
engine.list_saved_projects("")
projects = events[-1]["projects"]
assert {item["type_id"] for item in projects} == {"alpha", "zeta"}
assert all("kind" not in item for item in projects)
engine.settings("", "")
settings = events[-1]
assert settings["event"] == "settings", settings
assert settings["profile"]["type_id"] == "alpha"
assert [item["type_id"] for item in settings["project_types"]] == [
    "alpha", "zeta"
]
assert all("kind" not in item for item in settings["project_types"])
engine.close()
print(json.dumps({"type_ids": kilin.PROJECT_TYPE_IDS, "ok": True}))
'''
            result = subprocess.run(
                [sys.executable, "-c", script], cwd=root,
                env={**os.environ, "PYTHONPATH": str(root)},
                capture_output=True, text=True, timeout=60, check=False,
            )
            self.assertEqual(result.returncode, 0,
                             result.stdout + result.stderr)
            self.assertTrue(json.loads(result.stdout.splitlines()[-1])["ok"])
