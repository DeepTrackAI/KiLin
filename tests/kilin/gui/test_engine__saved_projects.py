"""Verify GUI startup and discovery of saved KíLín workspaces.

No model backend or Qt application is needed for these checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kilin.gui.engine import ChatEngine


class TestEngineSavedProjects(unittest.TestCase):
    """Test startup and explicit saved-workspace selection."""

    def test_startup_finds_recent_saved_project(self) -> None:
        """Only saved workspaces are candidates for automatic reopening."""

        with TemporaryDirectory() as temporary:
            sandbox = Path(temporary).resolve()
            for index, (name, kind) in enumerate(
                (("older", "code"), ("newer", "literature"))
            ):
                folder = sandbox / name
                (folder / "sources").mkdir(parents=True)
                profile = folder / "profile.ini"
                profile.write_text(
                    f"[project]\nkind = {kind}\nsource = sources\n",
                    encoding="utf-8",
                )
                os.utime(profile, (100 + index, 100 + index))
            (sandbox / "unrelated").mkdir()
            events: list[dict[str, object]] = []
            engine = ChatEngine(sandbox, events.append)

            engine.startup()

            self.assertEqual(events[-1]["event"], "startup")
            self.assertEqual(events[-1]["project"], str(sandbox / "newer"))
            self.assertEqual(events[-1]["kind"], "literature")

    def test_saved_project_tracks_type_and_deleted_folder(self) -> None:
        """Resume follows the selected type and a removed folder disappears."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            sandbox = root / "sandbox"
            for name, kind in (
                ("code-project", "code"),
                ("papers", "literature"),
            ):
                folder = sandbox / name
                (folder / "sources").mkdir(parents=True)
                (folder / "profile.ini").write_text(
                    f"[project]\nkind = {kind}\nsource = sources\n",
                    encoding="utf-8",
                )
            events: list[dict[str, object]] = []
            engine = ChatEngine(sandbox, events.append)

            engine.find_saved_project("code", "")
            self.assertEqual(
                events[-1]["project"], str(sandbox / "code-project")
            )
            engine.find_saved_project("literature", "")
            self.assertEqual(events[-1]["project"], str(sandbox / "papers"))

            (sandbox / "papers").rename(root / "removed-papers")
            engine.find_saved_project("literature", "")
            self.assertIsNone(events[-1]["project"])

    def test_saved_projects_are_newest_first(self) -> None:
        """The project-list refresh shares startup's discovery rules."""

        with TemporaryDirectory() as temporary:
            sandbox = Path(temporary).resolve()
            for index, (name, kind) in enumerate(
                (("older", "code"), ("newer", "literature"))
            ):
                folder = sandbox / name
                (folder / "sources").mkdir(parents=True)
                profile = folder / "profile.ini"
                profile.write_text(
                    f"[project]\nkind = {kind}\nsource = sources\n",
                    encoding="utf-8",
                )
                os.utime(profile, (100 + index, 100 + index))
            events: list[dict[str, object]] = []
            engine = ChatEngine(sandbox, events.append)

            engine.list_saved_projects("")

            self.assertEqual(events[-1]["event"], "projects")
            self.assertEqual(
                [project["name"] for project in events[-1]["projects"]],
                ["newer", "older"],
            )

    def test_saved_project_checks_custom_workspace(self) -> None:
        """A chosen workspace takes precedence over sandbox suggestions."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            custom = root / "my-project"
            (custom / "sources").mkdir(parents=True)
            (custom / "profile.ini").write_text(
                "[project]\nkind = literature\nsource = sources\n",
                encoding="utf-8",
            )
            events: list[dict[str, object]] = []
            engine = ChatEngine(root / "sandbox", events.append)

            engine.find_saved_project("literature", str(custom))
            self.assertEqual(events[-1]["project"], str(custom))
            engine.find_saved_project("code", str(custom))
            self.assertIsNone(events[-1]["project"])

            custom.rename(root / "moved-project")
            engine.find_saved_project("literature", str(custom))
            self.assertIsNone(events[-1]["project"])

    def test_empty_sandbox_starts_without_workspace(self) -> None:
        """An empty sandbox does not show an empty workspace sidebar."""

        with TemporaryDirectory() as temporary:
            events: list[dict[str, object]] = []
            ChatEngine(Path(temporary).resolve(), events.append).startup()

        self.assertEqual(events[-1], {"event": "startup", "project": None})
