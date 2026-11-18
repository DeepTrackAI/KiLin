"""Test GUI-independent project folders and tool-selected resources.

Exercise each core project creation path with temporary directories and
validated project-type configuration. Confirm that opening existing projects
cannot change user-owned templates and that invalid tool declarations fail
before installation.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kilin.project import (
    initialize_project,
    open_project,
    prepare_project_workspace,
    synchronize_project_workspace,
)
from kilin.project_layout import (
    ProjectLayout,
    ensure_project_folders,
    install_project_resources,
    plan_project_layout,
    source_scan_exclusions,
)
from kilin.tools.generate_paper_summary import (
    SummarySettings,
    choose_template,
    workspace_resources,
    SHIPPED_TEMPLATES_DIR,
)
from kilin.tools._registry import TOOL_REGISTRY, ToolUse
from kilin.tools.project_summary import (
    SHIPPED_TEMPLATES_DIR as PROJECT_TEMPLATES_DIR,
)
from kilin.workspace import prepare_sources_workspace
from kilin.repository import scan_repository
from kilin.workspace_resources import (
    WorkspaceResource,
    templates_folder,
)


class TestProjectLayout(unittest.TestCase):
    """Verify project layouts, safety checks and reopen semantics."""

    def setUp(self) -> None:
        """Create an isolated workspace parent.

        """

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _project(self, kind: str) -> Path:
        """Initialize one project through the core API.

        Parameters
        ----------
        kind: str
            Built-in project type identifier.

        Returns
        -------
        pathlib.Path
            Initialized project root.

        """

        root = self.root / kind
        root.mkdir()
        initialize_project(root, kind)
        return root

    def _fake_plan(
        self, *resources: WorkspaceResource
    ) -> ProjectLayout:
        """Plan resources for a synthetic configured intent.

        Parameters
        ----------
        *resources: WorkspaceResource
            Tool-provided declarations to validate.

        Returns
        -------
        ProjectLayout
            Planned and deduplicated resources.

        """

        tool = SimpleNamespace(workspace_resources=lambda _: resources)
        definition = SimpleNamespace(
            intents={
                "example": SimpleNamespace(
                    tool=ToolUse("generate_paper_summary", object())
                )
            }
        )
        with patch(
            "kilin.project_layout.project_type", return_value=definition
        ):
            with patch.dict(
                TOOL_REGISTRY, {"generate_paper_summary": tool}
            ):
                return plan_project_layout("synthetic")

    def test_code_project_has_empty_templates_directory(self) -> None:
        """All core directories exist even without configured templates."""

        root = self._project("code")
        for name in ("indexes", "logs", "memory", "outputs",
                     "templates"):
            with self.subTest(directory=name):
                self.assertTrue((root / name).is_dir())
        self.assertEqual(list((root / "templates").iterdir()), [])
        self.assertEqual(plan_project_layout("code").resources, ())

    def test_literature_installs_exactly_its_four_templates(self) -> None:
        """Install the literature templates and the guide to the format."""

        root = self._project("literature")
        expected = (
            "README.md",
            "experimental_protocol.md",
            "literature_table.md",
            "popular_science_summary.md",
            "project_overview.md",
            "research_briefing.md",
            "technical_summary.md",
        )
        actual = tuple(
            sorted(path.name for path in (root / "templates").iterdir())
        )
        self.assertEqual(actual, expected)
        self.assertEqual(len(plan_project_layout("literature").resources), 7)
        for name in expected:
            shipped = SHIPPED_TEMPLATES_DIR / name
            if not shipped.exists():  # A project report template.
                shipped = PROJECT_TEMPLATES_DIR / name
            self.assertEqual(
                (root / "templates" / name).read_bytes(),
                shipped.read_bytes(),
            )

    def test_reopening_preserves_edited_deleted_and_added_templates(
        self,
    ) -> None:
        """Opening never silently restores deleted packaged resources."""

        root = self._project("literature")
        edited = root / "templates/research_briefing.md"
        deleted = root / "templates/technical_summary.md"
        custom = root / "templates/lab_summary.md"
        edited.write_text("# Edited by the user\n", encoding="utf-8")
        deleted.unlink()
        custom.write_text("# User template\n", encoding="utf-8")
        open_project(root)
        with self.assertRaises(ValueError):
            initialize_project(root, "literature")
        self.assertEqual(
            edited.read_text(encoding="utf-8"), "# Edited by the user\n"
        )
        self.assertFalse(deleted.exists())
        self.assertEqual(
            custom.read_text(encoding="utf-8"), "# User template\n"
        )
        with self.assertRaises(ValueError):
            choose_template(
                "write a report", root / "templates", "technical_summary"
            )

    def test_existing_profile_without_templates_gets_empty_folder(
        self,
    ) -> None:
        """Legacy profiles get folders but never automatic templates."""

        root = self._project("literature")
        for path in (root / "templates").iterdir():
            path.unlink()
        (root / "templates").rmdir()
        self.assertEqual(open_project(root).root.resolve(), root.resolve())
        self.assertFalse((root / "templates").exists())
        ensure_project_folders(root)
        self.assertEqual(list((root / "templates").iterdir()), [])

    def test_multi_source_path_installs_selected_templates(self) -> None:
        """The multi-source Python API creates the same core layout."""

        original = self.root / "paper.md"
        original.write_text("# Paper\n", encoding="utf-8")
        workspace = self.root / "multiple"
        project = prepare_sources_workspace(
            [original], "literature", workspace
        )
        self.assertTrue((project.root / "sources/paper.md").is_file())
        self.assertTrue((project.root / "outputs").is_dir())
        self.assertEqual(
            len(list((project.root / "templates").glob("*.md"))), 7
        )

    def test_mirrored_workspace_installs_selected_templates(self) -> None:
        """The mirrored-source API installs resources without the GUI."""

        original = self.root / "papers"
        original.mkdir()
        (original / "paper.md").write_text("# Study", encoding="utf-8")
        sandbox = self.root / "sandbox"
        workspace = prepare_project_workspace(original, "literature", sandbox)
        self.assertTrue((workspace.root / "outputs").is_dir())
        self.assertTrue((workspace.root / "templates").is_dir())
        self.assertEqual(
            len(list((workspace.root / "templates").glob("*.md"))), 7
        )
        # Repeated synchronization cannot replace an edited template.
        modified = workspace.root / "templates/research_briefing.md"
        modified.write_text("# Custom\n", encoding="utf-8")
        synchronize_project_workspace(original, workspace.root)
        self.assertEqual(modified.read_text(encoding="utf-8"), "# Custom\n")

    def test_user_provided_project_type_template_needs_no_package(
        self,
    ) -> None:
        """A custom template can be supplied after creating a project."""

        settings = SummarySettings(template="custom_lab_format")
        self.assertEqual(workspace_resources(settings), ())
        root = self._project("code")
        custom = root / "templates/custom_lab_format.md"
        custom.write_text("# Lab report\n", encoding="utf-8")
        self.assertTrue(custom.is_file())

    def test_workspace_inventory_reports_empty_folders(self) -> None:
        """The core reports empty folders without importing the GUI."""

        from kilin.project_files import list_project_files

        root = self._project("code")
        listed = list_project_files(open_project(root))
        for name in ("indexes", "logs", "memory", "outputs",
                     "templates"):
            self.assertIn(f"{name}/", listed)
            self.assertNotIn(name, listed)

    def test_root_source_excludes_metadata_not_external_files(self) -> None:
        """Internal profile and manifests are excluded only at project root."""

        from kilin.project import configure_project

        root = self._project("code")
        profile = root / "profile.ini"
        profile.write_text(
            profile.read_text(encoding="utf-8").replace(
                "source = sources", "source = ."
            ), encoding="utf-8",
        )
        project = configure_project(
            root, {"extensions": [".py", ".ini", ".json"]}
        )
        (root / "analysis.py").write_text("x = 1\n", encoding="utf-8")
        (root / "sources.json").write_text("{}\n", encoding="utf-8")
        (root / "source_snapshot.json").write_text("{}\n", encoding="utf-8")
        report = scan_repository(
            project.source,
            extensions=project.profile.extensions,
            exclude_paths=source_scan_exclusions(
                project.root, project.source
            ),
        )
        self.assertEqual(report["selected_files"], ["analysis.py"])

        external = self.root / "external"
        external.mkdir()
        (external / "profile.ini").write_text("[external]\n", encoding="utf-8")
        (external / "sources.json").write_text("{}\n", encoding="utf-8")
        external_report = scan_repository(
            external, extensions=[".ini", ".json"]
        )
        self.assertEqual(
            external_report["selected_files"],
            ["profile.ini", "sources.json"],
        )


    def test_managed_source_versions_are_not_automatically_scanned(
        self,
    ) -> None:
        """Only recorded historical copies enter the retrieval scan."""

        source = self.root / "external"
        source.mkdir()
        (source / "a.py").write_text("value = 1\n", encoding="utf-8")
        project = prepare_sources_workspace(
            [source], "code", self.root / "managed"
        )
        self.assertEqual(
            source_scan_exclusions(project.root, project.source),
            (".kilin_versions",),
        )
        folder = project.source / ".kilin_versions" / "untracked"
        folder.mkdir(parents=True)
        (folder / "untracked.py").write_text(
            "untracked = True\n", encoding="utf-8"
        )
        report = scan_repository(
            project.source,
            extensions=project.profile.extensions,
            exclude_paths=source_scan_exclusions(
                project.root, project.source
            ),
        )
        self.assertNotIn(
            ".kilin_versions/untracked/untracked.py",
            report["selected_files"],
        )

    def test_root_source_excludes_managed_templates_from_scans(self) -> None:
        """Template Markdown is not indexed when source equals project root."""

        root = self._project("literature")
        profile = root / "profile.ini"
        text = profile.read_text(encoding="utf-8")
        text = text.replace("source = sources", "source = .")
        profile.write_text(text, encoding="utf-8")
        opened = open_project(root)
        self.assertEqual(opened.source.resolve(), root.resolve())
        (root / "study.md").write_text("# Source", encoding="utf-8")
        (root / "memory/feedback.json").write_text(
            '{"records": "not a source"}', encoding="utf-8"
        )
        excluded = source_scan_exclusions(opened.root, opened.source)
        report = scan_repository(
            opened.source,
            extensions=(".md",),
            exclude_paths=excluded,
        )
        self.assertEqual(report["selected_files"], ["study.md"])
        self.assertIn("memory", excluded)
        # An unrelated source directory named target is not managed.
        target = root / "target"
        target.mkdir()
        (target / "notes.md").write_text("# Notes", encoding="utf-8")
        report = scan_repository(
            opened.source, extensions=(".md",), exclude_paths=excluded,
        )
        self.assertIn("target/notes.md", report["selected_files"])
        self.assertEqual(
            source_scan_exclusions(root, root / "external"), ()
        )


    def test_resolved_root_alias_still_excludes_managed_folders(
        self,
    ) -> None:
        """Resolve filesystem aliases when applying source scan limits.

        """

        root = self._project("literature")
        alias = self.root / "project_link"
        try:
            alias.symlink_to(root, target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("Symbolic links are unavailable")
        exclusions = source_scan_exclusions(alias, root.resolve())
        self.assertIn("templates", exclusions)
        self.assertIn("outputs", exclusions)
        self.assertEqual(
            source_scan_exclusions(alias, self.root / "external"), ()
        )

    def test_template_accessor_has_no_side_effects(self) -> None:
        """A tool helper cannot create folders or install templates."""

        root = self.root / "uninitialized"
        self.assertEqual(templates_folder(root), root / "templates")
        self.assertFalse(root.exists())

    def test_manual_template_survives_initial_installation(self) -> None:
        """Preexisting project-owned content is never overwritten."""

        root = self.root / "preexisting"
        root.mkdir()
        (root / "templates").mkdir()
        template = root / "templates/research_briefing.md"
        template.write_text("# Custom\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            initialize_project(root, "literature")
        self.assertEqual(template.read_text(encoding="utf-8"), "# Custom\n")
        self.assertEqual(
            len(list((root / "templates").glob("*.md"))), 1
        )
        self.assertFalse((root / "profile.ini").exists())

    def test_explicit_repair_is_opt_in(self) -> None:
        """A separate core operation can restore selected resources."""

        root = self._project("literature")
        missing = root / "templates/technical_summary.md"
        missing.unlink()
        open_project(root)
        self.assertFalse(missing.exists())
        install_project_resources(root, plan_project_layout("literature"))
        self.assertEqual(
            missing.read_bytes(),
            (SHIPPED_TEMPLATES_DIR / missing.name).read_bytes(),
        )

    def test_duplicate_resource_declarations_are_deduplicated(self) -> None:
        """Multiple intents sharing a template request only one copy."""

        source = SHIPPED_TEMPLATES_DIR / "research_briefing.md"
        resource = WorkspaceResource(source, "templates/shared.md")
        planned = self._fake_plan(resource, resource)
        self.assertEqual(len(planned.resources), 1)

    def test_conflicting_resources_are_rejected(self) -> None:
        """Two source files cannot claim the same workspace path."""

        sources = sorted(SHIPPED_TEMPLATES_DIR.glob("*.md"))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self._fake_plan(
                WorkspaceResource(sources[0], "templates/report.md"),
                WorkspaceResource(sources[1], "templates/report.md"),
            )

    def test_generic_tool_resource_subdirectory_can_be_installed(
        self,
    ) -> None:
        """Resources need not be specific to the report-template tool."""

        root = self._project("code")
        source = SHIPPED_TEMPLATES_DIR / "research_briefing.md"
        resource = WorkspaceResource(source, "styles/reports/format.md")
        layout = self._fake_plan(resource)
        install_project_resources(root, layout)
        destination = root / "styles/reports/format.md"
        self.assertEqual(destination.read_bytes(), source.read_bytes())

    def test_invalid_destinations_are_rejected(self) -> None:
        """Tool-declared file paths cannot escape a managed workspace."""

        source = SHIPPED_TEMPLATES_DIR / "research_briefing.md"
        invalid = (
            "../escape.md", "/tmp/out.md", "C:/elsewhere.md",
            "templates/../escape.md", "templates//escape.md",
            "templates/./escape.md", "templates\\escape.md",
            "outputs/output.md", "indexes/index.md", "sources/file.md",
            "profile.ini", "templates/", "templates",
        )
        for destination in invalid:
            with self.subTest(destination=destination):
                with self.assertRaises(ValueError):
                    self._fake_plan(WorkspaceResource(source, destination))

    def test_failed_resource_plan_does_not_create_profile(self) -> None:
        """New profile creation validates tool assets before writing."""

        root = self.root / "invalid"
        root.mkdir()
        with patch(
            "kilin.project.plan_project_layout",
            side_effect=ValueError("Missing packaged template"),
        ):
            with self.assertRaisesRegex(ValueError, "Missing packaged"):
                initialize_project(root, "literature")
        self.assertFalse((root / "profile.ini").exists())

    def test_tool_cannot_install_another_packages_resource(self) -> None:
        """Enforce ownership of declared files in the registry package.

        """

        foreign = self.root / "outside.md"
        foreign.write_text("# Outside\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "outside its package"):
            self._fake_plan(
                WorkspaceResource(foreign, "templates/foreign.md")
            )
        other_tool = (
            Path(__file__).resolve().parents[2]
            / "kilin/tools/batch_classify/defaults.json"
        )
        with self.assertRaisesRegex(ValueError, "outside its package"):
            self._fake_plan(
                WorkspaceResource(other_tool, "templates/config.json")
            )

    def test_missing_packaged_resource_fails_during_planning(self) -> None:
        """Fail before writing a profile for unavailable tool resources."""

        root = self.root / "project"
        root.mkdir()
        missing = self.root / "not_packaged.md"
        with self.assertRaisesRegex(ValueError, "packaged resource"):
            self._fake_plan(
                WorkspaceResource(missing, "templates/missing.md")
            )
        self.assertFalse((root / "profile.ini").exists())

    def test_symlinked_project_resource_is_not_overwritten(self) -> None:
        """Resource installation refuses links pointing outside the root."""

        root = self._project("code")
        outside = self.root / "outside.md"
        outside.write_text("unchanged", encoding="utf-8")
        link = root / "templates" / "unsafe.md"
        try:
            link.symlink_to(outside)
        except (NotImplementedError, OSError):
            self.skipTest("Symbolic links are unavailable")
        layout = ProjectLayout((WorkspaceResource(
            SHIPPED_TEMPLATES_DIR / "research_briefing.md",
            "templates/unsafe.md",
        ),))
        with self.assertRaisesRegex(ValueError, "link"):
            install_project_resources(root, layout)
        self.assertEqual(outside.read_text(encoding="utf-8"), "unchanged")

    def test_linked_core_folder_is_rejected(self) -> None:
        """Core folders never follow symbolic links on reopening."""

        root = self._project("code")
        path = root / "templates"
        path.rmdir()
        try:
            path.symlink_to(self.root, target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("Symbolic links are unavailable")
        with self.assertRaisesRegex(ValueError, "link"):
            open_project(root)

    def test_existing_target_is_not_migrated_or_cleaned(self) -> None:
        """Core initialization leaves unrelated folders and guides intact.

        """

        root = self._project("code")
        (root / "outputs").rmdir()
        unrelated = root / "target"
        unrelated.mkdir()
        (unrelated / "user.md").write_text("# User\n", encoding="utf-8")
        guides = unrelated / "templates"
        guides.mkdir()
        generated = guides / "old_template.md"
        generated.write_text(
            "<!-- Generated by KíLín from a report template -->\n",
            encoding="utf-8",
        )
        ensure_project_folders(root)
        self.assertTrue((root / "outputs").is_dir())
        self.assertEqual(
            (unrelated / "user.md").read_text(encoding="utf-8"), "# User\n"
        )
        self.assertTrue(generated.is_file())
        self.assertFalse((root / "outputs/user.md").exists())


if __name__ == "__main__":
    unittest.main()
