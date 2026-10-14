"""Test package ownership and selective installation of tool assets.

Each project type selects registered tools and their resources. The core
must install only the declared files, without depending on the GUI or
copying resources from unrelated executable tool packages.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.project import initialize_project
from qachat.project_layout import plan_project_layout
from qachat.project_type import project_type
from qachat.tools._registry import TOOL_REGISTRY
from qachat.tools.generate_paper_summary import SHIPPED_TEMPLATES_DIR


class TestToolResources(unittest.TestCase):
    """Verify project-type resource selection and packaged ownership."""

    def test_project_types_only_declare_assets_their_tools_own(self) -> None:
        """Require every declared file to reside in its tool package.

        """

        for kind in ("code", "literature"):
            definition = project_type(kind)
            for intent in definition.intents.values():
                use = intent.tool
                if use is None:
                    continue
                declaration = TOOL_REGISTRY[use.name].workspace_resources
                if declaration is None:
                    continue
                package = Path(files(f"qachat.tools.{use.name}")).resolve()
                for asset in declaration(use.settings):
                    with self.subTest(type=kind, tool=use.name):
                        self.assertTrue(asset.source.is_file())
                        self.assertTrue(
                            asset.source.resolve().is_relative_to(package)
                        )
                        self.assertFalse(asset.source.is_symlink())

    def test_core_installs_exactly_the_planned_resource_files(self) -> None:
        """Create both project types using their core resource plans.

        """

        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            for kind in ("code", "literature"):
                with self.subTest(project_type=kind):
                    plan = plan_project_layout(kind)
                    root = parent / kind
                    root.mkdir()
                    initialize_project(root, kind)
                    self.assertTrue((root / "templates").is_dir())
                    installed = {
                        path.relative_to(root).as_posix()
                        for path in (root / "templates").rglob("*.md")
                    }
                    selected = {
                        resource.destination
                        for resource in plan.resources
                    }
                    self.assertEqual(installed, selected)
                    for resource in plan.resources:
                        self.assertEqual(
                            (root / resource.destination).read_bytes(),
                            resource.source.read_bytes(),
                        )

    def test_bundled_report_formats_are_tool_local(self) -> None:
        """Find report templates in their tool package, not in shared code.

        """

        owned = {
            resource.source
            for resource in plan_project_layout("literature").resources
        }
        self.assertEqual(
            owned,
            set(SHIPPED_TEMPLATES_DIR.glob("*.md")),
        )
        self.assertEqual(plan_project_layout("code").resources, ())


if __name__ == "__main__":
    unittest.main()
