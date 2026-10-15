"""Test the immutable workspace resource declaration contract.

Check construction, field preservation, dataclass equality and hashing,
and the absence of filesystem side effects. Destination validation belongs
to the project layout planner, not this declaration module.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.workspace_resources import WorkspaceResource


class TestWorkspaceResource(unittest.TestCase):
    """Verify the lightweight, immutable tool resource declaration."""

    def test_fields_preserve_source_and_destination(self) -> None:
        """Construct a resource without changing either field."""

        source = Path("package/templates/example.md")
        destination = "templates/example.md"
        resource = WorkspaceResource(source, destination)
        self.assertIs(resource.source, source)
        self.assertEqual(resource.destination, destination)
        self.assertEqual(
            tuple(field.name for field in fields(resource)),
            ("source", "destination"),
        )

    def test_resources_have_value_equality_and_hashing(self) -> None:
        """Equal declarations deduplicate in sets and dictionaries."""

        first = WorkspaceResource(Path("a.md"), "templates/a.md")
        same = WorkspaceResource(Path("a.md"), "templates/a.md")
        different = WorkspaceResource(Path("a.md"), "templates/b.md")
        self.assertEqual(first, same)
        self.assertEqual(hash(first), hash(same))
        self.assertNotEqual(first, different)
        self.assertEqual(len({first, same, different}), 2)
        self.assertNotEqual(first, object())

    def test_declarations_are_frozen_and_slotted(self) -> None:
        """Tool declarations cannot be mutated or given new fields."""

        resource = WorkspaceResource(Path("a.md"), "templates/a.md")
        with self.assertRaises(FrozenInstanceError):
            resource.destination = "templates/b.md"
        with self.assertRaises((AttributeError, TypeError)):
            resource.extra = "not supported"
        self.assertFalse(hasattr(resource, "__dict__"))

    def test_constructing_resource_has_no_filesystem_side_effects(
        self,
    ) -> None:
        """Declaring a missing resource never creates directories."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "missing" / "example.md"
            destination = "templates/example.md"
            resource = WorkspaceResource(source, destination)
            self.assertEqual(resource.source, source)
            self.assertEqual(resource.destination, destination)
            self.assertFalse(source.exists())
            self.assertFalse((root / "templates").exists())

    def test_declaration_does_not_validate_destination(self) -> None:
        """The core layout planner owns destination validation."""

        resource = WorkspaceResource(Path("a.md"), "../outside.md")
        self.assertEqual(resource.destination, "../outside.md")


if __name__ == "__main__":
    unittest.main()
