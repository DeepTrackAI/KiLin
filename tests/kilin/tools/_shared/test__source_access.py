"""Verify that tools can read only explicitly selected workspace sources.

Test both manifest authority and the filesystem boundary independently.
No selected-source filename should turn a symlink or external profile path
into permission to read unrelated local user data.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from kilin.tools._shared.source_access import (
    selected_paths,
    selected_source_file,
)


class TestSelectedSourceAccess(unittest.TestCase):
    """Test selected-file authorization and containment."""

    def test_manifest_is_authoritative_even_when_empty(self) -> None:
        indexed = SimpleNamespace(path=Path("paper.tex"))
        context = SimpleNamespace(
            chunks=[indexed], source_files=lambda: {},
        )
        self.assertEqual(selected_paths(context), frozenset())
        del context.source_files
        self.assertEqual(
            selected_paths(context), frozenset({Path("paper.tex")})
        )

    def test_only_selected_regular_workspace_files(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            source = root / "sources"
            source.mkdir(parents=True)
            paper = source / "paper.tex"
            paper.write_text("Text", encoding="utf-8")
            allowed = {Path("paper.tex")}
            self.assertEqual(
                selected_source_file(root, source, Path("paper.tex"),
                                     allowed), paper.resolve(),
            )
            for path in (
                Path("unselected.tex"), Path("../outside/secret.bib"),
                Path("/tmp/absolute.bib"), Path("paper.tex/../x"),
            ):
                with self.subTest(path=path):
                    self.assertIsNone(
                        selected_source_file(root, source, path, allowed)
                    )
            (root / "paper.tex").write_text("Wrong", encoding="utf-8")
            self.assertEqual(
                selected_source_file(root, source, Path("paper.tex"),
                                     allowed), paper.resolve(),
            )

    def test_external_saved_source_root_is_not_authorized(self) -> None:
        with TemporaryDirectory() as temporary:
            outer = Path(temporary)
            root = outer / "project"
            root.mkdir()
            source = outer / "elsewhere"
            source.mkdir()
            (source / "refs.bib").write_text("private", encoding="utf-8")
            self.assertIsNone(selected_source_file(
                root, source, Path("refs.bib"), {Path("refs.bib")}
            ))

    def test_symlink_and_linked_parent_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            outer = Path(temporary)
            root = outer / "project"
            source = root / "sources"
            source.mkdir(parents=True)
            private = outer / "private.bib"
            private.write_text("private", encoding="utf-8")
            link = source / "refs.bib"
            link.symlink_to(private)
            self.assertIsNone(selected_source_file(
                root, source, Path("refs.bib"), {Path("refs.bib")}
            ))
            subdir = outer / "linked"
            subdir.mkdir()
            (subdir / "refs.bib").write_text("private", encoding="utf-8")
            (source / "nested").symlink_to(subdir, target_is_directory=True)
            self.assertIsNone(selected_source_file(
                root, source, Path("nested/refs.bib"),
                {Path("nested/refs.bib")}
            ))
            self.assertIsNone(selected_source_file(
                root, source, Path("refs.bib"), set()
            ))


if __name__ == "__main__":
    unittest.main()
