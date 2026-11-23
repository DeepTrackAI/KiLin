"""Verify the single-source, tag-pinned README publication contract.

These tests use temporary checkout trees and never access GitHub or PyPI.
They also check that KíLín's committed PyPI description is reproducible.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from contextlib import redirect_stderr
import io
from io import StringIO
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from setup import generate_pypi_readme as readme
from setup.check_pypi_distribution import verify_distributions


ROOT = Path(__file__).resolve().parents[2]


class TestReadmeGeneration(unittest.TestCase):
    """Check the generator's supported links and safety boundaries.

    """

    def setUp(self):
        """Provide a checkout containing both files and directories.

        """

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "docs").mkdir()
        (self.root / "docs" / "guide.md").write_text("# Guide\n")
        (self.root / "assets").mkdir()
        (self.root / "assets" / "logo dark.svg").write_text("<svg/>\n")
        (self.root / "tutorials").mkdir()
        (self.root / "pyproject.toml").write_text(
            '[project]\nname = "kilin"\nversion = "0.1.0rc1"\n',
            encoding="utf-8",
        )

    def test_local_links_follow_the_exact_version_tag(self):
        """Distinguish source browsing from raw image references.

        """

        source = (
            '[guide](docs/guide.md#details)\n'
            '[tutorials](tutorials/)\n'
            '![brand](assets/logo%20dark.svg)\n'
            '<img src="assets/logo%20dark.svg">\n'
            '<source srcset="assets/logo%20dark.svg 2x">\n'
            '<a href="docs/guide.md#details">Guide</a>\n'
        )
        output = readme.render_pypi_readme(
            source, readme.read_project_version(self.root), self.root
        )
        self.assertIn(
            'https://github.com/DeepTrackAI/KiLin/blob/'
            'v0.1.0rc1/docs/guide.md#details',
            output,
        )
        self.assertIn(
            'https://github.com/DeepTrackAI/KiLin/tree/'
            'v0.1.0rc1/tutorials',
            output,
        )
        image = (
            'https://raw.githubusercontent.com/DeepTrackAI/KiLin/'
            'v0.1.0rc1/assets/logo%20dark.svg'
        )
        self.assertEqual(output.count(image), 3)
        self.assertIn('srcset="' + image + ' 2x"', output)
        self.assertIn('href="' + (
            'https://github.com/DeepTrackAI/KiLin/blob/'
            'v0.1.0rc1/docs/guide.md#details') + '"', output)

    def test_external_urls_fragments_and_code_are_unchanged(self):
        """Do not rewrite live GitHub pages, badges or code examples.

        """

        source = (
            '[release](https://github.com/DeepTrackAI/KiLin/releases)\n'
            '[same](#installation) [mail](mailto:team@example.org)\n'
            '![badge](https://img.shields.io/badge/test-green)\n'
            '`[literal](docs/guide.md)`\n'
            '```markdown\n[example](docs/guide.md)\n```\n'
            'Text [guide](docs/guide.md).\n'
        )
        output = readme.render_pypi_readme(
            source, "0.1.0b1", self.root
        )
        for unchanged in (
            'https://github.com/DeepTrackAI/KiLin/releases',
            '[same](#installation)',
            'mailto:team@example.org',
            'https://img.shields.io/badge/test-green',
            '`[literal](docs/guide.md)`',
            '```markdown\n[example](docs/guide.md)\n```',
        ):
            self.assertIn(unchanged, output)
        self.assertIn(
            '[guide](https://github.com/DeepTrackAI/KiLin/blob/'
            'v0.1.0b1/docs/guide.md)',
            output,
        )

    def test_missing_or_escaping_targets_fail_before_publication(self):
        """Do not silently create broken or escaping release links.

        """

        with self.assertRaisesRegex(ValueError, "Invalid local"):
            readme.render_pypi_readme(
                '[bad](docs/missing.md)', "0.1.0b1", self.root
            )
        with self.assertRaisesRegex(ValueError, "Invalid local"):
            readme.render_pypi_readme(
                '[bad](../outside.md)', "0.1.0b1", self.root
            )
        with self.assertRaisesRegex(ValueError, "must be relative"):
            readme.render_pypi_readme(
                '[bad](/docs/guide.md)', "0.1.0b1", self.root
            )
        with self.assertRaisesRegex(ValueError, "not a file"):
            readme.render_pypi_readme(
                '![bad](tutorials/)', "0.1.0b1", self.root
            )

    def test_check_mode_detects_stale_output_without_modifying_it(self):
        """A PR check must fail, not regenerate an unreviewed artifact.

        """

        (self.root / "README.md").write_text(
            '[guide](docs/guide.md)\n', encoding="utf-8"
        )
        target = self.root / "README.pypi.md"
        target.write_text("outdated\n", encoding="utf-8")
        with patch.object(readme, "_ROOT", self.root):
            with redirect_stderr(StringIO()) as errors:
                self.assertEqual(readme.main(["--check"]), 1)
            self.assertIn("stale", errors.getvalue())
            self.assertEqual(target.read_text(), "outdated\n")
            self.assertEqual(readme.main([]), 0)
            self.assertEqual(readme.main(["--check"]), 0)

    def test_project_version_is_validated(self):
        """The tag reference cannot contain an unsafe version value.

        """

        self.assertEqual(readme.read_project_version(self.root), "0.1.0rc1")
        (self.root / "pyproject.toml").write_text(
            '[project]\nversion = "bad/version"\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Invalid release"):
            readme.read_project_version(self.root)


class TestMaintainedReadme(unittest.TestCase):
    """Keep the public README and its generated release copy in sync.

    """

    def test_repository_readme_is_branch_relative(self):
        """Do not pin repository-owned documentation to main.

        """

        source = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("/DeepTrackAI/KiLin/blob/main/", source)
        self.assertNotIn("/DeepTrackAI/KiLin/tree/main/", source)
        self.assertNotIn(
            "raw.githubusercontent.com/DeepTrackAI/KiLin/main/",
            source,
        )

    def test_committed_pypi_readme_matches_generator(self):
        """Fail if source content or the package version changed alone.

        """

        version = readme.read_project_version(ROOT)
        source = (ROOT / "README.md").read_text(encoding="utf-8")
        expected = readme.render_pypi_readme(source, version, ROOT)
        actual = (ROOT / "README.pypi.md").read_text(
            encoding="utf-8"
        )
        self.assertEqual(actual, expected)
        self.assertIn(f"/v{version}/assets/brand/", actual)


class TestDistributionReadme(unittest.TestCase):
    """Inspect wheel and sdist bytes without network or build tools.

    """

    def setUp(self):
        """Build minimal metadata fixtures for one checked-out version.

        """

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dist = self.root / "dist"
        self.dist.mkdir()
        (self.root / "pyproject.toml").write_text(
            '[project]\nversion = "0.1.0b1"\n'
            'readme = "README.pypi.md"\n', encoding="utf-8"
        )
        (self.root / "README.md").write_text(
            "[guide](docs/guide.md)\n", encoding="utf-8"
        )
        self.description = (
            "KíLín — your own assistant\n"
            "[guide](https://github.com/DeepTrackAI/KiLin/"
            "blob/v0.1.0b1/docs/guide.md)\n"
        )
        (self.root / "README.pypi.md").write_text(
            self.description, encoding="utf-8"
        )
        self.wheel = self.dist / "kilin-0.1.0b1-py3-none-any.whl"
        self.sdist = self.dist / "kilin-0.1.0b1.tar.gz"
        self.write_wheel(self.description)
        self.write_sdist()

    def write_wheel(self, description: str) -> None:
        """Write a wheel fixture with the supplied description.

        """

        metadata = (
            "Metadata-Version: 2.4\n"
            "Name: kilin\n"
            "Version: 0.1.0b1\n"
            "Description-Content-Type: text/markdown\n\n"
        ).encode("utf-8") + description.encode("utf-8")
        with zipfile.ZipFile(self.wheel, mode="w") as archive:
            archive.writestr("kilin-0.1.0b1.dist-info/METADATA", metadata)

    def write_sdist(self, *, include_pypi: bool = True) -> None:
        """Write a minimal source archive with both root READMEs.

        """

        with tarfile.open(self.sdist, mode="w:gz") as archive:
            names = ["README.md"]
            if include_pypi:
                names.append("README.pypi.md")
            for name in names:
                data = (self.root / name).read_bytes()
                item = tarfile.TarInfo(f"kilin-0.1.0b1/{name}")
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))

    def test_matching_metadata_is_accepted(self):
        """The distribution verifier accepts matching package files.

        """

        verify_distributions(self.root, self.dist)

    def test_stale_wheel_description_is_rejected(self):
        """A wheel cannot quietly contain an earlier description.

        """

        self.write_wheel("stale\n")
        with self.assertRaisesRegex(AssertionError, "differs"):
            verify_distributions(self.root, self.dist)

    def test_missing_sdist_readme_is_rejected(self):
        """An sdist must rebuild with the same generated description.

        """

        self.write_sdist(include_pypi=False)
        with self.assertRaisesRegex(AssertionError, "root README.pypi.md"):
            verify_distributions(self.root, self.dist)


if __name__ == "__main__":
    unittest.main()
