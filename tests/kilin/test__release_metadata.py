"""Verify the version, authors, licence, and citation metadata.

The tests compare the source manifest, public version, lockfile, and
structured CFF citation. Wheel installation tests separately verify the
installed distribution metadata. This keeps source tests independent of
an unrelated or stale KíLín installation in the current environment.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

import yaml


_ROOT = Path(__file__).resolve().parents[2]
_EXPECTED_AUTHORS = (
    "Io̍k-uí Tiunn",
    "Joana B. Pereira",
    "Carlo Manzo",
    "Giovanni Volpe",
)


class TestReleaseMetadata(unittest.TestCase):
    """Keep the package and citation metadata mutually consistent.

    """

    def test_distribution_version(self):
        """Expose the checked-out version, not unrelated site metadata.

        """
        import kilin

        with (_ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
        self.assertEqual(kilin.__version__, project["version"])
        self.assertIn("__version__", kilin.__all__)

    def test_uninstalled_checkout_version(self):
        """Read the source manifest when distribution metadata is absent.

        """
        code = (
            "from importlib import metadata\n"
            "def missing(name):\n"
            "    if name == 'kilin':\n"
            "        raise metadata.PackageNotFoundError(name)\n"
            "    return original(name)\n"
            "original = metadata.version\n"
            "metadata.version = missing\n"
            "import kilin\n"
            "print(kilin.__version__)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        with (_ROOT / "pyproject.toml").open("rb") as stream:
            expected = tomllib.load(stream)["project"]["version"]
        self.assertEqual(result.stdout.strip(), expected)

    def test_stale_distribution_version(self):
        """Prefer source metadata over an obsolete installed KíLín.

        """
        code = (
            "from importlib import metadata\n"
            "original = metadata.version\n"
            "def stale(name):\n"
            "    if name == 'kilin':\n"
            "        return '0.1.0'\n"
            "    return original(name)\n"
            "metadata.version = stale\n"
            "import kilin\n"
            "print(kilin.__version__)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        with (_ROOT / "pyproject.toml").open("rb") as stream:
            expected = tomllib.load(stream)["project"]["version"]
        self.assertEqual(result.stdout.strip(), expected)

    def test_project_metadata(self):
        """Validate source metadata without assuming a matching install.

        Installed-wheel author/version checks are performed separately by
        ``tests/setup/check_installed_wheel.py``.

        """
        with (_ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
        names = [author["name"] for author in project["authors"]]
        self.assertEqual(tuple(names), _EXPECTED_AUTHORS)
        self.assertEqual(project["name"], "kilin")
        self.assertEqual(project["readme"], "README.pypi.md")
        self.assertEqual(project["version"], "0.1.0b1")

    def test_release_author_identities(self):
        """Reject unintended changes to the credited release authors.

        This assertion is independent of the manifest and citation, so
        modifying both files together cannot silently drop an author.

        """
        with (_ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
        self.assertEqual(
            tuple(author["name"] for author in project["authors"]),
            _EXPECTED_AUTHORS,
        )

    def test_prerelease_and_lockfile_versions(self):
        """Keep the first beta and editable uv package in agreement.

        """

        with (_ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
        self.assertEqual(project["version"], "0.1.0b1")
        with (_ROOT / "uv.lock").open("rb") as stream:
            lock = tomllib.load(stream)
        local = [
            package for package in lock["package"]
            if package["name"] == "kilin"
        ]
        self.assertEqual(len(local), 1)
        self.assertEqual(local[0]["version"], project["version"])
        self.assertEqual(
            project["urls"]["Repository"],
            "https://github.com/DeepTrackAI/KiLin",
        )

    def test_citation_metadata(self):
        """Validate the CFF structure, author order, and release version.

        """
        with (_ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
        citation_path = _ROOT / "CITATION.cff"
        citation = yaml.safe_load(citation_path.read_text(encoding="utf-8"))
        self.assertIsInstance(citation, dict)
        self.assertEqual(citation["cff-version"], "1.2.0")
        self.assertEqual(citation["type"], "software")
        self.assertEqual(
            citation["title"], "KíLín: Your Own Private AI Assistant"
        )
        self.assertTrue(citation["message"])
        self.assertEqual(citation["version"], project["version"])
        self.assertEqual(
            [set(author) for author in citation["authors"]],
            [{"given-names", "family-names"}] * len(citation["authors"]),
        )
        self.assertEqual(
            [
                f"{author['given-names']} {author['family-names']}"
                for author in citation["authors"]
            ],
            [author["name"] for author in project["authors"]],
        )

    def test_licence_is_shipped(self):
        """Ship the Apache 2.0 text that the welcome dialog refers to.

        """
        with (_ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
        self.assertEqual(project["license"], "Apache-2.0")
        self.assertEqual(project["license-files"], ["LICENSE"])
        licence = (_ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License\n", licence)
        self.assertIn("Version 2.0, January 2004", licence)
        self.assertIn("END OF TERMS AND CONDITIONS", licence)
        self.assertIn("APPENDIX: How to apply", licence)
        page = (_ROOT / "kilin/gui/web/index.html").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "https://github.com/DeepTrackAI/KiLin/blob/main/LICENSE", page
        )
