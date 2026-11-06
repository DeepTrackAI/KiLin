"""Verify the version, authors, licence, and citation metadata.

The tests compare the source manifest, installed distribution metadata,
and the structured CFF citation. ``pyproject.toml`` is the one list of
authors and the one version; the citation must repeat both unchanged, so
the tests hold no third copy that would need manual release updates.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from importlib import metadata
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

import yaml


_ROOT = Path(__file__).resolve().parents[2]


class TestReleaseMetadata(unittest.TestCase):
    """Keep the package and citation metadata mutually consistent.

    """

    def test_distribution_version(self):
        """Export the version declared by the installed distribution.

        """
        import kilin

        self.assertEqual(kilin.__version__, metadata.version("kilin"))
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

    def test_project_metadata(self):
        """Expose the author order in installed package metadata.

        """
        with (_ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
        names = [author["name"] for author in project["authors"]]
        self.assertTrue(names)
        self.assertEqual(project["version"], metadata.version("kilin"))
        self.assertEqual(project["name"], "kilin")
        self.assertEqual(project["readme"], "README.md")
        installed = metadata.metadata("kilin")
        self.assertEqual(installed["Author"], ", ".join(names))

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
        page = (_ROOT / "kilin/gui/web/index.html").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "https://github.com/giovannivolpe/KiLin/blob/main/LICENSE", page
        )
