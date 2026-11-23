"""Test automatic local README preparation in the unittest entry point.

Tests invoke the entry point with mocked discovery, so the full suite is
never run recursively. Temporary checkouts protect the source README.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from setup import generate_pypi_readme as readme
from tests import __main__ as runner


class TestReadmePreparation(unittest.TestCase):
    """Distinguish local regeneration from read-only CI checks.

    """

    def test_local_run_regenerates_before_discovery(self) -> None:
        """An outdated description is repaired before tests are loaded.

        """

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text(
                '[project]\nversion = "0.1.0b2"\n', encoding="utf-8"
            )
            (root / "README.md").write_text(
                "# KíLín\n", encoding="utf-8"
            )
            target = root / "README.pypi.md"
            target.write_text("outdated\n", encoding="utf-8")
            expected = readme.render_pypi_readme(
                "# KíLín\n", "0.1.0b2", root
            )

            def discover(**kwargs):
                """Assert regeneration has preceded unittest discovery.

                Parameters
                ----------
                **kwargs : object
                    Discovery arguments passed by the entry point.

                Returns
                -------
                unittest.TestSuite
                    Empty suite that avoids recursive discovery.

                """

                self.assertEqual(target.read_text(encoding="utf-8"),
                                 expected)
                return unittest.TestSuite()

            with (
                patch.dict(os.environ, {"CI": ""}),
                patch.object(readme, "_ROOT", root),
                patch.object(
                    runner.unittest.defaultTestLoader,
                    "discover", side_effect=discover
                ) as loader,
                patch.object(runner.unittest, "TextTestRunner") as tests,
            ):
                result = tests.return_value.run.return_value
                result.wasSuccessful.return_value = True
                self.assertEqual(runner.main(), 0)
                loader.assert_called_once()
                tests.return_value.run.assert_called_once()

            self.assertEqual(target.read_text(encoding="utf-8"), expected)

    def test_ci_run_does_not_regenerate(self) -> None:
        """GitHub Actions must not silently repair generated content.

        """

        with (
            patch.dict(os.environ, {"CI": "true"}),
            patch.object(runner, "generate_pypi_readme") as generate,
            patch.object(
                runner.unittest.defaultTestLoader, "discover",
                return_value=unittest.TestSuite()
            ) as loader,
            patch.object(runner.unittest, "TextTestRunner") as tests,
        ):
            result = tests.return_value.run.return_value
            result.wasSuccessful.return_value = True
            self.assertEqual(runner.main(), 0)
            generate.assert_not_called()
            loader.assert_called_once()

    def test_generation_failure_aborts_before_discovery(self) -> None:
        """Do not run tests after the README generator fails.

        """

        with (
            patch.dict(os.environ, {"CI": "false"}),
            patch.object(runner, "generate_pypi_readme", return_value=1),
            patch.object(
                runner.unittest.defaultTestLoader, "discover"
            ) as loader,
        ):
            self.assertEqual(runner.main(), 1)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
