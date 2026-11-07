"""Verify licence checks for wheel and source distributions.

Each test builds minimal in-memory archives, so no packaging backend or
network access is needed to exercise failure handling.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from io import BytesIO
import unittest
import zipfile

from tests.setup.check_built_distributions import check_built_distributions


class TestBuiltDistributions(unittest.TestCase):
    """Reject incomplete or inconsistent release artefacts.

    """

    def setUp(self) -> None:
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        (self.root / "dist").mkdir()
        self.licence = b"Apache License\nVersion 2.0, January 2004\n"
        (self.root / "LICENSE").write_bytes(self.licence)
        self.create_archives(self.licence, self.licence)

    def create_archives(
        self,
        wheel_licence: bytes | None,
        source_licence: bytes | None,
    ) -> None:
        """Create minimal distribution archives for a test.

        Parameters
        ----------
        wheel_licence : bytes | None
            Wheel licence text, or ``None`` to omit the file.
        source_licence : bytes | None
            Source licence text, or ``None`` to omit the file.

        """

        wheel = self.root / "dist/kilin-0.1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            if wheel_licence is not None:
                archive.writestr(
                    "kilin-0.1.0.dist-info/licenses/LICENSE",
                    wheel_licence,
                )
        sdist = self.root / "dist/kilin-0.1.0.tar.gz"
        with tarfile.open(sdist, "w:gz") as archive:
            if source_licence is not None:
                member = tarfile.TarInfo("kilin-0.1.0/LICENSE")
                member.size = len(source_licence)
                archive.addfile(member, BytesIO(source_licence))

    def test_valid_archives(self) -> None:
        check_built_distributions(self.root)

    def test_missing_wheel_licence(self) -> None:
        self.create_archives(None, self.licence)
        with self.assertRaisesRegex(AssertionError, "Wheel LICENSE"):
            check_built_distributions(self.root)

    def test_modified_wheel_licence(self) -> None:
        self.create_archives(b"Different", self.licence)
        with self.assertRaisesRegex(AssertionError, "Wheel LICENSE"):
            check_built_distributions(self.root)

    def test_missing_source_licence(self) -> None:
        self.create_archives(self.licence, None)
        with self.assertRaisesRegex(AssertionError, "tarball.*LICENSE"):
            check_built_distributions(self.root)

    def test_modified_source_licence(self) -> None:
        self.create_archives(self.licence, b"Different")
        with self.assertRaisesRegex(AssertionError, "tarball.*LICENSE"):
            check_built_distributions(self.root)

    def test_missing_wheel_archive(self) -> None:
        (self.root / "dist/kilin-0.1.0-py3-none-any.whl").unlink()
        with self.assertRaisesRegex(AssertionError, "Expected one"):
            check_built_distributions(self.root)


if __name__ == "__main__":
    unittest.main()
