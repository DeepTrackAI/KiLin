"""Verify that built KíLín distributions include the declared licence.

Run after ``uv build`` in the source checkout. This checks the actual
wheel and source distribution, rather than assuming that the packaging
metadata implies the presence of the licence text.

"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import tarfile
import zipfile


def check_built_distributions(root: Path) -> None:
    """Compare the licence bytes in both archives with the source file.

    Parameters
    ----------
    root : pathlib.Path
        Checkout containing ``LICENSE`` and the ``dist/`` directory.

    Raises
    ------
    AssertionError
        If an archive or its licence is missing or inconsistent.

    """

    expected = (root / "LICENSE").read_bytes()
    if not expected:
        raise AssertionError("The source LICENSE is empty")

    dist = root / "dist"
    wheels = sorted(dist.glob("kilin-*.whl"))
    sdists = sorted(dist.glob("kilin-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise AssertionError(
            "Expected one KíLín wheel and one source tarball"
        )

    with zipfile.ZipFile(wheels[0]) as archive:
        matches = [
            name for name in archive.namelist()
            if len(PurePosixPath(name).parts) == 3
            and PurePosixPath(name).parts[0].endswith(".dist-info")
            and PurePosixPath(name).parts[1:] == ("licenses", "LICENSE")
        ]
        if len(matches) != 1 or archive.read(matches[0]) != expected:
            raise AssertionError("Wheel LICENSE differs from source")

    with tarfile.open(sdists[0], "r:gz") as archive:
        matches = [
            member for member in archive.getmembers()
            if len(PurePosixPath(member.name).parts) == 2
            and PurePosixPath(member.name).parts[1] == "LICENSE"
            and member.isfile()
        ]
        if len(matches) != 1:
            raise AssertionError("Source tarball is missing LICENSE")
        stream = archive.extractfile(matches[0])
        if stream is None or stream.read() != expected:
            raise AssertionError("Source tarball LICENSE differs from source")


if __name__ == "__main__":
    check_built_distributions(Path(__file__).resolve().parents[2])
    print("Verified wheel and source distribution licences")
