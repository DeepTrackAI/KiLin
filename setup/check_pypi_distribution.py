"""Check built distribution descriptions against the generated PyPI README.

Run this after ``uv build`` and before publishing. The wheel metadata and
source archive must come from the same tagged checkout description; a
successful build alone does not establish this correspondence.

"""

from __future__ import annotations

from email import policy
from email.parser import BytesParser
from pathlib import Path
import sys
import tarfile
import tomllib
import zipfile


__all__ = ["verify_distributions", "main"]

_ROOT = Path(__file__).resolve().parents[1]


def _one(paths: list[Path], label: str) -> Path:
    """Require exactly one artifact of the declared version.

    Parameters
    ----------
    paths : list[Path]
        Matching distribution candidates.
    label : str
        Artifact type for a diagnostic error.

    Returns
    -------
    Path
        The unique artifact.

    Raises
    ------
    AssertionError
        If a required artifact is missing or ambiguous.

    """

    if len(paths) != 1:
        raise AssertionError(
            f"Expected one {label}, found {len(paths)}: {paths}"
        )
    return paths[0]


def verify_distributions(root: Path, dist: Path) -> None:
    """Verify wheel and sdist metadata before publication.

    Parameters
    ----------
    root : Path
        Repository checkout containing both READMEs and ``pyproject``.
    dist : Path
        Directory holding built wheel and source distribution.

    Raises
    ------
    AssertionError
        If an artifact is missing or does not embed the expected README.

    """

    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    version = project["version"]
    if project["readme"] != "README.pypi.md":
        raise AssertionError("pyproject must use generated PyPI README")
    expected = (root / "README.pypi.md").read_text(encoding="utf-8")
    original = (root / "README.md").read_bytes()
    wheel = _one(sorted(dist.glob(f"kilin-{version}-*.whl")), "wheel")
    sdist = _one(sorted(dist.glob(f"kilin-{version}.tar.gz")), "sdist")

    with zipfile.ZipFile(wheel) as archive:
        files = [
            item for item in archive.namelist()
            if item.endswith(".dist-info/METADATA")
        ]
        if len(files) != 1:
            raise AssertionError("Wheel must contain one METADATA file")
        metadata_bytes = archive.read(files[0])
        message = BytesParser(policy=policy.default).parsebytes(
            metadata_bytes, headersonly=True
        )
        if message["Version"] != version:
            raise AssertionError("Wheel metadata version is inconsistent")
        if message["Description-Content-Type"] != "text/markdown":
            raise AssertionError("Wheel description is not Markdown")
        _, separator, description = metadata_bytes.partition(b"\n\n")
        if not separator:
            raise AssertionError("Wheel METADATA lacks a description body")
        if description.decode("utf-8").rstrip("\n") != expected.rstrip("\n"):
            raise AssertionError("Wheel description differs from PyPI README")

    with tarfile.open(sdist, mode="r:gz") as archive:
        members = [item for item in archive if item.isfile()]
        for name, content in (
            ("README.md", original),
            ("README.pypi.md", expected.encode("utf-8")),
        ):
            matches = [
                item for item in members
                if Path(item.name).name == name
                and len(Path(item.name).parts) == 2
            ]
            if len(matches) != 1:
                raise AssertionError(f"Sdist must contain one root {name}")
            stream = archive.extractfile(matches[0])
            if stream is None or stream.read() != content:
                raise AssertionError(f"Sdist {name} differs from checkout")
    print(f"Verified PyPI README in kilin {version} wheel and sdist")


def main() -> int:
    """Validate distributions under the checkout's ``dist/`` folder.

    Returns
    -------
    int
        Zero when both distributions match the checked-out metadata.

    """

    try:
        verify_distributions(_ROOT, _ROOT / "dist")
    except (AssertionError, OSError, ValueError) as error:
        print(f"Invalid KíLín PyPI distributions: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
