"""Verify a wheel installation without importing from the source tree.

Run this file with ``python -I`` in an independent environment after
installing the wheel. It checks packaged resources and CLI discoverability,
without importing optional backends or opening a Qt application. The
resources expected are every data file of the source package beside this
checkout (not a list kept by hand), so a new tool's ``defaults.json`` that
``pyproject.toml`` forgets to package fails the check.

"""

from __future__ import annotations

from importlib import metadata, resources
from pathlib import Path
import subprocess
import sys
import sysconfig


# The source package of this checkout.
SOURCE = Path(__file__).resolve().parents[2] / "kilin"
# Data files of the source package that are documentation for developers,
# not runtime resources.
NOT_SHIPPED = frozenset({"gui/README.md"})


def source_resources() -> list[str]:
    """Return the data files of the source package that must be installed.

    Returns
    -------
    list[str]
        Paths relative to the package, in order: every file that is not
        Python code, a cache or hidden, except ``NOT_SHIPPED``.

    Raises
    ------
    AssertionError
        If the source package is not beside this file.

    """

    if not SOURCE.is_dir():
        raise AssertionError(f"Source package not found: {SOURCE}")
    names = []
    for path in sorted(SOURCE.rglob("*")):
        relative = path.relative_to(SOURCE)
        if (
            path.is_file()
            and path.suffix not in {".py", ".pyc"}
            and "__pycache__" not in relative.parts
            and not any(part.startswith(".") for part in relative.parts)
            and relative.as_posix() not in NOT_SHIPPED
        ):
            names.append(relative.as_posix())
    return names


def check_installed_package() -> None:
    """Validate the installed distribution and its resource inventory.

    Raises
    ------
    AssertionError
        If imports or packaged resources resolve incorrectly.
    subprocess.CalledProcessError
        If the installed console entry point cannot display help.

    """

    import kilin

    package_file = Path(kilin.__file__).resolve()
    if not package_file.is_relative_to(Path(sys.prefix).resolve()):
        raise AssertionError(
            "KíLín was imported outside the isolated installation: "
            f"{package_file}"
        )
    if not metadata.version("kilin"):
        raise AssertionError("Missing KíLín distribution metadata")
    if "PySide6" in sys.modules:
        raise AssertionError("Core imports must not load PySide6")

    package = resources.files("kilin")
    expected = source_resources()
    if not expected:
        raise AssertionError(f"No data files found in {SOURCE}")
    for name in expected:
        resource = package.joinpath(*name.split("/"))
        if not resource.is_file():
            raise AssertionError(f"Missing wheel resource: {name}")
        if not resource.read_bytes():
            raise AssertionError(f"Empty wheel resource: {name}")

    scripts = Path(sysconfig.get_path("scripts"))
    script = scripts / ("kilin.exe" if sys.platform == "win32" else "kilin")
    if not script.is_file():
        raise AssertionError("The kilin console script was not installed")
    result = subprocess.run(
        [script, "--help"],
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    if "--gui" not in result.stdout:
        raise AssertionError("The CLI help does not expose --gui")
    print(f"Verified kilin {metadata.version('kilin')} wheel resources")


if __name__ == "__main__":
    check_installed_package()
