"""Verify a wheel installation without importing from the source tree.

Run this file with ``python -I`` in an independent environment after
installing the wheel. It checks packaged resources and CLI discoverability,
without importing optional backends or opening a Qt application.

"""

from __future__ import annotations

from importlib import metadata, resources
from pathlib import Path
import subprocess
import sys
import sysconfig


TOOL_PACKAGES = (
    "project_inventory",
    "scan_and_list",
    "batch_classify",
    "batch_review",
    "filter_results",
    "find_fact",
    "find_passages",
    "paper_overview",
    "show_source",
    "generate_paper_summary",
)


def check_installed_package() -> None:
    """Validate the installed distribution and its resource inventory.

    Raises
    ------
    AssertionError
        If imports or packaged resources resolve incorrectly.
    subprocess.CalledProcessError
        If the installed console entry point cannot display help.

    """

    import qachat

    package_file = Path(qachat.__file__).resolve()
    if not package_file.is_relative_to(Path(sys.prefix).resolve()):
        raise AssertionError(
            "QAChat was imported outside the isolated installation: "
            f"{package_file}"
        )
    if not metadata.version("qachat"):
        raise AssertionError("Missing QAChat distribution metadata")
    if "PySide6" in sys.modules:
        raise AssertionError("Core imports must not load PySide6")

    package = resources.files("qachat")
    expected = [
        "qachat_defaults.json",
        "repository_reason_codes.json",
        "project_types/code.json",
        "project_types/literature.json",
        "gui/defaults.json",
        "gui/web/index.html",
        "gui/web/app.js",
        "gui/web/styles.css",
        "gui/icons/qachat-dark.png",
        "gui/icons/qachat-light.png",
        "tools/generate_paper_summary/templates/technical_summary.md",
        "tools/generate_paper_summary/templates/popular_science_summary.md",
        "tools/generate_paper_summary/templates/research_briefing.md",
        "tools/generate_paper_summary/templates/experimental_protocol.md",
    ]
    for tool in TOOL_PACKAGES:
        expected.extend((
            f"tools/{tool}/defaults.json",
            f"tools/{tool}/README.md",
        ))
    for name in expected:
        resource = package.joinpath(*name.split("/"))
        if not resource.is_file():
            raise AssertionError(f"Missing wheel resource: {name}")
        if not resource.read_bytes():
            raise AssertionError(f"Empty wheel resource: {name}")

    scripts = Path(sysconfig.get_path("scripts"))
    script = scripts / ("qachat.exe" if sys.platform == "win32" else "qachat")
    if not script.is_file():
        raise AssertionError("The qachat console script was not installed")
    result = subprocess.run(
        [script, "--help"],
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    if "--gui" not in result.stdout:
        raise AssertionError("The CLI help does not expose --gui")
    print(f"Verified qachat {metadata.version('qachat')} wheel resources")


if __name__ == "__main__":
    check_installed_package()
