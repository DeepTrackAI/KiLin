"""Validate candidate tool folders without changing the active registry.

Structural checks inspect files and compile source without executing it.
Trusted runtime checks use a disposable interpreter with the current
KíLín package and Python environment. This is not a security sandbox.

Key Features
------------
- Separate nonexecuting and explicitly trusted validation.
- Runtime checks do not replace active plugin modules or cached defaults.
- A bounded worker lifetime and structured diagnostic results.

Module Structure
----------------
- ``_validate_tool`` checks a candidate package.
- ``_worker`` imports a trusted candidate in a fresh process.

Examples
--------
>>> from kilin.plugins import validate_tool
>>> callable(validate_tool)
True

"""

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from kilin.plugins._discovery import (
    ToolPluginInfo, _check_defaults, _check_package, _package_digest,
)

__all__: list[str] = []

_TIMEOUT = 30


def _structural_check(path: Path) -> ToolPluginInfo:
    """Check a candidate's layout, source syntax and defaults JSON.

    Parameters
    ----------
    path: pathlib.Path
        Candidate package folder, not necessarily installed.

    Returns
    -------
    ToolPluginInfo
        Structural diagnostic and content digest, without executing code.

    """

    info = ToolPluginInfo(
        path.name, path.absolute(), "user", runtime_checked=False,
    )
    try:
        directory = _check_package(path)
        bundled = Path(__file__).resolve().parents[1] / "tools"
        if (bundled / directory.name / "__init__.py").is_file():
            raise ValueError(
                f"Tool ID {directory.name!r} is already registered."
            )
        digest = _package_digest(directory)
        for source in sorted(directory.rglob("*.py")):
            if "__pycache__" not in source.relative_to(directory).parts:
                compile(source.read_bytes(), str(source), "exec")
        _check_defaults(directory)
        return replace(info, path=directory, sha256=digest)
    except (OSError, ValueError, SyntaxError, RuntimeError) as error:
        return replace(info, error=f"{type(error).__name__}: {error}")


def _worker(path: Path, output: Path) -> None:
    """Import and register a trusted candidate in the disposable process.

    Parameters
    ----------
    path: pathlib.Path
        Candidate package folder.
    output: pathlib.Path
        Private result file used instead of the plugin's stdout.

    """

    from kilin.tools._registry import _register_user_tool

    try:
        tool = _register_user_tool(path)
        info = tool._info
    except (Exception, SystemExit) as error:
        info = ToolPluginInfo(
            path.name, path, "user",
            error=f"{type(error).__name__}: {error}",
        )
    data = asdict(info)
    data["path"] = str(info.path)
    output.write_text(json.dumps(data), encoding="utf-8")


def _validate_tool(path: Path, *, trusted: bool) -> ToolPluginInfo:
    """Perform structural and optionally isolated runtime validation.

    Parameters
    ----------
    path: pathlib.Path
        Candidate package location.
    trusted: bool
        Explicit permission to execute candidate code in a subprocess.

    Returns
    -------
    ToolPluginInfo
        Validation outcome, not a change to the runtime catalog.

    """

    if type(trusted) is not bool:
        raise TypeError("trusted must be a boolean.")
    info = _structural_check(path)
    if info.error is not None or not trusted:
        return info
    script = (
        "import sys; from pathlib import Path; "
        "sys.path.insert(0, sys.argv[1]); "
        "from kilin.plugins._validation import _worker; "
        "_worker(Path(sys.argv[2]), Path(sys.argv[3]))"
    )
    host = Path(__file__).resolve().parents[2]
    try:
        with TemporaryDirectory(prefix="kilin-plugin-check-") as temp:
            output = Path(temp).resolve() / "result.json"
            process = subprocess.run(
                [sys.executable, "-B", "-c", script, str(host),
                 str(info.path), str(output)],
                cwd=temp, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=_TIMEOUT, check=False,
            )
            if process.returncode != 0 or not output.is_file():
                raise ValueError(
                    "Plugin validation worker did not return a result "
                    f"(exit status {process.returncode})."
                )
            data = json.loads(output.read_text(encoding="utf-8"))
            data["path"] = Path(data["path"])
            data["warnings"] = tuple(data["warnings"])
            result = ToolPluginInfo(**data)
            if result.error is not None:
                return replace(info, error=result.error,
                               runtime_checked=True)
            if result.sha256 != info.sha256:
                raise ValueError("Plugin changed during validation.")
            return result
    except subprocess.TimeoutExpired:
        error = f"Plugin runtime validation exceeded {_TIMEOUT} seconds."
    except (OSError, ValueError, TypeError, KeyError) as failure:
        error = f"{type(failure).__name__}: {failure}"
    return replace(info, error=error, runtime_checked=True)
