"""Load trusted user packages and identify captured plugin resources.

Key Features
------------
- Separate import namespaces without adding user paths to ``sys.path``.
- Source-only loading avoids stale bytecode after a restart.
- Deterministic SHA-256 identity over relative paths and file contents.
- Immutable diagnostics, including rejected candidates.

Module Structure
----------------
- ``ToolPluginInfo`` describes one tool candidate.
- ``_package_digest`` and ``_load_user_package`` implement discovery.

Examples
--------
>>> from kilin.plugins import ToolPluginInfo
>>> ToolPluginInfo.__name__
'ToolPluginInfo'

"""

from dataclasses import dataclass
from hashlib import sha256
from importlib.machinery import (
    PathFinder, SourceFileLoader, SourcelessFileLoader,
)
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import re
import sys
from types import CodeType, ModuleType

__all__ = ["ToolPluginInfo"]

_PREFIX = "kilin.plugins._user_"


@dataclass(frozen=True, slots=True)
class ToolPluginInfo:
    """Describe a discovered or validated tool package.

    Attributes
    ----------
    name: str
        Candidate directory name.
    path: pathlib.Path
        Resolved package location, or the attempted path on failure.
    origin: str
        Distribution or user storage scope.
    version: str | None
        Declared individual version, when validated.
    api_version: int | None
        Declared API generation, when validated.
    sha256: str | None
        Deterministic digest of captured source and resource files.
    error: str | None
        Diagnostic for an unavailable candidate.
    warnings: tuple[str, ...]
        Nonfatal diagnostics, including malformed optional help.
    runtime_checked: bool
        Whether runtime validation was attempted rather than structural checks
        only. A failed runtime check does not imply code was executed.

    """

    name: str
    path: Path
    origin: str
    version: str | None = None
    api_version: int | None = None
    sha256: str | None = None
    error: str | None = None
    warnings: tuple[str, ...] = ()
    runtime_checked: bool = True

    @property
    def available(self) -> bool:
        """Report whether the candidate has no validation error."""

        return self.error is None and self.runtime_checked

    def identity(self) -> dict[str, object]:
        """Return portable execution identity without absolute user paths."""

        return {
            "name": self.name, "origin": self.origin,
            "version": self.version, "api_version": self.api_version,
            "sha256": self.sha256,
        }


def _scan_error(error: OSError) -> None:
    """Propagate a directory traversal failure rather than omit files.

    Parameters
    ----------
    error: OSError
        Error reported by os.walk.

    """

    raise error


def _package_digest(directory: Path) -> str:
    """Hash regular source/resource files with unambiguous path framing.

    Parameters
    ----------
    directory: pathlib.Path
        Resolved plugin package directory.

    Returns
    -------
    str
        SHA-256 over sorted relative paths and per-file SHA-256 hashes.

    Raises
    ------
    ValueError
        If an owned file or directory is linked or not a regular file.

    """

    paths = []
    for folder, dirs, names in os.walk(directory, onerror=_scan_error):
        root = Path(folder)
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in dirs:
            if (root / name).is_symlink():
                raise ValueError(f"Linked plugin directory: {root / name}")
        for name in names:
            path = root / name
            if path.suffix in (".pyc", ".pyo"):
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Invalid plugin file: {path}")
            paths.append(path)
    digest = sha256()
    ordered = sorted(paths, key=lambda p: p.relative_to(directory).as_posix())
    for path in ordered:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        content = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                content.update(block)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(content.digest())
    return digest.hexdigest()


def _check_package(directory: Path) -> Path:
    """Validate a candidate package without importing it.

    Parameters
    ----------
    directory: pathlib.Path
        User-supplied package location.

    Returns
    -------
    pathlib.Path
        Resolved directory with a regular package initializer.

    """

    if not re.fullmatch(r"[a-z][a-z0-9_]*", directory.name):
        raise ValueError(f"Invalid tool package ID: {directory.name!r}.")
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Expected a regular plugin directory: {directory}")
    directory = directory.resolve()
    initial = directory / "__init__.py"
    if initial.is_symlink() or not initial.is_file():
        raise ValueError(f"Missing regular package initializer: {initial}")
    return directory


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate keys in user-owned execution defaults.

    Parameters
    ----------
    pairs: list[tuple[str, object]]
        JSON object members before conversion to a dictionary.

    Returns
    -------
    dict[str, object]
        Object without duplicate keys.

    """

    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate plugin default key: {key!r}.")
        result[key] = value
    return result


def _check_defaults(directory: Path) -> None:
    """Check optional user defaults without invoking plugin code.

    Parameters
    ----------
    directory: pathlib.Path
        Validated candidate package root.

    """

    path = directory / "defaults.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"),
                           object_pairs_hook=_unique_pairs)
        if not isinstance(value, dict):
            raise ValueError(f"{path} must contain a JSON object.")
        # JSON's nonstandard NaN and infinity extensions are not accepted.
        json.dumps(value, allow_nan=False)


class _SourceLoader(SourceFileLoader):
    """Compile plugin source directly instead of trusting bytecode caches."""

    def get_code(self, fullname: str) -> CodeType:
        """Compile the current source for a module in the plugin namespace.

        Parameters
        ----------
        fullname: str
            Qualified module name selected by the import system.

        Returns
        -------
        types.CodeType
            Code compiled from the source file.

        """

        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)


class _UserSourceFinder:
    """Apply source-only loading to descendants of user tool packages."""

    def find_spec(self, fullname, path=None, target=None):
        """Find a source module inside an already loaded user package.

        Parameters
        ----------
        fullname: str
            Requested qualified name.
        path: Sequence[str] | None
            Parent package search locations.
        target: ModuleType | None
            Existing module during reload (not supported by KíLín).

        Returns
        -------
        ModuleSpec | None
            Source-only specification or None for unrelated modules.

        """

        if not fullname.startswith(_PREFIX) or path is None:
            return None
        if "__pycache__" in fullname.split("."):
            raise ImportError("Plugin bytecode-cache folders are reserved.")
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is not None and isinstance(spec.loader, SourcelessFileLoader):
            raise ImportError("User tools require Python source modules.")
        if spec is not None and isinstance(spec.loader, SourceFileLoader):
            spec.loader = _SourceLoader(fullname, spec.origin)
        return spec


_FINDER = _UserSourceFinder()


def _discard_user_package(name: str) -> None:
    """Remove a failed package and its descendants from the import cache.

    Parameters
    ----------
    name: str
        Candidate tool identifier.

    """

    prefix = _PREFIX + name
    for key in tuple(sys.modules):
        if key == prefix or key.startswith(prefix + "."):
            sys.modules.pop(key, None)


def _load_user_package(directory: Path) -> ModuleType:
    """Import one trusted package under its own qualified name.

    Parameters
    ----------
    directory: pathlib.Path
        Validated package directory.

    Returns
    -------
    ModuleType
        Imported package with its package-local relative imports enabled.

    """

    name = _PREFIX + directory.name
    if name in sys.modules:
        raise ValueError(f"Plugin namespace already loaded: {name}.")
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
    initial = directory / "__init__.py"
    spec = spec_from_file_location(
        name, initial, loader=_SourceLoader(name, str(initial)),
        submodule_search_locations=[str(directory)],
    )
    if spec is None:
        raise ValueError(f"Cannot load plugin package: {directory}.")
    module = module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        _discard_user_package(directory.name)
        raise
    return module
