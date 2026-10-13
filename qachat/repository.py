"""Select local source files and compare content manifests.

``scan_repository`` inspects metadata without reading file contents. It
applies literal suffix matching, relative include/exclude scopes, size and
entry limits, and built-in skip rules. Exclusions and protected-name rules
cannot be overridden by an inclusion. Descendant symbolic links and
junctions are skipped rather than traversed.

``build_repository_manifest`` requires a complete scan and then reads
selected files to compute SHA-256 hashes. Manifests contain relative paths,
sizes, and hashes, not the absolute source location or modification times.
Comparing manifests detects changes in selected content, not changes in every
file in the repository.

These routines are not a security sandbox or an atomic filesystem snapshot.
Manifest construction expects a trusted scan report and does not repeat all
scan-time path checks. A scan's ``complete`` flag reports recoverable I/O
failures; manifest construction rejects incomplete reports.
Relative repository paths use ``PROJECT_ROOT`` (the package's parent), while
relative saved-manifest paths use the process working directory.

Key Features
------------

- **Selection reports**

    Return sorted selected/skipped rows with metadata and explanatory reasons.

- **Content fingerprints**

    Read selected file bytes only when building a versioned manifest.

- **Change reports**

    Compare normalized size/hash records and return sorted added, removed, and
    modified paths.

- **Missing-baseline distinction**

    Treat an absent saved manifest as a non-match even when the current
    selection is empty.

Module Structure
----------------

- ``build_repository_manifest``

    Build a content manifest for selected repository files.

- ``compare_repository_manifests``

    Compare two repository content manifests.

- ``detect_repository_changes``

    Detect added, removed, or modified selected repository files.

- ``repository_has_changed``

    Return whether selected repository content has changed.

- ``normalize_extensions``

    Validate and normalize literal file suffixes across the configuration
    loader, profile reader, and repository scanner.

- ``scan_repository``

    Preview files selected from a local repository without reading them.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Module settings
---------------
``PROJECT_ROOT`` anchors relative repository paths to the package parent.
The packaged ``qachat_defaults.json`` contains the repository-selection
defaults. ``DEFAULT_EXTENSIONS`` supplies the literal filename suffixes;
``SKIP_DIRECTORIES``, ``SKIP_FILES``, and ``BLOCKED_SUFFIXES`` provide
name-based exclusions. These lists are loaded at import time and define
which extensions are considered, which directory and file names are skipped,
and which binary or sensitive suffixes are rejected. They can be edited in
the packaged JSON; changes require restarting QAChat. They are not
included in ``__all__`` or re-exported from the package root.

Examples
--------
>>> import qachat
>>> report = qachat.scan_repository(
...     "/path/to/repository", extensions=[".py", ".md"],
... )
>>> if not report["complete"]:
...     raise RuntimeError("Repository scan was incomplete.")
>>> manifest = qachat.build_repository_manifest(report)
>>> changes = qachat.compare_repository_manifests(manifest, manifest)

Use an absolute repository path to avoid dependence on the package location.

"""

from __future__ import annotations

import os
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat

from qachat.defaults import (
    load_defaults_section,
    require_default_strings,
)


__all__ = [
    "build_repository_manifest",
    "compare_repository_manifests",
    "detect_repository_changes",
    "normalize_extensions",
    "repository_has_changed",
    "scan_repository",
]

_MANIFEST_VERSION = 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Selection defaults are owned by this module; the JSON is loaded once.
_REPOSITORY_DEFAULTS = load_defaults_section("REPOSITORY")
DEFAULT_EXTENSIONS = require_default_strings(
    _REPOSITORY_DEFAULTS, "DEFAULT_EXTENSIONS", "REPOSITORY"
)
SKIP_DIRECTORIES = set(
    require_default_strings(
        _REPOSITORY_DEFAULTS, "SKIP_DIRECTORIES", "REPOSITORY"
    )
)
SKIP_FILES = set(
    require_default_strings(
        _REPOSITORY_DEFAULTS, "SKIP_FILES", "REPOSITORY"
    )
)
BLOCKED_SUFFIXES = set(
    require_default_strings(
        _REPOSITORY_DEFAULTS, "BLOCKED_SUFFIXES", "REPOSITORY"
    )
)


def _selection_paths(
    values: list[str] | tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    """Check paths used to include or exclude repository content.

    Parameters
    ----------
    values: list[str] | tuple[str, ...]
        Relative paths inside the selected repository.
    label: str
        Setting name used in an error message.

    Returns
    -------
    tuple[str, ...]
        Normalized paths with duplicates removed.

    Raises
    ------
    ValueError
        If values is not a list/tuple, an item is blank or non-string, or it
        contains absolute/drive-qualified paths, parent traversal, wildcards,
        colon, or a null byte.

    Notes
    -----
    Separators are normalized and duplicates are removed in first-seen order.
    Whitespace is tested for blank input but is not stripped from a nonblank
    path. The path "." denotes the whole selection scope.

    """

    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} must be a list, for example ['qachat'].")

    result: list[str] = []
    for value in values:

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} needs non-empty relative paths.")

        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        invalid_character = any(
            character in normalized for character in "*?[]:\0"
        )
        if (
            path.is_absolute()
            or bool(PureWindowsPath(value).drive)
            or ".." in path.parts
            or invalid_character
        ):
            raise ValueError(
                f"{label} must contain relative paths without wildcards "
                "or '..'."
            )

        result.append(path.as_posix())

    return tuple(dict.fromkeys(result))


def _within(path: str, parent: str) -> bool:
    """Return whether a path is the parent or one of its descendants.

    Parameters
    ----------
    path: str
        Relative path to check.
    parent: str
        Possible parent path.

    Returns
    -------
    bool
        Whether the path is inside the parent.

    """

    return parent == "." or path == parent or path.startswith(parent + "/")


def _skip_reason(
    relative: str,
    kind: str,
    size: int | None,
    extensions: tuple[str, ...],
    includes: tuple[str, ...],
    excludes: tuple[str, ...],
    max_file_bytes: int,
) -> str | None:
    """Explain why one entry should be skipped.

    Parameters
    ----------
    relative: str
        Entry path relative to the selected repository.
    kind: str
        ``file``, ``folder``, ``link``, or ``other``.
    size: int | None
        File size in bytes, or ``None`` for non-files.
    extensions: tuple[str, ...]
        Allowed lowercase file extensions.
    includes: tuple[str, ...]
        Paths that limit the selection.
    excludes: tuple[str, ...]
        Paths removed from the selection.
    max_file_bytes: int
        Largest file allowed.

    Returns
    -------
    str | None
        Skip reason, or ``None`` when the entry is selected or entered.

    """

    name = PurePosixPath(relative).name.lower()

    if kind == "link":
        return "links are not followed"

    if kind == "folder" and name in SKIP_DIRECTORIES:
        return "environment, cache, or generated folder"

    protected_environment = name.startswith(".env.")
    if name in SKIP_FILES or protected_environment:
        return "potentially secret or generated file"

    if PurePosixPath(name).suffix in BLOCKED_SUFFIXES:
        return "model weights or potentially secret file"

    if any(_within(relative, excluded) for excluded in excludes):
        return "excluded by your selection"

    in_scope = not includes or any(
        _within(relative, included) for included in includes
    )
    leads_to_scope = kind == "folder" and any(
        _within(included, relative) for included in includes
    )

    if not in_scope and not leads_to_scope:
        return "outside your included paths"

    if kind == "folder":
        return None

    if kind != "file":
        return "not a regular file"

    if not any(name.endswith(extension) for extension in extensions):
        return "file type not selected"

    if size == 0:
        return "empty file"

    if size is not None and size > max_file_bytes:
        return "larger than the file-size limit"

    return None


def normalize_extensions(
    extensions: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Check and normalize selected file extensions.

    Parameters
    ----------
    extensions: list[str] | tuple[str, ...]
        Extensions with or without their leading period.

    Returns
    -------
    tuple[str, ...]
        Lowercase extensions with duplicates removed.

    Raises
    ------
    ValueError
        If extensions is not a list/tuple or an item is not a supported literal
        suffix string.

    Notes
    -----
    Compound suffixes and special dot-file names are allowed by the literal
    suffix grammar. The empty sequence is valid; it selects no files.
    Whitespace and wildcard patterns are rejected.

    """

    if not isinstance(extensions, (list, tuple)):
        raise ValueError("extensions must be a list, such as ['.py', '.md'].")

    normalized: list[str] = []
    for extension in extensions:

        if not isinstance(extension, str):
            raise ValueError("Use file extensions such as '.py' or '.md'.")

        value = "." + extension.lower().lstrip(".")

        if not re.fullmatch(r"\.[a-z0-9_-]+(?:\.[a-z0-9_-]+)*", value):
            raise ValueError(
                "Use file extensions such as '.py' or '.md', without "
                "wildcards."
            )

        normalized.append(value)

    return tuple(dict.fromkeys(normalized))


def _entry_row(
    path: str,
    kind: str,
    size: int | None,
    reason: str | None,
) -> dict[str, object]:
    """Build one result row.

    Parameters
    ----------
    path: str
        Relative entry path.
    kind: str
        Entry kind.
    size: int | None
        File size in bytes.
    reason: str | None
        Skip reason, or ``None`` for a selected file.

    Returns
    -------
    dict[str, object]
        Result row used by :func:`scan_repository`.

    """

    return {
        "path": path,
        "kind": kind,
        "size_bytes": size,
        "status": "skipped" if reason else "selected",
        "reason": reason or "matches your selection",
    }


def scan_repository(
    repository: str | Path,
    *,
    extensions: list[str] | tuple[str, ...] = DEFAULT_EXTENSIONS,
    include_paths: list[str] | tuple[str, ...] = (),
    exclude_paths: list[str] | tuple[str, ...] = (),
    max_file_bytes: int = 1_048_576,
    max_entries: int = 10_000,
) -> dict[str, object]:
    """Preview files selected from a local repository without reading them.

    Parameters
    ----------
    repository: str | pathlib.Path
        Local source directory. Absolute paths are used directly; relative
        paths are resolved against repository.PROJECT_ROOT, the parent of the
        installed package directory.
    extensions: list[str] | tuple[str, ...], optional
        Literal filename suffixes with optional leading periods. They are
        normalized to lowercase and matched against lowercase names. An empty
        sequence selects no files. Defaults to ``DEFAULT_EXTENSIONS``.
    include_paths: list[str] | tuple[str, ...], optional
        Relative repository paths limiting the selection. An empty sequence
        permits all otherwise eligible paths. No glob patterns are accepted.
        Defaults to ``()``.
    exclude_paths: list[str] | tuple[str, ...], optional
        Relative repository paths excluded before inclusion checks. Excluded
        directories are not traversed. Defaults to ``()``.
    max_file_bytes: int, optional
        Maximum selected-file size in bytes; an exact positive integer. Files
        exactly at the limit are allowed. Defaults to ``1048576``.
    max_entries: int, optional
        Maximum encountered directory entries, including skipped files and
        directories. Exceeding the bound aborts without returning a partial
        report. Defaults to ``10000``.

    Returns
    -------
    dict[str, object]
        Scan report with repository (resolved absolute path), extensions,
        include_paths, exclude_paths, and max_file_bytes. visited_entries
        counts encountered entries; complete is False after recoverable
        descendant I/O errors. selected_files is a sorted list of relative
        POSIX paths and selected_bytes is their metadata-size sum. entries
        contains sorted rows with path, kind, size_bytes, status
        (selected/skipped), and reason. Traversed directories do not receive a
        selected row; skipped directories do. A kind of unknown records an
        inspection failure.

    Raises
    ------
    ValueError
        If the repository value or selection settings are invalid, a
        home/filesystem root is selected, or max_entries is exceeded.
    FileNotFoundError
        If the requested source root cannot be resolved because it does not
        exist.
    NotADirectoryError
        If the resolved source is not a directory.
    OSError
        If resolving the source or listing the root fails. Descendant
        listing/stat errors are recorded in an incomplete report.

    Notes
    -----
    No file contents are read and no .gitignore syntax is interpreted. Built-in
    name/suffix rules still apply inside included paths. Link and containment
    checks are best-effort traversal checks, not race-free access control. A
    chosen root may resolve through a link before traversal; descendant links
    and junctions are skipped. Check complete before treating the result as a
    full inventory.

    """

    if not isinstance(repository, (str, Path)) or not str(repository).strip():
        raise ValueError("Choose a local project folder.")

    if str(repository).lower().startswith(("https:", "http:", "git@")):
        raise ValueError("Choose a local folder, not a remote address.")

    limits = (
        (max_file_bytes, "max_file_bytes"),
        (max_entries, "max_entries"),
    )
    for value, label in limits:
        if type(value) is not int or value < 1:
            raise ValueError(f"{label} must be a positive whole number.")

    selected_extensions = normalize_extensions(extensions)
    includes = _selection_paths(include_paths, "include_paths")
    excludes = _selection_paths(exclude_paths, "exclude_paths")
    chosen = Path(repository).expanduser()
    if not chosen.is_absolute():
        chosen = PROJECT_ROOT / chosen
    root = chosen.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Choose a folder, not a file: {root}")
    if root == Path(root.anchor) or root == Path.home().resolve():
        raise ValueError(
            "Choose a project folder, not a home folder or drive."
        )

    rows: list[dict[str, object]] = []
    pending = [root]
    visited = 0
    complete = True
    while pending:
        folder = pending.pop()
        relative_folder = folder.relative_to(root).as_posix()
        try:
            unsafe = (
                folder.is_symlink()
                or folder.is_junction()
                or not folder.resolve().is_relative_to(root)
            )
            if unsafe:
                rows.append(
                    _entry_row(
                        relative_folder,
                        "link",
                        None,
                        "links are not followed",
                    )
                )
                continue
            with os.scandir(folder) as stream:
                children = []
                for entry in stream:
                    visited += 1
                    if visited > max_entries:
                        raise ValueError(
                            "Scan limit reached. Choose a smaller folder or "
                            "raise max_entries."
                        )
                    children.append(entry)
        except OSError as error:
            if folder == root:
                raise OSError(
                    f"Cannot list the chosen folder: {root}"
                ) from error
            complete = False
            reason = f"cannot list folder ({type(error).__name__})"
            rows.append(_entry_row(relative_folder, "folder", None, reason))
            continue

        for entry in sorted(children, key=lambda item: item.name):
            path = folder / entry.name
            relative = path.relative_to(root).as_posix()
            try:
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode) or path.is_junction():
                    kind = "link"
                elif stat.S_ISDIR(metadata.st_mode):
                    kind = "folder"
                elif stat.S_ISREG(metadata.st_mode):
                    kind = "file"
                else:
                    kind = "other"
                size = metadata.st_size if kind == "file" else None
                reason = _skip_reason(
                    relative,
                    kind,
                    size,
                    selected_extensions,
                    includes,
                    excludes,
                    max_file_bytes,
                )
            except OSError as error:
                complete = False
                kind = "unknown"
                size = None
                reason = f"cannot inspect entry ({type(error).__name__})"

            if kind == "folder" and reason is None:
                pending.append(path)
            else:
                rows.append(_entry_row(relative, kind, size, reason))

    rows.sort(key=lambda row: str(row["path"]))
    selected = [row for row in rows if row["status"] == "selected"]
    selected_bytes = sum(int(row["size_bytes"]) for row in selected)
    return {
        "repository": str(root),
        "extensions": list(selected_extensions),
        "include_paths": list(includes),
        "exclude_paths": list(excludes),
        "max_file_bytes": max_file_bytes,
        "visited_entries": visited,
        "complete": complete,
        "selected_files": [row["path"] for row in selected],
        "selected_bytes": selected_bytes,
        "entries": rows,
    }


def build_repository_manifest(
    report: dict[str, object],
) -> dict[str, object]:
    """Build a content manifest for selected repository files.

    Files listed in a report returned by ``scan_repository`` are read and
    hashed with SHA-256. The manifest contains only relative paths, file sizes,
    and hashes, so it remains stable when the same source tree is copied to a
    different directory.

    Parameters
    ----------
    report: dict[str, object]
        Result returned by :func:`scan_repository`.

    Returns
    -------
    dict[str, object]
        JSON-safe manifest describing the selected repository content.

    Raises
    ------
    ValueError
        If the scan is incomplete or report fields have invalid types;
        individual selected path strings are not fully validated.
    FileNotFoundError
        If the source root or selected regular file is missing.
    OSError
        If a file cannot be opened, read, or inspected.

    Notes
    -----
    This function requires a complete scan and reads selected files in binary
    blocks. It checks file metadata before and after hashing and rejects
    detectable concurrent modifications. It does not reject parent/absolute
    path strings or repeat scan-time link containment checks. Supply a
    trusted scan report. Duplicate normalized path keys overwrite earlier
    records.

    """

    if report.get("complete") is not True:
        raise ValueError("Cannot build a manifest from an incomplete scan.")

    repository_value = report.get("repository")
    selected_value = report.get("selected_files")
    if not isinstance(repository_value, str):
        raise ValueError("Repository report is missing its source folder.")
    if not isinstance(selected_value, list):
        raise ValueError("Repository report is missing selected files.")

    root = Path(repository_value).expanduser().resolve(strict=True)
    files: dict[str, dict[str, object]] = {}
    for relative_value in selected_value:
        if not isinstance(relative_value, str):
            raise ValueError("Repository report contains an invalid path.")
        relative = PurePosixPath(relative_value).as_posix()
        path = root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(
                f"Selected source file disappeared during scanning: {path}"
            )
        digest = sha256()
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(stream.fileno())
        identity = (
            "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"
        )
        if any(getattr(before, name) != getattr(after, name)
               for name in identity):
            raise OSError(
                f"Selected source file changed while hashing: {path}"
            )
        files[relative] = {
            "size_bytes": after.st_size,
            "sha256": digest.hexdigest(),
        }

    return {
        "format_version": _MANIFEST_VERSION,
        "files": files,
    }


def compare_repository_manifests(
    saved: dict[str, object] | None,
    current: dict[str, object],
) -> dict[str, object]:
    """Compare two repository content manifests.

    Parameters
    ----------
    saved: dict[str, object] | None
        Previous supported manifest. None denotes no baseline and always yields
        matches=False, including when the current file mapping is empty.
    current: dict[str, object]
        Current repository manifest.

    Returns
    -------
    dict[str, object]
        Comparison containing ``matches`` and sorted ``added``, ``removed``,
        and ``modified`` relative paths.

    Raises
    ------
    ValueError
        If either manifest has an unsupported format version or invalid file
        records.

    Notes
    -----
    Modified records differ in size_bytes or sha256 after extra record fields
    are discarded. Paths must be normalized relative POSIX paths, sizes must
    be nonnegative, and hashes must be valid SHA-256 hex digests. All change
    lists are sorted lexically; no files are read during comparison.

    """

    current_files = _manifest_files(current)
    if saved is None:
        return {
            "matches": False,
            "added": sorted(current_files),
            "removed": [],
            "modified": [],
        }

    saved_files = _manifest_files(saved)
    saved_paths = set(saved_files)
    current_paths = set(current_files)
    added = sorted(current_paths - saved_paths)
    removed = sorted(saved_paths - current_paths)
    modified = sorted(
        path
        for path in saved_paths & current_paths
        if saved_files[path] != current_files[path]
    )
    return {
        "matches": not added and not removed and not modified,
        "added": added,
        "removed": removed,
        "modified": modified,
    }



def _load_repository_manifest(
    manifest: str | Path | dict[str, object] | None,
) -> dict[str, object] | None:
    """Load a repository manifest supplied as a path or mapping.

    Parameters
    ----------
    manifest: str | pathlib.Path | dict[str, object] | None
        Saved manifest mapping or JSON file. A missing path is treated as no
        saved manifest, which means every currently selected file is new.

    Returns
    -------
    dict[str, object] | None
        Loaded manifest, or ``None`` when no saved manifest exists.

    Raises
    ------
    ValueError
        If the argument is neither a mapping nor a usable path, an existing
        path is not a file, or decoded JSON is not an object.
    json.JSONDecodeError
        If an existing file contains invalid JSON.
    OSError
        If the manifest file cannot be opened or read.

    Notes
    -----
    A dictionary is returned as supplied without copying or validating its
    manifest records. Relative paths use the current working directory. A
    nonexistent path and None both return None.

    """

    if manifest is None:
        return None
    if isinstance(manifest, dict):
        return manifest
    if not isinstance(manifest, (str, Path)) or not str(manifest).strip():
        raise ValueError("manifest must be a manifest mapping or JSON path.")

    path = Path(manifest).expanduser()
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"Manifest path is not a file: {path}")

    with path.open("r", encoding="utf-8") as stream:
        loaded = json.load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("Repository manifest JSON must contain an object.")
    return loaded


def detect_repository_changes(
    repository: str | Path,
    saved_manifest: str | Path | dict[str, object] | None,
    *,
    extensions: list[str] | tuple[str, ...] = DEFAULT_EXTENSIONS,
    include_paths: list[str] | tuple[str, ...] = (),
    exclude_paths: list[str] | tuple[str, ...] = (),
    max_file_bytes: int = 1_048_576,
    max_entries: int = 10_000,
) -> dict[str, object]:
    """Detect added, removed, or modified selected repository files.

    This is the standalone convenience function for checking whether a
    repository still matches a previously saved source manifest. It does not
    modify the repository or the saved manifest.

    Parameters
    ----------
    repository: str | pathlib.Path
        Local source directory. Absolute paths are used directly; relative
        paths are resolved against repository.PROJECT_ROOT, the parent of the
        installed package directory.
    saved_manifest: str | pathlib.Path | dict[str, object] | None
        Previously saved manifest mapping or JSON file path, or None. A missing
        path means no baseline, not an empty versioned manifest. Relative
        manifest paths use the process working directory.
    extensions: list[str] | tuple[str, ...], optional
        Literal filename suffixes with optional leading periods. They are
        normalized to lowercase and matched against lowercase names. An empty
        sequence selects no files. Defaults to ``DEFAULT_EXTENSIONS``.
    include_paths: list[str] | tuple[str, ...], optional
        Relative repository paths limiting the selection. An empty sequence
        permits all otherwise eligible paths. No glob patterns are accepted.
        Defaults to ``()``.
    exclude_paths: list[str] | tuple[str, ...], optional
        Relative repository paths excluded before inclusion checks. Excluded
        directories are not traversed. Defaults to ``()``.
    max_file_bytes: int, optional
        Maximum selected-file size in bytes; an exact positive integer. Files
        exactly at the limit are allowed. Defaults to ``1048576``.
    max_entries: int, optional
        Maximum encountered directory entries, including skipped files and
        directories. Exceeding the bound aborts without returning a partial
        report. Defaults to ``10000``.

    Returns
    -------
    dict[str, object]
        ``matches`` plus ``added``, ``removed``, and ``modified`` paths.

    Raises
    ------
    ValueError
        If scan settings or manifest records are invalid, or the entry limit is
        exceeded.
    OSError
        If required roots, selected files, or manifest files cannot be
        accessed.
    json.JSONDecodeError
        If the saved manifest file is not valid JSON.

    Notes
    -----
    The wrapper scans and hashes all currently selected files. An incomplete
    scan raises ValueError rather than reporting inaccessible files as removed.
    Use the same selection settings as the saved baseline to compare the same
    scope. None or a missing manifest always means matches=False.

    Examples
    --------
    >>> import qachat
    >>> changes = qachat.detect_repository_changes(
    ...     "/path/to/repository", "/path/to/source_manifest.json",
    ... )

    """

    report = scan_repository(
        repository,
        extensions=extensions,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        max_file_bytes=max_file_bytes,
        max_entries=max_entries,
    )
    current = build_repository_manifest(report)
    saved = _load_repository_manifest(saved_manifest)
    return compare_repository_manifests(saved, current)


def repository_has_changed(
    repository: str | Path,
    saved_manifest: str | Path | dict[str, object] | None,
    *,
    extensions: list[str] | tuple[str, ...] = DEFAULT_EXTENSIONS,
    include_paths: list[str] | tuple[str, ...] = (),
    exclude_paths: list[str] | tuple[str, ...] = (),
    max_file_bytes: int = 1_048_576,
    max_entries: int = 10_000,
) -> bool:
    """Return whether selected repository content has changed.

    This Boolean convenience wrapper rescans the repository through
    ``detect_repository_changes`` and negates the resulting ``matches`` value.

    Parameters
    ----------
    repository: str | pathlib.Path
        Local source directory. Absolute paths are used directly; relative
        paths are resolved against repository.PROJECT_ROOT, the parent of the
        installed package directory.
    saved_manifest: str | pathlib.Path | dict[str, object] | None
        Previously saved manifest mapping or JSON file path, or None. A missing
        path means no baseline, not an empty versioned manifest. Relative
        manifest paths use the process working directory.
    extensions: list[str] | tuple[str, ...], optional
        Literal filename suffixes with optional leading periods. They are
        normalized to lowercase and matched against lowercase names. An empty
        sequence selects no files. Defaults to ``DEFAULT_EXTENSIONS``.
    include_paths: list[str] | tuple[str, ...], optional
        Relative repository paths limiting the selection. An empty sequence
        permits all otherwise eligible paths. No glob patterns are accepted.
        Defaults to ``()``.
    exclude_paths: list[str] | tuple[str, ...], optional
        Relative repository paths excluded before inclusion checks. Excluded
        directories are not traversed. Defaults to ``()``.
    max_file_bytes: int, optional
        Maximum selected-file size in bytes; an exact positive integer. Files
        exactly at the limit are allowed. Defaults to ``1048576``.
    max_entries: int, optional
        Maximum encountered directory entries, including skipped files and
        directories. Exceeding the bound aborts without returning a partial
        report. Defaults to ``10000``.

    Returns
    -------
    bool
        ``True`` when selected repository content differs from the saved
        manifest, otherwise ``False``.

    Raises
    ------
    ValueError
        If selection settings or manifest data are invalid.
    OSError
        If the scan, hashing, or manifest read cannot complete.
    json.JSONDecodeError
        If the saved manifest file contains invalid JSON.

    Notes
    -----
    This performs the complete scan-and-hash workflow, not a timestamp-only
    shortcut. An absent baseline returns True even for an empty current
    selection. Selection, manifest, and I/O exceptions propagate from
    detect_repository_changes.

    """

    changes = detect_repository_changes(
        repository,
        saved_manifest,
        extensions=extensions,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        max_file_bytes=max_file_bytes,
        max_entries=max_entries,
    )
    return not bool(changes["matches"])


def _manifest_files(
    manifest: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Validate and return file records from a repository manifest.

    Parameters
    ----------
    manifest: dict[str, object]
        Repository manifest to validate.

    Returns
    -------
    dict[str, dict[str, object]]
        Checked mapping from relative paths to size and SHA-256 records.

    Raises
    ------
    ValueError
        If the manifest version is unsupported or its file records, sizes, or
        hashes have an invalid structure.

    Notes
    -----
    File keys must be normalized relative POSIX paths; sizes must be exact
    nonnegative integers; hashes must be 64 hexadecimal characters. Hashes
    are normalized to lowercase. Extra record keys are ignored.

    """

    if type(manifest) is not dict or (
        type(manifest.get("format_version")) is not int
        or manifest["format_version"] != _MANIFEST_VERSION
    ):
        raise ValueError("Unsupported repository manifest version.")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Repository manifest is missing its file records.")

    checked: dict[str, dict[str, object]] = {}
    for path, record in files.items():
        if not isinstance(path, str) or not isinstance(record, dict):
            raise ValueError("Repository manifest contains invalid records.")
        if (not path or "\\" in path or path.startswith("/")
                or any(part in ("", ".", "..") for part in path.split("/"))
                or PureWindowsPath(path).drive):
            raise ValueError("Repository manifest contains invalid paths.")
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if (type(size) is not int or size < 0
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None):
            raise ValueError("Repository manifest contains invalid hashes.")
        checked[path] = {
            "size_bytes": size,
            "sha256": digest.lower(),
        }
    return checked
