"""Preview local file selection using names and metadata only."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTENSIONS = (
    ".py",
    ".md",
    ".txt",
    ".rst",
    ".m",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".sh",
    ".ps1",
    ".ipynb",
)
SKIP_DIRECTORIES = {
    "models",
    "weights",
    "venv",
    "env",
    "node_modules",
    "build",
    "dist",
    "__pycache__",
    "logs",
    "htmlcov",
}
SKIP_FILES = {
    "credentials.json",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
}
BLOCKED_SUFFIXES = {
    ".safetensors",
    ".gguf",
    ".bin",
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
    ".incomplete",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".keystore",
}


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

    if name.startswith("."):
        return "hidden entry"

    if kind == "folder" and name in SKIP_DIRECTORIES:
        return "model, environment, or generated folder"

    if name in SKIP_FILES:
        return "model weights or potentially secret file"

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


def _normalize_extensions(
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

    """

    if not isinstance(extensions, (list, tuple)):
        raise ValueError("extensions must be a list, such as ['.py', '.md'].")

    normalized: list[str] = []
    for extension in extensions:

        if not isinstance(extension, str):
            raise ValueError("Use file extensions such as '.py' or '.md'.")

        value = "." + extension.lower().lstrip(".")

        if not re.fullmatch(r"\.[a-z0-9_]+(?:\.[a-z0-9_]+)*", value):
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
        Local project folder. A relative path starts at the QAChat root.
    extensions: list[str] | tuple[str, ...], optional
        File extensions to select.
    include_paths: list[str] | tuple[str, ...], optional
        Optional paths that limit the selection.
    exclude_paths: list[str] | tuple[str, ...], optional
        Optional paths removed from the selection. Exclusions win.
    max_file_bytes: int, optional
        Largest file allowed.
    max_entries: int, optional
        Maximum number of entries inspected.

    Returns
    -------
    dict[str, object]
        Selected paths, byte counts, skipped entries, and explanations.

    Raises
    ------
    FileNotFoundError
        If the repository does not exist.
    NotADirectoryError
        If the selected path is a file.
    ValueError
        If a setting is invalid or the entry limit is reached.

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

    selected_extensions = _normalize_extensions(extensions)
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
