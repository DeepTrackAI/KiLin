"""Preview local file selection using names and metadata only. Never read code."""

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
    "models", "weights", "venv", "env", "node_modules", "build", "dist",
    "__pycache__", "logs", "htmlcov",
}
SKIP_FILES = {"credentials.json", "secrets.json", "id_rsa", "id_ed25519"}
BLOCKED_SUFFIXES = {
    ".safetensors", ".gguf", ".bin", ".onnx", ".pt", ".pth", ".ckpt",
    ".incomplete", ".pem", ".key", ".p12", ".pfx", ".keystore",
}


def _paths(values, label):
    """Selection paths are literal paths inside the chosen folder, not globs."""
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} must be a list, for example ['app_core'].")
    result = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} needs non-empty relative paths.")
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts
                or any(character in normalized for character in "*?[]:\0")):
            raise ValueError(f"{label}: use a relative path inside the repository, without wildcards or '..'.")
        result.append(path.as_posix())
    return tuple(dict.fromkeys(result))


def _within(path, parent):
    """A folder selects its descendants, but src does not match src_extra."""
    return parent == "." or path == parent or path.startswith(parent + "/")


def _skip_reason(relative, kind, size, extensions, includes, excludes, max_file_bytes):
    name = PurePosixPath(relative).name.lower()
    if kind == "link":
        return "links are not followed"
    if name.startswith("."):
        return "hidden entry"
    if kind == "folder" and name in SKIP_DIRECTORIES:
        return "model, environment, or generated folder"
    if name in SKIP_FILES or PurePosixPath(name).suffix in BLOCKED_SUFFIXES:
        return "model weights or potentially secret file"
    if any(_within(relative, excluded) for excluded in excludes):
        return "excluded by your selection"
    in_scope = not includes or any(_within(relative, included) for included in includes)
    # Walk parent folders only when needed to reach an explicitly included path.
    leads_to_scope = kind == "folder" and any(_within(included, relative) for included in includes)
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
    if size > max_file_bytes:
        return "larger than the file-size limit"
    return None


def scan_repository(
    repository: str | Path,
    *,
    extensions=DEFAULT_EXTENSIONS,
    include_paths=(),
    exclude_paths=(),
    max_file_bytes: int = 1_048_576,
    max_entries: int = 10_000,
) -> dict[str, object]:
    """Return selected file paths and explained skips. Create or change nothing.

    Relative repository paths start at QAChat. Include/exclude paths start at
    the chosen repository. Exclusions win. Skipped folders are not entered.
    This is a metadata preview, not a Git-ignore parser or a text/security check.
    """

    if not isinstance(repository, (str, Path)) or not str(repository).strip():
        raise ValueError("Choose a local project folder.")

    if str(repository).lower().startswith(("https:", "http:", "git@")):
        raise ValueError("Choose a local folder. GitHub downloading comes later.")

    for value, label in ((max_file_bytes, "max_file_bytes"), (max_entries, "max_entries")):
        if type(value) is not int or value < 1:
            raise ValueError(f"{label} must be a positive whole number.")

    if not isinstance(extensions, (list, tuple)):
        raise ValueError("extensions must be a list, for example ['.py', '.md'].")

    normalized_extensions = []
    for extension in extensions:
        if not isinstance(extension, str):
            raise ValueError("Use file extensions such as '.py' or '.md'.")
        extension = "." + extension.lower().lstrip(".")
        if not re.fullmatch(r"\.[a-z0-9_]+(?:\.[a-z0-9_]+)*", extension):
            raise ValueError("Use file extensions such as '.py' or '.md', without wildcards.")
        normalized_extensions.append(extension)
    extensions = tuple(dict.fromkeys(normalized_extensions))
    includes = _paths(include_paths, "include_paths")
    excludes = _paths(exclude_paths, "exclude_paths")
    chosen = Path(repository).expanduser()
    root = (chosen if chosen.is_absolute() else PROJECT_ROOT / chosen).resolve(strict=True)

    if not root.is_dir():
        raise NotADirectoryError(f"Choose a folder, not a file: {root}")

    if root == Path(root.anchor) or root == Path.home().resolve():
        raise ValueError("Choose a project folder, not your whole home folder or drive.")

    rows = []
    pending = [root]
    visited = 0
    complete = True
    while pending:
        folder = pending.pop()
        relative_folder = folder.relative_to(root).as_posix()
        try:
            # Check again before entering: a folder may have changed since listing.
            if folder.is_symlink() or folder.is_junction() or not folder.resolve().is_relative_to(root):
                rows.append({"path": relative_folder, "kind": "link", "size_bytes": None,
                             "status": "skipped", "reason": "links are not followed"})
                continue
            with os.scandir(folder) as stream:
                children = []
                for entry in stream:
                    visited += 1
                    if visited > max_entries:
                        raise ValueError("Scan limit reached. Choose a smaller folder or raise max_entries.")
                    children.append(entry)
        except OSError as error:
            if folder == root:
                raise OSError(f"Cannot list the chosen folder: {root}") from error
            complete = False
            rows.append({"path": relative_folder, "kind": "folder", "size_bytes": None,
                         "status": "skipped", "reason": f"cannot list folder ({type(error).__name__})"})
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
                    relative, kind, size, extensions, includes, excludes, max_file_bytes,
                )
            except OSError as error:
                complete = False
                kind, size, reason = "unknown", None, f"cannot inspect entry ({type(error).__name__})"
            if kind == "folder" and reason is None:
                pending.append(path)
            else:
                rows.append({"path": relative, "kind": kind, "size_bytes": size,
                             "status": "skipped" if reason else "selected",
                             "reason": reason or "matches your selection"})

    rows.sort(key=lambda row: row["path"])
    selected = [row for row in rows if row["status"] == "selected"]
    return {
        "repository": str(root),
        "extensions": list(extensions),
        "include_paths": list(includes),
        "exclude_paths": list(excludes),
        "max_file_bytes": max_file_bytes,
        "visited_entries": visited,
        "complete": complete,
        "selected_files": [row["path"] for row in selected],
        "selected_bytes": sum(row["size_bytes"] for row in selected),
        "entries": rows,
    }
