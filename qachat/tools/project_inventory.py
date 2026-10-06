"""Tool ``project_inventory``: exact facts about the project's files.

Questions such as "how many papers are in this project?", "list the PDFs
in papers", or "which files could not be read?" are about the project, not
its content. Ranked retrieval cannot count, and a model guessing from a few
passages gets them wrong. This tool answers from the source manifest (every
selected file and its size) and the index (which files have searchable
text, and in how many parts), and Python writes the answer.

The question can narrow the listing: file types it names ("PDFs",
"markdown", "notebooks", ".py") and folders it names ("in papers"). Counts
are of files; the answer says so, because a folder of PDFs may hold
tutorials as well as papers.

Key Features
------------

- **Counts**

    Files by type and by top-level folder, with sizes.

- **Coverage**

    Which selected files have no searchable text.

- **Listing**

    Matching files with their parts and size, when the question asks for
    a list or narrows by type or folder.

Module Structure
----------------

- ``InventorySettings``
- ``parse_settings``
- ``run``
- ``file_types_in``

"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final

from qachat.tools.contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_int,
    reject_unknown,
)
from qachat.tools.scan_and_list import short_names

__all__ = [
    "InventorySettings",
    "file_types_in",
    "parse_settings",
    "run",
]

# Words that name file types, mapped to suffixes.
_TYPE_WORDS: Final[dict[str, tuple[str, ...]]] = {
    "pdf": (".pdf",),
    "pdfs": (".pdf",),
    "markdown": (".md",),
    "readme": (".md",),
    "readmes": (".md",),
    "txt": (".txt",),
    "notebook": (".ipynb",),
    "notebooks": (".ipynb",),
    "python": (".py",),
    "matlab": (".m",),
    "script": (".py", ".m"),
    "scripts": (".py", ".m"),
}
_TYPE_NAMES: Final[dict[str, str]] = {
    ".pdf": "PDF",
    ".md": "Markdown",
    ".txt": "text",
    ".ipynb": "notebook",
    ".py": "Python",
    ".m": "MATLAB",
}
# Questions about files that could not be read list only those files.
_UNREADABLE_WORDS = re.compile(
    r"\b(no (searchable )?text|not (been )?indexed|unindexed|failed|"
    r"could ?n[o']t be read|unreadable|missing text|empty)\b",
    flags=re.I,
)
_LIST_WORDS = re.compile(
    r"\b(list|which|what files|show|name|names|enumerate)\b", flags=re.I
)


@dataclass(frozen=True, slots=True)
class InventorySettings:
    """Settings of ``project_inventory`` in a project type.

    Attributes
    ----------
    max_listed: int
        Files listed by name (1-2000); counts include all.
    max_folders: int
        Folders shown in the per-folder counts (1-100).

    """

    max_listed: int = 200
    max_folders: int = 20


def parse_settings(data: dict, where: str) -> InventorySettings:
    """Validate ``project_inventory`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    InventorySettings
        Validated settings.

    Raises
    ------
    ValueError
        If a setting is unknown or out of range.

    """

    reject_unknown(where, data, ("max_listed", "max_folders"))
    defaults = InventorySettings()
    return InventorySettings(
        max_listed=check_int(
            where, data, "max_listed", defaults.max_listed, 1, 2000
        ),
        max_folders=check_int(
            where, data, "max_folders", defaults.max_folders, 1, 100
        ),
    )


def file_types_in(question: str) -> tuple[str, ...]:
    """Return the file suffixes a question names.

    Parameters
    ----------
    question: str
        The question.

    Returns
    -------
    tuple[str, ...]
        Suffixes such as ``(".pdf",)`` for "list the PDFs" or ".pdf"; empty
        when the question names no type.

    """

    found: list[str] = []
    text = question.casefold()
    # "text" alone is too common ("searchable text"); "text files" counts.
    if re.search(r"\btext files?\b", text):
        found.append(".txt")
    for word in re.findall(r"\.?[a-z]+", text):
        if word.startswith(".") and word in _TYPE_NAMES:
            found.append(word)
        for suffix in _TYPE_WORDS.get(word.lstrip("."), ()):
            found.append(suffix)
    return tuple(dict.fromkeys(found))


def run(
    settings: InventorySettings, request: ToolRequest, context: ToolContext
) -> ToolResult:
    """Answer a question about the project's files from exact counts.

    Parameters
    ----------
    settings: InventorySettings
        The intent's settings.
    request: ToolRequest
        The question.
    context: ToolContext
        QAChat's services (``source_files`` and ``chunks``).

    Returns
    -------
    ToolResult
        Counts, coverage, and (when asked or narrowed) a file list. A
        question about files that could not be read gets only those. No
        model is called and no evidence is cited.

    """

    context.progress("Counting the project's files")
    sizes = context.source_files()
    parts = Counter(chunk.path for chunk in context.chunks)
    files = sorted(set(sizes) | set(parts), key=lambda path: str(path))
    types = file_types_in(request.question)
    folders = _folders(files)
    named_folders = [
        folder
        for folder in folders
        if re.search(
            rf"(?<![\w/]){re.escape(folder.casefold())}(?![\w])",
            request.question.casefold(),
        )
    ]
    selected = [
        path
        for path in files
        if (not types or path.suffix.lower() in types)
        and (not named_folders or _folder(path) in named_folders)
    ]
    unsearchable = [path for path in files if not parts.get(path)]
    unreadable_question = bool(_UNREADABLE_WORDS.search(request.question))
    lines = [_headline(files, unsearchable)]
    lines += ["", "**By type**", ""]
    lines += [
        f"- {_type_name(suffix)}: {count}"
        for suffix, count in Counter(
            path.suffix.lower() for path in files
        ).most_common()
    ]
    folder_counts = Counter(_folder(path) for path in files)
    if len(folder_counts) > 1:
        lines += ["", "**By folder**", ""]
        for folder, count in folder_counts.most_common(settings.max_folders):
            kinds = Counter(
                _type_name(path.suffix.lower())
                for path in files
                if _folder(path) == folder
            )
            detail = ", ".join(f"{n} {kind}" for kind, n in kinds.items())
            lines.append(f"- {folder or '(top level)'}: {count} ({detail})")
        if len(folder_counts) > settings.max_folders:
            lines.append(
                f"- and {len(folder_counts) - settings.max_folders} more "
                "folders"
            )
    if unsearchable:
        names = short_names(files)
        lines += ["", "**No searchable text**", ""]
        lines += [f"- {names[path]}" for path in unsearchable]
    narrowed = bool(types or named_folders)
    if unreadable_question:
        # The "No searchable text" section above already lists them.
        selected = unsearchable
    elif narrowed or _LIST_WORDS.search(request.question):
        names = short_names(files)
        what = " ".join(
            filter(
                None,
                [
                    ", ".join(_type_name(suffix) for suffix in types),
                    "files",
                    (
                        "in " + ", ".join(named_folders)
                        if named_folders
                        else ""
                    ),
                ],
            )
        )
        lines += ["", f"**{len(selected)} {what}**", ""]
        for path in selected[: settings.max_listed]:
            size = _size(sizes.get(path))
            count = parts.get(path, 0)
            note = f"{count} parts" if count else "no searchable text"
            lines.append(f"- {names[path]} ({note}{size})")
        if len(selected) > settings.max_listed:
            lines.append(
                f"- and {len(selected) - settings.max_listed} more"
            )
    lines += [
        "",
        "Counts are of files QAChat selected for this project; a folder of "
        "PDFs can hold tutorials as well as papers.",
    ]
    summary: dict[str, object] = {
        "name": "project_inventory",
        "files": len(files),
        "searchable_files": len(files) - len(unsearchable),
        "unsearchable_files": [path.as_posix() for path in unsearchable],
        "types": dict(Counter(path.suffix.lower() for path in files)),
        "folders": dict(folder_counts),
        "filter_types": list(types),
        "filter_folders": named_folders,
        "selected": len(selected),
        "asked_about_unreadable": unreadable_question,
    }
    context.trace.emit(
        "tool.project_inventory.result",
        component="tool",
        data={key: value for key, value in summary.items() if key != "name"},
        turn_id=context.trace_turn_id,
    )
    context.done(f"Counted {len(files)} files", evidence_count=0)
    return ToolResult("\n".join(lines), (), summary)


def _headline(files: Sequence[Path], unsearchable: Sequence[Path]) -> str:
    total = len(files)
    noun = "file" if total == 1 else "files"
    line = f"This project has **{total} source {noun}**"
    if unsearchable:
        return (
            f"{line}; {total - len(unsearchable)} have searchable text and "
            f"{len(unsearchable)} do not."
        )
    return f"{line}, all with searchable text."


def _folder(path: Path) -> str:
    """Top-level folder below the source root, or "" for its files."""

    parts = path.parts
    return parts[1] if len(parts) > 2 else ""


def _folders(files: Sequence[Path]) -> list[str]:
    return sorted({_folder(path) for path in files} - {""})


def _type_name(suffix: str) -> str:
    return _TYPE_NAMES.get(suffix, suffix or "no extension")


def _size(size: int | None) -> str:
    if not size:
        return ""
    for unit, scale in (("GB", 2**30), ("MB", 2**20), ("kB", 2**10)):
        if size >= scale:
            return f", {size / scale:.1f} {unit}"
    return f", {size} B"
