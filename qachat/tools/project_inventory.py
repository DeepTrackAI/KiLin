"""Tool ``project_inventory``: exact facts about the project's files.

Questions such as "how many papers are in this project?", "list the PDFs
in papers", or "which files could not be read?" are about the project, not
its content. Ranked retrieval cannot count, and a model guessing from a few
passages gets them wrong. This tool answers from the source manifest (every
selected file and its size) and the index (which files have searchable
text, and in how many parts), and Python writes the answer. No model is
called.

The question can narrow the listing: file types it names ("PDFs",
"markdown", "notebooks", ".py") and folders it names ("in papers"). Counts
are of files; the answer says so, because a folder of PDFs may hold
tutorials as well as papers. Type and folder words are matched literally;
the tool does not understand other descriptions of files ("the long
ones", "last year's").

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

    Settings of ``project_inventory`` in a project type.

- ``parse_settings``

    Validate ``project_inventory`` settings from a project type.

- ``file_types_in``

    Return the file suffixes a question names.

- ``run``

    Answer a question about the project's files from exact counts.

Private helpers write the answer's sections and name types, folders, and
sizes; they are not part of the public API.

Examples
--------
>>> from qachat.tools.project_inventory import file_types_in
>>> file_types_in("list the PDFs and notebooks")
('.pdf', '.ipynb')

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
# How the answer names each suffix.
_TYPE_NAMES: Final[dict[str, str]] = {
    ".pdf": "PDF",
    ".md": "Markdown",
    ".txt": "text",
    ".ipynb": "notebook",
    ".py": "Python",
    ".m": "MATLAB",
}
# Questions about files that could not be read list only those files.
_UNREADABLE_WORDS: Final[re.Pattern[str]] = re.compile(
    r"\b(no (searchable )?text|not (been )?indexed|unindexed|failed|"
    r"could ?n[o']t be read|unreadable|missing text|empty)\b",
    flags=re.I,
)
# Questions with these words get a file list even when not narrowed.
_LIST_WORDS: Final[re.Pattern[str]] = re.compile(
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
    files = sorted(set(sizes) | set(parts), key=str)
    unsearchable = [path for path in files if not parts.get(path)]
    type_counts = Counter(path.suffix.lower() for path in files)
    folder_counts = Counter(_folder(path) for path in files)

    types = file_types_in(request.question)
    named_folders = _named_folders(request.question, _folders(files))
    unreadable_question = bool(_UNREADABLE_WORDS.search(request.question))
    if unreadable_question:
        # The "No searchable text" section already lists them.
        selected = unsearchable
    else:
        selected = [
            path
            for path in files
            if (not types or path.suffix.lower() in types)
            and (not named_folders or _folder(path) in named_folders)
        ]

    lines = [_headline(files, unsearchable)]
    lines += ["", "**By type**", ""]
    lines += [
        f"- {_type_name(suffix)}: {count}"
        for suffix, count in type_counts.most_common()
    ]
    if len(folder_counts) > 1:
        lines += _folder_lines(files, folder_counts, settings.max_folders)
    if unsearchable:
        names = short_names(files)
        lines += ["", "**No searchable text**", ""]
        lines += [f"- {names[path]}" for path in unsearchable]

    narrowed = bool(types or named_folders)
    if not unreadable_question and (
        narrowed or _LIST_WORDS.search(request.question)
    ):
        names = short_names(files)
        title = _selection_title(types, named_folders)
        lines += ["", f"**{len(selected)} {title}**", ""]
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
        "types": dict(type_counts),
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
    """Write the first line: how many files, and how many are searchable."""

    total = len(files)
    noun = "file" if total == 1 else "files"
    line = f"This project has **{total} source {noun}**"
    if unsearchable:
        return (
            f"{line}; {total - len(unsearchable)} have searchable text and "
            f"{len(unsearchable)} do not."
        )

    return f"{line}, all with searchable text."


def _folder_lines(
    files: Sequence[Path], folder_counts: Counter[str], max_folders: int
) -> list[str]:
    """Write the "By folder" section: the largest folders and their types.

    Parameters
    ----------
    files: Sequence[pathlib.Path]
        Every project file.
    folder_counts: Counter[str]
        Files per top-level folder ("" for files at the top level).
    max_folders: int
        Folders shown; the rest are counted in one closing line.

    Returns
    -------
    list[str]
        Markdown lines, starting with a blank line.

    """

    lines = ["", "**By folder**", ""]
    for folder, count in folder_counts.most_common(max_folders):
        kinds = Counter(
            _type_name(path.suffix.lower())
            for path in files
            if _folder(path) == folder
        )
        detail = ", ".join(f"{n} {kind}" for kind, n in kinds.items())
        lines.append(f"- {folder or '(top level)'}: {count} ({detail})")

    if len(folder_counts) > max_folders:
        lines.append(
            f"- and {len(folder_counts) - max_folders} more folders"
        )

    return lines


def _selection_title(
    types: Sequence[str], named_folders: Sequence[str]
) -> str:
    """Name the listed files, such as "PDF files in papers"."""

    words = [", ".join(_type_name(suffix) for suffix in types), "files"]
    if named_folders:
        words.append("in " + ", ".join(named_folders))

    return " ".join(word for word in words if word)


def _named_folders(question: str, folders: Sequence[str]) -> list[str]:
    """Return the folders whose names the question uses as whole words."""

    text = question.casefold()
    return [
        folder
        for folder in folders
        if re.search(
            rf"(?<![\w/]){re.escape(folder.casefold())}(?![\w])", text
        )
    ]


def _folder(path: Path) -> str:
    """Top-level folder below the source root, or "" for its files."""

    parts = path.parts
    return parts[1] if len(parts) > 2 else ""


def _folders(files: Sequence[Path]) -> list[str]:
    """Return the sorted top-level folders that hold any of the files."""

    return sorted({_folder(path) for path in files} - {""})


def _type_name(suffix: str) -> str:
    """Name a suffix for the answer, such as "PDF" for ``.pdf``."""

    return _TYPE_NAMES.get(suffix, suffix or "no extension")


def _size(size: int | None) -> str:
    """Write a file size such as ", 2.0 kB"; "" when unknown or zero."""

    if not size:
        return ""

    for unit, scale in (("GB", 2**30), ("MB", 2**20), ("kB", 2**10)):
        if size >= scale:
            return f", {size / scale:.1f} {unit}"

    return f", {size} B"
