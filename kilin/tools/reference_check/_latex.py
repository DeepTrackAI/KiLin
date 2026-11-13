"""Read the citations and bibliography of a LaTeX paper.

Private to ``reference_check``. A LaTeX paper cites with ``\\cite{key}``
and keeps its references in a ``thebibliography`` environment, a compiled
``.bbl`` file, or a ``.bib`` database, so the PDF-oriented reference list
and number search find nothing in it. This module turns a ``.tex`` paper
into what the checker already works with:

1. **Entries**: the references, numbered, with readable text for matching
   them to the project's files. ``\\bibitem`` entries keep the order LaTeX
   prints them in; ``.bib`` entries have inferred numbers in order of
   first citation, not necessarily the compiled bibliography's order.
   Only cited ``.bib`` entries are listed.
2. **Body**: the paper's passages with each citation command written as
   the numbers it prints ("[2, 5]") and the rest of the LaTeX reduced to
   readable text, line by line, so line numbers stay those of the file.

Everything is deterministic and model-free. Keys cited but missing from
the bibliography are reported, not numbered.

Key Features
------------

- **Three bibliography sources**

    ``thebibliography`` in the file, then the ``.bbl`` file beside it,
    then explicitly selected ``.bib`` files named by
    ``\\bibliography`` or ``\\addbibresource``. No file is
    discovered or opened outside the authorized source selection.

- **Citation commands**

    ``\\cite``, ``\\citep``, ``\\citet``, ``\\parencite``, ``\\textcite``,
    ``\\autocite`` and the other ``cite`` commands, with several keys and
    optional notes; ``\\nocite`` lists without citing.

Module Structure
----------------

- ``LatexReferences`` and ``latex_references``

    The entries, the readable body, and notes on the bibliography.

- ``parse_bib`` and ``bib_entry_text``

    A ``.bib`` database, and one entry as a printed reference.

Examples
--------
>>> from kilin.tools.reference_check._latex import (
...     bib_entry_text, parse_bib)
>>> entry = parse_bib("@article{k, author={Lee, A.}, title={T}, year=2020}")
>>> bib_entry_text(entry["k"])
'Lee, A. (2020). T.'

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
import posixpath
import re
from typing import Final

from kilin.chunking import Chunk
from kilin.tools._shared.references import ReferenceEntry

__all__ = [
    "LatexReferences",
    "bib_entry_text",
    "latex_references",
    "parse_bib",
]

# A citation command with its optional notes and its keys.
_CITE: Final[re.Pattern[str]] = re.compile(
    r"\\(?P<name>[A-Za-z]*[Cc]ite[A-Za-z]*)\*?"
    r"(?:\s*\[[^\]]*\]){0,2}\s*\{(?P<keys>[^{}]*)\}"
)
# Citation-like commands that do not cite in the text.
_NOT_CITING: Final[frozenset[str]] = frozenset({
    "nocite", "citeauthor", "citeyear", "citeyearpar",
})
_COMMENT: Final[re.Pattern[str]] = re.compile(r"(?<!\\)%.*$")
_INLINE_MATH: Final[re.Pattern[str]] = re.compile(
    r"\$([^$]*)\$|\\\((.*?)\\\)"
)
# A line that starts a section: its title becomes a sentence of its own.
_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^\s*\\(?:part|chapter|section|subsection|subsubsection|paragraph)"
    r"(?![A-Za-z])"
)
_REFERENCE: Final[re.Pattern[str]] = re.compile(
    r"\\(?:ref|eqref|autoref|cref|Cref|pageref)\*?\s*\{[^{}]*\}"
)
_DROPPED: Final[re.Pattern[str]] = re.compile(
    r"\\(?:label|begin|end|includegraphics|input|include|bibliography"
    r"|bibliographystyle|addbibresource|vspace|hspace)\*?"
    r"(?:\s*\[[^\]]*\])?\s*\{[^{}]*\}"
)
_WRAPPED: Final[re.Pattern[str]] = re.compile(
    r"\\[A-Za-z]+\*?\s*(?:\[[^\]]*\])?\{([^{}]*)\}"
)
_COMMAND: Final[re.Pattern[str]] = re.compile(r"\\[A-Za-z]+\*?")
# A command that writes a character with an accent ("\'e", "\"{o}",
# "\v{c}"); a letter accent is not the start of a longer command.
_ACCENT: Final[re.Pattern[str]] = re.compile(
    r"\\(?:[`'^\"~=.]|[uvHcdbtk](?![A-Za-z]))\s*\{?([A-Za-z])\}?"
)
_BIBITEM: Final[re.Pattern[str]] = re.compile(
    r"\\bibitem\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}"
)
_BIBLIOGRAPHY_BEGIN: Final[re.Pattern[str]] = re.compile(
    r"\\begin\s*\{thebibliography\}"
)
_BIBLIOGRAPHY_END: Final[re.Pattern[str]] = re.compile(
    r"\\end\s*\{thebibliography\}"
)
_BIB_FILES: Final[re.Pattern[str]] = re.compile(
    r"\\(?:bibliography|addbibresource)\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}"
)
_VERBATIM_BEGIN: Final[re.Pattern[str]] = re.compile(
    r"\\begin\s*\{(verbatim\*?|lstlisting|minted|comment|Verbatim)\}"
)
_BIB_HEAD: Final[re.Pattern[str]] = re.compile(r"@\s*(\w+)\s*([{(])")
_BIB_FIELD: Final[re.Pattern[str]] = re.compile(r"\s*,?\s*([\w-]+)\s*=\s*")
_BIB_BARE: Final[re.Pattern[str]] = re.compile(r"[^,#\s]+")
_BIB_JOIN: Final[re.Pattern[str]] = re.compile(r"\s*#\s*")
# Authors listed in a reference before "et al.".
_AUTHORS_SHOWN: Final[int] = 6
# Venue fields of a .bib entry, by preference.
_VENUES: Final[tuple[str, ...]] = (
    "journal", "journaltitle", "booktitle", "publisher", "school",
    "institution", "howpublished",
)


@dataclass(frozen=True, slots=True)
class LatexReferences:
    """A LaTeX paper's references, as the checker reads them.

    Attributes
    ----------
    entries: tuple[ReferenceEntry, ...]
        The numbered references, as printed text.
    body: tuple[Chunk, ...]
        The paper's passages as readable text, citations as numbers, the
        bibliography environment left out.
    keys: dict[int, str]
        The citation key of each reference number.
    source: str
        Where the references come from: ``thebibliography``, a ``.bbl`` or
        ``.bib`` file name, or ``""`` when none was found.
    undefined: tuple[str, ...]
        Keys cited in the text but not in the bibliography, in order.
    unavailable: tuple[str, ...]
        Bibliography declarations not resolved to an authorized file.

    """

    entries: tuple[ReferenceEntry, ...]
    body: tuple[Chunk, ...]
    keys: dict[int, str]
    source: str
    undefined: tuple[str, ...]
    unavailable: tuple[str, ...] = ()

    @property
    def note(self) -> str:
        """Return a note on where the references come from.

        Returns
        -------
        str
            One or two sentences for the report.

        """

        notes = []
        if not self.source:
            notes.append(
                "No bibliography found: no thebibliography environment, "
                "selected .bbl or .bib source file in the workspace."
            )
        elif self.source.endswith(".bib"):
            notes.append(
                f"References from {self.source}, numbered in order of first "
                "citation; entries never cited are not listed."
            )
        else:
            notes.append(f"References from {self.source}.")
        if self.unavailable:
            notes.append(
                "Bibliography files unavailable or ambiguous among selected "
                "sources: " + ", ".join(self.unavailable) + "."
            )
        if self.undefined:
            listed = ", ".join(f"`{key}`" for key in self.undefined)
            notes.append(f"Cited but not in the bibliography: {listed}.")
        return " ".join(notes)


def latex_references(
    passages: Sequence[Chunk],
    file: Path | None,
    *,
    paper: Path | None = None,
    bibliography_files: Mapping[Path, Path] | None = None,
) -> LatexReferences:
    """Read a LaTeX paper's references from authorized source files.

    Parameters
    ----------
    passages: Sequence[Chunk]
        The paper's indexed passages.
    file: pathlib.Path | None
        The citing ``.tex`` file, already authorized by the tool; None
        when the complete source text is unavailable.
    paper: pathlib.Path | None, optional
        Relative source path for the citing paper. When omitted, use the
        path of the first passage. Defaults to ``None``.
    bibliography_files: Mapping[pathlib.Path, pathlib.Path] | None, optional
        Explicitly selected and validated bibliography paths, mapped from
        source-relative names to authorized workspace files. No other
        bibliography file is opened. Defaults to ``None`` (no companions).

    Returns
    -------
    LatexReferences
        Entries, readable body, and notes. Missing or ambiguous selected
        bibliography files cannot silently resolve to unrelated files.

    """

    ordered = sorted(passages, key=lambda chunk: chunk.start_line)
    path = paper or (ordered[0].path if ordered else Path("paper.tex"))
    available = bibliography_files or {}
    source_text = _read(file) if file is not None else ""
    ignored = _verbatim_lines(source_text, ordered)
    if source_text:
        numbered = enumerate(source_text.splitlines(), 1)
    else:
        numbered = sorted({
            (chunk.start_line + offset, line)
            for chunk in ordered
            for offset, line in enumerate(chunk.text.splitlines())
        })
    uncommented = "\n".join(
        "" if number in ignored else _COMMENT.sub("", line)
        for number, line in numbered
    )
    cited = list(dict.fromkeys(
        key
        for match in _CITE.finditer(uncommented)
        if match["name"].lower() not in _NOT_CITING
        for key in _keys(match["keys"])
    ))

    items, source = _bibitems(uncommented), "thebibliography"
    if not items:
        bbl_path, _ = _companion(path.stem + ".bbl", path, available)
        if bbl_path is not None:
            items = _bibitems(_read(available[bbl_path]))
            source = bbl_path.as_posix()
    unavailable: tuple[str, ...] = ()
    if items:
        keys = {number: key for number, (key, _) in enumerate(items, 1)}
        entries = tuple(
            ReferenceEntry(number, text, True)
            for number, (_, text) in enumerate(items, 1)
        )
    else:
        database, source, unavailable = _bib_database(
            uncommented, path, available
        )
        listed = [key for key in cited if key in database]
        keys = dict(enumerate(listed, 1))
        entries = tuple(
            ReferenceEntry(number, bib_entry_text(database[key]), True)
            for number, key in keys.items()
        )
    numbers = {key: number for number, key in keys.items()}
    undefined = tuple(key for key in cited if key not in numbers)
    listed = _bibliography_lines(source_text, ordered) | ignored
    body = tuple(
        _readable_chunk(chunk, numbers, listed) for chunk in ordered
    )
    return LatexReferences(
        entries, body, keys, source if entries else "", undefined,
        unavailable,
    )


def _read(file: Path) -> str:
    """Return a text file's content, or ``""`` when it cannot be read.

    Parameters
    ----------
    file: pathlib.Path
        The file.

    Returns
    -------
    str
        Its text, undecodable bytes replaced.

    """

    try:
        return file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _keys(text: str) -> list[str]:
    """Return the keys of a citation command's argument.

    Parameters
    ----------
    text: str
        The argument, such as ``"smith20, lee21"``.

    Returns
    -------
    list[str]
        Nonempty keys, stripped.

    """

    return [key.strip() for key in text.split(",") if key.strip()]


def _bibitems(text: str) -> list[tuple[str, str]]:
    """Return the ``\\bibitem`` entries of a ``thebibliography``.

    Parameters
    ----------
    text: str
        LaTeX text without comments: a paper or a ``.bbl`` file.

    Returns
    -------
    list[tuple[str, str]]
        Each entry's key and readable text, in printed order; empty when
        there is no ``thebibliography`` environment.

    """

    begin = _BIBLIOGRAPHY_BEGIN.search(text)
    if begin is None:
        return []
    end = _BIBLIOGRAPHY_END.search(text, begin.end())
    block = text[begin.end():end.start() if end else len(text)]
    starts = list(_BIBITEM.finditer(block))
    items = []
    for index, match in enumerate(starts):
        stop = (
            starts[index + 1].start() if index + 1 < len(starts)
            else len(block)
        )
        words = " ".join(
            _readable_line(line, {})
            for line in block[match.end():stop].splitlines()
        )
        items.append((match.group(1).strip(), " ".join(words.split())))
    return items


def _companion(
    name: str,
    paper: Path,
    available: Mapping[Path, Path],
) -> tuple[Path | None, str]:
    """Find a selected companion, preferring the paper's own directory.

    Parameters
    ----------
    name: str
        A LaTeX bibliography argument or expected compiled bibliography.
    paper: pathlib.Path
        Source-relative path of the referring LaTeX document.
    available: Mapping[pathlib.Path, pathlib.Path]
        Selected bibliography files, validated by the calling tool.

    Returns
    -------
    tuple[pathlib.Path | None, str]
        Chosen source-relative path and an empty diagnostic, or ``None``
        and an explanation. Duplicate basenames without a local match
        remain ambiguous instead of selecting an unrelated bibliography.

    """

    raw = name.strip()
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        not raw or "\\" in raw or posix.is_absolute()
        or windows.is_absolute() or windows.drive
    ):
        return None, f"{name} (invalid path)"
    suffix = posix.suffix.lower()
    if suffix not in {".bib", ".bbl"}:
        posix = posix.with_name(posix.name + ".bib")
    relative = PurePosixPath(posixpath.normpath(
        (PurePosixPath(paper.parent.as_posix()) / posix).as_posix()
    ))
    if ".." in relative.parts or relative.is_absolute():
        return None, f"{name} (outside sources)"
    local = Path(relative.as_posix())
    if local in available:
        return local, ""
    if len(posix.parts) != 1:
        return None, name
    matches = [
        path for path in available
        if path.name.casefold() == posix.name.casefold()
    ]
    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        return None, f"{name} (ambiguous)"
    return None, name


def _bib_database(
    text: str,
    paper: Path,
    available: Mapping[Path, Path],
) -> tuple[dict[str, dict[str, str]], str, tuple[str, ...]]:
    """Read only selected ``.bib`` files named by a LaTeX paper.

    Parameters
    ----------
    text: str
        The paper's LaTeX without comments, setup included.
    paper: pathlib.Path
        Source-relative path of the citing paper.
    available: Mapping[pathlib.Path, pathlib.Path]
        Selected bibliography files with validated workspace paths.

    Returns
    -------
    tuple[dict[str, dict[str, str]], str, tuple[str, ...]]
        Entries, selected bibliography names, and unavailable/ambiguous
        declarations. Earlier selected entries win on duplicate keys.

    """

    database: dict[str, dict[str, str]] = {}
    found = []
    unavailable = []
    for match in _BIB_FILES.finditer(text):
        for name in _keys(match.group(1)):
            path, problem = _companion(name, paper, available)
            if path is None:
                unavailable.append(problem)
                continue
            content = _read(available[path])
            if not content:
                unavailable.append(name)
                continue
            found.append(path.as_posix())
            for key, fields in parse_bib(content).items():
                database.setdefault(key, fields)
    return database, ", ".join(found), tuple(unavailable)


def parse_bib(text: str) -> dict[str, dict[str, str]]:
    """Parse a BibTeX database.

    Parameters
    ----------
    text: str
        The ``.bib`` file's content.

    Returns
    -------
    dict[str, dict[str, str]]
        Each entry's fields by key, field names in lower case, values
        with their outer braces or quotes removed; ``type`` holds the
        entry type. ``@string`` abbreviations are expanded where they are
        used ("jn" for ``@string{jn = "Nature"}``); ``@comment`` and
        ``@preamble`` are skipped, and so are entries that do not close.

    """

    entries: dict[str, dict[str, str]] = {}
    strings: dict[str, str] = {}
    position = 0
    while True:
        at = text.find("@", position)
        if at < 0:
            return entries
        head = _BIB_HEAD.match(text, at)
        if head is None:
            position = at + 1
            continue
        kind = head.group(1).lower()
        closing = "}" if head.group(2) == "{" else ")"
        end = _closing(text, head.end(), head.group(2), closing)
        if end < 0:
            return entries
        position = end + 1
        if kind == "string":
            strings.update(_bib_fields(text[head.end():end], strings))
            continue
        if kind in ("comment", "preamble"):
            continue
        key, comma, rest = text[head.end():end].partition(",")
        if not comma or not key.strip():
            continue
        fields = _bib_fields(rest, strings)
        fields["type"] = kind
        entries.setdefault(key.strip(), fields)


def _closing(text: str, start: int, opening: str, closing: str) -> int:
    """Return the position of the bracket that closes an entry.

    Parameters
    ----------
    text: str
        The text.
    start: int
        Position just after the opening bracket.
    opening: str
        The opening bracket, ``{`` or ``(``.
    closing: str
        The closing bracket.

    Returns
    -------
    int
        The closing bracket's position, or -1 when it does not close.

    """

    depth = 1
    for position in range(start, len(text)):
        character = text[position]
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return position
    return -1


def _bib_fields(text: str, strings: dict[str, str]) -> dict[str, str]:
    """Return the fields of one ``.bib`` entry.

    Parameters
    ----------
    text: str
        The entry after its key: ``name = {value}, ...``.
    strings: dict[str, str]
        ``@string`` abbreviations by lower-case name.

    Returns
    -------
    dict[str, str]
        Values by lower-case field name; parts joined by ``#`` are
        concatenated, and a bare word that is an abbreviation is
        expanded.

    """

    fields: dict[str, str] = {}
    position = 0
    while True:
        match = _BIB_FIELD.match(text, position)
        if match is None:
            return fields
        position = match.end()
        parts = []
        while position < len(text):
            character = text[position]
            if character == "{":
                end = _closing(text, position + 1, "{", "}")
                end = len(text) if end < 0 else end
                parts.append(text[position + 1:end])
                position = end + 1
            elif character == '"':
                end = text.find('"', position + 1)
                end = len(text) if end < 0 else end
                parts.append(text[position + 1:end])
                position = end + 1
            else:
                bare = _BIB_BARE.match(text, position)
                if bare is None:
                    break
                word = bare.group(0)
                parts.append(strings.get(word.lower(), word))
                position = bare.end()
            joined = _BIB_JOIN.match(text, position)
            if joined is None:
                break
            position = joined.end()
        fields[match.group(1).lower()] = "".join(parts)


def _field(fields: dict[str, str], name: str) -> str:
    """Return one ``.bib`` field as readable text.

    Parameters
    ----------
    fields: dict[str, str]
        The entry's fields.
    name: str
        Field name.

    Returns
    -------
    str
        The field's readable text on one line, or ``""``.

    """

    return " ".join(_readable_line(fields.get(name, ""), {}).split())


def bib_entry_text(fields: dict[str, str]) -> str:
    """Return a ``.bib`` entry as a printed reference.

    Parameters
    ----------
    fields: dict[str, str]
        The entry's fields (``parse_bib``).

    Returns
    -------
    str
        "Authors (year). Title. Venue, volume, pages. doi:..." with the
        parts the entry has, LaTeX reduced to readable text, at most six
        authors before "et al.".

    """

    names = _field(fields, "author") or _field(fields, "editor")
    authors = [
        author.strip() for author in re.split(r"\s+and\s+", names)
        if author.strip()
    ]
    if len(authors) > _AUTHORS_SHOWN:
        authors = [*authors[:_AUTHORS_SHOWN], "et al."]
    year = _field(fields, "year") or _field(fields, "date")[:4]
    head = "; ".join(authors)
    if year:
        head = f"{head} ({year})" if head else f"({year})"
    venue = next(
        (_field(fields, name) for name in _VENUES if _field(fields, name)),
        "",
    )
    details = ", ".join(
        part
        for part in (venue, _field(fields, "volume"), _field(fields, "pages"))
        if part
    )
    parts = [
        part for part in (head, _field(fields, "title"), details) if part
    ]
    text = ". ".join(parts) + "." if parts else ""
    if _field(fields, "doi"):
        text += f" doi:{_field(fields, 'doi')}"
    elif _field(fields, "eprint"):
        text += f" arXiv:{_field(fields, 'eprint')}"
    return text.strip()


def _verbatim_lines(
    source_text: str, passages: Sequence[Chunk]
) -> frozenset[int]:
    """Locate LaTeX example lines whose citation commands are not prose.

    Parameters
    ----------
    source_text: str
        Complete selected document, or empty when unavailable.
    passages: Sequence[Chunk]
        Indexed lines used when the original document is not readable.

    Returns
    -------
    frozenset[int]
        One-based lines in verbatim-like environments, including their
        opening and closing commands. These cannot generate citations.

    """

    if source_text:
        numbered = list(enumerate(source_text.splitlines(), 1))
    else:
        numbered = sorted({
            (chunk.start_line + offset, line)
            for chunk in passages
            for offset, line in enumerate(chunk.text.splitlines())
        })
    inside: str | None = None
    ignored = set()
    for number, line in numbered:
        if inside is not None:
            ignored.add(number)
            if re.search(rf"\\end\s*\{{{re.escape(inside)}\}}", line):
                inside = None
            continue
        match = _VERBATIM_BEGIN.search(_COMMENT.sub("", line))
        if match is not None:
            ignored.add(number)
            name = match.group(1)
            if not re.search(
                rf"\\end\s*\{{{re.escape(name)}\}}", line
            ):
                inside = name
    return frozenset(ignored)


def _bibliography_lines(
    source_text: str, passages: Sequence[Chunk]
) -> frozenset[int]:
    """Return the line numbers of a ``thebibliography`` environment.

    Parameters
    ----------
    source_text: str
        The paper's file text, or ``""`` when it could not be read.
    passages: Sequence[Chunk]
        The paper's passages, used when the file text is missing.

    Returns
    -------
    frozenset[int]
        One-based lines from ``\\begin{thebibliography}`` to its end,
        whichever passages they fall in.

    """

    if source_text:
        numbered = list(enumerate(source_text.split("\n"), start=1))
    else:
        numbered = sorted({
            (chunk.start_line + offset, line)
            for chunk in passages
            for offset, line in enumerate(chunk.text.split("\n"))
        })
    inside = False
    lines = set()
    for number, line in numbered:
        line = _COMMENT.sub("", line)
        if _BIBLIOGRAPHY_BEGIN.search(line):
            inside = True
        if inside:
            lines.add(number)
            if _BIBLIOGRAPHY_END.search(line):
                inside = False
    return frozenset(lines)


def _readable_chunk(
    chunk: Chunk, numbers: dict[str, int], skipped: frozenset[int]
) -> Chunk:
    """Return a passage as readable text, citations as numbers.

    Parameters
    ----------
    chunk: Chunk
        A LaTeX passage.
    numbers: dict[str, int]
        Reference number by citation key.
    skipped: frozenset[int]
        One-based lines to leave blank: the bibliography.

    Returns
    -------
    Chunk
        The same passage, each line reduced to readable text, so line
        numbers are kept.

    """

    readable = [
        "" if chunk.start_line + offset in skipped
        else _readable_line(line, numbers)
        for offset, line in enumerate(chunk.text.split("\n"))
    ]
    return replace(chunk, text="\n".join(readable))


def _citation_numbers(match: re.Match[str], numbers: dict[str, int]) -> str:
    """Return a citation command as the numbers it prints.

    Parameters
    ----------
    match: re.Match[str]
        The citation command (``_CITE``).
    numbers: dict[str, int]
        Reference number by citation key.

    Returns
    -------
    str
        "[2, 5]"; "[?]" when no key has a number; ``""`` for ``\\nocite``.

    """

    if match["name"].lower() in _NOT_CITING:
        if match["name"].lower() == "nocite":
            return ""
        return ", ".join(_keys(match["keys"]))
    found = [
        str(numbers[key]) for key in _keys(match["keys"]) if key in numbers
    ]
    return f"[{', '.join(found)}]" if found else "[?]"


def _readable_line(line: str, numbers: dict[str, int]) -> str:
    """Return one line of LaTeX as readable text.

    Parameters
    ----------
    line: str
        One line.
    numbers: dict[str, int]
        Reference number by citation key.

    Returns
    -------
    str
        The line without its comment, citations as "[2, 5]"
        (``_citation_numbers``), math kept without brackets (so "[0, 1]"
        is not a citation), section titles ending with a full stop so
        they do not join the next sentence, cross-references as "(ref)",
        formatting
        commands reduced to their words, other commands and braces
        removed.

    """

    text = _COMMENT.sub("", line)
    text = _CITE.sub(lambda match: _citation_numbers(match, numbers), text)
    text = _INLINE_MATH.sub(
        lambda match: (match.group(1) or match.group(2) or "")
        .replace("[", "(").replace("]", ")"),
        text,
    )
    heading = _HEADING.match(text) is not None
    text = _REFERENCE.sub("(ref)", text)
    text = _DROPPED.sub("", text)
    text = _ACCENT.sub(r"\1", text)
    previous = None
    while previous != text:
        previous, text = text, _WRAPPED.sub(r"\1", text)
    text = text.replace("\\%", "%").replace("\\&", "&").replace("\\\\", " ")
    text = _COMMAND.sub("", text)
    text = text.replace("{", "").replace("}", "").replace("~", " ")
    if heading and text.strip() and text.rstrip()[-1] not in ".?!:":
        text = text.rstrip() + "."
    return text
