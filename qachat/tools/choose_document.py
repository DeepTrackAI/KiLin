"""Choose the source document a request is about, or ask the user.

``document_scope`` serves tools that may work on the whole project or on
one document (listing mentions, finding passages): it returns the one
document a request is limited to, or none. ``choose_document`` serves tools
that always need one document (a briefing, reading a paper in full).

Tools that work on one document (reading a whole paper, turning a paper
into a briefing) must not guess which one is meant. ``choose_document``
uses, in order:

1. a file the question names, by path, name, stem, or author ("Smith's
   paper", "smith20" for ``smith2020.pdf``);
2. the earlier turns the question points back to, newest first: a file
   named in that turn's question, or the file its evidence comes from
   (all of it, or at least three passages and 60% of them). The router model
   confirms that the request means that same file; when it asks for a
   different one ("another paper", "a different study", "the next one"),
   the files of the earlier turns are left out;
3. otherwise the user is asked, with up to four candidates: the files of
   the earlier turns and the best matches for the question. When the user
   cancels, the answer asks them to name the file. Where nobody can be
   asked (notebooks), the best match is used and marked as a guess.

``document_scope`` uses the same first two steps, but the router model
chooses between the whole project (ALL), the earlier file (SAME), and a
different file (DIFFERENT); anything but a clear SAME means the whole
project, and it never asks the user.

The router model only confirms or rejects a file found by these rules; it
never names a file itself. When the router call fails, ``choose_document``
keeps the earlier file and ``document_scope`` searches the whole project.

Key Features
------------

- **Named files first**

    A file the question names settles the choice without a model call.

- **Earlier turns**

    "That paper" means the file an earlier turn named or clearly
    discussed, when the router model agrees.

- **Asking instead of guessing**

    When the file is unclear, the user picks from a short list; only
    callers that cannot ask receive a marked guess.

Module Structure
----------------

- ``choose_document``

    Choose one indexed source file for a request.

- ``earlier_file``

    Return the file the newest earlier turn names or clearly discusses.

- ``means_same_file``

    Ask the router model whether a request means an earlier file.

- ``document_scope``

    Return the documents a request is limited to, if any.

Examples
--------
>>> from qachat.tools.choose_document import choose_document
>>> chosen = choose_document(
...     request,
...     context,
...     question="Which paper should I turn into a research briefing?",
...     example="turn smith2020.pdf into a research briefing",
...     prefer=(".pdf",),
... )
>>> if isinstance(chosen, str):
...     answer = chosen
... else:
...     path, reason, guessed = chosen

``request`` and ``context`` are the ``ToolRequest`` and ``ToolContext`` a
tool's ``run`` receives.

"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Final

from qachat.retrievals.document_reading import dominant_path, named_paths
from qachat.tools.contract import ToolContext, ToolRequest
from qachat.tools.planning import ask_choice

__all__ = [
    "choose_document",
    "document_scope",
    "earlier_file",
    "means_same_file",
]

# A file "dominates" an earlier turn's evidence with at least this many
# passages and this share of them.
_DOMINANT_PASSAGES: Final[int] = 3
_DOMINANT_SHARE: Final[float] = 0.6
# Most files offered when the user is asked to choose.
_MAX_CHOICES: Final[int] = 4
# Reason for a file the question names; options with it are never left out.
_NAMED_IN_QUESTION: Final[str] = "named in the question"
# Router reply words. The prompts below ask for exactly these words, so
# change both together.
_REPLY_ALL: Final[str] = "ALL"
_REPLY_SAME: Final[str] = "SAME"
_REPLY_DIFFERENT: Final[str] = "DIFFERENT"
# Longest router reply; one word is expected.
_ROUTER_REPLY_TOKENS: Final[int] = 8


def choose_document(
    request: ToolRequest,
    context: ToolContext,
    *,
    question: str,
    example: str,
    prefer: tuple[str, ...] = (),
    avoid: frozenset[Path] = frozenset(),
) -> tuple[Path, str, bool] | str:
    """Choose one indexed source file for a request.

    Parameters
    ----------
    request: ToolRequest
        The question, with the earlier turns it points back to.
    context: ToolContext
        QAChat's services.
    question: str
        What to ask when the file is unclear, such as "Which paper should
        I turn into a research briefing?".
    example: str
        A request that names a file, for the answer after a cancel, such as
        "turn smith2020.pdf into a research briefing".
    prefer: tuple[str, ...], optional
        File suffixes offered first, such as ``(".pdf",)``. Defaults to
        ``()``.
    avoid: frozenset[pathlib.Path], optional
        Files not to choose or offer unless the question names them, such
        as the papers already discussed when the user asks for "another
        paper". Defaults to an empty set.

    Returns
    -------
    tuple[pathlib.Path, str, bool] | str
        The file, why it was chosen, and whether it is a guess (only where
        nobody can be asked); or a message to answer with when no file
        fits or the user cancelled.

    """

    sources = list(dict.fromkeys(chunk.path for chunk in context.chunks))
    known = set(sources)
    hits = named_paths(request.question, sources, authors=True)
    if len(hits) == 1:
        return hits[0], _NAMED_IN_QUESTION, False

    candidates: list[tuple[Path, str]] = [
        (path, _NAMED_IN_QUESTION) for path in hits
    ]
    if not hits:
        earlier = earlier_file(request, sources, known)
        if earlier is not None and earlier[0] not in avoid:
            if means_same_file(context, request.question, earlier[0]):
                return earlier[0], earlier[1], False
            # A different file: leave out every file of the earlier turns.
            avoid = avoid | {
                path for turn in request.earlier for path in turn.paths
            }

        for turn in request.earlier:
            counts = Counter(path for path in turn.paths if path in known)
            candidates += [
                (path, f"used in turn {turn.number}")
                for path, _ in counts.most_common(2)
            ]

        ranked = context.rank(request.text) if context.chunks else []
        top = ranked[: context.retrieval_count]
        preferred = [
            path for path in sources if path.suffix.lower() in prefer
        ]
        best = dominant_path(top, preferred or None) or dominant_path(top)
        if best is not None:
            candidates.append((best, "best match for the question"))
        matches = [result.chunk.path for result in top]
        matches.sort(key=lambda path: path.suffix.lower() not in prefer)
        candidates += [(path, "matches the question") for path in matches]

    options: list[tuple[Path, str]] = []
    for path, why in candidates:
        if path in avoid and why != _NAMED_IN_QUESTION:
            continue
        if path not in {chosen for chosen, _ in options}:
            options.append((path, why))

    if len(options) < _MAX_CHOICES and not hits:
        # Fill the list with other files of the preferred types, so the
        # user always has something to pick from.
        listed = {path for path, _ in options}
        options += [
            (path, "in this project")
            for path in sorted(
                sources, key=lambda source: source.name.casefold()
            )
            if path not in listed
            and path not in avoid
            and (not prefer or path.suffix.lower() in prefer)
        ]

    options = options[:_MAX_CHOICES]
    names = ", ".join(path.name for path, _ in options)
    if not options:
        return (
            f"I couldn't tell which file you mean. Name it, for example: "
            f"{example}."
        )

    if context.choose is not None:
        index = ask_choice(
            question,
            tuple(f"{path.name} ({why})" for path, why in options),
            context,
        )
        if index is None:
            return (
                f"No file was chosen, so I stopped. Tell me which one you "
                f"mean, for example: {example}. Files that match your "
                f"question: {names}."
            )
        path, why = options[index]
        return path, f"chosen by you ({why})", False

    path, why = options[0]
    return path, why, True


def earlier_file(
    request: ToolRequest, sources: list[Path], known: set[Path]
) -> tuple[Path, str] | None:
    """Return the file the newest earlier turn names or clearly discusses.

    Parameters
    ----------
    request: ToolRequest
        The question, with the earlier turns it points back to.
    sources: list[pathlib.Path]
        Indexed files.
    known: set[pathlib.Path]
        The same files, as a set.

    Returns
    -------
    tuple[pathlib.Path, str] | None
        The file and why, or None when no earlier turn settles one.

    """

    for turn in request.earlier:
        named = named_paths(turn.question, sources, authors=True)
        if len(named) == 1:
            return named[0], f"named in turn {turn.number}"

        counts = Counter(path for path in turn.paths if path in known)
        if not counts:
            continue
        path, count = counts.most_common(1)[0]
        # All of a turn's evidence from one file, or most of plenty of it.
        if len(counts) == 1 or (
            count >= _DOMINANT_PASSAGES
            and count >= _DOMINANT_SHARE * sum(counts.values())
        ):
            return path, f"the file discussed in turn {turn.number}"

    return None


def means_same_file(
    context: ToolContext, question: str, path: Path
) -> bool:
    """Ask the router model whether a request means an earlier file.

    Parameters
    ----------
    context: ToolContext
        QAChat's services (``ask_router``).
    question: str
        The request.
    path: pathlib.Path
        The file the conversation was about.

    Returns
    -------
    bool
        False when the request asks for a different file ("another
        paper", "a different study"); a failed or unclear reply keeps the
        earlier file.

    """

    system = (
        "You decide which document a request is about. You reply with "
        "SAME or DIFFERENT only."
    )
    prompt = f"""The conversation was just about {path.name}.

New request: {question}

Does the new request mean {path.name} (SAME), or ask for a different
document (DIFFERENT)?"""
    try:
        reply = context.ask_router(system, prompt, _ROUTER_REPLY_TOKENS)
    except Exception:  # Keep the earlier file when the router fails.
        return True

    return _REPLY_DIFFERENT not in reply.upper()


def document_scope(
    request: ToolRequest,
    ask_router: Callable[[str, str, int], str],
    sources: list[Path],
) -> tuple[list[Path], str | None]:
    """Return the documents a request is limited to, if any.

    Parameters
    ----------
    request: ToolRequest
        The question, with the earlier turns it points back to.
    ask_router: Callable[[str, str, int], str]
        Asks the router model (``ToolContext.ask_router``); the answering
        session passes its own, so answers and tools decide alike.
    sources: list[pathlib.Path]
        Indexed files.

    Returns
    -------
    tuple[list[pathlib.Path], str | None]
        The files the question names; otherwise the file the earlier turns
        were about, when the router model judges that the request means it
        ("go through that paper", "in this paper", "does it say MRI?");
        otherwise no files (the whole project). The second value says why.

    """

    named = named_paths(request.question, sources, authors=True)
    if named:
        return named, _NAMED_IN_QUESTION

    earlier = earlier_file(request, sources, set(sources))
    if earlier is None:
        return [], None

    path, why = earlier
    system = (
        "You decide which documents a request is about. You reply with "
        "ALL, SAME, or DIFFERENT only."
    )
    prompt = f"""The conversation was just about {path.name}.

New request: {request.question}

Is the request about all documents in the project (ALL), about
{path.name} (SAME), or about a different particular document
(DIFFERENT)? Answer ALL when it asks about the project, all papers, or
does not point at one document; SAME only when it points at that document,
for example with "that paper", "this paper", or "it"."""
    try:
        reply = ask_router(system, prompt, _ROUTER_REPLY_TOKENS).upper()
    except Exception:  # Without an answer, search the whole project.
        return [], None

    if _REPLY_SAME in reply and _REPLY_ALL not in reply:
        return [path], why

    return [], None
