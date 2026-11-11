"""Build source descriptions from the existing selected files and chunks.

Source descriptors identify files and the chunks already extracted from them.
They do not read documents, generate summaries, infer scientific-paper kinds,
or require that every source produce a text chunk. An image or certificate can
remain present in the source inventory without receiving a source card.

Key Features
------------

- **Format-neutral identity**

    Preserve source paths independently of their card eligibility.

- **Existing provenance**

    Collect the stable identifiers of chunks that already belong to a file.

- **Empty content**

    Keep selected sources even when no searchable text was extracted.

Module Structure
----------------

- ``describe_sources``

    Create deterministic source descriptors from paths, titles, and chunks.

Examples
--------
>>> from pathlib import Path
>>> from kilin.retrievals.source_cards import describe_sources
>>> sources = describe_sources((Path("image.png"),), ())
>>> sources[Path("image.png")].chunk_ids
()

"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from kilin.chunking import Chunk
from kilin.retrievals.source_cards.models import (
    SourceDescriptor,
    _check_descriptor,
    _check_path,
)

__all__ = ["describe_sources"]


def describe_sources(
    paths: Iterable[Path],
    chunks: Sequence[Chunk],
    *,
    titles: Mapping[Path, str] | None = None,
) -> dict[Path, SourceDescriptor]:
    """Describe selected sources using existing chunk identities.

    Parameters
    ----------
    paths: Iterable[pathlib.Path]
        Selected repository-relative source paths, including chunkless files.
    chunks: Sequence[Chunk]
        Already extracted chunks; each must belong to one selected file.
    titles: Mapping[pathlib.Path, str] | None, optional
        Known titles, when available; other titles use file-name stems.
        Defaults to ``None``.

    Returns
    -------
    dict[pathlib.Path, SourceDescriptor]
        Ordered descriptions keyed by source path, including empty sources.

    Raises
    ------
    ValueError
        If paths, titles, or chunk ownership are inconsistent or duplicated.

    Notes
    -----
    A suffix describes a file format; it is not evidence that the content
    qualifies for any registered card kind.

    """

    selected: dict[Path, list[str]] = {}
    for path in paths:
        _check_path(path)
        if path in selected:
            raise ValueError("Duplicate selected source path.")
        selected[path] = []
    known = {} if titles is None else dict(titles)
    if any(path not in selected for path in known):
        raise ValueError("A title refers to an unselected source.")
    identifiers: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, Chunk):
            raise ValueError("Expected already extracted Chunk records.")
        if chunk.path not in selected:
            raise ValueError("Chunk belongs to an unselected source.")
        if (
            not isinstance(chunk.identifier, str)
            or not chunk.identifier.strip()
            or chunk.identifier in identifiers
        ):
            raise ValueError("Chunk identifiers must be unique strings.")
        selected[chunk.path].append(chunk.identifier)
        identifiers.add(chunk.identifier)
    result = {}
    for path, chunk_ids in selected.items():
        title = known.get(path, path.stem)
        source = SourceDescriptor(
            path, title, path.suffix.lower(), tuple(chunk_ids)
        )
        _check_descriptor(source)
        result[path] = source
    return result
