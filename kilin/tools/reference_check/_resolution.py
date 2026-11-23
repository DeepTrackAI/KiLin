"""Resolve missing bibliography entries without implicit network access.

A reference check is local by default. ``offline_link`` makes a useful
Google Scholar URL from a title or journal coordinates (or a DOI printed
in the reference) without a request.
``resolve_links`` can consult Crossref and OpenAlex only when given a
permission created for one explicitly approved lookup operation. Its
results are hints for finding publications, not evidence of a legal right
to download a PDF. Downloading belongs to a separate tool operation.

Key Features
------------

- **Offline links**

    Missing entries have a title or journal-coordinate search link even
    without permission.

- **Explicit permission**

    The online resolver rejects calls without an operation-scoped grant.

- **Conservative matching**

    A known article title requires corroborating author, year and title.
    Title-less citations can instead match journal, volume and first page,
    alongside author and year. Ambiguous candidates remain search links.

Module Structure
----------------

- ``ReferenceLink``

    A missing entry's preferred link and optional online metadata.

- ``LookupPermission``

    Approval for one specific set of numbered reference entries.

- ``offline_link`` and ``resolve_links``

    Build a local link, or resolve approved entries against services.

Examples
--------
>>> from kilin.tools.reference_check._resolution import offline_link
>>> link = offline_link(1, "Smith J. Example article. Nature (2020).")
>>> link.kind
'search'

"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
import unicodedata
from typing import Final
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from kilin.tools.reference_check._matching import first_author

__all__ = [
    "LookupPermission",
    "ReferenceLink",
    "offline_link",
    "linkable_entry",
    "resolve_links",
]

_CROSSREF: Final[str] = "https://api.crossref.org"
_OPENALEX: Final[str] = "https://api.openalex.org"
_SEARCH: Final[str] = "https://scholar.google.com/scholar?q="
_SERVICES: Final[tuple[str, ...]] = (_CROSSREF, _OPENALEX)
_DOI: Final[re.Pattern[str]] = re.compile(
    r"\b(10\.\d{4,9}/[^\s<>\"'\[\]]+)", re.I
)
_YEAR: Final[re.Pattern[str]] = re.compile(r"\b(?:19|20)\d{2}\b")
_WORD: Final[re.Pattern[str]] = re.compile(r"[a-z]{4,}")
_COMMON: Final[frozenset[str]] = frozenset(
    {"journal", "volume", "press", "article", "research", "issue",
     "pages", "et al", "https", "doi", "the", "with", "from"}
)
_RESPONSE_LIMIT: Final[int] = 512 * 1024
_TIMEOUT: Final[float] = 5.0
_MAX_QUERY_LENGTH: Final[int] = 350
_MAX_SEARCH_TITLE: Final[int] = 220
# Bibliographic coordinates in journal-first or volume-first styles.
_PAGE: Final[str] = r"[A-Za-z]?\d{1,12}[A-Za-z0-9]*"
_COORDINATES_YEAR_FIRST: Final[re.Pattern[str]] = re.compile(
    r"(?P<year>(?:19|20)\d{2})[a-z]?\s*[,;]\s*"
    r"(?P<volume>\d{1,6})(?:\s*\(\s*(?P<issue>[^()]{1,12})"
    r"\s*\))?\s*[:,]\s*(?P<page>" + _PAGE + r")",
    re.I,
)
_COORDINATES_YEAR_LAST: Final[re.Pattern[str]] = re.compile(
    r"(?P<volume>\d{1,6})(?:\s*\(\s*(?P<issue>[^()]{1,12})"
    r"\s*\))?\s*[:,]\s*(?P<page>" + _PAGE + r")"
    r"(?:\s*[-–]\s*" + _PAGE + r")?\s*\(\s*"
    r"(?P<year>(?:19|20)\d{2})[a-z]?\s*\)",
    re.I,
)
_COORDINATES_VOLUME_PAGE: Final[re.Pattern[str]] = re.compile(
    r"(?P<volume>\d{1,6})(?:\s*\(\s*(?P<issue>[^()]{1,12})"
    r"\s*\))?\s*[:,]\s*(?P<page>" + _PAGE + r")",
    re.I,
)


@dataclass(frozen=True, slots=True)
class ReferenceLink:
    """A local or resolved link for one missing bibliography entry.

    Attributes
    ----------
    number: int
        Number printed or inferred in the bibliography.
    url: str
        Preferred HTTPS publication or search URL.
    kind: str
        ``search``, ``printed_doi``, or ``verified_doi``.
    doi: str
        DOI copied from the entry or verified against a service.
    pdf_url: str
        OpenAlex candidate PDF URL; **not** a download permission.
    license: str
        License metadata reported by OpenAlex, if present.
    version: str
        Document version reported by OpenAlex, if present.

    """

    number: int
    url: str
    kind: str = "search"
    doi: str = ""
    pdf_url: str = ""
    license: str = ""
    version: str = ""


@dataclass(frozen=True, slots=True)
class LookupPermission:
    """One explicit grant to look up a specific set of references.

    Attributes
    ----------
    numbers: tuple[int, ...]
        The reference numbers covered by the user's approval.
    services: tuple[str, ...]
        Exact service origins approved for this lookup.

    Notes
    -----
    Only the reference-checking operation should construct this object,
    after an affirmative user confirmation. It never authorizes downloads.

    """

    numbers: tuple[int, ...]
    services: tuple[str, ...] = _SERVICES


class _NoRedirect(HTTPRedirectHandler):
    """Do not silently follow a redirect to an unapproved host."""

    def redirect_request(self, request, response, code, message, headers,
                         new_url):
        """Reject redirects so service boundaries cannot change.

        Parameters
        ----------
        request: object
            Original request.
        response: object
            Response being redirected.
        code: int
            HTTP redirect status.
        message: str
            HTTP redirect message.
        headers: object
            Response headers.
        new_url: str
            Proposed redirect destination.

        Returns
        -------
        None
            No redirect is authorized.

        """

        return None


@dataclass(frozen=True, slots=True)
class _Coordinates:
    """Bibliographic fields identifying a publication without its title.

    Attributes
    ----------
    journal: str
        Journal name or abbreviation, as printed in the reference.
    year: str
        Publication year, if present.
    volume: str
        Journal volume.
    page: str
        First page or article identifier.
    issue: str
        Issue number, if printed.

    """

    journal: str
    year: str
    volume: str
    page: str
    issue: str = ""


def _journal_before(text: str) -> str:
    """Extract the journal before the numeric citation coordinates.

    Parameters
    ----------
    text: str
        Text preceding the year or volume in a journal citation.

    Returns
    -------
    str
        Journal segment, preserving abbreviated journal names.

    Notes
    -----
    Short, period-separated segments such as ``Phys. Rev. Lett.`` are
    kept together; a preceding author list or article title is excluded.

    """

    parts = re.split(r"\.\s+", text.strip(" .,:;"))
    journal = parts[-1].strip(" .,:;")
    if not journal:
        return ""
    for index in range(len(parts) - 2, -1, -1):
        fragment = parts[index].strip(" .,:;")
        words = fragment.split()
        # An isolated author fragment is not part of the journal title.
        author = re.fullmatch(
            r"[A-Z][a-z'-]+,?\s+[A-Z]{1,4}(?:,.*)?", fragment
        )
        if (
            not words or len(words) > 2
            or any(len(word) > (12 if len(words) == 1 else 7)
                   for word in words)
            or any(not word[:1].isupper() for word in words)
            or fragment.casefold() in {"et al", "and others"}
            or author is not None
        ):
            break
        journal = fragment + ". " + journal
    return journal


def _coordinates(entry: str) -> _Coordinates | None:
    """Identify journal, year, volume and first page or article number.

    Parameters
    ----------
    entry: str
        Original bibliography entry; a title is not required.

    Returns
    -------
    _Coordinates | None
        A citation with a usable journal and numeric coordinates, or
        None if the required fields cannot be reliably extracted.

    """

    text = " ".join(entry.split())
    for pattern in (
        _COORDINATES_YEAR_FIRST, _COORDINATES_YEAR_LAST,
        _COORDINATES_VOLUME_PAGE,
    ):
        for match in pattern.finditer(text):
            preceding = text[:match.start()]
            year = match.groupdict().get("year") or ""
            if not year:
                # In APA styles the year can precede the journal:
                # "Smith (1998). Phys. Rev. Lett. 80(2), 1234".
                earlier = list(_YEAR.finditer(preceding))
                if earlier:
                    year = earlier[-1].group()
                    preceding = preceding[earlier[-1].end():].lstrip(
                        "). ,;"
                    )
            journal = _journal_before(preceding)
            if not journal or not re.search(r"[A-Za-z]{2}", journal):
                continue
            # An author list is not a journal. A year is optional for
            # searching, but required before verifying a publication.
            if re.fullmatch(
                r"(?:[A-Z][a-z-]+\s+[A-Z]{1,3}\s*,?\s*){2,}",
                journal,
            ):
                continue
            return _Coordinates(
                journal=journal,
                year=year,
                volume=match.group("volume"),
                page=match.group("page"),
                issue=match.group("issue") or "",
            )
    return None


def _article_title(entry: str) -> str:
    """Extract an article title only when distinct from journal metadata.

    Parameters
    ----------
    entry: str
        Original bibliography entry.

    Returns
    -------
    str
        A likely article title or an empty string if none is evident.

    """

    text = " ".join(entry.split())
    coordinates = _coordinates(text)
    segments = re.split(r"\.\s+", text)
    for segment in segments[1:]:
        candidate = segment.strip(" .")
        if re.fullmatch(r"et\s+al|\(?\d{4}[a-z]?\)?", candidate, re.I):
            continue
        # Initials and surnames can form separate segments in APA style.
        if re.fullmatch(
            r"(?:&\s*)?[A-Z][\w'-]*,?\s+[A-Z]{1,4}", candidate
        ):
            continue
        if coordinates is not None:
            # Bibliographic coordinates are not an article title. This
            # also excludes abbreviated journal fragments (Phys. Rev.).
            if (
                _COORDINATES_YEAR_FIRST.search(candidate)
                or _COORDINATES_YEAR_LAST.search(candidate)
                or _COORDINATES_VOLUME_PAGE.search(candidate)
            ):
                continue
            journal_words = _words(coordinates.journal)
            if _words(candidate) and _words(candidate) <= journal_words:
                continue
            if _journal_key(candidate) in _journal_key(coordinates.journal):
                continue
        if len(re.findall(r"[^\W\d_]{2,}", candidate)) >= 2:
            return candidate[:_MAX_SEARCH_TITLE].rstrip()
    return ""


def _search_query(entry: str) -> str:
    """Build a title search or a bibliographic-coordinate search.

    Parameters
    ----------
    entry: str
        Original bibliography entry.

    Returns
    -------
    str
        Bounded query with the most distinctive available fields.

    """

    title = _article_title(entry)
    if title:
        return title
    coordinates = _coordinates(entry)
    if coordinates is not None:
        no_author = entry.strip().casefold().startswith(
            coordinates.journal.casefold()
        )
        surname = "" if no_author else first_author(entry) or ""
        terms = (
            surname, coordinates.journal, coordinates.year,
            coordinates.volume, coordinates.page,
        )
        return " ".join(part for part in terms if part)[
            :_MAX_QUERY_LENGTH
        ].rstrip()
    return " ".join(entry.split())[:_MAX_QUERY_LENGTH].rstrip()


def linkable_entry(entry: str) -> bool:
    """Tell whether even a short entry carries a useful publication locator.

    Parameters
    ----------
    entry: str
        Original bibliography entry, possibly lacking an article title.

    Returns
    -------
    bool
        Whether the entry contains a DOI or usable journal coordinates.

    Notes
    -----
    The reference matcher may call a short entry unreadable even when it
    gives enough coordinates for a human search. This does not change the
    original local-matching status or imply a verified publication.

    """

    return bool(_doi(entry) or _coordinates(entry) is not None)


def offline_link(number: int, entry: str) -> ReferenceLink:
    """Construct a link without contacting a remote service.

    Parameters
    ----------
    number: int
        Reference number.
    entry: str
        Original bibliography entry.

    Returns
    -------
    ReferenceLink
        Printed DOI if present, otherwise a Google Scholar search URL.

    """

    doi = _doi(entry)
    if doi:
        return ReferenceLink(
            number, f"https://doi.org/{quote(doi, safe='/._-;')}",
            "printed_doi", doi,
        )
    return ReferenceLink(
        number, _SEARCH + quote(_search_query(entry), safe="")
    )


def resolve_links(
    entries: dict[int, str], permission: LookupPermission | None,
) -> dict[int, ReferenceLink]:
    """Resolve approved references without guessing uncertain matches.

    Parameters
    ----------
    entries: dict[int, str]
        Missing reference numbers and original bibliography entries.
    permission: LookupPermission | None
        Explicit approval for these entries and approved endpoints.

    Returns
    -------
    dict[int, ReferenceLink]
        Preferred links, falling back to offline links on failures.

    Raises
    ------
    PermissionError
        If there is no permission, the scope differs, or a service is
        not explicitly approved.

    Notes
    -----
    This function does not download PDFs. Its OpenAlex PDF links are only
    acquisition candidates for a separate approved operation.

    """

    if permission is None or set(entries) != set(permission.numbers):
        raise PermissionError("Online bibliography lookup requires approval")
    if tuple(permission.services) != _SERVICES:
        raise PermissionError("Unrecognized bibliographic service grant")

    results: dict[int, ReferenceLink] = {}
    for number, entry in entries.items():
        link = offline_link(number, entry)
        try:
            item = _crossref(entry, permission)
            if item is not None and _matches(entry, item):
                doi = _doi(str(item.get("DOI", "")))
                if doi:
                    link = ReferenceLink(
                        number, "https://doi.org/" +
                        quote(doi, safe="/._-;"), "verified_doi", doi,
                    )
                    link = _openalex(link, permission)
        except HTTPError as error:
            # Rate limiting is an instruction to stop, not to retry each
            # remaining reference immediately against the same service.
            if error.code == 429:
                results[number] = link
                break
        except (ValueError, TypeError, KeyError, OSError):
            # A failed service must not invalidate the offline link.
            pass
        results[number] = link
    for number, entry in entries.items():
        results.setdefault(number, offline_link(number, entry))
    return results


def _doi(text: str) -> str:
    """Extract a conservatively bounded DOI from untrusted text.

    Parameters
    ----------
    text: str
        Reference entry or bibliographic metadata.

    Returns
    -------
    str
        DOI without trailing sentence punctuation, or an empty string.

    """

    match = _DOI.search(text)
    return match.group(1).rstrip(".,;:)]}") if match else ""


def _words(text: str) -> set[str]:
    """Return distinctive normalized words for title comparison.

    Parameters
    ----------
    text: str
        Publication title or reference entry.

    Returns
    -------
    set[str]
        Normalized words of sufficient length, excluding common words.

    """

    ascii_text = unicodedata.normalize("NFKD", text).encode(
        "ascii", "ignore"
    ).decode("ascii").lower()
    return set(_WORD.findall(ascii_text)) - _COMMON


def _normalized_name(text: str) -> str:
    """Normalize surnames, including short names, for comparison.

    Parameters
    ----------
    text: str
        Author family name.

    Returns
    -------
    str
        Lowercase letters without spacing or diacritics.

    """

    normalized = unicodedata.normalize("NFKD", text).encode(
        "ascii", "ignore"
    ).decode("ascii").lower()
    return "".join(re.findall(r"[a-z]+", normalized))


def _matches(entry: str, work: dict) -> bool:
    """Accept a DOI match or corroborated title or journal coordinates.

    Parameters
    ----------
    entry: str
        Original bibliography entry.
    work: dict
        Crossref work metadata.

    Returns
    -------
    bool
        True only when the work is supported by the extracted citation.

    """

    doi = _doi(str(work.get("DOI", "")))
    printed = _doi(entry)
    if printed:
        return printed.casefold() == doi.casefold()

    title = _article_title(entry)
    coordinates = _coordinates(entry)
    if title:
        titles = work.get("title") or []
        if not isinstance(titles, list) or not titles:
            return False
        title_words = _words(str(titles[0]))
        if len(title_words) < 3:
            return False
        overlap = len(title_words & _words(title)) / len(title_words)
        if overlap < 0.8:
            return False
    elif coordinates is None or not _matches_coordinates(coordinates, work):
        return False

    # A coordinate-only citation has no author to compare. Require the
    # full set of bibliographic coordinates instead; _crossref also
    # rejects multiple candidates with the same coordinates.
    no_author = (
        not title and coordinates is not None
        and entry.strip().casefold().startswith(
            coordinates.journal.casefold()
        )
    )
    if not no_author:
        authors = work.get("author") or []
        surname = first_author(entry)
        if not surname or not isinstance(authors, list) or not authors:
            return False
        if not isinstance(authors[0], dict):
            return False
        family = str(authors[0].get("family", ""))
        if _normalized_name(surname) != _normalized_name(family):
            return False

    year = _YEAR.search(entry)
    published = work.get("published") or work.get("issued") or {}
    if not isinstance(published, dict):
        return False
    parts = published.get("date-parts") or []
    if not year or not parts or not parts[0]:
        return False
    return int(year.group()) == int(parts[0][0])


def _journal_key(value: str) -> str:
    """Normalize punctuation and accents for journal-name matching.

    Parameters
    ----------
    value: str
        Full or abbreviated journal name.

    Returns
    -------
    str
        Only lower-case letters and digits.

    """

    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize(
        "NFKD", value
    ).encode("ascii", "ignore").decode("ascii").lower())


def _first_page(value: str) -> str:
    """Select a page or article number before an optional page range.

    Parameters
    ----------
    value: str
        Publication pagination or article identifier.

    Returns
    -------
    str
        Lowercase first page, without leading punctuation or whitespace.

    """

    return re.split(r"\s*[-–]\s*", value.strip(), maxsplit=1)[
        0
    ].strip().casefold()


def _same_journal(expected: str, candidate: str) -> bool:
    """Match full and commonly abbreviated journal names conservatively.

    Parameters
    ----------
    expected: str
        Printed journal name or abbreviation.
    candidate: str
        Journal name in the bibliographic service response.

    Returns
    -------
    bool
        Whether normalized names or corresponding word prefixes agree.

    """

    if _journal_key(expected) == _journal_key(candidate):
        return True
    a = re.findall(r"[a-z0-9]+", unicodedata.normalize(
        "NFKD", expected
    ).encode("ascii", "ignore").decode("ascii").lower())
    b = re.findall(r"[a-z0-9]+", unicodedata.normalize(
        "NFKD", candidate
    ).encode("ascii", "ignore").decode("ascii").lower())
    if len(a) < 2 or len(a) != len(b):
        return False
    return all(
        left == right or (
            min(len(left), len(right)) >= 3
            and (left.startswith(right) or right.startswith(left))
        )
        for left, right in zip(a, b)
    )


def _matches_coordinates(coordinates: _Coordinates, work: dict) -> bool:
    """Require the same journal, volume and first page or article ID.

    Parameters
    ----------
    coordinates: _Coordinates
        Bibliographic coordinates extracted from the reference.
    work: dict
        Candidate metadata returned by Crossref.

    Returns
    -------
    bool
        True when all available identifying coordinates agree.

    Notes
    -----
    A journal abbreviation is compared with the publisher's short title
    as well as the full title. A missing field is not treated as a match.
    The first author and year are checked separately by ``_matches``.

    """

    journals = (
        work.get("container-title") or [],
        work.get("short-container-title") or [],
    )
    if not any(
        _same_journal(coordinates.journal, name)
        for names in journals if isinstance(names, list)
        for name in names if isinstance(name, str)
    ):
        return False
    if str(work.get("volume") or "").strip() != coordinates.volume:
        return False
    if coordinates.issue and work.get("issue"):
        if str(work["issue"]).strip() != coordinates.issue:
            return False
    pages = (work.get("page"), work.get("article-number"))
    return any(
        _first_page(value) == _first_page(coordinates.page)
        for value in pages if isinstance(value, str) and value.strip()
    )


def _crossref(
    entry: str, permission: LookupPermission
) -> dict | None:
    """Query Crossref for candidates using original bibliography text.

    Parameters
    ----------
    entry: str
        Missing bibliography entry.
    permission: LookupPermission
        Approval covering this reference and Crossref.

    Returns
    -------
    dict | None
        A corroborated work, if Crossref provides one.

    """

    doi = _doi(entry)
    if doi:
        path = "/works/" + quote(doi, safe="/._-;")
    else:
        query = _search_query(entry)
        path = "/works?" + urlencode(
            {"query.bibliographic": query, "rows": 5}
        )
    body = _json_request(_CROSSREF + path, _CROSSREF, permission)
    message = body.get("message", {})
    if doi:
        return message if isinstance(message, dict) else None
    items = message.get("items", []) if isinstance(message, dict) else []
    matches = [
        item for item in items
        if isinstance(item, dict) and _matches(entry, item)
    ]
    # Even precise journal coordinates can refer to several articles in
    # journals that restart pagination per issue. Never guess among them.
    return matches[0] if len(matches) == 1 else None


def _openalex(
    link: ReferenceLink, permission: LookupPermission
) -> ReferenceLink:
    """Attach open-access candidate metadata to a verified DOI link.

    Parameters
    ----------
    link: ReferenceLink
        Reference with a DOI corroborated by Crossref.
    permission: LookupPermission
        Approval covering this reference and OpenAlex.

    Returns
    -------
    ReferenceLink
        Original DOI link, optionally supplemented with PDF metadata.

    """

    url = _OPENALEX + "/works/https://doi.org/" + quote(
        link.doi, safe="/._-;"
    )
    work = _json_request(url, _OPENALEX, permission)
    locations = [work.get("best_oa_location"), work.get("primary_location")]
    for location in locations:
        if not isinstance(location, dict):
            continue
        pdf_url = str(location.get("pdf_url") or "")
        if urlsplit(pdf_url).scheme != "https":
            continue
        return replace(
            link, pdf_url=pdf_url,
            license=str(location.get("license") or ""),
            version=str(location.get("version") or ""),
        )
    return link


def _json_request(
    url: str, service: str, permission: LookupPermission
) -> dict:
    """Make one bounded HTTPS request with an explicit approval grant.

    Parameters
    ----------
    url: str
        Bibliographic API URL, never a PDF or arbitrary website.
    service: str
        Service origin associated with the requested API route.
    permission: LookupPermission
        Operation-scoped approval containing this service.

    Returns
    -------
    dict
        Decoded JSON object from the bibliographic service.

    Raises
    ------
    PermissionError
        If the service was not approved for the operation.
    ValueError
        If the URL or response violates the service boundary.

    """

    if (
        not isinstance(permission, LookupPermission)
        or tuple(permission.services) != _SERVICES
        or service not in permission.services
    ):
        raise PermissionError("Bibliographic network request not authorized")
    parsed = urlsplit(url)
    approved = urlsplit(service)
    if (
        parsed.scheme != "https" or parsed.netloc != approved.netloc
        or parsed.username or parsed.password or not parsed.path.startswith(
            "/works"
        )
    ):
        raise ValueError("Bibliographic lookup outside approved service")
    request = Request(
        url, headers={"Accept": "application/json",
                      "User-Agent": "KiLin-reference-check/0.1"},
    )
    with build_opener(_NoRedirect()).open(
        request, timeout=_TIMEOUT
    ) as response:
        body = response.read(_RESPONSE_LIMIT + 1)
        if len(body) > _RESPONSE_LIMIT:
            raise ValueError("Bibliographic service response too large")
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid bibliographic response")
    return data
