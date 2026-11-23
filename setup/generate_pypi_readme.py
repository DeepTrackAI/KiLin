"""Generate the tag-pinned PyPI description from the repository README.

The source README retains branch-relative file references on GitHub. PyPI
cannot resolve these paths, so this build-time document replaces only
repository-local links and image sources with URLs for the package's release
tag. This module uses the standard library and does not import KíLín.

"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import tomllib
from urllib.parse import quote, unquote, urlsplit


__all__ = ["render_pypi_readme", "read_project_version", "main"]

_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY = "https://github.com/DeepTrackAI/KiLin"
_RAW = "https://raw.githubusercontent.com/DeepTrackAI/KiLin"
_HEADER = (
    "<!-- Generated from README.md by setup/generate_pypi_readme.py; "
    "do not edit. -->\n\n"
)
_MARKDOWN_LINK = re.compile(
    r"(?P<start>!?\[[^\]\n]*\]\()"
    r"(?P<url><[^>\n]+>|[^\s()\n]+)"
    r"(?P<end>(?:\s+\"[^\"\n]*\")?\))"
)
_HTML_ATTRIBUTE = re.compile(
    r'(?P<name>\b(?:src|srcset|href))="(?P<url>[^"]*)"'
)
_INLINE_CODE = re.compile(r"(`+)(.*?)\1")
_FENCE = re.compile(r"^[ ]{0,3}(?P<ticks>`{3,}|~{3,})")


def read_project_version(root: Path) -> str:
    """Read the exact release version from the packaging manifest.

    Parameters
    ----------
    root : Path
        Root of the KíLín checkout.

    Returns
    -------
    str
        PEP 440 version used for the tag ``v<version>``.

    Raises
    ------
    ValueError
        If the version cannot be used safely in a Git ref.

    """

    with (root / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    if not isinstance(version, str) or not re.fullmatch(
        r"[0-9][A-Za-z0-9._-]*", version
    ):
        raise ValueError("Invalid release version in pyproject.toml")
    return version


def _absolute_url(
    url: str, *, root: Path, tag: str, image: bool
) -> str:
    """Resolve a repository-local reference against a release tag.

    Parameters
    ----------
    url : str
        URL or relative path used in the source README.
    root : Path
        Repository checkout that owns relative targets.
    tag : str
        Git tag identifying the exact distribution source.
    image : bool
        Whether to emit a raw-image URL instead of a GitHub page URL.

    Returns
    -------
    str
        Unchanged external reference or validated, tag-pinned URL.

    Raises
    ------
    ValueError
        If a relative path escapes the checkout or does not exist.

    """

    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return url
    path_text = unquote(parsed.path)
    if path_text.startswith(("/", "\\")) or "\\" in path_text:
        raise ValueError(f"Local README URL must be relative: {url}")
    resolved_root = root.resolve()
    target = (resolved_root / path_text).resolve()
    if not target.is_relative_to(resolved_root) or not target.exists():
        raise ValueError(f"Invalid local README target: {url}")
    if image and not target.is_file():
        raise ValueError(f"README image target is not a file: {url}")
    name = quote(target.relative_to(resolved_root).as_posix(), safe="/")
    if image:
        result = f"{_RAW}/{tag}/{name}"
    else:
        view = "blob" if target.is_file() else "tree"
        result = f"{_REPOSITORY}/{view}/{tag}/{name}"
    if parsed.query:
        result += f"?{parsed.query}"
    if parsed.fragment:
        result += f"#{parsed.fragment}"
    return result


def _rewrite_segment(text: str, *, root: Path, tag: str) -> str:
    """Rewrite links and HTML attributes in a non-code text segment.

    Parameters
    ----------
    text : str
        Markdown or HTML text that contains no inline-code spans.
    root : Path
        Repository checkout containing referenced files.
    tag : str
        Version-specific Git tag to use in destination URLs.

    Returns
    -------
    str
        Segment with local links and image URLs resolved to the tag.

    """

    def markdown(match: re.Match[str]) -> str:
        url = match.group("url")
        angled = url.startswith("<") and url.endswith(">")
        unwrapped = url[1:-1] if angled else url
        absolute = _absolute_url(
            unwrapped,
            root=root,
            tag=tag,
            image=match.group("start").startswith("!"),
        )
        if angled:
            absolute = f"<{absolute}>"
        return match.group("start") + absolute + match.group("end")

    def html(match: re.Match[str]) -> str:
        attribute = match.group("name")
        value = match.group("url")
        if attribute == "srcset" and value.startswith("data:"):
            return match.group()
        if attribute == "srcset":
            entries = []
            for entry in value.split(","):
                stripped = entry.strip()
                if not stripped:
                    continue
                parts = stripped.split(maxsplit=1)
                converted = _absolute_url(
                    parts[0], root=root, tag=tag, image=True
                )
                entries.append(" ".join((converted, *parts[1:])))
            value = ", ".join(entries)
        else:
            value = _absolute_url(
                value,
                root=root,
                tag=tag,
                image=attribute == "src",
            )
        return f'{attribute}="{value}"'

    return _HTML_ATTRIBUTE.sub(html, _MARKDOWN_LINK.sub(markdown, text))


def _rewrite_line(line: str, *, root: Path, tag: str) -> str:
    """Rewrite a prose line without modifying inline-code spans.

    Parameters
    ----------
    line : str
        One source README line outside a fenced code block.
    root : Path
        Checkout root used to resolve file and image references.
    tag : str
        Version-specific Git tag used for generated references.

    Returns
    -------
    str
        Line with only non-code local links converted.

    """

    parts = []
    end = 0
    for match in _INLINE_CODE.finditer(line):
        parts.append(
            _rewrite_segment(line[end:match.start()], root=root, tag=tag)
        )
        parts.append(match.group())
        end = match.end()
    parts.append(_rewrite_segment(line[end:], root=root, tag=tag))
    return "".join(parts)


def render_pypi_readme(source: str, version: str, root: Path) -> str:
    """Render the package description with links pinned to its Git tag.

    Parameters
    ----------
    source : str
        Maintained repository README text.
    version : str
        Exact package version from ``pyproject.toml``.
    root : Path
        Checkout root containing all referenced local files.

    Returns
    -------
    str
        Generated Markdown suitable for the PyPI long description.

    Notes
    -----
    Relative Markdown links and ``src``, ``srcset`` and ``href`` HTML
    attributes are rewritten outside fenced and inline code. External URLs,
    badges and same-page anchors remain unchanged. Use encoded filenames for
    unusual Markdown paths; paths must remain inside the checkout.

    """

    tag = f"v{version}"
    result = []
    fence_char = ""
    fence_width = 0
    for line in source.splitlines(keepends=True):
        marker = _FENCE.match(line)
        if marker:
            ticks = marker.group("ticks")
            if not fence_char:
                fence_char, fence_width = ticks[0], len(ticks)
            elif (
                ticks[0] == fence_char
                and len(ticks) >= fence_width
                and not line[marker.end():].strip()
            ):
                fence_char = ""
            result.append(line)
        elif fence_char:
            result.append(line)
        else:
            result.append(_rewrite_line(line, root=root, tag=tag))
    return _HEADER + "".join(result)


def main(argv: list[str] | None = None) -> int:
    """Regenerate or verify the committed PyPI README.

    Parameters
    ----------
    argv : list[str] or None, optional
        Command-line arguments; ``--check`` verifies without writing.

    Returns
    -------
    int
        Zero when the generated description is current, else one.

    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if output is stale"
    )
    args = parser.parse_args(argv)
    version = read_project_version(_ROOT)
    source = (_ROOT / "README.md").read_text(encoding="utf-8")
    expected = render_pypi_readme(source, version, _ROOT)
    target = _ROOT / "README.pypi.md"
    if args.check:
        if not target.is_file() or (
            target.read_text(encoding="utf-8") != expected
        ):
            print(
                "README.pypi.md is stale: run "
                "python -m setup.generate_pypi_readme",
                file=sys.stderr,
            )
            return 1
        print(f"README.pypi.md matches v{version}")
        return 0
    target.write_text(expected, encoding="utf-8", newline="\n")
    print(f"Generated README.pypi.md for v{version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
