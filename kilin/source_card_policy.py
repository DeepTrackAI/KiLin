"""Resolve core-owned source-card indexing policy for one project.

Project types provide defaults; ``profile.ini`` overrides the project-wide
selection. A separate, case-sensitive JSON file holds per-source exceptions.
The master enabled switch always takes precedence over per-source settings.
All paths remain relative to the project's source directory. Reading policy
never loads a model or alters any project file.

Key Features
------------

- **Project-level defaults**

    Inherit whether to index cards, the preferred kind and automatic work.

- **Per-source choice**

    Disable individual files or select another registered kind without
    inferring one from the extension.

- **Core persistence**

    Validate configuration and write only explicitly requested changes.

Module Structure
----------------

- ``SourceCardPolicy`` and ``SourceCardOverride``

    Immutable, validated policy records.

- ``read_source_card_policy`` and ``write_source_card_policy``

    Resolve the project-wide policy from ``profile.ini``.

- ``read_source_card_overrides`` and ``write_source_card_overrides``

    Manage per-source choices in a dedicated JSON document.

- ``select_source_card_kinds``

    Apply overrides and the enabled master switch to selected paths.

Examples
--------
>>> from kilin.source_card_policy import SourceCardPolicy
>>> SourceCardPolicy(False, None, False).enabled
False

"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from configparser import ConfigParser
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
from typing import Final

from kilin.atomic_writes import atomic_write_text
from kilin.project_type import ProjectType
from kilin.retrievals.source_cards.models import _check_path
from kilin.retrievals.source_cards.registry import SOURCE_CARD_KINDS

__all__ = [
    "SourceCardPolicy",
    "SourceCardOverride",
    "read_source_card_policy",
    "write_source_card_policy",
    "read_source_card_overrides",
    "write_source_card_overrides",
    "select_source_card_kinds",
]

_PROFILE: Final[str] = "profile.ini"
_SECTION: Final[str] = "source_cards"
_OVERRIDES: Final[str] = "source_card_sources.json"
_OVERRIDES_FORMAT: Final[str] = "source-card-source-settings-v1"


@dataclass(frozen=True, slots=True)
class SourceCardPolicy:
    """Project-wide source-card indexing choice.

    Parameters
    ----------
    enabled: bool
        Master switch for storing, generating, and searching cards.
    kind: str | None
        Default registered kind for selected sources, or ``None``.
    automatic: bool
        Whether missing or stale cards are generated on indexing.

    """

    enabled: bool
    kind: str | None
    automatic: bool

    def __post_init__(self) -> None:
        """Reject invalid project-wide source-card settings.

        Raises
        ------
        ValueError
            If a setting is malformed or the kind is not registered.

        """

        if type(self.enabled) is not bool or type(self.automatic) is not bool:
            raise ValueError("Source-card switches must be booleans.")
        _check_kind(self.kind)


@dataclass(frozen=True, slots=True)
class SourceCardOverride:
    """Case-sensitive override for one repository-relative source.

    Parameters
    ----------
    enabled: bool | None, optional
        Whether this source is selected; ``None`` inherits the project.
    kind: str | None, optional
        Registered kind to choose; ``None`` inherits the project.

    """

    enabled: bool | None = None
    kind: str | None = None

    def __post_init__(self) -> None:
        """Validate optional switches and kind identifiers.

        Raises
        ------
        ValueError
            If the override is not a valid, explicit selection.

        """

        if self.enabled is not None and type(self.enabled) is not bool:
            raise ValueError("Source-card enabled override must be boolean.")
        _check_kind(self.kind)
        if self.enabled is None and self.kind is None:
            raise ValueError("A source-card override must change a setting.")


def read_source_card_policy(
    root: Path, project_type: ProjectType
) -> SourceCardPolicy:
    """Read project overrides with packaged project-type defaults.

    Parameters
    ----------
    root: pathlib.Path
        Project root containing ``profile.ini``.
    project_type: ProjectType
        Validated project type with its packaged source-card defaults.

    Returns
    -------
    SourceCardPolicy
        Resolved core-owned indexing choice.

    """

    if not isinstance(project_type, ProjectType):
        raise ValueError("Expected a validated project type.")
    reader = ConfigParser(interpolation=None)
    reader.read(Path(root) / _PROFILE, encoding="utf-8")
    if not reader.has_section(_SECTION):
        return SourceCardPolicy(
            project_type.source_cards_enabled,
            project_type.source_card_kind,
            project_type.source_cards_automatic,
        )
    keys = set(reader.options(_SECTION))
    if keys - {"enabled", "kind", "automatic"}:
        raise ValueError("Unknown source-card project setting.")
    try:
        kind = reader.get(
            _SECTION, "kind", fallback=project_type.source_card_kind
        )
        return SourceCardPolicy(
            reader.getboolean(
                _SECTION, "enabled",
                fallback=project_type.source_cards_enabled,
            ),
            kind or None,
            reader.getboolean(
                _SECTION, "automatic",
                fallback=project_type.source_cards_automatic,
            ),
        )
    except ValueError as error:
        raise ValueError("Invalid source-card project settings.") from error


def write_source_card_policy(root: Path, policy: SourceCardPolicy) -> None:
    """Persist project-wide overrides without changing unrelated sections.

    Parameters
    ----------
    root: pathlib.Path
        Existing project root.
    policy: SourceCardPolicy
        Fully validated source-card indexing policy.

    Raises
    ------
    ValueError
        If the policy or its location is invalid.

    """

    if not isinstance(policy, SourceCardPolicy):
        raise ValueError("Expected a SourceCardPolicy.")
    policy.__post_init__()
    file = Path(root) / _PROFILE
    if not file.is_file():
        raise ValueError("Project profile.ini does not exist.")
    reader = ConfigParser(interpolation=None)
    reader.read(file, encoding="utf-8")
    if not reader.has_section(_SECTION):
        reader.add_section(_SECTION)
    reader.set(_SECTION, "enabled", str(policy.enabled).lower())
    reader.set(_SECTION, "kind", policy.kind or "")
    reader.set(_SECTION, "automatic", str(policy.automatic).lower())
    output = StringIO()
    reader.write(output)
    atomic_write_text(file, output.getvalue())


def read_source_card_overrides(
    root: Path,
) -> dict[Path, SourceCardOverride]:
    """Load source-relative choices without inferring kinds from suffixes.

    Parameters
    ----------
    root: pathlib.Path
        Root containing optional ``source_card_sources.json``.

    Returns
    -------
    dict[pathlib.Path, SourceCardOverride]
        Exact-case source paths and their validated choices.

    Raises
    ------
    ValueError
        If JSON contains duplicate keys, invalid paths, or unknown fields.

    """

    file = Path(root) / _OVERRIDES
    if not file.exists():
        return {}
    try:
        data = json.loads(
            file.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_keys,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(
            "Invalid source-card source settings JSON."
        ) from error
    if (
        not isinstance(data, dict)
        or set(data) != {"format", "sources"}
        or data["format"] != _OVERRIDES_FORMAT
        or not isinstance(data["sources"], dict)
    ):
        raise ValueError("Unknown source-card source-settings format.")
    result = {}
    for name, entry in data["sources"].items():
        if not isinstance(name, str):
            raise ValueError("Source override paths must be text.")
        path = Path(name)
        _check_path(path)
        if path.as_posix() != name:
            raise ValueError("Source override paths must be canonical.")
        if not isinstance(entry, dict) or (
            set(entry) - {"enabled", "kind"}
        ) or not entry:
            raise ValueError("Invalid source-card override object.")
        result[path] = SourceCardOverride(**entry)
    return result


def write_source_card_overrides(
    root: Path, overrides: Mapping[Path, SourceCardOverride]
) -> None:
    """Atomically save explicit, case-sensitive per-source choices.

    Parameters
    ----------
    root: pathlib.Path
        Project root.
    overrides: Mapping[pathlib.Path, SourceCardOverride]
        Source-relative paths and their validated selections.

    Raises
    ------
    ValueError
        If keys or values are invalid or ambiguous.

    """

    entries = {}
    for path, override in overrides.items():
        _check_path(path)
        if not isinstance(override, SourceCardOverride):
            raise ValueError("Expected SourceCardOverride values.")
        override.__post_init__()
        entries[path.as_posix()] = {
            key: value for key, value in (
                ("enabled", override.enabled), ("kind", override.kind),
            ) if value is not None
        }
    payload = {
        "format": _OVERRIDES_FORMAT,
        "sources": dict(sorted(entries.items())),
    }
    atomic_write_text(
        Path(root) / _OVERRIDES,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def select_source_card_kinds(
    paths: Iterable[Path], policy: SourceCardPolicy,
    overrides: Mapping[Path, SourceCardOverride],
) -> dict[Path, str]:
    """Apply the project master switch and individual source choices.

    Parameters
    ----------
    paths: Iterable[pathlib.Path]
        Selected source-relative paths from the current manifest.
    policy: SourceCardPolicy
        Project-wide master switch, kind and generation policy.
    overrides: Mapping[pathlib.Path, SourceCardOverride]
        Source-specific exceptions, also subject to the master switch.

    Returns
    -------
    dict[pathlib.Path, str]
        One registered kind per selected and enabled source.

    """

    if not policy.enabled:
        return {}
    selected = {}
    for path in paths:
        _check_path(path)
        override = overrides.get(path)
        if override is not None and override.enabled is False:
            continue
        kind = override.kind if override and override.kind else policy.kind
        if kind is not None:
            _check_kind(kind)
            selected[path] = kind
    return selected


def _check_kind(kind: str | None) -> None:
    """Ensure that kinds are explicit trusted registrations.

    Parameters
    ----------
    kind: str | None
        Kind to validate, or ``None`` for an unselected default.

    Raises
    ------
    ValueError
        If a nonregistered kind was supplied.

    """

    if kind is not None:
        if not isinstance(kind, str):
            raise ValueError("Source-card kind must be a string.")
        SOURCE_CARD_KINDS.get(kind)


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    """Reject duplicate JSON keys before they can hide source choices.

    Parameters
    ----------
    pairs: list[tuple[str, object]]
        Members of a JSON object in source order.

    Returns
    -------
    dict
        Object whose keys are unique.

    """

    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate source-card settings key.")
        result[key] = value
    return result
