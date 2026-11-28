"""Define compact representations of effective project configurations.

Only project-opening configuration observations use this representation.
A reference preserves the original event identity and identifies the latest
complete configuration by sequence number and a digest of its canonical JSON.
The original project path remains part of the recorded configuration.

Key Features
------------
- Canonical JSON equality independent of mapping order.
- Direct references to complete snapshots, never to other references.

Module Structure
----------------
- ``CONFIGURATION_EVENT`` and ``CONFIGURATION_REFERENCE`` identify the
  configuration observation and its compact payload.
- ``configuration_bytes`` encodes a complete snapshot for comparison.
- ``configuration_reference`` constructs a compact reference.
- ``resolve_configuration_reference`` verifies and expands a reference.

Examples
--------
>>> from kilin._trace_configuration import configuration_reference
>>> reference = configuration_reference(12, b"{}")
>>> reference["configuration_reference"]["sequence"]
12

"""

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
from typing import Final

__all__ = [
    "CONFIGURATION_EVENT",
    "CONFIGURATION_REFERENCE",
    "configuration_bytes",
    "configuration_reference",
    "resolve_configuration_reference",
]

CONFIGURATION_EVENT: Final[str] = "gui.project.configuration.effective"
CONFIGURATION_REFERENCE: Final[str] = "configuration_reference"
_CONFIGURATION_FIELDS: Final[frozenset[str]] = frozenset({
    "profile", "effort_levels", "router_models", "search_depth",
    "source_cards", "behavior", "tools",
})


def configuration_bytes(configuration: Mapping[str, object]) -> bytes:
    """Encode a complete JSON-ready configuration for exact comparison.

    Parameters
    ----------
    configuration: mapping of str to object
        Complete effective configuration, already converted to trace JSON
        values. Extra fields are retained and included in equality.

    Returns
    -------
    bytes
        UTF-8 JSON with ordered object keys and compact separators. List
        ordering and JSON scalar representations remain significant.

    Raises
    ------
    ValueError
        If required fields are missing, the payload is a reference, or a
        nonfinite number cannot be represented in standard JSON.
    TypeError
        If a value cannot be serialized as JSON.

    """

    if (CONFIGURATION_REFERENCE in configuration
            or not _CONFIGURATION_FIELDS <= configuration.keys()):
        raise ValueError("Expected a complete effective configuration.")
    return json.dumps(
        dict(configuration), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def configuration_reference(sequence: int, encoded: bytes) -> dict:
    """Construct the compact payload for an already published snapshot.

    Parameters
    ----------
    sequence: int
        Project-wide sequence number of the complete snapshot.
    encoded: bytes
        Canonical JSON of the complete snapshot.

    Returns
    -------
    dict
        Reference containing the sequence number and lowercase SHA-256 digest.

    """

    return {CONFIGURATION_REFERENCE: {
        "sequence": sequence, "sha256": sha256(encoded).hexdigest(),
    }}


def resolve_configuration_reference(
    details: dict, sequence: int, latest: tuple[int, dict] | None,
) -> dict:
    """Resolve only a direct, backward reference to the latest full snapshot.

    Parameters
    ----------
    details: dict
        Compact payload of the current configuration observation.
    sequence: int
        Sequence number of that observation.
    latest: tuple of int and dict or None
        Sequence and details of the most recent preceding full observation in
        this project trace. References do not replace this candidate.

    Returns
    -------
    dict
        Independent copy of the complete configuration.

    Raises
    ------
    ValueError
        If the payload is malformed, the target is missing or not the latest
        complete observation, or the digest does not match the target.

    """

    reference = details.get(CONFIGURATION_REFERENCE)
    if (set(details) != {CONFIGURATION_REFERENCE}
            or not isinstance(reference, dict)
            or set(reference) != {"sequence", "sha256"}
            or type(reference["sequence"]) is not int
            or not 0 < reference["sequence"] < sequence
            or not isinstance(reference["sha256"], str)):
        raise ValueError("Invalid project configuration reference.")
    if latest is None or latest[0] != reference["sequence"]:
        raise ValueError(
            "Configuration reference does not identify the latest snapshot."
        )
    expected = configuration_reference(
        latest[0], configuration_bytes(latest[1])
    )
    if details != expected:
        raise ValueError("Project configuration reference digest mismatch.")
    return deepcopy(latest[1])
