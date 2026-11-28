"""Inspect historical plugin identities and preflight fresh replay inputs.

Historical inspection uses only recorded data, never the inspecting runtime's
catalog. Fresh replay explicitly resolves trusted installed plugins before
creating or archiving a destination.

Key Features
------------
- Preserve unknown identities in older traces without reconstructing them.
- Compare captured identities at matching command boundaries.
- Resolve historical and current plugin requirements before replay.

Module Structure
----------------
- ``plugin_observations`` decodes recorded plugin identities.
- ``compare_plugins`` compares observations without loading plugins.
- ``preflight_plugins`` checks the executing runtime and captures identities.

Examples
--------
>>> from kilin.system_testing._plugin_provenance import plugin_observations
>>> plugin_observations(())
[]

"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import re

from .timeline import TimelineOperation
from .traces import TraceEvent

__all__ = []


def _observation(
    event: TraceEvent, scope: str, kind: str, name: str, identity: object,
) -> dict:
    """Decode an optional portable identity without consulting a catalog.

    Parameters
    ----------
    event: TraceEvent
        Historical event supplying the observation.
    scope: str
        Opening, turn, or tool execution boundary.
    kind: str
        Project type or tool identity category.
    name: str
        Identifier recorded independently of the identity payload.
    identity: object
        Optional identity payload. None represents unknown provenance.

    Returns
    -------
    dict
        Observation with recorded, unknown or invalid status.

    """

    result = {"sequence": event.sequence, "scope": scope, "kind": kind,
              "name": name, "status": "unknown", "identity": None}
    if identity is None:
        return result
    key = "type_id" if kind == "project_type" else "name"
    version = "schema_version" if kind == "project_type" else "api_version"
    required = {key, "origin", version, "sha256"}
    if kind == "tool":
        required.add("version")
    if (not isinstance(identity, dict) or set(identity) != required
            or identity[key] != name
            or identity["origin"] not in ("distribution", "user")
            or type(identity[version]) is not int or identity[version] < 1
            or not isinstance(identity["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is None
            or (kind == "tool" and (
                not isinstance(identity["version"], str)
                or not identity["version"].strip()))):
        result.update(status="invalid", error="Invalid plugin identity.")
    else:
        result.update(status="recorded", identity=deepcopy(identity))
    return result


def plugin_observations(events: tuple[TraceEvent, ...]) -> list[dict]:
    """Read the opening and execution identities from historical events.

    Parameters
    ----------
    events: tuple[TraceEvent, ...]
        Decoded events, including expanded configuration observations.

    Returns
    -------
    list[dict]
        Chronological observations. Legacy configurations retain known IDs
        but explicitly unknown identities. Invalid optional provenance is
        reported without pretending it is a verified identity.

    """

    result = []
    for event in events:
        data = event.details
        if event.event_type == "gui.project.configuration.effective":
            profile = data.get("profile")
            name = (profile.get("type_id")
                    if isinstance(profile, dict) else None)
            tools = data.get("tools")
            names = sorted({entry["name"] for entry in tools
                            if isinstance(entry, dict)
                            and isinstance(entry.get("name"), str)}) if (
                                isinstance(tools, list)) else []
            snapshot = data.get("plugins", {"project_type": None,
                                            "tools": dict.fromkeys(names)})
            if (not isinstance(snapshot, dict)
                    or set(snapshot) != {"project_type", "tools"}
                    or not isinstance(snapshot["tools"], dict)
                    or set(snapshot["tools"]) != set(names)):
                result.append({"sequence": event.sequence, "scope": "open",
                               "kind": "project_type", "name": name,
                               "status": "invalid", "identity": None,
                               "error": "Invalid plugin snapshot."})
                continue
            if isinstance(name, str):
                result.append(_observation(event, "open", "project_type",
                                           name, snapshot["project_type"]))
            for tool in names:
                result.append(_observation(event, "open", "tool", tool,
                                           snapshot["tools"][tool]))
        elif event.event_type == "turn.settings.snapshot":
            name = data.get("project_type_id")
            if isinstance(name, str):
                result.append(_observation(event, "turn", "project_type",
                                           name, data.get(
                                               "project_type_plugin")))
        elif event.event_type == "tool.run.started":
            name = data.get("name")
            if isinstance(name, str):
                result.append(_observation(event, "execution", "tool",
                                           name, data.get("plugin")))
    return result


def compare_plugins(
    original: TimelineOperation, replayed: TimelineOperation,
) -> list[dict]:
    """Compare matching historical plugin observations, never live code.

    Parameters
    ----------
    original: TimelineOperation
        Reference command with recorded observations.
    replayed: TimelineOperation
        Aligned replay command with its own observations.

    Returns
    -------
    list[dict]
        Pairs with equal, changed, unknown, invalid or one-sided status.
        An unobserved tool may reflect different routing, not changed code.

    """

    groups = []
    for op in (original, replayed):
        grouped = defaultdict(list)
        for item in plugin_observations(op.events):
            key = (item["scope"], item["kind"], item["name"] or "")
            grouped[key].append(item)
        groups.append(grouped)
    left, right = groups
    result = []
    for scope, kind, name in sorted(left.keys() | right.keys()):
        key = scope, kind, name
        before, after = left[key], right[key]
        for index in range(max(len(before), len(after))):
            a = before[index] if index < len(before) else None
            b = after[index] if index < len(after) else None
            if a is None or b is None:
                status = "one_sided"
            elif "invalid" in (a["status"], b["status"]):
                status = "invalid"
            elif "unknown" in (a["status"], b["status"]):
                status = "unknown"
            else:
                status = ("equal" if a["identity"] == b["identity"]
                          else "changed")
            result.append({"scope": scope, "kind": kind, "name": name,
                           "status": status, "original": a, "replayed": b})
    return result


def preflight_plugins(operations: tuple[TimelineOperation, ...]) -> dict:
    """Require usable historical and current plugins before fresh execution.

    Parameters
    ----------
    operations: tuple[TimelineOperation, ...]
        Validated reference operations, starting with project creation.

    Returns
    -------
    dict
        Portable identities of the current type and all required tools,
        including any tools required by the historical configuration or
        observed historical tool execution.

    Raises
    ------
    ValueError
        If a required plugin is missing or unavailable. A changed identity
        alone is not an error: replay intentionally uses current code.

    Notes
    -----
    Only fresh execution calls this function. Registry access imports
    trusted installed plugins; historical names never authorize imports.

    """

    from kilin.plugins import list_tools
    from kilin.project_type import project_type_catalog
    from kilin.tools._registry import _tool_identity, registered_tools

    type_id = operations[0].request["source_batch"]["type_id"]
    info = project_type_catalog().get(type_id)
    if info is None or not info.available:
        reason = info.error if info is not None else "Not installed."
        raise ValueError(f"Required project type {type_id!r} is unavailable: "
                         f"{reason}")
    required = set(info.required_tools)
    for op in operations:
        for item in plugin_observations(op.events):
            if item["kind"] == "tool":
                required.add(item["name"])
    missing = required - registered_tools().keys()
    if missing:
        diagnostics = {entry.name: entry.error for entry in list_tools()
                       if not entry.available}
        details = "; ".join(
            f"{name}: {diagnostics.get(name) or 'Not installed.'}"
            for name in sorted(missing)
        )
        raise ValueError(f"Required tool plugins unavailable: {details}")
    return {"project_type": info.identity(),
            "tools": {name: _tool_identity(name) for name in sorted(required)}}
