"""Resolve and edit project-local behavior over packaged project types.

A project stores only differences from the immutable packaged definition in
``project_behavior.json``. Every effective type passes the existing core
validator; saving an editor checks revision and validates before atomic
replacement. Tool bindings and the profile's settings remain independent.
Conversation-memory policy is a separate editor scope from source indexing,
and an intent's ordinary settings and memory policy are saved independently.

Key Features
------------

- **Project-local inheritance**

    Shared and per-intent settings preserve packaged defaults until edited.

- **Validated atomic saves**

    The entire effective project type is validated before an update.

- **Independent memory configuration**

    Global memory and per-intent policies use separate revision-checked
    editor scopes, without triggering source reindexing or changing the
    existing GUI's global and intent editor payloads.

Module Structure
----------------

- ``effective_project_type``

    Resolve a project's packaged and project-local settings.

- ``project_behavior_settings``

    Build editor payloads containing values, defaults and revision.

- ``save_project_behavior``

    Validate and persist a revision-checked project behavior editor.

Examples
--------
>>> from kilin.project_behavior import effective_project_type
>>> config = effective_project_type('/project/path')  # doctest: +SKIP

"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Final

from kilin.atomic_writes import atomic_write_text
from kilin.project import open_project
from kilin.project_type import (
    PROJECT_TYPES_DIR,
    Intent,
    ProjectType,
    _parse_project_type,
    _unique_keys,
    project_type,
)
from kilin.trace_logging import NullTraceLogger, TraceSink

__all__ = [
    "PROJECT_BEHAVIOR_FILE",
    "effective_project_type",
    "project_behavior_settings",
    "save_project_behavior",
]

PROJECT_BEHAVIOR_FILE: Final[str] = "project_behavior.json"
_FORMAT: Final[str] = "project-behavior"
_DELETE: Final[dict[str, bool]] = {"$delete": True}
_MAX_BYTES: Final[int] = 1_048_576
_GLOBAL_FIELDS: Final[frozenset[str]] = frozenset((
    "description", "system_prompt", "answer_guidance",
    "retrieval_instruction", "history", "default_intent",
    "welcome_prompts", "suggestions",
))
_CAPABILITY_FIELDS: Final[frozenset[str]] = frozenset((
    "code_graph", "document_headers",
))
_MEMORY_FIELDS: Final[frozenset[str]] = frozenset((
    "enabled", "questions", "answers", "max_hits", "max_tokens",
    "exclude_disliked_answers",
))
_INTENT_MEMORY_FIELDS: Final[frozenset[str]] = frozenset((
    "indexing", "read",
))
_INTENT_FIELDS: Final[frozenset[str]] = frozenset((
    "description", "system_prompt", "answer_guidance", "history",
    "retrieval", "shortcuts", "cues", "requires", "thinking",
))
_RETRIEVAL_FIELDS: Final[frozenset[str]] = frozenset((
    "mode", "strategy", "include_related_evidence", "prefer_source_code",
    "max_tokens", "max_sources", "passages_per_source", "max_passages",
    "evidence_tokens", "coverage",
))


def _global_values(definition: ProjectType) -> dict[str, object]:
    """Return editable fields shared by the project's intents.

    Parameters
    ----------
    definition: ProjectType
        Validated packaged or effective project definition.

    Returns
    -------
    dict[str, object]
        Resolved editable project-wide settings.

    """

    return {
        "description": definition.description,
        "system_prompt": definition.system_prompt,
        "answer_guidance": definition.answer_guidance,
        "retrieval_instruction": definition.retrieval_instruction,
        "history": {"recent_turns": definition.recent_turns},
        "default_intent": definition.default_intent,
        "welcome_prompts": list(definition.welcome_prompts),
        "suggestions": list(definition.suggestions),
    }


def _capability_values(definition: ProjectType) -> dict[str, object]:
    """Return the two advanced indexing capability settings.

    Parameters
    ----------
    definition: ProjectType
        Validated project definition.

    Returns
    -------
    dict[str, object]
        Indexed code-graph and document-header settings.

    """

    return {
        "code_graph": definition.code_graph,
        "document_headers": definition.document_headers,
    }


def _memory_values(definition: ProjectType) -> dict[str, object]:
    """Return editable memory and disliked-history settings.

    Parameters
    ----------
    definition: ProjectType
        Validated effective or packaged project definition.

    Returns
    -------
    dict[str, object]
        Independent memory limits, inclusion, and history filtering policy.

    """

    memory = definition.conversation_memory
    return {
        "enabled": memory.enabled,
        "questions": memory.questions,
        "answers": memory.answers,
        "max_hits": memory.max_hits,
        "max_tokens": memory.max_tokens,
        "exclude_disliked_answers": definition.exclude_disliked_answers,
    }


def _intent_memory_values(intent: Intent) -> dict[str, object]:
    """Return the per-intent indexing and memory-search policy.

    Parameters
    ----------
    intent: Intent
        Validated effective or packaged intent.

    Returns
    -------
    dict[str, object]
        Memory indexing mode and retrieval permission.

    """

    return {
        "indexing": intent.memory_indexing,
        "read": intent.memory_read,
    }


def _intent_values(intent: Intent) -> dict[str, object]:
    """Expose an intent's editable fields without its fixed tool binding.

    Parameters
    ----------
    intent: Intent
        Validated intent to serialize into an editor.

    Returns
    -------
    dict[str, object]
        JSON-compatible effective settings.

    """

    retrieval: dict[str, object] = {
        "mode": intent.retrieval_mode,
        "strategy": intent.retrieval_strategy,
        "include_related_evidence": intent.include_related_evidence,
        "prefer_source_code": intent.prefer_source_code,
        "max_tokens": intent.max_tokens,
    }
    if (
        intent.retrieval_mode == "search"
        and intent.retrieval_strategy == "source_first"
        and intent.tool is None
    ):
        limits = intent.source_limits()
        retrieval.update({
            "max_sources": limits.max_sources,
            "passages_per_source": limits.passages_per_source,
            "max_passages": limits.max_passages,
            "evidence_tokens": limits.max_tokens,
            "coverage": limits.coverage,
        })
    return {
        "description": intent.description,
        "system_prompt": intent.system_prompt,
        "answer_guidance": intent.answer_guidance,
        "history": {"policy": intent.history_policy},
        "retrieval": retrieval,
        "shortcuts": [item.pattern for item in intent.shortcuts],
        "cues": [item.pattern for item in intent.cues],
        "requires": [item.pattern for item in intent.requires],
        "thinking": intent.thinking,
    }


def _check_keys(value: object, allowed: frozenset[str], where: str) -> dict:
    """Require an object containing only explicitly permitted keys.

    Parameters
    ----------
    value: object
        Untrusted JSON value.
    allowed: frozenset[str]
        Keys allowed at this scope.
    where: str
        Context to report for invalid settings.

    Returns
    -------
    dict
        Checked mapping.

    Raises
    ------
    ValueError
        When a field is invalid or the value is not an object.

    """

    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object.")
    extra = set(value) - allowed
    if extra:
        raise ValueError(
            f"{where} has unknown settings: {', '.join(sorted(extra))}."
        )
    return value


def _check_override_shape(saved: dict, defaults: ProjectType) -> None:
    """Reject edits to tool assignments or unregistered intent IDs.

    Parameters
    ----------
    saved: dict
        Sparse project configuration to check.
    defaults: ProjectType
        Packaged type determining allowed intent IDs.

    """

    global_values = _check_keys(
        saved.get("global", {}), _GLOBAL_FIELDS, "Behavior global settings"
    )
    if "history" in global_values and global_values["history"] != _DELETE:
        _check_keys(
            global_values["history"], frozenset(("recent_turns",)),
            "Behavior global history",
        )
    _check_keys(
        saved.get("capabilities", {}), _CAPABILITY_FIELDS,
        "Behavior indexing capabilities",
    )
    _check_keys(
        saved.get("memory", {}), _MEMORY_FIELDS,
        "Behavior conversation memory",
    )
    intents = saved.get("intents", {})
    _check_keys(intents, frozenset(defaults.intents), "Behavior intents")
    for name, value in intents.items():
        fields = _check_keys(
            value, _INTENT_FIELDS | frozenset(("memory",)),
            f"Behavior intent {name}",
        )
        for key, allowed in (
            ("history", frozenset(("policy",))),
            ("retrieval", _RETRIEVAL_FIELDS),
            ("memory", _INTENT_MEMORY_FIELDS),
        ):
            if key in fields and fields[key] != _DELETE:
                _check_keys(fields[key], allowed,
                            f"Behavior intent {name}.{key}")


def _merge(base: dict, patch: dict) -> dict:
    """Apply sparse patches and optional-field removal without mutation.

    Parameters
    ----------
    base: dict
        Original packaged mapping.
    patch: dict
        Sparse overrides.

    Returns
    -------
    dict
        Fresh merged mapping.

    """

    result = deepcopy(base)
    for key, value in patch.items():
        if value == _DELETE:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _diff(default: dict, new: dict) -> dict:
    """Find changed and removed fields for a sparse override document.

    Parameters
    ----------
    default: dict
        Packaged normalized settings.
    new: dict
        Complete edited normalized settings.

    Returns
    -------
    dict
        Sparse differences from the defaults.

    """

    changed = {}
    for key in default.keys() | new.keys():
        if key not in new:
            changed[key] = deepcopy(_DELETE)
        elif key not in default:
            changed[key] = deepcopy(new[key])
        elif isinstance(default[key], dict) and isinstance(new[key], dict):
            nested = _diff(default[key], new[key])
            if nested:
                changed[key] = nested
        elif (type(default[key]) is not type(new[key])
              or default[key] != new[key]):
            changed[key] = deepcopy(new[key])
    return changed


def _state(root: Path, type_id: str) -> tuple[dict, str]:
    """Read a bounded, validated saved override and its SHA revision.

    Parameters
    ----------
    root: pathlib.Path
        Project workspace root.
    type_id: str
        Type ID stored in the project profile.

    Returns
    -------
    tuple[dict, str]
        Saved override and SHA-256 revision.

    Raises
    ------
    ValueError
        For corrupted or mismatched saved configuration.

    """

    path = root / PROJECT_BEHAVIOR_FILE
    if not path.exists() and not path.is_symlink():
        return {
            "global": {}, "intents": {}, "capabilities": {},
            "memory": {},
        }, (
            sha256(b"").hexdigest()
        )
    if path.is_symlink() or not path.is_file():
        raise ValueError("Project behavior must be a regular project file.")
    if path.stat().st_size > _MAX_BYTES:
        raise ValueError("Project behavior file is too large.")
    raw = path.read_bytes()
    if len(raw) > _MAX_BYTES:
        raise ValueError("Project behavior file is too large.")
    try:
        saved = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Project behavior file is not valid UTF-8 JSON.") \
            from error
    expected = frozenset((
        "format", "project_type_id", "global", "intents",
        "capabilities", "memory",
    ))
    _check_keys(saved, expected, "Project behavior")
    if saved.get("format") != _FORMAT:
        raise ValueError("Unsupported project behavior format.")
    if set(saved) != expected:
        raise ValueError("Project behavior has missing required settings.")
    if saved.get("project_type_id") != type_id:
        raise ValueError("Project behavior belongs to another project type.")
    _check_override_shape(saved, project_type(type_id))
    return saved, sha256(raw).hexdigest()


def _merged_overrides(saved: dict) -> dict:
    """Map saved scopes into the packaged project-type JSON structure.

    Parameters
    ----------
    saved: dict
        Checked sparse override document.

    Returns
    -------
    dict
        Merge-ready patch for shared, intent, and capability settings.

    """

    result = {
        **saved.get("global", {}),
        "capabilities": saved.get("capabilities", {}),
        "intents": saved.get("intents", {}),
    }
    memory = dict(saved.get("memory", {}))
    if "exclude_disliked_answers" in memory:
        history = result.get("history", {})
        if history == _DELETE:
            history = {}
        result["history"] = {
            **history,
            "exclude_disliked_answers": memory.pop(
                "exclude_disliked_answers"
            ),
        }
    if memory:
        result["conversation_memory"] = memory
    return result


def _resolved(root: Path, type_id: str) -> tuple[ProjectType, str]:
    """Validate the effective project type and return saved revision.

    Parameters
    ----------
    root: pathlib.Path
        Project root.
    type_id: str
        Registered project type ID.

    Returns
    -------
    tuple[ProjectType, str]
        Validated effective configuration and saved revision.

    """

    saved, revision = _state(root, type_id)
    if not any(saved.get(key) for key in (
        "global", "intents", "capabilities", "memory"
    )):
        return project_type(type_id), revision
    path = PROJECT_TYPES_DIR / f"{type_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"),
                     object_pairs_hook=_unique_keys)
    return _parse_project_type(
        type_id, _merge(raw, _merged_overrides(saved))
    ), revision


def effective_project_type(directory: str | Path) -> ProjectType:
    """Return the validated project type with project-local differences.

    Parameters
    ----------
    directory: str | pathlib.Path
        Existing KíLín project workspace.

    Returns
    -------
    ProjectType
        Effective project-specific shared and per-intent behavior.

    """

    project = open_project(directory)
    return _resolved(project.root, project.profile.type_id)[0]


def project_behavior_settings(directory: str | Path) -> dict[str, object]:
    """Return editable effective settings alongside packaged defaults.

    Parameters
    ----------
    directory: str | pathlib.Path
        Existing project workspace.

    Returns
    -------
    dict[str, object]
        Global, capability, memory, and ordered intent editors plus revision.

    """

    project = open_project(directory)
    defaults = project_type(project.profile.type_id)
    effective, revision = _resolved(project.root, defaults.id)
    return {
        "project": str(project.root),
        "project_type_id": defaults.id,
        "revision": revision,
        "global": {
            "values": _global_values(effective),
            "defaults": _global_values(defaults),
        },
        "capabilities": {
            "values": _capability_values(effective),
            "defaults": _capability_values(defaults),
        },
        "memory": {
            "values": _memory_values(effective),
            "defaults": _memory_values(defaults),
        },
        "intents": [
            {
                "id": intent_id,
                "tool": (intent.tool.name if intent.tool is not None
                         else None),
                "values": _intent_values(effective.intents[intent_id]),
                "defaults": _intent_values(intent),
                "memory": {
                    "values": _intent_memory_values(
                        effective.intents[intent_id]
                    ),
                    "defaults": _intent_memory_values(intent),
                },
                "source_first_defaults": {
                    "max_sources": (
                        effective.intents[intent_id]
                        .source_limits().max_sources
                    ),
                    "passages_per_source": (
                        effective.intents[intent_id]
                        .source_limits().passages_per_source
                    ),
                    "max_passages": (
                        effective.intents[intent_id]
                        .source_limits().max_passages
                    ),
                    "evidence_tokens": (
                        effective.intents[intent_id].source_limits().max_tokens
                    ),
                    "coverage": (
                        effective.intents[intent_id].source_limits().coverage
                    ),
                },
            }
            for intent_id, intent in defaults.intents.items()
        ],
    }


def save_project_behavior(
    directory: str | Path,
    scope: str,
    values: dict[str, object],
    *,
    intent_id: str | None = None,
    expected_revision: str | None = None,
    trace: TraceSink | None = None,
) -> dict[str, object]:
    """Persist one complete editor after validating the effective type.

    Parameters
    ----------
    directory: str | pathlib.Path
        Existing project directory.
    scope: str
        ``global``, ``intent``, ``capabilities``, ``memory``, or
        ``intent_memory``. Memory edits never require source reindexing.
    values: dict[str, object]
        Complete editor values excluding tool bindings.
    intent_id: str | None, optional
        Existing intent ID for intent saves. Defaults to None.
    expected_revision: str | None, optional
        Last seen file revision. Defaults to None.
    trace: TraceSink | None, optional
        Trace sink, if any. Defaults to None.

    Returns
    -------
    dict[str, object]
        Fresh editor payload with the saved revision.

    Raises
    ------
    ValueError
        For invalid, stale, or unsupported configuration.

    """

    project = open_project(directory)
    defaults = project_type(project.profile.type_id)
    saved, revision = _state(project.root, defaults.id)
    if expected_revision is not None and expected_revision != revision:
        raise ValueError(
            "Project behavior changed since it was loaded. Reload and "
            "review the latest settings before saving."
        )
    if not isinstance(values, dict):
        raise ValueError("Project behavior settings must be an object.")
    if scope == "global" and intent_id is None:
        if set(values) != _GLOBAL_FIELDS:
            raise ValueError("Supply all project-wide behavior fields.")
        _check_keys(values["history"], frozenset(("recent_turns",)),
                    "Project history")
        saved["global"] = _diff(_global_values(defaults), values)
    elif scope == "capabilities" and intent_id is None:
        if set(values) != _CAPABILITY_FIELDS:
            raise ValueError("Supply all indexing capability fields.")
        saved["capabilities"] = _diff(_capability_values(defaults), values)
    elif scope == "memory" and intent_id is None:
        if set(values) != _MEMORY_FIELDS:
            raise ValueError("Supply all conversation memory fields.")
        saved["memory"] = _diff(_memory_values(defaults), values)
    elif scope == "intent" and intent_id in defaults.intents:
        if set(values) != _INTENT_FIELDS:
            raise ValueError("Supply all editable intent fields.")
        _check_keys(values["history"], frozenset(("policy",)),
                    "Intent history")
        _check_keys(values["retrieval"], _RETRIEVAL_FIELDS,
                    "Intent retrieval")
        changed = _diff(_intent_values(defaults.intents[intent_id]), values)
        intents = dict(saved.get("intents", {}))
        # The existing GUI edits ordinary intent settings without touching
        # the separately owned memory policy saved for the same intent.
        memory = intents.get(intent_id, {}).get("memory")
        updated = dict(changed)
        if memory:
            updated["memory"] = memory
        if updated:
            intents[intent_id] = updated
        else:
            intents.pop(intent_id, None)
        saved["intents"] = intents
    elif scope == "intent_memory" and intent_id in defaults.intents:
        if set(values) != _INTENT_MEMORY_FIELDS:
            raise ValueError("Supply all intent memory fields.")
        intents = dict(saved.get("intents", {}))
        updated = dict(intents.get(intent_id, {}))
        updated.pop("memory", None)
        changed = _diff(
            _intent_memory_values(defaults.intents[intent_id]), values
        )
        if changed:
            updated["memory"] = changed
        if updated:
            intents[intent_id] = updated
        else:
            intents.pop(intent_id, None)
        saved["intents"] = intents
    else:
        raise ValueError(
            "Choose a global, indexing, memory, or intent editor."
        )
    _check_override_shape(saved, defaults)
    path = PROJECT_TYPES_DIR / f"{defaults.id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"),
                     object_pairs_hook=_unique_keys)
    _parse_project_type(defaults.id, _merge(raw, _merged_overrides(saved)))
    state = {
        "format": _FORMAT,
        "project_type_id": defaults.id,
        "global": saved["global"],
        "intents": saved["intents"],
        "capabilities": saved.get("capabilities", {}),
        "memory": saved.get("memory", {}),
    }
    content = json.dumps(state, indent=2, ensure_ascii=False) + "\n"
    if len(content.encode("utf-8")) > _MAX_BYTES:
        raise ValueError("Project behavior settings are too large.")
    atomic_write_text(project.root / PROJECT_BEHAVIOR_FILE, content)
    updated = project_behavior_settings(project.root)
    sink = trace if trace is not None else NullTraceLogger()
    sink.emit(
        "project.behavior.saved", component="project",
        data={"scope": scope, "intent": intent_id,
              "revision": updated["revision"]},
    )
    return updated
