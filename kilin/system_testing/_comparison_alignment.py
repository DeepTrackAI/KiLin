"""Align validated workspace histories without using generated identities.

Key Features
------------
- Align explicit chronological operations and durable identity transitions.
- Compare source outcomes independently of historical external paths.
- Stop at the first ambiguous or different input; never align by answer.

Module Structure
----------------
- ``Alignment`` holds the matched prefix, identity maps, and divergence.
- ``align`` reconstructs correspondences from two validated recordings.

Examples
--------
>>> from kilin.system_testing import Experiment
>>> # comparison = Experiment("010_original").compare()

"""

from dataclasses import dataclass, field

from kilin.repository import _manifest_files

from .timeline import TimelineOperation, one

__all__ = []


@dataclass(slots=True)
class Alignment:
    """Unambiguous operation prefix and independently reconstructed IDs."""

    pairs: list[tuple[TimelineOperation, TimelineOperation]] = field(
        default_factory=list
    )
    conversations: dict[str, str] = field(default_factory=dict)
    turns: dict[str, str] = field(default_factory=dict)
    reason: str | None = None


def _check(condition: bool, message: str) -> None:
    """Reject a mismatch before extending the matched prefix.

    Parameters
    ----------
    condition: bool
        Required correspondence.
    message: str
        Explanation if the correspondence fails.

    """
    if not condition:
        raise ValueError(message)


def _mapped(mapping: dict[str, str], value: str | None) -> str | None:
    """Map a previously identified object or a genuine absent identity.

    Parameters
    ----------
    mapping: dict[str, str]
        Established original-to-replay identities.
    value: str | None
        Recorded original identity.

    Returns
    -------
    str | None
        Corresponding replay identity.

    """
    return None if value is None else mapping[value]


def _register(mapping: dict[str, str], original: str, replayed: str) -> None:
    """Extend a one-to-one identity map only for newly allocated objects.

    Parameters
    ----------
    mapping: dict[str, str]
        Identity map for one object type.
    original: str
        Newly allocated original identity.
    replayed: str
        Newly allocated replay identity.

    """
    _check(original not in mapping and replayed not in mapping.values(),
           "A newly allocated identity was reused.")
    mapping[original] = replayed


def _source_effects(op: TimelineOperation) -> dict:
    """Describe durable source changes without reopening external inputs.

    Parameters
    ----------
    op: TimelineOperation
        Validated command interval.

    Returns
    -------
    dict
        Actual managed copy outcomes, additive sync actions, and manifests.
        A direct file's original basename may differ from its managed name.

    """
    copies = []
    for event in op.matching("workspace.source.copied"):
        data = event.details
        path = data["workspace_path"]
        copied = data["copied_files"]
        preserved = data["preserved_files"]
        if data["kind"] == "file":
            copied = [path] if copied else []
            preserved = [path] if preserved else []
        else:
            copied = sorted(f"{path}/{name}" for name in copied)
            preserved = sorted(f"{path}/{name}" for name in preserved)
        copies.append((path, data["kind"], copied, preserved))
    return {
        "copies": sorted(copies),
        "sync": sorted((e.details["path"], e.details["action"])
                       for e in op.matching("workspace.sync.file")),
        "manifests": [_manifest_files(e.details["source_manifest"])
                      for e in op.matching("index.check.completed",
                                           "index.empty")],
    }


def interaction_records(op: TimelineOperation) -> list[dict]:
    """Pair interaction proposals and outcomes in their original order.

    Parameters
    ----------
    op: TimelineOperation
        Validated question command.

    Returns
    -------
    list[dict]
        Proposals and decisions without regenerated interaction identifiers.

    """
    decisions = {e.details["interaction_id"]: e.details for e in op.matching(
        "routing.choice.decided", "tool.choice.decided", "tool.plan.decided"
    )}
    records = []
    for event in op.matching("routing.choice.proposed", "tool.choice.proposed",
                             "tool.plan.proposed"):
        proposal = event.details
        decision = decisions[proposal["interaction_id"]]
        records.append({
            "kind": event.event_type.removesuffix(".proposed"),
            "proposal": {k: v for k, v in proposal.items()
                         if k != "interaction_id"},
            "decision": {k: v for k, v in decision.items()
                         if k != "interaction_id"},
        })
    return records


def _user_choices(op: TimelineOperation) -> list[dict]:
    """Retain semantic responses rather than numerical option positions.

    Parameters
    ----------
    op: TimelineOperation
        Question with recorded interactive decisions.

    Returns
    -------
    list[dict]
        Explicit user decisions, excluding automatic choices and estimates.

    """
    responses = []
    for record in interaction_records(op):
        kind = record["kind"]
        proposal = record["proposal"]
        decision = record["decision"]
        if not proposal["asked"]:
            continue
        if kind == "tool.plan":
            keys = ["tool", "summary", "scope", "details"]
            if decision["choice"] == "quick":
                keys.extend(("quick_label", "quick_note"))
            if proposal.get("operation") == "network":
                keys.extend(("items", "batches", "batches_to_run",
                             "quick_items"))
            value = {k: proposal.get(k) for k in keys}
            value.update(operation=proposal.get("operation", "model"),
                         choice=decision["choice"])
        else:
            value = {k: proposal[k] for k in ("question", "allow_cancel")}
            value["selected_option"] = decision["selected_option"]
        # Replay uses the same semantic choice protocol for router and
        # tool prompts. A changed producer stage is a recorded observation.
        value["kind"] = kind if kind == "tool.plan" else "choice"
        responses.append(value)
    return responses


def _includes(expected: dict, submitted: dict) -> bool:
    """Check explicit overrides against the submitted nested settings form.

    Parameters
    ----------
    expected: dict
        Recorded sparse overrides; lists are atomic values.
    submitted: dict
        Settings actually submitted in the corresponding request.

    Returns
    -------
    bool
        Whether every expected field is present with the same typed value.

    """
    for key, value in expected.items():
        if key not in submitted:
            return False
        actual = submitted[key]
        if isinstance(value, dict):
            if not isinstance(actual, dict) or not _includes(value, actual):
                return False
        elif type(value) is not type(actual) or value != actual:
            return False
    return True


def _overrides(first: dict, second: dict, explicit_key: str,
               submitted_key: str) -> bool:
    """Allow an explicit value to become equal to a new inherited default.

    Parameters
    ----------
    first: dict
        Original recorded settings request.
    second: dict
        Replay recorded settings request.
    explicit_key: str
        Field holding classified sparse overrides.
    submitted_key: str
        Field holding the actual submitted settings.

    Returns
    -------
    bool
        Whether all original overrides were submitted and the replay has
        no additional explicit override. A changed default can make a
        previously explicit field appear inherited in a GUI full form.

    """
    old, new = first[explicit_key], second[explicit_key]
    return (_includes(old, second[submitted_key]) and _includes(new, old))


def _same_request(before: TimelineOperation, after: TimelineOperation,
                  conversations: dict[str, str],
                  turns: dict[str, str]) -> None:
    """Verify command inputs modulo mapped IDs and inherited settings.

    Parameters
    ----------
    before: TimelineOperation
        Original command.
    after: TimelineOperation
        Candidate corresponding replay command.
    conversations: dict[str, str]
        Established conversation map.
    turns: dict[str, str]
        Established turn map.

    """
    kind = before.kind
    first, second = before.request, after.request
    if kind == "create_project":
        _check(first["source_batch"]["type_id"] ==
               second["source_batch"]["type_id"], "Project type changed.")
        _check(_overrides(first["settings"], second["settings"],
                          "explicit_settings", "requested_settings"),
               "Explicit creation settings differ.")
        return
    if kind in ("add_sources", "sync_sources"):
        # Source paths and unretained no-effect selections can differ.
        # The actual imports and historical manifests are checked separately.
        return
    if kind == "ask":
        keys = ("question", "title_question", "retrieval_count", "effort")
        _check(all(first.get(k) == second.get(k) for k in keys),
               "Question or effort settings differ.")
        _check(_overrides(first, second, "explicit_generation_settings",
                          "generation_settings"),
               "Explicit question overrides differ.")
        _check(_user_choices(before) == _user_choices(after),
               "Semantic user responses differ.")
    elif kind == "create_conversation":
        _check(all(first.get(k) == second.get(k) for k in (
            "branch", "turn_count")), "Requested branch prefix differs.")
    elif kind == "rename_conversation":
        _check(first["name"] == second["name"], "Explicit title differs.")
    elif kind in ("feedback", "memory_exclusion"):
        keys = ("role", "saved", "reaction", "excluded")
        _check(all(first.get(k) == second.get(k) for k in keys),
               "Feedback or exclusion request differs.")
        _check(_mapped(turns, first["turn_uuid"]) == second["turn_uuid"],
               "Feedback refers to another turn.")
    elif kind == "sync_memory":
        _check(first["trigger"] == second["trigger"],
               "Memory synchronization trigger differs.")
    for key in ("conversation_id", "previous_conversation_id",
                "requested_conversation_id"):
        if key in first or key in second:
            _check(_mapped(conversations, first.get(key)) == second.get(key),
                   f"Different conversation transition: {key}.")


def _outcome(before: TimelineOperation, after: TimelineOperation,
             conversations: dict[str, str], turns: dict[str, str]) -> None:
    """Extend mappings from observed durable outcomes, checking lineage.

    Parameters
    ----------
    before: TimelineOperation
        Original operation.
    after: TimelineOperation
        Corresponding replay operation.
    conversations: dict[str, str]
        Candidate conversation map, updated only within this pair.
    turns: dict[str, str]
        Candidate turn map, updated only within this pair.

    """
    kind = before.kind
    if kind == "create_project":
        _register(conversations, before.request["open"]["conversation_id"],
                  after.request["open"]["conversation_id"])
    elif kind in ("create_conversation", "reset_conversation"):
        event_type = ("gui.conversation.create.result" if kind ==
                      "create_conversation" else
                      "gui.conversation.reset.result")
        first = one(before.events, event_type).details
        second = one(after.events, event_type).details
        _check(first.get("reused") == second.get("reused") and
               first.get("origin_turn_count") ==
               second.get("origin_turn_count") and
               _mapped(conversations, first.get("parent_conversation_id")) ==
               second.get("parent_conversation_id"),
               "Conversation creation or branch lineage differs.")
        if first.get("reused"):
            _check(conversations[first["conversation_id"]] ==
                   second["conversation_id"], "A different chat was reused.")
        else:
            _register(conversations, first["conversation_id"],
                      second["conversation_id"])
    elif kind == "delete_conversation":
        first = one(before.events, "gui.conversation.deleted").details
        second = one(after.events, "gui.conversation.deleted").details
        _check(all(first[k] == second[k] for k in (
            "deleted_active", "replacement_created")),
            "Archive or replacement policy differs.")
        old = first["replacement"]["conversation_id"]
        new = second["replacement"]["conversation_id"]
        if first["replacement_created"]:
            _register(conversations, old, new)
        else:
            _check(conversations[old] == new, "Replacement chat differs.")
    elif kind == "ask":
        first = one(before.events, "gui.question.saved").details
        second = one(after.events, "gui.question.saved").details
        _check(first["turn_count"] == second["turn_count"],
               "Committed turn count differs.")
        _register(turns, first["conversation_turn_uuid"],
                  second["conversation_turn_uuid"])
    old = before.matching("conversation.restored")
    new = after.matching("conversation.restored")
    _check(len(old) == len(new), "Restoration count differs.")
    for first, second in zip(old, new):
        _check([turns[uid] for uid in first.details["turn_uuids"]] ==
               second.details["turn_uuids"], "Restored turn prefix differs.")


def align(original: tuple[TimelineOperation, ...],
          replayed: tuple[TimelineOperation, ...]) -> Alignment:
    """Align only the unambiguous common prefix of validated recordings.

    Parameters
    ----------
    original: tuple[TimelineOperation, ...]
        Validated original timeline.
    replayed: tuple[TimelineOperation, ...]
        Independently validated replay timeline.

    Returns
    -------
    Alignment
        Matched prefix, reconstructed IDs, and first mismatch if any.

    """
    result = Alignment()
    for before, after in zip(original, replayed):
        conversations = dict(result.conversations)
        turns = dict(result.turns)
        try:
            _check(before.kind == after.kind, "Command kind differs.")
            _check(before.status == after.status == "completed",
                   "Command is not completed in both recordings.")
            _same_request(before, after, conversations, turns)
            _check(_source_effects(before) == _source_effects(after),
                   "Historical managed source effects differ.")
            _outcome(before, after, conversations, turns)
        except (ValueError, KeyError, TypeError) as error:
            result.reason = str(error)
            break
        result.conversations = conversations
        result.turns = turns
        result.pairs.append((before, after))
    if result.reason is None and len(original) != len(replayed):
        result.reason = "Recorded command counts differ."
    return result
