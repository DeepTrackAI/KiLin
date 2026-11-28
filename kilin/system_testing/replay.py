"""Execute a validated reference history in a separate managed workspace.

Key Features
------------
- Reuse the synchronous engine commands, without launching the frontend.
- Reproduce source batches, question inputs, and actual user decisions.
- Map newly allocated conversation and message identities across branches.
- Report effective settings differences and preserve failed replay attempts.

Module Structure
----------------
- ``ReplayResult`` and ``ReplayDifference`` describe the replay outcome.
- ``ReplayError`` identifies the first operation that could not be reproduced.
- ``replay`` runs a validated experiment using ordinary KíLín operations.

Examples
--------
>>> from kilin.system_testing import Experiment
>>> # result = Experiment("workspaces/010_original").replay()
>>> # result.workspace

"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kilin.conversation_memory.feedback import FeedbackStore

from ._plugin_provenance import compare_plugins, preflight_plugins
from ._replay_interactions import InteractionReplay
from ._replay_sources import SourceReplay
from .timeline import TimelineOperation, build_timeline, one
from .traces import read_recording
from .validation import ValidationReport

if TYPE_CHECKING:
    from .experiment import Experiment

__all__ = ["ReplayResult", "ReplayDifference", "ReplayError", "replay"]

ProgressCallback = Callable[[int, int, TimelineOperation], None]


@dataclass(frozen=True, slots=True)
class ReplayDifference:
    """An observed change in settings or a replay input representation.

    Attributes
    ----------
    sequence: int
        Start of the original operation, not the replay event number.
    field: str
        Dotted configuration or interaction field.
    original: object
        Original recorded value.
    replayed: object
        Actual value or None when the retained input is unavailable.

    """

    sequence: int
    field: str
    original: object
    replayed: object


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Outcome of a complete chronological replay.

    Attributes
    ----------
    reference: pathlib.Path
        Unmodified reference workspace.
    workspace: pathlib.Path
        Independently created replay workspace.
    operation_count: int
        Number of successfully reproduced reference commands.
    conversation_ids: dict[str, str]
        Original-to-replay conversation IDs, including archived records.
    turn_uuids: dict[str, str]
        Original-to-replay committed turn UUIDs, shared across branches.
    differences: tuple[ReplayDifference, ...]
        Configuration, interaction, and source-request differences. These
        are observations, not answer-quality evaluations.
    validation: ValidationReport
        Independent validation of the closed replay workspace.

    """

    reference: Path
    workspace: Path
    operation_count: int
    conversation_ids: dict[str, str]
    turn_uuids: dict[str, str]
    differences: tuple[ReplayDifference, ...]
    validation: ValidationReport


class ReplayError(RuntimeError):
    """A reference command or resulting recording could not be reproduced.

    Attributes
    ----------
    workspace: pathlib.Path
        Requested destination; it may contain a partial replay.
    sequence: int | None
        Original operation start, if the failure occurred in a command.
    completed_operations: int
        Number of commands that completed their immediate checks.
    differences: tuple[ReplayDifference, ...]
        Differences detected up to the failure.
    validation: ValidationReport | None
        Final replay validation if it was reached.

    """

    def __init__(self, workspace: Path, sequence: int | None,
                 completed_operations: int, message: str,
                 differences: tuple[ReplayDifference, ...] = (),
                 validation: ValidationReport | None = None) -> None:
        """Retain failure context without hiding a partial workspace.

        Parameters
        ----------
        workspace: pathlib.Path
            Replay destination.
        sequence: int | None
            Original command sequence, if known.
        completed_operations: int
            Number of successfully checked commands.
        message: str
            Specific reason reproduction stopped.
        differences: tuple[ReplayDifference, ...], optional
            Recorded differences. Defaults to an empty tuple.
        validation: ValidationReport | None, optional
            Final replay validation, if available. Defaults to None.

        """
        self.workspace = workspace
        self.sequence = sequence
        self.completed_operations = completed_operations
        self.differences = differences
        self.validation = validation
        where = f" at original sequence {sequence}" if sequence else ""
        super().__init__(f"Replay failed{where}: {message}")


def _require(condition: bool, message: str) -> None:
    """Enforce a replay outcome instead of silently changing its history.

    Parameters
    ----------
    condition: bool
        Required replay invariant.
    message: str
        Explanation on divergence.

    """
    if not condition:
        raise ValueError(message)


def _destination(reference: Path, destination: str | Path | None) -> Path:
    """Choose an absent sibling workspace without overwriting any path.

    Parameters
    ----------
    reference: pathlib.Path
        Resolved original workspace.
    destination: str | pathlib.Path | None
        Explicit sibling destination or the default replay name.

    Returns
    -------
    pathlib.Path
        New sibling path. The ordinary importer reserves it exclusively.

    """
    if destination is None:
        name = reference.name.removesuffix("_original") + "_replay"
        path = reference.with_name(name)
    else:
        if not str(destination).strip():
            raise ValueError("Choose a replay destination.")
        path = Path(destination).expanduser()
    if path.is_symlink():
        raise ValueError("Replay destination must not be a symbolic link.")
    path = path.resolve()
    if path == reference or path.parent != reference.parent:
        raise ValueError("Replay must be a separate sibling workspace.")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Replay destination already exists: {path}")
    if not path.parent.is_dir():
        raise ValueError("Replay destination parent does not exist.")
    return path


def _current_request(defaults: dict, requested: dict,
                     overrides: dict) -> dict:
    """Reproduce the submitted form's shape with current inherited values.

    Parameters
    ----------
    defaults: dict
        Current full settings form.
    requested: dict
        Historical submitted form, including its partial backend mappings.
    overrides: dict
        Historical explicit fields; lists are atomic settings.

    Returns
    -------
    dict
        Current values for the submitted fields and recorded overrides.

    Notes
    -----
    Displayed defaults contain fields never submitted by the frontend.
    Do not write those fields, or inactive backend settings, merely because
    they appeared in the defaults snapshot.

    """
    result = {}
    for key, value in requested.items():
        if isinstance(value, dict) and isinstance(defaults.get(key), dict):
            result[key] = _current_request(
                defaults[key], value, overrides.get(key, {})
            )
        elif key in overrides:
            result[key] = deepcopy(overrides[key])
        else:
            _require(key in defaults, f"Inherited setting unavailable: {key}")
            result[key] = deepcopy(defaults[key])
    return result


def _differences(sequence: int, field: str, original: object,
                 replayed: object) -> list[ReplayDifference]:
    """Compare JSON-compatible configuration without model judgment.

    Parameters
    ----------
    sequence: int
        Original operation start.
    field: str
        Current dotted setting name.
    original: object
        Recorded original value.
    replayed: object
        Observed replay value.

    Returns
    -------
    list[ReplayDifference]
        Changed fields; lists remain atomic, as in settings provenance.

    """
    if isinstance(original, dict) and isinstance(replayed, dict):
        changes = []
        for key in sorted(original.keys() | replayed.keys()):
            name = f"{field}.{key}"
            if key not in original or key not in replayed:
                changes.append(ReplayDifference(
                    sequence, name, original.get(key), replayed.get(key)
                ))
            else:
                changes.extend(_differences(
                    sequence, name, original[key], replayed[key]
                ))
        return changes
    if original != replayed or type(original) is not type(replayed):
        return [ReplayDifference(sequence, field, original, replayed)]
    return []


def _configuration_differences(
    original: tuple[TimelineOperation, ...],
    reproduced: tuple[TimelineOperation, ...],
) -> list[ReplayDifference]:
    """Compare actual effective settings at matching command boundaries.

    Parameters
    ----------
    original: tuple[TimelineOperation, ...]
        Reference operations.
    reproduced: tuple[TimelineOperation, ...]
        Complete or partial replay operations in command order.

    Returns
    -------
    list[ReplayDifference]
        Observed setting changes for the unambiguous common prefix.

    """
    changes = []
    for before, after in zip(original, reproduced):
        if before.kind != after.kind:
            break
        for event_type, field in (
            ("gui.project.configuration.effective", "configuration"),
            ("gui.question.settings.resolved", "question.resolved"),
            ("turn.settings.snapshot", "question.effective"),
        ):
            old, new = before.matching(event_type), after.matching(event_type)
            if len(old) != 1 or len(new) != 1:
                continue
            first, second = deepcopy(old[0].details), deepcopy(new[0].details)
            # The workspace is intentionally relocated. Its absolute source
            # path and the behavior editor's project ID are not settings.
            if event_type == "gui.project.configuration.effective":
                for values in (first, second):
                    values.pop("plugins", None)
                    if isinstance(values.get("behavior"), dict):
                        values["behavior"].pop("project", None)
            elif event_type == "turn.settings.snapshot":
                for values in (first, second):
                    values.pop("source", None)
                    values.pop("project_type_plugin", None)
            changes.extend(_differences(before.start_sequence, field,
                                        first, second))
    return changes


def _plugin_differences(
    original: tuple[TimelineOperation, ...],
    reproduced: tuple[TimelineOperation, ...],
) -> list[ReplayDifference]:
    """Report observed identity changes separately from setting changes.

    Parameters
    ----------
    original: tuple[TimelineOperation, ...]
        Reference operations.
    reproduced: tuple[TimelineOperation, ...]
        Complete or partial replay operations in command order.

    Returns
    -------
    list[ReplayDifference]
        Known differences in the common command prefix. Missing historical
        identities are not inferred differences; comparison reports retain
        their explicit unknown status.

    """

    changes = []
    for before, after in zip(original, reproduced):
        if before.kind != after.kind:
            break
        for pair in compare_plugins(before, after):
            if pair["status"] == "changed":
                field = ".".join(("plugins", pair["scope"], pair["kind"],
                                  pair["name"]))
                changes.extend(_differences(
                    before.start_sequence, field,
                    pair["original"]["identity"],
                    pair["replayed"]["identity"],
                ))
    return changes


class _Runner:
    """Coordinate one replay, using the same synchronous engine as the GUI."""

    def __init__(self, workspace: Path,
                 operations: tuple[TimelineOperation, ...],
                 sources: SourceReplay) -> None:
        """Create an idle headless command coordinator.

        Parameters
        ----------
        workspace: pathlib.Path
            New sibling replay destination.
        operations: tuple[TimelineOperation, ...]
            Validated operation history.
        sources: SourceReplay
            Preflighted additive source input plans.

        """
        # Import the engine only for execution, not reference inspection.
        from kilin.gui.engine import ChatEngine

        self.workspace = workspace
        self.operations = operations
        self.sources = sources
        self.interactions = InteractionReplay()
        self.conversations: dict[str, str] = {}
        self.turns: dict[str, str] = {}
        self.differences: list[ReplayDifference] = []
        self.completed = 0
        self.sequence: int | None = None
        self.events: set[str] = set()
        self.error: str | None = None
        self.engine = ChatEngine(workspace.parent, self._observe,
                                 trace_enabled=True,
                                 read_command=self.interactions.read_command)

    def _observe(self, event: dict) -> None:
        """Consume acknowledgements without retaining token streams.

        Parameters
        ----------
        event: dict
            Synchronous engine event, including its prompt or error payload.

        """
        self.interactions.observe(event)
        kind = event["event"]
        self.events.add(kind)
        if self.error is None and (
            kind in ("error", "workspace_conflict", "backend_failed",
                     "answer_cancelled", "memory_sync_failed")
            or kind in ("message_feedback", "memory_exclusion",
                        "memory_status") and event.get("error")
        ):
            self.error = str(event.get("error") or event.get("message")
                             or kind)

    def _call(self, method: Callable, *args: object,
              expected: str | None = None, **kwargs: object) -> None:
        """Invoke a command and check errors caught by the engine itself.

        Parameters
        ----------
        method: Callable
            Ordinary synchronous engine command.
        *args: object
            Positional command arguments.
        expected: str | None, optional
            Required successful acknowledgement. None if no such event is
            defined. Defaults to None.
        **kwargs: object
            Keyword command arguments.

        """
        self.events.clear()
        self.error = None
        method(*args, **kwargs)
        _require(self.error is None, self.error or "Engine command failed.")
        _require(expected is None or expected in self.events,
                 f"Missing success acknowledgement: {expected}")

    def _active(self) -> str:
        """Return the live replay conversation identity."""
        item = self.engine.active_conversation
        _require(item is not None and self.engine.session is not None,
                 "Replay has no active conversation.")
        return item.conversation_id

    def _catalog(self) -> dict:
        """Read the active replay store, including unused empty chats."""
        _require(self.engine.conversation_store is not None,
                 "Replay has no conversation store.")
        return {item.conversation_id: item for item in
                self.engine.conversation_store.conversations()}

    def _ids(self) -> list[str]:
        """Return the currently loaded committed turn identities."""
        self._active()
        return [turn.conversation_turn_uuid for turn in
                self.engine.session.turns]

    def _register(self, original: str, actual: str) -> None:
        """Register a newly allocated conversation without reusing identity.

        Parameters
        ----------
        original: str
            Reference conversation ID.
        actual: str
            Independently allocated replay conversation ID.

        """
        _require(original not in self.conversations and
                 actual not in self.conversations.values(),
                 "Conversation identity reused unexpectedly.")
        self.conversations[original] = actual

    def _check_active(self, original: str) -> None:
        """Require the recorded conversation to be active via its new ID.

        Parameters
        ----------
        original: str
            Original active conversation ID.

        """
        _require(self._active() == self.conversations[original],
                 "Active conversation differs from recorded transition.")

    def _create(self, op: TimelineOperation) -> None:
        """Open a new workspace with current defaults and explicit overrides.

        Parameters
        ----------
        op: TimelineOperation
            Original creation with source outcomes and settings provenance.

        """
        request = op.request
        type_id = request["source_batch"]["type_id"]
        provenance = request["settings"]
        explicit = provenance["explicit_settings"]
        if provenance["provenance"] == "displayed_defaults":
            # The existing blank-workspace settings command can borrow
            # saved preferences from a profile in the working directory.
            # Do not let that silently change the inherited replay inputs.
            _require(not (Path.cwd() / "profile.ini").is_file(),
                     "Run replay from a directory without profile.ini.")
            # Reuse the same default form that the current frontend uses.
            self._call(self.engine.settings, type_id, "", expected="settings")
            defaults = self.engine._creation_defaults
            _require(defaults is not None and defaults[0] == type_id,
                     "Current creation defaults are unavailable.")
            settings = _current_request(
                defaults[1], provenance["requested_settings"], explicit
            )
            settings["_gui_full_snapshot"] = True
        else:
            settings = deepcopy(explicit)
        source = self.sources.commands[op.start_sequence]
        self._call(self.engine.open_project, source.sources, type_id,
                   str(self.workspace), selected_files=source.selections,
                   settings=settings, expected="opened")
        _require(self.engine.session.project.root == self.workspace,
                 "Engine opened an unexpected workspace.")
        active = self._active()
        _require(set(self._catalog()) == {active} and not self._ids(),
                 "New replay did not start with one empty conversation.")
        self._register(request["open"]["conversation_id"], active)

    def _open(self, op: TimelineOperation) -> None:
        """Reopen using the original explicit or inherited selection policy.

        A validated relocation is reproduced as a normal resume with the
        mapped active conversation, not as a physical filesystem rename.

        Parameters
        ----------
        op: TimelineOperation
            Recorded resume operation.

        """
        selected = op.request["requested_conversation_id"]
        self._call(self.engine.resume_project, str(self.workspace),
                   self.conversations[selected] if selected else None,
                   expected="opened")
        self._check_active(op.request["conversation_id"])
        if "relocation" in op.request:
            self.differences.append(ReplayDifference(
                op.start_sequence, "project.relocation",
                op.request["relocation"],
                {"directory": str(self.workspace),
                 "policy": "retain_destination_directory"},
            ))

    def _ask(self, op: TimelineOperation) -> None:
        """Reissue a question and map its newly committed turn.

        Parameters
        ----------
        op: TimelineOperation
            Original question, settings provenance and interaction history.

        """
        request = op.request
        self._check_active(request["conversation_id"])
        before = self._ids()
        explicit = request["explicit_generation_settings"]
        if request["generation_provenance"] == "profile_difference":
            profile = self.engine.session.project.profile
            settings = {key: getattr(profile, key) for key in
                        request["generation_settings"]}
            settings.update(explicit)
            settings["_title_question"] = request["title_question"]
        else:
            settings = deepcopy(explicit)
        self.interactions.begin(op)
        try:
            self._call(self.engine.ask, request["question"],
                       retrieval_count=request["retrieval_count"],
                       generation_settings=settings,
                       effort=deepcopy(request["effort"]), expected="answered")
            self.interactions.finish()
        finally:
            self.differences.extend(ReplayDifference(
                op.start_sequence, field, old, new
            ) for field, old, new in self.interactions.differences)
        self._check_active(request["conversation_id"])
        after = self._ids()
        _require(len(after) == len(before) + 1 and after[:-1] == before,
                 "Question did not commit exactly one new turn.")
        expected = one(op.events, "gui.question.saved").details
        original = expected["conversation_turn_uuid"]
        _require(len(after) == expected["turn_count"] and
                 original not in self.turns and
                 after[-1] not in self.turns.values() and
                 self.engine.session.turns[-1].question ==
                 request["question"].strip(),
                 "Committed question or turn identity differs from history.")
        self.turns[original] = after[-1]

    def _create_conversation(self, op: TimelineOperation) -> None:
        """Reproduce a new chat, empty-chat reuse, branch or reset.

        Parameters
        ----------
        op: TimelineOperation
            Recorded conversation creation or reset operation.

        """
        request = op.request
        self._check_active(request["previous_conversation_id"])
        previous = self._active()
        before = set(self._catalog())
        inherited = []
        if op.kind == "reset_conversation":
            result = one(op.events, "gui.conversation.reset.result").details
            self._call(self.engine.reset, expected="reset")
        else:
            result = one(op.events, "gui.conversation.create.result").details
            if request["branch"]:
                inherited = self._ids()[:request["turn_count"]]
            self._call(self.engine.create_conversation,
                       branch=request["branch"],
                       turn_count=request["turn_count"],
                       expected="conversation_opened")
        active = self._active()
        after = self._catalog()
        original = result["conversation_id"]
        if result.get("reused"):
            self._check_active(original)
            _require(set(after) == before, "Empty-chat reuse created a chat.")
        else:
            _require(set(after) == before | {active} and active not in before,
                     "Conversation creation did not allocate one new chat.")
            self._register(original, active)
        parent = previous if request.get("branch") else None
        _require(after[active].parent_conversation_id == parent and
                 self._ids() == inherited,
                 "Conversation did not inherit the recorded turn prefix.")

    def _delete(self, op: TimelineOperation) -> None:
        """Archive a mapped chat and check the actual replacement policy.

        Parameters
        ----------
        op: TimelineOperation
            Recorded deletion, archive, and replacement metadata.

        """
        target = self.conversations[op.request["conversation_id"]]
        before = set(self._catalog())
        result = one(op.events, "gui.conversation.deleted").details
        _require((target == self._active()) == result["deleted_active"],
                 "Deletion targets a different active conversation.")
        self._call(self.engine.delete_conversation, target)
        active = self._active()
        expected = before - {target}
        original = result["replacement"]["conversation_id"]
        if result["replacement_created"]:
            _require(active not in before, "No new deletion replacement.")
            self._register(original, active)
            expected.add(active)
        self._check_active(original)
        _require(set(self._catalog()) == expected,
                 "Deletion changed the conversation catalog unexpectedly.")

    def _feedback(self, op: TimelineOperation) -> None:
        """Replay message feedback using the new UUID and current revision.

        Parameters
        ----------
        op: TimelineOperation
            Feedback or exclusion mutation with its committed result.

        """
        request = op.request
        store = FeedbackStore(self.workspace)
        data = {"project": str(self.workspace),
                "conversation_id": self.conversations[
                    request["conversation_id"]],
                "turn_uuid": self.turns[request["turn_uuid"]],
                "role": request["role"], "revision": store.load().revision,
                "request_id": op.start_sequence}
        if op.kind == "feedback":
            data.update(saved=request["saved"], reaction=request["reaction"])
            self._call(self.engine.save_message_feedback, data,
                       expected="message_feedback")
            event_type = "gui.conversation.feedback.saved"
        else:
            data["excluded"] = request["excluded"]
            self._call(self.engine.set_memory_exclusion, data,
                       expected="memory_exclusion")
            event_type = "gui.conversation.memory_exclusion.saved"
        rating = store.load().get(data["turn_uuid"], data["role"])
        recorded = one(op.events, event_type).details["rating"]
        _require(asdict(rating) == recorded,
                 "Feedback mutation differs from the recorded result.")

    def _execute(self, op: TimelineOperation) -> None:
        """Dispatch one validated operation without executing recorded code.

        Parameters
        ----------
        op: TimelineOperation
            Next chronological reference command.

        """
        kind, request = op.kind, op.request
        if kind == "create_project":
            self._create(op)
        elif kind == "open_project":
            self._open(op)
        elif kind == "close_project":
            self._check_active(request["conversation_id"])
            self._call(self.engine.close)
            _require(self.engine.session is None, "Project did not close.")
        elif kind == "ask":
            self._ask(op)
        elif kind in ("add_sources", "sync_sources"):
            command = self.sources.commands[op.start_sequence]
            if kind == "add_sources":
                self._call(self.engine.add_project_sources, command.sources,
                           decisions=command.decisions,
                           selected_files=command.selections or None,
                           expected="opened")
            else:
                self._call(self.engine.sync_sources, command.decisions,
                           expected="opened")
        elif kind in ("create_conversation", "reset_conversation"):
            self._create_conversation(op)
        elif kind == "select_conversation":
            self._call(self.engine.select_conversation,
                       self.conversations[request["conversation_id"]])
            self._check_active(request["conversation_id"])
        elif kind == "rename_conversation":
            target = self.conversations[request["conversation_id"]]
            self._call(self.engine.rename_conversation, target,
                       request["name"],
                       expected="conversations")
            name = one(op.events, "gui.conversation.renamed").details["name"]
            _require(self._catalog()[target].name == name,
                     "Explicit conversation name differs from recording.")
        elif kind == "delete_conversation":
            self._delete(op)
        elif kind in ("feedback", "memory_exclusion"):
            self._feedback(op)
        elif kind == "sync_memory":
            self._call(self.engine.sync_memory_index, str(self.workspace),
                       expected="memory_synced")
        elif kind == "regenerate_source_cards":
            self._call(self.engine.regenerate_source_cards,
                       expected="source_cards_regenerated")
        else:
            raise ValueError(f"Unsupported replay operation: {kind}")
        self.sources.verify(self.workspace, op)
        command = self.sources.commands.get(op.start_sequence)
        if command is not None:
            self.differences.extend(ReplayDifference(
                op.start_sequence, "source.unretained_request", item, None
            ) for item in command.omitted)
        for restored in op.matching("conversation.restored"):
            expected = [self.turns[uid] for uid in
                        restored.details["turn_uuids"]]
            _require(self._ids() == expected,
                     "Restored replay history differs from mapped turns.")

    def run(self, progress: ProgressCallback | None) -> None:
        """Execute the chronological operations, stopping at first failure.

        Parameters
        ----------
        progress: ProgressCallback | None
            Optional callback before each command (one-based index, total,
            original operation). No callback when None.

        """
        total = len(self.operations)
        try:
            for i, op in enumerate(self.operations, 1):
                self.sequence = op.start_sequence
                if progress is not None:
                    progress(i, total, op)
                self._execute(op)
                self.completed += 1
        except BaseException as error:
            # Never delete a partial replay or let a secondary close failure
            # obscure the operation that first diverged.
            try:
                self.engine.close()
            except BaseException as cleanup:
                error.add_note(f"Replay cleanup also failed: {cleanup!r}")
            raise
        else:
            self.engine.close()


def replay(experiment: Experiment, destination: str | Path | None = None,
           *, progress: ProgressCallback | None = None) -> ReplayResult:
    """Replay a closed reference into one independently created workspace.

    Parameters
    ----------
    experiment: Experiment
        Original reference, validated before any model or destination opens.
    destination: str | pathlib.Path | None, optional
        Absent sibling directory. Defaults to replacing a trailing
        ``_original`` with ``_replay``, or appending ``_replay`` otherwise.
    progress: ProgressCallback | None, optional
        Called before each operation with its one-based index, total count,
        and original TimelineOperation. Defaults to None.

    Returns
    -------
    ReplayResult
        Closed, independently validated replay and identity mappings.

    Raises
    ------
    InvalidRecordingError
        If the original cannot be validated.
    FileExistsError
        If the requested replay destination already exists.
    ValueError
        If the destination is not a separate sibling.
    ReplayError
        If replay inputs, command outcomes, or final validation diverge.

    Notes
    -----
    Uses current defaults for inherited settings and recorded explicit
    overrides for user settings. Only live matching user decisions are
    supplied. Models and explicitly approved network actions run normally;
    the original answers, outputs, templates and indexes are not copied.
    The original must remain closed and unchanged throughout replay.

    """
    root = experiment.workspace
    target = _destination(root, destination)
    report = experiment.validate()
    report.raise_for_errors()
    operations = report.operations
    try:
        preflight_plugins(operations)
        sources = SourceReplay(root, operations)
        runner = _Runner(target, operations, sources)
    except Exception as error:
        raise ReplayError(target, None, 0, str(error)) from error
    try:
        runner.run(progress)
    except Exception as error:
        # Read partial diagnostics only when the replay actually exists.
        # They are evidence, not permission to resume or overwrite it.
        if target.is_dir():
            try:
                partial = build_timeline(read_recording(target))
                runner.differences.extend(_configuration_differences(
                    operations, partial
                ))
                runner.differences.extend(_plugin_differences(
                    operations, partial
                ))
            except Exception as inspection:
                error.add_note(f"Partial inspection failed: {inspection!r}")
        raise ReplayError(target, runner.sequence, runner.completed,
                          str(error), tuple(runner.differences)) from error
    from .experiment import Experiment

    validation = Experiment(target).validate()
    runner.differences.extend(_configuration_differences(
        operations, validation.operations
    ))
    runner.differences.extend(_plugin_differences(
        operations, validation.operations
    ))
    if (not validation.valid or [op.kind for op in validation.operations] !=
            [op.kind for op in operations]):
        message = (validation.issues[0].message if validation.issues else
                   "Replay command sequence differs from reference.")
        raise ReplayError(target, None, runner.completed, message,
                          tuple(runner.differences), validation)
    return ReplayResult(root, target, runner.completed,
                        dict(runner.conversations), dict(runner.turns),
                        tuple(runner.differences), validation)
