"""Compare independent recorded workspaces without replaying either one.

Key Features
------------
- Validate both recordings and reconstruct their operation correspondences.
- Compare configuration, sources, saved answers, tools, and trace stages.
- Return a portable structured report with optional local answer embeddings.

Module Structure
----------------
- ``Comparison`` provides the report and presentation methods.
- ``compare`` performs fresh, read-only inspection of two workspace trees.

Examples
--------
>>> from kilin.system_testing import Experiment
>>> # comparison = Experiment("010_original").compare()
>>> # comparison.display()
>>> # comparison.save()

"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from kilin.conversation_memory.feedback import FeedbackStore

from ._comparison_alignment import align, interaction_records
from ._comparison_diagnostics import (
    context_comparison, independent_questions, structured_prompt,
    tool_comparison,
)
from ._comparison_metrics import (
    EmbeddingEvaluator, evidence_metrics, file_changes, inventory,
    inference_comparison, inference_summary, model_calls, operation_duration,
    performance_summary, ranking_metrics, stage_metrics, text_metrics, usage,
)
from ._comparison_report import markdown_report, save_reports
from ._inspection import note_inspection, relocation_event_differences
from ._plugin_provenance import compare_plugins, plugin_observations
from .replay import _configuration_differences
from .reference import inspect_receipt
from .replay_management import (
    execution_provenance, reference_execution_match,
)
from .timeline import TimelineOperation, one
from .traces import Recording, read_recording
from .validation import ValidationReport, validate_recording

if TYPE_CHECKING:
    from .experiment import Experiment

__all__ = ["Comparison", "compare"]


@dataclass(frozen=True, slots=True)
class Comparison:
    """One independently computed comparison, not a quality verdict.

    Attributes
    ----------
    reference: pathlib.Path
        Original workspace, unchanged by comparison.
    workspace: pathlib.Path
        Replayed workspace, unchanged by comparison.
    data: dict
        JSON-compatible report snapshot, including validation, alignment,
        identity maps, file inventories, and individual observations.

    """

    reference: Path
    workspace: Path
    data: dict

    @property
    def complete(self) -> bool:
        """Return whether the entire validated histories could be aligned."""
        return self.data["alignment"]["status"] == "complete"

    def markdown(self) -> str:
        """Return a human-readable report; JSON retains the detailed data."""
        return markdown_report(self.data)

    def display(self) -> None:
        """Display the Markdown report in a notebook, or print it in a CLI."""
        content = self.markdown()
        try:
            from IPython import get_ipython
            from IPython.display import Markdown, display
        except ImportError:
            print(content)
        else:
            if get_ipython() is None:
                print(content)
            else:
                display(Markdown(content))

    def save(self, *, overwrite: bool = False) -> tuple[Path, Path]:
        """Save JSON and Markdown reports beside the workspace directories.

        Parameters
        ----------
        overwrite: bool, optional
            Permit replacement of existing regular report files only.
            Defaults to False. No workspace contents are replaced.

        Returns
        -------
        tuple[pathlib.Path, pathlib.Path]
            JSON and Markdown report paths, in that order.

        """
        return save_reports(self.reference, self.data, self.markdown(),
                            overwrite=overwrite)


def _replay_path(root: Path, destination: str | Path | None) -> Path:
    """Resolve a separate sibling, allowing a missing replay for diagnosis.

    Parameters
    ----------
    root: pathlib.Path
        Resolved original workspace.
    destination: str | pathlib.Path | None
        Explicit replay directory or the default sibling.

    Returns
    -------
    pathlib.Path
        Replay directory; no directories are created.

    """
    if destination is None:
        path = root.with_name(root.name.removesuffix("_original") + "_replay")
    else:
        if not str(destination).strip():
            raise ValueError("Choose a replay workspace.")
        path = Path(destination).expanduser()
    if path.is_symlink():
        raise ValueError("Replay comparison does not follow workspace links.")
    path = path.resolve()
    if path == root or path.parent != root.parent:
        raise ValueError("Compare a separate sibling replay workspace.")
    if (path.exists() and not path.is_dir()) or (
            root.exists() and not root.is_dir()):
        raise ValueError("Workspace paths must be directories.")
    if path.exists() and root.exists() and path.samefile(root):
        raise ValueError("Cannot compare the original workspace to itself.")
    return path


def _saved_turns(recording: Recording) -> dict[str, dict]:
    """Index validated saved turns once, including inherited and archived.

    Parameters
    ----------
    recording: Recording
        Validated snapshot with canonical conversation states.

    Returns
    -------
    dict[str, dict]
        Unique committed turn UUIDs, not branch-local turn numbers.

    """
    return {turn["conversation_turn_uuid"]: turn
            for state in recording.states.values()
            for turn in state.get("turns", [])}


def _catalog(recording: Recording) -> list[dict]:
    """Describe the final active/archive catalog without changing names.

    Parameters
    ----------
    recording: Recording
        Validated recording with authoritative states.

    Returns
    -------
    list[dict]
        Trace locations, final names, lineage and inherited turn identities.

    """
    return [{"path": path, "id": state["conversation_id"],
             "name": state["name"],
             "archived": Path(path).parent.name == ".deleted",
             "parent_id": state.get("parent_conversation_id"),
             "origin_turn_count": state.get("origin_turn_count"),
             "origin_checkpoint_id": state.get("origin_checkpoint_id"),
             "turn_uuids": [t["conversation_turn_uuid"]
                            for t in state.get("turns", [])]}
            for path, state in sorted(recording.states.items())]


def _workspace(recording: Recording, report: ValidationReport) -> dict:
    """Collect independent workspace observations and validation diagnostics.

    Parameters
    ----------
    recording: Recording
        Fresh trace snapshot.
    report: ValidationReport
        Validation of that same snapshot.

    Returns
    -------
    dict
        Candidate command inventory, validation, recorded environment/model
        provenance, and authoritative conversation data only when validated.

    """
    environments = {}
    models = []
    for event in report.events:
        if event.event_type == "trace.started":
            environment = event.details.get("environment")
            if isinstance(environment, dict):
                environments[json.dumps(environment, sort_keys=True)] = (
                    environment
                )
        if event.event_type.startswith("model.") and (
                event.event_type.endswith(".resolved")):
            models.append({"sequence": event.sequence,
                           "type": event.event_type,
                           "details": event.details})
    assigned = {event.sequence for op in report.operations
                for event in op.events}
    unassigned = [event for event in report.events
                  if event.sequence not in assigned]
    result = {
        "path": str(recording.root), "valid": report.valid,
        "event_count": report.event_count,
        "validation_issues": [asdict(issue) for issue in report.issues],
        "validation_warnings": [asdict(w) for w in report.warnings],
        "operations": [{"kind": op.kind, "sequence": op.start_sequence,
                        "end_sequence": op.end_sequence, "status": op.status}
                       for op in report.operations],
        "operation_counts": dict(sorted(Counter(
            op.kind for op in report.operations).items())),
        "environments": list(environments.values()), "models": models,
        "plugin_provenance": plugin_observations(report.events),
        # A title generation occurs outside the outer question span. Retain
        # this diagnostic overhead without inventing a user operation.
        "outside_commands": {
            "event_counts": dict(sorted(Counter(e.event_type for e in
                                                unassigned).items())),
            "generated_titles": [{"sequence": e.sequence,
                                  "details": e.details} for e in unassigned
                                 if e.event_type ==
                                 "conversation.title.generated"],
        },
        "traced_chat_usage": _traced_usage(report),
        "event_counts": dict(sorted(Counter(e.event_type for e in
                                            report.events).items())),
        "saved_questions": independent_questions(recording, report),
        "conversations": None, "turn_count": None, "usage": None,
        "feedback": None, "configuration": None,
    }
    if report.valid:
        turns = _saved_turns(recording)
        names = {}
        for event in report.events:
            if event.event_type in ("gui.conversation.renamed",
                                    "conversation.title.generated"):
                uid = event.details.get("conversation_id")
                if isinstance(uid, str):
                    names[uid] = ("explicit" if event.event_type ==
                                  "gui.conversation.renamed" else "generated")
        catalog = _catalog(recording)
        for item in catalog:
            item["name_origin"] = names.get(item["id"], "unclassified")
        feedback = FeedbackStore(recording.root).load()
        counts = [usage(turn) for turn in turns.values()]
        result.update(conversations=catalog, turn_count=len(turns),
                      usage={key: {
                          "recorded_total": (sum(v[key] for v in counts
                                                 if v[key] is not None)
                                             if any(v[key] is not None
                                                    for v in counts)
                                             else None),
                          "available_turns": sum(v[key] is not None
                                                 for v in counts),
                          "total_turns": len(turns),
                      } for key in ("input_tokens", "output_tokens")},
                      feedback={uid: {role: asdict(rating) for role, rating
                                      in values.items()} for uid, values
                                in feedback.entries.items()},
                      configuration=[{"sequence": e.sequence,
                                      "details": e.details}
                                     for e in report.events
                                     if e.event_type ==
                                     "gui.project.configuration.effective"])
    return result


def _traced_usage(report: ValidationReport) -> dict:
    """Count recorded chat call tokens, separately from saved turn usage.

    Parameters
    ----------
    report: ValidationReport
        Decoded workspace diagnostics, including events outside commands.

    Returns
    -------
    dict
        Available token totals and coverage by call-input/output record.
        This is not a second sum to add to the saved chat counters.

    """
    result = {}
    for key, event_type, field in (
        ("input_tokens", "llm.call.input", "input_token_count"),
        ("output_tokens", "llm.call.output", "output_token_count"),
    ):
        records = [e.details.get(field) for e in report.events
                   if e.event_type == event_type]
        samples = [value for value in records
                   if type(value) is int and value >= 0]
        result[key] = {"recorded_total": sum(samples) if samples else None,
                       "available_records": len(samples),
                       "total_records": len(records)}
    return result


def _call_rows(op: TimelineOperation) -> list[dict]:
    """Annotate call rows by enclosing stage before occurrence comparison.

    Parameters
    ----------
    op: TimelineOperation
        One aligned recorded command.

    Returns
    -------
    list[dict]
        Call observations grouped by enclosing span type and occurrence.

    """
    events = {e.sequence: e for e in op.events}
    starts = {(e.metadata["trace_id"], e.span_id): e.event_type
              for e in op.events if e.span_id and
              e.event_type.endswith(".started") and
              "started_elapsed_ms" in e.metadata}
    rows = model_calls(op)
    for row in rows:
        event = events[row["input_sequence"] or row["output_sequence"]]
        row["stage"] = starts.get((event.metadata["trace_id"], event.span_id),
                                  "unscoped")
    return rows


def _call_comparison(original: TimelineOperation,
                     replayed: TimelineOperation, maps: tuple) -> list[dict]:
    """Compare available exact model texts by stage and occurrence.

    Parameters
    ----------
    original: TimelineOperation
        Original operation.
    replayed: TimelineOperation
        Corresponding replay operation.
    maps: tuple
        Original-to-replay conversation and turn identity maps.

    Returns
    -------
    list[dict]
        Paired observations or unavailable sides, text metrics where both
        artifacts exist, and text hashes instead of duplicated prompts.

    """
    conversations, turns = maps
    namespaces = (({k: k for k in turns}, {k: k for k in conversations}),
                  ({v: k for k, v in turns.items()},
                   {v: k for k, v in conversations.items()}))
    groups = []
    for op, namespace in zip((original, replayed), namespaces, strict=True):
        values = {}
        inputs = {e.sequence: e for e in op.matching("llm.call.input")}
        for row in _call_rows(op):
            row["structured_prompt"] = structured_prompt(
                inputs.get(row["input_sequence"]), op, *namespace)
            values.setdefault(row["stage"], []).append(row)
        groups.append(values)
    first, second = groups
    rows = []
    for stage in sorted(first.keys() | second.keys()):
        a, b = first.get(stage, []), second.get(stage, [])
        for i in range(max(len(a), len(b))):
            old = a[i] if i < len(a) else None
            new = b[i] if i < len(b) else None
            metrics = {}
            structures = [row.pop("structured_prompt", None) if row else None
                          for row in (old, new)]
            for key in ("raw", "normalized"):
                a_text, b_text = [entry.get(key) if entry else None
                                  for entry in structures]
                metrics["structured_" + key + "_equal"] = (
                    a_text == b_text if a_text is not None and
                    b_text is not None else None)
            for row, structure in zip((old, new), structures, strict=True):
                if row is not None:
                    row["prompt_metadata"] = (
                        {k: structure[k] for k in ("headers", "issue")}
                        if structure is not None else None)
            for key in ("prompt", "answer"):
                left = old.pop(key, None) if old else None
                right = new.pop(key, None) if new else None
                metrics[key] = (text_metrics(left, right)
                                if left is not None and right is not None
                                else None)
                for row, text in ((old, left), (new, right)):
                    if row is not None:
                        row[key + "_sha256"] = (sha256(text.encode("utf-8")
                                                       ).hexdigest()
                                                 if text is not None else None)
            rows.append({"stage": stage, "occurrence": i + 1,
                         "original": old, "replayed": new,
                         "text": metrics,
                         "inference": inference_comparison(old, new, metrics)})
    return rows


def _observations(op: TimelineOperation) -> dict:
    """Extract semantic stage observations without comparing generated IDs.

    Parameters
    ----------
    op: TimelineOperation
        A validated command.

    Returns
    -------
    dict
        Selected original diagnostic payloads for manual interpretation.

    """
    types = (
        "intent.classification.decided", "history.selection.result",
        "retrieval.plan", "retrieval.result", "evidence.finalized",
        "retrieval.document.selected", "retrieval.sources.ranked",
        "retrieval.context_expansion.planned", "memory.sync.result",
        "generation.input.prepared", "generation.incomplete",
        "conversation_memory.retrieved", "project.memory.sync.result",
        "index.rebuilt", "index.reused", "index.empty", "tool.run.started",
        "source_cards.prepared", "project.source_cards.regenerate.completed",
        "workspace.source.copied", "workspace.sync.file",
        "workspace.sync.started", "workspace.sync.completed",
        "gui.sources.add.requested", "gui.sources.add.completed",
    )
    rows = {}
    for event in op.events:
        kind = event.event_type
        if kind in types or (kind.startswith("tool.") and
                             kind.endswith(".result")):
            rows.setdefault(kind, []).append(event.details)
    return rows


def _source_manifest(op: TimelineOperation) -> list[dict]:
    """Expose the retained historical file manifests at a command boundary.

    Parameters
    ----------
    op: TimelineOperation
        One recorded command.

    Returns
    -------
    list[dict]
        Historical source manifests, not regenerated final indexes.

    """
    return [e.details["source_manifest"] for e in op.matching(
        "index.check.completed", "index.empty")]


def _question(original: TimelineOperation, replayed: TimelineOperation,
              old_turns: dict[str, dict], new_turns: dict[str, dict],
              evaluator: EmbeddingEvaluator) -> dict:
    """Compare the committed question once, even if inherited by branches.

    Parameters
    ----------
    original: TimelineOperation
        Original ask command.
    replayed: TimelineOperation
        Matched replay ask command.
    old_turns: dict[str, dict]
        Validated original committed turns.
    new_turns: dict[str, dict]
        Validated replay committed turns.
    evaluator: EmbeddingEvaluator
        Shared optional evaluator, reused across questions.

    Returns
    -------
    dict
        Text, citation, usage, tool, intent, and semantic similarity metrics.

    """
    first = one(original.events, "gui.question.saved")
    second = one(replayed.events, "gui.question.saved")
    old_id = first.details["conversation_turn_uuid"]
    new_id = second.details["conversation_turn_uuid"]
    a, b = old_turns[old_id], new_turns[new_id]
    left, right = a["parameters"], b["parameters"]
    issues = []
    try:
        evidence = evidence_metrics(a, b)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        evidence = None
        issues.append(f"Evidence metric unavailable: {error}")
    return {
        "question": a["question"],
        "original_turn_uuid": old_id, "replayed_turn_uuid": new_id,
        "original_trace": first.path, "replayed_trace": second.path,
        "answer": {"original": a["answer"], "replayed": b["answer"],
                   **text_metrics(a["answer"], b["answer"]),
                   "embedding": evaluator.measure(a["answer"], b["answer"])},
        "evidence": evidence, "issues": issues,
        "intent": {"original": left.get("intent"),
                   "replayed": right.get("intent")},
        "retrieval_mode": {"original": left.get("retrieval_mode"),
                           "replayed": right.get("retrieval_mode")},
        "tool": {"original": left.get("tool"), "replayed": right.get("tool")},
        "completion": {"original": left.get("completion"),
                       "replayed": right.get("completion")},
        "finish_reason": {"original": left.get("finish_reason"),
                          "replayed": right.get("finish_reason")},
        "usage": {"original": usage(a), "replayed": usage(b)},
        "interactions": {"original": interaction_records(original),
                         "replayed": interaction_records(replayed)},
    }


def _operation(original: TimelineOperation, replayed: TimelineOperation,
               old_turns: dict[str, dict], new_turns: dict[str, dict],
               evaluator: EmbeddingEvaluator, catalogs: tuple,
               maps: tuple) -> dict:
    """Measure a matched command and its observed intermediate stages.

    Parameters
    ----------
    original: TimelineOperation
        Original operation.
    replayed: TimelineOperation
        Replayed operation.
    old_turns: dict[str, dict]
        Unique original committed turns.
    new_turns: dict[str, dict]
        Unique replay committed turns.
    evaluator: EmbeddingEvaluator
        Shared optional evaluation model.
    catalogs: tuple
        Original and replayed conversation turn-prefix maps.
    maps: tuple
        Original-to-replay conversation and turn maps.

    Returns
    -------
    dict
        One structured comparison row, with explicit metric errors.

    """
    row = {"kind": original.kind, "original_sequence": original.start_sequence,
           "replayed_sequence": replayed.start_sequence,
           "duration": {"original": operation_duration(original),
                        "replayed": operation_duration(replayed)},
           "settings_differences": [asdict(d) for d in
                                    _configuration_differences(
                                        (original,), (replayed,))],
           "plugin_provenance": compare_plugins(original, replayed),
           "source_manifests": {"original": _source_manifest(original),
                                "replayed": _source_manifest(replayed)},
           "stages": stage_metrics(original, replayed),
           "observations": {"original": _observations(original),
                            "replayed": _observations(replayed)},
           "rankings": ranking_metrics(original, replayed), "issues": []}
    row["context"] = context_comparison(original, replayed, catalogs, maps)
    try:
        row["model_calls"] = _call_comparison(original, replayed, maps)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        row["model_calls"] = None
        row["issues"].append(f"Model call metrics unavailable: {error}")
    if original.kind in ("create_project", "add_sources", "sync_sources"):
        row["source_requests"] = {
            side: (op.request["source_batch"] if op.kind == "create_project"
                   else op.request) for side, op in (
                       ("original", original), ("replayed", replayed))
        }
    if original.kind == "open_project" and "relocation" in original.request:
        row["relocation"] = {
            "original": original.request["relocation"],
            "replayed": replayed.request.get("relocation"),
            "physical_rename_replayed": "relocation" in replayed.request,
        }
    if original.kind == "create_project":
        row["settings_provenance"] = {
            side: op.request["settings"] for side, op in (
                ("original", original), ("replayed", replayed))
        }
    if original.kind == "ask":
        row["settings_provenance"] = {
            side: {k: op.request[k] for k in (
                "generation_settings", "generation_provenance",
                "profile_generation_settings", "explicit_generation_settings",
                "effort", "retrieval_count")}
            for side, op in (("original", original), ("replayed", replayed))
        }
        row["question"] = _question(original, replayed, old_turns, new_turns,
                                    evaluator)
        question = row["question"]
        question["tool_comparison"] = tool_comparison(
            question["tool"], question["completion"],
            executions={side: len(op.matching("tool.run.started"))
                        for side, op in (("original", original),
                                         ("replayed", replayed))},
            finish_reasons=question["finish_reason"])
    return row


def _state_pairs(left: dict, right: dict, conversations: dict[str, str],
                 turns: dict[str, str]) -> dict:
    """Compare mapped final conversation structure and authoritative ratings.

    Parameters
    ----------
    left: dict
        Original validated workspace summary.
    right: dict
        Replay validated workspace summary.
    conversations: dict[str, str]
        Reconstructed original-to-replay conversation map.
    turns: dict[str, str]
        Reconstructed original-to-replay unique turn map.

    Returns
    -------
    dict
        Paired final metadata, mapped turn prefixes and feedback. For a
        divergent prefix the final snapshots may include unmatched suffixes.

    """
    result = {"conversations": [], "feedback": []}
    if not left["valid"] or not right["valid"]:
        return result
    a = {c["id"]: c for c in left["conversations"]}
    b = {c["id"]: c for c in right["conversations"]}
    for old, new in conversations.items():
        first, second = a[old], b[new]
        mapped = [turns.get(uid) for uid in first["turn_uuids"]]
        result["conversations"].append({
            "original_id": old, "replayed_id": new,
            "name_equal": first["name"] == second["name"],
            "name_origin": {"original": first["name_origin"],
                            "replayed": second["name_origin"]},
            "archival_equal": first["archived"] == second["archived"],
            "lineage_equal": (first["origin_turn_count"] ==
                              second["origin_turn_count"] and
                              (first["parent_id"] is None or
                               first["parent_id"] in conversations) and
                              conversations.get(first["parent_id"]) ==
                              second["parent_id"]),
            "turn_prefix_equal": mapped == second["turn_uuids"],
        })
    for old, new in turns.items():
        first = left["feedback"].get(old, {})
        second = right["feedback"].get(new, {})
        result["feedback"].append({"original_turn_uuid": old,
                                   "replayed_turn_uuid": new,
                                   "original": first, "replayed": second,
                                   "equal": first == second})
    return result


def _aggregate(operations: list[dict]) -> dict:
    """Summarize only available question metrics with coverage counts.

    Parameters
    ----------
    operations: list[dict]
        Matched operation observations.

    Returns
    -------
    dict
        Per-metric means and denominators, without thresholds or composites.

    """
    questions = [row["question"] for row in operations if "question" in row]
    values = {
        "token_f1": [q["answer"]["token_f1"] for q in questions],
        "embedding_cosine": [q["answer"]["embedding"]["cosine"]
                             for q in questions],
        "evidence_jaccard": [q["evidence"]["overlap"]["jaccard"]
                             if q["evidence"] else None for q in questions],
        "citation_jaccard": [q["evidence"]["citations"]["overlap"]["jaccard"]
                             if q["evidence"] else None for q in questions],
    }
    result = {"questions": len(questions), "exact_answers": sum(
        q["answer"]["exact_equal"] for q in questions),
        "normalized_equal_answers": sum(q["answer"]["normalized_equal"]
                                        for q in questions)}
    for name, samples in values.items():
        available = [value for value in samples if value is not None]
        result[name] = {"mean": math.fsum(available) / len(available)
                        if available else None,
                        "available": len(available), "total": len(samples)}
    durations = [(op["duration"]["original"]["ms"],
                  op["duration"]["replayed"]["ms"]) for op in operations]
    paired = [(a, b) for a, b in durations if a is not None and b is not None]
    result["paired_command_duration_ms"] = {
        "original": math.fsum(a for a, _ in paired) if paired else None,
        "replayed": math.fsum(b for _, b in paired) if paired else None,
        "available": len(paired), "total": len(durations),
    }
    result["model_reproducibility"] = inference_summary(operations)
    result["tool_transitions"] = dict(sorted(Counter(
        q["tool_comparison"]["transition"] for q in questions).items()))
    return result


def _portable(value: object, path: str, issues: list[str]) -> object:
    """Copy JSON values and identify invalid optional diagnostic numbers.

    Parameters
    ----------
    value: object
        Diagnostic value or report container.
    path: str
        Location within the report, for unavailable observations.
    issues: list[str]
        Collector for explicit nonfinite-value diagnostics.

    Returns
    -------
    object
        Independent JSON-compatible value; nonfinite numbers become None,
        never zero. The unavailable location is retained in the report.

    """
    if type(value) is float and not math.isfinite(value):
        issues.append(f"Nonfinite diagnostic value unavailable at {path}.")
        return None
    if isinstance(value, dict):
        return {k: _portable(v, f"{path}/{k}", issues)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(v, f"{path}/{i}", issues)
                for i, v in enumerate(value)]
    return value


def compare(experiment: Experiment, destination: str | Path | None = None,
            *, embedding_model: str | None = None,
            embedding_backend: str = "auto",
            embedding_revision: str | None = None) -> Comparison:
    """Inspect and compare the original and an existing replay independently.

    Parameters
    ----------
    experiment: Experiment
        Closed original reference workspace.
    destination: str | pathlib.Path | None, optional
        Existing sibling replay, defaulting to the usual replay path. A missing
        or invalid replay is reported without inventing correspondences.
    embedding_model: str | None, optional
        Explicit local/cached evaluation model. Defaults to None, disabling
        embeddings; deterministic metrics remain available without models.
    embedding_backend: str, optional
        Evaluation backend preference, defaulting to ``"auto"``.
    embedding_revision: str | None, optional
        Requested evaluation model revision, defaulting to None.

    Returns
    -------
    Comparison
        Portable report snapshot. Both workspaces remain unchanged.

    Notes
    -----
    Both complete recordings must validate before any matching. For valid
    but divergent histories, only their unambiguous prefix is compared.
    Recorded metrics are observations, not correctness or quality verdicts.
    The evaluator is optional, uses one model for both sides, and does not
    request downloads, retries, or fallback models.

    """
    if embedding_model is not None and (
            not isinstance(embedding_model, str) or not embedding_model.strip()
    ):
        raise ValueError("Choose an explicit embedding model or None.")
    if embedding_backend not in ("auto", "mlx", "torch"):
        raise ValueError("Invalid evaluation embedding backend.")
    if embedding_revision is not None and (
            not isinstance(embedding_revision, str) or
            not embedding_revision.strip()):
        raise ValueError("Invalid evaluation model revision.")
    if embedding_model is None and embedding_revision is not None:
        raise ValueError("An embedding revision requires an evaluation model.")
    root = experiment.workspace
    target = _replay_path(root, destination)
    old, new = read_recording(root), read_recording(target)
    before, after = validate_recording(old), validate_recording(new)
    evaluator = EmbeddingEvaluator(embedding_model, embedding_backend,
                                   embedding_revision)
    rows = []
    if before.valid and after.valid:
        matched = align(before.operations, after.operations)
        old_turns, new_turns = _saved_turns(old), _saved_turns(new)
        catalogs = tuple({c["id"]: c["turn_uuids"] for c in _catalog(r)}
                         for r in (old, new))
        maps = matched.conversations, matched.turns
        try:
            for a, b in matched.pairs:
                rows.append(_operation(a, b, old_turns, new_turns, evaluator,
                                       catalogs, maps))
        finally:
            evaluator.close()
        status = "complete" if matched.reason is None else "diverged"
        reason = matched.reason
        conversations, turns = matched.conversations, matched.turns
    else:
        status = "invalid_recording"
        reason = "Both closed workspace recordings must validate."
        conversations, turns = {}, {}
    left, right = _workspace(old, before), _workspace(new, after)
    data = {
        "format_version": 1, "original": left, "replayed": right,
        "alignment": {"status": status, "reason": reason,
                      "matched_operations": len(rows),
                      "original_unmatched": left["operations"][len(rows):],
                      "replayed_unmatched": right["operations"][len(rows):],
                      "conversation_ids": conversations, "turn_uuids": turns},
        "summary": _aggregate(rows), "operations": rows,
        "final_state": _state_pairs(left, right, conversations, turns),
        "embedding_evaluation": evaluator.provenance,
        "files": {directory: file_changes(inventory(root, directory),
                                          inventory(target, directory))
                  for directory in ("sources", "templates", "outputs")},
    }
    data["performance"] = {
        side: performance_summary(report.events) if report.valid else None
        for side, report in (("original", before), ("replayed", after))}
    data["inspection"] = note_inspection(data, before)
    data["reference_publication"] = inspect_receipt(root, before)
    data["replay_execution"] = execution_provenance(target, after)
    if data["replay_execution"]["status"] == "recorded":
        data["replay_execution"]["reference_match"] = (
            reference_execution_match(data["replay_execution"], before))
    data["expected_event_differences"] = relocation_event_differences(
        data, before, after)
    issues: list[str] = []
    snapshot = _portable(data, "", issues)
    snapshot["report_issues"] = issues
    # Fail before publishing if the report is not a valid JSON object.
    json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
    return Comparison(root, target, snapshot)
