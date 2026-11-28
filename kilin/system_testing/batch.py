"""Orchestrate reference experiments and pre-render their analysis notebooks.

Key Features
------------
- One validation/replay/comparison path per discovered reference.
- Independent notebook generation and execution from current templates.
- Explicit issue classification and complete per-experiment action logs.
- Explicit retain, archive, or error policy for existing replays.
- Reference publication and inspection-only extension diagnostics.

Module Structure
----------------
- Snapshot and notebook generation helpers.
- Batch orchestration and issue collection.

Examples
--------
>>> from pathlib import Path
>>> from kilin.system_testing.batch import discover_originals
>>> originals = discover_originals(Path('workspaces'))  # doctest: +SKIP

Only packaged KíLín templates are executed, in separate local kernels
using the interpreter running the batch. Saved report text is data only.

"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from importlib.resources import files
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nbformat import NotebookNode

from .analysis import (
    load_report, save_analysis, event_inventory, _release_measurements,
    _release_rows, _release_totals, _release_agreement_rows,
)

__all__ = [
    "discover_originals", "run_batch", "collect_issues", "load_snapshot",
    "save_batch_summary", "group_issues",
]

NOTEBOOKS = (
    ("comparison", "experiment_template.ipynb", ".ipynb"),
    ("answers", "explore_answers.ipynb", "_answers.ipynb"),
)
MARKER = 'REFERENCE = Path("__ORIGINAL_WORKSPACE__")'
SNAPSHOT_ENV = "KILIN_SYSTEM_TESTS_BATCH_REPORT"
SOURCE_ENV = "KILIN_SYSTEM_TESTS_BATCH_SOURCE"


def discover_originals(directory: Path) -> list[Path]:
    """Discover direct real original workspaces in deterministic order.

    Parameters
    ----------
    directory: pathlib.Path
        Explicit directory to inspect or write artifacts in.

    Returns
    -------
    list[pathlib.Path]
        Direct original workspaces, sorted by path.

    """
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Expected a real workspace directory: {directory}")
    return sorted(p for p in directory.glob("*_original")
                  if p.is_dir() and not p.is_symlink())


def _write_text(path: Path, text: str, *, overwrite: bool) -> None:
    """Publish a complete artifact without following linked destinations.

    Parameters
    ----------
    path: pathlib.Path
        Local artifact destination.
    text: str
        Text to publish or render.
    overwrite: bool
        Explicitly permit replacing an existing regular artifact.

    """
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"Nonregular artifact destination: {path}")
    temp = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                dir=path.parent) as stream:
            temp = Path(stream.name)
            stream.write(text)
        if overwrite:
            temp.replace(path)
        else:
            # Exclusive publication avoids truncation or partial notebooks.
            os.link(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _notebook(template: str, original: Path) -> "NotebookNode":
    """Instantiate a canonical template, removing all historical outputs.

    Parameters
    ----------
    template: str
        Name of a trusted notebook resource packaged with KíLín.
    original: pathlib.Path
        Reference whose name replaces the template marker.

    Returns
    -------
    NotebookNode
        Fresh validated notebook with its reference marker replaced.

    """
    import nbformat

    resource = files(__package__).joinpath("_notebooks", template)
    if not resource.is_file():
        raise FileNotFoundError(f"Missing packaged notebook: {template}")
    nb = nbformat.reads(resource.read_text(encoding="utf-8"),
                        as_version=4)
    nbformat.validate(nb)
    matches = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        lines = cell.source.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.rstrip("\r\n") == MARKER:
                lines[i] = f"REFERENCE = Path({original.name!r})\n"
                matches += 1
        cell.source = "".join(lines)
        cell.outputs = []
        cell.execution_count = None
    if matches != 1:
        raise ValueError("Expected exactly one template reference marker.")
    nb.metadata["kilin_system_tests"] = {
        "generated": True, "reference": original.name,
        "template": template,
    }
    nbformat.validate(nb)
    return nb


def _canonical_json(data: dict) -> str:
    """Use a stable representation for snapshot identity across views.

    Parameters
    ----------
    data: dict
        Saved comparison snapshot; never modified.

    """
    return json.dumps(data, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def load_snapshot(reference: Path) -> tuple[dict, str, str]:
    """Load the batch's exact snapshot or the ordinary saved local report.

    Parameters
    ----------
    reference: pathlib.Path
        Reference workspace associated with the snapshot.

    Returns
    -------
    tuple[dict, str, str]
        Comparison data, provenance, and SHA-256 of canonical JSON.
        A temporary batch snapshot is never stored as a notebook parameter.

    """
    reference = Path(reference)
    if (reference.name == "__ORIGINAL_WORKSPACE__" or
            not reference.name.endswith("_original")):
        raise ValueError("Open a generated workspace notebook instead.")
    override = os.environ.get(SNAPSHOT_ENV)
    if override:
        path = Path(override)
        source = os.environ.get(SOURCE_ENV, "Current batch snapshot")
    else:
        stem = reference.name.removesuffix("_original")
        path = reference.with_name(stem + "_comparison.json")
        source = "Saved report snapshot (not independently revalidated)"
    data = load_report(path)
    digest = sha256(_canonical_json(data).encode("utf-8")).hexdigest()
    return data, source, digest


def _execute(nb, directory: Path, snapshot: Path, source: str,
             timeout: int) -> None:
    """Execute a fresh template in the batch interpreter's local kernel.

    Parameters
    ----------
    nb: NotebookNode
        Fresh notebook instance to execute in place.
    directory: pathlib.Path
        Explicit directory to inspect or write artifacts in.
    snapshot: pathlib.Path
        Temporary file containing the exact comparison snapshot.
    source: str
        Human-readable snapshot provenance.
    timeout: int
        Per-cell execution timeout in seconds.

    """
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    # Do not silently use a different installed python3 kernelspec.
    with TemporaryDirectory(prefix="kilin-kernel-") as temporary:
        kernel = Path(temporary) / "kilin-local"
        kernel.mkdir()
        spec = {"argv": [sys.executable, "-m", "ipykernel_launcher",
                         "-f", "{connection_file}"],
                "display_name": "KíLín batch interpreter",
                "language": "python"}
        (kernel / "kernel.json").write_text(json.dumps(spec),
                                            encoding="utf-8")
        km = AsyncKernelManager(
            kernel_name="kilin-local",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]),
        )
        env = os.environ.copy()
        env[SNAPSHOT_ENV] = str(snapshot.resolve())
        env[SOURCE_ENV] = source
        client = NotebookClient(nb, km=km, timeout=timeout,
                                allow_errors=False)
        client.execute(cwd=str(directory.resolve()), env=env,
                       cleanup_kc=True)


def _prepare_notebooks(original, row, overwrite):
    """Create fresh templates or explicitly retain existing artifacts.

    Parameters
    ----------
    original: pathlib.Path
        Reference for the generated report notebooks.
    row: dict
        Current record or batch action row.
    overwrite: bool
        Explicitly permit replacing an existing regular artifact.

    Returns
    -------
    dict
        Fresh notebook instances ready for execution.

    """
    stem = original.name.removesuffix("_original")
    if not stem:
        raise ValueError("Original workspace requires a nonempty stem.")
    prepared = {}
    for kind, template, suffix in NOTEBOOKS:
        destination = original.with_name(stem + suffix)
        entry = {"path": str(destination), "status": "pending",
                 "error": None}
        row["notebooks"][kind] = entry
        try:
            import nbformat

            if destination.is_symlink() or (destination.exists() and
                                            not destination.is_file()):
                raise ValueError(f"Nonregular notebook: {destination}")
            existed = destination.exists()
            if existed and not overwrite:
                entry["status"] = "retained"
                continue
            nb = _notebook(template, original)
            _write_text(destination, nbformat.writes(nb) + "\n",
                        overwrite=overwrite)
            entry["status"] = "refreshed" if existed else "created"
            prepared[kind] = (nb, destination)
        except Exception as error:
            entry.update(status="error", error=str(error))
    return prepared


def _render_notebooks(prepared, row, snapshot, source, timeout):
    """Persist executed notebooks, retaining diagnostic outputs on failure.

    Parameters
    ----------
    prepared: dict
        Prepared notebook instances and their destination paths.
    row: dict
        Current record or batch action row.
    snapshot: pathlib.Path
        Temporary file containing the exact comparison snapshot.
    source: str
        Human-readable snapshot provenance.
    timeout: int
        Per-cell execution timeout in seconds.

    """
    for kind, (nb, path) in prepared.items():
        entry = row["notebooks"][kind]
        try:
            print(f"Executing {path.name}...", flush=True)
            _execute(nb, path.parent, snapshot, source, timeout)
            entry["status"] = "executed"
        except Exception as error:
            entry.update(status="error", error=str(error))
        # A failing cell's traceback is valuable, but not a success.
        try:
            import nbformat

            nbformat.validate(nb)
            _write_text(path, nbformat.writes(nb) + "\n", overwrite=True)
        except Exception as error:
            previous = entry.get("error")
            entry.update(status="error", error=(
                f"{previous}; notebook save: {error}" if previous else
                f"Notebook save: {error}"))


def _event_differences(data: dict, inventory: dict) -> list[dict]:
    """Attribute event-count changes to aligned commands when corroborated.

    Parameters
    ----------
    data: dict
        Comparison with per-operation recorded stage counts.
    inventory: dict
        Reconciled independent workspace event inventories.

    Returns
    -------
    list[dict]
        Per-command changes and residual outside-matched-command changes.
        Every side's counts must reconcile; no causal relationships are
        inferred, and opposite changes in different commands are retained.

    """
    sides = ("original", "replayed")
    counted = {side: Counter() for side in sides}
    result = []
    fallback = [{**item, "attribution": "unavailable"}
                for item in inventory["differences"]]
    for op in data["operations"]:
        local = {side: Counter() for side in sides}
        if not isinstance(op.get("stages"), list):
            return fallback
        for stage in op["stages"]:
            name = stage.get("event_type")
            for side in sides:
                count = (stage.get(side) or {}).get("count")
                if (not isinstance(name, str) or type(count) is not int
                        or count < 0):
                    # No reliable operation-level attribution is possible.
                    return fallback
                local[side][name] += count
        for side in sides:
            counted[side].update(local[side])
        for name in sorted(local["original"].keys() |
                           local["replayed"].keys()):
            a, b = (local[side][name] for side in sides)
            if a != b:
                result.append({"event_type": name, "original": a,
                               "replayed": b,
                               "sequence": op.get("original_sequence")})
    remainder = {}
    for side in sides:
        remainder[side] = Counter(inventory["counts"][side])
        remainder[side].subtract(counted[side])
        if any(count < 0 for count in remainder[side].values()):
            return fallback
    for name in sorted(remainder["original"].keys() |
                       remainder["replayed"].keys()):
        a, b = (remainder[side][name] for side in sides)
        if a != b:
            result.append({"event_type": name, "original": a,
                           "replayed": b, "sequence": None})
    return result


def _incomplete_answer(completion: object, finish: object) -> bool:
    """Identify explicit ordinary-answer incompletion without a tool.

    Parameters
    ----------
    completion: object
        Saved answer completion metadata.
    finish: object
        Saved model termination reason.

    Returns
    -------
    bool
        Whether recorded metadata explicitly indicates incompletion.

    """
    status = completion.get("status") if isinstance(completion, dict) else None
    return finish == "length" or status in (
        "incomplete", "partial", "failed", "error", "cancelled", "aborted",
        "interrupted")


def _tool_finding(tools: dict) -> tuple[str, str] | None:
    """Describe a paired tool outcome without treating unknown as success.

    Parameters
    ----------
    tools: dict
        Functional tool comparison with side-specific outcome observations.

    Returns
    -------
    tuple[str, str] | None
        Diagnostic code and concise comparison, when attention is warranted.

    """
    outcomes = tools.get("outcomes", {})
    transition = tools.get("transition")
    states = [outcomes.get(side, {}).get("status")
              for side in ("original", "replayed")]
    failed = [side for side, state in zip(("original", "replayed"), states)
              if state == "incomplete"]
    labels = {
        "incomplete_reproduced": "Incomplete outcome reproduced",
        "potential_regression": "Potential regression: complete to incomplete",
        "possible_improvement": "Changed outcome: incomplete to complete",
        "changed": "Recorded tool applicability or outcome changed",
    }
    if transition in labels or failed:
        label = labels.get(transition, "Incomplete recorded result")
        reasons = []
        for side, outcome in outcomes.items():
            completion = outcome.get("completion") or {}
            reason = completion.get("reason") or outcome.get("finish_reason")
            items = outcome.get("incomplete_items") or completion.get(
                "incomplete_items") or []
            item_reasons = sorted({item["reason"] for item in items
                                   if isinstance(item, dict) and isinstance(
                                       item.get("reason"), str)})
            if reason or item_reasons:
                reasons.append(f"{side}: {reason or 'incomplete items'}"
                               + (f" ({', '.join(item_reasons)})"
                                  if item_reasons else ""))
        code = ("capability.changed" if transition in (
            "possible_improvement", "changed") else "capability.incomplete")
        return code, (f"{label}; original: {states[0]}, replay: {states[1]}"
                      + ("; " + "; ".join(reasons) if reasons else ""))
    return None


def collect_issues(data: dict | None, row: dict) -> list[dict]:
    """Classify execution failures separately from descriptive differences.

    Critical means a failed requested step, invalid/divergent recording,
    or an explicit diagnostic error. Review means a measured difference or
    a warning requiring inspection. Text similarity never defines failure.
    Recording-limit warnings are informational, not per-run defects.

    Parameters
    ----------
    data: dict or None
        Current comparison snapshot, if available.
    row: dict
        Current record or batch action row.

    Returns
    -------
    list[dict]
        Classified individual observations.

    """
    issues = []

    def add(level, code, message, sequence=None, *, side=None):
        """Append an individual diagnostic finding.

        Parameters
        ----------
        level: str
            Severity of the diagnostic finding.
        code: str
            Diagnostic category identifier.
        message: object
            Finding detail, converted to text.
        sequence: int or None
            Recorded sequence associated with the finding.
        side: str or None, optional
            Recording side for an independently observed sequence.

        """
        issue = {"severity": level, "code": code,
                 "message": str(message), "sequence": sequence}
        if side is not None:
            issue["side"] = side
        issues.append(issue)

    for field in ("replay_error", "comparison_error", "save_error",
                  "analysis_error", "snapshot_error"):
        if row.get(field):
            add("critical", field, row[field])
    for kind, entry in row.get("notebooks", {}).items():
        if entry.get("error"):
            add("critical", f"notebook.{kind}", entry["error"])
        elif entry.get("status") == "retained" and row.get("render_requested"):
            add("review", f"notebook.{kind}.retained",
                "Existing notebook retained; not pre-executed.")
    if data is None:
        return issues
    alignment = data["alignment"]
    inspection = data.get("inspection", data.get("batch_inspection"))
    if (isinstance(inspection, dict) and
            inspection.get("status") == "inspection_only" and
            alignment.get("status") == "diverged"):
        add("info", "alignment.inspection", inspection["message"])
    elif alignment.get("status") != "complete":
        add("critical", "alignment", alignment.get("reason") or
            alignment.get("status"))
    receipt = data.get("reference_publication", {})
    if receipt.get("status") == "inspection_only":
        add("info", "reference.inspection",
            "Published reference retained; inspection sessions appended.")
    elif receipt.get("status") in ("changed", "extended", "error"):
        add("review", "reference." + receipt["status"],
            receipt.get("message"))
    execution = data.get("replay_execution", {})
    for observation in (row.get("replay_execution", {}), execution):
        status = observation.get("status")
        match = observation.get("reference_match")
        if status in ("changed", "error") or match in ("changed", "error"):
            add("review", "replay.provenance", observation.get("message") or
                "Execution receipt or recorded reference prefix differs.")
            break
    for hint in receipt.get("coverage", []):
        if isinstance(hint, str) and hint.startswith("REVIEW:"):
            add("info", "generator.coverage",
                "Reference at publication: " + hint)
    for side in ("original", "replayed"):
        workspace = data[side]
        if workspace.get("valid") is not True:
            add("critical", f"{side}.invalid", "Recording is not valid.")
        for field, level in (("validation_issues", "critical"),
                             ("validation_warnings", "review")):
            for issue in workspace.get(field, []):
                severity = ("info" if issue.get("code") in (
                    "recording.limit", "relocation.inferred",
                    "environment.difference") else level)
                add(severity, f'{side}.{issue.get("code")}',
                    issue.get("message"), issue.get("sequence"), side=side)
    for issue in data.get("report_issues", []):
        add("critical", "report.issue", issue)
    evaluation = data.get("embedding_evaluation", {})
    if evaluation.get("error"):
        add("review", "embedding.error", evaluation["error"])
    if alignment.get("status") != "complete":
        # The matched prefix is not evidence of a complete execution.
        add("info", "alignment.coverage", "Only reliable pairs are shown.")
    for op in data["operations"]:
        seq = op.get("original_sequence")
        for issue in op.get("issues", []):
            add("critical", "operation.issue", issue, seq)
        for diff in op.get("settings_differences", []):
            add("review", "settings.difference", diff, seq)
        for pair in op.get("plugin_provenance", []):
            if pair["status"] == "changed":
                add("review", "plugin.environment.difference", pair, seq)
        for stage in op.get("stages", []):
            for side in ("original", "replayed"):
                statuses = (stage.get(side) or {}).get("statuses", {})
                for status, count in statuses.items():
                    if count and status in {"error", "failed", "cancelled",
                                            "aborted", "interrupted"}:
                        # These are intermediate events in a matched valid
                        # command. Blocking command failures are handled by
                        # recording validation/alignment above.
                        add("review", "stage.failure",
                            f'{side}: {stage.get("event_type")} '
                            f'{status} ({count})', seq)
        question = op.get("question")
        for context in op.get("context", []):
            kind = context.get("event_type")
            for issue in context.get("issues", []):
                add("review", "context.unavailable", issue, seq)
            if context.get("identity_equal") is False:
                if kind == "history.selection.result":
                    flags = context.get("selection_equal", {})
                    different = [key for key, equal in flags.items()
                                 if equal is False]
                    add("review", "context.history", ", ".join(different),
                        seq)
                else:
                    add("review", "context.identity", kind, seq)
            elif context.get("identity_equal") is None and not context.get(
                    "issues"):
                add("review", "context.unavailable", kind, seq)
        for ranking in op.get("rankings", []):
            for issue in ranking.get("issues", []):
                add("review", "ranking.issue", issue, seq)
            overlap = ranking.get("overlap") or {}
            jaccard = overlap.get("jaccard")
            if type(jaccard) in (int, float) and jaccard < 1:
                add("review", "ranking.difference",
                    f'{ranking.get("event_type")}: {jaccard:.4f}', seq)
        for call in op.get("model_calls") or []:
            old, new = call.get("original"), call.get("replayed")
            if (old is None) != (new is None):
                add("review", "model_call.difference",
                    f'{call.get("stage")}: unmatched call', seq)
            for side, entry in (("original", old), ("replayed", new)):
                if isinstance(entry, dict) and entry.get("issue"):
                    add("review", "model_call.issue",
                        f'{side}: {entry["issue"]}', seq)
            flags = call.get("inference", {})
            label = f'{call.get("stage")} #{call.get("occurrence")}'
            changed = [key for key in ("model_equal", "settings_equal",
                                       "finish_reason_equal")
                       if flags.get(key) is False]
            if (flags.get("rendered_prompt_equal") is True and
                    flags.get("input_token_count_equal") is False):
                changed.append("input_token_count_equal")
            if changed:
                add("review", "model_call.configuration",
                    f'{label}: {", ".join(changed)}', seq)
            category = flags.get("conditional_category")
            if flags.get("output_equal") is False:
                if category == "zero_temperature":
                    add("review", "model_call.conditional_output",
                        f"{label}: different output with identical recorded "
                        "zero-temperature inputs.", seq)
                elif category == "nonzero_temperature":
                    add("info", "model_call.stochastic_output",
                        f"{label}: output differs at nonzero temperature.",
                        seq)
        if not isinstance(question, dict):
            continue
        for issue in question.get("issues", []):
            add("critical", "question.issue", issue, seq)
        tools = question.get("tool_comparison")
        if isinstance(tools, dict):
            if tools.get("functional_equal") is False:
                add("review", "question.tool", "Functional results differ.",
                    seq)
            finding = _tool_finding(tools)
            if finding is not None:
                add("review", *finding, seq)
            ordinary = [side for side in ("original", "replayed")
                        if tools.get("outcomes", {}).get(side, {}).get(
                            "status") in ("not_applicable", "unknown") and
                        _incomplete_answer(
                            (question.get("completion") or {}).get(side),
                            (question.get("finish_reason") or {}).get(side))]
            if ordinary:
                add("review", "answer.incomplete",
                    "Incomplete saved answer: " + ", ".join(ordinary), seq)
        fields = ("intent", "retrieval_mode", "completion",
                  "finish_reason", "interactions")
        if not isinstance(tools, dict):
            fields += ("tool",)
        for field in fields:
            pair = question.get(field) or {}
            if ("original" in pair and "replayed" in pair and
                    pair["original"] != pair["replayed"]):
                add("review", f"question.{field}",
                    "Recorded observations differ.", seq)
        evidence = question.get("evidence") or {}
        citations = evidence.get("citations") or {}
        for side in ("original", "replayed"):
            if citations.get(f"{side}_unresolved"):
                add("review", "citation.unresolved",
                    f"{side}: {citations[f'{side}_unresolved']}", seq)
        for name, item in (("evidence", evidence), ("citation", citations)):
            jaccard = (item.get("overlap") or {}).get("jaccard")
            if type(jaccard) in (int, float) and jaccard < 1:
                add("review", f"{name}.difference",
                    f"Exact passage overlap: {jaccard:.4f}", seq)
        if (question.get("answer") or {}).get("normalized_equal") is False:
            add("info", "answer.difference", "Visible answers differ.", seq)
    for directory, inventory in data.get("files", {}).items():
        for side in ("original", "replayed"):
            for issue in (inventory.get(side) or {}).get("issues", []):
                add("critical", "files.inventory",
                    f"{side}: {directory}: {issue}")
        changes = inventory.get("changes") or {}
        if any(changes.get(key) for key in ("added", "removed", "changed")):
            add("review", "files.difference", f"{directory}: files differ.")
    for item in data.get("final_state", {}).get("conversations", []):
        for field in ("name_equal", "lineage_equal", "archival_equal",
                      "turn_prefix_equal"):
            if item.get(field) is False:
                origins = item.get("name_origin", {})
                generated = (field == "name_equal" and
                             origins.get("original") == "generated" and
                             origins.get("replayed") == "generated")
                add("info" if generated else "review", "state.difference",
                    field)
    for item in data.get("final_state", {}).get("feedback", []):
        if item.get("equal") is False:
            add("review", "state.feedback", "Mapped feedback differs.")
    events = event_inventory(data)
    if events["complete"]:
        for item in _event_differences(data, events):
            # A runtime message is a diagnostic, not a failed command.
            level = ("info" if (item["event_type"] == "runtime.output" or
                     item["event_type"] in data.get(
                         "expected_event_differences", []) or
                     alignment.get("status") != "complete") else "review")
            location = (
                "Workspace total (operation attribution unavailable): "
                if item.get("attribution") == "unavailable" else
                "Outside matched commands: " if item.get("sequence") is None
                else "")
            add(level, "events.difference", location +
                f'{item["event_type"]}: {item["original"]} -> '
                f'{item["replayed"]}', item.get("sequence"))
    elif alignment.get("status") == "complete":
        add("review", "events.coverage", "Full event counts unavailable.")
    # A missing or invalid replay does not suppress independently saved
    # tool outcomes. Do not duplicate already aligned questions.
    for side in ("original", "replayed"):
        paired = {op["question"].get(side + "_turn_uuid")
                  for op in data["operations"] if op.get("question")}
        for question in data[side].get("saved_questions", []):
            if question.get("turn_uuid") in paired:
                continue
            status = question.get("tool_outcome", {}).get("status")
            if status == "incomplete":
                add("review", "capability.incomplete", f"{side}: "
                    "independently observed incomplete result.",
                    question.get("sequence"), side=side)
            elif status in ("not_applicable", "unknown", None) and (
                    _incomplete_answer(
                        question.get("completion"),
                        question.get("finish_reason"))):
                add("review", "answer.incomplete", f"{side}: "
                    "independently observed incomplete answer.",
                    question.get("sequence"), side=side)
    return issues


def group_issues(issues, data):
    """Group related observations without discarding individual findings.

    Groups are diagnostic categories, not proof of a causal relationship.

    Parameters
    ----------
    issues: list[dict]
        Individual classified diagnostic findings.
    data: dict or None
        Comparison providing alignment and workspace context.

    Returns
    -------
    list[dict]
        Groups retaining their complete underlying findings.

    """
    groups = {}
    missing = bool(data and not data["replayed"].get("valid") and
                   data["replayed"].get("event_count") == 0)
    original_invalid = bool(data and not data["original"].get("valid"))
    operations = {op["original_sequence"]: op for op in
                  (data or {}).get("operations", [])}
    questions = {op["original_sequence"]: i for i, op in enumerate(
        (op for op in operations.values() if op.get("question")), 1)}
    for i, issue in enumerate(issues):
        code, seq = issue["code"], issue["sequence"]
        key = (issue["severity"], code, i)
        paired = issue.get("side") is None and seq in operations
        if missing and issue["severity"] == "critical" and (
                code in ("alignment", "replay_error") or
                code.startswith("replayed.") or
                (original_invalid and code.startswith("original.")) or
                (code == "files.inventory" and
                 issue["message"].startswith("replayed:"))):
            key = ("critical", "recording.original" if original_invalid else
                   "recording.replayed", None)
        elif code == "plugin.environment.difference":
            key = (issue["severity"], "environment.plugins", seq)
        elif code.endswith(".environment.difference"):
            key = (issue["severity"], "environment.inspection", None)
        elif paired and code in (
                "citation.unresolved", "citation.difference"):
            key = (issue["severity"], "question.citations", seq)
        elif paired and (code.startswith(("context.",
                "ranking.", "model_call.")) or code in (
                    "question.intent", "question.retrieval_mode",
                    "question.interactions", "question.tool",
                    "question.completion", "question.finish_reason",
                    "events.difference", "settings.difference",
                    "stage.failure", "evidence.difference")):
            key = (issue["severity"], "question.execution" if seq in questions
                   else "operation.execution", seq)
        group = groups.setdefault(key, {"severity": key[0], "code": key[1],
                                       "sequence": seq, "issues": []})
        if paired and seq in questions:
            group["question_number"] = questions[seq]
        group["issues"].append(issue)
    return list(groups.values())


def _compare(original, row, replay, compare, save, overwrite, analysis,
             evaluation, existing_replay, root):
    """Run the KíLín API once per requested stage; isolate failures.

    Parameters
    ----------
    original: pathlib.Path
        Closed reference workspace.
    row: dict
        Current record or batch action row.
    replay: bool
        Whether to execute or retain a replay.
    compare: bool
        Whether to calculate a new comparison.
    save: bool
        Whether to save native comparison reports.
    overwrite: bool
        Explicitly permit replacing an existing regular artifact.
    analysis: bool
        Whether to write the visual Markdown companion.
    evaluation: dict or None
        Optional comparison embedding-evaluation settings.
    existing_replay: str
        Existing-replay policy: retain, archive, or error.
    root: pathlib.Path
        Explicit local repository or workspace directory.

    Returns
    -------
    dict or None
        Current comparison data, if requested and available.

    """
    data = None
    destination = original.with_name(
        original.name.removesuffix("_original") + "_replay")
    if replay or compare:
        try:
            from kilin.system_testing import Experiment
            experiment = Experiment(original)
        except Exception as error:
            row["comparison_error" if compare else "replay_error"] = str(error)
            return None
        if replay:
            try:
                action = experiment.run_replay(
                    destination=destination,
                    existing_replay=existing_replay,
                    archive_directory=root / "_replay_archive",
                    progress=lambda i, n, op: print(i, n, op.kind),
                )
                row.update(replay_status=action.status,
                           replay_error=action.error,
                           replay_execution=action.provenance,
                           completed_operations=action.completed_operations,
                           failure_sequence=action.sequence)
                if action.archived_workspace is not None:
                    row["archived_replay"] = str(action.archived_workspace)
                    print("Previous replay archived: "
                          f"{action.archived_workspace}", flush=True)
                if action.status == "retained":
                    print("Existing replay retained.", flush=True)
            except Exception as error:
                row.update(replay_status="error", replay_error=str(error))
        if compare:
            try:
                print("Comparing recordings...", flush=True)
                result = experiment.compare(destination=destination,
                                            **evaluation)
                data = result.data
                row["reference_publication"] = data.get(
                    "reference_publication", {"status": "unavailable"})
                row["inspection"] = data.get("inspection")
                row["snapshot"] = "current"
                row["alignment"] = data["alignment"]["status"]
            except Exception as error:
                row["comparison_error"] = str(error)
                return None
            if save:
                try:
                    paths = result.save(overwrite=overwrite)
                    row["reports"] = [str(path) for path in paths]
                except Exception as error:
                    row["save_error"] = str(error)
                else:
                    if analysis:
                        try:
                            row["analysis"] = str(save_analysis(
                                paths[0], overwrite=overwrite))
                        except Exception as error:
                            row["analysis_error"] = str(error)
    return data


def run_batch(originals: list[Path], root: Path, *, run_replay: bool,
              run_comparison: bool, save_reports: bool,
              overwrite_reports: bool, save_visual_analysis: bool,
              run_notebooks: bool = True, refresh_notebooks: bool = True,
              notebook_timeout: int = 180, evaluation: dict | None = None,
              existing_replay: str = "retain"
              ) -> tuple[dict, list[dict]]:
    """Run all requested stages and compile current data and action rows.

    Existing replays are retained by default. ``"archive"`` moves an
    existing replay outside discovery before a new execution; ``"error"``
    refuses reuse. The policy applies only when replay is requested.
    Both projects must be closed; no concurrent GUI use is supported.

    Each comparison is evaluated once. Its exact JSON data are provided to
    both report-only notebooks via a temporary file, even when native
    saving is disabled or fails. An old saved report never substitutes for
    a failed requested comparison. With comparison disabled, rendering an
    explicitly historical snapshot is allowed. The temporary copy is
    removed on exit and never becomes a notebook source parameter.

    Parameters
    ----------
    originals: list[pathlib.Path]
        Reference workspaces in the intended batch order.
    root: pathlib.Path
        Local experiment repository for replay archives, not templates.
    run_replay: bool
        Request managed replay of each reference.
    run_comparison: bool
        Compute a fresh comparison of each pair.
    save_reports: bool
        Save native JSON and Markdown comparison reports.
    overwrite_reports: bool
        Permit replacement of existing report files.
    save_visual_analysis: bool
        Write visual Markdown companions when native reports are
        saved.
    run_notebooks: bool
        Generate and execute report-only notebooks.
    refresh_notebooks: bool
        Refresh existing generated notebooks from packaged templates.
    notebook_timeout: int
        Maximum seconds allowed for one notebook cell.
    evaluation: dict or None
        Optional comparison embedding-evaluation settings.
    existing_replay: str
        Existing-replay policy: retain, archive, or error.

    Returns
    -------
    tuple[dict, list[dict]]
        Comparison snapshots keyed by stem and per-experiment action
        rows.

    """
    if existing_replay not in ("retain", "archive", "error"):
        raise ValueError(
            "Choose retain, archive, or error for existing replay.")
    reports, results = {}, []
    root = Path(root).resolve()
    claimed = {}
    all_originals = set(originals)
    for parent in {Path(p).parent for p in originals}:
        all_originals.update(discover_originals(parent))
    for candidate in all_originals:
        stem = candidate.name.removesuffix("_original")
        for _, _, suffix in NOTEBOOKS:
            path = candidate.parent.resolve() / (stem + suffix)
            claimed.setdefault(path, set()).add(candidate.name)
    for original in originals:
        original = Path(original)
        stem = original.name.removesuffix("_original")
        print(f"\n=== {stem} ===", flush=True)
        row = {"experiment": original.name, "snapshot": None,
               "replay_status": "not requested", "notebooks": {},
               "existing_replay": existing_replay, "archived_replay": None,
               "render_requested": run_notebooks, "replay_error": None,
               "comparison_error": None, "save_error": None,
               "analysis_error": None, "snapshot_error": None}
        try:
            conflicts = [original.with_name(stem + suffix) for _, _, suffix
                         in NOTEBOOKS if len(claimed.get(
                             original.parent.resolve() / (stem + suffix),
                             set())) > 1]
            if conflicts:
                raise ValueError(f"Notebook name collision: {conflicts}")
            prepared = _prepare_notebooks(original, row, refresh_notebooks)
        except Exception as error:
            row["snapshot_error"] = str(error)
            prepared = {}
        data = None
        try:
            data = _compare(original, row, run_replay, run_comparison,
                            save_reports, overwrite_reports,
                            save_visual_analysis, evaluation or {},
                            existing_replay, root)
        except Exception as error:
            row["comparison_error"] = str(error)
        if not run_comparison and run_notebooks:
            try:
                path = original.with_name(stem + "_comparison.json")
                data = load_report(path)
                row["snapshot"] = "saved"
                row["alignment"] = data["alignment"]["status"]
            except Exception as error:
                row["snapshot_error"] = str(error)
        if data is not None:
            try:
                row["inspection"] = data.get(
                    "inspection", data.get("batch_inspection"))
                row["reference_publication"] = data.get(
                    "reference_publication", {"status": "unavailable"})
                row["comparison_replay_execution"] = data.get(
                    "replay_execution", {"status": "unavailable"})
                matched = data["alignment"]["matched_operations"]
                questions = data["summary"]["questions"]
                if any(type(n) is not int or n < 0
                       for n in (matched, questions)):
                    raise ValueError("Invalid comparison coverage counts.")
                row["matched_operations"] = matched
                row["questions"] = questions
                row["release_measurements"] = _release_measurements(data)
                for metric in ("token_f1", "embedding_cosine",
                               "evidence_jaccard", "citation_jaccard"):
                    row[metric] = data["summary"].get(metric)
                # Ensure the snapshot is serializable before exposing it
                # to either notebook or the compiled dashboard.
                _canonical_json(data)
            except Exception as error:
                row["snapshot_error"] = str(error)
                data = None
                row["snapshot"] = None
                row.pop("release_measurements", None)
            else:
                reports[stem] = data
        if run_notebooks and prepared and data is not None:
            try:
                with TemporaryDirectory(prefix="kilin-report-") as temp:
                    snapshot = Path(temp) / (stem + "_comparison.json")
                    # All views are compiled from exactly the same data.
                    snapshot.write_text(_canonical_json(data),
                                        encoding="utf-8")
                    source = ("Current batch comparison" if
                              row["snapshot"] == "current" else
                              "Saved historical comparison snapshot")
                    row["snapshot_sha256"] = sha256(
                        snapshot.read_bytes()).hexdigest()
                    _render_notebooks(prepared, row, snapshot, source,
                                      notebook_timeout)
            except Exception as error:
                row["snapshot_error"] = str(error)
        elif run_notebooks and data is None:
            for kind in prepared:
                row["notebooks"][kind]["status"] = "no report"
        try:
            row["issues"] = collect_issues(data, row)
        except Exception as error:
            # Preserve stage failures even if the report is malformed.
            row["issues"] = collect_issues(None, row)
            row["issues"].append({"severity": "critical",
                                  "code": "issue_collection.error",
                                  "message": str(error), "sequence": None})
        row["issue_groups"] = group_issues(row["issues"], data)
        row["coverage_findings"] = [i for i in row["issues"]
                                    if i["code"] in ("capability.incomplete",
                                                     "capability.changed",
                                                     "generator.coverage")]
        row["tool_findings"] = [i for i in row["coverage_findings"]
                                if i["code"] != "generator.coverage"]
        counts = {level: sum(issue["severity"] == level
                             for issue in row["issue_groups"])
                  for level in ("critical", "review", "info")}
        row["issue_counts"] = counts
        # An old snapshot must never certify a newly requested replay.
        row["execution_status"] = (
            "blocked" if counts["critical"] else
            "not analyzed" if data is None else
            "saved snapshot" if row["snapshot"] != "current" else
            "inspection only" if row.get("inspection") else "complete")
        if counts["critical"]:
            row["status"] = "CRITICAL"
        elif data is None:
            row["status"] = "NOT ANALYZED"
        elif counts["review"]:
            row["status"] = "REVIEW"
        else:
            row["status"] = "COMPLETE"
        results.append(row)
        print(f'{row["status"]}: {counts}', flush=True)
    return reports, results


def save_batch_summary(directory: Path, rows: list[dict], *,
                       overwrite: bool) -> tuple[Path, Path]:
    """Persist the current batch action log and readable issue summary.

    Parameters
    ----------
    directory: pathlib.Path
        Explicit directory to inspect or write artifacts in.
    rows: list[dict]
        Complete action and diagnostic rows from the current batch.
    overwrite: bool
        Explicitly permit replacing an existing regular artifact.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        JSON and Markdown summary paths.

    """
    from .analysis import _md, _md_table

    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Expected a real summary directory.")
    path = directory / "_batch_summary.json"
    markdown = directory / "_batch_summary.md"
    for target in (path, markdown):
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"Nonregular summary destination: {target}")
        if target.exists() and not overwrite:
            raise FileExistsError(str(target))
    ordered = sorted(rows, key=lambda row: row["experiment"])
    totals = _release_totals(rows)
    lines = ["# KíLín · Release verification", "",
             "Current invocation only. Execution, behavioral agreement and "
             "capability observations are separate; no quality verdict.", "",
             f"Current complete comparisons: **{totals['complete']} / "
             f"{len(rows)}**; blocking experiments: **{totals['blocked']}**; "
             f"fresh completed replays: **{totals['fresh']}**; "
             f"inspection-only comparisons: **{totals['inspection']}**.", ""]
    lines += _md_table(("Experiment", "Execution", "Replay", "Operations M/O",
                        "Questions M/O", "Exact answers", "Token F1",
                        "Review groups"), _release_rows(ordered))
    lines += ["M/O = matched / recorded original.", "",
              "## Measurement coverage", ""]
    lines += _md_table(("Boundary", "Matched", "Original", "Replay"), [
        (name, *(value.get(key) if value.get(key + "_snapshots") ==
                 totals["snapshots"] and totals["snapshots"] else None
                 for key in ("matched", "original", "replayed")))
        for name, value in totals["coverage"].items()])
    lines += ["## Behavioral agreement (current comparisons)", ""]
    lines += _md_table(("Observation", "Equal", "Available", "Total",
                        "Snapshots"), _release_agreement_rows(rows))
    lines += ["Missing values are unavailable, not zero. Snapshots shows "
              "the number of reports containing each metric. Saved snapshots "
              "are excluded from current aggregate measurements.", "",
              "## Recorded tool outcomes", ""]
    lines += _md_table(("Transition", "Questions"), sorted(
        totals["tool_transitions"].items()))
    lines += ["Historical generator hints describe only the reference at "
              "publication. Unknown tool completion is not proof of success.",
              "", "## Findings by experiment", ""]
    for row in ordered:
        lines += [f'## {_md(row["experiment"])}', ""]
        if row.get("archived_replay"):
            lines += [f'Previous replay archived: '
                      f'{_md(row["archived_replay"])}', ""]
        if row.get("inspection"):
            lines += [_md(row["inspection"]["message"]), ""]
        receipt = row.get("reference_publication", {})
        if receipt.get("status") == "inspection_only":
            lines += ["Published reference preserved; inspection sessions "
                      "appended.", ""]
        if row.get("coverage_findings"):
            lines += [f'Capability coverage observations: '
                      f'{len(row["coverage_findings"])}.', ""]
        if row.get("replay_execution"):
            lines += ["Replay execution provenance: " + _md(
                row["replay_execution"].get("status")), ""]
        groups = [group for group in row.get("issue_groups", [])
                  if group["severity"] != "info"]
        if groups:
            for group in groups:
                where = (f"Q{group['question_number']} · " if
                         group.get("question_number") is not None else "")
                lines += [f"### {where}{_md(group['code'])} "
                          f"({_md(group['severity'])})", ""]
                lines += _md_table(("Code", "Sequence", "Observation"), [
                    (issue["code"], issue["sequence"], issue["message"])
                    for issue in group["issues"]])
        else:
            lines += ["No critical or review issues recorded.", ""]
    lines += ["The JSON includes informational observations, metrics, and "
              "complete action logs. Inspect the executed per-workspace "
              "notebooks for details.", ""]
    _write_text(path, json.dumps(rows, indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n", overwrite=overwrite)
    _write_text(markdown, "\n".join(lines), overwrite=overwrite)
    return path, markdown
