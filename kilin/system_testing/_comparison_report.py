"""Present and persist comparison results without modifying workspaces.

Key Features
------------
- Summarize measurements, coverage, divergence and file differences.
- Escape recorded strings when rendering Markdown.
- Save deterministic JSON and Markdown beside the two workspaces.

Module Structure
----------------
- ``markdown_report`` renders a compact human-readable report.
- ``save_reports`` writes the corresponding sibling report files.

Examples
--------
>>> from kilin.system_testing import Experiment
>>> # Experiment("010_original").compare().save()

"""

from html import escape
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

__all__ = []


def _cell(value: object) -> str:
    """Escape arbitrary recorded text for a Markdown table cell.

    Parameters
    ----------
    value: object
        Recorded value or unavailable measurement.

    Returns
    -------
    str
        Safe single-line table cell text; None is represented as n/a.

    """
    if value is None:
        return "n/a"
    if type(value) is float:
        return f"{value:.4g}"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = escape(" ".join(str(value).split()), quote=False)
    for char in ("\\", "|", "`", "*", "_", "[", "]", "~"):
        text = text.replace(char, "\\" + char)
    return text


def _row(*values: object) -> str:
    """Render one table row without interpreting recorded content.

    Parameters
    ----------
    *values: object
        Ordered cell values.

    Returns
    -------
    str
        Markdown table row.

    """
    return "| " + " | ".join(_cell(value) for value in values) + " |"


def markdown_report(data: dict) -> str:
    """Render the comparison snapshot, keeping details in the JSON report.

    Parameters
    ----------
    data: dict
        Structured comparison snapshot.

    Returns
    -------
    str
        Markdown report with validation, coverage, operation and question
        metrics, settings and final file differences.

    """
    a, b = data["original"], data["replayed"]
    alignment = data["alignment"]
    summary = data["summary"]
    lines = ["# KíLín workspace comparison", "",
             "This is a measurement report, not a correctness verdict.", "",
             f"Alignment: **{_cell(alignment['status'])}**", ""]
    if alignment["reason"] is not None:
        lines.extend((f"Reason: {_cell(alignment['reason'])}", ""))
    lines.extend(("| Workspace | Path | Valid recording | Events | Commands |",
                  "|---|---|---|---:|---:|",
                  _row("Original", a["path"], a["valid"], a["event_count"],
                       len(a["operations"])),
                  _row("Replayed", b["path"], b["valid"], b["event_count"],
                       len(b["operations"])), "",
                  f"Matched commands: {alignment['matched_operations']}. "
                  f"Matched committed questions: {summary['questions']}.",
                  "", "## Validation", ""))
    for label, workspace in (("Original", a), ("Replayed", b)):
        lines.append(f"### {label}")
        lines.append("")
        issues = workspace["validation_issues"]
        if not issues:
            lines.append("No blocking validation issues.")
        for issue in issues:
            lines.append(f"- {_cell(issue['code'])}: "
                         f"{_cell(issue['message'])} "
                         f"(sequence {_cell(issue['sequence'])}, "
                         f"{_cell(issue['path'])})")
        for warning in workspace["validation_warnings"]:
            lines.append(f"- Warning: {_cell(warning['message'])}")
        lines.append("")
    lines.extend(("## Question metrics", "",
                  "Each mean includes only available measurements. "
                  "An empty evidence union is not a similarity of zero.",
                  "", "| Metric | Mean | Available | Questions |",
                  "|---|---:|---:|---:|"))
    for key in ("token_f1", "embedding_cosine", "evidence_jaccard",
                "citation_jaccard"):
        metric = summary[key]
        lines.append(_row(key, metric["mean"], metric["available"],
                          metric["total"]))
    lines.extend(("", f"Exact saved answers: {summary['exact_answers']}; "
                  f"normalized visible answers equal: "
                  f"{summary['normalized_equal_answers']}.", "",
                  "| Original seq. | Question | Token F1 | Cosine | "
                  "Evidence Jaccard | Citation Jaccard |",
                  "|---:|---|---:|---:|---:|---:|"))
    for op in data["operations"]:
        q = op.get("question")
        if q is None:
            continue
        evidence = q["evidence"]
        lines.append(_row(op["original_sequence"], q["question"],
                          q["answer"]["token_f1"],
                          q["answer"]["embedding"]["cosine"],
                          evidence["overlap"]["jaccard"] if evidence else None,
                          evidence["citations"]["overlap"]["jaccard"]
                          if evidence else None))
    evaluator = data["embedding_evaluation"]
    lines.extend(("", "Embedding evaluation: " + _cell(evaluator), "",
                  "## Operations and recorded duration", "",
                  "Durations are milliseconds at the recorded boundaries; "
                  "point events have no measured duration. Nested stage "
                  "times are not summed into wall time.", "",
                  "| Original seq. | Replay seq. | Operation | Original ms | "
                  "Replay ms | Settings differences |",
                  "|---:|---:|---|---:|---:|---:|"))
    for op in data["operations"]:
        lines.append(_row(op["original_sequence"], op["replayed_sequence"],
                          op["kind"], op["duration"]["original"]["ms"],
                          op["duration"]["replayed"]["ms"],
                          len(op["settings_differences"])))
    for op in data["operations"]:
        relocation = op.get("relocation")
        if relocation is not None:
            recorded = relocation["original"]
            outcome = ("physical rename recorded" if relocation[
                "physical_rename_replayed"] else "replay directory retained")
            lines.extend(("", "Project relocation at original sequence "
                          f"{op['original_sequence']}: "
                          f"{_cell(recorded['source_directory'])} to "
                          f"{_cell(recorded['destination_directory'])} "
                          f"({recorded['provenance']}); {outcome}."))
    for side in ("original", "replayed"):
        unmatched = alignment[side + "_unmatched"]
        if unmatched:
            lines.extend(("", f"{side.capitalize()} unmatched commands: "
                          + str(len(unmatched))))
    duration = summary["paired_command_duration_ms"]
    lines.extend(("", "Paired command duration (ms): "
                  f"{_cell(duration['original'])} / "
                  f"{_cell(duration['replayed'])}; "
                  f"{duration['available']}/{duration['total']} commands.",
                  "", "## Workspace token usage", "",
                  "Saved-turn usage and traced call usage have different "
                  "boundaries and must not be added together.", "",
                  "| Measurement | Original total | Original coverage | "
                  "Replay total | Replay coverage |",
                  "|---|---:|---|---:|---|"))
    for key, unit in (("usage", "turns"),
                      ("traced_chat_usage", "records")):
        for token in ("input_tokens", "output_tokens"):
            values = []
            for side in (a, b):
                metric = (side[key] or {}).get(token, {})
                available = metric.get("available_" + unit)
                total = metric.get("total_" + unit)
                values.extend((metric.get("recorded_total"),
                               f"{available}/{total}" if total is not None
                               else None))
            lines.append(_row(f"{key}.{token}", *values))
    state = data["final_state"]
    lines.extend(("", "## Final mapped state", "",
                  "Final snapshots can include unmatched suffixes.", "",
                  "| Observation | Equal | Mapped |", "|---|---:|---:|"))
    for key in ("name_equal", "archival_equal", "lineage_equal",
                "turn_prefix_equal"):
        records = state["conversations"]
        lines.append(_row(key, sum(r[key] for r in records), len(records)))
    records = state["feedback"]
    lines.append(_row("feedback_equal", sum(r["equal"] for r in records),
                      len(records)))
    lines.extend(("", "## Recorded settings differences", "",
                  "| Original seq. | Field | Original | Replayed |",
                  "|---:|---|---|---|"))
    for op in data["operations"]:
        for diff in op["settings_differences"]:
            lines.append(_row(op["original_sequence"], diff["field"],
                              diff["original"], diff["replayed"]))
    lines.extend(("", "## Per-question execution details", ""))
    for op in data["operations"]:
        q = op.get("question")
        if q is None:
            continue
        lines.extend((f"### Original sequence {op['original_sequence']}", "",
                      "| Observation | Original | Replayed |",
                      "|---|---|---|",
                      _row("Intent", q["intent"]["original"],
                           q["intent"]["replayed"]),
                      _row("Retrieval mode", q["retrieval_mode"]["original"],
                           q["retrieval_mode"]["replayed"]),
                      _row("Tool summary", q["tool"]["original"],
                           q["tool"]["replayed"]),
                      _row("Input tokens", q["usage"]["original"][
                          "input_tokens"], q["usage"]["replayed"][
                              "input_tokens"]),
                      _row("Output tokens", q["usage"]["original"][
                          "output_tokens"], q["usage"]["replayed"][
                              "output_tokens"]),
                      _row("Saved trace", q["original_trace"],
                           q["replayed_trace"]), ""))
        if op["rankings"]:
            lines.extend(("| Ranking channel | Occurrence | Exact Jaccard | "
                          "File Jaccard | Mean absolute rank shift |",
                          "|---|---:|---:|---:|---:|"))
            for row in op["rankings"]:
                metrics = row["overlap"] or {}
                lines.append(_row(row["event_type"], row["occurrence"],
                                  metrics.get("jaccard"),
                                  metrics.get("source_jaccard"),
                                  metrics.get("mean_absolute_rank_shift")))
            lines.append("")
        for ranking in op["rankings"]:
            for message in ranking["issues"]:
                lines.append(f"- {_cell(ranking['event_type'])}: "
                             f"{_cell(message)}")
        timed = [stage for stage in op["stages"] if any(
            stage[side]["total_duration_ms"] is not None
            for side in ("original", "replayed"))]
        if timed:
            lines.extend(("", "| Timed diagnostic stage | Original count | "
                          "Original ms | Replay count | Replay ms |",
                          "|---|---:|---:|---:|---:|"))
            for stage in timed:
                lines.append(_row(stage["event_type"],
                                  stage["original"]["count"],
                                  stage["original"]["total_duration_ms"],
                                  stage["replayed"]["count"],
                                  stage["replayed"]["total_duration_ms"]))
            lines.append("")
        for message in op["issues"] + q["issues"]:
            lines.append(f"- {_cell(message)}")
        error = q["answer"]["embedding"]["reason"]
        if error not in (None, "disabled"):
            lines.append(f"- Embedding unavailable: {_cell(error)}")
        lines.append("")
    lines.extend(("## Final file inventories", "",
                  "Compared by workspace-relative name, SHA-256, and size. "
                  "Byte differences do not establish semantic differences.",
                  "", "| Directory | Added | Removed | Changed | Unchanged |",
                  "|---|---:|---:|---:|---:|"))
    for directory, result in data["files"].items():
        changes = result["changes"]
        counts = ([len(changes[key]) for key in (
            "added", "removed", "changed", "unchanged")]
                  if changes is not None else [None] * 4)
        lines.append(_row(directory, *counts))
    for directory, result in data["files"].items():
        changes = result["changes"]
        lines.extend(("", f"### {_cell(directory)}", ""))
        for side in ("original", "replayed"):
            for issue in result[side]["issues"]:
                lines.append(f"- {_cell(side)}: {_cell(issue)}")
        if changes is not None:
            for kind in ("added", "removed", "changed"):
                for path in changes[kind]:
                    lines.append(f"- {kind}: {_cell(path)}")
    if data.get("report_issues"):
        lines.extend(("", "## Unavailable diagnostic values", ""))
        for issue in data["report_issues"]:
            lines.append(f"- {_cell(issue)}")
    lines.extend(("", "The JSON report contains the full paired answer text, "
                  "mapped conversation and turn IDs, file hashes, all stage "
                  "measurements, model-call metadata and recorded outcomes. "
                  "It may contain sensitive project information.", ""))
    return "\n".join(lines)


def save_reports(reference: Path, data: dict, markdown: str, *,
                 overwrite: bool) -> tuple[Path, Path]:
    """Write two sibling report files, never a file inside either workspace.

    Parameters
    ----------
    reference: pathlib.Path
        Resolved original workspace path.
    data: dict
        JSON-compatible report snapshot.
    markdown: str
        Human-readable report text.
    overwrite: bool
        Explicit permission to replace existing regular report files.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Saved JSON and Markdown paths.

    Notes
    -----
    Each file is published atomically. Both files are staged before either is
    published, but the two-file publication is not a filesystem transaction.
    Existing files or links are refused by default. No path is followed for
    report writes, and no existing directory is removed.

    """
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a Boolean.")
    base = reference.name.removesuffix("_original") + "_comparison"
    paths = (reference.with_name(base + ".json"),
             reference.with_name(base + ".md"))
    contents = (json.dumps(data, indent=2, ensure_ascii=False,
                           sort_keys=True, allow_nan=False) + "\n", markdown)
    for path in paths:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Not a regular report destination: {path}")
        if not overwrite and path.exists():
            raise FileExistsError(f"Comparison report already exists: {path}")
    temporary = []
    try:
        for path, content in zip(paths, contents):
            with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                    prefix="." + path.name + ".",
                                    dir=path.parent, delete=False) as stream:
                temporary.append(Path(stream.name))
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        for source, path in zip(temporary, paths):
            if overwrite:
                # Replace the directory entry rather than following a link.
                # A concurrent change to a directory still fails normally.
                if path.is_symlink():
                    raise ValueError(f"Linked report destination: {path}")
                os.replace(source, path)
            else:
                # Publish exclusively, so even a concurrent creator cannot
                # be overwritten between the preflight and publication.
                os.link(source, path)
    finally:
        for source in temporary:
            source.unlink(missing_ok=True)
    return paths
