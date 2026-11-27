"""Present KíLín comparison snapshots without rerunning an experiment.

Key Features
------------
- Read-only report discovery and visual, coverage-aware summaries.
- Side-by-side saved answers and expandable execution diagnostics.
- Optional exact model texts verified against recorded SHA-256 digests.
- Independent visual Markdown exports alongside native reports.

Module Structure
----------------
- Report loading and presentation helpers.
- Overview, operation and question renderers.
- Restricted local trace lookup and Markdown export.

Examples
--------
>>> from pathlib import Path
>>> from kilin.system_testing import analysis
>>> report = Path('workspaces/020_hello_world_comparison.json')
>>> data = analysis.load_report(report)  # doctest: +SKIP
>>> analysis.overview(data)  # doctest: +SKIP

No model calls or network access are required. HTML is
escaped, and optional Markdown rendering disables active links and images.
The JSON snapshot is authoritative for the displayed metrics; inspecting
it does not revalidate the workspaces.

"""

from __future__ import annotations

from collections import Counter
from difflib import unified_diff
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

__all__ = [
    "load_report", "discover_reports", "overview", "batch_overview",
    "question_explorer", "operation_explorer", "analysis_markdown",
    "save_analysis", "event_inventory", "event_inventory_html",
]

SIDES = ("original", "replayed")
METRICS = (
    ("token_f1", "Token F1"), ("embedding_cosine", "Embedding cosine"),
    ("evidence_jaccard", "Evidence Jaccard"),
    ("citation_jaccard", "Citation Jaccard"),
)
SUFFIX = "_comparison.json"

# Scope every selector to the report. Jupyter's theme variables allow the
# same HTML output to be legible in light and dark themes.
STYLE = """<style>
.ks-review {font-family:var(--jp-ui-font-family,system-ui,sans-serif);
 color:var(--jp-ui-font-color1,#1e293b); line-height:1.5; font-size:13px;
 --ks-bg:var(--jp-layout-color1,#fff);
 --ks-muted:var(--jp-ui-font-color2,#526276);
 --ks-border:var(--jp-border-color2,#d6dce4);
 --ks-soft:var(--jp-layout-color2,#f4f6f9);}
.ks-review h2,.ks-review h3,.ks-review h4 {margin:18px 0 8px;}
.ks-review .ks-caption {color:var(--ks-muted);font-size:12px;}
.ks-review .ks-cards {display:grid;grid-template-columns:
 repeat(4,minmax(0,1fr));gap:10px;margin:12px 0;}
.ks-review .ks-card {padding:12px;border:1px solid var(--ks-border);
 border-radius:8px;background:var(--ks-bg);min-width:0;}
.ks-review .ks-value {font-size:21px;font-weight:650;overflow-wrap:anywhere;}
.ks-review .ks-ok {color:#16804a;} .ks-review .ks-warn {color:#bc7710;}
.ks-review .ks-bad {color:#c43d42;}
.ks-review .ks-muted {color:var(--ks-muted);}
.ks-review .ks-scroll {overflow-x:auto;}
.ks-review table {border-collapse:collapse;width:100%;margin:8px 0 14px;}
.ks-review th,.ks-review td {padding:6px 9px;vertical-align:top;
 border-bottom:1px solid var(--ks-border);text-align:left;
 overflow-wrap:anywhere;}
.ks-review th {background:var(--ks-soft);font-weight:600;}
.ks-review .ks-pair {display:grid;grid-template-columns:
 repeat(2,minmax(0,1fr));gap:12px;margin:10px 0;}
.ks-review .ks-pane {min-width:0;overflow-x:auto;padding:10px 12px;
 border:1px solid var(--ks-border);border-radius:6px;}
.ks-review .ks-original {border-top:3px solid #8493a7;}
.ks-review .ks-replayed {border-top:3px solid #3476b8;}
.ks-review .ks-pane h4 {margin:0 0 10px;}
.ks-review pre {white-space:pre-wrap;overflow-wrap:anywhere;
 text-align:left;margin:5px 0;padding:7px;background:var(--ks-soft);}
.ks-review .ks-text {white-space:pre-wrap;overflow-wrap:anywhere;}
.ks-review details {border:1px solid var(--ks-border);border-radius:6px;
 padding:9px 12px;margin:8px 0;}
.ks-review summary {cursor:pointer;font-weight:600;overflow-wrap:anywhere;}
.ks-review summary + div {margin-top:10px;}
.ks-review .ks-bar {height:7px;background:var(--ks-soft);border-radius:4px;
 min-width:75px;width:100%;overflow:hidden;margin:4px 0;}
.ks-review .ks-fill {height:100%;border-radius:4px;background:#3476b8;}
.ks-review .ks-fill-original {background:#8493a7;}
.ks-review .ks-diff-add {color:#16804a;}
.ks-review .ks-diff-remove {color:#c43d42;}
@media(max-width:760px) {.ks-review .ks-pair {grid-template-columns:1fr;}
 .ks-review .ks-cards {grid-template-columns:repeat(2,minmax(0,1fr));}}
@media(max-width:420px) {.ks-review .ks-cards {grid-template-columns:1fr;}}
</style>"""


def _dict(value):
    """Return a mapping or an empty mapping for unavailable sections.

    Parameters
    ----------
    value: object
        Recorded value to inspect or format.

    Returns
    -------
    dict
        Original mapping or an empty dictionary.

    """
    return value if isinstance(value, dict) else {}


def _list(value):
    """Return a list or an empty list for unavailable inventories.

    Parameters
    ----------
    value: object
        Recorded value to inspect or format.

    Returns
    -------
    list
        Original list or an empty list.

    """
    return value if isinstance(value, list) else []


def _at(value, *keys):
    """Retrieve nested report fields without substituting missing values.

    Parameters
    ----------
    value: object
        Recorded value to inspect or format.
    *keys: str
        Nested dictionary keys, in traversal order.

    Returns
    -------
    object
        Nested value, or None when unavailable.

    """
    for key in keys:
        value = _dict(value).get(key)
    return value


def _number(value):
    """Retain finite numeric measurements, excluding Boolean flags.

    Parameters
    ----------
    value: object
        Recorded value to inspect or format.

    Returns
    -------
    int or float or None
        Finite number, or None.

    """
    return (value if type(value) in (int, float) and math.isfinite(value)
            else None)


def _format(value):
    """Format measurements while distinguishing null, zero and false.

    Parameters
    ----------
    value: object
        Recorded value to inspect or format.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.4g}" if math.isfinite(value) else "n/a"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _seconds(value):
    """Convert a recorded millisecond value to a labeled duration.

    Parameters
    ----------
    value: object
        Recorded value to inspect or format.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    value = _number(value)
    return "n/a" if value is None else f"{value / 1000:.3f} s"


def _e(value):
    """Escape every data-derived value before insertion into HTML.

    Parameters
    ----------
    value: object
        Recorded value to inspect or format.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    return escape(_format(value), quote=True)


def _table(headers, rows):
    """Create a scrollable HTML table with escaped cells.

    Parameters
    ----------
    headers: sequence[str]
        Column headings in display order.
    rows: iterable
        Rows of values, one sequence per table row.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    head = "".join(f"<th>{_e(x)}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{_e(x)}</td>" for x in row)
                   + "</tr>" for row in rows)
    return (f'<div class="ks-scroll"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def _section(title, body, *, expanded=False):
    """Create a native collapsible panel, without JavaScript.

    Parameters
    ----------
    title: str
        Panel heading, escaped before insertion.
    body: str
        Trusted, already rendered HTML fragment.
    expanded: bool
        Whether the collapsible panel starts open.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    state = " open" if expanded else ""
    return (f'<details{state}><summary>{_e(title)}</summary>'
            f"<div>{body}</div></details>")


def _wrap(body):
    """Wrap an HTML fragment in scoped Jupyter-compatible styling.

    Parameters
    ----------
    body: str
        Trusted, already rendered HTML fragment.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    return STYLE + f'<div class="ks-review">{body}</div>'


def _raw(value):
    """Present complete escaped JSON for detailed diagnostic inspection.

    Parameters
    ----------
    value: object
        Recorded value to inspect or format.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    return "<pre>" + _e(json.dumps(value, indent=2, ensure_ascii=False)) + (
        "</pre>"
    )


def _pair(left, right):
    """Place trusted rendered fragments in labeled side-by-side panes.

    Parameters
    ----------
    left: str
        Original-side text or rendered content.
    right: str
        Replay-side text or rendered content.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    return ('<div class="ks-pair">'
            '<div class="ks-pane ks-original"><h4>Original</h4>'
            f'{left}</div><div class="ks-pane ks-replayed">'
            f'<h4>Replay</h4>{right}</div></div>')


def _card(label, value, detail="", tone="muted"):
    """Create a compact metric tile with an optional status tone.

    Parameters
    ----------
    label: str
        Human-readable label.
    value: object
        Recorded value to inspect or format.
    detail: str
        Optional supplementary text.
    tone: str
        Trusted CSS status class suffix.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    return (f'<div class="ks-card"><div class="ks-caption">{_e(label)}'
            f'</div><div class="ks-value ks-{tone}">{_e(value)}</div>'
            f'<div class="ks-caption">{_e(detail)}</div></div>')


def _bar(value, maximum=1, *, original=False):
    """Scale a nonnegative finite number; missing has no bar.

    Parameters
    ----------
    value: object
        Recorded value to inspect or format.
    maximum: float
        Positive reference scale for the bar.
    original: bool
        Use the original-side bar style.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    value = _number(value)
    if value is None or maximum <= 0 or value < 0:
        return ""
    width = min(value / maximum, 1) * 100
    cls = "ks-fill ks-fill-original" if original else "ks-fill"
    return (f'<div class="ks-bar"><div class="{cls}" '
            f'style="width:{width:.4f}%"></div></div>')


def _metric_value(value):
    """Format a similarity on its native scale, including negatives.

    Parameters
    ----------
    value: object
        Recorded value to inspect or format.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    value = _number(value)
    return "n/a" if value is None else f"{value:.4f}"


def _questions(data):
    """Return aligned committed questions in report order.

    Parameters
    ----------
    data: dict
        Saved comparison snapshot; never modified.

    Returns
    -------
    list[dict]
        Aligned question operations in chronological order.

    """
    return [op for op in data["operations"]
            if isinstance(op.get("question"), dict)]


def _question_metrics(op):
    """Extract the four similarity measurements from one question pair.

    Parameters
    ----------
    op: dict
        One aligned operation from the comparison.

    Returns
    -------
    tuple
        Token F1, embedding cosine, evidence and citation overlap.

    """
    q = op["question"]
    return (
        _at(q, "answer", "token_f1"),
        _at(q, "answer", "embedding", "cosine"),
        _at(q, "evidence", "overlap", "jaccard"),
        _at(q, "evidence", "citations", "overlap", "jaccard"),
    )


def _agreement(data):
    """Count observed equality separately from measurement coverage.

    Parameters
    ----------
    data: dict
        Saved comparison snapshot; never modified.

    Returns
    -------
    list[tuple]
        Agreement counts with measured and total coverage.

    """
    questions = _questions(data)
    rows = []
    for field, label in (("intent", "Intent"),
                         ("retrieval_mode", "Retrieval mode"),
                         ("tool", "Functional tool result")):
        if field == "tool":
            values = []
            for op in questions:
                q = op["question"]
                if "tool_comparison" in q:
                    values.append(_at(q, "tool_comparison",
                                      "functional_equal"))
                else:
                    pair = _dict(q.get("tool"))
                    if all(side in pair for side in SIDES):
                        values.append(pair["original"] == pair["replayed"])
            measured = [v for v in values if type(v) is bool]
            rows.append((label, sum(measured), len(measured),
                         len(questions)))
            continue
        pairs = [_dict(op["question"].get(field)) for op in questions]
        available = [pair for pair in pairs if all(
            side in pair and (field == "tool" or pair[side] is not None)
            for side in SIDES)]
        rows.append((label, sum(pair["original"] == pair["replayed"]
                                for pair in available),
                     len(available), len(questions)))
    return rows


def _mapped_state(data):
    """Count recorded final-state equality only for available flags.

    Parameters
    ----------
    data: dict
        Saved comparison snapshot; never modified.

    Returns
    -------
    list[tuple]
        Mapped-state agreement counts and coverage.

    """
    state = data["final_state"]
    rows = []
    for key, label in (("name_equal", "Conversation name"),
                       ("archival_equal", "Archival state"),
                       ("lineage_equal", "Lineage"),
                       ("turn_prefix_equal", "Mapped turn prefix"),
                       ("feedback_equal", "Feedback")):
        items = (_list(state.get("feedback")) if key == "feedback_equal"
                 else _list(state.get("conversations")))
        flag = "equal" if key == "feedback_equal" else key
        observed = [item[flag] for item in items
                    if type(item.get(flag)) is bool]
        rows.append((label, sum(observed), len(observed), len(items)))
    return rows


def _alignment_label(data):
    """Describe strict alignment with an optional inspection note.

    Parameters
    ----------
    data: dict
        Saved comparison snapshot; never modified.

    Returns
    -------
    str
        Alignment label, including inspection-only annotation.

    """
    if data.get("inspection", data.get("batch_inspection")):
        return "diverged (inspection only)"
    return data["alignment"].get("status")


def _status(data):
    """Translate only structural alignment status into a display tone.

    Parameters
    ----------
    data: dict
        Saved comparison snapshot; never modified.

    Returns
    -------
    str
        CSS status tone.

    """
    status = data["alignment"].get("status")
    return {"complete": "ok", "invalid_recording": "bad",
            "diverged": "warn"}.get(status, "muted")


def load_report(path: str | Path) -> dict:
    """Read one native format-version-1 comparison snapshot.

    Parameters
    ----------
    path : str or pathlib.Path
        Local regular JSON comparison report. Symlinks are rejected.

    Returns
    -------
    dict
        Saved report, with its original measurements unchanged.

    Raises
    ------
    ValueError
        Unsupported schema or malformed top-level sections.
    OSError
        Report cannot be read.

    """
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular comparison report: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if type(_at(data, "format_version")) is not int or (
            data["format_version"] != 1):
        raise ValueError("Unsupported comparison report format version.")
    for key in ("alignment", "summary", "original", "replayed", "files",
                "final_state", "embedding_evaluation"):
        if not isinstance(data.get(key), dict):
            raise ValueError(f"Invalid report section: {key}")
    for key in ("operations", "report_issues"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"Invalid report section: {key}")
    if any(not isinstance(op, dict) for op in data["operations"]):
        raise ValueError("Invalid operation record.")
    if data["alignment"].get("status") not in (
            "complete", "diverged", "invalid_recording"):
        raise ValueError("Unknown alignment status.")
    return data


def discover_reports(directory: str | Path) -> tuple[dict, dict]:
    """Discover direct native report files, isolating unreadable reports.

    Parameters
    ----------
    directory : str or pathlib.Path
        Real workspace parent directory.

    Returns
    -------
    tuple[dict, dict]
        Snapshots and error messages keyed by experiment stem. Discovery
        does not use absolute paths embedded in the recorded reports.

    """
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Expected a real directory: {directory}")
    reports, errors = {}, {}
    for path in sorted(directory.glob(f"*{SUFFIX}")):
        stem = path.name.removesuffix(SUFFIX)
        try:
            reports[stem] = load_report(path)
        except (OSError, ValueError, TypeError) as error:
            errors[stem] = str(error)
    return reports, errors


def _diagnostics(data):
    """Show validation, unmatched operations, and report-level issues.

    Parameters
    ----------
    data: dict
        Saved comparison snapshot; never modified.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    blocks = []
    for side in SIDES:
        value = data[side]
        rows = []
        for key, label in (("validation_issues", "Issue"),
                           ("validation_warnings", "Warning")):
            for issue in _list(value.get(key)):
                rows.append((label, issue.get("code"), issue.get("sequence"),
                             issue.get("path"), issue.get("message")))
        body = _table(("Level", "Code", "Sequence", "Path", "Message"), rows)
        blocks.append(body if rows else "<p>No recorded diagnostics.</p>")
    alignment = data["alignment"]
    body = _pair(*blocks)
    if alignment.get("reason") is not None:
        body += f'<p>{_e(alignment["reason"])}</p>'
    if data.get("inspection", data.get("batch_inspection")):
        note = data.get("inspection", data.get("batch_inspection"))
        body += "<p>" + _e(note["message"]) + "</p>"
    if data.get("reference_publication"):
        body += _section("Reference publication", _raw(
            data["reference_publication"]))
    if data.get("replay_execution"):
        body += _section("Replay execution provenance", _raw(
            data["replay_execution"]))
    for key in ("original_unmatched", "replayed_unmatched"):
        rows = _list(alignment.get(key))
        if rows:
            body += _section(key.replace("_", " ").title(), _raw(rows))
    if data["report_issues"]:
        body += _section("Report issues", _raw(data["report_issues"]),
                         expanded=True)
    return body


def _timings(data):
    """Plot paired outer command durations on one shared scale.

    Parameters
    ----------
    data: dict
        Saved comparison snapshot; never modified.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    ops = data["operations"]
    times = [_number(_at(op, "duration", side, "ms"))
             for op in ops for side in SIDES]
    maximum = max([x for x in times if x is not None] + [1])
    body = ('<p class="ks-caption">Original: slate · Replay: blue. '
            'One common scale; missing durations have no bar.</p>')
    for op in ops:
        title = f'{op.get("original_sequence")} · {op.get("kind")}'
        q = _at(op, "question", "question")
        if q is not None:
            title += " · " + str(q)
        body += f'<p><b>{_e(title)}</b></p><div class="ks-pair">'
        for side in SIDES:
            value = _at(op, "duration", side, "ms")
            body += (f'<div>{_e(side.title())}: {_e(_seconds(value))}'
                     + _bar(value, maximum, original=side == "original")
                     + "</div>")
        body += "</div>"
    return body if ops else "<p>No paired operations.</p>"


def _state(data):
    """Summarize workspace-level settings, tokens and final inventories.

    Parameters
    ----------
    data: dict
        Saved comparison snapshot; never modified.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    rows = []
    for key, label in (("usage", "Saved turn"),
                       ("traced_chat_usage", "Traced call")):
        for kind in ("input_tokens", "output_tokens"):
            cells = [f"{label} · {kind}"]
            for side in SIDES:
                entry = _dict(_at(data, side, key, kind))
                coverage = ("available_turns", "total_turns") if key == (
                    "usage") else ("available_records", "total_records")
                cells.extend((_format(entry.get("recorded_total")),
                              f'{_format(entry.get(coverage[0]))}/'
                              f'{_format(entry.get(coverage[1]))}'))
            rows.append(cells)
    body = ("<h3>Token usage and coverage</h3>"
            '<p class="ks-caption">Saved-turn and traced-call counters '
            'overlap; do not add them.</p>' + _table(
                ("Boundary", "Original", "Coverage", "Replay", "Coverage"),
                rows))
    diffs = [(op.get("original_sequence"), item)
             for op in data["operations"]
             for item in _list(op.get("settings_differences"))]
    body += "<h3>Recorded settings differences</h3>"
    body += (_table(("Sequence", "Difference"), diffs) if diffs else
             "<p>No differences reported for the matched operations.</p>")
    rows = []
    for kind, inventory in data["files"].items():
        changes = _dict(_at(inventory, "changes"))
        rows.append([kind] + [len(changes[k]) if isinstance(
            changes.get(k), list) else None for k in
            ("added", "removed", "changed", "unchanged")])
    body += "<h3>Final file inventories</h3>" + _table(
        ("Directory", "Added", "Removed", "Changed", "Unchanged"), rows)
    body += _section("Complete file identities and inventory issues",
                     _raw(data["files"]))
    body += "<h3>Final mapped state</h3>" + _table(
        ("Observation", "Equal", "Available", "Mapped"), _mapped_state(data))
    body += _section("Final mapped conversation and feedback state",
                     _raw(data["final_state"]))
    body += _section("Recorded environments and model identities", _pair(
        *[_raw({k: data[s].get(k) for k in ("environments", "models")})
          for s in SIDES]))
    return body


def overview(data: dict) -> str:
    """Render a visual overview of a comparison snapshot.

    Parameters
    ----------
    data : dict
        Native version-1 comparison report.

    Returns
    -------
    str
        HTML containing alignment, coverage, per-question similarities,
        paired timings, validation and workspace-state diagnostics.

    """
    a, s = data["alignment"], data["summary"]
    ops, questions = data["operations"], _questions(data)
    diffs = sum(len(_list(op.get("settings_differences"))) for op in ops)
    matched = a.get("matched_operations")
    body = ("<h2>Workspace comparison</h2>"
            '<p class="ks-caption">Saved measurement snapshot. '
            'Similarity is not a correctness or quality verdict.</p>')
    body += '<div class="ks-cards">' + _card(
        "Alignment", _alignment_label(data), a.get("reason") or "",
        _status(data))
    body += _card("Matched operations", matched,
                  "Original / replay: " + " / ".join(_format(len(
                      _list(data[k].get("operations")))) for k in SIDES))
    body += _card("Matched questions", s.get("questions"),
                  f'{_format(s.get("exact_answers"))} exact saved answers')
    body += _card("Settings differences", diffs, "Matched operations")
    for key, label in METRICS:
        metric = _dict(s.get(key))
        body += _card(label, _metric_value(metric.get("mean")),
                      f'{_format(metric.get("available"))}/'
                      f'{_format(metric.get("total"))} available')
    body += "</div>"
    body += _table(("Workspace", "Valid", "Events", "Commands"), [
        (side.title(), data[side].get("valid"), data[side].get("event_count"),
         len(_list(data[side].get("operations")))) for side in SIDES])
    body += _section("Validation and alignment diagnostics", _diagnostics(
        data), expanded=a.get("status") != "complete")
    body += "<h3>Question agreement</h3>"
    if not questions:
        body += "<p>No aligned committed question pairs.</p>"
    else:
        body += _table(("Question", "Exact", "Normalized", "Token F1",
                        "Cosine", "Evidence", "Citation",
                            "Intent O / R"), [
            (f'Q{i} · {op["question"].get("question")}',
             _at(op, "question", "answer", "exact_equal"),
             _at(op, "question", "answer", "normalized_equal"),
             *(_metric_value(x) for x in _question_metrics(op)),
             " / ".join(_format(_at(op, "question", "intent", side))
                        for side in SIDES))
            for i, op in enumerate(questions, 1)])
        for i, op in enumerate(questions, 1):
            value = _question_metrics(op)[0]
            body += (f'<div>Q{i} · Token F1: {_e(_metric_value(value))}</div>'
                     + _bar(value))
    body += "<h3>Recorded behavior agreement</h3>" + _table(
        ("Observation", "Equal", "Available", "Question pairs"),
        _agreement(data))
    body += ('<p class="ks-caption">Two null tool summaries are equal '
             'recorded absences, not proof that no tool ran.</p>')
    body += "<h3>Recorded operation durations</h3>" + _timings(data)
    duration = _dict(s.get("paired_command_duration_ms"))
    body += ('<p><b>Paired measured total:</b> '
             f'{_e(_seconds(duration.get("original")))} / '
             f'{_e(_seconds(duration.get("replayed")))} · '
             f'{_e(duration.get("available"))}/{_e(duration.get("total"))} '
             'operations. Nested diagnostics are not summed.</p>')
    body += _state(data)
    body += _section("Embedding evaluation", _raw(
        data["embedding_evaluation"]))
    return _wrap(body)


def event_inventory(data: dict) -> dict:
    """Count events in aligned commands and outside command boundaries.

    Prefer independent counts from PR5. Legacy reports require complete
    alignment and reconciled totals before claiming full coverage.

    Parameters
    ----------
    data: dict
        Saved comparison snapshot; never modified.

    Returns
    -------
    dict
        Count coverage, per-side inventories and event-type
        differences.

    """
    counts = {side: Counter() for side in SIDES}
    expected = {side: _at(data, side, "event_count") for side in SIDES}
    independent = all(isinstance(_at(data, s, "event_counts"), dict)
                      for s in SIDES)
    complete = independent or data["alignment"].get("status") == "complete"
    for side in SIDES:
        if independent:
            raw = data[side]["event_counts"]
            if all(isinstance(k, str) and type(v) is int and v >= 0
                   for k, v in raw.items()):
                counts[side].update(raw)
            else:
                complete = False
            if sum(counts[side].values()) != expected[side]:
                complete = False
            continue
        outside = _at(data, side, "outside_commands", "event_counts")
        if not isinstance(outside, dict):
            complete = False
        for name, count in _dict(outside).items():
            if type(count) is int and count >= 0:
                counts[side][name] += count
            else:
                complete = False
        for op in data["operations"]:
            for stage in _list(op.get("stages")):
                record = stage.get(side)
                if record is None:
                    continue
                name = stage.get("event_type")
                count = _at(record, "count")
                if (not isinstance(name, str) or type(count) is not int
                        or count < 0):
                    complete = False
                    continue
                counts[side][name] += count
        if sum(counts[side].values()) != expected[side]:
            complete = False
    differences = []
    if complete:
        for name in sorted(counts["original"].keys() |
                           counts["replayed"].keys()):
            old, new = (counts[side][name] for side in SIDES)
            if old != new:
                differences.append({"event_type": name, "original": old,
                                    "replayed": new, "delta": new - old})
    return {"complete": complete, "expected": expected,
            "counts": {side: dict(counts[side]) for side in SIDES},
            "differences": differences}


def _local_events(root, kinds):
    """Read local trace nodes of selected types, including merged ends.

    Parameters
    ----------
    root: pathlib.Path
        Explicit local repository or workspace directory.
    kinds: set[str]
        Recorded event types to include.

    Returns
    -------
    dict or None
        Local events and read errors, or None for an omitted root.

    """
    if root is None:
        return None
    root = Path(root)
    trace_dir = root / "logs" / "traces"
    if (root.is_symlink() or (root / "logs").is_symlink() or
            trace_dir.is_symlink() or not trace_dir.is_dir()):
        return {"error": "Local trace directory unavailable."}
    records, errors = [], []
    # Include archived conversation traces, but never follow links or
    # read unrelated JSON outside the trace tree.
    for path in sorted(trace_dir.rglob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            relative = path.relative_to(root).as_posix()
            safe = _trace_path(root, relative)
            trace = json.loads(safe.read_text(encoding="utf-8"))
            for node in _trace_events(trace):
                for key in ("event", "completion"):
                    meta = _dict(node.get(key))
                    if meta.get("event_type") in kinds:
                        record = {"path": relative, "metadata": meta}
                        if key == "event":
                            record.update(details=node.get("details"),
                                          exact_content=node.get(
                                              "exact_content"))
                        records.append(record)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            errors.append(f"{path.name}: {error}")
    records.sort(key=lambda row: row["metadata"].get("sequence", 0))
    return {"events": records, "errors": errors}


def event_inventory_html(data: dict, *, original=None, replayed=None,
                         include_details: bool = False) -> str:
    """Render reconciled event counts and optional local trace details.

    Local event text is for inspection only: it is not hash-verified or
    used to modify the saved comparison metrics. Events of a changed type
    are listed on both sides without claiming individual event pairing.

    Parameters
    ----------
    data: dict
        Saved comparison snapshot; never modified.
    original: str or pathlib.Path or None
        Explicit local reference directory, if available.
    replayed: str or pathlib.Path or None
        Explicit local replay directory, if available.
    include_details: bool
        Include locally read event details, not verified against the
        report.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    result = event_inventory(data)
    body = "<h2>Event inventory</h2>"
    body += _table(("Side", "Reported total", "Counted in report"), [
        (side, result["expected"][side],
         sum(result["counts"][side].values())) for side in SIDES])
    if not result["complete"]:
        return _wrap(body + "<p>Complete event inventory unavailable; "
                     "do not interpret partial counts as differences.</p>")
    differences = result["differences"]
    if differences:
        body += _table(("Event type", "Original", "Replay", "Delta"), [
            (r["event_type"], r["original"], r["replayed"], r["delta"])
            for r in differences])
        if include_details:
            kinds = {item["event_type"] for item in differences}
            body += _section("Local events of changed types", _pair(
                *[_raw(_local_events(root, kinds))
                  for root in (original, replayed)]))
            body += ("<p>Local trace details may differ from the saved "
                     "snapshot if the workspace was changed.</p>")
    else:
        body += "<p>Every recorded event-type count agrees.</p>"
    names = sorted(set(result["counts"]["original"]) |
                   set(result["counts"]["replayed"]))
    body += _section("Full event inventory", _table(
        ("Event type", "Original", "Replay"), [
            (name, *(result["counts"][side].get(name, 0) for side in SIDES))
            for name in names]))
    body += ("<p>Count differences are descriptive; they are not alone "
             "evidence of a replay defect.</p>")
    return _wrap(body)


def batch_overview(reports: dict, errors: dict | None = None,
                   *, runs: list | None = None) -> str:
    """Render cross-experiment measurements without a composite score.

    Parameters
    ----------
    reports : dict
        Comparison snapshots keyed by experiment stem.
    errors : dict, optional
        Report discovery errors keyed by stem.
    runs : list, optional
        Current batch execution rows. These remain separate from saved
        report measurements so an old report cannot imply a new run passed.

    Returns
    -------
    str
        Visual dashboard and isolated run or report errors.

    """
    body = "<h2>Experiment overview</h2>"
    if runs is not None and any("status" in row for row in runs):
        counts = Counter(row.get("status", "NOT ANALYZED") for row in runs)
        body += '<div class="ks-cards">' + "".join(
            _card(label, counts.get(label, 0), tone=tone)
            for label, tone in (("CRITICAL", "bad"), ("REVIEW", "warn"),
                                ("COMPLETE", "ok"),
                                ("NOT ANALYZED", "muted"))) + "</div>"
        body += ("<p>Critical: structural or execution failure. Review: "
                 "recorded differences or warnings. Complete is not an "
                 "answer-quality judgment. Counts refer to diagnostic "
                 "groups; underlying findings remain below.</p>")
        ordered = sorted(runs, key=lambda row: (
            {"CRITICAL": 0, "REVIEW": 1, "NOT ANALYZED": 2,
             "COMPLETE": 3}.get(row.get("status"), 4),
            row.get("experiment", "")))
        body += _table(("Experiment", "Status", "Snapshot", "Replay",
                        "Alignment", "Questions", "Critical", "Review",
                        "Coverage findings",
                        "Comparison notebook", "Answer notebook"), [
            (r.get("experiment"), r.get("status"), r.get("snapshot"),
             r.get("replay_status"),
             ("diverged (inspection only)" if r.get("inspection")
              else r.get("alignment")), r.get("questions"),
             _at(r, "issue_counts", "critical"),
             _at(r, "issue_counts", "review"),
             len(r.get("coverage_findings", [])),
             _at(r, "notebooks", "comparison", "status"),
             _at(r, "notebooks", "answers", "status")) for r in ordered])
        for row in ordered:
            issues = _list(row.get("issues"))
            if not issues:
                continue
            important = [i for i in issues if i["severity"] != "info"]
            info = [i for i in issues if i["severity"] == "info"]
            def issue_table(items):
                """Render a table of grouped or individual batch findings.

                Parameters
                ----------
                items: list
                    Recorded items in display order.

                Returns
                -------
                str
                    Rendered or formatted text.

                """
                return _table(("Severity", "Code", "Sequence", "Message"),
                              [(i["severity"], i["code"], i["sequence"],
                                i["message"]) for i in items])
            section = ""
            if row.get("issue_groups"):
                section += _table(("Severity", "Diagnostic group", "Count"), [
                    (g["severity"], g["code"], len(g["issues"]))
                    for g in row["issue_groups"] if g["severity"] != "info"])
            if important:
                section += issue_table(important)
            if info:
                section += _section("Informational observations",
                                    issue_table(info))
            body += _section(f'{row["experiment"]} · {row["status"]}',
                             section, expanded=bool(important))
    if not reports:
        body += "<p>No comparison snapshots available.</p>"
    rows = []
    for name, data in sorted(reports.items()):
        summary = data["summary"]
        metrics = []
        for key, _ in METRICS:
            item = _dict(summary.get(key))
            metrics.append(f'{_metric_value(item.get("mean"))} '
                           f'({_format(item.get("available"))}/'
                           f'{_format(item.get("total"))})')
        duration = _dict(summary.get("paired_command_duration_ms"))
        rows.append((name, _alignment_label(data),
                     data["alignment"].get("matched_operations"),
                     summary.get("questions"), *metrics,
                     _seconds(duration.get("original")),
                     _seconds(duration.get("replayed")),
                     f'{_format(duration.get("available"))}/'
                     f'{_format(duration.get("total"))}'))
    if rows:
        body += _table(("Experiment", "Alignment", "Operations", "Questions",
                        *(label + " (coverage)" for _, label in METRICS),
                        "Original time", "Replay time", "Timed pairs"), rows)
        for name, data in sorted(reports.items()):
            value = _at(data, "summary", "token_f1", "mean")
            body += (f'<div>{_e(name)} · F1: {_e(_metric_value(value))}</div>'
                     + _bar(value))
    body += ('<p class="ks-caption">Each experiment has its own coverage. '
             'No cross-experiment average or acceptance threshold.</p>')
    if errors:
        body += "<h3>Report discovery errors</h3>" + _table(
            ("Experiment", "Error"), sorted(errors.items()))
    if runs is not None and not any("status" in row for row in runs):
        body += "<h3>Current batch actions</h3>" + _table(
            ("Experiment", "Notebook", "Notebook error", "Alignment",
             "Replay error", "Comparison error", "Save error",
             "Analysis error"), [
                (row.get("experiment"), row.get("notebook_status"),
                 row.get("notebook_error"), row.get("alignment"),
                 row.get("replay_error"), row.get("comparison_error"),
                 row.get("save_error"),
                 row.get("analysis_error")) for row in runs])
    if runs is not None:
        body += _section("Full batch action log", _raw(runs))
    return _wrap(body)


def _answer_html(value):
    """Render saved Markdown as inert text and formatting, not active HTML.

    Parameters
    ----------
    value: object
        Recorded value to inspect or format.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    if not isinstance(value, str):
        return "<p>Unavailable.</p>"
    try:
        import mistune
    except ImportError:
        return f'<div class="ks-text">{_e(value)}</div>'

    class SafeRenderer(mistune.HTMLRenderer):
        """Suppress navigable links and externally loaded images."""

        def link(self, text, url, title=None):
            """Keep the formatted label and display the escaped URL.

            Parameters
            ----------
            text: str
                Already rendered link label.
            url: str
                Recorded URL; displayed as text, never fetched.
            title: str or None
                Optional link title.

            Returns
            -------
            str
                Rendered or formatted text.

            """
            return f"{text} <code>{escape(url)}</code>"

        def image(self, text, url, title=None):
            """Represent images as inert descriptions without loading.

            Parameters
            ----------
            text: str
                Image description to display as text.
            url: str
                Recorded URL; displayed as text, never fetched.
            title: str or None
                Optional image title.

            Returns
            -------
            str
                Rendered or formatted text.

            """
            return f"[Image: {escape(text)} — {escape(url)}]"

    markdown = mistune.create_markdown(renderer=SafeRenderer(escape=True),
                                       plugins=["table", "strikethrough"])
    return markdown(value)


def _diff(left, right):
    """Render a literal line diff of saved answers, not a semantic verdict.

    Parameters
    ----------
    left: str
        Original-side text or rendered content.
    right: str
        Replay-side text or rendered content.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    if not isinstance(left, str) or not isinstance(right, str):
        return "<p>Both texts are required.</p>"
    if left == right:
        return "<p>Saved answer strings are identical.</p>"
    lines = []
    for line in unified_diff(left.splitlines(keepends=True),
                             right.splitlines(keepends=True),
                             fromfile="Original", tofile="Replay"):
        tone = ("add" if line.startswith("+") else "remove" if
                line.startswith("-") else None)
        text = _e(line.rstrip("\n")) + "\n"
        lines.append(f'<span class="ks-diff-{tone}">{text}</span>'
                     if tone else text)
    if not lines:
        return "<p>Only line-ending or whitespace representation differs.</p>"
    return "<pre>" + "".join(lines) + "</pre>"


def _paired_fields(q):
    """Show user-facing routing, completion, and saved-turn token fields.

    Parameters
    ----------
    q: dict
        Question-pair observations from an aligned operation.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    rows = []
    for field, label in (("intent", "Intent"),
                         ("retrieval_mode", "Retrieval mode"),
                         ("tool", "Tool summary"),
                         ("completion", "Completion"),
                         ("finish_reason", "Finish reason")):
        rows.append((label, *(_at(q, field, side) for side in SIDES)))
    for key in ("input_tokens", "output_tokens"):
        rows.append((key, *(_at(q, "usage", side, key) for side in SIDES)))
    rows.append(("Saved turn UUID", q.get("original_turn_uuid"),
                 q.get("replayed_turn_uuid")))
    rows.append(("Trace", q.get("original_trace"), q.get("replayed_trace")))
    return _table(("Observation", "Original", "Replay"), rows)


def _passages(items):
    """Render ranked passage identities without pretending hashes are text.

    Parameters
    ----------
    items: list
        Recorded items in display order.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    if items is None:
        return "<p>Unavailable.</p>"
    if not items:
        return "<p>No recorded passages.</p>"
    return _table(("Rank", "Source", "Start", "End", "Section / symbol / cell",
                   "Text SHA-256"), [
        (i, x.get("path"), x.get("start_line"), x.get("end_line"),
         " / ".join(_format(x[k]) for k in ("section", "symbol", "cell")
                    if x.get(k) is not None), x.get("text_sha256"))
        for i, x in enumerate(items, 1)])


def _evidence(q):
    """Display evidence, citation overlap, and unresolved citation numbers.

    Parameters
    ----------
    q: dict
        Question-pair observations from an aligned operation.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    evidence = q.get("evidence")
    if not isinstance(evidence, dict):
        return "<p>Evidence comparison unavailable.</p>"
    citations = _dict(evidence.get("citations"))
    body = "<h4>Retrieved evidence identities</h4>" + _pair(
        *[_passages(evidence.get(side)) for side in SIDES])
    body += "<h4>Cited passage identities</h4>" + _pair(
        *[_passages(citations.get(side)) for side in SIDES])
    body += _table(("Observation", "Original", "Replay"), [
        ("Unresolved citation numbers", citations.get("original_unresolved"),
         citations.get("replayed_unresolved"))])
    body += ('<p class="ks-caption">Citation resolution is syntactic, not '
             'verification of scientific support. Empty evidence unions '
             'have unavailable overlap.</p>')
    body += _section("Evidence and citation overlap details", _raw({
        "evidence": evidence.get("overlap"),
        "citations": citations.get("overlap")}))
    return body


def _rankings(op):
    """Display ranked identities, overlap and per-channel coverage.

    Parameters
    ----------
    op: dict
        One aligned operation from the comparison.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    body = ""
    for row in _list(op.get("rankings")):
        title = (f'{row.get("event_type")} · '
                 f'occurrence {row.get("occurrence")}')
        panels = []
        for side in SIDES:
            items = row.get(side)
            if items is None:
                panels.append("<p>Unavailable.</p>")
            elif not items:
                panels.append("<p>No ranked entries.</p>")
            else:
                panels.append(_table(("Rank", "Identity"),
                                     enumerate(items, 1)))
        body += _section(title, _pair(*panels) + _section(
            "Overlap and issues", _raw({"overlap": row.get("overlap"),
                                        "issues": row.get("issues")})))
    return body or "<p>No aligned ranking records.</p>"


def _stages(op):
    """Show all stage counts, statuses and recorded durations.

    Parameters
    ----------
    op: dict
        One aligned operation from the comparison.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    return _table(("Component", "Event", "Original count", "Original time",
                   "Replay count", "Replay time", "Original statuses",
                   "Replay statuses"), [
        (x.get("component"), x.get("event_type"),
         _at(x, "original", "count"),
         _seconds(_at(x, "original", "total_duration_ms")),
         _at(x, "replayed", "count"),
         _seconds(_at(x, "replayed", "total_duration_ms")),
         _at(x, "original", "statuses"), _at(x, "replayed", "statuses"))
        for x in _list(op.get("stages"))])


def _trace_path(root, relative):
    """Resolve only nonlinked local JSON files under logs/traces.

    Parameters
    ----------
    root: pathlib.Path
        Explicit local repository or workspace directory.
    relative: str
        Workspace-relative recorded trace or artifact path.

    Returns
    -------
    pathlib.Path
        Validated local trace path.

    """
    if root is None or not isinstance(relative, str):
        raise ValueError("Local workspace or trace path unavailable.")
    root = Path(root)
    path = Path(relative)
    if (root.is_symlink() or not root.is_dir() or path.is_absolute()
            or len(path.parts) < 3 or path.parts[:2] != ("logs", "traces")
            or ".." in path.parts or path.suffix.lower() != ".json"):
        raise ValueError("Unsafe or unavailable local trace path.")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Linked trace path rejected.")
    if not current.is_file():
        raise ValueError("Local trace file unavailable.")
    return current


def _trace_events(data):
    """Yield stored event nodes from the hierarchical trace, not content.

    Parameters
    ----------
    data: dict
        Decoded local hierarchical trace JSON.

    """
    def visit(value):
        """Traverse stored trace containers without interpreting event text.

        Parameters
        ----------
        value: object
            Nested trace node or container to traverse.

        """
        if isinstance(value, dict):
            event = value.get("event")
            if isinstance(event, dict) and type(event.get("sequence")) is int:
                yield value
            for key, child in value.items():
                if key not in ("event", "completion", "details",
                               "exact_content", "timing"):
                    yield from visit(child)
        elif isinstance(value, list):
            for child in value:
                yield from visit(child)

    if (data.get("trace_format_version") != 1
            or data.get("sequence_scope") != "project"):
        raise ValueError("Unsupported local trace format.")
    yield from visit(data.get("sessions"))


def _trace_index(root, relative, cache):
    """Index the reported conversation trace and optional project trace.

    Parameters
    ----------
    root: pathlib.Path
        Explicit local repository or workspace directory.
    relative: str
        Workspace-relative recorded trace or artifact path.
    cache: dict
        Per-render local trace index cache.

    Returns
    -------
    dict
        Local trace nodes indexed by sequence.

    """
    relative = relative if isinstance(relative, str) else None
    key = (str(root), relative)
    if key in cache:
        return cache[key]
    index = {}
    paths = [relative, "logs/traces/project.json"]
    for path in dict.fromkeys(paths):
        try:
            file = _trace_path(root, path)
            data = json.loads(file.read_text(encoding="utf-8"))
            for node in _trace_events(data):
                meta = node["event"]
                index.setdefault(meta["sequence"], []).append(node)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    cache[key] = index
    return index


def _exact_model_text(row, field, index):
    """Recover only a unique model artifact matching the report digest.

    Parameters
    ----------
    row: dict
        Current record or batch action row.
    field: str
        Recorded field to inspect.
    index: dict
        Trace nodes keyed by project event sequence.

    Returns
    -------
    tuple
        Verified text or None, and an explanatory observation.

    """
    key = "input" if field == "prompt" else "output"
    artifact = "02_rendered_prompt" if key == "input" else (
        "01_generated_answer")
    sequence = _at(row, key + "_sequence")
    digest = _at(row, field + "_sha256")
    if type(sequence) is not int or not isinstance(digest, str):
        return None, "No exact artifact recorded in the report."
    events = index.get(sequence, [])
    if len(events) != 1:
        return None, "Local trace event missing or ambiguous."
    node = events[0]
    if _at(node, "event", "event_type") != "llm.call." + key:
        return None, "Local event type does not match the report."
    value = _at(node, "exact_content", artifact, "value")
    if not isinstance(value, str):
        return None, "Inline exact text unavailable."
    if sha256(value.encode("utf-8")).hexdigest() != digest:
        return None, "Local trace content differs from the report hash."
    return value, "Verified against report SHA-256."


def _model_calls(op, roots, cache, include_text):
    """Display aligned call metadata and optional verified local texts.

    Parameters
    ----------
    op: dict
        One aligned operation from the comparison.
    roots: dict
        Explicit workspace paths keyed by comparison side.
    cache: dict
        Per-render local trace index cache.
    include_text: bool
        Include verified exact local model texts.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    q = _dict(op.get("question"))
    body = ""
    for row in _list(op.get("model_calls")):
        title = f'{row.get("stage")} · occurrence {row.get("occurrence")}'
        pairs = []
        for field in ("model_id", "backend", "input_sequence",
                      "output_sequence", "input_tokens", "output_tokens",
                      "finish_reason", "reply_duration_ms",
                      "generation_duration_ms", "output_available"):
            pairs.append((field, *(_at(row, side, field) for side in SIDES)))
        panel = _table(("Call field", "Original", "Replay"), pairs)
        panel += _section("Prompt and response similarity / hashes", _raw({
            "similarities": row.get("text"),
            **{side: {key: _at(row, side, key) for key in
                       ("prompt_sha256", "answer_sha256", "model_source")}
               for side in SIDES}}))
        if include_text:
            for field in ("prompt", "answer"):
                panes = []
                for side in SIDES:
                    index = _trace_index(roots.get(side),
                                         q.get(side + "_trace"), cache)
                    value, note = _exact_model_text(row.get(side), field,
                                                    index)
                    panes.append(f'<p class="ks-caption">{_e(note)}</p>'
                                 + (f'<pre>{_e(value)}</pre>'
                                    if value is not None else ""))
                panel += _section("Exact " + field, _pair(*panes))
        body += _section(title, panel)
    return body or "<p>No aligned model-call records.</p>"


def _question_panel(op, ordinal, roots, cache, include_text, expanded):
    """Build one expandable case with answers and diagnostic layers.

    Parameters
    ----------
    op: dict
        One aligned operation from the comparison.
    ordinal: int
        One-based aligned question index.
    roots: dict
        Explicit workspace paths keyed by comparison side.
    cache: dict
        Per-render local trace index cache.
    include_text: bool
        Include verified exact local model texts.
    expanded: bool
        Whether the collapsible panel starts open.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    q = op["question"]
    a = _dict(q.get("answer"))
    metrics = _question_metrics(op)
    title = (f'Q{ordinal} · {q.get("question")} · '
             f'F1 {_metric_value(metrics[0])} · '
             + ("exact" if a.get("exact_equal") is True else
                "different" if a.get("exact_equal") is False else
                "equality unavailable"))
    body = '<div class="ks-cards">' + "".join(
        _card(label, _metric_value(value)) for (_, label), value in
        zip(METRICS, metrics, strict=True)) + "</div>"
    body += _table(("Sequence", "Original", "Replay"), [
        ("Command start", op.get("original_sequence"),
         op.get("replayed_sequence")),
        ("Duration", *(_seconds(_at(op, "duration", side, "ms"))
                       for side in SIDES)),
        ("Answer characters", a.get("original_characters"),
         a.get("replayed_characters")),
        ("Exact saved answer equality", a.get("exact_equal"), ""),
        ("Normalized visible equality", a.get("normalized_equal"), "")])
    body += "<h3>Complete saved answers</h3>" + _pair(
        *[_answer_html(a.get(side)) for side in SIDES])
    body += _section(
        "Exact saved text (including any recorded reasoning)",
        _pair(*[f'<pre>{_e(a.get(side))}</pre>' for side in SIDES]))
    body += _section("Literal line-by-line difference", _diff(
        a.get("original"), a.get("replayed")))
    body += _section("Routing, tool, token use and saved turn identity",
                     _paired_fields(q), expanded=True)
    if "tool_comparison" in q:
        body += _section("Functional tool result and completion", _raw(
            q["tool_comparison"]), expanded=True)
    if "context" in op:
        body += _section("Mapped history and memory selections", _raw(
            op["context"]))
    body += _section("Evidence and citations", _evidence(q))
    body += _section("Ranked retrieval comparisons", _rankings(op))
    body += _section("Execution stages (nested; not additive)", _stages(op))
    body += _section("Individual model calls", _model_calls(
        op, roots, cache, include_text))
    for key, label in (("observations", "History, routing and retrieval"),
                       ("settings_provenance", "Settings provenance")):
        body += _section(label, _pair(*[_raw(_at(op, key, s)) for s in SIDES]))
    body += _section("Recorded interactions", _pair(*[_raw(
        _at(q, "interactions", s)) for s in SIDES]))
    body += _section("Settings differences and diagnostic issues", _raw({
        "settings_differences": op.get("settings_differences"),
        "operation_issues": op.get("issues"),
        "question_issues": q.get("issues")}))
    return _section(title, body, expanded=expanded)


def question_explorer(data: dict, *, question: int | None = None,
                      differences_only: bool = False,
                      original: str | Path | None = None,
                      replayed: str | Path | None = None,
                      include_model_text: bool = False) -> str:
    """Explore matched saved answers and expandable execution diagnostics.

    Parameters
    ----------
    data : dict
        Native version-1 comparison report.
    question : int or None, optional
        One-based aligned question index; None shows all pairs.
    differences_only : bool, optional
        Include pairs with differing answers, routing, retrieval, tools,
        recorded settings, or reported issues. Not a quality filter.
    original: str, pathlib.Path or None, optional
        Explicit local original path, never a path taken from the report.
    replayed: str, pathlib.Path or None, optional
        Explicit local replay path, never a path taken from the report.
    include_model_text : bool, optional
        Inspect inline local prompt/response artifacts only when their
        sequences, event types and SHA-256 digests match the report.

    Returns
    -------
    str
        Standalone HTML with collapsible cases; no widgets or model calls.

    """
    questions = _questions(data)
    if question is not None and (type(question) is not int or
                                 not 1 <= question <= len(questions)):
        raise ValueError(
            "Question must be a one-based matched question index.")
    roots = {"original": original, "replayed": replayed}
    cache = {}
    body = ('<h2>Answer explorer</h2><p class="ks-caption">Expand any '
            'question to compare complete saved answers and diagnostics. '
            'Different answers are not automatically regressions. '
            'These are aligned pairs, not all historical turns.</p>')
    shown = 0
    for i, op in enumerate(questions, 1):
        if question is not None and question != i:
            continue
        q = op["question"]
        changed = (_at(q, "answer", "exact_equal") is not True or
                   any(_at(q, key, "original") != _at(q, key, "replayed")
                       for key in ("intent", "retrieval_mode")) or
                   (_at(q, "tool_comparison", "functional_equal") is False
                    if "tool_comparison" in q else
                    _at(q, "tool", "original") != _at(q, "tool", "replayed"))
                   or any(outcome.get("status") == "incomplete" for outcome
                          in _dict(_at(q, "tool_comparison", "outcomes")
                                   ).values()) or
                   bool(op.get("settings_differences") or op.get("issues")
                        or q.get("issues")))
        if differences_only and not changed:
            continue
        body += _question_panel(op, i, roots, cache, include_model_text,
                                expanded=question is not None)
        shown += 1
    body += f"<p>Showing {shown} of {len(questions)} matched questions.</p>"
    if question is None:
        for side in SIDES:
            paired = {op["question"].get(side + "_turn_uuid")
                      for op in questions}
            unmatched = [q for q in _list(_at(data, side, "saved_questions"))
                         if q.get("turn_uuid") not in paired]
            if unmatched:
                panel = ""
                for q in unmatched:
                    trust = ("validated" if q.get("validated") else
                             "unverified recording")
                    title = (f'{q.get("sequence")} · {q.get("question")}'
                             f" · {trust}")
                    panel += _section(
                        title,
                        _answer_html(q.get("answer")) + _raw({
                            "intent": q.get("intent"),
                            "tool_outcome": q.get("tool_outcome"),
                            "validated": q.get("validated")}), expanded=True)
                body += _section(f"{side.title()} — independently saved "
                                 "questions (not paired)", panel,
                                 expanded=True)
    if not questions:
        body += _diagnostics(data)
    return _wrap(body)


def operation_explorer(data: dict) -> str:
    """Render all aligned operation records, including non-question commands.

    Parameters
    ----------
    data : dict
        Native version-1 comparison snapshot.

    Returns
    -------
    str
        Collapsible operations with durations, settings, manifests, stage
        diagnostics and observations. Question answers are reviewed in the
        separate answer explorer.

    """
    body = "<h2>Operation explorer</h2>"
    for op in data["operations"]:
        label = (f'{op.get("kind")} · '
                 f'original #{op.get("original_sequence")} '
                 f'→ replay #{op.get("replayed_sequence")}')
        fields = {key: value for key, value in op.items()
                  if key not in ("question", "stages")}
        body += _section(label, _stages(op) + _section(
            "Complete operation metadata", _raw(fields)))
    if not data["operations"]:
        body += "<p>No aligned operations.</p>"
    return _wrap(body)


def _md(value):
    """Escape data used in Markdown cells, including raw HTML and pipes.

    Parameters
    ----------
    value: object
        Recorded value to inspect or format.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    value = _format(value).replace("\n", " ").replace("\r", " ")
    value = escape(value, quote=False).replace("\\", "\\\\")
    for char in "|`*_[]!~":
        value = value.replace(char, "\\" + char)
    return value


def _md_table(headers, rows):
    """Build a Markdown table with escaped report values.

    Parameters
    ----------
    headers: sequence[str]
        Column headings in display order.
    rows: iterable
        Rows of values, one sequence per table row.

    Returns
    -------
    list[str]
        Markdown table lines.

    """
    line = lambda row: "| " + " | ".join(_md(x) for x in row) + " |"
    return [line(headers), line(["---"] * len(headers)),
            *(line(row) for row in rows), ""]


def _md_bar(value, maximum=1, width=12):
    """Produce a portable text bar with a labeled scale or missing value.

    Parameters
    ----------
    value: object
        Recorded value to inspect or format.
    maximum: float
        Positive reference scale for the bar.
    width: int
        Number of characters in the portable bar.

    Returns
    -------
    str
        Rendered or formatted text.

    """
    value = _number(value)
    if value is None or maximum <= 0 or value < 0:
        return "n/a" if value is None else _format(value)
    filled = round(min(value / maximum, 1) * width)
    if value > 0:
        filled = max(1, filled)
    return "█" * filled + "░" * (width - filled)


def analysis_markdown(data: dict, name: str = "Workspace") -> str:
    """Create a portable visual summary of a saved comparison snapshot.

    Parameters
    ----------
    data : dict
        Native version-1 comparison snapshot.
    name : str, optional
        Experiment stem used in the heading.

    Returns
    -------
    str
        Markdown with portable text bars, coverage, per-question routing,
        paired durations and diagnostic summaries. The native reports
        remain the authoritative detailed exports.

    """
    summary, align = data["summary"], data["alignment"]
    questions = _questions(data)
    lines = [f"# {_md(name)} — visual analysis", "",
             "Saved comparison snapshot; not a correctness verdict.", "",
             f'**Alignment:** {_md(align.get("status"))}  ',
             f'**Matched operations:** {_md(align.get("matched_operations"))}'
             f' · **Questions:** {_md(summary.get("questions"))}  ',
             f'**Exact answers:** {_md(summary.get("exact_answers"))}', ""]
    if align.get("reason") is not None:
        lines.extend([_md(align["reason"]), ""])
    if data.get("inspection", data.get("batch_inspection")):
        note = data.get("inspection", data.get("batch_inspection"))
        lines += [_md(note["message"]), ""]
    receipt = data.get("reference_publication", {})
    if receipt:
        lines += ["**Reference publication:** " +
                  _md(receipt.get("status")), ""]
        if receipt.get("message"):
            lines += [_md(receipt["message"]), ""]
        lines += [f"- {_md(hint)}" for hint in receipt.get("coverage", [])]
        lines += [""]
    lines += ["## Validity and coverage", ""] + _md_table(
        ("Workspace", "Valid", "Events", "Commands", "Issues"), [
            (side.title(), data[side].get("valid"),
             data[side].get("event_count"),
             len(_list(data[side].get("operations"))),
             len(_list(data[side].get("validation_issues"))))
            for side in SIDES])
    lines += ["## Agreement", ""] + _md_table(
        ("Metric", "Mean", "Coverage", "0–1 scale"), [
            (label, _metric_value(_at(summary, key, "mean")),
             f'{_format(_at(summary, key, "available"))}/'
             f'{_format(_at(summary, key, "total"))}',
             _md_bar(_at(summary, key, "mean"))) for key, label in METRICS])
    lines += ["Missing values are not zeros. Negative cosine values are "
              "shown numerically without a nonnegative bar.", "",
              "## Questions", ""]
    if not questions:
        lines += ["No aligned committed question pairs.", ""]
    else:
        lines += _md_table(("Question", "Exact", "Token F1", "Overlap bar",
                            "Cosine", "Evidence", "Citation",
                            "Intent O / R"), [
            (f'Q{i} · {op["question"].get("question")}',
             _at(op, "question", "answer", "exact_equal"),
             _metric_value(_question_metrics(op)[0]),
             _md_bar(_question_metrics(op)[0]),
             *(_metric_value(x) for x in _question_metrics(op)[1:]),
             " / ".join(_format(_at(op, "question", "intent", side))
                        for side in SIDES))
            for i, op in enumerate(questions, 1)])
    lines += ["## Recorded behavior agreement", ""] + _md_table(
        ("Observation", "Equal", "Available", "Question pairs"),
        _agreement(data))
    lines += ["Equal null tool summaries mean equal recorded absences, "
              "not proof that no tool ran.", ""]
    lines += ["## Performance", "",
              "All operation bars share the same duration scale (rounded; "
              "positive values get at least one block). Nested "
              "stage times must not be added to these totals.", ""]
    maximum = max([_number(_at(op, "duration", side, "ms")) or 0
                   for op in data["operations"] for side in SIDES] + [1])
    lines += _md_table(("Operation", "Original", "Replay", "Original bar",
                        "Replay bar"), [
        (f'{op.get("original_sequence")} · {op.get("kind")}',
         *(_seconds(_at(op, "duration", side, "ms")) for side in SIDES),
         *(_md_bar(_at(op, "duration", side, "ms"), maximum)
           for side in SIDES)) for op in data["operations"]])
    duration = _dict(summary.get("paired_command_duration_ms"))
    lines += [f'**Paired measured total:** '
              f'{_seconds(duration.get("original"))} / '
              f'{_seconds(duration.get("replayed"))} · '
              f'{_format(duration.get("available"))}/'
              f'{_format(duration.get("total"))} operations.', ""]
    lines += _md_table(("Token boundary", "Original", "Replay"), [
        (f"{key}.{field}", *(_at(data, side, key, field, "recorded_total")
                             for side in SIDES)) for key in
        ("usage", "traced_chat_usage") for field in
        ("input_tokens", "output_tokens")])
    lines += ["Saved-turn and traced-call tokens overlap; do not sum them. "
              "See the native report for token coverage.", "",
              "## Configuration and final state", ""]
    diffs = [(op.get("original_sequence"), difference)
             for op in data["operations"] for difference in
             _list(op.get("settings_differences"))]
    lines += [f"Recorded settings differences: **{len(diffs)}** "
              "across matched operations.", ""]
    if diffs:
        lines += _md_table(("Sequence", "Difference"), diffs)
    lines += _md_table(("Directory", "Added", "Removed", "Changed",
                        "Unchanged"), [
        (directory, *(len(_at(entry, "changes", key)) if isinstance(
            _at(entry, "changes", key), list) else None for key in
            ("added", "removed", "changed", "unchanged")))
        for directory, entry in data["files"].items()])
    lines += _md_table(("Final state", "Equal", "Available", "Mapped"),
                       _mapped_state(data))
    events = event_inventory(data)
    lines += ["## Event inventory", ""]
    lines += _md_table(("Side", "Reported events", "Counted events"), [
        (side, events["expected"][side],
         sum(events["counts"][side].values())) for side in SIDES])
    if not events["complete"]:
        lines += ["Complete inventory unavailable; partial counts are not "
                  "interpreted as differences.", ""]
    elif events["differences"]:
        lines += _md_table(("Event type", "Original", "Replay", "Delta"), [
            (r["event_type"], r["original"], r["replayed"], r["delta"])
            for r in events["differences"]])
    else:
        lines += ["Every recorded event-type count agrees.", ""]
    lines += ["## Diagnostics", ""]
    for side in SIDES:
        lines += [f"### {side.title()}", ""]
        for key in ("validation_issues", "validation_warnings"):
            for issue in _list(data[side].get(key)):
                lines.append(f'- {_md(key)} · {_md(issue.get("code"))} · '
                             f'{_md(issue.get("sequence"))}: '
                             f'{_md(issue.get("message"))}')
        lines.append("")
    for issue in data["report_issues"]:
        lines.append(f'- Report: {_md(issue)}')
    lines += ["", "## Detailed inspection", "",
              f"Open `{_md(name)}_answers.ipynb` in `workspaces/` "
              "to compare full answers, "
              "evidence identities, citations, model calls and stages.", "",
              "The native `_comparison.json` and `_comparison.md` files "
              "contain the complete underlying comparison measurements. "
              "All reports may contain private information.", ""]
    return "\n".join(lines)


def save_analysis(report_path: str | Path, *, overwrite: bool = False
                  ) -> Path:
    """Write a visual Markdown companion to an existing native JSON report.

    Parameters
    ----------
    report_path : str or pathlib.Path
        Saved native ``*_comparison.json`` report, not an in-memory result.
    overwrite : bool, optional
        Replace an existing regular companion only when explicitly enabled.
        The native Markdown and JSON files are never modified.

    Returns
    -------
    pathlib.Path
        Path of the ``*_analysis.md`` companion.

    """
    report_path = Path(report_path)
    if not report_path.name.endswith(SUFFIX):
        raise ValueError("Expected a *_comparison.json report.")
    if report_path.parent.is_symlink():
        raise ValueError("Linked report directory rejected.")
    data = load_report(report_path)
    stem = report_path.name.removesuffix(SUFFIX)
    path = report_path.with_name(stem + "_analysis.md")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"Nonregular analysis destination: {path}")
    text = analysis_markdown(data, stem)
    if not overwrite:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        temp_path = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8",
                                    dir=path.parent, delete=False) as stream:
                temp_path = Path(stream.name)
                stream.write(text)
            temp_path.replace(path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    return path
