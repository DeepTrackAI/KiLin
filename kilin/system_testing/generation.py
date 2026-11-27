"""Generate closed system-testing references from trusted experiment recipes.

Key Features
------------
- Drive ordinary recorded engine commands and explicit local interactions.
- Validate command history, protect destinations and publish references.
- Report question-specific coverage hints and saved output observations.

Module Structure
----------------
- ``Recorder``: synchronous command and interaction adapter.
- ``generate_reference``: staging, validation and publication lifecycle.
- ``check_output``, ``reference_overview``: descriptive recipe diagnostics.

Examples
--------
>>> from kilin.system_testing.generation import generate_reference
>>> # Pass a trusted recipe and explicit creation settings.

"""

from __future__ import annotations

from html import escape
from pathlib import Path

from .experiment import Experiment
from .reference import publish_receipt, receipt_path

__all__ = ["Recorder", "generate_reference", "check_output",
           "reference_overview"]


class Recorder:
    """Drive the real synchronous engine and verify command outcomes."""

    def __init__(self, recording, relocated, project_type, settings):
        """Configure a traced engine without loading a model.

        Parameters
        ----------
        recording: pathlib.Path
            Fresh staging workspace path.
        relocated: pathlib.Path
            Staging sibling used by an explicit rename recipe.
        project_type: str
            Registered project type identifier.
        settings: dict
            Explicit project creation settings.

        """
        from ..gui.engine import ChatEngine

        self.project_type = project_type
        self.settings = dict(settings)
        self.relocated = relocated
        self.events = []
        self.results = []
        self.coverage = []
        self.prompt = None
        self.plans = {}
        self.options = {}
        self.responses = []
        self.failure = None
        self.request_id = 0
        self.directory = recording
        self.engine = ChatEngine(
            recording.parent, self.observe, trace_enabled=True,
            read_command=self.read_command,
        )

    def observe(self, event):
        """Collect acknowledgements and live interaction payloads.

        Parameters
        ----------
        event: dict
            Synchronous engine notification.

        """
        kind = event["event"]
        if kind == "answer_delta":
            return
        self.events.append(event)
        if kind == "status":
            print(event.get("message", ""))
        if kind == "answer_phase":
            for name in ("plan", "choice"):
                if name in event:
                    if self.prompt is not None:
                        self.failure = "Overlapping interaction prompts."
                        raise RuntimeError(self.failure)
                    self.prompt = (name, event[name])

    def read_command(self):
        """Answer only a predeclared local plan or unique option label.

        Returns
        -------
        dict
            Explicit approved engine response.

        """
        try:
            if self.prompt is None:
                raise RuntimeError("Unexpected interaction without a prompt.")
            kind, prompt = self.prompt
            self.prompt = None
            if kind == "plan":
                if prompt.get("operation", "model") != "model":
                    raise RuntimeError(
                        "Non-model plan requires manual review."
                    )
                tool = prompt.get("tool")
                choice = self.plans.get(tool)
                if choice not in ("quick", "continue"):
                    raise RuntimeError(f"Unapproved plan: {prompt}")
                self.responses.append(("plan", tool, choice))
            else:
                question = prompt.get("question")
                prefix = self.options.get(question)
                labels = prompt.get("options", [])
                matches = [i for i, label in enumerate(labels)
                           if prefix and label.startswith(prefix)]
                if len(matches) != 1:
                    raise RuntimeError(f"Unapproved choice: {prompt}")
                index = matches[0]
                choice = f"option:{index}"
                self.responses.append(("choice", question, labels[index]))
            print("Scripted response:", self.responses[-1])
            return {"action": "plan_choice", "choice": choice}
        except Exception as error:
            self.failure = str(error)
            raise

    def call(self, method, *args, expected=None, **kwargs):
        """Reject failures emitted by engine methods, not just exceptions.

        Parameters
        ----------
        method: callable
            Ordinary synchronous engine command.
        *args: object
            Positional arguments forwarded to the command.
        expected: str or None
            Required acknowledgement event type, if any.
        **kwargs: object
            Keyword arguments forwarded to the command.

        """
        self.events.clear()
        self.prompt = None
        self.failure = None
        method(*args, **kwargs)
        bad = [event for event in self.events if (
            event["event"] in ("error", "workspace_conflict",
                               "backend_failed", "answer_cancelled",
                               "memory_sync_failed")
            or event.get("error")
        )]
        if self.failure or bad or self.prompt is not None:
            raise RuntimeError(self.failure or str(bad or self.prompt))
        if expected and not any(e["event"] == expected for e in self.events):
            raise RuntimeError(f"Missing acknowledgement: {expected}")

    def active_id(self):
        """Return the current live conversation identity.

        Returns
        -------
        str
            Current conversation identity.

        """
        return self.engine.active_conversation.conversation_id

    def open(self, sources=(), *, selected_files=None):
        """Create a traced project, without GUI widgets.

        Parameters
        ----------
        sources: sequence[pathlib.Path]
            Local sources to copy into the new workspace.
        selected_files: list[str] or None
            Explicit file selection within imported source paths.

        """
        paths = [str(path.resolve(strict=True)) for path in sources]
        self.call(self.engine.open_project, paths, self.project_type,
                  str(self.directory), selected_files=selected_files,
                  settings=self.settings, expected="opened")
        if self.engine.session.project.root != self.directory:
            raise RuntimeError("Opened an unexpected workspace.")
        profile = self.engine.session.project.profile
        if (profile.temperature != self.settings["temperature"]
                or profile.thinking != self.settings["thinking"]
                or profile.max_new_tokens !=
                self.settings["max_new_tokens"]):
            raise RuntimeError("Creation generation settings differ.")

    def ask(self, label, question, *, terms=(), source_names=(),
            intent=None, plans=None, options=None, require=(), **kwargs):
        """Save a question and collect nonjudgmental review hints.

        Parameters
        ----------
        label: str
            Human-readable label.
        question: str
            Exact user question to record.
        terms: sequence[str]
            Case-insensitive answer-content hints.
        source_names: sequence[str]
            Expected evidence basenames, used only as review hints.
        intent: str or None
            Optional expected intent, recorded as a review hint.
        plans: dict or None
            Approved local tool-plan choices keyed by tool name.
        options: dict or None
            Approved option-label prefixes keyed by prompt question.
        require: sequence[str]
            Interaction kinds expected as coverage observations.
        **kwargs: object
            Question-specific settings forwarded to the engine.

        Returns
        -------
        object
            Newly committed conversation turn.

        """
        self.plans = dict(plans or {})
        self.options = dict(options or {})
        self.responses.clear()
        before = len(self.engine.session.turns)
        generation = kwargs.get("generation_settings") or {}
        profile = self.engine.session.project.profile
        expected_temperature = generation.get(
            "temperature", profile.temperature
        )
        print(f"\n{label}: {question}")
        try:
            self.call(self.engine.ask, question, expected="answered", **kwargs)
        finally:
            self.plans.clear()
            self.options.clear()
        turns = self.engine.session.turns
        if len(turns) != before + 1 or turns[-1].question != question:
            raise RuntimeError("Expected one newly committed question.")
        turn = turns[-1]
        if turn.parameters.get("temperature") != expected_temperature:
            raise RuntimeError("Recorded temperature differs from request.")
        paths = sorted({Path(e.path).name for e in turn.evidence})
        visible = turn.answer.rsplit("</think>", 1)[-1].strip()
        hints = [f"Expected text: {term}" for term in terms
                 if term.casefold() not in visible.casefold()]
        hints += [f"Expected evidence file: {name}" for name in source_names
                  if name not in paths]
        actual = turn.parameters.get("intent")
        if intent is not None and actual != intent:
            hints.append(f"Expected intent {intent}; observed {actual}")
        for kind in require:
            if not any(response[0] == kind for response in self.responses):
                hints.append(f"Planned {kind} interaction not exercised")
        self.results.append({
            "id": label, "question": question, "answer": visible,
            "intent": actual, "evidence": paths, "hints": hints,
            "temperature": turn.parameters["temperature"],
            "thinking": turn.parameters.get("thinking"),
            "responses": list(self.responses),
        })
        return turn

    def rename(self):
        """Exercise the recorded project-folder relocation command."""
        active = self.active_id()
        before = [turn.conversation_turn_uuid
                  for turn in self.engine.session.turns]
        self.call(self.engine.rename_project, self.relocated.name,
                  expected="opened")
        self.directory = self.relocated
        if (self.engine.session.project.root != self.directory
                or self.active_id() != active
                or [turn.conversation_turn_uuid
                    for turn in self.engine.session.turns] != before):
            raise RuntimeError("Project relocation did not restore the chat.")

    def resume(self, conversation_id):
        """Reopen a closed workspace at an explicitly selected chat.

        Parameters
        ----------
        conversation_id: str
            Existing conversation to restore.

        """
        self.call(self.engine.resume_project, str(self.directory),
                  conversation_id, expected="opened")
        if self.active_id() != conversation_id:
            raise RuntimeError("Incorrect conversation restored.")


def generate_reference(root: Path, experiment: str, project_type: str,
                       settings: dict, expected_operations, record,
                       check_coverage):
    """Record and publish one closed reference using a trusted recipe.

    Preserve incomplete staging directories and refuse existing references,
    replays or publication receipts. Only the recipe may authorize local
    interactions. Coverage observations are not answer-quality verdicts.

    Parameters
    ----------
    root: pathlib.Path
        Experiment repository containing the workspaces directory.
    experiment: str
        Unique single-component stem, without the original suffix.
    project_type: str
        Registered project type for the creation command.
    settings: dict
        Explicit creation settings, including temperature, thinking and
        maximum new tokens.
    expected_operations: sequence[str]
        Complete chronological command kinds expected from the recipe.
    record: callable
        Trusted function accepting a Recorder and issuing engine commands.
    check_coverage: callable
        Trusted function accepting the recorder and validated report,
        appending descriptive coverage observations before publication.

    Returns
    -------
    tuple[Recorder, ValidationReport]
        Recorder and validated published reference report.

    """
    root = Path(root).resolve()
    if (not isinstance(experiment, str) or not experiment or
            experiment in (".", "..") or any(c in experiment for c in
                ("/", "\\", ":", "\x00")) or
            experiment.endswith(("_original", "_replay"))):
        raise ValueError("Expected a single experiment stem.")
    workspaces = root / "workspaces"
    original = workspaces / f"{experiment}_original"
    replay = workspaces / f"{experiment}_replay"
    staging = root / "_recordings"
    recording = staging / f"{experiment}_recording"
    relocated = staging / f"{experiment}_relocated_recording"
    # Run the complete script once. Preserve failed or unfinished attempts.
    legacy = (workspaces / recording.name, workspaces / relocated.name)
    protected = (original, replay, recording, relocated,
                 receipt_path(original), *legacy)
    if staging.is_symlink() or (staging.exists() and not staging.is_dir()):
        raise ValueError(f"Expected a real staging directory: {staging}")
    if workspaces.is_symlink() or not workspaces.is_dir():
        raise ValueError(f"Expected a real workspace folder: {workspaces}")
    for path in protected:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Move the existing workspace first: {path}")

    staging.mkdir(exist_ok=True)
    recorder = Recorder(recording, relocated, project_type, settings)
    try:
        record(recorder)
    finally:
        recorder.engine.close()

    report = Experiment(recorder.directory).validate()
    for issue in report.issues:
        print(issue.code, issue.sequence, issue.message)
    report.raise_for_errors()
    actual_operations = [op.kind for op in report.operations]
    if actual_operations != list(expected_operations):
        raise RuntimeError(
            f"Unexpected command history:\n{actual_operations}\n"
            f"Expected:\n{expected_operations}"
        )
    asks = [op for op in report.operations if op.kind == "ask"]
    if [op.request["question"] for op in asks] != [
            row["question"] for row in recorder.results]:
        raise RuntimeError(
            "Saved question history differs from scripted inputs.")
    check_coverage(recorder, report)
    for row in recorder.results:
        recorder.coverage.extend(
            f'REVIEW: {row["id"]}: {hint}' for hint in row["hints"]
        )

    # Publication is an ordinary move of an already closed recording, not a
    # simulated GUI rename. Never move a live workspace or overwrite a target.
    if original.exists() or original.is_symlink():
        raise FileExistsError(original)
    recorder.directory.rename(original)
    recorder.directory = original
    report = Experiment(original).validate()
    report.raise_for_errors()
    receipt = publish_receipt(original, report, recorder.coverage)
    print("\nPublished reference:", original)
    print("Publication receipt:", receipt)
    print("Events:", report.event_count, "Operations:", len(report.operations))
    print("Questions:", len(asks))
    for warning in report.warnings:
        print("Validation warning:", warning.code, warning.message)
    return recorder, report


def check_output(directory: Path, summary, *, label: str, tool: str,
                 field: str, expected: str, template: str | None = None,
                 require_valid: bool = False) -> list[str]:
    """Check the output attributed to one particular recorded question.

    A file from another question cannot fulfill the check. A failed or
    partial tool result is retained as a coverage observation; it does not
    invalidate an otherwise valid operation recording.

    Parameters
    ----------
    directory: pathlib.Path
        Explicit directory to inspect or write artifacts in.
    summary: dict or None
        Tool summary from the particular committed question.
    label: str
        Human-readable label.
    tool: str
        Expected tool name.
    field: str
        Tool-summary field containing the saved filename.
    expected: str
        Expected workspace-relative saved output filename.
    template: str or None
        Expected report template identifier, if specified.
    require_valid: bool
        Require an explicit successful validation result.

    Returns
    -------
    list[str]
        Coverage observations, with REVIEW prefixes for missing
        coverage.

    """
    if not isinstance(summary, dict) or summary.get("name") != tool:
        return [f"REVIEW: {label}: expected tool {tool} did not complete."]
    if summary.get("status") in ("incomplete", "partial", "error", "failed"):
        return [f"REVIEW: {label}: {tool} did not produce a complete result."]
    if template is not None and summary.get("template") != template:
        return [f"REVIEW: {label}: expected template {template} not used."]
    if require_valid and summary.get("valid") is not True:
        return [f"REVIEW: {label}: {template} report was not validated."]
    if summary.get(field) != expected:
        return [f"REVIEW: {label}: expected saved file {expected} absent."]

    def present(relative: str) -> bool:
        """Check a safe, nonempty regular file under the outputs folder.

        Parameters
        ----------
        relative: str
            Workspace-relative recorded trace or artifact path.

        Returns
        -------
        bool
            Whether the reported output is a nonempty regular file.

        """
        if not isinstance(relative, str):
            return False
        path = Path(relative)
        if (path.is_absolute() or len(path.parts) < 2 or
                path.parts[0] != "outputs" or ".." in path.parts or
                "\\" in relative):
            return False
        location = directory
        for part in path.parts:
            location = location / part
            if location.is_symlink():
                return False
        return location.is_file() and location.stat().st_size > 0

    if not present(expected):
        return [f"REVIEW: {label}: missing or empty file {expected}."]
    messages = [f"{label}: verified {tool} output {expected}."]
    pdf = summary.get("saved_pdf")
    if pdf is not None:
        if present(pdf):
            messages.append(f"{label}: verified reported PDF file {pdf}.")
        else:
            messages.append(f"REVIEW: {label}: reported PDF is missing.")
    elif require_valid:
        messages.append(f"{label}: no PDF output reported (Markdown checked).")
    return messages


def reference_overview(recorder: Recorder, report) -> str:
    """Render saved reference answers and coverage without calling a model.

    Parameters
    ----------
    recorder: Recorder
        Completed reference recording adapter.
    report: ValidationReport
        Validation result for the published reference.

    """
    # Read the actual recorded answers; these are review hints, not pass/fail
    # criteria. No model, judge, or replay is called by this presentation cell.
    rows = []
    for result in recorder.results:
        hints = result["hints"]
        state = "REVIEW" if hints else "No scripted hints missing"
        title = f'{result["id"]} — {result["question"]} — {state}'
        meta = (f'Intent: {result["intent"]}; '
                f'Temperature: {result["temperature"]}; '
                f'Configured thinking: {result["thinking"]}; '
                f'Evidence: {", ".join(result["evidence"]) or "none"}; '
                f'Choices: {result["responses"] or "none"}')
        items = "".join(f"<li>{escape(hint)}</li>" for hint in hints)
        rows.append(
            f'<details style="border:1px solid #aaa;padding:10px;'
            f'margin:8px 0"><summary><b>{escape(title)}</b></summary>'
            f'<p>{escape(meta)}</p><ul>{items}</ul>'
            '<div style="white-space:pre-wrap;overflow-wrap:anywhere">'
            f'{escape(result["answer"])}</div></details>'
        )
    summary = [f"{len(report.operations)} recorded operations; "
               f"{report.event_count} events; "
               f"{len(recorder.results)} questions."]
    summary.extend(recorder.coverage)
    return "<h3>Recorded reference — valid</h3>" + "".join(
        f"<p>{escape(line)}</p>" for line in summary
    ) + "".join(rows)
