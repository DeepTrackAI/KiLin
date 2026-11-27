# System testing: reference inspection, replay, and comparison

The `kilin.system_testing` package inspects a closed, recorded project as a
reference experiment. It reads the ordinary project and conversation traces,
including archived conversations, and reconstructs the recorded user commands
in project-wide sequence order. Inspection does not start the GUI, initialize
an inference backend, load models, or download resources. Replay is a separate,
explicit operation that runs the validated history in a new workspace.
Comparison reads both existing recordings independently; optional embedding
evaluation is explicitly enabled by the researcher.

See [trace architecture](trace_architecture.md) for the recording format and
[conversation ownership](conversation_ownership.md) for checkpoint and archive
semantics.

## Preparing an original reference workspace

- Create a **new** project with ordinary tracing enabled throughout. Use the
  recording version supported by the installed system-testing package.
- Choose the project type, models and persistent project settings at creation.
  Keep persistent settings fixed during the experiment, except for generation
  settings submitted with a question and recorded as part of that request.
- Add sources through normal project operations, including selected folder
  subsets and additive synchronization. Do not remove, replace, rename or
  manually edit managed sources. Retain all managed copies.
- Ask questions, respond to choices and plans, and use normal conversation
  creation, selection, branching, renaming, deletion, reset, feedback,
  exclusion, memory synchronization and source-card regeneration commands.
- Do not manually alter templates, generated outputs, profile files, traces,
  indexes, or saved feedback. Avoid cancelled or interrupted operations and
  concurrent writers. Close the project before inspection or copying it.
- Renaming the project folder through KíLín is supported. The operation is
  recorded as a close followed by reopening the same active conversation.
  The replay retains its designated workspace name.
- Keep the original workspace unchanged. Replay requires a separate sibling
  workspace path that does not yet exist.

GUI appearance and view-only actions are not replay inputs. Older recordings
or direct core-only sessions that do not contain the required command and
creation metadata are not complete reference experiments. No migration of traces or
inference of missing commands is attempted. A narrow exception for earlier
recordings is an already-recorded internal reopening that can be identified
as a successful sibling-directory rename from independent path and lifecycle
metadata. Such identification is reported as a validation warning.

## Inspecting a recording

```python
from pathlib import Path

from kilin.system_testing import Experiment

experiment = Experiment(Path("workspaces/010_original"))
report = experiment.validate()
print(report.valid, report.event_count)
for issue in report.issues:
    print(issue.code, issue.path, issue.sequence, issue.message)
for warning in report.warnings:
    print(warning.code, warning.message)

if report.valid:
    for operation in experiment.timeline():
        print(operation.start_sequence, operation.kind, operation.status)
```

`validate()` returns a `ValidationReport`. `issues` contains blocking errors;
`warnings` contains environmental differences and inspection limitations.
`events` preserves every successfully decoded original event in allocation
order, including logger lifecycle and other diagnostics outside command
intervals. `operations` is the candidate command timeline, even for an invalid
reference. Use `timeline()` to obtain a validated timeline: it raises
`InvalidRecordingError` with the full report if errors exist. Both methods
read fresh data; the object does not cache or repair an earlier result.

`TimelineOperation` exposes the stable `kind`, `start_sequence`,
`end_sequence`, recorded `request`, `status`, and associated `events`.
`operation.matching("tool.choice.decided")`, for example, selects actual
recorded decisions. Each `TraceEvent` retains the relative file and hierarchy
location, original metadata, details, and separately retained exact content.
An operation interval contains intermediate diagnostics, not extra commands.
Generated titles remain diagnostic observations rather than user requests.

## Validation contract

Validation checks the current trace format and recording marker, counter
health and contiguous sequences, unique event identities, paired spans,
logger lifetimes, and unambiguous chronological command boundaries. A
missing counter, malformed trace, missing archive or interrupted command is
an error, not permission to reconstruct the missing history. The one
supported detached-writer case is an active conversation archived by a
recorded successful deletion. A project rename is validated using matching
request/completion markers around a successful close and internal reopening,
corroborated by recorded directory paths and the restored conversation.
For older traces without either marker, only independently corroborated
sibling moves are accepted. Incomplete markers, same-directory internal
reopens, mismatched active conversations, failed renames, and persistent
configuration mutations remain blocking issues. Recorded absolute paths
are compared lexically in their original platform's path syntax, never
opened on the inspection machine.

The source checks use recorded copy outcomes and historical indexing
manifests. The final managed source inventory and content hashes must agree
with the additive history. Original external paths are provenance, not
locations to reopen. Recorded folder selections and collision-adjusted
managed paths remain available in the timeline. The original source tree
need not be accessible on the inspection machine.

Conversation identities, branch prefixes, saved question/checkpoint
identities and final saved content hashes are checked against the operation
history, including archived state. Question settings and interactive proposals
and responses are correlated with their core turns. A failed outer request
with a committed checkpoint has the distinct status `committed_with_error`;
it remains a blocking reference issue but its durable effect is not erased
from the candidate history. Feedback mutations are checked against their
recorded revisions and final authoritative state.

Recorded explicit settings are interpreted using the original displayed
values or submitted request, not the current package defaults. Saved profile
settings and behavior revisions are checked against historical configuration
records. Python, platform and installed-package differences are reported as
warnings; no model availability or inference-quality check is performed.

## Replaying a reference

Keep the experiments in a separate repository (for example,
`KiLinSystemTests`), with original and replay workspaces at the same level so
both can be opened in the ordinary GUI. Notebooks and the comparison
reports also belong in that repository. The replay implementation itself
belongs to KíLín; it does not require an experiment repository to be initialized
as a Git repository.

```text
KiLinSystemTests/
    run_all.ipynb
    workspaces/
        010_literature_original/
        010_literature_replay/    # created by replay()
        010_literature.ipynb
        010_literature_comparison.json
        010_literature_comparison.md
        020_code_original/
        020_code_replay/
        020_code.ipynb
```

From a notebook in `workspaces/`:

```python
from kilin.system_testing import Experiment, ReplayError

experiment = Experiment("010_literature_original")
report = experiment.validate()
report.raise_for_errors()

result = experiment.replay(
    progress=lambda i, n, op: print(i, n, op.kind)
)
print(result.workspace, result.operation_count, result.validation.valid)
for difference in result.differences:
    print(difference.sequence, difference.field)
    print("original:", difference.original)
    print("replayed:", difference.replayed)
```

`replay()` validates again before any execution. An optional `destination`
argument selects another absent sibling directory. By default the suffix
`_original` is replaced with `_replay`, or `_replay` is appended. Existing
paths, including empty directories and symbolic links, are refused. Move or
remove an earlier replay yourself before running another. The package never
automatically replaces or deletes a workspace.

### Execution and settings

Replay calls the synchronous `ChatEngine` commands used by the ordinary
frontend. It does not launch the desktop GUI. This is real execution, not a
simulation: models must be available according to the current environment
and the project settings; retrieval, routing, answering, tool calls, and
persistence run normally. Tools producing PDFs may initialize the ordinary
offscreen Qt report renderer. Recorded, compatible approvals for online
reference lookup may also result in ordinary network requests. Replay does
not force offline mode or install missing models outside the normal policies.
Only replay trusted reference workspaces, and review their operations before
execution if they contain network approvals or downloads.

Run the notebook from the common parent directory, not from a workspace
containing `profile.ini`. The existing GUI defaults command can consult a
profile in the working directory even when requesting new-project defaults;
replay refuses that situation before creating a workspace.

The original project type is retained. Inherited creation settings and
inherited per-question settings use current defaults. Explicit historical
settings are supplied as overrides. The original submitted form's fields
are retained, including partial backend mappings. Effort and search-depth
choices are reapplied to the current configuration. Effective profile,
behavior, tool, model and question settings are compared with the recorded
values, not forced to match them. If a changed policy makes an original
operation impossible, replay stops rather than silently changing that policy.

The original managed source copies supply all source bytes. Copy outcomes
identify the collision-adjusted managed names and selected files; later files
are imported only at their recorded operation. Replayed source inventories
and hashes must agree with the recorded historical manifests. Original
external source paths are never opened. The final retained folder snapshot
can expose future file names to source-change previews earlier than in the
original, but not to the managed source index before import. No-effect
requests for unretained external inputs can be omitted with an explicit
reported difference. A source command with no identifiable retained input
cannot be reproduced and is rejected. Do not include irrelevant, unretained
sources in reference commands.

Every question is executed in the mapped conversation with its recorded
inputs. New conversation and turn IDs are allocated normally; branch
prefixes must consist of the corresponding replayed turns. Explicit names,
feedback, exclusions, memory synchronization, archive operations, selection,
reset, project reopenings and card regeneration are replayed through the
same engine. A validated project rename is replayed through the normal
close and explicit resume commands, restoring the mapped active conversation
without renaming the replay directory. Normal resume can also emit a
source-change diagnostic that the original internal reopening did not emit.
The relocation provenance and this normalization are reported separately
from effective settings differences. Automatic titles, answers, index files,
cards, templates and outputs are produced afresh; they are not copied from
the reference.

### Interactions and failure

Only responses actually requested from the original user are replayed.
Option responses require the same question and a unique matching label in
the live choices; the original numeric position is not reused. An option
reordering or other option-list difference is recorded. Confirmation plans
must retain their tool, action, scope, and semantic details, including the
meaning of quick mode when selected. Ordinary model-work counts may differ
and are reported. Network approvals additionally require matching work
counts. Timing estimates are not replay inputs. An unexpected, missing,
cancelled, or ambiguous interaction stops the run; there is no fallback
choice, retry, or new approval inferred from a changed model response.

`ReplayResult` contains the closed replay workspace path, completed operation
count, original-to-replay conversation IDs and turn UUIDs, observed
`ReplayDifference` records, and an independent validation report for the new
workspace. The replay must itself validate and contain the same sequence of
command kinds. Differences in regenerated answers are not replay failures;
answer comparison is a separate concern. The returned mappings allow later
comparison to align branches and messages without relying on generated IDs.

On divergence, `ReplayError` gives the original operation sequence (if
known), the count of successfully checked operations, the destination path,
observed differences, and final validation when available. The partial replay
is closed if possible and retained for diagnosis; it is never presented as a
successful replay. Invalid original recordings raise `InvalidRecordingError`
before replay. Destination conflicts are rejected before execution.

## Comparing recorded workspaces

Once the replay exists, comparison can run independently, including from a
fresh process using the current package. It does not require a `ReplayResult`,
load the original answering model, or issue another question:

```python
from kilin.system_testing import Experiment

experiment = Experiment("010_literature_original")
comparison = experiment.compare()
comparison.display()
paths = comparison.save()
print(paths)
```

`destination` optionally selects another sibling replay. Both workspaces
must remain closed and stable throughout comparison. Validation errors are
included in the report, and an invalid or missing replay is not treated as
an empty successful run. No command pairs are inferred from invalid records.
When both recordings validate but their operations diverge, only the
unambiguous common prefix is paired. Matching uses recorded requests,
source effects, conversation/turn identities, branch prefixes, and semantic
user interactions, never answer similarity. A historical override may become
identical to a new default: its submitted value must still be preserved,
while the changed provenance remains observable. `comparison.complete`
means the full valid command histories align, not that their outputs match
or that every metric is available.

`comparison.data` is a JSON-compatible snapshot with `format_version`,
`original`, `replayed`, `alignment`, `summary`, `operations`, `final_state`,
`embedding_evaluation`, `files`, and `report_issues`. The workspace sections
include validation issues, recorded model/environment provenance, command
counts, final catalogs, and token coverage. Alignment includes the mapped
conversation IDs and turn UUIDs and any unmatched command suffixes. Each
operation retains intermediate stage observations, source manifests and
settings differences. Each matched committed question is counted once,
including when its turn is inherited in multiple branches.

### Metrics and coverage

- Text: exact saved equality, normalized visible equality and multiset
  token F1 (Dice overlap of words and punctuation). Normalization removes
  model thinking blocks, applies Unicode NFC and collapses whitespace, but
  retains case, punctuation and numbered citations. Empty pairs are not
  assigned an invented similarity.
- Evidence: Jaccard overlap of exact passages, source-file overlap, common
  rank-prefix overlap, and mean absolute rank displacement of shared items.
  Exact passage identity uses managed relative paths, line coordinates,
  available symbol/section/cell metadata and text SHA-256. Scores and
  generated chunk identifiers are not identities. Source-ranking channels
  use source paths rather than passages. Ranking occurrences are paired by
  recorded channel and ordinal, a descriptive comparison rather than proof
  that the changed pipeline issued semantically identical searches.
- Citations: numbered bracket citations are resolved against each saved
  evidence list, so renumbering alone does not imply changed evidence. The
  parser reports unresolved references and is syntactic; it does not check
  whether cited evidence supports a claim or whether a bracketed number in
  code is a citation.
- Execution: recorded intents, tools, interaction proposals and decisions,
  model-call provenance and available exact-text artifact metrics, event
  counts, status, and durations. Only recorded token counts are used. Saved
  turn totals and traced chat-call totals are separate observations, not
  additive. Generated titles and diagnostics outside command intervals are
  also retained. Nested inclusive timings are not a wall-time breakdown.
  Cross-logger command timing uses recorded UTC boundaries when no matching
  outer span duration is available.
- Files: final regular files in `sources`, `templates` and `outputs` are
  compared by relative path, size and SHA-256. Added, removed and changed
  files are distinguished. Byte changes in generated files are not semantic
  differences; indexes are neither copied nor judged by their bytes.

Unavailable measurements are `None` in Python and `null` in JSON. Means
include only available measurements and report available and total counts.
Neither an empty evidence union nor a missing trace field is replaced by
zero. Malformed optional diagnostic numbers are reported as unavailable.
No composite quality score, pass threshold, LLM judge, or correctness verdict
is computed.

### Context, tool coverage and independent inspection

Each side's independent event inventory and saved-question observations are
retained even without alignment. Unverified saved text from an invalid
recording is labeled as such; unmatched questions do not contribute to paired
metrics. Aligned history selections resolve branch-local turn numbers through
validated saved prefixes and the reconstructed identity map. Memory inventories
compare mapped turn/role identities and origins without regard to inventory
order; retrieval order, content fingerprints, priorities and scores remain
separate. Missing observations and unknown identities are unavailable, not
empty selections.

In addition to raw rendered prompts, structured-input comparisons normalize
only corroborated memory headers in the final user message. Their recorded
labels, roles, UUIDs and origins must agree with the preceding selected-memory
record. The diagnostic maps these identities and masks the timestamp, not
ordinary question text or memory content. Ambiguous headers are unavailable.
Neither model input nor trace data are changed.

Functional tool-summary comparisons separate known elapsed-time and plan-time
fields. Explicit incomplete or invalid outcomes remain visible even when
both sides fail identically. Unreported completion is not proof of success.

Closed GUI inspection after replay can append open, select and close commands.
A validated suffix consisting solely of such sessions without source changes
is annotated as inspection-only if the replay matches the preceding prefix.
Strict alignment remains divergent; substantive changes and failed commands
never qualify. A fresh replay reproduces the entire valid history.

### Effective model settings, outcome transitions and performance

Each paired model call retains the recorded backend, device, model source,
recognized Hugging Face snapshot revision, effective generation settings
(including suppression fields), template settings, thinking flag, token
counts, and termination reason. Comparisons are by enclosing stage and
occurrence, not by semantic equivalence. Snapshot revisions are recognized
lexically in the matching repository cache path. Unversioned local model
paths do not establish revision identity; no model files are opened.

`summary.model_reproducibility` counts input calls and unpaired inputs in
matched commands. Each equality statistic has `equal`, `available` and
`total` counts. Conditional categories require equal recorded model identity,
effective settings, exact rendered prompt and input-token count. Metadata-
normalized structured prompts are reported separately and never used to
qualify exact-input reproducibility. For zero and nonzero temperatures the
report records `identical_inputs`, `available_outputs` and `equal_outputs`.
Missing, ambiguous or incomplete input metadata cannot qualify. Equal output
text is not proof of answer correctness or a universal determinism guarantee.

Tool outcomes are `complete`, `incomplete`, `unknown` or `not_applicable`.
An absent tool summary with no recorded tool start is not applicable; absence
of a completion flag in a returned tool is unknown. Explicit validation
failure, incomplete items or recorded incomplete completion takes precedence
over a nominal success. Completion of a trace span or existence of a filename
alone does not demonstrate success. The paired `transition` describes a
reproduced completion/limitation, potential regression, possible improvement,
changed outcome or unavailable comparison. Historical generator hints remain
reference observations. Saved output paths are observations; the independent
final file inventories retain their actual hashes and presence.

The `performance` section covers each independently valid full workspace,
including overhead outside matched commands. Categories select corroborated
recorded spans for project initialization, indexing, model loading, retrieval
and embeddings, model inference, and question duration. Intervals within each
category and logger clock domain are merged to avoid counting nested spans
more than once. Categories overlap and are not additive; measurements are
reported with span coverage and no automatic performance thresholds. The
recorded output-event generation duration is used when the duplicate detail
is omitted. Generation and reply throughput use only actual recorded generated
token counts with positive corresponding durations, as a ratio of total
covered tokens to total covered time. Neither is necessarily decoding-only
throughput. Missing samples remain unavailable. Hardware/cache conditions
are not controlled by these observations.

### Optional embedding evaluation

For semantic answer comparison, explicitly choose a local or already cached
embedding model, and preferably pin its revision for repeated experiments:

```python
comparison = experiment.compare(
    embedding_model="your/local-or-cached-embedding-model",
    embedding_backend="auto",
    embedding_revision=None,
)
comparison.display()
comparison.save(overwrite=True)
```

A single evaluator instance embeds both normalized visible answers and is
closed after use. Cosine similarity is reported in [-1, 1]. The evaluator
uses the same model on both sides regardless of their original embedding
settings. The evaluation model, backend, requested revision and resolved
source are reported when available. Downloads, retries and fallback models
are not requested. Model loading or embedding errors leave the metric
unavailable with an explanation. No additional answer-generation calls are
made. With `embedding_model=None` (the default), no model is loaded.

### Reports and batch execution

`display()` renders Markdown in a notebook or prints it in a terminal.
`save()` writes `<experiment>_comparison.json` and
`<experiment>_comparison.md` beside the workspaces, removing the original's
`_original` suffix. Existing regular report files are replaced only with
`overwrite=True`. This flag applies to reports, not replay workspaces.
Each report file is published atomically, but the pair is not a filesystem
transaction. The JSON report contains saved answers and potentially
sensitive trace observations; keep it with the private experiment data.

A root-level batch notebook can enumerate `workspaces/*_original` and
compare each independently. Leave replay disabled unless explicitly
requested, never automatically delete an existing replay, and collect errors
per experiment so a divergent run does not suppress unrelated comparisons.

## Limits and ownership

Inspection assumes a stable, closed workspace. It does not acquire a writer
lock, so it creates no lock companions, nor does it make an atomic snapshot
of a concurrently changing directory. Copying or reading files can affect
filesystem access times; no workspace file contents are changed by inspection.
Only retained recording evidence is validated. It is not a cryptographic
attestation of all past actions: unrecorded actions or changes restored before
inspection cannot always be detected. Templates and generated outputs do not
have a complete historical mutation inventory, so the preparation protocol
prohibits manual changes. Validation does not authenticate the producer or
certify answer quality, verify real-model availability, or compare generated
indexes or outputs. A valid report establishes the checked recording
contracts, not deterministic reproducibility of model inference.

Replay assumes the original remains closed and unchanged throughout the run.
It does not hold a lock on the reference and must not be used with concurrent
writers. Environment and real-model availability remain the researcher's
responsibility. Replay establishes that the recorded operations and their
required durable effects were reproduced, not that regenerated model outputs
are identical or correct. Comparison is a separate read-only operation and never executes replay.

The [module map](module_map.md) lists the package modules and their tests.

## Managed replay and reference publication

The low-level `Experiment.replay()` still requires an absent destination.
Use the managed API to make the policy for an existing replay explicit:

```python
from kilin.system_testing import Experiment

experiment = Experiment("workspaces/010_original")
action = experiment.run_replay(existing_replay="archive")
print(action.status, action.archived_workspace, action.error)
```

`retain` (the default) leaves an existing replay unchanged; `archive` moves
it into a unique directory under `_replay_archive` outside workspace
discovery before creating a fresh replay; `error` refuses an existing replay.
The original must validate before a previous replay is moved. An archive
uses a same-filesystem rename, not copy-and-delete. A failed replay preserves
both the previous archive and the partial new recording. No policy overwrites
an original. A nonstandard archive directory must be separate from the
workspace discovery directory, and only directories with the `_replay`
suffix can be archived. Close both projects before invoking this API.

A `ManagedReplay` distinguishes a completed execution, retained recording,
and failed action. Only a successful fresh execution writes an exclusive
`logs/system_testing_execution.json` receipt, containing the recorded input
and output trace fingerprints, package content digest, available Git revision
and dirty status, Python/platform information, and UTC execution bounds.
Model identities and resolved revisions remain in the recorded trace model
observations. A missing historical receipt is reported as unavailable; the
inspection machine's environment is never attributed to a retained replay.
Failure to publish provenance is reported separately from replay completion.
Receipts are local integrity observations, not cryptographic attestations of
which process executed the code or protection from a malicious editor.

Generators can publish an optional sibling `<stem>_reference.json` receipt:

```python
from kilin.system_testing import publish_receipt, inspect_receipt

report = experiment.validate()
report.raise_for_errors()
publish_receipt(experiment.workspace, report, coverage=[])
print(inspect_receipt(experiment.workspace, report))
```

Publish only after the reference is complete and closed. The receipt records
the original event prefix, command count, managed source/template/output
hashes and generator coverage observations. It is never rewritten by an
ordinary replay or inspection. Later closed GUI inspection may extend the
trace without modifying the published prefix or managed files. Additional
substantive commands are reported as an extension; changed prefix or file
contents are reported separately. These observations do not override the
recording validator or block intentionally extended valid references.
Older or manually recorded references without receipts remain usable.

Comparison reports include `reference_publication` and `replay_execution`
observations, including whether the recorded input matches the current
reference exactly or only a historical prefix. These receipts do not
replace structural validation or strict
command alignment. No historical environment is inferred when it was not
recorded. The optional receipt does not fingerprint every workspace file or
detect changes that have been restored before inspection.


## Scripted references and notebook batch workflows

The optional workflow modules consolidate the former KiLinSystemTests
Python helpers. The experimental repository owns the experiment recipes,
local fixtures and results, not another copy of system-testing logic or
notebook templates. Existing recordings and version-1 comparison reports
retain their format and interpretation.

- `kilin.system_testing.generation` provides `Recorder`,
  `generate_reference`, `check_output` and `reference_overview`. Recipes
  supply a trusted function receiving the recorder, explicit project
  settings, the expected operation sequence, and a coverage callback.
  Staging occurs in `_recordings/` beside `workspaces/`, outside normal GUI
  discovery. Successful closed recordings are validated before publication
  and receive a reference receipt. Failed staging recordings remain for
  diagnosis. Existing originals, replays and receipts are protected.
  Scripted interactions allow only explicitly approved local plans and
  uniquely matching option labels. Recipe hints are coverage observations,
  not scientific judgments. The adapter uses the ordinary engine and does
  not change answer generation, retrieval, routing or tool behavior.
- `kilin.system_testing.batch` provides `discover_originals`, `run_batch`,
  `load_snapshot`, `collect_issues`, `group_issues` and `save_batch_summary`.
  It manages replay through `Experiment.run_replay`, compares each pair
  once, and supplies that same snapshot to generated comparison and answer
  notebooks. The caller controls the replay policy, report overwrite,
  notebook refresh and optional evaluation model. Templates are packaged
  resources in `kilin/system_testing/_notebooks/`; only these templates
  are executed automatically. Local template copies are not required or
  consulted. Historical snapshots are explicitly
  distinguished from current comparisons; failures do not silently fall
  back to stale data.
- `kilin.system_testing.analysis` presents comparison snapshots as escaped
  HTML and Markdown. It includes answer and operation explorers, batch
  summaries, event inventories, and optional exact local trace text
  verified against its recorded hashes. Rendering does not load models or
  revalidate a saved report.

### Release verification

A release check is a comparison between a fixed reference collection and
replays generated by the current implementation. Set the existing-replay
policy to `archive` for fresh replays; retaining or comparing saved
workspaces is useful for diagnostic inspection but does not constitute a
fresh replay. No longitudinal benchmark or automatic release verdict is
produced.

The batch dashboard separates execution status, paired measurement
coverage, behavioral agreement and tool-outcome transitions. A complete
comparison may still contain review findings. Blocking recording, replay,
alignment, comparison and requested artifact failures are critical;
intermediate failed stages in otherwise valid matched commands, incomplete
tool outcomes, unresolved citations and configuration or behavior changes
require review. Ordinary text differences, known relocation differences,
nonzero-temperature variation with identical recorded inputs and historical
generator hints are informational. A reproduced limitation is not evidence
of a new regression, and an incomplete-to-complete transition is not proof
of scientific improvement.

Aggregate agreement counts sum observed numerators and denominators, not
experiment-level percentages. Each metric includes available observation
counts and snapshot coverage. Current-invocation aggregates exclude saved
comparison snapshots; partial comparisons retain their measured common
prefix without implying complete execution. A report-only dashboard labels
its observations as saved. Missing observations are never assumed equal.

Event-count differences are attributed to aligned operations when the
stage counts reconcile with the independent workspace inventories. The
remaining differences are reported outside matched commands. When the
attribution is unavailable, only reconciled workspace totals are reported.
Related observations are grouped by aligned operation and diagnostic
category, retaining every underlying observation, including independent
recording-side sequences. Grouping does not establish causation. The answer
explorer's difference filter includes identical answers with incomplete
outcomes, citation problems or execution differences; timing alone is not
a behavioral difference.

For example, in a separate experiment repository with `workspaces/` and
one batch notebook:

```python
from pathlib import Path
from kilin.system_testing.batch import discover_originals, run_batch
from kilin.system_testing.analysis import batch_overview

root = Path.cwd()
reports, runs = run_batch(
    discover_originals(root / "workspaces"), root,
    run_replay=False, run_comparison=True, save_reports=True,
    overwrite_reports=True, save_visual_analysis=True,
    run_notebooks=False, refresh_notebooks=False,
)
html = batch_overview(reports, runs=runs)
```

The workflow modules do not add required installation dependencies. For
notebook creation and execution, install JupyterLab (or the corresponding
`nbformat`, `nbclient`, `jupyter_client` and `ipykernel` packages) in the
same Python environment as KíLín. These dependencies are imported only at
the notebook stages. `mistune` 3 is optional for formatted Markdown answers;
without it, answer text remains escaped. Normal system-testing imports do
not load Jupyter, a model backend or Qt. Generating and replaying actual
experiments still requires the configured KíLín backend and models.
