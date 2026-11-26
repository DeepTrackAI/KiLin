# System testing: reference inspection and replay

The `kilin.system_testing` package inspects a closed, recorded project as a
reference experiment. It reads the ordinary project and conversation traces,
including archived conversations, and reconstructs the recorded user commands
in project-wide sequence order. Inspection does not start the GUI, initialize
an inference backend, load models, or download resources. Replay is a separate,
explicit operation that runs the validated history in a new workspace.

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
- Keep the original workspace unchanged. Replay requires a separate sibling
  workspace path that does not yet exist.

GUI appearance and view-only actions are not replay inputs. Older recordings
or direct core-only sessions that do not contain the required command and
creation metadata are not complete reference experiments. No migration or
inference of missing command history is attempted.

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
recorded successful deletion.

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
both can be opened in the ordinary GUI. Notebooks and the eventual comparison
reports also belong in that repository. The replay implementation itself
belongs to KíLín; it does not require an experiment repository to be initialized
as a Git repository.

```text
KiLinSystemTests/
    010_literature_original/
    010_literature_replay/       # created by replay()
    010_literature.ipynb
    020_code_original/
    020_code_replay/
    020_code.ipynb
```

From a notebook in this directory:

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
same engine. Automatic titles, answers, index files, cards, templates and
outputs are produced afresh; they are not copied from the reference.

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
are identical or correct. Quantitative comparison and reporting remain
separate from this API.

The [module map](module_map.md) lists the package modules and their tests.
