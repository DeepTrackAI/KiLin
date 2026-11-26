# System testing: reference inspection

The `kilin.system_testing` package inspects a closed, recorded project as a
reference experiment. It reads the ordinary project and conversation traces,
including archived conversations, and reconstructs the recorded user commands
in project-wide sequence order. It does not start the GUI, initialize an
inference backend, load models, download resources, or replay commands.

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
- Keep the original workspace unchanged. A subsequent replay, when supported,
  should use a separate empty workspace.

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

The [module map](module_map.md) lists the package modules and their tests.
