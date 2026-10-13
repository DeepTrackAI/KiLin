# `generate_paper_summary`

This tool converts one paper into a structured Markdown report using a
project-owned template. It reads a paper in parts, collects notes, writes
sections, checks their structure and length, and saves the report to the
project's `outputs/` directory. The model can still make factual errors:
validation checks structure, not scientific correctness.

The tool is a self-contained package:

- `tool.py`: settings validation, resource declarations, and execution.
- `_configuration.py`: validates all packaged defaults at import time.
- `_report_template.py`: reads templates and checks generated reports.
- `defaults.json`: default settings and private algorithm parameters.
- `templates/`: Markdown templates available for selective installation.

## Configuration and precedence

The `SETTINGS` section supplies defaults for the tool's
`SummarySettings`. Each intent's `tool` entry in
`qachat/project_types/<type>.json` may override these eight settings.
Missing keys use this tool's packaged defaults, **not** values from
`qachat_defaults.json`. The remaining sections are tool-internal
heuristics and estimation parameters; they are not project-type settings.
Project-type overrides of unrelated core QAChat defaults are not yet
supported and belong to a separate configuration-precedence change.

All packaged values are validated before the tool is usable. Values are
read once per Python process; edits require restarting QAChat. No tool
uses the global README to define its settings.

## `SETTINGS` — project-type-overridable values

| Key | Default | Meaning and validation |
| --- | --- | --- |
| `TEMPLATE` | `research_briefing` | Default template ID when no format is named; a safe filename stem, without `.md` or directory separators. |
| `PART_TOKENS` | 6000 | Approximate paper tokens read per model call; integer, 1000–16000. |
| `NOTE_TOKENS` | 500 | Maximum tokens of notes per reading part; integer, 100–1500. |
| `SECTION_TOKENS` | 600 | Maximum tokens when writing a section without a specified length; integer, 100–2000. |
| `REPAIR_ATTEMPTS` | 1 | Allowed targeted revisions of a nonconforming section; integer, 0–4. |
| `MAX_MODEL_CALLS` | 20 | Limit on paper-reading parts; integer, 1–200. |
| `CONFIRMATION` | `when_large` | Confirmation policy: `always`, `when_large`, or `never`. |
| `LARGE_BATCHES` | 3 | Reading parts above which `when_large` requests confirmation; integer, 0–100. |

## `PAPER` — initial paper analysis

| Key | Default | Meaning and validation |
| --- | --- | --- |
| `BEGINNING_PASSAGES` | 3 | First passages searched for title, authors and publication metadata; positive integer. |
| `BEGINNING_CHARACTERS` | 1500 | Maximum characters inspected near the paper beginning; positive integer. |
| `ORGANISATION_WORDS` | 10 words | Lowercase terms distinguishing an authoring organization from a person (e.g. `consortium`, `laboratory`); nonempty list of distinct words. |

## `REFERENCES` — reference extraction

| Key | Default | Meaning and validation |
| --- | --- | --- |
| `REFERENCE_CHARACTERS` | 40000 | Maximum characters considered after the references heading; positive integer. |
| `NUMBERED_ENTRIES_MIN` | 5 | Minimum numbered reference lines for a number-based split; positive integer. |
| `RUNNING_LINE_REPEATS` | 3 | Repetitions needed to classify a short line as running page furniture; positive integer. |
| `RUNNING_LINE_WORDS` | 12 | Maximum words in a candidate running line; positive integer. |
| `ENTRY_MIN_WORDS` | 5 | Minimum words in a candidate reference entry; positive integer. |
| `DEFAULT_REFERENCE_ITEMS` | `[1, 3]` | Lower and upper item limits if a references section has no explicit limit; two positive ascending integers. |
| `COMMON_WORDS` | 6 words | Lowercase words ignored when matching reference text (e.g. `this`, `paper`); nonempty list of distinct words. |

## `WRITING` — report-writing prompts

| Key | Default | Meaning and validation |
| --- | --- | --- |
| `EARLIER_SECTION_WORDS` | 30 | Words retained from previously written sections to discourage repetition; positive integer. |
| `HEADLINE_TOKENS` | 60 | Maximum response tokens for writing or repairing a headline; positive integer. |
| `REFERENCES_TOKENS` | 40 | Maximum response tokens for choosing reference-item numbers; positive integer. |

## `ESTIMATES` — model-time planning assumptions

These values are **planning estimates**, not model-specific measurements.
They preserve the original assumptions for a small local model on Apple
silicon. Model-speed calibration is a separate concern.

| Key | Default | Meaning and validation |
| --- | --- | --- |
| `PROMPT_TOKENS_PER_SECOND` | 700.0 | Estimated model prompt-processing throughput; positive finite number. |
| `REPLY_TOKENS_PER_SECOND` | 25.0 | Estimated generation throughput; positive finite number. |
| `CALL_SECONDS` | 1.0 | Estimated fixed latency per model call; nonnegative finite number. |
| `PART_PROMPT_TOKENS` | 300 | Estimated instruction tokens in each reading-part call; nonnegative integer. |
| `SECTION_PROMPT_TOKENS` | 400 | Estimated instruction tokens in a writing call; nonnegative integer. |
| `REPLY_SHARE` | 0.6 | Expected average proportion of the allowed output-token budget consumed; finite fraction from 0 to 1. |
| `REFERENCES_REPLY_TOKENS` | 20 | Estimated response tokens for choosing reference numbers; nonnegative integer. |

## `TEMPLATE_VALIDATION` — report checks

| Key | Default | Meaning and validation |
| --- | --- | --- |
| `LIMIT_TOLERANCE` | 0.1 | Permitted fractional deviation from word-count limits, but not item limits; finite fraction from 0 to 1. |
| `MIN_TOTAL_WORDS` | 50 | Smallest allowed template `max_words` front-matter value; positive integer. |

## Workspace resources

The core always creates `templates/` and `outputs/` for every project.
On **new** project creation, the tool's `workspace_resources(settings)`
returns the one bundled template selected by each configured intent.
The core deduplicates repeated declarations and copies only required files.
An unbundled custom template name results in no packaged resource:
users can create the Markdown file manually in `templates/`.

Opening an existing project never replaces, recreates, or deletes its
project-owned templates. Edited and custom files take precedence over the
bundled resources; a missing file is reported instead of silently read
from the package. See `docs/project_workspaces.md` for the workspace
resource contract.

## Tests

`tests/qachat/tools/test_generate_paper_summary.py` tests the settings,
registry integration, resource selection, parsing, and model-free error
paths. `tests/qachat/tools/test_tool_defaults.py` tests the shared private
defaults reader. Further model orchestration tests can exercise the
`ToolContext` callbacks with fakes instead of requiring an LLM.
