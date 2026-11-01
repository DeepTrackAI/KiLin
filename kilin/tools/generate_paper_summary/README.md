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
`kilin/project_types/<type>.json` may override these eight settings.
Missing keys use this tool's packaged defaults, **not** values from
`kilin_defaults.json`. The remaining sections are tool-internal
heuristics and estimation parameters; they are not project-type settings.
Project-type overrides of unrelated core KíLín defaults are not yet
supported and belong to a separate configuration-precedence change.

All packaged values are validated before the tool is usable. Values are
read once per Python process; edits require restarting KíLín. No tool
uses the global README to define its settings.

## `SETTINGS` — project-type-overridable values

| Key | Default | Meaning and validation |
| --- | --- | --- |
| `TEMPLATE` | `research_briefing` | Default template ID when no format is named; a safe filename stem, without `.md` or directory separators. |
| `PART_TOKENS` | 6000 | Approximate paper tokens read per model call; integer, 1000–16000. |
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

The reference list is split into entries by the shared, model-free
extractor in `kilin/tools/_shared/references.py`. This tool owns the
thresholds: the first five keys below are passed to it as
`REFERENCE_LAYOUT`, so other tools using the extractor keep their own
values. The tool passes the paper's file, so a PDF's reference list is
read again in column order (two-column lists are otherwise interleaved)
and the lines that indexed passages overlap on are read once. `DEFAULT_REFERENCE_ITEMS` and `COMMON_WORDS` are used only by this
tool, when choosing entries for a report.

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
| `HEADLINE_TOKENS` | 128 | Nonthinking cap for a headline; ignored when thinking is on. Positive integer. |
| `REFERENCES_TOKENS` | 128 | Nonthinking cap for reference identifiers; ignored when thinking is on. Positive integer. |

## `ESTIMATES` — model-time planning assumptions

- `NOTE_REPLY_TOKENS` (default `256`): expected reply tokens for
  duration estimates only; never a generation cap. Range: 1-131072.
- `SECTION_REPLY_TOKENS` (default `512`): expected reply tokens for
  duration estimates only; never a generation cap. Range: 1-131072.

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

## Report formatting and PDF output

This tool always writes the validated Markdown report to the project's
`outputs/` folder. Its private `_report_format.py` converts the supported
report Markdown to escaped, styled HTML and extracts the report title. It
has no GUI or PySide6 dependency. Formatting options are arguments of the
formatter, not tool defaults.

PDF generation is **optional**: the caller can supply the `ToolContext`
`write_pdf` callback to render the Markdown. The desktop GUI supplies the
private `kilin/gui/_report_pdf.py` adapter, which uses Qt's `QPdfWriter`
and tries to fit the report on an A4 page by reducing its font size.
When no callback is available (for example, from the CLI or a notebook),
the Markdown report is still saved and no PDF is attempted. PDF errors
or multi-page output are reported without discarding the Markdown file.

The tool provides the PDF renderer with a private staging path in `outputs/`.
Once a nonempty PDF is produced, it publishes the file under a numbered,
non-overwriting name. The renderer must write to the staging path it is
given, not select its own destination. Failed renders leave no partial
public PDF and do not remove the saved Markdown report.

PDF generation is not a project-type output-format setting. The caller
controls whether a PDF renderer is available through `ToolContext`.

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
from the package. Both `load_template` and `template_names` require an
explicit directory to avoid accidentally reading global shipped assets.
Symbolic links are not accepted as project templates. See
`docs/project_workspaces.md` for the workspace resource contract.

## Tests

`tests/kilin/tools/generate_paper_summary/test_tool.py` tests the settings,
registry integration, resource selection, parsing, and model-free error
paths. `tests/kilin/tools/test__defaults.py` tests the shared private
defaults reader. Further model orchestration tests can exercise the
`ToolContext` callbacks with fakes instead of requiring an LLM.

## Model completion

Model calls must finish with usable final text. A token-limit stop, unfinished
thinking block or missing final answer makes the operation incomplete; it is
not a zero-match result. The coordinator saves an explicit failure notice
and completion metadata without adding retries. Internal parsers never
receive thinking text.

A report with no usable cited notes is not written. Reports which fail the
final template or reference checks are returned with `status: incomplete`,
`valid: false` and the validation problems. Neither Markdown nor PDF is
published in that case; existing files remain untouched.


## Generation settings

Output allowances come from the active session, including per-question GUI
or API settings. They are ceilings for each model call, not target lengths
or a total budget for a multi-call tool. Prose length is controlled by the
prompt or report template; input passage budgets remain separate.

`MODEL_CALLS` contains one key: `THINKING` (packaged value `"inherit"`).
It accepts `inherit`, `on`, or `off` for internal `context.ask_model` calls.
It never infers thinking from a token count. `context.write` always inherits
main-session thinking. Router calls remain nonthinking and separately bounded.

Thinking requests use the user's full allowance. Only nonthinking requests
may use an explicit smaller cap, never exceeding the user's maximum.
Changing `THINKING` to `inherit` resets this tool to the main setting; `on`
forces thinking and bypasses local short-response caps. Restart after editing
cached defaults. See the [shared policy](../../../docs/configuration.md#internal-tool-thinking)
for precedence, the retained exceptions, and migration of obsolete keys.
