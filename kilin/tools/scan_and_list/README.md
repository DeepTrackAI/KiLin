# scan_and_list

This package owns the `scan_and_list` registered tool. Its `defaults.json` is independent of `kilin_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `CASE_SENSITIVE` (default `false`): Whether letter case must match.
- `WHOLE_WORDS` (default `true`): Whether a term must not continue into adjacent letters or digits.
- `CONTEXT_LINES` (default `1`): Lines of context kept around each mention (0-5). Allowed range: 0–5.
- `MAX_LISTED` (default `100`): Mentions listed with citations (1-1000); counts include all. Allowed range: 1–1000.

### `HEURISTICS`

- `UNSEARCHABLE_EXAMPLES` (default `5`): Maximum examples of files without indexed searchable text. Allowed range: 0–100.


## Internal model thinking

`MODEL_CALLS` in this tool's `defaults.json` controls only
`ToolContext.ask_model`, identically in the notebook and GUI.

Internal requests inherit the main thinking setting. This tool either produces narrative explanations or does not use internal answering-model calls; no short-reply override is applied.

| Key | Packaged value | Meaning |
| --- | --- | --- |
| `THINKING` | `inherit` | `inherit` follows the session subject to the short-reply rule; `on` or `off` forces all internal requests and takes precedence. |
| `SHORT_REPLY_THINKING` | `inherit` | `inherit`, `on`, or `off` for bounded short requests, only when `THINKING` is `inherit`. |
| `SHORT_REPLY_MAX_TOKENS` | `0` | Inclusive requested output-token bound; integer 0-100000. Zero disables the rule. `tokens=None` never counts as short. |

To restore full inheritance, set `THINKING` and `SHORT_REPLY_THINKING` to
`inherit` (or set the short bound to zero). To force thinking for every
internal request, set `THINKING` to `on`. Restart KíLín or the notebook kernel
after editing these cached packaged defaults. Existing projects need no
reindexing; saved answers are not changed. Project-type tool settings do not
override this section.

These controls do not change the selected model, temperature, output cap,
or number of model calls. They do not affect `context.write()` final answers
or the nonthinking router. A small output budget can still be exhausted when
thinking is forced on; the completion guard reports an incomplete operation
instead of a false result. Models must support the existing thinking control.
See [configuration policy](../../../docs/configuration.md#internal-tool-thinking).
