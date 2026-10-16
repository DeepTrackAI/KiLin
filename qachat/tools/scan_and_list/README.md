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

