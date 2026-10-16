# paper_overview

This package owns the `paper_overview` registered tool. Its `defaults.json` is independent of `kilin_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `MAX_TOKENS` (default `6000`): Most estimated tokens of the paper given to the model (1000-20000). Allowed range: 1000–20000.
- `ANSWER_TOKENS` (default `500`): Longest answer, in tokens (100-2000). Allowed range: 100–2000.

### `HEURISTICS`

- `OPENING_PASSAGES` (default `2`): Number of first passages used to introduce a paper. Allowed range: 1–100.

