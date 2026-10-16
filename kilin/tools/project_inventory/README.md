# project_inventory

This package owns the `project_inventory` registered tool. Its `defaults.json` is independent of `kilin_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `MAX_LISTED` (default `200`): Files listed by name (1-2000); counts include all. Allowed range: 1–2000.
- `MAX_FOLDERS` (default `20`): Folders shown in the per-folder counts (1-100). Allowed range: 1–100.

