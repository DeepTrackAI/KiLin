"""Plan additive replay inputs from retained managed source snapshots.

Key Features
------------
- Map historical external identities to actual retained source destinations.
- Reproduce historical selections without copying later files prematurely.
- Check source inventories and content hashes after replayed operations.

Module Structure
----------------
- ``SourceReplay`` prepares creation/import/sync arguments and checks copies.

Examples
--------
>>> from kilin.system_testing import Experiment
>>> # Experiment("workspaces/010_original").replay()

"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from kilin.repository import _manifest_files
from kilin.workspace import _hash_file

from .timeline import TimelineOperation, one
from .traces import safe_path

__all__ = []


@dataclass(slots=True)
class SourceCommand:
    """Prepared arguments for an ordinary creation, addition or sync."""

    sources: list[str] = field(default_factory=list)
    selections: dict[str, list[str]] = field(default_factory=dict)
    decisions: dict[str, str] | None = None
    manifest: dict[str, dict] = field(default_factory=dict)
    omitted: tuple[str, ...] = ()


def _scan_error(error: OSError) -> None:
    """Propagate an inaccessible source directory.

    Parameters
    ----------
    error: OSError
        Directory traversal failure.

    """
    raise error


class SourceReplay:
    """Preflight source commands without using historical external paths.

    Parameters
    ----------
    root: pathlib.Path
        Closed, validated original workspace.
    operations: tuple[TimelineOperation, ...]
        Complete chronological reference operations.

    Attributes
    ----------
    commands: dict[int, SourceCommand]
        Source arguments and expected inventories by original sequence.

    Notes
    -----
    The normal importer allocates the destination from the basename of the
    input. Use the already collision-adjusted managed basename, not the
    original external basename. All imported files are explicitly selected.
    Original folder snapshots can contain later additions, but those files
    must not enter the replay's managed sources until their recorded batch.

    """

    def __init__(self, root: Path,
                 operations: tuple[TimelineOperation, ...]) -> None:
        """Prepare every source command before opening a model.

        Parameters
        ----------
        root: pathlib.Path
            Validated reference root.
        operations: tuple[TimelineOperation, ...]
            Chronologically ordered reference operations.

        """
        self.root = root
        self.commands: dict[int, SourceCommand] = {}
        self._origins: dict[str, str] = {}
        self._manifest: dict[str, dict] = {}
        for op in operations:
            if op.kind in ("create_project", "add_sources", "sync_sources",
                           "open_project"):
                self.commands[op.start_sequence] = self._plan(op)

    def _plan(self, op: TimelineOperation) -> SourceCommand:
        """Reconstruct a source command using actual copy outcomes.

        Parameters
        ----------
        op: TimelineOperation
            Validated original operation.

        Returns
        -------
        SourceCommand
            Arguments and expected post-operation source inventory.

        """
        command = SourceCommand()
        copies = op.matching("workspace.source.copied")
        batch = op.matching("workspace.prepare.started")
        if len(batch) != bool(copies) and not (
            len(batch) == 1 and op.kind == "create_project" and not copies
            and not batch[0].details["source_paths"]
        ):
            raise ValueError("Missing or ambiguous source preparation batch.")
        actual = []
        for event in copies:
            data = event.details
            original = data["source"]
            managed = data["workspace_path"]
            path = safe_path(self.root, managed)
            if (not managed.startswith("sources/")
                    or len(Path(managed).parts) != 2
                    or original in self._origins
                    or data["preserved_files"]):
                raise ValueError("Expected a new, additive source copy.")
            selected = data["copied_files"]
            if data["kind"] == "file" and path.is_file():
                if len(selected) != 1:
                    raise ValueError("Ambiguous direct file copy.")
                selected = [path.name]
            elif data["kind"] == "folder" and path.is_dir():
                selected = list(selected)
                for name in selected:
                    if not safe_path(self.root, f"{managed}/{name}").is_file():
                        raise ValueError(f"Missing retained file: {name}")
            else:
                raise ValueError(f"Missing retained source: {managed}")
            if not selected or str(path) in command.selections:
                raise ValueError("Ambiguous retained source selection.")
            actual.append(str(path))
            self._origins[original] = str(path)
            command.selections[str(path)] = selected
        # A source batch preserves the order of unique resolved inputs. For
        # one-to-one batches this also establishes their submitted aliases.
        if batch:
            submitted = batch[0].details["source_paths"]
            if len(submitted) == len(actual):
                for raw, path in zip(submitted, actual):
                    if self._origins.get(raw, path) != path:
                        raise ValueError("Conflicting source identity.")
                    self._origins[raw] = path
        if op.kind == "create_project":
            command.sources = actual
        elif op.kind == "add_sources":
            completed = one(op.events, "gui.sources.add.completed").details
            new = completed["new_sources"]
            if len(new) != len(actual) or any(
                self._origins.get(raw) != path
                for raw, path in zip(new, actual)
            ):
                raise ValueError("Cannot map the recorded new source batch.")
            # The canonical new-input order is recorded separately from
            # the raw request. Aliases no longer on disk need not be guessed
            # to reproduce the actual new copies.
            command.sources = list(actual)
            omitted = []
            for raw in op.request["sources"]:
                path = self._origins.get(raw)
                if path is None:
                    # Skipped originals are not retained in the workspace.
                    # They cannot supply replay inputs or alter its sources.
                    omitted.append(raw)
                elif path not in command.sources:
                    command.sources.append(path)
            if not command.sources:
                raise ValueError("No unambiguous retained source request.")
            command.omitted = tuple(omitted)
        sync = op.matching("workspace.sync.started")
        if sync:
            request = one(op.events, "workspace.sync.started").details
            imports = {e.details["path"]: "import" for e in op.matching(
                "workspace.sync.file"
            ) if e.details["action"] == "import"}
            if len(imports) != len(op.matching("workspace.sync.file")):
                raise ValueError("Nonadditive synchronization is unsupported.")
            decisions = request["decisions"]
            # A default sync can act on newly visible external files. Use
            # explicit additive actions only, never the automatic policy.
            command.decisions = dict(imports)
            for path, action in (decisions or {}).items():
                if action == "import" and path in imports:
                    continue
                if action != "keep":
                    raise ValueError("Unsupported synchronization decision.")
                # Preserve a keep only when that unimported source is still
                # represented in the final snapshot. Never invent a file.
                retained = safe_path(self.root, path)
                relative = path.removeprefix("sources/")
                if (path.startswith("sources/") and retained.is_file()
                        and relative not in self._manifest
                        and path not in imports):
                    command.decisions[path] = "keep"
                else:
                    command.omitted += (f"keep: {path}",)
        elif op.kind == "add_sources":
            command.decisions = op.request["decisions"]
            if command.decisions:
                raise ValueError("Source decisions lack a recorded sync.")
        manifests = op.matching("index.check.completed", "index.empty")
        if manifests:
            self._manifest = _manifest_files(
                manifests[-1].details["source_manifest"]
            )
        command.manifest = dict(self._manifest)
        return command

    def verify(self, destination: Path, op: TimelineOperation) -> None:
        """Check actual source inventory and bytes after a source command.

        Parameters
        ----------
        destination: pathlib.Path
            Newly created replay workspace.
        op: TimelineOperation
            Original command that has just been executed.

        """
        if op.start_sequence not in self.commands:
            return
        expected = self.commands[op.start_sequence].manifest
        root = safe_path(destination, "sources")
        if not root.is_dir():
            raise ValueError("Replay has no managed source directory.")
        found = set()
        for parent, directories, files in os.walk(root, onerror=_scan_error):
            for name in directories + files:
                path = Path(parent) / name
                if path.is_symlink():
                    raise ValueError(f"Linked replay source: {path}")
                if name in files:
                    if not path.is_file():
                        raise ValueError(f"Nonregular replay source: {path}")
                    found.add(path.relative_to(root).as_posix())
        if found != set(expected):
            raise ValueError("Replay source inventory differs from history: "
                             f"missing={sorted(set(expected) - found)}, "
                             f"extra={sorted(found - set(expected))}.")
        for relative, record in expected.items():
            path = safe_path(root, relative)
            if (path.stat().st_size != record["size_bytes"]
                    or _hash_file(path) != record["sha256"]):
                raise ValueError(f"Replay source content differs: {relative}")
