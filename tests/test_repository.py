"""Tiny temporary folders test selection; no source contents need to be read."""

import builtins
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from app_core import repository


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "small project"
    files = {
        "README.md": "# A small project\n",
        "src/main.py": "raise RuntimeError('This source must never run')\n",
        "src/helpers.py": "def add(a, b): return a + b\n",
        "src_extra/other.py": "x = 1\n",
        "tests/test_main.py": "assert True\n",
        "docs/guide.MD": "Some instructions.\n",
        "settings.toml": "name = 'example'\n",
        "empty.py": "",
        "image.png": "not a selected text format",
        ".env": "FAKE_SECRET=not-a-real-secret",
        ".gitignore": "src/\n",
        ".git/config": "fake Git metadata",
        ".venv/package.py": "installed package",
        "models/config.json": "{}",
        "node_modules/package/index.js": "package code",
        "cache.safetensors": "fake weights",
        "credentials.json": "{}",
    }
    for name, text in files.items():
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return folder


def by_path(report):
    return {row["path"]: row for row in report["entries"]}


def test_default_preview_is_sorted_and_explains_skips(project):
    report = repository.scan_repository(project)
    assert report["complete"] is True
    assert report["selected_files"] == [
        "README.md", "docs/guide.MD", "settings.toml", "src/helpers.py",
        "src/main.py", "src_extra/other.py", "tests/test_main.py",
    ]
    rows = by_path(report)
    assert list(rows) == sorted(rows)
    assert rows["image.png"]["reason"] == "file type not selected"
    assert rows["empty.py"]["reason"] == "empty file"
    assert rows[".env"]["reason"] == "hidden entry"
    assert rows["models"]["kind"] == "folder"
    assert "models/config.json" not in rows
    assert report["selected_bytes"] == sum((project / name).stat().st_size for name in report["selected_files"])


def test_preview_neither_reads_contents_runs_programs_nor_changes_files(project, monkeypatch):
    before = {path.relative_to(project): (path.stat().st_size, path.stat().st_mtime_ns)
              for path in project.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Scanning must not read/write file contents or start another program")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    report = repository.scan_repository(project)
    assert "src/main.py" in report["selected_files"]
    after = {path.relative_to(project): (path.stat().st_size, path.stat().st_mtime_ns)
             for path in project.rglob("*") if path.is_file()}
    assert before == after


def test_rules_are_explicit_and_do_not_claim_to_parse_gitignore(project):
    # The fixture's .gitignore says src/, but this preview does not read it.
    assert "src/main.py" in repository.scan_repository(project)["selected_files"]
    report = repository.scan_repository(project, exclude_paths=["src"])
    assert "src/main.py" not in report["selected_files"]
    assert "src_extra/other.py" in report["selected_files"]


def test_relative_root_always_starts_at_qachat(project, monkeypatch, tmp_path):
    monkeypatch.setattr(repository, "PROJECT_ROOT", project.parent)
    monkeypatch.chdir(tmp_path.parent)
    report = repository.scan_repository("small project")
    assert report["repository"] == str(project.resolve())


def test_home_shortcut_can_select_a_project_subfolder(project, monkeypatch):
    original = Path.expanduser
    monkeypatch.setattr(Path, "expanduser", lambda path: project if str(path) == "~/small project" else original(path))
    assert repository.scan_repository("~/small project")["repository"] == str(project.resolve())


@pytest.mark.parametrize("includes,expected", [
    (["src"], ["src/helpers.py", "src/main.py"]),
    (["src/main.py"], ["src/main.py"]),
    (["src/main.py", "docs"], ["docs/guide.MD", "src/main.py"]),
    (["src\\main.py"], ["src/main.py"]),
    (["missing"], []),
])
def test_include_folder_file_and_multiple_paths(project, includes, expected):
    assert repository.scan_repository(project, include_paths=includes)["selected_files"] == expected


def test_exclusion_wins_over_inclusion(project):
    report = repository.scan_repository(project, include_paths=["src"], exclude_paths=["src/helpers.py"])
    assert report["selected_files"] == ["src/main.py"]
    assert by_path(report)["src/helpers.py"]["reason"] == "excluded by your selection"
    assert repository.scan_repository(project, exclude_paths=["."])["selected_files"] == []


def test_extensions_are_case_insensitive_and_can_be_empty(project):
    report = repository.scan_repository(project, extensions=["MD", ".md"])
    assert report["extensions"] == [".md"]
    assert report["selected_files"] == ["README.md", "docs/guide.MD"]
    assert repository.scan_repository(project, extensions=[])["selected_files"] == []


def test_compound_extension_is_literal(project):
    (project / "example.gen.m").write_text("some source")
    assert repository.scan_repository(project, extensions=[".gen.m"])["selected_files"] == ["example.gen.m"]


@pytest.mark.parametrize("name", ["models", "weights", "venv", "node_modules", "build", "dist", "__pycache__", "logs", ".private"])
def test_protected_folders_are_not_entered_even_when_included(tmp_path, monkeypatch, name):
    folder = tmp_path / "project"
    protected = folder / name
    protected.mkdir(parents=True)
    (protected / "private.py").write_text("never read")
    original = repository.os.scandir

    def guarded(path):
        assert Path(path) != protected
        return original(path)

    monkeypatch.setattr(repository.os, "scandir", guarded)
    report = repository.scan_repository(folder, include_paths=[name])
    assert report["selected_files"] == []
    assert list(by_path(report)) == [name]


@pytest.mark.parametrize("filename,extension", [("cache.safetensors", ".safetensors"), ("key.pem", ".pem"), ("credentials.json", ".json"), (".env", ".env")])
def test_sensitive_names_and_weights_cannot_be_selected(project, filename, extension):
    (project / filename).write_text("not real private data")
    report = repository.scan_repository(project, extensions=[extension], include_paths=[filename])
    assert report["selected_files"] == []
    assert by_path(report)[filename]["status"] == "skipped"


def test_metadata_selection_is_not_binary_detection(project):
    (project / "disguised.py").write_bytes(b"\x00\x01binary")
    assert "disguised.py" in repository.scan_repository(project)["selected_files"]


def test_exact_file_size_boundary(project):
    (project / "four.txt").write_bytes(b"1234")
    (project / "five.txt").write_bytes(b"12345")
    report = repository.scan_repository(project, extensions=[".txt"], max_file_bytes=4)
    assert report["selected_files"] == ["four.txt"]
    assert by_path(report)["five.txt"]["reason"] == "larger than the file-size limit"


@pytest.mark.parametrize("path", ["../outside", "src/../../outside", "/outside", "C:\\outside", "C:outside", "\\\\server\\share", "src/*.py", "src/?.py", "", "src\0bad"])
@pytest.mark.parametrize("setting", ["include_paths", "exclude_paths"])
def test_invalid_selection_paths_rejected(project, path, setting):
    with pytest.raises(ValueError):
        repository.scan_repository(project, **{setting: [path]})


@pytest.mark.parametrize("options", [
    {"include_paths": "src"}, {"exclude_paths": "tests"}, {"include_paths": [None]},
    {"extensions": ".py"}, {"extensions": [None]}, {"extensions": ["*"]},
    {"extensions": [".py/secret"]}, {"extensions": [""]},
    {"max_file_bytes": 0}, {"max_entries": True}, {"max_entries": 1.5},
])
def test_bad_options_are_clear(project, options):
    with pytest.raises(ValueError):
        repository.scan_repository(project, **options)


def test_missing_folder_and_file_root_have_clear_errors(project):
    with pytest.raises(FileNotFoundError):
        repository.scan_repository(project / "missing")
    with pytest.raises(NotADirectoryError, match="Choose a folder"):
        repository.scan_repository(project / "README.md")


@pytest.mark.parametrize("root", ["", None, "https://github.com/example/repo", "git@example:repo"])
def test_invalid_or_remote_roots_rejected(root):
    with pytest.raises(ValueError, match="local"):
        repository.scan_repository(root)


def test_whole_home_or_disk_is_not_a_project():
    for root in (Path.home(), Path(Path.cwd().anchor)):
        with pytest.raises(ValueError, match="project folder"):
            repository.scan_repository(root)


def test_limit_fails_explicitly_instead_of_returning_a_partial_selection(project):
    with pytest.raises(ValueError, match="Scan limit reached"):
        repository.scan_repository(project, max_entries=2)


def test_exact_entry_limit_and_empty_folder(tmp_path):
    report = repository.scan_repository(tmp_path, max_entries=1)
    assert report["complete"] and report["entries"] == []
    (tmp_path / "one.py").write_text("x = 1")
    report = repository.scan_repository(tmp_path, max_entries=1)
    assert report["visited_entries"] == 1 and report["selected_files"] == ["one.py"]


def test_links_are_never_followed_including_loops_and_broken_links(project, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "outside.py").write_text("outside source")
    links = {"linked-folder": outside, "linked.py": outside / "outside.py",
             "loop": project, "broken.py": outside / "missing"}
    try:
        for name, target in links.items():
            (project / name).symlink_to(target, target_is_directory=target.is_dir())
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this machine")
    report = repository.scan_repository(project)
    rows = by_path(report)
    for name in links:
        assert rows[name]["kind"] == "link"
        assert rows[name]["reason"] == "links are not followed"
    assert not any("outside.py" in name for name in report["selected_files"])


def test_junction_is_skipped_without_entering_it(project, monkeypatch):
    junction = project / "junction"
    junction.mkdir()
    (junction / "file.py").write_text("outside-like target")
    monkeypatch.setattr(Path, "is_junction", lambda path: path == junction)
    rows = by_path(repository.scan_repository(project))
    assert rows["junction"]["kind"] == "link"
    assert "junction/file.py" not in rows


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Named pipes are Unix-only")
def test_special_files_are_not_opened(project):
    os.mkfifo(project / "pipe.py")
    row = by_path(repository.scan_repository(project))["pipe.py"]
    assert row["reason"] == "not a regular file"


def test_unreadable_subfolder_is_reported_as_incomplete(project, monkeypatch):
    original = repository.os.scandir

    def guarded(path):
        if Path(path) == project / "docs":
            raise PermissionError("test denied")
        return original(path)

    monkeypatch.setattr(repository.os, "scandir", guarded)
    report = repository.scan_repository(project)
    assert not report["complete"]
    assert "PermissionError" in by_path(report)["docs"]["reason"]
    assert "src/main.py" in report["selected_files"]


def test_unreadable_root_raises_instead_of_looking_empty(project, monkeypatch):
    def denied(path):
        raise PermissionError("test denied")

    monkeypatch.setattr(repository.os, "scandir", denied)
    with pytest.raises(OSError, match="Cannot list"):
        repository.scan_repository(project)


def test_disappearing_file_is_reported(project, monkeypatch):
    def gone(**kwargs):
        assert kwargs == {"follow_symlinks": False}
        raise FileNotFoundError("removed during scan")

    @contextmanager
    def entries(path):
        yield iter([SimpleNamespace(name="gone.py", stat=gone)])

    monkeypatch.setattr(repository.os, "scandir", entries)
    report = repository.scan_repository(project)
    assert not report["complete"] and report["selected_files"] == []
    assert "FileNotFoundError" in by_path(report)["gone.py"]["reason"]


def test_import_needs_no_models_or_gui_packages():
    result = subprocess.run(
        [sys.executable, "-c", "from app_core.repository import scan_repository; import sys; "
         "assert not {'torch', 'mlx.core', 'transformers', 'huggingface_hub', 'PySide6'} & sys.modules.keys()"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_notebook_is_clean_and_uses_only_shared_selection():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/repository_files.ipynb").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in cells)
    source = "\n".join("".join(cell["source"]) for cell in cells)
    assert "from app_core.repository import scan_repository" in source
    assert "REPOSITORY = root" in source
    assert "include_paths=INCLUDE_PATHS" in source and "exclude_paths=EXCLUDE_PATHS" in source
    assert not any(term in source for term in ("os.walk", "rglob(", "scandir(", "read_text(", "load_model(", "download_model("))
