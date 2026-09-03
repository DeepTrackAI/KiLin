"""Small local stand-ins test model management without network or real weights."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qachat import models


@pytest.fixture
def tiny_model(monkeypatch, tmp_path):
    files = {"config.json": b"{}", "model.safetensors": b"small fake weights"}
    specs = []
    for name, data in files.items():
        is_weight = name.endswith(".safetensors")
        digest = hashlib.sha256(data) if is_weight else hashlib.sha1(
            f"blob {len(data)}\0".encode() + data
        )
        specs.append({
            "name": name, "size": len(data), "hash": digest.hexdigest(),
            "hash_type": "sha256" if is_weight else "git_sha1",
        })
    monkeypatch.setattr(models, "CATALOG", {
        "tiny": {"backend": "mlx", "repo_id": "example/tiny",
                 "revision": "a" * 40, "license": "apache-2.0", "files": specs},
    })
    monkeypatch.setenv("QACHAT_MODEL_HOME", str(tmp_path / "models"))
    return files


def write_model(files, directory=None):
    folder = models.model_path("tiny", directory)
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (folder / name).write_bytes(data)
    return folder


def test_catalog_has_fixed_safe_files_and_both_backends():
    assert {info["backend"] for info in models.CATALOG.values()} == {"mlx", "torch"}
    for name, info in models.CATALOG.items():
        assert len(info["revision"]) == 40
        int(info["revision"], 16)
        assert info["license"] == "apache-2.0"
        filenames = [file["name"] for file in info["files"]]
        assert len(filenames) == len(set(filenames))
        assert {"config.json", "tokenizer.json", "tokenizer_config.json"} <= set(filenames)
        assert any(filename.endswith(".safetensors") for filename in filenames)
        for file in info["files"]:
            assert Path(file["name"]).name == file["name"]
            assert file["size"] > 0
            assert len(file["hash"]) == (64 if file["hash_type"] == "sha256" else 40)


def test_default_storage_is_repo_models_not_current_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("QACHAT_MODEL_HOME", raising=False)
    project = tmp_path / "QAChat"
    monkeypatch.setattr(models, "PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)
    assert models.model_home() == project / "models"
    assert not (project / "models").exists()


def test_environment_selects_storage(tiny_model, tmp_path):
    assert models.model_home() == tmp_path / "models"


@pytest.mark.parametrize("through_environment", [False, True])
def test_relative_storage_is_anchored_to_project(monkeypatch, tmp_path, through_environment):
    project = tmp_path / "QAChat"
    monkeypatch.setattr(models, "PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QACHAT_MODEL_HOME", raising=False)
    if through_environment:
        monkeypatch.setenv("QACHAT_MODEL_HOME", "my models")
        actual = models.model_home()
    else:
        actual = models.model_home("my models")
    assert actual == project / "my models"


def test_storage_expands_user_home():
    assert models.model_home("~/qachat-custom-models") == Path("~/qachat-custom-models").expanduser().resolve()


def test_explicit_storage_overrides_environment(tiny_model, tmp_path):
    assert models.model_home(tmp_path / "chosen") == tmp_path / "chosen"


def test_existing_models_in_custom_folder_are_reused(tiny_model, tmp_path, monkeypatch):
    custom = tmp_path / "my existing models"
    folder = write_model(tiny_model, directory=custom)
    start = Mock()
    monkeypatch.setattr(models.subprocess, "Popen", start)
    assert models.check_model("tiny", directory=custom, verify=True)["status"] == "verified"
    assert models.list_models(directory=custom)[0]["path"] == str(folder)
    assert models.download_model("tiny", directory=custom) == folder
    start.assert_not_called()
    assert not models.model_home().exists()


@pytest.mark.parametrize("name", ["missing", "../../elsewhere", "/tmp/wrong"])
def test_unknown_model_is_rejected_before_creating_files(tiny_model, name):
    with pytest.raises(ValueError, match="Unknown model"):
        models.check_model(name)
    assert not models.model_home().exists()


def test_model_path_rejects_escaping_symlink(tiny_model, tmp_path):
    home = models.model_home()
    home.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    try:
        (home / "tiny").symlink_to(elsewhere, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this machine.")
    with pytest.raises(ValueError, match="inside"):
        models.model_path("tiny")


def test_missing_check_is_read_only_and_does_not_start_process(tiny_model, monkeypatch):
    start = Mock()
    monkeypatch.setattr(models.subprocess, "Popen", start)
    report = models.check_model("tiny")
    assert report["status"] == "missing"
    assert set(report["missing_files"]) == set(tiny_model)
    assert not models.model_home().exists()
    start.assert_not_called()


def test_partial_directory_is_not_downloaded(tiny_model):
    folder = write_model({"config.json": b"{}"})
    (folder / "model.safetensors.incomplete").write_bytes(b"partial")
    report = models.check_model("tiny")
    assert report["status"] == "incomplete"
    assert report["missing_files"] == ["model.safetensors"]


def test_empty_directory_is_not_downloaded(tiny_model):
    models.model_path("tiny").mkdir(parents=True)
    assert models.check_model("tiny")["status"] == "incomplete"


def test_wrong_size_is_reported(tiny_model):
    write_model({**tiny_model, "model.safetensors": b"short"})
    report = models.check_model("tiny")
    assert report["status"] == "incomplete"
    assert report["invalid_files"] == ["model.safetensors"]


def test_complete_files_are_downloaded_and_hashes_can_be_verified(tiny_model):
    write_model(tiny_model)
    assert models.check_model("tiny")["status"] == "downloaded"
    assert models.check_model("tiny", verify=True)["status"] == "verified"


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors"])
def test_full_check_catches_same_size_corruption(tiny_model, filename):
    corrupted = {**tiny_model, filename: b"x" * len(tiny_model[filename])}
    write_model(corrupted)
    assert models.check_model("tiny")["status"] == "downloaded"  # Size check only.
    report = models.check_model("tiny", verify=True)
    assert report["status"] == "corrupt"
    assert report["invalid_files"] == [filename]


def test_list_reports_local_status_without_network(tiny_model):
    assert models.list_models()[0]["status"] == "missing"
    write_model(tiny_model)
    assert models.list_models()[0]["status"] == "downloaded"


def test_package_versions_reports_missing_without_gpu_import(tiny_model, monkeypatch):
    def installed(name):
        if name == "mlx":
            raise models.PackageNotFoundError(name)
        return "1.2.3"
    monkeypatch.setattr(models, "version", installed)
    versions = models.package_versions("tiny")
    assert versions["mlx"] is None
    assert versions["mlx-lm"] == "1.2.3"


def test_complete_download_reuses_files_without_a_child(tiny_model, monkeypatch):
    folder = write_model(tiny_model)
    start = Mock()
    monkeypatch.setattr(models.subprocess, "Popen", start)
    assert models.download_model("tiny") == folder
    start.assert_not_called()


def test_fetch_downloads_pinned_files_and_checks_content(tiny_model, monkeypatch):
    def fetch(**kwargs):
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.write_bytes(tiny_model[kwargs["filename"]])
        return str(destination)
    download = Mock(side_effect=fetch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    models._fetch_files("tiny", models.model_home())
    assert models.check_model("tiny", verify=True)["status"] == "verified"
    assert download.call_count == len(tiny_model)
    for call in download.call_args_list:
        assert call.kwargs["revision"] == "a" * 40
        assert call.kwargs["repo_id"] == "example/tiny"
        assert call.kwargs["token"] is False
        assert call.kwargs["endpoint"] == "https://huggingface.co"
        assert call.kwargs["force_download"] is False
    models._fetch_files("tiny", models.model_home())
    assert download.call_count == len(tiny_model)  # A second run needs no network.


def test_fetch_only_repairs_the_invalid_file(tiny_model, monkeypatch):
    folder = write_model({**tiny_model, "config.json": b"xx"})
    def fetch(**kwargs):
        (folder / kwargs["filename"]).write_bytes(tiny_model[kwargs["filename"]])
    download = Mock(side_effect=fetch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    models._fetch_files("tiny", models.model_home())
    assert download.call_count == 1
    assert download.call_args.kwargs["filename"] == "config.json"
    assert download.call_args.kwargs["force_download"] is True


def test_fetch_rejects_bad_download(tiny_model, monkeypatch):
    def fetch(**kwargs):
        (Path(kwargs["local_dir"]) / kwargs["filename"]).write_bytes(b"wrong")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fetch))
    with pytest.raises(RuntimeError, match="File check failed"):
        models._fetch_files("tiny", models.model_home())


def test_network_error_does_not_mark_model_complete(tiny_model, monkeypatch):
    download = Mock(side_effect=OSError("offline"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    with pytest.raises(OSError, match="offline"):
        models._fetch_files("tiny", models.model_home())
    assert models.check_model("tiny")["status"] == "incomplete"


def fake_child(lines=()):
    child = Mock()
    child.stdout = Mock()
    child.stdout.__iter__ = Mock(return_value=iter(lines))
    child.wait.return_value = 0
    child.poll.return_value = 0
    return child


def test_download_child_uses_same_python_and_private_cache(tiny_model, monkeypatch):
    child = fake_child(["Downloading model\n"])
    def start(*args, **kwargs):
        write_model(tiny_model)
        return child
    start = Mock(side_effect=start)
    monkeypatch.setattr(models.subprocess, "Popen", start)
    assert models.download_model("tiny") == models.model_path("tiny")
    args, kwargs = start.call_args
    assert args[0][:5] == [sys.executable, "-u", "-m", "qachat.models", "_fetch"]
    assert kwargs["env"]["HF_HOME"] == str(models.model_home() / ".huggingface")
    assert kwargs["env"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    child.stdout.close.assert_called_once()


def test_failed_child_does_not_claim_success(tiny_model, monkeypatch):
    child = fake_child()
    child.wait.return_value = 1
    monkeypatch.setattr(models.subprocess, "Popen", Mock(return_value=child))
    with pytest.raises(RuntimeError, match="did not finish"):
        models.download_model("tiny")


def test_child_success_without_files_still_fails(tiny_model, monkeypatch):
    monkeypatch.setattr(models.subprocess, "Popen", Mock(return_value=fake_child()))
    with pytest.raises(RuntimeError, match="not complete"):
        models.download_model("tiny")


@pytest.mark.parametrize("needs_kill", [False, True])
def test_stop_terminates_child_and_keeps_partial_files(tiny_model, monkeypatch, needs_kill):
    write_model({"config.json": b"{}"})
    child = fake_child()
    child.stdout.__iter__.side_effect = KeyboardInterrupt
    child.poll.return_value = None
    if needs_kill:
        child.wait.side_effect = [subprocess.TimeoutExpired("download", 5), 0]
    monkeypatch.setattr(models.subprocess, "Popen", Mock(return_value=child))
    with pytest.raises(KeyboardInterrupt):
        models.download_model("tiny")
    child.terminate.assert_called_once()
    assert child.kill.called == needs_kill
    child.stdout.close.assert_called_once()
    assert models.check_model("tiny")["status"] == "incomplete"


def test_cli_list_does_not_download(tiny_model, monkeypatch, capsys):
    download = Mock()
    monkeypatch.setattr(models, "download_model", download)
    assert models.main([]) == 0
    assert "missing" in capsys.readouterr().out
    download.assert_not_called()


def test_cli_check_reports_missing_as_failure(tiny_model, capsys):
    assert models.main(["check", "tiny"]) == 1
    assert "status: missing" in capsys.readouterr().out


def test_cli_check_uses_chosen_directory(tiny_model, tmp_path, capsys):
    custom = tmp_path / "my existing models"
    write_model(tiny_model, directory=custom)
    assert models.main(["check", "tiny", "--directory", str(custom), "--verify"]) == 0
    output = capsys.readouterr().out
    assert str(custom) in output
    assert "status: verified" in output


def test_download_passes_custom_directory_to_child(tiny_model, tmp_path, monkeypatch):
    custom = tmp_path / "chosen models"
    child = fake_child()
    def start(*args, **kwargs):
        write_model(tiny_model, directory=custom)
        return child
    start = Mock(side_effect=start)
    monkeypatch.setattr(models.subprocess, "Popen", start)
    assert models.download_model("tiny", directory=custom) == models.model_path("tiny", custom)
    command = start.call_args.args[0]
    assert command[-2:] == ["--directory", str(custom)]
    assert start.call_args.kwargs["env"]["HF_HOME"] == str(custom / ".huggingface")


def test_cli_download_reports_success(tiny_model, monkeypatch, capsys):
    def download(*args):
        return write_model(tiny_model)
    monkeypatch.setattr(models, "download_model", download)
    assert models.main(["download", "tiny"]) == 0
    assert "status: downloaded" in capsys.readouterr().out


def test_cli_cancel_returns_130(tiny_model, monkeypatch, capsys):
    monkeypatch.setattr(models, "download_model", Mock(side_effect=KeyboardInterrupt))
    assert models.main(["download", "tiny"]) == 130
    assert "stopped" in capsys.readouterr().err


def test_cli_bad_model_is_clear(tiny_model, capsys):
    assert models.main(["check", "not-a-model"]) == 1
    assert "Unknown model" in capsys.readouterr().err


def test_import_does_not_load_backend_libraries():
    result = subprocess.run(
        [sys.executable, "-c", "from qachat import models; import sys; "
         "assert not {'mlx.core', 'torch', 'transformers', 'huggingface_hub'} & sys.modules.keys()"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_notebook_is_clean_and_downloads_are_opt_in():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/models.ipynb").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in cells)
    source = "\n".join("".join(cell["source"]) for cell in cells)
    assert "from qachat.models import" in source
    assert "DOWNLOAD = False" in source
    assert "if DOWNLOAD:" in source
    assert "MODEL_DIRECTORY = None" in source
    assert "model_home(MODEL_DIRECTORY)" in source
    assert "list_models(directory=MODEL_DIRECTORY)" in source
    assert "download_model(MODEL, directory=MODEL_DIRECTORY)" in source
    assert source.count("check_model(MODEL, directory=MODEL_DIRECTORY)") == 2
