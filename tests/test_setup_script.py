"""Exercise setup.sh with fake tools; never install or access the network."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.skipif(os.name == "nt", reason="The Unix wrapper needs sh.")
ROOT = Path(__file__).resolve().parents[1]


def command(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nset -eu\n" + body)
    path.chmod(0o755)


@pytest.fixture
def setup_workspace(tmp_path):
    """Make a temporary test project with fake tools, not a security sandbox."""
    project = tmp_path / "project with spaces"
    (project / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/setup.sh", project / "scripts/setup.sh")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    # Only these ordinary filesystem tools are reachable. uv/curl are fakes.
    for name in ("sh", "dirname", "mktemp", "rm", "mkdir", "cp", "chmod"):
        (fake_bin / name).symlink_to(shutil.which(name))
    log = tmp_path / "calls.txt"
    uv_stub = tmp_path / "uv-stub"
    command(uv_stub, 'printf "uv:%s:%s\\n" "$PWD" "$*" >> "$QACHAT_TEST_LOG"\n'
            'exit "${QACHAT_TEST_SYNC_EXIT:-0}"\n')
    shutil.copyfile(uv_stub, fake_bin / "uv")
    (fake_bin / "uv").chmod(0o755)
    command(fake_bin / "uname", 'if [ "$1" = "-s" ]; then echo "$QACHAT_TEST_OS"; '
            'else echo "$QACHAT_TEST_ARCH"; fi\n')
    command(project / ".venv/bin/python", 'printf "python:%s\\n" "$*" >> "$QACHAT_TEST_LOG"\n'
            'exit "${QACHAT_TEST_CHECK_EXIT:-0}"\n')
    installer = tmp_path / "fake-installer"
    command(installer, 'echo installer >> "$QACHAT_TEST_LOG"\n'
            'if [ "${QACHAT_TEST_INSTALL_EXIT:-0}" -ne 0 ]; then exit "$QACHAT_TEST_INSTALL_EXIT"; fi\n'
            'mkdir -p "$UV_INSTALL_DIR"\n'
            'cp "$QACHAT_TEST_UV_STUB" "$UV_INSTALL_DIR/uv"\n'
            'chmod +x "$UV_INSTALL_DIR/uv"\n')
    command(fake_bin / "curl", 'echo curl >> "$QACHAT_TEST_LOG"\n'
            '# Even a partial failed download must not be executed.\n'
            'cp "$QACHAT_TEST_INSTALLER" "$4"\n'
            'exit "${QACHAT_TEST_CURL_EXIT:-0}"\n')
    env = {
        **os.environ,
        "PATH": str(fake_bin),
        "UV_INSTALL_DIR": str(tmp_path / "uv install"),
        "QACHAT_TEST_LOG": str(log),
        "QACHAT_TEST_UV_STUB": str(uv_stub),
        "QACHAT_TEST_INSTALLER": str(installer),
        "QACHAT_TEST_OS": "Darwin",
        "QACHAT_TEST_ARCH": "arm64",
    }
    return project, fake_bin, log, env


def run_setup(setup_workspace):
    project, _, _, env = setup_workspace
    return subprocess.run(
        ["/bin/sh", str(project / "scripts/setup.sh")],
        cwd=project.parent, env=env, capture_output=True, text=True, timeout=10,
    )


@pytest.mark.parametrize("system,arch", [("Darwin", "arm64"), ("Linux", "x86_64"), ("Linux", "aarch64")])
def test_supported_unix_setup_uses_project_root_and_checks_backend(setup_workspace, system, arch):
    project, _, log, env = setup_workspace
    env.update(QACHAT_TEST_OS=system, QACHAT_TEST_ARCH=arch)
    result = run_setup(setup_workspace)
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines() == [
        f"uv:{project}:sync --extra mlx --extra torch", "python:scripts/check_setup.py",
    ]
    assert "Setup complete" in result.stdout


@pytest.mark.parametrize("system,arch", [("Darwin", "x86_64"), ("FreeBSD", "x86_64"), ("Linux", "i686")])
def test_unsupported_platform_stops_before_installation(setup_workspace, system, arch):
    _, _, log, env = setup_workspace
    env.update(QACHAT_TEST_OS=system, QACHAT_TEST_ARCH=arch)
    result = run_setup(setup_workspace)
    assert result.returncode != 0
    assert not log.exists()
    assert "Setup complete" not in result.stdout


def test_reuses_uv_outside_path(setup_workspace):
    _, fake_bin, log, env = setup_workspace
    (fake_bin / "uv").unlink()
    installed = Path(env["UV_INSTALL_DIR"]) / "uv"
    installed.parent.mkdir()
    shutil.copyfile(env["QACHAT_TEST_UV_STUB"], installed)
    installed.chmod(0o755)
    assert run_setup(setup_workspace).returncode == 0
    assert "curl" not in log.read_text()


def test_missing_uv_installs_before_sync(setup_workspace):
    _, fake_bin, log, _ = setup_workspace
    (fake_bin / "uv").unlink()
    result = run_setup(setup_workspace)
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines()[:2] == ["curl", "installer"]
    assert "Setup complete" in result.stdout


def test_failed_download_does_not_execute_partial_installer(setup_workspace):
    _, fake_bin, log, env = setup_workspace
    (fake_bin / "uv").unlink()
    env["QACHAT_TEST_CURL_EXIT"] = "22"
    result = run_setup(setup_workspace)
    assert result.returncode == 22
    assert log.read_text().splitlines() == ["curl"]
    assert "Setup complete" not in result.stdout


def test_installer_failure_stops_setup(setup_workspace):
    _, fake_bin, log, env = setup_workspace
    (fake_bin / "uv").unlink()
    env["QACHAT_TEST_INSTALL_EXIT"] = "9"
    result = run_setup(setup_workspace)
    assert result.returncode == 9
    assert log.read_text().splitlines() == ["curl", "installer"]


def test_sync_failure_does_not_run_backend_check(setup_workspace):
    _, _, log, env = setup_workspace
    env["QACHAT_TEST_SYNC_EXIT"] = "7"
    result = run_setup(setup_workspace)
    assert result.returncode == 7
    assert "python:" not in log.read_text()
    assert "Setup complete" not in result.stdout


def test_backend_failure_does_not_claim_setup_complete(setup_workspace):
    _, _, _, env = setup_workspace
    env["QACHAT_TEST_CHECK_EXIT"] = "3"
    result = run_setup(setup_workspace)
    assert result.returncode == 3
    assert "Setup complete" not in result.stdout
