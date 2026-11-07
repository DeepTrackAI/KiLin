"""Isolate macOS launcher execution from GUI prompts and real installs.

The harness runs an unchanged copy of the production installation logic,
except that absolute ``osascript`` calls are redirected to an executable
which records the requested message instead of opening a macOS dialog.
The fake ``lockf`` and ``uv`` commands preserve the relevant process and
filesystem behavior without requiring model or runtime downloads.

"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory

_LAUNCHER = (
    Path(__file__).resolve().parents[2] / "setup/macos/launcher.sh"
)


def executable(path: Path, content: str) -> None:
    """Create an executable script in the isolated test environment.

    Parameters
    ----------
    path: pathlib.Path
        Path to create, including its parent directory.
    content: str
        Complete script with the requested interpreter line.

    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


class LauncherSandbox:
    """Own a fake macOS application and an isolated installation tree.

    The app preserves the production launcher's shell code. It replaces
    only the system alert executable so negative-path tests never display
    blocking dialogs or trigger user approval prompts on macOS.

    """

    def __init__(
        self, spec: str = "kilin[app]==0.1.0", wheel_sha: str = "",
    ) -> None:
        """Create the application, installer, locking, and alert shims.

        Parameters
        ----------
        spec: str
            Application installation requirement to embed.
        wheel_sha: str
            Optional checksum used for rebuilt wheels.

        """

        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.support = self.root / "Application Support" / "kilin"
        self.runtime_root = self.support / "runtime"
        self.runtime = self.runtime_root / "app"
        self.spec = spec
        self.wheel_sha = wheel_sha
        self.uv_calls = self.root / "uv-calls.txt"
        self.gui_calls = self.root / "gui-calls.txt"
        self.venv_paths = self.root / "venv-paths.txt"
        self.osascript_calls = self.root / "osascript-calls.txt"
        self.started = self.root / "install-started"
        self.release = self.root / "release-install"
        self.gui_script = self.root / "fake-gui"
        self.python_script = self.root / "fake-python"
        self.mock_bin = self.root / "bin"
        self.contents = self.root / "K\u00edL\u00edn.app" / "Contents"
        self.resources = self.contents / "Resources"
        self.launcher = self.contents / "MacOS" / "KiLin"

        executable(
            self.gui_script,
            "#!/bin/sh\n"
            'printf "gui\\n" >> "$KILIN_TEST_GUI_CALLS"\n',
        )
        executable(
            self.python_script,
            "#!/bin/sh\n"
            'if [ "${KILIN_TEST_FAIL_VALIDATION:-}" = "1" ]; then\n'
            '  exit 23\n'
            'fi\n',
        )
        executable(
            self.resources / "uv",
            "#!/bin/sh\n"
            'printf "%s\\n" "$1" >> "$KILIN_TEST_UV_CALLS"\n'
            'if [ "$1" = "venv" ]; then\n'
            '  printf "%s\\n" "$4" >> "$KILIN_TEST_VENV_PATHS"\n'
            '  mkdir -p "$4/bin"\n'
            '  cp "$KILIN_TEST_GUI_SCRIPT" "$4/bin/kilin-gui"\n'
            '  cp "$KILIN_TEST_PYTHON_SCRIPT" "$4/bin/python"\n'
            '  if [ -n "${KILIN_TEST_RELEASE:-}" ]; then\n'
            '    : > "$KILIN_TEST_STARTED"\n'
            '    while [ ! -e "$KILIN_TEST_RELEASE" ]; do\n'
            '      sleep 0.05\n'
            '    done\n'
            '  fi\n'
            'fi\n'
            'if [ "${KILIN_TEST_FAIL_UV:-}" = "$1" ]; then\n'
            '  exit 12\n'
            'fi\n',
        )
        executable(self.mock_bin / "uname", "#!/bin/sh\necho arm64\n")
        executable(
            self.mock_bin / "osascript",
            "#!/bin/sh\n"
            'printf "%s\\n" "$*" >> "$KILIN_TEST_OSASCRIPT"\n',
        )
        executable(
            self.mock_bin / "mv",
            "#!/bin/sh\n"
            'if [ "${KILIN_TEST_FAIL_ACTIVATE:-}" = "1" ]; then\n'
            '  exit 12\n'
            'fi\n'
            'exec /bin/mv "$@"\n',
        )
        executable(
            self.mock_bin / "lockf",
            "#!/bin/sh\n"
            f"exec {shlex.quote(sys.executable)} - \"$@\" <<'PY'\n"
            "import fcntl\n"
            "import sys\n"
            "import time\n"
            "fd = int(sys.argv[-1])\n"
            "timeout = int(sys.argv[sys.argv.index('-t') + 1])\n"
            "deadline = time.monotonic() + timeout\n"
            "while True:\n"
            "    try:\n"
            "        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "        break\n"
            "    except BlockingIOError:\n"
            "        if time.monotonic() >= deadline:\n"
            "            sys.exit(75)\n"
            "        time.sleep(0.02)\n"
            "PY\n",
        )
        self.environment = {
            **os.environ,
            "KILIN_APP_SUPPORT": str(self.support),
            "KILIN_TEST_UV_CALLS": str(self.uv_calls),
            "KILIN_TEST_GUI_CALLS": str(self.gui_calls),
            "KILIN_TEST_GUI_SCRIPT": str(self.gui_script),
            "KILIN_TEST_PYTHON_SCRIPT": str(self.python_script),
            "KILIN_TEST_VENV_PATHS": str(self.venv_paths),
            "KILIN_TEST_OSASCRIPT": str(self.osascript_calls),
            "KILIN_TEST_STARTED": str(self.started),
            "PATH": str(self.mock_bin) + os.pathsep
            + os.environ.get("PATH", ""),
        }
        self.set_launcher(spec, wheel_sha)

    def set_launcher(self, spec: str, wheel_sha: str = "") -> None:
        """Select a new app build while preserving the installed runtime.

        Parameters
        ----------
        spec: str
            New version or wheel installation requirement.
        wheel_sha: str
            Optional checksum supplied by the app builder.

        """

        self.spec = spec
        self.wheel_sha = wheel_sha
        template = _LAUNCHER.read_text(encoding="utf-8")
        osascript = shlex.quote(str(self.mock_bin / "osascript"))
        # Keep production error/notification behavior, but redirect its
        # absolute executable before the script is launched in a test.
        template = template.replace("/usr/bin/osascript", osascript)
        executable(
            self.launcher,
            template.replace("@KILIN_SPEC@", spec)
            .replace("@KILIN_WHEEL_SHA@", wheel_sha),
        )

    def release_runtime(self) -> Path:
        """Compute the stable release directory selected by this app.

        Returns
        -------
        pathlib.Path
            Destination for the currently embedded installation source.

        """

        expanded = self.spec.replace(
            "@RESOURCES@", str(self.resources)
        )
        build = expanded + (
            " " + self.wheel_sha if self.wheel_sha else ""
        )
        digest = hashlib.sha256(build.encode("utf-8")).hexdigest()
        return self.runtime_root / "releases" / digest

    def run(
        self, environment: dict[str, str] | None = None,
        timeout: float = 15,
    ) -> subprocess.CompletedProcess[str]:
        """Execute the launcher and return its result without modal UI.

        Parameters
        ----------
        environment: dict[str, str] or None
            Environment variables to override for one invocation.
        timeout: float
            Maximum wall-clock time for the simulated launcher.

        Returns
        -------
        subprocess.CompletedProcess[str]
            Standard output, standard error, and launcher exit code.

        """

        configured = self.environment.copy()
        configured.update(environment or {})
        return subprocess.run(
            [str(self.launcher)],
            env=configured,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    def close(self) -> None:
        """Release all temporary application and installation files."""

        self.temporary.cleanup()
