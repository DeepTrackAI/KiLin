"""Automated tests organized to match the repository packages."""

import os
from pathlib import Path
import shutil
import tempfile

# Tests read and write a copy, never the real desktop defaults.
_DEFAULTS = Path(tempfile.mkdtemp(prefix="qachat-tests-")) / "defaults.json"
shutil.copyfile(
    Path(__file__).resolve().parents[1] / "qachat" / "gui" / "defaults.json",
    _DEFAULTS,
)
os.environ["QACHAT_DEFAULTS_FILE"] = str(_DEFAULTS)
