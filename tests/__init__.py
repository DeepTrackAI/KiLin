"""Automated tests organized to match the repository packages."""

import os
from pathlib import Path
import tempfile

# Tests never touch the user's desktop preferences.
os.environ["QACHAT_PREFERENCES_FILE"] = str(
    Path(tempfile.mkdtemp(prefix="qachat-tests-")) / "preferences.json"
)
