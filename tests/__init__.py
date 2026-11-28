"""KíLín test suite.

This package contains tests for the KíLín framework, organized to mirror
the structure of the kilin package.

Tests are designed to run independently, without modifying user data or
requiring external services unless explicitly stated.

"""
# Isolate both pytest and unittest before any test imports KíLín. Fresh
# interpreters launched by tests inherit the empty directory; plugin-specific
# tests explicitly select their own temporary directory.
import os
from pathlib import Path
from tempfile import TemporaryDirectory

_plugin_directory = TemporaryDirectory(prefix="kilin-test-plugins-")
os.environ["KILIN_PLUGINS_DIR"] = str(Path(_plugin_directory.name).resolve())
