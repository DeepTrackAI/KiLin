"""Provide the optional KíLín graphical-interface package.

The package initializer has no GUI-startup side effects. The ``app`` module
launches the Qt interface, ``window`` hosts a local web page, and ``engine``
runs model and repository operations in a separate process. Web assets live
in the sibling ``web`` directory. Importing ``kilin.gui`` does not start Qt.

Key Features
------------

- **Separate entry point**

    Keep graphical startup separate from the package-level model and retrieval
    API.

- **Process separation**

    Separate the Qt/web presentation layer from blocking backend operations.

Module Structure
----------------

This module is an entry point or namespace marker and does not define a
separate exported API. The related implementation modules are described above.

Examples
--------
>>> import kilin.gui

Launch the interface explicitly with ``python -m kilin.gui`` in a suitable
environment. This package does not re-export the Qt classes.

"""

__all__ = []
