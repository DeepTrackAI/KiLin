"""Start the optional GUI when invoked as ``python -m kilin.gui``.

This entry point delegates to ``kilin.gui.app.main`` and exits with its return
code. It does not perform the dependency synchronization offered by the
separate command-line helper in ``kilin.cli``.

Key Features
------------

- **Module entry point**

    Delegate interactive application startup to the GUI application module.

Module Structure
----------------

This module is an entry point or namespace marker and does not define a
separate exported API. The related implementation modules are described above.

Examples
--------
Run ``python -m kilin.gui`` from an environment with the GUI dependencies
installed. The ``main`` call is protected by a ``__name__`` guard;
importing this entry-point module does not start the event loop.

"""

from kilin.gui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
