"""
GUI module for OptiCalib / calpy
=================================

Provides the :class:`~opticalib.gui.app.CalpyGUI` main window and the
:func:`~opticalib.gui.app.launch_gui` convenience function that starts the Qt
application.

The GUI uses Qt through ``qtpy`` (PySide6 by default; set ``QT_API`` to pick
another binding) and runs the IPython session in a separate kernel process,
so long-running commands never freeze the window.

Typical usage
-------------
From Python::

    from opticalib.gui import launch_gui
    launch_gui(config_path='/path/to/configuration.yaml')

Via the CLI::

    calpy -f /path/to/experiment --gui
"""

import os as _os

# Prefer PySide6 unless the user explicitly selected another Qt binding,
# and make pyqtgraph use the same binding as qtpy (mixing bindings crashes).
_os.environ.setdefault("QT_API", "pyside6")
_PYQTGRAPH_LIBS = {"pyside6": "PySide6", "pyqt6": "PyQt6", "pyqt5": "PyQt5", "pyside2": "PySide2"}
if _os.environ["QT_API"].lower() in _PYQTGRAPH_LIBS:
    _os.environ.setdefault("PYQTGRAPH_QT_LIB", _PYQTGRAPH_LIBS[_os.environ["QT_API"].lower()])

__all__ = ["CalpyGUI", "launch_gui"]


def __getattr__(name: str):
    """
    Lazily import the Qt application objects.

    The kernel-side helpers (:mod:`opticalib.gui.kernel_side`) are imported
    inside the IPython kernel, which must not pay for (or require) a Qt
    import; the main window is therefore only loaded on first access.
    """
    if name in __all__:
        from . import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
