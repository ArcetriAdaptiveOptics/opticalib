"""
GUI module for opticalib / calpy
=================================

Provides the :class:`CalpyGUI` main window and the :func:`launch_gui`
convenience function that starts the Qt application.

Typical usage
-------------
From Python::

    from opticalib.gui import launch_gui
    launch_gui(config_path='/path/to/configuration.yaml')

Via the CLI::

    calpy -f /path/to/experiment --gui
"""

from .app import CalpyGUI, launch_gui

__all__ = ["CalpyGUI", "launch_gui"]
