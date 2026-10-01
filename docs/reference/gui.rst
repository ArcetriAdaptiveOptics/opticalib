GUI (``calpy``)
===============

.. automodule:: opticalib.gui
   :no-members:

The optional Qt front end that ships with the ``calpy`` entry point.  It is an
extra dependency group -- see :doc:`/installation` -- and is **not** required to
use the library programmatically.

.. warning::
   Importing this module requires ``PySide6`` and a usable Qt platform plugin.
   On headless machines (CI, Read the Docs) set ``QT_QPA_PLATFORM=offscreen``;
   this documentation build mocks Qt entirely, so the signatures below are
   rendered from source rather than from a live import.

Launching the application
-------------------------

.. autofunction:: opticalib.gui.app.launch_gui

.. autoclass:: opticalib.gui.app.CalpyGUI
   :members:

Task windows
------------

Each window implements one bench task.  They share the panel widgets below and
are registered with :class:`~opticalib.gui.app.CalpyGUI` as plugins.

.. autoclass:: opticalib.gui.app.DeformableMirrorCalibrationWindow
   :members:

.. autoclass:: opticalib.gui.app.AlignmentWindow
   :members:

.. autoclass:: opticalib.gui.app.TimeseriesWindow
   :members:

.. autoclass:: opticalib.gui.app.StitchingWindow
   :members:

.. autoclass:: opticalib.gui.app.SegmentsPhasingWindow
   :members:

Shared panels and plugin API
----------------------------

.. autoclass:: opticalib.gui.app.PluginWindowBase
   :members:

.. autoclass:: opticalib.gui.app.PluginPanel
   :members:

.. autoclass:: opticalib.gui.app.DevicePanel
   :members:

.. autoclass:: opticalib.gui.app.PlotPanel
   :members:

.. autoclass:: opticalib.gui.app.ConfigViewDialog
   :members:
