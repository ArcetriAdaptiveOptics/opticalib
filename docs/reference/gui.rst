GUI (``calpy --gui``)
=====================

.. automodule:: opticalib.gui
   :no-members:

The Qt front end started by ``calpy --gui``.  It is **not** required to use the
library programmatically; see :ref:`user-guide-calpy-gui` for a tour of the
interface.

.. warning::
   The GUI uses Qt through ``qtpy`` (PySide6 by default) and needs a display.
   On headless machines (CI, Read the Docs) set ``QT_QPA_PLATFORM=offscreen``;
   this documentation build mocks Qt entirely, so the signatures below are
   rendered from source rather than from a live import.

Architecture
------------

* The IPython session runs in a separate **kernel process**
  (:class:`~opticalib.gui.kernel.KernelBridge`); the window never freezes,
  commands can be interrupted and the kernel restarted.
* GUI actions are Python code queued as :class:`~opticalib.gui.kernel.Task`
  objects and echoed in the console, so every action is reproducible.
* Inside the kernel, :mod:`opticalib.gui.kernel_side` (bound as ``_gui``) and
  the :mod:`~opticalib.gui.kernel_backend` matplotlib backend publish figures
  and arrays to the GUI.

Launching the application
-------------------------

.. autofunction:: opticalib.gui.app.launch_gui

.. autoclass:: opticalib.gui.app.CalpyGUI
   :members:

Kernel
------

.. automodule:: opticalib.gui.kernel
   :members: KernelBridge, Task, CalpyConsole, parse_progress, strip_ansi

Inside the kernel
-----------------

These modules run in the kernel process and do not import Qt.

.. automodule:: opticalib.gui.kernel_side
   :members: view, publish_figures, workspace_items, workspace, folders, mark_baseline, install

.. automodule:: opticalib.gui.kernel_backend
   :no-members:

Panels
------

.. autoclass:: opticalib.gui.widgets.plot_viewer.PlotViewer
   :members:

.. autofunction:: opticalib.gui.widgets.plot_viewer.prepare_array

.. autoclass:: opticalib.gui.widgets.device_panel.DevicePanel
   :members:

.. autoclass:: opticalib.gui.widgets.device_panel.DeviceCard
   :members:

.. autoclass:: opticalib.gui.widgets.connect_dialog.ConnectDialog
   :members:

.. autoclass:: opticalib.gui.widgets.workspace.WorkspaceView
   :members:

.. autoclass:: opticalib.gui.widgets.data_browser.DataBrowser
   :members:

.. autoclass:: opticalib.gui.widgets.config_editor.ConfigEditorDialog
   :members:

Device registry
---------------

How the entries of the ``DEVICES`` section are mapped to device classes.

.. automodule:: opticalib.gui.widgets.device_registry
   :members: DeviceClass, DeviceEntry, resolve_entry, list_entries, build_command, set_entry_class, locate_entry

Activity and theme
------------------

.. autoclass:: opticalib.gui.activity.ActivityCenter
   :members:

.. autoclass:: opticalib.gui.activity.StartupOverlay
   :members:

.. autoclass:: opticalib.gui.activity.LocalJob
   :members:

.. automodule:: opticalib.gui.theme
   :members: ThemeManager, theme

Procedure windows
-----------------

.. automodule:: opticalib.gui.procedures
   :no-members:

Each procedure is a sequence of steps; running a step executes its code,
visibly, in the console.  Steps that move hardware ask for confirmation, and
the results of a step (e.g. a tracking number) pre-fill the next ones.

.. autoclass:: opticalib.gui.procedures.dm_calibration.DeformableMirrorCalibrationWindow

.. autoclass:: opticalib.gui.procedures.timeseries.TimeseriesWindow

.. autoclass:: opticalib.gui.procedures.stitching.StitchingWindow

.. autoclass:: opticalib.gui.procedures.alignment.AlignmentWindow

.. autoclass:: opticalib.gui.procedures.phasing.SegmentsPhasingWindow

Writing a procedure window
~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass :class:`~opticalib.gui.procedures.base.ProcedureWindow` and return
:class:`~opticalib.gui.procedures.base.Step` objects from ``steps()``; the
parameters come from :mod:`opticalib.gui.procedures.params`.

.. autoclass:: opticalib.gui.procedures.base.ProcedureWindow
   :members: steps, run_step, set_state, show_warnings, run_checks

.. autoclass:: opticalib.gui.procedures.base.Step
   :members:

.. autoclass:: opticalib.gui.procedures.base.ProcedureContext
   :members:

.. autofunction:: opticalib.gui.procedures.base.kwargs_code

.. automodule:: opticalib.gui.procedures.params
   :members: Param, IntParam, FloatParam, BoolParam, ChoiceParam, TextParam, ExprParam, VarParam, DeviceParam, DeviceListParam, TnParam, ParamError

.. autoclass:: opticalib.gui.plugins.PluginPanel
   :members:
