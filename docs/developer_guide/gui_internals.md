(dev-guide-gui-internals)=

# CalpyGUI internals

This page explains how the `calpy --gui` application (`opticalib.gui`, *CalpyGUI*)
is built: which process does what, how a click becomes code in the console, how
results come back, and where each piece lives. It describes the code as of
opticalib 2.2.0. For how to *use* the GUI see {ref}`user-guide-calpy-gui`; for
how to *extend* it see {doc}`extending_the_gui`.

## 1. The big picture

The application is made of **two processes**:

```text
┌──────────────────────────── GUI process (Qt 6, PySide6 via qtpy) ───────────────────────────┐
│                                                                                              │
│  CalpyGUI (app.py, QMainWindow)                                                              │
│   ├─ PlotViewer (center)           ├─ DevicePanel ── device_registry / ConnectDialog         │
│   ├─ WorkspaceView                 ├─ PluginPanel ── ProcedureWindow subclasses              │
│   ├─ DataBrowser                   ├─ ConfigEditorDialog                                     │
│   ├─ ActivityCenter / StartupOverlay                                                         │
│   └─ KernelBridge (kernel.py) ── CalpyConsole (qtconsole RichJupyterWidget)                  │
│              │  run(code)   → visible execute request (echoed in the console)                │
│              │  query(expr) → silent execute request + user_expressions                     │
└──────────────┼───────────────────────────────────────────────────────────────────────────────┘
               │  Jupyter messaging protocol over local IPC sockets (ZeroMQ)
┌──────────────┼────────────── Kernel process (ipykernel, same Python interpreter) ────────────┐
│  IPython user namespace: initCalpy.py aliases (np, xp, opt, osu, ifp, ifm, ...), devices,   │
│  data, plus `_gui` = opticalib.gui.kernel_side (no Qt here!)                                 │
│   ├─ kernel_side.workspace() / folders() / backend() / describe()  → JSON for the GUI        │
│   ├─ kernel_side.view(array)   → .npz in a temp dir + display_data(IMAGE_MIME)               │
│   └─ kernel_backend (matplotlib backend) → PNG + display_data(FIGURE_MIME)                   │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

Three design rules hold the whole thing together. Keep them when you add
features:

1. **Every action is visible Python.** Connecting a device, running a
   procedure step or switching experiment generates Python source that is
   *echoed and executed in the console*, as if typed. What the GUI does can
   always be copied into a script. The only exception is **silent queries**,
   which only read state (the workspace listing, the folders, a check).
2. **The kernel is the only owner of devices and data.** The GUI process never
   holds a device object or a data array produced by the kernel. It sees JSON
   summaries (`workspace()`), PNG images (figures) and temporary `.npz` files
   (`_gui.view`). This is why a crashing vendor SDK can kill the kernel but
   never the window.
3. **Nothing blocks the GUI thread.** Kernel work is a `Task`; work that the GUI
   process does itself (reading a FITS file for preview) is a `LocalJob` on a
   background thread. Both can be shown in the *Activity* panel (unless quiet).

The out-of-process design replaced an in-process kernel
(`QtInProcessKernelManager`). That kernel ran user code on the Qt thread: the
first build of a simulated Alpao-88 froze the window for about 54 s, and an
interrupt was impossible.

### Qt binding

The GUI uses Qt **6** through `qtpy`, PySide6 by default.
`opticalib/gui/__init__.py` sets `QT_API=pyside6` (unless already set) and pins
`PYQTGRAPH_QT_LIB` to the same binding, because mixing bindings crashes.
`app.py` refuses Qt 5 (`qtpy.QT5 is True` raises `ImportError`), because
pyqtgraph 0.14 segfaults on PyQt5. **Always import `opticalib.gui` before
anything that imports `qtpy`**, or qtpy may pick another installed binding.

## 2. File map

| File | Lines | Process | Role |
|---|---:|---|---|
| `gui/__init__.py` | 65 | both | Selects the Qt binding; lazily exposes `CalpyGUI`, `launch_gui`, `Procedure`, `PROCEDURES` (lazy so that the kernel can import `opticalib.gui.kernel_side` without Qt) |
| `gui/app.py` | 1425 | GUI | `CalpyGUI` main window, docks, menus, status bar (`KernelStatus`, `RamBar`, `BackendButton`), experiment switching, `launch_gui()` |
| `gui/kernel.py` | 1189 | GUI | `KernelBridge` (kernel lifecycle, `run`/`query`, watchdog), `Task`, `CalpyConsole`, progress parsing |
| `gui/kernel_side.py` | 548 | **kernel** | `_gui` helpers: figure publication, `view`, `workspace`, `folders`, `backend`, `describe`, `use_experiment`, `install` |
| `gui/kernel_backend.py` | 63 | **kernel** | Matplotlib backend `module://opticalib.gui.kernel_backend` |
| `gui/activity.py` | 1069 | GUI | `ActivityCenter` (floating activity cards and toasts), `StartupOverlay`, `LocalJob`, `TracebackDialog` |
| `gui/theme.py` | 598 | GUI | Light/dark color tokens, QSS style sheet, palette, `theme()` singleton, Material Design icons |
| `gui/plugins.py` | 302 | GUI | `Procedure` dataclass, `PROCEDURES` list, `PluginPanel` (the *Procedures* dock) |
| `gui/procedures/base.py` | 732 | GUI | `ProcedureContext`, `Step`, `ProcedureWindow`, `kwargs_code` |
| `gui/procedures/params.py` | 684 | GUI | Step parameters: `IntParam`, `FloatParam`, `BoolParam`, `ChoiceParam`, `TextParam`, `ExprParam`, `VarParam`, `DeviceParam`, `DeviceListParam`, `TnParam`, `TnListParam` |
| `gui/procedures/dm_calibration.py` | 381 | GUI | *Deformable Mirror Calibration* window |
| `gui/procedures/timeseries.py` | 123 | GUI | *Timeseries* window |
| `gui/procedures/stitching.py` | 116 | GUI | *Stitching* window |
| `gui/procedures/alignment.py` | 129 | GUI | *Alignment* window |
| `gui/procedures/phasing.py` | 190 | GUI | *Segments Phasing* window |
| `gui/widgets/device_registry.py` | 1164 | both (no Qt) | Maps `DEVICES` entries to device classes; `DEVICE.CLASSES` custom classes; line-level YAML edits |
| `gui/widgets/device_panel.py` | 921 | GUI | `DevicePanel`, `DeviceCard`, `DeviceInfo`, `SIMULATED_DEVICES`, `quick_actions` |
| `gui/widgets/connect_dialog.py` | 490 | GUI | `ConnectDialog`: pick the class or variable, define a custom class, preview and edit the command |
| `gui/widgets/workspace.py` | 218 | GUI | `WorkspaceView` (kernel variables) |
| `gui/widgets/data_browser.py` | 314 | GUI | `DataBrowser` (tracking-number tree), `load_array_file` |
| `gui/widgets/plot_viewer.py` | 839 | GUI | `PlotViewer` (figures + pyqtgraph data viewer, thumbnails, pop-out, export) |
| `gui/widgets/config_editor.py` | 556 | GUI | `ConfigEditorDialog`, `CodeEditor`, YAML highlighter, `validate_config_text` |
| `gui/widgets/common.py` | 300 | GUI | `ElidedLabel`, `DockTitleBar`, `CollapsibleSection` (foldable group with a summary line) |

Related code outside `gui/`:

- `setup_calpy.py`: the `calpy` / `OptiCalib` entry points. `_launch_gui()`
  checks for a display and for the GUI extra before importing
  `opticalib.gui`. It also creates the desktop launcher
  (`--install-launcher`).
- `opticalib/__init_script__/initCalpy.py`: the namespace of the console (`np`,
  `xp`, `opt`, `folders`, `osu`, `zern`, `az`, `sim`, `ifp`, `ifm`, `join`,
  `from matplotlib.pyplot import *`). It is shared with the plain `calpy`
  IPython shell.
- `opticalib/core/root.py: set_configuration_file()`: reloads opticalib's
  runtime modules so that every cached path follows the new configuration. It
  keeps the **same** `folders` object and updates its attributes.
- `opticalib/core/config.py`: `DeviceSection` makes `DEVICES` name lookup
  case-insensitive (an exact match wins, an ambiguous match raises
  `KeyError`). opticalib still imports when `SYSTEM.ALIGNMENT` or `PHASING` is
  missing: `PHASING` is read through a `LazySection` (`procedures/phasing.py`),
  and `procedures/alignment.py` serves its `_sc` lazily through a module
  `__getattr__`.

## 3. Startup, step by step

1. `calpy [-f exp] --gui` → `setup_calpy._launch_gui()`. This checks for a
   display (it prints an explanation over SSH without `-X`), then
   `from opticalib.gui import launch_gui`.
2. `launch_gui(config_path, window_class=CalpyGUI)` creates the
   `QApplication` and sets the name, desktop file and icon from the window
   class's attributes. It applies the theme, then builds `window_class(config_path)`.
3. `CalpyGUI.__init__`:
   - opens `QSettings("ArcetriAdaptiveOptics", SETTINGS_APP)` and resolves
     the configuration (default: `opticalib.core.root.CONFIGURATION_FILE`);
   - creates a private temporary folder (`calpygui-*`), used for the kernel
     sockets and the `_gui.view` files;
   - creates the `KernelBridge(config, initCalpy.py, tmp,
     extra_init_files=INIT_SCRIPTS)`; the console widget exists, the kernel
     does not yet;
   - builds the docks, menus and status bar, the `ActivityCenter` and the
     shared `ProcedureContext`, and connects the bridge signals;
   - saves the default dock layout (for *Reset layout*), restores the saved
     one, and shows the `StartupOverlay`;
   - `QTimer.singleShot(0, bridge.start)`: the kernel starts once the event
     loop runs.
4. `KernelBridge.start()`:
   - builds a `QtKernelManager` with a fixed `KernelSpec`: **the same
     interpreter as the GUI**, whatever kernelspecs the user has;
   - uses `ipc` transport in the temp folder on POSIX (only the local user
     can reach it), falling back to TCP if the path is too long;
   - starts the kernel with `kernel_env()`: the GUI environment, plus
     the variables of the conda environment (`conda_env_variables`; a
     desktop launcher does not "activate" conda), plus
     `AOCONF=<config>`, `MPLBACKEND=module://opticalib.gui.kernel_backend`
     and `CALPY_GUI_TMP=<tmp>`.
5. Bootstrap (`_begin_bootstrap` → `_on_kernel_info` → `_on_install_done` →
   `_run_init_files` → `_finish_bootstrap`):
   - `kernel_info` is sent, and re-sent every second (requests sent before
     the kernel listens are lost), for up to `STARTUP_TIMEOUT` = 120 s;
   - the bridge waits until the console has finished *its own* startup
     (`console._starting`). qtconsole calls `reset(clear=True)` on its first
     `kernel_info` reply, which wipes its request table;
   - a silent query imports `opticalib.gui.kernel_side` and calls `install()`;
     this is step *"Importing opticalib"* of the overlay;
   - `initCalpy.py` and then every `INIT_SCRIPTS` file run as **quiet
     bootstrap tasks** with `get_ipython().safe_execfile(path,
     user_ns, raise_exceptions=True)`, one after the other. At the first error
     the remaining scripts are **skipped**, and the overlay shows the error
     (step *"Loading the calpy environment"*) with a *Continue anyway* button.
     The bootstrap still completes, so the GUI comes up without the skipped
     scripts;
   - `kernel_side.mark_baseline()` records every name defined so far, so
     the workspace panel hides them;
   - `ready` is emitted. The window refreshes the data folders and the
     workspace, and the queued GUI tasks start.

The same bootstrap runs after every restart (manual, or automatic after a
crash).

## 4. The kernel bridge (`kernel.py`)

`KernelBridge` is the **only** object that talks to the kernel. Everything else
goes through `bridge.run`, `bridge.query`, `bridge.interrupt`,
`bridge.restart`, or the `ProcedureContext` that wraps them.

### 4.1 `run()` versus `query()`

|  | `run(code, title, on_done, on_error, quiet=False) -> Task` | `query(expressions, callback, code="")` |
|---|---|---|
| Shown in the console | yes, echoed like typed input | no |
| Stored in history, counts as `In[n]` | yes | no |
| Queued | yes, one visible execution at a time, FIFO | no, sent immediately (the kernel still executes them in order) |
| Shown in the Activity panel | yes, unless `quiet` | no |
| Result | `Task` states and callbacks; errors in `task.error` | `callback({key: value or Exception})` |
| Use it for | anything that **changes state** or that the user should see | **reading** state for the GUI |

Never use `query` to move hardware or to write data: that would break rule 1.

### 4.2 Task lifecycle

`Task` (a `QObject`) goes `queued → running → done | error | cancelled`:

- `changed` is emitted on any change of state, progress or detail; `finished`
  is emitted once.
- `on_done(task)` / `on_error(task)` are called through `_safe_call`: an
  exception in a callback is printed, never propagated into Qt.
- **Progress**: every `stream` output of the running request goes through
  `parse_progress()`, which recognises `tqdm` bars (`42%|###`) and counters
  like `12/88`. Printing `f"{i+1}/{n}"` in your code is enough to get a real
  progress bar in the Activity panel. `task.detail` is the last output line.
- The text the user was typing in the console is saved before the task runs
  and restored after it.
- `cancel(task)` removes a queued task; a running one can only be
  interrupted (`bridge.interrupt()`, which is SIGINT, like Ctrl+C).

Dispatching (`_dispatch`): a task is sent with `console.execute(code)` only
when no visible request is in flight, the kernel is not dead or restarting, and
(except for bootstrap tasks) the bootstrap is complete. `CalpyConsole._execute`
reports the `msg_id` of every request (`request_sent`), so the bridge can bind
the `msg_id` to the task.

**Completion is detected from the shell `execute_reply`, never from the
console's `executed` signal**, because the console's bookkeeping is reset on
startup and restarts. Commands the user types count too:
`execution_finished` fires for every visible execution, so the workspace also
refreshes after typed code.

### 4.3 Queries

`query({"key": "expression"}, callback, code="")` sends a silent `execute`
request with `user_expressions` and `stop_on_error=False` (a failing query
never aborts other requests). The reply is converted by
`_parse_user_expressions`:

1. IPython returns each value as its `repr` (`text/plain`);
2. the bridge applies `ast.literal_eval` to it (non-literal reprs stay
   strings);
3. **if the result is a `str`, it is tried as JSON** (`json.loads`).

Consequences:

- Kernel helpers return `json.dumps(...)` strings, and the GUI receives
  dicts and lists directly (`workspace()`, `folders()`, `backend()`).
- To receive a string that *looks like* JSON (for example `"0.05"`) as a
  string, encode it twice. `ProcedureWindow.refresh_config_defaults` does
  this: `json.dumps(describe(value))`.
- A failed expression arrives as an `Exception` *instance* in the dict.
  Check with `isinstance(value, Exception)`.

A query that comes back `aborted` (because a visible command failed just
before it) is retried once.

### 4.4 Restarts, generations and the watchdog

- `_generation` is incremented on every restart, shutdown or death.
  Callbacks wrapped with `_current()` are dropped if they belong to an older
  generation.
- `_fail_pending(reason)` fails running tasks and pending queries with
  `KernelRestarted`. **Queued tasks survive** and run on the new kernel.
- A **fresh kernel client** (new sockets) is created on every restart, because
  old sockets can keep a stale route to the dead kernel.
- The **watchdog** (every 2 s, `_watchdog_check`): after a restart about 3% of
  requests used to vanish. While the kernel is idle, a request that has had no
  iopub acknowledgement for `REQUEST_STALL` = 8 s counts as lost. The client is
  reconnected, lost *queries* are re-sent (they only read), and lost
  *visible* commands fail with `ConnectionLost` (they are never run twice).
  More than three reconnects within 60 s count as a dead kernel.
- If the kernel dies, `QtKernelManager` restarts it automatically
  (`_on_kernel_auto_restarted` → bootstrap again). If that fails,
  `kernel_died(reason)` is emitted and the window shows a dialog.

### 4.5 `CalpyConsole`

`CalpyConsole` is a qtconsole `RichJupyterWidget` with three changes:

- `_handle_display_data` intercepts the two custom MIME types
  (`application/vnd.calpy.figure+json`, `application/vnd.calpy.image+json`)
  and re-emits them as `figure_received(dict)` / `image_received(dict)`
  instead of printing them;
- `_execute` emits `request_sent(msg_id, hidden)`;
- `reset`/`clear` drop qtconsole's handle to the last prompt block, which
  otherwise dangles and segfaults the GUI after a few restarts.

`custom_restart = True`: the console's own restart shortcut asks the
application, which asks for confirmation and runs the bootstrap again.

## 5. Inside the kernel (`kernel_side.py`, `kernel_backend.py`)

These modules run **in the kernel** and **must not import Qt**.
`kernel_side.install()` registers a `post_execute` hook and binds the module as
`_gui` in the user namespace (hidden from `%who`). Since `%reset` can delete
`_gui`, the GUI reaches the module through
`KERNEL_SIDE = "__import__('sys').modules['opticalib.gui.kernel_side']"`, and
re-installs `_gui` whenever it refreshes the workspace.

| Function | Called by | Returns or does |
|---|---|---|
| `view(obj, title)` | user code, procedure steps (`_gui.view(...)`) | Converts xupy/cupy/masked arrays to host numpy, writes `view_<pid>_<n>.npz` (`data`, plus `mask` if any) into `$CALPY_GUI_TMP`, publishes `IMAGE_MIME {path, title, shape, dtype}`. The GUI loads the file in a `LocalJob` and shows it in the viewer. Accepts 1-D, 2-D and 3-D arrays (cubes: frames along the **last** axis). |
| `publish_figures()` | the `post_execute` hook, `plt.show()`, `plt.pause()` | Renders every *changed* figure to PNG (120 dpi) and publishes `FIGURE_MIME {uid, num, title, png(base64)}`. The viewer updates the item `fig:<uid>` in place. |
| `workspace()` | GUI query after every execution | JSON list of `{name, type, module, kind, summary}` for the user variables; arrays also have `shape` (list), `dtype`, `masked` (has a mask) and `full_mask` (the mask is an array of the shape of the data, which opticalib's image and cube types require) |
| `require(obj, kind)` | generated code of `ImageParam`, `MaskParam`, `CubeParam` | Returns `obj` if opticalib's `isinstance_(obj, kind)` holds, else raises `TypeError` saying what it is (and, for a masked array with a scalar mask, how to build a full one) |
| `folders()` | GUI query at start, after a config change or experiment switch | JSON `{base, config, categories: [[label, path]], paths: {ATTR: path}}` from `opticalib.core.root` |
| `backend()` | GUI query, with the workspace | JSON `{on_gpu, available}` of xupy |
| `describe(value)` | procedure windows (configuration hints) | Short text: integer ranges become `np.arange(a, b[, s])` and long values are cut |
| `use_experiment(config)` | `install()`, experiment switch | Puts the experiment folder first on `sys.path`, so `DEVICE.CLASSES` modules can be imported |
| `mark_baseline()` | end of the bootstrap | Hides the names defined so far from the workspace |

How the workspace classifies objects (`_classify`): it inspects **only the
type**, never the instance, because probing attributes of a Pyro4 proxy would
make network calls. numpy, cupy and xupy-masked arrays become `"array"`. Then the
first match among `dm` (has `set_shape` and `get_shape`), `wfs`
(`acquire_map` and `acquire_pupil`), `interferometer` (`acquire_map`) and
`camera` (`acquire_frames`); anything else is `"other"`. A custom device class
is therefore recognised as soon as it has the methods of its protocol.

The matplotlib backend (`kernel_backend.py`) is Agg with a `draw_idle` that
only marks the canvas dirty. Rendering happens at publication time, so a
plot updated in a loop with `plt.pause()` streams to the plot panel.
*Kernel → Plots → interactive windows* runs `%matplotlib qt` in the kernel.
The hook then publishes nothing and figures get native windows; the other menu
choice switches back.

## 6. The main window (`app.py`)

### 6.1 Extension attributes

`CalpyGUI` is meant to be **subclassed by overriding class attributes only**
(see {doc}`extending_the_gui`):

| Attribute | Default | Used for |
|---|---|---|
| `APP_NAME` | `"CalpyGUI"` | Window title, startup overlay, About box, `QApplication` name |
| `SETTINGS_APP` | `"CalpyGUI"` | `QSettings` application name for the **layout and recent experiments**. Theme and Activity-panel preferences always use `"CalpyGUI"`, so they are shared |
| `DESKTOP_FILE_NAME` | `"OptiCalib"` | Lets Linux desktops match the window to its launcher (`None`: none) |
| `ICON` | `gui/resources/opticalib.png` | Window and application icon |
| `PROCEDURES` | `plugins.PROCEDURES` | Cards of the *Procedures* dock |
| `INIT_SCRIPTS` | `[]` | Scripts run in the kernel after `initCalpy.py`, in the same namespace, at startup and after every restart |

Subclasses add panels by overriding the `add_panels()` hook and calling
`add_dock(name, title, widget, area="left", tabify_with=None)` from it. The
hook runs inside `_build_ui`, before the menus, the default layout and the
saved layout, so extra panels behave exactly like the standard ones. The
`docks` property returns every panel by name.

### 6.2 Docks and layout

- Central widget: `PlotViewer`. Docks: `devices`, `procedures` (tabbed with
  devices), `workspace`, `data` (tabbed with workspace), all on the left, and
  `console` on the right. They are created by `_make_dock(name, title,
  widget)`, which sets `objectName = "dock_<name>"` (needed by `saveState`) and
  a `DockTitleBar` (float and close buttons).
- A tabbed dock shows **no title text**, because the tab names it; a dock that
  is alone or floating shows its name. The title bar itself always stays, so
  every panel can be dragged out (`_update_dock_titles`). A zero-height title
  bar was tried and rejected: the panels could no longer be detached.
- Qt/PySide 6.11 leaves *stale* `QTabBar`s visible when an area has two or
  more tab groups. `_hide_stale_tab_bars()` hides every tab bar that is not
  geometrically attached to the panel of its tab.
- The layout is saved in `QSettings` (`window/geometry`, `window/state`) with
  `LAYOUT_VERSION = 2`. **Bump `LAYOUT_VERSION` when you add, remove or rename
  docks**: older saved states are then ignored instead of misplacing panels.

### 6.3 Status bar

`KernelStatus` shows idle, busy (with the elapsed time), starting, restarting or
dead, next to the Interrupt and Restart buttons. `RamBar` reads the RAM of the
whole machine with `psutil` **in the GUI process** every 2 s, split into this
session (kernel + GUI RSS) and everything else, and turns orange or red at 85%
and 95%. `BackendButton` shows the xupy backend (bright green: GPU, dim green: CPU,
grey: no CuPy) and switches it with a visible `xp.use_gpu()` / `xp.use_cpu()`.
Then come the *switch experiment* button, the configuration path (click to
edit it) and the opticalib version.

### 6.4 The refresh loop

```text
any visible execution finishes ── bridge.execution_finished
        └─► 150 ms single-shot timer (debounce) ─► _refresh_workspace()
              query({"ws": workspace(), "xp": backend()}, code=re-install _gui if missing)
                └─► _on_workspace: WorkspaceView.set_items, DevicePanel.update_workspace,
                                   ProcedureContext.set_workspace (→ DeviceParam combos),
                                   BackendButton.set_state
```

Only one workspace query is in flight at a time; a refresh requested
meanwhile sets a *dirty* flag and runs once the reply arrives. The folders
(`_refresh_folders` → `DataBrowser.set_folders`, `ProcedureContext.set_folders`)
refresh when the kernel becomes ready, after a configuration save and after an
experiment switch. Procedure windows react to `folders_changed` by re-running
their checks and configuration hints.

### 6.5 Configuration edits and experiment switching

- *Edit configuration* (`Ctrl+,`, or a click on the path) opens
  `ConfigEditorDialog`. It validates the YAML (`validate_config_text`),
  writes only if the text changed, and asks before discarding edits. After a
  save: `DevicePanel.reload()`, `PluginPanel.reload()`, then the visible
  `opticalib.set_configuration_file(<path>)`, then a folders refresh.
- *File → Open experiment…* / *Recent experiments* / *Open configuration
  file…* → `switch_experiment(path)`: resolve (folder →
  `SysConfig/configuration.yaml` or `configuration.yaml`), validate, confirm
  (warning that connected devices keep their old configuration), then run
  `opticalib.set_configuration_file(...)` **visibly**. On success
  `_apply_experiment` updates the title, the status bar,
  `bridge.set_config_path` (the `AOCONF` of future restarts),
  `kernel_side.use_experiment` (`sys.path`), the device and procedure panels,
  the recent list, the folders and the workspace. The kernel is *not*
  restarted.

### 6.6 Procedure windows and Qt object lifetime

`_open_plugin_window(name)` instantiates `window_cls(context=procedure_context,
parent=self)`. Each click opens a new window, so several can be open. A window
emits `closed(self)` from `closeEvent`; the main window drops its reference and
calls `deleteLater()`. **Do not use `WA_DeleteOnClose` together with
`destroyed.connect(lambda ...)`**: that combination segfaulted PySide6. Callbacks
that may outlive a widget look it up again by key (see
`DevicePanel._connect`).

On close, `closeEvent` asks first if code is still running. It then saves the
layout, closes the procedure windows, interrupts and shuts down the kernel, and
deletes the temp folder.

## 7. Panels (`widgets/`)

### 7.1 Devices: registry, panel, connect dialog

**Registry** (`device_registry.py`, no Qt). Every entry of `DEVICES` is
resolved by `resolve_entry(section, name, conf, classes)` into a
`DeviceEntry`. The class is chosen in this order:

1. an explicit `class:` key in the entry;
2. otherwise the **longest alias prefix** of the entry name, compared
   case-insensitively with the classes of that section (`phasecam6110` →
   `PhaseCam`, suffix `6110`);
3. otherwise the section default, if the section has exactly one default
   (`CAMERAS` → `GigaVision`, `WFS` → `Ingot`).

`DeviceClass` describes how to build the constructor arguments (`args`):
`suffix`, `alpao` (number of actuators from the name, or
`serial_number=`), `name`, `camera`, `none`, or `template` for custom
classes. It also lists which entry the class actually **reads** (`reads`;
`PetalMirror` always reads `PetalDM`) and the required field groups. An entry
is `ready` when it has a class, arguments and no problems. Otherwise the card
says **Needs setup** and the tooltip gives the reason. `build_command()`
produces `import opticalib.devices as devices\ndm = devices.AlpaoDm(88)`, or
`from <module> import <Cls>\n<var> = <Cls>(<args>)` for custom classes.

**Custom classes** come from the top-level `DEVICE.CLASSES` section
(`load_custom_classes`):

```yaml
DEVICE.CLASSES:
  MorfeoDM:
    module: morfeo_dm            # dotted module, or a .py file inside the experiment folder
    section: DEFORMABLE.MIRRORS  # optional; without it the class is offered in every section
    args: '{suffix}'             # optional; {name} and {suffix} are substituted
    description: MORFEO DM       # optional
```

Custom classes are listed **before** the opticalib ones, so they win on name
clashes. A custom class without `section` takes the variable and kind of the
section its entry sits in (`DeviceEntry.kind_section`). The experiment folder (the parent of `SysConfig`) is put on the
kernel's `sys.path`.

**Writing to the configuration file.** `set_entry_class` and
`set_custom_class` do **line-level** edits: comments and formatting are
preserved, the result is re-parsed and checked, and the file is replaced
atomically (`_write_atomic`). Entries written inline (`Name: {a: 1}`) are
refused with a message asking to edit them by hand.

**Panel** (`device_panel.py`). A `DeviceInfo` per card: the configured
entries (`configured_devices()`) plus `SIMULATED_DEVICES` (Alpao with a size
combo box, DP, PetalMirror, and Fake4DInterf, which `requires="dm"`).
Card states: `disconnected`, `queued`, `connecting`, `connected`, `error`,
`unavailable` (needs a DM first), `setup`. *Connect* runs the command as a task.
A card becomes *connected* when the variable (`dm`, `interf`, `cam`, `wfs`,
`motor`, or the one chosen in the dialog) holds an instance whose **type name
and module prefix** match the card. This is why a device created by hand in the
console is also detected, if exactly one card can explain it. Choices made in
the connect dialog are kept per session in `_overrides` and `_session_classes`,
or saved to the file on request. The `⋯` menu offers *Copy connect code*,
*Connect with options…*, *Edit configuration entry* and, once connected, the
`quick_actions()` (show, acquire a map or frame, show the DM command, reset a
simulated DM, which asks for confirmation).

The two groups of the panel (*Configured* and *Simulated*) are
`CollapsibleSection`s (`widgets/common.py`): a clickable header with a chevron,
a one-line summary and a body holding the cards. The summary is rebuilt by
`DevicePanel._refresh_summaries` whenever a card emits `status_changed` or the
panel reloads, and counts devices, connected ones and failed ones, so a folded
group never hides a connection or an error. Reuse `CollapsibleSection` for any
panel that needs foldable groups; give it a `key` to remember its state.

A connected card goes back to *disconnected* when a fresh workspace shows
that its variable is gone or holds another type (`update_workspace`,
`demote=True`). The stored, possibly stale, workspace used by `reload()`
never demotes cards.

**Connect dialog** (`connect_dialog.py`). Pick a class from those of the
section, or *Custom … class* (name, module or `.py` file, arguments). Set the
variable name, preview and edit the command, and optionally save `class:`
in the entry (and the class in `DEVICE.CLASSES`). *Edit configuration*
closes the dialog with the `EDIT_CONFIG` result.

### 7.2 Workspace, data browser, plot viewer, config editor

- `WorkspaceView`: a tree grouped into Devices, Arrays and Other. A double click
  runs `default_action()` (`_gui.view(x, 'x')` for arrays, otherwise just the
  name). Every action is emitted as `run_requested(code, title)` and runs
  visibly.
- `DataBrowser`: one node per category of `folders()["categories"]`, then the
  tracking numbers (newest first, at most `MAX_TNS` = 500), then the files. A
  `QFileSystemWatcher` plus a 500 ms debounce timer refresh it as new tracking
  numbers appear. Double click: preview (`.fits` through `osu.load_fits`,
  `.npy`, `.npz`) in a `LocalJob`. The context menu loads the file into the
  console (`load_code`), copies the path or tracking number, or opens the folder.
- `PlotViewer`: `PlotItem`s keyed `fig:<uid>` (kernel figures, PNG) or
  data items (pyqtgraph `ImageView` with levels, histogram, colormaps, pixel
  readout and a frame slider for cubes). It has a thumbnail strip, and
  items can be popped out into a `PlotWindow`, saved (PNG, FITS, NPY) or removed.
- `ConfigEditorDialog`: `CodeEditor` (line numbers, soft tabs) with a
  `YamlHighlighter`, live validation, and `goto_entry(section, name)`.

## 8. The procedures framework (`plugins.py`, `procedures/`)

### 8.1 From card to window

`Procedure(name, icon, description, window, section=None)` is a card of the
*Procedures* dock. `icon` is a Material Design icon name without the `mdi6.`
prefix. If `section` is given and the configuration does not define it (it
is missing, `null` or empty), the card is hidden and listed under *"Not
configured: …"* (`defined_sections`). This is re-checked after every
configuration save or experiment switch.

### 8.2 `ProcedureContext`

There is one context per main window, shared by all procedure windows. It is
the only way a procedure window reaches the kernel:

- `run(code, title, on_done, on_error) -> Task`: visible execution;
- `query(expressions, callback)`: silent evaluation;
- `interrupt()`;
- `view_file(path)`: preview a file in the plot viewer;
- `workspace` and `workspace_changed(list)`: the kernel variables (feeds
  `DeviceParam`);
- `folders`, `folders_changed(dict)` and `folder(attr)`: the data folders (feeds
  `TnParam` and `TnListParam`).

A window built with `context=None` still works as a *code preview* only:
Run stays disabled.

### 8.3 `Step` and `ProcedureWindow`

```python
Step(
    key="acquire",                 # identifier
    title="Acquire ...",           # shown in the step list
    description="...",             # shown above the form
    params=[...],                  # Param objects (one widget each)
    template=lambda v: "...",      # v = {param name: Python source}; returns the code
    outputs={"iff_tn": "{out}"},   # state key -> kernel expression, read after success
    confirm="This step moves ...", # confirmation for hardware-moving steps
    after=callback,                # after(window, results, values) once outputs are read
    section="Calibration",         # heading in the step list
)
```

What `ProcedureWindow` does, in order:

1. `__init__` calls `steps()` **once**: each window instance gets fresh `Param`
   objects. It builds the UI (step list with section headers, form, live code
   preview, *Copy code*, *Stop*, *Run step*, results log), runs `CHECKS` and
   reads the configuration defaults.
2. Forms are built **lazily**, the first time a step is shown
   (`_form_for`). Any parameter change re-renders the code. An invalid
   parameter (`ParamError`) shows its message and disables Run.
3. **Run** (`run_step`): `values = step.values()` (Python source of each
   parameter) → `code = template(values)`. The output expressions are formatted
   with the same values (`"{out}"` → `"tn_iff"`). If `confirm` is set, a dialog
   shows the message **and the exact code**. Then `context.run(...)`. Only one
   step per window runs at a time.
4. On success: `query({k: f"repr({expr})"})` for the outputs → each is
   `ast.literal_eval`'d → `set_state(key, value)`. `set_state` fills every
   built parameter whose `state_key` equals `key` (only `TnParam` has a
   `state_key`). Then a line in the results log, and the `after()` hook.
5. On error: the step shows a red icon and the log gives `ename: evalue`;
   the traceback is in the console and in the Activity card.
6. **Stop** calls `context.interrupt()` (KeyboardInterrupt in the kernel).

Pre-flight `CHECKS = [(expression, warning), ...]`: each expression is
evaluated silently, and the warning is shown on top only when the result **is
`False`**. An expression that raises (a bare `dm` gives `NameError`) or returns
another falsy value shows nothing, so write checks that always return a bool,
such as `'dm' in globals()`.
`Param.config_default` is a kernel expression giving the value used when an
optional field is left empty. It is shown as placeholder text through
`describe()` and re-read when the configuration reloads.

### 8.4 Parameters

| Class | Widget | `code()` renders | Notes |
|---|---|---|---|
| `IntParam(name, label, default, minimum, maximum)` | spin box | `5` | |
| `FloatParam(..., default, minimum=None, maximum=None)` | line edit | `1e-07` / `None` | scientific notation accepted; `None` only if `optional` |
| `BoolParam` | check box | `True`/`False` | |
| `ChoiceParam(..., choices=[(value, label)], default)` | combo | `repr(value)` | a `None` choice shows the configured value in its label |
| `TextParam(..., default, placeholder)` | line edit | `'text'` (string literal) | |
| `ExprParam` | line edit | the expression **verbatim** | syntax checked with `compile()` |
| `VarParam` | line edit | bare identifier | for output variables (`tn_iff = ...`) |
| `DeviceParam(..., kinds, default)` | editable combo | bare identifier | lists workspace variables of the given kinds |
| `DeviceListParam` | line edit | `[a, b]` | |
| `ImageParam`, `MaskParam`, `CubeParam` (`KernelArrayParam` subclasses) | editable combo | `_gui.require(img, 'ImageData')` | names a kernel array; suggestions come from the workspace (`shape`, `dtype`, `full_mask`), the type is **checked in the kernel** when the step runs |
| `TnParam(..., folder_attr, state_key)` | editable combo + refresh | `'YYYYMMDD_HHMMSS'` | lists the folders of `folders.<folder_attr>` |
| `TnListParam(..., folder_attr, minimum)` | line edit + *Add* menu | `['tn1', 'tn2']` | |

Every parameter takes `help=` (tooltip), `optional=` and `config_default=`.
`optional` only has an effect on text-like parameters (`FloatParam`,
`TextParam`, `ExprParam`, `DeviceParam`, `TnParam`), which render an empty
field as `None`. `DeviceListParam` then gives `[]`, and `IntParam`,
`BoolParam`, `ChoiceParam` and `TnListParam` always have a value. The
`config_default` placeholder is shown only by line-edit or editable-combo
parameters, plus the `None` choice of a `ChoiceParam`. `IntParam` and
`BoolParam` ignore it. `kwargs_code(values, "a", "b")` renders `a=..., b=...`
and leaves out values that are `None`, so library defaults apply.

### 8.5 The shipped windows

| Window | Card section | Steps | Verified |
|---|---|---|---|
| Deformable Mirror Calibration | `INFLUENCE.FUNCTIONS` | *Calibration:* acquire → process → singular values → apply flat → closed loop. *Post-processing:* Zernike filter, stack cubes, ROI processing | end to end on `AlpaoDm(88)` + `Fake4DInterf` |
| Timeseries | none (always shown) | acquire → average → running difference → noise vs time lag | analysis on synthetic OPDSeries data |
| Stitching | `STITCHING` | plan the grid → acquire the scan → stitch | code generation only (needs hardware) |
| Alignment | `SYSTEM.ALIGNMENT` | set up → read positions → calibrate → compute correction → apply | code generation only |
| Segments Phasing | `PHASING` | set up → dark → preview detection → acquire the sweep → analyse | code generation only |

The DM calibration window has a pre-flight check. Influence-function defaults
are read from `<data path>/SysConfig/configuration.yaml`, which may not be the
configuration loaded in the GUI, and the window warns when the two differ.

## 9. Activity, overlay and theme

- `ActivityCenter.track(job)` accepts a `Task` or a `LocalJob` (quiet ones are
  not tracked; for example, the loads triggered by `_gui.view` are silent). It
  keeps at most `MAX_CARDS` = 4 *finished* cards (running and queued ones
  are never evicted), each with a spinner or progress bar, the elapsed time,
  the last output line and Interrupt/Cancel. Finished jobs become a 4 s toast;
  errors stay, with a *Traceback* button. The panel defaults to the top-left
  corner of the plot area, can be dragged by its header, pins to the nearest
  corner, and can be locked or collapsed. Its position is saved
  (`activity/*` settings).
- `StartupOverlay` lists `KernelBridge.BOOTSTRAP_STEPS` and turns red with the
  message when a step fails; it can then be dismissed.
- `LocalJob(fn, title, on_done, on_error).start()` runs `fn` in a thread and
  calls back **in the GUI thread**. `fn` must not touch widgets.
- `theme()` is the shared `ThemeManager`: `tokens` (colors of the active
  light or dark theme: `bg`, `surface`, `text`, `text_muted`, `accent`,
  `success`, `warning`, `danger`, ...), `color(token)`, `icon(mdi_name, token)`
  and the `changed` signal. Widgets are styled through **dynamic properties**
  that the style sheet knows: `muted`, `heading`, `title`, `section`, `card`,
  `accent` (buttons), `floating`, and `state` (`done`/`error`). Call the
  module function `opticalib.gui.theme.repolish(widget)` after changing one at
  runtime.

## 10. What is stored where

| Store | Keys | Notes |
|---|---|---|
| `QSettings("ArcetriAdaptiveOptics", SETTINGS_APP)` | `window/geometry`, `window/state`, `recent/experiments` | per application (`SETTINGS_APP`) |
| `QSettings("ArcetriAdaptiveOptics", "CalpyGUI")` | `ui/theme`, `ui/collapsed_<key>` (foldable groups, e.g. `devices_configured`), `activity/anchor`, `activity/offset_x`, `activity/offset_y`, `activity/pinned`, `activity/collapsed` | shared by every GUI built on opticalib |
| `configuration.yaml` | `DEVICES.<section>.<entry>.class`, `DEVICE.CLASSES` | written only on explicit request, line-level and atomic |
| temp folder `calpygui-*` | kernel sockets, `view_*.npz` | deleted when the window closes |

On Linux, `QSettings` lives in `~/.config/ArcetriAdaptiveOptics/<app>.conf`.
Tests redirect it with `QSettings.setPath` (the `qapp` fixture).

## 11. Tests

| File | What it covers |
|---|---|
| `tests/test_gui_kernel_bridge.py` | `KernelBridge` against a fake kernel client: dispatching, queries, restarts, watchdog |
| `tests/test_gui_kernel_side.py` | `_gui` helpers: `view`, figure publication, workspace classification, folders |
| `tests/test_gui_procedures.py` | params and procedure windows with a `FakeContext` that records the generated code |
| `tests/test_gui_devices.py` | registry resolution, `DEVICE.CLASSES`, YAML line edits, device cards and the connect dialog |
| `tests/test_gui_widgets.py` | plot viewer, workspace, data browser, theme, `BackendButton`, `RamBar`, `LocalJob`, experiment switching |
| `tests/test_gui_config_editor.py` | editor and validation |
| `tests/test_gui_activity.py` | activity panel position, pinning and summary |
| `tests/test_gui_app.py` | **end to end** in a subprocess with a real kernel (`@pytest.mark.integration`): startup, docks, aliases, experiment switching, layout persistence |

Run them headless with `QT_QPA_PLATFORM=offscreen` (CI uses Xvfb).
`tests/conftest.py` selects PySide6 and provides the `qapp` fixture.
`CUDA_VISIBLE_DEVICES=""` keeps xupy on the CPU when CuPy cannot compile
kernels on the machine.

## 12. Known limitations

- Simulated `M4AU` has no device card: downloading its reference data timed out.
- Devices connected before an experiment switch keep their old configuration
  until they are reconnected.
- A query waits behind a running visible command, because the kernel serves one
  request at a time. Panels that poll the kernel should skip polling while
  `bridge.is_busy`.
- On 2026-09-30, several *library* (not GUI) issues were found while wiring
  the procedure windows: Alignment configuration keys, `TimeSeries` naming
  frames by the second, `closed_loop_flattening(iterations=None)` importing
  `msvcrt`, `SPL.acquire` without a dark frame, and others. Some have since
  been fixed. If a step fails, check the library call first: the GUI only
  generates it.
