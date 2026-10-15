(dev-guide-extending-gui)=

# Extending CalpyGUI: tutorials

These tutorials show how to add features to the opticalib GUI, and how to build
a GUI for your own experiment (with proprietary devices and procedures) on top
of it, **without forking or patching opticalib**. Read
{doc}`gui_internals` first, or keep it open alongside: these tutorials point
to it for details.

Every example below was run against opticalib 2.2.0, both headless with a fake
context and end to end with a real kernel.

## 0. Ground rules

The GUI stays maintainable because of a handful of invariants. Every tutorial
below respects them; your extensions should too.

1. **Put the logic in a Python function, not in the GUI.** A procedure step
   should generate a *short call* (`tn = poke_test(dm, interf, ...)`), not 40
   lines of loop. The function can then be used from scripts and tested
   without Qt, and the GUI stays a thin layer.
2. **Anything that changes state runs visibly** (`context.run` /
   `bridge.run`). Silent `query` calls only *read*.
3. **The GUI process never imports or holds devices.** It refers to them by
   *variable name* in the kernel (`dm`, `interf`, ...).
4. **Kernel-side code never imports Qt.** That covers your `INIT_SCRIPTS` and
   the helpers you call from queries.
5. **Never block the GUI thread.** Kernel work goes through tasks; heavy work in
   the GUI process goes in a `LocalJob`.
6. **Steps that move hardware set `confirm=`.** The dialog then shows the
   exact code before anything moves.
7. **Read `opticalib.folders.<X>` at call time**, not at import time. After
   `opticalib.set_configuration_file()` the same `folders` object gets new
   paths, but values copied into module globals or default arguments do not.
   Your own modules are *not* reloaded.
8. **Extend through the public API**: subclassing, class attributes
   (`PROCEDURES`, `INIT_SCRIPTS`, ...) and the `add_panels()` / `add_dock()`
   hook. Avoid private methods (`_name`); if you need one, pin your opticalib
   version.

### Where does my code go?

| You want to... | In opticalib (a PR to the library) | In your own package |
|---|---|---|
| Add a step to an opticalib procedure | `gui/procedures/<window>.py` | subclass the window, override `steps()` |
| Add a generic procedure | new `gui/procedures/<name>.py` + an entry in `plugins.PROCEDURES` | a `ProcedureWindow` in your package + `MyGUI.PROCEDURES` |
| Connect a new device class | `DEVICE_CLASSES` in `widgets/device_registry.py` | `DEVICE.CLASSES` in the experiment configuration |
| Lab-specific panels or start-up code | n/a | subclass `CalpyGUI` (+ `INIT_SCRIPTS`) |

## Tutorial 1: add a step to an existing procedure

Goal: in the *Timeseries* window, add a step that shows one frame of a series.

A `Step` is declarative: parameters, a template that turns their Python source
into code, and optionally outputs, confirmation, a post-run hook and a section.
Add it to the list returned by `TimeseriesWindow.steps()` in
`opticalib/gui/procedures/timeseries.py`:

```python
Step(
    "frame",
    "Show one frame",
    "Load one frame of a series and show it in the viewer.",
    [
        TnParam("tn", "Tracking number", folder_attr=_SERIES, state_key="ts_tn"),
        IntParam("index", "Frame index", 0, minimum=0),
    ],
    lambda v: (
        f"_frames = sorted(os.listdir(join(folders.OPD_SERIES_ROOT_FOLDER, {v['tn']})))\n"
        f"_gui.view(osu.load_fits(join(folders.OPD_SERIES_ROOT_FOLDER, {v['tn']}, "
        f"_frames[{v['index']}])), 'Frame {v['index']} of ' + {v['tn']})"
    ),
),
```

What happens:

- `TnParam(..., state_key="ts_tn")` pre-fills itself with the tracking number
  that the *Acquire* step stored under `outputs={"ts_tn": "{out}"}`. Its combo
  lists the folders of `folders.OPD_SERIES_ROOT_FOLDER`.
- `v['tn']` is the Python **source** of the value, so it is already quoted
  (`'20261013_120000'`); never wrap it in quotes again. `v['index']` is `0`.
- The template runs in the console namespace, so the `initCalpy.py` aliases
  (`os`, `np`, `osu`, `join`, `folders`, `_gui`, ...) are available. Variables
  starting with `_` are hidden from the Workspace panel.
- No `confirm=`: the step only reads data.

To see it, open the window: the code preview updates as you edit the fields.
Then add a test (see [Tutorial 8](#tutorial-8-test-your-extensions)).

If the window belongs to a library you can't modify, subclass it instead:

```python
from opticalib.gui.procedures.timeseries import TimeseriesWindow

class MyTimeseriesWindow(TimeseriesWindow):
    def steps(self):
        return super().steps() + [Step("frame", ...)]
```

Then register it under a **new name**, or replace the built-in entry in the
list. Cards are keyed by `Procedure.name`, so adding a second "Timeseries"
produces a duplicate card, not a replacement:

```python
PROCEDURES = [
    Procedure("Timeseries", "chart-line", "...", MyTimeseriesWindow) if p.name == "Timeseries" else p
    for p in opticalib.gui.PROCEDURES
]
```

## Tutorial 2: write a new procedure window

Goal: a *Poke test* that pushes DM actuators one by one, saves one map per
actuator under a new tracking number, and then shows any of them.

### 2.1 First, the library function (rule 1)

```python
# mylab/pokes.py
import os

import numpy as np
import opticalib
from opticalib.ground import osutils as osu


def poke_test(dm, interf, actuators, amplitude: float = 1e-7, nframes: int = 1) -> str:
    """
    Push the given actuators one at a time and save one map per actuator.

    Returns
    -------
    str
        The tracking number of the data, in OPDImages.
    """
    tn = osu.newtn()
    # Read the folder at call time: it follows opticalib.set_configuration_file.
    folder = os.path.join(opticalib.folders.OPD_IMAGES_ROOT_FOLDER, tn)
    os.makedirs(folder)
    actuators = [int(a) for a in np.atleast_1d(actuators)]
    for i, act in enumerate(actuators):
        poke = np.zeros(dm.n_acts)
        poke[act] = amplitude
        dm.set_shape(poke, differential=True)
        img = interf.acquire_map(nframes)
        dm.set_shape(-poke, differential=True)
        osu.save_fits(os.path.join(folder, f"act_{act:04d}.fits"), img)
        print(f"{i + 1}/{len(actuators)}")  # the GUI turns "i/N" into a progress bar
    return tn
```

It follows the opticalib conventions: a new `tn` (`osu.newtn`) inside a
`folders.*_ROOT_FOLDER`, `osu.save_fits` for I/O, and a returned `tn`.
The `print(f"{i+1}/{n}")` line is all the Activity panel needs for a real
progress bar (`parse_progress` also understands `tqdm`).

### 2.2 The window

```python
# mylab/gui.py (part)
from typing import List

# opticalib.gui first: it selects the Qt binding before anything imports qtpy.
from opticalib.gui.procedures.base import ProcedureWindow, Step, kwargs_code
from opticalib.gui.procedures.params import (
    DeviceParam, ExprParam, FloatParam, IntParam, TnParam, VarParam,
)


def _after_poke(window, results, values):
    """Called after a successful poke: reset the 'show' step to the first actuator."""
    window.step("show").param("actuator").set_value(0)


class PokeTestWindow(ProcedureWindow):
    TITLE = "Poke test"
    DESCRIPTION = "Push actuators one at a time and look at their influence functions."
    CHECKS = [("'dm' in globals()", "No 'dm' in the console: connect a deformable mirror first.")]

    def steps(self) -> List[Step]:
        return [
            Step(
                "poke", "Poke the actuators",
                "Push each actuator, measure, pull it back. Saves one FITS per actuator in OPDImages.",
                [
                    DeviceParam("dm", "Deformable mirror", kinds=("dm",), default="dm"),
                    DeviceParam("interf", "Interferometer", kinds=("interferometer", "wfs"), default="interf"),
                    ExprParam("actuators", "Actuators", "range(10)"),
                    FloatParam("amplitude", "Amplitude [m]", 1e-7, minimum=0),
                    IntParam("nframes", "Frames per map", 1, minimum=1, maximum=100),
                    VarParam("out", "Store the tracking number in", "tn_poke"),
                ],
                lambda v: (
                    f"{v['out']} = poke_test({v['dm']}, {v['interf']}, {v['actuators']}, "
                    + kwargs_code(v, "amplitude", "nframes") + ")"
                ),
                outputs={"poke_tn": "{out}"},
                confirm="This step moves the deformable mirror.",
                after=_after_poke,
                section="Acquisition",
            ),
            Step(
                "show", "Show one actuator",
                "Open the map of one actuator in the viewer.",
                [
                    TnParam("tn", "Tracking number", folder_attr="OPD_IMAGES_ROOT_FOLDER", state_key="poke_tn"),
                    IntParam("actuator", "Actuator", 0, minimum=0, maximum=10000),
                ],
                lambda v: (
                    f"_gui.view(osu.load_fits(join(folders.OPD_IMAGES_ROOT_FOLDER, {v['tn']}, "
                    f"'act_{{:04d}}.fits'.format({v['actuator']}))), 'Actuator {v['actuator']}')"
                ),
                section="Analysis",
            ),
        ]
```

The generated code for the first step, with the defaults, is:

```python
tn_poke = poke_test(dm, interf, range(10), amplitude=1e-07, nframes=1)
```

After it succeeds, the window evaluates `repr(tn_poke)` silently, stores it as
`state["poke_tn"]`, and the *Show* step's `TnParam(state_key="poke_tn")`
picks it up.

### 2.3 Register it

- **In opticalib**: add it to `plugins.PROCEDURES`:

  ```python
  Procedure("Poke test", "gesture-tap", "Push actuators one by one.", PokeTestWindow,
            section=None)  # or e.g. "INFLUENCE.FUNCTIONS" to show it only when configured
  ```

  Also list it in the `procedures/__init__.py` docstring and in
  `docs/reference/gui.rst`.
- **In your own package**: put it in your GUI subclass
  ([Tutorial 5](#tutorial-5-build-your-own-experiment-gui)).

Icons are Material Design names without the `mdi6.` prefix
(<https://pictogrammers.com/library/mdi/>).

### 2.4 Step recipes and gotchas

| Need | How |
|---|---|
| Optional argument that falls back to the library default | `optional=True` on the param, then `kwargs_code(v, "name", ...)`: values that are `None` are left out |
| Show the configured default as a placeholder | `config_default="<kernel expression>"`, e.g. `"__import__('opticalib.core.config', fromlist=['_']).get_iff_config('IFFUNC')['amplitude']"` |
| Hand a result to later steps | `outputs={"state_key": "<expr using {param}>"}` + `TnParam(state_key=...)` |
| Pre-fill a non-`TnParam` parameter | `after=lambda window, results, values: window.step("k").param("p").set_value(...)` (only `TnParam` has `state_key`) |
| Warn before the user starts | `CHECKS = [(expr, message)]`, re-evaluated after configuration reloads. The warning shows only when `expr` returns exactly `False`: write `'dm' in globals()`, not `dm` (an error shows nothing) |
| Plot something | emit matplotlib code (`figure(...)`, `plot(...)`); it appears in the plot panel by itself |
| Show an array or cube | `_gui.view(array, "title")` (cube frames along the **last** axis) |
| Group steps | `section="Calibration"` on consecutive steps |

Gotchas:

- `outputs` values go through `str.format(**values)`. Write a literal brace as
  `{{` / `}}`. The same applies to f-string templates (`'act_{{:04d}}'` above).
- Output values come back through `repr` + `ast.literal_eval`: return
  strings, numbers, lists or dicts, not objects.
- Each window calls `steps()` **once per instance**, and a `Param` owns
  exactly one widget. Never share a `Param` instance between two steps or two
  windows; use a small factory function if several steps need the same
  parameter (as M4's `_scripts_param()` does).
- Only one step per window runs at a time. Steps from different windows queue
  in the console in FIFO order.
- `ExprParam` is inserted **verbatim**. `TextParam` is quoted. `VarParam`,
  `DeviceParam` and `TnParam` are validated identifiers or literals.

## Tutorial 3: write a new parameter type

Goal: a `FileParam` with a *Browse…* button that renders as a string literal
and refuses files that do not exist.

The contract of a `Param`:

- `_make_widget()` returns the widget and emits `self.changed` on every edit;
- `code()` returns the Python source, or raises `self._error("message")`
  (a `ParamError`, shown in the window and disabling *Run*);
- `value()` and `set_value()` read and write the value;
- optionally `_setup()` (called once the widget exists; connect it to
  `self.context` there) and `_hint_edit()` (the `QLineEdit` that receives the
  configuration placeholder).

```python
import os

from qtpy.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QToolButton, QWidget
from opticalib.gui.procedures.params import Param


class FileParam(Param):
    """A file path, typed or picked with a dialog; rendered as a string literal."""

    def __init__(self, name, label, filter="All files (*)", **kwargs):
        super().__init__(name, label, **kwargs)
        self.filter = filter

    def _make_widget(self) -> QWidget:
        box = QWidget()
        self._edit = QLineEdit()
        self._edit.setPlaceholderText("None" if self.optional else "path to a file")
        self._edit.textChanged.connect(lambda *_: self.changed.emit())
        browse = QToolButton()
        browse.setText("…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._edit, 1)
        row.addWidget(browse)
        return box

    def _hint_edit(self):
        self.widget()
        return self._edit

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self._widget, self.label, "", self.filter)
        if path:
            self._edit.setText(path)

    def value(self):
        self.widget()
        text = self._edit.text().strip()
        if not text:
            if self.optional:
                return None
            raise self._error("a file is required")
        if not os.path.isfile(os.path.expanduser(text)):
            raise self._error(f"'{text}' does not exist")
        return os.path.expanduser(text)

    def set_value(self, value) -> None:
        self.widget()
        self._edit.setText("" if value is None else str(value))

    def code(self) -> str:
        path = self.value()
        return "None" if path is None else repr(path)
```

Use it like any other parameter:
`Step("load", "Load a command file", "...", [FileParam("path", "Command file", "FITS (*.fits)"),
VarParam("out", "Store it in", "cmd")], lambda v: f"{v['out']} = osu.load_fits({v['path']})")`.

```{note}
The check `os.path.isfile` runs in the **GUI** process. That is fine on one
machine; if the kernel ever runs elsewhere, check paths in the kernel (with a
`CHECKS` entry or in the library function) instead.
```

## Tutorial 4: connect proprietary hardware

### 4.1 Make the class look like an opticalib device

The GUI recognises devices by the **methods of their class**, never by
inheritance or by probing the instance (see {doc}`gui_internals`, section 5):

| Kind | The class needs | What you get in the GUI |
|---|---|---|
| `dm` | `set_shape` and `get_shape` (and `n_acts` for the opticalib procedures) | Workspace *Devices* group, `DeviceParam(kinds=("dm",))`, *Show command* quick action |
| `wfs` | `acquire_map` and `acquire_pupil` | *Acquire map* |
| `interferometer` | `acquire_map` | *Acquire map* |
| `camera` | `acquire_frames` | *Acquire frame* |

Satisfy the matching protocol of `opticalib.core._types` (see
{doc}`adding_a_device`) and every opticalib procedure works with your device
too. Deriving from `devices._API.base_devices.BaseDeformableMirror` gives you
actuator slaving for free.

### 4.2 Declare the class in the experiment configuration

No opticalib change is needed. Put the driver in the experiment folder (the
folder that contains `SysConfig`; the GUI adds it to the kernel's `sys.path`)
or install it as a package, then:

```yaml
DEVICES:
  DEFORMABLE.MIRRORS:
    AcmeDM12:                # "AcmeDM" + suffix "12"
      ip: 192.168.0.10
      port: 5000
  CAMERAS:
    LabCam:
      class: AcmeCamera      # explicit class: the name does not have to match

DEVICE.CLASSES:
  AcmeDM:
    module: drivers/acme_dm.py           # a .py file in the experiment folder...
    section: DEFORMABLE.MIRRORS
    args: "{suffix}, config_name='{name}'"
  AcmeCamera:
    module: acme_sdk.camera              # ...or an importable module
```

The device cards then run:

```python
from drivers.acme_dm import AcmeDM
dm = AcmeDM(12, config_name='AcmeDM12')

from acme_sdk.camera import AcmeCamera
cam = AcmeCamera()
```

Notes:

- `{name}` is the entry name and `{suffix}` is the entry name after the class
  name (case-insensitive). `args` is inserted verbatim, so quote strings yourself.
- `section:` is optional. With it, the class is offered only in that section.
  Without it, the class is offered in every section, and each entry takes
  the variable and kind of the section it sits in (`cam` under `CAMERAS`,
  `dm` under `DEFORMABLE.MIRRORS`, ...).
- The same declaration can be made from the GUI: the card's *Connect with
  options…* → *Custom … class*, with *save* checked. The edit keeps your
  comments and is checked before the file is written.
- Inside the driver, read the entry with
  `opticalib.core.config.get_device_config(...)` (or your own reader), at
  construction time.

### 4.3 Add the class to opticalib itself (for library devices)

For a device that belongs in `opticalib.devices`, add a `DeviceClass` to
`DEVICE_CLASSES` in `gui/widgets/device_registry.py`:

```python
DeviceClass(
    "AcmeDM",                  # name in opticalib.devices
    "DEFORMABLE.MIRRORS",      # section of its entries
    ("acmedm", "acme"),        # lower-case name prefixes, the longest wins
    "suffix",                  # args: suffix | alpao | name | camera | none
    "AcmeDM{suffix}",          # the entry the class reads (None: none, "*": the entry by name)
    (("ip", "port"),),         # required fields (alternative groups)
    description="Acme deformable mirror",
),
```

Add a case to `tests/test_gui_devices.py` (resolution, `build_command`,
the *Needs setup* reason when a field is missing). For a new `DEVICES`
section, also add it to `SECTION_VARIABLES` and `SECTION_KINDS`.

Simulated devices are listed in `SIMULATED_DEVICES` and quick actions in
`quick_actions()`, both in `widgets/device_panel.py`. A quick action that moves
hardware must return `needs_confirmation=True`.

## Tutorial 5: build your own experiment GUI

Goal: a *MyLab GUI* that is opticalib's GUI plus your procedures and
environment, launched from your own package. This is the pattern used by M4
(`m4/gui/calpygui/`).

### 5.1 Package layout

```text
mylab/
├── __init__.py        # from .pokes import bench_status, poke_test
├── pokes.py           # the library functions (Tutorial 2.1), no Qt
├── init_gui.py        # kernel start-up script, no Qt
├── gui.py             # windows, params, the CalpyGUI subclass, main()
└── ...
```

### 5.2 The kernel start-up script

It runs in the kernel after opticalib's `initCalpy.py`, **in the same
namespace**, at startup and after every restart:

```python
# mylab/init_gui.py
"""Kernel start-up script of the MyLab GUI (runs after initCalpy.py and after every restart)."""
import mylab
from mylab import poke_test

print("MyLab environment loaded.")
```

Keep it light, and **do not connect to hardware here**: it runs on every
restart. If it fails, the remaining init scripts are skipped, and the overlay
shows the error with *Continue anyway*, so the GUI comes up without your
environment. Let
the user connect devices from the device cards or a *Setup* step. Names
defined here are hidden from the Workspace panel (`mark_baseline`).

### 5.3 The window subclass

```python
# mylab/gui.py (part)
import os
from typing import Optional

from opticalib.gui import PROCEDURES, CalpyGUI, Procedure
from opticalib.gui.app import launch_gui

_HERE = os.path.dirname(os.path.abspath(__file__))


class MyLabGUI(CalpyGUI):
    APP_NAME = "MyLab GUI"           # title, startup screen, About
    SETTINGS_APP = "MyLabGUI"        # own layout + recent experiments (theme shared)
    DESKTOP_FILE_NAME = None
    INIT_SCRIPTS = [os.path.join(_HERE, "init_gui.py")]
    PROCEDURES = [
        Procedure("Poke test", "gesture-tap", "Push actuators one by one.", PokeTestWindow),
    ] + PROCEDURES                   # drop "+ PROCEDURES" to show only yours


def main(config_path: Optional[str] = None) -> None:
    launch_gui(config_path, window_class=MyLabGUI)
```

Everything else (console, devices with your `DEVICE.CLASSES`, data browser,
viewer, config editor, experiment switching, restarts) is inherited unchanged.

### 5.4 The launcher

opticalib (and possibly your package) reads the configuration **at import
time**, so set `AOCONF` *before* importing them:

```python
# mylab/launch.py
import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="MyLab GUI")
    parser.add_argument("-f", "--config", help="experiment folder or configuration.yaml")
    args = parser.parse_args()
    config_path = None
    if args.config:
        path = os.path.abspath(os.path.expanduser(args.config))
        if os.path.isdir(path):
            path = os.path.join(path, "SysConfig", "configuration.yaml")
        os.environ["AOCONF"] = config_path = path
    from mylab.gui import main as gui_main   # imports opticalib only now

    gui_main(config_path)
```

Expose it as an entry point in your `pyproject.toml`:

```toml
[project.gui-scripts]
mylab-gui = "mylab.launch:main"
```

The kernel inherits the GUI's environment (`PYTHONPATH` included) and always
uses the GUI's interpreter, so `mylab` must be importable in that environment.

## Tutorial 6: read kernel state from the GUI

Goal: show live information computed in the kernel (here, which devices the
console holds) in a GUI widget.

1. Write the helper **kernel-side**, fast and without hardware I/O, returning
   JSON:

   ```python
   # mylab/pokes.py
   def bench_status() -> str:
       """One-line bench summary, queried by the GUI (must be fast, no hardware I/O)."""
       import json
       ns = __import__("IPython").get_ipython().user_ns
       return json.dumps({"dm": type(ns.get("dm")).__name__, "interf": type(ns.get("interf")).__name__})
   ```

2. Query it from the GUI:

   ```python
   self.bridge.query({"s": "mylab.bench_status()"}, self.bench_panel.show_status)
   # in a procedure window: self.context.query({...}, callback)
   ```

   The callback receives `{"s": {"dm": "AcmeDM", "interf": "PhaseCam"}}`: the
   JSON string is decoded for you. If the expression failed, `results["s"]`
   is an **`Exception` instance**; always check with `isinstance`.

Rules for queries:

- **Read only.** Anything that changes state goes through `run` (rule 2).
- **Do not poll while the kernel is busy.** The kernel serves one request at a
  time, so queries wait behind a running acquisition. Guard timers with
  `if self.bridge.is_ready and not self.bridge.is_busy:`.
- **Prefer events to polling.** `bridge.execution_finished` fires after every
  visible execution, which is when the state can have changed (the
  Workspace panel uses it with a 150 ms debounce).
- Refer to the kernel helpers by module (`mylab.bench_status()`), not by a
  name that the user might overwrite in the console.
- To receive a string that looks like JSON (`"0.05"`) as a string, return
  `json.dumps(json.dumps(x))`.

## Tutorial 7: add a dock panel

Goal: show the `bench_status()` of Tutorial 6 in its own dock.

Override the `add_panels()` hook and call `add_dock()` from it.

```python
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QLabel


class BenchPanel(QLabel):
    """A tiny read-only panel showing which devices the console holds."""

    def __init__(self):
        super().__init__("Kernel not ready")
        self.setProperty("muted", True)          # styled by the theme's QSS
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setMargin(8)

    def show_status(self, results):
        status = results.get("s")
        if isinstance(status, Exception):
            self.setText(f"Status unavailable: {status}")
        else:
            self.setText("\n".join(f"{k}: {v}" for k, v in status.items()))


class MyLabGUI(CalpyGUI):
    ...  # class attributes as in Tutorial 5

    def add_panels(self) -> None:
        self.bench_panel = BenchPanel()
        self.add_dock("bench", "Bench", self.bench_panel, tabify_with="workspace")
        self._bench_timer = QTimer(self, interval=2000, timeout=self._poll_bench)
        self._bench_timer.start()

    def _poll_bench(self) -> None:
        # A query waits behind running code: do not pile them up while busy.
        if self.bridge.is_ready and not self.bridge.is_busy:
            self.bridge.query({"s": "mylab.bench_status()"}, self.bench_panel.show_status)
```

How it works:

- `add_panels()` is called while the window is built, after the standard
  panels and **before** the menus and the layouts. Your panel is therefore
  listed in *View*, remembered between sessions (under your `SETTINGS_APP`)
  and put back by *View → Reset layout*.
- `add_dock(name, title, widget, area="left", tabify_with=None)` sets the
  dock's `objectName` (`dock_<name>`; keep `name` stable, because the saved
  layout refers to it) and the draggable title bar. It places the panel in
  `area` (`"left"`, `"right"`, `"top"` or `"bottom"`), or as a tab next to an
  existing panel (`"devices"`, `"procedures"`, `"workspace"`, `"data"`,
  `"console"`, or one of yours). It raises `ValueError` for a duplicate
  name or title, or an unknown area or panel. It raises `RuntimeError` when
  called outside `add_panels()`.
- In `add_panels()`, `self.bridge` and the standard panels exist, but the
  kernel is not started yet (the timer guard `is_ready` covers that).
  The menus, the status bar, `self.activity` and `self.procedure_context`
  are not created yet: reach them from a slot that runs later, or with
  `QTimer.singleShot(0, ...)`.
- `window.docks` returns the panels by name.

For a widget that follows the theme, use `theme().tokens[...]`,
`theme().icon(name, token)` and connect to `theme().changed`, or set the
dynamic properties of the style sheet (`muted`, `heading`, `section`, `card`,
`accent`).

## Tutorial 8: test your extensions

### 8.1 Unit tests without a kernel

A `ProcedureContext` built from plain callables lets you check the code a
window would run. This is the pattern of `tests/test_gui_procedures.py`:

```python
import pytest

pytest.importorskip("qtconsole", exc_type=ImportError)


@pytest.fixture
def context(qapp):                      # qapp: tests/conftest.py (QApplication + private QSettings)
    from opticalib.gui.kernel import Task
    from opticalib.gui.procedures.base import ProcedureContext

    runs, answers = [], {}

    def run(code, title, on_done=None, on_error=None):
        task = Task(code, title, on_done=on_done, on_error=on_error)
        runs.append(task)
        return task

    def query(expressions, callback):
        callback({k: answers.get(k, RuntimeError("no answer")) for k in expressions})

    ctx = ProcedureContext(run=run, query=query, interrupt=lambda: None)
    ctx.runs, ctx.answers = runs, answers
    ctx.set_workspace([
        {"name": "dm", "type": "AcmeDM", "module": "m", "kind": "dm", "summary": ""},
        {"name": "interf", "type": "PhaseCam", "module": "m", "kind": "interferometer", "summary": ""},
    ])
    return ctx


def test_poke_code_and_state(context):
    from mylab.gui import PokeTestWindow

    window = PokeTestWindow(context=context)
    step = window.step("poke")
    assert step.code() == "tn_poke = poke_test(dm, interf, range(10), amplitude=1e-07, nframes=1)"

    context.answers["poke_tn"] = repr("20261013_120000")   # what repr(tn_poke) would return
    task = window.run_step(step, confirmed=True)
    task._start("msg"); task._finish("done")             # simulate the kernel
    assert window.state == {"poke_tn": "20261013_120000"}
    window.select_step("show")
    assert "'20261013_120000'" in window.step("show").code()
```

Outside the opticalib repository, copy the `qapp` fixture from
`tests/conftest.py`. It redirects `QSettings`, so tests never touch your real
`~/.config/ArcetriAdaptiveOptics/*.conf`.

### 8.2 End-to-end smoke test

`tests/test_gui_app.py` shows how to run a whole session in a subprocess:

- set `QT_QPA_PLATFORM=offscreen` and `AOCONF`;
- build the window;
- pump events until `window.bridge.is_ready`;
- call `window.bridge.query(...)` and `window._run(...)`;
- inspect `task.state`.

Use duck-typed fake devices defined in the kernel (a class with `set_shape`,
`get_shape` and `n_acts`) to exercise procedures without hardware or heavy
simulators. Mark such tests `@pytest.mark.integration`.

### 8.3 Running

```bash
QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES="" pytest tests/test_gui_*.py
flake8 . --count --select=E9,F63,F7,F82 --max-complexity=10 --max-line-length=127
```

## Pitfalls cheat-sheet

| Symptom | Cause | Fix |
|---|---|---|
| GUI crashes at import, or pyqtgraph segfaults | qtpy picked PyQt5, or another binding than pyqtgraph's | import `opticalib.gui` before anything that imports `qtpy`; install PySide6 |
| A step writes to the old experiment's folders after a switch | a path was copied at import time | read `opticalib.folders.X` when the function runs |
| Query result is a float instead of the string `"0.05"` | the bridge decodes JSON-looking strings | return `json.dumps(json.dumps(x))` |
| `KeyError` in an `outputs` expression or template | unescaped `{}` passed to `str.format` | double the braces |
| A widget appears in only one of two steps | a `Param` instance shared between steps | create a new `Param` per step (factory function) |
| GUI segfault when closing a window | `WA_DeleteOnClose` + `destroyed.connect(lambda ...)` | emit a `closed` signal; the owner calls `deleteLater()` |
| Workspace shows your device under *Other* | its class lacks the protocol methods | add the methods (Tutorial 4.1) |
| Startup overlay shows an error in "Loading the calpy environment" | exception in an `INIT_SCRIPTS` file (the later scripts were skipped) | fix the script, then Kernel → Restart |
| Two cards with the same name in *Procedures* | a subclassed window registered under a built-in name | use a new name, or replace the entry in the list |
| Saved layout puts panels in odd places after you change docks | stale `window/state` | bump `LAYOUT_VERSION` (in opticalib) or change `SETTINGS_APP` |
| A query never returns while an acquisition runs | the kernel runs one request at a time | expected: do not poll while `bridge.is_busy` |
