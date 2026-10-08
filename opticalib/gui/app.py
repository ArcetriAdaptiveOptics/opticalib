"""
CalpyGUI – Graphical User Interface for the Calpy / Opticalib toolchain
========================================================================

Layout
------
The main window is made of dockable panels around a central plot viewer;
panels can be moved, tabbed, floated or hidden (*View* menu), and the
layout is remembered between sessions.

* **Plots** (center): figures created in the console and arrays sent with
  ``_gui.view(array)`` or opened from the data browser, with an interactive
  image viewer and a thumbnail strip.
* **Devices**: one card per configured device and per simulator, with its
  connection status and quick actions.
* **Procedures**: windows dedicated to the bench procedures.
* **Workspace**: the variables of the IPython session.
* **Data**: the opticalib data folders, by tracking number.
* **Console**: the IPython console, identical to a ``calpy`` CLI session.

The IPython kernel runs in a separate process (see
:mod:`opticalib.gui.kernel`), so the window never freezes: long operations
are shown in the bottom-right activity panel, where they can be
interrupted, and the status bar shows the kernel state.

Author(s)
---------
- Pietro Ferraiuolo / Copilot : written in 2025
"""

import os
import shutil
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional

import qtpy

if getattr(qtpy, "QT5", False) is True:  # "is True": qtpy is mocked in the docs build
    # pyqtgraph (plot viewer) crashes with Qt 5 bindings.
    raise ImportError(
        f"CalpyGUI requires a Qt 6 binding (PySide6 or PyQt6), but qtpy selected "
        f"{qtpy.API_NAME} {qtpy.QT_VERSION}. Install PySide6-Essentials, or set "
        f"QT_API=pyside6 before starting."
    )

from qtpy.QtCore import QRectF, QSettings, Qt, QTimer, Signal
from qtpy.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QIcon,
    QKeySequence,
    QPainter,
    QPainterPath,
)
from qtpy.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QTabBar,
    QToolButton,
    QWidget,
)

from .activity import ActivityCenter, LocalJob, StartupOverlay, format_elapsed
from .kernel import KERNEL_MPL_BACKEND, KERNEL_SIDE, KernelBridge, Task
from .plugins import PLUGIN_WINDOWS, PluginPanel
from .procedures.base import ProcedureContext
from .theme import SETTINGS_APP, SETTINGS_ORG, THEME_MODES, console_style_sheet, theme
from .widgets.common import DockTitleBar, ElidedLabel
from .widgets.config_editor import ConfigEditorDialog, validate_config_text
from .widgets.data_browser import DataBrowser, load_array_file
from .widgets.device_panel import DevicePanel
from .widgets.plot_viewer import PlotViewer, load_npz_view
from .widgets.workspace import WorkspaceView

#: Version of the saved dock layout; bump it when the docks change.
LAYOUT_VERSION = 2

#: Application icon (also used by the OptiCalib desktop launchers).
ICON_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resources", "opticalib.png"
)

# Re-bind ``_gui`` in the kernel if the user deleted it (e.g. ``%reset``).
_REINSTALL_GUI = (
    "if '_gui' not in get_ipython().user_ns:\n" f"    {KERNEL_SIDE}.install()\n"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_experiment_name(config_path: str) -> str:
    """
    Derive a human-readable experiment name from the config file path.

    Uses the name of the directory that contains the configuration file
    (the experiment folder; ``SysConfig`` is skipped, as in the layout created
    by ``calpy --create``).

    Parameters
    ----------
    config_path : str
        Full path to the ``configuration.yaml`` file.

    Returns
    -------
    str
        Experiment/folder name, or an empty string when unavailable.
    """
    folder = os.path.dirname(os.path.abspath(config_path))
    if os.path.basename(folder) == "SysConfig":
        folder = os.path.dirname(folder)
    return os.path.basename(folder)


def resolve_experiment(path: str) -> str:
    """
    Return the configuration file of an experiment.

    Parameters
    ----------
    path : str
        A configuration file, or an experiment folder containing
        ``SysConfig/configuration.yaml`` (the ``calpy --create`` layout) or
        ``configuration.yaml``.

    Returns
    -------
    str
        Absolute path of the configuration file.

    Raises
    ------
    FileNotFoundError
        If no configuration file is found.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return path
    for candidate in (
        os.path.join(path, "SysConfig", "configuration.yaml"),
        os.path.join(path, "configuration.yaml"),
    ):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"No configuration file found in {path}")


def _resolve_init_file() -> Optional[str]:
    """
    Locate the ``initCalpy.py`` IPython bootstrap script.

    Searches in the installed package location and in the local development
    tree (for editable installs).

    Returns
    -------
    str or None
        Absolute path to the bootstrap script, or ``None`` when not found.
    """
    from pathlib import Path

    here = Path(__file__).resolve().parent
    candidates = [
        # Installed package layout: opticalib/gui/ -> opticalib/__init_script__/
        here / ".." / "__init_script__" / "initCalpy.py",
        # Development layout (running from repo root)
        here / ".." / ".." / "__init_script__" / "initCalpy.py",
    ]
    for path in candidates:
        resolved = path.resolve()
        if resolved.exists():
            return str(resolved)
    return None


# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------


class KernelStatus(QWidget):
    """
    Status-bar indicator of the kernel state, with the busy time.

    States are those of :attr:`KernelBridge.state_changed`.
    """

    _STYLES: Dict[str, tuple] = {
        "starting": ("Starting kernel…", "warning"),
        "idle": ("Kernel idle", "success"),
        "busy": ("Kernel busy", "accent"),
        "restarting": ("Restarting kernel…", "warning"),
        "dead": ("Kernel stopped", "danger"),
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Create the indicator in the ``dead`` state."""
        super().__init__(parent)
        self.state = "dead"
        self._busy_since: Optional[float] = None
        self._label = QLabel()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.addWidget(self._label)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._refresh)
        theme().changed.connect(self._refresh)
        self._refresh()

    def set_state(self, state: str) -> None:
        """
        Show a new kernel state.

        Parameters
        ----------
        state : str
            Kernel state.
        """
        import time

        if state == "busy" and self.state != "busy":
            self._busy_since = time.monotonic()
            self._timer.start()
        elif state != "busy":
            self._busy_since = None
            self._timer.stop()
        self.state = state
        self._refresh()

    def _refresh(self) -> None:
        import time

        label, token = self._STYLES.get(self.state, self._STYLES["dead"])
        if self._busy_since is not None:
            elapsed = time.monotonic() - self._busy_since
            if elapsed >= 1:
                label += f" · {format_elapsed(elapsed)}"
        color = theme().tokens[token]
        self._label.setText(f"<span style='color:{color}'>●</span> {label}")


class _RamGauge(QWidget):
    """Horizontal bar with two stacked segments: this session, then the rest."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Create an empty gauge."""
        super().__init__(parent)
        self.setFixedSize(150, 8)
        self.kernel = 0.0  # fractions of the total RAM: kernel + GUI
        self.others = 0.0
        self.colors = (QColor(), QColor(), QColor())  # track, kernel, others

    def paintEvent(self, event) -> None:
        """Draw the track, then the kernel and the other segments."""
        track, kernel, others = self.colors
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        rect = QRectF(self.rect())
        radius = rect.height() / 2
        clip = QPainterPath()
        clip.addRoundedRect(rect, radius, radius)
        painter.setClipPath(clip)
        painter.fillRect(rect, track)
        width = rect.width()
        k = width * min(max(self.kernel, 0.0), 1.0)
        o = min(width * max(self.others, 0.0), width - k)
        painter.fillRect(QRectF(0, 0, k, rect.height()), kernel)
        painter.fillRect(QRectF(k, 0, o, rect.height()), others)
        painter.end()


class RamBar(QWidget):
    """
    Status-bar gauge of the machine's memory.

    The bar shows the RAM in use on the whole machine, split in two
    segments: this session, kernel plus GUI (accent colour), and every
    other process.  The second segment turns orange, then red, when
    memory use approaches the limit.  Readings come from the operating
    system (``psutil``) in the GUI process every :attr:`INTERVAL_MS`, so the
    kernel is never involved.  Without ``psutil`` the widget stays hidden.

    Parameters
    ----------
    pid : callable
        Returns the process ID of the kernel (or ``None``).
    parent : QWidget, optional
        Parent widget.
    """

    #: Refresh period.
    INTERVAL_MS = 2000
    #: System memory use (fraction) above which the bar turns orange / red.
    WARNING, CRITICAL = 0.85, 0.95

    def __init__(
        self, pid: Callable[[], Optional[int]], parent: Optional[QWidget] = None
    ) -> None:
        """Create the gauge (hidden until the first reading)."""
        super().__init__(parent)
        try:
            import psutil
        except ImportError:  # pragma: no cover - psutil comes with ipykernel
            psutil = None
        self._psutil = psutil
        self._pid = pid
        self._process = None
        self._label = QLabel("RAM")
        self._label.setProperty("muted", True)
        self._gauge = _RamGauge()
        self._value = QLabel()
        self._value.setProperty("muted", True)
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 0, 6, 0)
        row.setSpacing(6)
        for widget in (self._label, self._gauge, self._value):
            row.addWidget(widget, 0, Qt.AlignmentFlag.AlignVCenter)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self._level = ""
        self._apply_level()
        self.hide()
        if psutil is not None:
            self._timer = QTimer(self)
            self._timer.setInterval(self.INTERVAL_MS)
            self._timer.timeout.connect(self.refresh)
            self._timer.start()
            theme().changed.connect(self._apply_level)

    @staticmethod
    def format_bytes(value: float) -> str:
        """``1536 MB`` → ``'1.5 GB'``."""
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024 or unit == "TB":
                return (
                    f"{value:.0f} {unit}"
                    if unit in ("B", "KB")
                    else f"{value:.1f} {unit}"
                )
            value /= 1024
        return f"{value:.1f} TB"  # pragma: no cover

    def refresh(self) -> None:
        """Read the memory of the kernel and of the system."""
        psutil = self._psutil
        pid = self._pid()
        if psutil is None or pid is None:
            self.hide()
            return
        try:
            if self._process is None or self._process.pid != pid:
                self._process = psutil.Process(pid)
            kernel = self._process.memory_info().rss
            system = psutil.virtual_memory()
            gui = psutil.Process().memory_info().rss
        except (psutil.Error, OSError):
            self._process = None
            self.hide()
            return
        self.set_reading(kernel, system.total, system.total - system.available, gui)

    def set_reading(self, kernel: int, total: int, used: int, gui: int = 0) -> None:
        """
        Show a reading.

        Parameters
        ----------
        kernel : int
            Memory of the kernel process [bytes].
        total : int
            RAM of the machine [bytes].
        used : int
            RAM in use on the whole machine, kernel included [bytes].
        gui : int, optional
            Memory of the GUI process [bytes].
        """
        fmt = self.format_bytes
        kernel = min(kernel, used)
        calpy = min(kernel + gui, used)  # this session: kernel + GUI
        fraction = used / total if total else 0.0
        self._gauge.kernel = calpy / total if total else 0.0
        self._gauge.others = (used - calpy) / total if total else 0.0
        self._gauge.update()
        self._value.setText(f"{fmt(used)} / {fmt(total)}")
        self.setToolTip(
            f"In use: {fmt(used)} of {fmt(total)} ({fraction:.0%})\n"
            f"Kernel: {fmt(kernel)}\n"
            f"GUI: {fmt(gui)}\n"
            f"Other processes: {fmt(max(used - kernel - gui, 0))}\n"
            f"Available: {fmt(max(total - used, 0))}"
        )
        level = (
            "critical"
            if fraction >= self.CRITICAL
            else "warning" if fraction >= self.WARNING else ""
        )
        if level != self._level:
            self._level = level
            self._apply_level()
        self.show()

    def _apply_level(self) -> None:
        tokens = theme().tokens
        others = {"warning": "warning", "critical": "danger"}.get(
            self._level, "text_muted"
        )
        self._gauge.colors = (
            QColor(tokens["surface_hover"]),
            QColor(tokens["accent"]),
            QColor(tokens[others]),
        )
        self._value.setStyleSheet(f"color: {tokens[others]};" if self._level else "")
        self._gauge.update()


class BackendButton(QToolButton):
    """
    Status-bar switch of the xupy array backend of the kernel.

    Bright green (glowing) when xupy creates GPU (CuPy) arrays, dim green
    when it uses the CPU (NumPy), grey when no GPU is available.  Clicking
    asks to switch backend (see :attr:`switch_requested`).

    Signals
    -------
    switch_requested(bool)
        The user asked to switch; ``True`` means to the GPU.
    """

    switch_requested = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Create the button in the unknown state."""
        super().__init__(parent)
        self.on_gpu: Optional[bool] = None
        self.available: Optional[bool] = None
        self.setText("GPU")
        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._glow = QGraphicsDropShadowEffect(self)
        self._glow.setBlurRadius(16)
        self._glow.setOffset(0, 0)
        self.setGraphicsEffect(self._glow)
        self.clicked.connect(self._on_click)
        theme().changed.connect(self._refresh)
        self._refresh()

    def set_state(self, on_gpu: Optional[bool], available: Optional[bool]) -> None:
        """
        Show the backend of the kernel.

        Parameters
        ----------
        on_gpu : bool or None
            Whether xupy uses the GPU (``None``: unknown).
        available : bool or None
            Whether a GPU (CuPy) is available.
        """
        self.on_gpu, self.available = on_gpu, available
        self._refresh()

    def _colors(self):
        t = theme()
        lit = t.color("success").lighter(125 if t.is_dark else 100)
        dead = QColor(t.tokens["success"])
        bg = QColor(t.tokens["bg"])
        # Dim green: the success color faded towards the background.
        dead = QColor(
            int(dead.red() * 0.35 + bg.red() * 0.65),
            int(dead.green() * 0.35 + bg.green() * 0.65),
            int(dead.blue() * 0.35 + bg.blue() * 0.65),
        )
        return lit, dead, t.color("text_muted")

    def _refresh(self) -> None:
        lit, dead, muted = self._colors()
        if self.on_gpu is None:
            color, glow, tip = muted, False, "xupy backend: unknown (kernel not ready)"
        elif not self.available:
            color, glow, tip = (
                muted,
                False,
                "xupy backend: CPU (NumPy). No GPU available (CuPy not found).",
            )
        elif self.on_gpu:
            color, glow, tip = (
                lit,
                True,
                "xupy backend: GPU (CuPy). Click to switch to the CPU (NumPy).",
            )
        else:
            color, glow, tip = (
                dead,
                False,
                "xupy backend: CPU (NumPy). Click to switch to the GPU (CuPy).",
            )
        self.setEnabled(bool(self.available) and self.on_gpu is not None)
        self.setIcon(
            theme().icon("chip", color_disabled=color.name(), color=color.name())
        )
        self.setStyleSheet(
            f"QToolButton {{ color: {color.name()}; font-weight: 600; }}"
        )
        self._glow.setColor(lit)
        self._glow.setEnabled(glow)
        self.setToolTip(tip + "\nArrays created before a switch are not converted.")

    def _on_click(self) -> None:
        if self.on_gpu is not None and self.available:
            self.switch_requested.emit(not self.on_gpu)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class CalpyGUI(QMainWindow):
    """
    Main window of the CalpyGUI application.

    Combines an IPython console (identical to a ``calpy`` CLI session, run in
    a separate kernel process), a plot viewer, device connection cards, the
    kernel workspace, a data browser, and the procedure windows.

    Parameters
    ----------
    config_path : str or None
        Path to the ``configuration.yaml`` file to load.  When *None* the
        default OptiCalib path (set via the ``AOCONF`` environment variable,
        or the package template) is used.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialise the main window and start the IPython kernel."""
        super().__init__()

        self._settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        if config_path is None:
            from opticalib.core.root import CONFIGURATION_FILE

            config_path = CONFIGURATION_FILE
        self._config_path: str = os.path.abspath(config_path)
        self._experiment = _get_experiment_name(self._config_path) or "opticalib"
        self.setWindowTitle(f"CalpyGUI – {self._experiment}")
        if os.path.isfile(ICON_FILE):
            self.setWindowIcon(QIcon(ICON_FILE))
        self.resize(1600, 950)

        self._tmp_dir = tempfile.mkdtemp(prefix="calpygui-")
        self._plugin_windows: List[QMainWindow] = []
        self._workspace_pending = False
        self._workspace_dirty = False
        self._overlay: Optional[StartupOverlay] = None
        # Ask before quitting while code runs (disabled by tests).
        self._confirm_close = True

        theme().apply()
        self.bridge = KernelBridge(
            self._config_path, _resolve_init_file(), self._tmp_dir, parent=self
        )
        self._build_ui()
        self._build_menus()
        self._build_status_bar()
        self.activity = ActivityCenter(
            self, interrupt=self.bridge.interrupt, cancel=self.bridge.cancel
        )
        #: Kernel access shared by the procedure windows.
        self.procedure_context = ProcedureContext(
            run=self._run,
            query=self.bridge.query,
            interrupt=self.bridge.interrupt,
            view_file=self._preview_file,
            parent=self,
        )
        self._connect_bridge()
        self._default_state = self.saveState(LAYOUT_VERSION)
        self._restore_layout_settings()
        self._update_dock_titles()
        theme().changed.connect(self._apply_theme)
        self._apply_theme()

        self._remember_experiment(self._config_path)
        self._show_overlay()
        QTimer.singleShot(0, self.bridge.start)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Create the central viewer and the dock panels."""
        self.setDockNestingEnabled(True)
        self.tabifiedDockWidgetActivated.connect(
            lambda dock: self._schedule_dock_titles()
        )
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(
            Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea
        )

        self.plot_viewer = PlotViewer()
        self.setCentralWidget(self.plot_viewer)

        self.device_panel = DevicePanel(self._config_path, runner=self._run)
        self.plugin_panel = PluginPanel(selection_callback=self._open_plugin_window)
        self.workspace_view = WorkspaceView()
        self.data_browser = DataBrowser()

        self._docks: Dict[str, QDockWidget] = {}
        devices = self._make_dock("devices", "Devices", self.device_panel)
        procedures = self._make_dock("procedures", "Procedures", self.plugin_panel)
        workspace = self._make_dock("workspace", "Workspace", self.workspace_view)
        data = self._make_dock("data", "Data", self.data_browser)
        console = self._make_dock("console", "Console", self.bridge.console)

        left = Qt.DockWidgetArea.LeftDockWidgetArea
        self.addDockWidget(left, devices, Qt.Orientation.Vertical)
        self.addDockWidget(left, workspace, Qt.Orientation.Vertical)
        self.tabifyDockWidget(devices, procedures)
        self.tabifyDockWidget(workspace, data)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, console)
        devices.raise_()
        workspace.raise_()
        self.resizeDocks([devices, console], [360, 620], Qt.Orientation.Horizontal)
        self.resizeDocks([devices, workspace], [520, 360], Qt.Orientation.Vertical)

        self.device_panel.edit_config_requested.connect(self._edit_config_entry)
        self.workspace_view.run_requested.connect(
            lambda code, title: self._run(code, title)
        )
        self.data_browser.run_requested.connect(
            lambda code, title: self._run(code, title)
        )
        self.data_browser.preview_requested.connect(self._preview_file)

        self._workspace_timer = QTimer(self)
        self._workspace_timer.setSingleShot(True)
        self._workspace_timer.setInterval(150)
        self._workspace_timer.timeout.connect(self._refresh_workspace)

    def _make_dock(self, name: str, title: str, widget: QWidget) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{name}")
        dock.setWidget(widget)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        dock.setTitleBarWidget(DockTitleBar(dock))
        self._docks[name] = dock
        # The tab groups change when the user moves, floats or closes panels.
        dock.dockLocationChanged.connect(lambda area: self._schedule_dock_titles())
        dock.topLevelChanged.connect(lambda floating: self._schedule_dock_titles())
        dock.visibilityChanged.connect(lambda visible: self._schedule_dock_titles())
        return dock

    def _schedule_dock_titles(self) -> None:
        QTimer.singleShot(0, self._update_dock_titles)

    def _update_dock_titles(self) -> None:
        """
        Show the panel names only where no tab shows them.

        A tabbed panel is named by its (selected) tab, so its title bar
        shows no text; panels alone or floating show their name.  The title
        bars stay, so every panel can still be dragged out or floated.
        """
        for dock in self._docks.values():
            tabbed = not dock.isFloating() and any(
                not other.isHidden() for other in self.tabifiedDockWidgets(dock)
            )
            bar = dock.titleBarWidget()
            if isinstance(bar, DockTitleBar):
                bar.set_title("" if tabbed else dock.windowTitle())
                bar._refresh_icons()
        self._hide_stale_tab_bars()

    def _hide_stale_tab_bars(self) -> None:
        """
        Hide the leftover tab bars Qt leaves on screen.

        With several groups of tabbed panels in one dock area, QMainWindow
        keeps stale tab bars visible (drawn as stray lines across the
        window).  A tab bar in use sits right on top of the panel of its
        current tab, with the same position and width; the others are hidden.
        """
        if not self.isVisible():
            return  # geometries are not final yet
        by_title = {dock.windowTitle(): dock for dock in self._docks.values()}
        # Newest first: after restoreState() the bars in use are the newest
        # ones, and older copies may sit exactly on top of the same panel.
        kept = set()
        bars = [c for c in self.children() if isinstance(c, QTabBar)]
        for bar in reversed(bars):
            if not bar.isVisible() or bar.count() == 0:
                continue
            key = bar.geometry().getRect()
            if key in kept:
                bar.hide()  # an older copy of a bar in use
                continue
            geo = bar.geometry()
            attached = False
            for index in range(bar.count()):
                dock = by_title.get(bar.tabText(index))
                if dock is None or dock.isHidden() or dock.isFloating():
                    continue
                dg = dock.geometry()
                aligned = (
                    abs(dg.left() - geo.left()) <= 2
                    and abs(dg.width() - geo.width()) <= 4
                )
                touching = (
                    abs(dg.top() - (geo.bottom() + 1)) <= 12
                    or abs(geo.top() - (dg.bottom() + 1)) <= 12
                )
                if aligned and touching:
                    attached = True
                    break
            if attached:
                kept.add(key)
            else:
                bar.hide()

    # Qt event handler: the camelCase name is required for Qt to call it.
    def showEvent(self, event) -> None:  # noqa: N802
        """Tidy the dock tab bars once the window has its final geometry."""
        super().showEvent(event)
        self._schedule_dock_titles()

    def _build_menus(self) -> None:
        """Create the menu bar."""
        bar = self.menuBar()

        file_menu = bar.addMenu("&File")
        open_experiment = file_menu.addAction(
            theme().icon("folder-open-outline"),
            "Open experiment…",
            self._choose_experiment,
        )
        open_experiment.setShortcut(QKeySequence.StandardKey.Open)
        file_menu.addAction("Open configuration file…", self._choose_configuration_file)
        self._recent_menu = file_menu.addMenu(
            theme().icon("history"), "Recent experiments"
        )
        self._recent_menu.aboutToShow.connect(self._fill_recent_menu)
        file_menu.addSeparator()
        self._action_config = file_menu.addAction(
            "Edit configuration…", self._view_config
        )
        self._action_config.setShortcut(QKeySequence("Ctrl+,"))
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit", self.close)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)

        kernel_menu = bar.addMenu("&Kernel")
        self._action_interrupt = kernel_menu.addAction(
            "Interrupt", self.bridge.interrupt
        )
        self._action_interrupt.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self._action_restart = kernel_menu.addAction("Restart…", self._restart_kernel)
        self._action_restart.setShortcut(QKeySequence("Ctrl+Shift+R"))
        kernel_menu.addSeparator()
        plots_menu = kernel_menu.addMenu("Plots")
        group = QActionGroup(self)
        self._plot_mode_actions: Dict[str, QAction] = {}
        for mode, label in (
            ("panel", "Show figures in the plot panel"),
            ("windows", "Show figures in interactive windows"),
        ):
            action = plots_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(mode == "panel")
            action.triggered.connect(
                lambda checked=False, m=mode: self._set_plot_mode(m)
            )
            group.addAction(action)
            self._plot_mode_actions[mode] = action

        view_menu = bar.addMenu("&View")
        for dock in self._docks.values():
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("Theme")
        theme_group = QActionGroup(self)
        for mode in THEME_MODES:
            action = theme_menu.addAction(mode.capitalize())
            action.setCheckable(True)
            action.setChecked(theme().mode == mode)
            action.triggered.connect(lambda checked=False, m=mode: theme().set_mode(m))
            theme_group.addAction(action)
        view_menu.addAction("Reset layout", self._reset_layout)

        help_menu = bar.addMenu("&Help")
        help_menu.addAction("About CalpyGUI", self._about)

    def _build_status_bar(self) -> None:
        """Create the status bar (kernel state, actions, configuration)."""
        status = self.statusBar()
        self.kernel_status = KernelStatus()
        self._btn_interrupt = QToolButton()
        self._btn_interrupt.setToolTip("Interrupt the running code (Ctrl+Shift+C)")
        self._btn_interrupt.clicked.connect(self.bridge.interrupt)
        self._btn_restart = QToolButton()
        self._btn_restart.setToolTip("Restart the kernel (Ctrl+Shift+R)")
        self._btn_restart.clicked.connect(self._restart_kernel)
        status.addWidget(self.kernel_status)
        status.addWidget(self._btn_interrupt)
        status.addWidget(self._btn_restart)

        from opticalib import __version__

        self._config_label = ElidedLabel(self._config_path)
        self._config_label.setProperty("muted", True)
        self._config_label.setMinimumWidth(220)
        self._config_label.setMaximumWidth(520)
        self._config_label.setToolTip(
            f"{self._config_path}\nClick to edit the configuration"
        )
        self._config_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._config_label.clicked.connect(self._view_config)
        version = QLabel(f"opticalib {__version__}")
        version.setProperty("muted", True)
        self.backend_button = BackendButton()
        self.backend_button.switch_requested.connect(self._switch_backend)
        self._btn_switch_experiment = QToolButton()
        self._btn_switch_experiment.setAutoRaise(True)
        self._btn_switch_experiment.setToolTip("Switch to another experiment (Ctrl+O)")
        self._btn_switch_experiment.clicked.connect(self._choose_experiment)
        self.ram_bar = RamBar(lambda: self.bridge.kernel_pid)
        # Centred in the free space between the left and the right widgets.
        slot = QWidget()
        row = QHBoxLayout(slot)
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        row.addWidget(self.ram_bar)
        row.addStretch(1)
        status.addWidget(slot, 1)
        status.addPermanentWidget(self.backend_button)
        status.addPermanentWidget(self._btn_switch_experiment)
        status.addPermanentWidget(self._config_label)
        status.addPermanentWidget(version)

    def _apply_theme(self) -> None:
        """Update the parts that are not styled by the style sheet."""
        self._btn_switch_experiment.setIcon(
            theme().icon("folder-swap-outline", "text_muted")
        )
        t = theme()
        self._btn_interrupt.setIcon(t.icon("stop-circle-outline"))
        self._btn_restart.setIcon(t.icon("restart"))
        console = self.bridge.console
        console.syntax_style = "monokai" if t.is_dark else "default"
        console.style_sheet = console_style_sheet(t.tokens)
        console._syntax_style_changed()
        console._style_sheet_changed()

    def _show_overlay(self) -> None:
        self._overlay = StartupOverlay(
            KernelBridge.BOOTSTRAP_STEPS,
            "CalpyGUI",
            f"Experiment: {self._experiment}\n{self._config_path}",
            parent=self,
        )
        self._overlay.closed.connect(self._on_overlay_closed)

    def _on_overlay_closed(self) -> None:
        self._overlay = None

    # ------------------------------------------------------------------
    # Layout persistence
    # ------------------------------------------------------------------

    def _restore_layout_settings(self) -> None:
        """Restore window geometry and dock layout from QSettings."""
        geometry = self._settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self._settings.value("window/state")
        if state is not None:
            self.restoreState(state, LAYOUT_VERSION)

    def _save_layout_settings(self) -> None:
        """Save window geometry and dock layout to QSettings."""
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("window/state", self.saveState(LAYOUT_VERSION))
        self._settings.sync()

    def _reset_layout(self) -> None:
        """Restore the default dock layout and activity panel position."""
        self.restoreState(self._default_state, LAYOUT_VERSION)
        for dock in self._docks.values():
            dock.show()
        self.activity.reset_position()
        self._schedule_dock_titles()

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def _connect_bridge(self) -> None:
        bridge = self.bridge
        bridge.state_changed.connect(self._on_kernel_state)
        bridge.bootstrap_step.connect(self._on_bootstrap_step)
        bridge.ready.connect(self._on_kernel_ready)
        bridge.task_added.connect(self.activity.track)
        bridge.execution_finished.connect(self._workspace_timer.start)
        bridge.console.figure_received.connect(self.plot_viewer.add_figure)
        bridge.console.image_received.connect(self._on_image_received)
        bridge.restart_requested.connect(self._restart_kernel)
        bridge.kernel_died.connect(self._on_kernel_died)

    def _run(
        self,
        code: str,
        title: Optional[str] = None,
        on_done=None,
        on_error=None,
    ) -> Task:
        """
        Execute *code* in the console (see :meth:`KernelBridge.run`).

        Parameters
        ----------
        code : str
            Python / IPython source.
        title : str, optional
            Description shown in the activity panel.
        on_done, on_error : callable, optional
            Called with the task when it succeeds or fails.

        Returns
        -------
        Task
            The queued task.
        """
        return self.bridge.run(code, title, on_done=on_done, on_error=on_error)

    def _on_kernel_state(self, state: str) -> None:
        self.kernel_status.set_state(state)
        running = state == "busy"
        self._btn_interrupt.setEnabled(running)
        self._action_interrupt.setEnabled(running)

    def _on_bootstrap_step(self, key: str, status: str, message: str) -> None:
        if self._overlay is not None:
            self._overlay.set_step(key, status, message)

    def _on_kernel_ready(self) -> None:
        if self._overlay is not None:
            self._overlay.finish()
        self._refresh_folders()
        self._refresh_workspace()

    def _restart_kernel(self) -> None:
        answer = QMessageBox.question(
            self,
            "Restart kernel",
            "Restart the kernel? All variables and device connections are lost.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        # A fresh overlay: the previous one may still show an old error.
        if self._overlay is not None:
            self._overlay.dismiss()
        self._show_overlay()
        self._overlay.set_step("kernel", "running")
        self.workspace_view.set_items([])
        self.device_panel.update_workspace([])
        self.backend_button.set_state(None, None)
        self.bridge.restart()

    def _switch_backend(self, to_gpu: bool) -> None:
        """Switch the xupy backend of the kernel (arrays created later)."""
        target = "gpu" if to_gpu else "cpu"
        self._run(
            f"xp.use_{target}()",  ## No need, as initCalpy imports xp
            f"Switching `xupy` to the {target.upper()}",
        )

    def _on_kernel_died(self, reason: str) -> None:
        self.backend_button.set_state(None, None)
        self.workspace_view.set_items([])
        self.device_panel.update_workspace([])
        QMessageBox.critical(
            self,
            "Kernel stopped",
            f"{reason}\n\nUse Kernel → Restart to start a new one.",
        )

    def _set_plot_mode(self, mode: str) -> None:
        """Show kernel figures in the plot panel or in interactive windows."""
        if mode == "windows":
            code = "%matplotlib qt"
        else:
            code = f"import matplotlib.pyplot as plt\nplt.switch_backend({KERNEL_MPL_BACKEND!r})"
        self._run(code, "Changing the plot mode")

    def _refresh_workspace(self) -> None:
        if not self.bridge.is_ready:
            return
        if self._workspace_pending:
            self._workspace_dirty = True
            return
        self._workspace_pending = True
        self.bridge.query(
            {"ws": f"{KERNEL_SIDE}.workspace()", "xp": f"{KERNEL_SIDE}.backend()"},
            self._on_workspace,
            code=_REINSTALL_GUI,
        )

    def _on_workspace(self, results: Dict[str, Any]) -> None:
        self._workspace_pending = False
        backend = results.get("xp")
        if isinstance(backend, dict):
            self.backend_button.set_state(
                backend.get("on_gpu"), backend.get("available")
            )
        items = results.get("ws")
        if isinstance(items, list):
            self.workspace_view.set_items(items)
            self.device_panel.update_workspace(items)
            self.procedure_context.set_workspace(items)
        if self._workspace_dirty:
            self._workspace_dirty = False
            self._workspace_timer.start()

    def _refresh_folders(self) -> None:
        self.bridge.query({"fo": f"{KERNEL_SIDE}.folders()"}, self._on_folders)

    def _on_folders(self, results: Dict[str, Any]) -> None:
        info = results.get("fo")
        if isinstance(info, dict):
            self.data_browser.set_folders(info)
            self.procedure_context.set_folders(info)

    # ------------------------------------------------------------------
    # Plots and data
    # ------------------------------------------------------------------

    def _on_image_received(self, payload: Dict[str, Any]) -> None:
        title = str(payload.get("title", "array"))
        job = LocalJob(
            lambda path=payload.get("path"): load_npz_view(path),
            f"Loading {title}",
            on_done=lambda j, t=title: self.plot_viewer.add_data(
                j.result, t, "console"
            ),
            on_error=self._on_local_job_failed,
        )
        job.quiet = True
        job.start()

    def _preview_file(self, path: str) -> None:
        name = os.path.basename(path)
        job = LocalJob(
            lambda: load_array_file(path),
            f"Opening {name}",
            on_done=lambda j: self.plot_viewer.add_data(j.result, name, path),
            on_error=self._on_local_job_failed,
        )
        self.activity.track(job)
        job.start()

    def _on_local_job_failed(self, job: LocalJob) -> None:
        if job.quiet:
            error = job.error or {}
            QMessageBox.warning(
                self, job.title, f"{error.get('ename')}: {error.get('evalue')}"
            )

    # ------------------------------------------------------------------
    # Configuration file actions
    # ------------------------------------------------------------------

    def _view_config(self) -> None:
        """Open the configuration file in the in-app editor dialog."""
        dlg = ConfigEditorDialog(
            self._config_path, on_saved=self._on_config_saved, parent=self
        )
        dlg.exec()
        dlg.deleteLater()

    def _edit_config_entry(self, section: str, name: str) -> None:
        """Open the configuration editor on a ``DEVICES`` entry."""
        dlg = ConfigEditorDialog(
            self._config_path, on_saved=self._on_config_saved, parent=self
        )
        dlg.goto_entry(section, name)
        dlg.exec()
        dlg.deleteLater()

    def _on_config_saved(self) -> None:
        """Refresh the device panel and reload the configuration in the kernel."""
        self.device_panel.reload()
        self._run(
            "import opticalib\n"
            f"opticalib.set_configuration_file({self._config_path!r})",
            "Reloading the configuration",
            on_done=lambda task: self._refresh_folders(),
        )

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    #: Number of experiments remembered in File → Recent experiments.
    MAX_RECENT = 8

    @property
    def config_path(self) -> str:
        """The configuration file of the current experiment."""
        return self._config_path

    def _choose_experiment(self) -> None:
        """Ask for an experiment folder and switch to it."""
        start = os.path.dirname(os.path.dirname(self._config_path))
        folder = QFileDialog.getExistingDirectory(self, "Open experiment", start)
        if folder:
            self.switch_experiment(folder)

    def _choose_configuration_file(self) -> None:
        """Ask for a configuration file and switch to it."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open configuration file",
            os.path.dirname(self._config_path),
            "Configuration (*.yaml *.yml)",
        )
        if path:
            self.switch_experiment(path)

    def recent_experiments(self) -> List[str]:
        """Configuration files of the recent experiments (newest first)."""
        value = self._settings.value("recent/experiments", [])
        if isinstance(value, str):
            value = [value]
        return [p for p in (value or []) if isinstance(p, str) and os.path.isfile(p)]

    def _remember_experiment(self, config_path: str) -> None:
        recent = [
            p
            for p in self.recent_experiments()
            if os.path.normpath(p) != os.path.normpath(config_path)
        ]
        self._settings.setValue(
            "recent/experiments", [config_path] + recent[: self.MAX_RECENT - 1]
        )

    def _fill_recent_menu(self) -> None:
        menu = self._recent_menu
        menu.clear()
        recent = self.recent_experiments()
        if not recent:
            menu.addAction("No recent experiments").setEnabled(False)
            return
        for path in recent:
            action = menu.addAction(f"{_get_experiment_name(path)}  —  {path}")
            action.setCheckable(True)
            action.setChecked(
                os.path.normpath(path) == os.path.normpath(self._config_path)
            )
            action.triggered.connect(
                lambda checked=False, p=path: self.switch_experiment(p)
            )
        menu.addSeparator()
        menu.addAction(
            "Clear list", lambda: self._settings.remove("recent/experiments")
        )

    def switch_experiment(self, path: str, confirm: bool = True) -> Optional[Task]:
        """
        Switch the session to another experiment, without restarting.

        The kernel runs ``opticalib.set_configuration_file`` (visible in the
        console); when it succeeds the GUI follows: window title, status bar,
        devices, data folders, procedure windows and the environment of the
        next kernel restarts.  Devices already connected keep the
        configuration they were created with.

        Parameters
        ----------
        path : str
            Experiment folder or configuration file.
        confirm : bool, optional
            Ask before switching.

        Returns
        -------
        Task or None
            The switching task, or ``None`` if nothing was switched.
        """
        try:
            config_path = resolve_experiment(path)
            with open(config_path, "r") as f:
                error = validate_config_text(f.read())
        except OSError as exc:
            QMessageBox.warning(self, "Open experiment", str(exc))
            return None
        if error is not None:
            QMessageBox.warning(
                self,
                "Open experiment",
                f"Invalid configuration file:\n{config_path}\n\n{error}",
            )
            return None
        if os.path.normpath(config_path) == os.path.normpath(self._config_path):
            return None
        name = _get_experiment_name(config_path)
        if confirm:
            connected = [
                i["name"]
                for i in self.workspace_view.items
                if i.get("kind") in ("dm", "interferometer", "wfs", "camera")
            ]
            message = f"Switch to the experiment '{name}'?\n\n{config_path}"
            if connected:
                message += (
                    "\n\nConnected devices ("
                    + ", ".join(connected)
                    + ") keep the configuration "
                    "they were created with: reconnect them to use the new one."
                )
            answer = QMessageBox.question(self, "Open experiment", message)
            if answer != QMessageBox.StandardButton.Yes:
                return None
        return self._run(
            f"import opticalib\nopticalib.set_configuration_file({config_path!r})",
            f"Switching to the experiment {name}",
            on_done=lambda task, p=config_path: self._apply_experiment(p),
            on_error=lambda task: QMessageBox.warning(
                self,
                "Open experiment",
                f"The experiment could not be loaded:\n{(task.error or {}).get('evalue', '')}",
            ),
        )

    def _apply_experiment(self, config_path: str) -> None:
        """Make the GUI follow the experiment the kernel switched to."""
        self._config_path = config_path
        self._experiment = _get_experiment_name(config_path) or "opticalib"
        self.setWindowTitle(f"CalpyGUI – {self._experiment}")
        self._config_label.setText(config_path)
        self._config_label.setToolTip(f"{config_path}\nClick to edit the configuration")
        self.bridge.set_config_path(config_path)
        self.device_panel.set_config_path(config_path)
        self._remember_experiment(config_path)
        self._refresh_folders()
        self._refresh_workspace()

    # ------------------------------------------------------------------
    # Plugins
    # ------------------------------------------------------------------

    def _open_plugin_window(self, plugin_name: str) -> None:
        """
        Open the GUI window associated with the selected plugin.

        Parameters
        ----------
        plugin_name : str
            Display label selected in the plugin panel.
        """
        window_cls = PLUGIN_WINDOWS.get(plugin_name)
        if window_cls is None:
            QMessageBox.warning(
                self,
                "Unknown plugin",
                f"No GUI mapping found for '{plugin_name}'.",
            )
            return

        window = window_cls(context=self.procedure_context, parent=self)
        # A bound method (not a lambda) is disconnected when the window dies.
        window.closed.connect(self._on_plugin_window_closed)
        self._plugin_windows.append(window)
        window.show()
        window.raise_()
        window.activateWindow()

    def _on_plugin_window_closed(self, window: QMainWindow) -> None:
        """
        Drop references to closed plugin windows.

        Parameters
        ----------
        window : QMainWindow
            Plugin window that has just been closed.
        """
        self._plugin_windows = [w for w in self._plugin_windows if w is not window]
        window.deleteLater()

    def _about(self) -> None:
        from opticalib import __version__

        QMessageBox.about(
            self,
            "About CalpyGUI",
            f"<b>CalpyGUI</b><br>opticalib {__version__}<br><br>"
            "Graphical interface for the calpy / opticalib toolchain.<br>"
            "Arcetri Adaptive Optics group.",
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    # Qt event handler: the camelCase name is required for Qt to call it.
    def closeEvent(self, event) -> None:  # noqa: N802
        """Save the layout and stop the kernel when the window is closed."""
        busy = self.bridge.is_busy or bool(self.bridge.pending_tasks)
        if busy and self._confirm_close:
            answer = QMessageBox.question(
                self,
                "Quit CalpyGUI",
                "Code is still running in the kernel (e.g. an acquisition).\n"
                "Quit anyway? The running code is interrupted.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._save_layout_settings()
        for window in list(self._plugin_windows):
            window.close()
        if busy:
            self.bridge.interrupt()
        self.bridge.shutdown(now=busy)
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Public launch function
# ---------------------------------------------------------------------------


def launch_gui(config_path: Optional[str] = None) -> None:
    """
    Launch the CalpyGUI application.

    Creates (or reuses) a :class:`QApplication` instance, instantiates
    the main window, and enters the Qt event loop.  This function blocks
    until the window is closed.

    Parameters
    ----------
    config_path : str or None
        Path to the ``configuration.yaml`` file.  When *None* the default
        path resolved by OptiCalib at import time is used (either the
        ``AOCONF`` environment variable or the package template file).
    """
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("CalpyGUI")
    app.setApplicationDisplayName("CalpyGUI")
    # Lets Linux desktops match the window to the OptiCalib launcher.
    app.setDesktopFileName("OptiCalib")
    if os.path.isfile(ICON_FILE):
        app.setWindowIcon(QIcon(ICON_FILE))
    theme().apply(app)
    window = CalpyGUI(config_path=config_path)
    window.show()
    sys.exit(app.exec())
