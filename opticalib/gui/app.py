"""
CalpyGUI – Graphical User Interface for the Calpy / Opticalib toolchain
========================================================================

Layout
------
The window is divided into two columns:

* **Left column** (≈60 % width)

  * Two pink buttons at the top: *View configuration file* and
    *Edit configuration file*.
  * A large plotting area that captures every matplotlib figure produced
    inside the embedded IPython session.  Navigation arrows let the user
    cycle through all figures; *Clear plot* and *Clear all plots* remove
    individual or all stored figures.

* **Right column** (≈40 % width)

  * **Device panel** (top): one connect-button per device whose YAML
    configuration block contains at least one non-empty field.  Pressing
    a button injects the corresponding instantiation command into the
    terminal.
  * **IPython terminal** (bottom): a full ``qtconsole`` widget backed by
    an in-process IPython kernel.  The kernel is initialised with the
    same ``initCalpy.py`` bootstrap script used by the ``calpy`` CLI tool
    and with the ``AOCONF`` environment variable pointing at the chosen
    configuration file.

Author(s)
---------
- Pietro Ferraiuolo / Copilot : written in 2025
"""

import io
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = ""
) -> Dict[str, Any]:
    """
    Recursively flatten a nested dict into a single-level dict.

    Parameters
    ----------
    d : dict
        The dictionary to flatten.
    parent_key : str, optional
        Prefix prepended to every key in the result.

    Returns
    -------
    dict
        Flattened dictionary where nested keys are joined with ``'.'``.
    """
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


def _is_connectable(device_conf: Any) -> bool:
    """
    Return ``True`` when *device_conf* contains at least one non-empty field.

    A device is considered *connectable* when its YAML configuration block
    has at least one value that is not ``None``, not an empty string, and
    not an empty collection.

    Parameters
    ----------
    device_conf : Any
        The value associated with a device name in the ``DEVICES`` section
        of the configuration YAML file.

    Returns
    -------
    bool
        Whether the device has at least one configured (non-empty) field.
    """
    if not isinstance(device_conf, dict):
        return device_conf not in (None, "", [], {})
    flat = _flatten_dict(device_conf)
    return any(v not in (None, "", [], {}) for v in flat.values())


def _get_connectable_devices(
    config_path: str,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Parse *config_path* and return a list of connectable devices.

    Parameters
    ----------
    config_path : str
        Full path to the ``configuration.yaml`` file.

    Returns
    -------
    list of (device_type, device_name, device_conf) tuples
        Each entry describes one connectable device.  ``device_type`` is
        the top-level YAML key (e.g. ``'INTERFEROMETER'``), ``device_name``
        is the name of the device entry, and ``device_conf`` is its raw
        YAML sub-dictionary.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        return []

    devices_section = config.get("DEVICES", {})
    result: List[Tuple[str, str, Dict[str, Any]]] = []
    for dev_type, dev_dict in devices_section.items():
        if not isinstance(dev_dict, dict):
            continue
        for dev_name, dev_conf in dev_dict.items():
            if _is_connectable(dev_conf):
                result.append((dev_type, dev_name, dev_conf))
    return result


def _get_experiment_name(config_path: str) -> str:
    """
    Derive a human-readable experiment name from the config file path.

    Uses the name of the directory that contains the configuration file
    (usually the experiment folder).

    Parameters
    ----------
    config_path : str
        Full path to the ``configuration.yaml`` file.

    Returns
    -------
    str
        Experiment/folder name, or an empty string when unavailable.
    """
    return os.path.basename(os.path.dirname(os.path.abspath(config_path)))


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
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # Installed package layout: opticalib/gui/ -> opticalib/__init_script__/
        os.path.join(here, "..", "__init_script__", "initCalpy.py"),
        # Development layout (running from repo root)
        os.path.join(here, "..", "..", "__init_script__", "initCalpy.py"),
    ]
    for path in candidates:
        resolved = os.path.normpath(path)
        if os.path.exists(resolved):
            return resolved
    return None


# ---------------------------------------------------------------------------
# Plot Panel
# ---------------------------------------------------------------------------


class PlotPanel(QFrame):
    """
    Widget that displays matplotlib figures captured from the IPython session.

    Figures are stored as PNG byte strings.  Navigation arrows allow the
    user to cycle through all figures produced during the session.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialise the plot panel with an empty figure list."""
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Sunken)
        self.setStyleSheet("background-color: white;")

        # List of PNG byte strings, one per captured figure
        self._figures: List[bytes] = []
        self._current_index: int = 0

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the internal layout (image area + navigation bar)."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Image display area
        self._image_label = QLabel("No plots yet")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self._image_label.setStyleSheet("color: gray; font-size: 14px;")
        layout.addWidget(self._image_label)

        # Navigation / control bar
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(0, 0, 0, 0)

        self._btn_prev = QPushButton("◀")
        self._btn_prev.setFixedWidth(36)
        self._btn_prev.setToolTip("Previous plot")
        self._btn_prev.clicked.connect(self._go_prev)

        self._counter_label = QLabel("0 / 0")
        self._counter_label.setAlignment(Qt.AlignCenter)
        self._counter_label.setFixedWidth(60)

        self._btn_next = QPushButton("▶")
        self._btn_next.setFixedWidth(36)
        self._btn_next.setToolTip("Next plot")
        self._btn_next.clicked.connect(self._go_next)

        self._btn_clear = QPushButton("Clear plot")
        self._btn_clear.setToolTip("Remove the current plot from the panel")
        self._btn_clear.clicked.connect(self._clear_current)

        self._btn_clear_all = QPushButton("Clear all plots")
        self._btn_clear_all.setToolTip("Remove all plots from the panel")
        self._btn_clear_all.clicked.connect(self._clear_all)

        nav_layout.addWidget(self._btn_prev)
        nav_layout.addWidget(self._counter_label)
        nav_layout.addWidget(self._btn_next)
        nav_layout.addStretch()
        nav_layout.addWidget(self._btn_clear)
        nav_layout.addWidget(self._btn_clear_all)
        layout.addLayout(nav_layout)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_figure(self, png_bytes: bytes) -> None:
        """
        Append a new figure (PNG bytes) to the panel and display it.

        Parameters
        ----------
        png_bytes : bytes
            PNG-encoded figure data.
        """
        self._figures.append(png_bytes)
        self._current_index = len(self._figures) - 1
        self._refresh_display()

    def update_figure(self, index: int, png_bytes: bytes) -> None:
        """
        Replace the figure at *index* with new PNG bytes.

        Parameters
        ----------
        index : int
            Zero-based position of the figure to replace.
        png_bytes : bytes
            Updated PNG-encoded figure data.
        """
        if 0 <= index < len(self._figures):
            self._figures[index] = png_bytes
            if self._current_index == index:
                self._refresh_display()

    # ------------------------------------------------------------------
    # Navigation callbacks
    # ------------------------------------------------------------------

    def _go_prev(self) -> None:
        """Navigate to the previous figure."""
        if self._figures and self._current_index > 0:
            self._current_index -= 1
            self._refresh_display()

    def _go_next(self) -> None:
        """Navigate to the next figure."""
        if self._figures and self._current_index < len(self._figures) - 1:
            self._current_index += 1
            self._refresh_display()

    def _clear_current(self) -> None:
        """Remove the currently displayed figure from the panel."""
        if self._figures:
            del self._figures[self._current_index]
            self._current_index = max(0, self._current_index - 1)
            self._refresh_display()

    def _clear_all(self) -> None:
        """Remove all figures from the panel."""
        self._figures.clear()
        self._current_index = 0
        self._refresh_display()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_display(self) -> None:
        """Update the image label and counter to reflect the current state."""
        n = len(self._figures)
        if n == 0:
            self._image_label.setPixmap(QPixmap())
            self._image_label.setText("No plots yet")
            self._counter_label.setText("0 / 0")
            return

        self._counter_label.setText(f"{self._current_index + 1} / {n}")
        png_bytes = self._figures[self._current_index]
        pixmap = QPixmap()
        pixmap.loadFromData(png_bytes)
        scaled = pixmap.scaled(
            self._image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
        self._image_label.setText("")

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Re-scale the current figure when the widget is resized."""
        super().resizeEvent(event)
        self._refresh_display()


# ---------------------------------------------------------------------------
# Device Panel
# ---------------------------------------------------------------------------


class DevicePanel(QGroupBox):
    """
    Panel containing one *Connect* button per connectable device.

    A device is listed here when its YAML configuration block contains at
    least one non-empty field (indicating that connection details have been
    filled in).  Pressing a button injects an instantiation command into the
    embedded IPython terminal.

    Parameters
    ----------
    config_path : str
        Full path to the ``configuration.yaml`` file.
    terminal_callback : callable
        Function that accepts a command string and executes it in the
        embedded IPython terminal.
    parent : QWidget, optional
        Parent widget.
    """

    # Mapping from (YAML section, device-name prefix) to opticalib class name.
    # Used to generate ready-to-run connect commands.
    _NAME_CLASS_MAP: Dict[str, str] = {
        # Interferometers
        "PhaseCam": "devices.PhaseCam",
        "AccuFiz": "devices.AccuFiz",
        "4DProcesser": "devices.Processer4D",
        # Deformable mirrors
        "Alpao": "devices.AlpaoDm",
        "Petal": "devices.PetalMirror",
        "Splatt": "devices.SplattDm",
        "DP": "devices.DP",
        "M4AU": "devices.M4AU",
        # Cameras
        "AVT": "devices.AVTCamera",
    }

    def __init__(
        self,
        config_path: str,
        terminal_callback,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialise the device panel and populate device buttons."""
        super().__init__("Available Devices", parent)
        self._config_path = config_path
        self._terminal_callback = terminal_callback
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build a scrollable list of device buttons."""
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        self._btn_layout = QVBoxLayout(container)
        self._btn_layout.setAlignment(Qt.AlignTop)
        self._btn_layout.setSpacing(4)

        self._populate_buttons()

        scroll.setWidget(container)
        outer_layout.addWidget(scroll)

    def _populate_buttons(self) -> None:
        """Create one button per connectable device found in the config."""
        devices = _get_connectable_devices(self._config_path)
        if not devices:
            label = QLabel("No configured devices found.")
            label.setStyleSheet("color: gray; font-style: italic;")
            self._btn_layout.addWidget(label)
            return

        for dev_type, dev_name, dev_conf in devices:
            btn = QPushButton(f"Connect  {dev_name}")
            tooltip_lines = [f"Type: {dev_type}"]
            tooltip_lines += [
                f"  {k}: {v}"
                for k, v in _flatten_dict(dev_conf).items()
            ]
            btn.setToolTip("\n".join(tooltip_lines))
            cmd = self._build_connect_command(dev_type, dev_name)
            btn.clicked.connect(
                lambda checked, c=cmd: self._terminal_callback(c)
            )
            self._btn_layout.addWidget(btn)

    # ------------------------------------------------------------------
    # Command generation
    # ------------------------------------------------------------------

    def _build_connect_command(
        self, dev_type: str, dev_name: str
    ) -> str:
        """
        Build an IPython command string that instantiates the device.

        The command is generated by matching *dev_name* against known
        name prefixes.  If no match is found a helpful comment is returned
        instead.

        Parameters
        ----------
        dev_type : str
            YAML device category (e.g. ``'INTERFEROMETER'``).
        dev_name : str
            Device name as it appears in the YAML file.

        Returns
        -------
        str
            A Python / IPython expression ready to be executed in the
            embedded terminal.
        """
        var_name = (
            dev_name.lower()
            .replace(".", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )
        # Look for a matching prefix
        class_name: Optional[str] = None
        for prefix, cls in self._NAME_CLASS_MAP.items():
            if dev_name.startswith(prefix):
                class_name = cls
                break

        if class_name:
            return (
                f"import opticalib.devices as devices\n"
                f"{var_name} = {class_name}()"
            )
        # No known class – produce a commented template
        return (
            f"# Connect '{dev_name}' ({dev_type})\n"
            f"# Example:\n"
            f"#   import opticalib.devices as devices\n"
            f"#   {var_name} = devices.<ClassName>()\n"
            f"print('Please instantiate {dev_name!r} manually.')"
        )


# ---------------------------------------------------------------------------
# Configuration dialogs
# ---------------------------------------------------------------------------


class ConfigViewDialog(QDialog):
    """
    Read-only dialog that displays the raw YAML configuration file.

    Parameters
    ----------
    config_path : str
        Full path to the ``configuration.yaml`` file.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self, config_path: str, parent: Optional[QWidget] = None
    ) -> None:
        """Open the config file and display its content."""
        super().__init__(parent)
        self.setWindowTitle(
            f"Configuration – {os.path.basename(config_path)}"
        )
        self.resize(700, 600)

        layout = QVBoxLayout(self)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Monospace", 10))
        try:
            with open(config_path, "r") as f:
                text_edit.setPlainText(f.read())
        except OSError as exc:
            text_edit.setPlainText(f"Could not read file:\n{exc}")
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class CalpyGUI(QMainWindow):
    """
    Main window of the CalpyGUI application.

    Combines an embedded IPython terminal (identical to a ``calpy`` CLI
    session), a matplotlib figure panel, configuration file utilities, and
    device connection buttons.

    Parameters
    ----------
    config_path : str or None
        Path to the ``configuration.yaml`` file to load.  When *None* the
        default opticalib path (set via the ``AOCONF`` environment variable,
        or the package template) is used.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialise the main window and start the IPython kernel."""
        super().__init__()

        # Resolve configuration path
        from opticalib.core.root import CONFIGURATION_FILE

        self._config_path: str = config_path or CONFIGURATION_FILE

        # Window title derived from the experiment folder name
        exp_name = _get_experiment_name(self._config_path)
        self.setWindowTitle(f"CalpyGUI – {exp_name or 'opticalib'}")
        self.resize(1400, 900)

        # Tracks matplotlib figure numbers -> panel index
        # {matplotlib_fig_num: panel_list_index}
        self._fig_map: Dict[int, int] = {}

        self._build_ui()
        self._start_kernel()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct the full window layout."""
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # ---- LEFT COLUMN ------------------------------------------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # Config buttons (pink, as in the mockup)
        _btn_style = (
            "QPushButton {"
            "  background-color: #f8c8d4;"
            "  border: 2px solid #e07090;"
            "  border-radius: 4px;"
            "  padding: 6px 14px;"
            "}"
            "QPushButton:hover { background-color: #f0a0b8; }"
        )
        config_row = QHBoxLayout()
        config_row.setContentsMargins(0, 0, 0, 0)

        btn_view = QPushButton("View configuration file")
        btn_view.setStyleSheet(_btn_style)
        btn_view.clicked.connect(self._view_config)

        btn_edit = QPushButton("Edit configuration file")
        btn_edit.setStyleSheet(_btn_style)
        btn_edit.clicked.connect(self._edit_config)

        config_row.addWidget(btn_view)
        config_row.addWidget(btn_edit)
        config_row.addStretch()
        left_layout.addLayout(config_row)

        # Plotting panel (blue border, as in the mockup)
        self._plot_panel = PlotPanel()
        self._plot_panel.setStyleSheet(
            "PlotPanel {"
            "  border: 2px solid #4070c0;"
            "  border-radius: 4px;"
            "  background-color: white;"
            "}"
        )
        left_layout.addWidget(self._plot_panel, stretch=1)

        # ---- RIGHT COLUMN -----------------------------------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # Device panel (teal border, as in the mockup)
        self._device_panel = DevicePanel(
            config_path=self._config_path,
            terminal_callback=self._execute_in_terminal,
        )
        self._device_panel.setStyleSheet(
            "QGroupBox {"
            "  border: 2px solid #40a080;"
            "  border-radius: 4px;"
            "  margin-top: 10px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 8px;"
            "  color: #206040;"
            "  font-weight: bold;"
            "}"
        )
        right_layout.addWidget(self._device_panel, stretch=1)

        # Terminal placeholder (replaced once the kernel is ready)
        self._terminal: QWidget = self._make_terminal_placeholder()
        right_layout.addWidget(self._terminal, stretch=2)

        # ---- SPLITTER ---------------------------------------------------
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter)

    @staticmethod
    def _make_terminal_placeholder() -> QFrame:
        """
        Create a placeholder widget shown while the kernel is starting.

        Returns
        -------
        QFrame
            A simple framed label used as a temporary stand-in.
        """
        placeholder = QFrame()
        placeholder.setFrameShape(QFrame.StyledPanel)
        placeholder.setFrameShadow(QFrame.Sunken)
        lbl = QLabel("Starting IPython kernel…")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: gray; font-size: 13px;")
        layout = QVBoxLayout(placeholder)
        layout.addWidget(lbl)
        return placeholder

    # ------------------------------------------------------------------
    # Kernel management
    # ------------------------------------------------------------------

    def _start_kernel(self) -> None:
        """
        Start the in-process IPython kernel and insert the terminal widget.

        Uses :class:`QtInProcessKernelManager` so that the kernel runs in
        the same process as the GUI, enabling direct matplotlib figure
        capture via a ``post_execute`` hook.
        """
        self._kernel_manager = QtInProcessKernelManager()
        self._kernel_manager.start_kernel(show_banner=False)

        kernel = self._kernel_manager.kernel
        kernel.gui = "qt5"

        # Force the Agg (non-interactive) backend before any matplotlib
        # import so that figures are rendered to in-memory buffers and can
        # be captured and shown in the plot panel.
        kernel.shell.run_cell(
            "import matplotlib; matplotlib.use('Agg')",
            silent=True,
        )

        # Register the figure-capture hook
        kernel.shell.events.register(
            "post_execute", self._on_post_execute
        )

        # Build and attach the qtconsole widget
        self._kernel_client = self._kernel_manager.client()
        self._kernel_client.start_channels()

        term = RichJupyterWidget()
        term.kernel_manager = self._kernel_manager
        term.kernel_client = self._kernel_client

        # Replace the placeholder in the right column
        right_layout = self._device_panel.parent().layout()
        old_widget = right_layout.itemAt(
            right_layout.count() - 1
        ).widget()
        right_layout.replaceWidget(old_widget, term)
        old_widget.deleteLater()
        self._terminal = term

        # Run the calpy init script after a short delay so that the
        # kernel event loop has time to settle.
        QTimer.singleShot(600, self._run_init_script)

    def _run_init_script(self) -> None:
        """
        Bootstrap the kernel with the calpy init script.

        Sets ``AOCONF`` to the resolved configuration path and then
        executes ``initCalpy.py`` inside the kernel, mirroring exactly
        what the ``calpy`` CLI does when launched with ``-f <path>``.
        """
        init_file = _resolve_init_file()
        if init_file is None:
            self._execute_in_terminal(
                "print('Warning: initCalpy.py not found; "
                "calpy environment not loaded.')"
            )
            return

        # Point opticalib at the chosen configuration file
        env_cmd = (
            f"import os\n"
            f"os.environ['AOCONF'] = {self._config_path!r}\n"
            f"import importlib, opticalib.core.root as _r\n"
            f"importlib.reload(_r)"
        )
        self._kernel_manager.kernel.shell.run_cell(env_cmd, silent=True)

        # Execute the init script (equivalent to IPython -i initCalpy.py)
        self._execute_in_terminal(f"%run -i {init_file!r}")

    def _execute_in_terminal(self, command: str) -> None:
        """
        Execute *command* in the embedded IPython terminal.

        Parameters
        ----------
        command : str
            Python / IPython code to run.
        """
        if isinstance(self._terminal, RichJupyterWidget):
            self._terminal.execute(command)

    # ------------------------------------------------------------------
    # Matplotlib figure capture
    # ------------------------------------------------------------------

    def _on_post_execute(self) -> None:
        """
        Kernel ``post_execute`` hook – capture new or updated figures.

        This callback fires after every cell execution in the embedded
        IPython kernel.  It iterates over all currently open matplotlib
        figure numbers, renders each to a PNG buffer, and either adds a
        new entry to the plot panel or updates an existing one.
        """
        try:
            import matplotlib.pyplot as _plt  # noqa: PLC0415
        except ImportError:
            return

        for num in _plt.get_fignums():
            fig = _plt.figure(num)
            buf = io.BytesIO()
            try:
                fig.savefig(
                    buf, format="png", dpi=100, bbox_inches="tight"
                )
            except Exception:
                continue
            buf.seek(0)
            png_bytes = buf.read()

            if num in self._fig_map:
                self._plot_panel.update_figure(
                    self._fig_map[num], png_bytes
                )
            else:
                self._plot_panel.add_figure(png_bytes)
                self._fig_map[num] = len(self._plot_panel._figures) - 1

    # ------------------------------------------------------------------
    # Configuration file actions
    # ------------------------------------------------------------------

    def _view_config(self) -> None:
        """Open the configuration file in a read-only in-app dialog."""
        dlg = ConfigViewDialog(self._config_path, parent=self)
        dlg.exec_()

    def _edit_config(self) -> None:
        """
        Open the configuration file in the system default text editor.

        On Linux the ``VISUAL`` or ``EDITOR`` environment variable is
        honoured; falls back to ``xdg-open``.  On macOS ``open`` is used.
        On Windows ``os.startfile`` is used.

        If the system editor cannot be launched a warning dialog is shown.
        """
        try:
            if sys.platform.startswith("win"):
                os.startfile(self._config_path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", self._config_path])
            else:
                editor = (
                    os.environ.get("VISUAL")
                    or os.environ.get("EDITOR")
                    or "xdg-open"
                )
                subprocess.Popen([editor, self._config_path])
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Editor Error",
                f"Could not open the system editor:\n{exc}\n\n"
                f"Configuration file path:\n{self._config_path}",
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        """Stop the kernel gracefully when the window is closed."""
        try:
            self._kernel_client.stop_channels()
            self._kernel_manager.shutdown_kernel()
        except Exception:
            pass
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
        path resolved by opticalib at import time is used (either the
        ``AOCONF`` environment variable or the package template file).
    """
    app = QApplication.instance() or QApplication(sys.argv)
    window = CalpyGUI(config_path=config_path)
    window.show()
    sys.exit(app.exec_())
