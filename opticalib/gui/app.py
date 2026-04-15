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
from PyQt5.QtCore import QSettings, Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressDialog,
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


def _flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
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
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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

    def get_figure_count(self) -> int:
        """
        Return the number of figures currently stored in the panel.

        Returns
        -------
        int
            Total number of captured figures.
        """
        return len(self._figures)

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

    _SIM_ALPAO_CMD = (
        "from opticalib.simulator import AlpaoDm\n" "dm = AlpaoDm(nActs={})"
    )
    # Simulated device labels and terminal commands.
    _SIMULATED_COMMANDS: Dict[str, str] = {
        "Alpao DM 88": _SIM_ALPAO_CMD.format(88),
        "Alpao DM 97": _SIM_ALPAO_CMD.format(97),
        "Alpao DM 192": _SIM_ALPAO_CMD.format(192),
        "Alpao DM 277": _SIM_ALPAO_CMD.format(277),
        "Alpao DM 468": _SIM_ALPAO_CMD.format(468),
        "Alpao DM 820": _SIM_ALPAO_CMD.format(820),
        "M4 Demonstration Prototype": (
            "from opticalib.simulator import DP\n" "dm = DP()"
        ),
        "Interferometer": (
            "from opticalib.simulator import Fake4DInterf\n"
            "if 'dm' not in globals():\n"
            "    raise RuntimeError('No DM available')\n"
            "interf = Fake4DInterf(dm)"
        ),
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
        """Build two internal sections: real and simulated devices."""
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(4, 4, 4, 4)
        outer_layout.setSpacing(6)

        real_label = QLabel("Configured devices")
        real_label.setStyleSheet("font-weight: 600; color: #206040;")
        outer_layout.addWidget(real_label)

        real_container = QWidget()
        real_container_layout = QVBoxLayout(real_container)
        real_container_layout.setContentsMargins(0, 0, 0, 0)

        real_scroll = QScrollArea()
        real_scroll.setWidgetResizable(True)
        real_scroll.setFrameShape(QFrame.NoFrame)

        real_scroll_container = QWidget()
        self._btn_layout = QVBoxLayout(real_scroll_container)
        self._btn_layout.setAlignment(Qt.AlignTop)
        self._btn_layout.setSpacing(4)

        self._populate_buttons()

        real_scroll.setWidget(real_scroll_container)
        real_container_layout.addWidget(real_scroll)
        outer_layout.addWidget(real_container, stretch=1)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        outer_layout.addWidget(separator)

        sim_label = QLabel("Simulated devices")
        sim_label.setStyleSheet("font-weight: 600; color: #705200;")
        outer_layout.addWidget(sim_label)

        sim_scroll = QScrollArea()
        sim_scroll.setWidgetResizable(True)
        sim_scroll.setFrameShape(QFrame.NoFrame)

        sim_scroll_container = QWidget()
        self._sim_layout = QVBoxLayout(sim_scroll_container)
        self._sim_layout.setContentsMargins(0, 0, 0, 0)
        self._sim_layout.setSpacing(4)
        self._sim_layout.setAlignment(Qt.AlignTop)
        self._populate_simulated_buttons()

        sim_scroll.setWidget(sim_scroll_container)
        outer_layout.addWidget(sim_scroll, stretch=1)

    def _populate_buttons(self) -> None:
        """Create one button per connectable device found in the config."""
        while self._btn_layout.count():
            widget = self._btn_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()

        devices = _get_connectable_devices(self._config_path)
        if not devices:
            label = QLabel("No configured devices found.")
            label.setStyleSheet("color: gray; font-style: italic;")
            self._btn_layout.addWidget(label)
            return

        for dev_type, dev_name, dev_conf in devices:
            btn = QPushButton(f"Connect  {dev_name}")
            tooltip_lines = [f"Type: {dev_type}"]
            tooltip_lines += [f"  {k}: {v}" for k, v in _flatten_dict(dev_conf).items()]
            btn.setToolTip("\n".join(tooltip_lines))
            cmd = self._build_connect_command(dev_type, dev_name)
            btn.clicked.connect(lambda checked, c=cmd: self._terminal_callback(c))
            self._btn_layout.addWidget(btn)

    def _populate_simulated_buttons(self) -> None:
        """Create fixed buttons for simulated devices."""
        while self._sim_layout.count():
            widget = self._sim_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()

        for label, cmd in self._SIMULATED_COMMANDS.items():
            btn = QPushButton(label)
            btn.setToolTip("Instantiate a simulated device in the terminal")
            btn.clicked.connect(
                lambda checked, c=cmd, l=label: self._terminal_callback(
                    c,
                    show_progress=True,
                    progress_text=(
                        f"Initializing simulated device: {l}. "
                        "This may take some time."
                    ),
                )
            )
            self._sim_layout.addWidget(btn)

        self._sim_layout.addStretch()

    # ------------------------------------------------------------------
    # Command generation
    # ------------------------------------------------------------------

    def _build_connect_command(self, dev_type: str, dev_name: str) -> str:
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
        var_name = self._get_variable_name(dev_type)

        # Look for a matching prefix
        class_name: Optional[str] = None
        for prefix, cls in self._NAME_CLASS_MAP.items():
            if dev_name.startswith(prefix):
                class_name = cls
                break

        model = self._get_device_model(dev_name)

        if class_name:
            return (
                f"import opticalib.devices as devices\n"
                f"{var_name} = {class_name}({model})"
            )
        # No known class – produce a commented template
        return (
            f"# Connect '{dev_name}' ({dev_type})\n"
            f"# Example:\n"
            f"#   import opticalib.devices as devices\n"
            f"#   {var_name} = devices.<ClassName>()\n"
            f"print('Please instantiate {dev_name!r} manually.')"
        )

    def _get_device_model(self, dev_name: str) -> str:
        """
        Get the device model from the configuration for a given device.

        Parameters
        ----------
        dev_name : str
            Device name as it appears in the YAML file.

        Returns
        -------
        str
            The value of the 'model' field for the specified device.
        """
        PREFIX_MAP: list[str] = [
            "PhaseCam",
            "AccuFiz",
            "Processer4D",
            "Alpao",
        ]

        EMPTY_MODELS: list[str] = [
            "PetalDM",
        ]

        for prefix in PREFIX_MAP:
            if dev_name.startswith(prefix):
                return dev_name.lstrip(prefix).strip()
            if dev_name in EMPTY_MODELS:
                return ""

        return dev_name.strip()

    def _get_variable_name(self, dev_type: str):
        """
        Generate a valid Python variable name based on the device type.

        Parameters
        ----------
        dev_type : str
            YAML device category (e.g. ``'INTERFEROMETER'``).

        Returns
        -------
        str
            A lowercase variable name derived from *dev_type*.
        """
        DEVICE_TO_VAR_MAP: Dict[str, str] = {
            "INTERFEROMETER": "interf",
            "DEFORMABLE.MIRRORS": "dm",
            "CAMERAS": "cam",
            "MOTORS": "motor",
        }
        return DEVICE_TO_VAR_MAP.get(dev_type.upper(), "device")


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

    def __init__(self, config_path: str, parent: Optional[QWidget] = None) -> None:
        """Open the config file and display its content."""
        super().__init__(parent)
        self.setWindowTitle(f"Configuration – {os.path.basename(config_path)}")
        self.resize(700, 600)

        layout = QVBoxLayout(self)

        text_edit = QTextEdit()
        text_edit.setReadOnly(False)
        text_edit.setFont(QFont("Monospace", 12))
        text_edit.setCursor(Qt.IBeamCursor)
        try:
            with open(config_path, "r") as f:
                text_edit.setPlainText(f.read())
        except OSError as exc:
            text_edit.setPlainText(f"Could not read file:\n{exc}")
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        text_edit.textChanged.connect(lambda: close_btn.setText("Save and Close"))
        close_btn.clicked.connect(
            lambda: self._save_changes(config_path, text_edit.toPlainText())
        )
        close_btn.clicked.connect(
            lambda: (
                self.parent()._device_panel._populate_buttons()
                if self.parent()
                else None
            )
        )

    def _save_changes(self, config_path: str, new_content: str) -> None:
        """
        Save the edited configuration back to disk.

        Parameters
        ----------
        config_path : str
            Full path to the ``configuration.yaml`` file.
        new_content : str
            The updated YAML content to write to the file.
        """
        try:
            with open(config_path, "w") as f:
                f.write(new_content)
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Error Saving Configuration",
                f"Could not save changes:\n{exc}",
            )


# ---------------------------------------------------------------------------
# Plugin window
# ---------------------------------------------------------------------------


class PluginPanel(QGroupBox):
    """
    Panel containing one button per available plugin GUI.

    Parameters
    ----------
    selection_callback : callable
        Callback invoked with the selected plugin label.
    parent : QWidget, optional
        Parent widget.
    """

    _PLUGIN_CHOICES: List[str] = [
        "Deformable Mirror Calibration",
        "Stitching",
        "Segments Phasing",
        "Alignment",
        "Timeseries",
    ]

    def __init__(
        self,
        selection_callback,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialise the plugin panel with one button per plugin."""
        super().__init__("Plugins", parent)
        self._selection_callback = selection_callback
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the plugin button list."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        caption = QLabel("Open a dedicated GUI for one of the available plugins.")
        caption.setWordWrap(True)
        caption.setStyleSheet("color: #555;")
        layout.addWidget(caption)

        for plugin_name in self._PLUGIN_CHOICES:
            btn = QPushButton(plugin_name)
            btn.clicked.connect(
                lambda checked, n=plugin_name: self._selection_callback(n)
            )
            layout.addWidget(btn)

        layout.addStretch()


class PluginWindowBase(QMainWindow):
    """
    Base window for plugin placeholder GUIs.

    Parameters
    ----------
    plugin_title : str
        Title shown in the window bar and as a heading.
    plugin_description : str
        Short text explaining the intent of the plugin GUI.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        plugin_title: str,
        plugin_description: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialise a placeholder plugin GUI window."""
        super().__init__(parent)
        self.setWindowTitle(plugin_title)
        self.resize(900, 600)
        self._build_ui(plugin_title, plugin_description)

    def _build_ui(self, title: str, description: str) -> None:
        """Build the plugin placeholder layout and dummy actions."""
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        heading = QLabel(title)
        heading.setStyleSheet("font-size: 20px; font-weight: 700;")

        subtitle = QLabel(description)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #555;")

        options_box = QGroupBox("Plugin options")
        options_layout = QGridLayout(options_box)
        options_layout.setHorizontalSpacing(10)
        options_layout.setVerticalSpacing(10)

        btn_configure = QPushButton("Configure")
        btn_load_data = QPushButton("Load data")
        btn_run = QPushButton("Run")
        btn_preview = QPushButton("Preview")
        btn_export = QPushButton("Export")

        btn_configure.clicked.connect(
            lambda: self._show_placeholder_action("Configure")
        )
        btn_load_data.clicked.connect(
            lambda: self._show_placeholder_action("Load data")
        )
        btn_run.clicked.connect(lambda: self._show_placeholder_action("Run"))
        btn_preview.clicked.connect(lambda: self._show_placeholder_action("Preview"))
        btn_export.clicked.connect(lambda: self._show_placeholder_action("Export"))

        options_layout.addWidget(btn_configure, 0, 0)
        options_layout.addWidget(btn_load_data, 0, 1)
        options_layout.addWidget(btn_run, 0, 2)
        options_layout.addWidget(btn_preview, 1, 0)
        options_layout.addWidget(btn_export, 1, 1)

        status_box = QGroupBox("Status")
        status_layout = QVBoxLayout(status_box)
        self._status_label = QLabel(
            "Placeholder GUI ready. Implement plugin logic here."
        )
        self._status_label.setWordWrap(True)
        status_layout.addWidget(self._status_label)

        root_layout.addWidget(heading)
        root_layout.addWidget(subtitle)
        root_layout.addWidget(options_box)
        root_layout.addWidget(status_box)
        root_layout.addStretch()

    def _show_placeholder_action(self, action_name: str) -> None:
        """
        Show a placeholder message for an unimplemented action.

        Parameters
        ----------
        action_name : str
            The action clicked by the user.
        """
        self._status_label.setText(
            f"Action '{action_name}' clicked. "
            "Replace this handler with the real implementation."
        )
        QMessageBox.information(
            self,
            "Placeholder action",
            f"{action_name} is not implemented yet.",
        )


class DeformableMirrorCalibrationWindow(PluginWindowBase):
    """Placeholder GUI for Deformable Mirror Calibration plugin."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialise the Deformable Mirror Calibration placeholder."""
        super().__init__(
            plugin_title="Deformable Mirror Calibration",
            plugin_description=(
                "Placeholder window for deformable mirror calibration "
                "workflows and controls."
            ),
            parent=parent,
        )


class StitchingWindow(PluginWindowBase):
    """Placeholder GUI for Stitching plugin."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialise the Stitching placeholder."""
        super().__init__(
            plugin_title="Stitching",
            plugin_description=(
                "Placeholder window for stitching configuration and "
                "processing tasks."
            ),
            parent=parent,
        )


class SegmentsPhasingWindow(PluginWindowBase):
    """Placeholder GUI for Segments Phasing plugin."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialise the Segments Phasing placeholder."""
        super().__init__(
            plugin_title="Segments Phasing",
            plugin_description=(
                "Placeholder window for segmented mirror phasing setup "
                "and execution."
            ),
            parent=parent,
        )


class AlignmentWindow(PluginWindowBase):
    """Placeholder GUI for Alignment plugin."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialise the Alignment placeholder."""
        super().__init__(
            plugin_title="Alignment",
            plugin_description=(
                "Placeholder window for alignment procedures and "
                "associated controls."
            ),
            parent=parent,
        )


class TimeseriesWindow(PluginWindowBase):
    """Placeholder GUI for Timeseries plugin."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialise the Timeseries placeholder."""
        super().__init__(
            plugin_title="Timeseries",
            plugin_description=(
                "Placeholder window for timeseries plugin options, "
                "analysis tasks, and outputs."
            ),
            parent=parent,
        )


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

        self._settings = QSettings("ArcetriAdaptiveOptics", "CalpyGUI")

        # Resolve configuration path
        from opticalib.core.root import CONFIGURATION_FILE

        self._config_path: str = config_path or CONFIGURATION_FILE

        # Window title derived from the experiment folder name
        exp_name = _get_experiment_name(self._config_path)
        self.setWindowTitle(f"CalpyGUI – {exp_name or 'opticalib'}")
        self.resize(1600, 900)

        # Tracks matplotlib figure numbers -> panel index
        # {matplotlib_fig_num: panel_list_index}
        self._fig_map: Dict[int, int] = {}

        # Strong references to plugin windows, kept while they are open.
        self._plugin_windows: List[QMainWindow] = []

        # Busy indicator state used for long terminal commands.
        self._busy_dialog: Optional[QProgressDialog] = None
        self._pending_busy_commands: int = 0

        self._build_ui()
        self._restore_layout_settings()
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

        config_row = QHBoxLayout()
        config_row.setContentsMargins(0, 0, 0, 0)

        btn_view = QPushButton("View configuration file")
        btn_view.setStyleSheet(
            (
                "QPushButton {"
                "  background-color: #f8c8d4;"
                "  border: 2px solid #e07090;"
                "  border-radius: 4px;"
                "  padding: 6px 14px;"
                "}"
                "QPushButton:hover { background-color: #f0a0b8; }"
            )
        )
        btn_view.clicked.connect(self._view_config)

        config_row.addWidget(btn_view)
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

        self._plugin_panel = PluginPanel(
            selection_callback=self._open_plugin_window,
            parent=right_widget,
        )
        self._plugin_panel.setStyleSheet(
            "QGroupBox {"
            "  border: 2px solid #b48a2b;"
            "  border-radius: 4px;"
            "  margin-top: 10px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 8px;"
            "  color: #705200;"
            "  font-weight: bold;"
            "}"
        )

        # Horizontal splitter for the top-right panels (devices/plugins).
        self._upper_splitter = QSplitter(Qt.Horizontal)
        self._upper_splitter.addWidget(self._device_panel)
        self._upper_splitter.addWidget(self._plugin_panel)
        self._upper_splitter.setStretchFactor(0, 1)
        self._upper_splitter.setStretchFactor(1, 1)

        # ---- LOWER ROW ------------------------------------------------
        # Terminal placeholder (replaced once the kernel is ready)
        self._terminal: QWidget = self._make_terminal_placeholder()

        # Vertical splitter for top-right panels vs terminal.
        self._right_splitter = QSplitter(Qt.Vertical)
        self._right_splitter.addWidget(self._upper_splitter)
        self._right_splitter.addWidget(self._terminal)
        self._right_splitter.setStretchFactor(0, 1)
        self._right_splitter.setStretchFactor(1, 2)
        right_layout.addWidget(self._right_splitter, stretch=1)

        # ---- SPLITTER ---------------------------------------------------
        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.addWidget(left_widget)
        self._main_splitter.addWidget(right_widget)
        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 2)
        root_layout.addWidget(self._main_splitter)

    def _restore_layout_settings(self) -> None:
        """Restore window geometry and splitter sizes from QSettings."""
        geometry = self._settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        self._restore_splitter_sizes(self._main_splitter, "splitter/main")
        self._restore_splitter_sizes(self._right_splitter, "splitter/right")
        self._restore_splitter_sizes(self._upper_splitter, "splitter/upper")

    def _save_layout_settings(self) -> None:
        """Save window geometry and splitter sizes to QSettings."""
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("splitter/main", self._main_splitter.sizes())
        self._settings.setValue("splitter/right", self._right_splitter.sizes())
        self._settings.setValue("splitter/upper", self._upper_splitter.sizes())

    def _restore_splitter_sizes(self, splitter: QSplitter, settings_key: str) -> None:
        """Restore one splitter's sizes from QSettings if available."""
        raw_sizes = self._settings.value(settings_key)
        if raw_sizes is None:
            return

        if isinstance(raw_sizes, str):
            parsed = [s.strip() for s in raw_sizes.split(",") if s.strip()]
            sizes = [int(s) for s in parsed]
        else:
            sizes = [int(s) for s in raw_sizes]

        if sizes:
            splitter.setSizes(sizes)

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
        kernel.gui = "qt"

        # Force the Agg (non-interactive) backend before any matplotlib
        # import so that figures are rendered to in-memory buffers and can
        # be captured and shown in the plot panel.
        kernel.shell.run_cell(
            "import matplotlib; matplotlib.use('Agg')",
            silent=True,
        )

        # Register the figure-capture hook
        kernel.shell.events.register("post_execute", self._on_post_execute)

        # Build and attach the qtconsole widget
        self._kernel_client = self._kernel_manager.client()
        self._kernel_client.start_channels()

        term = RichJupyterWidget()
        term.kernel_manager = self._kernel_manager
        term.kernel_client = self._kernel_client

        # Replace the terminal placeholder in either a layout-based parent
        # or a splitter-based parent.
        terminal_parent = self._terminal.parent()
        if isinstance(terminal_parent, QSplitter):
            index = terminal_parent.indexOf(self._terminal)
            terminal_parent.insertWidget(index, term)
            self._terminal.setParent(None)
        else:
            parent_layout = self._terminal.parentWidget().layout()
            parent_layout.replaceWidget(self._terminal, term)

        self._terminal.deleteLater()
        self._terminal = term

        # Re-apply saved vertical split after swapping placeholder/terminal,
        # because splitter proportions can be reset by widget replacement.
        QTimer.singleShot(
            0,
            lambda: self._restore_splitter_sizes(
                self._right_splitter, "splitter/right"
            ),
        )

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
        env_init = (
            f"import os\n"
            f"os.environ['AOCONF'] = {self._config_path!r}\n"
            f"import importlib; import opticalib.core.root as _r\n"
            f"importlib.reload(_r)\n"
        )

        # post-init reload to ensure the paths shows as updated in the terminal
        # after initCalpy runs, and to re-import any modules that may have
        # cached the old path.
        post_init_reload = (
            f"from importlib import reload; import types; gb = globals(); name=val=None\n"
            f"for name, val in gb.items():\n"
            f"    if isinstance(val, types.ModuleType) and 'opticalib' in name:\n"
            f"        reload(val)\n"
            f"del gb, reload, types\n"
            f"from opticalib.__init_script__.initCalpy import *\n"
        )

        self._kernel_manager.kernel.shell.run_cell(env_init, silent=True)
        self._execute_in_terminal(f"%run -i {init_file!r}")
        self._kernel_manager.kernel.shell.run_cell(post_init_reload, silent=True)

    def _execute_in_terminal(
        self,
        command: str,
        show_progress: bool = False,
        progress_text: str = "Working...",
    ) -> None:
        """
        Execute *command* in the embedded IPython terminal.

        Parameters
        ----------
        command : str
            Python / IPython code to run.
        show_progress : bool, optional
            Whether to show a busy indicator while the command runs.
        progress_text : str, optional
            Message displayed in the busy indicator dialog.
        """
        if isinstance(self._terminal, RichJupyterWidget):
            if show_progress:
                self._show_busy_dialog(progress_text)
                self._pending_busy_commands += 1
            self._terminal.execute(command)

    def _show_busy_dialog(self, message: str) -> None:
        """
        Show an indeterminate progress dialog for long-running commands.

        Parameters
        ----------
        message : str
            Text shown in the progress dialog.
        """
        if self._busy_dialog is None:
            dialog = QProgressDialog(message, "", 0, 0, self)
            dialog.setWindowTitle("Please wait")
            dialog.setCancelButton(None)
            # Keep terminal interaction available while simulated commands
            # are running in the embedded kernel.
            dialog.setWindowModality(Qt.NonModal)
            dialog.setMinimumDuration(0)
            dialog.setAutoClose(False)
            dialog.setAutoReset(False)
            self._busy_dialog = dialog
        else:
            self._busy_dialog.setLabelText(message)

        self._busy_dialog.show()
        QApplication.processEvents()

    def _hide_busy_dialog_if_done(self) -> None:
        """Close the busy dialog when all tracked commands are completed."""
        if self._pending_busy_commands > 0:
            self._pending_busy_commands -= 1

        if self._pending_busy_commands == 0 and self._busy_dialog is not None:
            self._busy_dialog.hide()

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
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            except Exception:
                continue
            buf.seek(0)
            png_bytes = buf.read()

            if num in self._fig_map:
                self._plot_panel.update_figure(self._fig_map[num], png_bytes)
            else:
                self._plot_panel.add_figure(png_bytes)
                self._fig_map[num] = self._plot_panel.get_figure_count() - 1

        if self._pending_busy_commands > 0:
            QTimer.singleShot(0, self._hide_busy_dialog_if_done)

    # ------------------------------------------------------------------
    # Configuration file actions
    # ------------------------------------------------------------------

    def _view_config(self, edit: bool = False) -> None:
        """Open the configuration file in a read-only in-app dialog."""
        dlg = ConfigViewDialog(self._config_path, parent=self)
        dlg.exec_()

    def _open_plugin_window(self, plugin_name: str) -> None:
        """
        Open the GUI window associated with the selected plugin.

        Parameters
        ----------
        plugin_name : str
            Display label selected in the plugin dialog.
        """
        plugin_window_map = {
            "Deformable Mirror Calibration": (DeformableMirrorCalibrationWindow),
            "Stitching": StitchingWindow,
            "Segments Phasing": SegmentsPhasingWindow,
            "Alignment": AlignmentWindow,
            "Timeseries": TimeseriesWindow,
        }

        window_cls = plugin_window_map.get(plugin_name)
        if window_cls is None:
            QMessageBox.warning(
                self,
                "Unknown plugin",
                f"No GUI mapping found for '{plugin_name}'.",
            )
            return

        window = window_cls(parent=self)
        window.setAttribute(Qt.WA_DeleteOnClose, True)
        window.destroyed.connect(
            lambda obj=None, w=window: self._on_plugin_window_closed(w)
        )
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
                # os.startfile is a Windows-only built-in; the type: ignore
                # suppresses the linter warning about the missing attribute on
                # non-Windows platforms.
                os.startfile(self._config_path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", self._config_path])
            else:
                editor = (
                    os.environ.get("VISUAL") or os.environ.get("EDITOR") or "xdg-open"
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
        self._save_layout_settings()
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
