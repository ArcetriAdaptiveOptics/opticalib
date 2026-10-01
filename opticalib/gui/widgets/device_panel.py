"""
Device panel of CalpyGUI
========================

Lists the devices that can be connected in the kernel:

* **configured** devices, i.e. entries of the ``DEVICES`` section of the
  configuration file with at least one filled-in field;
* **simulated** devices from :mod:`opticalib.simulator`.

Each device is shown as a :class:`DeviceCard` with its connection status,
which follows the kernel workspace: a device is *connected* while the
variable it was bound to (``dm``, ``interf``, ...) holds an instance of the
expected class.  Connecting a device runs its constructor in the console.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..theme import theme
from .common import ElidedLabel
from .connect_dialog import ConnectDialog
from .device_registry import (
    SECTION_KINDS,
    DeviceEntry,
    build_command,
    list_entries,
    resolve_entry,
    set_entry_class,
)

# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# Device descriptions
# ---------------------------------------------------------------------------

#: Icon of each device kind.
KIND_ICONS: Dict[str, str] = {
    "dm": "mirror",
    "interferometer": "waves",
    "wfs": "blur",
    "camera": "camera-outline",
    "motor": "engine-outline",
    "device": "chip",
}

@dataclass
class DeviceInfo:
    """
    Description of a connectable device.

    Attributes
    ----------
    key : str
        Unique identifier of the device in the panel.
    title : str
        Displayed name.
    kind : str
        ``'dm'``, ``'interferometer'``, ``'wfs'``, ``'camera'``, ``'motor'``
        or ``'device'``.
    var_name : str
        Variable the device is bound to in the kernel.
    class_name : str or None
        Class of the instance, used to detect the connection (``None`` when
        the command is only a template to complete manually).
    module_prefix : str
        Module prefix of that class (real or simulated devices).
    build_code : callable
        Returns the connect command, given the selected option.
    options : list
        Choices offered in a combo box (e.g. number of actuators).
    requires : str or None
        Kind of device that must be connected first.
    simulated : bool
        Whether the device is a simulator.
    tooltip : str
        Extra information shown on hover.
    entry : DeviceEntry or None
        The configuration entry (configured devices only).
    """

    key: str
    title: str
    kind: str
    var_name: str
    class_name: Optional[str]
    module_prefix: str
    build_code: Callable[[Any], str]
    options: List[Any] = field(default_factory=list)
    requires: Optional[str] = None
    simulated: bool = False
    tooltip: str = ""
    entry: Optional[DeviceEntry] = None

    @property
    def needs_setup(self) -> bool:
        """Whether the configuration entry cannot be connected in one click."""
        return self.entry is not None and not self.entry.ready


def _simulated(template: str) -> Callable[[Any], str]:
    return lambda option: template.format(option=option)


#: Simulated devices offered by the panel.
SIMULATED_DEVICES: List[DeviceInfo] = [
    DeviceInfo(
        key="sim:alpao",
        title="Alpao DM",
        kind="dm",
        var_name="dm",
        class_name="AlpaoDm",
        module_prefix="opticalib.simulator",
        build_code=_simulated(
            "from opticalib.simulator import AlpaoDm\ndm = AlpaoDm(n_acts={option})"
        ),
        options=[88, 97, 192, 277, 468, 820],
        simulated=True,
        tooltip="Simulated Alpao DM. The first build of each size computes "
        "the influence functions and can take a few minutes.",
    ),
    DeviceInfo(
        key="sim:dp",
        title="M4 Demonstration Prototype",
        kind="dm",
        var_name="dm",
        class_name="DP",
        module_prefix="opticalib.simulator",
        build_code=_simulated("from opticalib.simulator import DP\ndm = DP()"),
        simulated=True,
        tooltip="Simulated AdOptica DP (2 segments). The first build "
        "downloads data and computes the influence functions.",
    ),
    DeviceInfo(
        key="sim:petal",
        title="Petal mirror",
        kind="dm",
        var_name="dm",
        class_name="PetalMirror",
        module_prefix="opticalib.simulator",
        build_code=_simulated(
            "from opticalib.simulator import PetalMirror\ndm = PetalMirror()"
        ),
        simulated=True,
        tooltip="Simulated 6-segment petal mirror.",
    ),
    DeviceInfo(
        key="sim:interf",
        title="4D interferometer",
        kind="interferometer",
        var_name="interf",
        class_name="Fake4DInterf",
        module_prefix="opticalib.simulator",
        build_code=_simulated(
            "from opticalib.simulator import Fake4DInterf\ninterf = Fake4DInterf(dm)"
        ),
        requires="dm",
        simulated=True,
        tooltip="Simulated interferometer looking at the DM bound to `dm`.",
    ),
]


def configured_devices(
    config_path: str,
    overrides: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[DeviceInfo]:
    """
    Describe every device entry of *config_path*.

    All the entries of the ``DEVICES`` section are listed, also those that
    are not filled in yet; see :mod:`~opticalib.gui.widgets.device_registry`
    for how their class is found.

    Parameters
    ----------
    config_path : str
        Path to the ``configuration.yaml`` file.
    overrides : dict, optional
        Per entry key, a ``class`` and/or ``var`` chosen in the connect
        dialog for this session.

    Returns
    -------
    list of DeviceInfo
        One entry per device, in file order.
    """
    overrides = overrides or {}
    infos = []
    for entry in list_entries(config_path):
        override = overrides.get(entry.key, {})
        if override.get("class") and entry.source != "explicit":
            conf = dict(entry.conf, **{"class": override["class"]})
            entry = resolve_entry(entry.section, entry.name, conf)
        var = override.get("var") or entry.var_name
        section = entry.device_class.section if entry.device_class else entry.section
        fields = [f"{k}: {v}" for k, v in _flatten_dict(entry.conf).items()]
        klass = entry.device_class.name if entry.device_class else "unknown class"
        tooltip = [f"{entry.section} → {entry.name} ({klass})"] + fields
        if entry.problems or entry.notes:
            tooltip += [""] + entry.problems + entry.notes
        infos.append(
            DeviceInfo(
                key=entry.key,
                title=entry.name,
                kind=SECTION_KINDS.get(section.upper(), "device"),
                var_name=var,
                class_name=entry.device_class.name if entry.device_class else None,
                module_prefix="opticalib.devices",
                build_code=lambda option, e=entry, v=var: build_command(e, v),
                tooltip="\n".join(tooltip),
                entry=entry,
            )
        )
    return infos


def quick_actions(info: DeviceInfo) -> List[Tuple[str, str, bool]]:
    """
    Return the quick actions of a connected device.

    Parameters
    ----------
    info : DeviceInfo
        The device.

    Returns
    -------
    list of (label, code, needs_confirmation)
        Actions offered in the device menu.
    """
    var = info.var_name
    actions = [("Show in console", var, False)]
    if info.kind in ("interferometer", "wfs"):
        actions.append(
            ("Acquire map", f"img = {var}.acquire_map()\n_gui.view(img, 'img')", False)
        )
    elif info.kind == "camera":
        actions.append(
            (
                "Acquire frame",
                f"frame = {var}.acquire_frames()\n_gui.view(frame, 'frame')",
                False,
            )
        )
    elif info.kind == "dm":
        actions.append(
            ("Show command", f"_gui.view({var}.get_shape(), '{var} command')", False)
        )
        if info.simulated:
            actions.append(
                ("Reset to zero", f"{var}.set_shape(np.zeros({var}.n_acts))", True)
            )
    return actions


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

#: Connection states of a device card, with their label and color token.
STATUS_STYLES: Dict[str, Tuple[str, str]] = {
    "disconnected": ("Not connected", "text_muted"),
    "queued": ("Queued", "warning"),
    "connecting": ("Connecting…", "accent"),
    "connected": ("Connected", "success"),
    "error": ("Failed", "danger"),
    "unavailable": ("Needs a DM", "text_muted"),
    "setup": ("Needs setup", "warning"),
}


class DeviceCard(QFrame):
    """
    Card showing one device, its status and its actions.

    Parameters
    ----------
    info : DeviceInfo
        The device.
    parent : QWidget, optional
        Parent widget.

    Signals
    -------
    connect_requested(object)
        The user asked to connect the device (the card is passed).
    setup_requested(object)
        The user asked to connect with options (opens the connect dialog).
    edit_requested(object)
        The user asked to edit the configuration entry.
    action_requested(str, str)
        The user picked a quick action (code, title).
    """

    connect_requested = Signal(object)
    setup_requested = Signal(object)
    edit_requested = Signal(object)
    action_requested = Signal(str, str)

    def __init__(self, info: DeviceInfo, parent: Optional[QWidget] = None) -> None:
        """Build the card."""
        super().__init__(parent)
        self.info = info
        self.status = "disconnected"
        self.message = ""
        self.setProperty("card", True)
        self.setToolTip(info.tooltip)

        import qtawesome as qta

        self._icon = qta.IconWidget()
        self._icon.setIconSize(QSize(22, 22))
        self._title = ElidedLabel(info.title)
        self._title.setProperty("heading", True)
        self._subtitle = ElidedLabel()
        self._subtitle.setProperty("muted", True)
        self._problem = ElidedLabel()
        self._status = QLabel()
        self._combo: Optional[QComboBox] = None
        if info.options:
            self._combo = QComboBox()
            for option in info.options:
                self._combo.addItem(f"{option} actuators", option)
            self._combo.currentIndexChanged.connect(self._refresh_subtitle)
        self._button = QPushButton("Connect")
        self._button.setProperty("accent", True)
        self._button.clicked.connect(self._on_button)
        self._more = QToolButton()
        self._more.setToolTip("More actions")
        self._more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._more.setMenu(QMenu(self._more))
        self._more.menu().aboutToShow.connect(self._fill_menu)

        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(self._icon)
        header.addWidget(self._title, 1)
        header.addWidget(self._status)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        if self._combo is not None:
            actions.addWidget(self._combo, 1)
        else:
            actions.addStretch(1)
        actions.addWidget(self._button)
        actions.addWidget(self._more)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addWidget(self._subtitle)
        layout.addWidget(self._problem)
        layout.addLayout(actions)

        theme().changed.connect(self._refresh)
        self._refresh()

    @property
    def option(self) -> Any:
        """The selected option (``None`` without options)."""
        if self._combo is None:
            return None
        return self._combo.currentData()

    def connect_code(self) -> str:
        """
        Return the command that connects the device.

        Returns
        -------
        str
            Python source.
        """
        return self.info.build_code(self.option)

    def set_status(self, status: str, message: str = "") -> None:
        """
        Update the connection status.

        Parameters
        ----------
        status : str
            One of :data:`STATUS_STYLES`.
        message : str, optional
            Extra information (e.g. the error), shown on hover.
        """
        self.status = status
        self.message = message
        self._refresh()

    @property
    def display_status(self) -> str:
        """The status shown: ``'setup'`` for idle entries that need setup."""
        if self.status == "disconnected" and self.info.needs_setup:
            return "setup"
        return self.status

    def _on_button(self) -> None:
        if self.display_status == "setup":
            self.setup_requested.emit(self)
        else:
            self.connect_requested.emit(self)

    def _refresh_subtitle(self) -> None:
        lines = [l for l in self.connect_code().splitlines() if not l.startswith(("#", "import", "from"))]
        text = lines[-1] if lines else ""
        entry = self.info.entry
        if self.info.class_name is None:
            text = "Unknown device class"
        elif entry is not None and entry.args is None:
            text = f"{self.info.var_name} = devices.{self.info.class_name}(…)"
        self._subtitle.setText(text)
        self._subtitle.setToolTip(self.connect_code())

    def _refresh(self) -> None:
        t = theme()
        shown = self.display_status
        label, token = STATUS_STYLES.get(shown, STATUS_STYLES["disconnected"])
        self._icon.setIcon(t.icon(KIND_ICONS.get(self.info.kind, "chip"), "accent"))
        self._status.setText(f"<span style='color:{t.tokens[token]}'>●</span> {label}")
        self._status.setToolTip(self.message)
        self._more.setIcon(t.icon("dots-vertical"))
        busy = self.status in ("queued", "connecting")
        self._button.setEnabled(not busy and self.status != "unavailable")
        if shown == "setup":
            self._button.setText("Set up…")
        else:
            self._button.setText("Reconnect" if self.status == "connected" else "Connect")
        problems = self.info.entry.problems if self.info.entry is not None else []
        self._problem.setText(problems[0] if problems else "")
        self._problem.setToolTip("\n".join(problems))
        self._problem.setStyleSheet(f"color: {t.tokens['warning']};")
        self._problem.setVisible(bool(problems) and self.status != "connected")
        if self._combo is not None:
            self._combo.setEnabled(not busy)
        self._refresh_subtitle()

    def _fill_menu(self) -> None:
        menu = self._more.menu()
        menu.clear()
        copy = menu.addAction(theme().icon("content-copy"), "Copy connect code")
        copy.triggered.connect(
            lambda: QGuiApplication.clipboard().setText(self.connect_code())
        )
        if self.info.entry is not None:
            menu.addAction(theme().icon("tune-variant"), "Connect with options…").triggered.connect(
                lambda: self.setup_requested.emit(self)
            )
            menu.addAction(theme().icon("file-document-edit-outline"), "Edit configuration entry").triggered.connect(
                lambda: self.edit_requested.emit(self)
            )
        if self.status == "error" and self.message:
            menu.addAction(theme().icon("alert-circle", "danger"), self.message[:80]).setEnabled(False)
        if self.status != "connected":
            return
        menu.addSeparator()
        for label, code, confirm in quick_actions(self.info):
            action = menu.addAction(label)
            action.triggered.connect(
                lambda checked=False, c=code, l=label, k=confirm: self._run_action(c, l, k)
            )

    def _run_action(self, code: str, label: str, confirm: bool) -> None:
        if confirm:
            answer = QMessageBox.question(
                self,
                label,
                f"This moves the mirror bound to '{self.info.var_name}'.\n\n{code}\n\nContinue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.action_requested.emit(code, f"{self.info.title}: {label}")


class DevicePanel(QWidget):
    """
    Panel with one :class:`DeviceCard` per configured and simulated device.

    Parameters
    ----------
    config_path : str
        Path to the ``configuration.yaml`` file.
    runner : callable
        ``runner(code, title, on_done, on_error)`` executing code in the
        console and returning a task (see
        :meth:`~opticalib.gui.kernel.KernelBridge.run`).
    parent : QWidget, optional
        Parent widget.

    Signals
    -------
    edit_config_requested(str, str)
        Open the configuration editor on an entry (section, name).
    config_changed()
        The panel modified the configuration file (e.g. saved a ``class``).
    """

    edit_config_requested = Signal(str, str)
    config_changed = Signal()

    def __init__(self, config_path: str, runner, parent: Optional[QWidget] = None) -> None:
        """Build the panel and read the configuration."""
        super().__init__(parent)
        self._config_path = config_path
        self._runner = runner
        self._cards: Dict[str, DeviceCard] = {}
        self._workspace: Dict[str, Dict[str, str]] = {}
        # Class and variable chosen in the connect dialog, per entry key.
        self._overrides: Dict[str, Dict[str, str]] = {}

        self._configured_box = QVBoxLayout()
        self._configured_box.setSpacing(6)
        self._simulated_box = QVBoxLayout()
        self._simulated_box.setSpacing(6)

        self._refresh_btn = QToolButton()
        self._refresh_btn.setToolTip("Re-read the configuration file")
        self._refresh_btn.clicked.connect(self.reload)

        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(8, 8, 8, 8)
        column.setSpacing(6)
        header = QHBoxLayout()
        header.addWidget(self._section_label("CONFIGURED"))
        header.addStretch()
        header.addWidget(self._refresh_btn)
        column.addLayout(header)
        column.addLayout(self._configured_box)
        column.addSpacing(8)
        column.addWidget(self._section_label("SIMULATED"))
        column.addLayout(self._simulated_box)
        column.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

        for info in SIMULATED_DEVICES:
            self._simulated_box.addWidget(self._make_card(info))
        self.reload()
        theme().changed.connect(
            lambda: self._refresh_btn.setIcon(theme().icon("refresh"))
        )
        self._refresh_btn.setIcon(theme().icon("refresh"))

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("section", True)
        return label

    @property
    def cards(self) -> List[DeviceCard]:
        """All device cards."""
        return list(self._cards.values())

    def card(self, key: str) -> Optional[DeviceCard]:
        """
        Return the card with the given key.

        Parameters
        ----------
        key : str
            :attr:`DeviceInfo.key`.

        Returns
        -------
        DeviceCard or None
            The card.
        """
        return self._cards.get(key)

    def set_config_path(self, config_path: str) -> None:
        """
        Switch to another configuration file.

        Parameters
        ----------
        config_path : str
            Path to the ``configuration.yaml`` file.
        """
        self._config_path = config_path
        self.reload()

    def reload(self) -> None:
        """Re-read the configured devices, keeping known card states."""
        infos = configured_devices(self._config_path, self._overrides)
        keep = {info.key for info in infos}
        for key in [k for k in self._cards if k.startswith("cfg:") and k not in keep]:
            card = self._cards.pop(key)
            self._configured_box.removeWidget(card)
            card.deleteLater()
        while self._configured_box.count():
            item = self._configured_box.takeAt(0)
            widget = item.widget()
            if widget is not None and widget not in self._cards.values():
                widget.deleteLater()
        if not infos:
            empty = QLabel("No configured devices. Fill in a device in the configuration file.")
            empty.setProperty("muted", True)
            empty.setWordWrap(True)
            self._configured_box.addWidget(empty)
        for info in infos:
            card = self._cards.get(info.key)
            if card is None:
                card = self._make_card(info)
            else:
                card.info = info
                card.setToolTip(info.tooltip)
                card._refresh()
            self._configured_box.addWidget(card)
        # The stored workspace may be stale (the refresh can be queued behind a
        # long command): use it for the new cards, but never to disconnect.
        self._apply_workspace(demote=False)

    def _make_card(self, info: DeviceInfo) -> DeviceCard:
        card = DeviceCard(info)
        card.connect_requested.connect(self._connect)
        card.setup_requested.connect(self.open_connect_dialog)
        card.edit_requested.connect(
            lambda c: self.edit_config_requested.emit(c.info.entry.section, c.info.entry.name)
        )
        card.action_requested.connect(lambda code, title: self._runner(code, title, None, None))
        self._cards[info.key] = card
        return card

    def open_connect_dialog(self, card: DeviceCard) -> Optional[ConnectDialog]:
        """
        Let the user choose how to connect a configured device, then connect it.

        Parameters
        ----------
        card : DeviceCard
            A card of a configuration entry.

        Returns
        -------
        ConnectDialog or None
            The dialog (after it closed), or ``None`` for simulated devices.
        """
        entry = card.info.entry
        if entry is None:
            return None
        dialog = self._make_dialog(entry, card.info.var_name)
        result = dialog.exec()
        if result == ConnectDialog.EDIT_CONFIG:
            self.edit_config_requested.emit(entry.section, entry.name)
        elif result == ConnectDialog.DialogCode.Accepted:
            self.apply_connect_dialog(card, dialog)
        dialog.deleteLater()
        return dialog

    def _make_dialog(self, entry: DeviceEntry, var_name: str) -> ConnectDialog:
        """Create the connect dialog (separate for tests)."""
        return ConnectDialog(entry, var_name=var_name, parent=self)

    def apply_connect_dialog(self, card: DeviceCard, dialog: ConnectDialog) -> None:
        """
        Apply the choices of an accepted connect dialog and connect.

        Parameters
        ----------
        card : DeviceCard
            The card the dialog was opened from.
        dialog : ConnectDialog
            The accepted dialog.
        """
        entry = card.info.entry
        key = card.info.key
        self._overrides[key] = {"class": dialog.class_name(), "var": dialog.var_name()}
        if dialog.save_class():
            try:
                set_entry_class(self._config_path, entry.section, entry.name, dialog.class_name())
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Configuration not updated", str(exc))
            else:
                self.config_changed.emit()
        self.reload()
        card = self._cards.get(key, card)
        self._connect(card, code=dialog.code())

    def _connect(self, card: DeviceCard, code: Optional[str] = None) -> None:
        title = f"Connecting {card.info.title}"
        if card.option is not None:
            title += f" ({card.option})"

        # The callbacks look the card up by key when they run: a reload may
        # have replaced (or removed) it while the device was connecting.
        key = card.info.key

        def done(task):
            c = self._cards.get(key)
            if c is not None:
                c.set_status("connected" if c.info.class_name else "disconnected")
                self._mark_owner(c)

        def failed(task):
            c = self._cards.get(key)
            if c is not None:
                error = task.error or {}
                c.set_status("error", f"{error.get('ename', 'Error')}: {error.get('evalue', '')}")

        def started():
            c = self._cards.get(key)
            if c is not None and task.state == "running" and c.status == "queued":
                c.set_status("connecting")

        task = self._runner(code or card.connect_code(), title, done, failed)
        card.set_status("queued" if task is not None and task.state == "queued" else "connecting")
        if task is not None:
            task.changed.connect(started)

    def _mark_owner(self, owner: DeviceCard) -> None:
        """Only *owner* is connected to its variable after a (re)connection."""
        for card in self._cards.values():
            if card is not owner and card.info.var_name == owner.info.var_name:
                if card.status == "connected":
                    card.set_status("disconnected")

    def _matches(self, card: DeviceCard, item: Optional[Dict[str, str]]) -> bool:
        if item is None or card.info.class_name is None:
            return False
        return item.get("type") == card.info.class_name and item.get(
            "module", ""
        ).startswith(card.info.module_prefix)

    def update_workspace(self, items: List[Dict[str, str]]) -> None:
        """
        Update the card states from the kernel workspace.

        Parameters
        ----------
        items : list of dict
            Workspace entries (see
            :func:`opticalib.gui.kernel_side.workspace_items`).
        """
        self._workspace = {item["name"]: item for item in items}
        self._apply_workspace(demote=True)

    def _apply_workspace(self, demote: bool) -> None:
        """
        Update the card states from the stored workspace.

        Parameters
        ----------
        demote : bool
            Whether connected cards whose variable is gone become
            disconnected; only a fresh workspace from the kernel may do so.
        """
        cards = list(self._cards.values())
        for card in cards:
            item = self._workspace.get(card.info.var_name)
            if demote and card.status == "connected" and not self._matches(card, item):
                card.set_status("disconnected")
        # A device created by hand in the console is detected when exactly
        # one card can explain it.
        for name, item in self._workspace.items():
            candidates = [
                c for c in cards if c.info.var_name == name and self._matches(c, item)
            ]
            if len(candidates) == 1 and candidates[0].status in ("disconnected", "error"):
                if not any(c.status == "connected" for c in cards if c.info.var_name == name):
                    candidates[0].set_status("connected")
        for card in cards:
            if card.info.requires is None or card.status in ("queued", "connecting"):
                continue
            available = any(
                item.get("kind") == card.info.requires and name == card.info.requires
                for name, item in self._workspace.items()
            )
            if not available and card.status != "connected":
                card.set_status("unavailable", "Connect a (simulated) DM first.")
            elif available and card.status == "unavailable":
                card.set_status("disconnected")
