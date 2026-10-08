"""
Workspace panel of CalpyGUI
===========================

Lists the user variables of the kernel, grouped into devices, arrays and
other values, and refreshed after every execution.  Double-clicking a
variable shows it (arrays open in the plot viewer); the context menu offers
the other actions.  Every action runs as visible code in the console.
"""

from typing import Dict, List, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..theme import theme
from .device_panel import KIND_ICONS

#: Groups of the workspace tree, as (group, label, kinds).
GROUPS = [
    ("devices", "Devices", ("dm", "interferometer", "wfs", "camera")),
    ("arrays", "Arrays", ("array",)),
    ("other", "Other", ("other",)),
]


def default_action(item: Dict[str, str]) -> str:
    """
    Return the code run when a variable is double-clicked.

    Parameters
    ----------
    item : dict
        Workspace entry.

    Returns
    -------
    str
        Python source.
    """
    name = item["name"]
    if item.get("kind") == "array":
        return f"_gui.view({name}, {name!r})"
    return name


class WorkspaceView(QWidget):
    """
    Tree of the kernel user variables.

    Signals
    -------
    run_requested(str, str)
        Code to execute in the console, and its title.
    """

    run_requested = Signal(str, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Build the (initially empty) workspace view."""
        super().__init__(parent)
        self._items: List[Dict[str, str]] = []

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter variables")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._populate)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Name", "Value"])
        self._tree.setRootIsDecorated(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._context_menu)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        header = self._tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, header.ResizeMode.ResizeToContents)

        self._empty = QLabel("No variables yet. Connect a device or run some code.")
        self._empty.setProperty("muted", True)
        self._empty.setWordWrap(True)
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)

        top = QHBoxLayout()
        top.setContentsMargins(8, 8, 8, 0)
        top.addWidget(self._filter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addWidget(self._tree, 1)
        layout.addWidget(self._empty, 1)
        theme().changed.connect(self._populate)
        self._populate()

    @property
    def items(self) -> List[Dict[str, str]]:
        """The workspace entries currently shown."""
        return list(self._items)

    def set_items(self, items: List[Dict[str, str]]) -> None:
        """
        Replace the listed variables.

        Parameters
        ----------
        items : list of dict
            Entries from :func:`opticalib.gui.kernel_side.workspace_items`.
        """
        self._items = list(items)
        self._populate()

    def _populate(self) -> None:
        selected = self._selected_name()
        collapsed = {
            self._tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)
            for i in range(self._tree.topLevelItemCount())
            if not self._tree.topLevelItem(i).isExpanded()
        }
        text = self._filter.text().strip().lower()
        t = theme()
        self._tree.clear()
        shown = 0
        for group, label, kinds in GROUPS:
            entries = [
                it
                for it in self._items
                if it.get("kind") in kinds and (not text or text in it["name"].lower())
            ]
            if not entries:
                continue
            node = QTreeWidgetItem([f"{label} ({len(entries)})"])
            node.setData(0, Qt.ItemDataRole.UserRole, group)
            node.setFirstColumnSpanned(True)
            node.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._tree.addTopLevelItem(node)
            for entry in entries:
                child = QTreeWidgetItem([entry["name"], entry.get("summary", "")])
                child.setData(0, Qt.ItemDataRole.UserRole, entry)
                tip = f"{entry['name']}: {entry.get('type', '')}\n{entry.get('summary', '')}"
                child.setToolTip(0, tip)
                child.setToolTip(1, tip)
                icon = KIND_ICONS.get(entry.get("kind", ""), "variable")
                if entry.get("kind") == "array":
                    icon = "grid"
                elif entry.get("kind") == "other":
                    icon = "variable"
                child.setIcon(
                    0, t.icon(icon, "accent" if group == "devices" else "text_muted")
                )
                node.addChild(child)
                shown += 1
                if entry["name"] == selected:
                    self._tree.setCurrentItem(child)
            node.setExpanded(group not in collapsed)
        self._tree.setVisible(shown > 0)
        self._empty.setVisible(shown == 0)

    def _selected_name(self) -> Optional[str]:
        current = self._tree.currentItem()
        if current is None:
            return None
        entry = current.data(0, Qt.ItemDataRole.UserRole)
        return entry.get("name") if isinstance(entry, dict) else None

    def _on_double_click(self, tree_item: QTreeWidgetItem, column: int) -> None:
        entry = tree_item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(entry, dict):
            self.run_requested.emit(default_action(entry), f"Show {entry['name']}")

    def _context_menu(self, pos) -> None:
        tree_item = self._tree.itemAt(pos)
        entry = tree_item.data(0, Qt.ItemDataRole.UserRole) if tree_item else None
        if not isinstance(entry, dict):
            return
        name = entry["name"]
        t = theme()
        menu = QMenu(self)
        if entry.get("kind") == "array":
            menu.addAction(t.icon("image-outline"), "Show in viewer").triggered.connect(
                lambda: self.run_requested.emit(
                    f"_gui.view({name}, {name!r})", f"Show {name}"
                )
            )
        menu.addAction(t.icon("console"), "Print").triggered.connect(
            lambda: self.run_requested.emit(f"print({name})", f"Print {name}")
        )
        menu.addAction(t.icon("information-outline"), "Inspect").triggered.connect(
            lambda: self.run_requested.emit(f"{name}?", f"Inspect {name}")
        )
        menu.addAction(t.icon("content-copy"), "Copy name").triggered.connect(
            lambda: QGuiApplication.clipboard().setText(name)
        )
        menu.addSeparator()
        menu.addAction(t.icon("delete-outline", "danger"), "Delete…").triggered.connect(
            lambda: self._delete(name)
        )
        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _delete(self, name: str) -> None:
        answer = QMessageBox.question(
            self, "Delete variable", f"Delete '{name}' from the kernel namespace?"
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.run_requested.emit(f"del {name}", f"Delete {name}")
