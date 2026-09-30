"""
Data browser of CalpyGUI
========================

Browses the opticalib data tree: one node per data category (OPD images,
IF functions, flattening, ...), then one node per tracking number (newest
first), then the files of each tracking number.

The tree refreshes by itself when new tracking numbers appear (e.g. at the
end of an acquisition).  Double-clicking a data file previews it in the
plot viewer; the context menu can also load it in the console, copy its
path or tracking number, or open its folder.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
from qtpy.QtCore import QFileSystemWatcher, Qt, QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices, QGuiApplication
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..theme import theme
from .common import ElidedLabel

#: File extensions that can be previewed in the plot viewer.
PREVIEW_EXTENSIONS = (".fits", ".npy", ".npz")
#: Maximum number of tracking numbers listed per category.
MAX_TNS = 500

_ROLE = Qt.ItemDataRole.UserRole


def load_array_file(path: str) -> Any:
    """
    Load a data file for preview (runs in a background thread).

    Parameters
    ----------
    path : str
        A ``.fits`` (read with opticalib), ``.npy`` or ``.npz`` file.

    Returns
    -------
    ndarray or MaskedArray
        The data (the first array of multi-array files).
    """
    lower = path.lower()
    if lower.endswith(".fits"):
        from opticalib.ground.osutils import load_fits

        data = load_fits(path)
        if isinstance(data, list):
            data = next((d for d in data if d is not None and np.ndim(d) > 0), None)
        if data is None:
            raise ValueError("The FITS file contains no image data.")
    elif lower.endswith(".npz"):
        with np.load(path) as npz:
            data = npz[npz.files[0]]
    else:
        data = np.load(path)
    if isinstance(data, np.ma.MaskedArray):
        return np.ma.masked_array(np.asarray(data.data), mask=np.ma.getmaskarray(data))
    return np.asarray(data)


def load_code(path: str) -> str:
    """
    Return the console code that loads *path* into a variable.

    Parameters
    ----------
    path : str
        Data file.

    Returns
    -------
    str
        Python source.
    """
    if path.lower().endswith(".fits"):
        return f"data = osu.load_fits({path!r})"
    if path.lower().endswith((".npy", ".npz")):
        return f"data = np.load({path!r})"
    return f"data = open({path!r}).read()"


def _list_dir(path: str) -> List[os.DirEntry]:
    try:
        with os.scandir(path) as it:
            return list(it)
    except OSError:
        return []


class DataBrowser(QWidget):
    """
    Tree of the opticalib data folders.

    Signals
    -------
    preview_requested(str)
        Path of a file to preview in the plot viewer.
    run_requested(str, str)
        Code to execute in the console, and its title.
    """

    preview_requested = Signal(str)
    run_requested = Signal(str, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Build the (initially empty) browser."""
        super().__init__(parent)
        self._categories: List[List[str]] = []
        self._base = ""

        self._base_label = ElidedLabel("Waiting for the kernel…")
        self._base_label.setProperty("muted", True)
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter tracking numbers")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        self._refresh_btn = QToolButton()
        self._refresh_btn.setToolTip("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._context_menu)
        self._tree.itemExpanded.connect(self._on_expanded)
        self._tree.itemDoubleClicked.connect(self._on_double_click)

        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(lambda path: self._refresh_timer.start())
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self.refresh)

        top = QHBoxLayout()
        top.addWidget(self._filter, 1)
        top.addWidget(self._refresh_btn)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 0)
        layout.setSpacing(6)
        layout.addWidget(self._base_label)
        layout.addLayout(top)
        layout.addWidget(self._tree, 1)
        theme().changed.connect(self._on_theme)
        self._on_theme()

    def set_folders(self, info: Dict[str, Any]) -> None:
        """
        Set the folders to browse.

        Parameters
        ----------
        info : dict
            ``base`` and ``categories`` as returned by
            :func:`opticalib.gui.kernel_side.folders`.
        """
        self._base = info.get("base", "")
        self._categories = [list(c) for c in info.get("categories", [])]
        self._base_label.setText(self._base or "No data folder")
        self._base_label.setToolTip(self._base)
        watched = self._watcher.directories()
        if watched:
            self._watcher.removePaths(watched)
        paths = [p for _, p in self._categories if os.path.isdir(p)]
        if paths:
            self._watcher.addPaths(paths)
        self.refresh()

    def refresh(self) -> None:
        """Re-read the category folders, keeping expanded nodes open."""
        expanded = set()
        for i in range(self._tree.topLevelItemCount()):
            cat = self._tree.topLevelItem(i)
            if cat.isExpanded():
                expanded.add(cat.data(0, _ROLE))
                for j in range(cat.childCount()):
                    if cat.child(j).isExpanded():
                        expanded.add(cat.child(j).data(0, _ROLE))
        self._tree.clear()
        t = theme()
        for label, path in self._categories:
            all_tns = sorted((e.name for e in _list_dir(path) if e.is_dir()), reverse=True)
            tns = all_tns[:MAX_TNS]
            count = f"{len(all_tns)}" if len(all_tns) <= MAX_TNS else f"latest {MAX_TNS} of {len(all_tns)}"
            node = QTreeWidgetItem([f"{label}  ({count})"])
            node.setData(0, _ROLE, path)
            node.setToolTip(0, path)
            node.setIcon(0, t.icon("folder-outline", "accent"))
            self._tree.addTopLevelItem(node)
            for tn in tns:
                child = QTreeWidgetItem([tn])
                child_path = os.path.join(path, tn)
                child.setData(0, _ROLE, child_path)
                child.setIcon(0, t.icon("folder-outline", "text_muted"))
                child.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
                node.addChild(child)
                if child_path in expanded:
                    child.setExpanded(True)
            if path in expanded:
                node.setExpanded(True)
        self._apply_filter()

    def _on_expanded(self, item: QTreeWidgetItem) -> None:
        path = item.data(0, _ROLE)
        if item.parent() is None or item.childCount() or not path:
            return
        t = theme()
        entries = sorted(_list_dir(path), key=lambda e: (not e.is_dir(), e.name))
        for entry in entries:
            child = QTreeWidgetItem([entry.name])
            child.setData(0, _ROLE, entry.path)
            child.setToolTip(0, entry.path)
            if entry.is_dir():
                child.setIcon(0, t.icon("folder-outline", "text_muted"))
            elif entry.name.lower().endswith(PREVIEW_EXTENSIONS):
                child.setIcon(0, t.icon("image-outline", "accent"))
            else:
                child.setIcon(0, t.icon("file-outline", "text_muted"))
            item.addChild(child)
        if not entries:
            empty = QTreeWidgetItem(["(empty)"])
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            item.addChild(empty)

    def _on_double_click(self, item: QTreeWidgetItem, column: int) -> None:
        path = item.data(0, _ROLE)
        if path and os.path.isfile(path) and path.lower().endswith(PREVIEW_EXTENSIONS):
            self.preview_requested.emit(path)

    def _apply_filter(self) -> None:
        text = self._filter.text().strip().lower()
        for i in range(self._tree.topLevelItemCount()):
            cat = self._tree.topLevelItem(i)
            visible = 0
            for j in range(cat.childCount()):
                child = cat.child(j)
                hide = bool(text) and text not in child.text(0).lower()
                child.setHidden(hide)
                visible += not hide
            if text:
                cat.setExpanded(visible > 0)

    def _context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        path = item.data(0, _ROLE) if item else None
        if not path:
            return
        t = theme()
        menu = QMenu(self)
        is_file = os.path.isfile(path)
        if is_file and path.lower().endswith(PREVIEW_EXTENSIONS):
            menu.addAction(t.icon("eye-outline"), "Preview").triggered.connect(
                lambda: self.preview_requested.emit(path)
            )
            menu.addAction(t.icon("console"), "Load in console").triggered.connect(
                lambda: self.run_requested.emit(load_code(path), f"Load {os.path.basename(path)}")
            )
        menu.addAction(t.icon("content-copy"), "Copy path").triggered.connect(
            lambda: QGuiApplication.clipboard().setText(path)
        )
        tn = self._tracking_number(item)
        if tn:
            menu.addAction(t.icon("content-copy"), f"Copy tracking number ({tn})").triggered.connect(
                lambda: QGuiApplication.clipboard().setText(tn)
            )
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        menu.addAction(t.icon("folder-open-outline"), "Open folder").triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
        )
        menu.exec(self._tree.viewport().mapToGlobal(pos))

    @staticmethod
    def _tracking_number(item: QTreeWidgetItem) -> str:
        """Return the tracking number an item belongs to."""
        while item is not None and item.parent() is not None and item.parent().parent() is not None:
            item = item.parent()
        if item is not None and item.parent() is not None:
            return item.text(0)
        return ""

    def _on_theme(self) -> None:
        self._refresh_btn.setIcon(theme().icon("refresh"))
        if self._categories:
            self.refresh()
