"""
Plot viewer of CalpyGUI
=======================

The central panel of the main window.  It collects two kinds of items:

* **figures**: matplotlib figures published by the kernel, rendered as
  images and updated in place when the figure changes;
* **data**: arrays sent with ``_gui.view(array)`` or opened from the data
  browser, shown in an interactive ``pyqtgraph`` viewer (zoom, pan, levels
  and histogram, colormap, pixel readout; 3-D cubes get a frame slider).

A thumbnail strip gives access to every item; items can be saved, popped
out into their own window, or removed.
"""

import itertools
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QIcon, QImage, QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..theme import theme

#: Colormaps offered for data items (pyqtgraph names).
COLORMAPS = ["viridis", "inferno", "magma", "plasma", "cividis", "turbo", "CET-L1", "CET-D1"]
#: Size of the thumbnails in the gallery strip.
THUMB_SIZE = QSize(96, 72)

_keys = itertools.count(1)


@dataclass
class PlotItem:
    """
    One entry of the plot viewer.

    Attributes
    ----------
    key : str
        Unique identifier (``'fig:<uid>'`` for kernel figures).
    kind : str
        ``'figure'`` or ``'data'``.
    title : str
        Displayed title.
    png : bytes or None
        Rendered figure (figures only).
    data : ndarray or None
        Float array with NaN for masked values (data only).
    source : str
        Where the item comes from (e.g. a file path).
    """

    key: str
    kind: str
    title: str
    png: Optional[bytes] = None
    data: Optional[np.ndarray] = None
    source: str = ""
    raw: Any = None
    created: float = field(default_factory=time.time)


def prepare_array(obj: Any) -> np.ndarray:
    """
    Convert an array to a float array suitable for display.

    Masked values become NaN; 3-D arrays are reordered from the opticalib
    cube layout ``(y, x, frame)`` to ``(frame, y, x)``.

    Parameters
    ----------
    obj : array_like
        1-D to 3-D numeric array (masked arrays are supported).

    Returns
    -------
    ndarray
        The display array.

    Raises
    ------
    ValueError
        For arrays with more than three dimensions.
    """
    if isinstance(obj, np.ma.MaskedArray):
        data = np.ma.filled(obj.astype(np.float32), np.nan)
    else:
        data = np.asarray(obj, dtype=np.float32)
    if data.ndim == 0 or data.ndim > 3:
        raise ValueError(f"Cannot display a {data.ndim}-D array.")
    if data.ndim == 3:
        data = np.moveaxis(data, -1, 0)
    return np.ascontiguousarray(data)


def load_npz_view(path: str) -> np.ndarray:
    """
    Load an array handed over by :func:`opticalib.gui.kernel_side.view`.

    Parameters
    ----------
    path : str
        The ``.npz`` file written by the kernel (deleted after reading).

    Returns
    -------
    ndarray or MaskedArray
        The array.
    """
    try:
        with np.load(path) as npz:
            data = npz["data"]
            mask = npz["mask"] if "mask" in npz.files else None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if mask is not None:
        return np.ma.masked_array(data, mask=mask)
    return data


def _thumbnail_for_data(data: np.ndarray) -> QPixmap:
    """Render a small grayscale preview of a data item."""
    frame = data[0] if data.ndim == 3 else data
    pixmap = QPixmap(THUMB_SIZE)
    if frame.ndim != 2 or not np.isfinite(frame).any():
        pixmap.fill(theme().color("surface_alt"))
        return pixmap
    lo, hi = np.nanpercentile(frame, [1, 99])
    scaled = np.nan_to_num((frame - lo) / (hi - lo if hi > lo else 1.0), nan=0.0)
    img8 = np.ascontiguousarray((np.clip(scaled, 0, 1) * 255).astype(np.uint8))
    image = QImage(img8.data, img8.shape[1], img8.shape[0], img8.strides[0], QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(image.copy()).scaled(
        THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )


class FigureView(QLabel):
    """Label showing a rendered figure scaled to fit its size."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Create an empty figure view."""
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Without a small minimum size the label could only grow.
        self.setMinimumSize(1, 1)
        self._pixmap = QPixmap()

    def set_png(self, png: bytes) -> None:
        """
        Show a PNG image.

        Parameters
        ----------
        png : bytes
            PNG-encoded image.
        """
        self._pixmap = QPixmap()
        self._pixmap.loadFromData(png)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap.isNull():
            self.clear()
            return
        ratio = self.devicePixelRatioF()
        scaled = self._pixmap.scaled(
            self.size() * ratio,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(ratio)
        self.setPixmap(scaled)

    # Qt event handler: the camelCase name is required for Qt to call it.
    def resizeEvent(self, event) -> None:  # noqa: N802
        """Re-scale the figure when the view is resized."""
        super().resizeEvent(event)
        self._rescale()


class DataView(QWidget):
    """
    Interactive viewer for 1-D (line plot), 2-D (image) and 3-D (cube) data.

    Signals
    -------
    hover(str)
        Text describing the value under the cursor.
    """

    hover = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Create the pyqtgraph widgets."""
        super().__init__(parent)
        import pyqtgraph as pg

        pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
        self._pg = pg
        self._image = pg.ImageView()
        self._image.ui.roiBtn.hide()
        self._image.ui.menuBtn.hide()
        self._plot = pg.PlotWidget()
        self._stack = QStackedWidget()
        self._stack.addWidget(self._image)
        self._stack.addWidget(self._plot)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack)
        self._data: Optional[np.ndarray] = None
        self._cmap = "viridis"
        self._image.scene.sigMouseMoved.connect(self._on_mouse_moved)
        theme().changed.connect(self._apply_theme)
        self._apply_theme()

    @property
    def image_view(self):
        """The pyqtgraph ``ImageView`` (2-D and 3-D data)."""
        return self._image

    def set_data(self, data: np.ndarray, title: str = "") -> None:
        """
        Show *data* (as returned by :func:`prepare_array`).

        Parameters
        ----------
        data : ndarray
            1-D, 2-D or 3-D float array.
        title : str, optional
            Plot title (1-D data).
        """
        self._data = data
        if data.ndim == 1:
            self._plot.clear()
            color = theme().tokens["accent"]
            self._plot.plot(np.arange(data.size), data, pen=self._pg.mkPen(color, width=1.6))
            self._plot.setTitle(title)
            self._plot.showGrid(x=True, y=True, alpha=0.25)
            self._stack.setCurrentWidget(self._plot)
            return
        finite = data[np.isfinite(data)]
        levels = None
        if finite.size:
            lo, hi = np.percentile(finite, [0.5, 99.5])
            levels = (float(lo), float(hi if hi > lo else lo + 1))
        if data.ndim == 3:
            self._image.setImage(data, xvals=np.arange(data.shape[0]), levels=levels, autoRange=True)
        else:
            self._image.setImage(data, levels=levels, autoRange=True)
        self.set_colormap(self._cmap)
        self._stack.setCurrentWidget(self._image)

    def set_colormap(self, name: str) -> None:
        """
        Select the colormap of images.

        Parameters
        ----------
        name : str
            A pyqtgraph colormap name (see :data:`COLORMAPS`).
        """
        self._cmap = name
        try:
            self._image.setColorMap(self._pg.colormap.get(name))
        except Exception:
            pass

    def auto_range(self) -> None:
        """Reset the zoom to show the whole data."""
        if self._stack.currentWidget() is self._image:
            self._image.autoRange()
        else:
            self._plot.enableAutoRange()

    def _on_mouse_moved(self, pos) -> None:
        data = self._data
        if data is None or data.ndim < 2:
            return
        item = self._image.getImageItem()
        point = item.mapFromScene(pos)
        x, y = int(np.floor(point.x())), int(np.floor(point.y()))
        frame = data[self._image.currentIndex] if data.ndim == 3 else data
        if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            value = frame[y, x]
            text = "masked" if not np.isfinite(value) else f"{value:.6g}"
            prefix = f"frame {self._image.currentIndex}  " if data.ndim == 3 else ""
            self.hover.emit(f"{prefix}x={x}  y={y}  value={text}")
        else:
            self.hover.emit("")

    def _apply_theme(self) -> None:
        t = theme().tokens
        self._image.view.setBackgroundColor(t["plot_bg"])
        self._image.ui.graphicsView.setBackground(t["plot_bg"])
        self._image.ui.histogram.setBackground(t["plot_bg"])
        from qtpy.QtGui import QColor

        fill = QColor(t["accent"])
        fill.setAlpha(90)
        hist = self._image.getHistogramWidget().item
        hist.plot.setBrush(fill)
        hist.plot.setPen(self._pg.mkPen(t["accent"]))
        hist.axis.setPen(t["plot_fg"])
        hist.axis.setTextPen(t["plot_fg"])
        self._plot.setBackground(t["plot_bg"])
        for axis in ("left", "bottom"):
            ax = self._plot.getAxis(axis)
            ax.setPen(t["plot_fg"])
            ax.setTextPen(t["plot_fg"])


class ItemView(QStackedWidget):
    """Shows one :class:`PlotItem` (figure or data)."""

    hover = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Create the figure and data views."""
        super().__init__(parent)
        self.figure_view = FigureView()
        self.data_view = DataView()
        self.data_view.hover.connect(self.hover.emit)
        self.addWidget(self.figure_view)
        self.addWidget(self.data_view)

    def show_item(self, item: PlotItem) -> None:
        """
        Display *item*.

        Parameters
        ----------
        item : PlotItem
            The item to show.
        """
        if item.kind == "figure":
            self.figure_view.set_png(item.png or b"")
            self.setCurrentWidget(self.figure_view)
        else:
            self.data_view.set_data(item.data, item.title)
            self.setCurrentWidget(self.data_view)


class PlotWindow(QMainWindow):
    """
    Separate window showing a single plot item.

    Signals
    -------
    closed(object)
        The window was closed (the window is passed); its owner deletes it.
    """

    closed = Signal(object)

    def __init__(self, item: PlotItem, colormap: str, parent: Optional[QWidget] = None) -> None:
        """Create the window for *item*."""
        super().__init__(parent)
        self.setWindowTitle(item.title)
        self.resize(820, 640)
        view = ItemView()
        view.data_view.set_colormap(colormap)
        view.show_item(item)
        self.statusBar()
        view.hover.connect(self.statusBar().showMessage)
        self.setCentralWidget(view)

    # Qt event handler: the camelCase name is required for Qt to call it.
    def closeEvent(self, event) -> None:  # noqa: N802
        """Notify the owner, which deletes the window."""
        super().closeEvent(event)
        self.closed.emit(self)


class PlotViewer(QWidget):
    """
    Gallery of figures and data with an interactive viewer.

    Signals
    -------
    current_changed(str)
        Key of the item shown (empty when there is none).
    """

    current_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Build the viewer (initially empty)."""
        super().__init__(parent)
        self._items: Dict[str, PlotItem] = {}
        self._order: List[str] = []
        self._windows: List[QMainWindow] = []

        # Toolbar
        toolbar = QFrame()
        toolbar.setObjectName("Toolbar")
        self._title = QLabel("Plots")
        self._title.setProperty("heading", True)
        self._counter = QLabel("")
        self._counter.setProperty("muted", True)
        self._btn_prev = self._tool("chevron-left", "Previous plot", self.show_previous)
        self._btn_next = self._tool("chevron-right", "Next plot", self.show_next)
        self._cmap = QComboBox()
        self._cmap.setToolTip("Colormap")
        self._cmap.addItems(COLORMAPS)
        self._cmap.currentTextChanged.connect(self._on_colormap)
        self._btn_fit = self._tool("fit-to-screen-outline", "Reset zoom", self._fit)
        self._btn_save = self._tool("content-save-outline", "Save…", self.save_current)
        self._btn_pop = self._tool("open-in-new", "Open in a separate window", self.pop_out_current)
        self._btn_remove = self._tool("delete-outline", "Remove this plot", self.remove_current)
        self._btn_clear = self._tool("broom", "Remove all plots", self.clear)
        bar = QHBoxLayout(toolbar)
        bar.setContentsMargins(10, 6, 8, 6)
        bar.setSpacing(4)
        bar.addWidget(self._btn_prev)
        bar.addWidget(self._btn_next)
        bar.addSpacing(6)
        bar.addWidget(self._title)
        bar.addWidget(self._counter)
        bar.addStretch()
        bar.addWidget(self._cmap)
        for button in (self._btn_fit, self._btn_save, self._btn_pop, self._btn_remove, self._btn_clear):
            bar.addWidget(button)

        # Views
        self._view = ItemView()
        self._empty = self._make_empty_state()
        self._stack = QStackedWidget()
        self._stack.addWidget(self._empty)
        self._stack.addWidget(self._view)

        self._readout = QLabel("")
        self._readout.setProperty("muted", True)
        self._view.hover.connect(self._readout.setText)

        # Thumbnails
        self._strip = QListWidget()
        self._strip.setViewMode(QListWidget.ViewMode.IconMode)
        self._strip.setFlow(QListWidget.Flow.LeftToRight)
        self._strip.setWrapping(False)
        self._strip.setMovement(QListWidget.Movement.Static)
        self._strip.setIconSize(THUMB_SIZE)
        self._strip.setFixedHeight(THUMB_SIZE.height() + 44)
        self._strip.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._strip.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._strip.setSpacing(4)
        self._strip.currentItemChanged.connect(self._on_strip_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(self._stack, 1)
        readout_row = QHBoxLayout()
        readout_row.setContentsMargins(10, 2, 10, 2)
        readout_row.addWidget(self._readout)
        layout.addLayout(readout_row)
        layout.addWidget(self._strip)

        theme().changed.connect(self._apply_theme)
        self._apply_theme()
        self._refresh_controls()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _tool(self, icon: str, tip: str, slot) -> QToolButton:
        button = QToolButton()
        button.setToolTip(tip)
        button.setProperty("icon_name", icon)
        button.clicked.connect(slot)
        return button

    def _make_empty_state(self) -> QWidget:
        import qtawesome as qta

        widget = QWidget()
        self._empty_icon = qta.IconWidget()
        self._empty_icon.setIconSize(QSize(56, 56))
        title = QLabel("No plots yet")
        title.setProperty("heading", True)
        hint = QLabel(
            "Figures created in the console appear here automatically.\n"
            "Use _gui.view(array) to explore an image or a cube interactively,\n"
            "or double-click a file in the Data panel."
        )
        hint.setProperty("muted", True)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(widget)
        layout.addStretch()
        for w in (self._empty_icon, title, hint):
            layout.addWidget(w, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        return widget

    def _apply_theme(self) -> None:
        t = theme()
        for button in self.findChildren(QToolButton):
            name = button.property("icon_name")
            if name:
                button.setIcon(t.icon(name))
        self._empty_icon.setIcon(t.icon("chart-box-outline", "text_muted"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def items(self) -> List[PlotItem]:
        """Items in gallery order."""
        return [self._items[k] for k in self._order]

    def get_figure_count(self) -> int:
        """
        Return the number of items in the viewer.

        Returns
        -------
        int
            Number of figures and data items.
        """
        return len(self._order)

    def current_key(self) -> str:
        """Key of the item shown, or an empty string."""
        item = self._strip.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else ""

    def add_figure(self, payload: Dict[str, Any]) -> PlotItem:
        """
        Add or update a figure published by the kernel.

        Parameters
        ----------
        payload : dict
            ``uid``, ``num``, ``title`` and ``png`` (bytes).

        Returns
        -------
        PlotItem
            The new or updated item.
        """
        key = f"fig:{payload.get('uid')}"
        item = PlotItem(
            key=key,
            kind="figure",
            title=str(payload.get("title") or f"Figure {payload.get('num')}"),
            png=payload.get("png") or b"",
            source="console",
        )
        pixmap = QPixmap()
        pixmap.loadFromData(item.png)
        thumb = pixmap.scaled(THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        return self._put(item, thumb)

    def add_data(self, data: Any, title: str, source: str = "") -> PlotItem:
        """
        Add an array to the viewer and show it.

        Parameters
        ----------
        data : array_like
            1-D to 3-D data (masked arrays are supported; 3-D arrays use the
            opticalib ``(y, x, frame)`` layout).
        title : str
            Displayed title.
        source : str, optional
            Origin of the data (e.g. a file path).

        Returns
        -------
        PlotItem
            The new item.
        """
        display = prepare_array(data)
        item = PlotItem(
            key=f"data:{next(_keys)}",
            kind="data",
            title=title,
            data=display,
            source=source,
            raw=data,
        )
        return self._put(item, _thumbnail_for_data(display))

    def show_key(self, key: str) -> None:
        """
        Show the item with the given key.

        Parameters
        ----------
        key : str
            Item key.
        """
        for row in range(self._strip.count()):
            entry = self._strip.item(row)
            if entry.data(Qt.ItemDataRole.UserRole) == key:
                self._strip.setCurrentItem(entry)
                return

    def show_previous(self) -> None:
        """Show the previous item."""
        row = self._strip.currentRow()
        if row > 0:
            self._strip.setCurrentRow(row - 1)

    def show_next(self) -> None:
        """Show the next item."""
        row = self._strip.currentRow()
        if row < self._strip.count() - 1:
            self._strip.setCurrentRow(row + 1)

    def remove_current(self) -> None:
        """Remove the item shown."""
        key = self.current_key()
        if not key:
            return
        row = self._strip.currentRow()
        self._order.remove(key)
        del self._items[key]
        self._strip.takeItem(row)
        if not self._order:
            self._show(None)
        self._refresh_controls()

    def clear(self) -> None:
        """Remove every item."""
        self._items.clear()
        self._order.clear()
        self._strip.clear()
        self._show(None)
        self._refresh_controls()

    def pop_out_current(self) -> Optional[QMainWindow]:
        """
        Open the item shown in a separate window.

        Returns
        -------
        QMainWindow or None
            The new window.
        """
        key = self.current_key()
        if not key:
            return None
        window = PlotWindow(self._items[key], self._cmap.currentText(), self)
        # A bound method (not a lambda) is disconnected when the viewer dies.
        window.closed.connect(self._on_window_closed)
        self._windows.append(window)
        window.show()
        return window

    def _on_window_closed(self, window: "PlotWindow") -> None:
        if window in self._windows:
            self._windows.remove(window)
        window.deleteLater()

    def save_current(self) -> None:
        """Save the item shown to a file chosen by the user."""
        key = self.current_key()
        if not key:
            return
        item = self._items[key]
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in item.title)[:60] or "plot"
        if item.kind == "figure":
            path, _ = QFileDialog.getSaveFileName(self, "Save figure", f"{safe}.png", "PNG image (*.png)")
            if path:
                with open(path, "wb") as f:
                    f.write(item.png or b"")
            return
        path, chosen = QFileDialog.getSaveFileName(
            self, "Save data", f"{safe}.fits", "FITS file (*.fits);;NumPy array (*.npy)"
        )
        if not path:
            return
        try:
            save_array(path, item.raw if item.raw is not None else item.data)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save {path}:\n{exc}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _put(self, item: PlotItem, thumb: QPixmap) -> PlotItem:
        existing = item.key in self._items
        self._items[item.key] = item
        if existing:
            for row in range(self._strip.count()):
                entry = self._strip.item(row)
                if entry.data(Qt.ItemDataRole.UserRole) == item.key:
                    entry.setIcon(QIcon(thumb))
                    entry.setText(_short(item.title))
                    entry.setToolTip(item.title)
                    if self._strip.currentRow() == row:
                        self._show(item)
                    break
        else:
            self._order.append(item.key)
            entry = QListWidgetItem(QIcon(thumb), _short(item.title))
            entry.setToolTip(item.title)
            entry.setData(Qt.ItemDataRole.UserRole, item.key)
            entry.setSizeHint(QSize(THUMB_SIZE.width() + 16, THUMB_SIZE.height() + 34))
            self._strip.addItem(entry)
            self._strip.setCurrentItem(entry)
        self._refresh_controls()
        return item

    def _on_strip_changed(self, current, previous) -> None:
        key = current.data(Qt.ItemDataRole.UserRole) if current is not None else ""
        self._show(self._items.get(key))
        self._refresh_controls()
        self.current_changed.emit(key)

    def _show(self, item: Optional[PlotItem]) -> None:
        self._readout.setText("")
        if item is None:
            self._stack.setCurrentWidget(self._empty)
            self._title.setText("Plots")
            return
        self._title.setText(item.title)
        self._view.show_item(item)
        self._stack.setCurrentWidget(self._view)

    def _refresh_controls(self) -> None:
        n = len(self._order)
        row = self._strip.currentRow()
        self._counter.setText(f"{row + 1} / {n}" if n else "")
        has = n > 0
        item = self._items.get(self.current_key())
        is_data = item is not None and item.kind == "data"
        self._btn_prev.setEnabled(has and row > 0)
        self._btn_next.setEnabled(has and row < n - 1)
        for button in (self._btn_save, self._btn_pop, self._btn_remove, self._btn_clear):
            button.setEnabled(has)
        self._btn_fit.setEnabled(is_data)
        self._cmap.setVisible(is_data and item.data is not None and item.data.ndim >= 2)

    def _on_colormap(self, name: str) -> None:
        self._view.data_view.set_colormap(name)

    def _fit(self) -> None:
        self._view.data_view.auto_range()


def _short(title: str, size: int = 16) -> str:
    return title if len(title) <= size else title[: size - 1] + "…"


def save_array(path: str, data: Any) -> None:
    """
    Save an array to FITS (opticalib format, mask as second HDU) or NumPy.

    Parameters
    ----------
    path : str
        Destination; the format follows the extension (``.fits`` or
        ``.npy``).
    data : array_like
        The array (masked arrays keep their mask in FITS files).
    """
    if path.lower().endswith(".npy"):
        np.save(path, np.ma.getdata(data))
        return
    from opticalib.ground.osutils import save_fits

    save_fits(path, data)
