"""
Procedure plugins of CalpyGUI
=============================

The *Procedures* panel lists the bench procedures that get a dedicated
window (see :mod:`opticalib.gui.procedures`): DM calibration, stitching,
segments phasing, alignment and timeseries.
"""

from typing import Callable, Dict, List, Optional, Tuple

from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .procedures.alignment import AlignmentWindow
from .procedures.base import ProcedureWindow
from .procedures.dm_calibration import DeformableMirrorCalibrationWindow
from .procedures.phasing import SegmentsPhasingWindow
from .procedures.stitching import StitchingWindow
from .procedures.timeseries import TimeseriesWindow
from .theme import theme

#: Available plugins, as (name, icon, description).
PLUGINS: List[Tuple[str, str, str]] = [
    (
        "Deformable Mirror Calibration",
        "mirror",
        "Influence functions acquisition and processing, flattening.",
    ),
    ("Stitching", "image-multiple-outline", "Sub-aperture acquisition and stitching."),
    ("Segments Phasing", "flare", "Segment piston measurement with the SPL sensor."),
    ("Alignment", "axis-arrow", "Calibrate and correct the optical alignment."),
    ("Timeseries", "chart-line", "Acquire and analyse sequences of frames."),
]


class PluginPanel(QWidget):
    """
    Panel with one card per available plugin window.

    Parameters
    ----------
    selection_callback : callable
        Callback invoked with the selected plugin name.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        selection_callback: Callable[[str], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        """Build one card per plugin."""
        super().__init__(parent)
        self._selection_callback = selection_callback
        self._icons: List[Tuple[object, str]] = []

        import qtawesome as qta

        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(8, 8, 8, 8)
        column.setSpacing(6)
        caption = QLabel("Open a dedicated window for a bench procedure.")
        caption.setProperty("muted", True)
        caption.setWordWrap(True)
        column.addWidget(caption)
        for name, icon_name, description in PLUGINS:
            card = QFrame()
            card.setProperty("card", True)
            icon = qta.IconWidget()
            icon.setIconSize(QSize(22, 22))
            self._icons.append((icon, icon_name))
            title = QLabel(name)
            title.setProperty("heading", True)
            text = QLabel(description)
            text.setProperty("muted", True)
            text.setWordWrap(True)
            button = QPushButton("Open")
            button.clicked.connect(
                lambda checked=False, n=name: self._selection_callback(n)
            )
            texts = QVBoxLayout()
            texts.setSpacing(0)
            texts.addWidget(title)
            texts.addWidget(text)
            row = QHBoxLayout(card)
            row.setContentsMargins(10, 8, 8, 8)
            row.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
            row.addLayout(texts, 1)
            row.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
            column.addWidget(card)
        column.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        theme().changed.connect(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        for icon, name in self._icons:
            icon.setIcon(theme().icon(name, "accent"))


#: Window class of each plugin, by name.
PLUGIN_WINDOWS: Dict[str, type] = {
    "Deformable Mirror Calibration": DeformableMirrorCalibrationWindow,
    "Stitching": StitchingWindow,
    "Segments Phasing": SegmentsPhasingWindow,
    "Alignment": AlignmentWindow,
    "Timeseries": TimeseriesWindow,
}

#: Former name of the procedure window base class.
PluginWindowBase = ProcedureWindow
