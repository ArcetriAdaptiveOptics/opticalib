"""
Procedure plugins of CalpyGUI
=============================

The *Procedures* panel lists the bench procedures that get a dedicated
window (see :mod:`opticalib.gui.procedures`): DM calibration, stitching,
segments phasing, alignment and timeseries.  A procedure that needs a
section of the configuration file is listed only when the experiment
defines that section: unused sections can simply be deleted.

Each procedure is a :class:`Procedure`; :data:`PROCEDURES` lists opticalib's.
Packages building on opticalib list their own in
:attr:`opticalib.gui.CalpyGUI.PROCEDURES` (see the *Extending the GUI*
section of the GUI reference).
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import yaml

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

@dataclass(frozen=True)
class Procedure:
    """
    A card of the *Procedures* panel and the window it opens.

    Attributes
    ----------
    name : str
        Title of the card (unique in the panel).
    icon : str
        `Material Design <https://pictogrammers.com/library/mdi/>`_ icon name
        (``mdi6.`` prefix omitted).
    description : str
        One-line description shown on the card.
    window : type
        The :class:`~opticalib.gui.procedures.base.ProcedureWindow` subclass
        opened by the card.
    section : str, optional
        Configuration section the procedure needs: without it the card is
        not listed.
    """

    name: str
    icon: str
    description: str
    window: type
    section: Optional[str] = None


#: opticalib's procedures.
PROCEDURES: List[Procedure] = [
    Procedure(
        "Deformable Mirror Calibration",
        "mirror",
        "Influence functions acquisition and processing, flattening.",
        DeformableMirrorCalibrationWindow,
        "INFLUENCE.FUNCTIONS",
    ),
    Procedure(
        "Stitching",
        "image-multiple-outline",
        "Sub-aperture acquisition and stitching.",
        StitchingWindow,
        "STITCHING",
    ),
    Procedure(
        "Segments Phasing",
        "flare",
        "Segment piston measurement with the SPL sensor.",
        SegmentsPhasingWindow,
        "PHASING",
    ),
    Procedure(
        "Alignment",
        "axis-arrow",
        "Calibrate and correct the optical alignment.",
        AlignmentWindow,
        "SYSTEM.ALIGNMENT",
    ),
    Procedure(
        "Timeseries",
        "chart-line",
        "Acquire and analyse sequences of frames.",
        TimeseriesWindow,
    ),
]

#: Former description of the procedures, as (name, icon, description); the
#: configuration section each one needs is in :data:`PROCEDURES`.
PLUGINS: List[Tuple[str, str, str]] = [
    (p.name, p.icon, p.description) for p in PROCEDURES
]


def defined_sections(config_path: Optional[str]) -> Optional[Set[str]]:
    """
    Return the top-level sections that a configuration file defines.

    A section set to ``null`` (or left empty) counts as not defined.

    Parameters
    ----------
    config_path : str or None
        Path to the ``configuration.yaml`` file.

    Returns
    -------
    set of str or None
        The defined sections, or ``None`` when the file cannot be read (then
        every procedure is listed).
    """
    if not config_path:
        return None
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(config, dict):
        return None
    return {str(k) for k, v in config.items() if v not in (None, "", [], {})}


class PluginPanel(QWidget):
    """
    Panel with one card per available plugin window.

    Parameters
    ----------
    selection_callback : callable
        Callback invoked with the selected plugin name.
    config_path : str, optional
        Configuration file: procedures whose section it does not define are
        not listed (all are listed without a file).
    procedures : sequence of Procedure, optional
        The cards, in order (default: :data:`PROCEDURES`).
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        selection_callback: Callable[[str], None],
        parent: Optional[QWidget] = None,
        *,
        config_path: Optional[str] = None,
        procedures: Optional[Sequence[Procedure]] = None,
    ) -> None:
        """Build one card per plugin."""
        super().__init__(parent)
        self._selection_callback = selection_callback
        self.procedures: List[Procedure] = list(
            PROCEDURES if procedures is None else procedures
        )
        self._icons: List[Tuple[object, str]] = []
        self._cards: Dict[str, QFrame] = {}

        import qtawesome as qta

        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(8, 8, 8, 8)
        column.setSpacing(6)
        caption = QLabel("Open a dedicated window for a bench procedure.")
        caption.setProperty("muted", True)
        caption.setWordWrap(True)
        column.addWidget(caption)
        for procedure in self.procedures:
            name, icon_name, description = (
                procedure.name,
                procedure.icon,
                procedure.description,
            )
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
            self._cards[name] = card
        self._hidden_note = QLabel()
        self._hidden_note.setProperty("muted", True)
        self._hidden_note.setWordWrap(True)
        column.addWidget(self._hidden_note)
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
        self.set_config_path(config_path)

    def available(self) -> List[str]:
        """
        Return the procedures listed in the panel.

        Returns
        -------
        list of str
            Plugin names, in panel order.
        """
        return [name for name, card in self._cards.items() if not card.isHidden()]

    def set_config_path(self, config_path: Optional[str]) -> None:
        """
        List the procedures that the configuration file sets up.

        Parameters
        ----------
        config_path : str or None
            Path to the ``configuration.yaml`` file (re-read on every call).
        """
        self._config_path = config_path
        sections = defined_sections(config_path)
        hidden = []
        for procedure in self.procedures:
            name, section = procedure.name, procedure.section
            shown = section is None or sections is None or section in sections
            self._cards[name].setHidden(not shown)
            if not shown:
                hidden.append(f"{name} ({section})")
        self._hidden_note.setText(
            "Not configured: " + ", ".join(hidden) + ". Add the section to the "
            "configuration file to use them."
            if hidden
            else ""
        )
        self._hidden_note.setVisible(bool(hidden))

    def reload(self) -> None:
        """Re-read the configuration file."""
        self.set_config_path(self._config_path)

    def window_class(self, name: str) -> Optional[type]:
        """
        Return the window class of a procedure.

        Parameters
        ----------
        name : str
            Procedure name.

        Returns
        -------
        type or None
            The window class, or ``None`` for an unknown name.
        """
        return next((p.window for p in self.procedures if p.name == name), None)

    def _apply_theme(self) -> None:
        for icon, name in self._icons:
            icon.setIcon(theme().icon(name, "accent"))


#: Window class of each of opticalib's procedures, by name.
PLUGIN_WINDOWS: Dict[str, type] = {p.name: p.window for p in PROCEDURES}

#: Former name of the procedure window base class.
PluginWindowBase = ProcedureWindow
