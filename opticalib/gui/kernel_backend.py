"""
Matplotlib backend used inside the CalpyGUI kernel
==================================================

Figures are rendered off-screen with Agg and published to the GUI by
:func:`opticalib.gui.kernel_side.publish_figures`, which runs after every
cell and whenever ``plt.show()`` / ``plt.pause()`` is called, so plots
updated inside a loop are streamed live to the plot panel.

The backend is selected by the GUI through the ``MPLBACKEND`` environment
variable of the kernel process::

    MPLBACKEND=module://opticalib.gui.kernel_backend

This module must not import Qt.
"""

from matplotlib.backend_bases import FigureManagerBase
from matplotlib.backends.backend_agg import FigureCanvasAgg


def _publish(figures=None) -> None:
    """Publish changed figures (or the given *figures*) to the GUI."""
    from . import kernel_side

    kernel_side.publish_figures(figures=figures)


class FigureManagerCalpy(FigureManagerBase):
    """Figure manager that sends figures to the GUI instead of a window."""

    def show(self) -> None:
        """Publish this figure to the GUI plot panel."""
        _publish([self.canvas.figure])

    @classmethod
    def pyplot_show(cls, *, block=None) -> None:
        """Publish every changed figure to the GUI plot panel."""
        _publish()


class FigureCanvasCalpy(FigureCanvasAgg):
    """
    Agg canvas whose ``draw_idle`` only marks the figure as changed.

    In interactive mode (``ion()``) matplotlib calls ``draw_idle`` after
    every pyplot command; rendering is deferred to publication time instead.
    """

    manager_class = FigureManagerCalpy

    def draw_idle(self, *args, **kwargs) -> None:
        """Mark the figure for publication instead of rendering it now."""
        self._calpy_dirty = True


FigureCanvas = FigureCanvasCalpy
FigureManager = FigureManagerCalpy


def show(*args, **kwargs) -> None:
    """``plt.show()`` entry point: publish every changed figure."""
    _publish()
