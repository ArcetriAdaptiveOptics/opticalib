"""
Tests for the CalpyGUI activity panel (opticalib.gui.activity.ActivityCenter).
"""

import pytest

pytest.importorskip("qtconsole", exc_type=ImportError)
pytest.importorskip("qtawesome", exc_type=ImportError)


@pytest.fixture
def window(qapp):
    """A main window with a central widget, a menu bar and a status bar."""
    from qtpy.QtCore import QSettings
    from qtpy.QtWidgets import QLabel, QMainWindow

    from opticalib.gui.theme import SETTINGS_APP, SETTINGS_ORG

    settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
    for key in ("anchor", "offset_x", "offset_y", "pinned", "collapsed"):
        settings.remove(f"activity/{key}")
    win = QMainWindow()
    win.setCentralWidget(QLabel("plots"))
    win.menuBar().addMenu("File")
    win.statusBar().showMessage("status")
    win.resize(1000, 700)
    win.show()
    yield win
    win.close()


@pytest.fixture
def center(window):
    """An activity panel showing one running task."""
    from opticalib.gui.activity import ActivityCenter
    from opticalib.gui.kernel import Task

    panel = ActivityCenter(window)
    task = Task("x = 1", "running task")
    task._start("m1")
    panel.track(task)
    panel._reposition()
    return panel


def test_default_position_is_the_plot_area(window, center):
    assert center.anchor is None
    expected = window.centralWidget().geometry().topLeft() + center.DEFAULT_OFFSET
    assert center.geometry().topLeft() == expected


@pytest.mark.parametrize(
    "fx, fy, anchor",
    [(0.1, 0.1, "top-left"), (0.9, 0.1, "top-right"), (0.1, 0.9, "bottom-left"), (0.9, 0.9, "bottom-right")],
)
def test_drop_pins_to_nearest_corner(window, center, fx, fy, anchor):
    from qtpy.QtCore import QPoint, QRect

    area = center._area()
    size = center.size()
    x = int(area.left() + fx * (area.width() - size.width()))
    y = int(area.top() + fy * (area.height() - size.height()))
    got, offset = center.nearest_anchor(QRect(QPoint(x, y), size))
    assert got == anchor
    assert offset.x() >= 0 and offset.y() >= 0


def test_anchor_follows_window_resize(window, qt_wait, center):
    from qtpy.QtCore import QPoint

    center.move_to("bottom-right", QPoint(20, 30))
    area = center._area()
    geo = center.geometry()
    assert area.left() + area.width() - (geo.left() + geo.width()) == 20
    assert area.top() + area.height() - (geo.top() + geo.height()) == 30
    window.resize(800, 600)
    qt_wait(lambda: False, timeout=0.2)
    area, geo = center._area(), center.geometry()
    assert area.left() + area.width() - (geo.left() + geo.width()) == 20


def test_drag_moves_and_pins(window, center):
    from qtpy.QtCore import QPoint

    header = center._header
    start = header.mapToGlobal(QPoint(30, 10))
    target = window.mapToGlobal(QPoint(window.width() - 100, window.height() - 100))
    header.drag_started.emit(start)
    header.drag_moved.emit(target)
    header.drag_finished.emit()
    assert center.anchor == "bottom-right"


def test_pin_blocks_dragging(window, center):
    from qtpy.QtCore import QPoint, QSettings

    from opticalib.gui.theme import SETTINGS_APP, SETTINGS_ORG

    center.move_to("top-right", QPoint(10, 10))
    center.set_pinned(True)
    assert not center._header.draggable
    center.set_pinned(False)
    assert center._header.draggable
    center.set_collapsed(True)
    assert not center._cards_box.isVisibleTo(center)
    settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
    assert settings.value("activity/anchor") == "top-right"
    assert str(settings.value("activity/collapsed")).lower() == "true"


def test_state_is_restored_and_reset(window, center):
    from qtpy.QtCore import QPoint

    from opticalib.gui.activity import ActivityCenter

    center.move_to("bottom-left", QPoint(12, 34))
    center.set_pinned(True)
    other = ActivityCenter(window)
    assert other.anchor == "bottom-left" and other.offset == QPoint(12, 34)
    assert other.pinned
    other.reset_position()
    assert ActivityCenter(window).anchor is None


def test_summary_and_clear_finished(window, center):
    from opticalib.gui.kernel import Task

    done = Task("y = 2", "finished task")
    done._start("m2")
    done._finish("done")
    center.track(done)
    assert center.summary_text() == "Activity · 1 running · 1 done"
    center.clear_finished()
    assert [c.job.title for c in center.cards] == ["running task"]
