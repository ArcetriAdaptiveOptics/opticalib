"""
Tests for ``opticalib.devices.cameras`` without the ``vmbpy`` SDK.

``vmbpy`` has no macOS distribution, so it is not installed there (see the
``sys_platform`` marker in ``pyproject.toml``).  ``opticalib.devices`` must
still import, and only creating a ``GigaVision`` camera must fail.  This module
lives apart from ``test_devices_cameras.py``, which is skipped entirely when
``vmbpy`` is missing.
"""

import sys
import importlib

import pytest

import opticalib.devices as devices
from opticalib.devices import cameras


@pytest.fixture
def cameras_without_vmbpy(monkeypatch):
    """Reload ``cameras`` with ``import vmbpy`` failing, restore it afterwards."""
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "vmbpy", None)  # None -> ImportError on import
        yield importlib.reload(cameras)
    importlib.reload(cameras)
    # reload() creates a new class: keep the package re-export pointing at it.
    devices.GigaVision = cameras.GigaVision


def test_module_imports_without_vmbpy(cameras_without_vmbpy):
    assert cameras_without_vmbpy._vmbpy is None
    assert cameras_without_vmbpy._FEATURE_ERRORS == (AttributeError,)
    assert cameras_without_vmbpy._ACQUISITION_ERRORS == (AttributeError,)


def test_gigavision_raises_import_error_without_vmbpy(cameras_without_vmbpy):
    with pytest.raises(ImportError, match="vmbpy") as excinfo:
        cameras_without_vmbpy.GigaVision("any_camera")
    # the original import failure is chained, for diagnosing broken installs
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_devices_package_exports_gigavision():
    assert "GigaVision" in devices.__all__
    assert devices.GigaVision is cameras.GigaVision


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only: vmbpy is excluded from the install there")
def test_vmbpy_not_installed_on_macos():
    assert cameras._vmbpy is None
