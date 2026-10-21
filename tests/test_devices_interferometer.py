"""Tests for opticalib.devices.interferometer static helper methods."""

import os
from unittest.mock import MagicMock, call

import numpy as np
import numpy.ma as ma
import pytest

from opticalib.devices.interferometer import _4DInterferometer


class Test4DInterferometerStatics:
    """Tests for static methods in _4DInterferometer."""

    def test_get_camera_settings_uses_new_width_method_no_tn(self, monkeypatch):
        """Test camera settings retrieval with the renamed width accessor."""

        class ReaderStub:
            """Reader stub exposing only the new width API."""

            def get_image_width_in_pixels(self):
                return 2000

            def get_image_height_in_pixels(self):
                return 1500

            def get_offset_x(self):
                return 11

            def get_offset_y(self):
                return 22

        monkeypatch.setattr(
            "opticalib.devices.interferometer._confReader",
            lambda _path: ReaderStub(),
        )
        monkeypatch.setattr(
            "opticalib.devices.interferometer._Folds.SETTINGS_CONF_FILE",
            "/file/to/config.ini",
        )

        settings = _4DInterferometer.get_camera_settings()

        assert settings == [2000, 1500, 11, 22]

    def test_get_frame_rate_reads_from_reader_no_tn(self, monkeypatch):
        """Test frame-rate retrieval through the configured reader."""

        class ReaderStub:
            """Reader stub for frame-rate retrieval."""

            def get_frame_rate(self):
                return 77.5

        monkeypatch.setattr(
            "opticalib.devices.interferometer._confReader",
            lambda _path: ReaderStub(),
        )
        monkeypatch.setattr(
            "opticalib.devices.interferometer._Folds.SETTINGS_CONF_FILE",
            "/file/to/config.ini",
        )

        frame_rate = _4DInterferometer.get_frame_rate()

        assert frame_rate == 77.5

    def test_into_full_frame_uses_camera_offsets_when_offset_is_none(
        self, monkeypatch
    ):
        """Test that into_full_frame works with default optional offset."""
        monkeypatch.setattr(
            "opticalib.devices.interferometer._4DInterferometer.get_camera_settings",
            staticmethod(lambda _tn=None: [2048, 2048, 4, 7]),
        )

        img = ma.masked_array(np.array([[1.0, 2.0], [3.0, 4.0]]))

        fullimg = _4DInterferometer.into_full_frame(img)

        assert fullimg.shape == (2048, 2048)
        np.testing.assert_array_equal(fullimg[7:9, 4:6], img.data)


    def test_into_full_frame_reads_offsets_from_config_path(self, monkeypatch):
        """Test config_path-based offset loading for full-frame insertion."""
        reader = MagicMock()
        reader.get_offset_x.return_value = 0
        reader.get_offset_y.return_value = 0
        monkeypatch.setattr(
            "opticalib.devices.interferometer._fn.ConfSettingReader4D",
            lambda _path: reader,
        )

        img = ma.masked_array(np.array([[3.0, 1.0]]))

        fullimg = _4DInterferometer.into_full_frame(
            img,
            config_path="/tmp/fake_4d_settings.ini",
        )

        np.testing.assert_array_equal(fullimg[0:1, 0:2], img.data)


# ---------------------------------------------------------------------------
# Settings-file read/write methods
# ---------------------------------------------------------------------------
_MOD = "opticalib.devices.interferometer"
_INI_TEMPLATE = """[ACA2440]
FrameRate = {fr}
ImageWidthInPixels = {w}
ImageHeightInPixels = {h}
OffsetX = {ox}
OffsetY = {oy}
PixelFormat = Mono12
"""


def _write_ini(
    path, fr: float = 10.5, w: int = 100, h: int = 80, ox: int = 5, oy: int = 7
) -> str:
    """Write a real camera-settings ini file and return its path as str."""
    path.write_text(_INI_TEMPLATE.format(fr=fr, w=w, h=h, ox=ox, oy=oy))
    return str(path)


@pytest.fixture
def inst() -> _4DInterferometer:
    """Hardware-free interferometer with mocked ``_i4d`` and ``_logger``."""
    obj = object.__new__(_4DInterferometer)
    obj._i4d = MagicMock()
    obj._logger = MagicMock()
    return obj


def _reject_directories(path):
    """Backend side effect modelling a 4D server that rejects directories."""
    if os.path.splitext(str(path))[1] == "":
        raise RuntimeError(f"cannot write to directory '{path}'")


class TestSaveLoadConfiguration:
    """Tests for ``save_configuration`` and ``load_configuration``."""

    @pytest.fixture(params=["save", "load"])
    def op(self, request, inst):
        """Return (method, i4d_mock_method) with a directory-rejecting backend."""
        if request.param == "save":
            method, target = inst.save_configuration, inst._i4d.saveConfiguration
        else:
            method, target = inst.load_configuration, inst._i4d.loadConfiguration
        target.side_effect = _reject_directories
        return method, target

    def test_full_path_accepted_first_try(self, op):
        """A full (e.g. remote) file path is forwarded once, without a join."""
        method, target = op
        remote = "/remote/4d/capture/tn/SoftwareSettings.4dini"
        method(remote)
        target.assert_called_once_with(remote)

    def test_existing_file_used_as_is(self, op, tmp_path):
        """An existing local file path is forwarded unchanged, once."""
        f = tmp_path / "my.4dini"
        f.write_text("x")
        method, target = op
        method(str(f), filename="other.4dini")
        target.assert_called_once_with(str(f))

    def test_directory_rejected_then_default_filename(self, op, tmp_path):
        """A rejected directory is retried with the default filename."""
        method, target = op
        method(str(tmp_path))
        assert target.call_args_list == [
            call(str(tmp_path)),
            call(os.path.join(str(tmp_path), "SoftwareSettings.4dini")),
        ]

    def test_directory_rejected_then_custom_filename(self, op, tmp_path):
        """A rejected directory is retried with the custom filename."""
        method, target = op
        method(str(tmp_path), filename="custom.4dini")
        assert target.call_args_list == [
            call(str(tmp_path)),
            call(os.path.join(str(tmp_path), "custom.4dini")),
        ]

    def test_error_is_printed_on_first_failure(self, op, tmp_path, capsys):
        """The first exception is printed before the retry."""
        method, _ = op
        method(str(tmp_path))
        assert "cannot write to directory" in capsys.readouterr().out

    def test_existing_file_rejected_retried_with_joined_path(self, op, tmp_path):
        """A rejected path is always retried joined, even if it is a local file."""
        f = tmp_path / "my.4dini"
        f.write_text("x")
        method, target = op
        target.side_effect = [RuntimeError("transient"), None]
        method(str(f))
        assert target.call_args_list == [
            call(str(f)),
            call(os.path.join(str(f), "SoftwareSettings.4dini")),
        ]

    def test_error_logged_once_on_retry_path(self, op, inst, tmp_path):
        """``_logger.error`` is called exactly once when the retry happens."""
        method, _ = op
        method(str(tmp_path))
        inst._logger.error.assert_called_once()
        assert str(tmp_path) in inst._logger.error.call_args[0][0]

    def test_no_error_logged_on_first_try_success(self, op, inst):
        """No error is logged when the first call succeeds."""
        method, _ = op
        method("/remote/x.4dini")
        inst._logger.error.assert_not_called()

    def test_existing_file_rejected_twice_propagates(self, op, tmp_path):
        """A path failing twice propagates the second error."""
        f = tmp_path / "my.4dini"
        f.write_text("x")
        method, target = op
        target.side_effect = [RuntimeError("first"), ValueError("second")]
        with pytest.raises(ValueError, match="second"):
            method(str(f))
        assert target.call_count == 2

    def test_both_calls_fail_propagates_and_no_info_log(self, op, inst, tmp_path):
        """If both calls fail the error propagates and no info is logged."""
        method, target = op
        target.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            method(str(tmp_path))
        assert target.call_count == 2
        inst._logger.error.assert_called_once()
        inst._logger.info.assert_not_called()

    def test_nonexistent_path_failing_backend(self, op):
        """A failing backend on a missing path is retried once, then raises."""
        method, target = op
        target.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            method("/nonexistent/dir")
        assert target.call_args_list == [
            call("/nonexistent/dir"),
            call(os.path.join("/nonexistent/dir", "SoftwareSettings.4dini")),
        ]

    def test_lenient_backend_directory_used_as_is(self, op, tmp_path):
        """Pin current behavior: a backend not rejecting directories gets them.

        This relies on the 4D server rejecting directory paths; if it does
        not, the default filename is never appended.
        """
        method, target = op
        target.side_effect = None
        method(str(tmp_path))
        target.assert_called_once_with(str(tmp_path))

    def test_logs_info_with_final_path(self, op, inst, tmp_path):
        """The operation is logged once with the final (joined) path."""
        method, _ = op
        method(str(tmp_path))
        inst._logger.info.assert_called_once()
        msg = inst._logger.info.call_args[0][0]
        assert os.path.join(str(tmp_path), "SoftwareSettings.4dini") in msg

    def test_logs_info_with_given_path_when_accepted(self, op, inst):
        """When accepted on the first try the given path is logged."""
        method, _ = op
        method("/remote/x.4dini")
        inst._logger.info.assert_called_once()
        assert "/remote/x.4dini" in inst._logger.info.call_args[0][0]

    def test_filename_none_raises_on_retry(self, op, tmp_path):
        """``filename`` must be a str: ``None`` raises TypeError on the retry."""
        method, target = op
        with pytest.raises(TypeError):
            method(str(tmp_path), filename=None)
        target.assert_called_once_with(str(tmp_path))

    def test_filename_none_not_reached_when_accepted(self, op):
        """``filename=None`` is harmless if the first call succeeds."""
        method, target = op
        method("/remote/x.4dini", filename=None)
        target.assert_called_once_with("/remote/x.4dini")

    def test_none_path_lenient_backend_receives_none(self, op, inst):
        """Pin current behavior: ``conf_path=None`` is forwarded to the backend."""
        method, target = op
        target.side_effect = None
        method(None)
        target.assert_called_once_with(None)

    def test_none_path_failing_backend_raises_type_error(self, op, inst):
        """Pin current behavior: None path + failing backend -> TypeError.

        The retry's ``os.path.join(None, filename)`` raises after the error
        has been logged once.
        """
        method, target = op
        target.side_effect = RuntimeError("boom")
        with pytest.raises(TypeError):
            method(None)
        target.assert_called_once_with(None)
        inst._logger.error.assert_called_once()


class TestCaptureProduceConfigIntegration:
    """Check which path ``capture``/``produce`` pass to the I4D backend."""

    @pytest.fixture(autouse=True)
    def _patches(self, monkeypatch, inst, tmp_path):
        """Patch folders, shutil, rename and settings copy."""
        for attr, val in {
            "CAPTURE_FOLDER_NAME_4D_PC": "/4d/capture",
            "PRODUCE_FOLDER_NAME_4D_PC": "/4d/produce",
            "CAPTURE_FOLDER_NAME_LOCAL_PC": str(tmp_path / "cap"),
            "PRODUCE_FOLDER_NAME_LOCAL_PC": str(tmp_path / "prod"),
            "OPD_IMAGES_ROOT_FOLDER": str(tmp_path / "opd"),
        }.items():
            monkeypatch.setattr(f"{_MOD}._Folds.{attr}", val)
        self.move = MagicMock()
        self.copy = MagicMock()
        monkeypatch.setattr(f"{_MOD}._sh.move", self.move)
        monkeypatch.setattr(f"{_MOD}._sh.copy", self.copy)
        inst._rename_4d_files = MagicMock()
        self.cp4d = MagicMock()
        inst.copy_4d_settings = self.cp4d
        monkeypatch.setattr(f"{_MOD}._osu.newtn", lambda: "20240101_000000")
        self.inst = inst

    def test_capture_saves_remote_path(self):
        """capture() forwards the full remote file path exactly once."""
        self.inst.capture(3, folder_name="tnA")
        self.inst._i4d.burstFramesToSpecificDirectory.assert_called_once_with(
            "/4d/capture/tnA", 3
        )
        self.inst._i4d.saveConfiguration.assert_called_once_with(
            "/4d/capture/tnA/SoftwareSettings.4dini"
        )

    def test_capture_burst_and_return(self):
        """capture() bursts into the 4D PC folder and returns the name."""
        tn = self.inst.capture(3, folder_name="tnA")
        assert tn == "tnA"
        self.inst._i4d.burstFramesToSpecificDirectory.assert_called_once_with(
            "/4d/capture/tnA", 3
        )

    def test_capture_generates_tn(self):
        """capture() without folder name uses ``newtn``."""
        assert self.inst.capture(1) == "20240101_000000"

    def test_produce_true_loads_capture_path(self):
        """produce(load_interf_config=True) loads the capture-folder settings."""
        self.inst.produce("tnA", load_interf_config=True)
        self.inst._i4d.loadConfiguration.assert_called_once_with(
            "/4d/capture/tnA/SoftwareSettings.4dini"
        )
        self.inst._rename_4d_files.assert_called_once_with("tnA")

    def test_produce_str_loads_given_path(self):
        """A string load_interf_config is the path handed to loadConfiguration."""
        self.inst.produce("tnA", load_interf_config="/some/remote/x.4dini")
        self.inst._i4d.loadConfiguration.assert_called_once_with(
            "/some/remote/x.4dini"
        )

    def test_produce_false_no_load(self):
        """By default no configuration is loaded."""
        self.inst.produce("tnA")
        self.inst._i4d.loadConfiguration.assert_not_called()

    def test_produce_list_of_tns(self):
        """produce() iterates over a list of tracking numbers."""
        self.inst.produce(["a", "b"], load_interf_config=True)
        assert self.inst._i4d.loadConfiguration.call_count == 2
        assert self.cp4d.call_count == 2
        assert self.inst._rename_4d_files.call_count == 2

    def test_produce_copy_failure_is_swallowed(self):
        """A failing SoftwareSettings copy is printed, not raised."""
        self.copy.side_effect = OSError("nope")
        self.inst.produce("tnA")
        self.cp4d.assert_called_once()
        assert self.cp4d.call_args.kwargs["iscapture"] is False


class TestCopy4DSettings:
    """Tests for ``copy_4d_settings`` with real files."""

    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        """Create src file, dest dir and neutral config values."""
        src_dir = tmp_path / "src"
        dest = tmp_path / "dest"
        src_dir.mkdir()
        dest.mkdir()
        monkeypatch.setattr(f"{_MOD}._Folds.COPIED_SETTINGS_CONF_FILE", None)
        monkeypatch.setattr(f"{_MOD}._Folds.SETTINGS_CONF_FILE", None)
        return src_dir, dest

    def test_capture_with_src_file(self, inst, env):
        """iscapture=True: src is the file itself, copied under default name."""
        src_dir, dest = env
        f = src_dir / "orig.ini"
        f.write_text("content")
        inst.copy_4d_settings(str(dest), src=str(f))
        assert (dest / "CameraSettings.ini").read_text() == "content"
        inst._logger.info.assert_called_once()

    def test_capture_src_none_uses_config(self, inst, env, monkeypatch):
        """src=None uses ``_Folds.SETTINGS_CONF_FILE``."""
        src_dir, dest = env
        f = src_dir / "live.ini"
        f.write_text("live")
        monkeypatch.setattr(f"{_MOD}._Folds.SETTINGS_CONF_FILE", str(f))
        inst.copy_4d_settings(str(dest))
        assert (dest / "CameraSettings.ini").read_text() == "live"

    def test_not_capture_src_is_folder(self, inst, env):
        """iscapture=False: src is a folder containing ``copied_name``."""
        src_dir, dest = env
        (src_dir / "CameraSettings.ini").write_text("folder")
        inst.copy_4d_settings(str(dest), src=str(src_dir), iscapture=False)
        assert (dest / "CameraSettings.ini").read_text() == "folder"

    def test_custom_copied_name(self, inst, env):
        """An explicit copied_name is used when config value is None."""
        src_dir, dest = env
        f = src_dir / "a.ini"
        f.write_text("c")
        inst.copy_4d_settings(str(dest), src=str(f), copied_name="mine.ini")
        assert (dest / "mine.ini").read_text() == "c"
        assert not (dest / "CameraSettings.ini").exists()

    def test_config_name_used_when_set(self, inst, env, monkeypatch):
        """COPIED_SETTINGS_CONF_FILE from config names the copy."""
        src_dir, dest = env
        f = src_dir / "a.ini"
        f.write_text("c")
        monkeypatch.setattr(f"{_MOD}._Folds.COPIED_SETTINGS_CONF_FILE", "cfg.ini")
        inst.copy_4d_settings(str(dest), src=str(f))
        assert (dest / "cfg.ini").exists()

    def test_explicit_name_beats_config(self, inst, env, monkeypatch):
        """An explicitly passed copied_name wins over the config value."""
        src_dir, dest = env
        f = src_dir / "a.ini"
        f.write_text("c")
        monkeypatch.setattr(f"{_MOD}._Folds.COPIED_SETTINGS_CONF_FILE", "cfg.ini")
        inst.copy_4d_settings(str(dest), src=str(f), copied_name="explicit.ini")
        assert (dest / "explicit.ini").exists()
        assert not (dest / "cfg.ini").exists()

    def test_default_name_config_none(self, inst, env):
        """copied_name=None and no config -> 'CameraSettings.ini'."""
        src_dir, dest = env
        f = src_dir / "a.ini"
        f.write_text("c")
        inst.copy_4d_settings(str(dest), src=str(f), copied_name=None)
        assert (dest / "CameraSettings.ini").exists()

    def test_default_name_config_empty_string(self, inst, env, monkeypatch):
        """An empty-string config value is falsy -> 'CameraSettings.ini'."""
        src_dir, dest = env
        f = src_dir / "a.ini"
        f.write_text("c")
        monkeypatch.setattr(f"{_MOD}._Folds.COPIED_SETTINGS_CONF_FILE", "")
        inst.copy_4d_settings(str(dest), src=str(f))
        assert (dest / "CameraSettings.ini").exists()

    def test_default_name_from_config_not_capture(self, inst, env, monkeypatch):
        """iscapture=False: the config name is also used to locate the source."""
        src_dir, dest = env
        (src_dir / "cfg.ini").write_text("cfgdata")
        monkeypatch.setattr(f"{_MOD}._Folds.COPIED_SETTINGS_CONF_FILE", "cfg.ini")
        inst.copy_4d_settings(str(dest), src=str(src_dir), iscapture=False)
        assert (dest / "cfg.ini").read_text() == "cfgdata"

    def test_explicit_name_config_none(self, inst, env):
        """An explicit copied_name is used when the config value is None."""
        src_dir, dest = env
        f = src_dir / "a.ini"
        f.write_text("c")
        inst.copy_4d_settings(str(dest), src=str(f), copied_name="mine.ini")
        assert (dest / "mine.ini").exists()

    def test_missing_source_raises(self, inst, env):
        """A missing source file raises FileNotFoundError."""
        src_dir, dest = env
        with pytest.raises(FileNotFoundError):
            inst.copy_4d_settings(str(dest), src=str(src_dir / "nope.ini"))
        inst._logger.info.assert_not_called()

    def test_missing_source_in_folder_raises(self, inst, env):
        """iscapture=False with an empty source folder raises."""
        src_dir, dest = env
        with pytest.raises(FileNotFoundError):
            inst.copy_4d_settings(str(dest), src=str(src_dir), iscapture=False)

    def test_missing_dest_raises(self, inst, env):
        """A non-existing destination folder raises FileNotFoundError."""
        src_dir, dest = env
        f = src_dir / "a.ini"
        f.write_text("c")
        with pytest.raises(FileNotFoundError):
            inst.copy_4d_settings(str(dest / "missing"), src=str(f))

    def test_no_src_and_no_config_raises(self, inst, env):
        """src=None with SETTINGS_CONF_FILE None raises (TypeError today)."""
        _, dest = env
        with pytest.raises((TypeError, FileNotFoundError)):
            inst.copy_4d_settings(str(dest))

    def test_no_src_not_capture_no_config_raises(self, inst, env):
        """iscapture=False and no source at all raises."""
        _, dest = env
        with pytest.raises((TypeError, FileNotFoundError)):
            inst.copy_4d_settings(str(dest), iscapture=False)


class TestGetCameraSettingsAndFrameRate:
    """Tests for ``get_camera_settings`` and ``get_frame_rate`` on real ini."""

    @pytest.fixture(params=["settings", "rate"])
    def getter(self, request):
        """Parametrize over the two sibling static methods."""
        if request.param == "settings":
            return _4DInterferometer.get_camera_settings, [100, 80, 5, 7]
        return _4DInterferometer.get_frame_rate, 10.5

    @pytest.fixture
    def tnfold(self, tmp_path, monkeypatch):
        """Tn folder with a real ini; find_tracknum patched to return it."""
        fold = tmp_path / "tn"
        fold.mkdir()
        monkeypatch.setattr(f"{_MOD}._Folds.SETTINGS_CONF_FILE", None)
        monkeypatch.setattr(f"{_MOD}._osu.find_tracknum", lambda tn, complete_path=False: str(fold))
        monkeypatch.setattr(f"{_MOD}._fn.COPIED_SETTINGS_CONF_FILE", "Copied.ini")
        return fold

    def test_no_tn_real_file(self, getter, tmp_path, monkeypatch):
        """Without tn, the live settings file is read with correct types."""
        fn, expected = getter
        ini = _write_ini(tmp_path / "live.ini")
        monkeypatch.setattr(f"{_MOD}._Folds.SETTINGS_CONF_FILE", ini)
        out = fn()
        assert out == expected
        if isinstance(expected, list):
            assert all(type(v) is int for v in out)
        else:
            assert type(out) is float

    def test_no_tn_no_file_raises(self, getter, monkeypatch):
        """No tn and SETTINGS_CONF_FILE None raises FileNotFoundError."""
        fn, _ = getter
        monkeypatch.setattr(f"{_MOD}._Folds.SETTINGS_CONF_FILE", None)
        with pytest.raises(FileNotFoundError):
            fn()

    def test_tn_with_copied_name(self, getter, tnfold):
        """With tn, the COPIED_SETTINGS_CONF_FILE file is read."""
        fn, expected = getter
        _write_ini(tnfold / "Copied.ini")
        assert fn(tn="x") == expected

    def test_tn_copied_name_none_falls_back(self, getter, tnfold, monkeypatch):
        """COPIED_SETTINGS_CONF_FILE None -> TypeError -> default filename."""
        fn, expected = getter
        monkeypatch.setattr(f"{_MOD}._fn.COPIED_SETTINGS_CONF_FILE", None)
        _write_ini(tnfold / "CameraSettings.ini")
        assert fn(tn="x") == expected

    def test_tn_custom_filename_fallback(self, getter, tnfold, monkeypatch):
        """A custom ``filename`` is used on fallback."""
        fn, expected = getter
        monkeypatch.setattr(f"{_MOD}._fn.COPIED_SETTINGS_CONF_FILE", None)
        _write_ini(tnfold / "my.ini")
        assert fn(tn="x", filename="my.ini") == expected

    def test_tn_copied_name_wins_over_filename(self, getter, tnfold):
        """When the copied-name file exists, ``filename`` is not consulted."""
        fn, expected = getter
        _write_ini(tnfold / "Copied.ini")
        _write_ini(tnfold / "other.ini", fr=99.0, w=1, h=2, ox=3, oy=4)
        assert fn(tn="x", filename="other.ini") == expected

    def test_tn_copied_name_missing_falls_back(self, getter, tnfold):
        """A missing copied-name file falls back to ``filename``."""
        fn, expected = getter
        _write_ini(tnfold / "CameraSettings.ini")
        assert fn(tn="x") == expected

    def test_find_tracknum_list_resolved_end_to_end(self, getter, tnfold, monkeypatch):
        """A list from find_tracknum is resolved and the right values are read."""
        fn, expected = getter
        other = tnfold.parent / "other"
        other.mkdir()
        _write_ini(tnfold / "Copied.ini")
        monkeypatch.setattr(
            f"{_MOD}._osu.find_tracknum",
            lambda tn, complete_path=False: [str(other), str(tnfold)],
        )
        assert fn(tn="x") == expected

    def test_tn_nothing_on_disk_raises_on_disk_raises(self, getter, tnfold, monkeypatch):
        """No ini at all in the tn folder raises FileNotFoundError."""
        fn, _ = getter
        monkeypatch.setattr(f"{_MOD}._fn.COPIED_SETTINGS_CONF_FILE", None)
        with pytest.raises(FileNotFoundError):
            fn(tn="x")

    def test_find_tracknum_called_with_complete_path(self, getter, tnfold, monkeypatch):
        """find_tracknum receives the tn and complete_path=True."""
        fn, _ = getter
        _write_ini(tnfold / "Copied.ini")
        mock = MagicMock(return_value=str(tnfold))
        monkeypatch.setattr(f"{_MOD}._osu.find_tracknum", mock)
        fn(tn="abc")
        mock.assert_called_once_with("abc", complete_path=True)

    def test_malformed_values_raise_value_error(self, tmp_path, monkeypatch):
        """Non-numeric values in the ini raise ValueError."""
        f = tmp_path / "bad.ini"
        f.write_text("[ACA2440]\nFrameRate = abc\nImageWidthInPixels = x\n")
        monkeypatch.setattr(f"{_MOD}._Folds.SETTINGS_CONF_FILE", str(f))
        with pytest.raises(ValueError):
            _4DInterferometer.get_frame_rate()
        with pytest.raises(ValueError):
            _4DInterferometer.get_camera_settings()


class TestLocateSettingsFile:
    """Tests for ``_4DInterferometer._locate_settings_file``."""

    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        """Patch find_tracknum, OPDImages root and the copied-name config."""
        opd = tmp_path / "opd"
        opd.mkdir()
        monkeypatch.setattr(f"{_MOD}._Folds.OPD_IMAGES_ROOT_FOLDER", str(opd))
        monkeypatch.setattr(f"{_MOD}._fn.COPIED_SETTINGS_CONF_FILE", "Copied.ini")
        state = {"ret": []}
        monkeypatch.setattr(
            f"{_MOD}._osu.find_tracknum",
            lambda tn, complete_path=False: state["ret"],
        )
        return opd, state, tmp_path

    @staticmethod
    def _folder(root, name):
        d = root / name
        d.mkdir()
        return d

    @pytest.mark.parametrize("as_list", [False, True])
    def test_str_or_single_element_list(self, env, as_list):
        """A str and a one-element list from find_tracknum are equivalent."""
        _, state, root = env
        d = self._folder(root, "a")
        f = _write_ini(d / "Copied.ini")
        state["ret"] = [str(d)] if as_list else str(d)
        assert _4DInterferometer._locate_settings_file("tn") == f

    @pytest.mark.parametrize("empty", [[], ""])
    def test_empty_raises_mentioning_tn(self, env, empty):
        """An empty result raises FileNotFoundError mentioning the tn."""
        _, state, _ = env
        state["ret"] = empty
        with pytest.raises(FileNotFoundError, match="20240101_000000"):
            _4DInterferometer._locate_settings_file("20240101_000000")

    def test_copied_name_preferred(self, env):
        """The copied name wins over ``filename`` in the same folder."""
        _, state, root = env
        d = self._folder(root, "a")
        f = _write_ini(d / "Copied.ini")
        _write_ini(d / "CameraSettings.ini")
        state["ret"] = str(d)
        assert _4DInterferometer._locate_settings_file("tn") == f

    @pytest.mark.parametrize("copied", [None, "", "Missing.ini"])
    def test_fallback_to_filename(self, env, monkeypatch, copied):
        """Falls back to ``filename`` if the copied name is None/empty/missing."""
        _, state, root = env
        monkeypatch.setattr(f"{_MOD}._fn.COPIED_SETTINGS_CONF_FILE", copied)
        d = self._folder(root, "a")
        f = _write_ini(d / "CameraSettings.ini")
        state["ret"] = str(d)
        assert _4DInterferometer._locate_settings_file("tn") == f

    def test_custom_filename(self, env):
        """A custom ``filename`` is looked up."""
        _, state, root = env
        d = self._folder(root, "a")
        f = _write_ini(d / "my.ini")
        _write_ini(d / "CameraSettings.ini")
        state["ret"] = str(d)
        assert _4DInterferometer._locate_settings_file("tn", "my.ini") == f

    def test_directory_named_like_file_is_ignored(self, env):
        """A directory with the settings name is not a match."""
        _, state, root = env
        d = self._folder(root, "a")
        (d / "Copied.ini").mkdir()
        state["ret"] = str(d)
        with pytest.raises(FileNotFoundError):
            _4DInterferometer._locate_settings_file("tn")

    def test_several_folders_only_one_has_file(self, env):
        """Only the folder containing a settings file is returned."""
        _, state, root = env
        a, b, c = (self._folder(root, n) for n in "abc")
        f = _write_ini(b / "CameraSettings.ini")
        state["ret"] = [str(a), str(b), str(c)]
        assert _4DInterferometer._locate_settings_file("tn") == f

    def test_several_folders_one_under_opd(self, env):
        """Among several matches the single OPDImages one is returned."""
        opd, state, root = env
        a = self._folder(root, "a")
        o = self._folder(opd, "tn")
        _write_ini(a / "Copied.ini")
        f = _write_ini(o / "Copied.ini")
        state["ret"] = [str(a), str(o)]
        assert _4DInterferometer._locate_settings_file("tn") == f

    def test_several_folders_none_under_opd_raises(self, env):
        """Ambiguous matches outside OPDImages raise ValueError."""
        _, state, root = env
        a, b = self._folder(root, "a"), self._folder(root, "b")
        _write_ini(a / "Copied.ini")
        _write_ini(b / "Copied.ini")
        state["ret"] = [str(a), str(b)]
        with pytest.raises(ValueError, match="Ambiguous"):
            _4DInterferometer._locate_settings_file("tn")

    def test_several_folders_two_under_opd_raises(self, env):
        """Two matches under OPDImages are still ambiguous."""
        opd, state, root = env
        o1, o2 = self._folder(opd, "t1"), self._folder(opd, "t2")
        _write_ini(o1 / "Copied.ini")
        _write_ini(o2 / "Copied.ini")
        state["ret"] = [str(o1), str(o2)]
        with pytest.raises(ValueError):
            _4DInterferometer._locate_settings_file("tn")

    def test_one_folder_with_both_names_counts_once(self, env):
        """Both names in one folder is a single match (no ambiguity)."""
        _, state, root = env
        a = self._folder(root, "a")
        f = _write_ini(a / "Copied.ini")
        _write_ini(a / "CameraSettings.ini")
        b = self._folder(root, "b")
        state["ret"] = [str(a), str(b)]
        assert _4DInterferometer._locate_settings_file("tn") == f

    def test_no_file_anywhere_raises(self, env):
        """No settings file in any folder raises FileNotFoundError."""
        _, state, root = env
        state["ret"] = [str(self._folder(root, "a")), str(self._folder(root, "b"))]
        with pytest.raises(FileNotFoundError, match="tn"):
            _4DInterferometer._locate_settings_file("tn")

    def test_find_tracknum_called_with_complete_path(self, env, monkeypatch):
        """find_tracknum is queried with ``complete_path=True``."""
        _, state, root = env
        d = self._folder(root, "a")
        _write_ini(d / "Copied.ini")
        mock = MagicMock(return_value=str(d))
        monkeypatch.setattr(f"{_MOD}._osu.find_tracknum", mock)
        _4DInterferometer._locate_settings_file("abc")
        mock.assert_called_once_with("abc", complete_path=True)


class TestIntoFullFrameRealConfig:
    """Tests for ``into_full_frame`` with real ini files."""

    def test_config_path_placement_and_nan_fill(self, tmp_path):
        """Image is placed at (offy, offx) and the rest is NaN/masked."""
        ini = _write_ini(tmp_path / "c.ini", ox=10, oy=20)
        img = ma.masked_array(np.arange(6, dtype=float).reshape(2, 3))
        full = _4DInterferometer.into_full_frame(img, config_path=ini)
        assert full.shape == (2048, 2048)
        np.testing.assert_array_equal(full.data[20:22, 10:13], img.data)
        assert np.ma.getmaskarray(full).sum() == 2048 * 2048 - 6
        assert np.isnan(full.data[0, 0])
        assert not np.ma.getmaskarray(full)[20:22, 10:13].any()

    def test_config_path_takes_precedence_over_tn(self, tmp_path, monkeypatch):
        """config_path given: get_camera_settings is not consulted."""
        ini = _write_ini(tmp_path / "c.ini", ox=1, oy=2)
        called = MagicMock()
        monkeypatch.setattr(
            f"{_MOD}._4DInterferometer.get_camera_settings", staticmethod(called)
        )
        img = ma.masked_array(np.ones((2, 2)))
        full = _4DInterferometer.into_full_frame(img, tn="x", config_path=ini)
        called.assert_not_called()
        np.testing.assert_array_equal(full.data[2:4, 1:3], 1.0)

    def test_tn_forwarded(self, monkeypatch):
        """tn is forwarded to get_camera_settings."""
        mock = MagicMock(return_value=[0, 0, 3, 4])
        monkeypatch.setattr(
            f"{_MOD}._4DInterferometer.get_camera_settings", staticmethod(mock)
        )
        img = ma.masked_array(np.ones((1, 1)))
        full = _4DInterferometer.into_full_frame(img, tn="tnZ")
        mock.assert_called_once_with("tnZ")
        assert full.data[4, 3] == 1.0

    def test_neither_given_no_interferometer(self, monkeypatch):
        """No tn, no config_path and no live settings raises FileNotFoundError."""
        monkeypatch.setattr(f"{_MOD}._Folds.SETTINGS_CONF_FILE", None)
        with pytest.raises(FileNotFoundError):
            _4DInterferometer.into_full_frame(ma.masked_array(np.ones((2, 2))))

    def test_neither_given_uses_live_file(self, tmp_path, monkeypatch):
        """No tn/config_path: live settings file offsets are used."""
        ini = _write_ini(tmp_path / "live.ini", ox=2, oy=3)
        monkeypatch.setattr(f"{_MOD}._Folds.SETTINGS_CONF_FILE", ini)
        full = _4DInterferometer.into_full_frame(ma.masked_array(np.ones((1, 1))))
        assert full.data[3, 2] == 1.0

    def test_offsets_overflow_raises(self, tmp_path):
        """An image that does not fit at the offset raises ValueError."""
        ini = _write_ini(tmp_path / "c.ini", ox=2047, oy=0)
        with pytest.raises(ValueError):
            _4DInterferometer.into_full_frame(
                ma.masked_array(np.ones((2, 2))), config_path=ini
            )

    def test_missing_config_path_raises(self, tmp_path):
        """A non-existent config_path yields a configparser error."""
        import configparser

        with pytest.raises(configparser.Error):
            _4DInterferometer.into_full_frame(
                ma.masked_array(np.ones((2, 2))),
                config_path=str(tmp_path / "nope.ini"),
            )
