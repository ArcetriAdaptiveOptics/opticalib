"""
Tests for opticalib.core.config module.
"""

import pytest
import os
import tempfile
import shutil
import yaml
import numpy as np
from opticalib.core.exceptions import DeviceNotFoundError
from opticalib.core import config as read_config


class TestLoadYamlConfig:
    """Test load_yaml_config function."""

    def test_load_yaml_config_default(self, temp_dir, monkeypatch):
        """Test loading default configuration."""
        # Create a config file in temp_dir
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "SYSTEM": {"data_path": ""},
            "DEVICES": {"CAMERAS": {"TestCam": {"id": "test"}}},
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Mock the configuration folder
        monkeypatch.setattr(read_config, "_cfold", temp_dir)
        # Also need to update the module-level variable
        import opticalib.core.config as rc_module

        monkeypatch.setattr(rc_module, "_cfold", temp_dir)
        config = read_config.load_yaml_config(path=temp_dir)
        assert "SYSTEM" in config
        assert "DEVICES" in config

    def test_load_yaml_config_custom_path(self, temp_dir):
        """Test loading configuration from custom path."""
        config_file = os.path.join(temp_dir, "custom_config.yaml")
        config_data = {"TEST": {"key": "value"}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = read_config.load_yaml_config(path=config_file)
        assert "TEST" in config
        assert config["TEST"]["key"] == "value"


class TestDumpYamlConfig:
    """Test dump_yaml_config function."""

    def test_dump_yaml_config(self, temp_dir, monkeypatch):
        """Test dumping configuration to file."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        monkeypatch.setattr(read_config, "_cfold", temp_dir)

        config_data = {"TEST": {"key": "value"}}
        read_config.dump_yaml_config(config_data, path=temp_dir)

        assert os.path.exists(config_file)
        with open(config_file, "r") as f:
            loaded = yaml.safe_load(f)
        assert loaded["TEST"]["key"] == "value"


class TestLoadDump:
    """Test the refactored load/dump functions."""

    def test_load_from_iff_file(self, temp_dir):
        """Test loading iffConfig from an explicit file path."""
        iff_file = os.path.join(temp_dir, "iffConfig.yaml")
        with open(iff_file, "w") as f:
            yaml.safe_dump({"TEST": {"value": 1}}, f)

        config = read_config.load(path=iff_file)
        assert config["TEST"]["value"] == 1

    def test_dump_to_explicit_file(self, temp_dir):
        """Test dumping config to an explicit file path."""
        outfile = os.path.join(temp_dir, "custom.yaml")
        data = {"A": {"B": 2}}
        read_config.dump(data, outfile)

        with open(outfile, "r") as f:
            loaded = yaml.safe_load(f)
        assert loaded == data


class TestGetSectionConfig:
    """Test get_section_config function."""

    def test_get_section_config_section_and_subsection(self, temp_dir, monkeypatch):
        """Test reading both section and subsection values."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "PHASING": {"expected_psfs": 6},
            "DEVICES": {"CAMERAS": {"TestCam": {"id": "ID"}}},
        }
        with open(config_file, "w") as f:
            yaml.safe_dump(config_data, f)

        monkeypatch.setattr(read_config, "_cfile", config_file)
        assert read_config.get_section_config("PHASING")["expected_psfs"] == 6
        assert (
            read_config.get_section_config("DEVICES", "CAMERAS")["TestCam"]["id"]
            == "ID"
        )

    def test_get_section_config_missing_section(self, temp_dir, monkeypatch):
        """Test missing section handling."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        with open(config_file, "w") as f:
            yaml.safe_dump({"A": {}}, f)

        monkeypatch.setattr(read_config, "_cfile", config_file)
        with pytest.raises(KeyError):
            read_config.get_section_config("MISSING")


class TestGetIffConfig:
    """Test get_iff_config function."""

    def test_get_iff_config(self, temp_dir, monkeypatch):
        """Test getting IFF configuration."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "INFLUENCE.FUNCTIONS": {
                "IFFUNC": {
                    "trailing_zeros": 2,
                    "modes_list": [1, 2, 3],
                    "amplitude": [0.1, 0.2, 0.3],
                    "template": [[1, 2], [3, 4]],
                    "modal_base": "test_base",
                }
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setattr(read_config, "_cfold", temp_dir)
        config = read_config.get_iff_config("IFFUNC", bpath=temp_dir)

        assert config["trailing_zeros"] == 2
        assert isinstance(config["modes_list"], np.ndarray)
        assert isinstance(config["amplitude"], np.ndarray)
        assert isinstance(config["template"], np.ndarray)
        assert config["modal_base"] == "test_base"


class TestGetDmConfig:
    """Test get_dm_config function."""

    def test_get_dm_config_success(self, temp_dir, monkeypatch):
        """Test getting DM configuration successfully."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "DEVICES": {
                "DEFORMABLE.MIRRORS": {"TestDM": {"ip": "127.0.0.1", "port": 9090}}
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setattr(read_config, "_cfile", config_file)
        config = read_config.get_dm_config("TestDM")

        assert config["ip"] == "127.0.0.1"
        assert config["port"] == 9090

    def test_get_dm_config_not_found(self, temp_dir, monkeypatch):
        """Test getting DM configuration when device not found."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {"DEVICES": {"DEFORMABLE.MIRRORS": {}}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setattr(read_config, "_cfile", config_file)

        with pytest.raises(DeviceNotFoundError):
            read_config.get_dm_config("NonExistentDM")


class TestGetInterfConfig:
    """Test get_interf_config function."""
    def test_get_interf_config_success(self, temp_dir, monkeypatch):
        """Test getting interferometer configuration successfully."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "DEVICES": {
                "INTERFEROMETERS": {"TestInterf": {"ip": "127.0.0.1", "port": 8011}}
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setattr(read_config, "_cfile", config_file)
        config = read_config.get_interf_config("TestInterf")

        assert config["ip"] == "127.0.0.1"
        assert config["port"] == 8011

    def test_get_interf_config_not_found(self, temp_dir, monkeypatch):
        """Test getting interferometer configuration when device not found."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {"DEVICES": {"INTERFEROMETERS": {}}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setattr(read_config, "_cfile", config_file)

        with pytest.raises(DeviceNotFoundError):
            read_config.get_interf_config("NonExistentInterf")


class TestGetCamerasConfig:
    """Test get_cameras_config function."""

    def test_get_cameras_config_all(self, temp_dir, monkeypatch):
        """Test getting all cameras configuration."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "DEVICES": {"CAMERAS": {"Cam1": {"id": "cam1"}, "Cam2": {"id": "cam2"}}}
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setattr(read_config, "_cfile", config_file)
        config = read_config.get_cameras_config()

        assert "Cam1" in config
        assert "Cam2" in config

    def test_get_cameras_config_specific(self, temp_dir, monkeypatch):
        """Test getting specific camera configuration."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {"DEVICES": {"CAMERAS": {"TestCam": {"id": "test_cam"}}}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setattr(read_config, "_cfile", config_file)
        config = read_config.get_cameras_config("TestCam")

        assert config["id"] == "test_cam"

    def test_get_cameras_config_not_found(self, temp_dir, monkeypatch):
        """Test getting camera configuration when device not found."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {"DEVICES": {"CAMERAS": {}}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setattr(read_config, "_cfile", config_file)

        with pytest.raises(DeviceNotFoundError):
            read_config.get_cameras_config("NonExistentCam")


class TestGetNActs:
    """Test get_n_acts function."""

    def test_get_nacts(self, temp_dir, monkeypatch):
        """Test getting number of actuators."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {"INFLUENCE.FUNCTIONS": {"DM": {"nacts": 100}}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setattr(read_config, "_cfold", temp_dir)
        nacts = read_config.get_n_acts(bpath=temp_dir)

        assert nacts == 100
        assert isinstance(nacts, int)


class TestGetTiming:
    """Test get_timing function."""

    def test_get_timing(self, temp_dir, monkeypatch):
        """Test getting timing configuration."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {"INFLUENCE.FUNCTIONS": {"DM": {"timing": 10}}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setattr(read_config, "_cfold", temp_dir)
        timing = read_config.get_timing(bpath=temp_dir)

        assert timing == 10
        assert isinstance(timing, int)


class TestGetCmdDelay:
    """Test get_cmd_delay function."""

    def test_get_cmd_delay(self, temp_dir, monkeypatch):
        """Test getting command delay."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {"INFLUENCE.FUNCTIONS": {"DM": {"sequentialDelay": 0.1}}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setattr(read_config, "_cfold", temp_dir)
        cmd_delay = read_config.get_cmd_delay(bpath=temp_dir)

        assert cmd_delay == 0.1
        assert isinstance(cmd_delay, float)


class TestParseVal:
    """Test _parse_val function."""

    def test_parse_val_list(self):
        """Test parsing a list value."""
        val = [1, 2, 3]
        result = read_config._parse_val(val)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_parse_val_np_arange_string(self):
        """Test parsing np.arange string."""
        val = "np.arange(0, 10)"
        result = read_config._parse_val(val)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.arange(0, 10))

    def test_parse_val_float_string(self):
        """Test parsing float string."""
        val = "3.14"
        result = read_config._parse_val(val)
        assert result == 3.14

    def test_parse_val_int(self):
        """Test parsing integer."""
        val = 42
        result = read_config._parse_val(val)
        assert result == 42
        assert isinstance(result, int)

    def test_parse_val_float(self):
        """Test parsing float."""
        val = 3.14
        result = read_config._parse_val(val)
        assert result == 3.14
        assert isinstance(result, float)


class TestGetAlignmentConfig:
    """Test get_alignment_config function."""

    def test_get_alignment_config(self, temp_dir, monkeypatch):
        """Test getting alignment configuration."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "SYSTEM.ALIGNMENT": {
                "slices": [{"start": 0, "stop": 100}, {"start": 100, "stop": 200}]
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setattr(read_config, "_cfile", config_file)
        config = read_config.get_alignment_config()

        assert hasattr(config, "slices")
        assert len(config.slices) == 2
        assert isinstance(config.slices[0], slice)


class TestGetStitchingConfig:
    """Test get_stitching_config function."""

    def test_get_stitching_config(self, temp_dir, monkeypatch):
        """Test getting stitching configuration."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {"STITCHING": {"overlap": 0.1, "method": "test_method"}}
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        monkeypatch.setattr(read_config, "_cfile", config_file)
        config = read_config.get_stitching_config()

        assert config["overlap"] == 0.1
        assert config["method"] == "test_method"
