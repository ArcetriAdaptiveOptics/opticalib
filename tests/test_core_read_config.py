"""
Tests for opticalib.core.read_config module.
"""
import pytest
import os
import tempfile
import shutil
import yaml
import numpy as np
from opticalib.core.exceptions import DeviceNotFoundError
from opticalib.core import read_config


class TestLoadYamlConfig:
    """Test load_yaml_config function."""

    def test_load_yaml_config_default(self, temp_dir, monkeypatch):
        """Test loading default configuration."""
        # Create a config file in temp_dir
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "SYSTEM": {"data_path": ""},
            "DEVICES": {"CAMERAS": {"TestCam": {"id": "test"}}}
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        # Mock the configuration folder
        monkeypatch.setattr(read_config, "_cfold", temp_dir)
        # Also need to update the module-level variable
        import opticalib.core.read_config as rc_module
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


class TestGetIffConfig:
    """Test getIffConfig function."""

    def test_get_iff_config(self, temp_dir, monkeypatch):
        """Test getting IFF configuration."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "INFLUENCE.FUNCTIONS": {
                "IFFUNC": {
                    "numberofzeros": 2,
                    "modeid": [1, 2, 3],
                    "modeamp": [0.1, 0.2, 0.3],
                    "template": [[1, 2], [3, 4]],
                    "modalbase": "test_base"
                }
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        monkeypatch.setattr(read_config, "_cfold", temp_dir)
        config = read_config.getIffConfig("IFFUNC", bpath=temp_dir)
        
        assert config["zeros"] == 2
        assert isinstance(config["modes"], np.ndarray)
        assert isinstance(config["amplitude"], np.ndarray)
        assert isinstance(config["template"], np.ndarray)
        assert config["modalBase"] == "test_base"


class TestGetDmConfig:
    """Test getDmConfig function."""

    def test_get_dm_config_success(self, temp_dir, monkeypatch):
        """Test getting DM configuration successfully."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "DEVICES": {
                "DEFORMABLE.MIRRORS": {
                    "TestDM": {
                        "ip": "127.0.0.1",
                        "port": 9090
                    }
                }
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        monkeypatch.setattr(read_config, "_cfile", config_file)
        config = read_config.getDmConfig("TestDM")
        
        assert config["ip"] == "127.0.0.1"
        assert config["port"] == 9090

    def test_get_dm_config_not_found(self, temp_dir, monkeypatch):
        """Test getting DM configuration when device not found."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "DEVICES": {
                "DEFORMABLE.MIRRORS": {}
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        monkeypatch.setattr(read_config, "_cfile", config_file)
        
        with pytest.raises(DeviceNotFoundError):
            read_config.getDmConfig("NonExistentDM")


class TestGetInterfConfig:
    """Test getInterfConfig function."""

    def test_get_interf_config_success(self, temp_dir, monkeypatch):
        """Test getting interferometer configuration successfully."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "DEVICES": {
                "INTERFEROMETER": {
                    "TestInterf": {
                        "ip": "127.0.0.1",
                        "port": 8011
                    }
                }
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        monkeypatch.setattr(read_config, "_cfile", config_file)
        config = read_config.getInterfConfig("TestInterf")
        
        assert config["ip"] == "127.0.0.1"
        assert config["port"] == 8011

    def test_get_interf_config_not_found(self, temp_dir, monkeypatch):
        """Test getting interferometer configuration when device not found."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "DEVICES": {
                "INTERFEROMETER": {}
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        monkeypatch.setattr(read_config, "_cfile", config_file)
        
        with pytest.raises(DeviceNotFoundError):
            read_config.getInterfConfig("NonExistentInterf")


class TestGetCamerasConfig:
    """Test getCamerasConfig function."""

    def test_get_cameras_config_all(self, temp_dir, monkeypatch):
        """Test getting all cameras configuration."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "DEVICES": {
                "CAMERAS": {
                    "Cam1": {"id": "cam1"},
                    "Cam2": {"id": "cam2"}
                }
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        monkeypatch.setattr(read_config, "_cfile", config_file)
        config = read_config.getCamerasConfig()
        
        assert "Cam1" in config
        assert "Cam2" in config

    def test_get_cameras_config_specific(self, temp_dir, monkeypatch):
        """Test getting specific camera configuration."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "DEVICES": {
                "CAMERAS": {
                    "TestCam": {"id": "test_cam"}
                }
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        monkeypatch.setattr(read_config, "_cfile", config_file)
        config = read_config.getCamerasConfig("TestCam")
        
        assert config["id"] == "test_cam"

    def test_get_cameras_config_not_found(self, temp_dir, monkeypatch):
        """Test getting camera configuration when device not found."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "DEVICES": {
                "CAMERAS": {}
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        monkeypatch.setattr(read_config, "_cfile", config_file)
        
        with pytest.raises(DeviceNotFoundError):
            read_config.getCamerasConfig("NonExistentCam")


class TestGetNActs:
    """Test getNActs function."""

    def test_get_nacts(self, temp_dir, monkeypatch):
        """Test getting number of actuators."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "INFLUENCE.FUNCTIONS": {
                "DM": {
                    "nacts": 100
                }
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        monkeypatch.setattr(read_config, "_cfold", temp_dir)
        nacts = read_config.getNActs(bpath=temp_dir)
        
        assert nacts == 100
        assert isinstance(nacts, int)


class TestGetTiming:
    """Test getTiming function."""

    def test_get_timing(self, temp_dir, monkeypatch):
        """Test getting timing configuration."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "INFLUENCE.FUNCTIONS": {
                "DM": {
                    "timing": 10
                }
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        monkeypatch.setattr(read_config, "_cfold", temp_dir)
        timing = read_config.getTiming(bpath=temp_dir)
        
        assert timing == 10
        assert isinstance(timing, int)


class TestGetCmdDelay:
    """Test getCmdDelay function."""

    def test_get_cmd_delay(self, temp_dir, monkeypatch):
        """Test getting command delay."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "INFLUENCE.FUNCTIONS": {
                "DM": {
                    "sequentialDelay": 0.1
                }
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        monkeypatch.setattr(read_config, "_cfold", temp_dir)
        cmd_delay = read_config.getCmdDelay(bpath=temp_dir)
        
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
    """Test getAlignmentConfig function."""

    def test_get_alignment_config(self, temp_dir, monkeypatch):
        """Test getting alignment configuration."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "SYSTEM.ALIGNMENT": {
                "slices": [
                    {"start": 0, "stop": 100},
                    {"start": 100, "stop": 200}
                ]
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        monkeypatch.setattr(read_config, "_cfile", config_file)
        config = read_config.getAlignmentConfig()
        
        assert hasattr(config, "slices")
        assert len(config.slices) == 2
        assert isinstance(config.slices[0], slice)


class TestGetStitchingConfig:
    """Test getStitchingConfig function."""

    def test_get_stitching_config(self, temp_dir, monkeypatch):
        """Test getting stitching configuration."""
        config_file = os.path.join(temp_dir, "configuration.yaml")
        config_data = {
            "STITCHING": {
                "overlap": 0.1,
                "method": "test_method"
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        monkeypatch.setattr(read_config, "_cfile", config_file)
        config = read_config.getStitchingConfig()
        
        assert config["overlap"] == 0.1
        assert config["method"] == "test_method"

