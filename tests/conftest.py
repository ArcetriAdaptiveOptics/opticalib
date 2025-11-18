"""
Pytest configuration and shared fixtures for opticalib tests.
"""
import pytest
import tempfile
import shutil
import os
import numpy as np
import numpy.ma as ma
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_config_file(temp_dir):
    """Create a temporary configuration file."""
    config_path = os.path.join(temp_dir, "configuration.yaml")
    # Create a minimal config file
    config_content = """SYSTEM:
  data_path: ''
DEVICES:
  INTERFEROMETER:
    TestInterf:
      ip: 127.0.0.1
      port: 8011
  DEFORMABLE.MIRRORS:
    TestDM:
      ip: 127.0.0.1
      port: 9090
  CAMERAS:
    TestCam:
      id: 'TEST_CAM'
INFLUENCE.FUNCTIONS:
  DM:
    nacts: 100
    timing: 10
    sequentialDelay: 0.1
  IFFUNC:
    numberofzeros: 0
    modeid: [1, 2, 3]
    modeamp: [0.1, 0.2, 0.3]
    template: [[1, 2], [3, 4]]
    modalbase: 'test_base'
SYSTEM.ALIGNMENT:
  slices: []
STITCHING:
  overlap: 0.1
"""
    with open(config_path, "w") as f:
        f.write(config_content)
    return config_path


@pytest.fixture
def sample_image():
    """Create a sample masked image for testing."""
    from skimage.draw import disk
    data = np.random.randn(100, 100).astype(np.float32)
    mask = np.ones((100, 100), dtype=bool)
    rr, cc = disk((50, 50), 30)
    mask[rr, cc] = False
    return ma.masked_array(data, mask=mask)


@pytest.fixture
def sample_cube():
    """Create a sample cube for testing."""
    from skimage.draw import disk
    data = np.random.randn(100, 100, 10).astype(np.float32)
    mask = np.ones((100,100), dtype=bool)
    rr, cc = disk((50, 50), 30)
    mask[rr, cc] = False
    # Apply same mask to all frames
    cube = ma.masked_array(
        data,
        mask=np.broadcast_to(mask[..., np.newaxis], data.shape)
    )
    return cube


@pytest.fixture
def circular_mask():
    """Create a circular mask for testing."""
    size = 100
    y, x = np.ogrid[-size/2:size/2, -size/2:size/2]
    radius = size / 2 - 5
    mask = (x**2 + y**2) > radius**2
    return mask.astype(bool)


@pytest.fixture
def tracking_number():
    """Generate a valid tracking number."""
    from opticalib.ground.osutils import newtn
    return newtn()


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch, temp_dir):
    """Reset environment variables for testing."""
    # Set a test data path
    test_data_path = os.path.join(temp_dir, "test_data")
    os.makedirs(test_data_path, exist_ok=True)
    monkeypatch.setenv("AOCONF", "")
    # This will be handled by individual test modules that need to mock the config


# Fixtures for dmutils tests
from unittest.mock import MagicMock, Mock


@pytest.fixture
def mock_dm():
    """Create a mock deformable mirror device."""
    dm = MagicMock()
    dm.nActs = 100
    dm.mirrorModes = np.random.randn(100, 100).astype(np.float32)
    dm.name = "TestDM"
    dm.uploadCmdHistory = Mock()
    dm.runCmdHistory = Mock()
    dm.get_shape = Mock(return_value=np.zeros(100))
    dm.set_shape = Mock()
    return dm


@pytest.fixture
def mock_interferometer():
    """Create a mock interferometer device."""
    interf = MagicMock()
    interf.name = "TestInterf"
    interf.acquire_map = Mock(return_value=ma.masked_array(
        np.random.randn(200, 200).astype(np.float32),
        mask=np.zeros((200, 200), dtype=bool)
    ))
    interf.capture = Mock(return_value="test_tn")
    return interf


@pytest.fixture
def sample_int_cube():
    """Create a sample interaction cube for dmutils tests."""
    from skimage.draw import disk
    data = np.random.randn(100, 100, 10).astype(np.float32)
    mask = np.ones((100,100), dtype=bool)
    rr, cc = disk((50, 50), 30)
    mask[rr, cc] = False
    cube = ma.masked_array(
        data,
        mask=np.broadcast_to(mask[..., np.newaxis], data.shape)
    )
    return cube


@pytest.fixture
def sample_cmd_matrix():
    """Create a sample command matrix."""
    return np.random.randn(100, 10).astype(np.float32)


@pytest.fixture
def sample_iff_folder_structure(temp_dir, monkeypatch):
    """Create a sample IFF folder structure with required files."""
    from opticalib.core.root import folders
    from opticalib.dmutils import iff_processing as ifp
    from opticalib.ground import osutils
    
    # Patch _OPTDATA to point to temp_dir
    optdata = os.path.join(temp_dir, "OPTData")
    os.makedirs(optdata, exist_ok=True)
    monkeypatch.setattr(osutils, "_OPTDATA", optdata)
    
    # Create IFF folder
    iff_folder = os.path.join(optdata, "IFFunctions")
    os.makedirs(iff_folder, exist_ok=True)
    monkeypatch.setattr(folders, "IFFUNCTIONS_ROOT_FOLDER", iff_folder)
    # Also patch the module-level variable
    monkeypatch.setattr(ifp, "_ifFold", iff_folder)
    
    # Create tracking number folder
    tn = "20240101_120000"
    tn_folder = os.path.join(iff_folder, tn)
    os.makedirs(tn_folder, exist_ok=True)
    
    # Create required files
    from opticalib.ground import osutils
    
    # Create dummy files
    files_to_create = {
        "ampVector.fits": np.array([0.1, 0.2, 0.3]),
        "modesVector.fits": np.array([1, 2, 3]),
        "template.fits": np.array([1, -1, 1, -1]),
        "indexList.fits": np.array([0, 1, 2]),
        "regActs.fits": np.array([1, 2]),
        "cmdMatrix.fits": np.random.randn(100, 3).astype(np.float32),
        "shuffle.dat": "0"
    }
    
    for fname, data in files_to_create.items():
        if fname.endswith('.fits'):
            osutils.save_fits(
                os.path.join(tn_folder, fname),
                data,
                overwrite=True
            )
        else:
            with open(os.path.join(tn_folder, fname), "w") as f:
                f.write(str(data))
    
    # Create mode_ files for saveCube tests
    for i in range(3):
        mode_data = np.random.randn(50, 50).astype(np.float32)
        mode_mask = np.zeros((50, 50), dtype=bool)
        mode_img = ma.masked_array(mode_data, mask=mode_mask)
        osutils.save_fits(
            os.path.join(tn_folder, f"mode_{i:04d}.fits"),
            mode_img,
            overwrite=True
        )
    
    return tn, tn_folder


@pytest.fixture
def sample_int_matrix_folder(temp_dir, monkeypatch, sample_int_cube):
    """Create a sample INTMatrices folder structure."""
    from opticalib.core.root import folders
    from opticalib.ground import osutils
    from opticalib.dmutils import iff_processing as ifp
    from opticalib.dmutils import flattening as flt
    
    int_folder = os.path.join(temp_dir, "INTMatrices")
    os.makedirs(int_folder, exist_ok=True)
    monkeypatch.setattr(folders, "INTMAT_ROOT_FOLDER", int_folder)
    # Also patch the module-level variables
    monkeypatch.setattr(ifp, "_intMatFold", int_folder)
    monkeypatch.setattr(flt._ifp, "_intMatFold", int_folder)
    
    tn = "20240101_120000"
    tn_folder = os.path.join(int_folder, tn)
    os.makedirs(tn_folder, exist_ok=True)
    
    # Create cube
    cube = sample_int_cube
    osutils.save_fits(
        os.path.join(tn_folder, "IMCube.fits"),
        cube,
        overwrite=True,
        header={"REBIN": 1, "FILTERED": False}
    )
    
    # Create command matrix
    cmd_mat = np.random.randn(100, 10).astype(np.float32)
    osutils.save_fits(
        os.path.join(tn_folder, "cmdMatrix.fits"),
        cmd_mat,
        overwrite=True
    )
    
    # Create modes vector
    modes_vec = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    osutils.save_fits(
        os.path.join(tn_folder, "modesVector.fits"),
        modes_vec,
        overwrite=True
    )
    
    return tn, tn_folder

