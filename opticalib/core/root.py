"""
this module is at the heart of the package, as it defines its folder structure
and the configuration file reader and writer. Also, it is fundamental for the
`calpy` custom entry point functionalities.
"""

import os as _os
import configparser as _cp
from shutil import copy as _copy
from ruamel.yaml import YAML as _YAML

_gyml = _YAML()
_gyml.preserve_quotes = True

global TEMPLATE_CONF_FILE

global SETTINGS_CONF_FILE
global COPIED_SETTINGS_CONF_FILE
global CAPTURE_FOLDER_NAME_4D_PC
global PRODUCE_FOLDER_NAME_4D_PC
global PRODUCE_FOLDER_NAME_LOCAL_PC


def _create_folder(path: str) -> None:
    if not _os.path.exists(path):
        _os.makedirs(path)


def create_folder_tree(BASE_DATA_PATH: str) -> None:
    """
    Create the folder tree for the package.
    """
    OPT_DATA_ROOT_FOLDER = _os.path.join(BASE_DATA_PATH, "OPTData")
    LOGGING_ROOT_FOLDER = _os.path.join(BASE_DATA_PATH, "Logging")
    CONFIGURATION_FOLDER = _os.path.join(BASE_DATA_PATH, "SysConfig")
    FLAT_ROOT_FOLDER = _os.path.join(OPT_DATA_ROOT_FOLDER, "Flattening")
    INTMAT_ROOT_FOLDER = _os.path.join(OPT_DATA_ROOT_FOLDER, "INTMatrices")
    MODALBASE_ROOT_FOLDER = _os.path.join(OPT_DATA_ROOT_FOLDER, "ModalBases")
    OPD_SERIES_ROOT_FOLDER = _os.path.join(OPT_DATA_ROOT_FOLDER, "OPDSeries")
    OPD_IMAGES_ROOT_FOLDER = _os.path.join(OPT_DATA_ROOT_FOLDER, "OPDImages")
    IFFUNCTIONS_ROOT_FOLDER = _os.path.join(OPT_DATA_ROOT_FOLDER, "IFFunctions")
    ALIGNMENT_ROOT_FOLDER = _os.path.join(OPT_DATA_ROOT_FOLDER, "Alignment")
    CONTROL_MATRIX_FOLDER = _os.path.join(ALIGNMENT_ROOT_FOLDER, "ControlMatrices")
    ALIGN_CALIBRATION_ROOT_FOLDER: str = _os.path.join(
        ALIGNMENT_ROOT_FOLDER, "Calibration"
    )
    for p in [
        BASE_DATA_PATH,
        OPT_DATA_ROOT_FOLDER,
        LOGGING_ROOT_FOLDER,
        CONFIGURATION_FOLDER,
        FLAT_ROOT_FOLDER,
        INTMAT_ROOT_FOLDER,
        MODALBASE_ROOT_FOLDER,
        OPD_SERIES_ROOT_FOLDER,
        OPD_IMAGES_ROOT_FOLDER,
        IFFUNCTIONS_ROOT_FOLDER,
        ALIGNMENT_ROOT_FOLDER,
        CONTROL_MATRIX_FOLDER,
        ALIGN_CALIBRATION_ROOT_FOLDER,
    ]:
        _create_folder(p)


def _updateInterfPaths(paths: dict[str, str]) -> None:
    """
    Update the path of the configuration file and the folders.

    This function reads the configuration file and updates the paths of the
    settings file and the folders used in the package.
    """
    global SETTINGS_CONF_FILE
    global COPIED_SETTINGS_CONF_FILE
    global CAPTURE_FOLDER_NAME_4D_PC
    global PRODUCE_FOLDER_NAME_4D_PC
    global PRODUCE_FOLDER_NAME_LOCAL_PC
    SETTINGS_CONF_FILE = paths["settings"]
    COPIED_SETTINGS_CONF_FILE = paths["copied_settings"]
    CAPTURE_FOLDER_NAME_4D_PC = paths["capture_4dpc"]
    PRODUCE_FOLDER_NAME_4D_PC = paths["produce_4dpc"]
    PRODUCE_FOLDER_NAME_LOCAL_PC = paths["produce"]


def create_configuration_file(path: str = "", data_path: str | bool = False) -> None:
    """
    Create a configuration file in the specified path.

    Parameters
    ----------
    path : str
        The path to the configuration file.
    data_path : str | bool
        The path to the data folder. If True, it will be set to the same
        directory as the configuration file. If False, it will not be set.
        If a string, a path must be provided, and the `data_path` will be
        set to that path.
    """
    global TEMPLATE_CONF_FILE
    bp = _os.path.expanduser("~")
    if not bp in path:
        if "mnt" in path or "media" in path:
            pass
        else:
            path = _os.path.join(bp, path)
    if not ".yaml" in path:
        file = _os.path.join(path, "configuration.yaml")
        _create_folder(path)
        if not _os.path.isdir(path):
            raise OSError(f"Invalid Path: {path}")
    else:
        file = path
        _create_folder(_os.path.dirname(path))
    _copy(TEMPLATE_CONF_FILE, file)
    print(f"Configuration file created in {path}")
    if data_path:
        data_path = _os.path.dirname(file)
        with open(file, "r") as _f:
            config = _gyml.load(_f)
        config["SYSTEM"]["data_path"] = data_path
        with open(file, "w") as _f:
            _gyml.dump(config, _f)
        with open(file, "r") as _f:
            _config = _gyml.load(_f)
        create_folder_tree(data_path)
        from shutil import move

        conf_folder = _os.path.join(data_path, "SysConfig")
        move(file, _os.path.join(conf_folder, "configuration.yaml"))
        print(f"Configuration file moved to {conf_folder}")


TEMPLATE_CONF_FILE: str = (
    _os.path.dirname(_os.path.abspath(__file__)) + "/_configurations/configuration.yaml"
)
CONFIGURATION_ROOT_FOLDER: str = _os.path.dirname(TEMPLATE_CONF_FILE)
CONFIGURATION_FILE: str = _os.getenv("AOCONF", TEMPLATE_CONF_FILE)

with open(CONFIGURATION_FILE, "r") as _f:
    _config = _gyml.load(_f)

_bdp = _config["SYSTEM"].get("data_path")
_fallback_bdp = _os.path.join(_os.path.expanduser("~"), ".tmp_opticalib")
BASE_DATA_PATH: str = _bdp if not _bdp == "" else _fallback_bdp

create_folder_tree(BASE_DATA_PATH)
if BASE_DATA_PATH == _fallback_bdp:
    CONFIGURATION_FILE = _os.path.join(
        BASE_DATA_PATH, "SysConfig", "configuration.yaml"
    )
    if not _os.path.exists(CONFIGURATION_FILE):
        _copy(TEMPLATE_CONF_FILE, CONFIGURATION_FILE)
    with open(CONFIGURATION_FILE, "r") as _f:
        _config = _gyml.load(_f)

OPT_DATA_ROOT_FOLDER: str = _os.path.join(BASE_DATA_PATH, "OPTData")
LOGGING_ROOT_FOLDER: str = _os.path.join(BASE_DATA_PATH, "Logging")
CONFIGURATION_FOLDER: str = _os.path.join(BASE_DATA_PATH, "SysConfig")
OPD_SERIES_ROOT_FOLDER: str = _os.path.join(OPT_DATA_ROOT_FOLDER, "OPDSeries")
OPD_IMAGES_ROOT_FOLDER: str = _os.path.join(OPT_DATA_ROOT_FOLDER, "OPDImages")
ALIGNMENT_ROOT_FOLDER: str = _os.path.join(OPT_DATA_ROOT_FOLDER, "Alignment")
FLAT_ROOT_FOLDER: str = _os.path.join(OPT_DATA_ROOT_FOLDER, "Flattening")
MODALBASE_ROOT_FOLDER: str = _os.path.join(OPT_DATA_ROOT_FOLDER, "ModalBases")
IFFUNCTIONS_ROOT_FOLDER: str = _os.path.join(OPT_DATA_ROOT_FOLDER, "IFFunctions")
INTMAT_ROOT_FOLDER: str = _os.path.join(OPT_DATA_ROOT_FOLDER, "INTMatrices")
CONTROL_MATRIX_FOLDER: str = _os.path.join(ALIGNMENT_ROOT_FOLDER, "ControlMatrices")
ALIGN_CALIBRATION_ROOT_FOLDER: str = _os.path.join(ALIGNMENT_ROOT_FOLDER, "Calibration")
ALIGN_RESULTS_ROOT_FOLDER: str = _os.path.join(ALIGNMENT_ROOT_FOLDER, "Results")

########################
# INTERFEROMETER PATHS #
########################
SETTINGS_CONF_FILE: str = None
COPIED_SETTINGS_CONF_FILE: str = None
CAPTURE_FOLDER_NAME_4D_PC: str = None
PRODUCE_FOLDER_NAME_4D_PC: str = None
PRODUCE_FOLDER_NAME_LOCAL_PC: str = None


###############################################################################
# CLASSES DEFINITIONS: THE FOLDER TREE WRAPPER AND THE 4D CONFIGURATION READER
###############################################################################
class _folds:
    """Wrapper class for the folder tree of the package"""

    def __init__(self):
        self.BASE_DATA_PATH = BASE_DATA_PATH
        self.CONFIGURATION_FILE = CONFIGURATION_FILE
        self.OPT_DATA_ROOT_FOLDER = OPT_DATA_ROOT_FOLDER
        self.LOGGING_ROOT_FOLDER = LOGGING_ROOT_FOLDER
        self.CONFIGURATION_FOLDER = CONFIGURATION_FOLDER
        self.OPD_SERIES_ROOT_FOLDER = OPD_SERIES_ROOT_FOLDER
        self.OPD_IMAGES_ROOT_FOLDER = OPD_IMAGES_ROOT_FOLDER
        self.ALIGNMENT_ROOT_FOLDER = ALIGNMENT_ROOT_FOLDER
        self.FLAT_ROOT_FOLDER = FLAT_ROOT_FOLDER
        self.MODALBASE_ROOT_FOLDER = MODALBASE_ROOT_FOLDER
        self.IFFUNCTIONS_ROOT_FOLDER = IFFUNCTIONS_ROOT_FOLDER
        self.INTMAT_ROOT_FOLDER = INTMAT_ROOT_FOLDER
        self.CONTROL_MATRIX_FOLDER = CONTROL_MATRIX_FOLDER
        self.ALIGN_CALIBRATION_ROOT_FOLDER = ALIGN_CALIBRATION_ROOT_FOLDER
        self.ALIGN_RESULTS_ROOT_FOLDER = ALIGN_RESULTS_ROOT_FOLDER

        self.SETTINGS_CONF_FILE = SETTINGS_CONF_FILE
        self.COPIED_SETTINGS_CONF_FILE = COPIED_SETTINGS_CONF_FILE
        self.CAPTURE_FOLDER_NAME_4D_PC = CAPTURE_FOLDER_NAME_4D_PC
        self.PRODUCE_FOLDER_NAME_4D_PC = PRODUCE_FOLDER_NAME_4D_PC
        self.PRODUCE_FOLDER_NAME_LOCAL_PC = PRODUCE_FOLDER_NAME_LOCAL_PC

    @property
    def print_all(self):
        """Print all the folders"""
        for key, value in self.__dict__.items():
            print(f"{key:<30} = {value}")

    def _update_interf_paths(self):
        """
        Update the paths of the configuration file and the folders.

        This function reads the configuration file and updates the paths of the
        settings file and the folders used in the package.
        """
        global SETTINGS_CONF_FILE
        global COPIED_SETTINGS_CONF_FILE
        global CAPTURE_FOLDER_NAME_4D_PC
        global PRODUCE_FOLDER_NAME_4D_PC
        global PRODUCE_FOLDER_NAME_LOCAL_PC
        self.SETTINGS_CONF_FILE = SETTINGS_CONF_FILE
        self.COPIED_SETTINGS_CONF_FILE = COPIED_SETTINGS_CONF_FILE
        self.CAPTURE_FOLDER_NAME_4D_PC = CAPTURE_FOLDER_NAME_4D_PC
        self.PRODUCE_FOLDER_NAME_4D_PC = PRODUCE_FOLDER_NAME_4D_PC
        self.PRODUCE_FOLDER_NAME_LOCAL_PC = PRODUCE_FOLDER_NAME_LOCAL_PC


folders = _folds()


class ConfSettingReader4D:
    """
    Class which reads an interferometer configuration settings file '4DSettings.ini'

    Methods
    -------
    getFrameRate() :
        Gets the camera frame rate in Hz.

    getImageWidthInPixels() :
        Get the width of the frame in pixel units.

    getImageHeightInPixels() :
        Get the height of the frame in pixel units.

    getOffsetX() :
        Get the frame offset in x-axis.

    getOffsetY() :
        Get the frame offset in y-axis.

    getPixelFormat() :
        Get the format of the pixels.

    getUserSettingFilePath() :
        Get the path of the configuration file.

    How to Use it
    -------------
    After initializing the class with a file path, just call methods on the defined
    object

    >>> cr = ConfSettingReader(file_path)
    >>> cr.getImageWidhtInPixels()
    2000
    >>> cr.getImageHeightInPixels()
    2000

    Notes
    -----
    Note that there is no need to directly use this module, as the settings information
    retrievement is handled by m4.urils.osutils, with its functions
    ''getConf4DSettingsPath'' and ''getCameraSettings''.
    """

    def __init__(self, file_path):
        self.config = _cp.ConfigParser()
        self.config.read(file_path)
        self.camera_section = "ACA2440"
        self.path_section = "Paths"

    # CAMERA
    def getFrameRate(self):
        """
        Returns the acquisition frame rate of the interferometer in Hz

        Returns
        -------
        frame_rate : float
            The frame rate.
        """
        frame_rate = self.config.get(self.camera_section, "FrameRate")
        return float(frame_rate)

    def getImageWidhtInPixels(self):
        """
        Returns the image widht in pixel scale

        Returns
        -------
        image_wight_in_pixels : int
            Image pixel width.
        """
        image_width_in_pixels = self.config.get(
            self.camera_section, "ImageWidthInPixels"
        )
        return int(image_width_in_pixels)

    def getImageHeightInPixels(self):
        """
        Returns the image height in pixel scale

        Returns
        -------
        image_height_in_pixels : int
            Image pixel height.
        """
        image_height_in_pixels = self.config.get(
            self.camera_section, "ImageHeightInPixels"
        )
        return int(image_height_in_pixels)

    def getOffsetX(self):
        """
        Returns the camera offset, in pixels, along the x-axis.

        Returns
        -------
        offset_x : int
            Pixel offset in the x-axis.
        """
        offset_x = self.config.get(self.camera_section, "OffsetX")
        return int(offset_x)

    def getOffsetY(self):
        """
        Returns the camera offset, in pixels, along the y-axis.

        Returns
        -------
        offset_y : int
            Pixel offset in the y-axis.
        """
        offset_y = self.config.get(self.camera_section, "OffsetY")
        return int(offset_y)

    def getPixelFormat(self):
        """
        Returns the format of the pixel.

        Returns
        -------
        pixel_format : str
            Pixel format.
        """
        pixel_format = self.config.get(self.camera_section, "PixelFormat")
        return pixel_format

    # PATH
    def getUserSettingFilePath(self):
        """
        Returns the complete filepath of the settings configuration file.

        Returns
        -------
        user_setting_file_path : str
            Settings file path.
        """
        user_setting_file_path = self.config.get(
            self.path_section, "UserSettingsFilePath"
        )
        return user_setting_file_path
