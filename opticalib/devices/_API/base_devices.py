from abc import ABC, abstractmethod
from opticalib.ground import logger as _logger
from opticalib.ground.osutils import newtn as _newtn
from opticalib.core.read_config import getInterfConfig
from opticalib.core.root import _updateInterfPaths, folders as _folds


class BaseInterferometer(ABC):
    """
    Base class for all interferometer devices.
    """

    def __init__(self, name: str, ip: str, port: int):
        """
        Initializes the interferometer with a name, in order to retrieve
        all the information from the configuration file.
        """
        self._name = name
        if (ip and port) is None:
            config = getInterfConfig(name)
            ip = config["ip"]
            port = config["port"]
            _updateInterfPaths(config["Paths"])
        self.ip = ip
        self.port = port
        self._logger = _logger.set_up_logger(f"{self._name}.log", 20)
        self._logger.info(
            f"Interferometer {self._name} initialized on addess {self.ip}:{self.port}"
        )
        self._ts = _newtn
        _folds._update_interf_paths()

    @abstractmethod
    def acquire_map(self):
        """
        Abstract method to measure the interference pattern.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def intoFullFrame(self, img):
        """
        Abstract method to convert the interference pattern into a full frame image.
        Must be implemented by subclasses.

        Parameters
        ----------
        img: _ot.ImageData
            The image data to be converted.

        Returns
        -------
        _ot.ImageData
            The full frame image data.
        """
        pass

    def acquireFullFrame(self, **kwargs):
        """
        Wrapper for the consecutive execution of `acquire_mapo` and `intoFullFrame`.

        Parameters
        ----------
        **kwargs: dict
            Additional keyword arguments to be passed to `acquire_map`.

        Returns
        -------
        _ot.ImageData
            The full frame image data.
        """
        img = self.acquire_map(**kwargs)
        full_frame = self.intoFullFrame(img)
        return full_frame


class BaseDeformableMirror(ABC):
    """
    Base class for all deformable mirror devices.
    """

    @abstractmethod
    def set_shape(self, cmd):
        """
        Abstract method to set the shape of the deformable mirror.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_shape(self):
        """
        Abstract method to get the shape of the deformable mirror.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def uploadCmdHistory(self, tcmdhist):
        """
        Abstract method to upload the command history to the deformable mirror.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def runCmdHistory(self, interf, differential, save):
        """
        Abstract method to run the command history on the deformable mirror.
        Must be implemented by subclasses.
        """
        pass

    def _slaveCmd(self, cmd, method: str):
        """ """
        from opticalib.dmutils.utils import compute_slave_cmd

        if len(self.slaveIds) == 0:
            return cmd
        else:
            return compute_slave_cmd(self, cmd, method=method)
