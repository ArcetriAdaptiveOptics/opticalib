from abc import ABC, abstractmethod
from opticalib.ground import logger as _logger
from opticalib.ground.osutils import newtn as _newtn
from opticalib.core.read_config import getInterfConfig
from opticalib.core.root import _updateInterfPaths, folders as _folds


class BaseWavefrontSensor(ABC):
    """
    Base class for all wavefront sensor devices.
    """

    @abstractmethod
    def acquire_map(self):
        """
        Abstract method to measure the interference pattern.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement acquire_map method")


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
        raise NotImplementedError("Subclasses must implement set_shape method")

    @abstractmethod
    def get_shape(self):
        """
        Abstract method to get the shape of the deformable mirror.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_shape method")

    @abstractmethod
    def uploadCmdHistory(self, tcmdhist):
        """
        Abstract method to upload the command history to the deformable mirror.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement uploadCmdHistory method")

    @abstractmethod
    def runCmdHistory(self, interf, differential, save):
        """
        Abstract method to run the command history on the deformable mirror.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement runCmdHistory method")

    def _slaveCmd(self, cmd, method: str):
        """ """
        from opticalib.dmutils.slaving import compute_slave_cmd

        if len(self.slaveIds) == 0:
            return cmd
        else:
            return compute_slave_cmd(self, cmd, method=method)
