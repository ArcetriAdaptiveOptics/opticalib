from abc import ABC, abstractmethod
from opticalib.core import _types as _ot
from opticalib.ground import logger as _logger
from opticalib.dmutils.slaving import compute_slave_cmd, compute_slaved_command_matrix


class BaseWavefrontSensor(ABC):
    """
    Base class for all wavefront sensor devices.
    """

    @abstractmethod
    def acquire_map(self, nframes):
        """
        Abstract method to measure the interference pattern.
        Must be implemented by subclasses.

        Parameters
        ----------
        nframes : int
            Number of frames to acquire.
        """
        raise NotImplementedError("Subclasses must implement acquire_map method")

    @abstractmethod
    def acquire_detector(self, nframes):
        """
        Abstract method to acquire frames from the detector.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement acquire_detector method")


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
    def upload_cmd_history(self, tcmdhist, *, slave: bool | str = False):
        """
        Abstract method to upload the command history to the deformable mirror.
        Must be implemented by subclasses.

        Parameters
        ----------
        tcmdhist : np.array
            Command history to be uploaded. Should be a 2D matrix of shape
            (nacts, nmodes).
        slave : bool | str, optional
            Slaving option for the input command history. If ``True``, the slaving method
            is chosen automatically; if a string is provided, it is used as method.
        """
        raise NotImplementedError("Subclasses must implement upload_cmd_history method")

    @abstractmethod
    def run_cmd_history(self, wfs, differential, save):
        """
        Abstract method to run the command history on the deformable mirror.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement run_cmd_history method")

    def _get_slaving_method(self, slave: bool | str) -> _ot.Optional[str]:
        """
        Resolve the slaving method from the ``slave`` argument.
        """
        if isinstance(slave, str):
            return slave

        if not slave:
            return None

        slave_ids = getattr(self, "slave_ids", [])
        border_ids = getattr(self, "border_ids", [])
        s, b = len(slave_ids), len(border_ids)
        if not (s or b):
            _logger.warning(
                "Slaving requested but no slave or border actuators defined. Defaulting to zero-force slaving."
            )
            return None
        return "minimum-rms" if (s and b) else "zero-force"

    def _apply_slaving(self, cmd: _ot.ArrayLike, slave: bool | str) -> _ot.ArrayLike:
        """
        Apply actuator slaving to a command when requested.
        """
        method = self._get_slaving_method(slave)
        if method is None:
            return cmd
        return compute_slave_cmd(self, cmd, method=method)

    def _slave_cmdmat(self, cmdmat: _ot.ArrayLike, slave: bool | str) -> _ot.MatrixLike:
        """
        Apply actuator slaving to a command matrix when requested.
        """
        method = self._get_slaving_method(slave)
        if method is None:
            return cmdmat
        return compute_slaved_command_matrix(self, cmdmat, method=method)


class BaseCamera(ABC):

    @abstractmethod
    def acquire_frames(self, nframes, *args):
        """Main function for acquiring frames from the camera"""
        raise NotImplementedError("Camera classes must implement this method!")

    @abstractmethod
    def reconnect(self, *args):
        """Function for reconnecting to the camera device in case of disconnection.
        Used also for the ``allow_reconnection`` decorator"""
        raise NotImplementedError("Camera classes must implement this method!")
