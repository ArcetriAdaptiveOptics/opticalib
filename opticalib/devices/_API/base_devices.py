import warnings as _warnings
from abc import ABC, ABCMeta, abstractmethod
from opticalib.core import _types as _ot
from opticalib.core import config as _rc
from opticalib.core.exceptions import DeviceAttributeError
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


class _DeformableMirrorMeta(ABCMeta):
    """
    Metaclass of :class:`BaseDeformableMirror`.

    After the *outermost* ``__init__`` of a concrete DM class has completed,
    it calls :meth:`BaseDeformableMirror._check_required_attributes`, so that
    a DM class which forgets to define one of the required base attributes
    fails loudly at construction time instead of deep inside a procedure.
    """

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance._check_required_attributes()
        return instance


class BaseDeformableMirror(ABC, metaclass=_DeformableMirrorMeta):
    """
    Base class for all deformable mirror devices, real and simulated.

    Besides the abstract methods, every concrete DM **must** define the
    attributes listed below by the end of its ``__init__`` (they may be
    plain instance attributes or properties). Missing attributes raise a
    :class:`~opticalib.core.exceptions.DeviceAttributeError` on construction.

    Attributes
    ----------
    _name : str
        Device name. For real devices it is also the key of the device block
        under ``DEVICES/DEFORMABLE.MIRRORS`` in the configuration file.
    n_acts : int
        Total number of actuators.
    is_segmented : bool
        Whether the DM is made of independently controlled segments.
    n_segments : int
        Number of segments (``1`` for monolithic DMs).
    n_acts_per_segment : int
        Number of actuators per segment (``n_acts`` for monolithic DMs).
    mirror_modes : MatrixLike or None
        Mirror modal base, of shape ``(n_acts, n_modes)``, or ``None`` if
        not available.
    act_coord : ArrayLike or None
        Actuator coordinates, of shape ``(n_acts, 2)`` with columns
        ``(x, y)``, or ``None`` if not available.
    ff : MatrixLike or None
        Feed-forward (stiffness) matrix, of shape ``(n_acts, n_acts)``, used
        by actuator slaving, or ``None`` if not available.
    _cmd_history : MatrixLike or None
        Last uploaded command history (read through :attr:`cmd_history`).
    _slave_ids, _border_ids : list[int]
        Slave and border actuator indices (read through :attr:`slave_ids`
        and :attr:`border_ids`). Empty lists if none are defined; see
        :meth:`_init_slaving_ids`.
    """

    #: Attributes every concrete DM must define by the end of ``__init__``.
    _REQUIRED_ATTRIBUTES: tuple[str, ...] = (
        "_name",
        "n_acts",
        "is_segmented",
        "n_segments",
        "n_acts_per_segment",
        "mirror_modes",
        "act_coord",
        "ff",
        "_cmd_history",
        "_slave_ids",
        "_border_ids",
    )

    _name: str
    n_acts: int
    is_segmented: bool
    n_segments: int
    n_acts_per_segment: int
    mirror_modes: _ot.Optional[_ot.MatrixLike]
    act_coord: _ot.Optional[_ot.ArrayLike]
    ff: _ot.Optional[_ot.MatrixLike]
    _cmd_history: _ot.Optional[_ot.MatrixLike]
    _slave_ids: list[int]
    _border_ids: list[int]

    @property
    def slave_ids(self) -> list[int]:
        """Indices of the slave actuators."""
        return self._slave_ids

    @property
    def border_ids(self) -> list[int]:
        """Indices of the border actuators."""
        return self._border_ids

    @property
    def n_actuators(self) -> int:
        """Number of actuators of the DM (alias of ``n_acts``)."""
        return self.n_acts

    @property
    def cmd_history(self) -> _ot.Optional[_ot.MatrixLike]:
        """
        Get the command history of the deformable mirror (DM).

        Returns
        -------
        cmd_history : _ot.Optional[_ot.MatrixLike]
            The command history of the DM, or None if no history has been uploaded.
        """
        return self._cmd_history

    @abstractmethod
    def set_shape(
        self,
        cmd: _ot.ArrayLike,
        differential: bool = False,
        *,
        slaving: bool | str = False,
    ) -> None:
        """
        Abstract method to set the shape of the deformable mirror.
        Must be implemented by subclasses.

        Parameters
        ----------
        cmd : ArrayLike
            Command vector of length ``n_acts``.
        differential : bool, optional
            If True, the command is applied on top of the current shape.
        slaving : bool | str, optional
            Slaving option for the input command. If ``True``, the slaving
            method is chosen automatically; if a string is provided, it is
            used as method.
        """
        raise NotImplementedError("Subclasses must implement set_shape method")

    @abstractmethod
    def get_shape(self) -> _ot.ArrayLike:
        """
        Abstract method to get the shape of the deformable mirror.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_shape method")

    @abstractmethod
    def upload_cmd_history(
        self, tcmdhist: _ot.MatrixLike, *, slaving: bool | str = False
    ) -> None:
        """
        Abstract method to upload the command history to the deformable mirror.
        Must be implemented by subclasses.

        Parameters
        ----------
        tcmdhist : MatrixLike
            Command history to be uploaded. Should be a 2D matrix of shape
            (n_acts, n_commands).
        slaving : bool | str, optional
            Slaving option for the input command history. If ``True``, the slaving method
            is chosen automatically; if a string is provided, it is used as method.
        """
        raise NotImplementedError("Subclasses must implement upload_cmd_history method")

    @abstractmethod
    def run_cmd_history(
        self,
        wfs: _ot.Optional[_ot.InterferometerDevice | _ot.WFSDevice] = None,
        save: _ot.Optional[str] = None,
        **setshape_kwargs: _ot.Any,
    ) -> _ot.Any:
        """
        Abstract method to run the command history on the deformable mirror.
        Must be implemented by subclasses.

        Parameters
        ----------
        wfs : InterferometerDevice | WFSDevice, optional
            Sensor used to acquire one frame per command.
        save : str, optional
            Tracking number of the folder where the frames are saved. A new
            one is generated if not given.
        **setshape_kwargs
            Options forwarded to the command application, at least
            ``differential`` (default ``True``) and ``slaving`` (default
            ``False``).

        Returns
        -------
        tn : str or None
            Tracking number of the acquired data, if any.
        """
        raise NotImplementedError("Subclasses must implement run_cmd_history method")

    def _init_slaving_ids(self, config_name: _ot.Optional[str] = None) -> None:
        """
        Set ``_slave_ids`` and ``_border_ids`` from the device configuration.

        Reads the ``slave_ids`` and ``border_ids`` keys of the
        ``DEVICES/DEFORMABLE.MIRRORS/<config_name>`` block; both default to
        empty lists if the block or the keys are missing.

        Parameters
        ----------
        config_name : str, optional
            Name of the configuration block. Defaults to ``self._name``.
        """
        name = config_name if config_name is not None else self._name
        try:
            dm_config = _rc.get_device_config("DEFORMABLE.MIRRORS", name) or {}
        except KeyError:  # includes DeviceNotFoundError
            dm_config = {}
        self._slave_ids = list(dm_config.get("slave_ids", None) or [])
        self._border_ids = list(dm_config.get("border_ids", None) or [])

    def _check_required_attributes(self) -> None:
        """
        Verify that all the attributes in ``_REQUIRED_ATTRIBUTES`` are defined.

        Called automatically after construction. Also warns if the
        segmentation attributes are not consistent with ``n_acts``.

        Raises
        ------
        DeviceAttributeError
            If one or more required attributes are missing.
        """
        missing = [a for a in self._REQUIRED_ATTRIBUTES if not hasattr(self, a)]
        if missing:
            raise DeviceAttributeError(
                f"`{self.__class__.__name__}` does not define the required "
                f"deformable mirror attribute(s): {', '.join(missing)}",
                attribute=missing[0],
                obj=self,
            )
        if self.n_segments * self.n_acts_per_segment != self.n_acts:
            self._warn(
                f"{self.__class__.__name__}: n_segments ({self.n_segments}) x "
                f"n_acts_per_segment ({self.n_acts_per_segment}) != n_acts "
                f"({self.n_acts})"
            )

    def _warn(self, message: str) -> None:
        """Log a warning via the device logger, or emit a Python warning."""
        logger = getattr(self, "_logger", None)
        if logger is not None:
            logger.warning(message)
        else:
            _warnings.warn(message, RuntimeWarning, stacklevel=3)

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
        if not s:
            self._warn(
                "Slaving requested but no slave actuators defined: "
                "the command is applied without slaving."
            )
            return None
        return "minimum-rms" if b else "zero-force"

    def _apply_slaving(
        self, cmd: _ot.ArrayLike | _ot.MatrixLike, slave: bool | str
    ) -> _ot.ArrayLike | _ot.MatrixLike:
        """
        Apply actuator slaving to a command (or command matrix) when requested.
        """
        method = self._get_slaving_method(slave)
        if method is None:
            return cmd
        if _ot.isinstance_(cmd, "MatrixLike"):
            return compute_slaved_command_matrix(self, cmdmat=cmd, method=method)
        return compute_slave_cmd(self, cmd, method=method)


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
