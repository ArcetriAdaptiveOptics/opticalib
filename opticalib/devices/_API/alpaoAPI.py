import numpy as _np
from opticalib.core.read_config import getDmConfig
from opticalib.core.exceptions import CommandError
from opticalib import typings as _t


class BaseAlpaoMirror:
    """
    Base class for Alpao deformable mirrors using the Alpao SDK directly.

    Connects to the hardware via the Alpao SDK (``asdk`` module) and
    provides actuator-coordinate helpers, command-integrity checking,
    shape read/write, and configuration look-up.

    The Alpao SDK has no built-in position readback, so the last
    commanded vector is cached internally and returned by
    :meth:`get_shape`.

    Parameters
    ----------
    serial_number : str or None
        Hardware serial number of the DM (e.g. ``"BAXXX"``).  May be
        ``None`` when *nActs* is given and the serial number is stored
        in the configuration file.
    nActs : int, str or None
        Number of actuators.  Used to look up the DM configuration
        when *serial_number* is ``None``.

    Notes
    -----
    The ``asdk`` module is imported lazily inside :meth:`_init_sdk` so
    that the rest of the package can be used on systems where the
    Alpao SDK is not installed.
    """

    def __init__(
        self,
        serial_number: str | None,
        nActs: int | str | None,
    ) -> None:
        """
        Initialise the mirror, connecting to the SDK and loading the
        actuator layout.

        Parameters
        ----------
        serial_number : str or None
            Hardware serial number.  ``None`` if *nActs* is provided
            and the serial number will be read from the config file.
        nActs : int, str or None
            Number of actuators.  ``None`` if *serial_number* is
            provided directly.

        Raises
        ------
        ValueError
            If neither *serial_number* nor *nacts* is provided.
        ModuleNotFoundError
            If the ``asdk`` module (Alpao SDK) is not installed.
        Exception
            Any hardware-level exception raised by ``asdk.DM()`` on
            connection failure is propagated to the caller.
        """
        self._dmCoords = {
            "dm97": [5, 7, 9, 11],
            "dm192": [4, 8, 12, 12, 16, 16, 18],
            "dm277": [7, 9, 11, 13, 15, 17, 19],
            "dm468": [8, 12, 16, 18, 20, 20, 22, 22, 24],
            "dm820": [10, 14, 18, 20, 22, 24, 26, 28, 28, 30, 30, 32],
        }
        self._init_sdk(serial_number, nActs)
        self.nActs = int(self._sdk_dm.Get("NbOfActuator"))
        self._last_cmd: _t.ArrayLike = _np.zeros(self.nActs)
        self._name = f"Alpao{self.nActs}"
        self.actCoord = self._initActCoord()
        self.diameter = getDmConfig(self._name).get("diameter", None)
        self.mirrorModes = None
        self.cmdHistory = None
        self.refAct = None

    # ------------------------------------------------------------------
    # SDK-level interface
    # ------------------------------------------------------------------

    def get_shape(self) -> _t.ArrayLike:
        """
        Return the last commanded actuator positions.

        The Alpao SDK does not provide a hardware readback; the last
        vector sent via :meth:`set_shape` is returned instead.

        Returns
        -------
        numpy.ndarray
            Array of length ``nActs`` with the
            last commanded positions (zeros before the first command).
        """
        return self._last_cmd.copy()

    def set_shape(self, cmd: _t.ArrayLike) -> None:
        """
        Send an absolute command vector to the DM via the Alpao SDK.

        Parameters
        ----------
        cmd : array_like
            Command vector of length ``nActs``.
            Values must be in the range ``[-1, 1]`` (normalised units).

        Raises
        ------
        ValueError
            If the length of *cmd* does not match the number of
            actuators.
        """
        cmd = _np.asarray(cmd, dtype=float)
        if cmd.size != self.nActs:
            raise ValueError(
                f"Command length {cmd.size} does not match the number "
                f"of actuators ({self.nActs})."
            )
        self._sdk_dm.Send(cmd)
        self._last_cmd = cmd.copy()

    def get_version(self) -> int:
        """
        Return the SDK version reported by the firmware.

        Returns
        -------
        int
            Integer version code.
        """
        return int(self._sdk_dm.Get("VersionInfo"))

    def deinitialize(self) -> None:
        """
        Stop the DM and release hardware resources.

        Should be called when the DM object is no longer needed to
        ensure a clean shutdown of the Alpao SDK connection.
        Does nothing if the SDK handle was never successfully created.
        """
        if not hasattr(self, "_sdk_dm"):
            return
        self._sdk_dm.Stop()
        self._sdk_dm.Reset()

    # ------------------------------------------------------------------
    # Higher-level helpers
    # ------------------------------------------------------------------

    @property
    def nActuators(self) -> int:
        """Number of actuators on the DM."""
        return self.nActs

    def setReferenceActuator(self, refAct: int) -> None:
        """
        Set the reference actuator index for calibration purposes.

        Parameters
        ----------
        refAct : int
            Zero-based index of the reference actuator.

        Raises
        ------
        ValueError
            If *refAct* is outside the valid range ``[0, nActs)``.
        """
        if refAct < 0 or refAct >= self.nActs:
            raise ValueError(f"Reference actuator {refAct} is out of range.")
        self.refAct = refAct

    def _checkCmdIntegrity(self, cmd: _t.ArrayLike, amp_threshold: float = 0.9) -> None:
        """
        Validate a command vector before sending it to the hardware.

        Parameters
        ----------
        cmd : array_like
            Command vector to check.
        amp_threshold : float, optional
            Maximum allowed absolute value (default ``0.9``).

        Raises
        ------
        CommandError
            If any element exceeds *amp_threshold* or the standard
            deviation exceeds ``sqrt(amp_threshold) / 2``.
        """
        at = amp_threshold
        stdt = _np.sqrt(at) / 2
        mcmd = _np.max(cmd)
        if mcmd > at:
            raise CommandError(f"Command value {mcmd} is greater than {at:.2f}")
        mcmd = _np.min(cmd)
        if mcmd < -at:
            raise CommandError(f"Command value {mcmd} is smaller than {-at:.2f}")
        scmd = _np.std(cmd)
        if scmd > stdt:
            raise CommandError(
                f"Command standard deviation {scmd} is greater than {stdt:.2f}."
            )

    # ------------------------------------------------------------------
    # Private initialisation helpers
    # ------------------------------------------------------------------

    def _initActCoord(self) -> _t.ArrayLike:
        """
        Build the 2-D actuator coordinate array from the DM layout table.

        Returns
        -------
        numpy.ndarray or None
            Array of shape ``(2, nActs)`` with ``(x, y)`` pixel
            coordinates, or an empty array if the model is unknown.
        """
        try:
            nacts_row_sequence = self._dmCoords[f"dm{self.nActs}"]
        except KeyError:
            self.actCoord = _np.array([], dtype=int)
            return
        n_dim = nacts_row_sequence[-1]
        upper_rows = nacts_row_sequence[:-1]
        lower_rows = list(reversed(upper_rows))
        center_rows = [n_dim] * upper_rows[0]
        rows_number_of_acts = upper_rows + center_rows + lower_rows
        n_rows = len(rows_number_of_acts)
        cx = _np.array([], dtype=int)
        cy = _np.array([], dtype=int)
        for i in range(n_rows):
            cx = _np.concatenate(
                (
                    cx,
                    _np.arange(rows_number_of_acts[i])
                    + (n_dim - rows_number_of_acts[i]) // 2,
                )
            )
            cy = _np.concatenate((cy, _np.full(rows_number_of_acts[i], i)))
        self.actCoord = _np.array([cx, cy])
        return self.actCoord

    def _init_sdk(
        self,
        serial_number: str | None,
        nacts: int | str | None,
    ) -> None:
        """
        Connect to the Alpao SDK and store the raw DM handle.

        The serial number is resolved in the following priority order:

        1. The *serial_number* argument, if not ``None``.
        2. The ``serialNumber`` key in the device configuration file
           (looked up by *nacts*).

        Parameters
        ----------
        serial_number : str or None
            Hardware serial number supplied directly by the caller.
        nacts : int, str or None
            Number of actuators used to look up the configuration when
            *serial_number* is ``None``.

        Raises
        ------
        RuntimeError
            If neither *serial_number* nor *nacts* is provided.
        """
        if serial_number is None and nacts is not None:
            name = f"Alpao{int(nacts)}"
            config = getDmConfig(name)
            serial_number = config.get("serialNumber")
        elif (serial_number, nacts) == (None, None):
            raise RuntimeError("Either 'serial_number' or 'nacts' must be provided.")
        self.serial_number = serial_number

        try:
            from ...core.root import CONFIGURATION_FOLDER
            from os.path import join
            import sys

            sys.path.append(join(CONFIGURATION_FOLDER, "alpao_sdk"))
            from Lib64 import asdk  # type:ignore
        except Exception as e:
            if isinstance(e, ModuleNotFoundError):
                raise ModuleNotFoundError(
                    "The 'asdk' module (Alpao SDK) is not installed or "
                    "not found in the configuration folder."
                    f"\nEnsure to have put your {self.serial_number} Lib64 folder"
                    "in your `.../SysConfig/alpao_sdk/` folder."
                ) from e
            else:
                raise e

        self._sdk_dm = asdk.DM(serial_number)
        self._sdk_dm.Reset()
