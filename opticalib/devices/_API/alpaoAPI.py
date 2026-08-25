import numpy as _np
from opticalib.core.config import get_section_config
from opticalib.core.exceptions import CommandError
from opticalib.core import _types as _t


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
        ``None`` when *n_acts* is given and the serial number is stored
        in the configuration file.
    n_acts : int, str or None
        Number of actuators.  Used to look up the DM configuration
        when *serial_number* is ``None``.
    reset_on_connect : bool, optional
        Zero actuators after connecting (default ``False``).
    reset_on_close : bool, optional
        Zero actuators on ``deinitialize`` / teardown (default ``False``).

    Notes
    -----
    The ``asdk`` module is imported lazily inside :meth:`_init_sdk` so
    that the rest of the package can be used on systems where the
    Alpao SDK is not installed.

    By default the last commanded shape is not cleared on connect or
    exit.  Pass ``reset_on_connect=True`` / ``reset_on_close=True``
    (or ``deinitialize(reset=True)``) when a safe zero is required.

    Exit without reset relies on the Alpao SDK parameter
    ``ResetOnClose`` (default ``true`` in the SDK): opticalib sets it
    from ``reset_on_close`` so destroying the ``asdk.DM`` handle
    (normal exit, GC, or Ctrl-C) does not zero the electronics.

    There is no hardware readback: after reconnect without reset,
    :meth:`get_shape` starts at zeros until the next :meth:`set_shape`.
    """

    def __init__(
        self,
        serial_number: str | None,
        n_acts: int | str | None,
        use_plico: bool = False,
        *,
        reset_on_connect: bool = False,
        reset_on_close: bool = False,
    ) -> None:
        """
        Initialise the mirror, connecting to the SDK and loading the
        actuator layout.

        Parameters
        ----------
        serial_number : str or None
            Hardware serial number.  ``None`` if *n_acts* is provided
            and the serial number will be read from the config file.
        n_acts : int, str or None
            Number of actuators.  ``None`` if *serial_number* is
            provided directly.
        reset_on_connect : bool, optional
            If ``True``, call SDK ``Reset()`` after opening the
            connection (actuators to zero).  Default ``False`` so a
            previously held shape is not cleared on connect.
        reset_on_close : bool, optional
            If ``True``, ``deinitialize()`` / process teardown will
            ``Reset()`` the mirror.  Default ``False`` so the last
            commanded shape is left on the electronics when possible.

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
            "dm88": [6, 8, 10],
            "dm97": [5, 7, 9, 11],
            "dm192": [4, 8, 12, 12, 16, 16, 18],
            "dm277": [7, 9, 11, 13, 15, 17, 19],
            "dm292": [4, 8, 12, 14, 16, 16, 18, 18, 20],
            "dm468": [8, 12, 16, 18, 20, 20, 22, 22, 24],
            "dm820": [10, 14, 18, 20, 22, 24, 26, 28, 28, 30, 30, 32],
        }
        self._reset_on_close = bool(reset_on_close)
        self._init_sdk(
            serial_number, n_acts, reset_on_connect=reset_on_connect
        )
        self.n_acts = int(self._sdk_dm.Get("NbOfActuator"))
        self._last_cmd: _t.ArrayLike = _np.zeros(self.n_acts)
        self._name = f"Alpao{self.n_acts}"
        self.act_coord = self._init_act_coord()
        self.diameter = get_section_config("DEVICES", "DEFORMABLE.MIRRORS")[
            self._name
        ].get("diameter", None)
        self.mirrorModes = None
        self.cmdHistory = None
        self.refAct = None
        self.ff = None

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
            Array of length ``n_acts`` with the
            last commanded positions (zeros before the first command).
        """
        return self._last_cmd.copy()

    def set_shape(self, cmd: _t.ArrayLike) -> None:
        """
        Send an absolute command vector to the DM via the Alpao SDK.

        Parameters
        ----------
        cmd : array_like
            Command vector of length ``n_acts``.
            Values must be in the range ``[-1, 1]`` (normalised units).

        Raises
        ------
        CommandError
            If the length of *cmd* does not match the number of
            actuators.
        """
        cmd = _np.asarray(cmd, dtype=float)
        if cmd.size != self.n_acts:
            raise CommandError(
                f"Command length {cmd.size} does not match the number "
                f"of actuators ({self.n_acts})."
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

    def deinitialize(self, reset: bool | None = None) -> None:
        """
        Stop transfers and release the SDK handle.

        Parameters
        ----------
        reset : bool or None, optional
            If ``True``, call SDK ``Reset()`` (actuators to zero)
            before releasing.  If ``None`` (default), use the
            ``reset_on_close`` value from construction.

        Notes
        -----
        Shape retention on teardown (including Ctrl-C, which often
        skips this method) is controlled by SDK ``ResetOnClose``, set
        at connect from ``reset_on_close``.  An explicit ``Reset()``
        here is only sent when *reset* / ``reset_on_close`` is true.
        """
        if not hasattr(self, "_sdk_dm") or self._sdk_dm is None:
            return
        do_reset = self._reset_on_close if reset is None else bool(reset)
        try:
            # Keep SDK destructor behavior aligned with this call.
            self._sdk_dm.Set("ResetOnClose", int(do_reset))
        except Exception:
            pass
        try:
            self._sdk_dm.Stop()
        except Exception:
            pass
        if do_reset:
            try:
                self._sdk_dm.Reset()
            except Exception:
                pass
        # Drop the Python reference; SWIG ~DM / asdkRelease follows.
        # With ResetOnClose=0 the electronics keep the last command.
        self._sdk_dm = None

    # ------------------------------------------------------------------
    # Higher-level helpers
    # ------------------------------------------------------------------

    @property
    def n_actuators(self) -> int:
        """Number of actuators on the DM."""
        return self.n_acts

    def set_reference_actuator(self, refAct: int) -> None:
        """
        Set the reference actuator index for calibration purposes.

        Parameters
        ----------
        refAct : int
            Zero-based index of the reference actuator.

        Raises
        ------
        ValueError
            If *refAct* is outside the valid range ``[0, n_acts)``.
        """
        if refAct < 0 or refAct >= self.n_acts:
            raise ValueError(f"Reference actuator {refAct} is out of range.")
        self.refAct = refAct

    def _check_cmd_integrity(
        self, cmd: _t.ArrayLike, amp_threshold: float = 0.9
    ) -> None:
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

    def _init_act_coord(self) -> _t.ArrayLike:
        """
        Build the 2-D actuator coordinate array from the DM layout table.

        Returns
        -------
        numpy.ndarray or None
            Array of shape ``(2, n_acts)`` with ``(x, y)`` pixel
            coordinates, or an empty array if the model is unknown.
        """
        try:
            nacts_row_sequence = self._dmCoords[f"dm{self.n_acts}"]
        except KeyError:
            self.act_coord = _np.array([], dtype=int)
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
        self.act_coord = _np.array([cx, cy])
        return self.act_coord

    def _init_sdk(
        self,
        serial_number: str | None,
        nacts: int | str | None,
        *,
        reset_on_connect: bool = False,
    ) -> None:
        """
        Connect to the Alpao SDK and store the raw DM handle.

        The serial number is resolved in the following priority order:

        1. The *serial_number* argument, if not ``None``.
        2. The ``serialNumber`` key in the device configuration file
           (looked up by *nacts*).

        When *nacts* is provided the device configuration block
        ``Alpao{nacts}`` is read from the configuration file, allowing
        ``serialNumber``, ``sdk_folder_path``, and ``acfg_path`` to be
        set there instead of being passed as arguments.  When only
        *serial_number* is given the configuration lookup is skipped and
        the caller is responsible for having ``ACECFG`` set in the
        environment.

        Parameters
        ----------
        serial_number : str or None
            Hardware serial number supplied directly by the caller.
        nacts : int, str or None
            Number of actuators used to look up the configuration when
            *serial_number* is ``None``.
        reset_on_connect : bool, optional
            If ``True``, zero the mirror after connecting.

        Raises
        ------
        RuntimeError
            If neither *serial_number* nor *nacts* is provided.
        FileNotFoundError
            If the resolved SDK path does not exist on disk.
        ModuleNotFoundError
            If the ``asdk`` module cannot be imported from the SDK path.
        """
        import os
        import sys
        from ...core.root import CONFIGURATION_FOLDER

        if serial_number is None and nacts is None:
            raise RuntimeError("Either 'serial_number' or 'nacts' must be provided.")

        # Config lookup is only possible when nacts is known.
        config: dict = {}
        if nacts is not None:
            name = f"Alpao{int(nacts)}"
            config = get_section_config("DEVICES", "DEFORMABLE.MIRRORS")[name]
            serial_number = config.get("serialNumber", serial_number)

        self.serial_number = serial_number
        sdkp = config.get("sdk_folder_path", None)
        acfg = config.get("acfg_path", None)

        # Set the ACECFG environment variable so the native libasdk.so can
        # locate the .acfg hardware-configuration file (contains IP, port,
        # etc.).  If not set here the caller must have ACECFG in the
        # environment already.
        if acfg is not None:
            os.environ["ACECFG"] = acfg

        try:
            sdk_path = sdkp or os.path.join(CONFIGURATION_FOLDER, "alpao_sdk")

            if sdkp is not None and not os.path.exists(sdkp):
                sdk_path = os.path.join(CONFIGURATION_FOLDER, sdkp)

            if not os.path.exists(sdk_path):
                raise FileNotFoundError(
                    f"SDK path '{sdk_path}' does not exist. "
                    f"Set 'sdk_folder_path' in the configuration file to the "
                    f"parent folder that contains the 'Lib64/' directory, or "
                    f"place the SDK under "
                    f"'<SysConfig>/alpao_sdk/'."
                )

            sys.path.insert(0, sdk_path)

            from Lib64 import asdk  # type: ignore

        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "The 'asdk' module (Alpao SDK) could not be imported. "
                "Ensure 'sdk_folder_path' points to the parent folder "
                f"containing 'Lib64/' (current path: '{sdk_path}')."
            ) from e

        self._sdk_dm = asdk.DM(serial_number)
        # SDK default ResetOnClose=true zeros the mirror when ~DM runs
        # (process exit / GC / Ctrl-C).  Mirror construction flag.
        try:
            self._sdk_dm.Set("ResetOnClose", int(bool(self._reset_on_close)))
        except Exception:
            # Older SDK builds may lack the parameter; keep going.
            pass
        if reset_on_connect:
            self._sdk_dm.Reset()
