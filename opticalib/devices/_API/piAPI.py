import numpy as np
from opticalib import typings as _ot
from pipython import GCSDevice, GCSError
from opticalib.ground.logger import SystemLogger as _SL
from pipython.pidevice.interfaces.pisocket import PISocket
from opticalib.core.read_config import getDmConfig, getDmIffConfig as _dmc

class BasePetalMirror:
    """
    Base class for controlling a petal mirror device.

    Parameters
    ----------
    ip_addresses : list[str], optional
        List of IP addresses for the petal mirror segments. If None, the addresses
        will be retrieved from the configuration file.
    """

    def __init__(self, ip_addresses: list[str] = None):
        """
        Initialize the petal mirror device with the given addresses.
        """
        self._had_error = False

        if ip_addresses is None:
            self._ip_addresses = [ip for ip in getDmConfig("PetalDM").values()]
        else:
            self._ip_addresses = ip_addresses

        self._gateways = [PISocket(host=ip) for ip in self._ip_addresses]
        self._devices = [
            GCSDevice(gateway=gateway).gcsdevice for gateway in self._gateways
        ]

        if not all([dev.connected for dev in self._devices]):
            self._logger.error("Some connection did not get established")
            raise RuntimeError("Some connection did not get established")
        else:
            self._logger.info("All connections to petal mirror segments established")
            self._check_servos()
            self._morning_routine()

        self.is_segmented = True
        self.nSegments = len(self._devices)
        self.nActsPerSegment = 3
        self.nActs = self.nSegments * self.nActsPerSegment
        self._slaveIds = _dmc().get("slaveIds", [])
        self._borderIds = _dmc().get("borderIds", [])

    @property
    def slaveIds(self):
        """Get the IDs of the slave segments."""
        return self._slaveIds

    @property
    def borderIds(self):
        """Get the IDs of the border segments."""
        return self._borderIds
    
    def get_acc(self) -> float:
        """
        Get the acceleration setting for the piezo actuators.

        Returns
        -------
        float
            The acceleration value.
        """
        self._check_axes()
        try:
            acc = []
            for k, dev in enumerate(self._devices):
                self._logger.info(f"Getting acceleration from segment {k} : {self._ip_addresses[k]}")
                accx = dev.qACC()
                accx = [accx["1"], accx["2"], accx["3"]]
                acc.extend(accx)
            return np.asarray(acc)
        except GCSError as err:
            self._logger.error(f"Error getting acceleration: {err}")
            self._had_error = True
            raise RuntimeError("Failed to get acceleration") from err
    
    # TODO: Is this needed?
    # def set_acc(self, acc: float|_ot.ArrayLike) -> None:
    #     """
    #     Set the acceleration for all piezo actuators.

    #     Parameters
    #     ----------
    #     acc: float
    #         The acceleration value to set.
    #     """
    #     self._check_axes()
    #     try:
    #         for k, dev in enumerate(self._devices):
    #             L.log(20, f"Setting acceleration for segment {k} : {self._ip_addresses[k]}")
    #             odict = {"1": acc, "2": acc, "3": acc}
    #             dev.ACC(odict)
    #             dev.checkerror()
    #     except GCSError as err:
    #         L.error(f"Error setting acceleration: {err}")
    #         self._had_error = True
    #         raise RuntimeError("Failed to set acceleration") from err

    def _read_act_position(self) -> _ot.ArrayLike:
        """
        Read the current actuator positions from all segments.

        Returns
        -------
        np.ndarray
            An array containing the positions of all actuators.
        """
        self._check_axes()
        try:
            pos = []
            for k, dev in enumerate(self._devices):
                self._logger.info(f"Reading position from segment {k} : {self._ip_addresses[k]}")
                posx = dev.qPOS()
                posx = [posx["1"], posx["2"], posx["3"]]
                pos.extend(posx)
            return np.asarray(pos)
        except GCSError as err:
            self._logger.error(f"Error reading actuator positions: {err}")
            self._had_error = True
            raise RuntimeError("Failed to read actuator positions") from err

    def _get_last_cmd(self) -> _ot.ArrayLike:
        """
        Get the last command sent to the mirror.

        Returns
        -------
        np.ndarray
            An array containing the last command for all actuators.
        """
        self._check_axes()
        try:
            pos = []
            for k, dev in enumerate(self._devices):
                self._logger.info(
                    f"Getting target position from segment {k} : {self._ip_addresses[k]}"
                )
                posx = dev.qMOV()
                posx = [posx["1"], posx["2"], posx["3"]]
                pos.extend(posx)
            return np.asarray(pos)
        except GCSError as err:
            self._logger.error(f"Error getting last command positions: {err}")
            self._had_error = True
            raise RuntimeError("Failed to get last command positions") from err

    def _mirror_command(self, cmd: _ot.ArrayLike, differential: bool = False) -> None:
        """
        Send commands to set the mirror shape.

        Parameters
        ----------
        cmd: _ot.ArrayLike
            An array of commands for the actuators.
        differential: bool, optional
            If True, the command is treated as a differential adjustment to the current shape.
        """

        if not len(cmd) == self.nActs:
            raise ValueError(f"command length must be {self.nActs}")

        if differential:
            cmd += self._get_last_cmd()

        self._check_axes()
        try:
            for k, dev in enumerate(self._devices):
                self._logger.info(f"Commanding position for segment {k} : {self._ip_addresses[k]}")
                segcmd = cmd[k * 3 : k * 3 + 3]
                odict = {"1": segcmd[0], "2": segcmd[1], "3": segcmd[2]}
                dev.MOV(odict)
                dev.checkerror()
        except GCSError as err:
            self._logger.error(f"Error sending mirror command: {err}")
            self._had_error = True
            raise RuntimeError("Failed to send mirror command") from err

    def _morning_routine(self) -> None:
        """
        On system startup, this will warm up the piezos by moving the piston
        mode back and forth from `start->mid->end->mid->start` a few times, 
        settling at mid range in the end.
        """
        import time

        try:
            for k, dev in enumerate(self._devices):
                self._logger.info(f"Warming up piezos for segment {k} : {self._ip_addresses[k]}")
                for c in [0,6,12,6,0,6,12,6,0,6]:
                    dev.MOV({"1": c})
                    time.sleep(0.25)
                    dev.checkerror()
        except GCSError as err:
            self._logger.error(f"Error during warming up: {err}")
            self._had_error = True
            raise RuntimeError("Morning routine failed") from err
    
    def _check_servos(self) -> None:
        """
        Check the servo status of all segments.
        
        If servos are disabled, enable them.
        """
        self._check_axes()
        try:
            for k, dev in enumerate(self._devices):
                self._logger.info(f"Checking servos for segment {k} : {self._ip_addresses[k]}")
                status = dev.qSVO()
                self._logger.info(f"Servo status for segment {k}: {status}")
                if all(value == 0 for value in status.values()):
                    self._logger.info(f"Enabling servos for segment {k}")
                    dev.SVO({"1": 1, "2": 1, "3": 1})
                    dev.checkerror()
        except GCSError as err:
            self._logger.error(f"Error checking/enabling servos: {err}")
            self._had_error = True
            raise RuntimeError("Failed to check/enable servos") from err
    
    def _check_axes(self) -> None:
        """
        Check the axis status of all segments.
        
        If any axis is disabled, enable it.
        """
        if self._had_error:
            self._logger.warning("Previous error detected, enabling axes")
            self._enable_axes()
    
    def _enable_axes(self) -> None:
        """
        Enable axes.
        - GCS3: use EAX/qEAX.
        - GCS2 fallback: use SVO/qSVO (servo on).
        """
        try:
            for k, dev in enumerate(self._devices):
                self._logger.info(f"Checking segment {k} : {self._ip_addresses[k]}")
                try:
                    # Preferred for GCS3.x
                    status = dev.qEAX()
                    self._logger.info(f"         Axis status (EAX): {status}")
                    for axis, value in status.items():
                        if value == 0:
                            self._logger.info(f"         Enabling axis {axis} via EAX")
                            dev.EAX({axis: 1})
                            dev.checkerror()
                except (AttributeError, GCSError) as err:
                    # Fallback for GCS2.x
                    self._logger.info(f"EAX/qEAX not supported ({err}); falling back to SVO")
                    status = dev.qSVO()
                    self._logger.info(f"         Servo status (SVO): {status}")
                    to_enable = {ax: 1 for ax, val in status.items() if val == 0}
                    if to_enable:
                        self._logger.info(f"         Enabling servos via SVO: {to_enable}")
                        dev.SVO(to_enable)
                        dev.checkerror()
        except GCSError as err:
            self._logger.error(f"Error checking/enabling axes: {err}")
            self._had_error = True
            raise RuntimeError("Failed to check/enable axes") from err
