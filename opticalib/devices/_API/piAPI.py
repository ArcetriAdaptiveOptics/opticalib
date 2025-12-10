import numpy as np
from pipython import GCSDevice
from opticalib.typings import _ot
from pipython.pidevice.interfaces.pisocket import PISocket
from opticalib.core.read_config import getDmConfig, getDmIffConfig as _dmc
from opticalib.ground import logger

L = logger.getSystemLogger()


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

        if ip_addresses is None:
            self._ip_addresses = getDmConfig("PetalDM")
        else:
            self._ip_addresses = ip_addresses

        self._gateways = [PISocket(host=ip) for ip in self._ip_addresses]
        self._devices = [
            GCSDevice(gateway=gateway).gcsdevice for gateway in self._gateways
        ]

        if not all([dev.connected for dev in self._devices]):
            raise RuntimeError("Some connection did not get established")

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

    def _read_act_position(self) -> _ot.ArrayLike:
        """
        Read the current actuator positions from all segments.

        Returns
        -------
        np.ndarray
            An array containing the positions of all actuators.
        """
        pos = []
        for k, dev in enumerate(self._devices):
            L.log(20, f"Reading position from segment {k} : {self._ip_addresses[k]}")
            posx = dev.qPOS()
            posx = [posx["1"], posx["2"], posx["3"]]
            pos.extend(posx)
        return np.asarray(pos)

    def _get_last_cmd(self) -> _ot.ArrayLike:
        """
        Get the last command sent to the mirror.

        Returns
        -------
        np.ndarray
            An array containing the last command for all actuators.
        """
        pos = []
        for k, dev in enumerate(self._devices):
            L.log(
                20,
                f"Getting target position from segment {k} : {self._ip_addresses[k]}",
            )
            posx = dev.qMOV()
            posx = [posx["1"], posx["2"], posx["3"]]
            pos.extend(posx)
        return np.asarray(pos)

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

        for k, dev in enumerate(self._devices):
            L.log(20, f"Commanding position for segment {k} : {self._ip_addresses[k]}")
            segcmd = cmd[k * 3 : k * 3 + 3]
            odict = {"1": segcmd[0], "2": segcmd[1], "3": segcmd[2]}
            dev.MOV(odict)
            dev.checkerror()
