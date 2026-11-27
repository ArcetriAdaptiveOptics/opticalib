import numpy as _np
import socket
from opticalib.core.config import get_section_config
from opticalib.core.exceptions import CommandError
from opticalib.core import _types as _t
from ..base_devices import BaseDeformableMirror


class NACOMirror(BaseDeformableMirror):
    """
    Class to send commands to the NACO deformable mirror via PLC.

    The NACO mirror has no built-in position readback, so the last
    commanded vector is cached internally and returned by
    :meth:`get_shape`.

    Parameters
    ----------
    host : str
        IP address or hostname of the NACO mirror.
    port : int
        Port number for the PLC connection.
    """

    def __init__(
        self,
        host: str,
        port: int,
        reply_timeout: float = 1.0,
    ) -> None:
        """
        Initialize the NACO mirror interface.


        Parameters
        ----------
        host : str
            IP address or hostname of the NACO mirror.
        port : int
            Port number for the PLC connection.
        reply_timeout : float, optional
            Seconds to wait for the PLC reply after each command.
            Default is 1.0.

        Raises
        ------
        socket.gaierror
            If *host* cannot be resolved.
        """
        self._host = host
        self._port = port
        self._reply_timeout = reply_timeout
        # A connected UDP socket only receives datagrams from the PLC
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.connect((host, port))
        self.last_reply: _t.ArrayLike | None = None

        self.n_acts = 185
        self._dmCoords = {
            "dm185": [7, 9, 11, 13, 15],
        }

        self._last_cmd: _t.ArrayLike = _np.zeros(self.n_acts)
        self.act_coord = self._init_act_coord()
        self.is_segmented = False
        self.n_segments = 1
        self.n_acts_per_segment = self.n_acts
        self._cmd_history = None
        self.ref_act = None
        self.mirror_modes = None
        self.ff = None

    # ------------------------------------------------------------------
    # SDK-level interface
    # ------------------------------------------------------------------

    def get_shape(self) -> _t.ArrayLike:
        """
        Return the last commanded actuator positions.

        The NACO mirror does not provide a hardware readback; the last
        vector sent via :meth:`set_shape` is returned instead.

        Returns
        -------
        numpy.ndarray
            Array of length ``n_acts`` with the
            last commanded positions (zeros before the first command).
        """
        return self._last_cmd.copy()

    def _mirror_command(self, cmd: _t.ArrayLike) -> None:
        """
        Send an absolute command vector to the NACO mirror via the PLC.

        Data is sent as an UDP packet with 185 values in raw float32 format.
        The PLC will convert the values to the appropriate DAC units and send them to the mirror.

        The PLC replies to every packet: with the 185 computed actuator
        commands (raw float32) on success, or with a short error string
        if the packet was malformed. This method waits for that reply and
        stores the returned commands in ``last_reply``.

        Parameters
        ----------
        cmd : array_like
            Command vector of length ``n_acts``.

        Raises
        ------
        CommandError
            If the length of *cmd* does not match the number of
            actuators, if no reply is received from the PLC within
            the timeout, or if the PLC replies with an error.
        """
        cmd = _np.asarray(cmd, dtype=float)
        if cmd.size != self.n_acts:
            raise CommandError(
                f"Command length {cmd.size} does not match the number "
                f"of actuators ({self.n_acts})."
            )
        # Discard stale replies (e.g. a late answer to a timed-out command)
        self._sock.setblocking(False)
        try:
            while True:
                self._sock.recv(65535)
        except OSError:  # nothing left to read
            pass
        self._sock.settimeout(self._reply_timeout)
        # Send UDP packet to PLC (raw float32) and wait for its reply
        self._sock.send(cmd.astype(_np.float32).tobytes())
        try:
            reply = self._sock.recv(65535)
        except OSError as e:  # timeout, or ICMP port unreachable
            raise CommandError(
                f"No reply from NACO PLC at {self._host}:{self._port}: {e!r}"
            ) from e
        if len(reply) != 4 * self.n_acts:
            raise CommandError(f"NACO PLC rejected the command, reply: {reply!r}")
        self.last_reply = _np.frombuffer(reply, dtype=_np.float32).astype(float)
        self._last_cmd = cmd.copy()

    # ------------------------------------------------------------------
    # Higher-level helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Private initialisation helpers
    # ------------------------------------------------------------------

    def _init_act_coord(self) -> _t.Optional[_t.ArrayLike]:
        """
        Build the 2-D actuator coordinate array from the DM layout table.

        Returns
        -------
        numpy.ndarray or None
            Array of shape ``(n_acts, 2)`` with columns ``(x, y)`` pixel
            coordinates, or ``None`` if the model is unknown.
        """
        try:
            nacts_row_sequence = self._dmCoords[f"dm{self.n_acts}"]
        except KeyError:
            return None
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
        return _np.stack([cx, cy], axis=1)
