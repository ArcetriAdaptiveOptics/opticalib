import numpy as _np
from opticalib.core.read_config import getDmConfig
from opticalib.core.exceptions import CommandError
from opticalib import typings as _t


class BaseAlpaoMirror:

    def __init__(self, ip: str | None, port: int | None, nActs: int | str | None):
        self._dmCoords = {
            "dm97": [5, 7, 9, 11],
            "dm192": [4, 8, 12, 12, 16, 16, 18],
            "dm277": [7, 9, 11, 13, 15, 17, 19],
            "dm468": [8, 12, 16, 18, 20, 20, 22, 22, 24],
            "dm820": [10, 14, 18, 20, 22, 24, 26, 28, 28, 30, 30, 32],
            
        }
        self._dm = self._init_dm(ip, port, nActs)
        self.nActs = self._initNactuators()
        self._name = f"Alpao{self.nActs}"
        self.actCoord = self._initActCoord()
        self.diameter = getDmConfig(self._name).get("diameter", None)
        self.mirrorModes = None
        self.cmdHistory = None
        self.refAct = None

    @property
    def nActuators(self) -> int:
        return self.nActs

    def setReferenceActuator(self, refAct: int) -> None:
        if refAct < 0 or refAct > self.nActs:
            raise ValueError(f"Reference actuator {refAct} is out of range.")
        self.refAct = refAct

    def _checkCmdIntegrity(self, cmd: list[float], amp_threshold: float = 0.9) -> None:
        """
        Checks the integrity of the command vector.
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

    def _initNactuators(self) -> int:
        return self._dm.get_number_of_actuators()

    def _initActCoord(self):
        try:
            nacts_row_sequence = self._dmCoords[f"dm{self.nActs}"]
        except KeyError:
            self.actCoord = _np.array([], dtype=int)
            return
        n_dim = nacts_row_sequence[-1]
        upper_rows = nacts_row_sequence[:-1]
        lower_rows = [l for l in reversed(upper_rows)]
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

    def _init_dm(
        self, ip: str | None, port: int | None, nacts: int | str | None
    ) -> object:
        import plico_dm

        if (ip, port) == (None, None) and nacts is not None:
            name = f"Alpao{int(nacts)}"
            config = getDmConfig(name)
            self.ip, self.port = config.get("ip"), config.get("port")
        elif (ip, port, nacts) == (None, None, None):
            raise ValueError("Either (ip, port) or nacts must be provided.")
        else:
            self.ip, self.port = ip, port
        return plico_dm.deformableMirror(self.ip, self.port)
