"""
This module contains the classes for the high-level use of deformable mirrors.

Author(s)
---------
- Pietro Ferraiuolo : written in 2025

Description
-----------

"""

import os as _os
import numpy as _np
import time as _time
from . import _API as _api
from opticalib import typings as _ot
from contextlib import contextmanager as _contextmanager
from opticalib.core.root import OPD_IMAGES_ROOT_FOLDER as _opdi
from opticalib.ground.osutils import newtn as _ts, save_fits as _sf
from opticalib.core import exceptions as _oe
from opticalib.core.read_config import getDmIffConfig as _dmc
import types as _types


class AdOpticaDm(_api.BaseAdOpticaDm, _api.base_devices.BaseDeformableMirror):
    """
    AdOptica Deformable Mirror interface.

    Used with the AdOptica AO Client. In use for the DP, and will later be used for M4.
    """

    def __init__(self, tn: _ot.Optional[str] = None):
        """The Constructor"""
        self._name = "AdOpticaDm"
        super().__init__(tn)
        self._lastCmd = _np.zeros(self.nActs)
        self._lastCmdDiff = False

    def get_shape(self):
        """
        Retrieve the actuators positions
        """
        pos = self._aoClient.getPosition()
        return pos

    def set_shape(
        self,
        cmd: _ot.ArrayLike | list[float],
        differential: bool = True,
        incremental: float = False,
    ):  # cmd, segment=None):
        """
        Applies the given command to the DM actuators.

        Parameters
        ----------
        cmd : ArrayLike | list[float]
            The command to be applied to the DM actuators, of lenght equal
            the number of actuators.
        differential : bool, optional
            If True, the command will be applied as a differential command
            with respect to the current shape (default is False).
        incremental : float, optional
            If provided, the command will be applied incrementally in steps of
            size `incremental` (if <1) of in `N=incremental` steps (if >1)
            (default is False, meaning the command is applied in one go).
        """
        if not len(cmd) == self.nActs:
            raise _oe.CommandError(
                f"Command length {len(cmd)} does not match the number of actuators {self.nActs}."
            )
        if differential:
            self._lastCmd += cmd
        self._lastCmd = cmd
        if incremental:
            dc = _np.ceil((1 / incremental))
            if dc < 1 and incremental > 1.0:
                dc = incremental
                incremental = 1.0 / incremental
            for i in range(dc):
                if i * incremental > 1.0:
                    self._aoClient.mirrorCommand(cmd)
                else:
                    self._aoClient.mirrorCommand(cmd * i * incremental)
        else:
            self._aoClient.mirrorCommand(cmd)

    def uploadCmdHistory(self, tcmdhist: _ot.MatrixLike) -> None:
        """
        Uploads the (timed) command history in the DM. if `for_triggered` is true,
        then it is loaded direclty in the AO client for the triggere mode run.

        Parameters
        ----------
        tcmdhist : _ot.MatrixLike
            The command history to be uploaded, of shape (used_acts, nmodes).
        tfor_triggered : bool, optional
            If True, the command history will be uploaded directly to the AO client for
            the triggered mode run. If False, it will be stored in the `cmdHistory`
            attribute of the DM instance (default is False).
        """
        if not _ot.isinstance_(tcmdhist, "MatrixLike"):
            raise _oe.MatrixError(
                f"Expecting a 2D Matrix of shape (used_acts, nmodes), got instead: {tcmdhist.shape}"
            )
        if all(self._lastCmd != _np.zeros(self.nActs)):
            tcmdhist += self._lastCmd[:, None]
        trig = _dmc()["triggerMode"]
        if trig is not False:
            self.cmdHistory = tcmdhist.copy()
            self._aoClient.timeHistoryUpload(tcmdhist)
        else:
            self.cmdHistory = tcmdhist.copy()
        print("Time History uploaded!")

    def runCmdHistory(
        self,
        interf: _ot.Optional[_ot.InterferometerDevice] = None,
        differential: bool = False,
        save: _ot.Optional[str] = None,
    ) -> None:
        """
        Runs the loaded command history on the DM. If `triggered` is not False, it must
        be a dictionary containing the low level arguments for the `aoClient.timeHistoryRun` function.

        Parameters
        ----------
        interf : _ot.InterferometerDevice
            The interferometer device to be used for acquiring images during the command history run.
        differential : bool, optional
            If True, the commands will be applied as differential commands (default is False).
        triggered : bool | dict[str, _ot.Any], optional
            If False, the command history will be run in a sequential mode.
            If not False, a dictionary must be provided, where it should contain the keys
            'freq', 'wait', and 'delay' for the triggered mode.
        sequential_delay : int | float, optional
            The delay between each command execution in seconds (only if not in
            triggered mode).
        save : str, optional
            If provided, the command history will be saved with this name as a timestamp.
        """
        dmifconf = _dmc()
        triggered = dmifconf["triggerMode"]
        sequential_delay = dmifconf["sequentialDelay"]
        if triggered is not False:
            for arg in triggered.keys():
                if not arg in ["frequency", "cmdDelay"]:
                    raise _oe.CommandError(
                        f"Invalid argument '{arg}' in triggered commands."
                    )
            freq = triggered.get("frequency", 1.0)
            tdelay = triggered.get("cmdDelay", 0.8)
            ins = _np.zeros(self.nActs)
            self._aoClient.timeHistoryRun(freq, 0, tdelay)
            nframes = self._tCmdHistory.shape[-1]
            if interf is not None:
                interf.capture(nframes - 2, save)
            self.set_shape(ins)
        else:
            if self.cmdHistory is None:
                raise _oe.CommandError("No Command History to run!")
            else:
                tn = _ts() if save is None else save
                print(f"{tn} - {self.cmdHistory.shape[-1]} images to go.")
                datafold = _os.path.join(self.baseDataPath, tn)
                s = self.get_shape() - self._biasCmd
                if not _os.path.exists(datafold) and interf is not None:
                    _os.mkdir(datafold)
                for i, cmd in enumerate(self.cmdHistory.T):
                    print(f"{i+1}/{self.cmdHistory.shape[-1]}", end="\r", flush=True)
                    if differential:
                        cmd = cmd + s
                    self.set_shape(cmd)
                    if interf is not None:
                        _time.sleep(sequential_delay)
                        img = interf.acquire_map()
                        path = _os.path.join(datafold, f"image_{i:05d}.fits")
                        _sf(path, img)


class DP(AdOpticaDm):
    """
    Deformable Mirror interface for the Deformable Platform (DP) of the ELT.

    Used with the AdOptica AO Client.
    """

    def __init__(self, tn: _ot.Optional[str] = None):
        """The Constructor"""
        self._name = "DP"
        super().__init__(tn)

    def set_shape(
        self,
        cmd: _ot.ArrayLike | list[float],
        differential: bool = True,
        incremental: float = False,
    ):  # cmd, segment=None):
        """
        Applies the given command to the DM actuators.

        Parameters
        ----------
        cmd : ArrayLike | list[float]
            The command to be applied to the DM actuators, of lenght equal
            the number of actuators.
        differential : bool, optional
            If True, the command will be applied as a differential command
            with respect to the current shape (default is False).
        incremental : float, optional
            If provided, the command will be applied incrementally in steps of
            size `incremental` (if <1) of in `N=incremental` steps (if >1)
            (default is False, meaning the command is applied in one go).
        """
        if not len(cmd) == self.nActs:
            raise _oe.CommandError(
                f"Command length {len(cmd)} does not match the number of actuators {self.nActs}."
            )
        if differential:
            self._lastCmd += cmd
        else:
            self._lastCmd = cmd
        if incremental:
            dc = _np.ceil((1 / incremental))
            if dc < 1 and incremental > 1.0:
                dc = incremental
                incremental = 1.0 / incremental
            else:
                dc = int(dc)
            for i in range(dc):
                if i * incremental > 1.0:
                    self._aoClient.mirrorCommand(self._lastCmd)
                else:
                    self._aoClient.mirrorCommand(self._lastCmd * i * incremental)
        else:
            self._aoClient.mirrorCommand(self._lastCmd)

    @_contextmanager
    def read_buffer(
        self, segment: int = 0, npoints_per_cmd: int = 100, total_frames: int = None
    ):
        """
        Context manager for reading internal buffers of the DP DM during operations.

        The buffer data is acquired while executing commands within the context,
        and stored in `self.bufferData` upon exit.

        Parameters
        ----------
        segment : int, optional
            Segment number to read from (0 or 1 for DP, default: 0)
        npoints_per_cmd : int, optional
            Number of data points to acquire per command (default: 100)
        total_frames : int, optional
            Total number of frames to read (default: None, meaning use command history length)

        Yields
        ------
        dict
            A dictionary that will be populated with buffer results:
            - 'actPos': actuator positions (222, buffer_length)
            - 'actForce': actuator forces (222, buffer_length)

        Example
        -------
        >>> with dm.read_buffer(npoints_per_cmd=150) as buf:
        ...     dm.runCmdHistory(interf=myInterf, save='test_run')
        >>> print(buf['actPos'].shape)  # Access the buffer data
        (111, 33300)
        >>> # Or access via class attribute
        >>> print(dm.bufferData['actPos'].shape)
        """
        # Setup: Configure and start buffer acquisition
        nActs = 222
        if self.cmdHistory is not None:
            totframes = self.cmdHistory.shape[-1]
        elif total_frames is not None:
            totframes = total_frames
        else:
            raise _oe.BufferError(
                "Missing `total_frames` value: either load a command history or provide the variable's value"
            )
        triggered = _dmc()["triggerMode"]
        if triggered is not False:
            thistfreq = triggered.get("frequency", 1.0)
        if segment == 0:
            subsys = self._aoClient.aoSystem.aoSubSystem0
        else:
            subsys = self._aoClient.aoSystem.aoSubSystem1
        buffer_len = npoints_per_cmd * totframes + (nActs * 2)  # Extra margin
        clockfreq = subsys.sysConf.gen.cntFreq()
        thistdecim = int(clockfreq / thistfreq)
        diagdecim = int(thistdecim / npoints_per_cmd)

        subsys.support.diagBuf.config(
            _np.r_[0:nActs],
            buffer_len,
            "mirrActMap",
            decFactor=diagdecim,
            startPointer=0,
        )
        subsys.support.diagBuf.start()

        # Create a result container that will be populated on exit
        result = {}

        try:
            # Yield control back to the caller
            # Here you can call e.g. `runCmdHistory`
            yield result

        finally:
            # Cleanup: Stop acquisition and read data
            subsys.support.diagBuf.waitStop()
            bufData = subsys.support.diagBuf.read()

            # Process the buffer data
            actPos = _np.zeros((nActs, buffer_len))
            actForce = _np.zeros((nActs, buffer_len))

            for act_idx in range(nActs):
                tmp = bufData[f"ch{act_idx:04d}"]
                actPos[act_idx, :] = tmp[:, 4]
                actForce[act_idx, :] = tmp[:, 16]

            # Store in both the yielded dict and class attribute
            result["actPos"] = actPos
            result["actForce"] = actForce
            result["rawData"] = bufData

            self.bufferData = result.copy()


class M4AU(AdOpticaDm):
    """
    Deformable Mirror interface for the M4 Auxiliary Unit (M4AU) of the ELT.

    Used with the AdOptica AO Client.
    """

    def __init__(self, tn: _ot.Optional[str] = None):
        """The Constructor"""
        self._name = "M4AU"
        super().__init__(tn)


class AlpaoDm(_api.BaseAlpaoMirror, _api.base_devices.BaseDeformableMirror):
    """
    Alpao Deformable Mirror interface.
    """

    def __init__(
        self,
        nacts: _ot.Optional[int | str] = None,
        ip: _ot.Optional[str] = None,
        port: _ot.Optional[int] = None,
    ):
        """The Contructor"""
        super().__init__(ip, port, nacts)
        self.baseDataPath = _opdi

    def get_shape(self) -> _ot.ArrayLike:
        shape = self._dm.get_shape()
        return shape

    def set_shape(self, cmd: _ot.ArrayLike, differential: bool = False) -> None:
        if differential:
            shape = self._dm.get_shape()
            cmd = cmd + shape
        self._checkCmdIntegrity(cmd)
        self._dm.set_shape(cmd)

    def setZeros2Acts(self):
        zero = _np.zeros(self.nActs)
        self.set_shape(zero)

    def uploadCmdHistory(self, tcmdhist: _ot.MatrixLike) -> None:
        if not _ot.isinstance_(tcmdhist, "MatrixLike"):
            raise _oe.MatrixError(
                f"Expecting a 2D Matrix of shape (used_acts, nmodes), got instead: {tcmdhist.shape}"
            )
        self.cmdHistory = tcmdhist

    def runCmdHistory(
        self,
        interf: _ot.InterferometerDevice = None,
        delay: int | float = 0.2,
        save: str = None,
        differential: bool = True,
    ) -> str:
        if self.cmdHistory is None:
            raise _oe.MatrixError("No Command History to run!")
        s = self.get_shape()
        if isinstance(interf, tuple):
            if isinstance(interf[0], (_types.FunctionType, _types.MethodType)):
                # interf = interf[0](*interf[1:]) That is how to call
                tn = []
                for i, cmd in enumerate(self.cmdHistory.T):
                    if differential:
                        cmd = cmd + s
                    self.set_shape(cmd)
                    if interf is not None:
                        _time.sleep(delay)
                        img = interf[0](*interf[1:])
                        tn.append(img)
        else:
            tn = _ts.now() if save is None else save
            print(f"{tn} - {self.cmdHistory.shape[-1]} images to go.")
            datafold = _os.path.join(self.baseDataPath, tn)
            if not _os.path.exists(datafold) and interf is not None:
                _os.mkdir(datafold)
            for i, cmd in enumerate(self.cmdHistory.T):
                print(f"{i+1}/{self.cmdHistory.shape[-1]}", end="\r", flush=True)
                if differential:
                    cmd = cmd + s
                self.set_shape(cmd)
                if interf is not None:
                    _time.sleep(delay)
                    img = interf.acquire_map()
                    path = _os.path.join(datafold, f"image_{i:05d}.fits")
                    _sf(path, img)
        self.set_shape(s)
        return tn


class SplattDm(_api.base_devices.BaseDeformableMirror):
    """
    SPLATT deformable mirror interface.
    """

    def __init__(self, ip: str = None, port: int = None):
        """The Constructor"""
        self._name = "Splatt"
        self._dm = _api.SPLATTEngine(ip, port)
        self.nActs = self._dm.nActs
        self.mirrorModes = self._dm.mirrorModes
        self.actCoord = self._dm.actCoords
        self.cmdHistory = None
        self.baseDataPath = _opdi
        self.refAct = 16

    def get_shape(self):
        shape = self._dm.get_position()
        return shape

    def set_shape(self, cmd: _ot.ArrayLike, differential: bool = False) -> None:
        if differential:
            lastCmd = self._dm.get_position_command()
            cmd = cmd + lastCmd
        self._checkCmdIntegrity(cmd)
        self._dm.set_position(cmd)

    def uploadCmdHistory(self, tcmdhist: _ot.MatrixLike) -> None:
        if not _ot.isinstance_(tcmdhist, "MatrixLike"):
            raise _oe.MatrixError(
                f"Expecting a 2D Matrix of shape (used_acts, nmodes), got instead: {tcmdhist.shape}"
            )
        self.cmdHistory = tcmdhist

    def runCmdHistory(
        self,
        interf: _ot.Optional[_ot.InterferometerDevice] = None,
        delay: int | float = 0.2,
        save: _ot.Optional[str] = None,
        differential: bool = True,
        read_buffers: bool = False,
    ) -> str:
        if self.cmdHistory is None:
            raise _oe.MatrixError("No Command History to run!")
        else:
            tn = _ts() if save is None else save
            print(f"{tn} - {self.cmdHistory.shape[-1]} images to go.")
            datafold = _os.path.join(self.baseDataPath, tn)
            s = self._dm.get_position_command()  # self._dm.flatPos # self.get_shape()
            if read_buffers is True:
                delay = 0.0
            if not _os.path.exists(datafold) and interf is not None:
                _os.mkdir(datafold)
            for i, cmd in enumerate(self.cmdHistory.T):
                print(f"{i+1}/{self.cmdHistory.shape[-1]}", end="\r", flush=True)
                if differential:
                    cmd = cmd + s
                self.set_shape(cmd)
                if read_buffers is True:
                    pos, cur, bufTN = self._dm.read_buffers(
                        external=True, n_samples=300
                    )
                    path = _os.path.join(datafold, f"buffer_{i:05d}.fits")
                    hdr_dict = {"BUF_TN": str(bufTN)}
                    _sf(path, [pos, cur], hdr_dict)
                if interf is not None:
                    _time.sleep(delay)
                    img = interf.acquire_map()
                    path = _os.path.join(datafold, f"image_{i:05d}.fits")
                    _sf(path, img)
        self.set_shape(s)
        return tn

    def plot_command(self, cmd: _ot.ArrayLike) -> None:
        self._dm.plot_splatt_vec(cmd)

    def sendBufferCommand(
        self, cmd: _ot.ArrayLike, differential: bool = False, delay: int | float = 1.0
    ) -> str:
        # cmd is a command relative to self._dm.flatPos
        if differential:
            lastCmd = self._dm.get_position_command()
            cmd = cmd + lastCmd
        self._checkCmdIntegrity(cmd)
        cmd = cmd.tolist()
        tn = self._dm._eng.read(f"prepareCmdHistory({cmd})")
        # if accelerometers is not None:
        #   accelerometers.start_schedule()
        self._dm._eng.oneway_send(f"pause({delay}); sendCmdHistory(buffer)")
        return tn

    @property
    def nActuators(self) -> int:
        return self.nActs

    def integratePosition(self, Nits: int = 3):
        self._dm._eng.send(f"splattIntegrateMeasPos({Nits})")

    def _checkCmdIntegrity(self, cmd: _ot.ArrayLike) -> None:
        pos = cmd + self._dm.flatPos
        if _np.max(pos) > 1.2e-3:
            raise _oe.CommandError(
                f"End position is too high at {_np.max(pos)*1e+3:1.2f} [mm]"
            )
        if _np.min(pos) < 450e-6:
            raise _oe.CommandError(
                f"End position is too low at {_np.min(pos)*1e+3:1.2f} [mm]"
            )
