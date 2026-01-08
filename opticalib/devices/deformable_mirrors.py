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
from opticalib.core import exceptions as _oe
from contextlib import contextmanager as _contextmanager
from opticalib.core.read_config import getDmIffConfig as _dmc
from opticalib.core.root import OPD_IMAGES_ROOT_FOLDER as _opdi
from opticalib.ground.osutils import newtn as _ts, save_fits as _sf
from opticalib.ground.logger import SystemLogger as _SL

class PetalMirror(_api.BasePetalMirror, _api.base_devices.BaseDeformableMirror):
    """
    Petal Deformable Mirror interface.

    Used with the AdOptica AO Client.
    """

    def __init__(self, ip_addresses: _ot.Optional[list[str]] = None):
        """The Constructor"""
        self._name = "PetalDM"
        self._logger = _SL(the_class=__class__)
        super().__init__(ip_addresses)
        self.mirrorModes = None
        self.cmdHistory = None

    def get_shape(self) -> _ot.ArrayLike:
        """
        Retrieve the actuators positions
        """
        return self._read_act_position()

    def set_shape(self, cmd: _ot.ArrayLike, differential: bool = False) -> None:
        """
        Applies the given command to the DM actuators.

        Parameters
        ----------
        cmd : ArrayLike
            The command to be applied to the DM actuators, of lenght equal
            the number of actuators.
        differential : bool, optional
            If True, the command will be applied as a differential command
            with respect to the current shape (default is False).
        """
        self._mirror_command(cmd, differential)

    def uploadCmdHistory(self, tcmdhist: _ot.MatrixLike) -> None:
        """
        Uploads the (timed) command history to the DM.

        Parameters
        ----------
        tcmdhist : _ot.MatrixLike
            The command history to be uploaded, of shape (nActs, nmodes).
        """
        if not _ot.isinstance_(tcmdhist, "MatrixLike"):
            self._logger.error(f"MatrixError: Expecting a 2D Matrix of shape (nActs, nmodes), got instead: {tcmdhist.shape}")
            raise _oe.MatrixError(
                f"Expecting a 2D Matrix of shape (nActs, nmodes), got instead: {tcmdhist.shape}"
            )
        self.cmdHistory = tcmdhist
        self._logger.info(f'Loaded Timed command history of shape {tcmdhist.shape}')

    def runCmdHistory(
        self,
        interf: _ot.Optional[_ot.InterferometerDevice] = None,
        differential: bool = False,
        save: _ot.Optional[str] = None,
    ) -> None:
        """
        Runs the loaded command history on the DM.

        Parameters
        ----------
        interf : _ot.InterferometerDevice
            The interferometer device to be used for acquiring images during the command history run.
        differential : bool, optional
            If True, the commands will be applied as differential commands (default is False).
        save : str, optional
            If provided, the data will be saved in a folder with this name, instead of a freshly
            generated timestamp.
        """
        self._logger.info('Starting to run the command history')

        iff_config = _dmc()

        if self.cmdHistory is None:
            self._logger.error("MatrixError: No Command History to run!")
            raise _oe.MatrixError("No Command History to run!")

        else:
            tn = _ts() if save is None else save
            self._logger.info(f"Acquiring {tn} - {self.cmdHistory.shape[-1]} images")
            print(f"{tn} - {self.cmdHistory.shape[-1]} images to go.")

            # Create the Data folder
            datafold = _os.path.join(_opdi, tn)
            if not _os.path.exists(datafold) and interf is not None:
                _os.mkdir(datafold)

            # Getting starting position for differential commands
            s = self.get_shape()

            # Main loop for running the history
            for i, cmd in enumerate(self.cmdHistory.T):
                print(f"{i+1}/{self.cmdHistory.shape[-1]}", end="\r", flush=True)

                if differential:
                    cmd = cmd + s
                self.set_shape(cmd)

                if interf is not None:
                    _time.sleep(iff_config["delay"])
                    img = interf.acquire_map()
                    _sf(_os.path.join(datafold, f"image_{i:05d}.fits"), img)

            # Return to starting position
            self.set_shape(s)


class AdOpticaDm(_api.BaseAdOpticaDm, _api.base_devices.BaseDeformableMirror):
    """
    AdOptica Deformable Mirror interface.

    Used with the AdOptica AO Client. In use for the DP, and will later be used for M4.
    """

    def __init__(self, tn: _ot.Optional[str] = None):
        """The Constructor"""
        self._name = "AdOpticaDM"
        super().__init__(tn)
        self._lastCmd = _np.zeros(self.nActs)
        self._slaveIds = _dmc().get("slaveIds", [])
        self._borderIds = _dmc().get("borderIds", [])

    @property
    def slaveIds(self):
        return self._slaveIds

    @property
    def borderIds(self):
        return self._borderIds

    def get_shape(self):
        """
        Retrieve the actuators positions
        """
        pos = self._aoClient.getPosition()
        return pos

    def set_shape(
        self,
        cmd: _ot.ArrayLike | list[float],
        differential: bool = False,
        incremental: float | int = False,
        *,
        slaving_method: str = "zero-force",
    ) -> None:
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
        incremental: float|int, optional
            If provided, the command will be applied incrementally in steps of
            size `incremental` (if <1) or in `N=incremental` steps (if >1)
            (default is False, meaning the command is applied in one go).

            If incremental is positive, the command is applied from the current
            shape to the target shape, while if negative, it is applied in reverse
            (so, if a `lastCmd` is available, it returns to it, else it goes to 0 cmd).
        slaving_method : str, optional
            Method to compute the master-to-slave matrix. Options are:
            - 'zero-force' : zero-force slaving, in which the slave actuators are
                commanded a position which needs zero force to be used (my require
                nearby actuators to apply more force)
            - 'minimum-rms' : minimum-RMS-force slaving, in which the slave actuators
                are set to minimize the overall force of nearby actuators.
        """
        if not len(cmd) == self.nActs:
            raise _oe.CommandError(
                f"Command length {len(cmd)} does not match the number of actuators {self.nActs}."
            )

        cmd = self._slaveCmd(cmd=cmd, method=slaving_method)

        fc1 = self._get_frame_counter()

        # Incremental case
        if incremental:

            # Compute increment
            # Determine direction and number of steps
            positive = True if incremental > 0 else False
            if abs(incremental) >= 1.0:
                # incremental is number of steps
                n_steps = int(abs(incremental))
                step_fraction = 1.0 / n_steps
            else:
                # incremental is step size
                step_fraction = abs(incremental)
                n_steps = int(_np.ceil(1.0 / step_fraction))

            # Create iteration (reverse if incremental is negative)
            dc = range(n_steps) if incremental > 0 else reversed(range(n_steps))
            incremental = step_fraction

            # Differential case
            if differential:
                for i in dc:
                    if i * incremental > 1.0:
                        self._aoClient.mirrorCommand(cmd + self._lastCmd)
                    else:
                        self._aoClient.mirrorCommand(
                            self._lastCmd + (cmd * i * incremental)
                        )
                cmd = (self._lastCmd + cmd) if positive else (self._lastCmd - cmd)

            # Absolute case
            else:
                for i in dc:
                    if i * incremental > 1.0:
                        self._aoClient.mirrorCommand(cmd)
                    else:
                        self._aoClient.mirrorCommand(cmd * i * incremental)
                cmd = cmd if positive else _np.zeros(self.nActs)

        # Not incremental case
        else:
            if differential:
                cmd += self._lastCmd
            self._aoClient.mirrorCommand(cmd)

        _time.sleep(0.2)  # needed to get fc updated
        fc2 = self._get_frame_counter()
        if not fc2 == fc1:
            self._logger.error(f"FRAME SKIPPED.")
        else:
            self._lastCmd = cmd.copy()

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
        tcmdhist += self._lastCmd[:, None]
        trig = _dmc()["triggerMode"]
        self.cmdHistory = tcmdhist.copy()
        if trig is not False:
            self._aoClient.timeHistoryUpload(tcmdhist)
        self._logger.info(f"Command History uploaded to the {self._name} DM.")
        print("Command History uploaded!")

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
            if self.cmdHistory is None:
                raise _oe.CommandError("No Command History uploaded!")
            freq = triggered.get("frequency", 1.0)
            tdelay = triggered.get("cmdDelay", 0.8)
            ins = self._lastCmd.copy()
            self._logger.info("Executing Command history")
            nframes = self.cmdHistory.shape[-1]
            self._aoClient.timeHistoryRun(freq, 0, tdelay)
            if interf is not None:
                interf.capture(nframes - 2, save)
            self.set_shape(ins)
            self._logger.info("Command history execution completed")
        else:
            if self.cmdHistory is None:
                raise _oe.CommandError("No Command History uploaded!")
            else:
                tn = _ts() if save is None else save
                print(f"{tn} - {self.cmdHistory.shape[-1]} images to go.")
                datafold = _os.path.join(self.baseDataPath, tn)
                s = self.get_shape() - self._biasCmd
                if not _os.path.exists(datafold) and interf is not None:
                    _os.mkdir(datafold)
                self._logger.info("Executing Command history")
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
                self._logger.info("Command history execution completed")

    def _get_frame_counter(self) -> int:
        """
        Get the current frame counter from the AO client

        Returns
        -------
        total_skipped_frames : int
            Current total skipped frames counter value
        """
        cc = self._aoClient.getCounters()
        values = []
        keyse = ["skipByCommand", "skipByDeltaCommand", "skipByForce"]
        for key in keyse:
            values.append(getattr(cc, key))
        total_skipped_frames = sum(values)
        return total_skipped_frames


class DP(AdOpticaDm):
    """
    Deformable Mirror interface for the Deformable Platform (DP) of the ELT.

    Used with the AdOptica AO Client.
    """

    def __init__(self, tn: _ot.Optional[str] = None):
        """The Constructor"""
        self._logger = _SL(the_class=__class__)
        super().__init__(tn)
        self._name = self._name.replace("DM", "DP")
        self.bufferData = None
        self.is_segmented = True
        self.nSegments: int = 2
        self.nActsPerSegment: int = 111

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
        subsys_nacts = 111
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
        buffer_len = npoints_per_cmd * totframes + (subsys_nacts * 2)  # Extra margin
        clockfreq = subsys.sysConf.gen.cntFreq
        thistdecim = int(clockfreq / thistfreq)
        diagdecim = int(thistdecim / npoints_per_cmd)

        subsys.support.diagBuf.config(
            _np.r_[0:subsys_nacts],
            buffer_len,
            "mirrActMap",
            decFactor=diagdecim,
            startPointer=0,
        )
        self._logger.info(
            f"DP Buffer readout configured: {buffer_len} samples at {clockfreq/diagdecim:1.2f} Hz"
        )
        self._logger.info("Starting DP Buffer readout")
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
            self._logger.info("DP Buffer readout completed")
            # Process the buffer data
            keys = [
                "globCounter",  #  0
                "statusBits",  #  1
                "ADCHigh",  #  2
                "ADCLow",  #  3
                "actPos",  #  4
                "posError",  #  5
                "preshapedBiadCmd",  #  6
                "preshapedBiadForce",  #  7
                "newFFcmd",  #  8
                "newFFforce",  #  9
                "controlPropForce",  # 10
                "controlDerivForce",  # 11
                "controlIntegForce",  # 12
                "dynamicFFmassForce",  # 13
                "dynamicFFdampForce",  # 14
                "dynamicFFposForce",  # 15
                "actForce",  # 16
            ]

            for act_idx in range(subsys_nacts):
                tmp = bufData[f"ch{act_idx:04d}"]
                for k, idx in zip(keys, range(tmp.shape[1])):
                    result[k] = tmp[:, idx]

            # Store in both the yielded dict and class attribute
            self.bufferData = result.copy()


class M4AU(AdOpticaDm):
    """
    Deformable Mirror interface for the M4 Auxiliary Unit (M4AU) of the ELT.

    Used with the AdOptica AO Client.
    """

    def __init__(self, tn: _ot.Optional[str] = None):
        """The Constructor"""
        super().__init__(tn)
        self._name = "M4AU"
        self.is_segmented = True
        self.nSegments = 6
        self.nActsPerSegment = 892


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
        self._logger = _SL(the_class=__class__)
        super().__init__(ip, port, nacts)
        self.baseDataPath = _opdi
        self.is_segmented = False
        self._slaveIds = _dmc().get("slaveIds", [])
        self._borderIds = _dmc().get("borderIds", [])
        self.has_slaved_acts = False if len(self._slaveIds) == 0 else True

    @property
    def slaveIds(self):
        return self._slaveIds

    @property
    def borderIds(self):
        return self._borderIds

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
        save: str = None,
        differential: bool = True,
    ) -> str:
        """ """
        iff_config = _dmc()
        delay: float = iff_config.get("delay", 0.0)

        if self.cmdHistory is None:
            raise _oe.MatrixError("No Command History to run!")

        s = self.get_shape()

        if isinstance(interf, tuple):
            import types

            if isinstance(interf[0], (types.FunctionType, types.MethodType)):
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

            tn = _ts() if save is None else save
            print(f"{tn} - {self.cmdHistory.shape[-1]} images to go.")
            datafold = _os.path.join(_opdi, tn)
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
                    _sf(_os.path.join(datafold, f"image_{i:05d}.fits"), img)
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
        self.is_segmented = False
        self._slaveIds = _dmc().get("slaveIds", [])
        self._borderIds = _dmc().get("borderIds", [])
        self._logger = _SL(the_class=__class__)

    @property
    def slaveIds(self):
        return self._slaveIds

    @property
    def borderIds(self):
        return self._borderIds

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
