"""
This module contains the necessary high/user-leve functions to acquire the IFF data,
given a deformable mirror and an interferometer.

Author(s):
----------
- Pietro Ferraiuolo: pietro.ferraiuolo@inaf.it
- Runa Briguglio: runa.briguglio@inaf.it

"""

import os as _os
import numpy as _np
from opticalib.core.root import folders as _fn
from opticalib.core import read_config as _rif, exceptions as _oe
from . import iff_acquisition_preparation as _ifa
from opticalib.ground import osutils as _osu
from opticalib import typings as _ot


def iffDataAcquisition(
    dm: _ot.DeformableMirrorDevice,
    interf: _ot.InterferometerDevice,
    modesList: _ot.Optional[_ot.ArrayLike] = None,
    amplitude: _ot.Optional[float | _ot.ArrayLike] = None,
    template: _ot.Optional[_ot.ArrayLike] = None,
    shuffle: bool = False,
    differential: bool = False,
    read_buffer: bool | dict[str, _ot.Any] = False,
) -> str:
    """
    This is the user-lever function for the acquisition of the IFF data, given a
    deformable mirror and an interferometer.

    Except for the devices, all the arguments are optional, as, by default, the
    values are taken from the `iffConfig.ini` configuration file.

    Parameters
    ----------
    dm: DeformableMirrorDevice
        The inizialized deformable mirror object
    interf: InterferometerDevice
        The initialized interferometer object to take measurements
    modesList: ArrayLike , optional
        list of modes index to be measured, relative to the command matrix to be used
    amplitude: float | ArrayLike, optional
        command amplitude
    template: ArrayLike , oprional
        template file for the command matrix
    shuffle: bool , optional
        if True, shuffle the modes before acquisition
    differential: bool , optional
        if True, applies the commands differentially w.r.t. the initial shape of
        the DM.
    read_buffer: bool | dict[str, Any], optional
        If False (default) do not read the buffer data during the acquisition.
        If True, read the buffer data with default parameters.
        If a dictionary is provided, it is passed as keyword arguments to the
        `read_buffer` method of the deformable mirror device.

    Returns
    -------
    tn: str
        The tracking number of the dataset acquired, saved in the OPDImages folder
    """
    ifc = _ifa.IFFCapturePreparation(dm)
    tch = ifc.createTimedCmdHistory(
        modesList=modesList, modesAmp=amplitude, template=template, shuffle=shuffle
    )
    info = ifc.getInfoToSave()
    tn, _ = _prepareData2Save(info)
    _rif.copyIffConfigFile(tn)
    for param, value in zip(
        ["modeid", "modeamp", "template"], [modesList, amplitude, template]
    ):
        if value is not None:
            _rif.updateIffConfig(tn, param, value)
    dm.uploadCmdHistory(tch)
    if read_buffer is not False:
        try:
            if not hasattr(dm, "read_buffer"):
                raise _oe.BufferError(
                    f"The `{dm.__class__.__name__}` device cannot read buffer data."
                )
            if not type(read_buffer) == bool:
                rb_kwargs = read_buffer
            else:
                rb_kwargs = {}
            with dm.read_buffer(**rb_kwargs):
                try:
                    interf.setTriggerMode(True)
                except Exception as e:
                    interf._logger.error(f"Could not enable triggered mode on the interferometer.\nError: {e}")
                    pass
                dm.runCmdHistory(interf, save=tn, differential=differential)
            saveBufferData(dm, tn)
        except _oe.BufferError as be:
            print(be)
    else:
        dm.runCmdHistory(interf, save=tn, differential=differential)
    return tn


def acquirePistonData(
    dm: _ot.DeformableMirrorDevice,
    interf: _ot.InterferometerDevice,
    segmentID: int = 0,
    *,
    template: list[int],
    stepamp: float = 70e-9,
    nstep: int = 50,
    reverse: bool = False,
    differential: bool = False,
    read_buffer: bool = False,
) -> str:
    """
    This is the user-lever function for the acquisition of piston data of a
    segmented DM.

    The logic is to leave one of the segments fixed at "0" for reference and
    to move all the others with a stepping function which pistons all actuators
    back a forth on a Push-Pull basis.

    Parameters
    ----------
    dm: DeformableMirrorDevice
        The inizialized deformable mirror object
    interf: InterferometerDevice
        The initialized interferometer object to take measurements
    template: list[int]
        The template defining the stepping pattern. Must have an odd length.
    stepamp: float, optional
        The amplitude of each step. Default is 70e-9 m.
    nstep: int, optional
        The number of steps in the sequence. Default is 50.
    reverse: bool, optional
        If True, appends the reverse of the sequence to itself. Default is False.
    differential: bool , optional
        if True, applies the commands differentially w.r.t. the initial shape of
        the DM.
    read_buffer: bool | dict[str, Any], optional
        If False (default) do not read the buffer data during the acquisition.
        If True, read the buffer data with default parameters.
        If a dictionary is provided, it is passed as keyword arguments to the
        `read_buffer` method of the deformable mirror device.

    Returns
    -------
    tn: str
        The tracking number of the dataset acquired, saved in the OPDImages folder
    """
    ifc = _ifa.IFFCapturePreparation(dm)
    amps = _prepareSteppingAmplitudes(template, nstep, stepamp, reverse)
    cmdmat = _np.full((dm.nActs, len(amps)), 1.0)

    # check if dm is segmented
    try:
        if dm.is_segmented and not segmentID > dm.nSegments - 1:
            for ns in range(dm.nSegments):
                if not ns == segmentID:
                    idx = ns * dm.nActsPerSegment
                    cmdmat[idx : idx + dm.nActsPerSegment, :] = 0.0
    except AttributeError:
        print(
            f"--WARNING-- `{dm.__class__.__name__}` does not have the `is_segmented` attribute. Assuming monolitic DM."
        )
    finally:
        cmdmat *= amps[None, :]

    # create AmpVector compatible with iff processing
    ampvec = []
    ki = 0
    for kf in range(ki + len(template) - 1, len(amps), len(template)):
        ampvec.append(amps[ki:kf].max())
        ki = kf

    modeslist = _np.arange(len(ampvec))

    tch = ifc.createTimedCmdHistory(cmdmat, modeslist, ampvec, template, shuffle=False)
    info = ifc.getInfoToSave()

    # Hacking the standard IFF procedure
    info["ampVector"] = _np.asarray(ampvec)
    info["template"] = _np.asarray(template)
    info["cmdMatrix"] = _np.full((dm.nActs, len(amps)), 1.0)
    info["modesVector"] = modeslist
    info["indexList"] = modeslist
    info["shuffle"] = 0
    tn, _ = _prepareData2Save(info)

    _rif.copyIffConfigFile(tn)
    for param, value in zip(
        ["modeid", "modeamp", "template"], [modeslist, ampvec, template]
    ):
        if value is not None:
            _rif.updateIffConfig(tn, param, value)
    dm.uploadCmdHistory(tch)
    if read_buffer is not False:
        try:
            if not hasattr(dm, "read_buffer"):
                raise _oe.BufferError(
                    f"The `{dm.__class__.__name__}` device cannot read buffer data."
                )
            if not type(read_buffer) == bool:
                rb_kwargs = read_buffer
            else:
                rb_kwargs = {}
            with dm.read_buffer(**rb_kwargs):
                dm.runCmdHistory(interf, save=tn, differential=differential)
            saveBufferData(dm, tn)
        except _oe.BufferError as be:
            print(be)
    else:
        dm.runCmdHistory(interf, save=tn, differential=differential)
    return tn


def saveBufferData(dm: _ot.DeformableMirrorDevice, tn_or_fp: str):
    """
    Saves the buffer data from the deformable mirror device into a FITS file.

    Parameters
    ----------
    dm: DeformableMirrorDevice
        The initialized deformable mirror object
    tn_or_fp: str
        The tracking number or full path where to save the buffer data.
    """
    if not hasattr(dm, "read_buffer"):
        raise _oe.BufferError(
            f"The `{dm.__class__.__name__}` device cannot read buffer data."
        )
    if _osu.is_tn(tn_or_fp):
        iffpath = _os.path.join(_fn.IFFUNCTIONS_ROOT_FOLDER, tn_or_fp, "buffer_data.h5")
    elif not _os.path.exists(tn_or_fp):
        raise _oe.PathError(f"The path `{tn_or_fp}` does not exist.")
    else:
        iffpath = _os.path.join(tn_or_fp, "buffer_data.h5")
    bdata = dm.bufferData.copy()
    _osu.save_dict(bdata, iffpath, overwrite=True)


def _prepareData2Save(info: dict[str, _ot.Any]) -> tuple[str, str]:
    """
    Manages the creation of the folder to save the IFF data and saves
    the info dictionary in it, which comprehends:
    - the command history
    - the command amplitudes
    - the modes list
    - the template used
    - the shuffle flag

    Parameters
    ----------
    info: dict[str, Any]
        The info dictionary to be saved, gotten from the IFFCapturePreparation object

    Returns
    -------
    tn: str
        The tracking number of the dataset acquired, saved in the OPDImages folder
    iffpath: str
        The path to the folder where the IFF data are saved
    """
    tn = _osu.newtn()
    iffpath = _os.path.join(_fn.IFFUNCTIONS_ROOT_FOLDER, tn)
    if not _os.path.exists(iffpath):
        _os.mkdir(iffpath)
    try:
        for key, value in info.items():
            if not isinstance(value, _np.ndarray):
                tvalue = _np.asarray(value)
            else:
                tvalue = value
            if key == "shuffle":
                with open(_os.path.join(iffpath, f"{key}.dat"), "w") as f:
                    f.write(str(value))
            else:
                _osu.save_fits(
                    _os.path.join(iffpath, f"{key}.fits"), tvalue, overwrite=True
                )
    except KeyError as e:
        print(f"KeyError: {key}, {e}")
    return tn, iffpath


def _prepareSteppingAmplitudes(
    template: list[int], nstep: int, stepamp: float = 70e-9, reverse: bool = False
) -> _ot.ArrayLike:
    """
    Prepares a stepping amplitude sequence based on the provided template.

    Parameters
    ----------
    template: list[int]
        The template defining the stepping pattern. Must have an odd length.
    nstep: int
        The number of steps in the sequence.
    stepamp: float, optional
        The amplitude of each step. Default is 25e-6.
    substep_amp: int, optional
        The amplitude of the substep. Default is 25e-6.
    reverse: bool, optional
        If True, appends the reverse of the sequence to itself. Default is False.
    """
    M = len(template)

    if not M % 2:
        raise ValueError("Template must return to starting point (e.g, [1,-1,1])")

    fk = (
        _np.array(
            [i + (j % 2 == 0) for i in range(nstep) for j in range(1, M + 1)] + [nstep]
        )
        * stepamp
    )

    if reverse:
        fk = _np.concatenate((fk, _np.flip(fk)[1:]))

    # get rid of 0 amplitude at the beginning
    return fk[1:]
