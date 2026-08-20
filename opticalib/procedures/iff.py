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
from opticalib.core import _types as _ot
from opticalib.ground import osutils as _osu
from ..dmutils import iff_preparation as _ifa
from opticalib.core.root import folders as _fn
from opticalib.core import config as _rif, exceptions as _oe


def iff_data_acquisition(
    dm: _ot.DeformableMirrorDevice,
    wfs: _ot.InterferometerDevice | _ot.WFSDevice,
    modesList: _ot.Optional[_ot.ArrayLike] = None,
    amplitude: _ot.Optional[float | _ot.ArrayLike] = None,
    template: _ot.Optional[_ot.ArrayLike] = None,
    modalbase: _ot.Optional[str] = None,
    shuffle: bool = False,
    n_repetitions: int = 1,
    read_buffer: bool | dict[str, _ot.Any] = False,
    parallel_spacing: _ot.Optional[float] = None,
    **setshape_kwargs: dict[str, _ot.Any],
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
    wfs: InterferometerDevice | WFSDevice
        The initialized wavefront sensor object to take measurements
    modesList: ArrayLike , optional
        list of modes index to be measured, relative to the command matrix to be used
    amplitude: float | ArrayLike, optional
        command amplitude
    template: ArrayLike , oprional
        template file for the command matrix
    modalbase: str, optional
        Modal base to use. Default is None, which means the modal base is loaded
        from the 'iffconfig.ini' file.
    shuffle: bool , optional
        if True, shuffle the modes before acquisition
    read_buffer: bool | dict[str, Any], optional
        If False (default) do not read the buffer data during the acquisition.
        If True, read the buffer data with default parameters.
        If a dictionary is provided, it is passed as keyword arguments to the
        `read_buffer` method of the deformable mirror device.
    parallel_spacing: float, optional
        If > 0, pack zonal actuators into parallel poke groups with this
        minimum Euclidean spacing in ``dm.act_coord`` units. Default is None
        (read from config, else 0 = sequential).
    slave: bool | str, optional
        If True, the deformable mirror device is set to slave mode during the
        acquisition. If a string is provided, it specifies the slaving method to
        be used. Default to False

    Other Parameters
    ----------------
    **dm_wkargs: dict[str, Any]
        Additional keyword arguments to be passed to the deformable mirror device
        ``set_shape`` method.

    Returns
    -------
    tn: str
        The tracking number of the dataset acquired, saved in the OPDImages folder
    """
    ifc = _ifa.IFFCapturePreparation(dm)
    tch = ifc.create_timed_cmd_history(
        modesList=modesList,
        modesAmp=amplitude,
        template=template,
        shuffle=shuffle,
        modalBase=modalbase,
        n_repetitions=n_repetitions,
        parallel_spacing=parallel_spacing,
    )
    info = ifc.get_info_to_save()
    tn, _ = _prepare_data2_save(info)

    _rif.copy_iff_config_file(tn)
    # When parallel packing is active, modes_list in FILES is group indices;
    # keep iffConfig in sync with what was actually commanded.
    commanded_modes = info.get("modes_list")
    if commanded_modes is not None:
        modes_for_cfg = _np.asarray(commanded_modes).ravel()
        # Drop repetition tiling for config (unique group ids / mode ids)
        n_rep = int(info.get("n_repetitions", 1) or 1)
        if n_rep > 1 and modes_for_cfg.size % n_rep == 0:
            modes_for_cfg = modes_for_cfg[: modes_for_cfg.size // n_rep]
    else:
        modes_for_cfg = modesList

    spacing_to_save = parallel_spacing
    if spacing_to_save is None:
        raw = info.get("parallel_spacing", 0)
        spacing_to_save = float(_np.asarray(raw).ravel()[0]) if raw is not None else 0.0

    pars2update = dict(
        zip(
            [
                "modes_list",
                "amplitude",
                "template",
                "shuffle",
                "n_repetitions",
                "modal_base",
                "parallel_spacing",
            ],
            [
                modes_for_cfg,
                amplitude,
                template,
                shuffle,
                n_repetitions,
                modalbase,
                spacing_to_save,
            ],
        )
    )
    pars2update = {k: v for k, v in pars2update.items() if v is not None}

    _rif.update_iff_config(
        tn, item=list(pars2update.keys()), value=list(pars2update.values())
    )
    slaving = setshape_kwargs.pop("slave", False)
    dm.upload_cmd_history(tch, slave=slaving)
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
                dm.run_cmd_history(wfs, save=tn, **setshape_kwargs)
            save_buffer_data(dm, tn)
        except _oe.BufferError as be:
            print(be)
    else:
        _ = dm.run_cmd_history(wfs, save=tn, **setshape_kwargs)
    return tn


def piston_data_acquisition(
    dm: _ot.DeformableMirrorDevice,
    wfs: _ot.InterferometerDevice | _ot.WFSDevice,
    segmentID: int = 0,
    *,
    template: list[int],
    stepamp: float = 70e-9,
    nstep: int = 50,
    reverse: bool = False,
    differential: bool = False,
    read_buffer: bool | dict[str, _ot.Any] = False,
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
    wfs: InterferometerDevice | WFSDevice
        The initialized wavefront sensor object to take measurements
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
    amps = _prepare_stepping_amplitudes(template, nstep, stepamp, reverse)
    cmdmat = _np.full((dm.n_acts, len(amps)), 1.0)

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

    tch = ifc.create_timed_cmd_history(
        cmdmat, modeslist, ampvec, template, shuffle=False
    )
    info = ifc.get_info_to_save()

    # Hacking the standard IFF procedure
    info["amplitude"] = _np.asarray(ampvec)
    info["template"] = _np.asarray(template)
    info["cmd_matrix"] = _np.full((dm.n_acts, len(amps)), 1.0)
    info["modes_list"] = modeslist
    info["index_list"] = modeslist
    info["shuffle"] = 0
    tn, _ = _prepare_data2_save(info)

    _rif.copy_iff_config_file(tn)
    for param, value in zip(
        ["modeid", "modeamp", "template"], [modeslist, ampvec, template]
    ):
        if value is not None:
            _rif.update_iff_config(tn, param, value)
    dm.upload_cmd_history(tch)
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
                dm.run_cmd_history(wfs, save=tn, differential=differential)
            save_buffer_data(dm, tn)
        except _oe.BufferError as be:
            print(be)
    else:
        _ = dm.run_cmd_history(wfs, save=tn, differential=differential)
    return tn


def save_buffer_data(dm: _ot.DeformableMirrorDevice, tn_or_fp: str):
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
    _osu.save_h5(bdata, iffpath, overwrite=True)


def _prepare_data2_save(info: dict[str, _ot.Any]) -> tuple[str, str]:
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
            if key in ["shuffle", "n_repetitions"]:
                continue
            if not isinstance(value, _np.ndarray):
                tvalue = _np.asarray(value)
            else:
                tvalue = value
            if tvalue is None or (isinstance(tvalue, _np.ndarray) and tvalue.dtype == object):
                continue
            # FITS ImageHDU requires ndim >= 1
            if isinstance(tvalue, _np.ndarray) and tvalue.ndim == 0:
                tvalue = tvalue.reshape(1)
            _osu.save_fits(
                _os.path.join(iffpath, f"{key}.fits"), tvalue, overwrite=True
            )
    except KeyError as e:
        print(f"KeyError: {key}, {e}")
    return tn, iffpath


def _prepare_stepping_amplitudes(
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
        The amplitude of each step. Default is 70e-9.
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
