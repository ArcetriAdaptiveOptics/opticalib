import xupy as _xp
import numpy as _np
from opticalib import typings as _ot
from opticalib.core import exceptions as _oe


def compute_slave_cmd(
    dm: _ot.DeformableMirror,
    cmd: _ot.ArrayLike,
    method: str = "zero-force",
) -> _ot.ArrayLike:
    """
    Compute the command vector including slaved actuators.

    Parameters
    ----------
    dm : opticalib.DeformableMirror
        Deformable mirror object with slaved actuators. Must have ther properties:
        - slaveIds : list of int
            List of indices of the slaved actuators.
        - ff : opticalib.MatrixLike
            Feed-Forward matrix of the deformable mirror.
    cmd : opticalib.ArrayLike
        Command vector for master actuators.
    method : str, optional
        Method to compute the master-to-slave matrix. Options are:
        - 'zero-force' : zero-force slaving, in which the slave actuators are
            commanded a position which needs zero force to be used (my require
            nearby actuators to apply more force)
        - 'minimum-rms' : minimum-RMS-force slaving, in which the slave actuators
            are set to minimize the overall force of nearby actuators.

        Defaults to 'zero-force'.

    Returns
    -------
    slaved_cmd : opticalib.ArrayLike
        The recomputed command with slaved actuators following the specified method.
        
    Raises
    ------
    opticalib.exceptions.DeviceError
        If the deformable mirror does not have the `.ff` method, i.e. a feed-forward matrix.
    opticalib.exceptions.ValueError
        If an unknown slaving method is specified.
    """
    sid = _np.array(sorted(dm.slaveIds))  # slave ids
    mid = _np.array([_i for _i in range(dm.nActs) if _i not in sid])  # master ids

    if not hasattr(dm, "ff"):
        raise _oe.DeviceError(
            f"Feed-Forward matrix not available in {dm.__class__.__name__}."
        )

    if method == "zero-force":
        return _zero_force_slaving(sid, mid, dm.ff, cmd)
    elif method == "minimum-rms":
        # get border actuators
        return _minimum_rms_slaving()
    else:
        raise _oe.ValueError(
            f"Unknown slaving method '{method}'. Available methods are 'zero-force' and 'minimum-rms'."
        )


def compute_slaved_IM(
    dm: _ot.DeformableMirror,
    IM: _ot.MatrixLike,
    method: str = "zero-force",
) -> _ot.MatrixLike:
    """
    Compute a new interaction matrix taking into account slaved actuators.

    Parameters
    ----------
    dm : opticalib.DeformableMirror
        Deformable mirror object with slaved actuators.
    IM : opticalib.MatrixLike
        Original interaction matrix.

    Returns
    -------
    nIM : opticalib.MatrixLike
        New interaction matrix with slaved actuators taken into account.
    """
    im = _xp.asarray(IM)

    sid = _np.array(sorted(dm.slaveIds))  # slave ids
    mid = _np.array([_i for _i in range(dm.nActs) if _i not in sid])  # master ids

    # compute new IM with slaved actuators
    try:
        ffwd = _xp.asarray(dm.ff)
    except AttributeError:
        raise _oe.DeviceError(
            f"Feed-Forward matrix not available in {dm.__class__.__name__}."
        )

    _, _, vt = _xp.linalg.svd(ffwd)
    zim = vt.T @ im  # zonal interaction matrix
    temp = vt.T[mid, :]  # mid
    nv = temp[:, mid]  # new Vt matrix slaved
    nIM = nv.T @ zim[mid, :]  # new interaction matrix
    return _xp.asnumpy(nIM)


def project_IM_into_zonal_IM(
    IM: _ot.MatrixLike,
    FFWD: _ot.MatrixLike,
) -> _ot.MatrixLike:
    """
    Project an interaction matrix into a zonal interaction matrix
    using the deformable mirror influence functions.

    Parameters
    ----------
    IM : MatrixLike
        General Interaction matrix to project into a Zonal IM.
    FFWD : MatrixLike
        Feed-Forward matrix of the deformable mirror.

    Returns
    -------
    ZIM : MatrixLike
        Zonal interaction matrix.
    """
    im, ff = _xp.asarray(IM), _xp.asarray(FFWD)
    _, _, vt = _xp.linalg.svd(ff)
    ZIM = vt.T @ im
    return _xp.asnumpy(ZIM)


def _zero_force_slaving(
    slaveIds: _ot.ArrayLike,
    masterIds: _ot.ArrayLike,
    ffwd: _ot.MatrixLike,
    cmd: _ot.ArrayLike,
) -> _ot.ArrayLike:
    """
    Computes the slave-to-master matrix using the zero-force method,
    and updates the command vector accordingly.

    The zero-force methode sets the slave actuators to positions that require
    zero force. Given the sub-matrices of the feed-forward matrix:

    ... math::
        K = \\begin{pmatrix} K_{mm} & K_{ms} \\\\ K_{sm} & K_{ss} \\end{pmatrix}

    the slaved command is computed as:

    ... math::
        c_s = -K_{ss}^{T} K_{sm} c_m

    Parameters
    ----------
    slaveIds : ArrayLike
        Indices of the slave actuators.
    masterIds : ArrayLike
        Indices of the master actuators.
    ffwd : MatrixLike
        Feed-Forward matrix of the deformable mirror.
    cmd : ArrayLike
        Command vector for master actuators.

    Returns
    -------
    slaved_cmd : ArrayLike
        Command vector including slaved actuators.
    """
    ffwd = _xp.asarray(ffwd)
    # s-s ffwd
    temp = ffwd[:, slaveIds]
    Kss = temp[slaveIds, :]

    # m-s ffwd
    temp = ffwd[:, masterIds]
    Kms = temp[slaveIds, :]

    # slave 2 master matrix
    S2M = -_xp.asnumpy(_xp.linalg.pinv(Kss) @ Kms)

    cmd[slaveIds] = S2M @ cmd[masterIds]
    return cmd


def _minimum_rms_slaving():
    """
    Docstring for _minimum_rms_slaving
    """
    raise NotImplementedError("Minimum-RMS slaving not implemented yet.")
