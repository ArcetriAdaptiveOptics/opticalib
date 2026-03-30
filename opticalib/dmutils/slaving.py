import xupy as _xp
import numpy as _np
from opticalib import typings as _ot
from opticalib.core import exceptions as _oe


def compute_slave_cmd(
    dm: _ot.DeformableMirrorDevice,
    cmd: _ot.ArrayLike,
    method: str = "zero-force",
) -> _ot.ArrayLike:
    """
    Compute the command vector including slaved actuators.

    Parameters
    ----------
    dm : opticalib.DeformableMirror
        Deformable mirror object with slaved actuators. Must have the properties:
        - slaveIds : list of int
            List of indices of the slaved actuators.
        - ff : opticalib.MatrixLike
            Feed-Forward matrix of the deformable mirror.
    cmd : opticalib.ArrayLike
        Command vector for master actuators.
    method : str, optional
        Method to compute the master-to-slave matrix. Options are:
        - 'zero-force' : zero-force slaving, in which the slave actuators are
            commanded a position which needs zero force to be used (may require
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
    DeviceError
        If the deformable mirror does not have the `.ff` method, i.e. a feed-forward matrix.
    ValueError
        If an unknown slaving method is specified.
    """
    sid, bid, mid = _get_act_roles(dm)

    if sid is None or len(sid) == 0:
        raise _oe.DeviceError(f"No slave actuators defined in {dm.__class__.__name__}.")

    if not hasattr(dm, "ff"):
        raise _oe.DeviceAttributeError(
            f"Feed-Forward matrix not available in {dm.__class__.__name__}."
        )

    if method == "zero-force":
        return _zero_force_slaving(sid, mid, dm.ff, cmd)
    elif method == "minimum-rms":
        return _minimum_rms_slaving(sid, mid, bid, dm.ff, cmd)

    else:
        raise ValueError(
            f"Unknown slaving method '{method}'. "
            "Available methods are 'zero-force' and 'minimum-rms'."
        )


def compute_slaved_IM(
    dm: _ot.DeformableMirrorDevice,
    IM: _ot.MatrixLike,
    method: str = None,
) -> _ot.MatrixLike:
    """
    Compute a new interaction matrix taking into account slaved actuators.

    Parameters
    ----------
    dm : opticalib.DeformableMirror
        Deformable mirror object with slaved actuators.
    IM : opticalib.MatrixLike
        Original interaction matrix.
    method : str, optional
        Method to compute the master-to-slave matrix. Options are:
        - None: Creates an IM with the same number of rows as the original one,
            controlling only the master actuators.
        - 'zero-force' : zero-force slaving, in which the slave actuators are
            commanded a position which needs zero force to be used (may require
            nearby actuators to apply more force)
        - 'minimum-rms' : minimum-RMS-force slaving, in which the slave actuators
            are set to minimize the overall force of nearby actuators.

    Returns
    -------
    nIM : opticalib.MatrixLike
        New interaction matrix with slaved actuators taken into account.
    """
    try:
        ffwd = _xp.asarray(dm.ff)
    except AttributeError:
        raise _oe.DeviceAttributeError(
            f"Feed-Forward matrix not available in {dm.__class__.__name__}."
        )

    im = _xp.asarray(IM)
    _, _, vt = _xp.linalg.svd(ffwd)
    zim = vt.T @ im  # zonal interaction matrix

    if method is not None:
        compute_slaved_mat(
            dm, zim, method=method
        )  # check method and get slaving matrix Q
    else:
        temp = vt.T[mid, :]  # mid
        nv = temp[:, mid]  # new Vt matrix slaved
        nIM = nv.T @ zim[mid, :]  # new interaction matrix

    return _xp.asnumpy(nIM)


def compute_slaved_mat(
    dm: _ot.DeformableMirrorDevice,
    M: _ot.MatrixLike,
    method: str = "zero-force",
) -> _ot.MatrixLike:
    """
    Compute a new interaction matrix taking into account slaved actuators.

    This matrix slaving method works for IFF matrices, FeedForward matrices and
    Interaction Matrix (zonal, not modal).

    Parameters
    ----------
    dm : opticalib.DeformableMirror
        Deformable mirror object with slaved actuators.
    M : opticalib.MatrixLike
        Original matrix to slaved.

        Works on:
        - `FeedForward` (nactuators, nactuators)
        - `IFF` (nactuators, npixels)
    method : str, optional
        Method to compute the master-to-slave matrix. Options are:
        - 'zero-force' : zero-force slaving, in which the slave actuators are
            commanded a position which needs zero force to be used (may require
            nearby actuators to apply more force)
        - 'minimum-rms' : minimum-RMS-force slaving, in which the slave actuators
            are set to minimize the overall force of nearby actuators.

    Returns
    -------
    nM : opticalib.MatrixLike
        New matrix with slaved actuators taken into account.
    """
    m = _xp.asarray(M)

    sid, bid, mid = _get_act_roles(dm)

    try:
        ffwd = _xp.asarray(dm.ff)
    except AttributeError:
        raise _oe.DeviceAttributeError(
            f"Feed-Forward matrix not available in {dm.__class__.__name__}."
        )

    Q = _get_slaving_matrix(method, ffwd, sid, mid, borderIds=bid)

    nM = m[mid, :] + Q.T @ m[sid, :]
    return _xp.asnumpy(nM)


def project_IM_into_zonal_IM(
    IM: _ot.MatrixLike,
    FFWD: _ot.MatrixLike,
) -> _ot.MatrixLike:
    """
    Project a modal interaction matrix into a zonal one using the deformable mirror
    Feed-Forward matrix.

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


def _get_act_roles(
    dm: _ot.DeformableMirrorDevice,
) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """
    Get the roles of actuators in the deformable mirror.

    Parameters
    ----------
    dm : opticalib.DeformableMirrorDevice
        Deformable mirror device.

    Returns
    -------
    sid : ndarray
        Indices of the slave actuators.
    bid : ndarray
        Indices of the border actuators.
    mid : ndarray
        Indices of the master actuators.
    """
    sid = _np.array(sorted(dm.slaveIds), dtype=int)  # slave ids
    bid = getattr(dm, "borderIds", None)

    if bid is None:
        mid = _np.array(
            [_i for _i in range(dm.nActs) if _i not in sid], dtype=int
        )  # master ids
        bid = mid.copy()  # border ids
    else:
        bid = _np.array(sorted(bid), dtype=int)  # border ids
        mid = _np.array(  # Master ids
            [_i for _i in range(dm.nActs) if _i not in sid and _i not in bid], dtype=int
        )

    return sid, bid, mid


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
        c_s = -K_{ss}^{-1} K_{sm} c_m

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

    Acknowledgements
    ----------------
    Method from <a href="https://arxiv.org/abs/2101.04801"> Riccardi,A.; 2021 (arXiv:2101.04801)</a>
    """
    cmd = _xp.asarray(cmd)

    K = _get_decomposed_ffwd(slaveIds, masterIds, ffwd, method="zero-force")
    Kss = K["ss"]
    Ksm = K["sm"]

    # slave 2 master matrix
    Q = -_xp.linalg.pinv(Kss) @ Ksm

    cmd[slaveIds] = Q @ cmd[masterIds]
    return _xp.asnumpy(cmd)


def _minimum_rms_slaving(
    slaveIds: _ot.ArrayLike,
    borderIds: _ot.ArrayLike,
    masterIds: _ot.ArrayLike,
    ffwd: _ot.MatrixLike,
    cmd: _ot.ArrayLike,
) -> _ot.ArrayLike:
    """
    Computes the slave-to-master matrix using the minimum-RMS-force method,
    and updates the command vector accordingly.

    With this method, the actuators are deviden in three groups:
    - slave actuators: actuators that are slaved (`s`)
    - border actuators: master actuators in a given area surrounding the border between master and slaved actuators (`b`)
    - master actuators: The rest of the master actuator (`m`)

    The minimum-RMS-force method sets the slave actuators to positions that minimize
    the overall force applied by the joint set of slaved and border actuators.
    In this way the slaved actuators can drive forces helping the border actuators in keeping
    their commanded positions, but reducing their peak force.

    The feed-forward matrix can be rewritten as:

    ... math::
        K = \\begin{pmatrix} K_{mm} & K_{mb} & K_{ms} \\\\ K_{bm} & K_{bb} & K_{bs} \\\\ K_{sm} & K_{sb} & K_{ss} \\end{pmatrix}

    and the slaved command is computed as:

    ... math::
        c_s = - (K^T_{bs}K_{bs} + K_{ss}K_{ss})^{-1} \\big[(K^T_{bs}K_{bi} + K^T_{ss}K_{si})c_i + (K^T_{bs}K_{bb} + K^T_{ss}K_{sb})c_b\\big]


    Parameters
    ----------
    slaveIds : ArrayLike
        Indices of the slave actuators.
    borderIds : ArrayLike
        Indices of the border actuators.
    masterIds : ArrayLike
        Indices of the master actuators.
    ffwd : MatrixLike
        Feed-Forward matrix of the deformable mirror.
    cmd : ArrayLike
        Command vector to recompute.

    Returns
    -------
    slaved_cmd : ArrayLike
        Command vector including slaved actuators.

    Acknowledgements
    ----------------
    Method from <a href="https://arxiv.org/abs/2101.04801"> Riccardi,A.; 2021 (arXiv:2101.04801)</a>
    """
    cmd = _xp.asarray(cmd)
    K = _get_decomposed_ffwd(slaveIds, masterIds, ffwd, borderIds, method="minimum-rms")

    Q0 = -_xp.linalg.pinv(K["bs"].T @ K["bs"] + K["ss"].T @ K["ss"])

    ci = (K["bs"].T @ K["bi"] + K["ss"].T @ K["si"]) @ cmd[masterIds]
    cb = (K["bs"].T @ K["bb"] + K["ss"].T @ K["sb"]) @ cmd[borderIds]

    n = 2 if all(borderIds == masterIds) else 1

    cmd[slaveIds] = Q0 @ (ci + cb) / n
    return _xp.asnumpy(cmd)


def _get_slaving_matrix(
    method: str,
    FF: _ot.MatrixLike,
    slaveIds: _ot.ArrayLike,
    masterIds: _ot.ArrayLike,
    borderIds: _ot.ArrayLike,
) -> _ot.MatrixLike:
    """
    Computes the slave-to-master matrix according to the specified method.

    Parameters
    ----------
    method : str
        Method to compute the master-to-slave matrix. Options are:
        - 'zero-force' : zero-force slaving
        - 'minimum-rms' : minimum-RMS-force slaving.
    FF : MatrixLike
        Feed-Forward matrix of the deformable mirror.
    slaveIds : ArrayLike
        Indices of the slave actuators.
    masterIds : ArrayLike
        Indices of the master actuators.
    borderIds : ArrayLike, optional
        Indices of the border actuators. Required for minimum-RMS-force slaving.

    Returns
    -------
    Q : MatrixLike
        Slave-to-master matrix.
    """
    K = _get_decomposed_ffwd(slaveIds, masterIds, FF, borderIds, method=method)

    if method == "zero-force":
        Kss = K["ss"]
        Ksm = K["sm"]
        Q = -_xp.linalg.pinv(Kss) @ Ksm
    elif method == "minimum-rms":
        Q0 = -_xp.linalg.pinv(K["bs"].T @ K["bs"] + K["ss"].T @ K["ss"])
        Q = Q0 @ (K["bs"].T @ K["bi"] + K["ss"].T @ K["si"])
    else:
        raise _oe.ValueError(
            f"Unknown slaving method '{method}'. Available methods are 'zero-force' and 'minimum-rms'."
        )

    return Q


def _get_decomposed_ffwd(
    slaveIds: _ot.ArrayLike,
    masterIds: _ot.ArrayLike,
    ffwd: _ot.MatrixLike,
    borderIds: _ot.ArrayLike,
    method: str = "zero-force",
) -> dict[str, _ot.MatrixLike]:
    """
    Decomposes the feed-forward matrix into its sub-matrices
    according to the slave, border and master actuator indices.

    Parameters
    ----------
    slaveIds : ArrayLike
        Indices of the slave actuators.
    masterIds : ArrayLike
        Indices of the master actuators.
    ffwd : MatrixLike
        Feed-Forward matrix of the deformable mirror.
    borderIds : ArrayLike, optional
        Indices of the border actuators.
    method : str, optional
        Method to compute the master-to-slave matrix. Options are:
        - 'zero-force' : zero-force slaving
        - 'minimum-rms' : minimum-RMS-force slaving.

        Defaults to 'zero-force'.

    Returns
    -------
    K : dict[str, MatrixLike]
        Sub-matrix K_mm.
    """
    K = _xp.asarray(ffwd)

    if method == "zero-force":
        nK = {
            "mm": K[_xp.ix_(masterIds, masterIds)],
            "ms": K[_xp.ix_(masterIds, slaveIds)],
            "sm": K[_xp.ix_(slaveIds, masterIds)],
            "ss": K[_xp.ix_(slaveIds, slaveIds)],
        }
    elif method == "minimum-rms":
        nK = {
            "ii": K[_xp.ix_(masterIds, masterIds)],
            "ib": K[_xp.ix_(masterIds, borderIds)],
            "is": K[_xp.ix_(masterIds, slaveIds)],
            "bi": K[_xp.ix_(borderIds, masterIds)],
            "bb": K[_xp.ix_(borderIds, borderIds)],
            "bs": K[_xp.ix_(borderIds, slaveIds)],
            "si": K[_xp.ix_(slaveIds, masterIds)],
            "sb": K[_xp.ix_(slaveIds, borderIds)],
            "ss": K[_xp.ix_(slaveIds, slaveIds)],
        }

    return nK
