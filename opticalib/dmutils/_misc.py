import numpy as _np
from ..ground import modal_decomposer as _md
from .. import typings as _ot


def make_modal_base(
    RM: _ot.MatrixLike,
    modes: int | list[int],
    mask: _ot.MaskData,
    basis: str = "zernike",
) -> _ot.MatrixLike:
    """
    Make a modal base from a reconstruction matrix.

    Parameters
    ----------
    RM : MatrixLike
        The reconstruction matrix.
    modes : int or list of int
        The number of modes to compute, or the list of mode indices to compute.
    mask : MaskData
        The mask where to define the modal base.
    basis : str, optional
        The type of modal base to compute. Options are:

        - 'zernike'
        - 'kl' for Karhunen-Loève modes.

        Default is 'zernike'.

    Returns
    -------
    array_like
        The modal base.

    """
    if isinstance(modes, int):
        modes_list = list(range(1, modes + 1))

    if not all([i != 0 for i in modes_list]):
        raise ValueError("Index 0 not permitted.")

    match basis:
        case "zernike":
            if isinstance(modes, int):
                modes_list = list(range(1, modes + 1))
            fit = _md.ZernikeFitter(mask)
        case "kl":
            if not isinstance(modes, int):
                raise ValueError("For KL modes, modes must be an integer.")
            modes_list = list(range(0, modes))
            fit = _md.KLFitter(modes, mask)
        case _:
            raise ValueError(f"Unknown basis: {basis}")

    validpix = _np.sum(mask == 0)

    nmodes = len(modes_list)
    mat = _np.zeros((nmodes, validpix))
    for i in range(nmodes):
        surf = fit.makeSurface([modes_list[i]])
        masked_data = surf[~mask]
        mat[i, :] = masked_data

    MB = (mat @ RM).T
    return MB


def get_buffer_mean_values(
    position: _ot.ArrayLike,
    position_error: _ot.ArrayLike,
    k: int = 12,
    min_cmd: float = 1e-9,
):
    """
    Get mean position values for position and position error buffers

    Parameters
    ----------
    position : np.ndarray
        Position values array (nActs, nSamples)
    position_error : np.ndarray
        Position error values array (nActs, nSamples)
    k : int, optional
        Number of samples to wait before averaging each command. Defaults to 12
    min_cmd : float, optional
        Minimum command change to detect a new command buffer. Defaults to 1 nm

    Returns
    -------
    posMeans : np.ndarray
        Mean position values for each command buffer [nActuators, nCommands]
    cmdIds : np.ndarray
        Indices of samples corresponding to each command [nActuators, nCommands*cmdLen]
    """
    # Detect command jumps
    command = position + position_error
    delta_command = command[:, 1:] - command[:, :-1]
    delta_bool = abs(delta_command) > min_cmd  # 1 nm command threshold

    nActs, nSteps = _np.shape(command)
    cmd_ids = []

    for i in range(nActs):
        ids = _np.arange(nSteps)
        ids = ids[1:][delta_bool[i, :]]
        filt_ids = []
        for i in range(len(ids) - 1):
            if ids[i + 1] - ids[i] > 1:
                filt_ids.append(ids[i])
        filt_ids.append(ids[-1])
        cmd_ids.append(filt_ids)

    cmd_ids = _np.array(cmd_ids, dtype=int)
    cmd_ids = cmd_ids[:, 3:]  # remove trigger

    minCmdLen = _np.min(cmd_ids[:, 1:] - cmd_ids[:, :-1])
    startIds = cmd_ids.copy()
    nCmds = _np.shape(startIds)[1]

    cmdIds = _np.tile(_np.arange(minCmdLen), (nActs, nCmds))
    cmdIds += _np.repeat(startIds, (minCmdLen,)).reshape([nActs, -1])
    posMeans = _np.zeros((nActs, nCmds))

    chunk_size = 10  # 10 acts at a time
    posMeans = _np.zeros((nActs, nCmds))
    for i in range(0, nActs, chunk_size):
        end_i = min(i + chunk_size, nActs)
        cmd_indices = cmdIds[i:end_i].reshape(-1, nCmds, minCmdLen)[:, :, k:]
        act_idx = _np.arange(end_i - i)[:, None, None]
        posMeans[i:end_i] = _np.mean(position[i:end_i][act_idx, cmd_indices], axis=2)

    return posMeans, cmdIds


__all__ = ["make_modal_base", "get_buffer_mean_values"]
