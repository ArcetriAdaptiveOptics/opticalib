import numpy as _np
from opticalib import typings as _ot


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
