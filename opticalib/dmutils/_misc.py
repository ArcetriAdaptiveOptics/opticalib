import numpy as _np
from ..ground import modal_decomposer as _md
from ..core import _types as _ot


def make_modal_base(
    rm: _ot.MatrixLike,
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
        surf = fit.make_surface([modes_list[i]])
        masked_data = surf[~mask]
        mat[i, :] = masked_data

    MB = (mat @ rm).T
    return MB


__all__ = ["make_modal_base"]
