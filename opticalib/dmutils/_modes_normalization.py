import numpy as np
from ..analyzer import image_processing as _ip
from ..core import _types as _ot

def resolve(normalization: _ot.Number|bool|str|callable) -> _ot.Number|callable:
    """
    Resolve the normalization factor based on the input type.

    Parameters
    ----------
    normalization : number or bool or str
        The normalization parameter to resolve.
    T : int, optional
        The number of frames in the push-pull template sequence, 
        used for the default normalization method.

    Returns
    -------
    norm, flag : number or callable and str
        The resolved normalization factor or function and the corresponding flag
        for the type of normalization used.
    """
    # Flat scaling factor applied to the data
    if isinstance(normalization, (int, float)):
        return normalization, 'flat'
    # Default normalization method based on the push-pull template sequence length
    # needs T and A as input when called
    if isinstance(normalization, bool):
        return lambda T, A: 2 * max(T - 1, 1) * A, 'default'
    # String-based normalization method, resolved via a helper function
    if isinstance(normalization, str):
        return get_str_normalization(normalization), 'str'
    # custom callable 
    if callable(normalization):
        return normalization, 'callable'

    raise ValueError(f"Unsupported normalization type: {type(normalization)}")


def get_str_normalization(norm: str):
    """
    Function used to retrieve the correct function defined by the string normalization
    parameter.
    
    Parameters
    ----------
    norm : str
        The normalization parameter passed down as a string, which specifies the
        normalization method to be used.
    
    Return
    ------
    func : callable
        The function corresponding to the specified normalization method.
    """
    match norm:
        case "shlike":
            return lambda dat: dat / np.where(dat > 0, dat, 1)
        case "total":
            return lambda dat: dat / np.sum(np.where(dat > 0, dat, 0))
        case _:
            raise ValueError(f"Unknown normalization method: {norm}")