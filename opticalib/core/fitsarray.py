import numpy as _np
from astropy.io import fits as _fits
import xupy as _xp

from .. import typings as _ot


def _prepare_header(
    header: dict[str, int | float | str | bool] | _fits.Header | None,
) -> _fits.Header | None:
    if header is None:
        return {}
    if isinstance(header, _fits.Header):
        return header.copy()
    if isinstance(header, dict):
        h = _fits.Header()
        for key, value in header.items():
            h[key] = value
        return h
    raise TypeError("header must be a dict, astropy.io.fits.Header, or None")


class FitsArray(_np.ndarray):
    """
    Numpy ndarray subclass that keeps a FITS header alongside the numeric data.

    Use this class whenever your data have no mask but you still want to preserve
    ancillary FITS metadata such as exposure time or instrument keywords. The
    array behaves like a plain ``numpy.ndarray`` in numerical operations while
    carrying the header around every time a new view is created.
    """

    def __new__(
        cls,
        data: _ot.ArrayLike,
        *,
        header: _ot.Optional[dict[str, _ot.Any] | _ot.Header] = None,
        dtype: _ot.Optional[_ot.DTypeLike] = None
    ):
        """
        Parameters
        ----------
        data :
            Any array-like object accepted by ``numpy.array``.
        header :
            FITS header to associate with the array (dict or ``astropy.io.fits.Header``).
        dtype :
            Target dtype, forwarded to ``numpy.array``.
        copy :
            Whether to force a copy of the data buffer.
        """
        obj = _np.array(data, dtype=dtype).view(cls)
        obj.header = _prepare_header(header)
        return obj

    def __array_finalize__(self, obj):
        """Ensure the header is propagated to numpy-created views."""
        if obj is None:
            return
        self.header = getattr(obj, "header", None)

    def writeto(self, filename: str, overwrite: bool = False):
        """
        Saves the array to a FITS file.

        Parameters
        ----------
        filename : str
            Path to the FITS file.
        overwrite : bool, optional
            Whether to overwrite an existing file. Default is False.
        """
        data = _np.asanyarray(self, dtype=_np.float32)
        _fits.writeto(filename, data, header=self.header, overwrite=overwrite)

    @classmethod
    def fromFits(cls, filename: str) -> "FitsArray":
        from opticalib.ground.osutils import load_fits

        data = load_fits(filename, return_header=True)
        return cls(data, header=data.header)

class FitsMaskedArray(_np.ma.MaskedArray):
    """
    MaskedArray subclass that keeps an associated FITS header.

    This version mirrors ``numpy.ma.MaskedArray`` behaviour, so masks propagate
    through numpy operations while FITS metadata are preserved via subclassing.
    """

    def __new__(
        cls,
        data: _ot.ArrayLike,
        *,
        mask: _ot.Optional[_ot.MaskData] = None,
        header: _ot.Optional[dict[str, _ot.Any] | _ot.Header] = None,
        dtype: _ot.Optional[_ot.DTypeLike] = None,
        fill_value: _ot.Optional[_ot.Number] = None,
        keep_mask: bool = True,
    ):
        """
        Parameters
        ----------
        data, mask, dtype, copy, fill_value, keep_mask :
            Same semantics as ``numpy.ma.MaskedArray``.
        header :
            FITS header (dict or ``astropy.io.fits.Header``) stored with the data.
        """
        obj = _np.ma.MaskedArray.__new__(
            cls,
            data=data,
            mask=mask,
            dtype=dtype,
            keep_mask=keep_mask,
            fill_value=fill_value,
            subok=False,
        )
        obj.header = _prepare_header(header)
        return obj

    def __array_finalize__(self, obj):
        """
        Called automatically whenever numpy creates a view of the subclass,
        ensuring header metadata follow the data buffer.
        """
        if obj is None:
            return
        super().__array_finalize__(obj)
        self.header = getattr(obj, "header", {})

    def _update_from(self, obj):
        """
        Copies attributes from obj to self, called during view operations.
        This is the standard way to propagate custom attributes in MaskedArray.
        """
        super()._update_from(obj)
        self.header = getattr(obj, "header", {})

    def writeto(self, filename: str, overwrite: bool = False):
        """
        Saves the array to a FITS file.

        Parameters
        ----------
        filename : str
            Path to the FITS file.
        overwrite : bool, optional
            Whether to overwrite an existing file. Default is False.
        """
        data = _np.asanyarray(self.data, dtype=_np.float32)
        _fits.writeto(filename, data, header=self.header, overwrite=overwrite)
        _fits.append(filename, self.mask.astype(_np.uint8))

    @classmethod
    def fromFits(cls, filename: str) -> "FitsMaskedArray":
        from opticalib.ground.osutils import load_fits

        data = load_fits(filename, return_header=True)
        return cls(data, header=data.header)

class FitsArrayGpu(_xp.ndarray):
    """
    Cupy ndarray subclass that keeps a FITS header alongside the numeric data.

    Use this class whenever your data have no mask but you still want to preserve
    ancillary FITS metadata such as exposure time or instrument keywords. The
    array behaves like a plain ``cupy.ndarray`` in numerical operations while
    carrying the header around every time a new view is created.
    """

    def __new__(
        cls,
        data: _ot.ArrayLike,
        *,
        header: _ot.Optional[dict[str, _ot.Any] | _ot.Header] = None,
        dtype: _ot.Optional[_ot.DTypeLike] = None,
    ):
        """
        Parameters
        ----------
        data :
            Any array-like object accepted by ``numpy.array``.
        header :
            FITS header to associate with the array (dict or ``astropy.io.fits.Header``).
        dtype :
            Target dtype, forwarded to ``cupy.array``.
        """
        obj = _xp.array(data, dtype=dtype).view(cls)
        obj.header = _prepare_header(header)
        return obj

    def __array_finalize__(self, obj):
        """Ensure the header is propagated to numpy-created views."""
        if obj is None:
            return
        self.header = getattr(obj, "header", None)

    def writeto(self, filename: str, overwrite: bool = False):
        """
        Saves the array to a FITS file.

        Parameters
        ----------
        filename : str
            Path to the FITS file.
        overwrite : bool, optional
            Whether to overwrite an existing file. Default is False.
        """
        data = _xp.asnumpy(self)
        # force float32 dtype on save
        if data.dtype != _np.float32:
            data = _np.asanyarray(data, dtype=_np.float32)
        _fits.writeto(filename, data, header=self.header, overwrite=overwrite)

    @classmethod
    def fromFits(cls, filename: str) -> "FitsArrayGpu":
        from opticalib.ground.osutils import load_fits

        data = load_fits(filename, return_header=True)
        return cls(_xp.asarray(data), header=data.header)

class FitsMaskedArrayGpu(_xp.ma.masked_array):
    """
    GPU MaskedArray subclass that keeps an associated FITS header.

    This version mirrors ``xupy.ma.MaskedArray`` behaviour.
    """
    
    def __init__(self, **kwargs:dict[str,_ot.Any]):
        header = kwargs.pop("header", None)
        super().__init__(**kwargs)
        self.header = _prepare_header(header)
    
    def writeto(self, filename: str, overwrite: bool = False):
        """
        Saves the array to a FITS file.

        Parameters
        ----------
        filename : str
            Path to the FITS file.
        overwrite : bool, optional
            Whether to overwrite an existing file. Default is False.
        """
        data = self.asmarray(dtype=_np.float32)
        _fits.writeto(filename, data.data, header=self.header, overwrite=overwrite)
        _fits.append(filename, data.mask.astype(_np.uint8))

    @classmethod
    def fromFits(cls, filename: str) -> "FitsArrayGpu":
        from opticalib.ground.osutils import load_fits

        data = load_fits(filename, return_header=True)
        return cls(data=data, header=data.header)
    
    def asmarray(self, **kwargs: dict[str,_ot.Any]) -> "FitsMaskedArray":
        """
        Returns the data as a cupy ndarray, optionally casting to a specified dtype.

        Parameters
        ----------
        dtype : DtypeLike, optional
            Target data type for the returned array.

        Returns
        -------
        array : cupy.ndarray
            The data as a cupy ndarray.
        """
        ma = super().asmarray(**kwargs)
        arr = FitsMaskedArray(ma, header=self.header)
        return arr


def fits_array(data: _ot.ArrayLike, **kwargs: dict[str,_ot.Any]) -> FitsArray | FitsMaskedArray:
    """
    Wrapper aound numpy's array and masked array classes that keeps an associated FITS header.

    Parameters
    ----------
    data : ArrayLike
        Data to be wrapped.
    mask : MaskData, optional
        Mask to be applied to the data.
    header : dict[str, Any] | fits.Header, optional
        Header to be associated with the data.
    dtype : DtypeLike, optional
        Data type to be used for the data.
    copy : bool, optional
        Whether to copy the data.
    fill_value : Number, optional
        Fill value to be used for the data.
    keep_mask : bool, optional
        Whether to keep the mask.
        
    Returns
    -------
    array : FitsArray | FitsMaskedArray
        The wrapped array.

    Examples
    --------
    >>> data = np.array([[1, 2], [3, 4]])
    >>> header = {"TEST": "value", "EXPTIME": 1.5, "BITPIX": 16}
    >>> array = fits_array(data, header=header)
    >>> print(array.header)
    >>> print(array.data)
    [[1 2]
     [3 4]]
    """
    mask = kwargs.get("mask", None)
    
    # Convert lists to numpy arrays
    if isinstance(data, list):
        # default on CPU
        data = _np.asarray(data)

    # Check if data has a .data attribute (masked arrays, FitsArray, etc.)
    # If not, use the data itself to check type
    if hasattr(data, "data"):
        data_type_str = str(type(data.data))
    else:
        # For plain arrays, check the type of the array itself
        data_type_str = str(type(data))

    # Determine array type based on underlying data type
    if 'cupy' in data_type_str:
        if hasattr(data, "mask") or mask is not None:
            array_type = FitsMaskedArrayGpu
        else:
            array_type = FitsArrayGpu
    elif any(['numpy' in data_type_str, 'memoryview' in data_type_str]):
        if hasattr(data, "mask") or mask is not None:
            array_type = FitsMaskedArray
        else:
            array_type = FitsArray
    else:
        # Default to plain FitsArray for unknown types
        if hasattr(data, "mask") or mask is not None:
            array_type = FitsMaskedArray
        else:
            array_type = FitsArray

    return array_type(data=data, **kwargs)