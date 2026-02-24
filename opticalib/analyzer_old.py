"""
ANALYZER module
===============
2020-2024

In this module are present all useful functions for data analysis.

Author(s)
---------
- Runa Briguglio: runa.briguglio@inaf.it
- Pietro Ferraiuolo: pietro.ferraiuolo@inaf.it

Description
-----------

"""

from ast import Not
import os as _os
import xupy as _xp
import numpy as _np
import jdcal as _jdcal
import matplotlib.pyplot as _plt
from . import typings as _ot
from .ground import modal_decomposer as zern
from .ground import osutils as osu
from .core import root as _foldname, fitsarray as _fa
from .ground.geo import qpupil as _qpupil
from scipy import stats as _stats, fft as _fft, ndimage as _ndimage

_OPDSER = _foldname.OPD_SERIES_ROOT_FOLDER


# TODO: TO REMOVE -> equal to `intoFullFrame`
def frame2ottFrame(
    img: _ot.ImageData, croppar: list[int], flipOffset: bool = True
) -> _ot.ImageData:
    """
    Reconstructs a full 2048x2048 image from a cropped image and its cropping parameters.

    Parameters
    ----------
    img : _ot.ImageData
        Cropped image data.
    croppar : list[int]
        Cropping parameters [x, y, width, height].
    flipOffset : bool, optional
        If True, flips the cropping offset. The default is True.

    Returns
    -------
    fullimg : _ot.ImageData
        Reconstructed full image.
    """
    off = croppar.copy()
    if flipOffset is True:
        off = _np.flip(croppar)
        print(f"Offset values flipped: {str(off)}")
    nfullpix = _np.array([2048, 2048])
    fullimg = _np.zeros(nfullpix)
    fullmask = _np.ones(nfullpix)
    offx = off[0]
    offy = off[1]
    sx = _np.shape(img)[0]  # croppar[2]
    sy = _np.shape(img)[1]  # croppar[3]
    fullimg[offx : offx + sx, offy : offy + sy] = img.data
    fullmask[offx : offx + sx, offy : offy + sy] = img.mask
    fullimg = _np.ma.masked_array(fullimg, fullmask)
    return fullimg


# TODO
def readTemperatures(tn: str):
    """
    Reads temperature data from a FITS file associated with a tracking number.

    Parameters
    ----------
    tn : str
        Tracking number of the frames to process.

    Returns
    -------
    temperatures : _ot.ArrayLike
        Array of temperature values for each frame.

    """
    fold = osu.findTracknum(tn, complete_path=True)
    fname = _os.path.join(fold, "temperature.fits")
    temperatures = osu.load_fits(fname)
    return temperatures


# TODO
def readZernike(tn: str):
    """
    Reads Zernike coefficients from a FITS file associated with a tracking number.

    Parameters
    ----------
    tn : str
        Tracking number of the frames to process.

    Returns
    -------
    zernikes : _ot.ArrayLike
        Array of Zernike coefficients for each frame.
    """
    fold = osu.findTracknum(tn, complete_path=True)
    fname = _os.path.join(fold, "zernike.fits")
    zernikes = osu.load_fits(fname)
    return zernikes


# TODO
def zernikePlot(
    mylist: _ot.CubeData | list[_ot.ImageData], zmodes: _ot.ArrayLike = None
) -> _ot.ArrayLike:
    """
    Computes Zernike coefficients for each frame in a cube or a list of images.

    Parameters
    ----------
    mylist : _ot.CubeData | list[_ot.ImageData]
        Input image data.
    zmodes : _ot.ArrayLike, optional
        Zernike modes to compute. The default is _np.array(range(1, 11)).

    Returns
    -------
    zcoeff : _ot.ArrayLike
        Zernike coefficients for each frame.
    """
    zfit = zern.ZernikeFitter()
    if zmodes is None:
        zmodes = _np.array(range(1, 11))
    if isinstance(mylist, list):
        imgcube = createCube(mylist)
    elif isinstance(mylist, _np.ma.MaskedArray):
        imgcube = mylist
    zlist = []
    for i in range(imgcube.shape[-1]):
        coeff, _ = zfit.fit(imgcube[:, :, i], zmodes)
        zlist.append(coeff)
    zcoeff = _np.array(zlist)
    zcoeff = zcoeff.T
    return zcoeff
