import numpy as _np
from opticalib import typings as _ot
from photutils.centroids import centroid_2dg
from opticalib.ground import geometry as _geo

center_act = 313

def findFrameCoord(imglist, actlist, actcoord):
    """
    returns the position of given actuators from a list of frames
    """
    pos = []
    for i in imglist:
        pos.append(findActuator(i))
    pos = (_np.array(pos)).T

    frameCenter = marker_general_remap(
        actcoord[:, actlist], pos, actcoord[:, (center_act, center_act)]
    )
    # the last variable has been vectorized (by adding a second element) don't know why but so it works
    frameCenter = frameCenter[:, 0]
    return frameCenter


def findActuator(image: _ot.ImageData) -> _np.ndarray:
    """
    Finds the coordinates of an actuator, given the image with the Influence
    Function masked around the actuators.
    
    Parameters
    ----------
    image: _ot.ImageData
        Image where the actuator is to be searched
        
    Return
    -------
    pos: np.ndarray
        Coordinates of the actuator
    """
    imgw = extractPeak(image, radius=50)
    pos = centroid_2dg(imgw)
    return pos


def extractPeak(img: _ot.ImageData, radius: int) -> _ot.ImageData:
    """
    Extract a circular area around the peak in the image
    
    Parameters
    ----------
    img: ImageData
        Input image
    radius: int
        Radius of the circular area to extract
        
    Returns
    -------
    imgout: ImageData
        Image with only the circular area around the peak, rest masked
    """
    yp, xp = _np.where(img == _np.max(abs(img)))
    mm = _geo.draw_circular_pupil(img.shape, radius, center=[yp, xp])
    imgout = _np.ma.masked_array(img, mask=mm)
    return imgout


def marker_general_remap(cghf, ottf, pos2t):
    """
    transforms the pos2t coordinates, using the cghf and ottf coordinates to create the trnasformation
    """
    polycoeff = fit_trasformation_parameter(cghf, ottf)
    base_cgh = _expandbase(pos2t[0, :], pos2t[1, :])
    cghf_tra = _np.transpose(_np.dot(_np.transpose(base_cgh), _np.transpose(polycoeff)))
    return cghf_tra

def fit_trasformation_parameter(cghf, ottf, forder=10):
    """
    Fits the transformation parameters between cghf and ottf coordinates.

    Parameters
    ----------
    cghf: np.ndarray
        Coordinates in the cghf system
    ottf: np.ndarray
        Coordinates in the ottf system
    forder: int, optional
        Order of the polynomial fit (default is 10)

    Returns
    -------
    polycoeff: np.ndarray
        Coefficients of the polynomial transformation
    """
    
    import scipy.linalg as sl
    
    base_cgh = _expandbase(cghf[0, :], cghf[1, :], forder=forder)
    base_ott = _expandbase(ottf[0, :], ottf[1, :], forder=forder)

    base_cgh_plus = sl.pinv(base_cgh)
    polycoeff = _np.matmul(base_ott, base_cgh_plus)
    return polycoeff

def _expandbase(cx: _ot.ArrayLike, cy: _ot.ArrayLike, forder: int = 10):
    """
    Expands the base functions for polynomial fitting of given order.
    
    Parameters
    ----------
    cx: ArrayLike
        x-coordinates
    cy: ArrayLike
        y-coordinates
    forder: int, optional
        Order of the polynomial fit (default is 10)
        
    Returns
    -------
    zz: ArrayLike
        Expanded base functions for fitting
    """
    print("Fitting order %i" % forder)
    if forder == 3:
        zz = _np.stack((cx, cy, _np.ones(cx.size)), axis=0)
    if forder == 6:
        zz = _np.stack((cx**2, cy**2, cx * cy, cx, cy, _np.ones(cx.size)), axis=0)
    if forder == 5:
        zz = _np.stack((cx**2, cy**2, cx, cy, _np.ones(cx.size)), axis=0)
    if forder == 10:
        zz = _np.stack(
            (
                cx**3,
                cy**3,
                cx**2 * cy,
                cy**2 * cx,
                cx**2,
                cy**2,
                cx * cy,
                cx,
                cy,
                _np.ones(cx.size),
            ),
            axis=0,
        )

    return zz