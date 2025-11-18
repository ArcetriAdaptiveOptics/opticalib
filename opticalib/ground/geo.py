"""
This module contains functions for geometric operations on images.

Autor(s)
---------
- Runa Briguglio : created Mar 2020
- Federico Miceli : added funcitonality on 2022
- Pietro Ferraiuolo : polished on 2024

"""

import numpy as _np
from .. import typings as _t
from skimage.draw import disk as _disk
from skimage.measure import CircleModel as _CM


def qpupil_circle(image, pixel_dir=0):
    """
    Function for...
    Created by Federico
    NOTA: la funzione usa come standard la direzione y per determinare la dimensione dei pixel

    pixel_dir: int
        indicates which direction to use for counting the number of pixels in the image.
        Y direction as standard
    """
    aa = _np.shape(image)
    imagePixels = aa[pixel_dir]  # standard dir y
    circ = _CM()
    cnt = _find_img_borders(image, imagePixels)
    circ.estimate(cnt)
    xc, yc, radius = _np.array(circ.params, dtype=int)
    maskedd = _np.zeros((imagePixels, imagePixels), dtype=_np.uint8)
    rr, cc = _disk((xc, yc), int(radius))
    maskedd[rr, cc] = 1
    idx = _np.where(maskedd == 1)
    ss = _np.shape(maskedd)
    x = _np.arange(ss[0]).astype(float)
    x = _np.transpose(_np.tile(x, [ss[1], 1]))
    y = _np.arange(ss[1]).astype(float)
    y = _np.tile(y, [ss[0], 1])
    xx = x
    yy = y
    xx = xx - xc
    xx = xx / radius
    yy = yy - yc
    yy = yy / radius
    return xx, yy


def qpupil(mask, xx=None, yy=None, nocircle=0):
    """
    Function for....
    created by Runa

    Parameters
    ----------
    mask: numpy array

    Returns
    ------
    x0:
    y0:
    r:
    xx: numpy array
        grid of coordinates of the same size as input mask
    yy: numpy array
        grid of coordinates of the same size as input mask
    """
    idx = _np.where(mask == 1)
    ss = _np.shape(mask)
    x = _np.arange(ss[0]).astype(float)
    x = _np.transpose(_np.tile(x, [ss[1], 1]))
    y = _np.arange(ss[1]).astype(float)
    y = _np.tile(y, [ss[0], 1])
    xx = x
    yy = y
    x0 = 0
    y0 = 0
    r = 0
    if nocircle == 0:
        maxv = max(xx[idx])
        minv = min(xx[idx])
        r1 = (maxv - minv) / 2
        x0 = r1 + minv
        xx = xx - (minv + maxv) / 2
        xx = xx / ((maxv - minv) / 2)
        mx = [minv, maxv]
        maxv = max(yy[idx])
        minv = min(yy[idx])
        r2 = (maxv - minv) / 2
        y0 = r2 + minv
        yy = yy - (minv + maxv) / 2
        yy = yy / ((maxv - minv) / 2)
        r = _np.mean([r1, r2])
        my = [minv, maxv]
    return x0, y0, r, xx, yy



def draw_mask(img, cx, cy, r, out=0):
    """Function to create circular mask
    Created by Runa

    Parameters
    ----------
    img: numpy array
        image to mask
    cx: int [pixel]
        center x of the mask
    cy: int [pixel]
        center y of the mask
    r: int [pixel]
        radius of the mask

    Returns
    -------
    img1: numpy array
        start image mask whit circular new mask
    """
    ss = _np.shape(img)
    x = _np.arange(ss[0])
    x = _np.transpose(_np.tile(x, [ss[1], 1]))
    y = _np.arange(ss[1])
    y = _np.tile(y, [ss[0], 1])
    x = x - cx
    y = y - cy
    nr = _np.size(r)
    if nr == 2:
        rr = x * x / r[0] ** 2 + y * y / r[1] ** 2
        r1 = 1
    else:
        rr = x * x + y * y
        r1 = r**2
    pp = _np.where(rr < r1)
    img1 = img.copy()
    if out == 1:
        img1[pp] = 0
    else:
        img1[pp] = 1
    # plt.imshow(img1)
    return img1


# from arte.types.mask import CircularMask

# CircularMask


def draw_circular_mask(img_or_mask: _t.ImageData, radius: float) -> _t.ImageData:
    """
    Function to create a circular mask that fits the input image or mask.

    Parameters
    ----------
    img_or_mask: np.ndarray or np.ma.maskedArray
        Input image or mask to fit the circular mask.

    Returns
    -------
    circular_mask: np.ma.maskedArray
        Circular mask fitting the input image or mask.
    """
    if hasattr(img_or_mask, "mask"):
        img_data = img_or_mask.data
        img_mask = img_or_mask.mask
    else:
        img_data = img_mask = img_or_mask
    coords = qpupil(-1 * img_mask + 1)
    circular_mask = draw_mask(img_data * 0, coords[0], coords[1], radius, out=0)
    return circular_mask


def _find_img_borders(image, imagePixels):
    """
    Function for...
    Created by Federico
    """
    x = image
    val = []
    i = 0
    while i < imagePixels:
        a = x[i, :]
        aa = _np.where(a.mask.astype(int) == 0)
        q = _np.asarray(aa)
        if q.size < 2:
            i = i + 1
        else:
            val.append(_np.array([[i, q[0, 0]], [i, q[0, q.size - 1]]]))
            i = i + 1
    cut = _np.concatenate(val)
    return cut
