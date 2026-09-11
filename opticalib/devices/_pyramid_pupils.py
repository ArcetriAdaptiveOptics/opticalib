"""
Pyramid WFS pupil geometry helpers.

Ported from ArcetriLAB ``pupEstimator.PupilDesc`` (no ArcetriLAB dependency).
FITS save/load layout is compatible with ArcetriLAB pupil files so existing
reconstructor pupdata can be reused.

ArcetriLAB ``indpup`` order (list of 4 flat index arrays)::

    [TL, BL, BR, TR]

Specula ``PyrSlopec`` / ``PupData`` pupil columns A,B,C,D map as::

    A = TR, B = TL, C = BL, D = BR

i.e. Specula order indices into ArcetriLAB ``indpup`` are ``[3, 0, 1, 2]``.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import optimize


# Specula A,B,C,D <- ArcetriLAB [TL, BL, BR, TR]
_SPECULA_FROM_ARCETRI = (3, 0, 1, 2)


def _params_to_verts(side, alpha, off_x, off_y):
    sdiag = np.sqrt(2.0) * side / 2.0
    return np.asarray(
        [
            np.cos(np.pi / 4.0 + alpha) * sdiag + off_x,
            np.sin(np.pi / 4.0 + alpha) * sdiag + off_y,
            np.cos(3 * np.pi / 4.0 + alpha) * sdiag + off_x,
            np.sin(3 * np.pi / 4.0 + alpha) * sdiag + off_y,
            np.cos(5 * np.pi / 4.0 + alpha) * sdiag + off_x,
            np.sin(5 * np.pi / 4.0 + alpha) * sdiag + off_y,
            np.cos(7 * np.pi / 4.0 + alpha) * sdiag + off_x,
            np.sin(7 * np.pi / 4.0 + alpha) * sdiag + off_y,
        ]
    )


def _objective_function(xx, *args):
    return np.sqrt(np.sum(np.square(np.asarray(args) - _params_to_verts(*xx))) / 4.0)


def _find_points(img1, th=1000, verbose=False):
    """Detect 4 quadrant pupil centroids and fit a square geometry.

    If ``th`` leaves empty quadrants, retry with an adaptive threshold
    based on the frame intensity (handles short exposures / scaled frames).
    """
    img_orig = np.asarray(img1, dtype=float)
    thresholds = [float(th)]
    finite = img_orig[np.isfinite(img_orig)]
    if finite.size:
        peak = float(np.max(finite))
        med = float(np.median(finite))
        # Adaptive candidates: fraction of peak, then above background
        for cand in (0.25 * peak, 0.1 * peak, med + 0.5 * (peak - med)):
            if cand > 0 and cand not in thresholds:
                thresholds.append(cand)

    last_sizes = None
    for thr in thresholds:
        mask = np.where(img_orig > thr, 1, 0)
        mx = int(mask.shape[0] / 2)
        my = int(mask.shape[1] / 2)
        sizes = [
            np.count_nonzero(mask[0:mx, 0:my]),
            np.count_nonzero(mask[0:mx, my : 2 * my]),
            np.count_nonzero(mask[mx : 2 * mx, my : 2 * my]),
            np.count_nonzero(mask[mx : 2 * mx, 0:my]),
        ]
        last_sizes = sizes
        if all(s > 0 for s in sizes):
            if verbose or thr != float(th):
                print(f"Pupil detection using threshold={thr:g} (sizes={sizes})")
            return _find_points_at_threshold(img_orig, thr, verbose=verbose)

    raise RuntimeError(
        f"Pupil detection failed: empty quadrant(s). Tried thresholds={thresholds}. "
        f"last sizes={last_sizes}. Check ROI / exposure / illumination."
    )


def _find_points_at_threshold(img_orig, th, verbose=False):
    """Core ArcetriLAB quadrant centroid + square fit at a fixed threshold."""
    img1 = np.where(img_orig > th, 1, 0)
    mx = int(img1.shape[0] / 2)
    my = int(img1.shape[1] / 2)
    sizes = [
        np.count_nonzero(img1[0:mx, 0:my]),
        np.count_nonzero(img1[0:mx, my : 2 * my]),
        np.count_nonzero(img1[mx : 2 * mx, my : 2 * my]),
        np.count_nonzero(img1[mx : 2 * mx, 0:my]),
    ]
    y_center1, x_center1 = np.argwhere(img1[0:mx, 0:my] >= 1).sum(0) / sizes[0]
    y_center2, x_center2 = np.argwhere(img1[0:mx, my : 2 * my] >= 1).sum(0) / sizes[1]
    y_center3, x_center3 = (
        np.argwhere(img1[mx : 2 * mx, my : 2 * my] >= 1).sum(0) / sizes[2]
    )
    y_center4, x_center4 = np.argwhere(img1[mx : 2 * mx, 0:my] >= 1).sum(0) / sizes[3]
    # ncoords order: BL, BR, TR, TL  (ArcetriLAB convention)
    coords = [
        [x_center4, y_center4 + mx],
        [x_center3 + my, y_center3 + mx],
        [x_center2 + my, y_center2],
        [x_center1, y_center1],
    ]
    intensity = [
        (img_orig * img1)[:mx, :my].sum(),
        (img_orig * img1)[:mx, my:].sum(),
        (img_orig * img1)[mx:, my:].sum(),
        (img_orig * img1)[mx:, :my].sum(),
    ]
    mask = img1.astype(np.uint8)
    ncoords = np.asarray(coords)
    x0 = np.asarray([1000.0, 0.0, 1000.0, 1000.0])
    pps = (
        ncoords[2][0],
        ncoords[2][1],
        ncoords[1][0],
        ncoords[1][1],
        ncoords[0][0],
        ncoords[0][1],
        ncoords[3][0],
        ncoords[3][1],
    )
    res = optimize.minimize(_objective_function, x0, args=pps, method="Nelder-Mead")
    if verbose:
        print(res)
    avg_diam = np.sqrt(np.average(sizes) / np.pi) * 2
    return res.x, ncoords, sizes, avg_diam, mask, intensity


def compute_indpup(mask, ncoords):
    """
    Build four aligned flat-index pupil arrays (ArcetriLAB order).

    Returns
    -------
    list[np.ndarray]
        ``[TL, BL, BR, TR]`` each of length ``n_subap``.
    """
    ss = mask.shape
    quadrant = np.zeros(ss, dtype=np.int32)
    quadrant[: ss[0] // 2, : ss[1] // 2] = 1

    c = ncoords
    mask1 = mask * quadrant
    mask2 = (
        np.roll(
            mask,
            (
                int(np.round(c[3][1] - c[0][1])),
                int(np.round(c[3][0] - c[0][0])),
            ),
            axis=(0, 1),
        )
        * quadrant
    )
    mask3 = (
        np.roll(
            mask,
            (
                int(np.round(c[3][1] - c[1][1])),
                int(np.round(c[3][0] - c[1][0])),
            ),
            axis=(0, 1),
        )
        * quadrant
    )
    mask4 = (
        np.roll(
            mask,
            (
                int(np.round(c[3][1] - c[2][1])),
                int(np.round(c[3][0] - c[2][0])),
            ),
            axis=(0, 1),
        )
        * quadrant
    )

    x, y = np.where(mask1 + mask2 + mask3 + mask4 == 4)
    if x.size == 0:
        raise RuntimeError("Pupil index intersection is empty; check detection.")

    pup1 = np.zeros(mask.shape, dtype=bool)
    pup1[x, y] = True
    pup2 = np.roll(
        pup1,
        (
            int(-np.round(c[3][1] - c[0][1])),
            int(-np.round(c[3][0] - c[0][0])),
        ),
        axis=(0, 1),
    )
    pup3 = np.roll(
        pup1,
        (
            int(-np.round(c[3][1] - c[1][1])),
            int(-np.round(c[3][0] - c[1][0])),
        ),
        axis=(0, 1),
    )
    pup4 = np.roll(
        pup1,
        (
            int(-np.round(c[3][1] - c[2][1])),
            int(-np.round(c[3][0] - c[2][0])),
        ),
        axis=(0, 1),
    )
    return [
        np.ravel_multi_index(np.where(pup), pup.shape)
        for pup in (pup1, pup2, pup3, pup4)
    ]


def indpup_to_specula(indpup_arcetri):
    """
    Reorder ArcetriLAB ``[TL, BL, BR, TR]`` to Specula ``[A, B, C, D]``.

    Specula order is ``A=TR, B=TL, C=BL, D=BR``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_subap, 4)``.
    """
    ordered = [indpup_arcetri[i] for i in _SPECULA_FROM_ARCETRI]
    return np.column_stack(ordered).astype(np.int64)


class PyramidPupilData:
    """
    Four-pupil geometry for a Pyramid WFS.

    Compatible with ArcetriLAB ``PupilDesc`` FITS files (version 1).
    """

    version = 1

    def __init__(self, vvo, ncoords, rx, sizes, avg_d, mask, intensity):
        self.vvo = np.asarray(vvo, dtype=float)
        self.ncoords = np.asarray(ncoords, dtype=float)
        self.rx = np.asarray(rx, dtype=float)
        self.sizes = np.asarray(sizes, dtype=float)
        self.avg_d = float(avg_d)
        self.mask = np.asarray(mask).astype(bool)
        self.intensity = np.asarray(intensity, dtype=float)
        self.name = ""
        self._indpup = None
        self._slopes_display_info = None

    @property
    def separation(self):
        return float(self.rx[0])

    @property
    def angle(self):
        return float(self.rx[1])

    @property
    def indpup(self):
        """ArcetriLAB order: ``[TL, BL, BR, TR]``."""
        if self._indpup is None:
            self._indpup = compute_indpup(self.mask, self.ncoords)
        return self._indpup

    @property
    def ind_pup_specula(self):
        """Specula ``PupData`` layout ``(n_subap, 4)`` as A,B,C,D."""
        return indpup_to_specula(self.indpup)

    @property
    def n_subap(self):
        return int(len(self.indpup[0]))

    @property
    def framesize(self):
        return tuple(int(x) for x in self.mask.shape)

    @classmethod
    def from_img(cls, img, threshold=1000, verbose=False):
        """Detect pupils from a science frame."""
        rx, ncoords, sizes, avg_d, mask, intensity = _find_points(
            np.asarray(img, dtype=float), th=threshold, verbose=verbose
        )
        vvo = _params_to_verts(rx[0], rx[1], rx[2], rx[3])
        return cls(vvo, ncoords, rx, sizes, avg_d, mask, intensity)

    def save(self, filename, overwrite=False):
        """
        Save pupil geometry to a FITS file (ArcetriLAB-compatible layout).

        HDU0 holds ``vstack(indpup)`` plus AVGDIAM/SEPARAT/ANGLE/VERSION
        headers; subsequent HDUs store vvo, ncoords, rx, mask, sizes, intensity.
        """
        path = Path(filename)
        self.name = path.stem
        hdulist = fits.HDUList()
        hdulist.append(fits.ImageHDU(np.vstack(self.indpup)))
        hdulist.append(fits.ImageHDU(self.vvo))
        hdulist.append(fits.ImageHDU(self.ncoords))
        hdulist.append(fits.ImageHDU(self.rx))
        hdulist.append(fits.ImageHDU(self.mask.astype(np.uint8)))
        hdulist.append(fits.ImageHDU(self.sizes))
        hdulist.append(fits.ImageHDU(self.intensity))
        hdulist[0].header["AVGDIAM"] = self.avg_d
        hdulist[0].header["SEPARAT"] = self.separation
        hdulist[0].header["ANGLE"] = self.angle
        hdulist[0].header["VERSION"] = self.version
        hdulist.writeto(path, overwrite=overwrite)

    @classmethod
    def load(cls, filename):
        """Load pupil geometry from an ArcetriLAB-compatible FITS file."""
        with fits.open(filename) as hdulist:
            version = hdulist[0].header["VERSION"]
            if version != 1:
                raise ValueError(f"Cannot load pupil file with version {version}")
            avg_d = hdulist[0].header["AVGDIAM"]
            vvo = hdulist[1].data
            ncoords = hdulist[2].data
            rx = hdulist[3].data
            mask = hdulist[4].data.astype(bool)
            sizes = hdulist[5].data
            intensity = hdulist[6].data if len(hdulist) > 6 else np.zeros(4)
            pupil = cls(vvo, ncoords, rx, sizes, avg_d, mask, intensity)
            pupil.name = Path(filename).stem
            return pupil

    @property
    def slopes_display_info(self):
        """Indices for packing Specula 1D slopes into a side-by-side 2D map.

        Layout matches ArcetriLAB ``PupilDesc.slopes_display_info``: one pupil
        mask (from ``indpup[0]``) cropped to its bounding box, then
        ``hstack(mask, mask)`` so Sx occupies the left half and Sy the right.
        """
        if self._slopes_display_info is not None:
            return self._slopes_display_info
        pup1 = np.zeros(self.mask.shape)
        pup1.flat[self.indpup[0]] = 1
        x, y = np.where(pup1)
        x1, x2, y1, y2 = x.min(), x.max(), y.min(), y.max()
        x = x - x1
        y = y - y1
        f1 = np.zeros((x2 - x1 + 1, y2 - y1 + 1))
        f1[x, y] = 1
        ff = np.hstack((f1, f1 * 2))
        Info = namedtuple("Info", "xindex yindex shape")
        self._slopes_display_info = Info(
            xindex=np.ravel_multi_index(np.where(ff == 1), ff.shape),
            yindex=np.ravel_multi_index(np.where(ff == 2), ff.shape),
            shape=ff.shape,
        )
        return self._slopes_display_info

    def slopes2d(self, slopes, useNaN=True):
        """Remap Specula 1D slopes into a 2D (side-by-side Sx|Sy) map.

        Parameters
        ----------
        slopes : array-like
            Length ``2 * n_subap`` vector, or a 2-D stack of such vectors
            (shape ``(n, 2 * n_subap)``).
        useNaN : bool, optional
            If True (default), non-pupil pixels are NaN and the result is a
            masked array. If False, they are zeros (plain ndarray).

        Returns
        -------
        np.ndarray | np.ma.MaskedArray
            Shape ``(h, 2w)`` for a single vector, or ``(n, h, 2w)`` for a stack.
        """
        slopes = np.asarray(slopes, dtype=np.float64)
        if slopes.ndim == 2:
            mapped = [self.slopes2d(s, useNaN=useNaN) for s in slopes]
            stacked = np.stack([np.ma.filled(m, np.nan) for m in mapped], axis=0)
            if useNaN:
                return np.ma.masked_invalid(stacked)
            return stacked

        info = self.slopes_display_info
        nsubaps = self.n_subap
        if slopes.ndim != 1 or slopes.size != 2 * nsubaps:
            raise ValueError(
                f"Expected 1D slopes of length {2 * nsubaps}, got shape {slopes.shape}"
            )
        frame = np.zeros(info.shape, dtype=np.float64)
        if useNaN:
            frame.fill(np.nan)
        frame.flat[info.xindex] = slopes[:nsubaps]
        frame.flat[info.yindex] = slopes[nsubaps : nsubaps * 2]
        if useNaN:
            return np.ma.masked_invalid(frame)
        return frame

    def slopes1d(self, frame):
        """Inverse of ``slopes2d``: 2D map(s) back to Specula 1D slopes.

        Parameters
        ----------
        frame : array-like
            Shape ``(h, 2w)`` or a stack ``(n, h, 2w)``. Masked arrays are
            filled with 0.0 before gathering.

        Returns
        -------
        np.ndarray
            Length ``2 * n_subap`` vector, or shape ``(n, 2 * n_subap)``.
        """
        info = self.slopes_display_info
        arr = np.ma.filled(np.asanyarray(frame, dtype=np.float64), 0.0)
        if arr.ndim == 3:
            return np.stack([self.slopes1d(f) for f in arr], axis=0)
        if arr.ndim != 2 or tuple(arr.shape) != tuple(info.shape):
            raise ValueError(
                f"Expected 2D frame of shape {info.shape}, got {arr.shape}"
            )
        flat = arr.ravel()
        sx = flat[info.xindex]
        sy = flat[info.yindex]
        return np.concatenate([sx, sy])

    def remap_pupils(self, flat_pixels):
        """
        Extract four intensity maps of shape ``framesize`` from a flat frame.

        Returns
        -------
        np.ndarray
            Array of shape ``(4, H, W)`` in ArcetriLAB order TL,BL,BR,TR.
        """
        flat = np.asarray(flat_pixels, dtype=np.float64).ravel()
        h, w = self.framesize
        out = np.zeros((4, h, w), dtype=np.float64)
        for i, idx in enumerate(self.indpup):
            out[i].ravel()[idx] = flat[idx]
        return out
