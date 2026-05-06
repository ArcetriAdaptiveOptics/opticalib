"""
Module: images_processing
=========================

Author(s)
---------
- Pietro Ferraiuolo

Description
-----------

This module provides utilities to process masked optical phase maps and
image cubes, including piston unwrapping, push-pull reduction, rebinning,
Zernike-mode removal, Fourier-domain filtering, and power spectral density
analysis.

The functions are designed to operate on Opticalib typing aliases and
``fitsarray``-compatible masked arrays used across the analysis pipeline.
"""

import xupy as _xp
import numpy as _np
from .. import typings as _ot
from ..ground import osutils as osu
from ..core import fitsarray as _fa
from scipy import stats as _stats, fft as _fft, ndimage as _ndimage
from skimage.transform import resize as _resize


def frame(idx: int, mylist: list[_ot.ImageData] | _ot.CubeData) -> _ot.ImageData:
    """
    Returns a single frame from a list of files or from a cube.

    Parameters
    ----------
    idx : int
        Index of the frame to retrieve.
    mylist : list or cube
        1) list of strings with the paths to the files to read;
        2) list of ImageData objects;
        3) cube of images (3D masked array).

    Returns
    -------
    img : _ot.ImageData
        The requested image frame.
    """
    if isinstance(mylist, list):
        if idx >= len(mylist):
            raise IndexError("Index out of range")
        if isinstance(mylist[idx], str):
            img = osu.read_phasemap(mylist[idx])
        elif _ot.isinstance_(mylist[idx], "ImageData"):
            img = mylist[idx]
    else:
        img = mylist[:, :, idx]
    return img


def piston_unwrap(
    piston_vec: _ot.ArrayLike,
    commanded_piston_vec: _ot.ArrayLike = None,
    wavelength: float = None,
    period: int = 2,
) -> _ot.ArrayLike:
    """
    Unwraps a piston vector by correcting for jumps that exceed a specified threshold.

    Parameters
    ----------
    piston_vec : _ot.ArrayLike
        Input piston vector to be unwrapped.
    commanded_piston_vec : _ot.ArrayLike, optional
        Commanded piston vector. The default is None.
    wavelength : float, optional
        Wavelength of the piston measurements. If None, the default is 632.8 nm
        (He-Ne laser).
    period : int, optional
        Period for unwrapping. The default is 2.

    Returns
    -------
    unwrapped : _ot.ArrayLike
        Unwrapped piston vector.
    """
    if wavelength is None:
        print(
            "Wavelength not specified, using default value of 632.8 nm\nWARNING! Pass input `piston_vec` in nm"
        )
        wavelength = 632.8  # nm

    # checking wavelength and the piston vector units are consistent
    if wavelength < 1:  # assuming input in m
        wavelength *= 1e9  # convert to nm

    if _np.max(piston_vec) < 1:  # assuming input in nm
        piston_vec *= 1e9  # convert to nm

    pwl = wavelength / period

    if commanded_piston_vec is None:
        reconstructed_piston = _np.unwrap(piston_vec, discont=wavelength, period=pwl)
    else:
        k = _np.round((commanded_piston_vec - piston_vec) / pwl)
        reconstructed_piston = piston_vec + k * pwl

    return reconstructed_piston


def pushPullReductionAlgorithm(
    imagelist: list[_ot.ImageData] | _ot.CubeData,
    template: _ot.ArrayLike,
    normalization: _ot.Optional[float | int] = None
) -> _ot.ImageData:
    """
    Performs the basic operation of processing PushPull data.

    Parameters
    ----------
    imagelist : list of ImageData | CubeData
        List of images for the PushPull acquisition, organized according to the template.
    template: int | ArrayLike
        Template for the PushPull acquisition.
    normalization : float | int, optional
        Normalization factor for the final image. If None, the normalization factor
        is set to the template length minus one.

    Returns
    -------
    image: masked_array
        Final processed mode's image.
    """
    template = _np.asarray(template)
    n_images = len(imagelist)

    # Template weights computation
    w = _xp.asarray(
        template.astype(_np.result_type(template, imagelist[0].data), copy=True),
        dtype=_xp.float,
    )
    if n_images > 2:
        w[1:-1] *= 2.0
    # OR-reduce all masks once
    master_mask = _np.logical_or.reduce([ima.mask for ima in imagelist])
    # Compute weighted sum over realizations on raw data
    stack = _xp.stack(
        [_xp.asarray(ima.data, dtype=_xp.float) for ima in imagelist],
        axis=0,
        dtype=_xp.float,
    )  # (n, H, W)
    image = _xp.asnumpy(_xp.tensordot(w, stack, axes=(0, 0)))  # (H, W)

    if normalization is None:
        norm_factor = _np.max(((template.shape[0] - 1), 1))
    else:
        norm_factor = normalization

    image = _np.ma.masked_array(image, mask=master_mask) / norm_factor
    return image


def createCube(fl_or_il: list[str], register: bool = False) -> _ot.CubeData:
    """
    Creates a cube of images from an images file list

    Parameters
    ----------
    fl_or_il : list of str
        Either:
        - the list of image file paths;
        - a list of ImageData.
    register : int or tuple, optional
        If not False, and int or a tuple of int must be passed as value, and
        the registration algorithm is performed on the images before stacking them
        into the cube. Default is False.

    Returns
    -------
    cube : ndarray
        Data cube containing the images/frames stacked, with shape (npx, npy, nframes).
    """
    # check it is a list
    if not isinstance(fl_or_il, list):
        raise TypeError("filelist must be a list of strings or images")

    # check if it is composed of file paths to load
    if all([isinstance(item, str) for item in fl_or_il]):
        fl_or_il = [osu.read_phasemap(f) for f in fl_or_il]

        # Is the list now full of images?
        if not any(
            [
                all([_ot.isinstance_(item, "ImageData") for item in fl_or_il]),
                all([_ot.isinstance_(item, "MatrixLike") for item in fl_or_il]),
            ]
        ):
            raise TypeError("Data different from `images` loaded. Check filelist.")

    # finally check if it is a list of ImageData
    elif not all([_ot.isinstance_(item, "ImageData") for item in fl_or_il]):
        try:
            cube = _fa.fits_array(_np.ma.dstack(fl_or_il))
            return cube
        except Exception as e:
            raise TypeError(
                "filelist must be either a list of strings or ImageData"
            ) from e

    if register:
        print("Registration Not implemented yet!")

    header = {}
    for item in fl_or_il:
        if hasattr(item, "header"):
            header.update(item.header)

    cube = _fa.fits_array(_np.ma.dstack(fl_or_il), header=header)

    return cube


def removeZernikeFromCube(
    cube: _ot.CubeData, zmodes: _ot.ArrayLike = None, mode="global"
) -> _ot.CubeData:
    """
    Removes Zernike modes from each frame in a cube of images.

    Parameters
    ----------
    cube : ndarray
        Data cube containing the images/frames stacked.
    zmodes : ndarray, optional
        Zernike modes to remove. If None, the first 3 modes are removed.
    mode : str, optional
        Mode of Zernike removal, either 'global' or 'local'. The default is 'global'.

    Returns
    -------
    newCube : ndarray
        Cube with Zernike modes removed from each frame.
    """
    from tqdm import tqdm
    import sys as _sys
    from io import StringIO as _sIO
    from ..ground.modal_decomposer import ZernikeFitter

    zfit = ZernikeFitter()
    if zmodes is None:
        zmodes = _np.array(range(1, 4))

    if isinstance(cube, (_fa.FitsMaskedArray, _fa.FitsArray)):
        zmodes_str = "[" + ",".join(map(str, zmodes)) + "]"
        cube.header["FILTERED"] = (True, "has zernike removed")
        cube.header["ZREMOVED"] = (zmodes_str, "zernike modes removed")

    newCube = _fa.fits_array(_np.ma.empty_like(cube), header=cube.header)
    for i in tqdm(
        range(cube.shape[-1]),
        desc=f"Removing Z[{', '.join(map(str, zmodes))}]...",
        unit="image",
        ncols=80,
    ):
        old_stdout = _sys.stdout
        _sys.stdout = _sIO()
        zfit.removeZernike(cube[:, :, i], zmodes, mode=mode)
        _sys.stdout = old_stdout
        newCube[:, :, i] = zfit.removeZernike(cube[:, :, i], zmodes, mode=mode)
    return newCube


def modeRebinner(
    img: _ot.ImageData,
    rebin: int,
    method: str = "averaging",
    anti_aliasing: bool = True,
    preserve_flux: bool = False,
    mode: str = "reflect",
    cval: float = 0.0,
) -> _ot.ArrayLike:
    """
    Image rebinner

    Rebins a masked array image by a factor rebin.

    Parameters
    ----------
    img : masked_array
        Image to rebin.
    rebin : int
        Rebinning factor.
    method : str, optional
        Rebinning method. Supported values are:
        'averaging'/'mean', 'sum', 'median',
        'sampling'/'nearest', 'bilinear', 'bicubic'.
        The default is 'averaging'.
    anti_aliasing : bool, optional
        If True, apply anti-aliasing when using interpolation methods during
        downsampling. Default is False.
    preserve_flux : bool, optional
        If True, scale interpolated images to preserve integrated flux.
        Default is False.
    mode : str, optional
        Pixel extrapolation mode for interpolation methods.
        Default is 'reflect'.
    cval : float, optional
        Constant value used when ``mode='constant'``. Default is 0.0.

    Returns
    -------
    newImg : masked_array
        Rebinned image.
    """
    shape = img.shape
    new_shape = (shape[0] // rebin, shape[1] // rebin)
    newImg = rebin2DArray(
        img,
        new_shape,
        method=method,
        anti_aliasing=anti_aliasing,
        preserve_flux=preserve_flux,
        mode=mode,
        cval=cval,
    )
    return newImg


def cubeRebinner(
    cube: _ot.CubeData,
    rebin: int,
    method: str = "averaging",
    anti_aliasing: bool = False,
    preserve_flux: bool = False,
    mode: str = "reflect",
    cval: float = 0.0,
) -> _ot.CubeData:
    """
    Cube rebinner

    Parameters
    ----------
    cube : ndarray
        Cube to rebin.
    rebin : int
        Rebinning factor.
    method : str, optional
        Rebinning method. Supported values are:
        'averaging'/'mean', 'sum', 'median',
        'sampling'/'nearest', 'bilinear', 'bicubic'.
        The default is 'averaging'.
    anti_aliasing : bool, optional
        If True, apply anti-aliasing when using interpolation methods during
        downsampling. Default is True.
    preserve_flux : bool, optional
        If True, scale interpolated images to preserve integrated flux.
        Default is False.
    mode : str, optional
        Pixel extrapolation mode for interpolation methods.
        Default is 'reflect'.
    cval : float, optional
        Constant value used when ``mode='constant'``. Default is 0.0.

    Returns
    -------
    newCube : ndarray
        Rebinned cube.
    """
    if hasattr(cube, "header"):
        header = cube.header.copy()
    else:
        header = {}

    newCube = []
    for i in range(cube.shape[-1]):
        newCube.append(
            modeRebinner(
                cube[:, :, i],
                rebin,
                method=method,
                anti_aliasing=anti_aliasing,
                preserve_flux=preserve_flux,
                mode=mode,
                cval=cval,
            )
        )
    return _fa.fits_array(_np.ma.dstack(newCube), header=header)


def comp_filtered_image(
    imgin: _ot.ImageData,
    verbose: bool = False,
    disp: bool = False,
    d: int = 1,
    freq2filter: _ot.Optional[tuple[float, float]] = None,
):
    """


    Parameters
    ----------
    imgin : _ot.ImageData
        Input image data.
    verbose : bool, optional
        If True, print detailed information. The default is False.
    disp : bool, optional
        If True, display intermediate results. The default is False.
    d : int, optional
        Spacing between samples. The default is 1.
    freq2filter : tuple[float, float], optional
        Frequency range to filter. The default is None.

    Returns
    -------
    imgout : _ot.ImageData
        Filtered image data.
    """
    img = imgin.copy()
    sx = (_np.shape(img))[0]
    mask = _np.invert(img.mask)
    img[mask == 0] = 0
    norm = "ortho"
    tf2d = _fft.fft2(img.data, norm=norm)
    kfreq = _fft.fftfreq(sx, d=d)  # frequency in cicles
    kfreq2D = _np.meshgrid(kfreq, kfreq)  # frequency grid x,y
    knrm = _np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)  # freq. grid distance
    # TODO optional mask to get the circle and not the square
    fmask1 = 1.0 * (knrm > _np.max(kfreq))
    if freq2filter is None:
        fmin = -1
        fmax = _np.max(kfreq)
    else:
        fmin, fmax = freq2filter
    fmask2 = 1.0 * (knrm > fmax)
    fmask3 = 1.0 * (knrm < fmin)
    fmask = (fmask1 + fmask2 + fmask3) > 0
    tf2d_filtered = tf2d.copy()
    tf2d_filtered[fmask] = 0
    imgf = _fft.ifft2(tf2d_filtered, norm=norm)
    imgout = _np.ma.masked_array(_np.real(imgf), mask=imgin.mask)
    if disp:
        import matplotlib.pyplot as plt

        imgs = [imgin, imgout, knrm, fmask1, fmask2, fmask3, fmask]
        titles = [
            "Initial image",
            "Filtered image",
            "Frequency",
            "Fmask1",
            "Fmask2",
            "Fmask3",
            "Fmask",
        ]
        for i in range(len(imgs)):
            plt.figure()
            plt.imshow(imgs[i])
            plt.title(titles[i])
            plt.colorbar()
        plt.show()
    if verbose:
        e1 = _np.sqrt(_np.sum(img[mask] ** 2) / _np.sum(mask)) * 1e9
        e2 = _np.sqrt(_np.sum(imgout[mask] ** 2) / _np.sum(mask)) * 1e9
        e3 = _np.sqrt(_np.sum(_np.abs(tf2d) ** 2) / _np.sum(mask)) * 1e9
        e4 = _np.sqrt(_np.sum(_np.abs(tf2d_filtered) ** 2) / _np.sum(mask)) * 1e9
        print(f"RMS image [nm]            {e1:.2f}")
        print(f"RMS image filtered [nm]   {e2:.2f}")
        print(f"RMS spectrum              {e3:.2f}")
        print(f"RMS spectrum filtered     {e4:.2f}")
    return imgout


def compute_psd(
    imgin: _ot.ImageData,
    nbins: _ot.Optional[int] = None,
    norm: str = "backward",
    verbose: bool = False,
    show: bool = False,
    d: int = 1,
    sigma: _ot.Optional[float] = None,
    crop: bool = True,
):
    """
    Computes the power spectral density (PSD) of a 2D image.

    Parameters
    ----------
    imgin : _ot.ImageData
        Input image data.
    nbins : _ot.Optional[int], optional
        Number of bins for the power spectrum. The default is None.
    norm : str, optional
        Normalization mode for the FFT. The default is "backward".
    verbose : bool, optional
        If True, print detailed information. The default is False.
    show : bool, optional
        If True, display intermediate results. The default is False.
    d : int, optional
        Spacing between samples. The default is 1.
    sigma : _ot.Optional[float], optional
        Standard deviation for Gaussian smoothing. The default is None.
    crop : bool, optional
        If True, crop the image to the circular region. The default is True.

    Returns
    -------
    fout : _ot.ArrayLike
        Frequency bins.
    Aout : _ot.ArrayLike
        Amplitude spectrum.

    """
    from opticalib.ground.roi import imgCut

    img = imgCut(imgin) if crop else imgin.copy()
    sx = (_np.shape(img))[0]
    if nbins is None:
        nbins = sx // 2
    img = img - _np.mean(img)
    mask = _np.invert(img.mask)
    img[mask == 0] = 0
    if sigma is not None:
        img = _ndimage.fourier_gaussian(img, sigma=sigma)
    tf2d = _fft.fft2(img, norm=norm)
    tf2d[0, 0] = 0
    tf2d_power_spectrum = _np.abs(tf2d) ** 2
    kfreq = _fft.fftfreq(sx, d=d)  # frequency in cicles
    kfreq2D = _np.meshgrid(kfreq, kfreq)  # freq. grid
    knrm = _np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)  # freq. grid distance
    fmask = knrm < _np.max(kfreq)
    knrm = knrm[fmask].flatten()
    fourier_amplitudes = tf2d_power_spectrum[fmask].flatten()
    Abins, _, _ = _stats.binned_statistic(
        knrm, fourier_amplitudes, statistic="sum", bins=nbins
    )
    e1 = _np.sum(img[mask] ** 2 / _np.sum(mask))
    e2 = _np.sum(Abins) / _np.sum(mask)
    ediff = _np.abs(e2 - e1) / e1
    fout = kfreq[0 : sx // 2]
    Aout = Abins / _np.sum(mask)
    if verbose:
        print(f"Sampling          {d:}")
        print(f"Energy signal     {e1}")
        print(f"Energy spectrum   {e2}")
        print(f"Energy difference {ediff}")
        print(kfreq[0:4])
        print(kfreq[-4:])
    else:
        print(f"RMS from spectrum {_np.sqrt(e2)}")
        print(f"RMS [nm]          {(_np.std(img[mask])*1e9):.2f}")
    if show is True:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(fout[1:], Aout[1:] * fout[1:], ".")
        plt.yscale("log")
        plt.xscale("log")
        plt.title("Power spectrum")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude [A^2]")
        plt.grid()
        plt.show()

    return fout, Aout


def integrate_psd(y: _ot.ArrayLike, img: _ot.ImageData) -> _ot.ArrayLike:
    """
    Integrates the power spectral density (PSD) over the image.

    Parameters
    ----------
    y : _ot.ArrayLike
        Power spectral density values.
    img : _ot.ImageData
        Input image data.

    Returns
    -------
    _ot.ArrayLike
        Integrated PSD values.
    """
    nn = _np.sqrt(_np.sum(-1 * img.mask + 1))
    yint = _np.sqrt(_np.cumsum(y)) / nn
    return yint


def _normalize_rebin_method(method: str) -> str:
    """Map legacy and alias method names to canonical names."""
    method_aliases = {
        "averaging": "mean",
        "sampling": "nearest",
        "mean": "mean",
        "sum": "sum",
        "median": "median",
        "nearest": "nearest",
        "bilinear": "bilinear",
        "bicubic": "bicubic",
    }
    norm_method = method_aliases.get(method.lower())
    if norm_method is None:
        valid = ", ".join(sorted(method_aliases))
        raise ValueError(f"Unsupported rebin method '{method}'. Valid: {valid}")
    return norm_method


def _block_reduce_2d(
    arr: _np.ma.MaskedArray, new_shape: tuple[int, int], method: str
) -> _np.ma.MaskedArray:
    """Downsample by integer factors using block statistics."""
    m, n = new_shape
    M, N = arr.shape
    if m > M or n > N:
        raise ValueError(
            "Block reduction methods support downsampling only. "
            "Use an interpolation method to upsample."
        )
    if M % m != 0 or N % n != 0:
        raise ValueError(
            "Block reduction requires integer factors between input and "
            "output shapes."
        )

    fy = M // m
    fx = N // n
    block = _np.ma.asarray(arr).reshape(m, fy, n, fx)

    if method == "mean":
        return _np.ma.mean(block, axis=(1, 3))
    if method == "sum":
        return _np.ma.sum(block, axis=(1, 3))

    # median
    block_flat = _np.ma.transpose(block, (0, 2, 1, 3)).reshape(m, n, fy * fx)
    return _np.ma.median(block_flat, axis=2)


def _interpolate_2d(
    arr: _np.ma.MaskedArray,
    new_shape: tuple[int, int],
    method: str,
    anti_aliasing: bool,
    preserve_flux: bool,
    mode: str,
    cval: float,
) -> _np.ma.MaskedArray:
    """Resize 2D data with interpolation and mask-aware normalization."""
    interpolation_order = {
        "nearest": 0,
        "bilinear": 1,
        "bicubic": 3,
    }[method]

    M, N = arr.shape
    m, n = new_shape
    data = _np.ma.getdata(arr).astype(float, copy=False)
    mask = _np.ma.getmaskarray(arr)
    valid = (~mask).astype(float)

    data_filled = _np.where(mask, 0.0, data)
    resized_data = _resize(
        data_filled,
        (m, n),
        order=interpolation_order,
        mode=mode,
        cval=cval,
        anti_aliasing=anti_aliasing,
        preserve_range=True,
    )
    resized_valid = _resize(
        valid,
        (m, n),
        order=0 if interpolation_order == 0 else 1,
        mode="constant",
        cval=0.0,
        anti_aliasing=False,
        preserve_range=True,
    )

    eps = _np.finfo(float).eps
    out_mask = resized_valid <= eps
    out_data = _np.zeros((m, n), dtype=float)
    out_data[~out_mask] = resized_data[~out_mask] / resized_valid[~out_mask]

    if preserve_flux and (m > 0 and n > 0):
        out_data *= (M * N) / (m * n)

    return _np.ma.masked_array(out_data, mask=out_mask)


def rebin2DArray(
    a: _ot.ArrayLike,
    new_shape: tuple[int, int],
    method: str = "mean",
    anti_aliasing: bool = True,
    preserve_flux: bool = False,
    mode: str = "reflect",
    cval: float = 0.0,
) -> _ot.ArrayLike:
    """
    Rebin a 2D array with selectable aggregation or interpolation methods.

    Parameters
    ----------
    a : _ot.ArrayLike
        Input 2D array or masked array.
    new_shape : tuple[int, int]
        Target shape ``(rows, cols)``.
    method : str, optional
        Rebinning method. Supported values are:
        'averaging'/'mean', 'sum', 'median',
        'sampling'/'nearest', 'bilinear', 'bicubic'.
        The default is 'mean'.
    anti_aliasing : bool, optional
        If True, apply anti-aliasing for interpolation-based downsampling.
        Default is True.
    preserve_flux : bool, optional
        If True, scale interpolation output to preserve integrated flux.
        Default is False.
    mode : str, optional
        Pixel extrapolation mode used by interpolation methods.
        Default is 'reflect'.
    cval : float, optional
        Constant value used when ``mode='constant'``. Default is 0.0.

    Returns
    -------
    _ot.ArrayLike
        Rebinned 2D masked array.

    Raises
    ------
    ValueError
        If the input is not 2D, output shape is invalid, or method is unknown.
    """
    arr = _np.ma.asarray(a)
    if arr.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    try:
        m, n = map(int, new_shape)
    except Exception as e:
        raise ValueError("new_shape must be a 2-element integer sequence") from e

    if m <= 0 or n <= 0:
        raise ValueError("new_shape values must be positive")

    if arr.shape == (m, n):
        return arr.copy()

    norm_method = _normalize_rebin_method(method)
    if norm_method in {"mean", "sum", "median"}:
        return _block_reduce_2d(arr, (m, n), method=norm_method)

    return _interpolate_2d(
        arr,
        (m, n),
        method=norm_method,
        anti_aliasing=anti_aliasing,
        preserve_flux=preserve_flux,
        mode=mode,
        cval=cval,
    )


__all__ = [
    "frame",
    "piston_unwrap",
    "pushPullReductionAlgorithm",
    "createCube",
    "removeZernikeFromCube",
    "rebin2DArray",
    "modeRebinner",
    "cubeRebinner",
    "comp_filtered_image",
    "compute_psd",
    "integrate_psd",
]
