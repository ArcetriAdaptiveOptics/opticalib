"""
Module: timeseries
==================

Author(s)
---------
- Runa Briguglio
- Pietro Ferraiuolo

Description
-----------

This module provides functions to handle and analyze time series data in the context of Optical Imaging.
It includes utilities for frame extraction, averaging frames, saving and loading averaged images,
computing running differences, calculating running means, and estimating structure functions from
time series data.

Functions:
- frame: Retrieve a single frame from a list or cube.
- averageFrames: Average multiple frames to create an averaged image.
- saveAverage: Save the averaged image to a file.
- openAverage: Load an averaged image from a file.
- runningDiff: Compute the running difference between frames with optional zernike removal.
- timevec: Generate a time vector for a set of frames based on their timestamps.
- runningMean: Calculate the running mean of a 1D array.
- structfunc: Compute the structure function for a given time series.


This module is part of the OPTICALIB library and relies on other modules within the package for data handling and processing.
"""

import os as _os
import numpy as _np
from .. import typings as _ot
from ..ground import osutils as osu, modal_decomposer as _md
from . import images_processing as _ip
from ..core.root import OPD_SERIES_ROOT_FOLDER as _OPDSER


def averageFrames(
    tn_or_fl: str | list[_ot.ImageData] | _ot.CubeData,
    first: int = 0,
    last: int = -1,
    file_selector: list[int] | None = None,
    thresh: bool = False,
) -> _ot.ImageData:
    """
    Perform the average of a list of images, retrievable through a tracking
    number.

    Parameters
    ----------
    tn_or_fl : str | list[ImageData] | CubeData
        The data Tracking Number, the list of images or the cube of images to
        average.
    first : int, optional
        Index number of the first file to consider. Defaults to first item
        in the list.
    last : int, optional
        Index number of the last file to consider. Defaults to last item in
        the list.
    file_selector : list, optional
        A list of integers, representing the specific files to load. If None,
        the range (first->last) is considered.
    thresh : bool, optional
        If True, apply a threshold to the averaging process. The default is False.

    Returns
    -------
    aveimg : ImageData
        Final image of averaged frames.

    """
    s = slice(first, last) if last != -1 else slice(first, None)

    if osu.is_tn(tn_or_fl):
        fl = osu.getFileList(tn_or_fl, fold=_OPDSER.split("/")[-1], key="20")
        fl = fl[s] if file_selector is None else fl[file_selector]
        imcube = _ip.createCube(fl)
    elif _ot.isinstance_(tn_or_fl, "CubeData"):
        imcube = (
            tn_or_fl[:, :, s]
            if file_selector is None
            else tn_or_fl[:, :, file_selector]
        )
    elif isinstance(tn_or_fl, list) and all(
        _ot.isinstance_(item, "ImageData") for item in tn_or_fl
    ):
        fl = (
            tn_or_fl[s]
            if file_selector is None
            else [tn_or_fl[i] for i in file_selector]
        )
        imcube = _ip.createCube(fl)

    if thresh is False:
        aveimg = _np.ma.mean(imcube, axis=2)
    else:
        ## TODO: test new implementation

        valid_frames = ~imcube.mask  # Boolean array of valid data
        n_valid = valid_frames.sum(axis=2)  # Count valid frames per pixel

        # Sum only valid data
        img_sum = _np.ma.sum(imcube, axis=2).filled(0)

        # Avoid division by zero
        with _np.errstate(divide="ignore", invalid="ignore"):
            img = img_sum / n_valid
            img = _np.where(n_valid > 0, img, 0)

        # Create mask
        mmask = n_valid == 0
        aveimg = _np.ma.masked_array(img, mask=mmask)

    return aveimg


def saveAverage(
    tn: str,
    average_img: _ot.ImageData = None,
    overwrite: bool = False,
    **kwargs: dict[str, _ot.Any],
):
    """
    Saves an averaged frame, in the same folder as the original frames. If no
    averaged image is passed as argument, it will create a new average for the
    specified tracking number, and additional arguments, the same as ''averageFrames''
    can be specified.

    Parameters
    ----------
    tn : str
        Tracking number where to save the average frame file. If average_img is
        None, it is the tracking number of the data that will be averaged
    average_img : ndarray, optional
        Result average image of multiple frames. If it's None, it will be generated
        from data found in the tracking number folder. Additional arguments can
        be passed on
    **kwargs : additional optional arguments
        The same arguments as `averageFrames`, to specify the averaging method.
        - first : int, optional
            Index number of the first file to consider. If None, the first file in
            the list is considered.
        - last : int, optional
            Index number of the last file to consider. If None, the last file in
            list is considered.
        - file_selector : list of ints, optional
            A list of integers, representing the specific files to load. If None,
            the range (first->last) is considered.
        - thresh : bool, optional
            DESCRIPTION. The default is None.
    """
    fname = _os.path.join(_OPDSER, tn, "average.fits")
    if _os.path.isfile(fname):
        print(f"Average '{fname}' already exists")
        return
    else:
        if average_img is None:
            first = kwargs.get("first", None)
            last = kwargs.get("last", None)
            fsel = kwargs.get("file_selector", None)
            thresh = kwargs.get("tresh", False)
            average_img = averageFrames(
                tn, first=first, last=last, file_selector=fsel, thresh=thresh
            )
    osu.save_fits(fname, average_img, overwrite=overwrite)
    print(f"Saved average at '{fname}'")


def openAverage(tn: str):
    """
    Loads an averaged frame from an 'average.fits' file, found inside the input
    tracking number

    Parameters
    ----------
    tn : str
        Tracking number of the averaged frame.

    Returns
    -------
    image : ndarray
        Averaged image.

    Raises
    ------
    FileNotFoundError
        Raised if the file does not exist.
    """
    fname = _os.path.join(_OPDSER, tn, "average.fits")
    try:
        image = osu.load_fits(fname)
        print(f"Average loaded: '{fname}'")
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Average file '{fname}' does not exist!") from err
    return image


def runningDiff(
    tn_or_fl: str | list[str] | list[_ot.ImageData] | _ot.CubeData,
    gap: int = 2,
    remove_zernikes: bool | list[int] = False,
    stds_out: bool = True,
) -> tuple[list[_ot.ImageData], _ot.ArrayLike] | list[_ot.ImageData]:
    """
    Computes the running difference of the frames in a given tracking number.

    Parameters
    ----------
    tn_or_fl : str or list[str] or list[ImageData] or CubeData
        It can either be:
        - a tracking number where the frames to process are;
        - a list of strings with the file list of images to process;
        - a list of ImageData objects;
        - a CubeData object.
    gap : int, optional
        Number of frames to skip between each difference calculation. The default is 2.
    remove_zernikes : bool or list[int]
        If not False, the zernikes modes to remove from the difference, before computing the std
    stds_out : bool, optional
        If True, returns the standard deviations of the differences. The default is True.

    Returns
    -------
    diff_vec : list[ImageData]
        Array of differences between frames.
    svec : ArrayLike
        Array of standard deviations for each frame difference.

    """
    from tqdm import trange
    import sys as _sys
    from io import StringIO as _sIO
    from opticalib.ground.modal_decomposer import ZernikeFitter

    zfit = ZernikeFitter()
    if isinstance(tn_or_fl, str):
        if osu.is_tn(tn_or_fl):
            llist = osu.getFileList(tn_or_fl)
        else:
            raise ValueError("Invalid tracking number")
    else:
        llist = tn_or_fl
    nfile = len(llist)
    npoints = int(nfile / gap) - 2
    idx0 = _np.arange(0, npoints * gap, gap)
    idx1 = idx0 + 1
    svec = _np.empty(npoints)
    diff_vec = []
    for i in trange(npoints, total=npoints, ncols=88, unit=" diffs"):
        diff = _ip.frame(idx1[i], llist) - _ip.frame(idx0[i], llist)
        if remove_zernikes:
            old_stdout = _sys.stdout
            _sys.stdout = _sIO()
            diff = zfit.removeZernike(diff)
            _sys.stdout = old_stdout
        diff_vec.append(diff)
        svec[i] = diff.std()
    if stds_out:
        return diff_vec, svec
    return diff_vec


def timevec(tn: str) -> _ot.ArrayLike:
    """
    Parameters
    ----------
    tn : str
        Tracking number of the frames to process.

    Returns
    -------
    timevector : _np.ndarray
        Array of time values for each frame.

    """
    fold = osu.findTracknum(tn)
    if "OPDImages" in fold:
        flist = osu.getFileList(tn)
        nfile = len(flist)
        tspace = 1.0 / 28.57  # TODO: hardcoded!!
        timevector = range(nfile) * tspace
    elif "OPDSeries" in fold:
        # Assuming files named as 'YYYYMMDD_HHMMSS.fits'
        flist = osu.getFileList(tn, key="20")
        timevector = []
        for f in flist:
            tni = (f.split("/")[-1]).replace(".fits", "")
            jdi = _track2jd(tni)
            timevector.append(jdi)
        timevector = _np.array(timevector)
    return timevector


def runningMean(vec: _ot.ArrayLike, npoints: int) -> _ot.ArrayLike:
    """
    Computes the running mean of a 1D array.

    Parameters
    ----------
    vec : _ot.ArrayLike
        Input array.
    npoints : int
        Number of points to average over.

    Returns
    -------
    _ot.ArrayLike
        Running mean of the input array.
    """
    return _np.convolve(vec, _np.ones(npoints), "valid") / npoints


def structfunc(vect: _ot.ArrayLike, gapvect: _ot.ArrayLike) -> _ot.ArrayLike:
    """
    Computes the structure function for a given time series.

    Parameters
    ----------
    vect : _ot.ArrayLike
        Input time series data.
    gapvect : _ot.ArrayLike
        Array of gap values to compute the structure function.

    Returns
    -------
    _ot.ArrayLike
        Structure function values for each gap.
    """
    nn = _np.shape(vect)[0]
    maxgap = _np.max(gapvect)
    ngap = len(gapvect)
    n2ave = int(nn / (maxgap)) - 1  # or -maxgap??
    jump = maxgap
    st = _np.zeros(ngap)
    for j in range(ngap):
        tx = []
        for k in range(n2ave):
            print("Using positions:")
            print(k * jump, k * jump + gapvect[j])
            tx.append((vect[k * jump] - vect[k * jump + gapvect[j]]) ** 2)
        st[j] = _np.mean(_np.sqrt(tx))
    return st


def noise_strfunct(
    tn: str, tau_vector: _ot.ArrayLike, zernike_vector: list[int] = [1, 2, 3]
):
    """
    Computes the noise structure function for a given time series.

    Parameters
    ----------
    tn : str
        Tracking number of the frames to process.
    tau_vector : _ot.ArrayLike
        Array of gap values to compute the structure function.
    zernike_vector : list[int], optional
        List of Zernike modes to remove from the images before computing the
        noise structure function. The default is [1, 2, 3].

    Returns
    -------
    mean_rms : _ot.ArrayLike
        Array of mean RMS values for each gap.
    n_meas : int
        Number of measurements used in the computation.
    """
    zf = _md.ZernikeFitter()

    fold = osu.findTracknum(tn)
    fl = osu.getFileList(tn, key=("20" if fold == "OPDSeries" else ".4D"))
    cube = osu.loadCubeFromFilelist(fl)
    i_max = int(
        (len(fl) - tau_vector[tau_vector.shape[0] - 1])
        / (tau_vector[tau_vector.shape[0] - 1] * 2)
    )
    if i_max <= 10:
        print("WARNING! low sampling...")
    mean_rms_list = []
    for j in range(tau_vector.shape[0]):
        dist = tau_vector[j]
        rms_list = []
        for i in range(i_max):
            k = i * dist * 2
            image_diff = cube[:, :, k] - cube[:, :, k + dist]
            image_ttr = zf.removeZernike(image_diff, zernike_vector)
            rms = image_ttr.std()
            rms_list.append(rms)
        rms_vector = _np.array(rms_list)
        aa = rms_vector.mean()
        mean_rms_list.append(aa)
    mean_rms = _np.array(mean_rms_list)
    n_meas = rms_vector.shape[0] * 2 * tau_vector.shape[0]
    return mean_rms, n_meas


def noise_pushpull(tn: str, template: list[int], zern2remove: list[int] = [1, 2, 3]):
    """
    Computes the noise structure function using a push-pull reduction algorithm.

    Parameters
    ----------
    tn : str
        Tracking number of the frames to process.
    template : list[int]
        List of integers representing the push-pull template to apply to the frames.
    zern2remove : list[int], optional
        List of Zernike modes to remove from the images before computing the
        noise structure function. The default is [1, 2, 3].

    Returns
    -------
    resrms : _ot.ArrayLike
        Array of residual RMS values for each template configuration.
    restt : _ot.ArrayLike
        Array of residual Tip/TiltZernike coefficients for each template configuration.
    """
    zf = _md.ZernikeFitter()

    fold = osu.findTracknum(tn)
    fl = osu.getFileList(tn, key=("20" if fold == "OPDSeries" else "4D"))
    nfiles = len(fl)
    resrms = []
    restt = []
    for i in _np.arange(len(template)):
        template = _np.ones(template[i])
        template[1::2] = -1
        nframes2use = int(nfiles / template[i] * template[i])
        img = _ip.pushPullReductionAlgorithm(fl[:nframes2use], template)
        cc = zf.fit(img, zern2remove)
        # qui fare rimuovi fit # FIXME
        resrms.append(img.std())
        restt.append(cc)
    resrms = _np.array(resrms)
    restt = _np.array(restt)
    return resrms, restt


def _track2jd(tni: str) -> float:
    """
    Converts a tracking number timestamp to Julian Date.

    Parameters
    ----------
    tni : str
        Tracking number timestamp in the format 'YYYYMMDD_HHMMSS'.

    Returns
    -------
    jdi : float
        Julian Date corresponding to the tracking number timestamp.
    """
    import jdcal

    y = int(tni[0:4])
    m = int(tni[4:6])
    d = int(tni[6:8])
    h = int(tni[9:11])
    mi = float(tni[11:13])
    s = float(tni[13:15])
    t = [y, m, d, h, mi, s]
    jdi = sum(jdcal.gcal2jd(t[0], t[1], t[2])) + t[3] / 24 + t[4] / 1440 + t[5] / 86400
    return jdi


__all__ = [
    "averageFrames",
    "saveAverage",
    "openAverage",
    "runningDiff",
    "timevec",
    "runningMean",
    "structfunc",
    "noise_strfunct",
    "noise_pushpull",
]
