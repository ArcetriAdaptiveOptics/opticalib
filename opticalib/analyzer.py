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


def averageFrames(
    tn: str,
    first: int = 0,
    last: int = -1,
    file_selector: list[int] = None,
    thresh: bool = False,
):
    """
    Perform the average of a list of images, retrievable through a tracking
    number.

    Parameters
    ----------
    tn : str
        Data Tracking Number.
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
        DESCRIPTION. The default is None.

    Returns
    -------
    aveimg : ndarray
        Final image of averaged frames.

    """
    fl = osu.getFileList(tn, fold=_OPDSER.split("/")[-1], key="20")
    s = slice(first, last) if last != -1 else slice(first, None)
    fl = fl[s] if file_selector is None else fl[file_selector]
    # elif file_selector is not None:
    #     first = 0
    #     last = len(fl)
    #     fl = [
    #         fl[x]
    #         for x in _np.arange(first, last, 1)
    #         if file_selector is None or x in file_selector
    #     ]
    imcube = createCube(fl)

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

        # img = imcube[:, :, 0].data * 0
        # mmask = imcube[:, :, 0].mask
        # nn = 0
        # for j in range(imcube.shape[2]):
        #     im = imcube[:, :, j]
        #     size = im.data.compressed.size
        #     if size > 1:
        #         nn += 1
        #         img += im.data
        #         mmask = _np.ma.mask_or(im.mask, mmask)
        # img = img / nn
        # aveimg = _np.ma.masked_array(img, mask=mmask)
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
    tn_or_fl: str | list[str] | list[_ot.ImageData] | _ot.CubeData, gap: int = 2
):
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

    Returns
    -------
    svec : ndarray
        Array of standard deviations for each frame difference.

    """
    zfit = zern.ZernikeFitter()
    if isinstance(tn_or_fl, str):
        if osu.is_tn(tn_or_fl):
            fl = osu.getFileList(tn_or_fl)
            llist = [osu.load_fits(x) for x in fl]
        else:
            raise ValueError("Invalid tracking number")
    elif isinstance(tn_or_fl, list):
        if all(isinstance(item, str) for item in tn_or_fl):
            llist = [osu.load_fits(x) for x in tn_or_fl]
        elif all(_ot.isinstance_(item, "ImageData") for item in tn_or_fl):
            llist = [item.file for item in tn_or_fl]
        else:
            raise TypeError("Invalid list elements")
    elif _ot.isinstance_(tn_or_fl, "CubeData"):
        llist = tn_or_fl.transpose(2, 0, 1).tolist()
    nfile = len(llist)
    npoints = int(nfile / gap) - 2
    idx0 = _np.arange(0, npoints * gap, gap)
    idx1 = idx0 + 1
    svec = _np.empty(npoints)
    for i in range(npoints):
        diff = llist[idx1[i]] - llist[idx0[i]]
        diff = zfit.removeZernike(diff)
        svec[i] = diff.std()
    return svec


# TODO: TO REMOVE
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
        if isinstance(mylist[0], str):
            img = osu.read_phasemap(mylist[idx])
        elif _ot.isinstance_(mylist[0], _ot.ImageData):
            img = mylist[idx]
    else:
        img = mylist[:, :, idx]
    return img


# TODO: Check for hardcoded assumptions on dimensions ecc...
def spectrum(
    signal: _ot.ArrayLike, dt: float = 1, show: bool = None
) -> tuple[_ot.ArrayLike, _ot.ArrayLike]:
    """
    Computes the one-dimensional power spectrum of a signal or a set of signals.

    Parameters
    ----------
    signal : ndarray
        Input signal or signals.
    dt : float, optional
        Time spacing between samples. The default is 1.
    show : bool, optional
        If True, displays the power spectrum. The default is None.

    Returns
    -------
    spe : float | ndarray
        Power spectrum of the input signal(s).
    freq : float | ArrayLike
        Frequency bins corresponding to the power spectrum.
    """
    nsig = signal.shape
    thedim = 0 if _np.size(nsig) == 1 else 1
    spe = _fft.rfft(signal, axis=thedim, norm="ortho")
    nn = _np.sqrt(spe.shape[thedim])  # modRB
    spe = (_np.abs(spe)) / nn
    freq = _fft.rfftfreq(signal.shape[thedim], d=dt)
    if _np.size(nsig) == 1:
        spe[0] = 0
    else:
        spe[:, 0] = 0
    if show is not None:
        _plt.figure()
        for i in range(0, len(spe)):
            _plt.plot(freq, spe[i, :], label=f"Channel {i}")
        _plt.xlabel(r"Frequency $[\mathrm{Hz}]$")
        _plt.ylabel("PS Amplitude")
        _plt.legend(loc="best")
        _plt.show()
    return spe, freq


# TODO: TO REMOVE -> equan to `intoFullFrame`
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
    flist = osu.getFileList(tn)
    nfile = len(flist)
    if "OPDImages" in fold:
        tspace = 1.0 / 28.57  # TODO: hardcoded!!
        timevector = range(nfile) * tspace
    elif "OPDSeries" in fold:
        timevector = []
        for f in flist:
            tni = f.split("/")[-2]
            jdi = track2jd(tni)
            timevector.append(jdi)
        timevector = _np.array(timevector)
    return timevector


def track2jd(tni: str):
    """


    Parameters
    ----------
    tni : TYPE
        DESCRIPTION.

    Returns
    -------
    jdi : TYPE
        DESCRIPTION.

    """
    y = tni[0:4]
    mo = tni[4:6]
    d = tni[6:8]
    h = float(tni[9:11])
    mi = float(tni[11:13])
    s = float(tni[13:15])
    t = [y, mo, d, h, mi, s]
    jdi = sum(_jdcal.gcal2jd(t[0], t[1], t[2])) + t[3] / 24 + t[4] / 1440 + t[5] / 86400
    return jdi


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


def strfunct(vect: _ot.ArrayLike, gapvect: _ot.ArrayLike) -> _ot.ArrayLike:
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
    nn = _np.shape(vect)
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
            _plt.figure()
            _plt.imshow(imgs[i])
            _plt.title(titles[i])
            _plt.colorbar()
        _plt.show()
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


def comp_psd(
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
    Computes the power spectrum of a 2D image.

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
    if crop:
        cir = _qpupil(-1 * imgin.mask + 1)
        cir = _np.array(cir[0:3]).astype(int)
        img = imgin.data[
            cir[0] - cir[2] : cir[0] + cir[2], cir[1] - cir[2] : cir[1] + cir[2]
        ]
        m = imgin.mask[
            cir[0] - cir[2] : cir[0] + cir[2], cir[1] - cir[2] : cir[1] + cir[2]
        ]
        img = _np.ma.masked_array(img, m)
    else:
        img = imgin.copy()
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
        _plt.figure()
        _plt.plot(fout[1:], Aout[1:] * fout[1:], ".")
        _plt.yscale("log")
        _plt.xscale("log")
        _plt.title("Power spectrum")
        _plt.xlabel("Frequency [Hz]")
        _plt.ylabel("Amplitude [A^2]")
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


def getDataFileList(tn: str) -> list[str]:
    """
    Returns a list of data files for the given tracking number.

    Parameters
    ----------
    tn : str
        Tracking number.

    Returns
    -------
    filelist : list of str
        List of file paths to the data files.
    """
    fold = osu.findTracknum(tn, complete_path=True)
    filelist = osu.getFileList(tn, fold=fold)
    return filelist


def pushPullReductionAlgorithm(
    imagelist: list[_ot.ImageData] | _ot.CubeData,
    template: _ot.ArrayLike,
    normalization: _ot.Optional[float | int] = None,
    shuffle: int = 0,
):
    """
    Performs the basic operation of processing PushPull data.

    Parameters
    ----------
    imagelist : list of ImageData ! CubeData
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
    if shuffle == 0:
        # Template weights computation
        w = template.astype(_np.result_type(template, imagelist[0].data), copy=True)
        if n_images > 2:
            w[1:-1] *= 2.0
        # OR-reduce all masks once
        master_mask = _np.logical_or.reduce(
            [_xp.asnumpy(ima.mask) for ima in imagelist]
        )
        # Compute weighted sum over realizations on raw data
        stack = _np.stack([ima.data for ima in imagelist], axis=0)  # (n, H, W)
        image = _xp.asnumpy(  # (H, W)
            _xp.tensordot(_xp.asarray(w), _xp.asarray(stack), axes=(0, 0))
        )
    else:
        print("Shuffle option")
        for i in range(0, shuffle - 1):
            for x in range(1, 2):
                opd2add = (
                    imagelist[i * 3 + x] * template[x]
                    + imagelist[i * 3 + x - 1] * template[x - 1]
                )
                master_mask2add = _np.ma.mask_or(
                    imagelist[i * 3 + x].mask, imagelist[i * 3 + x - 1].mask
                )
                if i == 0 and x == 1:
                    master_mask = master_mask2add
                else:
                    master_mask = _np.ma.mask_or(master_mask, master_mask2add)
                image += opd2add
    if normalization is None:
        norm_factor = _np.max(((template.shape[0] - 1), 1))
    else:
        norm_factor = normalization
    image = _np.ma.masked_array(image, mask=master_mask) / norm_factor
    return image


def createCube(fl_or_il: list[str], register: bool = False):
    """
    Creates a cube of images from an images file list

    Parameters
    ----------
    fl_or_il : list of str
        Either:
        - the list of image files;
        - a list of ImageData.
    register : int or tuple, optional
        If not False, and int or a tuple of int must be passed as value, and
        the registration algorithm is performed on the images before stacking them
        into the cube. Default is False.

    Returns
    -------
    cube : ndarray
        Data cube containing the images/frames stacked.
    """
    # check it is a list
    if not isinstance(fl_or_il, list):
        raise TypeError("filelist must be a list of strings or images")
    # check if it is composed of file paths to load
    if all(isinstance(item, str) for item in fl_or_il):
        fl_or_il = [osu.read_phasemap(f) for f in fl_or_il]
        if not all(_ot.isinstance_(item, "ImageData") for item in fl_or_il):
            raise TypeError("Data different from `images` loaded. Check filelist.")
    # finally check if it is a list of ImageData
    elif not all(_ot.isinstance_(item, "ImageData") for item in fl_or_il):
        raise TypeError("filelist must be either a list of strings or ImageData")
    cube = _np.ma.dstack(fl_or_il)
    return cube


def removeZernikeFromCube(
    cube: _ot.CubeData, zmodes: _ot.ArrayLike = None
) -> _ot.CubeData:
    """
    Removes Zernike modes from each frame in a cube of images.

    Parameters
    ----------
    cube : ndarray
        Data cube containing the images/frames stacked.
    zmodes : ndarray, optional
        Zernike modes to remove. If None, the first 3 modes are removed.

    Returns
    -------
    newCube : ndarray
        Cube with Zernike modes removed from each frame.
    """
    from tqdm import tqdm
    zfit = zern.ZernikeFitter()
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
        newCube[:, :, i] = zfit.removeZernike(cube[:, :, i], zmodes)
    return newCube


def makeCubeMasterMask(cube: _ot.CubeData, apply: bool = False) -> _ot.CubeData:
    """
    Creates a master mask for a cube of images by performing a logical OR operation
    across all individual image masks.

    Parameters
    ----------
    cube : ndarray
        Data cube containing the images/frames stacked.
    apply : bool, optional
        If True, applies the master mask to all frames in the cube and
        returns the modified cube.

    Returns
    -------
    master_mask or cube: MaskData or CubeData
        Master mask for the cube or the cube with the master mask applied.
    """
    master_mask = _np.logical_or.reduce(
        [cube[:, :, i].mask for i in range(cube.shape[2])]
    )
    if apply:
        cube.mask = _np.broadcast_to(master_mask[:, :, None], cube.shape)
        return cube
    else:
        return master_mask


def modeRebinner(
    img: _ot.ImageData, rebin: int, method: str = "averaging"
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
        Rebinning method, either 'averaging' or 'sampling'. The default is 'averaging'.

    Returns
    -------
    newImg : masked_array
        Rebinned image.
    """
    shape = img.shape
    new_shape = (shape[0] // rebin, shape[1] // rebin)
    sample = False if method == "sampling" else True
    newImg = _rebin2DArray(img, new_shape, sample=sample)
    return newImg


def cubeRebinner(
    cube: _ot.CubeData, rebin: int, method: str = "averaging"
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
        Rebinning method, either 'averaging' or 'sampling'. The default is
        'averaging'.

    Returns
    -------
    newCube : ndarray
        Rebinned cube.
    """
    newCube = []
    for i in range(cube.shape[-1]):
        newCube.append(modeRebinner(cube[:, :, i], rebin, method=method))
    return _np.ma.dstack(newCube)


# From ARTE #
def _rebin2DArray(
    a: _ot.ArrayLike, new_shape: tuple[int, int], sample: bool = False
) -> _ot.ArrayLike:
    """
    Replacement of IDL's rebin() function for 2d arrays.
    Resizes a 2d array by averaging or repeating elements.
    New dimensions must be integral factors of original dimensions,
    otherwise a ValueError exception will be raised.

    Parameters
    ----------
    a : ndarray
        Input array.
    new_shape : 2-elements sequence
        Shape of the output array
    sample : bool
        if True, when reducing the array side elements are set
        using a nearest-neighbor algorithm instead of averaging.
        This parameter has no effect when enlarging the array.

    Returns
    -------
    rebinned_array : ndarray
        If the new shape is smaller of the input array  the data are averaged,
        unless the sample parameter is set.
        If the new shape is bigger array elements are repeated.

    Raises
    ------
    ValueError
        in the following cases:
         - new_shape is not a sequence of 2 values that can be converted to int
         - new dimensions are not an integral factor of original dimensions
    NotImplementedError
         - one dimension requires an upsampling while the other requires
           a downsampling

    Examples
    --------
    >>> a = np.array([[0, 1], [2, 3]])
    >>> b = rebin(a, (4, 6)) #upsize
    >>> b
    array([[0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1, 1],
           [2, 2, 2, 3, 3, 3],
           [2, 2, 2, 3, 3, 3]])
    >>> rebin(b, (2, 3)) #downsize
    array([[0. , 0.5, 1. ],
           [2. , 2.5, 3. ]])
    >>> rebin(b, (2, 3), sample=True) #downsize
    array([[0, 0, 1],
           [2, 2, 3]])
    """

    # unpack early to allow any 2-length type for new_shape
    m, n = map(int, new_shape)

    if a.shape == (m, n):
        return a

    M, N = a.shape

    if m <= M and n <= M:
        if (M // m != M / m) or (N // n != N / n):
            raise ValueError("Cannot downsample by non-integer factors")

    elif M <= m and M <= m:
        if (m // M != m / M) or (n // N != n / N):
            raise ValueError("Cannot upsample by non-integer factors")

    else:
        raise NotImplementedError(
            "Up- and down-sampling in different axes " "is not supported"
        )

    if sample:
        slices = [slice(0, old, float(old) / new) for old, new in zip(a.shape, (m, n))]
        idx = _np.mgrid[slices].astype(int)
        return a[tuple(idx)]
    else:
        if m <= M and n <= N:
            return a.reshape((m, M // m, n, N // n)).mean(3).mean(1)
        elif m >= M and n >= M:
            return _np.repeat(_np.repeat(a, m / M, axis=0), n / N, axis=1)
