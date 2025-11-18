"""
Module containing all the functions necessary to process the data acquired for
the Influence Function measurements.

Author(s):
----------
- Pietro Ferraiuolo: pietro.ferraiuolo@inaf.it
- Runa Briguglio: runa.briguglio@inaf.it

High-level Functions
--------------------
process(tn, registration=False, roi=None, save=False, rebin=1)
    Function that processes the data contained in the OPDImages/tn folder. By
    performing the differential algorithm, it produces fits images for each
    commanded mode into the IFFunctions/tn folder, and creates a cube from these
    into INTMatrices/tn. If 'registration is not False', upon createing the cube,
    the registration algorithm is performed.

stackCubes(tnlist)
    Function that, given as imput a tracking number list containing cubes data,
    will stack the found cubes into a new one with a new tracking number, into
    INTMatrices/new_tn. A 'flag.txt' file will be created to give more information
    on the process.

Example
-------

```python
tn1 = '20160516_114916'
tn2 = '20160516_114917' # A copy of tn1 (simulated) data
ifp.process(tn1, save=True)
Cube saved in '.../path/to/data/OPTData/INTMatrices/20160516_114916/IMcube.fits'
ifp.process(tn2, save=True)
Cube saved in '.../path/to/data/OPTData/INTMatrices/20160516_114917/IMcube.fits'
tnlist = [tn1, tn2]
ifp.stackCubes(tnlist)
Stacked cube and matrices saved in '.../path/to/data/OPTData/INTMatrices/'new_tn'/IMcube.fits'
```
"""

import os as _os
import xupy as _xp
import numpy as _np
import shutil as _sh
import configparser as _cp
from tqdm import tqdm as _tqdm
from opticalib import typings as _ot
from opticalib.core.root import _folds
from opticalib.core import fitsarray as _fa
from opticalib.core import read_config as _rif
from concurrent.futures import ThreadPoolExecutor as _tpe
from opticalib.ground import modal_decomposer as _zern, osutils as _osu, roi as _roi

# from scripts.misc.IFFPackage import actuator_identification_lib as _fa

_fn = _folds()
_config = _cp.ConfigParser()
_ifFold = _fn.IFFUNCTIONS_ROOT_FOLDER
_intMatFold = _fn.INTMAT_ROOT_FOLDER
_frameCenter = [200, 200]
_ts = _osu.newtn

ampVecFile = "ampVector.fits"
modesVecFile = "modesVector.fits"
templateFile = "template.fits"
regisActFile = "regActs.fits"
shuffleFile = "shuffle.dat"
indexListFile = "indexList.fits"
coordfile = ""  # TODO
cmdMatFile = "cmdMatrix.fits"
cubeFile = "IMCube.fits"
flagFile = "flag.txt"


def process(
    tn: str,
    register: bool = False,
    save: bool = False,
    rebin: int = 1,
    *,
    nworkers: int = 1,
    nmode_prefetch: int = 0,
) -> None:
    """
    High level function with processes the data contained in the given tracking
    number OPDimages folder, performing the differential algorithm and saving
    the final cube.

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.
    register : bool, optional
        Parameter which enables the registration option. The default is False.
    save : bool, optional
        If True, the final cube is saved into the INTMatrices/tn folder. The
        default is False.
    rebin : int, optional
        Rebinning factor to apply to the images before stacking them into the
        cube. The default is 1, which means no rebinning.
    roi : int, optional
        If not None, it defines the size of the square ROI to be used for the
        registration algorithm. The default is None.
    nworkers : int, optional
        Number of workers to use for the processing. The default is 1 for no
        parallelization.
    nmode_prefetch : int, optional
        Number of modes to prefetch during the processing. The default is 0
        for no prefetching.
    """
    ampVector, modesVector, template, _, registrationActs, shuffle = _getAcqPar(tn)
    if not modesVector.dtype.type is _np.int_:
        modesVector = modesVector.astype(int)
    new_fold = _os.path.join(_intMatFold, tn)
    if not _os.path.exists(new_fold):
        _os.mkdir(new_fold)
    trigFrame = getTriggerFrame(tn)
    regMat = getRegFileMatrix(tn, trigFrame)
    modesMat = getIffFileMatrix(tn, trigFrame)
    modesMatReorg = _modesReorganization(modesMat)
    iffRedux(
        tn=tn,
        fileMat=modesMatReorg,
        ampVect=ampVector,
        modeList=modesVector,
        template=template,
        shuffle=shuffle,
        io_workers=nworkers,
        prefetch=nmode_prefetch,
    )
    if register and not len(regMat) == 0:
        actImgList = registrationRedux(tn, regMat)
        dx = findFrameOffset(tn, actImgList, registrationActs)
    else:
        dx = register
    if save:
        saveCube(tn, rebin=rebin, register=dx)


def saveCube(
    tn: str,
    rebin: int = 1,
    cube_header: _ot.Optional[dict[str, _ot.Any] | _ot.Header] = None,
    register: bool = False,
) -> _ot.CubeData:
    """
    Creates and save a cube from the fits files contained in the tn folder,
    along with the command matrix and the modes vector fits.

    Parameters
    ----------
    tn : str
        Tracking number of the IFFunctions data folder from which create the cu
        be.
    rebin : int
        Rebinning factor to apply to the images before stacking them into the
        cube.
    register : int or tuple, optional
        If not False, an int or a tuple of int must be passed as value, and
        the registration algorithm is performed on the images before stacking them
        into the cube. Default is False.
    cube_header : dict | Header, optional
        Header to be used for the cube. If None, a default header is created.

    Returns
    -------
    cube : masked_array
        Data cube of the images, with shape (npx, npx, nmodes).
    """
    from opticalib.analyzer import cubeRebinner

    cube = _osu.loadCubeFromFilelist(tn_or_fl=tn, fold=_ifFold, key="mode_")
    # cube.mask = _roi.cubeMasterMask(cube) # w/ DP weird behavior

    ## TODO ??
    # if register is not False:
    #     print(f"Applying registration with offset {register}...")
    #     for i in _tqdm(range(cube.shape[-1]), desc="Registering cube...", unit='modes'):
    #         cube[:, :, i] = _osu.shiftImage(cube[:, :, i], register)

    # Rebinning the cube
    header = {}
    header["REBIN"] = (rebin, "Rebinning factor applied")
    if cube_header is not None:
        header.update(cube_header)
    if rebin > 1:
        cube = cubeRebinner(cube, rebin)
    # Saving the cube
    new_fold = _os.path.join(_intMatFold, tn)
    if not _os.path.exists(new_fold):
        _os.mkdir(new_fold)
    cube_path = _os.path.join(new_fold, cubeFile)
    _osu.save_fits(cube_path, cube, overwrite=True, header=header)
    # Copying the cmdMatrix and the ModesVector into the INTMAT Folder
    _copyFromIffToIM(name="cmdMatrix.fits", tn=tn)
    _copyFromIffToIM(name="modesVector.fits", tn=tn)
    print(
        f"Cube of shape {cube.shape} saved in '.../{'/'.join(cube_path.split('/')[-2:])}'"
    )
    return cube


def stackCubes(tnlist: str, cubeNames: _ot.Optional[list[str]]) -> None:
    """
    Stack the cubes sontained in the corresponding tracking number folder, creating
    a new cube, along with stacked command matrix and modes vector.

    Parameters
    ----------
    tnlist : list of str
        List containing the tracking numbers of the cubes to stack.
    cubeNames : list of str, optional
        List containing the names of the cube files, corresponding to the tn, to stack.
        If not provided, the default names `IMCube.fits` will be used.

    Returns
    -------
    stacked_cube : masked_array
        Final cube, stacked along the 3th axis.
    """
    # TODO: aggiungere variabile `cubeNames: str|list[str]` che deve matchare
    # `len(tnlist)` e che contiene il nome del file da caricare per ogni tn
    new_tn = _ts()
    stacked_cube_fold = _os.path.join(_intMatFold, new_tn)
    _os.mkdir(stacked_cube_fold)
    cube_parameters = _getCubeList(tnlist, cubeNames)
    flag = _checkStackedCubes(tnlist)
    # Stacking the cube and the matrices
    stacked_cube = _np.ma.dstack(cube_parameters[0])
    stacked_cmat = _np.hstack(cube_parameters[1])
    stacked_mvec = _np.dstack(cube_parameters[2])
    # Saving everithing to a new file into a new tn
    nchead = {}
    nchead["FLAG"] = (flag, "stacking mode")
    nchead["NSTACK"] = (len(tnlist), "number of stacked cubes")
    for i, tn in enumerate(tnlist):
        nchead[f"TN{i+1}"] = (tn, f"TN of stacked cube {i+1}")
    nchead["REBIN"] = (cube_parameters[3], "common rebinning factor")
    save_cube = _os.path.join(stacked_cube_fold, cubeFile)
    save_cmat = _os.path.join(stacked_cube_fold, "cmdMatrix.fits")
    save_mvec = _os.path.join(stacked_cube_fold, "modesVector.fits")
    _osu.save_fits(save_cube, stacked_cube, header=nchead)
    _osu.save_fits(save_cmat, stacked_cmat)
    _osu.save_fits(save_mvec, stacked_mvec)
    print(f"Stacked cube and matrices saved in {new_tn}")


def filterZernikeCube(
    tn: str,
    zern_modes: _ot.Optional[list[int]] = None,
    save: bool = True,
) -> tuple[_ot.CubeData, str]:
    """
    Function which filters out the desired zernike modes from a cube.

    Parameters
    ----------
    tn : str
        Tracking number of the cube to filter.
    zern_modes : list, optional
        List of zernike modes to filter out. The default is [1,2,3]
        (piston, tip and tilt).
    save : bool, optional
        If True, the filtered cube will be saved to disk. The default is True.

    Returns
    -------
    ffcube : masked array
        Filtered cube.
    new_tn : str
        Tracking Number of the new folder where the filtered cube is saved.
    """
    new_tn = _os.path.join(_intMatFold, _ts())
    CmdMat = _os.path.join(_intMatFold, tn, cmdMatFile)
    ModesVec = _os.path.join(_intMatFold, tn, modesVecFile)
    cube = _fa.FitsMaskedArray.fromFits(_os.path.join(_intMatFold, tn, cubeFile))
    zern_modes = zern_modes if zern_modes is not None else [1, 2, 3]
    from opticalib.analyzer import removeZernikeFromCube

    ffcube = removeZernikeFromCube(cube, zern_modes)
    # TODO: Problem with master mask... is it the data?
    # ffcube.mask = _roi.cubeMasterMask(ffcube)

    if save:
        _os.mkdir(new_tn)
        ffcube.writeto(_os.path.join(new_tn, cubeFile), overwrite=True)
        _sh.copyfile(CmdMat, _os.path.join(new_tn, cmdMatFile))
        _sh.copyfile(ModesVec, _os.path.join(new_tn, modesVecFile))
        print(f"Filtered cube saved at {new_tn}")
    return ffcube, new_tn.split("/")[-1]


def iffRedux(
    tn: str,
    fileMat: list[list[str]],
    ampVect: _ot.ArrayLike,
    modeList: _ot.ArrayLike,
    template: _ot.ArrayLike,
    *,
    shuffle: int = 0,
    io_workers: int = 1,
    prefetch: int = 0,
) -> None:
    """
    Reduction function that performs the push-pull analysis on each mode, saving
    out the final processed image for each mode.<br>
    The differential algorithm for each mode is the sum over the push-pull
    realizations of the images, and it is performed as follows:

    :math::
        \\sum_i \\dfrac{I_i \\cdot t_i - I_{i-1}\\cdot t_{i-1}}{A\\cdot(n-1)}

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.
    fileMat : ndarray
        A matrix of images in string format, in which each row is a mode and the
        columns are its template realization.
    ampVect : float | ArrayLike
        Vector containing the amplitude for each commanded mode.
    modeList : int | ArrayLike
        Vector conaining the list of commanded modes.
    template : int | ArrayLike
        Template for the push-pull command actuation.
    shuffle : int, optional
        A value different from 0 activates the shuffle option, and the imput
        value is the number of repetition for each mode's push-pull packet. The
        default is 0, which means the shuffle is OFF.
    io_workers : int, optional
        Number of threads to use for I/O prefetching. The default is 1.
    prefetch : int, optional
        Number of future modes to keep in-flight. The default is 0.

    Notes on performance
    --------------------
    - I/O is overlapped with computation via a small prefetch window.
    - io_workers controls the number of threads used to prefetch mode blocks.
    - prefetch controls how many future modes to keep in-flight.
    """
    from opticalib.analyzer import pushPullReductionAlgorithm

    fold = _os.path.join(_ifFold, tn)
    nmodes = len(modeList)

    # Helper: read one mode's frame block
    def _read_mode(paths: list[str]) -> list[_ot.ImageData]:
        # Sequential reads inside a worker to keep per-mode locality
        return [_osu.read_phasemap(p) for p in paths]

    ## NEW METHOD: THREADING I/O WORKERS WITH PREFETCHING ##
    # ---------------------------------------------------- #

    # just to be sure...
    io_workers = max(1, int(io_workers))
    prefetch = max(0, int(prefetch))

    futures: dict[int, _ot.Any] = {}
    with _tpe(max_workers=io_workers) as ex:
        for j in range(min(prefetch, nmodes)):
            futures[j] = ex.submit(_read_mode, fileMat[j, :])

        for i in _tqdm(
            range(nmodes), desc="Processing...", total=nmodes, unit="modes", ncols=80
        ):
            # Ensure current mode is scheduled
            if i not in futures:
                futures[i] = ex.submit(_read_mode, fileMat[i, :])

            # Fetch prefetched images (waits here if not yet done)
            imagelist = futures[i].result()
            del futures[i]

            # Schedule next mode to keep window full
            j = i + prefetch
            if j < nmodes and j not in futures:
                futures[j] = ex.submit(_read_mode, fileMat[j, :])

            # Compute (already vectorized inside pushPullRedux)

            # img = pushPullRedux(imagelist, template, shuffle)
            # norm_img = img / (2 * ampVect[i])

            norm_img = pushPullReductionAlgorithm(
                imagelist,
                template,
                normalization=(_np.max(((template.shape[0] - 1), 1)) * 2 * ampVect[i]),
                shuffle=shuffle,
            )

            img_name = _os.path.join(fold, f"mode_{int(modeList[i]):04d}.fits")
            header = {
                "MODEID": (int(modeList[i]), "mode id"),
                "AMP": (float(ampVect[i]), "mode amplitude"),
                "TEMPLATE": (len(template), "push-pull length"),
            }
            _osu.save_fits(img_name, norm_img, overwrite=True, header=header)


def registrationRedux(tn: str, fileMat: list[str]) -> list[_ot.ImageData]:
    """
    Reduction function that performs the push-pull analysis on the registration
    data.

    Parameters
    ----------
    fileMat : ndarray
        A matrix of images in string format, in which each row is a mode and the
        columns are its template realization.

    Returns
    -------
    imgList : ArrayLike
        List of the processed registration images.
    """
    from opticalib.analyzer import pushPullReductionAlgorithm

    _, infoR, _, _ = _getAcqInfo(tn)
    template = infoR["template"]
    if _np.array_equal(fileMat, _np.array([])) and len(infoR["modesid"]) == 0:
        print("No registration data found")
        return []
    nActs = fileMat.shape[0]
    imglist = []
    for i in range(0, nActs - 1):
        imgs = [_osu.read_phasemap(x) for x in fileMat[i, :]]
        img = pushPullReductionAlgorithm(imgs, template)
        imglist.append(img)
    # cube = _np.ma.masked_array(imglist)
    # _osu.save_fits(_os.path.join(_intMatFold, tn, "regActCube.fits"), cube)
    return imglist


def findFrameOffset(
    tn: str, imglist: list[_ot.ImageData], actlist: _ot.ArrayLike
) -> float:
    """
    This function computes the position difference between the current frame and
    a reference one.

    Parameters
    ----------
    tn : str
        Tracking number
    imglist : list | masked arrays
        List of the actuator images to be used
    actlist: int | array
        List of actuators (index)

    Returns
    -------
    dp: float
        Position difference
    """
    actCoordFile = _os.path.join(_ifFold, tn, coordfile)
    actCoord = _osu.load_fits(actCoordFile)
    xy = _fa.findFrameCoord(imglist, actlist, actCoord)  # type: ignore
    dp = xy - _frameCenter
    return dp


def getTriggerFrame(tn: str, amplitude: int | float = None, roi: int = None) -> int:
    """
    Analyze the tracking number's images list and search for the trigger frame.

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.
    amplitude : int or float, optional
        Amplitude of the commanded trigger mode, which serves as the check value
        for finding the frame. If no value is passed it is loaded from the iffConfig.ini
        file.

    Returns
    -------
    trigFrame : int
        Index which identifies the trigger frame in the images folder file list.

    Raises
    ------
    RuntimeError
        Error raised if the file iteration goes beyon the expected trigger frame
        wich can be inferred through the number of trigger zeros in the iffConfig.ini
        file.
    """
    zfit = _zern.ZernikeFitter()
    infoT, _, _, _ = _getAcqInfo(tn)
    if amplitude is not None:
        infoT["amplitude"] = amplitude
    fileList = _osu.getFileList(tn)
    img0 = _osu.read_phasemap(fileList[0])
    go = i = 1
    thresh = infoT["amplitude"] / _np.sqrt(3)
    print(f"Trigger threshold: {thresh:.2e}")
    if infoT["zeros"] == 0 and len(infoT["modes"]) == 0:
        trigFrame = 0
        return trigFrame
    # listout = [] # ??
    while go != 0:
        img1 = _osu.read_phasemap(fileList[i])
        if not roi is None:
            rois = _roi.roiGenerator(img0)
            roi2use = rois[roi]  # ??
            _ = rois.pop(roi)
            for r in rois:
                img1.mask[r == 0] = True
                img0.mask[r == 0] = True
        rr2check = _np.nanstd(zfit.removeZernike(img1 - img0, [1, 2, 3]))
        print(f"Frame {i-1}: std = {rr2check:.2e}")
        if go > infoT["zeros"] + 1:
            msg = f"Frame {go}. Heading Zeros exceeded: std = {rr2check:.2e} < {thresh:.2e} (Amp/sqrt(3))"
            raise RuntimeError(msg)
        if rr2check > thresh:
            print(f"↑ Trigger Frame found!")
            go = 0
        else:
            i += 1
            go += 1
            img0 = img1
    trigFrame = i
    return trigFrame


def getRegFrames(tn: str, trigFrame: int) -> tuple[int, _ot.ArrayLike]:
    """
    Search for the registration frames in the images file list.

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.
    trigFrame : int
        Trigger frame index.

    Returns
    -------
    regStart : int
        Index which identifies the first registration frame in the images file
        list.
    regEnd : int
        Index which identifies the last registration frame in the images file
        list.
    """
    _, infoR, _, _ = _getAcqInfo(tn)
    timing = _rif.getTiming()
    if infoR["zeros"] == 0 and len(infoR["modes"]) == 0:
        regStart = regEnd = (trigFrame + 1) if trigFrame != 0 else 0
    else:
        regStart = trigFrame + infoR["zeros"] * timing + (1 if trigFrame != 0 else 0)
        regEnd = regStart + len(infoR["modes"]) * len(infoR["template"]) * timing
    return regStart, regEnd


def getRegFileMatrix(tn: str, trigFrame: int) -> tuple[int, _ot.ArrayLike]:
    """
    Search for the registration frames in the images file list, and creates the
    registration file matrix.

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.

    Returns
    -------
    regMat : ndarray
        A matrix of images in string format, containing the registration frames.
        It has shape (registration_modes, n_push_pull).
    """
    if not _osu.is_tn(tn):
        fold = tn[:-15]
        _os.path.isdir(fold)
    else:
        fold = None
    fileList = _osu.getFileList(tn, fold=fold)
    _, infoR, _, _ = _getAcqInfo(tn)
    regStart, regEnd = getRegFrames(tn, trigFrame)
    regList = fileList[regStart:regEnd]
    regMat = _np.reshape(regList, (len(infoR["modes"]), len(infoR["template"])))
    return regMat


def getIffFileMatrix(tn: str, trigFrame: int = None) -> _ot.ArrayLike:
    """
    Creates the iffMat

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.

    Returns
    -------
    iffMat : ndarray
        A matrix of images in string format, conatining all the images for the
        IFF acquisition, that is all the modes with each push-pull realization.
        It has shape (modes, n_push_pull)
    """
    if not _osu.is_tn(tn):
        fold = tn[:-15]
        _os.path.isdir(fold)
    else:
        fold = None
    fileList = _osu.getFileList(tn, fold=fold)
    _, _, infoIF, _ = _getAcqInfo(tn)
    _, regEnd = getRegFrames(tn, trigFrame)
    n_useful_frames = len(infoIF["modes"]) * len(infoIF["template"])
    k = regEnd + infoIF["zeros"]
    iffList = fileList[k : k + n_useful_frames]
    iffMat = _np.reshape(iffList, (len(infoIF["modes"]), len(infoIF["template"])))
    return iffMat


def _copyFromIffToIM(name: str, tn: str) -> None:
    """
    Copies an IFFunctions file from the IFFunctions folder to the IntMatrices folder.

    Parameters
    ----------
    name : str
        Name of the file to copy.
    tn : str
        Tracking number of the data.
    """
    opd_path = _os.path.join(_ifFold, tn, name)
    iff_path = _os.path.join(_intMatFold, tn, name)
    _os.makedirs(_os.path.dirname(iff_path), exist_ok=True)
    _sh.copy2(opd_path, iff_path)


def _getCubeList(
    tnlist: str, cubeNames: _ot.Optional[list[str]] = None
) -> tuple[list[_ot.ImageData], list[_ot.MatrixLike], _ot.ArrayLike, int]:
    """
    Retireves the cubes from each tn in the tnlist.

    Parameters
    ----------
    tnlist : list of str
        List containing the tracking number of the cubes to stack.
    cubeNames : list of str, optional
        List containing the names of the cube files, corresponding to the tn, to stack.
        If not provided, the default names `IMCube.fits` will be used.

    Returns
    -------
    cubeList : list of masked_array
        List containing the cubes to stack.
    matrixList : list of ndarray
        List containing the command matrices for each cube.
    modesVectList : list of ndarray
        List containing the modes vectors for each cube.
    """
    cubeList = []
    matrixList = []
    modesVectList = []
    rebins = []
    if cubeNames is None:
        cubeNames = [cubeFile] * len(tnlist)
    for tn, cname in zip(tnlist, cubeNames):
        fold = _os.path.join(_intMatFold, tn)
        cube_name = _os.path.join(fold, cname)
        matrix_name = _os.path.join(fold, "cmdMatrix.fits")
        modesVec_name = _os.path.join(fold, "modesVector.fits")
        cube = _osu.load_fits(cube_name, True)
        cubeList.append(cube)
        matrixList.append(_osu.load_fits(matrix_name))
        modesVectList.append(_osu.load_fits(modesVec_name))
        rebins.append(int(cube.header.get("REBIN", 1)))
    if not all([rebin == rebins[0] for rebin in rebins]):
        raise ValueError("Cubes have different rebinning factors")
    rebin = rebins[0]
    return cubeList, matrixList, modesVectList, rebin


def _getAcqPar(
    tn: str,
) -> tuple[
    _ot.ArrayLike, _ot.ArrayLike, _ot.ArrayLike, _ot.ArrayLike, _ot.ArrayLike, int
]:
    """
    Reads ad returns the acquisition parameters from fits files.

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.

    Returns
    -------
    ampVector : float | ArrayLike
        Vector containg the amplitude of each commanded mode.
    modesVector : int | ArrayLike
        Vector containing the list of commanded modes.
    template : int | ArrayLike
        Sampling template ampplied on each mode.
    indexList : int | ArrayLike
        Indexing of the modes inside the commanded matrix.
    registrationActs : int | ArrayLike
        Vector containing the commanded actuators for the registration.
    shuffle : int
        Shuffle information. If it's nor 0, the values indicates the number of
        template sampling repetition for each mode.
    """
    base = _os.path.join(_ifFold, tn)
    ampVector = _osu.load_fits(_os.path.join(base, ampVecFile))
    template = _osu.load_fits(_os.path.join(base, templateFile))
    modesVector = _osu.load_fits(_os.path.join(base, modesVecFile))
    indexList = _osu.load_fits(_os.path.join(base, indexListFile))
    registrationActs = _osu.load_fits(_os.path.join(base, regisActFile))
    with open(_os.path.join(base, shuffleFile), "r", encoding="UTF-8") as shf:
        shuffle = int(shf.read())
    return ampVector, modesVector, template, indexList, registrationActs, shuffle


def _getAcqInfo(
    tn: str = None,
) -> tuple[
    dict[str, _ot.Any], dict[str, _ot.Any], dict[str, _ot.Any], dict[str, _ot.Any]
]:
    """
    Returns the information read from the iffConfig.ini file.

    Parameters
    ----------
    tn : str, optional
        Tracking number of the data in the IFFunctions folder. The default is None,
        which points to configuration root folder.

    Returns
    -------
    infoT : dict
        Information read about the TRIGGER options.
    infoR : dict
        Information read about the REGISTRATION options.
    infoIF : dict
        Information read about the IFFUNC option.
    """
    path = _os.path.join(_ifFold, tn) if tn is not None else _fn.CONFIGURATION_FOLDER
    infoT = _rif.getIffConfig("TRIGGER", bpath=path)
    infoR = _rif.getIffConfig("REGISTRATION", bpath=path)
    infoIF = _rif.getIffConfig("IFFUNC", bpath=path)
    infoDM = _rif.getDmIffConfig(bpath=path)
    return infoT, infoR, infoIF, infoDM


def _checkStackedCubes(tnlist: str) -> dict[str, _ot.Any]:
    """
    Inspect the cubes to stack, to check whether there are shared modes, or not.

    Parameters
    ----------
    tnlist : list of str
        List containing the tracking number of the cubes to stack.

    Returns
    -------
    flag : dict
        Dictionary containing the flagging information about the stacked cube,
        to be later dump into the 'flag.txt' file.
    """
    _, _, modesVectList, rebin = _getCubeList(tnlist)
    nmodes = len(modesVectList[0])
    nvects = len(modesVectList)
    for i in range(nvects):
        for j in range(i + 1, nvects):
            common_modes = set(modesVectList[i]).intersection(modesVectList[j])
    c_nmodes = len(common_modes)
    if c_nmodes in range(1, nmodes):
        flag = __flag(tnlist, modesVectList, rebin, 2)
    elif c_nmodes == nmodes:
        flag = __flag(tnlist, modesVectList, rebin, 1)
    else:
        flag = __flag(tnlist, modesVectList, rebin, 0)
    return flag


def __flag(
    tnlist: list[str],
    modesVectList: list[int] | list[_ot.ArrayLike],
    rebin: int,
    type: int,
) -> dict[str, _ot.Any]:
    """
    Creates the dictionary to dump into the 'flag.txt' file accordingly to
    sequentially stacked cubes with no repeated modes.

    Parameters
    ----------
    tnlist : list of str
        List containing the tracking number of the cubes to stack.
    modesVectList : list of ndarray
        A list containing the modes vectors for each cube.
    type : int
        Type of stacked cube created.
        0 for sequential, 1 for mean, 2 for shared modes.

    Returns
    -------
    config : dict
        Dictionary containing the flagging information about the stacked cube.
    """
    c_type = [
        "sequential stack",
        "mean stack",
        "(WARN)repeated modes",
    ]
    text = ""
    for i, tn in enumerate(tnlist):
        if _np.array_equal(
            modesVectList[i],
            _np.arange(modesVectList[i][0], modesVectList[i][-1] + 1, 1),
        ):
            text += f"""
{tn}, modes {modesVectList[i][0]} to {modesVectList[i][-1]}"""
        else:
            text += f"""
{tn}, modes {list(modesVectList[i])}"""
    flag = {
        "Flag": {
            "Rebin": str(rebin),
            "Cube type": c_type[type],
            "Source cubes": text,
        }
    }
    _config["Flag"] = {}
    for key, value in flag["Flag"].items():
        _config["Flag"][key] = value
    return _config


# TODO
def _ampReorganization(ampVector: _ot.ArrayLike):
    reorganizaed_amps = ampVector
    return reorganizaed_amps


# TODO
def _modesReorganization(modesVector: _ot.ArrayLike):
    # if isinstance(modesVector, _np.ndarray):
    #     modesVector = modesVector.astype(int)
    # else:
    #     modesVector = _np.asarray(modesVector)
    reorganizaed_modes = modesVector
    return reorganizaed_modes
