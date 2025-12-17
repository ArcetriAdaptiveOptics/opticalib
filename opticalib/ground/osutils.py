"""
Module containing various utility functions for handling files and directories,
especially related to tracking numbers and interferometric data, within the
Opticalib framework.

Author(s)
---------
- Chiara Selmi:  written in 2019
- Pietro Ferraiuolo: updated in 2025

"""

import os as _os
import h5py as _h5
import numpy as _np
import time as _time
import h5py as _h5py
from numpy import uint8 as _uint8
from astropy.io import fits as _fits
from opticalib import typings as _ot
from numpy.ma import masked_array as _masked_array
from opticalib.core import fitsarray as _fa
from opticalib.core import root as _fn

_OPTDATA = _fn.OPT_DATA_ROOT_FOLDER


def is_tn(string: str) -> bool:
    """
    Check if a given string is a valid tracking number or the full path
    of a tracking number.

    Parameters
    ----------
    string : str
        The string to check.

    Returns
    -------
    bool
        True if the string is a valid tracking number, False otherwise.
    """
    if len(string) != 15:
        return False
    date_part = string[:8]
    time_part = string[9:]
    if string[8] != "_":
        return False
    if not (date_part.isdigit() and time_part.isdigit()):
        return False
    try:
        _time.strptime(date_part + time_part, "%Y%m%d%H%M%S")
        return True
    except ValueError:
        return False


def findTracknum(tn: str, complete_path: bool = False) -> str | list[str]:
    """
    Search for the tracking number given in input within all the data path subfolders.

    Parameters
    ----------
    tn : str
        Tracking number to be searched.
    complete_path : bool, optional
        Option for wheter to return the list of full paths to the folders which
        contain the tracking number or only their names.

    Returns
    -------
    tn_path : list of str
        List containing all the folders (within the OPTData path) in which the
        tracking number is present, sorted in alphabetical order.
    """
    tn_path = []
    for fold in _os.listdir(_OPTDATA):
        search_fold = _os.path.join(_OPTDATA, fold)
        if not _os.path.isdir(search_fold):
            continue
        if tn in _os.listdir(search_fold):
            if complete_path:
                tn_path.append(_os.path.join(search_fold, tn))
            else:
                tn_path.append(fold)
    path_list = sorted(tn_path)
    if len(path_list) == 1:
        path_list = path_list[0]
    return path_list


def getFileList(tn: str = None, fold: str = None, key: str = None) -> list[str]:
    """
    Search for files in a given tracking number or complete path, sorts them and
    puts them into a list.

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.
    fold : str, optional
        Folder in which searching for the tracking number. If None, the default
        folder is the OPD_IMAGES_ROOT_FOLDER.
    key : str, optional
        A key which identify specific files to return

    Returns
    -------
    fl : list of str
        List of sorted files inside the folder.

    How to Use it
    -------------
    If the complete path for the files to retrieve is available, then this function
    should be called with the 'fold' argument set with the path, while 'tn' is
    defaulted to None.

    In any other case, the tn must be given: it will search for the tracking
    number into the OPDImages folder, but if the search has to point another
    folder, then the fold argument comes into play again. By passing both the
    tn (with a tracking number) and the fold argument (with only the name of the
    folder) then the search for files will be done for the tn found in the
    specified folder. Hereafter there is an example, with the correct use of the
    key argument too.

    Examples
    --------

    Here are some examples regarding the use of the 'key' argument. Let's say we
    need a list of files inside ''tn = '20160516_114916' '' in the IFFunctions
    folder.

    ```python
    iffold = 'IFFunctions'
    tn = '20160516_114916'
    getFileList(tn, fold=iffold)
    ['.../OPTData/IFFunctions/20160516_114916/cmdMatrix.fits',
     '.../OPTData/IFFunctions/20160516_114916/mode_0000.fits',
     '.../OPTData/IFFunctions/20160516_114916/mode_0001.fits',
     '.../OPTData/IFFunctions/20160516_114916/mode_0002.fits',
     '.../OPTData/IFFunctions/20160516_114916/mode_0003.fits',
     '.../OPTData/IFFunctions/20160516_114916/modesVector.fits']
    ```

    Let's suppose we want only the list of 'mode_000x.fits' files:

    ```python
    getFileList(tn, fold=iffold, key='mode_')
    ['.../OPTData/IFFunctions/20160516_114916/mode_0000.fits',
     '.../OPTData/IFFunctions/20160516_114916/mode_0001.fits',
     '.../OPTData/IFFunctions/20160516_114916/mode_0002.fits',
     '.../OPTData/IFFunctions/20160516_114916/mode_0003.fits']
    ```

    Notice that, in this specific case, it was necessary to include the underscore
    after 'mode' to exclude the 'modesVector.fits' file from the list.
    """
    if tn is None and fold is not None:
        fl = sorted([_os.path.join(fold, file) for file in _os.listdir(fold)])
    else:
        try:
            paths = findTracknum(tn, complete_path=True)
            if isinstance(paths, str):
                paths = [paths]
            for path in paths:
                if fold is None:
                    fl = []
                    fl.append(
                        sorted(
                            [_os.path.join(path, file) for file in _os.listdir(path)]
                        )
                    )
                elif fold in path.split("/")[-2]:
                    fl = sorted(
                        [_os.path.join(path, file) for file in _os.listdir(path)]
                    )
                else:
                    continue
        except Exception as exc:
            raise FileNotFoundError(
                f"Invalid Path: no data found for tn '{tn}'"
            ) from exc
    if len(fl) == 1:
        fl = fl[0]
    if key is not None:
        try:
            selected_list = []
            for file in fl:
                if key in file.split("/")[-1]:
                    selected_list.append(file)
        except TypeError as err:
            raise TypeError("'key' argument must be a string") from err
        fl = selected_list
    if len(fl) == 1:
        fl = fl[0]
    return fl


def tnRange(tn0: str, tn1: str, complete_paths: bool = False) -> list[str]:
    """
    Returns the list of tracking numbers between tn0 and tn1, within the same
    folder, if they both exist in it.

    Parameters
    ----------
    tn0 : str
        Starting tracking number.
    tn1 : str
        Finish tracking number.
    complete_paths : bool, optional
        Whether to return the full path of the tracking numbers or only their names.

    Returns
    -------
    tnMat : list of str
        A list or a matrix of tracking number in between the start and finish ones.

    Raises
    ------
    FileNotFoundError
        An exception is raised if the two tracking numbers are not found in the same folder
    """
    tn0_fold = findTracknum(tn0)
    tn1_fold = findTracknum(tn1)
    if isinstance(tn0_fold, str):
        tn0_fold = [tn0_fold]
    if isinstance(tn1_fold, str):
        tn1_fold = [tn1_fold]
    if len(tn0_fold) == 1 and len(tn1_fold) == 1:
        if tn0_fold[0] == tn1_fold[0]:
            fold_path = _os.path.join(_OPTDATA, tn0_fold[0])
            tn_folds = sorted(_os.listdir(fold_path))
            id0 = tn_folds.index(tn0)
            id1 = tn_folds.index(tn1)
            if complete_paths:
                tnMat = [_os.path.join(fold_path, tn) for tn in tn_folds[id0 : id1 + 1]]
            else:
                tnMat = tn_folds[id0 : id1 + 1]
        else:
            raise FileNotFoundError("The tracking numbers are in different foldes")
    else:
        tnMat = []
        for ff in tn0_fold:
            if ff in tn1_fold:
                fold_path = _os.path.join(_OPTDATA, ff)
                tn_folds = sorted(_os.listdir(fold_path))
                id0 = tn_folds.index(tn0)
                id1 = tn_folds.index(tn1)
                if not complete_paths:
                    tnMat.append(tn_folds[id0 : id1 + 1])
                else:
                    tnMat.append(
                        [_os.path.join(fold_path, tn) for tn in tn_folds[id0 : id1 + 1]]
                    )
    return tnMat


def loadCubeFromFilelist(
    tn_or_fl: str, fold: _ot.Optional[str] = None, key: _ot.Optional[str] = None
) -> _ot.CubeData:
    """
    Loads a cube from a list of files obtained from a tracking number or a folder.

    Parameters
    ----------
    tn_or_fl : str
        Either the filelist of the data to be put into the cube, or the tracking
        number. In the second case, the filelist is obtained searching for the
        tracking number, for which the additional parameters `fold` and `key` can
        be used (see the `getFileList` function).
    fold : str, optional
        Folder in which searching for the tracking number.
    key : str, optional
        A key which identify specific files to load.

    Returns
    -------
    cube : CubeData
        Cube containing all the images loaded from the files.
    """
    from ..analyzer import createCube

    if is_tn(tn_or_fl):
        if fold is None:
            raise ValueError(
                "When passing a tracking number, the 'fold' argument must be specified"
            )
        path = findTracknum(tn_or_fl, complete_path=True)
        if isinstance(path, str):
            path = [path]
        for p in path:
            if fold in p:
                fold = p
                break
        fl = getFileList(fold=fold, key=key)
    else:
        fl = tn_or_fl
    cube = createCube(fl)
    return cube


def read_phasemap(file_path: str) -> _ot.ImageData:
    """
    Function to read interferometric data, in the three possible formats
    (FITS, 4D, H5)

    Parameters
    ----------
    file_path: str
        Complete filepath of the file to load.

    Returns
    -------
    image: ImageData
        Image as a masked array.
    """
    ext = file_path.split(".")[-1]
    if ext in ["fits", "4Ds"]:
        image = load_fits(file_path)
    elif ext in ["4D", "h5"]:
        image = _InterferometerConverter.fromPhaseCam6110(file_path)
    return image


def load_fits(
    filepath: str, on_gpu: bool = False
) -> _ot.FitsData:
    """
    Loads a FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.
    on_gpu : bool, optional
        Whether to load the data on GPU as a `cupy.ndarray` or a
        `xupy.ma.MaskedArray` (if masked). Default is False.

    Returns
    -------
    fit : ArrayLike
        The loaded FITS file data (masked) array, on CPU or GPU, with attached header
        (as Fits<...>Array).
    """
    with _fits.open(filepath) as hdul:
        fit = hdul[0].data
        header = hdul[0].header
        if (len(hdul) > 1 and len(hdul) < 3) and hasattr(hdul[1], "data"):
            mask = hdul[1].data.astype(bool)
            fit = _masked_array(fit, mask=mask)
        elif len(hdul) > 2:
            header = [hdu.header for hdu in hdul if hasattr(hdu, "header")]
            fit = [hdu.data for hdu in hdul if hasattr(hdu, "data")]
            if on_gpu:
                raise NotImplementedError(
                    "Loading multi-extension FITS files on GPU is not supported."
                )
    if on_gpu:
        import xupy as _xu

        if isinstance(fit, _masked_array):
            fit = _xu.ma.MaskedArray(fit)
        else:
            fit = _xu.asarray(fit)

    out = _fa.fits_array(fit, header=header)
    return out


def save_fits(
    filepath: str,
    data: _ot.ImageData | _ot.CubeData | _ot.MatrixLike | _ot.ArrayLike | _ot.Any,
    overwrite: bool = True,
    header: dict[str, _ot.Any] | _fits.Header = None,
) -> None:
    """
    Saves a FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.
    data : ArrayLike
        Data to be saved.
    overwrite : bool, optional
        Whether to overwrite an existing file. Default is True.
    header : dict[str, any] | fits.Header, optional
        Header information to include in the FITS file. Can be a dictionary or
        a fits.Header object.
    """
    data = _ensure_on_cpu(data)

    # Check if lowering precision is safe
    if data.dtype == _np.float64:
        data = _reduce_dtype_safely(data, preserve_float64=False)
    elif any([data.dtype == _np.int64, data.dtype == _np.int32]):
        data = _reduce_dtype_safely(data)

    if isinstance(data, (_fa.FitsArray, _fa.FitsMaskedArray)):
        data.writeto(filepath, overwrite=overwrite)
        return

    # Prepare the header
    if header is not None:
        header = _header_from_dict(header)

    # Save the FITS file
    if isinstance(data, _masked_array):
        _fits.writeto(filepath, data.data, header=header, overwrite=overwrite)
        if not data.mask is _np.ma.nomask:
            _fits.append(filepath, data.mask.astype(_uint8))
    else:
        _fits.writeto(filepath, data, header=header, overwrite=overwrite)


def save_dict(
    datadict: dict[str, _ot.ArrayLike],
    filepath: str,
    overwrite: bool = True,
    reduce_precision: bool = True,
) -> None:
    """
    Saves a dictionary of data arrays to an HDF5 file.

    Each key-value pair in the dictionary is stored as a separate dataset
    in the HDF5 file, where the key becomes the dataset name and the value
    is stored as the dataset.

    Parameters
    ----------
    datadict: dict[str, _ot.ArrayLike]
        Dictionary where keys are strings and values are data arrays to be saved.
    filepath: str
        Full path where to save the HDF5 file. Should end with '.h5' or '.hdf5'
    overwrite: bool, optional
        If True, overwrites existing file. Default is True.

    Raises
    ------
    FileExistsError
        If file exists and overwrite is False

    Examples
    --------
    >>> datadict = {
    ...     'sensor_1': np.random.randn(111, 100),
    ...     'sensor_2': np.random.randn(111, 100),
    ... }
    >>> save_dict(datadict, '/path/to/data.h5')
    """
    if not filepath.endswith((".h5", ".hdf5")):
        filepath += ".h5"

    if _os.path.exists(filepath) and not overwrite:
        raise FileExistsError(f"File {filepath} already exists and overwrite=False")

    with _h5.File(filepath, "w") as hf:
        # Store metadata
        hf.attrs["n_keys"] = len(datadict)
        hf.attrs["creation_date"] = newtn()

        # Store each array as a dataset
        for key, data in datadict.items():
            if not isinstance(data, _np.ndarray):
                data = _np.asarray(data)

            original_dtype = data.dtype.type

            # Reduce precision if requested
            if reduce_precision:
                data, conversion_info = _reduce_dtype_safely(
                    data, preserve_float64=False, return_info=True
                )
            else:
                conversion_info = None

            # Create dataset with compression for better storage efficiency
            hf.create_dataset(
                key,
                data=data,
                compression="gzip",
                compression_opts=4,  # Compression level 0-9
                chunks=True,  # Enable chunking for better I/O performance
            )

            # Store shape information as attributes for verification
            hf[key].attrs["shape"] = data.shape
            hf[key].attrs["dtype"] = str(data.dtype.type)
            hf[key].attrs["original_dtype"] = str(original_dtype)

            if conversion_info is not None:
                for attr_key, attr_value in conversion_info.items():
                    hf[key].attrs[attr_key] = attr_value


def load_dict(
    filepath: str, keys: _ot.Optional[list[str]] = None
) -> dict[str, _ot.ArrayLike]:
    """
    Loads a dictionary of data arrays from an HDF5 file.

    Parameters
    ----------
    filepath: str
        Full path to the HDF5 file to load
    keys: list[str], optional
        If provided, only loads the specified keys. Otherwise loads all datasets.
        This is useful for memory management when dealing with large files.

    Returns
    -------
    dict: dict[str, _ot.ArrayLike]
        Dictionary with keys as dataset names and values as numpy arrays

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    KeyError
        If a requested key is not found in the file

    Examples
    --------
    >>> # Load all data
    >>> buffer_dict = loadBufferDataDict('/path/to/buffer_data.h5')
    >>>
    >>> # Load only specific keys (memory efficient)
    >>> datadict = load_dict(
    ...     '/path/to/data.h5',
    ...     keys=['sensor_1', 'sensor_3']
    ... )
    """
    if not filepath.endswith((".h5", ".hdf5")):
        filepath += ".h5"

    if not _os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")

    datadict = {}

    with _h5.File(filepath, "r") as hf:
        # Get list of keys to load
        if keys is None:
            keys_to_load = list(hf.keys())
        else:
            # Verify requested keys exist
            missing_keys = set(keys) - set(hf.keys())
            if missing_keys:
                raise KeyError(f"Keys not found in file: {missing_keys}")
            keys_to_load = keys

        # Load each dataset
        for key in keys_to_load:
            datadict[key] = hf[key][:]  # [:] loads data into memory

            # Verify shape if needed (for debugging)
            expected_shape = hf[key].attrs.get("shape", None)
            if expected_shape is not None:
                actual_shape = tuple(datadict[key].shape)
                if actual_shape != tuple(expected_shape):
                    print(
                        f"Warning: Shape mismatch for key '{key}': "
                        f"expected {expected_shape}, got {actual_shape}"
                    )

    return datadict


def get_h5file_info(filepath: str) -> dict[str, _ot.Any]:
    """
    Retrieves metadata and information about the dict data file without loading
    the full datasets into memory.

    This is useful for inspecting large HDF5 files without memory overhead.

    Parameters
    ----------
    filepath: str
        Full path to the HDF5 file

    Returns
    -------
    info: dict[str, Any]
        Dictionary containing:
        - 'keys': list of dataset names
        - 'n_keys': number of datasets
        - 'creation_date': tracking number when file was created
        - 'shapes': dict mapping each key to its array shape
        - 'dtypes': dict mapping each key to its data type
        - 'file_size_mb': file size in megabytes

    Examples
    --------
    >>> info = get_dict_info('/path/to/data.h5')
    >>> print(f"File contains {info['n_keys']} datasets")
    >>> print(f"Keys: {info['keys']}")
    >>> print(f"Shapes: {info['shapes']}")
    """
    if not filepath.endswith((".h5", ".hdf5")):
        filepath += ".h5"

    if not _os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")

    info = {
        "keys": [],
        "shapes": {},
        "dtypes": {},
    }

    with _h5.File(filepath, "r") as hf:
        # Get file-level metadata
        info["n_keys"] = hf.attrs.get("n_keys", len(hf.keys()))
        info["creation_date"] = hf.attrs.get("creation_date", "unknown")

        # Get dataset information
        for key in hf.keys():
            info["keys"].append(key)
            info["shapes"][key] = tuple(hf[key].shape)
            info["dtypes"][key] = str(hf[key].dtype.type)

    # Get file size
    file_size_bytes = _os.path.getsize(filepath)
    info["file_size_mb"] = file_size_bytes / (1024 * 1024)
    return info


def newtn() -> str:
    """
    Returns a timestamp in a string of the format `YYYYMMDD_HHMMSS`.

    Returns
    -------
    str
        Current time in a string format.
    """
    return _time.strftime("%Y%m%d_%H%M%S")

def create_data_folder(basepath: str = _fn.OPD_IMAGES_ROOT_FOLDER) -> str:
    """
    Creates a new data folder with a unique tracking number in the specified base path.
    
    Parameters
    ----------
    basepath : str, optional
        The base directory where the new tracking number folder will be created.
        Default is the OPTImages root folder.
    
    Returns
    -------
    tn_path : str
        The path to the newly created tracking number folder.
    """
    tn = newtn()
    tn_path = _os.path.join(basepath, tn)
    _os.makedirs(tn_path, exist_ok=True)
    return tn_path


def _header_from_dict(
    dictheader: dict[str, _ot.Any | tuple[_ot.Any, str]],
) -> _fits.Header:
    """
    Converts a dictionary to an astropy.io.fits.Header object.

    Parameters
    ----------
    dictheader : dict
        Dictionary containing header information. Each key should be a string,
        and the value can be a tuple of length 2, where the first element is the
        value and the second is a comment.

    Returns
    -------
    header : astropy.io.fits.Header
        The converted FITS header object.
    """
    if isinstance(dictheader, _fits.Header):
        return dictheader
    header = _fits.Header()
    for key, value in dictheader.items():
        if isinstance(value, tuple) and len(value) > 2:
            raise ValueError(
                "Header values must be a tuple of length 2 or less, "
                "where the first element is the value and the second is the comment."
                f"{value}"
            )
        else:
            header[key] = value
    return header


def _ensure_on_cpu(data: _ot.ArrayLike) -> _ot.ArrayLike:
    """
    Ensures that the input data is on the CPU as a NumPy array or masked array.

    Parameters
    ----------
    data : ArrayLike
        Input data which may be on GPU or CPU. Handles:
        - numpy arrays / masked arrays (CPU)
        - xupy arrays / masked arrays (GPU)
        - FitsArray and FitsMaskedArray (CPU)
        - FitsArrayGpu and FitsMaskedArrayGpu (GPU)

    Returns
    -------
    ArrayLike
        Data ensured to be on CPU as a NumPy array or masked array.
    """
    try:
        import xupy as _xu

        if _xu.on_gpu:

            # Handle both FitsMA and plain MA (on gpu)
            if isinstance(data, _xu.ma.MaskedArray):
                data_cpu = data.asmarray()
                return data_cpu

            # Handling normal Arrays/FitsArrays (on gpu)
            elif isinstance(data, _xu.ndarray):
                if isinstance(data, _fa.FitsArrayGpu):
                    data_cpu = _fa.fits_array(
                        _xu.asnumpy(data), header=data.header.copy()
                    )
                    return data_cpu
                else:
                    return _xu.asnumpy(data)

            # fallback for cpu data
            elif "numpy" in str(type(data)):
                return data
        else:
            raise ImportError
    # extra safety
    except ImportError:
        return data
    return data


def _reduce_dtype_safely(
    data: _ot.ArrayLike, preserve_float64: bool = True, return_info: bool = False
) -> tuple[_ot.ArrayLike, dict[str, _ot.Any]]:
    """
    Reduces the dtype of an array to save space, with safety checks.

    This function performs intelligent dtype reduction:
    - Checks if data range fits in smaller dtype
    - Estimates precision loss for floating point conversions
    - Preserves float64 if precision loss is significant

    Parameters
    ----------
    data: np.ndarray
        Input array to reduce precision
    preserve_float64: bool, optional
        If True, keeps float64 when conversion would cause significant
        precision loss. Default is True.

    Returns
    -------
    reduced_data: np.ndarray
        Array with reduced precision dtype
    info: dict[str, Any], optional
        Information about the conversion:
        - 'conversion': description of the conversion performed
        - 'precision_loss': estimated relative precision loss (for floats)
        - 'space_saving_ratio': ratio of reduced size to original

    Notes
    -----
    Precision loss is estimated as the relative error in representing
    the maximum absolute value in the new dtype.
    """
    original_dtype = data.dtype.type
    reduced_data = data
    conversion_info = {
        "conversion": "none",
        "precision_loss": 0.0,
        "space_saving_ratio": 1.0,
    }

    # Handle floating point types
    if original_dtype == _np.float64:
        # Check if we can safely convert to float32
        if data.size > 0:
            max_val = _np.abs(data).max()
            min_val = _np.abs(data[data != 0]).min() if _np.any(data != 0) else 0

            # float32 has ~7 significant decimal digits
            # Check dynamic range: can float32 represent both max and min?
            float32_max = _np.finfo(_np.float32).max
            float32_min = _np.finfo(_np.float32).tiny

            can_convert = max_val < float32_max * 0.9 and (
                min_val == 0 or min_val > float32_min * 10
            )

            if can_convert:
                # Estimate precision loss
                test_val = data.flat[data.size // 2]  # Middle value
                test_float32 = _np.float32(test_val)
                if test_val != 0:
                    rel_error = abs(test_float32 - test_val) / abs(test_val)
                else:
                    rel_error = 0

                # Only convert if precision loss is acceptable
                if not preserve_float64 or rel_error < 1e-6:
                    reduced_data = data.astype(_np.float32)
                    conversion_info["conversion"] = "float64 → float32"
                    conversion_info["precision_loss"] = float(rel_error)
                    conversion_info["space_saving_ratio"] = 0.5
                else:
                    conversion_info["conversion"] = "float64 preserved (precision)"
        else:
            # Empty array, safe to convert
            reduced_data = data.astype(_np.float32)
            conversion_info["conversion"] = "float64 → float32 (empty array)"
            conversion_info["space_saving_ratio"] = 0.5

    # Handle integer types
    elif original_dtype == _np.int64:
        if data.size > 0:
            min_val, max_val = data.min(), data.max()

            # Try int32 first
            if (
                _np.iinfo(_np.int32).min <= min_val
                and max_val <= _np.iinfo(_np.int32).max
            ):
                reduced_data = data.astype(_np.int32)
                conversion_info["conversion"] = "int64 → int32"
                conversion_info["space_saving_ratio"] = 0.5

                # Try int16 if range allows
                if (
                    _np.iinfo(_np.int16).min <= min_val
                    and max_val <= _np.iinfo(_np.int16).max
                ):
                    reduced_data = data.astype(_np.int16)
                    conversion_info["conversion"] = "int64 → int16"
                    conversion_info["space_saving_ratio"] = 0.25
        else:
            reduced_data = data.astype(_np.int32)
            conversion_info["conversion"] = "int64 → int32 (empty array)"
            conversion_info["space_saving_ratio"] = 0.5

    elif original_dtype == _np.int32:
        if data.size > 0:
            min_val, max_val = data.min(), data.max()
            if (
                _np.iinfo(_np.int16).min <= min_val
                and max_val <= _np.iinfo(_np.int16).max
            ):
                reduced_data = data.astype(_np.int16)
                conversion_info["conversion"] = "int32 → int16"
                conversion_info["space_saving_ratio"] = 0.5
        else:
            reduced_data = data.astype(_np.int16)
            conversion_info["conversion"] = "int32 → int16 (empty array)"
            conversion_info["space_saving_ratio"] = 0.5

    if return_info:
        return reduced_data, conversion_info
    else:
        return reduced_data


class _InterferometerConverter:
    """
    This class is crucial to convert H5 files into masked array
    """

    @staticmethod
    def fromPhaseCam4020(h5filename: str) -> _ot.ImageData:
        """
        Function for PhaseCam4020

        Parameters
        ----------
        h5filename: string
            Path of the h5 file to convert

        Returns
        -------
        ima: numpy masked array
            Masked array image
        """
        file = _h5py.File(h5filename, "r")
        genraw = file["measurement0"]["genraw"]["data"]
        data = _np.array(genraw)
        mask = _np.zeros(data.shape, dtype=bool)
        mask[_np.where(data == data.max())] = True
        ima = _np.ma.masked_array(data * 632.8e-9, mask=mask)
        return ima

    @staticmethod
    def fromPhaseCam6110(i4dfilename: str) -> _ot.ImageData:
        """
        Function for PhaseCam6110

        Parameters
        ----------
        i4dfilename: string
            Path of the 4D file to convert

        Returns
        -------
        ima: numpy masked array
            Masked array image
        """
        with _h5py.File(i4dfilename, "r") as ff:
            data = ff.get("/Measurement/SurfaceInWaves/Data")
            meas = data[()]
            mask = _np.invert(_np.isfinite(meas))
        image = _np.ma.masked_array(meas * 632.8e-9, mask=mask, dtype=_np.float32)
        return image

    @staticmethod
    def fromFakeInterf(filename: str) -> _ot.ImageData:
        """
        Function for fake interferometer

        Parameters
        ----------
        filename: string
            Path name for data

        Returns
        -------
        ima: numpy masked array
            Masked array image
        """
        masked_ima = load_fits(filename)
        return masked_ima

    @staticmethod
    def fromI4DToSimplerData(i4dname: str, folder: str, h5name: str) -> str:
        """
        Function for converting files from 4D 6110 files to H5 files

        Parameters
        ----------
        i4dname: string
            File name path of 4D data
        folder: string
            Folder path for new data
        h5name: string
            Name for H5 data

        Returns
        -------
        file_name: string
            Final path name
        """
        file = _h5py.File(i4dname, "r")
        data = file.get("/Measurement/SurfaceInWaves/Data")
        file_name = _os.path.join(folder, h5name)
        hf = _h5py.File(file_name, "w")
        hf.create_dataset("Data", data=data)
        return file_name
