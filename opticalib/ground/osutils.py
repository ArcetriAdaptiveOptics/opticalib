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
    filepath: str, return_header: bool = False, on_gpu: bool = False
) -> tuple[_ot.ImageData | _ot.CubeData | _ot.MatrixLike | _ot.ArrayLike, _ot.Any]:
    """
    Loads a FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.
    return_header: bool
        Wether to return the header of the loaded fits file. Default is False.
    on_gpu : bool, optional
        Whether to load the data on GPU as a `cupy.ndarray` or a
        `xupy.ma.MaskedArray` (if masked). Default is False.

    Returns
    -------
    fit : np.ndarray or np.ma.MaskedArray of cupy.ndarray or xupy.ma.MaskedArray
        The loaded FITS file data.
    header : dict | fits.Header, optional
        The header of the loaded fits file.
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
    if return_header:
        out = _fa.fits_array(fit, header=header)
    else:
        out = fit
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
    # If data is a python dictionary:
    if isinstance(data, dict):
        import pandas as _pd
        from astropy.table import Table

        df = _pd.DataFrame(data)
        table = Table.from_pandas(df)
        table.write(filepath, format="fits", overwrite=True)

    else:
        data = _ensure_on_cpu(data)
        # force float32 dtype on save
        if data.dtype != _np.float32:
            data = _np.asanyarray(data, dtype=_np.float32)
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
    datadict: dict[str,_ot.ArrayLike],
    filepath: str,
    overwrite: bool = True
) -> None:
    """
    Saves a dictionary of buffer data arrays to an HDF5 file.
    
    Each key-value pair in the dictionary is stored as a separate dataset
    in the HDF5 file, where the key becomes the dataset name and the value
    (expected to be a numpy array of shape (111, N)) is stored as the dataset.
    
    Parameters
    ----------
    buffer_dict: dict[str, np.ndarray]
        Dictionary where keys are strings and values are numpy arrays of shape (111, N)
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
    >>> buffer_dict = {
    ...     'sensor_1': np.random.randn(111, 100),
    ...     'sensor_2': np.random.randn(111, 100),
    ... }
    >>> saveBufferDataDict(buffer_dict, '/path/to/buffer_data.h5')
    """
    if not filepath.endswith(('.h5', '.hdf5')):
        filepath += '.h5'
    
    if _os.path.exists(filepath) and not overwrite:
        raise FileExistsError(f"File {filepath} already exists and overwrite=False")
    
    with _h5.File(filepath, 'w') as hf:
        # Store metadata
        hf.attrs['n_keys'] = len(datadict)
        hf.attrs['creation_date'] = newtn()
        
        # Store each array as a dataset
        for key, data in datadict.items():
            if not isinstance(data, _np.ndarray):
                data = _np.asarray(data)
            
            # Create dataset with compression for better storage efficiency
            hf.create_dataset(
                key,
                data=data,
                compression='gzip',
                compression_opts=4,  # Compression level 0-9
                chunks=True  # Enable chunking for better I/O performance
            )
            
            # Store shape information as attributes for verification
            hf[key].attrs['shape'] = data.shape
            hf[key].attrs['dtype'] = str(data.dtype)


def load_dict(
    filepath: str,
    keys: _ot.Optional[list[str]] = None
) -> dict[str,_ot.ArrayLike]:
    """
    Loads buffer data from an HDF5 file into a dictionary.
    
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
    >>> buffer_dict = loadBufferDataDict(
    ...     '/path/to/buffer_data.h5',
    ...     keys=['sensor_1', 'sensor_3']
    ... )
    """
    if not _os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    
    datadict = {}
    
    with _h5.File(filepath, 'r') as hf:
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
            expected_shape = hf[key].attrs.get('shape', None)
            if expected_shape is not None:
                actual_shape = tuple(datadict[key].shape)
                if actual_shape != tuple(expected_shape):
                    print(f"Warning: Shape mismatch for key '{key}': "
                          f"expected {expected_shape}, got {actual_shape}")
    
    return dict


def get_h5file_info(filepath: str) -> dict[str, _ot.Any]:
    """
    Retrieves metadata and information about the buffer data file without loading
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
    >>> info = getBufferDataInfo('/path/to/buffer_data.h5')
    >>> print(f"File contains {info['n_keys']} datasets")
    >>> print(f"Keys: {info['keys']}")
    >>> print(f"Shapes: {info['shapes']}")
    """
    if not _os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    
    info = {
        'keys': [],
        'shapes': {},
        'dtypes': {},
    }
    
    with _h5.File(filepath, 'r') as hf:
        # Get file-level metadata
        info['n_keys'] = hf.attrs.get('n_keys', len(hf.keys()))
        info['creation_date'] = hf.attrs.get('creation_date', 'unknown')
        
        # Get dataset information
        for key in hf.keys():
            info['keys'].append(key)
            info['shapes'][key] = tuple(hf[key].shape)
            info['dtypes'][key] = str(hf[key].dtype)
    
    # Get file size
    file_size_bytes = _os.path.getsize(filepath)
    info['file_size_mb'] = file_size_bytes / (1024 * 1024)
    
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
        Input data which may be on GPU or CPU.

    Returns
    -------
    ArrayLike
        Data ensured to be on CPU as a NumPy array or masked array.
    """
    try:
        import xupy as _xu

        if _xu.on_gpu:
            if isinstance(data, _xu.ma.MaskedArray):
                data_cpu = data.asmarray()
                return data_cpu
            elif isinstance(data, _xu.ndarray):
                return _xu.asnumpy(data)
            elif "numpy" in str(type(data)):
                return data
        else:
            raise ImportError
    except ImportError:
        return data
    return data


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
