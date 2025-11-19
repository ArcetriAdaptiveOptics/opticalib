import os
import xupy as xp
import numpy as np
from ... import typings as _t
from .factory_functions import *
from ...core import root as _root
from abc import ABC, abstractmethod
from opticalib.ground import osutils as osu

######################################
## Utility classes for creating the ##
##       simulated Alpao            ##
######################################


def generate_zernike_matrix(noll_ids: list[int], img_mask: _t.ImageData, scale_length: float = None):
    """
    Generates the interaction matrix of the Zernike modes with Noll index
    in noll_ids on the mask in input

    Parameters
    ----------
    noll_ids : ndarray(int) [Nzern,]
        Array of Noll indices to fit.
    img_mask : matrix bool
        Mask of the desired image.
    scale_length : float, optional
        The scale length to use for the Zernike fit.
        The default is the maximum of the image mask shape.

    Returns
    -------
    ZernMat : ndarray(float) [Npix,Nzern]
        The Zernike interaction matrix of the given indices on the given mask.
    """
    n_pix = np.sum(1 - img_mask)
    if isinstance(noll_ids, int):
        noll_ids = np.arange(1, noll_ids + 1, 1)
    n_zern = len(noll_ids)
    ZernMat = np.zeros([n_pix, n_zern])
    for i in range(n_zern):
        ZernMat[:, i] = _project_zernike_on_mask(noll_ids[i], img_mask, scale_length)
    return ZernMat


def _project_zernike_on_mask(noll_number: int, mask, scale_length: float = None):
    """
    Project the Zernike polynomials identified by the Noll number in input
    on a given mask.
    The polynomials are computed on the circle inscribed in the mask by default,
    or on a circle of radius scale_length if the corresponding input is given
    Masked data is then normalized as follows:
    data = ma.data[~ma.mask], data = (data - mean(data))/std(data)

    Parameters
    ----------
    noll_number : int
        Noll index of the desired Zernike polynomial.
    mask : matrix bool
        Mask of the desired image.
    scale_length : float, optional
        The scale length to use for the Zernike fit.
        The default is the maximum of the image mask shape.

    Returns
    -------
    masked_data : ndarray
        Flattenned array of the masked values of the Zernike
        shape projected on the mask.

    """
    if noll_number < 1:
        raise ValueError("Noll index must be equal to or greater than 1")
    # Image dimensions
    X, Y = np.shape(mask)
    # Determine circle radius on to which define the Zernike
    if scale_length is not None:
        r = scale_length
    else:
        r = np.max([X, Y]) / 2
    # Conversion to polar coordinates on circle of radius r
    phi = lambda i, j: np.arctan2((j - Y / 2.0) / r, (i - X / 2.0) / r)
    rho = lambda i, j: np.sqrt(((j - Y / 2.0) / r) ** 2 + ((i - X / 2.0) / r) ** 2)
    mode = np.fromfunction(
        lambda i, j: zern._zernikel(noll_number, rho(i, j), phi(i, j)), [X, Y]
    )
    masked_data = mode[~mask]
    # Normalization of the masked data: null mean and unit STD
    if noll_number > 1:
        masked_data = (masked_data - np.mean(masked_data)) / np.std(masked_data)
    return masked_data


#################################
## Base class that creates the ##
##       simulated Alpao       ##
#################################


class BaseFakeAlpao(ABC):
    """
    Base class for deformable mirrors.
    """

    def __init__(self, nActs: int):
        """
        Initializes the base deformable mirror with the number of actuators.
        """
        self._name = f"AlpaoDM{nActs}"
        self.mirrorModes = None
        self.nActs = nActs
        self._pxScale = pixel_scale(self.nActs)
        self.actCoords, self.mask = getAlpaoCoordsMask(self.nActs)
        self._scaledActCoords = self._scaleActCoords()
        self._iffCube = None
        self.IM = None
        self.ZM = None
        self.RM = None
        self._load_matrices()

    @abstractmethod
    def set_shape(self, command: _t.ArrayLike, differential: bool = False):
        """
        Applies the DM to a wavefront.

        Parameters
        ----------
        command : np.array
            Wavefront to which the DM will be applied.

        differential : bool
            If True, the command is the differential wavefront.

        Returns
        -------
        np.array
            Modified wavefront.
        """
        raise NotImplementedError

    @abstractmethod
    def get_shape(self):
        """
        Returns the current shape of the DM.

        Returns
        -------
        np.array
            Current shape of the DM.
        """
        raise NotImplementedError
    
    @abstractmethod
    def uploadCmdHistory(self, timed_command_history: _t.MatrixLike):
        """
        Uploads a history of commands to the DM.

        Parameters
        ----------
        timed_command_history : _t.MatrixLike
            A 2D array where each column represents a command to be applied to the DM.
        """
        raise NotImplementedError

    @abstractmethod
    def runCmdHistory(self):
        """
        Executes the uploaded command history on the DM.
        """
        raise NotImplementedError
    

    def _load_matrices(self):
        """
        Loads the required matrices for the deformable mirror's operations.
        """
        if not os.path.exists(IffFile(self.nActs)):
            print(
                f"First time simulating DM {self.nActs}. Generating influence functions..."
            )
            self._simulate_Zonal_Iff_Acquisition()
        else:
            print(f"Loaded influence functions.")
            self._iffCube = np.ma.masked_array(
                osu.load_fits(IffFile(self.nActs))
            )
        self._create_int_and_rec_matrices()
        self._create_zernike_matrix()

    def _create_zernike_matrix(self):
        """
        Create the Zernike matrix for the DM.
        """
        if not os.path.exists(ZernMatFile(self.nActs)):
            n_zern = self.nActs
            print("Computing Zernike matrix...")
            self.ZM = xp.asnumpy(generate_zernike_matrix(n_zern, self.mask))
            osu.save_fits(ZernMatFile(self.nActs), self.ZM)
        else:
            print(f"Loaded Zernike matrix.")
            self.ZM = osu.load_fits(ZernMatFile(self.nActs))

    def _create_int_and_rec_matrices(self):
        """
        Create the interaction matrices for the DM.
        """
        if not os.path.exists(IntMatFile(self.nActs)):
            print("Computing interaction matrix...")
            im = xp.array(
                [
                    (self._iffCube[:, :, i].data)[self.mask == 0]
                    for i in range(self._iffCube.shape[2])
                ]
            )
            self.IM = xp.asnumpy(im)
            osu.save_fits(IntMatFile(self.nActs), self.IM)
        else:
            print(f"Loaded interaction matrix.")
            self.IM = osu.load_fits(IntMatFile(self.nActs))
        if not os.path.exists(RecMatFile(self.nActs)):
            print("Computing reconstruction matrix...")
            self.RM = xp.asnumpy(xp.linalg.pinv(im))
            osu.save_fits(RecMatFile(self.nActs), self.RM)
        else:
            print(f"Loaded reconstruction matrix.")
            self.RM = osu.load_fits(RecMatFile(self.nActs))

    def _simulate_Zonal_Iff_Acquisition(self):
        """
        Simulate the influence functions by imposing 'perfect' zonal commands.

        Parameters
        ----------
        amps : float or np.ndarray, optional
            Amplitude(s) for the actuator commands. If a single float is provided,
            it is applied to all actuators. Default is 1.0.

        Returns
        -------
        np.ma.MaskedArray
            A masked cube of influence functions with shape (height, width, nActs).
        """
        # Get the number of actuators from the coordinates array.
        n_acts = self.actCoords.shape[1]
        max_x, max_y = self.mask.shape
        # Create pixel grid coordinates.
        pix_coords = np.zeros((max_x * max_y, 2))
        pix_coords[:, 0] = np.repeat(np.arange(max_x), max_y)
        pix_coords[:, 1] = np.tile(np.arange(max_y), max_x)
        
        act_pix_coords = self._scaleActCoords()
        
        # Create Empty cube for IFF
        img_cube = np.zeros((max_x, max_y, n_acts))
        # For each actuator, compute the influence function with a TPS interpolation.

        from tps import ThinPlateSpline
        for k in range(n_acts):
            print(f"{k+1}/{n_acts}", end='\r', flush=True)
            # Create a command vector with a single nonzero element (ZONAL IFF).
            act_data = np.zeros(n_acts)
            act_data[k] = 1
            tps = ThinPlateSpline(alpha=0.0)
            tps.fit(act_pix_coords, act_data)
            flat_img = tps.transform(pix_coords)
            img_cube[:, :, k] = flat_img.reshape((max_x, max_y))
        
        # Create a cube mask that tiles the local mirror mask for each actuator.
        cube_mask = np.tile(self.mask, n_acts).reshape(img_cube.shape, order="F")
        cube = np.ma.masked_array(img_cube, mask=cube_mask)
        # Save the cube to a FITS file.
        fits_file = IffFile(self.nActs)
        osu.save_fits(fits_file, cube)
        self._iffCube = cube

    def _scaleActCoords(self):
        """
        Scales the actuator coordinates to the mirror's pixel scale.
        """
        max_x, max_y = self.mask.shape
        if not self.actCoords.shape[1] == 2:
            act_coords = self.actCoords.T  # shape: (n_acts, 2)
        act_pix_coords = np.zeros((self.nActs, 2), dtype=int)
        act_pix_coords[:, 0] = (
            act_coords[:, 1] / np.max(act_coords[:, 1]) * max_x
        ).astype(int)
        act_pix_coords[:, 1] = (
            act_coords[:, 0] / np.max(act_coords[:, 0]) * max_y
        ).astype(int)
        return act_pix_coords
