import os
import numpy as np
from tps import ThinPlateSpline
#from scipy.interpolate import Rbf
from opticalib import folders as fp
from opticalib import typings as _t
from opticalib import load_fits as lf, save_fits as sf
from skimage.draw import polygon_perimeter, polygon2mask
from opticalib.ground.modal_decomposer import ZernikeFitter as _ZF
from opticalib.ground import geometry as geo

join = os.path.join


class BaseFakeDp:

    def __init__(self):
        """The constuctor"""
        self._name = "DP"
        dir = join(os.path.dirname(__file__), "AdOpticaMirrorsData")
        self.coords_file = os.path.join(dir, "dp_coords.fits")
        self.mirrorModes = lf(os.path.join(dir, "dp_cmdmat.fits"))
        self._mask, self._act_px_coords = self._createDpMaskAndCoords()
        self.nActs = self._act_px_coords.shape[0]
        self._load_matrices()
        self.cmdHistory = None
        self._shape = np.ma.masked_array(self._mask * 0, mask=self._mask, dtype=float)
        self._idx = np.where(self._mask == 0)
        self._actPos = np.zeros(self.nActs)
        self._zern = _ZF(self._mask)

    def _wavefront(self, **kwargs: dict[str, _t.Any]) -> _t.ImageData:
        """
        Current shape of the mirror's surface. Only used for the interferometer's
        live viewer (see `interferometer.py`).

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments for customization.
            - zernike : int ,
                Zernike mode to be removed from the wavefront.
            - surf : bool ,
                If True, the shape is returned instead of
                the wavefront.
            - noisy : bool ,
                If True, adds noise to the wavefront.

        Returns
        -------
        wf : np.array
            Phase map of the interferometer.
        """
        zernike = kwargs.get("zernike", None)
        surf = kwargs.get("surf", True)
        noisy = kwargs.get("noisy", False)
        img = np.ma.masked_array(self._shape, mask=self._mask)
        if zernike is not None:
            img = self._zern.removeZernike(img, zernike)
        if not surf:
            Ilambda = 632.8e-9
            phi = np.random.uniform(-0.25 * np.pi, 0.25 * np.pi) if noisy else 0
            wf = np.sin(2 * np.pi / Ilambda * img + phi)
            A = np.std(img) / np.std(wf)
            wf *= A
            img = wf
        dx, dy = 650-img.shape[0], 650-img.shape[1]
        if dx > 0 or dy > 0:
            pimg = np.pad(img.data, ((dx//2, dx - dx//2), (dy//2, dy - dy//2)), mode='constant', constant_values=0)
            pmask = np.pad(img.mask, ((dx//2, dx - dx//2), (dy//2, dy - dy//2)), mode='constant', constant_values=1)
            img = np.ma.masked_array(pimg, mask=pmask)
        return geo.rotate_image(img, angle_deg=45)

    def _mirror_command(self, cmd: _t.ArrayLike, diff: bool, modal: bool):
        """
        Applies the given command to the deformable mirror.

        Parameters
        ----------
        cmd : np.array
            Command to be processed by the deformable mirror.

        diff : bool
            If True, process the command differentially.

        Returns
        -------
        np.array
            Processed shape based on the command.
        """
        if modal:
            mode_img = np.dot(self.ZM, cmd)
            cmd = np.dot(mode_img, self.RM)
        cmd_amp = cmd
        if not diff:
            cmd_amp = cmd - self._actPos
        self._shape[self._idx] += np.dot(cmd_amp, self.IM)
        self._actPos += cmd_amp

    def _load_matrices(self):
        """
        Loads the required matrices for the deformable mirror's operations.
        """
        if not os.path.exists(fp.SIM_DATA_FILE(self._name, 'IF')):
            print(
                f"First time simulating DM {self.nActs}. Generating influence functions..."
            )
            self._simulateDP()
        else:
            print(f"Loaded influence functions.")
            self._iffCube = np.ma.masked_array(lf(fp.SIM_DATA_FILE(self._name, 'IF')))
        self._create_int_and_rec_matrices()
        self._create_zernike_matrix()

    def _create_zernike_matrix(self):
        """
        Create the Zernike matrix for the DM.
        """
        if not os.path.exists(fp.SIM_DATA_FILE(self._name, 'ZM')):
            n_zern = self.nActs
            print("Computing Zernike matrix...")
            from .factory_functions import generateZernikeMatrix

            self.ZM = generateZernikeMatrix(n_zern, self._mask)
            sf(fp.SIM_DATA_FILE(self._name, 'ZM'), self.ZM)
        else:
            print(f"Loaded Zernike matrix.")
            self.ZM = lf(fp.SIM_DATA_FILE(self._name, 'ZM'))

    def _create_int_and_rec_matrices(self):
        """
        Create the interaction matrices for the DM.
        """
        imfile = fp.SIM_DATA_FILE(self._name, 'IM')
        if not os.path.exists(imfile):
            print("Computing interaction matrix...")
            self.IM = np.array(
                [
                    (self._iffCube[:, :, i].data)[self._mask == 0]
                    for i in range(self._iffCube.shape[2])
                ]
            )
            sf(imfile, self.IM)
        else:
            print(f"Loaded interaction matrix.")
            self.IM = lf(imfile)
        rmfile = fp.SIM_DATA_FILE(self._name, 'RM')
        if not os.path.exists(rmfile):
            print("Computing reconstruction matrix...")
            self.RM = np.linalg.pinv(self.IM)
            sf(rmfile, self.RM)
        else:
            print(f"Loaded reconstruction matrix.")
            self.RM = lf(rmfile)

    def _simulateDP(self):
        """
        Simulates the influence function of the DP by TPS interpolation
        """
        dp_mask = self._mask.copy()
        act_px_coords = self._act_px_coords.copy()
        X, Y = dp_mask.shape
        # Create pixel grid coordinates.
        pix_coords = np.zeros((X * Y, 2))
        pix_coords[:, 0] = np.repeat(np.arange(X), Y)
        pix_coords[:, 1] = np.tile(np.arange(Y), X)
        img_cube = np.zeros((X, Y, self.nActs))
        amps = np.ones(self.nActs)
        # For each actuator, compute the influence function with a TPS interpolation.
        for k in range(self.nActs):
            print(f"{k+1}/{self.nActs}", end="\r", flush=True)
            # Create a command vector with a single nonzero element.
            act_data = np.zeros(self.nActs)
            act_data[k] = amps[k]
            tps = ThinPlateSpline(alpha=0.0)
            tps.fit(act_px_coords, act_data)
            flat_img = tps.transform(pix_coords)
            img_cube[:, :, k] = flat_img.reshape((X, Y))
        # Create a cube mask that tiles the local mirror mask for each actuator.
        cube_mask = np.tile(self._mask, self.nActs).reshape(img_cube.shape, order="F")
        cube = np.ma.masked_array(img_cube, mask=cube_mask)
        # Save the cube to a FITS file.
        fits_file = fp.SIM_DATA_FILE(self._name, "IF")
        sf(fits_file, cube)
        self._iffCube = cube

    def _createDpMaskAndCoords(self):
        """
        Creates the mask and the actuator pixel coordinates for the DP
        """
        if self.coords_file is not None:
            dp_coords = lf(self.coords_file)
        else:
            dp_coords = lf("/mnt/nas/m4data/dp_plot_coord.fits")

        # Get DP's shape vertex coordinates (only 1 segment)
        
        x,y = dp_coords[dp_coords[:,1] > dp_coords[:,1].max()/2].T*1000 # upper segment
        _,yl= dp_coords[dp_coords[:,1] < dp_coords[:,1].max()/2].T*1000 # lower segment
        
        y0,y1,y2,y3 = y.min(), y.max(), y.max(), y.min()
        x0,x1,x2,x3 = x[y==y0].min(), x[y==y1].min(), x[y==y2].max(), x[y==y3].max()
        
        # vertex coordinates of the upper segment
        cols = np.array([x0,x1,x2,x3])
        rows = np.array([y0,y1,y2,y3])

        ylm = yl.max()
        # FIXME: gap is too big, should be half, maybe 1/4
        gap = np.abs((ylm - y1)//2).astype(np.int8) # in px

        # rescale to mm/px
        cols = cols - cols.min()
        rows = rows - rows.min()

        sx = int(np.ceil(np.max(cols))) + 1
        sy = int(np.ceil(np.max(rows))) + 1

        # Creates the mask of the single segment
        mm = geo.draw_polygonal_mask((sy, sx), np.stack((cols, rows)).T)
        mm = np.pad(mm, ((gap, 0), (0, 0)), mode="constant", constant_values=1)

        # Creates the full DP mask by mirroring the single segment
        mask_dp = np.zeros((mm.shape[0] * 2, mm.shape[1]))
        mask_dp[mm.shape[0] :, :] = mm
        mask_dp += np.flipud(mask_dp)  # flipping up-down
        
        mask_dp = mask_dp.astype(bool)

        # creating the valid pixel coordinates as a meshgrid
        X = np.arange(mask_dp.shape[1])
        Y = np.arange(mask_dp.shape[0])
        XX, YY = np.meshgrid(X, Y)
        xvalid = XX[~mask_dp].flatten()
        yvalid = YY[~mask_dp].flatten()

        # calculating the pixel to meter scale factors
        xlen = xvalid.max() - xvalid.min()  # px
        ylen = yvalid.max() - yvalid.min()  # px

        # dp dimensions in meters
        x_dp = dp_coords[:, 0].max() - dp_coords[:, 0].min()  # m
        y_dp = dp_coords[:, 1].max() - dp_coords[:, 1].min()  # m

        # pixel scale factor
        xscale = x_dp / xlen
        yscale = y_dp / ylen

        # scaling the dp coordinates to pixel coordinates
        dp2c = np.copy(dp_coords)
        dp2c *= 1000
        dp2c[:, 0] -= dp2c[:, 0].min()
        dp2c[:, 1] -= dp2c[:, 1].min()

        # padding the mask to avoid tangent frame
        final_mask = np.pad(mask_dp, 5, mode="constant", constant_values=1)

        return final_mask, dp2c
