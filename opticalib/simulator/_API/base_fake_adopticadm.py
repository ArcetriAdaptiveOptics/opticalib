import os
import numpy as np
from tps import ThinPlateSpline
from scipy.interpolate import Rbf
from opticalib import folders as fp
from opticalib import typings as _t
from opticalib.ground import osutils as osu
from opticalib import load_fits as lf, save_fits as sf
from skimage.draw import polygon_perimeter, polygon2mask

join = os.path.join

class BaseFakeDp:
    
    def __init__(self):
        """The constuctor"""
        self._name = 'AdOpticaDm'
        self.coords_file = join('AdOpticaMirrorsData','dp_coords.fits')
        self.mirrorModes = lf(join('AdOpticaMirrorsData','dp_cmdmat.fits'))
        self._mask, self._act_px_coords = self._createDpMaskAndCoords()
        self.nacts = self._act_px_coords.shape[0]
        self._basedir = fp.BASE_DATA_PATH
        self._load_matrices()
        self.cmdHistory = None
        self._shape = np.ma.masked_array(self.mask * 0, mask=self.mask, dtype=float)
        self._idx = np.where(self.mask == 0)
        self._actPos = np.zeros(self.nacts)


    def _wavefront(self, **kwargs: dict[str,_t.Any]) -> _t.ImageData:
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
        img = np.ma.masked_array(self._shape, mask=self.mask)
        if zernike is not None:
            from opticalib.ground.modal_decomposer import ZernikeFitter
            zfit = ZernikeFitter(self.mask)
            img = zfit.removeZernike(img, zernike)
        if not surf:
            Ilambda = 632.8e-9
            phi = np.random.uniform(-0.25*np.pi, 0.25*np.pi) if noisy else 0
            wf = np.sin(2*np.pi/Ilambda * img + phi)
            A = np.std(img)/np.std(wf)
            wf *= A
            img = wf
        return img

    def _mirror_command(self, cmd, diff, modal):
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
        if not os.path.exists(join(self._basedir, f"dp_iff.fits")):
            print(f"First time simulating DM {self.nacts}. Generating influence functions...")
            self._simulateDP()
        else:
            print(f"Loaded influence functions.")
            self._iffCube = np.ma.masked_array(lf(join(self._basedir, f"dp_iff.fits")))
        self._create_int_and_rec_matrices()
        self._create_zernike_matrix()

    
    def _create_zernike_matrix(self):
        """
        Create the Zernike matrix for the DM.
        """
        if not os.path.exists(join(self._basedir, f"dp_zernike_matrix.fits")):
            n_zern = self.nacts
            print("Computing Zernike matrix...")
            from opticalib.ground import zernike as zern
            self.ZM = zern.generate_zernike_matrix(n_zern, self.mask)
            sf(join(self._basedir, f"dp_zernike_matrix.fits"), self.ZM)
        else:
            print(f"Loaded Zernike matrix.")
            self.ZM = lf(join(self._basedir, f"dp_zernike_matrix.fits"))


    def _create_int_and_rec_matrices(self):
        """
        Create the interaction matrices for the DM.
        """
        if not os.path.exists(join(self._basedir, f"dp_int_matrix.fits")):
            print("Computing interaction matrix...")
            self.IM = np.array(
                [
                    (self._iffCube[:, :, i].data)[self.mask == 0]
                    for i in range(self._iffCube.shape[2])
                ]
            )
            sf(join(self._basedir, f"dp_int_matrix.fits"), self.IM)
        else:
            print(f"Loaded interaction matrix.")
            self.IM = lf(join(self._basedir, f"dp_int_matrix.fits"))
        if not os.path.exists(join(self._basedir, f"dp_rec_matrix.fits")):
            print("Computing reconstruction matrix...")
            self.RM = np.linalg.pinv(self.IM)
            sf(join(self._basedir, f"dp_rec_matrix.fits"), self.RM)
        else:
            print(f"Loaded reconstruction matrix.")
            self.RM = lf(join(self._basedir, f"dp_rec_matrix.fits"))

    def _simulateDP(self):
        """
        Simulates the influence function of the DP by TPS interpolation
        """
        dp_mask = self._mask.copy()
        act_px_coords = self._act_px_coords.copy()
        X,Y = dp_mask.shape
        # Create pixel grid coordinates.
        pix_coords = np.zeros((X * Y, 2))
        pix_coords[:, 0] = np.repeat(np.arange(X), Y)
        pix_coords[:, 1] = np.tile(np.arange(Y), X)
        img_cube = np.zeros((X, Y, self.nacts))
        amps = np.ones(self.nacts)
        # For each actuator, compute the influence function with a TPS interpolation.
        for k in range(self.nacts):
            print(f"{k+1}/{self.nacts}", end='\r', flush=True)
            # Create a command vector with a single nonzero element.
            act_data = np.zeros(self.nacts)
            act_data[k] = amps[k]
            tps = ThinPlateSpline(alpha=0.0)
            tps.fit(act_px_coords, act_data)
            flat_img = tps.transform(pix_coords)
            img_cube[:, :, k] = flat_img.reshape((X, Y))
        # Create a cube mask that tiles the local mirror mask for each actuator.
        cube_mask = np.tile(self.mask, self.nacts).reshape(img_cube.shape, order='F')
        cube = np.ma.masked_array(img_cube, mask=cube_mask)
        # Save the cube to a FITS file.
        fits_file = os.path.join(self._basedir, f"dp_iff.fits")
        sf(fits_file, cube)
        self._iffCube = cube
        

    def _createDpMaskAndCoords(self):
        """
        Creates the mask and the actuator pixel coordinates for the DP
        """
        if self.coords_file is not None:
            dp_coords = lf(self.coords_file)
        else:
            dp_coords = lf('/mnt/nas/m4data/dp_plot_coord.fits')

        # Get DP's shape vertex coordinates (only 1 segment)
        ymax = dp_coords[dp_coords[:,1]>0][:,1].max()
        ymin = dp_coords[dp_coords[:,1]>0][:,1].min()
        xmax = dp_coords[dp_coords[:,1]>0][:,0].max()
        xmin = dp_coords[dp_coords[:,1]>0][:,0].min()
        x2min = dp_coords[dp_coords[:,1]>0.14][:,0].min()
        x2max = dp_coords[dp_coords[:,1]>0.14][:,0].max()

        rows = np.array([ymax, ymax, ymin,ymin])
        cols = np.array([x2min, x2max, xmax, xmin])

        # Shift to eliminate negative values
        cols -= np.min(cols)

        # Creates the mask's perimeter array
        perimeter = polygon_perimeter(rows*1000, cols*1000, (1000,1000))
        poly = np.array(perimeter).T
        # Creates the mask of the single segment
        mm = polygon2mask((poly[:,0].max(), poly[:,1].max()), poly)
        # Creates the full DP mask by mirroring the single segment
        totalm = np.zeros((mm.shape[0]*2, mm.shape[1]))
        totalm[mm.shape[0]:, :] = mm
        totalm += np.flipud(totalm) # flipping up-down

        mask_dp = ~(totalm.astype(bool))

        # creating the valid pixel coordinates as a meshgrid
        X = np.arange(mask_dp.shape[1])
        Y = np.arange(mask_dp.shape[0])
        XX, YY = np.meshgrid(X, Y)
        xvalid = XX[~mask_dp].flatten()
        yvalid = YY[~mask_dp].flatten()

        # calculating the pixel to meter scale factors
        xlen = xvalid.max() - xvalid.min() # px
        ylen = yvalid.max() - yvalid.min() # px

        # dp dimensions in meters
        x_dp = dp_coords[:,0].max() - dp_coords[:,0].min() # m
        y_dp = dp_coords[:,1].max() - dp_coords[:,1].min() # m

        # pixel scale factor
        xscale = x_dp/xlen
        yscale = y_dp/ylen

        # scaling the dp coordinates to pixel coordinates
        dp2c = np.copy(dp_coords)
        dp2c *= 1000
        dp2c[:,0] -= dp2c[:,0].min()
        dp2c[:,1] -= dp2c[:,1].min()

        # padding the mask to avoid tangent frame
        npix = 10
        final_mask = np.pad(mask_dp, npix, mode='constant', constant_values=1)
        
        return final_mask, dp2c