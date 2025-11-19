import os
import numpy as np
from skimage.draw import polygon_perimeter, polygon2mask
from opticalib import load_fits as lf, save_fits as sf
from tps import ThinPlateSpline
from opticalib import folders as fp
from opticalib.ground import osutils as osu
from scipy.interpolate import Rbf
join = os.path.join


class DPSimulator():
    """
    """
    def __init__(self, coords_file: str = None):
        self._name = 'AdOpticaDm'
        self.coords_file = coords_file
        self.mirrorModes = lf(os.path.join(fp.MODALBASE_ROOT_FOLDER, 'dp_cmdmat.fits'))
        self._mask, self._act_px_coords = self._createDpMaskAndCoords()
        self.nacts = self._act_px_coords.shape[0]
        self._basedir = fp.BASE_DATA_PATH
        self._load_matrices()
        self.cmdHistory = None
        self._shape = np.ma.masked_array(self.mask * 0, mask=self.mask, dtype=float)
        self._idx = np.where(self.mask == 0)
        self._actPos = np.zeros(self.nacts)
        
    @property
    def nActs(self):
        return self.nacts
        
    @property
    def mask(self):
        return self._mask


    def set_shape(self, command, differential: bool = False, modal: bool = False):
        """
        Applies the given command to the deformable mirror.

        Parameters
        ----------
        command : np.array
            Command to be applied to the deformable mirror.

        differential : bool
            If True, the command is applied differentially.
        """
        scaled_cmd = command * 1e-5  # more realistic command
        self._mirror_command(scaled_cmd, differential, modal)


    def get_shape(self):
        """
        Returns the current amplitudes commanded to the dm's actuators.

        Returns
        -------
        np.array
            Current amplitudes commanded to the dm's actuators.
        """
        return self._actPos.copy()

    def uploadCmdHistory(self, cmdhist):
        """
        Upload the command history to the deformable mirror memory.
        Ready to run the `runCmdHistory` method.
        """
        self.cmdHistory = cmdhist

    def runCmdHistory(
        self,
        interf=None,
        save: str = None,
        rebin: int = 1,
        modal: bool = False,
        differential: bool = True,
        delay: float = 0,
    ):
        """
        Runs the command history on the deformable mirror.

        Parameters
        ----------
        interf : Interferometer
            Interferometer object to acquire the phase map.
        rebin : int
            Rebinning factor for the acquired phase map.
        modal : bool
            If True, the command history is modal.
        differential : bool
            If True, the command history is applied differentially
            to the initial shape.

        Returns
        -------
        tn :str
            Timestamp of the data saved.
        """
        import time
        if self.cmdHistory is None:
            raise Exception("No Command History to run!")
        else:
            tn = osu.newtn() if save is None else save
            print(f"{tn} - {self.cmdHistory.shape[-1]} images to go.")
            datafold = os.path.join(fp.OPD_IMAGES_ROOT_FOLDER, tn)
            s = self.get_shape()
            if not os.path.exists(datafold):
                os.mkdir(datafold)
            for i, cmd in enumerate(self.cmdHistory.T):
                print(f"{i+1}/{self.cmdHistory.shape[-1]}", end="\r", flush=True)
                if differential:
                    cmd = cmd + s
                self.set_shape(cmd, modal=modal)
                if interf is not None:
                    time.sleep(delay)
                    img = interf.acquire_map(rebin=rebin)
                    path = os.path.join(datafold, f"image_{i:05d}.fits")
                    sf(path, img)
        self.set_shape(s)
        return tn

    def visualize_shape(self, cmd=None):
        """
        Visualizes the command amplitudes on the mirror's actuators.

        Parameters
        ----------
        cmd : np.array, optional
            Command to be visualized on the mirror's actuators. If none, will plot
            the current position of the actuators.

        Returns
        -------
        np.array
            Processed shape based on the command.
        """
        import matplotlib.pyplot as plt
        if cmd is None:
            cmd = self._actPos.copy()
        plt.figure(figsize=(7, 6))
        size = (120 * 97) / self.nActs
        plt.scatter(
            self._act_px_coords[:, 0], self._act_px_coords[:, 1], c=cmd, s=size
        )
        plt.xlabel(r"$x$ $[px]$")
        plt.ylabel(r"$y$ $[px]$")
        plt.title(f"DM {self.nActs} Actuator's Coordinates")
        plt.colorbar()
        plt.show()
        
    def wavefront(self, **kwargs):
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
    
    @staticmethod
    def createMaskAndCoords(
        coords: str | np.ndarray,
        geometry: str = 'circle',
        radius: float = None,
        vertices: np.ndarray = None,
        mask_size: int = 1000,
        padding: int = 10,
        mirror_axis: str = None
    ):
        """
        General function to create a mask and scaled actuator coordinates for any deformable mirror.
        
        Parameters
        ----------
        coords_file : str
            Path to the FITS file containing actuator coordinates in meters.
            Expected format: Nx2 array with [x, y] coordinates.
        
        geometry : str, optional
            Type of mirror geometry. Options are 'circle' or 'polygon'. Default is 'circle'.
        
        radius : float, optional
            Radius of the circular mirror aperture in meters. Required if geometry='circle'.
        
        vertices : np.ndarray, optional
            Array of polygon vertices in meters with shape (N, 2) for [x, y] coordinates.
            Required if geometry='polygon'. Vertices should be ordered sequentially.
        
        mask_size : int, optional
            Size of the mask in pixels (for the largest dimension). Default is 1000.
        
        padding : int, optional
            Number of pixels to pad around the mask to avoid edge effects. Default is 10.
        
        mirror_axis : str, optional
            If provided, creates a symmetric mask by mirroring along the specified axis.
            Options: 'horizontal', 'vertical', None. Default is None (no mirroring).
        
        Returns
        -------
        final_mask : np.ndarray
            Boolean mask where True indicates invalid/masked pixels.
        
        act_px_coords : np.ndarray
            Actuator coordinates in pixel space (Nx2 array).
        
        Raises
        ------
        ValueError
            If geometry parameters are invalid or missing.
        
        Examples
        --------
        For a circular mirror:
        >>> mask, coords = DPSimulator.createMaskAndCoords(
        ...     'actuator_coords.fits', 
        ...     geometry='circle', 
        ...     radius=0.05
        ... )
        
        For a polygonal mirror:
        >>> vertices = np.array([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]])
        >>> mask, coords = DPSimulator.createMaskAndCoords(
        ...     'actuator_coords.fits',
        ...     geometry='polygon',
        ...     vertices=vertices
        ... )
        """
        # Load actuator coordinates in meters
        if isinstance(coords, str):
            act_coords = lf(coords)
        else:
            act_coords = coords
        if act_coords.shape[1] != 2:
            raise ValueError(f"Expected actuator coordinates with shape (N, 2), got {act_coords.shape}")
        
        # Determine mirror physical dimensions
        x_min, y_min = act_coords.min(axis=0)
        x_max, y_max = act_coords.max(axis=0)
        x_extent = x_max - x_min  # meters
        y_extent = y_max - y_min  # meters
        
        # Create mask based on geometry
        if geometry == 'circle':
            if radius is None:
                raise ValueError("Radius must be provided for circular geometry")
            
            # Create circular mask
            mask_diameter = mask_size
            center = mask_diameter / 2
            y_grid, x_grid = np.ogrid[:mask_diameter, :mask_diameter]
            distance_from_center = np.sqrt((x_grid - center)**2 + (y_grid - center)**2)
            mask = distance_from_center > (mask_diameter / 2 - 1)
            
            # Physical scale: map 2*radius (diameter) to mask_diameter pixels
            pixel_scale = (2 * radius) / mask_diameter
            
        elif geometry == 'polygon':
            if vertices is None:
                raise ValueError("Vertices must be provided for polygon geometry")
            
            vertices = np.asarray(vertices)
            if vertices.shape[1] != 2:
                raise ValueError(f"Expected vertices with shape (N, 2), got {vertices.shape}")
            
            # Normalize vertices to [0, 1] range
            v_min = vertices.min(axis=0)
            v_max = vertices.max(axis=0)
            v_extent = v_max - v_min
            
            # Scale vertices to pixel coordinates
            scale_factor = mask_size / v_extent.max()
            vertices_px = (vertices - v_min) * scale_factor
            
            # Create polygon mask
            mask_shape = (int(np.ceil(vertices_px[:, 1].max())) + 1,
                         int(np.ceil(vertices_px[:, 0].max())) + 1)
            
            # Convert vertices to row, col format for polygon2mask
            poly_coords = np.column_stack([vertices_px[:, 1], vertices_px[:, 0]])
            mask = ~polygon2mask(mask_shape, poly_coords)
            
            # Physical scale
            pixel_scale = v_extent.max() / mask_size
            
        else:
            raise ValueError(f"Unknown geometry type: {geometry}. Use 'circle' or 'polygon'.")
        
        # Apply mirroring if requested
        if mirror_axis == 'horizontal':
            # Mirror horizontally (left-right)
            half_mask = mask[:, :mask.shape[1]//2]
            mirrored = np.fliplr(half_mask)
            mask = np.hstack([half_mask, mirrored])
        elif mirror_axis == 'vertical':
            # Mirror vertically (up-down)
            half_mask = mask[:mask.shape[0]//2, :]
            mirrored = np.flipud(half_mask)
            mask = np.vstack([half_mask, mirrored])
        
        # Convert actuator coordinates to pixel coordinates
        # Center the coordinates
        act_coords_centered = act_coords - act_coords.min(axis=0)
        
        # Scale to pixels
        act_px_coords = act_coords_centered / pixel_scale
        
        # Center actuators in the mask
        mask_center = np.array([mask.shape[1] / 2, mask.shape[0] / 2])
        act_center = np.array([act_px_coords[:, 0].mean(), act_px_coords[:, 1].mean()])
        act_px_coords += (mask_center - act_center)
        
        # Apply padding to mask
        final_mask = np.pad(mask, padding, mode='constant', constant_values=True)
        
        # Adjust actuator coordinates for padding
        act_px_coords += padding
        
        return final_mask, act_px_coords

