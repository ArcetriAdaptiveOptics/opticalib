import os as _os
import io as _io
import numpy as _np
import contextlib as _clib
from shutil import copyfile as _cp
from opticalib import folders as _fn
from opticalib.ground import osutils as _osu
from . import iff_module as _iff, iff_processing as _ifp
from opticalib import typings as _ot
from opticalib.core.read_config import getStitchingConfig as _gsc
from ._stitching_algorithm import map_stitching as _map_stitching
from skimage.draw import disk as _disk


_ts = _osu.newtn


def _update_imports() -> None:
    """
    Refresh the cached module-level folder reference from the current root config.

    Call this after reloading ``opticalib.core.root`` (and rebinding
    ``opticalib.folders``) so that ``_fn`` points to the correct
    :class:`~opticalib.core.root._folds` instance and stitching data is saved
    in the right location.
    """
    global _fn
    from opticalib.core.root import folders
    _fn = folders


class StitchAnalysis:
    """
    Class to process and analyze acquisitions in sub-aperture mode of a mirror,
    to perform the stitching algorithm and produce stitched images.
    """

    def __init__(self, tn: _ot.Optional[str] = None):
        """The Initiation"""
        self.constants = _gsc()
        self.tn = tn

    def processTns(self, tnvec: tuple[str, tuple[float,float] | list[float]]) -> str:
        """
        Process the IFF obtained during the acquisition, and produces the modes and cubes
        for each position, and produces a cube for each IFF in different positions.

        Parameters
        ----------
        tnvec: tuple
            A tuple of tuples, where each inner tuple contains the scan name and the coordinates
            e.g. (("scan1", (x1, z1)), ("scan2", (x2, z2)), ...)

        Returns
        -------
        str
            The new TN where everything is saved
        """
        newtn = _ts()
        dir = _os.path.join(_fn.INTMAT_ROOT_FOLDER, newtn)
        if not _os.path.exists(dir):
            _os.mkdir(dir)
        full_coord_header = {}
        for j, (tn, xz) in enumerate(tnvec):
            print(f"Processing {(j+1)/len(tnvec)*100:d}%: {tn}", end="\r", flush=True)
            _ifp.process(tn=tn)
            header = {f"X{j}": xz[0], f"Z{j}": xz[1]}
            full_coord_header.update(header)
            captured_output = _io.StringIO()
            with _clib.redirect_stdout(captured_output):
                _ifp.saveCube(tn=tn, cube_header=header)
        amat = _osu.getFileList(tnvec[0][0], fold="IntMatrices", key="cmdMat")
        _cp(amat, _os.path.join(dir, _ifp.cmdMatFile))
        for i in range(self.dm.nActs):
            modevec = []
            for k, (tn, _) in enumerate(tnvec):
                print(
                    f"Processing Mode {i+1}/{self.dm.nActs} : {k/len(tnvec)*100}",
                    end="\r",
                    flush=True,
                )
                fl = _osu.getFileList(tn, fold="IFFunctions", key="mode_")
                img = _osu.load_fits(fl[i])
                modevec.append(img)
            modevec = _np.ma.dstack(modevec)
            _osu.save_fits(
                _os.path.join(dir, f"mode_{i:03d}_cube.fits"),
                modevec,
                header=full_coord_header,
            )
        return newtn

    def stitchAllIffCubes(
        self, tn: str = None, **stitchargs: dict[str, _ot.Any]
    ) -> _ot.CubeData:
        """
        Stitch the IFF cubes obtained during the acquisition, and produces a single cube
        for each IFF in different positions.

        Parameters
        ----------
        tn: str
            The tracking number where the IFF modes cubes are stored.

        Additional Parameters
        ---------------------
        **stitchargs : dict[str, _ot.Any]
            Additional arguments for stitching, including:
        - remask : float, optional
            The new mask radius, in mm, to apply to the cube images. Default is None, meaning no remask.
        - step_size : int, optional
            The step size of the re-sampling of the iff. Default is None, meaning no re-sampling.
        - mask_threshold : float, optional
            Pixel threshold to trim images. Default is 0.2, meaning that images with less than 20% of pixels
            masked will be discarded.
        - deg : float, optional
            The rotation angle in degrees to apply to the coordinates. Default is None, meaning no rotation.
        - average : np.ndarray, optional
            The average image to subtract from the cube images. Default is None, meaning no subtraction.


        Returns
        -------
        newtn = str
            The tracking number of the stitched IFF cube.
        """
        if tn is None and self.tn is not None:
            tn = self.tn
        else:
            raise ValueError("Tracking number (tn) must be provided.")
        remask_size = stitchargs.get("remask", False)
        newtn = _ts()
        print(newtn)
        dir = _os.path.join(_fn.INTMAT_ROOT_FOLDER, newtn)
        if not _os.path.exists(dir):
            _os.mkdir(dir)
        cubelist = _osu.getFileList(tn, fold="INTMatrices", key="mode_")
        cmdmat = _osu.getFileList(tn, fold="INTMatrices", key="cmdMat")
        _cp(cmdmat, _os.path.join(dir, _ifp.cmdMatFile))
        stitch_list = []
        for m, cube in enumerate(cubelist):
            print(f"Mode {m}", flush=True)
            cube, header = _osu.load_fits(cube, True)
            stitch_list.append(
                self.stitchSingleIffCube(cube=cube, header=header, **stitchargs)
            )
        rebin = header.get("REBIN", 1)
        stitched = _np.ma.dstack(stitch_list)
        nheader = {}
        nheader.update(header)
        nheader = {
            "STITCHED": (True, "if the cube is the result of stitching"),
            "MASKSIZE": (
                remask_size * 2,
                "sub aperture mask diameter in mm, 0 if not remasked",
            ),
            "FILTERED": (True, "has zernike removed"),
            "ZREMOVED": ("[1,2,3]", "the zernike modes filtered out"),
            "REBIN": (rebin, "cube rebinning factor"),
        }
        header = _ot.Header()
        for key, value in nheader.items():
            header[key] = value
        _osu.save_fits(_os.path.join(dir, "IMCube.fits"), stitched, header=header)
        return newtn

    def stitchSingleIffCube(
        self,
        cube: _ot.CubeData,
        header: dict[str, _ot.Any] | _ot.Header,
        remask: float = None,
        mask_threshold: float = 0.2,
        step_size: _ot.Optional[float | int] = None,
        deg: _ot.Optional[float] = None,
        average: _ot.Optional[_ot.ImageData] = None,
    ):
        """
        Analyze a single scan and performs the stitching

        Parameters
        ----------
        cube : _ot.CubeData
            The cube data to be stitched.
        header : dict or astropy.Header
            The header of the cube containing the coordinates.
        remask : float, optional
            The new mask radius, in mm, to apply to the cube images. Default
            is False, meaning no remask.
        mask_threshold : float, optional
            Pixel threshold to trim images.
        step_size : float | int, optional
            The step size of the re-sampling of the iff.
        average: np.ndarray, optional
            The average image to subtract from the cube images.
        deg: float, optional
            The rotation angle in degrees to apply to the coordinates.

        Returns
        -------
        np.MaskedArray
            The stitched image
        """
        cube = self._check_cube_dimension(cube)
        coords = self.retrieveCubeCoords(n_positions=cube.shape[0], header=header)
        step = _np.abs(coords[0, 0] - coords[1, 0])
        coords = self._transform_coord(coords, deg=deg)
        ocube = cube.copy()
        ocoords = coords.copy()
        try:
            if step_size:
                N = ocube.shape[0]
                imgshape = ocube.shape[1:]
                Nrows = _np.floor(N**0.5).astype(int)
                Ncols = _np.ceil(N**0.5).astype(int)
                ocube = ocube.reshape((Nrows, Ncols, imgshape[0], imgshape[1]))
                ocoords = ocoords.reshape((Nrows, Ncols, 2))
                coords = []
                cube = []
                for c in range(0, ocube.shape[0], step_size//step):
                    for k in range(0, ocube.shape[1], step_size // step):
                        cube.append(ocube[c,k])
                        coords.append([ocoords[c, k, 0], ocoords[c, k, 1]])
                coords = _np.array(coords)
                cube = _np.ma.dstack(cube)
        finally:
            import gc

            del ocube
            del ocoords
            gc.collect()
        if remask:
            cube, coords = self.remaskCube(remask, cube, coords, mask_threshold)
        if average is not None:
            pass
        fm, iv = self._prepare_masks_and_images(cube, coords)
        stitched = _map_stitching(iv, fm, [1, 2, 3], iv.shape[0])
        return stitched

    def stitchSingleScansionCube(
        self,
        tn: str,
        deg: float = None,
        average: _ot.Optional[_ot.ImageData] = None,
        chunk_size: int = 128,
    ) -> _ot.ImageData:
        """
        Analyze a single scansion cube and performs the stitching.

        Parameters
        ----------
        tn; str
            The tracking number of the scansion to analyze.
        average: np.ndarray, optional
            The average image to subtract from the cube images.
        deg: float, optional
            The rotation angle in degrees to apply to the coordinates.

        Returns
        -------
        np.MaskedArray
            The stitched image.
        """
        if deg is None:
            deg = self.constants["alpha"]
        cube, header = self.getCubeAndHeader(
            _os.path.join(_fn.OPD_IMAGES_ROOT_FOLDER, tn, "cube.fits")
        )
        coords = self.retrieveCubeCoords(n_positions=cube.shape[0], header=header)
        coords = self._transform_coord(coords, deg=deg)
        if average is not None:
            pass
        fm, iv = self._prepare_masks_and_images(cube, coords)
        stitched = _map_stitching(iv, fm, [1, 2, 3], mp_chunk_size=chunk_size)
        return stitched

    def remaskCube(
        self,
        mask_radius: float,
        cube: _ot.CubeData,
        coords: _ot.ArrayLike,
        threshold: float = 0.2,
    ) -> _ot.CubeData:
        """
        Remask all the images in the cube by intersecting a circular mask with
        a specified radius with the already existing one.

        Parameters
        ----------
        mask_radius : float
            The radius of the circular mask in millimeters.
        cube : _ot.CubeData
            The input image cube to be remasked.
        coords : ArrayLike
            The transformed coordinates written on the cube header.
        threshold : float, optional
            The pixel threshold used to reject images.Percentage of useful pixels.

        Returns
        -------
        new_cube : _ot.CubeData
            The remasked image cube.
        new_coords : ArrayLike
            The updated transformed coordinates of the remasked images.
        """
        cube = self._check_cube_dimension(cube)
        img_shape = (cube.shape[1], cube.shape[2])
        mask_center = (img_shape[0] // 2, img_shape[1] // 2)
        mask_radius_px = int(mask_radius // self.constants["pixel_scale"])
        mask = _np.ones(img_shape)
        mx, my = _disk(mask_center, mask_radius_px)
        mask[mx, my] = 0
        mask_pixels = _np.sum(~mask.astype(bool))
        new_cube = []
        new_coords = []
        for i, img in enumerate(cube):
            new_mask = _np.logical_or(img.mask, mask)
            newimg = _np.ma.masked_array((img.copy()).data, mask=new_mask)
            newimg_pixels = _np.sum(~newimg.mask)
            if not (newimg_pixels < threshold * mask_pixels):
                new_cube.append(newimg)
                new_coords.append([coords[i, 0], coords[i, 1]])
        new_cube = _np.ma.dstack(new_cube)
        new_cube = _np.transpose(new_cube, (2, 0, 1))
        return new_cube, _np.array(new_coords)

    def retrieveCubeCoords(
        self, n_positions: int, header: dict[str, _ot.Any] | _ot.Header
    ) -> _ot.ArrayLike:
        """
        Returns the coordinates written in the cube's header

        Parameters
        ----------
        n_positions : int
            The number of pair coordinate positions in the cube.
        header : dict or astropy.Header
            The header of the cube containing the coordinates.

        Returns
        -------
        coords : _ot.ArrayLike
            An array of coordinates in the form of (x, z) for each position.
        """
        coords = []
        for ii in range(n_positions):
            coords.append((header[f"X{ii}"], header[f"Z{ii}"]))
        return _np.array(coords)

    def getCubeAndHeader(
        self, filepath: str
    ) -> tuple[_ot.CubeData, dict[str, _ot.Any]]:
        """
        Load a cube and its header from a FITS file.

        Parameters
        ----------
        filepath : str
            The path to the FITS file.

        Returns
        -------
        tuple
            A tuple containing the transposed cube data (shape `[n_img,n_px,n_px]`)
            and the header.
        """
        cube = _osu.load_fits(filepath, True)
        cube = self._check_cube_dimension(cube)
        return cube, cube.header

    def reloadConstants(self) -> None:
        """Reload the constants from the configuration file"""
        self.constants = _gsc()
        print("Constants reloaded")

    def _check_cube_dimension(self, cube: _ot.CubeData) -> _ot.CubeData:
        """
        Check and returns the cube with dimension `(n_img,n_px,n_px)`
        """
        n1, n2, n3 = _np.shape(cube)
        if n1 == n2:
            cube = _np.transpose(cube.copy(), (2, 0, 1))
        elif n1 == n3:
            cube = _np.transpose(cube.copy(), (1, 0, 2))
        elif n2 == n3:
            pass
        else:
            print(
                "Warning: could not determine the right cube orientation. Be sure it is `(n_img, n_px, n_px)`"
            )
        return cube

    def _transform_coord(
        self, coords: _ot.ArrayLike, deg: float = None
    ) -> _ot.ArrayLike:
        """
        Transform the coordinates by applying a rotation and flipping them.

        Parameters
        ----------
        coords : _ot.ArrayLike
            The coordinates to be transformed, expected to be in the shape of `(n_img, 2)`.
        deg : float, optional
            The rotation angle in degrees to apply to the coordinates. If None,
            uses the default from constants.

        Returns
        -------
        _ot.ArrayLike
            The transformed coordinates after rotation and flipping.
        """
        if deg is None:
            deg = self.constants["alpha"]
        rot_mat = _np.array(
            [
                [_np.cos(_np.deg2rad(deg)), -_np.sin(_np.deg2rad(deg))],
                [_np.sin(_np.deg2rad(deg)), _np.cos(_np.deg2rad(deg))],
            ]
        )
        coords_m = coords / 1000
        coords_m = _np.flip(coords_m, axis=0)
        new_coords = []
        for ii in range(len(coords_m)):
            cc = _np.matmul(rot_mat, coords_m[ii])
            new_coords.append(cc)
        new_coords = _np.array(new_coords)
        rot_coords = new_coords + [-new_coords[:, 0].min(), -new_coords[:, 1].min()]
        nsteps = int(_np.sqrt(len(coords)))
        if (nsteps % 2) != 0:
            new_ax = []
            for k in range(0, len(rot_coords), nsteps):
                new_ax.append(_np.flip(rot_coords[k : k + nsteps, 0], axis=0))
            nax = _np.array(new_ax[0])
            for i in range(1, len(new_ax)):
                nax = _np.hstack((nax, new_ax[i]))
            rot_coords[:, 0] = nax
        return rot_coords

    def _prepare_masks_and_images(
        self, imgcube: _ot.CubeData, pos: _ot.ArrayLike
    ) -> tuple[_ot.ImageData, _ot.CubeData]:
        """
        Extract the mask vector from the cube, created the full mask and prepares
        the images cube masked in the right place.

        Parameters
        ----------
        imgcube : _ot.CubeData
            The input image cube, expected to be in the shape of `(n_img, n_px, n_px)`.
        pos : _ot.ArrayLike
            The positions where the images should be placed, expected to be in the shape of `(n_img, 2)`.

        Returns
        -------
        fullmask : np.ndarray
            A full mask array that combines all individual masks, sized according to the maximum position.
        imgvecout : _ot.CubeData
            A cube of images masked and placed according to the specified positions.
        """
        imgcube = self._check_cube_dimension(imgcube)
        maskvec = []
        for ii in imgcube:
            maskvec.append(ii.mask)
        ps = 1 / self.constants["pixel_scale"] * 1000
        pixpos = (pos * ps).astype(int)
        imgsize = _np.array(_np.shape(maskvec[0])).astype(int)
        imgfsize = imgsize + pixpos.max()  # -> cambiato da `.max(0)`
        fullmask = _np.ones([int(imgfsize[0]), int(imgfsize[1])])
        imgvecout = []
        for i in range(len(pixpos)):
            yi = slice(pixpos[i, 0], pixpos[i, 0] + imgsize[0])
            xi = slice(pixpos[i, 1], pixpos[i, 1] + imgsize[1])
            fullmask[xi, yi] = fullmask[xi, yi] * (maskvec[i])
            imgf = _np.ma.zeros([int(imgfsize[0]), int(imgfsize[1])])
            maskf = _np.ones([int(imgfsize[0]), int(imgfsize[1])])
            imgf[xi, yi] = imgcube[i]
            maskf[xi, yi] = maskf[xi, yi] * (maskvec[i])
            imgf.mask = maskf
            imgvecout.append(imgf)
        imgvecout = _np.ma.dstack(imgvecout)
        return fullmask, _np.transpose(imgvecout, (2, 0, 1))


class StitchAcquire:
    """
    Class to acquire images in sub-aperture mode of a mirror, to be later processed
    and analyzed with the `StitchAnalysis` class.

    Parameters
    ----------
    dm : _ot.DeformableMirrorDevice
        The deformable mirror device used for the acquisition.
    interf : _ot.InterferometerDevice
        The interferometer device used for the acquisition.
    motors : _ot.GenericDevice
        The motor device controlling the axis of the acquisition.
    """

    def __init__(
        self,
        dm: _ot.DeformableMirrorDevice,
        interf: _ot.InterferometerDevice,
        motors: _ot.GenericDevice,
    ):
        self.dm = dm
        self.interf = interf
        self.axis = motors
        self.dataFold = _fn.BASE_DATA_PATH
        self._lastCommandedCoords = [-999,-999]
        self.cvec = None

    def getAxisPosition(self) -> dict[str, float]:
        """
        Get the current position of the motor's axis.

        Returns
        -------
        dict[str, float]
            A dictionary with the current position of the axis in the format:
            {"x": x_position, "y": y_position, "z": z_position}
        """
        x = self.axis.get_position("x")
        y = self.axis.get_position("y")
        z = self.axis.get_position("z")
        positions = {"x": x, "y": y, "z": z}
        self.axis_pos = positions
        return positions

    def setAxisPosition(self, coord: list[float]):
        """
        Set the position of the motor's axis to the specified coordinates.

        Parameters
        ----------
        coord : list[float]
            A list containing the x and z coordinates to set the axis position.
            e.g. [x, z]
        """
        x, z = coord
        xo, zo = self._lastCommandedCoords
        if x == xo:
            self.axis.set_position("z", z)
        elif z == zo:
            self.axis.set_position("x", x)
        else: 
            self.axis._go2coord(coord)
        self._lastCommandedCoords = coord
        print(f"Reached position [{x},{z}]")

    def getCoordinatesVector(
        self, nstep: int, step_in_mm: tuple[int, int] = (3, 3), live_pos: bool = False
    ) -> list[tuple[float, float]]:
        """
        Get the coordinates vector of the grid for scanning.

        Parameters
        ----------
        nstep : int
            The number of steps in each direction (x and z).
        step_in_mm : tuple[int, int], optional
            The step size in millimeters for the x and z directions. Default is (3, 3).
        live_pos : bool, optional
            If True, the starting position will be the current position of the axis.
            If False, it will use the starting coordinates defined in the constants.
            Default is False.

        Returns
        -------
        coord_vector : list[tuple[float, float]]
            A list of tuples containing the coordinates (x, z) for each step in the grid.
            The coordinates are calculated based on the starting position and the step size.
        """
        xstep, ystep = step_in_mm
        coord = []
        if live_pos:
            curr_pos = self.getAxisPosition()
            start_pos = [int(curr_pos["x"]), int(curr_pos["z"])]
        else:
            start_pos = self.constants["starting_coords"]
        for i in range(nstep):
            row = []
            for j in range(nstep):
                xx = start_pos[0] + j * xstep
                zz = start_pos[1] + i * ystep
                ci = xx, zz
                row.append(ci)
            if i % 2 == 1:
                row.reverse()
            coord.extend(row)
        return coord

    def acquireSingleScan(self, coord_vec: list[float], nframes: int = 1, homing:bool = True) -> str:
        """
        Acquire a single scan at each position in the coordinate vector.

        Parameters
        ----------
        coord_vec : list[float]
            A list of tuples with the coordinates (x, z) where the scan will be acquired.
            e.g. [(x1, z1), (x2, z2), ...]
        nframes : int, optional
            The number of frames to acquire at each position. Default is 1.
        homing: bool, optional
            If True, the axis will be homed after the acquisition. Default is True.

        Returns
        -------
        tn : str
            The tracking number (TN) of the scan, where is located the cube of acquired
            images at each position.
        """
        tn = _ts()
        print(tn)
        ddir = _os.path.join(_fn.OPD_IMAGES_ROOT_FOLDER, tn)
        if not _os.path.exists(ddir):
            _os.mkdir(ddir)
        header = {}
        imglist = []
        for i, xz in enumerate(coord_vec):
            print(f"X = {xz[0]}, Z = {xz[1]}", end="\r", flush=True)
            header.update({f"X{i}": xz[0],f"Z{i}": xz[1],})
            co = _io.StringIO()
            with _clib.redirect_stdout(co):
                self.setAxisPosition(xz)
            imglist.append(self.interf.acquire_map(nframes=nframes))
        cube = _np.ma.dstack(imglist)
        _osu.save_fits(_os.path.join(ddir, "cube.fits"), cube, header=header)
        if homing:
            self.axis.homing()
        return tn

    def acquireSubApertureIFF(
        self, coord_vec: list[float]
    ) -> list[tuple[str, tuple[float, float]]]:
        """
        Acquire the IFF at each position in the coordinate vector.

        Parameters
        ----------
        coord_vec: list
            A list of tuples with the coordinates (x, z) where the IFF will be acquired.

        Returns
        -------
        tnvec : list
            A list of lists, where each list contains the TN and the coordinates.
        """
        tnvec = []
        for _, xz in enumerate(coord_vec):
            self.setAxisPosition(xz)
            tn = _iff.iffDataAcquisition(self.dm, self.interf)
            tnvec.append([tn, xz])
        self.axis.homing()
        return tnvec
