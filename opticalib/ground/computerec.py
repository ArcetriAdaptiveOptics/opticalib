"""
Module containing the ComputeReconstructor class, which, from a dm calibration,
computes the reconstruction matrix.

Author(s):
----------
- Marco Xompero : written in 2024
- Pietro Ferraiuolo : modified in 2024
"""

import numpy as _np
import xupy as _xp
from . import logger as _log
from . import osutils as _osu
import matplotlib.pyplot as _plt
from opticalib import typings as _ot
from opticalib.core.root import folders as _fn

_intMatFold = _fn.INTMAT_ROOT_FOLDER


class ComputeReconstructor:
    """
    This class analyzes the measurements made through the IFF class
    and calculates the reconstructor to be used in the control loop.

    HOW TO USE IT::

        tn = "YYYYMMDD_HHMMSS"
        cr = ComputeReconstructor.loadInteractionMatrix(tn)
        rec = cr.run()

    OR

        cr = ComputeReconstructor(interation_matrix_cube)
        rec = cr.run(Interactive=True)

        where the interaction_matrix_cube is a masked_array dstack of
        shape [pixels, pixels, n_images]
    """

    def __init__(
        self,
        interaction_matrix_cube: _ot.MatrixLike,
        mask2intersect: _ot.Optional[_ot.MatrixLike] = None,
    ):
        """The constructor"""
        self._logger = _log.set_up_logger("computerec.log", 20)
        self._intMatCube = interaction_matrix_cube
        self._cubeMask = self._intersectCubeMask()
        self._imgMask = self._mask2intersect(mask2intersect)
        self._analysisMask = self._setAnalysisMask()
        self._intMat = self._computeIntMat()
        self._intMat_U = None
        self._intMat_S = None
        self._intMat_Vt = None
        self._threshold = None
        self._filtered_sv = None

    def run(
        self, sv_threshold: int | float = None, interactive: bool = False
    ) -> _ot.MatrixLike:
        """
        Compute the reconstruction matrix from the interaction matrix and the image
        to flatten.

        Parameters
        ----------
        Interactive : bool, optional
            If True, the function will show an interactive plot to choose the
            threshold for the singular values. Default is False.
        sv_threshold : int | float, optional
            The threshold for the singular values. If None, the function will
            compute the pseudo-inverse of the interaction matrix. If an integer
            is provided, it will be used as the threshold. If a float is provided,
            it will be used as the threshold. Default is None.

        Returns
        -------
        recMat : MatrixLike
            Reconstruction matrix.
        """
        self._logger.info("Reconstructor Computation:")
        self._computeIntMat()
        self._logger.info("SVD of Interaction Matrix")
        IM = _xp.asarray(self._intMat, dtype=_xp.float)
        U, S, Vt = _xp.linalg.svd(IM, full_matrices=False)
        self._intMat_U, self._intMat_S, self._intMat_Vt = [
            _xp.asnumpy(x) for x in (U, S, Vt)
        ]
        if interactive:
            self._threshold = self.make_interactive_plot(self._intMat_S)
        else:
            if sv_threshold is None:
                return _xp.asnumpy(_xp.linalg.pinv(IM))
            elif isinstance(sv_threshold, int):
                self._threshold = {
                    "y": _np.finfo(_np.float32).eps,
                    "x": -sv_threshold,
                }
            else:
                self._threshold = {
                    "y": sv_threshold,
                    "x": _np.argmin(_np.abs(self._intMat_S - sv_threshold)),
                }
        sv_threshold = S.copy()
        sv_threshold[self._threshold["x"] :] = 0
        sv_inv_threshold = sv_threshold * 0
        sv_inv_threshold[0 : self._threshold["x"]] = (
            1 / sv_threshold[0 : self._threshold["x"]]
        )
        self._filtered_sv = sv_inv_threshold
        self._logger.info("Computing Reconstructor Matrix: Vt.T @ S_inv @ U.T")
        return _xp.asnumpy(Vt.T @ _xp.diag(sv_inv_threshold) @ U.T)

    def getSVD(self):
        """
        Returns the SVD components of the interaction matrix.

        Returns
        -------
        U : MatrixLike
            Left singular vectors.
        S : ArrayLike
            Singular values.
        Vt : MatrixLike
            Right singular vectors transposed.
        """
        if not all(
            [x is not None for x in (self._intMat_U, self._intMat_S, self._intMat_Vt)]
        ):
            return self._intMat_U, self._intMat_S, self._intMat_Vt
        else:
            print("SVD has not been computed yet. Run the 'run' method first.")

    def loadShape2Flat(self, img: _ot.ImageData) -> "ComputeReconstructor":
        """
        Function intended as a reloader for the image mask to intersect, in order
        to create a new recontructor matrix.

        Parameters
        ----------
        img : ImageData
            The image to compute the new recontructor.
        """
        self._shape2flat = img.copy()
        self._imgMask = self._shape2flat.mask
        return self

    def loadInteractionCube(
        self, intCube: _ot.Optional[_ot.MatrixLike] = None, tn: str = None
    ) -> "ComputeReconstructor":
        """
        Function intended as a reloader for the interaction matrix cube, to use
        a different IFF for recontructor creation.

        Parameters
        ----------
        intCube : ndarray, optional
            The data cube itself.
        tn : str, optional
            The tracking number where to find the data cube. Default is None.

        Returns
        -------
        """
        import os

        if intCube is not None:
            self._intMatCube = intCube
        elif tn is not None:
            cube_path = os.path.join(_intMatFold, tn, "IMCube.fits")
            self._intMatCube = _osu.read_phasemap(cube_path)
        else:
            raise KeyError("No cube or tracking number was provided.")
        self._computeIntMat()
        return self

    def _computeIntMat(self):
        """
        Subroutine which computes the interaction matrix and stores it in a
        class variable.
        """
        self._logger.info(
            "Computing interaction matrix from cube of size %s", self._intMatCube.shape
        )
        print("Computing Interaction Matrix", end="\a")
        print("...", end="\a", flush=True)
        try:
            self._setAnalysisMask()
            # New efficient way - TOTRY
            n_images = self._intMatCube.shape[2]
            reshaped = self._intMatCube.data.reshape(-1, n_images)
            mask_flat = (self._analysisMask == 0).ravel()
            intMat = reshaped[mask_flat, :].T
            # self._intMat = _np.array(
            #     [
            #         (self._intMatCube[:, :, i].data)[self._analysisMask == 0]
            #         for i in range(self._intMatCube.shape[2])
            #     ]
            # )
        except Exception as e:
            self._logger.error("Error in computing interaction matrix from cube:%s", e)
            raise e
        self._logger.info("Computed interaction matrix of shape %s", intMat.shape)
        return intMat

    def _setAnalysisMask(self):
        """
        Sets the analysis mask as the mask resulting from the 'logical_or' between
        the cube mask and the image to flatten mask.
        """
        try:
            if self._imgMask is None:
                analysisMask = self._cubeMask
            else:
                analysisMask = _np.logical_or(self._cubeMask, self._imgMask)
        except Exception as e:
            raise e
        self._analysisMask = analysisMask

    def _mask2intersect(
        self, img_or_mask: _ot.Optional[_ot.MatrixLike | _ot.ImageData] = None
    ) -> _ot.Optional[_ot.MatrixLike]:
        """
        Returns the mask from the input image or mask.

        Parameters
        ----------
        img_or_mask : ndarray or MaskedArray
            The image's mask or mask itself to use to compute the reconstructor.

        Returns
        -------
        mask : ndarray
            The mask.
        """
        if img_or_mask is not None:
            try:
                mask = img_or_mask.mask
            except AttributeError:
                mask = img_or_mask
        else:
            mask = None
        return mask

    def _intersectCubeMask(self) -> _ot.MatrixLike:
        """
        Creates the cube's mask by intersecating the masks of each frame.

        Returns
        -------
        cube_mask : MatrixLike
            the intersection mask for the cube.
        """
        mask = self._intMatCube[:, :, 0].mask
        for i in range(1, self._intMatCube.shape[2]):
            cube_mask = _np.logical_or(mask, self._intMatCube[:, :, i].mask)
            mask = cube_mask
        return cube_mask

    # ______________________________________________________________________________
    @staticmethod
    def make_interactive_plot(singular_values: _ot.ArrayLike):
        # Creare il grafico
        fig, ax = _plt.subplots()
        modelist = _np.arange(len(singular_values))
        ax.loglog(modelist, singular_values, "b-o")
        ax.set_xlabel("Mode number")
        ax.set_ylabel("Singular value")
        ax.grid()
        ax.autoscale(tight=False)
        ax.set_ylim([min(singular_values), max(singular_values)])
        ax.title.set_text("Singular values")
        threshold = dict()

        threshold["y"] = _np.finfo(_np.float32).eps  # 0.01
        threshold["x"] = 0  # 5 % len(singular_values)
        ax.axhline(threshold["y"], color="g", linestyle="-", linewidth=1)
        ax.axvline(threshold["x"], color="g", linestyle="-", linewidth=1)
        ax.plot(
            modelist[singular_values < threshold["y"]],
            singular_values[singular_values < threshold["y"]],
            "r-x",
        )

        # Funzione per aggiornare la posizione della croce rossa
        def update_crosshair(event):
            if event.inaxes:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                ax.cla()
                ax.loglog(modelist, singular_values, "b-o")
                ax.plot(
                    modelist[singular_values < threshold["y"]],
                    singular_values[singular_values < threshold["y"]],
                    "r-x",
                )
                ax.autoscale(tight=False)
                ax.set_xlabel("Mode number")
                ax.set_ylabel("Singular value")
                ax.grid()
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.axhline(threshold["y"], color="g", linestyle="-", linewidth=1)
                ax.axvline(threshold["x"], color="g", linestyle="-", linewidth=1)
                ax.title.set_text("Singular values")
                x_mouse, y_mouse = event.xdata, event.ydata
                ax.axhline(y_mouse, color="r", linestyle="-", linewidth=1)
                ax.axvline(
                    _np.argmin(_np.abs(singular_values - y_mouse)),
                    color="r",
                    linestyle="-",
                    linewidth=1,
                )
                ax.figure.canvas.draw()
                # print(
                #    f"Current threshold: X = {threshold['x']:.2f}, \
                #        Y = {threshold['y']:.5f}"
                # )

        # Connettere la funzione all'evento di movimento del mouse
        ax.add_artist(ax.patch)
        fig.canvas.mpl_connect(
            "motion_notify_event", lambda event: update_crosshair(event)
        )

        # Funzione per registrare le coordinate del click

        def record_click(event):
            if event.inaxes and event.dblclick:
                x_click, y_click = event.xdata, event.ydata
                threshold["y"] = y_click
                threshold["x"] = _np.argmin(_np.abs(singular_values - y_click))
                ax.axhline(threshold["x"], color="g", linestyle="-", linewidth=1)
                ax.axvline(threshold["y"], color="g", linestyle="-", linewidth=1)
                # print(
                #    f"New eigenvalues threshold: X = {x_click:.2f}, \
                #            Y = {y_click:.2f}"
                # )
                return threshold["x"], threshold["y"]

        # Connettere la funzione all'evento di click del tasto sinistro
        fig.canvas.mpl_connect("button_press_event", lambda event: record_click(event))

        # Visualizzare il grafico
        _plt.show()
        return threshold
