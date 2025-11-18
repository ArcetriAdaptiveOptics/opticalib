import numpy as _np
import matplotlib.pyplot as _plt
from opticalib.ground import zernike as zern
from opticalib import typings as _t
from opticalib.analyzer import modeRebinner as rebinned
from matplotlib.animation import FuncAnimation as _FuncAnimation

_conf = {
    "width": 512,
    "height": 512,
    "full_width": 1024,
    "full_height": 1024,
    "x-offset": 0,
    "y-offset": 0,
}


class Fake4DInterferometer:

    def __init__(self, dm: _t.DeformableMirrorDevice):
        self.model = "4D"
        self.full_frame = False
        self.shapesRemoved = None
        self._dm = dm
        self._lambda = 632.8e-9  # λ[m]
        self._anim = None
        self._live = False
        self._surf = False
        self._freeze = False
        self._noisy = False
        self._fps = 10
        self._fW, self._fH = self._readFullFrameSize()

    def live(
        self,
        zernike2remove: list[int] = None,
        framerate: int = 10,
        **kwargs: dict[str,_t.Any],
    ):
        """
        Runs the live-view animation for the simulated Interferometer
        instance.

        Parameters
        ----------
        shape2remove : np.array, optional
            Zernike modes to be removed from the wavefront.
        framerate : int, optional
            Framerate of the live-view animation.
        **kwargs : dict, optional
            Additional keyword arguments for imshow customization.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object of the live-view animation.
        anim : matplotlib.animation.FuncAnimation
            Animation object of the live-view animation (needed to keep the plot
            alive).
        """
        self._fps = framerate
        if zernike2remove is not None:
            self.shapeRemoval(zernike2remove)
        global _anim
        cmap = kwargs.get("cmap", "gray")

        self._live = True
        self._dm._live = True

        # Main plot creation
        _plt.ion()
        fig, ax = _plt.subplots(figsize=(7, 7.5))
        fig.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)
        fig.canvas.manager.set_window_title(f"Live View - Alpao DM {self._dm.nActs}")
        simg = self._dm._wavefront(
            zernike=zernike2remove, surf=self._surf, noisy=self._noisy
        )
        if self.full_frame:
            simg = self.intoFullFrame(simg)
        im = ax.imshow(simg, cmap=cmap)
        ax.axis("off")
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.9)
        pv_txt = fig.text(0.5, 0.1, "", ha="center", va="center", fontsize=15)
        shape_txt = fig.text(0.5, 0.925, "", ha="center", va="center", fontsize=15)
        fps_txt = fig.text(0.5, 0.1, "", ha="center", va="center", fontsize=15)

        # Closing Event
        def on_close(event):
            self._live = False
            self._dm._live = False

        fig.canvas.mpl_connect("close_event", on_close)

        # Update Event
        def update(frame):
            new_img = self._dm._wavefront(
                zernike=self.shapesRemoved, surf=self._surf, noisy=self._noisy
            )
            if self.full_frame:
                new_img = self.intoFullFrame(new_img)
            if not self._surf:
                fps_txt.set_text(f"FPS: {framerate:.1f}")
                pv_txt.set_text("")
                shape_txt.set_text("")
            else:
                pv = (_np.max(new_img) - _np.min(new_img)) * 1e6
                rms = _np.std(new_img) * 1e6
                pv_txt.set_text(
                    r"PV={:.3f} $\mu m$".format(pv)
                    + " " * 10
                    + r"RMS={:.5f} $\mu m$".format(rms)
                )
                stext = (
                    f"Removing Zernike modes {self.shapesRemoved}"
                    if self.shapesRemoved is not None
                    else ""
                )
                shape_txt.set_text(stext)
                fps_txt.set_text("")
            im.set_clim(
                vmin=new_img.min(), vmax=new_img.max()
            )  # to not have blank plot
            im.set_data(new_img)
            return (im,)

        # Create and hold a reference to the animation.
        self._anim = _FuncAnimation(
            fig,
            func=update,
            interval=(1000 / framerate),
            blit=False,
            cache_frame_data=False,
        )
        _plt.show(block=False)

        # force an `update()` to update the figure
        _plt.pause(0.5)
        update(0)

        return fig, self._anim

    def acquire_map(self, nframes: int = 1, rebin: int = 1):
        """
        Acquires the phase map of the interferometer.
        
        Parameters
        ----------
        nframes : int, optional
            Number of frames to be averaged.
        rebin : int, optional
            Rebin factor to reduce the resolution of the image.

        Returns
        -------
        np.array
            Phase map of the interferometer.
        """
        imglist = []
        for _ in range(nframes):
            img = self._dm._shape
            kk = _np.floor(_np.random.random(1) * 5 - 2)
            masked_ima = img + _np.ones(img.shape) * self._lambda * kk
            imglist.append(masked_ima)
        image = _np.ma.dstack(imglist)
        image = _np.mean(image, axis=2)
        masked_img = _np.ma.masked_array(image, mask=self._dm.mask)
        fimage = rebinned(masked_img, rebin)
        if self.full_frame:
            fimage = self.intoFullFrame(fimage)
        if self.shapesRemoved is not None:
            fimage = zern.removeZernike(fimage, self.shapesRemoved)
        if self._freeze:
            if self._live:
                self._surf = True
                _plt.pause(1)
                self._surf = False
        return fimage

    def intoFullFrame(self, img: _t.ImageData = None):
        """
        Converts the image to a full frame image of 2000x2000 pxs.

        Parameters
        ----------
        img : np.array
            Image to be converted to a full frame.

        Returns
        -------
        full_frame : np.array
            Full frame image.
        """
        if img is None:
            self.full_frame = True
            return
        params = self.getCameraSettings()
        ocentre = (params["Width"] // 2 - 1, params["Height"] // 2 - 1)
        ncentre = (self._fW // 2 - 1, self._fH // 2 - 1)
        offset = (ncentre[0] - ocentre[0], ncentre[1] - ocentre[1])
        newidx = (self._dm._idx[0] + offset[0], self._dm._idx[1] + offset[1])
        full_frame = _np.zeros((self._fW, self._fH))
        full_frame[newidx] = img.compressed()
        new_mask = full_frame == 0
        full_frame = _np.ma.masked_array(full_frame, mask=new_mask)
        return full_frame
    
    def acquireFullFrame(self, **kwargs: dict[str,_t.Any]):
        """
        Acquires the phase map of the interferometer in full frame mode.
        
        Parameters
        ----------
        nframes : int, optional
            Number of frames to be averaged.
        rebin : int, optional
            Rebin factor to reduce the resolution of the image.
        

        Returns
        -------
        np.array
            Full frame phase map of the interferometer.
        """
        return self.intoFullFrame(self.acquire_map(**kwargs))


    # --------------------------------------------------------------------------
    # Series of functions to control the behaviour of the live interferometer
    # --------------------------------------------------------------------------
    
    def toggleShapeRemoval(self, modes: list[int]):
        """
        Removes the acquired shape by the define Zernike modes.

        Parameters
        ----------
        modes : np.array
            Modes to be filtered out.
        """
        self.shapesRemoved = modes

    def toggleSurfaceView(self):
        """
        Continuously acquires the phase map of the interferometer.

        In reality, instead of the fringes, it will show the surface
        shape acquired of the dm.
        """
        self._surf = not self._surf

    def toggleAcquisitionLiveFreeze(self):
        """
        Freezes the live wavefront when acquiring, to show the
        measured surface.
        """
        self._freeze = not self._freeze

    def toggleLiveNoise(self):
        """
        Adds noise to the live wavefront.
        """
        self._noisy = not self._noisy

    def live_info(self):
        """
        Prints the current state of the interferometer.

        Returns
        -------
        dict
            Current state of the live interferometer.
        """
        params = self.getCameraSettings()
        state = f"""
{self.model}
--------------------------
Full Frame Size    : {self._fW}x{self._fH}
Actual Frame Size  : {params['Width']}x{params['Height']}
Offsets            : ({params['x-offset']}, {params['y-offset']})
Framerate          : {self._fps:.2f} Hz

Live Interferometer Info:
--------------------------
Shape Removal      : {self.shapesRemoved}
Surface View       : {self._surf}
Freeze on Acqu.    : {self._freeze}
Noise              : {self._noisy}"""
        print(state)

    # ==========================================================================

    def getCameraSettings(self):
        """
        Reads the configuration of the 4D interferometer.

        Returns
        -------
        dict
            Configuration file of the 4D interferometer.
        """
        data = _conf
        params = {}
        params["Width"] = int(data["width"])
        params["Height"] = int(data["height"])
        params["x-offset"] = int(data["x-offset"])
        params["y-offset"] = int(data["y-offset"])
        return params

    def _readFullFrameSize(self):
        """
        Reads the full frame size of the 4D interferometer.

        Returns
        -------
        tuple
            Full frame size of the 4D interferometer.
        """
        data = _conf
        return (int(data["full_width"]), int(data["full_height"]))
