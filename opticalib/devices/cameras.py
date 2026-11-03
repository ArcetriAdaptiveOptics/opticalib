"""
Camera drivers — Allied Vision GigE cameras via Vimba SDK
==========================================================

High-level wrapper for Allied Vision cameras using the Vimba SDK
(via ``vmbpy``).  Provides ``acquire_frames``, exposure-time control
and frame retrieval.

"""

import time as _time
import vmbpy as _vmbpy
import threading as _th

from ..core import _types as _ot
from ..core.decorators import allow_reconnect as _ar
from ..ground.logger import SystemLogger as _sl
from ..core.config import get_cameras_config as _gcc
from ._API.base_devices import BaseCamera
from ._API.cameras.frame_buffer import FrameBuffer as _FrameBuffer
from ._API.cameras.GigE_live_viewer import ViewerProcess as _ViewerProcess


def _frame_to_array(frame: _vmbpy.Frame) -> _ot.ImageData:
    """Copy a Vimba frame into a numpy array: (H, W) mono, (C, H, W) color."""
    img = frame.as_numpy_ndarray()  # a *view* on the SDK buffer, shape (H, W, C)
    img = img[:, :, 0] if img.shape[2] == 1 else img.transpose(2, 0, 1)
    # copy: the SDK reuses the buffer as soon as the frame is re-queued
    return img.copy()


class GigaVision(BaseCamera):

    def __init__(self, name: str):
        """
        Class which interfaces AVT cameras using the VimbaXPy API.

        Parameters
        -----------
        name : str
            The name of the camera to be used, as defined in the configuration
            file.
        """
        self._name = name
        self._cam_config = _gcc(device_name=self._name)
        self._logger = _sl(__class__)
        self._base_timeout = 2000  # milliseconds

        # Streaming state, shared between threads:
        # - _buffer: latest frames, filled by the SDK thread (see _on_frame);
        # - _viewer: live viewer process handle, or None. Read by the SDK
        #   thread, set/cleared by user threads: guarded by _viewer_lock;
        # - _stream_lock: serializes start/stop of streaming, which can be
        #   requested by the user and by the viewer watcher thread at once.
        self._buffer = _FrameBuffer(self._cam_config.get("buffer_size") or 100)
        self._viewer = None
        self._viewer_lock = _th.Lock()
        self._stream_lock = _th.Lock()

        # retrieve device ID or IP
        try:
            self.cam_id = self._cam_config["id"]
        except KeyError:
            self.cam_id = None

        try:
            self.cam_ip = self._cam_config["ip"]
        except KeyError:
            self.cam_ip = None

        if all([self.cam_id is None, self.cam_ip is None]):
            raise ValueError(
                f"Camera configuration for {self._name} must contain either 'id' or 'ip'."
            )

        # Connect to Vimba and the Camera persistently
        try:
            self._vimba = _vmbpy.VmbSystem.get_instance()
            self._vimba.__enter__()

            if self.cam_id is not None:
                self._cam = self._vimba.get_camera_by_id(self.cam_id)
            else:
                self._cam = self._vimba.get_camera_by_id(self.cam_ip)

            self._cam.__enter__()

            # Try to adjust GeV packet size once upon connection. This Feature is only available for GigE - Cameras.
            try:
                stream = self._cam.get_streams()[0]
                stream.GVSPAdjustPacketSize.run()
                while not stream.GVSPAdjustPacketSize.is_done():
                    pass
            except (AttributeError, _vmbpy.VmbFeatureError):
                pass

            repr_str = self.__str__()
            if "ip" in self._cam_config.keys():
                ip = self._cam_config["ip"]
                repr_str += f"/// IP Address    : {ip}"
            print(f"Connected to camera:\n{repr_str}")

        except Exception as e:
            self.close()
            raise RuntimeError(
                f"Could not connect to camera {self._name} with ID {self.cam_id}."
            ) from e

        self._exptime = None
        self._exptime = self.get_exptime()

    def reconnect(self, max_attempts: int = 2) -> None:
        """
        Attempt to reconnect to the camera after a disconnection.

        This method is called by the ``allow_reconnection`` decorator when a
        disconnection related error is detected. It closes the current connection
        and attempts to re-establish it.

        Parameters
        ----------
        max_attempts : int, optional
            Maximum number of reconnection attempts. The default is 2.

        Raises
        ------
        RuntimeError
            If reconnection fails after all attempts.
        """
        attempt = 0

        while attempt < max_attempts:
            try:
                self._logger.info(
                    f"Attempting to reconnect to camera {self._name} "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
                self.close()
                _time.sleep(0.25)

                self._vimba = _vmbpy.VmbSystem.get_instance()
                self._vimba.__enter__()

                if self.cam_id is not None:
                    self._cam = self._vimba.get_camera_by_id(self.cam_id)
                else:
                    self._cam = self._vimba.get_camera_by_id(self.cam_ip)

                self._cam.__enter__()
                self._exptime = self.get_exptime()
                self._exptime = self.get_exptime()
                self._logger.info(f"Successfully reconnected to camera {self._name}")
                return

            except Exception as e:
                self._logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
                attempt += 1

        raise RuntimeError(
            f"Failed to reconnect to camera {self._name} after "
            f"{max_attempts} attempts"
        )

    def close(self):
        """
        Gracefully close the camera and VmbSystem context.

        This method safely closes both the camera and VmbSystem connections,
        suppressing any exceptions that may occur during cleanup. This
        ensures that even if the camera is in a disconnected or error state,
        the cleanup completes without raising exceptions.
        """
        # the viewer and the stream must go before the camera they depend on
        # (getattr: close() may run on a half-initialized object; try: the
        # camera may already be dead, e.g. when called by reconnect())
        try:
            if getattr(self, "_viewer", None) is not None:
                self._logger.warning("Closing camera: live view stopped")
                self.stop_live_view()
            if getattr(self, "_stream_lock", None) is not None:
                self.stop_streaming()
        except Exception as e:
            self._logger.debug(f"Exception while stopping stream: {e}")

        if hasattr(self, "_cam") and self._cam is not None:
            try:
                self._cam.__exit__(None, None, None)
            except Exception as e:
                self._logger.debug(
                    f"Exception while closing camera: {type(e).__name__}: {e}"
                )
            finally:
                self._cam = None

        if hasattr(self, "_vimba") and self._vimba is not None:
            try:
                self._vimba.__exit__(None, None, None)
            except Exception as e:
                self._logger.debug(
                    f"Exception while closing VmbSystem: {type(e).__name__}: {e}"
                )
            finally:
                self._vimba = None

    @_ar(error_instance=(_vmbpy.VmbFeatureError, AttributeError))
    def get_exptime(self) -> float:
        """
        Get the exposure time of the camera in micro-seconds.

        Returns
        -------
        float
                The exposure time in micro-seconds.
        """
        if self._exptime is None:
            exptimeFeat = self._cam.get_feature_by_name("ExposureTimeAbs")
            self._exptime = exptimeFeat.get()
        return self._exptime

    @_ar(error_instance=(_vmbpy.VmbFeatureError, AttributeError))
    def set_exptime(self, exptime_us: float):
        """
        Set the exposure time of the camera.

        Parameters
        ----------
        exptime_us : float
                The exposure time in micro-seconds.
        """
        if self._exptime == exptime_us:
            self._logger.info(
                f"Exposure time is already set to {exptime_us} us, skipping."
            )
            return
        self._logger.info("Setting exposure time to {} us".format(exptime_us))
        exptimeFeat = self._cam.get_feature_by_name("ExposureTimeAbs")
        exptimeFeat.set(exptime_us)
        self._exptime = exptime_us
        # frames stored so far were exposed with the old value: drop them.
        # (A frame already being exposed may still arrive with the old value.)
        self._buffer.clear()

    # ------------------------------------------------------------------ #
    # Streaming: the camera pushes frames continuously into self._buffer #
    # ------------------------------------------------------------------ #
    @property
    def is_streaming(self) -> bool:
        """Whether the camera is currently streaming frames."""
        return self._cam is not None and self._cam.is_streaming()

    def start_streaming(self, allocation_mode: int = 0) -> None:
        """
        Start continuous acquisition into the internal frame buffer.

        Frames are delivered by the Vimba SDK on its own thread (see
        ``_on_frame``): this call returns immediately.

        Parameters
        ----------
        allocation_mode : int
            0 for buffers allocated by ``vmbpy``, 1 for buffers allocated by
            the Transport Layer.
        """
        with self._stream_lock:
            if self.is_streaming:
                return
            am = (
                _vmbpy.AllocationMode.AnnounceFrame
                if allocation_mode == 0
                else _vmbpy.AllocationMode.AllocAndAnnounceFrame
            )
            self._buffer.clear()
            self._logger.info("Starting streaming")
            self._cam.start_streaming(
                handler=self._on_frame, buffer_count=10, allocation_mode=am
            )

    def stop_streaming(self) -> None:
        """Stop continuous acquisition (blocks until the SDK thread is done)."""
        with self._stream_lock:
            if self.is_streaming:
                self._logger.info("Stopping streaming")
                self._cam.stop_streaming()
                self._buffer.interrupt()  # waiters raise now, not at timeout
                self._buffer.clear()  # free the memory of the stored frames

    def _on_frame(self, cam: _vmbpy.Camera, stream: _vmbpy.Stream, frame: _vmbpy.Frame):
        """
        Frame handler: runs in the **Vimba SDK thread**, once per frame.

        It must be fast and must never block: the SDK owns only a few frame
        buffers (``buffer_count``) and each one comes back to the camera only
        when we re-queue it. Do not call ``stop_streaming`` from here: it waits
        for this very handler to return, i.e. it would deadlock.
        """
        try:
            if frame.get_status() == _vmbpy.FrameStatus.Complete:
                img = _frame_to_array(frame)
                self._buffer.push(img)
                # hold the lock while writing, so that the viewer cannot be
                # closed (and its shared memory freed) in the middle of it
                with self._viewer_lock:
                    if self._viewer is not None:
                        self._viewer.write(img)
        finally:
            # always give the buffer back to the camera, even on errors
            cam.queue_frame(frame)

    @_ar(error_instance=(_vmbpy.VmbCameraError, _vmbpy.VmbTimeout, AttributeError))
    def acquire_frames(
        self,
        nframes: int | None = None,
        multiframe_out_mode: str = "mean",
        mode: str = "sync",
        allocation_mode: int = 0,
    ) -> _ot.ImageData | _ot.CubeData:
        """
        Acquire frames from the camera.

        If the camera is streaming (e.g. a live view is open), the frames are
        taken from the stream, whatever the ``mode``: the call waits for
        ``nframes`` frames delivered *after* it started (never older ones).
        In this case ``nframes`` cannot exceed the buffer size (config key
        ``buffer_size``, default 100).

        Parameters
        ----------
        nframes : int | None
                The number of frames to acquire. If in `sync` mode and None,
                acquires a single frame, while if in `async` mode and None,
                acquires frames until Enter is pressed.
        multiframe_out_mode : str
                The output mode for multiple frames. Can be 'cube' to return
                a cube of frames, or 'mean' to return the mean frame.
                Defaults to `mean`.
        mode : str
                The acquisition mode. Can be 'sync' (synchronous) or
                'async' (asynchronous).
        allocation_mode : vmbpy.AllocationMode
                The allocation mode for asynchronous acquisition. Options are:

                - 0 (vmbpy.AllocationMode.AnnounceFrame): buffer allocated
                  by `vmbpy`
                - 1 (vmbpy.AllocationMode.AllocAndAnnounceFrame): buffer
                  allocated by the Transport Layer

        Returns
        -------
        ImageData | CubeData
                Acquired frame(s) as numpy array or cube depending on
                multiframe_out_mode.
        """
        if mode not in ("sync", "async"):
            self._logger.error("Invalid acquisition mode specified")
            raise ValueError("Invalid mode. Choose either 'sync' or 'async'.")

        frames = []
        cam = self._cam
        exptimeInMs = max(1, int(self.get_exptime() / 1000))

        def timeout_s(n: int) -> float:
            # timeout for n frames, in seconds (same budget as the sync path)
            return self._base_timeout * exptimeInMs * n / 1000

        if self.is_streaming:
            # The camera is busy streaming: get_frame() would fail, so wait
            # for fresh frames in the buffer filled by the SDK thread.
            n = nframes or 1
            self._logger.info(f"Acquiring {n} frames from the running stream")
            frames = self._buffer.wait_new(n, timeout=timeout_s(n))

        elif mode == "sync":
            self._logger.info("Starting synchronous acquisition")
            self._logger.info(
                f"Acquiring {nframes} frames with timeout {self._base_timeout*exptimeInMs} ms"
            )
            if nframes is not None and nframes > 1:
                import copy

                for f in cam.get_frame_generator(
                    limit=nframes,
                    timeout_ms=int(self._base_timeout * exptimeInMs * nframes),
                ):
                    frames.append(
                        copy.deepcopy(f).as_numpy_ndarray().transpose(2, 0, 1)
                    )
            else:
                frames.append(
                    cam.get_frame(timeout_ms=int(self._base_timeout * exptimeInMs))
                    .as_numpy_ndarray()
                    .transpose(2, 0, 1)
                )

        else:  # async: stream just for this acquisition, then stop
            self._logger.info("Starting asynchronous acquisition")
            self.start_streaming(allocation_mode=allocation_mode)
            try:
                if nframes is None:
                    start = self._buffer.count
                    # the SDK thread fills the buffer while we wait here
                    input("Acquiring frames: press Enter to stop...")
                    frames = self._buffer.snapshot()
                    if not frames:
                        raise RuntimeError("No frames acquired before stopping.")
                    if self._buffer.count - start > self._buffer.capacity:
                        self._logger.warning(
                            f"Only the last {self._buffer.capacity} frames kept"
                        )
                else:
                    frames = self._buffer.wait_new(nframes, timeout=timeout_s(nframes))
            finally:
                self.stop_streaming()

        # Remove first dimension, since it's 1
        frames = [f.squeeze(0) if f.shape[0] == 1 else f for f in frames]
        if len(frames) == 1:
            frames = frames[0]
        else:
            from ..analyzer import create_cube as _cC

            frames = _cC(frames)

            if multiframe_out_mode == "mean":
                from numpy.ma import mean

                frames = mean(frames, axis=2)

        return frames

    # ------------------------------------------------------------------ #
    # Live view: a separate process displaying the stream               #
    # ------------------------------------------------------------------ #
    @property
    def is_live(self) -> bool:
        """Whether a live view window is open."""
        return self._viewer is not None

    def live_view(self, interval_ms: int = 40, **plotkwargs: dict[str, _ot.Any]):
        """
        Open a live view of the camera feed, in a separate process.

        Returns immediately: the terminal stays usable, and busy commands do
        not freeze the window. Streaming is started if needed, and stopped
        again when the view is closed (by the window or ``stop_live_view``)
        if it was started here. While live, ``acquire_frames`` reads from the
        stream.

        Parameters
        ----------
        interval_ms : int, optional
            Refresh period of the window in milliseconds. Default 40 (25 Hz).
        plotkwargs : dict[str, Any]
            ``title``, ``xlabel``, ``ylabel``, ``clabel`` and any keyword
            accepted by ``imshow`` (e.g. ``cmap``, ``vmin``, ``vmax``). Must
            be JSON-serializable. Giving ``vmin``/``vmax`` disables
            autoscaling.
        """
        if self.is_live:
            self._logger.info("Live view already running")
            return
        owns_stream = not self.is_streaming
        self.start_streaming()
        try:
            exptimeInMs = max(1, int(self.get_exptime() / 1000))
            # the first frame tells us the shape/dtype of the shared memory
            first = self._buffer.wait_new(1, timeout=self._base_timeout * exptimeInMs / 1000)[0]
            if first.ndim != 2:
                raise ValueError("Live view supports only monochrome cameras.")
            viewer = _ViewerProcess(
                first.shape, first.dtype, f"{self._name} live view", plotkwargs, interval_ms
            )
        except Exception:
            if owns_stream:
                self.stop_streaming()
            raise
        viewer.owns_stream = owns_stream
        viewer.write(first)
        with self._viewer_lock:
            self._viewer = viewer  # from now on, _on_frame feeds the viewer

        # A "watcher" thread blocks on the child process, so that closing the
        # window cleans up without anyone polling. daemon=True: it does not
        # keep the interpreter alive at exit.
        _th.Thread(
            target=self._watch_viewer, args=(viewer,), name=f"{self._name}-viewer-watch", daemon=True
        ).start()
        self._logger.info("Live view started")

    def stop_live_view(self) -> None:
        """Close the live view window (no-op if none is open)."""
        viewer = self._viewer
        if viewer is not None:
            self._detach_viewer(viewer)

    def _watch_viewer(self, viewer: _ViewerProcess) -> None:
        """Runs in the watcher thread: wait for the window to close."""
        viewer.wait()  # blocks this thread only
        self._detach_viewer(viewer)

    def _detach_viewer(self, viewer: _ViewerProcess) -> None:
        """
        Disconnect and close ``viewer``. Safe to call from several threads:
        only the first caller for a given viewer does the cleanup.
        """
        with self._viewer_lock:
            # "check and set" must be atomic, hence inside the lock. The
            # identity check also stops an old watcher from closing a newer
            # viewer opened in the meantime.
            if self._viewer is not viewer:
                return
            self._viewer = None
        # outside the lock: these calls can take a while, and _on_frame needs
        # the lock for every frame
        viewer.close()
        if viewer.owns_stream:
            self.stop_streaming()
        self._logger.info("Live view stopped")

    def set_base_timeout(self, timeout_ms: int):
        """
        Sets the base timeout for camera operations.

        Parameters
        -----------
        timeout_ms : int
            The base timeout in milliseconds.
        """
        self._base_timeout = timeout_ms

    def __del__(self):
        """
        Ensure connection is closed upon object deletion.

        This method is called when the object is garbage collected.
        It safely closes the connection without raising exceptions,
        even if the camera is already in a disconnected state.
        """
        try:
            self.close()
        except Exception:
            pass

    def __str__(self):
        """
        Returns a string representation of the camera.
        """
        text = ""
        text += "/// Camera Name   : {}\n".format(
            " ".join(self._cam.get_name().split(" ")[:-3])
        )
        text += "/// Model Name    : {}\n".format(self._cam.get_model())
        text += "/// Camera ID     : {}\n".format(self._cam.get_id())
        text += "/// Serial Number : {}\n".format(self._cam.get_serial())
        text += "/// Interface ID  : {}\n".format(self._cam.get_interface_id())
        return text

    def __repr__(self):
        arg1 = f"id={self.cam_id}" if self.cam_id is not None else f"ip={self.cam_ip}"
        return f"{self._name}({arg1}, exptime={self._exptime} us)"
