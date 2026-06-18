import time as _time

import vmbpy as _vmbpy

from .. import typings as _ot
from ..core.decorators import ReconnectionError as _re
from ..core.decorators import allow_reconnect as _ar
from ..ground.logger import SystemLogger as _sl
from ..core.read_config import getCamerasConfig as _gcc


class GigaVision:

    def __init__(self, name: str):
        """
        Class which interfaces AVT cameras using the VimbaXPy API.

        Parameters:
        -----------
        name : str
            The name of the camera to be used, as defined in the configuration
            file.
        """
        self._name = name
        self._cam_config = _gcc(device_name=self._name)
        self._logger = _sl(__class__)
        self._base_timeout = 2000  # milliseconds
        self._exptime = None

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
                self._exptime = None
                self._logger.info(
                    f"Successfully reconnected to camera {self._name}"
                )
                return

            except Exception as e:
                self._logger.warning(
                    f"Reconnection attempt {attempt + 1} failed: {e}"
                )
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
        if hasattr(self, "_cam") and self._cam is not None:
            try:
                self._cam.__exit__(None, None, None)
            except Exception as e:
                self._logger.debug(f"Exception while closing camera: {type(e).__name__}: {e}")
            finally:
                self._cam = None

        if hasattr(self, "_vimba") and self._vimba is not None:
            try:
                self._vimba.__exit__(None, None, None)
            except Exception as e:
                self._logger.debug(f"Exception while closing VmbSystem: {type(e).__name__}: {e}")
            finally:
                self._vimba = None

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

    @_ar(error_instance=(_vmbpy.VmbFeatureError,AttributeError))
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

    @_ar(error_instance=(_vmbpy.VmbFeatureError,AttributeError))
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

    @_ar(error_instance=(_vmbpy.VmbCameraError,_vmbpy.VmbTimeout,AttributeError))
    def acquire_frames(
        self,
        nframes: int | None = None,
        multiframe_out_mode: str = "mean",
        mode: str = "sync",
        allocation_mode: int = 0,
    ) -> _ot.ImageData | _ot.CubeData:
        """
        Acquire frames from the camera.

        Parameters
        ----------
        nframes : int | None
                The number of frames to acquire. If in `sync` mode and None,
                acquires a single frame, while if in `async` mode and None,
                acquires frames until stopped.
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
        frames = []
        cam = self._cam

        if mode == "sync":
            exptimeInMs = max(1, int(self.get_exptime() / 1000))
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
                    cam.get_frame(
                        timeout_ms=int(self._base_timeout * exptimeInMs)
                    )
                    .as_numpy_ndarray()
                    .transpose(2, 0, 1)
                )

        elif mode == "async":
            import time

            exposure_time = self.get_exptime()

            self._logger.info("Starting asynchronous acquisition")
            self._logger.info(f"Acquiring frames until Enter is pressed")
            aframes = []

            def frame_handler(
                cam: _vmbpy.Camera, stream: _vmbpy.Stream, frame: _vmbpy.Frame
            ):
                print("{} acquired {}".format(cam, frame), flush=True)
                aframes.append(frame)
                cam.queue_frame(frame)

            am = (
                _vmbpy.AllocationMode.AnnounceFrame
                if allocation_mode == 0
                else _vmbpy.AllocationMode.AllocAndAnnounceFrame
            )
            self._logger.info("Waiting for stop trigger (Enter)...")
            cam.start_streaming(
                handler=frame_handler, buffer_count=10, allocation_mode=am
            )
            if nframes is None:
                input()  # wait until Enter is pressed
                cam.stop_streaming()
            else:
                while len(aframes) < nframes:
                    time.sleep(exposure_time / 1_000_000)
                cam.stop_streaming()

            frames = [f.as_numpy_ndarray().transpose(2, 0, 1) for f in aframes]

        else:
            self._logger.error("Invalid acquisition mode specified")
            raise ValueError("Invalid mode. Choose either 'sync' or 'async'.")

        # Remove first dimension, since it's 1
        frames = [f.squeeze(0) if f.shape[0] == 1 else f for f in frames]
        if len(frames) == 1:
            frames = frames[0]
        else:
            from ..analyzer import createCube as _cC

            frames = _cC(frames)

            if multiframe_out_mode == "mean":
                from numpy.ma import mean

                frames = mean(frames, axis=2)

        return frames

    def set_base_timeout(self, timeout_ms: int):
        """
        Sets the base timeout for camera operations.

        Parameters:
        -----------
        timeout_ms : int
            The base timeout in milliseconds.
        """
        self._base_timeout = timeout_ms

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
