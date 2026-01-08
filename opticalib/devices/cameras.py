import vmbpy as _vmbpy
from .. import typings as _ot
from ..ground.logger import SystemLogger as _sl
from ..core.read_config import getCamerasConfig as _gcc
from contextlib import contextmanager as _contextmanager


class AVTCamera:

    def __init__(self, name: str):
        """
        Class which interfaces AVT cameras using the VimbaXPy API.

        Parameters:
        -----------
        cam_id : str
            The ID of the camera to be used, as defined in the configuration
            file.
        """
        self._name = name
        self._cam_config = _gcc(device_name=self._name)

        # retrieve device ID or IP
        try:
            self.cam_id = self._cam_config["id"]
        except KeyError:
            self.cam_id = self._cam_config["ip"]

        # Try to connect to the camera
        try:
            _ = self._get_camera()
            repr = self.__str__()
            if "ip" in self._cam_config.keys():
                ip = self._cam_config["ip"]
                repr += f"/// IP Address    : {ip}"
            print(f"Connected to camera:\n{repr}")
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to camera {self._name} with ID {self.cam_id}."
            ) from e
        
        self._logger = _sl(__class__)

    def get_exptime(self) -> float:
        """
        Gets the exposure time of the camera in micro-seconds.

        Returns:
        --------
        exposure_time : float
            The exposure time in micro-seconds.
        """
        with self._prepare_camera() as cam:
            exptimeFeat = cam.get_feature_by_name("ExposureTimeAbs")
            exposure_time = exptimeFeat.get()
        return exposure_time

    def set_exptime(self, exptime_us: float):
        """
        Sets the exposure time of the camera.

        Parameters:
        -----------
        exptime_us : float
            The exposure time in micro-seconds.
        """
        with self._prepare_camera() as cam:
            self._logger.info('Setting exposure time to {} us'.format(exptime_us))
            exptimeFeat = cam.get_feature_by_name("ExposureTimeAbs")
            exptimeFeat.set(exptime_us)

    def acquire_frames(
        self,
        n_frames: int = 1,
        timeout: int = 1000,
        mode: str = "sync",
        allocation_mode: int = 0,
    ) -> _ot.ImageData | _ot.CubeData:
        """
        Acquires frames from the camera.

        Parameters:
        -----------
        n_frames : int
            The number of frames to acquire.
        timeout : int
            The timeout in milliseconds.
        mode : str
            The acquisition mode. Can be 'sync' (synchronous) or 'async' (asynchronous).
        allocation_mode : vmbpy.AllocationMode
            The allocation mode for asynchronous acquisition. Options are:
            - 0 (vmbpy.AllocationMode.AnnounceFrame) : buffer allocated by `vmbpy`
            - 1 (vmbpy.AllocationMode.AllocAndAnnounceFrame) : buffer allocated by the Transport Layer
        """
        frames = []
        with self._prepare_camera() as cam:

            if mode == "sync":
                self._logger.info('Starting synchronous acquisition')
                self._logger.info(f'Acquiring {n_frames} frames with timeout {timeout} ms')
                if n_frames > 1:
                    for f in cam.get_frame_generator(
                        limit=n_frames, timeout_ms=timeout
                    ):
                        frames.append(f.as_numpy_ndarray().transpose(2, 0, 1))
                else:
                    frames.append(
                        cam.get_frame(timeout_ms=timeout)
                        .as_numpy_ndarray()
                        .transpose(2, 0, 1)
                    )

            elif mode == "async":
                self._logger.info('Starting asynchronous acquisition')
                self._logger.info(f'Acquiring frames until Enter is pressed')
                aframes = []

                def frame_handler(
                    cam: _vmbpy.Camera, stream: _vmbpy.Stream, frame: _vmbpy.Frame
                ):
                    print("{} acquired {}".format(cam, frame), flush=True)
                    aframes.append(frame)
                    cam.queue_frame(frame)

                try:
                    am = (
                        _vmbpy.AllocationMode.AnnounceFrame
                        if allocation_mode == 0
                        else _vmbpy.AllocationMode.AllocAndAnnounceFrame
                    )
                    self._logger.info("Waiting for stop trigger (Enter)...")
                    cam.start_streaming(
                        handler=frame_handler, buffer_count=10, allocation_mode=am
                    )
                    input()

                finally:
                    cam.stop_streaming()

                frames = [f.as_numpy_ndarray().transpose(2, 0, 1) for f in aframes]

            else:
                self._logger.error('Invalid acquisition mode specified')
                raise ValueError("Invalid mode. Choose either 'sync' or 'async'.")

        # Remove first dimension, since it's 1
        frames = [f.squeeze(0) if f.shape[0] == 1 else f for f in frames]
        if len(frames) == 1:
            frames = frames[0]
        else:
            from ..analyzer import createCube as _cC
            
            frames = _cC(frames)

        return frames

    @_contextmanager
    def _prepare_camera(self):
        """
        Context manager to prepare the camera for use.
        """
        self._logger.info('Retrieving camera instance')
        with _vmbpy.VmbSystem.get_instance():
            with self._get_camera() as cam:
                # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
                try:
                    stream = cam.get_streams()[0]
                    stream.GVSPAdjustPacketSize.run()
                    while not stream.GVSPAdjustPacketSize.is_done():
                        pass

                except (AttributeError, _vmbpy.VmbFeatureError):
                    pass
                self._logger.info('Camera instance ready')
                yield cam

    def _get_camera(self):
        """
        Retrieves the camera object using the Vimba API.

        Returns:
        --------
        cam : vmbpy.Camera
            The camera object.
        """
        self._logger.info(f'Getting camera with ID: {self.cam_id}')
        with _vmbpy.VmbSystem.get_instance() as vimba:
            return vimba.get_camera_by_id(self.cam_id)

    def __str__(self):
        """
        Returns a string representation of the camera.
        """
        with _vmbpy.VmbSystem.get_instance():
            with self._get_camera() as cam:
                text = ""
                text += "/// Camera Name   : {}\n".format(
                    " ".join(cam.get_name().split(" ")[:-3])
                )
                text += "/// Model Name    : {}\n".format(cam.get_model())
                text += "/// Camera ID     : {}\n".format(cam.get_id())
                text += "/// Serial Number : {}\n".format(cam.get_serial())
                text += "/// Interface ID  : {}\n".format(cam.get_interface_id())
                return text
