"""
This module contains the high-level classes for the use of interferometer devices.

Author(s)
---------
- Pietro Ferraiuolo : pietro.ferraiuolo@inaf.it

"""

import os as _os
import numpy as _np
import time as _time
import shutil as _sh
from . import _API as _api
from opticalib.analyzer import modeRebinner as _modeRebinner
from opticalib.ground.osutils import findTracknum as _ft, _InterferometerConverter
from opticalib.core import root as _fn
from opticalib import typings as _ot

_folds = _fn.folders
_confReader = _fn.ConfSettingReader4D
_OPDIMG = _folds.OPD_IMAGES_ROOT_FOLDER


class _4DInterferometer(_api.BaseInterferometer):
    """
    Class for the 4D Laser Interferometer.
    """

    def __init__(
        self, model: _ot.Optional[str | int] = None, ip: str = None, port: int = None
    ):
        """The constructor"""
        super().__init__(self.name, ip, port)
        self._i4d = _api.I4D(self.ip, self.port)
        self._ic = _InterferometerConverter()

    def acquire_map(
        self, nframes: int = 1, delay: int | float = 0, rebin: int = 1
    ) -> _ot.ImageData:
        """
        Acquires the interferometer image and returns it as a masked array.

        Parameters
        ----------
        nframes: int
            Number of frames to be averaged that produce the measurement.
        delay: int
            Delay between images in seconds.
        rebin: int
            Rebin factor for the image.

        Returns
        -------
        masked_ima: ImageData
            Interferometer image.
        """
        if nframes == 1:
            width, height, _, data_array = self._i4d.takeSingleMeasurement()
            masked_ima = self._fromDataArrayToMaskedArray(
                width, height, data_array * 632.8e-9
            )
            masked_ima = _modeRebinner(masked_ima, rebin)
        else:
            image_list = []
            for __ in range(nframes):
                width, height, _, data_array = self._i4d.takeSingleMeasurement()
                masked_ima = self._fromDataArrayToMaskedArray(
                    width, height, data_array * 632.8e-9
                )
                image_list.append(masked_ima)
                _time.sleep(delay)
            images = _np.ma.dstack(image_list)
            masked_ima = _np.ma.mean(images, 2)
            masked_ima = _modeRebinner(masked_ima, rebin)
        return masked_ima

    def acquire_detector(
        self, nframes: int = 1, delay: int | float = 0
    ) -> _ot.ImageData:
        """
        Parameters
        ----------
        nframes: int
            number of frames
        delay: int | flaot [s]
            delay between images

        Returns
        -------
        data2d: numpy masked array
                detector interferometer image
        """
        self.acquire_map()
        if nframes == 1:
            data, height, _, width = self._i4d.getFringeAmplitudeData()
            data2d = _np.reshape(data, (width, height))
        else:
            image_list = []
            for __ in range(nframes):
                data, height, _, width = self._i4d.getFringeAmplitudeData()
                data2d_t = _np.reshape(data, (width, height))
                image_list.append(data2d_t)
                _time.sleep(delay)
            images = _np.ma.dstack(image_list)
            data2d = _np.ma.mean(images, 2)
        return data2d

    def get_interferogram(self, index: int, rebin: int = 1) -> _ot.ImageData:
        """
        Acquires the interferogram at the specified index.

        Parameters
        ----------
        index: int
            Index of the interferogram to acquire.

        Returns
        -------
        data: numpy array
            Interferogram data.
        """
        json_data = self._i4d.getInterferogram(index)
        data = _np.array(json_data["Data"], dtype=float)
        # width = json_data["Width"]
        # height = json_data["Height"]
        # masked_ima = self._fromDataArrayToMaskedArray(
        #     width, height, data #* 632.8e-9
        # )
        # if rebin > 1:
        #     masked_ima = _modeRebinner(masked_ima, rebin)
        return data

    def capture(self, numberOfFrames: int, folder_name: str = None) -> str:
        """
        Parameters
        ----------
        numberOfFrames: int
            number of frames to acquire

        Other parameters
        ---------------
        folder_name: string
            if None a tacking number is generate

        Returns
        -------
        folder_name: string
            name of folder measurements
        """
        if folder_name is None:
            folder_name = self._ts()
        print(folder_name)

        self._i4d.burstFramesToSpecificDirectory(
            _os.path.join(_folds.CAPTURE_FOLDER_NAME_4D_PC, folder_name), numberOfFrames
        )
        return folder_name

    def produce(self, tn: str | list[str]) -> None:
        """
        Parameters
        ----------
        folder_name: string
            name of folder measurements to convert
        """
        if not isinstance(tn, list):
            tn = [tn]
        for t in tn:
            self._i4d.convertRawFramesInDirectoryToMeasurementsInDestinationDirectory(
                _os.path.join(_folds.PRODUCE_FOLDER_NAME_4D_PC, t),
                _os.path.join(_folds.CAPTURE_FOLDER_NAME_4D_PC, t),
            )
            _sh.move(
                _os.path.join(_folds.PRODUCE_FOLDER_NAME_LOCAL_PC, t),
                _folds.OPD_IMAGES_ROOT_FOLDER,
            )
            self._rename4D(t)
            self.copy4DSettings(t)

    def setTriggerMode(self, enable: bool) -> None:
        """
        Parameters
        ----------
        folder_name: string
            name of folder measurements to convert
        """
        enable = 1 if enable is True else 0
        self._i4d.setTriggerMode(enable)
        if enable == 1:
            print("Triggered mode enabled, waiting for TTL")
        else:
            print("Triggered mode disabled")

    def loadConfiguration(self, conffile: str) -> None:
        """
        Read and loads the configuration file of the interferometer.

        Parameters
        ----------
        conffile: string
            name of the configuration file to load
        """
        self._i4d.loadConfiguration(conffile)

    def copy4DSettings(self, destination: str) -> None:
        """
        Copies the interferometer settings file to the specified destination.
        """
        import shutil

        dest_fold = _os.path.join(_folds.OPD_IMAGES_ROOT_FOLDER, destination)
        shutil.copy(_folds.SETTINGS_CONF_FILE, dest_fold)
        shutil.move(
            _os.path.join(dest_fold, "AppSettings.ini"),
            _os.path.join(dest_fold, "4DSettings.ini"),
        )

    @staticmethod
    def getCameraSettings(tn: str = None) -> list[int]:
        """
        Reads che actual interferometer settings from its configuration file.

        Return
        ------
        output: list
        list of camera settings: [width_pixel, height_pixel, offset_x, offset_y]
        """
        if not tn is None:
            path = _ft(tn, complete_path=True)
            try:
                file_path = _os.path.join(path, _fn.COPIED_SETTINGS_CONF_FILE)
                setting_reader = _fn.ConfSettingReader4D(file_path)
            except Exception as e:
                print(f"Error: {e}")
                file_path = _os.path.join(path, "4DSettings.ini")
                setting_reader = _fn.ConfSettingReader4D(file_path)
        else:
            file_path = _folds.SETTINGS_CONF_FILE
            setting_reader = _confReader(file_path)
        width_pixel = setting_reader.getImageWidhtInPixels()
        height_pixel = setting_reader.getImageHeightInPixels()
        offset_x = setting_reader.getOffsetX()
        offset_y = setting_reader.getOffsetY()
        return [width_pixel, height_pixel, offset_x, offset_y]

    @staticmethod
    def getFrameRate(tn: str = None) -> float:
        """
        Reads the frame rate the interferometer is working at.

        Return
        ------
        frame_rate: float
            Frame rate of the interferometer
        """
        if not tn is None:
            path = _ft(tn, complete_path=True)
            try:
                file_path = _os.path.join(path, _fn.COPIED_SETTINGS_CONF_FILE)
                setting_reader = _fn.ConfSettingReader4D(file_path)
            except Exception as e:
                print(f"Error: {e}")
                file_path = _os.path.join(path, "4DSettings.ini")
                setting_reader = _fn.ConfSettingReader4D(file_path)
        else:
            file_path = _folds.SETTINGS_CONF_FILE
            setting_reader = _confReader(file_path)
        frame_rate = setting_reader.getFrameRate()
        return frame_rate

    def intoFullFrame(self, img: _ot.ImageData) -> _ot.ImageData:
        """
        The function fits the passed frame (expected cropped) into the
        full interferometer frame (2048x2048), after reading the cropping
        parameters.

        Parameters
        ----------
        img: ImageData
            The image to be fitted into the full frame.

        Return
        ------
        output: ImageData
            The output image, in the interferometer full frame.
        """
        off = (self.getCameraSettings())[2:4]
        off = _np.flip(off)
        nfullpix = _np.array([2048, 2048])
        fullimg = _np.full(nfullpix, _np.nan)  # was   _np.zeros(nfullpix)
        fullmask = _np.ones(nfullpix)
        offx = off[0]
        offy = off[1]
        sx = _np.shape(img)[0]  # croppar[2]
        sy = _np.shape(img)[1]  # croppar[3]
        fullimg[offx : offx + sx, offy : offy + sy] = img.data
        fullmask[offx : offx + sx, offy : offy + sy] = img.mask
        fullimg = _np.ma.masked_array(fullimg, fullmask)
        return fullimg

    def _fromDataArrayToMaskedArray(
        self, width: int, height: int, data_array: _ot.MatrixLike
    ) -> _ot.ImageData:
        """
        Converts the data array to a masked array.

        Parameters
        ----------
        width: int
            Width of the image.
        height: int
            Height of the image.
        data_array: MatrixLike
            Data array to be converted.

        Returns
        -------
        masked_ima: ImageData
            Masked image.
        """
        data = _np.reshape(data_array, (height, width))
        idx, idy = _np.where(_np.isnan(data))
        mask = _np.zeros((data.shape[0], data.shape[1]))
        mask[idx, idy] = 1
        masked_ima = _np.ma.masked_array(data, mask=mask.astype(bool))
        return masked_ima

    def _rename4D(self, folder: str) -> None:
        """
        Renames the produced 'x.4D' files into '0000x.4D'

        Parameters
        ----------
        folder : str
            The folder where the 4D data is stored.
        """
        fold = _os.path.join(_OPDIMG, folder)
        files = _os.listdir(fold)
        for file in files:
            if file.endswith(".4D"):
                num_str = file.split(".")[0]
                if num_str.isdigit():
                    num = int(num_str)
                    new_name = f"{num:05d}.4D"
                    old_file = _os.path.join(fold, file)
                    new_file = _os.path.join(fold, new_name)
                    _os.rename(old_file, new_file)


class AccuFiz(_4DInterferometer):
    """
    Class for the AccuFiz Laser Interferometer.
    """

    def __init__(
        self, model: _ot.Optional[str | int] = None, ip: str = None, port: int = None
    ):
        """The constructor"""
        self.name = "AccuFiz" + str(model)
        super().__init__(model, ip, port)


class PhaseCam(_4DInterferometer):
    """
    Class for the 4D Twyman-Green PhaseCam Laser Interferometer.
    """

    def __init__(
        self, model: _ot.Optional[str | int] = None, ip: str = None, port: int = None
    ):
        """The constructor"""
        self.name = "PhaseCam" + str(model)
        super().__init__(model, ip, port)


# class PhaseCam(_api.BaseInterferometer):
#     """
#     Class for the 4D Twyman-Green PhaseCam Laser Interferometer.
#     """

#     def __init__(
#         self, model: _ot.Optional[str | int] = None, ip: str = None, port: int = None
#     ):
#         """The constructor"""
#         self.name = "PhaseCam" + str(model)
#         super().__init__(self.name, ip, port)
#         self._i4d = _api.I4D(self.ip, self.port)
#         self._ic = _InterferometerConverter()

#     def acquire_map(
#         self, nframes: int = 1, delay: int | float = 0, rebin: int = 1
#     ) -> _ot.ImageData:
#         """
#         Acquires the interferometer image and returns it as a masked array.

#         Parameters
#         ----------
#         nframes: int
#             Number of frames to be averaged that produce the measurement.
#         delay: int
#             Delay between images in seconds.
#         rebin: int
#             Rebin factor for the image.

#         Returns
#         -------
#         masked_ima: ImageData
#             Interferometer image.
#         """
#         if nframes == 1:
#             width, height, _, data_array = self._i4d.takeSingleMeasurement()
#             masked_ima = self._fromDataArrayToMaskedArray(
#                 width, height, data_array * 632.8e-9
#             )
#             masked_ima = _modeRebinner(masked_ima, rebin)
#         else:
#             image_list = []
#             for __ in range(nframes):
#                 width, height, _, data_array = self._i4d.takeSingleMeasurement()
#                 masked_ima = self._fromDataArrayToMaskedArray(
#                     width, height, data_array * 632.8e-9
#                 )
#                 image_list.append(masked_ima)
#                 _time.sleep(delay)
#             images = _np.ma.dstack(image_list)
#             masked_ima = _np.ma.mean(images, 2)
#             masked_ima = _modeRebinner(masked_ima, rebin)
#         return masked_ima

#     def acquire_detector(
#         self, nframes: int = 1, delay: int | float = 0
#     ) -> _ot.ImageData:
#         """
#         Parameters
#         ----------
#         nframes: int
#             number of frames
#         delay: int | flaot [s]
#             delay between images

#         Returns
#         -------
#         data2d: numpy masked array
#                 detector interferometer image
#         """
#         self.acquire_map()
#         if nframes == 1:
#             data, height, _, width = self._i4d.getFringeAmplitudeData()
#             data2d = _np.reshape(data, (width, height))
#         else:
#             image_list = []
#             for __ in range(nframes):
#                 data, height, _, width = self._i4d.getFringeAmplitudeData()
#                 data2d_t = _np.reshape(data, (width, height))
#                 image_list.append(data2d_t)
#                 _time.sleep(delay)
#             images = _np.ma.dstack(image_list)
#             data2d = _np.ma.mean(images, 2)
#         return data2d

#     def _fromDataArrayToMaskedArray(
#         self, width: int, height: int, data_array: _ot.MatrixLike
#     ) -> _ot.ImageData:
#         """
#         Converts the data array to a masked array.

#         Parameters
#         ----------
#         width: int
#             Width of the image.
#         height: int
#             Height of the image.
#         data_array: MatrixLike
#             Data array to be converted.

#         Returns
#         -------
#         masked_ima: ImageData
#             Masked image.
#         """
#         data = _np.reshape(data_array, (height, width))
#         idx, idy = _np.where(_np.isnan(data))
#         mask = _np.zeros((data.shape[0], data.shape[1]))
#         mask[idx, idy] = 1
#         masked_ima = _np.ma.masked_array(data, mask=mask.astype(bool))
#         return masked_ima

#     def capture(self, numberOfFrames: int, folder_name: str = None) -> str:
#         """
#         Parameters
#         ----------
#         numberOfFrames: int
#             number of frames to acquire

#         Other parameters
#         ---------------
#         folder_name: string
#             if None a tacking number is generate

#         Returns
#         -------
#         folder_name: string
#             name of folder measurements
#         """
#         if folder_name is None:
#             folder_name = self._ts()
#         print(folder_name)

#         self._i4d.burstFramesToSpecificDirectory(
#             _os.path.join(_folds.CAPTURE_FOLDER_NAME_4D_PC, folder_name), numberOfFrames
#         )
#         return folder_name

#     def produce(self, tn: str | list[str]) -> None:
#         """
#         Parameters
#         ----------
#         folder_name: string
#             name of folder measurements to convert
#         """
#         if not isinstance(tn, list):
#             tn = [tn]
#         for t in tn:
#             self._i4d.convertRawFramesInDirectoryToMeasurementsInDestinationDirectory(
#                 _os.path.join(_folds.PRODUCE_FOLDER_NAME_4D_PC, t),
#                 _os.path.join(_folds.CAPTURE_FOLDER_NAME_4D_PC, t),
#             )
#             _sh.move(
#                 _os.path.join(_folds.PRODUCE_FOLDER_NAME_LOCAL_PC, t),
#                 _folds.OPD_IMAGES_ROOT_FOLDER,
#             )
#             self._rename4D(t)
#             self.copy4DSettings(t)

#     def setTriggerMode(self, enable: bool) -> None:
#         """
#         Parameters
#         ----------
#         folder_name: string
#             name of folder measurements to convert
#         """
#         enable = 1 if enable is True else 0
#         self._i4d.setTriggerMode(enable)
#         if enable == 1:
#             print("Triggered mode enabled, waiting for TTL")
#         else:
#             print("Triggered mode disabled")

#     def loadConfiguration(self, conffile: str) -> None:
#         """
#         Read and loads the configuration file of the interferometer.

#         Parameters
#         ----------
#         conffile: string
#             name of the configuration file to load
#         """
#         self._i4d.loadConfiguration(conffile)

#     def copy4DSettings(self, destination: str) -> None:
#         """
#         Copies the interferometer settings file to the specified destination.
#         """
#         import shutil

#         dest_fold = _os.path.join(_folds.OPD_IMAGES_ROOT_FOLDER, destination)
#         shutil.copy(_folds.SETTINGS_CONF_FILE, dest_fold)
#         shutil.move(
#             _os.path.join(dest_fold, "AppSettings.ini"),
#             _os.path.join(dest_fold, "4DSettings.ini"),
#         )

#     @staticmethod
#     def getCameraSettings(tn: str = None) -> list[int]:
#         """
#         Reads che actual interferometer settings from its configuration file.

#         Return
#         ------
#         output: list
#         list of camera settings: [width_pixel, height_pixel, offset_x, offset_y]
#         """
#         if not tn is None:
#             path = _ft(tn, complete_path=True)
#             try:
#                 file_path = _os.path.join(path, _fn.COPIED_SETTINGS_CONF_FILE)
#                 setting_reader = _fn.ConfSettingReader4D(file_path)
#             except Exception as e:
#                 print(f"Error: {e}")
#                 file_path = _os.path.join(path, "4DSettings.ini")
#                 setting_reader = _fn.ConfSettingReader4D(file_path)
#         else:
#             file_path = _folds.SETTINGS_CONF_FILE
#             setting_reader = _confReader(file_path)
#         width_pixel = setting_reader.getImageWidhtInPixels()
#         height_pixel = setting_reader.getImageHeightInPixels()
#         offset_x = setting_reader.getOffsetX()
#         offset_y = setting_reader.getOffsetY()
#         return [width_pixel, height_pixel, offset_x, offset_y]

#     @staticmethod
#     def getFrameRate(tn: str = None) -> float:
#         """
#         Reads the frame rate the interferometer is working at.

#         Return
#         ------
#         frame_rate: float
#             Frame rate of the interferometer
#         """
#         if not tn is None:
#             path = _ft(tn, complete_path=True)
#             try:
#                 file_path = _os.path.join(path, _fn.COPIED_SETTINGS_CONF_FILE)
#                 setting_reader = _fn.ConfSettingReader4D(file_path)
#             except Exception as e:
#                 print(f"Error: {e}")
#                 file_path = _os.path.join(path, "4DSettings.ini")
#                 setting_reader = _fn.ConfSettingReader4D(file_path)
#         else:
#             file_path = _folds.SETTINGS_CONF_FILE
#             setting_reader = _confReader(file_path)
#         frame_rate = setting_reader.getFrameRate()
#         return frame_rate

#     def intoFullFrame(self, img: _ot.ImageData) -> _ot.ImageData:
#         """
#         The function fits the passed frame (expected cropped) into the
#         full interferometer frame (2048x2048), after reading the cropping
#         parameters.

#         Parameters
#         ----------
#         img: ImageData
#             The image to be fitted into the full frame.

#         Return
#         ------
#         output: ImageData
#             The output image, in the interferometer full frame.
#         """
#         off = (self.getCameraSettings())[2:4]
#         off = _np.flip(off)
#         nfullpix = _np.array([2048, 2048])
#         fullimg = _np.full(nfullpix, _np.nan)  # was   _np.zeros(nfullpix)
#         fullmask = _np.ones(nfullpix)
#         offx = off[0]
#         offy = off[1]
#         sx = _np.shape(img)[0]  # croppar[2]
#         sy = _np.shape(img)[1]  # croppar[3]
#         fullimg[offx : offx + sx, offy : offy + sy] = img.data
#         fullmask[offx : offx + sx, offy : offy + sy] = img.mask
#         fullimg = _np.ma.masked_array(fullimg, fullmask)
#         return fullimg

#     def _rename4D(self, folder: str) -> None:
#         """
#         Renames the produced 'x.4D' files into '0000x.4D'

#         Parameters
#         ----------
#         folder : str
#             The folder where the 4D data is stored.
#         """
#         fold = _os.path.join(_OPDIMG, folder)
#         files = _os.listdir(fold)
#         for file in files:
#             if file.endswith(".4D"):
#                 num_str = file.split(".")[0]
#                 if num_str.isdigit():
#                     num = int(num_str)
#                     new_name = f"{num:05d}.4D"
#                     old_file = _os.path.join(fold, file)
#                     new_file = _os.path.join(fold, new_name)
#                     _os.rename(old_file, new_file)
