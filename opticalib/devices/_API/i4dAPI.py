import json
import numpy as np
import urllib


class I4D:
    """
    Base low-level class for the Interferometer devices
    """

    def __init__(self, IP: str, PORT: int):
        """The constructor"""
        self._ip = IP
        self._port = PORT
        self._dataServiceAddress = "http://%s:%i/DataService/" % (self._ip, self._port)
        self._systemServiceAddress = "http://%s:%i/SystemService/" % (
            self._ip,
            self._port,
        )
        self._frameBurstServiceAddress = "http://%s:%i/FrameBurstService/" % (
            self._ip,
            self._port,
        )

    def _read_json_data(self, url, data=None):
        """
        Parameters
        ----------
        url: string

        Other Parameters
        ---------------
        data:

        Returns
        -------
        json_data:

        """
        if data:
            dumped_data = json.dumps(data)
            encoded_data = dumped_data.encode("utf-8")
            content_length = len(encoded_data)
            request_headers = {
                "Content-type": "application/json",
                "Accept": "application/json",
                "Content-length": content_length,
            }
            req = urllib.request.Request(
                url, data=encoded_data, headers=request_headers
            )
        else:
            req = url
        try:
            response = urllib.request.urlopen(req)

            response_contents = response.read()
            if response_contents != b"":
                json_data = json.loads(response_contents)
                return json_data
        except urllib.request.HTTPError as e:
            error_message = e.read()
            print(error_message)
            open("/tmp/out.html", "w+").write(error_message.decode("utf-8"))
            raise Exception("Response error see /tmp/out.html for details")

    ### DATA PROXY ###
    def get_feature_analysis_results(self):
        """
        Returns
        -------
        features:
        fractionalFeatureArea:
        numberOfFeatures:
        totalFeatureAreaInSquareMicrons:
        """
        url = "%s%s" % (self._dataServiceAddress, "get_feature_analysis_results")
        json_data = self._read_json_data(url)
        features = np.array(json_data["Features"])
        fractionalFeatureArea = json_data["FractionalFeatureArea"]
        numberOfFeatures = json_data["NumberOfFeatures"]
        totalFeatureAreaInSquareMicrons = json_data["TotalFeatureAreaInSquareMicrons"]
        return (
            features,
            fractionalFeatureArea,
            numberOfFeatures,
            totalFeatureAreaInSquareMicrons,
        )

    def get_first_nine_zernike_terms(self):
        """
        Returns
        -------
        zernike_waves_terms:
        """
        """ Return the first nine zernike of image"""
        url = "%s%s" % (self._dataServiceAddress, "get_first_nine_zernike_terms")
        json_data = self._read_json_data(url)
        zernike_waves_terms = np.array(json_data)
        return zernike_waves_terms

    def get_fringe_amplitude_data(self):
        """
        Returns
        -------
        data: numpy array
            vector containing image data
        height: int
            height of the image in pixel
        pixel_size_in_microns: int
        width: int
            width of the image in pixel
        """
        url = "%s%s" % (self._dataServiceAddress, "get_fringe_amplitude_data")
        json_data = self._read_json_data(url)
        width = json_data["Width"]
        height = json_data["Height"]
        pixel_size_in_microns = json_data["PixelSizeInMicrons"]
        data = np.array(json_data["Data"], dtype=float)
        return data, height, pixel_size_in_microns, width

    def get_intensity_data(self):
        """
        Returns
        -------
        data: numpy array
            vector containing image data
        height: int
            height of the image in pixel
        pixel_size_in_microns: int
        width: int
            width of the image in pixel
        """
        url = "%s%s" % (self._dataServiceAddress, "get_intensity_data")
        json_data = self._read_json_data(url)
        width = json_data["Width"]
        height = json_data["Height"]
        pixel_size_in_microns = json_data["PixelSizeInMicrons"]
        data = np.array(json_data["Data"], dtype=float)
        return data, height, pixel_size_in_microns, width

    def get_interferogram(self, index):
        """
        Parameters
        ----------
        index:

        Returns
        -------
        json_data:
        """
        url = "%s%s" % (self._dataServiceAddress, "get_interferogram")
        data = index
        json_data = self._read_json_data(url, data)
        return json_data

    def get_measurement_info(self):
        """
        Returns
        -------
        averageFringeAmplitude:
        averageIntensity:
        averageModulation:
        fringeAmpThresholdPercentage:
        intensityThresholdPercentage:
        modulationThresholdPercentage:
        numberOfSamples:
        numberOfValidPixels:
        pathMatchPositionInMM:
        RMSInNM:
        userSettingsFilePath:
        wavelengthInNM:
        wedge:
        """
        url = "%s%s" % (self._dataServiceAddress, "get_measurement_info")
        json_data = self._read_json_data(url)
        averageFringeAmplitude = json_data["AverageFringeAmplitude"]
        averageIntensity = json_data["AverageIntensity"]
        averageModulation = json_data["AverageModulation"]
        fringeAmpThresholdPercentage = json_data["FringeAmpThresholdPercentage"]
        intensityThresholdPercentage = json_data["IntensityThresholdPercentage"]
        modulationThresholdPercentage = json_data["ModulationThresholdPercentage"]
        numberOfSamples = json_data["NumberOfSamples"]
        numberOfValidPixels = json_data["NumberOfValidPixels"]
        pathMatchPositionInMM = json_data["PathMatchPositionInMM"]
        RMSInNM = json_data["RMSInNM"]
        userSettingsFilePath = json_data["UserSettingsFilePath"]
        wavelengthInNM = json_data["WavelengthInNM"]
        wedge = json_data["Wedge"]
        return (
            averageFringeAmplitude,
            averageIntensity,
            averageModulation,
            fringeAmpThresholdPercentage,
            intensityThresholdPercentage,
            modulationThresholdPercentage,
            numberOfSamples,
            numberOfValidPixels,
            pathMatchPositionInMM,
            RMSInNM,
            userSettingsFilePath,
            wavelengthInNM,
            wedge,
        )

    def data_service_get_modulation_data(self):
        """
        Returns
        -------
        width: int
            width of the image in pixel
        height: int
            height of the image in pixel
        pixel_size_in_microns: int
        data_array: numpy array
            vector containing image data
        """
        url = "%s%s" % (self._dataServiceAddress, "GetModulationData/")
        json_data = self._read_json_data(url)
        width = json_data["Width"]
        height = json_data["Height"]
        pixel_size_in_microns = json_data["PixelSizeInMicrons"]
        data_list = json_data["Data"]
        data_array = np.array(data_list, dtype=float)
        return width, height, pixel_size_in_microns, data_array

    def get_phase_step_calculator_results(self):
        """
        Returns
        -------
        averagePhaseStepInDegrees:
        height:
        phaseStepsInDegrees:
        width:
        """
        url = "%s%s" % (self._dataServiceAddress, "get_phase_step_calculator_results")
        json_data = self._read_json_data(url)
        averagePhaseStepInDegrees = json_data["AveragePhaseStepInDegrees"]
        height = json_data["Height"]
        phaseStepsInDegrees = np.array(json_data["PhaseStepsInDegrees"])
        width = json_data["Width"]
        return averagePhaseStepInDegrees, height, phaseStepsInDegrees, width

    def get_surface_data(self):
        """
        Returns
        -------
        data: numpy array
            vector containing image data
        height: int
            height of the image in pixel
        pixel_size_in_microns: int
        width: int
            width of the image in pixel
        """
        url = "%s%s" % (self._dataServiceAddress, "get_surface_data")
        json_data = self._read_json_data(url)
        data = np.array(json_data["Data"], dtype=float)
        height = json_data["Height"]
        pixel_size_in_microns = json_data["PixelSizeInMicrons"]
        width = json_data["Width"]
        return data, height, pixel_size_in_microns, width

    def get_unprocessed_surface_data(self):
        """
        Returns
        -------
        data: numpy array
            vector containing image data
        height: int
            height of the image in pixel
        pixel_size_in_microns: int
        width: int
            width of the image in pixel
        """
        url = "%s%s" % (self._dataServiceAddress, "get_unprocessed_surface_data")
        json_data = self._read_json_data(url)
        data = np.array(json_data["Data"], dtype=float)
        height = json_data["Height"]
        pixel_size_in_microns = json_data["PixelSizeInMicrons"]
        width = json_data["Width"]
        return data, height, pixel_size_in_microns, width

    def save_data_to_disk(self, path):
        """
        Parameters
        ----------
        path: string
            path where to save the measurements
        """
        url = "%s%s" % (self._dataServiceAddress, "save_data_to_disk/")
        data = path
        self._read_json_data(url, data)

    ### SYSTEM PROXY ###
    def take_single_measurement(self):
        """
        Returns
        -------
        width: int
            width of the image in pixel
        height: int
            height of the image in pixel
        pixel_size_in_microns: int
        data_array: numpy array
            vector containing image data
        """
        url = "%s%s" % (self._systemServiceAddress, "take_single_measurement/")
        json_data = self._read_json_data(url)
        width = json_data["Width"]
        height = json_data["Height"]
        pixel_size_in_microns = json_data["PixelSizeInMicrons"]
        data_list = json_data["Data"]
        data_array = np.array(data_list, dtype=float)
        return width, height, pixel_size_in_microns, data_array

    def set_detector_mask(self, mask):
        """
        Parameters
        ----------
        mask: numpy array
            numpy 2d array with np.nan in the obscured area
        """
        url = "%s%s" % (self._systemServiceAddress, "set_detector_mask")
        height_str = "%i" % mask.shape[0]
        width_str = "%i" % mask.shape[1]
        data = {
            "Height": height_str,
            "Width": width_str,
            "MaskArray": mask.flatten().tolist(),
        }
        self._read_json_data(url, data)
        print("reload")
        return

    def get_system_info(self):
        """
        Returns
        -------
        serialNumber:
        """
        url = "%s%s" % (self._systemServiceAddress, "get_system_info")
        json_data = self._read_json_data(url)
        serialNumber = json_data["SystemSerialNumber"]
        return serialNumber

    def convert_raw_frames_in_directory_to_measurements_in_destination_directory(
        self, measurementsDirectory, rawFramesDirectory
    ):
        """
        Parameters
        ----------
        measurementsDirectory: string
            path where to save the measurements converted
        rawFramesDirectory: string
            path where raw frames are located
        """
        url = "%s%s" % (
            self._systemServiceAddress,
            "convert_raw_frames_in_directory_to_measurements_in_destination_directory",
        )
        data = {
            "MeasurementsDirectory": measurementsDirectory,
            "RawFramesDirectory": rawFramesDirectory,
        }
        self._read_json_data(url, data)

    def set_trigger_mode(self, trigger: int):
        """
        Enables/Disables external triggering mode.

        Parameters
        ----------
        trigger: int
            0 = Disabled
            1 = Enabled
        """
        url = "%s%s" % (self._systemServiceAddress, "set_trigger_mode")
        data = {"CameraIsExternallyTriggered": trigger}
        self._read_json_data(url, data)

    def take_averaged_measurement(self, numberOfSamples):
        """
        Parameters
        ----------
        numberOfSamples: int
             numbers of measurements to average
        """
        url = "%s%s" % (self._systemServiceAddress, "take_averaged_measurement")
        data = numberOfSamples
        self._read_json_data(url, data)

    def save_configuration(self, newConfigurationPath: str):
        """
        Parameters
        ---------
        configurationPath: string
            file path for configuration to save
        """
        url = "%s%s" % (self._systemServiceAddress, "save_configuration")
        self._read_json_data(url, newConfigurationPath)

    def load_configuration(self, configurationPath: str):
        """
        Parameters
        ---------
        configurationPath: string
            file path for configuration to load
        """
        url = "%s%s" % (self._systemServiceAddress, "load_configuration")
        self._read_json_data(url, configurationPath)

    ### FRAME BURST PROXY ###
    def burst_frames_to_specific_directory(self, directory, numberOfFrames):
        """
        Parameters
        ----------
        directory: string
            directory where to save files
        numberOfFrames: int
            number of frames to acquire
        """
        url = "%s%s" % (
            self._frameBurstServiceAddress,
            "burst_frames_to_specific_directory",
        )
        data = {"BurstDirectory": directory, "NumberOfFrames": numberOfFrames}
        self._read_json_data(url, data)

    def burst_frames_to_disk(self, numberOfFrames):
        """
        Parameters
        ----------
        numberOfFrames: int
            number of frames to acquire

        """
        url = "%s%s" % (self._frameBurstServiceAddress, "burst_frames_to_disk")
        data = numberOfFrames
        self._read_json_data(url, data)
