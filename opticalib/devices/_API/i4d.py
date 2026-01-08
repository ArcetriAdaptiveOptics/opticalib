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

    def _readJsonData(self, url, data=None):
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
    def getFeatureAnalysisResults(self):
        """
        Returns
        -------
        features:
        fractionalFeatureArea:
        numberOfFeatures:
        totalFeatureAreaInSquareMicrons:
        """
        url = "%s%s" % (self._dataServiceAddress, "GetFeatureAnalysisResults")
        json_data = self._readJsonData(url)
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

    def getFirstNineZernikeTerms(self):
        """
        Returns
        -------
        zernike_waves_terms:
        """
        """ Return the first nine zernike of image"""
        url = "%s%s" % (self._dataServiceAddress, "GetFirstNineZernikeTerms")
        json_data = self._readJsonData(url)
        zernike_waves_terms = np.array(json_data)
        return zernike_waves_terms

    def getFringeAmplitudeData(self):
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
        url = "%s%s" % (self._dataServiceAddress, "GetFringeAmplitudeData")
        json_data = self._readJsonData(url)
        width = json_data["Width"]
        height = json_data["Height"]
        pixel_size_in_microns = json_data["PixelSizeInMicrons"]
        data = np.array(json_data["Data"], dtype=float)
        return data, height, pixel_size_in_microns, width

    def getIntensityData(self):
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
        url = "%s%s" % (self._dataServiceAddress, "GetIntensityData")
        json_data = self._readJsonData(url)
        width = json_data["Width"]
        height = json_data["Height"]
        pixel_size_in_microns = json_data["PixelSizeInMicrons"]
        data = np.array(json_data["Data"], dtype=float)
        return data, height, pixel_size_in_microns, width

    def getInterferogram(self, index):
        """
        Parameters
        ----------
        index:

        Returns
        -------
        json_data:
        """
        url = "%s%s" % (self._dataServiceAddress, "GetInterferogram")
        data = index
        json_data = self._readJsonData(url, data)
        return json_data

    def getMeasurementInfo(self):
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
        url = "%s%s" % (self._dataServiceAddress, "GetMeasurementInfo")
        json_data = self._readJsonData(url)
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

    def dataServiceGetModulationData(self):
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
        json_data = self._readJsonData(url)
        width = json_data["Width"]
        height = json_data["Height"]
        pixel_size_in_microns = json_data["PixelSizeInMicrons"]
        data_list = json_data["Data"]
        data_array = np.array(data_list, dtype=float)
        return width, height, pixel_size_in_microns, data_array

    def getPhaseStepCalculatorResults(self):
        """
        Returns
        -------
        averagePhaseStepInDegrees:
        height:
        phaseStepsInDegrees:
        width:
        """
        url = "%s%s" % (self._dataServiceAddress, "GetPhaseStepCalculatorResults")
        json_data = self._readJsonData(url)
        averagePhaseStepInDegrees = json_data["AveragePhaseStepInDegrees"]
        height = json_data["Height"]
        phaseStepsInDegrees = np.array(json_data["PhaseStepsInDegrees"])
        width = json_data["Width"]
        return averagePhaseStepInDegrees, height, phaseStepsInDegrees, width

    def getSurfaceData(self):
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
        url = "%s%s" % (self._dataServiceAddress, "GetSurfaceData")
        json_data = self._readJsonData(url)
        data = np.array(json_data["Data"], dtype=float)
        height = json_data["Height"]
        pixel_size_in_microns = json_data["PixelSizeInMicrons"]
        width = json_data["Width"]
        return data, height, pixel_size_in_microns, width

    def getUnprocessedSurfaceData(self):
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
        url = "%s%s" % (self._dataServiceAddress, "GetUnprocessedSurfaceData")
        json_data = self._readJsonData(url)
        data = np.array(json_data["Data"], dtype=float)
        height = json_data["Height"]
        pixel_size_in_microns = json_data["PixelSizeInMicrons"]
        width = json_data["Width"]
        return data, height, pixel_size_in_microns, width

    def saveDataToDisk(self, path):
        """
        Parameters
        ----------
        path: string
            path where to save the measurements
        """
        url = "%s%s" % (self._dataServiceAddress, "SaveDataToDisk/")
        data = path
        self._readJsonData(url, data)

    ### SYSTEM PROXY ###
    def takeSingleMeasurement(self):
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
        url = "%s%s" % (self._systemServiceAddress, "TakeSingleMeasurement/")
        json_data = self._readJsonData(url)
        width = json_data["Width"]
        height = json_data["Height"]
        pixel_size_in_microns = json_data["PixelSizeInMicrons"]
        data_list = json_data["Data"]
        data_array = np.array(data_list, dtype=float)
        return width, height, pixel_size_in_microns, data_array

    def setDetectorMask(self, mask):
        """
        Parameters
        ----------
        mask: numpy array
            numpy 2d array with np.nan in the obscured area
        """
        url = "%s%s" % (self._systemServiceAddress, "SetDetectorMask")
        height_str = "%i" % mask.shape[0]
        width_str = "%i" % mask.shape[1]
        data = {
            "Height": height_str,
            "Width": width_str,
            "MaskArray": mask.flatten().tolist(),
        }
        self._readJsonData(url, data)
        print("reload")
        return

    def getSystemInfo(self):
        """
        Returns
        -------
        serialNumber:
        """
        url = "%s%s" % (self._systemServiceAddress, "GetSystemInfo")
        json_data = self._readJsonData(url)
        serialNumber = json_data["SystemSerialNumber"]
        return serialNumber

    def convertRawFramesInDirectoryToMeasurementsInDestinationDirectory(
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
            "ConvertRawFramesInDirectoryToMeasurementsInDestinationDirectory",
        )
        data = {
            "MeasurementsDirectory": measurementsDirectory,
            "RawFramesDirectory": rawFramesDirectory,
        }
        self._readJsonData(url, data)

    def setTriggerMode(self, trigger: int):
        """
        Enables/Disables external triggering mode.

        Parameters
        ----------
        trigger: int
            0 = Disabled
            1 = Enabled
        """
        url = "%s%s" % (self._systemServiceAddress, "SetTriggerMode")
        data = {"CameraIsExternallyTriggered": trigger}
        self._readJsonData(url, data)

    def takeAveragedMeasurement(self, numberOfSamples):
        """
        Parameters
        ----------
        numberOfSamples: int
             numbers of measurements to average
        """
        url = "%s%s" % (self._systemServiceAddress, "TakeAveragedMeasurement")
        data = numberOfSamples
        self._readJsonData(url, data)
    
    def saveConfiguration(self, newConfigurationPath: str):
        """
        Parameters
        ---------
        configurationPath: string
            file path for configuration to save
        """
        url = "%s%s" % (self._systemServiceAddress, "SaveConfiguration")
        self._readJsonData(url, newConfigurationPath)

    def loadConfiguration(self, configurationPath: str):
        """
        Parameters
        ---------
        configurationPath: string
            file path for configuration to load
        """
        url = "%s%s" % (self._systemServiceAddress, "LoadConfiguration")
        self._readJsonData(url, configurationPath)

    ### FRAME BURST PROXY ###
    def burstFramesToSpecificDirectory(self, directory, numberOfFrames):
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
            "BurstFramesToSpecificDirectory",
        )
        data = {"BurstDirectory": directory, "NumberOfFrames": numberOfFrames}
        self._readJsonData(url, data)

    def burstFramesToDisk(self, numberOfFrames):
        """
        Parameters
        ----------
        numberOfFrames: int
            number of frames to acquire

        """
        url = "%s%s" % (self._frameBurstServiceAddress, "BurstFramesToDisk")
        data = numberOfFrames
        self._readJsonData(url, data)
