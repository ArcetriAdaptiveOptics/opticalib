# CONFIGURATION FILE DOCUMENTATION

### Author(s)

- Pietro Ferraiuolo : written in 2025 (pietro.ferraiuolo@inaf.it)

### General description
The `configuration.yaml` is, as the name suggests, a configuration file for the optimal functioning of the `OptiCalib` python package, which includes the following main sections:

- [System](#system-section)
- [Devices](#devices-section)
    - [Wavefront Sensors](#variables-overview-wfs)
    - [Interferometers](#variables-overview-interferometers)
    - [Deformable Mirrors](#variables-overview-deformablemirrors)
    - [Cameras](#variables-overview-cameras)
    - [Motors](#variables-overview-motors)
- [Phasing](#phasing-section)
- [Influence Functions](#iff-section)
- [Stitching](#stitching-section)
- [Alignment](#alignment-section)

<ins>NOTE</ins>: every section is optional. A section which is missing (or set to `null`) does not prevent the package from being imported, but the procedures which need it will raise an error when used.

## System Section
### Description
In this first section of the configuration file, under the `SYSTEM` key, there are the general definitions the software needs to properly operate. Here we find the base data path, where obtained data and configurations are stored and retrieved, and the schema version of the configuration file.

### Variables Overview
- **data_path : str**

    This is the base bath for the construction of the folder tree of the package.
    If left undefined (empty string), the package will automatically asign as base data path the hidden folder in the home directory, and will use the copy of the `configuration.yaml` stored in its `SysConfig` folder (created from the template if not present), so that
    for future modifications there will be non need to touch the "root" configuration file:

    `~/.opticalib/`

    The folder tree (`OPTData`, `Logging`, `SysConfig`) is created at import time.

- **version : str**

    The *configuration schema version*, in `MAJOR.MINOR` format (e.g. `'2.5'`). It is independent of the package version and is bumped by hand (`CONFIG_SCHEMA_VERSION` in `opticalib.core.config`) only when the layout of the configuration file changes (sections or keys added, renamed or removed). At import time (and after `set_configuration_file`) only `MAJOR.MINOR` is compared with the schema version expected by the installed package, and a `UserWarning` is raised if:
    - the key is missing or unparseable (legacy file, may need alignment);
    - the configuration is *older* than expected: check this document, update the file and set `version` to the expected value;
    - the configuration is *newer* than expected: it was written for a newer `opticalib`, upgrade the package.

    Always quote the value (`version: '2.5'`): unquoted, YAML reads `2.10` as the number `2.1`.

- **created_with : str**

    The version of the `opticalib` package which created the file. Provenance only, it is never compared with anything.

## Devices Section
### Description
The devices part of the configuration file, under the `DEVICES` key, is divided into five sub-sections: Wavefront Sensors (`WFS`), Interferometers (`INTERFEROMETERS`), Deformable Mirrors (`DEFORMABLE.MIRRORS`), Cameras (`CAMERAS`) and Motors (`MOTORS`). This section of the configuration serves to define key elements for the connection with the instruments, such as their `ip:port` adresses for ethernet connection.

Each device is identified by its name, which is the key under which its parameters are defined. The lookup of a device by name is case-insensitive (i.e. an entry named `phasecam6110` is found when asking for `PhaseCam6110`), as long as it is not ambiguous.

Devices can refer to other devices with the `TYPE:name` notation, for example `camera: CAMERAS:GigaVision` indicates the `GigaVision` device defined in the `CAMERAS` sub-section.

### Variables Overview: `WFS`
The wavefront sensors (currently the `Ingot` three pupils sensor and the `Pyramid` four pupils sensor), which are based on a camera and a pupil detection algorithm. The parameters are the same for each device:

- **camera : str**

    The camera used by the sensor, in the `CAMERAS:name` notation.

- **camera_base_exptime : float**

    The base exposure time of the camera, in ms.

- **camera_fps : float**

    Frames per second of the camera (if supported).

- **camera_binning : int**

    Software binning factor of the camera.

- **sigma_threshold : float**

    Sigma threshold used by the edge filter (Canny algorithm) for the pupil detection.

- **n_pupils : int**

    The total number of pupils (3 for `Ingot`, 4 for `Pyramid`).

- **subapertures : int**

    The number of pixels across the pupil.

- **delta_r : int**

    The delta, in pixels, for the pupil search range, which is centered on `subapertures`.

- **pupils_centers : list of [int, int]**

    An estimate of the centers of the pupils, to help the search. It has one `[x, y]` element for each pupil.

### Variables Overview: `INTERFEROMETERS`
Each interferometer is defined by its own name (e.g. `PhaseCamX`, `AccuFizX`, `4DProcesser1`), under which are present the following variables:

- **ip : str**

    This is the IP address of the interferometer instrument.

- **port : int**

    This is the network PORT of the interferometer.

- **Paths**

    In this sub-section are all the necessary paths to work with the interferometer.

    - **settings : str**

        This is the path to the `AppSettings.ini` configuration file of the interferometer, which needs to be network-mounted into the user's machine.

    - **copied_settings : str**

        This is the file name (e.g. `CameraSettings.ini`) given to the copy of the interferometer `AppSettings.ini` file, which is saved along with the acquired data.

    - **capture_4dpc : str**

        This is the `capture` folder in the proprietary `4D` computer, which needs to be network-mounted in the user machine.

    - **produce_4dpc : str**

        This is the `produce` folder in the proprietary `4D` computer, which needs to be network-mounted in the user machine.

    - **produce : str**

        This is the destination folder for the `produce`, which is in the user's machine.

    - **capture : str**

        This is the destination folder for the `capture`, which is in the user's machine.


### Variables Overview: `DEFORMABLE.MIRRORS`

- **"device_name"** (Here goes the device name that has been defined, i.e. `AlpaoXXX` or `PetalDM`)

    The variables depend on the kind of device. Those shared by (almost) all of them are:

    - **ip : str**

        The `IP` address of the DM device to be connected.

    - **port : int**

        Network `PORT` of the DM device to be connected

    - **slave_ids : list of int**

        The indices of the actuators to be slaved (see the actuator slaving of the `slave` option when commanding the DM).

    - **border_ids : list of int**

        The indices of the border actuators of the DM, used by the slaving algorithms.

    The `Alpao` DMs (named `Alpao<nacts>`, e.g. `Alpao97`) can be controlled via two different backends, and the variables depend on the one in use:

    - Native SDK backend:

        - **serial_number : str**

            The serial number of the DM, in the form `BAX<YYY>`.

        - **sdk_folder_path : str**

            The parent folder containing `Lib64/` (e.g. `.../Linux/Samples/Python3`).

        - **acfg_path : str**

            The folder containing the `.acfg` hardware configuration file (it sets the `ACECFG` environment variable).

    - `plico_dm` backend:

        - **plico_ip : str**

            The `IP` address of the `plico_dm` server.

        - **plico_port : int**

            The `PORT` of the `plico_dm` server.

    - **diameter : float**

        The (optical) diameter of the DM.

    The `PetalDM` is made of six segments, each one with its own address:

    - **ip0, ..., ip5 : str**

        The `IP` address of each one of the six petals.

<ins>NOTE</ins>: multiple devices can be defined in the configuration file, and actually it is recommended to define ahead all the possible available devices, to have a smoother experience on instancing the DM objects.

### Variables Overview: `CAMERAS`

- **"device_name"** (Here goes the camera name that has been defined, i.e. `GigaVision`)

    - **id : str**

        The identifier of the camera (e.g. `DEV_000F314DE425`).

    - **ip : str**

        The `IP` address of the camera, if applicable.

    - **buffer_size : int**

        The number of frames kept in memory while streaming. If left undefined, it defaults to `100`.

### Variables Overview: `MOTORS`
The motor devices usually use the `plico_motor` interface, as their SDK is not implemented in Linux.

- **"device_name"** (Here goes the motor name that has been defined, i.e. `TunableFilter`)

    - **ip : str**

        The `IP` address of the `plico_motor` server.

    - **port : int**

        The `PORT` of the `plico_motor` server.

## Phasing Section
### Description
In this section of the configuration file, under the `PHASING` key, are defined the parameters for the SPL phasing sensor, which is usually a camera plus a tunable filter (both to be defined in the `DEVICES` section), looking at a frame in which a certain number of PSFs is expected.

### Variables Overview
- **camera : str**

    The camera of the sensor, in the `CAMERAS:name` notation.

- **filter : str**

    The tunable filter of the sensor, in the `MOTORS:name` notation.

- **expected_psfs : int**

    The number of different PSFs to expect in the frame.

- **psfs_angles : list of float**

    The angles of the PSFs, in degrees, with respect to the horizontal axis of the camera.

- **expected_psfs_pos : list of lists**

    The expected coordinates of the PSFs, ordered as the angles. This is used to better identify the correct centroids.

- **initial_crop_half_size : int**

    Half size, in pixels, of the crop used for the first PSF detection, which is then refined with a photometric centroid detection.

- **crop_half_size : int or list of int**

    Half size, in pixels, of the final crop. If a list, it is `[y, x]`.

- **min_px_threshold : int**

    The minimum number of pixels above the threshold to consider a PSF as detected.

- **sigma_threshold : float**

    The number of sigma above the background to use as threshold for the detection.

## IFF Section
### Description
In this section of the configuration file, under the `INFLUENCE.FUNCTIONS` key, are defined all the essential parameters needed to acquire the influence function for a deformable mirror. For each acquisition, a copy of this section is saved along with the data (`iffConfig.yaml`), for provenance.

### Variables Overview
There are two groups of parameters to define for the IFF acquisition: the general ones, which are related to the synchronization between the deformable mirror and the sensor, and three parameter classes defining the command to apply.

#### General parameters
In here we find information related to the hardware behaviour of the deformable mirror and its synchronization with the acquisition:

- **timing : int**

    The timing information is needed for the synchronization between the dm and the interferometer acquisition, and is effectively done by repeating each column of the `timed command history` by the amount specified in this parameter. This is a tentative of replacement for the interferometer trigger line, which is the best way to perform synchronization between interferometer and Deformable Mirror's command actuation.

- **sequential_delay : float**

    This is the delay, in seconds, artificially introduced between an actuated command and it's subsequent sensor acquisition, useful if the device at hand has a setting time for the commanded shape.

- **triggered_mode**

    The parameters for the triggered acquisition, in which the DM runs the command history by itself, synchronized with the sensor. If it is not available, set it to `false` and comment the next two lines.

    - **frequency : float**

        The frequency, in Hz, at which the commands are applied.

    - **cmd_delay : float**

        The delay, in seconds, between the start of the command history and the start of the acquisition.

#### `TRIGGER - REGISTRATION - IFFUNC`
The following three parameter classes share almost the same structure, and define how is the `Timed Command Matrix History` (TCMH) is built, that is the command matrix that will be uploaded and run by the deformable mirror.

- `TRIGGER`

    The trigger, the first part of the `TCMH`, consists of a mode "shot" with high amplitude to identify the start of the measures on the resulting data cube.

- `REGISTRATION`

    The second part of the `TCMH` is the registration, where 3 (or more) actuators are individually actuated following a template, to allow the re-alignment of the images within the measured data, in the case of, for example, the system's position has drifted during the measurements.

- `IFFUNC`

    The final, bigger, part which makes up the `TCMH` is the Influence function itself, which is a series of modes repeated on a template, later analized to obtain the calibration.

As for the variables iside each parameter class, they are:

- **trailing_zeros : int**

    This is the number of zeros at the start of each section. Serves as a sort of "spacing" between sections of the `TCMH`.

- **modes_list : list or str**

    This variables indicates the modes to command. It can be an hand-written list or, more conveniently in the case of modes equal to the number of actuators of the DM, a string can be used to form an array in python, like

    ```yaml
    modes_list : np.arange(0,88,1)
    ```

    In this way, `modes_list` is the list of integers from 0 to 87, which tells the system to run all the modes from 1 to 88 on the selected modal base. Only the `np.arange(start, stop[, step])` expression is accepted.

- **amplitude : float or list**

    This is the amplitude each command receives.<br> 
    If passed as a single float number, it is then transformed into an array with lenght equal to the number of measured modes, while if passed as list, it must have same size as the `modes_list` array, as each value corresponds to the amplitude of the command at same inxed position.

- **template : list of int**

    A list of integers defining how each mode is executed. Usually is `[1,-1,1]`, which means that each mode is executed in a `push-pull-push` way.

- **modal_base : str**

    This string defines the modal base to use for the calibration. There are several options:

    - **zonal**

        This is the zonal modal base, which is represented by the identity matrix of dimentions $[N_{acts}\times N_{acts}]$ and means that each mode corresponds to a single actuator.

    - **hadamard**

        The [hadamard matrix](https://en.wikipedia.org/wiki/Hadamard_matrix), created as $H_{2^{10}}$ and then cut down on the number of actuators.

    - **mirror**

        This option uses the mirror's modes defined in the mirror class itself, which must be present as `*devicename*Mirror.fits` in the `ModalBases` folder.

    - **"custom"**

        A custom modal base is admitted, as long as the string put in the `modal_base` variable corresponds to the name of the actual modal base `.fits` file in the `ModalBases` folder.

Only the `IFFUNC` class has the following additional variables:

- **padding_zeros : int**

    The number of zeros after the modes.

- **shuffle : bool**

    Whether to shuffle the order of the commanded modes.

- **n_repetitions : int**

    The number of times the application of each mode is repeated.

## Stitching Section
### Description
In this section of the configuration file, under the `STITCHING` key, are defined the parameters for the stitching procedure, in which a motorized stage moves the sensor (or the optic) to acquire different sub-apertures of a larger surface.

### Variables Overview
- **motor : str**

    The motor device which moves the stage, in the `MOTORS:name` notation.

- **pixel_scale : float**

    The pixel scale of the image, in mm/px.

- **alpha : float**

    The rotation angle, in degrees, between the motor axes and the image.

- **starting_coords : list of float**

    The motors position, in mm, at which the stitching procedure starts.

- **home_coords : list of float**

    The motors position, in mm, of the "home" position.

## Alignment Section
### Description
In this section of the configuration file, under the `SYSTEM.ALIGNMENT` key, are defined all the parameters and function calls of the various devices for the optical alignment of the system.

### <ins>Important Notes</ins>
The element order of the lists is <ins>**essential**</ins>, as the code will use the same index
for all the lists to access the right information. The order of the devices in the
lists must be the same as the order in the command matrix supplied, eg. if `device_y` 
is the second in place going through the command vector, then it's position must be
the second for every list in this configuration file.

### Variables Overview
- **names : list of str**
    
    This list contains the names of the devices, in the same order as all the 
    other lists. This is mainly used for logging purposes and print fancyness.<br>
    <ins>Use the same order of the devices used here for every subsequent variable in the configuration.</ins>

- **devices_move_calls : list of str**
    
    This list must contain the callables for moving each device, in the same 
    order as all the other lists.<br>

- **devices_read_calls : list of str**
    
    This list must contain the callables used to read the devices positions, in 
    the same order as all the other lists.

- **ccd_acquisition : list of str**
    
    This list contains the callables for the acquisition device (Interferometer, camera,...). The first item must be the acquisition method.

- **devices_dof : list of int | int**

    This variable represents the total degrees of freedom, at the software level, each device has. This parameter, then, will be the length of the accepted command vector by the devices. It can be an `int` (if all the devices use have the same number of accepted dof) or a list of integers, where each int corresponds to the dof of a device.

- **dof : list of lists**

    This variable, initialized as an empty list, will contain the degrees of
    freedom each device <ins>can actually move</ins>.<br>

    <ins>Example</ins>:<br>
    If a device can move only 3 DoF, say `piston`, `tip` and `tilt`, then, if the
    `devices_dof` for the device if 6 (i.e the device accept a vector of 6 elements), the 'dof' list should be the index at which these degrees of freedom are located in the
    accepted command vector. In this case, the list could be [2, 3, 4] (as 
    in the case for the M4's OTT) and the total vector would then be `[0,0,p,t,t,0]`.

- **slices : list**

    This variable is a list of slices that will be used to extract the right dof
    positions from the command matrix full vector.<br>
    Each slice is defined with the `start` and `stop` points:

    - **start : int**

        Starting point of the slice

    - **stop : int**

        Ending point of the slice (python notation, so the actual index used is `stop - 1`)

    <ins>Example</ins>:<br>
    If the command matrix is a column vector (the full command) of 7 elements,
    in which the first 3 elements are for the `device_1` dof, the second 2 
    for `device_2` dof and the last 3 for `device_3` dof, then the slices parameters should be:

    ```yaml
      slices:
        # device_1
        - start : 0
          stop  : 3
        # device_2
        - start : 3
          stop  : 5
        # device_3
        - start : 5
          stop  : 7
    ```

- **zernike_to_use : list of int**

    This variable contains the zernike modes indices which will be used for the 
    Interaction Matrix Construction, extracted from the general zernike fit. It is a list of integers, where each value corresponds to a mode order.

- **push_pull_template : list of int**

    Template for the push-pull algorithm. Being the command differential, the 
    push and pulls are to be written with respect to zero.<br>

    <ins>Example</ins>: A classic `[1,-1,1]` template, translates into `[1,-2,1]`.

- **commandMatrix : str**
    
    Control matrix file name, which must be a `.fits` file located under the `base_data_path/OPTData/Alignment/ControlMatrices` folder.

- **fitting_surface : str**
    
    Path to the calibration fits file, whose mask will be used for zernike fitting
    purposes. If no mask is to be provided, leave it as an empty string.

### Builtin Example: M4's OTT
The template (root) configuration file has the alignment section filled with the configuration for the ELT@M4 Optical Test Tower (OTT). Here we we have three devices: the Parabola, the Reference Mirror and the M4 Exapode. Each device accept a vector of 6 elements, which means they "speak" in 6 DoF, however, for 'reality' constraints, they can only move certain dof. In particular: the Parabola can move in piston, tip and tilt, so the DoF are 2, 3 and 4; the Reference Mirror can move in tip and tilt, so the DoF are 3 and 4; the M4 Exapode can move in tip and tilt, so the DoF are 3 and 4. The command matrix (7x7) has a 7 element columns, which is the alignment command, and these devices appear in the same order as are listed above, so the ordering inside every list will be Parabola-ReferenceMirror-M4Exapode.