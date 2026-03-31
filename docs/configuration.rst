Configuration
=============

``opticalib`` uses a single YAML file, ``configuration.yaml``, to describe your
experimental setup.  The file is generated automatically when you run
``calpy -f <path> --create`` and lives under ``<path>/SysConfig/``.

The file is divided into the following top-level sections:

.. contents::
   :depth: 2
   :local:

----

SYSTEM
------

The ``SYSTEM`` section contains global settings that control where data are
stored and whether hardware devices are real or simulated.

.. code-block:: yaml

   SYSTEM:
     data_path: ''
     simulated.devices:
       dm: true
       interferometer: true

``data_path`` : *str*
    Base path used to build the package's data folder tree (see
    :func:`~opticalib.core.root.create_folder_tree`).  A copy of
    ``configuration.yaml`` is written there automatically so future edits do
    not need to touch the root file.  Leave empty to fall back to
    ``~/opticalib_data/``.

``simulated.devices.dm`` : *bool*
    Set to ``true`` to use the simulated deformable mirror from
    :mod:`opticalib.simulator` instead of real hardware.

``simulated.devices.interferometer`` : *bool*
    Set to ``true`` to use the simulated interferometer from
    :mod:`opticalib.simulator` instead of real hardware.

----

DEVICES
-------

The ``DEVICES`` section registers the hardware instruments attached to the
system.  It has four sub-sections: ``INTERFEROMETER``, ``DEFORMABLE.MIRRORS``,
``CAMERAS``, and ``MOTORS``.  Multiple devices of the same type can be defined
simultaneously; the device name is an arbitrary key chosen by the user and is
the identifier passed to the device constructors at runtime.

Interferometers (``INTERFEROMETER``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supported models: **PhaseCam** series, **AccuFiz**, **4D Processer**.  All
share the same key structure.

.. code-block:: yaml

   DEVICES:
     INTERFEROMETER:
       PhaseCam6110:             # arbitrary name – used as device identifier
         ip:   192.168.1.10
         port: 8011
         Paths:
           settings:        /mnt/4dpc/AppSettings.ini
           copied_settings: /home/user/settings/AppSettings.ini
           capture_4dpc:    /mnt/4dpc/Capture
           produce_4dpc:    /mnt/4dpc/Produce
           produce:         /home/user/produce
           capture:         /home/user/capture

``ip`` : *str*
    IP address of the 4D interferometer computer.

``port`` : *int*
    Network port of the 4D server.

``Paths.settings`` : *str*
    Path to the ``AppSettings.ini`` file on the 4D PC.  Must be accessible
    from the user's machine (e.g. via a network mount).

``Paths.copied_settings`` : *str*
    Local destination for a copy of ``AppSettings.ini``, kept for
    record-keeping alongside acquired data.

``Paths.capture_4dpc`` : *str*
    The *capture* directory on the 4D PC (network-mounted).  Raw
    interferometer frames are written there by the 4D software.

``Paths.produce_4dpc`` : *str*
    The *produce* directory on the 4D PC (network-mounted).  Processed
    phase maps are deposited there by the 4D software.

``Paths.produce`` : *str*
    Local directory where produced phase maps are copied for analysis.

``Paths.capture`` : *str*
    Local directory used for raw captures.

Deformable Mirrors (``DEFORMABLE.MIRRORS``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The required keys depend on the mirror type.  All supported variants are shown
below.

**Alpao mirrors** — identified by serial number:

.. code-block:: yaml

   DEVICES:
     DEFORMABLE.MIRRORS:
       AlpaoDm97:               # arbitrary name
         serialNumber: BAX240
         diameter: 10.5        # aperture in mm

``serialNumber`` : *str*
    Hardware serial number printed on the Alpao mirror unit.  Used to open
    the SDK connection.

``diameter`` : *float*
    Aperture diameter in millimetres.

**Petal / AdOptica mirrors** — six independent controllers, one per petal:

.. code-block:: yaml

   DEVICES:
     DEFORMABLE.MIRRORS:
       PetalDM:                 # arbitrary name
         ip0: 192.168.10.1
         ip1: 192.168.10.2
         ip2: 192.168.10.3
         ip3: 192.168.10.4
         ip4: 192.168.10.5
         ip5: 192.168.10.6

``ip0`` … ``ip5`` : *str*
    IP addresses of the six petal DM controllers.  The index maps to the
    petal segment number (0-based).

.. note::
   It is recommended to pre-define all available devices in the file even if
   they are not all used simultaneously.  The device is instantiated only when
   its name is passed to the corresponding constructor.

Cameras (``CAMERAS``)
~~~~~~~~~~~~~~~~~~~~~~

Currently supported: AVT cameras via the VimbaX SDK.

.. code-block:: yaml

   DEVICES:
     CAMERAS:
       AVT_MANTA_5G:            # arbitrary name
         id: DEV_000F314DE425
         ip: 192.168.1.20

``id`` : *str*
    Unique Vimba device identifier (``DEV_XXXXXXXXXXXX`` format).

``ip`` : *str*
    IP address for GigE cameras.  Leave empty for USB cameras.

Motors (``MOTORS``)
~~~~~~~~~~~~~~~~~~~~

Motor devices are typically controlled via the ``plico_motor`` interface.
A common use case is a tunable optical filter used in the
:class:`~opticalib.phasing.SPL` phasing sensor.

.. code-block:: yaml

   DEVICES:
     MOTORS:
       TunableFilter:           # arbitrary name
         ip:   192.168.1.30
         port: 7100

``ip`` : *str*
    IP address of the motor controller.

``port`` : *int*
    Network port of the motor controller server.

----

PHASING
-------

The ``PHASING`` section provides default parameters for the
:class:`~opticalib.phasing.SPL` (Sensor for Phase Lag) system, which
detects and measures the co-phasing of segmented mirrors using PSF images
acquired by a camera through a tunable filter.

.. code-block:: yaml

   PHASING:
     camera: CAMERAS:AVT_MANTA_5G
     filter: MOTORS:TunableFilter
     expected_psfs: 6
     psfs_angles: [0, -60, 60, -120, 120, 180]
     expected_psfs_pos:
     - []
     - []
     - []
     - []
     - []
     - []
     initial_crop_half_size: 150
     crop_half_size: [75, 23]
     min_px_threshold: 30
     sigma_threshold: 2.0

``camera`` : *str*
    Reference to the camera device defined in ``DEVICES``, in the form
    ``CAMERAS:<device_name>``  (e.g. ``CAMERAS:AVT_MANTA_5G``).

``filter`` : *str*
    Reference to the tunable filter defined in ``DEVICES``, in the form
    ``MOTORS:<device_name>``  (e.g. ``MOTORS:TunableFilter``).

``expected_psfs`` : *int*
    Number of PSF spots expected in the camera frame (one per segment gap).
    For a six-petal mirror this is typically ``6``.

``psfs_angles`` : *list of float*
    Angular positions of the PSF spots, in degrees, measured from the
    horizontal axis of the camera frame.  Must be ordered consistently with
    ``expected_psfs_pos``.

``expected_psfs_pos`` : *list of [y, x] pairs*
    Expected pixel coordinates ``[row, col]`` of each PSF centroid.  Used
    to seed the centroid search and improve robustness.  Leave inner lists
    empty to let the algorithm search the full frame.

``initial_crop_half_size`` : *int*
    Half-size in pixels of the initial search window applied to the full
    frame for the first coarse PSF detection.  Defaults to ``150``.

``crop_half_size`` : *int or [y, x]*
    Half-size in pixels of the final crop window around each detected PSF.
    If a list ``[y, x]`` is provided the window is rectangular.

``min_px_threshold`` : *int*
    Minimum number of pixels above the detection threshold for a source to
    be accepted as a valid PSF.  Defaults to ``30``.

``sigma_threshold`` : *float*
    Detection threshold expressed as the number of standard deviations above
    the background level.  Defaults to ``2.0``.

----

INFLUENCE.FUNCTIONS
-------------------

The ``INFLUENCE.FUNCTIONS`` section (key ``INFLUENCE.FUNCTIONS``) defines all
parameters controlling the Timed Command Matrix History (TCMH) used for
Influence Function (IFF) acquisition.  It has four sub-sections: ``DM``,
``TRIGGER``, ``REGISTRATION``, and ``IFFUNC``.

DM parameters (``DM``)
~~~~~~~~~~~~~~~~~~~~~~~

Hardware-level DM parameters that govern timing and actuator grouping.

.. code-block:: yaml

   INFLUENCE.FUNCTIONS:
     DM:
       nacts: 88
       slaveIds: []
       borderIds: []
       timing: 1
       delay: 0.0
       triggeredMode:         # comment out or set to false to disable
         frequency: 5.0
         cmdDelay: 0.0

``nacts`` : *int*
    Number of actuators of the deformable mirror used for the acquisition.

``slaveIds`` : *list of int*
    Indices of the slaved actuators, i.e. actuators that are not driven
    independently but follow a master actuator via the slaving algorithm
    (see :mod:`~opticalib.dmutils.slaving`).  Leave empty (``[]``) if
    the mirror has no slaved actuators.

``borderIds`` : *list of int*
    Indices of the border actuators — master actuators surrounding the
    slaved region.  Used only by the minimum-RMS slaving method.  Leave
    empty (``[]``) if not needed.

``timing`` : *int*
    Each column of the TCMH is repeated this many times to synchronise the
    DM actuation with the interferometer acquisition.  Effectively replaces
    a hardware trigger line: set to the number of interferometer frames
    recorded per DM step.

``delay`` : *float*
    Artificial delay in seconds introduced between applying a commanded
    shape and the subsequent interferometer acquisition.  Useful for mirrors
    with a non-negligible settling time.  Defaults to ``0.0``.

``triggeredMode`` : *mapping or false*
    If set, enables hardware-triggered synchronisation (Microgate DMs only).
    Set to ``false`` or comment out the block for all other mirror types.

    ``frequency`` : *float*
        Trigger frequency in Hz.

    ``cmdDelay`` : *float*
        Delay in seconds between issuing the DM command and the trigger
        pulse.

Acquisition blocks (``TRIGGER``, ``REGISTRATION``, ``IFFUNC``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The TCMH is composed of three sequential blocks, each described by the same
set of keys:

``TRIGGER``
    A short high-amplitude burst used to identify the start of the
    measurement sequence in the resulting data cube.

``REGISTRATION``
    A set of individually actuated reference actuators (typically 3 or more)
    used to re-align images within the data cube in case of system drift
    during the acquisition.

``IFFUNC``
    The main IFF block: a full sweep of all modes on the selected modal
    base.

.. code-block:: yaml

   INFLUENCE.FUNCTIONS:
     TRIGGER:
       numberofzeros: 0
       modeid: []
       modeamp: 0.05
       template: []
       modalbase: hadamard
     REGISTRATION:
       numberofzeros: 0
       modeid: []
       modeamp: 0.1
       template: []
       modalbase: zonal
     IFFUNC:
       numberofzeros: 0
       paddingZeros: 5
       modeid: np.arange(0,88,1)
       modeamp: 0.05
       template: [1, -1, 1]
       modalbase: zonal

The shared keys are:

``numberofzeros`` : *int*
    Number of zero commands prepended to the block.  Acts as a spacing
    separator between consecutive sections of the TCMH.

``modeid`` : *list of int, or np.arange() string*
    Indices of the modes to command in this block.  Can be an explicit
    Python list (``[0, 1, 2, …]``) or a compact ``numpy`` range string:

    .. code-block:: yaml

       modeid: np.arange(0, 88, 1)   # equivalent to list(range(88))

``modeamp`` : *float or list of float*
    Command amplitude applied to each mode.  If a single float, the same
    amplitude is used for all modes.  If a list, its length must match that
    of ``modeid``.

``template`` : *list of int*
    Push-pull pattern for each mode.  The standard ``[1, -1, 1]`` template
    applies a *push–pull–push* cycle.  Each step is multiplied by
    ``modeamp`` before being sent to the mirror.

``modalbase`` : *str*
    Name of the modal basis used to build the command matrix:

    * ``zonal`` — identity matrix; each mode corresponds to a single
      actuator.
    * ``hadamard`` — Hadamard matrix :math:`H_{2^{10}}`, trimmed to
      ``nacts`` rows.
    * ``mirror`` — mirror's own native modes, loaded from
      ``<data_path>/ModalBases/<device_name>Mirror.fits``.
    * *custom name* — any other string is interpreted as a FITS filename
      in ``<data_path>/ModalBases/``.

``paddingZeros`` : *int* (``IFFUNC`` only)
    Number of zero commands appended after the last commanded mode in the
    IFFUNC block.  Pads the data cube to ensure the last push-pull cycle is
    fully captured.  Defaults to ``0``.

----

STITCHING
---------

The ``STITCHING`` section provides geometric parameters for the mirror pupil
stitching procedure, which mosaics multiple interferometer sub-aperture
measurements into a single full-pupil wavefront map.

.. code-block:: yaml

   STITCHING:
     pixel_scale:      0.09095   # mm per pixel
     alpha:            0         # rotation angle in degrees
     starting_coords: [0, 0]     # motor start position [x, y] in mm
     home_coords:     [0, 0]     # motor home position  [x, y] in mm

``pixel_scale`` : *float*
    Physical size of one camera pixel in millimetres per pixel.  Used to
    convert pixel displacements to physical translations on the mirror
    surface.

``alpha`` : *float*
    Rotation angle in degrees between the camera frame and the motor
    translation axes.  Set to ``0`` if the axes are already aligned.

``starting_coords`` : *[x, y]*
    Initial motor position (in mm) before the stitching scan begins.

``home_coords`` : *[x, y]*
    Home (reference) motor position (in mm) to which the motors are
    returned after the stitching scan.

----

SYSTEM.ALIGNMENT
----------------

The ``SYSTEM.ALIGNMENT`` section (key ``SYSTEM.ALIGNMENT``) configures the
optical alignment procedure managed by :class:`~opticalib.alignment.Alignment`.
It describes the set of optomechanical devices (e.g. mirrors, stages) and the
callable strings used to move and read them.

.. important::
   The element order of every list is **essential**.  All lists must follow
   the same device ordering.  The device whose command occupies positions
   ``slices[i]`` of the alignment vector must appear at index ``i`` in
   ``names``, ``devices_move_calls``, ``devices_read_calls``, ``dof``, and
   ``slices``.

.. code-block:: yaml

   SYSTEM.ALIGNMENT:
     names:
     - Parabola
     - Reference Mirror
     - M4 Exapode

     devices_move_calls:
     - parabola.setPosition
     - referenceMirror.setPosition
     - m4Exapode.setPosition

     devices_read_calls:
     - parabola.getPosition
     - referenceMirror.getPosition
     - m4Exapode.getPosition

     ccd_acquisition:
     - acquire_map

     devices_dof: 6

     dof:
     - [2, 3, 4]     # Parabola: piston, tip, tilt
     - [3, 4]        # Reference Mirror: tip, tilt
     - [3, 4]        # M4 Exapode: tip, tilt

     slices:
     - start: 0      # Parabola occupies elements 0..2
       stop:  3
     - start: 3      # Reference Mirror occupies elements 3..4
       stop:  5
     - start: 5      # M4 Exapode occupies elements 5..6
       stop:  7

     zernike_to_use: [1, 2, 3, 6, 7]
     push_pull_template: [1, -2, 1]
     commandMatrix:  ottCmdMat.fits
     fitting_surface: ''

``names`` : *list of str*
    Human-readable display names for each optomechanical device.  Used for
    logging and console output only.

``devices_move_calls`` : *list of str*
    Dotted callable strings for sending a position command to each device
    (e.g. ``parabola.setPosition``).  At runtime these strings are resolved
    against the objects in scope.

``devices_read_calls`` : *list of str*
    Dotted callable strings for reading back the current position of each
    device.

``ccd_acquisition`` : *list of str*
    Callable string(s) for the wavefront acquisition step (interferometer or
    camera).  The first element must be the acquisition method name (e.g.
    ``acquire_map``).

``devices_dof`` : *int or list of int*
    Total number of degrees of freedom (DOF) accepted by the device at the
    software interface level, i.e. the length of the command vector the
    device understands.  A single integer applies the same value to all
    devices; a list allows per-device values.

``dof`` : *list of lists of int*
    For each device, the indices within the full ``devices_dof``-element
    command vector that correspond to the DOF the device *can actually move*.

    **Example** — a device that speaks 6 DOF but can only move piston (2),
    tip (3), and tilt (4):

    .. code-block:: yaml

       dof:
       - [2, 3, 4]

    This yields an effective command ``[0, 0, p, t, t, 0]``.

``slices`` : *list of {start, stop}*
    Defines how to split the global alignment command vector into
    per-device sub-vectors.  Each entry is a mapping with ``start`` and
    ``stop`` keys (Python ``slice(start, stop)`` convention — ``stop`` is
    exclusive).

    **Example** — a 7-element command vector split across three devices:

    .. code-block:: yaml

       slices:
       - start: 0   # device 1 gets elements [0, 1, 2]
         stop:  3
       - start: 3   # device 2 gets elements [3, 4]
         stop:  5
       - start: 5   # device 3 gets elements [5, 6]
         stop:  7

``zernike_to_use`` : *list of int*
    Zernike polynomial indices (1-based Noll convention) extracted from the
    wavefront fit and used to build the alignment interaction matrix.

``push_pull_template`` : *list of int*
    Push-pull sequence applied to each Zernike mode during interaction
    matrix acquisition.  Commands are differential (relative to zero), so a
    classic ``[1, -1, 1]`` template is written as ``[1, -2, 1]``.

``commandMatrix`` : *str*
    Filename of the alignment control matrix (``.fits``), located in
    ``<data_path>/Alignment/ControlMatrices/``.

``fitting_surface`` : *str*
    Path to a FITS file whose mask defines the pupil region used for the
    Zernike fit.  Leave empty to use the full interferometer pupil.

Builtin example: M4's OTT
~~~~~~~~~~~~~~~~~~~~~~~~~~

The template ``configuration.yaml`` ships pre-filled with the configuration
for the ELT@M4 Optical Test Tower (OTT).  The OTT has three optomechanical
devices — Parabola, Reference Mirror, and M4 Exapode — each accepting a
6-element command vector.  Their movable DOF and the 7-element global
alignment vector are set up as shown in the YAML snippet above.

----

Configuration API
-----------------

The configuration file is read and written programmatically through
:mod:`opticalib.core.read_config`.  The most commonly used functions are:

.. code-block:: python

    from opticalib.core import read_config

    # Load the full configuration dictionary
    cfg = read_config.load_yaml_config()

    # --- Device access ---
    # Get configuration for a named interferometer
    interf_cfg = read_config.getInterfConfig('PhaseCam6110')

    # Get configuration for a named DM
    dm_cfg = read_config.getDmConfig('AlpaoDm97')

    # Get configuration for a named camera
    cam_cfg = read_config.getCamerasConfig('AVT_MANTA_5G')

    # Generic device config lookup (any DEVICES sub-section)
    raw = read_config.getDeviceConfig('MOTORS', 'TunableFilter')

    # --- IFF access ---
    # Get DM hardware parameters
    dm_iff = read_config.getDmIffConfig()
    nacts  = read_config.getNActs()
    timing = read_config.getTiming()

    # Get one IFF acquisition block ('TRIGGER', 'REGISTRATION', or 'IFFUNC')
    iff_block = read_config.getIffConfig('IFFUNC')
    # Returns: {zeros, modes, amplitude, template, modalBase, paddingZeros}

    # --- Phasing access ---
    phasing_cfg = read_config.getPhasingConfig()

    # --- Alignment and stitching access ---
    align_cfg    = read_config.getAlignmentConfig()   # returns attribute-access object
    stitching_cfg = read_config.getStitchingConfig()  # returns dict
