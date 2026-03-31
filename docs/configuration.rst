Configuration
=============

``opticalib`` uses a single YAML file, ``configuration.yaml``, to describe your
experimental setup.  The file is generated automatically when you run
``calpy -f <path> --create`` and lives under ``<path>/SysConfig/``.

.. contents:: Sections
   :depth: 2
   :local:

----

SYSTEM section
--------------

.. code-block:: yaml

   SYSTEM:
     data_path: ''
     simulated.devices:
       dm: true
       interferometer: true

``data_path``
    Base path where the data folder tree is stored.  Leave empty to use
    ``~/opticalib_data/``.

``simulated.devices.dm``
    Set to ``true`` to use the simulated deformable mirror from
    :mod:`opticalib.simulator` instead of real hardware.

``simulated.devices.interferometer``
    Set to ``true`` to use the simulated interferometer from
    :mod:`opticalib.simulator` instead of real hardware.

----

DEVICES section
---------------

The ``DEVICES`` section contains three sub-sections: ``INTERFEROMETER``,
``DEFORMABLE.MIRRORS``, and ``CAMERAS``.

Interferometers
~~~~~~~~~~~~~~~

.. code-block:: yaml

   DEVICES:
     INTERFEROMETER:
       PhaseCam6110:        # arbitrary name you choose
         ip:   192.168.1.10
         port: 8011
         Paths:
           settings:        /mnt/4dpc/AppSettings.ini
           copied_settings: /home/user/4d_settings/AppSettings.ini
           capture_4dpc:    /mnt/4dpc/Capture
           produce_4dpc:    /mnt/4dpc/Produce
           produce:         /home/user/produce
           capture:         /home/user/capture

``ip``
    IP address of the 4D interferometer computer.

``port``
    Network port of the 4D server.

``Paths.settings``
    Path to the 4D ``AppSettings.ini`` file (must be network-mounted).

``Paths.copied_settings``
    Local copy of ``AppSettings.ini`` used for record-keeping.

``Paths.capture_4dpc`` / ``Paths.produce_4dpc``
    Capture and produce directories on the 4D PC (network-mounted).

``Paths.produce`` / ``Paths.capture``
    Local destination directories for data transfer.

Deformable Mirrors
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   DEVICES:
     DEFORMABLE.MIRRORS:
       AlpaoDm97:
         serialNumber: BAX240
         diameter: 10.5

``serialNumber``
    Hardware serial number of the Alpao mirror.

``diameter``
    Aperture diameter in millimetres.

Cameras
~~~~~~~

.. code-block:: yaml

   DEVICES:
     CAMERAS:
       AVT_MANTA_5G:
         id:  DEV_000F314DE425
         ip:

``id``
    Unique Vimba device identifier.

``ip``
    IP address (for GigE cameras).

----

IFF section
-----------

The ``IFF`` section controls the default parameters for Influence Function
acquisition.

.. code-block:: yaml

   IFF:
     modesList:       [1, 2, 3, 4, 5]
     modesAmplitude:  0.1
     template:        [1, -1, 1]
     nFrames:         5
     delay:           0.05

``modesList``
    List of actuator or mode indices to drive during the IFF acquisition.

``modesAmplitude``
    Push/pull amplitude (in mirror units, typically micrometers).

``template``
    Push-pull template.  A classic ``[1, -1, 1]`` template applies
    ``[+A, -A, +A]`` voltage steps for each mode.

``nFrames``
    Number of interferometer frames to average per acquisition step.

``delay``
    Delay in seconds between consecutive interferometer acquisitions.

----

ALIGNMENT section
-----------------

The ``ALIGNMENT`` section is used by :class:`~opticalib.alignment.Alignment`
to calibrate and correct the optical alignment of the system.

.. code-block:: yaml

   ALIGNMENT:
     names:              [Parabola, RefMirror, M4Exapode]
     devices_move_calls: [par.move, rm.move, m4.move]
     devices_read_calls: [par.read, rm.read, m4.read]
     ccd_acquisition:    [interf.acquire_map]
     devices_dof:        6
     dof:
       - [2, 3, 4]
       - [3, 4]
       - [3, 4]
     slices:
       - {start: 0, stop: 3}
       - {start: 3, stop: 5}
       - {start: 5, stop: 7}
     zernike_to_use:     [2, 3, 4, 5, 6, 7]
     push_pull_template: [1, -2, 1]
     commandMatrix:      alignment_matrix.fits
     fitting_surface:    ''

``names``
    Display names of the optomechanical devices (for logging).

``devices_move_calls``
    Callable strings used to move each device.

``devices_read_calls``
    Callable strings used to read back each device position.

``ccd_acquisition``
    Callable strings for the wavefront acquisition device.

``devices_dof``
    Number of degrees of freedom accepted by each device (can be a single
    integer if all devices share the same DOF count, or a list).

``dof``
    For each device, the subset of DOF indices that the device can *actually*
    move.  Indices correspond to positions within the full DOF vector.

``slices``
    Start/stop slice definitions that map the alignment command vector into
    per-device sub-vectors.

``zernike_to_use``
    Zernike mode indices extracted from the wavefront fit and used to build
    the interaction matrix.

``push_pull_template``
    Push-pull template for alignment interaction matrix acquisition.

``commandMatrix``
    File name of the control matrix (``.fits``), relative to
    ``<data_path>/Alignment/ControlMatrices/``.

``fitting_surface``
    Path to a FITS file whose mask defines the pupil for the Zernike fit.
    Leave empty to use the full interferometer pupil.

----

Configuration API
-----------------

The configuration file is managed programmatically through
:mod:`opticalib.core.read_config`:

.. code-block:: python

    from opticalib.core import read_config

    # Load the full configuration dict
    cfg = read_config.load_yaml_config()

    # Get parameters for a specific device
    dm_cfg = read_config.getDmConfig('AlpaoDm97')

    # Get IFF acquisition parameters
    iff_cfg = read_config.getIffConfig()
