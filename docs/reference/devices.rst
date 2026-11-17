Devices
=======

.. automodule:: opticalib.devices
   :no-members:

Hardware drivers for the instruments on an optical bench.  Every concrete class
satisfies one of the :doc:`device protocols <typings>`, so higher-level code
never needs to know which vendor it is talking to.

.. list-table:: Supported hardware
   :header-rows: 1
   :widths: 22 26 52

   * - Kind
     - Class
     - Vendor / notes
   * - Interferometer
     - :class:`~opticalib.devices.interferometer.PhaseCam`
     - 4D Technology PhaseCam (4D Technology ``4D API``)
   * - Interferometer
     - :class:`~opticalib.devices.interferometer.AccuFiz`
     - 4D Technology AccuFiz
   * - Interferometer
     - :class:`~opticalib.devices.interferometer.Processer4D`
     - 4D Technology Processer
   * - Deformable mirror
     - :class:`~opticalib.devices.deformable_mirrors.AlpaoDm`
     - Alpao, via the ``asdk`` vendor SDK
   * - Deformable mirror
     - :class:`~opticalib.devices.deformable_mirrors.SplattDm`
     - SPLATT, via ``Pyro4`` remote objects
   * - Deformable mirror
     - :class:`~opticalib.devices.deformable_mirrors.AdOpticaDm`
     - AdOptica
   * - Deformable mirror
     - :class:`~opticalib.devices.deformable_mirrors.DP`
     - Adaptive Optics Associates / INAF ``DP``
   * - Deformable mirror
     - :class:`~opticalib.devices.deformable_mirrors.M4AU`
     - ELT M4 Actuator Unit (Microgate)
   * - Deformable mirror
     - :class:`~opticalib.devices.deformable_mirrors.PetalMirror`
     - Segmented / petal mirror (PI stage)
   * - Wavefront sensor
     - :class:`~opticalib.devices.wfs.Ingot`
     - INGO-T style WFS
   * - Camera
     - :class:`~opticalib.devices.cameras.GigaVision`
     - Allied Vision, via ``vmbpy``

Interferometers
---------------

.. automodule:: opticalib.devices.interferometer
   :members:

All three interferometers above share the 4D Technology base class below, which
is where ``acquire_map``, ``capture`` and ``produce`` are actually implemented.

.. autoclass:: opticalib.devices.interferometer._4DInterferometer
   :members:

Deformable mirrors
------------------

.. automodule:: opticalib.devices.deformable_mirrors
   :members:

Wavefront sensors
-----------------

.. automodule:: opticalib.devices.wfs
   :members:

Cameras
-------

.. automodule:: opticalib.devices.cameras
   :members:

Device base classes
-------------------

Abstract bases in :mod:`opticalib.devices._API.base_devices`.  Subclass these
when adding support for new hardware -- see :doc:`/developer_guide/adding_a_device`.

.. automodule:: opticalib.devices._API.base_devices
   :members:

.. note::
   The remaining modules under ``opticalib.devices._API`` are thin,
   vendor-specific wrappers around SDKs that are not redistributable
   (``asdk``, ``Pyro4``, ``pipython``, ``vmbpy``).  They are intentionally not
   part of the published reference; use the high-level classes above.
