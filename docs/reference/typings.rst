Type System
===========

.. py:module:: opticalib.core._types

OptiCalib is deliberately *hardware agnostic*: algorithms accept anything
that behaves like an image, a cube or a deformable mirror, rather than a
concrete class.  This module defines that contract.

It is re-exported publicly as ``opticalib.typings``:

.. code-block:: python

   import opticalib as tn
   img: tn.typings.ImageData = interf.acquire_map()

There are two layers:

1. **Data aliases** -- ``MatrixLike``, ``MaskData``, ``ImageData``,
   ``CubeData``, ``FitsData`` -- describe array-shaped payloads.
2. **Device protocols** -- ``InterferometerDevice``, ``WFSDevice``,
   ``CameraDevice``, ``DeformableMirrorDevice`` and their ``Fake*`` variants --
   describe the *methods* a device must expose.

Every alias is a :class:`~typing.TypeVar` bound to a structural
:class:`~typing.Protocol`, so static type checkers verify shape/method
compatibility without any inheritance relationship.

Data aliases
------------

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Alias
     - Kind
     - Meaning
   * - ``MatrixLike``
     - 2-D
     - Anything indexable with a ``shape`` -- ``numpy.ndarray``, a masked
       array, or a :class:`~opticalib.core.fitsarray.FitsArray`.
   * - ``MaskData``
     - 2-D
     - A boolean/integer pupil mask, same container rules as ``MatrixLike``.
   * - ``ImageData``
     - 2-D
     - A single interferogram or phase map: has ``data``, ``mask`` and
       ``__array__``.
   * - ``CubeData``
     - 3-D
     - A stack of ``ImageData`` frames, as produced by an IFF acquisition.
   * - ``FitsData``
     - 2-D/3-D
     - Adds ``writeto`` / ``fromfits`` so the object can round-trip to FITS.
   * - ``Reconstructor``
     - object
     - ``ComputeReconstructor | None``.

.. autodata:: opticalib.core._types.MatrixLike
   :no-value:

.. autodata:: opticalib.core._types.MaskData
   :no-value:

.. autodata:: opticalib.core._types.ImageData
   :no-value:

.. autodata:: opticalib.core._types.CubeData
   :no-value:

.. autodata:: opticalib.core._types.FitsData
   :no-value:

.. autodata:: opticalib.core._types.Reconstructor
   :no-value:

.. autodata:: opticalib.core._types.Number
   :no-value:

Device aliases
--------------

These are what you annotate a parameter with when a function accepts "any
interferometer" or "any deformable mirror", real or simulated.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Alias
     - Required behaviour
   * - ``InterferometerDevice``
     - ``acquire_map``, ``acquire_full_frame``, ``capture``, ``produce``
   * - ``WFSDevice``
     - ``acquire_map``, ``acquire_pupil``, ``acquire_detector``
   * - ``CameraDevice``
     - ``acquire_frames``, ``set_exptime``, ``get_exptime``
   * - ``DeformableMirrorDevice``
     - ``n_acts``, ``set_shape``, ``get_shape``, ``upload_cmd_history``,
       ``run_cmd_history``
   * - ``FakeDeformableMirrorDevice``
     - the above, plus ``_mask``, ``_zern``, ``_wavefront``
   * - ``FakeInterferometerDevice``
     - the interferometer set, plus the live-view controls
   * - ``GenericDevice``
     - unconstrained; used where any device object is acceptable

.. autodata:: opticalib.core._types.InterferometerDevice
   :no-value:

.. autodata:: opticalib.core._types.CameraDevice
   :no-value:

.. autodata:: opticalib.core._types.WFSDevice
   :no-value:

.. autodata:: opticalib.core._types.DeformableMirrorDevice
   :no-value:

.. autodata:: opticalib.core._types.FakeDeformableMirrorDevice
   :no-value:

.. autodata:: opticalib.core._types.FakeInterferometerDevice
   :no-value:

.. autodata:: opticalib.core._types.GenericDevice
   :no-value:

Structural protocols
--------------------

The ``Protocol`` classes below are the actual contracts.  They are listed for
completeness and because they appear in rendered signatures; you should
normally annotate with the aliases above instead.

.. autoclass:: opticalib.core._types._MatrixProtocol
   :no-members:

.. autoclass:: opticalib.core._types._ImageDataProtocol
   :no-members:

.. autoclass:: opticalib.core._types._CubeProtocol
   :no-members:

.. autoclass:: opticalib.core._types._FitsArrayProtocol
   :no-members:

.. autoclass:: opticalib.core._types._FitsMaskedArrayProtocol
   :no-members:

.. autoclass:: opticalib.core._types._InterfProtocol
   :no-members:

.. autoclass:: opticalib.core._types._WFSProtocol
   :no-members:

.. autoclass:: opticalib.core._types._CameraProtocol
   :no-members:

.. autoclass:: opticalib.core._types._DMProtocol
   :no-members:

.. autoclass:: opticalib.core._types._FakeDMProtocol
   :no-members:

.. autoclass:: opticalib.core._types._FakeInterfProtocol
   :no-members:

Runtime type checking
---------------------

Because the aliases are structural, ``isinstance(x, ImageData)`` does not work.
Use :func:`~opticalib.core._types.isinstance_` instead -- it dispatches to the
right check for the given name.

.. autoclass:: opticalib.core._types.InstanceCheck
   :members:

.. autofunction:: opticalib.core._types.isinstance_

.. autofunction:: opticalib.core._types.get_device_type

.. autofunction:: opticalib.core._types.array_str_formatter
