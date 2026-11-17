Procedures
==========

.. automodule:: opticalib.procedures
   :no-members:

Procedures are **stateful, multi-step operations** that drive real (or
simulated) hardware over time.  They are classes you instantiate, configure,
and step through -- unlike the mostly stateless helpers in :doc:`dmutils`.

.. list-table::
   :header-rows: 1
   :widths: 24 26 50

   * - Procedure
     - Entry point
     - Purpose
   * - Alignment
     - :class:`~opticalib.procedures.alignment.Alignment`
     - Centre and rotate the beam on the detector
   * - Influence functions
     - :func:`~opticalib.procedures.iff.iff_data_acquisition`
     - Acquire the per-actuator IFF cube
   * - Piston
     - :func:`~opticalib.procedures.iff.piston_data_acquisition`
     - Acquire the piston-only reference data
   * - Measurements
     - :class:`~opticalib.procedures.measurements.TimeSeries`
     - Closed-loop / open-loop time-series measurements
   * - Phasing
     - :class:`~opticalib.procedures.phasing.SPL`
     - Segmented-mirror phasing (SPL algorithm)
   * - Stitching
     - :class:`~opticalib.procedures.stitching.StitchAcquire`,
       :class:`~opticalib.procedures.stitching.StitchAnalysis`
     - Multi-subfield capture and offline stitching

Alignment
---------

.. automodule:: opticalib.procedures.alignment
   :members:

Influence-function and piston acquisition
-----------------------------------------

.. automodule:: opticalib.procedures.iff
   :members:

Measurements
------------

.. automodule:: opticalib.procedures.measurements
   :members:

Phasing
-------

.. automodule:: opticalib.procedures.phasing
   :members:

Stitching
---------

.. automodule:: opticalib.procedures.stitching
   :members:
