DM Utilities
============

.. automodule:: opticalib.dmutils
   :no-members:

Stateless and semi-stateless helpers for deformable-mirror calibration:
planning an influence-function capture, reducing the acquired data, running a
flattening loop, and slaving one mirror to another.

For the *multi-step, stateful* orchestration of a full calibration run, see
:doc:`procedures`.

Influence-function capture planning
-----------------------------------

.. automodule:: opticalib.dmutils.iff_preparation
   :members:

Influence-function processing
-----------------------------

.. automodule:: opticalib.dmutils.iff_processing
   :members:

Flattening
----------

.. automodule:: opticalib.dmutils.flattening
   :members:

Slaving
-------

.. automodule:: opticalib.dmutils.slaving
   :members:

Modal bases
-----------

.. autofunction:: opticalib.dmutils.make_modal_base

Backward-compatible aliases
---------------------------

The following names are re-exported from :mod:`opticalib.dmutils` for
compatibility with scripts written against earlier releases.  They are *aliases*
for the canonical objects documented elsewhere -- new code should import from
the canonical location.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ``opticalib.dmutils`` alias
     - Canonical location
   * - ``iff_module``
     - :mod:`opticalib.procedures.iff`
   * - ``stitching``
     - :mod:`opticalib.procedures.stitching`
   * - ``FlatData``, ``IffData``
     - :mod:`opticalib.core.data_classes`

.. warning::
   The camelCase entry points that older scripts used
   (``iff_module.iffDataAcquisition``, ``iff_module.pistonDataAcquisition``)
   have been **removed**.  Use
   :func:`~opticalib.procedures.iff.iff_data_acquisition` and
   :func:`~opticalib.procedures.iff.piston_data_acquisition` instead.
