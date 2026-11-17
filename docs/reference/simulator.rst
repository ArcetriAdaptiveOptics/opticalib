Simulator
=========

.. automodule:: opticalib.simulator
   :no-members:

Software stand-ins for the bench.  The simulator provides *fake* interferometers
and deformable mirrors that satisfy the same :doc:`device protocols <typings>`
as the real drivers, so any procedure, analysis script or GUI can be developed
and tested without hardware attached.

.. tip::
   Point ``DEVICES`` at a simulated device in ``configuration.yaml`` and the
   rest of the library is none the wiser.  See :doc:`/configuration`.

Factory helpers
---------------

Small utilities for generating the coordinate grids and masks that the fake
devices need.

.. automodule:: opticalib.simulator.factory
   :members:

Fake deformable mirrors
-----------------------

These mirror (in name and geometry) their counterparts in
:mod:`opticalib.devices.deformable_mirrors`.

.. automodule:: opticalib.simulator.fake_dms
   :members:

Fake interferometers
--------------------

.. automodule:: opticalib.simulator.fake_interf
   :members:

Simulator base classes
----------------------

The bases in ``opticalib.simulator._API`` hold the shared physics and
rendering behaviour.  Subclass them when adding a new simulated instrument.

.. automodule:: opticalib.simulator._API.base_petalmirror
   :members:

.. automodule:: opticalib.simulator._API.base_fake_alpao
   :members:

.. automodule:: opticalib.simulator._API.base_fake_adopticadm
   :members:

.. note::
   ``opticalib.simulator._API.simdata`` and ``_rbf_gpu`` are internal support
   modules for loading simulated datasets and for the optional GPU-accelerated
   radial-basis reconstruction path.  They are not part of the published
   reference; both require data files or a CUDA toolchain that are not shipped
   with the library.
