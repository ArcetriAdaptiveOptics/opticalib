Ground
======

.. automodule:: opticalib.ground
   :no-members:

The computational "ground segment": reconstruction, modal decomposition,
pupil geometry, regions of interest, plus the logging and file-system helpers
used throughout the library.

Reconstruction
--------------

.. automodule:: opticalib.ground.reconstructor
   :members:

Modal decomposition
-------------------

.. automodule:: opticalib.ground.modal_decomposer
   :members:

The abstract base below defines the fitting interface shared by all three
fitters.

.. autoclass:: opticalib.ground.modal_decomposer._ModeFitter
   :members:

Pupil geometry
--------------

.. automodule:: opticalib.ground.geometry
   :members:

Regions of interest
-------------------

.. automodule:: opticalib.ground.roi
   :members:

Logging
-------

.. automodule:: opticalib.ground.logger
   :members:

OS and file helpers
-------------------

.. automodule:: opticalib.ground.osutils
   :members:
