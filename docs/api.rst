API Reference
=============

This reference is organised by *functional area* rather than as a flat dump of
every module.  Each page documents the modules that define its symbols, so
every class and function has exactly one canonical location -- re-exports from
a package ``__init__`` link back here instead of duplicating the entry.

.. tip::
   Looking for the high-level workflow instead of a symbol?  Start from
   :doc:`/quickstart`, or read the :doc:`/user_guide/calpy` guide.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: :octicon:`gear` Core
      :link: reference/core
      :link-type: doc

      Configuration, data folders, exceptions, decorators and the FITS array
      wrappers.  The foundation every other area builds on.

   .. grid-item-card:: :octicon:`code` Type System
      :link: reference/typings
      :link-type: doc

      The ``ImageData`` / ``CubeData`` / ``MatrixLike`` aliases and the device
      protocols that make OptiCalib hardware-agnostic.

   .. grid-item-card:: :octicon:`device-camera` Devices
      :link: reference/devices
      :link-type: doc

      Interferometers, deformable mirrors, wavefront sensors and cameras.

   .. grid-item-card:: :octicon:`sliders` DM Utilities
      :link: reference/dmutils
      :link-type: doc

      Influence-function capture and processing, flattening, slaving.

   .. grid-item-card:: :octicon:`play` Procedures
      :link: reference/procedures
      :link-type: doc

      Multi-step, stateful bench operations: alignment, IFF, measurements,
      phasing and stitching.

   .. grid-item-card:: :octicon:`graph` Ground
      :link: reference/ground
      :link-type: doc

      Reconstruction, modal decomposition, geometry, ROIs, logging and OS/file
      helpers.

   .. grid-item-card:: :octicon:`pulse` Analyzer
      :link: reference/analyzer
      :link-type: doc

      Frame and cube analysis, time series, spectra and noise diagnostics.

   .. grid-item-card:: :octicon:`beaker` Simulator
      :link: reference/simulator
      :link-type: doc

      Fake interferometers and deformable mirrors for offline development.

   .. grid-item-card:: :octicon:`eye` Visualization
      :link: reference/visualization
      :link-type: doc

      Matplotlib helpers for images, surfaces and command plots.

   .. grid-item-card:: :octicon:`browser` GUI
      :link: reference/gui
      :link-type: doc

      The optional ``calpy`` Qt application shell.

.. toctree::
   :maxdepth: 2
   :hidden:

   reference/core
   reference/typings
   reference/devices
   reference/dmutils
   reference/procedures
   reference/ground
   reference/analyzer
   reference/simulator
   reference/visualization
   reference/gui

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
