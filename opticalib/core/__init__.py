"""
Core — configuration, data folders, shared types and I/O
========================================================

The :mod:`opticalib.core` package is the foundation of the library.  It
owns the YAML configuration file, the on-disk data folder tree, the
exception hierarchy, shared decorators, structured data classes for
passing calibration results between components, and the FITS-aware masked
array wrappers.

Every other area of the library depends on core.  Importing it reads
``AOCONF``, opens the configuration file, and calls
:func:`~opticalib.core.root.create_folder_tree` — set the environment
variable before import (or use the ``calpy`` entry point, which handles
it automatically).

"""
