Core
====

.. automodule:: opticalib.core
   :no-members:

The :mod:`opticalib.core` package is the foundation of the library.  It owns
the configuration file, the on-disk data folder layout, the exception
hierarchy, the shared decorators, the structured data classes used to pass
calibration results around, and the FITS-aware array wrappers.

Configuration
-------------

.. automodule:: opticalib.core.config
   :members:

.. note::
   :func:`~opticalib.core.config.load_yaml_config` and
   :func:`~opticalib.core.config.dump_yaml_config` are thin
   backward-compatibility wrappers around
   :func:`~opticalib.core.config.load` and
   :func:`~opticalib.core.config.dump`.  Prefer the shorter names in new code.

Data folders and experiment root
--------------------------------

.. automodule:: opticalib.core.root
   :members:
   :exclude-members: ConfSettingReader4D, __weakref__, __dict__, __module__, __qualname__, __slots__

.. autoclass:: opticalib.core.root.ConfSettingReader4D
   :members:

.. admonition:: Import-time side effects
   :class: warning

   Importing :mod:`opticalib.core.root` reads ``AOCONF``, opens the
   configuration file and calls :func:`~opticalib.core.root.create_folder_tree`
   on the resolved data path.  Set ``AOCONF`` *before* importing ``opticalib``
   -- notably in tests and in this documentation build
   (see ``docs/conf.py::_bootstrap_aoconf``).

Structured data classes
-----------------------

.. automodule:: opticalib.core.data_classes
   :members:

Exceptions
----------

.. automodule:: opticalib.core.exceptions
   :members:
   :show-inheritance:

Decorators
----------

.. automodule:: opticalib.core.decorators
   :members:
   :exclude-members: P, R, __weakref__, __dict__, __module__, __qualname__, __slots__

FITS array wrappers
-------------------

.. automodule:: opticalib.core.fitsarray
   :members:
