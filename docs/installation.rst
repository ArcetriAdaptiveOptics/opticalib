Installation
============

Requirements
------------

OptiCalib requires **Python 3.10** or newer.  The package depends on several
scientific Python libraries that are listed in the ``dependencies`` of
``pyproject.toml`` at the root of the repository.  All of them are installed
automatically when you use ``pip``.  The graphical interface and the GPU backend
are optional, see :ref:`optional-extras`.

Installing from PyPI (stable release)
--------------------------------------

The latest stable version is available on the `Python Package Index`_::

    pip install opticalib

.. _Python Package Index: https://pypi.org/project/opticalib/

Installing the development version
------------------------------------

To install the bleeding-edge version directly from GitHub::

    pip install git+https://github.com/ArcetriAdaptiveOptics/opticalib.git

.. warning::
   The development version may contain bugs or incomplete features.

Installing from source
-----------------------

If you want to contribute to the package or inspect the source code, clone the
repository and install it in *editable* mode::

    git clone https://github.com/ArcetriAdaptiveOptics/opticalib.git
    cd opticalib
    pip install -e ".[dev]"

``dev`` installs the optional graphical interface together with the test tools;
use plain ``pip install -e .`` for the base package only.

Setting up the experiment environment
--------------------------------------

Upon installation, the ``calpy`` entry-point script is added to your
``PATH``.  Use it to create and activate a workspace for your experiment::

    # Create a new environment in ~/my_experiment
    calpy -f ~/my_experiment --create

    # Activate an existing environment
    calpy -f ~/my_experiment

See :doc:`configuration` for details on editing the generated configuration
file.

.. _optional-extras:

Optional extras
---------------

Some features require packages that are not installed by default.  They are
installed as *extras*, e.g. ``pip install "opticalib[gui]"``:

.. list-table::
   :header-rows: 1
   :widths: 25 50

   * - Extra
     - Feature enabled
   * - ``gui``
     - The graphical interface (``calpy --gui`` and the ``OptiCalib`` launcher):
       PySide6, QtPy, qtconsole, ipykernel, pyqtgraph, ...
   * - ``cuda12`` or ``cuda13``
     - GPU-accelerated processing tools (iff, slaving, modal decomposition).
       Installs ``xupy`` with CuPy, enabling GPU-accelerated masked arrays.
   * - ``test``
     - ``pytest`` and ``pytest-cov``, to run the test suite.
   * - ``dev``
     - ``gui`` + ``test`` + ``flake8``, for contributors.

Extras can be combined, e.g. ``pip install "opticalib[gui,cuda13]"``.

.. note::
   Pick ``cuda12`` or ``cuda13`` according to your CUDA Toolkit version
   (``nvidia-smi`` shows the maximum version the driver supports).  Without an
   extra, ``xupy`` runs on the CPU with NumPy.

Verifying the installation
--------------------------

After installation you can verify that the package is importable::

    python -c "import opticalib; print(opticalib.__version__)"
