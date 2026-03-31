Installation
============

Requirements
------------

``opticalib`` requires **Python 3.10** or newer.  The package depends on several
scientific Python libraries that are listed in ``requirements.txt`` at the root
of the repository.  All of them are installed automatically when you use
``pip``.

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
    pip install -e .

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

Optional dependencies
---------------------

Some features require packages that are not installed by default:

.. list-table::
   :header-rows: 1
   :widths: 25 50

   * - Package
     - Feature enabled
   * - ``cupy-cuda12x`` or ``cupy-cuda13x``
     - GPU-accelerated processing tools (iff, slaving, modal decomposition). Enables ``xupy`` on gpu, allowing for GPU-accelerated masked arrays.
      

.. note::
   The appropriate version of ``cupy`` depends on your CUDA Toolkit version.

Verifying the installation
--------------------------

After installation you can verify that the package is importable::

    python -c "import opticalib; print(opticalib.__version__)"
