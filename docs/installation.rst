Installation
============

Requirements
------------

OptiCalib requires **Python 3.11** or newer.  The package depends on several
scientific Python libraries that are listed in the ``dependencies`` of
``pyproject.toml`` at the root of the repository.  All of them are installed
automatically when you use ``pip``.  The graphical interface and the GPU backend
are optional, see :ref:`optional-extras`.

.. note::
   **macOS**: ``vmbpy`` (the Allied Vision Vimba X SDK) is distributed only for
   Linux and Windows, so it is not installed on macOS.  Everything else works;
   only the :class:`~opticalib.devices.cameras.GigaVision` camera (and the
   devices built on it, such as the ``Ingot`` WFS camera) raises an
   ``ImportError`` when created.  With a Python from the python.org installer,
   see also :ref:`macos-certificates`.

.. note::
   **Windows**: the camera live view (:meth:`GigaVision.live_view
   <opticalib.devices.cameras.GigaVision.live_view>`) is available only on
   Linux and macOS.  Stitching (``opticalib.procedures.stitching`` /
   ``map_stitching``) on Windows requires the GPU backend (``xupy`` running on
   GPU, e.g. ``pip install "opticalib[cuda12]"``).

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

.. note::
   Development installs report versions such as ``2.5.0.dev12+g1a2b3c4``,
   derived from the git tags (see :doc:`contributing`).

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

Troubleshooting
---------------

.. _macos-certificates:

macOS: certificate error when downloading the simulator data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simulators download their reference data from GitHub the first time they
need it (see :func:`~opticalib.simulator.prefetch_simdata`).  On macOS this can
fail with an error like::

    FileNotFoundError: Cannot download simulator data file 'm4_data.h5' from
    'https://github.com/...': <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
    certificate verify failed: unable to get local issuer certificate (...)>

**Cause.** Your Python comes from the `python.org installer`_, which does not
use the macOS system certificates: it needs its own certificate bundle,
installed by the *Install Certificates.command* script that comes with it.
Until that script is run, Python cannot verify any HTTPS server.  ``pip``
still works, because it carries its own certificates, so installing
``opticalib`` succeeds and only the download fails.  Pythons from conda
(Anaconda, Miniconda, Miniforge) or Homebrew are not affected.

**Fix: run Install Certificates.command.**  It can be run at any time, also
long after installing Python.  First check which Python version you use::

    python3 -c "import sys; print('%d.%d' % sys.version_info[:2])"

then run the script of that version (here 3.11)::

    open "/Applications/Python 3.11/Install Certificates.command"

or double-click *Install Certificates.command* in the *Python 3.11* folder
inside *Applications* in Finder.  It installs the ``certifi`` certificate bundle and
links it where that Python's OpenSSL looks for certificates.  The fix applies
to that Python installation and to every virtual environment created from it.
Restart your Python session (or ``calpy``) afterwards.

If the *Python 3.11* folder is missing from *Applications*, run the
python.org installer of the same version again to restore it, or use the
alternative below.

**Alternative: point Python to the certifi bundle.**  Without running the
script, you can tell OpenSSL which certificate file to use with the
``SSL_CERT_FILE`` environment variable.  ``certifi`` is installed together with
``opticalib``; from the environment where ``opticalib`` is installed, run::

    export SSL_CERT_FILE="$(python -m certifi)"

This lasts for the current terminal only.  To make it permanent, add that
line, with the path printed by ``python -m certifi`` written out in full, to
your ``~/.zshrc``.

**Check.**  Both commands must end without errors::

    python -c "import urllib.request; urllib.request.urlopen('https://github.com'); print('OK')"
    python -c "from opticalib.simulator import prefetch_simdata; prefetch_simdata()"

The second one downloads every simulator data file into the local cache, so
later sessions work offline.

.. _python.org installer: https://www.python.org/downloads/macos/
