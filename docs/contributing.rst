Contributing
============

We welcome contributions to OptiCalib!  Please follow these guidelines to
keep the codebase consistent and maintainable.

Getting started
---------------

1. **Fork** the repository on GitHub and clone your fork::

       git clone https://github.com/<your-username>/opticalib.git
       cd opticalib

2. Create a feature branch::

       git checkout -b feature/my-feature

3. Install the package in editable mode together with the development
   dependencies::

       pip install -e ".[dev]"

4. Make your changes, add tests, and open a pull request against ``main``.

Code style
----------

* The project follows **PEP 8**.  Use the ``flake8`` linter to check your code
  before committing.
* Use **type hints** for all public functions and methods.
* Write **NumPy-style docstrings** for all public symbols (functions, classes,
  methods).  See the example below.

Example docstring
~~~~~~~~~~~~~~~~~

.. code-block:: python

    def compute_flat_cmd(
        dm: DeformableMirrorDevice,
        interf: InterfDevice,
        n_modes: int = 50,
    ) -> np.ndarray:
        """
        Compute a flattening command for *dm* using *interf*.

        Parameters
        ----------
        dm : DeformableMirrorDevice
            The deformable mirror to flatten.
        interf : InterfDevice
            The interferometer used to measure the wavefront.
        n_modes : int, optional
            Number of Zernike modes to include in the reconstruction.
            Defaults to 50.

        Returns
        -------
        flat_cmd : np.ndarray
            Command vector of length ``dm.nActs``.

        Raises
        ------
        MatrixError
            If the interaction matrix is singular.
        """

Tests
-----

All new features must be accompanied by unit tests located in the ``tests/``
directory.  Run the test suite with::

    pytest

Run only a targeted subset to speed up iteration::

    pytest tests/test_dmutils_flattening.py -v

Use the following markers to classify your tests:

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Marker
     - Description
   * - ``@pytest.mark.unit``
     - Fast, dependency-free unit tests
   * - ``@pytest.mark.integration``
     - Tests that require file I/O or multiple components
   * - ``@pytest.mark.slow``
     - Long-running tests (excluded from the default CI run)

Building the documentation
--------------------------

Documentation sources live in the ``docs/`` folder.  To build the HTML docs
locally::

    cd docs
    pip install -r requirements.txt
    make html

The output is written to ``docs/_build/html/``.  Open
``docs/_build/html/index.html`` in your browser to preview the result.

.. note::
   The docs are built automatically on `Read the Docs`_ on every push to
   ``main``.

.. _Read the Docs: https://readthedocs.org/projects/opticalib/

Versioning and releases
-----------------------

The version of ``opticalib`` is not stored in the source: it is derived from
the git tags by `setuptools-scm`_.  There are no "bump the version" commits.

* A commit exactly on a release tag ``vX.Y.Z`` is versioned ``X.Y.Z``.
* Any other commit is a development version, such as ``2.5.0.dev12+g1a2b3c4``
  (``devN`` is the number of commits since the last anchor tag, ``g<hash>`` the
  commit).
* A new development cycle is opened with a ``.dev0`` *anchor* tag, on the
  first commit of the cycle::

      git tag v2.6.0.dev0
      git push origin v2.6.0.dev0

* A release is made by creating a GitHub release with the tag ``vX.Y.Z`` on
  ``main``.  Publishing the release triggers the workflow that builds the
  package and uploads it to PyPI; it fails if the built version does not match
  the tag.

``opticalib.__version__`` is read from a ``_version.py`` file generated at
build/install time.  In an editable install it is refreshed only on
reinstall, so run ``pip install -e .`` again to update it after new commits or
tags.

.. note::
   This is unrelated to the *configuration schema version*
   (``opticalib.core.config.CONFIG_SCHEMA_VERSION``), which describes the
   layout of ``configuration.yaml``.  It is bumped only when that layout
   changes; see the configuration documentation.

.. _setuptools-scm: https://setuptools-scm.readthedocs.io/

Reporting issues
----------------

Open an issue on the `GitHub issue tracker`_.  Please include:

* A minimal, reproducible example.
* The version of ``opticalib`` (``python -c "import opticalib; print(opticalib.__version__)"``).
* The Python version and operating system.

.. _GitHub issue tracker: https://github.com/ArcetriAdaptiveOptics/opticalib/issues

License
-------

By contributing you agree that your contributions will be licensed under the
`MIT License <https://opensource.org/licenses/MIT>`_ that covers the project.
