Quick Start
===========

This guide shows the most common workflow: connecting to instruments, acquiring
interferometer images, and running a DM calibration.

Setting up an experiment
------------------------

After :doc:`installing <installation>` ``opticalib``, create a workspace with
the ``calpy`` command::

    calpy -f ~/alpao_experiment --create

This generates the following folder tree under ``~/alpao_experiment``:

.. code-block:: text

    alpao_experiment/
    ├── OPTData/
    │   ├── Flattening/
    │   ├── INTMatrices/
    │   ├── ModalBases/
    │   ├── OPDImages/
    │   ├── OPDSeries/
    |   ├── SPL/
    |   |   ├── Fringes/
    │   └── IFFunctions/
    ├── Logging/
    └── SysConfig/
        └── configuration.yaml

Edit ``SysConfig/configuration.yaml`` to describe your hardware (see
:doc:`configuration`).

Activating the environment
--------------------------

Run::

    calpy -f ~/alpao_experiment

``calpy`` will import ``opticalib`` (aliased as ``opt``) and
``opticalib.dmutils`` (aliased as ``dmutils``) and set the data root to your
experiment folder.

Connecting to instruments
--------------------------

Device classes are re-exported at the top level, so the common case is simply
``import opticalib as opt``:

.. code-block:: python

    import opticalib as opt

    # Connect to a 4D PhaseCam interferometer.
    # NOTE: the first positional argument is the *model*, not the address --
    # always pass ``ip``/``port`` by keyword.
    interf = opt.PhaseCam(ip='192.168.1.10', port=8011)

    # Connect to an Alpao deformable mirror with 820 actuators.
    # When several mirrors are declared in configuration.yaml, prefer the
    # serial number, which is unambiguous.
    dm = opt.AlpaoDm(820)
    dm = opt.AlpaoDm(serial_number='BAX751')

.. tip::
   ``ip``, ``port`` and ``serial_number`` are read from ``configuration.yaml``
   when omitted, so a correctly configured bench needs only
   ``opt.PhaseCam()`` / ``opt.AlpaoDm()``.  See :doc:`configuration`.

Acquiring a wavefront image
----------------------------

.. code-block:: python

    # Acquire a single wavefront map (returns an ImageData masked array)
    wf = interf.acquire_map()

    import matplotlib.pyplot as plt
    plt.imshow(wf)
    plt.colorbar(label='OPD [m]')
    plt.title('Wavefront map')
    plt.show()

``acquire_map`` accepts ``nframes``, ``delay`` and ``rebin`` to average several
exposures, wait for the mirror to settle, and downsample the result.  It is
implemented once on the shared 4D base class -- see
:class:`~opticalib.devices.interferometer._4DInterferometer`.

Acquiring Influence Functions
-------------------------------

:func:`~opticalib.procedures.iff.iff_data_acquisition` orchestrates the full
push-pull measurement loop.  Parameters that are not supplied are read from the
``INFLUENCE.FUNCTIONS`` section of ``configuration.yaml``:

.. code-block:: python

    from opticalib.procedures import iff_data_acquisition

    # Acquire IFF data – returns a tracking number (timestamp string)
    tn = iff_data_acquisition(dm, interf)
    print(f"IFF data saved under tracking number: {tn}")

.. warning::
   Older releases exposed this as ``opticalib.dmutils.iff_module.iffDataAcquisition``.
   That camelCase name is gone; ``opticalib.dmutils.iff_module`` remains only as
   an alias for :mod:`opticalib.procedures.iff`.

Processing Influence Functions
--------------------------------

.. code-block:: python

    from opticalib.dmutils import iff_processing

    # Process the acquired data
    iff_processing.process(tn)

Flattening the deformable mirror
----------------------------------

.. code-block:: python

    from opticalib.dmutils import Flattening

    # Flattening is keyed by tracking number: it loads the interaction matrix
    # derived from that IFF capture.
    flat = Flattening(tn, dm, interf)
    flat.compute_rec_mat(threshold=30)
    flat.apply_flat_command()      # solve, send and measure the result

.. warning::
   The pre-2.0 spelling ``Flattening(dm, interf).applyFlatCommand()`` no longer
   works: the tracking number is now the first argument and the method is
   :meth:`~opticalib.dmutils.flattening.Flattening.apply_flat_command`.

Using simulated devices (no hardware)
---------------------------------------

``opticalib`` ships with a :mod:`~opticalib.simulator` sub-package for
offline testing:

.. code-block:: python

    from opticalib.simulator import AlpaoDm, Fake4DInterf

    dm    = AlpaoDm(nActs=97)
    interf = Fake4DInterf()

    wf = interf.acquire_map()
    print(wf.shape)

Next steps
----------

* :doc:`user_guide/calpy` -- the experiment framework and a full calibration
  session, step by step.
* :doc:`configuration` -- configure your hardware devices.
* :doc:`api` -- the curated API reference.
