Visualization
=============

.. automodule:: opticalib.visualization
   :no-members:

Thin Matplotlib conveniences for the plots that come up constantly on an
adaptive-optics bench: pupil masks and phase maps as images, DM command vectors
as mirror-shaped surfaces, and quick-look diagnostics.

All of these are interactive-friendly -- each returns the Matplotlib artists it
created so you can keep customising the figure.

.. autofunction:: opticalib.visualization.matshow

.. autofunction:: opticalib.visualization.myimshow

.. autofunction:: opticalib.visualization.superimshow

.. autofunction:: opticalib.visualization.surfshow

.. autofunction:: opticalib.visualization.cmdplot
