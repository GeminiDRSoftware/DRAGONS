.. ex2_gmosim_separateCCDs.rst

.. _separateCCDs_example:

*************************
Example 2 - Separate CCDs
*************************

Both GMOS-N and GMOS-S are currently equipped with three Hamamatsu CCDs.  In
each instrument, the quantum efficiency of each CCD is different.  This is
beneficial for spectroscopy, where the most red-sensitive CCD is placed at
the red-end of the spectrum, and similarly for the blue-end.  However, for
precision photometry from imaging data, the different color responses of
each CCD do create significant issues.

The only way to get precision photometry is to reduce each CCD individually
and stack each CCD individually, never mosaicing the data to avoid mixing the
CCD responses.  Additionally, photometric standards, preferably covering a
range of color, need to be observed on each of the 3 CCDs for proper
photometric calibration.

In this example, we will show how to trigger the special recipe that will
reduce and stack the CCDs separately.

.. toctree::
   :maxdepth: 1

   ex2_gmosim_separateCCDs_dataset
   ex2_gmosim_separateCCDs_cmdline
   ex2_gmosim_separateCCDs_api
