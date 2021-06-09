.. 05_tips_and_tricks.rst

.. include:: DRAGONSlinks.txt

.. _tips_and_tricks:

***************
Tips and Tricks
***************

.. _plot_1d:

Plot a 1-D spectrum
===================
Here is how to plot an extracted spectrum produced by DRAGONS.

.. code-block:: python
    :linenos:

    import matplotlib.pyplot as plt
    import numpy as np

    import astrodata
    import gemini_instruments

    ad = astrodata.open('N20180526S1024_1D.fits')
    ad.info()

    data = ad[0].data
    wavelength = ad[0].wcs(np.arange(data.size)).astype(np.float32)
    units = ad[0].wcs.output_frame.unit[0]

    # add aperture number and location in the title.
    # check that plt.xlabel call.  Not sure it's right, it works though.
    plt.xlabel(units)
    plt.plot(wavelength, data)
    plt.show()


.. _plot_sensfunc:

Inspect the sensitivity function
================================
