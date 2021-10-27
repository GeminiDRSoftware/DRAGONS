.. 05_tips_and_tricks.rst

.. _tips_and_tricks:

***************
Tips and Tricks
***************

.. _plot_1d:

Plot a 1-D spectrum
===================
Here is how to plot an extracted spectrum produced by DRAGONS with Python and matplotlib.

.. code-block:: python
    :linenos:

    import matplotlib.pyplot as plt
    import numpy as np

    import astrodata
    import gemini_instruments

    ad = astrodata.open('S20171022S0087_1D.fits')
    ad.info()

    data = ad[0].data
    wavelength = ad[0].wcs(np.arange(data.size)).astype(np.float32)
    units = ad[0].wcs.output_frame.unit[0]

    # add aperture number and location in the title.
    # check that plt.xlabel call.  Not sure it's right, it works though.
    plt.xlabel(f'Wavelength ({units})')
    plt.ylabel(f'Signal ({ad[0].hdr["BUNIT"]})')
    plt.plot(wavelength, data)
    plt.show()


.. _plot_sensfunc:

Inspect the sensitivity function
================================
Plotting the sensitivity function is not obvious.  Using Python, here's a way to
do it.

.. code-block:: python
    :linenos:

    from scipy.interpolate import BSpline
    import numpy as np
    import matplotlib.pyplot as plt

    import astrodata
    import gemini_instruments

    ad = astrodata.open('S20170826S0160_ql_standard.fits')

    sensfunc = ad[0].SENSFUNC

    order = sensfunc.meta['header'].get('ORDER', 3)
    func = BSpline(sensfunc['knots'].data, sensfunc['coefficients'].data, order)
    std_wave_unit = sensfunc['knots'].unit
    std_flux_unit = sensfunc['coefficients'].unit

    w1 = ad[0].wcs(0)
    w2 = ad[0].wcs(ad[0].data.size)

    x = np.arange(w1, w2)
    plt.xlabel(f'Wavelength ({std_wave_unit})')
    plt.ylabel(f'{std_flux_unit}')
    plt.plot(x, func(x))
    plt.show()

In the science-approved version of the GMOS longslit support in DRAGONS, there
will be an interactive tool to inspect and adjust the sensitivity function.



