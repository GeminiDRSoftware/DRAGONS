.. tips_and_tricks.rst

.. _tips_and_tricks:

***************
Tips and Tricks
***************

.. _arc_lampoff:

Using the Lamp-off Flat for the K-band Arc
==========================================

There is continuum flux in the K-band arcs.  As part of the standard
calibrations for R3K + K-band filter observations, a lamp-off flat with
an exposure time equal to the arc's exposure time is taken right before
or after the arc.  This lamp-off flat is needed when reducing the arc
with Gemini IRAF.   It is **not** needed with DRAGONS.

Nevertheless, DRAGONS can use it if you think it is necessary.  Here's how.

::

    dataselect ../playdata/example3/*.fits --tags ARC -o arc.lis
    dataselect ../playdata/example3/*.fits --tags FLAT,LAMPOFF -o lampoff.lis

    reduce @arc.lis @lampoff.lis

The lamp-off flat(s), instead of a dark, will be subtracted from the arc.


.. _wavzero:

Adjusting the Wavelength Zeropoint
==================================

Following the wavelength calibration, the default recipe has an optional
step to adjust the wavelength zero point using the sky lines.  By default,
this step will NOT make any adjustment.  We found that in general, the
adjustment is so small as being in the noise.  If you wish to make an
adjustment, in pixels, use the ``shift`` parameter.  A value of 0 is the default and
applies no shift.  The parameter can be set to a value set my you, eg.
``-p adjustWavelengthZeroPoint:shift=1.3``.  Or, you can let the software
measure the shift for you by setting ``shift`` to ``None``.  This will trigger
the algorithm that tries to calculate the shift on it's own.


Inspect the sensitivity function
================================
The sensitivity function is stored in the processed telluric star file.
To inspect the sensitivity function, you can use the following Python code.

.. code-block:: python
    :linenos:

    import numpy as np
    import matplotlib.pyplot as plt

    import astrodata
    import gemini_instruments

    from gempy.library import astromodels as am

    ad = astrodata.open('N20210407S0188_telluric.fits')
    sensfunc = am.table_to_model(ad[0].SENSFUNC)
    w = ad[0].wcs(np.arange(ad[0].data.size))

    std_wave_unit = ad[0].SENSFUNC['knots'].unit
    std_flux_unit = ad[0].SENSFUNC['coefficients'].unit

    plt.xlabel(f'Wavelength ({std_wave_unit})')
    plt.ylabel(f'{std_flux_unit}')
    plt.plot(w, sensfunc(w))
    plt.show()


Useful parameters
=================

skip_primitive
--------------
I might happen that you will want or need to not run a primitive in a recipe.
You could copy the recipe over and edit it.  Or you could invoke the
``skip_primitive`` parameter to tell DRAGONS to completely skip that step.

Let's say that you want the data aligned but not stacked.  You would do::

    reduce @sci.lis -p stackFrames:skip_primitive=True


write_outputs
-------------
When debugging or when there's a need to inspect intermediate products, you
might want to write the output of a specific primitive to disk.  This is done
with the ``write_outputs`` parameter.

For example, to write the extracted spectrum before it is corrected for
telluric features and flux calibrated, you would do::

    reduce @sci.lis -p extractSpectra:write_outputs=True
