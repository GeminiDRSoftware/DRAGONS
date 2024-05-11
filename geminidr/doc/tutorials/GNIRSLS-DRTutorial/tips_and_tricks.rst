.. tips_and_tricks.rst

.. _tips_and_tricks:

***************
Tips and Tricks
***************

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



.. _getBPM:

Get the BPMs
============

.. _skywavecal:

From Olesja:
I think this summarizes everything one can try if automatic line identification fails, especially for sky lines:
Try to reduce the fit order;
Try to reduce/increase min_snr;
Central wavelength of the observation may be off by a larger value than expected (the expected maximum shift is +/-10 nm for all instruments but GNIRS. For GNIRS it is +/-7% of the wavelength coverage). This often is the case for the commissioning data (SV programs), very low resolutions in GNIRS, or sometimes it just happens. Try different central_wavelength value, or do line identification manually.
Check if the correct linelist is used
Check if the extraction columns/rows are OK (especially when calibrating from sky lines). Use different columns/rows for 1d-spectrum extraction if necessary.