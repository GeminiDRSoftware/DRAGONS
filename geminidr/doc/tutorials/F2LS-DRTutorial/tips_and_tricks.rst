.. tips_and_tricks.rst

.. _tips_and_tricks:

***************
Tips and Tricks
***************

.. _arc_lampoff:

Using the Lamp-off Flat for the K-band Arc
==========================================

Show how to create the dataselect list with arc and lamp-off flats
and how to make the reduce call.  Explain that both flats will be
stacked and then subtracted off each arc (as if a dark).

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
