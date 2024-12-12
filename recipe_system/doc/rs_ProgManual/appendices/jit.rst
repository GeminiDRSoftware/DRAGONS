.. jit.rst

.. _jit:

JIT Calibration Request
***********************

It is important to understand that when a calibration request is made, "live"
metadata are passed to the calibration manager at the current stage of
processing. This kind of operation is called "just in time" (jit), which
indicates that one only requests a calibration at the processing stage where
and when it is needed.

This is necessary because the correct association of a processed calibration
product can actually depend on the processing history of the dataset up to the
point where the calibration is needed.

For example, data from a CCD comes with an overscan section.  The common
reduction steps involve correcting for the signal in the overscan and then
trimming that section off.  When the processing comes to requesting a
processed master bias, that bias must have also been corrected for the
overscan signal and match in size with the dataset being reduced, it must
also have been trimmed.  If the calibration request were to be made on the
raw CCD frame, before overscan correction and trimming, that processed bias
would not be found, or another one, a mismatched one could be.

This principle would work if for some reason the user decides not to subtract
the overscan signal.  Then a processed bias still containing the overscan
signal would be required.  (Note that in DRAGONS, a "processed" calibration
is expected to be ready to use, without additional processing.)

Therefore, the Recipe System uses JIT calibration requests.  The calibration
found will match the data at that point in the recipe where the calibration
is needed.
