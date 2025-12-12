.. issues_and_limitations.rst

.. _issues_and_limitations:

**********************
Issues and Limitations
**********************

.. _why_darks:

Why are darks used
==================


Show what a dark looks like to convey why a dark correction is required.
Since no lampoff for flats, we need to dark correct.  Line detection in
arc lamps would be compromised with those dark lines still in.  And
for science/telluric, the dark current/features are known to be unstable
so extra steps can help.  (But I need to try without.)

.. _badwcs:

Recognizing and fixing bad WCS
==============================
The WCS in Flamingos 2 data is commonly not quite right.  There is a step
in the ``prepare`` primitive that tries to identify issues and raise an
error.  Unfortunately, for Flamingos 2, the discrepancies are not
always detected by the current algorithm.

There is a way for the user to detect the problem however.  It is by
inspecting the spatial profile with the ``findApertures`` interactive tool.
There are indications in the logs before that step, but the visual cue is
easier to spot.

When the WCS of the raw data are wrong, the profile in ``findApertures`` will
look like this, one positive, one negative:

<screenshot of bad WCS profile>

When the WCS is correct or has been fixed, the standard AB dither pattern
should lead to a negative-positive-negative profile, like this:

<screenshot of good WCS profile>

When you run the telluric or the science reduction on new data, run
``findApertures`` in interactive mode (``-p findApertures:interactive=True`` or
``-p interactive=True`` to turn everything interactive).  If you see the
"bad WCS" profile, abort and re-run the reduction with
``-p prepare:bad_wcs=new``.  For example::

    reduce @sci.lis -p prepare:bad_wcs=new interactive=True

That should lead to the "good WCS" profile.  If not, the problem is elsewhere.


