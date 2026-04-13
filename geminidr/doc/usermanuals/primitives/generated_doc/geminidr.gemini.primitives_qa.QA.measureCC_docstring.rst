
This primitive will determine the zeropoint by looking at sources in
the OBJCAT for which a reference catalog magnitude has been determined

It will also compare the measured zeropoint against the nominal
zeropoint for the instrument and the nominal atmospheric extinction
as a function of airmass, to compute the estimated cloud attenuation.

This function is for use with SExtractor-style source-detection.
It relies on having already added a reference catalog and done the
cross match to populate the refmag column of the objcat

The reference magnitudes (refmag) are straight from the reference
catalog. The measured magnitudes (mags) are straight from the object
detection catalog.

We correct for atmospheric extinction at the point where we
calculate the zeropoint, ie we define::

    actual_mag = zeropoint + instrumental_mag + extinction_correction

where in this case, actual_mag is the refmag, instrumental_mag is
the mag from the objcat, and we use the nominal extinction value as
we don't have a measured one at this point. ie  we're actually
computing zeropoint as::

    zeropoint = refmag - mag - nominal_extinction_correction

Then we can treat zeropoint as::

    zeropoint = nominal_photometric_zeropoint - cloud_extinction

to estimate the cloud extinction.

Parameters
----------
suffix : str
    suffix to be added to output files
