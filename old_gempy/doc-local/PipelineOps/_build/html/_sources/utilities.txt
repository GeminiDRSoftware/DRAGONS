.. utilities:

*********
Utilities
*********

There are a few utility scripts that might be of interest.  Those were developed mostly for the pipeline team
as part of performance analysis.  Since there is no reason for hiding them and since they are installed on
the server anyway for anyone to use, we describe them here.

fwhm_histogram
==============
Plots the histogram of FWHMs from the object catalog (OBJCAT) in a reduced
data FITS file. Plots general sources, point sources and sources selected
for IQ measurement in different colors.

Usage - Simply run it on any MEF file that contains an OBJCAT table: ::

   fwhm_histogram N20010203S0456_forStack.fits


zp_histogram
============
Plots the histogram of Zero Points from the object catalog (OBJCAT) in a
reduced data FITS file. The data must have reference magnitudes (ie. a
REFCAT table) present.

Usage - Simply run it on any MEF file that contains an OBJCAT and a REFCAT tables: ::

   zp_histogram N20010203S0456_forStack.fits


psf_plot
========
Shows a source image, radial profile and encircled energy profile for a
given source detected in a pipeline-reduced image.

Usage - Simply run it on any MEF file that contains an OBJCAT table: ::

   psf_plot N20010203S0456_forStack.fits

The script will prompt for the OBJCAT ID of the source you wish to plot,
and give the opportunity to enter your own source center co-ordinates
(just hit return to use the ones from the objcat).
