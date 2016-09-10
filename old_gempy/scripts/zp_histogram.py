#!/usr/bin/env python
# This tool plots some useful plots for seeing what's going on with zeropoint
# estimates from the QAP.
#
# This was developed as a quick analysis tool to asess QAP ZP  and CC numbers
# and to understand comparisons with certain people's skygazing results, but I'm sure
# it would be useful for the DA/SOSs.
# It can be run on any file with a sextractor style OBJCAT in it, for example
# the _forStack files output by the QAP. If no REFCAT is present, it bails out as no refmags
#
# Paul Hirst 20120321
import math
import sys
from astrodata import AstroData
import numpy as np
import matplotlib.pyplot as plt

from random import random

filename = sys.argv[1]

ad = AstroData(filename)
objcat = ad['OBJCAT']

if (ad['REFCAT'] is None):
  print "No Reference Catalog in this file, thus no Zeropoints. Sorry"
  sys.exit(0)

mag = objcat.data.field("MAG_AUTO")
magerr = objcat.data.field("MAGERR_AUTO")
refmag = objcat.data.field("REF_MAG")
refmagerr = objcat.data.field("REF_MAG_ERR")
sxflag = objcat.data.field("FLAGS")
dqflag = objcat.data.field("IMAFLAGS_ISO")

# set mag to None where we don't want to use the object
mag = np.where((mag==-999), None, mag)
mag = np.where((mag==99), None, mag)
mag = np.where((refmag==-999), None, mag)
mag = np.where((np.isnan(refmag)), None, mag)
mag = np.where((sxflag==0), mag, None)
mag = np.where((dqflag==0), mag, None)

# Now ditch the values out of the arrays where mag is None
# NB do mag last
magerr = magerr[np.flatnonzero(mag)]
refmag = refmag[np.flatnonzero(mag)]
refmagerr = refmagerr[np.flatnonzero(mag)]
mag = mag[np.flatnonzero(mag)]

if(len(mag) == 0):
  print "No good sources to plot"
  sys.exit(1)

# Now apply the exposure time and nom_at_ext corrections to mag
et = float(ad.exposure_time())
if 'GMOS_NODANDSHUFFLE' in ad.types:
    print "Imaging Nod-And-Shuffle. Photometry may be dubious"
    et /= 2.0

etmag = 2.5*math.log10(et)
nom_at_ext = float(ad.nominal_atmospheric_extinction())

mag += etmag
mag += nom_at_ext

# Can now calculate the zp array
zp = refmag - mag
zperr = np.sqrt(refmagerr*refmagerr + magerr*magerr)

# Trim values out of zp where the zeropoint error is > 0.1
zp_trim = np.where((zperr<0.1), zp, None)
zperr_trim = zperr[np.flatnonzero(zp_trim)]
refmag_trim = refmag[np.flatnonzero(zp_trim)]
refmagerr_trim = refmagerr[np.flatnonzero(zp_trim)]
zp_trim = zp[np.flatnonzero(zp_trim)]

nzp = float(ad.nominal_photometric_zeropoint())

plt.figure(1)

# Plot the mag-mag plot
plt.subplot(2,2,1)
plt.scatter(refmag, mag)
plt.errorbar(refmag, mag, xerr=refmagerr, yerr=magerr, fmt=None)
plt.xlabel('Reference Magnitude')
plt.ylabel('Instrumental Magnitdute')

# Plot the mag - zp plot
plt.subplot(2,2,2)
plt.scatter(refmag, zp)
plt.errorbar(refmag, zp, xerr=refmagerr, yerr=zperr, fmt=None)
plt.scatter(refmag_trim, zp_trim, color='g')
plt.errorbar(refmag_trim, zp_trim, xerr=refmagerr_trim, yerr=zperr_trim, c='g', fmt=None)
plt.axhline(y=nzp)
plt.xlabel('Reference Magnitude')
plt.ylabel('Zeropoint')

# Plot the zp histogram
plt.subplot(2,2,3)
plt.hist(zp, bins=40)
plt.hist(zp_trim, bins=40, range=(zp.min(), zp.max()))
plt.axvline(x=nzp)
plt.xlabel('Zeropoint')
plt.ylabel('Number')

# Now plot in CC extinction space
zp -= nzp
zp_trim -= nzp
zp *=-1
zp_trim *= -1
plt.subplot(2,2,4)
plt.hist(zp, bins=40)
plt.hist(zp_trim, bins=40, range=(zp.min(), zp.max()))
plt.xlabel('Cloud Extinction')
plt.ylabel('Number')


plt.show()
