#!/usr/bin/env python
# This utility plots the psf, radial profile and encircled energy profile

import sys
from astrodata import AstroData
import scipy.ndimage
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

filename = sys.argv[1]

ad = AstroData(filename)
objcat = ad['OBJCAT']
catx = objcat.data.field("X_IMAGE")
caty = objcat.data.field("Y_IMAGE")
catfwhm = objcat.data.field("FWHM_IMAGE")
catbg = objcat.data.field("BACKGROUND")

sci = ad['SCI']

i = int(raw_input('OBJCAT no: '))
i -= 1
xc = catx[i]
yc = caty[i]
bg = catbg[i]

print "X, Y:  %.2f, %.2f" % (xc, yc)

xc = raw_input('X: ')
yc = raw_input('Y: ')

if xc == '':
  xc = catx[i]
else:
  xc = float(xc)
if yc == '':
  yc = caty[i]
else:
  yc = float(yc)

# co-ords in python are 0 based not 1 based
xc -= 0.5
yc -= 0.5
print "xc, yc: %.2f, %.2f" % (xc, yc)

size=10
stamp=sci.data[int(yc)-size:int(yc)+size,int(xc)-size:int(xc)+size]

# Print a small pixel table:
#print "%8s %8d %8d %8d" % ('', int (xc-1), int(xc), int(xc+1))
#for y in range(int (yc)-1, int(yc)+2):
    #print "%8d %8f %8f %8f" % (y, sci.data[y,int (xc-1)], sci.data[y,int (xc)], sci.data[y,int (xc+1)])
  
plt.figure(1)
plt.subplot(1, 3, 1)
plt.imshow(stamp-bg, origin='lower', extent=[int(xc)-size,int(xc)+size,int(yc)-size,int(yc)+size], interpolation='nearest')
plt.axhline(y=yc)
plt.axvline(x=xc)

# Build the radial profile
ctr_coord = np.mgrid[int(yc)-size:int(yc)+size,int(xc)-size:int(xc)+size] + 0.5
dist = np.array([ctr_coord[0]-yc,ctr_coord[1]-xc])
dist = np.sqrt(np.sum(dist**2,axis=0))
rpr = dist.flatten()
rpv = sci.data[int(yc)-size:int(yc)+size,int(xc)-size:int(xc)+size].flatten()

# Find the first point where the flux falls below half
radial_profile = np.rec.fromarrays([rpr,rpv-bg],names=["radius","flux"])
maxflux = np.max(radial_profile["flux"])
halfflux = maxflux/2.0
radial_profile.sort(order="radius")
first_halfflux = np.where(radial_profile["flux"]<=halfflux)[0][0]
hwhm = radial_profile["radius"][first_halfflux]

print "HWHM: %.2f   FWHM: %.2f" % (hwhm, hwhm*2.0)

plt.subplot(1,3,2)
plt.scatter(rpr, rpv)
plt.axhline(y=maxflux+bg)
plt.axhline(y=halfflux+bg)
plt.axhline(y=bg)
plt.axvline(x=hwhm)
plt.grid(True)
plt.xlim(0,15)
plt.xlabel('radius')
plt.ylabel('counts')

# OK, now calculate the total flux
totalflux = np.sum(radial_profile["flux"])
print "Total Flux = %.1f" % totalflux

# Find the first radius that encircles half the total flux
halfflux = totalflux / 2.0
sumflux = np.cumsum(radial_profile["flux"])
first_50pflux = np.where(sumflux>=halfflux)[0][0]
ee50r = radial_profile["radius"][first_50pflux]
print "EE50 radius: %.2f       diameter: %.2f" % (ee50r, 2.0*ee50r)

# Now make an encircled energy plot
eer=radial_profile["radius"]
eev=sumflux
plt.subplot(1,3,3)
plt.scatter(eer, eev)
plt.grid(True)
plt.xlim(0,15)
plt.axhline(y=totalflux)
plt.axhline(y=halfflux)
plt.axvline(x=ee50r)
plt.xlabel('radius')
plt.ylabel('Enclircled Energy')

plt.show()
