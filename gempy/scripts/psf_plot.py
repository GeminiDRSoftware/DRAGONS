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
rpr=[]
rpv=[]
for y in range(int(yc)-size, int(yc)+size):
  for x in range(int(xc)-size, int(xc)+size):
    # Compute the distance of the center of this pixel from the centroid location
    dx = (float(x)+0.5) - xc
    dy = (float(y)+0.5) - yc
    d = math.sqrt(dx*dx + dy*dy)
    rpr.append(d)
    # And record the flux above the background
    rpv.append(sci.data[y, x] - bg)

maxflux = np.max(rpv)
halfflux = maxflux/2.0

# Sort into flux value order
sort = np.argsort(rpv)

# Walk through the flux values and find the index in the sort array of the point closest to half flux.
halfindex = None
best_d = maxflux
for i in range(len(rpv)):
  d = abs(rpv[sort[i]] - halfflux)
  if(d<best_d):
    best_d = d
    halfindex = i

# Find the average radius of the num_either_side=3 points either side of that in flux.
num_either_side=5
sum = 0
for i in range(halfindex-num_either_side, halfindex+num_either_side+1):
  sum += rpr[sort[i]]
hwhm = sum / float(num_either_side*2+1)

print "HWHM: %.2f   FWHM: %.2f" % (hwhm, hwhm*2.0)

plt.subplot(1,3,2)
plt.scatter(rpr, rpv)
plt.axhline(y=maxflux)
plt.axhline(y=halfflux)
plt.axhline(y=0)
plt.axvline(x=hwhm)
plt.grid(True)
plt.xlim(0,15)
plt.xlabel('radius')
plt.ylabel('counts')

# OK, now calculate the total flux
bgsub = stamp - bg
totalflux = np.sum(bgsub)

# Now make the radial profile into a 2d numpy array
rp = np.array([rpr, rpv], dtype=np.float32)
sort = np.argsort(rp[0])

print "Total Flux = %.1f" % totalflux

halfflux = totalflux / 2.0

# Now step out through the rp until we get half the flux
flux=0
i=0
while (flux < halfflux):
  #print "adding in r=%.2f v=%.1f" % (rp[0][sort[i]], rp[1][sort[i]]-bg)
  flux+= rp[1][sort[i]]
  i+=1

ee50r = rp[0][sort[i]]

print "EE50 radius: %.2f       diameter: %.2f" % (ee50r, 2.0*ee50r)

# Now step through to make an encircled energy plot
eer=[]
eev=[]
flux=0
for i in range(len(sort)):
  eer.append(rp[0][sort[i]])
  flux += rp[1][sort[i]]
  eev.append(flux)

# plot encircled energy
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
