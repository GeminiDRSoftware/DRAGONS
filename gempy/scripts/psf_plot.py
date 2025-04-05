#!/usr/bin/env python
'''
This utility plots the psf, radial profile and encircled energy profile
'''

import sys
import astrodata
import gemini_instruments
import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    filename = sys.argv[1]

    ad = astrodata.from_file(filename)
    objcat = ad[0].OBJCAT
    catx = objcat.field("X_IMAGE")
    caty = objcat.field("Y_IMAGE")
    catbg = objcat.field("BACKGROUND")
    cattotalflux = objcat.field("FLUX_AUTO")
    catmaxflux = objcat.field("FLUX_MAX")

    i = int(input('OBJCAT no: '))
    i -= 1
    xcenter = catx[i]
    ycenter = caty[i]
    bkg = catbg[i]
    totalflux = cattotalflux[i]
    maxflux = catmaxflux[i]

    print("X, Y:  %.2f, %.2f" % (xcenter, ycenter))

    xcenter = input('X: ')
    ycenter = input('Y: ')

    if xcenter == '':
        xcenter = catx[i]
    else:
        xcenter = float(xcenter)
    if ycenter == '':
        ycenter = caty[i]
    else:
        ycenter = float(ycenter)

    # co-ords in python are 0 based not 1 based
    xcenter -= 0.5
    ycenter -= 0.5
    print("xcenter, ycenter: %.2f, %.2f" % (xcenter, ycenter))

    size = 10
    stamp = ad[0].data[int(ycenter)-size:int(ycenter)+size,
                       int(xcenter)-size:int(xcenter)+size]

    # Print a small pixel table:
    #print("%8s %8d %8d %8d" %
    #      ('', int (xcenter-1), int(xcenter), int(xcenter+1)))
    #for y in range(int (ycenter)-1, int(ycenter)+2):
        #print("%8d %8f %8f %8f" % (y, ad[0].data[y,int (xcenter-1)],
        # ad[0].data[y,int (xcenter)], ad[0].data[y,int (xcenter+1)]))

    plt.figure(1)
    plt.subplot(1, 3, 1)
    plt.imshow(stamp-bkg, origin='lower',
               extent=[int(xcenter)-size, int(xcenter)+size,
                       int(ycenter)-size, int(ycenter)+size],
               interpolation='nearest')
    plt.axhline(y=ycenter)
    plt.axvline(x=xcenter)

    # Build the radial profile
    rpr = []
    rpv = []
    for y in range(int(ycenter)-size, int(ycenter)+size):
        for x in range(int(xcenter)-size, int(xcenter)+size):
            # Compute the distance of the center of this pixel from the
            # centroid location
            dx = (float(x)+0.5) - xcenter
            dy = (float(y)+0.5) - ycenter
            distance = math.sqrt(dx*dx + dy*dy)
            rpr.append(distance)
            # And record the flux above the background
            rpv.append(ad[0].data[y, x] - bkg)

    halfflux = maxflux / 2.0

    # Sort into radius value order
    sort = np.argsort(rpr)

    # Walk through the values and find the first point below the half flux and
    # the last point above the half flux preceeding 10 points below it...
    inner = None
    outer = None
    below = 0
    for i in range(len(rpr)):
        if rpv[sort[i]] < halfflux and inner is None:
            inner = rpr[sort[i]]
            print("inner: %.2f" % rpr[sort[i]])
        if below < 10 and rpv[sort[i]] > halfflux:
            outer = rpr[sort[i]]
            print("outer: %.2f" % rpr[sort[i]])
        if rpv[sort[i]] < halfflux:
            below += 1

    hwhm = (inner + outer) / 2.0

    print("HWHM: %.2f   FWHM: %.2f" % (hwhm, hwhm*2.0))

    plt.subplot(1, 3, 2)
    plt.scatter(rpr, rpv)
    plt.axhline(y=maxflux)
    plt.axhline(y=halfflux)
    plt.axhline(y=0)
    plt.axvline(x=hwhm)
    plt.grid(True)
    plt.xlim(0, 15)
    plt.xlabel('radius')
    plt.ylabel('counts')

    # Now make the radial profile into a 2d numpy array
    radial_profile = np.array([rpr, rpv], dtype=np.float32)
    sort = np.argsort(radial_profile[0])

    print("Total Flux = %.1f" % totalflux)

    halfflux = totalflux / 2.0

    # Now step out through the radial_profile until we get half the flux
    flux = 0
    i = 0
    while (flux < halfflux):
        #print("adding in r=%.2f v=%.1f" % (radial_profile[0][sort[i]],
        # radial_profile[1][sort[i]]-bkg))
        flux += radial_profile[1][sort[i]]
        i += 1

    ee50r = radial_profile[0][sort[i]]

    print("EE50 radius: %.2f       diameter: %.2f" % (ee50r, 2.0*ee50r))

    # Now step through to make an encircled energy plot
    eer = []
    eev = []
    flux = 0
    for i in range(len(sort)):
        eer.append(radial_profile[0][sort[i]])
        flux += radial_profile[1][sort[i]]
        eev.append(flux)

    # plot encircled energy
    plt.subplot(1, 3, 3)
    plt.scatter(eer, eev)
    plt.grid(True)
    plt.xlim(0, 15)
    plt.axhline(y=totalflux)
    plt.axhline(y=halfflux)
    plt.axvline(x=ee50r)
    plt.xlabel('radius')
    plt.ylabel('Enclircled Energy')

    plt.show()

if __name__ == '__main__':
    sys.exit(main())
