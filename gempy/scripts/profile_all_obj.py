#!/usr/bin/env python


import sys
import datetime
import math
import numpy as np
import astrodata
import gemini_instruments

def profile_numpy(data, xcenter, ycenter, bkg, total_flux, max_flux,
                  stamp_size=10):

    # Check that there's enough room for a stamp
    size = stamp_size
    if (int(ycenter)-size < 0 or int(xcenter)-size < 0 or
          int(ycenter)+size >= data.shape[0] or
          int(xcenter)+size >= data.shape[1]):
        return (None, None)

    # Get image stamp around center point
    stamp = data[int(ycenter)-size:int(ycenter)+size,
                 int(xcenter)-size:int(xcenter)+size]

    # Get an array of the coordinates of the centers of all the pixels
    # in the stamp
    dist = np.mgrid[int(ycenter)-size:int(ycenter)+size,
                    int(xcenter)-size:int(xcenter)+size] + 0.5

    # Subtract the center coordinates
    dist[0] -= ycenter
    dist[1] -= xcenter

    # Square root of the sum of the squares of the distances
    dist = np.sqrt(np.sum(dist**2, axis=0))

    # Radius and flux arrays for the radial profile
    rpr = dist.flatten()
    rpv = stamp.flatten() - bkg

    # Sort by the radius
    sort_order = np.argsort(rpr)
    radius = rpr[sort_order]
    flux = rpv[sort_order]

    # Find the first 10 points below the half-flux
    halfflux = max_flux / 2.0
    below = np.where(flux <= halfflux)[0]
    if below.size > 0:
        if len(below) >= 10:
            first_below = below[0:10]
        else:
            first_below = below
        inner = radius[first_below[0]]
        if first_below[0] > 0:
            minimum = first_below[0] - 1
        else:
            minimum = first_below[0]
        nearest_r = radius[minimum:first_below[-1]]
        nearest_f = flux[minimum:first_below[-1]]
        possible_outer = nearest_r[nearest_f >= halfflux]
        if possible_outer.size > 0:
            outer = np.max(possible_outer)
            hwhm = (inner + outer) / 2.0
            print(inner, outer)
        else:
            hwhm = None
    else:
        hwhm = None

    # Find the first radius that encircles half the total flux
    sumflux = np.cumsum(flux)
    halfflux = total_flux / 2.0
    first_50pflux = np.where(sumflux >= halfflux)[0]
    if first_50pflux.size > 0:
        ee50r = radius[first_50pflux[0]]
    else:
        ee50r = None

    return (hwhm, ee50r)

def profile_loop(data, xcenter, ycenter, bkg, total_flux, max_flux, size):

    # Check that there's enough room for a stamp
    if (int(ycenter)-size < 0 or int(xcenter)-size < 0 or
          int(ycenter)+size >= data.shape[0] or
          int(xcenter)+size >= data.shape[1]):

        return (None, None)

    # Build the radial profile
    rpr = []
    rpv = []
    for y in range(int(ycenter)-size, int(ycenter)+size):
        for x in range(int(xcenter)-size, int(xcenter)+size):
            # Compute the distance of the center of this pixel
            # from the centroid location
            dx = (float(x) + 0.5) - xcenter
            dy = (float(y) + 0.5) - ycenter
            distance = math.sqrt(dx*dx + dy*dy)
            rpr.append(distance)
            rpv.append(data[y, x] - bkg)

    halfflux = max_flux / 2.0

    # Sort into radius value order
    sort = np.argsort(rpr)

    # Walk through the values and find the first point below the half flux and
    # the last point above the half flux preceeding 10 points below it...
    inner = None
    outer = None
    below = 0
    for i in range(len(rpr)):
        if rpv[sort[i]] <= halfflux and inner is None:
            inner = rpr[sort[i]]
            #print("inner: %.2f" % rpr[sort[i]], i)

        if below < 10 and rpv[sort[i]] >= halfflux:
            outer = rpr[sort[i]]
            #print("outer: %.2f" % rpr[sort[i]])
        if rpv[sort[i]] < halfflux:
            below += 1
        if below > 10:
            break

    if inner is None or outer is None:
        hwhm = None
    else:
        print(inner, outer)
        hwhm = (inner + outer) / 2.0

    # Now make the radial profile into a 2d numpy array
    radial_profile = np.array([rpr, rpv], dtype=np.float32)
    sort = np.argsort(radial_profile[0])

    halfflux = total_flux / 2.0

    # Now step out through the radial_profile until we get half the flux
    flux = 0
    i = 0
    while flux < halfflux and i < len(rpr):
        #print("adding in r=%.2f v=%.1f" %
        # (radial_profile[0][sort[i]], radial_profile[1][sort[i]]-bg))
        flux += radial_profile[1][sort[i]]
        i += 1

    # subtract 1 from the index -- 1 gets added after the right flux is found
    if i > 0:
        i -= 1
    ee50r = radial_profile[0][sort[i]]

    return (hwhm, ee50r)

def main():
    filename = sys.argv[1]
    ad = astrodata.from_file(filename)
    objcat = ad[0].OBJCAT
    data = ad[0].data
    catx = objcat.field("X_IMAGE")
    caty = objcat.field("Y_IMAGE")
    catbg = objcat.field("BACKGROUND")
    cattotalflux = objcat.field("FLUX_AUTO")
    catmaxflux = objcat.field("FLUX_MAX")

    nobj = len(catx)
    print("%d objects" % nobj)

    # the numpy way
    print('numpy')
    now = datetime.datetime.now()
    hwhm_list = []
    e50r_list = []
    for i in range(0, len(objcat)):
        if i > 20:
            break
        xcenter = catx[i]
        ycenter = caty[i]
        bkg = catbg[i]
        total_flux = cattotalflux[i]
        max_flux = catmaxflux[i]

        xcenter -= 0.5
        ycenter -= 0.5

        hwhm, e50r = profile_numpy(data, xcenter, ycenter, bkg,
                                   total_flux, max_flux)
        #print(i,hwhm,e50r)
        if hwhm is not None and e50r is not None:
            hwhm_list.append(hwhm)
            e50r_list.append(e50r)
    print("  mean HWHM %.2f" % np.mean(hwhm_list))
    print("  mean E50R %.2f" % np.mean(e50r_list))
    elap = datetime.datetime.now() - now
    print("  %.2f s" % ((elap.seconds * 10**6 + elap.microseconds)/10.**6))

    # the loopy way
    print('loopy')
    now = datetime.datetime.now()
    hwhm_list = []
    e50r_list = []
    for i in range(0, len(objcat)):
        if i > 20:
            break
        xcenter = catx[i]
        ycenter = caty[i]
        bkg = catbg[i]
        total_flux = cattotalflux[i]
        max_flux = catmaxflux[i]

        xcenter -= 0.5
        ycenter -= 0.5

        size = 10

        hwhm, e50r = profile_loop(data, xcenter, ycenter, bkg,
                                  total_flux, max_flux, size)
        #print(i,hwhm,e50r)
        if hwhm is not None and e50r is not None:
            hwhm_list.append(hwhm)
            e50r_list.append(e50r)

    print("  mean HWHM %.2f" % np.mean(hwhm_list))
    print("  mean E50R %.2f" % np.mean(e50r_list))
    elap = datetime.datetime.now() - now
    print("  %.2f s" % ((elap.seconds*10**6 + elap.microseconds)/10.**6))


if __name__ == '__main__':
    sys.exit(main())
