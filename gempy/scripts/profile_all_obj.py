import sys
import datetime
import math
import numpy as np
from astrodata import AstroData

def profile_numpy(data,xc,yc,bg,stamp_size=10):
    
    # Check that there's enough room for a stamp
    sz = stamp_size
    if (int(yc)-sz<0 or int(xc)-sz<0 or
        int(yc)+sz>=data.shape[0] or int(xc)+sz>=data.shape[1]):
        return (None,None)

    # Get image stamp around center point
    stamp=data[int(yc)-sz:int(yc)+sz,int(xc)-sz:int(xc)+sz]

    # Get an array of the coordinates of the centers of all the pixels 
    # in the stamp
    dist = np.mgrid[int(yc)-sz:int(yc)+sz,int(xc)-sz:int(xc)+sz] + 0.5

    # Subtract the center coordinates
    dist[0] -= yc
    dist[1] -= xc
    
    # Square root of the sum of the squares of the distances
    dist = np.sqrt(np.sum(dist**2,axis=0))

    # Radius and flux arrays for the radial profile
    rpr = dist.flatten()
    rpv = stamp.flatten() - bg
    
    # Sort by the flux
    sort_order = np.argsort(rpv) 
    radius = rpr[sort_order]
    flux = rpv[sort_order]

    # Find the distance (in flux) of each point from the half-flux
    maxflux = flux[-1]
    halfflux = maxflux/2.0
    flux_dist = np.abs(flux - halfflux)

    # Find the point that is closest to the half-flux
    closest_ind = np.argmin(flux_dist)

    # Average the radius of this point with the five points higher and lower
    # in flux
    num_either_side = 5
    min_pt = closest_ind-num_either_side
    max_pt = closest_ind+num_either_side+1
    if min_pt<0:
        min_pt = 0
    if max_pt>radius.size:
        max_pt = radius.size
    nearest_pts = radius[min_pt:max_pt]
    hwhm = np.mean(nearest_pts)


    # Resort by radius
    sort_order = np.argsort(rpr) 
    radius = rpr[sort_order]
    flux = rpv[sort_order]

    # Find the first radius that encircles half the total flux
    sumflux = np.cumsum(flux)
    totalflux = sumflux[-1]
    halfflux = totalflux / 2.0
    first_50pflux = np.where(sumflux>=halfflux)[0]
    if first_50pflux.size<=0:
        ee50r = radius[-1]
    else:
        ee50r = radius[first_50pflux[0]]

    return (hwhm, ee50r)

def profile_loop(data,xc,yc,bg,sz):

    # Check that there's enough room for a stamp
    if (int(yc)-sz<0 or int(xc)-sz<0 or
        int(yc)+sz>=data.shape[0] or int(xc)+sz>=data.shape[1]):
        return (None,None)

    stamp=data[int(yc)-sz:int(yc)+sz,int(xc)-sz:int(xc)+sz]

    # Build the radial profile
    rpr=[]
    rpv=[]
    for y in range(int(yc)-sz, int(yc)+sz):
        for x in range(int(xc)-sz, int(xc)+sz):
            # Compute the distance of the center of this pixel 
            # from the centroid location
            dx = (float(x)+0.5) - xc
            dy = (float(y)+0.5) - yc
            d = math.sqrt(dx*dx + dy*dy)
            rpr.append(d)
            rpv.append(data[y, x] - bg)

    maxflux = np.max(rpv)
    halfflux = maxflux/2.0

    # Sort into flux value order
    sort = np.argsort(rpv)

    # Walk through the flux values and find the index in the
    # sort array of the point closest to half flux.
    halfindex = None
    best_d = maxflux
    for i in range(len(rpv)):
        d = abs(rpv[sort[i]] - halfflux)
        if(d<best_d):
            best_d = d
            halfindex = i

    # Find the average radius of the num_either_side=3 points
    # either side of that in flux.
    num_either_side=5
    sum = 0
    num = 0
    nearest_pts = []
    for i in range(halfindex-num_either_side, halfindex+num_either_side+1):
        if i<0 or i>=len(sort):
            continue
        sum += rpr[sort[i]]
        num += 1
        nearest_pts.append(rpr[sort[i]])
    hwhm = sum / num

    # OK, now calculate the total flux
    bgsub = stamp - bg
    totalflux = np.sum(bgsub)

    # Now make the radial profile into a 2d numpy array
    rp = np.array([rpr, rpv], dtype=np.float32)
    sort = np.argsort(rp[0])

    halfflux = totalflux / 2.0

    # Now step out through the rp until we get half the flux
    flux=0
    i=0
    while (flux < halfflux):
      #print "adding in r=%.2f v=%.1f" % (rp[0][sort[i]], rp[1][sort[i]]-bg)
      flux+= rp[1][sort[i]]
      i+=1

    # subtract 1 from the index -- 1 gets added after the right flux is found
    if i>0:
        i-=1
    ee50r = rp[0][sort[i]]

    return (hwhm, ee50r)

filename = sys.argv[1]
ad = AstroData(filename)
objcat = ad['OBJCAT',1]
sci = ad['SCI',1]
catx = objcat.data.field("X_IMAGE")
caty = objcat.data.field("Y_IMAGE")
catfwhm = objcat.data.field("FWHM_IMAGE")
catbg = objcat.data.field("BACKGROUND")
data = sci.data

nobj = len(catx)
print "%d objects" % nobj

# the numpy way
print 'numpy'
now = datetime.datetime.now()
hwhm_list = []
e50r_list = []
for i in range(0,len(objcat.data)):
    xc = catx[i]
    yc = caty[i]
    bg = catbg[i]

    xc -= 0.5
    yc -= 0.5

    hwhm,e50r = profile_numpy(data,xc,yc,bg)
    #print i,hwhm,e50r
    if (hwhm is not None and e50r is not None):
        hwhm_list.append(hwhm)
        e50r_list.append(e50r)
print "  mean HWHM %.2f" % np.mean(hwhm_list)
print "  mean E50R %.2f" % np.mean(e50r_list)
elap = datetime.datetime.now() - now
print "  %.2f s" % ((elap.seconds*10**6 + elap.microseconds)/10.**6)

# the loopy way
print 'loopy'
now = datetime.datetime.now()
hwhm_list = []
e50r_list = []
for i in range(0,len(objcat.data)):
    xc = catx[i]
    yc = caty[i]
    bg = catbg[i]

    xc -= 0.5
    yc -= 0.5

    sz=10

    hwhm,e50r = profile_loop(data,xc,yc,bg,sz)
    #print i,hwhm,e50r
    if (hwhm is not None and e50r is not None):
        hwhm_list.append(hwhm)
        e50r_list.append(e50r)
print "  mean HWHM %.2f" % np.mean(hwhm_list)
print "  mean E50R %.2f" % np.mean(e50r_list)
elap = datetime.datetime.now() - now
print "  %.2f s" % ((elap.seconds*10**6 + elap.microseconds)/10.**6)
