# The astroTools module contains astronomy specific utility functions

import os
import re
import math
import numpy as np
import scipy.optimize as opt
import pandas as pd
#import matplotlib.pyplot as plt

import warnings

def rasextodec(string):
    """
    Convert hh:mm:ss.sss to decimal degrees
    """
    m = re.match("(\d+):(\d+):(\d+\.\d+)", string)
    if m:
        hours = float(m.group(1))
        minutes = float(m.group(2))
        secs = float(m.group(3))
        
        minutes += (secs/60.0)
        hours += (minutes/60.0)
        
        degrees = hours * 15.0
    
    return degrees

def degsextodec(string):
    """
    Convert [-]dd:mm:ss.sss to decimal degrees
    """
    m = re.match("(-*)(\d+):(\d+):(\d+\.\d+)", string)
    if m:
        sign = m.group(1)
        if sign == '-':
            sign = -1.0
        else:
            sign = +1.0
        
        degs = float(m.group(2))
        minutes = float(m.group(3))
        secs = float(m.group(4))
        
        minutes += (secs/60.0)
        degs += (minutes/60.0)
        
        degs *= sign
    
    return degs

class GaussFit:
    """
    This class provides access to a Gaussian model, intended
    to be fit to a small stamp of data, via a  minimization of the 
    differences between the model and the data.

    Example usage:
    pars = (bg, peak, x_ctr, y_ctr, x_width, y_width, theta)
    gf = GaussFit(stamp_data)
    new_pars, success = scipy.optimize.leastsq(gf.calcDiff, pars, maxfev=1000)
    """


    def __init__(self, stamp_data):
        """
        This instantiates the fitting object.
        
        :param stamp_data: array containing image data, preferably the
                           source to fit plus a little padding
        :type stamp_data: NumPy array
        """
        self.stamp = stamp_data

    def model_gauss_1d(self, pars):
        """
        This function returns a Gaussian source in an 1D array the
        same shape as the stamp_data.  The Gaussian is determined by
        the parameters in pars.
        
        :param pars: Gaussian parameters in this order: background, peak,
                     center, width
        :type pars: 4-element tuple
        """
        bg, peak, c, w = pars
        
        model_fn = lambda x: bg + peak*np.exp((-((x-c)/w)**2)/2)
        gauss_array = np.fromfunction(model_fn, self.stamp.shape)
        return gauss_array

    def model_gauss_2d(self, pars):
        """
        This function returns a Gaussian source in an image array the
        same shape as the stamp_data.  The Gaussian is determined by
        the parameters in pars.
        
        :param pars: Gaussian parameters in this order: background, peak,
                     x-center, y-center, x-width, y-width, position angle
                     (in degrees)
        :type pars: 7-element tuple
        """
        bg, peak, cx, cy, wx, wy, theta = pars
        
        model_fn = lambda y,x: bg + peak*np.exp(-(   ((  (x-cx)*np.cos(theta)
                                                       - (y-cy)*np.sin(theta))
                                                       /wx)**2
                                                   + ((  (x-cx)*np.sin(theta)
                                                       + (y-cy)*np.cos(theta))
                                                       /wy)**2)
                                                 /2)
        gauss_array = np.fromfunction(model_fn, self.stamp.shape)
        return gauss_array

    def calc_diff(self, pars):
        """
        This function returns an array of the differences between
        the model stamp and the data stamp.  It is intended to be fed
        to an optimization algorithm, such as scipy.optimize.leastsq.

        :param pars: Gaussian parameters in this order: background, peak,
                     x-center, y-center, x-width, y-width, position angle
                     (in degrees)
        :type pars: 7-element tuple
        """
        if len(self.stamp.shape)==1:
            model = self.model_gauss_1d(pars).flatten()
        else:
            model = self.model_gauss_2d(pars).flatten()
        diff = self.stamp.flatten() - model
        return diff

class MoffatFit:
    """
    This class provides access to a Moffat model, intended
    to be fit to a small stamp of data, via a minimization of the 
    differences between the model and the data.

    Example usage:
    pars = (bg, peak, x_ctr, y_ctr, x_width, y_width, theta, beta)
    mf = MoffatFit(stamp_data)
    new_pars, success = scipy.optimize.leastsq(mf.calcDiff, pars, maxfev=1000)
    """


    def __init__(self, stamp_data):
        """
        This instantiates the fitting object.
        
        :param stamp_data: array containing image data, preferably the
                           source to fit plus a little padding
        :type stamp_data: NumPy array
        """
        self.stamp = stamp_data

    def model_moffat_2d(self, pars):
        """
        This function returns a Moffat source in an image array the
        same shape as the stamp_data.  The Moffat model is determined by
        the parameters in pars.
        
        :param pars: Moffat parameters in this order: background, peak,
                     x-center, y-center, x-width, y-width, position angle
                     (in degrees), beta
        :type pars: 8-element tuple
        """
        bg, peak, cx, cy, wx, wy, theta, beta = pars
        
        model_fn = lambda y,x: bg + peak*(1 + (((   (x-cx)*np.cos(theta)
                                                  - (y-cy)*np.sin(theta))
                                                 /wx)**2
                                             + ((   (x-cx)*np.sin(theta)
                                                  + (y-cy)*np.cos(theta))
                                                 /wy)**2)
                                          )**(-beta)

        moffat_array = np.fromfunction(model_fn, self.stamp.shape)
        return moffat_array

    def calc_diff(self, pars):
        """
        This function returns an array of the differences between
        the model stamp and the data stamp.  It is intended to be fed
        to an optimization algorithm, such as scipy.optimize.leastsq.

        :param pars: Moffat parameters in this order: background, peak,
                     x-center, y-center, x-width, y-width, position angle
                     (in degrees), beta
        :type pars: 8-element tuple
        """
        model = self.model_moffat_2d(pars).flatten()
        diff = self.stamp.flatten() - model
        return diff


def get_corners(shape):
    """
    This is a recursive function to calculate the corner indices 
    of an array of the specified shape.

    :param shape: length of the dimensions of the array
    :type shape: tuple of ints, one for each dimension

    """
    if not type(shape)==tuple:
        raise Errors.TypeError('get_corners argument is non-tuple')

    if len(shape)==1:
        corners = [(0,), (shape[0]-1,)]
    else:
        shape_less1 = shape[1:len(shape)]
        corners_less1 = get_corners(shape_less1)
        corners = []
        for corner in corners_less1:
            newcorner = (0,) + corner
            corners.append(newcorner)
            newcorner = (shape[0]-1,) + corner
            corners.append(newcorner)
        
    return corners

def rotate_2d(degs):
    """
    Little helper function to return a basic 2-D rotation matrix.

    :param degs: rotation amount, in degrees
    :type degs: float
    """
    rads = np.radians(degs)
    s = np.sin(rads)
    c = np.cos(rads)
    return np.array([[c,-s],
                     [s,c]])
class WCSTweak:
    """
    This class allows slight tweaking of an image's WCS, to fit to a
    reference WCS, via a minimization of the differences between
    reference points in both images.

    Example usage:
    wcstweak = WCSTweak(inp_wcs, inp_xy, ref_radec)
    pars = [0,0]
    new_pars,success = scipy.optimize.leastsq(wcstweak.calc_diff, pars,
                                              maxfev=1000)
    """

    def __init__(self, wcs, inp, ref, rotate=False, scale=False):
        """
        This instantiates the WCSTweak object.
        
        :param wcs: the input image WCS
        :type wcs: pywcs WCS object

        :param inp: input object position in input pixel frame
        :type inp: NumPy array of [x,y] positions

        :param ref: reference object positions in sky frame (RA/Dec)
        :type ref: NumPy array of [ra,dec] positions

        :param rotate: flag to indicate whether to allow rotation of
                       input WCS with respect to reference WCS
        :type rotate: bool

        :param scale: flag to indicate whether to allow scaling of
                      input WCS with respect to reference WCS
        :type scale: bool
        """
        self.wcs = wcs
        self.inp = inp.flatten() # in input pixel frame
        self.ref = ref           # in ra/dec
        self.rotate = rotate
        self.scale = scale
        self.crval = wcs.wcs.crval.copy()
        self.cd = wcs.wcs.cd.copy()

    def transform_ref(self, pars):
        """
        This function transforms reference RA/Dec into input pixel
        frame, via a WCS tweaked by parameters pars.
        
        :param pars: list of parameters to tweak WCS by. Number of 
                     elements is determined by whether rotation/scaling
                     is allowed.  Order is [dRA, dDec, dTheta, dMag].
                     dTheta and dMag are optional.
        :type pars: list of 2, 3, or 4 elements.
        """
        if self.rotate and self.scale:
            d_ra, d_dec, d_theta, d_mag = pars
            self.wcs.wcs.cd = np.dot(d_mag*rotate_2d(d_theta),self.cd)
        elif self.rotate:
            d_ra, d_dec, d_theta = pars
            self.wcs.wcs.cd = np.dot(rotate_2d(d_theta),self.cd)
        elif self.scale:
            d_ra, d_dec, d_mag = pars
            self.wcs.wcs.cd = d_mag*self.cd
        else:
            d_ra, d_dec = pars

        self.wcs.wcs.crval = self.crval + np.array([d_ra, d_dec])/3600.0

        new_ref = self.wcs.wcs_sky2pix(self.ref,1)
        return new_ref.flatten()

    # calculate residual (called by scipy.optimize.leastsq)
    def calc_diff(self, pars):
        """
        This function returns an array of the differences between the
        input sources and the reference sources in the input pixel frame.
        It is intended to be fed to an optimization algorithm, such as 
        scipy.optimize.leastsq.

        :param pars: list of parameters to tweak WCS by. Number of 
                     elements is determined by whether rotation/scaling
                     is allowed.  Order is [dRA, dDec, dTheta, dMag].
                     dTheta and dMag are optional.
        :type pars: list of 2, 3, or 4 elements.
        """
        new_ref = self.transform_ref(pars)
        diff = self.inp - new_ref
        return diff


def match_cxy (xx, sx, yy, sy, firstPass=50, delta=10, log=None):
    """
    Match reference positions (sx,sy) with those of the 
    object catalog (xx,yy). 
    Select those that are within delta pixels from
    the object positions.
    
    firstPass:  (50) First pass radius.

    This matching is a 2 pass algorithm. A form of cross-correlation

    OUTPUT:
    - obj_index: Index array of the objects matched.
    - ref_index: Index array of the references matched.
    """

    # Turn to numpy arrays
    xx, sx, yy, sy = map(np.asarray,(xx,sx,yy,sy))
    if len(xx) == 0:
        return [], []
    
    deltax = firstPass
    deltay = firstPass
    # current_delta is an estimate of final fit quality if delta=None
    if delta is None:
        current_delta = 0.2*firstPass
    else:
        current_delta = delta
    sigmasq = 0.25*current_delta**2
    hw = int(firstPass+1)
    # Make a "landscape" of Gaussian "mountains" onto which we're
    # going to cross-correlate the REFCAT sources
    landscape = np.zeros((int(np.max(yy)+hw),int(np.max(xx)+hw)))
    lysize, lxsize = landscape.shape
    xgrid, ygrid = np.mgrid[0:hw*2+1,0:hw*2+1]
    rsq = (ygrid-hw)**2 + (xgrid-hw)**2
    mountain = np.exp(-0.5*rsq/sigmasq)
    for i in range(len(xx)):
        if xx[i] > -999:
            mx1, mx2, my1, my2 = 0, hw*2+1, 0, hw*2+1
            lx1, lx2 = int(xx[i])-hw, int(xx[i])+hw+1
            ly1, ly2 = int(yy[i])-hw, int(yy[i])+hw+1
            if lx2<0 or lx1>=lxsize or ly2<0 or ly1>=lysize:
                continue
            if lx1 < 0:
                mx1 -= lx1
                lx1 = 0
            if lx2 > lxsize:
                mx2 -= (lx2-lxsize)
                lx2 = lxsize
            if ly1 < 0:
                my1 -= ly1
                ly1 = 0
            if ly2 > lysize:
                my2 -= (ly2-lysize)
                ly2 = lysize
            try:
                landscape[ly1:ly2,lx1:lx2] += mountain[my1:my2,mx1:mx2]
            except ValueError as e:
                print yy[i], xx[i], landscape.shape
                print ly1,ly2,lx1,lx2
                print my1,my2,mx1,mx2

    # We've got the full REFCAT, so first cull that to rough image area
    in_image = np.all((sx>-firstPass, sx<lxsize,
                        sy>-firstPass, sy<lysize),axis=0)

    # We can only do about 2500 cross-correlations per second, so we need
    # to limit the number we do. If the catalogs are sparse, we can just
    # take a list of offsets from pairs of (OBJCAT,REFCAT) sources.
    # Otherwise, we need to test on a grid or something. Which should we do?
    # Count the number of pairs for the first method;
    # if we exceed our limit, make a grid instead
    grid_step = 0.25*current_delta
    num_grid_tests = np.pi*(firstPass/grid_step)**2
    dax = []
    day = []
    height = []
    for sxx,syy in zip(sx[in_image],sy[in_image]):
        gindx, = np.where((xx-sxx)**2+(yy-syy)**2<firstPass*firstPass)
        dax.extend(xx[gindx]-sxx)
        day.extend(yy[gindx]-syy)
    if len(dax) > num_grid_tests:
        # There are fewer in the grid search, so do that
        dax=[]; day=[]
        for dx in np.arange(-firstPass,firstPass,grid_step):
            for dy in np.arange(-firstPass,firstPass,grid_step):
                if dx*dx+dy*dy < firstPass*firstPass:
                    dax.append(dx)
                    day.append(dy)
        
    # For each shift, sum the landscape pixel values at all shifted
    # coordinates -- remember to test whether they're in the landscape
    for dx,dy in zip(dax,day):
        new_sx = (sx[in_image]+dx+0.5).astype(int)
        new_sy = (sy[in_image]+dy+0.5).astype(int)
        indices = np.all((new_sx>=0,new_sx<lxsize,
                    new_sy>=0,new_sy<lysize),axis=0)
        height.append(np.sum(landscape[new_sy[indices],new_sx[indices]]))

    # We've calculated offsets without matching objects, which is what we need
    # This is a two-pass algorithm; first get better offsets by matching and
    # using the median offset; then apply this correction and go again
    # The extra iteration is most likely needed when the grid search has been
    # performed, since the best offsets are not calculated from an actual match
    if len(dax) > 0:
        xoffset, yoffset = dax[np.argmax(height)], day[np.argmax(height)]
        log.info("First pass offsets (x,y): %.2f %.2f" % (xoffset,yoffset))
        sx += xoffset; sy += yoffset

        for iter in range(2):
            g =[]; r=[]
            dax=[]; day=[]
            for k in [kk for kk in range(len(sx)) if in_image[kk]]:
                gindx,= np.where((xx-sx[k])**2+(yy-sy[k])**2<current_delta**2)
                for i in gindx:
                    dx = xx[i] - sx[k] 
                    dy = yy[i] - sy[k] 

                    # if there are multiple matches, keep only the closest
                    if (i in g or k in r):
                        if i in g:
                            first_ind = g.index(i)
                        else:
                            first_ind = r.index(k)
                        first_dist = dax[first_ind]**2 + day[first_ind]**2
                        this_dist = dx**2 + dy**2
                        if (first_dist > this_dist):
                            del dax[first_ind]
                            del day[first_ind]
                            del g[first_ind]
                            del r[first_ind]
                            dax.append(dx)
                            day.append(dy)
                            g.append(i)
                            r.append(k)
                    else:
                        dax.append(dx)
                        day.append(dy)
                        g.append(i)
                        r.append(k)
            dx = np.mean(dax)
            dy = np.mean(day)
            sx += dx
            sy += dy
            xoffset += dx
            yoffset += dy
            log.info("Tweaked offsets by: %.2f %.2f" % (dx,dy))
            # Use scatter in points to determine quality of fit for final pass
            if delta is None and iter==0:
                current_delta = 2*np.sqrt(np.std(dax-dx)*np.std(dax-dy))
                log.info("Using %.3f pixels as final matching "
                         "radius" % current_delta)
        g,r,dax,day = map(np.asarray, (g,r,dax,day))
        #for i,k,dx,dy in zip(g,r,dax,day):
        #    print i+1,k+1,dx,dy

    # dax may have been >0 before but now 0 if there are no good matches
    if len(dax) > 0:
        indxy = g
        indr = r
        stdx = np.std(dax)
        stdy = np.std(day)
        # Add 1 to debug display to match directly with catalogs
        #print "indxy = ", indxy+1
        #print "indr = ", indr+1
        log.info('Final offset (x,y): %.2f %.2f (%.2f %.2f)' %
             (xoffset,yoffset,stdx,stdy))
    else:
        indxy,indr=[],[]
        log.info('No matched sources')

    return indxy, indr

def clipped_mean(data):
    num_total = len(data)
    mean = data.mean()
    sigma = data.std()

    if num_total<3:
        return mean, sigma

    num = num_total
    clipped_data = data
    clip = 0
    while (num>0.5*num_total):
        # CJS: edited this as upper limit was mean+1*sigma => bias
        clipped_data = data[(data<mean+3*sigma) & (data>mean-3*sigma)]
        num = len(clipped_data)

        if num>0:
            mean = clipped_data.mean() 
            sigma = clipped_data.std()
        elif clip==0:
            return mean, sigma
        else:
            break

        clip+=1
        if clip>10:
            break

    return mean,sigma


# The following functions and classes were borrowed from STSCI's spectools
# package, currently under development.  They might be able to be
# replaced with a direct import of spectools.util if/when it is available

iraf_models_map = {1.: 'chebyshev',
                   2.: 'legendre',
                   3.: 'spline3',
                   4.: 'spline1'}
inverse_iraf_models_map = {'chebyshev': 1.,
                           'legendre': 2.,
                           'spline3': 3.,
                           'spline1': 4.}

def get_records(fname):
    """
    Read the records of an IRAF database file ionto a python list
    
    Parameters
    ----------
    fname: string
           name of an IRAF database file
           
    Returns
    -------
        A list of records
    """
    f = open(fname)
    dtb = f.read()
    f.close()
    records = []
    recs = dtb.split('begin')[1:]
    records = [Record(r) for r in recs]
    return records

def get_database_string(fname):
    """
    Read an IRAF database file
    
    Parameters
    ----------
    fname: string
          name of an IRAF database file
           
    Returns
    -------
        the database file as a string
    """
    f = open(fname)
    dtb = f.read()
    f.close()
    return dtb

class Record(object):
    """
    A base class for all records - represents an IRAF database record
    
    Attributes
    ----------
    recstr: string 
            the record as a string
    fields: dict
            the fields in the record
    taskname: string
            the name of the task which created the database file
    """
    def __init__(self, recstr):
        self.recstr = recstr
        self.fields = self.get_fields()
        self.taskname = self.get_task_name()
        
    def aslist(self):
        reclist = self.recstr.split('\n')
        reclist = [l.strip() for l in reclist]
        out = [reclist.remove(l) for l in reclist if len(l)==0]
        return reclist
    
    def get_fields(self):
        # read record fields as an array
        fields = {}
        flist = self.aslist()
        numfields = len(flist)
        for i in range(numfields):
            line = flist[i]
            if line and line[0].isalpha():
                field =  line.split()
                if i+1 < numfields:
                    if not flist[i+1][0].isalpha():
                        fields[field[0]] = self.read_array_field(
                                             flist[i:i+int(field[1])+1])
                    else:
                        fields[field[0]] = " ".join(s for s in field[1:])
                else:
                    fields[field[0]] = " ".join(s for s in field[1:])
            else:
                continue
        return fields
    
    def get_task_name(self):
        try:
            return self.fields['task']
        except KeyError:
            return None
    
    def read_array_field(self, fieldlist):
        # Turn an iraf record array field into a numpy array
        fieldline = [l.split() for l in fieldlist[1:]]
        # take only the first 3 columns
        # identify writes also strings at the end of some field lines
        xyz = [l[:3] for l in fieldline]
        try:
            farr = np.array(xyz)
        except:
            print "Could not read array field %s" % fieldlist[0].split()[0]
        return farr.astype(np.float64)
    
class IdentifyRecord(Record):
    """
    Represents a database record for the longslit.identify task
    
    Attributes
    ----------
    x: array
       the X values of the identified features
       this represents values on axis1 (image rows)
    y: int
       the Y values of the identified features 
       (image columns)
    z: array
       the values which X maps into
    modelname: string
        the function used to fit the data
    nterms: int
        degree of the polynomial which was fit to the data
        in IRAF this is the number of coefficients, not the order
    mrange: list
        the range of the data
    coeff: array
        function (modelname) coefficients
    """
    def __init__(self, recstr):
        super(IdentifyRecord, self).__init__(recstr)
        self._flatcoeff = self.fields['coefficients'].flatten()
        self.x = self.fields['features'][:,0]
        self.y = self.get_ydata()
        self.z = self.fields['features'][:,1]
####here - ref?
        self.zref = self.fields['features'][:,2]
        self.modelname = self.get_model_name()
        self.nterms = self.get_nterms()
        self.mrange = self.get_range()
        self.coeff = self.get_coeff()
        
    def get_model_name(self):
        return iraf_models_map[self._flatcoeff[0]]
    
    def get_nterms(self):
        return self._flatcoeff[1]
    
    def get_range(self):
        low = self._flatcoeff[2]
        high = self._flatcoeff[3]
        return [low, high]
    
    def get_coeff(self):
        return self._flatcoeff[4:]
    
    def get_ydata(self):
        image = self.fields['image']
        left = image.find('[')+1
        right = image.find(']')
        section = image[left:right]
        if ',' in section:
            yind = image.find(',')+1
            return int(image[yind:-1])
        else:
            return int(section)
        #xind = image.find('[')+1
        #yind = image.find(',')+1
        #return int(image[yind:-1])

class FitcoordsRecord(Record):
    """
    Represents a database record for the longslit.fitccords task
    
    Attributes
    ----------
    modelname: string
        the function used to fit the data
    xorder: int
        number of terms in x
    yorder: int
        number of terms in y
    xbounds: list
        data range in x
    ybounds: list
        data range in y
    coeff: array
        function coefficients

    """
    def __init__(self, recstr):
        super(FitcoordsRecord, self).__init__(recstr)
        self._surface = self.fields['surface'].flatten()
        self.modelname = iraf_models_map[self._surface[0]]
        self.xorder = self._surface[1]
        self.yorder = self._surface[2]
        self.xbounds = [self._surface[4], self._surface[5]]
        self.ybounds = [self._surface[6], self._surface[7]]
        self.coeff = self.get_coeff()
        
    def get_coeff(self):
        return self._surface[8:]
        
class IDB(object):
    """
    Base class for an IRAF identify database 
    
    Attributes
    ----------
    records: list
             a list of all `IdentifyRecord` in the database
    numrecords: int
             number of records
    """
    def __init__(self, dtbstr):
        lst = self.aslist(dtbstr)
        self.records = [IdentifyRecord(rstr) for rstr in self.aslist(dtbstr)]
        self.numrecords = len(self.records)

    def aslist(self, dtb):
        # return a list of records
        # if the first one is a comment remove it from the list
        rl = dtb.split('begin')
        try:
            rl0 = rl[0].split('\n')
        except:
            return rl
        if len(rl0) == 2 and rl0[0].startswith('#') and not rl0[1].strip():
            return rl[1:]
        elif len(rl0)==1 and not rl0[0].strip():
            return rl[1:]
        else:
            return rl
        
class ReidentifyRecord(IDB):
    """
    Represents a database record for the onedspec.reidentify task
    """
    def __init__(self, databasestr):
        super(ReidentifyRecord, self).__init__(databasestr)
        self.x = np.array([r.x for r in self.records])
        self.y = self.get_ydata()
        self.z = np.array([r.z for r in self.records])
        
        
    def get_ydata(self):
        y = np.ones(self.x.shape)
        y = y*np.array([r.y for r in self.records])[:,np.newaxis]
        return y


# This class pulls together fitcoords and identify databases into
# a single entity that can be written to or read from disk files
# or pyfits binary tables
class SpectralDatabase(object):
    def __init__(self, database_name=None, record_name=None, 
                 binary_table=None):
        """
        database_name is the name of the database directory
        on disk that contains the database files associated with
        record_name.  For example, database_name="database",
        record_name="image_001" (corresponding to the first science
        extention in a data file called image.fits
        """
        self.database_name = database_name
        self.record_name = record_name
        self.binary_table = binary_table

        self.identify_database = None
        self.fitcoords_database = None

        # Initialize from database on disk
        if (database_name is not None and record_name is not None):

            if not os.path.isdir(database_name):
                raise IOError('Database directory %s does not exist' %
                              database_name)

            # Read in identify database
            db_filename = "%s/id%s" % (database_name,record_name)
            if not os.access(db_filename,os.R_OK):
                raise IOError("Database file %s does not exist " \
                              "or cannot be accessed" % db_filename)

            db_str = get_database_string(db_filename)
            self.identify_database = IDB(db_str)

            # Read in fitcoords database
            db_filename = "%s/fc%s" % (database_name,record_name)
            if not os.access(db_filename,os.R_OK):
                raise IOError("Database file %s does not exist " \
                              "or cannot be accessed" % db_filename)

            db_str = get_database_string(db_filename)
            self.fitcoords_database = FitcoordsRecord(db_str)

        # Initialize from pyfits binary table in memory
        elif binary_table is not None:

            # Get record_name from header if not passed
            if record_name is not None:
                self.record_name = record_name
            else:
                self.record_name = binary_table.header["RECORDNM"]

            # Format identify information from header and table
            # data into a database string
            db_str = self._identify_db_from_table(binary_table)
            self.identify_database = IDB(db_str)
            
            # Format fitcoords information from header
            # into a database string
            db_str = self._fitcoords_db_from_table(binary_table)
            self.fitcoords_database = FitcoordsRecord(db_str)

        else:
            raise TypeError("Both database and binary table are None.")

    def _identify_db_from_table(self, tab):

        # Get feature information from table data
        features = tab.data
        nrows = len(features)
        nfeat = features["spectral_coord"].shape[1]
        ncoeff = features["fit_coefficients"].shape[1]
        
        db_str = ""

        for row in range(nrows):

            feature = features[row]

            # Make a dictionary to hold information gathered from
            # the table.  This structure is not quite the same as
            # the fields member of the Record class, but it is the
            # same principle
            fields = {}
            fields["id"] = self.record_name
            fields["task"] = "identify"
            fields["image"] = "%s[*,%d]" % (self.record_name,
                                            feature["spatial_coord"])
            fields["units"] = tab.header["IDUNITS"]
            
            zip_feature = np.array([feature["spectral_coord"],
                                    feature["fit_wavelength"],
                                    feature["ref_wavelength"]])
            fields["features"] = zip_feature.swapaxes(0,1)

            fields["function"] = tab.header["IDFUNCTN"]
            fields["order"] = tab.header["IDORDER"]
            fields["sample"] = tab.header["IDSAMPLE"]
            fields["naverage"] = tab.header["IDNAVER"]
            fields["niterate"] = tab.header["IDNITER"]

            reject = tab.header["IDREJECT"].split()
            fields["low_reject"] = float(reject[0])
            fields["high_reject"] = float(reject[1])
            fields["grow"] = tab.header["IDGROW"]

            # coefficients is a list of numbers with the following elements:
            # 0: model number (function type)
            # 1: order
            # 2: x min
            # 3: x max
            # 4 on: function coefficients
            coefficients = []
            
            model_num = inverse_iraf_models_map[fields["function"]]
            coefficients.append(model_num)

            coefficients.append(fields["order"])
            
            idrange = tab.header["IDRANGE"].split()
            coefficients.append(float(idrange[0]))
            coefficients.append(float(idrange[1]))

            fit_coeff = feature["fit_coefficients"].tolist()
            coefficients.extend(fit_coeff)
            fields["coefficients"] = np.array(coefficients).astype(np.float64)
            

            # Compose fields into a single string
            rec_str = "%-8s%-8s %s\n" % ("begin",fields["task"],fields["image"])
            for field in ["id","task","image","units"]:
                rec_str += "%-8s%-8s%s\n" % ("",field,str(fields[field]))
            rec_str += "%-8s%-8s %d\n" % ("","features",len(fields["features"]))
            for feat in fields["features"]:
                rec_str += "%16s%10f %10f %10f\n" % ("",feat[0],feat[1],feat[2])
            for field in ["function","order","sample",
                          "naverage","niterate","low_reject",
                          "high_reject","grow"]:
                rec_str += "%-8s%s %s\n" % ("",field,str(fields[field]))
            rec_str += "%-8s%-8s %d\n" % ("","coefficients",
                                         len(fields["coefficients"]))
            for coeff in fields["coefficients"]:
                rec_str += "%-8s%-8s%E\n" % ("","",coeff)
            rec_str+="\n"

            db_str += rec_str

        return db_str

    def _fitcoords_db_from_table(self, tab):

        # Make a dictionary to hold information gathered from
        # the table.  This structure is not quite the same as
        # the fields member of the Record class, but it is the
        # same principle
        fields = {}

        fields["begin"] = self.record_name
        fields["task"] = "fitcoords"
        fields["axis"] = tab.header["FCAXIS"]
        fields["units"] = tab.header["FCUNITS"]

        # The surface is a list of numbers with the following elements:
        # 0: model number (function type)
        # 1: x order
        # 2: y order
        # 3: cross-term type (always 1. for fitcoords)
        # 4. xmin
        # 5: xmax
        # 6. xmin
        # 7: xmax
        # 8 on: function coefficients
        surface = []
        
        model_num = inverse_iraf_models_map[tab.header["FCFUNCTN"]]
        surface.append(model_num)            

        xorder = tab.header["FCXORDER"]
        yorder = tab.header["FCYORDER"]
        surface.append(xorder)
        surface.append(yorder)
        surface.append(1.)

        xrange = tab.header["FCXRANGE"].split()
        surface.append(float(xrange[0]))
        surface.append(float(xrange[1]))
        yrange = tab.header["FCYRANGE"].split()
        surface.append(float(yrange[0]))
        surface.append(float(yrange[1]))
            
        for i in range(int(xorder)*int(yorder)):
            coeff = tab.header["FCCOEF%d" % i]
            surface.append(coeff)

        fields["surface"] = np.array(surface).astype(np.float64)

        # Compose fields into a single string
        db_str = "%-8s%s\n" % ("begin",fields["begin"])
        for field in ["task","axis","units"]:
            db_str += "%-8s%-8s%s\n" % ("",field,str(fields[field]))
        db_str += "%-8s%-8s%d\n" % ("","surface",len(fields["surface"]))
        for coeff in fields["surface"]:
            db_str += "%-8s%-8s%E\n" % ("","",coeff)

        return db_str

    def write_to_disk(self, database_name=None, record_name=None):

        # Check for provided names; use names from self if not
        # provided as input
        if database_name is None and self.database_name is None:
            raise TypeError("No database_name provided")
        elif database_name is None and self.database_name is not None:
            database_name = self.database_name
        if record_name is None and self.record_name is None:
            raise TypeError("No record_name provided")
        elif record_name is None and self.record_name is not None:
            record_name = self.record_name
        
        # Make the directory if needed
        if not os.path.exists(database_name):
            os.mkdir(database_name)

        # Timestamp
        import datetime
        timestamp = str(datetime.datetime.now())

        # Write identify files
        id_db = self.identify_database
        if id_db is not None:
            db_filename = "%s/id%s" % (database_name,record_name)
            db_file = open(db_filename,"w")
            db_file.write("# "+timestamp+"\n")
            for record in id_db.records:
                db_file.write("begin")
                db_file.write(record.recstr)
            db_file.close()

        # Write fitcoords files
        fc_db = self.fitcoords_database
        if fc_db is not None:
            db_filename = "%s/fc%s" % (database_name,record_name)
            db_file = open(db_filename,"w")
            db_file.write("# "+timestamp+"\n")
            db_file.write(fc_db.recstr)
            db_file.close()

    def as_binary_table(self,record_name=None):
        
        # Should this be lazy loaded?
        import pyfits as pf

        if record_name is None:
            record_name = self.record_name

        # Get the maximum number of features identified in any
        # record.  Use this as the length of the array in the
        # wavelength_coord and fit_wavelength fields
        nfeat = max([len(record.x)
                     for record in self.identify_database.records])

        # The number of coefficients should be the same for all
        # records, so take the value from the first record
        ncoeff = self.identify_database.records[0].nterms

        # Get the number of rows from the number of identify records
        nrows = self.identify_database.numrecords

        # Create pyfits Columns for the table
        column_formats = [{"name":"spatial_coord","format":"I"},
                          {"name":"spectral_coord","format":"%dE"%nfeat},
                          {"name":"fit_wavelength","format":"%dE"%nfeat},
                          {"name":"ref_wavelength","format":"%dE"%nfeat},
                          {"name":"fit_coefficients","format":"%dE"%ncoeff},]
        columns = [pf.Column(**format) for format in column_formats]

        # Make the empty table.  Use the number of records in the
        # database as the number of rows
        table = pf.new_table(columns,nrows=nrows)

        # Populate the table from the records
        for i in range(nrows):
            record = self.identify_database.records[i]
            row = table.data[i]
            row["spatial_coord"] = record.y
            row["fit_coefficients"] = record.coeff
            if len(row["spectral_coord"])!=len(record.x):
                row["spectral_coord"][:len(record.x)] = record.x
                row["spectral_coord"][len(record.x):] = -999
            else:
                row["spectral_coord"] = record.x
            if len(row["fit_wavelength"])!=len(record.z):
                row["fit_wavelength"][:len(record.z)] = record.z
                row["fit_wavelength"][len(record.z):] = -999
            else:
                row["fit_wavelength"] = record.z
            if len(row["ref_wavelength"])!=len(record.zref):
                row["ref_wavelength"][:len(record.zref)] = record.zref
                row["ref_wavelength"][len(record.zref):] = -999
            else:
                row["ref_wavelength"] = record.zref
            
        # Store the record name in the header
        table.header.update("RECORDNM",record_name)

        # Store other important values from the identify records in the header
        # These should be the same for all records, so take values
        # from the first record
        first_record = self.identify_database.records[0]
        table.header.update("IDUNITS",first_record.fields["units"])
        table.header.update("IDFUNCTN",first_record.modelname)
        table.header.update("IDORDER",first_record.nterms)
        table.header.update("IDSAMPLE",first_record.fields["sample"])
        table.header.update("IDNAVER",first_record.fields["naverage"])
        table.header.update("IDNITER",first_record.fields["niterate"])
        table.header.update("IDREJECT","%s %s" % 
                            (first_record.fields["low_reject"],
                             first_record.fields["high_reject"]))
        table.header.update("IDGROW",first_record.fields["grow"])
        table.header.update("IDRANGE","%s %s" % 
                            (first_record.mrange[0],first_record.mrange[1]))

        # Store fitcoords information in the header
        fc_record = self.fitcoords_database
        table.header.update("FCUNITS",fc_record.fields["units"])
        table.header.update("FCAXIS",fc_record.fields["axis"])
        table.header.update("FCFUNCTN",fc_record.modelname)
        table.header.update("FCXORDER",fc_record.xorder)
        table.header.update("FCYORDER",fc_record.yorder)
        table.header.update("FCXRANGE","%s %s" % 
                            (fc_record.xbounds[0],fc_record.xbounds[1]))
        table.header.update("FCYRANGE","%s %s" % 
                            (fc_record.ybounds[0],fc_record.ybounds[1]))
        for i in range(len(fc_record.coeff)):
            coeff = fc_record.coeff[i]
            table.header.update("FCCOEF%d" % i, coeff)
####here -- comments

        return table

class FittedFunction:
    """
    Represents the result of trying to fit a function to some data.
    
    Members
    ----------
    get_model_function(): ndarray
        returns an ndarray representing values of the fitted function
    get_fwhm(): float
        the full width half maximum of the fitted function
    get_rsquared(): float
        a number between zero and one describing how good the fit is
    get_success(): int
        the value returned from scipy.optimize.leastsq
    get_name(): str
        the name of the fitted function ('moffat' or 'gaussian')
    get_background(): float
    get_peak(): float
    get_center(): (float, float)
        the center of the function
    get_width(): (float, float)
        the width of the function in x and y dimensions
    get_theta(): float
        the rotation of the function in degrees
    get_fwhm_ellipticity(): (float, float)
        returns the function width converted to FWHM and ellipticity
    get_beta(): float
        the beta of the moffat function or None if not a moffat function
    """
    def __init__(self, function, function_name, success, bg, peak, x_ctr, y_ctr, x_width, y_width, theta, beta=None):
        self.function = function
        self.success = success
        self.function_name = function_name
        self.background = bg
        self.peak = peak
        self.x_ctr = x_ctr
        self.y_ctr = y_ctr
        self.x_width = x_width
        self.y_width = y_width
        self.theta = theta
        self.beta = beta

    def get_params(self):
        pars = (self.background,
                self.peak,
                self.x_ctr,
                self.y_ctr,
                self.x_width,
                self.y_width,
                self.theta)
        
        if self.function_name == "moffat":
            pars = pars + (self.beta,)

        return pars        

    def get_model_function(self):
        pars = self.get_params()
        if self.function_name == "gauss":
            return self.function.model_gauss_2d(pars)
        elif self.function_name == "moffat":
            return self.function.model_moffat_2d(pars)
        else:
            raise Errors.InputError("Function %s not supported" % self.function_name)

    def get_stamp_data(self):
        return self.function.stamp

    def get_rsquared(self):
        pars = self.get_params()

        func = get_function_with_penalties(self.function)
        residuals = func(pars)
        ss_err = (residuals**2).sum()

        data = self.get_stamp_data()
        ss_tot = ((data - data.mean())**2).sum()

        return 1.0 - (ss_err / ss_tot)

    def get_success(self):
        return self.success

    def get_name(self):
        return self.function_name

    def get_background(self):
        return self.background

    def get_peak(self):
        return self.peak

    def get_center(self):
        return (self.x_ctr, self.y_ctr)

    def get_fwhm(self):
        if self.function_name == "moffat":
            avg_width = (self.x_width + self.y_width) / 2.0
            fwhm = avg_width * 2 * np.sqrt((2**(1 / self.beta)) - 1)
            return fwhm
        else:
            fwhmx = abs(2*np.sqrt(2*np.log(2))*self.x_width)
            fwhmy = abs(2*np.sqrt(2*np.log(2))*self.y_width)
            return np.sqrt(fwhmx * fwhmy)

    def get_width(self):
        return (self.x_width, self.y_width)

    def get_theta(self):
        return math.degrees(math.acos(math.cos(self.theta)))

    def get_beta(self):
        return self.beta

    def get_fwhm_ellipticity(self):
        x_width = self.x_width
        y_width = self.y_width

        # convert fit parameters to FWHM, ellipticity
        if self.function_name == "moffat":
            # convert width to Gaussian-type sigma
            x_width = x_width*np.sqrt(((2**(1/beta)-1)/(2*np.log(2))))
            y_width = y_width*np.sqrt(((2**(1/beta)-1)/(2*np.log(2))))
        
        fwhmx = abs(2*np.sqrt(2*np.log(2))*x_width)
        fwhmy = abs(2*np.sqrt(2*np.log(2))*y_width)
        pa = (self.theta * (180 / np.pi))
        pa = pa % 360
        
        if fwhmy < fwhmx:
            ellip = 1 - fwhmy / fwhmx
        elif fwhmx < fwhmy:
            ellip = 1 - fwhmx / fwhmy
            pa = pa - 90 
        else:
            ellip = 0

        # FWHM is geometric mean of x and y FWHM
        fwhm = np.sqrt(fwhmx * fwhmy)
        
        return fwhm, ellip

def get_function_with_penalties(function):
    """ heavily penalize the function when it wants to start using a negative width """
    
    def bounded_function(pars):
        diff = function.calc_diff(pars)

        bg, peak, x_ctr, y_ctr, x_width, y_width, theta = pars[:7]
        if x_width < 0.0:
            diff = diff * abs(x_width) * 100
        
        if y_width < 0.0:
            diff = diff * abs(y_width) * 100

        return diff
    return bounded_function

            
def get_fitted_function(stamp_data, default_fwhm, default_bg=None, centroid_function="moffat"):
    """
    This function returns a FittedFunction object containing the
    parameters needed to fit the `centroid_function` to the
    `stamp_data` using `default_fwhm` and `default_bg` as starting conditions.

    :param stamp_data: subset of data to fit
    :type stamp_data: 2D NumPy array

    :param default_fwhm: the initial guess of the FWHM, related to pixel scale?
    :type default_fwhm: float

    :param default_bg: the initial guess of the background to use,
      if None, uses the median of the data passed to stamp_data
    :type default_bg: float

    :param centroid_function: function to fit, either 'moffat' or 'gaussian'
    :type  centroid_function: str
    """
    if default_bg is None:
        default_bg = np.median(stamp_data)
    
    # starting values for model fit
    bg = default_bg
    peak = stamp_data.max() - bg
    x_ctr = (stamp_data.shape[1] - 1) / 2.0
    y_ctr = (stamp_data.shape[0] - 1) / 2.0
    x_width = default_fwhm
    y_width = default_fwhm
    theta = 0.0
    beta = 1.0

    pars = (bg, peak, x_ctr, y_ctr, x_width, y_width, theta)

    # instantiate model fit object and initial parameters
    if centroid_function == "gauss":
        mf = GaussFit(stamp_data)
    elif centroid_function == "moffat":
        pars = pars + (beta,)
        mf = MoffatFit(stamp_data)
    else:
        raise Errors.InputError("Centroid function %s not supported" %
                                centroid_function)

    func = get_function_with_penalties(mf)
    
    # least squares fit of model to data
    try:
        # for scipy versions < 0.9
        new_pars, success = opt.leastsq(func,
                                        pars,
                                        maxfev=100, 
                                        warning=False)
    except:
        # for scipy versions >= 0.9
        import warnings
        warnings.simplefilter("ignore")
        new_pars, success = opt.leastsq(func,
                                        pars,
                                        maxfev=100)

    beta = None
    if centroid_function == "moffat":
        beta = new_pars[7]

    # strip off the beta from moffat
    pars = new_pars[:7]
    
    return FittedFunction(mf, centroid_function, success, *pars, beta=beta)
