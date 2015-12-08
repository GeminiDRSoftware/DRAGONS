# The astroTools module contains astronomy specific utility functions

import os
import re
import math
import numpy as np
import scipy.optimize as opt
import pandas as pd

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


def match_cxy (xx, sx, yy, sy, firstPass=50, delta=None, log=None):
    """
    Match reference positions (sx,sy) with those of the 
    object catalog (xx,yy). 
    Select those that are within delta pixels from
    the object positions.
    
    firstPass:  (50) First pass delta radius.

    This matching is a 2 pass algorithm. First pass takes the
    larger delta and adjust the reference positions by the median 
    of the x,y offset. The second pass takes 'delta' value and 
    look for those x,y now closer to the reference positions.
    The units are pixels for the deltas. Positions in degrees 
    should not be passed to this function, since at least a 
    cos(dec) correction is required!


    OUTPUT:
    - obj_index: Index array of the objects matched.
    - ref_index: Index array of the references matched.
    """

    # turn to numpy arrays
    xx, sx, yy, sy = map(np.asarray,(xx,sx,yy,sy))
        
    def getg(xx, sx, yy, sy, deltax=2.5, deltay=2.5):
        """ Return object(xx) and reference(sx) indices of
        common positions.
        OUTPUT
        g:    Indices of the object position common to
        r:    indices of the reference position
        """
        dax=[]; day=[]; g=[]; r=[]
        for k in range(len(sx)):
            gindx,= np.where((abs(xx-sx[k])<deltax) & 
                             (abs(yy-sy[k])<deltay))
            for i in gindx:
                dx = xx[i] - sx[k] 
                dy = yy[i] - sy[k] 

                # if there are multiple matches, keep only the
                # closest one
                if i in g or k in r:
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
                            
        dax,day = map(np.asarray, (dax,day))
        
        #DEBUG # For debugging or improvement purpose, save the match offsets
        #DEBUG # to disk.  Uncomment if necessary.
        #DEBUG from datetime import datetime
        #DEBUG timestr = datetime.now().time().isoformat()
        #DEBUG fout = open('daxy-'+str(deltax)+'-'+timestr+'.dat', mode='w')
        #DEBUG for i in range(len(dax)):
        #DEBUG     fout.write(str(dax[i])+'\t'+str(day[i])+'\n')
        #DEBUG fout.close()

        # KL: The clump finding code needs to be functionalize.
        # KL: This was done in a hurry while preparing for deployment
        # KL: It works with several type of fields, sparse and crowded
        # KL: and several types of instruments.  It just needs to be
        # KL: cleaned up a bit.
        #
        # Identify the location of the clump of good matches then do find
        # its center.  This technique helps when the WCS is not too good
        # and the sources are above some density causing several "matches"
        # to be completely wrong.  Even in those cases, there is generally
        # an obviously clump of good matches around the correct x, y offset
        
        if len(dax) > 10 and len(day) > 10:
            # Get an histogram of dax and day offsets to locate the clump.
            # The clump approximate position will be where the tallest histogram
            # bar is located.
            df = pd.DataFrame({'dx' : dax, 'dy' : day})
            counts, divisions = np.histogram(df['dx'], bins=10)
            xbinsize = abs(divisions[0] - divisions[1])
            counts, divisions = np.histogram(df['dy'], bins=10)
            ybinsize = abs(divisions[0] - divisions[1])
            
            apprx_xoffset = df['dx'].value_counts(bins=10).idxmax() + \
                            (xbinsize / 2.0)
            apprx_yoffset = df['dy'].value_counts(bins=10).idxmax() + \
                            (ybinsize / 2.0)
            stdx, stdy = (df['dx'].std(), df['dy'].std())
            
            # For crowded field with good matches, the histogram might have
            # most sources in one bin, which might be wide.  To increase
            # precision, we catch those cases and focus on that populous bin.
            # Otherwise, all the data is used.
            # threshold: fraction of matches in the top bin.
            # thr_binsize: size of the bin, we do this only if the bin is large.
            threshold = 0.7
            thr_binsize = 10
            if df['dx'].value_counts(bins=10).max() / float(len(df['dx'])) \
                  >= threshold:
                if xbinsize >= thr_binsize:
                    dfsub =  df[(df['dx'] > apprx_xoffset - 2*thr_binsize) & \
                                (df['dx'] < apprx_xoffset + 2*thr_binsize)]
                    counts, divisions = np.histogram(dfsub['dx'], bins=10)
                    subbinsize = abs(divisions[0] - divisions[1])
                    apprx_xoffset = dfsub['dx'].value_counts(bins=10).idxmax() + \
                                    (subbinsize / 2.0)
                    stdx = dfsub['dx'].std()
            if df['dy'].value_counts(bins=10).max() / float(len(df['dy'])) \
                  >= threshold:
                if ybinsize >= thr_binsize:
                    dfsub =  df[(df['dy'] > apprx_xoffset - 2*thr_binsize) & \
                                (df['dy'] < apprx_xoffset + 2*thr_binsize)]
                    counts, divisions = np.histogram(dfsub['dy'], bins=10)
                    subbinsize = abs(divisions[0] - divisions[1])
                    apprx_yoffset = dfsub['dy'].value_counts(bins=10).idxmax() + \
                                    (subbinsize / 2.0)
                    stdy = dfsub['dy'].std()

            # Get the center of that clump, the actually x, y offsets.
            # Focus on the area around the clump.  Use the standard deviation
            # to set a box around the clump on which stats will be derived.
            # We already know now that anything outside that box is a bad match.
            # Median appears to work better than mean for this.  
            llimitx, ulimitx = (apprx_xoffset - stdx, apprx_xoffset + stdx)
            llimity, ulimity = (apprx_yoffset - stdy, apprx_yoffset + stdy)
            
            xoffset = df[(df['dx'] > llimitx) & (df['dx'] < ulimitx)]['dx'].median()
            yoffset = df[(df['dy'] > llimity) & (df['dy'] < ulimity)]['dy'].median()
            stdx = df[(df['dx'] > llimitx) & (df['dx'] < ulimitx)]['dx'].std()
            stdy = df[(df['dy'] > llimity) & (df['dy'] < ulimity)]['dy'].std()
        elif len(dax) > 1 and len(day) > 1:
            # Too few source for clump-finding.  Use the old technique.
            xoffset = np.median(dax)
            yoffset = np.median(day)
            stdx = np.std(dax)
            stdy = np.std(day)
        elif len(dax) == 1 and len(day) == 1:
            xoffset = dax[0]
            yoffset = day[0]
            stdx = 0.0
            stdy = 0.0
        else:
            xoffset = float('nan')
            yoffset = float('nan')
            stdx = float('nan')
            stdy = float('nan')
        
        return np.asarray(g), np.asarray(r), xoffset, yoffset ,stdx, stdy
        
        # Below is the old code that was just doing the median instead
        # of trying to find the clump of good matches.  I (KL) keep it
        # here for now until I have confirmed that the new clump-detection
        # technique works well on a variety of data.
        #
        ## When dax and/or day are empty, np.median and np.std issue a 
        ## RuntimeWarning.  We are suppressing that warning.  NaN are 
        ## returned when median and std are applied to an empty array.
        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        #    mx = np.median(dax); stdx = np.std(dax)
        #    my = np.median(day); stdy = np.std(day)
            
        #return np.asarray(g),np.asarray(r),mx,my,stdx,stdy 


    # Select only those standards with less than 10 pixels from objects.
    # Get the median values (mx,my) of the differences and add these
    # to the standard positions.

    #NOTE: We are setting a large delta here, we would need to see
    #      median 1st...

    ig,r,mx,my,stdx,stdy = getg(xx,sx,yy,sy, deltax=firstPass,deltay=firstPass)
    log.info('Median differences (x,y):%.2f %.2f, %.2f %.2f' % 
             (mx,my,stdx,stdy)+"[First iteration]")
    if len(r) == 0 or len(r) == 1:
        if len(r) == 1:
            log.info('Only one good source is available, so the '
                     'cross-correlation routine will be run once only')
        return ig,r
        
    # Now shift reference position by adding the median of the
    # differences. The standards are now closer to the object positions.
    sx = sx + mx   
    sy = sy + my

    # Select only those that are closer than delta or default(6.5) pixels.
    xxx = xx[ig]; yyy = yy[ig] 
    deltax=delta; deltay=delta
    if delta == None:
        deltax=2*stdx; deltay=2*stdy
    g,r,mx,my,stdx,stdy = getg (xxx, sx, yyy, sy, 
                                deltax=deltax, deltay=deltay)
    log.info('Median differences (x,y):%.2f %.2f %.2f %.2f' %
             (mx,my,stdx,stdy)+"[Second iteration]")

    if g.size == 0:
        indxy,indr=[],[]
    else:
        indxy = ig[g]
        indr = r
        
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
        clipped_data = data[(data<mean+sigma) & (data>mean-3*sigma)]
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
