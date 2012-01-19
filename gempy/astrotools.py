# The astroTools module contains astronomy specific utility functions

import re
import math
import numpy as np

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
    The units are pixels for the deltas. If you are passing
    degrees, the deltas need to be consistent.


    OUTPUT:
    - obj_index: Index array of the objects matched.
    - ref_index: Index array of the referecences matched.
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
        mx = np.median(dax); stdx = np.std(dax)
        my = np.median(day); stdy = np.std(day)
            
        return np.asarray(g),np.asarray(r),mx,my,stdx,stdy 


    # Select only those standards with less than 10 pixels from objects.
    # Get the median values (mx,my) of the differences and add these
    # to the standard positions.

    #NOTE: We are setting a large delta here, we would need to see
    #      median 1st...

    ig,r,mx,my,stdx,stdy = getg(xx,sx,yy,sy, deltax=firstPass,deltay=firstPass)
    log.info('Median differences (x,y):%.2f %.2f, %.2f %.2f' % 
             (mx,my,stdx,stdy)+"[First iteration]")

    if len(r) == 0:
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
